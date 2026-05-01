//! JIT-enabled evaluation — the sole entry point for JIT execution.
//!
//! This module owns the JitDriver, tracing hooks, and compiled-code
//! execution. pyre-interpreter provides the pure interpreter (eval_frame_plain)
//! and the opcode trait implementations on PyFrame.
//!
//! Equivalent to PyPy's `pypyjit/interp_jit.py` — the JIT is injected
//! from outside the interpreter.

use crate::jit::state::{PyreEnv, PyreJitState};
use crate::jit::trace::trace_bytecode;
use pyre_interpreter::PyExecutionContext;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{
    PyResult, StepResult, decode_instruction_for_dispatch, execute_opcode_step,
};
use std::cell::{Cell, UnsafeCell};
use std::collections::HashMap;

use majit_backend::Backend;
use majit_gc::GcAllocator;
use majit_gc::trace::TypeInfo;
use majit_ir::{Type, Value};
use majit_metainterp::blackhole::ExceptionState;
use majit_metainterp::{CompiledExitLayout, DetailedDriverRunOutcome, JitState};

/// Host tracer registered with majit-gc so `walk_jf_roots` can reach
/// the interior Ref slots of our libc-allocated jitframes. The
/// collector expects a callback that, given a jitframe payload
/// address, reports each Ref slot pointer via `update`.
///
/// `jitframe_trace` reads `jf_gcmap` to know which of the trailing
/// `jf_frame` slots hold Refs and calls back for each bit.
unsafe fn pyre_libc_jitframe_tracer(obj_addr: usize, update: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    unsafe {
        majit_metainterp::jitframe::jitframe_trace(
            obj_addr as *mut majit_metainterp::jitframe::JitFrame,
            |slot_ptr| {
                update(slot_ptr as *mut majit_ir::GcRef);
            },
        );
    }
}

/// Bridge pyre-object's `GcAllocHookFn` to `majit_gc::alloc_nursery_typed`.
/// pyre-object deliberately carries no majit-gc dep, so pyre-jit owns
/// the `GcRef` → `*mut u8` conversion.
fn pyre_object_gc_alloc_trampoline(type_id: u32, size: usize) -> *mut u8 {
    majit_gc::alloc_nursery_typed(type_id, size).0 as *mut u8
}

/// Task #141 trampoline for stable-address host-side allocations.
/// Routes pyre-object's stable-allocation hook to the backend's
/// `alloc_oldgen_typed`. MiniMark's old-gen is mark-sweep
/// (non-moving), so the returned pointer is safe to hold on the Rust
/// stack across subsequent allocations.
fn pyre_object_gc_alloc_stable_trampoline(type_id: u32, size: usize) -> *mut u8 {
    majit_gc::alloc_oldgen_typed(type_id, size).0 as *mut u8
}

/// Task #141 option (a) trampoline: register a caller-owned slot as
/// a GC root with the active backend. Bridges `*mut *mut u8` (the
/// pyre-object-facing shape that does not depend on majit-gc) to
/// `*mut GcRef` expected by `majit_gc::gc_add_root`. `GcRef` is
/// `#[repr(transparent)]` over `usize`, so the pointer-pointer and
/// `*mut GcRef` share representation.
///
/// # Safety
/// Caller must keep `slot` valid until
/// [`pyre_object_gc_remove_root_trampoline`] is called with the same
/// pointer.
unsafe fn pyre_object_gc_add_root_trampoline(slot: *mut *mut u8) {
    unsafe { majit_gc::gc_add_root(slot as *mut majit_ir::GcRef) };
}

/// Companion to [`pyre_object_gc_add_root_trampoline`].
fn pyre_object_gc_remove_root_trampoline(slot: *mut *mut u8) {
    majit_gc::gc_remove_root(slot as *mut majit_ir::GcRef);
}

/// Bridge pyre-object's `is_managed_heap_object` query to
/// `majit_gc::gc_owns_object`. Used by host-side allocators
/// (`pyre_object::dealloc_items_block`) to discriminate
/// `try_gc_alloc_stable`-allocated blocks from `std::alloc`-backed
/// fallback blocks.
fn pyre_object_gc_owns_object_trampoline(addr: usize) -> bool {
    majit_gc::gc_owns_object(addr)
}

/// resume.py:1312 blackhole_from_resumedata parity: preserve per-frame
/// resume data from the last guard failure. rd_numb provides frame
/// boundaries (jitcode_index, pc); values are resolved from deadframe.
thread_local! {
    static LAST_GUARD_FRAMES: std::cell::RefCell<Option<Vec<crate::call_jit::ResumedFrame>>> =
        const { std::cell::RefCell::new(None) };
}

/// Take the last guard frames (consuming them).
pub(crate) fn take_last_guard_frames() -> Option<Vec<crate::call_jit::ResumedFrame>> {
    LAST_GUARD_FRAMES.with(|c| c.borrow_mut().take())
}

/// RPython jitexc.py:53 ContinueRunningNormally parity.
pub(crate) enum LoopResult {
    Done(PyResult),
    ContinueRunningNormally,
}

/// Action from handle_jit_outcome for eval_loop_jit dispatch.
enum JitAction {
    Return(PyResult),
    Continue,
    /// RPython jitexc.py:53: guard-restored → restart portal.
    ContinueRunningNormally,
}

use crate::jit::descr::{
    BUILTIN_CODE_GC_TYPE_ID, FUNCTION_GC_TYPE_ID, JITFRAME_GC_TYPE_ID, OBJECT_GC_TYPE_ID,
    PY_OBJECT_ARRAY_GC_TYPE_ID, PYFRAME_GC_TYPE_ID, RANGE_ITER_GC_TYPE_ID,
    SPECIALISED_TUPLE_FF_GC_TYPE_ID, SPECIALISED_TUPLE_II_GC_TYPE_ID,
    SPECIALISED_TUPLE_OO_GC_TYPE_ID, VREF_GC_TYPE_ID, W_BOOL_GC_TYPE_ID, W_BYTEARRAY_GC_TYPE_ID,
    W_BYTES_GC_TYPE_ID, W_CELL_GC_TYPE_ID, W_CLASSMETHOD_GC_TYPE_ID, W_COUNT_GC_TYPE_ID,
    W_DICT_GC_TYPE_ID, W_EXCEPTION_GC_TYPE_ID, W_FLOAT_GC_TYPE_ID, W_GENERATOR_GC_TYPE_ID,
    W_INT_GC_TYPE_ID, W_LIST_GC_TYPE_ID, W_LONG_GC_TYPE_ID, W_MEMBER_GC_TYPE_ID,
    W_METHOD_GC_TYPE_ID, W_MODULE_GC_TYPE_ID, W_PROPERTY_GC_TYPE_ID, W_REPEAT_GC_TYPE_ID,
    W_SEQ_ITER_GC_TYPE_ID, W_SET_GC_TYPE_ID, W_SLICE_GC_TYPE_ID, W_STATICMETHOD_GC_TYPE_ID,
    W_STR_GC_TYPE_ID, W_SUPER_GC_TYPE_ID, W_TUPLE_GC_TYPE_ID, W_TYPE_GC_TYPE_ID,
    W_UNION_GC_TYPE_ID,
};
use majit_gc::collector::MiniMarkGC;
use majit_metainterp::JitDriver;
use pyre_jit_trace::frame_layout::build_pyframe_virtualizable_info;
use pyre_object::floatobject::{FLOAT_FLOATVAL_OFFSET, W_FloatObject};
use pyre_object::intobject::{INT_INTVAL_OFFSET, W_IntObject};
use pyre_object::{w_bool_from, w_int_new, w_none, w_str_new, w_tuple_new};

const JIT_THRESHOLD: u32 = 200;
type JitDriverPair = (
    JitDriver<PyreJitState>,
    std::sync::Arc<majit_metainterp::virtualizable::VirtualizableInfo>,
);

thread_local! {
    static JIT_DRIVER: UnsafeCell<JitDriverPair> = UnsafeCell::new({
        let info = build_pyframe_virtualizable_info();
        let mut d = JitDriver::new(JIT_THRESHOLD);
        d.set_virtualizable_info(info.clone());
        d.meta_interp_mut().num_scalar_inputargs =
            pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        // info.py:810-822 `ConstPtrInfo.getstrlen1(mode)` — install pyre's
        // `W_StrObject` length reader so constant STRLEN / UNICODELEN ops
        // fold to `IntBound::from_constant(len)` during intbounds
        // postprocessing.
        //
        // PyPy returns the exact length for both modes:
        //
        //     def getstrlen1(self, mode):
        //         from rpython.jit.metainterp.optimizeopt import vstring
        //         if mode is vstring.mode_string:
        //             s = self._unpack_str(vstring.mode_string)
        //             ...
        //             return len(s)
        //         elif mode is vstring.mode_unicode:
        //             s = self._unpack_str(vstring.mode_unicode)
        //             ...
        //             return len(s)
        //
        // Pyre's `W_StrObject.value` is a Rust `String` whose
        // `len()` returns the UTF-8 BYTE length and whose
        // `chars().count()` returns the codepoint count, so the resolver
        // needs different reads per mode:
        //
        //   * mode == 0 (`vstring.mode_string`, byte string) — return the
        //     UTF-8 byte length, which is what PyPy's `str.len()` would
        //     produce for an RPython byte string.
        //   * mode == 1 (`vstring.mode_unicode`, unicode string) — return
        //     the codepoint count, which is what Python 3's
        //     `len(str_object)` produces.
        d.meta_interp_mut().set_string_length_resolver(std::sync::Arc::new(
            |gcref: majit_ir::GcRef, mode: u8| -> Option<i64> {
                if gcref.is_null() {
                    return None;
                }
                let obj = gcref.0 as pyre_object::pyobject::PyObjectRef;
                if !unsafe { pyre_object::strobject::is_str(obj) } {
                    return None;
                }
                match mode {
                    // vstring.mode_string — UTF-8 byte length.
                    0 => Some(unsafe { pyre_object::strobject::w_str_len(obj) } as i64),
                    // vstring.mode_unicode — codepoint count.
                    1 => {
                        let s = unsafe { pyre_object::strobject::w_str_get_value(obj) };
                        Some(s.chars().count() as i64)
                    }
                    _ => None,
                }
            },
        ));
        let mut gc = MiniMarkGC::new();
        // rclass.OBJECT root (rclass.py:160-166). pyre's static
        // `INSTANCE_TYPE` is the `name = "object"` PyType — every
        // other `PyObject`-layout class chains its `parent` field to
        // this id so `assign_inheritance_ids` (normalizecalls.py:373-389)
        // produces a `subclassrange_{min,max}` covering every
        // descendant. The size is `sizeof(PyObject)` because instances
        // tagged with `&INSTANCE_TYPE` (i.e. user `object()` calls)
        // carry only the `ob_type` header.
        let object_tid =
            gc.register_type(TypeInfo::object(std::mem::size_of::<pyre_object::PyObject>()));
        debug_assert_eq!(object_tid, OBJECT_GC_TYPE_ID);
        // W_IntObject / W_FloatObject carry `PyObject.ob_type` at offset 0,
        // matching RPython `rclass.OBJECT` layout (T_IS_RPYTHON_INSTANCE,
        // gc.py:642). They are NewWithVtable allocation targets so the
        // payload size must be the actual struct size, and they sit one
        // level below the OBJECT root (`int.__bases__ == (object,)`,
        // `float.__bases__ == (object,)`).
        let w_int_tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<W_IntObject>(),
            object_tid,
        ));
        debug_assert_eq!(w_int_tid, W_INT_GC_TYPE_ID);
        let w_float_tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<W_FloatObject>(),
            object_tid,
        ));
        debug_assert_eq!(w_float_tid, W_FLOAT_GC_TYPE_ID);
        // jitframe.py:49 — rgc.register_custom_trace_hook(JITFRAME, jitframe_trace)
        let jitframe_tid = gc.register_type(majit_metainterp::jitframe::jitframe_type_info());
        debug_assert_eq!(jitframe_tid, JITFRAME_GC_TYPE_ID);
        // pyre allocates jitframes via `libc::calloc` (not nursery/oldgen),
        // so the collector's standard `walk_jf_roots` visitor can't
        // route them through `trace_and_update_object`. Register a
        // host-side tracer that invokes `jitframe_trace` directly so
        // Refs pinned to frame slots are visible to GC across minor
        // collections triggered by CallMallocNursery slow paths.
        majit_gc::shadow_stack::register_libc_jitframe_tracer(
            pyre_libc_jitframe_tracer,
        );
        // virtualref.py — JIT_VIRTUAL_REF as a proper GC type.
        // Layout: type_tag(u64, offset 0) | virtual_token(i64, offset 8) | forced(Ref, offset 16)
        // `forced` is a GC pointer → gc_ptr_offsets = [16].
        let vref_tid = gc.register_type(majit_gc::trace::TypeInfo::with_gc_ptrs(
            std::mem::size_of::<majit_metainterp::virtualref::JitVirtualRef>(),
            vec![std::mem::offset_of!(majit_metainterp::virtualref::JitVirtualRef, forced)],
        ));
        debug_assert_eq!(vref_tid, VREF_GC_TYPE_ID);
        // Tell the virtualref optimizer about the registered type id.
        majit_metainterp::virtualref::set_vref_gc_type_id(vref_tid);
        // Dedicated typeids for the JIT-NEW'd / JIT-guard'd PyObject
        // subclasses whose payload is NOT `sizeof(PyObject)`. RPython
        // registers one typeid per distinct STRUCT through
        // `heaptracker.setup_cache_gcstruct2vtable` (heaptracker.py:23-30)
        // and `add_vtable_after_typeinfo` (gctypelayout.py:359-374). pyre's
        // earlier one-typeid-per-root-layout approximation under-walked
        // lists/tuples/range-iters as soon as their descr groups carried
        // `type_id = 0`. `gc_ptr_offsets` stays empty for all four — this
        // slice is pure bookkeeping (see gc_retype_epic_plan_2026_04_24.md
        // Session 1); Sessions 2-5 will add the real pointer fields.
        let w_bool_tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::boolobject::W_BoolObject>(),
            w_int_tid,
        ));
        debug_assert_eq!(w_bool_tid, W_BOOL_GC_TYPE_ID);
        let range_iter_tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::rangeobject::W_RangeIterator>(),
            object_tid,
        ));
        debug_assert_eq!(range_iter_tid, RANGE_ITER_GC_TYPE_ID);
        // rlist.py:116 parity: W_ListObject has a single GC pointer
        // field — `items: Ptr(GcArray(OBJECTPTR))` — directly at
        // `offset_of!(items)`. Phase L1 retired the `PyObjectArray`
        // fat wrapper, so the GC offset no longer goes through an
        // intermediate block-start field.
        //
        // STEPPING-STONE (Phase L2 pending). The pointer shape is now
        // correct but the block `items` points to is still
        // `std::alloc`-owned (`alloc_items_block` in
        // `pyre_object::object_array`), so
        // `is_nursery_object_start` (collector.rs:377) rejects it
        // and the walker no-ops. Activates end-to-end only after
        // Phase L2's allocator cutover (blocked on Task #141 GC-root
        // infrastructure). Do NOT promote `W_LIST_GC_TYPE_ID` to
        // "fully-parity" status in docs/MEMORY while this stepping-
        // stone state holds.
        let mut w_list_ti = TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::listobject::W_ListObject>(),
            object_tid,
        );
        w_list_ti.gc_ptr_offsets = vec![std::mem::offset_of!(
            pyre_object::listobject::W_ListObject,
            items
        )];
        w_list_ti.has_gc_ptrs = true;
        let w_list_tid = gc.register_type(w_list_ti);
        debug_assert_eq!(w_list_tid, W_LIST_GC_TYPE_ID);
        // Same stepping-stone caveat as W_LIST above; Phase T1-full
        // (specialised arity-2 variants per
        // `pypy/objspace/std/specialisedtupleobject.py`) + Phase L2
        // allocator cutover together complete the tuple convergence.
        let mut w_tuple_ti = TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::tupleobject::W_TupleObject>(),
            object_tid,
        );
        w_tuple_ti.gc_ptr_offsets = vec![std::mem::offset_of!(
            pyre_object::tupleobject::W_TupleObject,
            wrappeditems
        )];
        w_tuple_ti.has_gc_ptrs = true;
        let w_tuple_tid = gc.register_type(w_tuple_ti);
        debug_assert_eq!(w_tuple_tid, W_TUPLE_GC_TYPE_ID);
        // `rlist.py Ptr(GcArray(OBJECTPTR))` — the variable-length
        // backing block behind `PyObjectArray`. `base=8` single-slot
        // header (`capacity`), `item_size=8` Ref, `length_offset=0`
        // so `gctypelayout.py:266-291` reads `capacity` as the
        // GcArray length (rlist.py:251 `len(l.items)` = allocated
        // slot count — upstream's GcArray header IS the capacity,
        // not live length).  `items_have_gc_ptrs=true` activates
        // `T_IS_GCARRAY_OF_GCPTR` so the nursery walker traces every
        // item slot as a Ref; NULL-initialized spare slots past the
        // live length are benign.
        //
        // STEPPING-STONE (metadata precedes runtime). This typeid
        // only governs blocks allocated *through the GC*. Today
        // `alloc_items_block` uses `std::alloc::alloc`, so no
        // concrete allocation carries this typeid at runtime — the
        // registration shapes the GC's type table but no walker
        // ever visits a PY_OBJECT_ARRAY_GC_TYPE_ID-tagged object.
        // Activates once Phase L2 swaps the allocator (blocked on
        // Task #141). See comments on
        // `pyre_jit_trace::descr::PY_OBJECT_ARRAY_GC_TYPE_ID` and
        // `pyre_object::object_array::ItemsBlock` for the
        // companion stepping-stone notices.
        let py_object_array_tid = gc.register_type(TypeInfo::varsize(
            pyre_object::object_array::ITEMS_BLOCK_ITEMS_OFFSET,
            std::mem::size_of::<pyre_object::pyobject::PyObjectRef>(),
            0,
            true,
            Vec::new(),
        ));
        debug_assert_eq!(py_object_array_tid, PY_OBJECT_ARRAY_GC_TYPE_ID);
        // `pypy/objspace/std/specialisedtupleobject.py` `Cls_ii / Cls_ff
        // / Cls_oo` — three subclasses of `W_AbstractTupleObject` with
        // inline `value0` / `value1` fields. Each gets a distinct
        // `ob_type` so the JIT's `GUARD_CLASS` reaches the inline-field
        // shape directly. `Cls_oo` carries two GC-pointer slots; the
        // other two are GC-leaf for the payload (header still has w_class).
        let mut spec_tuple_ii_ti = TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::specialisedtupleobject::W_SpecialisedTupleObject_ii>(),
            object_tid,
        );
        spec_tuple_ii_ti.has_gc_ptrs = false;
        let spec_tuple_ii_tid = gc.register_type(spec_tuple_ii_ti);
        debug_assert_eq!(spec_tuple_ii_tid, SPECIALISED_TUPLE_II_GC_TYPE_ID);
        let mut spec_tuple_ff_ti = TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::specialisedtupleobject::W_SpecialisedTupleObject_ff>(),
            object_tid,
        );
        spec_tuple_ff_ti.has_gc_ptrs = false;
        let spec_tuple_ff_tid = gc.register_type(spec_tuple_ff_ti);
        debug_assert_eq!(spec_tuple_ff_tid, SPECIALISED_TUPLE_FF_GC_TYPE_ID);
        let mut spec_tuple_oo_ti = TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::specialisedtupleobject::W_SpecialisedTupleObject_oo>(),
            object_tid,
        );
        spec_tuple_oo_ti.gc_ptr_offsets = vec![
            pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_VALUE0_OFFSET,
            pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_VALUE1_OFFSET,
        ];
        spec_tuple_oo_ti.has_gc_ptrs = true;
        let spec_tuple_oo_tid = gc.register_type(spec_tuple_oo_ti);
        debug_assert_eq!(spec_tuple_oo_tid, SPECIALISED_TUPLE_OO_GC_TYPE_ID);
        // Tell the cranelift backend which type id to use for the
        // nursery allocations that it issues for jitframes. Without
        // this, the backend's default u32::MAX sentinel would trip the
        // allocation assert in run_compiled_code_inner, or — worse,
        // before this fix — the backend's stale hard-coded `2` would
        // collide with W_FLOAT_GC_TYPE_ID and GC would copy jitframes
        // with the wrong TypeInfo (24-byte float payload instead of
        // the real 64 + 8*depth layout), silently truncating every
        // ref root slot past the first three bytes.
        #[cfg(feature = "cranelift")]
        majit_backend_cranelift::set_jitframe_gc_type_id(jitframe_tid);
        #[cfg(feature = "dynasm")]
        majit_backend_dynasm::set_jitframe_gc_type_id(jitframe_tid);
        // llsupport/gc.py:563 vtable→typeid mapping. RPython derives the
        // typeid arithmetically from gc_get_type_info_group; pyre keeps an
        // explicit table because every PyType is a static global
        // unrelated to the GC's internal layout. The OBJECT root and
        // INT/FLOAT are wired up first so subsequent foreign-pytype
        // entries can resolve their parents through the same map.
        let mut pytype_to_tid: HashMap<usize, u32> = HashMap::new();
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::pyobject::INSTANCE_TYPE as *const _ as usize,
            object_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::pyobject::INSTANCE_TYPE as *const _ as usize,
            object_tid,
        );
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::pyobject::INT_TYPE as *const _ as usize,
            w_int_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::pyobject::INT_TYPE as *const _ as usize,
            w_int_tid,
        );
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::pyobject::FLOAT_TYPE as *const _ as usize,
            w_float_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::pyobject::FLOAT_TYPE as *const _ as usize,
            w_float_tid,
        );
        // Bind the four dedicated typeids registered above to their
        // static PyType pointers. The foreign-pytype loop below skips
        // any PyType already present in `pytype_to_tid`, so these four
        // pre-bindings override the loop's would-be
        // `object_subclass(sizeof(PyObject))` registration with the
        // correct per-struct size.
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::pyobject::BOOL_TYPE as *const _ as usize,
            w_bool_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::pyobject::BOOL_TYPE as *const _ as usize,
            w_bool_tid,
        );
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::rangeobject::RANGE_ITER_TYPE as *const _ as usize,
            range_iter_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::rangeobject::RANGE_ITER_TYPE as *const _ as usize,
            range_iter_tid,
        );
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::pyobject::LIST_TYPE as *const _ as usize,
            w_list_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::pyobject::LIST_TYPE as *const _ as usize,
            w_list_tid,
        );
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::pyobject::TUPLE_TYPE as *const _ as usize,
            w_tuple_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::pyobject::TUPLE_TYPE as *const _ as usize,
            w_tuple_tid,
        );
        // BuiltinCode is pre-registered (rather than picked up by the
        // foreign-pytype loop below) because the loop hard-codes
        // `size_of::<PyObject>()` as the payload size, while the
        // GC needs `size_of::<BuiltinCode>()` to walk live instances
        // correctly. Mirror W_INT/W_FLOAT pattern so future GC
        // integration finds an already-registered tid + size pair.
        let builtin_code_tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<pyre_interpreter::gateway::BuiltinCode>(),
            object_tid,
        ));
        debug_assert_eq!(builtin_code_tid, BUILTIN_CODE_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_interpreter::gateway::BUILTIN_CODE_TYPE as *const _ as usize,
            builtin_code_tid,
        );
        pytype_to_tid.insert(
            &pyre_interpreter::gateway::BUILTIN_CODE_TYPE as *const _ as usize,
            builtin_code_tid,
        );
        // Function carries 4 inline `PyObjectRef` fields (closure /
        // defs_w / w_kw_defs / w_module) that the collector must walk
        // — `object_subclass_with_gc_ptrs` records the offsets so
        // mark traversal reaches them. `BUILTIN_FUNCTION_TYPE` is a
        // separate static `PyType` for module-level builtins
        // (`pypy/interpreter/function.py:706 BuiltinFunction`) but its
        // instances are the same Rust struct, so the vtable map sends
        // both PyTypes to `function_tid`.
        let function_tid =
            gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
                std::mem::size_of::<pyre_interpreter::function::Function>(),
                object_tid,
                pyre_interpreter::function::FUNCTION_GC_PTR_OFFSETS.to_vec(),
            ));
        debug_assert_eq!(function_tid, FUNCTION_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_interpreter::function::FUNCTION_TYPE as *const _ as usize,
            function_tid,
        );
        pytype_to_tid.insert(
            &pyre_interpreter::function::FUNCTION_TYPE as *const _ as usize,
            function_tid,
        );
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_interpreter::function::BUILTIN_FUNCTION_TYPE as *const _ as usize,
            function_tid,
        );
        pytype_to_tid.insert(
            &pyre_interpreter::function::BUILTIN_FUNCTION_TYPE as *const _ as usize,
            function_tid,
        );
        // W_CellObject (closure cell) is pre-registered with its real
        // payload size and `gc_ptr_offsets = [contents]`, mirroring the
        // BuiltinCode/Function pattern. The foreign-pytype loop hard-codes
        // `size_of::<PyObject>()`, missing the `contents` slot — so any
        // cell live across a minor collection would lose the value if it
        // went through the loop.
        let w_cell_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::cellobject::W_CellObject>(),
            object_tid,
            pyre_object::cellobject::W_CELL_GC_PTR_OFFSETS.to_vec(),
        ));
        debug_assert_eq!(w_cell_tid, W_CELL_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::cellobject::CELL_TYPE as *const _ as usize,
            w_cell_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::cellobject::CELL_TYPE as *const _ as usize,
            w_cell_tid,
        );
        // W_MethodObject (bound method) carries 3 inline `PyObjectRef`
        // fields (w_function / w_self / w_class). Pre-registered ahead
        // of the foreign-pytype loop for the same reason as W_Cell:
        // the loop's `size_of::<PyObject>()` approximation drops the
        // gc_ptr_offsets, leaving live methods unscanned across a
        // minor collection.
        let w_method_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::methodobject::W_MethodObject>(),
            object_tid,
            pyre_object::methodobject::W_METHOD_GC_PTR_OFFSETS.to_vec(),
        ));
        debug_assert_eq!(w_method_tid, W_METHOD_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::methodobject::METHOD_TYPE as *const _ as usize,
            w_method_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::methodobject::METHOD_TYPE as *const _ as usize,
            w_method_tid,
        );
        // W_SliceObject (Python slice) carries 3 inline `PyObjectRef`
        // fields (start / stop / step). Pre-registered ahead of the
        // foreign-pytype loop for the same reason as W_Cell/W_Method.
        let w_slice_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::sliceobject::W_SliceObject>(),
            object_tid,
            pyre_object::sliceobject::W_SLICE_GC_PTR_OFFSETS.to_vec(),
        ));
        debug_assert_eq!(w_slice_tid, W_SLICE_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::sliceobject::SLICE_TYPE as *const _ as usize,
            w_slice_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::sliceobject::SLICE_TYPE as *const _ as usize,
            w_slice_tid,
        );
        // W_SuperObject (super proxy) carries 2 inline `PyObjectRef`
        // fields (super_type / obj). Pre-registered ahead of the
        // foreign-pytype loop for the same reason as W_Cell/W_Method/
        // W_Slice.
        let w_super_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::superobject::W_SuperObject>(),
            object_tid,
            pyre_object::superobject::W_SUPER_GC_PTR_OFFSETS.to_vec(),
        ));
        debug_assert_eq!(w_super_tid, W_SUPER_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::superobject::SUPER_TYPE as *const _ as usize,
            w_super_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::superobject::SUPER_TYPE as *const _ as usize,
            w_super_tid,
        );
        // W_PropertyObject (3 PyObjectRef fields: fget/fset/fdel),
        // W_StaticMethodObject and W_ClassMethodObject (1 PyObjectRef
        // field each: w_function). Pre-registered ahead of the
        // foreign-pytype loop so the GC walker reaches the inline
        // descriptor refs.
        let w_property_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::propertyobject::W_PropertyObject>(),
            object_tid,
            pyre_object::propertyobject::W_PROPERTY_GC_PTR_OFFSETS.to_vec(),
        ));
        debug_assert_eq!(w_property_tid, W_PROPERTY_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::propertyobject::PROPERTY_TYPE as *const _ as usize,
            w_property_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::propertyobject::PROPERTY_TYPE as *const _ as usize,
            w_property_tid,
        );
        let w_staticmethod_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::propertyobject::W_StaticMethodObject>(),
            object_tid,
            pyre_object::propertyobject::W_STATICMETHOD_GC_PTR_OFFSETS.to_vec(),
        ));
        debug_assert_eq!(w_staticmethod_tid, W_STATICMETHOD_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::propertyobject::STATICMETHOD_TYPE as *const _ as usize,
            w_staticmethod_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::propertyobject::STATICMETHOD_TYPE as *const _ as usize,
            w_staticmethod_tid,
        );
        let w_classmethod_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::propertyobject::W_ClassMethodObject>(),
            object_tid,
            pyre_object::propertyobject::W_CLASSMETHOD_GC_PTR_OFFSETS.to_vec(),
        ));
        debug_assert_eq!(w_classmethod_tid, W_CLASSMETHOD_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::propertyobject::CLASSMETHOD_TYPE as *const _ as usize,
            w_classmethod_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::propertyobject::CLASSMETHOD_TYPE as *const _ as usize,
            w_classmethod_tid,
        );
        // W_UnionType (PEP 604 `X | Y`) carries one inline `PyObjectRef`
        // field (`args` — tuple of union members). Pre-registered ahead
        // of the foreign-pytype loop for the same reason as W_Cell:
        // the loop's `size_of::<PyObject>()` approximation drops the
        // gc_ptr_offsets, leaving live unions unscanned across a
        // minor collection.
        let w_union_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::unionobject::W_UnionType>(),
            object_tid,
            pyre_object::unionobject::W_UNION_GC_PTR_OFFSETS.to_vec(),
        ));
        debug_assert_eq!(w_union_tid, W_UNION_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::unionobject::UNION_TYPE as *const _ as usize,
            w_union_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::unionobject::UNION_TYPE as *const _ as usize,
            w_union_tid,
        );
        // W_SeqIterator (list/tuple iterator) carries one inline
        // `PyObjectRef` field (`seq`) plus two i64 scalars
        // (`index`/`length`). Pre-registered ahead of the foreign-pytype
        // loop so the GC walker reaches `seq`.
        let w_seq_iter_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::rangeobject::W_SeqIterator>(),
            object_tid,
            pyre_object::rangeobject::W_SEQ_ITER_GC_PTR_OFFSETS.to_vec(),
        ));
        debug_assert_eq!(w_seq_iter_tid, W_SEQ_ITER_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::rangeobject::SEQ_ITER_TYPE as *const _ as usize,
            w_seq_iter_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::rangeobject::SEQ_ITER_TYPE as *const _ as usize,
            w_seq_iter_tid,
        );
        // W_Count (`itertools.count`) carries two inline `PyObjectRef`
        // fields (`w_c` / `w_step`) so the iterator state survives a
        // minor collection. `COUNT_TYPE` is not in
        // `all_foreign_pytypes()`, so the foreign-pytype loop never
        // visits it; pre-registration is the only path through which
        // its instances become GC-managed.
        let w_count_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::itertoolsmodule::W_Count>(),
            object_tid,
            pyre_object::itertoolsmodule::W_COUNT_GC_PTR_OFFSETS.to_vec(),
        ));
        debug_assert_eq!(w_count_tid, W_COUNT_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::itertoolsmodule::COUNT_TYPE as *const _ as usize,
            w_count_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::itertoolsmodule::COUNT_TYPE as *const _ as usize,
            w_count_tid,
        );
        // W_Repeat (`itertools.repeat`) carries one inline
        // `PyObjectRef` field (`w_obj`) plus a bool/i64 pair. Same
        // foreign-pytype caveat as W_Count.
        let w_repeat_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::itertoolsmodule::W_Repeat>(),
            object_tid,
            pyre_object::itertoolsmodule::W_REPEAT_GC_PTR_OFFSETS.to_vec(),
        ));
        debug_assert_eq!(w_repeat_tid, W_REPEAT_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::itertoolsmodule::REPEAT_TYPE as *const _ as usize,
            w_repeat_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::itertoolsmodule::REPEAT_TYPE as *const _ as usize,
            w_repeat_tid,
        );
        // W_MemberDescr (`__slots__` member descriptor) carries one
        // inline `PyObjectRef` field (`w_cls`) plus a `*const String`
        // (`name`) and a `u32` index. Pre-registered ahead of the
        // foreign-pytype loop so the GC walker reaches `w_cls`. The
        // `name` pointer is intentionally outside `gc_ptr_offsets`
        // because it points into a non-PyObject heap allocation.
        let w_member_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::memberobject::W_MemberDescr>(),
            object_tid,
            pyre_object::memberobject::W_MEMBER_GC_PTR_OFFSETS.to_vec(),
        ));
        debug_assert_eq!(w_member_tid, W_MEMBER_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::memberobject::MEMBER_TYPE as *const _ as usize,
            w_member_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::memberobject::MEMBER_TYPE as *const _ as usize,
            w_member_tid,
        );
        // W_BytesObject (immutable byte sequence) carries a raw
        // `*const Vec<u8>` (`data`) and a `usize` length, neither a
        // `PyObjectRef`. Pre-registered with `object_subclass(size, ...)`
        // so the foreign-pytype loop's `sizeof(PyObject)` approximation
        // does not under-count the payload.
        let w_bytes_tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::bytesobject::W_BytesObject>(),
            object_tid,
        ));
        debug_assert_eq!(w_bytes_tid, W_BYTES_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::bytesobject::BYTES_TYPE as *const _ as usize,
            w_bytes_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::bytesobject::BYTES_TYPE as *const _ as usize,
            w_bytes_tid,
        );
        // W_BytearrayObject (mutable byte sequence) carries a raw
        // `*mut Vec<u8>` (`data`). Same registration shape as
        // W_BytesObject.
        let w_bytearray_tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::bytearrayobject::W_BytearrayObject>(),
            object_tid,
        ));
        debug_assert_eq!(w_bytearray_tid, W_BYTEARRAY_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::bytearrayobject::BYTEARRAY_TYPE as *const _ as usize,
            w_bytearray_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::bytearrayobject::BYTEARRAY_TYPE as *const _ as usize,
            w_bytearray_tid,
        );
        // W_DictObject carries `entries: *mut Vec<...>` (raw heap),
        // a `usize` length, and `dict_storage_proxy: *mut u8`. None
        // of those are direct `PyObjectRef` fields (the (key, value)
        // pairs live behind a raw `Vec` pointer), so registration is
        // size-only. The Vec's PyObjectRefs reaching the GC is a
        // pre-existing limitation common to set/dict storage.
        let w_dict_tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::dictobject::W_DictObject>(),
            object_tid,
        ));
        debug_assert_eq!(w_dict_tid, W_DICT_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::DICT_TYPE as *const _ as usize,
            w_dict_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::DICT_TYPE as *const _ as usize,
            w_dict_tid,
        );
        // W_SetObject carries `items: *mut Vec<PyObjectRef>` and a
        // `usize` length. Same size-only registration shape as
        // W_DictObject. Both `set` and `frozenset` PyTypes share the
        // `W_SetObject` Rust struct so they map to the same tid.
        let w_set_tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::setobject::W_SetObject>(),
            object_tid,
        ));
        debug_assert_eq!(w_set_tid, W_SET_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::setobject::SET_TYPE as *const _ as usize,
            w_set_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::setobject::SET_TYPE as *const _ as usize,
            w_set_tid,
        );
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::setobject::FROZENSET_TYPE as *const _ as usize,
            w_set_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::setobject::FROZENSET_TYPE as *const _ as usize,
            w_set_tid,
        );
        // W_ExceptionObject carries an `ExcKind` tag and a
        // `*mut String` pointer (raw heap, not a `PyObjectRef`).
        // Pre-registered with `object_subclass(size, ...)` so the
        // foreign-pytype loop's `sizeof(PyObject)` approximation does
        // not under-count the payload.
        let w_exception_tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::excobject::W_ExceptionObject>(),
            object_tid,
        ));
        debug_assert_eq!(w_exception_tid, W_EXCEPTION_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::excobject::EXCEPTION_TYPE as *const _ as usize,
            w_exception_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::excobject::EXCEPTION_TYPE as *const _ as usize,
            w_exception_tid,
        );
        // W_GeneratorObject carries `frame_ptr: *mut u8` (opaque
        // PyFrame pointer, owned by the generator) plus three bools.
        // No direct `PyObjectRef` fields; the suspended frame's
        // PyObjectRefs are reachable through the generator only via
        // the PyFrame indirection (pre-existing limitation).
        let w_generator_tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::generatorobject::W_GeneratorObject>(),
            object_tid,
        ));
        debug_assert_eq!(w_generator_tid, W_GENERATOR_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::generatorobject::GENERATOR_TYPE as *const _ as usize,
            w_generator_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::generatorobject::GENERATOR_TYPE as *const _ as usize,
            w_generator_tid,
        );
        // W_TypeObject carries one inline `PyObjectRef` (`bases`)
        // plus several non-PyObject raw pointers (`name`, `dict`,
        // `mro_w`, `layout`). Pre-registered ahead of the
        // foreign-pytype loop because `TYPE_TYPE` is in
        // `all_foreign_pytypes()` and the loop's
        // `sizeof(PyObject)` approximation drastically under-counts
        // the W_TypeObject payload.
        let w_type_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::typeobject::W_TypeObject>(),
            object_tid,
            pyre_object::typeobject::W_TYPE_GC_PTR_OFFSETS.to_vec(),
        ));
        debug_assert_eq!(w_type_tid, W_TYPE_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::TYPE_TYPE as *const _ as usize,
            w_type_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::TYPE_TYPE as *const _ as usize,
            w_type_tid,
        );
        // W_StrObject carries a `*mut String` (raw heap) plus a
        // `usize` length. No direct `PyObjectRef` field. Pre-registered
        // so the foreign-pytype loop's `sizeof(PyObject)` approximation
        // does not under-count the payload.
        let w_str_tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::strobject::W_StrObject>(),
            object_tid,
        ));
        debug_assert_eq!(w_str_tid, W_STR_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::STR_TYPE as *const _ as usize,
            w_str_tid,
        );
        pytype_to_tid.insert(&pyre_object::STR_TYPE as *const _ as usize, w_str_tid);
        // W_LongObject carries a `*mut BigInt` (raw heap) only. Same
        // size-only registration shape as W_StrObject.
        let w_long_tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::longobject::W_LongObject>(),
            object_tid,
        ));
        debug_assert_eq!(w_long_tid, W_LONG_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::LONG_TYPE as *const _ as usize,
            w_long_tid,
        );
        pytype_to_tid.insert(&pyre_object::LONG_TYPE as *const _ as usize, w_long_tid);
        // W_ModuleObject carries `name: *mut String` and `dict: *mut u8`,
        // both non-PyObject heap pointers.
        let w_module_tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::moduleobject::W_ModuleObject>(),
            object_tid,
        ));
        debug_assert_eq!(w_module_tid, W_MODULE_GC_TYPE_ID);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            &pyre_object::MODULE_TYPE as *const _ as usize,
            w_module_tid,
        );
        pytype_to_tid.insert(
            &pyre_object::MODULE_TYPE as *const _ as usize,
            w_module_tid,
        );
        // `pyre-interpreter::pyframe::PyFrame` — execution frame for a
        // Python code block. NOT an `rclass.OBJECT`-shaped instance
        // (no `ob_type` header — virtualizable struct laid out for the
        // JIT virtualize pass), so register through `with_gc_ptrs`
        // rather than `object_subclass`.
        //
        // GC-traceable fields (the standard nursery tracer forwards
        // these when a nursery `PyFrame` survives a minor collection):
        //   - `locals_cells_stack_w`: `*mut FixedObjectArray` — when
        //     `emit_new_pyframe_inline_self_recursive` emits
        //     `NewArrayClear` for the locals array the array itself
        //     lives in the nursery, so its pointer must be forwarded.
        //   - `f_generator_nowref` / `w_yielding_from`: `PyObjectRef`
        //     slots that may point at nursery objects.
        //   - `f_backref`: `*mut PyFrame` — once chained inline calls
        //     produce nested nursery `PyFrame`s the parent pointer
        //     must be forwarded to the new address.
        // Excluded: `execution_context` (Rc::into_raw, persistent),
        // `pycode` (static W_CodeObject), `debugdata` / `lastblock`
        // (heap-allocated, not GC), `w_globals` (Box-allocated
        // DictStorage, not GC).
        //
        // `walk_pyframe_roots` (`pyre-interpreter::eval`) still walks
        // `CURRENT_FRAME → f_backref` and visits the items inside
        // `locals_cells_stack_w`; that path covers `std::alloc`-backed
        // PyFrames today and is the entry point that hands control to
        // the standard tracer for nursery-backed frames once
        // Slice 2 Step 2 teaches it about forwarding.
        let pyframe_tid = gc.register_type(majit_gc::trace::TypeInfo::with_gc_ptrs(
            std::mem::size_of::<pyre_interpreter::pyframe::PyFrame>(),
            vec![
                pyre_interpreter::pyframe::PYFRAME_LOCALS_CELLS_STACK_OFFSET,
                pyre_interpreter::pyframe::PYFRAME_F_GENERATOR_NOWREF_OFFSET,
                pyre_interpreter::pyframe::PYFRAME_W_YIELDING_FROM_OFFSET,
                pyre_interpreter::pyframe::PYFRAME_F_BACKREF_OFFSET,
            ],
        ));
        debug_assert_eq!(pyframe_tid, PYFRAME_GC_TYPE_ID);
        // W_InstanceObject is intentionally NOT pre-registered: its
        // PyType (`INSTANCE_TYPE`) is the `object` root, already
        // covered by `object_tid` (`OBJECT_GC_TYPE_ID = 0`). Adding
        // a separate tid for the same conceptual root would corrupt
        // the preorder-range hierarchy (`subclass_range` would yield
        // disjoint sub-ranges for the same root, breaking
        // `object ⊇ int` and friends — see eval::tests
        // ::test_subclass_range_preorder_bounds).
        // Walk every remaining built-in PyType and register one
        // `TypeInfo::object_subclass` per class, mirroring how
        // `assign_inheritance_ids` (normalizecalls.py:373-389) walks
        // `bk.bookkeeper.classdefs`. Each entry resolves its parent
        // through `pytype_to_tid`, so the resulting hierarchy obeys
        // `int_between(cls.min, subcls.min, cls.max)` (rclass.py:1133).
        // `pyre_object::pyobject::all_foreign_pytypes()` covers object
        // module PyTypes; `pyre_interpreter::all_foreign_pytypes()`
        // covers interpreter-level PyTypes (CODE_TYPE / FUNCTION_TYPE
        // / BUILTIN_CODE_TYPE) that flow through tracing as constant
        // callable/code pointers.
        for (pytype, parent) in pyre_object::pyobject::all_foreign_pytypes()
            .iter()
            .chain(pyre_interpreter::all_foreign_pytypes().iter())
        {
            let pytype_ptr = *pytype as *const _ as usize;
            // BOOL_TYPE / LIST_TYPE / TUPLE_TYPE / RANGE_ITER_TYPE are
            // pre-registered above with their real struct sizes. Leave
            // those bindings intact instead of overwriting them with a
            // `sizeof(PyObject)` approximation.
            if pytype_to_tid.contains_key(&pytype_ptr) {
                continue;
            }
            let parent_tid = *pytype_to_tid
                .get(&(*parent as *const _ as usize))
                .expect("foreign pytype parent must be registered before its subclass");
            let tid = gc.register_type(TypeInfo::object_subclass(
                std::mem::size_of::<pyre_object::PyObject>(),
                parent_tid,
            ));
            majit_gc::GcAllocator::register_vtable_for_type(
                &mut gc,
                pytype_ptr,
                tid,
            );
            pytype_to_tid.insert(pytype_ptr, tid);
        }
        // rclass.py:340-346 — assign subclassrange_{min,max} to each
        // vtable entry. freeze_types() runs assign_inheritance_ids
        // (normalizecalls.py:373-389), then we write the computed ranges
        // back into the static PyType structs so that ll_issubclass
        // (rclass.py:1133-1137) can read them directly from the typeptr.
        gc.freeze_types();
        for (&classptr, &_tid) in &pytype_to_tid {
            if let Some((min, max)) = gc.subclass_range(classptr) {
                let tp = unsafe { &*(classptr as *const pyre_object::pyobject::PyType) };
                pyre_object::pyobject::assign_subclass_range(tp, min, max);
            }
        }
        d.set_gc_allocator(Box::new(gc));
        // framework.py `root_walker.walk_roots` parity: the interpreter's
        // `PyFrame.locals_cells_stack_w` stores GC refs that must survive
        // minor collection. Compiled JIT code registers its own jitframe
        // shadow stack and blackhole register banks; the interpreter
        // path (`eval_with_jit` → `eval_loop_jit`) has no equivalent
        // until we plug this extra walker in. Register once per process;
        // `register_extra_root_walker` dedups on identity.
        pyre_interpreter::eval::register_pyframe_root_walker();
        // framework.py `root_walker.walk_roots` parity for JIT-side const
        // pools: every compiled guard's `rd_consts` (resume.py:451) may
        // hold nursery-resident GC refs for TAGCONST-encoded Ref values.
        // Without this walker, minor collection would leave stale
        // pointers in `rd_consts` and the next guard failure would
        // dereference freed memory. See
        // `majit_metainterp::MetaInterp::walk_rd_consts_refs`.
        majit_gc::shadow_stack::register_extra_root_walker(rd_consts_root_walker);
        // Route pyre-object host-side allocators through the backend's
        // nursery. `set_gc_allocator` populated
        // `majit_gc::ACTIVE_ALLOC_NURSERY_TYPED` with the active
        // backend's trampoline; the one registered here converts
        // `GcRef` -> `*mut u8` for the pyre-object side. pyre-object
        // deliberately does not depend on majit-gc, so the trampoline
        // lives here.
        pyre_object::register_gc_alloc_hook(pyre_object_gc_alloc_trampoline);
        pyre_object::register_gc_alloc_stable_hook(pyre_object_gc_alloc_stable_trampoline);
        pyre_object::register_gc_root_hooks(
            pyre_object_gc_add_root_trampoline,
            pyre_object_gc_remove_root_trampoline,
        );
        pyre_object::register_gc_owns_object_hook(pyre_object_gc_owns_object_trampoline);
        // Task #145 Step 2.4 Phase 2c — host-side `pyre_object::gc_roots`
        // shadow stack mirror of `framework.shadowstack`. Pinned roots
        // come from manual `pyre_object::gc_roots::pin_root` calls
        // bracketed by `push_roots()`; the active `MiniMarkGC`
        // instance walks them through this adapter so they survive
        // across nursery collection.
        majit_gc::shadow_stack::register_extra_root_walker(pyre_object_root_walker);
        // llmodel.py:67-69 self.vtable_offset, _ = symbolic.get_field_token(
        //     rclass.OBJECT, 'typeptr', translate_support_code)
        // pyre's PyObject.ob_type is the equivalent of RPython's typeptr.
        d.set_vtable_offset(Some(pyre_object::pyobject::OB_TYPE_OFFSET));
        // resume.py:1367 — BlackholeAllocator for virtual materialization.
        d.register_blackhole_allocator(PyreBlackholeAllocator);
        // warmspot.py:1039 handle_jitexception_from_blackhole parity:
        // portal_runner is called when ContinueRunningNormally is raised
        // at a recursive portal level during blackhole execution.
        d.register_portal_runner(pyre_portal_runner);
        // pypy/module/pypyjit/interp_jit.py:72-78 PyPyJitDriver(..., is_recursive=True).
        // Drives MetaInterp.is_main_jitcode() / is_portal_jitcode dispatch
        // — without this flag the recursive-portal bookkeeping stays
        // disabled while is_main_jitcode() callers still assume it was
        // set, leaving the metadata internally inconsistent.
        d.set_is_recursive(true);
        // warmspot.py:449 — jd.result_type = getkind(portal.getreturnvar().concretetype)[0]
        // PyPy dispatch() returns W_Root → Ref.
        d.set_result_type(majit_ir::Type::Ref);
        (d, info)
    });
}

#[inline]
pub fn driver_pair() -> &'static mut JitDriverPair {
    JIT_DRIVER.with(|cell| unsafe { &mut *cell.get() })
}

/// framework.py `root_walker.walk_roots` hook for
/// `storage.rd_consts` (resume.py:451) across every live compiled
/// trace.
///
/// Registered once during `JIT_DRIVER` init (see
/// `register_extra_root_walker` call above). Routes into the
/// thread-local `JitDriver`'s `walk_rd_consts_refs`, which in turn
/// iterates `MetaInterp::compiled_loops` and visits the Ref-typed
/// entries in every `StoredExitLayout::rd_consts`.
fn rd_consts_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    // SAFETY: the GC collection happens on the same thread that owns
    // `JIT_DRIVER`; no re-entrant collection touches the MetaInterp
    // concurrently. `driver_pair()` returns a `&'static mut`, which is
    // fine because the thread-local `UnsafeCell` is single-owner.
    let pair = driver_pair();
    pair.0.walk_rd_consts_refs(visitor);
}

/// `framework.shadowstack walk_stack_root` adapter — walk every
/// pinned slot in `pyre_object::gc_roots`'s thread-local shadow
/// stack and forward each `&mut PyObjectRef` slot to the GC's
/// `&mut GcRef` visitor. Both types are pointer-sized:
/// `PyObjectRef = *mut PyObject` and `GcRef` is
/// `#[repr(transparent)]` over `usize`, so the cast is layout-safe.
fn pyre_object_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    pyre_object::gc_roots::walk_shadow_stack(|slot: &mut pyre_object::PyObjectRef| {
        // SAFETY: `PyObjectRef` and `GcRef` are both pointer-sized
        // and trivially transmutable (one is `*mut PyObject`, the
        // other is `#[repr(transparent)] struct GcRef(pub usize)`).
        // Reinterpreting the slot in place lets a moving collector
        // rewrite the address through the visitor — the rewrite is
        // observed by subsequent `pin_root` / `shadow_stack_get`
        // callers.
        let gcref: &mut majit_ir::GcRef =
            unsafe { &mut *(slot as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef) };
        visitor(gcref);
    });
}

// GREEN_KEY_ALIASES removed: compile.py:269 parity — cross-loop cut
// traces are now stored directly under the inner loop's green_key
// (cut_inner_green_key) in compile_loop, matching RPython's
// jitcell_token = cross_loop.jitcell_token. No alias dispatch needed.

/// Return a raw pointer to the thread-local VirtualizableInfo.
/// Used by the blackhole to implement BC_GETFIELD_VABLE_* bytecodes.
pub(crate) fn get_virtualizable_info() -> *const majit_metainterp::virtualizable::VirtualizableInfo
{
    let pair = driver_pair();
    std::sync::Arc::as_ptr(&pair.1)
}

/// pypy/module/pypyjit/interp_jit.py → PyPyJitDriver(JitDriver).
///
/// Mirrors RPython JitDriver (`rpython/rlib/jit.py:610-693`) field set:
/// class-level attrs (`virtualizables`, `greens`, `reds`) from
/// interp_jit.py:67-71 and constructor kwargs from interp_jit.py:72-78
/// frozen onto a single static instance, matching the upstream
/// `pypyjitdriver = PyPyJitDriver(...)` module-scope binding.
#[derive(Clone, Copy)]
pub struct PyPyJitDriver {
    /// rlib/jit.py:617 `active = True` — class attr controlling whether
    /// the marker fires.
    pub active: bool,
    /// rlib/jit.py:618 / interp_jit.py:70 `virtualizables = ['frame']`.
    pub virtualizables: &'static [&'static str],
    /// rlib/jit.py:619 / interp_jit.py:77 `name = 'pypyjit'`.
    pub name: &'static str,
    /// rlib/jit.py:620 `inline_jit_merge_point = False`.
    pub inline_jit_merge_point: bool,
    /// rlib/jit.py:649-650 / interp_jit.py:69
    /// `greens = ['next_instr', 'is_being_profiled', 'pycode']`.
    pub greens: &'static [&'static str],
    /// rlib/jit.py:652-662 / interp_jit.py:68 `reds = ['frame', 'ec']`.
    pub reds: &'static [&'static str],
    /// rlib/jit.py:653/661 — True iff `reds='auto'`.
    pub autoreds: bool,
    /// rlib/jit.py:655/662 — `len(reds)`; `None` when `autoreds`.
    pub numreds: Option<usize>,
    /// rlib/jit.py:684 — `has_unique_id = (get_unique_id is not None)`.
    /// Stays in sync with `get_unique_id` below.
    pub has_unique_id: bool,
    /// rlib/jit.py:691 `check_untranslated=True` default.
    pub check_untranslated: bool,
    /// rlib/jit.py:692 / interp_jit.py:78 `is_recursive=True`.
    pub is_recursive: bool,
    /// rlib/jit.py:693 `vec = vectorize` default False.
    pub vec: bool,

    /// rlib/jit.py:682 — `get_printable_location` hook callable.
    pub get_printable_location: Option<fn(usize, bool, pyre_object::PyObjectRef) -> String>,
    /// rlib/jit.py:683 — `get_location` hook callable.
    pub get_location: Option<fn(usize, bool, pyre_object::PyObjectRef) -> pyre_object::PyObjectRef>,
    /// rlib/jit.py:685-687 — `get_unique_id` hook callable.
    pub get_unique_id: Option<fn(usize, bool, pyre_object::PyObjectRef) -> usize>,
    /// rlib/jit.py:690 — `should_unroll_one_iteration` hook callable.
    pub should_unroll_one_iteration: Option<fn(usize, bool, pyre_object::PyObjectRef) -> bool>,
    /// rlib/jit.py:688 — `confirm_enter_jit` hook (concrete pyre signature
    /// is wired alongside S1.3 specialize_call; until then, `None`).
    pub confirm_enter_jit: Option<fn() -> bool>,
    /// rlib/jit.py:689 — `can_never_inline` hook (signature ported with S1.3).
    pub can_never_inline: Option<fn() -> bool>,
}

impl PyPyJitDriver {
    /// interp_jit.py:85-87 — jit_merge_point inside dispatch loop.
    /// API-parity stub: the merge point is handled inside
    /// `eval_loop_jit`'s `jit_merge_point_hook` until the S3 cutover
    /// replaces this with the upstream marker call.
    pub fn jit_merge_point(
        &self,
        frame: &mut PyFrame,
        ec: *const PyExecutionContext,
        next_instr: usize,
        pycode: pyre_object::PyObjectRef,
        is_being_profiled: bool,
    ) {
        let _ = (frame, ec, next_instr, pycode, is_being_profiled);
    }

    /// interp_jit.py:114-117 — can_enter_jit at back-edge.
    /// API-parity stub: handled by `eval_loop_jit`'s
    /// `maybe_compile_and_run` on `StepResult::CloseLoop`.
    pub fn can_enter_jit(
        &self,
        frame: &mut PyFrame,
        ec: *const PyExecutionContext,
        next_instr: usize,
        pycode: pyre_object::PyObjectRef,
        is_being_profiled: bool,
    ) {
        let _ = (frame, ec, next_instr, pycode, is_being_profiled);
    }
}

/// pypy/module/pypyjit/interp_jit.py:72-78 —
/// `pypyjitdriver = PyPyJitDriver(...)`.
///
/// All four upstream hook kwargs that interp_jit.py:72-76 passes are
/// wired to the per-hook pyre implementations defined later in this
/// file (`get_printable_location`, `get_location`, `get_unique_id`,
/// `should_unroll_one_iteration`). `has_unique_id` mirrors
/// `get_unique_id` per rlib/jit.py:684 so the two cannot drift.
///
/// Field defaults that match `JitDriver.__init__` (rlib/jit.py:610-693)
/// when the corresponding kwarg is not passed:
///
///   - `active = true`               ← rlib/jit.py:617 class attr.
///   - `inline_jit_merge_point = false` ← rlib/jit.py:670.
///   - `autoreds = false`            ← interp_jit.py passes a list, not 'auto'.
///   - `check_untranslated = true`   ← rlib/jit.py:674.
///   - `vec = false`                 ← rlib/jit.py:693.
///   - `confirm_enter_jit = None`    ← interp_jit.py omits the kwarg, so
///                                     `JitDriver.__init__` (rlib/jit.py:680)
///                                     leaves the slot as the class-level
///                                     `confirm_enter_jit = None` default.
///   - `can_never_inline = None`     ← same path: rlib/jit.py:681 default
///                                     because interp_jit.py omits it.
pub const pypyjitdriver: PyPyJitDriver = PyPyJitDriver {
    active: true,
    virtualizables: &["frame"],
    name: "pypyjit",
    inline_jit_merge_point: false,
    greens: &["next_instr", "is_being_profiled", "pycode"],
    reds: &["frame", "ec"],
    autoreds: false,
    numreds: Some(2),
    has_unique_id: true,
    check_untranslated: true,
    is_recursive: true,
    vec: false,
    get_printable_location: Some(get_printable_location),
    get_location: Some(get_location),
    get_unique_id: Some(get_unique_id),
    should_unroll_one_iteration: Some(should_unroll_one_iteration),
    // interp_jit.py:72-78 omits these kwargs — keep at upstream
    // default `None` (see field-default block above).
    confirm_enter_jit: None,
    can_never_inline: None,
};

/// interp_jit.py:77 — class __extend__(PyFrame)
///
/// In RPython, __extend__ adds methods to PyFrame. In Rust, PyFrame methods
/// are defined directly; this struct provides the interp_jit.py API surface.
pub struct __extend__;

impl __extend__ {
    /// interp_jit.py:79-96 — dispatch(self, pycode, next_instr, ec).
    ///
    /// RPython:
    ///   while True:
    ///       pypyjitdriver.jit_merge_point(ec=ec, frame=self, ...)
    ///       next_instr = self.handle_bytecode(co_code, next_instr, ec)
    ///   except Yield: ...
    ///   except ExitFrame: ...
    ///
    /// In pyre, the JIT-instrumented dispatch loop is eval_loop_jit().
    /// pycode and ec are stored on the frame; eval_loop_jit reads them
    /// from frame.pycode and frame.execution_context respectively.
    pub fn dispatch(
        frame: &mut PyFrame,
        _pycode: pyre_object::PyObjectRef,
        next_instr: usize,
        _ec: *const PyExecutionContext,
    ) -> PyResult {
        frame.set_last_instr_from_next_instr(next_instr);
        // interp_jit.py:79-96 dispatch: the while-True loop runs until
        // Yield or ExitFrame. ContinueRunningNormally means portal
        // re-entry (warmspot.py:976), not a silent return.
        handle_jitexception(frame)
    }

    /// interp_jit.py:98-117 — jump_absolute(self, jumpto, ec).
    ///
    /// RPython:
    ///   if we_are_jitted():
    ///       decr_by = _get_adapted_tick_counter()
    ///       self.last_instr = intmask(jumpto)
    ///       ec.bytecode_trace(self, decr_by)
    ///       jumpto = r_uint(self.last_instr)   # re-read after trace hook
    ///   pypyjitdriver.can_enter_jit(...)
    ///   return jumpto
    pub fn jump_absolute(
        frame: &mut PyFrame,
        mut jumpto: usize,
        ec: *mut PyExecutionContext,
    ) -> usize {
        if majit_metainterp::we_are_jitted() {
            let decr_by = _get_adapted_tick_counter();
            frame.set_last_instr_from_next_instr(jumpto);
            if !ec.is_null() {
                unsafe {
                    (*ec).bytecode_trace(frame as *mut PyFrame, decr_by);
                }
            }
            // Re-read: trace/profile hook may have changed the jump target
            // (interp_jit.py:112 — jumpto = r_uint(self.last_instr))
            jumpto = frame.next_instr();
        }
        // can_enter_jit is handled by eval_loop_jit's StepResult::CloseLoop
        // path which calls maybe_compile_and_run.
        jumpto
    }
}

/// interp_jit.py:119-131 — _get_adapted_tick_counter().
///
/// Normally the tick counter is decremented by 100 for every Python opcode.
/// Here, to better support JIT compilation of small loops, we decrement it
/// by a possibly smaller constant.  We get the maximum 100 when the
/// (unoptimized) trace length is at least 3200 (a bit randomly).
#[inline]
fn _get_adapted_tick_counter() -> usize {
    let (driver, _) = driver_pair();
    let trace_length = driver.current_trace_length();
    // current_trace_length() returns -1 when not tracing
    let decr_by = if trace_length < 0 {
        100 // also if current_trace_length() returned -1
    } else {
        (trace_length as usize) / 32
    };
    decr_by.clamp(1, 100)
}

#[derive(Clone, Copy)]
pub struct W_NotFromAssembler {
    space: pyre_object::PyObjectRef,
    w_callable: pyre_object::PyObjectRef,
}

impl W_NotFromAssembler {
    pub fn __init__(
        &mut self,
        space: pyre_object::PyObjectRef,
        w_callable: pyre_object::PyObjectRef,
    ) {
        self.space = space;
        self.w_callable = w_callable;
    }

    pub fn descr_call(&self, __args__: &[pyre_object::PyObjectRef]) -> Self {
        _call_not_in_trace(self.space, self.w_callable, __args__);
        *self
    }
}

pub fn not_from_assembler_new(
    space: pyre_object::PyObjectRef,
    _w_subtype: pyre_object::PyObjectRef,
    w_callable: pyre_object::PyObjectRef,
) -> W_NotFromAssembler {
    let _ = _w_subtype;
    W_NotFromAssembler { space, w_callable }
}

#[allow(unused_variables)]
pub fn _call_not_in_trace(
    space: pyre_object::PyObjectRef,
    w_callable: pyre_object::PyObjectRef,
    args: &[pyre_object::PyObjectRef],
) {
    let _ = space;
    let _ = pyre_interpreter::baseobjspace::call_function(w_callable, args);
}

#[inline]
fn green_key_from_pycode(next_instr: usize, w_pycode: pyre_object::PyObjectRef) -> Option<u64> {
    // Safety: this follows existing wrappers that treat `W_CodeObject`
    // as an owned pointer to a `CodeObject`.
    let code_ptr = unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) };
    if code_ptr.is_null() {
        return None;
    }
    Some(make_green_key(code_ptr, next_instr))
}

/// RPython interp_jit.py helper: get_printable_location.
pub fn get_printable_location(
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> String {
    let mut opcode = "<eof>".to_string();
    let mut code_name = "<unknown>".to_string();
    let code_ptr = unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) };
    if !code_ptr.is_null() {
        let code = unsafe { &*code_ptr.cast::<pyre_interpreter::CodeObject>() };
        code_name = code.obj_name.to_string();
        if let Some((instr, _)) = pyre_interpreter::decode_instruction_at(code, next_instr) {
            opcode = format!("{:?}", instr);
        }
    }
    format!("{code_name} #{next_instr} {opcode}")
}

/// RPython interp_jit.py helper: get_unique_id.
pub fn get_unique_id(
    _next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> usize {
    // A stable process-local unique-id equivalent using the code pointer.
    unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) as usize }
}

/// RPython interp_jit.py helper: get_location.
pub fn get_location(
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    let (filename, line, name, opcode) =
        match unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) } {
            x if x.is_null() => (
                "<unknown>".to_string(),
                0,
                "<unknown>".to_string(),
                "<eof>".to_string(),
            ),
            code_ptr => {
                let code = unsafe { &*code_ptr.cast::<pyre_interpreter::CodeObject>() };
                let (_opcode, opname) =
                    match pyre_interpreter::decode_instruction_at(code, next_instr) {
                        Some((instruction, _)) => {
                            (format!("{instruction:?}"), format!("{:?}", instruction))
                        }
                        None => ("<eof>".to_string(), "<eof>".to_string()),
                    };
                let line = code
                    .locations
                    .get(next_instr)
                    .and_then(|(start, _)| Some(start.line.get() as usize))
                    .unwrap_or_else(|| {
                        code.first_line_number
                            .map(|line| line.get())
                            .unwrap_or(0)
                            .saturating_add(next_instr)
                    });
                (
                    code.source_path.to_string(),
                    line,
                    code.obj_name.to_string(),
                    opname,
                )
            }
        };
    let _ = opcode;
    w_tuple_new(vec![
        w_str_new(&filename),
        w_int_new(line as i64),
        w_str_new(&name),
        w_int_new(next_instr as i64),
        w_str_new(&opcode),
    ])
}

/// RPython interp_jit.py helper: should_unroll_one_iteration.
pub fn should_unroll_one_iteration(
    _next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> bool {
    match unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) } {
        ptr if ptr.is_null() => false,
        code_ptr => {
            let code = unsafe { &*code_ptr.cast::<pyre_interpreter::CodeObject>() };
            code.flags.contains(pyre_interpreter::CodeFlags::GENERATOR)
        }
    }
}

/// interp_jit.py:216 — get_jitcell_at_key.
///
/// Returns True if a jitcell exists for this green key, regardless of
/// whether machine code has been compiled. A cell is created when the
/// counter first ticks, so this returns True even before compilation.
/// interp_jit.py:215 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn get_jitcell_at_key(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    let key = green_key_from_pycode(next_instr, w_pycode);
    let (driver, _) = driver_pair();
    w_bool_from(key.is_some_and(|green_key| {
        driver
            .meta_interp_mut()
            .warm_state_mut()
            .get_cell(green_key)
            .is_some()
    }))
}

/// interp_jit.py:222 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn dont_trace_here(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) {
    let Some(green_key) = green_key_from_pycode(next_instr, w_pycode) else {
        return;
    };
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .disable_noninlinable_function(green_key);
}

/// interp_jit.py:233 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn mark_as_being_traced(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) {
    let Some(green_key) = green_key_from_pycode(next_instr, w_pycode) else {
        return;
    };
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .mark_as_being_traced(green_key);
}

/// interp_jit.py:245 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn trace_next_iteration(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) {
    let Some(green_key) = green_key_from_pycode(next_instr, w_pycode) else {
        return;
    };
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .trace_next_iteration(green_key);
}

/// interp_jit.py:253 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn trace_next_iteration_hash(_space: pyre_object::PyObjectRef, green_key_hash: usize) {
    let _ = _space;
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .trace_next_iteration(green_key_hash as u64);
}

/// interp_jit.py:169 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn residual_call(
    _space: pyre_object::PyObjectRef,
    callable: pyre_object::PyObjectRef,
    args: &[pyre_object::PyObjectRef],
) -> pyre_object::PyObjectRef {
    let _ = _space;
    pyre_interpreter::baseobjspace::call_function(callable, args)
}

/// interp_jit.py:138-167 — set_param(space, __args__).
///
/// Configure the tunable JIT parameters.
///   * set_param(name=value, ...)            # as keyword arguments
///   * set_param("name=value,name=value")    # as a user-supplied string
///   * set_param("off")                      # disable the jit
///   * set_param("default")                  # restore all defaults
pub fn set_param(
    _space: pyre_object::PyObjectRef,
    __args__: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let _ = _space;
    let (driver, _) = driver_pair();

    // Separate positional args from kwargs dict (last arg with __pyre_kw__ marker).
    let (pos_args, kwds) = split_kwargs(__args__);

    // interp_jit.py:147-148
    if pos_args.len() > 1 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "set_param() takes at most 1 non-keyword argument, {} given",
            pos_args.len()
        )));
    }

    // interp_jit.py:151-156 — positional string → jit.set_user_param(None, text)
    if pos_args.len() == 1 {
        let w_text = pos_args[0];
        if !unsafe { pyre_object::is_str(w_text) } {
            return Ok(w_none());
        }
        let text = unsafe { pyre_object::w_str_get_value(w_text) };
        // rlib/jit.py:842-845
        if text == "off" {
            let ws = driver.meta_interp_mut().warm_state_mut();
            ws.set_param("threshold", -1);
            ws.set_param("function_threshold", -1);
        } else if text == "default" {
            driver
                .meta_interp_mut()
                .warm_state_mut()
                .set_default_params();
        } else {
            // rlib/jit.py:850-862 — "name=value,name=value"
            let ws = driver.meta_interp_mut().warm_state_mut();
            for s in text.split(',') {
                let s = s.trim();
                if s.is_empty() {
                    continue;
                }
                // rlib/jit.py:853 — len(parts) != 2 → raise ValueError
                let Some((name, value)) = s.split_once('=') else {
                    return Err(pyre_interpreter::PyError::new(
                        pyre_interpreter::PyErrorKind::ValueError,
                        "error in JIT parameters string".to_string(),
                    ));
                };
                let value = value.trim();
                if name == "enable_opts" {
                    ws.set_param_enable_opts(value);
                } else if let Ok(parsed) = value.parse::<i64>() {
                    ws.set_param(name, parsed);
                } else {
                    return Err(pyre_interpreter::PyError::new(
                        pyre_interpreter::PyErrorKind::ValueError,
                        "error in JIT parameters string".to_string(),
                    ));
                }
            }
        }
    }

    // interp_jit.py:157-167 — keyword arguments
    if let Some(kw_dict) = kwds {
        let ws = driver.meta_interp_mut().warm_state_mut();
        let d = unsafe { &*(kw_dict as *const pyre_object::dictobject::W_DictObject) };
        for &(k, v) in unsafe { &*d.entries } {
            if !unsafe { pyre_object::is_str(k) } {
                continue;
            }
            let key = unsafe { pyre_object::w_str_get_value(k) };
            if key == "__pyre_kw__" {
                continue;
            }
            // interp_jit.py:158-159
            if key == "enable_opts" {
                if unsafe { pyre_object::is_str(v) } {
                    ws.set_param_enable_opts(unsafe { pyre_object::w_str_get_value(v) });
                }
                continue;
            }
            // interp_jit.py:160-167 — validate parameter name
            if !is_known_jit_param(key) {
                return Err(pyre_interpreter::PyError::type_error(format!(
                    "no JIT parameter '{key}'"
                )));
            }
            if unsafe { pyre_object::is_int(v) } {
                ws.set_param(key, unsafe { pyre_object::w_int_get_value(v) });
            }
        }
    }

    Ok(w_none())
}

/// rlib/jit.py:588-605 PARAMETERS — valid parameter names.
fn is_known_jit_param(name: &str) -> bool {
    matches!(
        name,
        "threshold"
            | "function_threshold"
            | "trace_eagerness"
            | "decay"
            | "trace_limit"
            | "inlining"
            | "loop_longevity"
            | "retrace_limit"
            | "pureop_historylength"
            | "max_retrace_guards"
            | "max_unroll_loops"
            | "disable_unrolling"
            | "enable_opts"
            | "max_unroll_recursion"
            | "vec"
            | "vec_all"
            | "vec_cost"
    )
}

/// Split args into (positional, optional kwargs dict).
fn split_kwargs(
    args: &[pyre_object::PyObjectRef],
) -> (
    &[pyre_object::PyObjectRef],
    Option<pyre_object::PyObjectRef>,
) {
    if let Some(&last) = args.last() {
        if !last.is_null()
            && unsafe { pyre_object::is_dict(last) }
            && unsafe {
                pyre_object::w_dict_lookup(last, pyre_object::w_str_new("__pyre_kw__")).is_some()
            }
        {
            return (&args[..args.len() - 1], Some(last));
        }
    }
    (args, None)
}

/// interp_jit.py:258 — `@dont_look_inside`
///
/// Mark all current machine code objects as ready to release.
/// They will be released at the next GC (unless in use on a thread stack).
#[majit_macros::dont_look_inside]
pub fn releaseall(_space: pyre_object::PyObjectRef) {
    let _ = _space;
    let (driver, _) = driver_pair();
    // memmgr.py:85 release_all_loops parity.
    driver.mark_all_loops_for_release();
}

fn init_callbacks() {
    use pyre_jit_trace::callbacks::{self, CallJitCallbacks};
    thread_local! {
        static INIT: Cell<bool> = const { Cell::new(false) };
    }
    INIT.with(|c| {
        if !c.get() {
            c.set(true);
            let cb = Box::leak(Box::new(CallJitCallbacks {
                callee_frame_helper: crate::call_jit::callee_frame_helper,
                recursive_force_cache_safe: crate::call_jit::recursive_force_cache_safe,
                jit_drop_callee_frame: crate::call_jit::jit_drop_callee_frame as *const (),
                jit_force_callee_frame: crate::call_jit::jit_force_callee_frame as *const (),
                jit_force_recursive_call_1: crate::call_jit::jit_force_recursive_call_1
                    as *const (),
                jit_force_recursive_call_argraw_boxed_1:
                    crate::call_jit::jit_force_recursive_call_argraw_boxed_1 as *const (),
                jit_force_self_recursive_call_argraw_boxed_1:
                    crate::call_jit::jit_force_self_recursive_call_argraw_boxed_1 as *const (),
                jit_create_callee_frame_1: crate::call_jit::jit_create_callee_frame_1 as *const (),
                jit_create_callee_frame_1_raw_int:
                    crate::call_jit::jit_create_callee_frame_1_raw_int as *const (),
                jit_create_self_recursive_callee_frame_1:
                    crate::call_jit::jit_create_self_recursive_callee_frame_1 as *const (),
                jit_create_self_recursive_callee_frame_1_raw_int:
                    crate::call_jit::jit_create_self_recursive_callee_frame_1_raw_int as *const (),
                driver_pair: || JIT_DRIVER.with(|cell| cell.get() as *mut u8),
                ensure_published_jitcode: |code, w_code| {
                    if !code.is_null() {
                        let _ = crate::jit::codewriter::ensure_published_jitcode_for_w_code(
                            code, w_code,
                        );
                    }
                },
            }));
            callbacks::init(cb);
        }
    });
}

// JIT_TRACING_DEPTH removed — now MetaInterp.tracing_call_depth field.
// RPython portal_call_depth parity: state colocated with tracing context.

/// Read the call depth from pyre-interpreter's CALL_DEPTH TLS.
/// Replaces the separate JIT_CALL_DEPTH — single source of truth.
#[inline(always)]
pub(crate) fn call_depth() -> u32 {
    pyre_interpreter::call::call_depth()
}

/// RPython green_key = (pycode, next_instr).
/// Each (code, pc) pair has independent warmup counter and compiled loop.
#[inline(always)]
pub fn make_green_key(code_ptr: *const (), pc: usize) -> u64 {
    (code_ptr as u64).wrapping_mul(1000003) ^ (pc as u64)
}

// JIT_CALL_DEPTH removed — pyre-interpreter::call::CALL_DEPTH is the single
// source of truth. call_depth() reads it. No more Box<dyn Any> allocation.

/// RPython compile.py:204-207 (record_loop_or_bridge) parity:
/// Register the compiled loop's invalidation flag with all quasi-immutable
/// dependencies collected during optimization. The optimizer records
/// namespace pointers in quasi_immutable_deps when processing
/// QUASIIMMUT_FIELD ops. After compilation, this function reads them
/// from MetaInterp and registers watchers so GUARD_NOT_INVALIDATED
/// fails when the namespace mutates.
fn register_quasi_immutable_deps(green_key: u64) {
    let (driver, _) = driver_pair();
    let deps: Vec<(u64, u32)> =
        std::mem::take(&mut driver.meta_interp_mut().last_quasi_immutable_deps);
    if deps.is_empty() {
        return;
    }
    let Some(token) = driver.get_loop_token(green_key) else {
        return;
    };
    let flag = token.invalidation_flag();
    for (ns_ptr, slot) in deps {
        let ns = unsafe { &mut *(ns_ptr as *mut pyre_interpreter::DictStorage) };
        ns.register_slot_watcher(slot as usize, &flag);
    }
}

/// rpython/rlib/rstack.py:75-90 `stack_almost_full` parity — delegates
/// to [`pyre_interpreter::stack_check::stack_almost_full`], which reads
/// the shared [`PYRE_STACKTOOBIG`](pyre_interpreter::stack_check::
/// PYRE_STACKTOOBIG) budget maintained by `sys.setrecursionlimit`. Kept
/// as a thin wrapper so existing call sites in this module stay short.
#[inline]
fn stack_almost_full() -> bool {
    pyre_interpreter::stack_check::stack_almost_full()
}

/// Evaluate a Python frame with JIT compilation.
///
/// This is the main entry point for pyre-jit.
pub fn eval_with_jit(frame: &mut PyFrame) -> PyResult {
    eval_with_jit_inner(frame)
}

fn eval_with_jit_inner(frame: &mut PyFrame) -> PyResult {
    // PYRE_JIT=0 disables JIT entirely, falling back to plain interpreter.
    static PYRE_JIT_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *PYRE_JIT_DISABLED.get_or_init(|| std::env::var("PYRE_JIT").as_deref() == Ok("0")) {
        return pyre_interpreter::eval::eval_frame_plain(frame);
    }
    pyre_interpreter::call::register_eval_override(eval_with_jit);
    #[cfg(not(target_arch = "wasm32"))]
    crate::call_jit::install_jit_call_bridge();
    init_callbacks();
    #[cfg(feature = "cranelift")]
    majit_backend_cranelift::register_rebuild_state_after_failure(rebuild_state_after_failure);
    frame.fix_array_ptrs();
    // Set CURRENT_FRAME so zero-arg super() can find __class__ in the caller.
    let _frame_guard = pyre_interpreter::eval::install_current_frame(frame);

    // RPython blackhole.py parity: during bridge tracing, concrete
    // (force helper) calls must use the plain interpreter to avoid
    // corrupting the bridge trace's symbolic state via eval_loop_jit's
    // jit_merge_point_hook. RPython's blackhole interpreter has no
    // JIT hooks; pyre's equivalent is eval_frame_plain.
    {
        let (drv, _) = driver_pair();
        if drv.is_bridge_tracing() {
            return pyre_interpreter::eval::eval_frame_plain(frame);
        }
    }

    // RPython warmspot.py ll_portal_runner:
    //   maybe_compile_and_run(increment_threshold, *args)
    //   return portal_ptr(*args)
    //
    // maybe_compile_and_run = try_function_entry_jit: checks for compiled
    // code (dispatch) or threshold (start tracing). Internally guards on
    // JC_TRACING (driver.is_tracing()) to avoid re-entry during tracing.
    //
    // portal_ptr = eval_loop_jit at depth 0 (has jit_merge_point +
    // can_enter_jit back-edge), plain interpreter at depth > 0.
    if let Some(result) = try_function_entry_jit(frame) {
        if majit_metainterp::majit_log_enabled() {
            log_named_global_result(frame, "eval_with_jit_inner.try_function_entry_jit");
        }
        return result;
    }
    let result = handle_jitexception(frame);
    if majit_metainterp::majit_log_enabled() {
        log_named_global_result(frame, "eval_with_jit_inner.handle_jitexception");
    }
    result
}

fn log_named_global_result(frame: &PyFrame, label: &str) {
    unsafe {
        if frame.w_globals.is_null() {
            return;
        }
        let Some(&value) = (*frame.w_globals).get("result") else {
            return;
        };
        if value.is_null() {
            eprintln!("[jit][{label}] result=NULL");
            return;
        }
        // pyobject.rs:308 `is_int` returns true for both INT_TYPE and
        // BOOL_TYPE, but W_BoolObject's layout differs from W_IntObject —
        // calling w_int_get_value on a bool reads the wrong field. Match
        // INT_TYPE strictly so the int payload read stays sound.
        if pyre_object::pyobject::py_type_check(value, &pyre_object::pyobject::INT_TYPE) {
            eprintln!(
                "[jit][{label}] result_ptr=0x{:x} kind=int intval={}",
                value as usize,
                pyre_object::intobject::w_int_get_value(value),
            );
        } else if pyre_object::pyobject::is_bool(value) {
            eprintln!("[jit][{label}] result_ptr=0x{:x} kind=bool", value as usize,);
        } else {
            eprintln!(
                "[jit][{label}] result_ptr=0x{:x} kind=other",
                value as usize,
            );
        }
    }
}

/// warmspot.py:970-983 ContinueRunningNormally → portal_ptr(*args) parity.
///
/// warmspot.py:961-983 handle_jitexception: ContinueRunningNormally path.
///
/// Called from handle_jitexception_in_portal (via portal_runner callback)
/// when ContinueRunningNormally is raised at a recursive portal level.
/// Extracts the red_ref values (frame locals as PyObjectRef pointers)
/// and calls the portal function (eval_with_jit) with those values.
///
/// Returns Ok((return_type, value)) or Err(JitException) if the portal
/// itself raises a JitException (warmspot.py:979-980 loop back).
pub(crate) fn pyre_portal_runner(
    exc: &majit_metainterp::jitexc::JitException,
) -> Result<(majit_metainterp::blackhole::BhReturnType, i64), majit_metainterp::jitexc::JitException>
{
    use majit_metainterp::blackhole::BhReturnType;
    use majit_metainterp::jitexc::JitException;

    let JitException::ContinueRunningNormally {
        green_int,
        green_ref,
        green_float,
        red_int,
        red_ref,
        red_float,
    } = exc
    else {
        return Ok((BhReturnType::Void, 0));
    };

    // warmspot.py:972-975: portalfunc_ARGS extraction.
    // Build merged arg lists like blackhole.py:1113-1116.
    let mut all_i = green_int.clone();
    all_i.extend(red_int);
    let mut all_r = green_ref.clone();
    all_r.extend(red_ref);
    let _all_f = (green_float, red_float);

    // warmspot.py:976-978: result = portal_ptr(*args)
    let next_instr = all_i.first().copied().unwrap_or(0) as usize;
    let pycode = all_r.first().copied().unwrap_or(0) as pyre_object::PyObjectRef;
    let frame_ptr = all_r.get(1).copied().unwrap_or(0) as *mut PyFrame;
    let ec = all_r.get(2).copied().unwrap_or(0) as *const pyre_interpreter::PyExecutionContext;
    if frame_ptr.is_null() {
        return Err(JitException::ExitFrameWithExceptionRef(majit_ir::GcRef(0)));
    }
    let frame = unsafe { &mut *frame_ptr };
    if !pycode.is_null() {
        frame.pycode = pycode as *const ();
    }
    if !ec.is_null() {
        frame.execution_context = ec;
    }
    frame.set_last_instr_from_next_instr(next_instr);
    match portal_runner_result(frame) {
        Ok(result) => Ok((BhReturnType::Ref, result as i64)),
        Err(err) => Err(JitException::ExitFrameWithExceptionRef(majit_ir::GcRef(
            err.exc_object as usize,
        ))),
    }
}

/// warmspot.py:961-1007 handle_jitexception.
///
/// RPython: CRN → portal_ptr(*args) re-invokes the interpreter.
/// pyre: CRN → re-loop eval_loop_jit(frame). This does NOT call
/// maybe_compile_and_run (warmspot.py:948); portal_ptr is a plain
/// interpreter dispatch, and pyre's eval_loop_jit is the equivalent.
/// TODO: exact portal_ptr(*args) parity (currently `continue`
/// re-enters without re-extracting CRN args from the exception).
#[inline(always)]
fn handle_jitexception(frame: &mut PyFrame) -> PyResult {
    loop {
        match eval_loop_jit(frame) {
            LoopResult::Done(result) => return result,
            LoopResult::ContinueRunningNormally => {
                // RPython warmspot.py:976-978: result = portal_ptr(*args).
                // The blackhole has already written back the merge point
                // state to the frame (call_jit.rs:999-1013). Re-enter
                // eval_loop_jit with that state — do NOT reset to entry.
                frame.fix_array_ptrs();
                continue;
            }
        }
    }
}

fn debug_first_arg_int(frame: &PyFrame) -> Option<i64> {
    if frame.locals_w().len() == 0 {
        return None;
    }
    let value = frame.locals_w()[0];
    if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) } {
        return None;
    }
    Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
}

/// warmspot.py:941 ll_portal_runner parity: execute a frame through the
/// JIT-enabled portal runner. Used by bhimpl_recursive_call
/// (blackhole.py:1101-1116) for recursive portal depth.
///
/// warmspot.py:941-959:
///   maybe_compile_and_run(state.increment_function_threshold, *args)
///   return portal_ptr(*args)
///
/// warmspot.py:997-1005: ExitFrameWithExceptionRef → re-raise.
pub(crate) fn portal_runner_result(frame: &mut PyFrame) -> PyResult {
    // warmspot.py:941-955 ll_portal_runner:
    //   maybe_compile_and_run(state.increment_function_threshold, *args)
    //   return portal_ptr(*args)
    //
    // portal_ptr is the JIT-aware interpreter (jit_merge_point +
    // can_enter_jit). pyre's equivalent is handle_jitexception →
    // eval_loop_jit, NOT eval_frame_plain. Routing through
    // eval_frame_plain here would skip maybe_enter_jit at every
    // opcode of the recursive portal frame, which breaks parity for
    // bhimpl_recursive_call_* paths.
    frame.fix_array_ptrs();
    let _frame_guard = pyre_interpreter::eval::install_current_frame(frame);
    portal_runner_dispatch(frame)
}

fn portal_runner_dispatch(frame: &mut PyFrame) -> PyResult {
    if let Some(result) = try_function_entry_jit(frame) {
        result
    } else {
        handle_jitexception(frame)
    }
}

pub fn portal_runner(frame: &mut PyFrame) -> pyre_object::PyObjectRef {
    match portal_runner_result(frame) {
        Ok(r) => r,
        Err(err) => {
            #[cfg(feature = "cranelift")]
            majit_backend_cranelift::jit_exc_raise(err.exc_object as i64);
            #[cfg(feature = "dynasm")]
            majit_backend_dynasm::jit_exc_raise(err.exc_object as i64);
            #[cfg(not(any(feature = "cranelift", feature = "dynasm")))]
            let _ = err;
            pyre_object::PY_NULL
        }
    }
}

/// pyre-local debug instrumentation (no PyPy counterpart).
/// `@not_in_trace` so that compiled code does not include this call.
#[majit_macros::not_in_trace]
fn trace_jit_bytecode(_pc: usize, _instruction_name: &str) {
    // Debug logging disabled — per-bytecode eprintln causes O(n) slowdown.
}

/// JIT-enabled evaluation loop (PyPy interp_jit.py dispatch()).
///
/// Calls merge_point on EVERY iteration (PyPy line 85-87), not just
/// when tracing. This matches PyPy's jit_merge_point placement.
/// RPython interp_jit.py dispatch() parity.
///
/// The hot loop mirrors RPython's structure exactly:
///   while True:
///       jit_merge_point(...)      # thin inline check
///       next_instr = handle_bytecode(...)
///
/// warmspot.py portal_runner parity: execute a frame through the JIT-enabled
/// interpreter. Used by bhimpl_recursive_call (blackhole.py:1074-1093) for
/// recursive portal depth. Returns PyObjectRef (NULL on void/exception).
/// JIT hooks are thin inline checks; all heavy logic is in #[cold] helpers.
fn eval_loop_jit(frame: &mut PyFrame) -> LoopResult {
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame) };
    let env = PyreEnv;
    let (driver, info) = driver_pair();
    // The codewriter-side portal check
    // (`CallControl::jitdriver_sd_from_portal_graph`, codewriter.py:37)
    // is the canonical "is this code a portal" answer once
    // `setup_jitdriver` has registered it. The eval-side gate below
    // is a different question — "should this **frame** go through the
    // JIT machinery at all" — and stays a structural pyre-specific
    // check because module-level `<module>` frames lack the
    // function-scope PyFrame layout the JIT relies on.
    //
    // PRE-EXISTING-ADAPTATION: pyre routes every function-scope
    // CodeObject through `jit_merge_point_hook` and `can_enter_jit`
    // so that recursive calls into a previously-traced function reach
    // `maybe_compile_and_run` even before the function's own loop
    // runs. RPython does not need this because portals are an
    // explicit registry (`jitdrivers_sd`), not an inferred property,
    // and recursion goes through the portal_runner. Two narrowing
    // alternatives both regress benchmarks:
    //   - "is registered portal" alone (post-`setup_jitdriver`):
    //     non-loop function frames never trigger registration, so
    //     recursive entry never reaches `maybe_compile_and_run` —
    //     surfaces as a TLS-drop panic in
    //     `test_inline_residual_user_call_with_many_args_stays_correct`.
    //   - "has back-edge AND name != <module>": same problem —
    //     non-loop function frames are skipped.
    //
    // Until pyre's portal/interpreter split is ported, the eval-side
    // gate stays pyre-specific: every non-module CodeObject reaches
    // the JIT machinery; the codewriter-side decides portal-ness via
    // the registry.
    let is_portal: bool = &*code.obj_name != "<module>";
    // interp_jit.py:66 — next_instr, pycode are greens (managed by jit_merge_point).
    // No explicit promote needed; the JitDriver green-key mechanism handles this.

    loop {
        if frame.next_instr() >= code.instructions.len() {
            return LoopResult::Done(Ok(w_none()));
        }

        let pc = frame.next_instr();
        let (opcode_pc, instruction, op_arg) = match decode_instruction_for_dispatch(code, pc) {
            Ok(decoded) => decoded,
            Err(err) => return LoopResult::Done(Err(err.into())),
        };

        // ── jit_merge_point (RPython interp_jit.py:85-87) ──
        // Runtime no-op. Only handles trace feed when tracing is active.
        if is_portal {
            let tracing_depth: Option<u32> = driver.meta_interp().tracing_call_depth;
            if let Some(depth) = tracing_depth {
                if call_depth() == depth {
                    if let Some(loop_result) =
                        jit_merge_point_hook(frame, code, pc, driver, info, &env)
                    {
                        return loop_result;
                    }
                }
            } else if driver.is_tracing() {
                // First merge_point after trace start — depth not yet set.
                if let Some(loop_result) = jit_merge_point_hook(frame, code, pc, driver, info, &env)
                {
                    return loop_result;
                }
            }
        }

        // ── handle_bytecode (RPython interp_jit.py:90) ──
        trace_jit_bytecode(pc, "");
        frame.last_instr = pc as isize;
        frame.set_last_instr_from_next_instr(opcode_pc + 1);
        let mut next_instr = frame.next_instr();
        match execute_opcode_step(frame, code, instruction, op_arg, next_instr) {
            Ok(StepResult::Continue) => {
                // pyjitpl.py:2843 blackhole_if_trace_too_long — check after
                // every traced step to prevent infinite trace recording.
                driver.blackhole_if_trace_too_long();
            }
            Ok(StepResult::CloseLoop { loop_header_pc, .. }) if is_portal => {
                // ── can_enter_jit (RPython interp_jit.py:114) ──
                // RPython interp_jit.py:114 → warmstate.py:446
                let green_key = make_green_key(frame.pycode, loop_header_pc);
                if let Some(loop_result) =
                    maybe_compile_and_run(frame, green_key, loop_header_pc, driver, info, &env)
                {
                    return loop_result;
                }
            }
            Ok(StepResult::CloseLoop { .. }) => {}
            Ok(StepResult::Return(result)) => return LoopResult::Done(Ok(result)),
            Ok(StepResult::Yield(result)) => return LoopResult::Done(Ok(result)),
            Err(err) => {
                if pyre_interpreter::eval::handle_exception(frame, &err, &mut next_instr) {
                    frame.set_last_instr_from_next_instr(next_instr);
                    continue;
                }
                return LoopResult::Done(Err(err));
            }
        }
    }
}

/// pyjitpl.py:2837-2845 _interpret() parity for bridge tracing.
///
/// RPython's bridge tracing uses the same MetaInterp._interpret() loop
/// as normal tracing. This function provides the same eval loop as
/// eval_loop_jit, but always calls jit_merge_point_hook since tracing
/// is already active from start_bridge_tracing.
pub(crate) fn eval_loop_jit_bridge(frame: &mut PyFrame) -> LoopResult {
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame) };
    let env = PyreEnv;
    let (driver, info) = driver_pair();

    loop {
        if frame.next_instr() >= code.instructions.len() {
            return LoopResult::Done(Ok(w_none()));
        }

        let pc = frame.next_instr();
        let (opcode_pc, instruction, op_arg) = match decode_instruction_for_dispatch(code, pc) {
            Ok(decoded) => decoded,
            Err(err) => return LoopResult::Done(Err(err.into())),
        };

        // pyjitpl.py:1892-1914 run_one_step: trace + execute.
        if driver.is_tracing() {
            if let Some(loop_result) = jit_merge_point_hook(frame, code, pc, driver, info, &env) {
                return loop_result;
            }
        } else {
            // Tracing ended (bridge compiled or aborted).
            return LoopResult::Done(Ok(w_none()));
        }

        // handle_bytecode: execute the bytecode on the concrete frame.
        let next_instr = opcode_pc + 1;
        frame.set_last_instr_from_next_instr(next_instr);
        match execute_opcode_step(frame, code, instruction, op_arg, next_instr) {
            Ok(StepResult::Continue) => {}
            Ok(StepResult::CloseLoop { .. }) => {}
            Ok(StepResult::Return(result)) => return LoopResult::Done(Ok(result)),
            Ok(StepResult::Yield(result)) => return LoopResult::Done(Ok(result)),
            Err(err) => {
                let mut next_instr = frame.next_instr();
                if pyre_interpreter::eval::handle_exception(frame, &err, &mut next_instr) {
                    frame.set_last_instr_from_next_instr(next_instr);
                    continue;
                }
                return LoopResult::Done(Err(err));
            }
        }
    }
}

/// RPython jit_merge_point slow path — only called when tracing is active.
#[cold]
#[inline(never)]
fn jit_merge_point_hook(
    frame: &mut PyFrame,
    code: &pyre_interpreter::CodeObject,
    pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    let concrete_frame = frame as *mut PyFrame as usize;
    let green_key = make_green_key(frame.pycode, pc);

    // S2.5 (wiggly-barto plan, Stage 2 — dual verification mode).
    //
    // TODO(delete in S3.2): the env var and this whole soak block ship
    // out alongside the direct-hook deletion. Upstream has no eval-side
    // marker gate (rlib/jit.py:881-1006 + warmspot.py:874-1075 run
    // unconditionally). The gate exists only as scaffolding for the
    // S3.1 cutover so we can bring up the marker/static-data path
    // alongside the legacy `make_green_key` hash flow without flipping
    // production default in one commit.
    //
    // While `PYRE_JIT_MARKER_PARITY_SOAK=1` is set, this block runs
    // shadow-style next to the production decision: it constructs the
    // typed `GreenKey` mirroring `pypyjitdriver.greens` (rlib/jit.py:649,
    // interp_jit.py:69 — `[next_instr, is_being_profiled, pycode]`) and
    // exercises `WarmEnterState::lookup_chain_with_key` so the typed-key
    // surface stays warm for the cutover; the result is intentionally
    // discarded so the legacy hash-only flow below still owns the
    // decision (see also Pre-S3.1 Task #17 — incremental hash unification
    // is blocked by a fannkuch perf regression, so the soak gate is the
    // S3.1 prereq instead).
    static PYRE_JIT_MARKER_PARITY_SOAK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *PYRE_JIT_MARKER_PARITY_SOAK
        .get_or_init(|| std::env::var_os("PYRE_JIT_MARKER_PARITY_SOAK").is_some())
    {
        // pypyjitdriver greens declaration order
        // (interp_jit.py:69 / rlib/jit.py:649): pc first, then
        // is_being_profiled, then pycode — verify_green_args upstream
        // walks the same prefix (pyjitpl.py:1532-1535).
        //
        // The hot-path overhead would dominate variance if we allocated
        // three Vecs per tick; reuse a thread-local GreenKey with
        // pre-built `types` so each call only mutates `values[0..3]`.
        thread_local! {
            static SOAK_KEY: std::cell::RefCell<majit_ir::GreenKey> =
                std::cell::RefCell::new(majit_ir::GreenKey::with_types(
                    vec![0_i64, 0, 0],
                    vec![
                        majit_ir::Type::Int,
                        majit_ir::Type::Int,
                        majit_ir::Type::Ref,
                    ],
                ));
        }
        SOAK_KEY.with(|cell| {
            let mut key = cell.borrow_mut();
            key.values[0] = pc as i64;
            key.values[1] = if frame.get_is_being_profiled() { 1 } else { 0 };
            key.values[2] = frame.pycode as i64;
            // Shadow lookup — read-only, no install. Discarded result.
            // Pre-S3.1 the typed `get_uhash` and the legacy
            // `make_green_key` hash differ (see Task #17), so this lookup
            // intentionally misses. The soak's value is exercising the
            // typed-API call site so a regression in
            // `lookup_chain_with_key` surfaces under
            // `PYRE_JIT_MARKER_PARITY_SOAK=1` before S3.1 cutover.
            let _shadow = driver
                .meta_interp()
                .warm_state_ref()
                .lookup_chain_with_key(&key);
        });
    }

    let mut jit_state = build_jit_state(frame, info);
    let current_depth = call_depth();
    let was_tracing = driver.is_tracing();
    // warmstate.py:437-444: capture the starting cell's key before
    // entering the trace body so we can unconditionally clear its
    // TRACING flag in the post-trace finally block. May differ from
    // `green_key` when we are mid-trace and the current merge point's
    // key is not the tracing origin.
    let starting_tracing_key = driver.starting_green_key();
    if let Some(outcome) = driver.jit_merge_point_keyed(
        green_key,
        pc,
        &mut jit_state,
        env,
        || {},
        |ctx, sym| {
            let (driver, _) = driver_pair();
            driver.meta_interp_mut().tracing_call_depth = Some(current_depth);
            // RPython parity: codewriter.make_jitcodes() runs before tracing
            // starts, populating all_liveness. In pyre, JitCode compilation is
            // lazy — ensure the code's JitCode (with liveness) exists before
            // tracing so get_list_of_active_boxes can use it.
            crate::jit::codewriter::register_portal_jitdriver(code, frame.pycode, Some(pc));
            let snapshot = frame.snapshot_for_tracing();
            let _ = concrete_frame;
            let (action, _executed_frame) = trace_bytecode(ctx, sym, code, pc, snapshot);
            action
        },
    ) {
        match handle_jit_outcome(outcome, &jit_state, frame, info, green_key) {
            JitAction::Return(result) => return Some(LoopResult::Done(result)),
            JitAction::ContinueRunningNormally => return Some(LoopResult::ContinueRunningNormally),
            JitAction::Continue => {}
        }
    }
    // Trace completed or aborted — clear tracing depth.
    if !driver.is_tracing() {
        driver.meta_interp_mut().tracing_call_depth = None;
        // compile.py:269: cross-loop cut stores under inner key.
        // Use the actual compiled key for post-compilation steps.
        let compiled_key = driver.last_compiled_key().unwrap_or(green_key);
        // warmstate.py:444 `finally: cell.flags &= ~JC_TRACING` parity.
        // `starting_tracing_key` was captured before jit_merge_point_keyed;
        // its TRACING must be cleared unconditionally — even if cross-loop
        // cut compiled under a different key, or if the trace aborted.
        if let Some(k) = starting_tracing_key {
            driver
                .meta_interp_mut()
                .warm_state_mut()
                .clear_tracing_flag(k);
        }
        register_quasi_immutable_deps(compiled_key);
        // RPython pyjitpl.py:3048-3061 raise_continue_running_normally:
        // after trace compilation, restart so maybe_compile_and_run
        // (try_function_entry_jit) dispatches to compiled code.
        if was_tracing {
            return Some(LoopResult::ContinueRunningNormally);
        }
    }
    None
}

/// RPython warmstate.py:446-511 maybe_compile_and_run.
///
/// Entry point to the JIT. Called at can_enter_jit (back-edge).
///
/// RPython order: cell lookup (JC_TRACING → skip, JC_COMPILED → enter)
/// BEFORE counter.tick(). This prevents compiled loops from occupying
/// counter hash-table slots and evicting non-compiled loops (the 5-way
/// associative cache has only 5 slots per bucket).
#[cold]
#[inline(never)]
fn maybe_compile_and_run(
    frame: &mut PyFrame,
    green_key: u64,
    loop_header_pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    // pyre-local extension: PYRE_NO_JIT disables JIT entirely.
    // No RPython counterpart — kept for development debugging only.
    // TODO: remove when JIT is stable enough to not need a kill switch.
    static NO_JIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *NO_JIT.get_or_init(|| std::env::var_os("PYRE_NO_JIT").is_some()) {
        return None;
    }
    // warmstate.py:473-477: JC_TRACING → skip entirely (no counter tick)
    if driver.is_tracing() {
        return None;
    }
    // warmstate.py:503-511: procedure_token exists → EnterJitAssembler.
    // RPython enters assembler unconditionally when a compiled loop is
    // available for this green_key.
    if driver.has_compiled_loop(green_key) {
        return execute_assembler(frame, green_key, loop_header_pc, driver, info, env);
    }
    // warmstate.py:484: DONT_TRACE_HERE → skip counter tick entirely
    if driver
        .meta_interp()
        .warm_state_ref()
        .is_dont_trace_here(green_key)
    {
        return None;
    }
    // warmstate.py:496-511: counter.tick → threshold reached → bound_reached
    if driver
        .meta_interp_mut()
        .warm_state_mut()
        .counter
        .tick(green_key)
    {
        if driver
            .meta_interp()
            .is_tracing_key((frame.pycode as usize, loop_header_pc))
        {
            return None;
        }
        return bound_reached(frame, green_key, loop_header_pc, driver, info, env);
    }
    None
}

/// compile.py:701-717 handle_fail outcome.
/// compile.py:701-717: handle_fail NEVER returns in RPython — it raises
/// ContinueRunningNormally or DoneWithThisFrame. In pyre, we return the
/// equivalent BlackholeResult.
/// compile.py:701-717 handle_fail outcome.
enum HandleFailOutcome {
    /// Bridge compiled successfully — continue in compiled code.
    BridgeCompiled,
    /// Resume in blackhole interpreter.
    ResumeInBlackhole,
}

/// compile.py:701-717 handle_fail.
///
/// Single function containing the complete guard failure handling:
/// compile.py:701-717 handle_fail.
///
/// RPython: handle_fail NEVER returns — both paths raise
/// ContinueRunningNormally or DoneWithThisFrame.
/// pyre: returns BlackholeResult (equivalent to RPython's exceptions).
fn handle_fail(
    frame: &mut PyFrame,
    _green_key: u64,
    trace_id: u64,
    fail_index: u32,
    should_bridge: bool,
    owning_key: u64,
    descr_addr: usize,
    exit_layout: &CompiledExitLayout,
    raw_values: &[i64],
    _info: &majit_metainterp::virtualizable::VirtualizableInfo,
) -> HandleFailOutcome {
    // compile.py:702-703: must_compile() AND not stack_almost_full()
    if should_bridge && !stack_almost_full() {
        let is_tracing = {
            let (driver, _) = driver_pair();
            driver.is_tracing()
        };
        if !is_tracing {
            // compile.py:704: self.start_compiling() (set ST_BUSY_FLAG)
            {
                let (driver, _) = driver_pair();
                driver.meta_interp_mut().start_guard_compiling(descr_addr);
            }
            // compile.py:706-708: _trace_and_compile_from_bridge(deadframe)
            // force_plain_eval prevents concrete calls during bridge
            // tracing from re-entering compiled code.
            let compiled = {
                let _plain = pyre_interpreter::call::force_plain_eval();
                crate::call_jit::trace_and_compile_from_bridge(
                    owning_key,
                    trace_id,
                    fail_index,
                    frame,
                    raw_values,
                    exit_layout,
                )
            };
            // compile.py:709: done_compiling (clear ST_BUSY_FLAG)
            {
                let (driver, _) = driver_pair();
                driver.meta_interp_mut().done_guard_compiling(descr_addr);
            }
            if compiled {
                // compile.py:708: bridge compiled → ContinueRunningNormally.
                // RPython: the bridge is attached to the guard descr;
                // re-entering compiled code will follow the bridge.
                return HandleFailOutcome::BridgeCompiled;
            }
        }
    }
    // compile.py:710-716 / pyjitpl.py:2906 (SwitchToBlackhole):
    // resume_in_blackhole(metainterp_sd, jitdriver_sd, self, deadframe)
    HandleFailOutcome::ResumeInBlackhole
}

/// compile.py:710-716 resume_in_blackhole parity.
///
/// RPython: resume_in_blackhole → blackhole_from_resumedata →
/// consume_one_section → _run_forever → raises.
///
fn resume_in_blackhole_from_exit_layout(
    frame: &mut PyFrame,
    raw_values: &[i64],
    exit_layout: &CompiledExitLayout,
) -> crate::call_jit::BlackholeResult {
    use crate::call_jit::BlackholeResult;

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[dynasm-debug] resume_in_blackhole: raw_values.len={} exit_types.len={} rd_numb={:?}",
            raw_values.len(),
            exit_layout.exit_types.len(),
            exit_layout.storage.as_deref().map(|s| s.rd_numb.len())
        );
    }
    // Save frame state before consume_vable_info modifies it.
    // RPython's blackhole always completes; pyre currently keeps a
    // partial rollback path for the `BlackholeResult::Failed` escape
    // (defensive null/bounds checks inside `resume_in_blackhole`
    // fall back here). The full removal of `Failed` depends on
    // auditing those remaining sites separately — see
    // memory/rd_numb_audit_2026_04_19.md step 5.
    let saved_ni = frame.next_instr();
    let saved_code = frame.pycode;
    let saved_vsd = frame.valuestackdepth;
    let saved_ns = frame.w_globals;
    let saved_array: Vec<pyre_object::PyObjectRef> = frame.locals_w().as_slice().to_vec();

    build_blackhole_frames_from_deadframe(raw_values, exit_layout);
    // `build_blackhole_frames_from_deadframe` asserts that rd_numb is
    // present and non-empty (commit c7ea7cb58b + the fallback removal
    // above), so `LAST_GUARD_FRAMES` is always populated by the time we
    // read it back here. `take_last_guard_frames` must return Some.
    let frames = take_last_guard_frames()
        .expect("LAST_GUARD_FRAMES must be set by build_blackhole_frames_from_deadframe");
    let result = crate::call_jit::resume_in_blackhole(frame, &frames);
    if matches!(result, BlackholeResult::Failed) {
        // Restore frame to pre-consume state so the fallback
        // interpreter can continue from the original frame.
        frame.set_last_instr_from_next_instr(saved_ni);
        frame.pycode = saved_code;
        frame.valuestackdepth = saved_vsd;
        frame.w_globals = saved_ns;
        let dest = frame.locals_w_mut().as_mut_slice();
        let n = dest.len().min(saved_array.len());
        dest[..n].copy_from_slice(&saved_array[..n]);
    }
    result
}

/// RPython warmstate.py:387-423 execute_assembler.
///
/// Run compiled machine code for a given green_key. Handles the
/// fail_descr outcomes: DoneWithThisFrame, GuardFailure, etc.
#[cold]
#[inline(never)]
fn execute_assembler(
    frame: &mut PyFrame,
    green_key: u64,
    entry_pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    frame.set_last_instr_from_next_instr(entry_pc);

    if majit_metainterp::majit_log_enabled() {
        let locals: Vec<(usize, Option<i64>)> = (0..frame.locals_w().len().min(5))
            .map(|i| {
                let value = frame.locals_w()[i];
                let decoded = if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) }
                {
                    None
                } else {
                    Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
                };
                (value as usize, decoded)
            })
            .collect();
        eprintln!("[jit][execute-assembler][locals] {:?}", locals);
    }

    let mut jit_state = build_jit_state(frame, info);

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][execute-assembler] key={} pc={} arg0={:?}",
            green_key,
            entry_pc,
            debug_first_arg_int(frame),
        );
    }

    // warmstate.py:395 func_execute_token(loop_token, *args) → deadframe
    let outcome = driver.run_compiled_detailed_with_bridge_keyed(
        green_key,
        entry_pc,
        &mut jit_state,
        env,
        || {},
    );

    // rstack.stack_check_slowpath → _StackOverflow parity: drain the
    // JIT-overflow flag the backend probe records when it trips. The
    // backend detects the overflow inside compiled code and exits via
    // the dedicated stack-overflow block; we surface the user-visible
    // RecursionError here on the way back to the interpreter loop.
    if let Err(exc) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
        return Some(LoopResult::Done(Err(exc)));
    }

    // warmspot.py:998 ExitFrameWithExceptionRef: check for exceptions
    // stashed by blackhole/force callbacks across FFI boundaries.
    if let Some(exc) = crate::call_jit::take_ca_exception() {
        return Some(LoopResult::Done(Err(exc)));
    }

    if majit_metainterp::majit_log_enabled() {
        let kind = match &outcome {
            DetailedDriverRunOutcome::Finished { .. } => "finished",
            DetailedDriverRunOutcome::Jump { .. } => "jump",
            DetailedDriverRunOutcome::Abort { .. } => "abort",
            DetailedDriverRunOutcome::GuardFailure { .. } => "guard-failure",
        };
        eprintln!(
            "[jit][execute-assembler] outcome key={} pc={} kind={}",
            green_key, entry_pc, kind
        );
    }

    // warmstate.py:402-422 handle fail_descr outcome
    match outcome {
        // warmstate.py:402-415 fast path: DoneWithThisFrame
        DetailedDriverRunOutcome::Finished {
            typed_values,
            raw_int_result,
            is_exit_frame_with_exception,
            ..
        } => {
            let raw_int_result = raw_int_result || driver.has_raw_int_finish();
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][handle-outcome] finished key={} raw_flag={} exc_exit={} typed_values={:?}",
                    green_key, raw_int_result, is_exit_frame_with_exception, typed_values
                );
            }
            // compile.py:658-662 ExitFrameWithExceptionDescrRef parity.
            // warmspot.py:998 handle_jitexception:
            //   ExitFrameWithExceptionRef.handle_fail raises the stored Ref
            //   into the outer interpreter's exception machinery.
            if is_exit_frame_with_exception {
                let exc_ref = match typed_values.as_slice() {
                    [majit_ir::Value::Ref(r)] => r.as_usize() as pyre_object::PyObjectRef,
                    _ => {
                        return Some(LoopResult::Done(Err(
                            pyre_interpreter::PyError::type_error(
                                "compiled exit_frame_with_exception did not produce a single Ref value",
                            ),
                        )));
                    }
                };
                let err = unsafe { pyre_interpreter::PyError::from_exc_object(exc_ref) };
                return Some(LoopResult::Done(Err(err)));
            }
            let [value] = typed_values.as_slice() else {
                return Some(LoopResult::Done(Err(
                    pyre_interpreter::PyError::type_error(
                        "compiled finish did not produce a single object return value",
                    ),
                )));
            };
            let result = match value {
                majit_ir::Value::Int(raw) => {
                    // compile.py:631 DoneWithThisFrameDescrInt parity —
                    // unused in pyre (result_type=Ref), but handle
                    // gracefully just in case.
                    pyre_object::intobject::w_int_new(*raw)
                }
                majit_ir::Value::Ref(value) => {
                    // compile.py:640 DoneWithThisFrameDescrRef parity:
                    // return get_result() as-is. jitframe GC trace hook
                    // (jitframe.rs:293) keeps interior refs alive.
                    value.as_usize() as pyre_object::PyObjectRef
                }
                majit_ir::Value::Float(f) => pyre_object::floatobject::w_float_new(*f),
                majit_ir::Value::Void => {
                    return Some(LoopResult::Done(Err(
                        pyre_interpreter::PyError::type_error(
                            "compiled finish produced a void return value",
                        ),
                    )));
                }
            };
            Some(LoopResult::Done(Ok(result)))
        }
        // warmstate.py:416-422 general: handle_fail
        // compile.py:701-717 → bridge or blackhole
        DetailedDriverRunOutcome::GuardFailure {
            fail_index,
            trace_id,
            should_bridge,
            owning_key,
            descr_addr,
            ref raw_values,
            ref exit_layout,
        } => {
            match handle_fail(
                frame,
                green_key,
                trace_id,
                fail_index,
                should_bridge,
                owning_key,
                descr_addr,
                exit_layout,
                raw_values,
                info,
            ) {
                HandleFailOutcome::BridgeCompiled => Some(LoopResult::ContinueRunningNormally),
                HandleFailOutcome::ResumeInBlackhole => {
                    // compile.py:710-716 / pyjitpl.py:2906 SwitchToBlackhole
                    let bh_result =
                        resume_in_blackhole_from_exit_layout(frame, raw_values, exit_layout);
                    match &bh_result {
                        crate::call_jit::BlackholeResult::ContinueRunningNormally {
                            green_int,
                            ..
                        } => {
                            // warmspot.py:961 handle_jitexception parity:
                            // CRN carries merge-point args. Write next_instr
                            // back to the frame so eval_loop_jit restarts at
                            // the merge point, not the guard-failure PC.
                            if let Some(&ni) = green_int.first() {
                                frame.set_last_instr_from_next_instr(ni as usize);
                            }
                            Some(LoopResult::ContinueRunningNormally)
                        }
                        crate::call_jit::BlackholeResult::DoneWithThisFrameRef(v) => {
                            Some(LoopResult::Done(Ok(*v)))
                        }
                        crate::call_jit::BlackholeResult::DoneWithThisFrameInt(v) => {
                            // warmspot.py:988-990: box Int to Ref for portal result_type=Ref
                            Some(LoopResult::Done(Ok(
                                pyre_object::intobject::w_int_new(*v) as pyre_object::PyObjectRef
                            )))
                        }
                        crate::call_jit::BlackholeResult::ExitFrameWithExceptionRef(exc) => {
                            // warmspot.py:998-1005 ExitFrameWithExceptionRef:
                            // propagate the Python exception, don't swallow it.
                            Some(LoopResult::Done(Err(exc.clone())))
                        }
                        crate::call_jit::BlackholeResult::Failed => {
                            // RPython: blackhole resume never fails — rd_numb
                            // is always complete (`blackhole.py:1679` raises
                            // `ExitFrameWithExceptionRef` for uncaught
                            // exceptions, never returns a failure code).
                            // Pyre's `BlackholeResult::Failed` is a layered
                            // adaptation that should be removed once the
                            // upstream SSA-authoritative live_r work lands
                            // (Task #110 / Option α step 6).  Until then
                            // the bare `invalidate_loop` keeps the cell
                            // retraceable; the failure surfaces in
                            // check.sh rather than being masked.
                            if majit_metainterp::majit_log_enabled() {
                                eprintln!(
                                    "[jit][BUG] blackhole failed key={} trace={} guard={} — invalidating",
                                    green_key, trace_id, fail_index,
                                );
                            }
                            driver.invalidate_loop(green_key);
                            None
                        }
                        _ => bh_result.to_pyresult().map(LoopResult::Done),
                    }
                }
            }
        }
        DetailedDriverRunOutcome::Jump { .. } | DetailedDriverRunOutcome::Abort { .. } => None,
    }
}

/// RPython warmstate.py:425-444 bound_reached.
///
/// Called when counter threshold fires and no compiled code exists.
/// Starts tracing via back_edge_or_run_compiled_keyed.
#[cold]
#[inline(never)]
fn bound_reached(
    frame: &mut PyFrame,
    green_key: u64,
    loop_header_pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    if majit_metainterp::majit_log_enabled() {
        let locals: Vec<(usize, Option<i64>)> = (0..frame.locals_w().len().min(5))
            .map(|i| {
                let value = frame.locals_w()[i];
                let decoded = if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) }
                {
                    None
                } else {
                    Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
                };
                (value as usize, decoded)
            })
            .collect();
        eprintln!(
            "[jit][bound-reached] key={} pc={} arg0={:?} locals={:?}",
            green_key,
            loop_header_pc,
            debug_first_arg_int(frame),
            locals,
        );
    }
    // warmstate.py:429: jitcounter.decay_all_counters()
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .counter
        .decay_all_counters();
    // warmstate.py:430
    if stack_almost_full() {
        return None;
    }
    // warmstate.py:437-444: MetaInterp.compile_and_run_once
    frame.set_last_instr_from_next_instr(loop_header_pc);
    let mut jit_state = build_jit_state(frame, info);
    // warmstate.py:473-477 JC_TRACING
    if driver
        .meta_interp()
        .is_tracing_key((frame.pycode as usize, loop_header_pc))
    {
        return None;
    }
    // warmstate.py:503-511: procedure_token → EnterJitAssembler.
    let outcome = if driver.has_compiled_loop(green_key) {
        Some(driver.run_compiled_detailed_with_bridge_keyed(
            green_key,
            loop_header_pc,
            &mut jit_state,
            env,
            || {},
        ))
    } else if !driver.is_tracing() {
        // warmstate.py:437-444 compile_and_run_once parity:
        // start tracing AND trace synchronously in a single call.
        let had_compiled = driver.has_compiled_loop(green_key);
        driver.bound_reached(green_key, loop_header_pc, &mut jit_state, env);
        // force_start_tracing may return RunCompiled (retargeted trace
        // already compiled for this cell). In that case, enter compiled.
        // compile.py:269: actual key may be inner key after cross-loop cut.
        let actual_key = driver.last_compiled_key().unwrap_or(green_key);
        if !driver.is_tracing() && driver.has_compiled_loop(actual_key) {
            Some(driver.run_compiled_detailed_with_bridge_keyed(
                actual_key,
                loop_header_pc,
                &mut jit_state,
                env,
                || {},
            ))
        } else if driver.is_tracing() {
            // RPython pyjitpl.py:2876-2888 _compile_and_run_once:
            // interpret() traces the entire loop synchronously.
            // Set tracing_call_depth so inner function calls (which
            // run their own eval_loop_jit) don't trigger jit_merge_point_hook.
            driver.meta_interp_mut().tracing_call_depth = Some(call_depth());
            let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame) };
            let outcome = driver.jit_merge_point_keyed(
                green_key,
                loop_header_pc,
                &mut jit_state,
                env,
                || {},
                |ctx, sym| {
                    use pyre_jit_trace::trace::trace_bytecode;
                    crate::jit::codewriter::register_portal_jitdriver(
                        code,
                        frame.pycode,
                        Some(loop_header_pc),
                    );
                    let concrete_frame = frame.snapshot_for_tracing();
                    let (action, _) =
                        trace_bytecode(ctx, sym, code, loop_header_pc, concrete_frame);
                    action
                },
            );
            driver.meta_interp_mut().tracing_call_depth = None;
            let compiled_key = driver.last_compiled_key().unwrap_or(green_key);
            if !had_compiled && driver.has_compiled_loop(compiled_key) {
                register_quasi_immutable_deps(compiled_key);
            }
            // pyjitpl.py:3048-3061 raise_continue_running_normally:
            // after compilation, restart so execute_assembler runs.
            if !driver.is_tracing() {
                // warmstate.py:444 `finally: cell.flags &= ~JC_TRACING`
                // — green_key is the starting cell. Cross-loop cut
                // (compile.py:269) installs the token on an inner cell,
                // so attach_procedure_to_interp does not clear TRACING
                // on green_key. Restore the clear here. The full
                // gate `!had_compiled && has_compiled_loop(compiled_key)
                // && compiled_key != green_key` narrows to "this round
                // cross-loop-compiled under a different inner key";
                // without it stale `last_compiled_key` values from
                // prior iterations trigger spurious clears that can
                // destabilize active traces (cranelift fannkuch regresses
                // without this gate).
                if !had_compiled
                    && driver.has_compiled_loop(compiled_key)
                    && compiled_key != green_key
                {
                    driver
                        .meta_interp_mut()
                        .warm_state_mut()
                        .clear_tracing_flag(green_key);
                }
                return Some(LoopResult::ContinueRunningNormally);
            }
            outcome
        } else {
            None
        }
    } else {
        None
    };
    if let Some(outcome) = outcome {
        // rstack.stack_check_slowpath → _StackOverflow parity: drain
        // the JIT-overflow flag the backend probe records when it
        // trips. The backend's prologue exits via the dedicated
        // stack-overflow block; we surface RecursionError here on the
        // way back to the interpreter loop.
        if let Err(exc) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
            return Some(LoopResult::Done(Err(exc)));
        }
        // compile.py:701-717 handle_fail: bridge/blackhole decision.
        if let DetailedDriverRunOutcome::GuardFailure {
            fail_index,
            trace_id,
            should_bridge,
            owning_key,
            descr_addr,
            ref raw_values,
            ref exit_layout,
        } = outcome
        {
            match handle_fail(
                frame,
                green_key,
                trace_id,
                fail_index,
                should_bridge,
                owning_key,
                descr_addr,
                exit_layout,
                raw_values,
                info,
            ) {
                HandleFailOutcome::BridgeCompiled => {
                    return Some(LoopResult::ContinueRunningNormally);
                }
                HandleFailOutcome::ResumeInBlackhole => {
                    let bh_result =
                        resume_in_blackhole_from_exit_layout(frame, raw_values, exit_layout);
                    match &bh_result {
                        crate::call_jit::BlackholeResult::ContinueRunningNormally {
                            green_int,
                            ..
                        } => {
                            // warmspot.py:961 parity: write merge-point PC
                            if let Some(&ni) = green_int.first() {
                                frame.set_last_instr_from_next_instr(ni as usize);
                            }
                            return Some(LoopResult::ContinueRunningNormally);
                        }
                        crate::call_jit::BlackholeResult::Failed => {}
                        _ => {
                            if let Some(r) = bh_result.to_pyresult() {
                                return Some(LoopResult::Done(r));
                            }
                        }
                    }
                }
            }
        } else {
            match handle_jit_outcome(outcome, &jit_state, frame, info, green_key) {
                JitAction::Return(result) => return Some(LoopResult::Done(result)),
                JitAction::ContinueRunningNormally | JitAction::Continue => {}
            }
        }
    }
    driver.meta_interp_mut().tracing_call_depth = None;
    None
}

/// RPython warmstate.py maybe_compile_and_run parity.
///
/// Called at every portal entry (function call). Must be fast for the
/// common case (no compiled code, not tracing, threshold not reached).
pub fn try_function_entry_jit(frame: &mut PyFrame) -> Option<PyResult> {
    // warmstate.py parity: PYRE_NO_JIT disables ALL JIT paths.
    static NO_JIT_FN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *NO_JIT_FN.get_or_init(|| std::env::var_os("PYRE_NO_JIT").is_some()) {
        return None;
    }
    if std::env::var_os("MAJIT_DUMP_BYTECODE").is_some() {
        let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame) };
        if code.obj_name.as_str() == "fannkuch" && frame.next_instr() == 0 {
            use std::sync::OnceLock;
            static DUMPED: OnceLock<()> = OnceLock::new();
            if DUMPED.get().is_none() {
                let _ = DUMPED.set(());
                let mut state = pyre_interpreter::OpArgState::default();
                eprintln!("-- fannkuch bytecode dump --");
                for (pc, unit) in code.instructions.iter().copied().enumerate() {
                    let (instr, oparg) = state.get(unit);
                    eprintln!("{pc:03}: {instr:?} oparg={oparg:?}");
                }
                for pc in [
                    72usize, 99, 129, 131, 141, 155, 168, 179, 234, 245, 447, 449,
                ] {
                    eprintln!(
                        "decode[{pc}] = {:?}",
                        pyre_interpreter::decode_instruction_at(code, pc)
                    );
                }
            }
        }
    }
    let green_key = make_green_key(frame.pycode, frame.next_instr());
    let (driver, info) = driver_pair();

    // RPython warmstate.py maybe_compile_and_run fast path:
    // if no compiled loop and not tracing, just tick the counter.
    if !driver.has_compiled_loop(green_key) && !driver.is_tracing() {
        let should_trace = driver
            .meta_interp_mut()
            .warm_state_mut()
            .should_trace_function_entry(green_key);
        if !should_trace {
            return None;
        }
    }

    // RPython warmstate.py:473-477: per-cell JC_TRACING.
    if driver
        .meta_interp()
        .is_tracing_key((frame.pycode as usize, frame.next_instr()))
    {
        return None;
    }
    if driver.has_compiled_loop(green_key) {
        // Same gate as maybe_compile_and_run: only enter compiled code
        // when a compiled loop exists for this green_key.
        // warmstate.py:503-511: procedure_token → enter unconditionally.
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][func-entry] run compiled frame=0x{:x} locals=0x{:x} key={} arg0={:?} depth={} raw_finish_known={}",
                frame as *mut PyFrame as usize,
                frame.locals_cells_stack_w as usize,
                green_key,
                debug_first_arg_int(frame),
                call_depth(),
                driver.has_raw_int_finish()
            );
        }
        let env = PyreEnv;
        let mut jit_state = build_jit_state(frame, info);
        let outcome = driver.run_compiled_detailed_with_bridge_keyed(
            green_key,
            frame.next_instr(),
            &mut jit_state,
            &env,
            || {},
        );
        // rstack.stack_check_slowpath → _StackOverflow parity: drain
        // the JIT-overflow flag the backend probe records when it
        // trips during compiled execution at function entry.
        if let Err(exc) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
            return Some(Err(exc));
        }
        // warmspot.py:998 ExitFrameWithExceptionRef: check for exceptions
        // stashed by blackhole/force callbacks across FFI boundaries.
        if let Some(exc) = crate::call_jit::take_ca_exception() {
            return Some(Err(exc));
        }
        if majit_metainterp::majit_log_enabled() {
            let kind = match &outcome {
                DetailedDriverRunOutcome::Finished { .. } => "finished",
                DetailedDriverRunOutcome::Jump { .. } => "jump",
                DetailedDriverRunOutcome::Abort { .. } => "abort",
                DetailedDriverRunOutcome::GuardFailure { .. } => "guard-failure",
            };
            eprintln!(
                "[jit][func-entry] compiled outcome frame=0x{:x} locals=0x{:x} key={} arg0={:?} kind={}",
                frame as *mut PyFrame as usize,
                frame.locals_cells_stack_w as usize,
                green_key,
                debug_first_arg_int(frame),
                kind
            );
        }

        // compile.py:701-717 handle_fail parity.
        if let DetailedDriverRunOutcome::GuardFailure {
            fail_index,
            trace_id,
            should_bridge,
            owning_key,
            descr_addr,
            ref raw_values,
            ref exit_layout,
        } = outcome
        {
            match handle_fail(
                frame,
                green_key,
                trace_id,
                fail_index,
                should_bridge,
                owning_key,
                descr_addr,
                exit_layout,
                raw_values,
                info,
            ) {
                HandleFailOutcome::BridgeCompiled => {
                    // Bridge compiled → ContinueRunningNormally → re-enter
                    // compiled code which will follow the new bridge.
                    // Fall through to eval_loop_jit below.
                }
                HandleFailOutcome::ResumeInBlackhole => {
                    let bh_result =
                        resume_in_blackhole_from_exit_layout(frame, raw_values, exit_layout);
                    match &bh_result {
                        crate::call_jit::BlackholeResult::ContinueRunningNormally { .. } => {
                            // Fall through to eval_loop_jit
                        }
                        crate::call_jit::BlackholeResult::Failed => {
                            // RPython blackhole resume cannot fail
                            // (`blackhole.py:1679` raises
                            // `ExitFrameWithExceptionRef` instead).  The
                            // `BlackholeResult::Failed` variant is a pyre
                            // layering on top of an upstream SSA liveness
                            // gap (Task #110 / Option α step 6); when it
                            // hits, surface the failure to check.sh by
                            // invalidating the loop without forcing a
                            // permanent dont-trace.
                            if majit_metainterp::majit_log_enabled() {
                                eprintln!(
                                    "[jit][BUG] blackhole failed key={} — invalidating",
                                    green_key,
                                );
                            }
                            let (driver, _) = driver_pair();
                            driver.invalidate_loop(green_key);
                        }
                        _ => {
                            if let Some(r) = bh_result.to_pyresult() {
                                if majit_metainterp::majit_log_enabled() {
                                    let returned_intval = match &r {
                                        Ok(obj)
                                            if !obj.is_null()
                                                && unsafe {
                                                    pyre_object::pyobject::is_int(*obj)
                                                } =>
                                        {
                                            Some(unsafe {
                                                pyre_object::intobject::w_int_get_value(*obj)
                                            })
                                        }
                                        _ => None,
                                    };
                                    eprintln!(
                                        "[jit][handle-outcome] bh-return arg0={:?} intval={:?}",
                                        debug_first_arg_int(frame),
                                        returned_intval,
                                    );
                                }
                                return Some(r);
                            }
                        }
                    }
                }
            }
        } else {
            match handle_jit_outcome(outcome, &jit_state, frame, info, green_key) {
                JitAction::Return(result) => return Some(result),
                JitAction::ContinueRunningNormally | JitAction::Continue => {}
            }
        }

        // After compiled code guard-restored fallback, re-establish the
        // frame's array pointer.
        frame.fix_array_ptrs();
        return None;
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][func-entry] probe key={} arg0={:?} tracing={}",
            green_key,
            debug_first_arg_int(frame),
            driver.is_tracing(),
        );
    }

    if driver.is_tracing() {
        return None;
    }

    // warmstate.py:467 jitcounter.tick(hash, increment_threshold). The
    // fast path above already fired the counter for this entry, so go
    // straight to bound_reached without re-ticking.
    if majit_metainterp::majit_log_enabled() {
        let function_threshold = driver.meta_interp().warm_state_ref().function_threshold();
        eprintln!(
            "[jit][func-entry] fired key={} arg0={:?} threshold={}",
            green_key,
            debug_first_arg_int(frame),
            function_threshold,
        );
    }
    // warmstate.py:425-444 bound_reached parity:
    //   if not confirm_enter_jit(*args): return
    //   jitcounter.decay_all_counters()
    //   if rstack.stack_almost_full(): return
    //   metainterp.compile_and_run_once(jitdriver_sd, *args)
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .counter
        .decay_all_counters();
    if stack_almost_full() {
        return None;
    }
    let env = PyreEnv;
    let mut jit_state = build_jit_state(frame, info);
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][func-entry] start tracing key={} arg0={:?}",
            green_key,
            debug_first_arg_int(frame),
        );
    }
    driver.force_start_tracing(green_key, frame.next_instr(), &mut jit_state, &env);
    None
}

fn handle_jit_outcome(
    outcome: DetailedDriverRunOutcome,
    _jit_state: &PyreJitState,
    frame: &mut PyFrame,
    _info: &majit_metainterp::virtualizable::VirtualizableInfo,
    green_key: u64,
) -> JitAction {
    match outcome {
        DetailedDriverRunOutcome::Finished {
            typed_values,
            raw_int_result,
            is_exit_frame_with_exception,
            ..
        } => {
            let (driver, _) = driver_pair();
            let raw_int_result = raw_int_result || driver.has_raw_int_finish();
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][handle-outcome] finished key={} raw_flag={} exc_exit={} typed_values={:?}",
                    green_key, raw_int_result, is_exit_frame_with_exception, typed_values
                );
            }
            // compile.py:658-662 ExitFrameWithExceptionDescrRef parity.
            if is_exit_frame_with_exception {
                let exc_ref = match typed_values.as_slice() {
                    [majit_ir::Value::Ref(r)] => r.as_usize() as pyre_object::PyObjectRef,
                    _ => {
                        return JitAction::Return(Err(pyre_interpreter::PyError::type_error(
                            "compiled exit_frame_with_exception did not produce a single Ref value",
                        )));
                    }
                };
                let err = unsafe { pyre_interpreter::PyError::from_exc_object(exc_ref) };
                return JitAction::Return(Err(err));
            }
            let [value] = typed_values.as_slice() else {
                return JitAction::Return(Err(pyre_interpreter::PyError::type_error(
                    "compiled finish did not produce a single object return value",
                )));
            };
            let value = match value {
                majit_ir::Value::Int(raw) => {
                    // compile.py:631 DoneWithThisFrameDescrInt parity —
                    // unused in pyre (result_type=Ref), but handle
                    // gracefully just in case.
                    pyre_object::intobject::w_int_new(*raw)
                }
                majit_ir::Value::Ref(value) => {
                    // compile.py:640 DoneWithThisFrameDescrRef parity.
                    value.as_usize() as pyre_object::PyObjectRef
                }
                majit_ir::Value::Float(f) => pyre_object::floatobject::w_float_new(*f),
                majit_ir::Value::Void => {
                    return JitAction::Return(Err(pyre_interpreter::PyError::type_error(
                        "compiled finish produced a void return value",
                    )));
                }
            };
            if majit_metainterp::majit_log_enabled() {
                let returned_intval =
                    if !value.is_null() && unsafe { pyre_object::pyobject::is_int(value) } {
                        Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
                    } else {
                        None
                    };
                eprintln!(
                    "[jit][handle-outcome] return arg0={:?} intval={:?} ref=0x{:x}",
                    debug_first_arg_int(frame),
                    returned_intval,
                    value as usize
                );
            }
            JitAction::Return(Ok(value))
        }
        DetailedDriverRunOutcome::Jump { .. } => {
            let _ = frame;
            JitAction::Continue
        }
        DetailedDriverRunOutcome::GuardFailure { .. } => {
            // Guard failure handled by handle_fail() before reaching here.
            // If we reach handle_jit_outcome with a GuardFailure, state was
            // already restored — proceed to blackhole resume.
            JitAction::ContinueRunningNormally
        }
        DetailedDriverRunOutcome::Abort { .. } => JitAction::Continue,
    }
}

/// resume.py:1441-1442 allocate_struct(typedescr) → cpu.bh_new(typedescr).
fn allocate_struct(typedescr: &dyn majit_ir::SizeDescr) -> usize {
    let size = typedescr.size();
    let descr = majit_translate::jitcode::BhDescr::Size {
        size,
        type_id: typedescr.type_id(),
        vtable: 0,
        owner: String::new(),
        all_fielddescrs: majit_translate::jitcode::bh_field_specs_from_size_descr(typedescr),
    };
    let (driver, _) = driver_pair();
    driver.meta_interp().backend().bh_new(&descr) as usize
}

/// resume.py:1437-1439 allocate_with_vtable(descr) → exec_new_with_vtable(cpu, descr).
/// llmodel.py:778-782: bh_new_with_vtable uses sizedescr.get_vtable().
fn allocate_with_vtable(descr: &dyn majit_ir::SizeDescr) -> usize {
    let size = descr.size();
    let vtable = descr.vtable();
    let bh_descr = majit_translate::jitcode::BhDescr::Size {
        size,
        type_id: descr.type_id(),
        vtable,
        owner: String::new(),
        all_fielddescrs: majit_translate::jitcode::bh_field_specs_from_size_descr(descr),
    };
    let (driver, _) = driver_pair();
    driver.meta_interp().backend().bh_new_with_vtable(&bh_descr) as usize
}

/// resume.py:945-956 getvirtual_ptr parity.
///
/// Lazily materializes a virtual from rd_virtuals[vidx].
/// Pattern: check cache → allocate_with_vtable/allocate_struct → cache → setfields.
/// RPython caches the REAL object pointer before filling fields, enabling
/// recursive/shared virtual resolution without NULL placeholders.
fn materialize_virtual_from_rd(
    vidx: usize,
    dead_frame: &[Value],
    num_failargs: i32,
    rd_consts: &[majit_ir::Const],
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    virtuals_cache: &mut HashMap<usize, Value>,
) -> Value {
    // resume.py:951: v = self.virtuals_cache.get_ptr(index)
    if let Some(cached) = virtuals_cache.get(&vidx) {
        return cached.clone();
    }
    let Some(virtuals) = rd_virtuals else {
        return Value::Ref(majit_ir::GcRef::NULL);
    };
    let Some(entry) = virtuals.get(vidx) else {
        return Value::Ref(majit_ir::GcRef::NULL);
    };
    // resume.py:1552-1588 decode_* parity.
    fn decode_tagged_fieldnum(
        tagged: i16,
        dead_frame: &[Value],
        num_failargs: i32,
        rd_consts: &[majit_ir::Const],
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> Option<Value> {
        if tagged == majit_ir::resumedata::UNINITIALIZED_TAG {
            return None;
        }
        let (val, tagbits) = majit_metainterp::resume::untag(tagged);
        Some(match tagbits {
            majit_ir::resumedata::TAGBOX => {
                // resume.py:1556-1564: negative index → num + count
                let idx = if val < 0 {
                    (val + num_failargs) as usize
                } else {
                    val as usize
                };
                dead_frame.get(idx).cloned().unwrap_or(Value::Int(0))
            }
            majit_ir::resumedata::TAGINT => Value::Int(val as i64),
            majit_ir::resumedata::TAGCONST => {
                // resume.py:1552-1564: type-aware constant decode
                if tagged == majit_ir::resumedata::NULLREF {
                    return Some(Value::Ref(majit_ir::GcRef::NULL));
                }
                let ci = (val - majit_ir::resumedata::TAG_CONST_OFFSET) as usize;
                rd_consts
                    .get(ci)
                    .copied()
                    .unwrap_or(majit_ir::Const::Int(0))
                    .to_value()
            }
            majit_ir::resumedata::TAGVIRTUAL => {
                return Some(materialize_virtual_from_rd(
                    val as usize,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                ));
            }
            _ => Value::Int(0),
        })
    }
    /// resume.py:1549 decode_int(fieldnum)
    /// Returns the raw i64 value for integer-typed fields.
    fn decode_tagged_fieldnum_int(
        tagged: i16,
        dead_frame: &[Value],
        num_failargs: i32,
        rd_consts: &[majit_ir::Const],
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> i64 {
        match decode_tagged_fieldnum(
            tagged,
            dead_frame,
            num_failargs,
            rd_consts,
            rd_virtuals,
            virtuals_cache,
        ) {
            Some(Value::Int(n)) => n,
            Some(Value::Float(f)) => f.to_bits() as i64,
            Some(Value::Ref(gc)) => gc.0 as i64,
            _ => 0,
        }
    }

    /// resume.py:1546 decode_float(fieldnum)
    /// Returns the raw f64 value for float-typed fields.
    fn decode_tagged_fieldnum_float(
        tagged: i16,
        dead_frame: &[Value],
        num_failargs: i32,
        rd_consts: &[majit_ir::Const],
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> f64 {
        match decode_tagged_fieldnum(
            tagged,
            dead_frame,
            num_failargs,
            rd_consts,
            rd_virtuals,
            virtuals_cache,
        ) {
            Some(Value::Float(f)) => f,
            Some(Value::Int(n)) => f64::from_bits(n as u64),
            _ => 0.0,
        }
    }

    fn box_opt_value(v: &Option<Value>) -> pyre_object::PyObjectRef {
        match v {
            Some(Value::Ref(gc)) => gc.0 as pyre_object::PyObjectRef,
            Some(Value::Int(n)) => pyre_object::intobject::w_int_new(*n),
            Some(Value::Float(f)) => pyre_object::floatobject::w_float_new(*f),
            _ => std::ptr::null_mut(),
        }
    }
    // resume.py:643-760: dispatch by virtual kind.
    match entry.as_ref() {
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            kind, fieldnums, ..
        }
        | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            kind, fieldnums, ..
        } => {
            let clear = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VArrayInfoClear { .. }
            );
            // resume.py:650-670: allocate_array(len, arraydescr, clear)
            let arr_kind = match kind {
                2 => pyre_object::ArrayKind::Float,
                1 => pyre_object::ArrayKind::Int,
                _ => pyre_object::ArrayKind::Ref,
            };
            let array = pyre_object::allocate_array(fieldnums.len(), arr_kind, clear);
            // resume.py:654: cache BEFORE filling — recursive/shared virtuals
            // may reference this vidx during element decoding.
            let result = Value::Ref(majit_ir::GcRef(array as usize));
            virtuals_cache.insert(vidx, result.clone());
            // resume.py:656-670: element kind dispatch + UNINITIALIZED skip.
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue; // resume.py:659: skip UNINITIALIZED
                }
                let v = decode_tagged_fieldnum(
                    fnum,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                if let Some(val) = v {
                    // resume.py:656-670: dispatch by element kind.
                    match val {
                        Value::Float(f) => pyre_object::setarrayitem_float(array, i, f),
                        Value::Int(n) => pyre_object::setarrayitem_int(array, i, n),
                        _ => pyre_object::setarrayitem_ref(array, i, box_opt_value(&Some(val))),
                    }
                }
            }
            return result;
        }
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr,
            size,
            fielddescrs,
            item_size,
            fieldnums,
            ..
        } => {
            // resume.py:748-760: VArrayStructInfo.allocate
            let num_fields = fielddescrs.len();
            // resume.py:749: array = decoder.allocate_array(self.size, self.arraydescr, clear=True)
            // item_size from arraydescr (RPython: self.arraydescr)
            let is = arraydescr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .map(|ad| ad.item_size())
                .unwrap_or(*item_size);
            let array = pyre_object::allocate_array_struct(*size, is);
            // resume.py:751: decoder.virtuals_cache.set_ptr(index, array)
            let result = Value::Ref(majit_ir::GcRef(array as usize));
            virtuals_cache.insert(vidx, result.clone());
            // resume.py:752-759:
            //   p = 0
            //   for i in range(self.size):
            //       for j in range(len(self.fielddescrs)):
            //           num = self.fieldnums[p]
            //           if not tagged_eq(num, UNINITIALIZED):
            //               decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
            //           p += 1
            let mut p = 0;
            for i in 0..*size {
                for j in 0..num_fields {
                    if p >= fieldnums.len() {
                        break;
                    }
                    let fnum = fieldnums[p];
                    p += 1;
                    if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                        continue;
                    }
                    let v = decode_tagged_fieldnum(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    );
                    if let Some(val) = v {
                        // resume.py:757: decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
                        let raw = match val {
                            Value::Int(i) => i,
                            Value::Float(f) => f.to_bits() as i64,
                            Value::Ref(r) => r.0 as i64,
                            Value::Void => 0,
                        };
                        let (fo, fs, ft) = extract_interior_field_info(&fielddescrs[j]);
                        pyre_object::setinteriorfield(array, i, fo, fs, is, ft, raw);
                    }
                }
            }
            return result;
        }
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets,
            descrs,
            fieldnums,
        } => {
            assert_eq!(offsets.len(), descrs.len());
            assert_eq!(offsets.len(), fieldnums.len());
            // resume.py:701-703: buffer = decoder.allocate_raw_buffer(func, size)
            let (driver, _) = driver_pair();
            let calldescr = majit_translate::jitcode::BhCallDescr::from_arg_classes(
                "i".into(),
                'i',
                majit_ir::descr::EffectInfo::MOST_GENERAL,
            );
            let buffer = driver.meta_interp().backend().bh_call_i(
                *func,
                Some(&[*size as i64]),
                None,
                None,
                &calldescr,
            );
            // resume.py:704: cache BEFORE filling fields.
            let result = Value::Int(buffer);
            virtuals_cache.insert(vidx, result.clone());
            let backend = driver.meta_interp().backend();
            // resume.py:705-708: for i in range(len(self.offsets)):
            //     offset = self.offsets[i]; descr = self.descrs[i]
            //     decoder.setrawbuffer_item(buffer, fieldnums[i], offset, descr)
            for (i, &fnum) in fieldnums.iter().enumerate() {
                let di = &descrs[i];
                let bh_descr = majit_translate::jitcode::BhDescr::from_array_descr_info(di);
                // resume.py:1544: assert not descr.is_array_of_pointers()
                assert!(
                    !bh_descr.is_array_of_pointers(),
                    "raw buffer entry must not be pointer type"
                );
                let offset = offsets[i] as i64;
                // resume.py:1545-1550: descr drives decode AND store
                if di.item_type == 2 {
                    // resume.py:1546: newvalue = self.decode_float(fieldnum)
                    let fval = decode_tagged_fieldnum_float(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    );
                    // resume.py:1547: self.cpu.bh_raw_store_f(buffer, offset, newvalue, descr)
                    backend.bh_raw_store_f(buffer, offset, fval, &bh_descr);
                } else {
                    // resume.py:1549: newvalue = self.decode_int(fieldnum)
                    let ival = decode_tagged_fieldnum_int(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    );
                    // resume.py:1550: self.cpu.bh_raw_store_i(buffer, offset, newvalue, descr)
                    backend.bh_raw_store_i(buffer, offset, ival, &bh_descr);
                }
            }
            return result;
        }
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            // resume.py:723-727: base_buffer + offset
            if let Some(fnum) = fieldnums.first() {
                if let Some(Value::Int(base)) = decode_tagged_fieldnum(
                    *fnum,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                ) {
                    let result = Value::Int(base + *offset as i64);
                    // resume.py:727: virtuals_cache.set_int(index, buffer)
                    virtuals_cache.insert(vidx, result.clone());
                    return result;
                }
            }
            return Value::Int(0);
        }
        majit_ir::RdVirtualInfo::Empty => {
            panic!("[jit] materialize_virtual: rd_virtuals[{vidx}] is Empty");
        }
        // resume.py:763-775 VStrPlainInfo.allocate /
        // resume.py:817-829 VUniPlainInfo.allocate —
        //     string = decoder.allocate_string(length)
        //     decoder.virtuals_cache.set_ptr(index, string)
        //     for i, fieldnum in enumerate(self.fieldnums):
        //         if not tagged_eq(fieldnum, UNINITIALIZED):
        //             decoder.string_setitem(string, i, fieldnum)
        majit_ir::RdVirtualInfo::VStrPlainInfo { fieldnums }
        | majit_ir::RdVirtualInfo::VUniPlainInfo { fieldnums } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniPlainInfo { .. }
            );
            let length = fieldnums.len() as i64;
            let (driver, _) = driver_pair();
            let backend = driver.meta_interp().backend();
            // resume.py:1449 allocate_string / resume.py:1482 allocate_unicode.
            let string = if is_unicode {
                backend.bh_newunicode(length)
            } else {
                backend.bh_newstr(length)
            };
            // resume.py:766/820 virtuals_cache.set_ptr BEFORE filling.
            let result = Value::Ref(majit_ir::GcRef(string as usize));
            virtuals_cache.insert(vidx, result.clone());
            // resume.py:771-774/824-827 per-char string_setitem loop.
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                let char_val = decode_tagged_fieldnum_int(
                    fnum,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                if is_unicode {
                    driver
                        .meta_interp()
                        .backend()
                        .bh_unicodesetitem(string, i as i64, char_val);
                } else {
                    driver
                        .meta_interp()
                        .backend()
                        .bh_strsetitem(string, i as i64, char_val);
                }
            }
            return result;
        }
        // resume.py:781-793 VStrConcatInfo.allocate /
        // resume.py:836-848 VUniConcatInfo.allocate —
        //     left  = decoder.decode_ref(self.fieldnums[0])
        //     right = decoder.decode_ref(self.fieldnums[1])
        //     string = decoder.concat_strings(left, right)
        //     decoder.virtuals_cache.set_ptr(index, string)
        majit_ir::RdVirtualInfo::VStrConcatInfo { fieldnums }
        | majit_ir::RdVirtualInfo::VUniConcatInfo { fieldnums } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniConcatInfo { .. }
            );
            let oopspec = if is_unicode {
                majit_ir::effectinfo::OopSpecIndex::UniConcat
            } else {
                majit_ir::effectinfo::OopSpecIndex::StrConcat
            };
            let left_val = decode_tagged_fieldnum_int(
                fieldnums[0],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            let right_val = decode_tagged_fieldnum_int(
                fieldnums[1],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            let (driver, _) = driver_pair();
            let cic = driver
                .meta_interp()
                .callinfocollection()
                .expect(
                    "materialize_virtual_from_rd: MetaInterp.callinfocollection \
                     required for VStr/VUni Concat recovery (resume.py:1143)",
                )
                .clone();
            let (calldescr, func) = cic
                .callinfo_for_oopspec(oopspec)
                .expect("callinfo_for_oopspec missing OS_STR_CONCAT / OS_UNI_CONCAT");
            let cd = calldescr
                .as_call_descr()
                .expect("VStr/VUni Concat calldescr must downcast to CallDescr");
            let bh_calldescr = majit_translate::jitcode::BhCallDescr::from_call_descr(cd);
            // resume.py:1462-1470 concat_strings / resume.py:1489-1497
            // concat_unicodes — cpu.bh_call_r(func, [left, right], descr).
            let backend = driver.meta_interp().backend();
            let result = backend.bh_call_r(
                *func as i64,
                None,
                Some(&[left_val, right_val]),
                None,
                &bh_calldescr,
            );
            let value = Value::Ref(majit_ir::GcRef(result.0));
            virtuals_cache.insert(vidx, value.clone());
            return value;
        }
        // resume.py:799-813 VStrSliceInfo.allocate /
        // resume.py:854-868 VUniSliceInfo.allocate —
        //     largerstr = decoder.decode_ref(self.fieldnums[0])
        //     start     = decoder.decode_int(self.fieldnums[1])
        //     length    = decoder.decode_int(self.fieldnums[2])
        //     string = decoder.slice_string(largerstr, start, length)
        //     decoder.virtuals_cache.set_ptr(index, string)
        majit_ir::RdVirtualInfo::VStrSliceInfo { fieldnums }
        | majit_ir::RdVirtualInfo::VUniSliceInfo { fieldnums } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniSliceInfo { .. }
            );
            let oopspec = if is_unicode {
                majit_ir::effectinfo::OopSpecIndex::UniSlice
            } else {
                majit_ir::effectinfo::OopSpecIndex::StrSlice
            };
            let str_val = decode_tagged_fieldnum_int(
                fieldnums[0],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            let start_val = decode_tagged_fieldnum_int(
                fieldnums[1],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            let length_val = decode_tagged_fieldnum_int(
                fieldnums[2],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            // resume.py:1474 / 1501 — slice_string(str, start, start + length)
            // passes the stop index, not the length.
            let stop_val = start_val + length_val;
            let (driver, _) = driver_pair();
            let cic = driver
                .meta_interp()
                .callinfocollection()
                .expect(
                    "materialize_virtual_from_rd: MetaInterp.callinfocollection \
                     required for VStr/VUni Slice recovery (resume.py:1143)",
                )
                .clone();
            let (calldescr, func) = cic
                .callinfo_for_oopspec(oopspec)
                .expect("callinfo_for_oopspec missing OS_STR_SLICE / OS_UNI_SLICE");
            let cd = calldescr
                .as_call_descr()
                .expect("VStr/VUni Slice calldescr must downcast to CallDescr");
            let bh_calldescr = majit_translate::jitcode::BhCallDescr::from_call_descr(cd);
            // resume.py:1472-1480 slice_string / resume.py:1499-1507
            // slice_unicode — cpu.bh_call_r(func, [str, start, stop], descr).
            let backend = driver.meta_interp().backend();
            let result = backend.bh_call_r(
                *func as i64,
                Some(&[start_val, stop_val]),
                Some(&[str_val]),
                None,
                &bh_calldescr,
            );
            let value = Value::Ref(majit_ir::GcRef(result.0));
            virtuals_cache.insert(vidx, value.clone());
            return value;
        }
        _ => {} // Instance/Struct: fall through
    }
    // Instance/Struct: extract fields for ob_type-based materialization.
    // resume.py:593 fielddescrs + fieldnums
    enum VirtualKind<'a> {
        /// resume.py:612 VirtualInfo — allocate_with_vtable(descr=self.descr).
        Instance {
            descr: &'a Option<majit_ir::DescrRef>,
            known_class: Option<i64>,
        },
        /// resume.py:628 VStructInfo — allocate_struct(self.typedescr).
        Struct {
            typedescr: &'a Option<majit_ir::DescrRef>,
        },
    }
    let (kind, fielddescrs, fieldnums, descr_size) = match entry.as_ref() {
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            known_class,
            fielddescrs,
            fieldnums,
            descr_size,
            ..
        } => (
            VirtualKind::Instance {
                descr,
                known_class: *known_class,
            },
            fielddescrs.as_slice(),
            fieldnums.as_slice(),
            *descr_size,
        ),
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            fielddescrs,
            fieldnums,
            descr_size,
            ..
        } => (
            VirtualKind::Struct { typedescr },
            fielddescrs.as_slice(),
            fieldnums.as_slice(),
            *descr_size,
        ),
        _ => unreachable!(),
    };

    // resume.py:617-621 VirtualInfo.allocate / resume.py:634-637 VStructInfo.allocate
    //   Phase 1: allocate (allocate_with_vtable or allocate_struct)
    //   Phase 2: virtuals_cache.set_ptr(index, struct)  ← BEFORE setfields
    //   Phase 3: self.setfields(decoder, struct)         ← fields filled AFTER

    // Phase 1: allocate.
    let obj_ptr: usize = match kind {
        // resume.py:617-621: VirtualInfo.allocate(descr) → allocate_with_vtable.
        VirtualKind::Instance { descr, known_class } => {
            let ob_type = known_class.unwrap_or(0);
            let int_type_addr = &pyre_object::INT_TYPE as *const _ as i64;
            let float_type_addr = &pyre_object::FLOAT_TYPE as *const _ as i64;
            if ob_type == int_type_addr {
                let tp = unsafe { &*(ob_type as *const pyre_object::pyobject::PyType) };
                let obj = Box::new(pyre_object::intobject::W_IntObject {
                    ob_header: pyre_object::pyobject::PyObject {
                        ob_type: tp,
                        w_class: pyre_object::pyobject::get_instantiate(tp),
                    },
                    intval: 0,
                });
                Box::into_raw(obj) as usize
            } else if ob_type == float_type_addr {
                let tp = unsafe { &*(ob_type as *const pyre_object::pyobject::PyType) };
                let obj = Box::new(pyre_object::floatobject::W_FloatObject {
                    ob_header: pyre_object::pyobject::PyObject {
                        ob_type: tp,
                        w_class: pyre_object::pyobject::get_instantiate(tp),
                    },
                    floatval: 0.0,
                });
                Box::into_raw(obj) as usize
            } else if ob_type != 0 {
                // resume.py:619: allocate_with_vtable(descr=self.descr).
                if let Some(d) = descr {
                    allocate_with_vtable(
                        d.as_size_descr()
                            .expect("VirtualInfo descr must be SizeDescr"),
                    )
                } else {
                    // Fallback: no live descr (decoded from EncodedResumeData).
                    debug_assert!(descr_size > 0, "VirtualInfo must have descr_size");
                    let size = if descr_size > 0 { descr_size } else { 16 };
                    let fallback =
                        majit_ir::make_size_descr_with_vtable(0, size, 0, ob_type as usize);
                    allocate_with_vtable(fallback.as_size_descr().unwrap())
                }
            } else {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit] materialize_virtual: vidx={vidx} Instance with no known_class",
                    );
                }
                return Value::Ref(majit_ir::GcRef::NULL);
            }
        }
        // resume.py:635: VStructInfo.allocate → allocate_struct(self.typedescr)
        VirtualKind::Struct { typedescr, .. } => {
            if let Some(td) = typedescr {
                let sd = td
                    .as_size_descr()
                    .expect("VStruct typedescr must be SizeDescr");
                allocate_struct(sd)
            } else if descr_size > 0 {
                let fallback = majit_ir::make_size_descr_full(0, descr_size, 0);
                let sd = fallback.as_size_descr().unwrap();
                allocate_struct(sd)
            } else {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!("[jit] materialize_virtual: vidx={vidx} Struct with no typedescr",);
                }
                return Value::Ref(majit_ir::GcRef::NULL);
            }
        }
    };

    // Phase 2: cache REAL object pointer BEFORE setting fields.
    // resume.py:620: decoder.virtuals_cache.set_ptr(index, struct)
    let obj_ref = Value::Ref(majit_ir::GcRef(obj_ptr));
    virtuals_cache.insert(vidx, obj_ref.clone());

    // Phase 3: setfields — decode each field and write to object.
    // resume.py:596-603: for each fielddescr, decoder.setfield(struct, num, descr)
    let is_instance = matches!(kind, VirtualKind::Instance { .. });
    match kind {
        VirtualKind::Instance { known_class, .. }
            if known_class == Some(&pyre_object::INT_TYPE as *const _ as i64) =>
        {
            // W_IntObject fast path: find intval field.
            // fielddescrs may include ob_type (offset 0) first.
            let intval_idx = fielddescrs
                .iter()
                .position(|fd| fd.offset == INT_INTVAL_OFFSET)
                .unwrap_or(0);
            if let Some(&tagged) = fieldnums.get(intval_idx) {
                let val = decode_tagged_value(
                    tagged,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                let intval = match val {
                    Value::Int(n) => n,
                    Value::Ref(gc) if !gc.is_null() => unsafe {
                        pyre_object::intobject::w_int_get_value(gc.0 as pyre_object::PyObjectRef)
                    },
                    _ => 0,
                };
                unsafe {
                    (*(obj_ptr as *mut pyre_object::intobject::W_IntObject)).intval = intval;
                }
            }
        }
        VirtualKind::Instance { known_class, .. }
            if known_class == Some(&pyre_object::FLOAT_TYPE as *const _ as i64) =>
        {
            // W_FloatObject fast path: find floatval field.
            let floatval_idx = fielddescrs
                .iter()
                .position(|fd| fd.offset == FLOAT_FLOATVAL_OFFSET)
                .unwrap_or(0);
            if let Some(&tagged) = fieldnums.get(floatval_idx) {
                let val = decode_tagged_value(
                    tagged,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                let floatval = match val {
                    Value::Float(f) => f,
                    Value::Int(bits) => f64::from_bits(bits as u64),
                    _ => 0.0,
                };
                unsafe {
                    (*(obj_ptr as *mut pyre_object::floatobject::W_FloatObject)).floatval =
                        floatval;
                }
            }
        }
        _ => {
            // resume.py:598-602 AbstractVirtualStructInfo.setfields:
            // for each fielddescr, decoder.setfield(struct, num, descr)
            for (i, &tagged) in fieldnums.iter().enumerate() {
                if tagged == majit_ir::resumedata::NULLREF
                    || tagged == majit_ir::resumedata::UNINITIALIZED_TAG
                {
                    continue;
                }
                let val = decode_tagged_value(
                    tagged,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                let raw = match val {
                    Value::Int(n) => n,
                    Value::Float(f) => f.to_bits() as i64,
                    Value::Ref(gc) => gc.0 as i64,
                    _ => 0,
                };
                let Some(descr) = fielddescrs.get(i) else {
                    debug_assert!(false, "fielddescrs missing for field {}", i);
                    continue;
                };
                // Skip vtable slot (offset 0) for Instance — already set by allocate_with_vtable.
                if descr.offset == 0 && is_instance {
                    continue;
                }
                unsafe {
                    let addr = (obj_ptr as *mut u8).add(descr.offset);
                    match descr.field_type {
                        majit_ir::Type::Ref => {
                            let p = match val {
                                Value::Ref(gc) => gc.0 as i64,
                                Value::Int(n) => n,
                                _ => 0,
                            };
                            std::ptr::write(addr as *mut i64, p);
                        }
                        majit_ir::Type::Float => {
                            let bits = match val {
                                Value::Float(f) => f.to_bits(),
                                Value::Int(n) => n as u64,
                                _ => 0,
                            };
                            std::ptr::write(addr as *mut u64, bits);
                        }
                        _ => match descr.field_size {
                            1 => std::ptr::write(addr, raw as u8),
                            2 => std::ptr::write(addr as *mut u16, raw as u16),
                            4 => std::ptr::write(addr as *mut u32, raw as u32),
                            _ => std::ptr::write(addr as *mut i64, raw),
                        },
                    }
                }
            }
        }
    }
    obj_ref
}

/// resume.py:1552-1588 ResumeDataDirectReader decode_int/decode_ref parity.
///
/// Decode a tagged value from rd_numb into a concrete Value.
/// Handles TAGBOX (deadframe), TAGINT (inline), TAGCONST (constant pool),
/// and TAGVIRTUAL (lazy materialization via materialize_virtual_from_rd).
fn decode_tagged_value(
    tagged: i16,
    dead_frame: &[Value],
    num_failargs: i32,
    rd_consts: &[majit_ir::Const],
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    virtuals_cache: &mut HashMap<usize, Value>,
) -> Value {
    let (val, tagbits) = majit_metainterp::resume::untag(tagged);
    match tagbits {
        majit_metainterp::resume::TAGBOX => {
            let idx = if val < 0 {
                (val + num_failargs) as usize
            } else {
                val as usize
            };
            dead_frame.get(idx).cloned().unwrap_or(Value::Int(0))
        }
        majit_metainterp::resume::TAGINT => Value::Int(val as i64),
        majit_metainterp::resume::TAGCONST => rd_consts
            .get((val - majit_metainterp::resume::TAG_CONST_OFFSET) as usize)
            .copied()
            .unwrap_or(majit_ir::Const::Int(0))
            .to_value(),
        majit_metainterp::resume::TAGVIRTUAL => {
            // resume.py:1572: decode_ref(TAGVIRTUAL) → getvirtual_ptr(num)
            materialize_virtual_from_rd(
                val as usize,
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            )
        }
        _ => Value::Int(0),
    }
}

fn decode_exit_layout_values(raw_values: &[i64], layout: &CompiledExitLayout) -> Vec<Value> {
    layout
        .exit_types
        .iter()
        .enumerate()
        .map(|(index, tp)| {
            let raw = raw_values.get(index).copied().unwrap_or(0);
            match tp {
                majit_ir::Type::Int => Value::Int(raw),
                majit_ir::Type::Ref => Value::Ref(majit_ir::GcRef(raw as usize)),
                majit_ir::Type::Float => Value::Float(f64::from_bits(raw as u64)),
                majit_ir::Type::Void => Value::Void,
            }
        })
        .collect()
}

/// Phase A: decode rd_numb + materialize virtuals + restore frame state.
/// RPython: this corresponds to rebuild_from_resumedata (resume.py:1042)
/// which decodes the deadframe into typed values and writes them to the
/// virtualizable/MIFrames. Returns typed values for Phase B and resume PC.
pub(crate) fn decode_and_restore_guard_failure(
    jit_state: &mut PyreJitState,
    meta: &crate::jit::state::PyreMeta,
    raw_values: &[i64],
    exit_layout: &CompiledExitLayout,
) -> Option<(Vec<Value>, usize)> {
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] exit-layout trace_id={} fail_idx={} source_op={:?} rd_numb={} recovery={} resume_layout={}",
            exit_layout.trace_id,
            exit_layout.fail_index,
            exit_layout.source_op_index,
            exit_layout
                .storage
                .as_deref()
                .map(|s| s.rd_numb.len())
                .unwrap_or(0),
            exit_layout.recovery_layout.is_some(),
            exit_layout.resume_layout.is_some(),
        );
    }
    if majit_metainterp::majit_log_enabled() {
        let nraw = raw_values.len();
        let slots: Vec<String> = (0..nraw)
            .map(|i| format!("{:#x}", raw_values[i] as usize))
            .collect();
        eprintln!(
            "[jit] guard-fail: fail_idx={} types={:?} raw_len={} raw=[{}]",
            exit_layout.fail_index,
            exit_layout.exit_types,
            nraw,
            slots.join(", ")
        );
    }
    let dead_frame_typed = decode_exit_layout_values(raw_values, exit_layout);
    // resume.py:1042 rebuild_from_resumedata: decode rd_numb into typed values.
    // compile.py:853 `ResumeGuardDescr` storage — borrow rd_numb / rd_consts
    // from the guard-owned shared Arc instead of a per-guard Vec copy.
    let (typed, mut pending_virtuals_cache) = {
        let storage = exit_layout.storage.as_deref();
        let rd_numb = storage.map(|s| s.rd_numb.as_slice()).unwrap_or(&[]);
        let empty_consts: Vec<majit_ir::Const> = Vec::new();
        let rd_consts: &[majit_ir::Const] = storage.map(|s| s.rd_consts()).unwrap_or(&empty_consts);
        if rd_numb.is_empty() {
            (dead_frame_typed.clone(), HashMap::new())
        } else {
            let (t, rd_numb_pc, virtuals_cache) =
                rebuild_typed_from_rd_numb(raw_values, rd_numb, rd_consts, exit_layout);
            // blackhole.py:337 parity: setposition(jitcode, pc) before
            // consume_one_section. rd_numb_pc = orgpc used by
            // get_list_of_active_boxes during encoding.
            jit_state.resume_pc = rd_numb_pc;
            (t, virtuals_cache)
        }
    };
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] rebuilt typed prefix: {:?}",
            typed.iter().take(6).collect::<Vec<_>>()
        );
    }
    // resume.py:924-926 + 993 parity: `_prepare_next_section` already
    // materializes rd_virtuals lazily via `materialize_virtual_from_rd`.
    // Replay pending fields against the original exit slots plus that
    // shared virtual cache; do not run the legacy pyre-only
    // `recovery_layout` materialization pass here.
    replay_pending_fields(&dead_frame_typed, exit_layout, &mut pending_virtuals_cache);

    // resume.py:1042 rebuild_from_resumedata + pyjitpl.py:3400-3430
    // rebuild_state_after_failure parity: decode rd_numb to reconstruct
    // per-frame values AND write the captured virtualizable_boxes back
    // onto the physical frame via synchronize_virtualizable/write_boxes.
    // pyjitpl.py:3419-3430 — `if vinfo is not None: ... self.synchronize_virtualizable()` —
    // fires on bridge tracing entry so the tracer's subsequent
    // vable_getarrayitem_ref reads see the resume-data values, not the
    // pre-guard heap. pyre mirrors this by selecting the guard-failure
    // vable-sync mode inside `build_resumed_frames`.
    //
    // RPython parity: every guard reaching this path MUST carry rd_numb.
    // `store_final_boxes_in_guard` (optimizeopt/mod.rs:2936) populates
    // it for tracer-origin guards; backend-origin layouts propagate it
    // via `FailDescrLayout.rd_numb` (commit c7ea7cb58b). An empty
    // `rd_numb` here indicates an unported guard-emission site — hard
    // assert so the gap surfaces rather than silently degrade via a
    // pyre-only single-frame synthesis.
    let resumed_frames = {
        // compile.py:853 `ResumeGuardDescr` storage — borrow rd_numb /
        // rd_consts from the guard-owned shared Arc instead of a
        // per-guard Vec copy.
        let storage = exit_layout
            .storage
            .as_deref()
            .expect("rebuild_guard_fail_state: exit_layout.storage missing");
        assert!(
            !storage.rd_numb.is_empty(),
            "rebuild_guard_fail_state: storage.rd_numb is empty (fail_index={})",
            exit_layout.fail_index
        );
        build_resumed_frames(
            raw_values,
            storage.rd_numb.as_slice(),
            storage.rd_consts(),
            exit_layout,
            ResumeVableMode::GuardFailureSync,
        )
    };
    LAST_GUARD_FRAMES.with(|c| *c.borrow_mut() = Some(resumed_frames));

    // virtualizable.py:126: write fields from resumedata to frame.
    let restored = jit_state.restore_guard_failure_values(meta, &typed, &ExceptionState::default());
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] guard-fail restored: ni={} vsd={}",
            jit_state.next_instr(),
            jit_state.valuestackdepth(),
        );
    }

    if restored {
        Some((typed, jit_state.next_instr()))
    } else {
        None
    }
}

/// compile.py:710 resume_in_blackhole(deadframe) →
/// resume.py:1312 blackhole_from_resumedata(deadframe) parity:
/// Build LAST_GUARD_FRAMES directly from deadframe.
/// RPython does NOT call rebuild_from_resumedata (guard restore)
/// before the blackhole path — the blackhole chain consumes
/// deadframe values directly via consume_one_section.
pub(crate) fn build_blackhole_frames_from_deadframe(
    raw_values: &[i64],
    exit_layout: &CompiledExitLayout,
) {
    // resume.py:1312-1343 blackhole_from_resumedata: rd_numb is mandatory.
    // RPython dereferences `storage.rd_numb_list` inside `ResumeDataDirectReader._prepare`
    // (resume.py:1369-1372); there is no fallback for a missing numbering.
    // Pyre borrows `rd_numb` / `rd_consts` from the guard-owned shared
    // `ResumeStorage` Arc (compile.py:853 `ResumeGuardDescr`) so post-
    // eviction backend-origin layouts still carry it.
    let storage = exit_layout
        .storage
        .as_deref()
        .expect("build_blackhole_frames_from_deadframe: exit_layout.storage missing");
    let rd_numb = storage.rd_numb.as_slice();
    let rd_consts = storage.rd_consts();
    assert!(
        !rd_numb.is_empty(),
        "build_blackhole_frames_from_deadframe: storage.rd_numb is empty (fail_index={})",
        exit_layout.fail_index
    );
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][deadframe] fail_index={} rd_numb_len={} exit_types={}",
            exit_layout.fail_index,
            rd_numb.len(),
            exit_layout.exit_types.len(),
        );
    }
    // resume.py:1399 consume_vable_info parity: blackhole path writes the
    // captured virtualizable fields directly from the resume reader stream.
    let resumed_frames = build_resumed_frames(
        raw_values,
        rd_numb,
        rd_consts,
        exit_layout,
        ResumeVableMode::BlackholeConsume,
    );
    LAST_GUARD_FRAMES.with(|c| *c.borrow_mut() = Some(resumed_frames));
}

/// Decode rd_numb to produce typed values via
/// `majit_ir::resumedata::rebuild_from_numbering`. Each slot is TAGBOX
/// (deadframe), TAGCONST (constant), TAGINT (small int), or TAGVIRTUAL
/// (virtual to materialize). Single-frame only (no per-jitcode liveness).
///
/// Returns `(typed_values, rd_numb_frame_pc)`. The frame PC from rd_numb
/// is the liveness PC used by get_list_of_active_boxes during encoding.
/// The recovery side MUST use this same PC for expand — NOT next_instr
/// (which may differ by 1+ due to cache slots).
fn rebuild_typed_from_rd_numb(
    raw_values: &[i64],
    rd_numb: &[u8],
    rd_consts: &[majit_ir::Const],
    exit_layout: &CompiledExitLayout,
) -> (Vec<Value>, Option<usize>, HashMap<usize, Value>) {
    use majit_ir::resumedata::rebuild_from_numbering;

    let (_num_failargs, vable_values, _vref_values, frames) =
        rebuild_from_numbering(rd_numb, rd_consts, &exit_layout.exit_types, None);

    // resume.py:1045 consume_vref_and_vable_boxes parity.
    // vable_array format: [frame_ptr, ni, code, vsd, ns, locals..., stack...]
    // (opencoder.py:722 moves virtualizable_ptr to front).
    if majit_metainterp::majit_log_enabled() && !vable_values.is_empty() {
        eprintln!(
            "[jit] guard-fail: vable_values={} items: {:?}",
            vable_values.len(),
            vable_values.iter().take(6).collect::<Vec<_>>()
        );
    }

    let dead_frame_typed = decode_exit_layout_values(raw_values, exit_layout);
    let mut virtuals_cache: HashMap<usize, Value> = HashMap::new();

    // resume.py:1083 + pyjitpl.py:3400-3428 parity:
    // Decode vable_values into typed prefix [frame_ptr, ni, code, vsd, ns, locals..., stack...].
    // In RPython, virtualizable_boxes are restored first, then synchronize_virtualizable
    // writes them back to the actual frame object.
    fn decode_rv(
        rv: &majit_ir::resumedata::RebuiltValue,
        dead_frame_typed: &[Value],
        exit_layout: &CompiledExitLayout,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> Value {
        use majit_ir::resumedata::RebuiltValue;
        match rv {
            RebuiltValue::Box(idx, _tp) => {
                dead_frame_typed.get(*idx).cloned().unwrap_or(Value::Int(0))
            }
            // history.py:220-360 Const → Value: direct variant projection.
            RebuiltValue::Const(c) => c.to_value(),
            RebuiltValue::Virtual(vidx) => {
                let storage = exit_layout.storage.as_deref();
                let rd_consts = storage.map(|s| s.rd_consts()).unwrap_or(&[]);
                let rd_virtuals = storage.map(|s| s.rd_virtuals.as_slice());
                materialize_virtual_from_rd(
                    *vidx,
                    dead_frame_typed,
                    exit_layout.exit_types.len() as i32,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                )
            }
            _ => Value::Int(0),
        }
    }
    // resume.py:1042-1057 rebuild_from_resumedata parity:
    // RPython produces TWO streams:
    //   1. virtualizable_boxes (consume_vref_and_vable → synchronize_virtualizable)
    //   2. frame registers (consume_boxes per frame)
    // pyjitpl.py:3419-3430: virtualizable_boxes restored, then
    // synchronize_virtualizable writes them back to the heap.
    // Frame registers fill frame.registers_i/r/f independently.

    // `vable_values` is heap-layout (opencoder.py:718 `_list_of_boxes_virtualizable`):
    //   [frame_ptr, vable_static_fields..., array_items...]
    // `_list_of_boxes_virtualizable` excludes any reds that are not virtualizable
    // static fields (ec is a per-thread global), so the encoded prefix has
    // `1 + NUM_VABLE_SCALARS` entries — never `NUM_SCALAR_INPUTARGS`, which counts
    // `NUM_EXTRA_REDS` ec slot(s) on the trace inputarg side.
    //
    // `restore_guard_failure_values` and downstream consumers index this header
    // with `SYM_*_IDX`, which include the `NUM_EXTRA_REDS` shift. Inject placeholder
    // ec slot(s) between the frame and the static fields here so the trace-layout
    // indices align. The ec value itself is never written back (ec is reloaded from
    // `get_execution_context()` on resume), so a `Value::Void` placeholder is safe.
    let num_scalars = pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS;
    let num_extra_reds = pyre_jit_trace::virtualizable_gen::NUM_EXTRA_REDS;
    let heap_scalar_count = 1 + pyre_jit_trace::virtualizable_gen::NUM_VABLE_SCALARS;
    let header: Vec<Value> = if vable_values.len() >= heap_scalar_count {
        let mut h = Vec::with_capacity(num_scalars);
        h.push(decode_rv(
            &vable_values[0],
            &dead_frame_typed,
            exit_layout,
            &mut virtuals_cache,
        ));
        for _ in 0..num_extra_reds {
            h.push(Value::Void);
        }
        for i in 1..heap_scalar_count {
            h.push(decode_rv(
                &vable_values[i],
                &dead_frame_typed,
                exit_layout,
                &mut virtuals_cache,
            ));
        }
        h
    } else {
        Vec::new()
    };

    // resume.py:1049-1056: rebuild_from_resumedata iterates all frames
    // via newframe()+consume_boxes(). For guard-failure restore into JIT
    // state (restore_guard_failure_values), only the outermost frame's
    // values matter — inner frames are handled by build_resumed_frames →
    // resume_in_blackhole. rd_numb frames are innermost-first; last = outermost.
    let mut typed = header;
    if let Some(outermost) = frames.last() {
        _prepare_next_section(
            outermost,
            &dead_frame_typed,
            exit_layout,
            &mut typed,
            &mut virtuals_cache,
        );
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] guard-fail: rd_numb decoded {} slots from {} frame(s)",
            typed.len(),
            frames.len(),
        );
    }

    // resume.py:1383 parity: liveness PC = frame.pc from rd_numb
    // (the same PC used by get_list_of_active_boxes during encoding).
    let rd_numb_pc = frames.last().map(|f| f.pc as usize);
    (typed, rd_numb_pc, virtuals_cache)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResumeVableMode {
    GuardFailureSync,
    BlackholeConsume,
}

fn value_to_static_vable_bits(value: &Value, expected_type: Type, field_index: usize) -> i64 {
    match (expected_type, value) {
        (Type::Int, Value::Int(v)) => *v,
        (Type::Float, Value::Float(v)) => v.to_bits() as i64,
        (Type::Ref, Value::Ref(r)) => r.as_usize() as i64,
        (ty, other) => {
            panic!("virtualizable static field {field_index} expected {ty:?}, got {other:?}")
        }
    }
}

fn value_to_vable_array_item_bits(
    value: &Value,
    expected_type: Type,
    array_index: usize,
    item_index: usize,
) -> i64 {
    match expected_type {
        Type::Ref => match value {
            Value::Ref(r) => r.as_usize() as i64,
            Value::Int(i) => pyre_object::intobject::w_int_new(*i) as i64,
            Value::Float(f) => pyre_object::floatobject::w_float_new(*f) as i64,
            other => panic!(
                "virtualizable array item [{array_index}][{item_index}] expected Ref, got {other:?}"
            ),
        },
        Type::Int => match value {
            Value::Int(v) => *v,
            other => panic!(
                "virtualizable array item [{array_index}][{item_index}] expected Int, got {other:?}"
            ),
        },
        Type::Float => match value {
            Value::Float(v) => v.to_bits() as i64,
            other => panic!(
                "virtualizable array item [{array_index}][{item_index}] expected Float, got {other:?}"
            ),
        },
        ty => {
            panic!("virtualizable array item [{array_index}][{item_index}] unsupported type {ty:?}")
        }
    }
}

fn value_to_vable_identity_bits(value: &Value) -> i64 {
    match value {
        Value::Ref(r) => r.as_usize() as i64,
        other => panic!("virtualizable identity expected Ref, got {other:?}"),
    }
}

fn sync_virtualizable_after_guard_failure(
    resolved_vable: &[Value],
    frame_u8: *mut u8,
    vinfo: &majit_metainterp::virtualizable::VirtualizableInfo,
) {
    unsafe {
        // pyjitpl.py:3427-3429: reset token before synchronize_virtualizable().
        vinfo.reset_vable_token(frame_u8);
    }
    let expected_total_without_identity = vinfo.num_static_extra_boxes
        + (0..vinfo.array_fields.len())
            .map(|array_index| unsafe {
                vinfo.get_array_length(frame_u8.cast_const(), array_index)
            })
            .sum::<usize>();
    assert_eq!(
        resolved_vable.len(),
        expected_total_without_identity + 1,
        "rebuild_guard_fail_state: virtualizable box count mismatch (expected {}, got {})",
        expected_total_without_identity + 1,
        resolved_vable.len(),
    );

    let mut boxes: Vec<i64> = Vec::with_capacity(expected_total_without_identity + 1);
    let mut cursor = 1;
    for (field_index, field) in vinfo.static_fields.iter().enumerate() {
        boxes.push(value_to_static_vable_bits(
            &resolved_vable[cursor],
            field.field_type,
            field_index,
        ));
        cursor += 1;
    }
    for (array_index, array_field) in vinfo.array_fields.iter().enumerate() {
        let array_len = unsafe { vinfo.get_array_length(frame_u8.cast_const(), array_index) };
        for item_index in 0..array_len {
            boxes.push(value_to_vable_array_item_bits(
                &resolved_vable[cursor],
                array_field.item_type,
                array_index,
                item_index,
            ));
            cursor += 1;
        }
    }
    debug_assert_eq!(cursor, resolved_vable.len());
    boxes.push(value_to_vable_identity_bits(&resolved_vable[0]));

    unsafe {
        vinfo.write_boxes_to_heap(frame_u8, &boxes);
    }
}

/// Decode rd_numb into per-frame ResumedFrame chain via
/// `majit_ir::resumedata::rebuild_from_numbering`.
/// Single-frame only (RPython's blackhole_from_resumedata uses
/// per-jitcode liveness for multi-frame decode).
fn build_resumed_frames(
    raw_values: &[i64],
    rd_numb: &[u8],
    rd_consts: &[majit_ir::Const],
    exit_layout: &CompiledExitLayout,
    vable_mode: ResumeVableMode,
) -> Vec<crate::call_jit::ResumedFrame> {
    use majit_ir::resumedata::rebuild_from_numbering;

    // resume.py:1049-1055 parity: consume_boxes(f.get_current_position_info())
    // RPython uses jitcode liveness (jitcode.position_info) to know how many
    // boxes each frame contributes. There is no out-of-band frame size — the
    // decoder reads jitcode liveness at the frame's resume pc.
    let cb = pyre_jit_trace::state::frame_value_count_at;
    let (_num_failargs, vable_values, _vref_values, frames) =
        rebuild_from_numbering(rd_numb, rd_consts, &exit_layout.exit_types, Some(&cb));

    let dead_frame_typed = decode_exit_layout_values(raw_values, exit_layout);
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][resume] exit_types={:?} dead_frame={:?} vable={} frames={}",
            exit_layout.exit_types,
            dead_frame_typed,
            vable_values.len(),
            frames.len()
        );
    }
    let mut virtuals_cache: HashMap<usize, Value> = HashMap::new();

    // resume.py:1045 consume_vref_and_vable parity:
    // Reconstruct header [frame_ptr, ni, code, vsd, ns] from vable_values.
    fn resolve_rebuilt_value(
        rv: &majit_ir::resumedata::RebuiltValue,
        dead_frame_typed: &[Value],
        exit_layout: &CompiledExitLayout,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> Value {
        use majit_ir::resumedata::RebuiltValue;
        match rv {
            RebuiltValue::Box(idx, _tp) => {
                dead_frame_typed.get(*idx).cloned().unwrap_or(Value::Int(0))
            }
            // history.py:220-360 Const → Value: direct variant projection.
            RebuiltValue::Const(c) => c.to_value(),
            RebuiltValue::Virtual(vidx) => {
                let storage = exit_layout.storage.as_deref();
                let rd_consts = storage.map(|s| s.rd_consts()).unwrap_or(&[]);
                let rd_virtuals = storage.map(|s| s.rd_virtuals.as_slice());
                materialize_virtual_from_rd(
                    *vidx,
                    dead_frame_typed,
                    exit_layout.exit_types.len() as i32,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                )
            }
            _ => Value::Int(0),
        }
    }
    // resume.py:1045 consume_vref_and_vable: vable header is extracted
    // AFTER _prepare_next_section materializes virtuals. The post-section
    // block below is the authoritative extraction. vable_values is always
    // non-empty for guards with complete resume data (resume.py:397 asserts
    // resume_position >= 0). The no-snapshot fallback in store_final_boxes_in_guard
    // now encodes fail_args[0..3] as vable_array to maintain this invariant.

    let mut all_values: Vec<Vec<Value>> = Vec::with_capacity(frames.len());
    for (fidx, frame) in frames.iter().enumerate() {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[dynasm-debug] _prepare_next_section frame={}/{} pc={} values_len={}",
                fidx,
                frames.len(),
                frame.pc,
                frame.values.len()
            );
        }
        let mut values = Vec::new();
        _prepare_next_section(
            frame,
            &dead_frame_typed,
            exit_layout,
            &mut values,
            &mut virtuals_cache,
        );
        all_values.push(values);
    }
    // RPython parity: _prepare_next_section + materialize_virtual_from_rd
    // is the authoritative path for virtual materialization.
    // Pending-field replay must consume the same deadframe slots and shared
    // virtual cache; the legacy pyre-only recovery_layout materializer has
    // been removed.
    // resume.py:993 _prepare_pendingfields: apply ONCE for the whole reader.
    // No header — values = slot registers only.
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[dynasm-debug] before replay_pending_fields, frames={}",
            all_values.len()
        );
    }
    replay_pending_fields(&dead_frame_typed, exit_layout, &mut virtuals_cache);
    if majit_metainterp::majit_log_enabled() {
        eprintln!("[dynasm-debug] after replay_pending_fields");
    }

    // opencoder.py:722 _list_of_boxes_virtualizable: snapshot reorders
    // virtualizable_ptr from end to front. `vable_values` is heap-layout
    // (no ec): [frame, vable_static_fields..., array_items...]. SYM_*_IDX
    // include the `NUM_EXTRA_REDS` shift for trace inputarg layout, so
    // subtract `NUM_EXTRA_REDS` to land on heap-layout positions.
    let extra = pyre_jit_trace::virtualizable_gen::NUM_EXTRA_REDS;
    let ni_idx = pyre_jit_trace::virtualizable_gen::SYM_LAST_INSTR_IDX as usize - extra;
    let code_idx = pyre_jit_trace::virtualizable_gen::SYM_PYCODE_IDX as usize - extra;
    let vsd_idx = pyre_jit_trace::virtualizable_gen::SYM_VALUESTACKDEPTH_IDX as usize - extra;
    let ns_idx = pyre_jit_trace::virtualizable_gen::SYM_W_GLOBALS_IDX as usize - extra;

    // Resolve ALL vable fields from resume data.
    // vable_values = [frame_ptr, last_instr, pycode, valuestackdepth,
    //                  debugdata, lastblock, w_globals, array...]
    // RPython reader.load_next_value_of_type reads ALL values sequentially.
    let resolved_vable: Vec<Value> = (0..vable_values.len())
        .map(|i| {
            resolve_rebuilt_value(
                &vable_values[i],
                &dead_frame_typed,
                exit_layout,
                &mut virtuals_cache,
            )
        })
        .collect();
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][resume][vable-values] rebuilt={:?} resolved={:?}",
            vable_values, resolved_vable
        );
    }

    let vable_frame_ptr = resolved_vable
        .first()
        .map(|v| match v {
            Value::Ref(r) => r.as_usize() as *mut pyre_interpreter::pyframe::PyFrame,
            Value::Int(v) => *v as *mut pyre_interpreter::pyframe::PyFrame,
            _ => std::ptr::null_mut(),
        })
        .unwrap_or(std::ptr::null_mut());
    let vable_ni = resolved_vable
        .get(ni_idx)
        .map(|v| match v {
            Value::Int(v) => (*v + 1) as usize,
            _ => 0,
        })
        .unwrap_or(0);

    let vable_vsd = resolved_vable
        .get(vsd_idx)
        .map(|v| match v {
            Value::Int(v) => *v as usize,
            _ => 0,
        })
        .unwrap_or(0);

    // virtualizable.py:86-99 read_boxes: ALL static fields in declared order.
    let vable_pycode: *const () = resolved_vable
        .get(code_idx)
        .map(|v| match v {
            Value::Ref(r) => r.as_usize() as *const (),
            Value::Int(v) => *v as *const (),
            _ => std::ptr::null(),
        })
        .unwrap_or(std::ptr::null());

    let vable_ns: *const () = resolved_vable
        .get(ns_idx)
        .map(|v| match v {
            Value::Ref(r) => r.as_usize() as *const (),
            Value::Int(v) => *v as *const (),
            _ => std::ptr::null(),
        })
        .unwrap_or(std::ptr::null());

    // resume.py:1399-1408 consume_vable_info literal port:
    //
    //     def consume_vable_info(self, vinfo, vable_size):
    //         assert vable_size
    //         virtualizable = self.next_ref()
    //         assert vinfo.get_total_size(virtualizable) == vable_size - 1
    //         vinfo.reset_token_gcref(virtualizable)
    //         vinfo.write_from_resume_data_partial(virtualizable, self)
    //
    // Called from `consume_vref_and_vable` (resume.py:1424-1431) in the
    // ResumeDataBoxReader path that `blackhole_from_resumedata`
    // (resume.py:1312-1342) drives. majit's blackhole entry mirrors that
    // path by selecting `ResumeVableMode::BlackholeConsume`, which writes
    // the captured vable payload to the heap BEFORE any blackhole
    // interpreter starts running.
    //
    // Guard-failure recovery is a different upstream helper chain:
    // pyjitpl.py:3419-3430 stores `self.virtualizable_boxes`, resets the
    // token, then calls `self.synchronize_virtualizable()` which ends at
    // virtualizable.py:101-113 `write_boxes`. That exact path is modeled by
    // `ResumeVableMode::GuardFailureSync`.
    if !vable_frame_ptr.is_null() {
        let frame_u8 = vable_frame_ptr as *mut u8;
        // resume.py:1312-1314 blackhole_from_resumedata parity:
        //     vinfo = self.jitdriver_sd.virtualizable_info
        // Use the JIT driver's cached `Arc<VirtualizableInfo>` set once by
        // `set_virtualizable_info` at JIT_DRIVER init rather than rebuilding
        // a fresh instance, so the guard-failure recovery path shares a
        // single vinfo identity with the tracing / blackhole consumers.
        let vinfo = crate::eval::driver_pair().1.clone();
        match vable_mode {
            ResumeVableMode::GuardFailureSync => {
                sync_virtualizable_after_guard_failure(&resolved_vable, frame_u8, &vinfo);
            }
            ResumeVableMode::BlackholeConsume => unsafe {
                // resume.py:1407 reset_token_gcref
                vinfo.reset_vable_token(frame_u8);

                // virtualizable.py:126-137 write_from_resume_data_partial:
                //
                //     for FIELDTYPE, fieldname in unroll_static_fields:
                //         x = reader.load_next_value_of_type(FIELDTYPE)
                //         setattr(virtualizable, fieldname, x)
                //
                // resolved_vable layout (opencoder.py:722
                // _list_of_boxes_virtualizable parity):
                //   [0]            virtualizable_ptr  (= the frame itself)
                //   [1..num_scalars] static fields
                //     (last_instr, pycode, valuestackdepth, debugdata,
                //      lastblock, w_globals — line-by-line PyPy parity
                //      with `interp_jit.py:25-31`)
                //   [num_scalars..] array items
                //
                // resume.py:1406 exact-size invariant:
                // assert vinfo.get_total_size(virtualizable) == vable_size - 1.
                let expected = vinfo.num_static_extra_boxes
                    + (0..vinfo.array_fields.len())
                        .map(|array_index| {
                            vinfo.get_array_length(frame_u8.cast_const(), array_index)
                        })
                        .sum::<usize>();
                assert_eq!(
                    resolved_vable.len(),
                    expected + 1,
                    "consume_vable_info: virtualizable box count mismatch (expected {}, got {})",
                    expected + 1,
                    resolved_vable.len(),
                );
                let static_boxes: Vec<i64> = vinfo
                    .static_fields
                    .iter()
                    .enumerate()
                    .map(|(field_index, field)| {
                        value_to_static_vable_bits(
                            &resolved_vable[field_index + 1],
                            field.field_type,
                            field_index,
                        )
                    })
                    .collect();

                let mut cursor = 1 + vinfo.num_static_extra_boxes;
                let mut array_boxes: Vec<Vec<i64>> = Vec::with_capacity(vinfo.array_fields.len());
                for (array_index, array_field) in vinfo.array_fields.iter().enumerate() {
                    let array_len = vinfo.get_array_length(frame_u8.cast_const(), array_index);
                    let mut array_items: Vec<i64> = Vec::with_capacity(array_len);
                    for item_index in 0..array_len {
                        array_items.push(value_to_vable_array_item_bits(
                            &resolved_vable[cursor],
                            array_field.item_type,
                            array_index,
                            item_index,
                        ));
                        cursor += 1;
                    }
                    array_boxes.push(array_items);
                }
                debug_assert_eq!(cursor, resolved_vable.len());
                vinfo.write_all_boxes(frame_u8, &static_boxes, &array_boxes);
            },
        }
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][resume][vable-sync] mode={vable_mode:?} frame {:?}",
                vable_frame_ptr
            );
            if !vable_frame_ptr.is_null() {
                let f = unsafe { &*vable_frame_ptr };
                eprintln!(
                    "[jit][resume][vable-sync] frame after write: ni={} vsd={} code={:?} ns={:?} debugdata={:?} lastblock={:?} vable_token={} array_len={}",
                    f.next_instr(),
                    f.valuestackdepth,
                    f.pycode,
                    f.w_globals,
                    f.debugdata,
                    f.lastblock,
                    f.vable_token,
                    f.locals_w().len(),
                );
            }
        }
    }

    let mut result = Vec::with_capacity(frames.len());
    for (idx, (frame, values)) in frames.iter().zip(all_values.into_iter()).enumerate() {
        // resume.py:1338 read_jitcode_pos_pc parity:
        // py_pc comes from rd_numb frame header (frame.pc = orgpc).
        // pc=0 is valid (function start). pc=-1 = no-snapshot sentinel.
        let py_pc = if frame.pc >= 0 {
            frame.pc as usize
        } else {
            // No-snapshot guard: fall back to vable ni.
            vable_ni
        };
        // resume.py:1339 jitcodes[jitcode_pos]:
        // Outermost frame (last): code from vable resume data.
        // Inner frames: code from jitcode_index registry (inlined calls).
        let is_outermost = frames.len() == 1 || idx == frames.len() - 1;
        let w_code = if is_outermost {
            // virtualizable.py:86-99: code from resume data, not heap.
            if !vable_pycode.is_null() {
                vable_pycode
            } else if !vable_frame_ptr.is_null() {
                unsafe { (*vable_frame_ptr).pycode }
            } else {
                std::ptr::null()
            }
        } else {
            pyre_jit_trace::state::code_for_jitcode_index(frame.jitcode_index)
                .unwrap_or(std::ptr::null())
        };
        let raw_code = if !w_code.is_null() {
            unsafe {
                pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                    as *const pyre_interpreter::CodeObject
            }
        } else {
            std::ptr::null()
        };
        // resume.py:1095/1067 parity: virtualizable_ptr is the JIT driver's
        // PyFrame, shared across the entire blackhole chain. RPython's
        // newframe(jitcode) creates a fresh MIFrame for each inner section
        // (inlined call) but the virtualizable on the JIT driver is the same.
        // pyre's BlackholeInterpreter holds its own register state per
        // section, so inner frames don't need a PyFrame — they only need
        // virtualizable_ptr to write back to the outermost frame at the
        // merge point. Use vable_frame_ptr for ALL sections.
        let frame_ptr = vable_frame_ptr;
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[dynasm-debug] frame_ptr={:?} py_pc={} idx={}",
                frame_ptr, py_pc, idx
            );
        }
        // Per-frame VSD: outermost uses vable_vsd, inner frames derive
        // from their code's nlocals + snapshot stack depth.
        let vsd = if frames.len() == 1 || idx == frames.len() - 1 {
            // resume.py:1399 parity: outermost frame's vsd comes directly
            // from the virtualizable. RPython does not sentinel-check 0.
            vable_vsd
        } else if !raw_code.is_null() {
            let nlocals = unsafe { &*raw_code }.varnames.len();
            nlocals + values.len().saturating_sub(nlocals)
        } else {
            values.len()
        };
        // virtualizable.py:86-99: namespace from resume data.
        let namespace = if is_outermost {
            if !vable_ns.is_null() {
                vable_ns
            } else if !vable_frame_ptr.is_null() {
                unsafe { (*vable_frame_ptr).w_globals as *const () }
            } else {
                std::ptr::null()
            }
        } else {
            // Inner frames share the chain virtualizable's namespace.
            vable_ns
        };
        result.push(crate::call_jit::ResumedFrame {
            code: w_code,
            py_pc,
            rd_numb_pc: if frame.pc >= 0 {
                Some(frame.pc as usize)
            } else {
                None
            },
            frame_ptr,
            vsd,
            namespace,
            values,
        });
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] build_resumed_frames: {} frame(s) from rd_numb",
            result.len()
        );
    }

    result
}

/// resume.py:1017-1026 _prepare_next_section: decode one frame's slots
/// from rd_numb tagged values into typed Value vector.
fn _prepare_next_section(
    frame: &majit_ir::resumedata::RebuiltFrame,
    dead_frame_typed: &[Value],
    exit_layout: &CompiledExitLayout,
    typed: &mut Vec<Value>,
    virtuals_cache: &mut HashMap<usize, Value>,
) {
    use majit_ir::resumedata::RebuiltValue;
    let storage = exit_layout.storage.as_deref();
    let rd_consts = storage.map(|s| s.rd_consts()).unwrap_or(&[]);
    let rd_virtuals = storage.map(|s| s.rd_virtuals.as_slice());
    let num_failargs = exit_layout.exit_types.len() as i32;
    for val in &frame.values {
        typed.push(match val {
            RebuiltValue::Box(idx, _tp) => {
                dead_frame_typed.get(*idx).cloned().unwrap_or(Value::Int(0))
            }
            // history.py:220-360 Const → Value: direct variant projection.
            RebuiltValue::Const(c) => c.to_value(),
            // resume.py:1572: decode_ref(TAGVIRTUAL) → getvirtual_ptr(num)
            RebuiltValue::Virtual(vidx) => materialize_virtual_from_rd(
                *vidx,
                dead_frame_typed,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            ),
            // resume.py:131 UNINITIALIZED parity: dead/uninitialized slots
            // stay at default. In pyre, PY_NULL via Value::Void.
            RebuiltValue::Unassigned => Value::Void,
        });
    }
}

/// Guard failure recovery: reconstruct virtual objects from their
/// field values stored as extra fail_args after null (NONE) slots.
///
/// When the optimizer places a virtual in fail_args, it sets the
/// resume.py:945/993 parity: virtual materialization via rd_virtuals.
/// Called from Cranelift's guard failure handler via TLS callback.
/// RPython uses rd_virtuals/rd_pendingfields for precise materialization.
/// The backend callback itself stays a no-op; runtime restore goes through
/// rebuild_from_resumedata + materialize_virtual_from_rd.
fn rebuild_state_after_failure(_outputs: &mut [i64], _types: &[majit_ir::Type]) {
    // RPython: materialization happens in rebuild_from_resumedata via
    // getvirtual_ptr (resume.py:945) and _prepare_pendingfields (resume.py:993).
    // The Cranelift callback is a no-op; decode_and_restore_guard_failure
    // performs the real resume-data rebuild.
}

/// virtual's slot to NONE and appends field values (ob_type, intval).
/// On guard failure, we detect contiguous null Ref slots at the end
/// of the locals/stack region and pair them with trailing Int fields.
///
/// resume.py:993-1007 _prepare_pendingfields: replay deferred field writes.
///
/// After virtual materialization, pending SETFIELD_GC/SETARRAYITEM_GC
/// ops stored in rd_pendingfields are replayed on the materialized objects.
/// This ensures lazy field writes that were deferred during optimization
/// take effect when the guard fires.
fn replay_pending_fields(
    dead_frame_typed: &[Value],
    exit_layout: &CompiledExitLayout,
    virtuals_cache: &mut HashMap<usize, Value>,
) {
    let Some(ref recovery) = exit_layout.recovery_layout else {
        return;
    };
    if recovery.pending_field_layouts.is_empty() {
        return;
    }

    let empty_consts: Vec<majit_ir::Const> = Vec::new();
    let rd_consts: &[majit_ir::Const] = exit_layout
        .storage
        .as_deref()
        .map(|s| s.rd_consts())
        .unwrap_or(&empty_consts);
    let rd_virtuals = exit_layout
        .storage
        .as_deref()
        .map(|s| s.rd_virtuals.as_slice());
    let num_failargs = exit_layout.exit_types.len() as i32;
    let value_to_raw_bits = |value: Value| match value {
        Value::Int(i) => i,
        Value::Float(f) => f.to_bits() as i64,
        Value::Ref(r) => r.0 as i64,
        Value::Void => 0,
    };
    let mut resolve_value = |src: &majit_backend::ExitValueSourceLayout| -> Option<i64> {
        match src {
            majit_backend::ExitValueSourceLayout::ExitValue(idx) => {
                dead_frame_typed.get(*idx).cloned().map(value_to_raw_bits)
            }
            majit_backend::ExitValueSourceLayout::Constant(c) => Some(*c),
            majit_backend::ExitValueSourceLayout::Virtual(vidx) => {
                Some(value_to_raw_bits(materialize_virtual_from_rd(
                    *vidx,
                    dead_frame_typed,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                )))
            }
            majit_backend::ExitValueSourceLayout::Uninitialized
            | majit_backend::ExitValueSourceLayout::Unavailable => None,
        }
    };

    for pf in &recovery.pending_field_layouts {
        let Some(target_ptr) = resolve_value(&pf.target) else {
            continue;
        };
        let Some(value_raw) = resolve_value(&pf.value) else {
            continue;
        };
        if target_ptr == 0 {
            continue; // null target — skip
        }
        // resume.py:1003-1007 _prepare_pendingfields parity:
        //   if itemindex < 0: setfield(struct, fieldnum, descr)
        //   else:             setarrayitem(struct, itemindex, fieldnum, descr)
        //
        // resume.py:1509-1518 setfield / 1520-1530 setarrayitem:
        //   descr.is_pointer_field() → bh_setfield_gc_r / bh_setarrayitem_gc_r
        //   descr.is_float_field()   → bh_setfield_gc_f / bh_setarrayitem_gc_f
        //   else                     → bh_setfield_gc_i / bh_setarrayitem_gc_i
        let addr = if pf.is_array_item {
            // setarrayitem: base + offset + item_index * item_size
            let item_index = pf.item_index.unwrap_or(0);
            target_ptr as usize + pf.field_offset + item_index * pf.field_size
        } else {
            // setfield: base + offset
            target_ptr as usize + pf.field_offset
        };
        unsafe {
            match pf.field_type {
                majit_ir::Type::Ref => {
                    // bh_setfield_gc_r: store pointer
                    std::ptr::write(addr as *mut usize, value_raw as usize);
                }
                majit_ir::Type::Float => {
                    // bh_setfield_gc_f: store f64
                    std::ptr::write(addr as *mut u64, value_raw as u64);
                }
                majit_ir::Type::Int | majit_ir::Type::Void => {
                    // bh_setfield_gc_i: store integer (size-aware)
                    match pf.field_size {
                        8 => std::ptr::write(addr as *mut i64, value_raw),
                        4 => std::ptr::write(addr as *mut i32, value_raw as i32),
                        2 => std::ptr::write(addr as *mut i16, value_raw as i16),
                        1 => std::ptr::write(addr as *mut u8, value_raw as u8),
                        _ => std::ptr::write(addr as *mut i64, value_raw),
                    }
                }
            }
        }
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit] replay_pending_field: type={:?} offset={} size={} target={:#x} value={:#x}",
                pf.field_type,
                pf.field_offset,
                pf.field_size,
                target_ptr as usize,
                value_raw as usize
            );
        }
    }
}

pub(crate) fn build_jit_state(
    frame: &PyFrame,
    virtualizable_info: &majit_metainterp::virtualizable::VirtualizableInfo,
) -> PyreJitState {
    let mut jit_state = PyreJitState {
        frame: frame as *const PyFrame as usize,
        resume_pc: None,
    };
    assert!(
        jit_state.sync_from_virtualizable(virtualizable_info),
        "build_jit_state: frame must be a valid PyFrame with readable fields"
    );
    jit_state
}

/// Extract (field_offset, field_size, field_type_code) from a live InteriorFieldDescr.
/// field_type_code: 0=ref, 1=int, 2=float — matches pyre_object::setinteriorfield API.
fn extract_interior_field_info(descr: &majit_ir::DescrRef) -> (usize, usize, u8) {
    if let Some(ifd) = descr.as_interior_field_descr() {
        let fld = ifd.field_descr();
        let ft = if fld.is_pointer_field() {
            0u8
        } else if fld.is_float_field() {
            2u8
        } else {
            1u8
        };
        (fld.offset(), fld.field_size(), ft)
    } else {
        (0, 8, 1)
    }
}

/// resume.py:1437-1541 — BlackholeAllocator for pyre's object model.
///
/// Used by ResumeDataDirectReader during guard failure blackhole resume
/// to allocate virtual objects and replay pending field writes.
/// RPython delegates to self.cpu (metainterp_sd.cpu) for allocation.
pub(crate) struct PyreBlackholeAllocator;

impl majit_metainterp::resume::BlackholeAllocator for PyreBlackholeAllocator {
    fn allocate_struct(&self, typedescr: &majit_ir::DescrRef) -> i64 {
        // resume.py:1441-1442 allocate_struct → cpu.bh_new(typedescr)
        // llmodel.py:775-776 bh_new(sizedescr): plain malloc, no vtable.
        let sd = typedescr
            .as_size_descr()
            .expect("allocate_struct: not a SizeDescr");
        let bh_descr = majit_translate::jitcode::BhDescr::Size {
            size: sd.size(),
            type_id: sd.type_id(),
            vtable: 0,
            owner: String::new(),
            all_fielddescrs: majit_translate::jitcode::bh_field_specs_from_size_descr(sd),
        };
        let (driver, _) = driver_pair();
        driver.meta_interp().backend().bh_new(&bh_descr)
    }

    fn allocate_with_vtable(&self, descr: &majit_ir::DescrRef, vtable: usize) -> i64 {
        // resume.py:1437-1439 allocate_with_vtable →
        //   exec_new_with_vtable(self.cpu, descr)
        // llmodel.py:778-782 bh_new_with_vtable: allocate AND set vtable.
        use pyre_jit_trace::descr::{W_FLOAT_GC_TYPE_ID, W_INT_GC_TYPE_ID};
        let sd = descr
            .as_size_descr()
            .expect("allocate_with_vtable: not a SizeDescr");
        let descr_index = sd.type_id();
        let descr_size = sd.size();
        match descr_index {
            W_INT_GC_TYPE_ID => {
                let obj = Box::new(pyre_object::intobject::W_IntObject {
                    ob_header: pyre_object::pyobject::PyObject {
                        ob_type: &pyre_object::pyobject::INT_TYPE as *const _,
                        w_class: pyre_object::pyobject::get_instantiate(
                            &pyre_object::pyobject::INT_TYPE,
                        ),
                    },
                    intval: 0,
                });
                Box::into_raw(obj) as i64
            }
            W_FLOAT_GC_TYPE_ID => {
                let obj = Box::new(pyre_object::floatobject::W_FloatObject {
                    ob_header: pyre_object::pyobject::PyObject {
                        ob_type: &pyre_object::pyobject::FLOAT_TYPE as *const _,
                        w_class: pyre_object::pyobject::get_instantiate(
                            &pyre_object::pyobject::FLOAT_TYPE,
                        ),
                    },
                    floatval: 0.0,
                });
                Box::into_raw(obj) as i64
            }
            _ => {
                let bh_descr = majit_translate::jitcode::BhDescr::Size {
                    size: descr_size,
                    type_id: descr_index,
                    vtable,
                    owner: String::new(),
                    all_fielddescrs: majit_translate::jitcode::bh_field_specs_from_size_descr(sd),
                };
                let (driver, _) = driver_pair();
                driver.meta_interp().backend().bh_new_with_vtable(&bh_descr)
            }
        }
    }

    fn setfield_typed(
        &self,
        struct_ptr: i64,
        value: i64,
        _descr: u32,
        field_offset: usize,
        field_size: usize,
    ) {
        // resume.py:1509-1528 setfield — write field at byte offset.
        // field_offset > 0: offset 0 is the ob_type header set by
        // allocate_struct/allocate_with_vtable; never let resume data
        // overwrite it.
        if struct_ptr != 0 && field_offset > 0 {
            unsafe {
                let ptr = (struct_ptr as *mut u8).add(field_offset);
                match field_size {
                    8 => (ptr as *mut i64).write(value),
                    4 => (ptr as *mut i32).write(value as i32),
                    2 => (ptr as *mut i16).write(value as i16),
                    1 => ptr.write(value as u8),
                    _ => (ptr as *mut i64).write(value),
                }
            }
        }
    }

    fn setarrayitem_typed(&self, array: i64, index: usize, value: i64, _descr: u32) {
        // resume.py:1009-1015 setarrayitem dispatch by type
        if array != 0 {
            // pyre list items are PyObjectRef (pointer-sized)
            let item_size = std::mem::size_of::<usize>();
            unsafe {
                let base = array as *mut u8;
                let ptr = base.add(index * item_size) as *mut i64;
                ptr.write(value);
            }
        }
    }

    // resume.py:1520-1529: setinteriorfield dispatch by descr
    // llmodel.py:648-665: bh_setinteriorfield_gc_{i,r,f}
    fn setinteriorfield_gc_i(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        if array != 0 {
            let (fo, fs, ft) = extract_interior_field_info(descr);
            let is = descr
                .as_interior_field_descr()
                .map(|ifd| ifd.array_descr())
                .map(|ad| ad.item_size())
                .unwrap_or(fo + fs);
            pyre_object::setinteriorfield(array as *mut _, index, fo, fs, is, ft, value);
        }
    }

    fn setinteriorfield_gc_r(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        self.setinteriorfield_gc_i(array, index, value, descr);
    }

    fn setinteriorfield_gc_f(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        self.setinteriorfield_gc_i(array, index, value, descr);
    }

    /// resume.py:1452-1456 allocate_raw_buffer(func, size)
    /// Concrete reader: cpu.bh_call_i(func, [size], None, None, calldescr)
    fn allocate_raw_buffer(&self, func: i64, size: usize) -> i64 {
        let (driver, _) = driver_pair();
        let calldescr = majit_translate::jitcode::BhCallDescr::from_arg_classes(
            "i".into(),
            'i',
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        driver
            .meta_interp()
            .backend()
            .bh_call_i(func, Some(&[size as i64]), None, None, &calldescr)
    }

    /// resume.py:1543-1550 setrawbuffer_item
    /// Concrete reader: descr-driven dispatch to cpu.bh_raw_store_f/i
    fn setrawbuffer_item(
        &self,
        buffer: i64,
        offset: usize,
        value: i64,
        descr: &majit_ir::ArrayDescrInfo,
    ) {
        let bh_descr = majit_translate::jitcode::BhDescr::from_array_descr_info(descr);
        // resume.py:1544: assert not descr.is_array_of_pointers()
        assert!(
            !bh_descr.is_array_of_pointers(),
            "raw buffer entry must not be pointer type"
        );
        let (driver, _) = driver_pair();
        let backend = driver.meta_interp().backend();
        if descr.item_type == 2 {
            // resume.py:1545-1547: descr.is_array_of_floats()
            //   newvalue = self.decode_float(fieldnum)
            //   self.cpu.bh_raw_store_f(buffer, offset, newvalue, descr)
            backend.bh_raw_store_f(
                buffer,
                offset as i64,
                f64::from_bits(value as u64),
                &bh_descr,
            );
        } else {
            // resume.py:1548-1550: else (int)
            //   newvalue = self.decode_int(fieldnum)
            //   self.cpu.bh_raw_store_i(buffer, offset, newvalue, descr)
            backend.bh_raw_store_i(buffer, offset as i64, value, &bh_descr);
        }
    }

    fn box_int(&self, value: i64) -> i64 {
        pyre_object::intobject::w_int_new(value) as i64
    }

    fn box_float(&self, bits: i64) -> i64 {
        pyre_object::floatobject::w_float_new(f64::from_bits(bits as u64)) as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestJitParamsGuard;

    impl TestJitParamsGuard {
        fn low_threshold() -> Self {
            let (driver, _) = driver_pair();
            driver
                .meta_interp_mut()
                .warm_state_mut()
                .set_default_params();
            driver.set_param("threshold", 2);
            driver.set_param("function_threshold", 2);
            Self
        }
    }

    impl Drop for TestJitParamsGuard {
        fn drop(&mut self) {
            let (driver, _) = driver_pair();
            driver
                .meta_interp_mut()
                .warm_state_mut()
                .set_default_params();
            driver.set_param("threshold", JIT_THRESHOLD as i64);
        }
    }

    fn function_code_from_module(
        module: &pyre_interpreter::CodeObject,
        name: &str,
    ) -> pyre_interpreter::CodeObject {
        use pyre_interpreter::ConstantData;

        module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } if code.obj_name.as_str() == name => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("test source should contain function code {name}"))
    }

    fn ensure_test_jit_callbacks() {
        super::init_callbacks();
    }

    /// Drive the codewriter `register_portal_jitdriver` setup path so
    /// `pyre_jit_trace::state::ensure_jitcode_ptr` can resolve the
    /// installed entry. Mirrors RPython warmspot.py:281-282 — the
    /// trace-side staticdata is populated only by the make_jitcodes
    /// drain.
    ///
    /// The portal must be registered using the canonical `CodeObject*`
    /// that backs `w_code` (the inner pointer obtained by
    /// `w_code_get_ptr`), not an arbitrary copy of the same source —
    /// `CallControl.jitcodes` is keyed by raw pointer identity.
    fn register_test_portal(_unused: &pyre_interpreter::CodeObject, w_code: *const ()) {
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject
        };
        let canonical_code = unsafe { &*raw_code };
        crate::jit::codewriter::register_portal_jitdriver(canonical_code, w_code, None);
    }

    /// Translate Python-stack depths into the post-regalloc Ref-bank
    /// colors the dispatcher would actually touch. Phase 2.1c removed the
    /// `register_color = nlocals + depth` identity (chordal coloring may
    /// coalesce disjointly-live slots), so callers must consult
    /// `metadata.stack_slot_color_map` before checking liveness.
    fn stack_slot_colors_for_depths(jitcode_index: i32, depths: &[usize]) -> Vec<u32> {
        let map = pyre_jit_trace::state::stack_slot_color_map_at(jitcode_index);
        depths
            .iter()
            .map(|&d| {
                u32::from(*map.get(d).unwrap_or_else(|| {
                    panic!(
                        "stack_slot_color_map for jitcode_index={jitcode_index} \
                         lacks entry for depth {d}; got {map:?}"
                    )
                }))
            })
            .collect()
    }

    fn live_pc_containing_all(
        jitcode_index: i32,
        code: &pyre_interpreter::CodeObject,
        regs: &[u32],
    ) -> (usize, Vec<u32>) {
        let live_by_pc: Vec<(usize, Vec<u32>)> = (0..code.instructions.len())
            .map(|pc| {
                let live =
                    pyre_jit_trace::state::frame_liveness_reg_indices_at(jitcode_index, pc as i32);
                (pc, live)
            })
            .collect();
        live_by_pc
            .iter()
            .find_map(|(pc, live)| {
                regs.iter()
                    .all(|reg| live.contains(reg))
                    .then_some((*pc, live.clone()))
            })
            .unwrap_or_else(|| {
                panic!(
                    "compiled liveness should expose regs {regs:?}; got {:?}",
                    live_by_pc
                )
            })
    }

    fn compiled_trace_fixture(
        source: &str,
        function_name: &str,
        live_locals: &[u32],
        live_stack_depths: &[usize],
        init: impl FnOnce(&mut PyFrame),
    ) -> (Box<PyFrame>, *const (), usize) {
        use pyre_interpreter::compile_exec;
        use pyre_jit_trace::state as trace_state;

        ensure_test_jit_callbacks();
        let module = compile_exec(source).expect("test code should compile");
        let code = function_code_from_module(&module, function_name);
        let mut frame = Box::new(PyFrame::new(code.clone()));
        init(&mut frame);
        frame.fix_array_ptrs();

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        // Phase 2.1c removed `register_color = nlocals + depth` identity:
        // stack-slot colors come from `stack_slot_color_map_at`. Locals
        // remain at colors `0..nlocals` (input-arg pinning kept that
        // half).
        let mut live_regs: Vec<u32> = live_locals.to_vec();
        if !live_stack_depths.is_empty() {
            live_regs.extend_from_slice(&stack_slot_colors_for_depths(
                jitcode_index,
                live_stack_depths,
            ));
        }
        let (resume_pc, _) = live_pc_containing_all(jitcode_index, &code, &live_regs);
        (frame, jitcode_ptr, resume_pc)
    }

    /// Stack-depth-based variant of `compiled_trace_fixture`. Locates the
    /// first Python PC where the bytecode-level forward stack analysis
    /// reports `target_depth`, independent of which Ref-bank colors land
    /// in the encoded `-live-` set. Stable across Task #158 force-add
    /// removal: the codewriter `live_r` is now SSA-driven only, so
    /// stack-slot colors no longer always appear there even when those
    /// slots are runtime-live (the consume_one_section heap-read
    /// fallback fills them at decode time). Tests that need a PC with
    /// a specific symbolic stack depth should use this helper.
    fn compiled_trace_fixture_at_depth(
        source: &str,
        function_name: &str,
        target_depth: u16,
        init: impl FnOnce(&mut PyFrame),
    ) -> (Box<PyFrame>, *const (), usize) {
        use pyre_interpreter::compile_exec;
        use pyre_jit_trace::state as trace_state;

        ensure_test_jit_callbacks();
        let module = compile_exec(source).expect("test code should compile");
        let code = function_code_from_module(&module, function_name);
        let mut frame = Box::new(PyFrame::new(code.clone()));
        init(&mut frame);
        frame.fix_array_ptrs();

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let _ = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let depth_table =
            pyre_jit_trace::liveness::liveness_for(&code as *const _).depth_at_py_pc();
        let resume_pc = depth_table
            .iter()
            .position(|&d| d == target_depth)
            .unwrap_or_else(|| {
                panic!(
                    "test source should reach stack depth {target_depth}; depth_table={depth_table:?}"
                )
            });
        (frame, jitcode_ptr, resume_pc)
    }

    fn single_local_test_state(
        ctx: &mut majit_metainterp::TraceCtx,
        frame: &PyFrame,
        frame_ptr: usize,
        jitcode_ptr: *const (),
        resume_pc: usize,
        local_type: majit_ir::Type,
        local: majit_ir::OpRef,
    ) -> pyre_jit_trace::state::TestSymState {
        use pyre_jit_trace::state as trace_state;

        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(ctx, frame_ref);
        pyre_jit_trace::state::TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 1,
            valuestackdepth: 1,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![local_type],
            symbolic_stack_types: vec![],
            registers_r: vec![local],
            concrete_stack: vec![],
            concrete_namespace: frame.w_globals,
            vable_last_instr: ctx.const_int(resume_pc as i64 - 1),
            vable_pycode: ctx.const_ref(frame.pycode as usize as i64),
            vable_valuestackdepth: ctx.const_int(1),
            vable_debugdata: ctx.const_ref(frame.debugdata as usize as i64),
            vable_lastblock: ctx.const_ref(frame.lastblock as usize as i64),
            vable_w_globals: ctx.const_ref(frame.w_globals as usize as i64),
        }
    }

    #[test]
    fn test_restore_guard_failure_uses_runtime_value_kinds_with_compiled_trace_jitcode() {
        use majit_ir::{GcRef, Type, Value};
        use majit_metainterp::JitState;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_interpreter::{ConstantData, compile_exec};
        use pyre_jit_trace::state::{self as trace_state, PyreJitState, PyreMeta};
        use pyre_object::pyobject::is_int;
        use pyre_object::{w_int_get_value, w_int_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(a, b, c):\n    i = 0\n    return i\nf(1, 2, 3)\n")
            .expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let mut frame = Box::new(PyFrame::new(code.clone()));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let resume_pc = (0..code.instructions.len())
            .find(|&pc| {
                trace_state::frame_liveness_reg_indices_at(jitcode_index, pc as i32).contains(&3)
            })
            .expect("compiled liveness should expose local i at some Python PC");
        let live_regs = trace_state::frame_liveness_reg_indices_at(jitcode_index, resume_pc as i32);
        assert!(
            live_regs.contains(&3),
            "selected resume pc must decode the raw-int local slot"
        );
        assert_eq!(
            trace_state::frame_value_count_at(jitcode_index, resume_pc as i32),
            live_regs.len(),
            "frame-value count must come from the same compiled jitcode liveness block"
        );

        let mut state = PyreJitState {
            frame: frame_ptr,
            resume_pc: Some(resume_pc),
        };
        state.set_next_instr(0);
        state.set_valuestackdepth(4);
        let meta = PyreMeta {
            num_locals: 4,
            ns_len: 0,
            valuestackdepth: 4,
            array_capacity: 4,
            trace_extra_reds: 1,
            has_virtualizable: true,
            // Trace-entry slot types can be stale; guard failure must still
            // respect the runtime Value tags recovered from resume data.
            slot_types: vec![Type::Ref, Type::Ref, Type::Ref, Type::Ref],
        };

        let ec_value = unsafe { (*(frame_ptr as *const PyFrame)).execution_context as usize };
        let mut values = vec![
            Value::Ref(GcRef(frame_ptr)),                // frame
            Value::Ref(GcRef(ec_value)),                 // ec extra red
            Value::Int(8),                               // last_instr
            Value::Ref(GcRef(frame.pycode as usize)),    // pycode
            Value::Int(4),                               // valuestackdepth
            Value::Ref(GcRef(0)),                        // debugdata
            Value::Ref(GcRef(0)),                        // lastblock
            Value::Ref(GcRef(frame.w_globals as usize)), // w_globals
        ];
        for reg in live_regs.iter() {
            match reg {
                0 => values.push(Value::Ref(GcRef(w_int_new(1) as usize))), // local a
                1 => values.push(Value::Ref(GcRef(w_int_new(2) as usize))), // local b
                2 => values.push(Value::Ref(GcRef(w_int_new(3) as usize))), // local c
                3 => values.push(Value::Int(7)),                            // local i
                // pypy/module/pypyjit/interp_jit.py:68 reds = ['frame',
                // 'ec'] — portal red args ride the live_r mask after
                // codewriter commit 9f0d0f6c18 (interp_jit.py parity).
                // The two trailing live regs are portal_frame_reg /
                // portal_ec_reg holding the runtime frame_ptr and ec.
                _ if Some(reg) == live_regs.iter().rev().nth(1) => {
                    values.push(Value::Ref(GcRef(frame_ptr)));
                }
                _ if Some(reg) == live_regs.iter().rev().next() => {
                    values.push(Value::Ref(GcRef(ec_value)));
                }
                other => panic!("unexpected live reg {other} at resume pc {resume_pc}"),
            }
        }

        assert!(<PyreJitState as JitState>::restore_guard_failure_values(
            &mut state,
            &meta,
            &values,
            &majit_metainterp::blackhole::ExceptionState::default(),
        ));

        assert_eq!(state.next_instr(), 9);
        assert_eq!(state.valuestackdepth(), 4);
        let restored_i = state.local_at(3).expect("local i should be restored");
        assert!(unsafe { is_int(restored_i) });
        assert_eq!(unsafe { w_int_get_value(restored_i) }, 7);
    }

    #[test]
    fn test_current_fail_args_flushes_header_with_compiled_trace_jitcode() {
        use majit_ir::{OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::{FrameBlock, PyFrame};
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(x):\n    i = 7\n    return x[i - 7]\nf([1])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = Box::new(PyFrame::new(code.clone()));
        frame.locals_w_mut()[0] = w_list_new(vec![w_int_new(11)]);
        frame.locals_w_mut()[1] = w_int_new(7);
        frame.locals_w_mut()[2] = w_list_new(vec![w_int_new(21)]);
        frame.locals_w_mut()[3] = w_int_new(5);
        frame.valuestackdepth = 4;
        let _ = frame.getorcreatedebug(123);
        frame.append_block(FrameBlock {
            valuestackdepth: 0,
            handlerposition: 55,
            previous: std::ptr::null_mut(),
        });
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let (resume_pc, live_regs) = live_pc_containing_all(
            jitcode_index,
            &code,
            &stack_slot_colors_for_depths(jitcode_index, &[0, 1]),
        );

        let mut ctx = TraceCtx::for_test(2);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let stack0 = ctx.const_ref(0xb0);
        let stack1 = ctx.const_ref(0xb1);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 2,
            valuestackdepth: 4,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref, Type::Int],
            symbolic_stack_types: vec![Type::Ref, Type::Int],
            registers_r: vec![OpRef::NONE, OpRef::NONE, stack0, stack1],
            concrete_stack: vec![],
            concrete_namespace: frame.w_globals,
            vable_last_instr: ctx.const_int(999),
            vable_pycode: ctx.const_ref(0xdead),
            vable_valuestackdepth: ctx.const_int(111),
            vable_debugdata: ctx.const_ref(0xbeef),
            vable_lastblock: ctx.const_ref(0xcafe),
            vable_w_globals: ctx.const_ref(0xfeed),
        });
        let ec_ref = ctx.const_ref(frame.execution_context as usize as i64);
        sym.set_test_execution_context(ec_ref);
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

        let fail_args = state.capture_current_fail_args();

        assert_eq!(
            fail_args.len(),
            pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS + live_regs.len(),
        );
        assert_eq!(fail_args[0], frame_ref);
        assert_eq!(fail_args[1], ec_ref);
        // last_instr / valuestackdepth are guard-time-overridden by
        // flush_to_frame_for_guard (orgpc - 1, pre-opcode depth).
        assert_eq!(fail_args[2], ctx.const_int(resume_pc as i64 - 1));
        assert_eq!(fail_args[4], ctx.const_int(4));
        // pycode / debugdata / lastblock / w_globals are JIT-scope
        // invariant under CPython 3.14 bytecode (`lastblock` is mutated
        // only by SETUP_*/POP_BLOCK paths the tracer never enters) and
        // stay bound to the trace-start inputarg OpRefs the fixture
        // seeded above.
        assert_eq!(fail_args[3], ctx.const_ref(0xdead));
        assert_eq!(fail_args[5], ctx.const_ref(0xbeef));
        assert_eq!(fail_args[6], ctx.const_ref(0xcafe));
        assert_eq!(fail_args[7], ctx.const_ref(0xfeed));
    }

    #[test]
    fn test_current_fail_args_materializes_symbolic_holes_with_compiled_trace_jitcode() {
        use majit_ir::{OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(x):\n    i = 7\n    return x[i - 7]\nf([1])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = Box::new(PyFrame::new(code.clone()));
        frame.locals_w_mut()[0] = w_list_new(vec![w_int_new(11)]);
        frame.locals_w_mut()[1] = w_int_new(7);
        frame.locals_w_mut()[2] = w_list_new(vec![w_int_new(21)]);
        frame.locals_w_mut()[3] = w_int_new(5);
        frame.valuestackdepth = 4;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let stack_colors = stack_slot_colors_for_depths(jitcode_index, &[0, 1]);
        let (resume_pc, live_regs) = live_pc_containing_all(jitcode_index, &code, &stack_colors);

        let mut ctx = TraceCtx::for_test(2);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let stack0 = ctx.const_ref(0xb0);
        let stack1 = ctx.const_ref(0xb1);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 2,
            valuestackdepth: 4,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref, Type::Int],
            symbolic_stack_types: vec![Type::Ref, Type::Int],
            registers_r: vec![OpRef::NONE, OpRef::NONE, stack0, stack1],
            concrete_stack: vec![],
            concrete_namespace: frame.w_globals,
            vable_last_instr: ctx.const_int(0),
            vable_pycode: ctx.const_ref(0),
            vable_valuestackdepth: ctx.const_int(0),
            vable_debugdata: ctx.const_ref(0),
            vable_lastblock: ctx.const_ref(0),
            vable_w_globals: ctx.const_ref(0),
        });
        let ec_ref = ctx.const_ref(frame.execution_context as usize as i64);
        sym.set_test_execution_context(ec_ref);
        trace_state::seed_compiled_trace_jitcode_test_state(
            &mut sym,
            &mut ctx,
            jitcode_index,
            resume_pc as i32,
            &[(0, stack0), (1, stack1)],
        );
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

        let fail_args = state.capture_current_fail_args();

        for &color in &stack_colors {
            assert!(live_regs.contains(&color));
        }
        assert_eq!(
            fail_args.len(),
            pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS + live_regs.len(),
        );
        assert_eq!(fail_args[0], frame_ref);
        assert!(
            fail_args.iter().all(|arg| !arg.is_none()),
            "materialized fail args should not contain OpRef::NONE holes"
        );
        // `registers_r` is keyed by semantic slot index (locals 0..nlocals,
        // stack nlocals..nlocals+stack_only) — `load_local_value` writes
        // there.  Phase 2.1c only changed the post-rename color landscape,
        // not the semantic indexing.
        assert!(
            state.symbolic_registers_r()[2..4]
                .iter()
                .all(|opref| !opref.is_none()),
            "live stack slots should be materialized into the symbolic register file"
        );
        for (depth, &color) in stack_colors.iter().enumerate() {
            let color = color as usize;
            if color < 2 {
                let stack_value = [stack0, stack1][depth];
                assert_ne!(
                    state.symbolic_registers_r()[color],
                    stack_value,
                    "Ref-bank color {} for stack depth {} must not overwrite semantic local slot {}",
                    color,
                    depth,
                    color
                );
            }
        }
    }

    #[test]
    fn test_load_local_checked_value_respects_symbolic_local_type_with_compiled_trace_jitcode() {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_interpreter::{LocalOpcodeHandler, compile_exec};
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module =
            compile_exec("def f(b):\n    return b\nf(1)\n").expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = Box::new(PyFrame::new(code.clone()));
        frame.locals_w_mut()[0] = w_list_new(vec![w_int_new(11)]);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let (resume_pc, _) = live_pc_containing_all(jitcode_index, &code, &[0]);

        let run_case = |symbolic_type: Type, name: &str, expected_guard: Option<OpCode>| {
            let mut ctx = TraceCtx::for_test_types(&[symbolic_type]);
            let local = OpRef(0);
            let frame_ref = ctx.const_ref(frame_ptr as i64);
            let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
            let mut sym = PyreSym::from_test_state(TestSymState {
                frame: frame_ref,
                jitcode: jitcode_ptr,
                nlocals: 1,
                valuestackdepth: 1,
                locals_cells_stack_array_ref: locals_array,
                symbolic_local_types: vec![symbolic_type],
                symbolic_stack_types: vec![],
                registers_r: vec![local],
                concrete_stack: vec![],
                concrete_namespace: frame.w_globals,
                vable_last_instr: ctx.const_int(resume_pc as i64 - 1),
                vable_pycode: ctx.const_ref(frame.pycode as usize as i64),
                vable_valuestackdepth: ctx.const_int(1),
                vable_debugdata: ctx.const_ref(frame.debugdata as usize as i64),
                vable_lastblock: ctx.const_ref(frame.lastblock as usize as i64),
                vable_w_globals: ctx.const_ref(frame.w_globals as usize as i64),
            });
            let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

            let loaded =
                <MIFrame as LocalOpcodeHandler>::load_local_checked_value(&mut state, 0, name)
                    .expect("local should load");
            assert_eq!(loaded.opref, local);

            let recorder = ctx.into_recorder();
            match expected_guard {
                Some(opcode) => {
                    assert!(
                        recorder.ops().iter().any(|op| op.opcode == opcode),
                        "expected guard opcode {opcode:?} in {:?}",
                        recorder.ops()
                    );
                }
                None => assert_eq!(recorder.num_guards(), 0),
            }
        };

        run_case(Type::Int, "j", None);
        run_case(Type::Ref, "b", Some(OpCode::GuardNonnull));
    }

    #[test]
    fn test_guard_class_uses_guard_nonnull_class_with_compiled_trace_jitcode() {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{INT_TYPE, w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(x):\n    i = 7\n    return x[i - 7]\nf([1])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = Box::new(PyFrame::new(code.clone()));
        frame.locals_w_mut()[0] = w_list_new(vec![w_int_new(11)]);
        frame.locals_w_mut()[1] = w_int_new(7);
        frame.locals_w_mut()[2] = w_list_new(vec![w_int_new(21)]);
        frame.locals_w_mut()[3] = w_int_new(5);
        frame.valuestackdepth = 4;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let (resume_pc, _) = live_pc_containing_all(
            jitcode_index,
            &code,
            &stack_slot_colors_for_depths(jitcode_index, &[0, 1]),
        );

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let obj = OpRef(0);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 2,
            valuestackdepth: 4,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref, Type::Int],
            symbolic_stack_types: vec![Type::Ref, Type::Int],
            registers_r: vec![OpRef::NONE; 4],
            concrete_stack: vec![],
            concrete_namespace: frame.w_globals,
            vable_last_instr: ctx.const_int(0),
            vable_pycode: ctx.const_ref(0),
            vable_valuestackdepth: ctx.const_int(0),
            vable_debugdata: ctx.const_ref(0),
            vable_lastblock: ctx.const_ref(0),
            vable_w_globals: ctx.const_ref(0),
        });
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

        state.capture_guard_class(obj, &INT_TYPE as *const _);

        let recorder = ctx.into_recorder();
        let op = recorder.ops().last().expect("guard op should be present");
        assert_eq!(op.opcode, OpCode::GuardNonnullClass);
        assert_eq!(op.args[0], obj);
    }

    #[test]
    fn test_trace_guarded_int_payload_uses_guard_nonnull_class_and_pure_payload_with_compiled_trace_jitcode()
     {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(x):\n    i = 7\n    return x[i - 7]\nf([1])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = Box::new(PyFrame::new(code.clone()));
        frame.locals_w_mut()[0] = w_list_new(vec![w_int_new(11)]);
        frame.locals_w_mut()[1] = w_int_new(7);
        frame.locals_w_mut()[2] = w_list_new(vec![w_int_new(21)]);
        frame.locals_w_mut()[3] = w_int_new(5);
        frame.valuestackdepth = 4;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let (resume_pc, _) = live_pc_containing_all(
            jitcode_index,
            &code,
            &stack_slot_colors_for_depths(jitcode_index, &[0, 1]),
        );

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let int_obj = OpRef(0);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 2,
            valuestackdepth: 4,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref, Type::Int],
            symbolic_stack_types: vec![Type::Ref, Type::Int],
            registers_r: vec![OpRef::NONE; 4],
            concrete_stack: vec![],
            concrete_namespace: frame.w_globals,
            vable_last_instr: ctx.const_int(0),
            vable_pycode: ctx.const_ref(0),
            vable_valuestackdepth: ctx.const_int(0),
            vable_debugdata: ctx.const_ref(0),
            vable_lastblock: ctx.const_ref(0),
            vable_w_globals: ctx.const_ref(0),
        });
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

        let _ = state.capture_trace_guarded_int_payload(int_obj);

        let recorder = ctx.into_recorder();
        let mut saw_guard_nonnull_class = false;
        let mut saw_pure_payload = false;
        let recorded_ops: Vec<(OpCode, Vec<OpRef>)> = recorder
            .ops()
            .iter()
            .map(|op| (op.opcode, op.args.to_vec()))
            .collect();
        for op in recorder.ops() {
            if op.opcode == OpCode::GuardNonnullClass {
                saw_guard_nonnull_class = true;
            }
            if op.opcode == OpCode::GetfieldGcPureI && op.args.as_slice() == &[int_obj] {
                saw_pure_payload = true;
            }
        }
        assert!(
            saw_guard_nonnull_class,
            "int payload fast path should guard object class via GuardNonnullClass: {:?}",
            recorded_ops
        );
        assert!(
            saw_pure_payload,
            "int payload fast path should read the immutable payload with GetfieldGcPureI: {:?}",
            recorded_ops
        );
    }

    #[test]
    fn test_branch_guard_preserves_pre_pop_stack_shape_with_compiled_trace_jitcode() {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use majit_metainterp::recorder::SnapshotTagged;
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(x):\n    i = 7\n    return x[i - 7]\nf([1])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = Box::new(PyFrame::new(code.clone()));
        frame.locals_w_mut()[0] = w_list_new(vec![w_int_new(11)]);
        frame.locals_w_mut()[1] = w_int_new(7);
        frame.locals_w_mut()[2] = w_list_new(vec![w_int_new(21)]);
        frame.locals_w_mut()[3] = w_int_new(1);
        frame.valuestackdepth = 4;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let (resume_pc, live_regs) = live_pc_containing_all(
            jitcode_index,
            &code,
            &stack_slot_colors_for_depths(jitcode_index, &[0, 1]),
        );

        let run_case = |record_branch_guard: bool| {
            let mut ctx = TraceCtx::for_test_types(&[Type::Ref, Type::Int]);
            let lower_stack = OpRef(0);
            let truth = OpRef(1);
            let frame_ref = ctx.const_ref(frame_ptr as i64);
            let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
            let mut sym = PyreSym::from_test_state(TestSymState {
                frame: frame_ref,
                jitcode: jitcode_ptr,
                nlocals: 2,
                valuestackdepth: 4,
                locals_cells_stack_array_ref: locals_array,
                symbolic_local_types: vec![Type::Ref, Type::Int],
                symbolic_stack_types: vec![Type::Ref, Type::Int],
                registers_r: vec![OpRef::NONE, OpRef::NONE, lower_stack, truth],
                concrete_stack: vec![],
                concrete_namespace: frame.w_globals,
                vable_last_instr: ctx.const_int(resume_pc as i64 - 1),
                vable_pycode: ctx.const_ref(frame.pycode as usize as i64),
                vable_valuestackdepth: ctx.const_int(4),
                vable_debugdata: ctx.const_ref(frame.debugdata as usize as i64),
                vable_lastblock: ctx.const_ref(frame.lastblock as usize as i64),
                vable_w_globals: ctx.const_ref(frame.w_globals as usize as i64),
            });
            trace_state::seed_compiled_trace_jitcode_test_state(
                &mut sym,
                &mut ctx,
                jitcode_index,
                resume_pc as i32,
                &[(0, lower_stack), (1, truth)],
            );
            let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);
            if record_branch_guard {
                state.capture_record_branch_guard(OpRef::NONE, truth, true, resume_pc);
            } else {
                state.capture_generate_guard(OpCode::GuardTrue, &[truth]);
            }

            // Task #91 slice 4a: production guard recording goes through
            // `record_guard_typed` + `capture_resumedata` —
            // `op.fail_args` stays None until the optimizer's
            // `store_final_boxes_in_guard` writes it back from the
            // snapshot.  Inspect the snapshot directly (the canonical
            // RPython resume oracle) instead of the raw recorder buffer.
            //
            // Snapshot layout (opencoder.py:806 / build_framestack_snapshot):
            //  - `vable_boxes` = full virtualizable image
            //    `[frame_ptr, scalar_fields..., array_items...]`
            //    (NUM_SCALAR_INPUTARGS scalars + locals/stack array slots).
            //  - `frames[0].boxes` = top frame's active boxes (one per
            //    live register at the resume PC).
            let guard = ctx
                .ops()
                .last()
                .expect("branch guard should be recorded")
                .clone();
            assert_eq!(guard.opcode, OpCode::GuardTrue);
            let snapshot_id = guard.rd_resume_position;
            assert!(
                snapshot_id >= 0,
                "branch guard must carry rd_resume_position pointing at its captured snapshot",
            );
            let snapshot = &ctx.snapshots()[snapshot_id as usize];
            let n = pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS;
            assert!(
                snapshot.vable_boxes.len() >= n,
                "vable_boxes must contain at least the scalar virtualizable header: {:?}",
                snapshot.vable_boxes
            );
            // vable_boxes[0] = frame_ptr — the encoded form of `frame_ref`,
            // which the test seeded as `ctx.const_ref(frame_ptr as i64)`.
            assert_eq!(
                snapshot.vable_boxes[0],
                SnapshotTagged::Const(frame_ptr as i64, Type::Ref)
            );
            let active_boxes = snapshot
                .frames
                .first()
                .map(|f| f.boxes.as_slice())
                .unwrap_or(&[]);
            assert_eq!(active_boxes.len(), live_regs.len());
            // Task #134 restored kind-segregated liveness emission
            // (Int regs first, then Ref); Phase 2.1c further removed
            // the `register_color = nlocals + depth` identity, so the
            // active_boxes order no longer reflects Python stack
            // depth. Verify both stack OpRefs are present without
            // asserting an order that the protocol no longer
            // guarantees.
            assert!(
                active_boxes.iter().any(|b| matches!(
                    b,
                    SnapshotTagged::Box(li, _) if *li == lower_stack.0
                )),
                "pre-pop snapshot must capture lower stack slot: {:?}",
                active_boxes
            );
            assert!(
                active_boxes.iter().any(|b| matches!(
                    b,
                    SnapshotTagged::Box(ti, _) if *ti == truth.0
                )),
                "pre-pop snapshot must capture truth slot: {:?}",
                active_boxes
            );
        };

        run_case(true);
        run_case(false);
    }

    #[test]
    fn test_branch_truth_uses_concrete_parameter_with_compiled_trace_jitcode() {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use majit_metainterp::recorder::SnapshotTagged;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_interpreter::{BranchOpcodeHandler, compile_exec};
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(x):\n    i = 7\n    return x[i - 7]\nf([1])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = Box::new(PyFrame::new(code.clone()));
        frame.locals_w_mut()[0] = w_list_new(vec![w_int_new(11)]);
        frame.locals_w_mut()[1] = w_int_new(7);
        frame.locals_w_mut()[2] = w_list_new(vec![w_int_new(21)]);
        frame.locals_w_mut()[3] = w_int_new(1);
        frame.valuestackdepth = 4;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let (resume_pc, live_regs) = live_pc_containing_all(
            jitcode_index,
            &code,
            &stack_slot_colors_for_depths(jitcode_index, &[0, 1]),
        );

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref, Type::Int]);
        let lower_stack = OpRef(0);
        let truth = OpRef(1);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 2,
            valuestackdepth: 4,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref, Type::Int],
            symbolic_stack_types: vec![Type::Ref, Type::Int],
            registers_r: vec![OpRef::NONE, OpRef::NONE, lower_stack, truth],
            concrete_stack: vec![],
            concrete_namespace: frame.w_globals,
            vable_last_instr: ctx.const_int(resume_pc as i64 - 1),
            vable_pycode: ctx.const_ref(frame.pycode as usize as i64),
            vable_valuestackdepth: ctx.const_int(4),
            vable_debugdata: ctx.const_ref(frame.debugdata as usize as i64),
            vable_lastblock: ctx.const_ref(frame.lastblock as usize as i64),
            vable_w_globals: ctx.const_ref(frame.w_globals as usize as i64),
        });
        trace_state::seed_compiled_trace_jitcode_test_state(
            &mut sym,
            &mut ctx,
            jitcode_index,
            resume_pc as i32,
            &[(0, lower_stack), (1, truth)],
        );
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

        state.capture_generate_guard(OpCode::GuardTrue, &[truth]);
        assert_eq!(
            state
                .capture_concrete_branch_truth_for_value(truth, w_int_new(1))
                .unwrap(),
            true
        );
        <MIFrame as BranchOpcodeHandler>::leave_branch_truth(&mut state).unwrap();

        // Task #91 slice 4a: snapshot is the resume-data oracle, not
        // `op.fail_args` (None until the optimizer's
        // `store_final_boxes_in_guard` writes it back).
        let guard = ctx
            .ops()
            .last()
            .expect("guard op should be present")
            .clone();
        assert_eq!(guard.opcode, OpCode::GuardTrue);
        let snapshot_id = guard.rd_resume_position;
        assert!(
            snapshot_id >= 0,
            "guard must carry rd_resume_position pointing at its captured snapshot",
        );
        let snapshot = &ctx.snapshots()[snapshot_id as usize];
        let n = pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        assert!(
            snapshot.vable_boxes.len() >= n,
            "vable_boxes must contain at least the scalar virtualizable header: {:?}",
            snapshot.vable_boxes
        );
        let active_boxes = snapshot
            .frames
            .first()
            .map(|f| f.boxes.as_slice())
            .unwrap_or(&[]);
        assert_eq!(active_boxes.len(), live_regs.len());
        // See note on the sibling test
        // `test_branch_guard_preserves_pre_pop_stack_shape_*`: kind
        // segregation (Task #134) + Phase 2.1c color reassignment mean
        // active_boxes order is dictated by liveness format, not by
        // stack depth.
        assert!(
            active_boxes.iter().any(|b| matches!(
                b,
                SnapshotTagged::Box(li, _) if *li == lower_stack.0
            )),
            "mixed-bank guard must capture lower stack slot: {:?}",
            active_boxes
        );
        assert!(
            active_boxes.iter().any(|b| matches!(
                b,
                SnapshotTagged::Box(ti, _) if *ti == truth.0
            )),
            "mixed-bank guard must capture truth slot: {:?}",
            active_boxes
        );
    }

    #[test]
    fn test_close_loop_args_at_target_pc_preserves_virtualizable_stack_with_compiled_trace_jitcode()
    {
        use majit_ir::Type;
        use majit_metainterp::TraceCtx;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::w_int_new;

        let _ = driver_pair();
        init_callbacks();
        // Symbolic state below has nlocals=1 + 2 stack slots, so the
        // target PC needs depth=2 (post-Task #158 force-add removal,
        // stack-slot colors no longer always appear in `live_r`, so a
        // depth-based locator is needed instead of `&[1, 2]` regs).
        let (frame, jitcode_ptr, target_pc) = compiled_trace_fixture_at_depth(
            "def f(x):\n    return (x, x)\nf(1)\n",
            "f",
            2,
            |frame| {
                frame.locals_w_mut()[0] = w_int_new(7);
            },
        );
        let frame_ptr = (&*frame) as *const PyFrame as usize;

        let mut ctx = TraceCtx::for_test(0);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let local0 = ctx.const_ref(w_int_new(11) as usize as i64);
        let stack0 = ctx.const_ref(w_int_new(22) as usize as i64);
        let stack1 = ctx.const_ref(w_int_new(33) as usize as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 1,
            valuestackdepth: 3,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref],
            symbolic_stack_types: vec![Type::Ref, Type::Ref],
            registers_r: vec![local0, stack0, stack1],
            concrete_stack: vec![],
            concrete_namespace: frame.w_globals,
            vable_last_instr: ctx.const_int(target_pc as i64 - 1),
            vable_pycode: ctx.const_ref(frame.pycode as usize as i64),
            vable_valuestackdepth: ctx.const_int(3),
            vable_debugdata: ctx.const_ref(frame.debugdata as usize as i64),
            vable_lastblock: ctx.const_ref(frame.lastblock as usize as i64),
            vable_w_globals: ctx.const_ref(frame.w_globals as usize as i64),
        });
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, target_pc, target_pc);

        let jump_args = state.capture_close_loop_args_at(Some(target_pc));

        assert_eq!(
            jump_args.len(),
            pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS + 3,
            "JUMP carries local and stack slots from the virtualizable array"
        );
        assert_eq!(state.symbolic_valuestackdepth(), 3);
        let nlocals = state.symbolic_nlocals();
        let stack_only = state.symbolic_valuestackdepth() - nlocals;
        assert_eq!(state.symbolic_registers_r().len(), nlocals + stack_only);
        assert!(
            state.symbolic_registers_r()[nlocals..]
                .iter()
                .all(|opref| !opref.is_none())
        );
    }

    #[test]
    fn test_trace_dynamic_list_index_typed_int_skips_object_unbox_with_compiled_trace_jitcode() {
        use majit_ir::{OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_jit_trace::state::{MIFrame, PyreSym};
        use pyre_object::w_int_new;

        let (frame, jitcode_ptr, resume_pc) =
            compiled_trace_fixture("def f(b):\n    return b\nf(1)\n", "f", &[0], &[], |frame| {
                frame.locals_w_mut()[0] = w_int_new(2);
            });
        let frame_ptr = (&*frame) as *const PyFrame as usize;

        let mut ctx = TraceCtx::for_test_types(&[Type::Int, Type::Int]);
        let key = OpRef(0);
        let len = OpRef(1);
        let mut sym = PyreSym::from_test_state(single_local_test_state(
            &mut ctx,
            &frame,
            frame_ptr,
            jitcode_ptr,
            resume_pc,
            Type::Int,
            key,
        ));
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

        let raw_index = state.capture_trace_dynamic_list_index(key, len, 2);
        assert_eq!(raw_index, key);

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_guards(), 2);
        assert!(
            recorder
                .ops()
                .iter()
                .all(|op| op.opcode != majit_ir::OpCode::GuardNonnullClass),
            "typed-int index should not guard object class for an unbox fast path: {:?}",
            recorder.ops()
        );
        assert!(
            recorder
                .ops()
                .iter()
                .all(|op| op.opcode != majit_ir::OpCode::GetfieldGcPureI),
            "typed-int index should not read boxed int payloads: {:?}",
            recorder.ops()
        );
    }

    #[test]
    fn test_direct_len_value_returns_typed_raw_len_for_integer_list_with_compiled_trace_jitcode() {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_jit_trace::state::{MIFrame, PyreSym};
        use pyre_object::{w_int_new, w_list_new};

        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(pyre_object::listobject::w_list_uses_int_storage(list));
        }
        let (frame, jitcode_ptr, resume_pc) = compiled_trace_fixture(
            "def f(x):\n    return len(x)\nf([1, 2, 3])\n",
            "f",
            &[0],
            &[],
            |frame| {
                frame.locals_w_mut()[0] = list;
            },
        );
        let frame_ptr = (&*frame) as *const PyFrame as usize;

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref, Type::Ref]);
        let value = OpRef(0);
        let callable = OpRef(1);
        let mut sym = PyreSym::from_test_state(single_local_test_state(
            &mut ctx,
            &frame,
            frame_ptr,
            jitcode_ptr,
            resume_pc,
            Type::Ref,
            value,
        ));
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

        let len = state
            .capture_direct_len_value(callable, value, list)
            .expect("integer-list len fast path should trace");
        assert_eq!(state.capture_value_type(len), Type::Int);

        let recorder = ctx.into_recorder();
        assert_ne!(
            recorder.ops().last().map(|op| op.opcode),
            Some(OpCode::CallI)
        );
        let mut saw_len_field = false;
        let mut saw_new = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            if op.opcode == OpCode::New {
                saw_new = true;
            }
            if op.opcode == OpCode::GetfieldGcI
                && op.descr.as_ref().map(|d| d.index())
                    == Some(pyre_jit_trace::descr::list_int_items_len_descr().index())
            {
                saw_len_field = true;
            }
        }
        assert!(saw_len_field);
        assert!(!saw_new);
    }

    #[test]
    fn test_trace_direct_float_list_getitem_uses_gc_field_loads_for_list_object_with_compiled_trace_jitcode()
     {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_jit_trace::state::{MIFrame, PyreSym};

        let float_list = pyre_object::w_list_new(vec![
            pyre_object::floatobject::w_float_new(1.5),
            pyre_object::floatobject::w_float_new(2.5),
            pyre_object::floatobject::w_float_new(3.5),
        ]);
        unsafe {
            assert!(pyre_object::listobject::w_list_uses_float_storage(
                float_list
            ));
        }
        let (frame, jitcode_ptr, resume_pc) = compiled_trace_fixture(
            "def f(x):\n    return x[2]\nf([1.5, 2.5, 3.5])\n",
            "f",
            &[0],
            &[],
            |frame| {
                frame.locals_w_mut()[0] = float_list;
            },
        );
        let frame_ptr = (&*frame) as *const PyFrame as usize;

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref, Type::Int]);
        let list = OpRef(0);
        let key = OpRef(1);
        let mut sym = PyreSym::from_test_state(single_local_test_state(
            &mut ctx,
            &frame,
            frame_ptr,
            jitcode_ptr,
            resume_pc,
            Type::Ref,
            list,
        ));
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

        let result = state.capture_generated_list_getitem_by_strategy(list, key, 2, 2);
        assert_eq!(state.capture_value_type(result), Type::Float);

        let recorder = ctx.into_recorder();
        let mut saw_gc_field = false;
        let mut saw_raw_field = false;
        let mut saw_raw_array = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            match op.opcode {
                OpCode::GetfieldGcI if op.args.first().copied() == Some(list) => {
                    saw_gc_field = true
                }
                OpCode::GetfieldRawI if op.args.first().copied() == Some(list) => {
                    saw_raw_field = true
                }
                OpCode::GetarrayitemRawF => saw_raw_array = true,
                _ => {}
            }
        }
        assert!(saw_gc_field);
        assert!(!saw_raw_field);
        assert!(saw_raw_array);
    }

    #[test]
    fn test_list_append_value_uses_raw_storage_fast_paths_with_compiled_trace_jitcode() {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_jit_trace::state::{MIFrame, PyreSym};

        let run_case = |concrete_list: pyre_object::PyObjectRef,
                        symbolic_value_type: Type,
                        concrete_value: pyre_object::PyObjectRef,
                        expected_len_descr_idx: u32,
                        expect_box_alloc: bool| {
            let (frame, jitcode_ptr, resume_pc) = compiled_trace_fixture(
                "def f(x):\n    return x\nf([1])\n",
                "f",
                &[0],
                &[],
                |frame| {
                    frame.locals_w_mut()[0] = concrete_list;
                },
            );
            let frame_ptr = (&*frame) as *const PyFrame as usize;

            let mut ctx = TraceCtx::for_test_types(&[Type::Ref, symbolic_value_type]);
            let list = OpRef(0);
            let value = OpRef(1);
            let mut sym = PyreSym::from_test_state(single_local_test_state(
                &mut ctx,
                &frame,
                frame_ptr,
                jitcode_ptr,
                resume_pc,
                Type::Ref,
                list,
            ));
            let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

            state
                .capture_list_append_value(list, value, concrete_list, concrete_value)
                .expect("raw-storage append fast path should trace");

            let recorder = ctx.into_recorder();
            let mut saw_raw_setitem = false;
            let mut saw_len_update = false;
            let mut saw_call = false;
            let mut saw_new = false;
            for pos in 2..(2 + recorder.num_ops() as u32) {
                let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                    continue;
                };
                if matches!(
                    op.opcode,
                    OpCode::CallI | OpCode::CallN | OpCode::CallR | OpCode::CallF
                ) {
                    saw_call = true;
                }
                if op.opcode == OpCode::New {
                    saw_new = true;
                }
                if op.opcode == OpCode::SetarrayitemRaw {
                    saw_raw_setitem = true;
                }
                if op.opcode == OpCode::SetfieldGc
                    && op.descr.as_ref().map(|d| d.index()) == Some(expected_len_descr_idx)
                {
                    saw_len_update = true;
                }
            }
            assert!(saw_raw_setitem);
            assert!(saw_len_update);
            assert!(!saw_call);
            assert_eq!(saw_new, expect_box_alloc);
        };

        let int_list =
            pyre_object::w_list_new(vec![pyre_object::w_int_new(1), pyre_object::w_int_new(2)]);
        unsafe {
            assert!(pyre_object::listobject::w_list_uses_int_storage(int_list));
            assert!(pyre_object::listobject::w_list_can_append_without_realloc(
                int_list
            ));
        }
        run_case(
            int_list,
            Type::Int,
            pyre_object::w_int_new(42),
            pyre_jit_trace::descr::list_int_items_len_descr().index(),
            false,
        );

        let float_list = pyre_object::w_list_new(vec![
            pyre_object::w_float_new(1.5),
            pyre_object::w_float_new(2.5),
        ]);
        unsafe {
            assert!(pyre_object::listobject::w_list_uses_float_storage(
                float_list
            ));
            assert!(pyre_object::listobject::w_list_can_append_without_realloc(
                float_list
            ));
        }
        run_case(
            float_list,
            Type::Float,
            pyre_object::w_float_new(3.14),
            pyre_jit_trace::descr::list_float_items_len_descr().index(),
            false,
        );
    }

    #[test]
    fn test_iter_next_value_for_range_iterator_uses_gc_fields_and_returns_raw_int_with_compiled_trace_jitcode()
     {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::IterOpcodeHandler;
        use pyre_jit_trace::state::{MIFrame, PyreSym};

        let range_iter = pyre_object::w_range_iter_new(0, 2, 1);
        let (frame, jitcode_ptr, resume_pc) = compiled_trace_fixture(
            "def f(it):\n    return it\nf(range(2))\n",
            "f",
            &[0],
            &[],
            |frame| {
                frame.locals_w_mut()[0] = range_iter;
            },
        );
        let frame_ptr = (&*frame) as *const PyFrame as usize;

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let iter = OpRef(0);
        let mut sym = PyreSym::from_test_state(single_local_test_state(
            &mut ctx,
            &frame,
            frame_ptr,
            jitcode_ptr,
            resume_pc,
            Type::Ref,
            iter,
        ));
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

        let next = state
            .capture_iter_next_value(iter, range_iter)
            .expect("range iterator fast path should trace");
        assert_eq!(state.capture_value_type(next.opref), Type::Int);
        <MIFrame as IterOpcodeHandler>::guard_optional_value(&mut state, next, true)
            .expect("typed range next should not need optional guard");

        let recorder = ctx.into_recorder();
        let mut saw_getfield_gc = false;
        let mut saw_setfield_gc = false;
        let mut saw_setfield_raw = false;
        let mut saw_getfield_raw = false;
        let mut saw_new = false;
        let mut saw_optional_guard = false;
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            match op.opcode {
                OpCode::GetfieldGcI if op.args.first().copied() == Some(iter) => {
                    saw_getfield_gc = true
                }
                OpCode::SetfieldGc if op.args.first().copied() == Some(iter) => {
                    saw_setfield_gc = true
                }
                OpCode::SetfieldRaw if op.args.first().copied() == Some(iter) => {
                    saw_setfield_raw = true
                }
                OpCode::GetfieldRawI if op.args.first().copied() == Some(iter) => {
                    saw_getfield_raw = true
                }
                OpCode::New => saw_new = true,
                OpCode::GuardNonnull | OpCode::GuardIsnull => saw_optional_guard = true,
                _ => {}
            }
        }
        assert!(saw_getfield_gc);
        assert!(saw_setfield_gc);
        assert!(!saw_setfield_raw);
        assert!(!saw_getfield_raw);
        assert!(!saw_new);
        assert!(!saw_optional_guard);
    }

    #[test]
    fn test_eval_simple_addition() {
        let source = "x = 1 + 2";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let x = *(*frame.w_globals).get("x").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(x), 3);
        }
    }

    #[test]
    fn test_eval_while_loop() {
        let _jit_params = TestJitParamsGuard::low_threshold();
        let source = "\
i = 0
s = 0
while i < 20:
    s = s + i
    i = i + 1";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = *(*frame.w_globals).get("s").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 190);
        }
    }

    #[test]
    fn test_eval_with_jit_redecodes_opargs_after_extended_arg_jumps() {
        let _jit_params = TestJitParamsGuard::low_threshold();
        let mut source = String::from(
            "\
i = 0
acc = 0
if i == 1:
",
        );
        for _ in 0..80 {
            source.push_str("    acc = acc + 1000\n");
        }
        source.push_str(
            "\
while i < 6:
    acc = acc + 1
    i = i + 1
r = acc",
        );
        let code = pyre_interpreter::compile_exec(&source).expect("compile failed");
        assert!(
            code.instructions.windows(2).any(|pair| {
                matches!(
                    pair[0].op,
                    pyre_interpreter::bytecode::Instruction::ExtendedArg
                ) && !matches!(
                    pair[1].op,
                    pyre_interpreter::bytecode::Instruction::ExtendedArg
                )
            }),
            "expected an instruction with an ExtendedArg prefix"
        );
        if std::env::var_os("MAJIT_DUMP_BYTECODE").is_some() {
            let mut state = pyre_interpreter::OpArgState::default();
            for (pc, unit) in code.instructions.iter().copied().enumerate() {
                let (instr, oparg) = state.get(unit);
                eprintln!("{pc:03}: {instr:?} oparg={oparg:?}");
            }
            for (pc, pair) in code.instructions.windows(2).enumerate() {
                if matches!(
                    pair[0].op,
                    pyre_interpreter::bytecode::Instruction::ExtendedArg
                ) && !matches!(
                    pair[1].op,
                    pyre_interpreter::bytecode::Instruction::ExtendedArg
                ) {
                    let target_pc = pc + 1;
                    eprintln!(
                        "decode[{target_pc}] = {:?}",
                        pyre_interpreter::decode_instruction_at(&code, target_pc)
                    );
                    break;
                }
            }
        }
        let mut frame = PyFrame::new(code);
        let result = eval_with_jit(&mut frame);
        if std::env::var_os("MAJIT_DUMP_BYTECODE").is_some() {
            let mut keys: Vec<_> = unsafe { (*frame.w_globals).keys().cloned().collect() };
            keys.sort();
            eprintln!("module result: {:?}", result);
            eprintln!("module namespace keys: {:?}", keys);
        }
        unsafe {
            let r = *(*frame.w_globals).get("r").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(r), 6);
        }
    }

    /// Regression test for the recursive portal Ref ABI.
    ///
    /// RPython portal return type is always REF (warmspot.py:449).
    /// The self-recursive call uses CALL_ASSEMBLER_R, FINISH records with
    /// done_with_this_frame_descr_ref, and the caller unboxes via
    /// GuardNonnullClass + GetfieldGcPureI (pyjitpl.py:3198-3220).
    ///
    /// A previous bug used CALL_ASSEMBLER_I + FINISH(Int) + forced unbox
    /// at the blackhole boundary, causing pointer-like-integer corruption
    /// in the recursive return path.
    #[test]
    fn test_recursive_fib_returns_correct_result_through_jit() {
        let source = "\
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
result = fib(12)
";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(
                pyre_object::intobject::w_int_get_value(result),
                144,
                "fib(12) should be 144 — recursive portal Ref ABI regression"
            );
        }
    }

    #[test]
    #[cfg_attr(
        feature = "cranelift",
        ignore = "cranelift CALL_ASSEMBLER Rust trampoline \
                  (call_assembler_guard_failure_inner / call_assembler_shim_inner) \
                  adds a native frame per recursive compiled entry. At the low JIT \
                  threshold used here, g(9)×2 runs enough compiled invocations to \
                  overflow the 2 MiB default cargo-test thread stack. Dynasm is \
                  unaffected (jmp trampoline). See \
                  memory/fib_recursive_sigbus_2026_04_19.md."
    )]
    fn test_recursive_global_reads_do_not_reuse_force_cache_across_global_mutation() {
        let _jit_params = TestJitParamsGuard::low_threshold();
        let source = "\
factor = 1
def g(n):
    if n < 2:
        return n * factor
    return g(n - 1) + g(n - 2) + factor

first = g(9)
factor = 2
second = g(9)";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let first = *(*frame.w_globals).get("first").unwrap();
            let second = *(*frame.w_globals).get("second").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(first), 88);
            assert_eq!(pyre_object::intobject::w_int_get_value(second), 176);
        }
    }

    #[test]
    fn test_inline_residual_user_call_with_many_args_stays_correct() {
        let _jit_params = TestJitParamsGuard::low_threshold();
        let source = "\
def helper(a, b, c, d, e):
    return a + b + c + d + e

def outer(x):
    return helper(x, x, x, x, x)

s = 0
i = 0
while i < 40:
    s = s + outer(i)
    i = i + 1";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = *(*frame.w_globals).get("s").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 3_900);
        }
    }

    #[test]
    fn test_nested_direct_helper_calls_stay_correct() {
        let _jit_params = TestJitParamsGuard::low_threshold();
        let source = "\
def add(a, b):
    return a + b

def mul(a, b):
    return a * b

def square(x):
    return mul(x, x)

def compute(i):
    return add(square(i), i)

s = 0
i = 0
while i < 40:
    s = add(s, compute(i))
    i = add(i, 1)";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = *(*frame.w_globals).get("s").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 21_320);
        }
    }

    /// rclass.py:1133-1137 `ll_issubclass(subcls, cls)` parity. After
    /// `set_gc_allocator` runs `freeze_types`, the materialized
    /// `(subclassrange_min, subclassrange_max)` for each registered
    /// PyType must satisfy `int_between(cls.min, subcls.min, cls.max)`
    /// for every (cls, subcls) pair where `subcls` Python-inherits from
    /// `cls`. This test exercises the `assign_inheritance_ids`
    /// (normalizecalls.py:373-389) preorder walk by verifying:
    ///   1. `INSTANCE_TYPE` (root `object`) range contains every other
    ///      PyType's range.
    ///   2. `INT_TYPE` range contains `BOOL_TYPE` range
    ///      (`bool.__bases__ == (int,)`).
    ///   3. Sibling classes (`INT_TYPE` vs `FLOAT_TYPE`, `STR_TYPE` vs
    ///      `LIST_TYPE`) are disjoint.
    #[test]
    fn test_subclass_range_preorder_bounds() {
        // Force JIT_DRIVER initialization so set_gc_allocator runs and
        // installs the active subclass_range hook.
        let _ = driver_pair();

        fn range(t: &pyre_object::pyobject::PyType) -> (i64, i64) {
            majit_gc::subclass_range(t as *const _ as usize)
                .expect("every built-in PyType must be registered with the GC")
        }

        // ll_issubclass(subcls, cls): a <= b < c.
        let contains = |outer: (i64, i64), inner: (i64, i64)| {
            outer.0 <= inner.0 && inner.0 < outer.1 && inner.1 <= outer.1
        };
        let disjoint = |a: (i64, i64), b: (i64, i64)| a.1 <= b.0 || b.1 <= a.0;

        let object_r = range(&pyre_object::pyobject::INSTANCE_TYPE);
        let int_r = range(&pyre_object::pyobject::INT_TYPE);
        let float_r = range(&pyre_object::pyobject::FLOAT_TYPE);
        let bool_r = range(&pyre_object::pyobject::BOOL_TYPE);
        let str_r = range(&pyre_object::pyobject::STR_TYPE);
        let list_r = range(&pyre_object::pyobject::LIST_TYPE);
        let none_r = range(&pyre_object::pyobject::NONE_TYPE);

        // (1) object encompasses every descendant.
        assert!(contains(object_r, int_r), "object ⊇ int");
        assert!(contains(object_r, float_r), "object ⊇ float");
        assert!(contains(object_r, bool_r), "object ⊇ bool");
        assert!(contains(object_r, str_r), "object ⊇ str");
        assert!(contains(object_r, list_r), "object ⊇ list");
        assert!(contains(object_r, none_r), "object ⊇ NoneType");

        // (2) int ⊇ bool (PyPy: W_BoolObject inherits from W_IntObject).
        assert!(contains(int_r, bool_r), "int ⊇ bool");

        // (3) Disjoint siblings.
        assert!(disjoint(int_r, float_r), "int ⊥ float");
        assert!(disjoint(int_r, str_r), "int ⊥ str");
        assert!(disjoint(float_r, str_r), "float ⊥ str");
        assert!(disjoint(str_r, list_r), "str ⊥ list");
        assert!(disjoint(float_r, bool_r), "float ⊥ bool");

        // (4) rclass.py:340-346 parity: subclassrange_{min,max} assigned
        // directly on the PyType (OBJECT_VTABLE) struct, not only in
        // the GC's TypeInfo table. ll_issubclass reads them from the
        // typeptr without a GC indirection.
        use pyre_object::pyobject::{BOOL_TYPE, FLOAT_TYPE, INSTANCE_TYPE, INT_TYPE};
        use std::sync::atomic::Ordering;
        assert_eq!(
            INSTANCE_TYPE.subclassrange_min.load(Ordering::Relaxed),
            object_r.0
        );
        assert_eq!(
            INSTANCE_TYPE.subclassrange_max.load(Ordering::Relaxed),
            object_r.1
        );
        assert_eq!(INT_TYPE.subclassrange_min.load(Ordering::Relaxed), int_r.0);
        assert_eq!(INT_TYPE.subclassrange_max.load(Ordering::Relaxed), int_r.1);
        assert_eq!(
            BOOL_TYPE.subclassrange_min.load(Ordering::Relaxed),
            bool_r.0
        );
        assert_eq!(
            BOOL_TYPE.subclassrange_max.load(Ordering::Relaxed),
            bool_r.1
        );
        assert_eq!(
            FLOAT_TYPE.subclassrange_min.load(Ordering::Relaxed),
            float_r.0
        );
        assert_eq!(
            FLOAT_TYPE.subclassrange_max.load(Ordering::Relaxed),
            float_r.1
        );

        // (5) ll_issubclass direct PyType reads match GC callback.
        assert!(pyre_object::pyobject::ll_issubclass(&BOOL_TYPE, &INT_TYPE));
        assert!(pyre_object::pyobject::ll_issubclass(
            &INT_TYPE,
            &INSTANCE_TYPE
        ));
        assert!(!pyre_object::pyobject::ll_issubclass(
            &INT_TYPE,
            &FLOAT_TYPE
        ));
        assert!(!pyre_object::pyobject::ll_issubclass(
            &FLOAT_TYPE,
            &INT_TYPE
        ));
    }

    #[test]
    fn test_dynamic_int_list_indexing_stays_correct() {
        let _jit_params = TestJitParamsGuard::low_threshold();
        let source = "\
q = [0, 1, 2, 3, 4]
i = 0
s = 0
while i < 40:
    q0 = i % 5
    s = s + q[q0]
    q[q0] = q[q0] + 1
    i = i + 1";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = *(*frame.w_globals).get("s").unwrap();
            let q = *(*frame.w_globals).get("q").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 220);
            assert_eq!(
                pyre_object::intobject::w_int_get_value(
                    pyre_object::listobject::w_list_getitem(q, 0).unwrap()
                ),
                8
            );
            assert_eq!(
                pyre_object::intobject::w_int_get_value(
                    pyre_object::listobject::w_list_getitem(q, 4).unwrap()
                ),
                12
            );
        }
    }
}
