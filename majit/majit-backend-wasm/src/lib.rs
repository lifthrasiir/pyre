/// WebAssembly backend for majit.
///
/// Generates wasm bytecodes via wasm-encoder. On wasm32 targets,
/// instantiates modules via JS WebAssembly.instantiate() and executes
/// via JS interop. On native targets, compile_loop succeeds but
/// execute_token requires the JS runtime (unreachable natively).
pub mod codegen;
pub mod failguard;

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
mod js_glue;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use failguard::{CompiledWasmLoop, WasmFailDescr, WasmFrameData};
use majit_backend::{AsmInfo, BackendError, DeadFrame, JitCellToken};
use majit_gc::GcAllocator;
use majit_ir::{FailDescr, GcRef, InputArg, Op, Value};

thread_local! {
    /// llmodel.py self.gc_ll_descr — owned by the active wasm
    /// backend on this thread. Stored as a thread-local so the
    /// backend-agnostic `majit_gc::ActiveGcGuardHooks` shims can
    /// reach the live allocator without taking a wasm dependency.
    /// Mirrors `cranelift::compiler::CRANELIFT_ACTIVE_GC` and
    /// `dynasm::runner::DYNASM_ACTIVE_GC` — RPython's
    /// `cpu.gc_ll_descr` parity, single-slot per thread.
    static WASM_ACTIVE_GC: RefCell<Option<Box<dyn GcAllocator>>> = const { RefCell::new(None) };
}

fn with_wasm_active_gc<R>(f: impl FnOnce(&dyn GcAllocator) -> R) -> Option<R> {
    WASM_ACTIVE_GC.with(|cell| {
        let guard = cell.borrow();
        guard.as_deref().map(f)
    })
}

/// `majit_gc::CheckIsObjectFn` installed by `set_gc_allocator`.
/// Mirrors cranelift's `check_is_object_via_active_runtime`: dispatches
/// through the wasm-thread-local GC allocator.
fn wasm_check_is_object(gcref: GcRef) -> bool {
    with_wasm_active_gc(|gc| gc.check_is_object(gcref)).unwrap_or(false)
}

fn wasm_get_actual_typeid(gcref: GcRef) -> Option<u32> {
    with_wasm_active_gc(|gc| gc.get_actual_typeid(gcref)).flatten()
}

fn wasm_subclass_range(classptr: usize) -> Option<(i64, i64)> {
    with_wasm_active_gc(|gc| gc.subclass_range(classptr)).flatten()
}

fn wasm_typeid_subclass_range(typeid: u32) -> Option<(i64, i64)> {
    with_wasm_active_gc(|gc| gc.typeid_subclass_range(typeid)).flatten()
}

fn wasm_typeid_is_object(typeid: u32) -> Option<bool> {
    with_wasm_active_gc(|gc| gc.typeid_is_object(typeid)).flatten()
}

/// Host-side nursery allocation trampoline. Published via
/// `majit_gc::set_active_alloc_nursery_typed` so backend-agnostic
/// callers (pyre-object `w_int_new`, …) can route through the
/// wasm-owned GC.
fn wasm_alloc_nursery_typed(type_id: u32, size: usize) -> GcRef {
    // See cranelift/dynasm counterparts: host-side allocation must not
    // trigger collection because the caller holds a raw pointer that
    // is not a registered GC root.
    WASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        match guard.as_deref_mut() {
            Some(gc) => gc.alloc_nursery_no_collect_typed(type_id, size),
            None => GcRef(0),
        }
    })
}

/// Host-side old-gen allocation trampoline (Task #141). Stable
/// across minor/major collections — see dynasm counterpart.
fn wasm_alloc_oldgen_typed(type_id: u32, size: usize) -> GcRef {
    WASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        match guard.as_deref_mut() {
            Some(gc) => gc.alloc_oldgen_typed(type_id, size),
            None => GcRef(0),
        }
    })
}

/// Host-side root-register trampoline (Task #141 option a).
///
/// # Safety
/// Caller must keep `slot` valid until [`wasm_gc_remove_root`] is
/// called with the same pointer.
unsafe fn wasm_gc_add_root(slot: *mut GcRef) {
    WASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        if let Some(gc) = guard.as_deref_mut() {
            unsafe { gc.add_root(slot) };
        }
    });
}

/// Companion to [`wasm_gc_add_root`].
fn wasm_gc_remove_root(slot: *mut GcRef) {
    WASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        if let Some(gc) = guard.as_deref_mut() {
            gc.remove_root(slot);
        }
    });
}

/// Host-side `is_managed_heap_object` trampoline.
fn wasm_gc_owns_object(addr: usize) -> bool {
    WASM_ACTIVE_GC.with(|cell| {
        let guard = cell.borrow();
        match guard.as_deref() {
            Some(gc) => gc.is_managed_heap_object(addr),
            None => false,
        }
    })
}

pub struct WasmBackend {
    trace_counter: u64,
    /// Optimizer constant pool (constant-namespace OpRef → i64 value).
    constants: HashMap<u32, i64>,
    /// llmodel.py:64-69 self.vtable_offset.
    vtable_offset: Option<usize>,
}

impl WasmBackend {
    pub fn new() -> Self {
        WasmBackend {
            trace_counter: 0,
            constants: HashMap::new(),
            vtable_offset: None,
        }
    }

    /// Active vtable_offset for wasm codegen.
    pub fn vtable_offset(&self) -> Option<usize> {
        self.vtable_offset
    }

    // `set_constants`, `set_next_trace_id`, `set_next_header_pc` are
    // provided via the `Backend` trait impl below.  wasm has no
    // `set_constant_types` inherent — the trait default (no-op) is used.

    /// llmodel.py:53-54: store gc_ll_descr on the cpu instance.
    ///
    /// Mirrors `CraneliftBackend::set_gc_allocator`: stores the box in
    /// the wasm thread-local seam and publishes the same five
    /// `ActiveGcGuardHooks` so the backend-agnostic optimizer /
    /// blackhole executor reach the live allocator without taking a
    /// wasm dependency.
    pub fn set_gc_allocator(&mut self, mut gc: Box<dyn majit_gc::GcAllocator>) {
        // gctypelayout.encode_type_shapes_now parity: close the
        // type-registration phase before any compile embeds the
        // type_info_group base address. Mirrors
        // `CraneliftBackend::set_gc_allocator`.
        gc.freeze_types();
        let supports_guard_gc_type = gc.supports_guard_gc_type();
        WASM_ACTIVE_GC.with(|cell| *cell.borrow_mut() = Some(gc));
        majit_gc::set_active_gc_guard_hooks(majit_gc::ActiveGcGuardHooks {
            check_is_object: Some(wasm_check_is_object),
            get_actual_typeid: Some(wasm_get_actual_typeid),
            subclass_range: Some(wasm_subclass_range),
            typeid_subclass_range: Some(wasm_typeid_subclass_range),
            typeid_is_object: Some(wasm_typeid_is_object),
            supports_guard_gc_type,
        });
        majit_gc::set_active_alloc_nursery_typed(Some(wasm_alloc_nursery_typed));
        majit_gc::set_active_alloc_oldgen_typed(Some(wasm_alloc_oldgen_typed));
        majit_gc::set_active_root_hooks(Some(wasm_gc_add_root), Some(wasm_gc_remove_root));
        majit_gc::set_active_gc_owns_object(Some(wasm_gc_owns_object));
    }

    /// llmodel.py:64-69 self.vtable_offset configuration.
    pub fn set_vtable_offset(&mut self, offset: Option<usize>) {
        self.vtable_offset = offset;
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Resolves a vtable pointer to its registered GC type id via the
    /// installed gc_ll_descr.
    pub fn lookup_typeid_from_classptr(&self, classptr: usize) -> Option<u32> {
        with_wasm_active_gc(|gc| gc.get_typeid_from_classptr_if_gcremovetypeptr(classptr)).flatten()
    }

    /// Pre-compute classptr → expected_typeid pairs for every GuardClass /
    /// GuardNonnullClass operand seen in `ops`. wasm codegen runs without a
    /// borrow of `self`, so we materialize the resolver as a HashMap.
    /// Only GuardClass / GuardNonnullClass need this table — GuardGcType
    /// already carries an immediate typeid (assembler.py:1919-1922) and
    /// GUARD_IS_OBJECT / GUARD_SUBCLASS use a different lookup path.
    fn collect_classptr_typeid_table(&self, ops: &[Op]) -> HashMap<i64, u32> {
        let mut table = HashMap::new();
        if self.vtable_offset.is_some() {
            return table;
        }
        if WASM_ACTIVE_GC.with(|cell| cell.borrow().is_none()) {
            return table;
        }
        for op in ops {
            if matches!(
                op.opcode,
                majit_ir::OpCode::GuardClass | majit_ir::OpCode::GuardNonnullClass
            ) && op.args.len() >= 2
            {
                if let Some(&classptr) = self.constants.get(&op.args[1].raw()) {
                    if let Some(tid) = self.lookup_typeid_from_classptr(classptr as usize) {
                        table.insert(classptr, tid);
                    }
                }
            }
        }
        table
    }

    /// Pre-fetch `GuardGcTypeInfo` from the installed `gc_ll_descr`.
    ///
    /// Mirrors the `self.cpu.gc_ll_descr.get_translated_info_*` /
    /// `cpu.subclassrange_min_offset` lookups that RPython's
    /// `genop_guard_guard_is_object` (x86/assembler.py:1924-1943) and
    /// `genop_guard_guard_subclass` (x86/assembler.py:1945-1980) do at
    /// codegen time. The returned struct is handed to
    /// `codegen::build_wasm_module`; the codegen arms assert
    /// `supports_guard_gc_type` before reading any other field.
    ///
    /// Also pre-computes `(subclassrange_min, subclassrange_max)` for
    /// every constant classptr argument of a `GuardSubclass` op
    /// (assembler.py:1971-1974 reads these bounds at codegen time).
    fn collect_guard_gc_type_info(&self, ops: &[Op]) -> codegen::GuardGcTypeInfo {
        with_wasm_active_gc(|gc| {
            let mut info = codegen::GuardGcTypeInfo::default();
            info.supports_guard_gc_type = gc.supports_guard_gc_type();
            if !info.supports_guard_gc_type {
                return info;
            }
            // assembler.py:1934-1937: gc_ll_descr lookups.
            let (base, shift, sizeof_ti) = gc.get_translated_info_for_typeinfo();
            info.base_type_info = base;
            info.shift_by = shift;
            info.sizeof_ti = sizeof_ti;
            let (infobits_off, is_object_flag) = gc.get_translated_info_for_guard_is_object();
            info.infobits_offset = infobits_off;
            info.is_object_flag = is_object_flag;
            // assembler.py:1951: cpu.subclassrange_min_offset.
            info.subclassrange_min_offset = gc.subclassrange_min_offset();
            // assembler.py:1971-1974: (subclassrange_min, subclassrange_max)
            // for every constant GuardSubclass arg1.
            for op in ops {
                if op.opcode == majit_ir::OpCode::GuardSubclass && op.args.len() >= 2 {
                    if let Some(&classptr) = self.constants.get(&op.args[1].raw()) {
                        if let Some(range) = gc.subclass_range(classptr as usize) {
                            info.subclass_ranges.insert(classptr, range);
                        }
                    }
                }
            }
            info
        })
        .unwrap_or_default()
    }

    /// Collect constants from ops (constant OpRefs that appear as args).
    fn collect_constants_from_ops(&mut self, ops: &[Op]) {
        for op in ops {
            for &arg in &op.args {
                if arg.is_constant() && !self.constants.contains_key(&arg.raw()) {
                    // Default to 0 if not already registered
                    self.constants.insert(arg.raw(), 0);
                }
            }
            if let Some(ref fail_args) = op.fail_args {
                for &arg in fail_args.iter() {
                    if arg.is_constant() && !self.constants.contains_key(&arg.raw()) {
                        self.constants.insert(arg.raw(), 0);
                    }
                }
            }
        }
    }
}

unsafe impl Send for WasmBackend {}

impl majit_backend::Backend for WasmBackend {
    fn compile_loop(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        token: &mut JitCellToken,
    ) -> Result<AsmInfo, BackendError> {
        self.collect_constants_from_ops(ops);
        let trace_id = self.trace_counter;
        self.trace_counter += 1;

        let typeid_table = self.collect_classptr_typeid_table(ops);
        let guard_gc_type_info = self.collect_guard_gc_type_info(ops);
        let (wasm_bytes, guard_exits) = codegen::build_wasm_module(
            inputargs,
            ops,
            &self.constants,
            self.vtable_offset,
            &typeid_table,
            &guard_gc_type_info,
        )?;

        // Build fail descriptors
        let fail_descrs: Vec<Arc<WasmFailDescr>> = guard_exits
            .iter()
            .map(|g| {
                Arc::new(WasmFailDescr {
                    fail_index: g.fail_index,
                    trace_id,
                    fail_arg_types: g.fail_arg_types.clone(),
                    is_finish: g.is_finish,
                })
            })
            .collect();

        let max_output_slots = guard_exits
            .iter()
            .map(|g| g.fail_arg_refs.len())
            .max()
            .unwrap_or(0)
            .max(inputargs.len());

        // Compile via JS on wasm32, or store bytes for testing on native
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        let func_handle = js_glue::compile_module(&wasm_bytes);
        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        let func_handle = 0u32; // Placeholder — no JS runtime available

        let compiled = CompiledWasmLoop {
            trace_id,
            input_types: inputargs.iter().map(|ia| ia.tp).collect(),
            func_handle,
            fail_descrs,
            num_inputs: inputargs.len(),
            max_output_slots,
        };

        token.compiled = Some(Box::new(compiled));

        Ok(AsmInfo {
            code_addr: 0,
            code_size: wasm_bytes.len(),
        })
    }

    fn set_constants(&mut self, constants: HashMap<u32, i64>) {
        self.constants = constants;
    }

    fn set_next_trace_id(&mut self, trace_id: u64) {
        self.trace_counter = trace_id;
    }

    // `set_constant_types` / `set_next_header_pc` use the trait default
    // (no-op) — wasm does not currently honour either value.

    fn compile_bridge(
        &mut self,
        _fail_descr: &dyn FailDescr,
        _inputargs: &[InputArg],
        _ops: &[Op],
        _original_token: &JitCellToken,
        _previous_tokens: &[std::sync::Arc<JitCellToken>],
    ) -> Result<AsmInfo, BackendError> {
        Err(BackendError::Unsupported(
            "wasm backend: bridge compile not yet implemented".into(),
        ))
    }

    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame {
        let compiled = token
            .compiled
            .as_ref()
            .expect("no compiled code")
            .downcast_ref::<CompiledWasmLoop>()
            .expect("not CompiledWasmLoop");

        // Allocate frame area large enough for slots + call trampoline area.
        // MIN_FRAME_BYTES accommodates the call area at offset 2000+.
        let min_slots = codegen::MIN_FRAME_BYTES / 8;
        let frame_size = min_slots.max(1 + compiled.max_output_slots.max(compiled.num_inputs));
        let mut frame = vec![0i64; frame_size];

        // Write inputs to frame[1..]
        for (i, arg) in args.iter().enumerate() {
            frame[1 + i] = match arg {
                Value::Int(v) => *v,
                Value::Float(v) => v.to_bits() as i64,
                Value::Ref(r) => r.0 as i64,
                Value::Void => 0,
            };
        }

        // frame_ptr = byte offset of frame[0] in wasm linear memory
        let _frame_ptr = frame.as_mut_ptr() as usize as u32;

        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        {
            panic!("wasm backend execute_token requires JS runtime");
        }
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        {
            js_glue::execute(compiled.func_handle, _frame_ptr);

            // Read fail_index from frame[0]
            let fail_index = frame[0] as u32;
            let fail_descr = compiled
                .fail_descrs
                .get(fail_index as usize)
                .expect("invalid fail_index from compiled wasm");

            // Read output values
            let num_outputs = fail_descr.fail_arg_types.len();
            let raw_values: Vec<i64> = (0..num_outputs).map(|i| frame[1 + i]).collect();

            DeadFrame {
                data: Box::new(WasmFrameData {
                    raw_values,
                    fail_descr: fail_descr.clone(),
                }),
            }
        }
    }

    fn execute_token_ints(&self, token: &JitCellToken, args: &[i64]) -> DeadFrame {
        let values: Vec<Value> = args.iter().map(|&v| Value::Int(v)).collect();
        self.execute_token(token, &values)
    }

    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr {
        let data = frame
            .data
            .downcast_ref::<WasmFrameData>()
            .expect("not WasmFrameData");
        data.fail_descr.as_ref()
    }

    fn get_latest_descr_arc(&self, frame: &DeadFrame) -> Arc<dyn FailDescr> {
        let data = frame
            .data
            .downcast_ref::<WasmFrameData>()
            .expect("not WasmFrameData");
        Arc::clone(&data.fail_descr) as Arc<dyn FailDescr>
    }

    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64 {
        let data = frame
            .data
            .downcast_ref::<WasmFrameData>()
            .expect("not WasmFrameData");
        data.raw_values[index]
    }

    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64 {
        let data = frame
            .data
            .downcast_ref::<WasmFrameData>()
            .expect("not WasmFrameData");
        f64::from_bits(data.raw_values[index] as u64)
    }

    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> GcRef {
        let data = frame
            .data
            .downcast_ref::<WasmFrameData>()
            .expect("not WasmFrameData");
        GcRef(data.raw_values[index] as usize)
    }

    fn invalidate_loop(&self, _token: &JitCellToken) {
        // No native code to invalidate — wasm modules are immutable.
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Resolves a vtable pointer through the installed gc_ll_descr.
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, classptr: usize) -> Option<u32> {
        self.lookup_typeid_from_classptr(classptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_backend::Backend;
    use majit_gc::collector::MiniMarkGC;
    use majit_gc::trace::TypeInfo;

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr
    /// Verify the wasm backend's gc_ll_descr round-trips a registered
    /// vtable→type_id mapping.
    #[test]
    fn test_backend_typeid_from_classptr_via_gc_ll_descr() {
        let mut gc = MiniMarkGC::new();
        let int_tid = gc.register_type(TypeInfo::simple(16));
        let int_vtable: usize = 0x3333_4400;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, int_vtable, int_tid);

        let mut backend = WasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let resolved = backend.get_typeid_from_classptr_if_gcremovetypeptr(int_vtable);
        assert_eq!(resolved, Some(int_tid));
        let unknown = backend.get_typeid_from_classptr_if_gcremovetypeptr(0xCAFE_F00D);
        assert_eq!(unknown, None);
    }
}
