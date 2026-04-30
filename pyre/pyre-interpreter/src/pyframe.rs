//! PyFrame — execution frame for Python bytecode.
//!
//! Each function call creates a new frame with its own value stack,
//! local variables, and instruction pointer. The JIT virtualizes
//! these fields so they live in registers instead of memory.

use std::collections::VecDeque;
use std::rc::Rc;

use crate::{CodeFlags, CodeObject};
use crate::{DictStorage, PyExecutionContext};
use pyre_object::FixedObjectArray;
use pyre_object::*;

// Ensure *const PyExecutionContext and Rc<PyExecutionContext> have the same
// size so that PyFrame field offsets are preserved after the switch.
const _: () = assert!(
    std::mem::size_of::<*const PyExecutionContext>()
        == std::mem::size_of::<Rc<PyExecutionContext>>()
);

#[derive(Debug, Clone, Copy)]
pub enum PendingInlineResult {
    Ref(PyObjectRef),
    Int(i64),
    Float(f64),
}

/// Execution frame for a single Python code block.
///
/// Unified `locals_cells_stack_w` array layout:
///   - indices `0..nlocals` — local variables
///   - indices `nlocals..nlocals+ncells` — cell/free variable slots
///   - indices `nlocals+ncells..` — operand stack
///
/// `valuestackdepth` is the absolute index into this array; it starts at
/// `nlocals + ncells` (empty stack) and grows upward on push.
///
/// The JIT's Virtualize pass keeps `locals_cells_stack_w` slots in CPU
/// registers during compiled code execution, eliminating heap reads/writes
/// for the hottest interpreter state.
///
/// The `vable_token` field coordinates ownership: when JIT code is
/// running, the token is nonzero and the canonical field values live
/// in registers. A "force" flushes them back to the heap.
#[repr(C)]
pub struct PyFrame {
    /// Raw pointer to the shared execution context.
    /// The top-level frame leaks the Rc via `Rc::into_raw`.
    /// Callee frames just copy the pointer (no atomic refcount ops).
    pub execution_context: *const PyExecutionContext,
    /// Pointer to the Code object (W_CodeObject).
    ///
    /// PyPy: pyframe.py `self.pycode = code` — stores the PyCode instance.
    /// Same pointer as `func.getcode()`, so `getcode(func) == frame.pycode`.
    pub pycode: *const (),
    /// pypy/interpreter/pyframe.py:84,110-112 locals_cells_stack_w
    /// `[None] * size; make_sure_not_resized(...)` → fixed-length GcArray.
    pub locals_cells_stack_w: *mut FixedObjectArray,
    /// Absolute index into `locals_cells_stack_w` marking the top of the
    /// operand stack. Starts at `nlocals + ncells` (empty stack), grows upward.
    pub valuestackdepth: usize,
    /// `pyframe.py:72 last_instr` — index of the last executed instruction.
    /// PyPy initializes to -1; `get_last_lineno` uses this for `offset2lineno`.
    ///
    /// PRE-EXISTING-ADAPTATION (Stage 3 / Task #118): pyre stores the PC of
    /// the last executed instruction here (matches CPython's `f_lasti`
    /// semantic, returned directly by `fget_f_lasti`). RPython's runtime
    /// stores `last_instr = next_instr` instead, set every dispatch by
    /// `pypy/interpreter/pyopcode.py:172 self.last_instr = intmask(next_instr)`,
    /// so RPython's `last_instr` is one ahead of pyre's. The two conventions
    /// are equivalent modulo ±1; `next_instr()` below adds one to recover
    /// PyPy's view, and JIT shadow code (`vable_last_instr`) is consistently
    /// produced as `pc - 1` and consumed as such across both pyre-jit and
    /// pyre-jit-trace. Flipping storage to RPython's convention is
    /// achievable but requires an atomic 30+ site mechanical rewrite (every
    /// `vable_last_instr = ctx.const_int(pc - 1)` producer plus the
    /// downstream readers) with no observable behavior change. Until that is
    /// done the divergence is documented but accepted; cited at
    /// `pypy/interpreter/pyopcode.py:172` and
    /// `pypy/module/pypyjit/interp_jit.py:25`.
    pub last_instr: isize,
    /// pyframe.py:80 escaped — see mark_as_escaped()
    pub escaped: bool,
    /// pyframe.py:82 debugdata — lazily allocated tracing/debug payload.
    /// Virtualizable static field (interp_jit.py:28).
    pub debugdata: *mut FrameDebugData,
    /// pyframe.py:86 lastblock — head of the FrameBlock linked list.
    /// Virtualizable static field (interp_jit.py:29).
    pub lastblock: *mut FrameBlock,
    /// pyframe.py:49 / interp_jit.py:31 w_globals.
    pub w_globals: *mut DictStorage,
    /// Virtualizable token — set by JIT when this frame is virtualized.
    /// 0 = not virtualized, nonzero = pointer to JIT state.
    pub vable_token: usize,
    /// PyPy: `frame_finished_execution = False`.
    pub frame_finished_execution: bool,
    /// PyPy: `f_generator_nowref = None`.
    pub f_generator_nowref: PyObjectRef,
    /// PyPy: `w_yielding_from = None`.
    pub w_yielding_from: PyObjectRef,
    /// PyPy: `f_backref = jit.vref_None`.
    pub f_backref: *mut PyFrame,
    /// Concrete inline-trace replay results owned by this frame.
    ///
    /// PyPy's `finishframe()` writes each child result into the parent in
    /// bytecode order. Tracing can run ahead of concrete execution and queue
    /// multiple inline-handled CALL results on the same caller frame before
    /// the interpreter replays the first CALL opcode, so this must preserve
    /// ordering instead of using a single overwrite-prone slot.
    pub pending_inline_results: VecDeque<PendingInlineResult>,
    /// Outermost inline trace-through resumed this frame past the CALL at the
    /// recorded pc.
    ///
    /// Nested inline frames already follow the MIFrame-owned
    /// make_result_of_lastop path directly. This marker narrows the same
    /// protocol to the outermost interpreter loop without conflating it with
    /// unrelated next_instr changes.
    pub pending_inline_resume_pc: Option<usize>,
}

/// GC header size in bytes.  Matches `majit_gc::header::GcHeader::SIZE`.
/// Every PyObjectArray and PyFrame allocation prepends this many zero bytes
/// so that RPython-style write barriers (`*(obj + wb_byteofs) & mask`) read a
/// valid header with `TRACK_YOUNG_PTRS=0` and skip the slow path.
pub const GC_HEADER_SIZE: usize = 8;

/// Allocate a value of type `T` with a zeroed GC header prepended.
///
/// incminimark write barrier reads at `obj - HEADER_SIZE`;
/// zeroed header => `TRACK_YOUNG_PTRS` clear => barrier fast-path skips.
///
/// RPython objects are GC-managed references. The frame holds non-owning
/// pointers; the GC (or, in pyre's simplified model, the allocator) owns
/// the allocation. Never manually free pointers returned by this function
/// from frame code — that would violate the GC ref contract and cause
/// dangling pointers when the JIT captures these refs in snapshots.
unsafe fn alloc_with_gc_header<T>(value: T) -> *mut T {
    unsafe {
        let total = GC_HEADER_SIZE + std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>().max(8);
        let layout = std::alloc::Layout::from_size_align(total, align).unwrap();
        let raw = std::alloc::alloc_zeroed(layout);
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let ptr = raw.add(GC_HEADER_SIZE) as *mut T;
        std::ptr::write(ptr, value);
        ptr
    }
}

/// Allocation size (in bytes, including the GC header) for a
/// `FixedObjectArray` of the given length.
#[inline]
fn fixed_array_alloc_size(len: usize) -> usize {
    GC_HEADER_SIZE
        + pyre_object::FIXED_ARRAY_ITEMS_OFFSET
        + len * std::mem::size_of::<pyre_object::PyObjectRef>()
}

#[inline]
fn fixed_array_layout(len: usize) -> std::alloc::Layout {
    std::alloc::Layout::from_size_align(fixed_array_alloc_size(len), 8).unwrap()
}

/// Allocate a fixed-length GcArray-shaped `FixedObjectArray` with all
/// slots initialised to `fill`. The allocation is prefixed with a
/// zeroed GC header (same convention as [`alloc_with_gc_header`]) so the
/// write barrier fast-path sees `TRACK_YOUNG_PTRS=0`.
///
/// The returned pointer points at the length prefix; items follow
/// immediately in memory at `FIXED_ARRAY_ITEMS_OFFSET` — matching
/// RPython `Ptr(GcArray(PyObjectRef))`.
pub unsafe fn alloc_fixed_array_with_header(
    len: usize,
    fill: pyre_object::PyObjectRef,
) -> *mut FixedObjectArray {
    unsafe {
        let layout = fixed_array_layout(len);
        let raw = std::alloc::alloc_zeroed(layout);
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let arr = raw.add(GC_HEADER_SIZE) as *mut FixedObjectArray;
        (*arr).len = len;
        let items = (arr as *mut u8).add(pyre_object::FIXED_ARRAY_ITEMS_OFFSET)
            as *mut pyre_object::PyObjectRef;
        for i in 0..len {
            items.add(i).write(fill);
        }
        arr
    }
}

/// Allocate a `FixedObjectArray` pre-populated from `values`. The
/// resulting array has `values.len()` slots; allocation layout matches
/// [`alloc_fixed_array_with_header`].
pub unsafe fn alloc_fixed_array_from_vec(
    values: Vec<pyre_object::PyObjectRef>,
) -> *mut FixedObjectArray {
    unsafe {
        let len = values.len();
        let layout = fixed_array_layout(len);
        let raw = std::alloc::alloc_zeroed(layout);
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let arr = raw.add(GC_HEADER_SIZE) as *mut FixedObjectArray;
        (*arr).len = len;
        let items = (arr as *mut u8).add(pyre_object::FIXED_ARRAY_ITEMS_OFFSET)
            as *mut pyre_object::PyObjectRef;
        for (i, v) in values.into_iter().enumerate() {
            items.add(i).write(v);
        }
        arr
    }
}

/// Deallocate a `FixedObjectArray` allocated with
/// [`alloc_fixed_array_with_header`] or [`alloc_fixed_array_from_vec`].
pub unsafe fn dealloc_array_with_gc_header(ptr: *mut FixedObjectArray) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let len = (*ptr).len;
        let raw = (ptr as *mut u8).sub(GC_HEADER_SIZE);
        std::alloc::dealloc(raw, fixed_array_layout(len));
    }
}

unsafe fn clone_debugdata_ptr(ptr: *mut FrameDebugData) -> *mut FrameDebugData {
    unsafe {
        if ptr.is_null() {
            std::ptr::null_mut()
        } else {
            pyre_object::lltype::malloc_raw((*ptr).clone())
        }
    }
}

unsafe fn clear_debugdata_ptr(ptr: &mut *mut FrameDebugData) {
    unsafe {
        if !(*ptr).is_null() {
            drop(Box::from_raw(*ptr));
            *ptr = std::ptr::null_mut();
        }
    }
}

unsafe fn clone_block_chain(ptr: *mut FrameBlock) -> *mut FrameBlock {
    unsafe {
        if ptr.is_null() {
            std::ptr::null_mut()
        } else {
            pyre_object::lltype::malloc_raw(FrameBlock {
                handlerposition: (*ptr).handlerposition,
                valuestackdepth: (*ptr).valuestackdepth,
                previous: clone_block_chain((*ptr).previous),
            })
        }
    }
}

unsafe fn clear_block_chain(ptr: &mut *mut FrameBlock) {
    unsafe {
        let mut current = *ptr;
        while !current.is_null() {
            let block = Box::from_raw(current);
            current = block.previous;
        }
        *ptr = std::ptr::null_mut();
    }
}

impl Drop for PyFrame {
    fn drop(&mut self) {
        if !self.locals_cells_stack_w.is_null() {
            unsafe { dealloc_array_with_gc_header(self.locals_cells_stack_w) };
            self.locals_cells_stack_w = std::ptr::null_mut();
        }
        unsafe {
            clear_debugdata_ptr(&mut self.debugdata);
            clear_block_chain(&mut self.lastblock);
        }
    }
}

impl PyFrame {
    /// Access locals_cells_stack_w (deref the pointer).
    #[inline]
    pub fn locals_w(&self) -> &FixedObjectArray {
        unsafe { &*self.locals_cells_stack_w }
    }

    /// Mutably access locals_cells_stack_w.
    #[inline]
    pub fn locals_w_mut(&mut self) -> &mut FixedObjectArray {
        unsafe { &mut *self.locals_cells_stack_w }
    }
}

/// Extract raw CodeObject from frame's W_CodeObject.
///
/// PyPy: `frame.pycode` gives `PyCode` which IS the code object.
/// pyre: W_CodeObject wraps a raw CodeObject — this extracts it.
#[inline]
pub unsafe fn pyframe_get_pycode(frame: &PyFrame) -> *const CodeObject {
    unsafe { crate::w_code_get_ptr(frame.pycode as pyre_object::PyObjectRef) as *const CodeObject }
}

#[repr(C)]
#[derive(Clone)]
pub struct FrameDebugData {
    /// pyframe.py:44
    pub w_locals: *mut DictStorage,
    /// pyframe.py:49 — set in __init__ from pycode.w_globals
    pub w_globals: *mut DictStorage,
    /// pyframe.py:37
    pub w_f_trace: PyObjectRef,
    /// pyframe.py:40
    pub is_being_profiled: bool,
    /// pyframe.py:41
    pub is_in_line_tracing: bool,
    /// pyframe.py:42
    pub f_trace_lines: bool,
    /// pyframe.py:43
    pub f_trace_opcodes: bool,
    /// pyframe.py:38
    pub instr_prev_plus_one: isize,
    /// pyframe.py:39
    pub f_lineno: isize,
    /// pyframe.py:45
    pub hidden_operationerr: PyObjectRef,
}

impl FrameDebugData {
    pub fn new(pycode: *const (), init_lineno: isize) -> Self {
        Self {
            w_locals: std::ptr::null_mut(),
            w_globals: unsafe { crate::w_code_get_w_globals(pycode as PyObjectRef) },
            w_f_trace: pyre_object::PY_NULL,
            is_being_profiled: false,
            is_in_line_tracing: false,
            f_trace_lines: true,
            f_trace_opcodes: false,
            instr_prev_plus_one: 0,
            f_lineno: init_lineno,
            hidden_operationerr: pyre_object::PY_NULL,
        }
    }
}

impl Default for FrameDebugData {
    fn default() -> Self {
        Self::new(std::ptr::null(), -1)
    }
}

/// pyopcode.py:1875-1897 FrameBlock — linked list node for the block stack.
/// `previous` forms a singly-linked list; `lastblock` in PyFrame is the head.
#[derive(Debug, Clone, Copy)]
pub struct FrameBlock {
    /// pyopcode.py:1883
    pub valuestackdepth: usize,
    /// pyopcode.py:1882
    pub handlerposition: usize,
    /// pyopcode.py:1884 — pointer to the previous FrameBlock (null = None).
    pub previous: *mut FrameBlock,
}

impl FrameBlock {
    /// pyopcode.py:1886-1887
    #[inline]
    pub fn cleanupstack(&self, frame: &mut PyFrame) {
        frame.dropvaluesuntil(self.valuestackdepth);
    }
}

#[inline]
pub fn get_block_class(opname: &str) -> &'static str {
    match opname {
        "SETUP_LOOP" | "SETUP_EXCEPT" | "SETUP_FINALLY" | "SETUP_WITH" => "FrameBlock",
        _ => "FrameBlock",
    }
}

#[inline]
pub fn unpickle_block(_space: PyObjectRef, w_tup: PyObjectRef) -> FrameBlock {
    let _ = _space;
    let handlerposition = unsafe {
        w_tuple_getitem(w_tup, 0).and_then(|v| {
            if is_int(v) {
                Some(w_int_get_value(v) as usize)
            } else {
                None
            }
        })
    }
    .unwrap_or(0);
    let valuestackdepth = unsafe {
        w_tuple_getitem(w_tup, 2).and_then(|v| {
            if is_int(v) {
                Some(w_int_get_value(v) as usize)
            } else {
                None
            }
        })
    }
    .unwrap_or(0);
    FrameBlock {
        handlerposition,
        valuestackdepth,
        previous: std::ptr::null_mut(),
    }
}

// ── Virtualizable field offsets ───────────────────────────────────────
//
// These constants tell the JIT where each virtualizable field lives
// inside a PyFrame, so it can read/write them via raw pointer arithmetic.
// Equivalent to PyPy's `_virtualizable_` descriptor on pyframe.py.

/// Byte offset of `pycode` in `PyFrame`.
pub const PYFRAME_PYCODE_OFFSET: usize = std::mem::offset_of!(PyFrame, pycode);

/// Byte offset of `vable_token` in `PyFrame`.
pub const PYFRAME_VABLE_TOKEN_OFFSET: usize = std::mem::offset_of!(PyFrame, vable_token);

/// Byte offset of `last_instr` in `PyFrame`.
pub const PYFRAME_LAST_INSTR_OFFSET: usize = std::mem::offset_of!(PyFrame, last_instr);

/// Byte offset of `valuestackdepth` in `PyFrame`.
pub const PYFRAME_VALUESTACKDEPTH_OFFSET: usize = std::mem::offset_of!(PyFrame, valuestackdepth);

/// Byte offset of `locals_cells_stack_w` in `PyFrame`.
pub const PYFRAME_LOCALS_CELLS_STACK_OFFSET: usize =
    std::mem::offset_of!(PyFrame, locals_cells_stack_w);

/// Byte offset of `debugdata` in `PyFrame`.
pub const PYFRAME_DEBUGDATA_OFFSET: usize = std::mem::offset_of!(PyFrame, debugdata);

/// Byte offset of `lastblock` in `PyFrame`.
pub const PYFRAME_LASTBLOCK_OFFSET: usize = std::mem::offset_of!(PyFrame, lastblock);

/// Byte offset of `w_globals` in `PyFrame`.
pub const PYFRAME_W_GLOBALS_OFFSET: usize = std::mem::offset_of!(PyFrame, w_globals);

// Backward-compat aliases used by JIT code.
pub const PYFRAME_STACK_DEPTH_OFFSET: usize = PYFRAME_VALUESTACKDEPTH_OFFSET;
pub const PYFRAME_LOCALS_OFFSET: usize = PYFRAME_LOCALS_CELLS_STACK_OFFSET;

/// pytraceback.py offset2lineno(c, stopat) — convert instruction index to line number.
/// Matches RPython: negative `stopat` means "frame not yet started", returns
/// first-line.
#[inline]
pub fn offset2lineno(code: &CodeObject, stopat: isize) -> usize {
    let lineno = code.first_line_number.map(|n| n.get()).unwrap_or(1);
    if stopat < 0 {
        return lineno;
    }
    code.locations
        .get(stopat as usize)
        .map(|(start, _)| start.line.get())
        .unwrap_or(lineno)
}

/// pyframe.py:105-106 — cell + free variable slot count.
/// PyPy: ncellvars = len(code.co_cellvars), nfreevars = len(code.co_freevars)
/// No overlap filtering — cells and locals occupy separate slots even if
/// a cellvar has the same name as a local variable.
#[inline]
pub fn ncells(code: &CodeObject) -> usize {
    code.cellvars.len() + code.freevars.len()
}

impl PyFrame {
    /// pyframe.py:121 getdebug → self.debugdata
    #[inline]
    fn getdebug_data(&self) -> Option<&FrameDebugData> {
        (!self.debugdata.is_null()).then(|| unsafe { &*self.debugdata })
    }

    /// pyframe.py:124 getorcreatedebug
    #[inline]
    fn getorcreate_debug_data(&mut self, init_lineno: isize) -> &mut FrameDebugData {
        if self.debugdata.is_null() {
            self.debugdata =
                pyre_object::lltype::malloc_raw(FrameDebugData::new(self.pycode, init_lineno));
        }
        unsafe { &mut *self.debugdata }
    }

    /// PyPy-compatible `getdebug()`.
    #[inline]
    pub fn getdebug(&self) -> Option<&FrameDebugData> {
        self.getdebug_data()
    }

    /// PyPy-compatible `getorcreatedebug()`.
    #[inline]
    pub fn getorcreatedebug(&mut self, init_lineno: isize) -> &mut FrameDebugData {
        self.getorcreate_debug_data(init_lineno)
    }

    /// PyPy-compatible alias for `code()`.
    #[inline]
    pub fn getcode(&self) -> &CodeObject {
        self.code()
    }

    /// PyPy-compatible `fget_code`.
    #[inline]
    pub fn fget_code(&self) -> &CodeObject {
        self.code()
    }

    /// pyframe.py:129-133 get_w_globals
    #[inline]
    pub fn get_w_globals(&self) -> *mut DictStorage {
        match self.getdebug_data() {
            Some(data) => data.w_globals,
            None => unsafe { crate::w_code_get_w_globals(self.pycode as PyObjectRef) },
        }
    }

    /// pyframe.py:135 get_w_f_trace
    #[inline]
    pub fn get_w_f_trace(&self) -> PyObjectRef {
        self.getdebug_data()
            .and_then(|data| (!data.w_f_trace.is_null()).then_some(data.w_f_trace))
            .unwrap_or(pyre_object::PY_NULL)
    }

    /// pyframe.py:141 get_is_being_profiled
    #[inline]
    pub fn get_is_being_profiled(&self) -> bool {
        self.getdebug_data()
            .map_or(false, |data| data.is_being_profiled)
    }

    /// pyframe.py:147 get_w_locals
    #[inline]
    pub fn get_w_locals(&self) -> *mut DictStorage {
        self.getdebug_data()
            .map_or(std::ptr::null_mut(), |data| data.w_locals)
    }

    /// pyframe.py:583-588 getdictscope
    #[inline]
    pub fn getdictscope(&mut self) -> *mut DictStorage {
        self.fast2locals();
        self.get_w_locals()
    }

    /// PyPy-compatible `__init__` hook.
    #[inline]
    pub fn __init__(
        &mut self,
        code: *const (),
        w_globals: *mut DictStorage,
        outer_func: PyObjectRef,
    ) {
        let _ = outer_func;
        self.pycode = code;
        let raw =
            unsafe { crate::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject };
        self.w_globals = w_globals;
        unsafe { dealloc_array_with_gc_header(self.locals_cells_stack_w) };
        self.locals_cells_stack_w = unsafe {
            alloc_fixed_array_with_header(
                (&*raw).varnames.len() + ncells(&*raw) + (&*raw).max_stackdepth as usize,
                PY_NULL,
            )
        };
        self.valuestackdepth = unsafe { (&*raw).varnames.len() + ncells(&*raw) };
        self.last_instr = -1;
        self.escaped = false;
        self.frame_finished_execution = false;
        self.f_generator_nowref = PY_NULL;
        self.w_yielding_from = PY_NULL;
        self.f_backref = std::ptr::null_mut();
        unsafe {
            clear_debugdata_ptr(&mut self.debugdata);
            clear_block_chain(&mut self.lastblock);
        }
        if unsafe { crate::w_code_frame_stores_global(code as PyObjectRef, w_globals) } {
            self.getorcreate_debug_data(-1).w_globals = w_globals;
        }
        self.pending_inline_results.clear();
        self.pending_inline_resume_pc = None;
        self.initialize_frame_scopes(outer_func, code);
    }

    /// PyPy-compatible `__repr__`.
    #[inline]
    pub fn __repr__(&self) -> String {
        format!(
            "<{}.{} executing {} at line {}>",
            module_path!(),
            std::any::type_name::<Self>()
                .rsplit("::")
                .next()
                .unwrap_or("PyFrame"),
            self.code().obj_name.as_str(),
            self.get_last_lineno()
        )
    }

    /// PyPy-compatible `fget_getdictscope`.
    #[inline]
    pub fn fget_getdictscope(&mut self) -> *mut DictStorage {
        self.getdictscope()
    }

    /// PyPy-compatible `fget_w_globals`.
    #[inline]
    pub fn fget_w_globals(&self) -> *mut DictStorage {
        self.get_w_globals()
    }

    /// PyPy-compatible `_getcell`.
    #[inline]
    pub fn _getcell(&self, varindex: usize) -> PyObjectRef {
        self.locals_w()
            .as_slice()
            .get(self.nlocals() + varindex)
            .copied()
            .unwrap_or(PY_NULL)
    }

    /// PyPy-compatible `getclosure`.
    #[inline]
    pub fn getclosure(&self) -> PyObjectRef {
        PY_NULL
    }

    /// PyPy-compatible `initialize_frame_scopes`.
    #[inline]
    pub fn initialize_frame_scopes(&mut self, outer_func: PyObjectRef, _code: *const ()) {
        let code = unsafe { &*pyframe_get_pycode(self) };
        let flags = code.flags;
        if !flags.contains(CodeFlags::OPTIMIZED) {
            let w_locals = if flags.contains(CodeFlags::NEWLOCALS) {
                pyre_object::lltype::malloc_raw(DictStorage::new())
            } else {
                self.get_w_globals()
            };
            self.getorcreate_debug_data(-1).w_locals = w_locals;
        }

        let ncellvars = code.cellvars.len();
        let nfreevars = code.freevars.len();
        if ncellvars == 0 && nfreevars == 0 {
            return;
        }

        let closure = if !outer_func.is_null() && unsafe { crate::is_function(outer_func) } {
            unsafe { crate::function_get_closure(outer_func) }
        } else {
            PY_NULL
        };
        let closure_size = if closure.is_null() {
            0
        } else {
            unsafe { w_tuple_len(closure) }
        };
        assert!(
            nfreevars == 0 || !outer_func.is_null(),
            "directly executed code object may not contain free variables"
        );
        assert!(
            closure_size == nfreevars,
            "code object received a closure with an unexpected number of free variables"
        );

        let mut index = code.varnames.len();
        for _ in 0..ncellvars {
            self.locals_w_mut()[index] = pyre_object::w_cell_new(PY_NULL);
            index += 1;
        }
        for i in 0..nfreevars {
            self.locals_w_mut()[index] =
                unsafe { w_tuple_getitem(closure, i as i64).unwrap_or(PY_NULL) };
            index += 1;
        }
    }

    /// pyframe.py:547-552 setdictscope(w_locals, skip_free_vars=False)
    #[inline]
    pub fn setdictscope(&mut self, w_locals: *mut DictStorage) {
        self.setdictscope_with_options(w_locals, false);
    }

    /// pyframe.py:547-552 setdictscope(w_locals, skip_free_vars=False)
    #[inline]
    pub fn setdictscope_with_options(&mut self, w_locals: *mut DictStorage, skip_free_vars: bool) {
        self.getorcreate_debug_data(-1).w_locals = w_locals;
        self.locals2fast(skip_free_vars);
    }

    /// Create a minimal frame stub for passing to call dispatch.
    /// Used by MIFrame Box tracking when concrete_frame is unavailable.
    pub fn new_minimal(
        code: *const (),
        w_globals: *mut crate::DictStorage,
        execution_context: *const PyExecutionContext,
    ) -> Self {
        let raw =
            unsafe { crate::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject };
        let nlocals = unsafe { (&*raw).varnames.len() };
        let ncells = unsafe { (&*raw).cellvars.len() + (&*raw).freevars.len() };
        let size = nlocals + ncells + 16; // small stack
        let stores_global =
            unsafe { crate::w_code_frame_stores_global(code as PyObjectRef, w_globals) };
        let mut frame = PyFrame {
            execution_context,
            pycode: code,
            locals_cells_stack_w: unsafe {
                alloc_fixed_array_from_vec(vec![pyre_object::PY_NULL; size])
            },
            valuestackdepth: nlocals + ncells,
            last_instr: -1,
            escaped: false,
            debugdata: std::ptr::null_mut(),
            lastblock: std::ptr::null_mut(),
            w_globals,
            vable_token: 0,
            frame_finished_execution: false,
            f_generator_nowref: PY_NULL,
            w_yielding_from: PY_NULL,
            f_backref: std::ptr::null_mut(),
            pending_inline_results: std::collections::VecDeque::new(),
            pending_inline_resume_pc: None,
        };
        if stores_global {
            frame.getorcreate_debug_data(-1).w_globals = w_globals;
        }
        frame
    }

    /// Create a new frame for executing a code object with a fresh execution context.
    pub fn new(code: CodeObject) -> Self {
        Self::new_with_context(code, Rc::new(PyExecutionContext::default()))
    }

    /// Create a new frame for executing a code object in the given context.
    ///
    /// The `Rc` is leaked via `Rc::into_raw` — consistent with pyre's
    /// memory model where code objects and namespaces are also leaked.
    pub fn new_with_context(code: CodeObject, execution_context: Rc<PyExecutionContext>) -> Self {
        let mut w_globals = Box::new(execution_context.fresh_dict_storage());
        w_globals.fix_ptr();
        // Set __name__ — PyPy: Module.__init__ sets __name__ in w_dict
        crate::dict_storage_store(
            &mut w_globals,
            "__name__",
            pyre_object::w_str_new("__main__"),
        );
        let w_globals = Box::into_raw(w_globals);
        let code_ptr = Box::into_raw(Box::new(code));
        let w_code = crate::w_code_new(code_ptr as *const ());
        unsafe {
            crate::w_code_set_w_globals(w_code, w_globals);
        }
        let ctx_ptr = Rc::into_raw(execution_context);
        Self::new_with_namespace(w_code as *const (), ctx_ptr, w_globals)
    }

    /// Create a new frame with an explicitly provided namespace pointer.
    pub fn new_with_namespace(
        code: *const (),
        execution_context: *const PyExecutionContext,
        w_globals: *mut DictStorage,
    ) -> Self {
        let raw =
            unsafe { crate::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject };
        let code_ref = unsafe { &*raw };
        let num_locals = code_ref.varnames.len();
        let num_cells = ncells(code_ref);
        let max_stack = code_ref.max_stackdepth as usize;

        let stores_global =
            unsafe { crate::w_code_frame_stores_global(code as PyObjectRef, w_globals) };
        let mut frame = PyFrame {
            execution_context,
            pycode: code,
            locals_cells_stack_w: unsafe {
                alloc_fixed_array_with_header(num_locals + num_cells + max_stack, PY_NULL)
            },
            valuestackdepth: num_locals + num_cells,
            last_instr: -1,
            escaped: false,
            debugdata: std::ptr::null_mut(),
            lastblock: std::ptr::null_mut(),
            w_globals,
            vable_token: 0,
            frame_finished_execution: false,
            f_generator_nowref: PY_NULL,
            w_yielding_from: PY_NULL,
            f_backref: std::ptr::null_mut(),
            pending_inline_results: VecDeque::new(),
            pending_inline_resume_pc: None,
        };
        if stores_global {
            frame.getorcreate_debug_data(-1).w_globals = w_globals;
        }
        // Module-level semantics (pypy/interpreter/module.py): locals == globals.
        // PyPy doesn't set NEWLOCALS on module code, so initialize_frame_scopes
        // does `w_locals = w_globals` for !OPTIMIZED && !NEWLOCALS. rustpython's
        // codegen, however, sets NEWLOCALS on module code too — so the
        // flag-based branch would hand us a fresh empty dict. new_with_namespace
        // is only used for the module-level entry frame, so force the PyPy
        // semantics here directly.
        frame.getorcreate_debug_data(-1).w_locals = w_globals;
        frame
    }

    /// RPython MetaInterp traces against its own MIFrame stack instead of
    /// mutating the live interpreter frame in place. pyre still executes
    /// bytecodes concretely during tracing, so use an owned snapshot when
    /// recording a trace to keep the real frame state unchanged until the
    /// interpreter actually executes the same path.
    pub fn snapshot_for_tracing(&self) -> Box<Self> {
        let mut frame = Box::new(PyFrame {
            execution_context: self.execution_context,
            pycode: self.pycode,
            locals_cells_stack_w: unsafe { alloc_fixed_array_from_vec(self.locals_w().to_vec()) },
            valuestackdepth: self.valuestackdepth,
            last_instr: self.last_instr,
            escaped: self.escaped,
            debugdata: unsafe { clone_debugdata_ptr(self.debugdata) },
            lastblock: unsafe { clone_block_chain(self.lastblock) },
            w_globals: self.get_w_globals(),
            vable_token: self.vable_token,
            frame_finished_execution: self.frame_finished_execution,
            f_generator_nowref: self.f_generator_nowref,
            w_yielding_from: self.w_yielding_from,
            f_backref: self.f_backref,
            pending_inline_results: self.pending_inline_results.clone(),
            pending_inline_resume_pc: self.pending_inline_resume_pc,
        });
        // fix_array_ptrs AFTER Box allocation: inline_buf ptr must
        // point to the heap-allocated frame, not a stale stack address.
        frame.fix_array_ptrs();
        frame
    }

    /// Number of local variable slots (from code object).
    #[inline]
    pub fn nlocals(&self) -> usize {
        unsafe { (&*pyframe_get_pycode(self)).varnames.len() }
    }

    /// Number of cell + free variable slots.
    #[inline]
    pub fn ncells(&self) -> usize {
        unsafe { ncells(&*pyframe_get_pycode(self)) }
    }

    /// First index of the operand stack (after locals and cells).
    #[inline]
    pub fn stack_base(&self) -> usize {
        self.nlocals() + self.ncells()
    }

    // ── Stack operations ──────────────────────────────────────────────

    #[inline]
    pub fn push(&mut self, value: PyObjectRef) {
        self.assert_stack_index(self.valuestackdepth);
        let idx = self.valuestackdepth;
        self.locals_w_mut()[idx] = value;
        self.valuestackdepth += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> PyObjectRef {
        assert!(self.valuestackdepth > self.stack_base());
        let depth = self.valuestackdepth - 1;
        let value = self.locals_w()[depth];
        self.locals_w_mut()[depth] = PY_NULL;
        self.valuestackdepth = depth;
        value
    }

    #[inline]
    pub fn peek(&self) -> PyObjectRef {
        self.locals_w()[self.valuestackdepth - 1]
    }

    #[inline]
    #[allow(dead_code)]
    pub fn peek_at(&self, depth: usize) -> PyObjectRef {
        self.locals_w()[self.valuestackdepth - 1 - depth]
    }

    /// PyPy-compatible stack operation aliases.
    #[inline]
    pub fn pushvalue(&mut self, value: PyObjectRef) {
        self.push(value)
    }

    /// pyframe.py:304-307 pushvalue_none
    #[inline]
    pub fn pushvalue_none(&mut self) {
        let depth = self.valuestackdepth;
        debug_assert!(self.locals_w()[depth].is_null());
        self.valuestackdepth = depth + 1;
    }

    /// PyPy-compatible stack index guard.
    #[inline]
    pub fn assert_stack_index(&self, index: usize) {
        debug_assert!(self._check_stack_index(index));
    }

    /// PyPy-compatible stack index validator.
    ///
    /// Asserts both lower and upper bounds: a valid stack write goes to
    /// `stack_base() <= index < locals_cells_stack_w.len()`. Pyre's
    /// `PyObjectArray` is allocated with `nlocals + ncells +
    /// max_stackdepth` slots (pyframe.rs:1091), so writing at or past
    /// `array_len` overruns the heap buffer — catastrophic in release
    /// mode, where `PyObjectArray` indexing is unchecked. This guard
    /// converts the heap overrun into a debug-mode assertion failure
    /// that surfaces tracer/JIT vsd miscalculations at the source
    /// rather than as silent malloc corruption later.
    #[inline]
    pub fn _check_stack_index(&self, index: usize) -> bool {
        index >= self.stack_base() && index < self.locals_w().len()
    }

    /// pyframe.py:313-314 popvalue
    #[inline]
    pub fn popvalue(&mut self) -> PyObjectRef {
        let value = self.popvalue_maybe_none();
        assert!(!value.is_null());
        value
    }

    /// pyframe.py:316-322 popvalue_maybe_none
    #[inline]
    pub fn popvalue_maybe_none(&mut self) -> PyObjectRef {
        let depth = self.valuestackdepth - 1;
        self.assert_stack_index(depth);
        let w_object = self.locals_w()[depth];
        self.locals_w_mut()[depth] = PY_NULL;
        self.valuestackdepth = depth;
        w_object
    }

    /// PyPy `PyFrame._new_popvalues` factory.
    #[inline]
    pub fn _new_popvalues() -> fn(&mut Self, usize) -> Vec<PyObjectRef> {
        Self::popvalues
    }

    /// PyPy-compatible pop-values helper.
    #[inline]
    pub fn popvalues(&mut self, n: usize) -> Vec<PyObjectRef> {
        let mut out = vec![PY_NULL; n];
        let mut idx = n;
        while idx > 0 {
            idx -= 1;
            out[idx] = self.popvalue();
        }
        out
    }

    /// PyPy-compatible `popvalues_mutable`.
    #[inline]
    pub fn popvalues_mutable(&mut self, n: usize) -> Vec<PyObjectRef> {
        self.popvalues(n)
    }

    /// pyframe.py:337-345 peekvalues
    #[inline]
    pub fn peekvalues(&self, n: usize) -> Vec<PyObjectRef> {
        let base = self.valuestackdepth - n;
        self.assert_stack_index(base);
        let mut values_w = vec![PY_NULL; n];
        let mut idx = n;
        while idx > 0 {
            idx -= 1;
            values_w[idx] = self.locals_w()[base + idx];
        }
        values_w
    }

    /// pyframe.py:348-355 dropvalues
    #[inline]
    pub fn dropvalues(&mut self, n: usize) {
        let finaldepth = self.valuestackdepth - n;
        self.assert_stack_index(finaldepth);
        while self.valuestackdepth > finaldepth {
            let idx = self.valuestackdepth - 1;
            self.locals_w_mut()[idx] = PY_NULL;
            self.valuestackdepth -= 1;
        }
    }

    /// PyPy-compatible `pushrevvalues`.
    #[inline]
    pub fn pushrevvalues(&mut self, _n: usize, values_w: &[PyObjectRef]) {
        let n = if _n == 0 { values_w.len() } else { _n };
        assert!(n <= values_w.len());
        let mut idx = n;
        while idx > 0 {
            idx -= 1;
            self.push(values_w[idx]);
        }
    }

    /// PyPy-compatible `dupvalues`.
    #[inline]
    pub fn dupvalues(&mut self, n: usize) {
        let values = self.peekvalues(n);
        for value in values {
            self.push(value);
        }
    }

    /// PyPy-compatible `peekvalue()`.
    #[inline]
    pub fn peekvalue(&self, index_from_top: usize) -> PyObjectRef {
        self.peek_at(index_from_top)
    }

    /// PyPy-compatible `peekvalue_maybe_none()`.
    #[inline]
    pub fn peekvalue_maybe_none(&self, index_from_top: usize) -> PyObjectRef {
        let index = self
            .valuestackdepth
            .checked_sub(index_from_top + 1)
            .unwrap_or(usize::MAX);
        if index == usize::MAX || index < self.stack_base() {
            return PY_NULL;
        }
        self.locals_w()[index]
    }

    /// PyPy-compatible `settopvalue()`.
    #[inline]
    pub fn settopvalue(&mut self, value: PyObjectRef, index_from_top: usize) {
        let index = self
            .valuestackdepth
            .checked_sub(index_from_top + 1)
            .unwrap_or(0);
        self.assert_stack_index(index);
        assert!(index < self.valuestackdepth);
        self.locals_w_mut()[index] = value;
    }

    /// PyPy-compatible `dropvaluesuntil()`.
    #[inline]
    pub fn dropvaluesuntil(&mut self, finaldepth: usize) {
        self.assert_stack_index(finaldepth);
        while self.valuestackdepth > finaldepth {
            let idx = self.valuestackdepth - 1;
            self.locals_w_mut()[idx] = PY_NULL;
            self.valuestackdepth -= 1;
        }
    }

    /// pyframe.py:186 append_block
    #[inline]
    pub fn append_block(&mut self, mut block: FrameBlock) {
        block.previous = self.lastblock;
        self.lastblock = pyre_object::lltype::malloc_raw(block);
    }

    /// pyframe.py:190 pop_block
    #[inline]
    pub fn pop_block(&mut self) -> Option<FrameBlock> {
        if self.lastblock.is_null() {
            return None;
        }
        unsafe {
            let block = Box::from_raw(self.lastblock);
            self.lastblock = block.previous;
            let mut result = *block;
            result.previous = std::ptr::null_mut();
            Some(result)
        }
    }

    /// pyframe.py:195 blockstack_non_empty
    #[inline]
    pub fn blockstack_non_empty(&self) -> bool {
        !self.lastblock.is_null()
    }

    /// PyPy-compatible exception-info unwind helper.
    #[inline]
    pub fn _exc_info_unroll(&self, _for_hidden: bool) -> PyObjectRef {
        let _ = _for_hidden;
        pyre_object::w_none()
    }

    /// PyPy-compatible unexpected-exception converter.
    #[inline]
    pub fn _convert_unexpected_exception(&self, _e: PyObjectRef) -> PyObjectRef {
        let _ = _e;
        pyre_object::w_none()
    }

    /// PyPy-compatible pickle state helper.
    #[inline]
    pub fn _reduce_state(&self) -> PyObjectRef {
        pyre_object::w_tuple_new(vec![
            pyre_object::w_none(),
            pyre_object::w_none(),
            pyre_object::w_none(),
            pyre_object::w_int_new(self.last_instr as i64),
            pyre_object::w_int_new(self.valuestackdepth as i64),
        ])
    }

    /// PyPy-compatible `descr__reduce__`.
    #[inline]
    pub fn descr__reduce__(&self) -> PyObjectRef {
        pyre_object::w_tuple_new(vec![
            pyre_object::w_none(),
            pyre_object::w_none(),
            self._reduce_state(),
        ])
    }

    /// PyPy-compatible `descr__setstate__`.
    #[inline]
    pub fn descr__setstate__(&mut self, _state: PyObjectRef) {
        let _ = _state;
    }

    /// pyframe.py:198 get_blocklist — walk linked list, return in reverse order.
    #[inline]
    pub fn get_blocklist(&self) -> Vec<FrameBlock> {
        let mut lst = Vec::new();
        let mut block = self.lastblock;
        while !block.is_null() {
            unsafe {
                let mut entry = *block;
                entry.previous = std::ptr::null_mut();
                lst.push(entry);
                block = (*block).previous;
            }
        }
        lst
    }

    /// pyframe.py:207 set_blocklist — rebuild linked list from slice.
    #[inline]
    pub fn set_blocklist(&mut self, lst: &[FrameBlock]) {
        unsafe { clear_block_chain(&mut self.lastblock) };
        let mut i = lst.len();
        while i > 0 {
            i -= 1;
            self.append_block(lst[i]);
        }
    }

    /// PyPy-compatible execution entrypoint.
    #[inline]
    pub fn run(&mut self) -> crate::PyResult {
        if self._is_generator_or_coroutine() {
            self.initialize_as_generator()
        } else {
            self.execute_frame(None, None)
        }
    }

    /// pyframe.py:300 resume_execute_frame (send-path only).
    ///
    /// pyre does not emit YIELD_FROM/SEND yet, so `w_yielding_from` is
    /// expected to remain null; asserting makes the gap visible instead of
    /// silently dropping the delegate. The SApplicationException branch
    /// (pyframe.py:320) is handled by the caller in `execute_frame`: if
    /// `operr.is_some()`, resume_execute_frame is skipped and
    /// `eval_frame_plain_with_operr` routes the error through
    /// `handle_exception` at `last_instr + 1`, matching PyPy's
    /// `handle_generator_error`.
    #[inline]
    pub fn resume_execute_frame(
        &mut self,
        w_arg_or_err: PyObjectRef,
    ) -> Result<usize, crate::PyError> {
        debug_assert!(
            self.w_yielding_from.is_null(),
            "YIELD_FROM delegation not yet ported; see pyframe.py:305-318",
        );
        if self.last_instr != -1 {
            self.pushvalue(w_arg_or_err);
            Ok(self.last_instr as usize + 1)
        } else {
            Ok(0)
        }
    }

    /// PyPy-compatible execution entrypoint with optional inbound values.
    #[inline]
    #[allow(unused_variables)]
    pub fn execute_frame(
        &mut self,
        w_inputvalue: Option<PyObjectRef>,
        operr: Option<crate::PyError>,
    ) -> crate::PyResult {
        if operr.is_none() {
            if let Some(w_arg_or_err) = w_inputvalue {
                let _ = self.resume_execute_frame(w_arg_or_err)?;
            }
        }
        crate::eval::eval_frame_plain_with_operr(self, operr)
    }

    /// PyPy-compatible `hide`.
    #[inline]
    pub fn hide(&self) -> bool {
        false
    }

    /// pyframe.py:183 mark_as_escaped
    #[inline]
    pub fn mark_as_escaped(&mut self) {
        self.escaped = true;
    }

    /// pyframe.py:200-204 get_builtin → space.builtin (same dict every call).
    #[inline]
    pub fn get_builtin(&self) -> PyObjectRef {
        if self.execution_context.is_null() {
            return pyre_object::PY_NULL;
        }
        unsafe { (*self.execution_context).get_builtin() }
    }

    /// PyPy-compatible `get_f_back`.
    #[inline]
    pub fn get_f_back(&self) -> *mut PyFrame {
        self.f_backref
    }

    /// PyPy-compatible `fget_f_builtins`.
    #[inline]
    pub fn fget_f_builtins(&self) -> PyObjectRef {
        self.get_builtin()
    }

    /// PyPy-compatible `fget_f_back`.
    #[inline]
    pub fn fget_f_back(&self) -> *mut PyFrame {
        self.get_f_back()
    }

    /// pyframe.py:773 fget_f_lasti → space.newint(self.last_instr)
    #[inline]
    pub fn fget_f_lasti(&self) -> isize {
        self.last_instr
    }

    /// PyPy-compatible `fget_f_trace`.
    #[inline]
    pub fn fget_f_trace(&self) -> PyObjectRef {
        self.get_w_f_trace()
    }

    /// pyframe.py:785-791 fset_f_trace
    #[inline]
    pub fn fset_f_trace(&mut self, w_trace: PyObjectRef) {
        if w_trace.is_null() || w_trace == pyre_object::w_none() {
            self.getorcreate_debug_data(-1).w_f_trace = pyre_object::PY_NULL;
        } else {
            let lineno = self.get_last_lineno();
            let d = self.getorcreate_debug_data(-1);
            d.w_f_trace = w_trace;
            d.f_lineno = lineno;
        }
    }

    /// pyframe.py:793-794 fdel_f_trace
    #[inline]
    pub fn fdel_f_trace(&mut self) {
        self.getorcreate_debug_data(-1).w_f_trace = pyre_object::PY_NULL;
    }

    /// pyframe.py:153-157 get_f_trace_lines
    #[inline]
    pub fn get_f_trace_lines(&self) -> bool {
        self.getdebug_data().map_or(true, |d| d.f_trace_lines)
    }

    /// pyframe.py:159-163 get_f_trace_opcodes
    #[inline]
    pub fn get_f_trace_opcodes(&self) -> bool {
        self.getdebug_data().map_or(false, |d| d.f_trace_opcodes)
    }

    /// pyframe.py:796-797 fget_f_trace_lines
    #[inline]
    pub fn fget_f_trace_lines(&self) -> bool {
        self.get_f_trace_lines()
    }

    /// pyframe.py:799-800 fset_f_trace_lines
    #[inline]
    pub fn fset_f_trace_lines(&mut self, value: bool) {
        self.getorcreate_debug_data(-1).f_trace_lines = value;
    }

    /// pyframe.py:802-803 fget_f_trace_opcodes
    #[inline]
    pub fn fget_f_trace_opcodes(&self) -> bool {
        self.get_f_trace_opcodes()
    }

    /// pyframe.py:805-806 fset_f_trace_opcodes
    #[inline]
    pub fn fset_f_trace_opcodes(&mut self, value: bool) {
        self.getorcreate_debug_data(-1).f_trace_opcodes = value;
    }

    /// PyPy-compatible `fget_f_exc_type`.
    #[inline]
    pub fn fget_f_exc_type(&self) -> PyObjectRef {
        pyre_object::PY_NULL
    }

    /// PyPy-compatible `fget_f_exc_value`.
    #[inline]
    pub fn fget_f_exc_value(&self) -> PyObjectRef {
        pyre_object::PY_NULL
    }

    /// PyPy-compatible `fget_f_exc_traceback`.
    #[inline]
    pub fn fget_f_exc_traceback(&self) -> PyObjectRef {
        pyre_object::PY_NULL
    }

    /// PyPy-compatible `fget_f_restricted`.
    #[inline]
    pub fn fget_f_restricted(&self) -> bool {
        false
    }

    /// pyframe.py:861-863 get_last_lineno → pytraceback.offset2lineno(pycode, last_instr)
    #[inline]
    pub fn get_last_lineno(&self) -> isize {
        offset2lineno(self.code(), self.last_instr) as isize
    }

    /// pyframe.py:660-671 fget_f_lineno
    #[inline]
    pub fn fget_f_lineno(&self) -> isize {
        if self.get_w_f_trace().is_null() {
            self.get_last_lineno()
        } else {
            let f_lineno = self.getdebug_data().map_or(-1, |dd| dd.f_lineno);
            if f_lineno == -1 {
                self.code()
                    .first_line_number
                    .map_or(-1, |n| n.get() as isize)
            } else {
                f_lineno
            }
        }
    }

    /// pyframe.py:680 fset_f_lineno (simplified — full version validates jumps)
    #[inline]
    pub fn fset_f_lineno(&mut self, new_f_lineno: isize) {
        self.getorcreate_debug_data(-1).f_lineno = new_f_lineno;
    }

    /// PyPy-compatible `setfastscope`.
    #[inline]
    pub fn setfastscope(&mut self, scope_w: &[PyObjectRef]) {
        assert!(scope_w.len() <= self.nlocals());
        for (index, value) in scope_w.iter().copied().enumerate() {
            self.locals_w_mut()[index] = value;
        }
        // In this port, cell initialization is performed as part of scope load.
        self.init_cells();
    }

    /// pyframe.py:601-636 locals2fast(skip_free_vars=False)
    pub fn locals2fast(&mut self, skip_free_vars: bool) {
        let w_locals = self.getorcreate_debug_data(-1).w_locals;
        assert!(!w_locals.is_null());
        let w_locals_ref = unsafe { &*w_locals };

        let code_ptr = unsafe { pyframe_get_pycode(self) };
        let code = unsafe { &*code_ptr };
        let numlocals = code.varnames.len();

        // pyframe.py:609-615: copy locals from dict to fast slots
        let mut new_fastlocals_w = vec![PY_NULL; numlocals];
        for i in 0..numlocals {
            let name = &code.varnames[i];
            if let Some(&w_value) = w_locals_ref.get(name.as_ref()) {
                new_fastlocals_w[i] = w_value;
            }
        }
        self.setfastscope(&new_fastlocals_w);

        // pyframe.py:619-636: freevarnames = co_cellvars
        // if CO_OPTIMIZED and not skip_free_vars: freevarnames += co_freevars
        let ncellvars = code.cellvars.len();
        let include_freevars = code.flags.contains(CodeFlags::OPTIMIZED) && !skip_free_vars;
        let freevarnames_len = if include_freevars {
            ncellvars + code.freevars.len()
        } else {
            ncellvars
        };
        for i in 0..freevarnames_len {
            let (name, idx) = if i < ncellvars {
                (&code.cellvars[i], numlocals + i)
            } else {
                (&code.freevars[i - ncellvars], numlocals + i)
            };
            if idx < self.locals_w().len() {
                let w_value = w_locals_ref.get(name.as_ref()).copied().unwrap_or(PY_NULL);
                // pyframe.py:632-634: cell.set(w_value) / cell.set(None)
                let slot = self.locals_w()[idx];
                if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
                    unsafe { pyre_object::w_cell_set(slot, w_value) };
                } else {
                    self.locals_w_mut()[idx] = w_value;
                }
            }
        }
    }

    /// pyframe.py:640-651 init_cells
    #[inline]
    pub fn init_cells(&mut self) {
        let code = unsafe { &*pyframe_get_pycode(self) };
        let mut argcount = code.arg_count as usize + code.kwonlyarg_count as usize;
        if code.flags.contains(CodeFlags::VARARGS) {
            argcount += 1;
        }
        if code.flags.contains(CodeFlags::VARKEYWORDS) {
            argcount += 1;
        }
        let args_to_copy =
            crate::pycode::_compute_args_as_cellvars(&code.varnames, &code.cellvars, argcount);
        let mut index = code.varnames.len(); // co_nlocals
        for i in 0..args_to_copy.len() {
            let argnum = args_to_copy[i];
            if argnum >= 0 {
                let cell = self.locals_w()[index];
                let val = self.locals_w()[argnum as usize];
                if !cell.is_null() && unsafe { pyre_object::is_cell(cell) } {
                    unsafe { pyre_object::w_cell_set(cell, val) };
                } else {
                    self.locals_w_mut()[index] = val;
                }
            }
            index += 1;
        }
    }

    /// pyframe.py:554-598 fast2locals
    pub fn fast2locals(&mut self) {
        let d = self.getorcreate_debug_data(-1);
        let mut w_locals = d.w_locals;
        let mut write = false;
        if w_locals.is_null() {
            w_locals = pyre_object::lltype::malloc_raw(DictStorage::new());
            write = true;
        }
        let w_locals_ref = unsafe { &mut *w_locals };

        let code_ptr = unsafe { pyframe_get_pycode(self) };
        let code = unsafe { &*code_ptr };
        let varnames = &code.varnames;
        let numlocals = varnames.len();

        // pyframe.py:564-575: copy local variables
        for i in 0..numlocals {
            let name = &varnames[i];
            let w_value = self.locals_w()[i];
            if !w_value.is_null() {
                w_locals_ref.insert(name.to_string(), w_value);
            } else {
                // pyframe.py:571-574: space.delitem(w_locals, w_name)
                w_locals_ref.remove(name.as_ref());
            }
        }

        // pyframe.py:580-581: freevarnames = co_cellvars
        // if CO_OPTIMIZED: freevarnames += co_freevars
        let ncellvars = code.cellvars.len();
        let include_freevars = code.flags.contains(CodeFlags::OPTIMIZED);
        let freevarnames_len = if include_freevars {
            ncellvars + code.freevars.len()
        } else {
            ncellvars
        };
        // pyframe.py:584-596: copy cell/free variables
        for i in 0..freevarnames_len {
            let (name, idx) = if i < ncellvars {
                (&code.cellvars[i], numlocals + i)
            } else {
                (&code.freevars[i - ncellvars], numlocals + i)
            };
            if idx < self.locals_w().len() {
                // pyframe.py:586: cell = self._getcell(i)
                let slot = self.locals_w()[idx];
                // pyframe.py:588: w_value = cell.get() — dereference cell
                let w_value = if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
                    unsafe { pyre_object::w_cell_get(slot) }
                } else {
                    slot
                };
                if !w_value.is_null() {
                    w_locals_ref.insert(name.to_string(), w_value);
                } else {
                    // pyframe.py:589-593: space.delitem(w_locals, w_name)
                    w_locals_ref.remove(name.as_ref());
                }
            }
        }

        if write {
            self.getorcreate_debug_data(-1).w_locals = w_locals;
        }
    }

    /// PyPy-compatible `make_arguments`.
    #[inline]
    pub fn make_arguments(&self, nargs: usize, _methodcall: bool) -> Vec<PyObjectRef> {
        self.peekvalues(nargs)
    }

    /// PyPy-compatible argument list builder.
    #[inline]
    #[allow(unused_variables)]
    pub fn argument_factory(
        &self,
        _arguments: &[PyObjectRef],
        _keywords: &[PyObjectRef],
        _keywords_w: &[PyObjectRef],
        _w_star: PyObjectRef,
        _w_starstar: PyObjectRef,
        _methodcall: bool,
    ) -> Vec<PyObjectRef> {
        let mut args = Vec::new();
        args.extend_from_slice(_arguments);
        args.extend_from_slice(_keywords);
        args.extend_from_slice(_keywords_w);
        if !_w_star.is_null() {
            args.push(_w_star);
        }
        if !_w_starstar.is_null() {
            args.push(_w_starstar);
        }
        args
    }

    /// Create a new frame for a function call.
    ///
    /// The `globals` pointer is shared from the function object -- no clone.
    /// The `code` pointer is shared from the function object -- no clone.
    /// `closure` is a tuple of cell objects from the enclosing scope,
    /// or PY_NULL if the function has no free variables.
    pub fn new_for_call(
        code: *const (),
        args: &[PyObjectRef],
        globals: *mut DictStorage,
        execution_context: *const PyExecutionContext,
    ) -> Self {
        Self::new_for_call_with_closure(code, args, globals, execution_context, PY_NULL)
    }

    /// Create a new frame for a function call with a closure.
    pub fn new_for_call_with_closure(
        code: *const (),
        args: &[PyObjectRef],
        globals: *mut DictStorage,
        execution_context: *const PyExecutionContext,
        closure: PyObjectRef,
    ) -> Self {
        let code_ref = unsafe {
            &*(crate::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject)
        };
        let num_locals = code_ref.varnames.len();
        let num_cells = ncells(code_ref);
        let max_stack = code_ref.max_stackdepth as usize;

        let locals_cells_stack_w =
            unsafe { alloc_fixed_array_with_header(num_locals + num_cells + max_stack, PY_NULL) };

        {
            // Populate the freshly-allocated array via its mutable slice.
            let arr = unsafe { &mut *locals_cells_stack_w };

            // Bind positional arguments directly -- no intermediate Vec.
            let nargs = args.len().min(num_locals);
            for i in 0..nargs {
                arr[i] = args[i];
            }

            // pyframe.py:229-236: copy free variables from closure into freevar slots
            for i in 0..code_ref.cellvars.len() {
                arr[num_locals + i] = pyre_object::w_cell_new(PY_NULL);
            }
            if !closure.is_null() {
                let ncellvars = code_ref.cellvars.len();
                let nfreevars = code_ref.freevars.len();
                for i in 0..nfreevars {
                    let cell = unsafe { w_tuple_getitem(closure, i as i64).unwrap() };
                    arr[num_locals + ncellvars + i] = cell;
                }
            }
        }

        let stores_global =
            unsafe { crate::w_code_frame_stores_global(code as PyObjectRef, globals) };

        let mut frame = PyFrame {
            execution_context,
            pycode: code,
            locals_cells_stack_w,
            valuestackdepth: num_locals + num_cells,
            last_instr: -1,
            escaped: false,
            debugdata: std::ptr::null_mut(),
            lastblock: std::ptr::null_mut(),
            w_globals: globals,
            vable_token: 0,
            frame_finished_execution: false,
            f_generator_nowref: PY_NULL,
            w_yielding_from: PY_NULL,
            f_backref: std::ptr::null_mut(),
            pending_inline_results: VecDeque::new(),
            pending_inline_resume_pc: None,
        };
        frame.init_cells();
        if stores_global {
            frame.getorcreate_debug_data(-1).w_globals = globals;
        }
        frame
    }

    /// Borrow the raw CodeObject.
    #[inline]
    pub fn code(&self) -> &CodeObject {
        unsafe { &*pyframe_get_pycode(self) }
    }

    #[inline]
    pub fn _is_generator_or_coroutine(&self) -> bool {
        self.code().flags.intersects(
            crate::CodeFlags::GENERATOR
                | crate::CodeFlags::COROUTINE
                | crate::CodeFlags::ITERABLE_COROUTINE,
        )
    }

    /// pyframe.py:276 initialize_as_generator
    ///
    /// Adaptation: pyre builds the caller's PyFrame on the interpreter stack,
    /// so we snapshot it onto the heap before handing ownership to the
    /// generator object. The backref (`f_generator_nowref`) is set on that
    /// heap-owned snapshot — not on the temporary caller frame — so later
    /// `get_generator()` calls through the surviving frame pointer return
    /// the right object.
    #[inline]
    pub fn initialize_as_generator(&mut self) -> crate::PyResult {
        let mut gen_frame = self.snapshot_for_tracing();
        gen_frame.fix_array_ptrs();
        let gen_frame_ptr = Box::into_raw(gen_frame);
        let generator = pyre_object::generatorobject::w_generator_new(gen_frame_ptr as *mut u8);
        unsafe {
            (*gen_frame_ptr).f_generator_nowref = generator;
        }
        Ok(generator)
    }

    #[inline]
    pub fn get_generator(&self) -> PyObjectRef {
        self.f_generator_nowref
    }

    /// Returns RPython's `last_instr` (= `next_instr`) value, derived from
    /// pyre's CPython-aligned storage. See the field comment on `last_instr`
    /// for the convention difference. PyPy `pyopcode.py:172` would read this
    /// field directly; pyre derives it because storage matches `f_lasti`.
    #[inline]
    pub fn next_instr(&self) -> usize {
        if self.last_instr < 0 {
            0
        } else {
            self.last_instr as usize + 1
        }
    }

    /// Inverse of `next_instr()`. Stores `next_instr - 1` so the field
    /// continues to satisfy `f_lasti = last_instr` (CPython convention).
    /// RPython would write `next_instr` directly here.
    #[inline]
    pub fn set_last_instr_from_next_instr(&mut self, next_instr: usize) {
        self.last_instr = next_instr as isize - 1;
    }

    /// Repoint internal array pointers after a struct move.
    ///
    /// `PyObjectArray` with small-buffer optimization stores an inline
    /// buffer whose address changes on move. Call this once after the
    /// frame is at its final stack location.
    #[inline]
    pub fn fix_array_ptrs(&mut self) {
        // locals_cells_stack_w is now a heap-allocated pointer;
        // fix_ptr on PyObjectArray is a no-op.
    }

    /// Load a constant from the code object by raw index.
    /// Used by the blackhole interpreter's bh_load_const_fn.
    pub fn load_const_pyobj(&self, idx: usize) -> PyObjectRef {
        let code = self.code();
        // RPython: constants are in JitCode.constants_r. In pyre, we resolve
        // from the CodeObject's constant table at runtime.
        let constants = code_constants(code);
        if idx >= constants.len() {
            return pyre_object::w_none();
        }
        pyobject_from_constant(&constants[idx])
    }
}

/// Load a constant from a CodeObject without a PyFrame.
/// Used by the blackhole's bh_load_const_fn when the code pointer
/// comes from a virtualizable field read.
pub fn load_const_from_code(code: &CodeObject, idx: usize) -> PyObjectRef {
    let constants = code_constants(code);
    if idx >= constants.len() {
        return pyre_object::w_none();
    }
    pyobject_from_constant(&constants[idx])
}

fn code_constants(code: &CodeObject) -> &[crate::bytecode::ConstantData] {
    unsafe {
        std::slice::from_raw_parts(
            code.constants.as_ptr() as *const crate::bytecode::ConstantData,
            code.constants.len(),
        )
    }
}

fn pyobject_from_constant(constant: &crate::bytecode::ConstantData) -> PyObjectRef {
    use crate::bytecode::ConstantData;
    use num_traits::ToPrimitive;
    match constant {
        ConstantData::Integer { value } => {
            pyre_object::intobject::w_int_new(value.to_i64().unwrap_or(0))
        }
        ConstantData::Float { value } => pyre_object::floatobject::w_float_new(*value),
        ConstantData::Boolean { value } => {
            pyre_object::intobject::w_int_new(if *value { 1 } else { 0 })
        }
        ConstantData::Str { value } => pyre_object::w_str_new(value.as_str().unwrap_or("")),
        ConstantData::None => pyre_object::w_none(),
        ConstantData::Ellipsis => pyre_object::noneobject::w_ellipsis(),
        _ => pyre_object::w_none(),
    }
}

// Virtualizable configuration is in jit/frame_layout.rs

#[cfg(test)]
mod tests {
    use super::load_const_from_code;

    #[test]
    fn load_const_from_code_returns_ellipsis_singleton() {
        let code = crate::compile_eval("...");
        let code = code.expect("compile should succeed");
        let ellipsis_index = super::code_constants(&code)
            .iter()
            .position(|constant| matches!(constant, crate::bytecode::ConstantData::Ellipsis))
            .expect("compiled code should contain an Ellipsis constant");

        let loaded = load_const_from_code(&code, ellipsis_index);
        assert_eq!(loaded, pyre_object::noneobject::w_ellipsis());
    }
}
