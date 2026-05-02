use pyre_object::PyObjectRef;
use pyre_object::object_array::{
    ITEMS_BLOCK_ITEMS_OFFSET, ItemsBlock, alloc_list_items_block, dealloc_list_items_block,
    grow_list_items_block, items_block_capacity, items_block_items_base,
};

use crate::{PyFrame, new_builtin_dict_storage};

/// Python-level entrypoint for profiling callback compatibility.
/// The PyPy implementation calls through to `space.call_function()` with
/// `(w_callable, frame, event, w_arg)`.
pub fn app_profile_call(
    _space: PyObjectRef,
    _w_callable: PyObjectRef,
    _frame: *mut PyFrame,
    _event: &str,
    _w_arg: PyObjectRef,
) {
    let _ = (_space, _w_callable, _frame, _event, _w_arg);
}

/// Byte offset of the GC-owning `values` pointer inside `DictStorage`.
/// Post-L1 DictStorage holds a `*mut ItemsBlock` directly (no fat
/// wrapper); the JIT reads this field to obtain the items-block
/// pointer, then adds `ITEMS_BLOCK_ITEMS_OFFSET` for the items base.
pub const DICT_STORAGE_VALUES_OFFSET: usize = std::mem::offset_of!(DictStorage, values);

/// Byte offset of the live dict slot count inside `DictStorage`.
/// Post-L1 reads `DictStorage.length` directly (upstream
/// `rdict.py` `l.length` equivalent for the values array).
pub const DICT_STORAGE_VALUES_LEN_OFFSET: usize = std::mem::offset_of!(DictStorage, length);

/// Internal dict backing used for globals, module dicts, and type dicts.
///
/// PyPy correspondence: this is not `cpyext._PyNamespace_New()` or
/// `types.SimpleNamespace`. It is pyre's internal storage used where PyPy keeps
/// string-keyed dict state such as `w_globals`, module dictionaries backed by
/// `W_DictMultiObject`/`ModuleDictStrategy`, and type-level `dict_w`.
///
/// Names are stored in insertion order. Values live in an `ItemsBlock`
/// GcArray body with an `(length, items)` pair matching upstream
/// `rpython/rtyper/lltypesystem/rlist.py:116`
/// `GcStruct("list", ("length", Signed), ("items", Ptr(ITEMARRAY)))`.
/// The JIT reads `length` at a fixed offset and combines `values` with
/// `ITEMS_BLOCK_ITEMS_OFFSET` for the items base pointer.
#[repr(C)]
pub struct DictStorage {
    names: Vec<String>,
    /// Number of live entries. Matches upstream `l.length` (rlist.py:
    /// 116).
    length: usize,
    /// `Ptr(GcArray(OBJECTPTR))` — `l.items` (rlist.py:116). Points
    /// at the `ItemsBlock` whose offset-0 header is the allocated
    /// capacity (= upstream `len(l.items)` per rlist.py:251). Never
    /// null — `DictStorage::new()` seeds a 1-slot block.
    values: *mut ItemsBlock,
    /// Per-slot JIT invalidation watchers.
    /// RPython quasiimmut.py parity: each dict entry has its own
    /// QuasiImmut watcher list. Only loops that depend on a specific
    /// slot are invalidated when that slot is overwritten.
    slot_watchers: Vec<Vec<std::sync::Weak<std::sync::atomic::AtomicBool>>>,
}

impl Clone for DictStorage {
    fn clone(&self) -> Self {
        let snapshot = unsafe {
            let base = items_block_items_base(self.values);
            std::slice::from_raw_parts(base, self.length).to_vec()
        };
        let values = unsafe { alloc_list_items_block(&snapshot) };
        Self {
            names: self.names.clone(),
            length: snapshot.len(),
            values,
            // Cloned storages start with no registered invalidation watchers,
            // but the per-slot shape must stay aligned with names/values.
            slot_watchers: vec![Vec::new(); self.names.len()],
        }
    }
}

impl Default for DictStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl DictStorage {
    pub fn new() -> Self {
        let values = unsafe { alloc_list_items_block(&[]) };
        Self {
            names: Vec::new(),
            length: 0,
            values,
            slot_watchers: Vec::new(),
        }
    }

    /// Reallocate `values` to fit at least `min_cap` entries, copying
    /// the live prefix. Upstream `_ll_list_resize_really`
    /// (rlist.py:262-267) parity.
    unsafe fn grow(&mut self, min_cap: usize) {
        let current_cap = items_block_capacity(self.values);
        let target_cap = min_cap.max(current_cap.saturating_mul(2).max(4));
        self.values = grow_list_items_block(self.values, target_cap, self.length);
    }

    unsafe fn push(&mut self, value: PyObjectRef) {
        if self.length == items_block_capacity(self.values) {
            self.grow(self.length + 1);
        }
        let base = items_block_items_base(self.values);
        *base.add(self.length) = value;
        self.length += 1;
    }

    unsafe fn remove_at(&mut self, idx: usize) -> PyObjectRef {
        assert!(idx < self.length);
        let base = items_block_items_base(self.values);
        let value = *base.add(idx);
        let p = base.add(idx);
        std::ptr::copy(p.add(1), p, self.length - idx - 1);
        self.length -= 1;
        value
    }

    unsafe fn values_slice(&self) -> &[PyObjectRef] {
        let base = items_block_items_base(self.values);
        std::slice::from_raw_parts(base, self.length)
    }

    unsafe fn values_slice_mut(&mut self) -> &mut [PyObjectRef] {
        let base = items_block_items_base(self.values);
        std::slice::from_raw_parts_mut(base, self.length)
    }

    /// Replace the entire values backing with a freshly allocated
    /// empty block. Used by `clear()`.
    unsafe fn reset_values(&mut self) {
        dealloc_list_items_block(self.values);
        self.values = alloc_list_items_block(&[]);
        self.length = 0;
    }

    #[inline]
    pub fn fix_ptr(&mut self) {
        // Post-L1 `values` is a direct `*mut ItemsBlock` — no moves,
        // no interior pointer to rebase. Retained as a no-op for
        // source-compatibility with existing callers.
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    #[inline]
    pub fn slot_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|candidate| candidate == name)
    }

    pub fn get(&self, name: &str) -> Option<&PyObjectRef> {
        let idx = self.slot_of(name)?;
        Some(unsafe { &self.values_slice()[idx] })
    }

    /// Iterate over (name, value) pairs, skipping tombstoned slots.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &PyObjectRef)> {
        let values = unsafe { self.values_slice() };
        self.names
            .iter()
            .zip(values.iter())
            .filter(|(name, _)| !name.is_empty())
            .map(|(name, value)| (name.as_str(), value))
    }

    #[inline]
    pub fn get_slot(&self, idx: usize) -> Option<PyObjectRef> {
        unsafe { self.values_slice().get(idx).copied() }
    }

    #[inline]
    pub fn values_mut(&mut self) -> &mut [PyObjectRef] {
        unsafe { self.values_slice_mut() }
    }

    pub fn get_or_insert_with(
        &mut self,
        name: &str,
        make: impl FnOnce() -> PyObjectRef,
    ) -> PyObjectRef {
        if let Some(idx) = self.slot_of(name) {
            return unsafe { self.values_slice()[idx] };
        }
        let value = make();
        self.names.push(name.to_string());
        unsafe { self.push(value) };
        self.slot_watchers.push(Vec::new());
        value
    }

    pub fn insert(&mut self, name: String, value: PyObjectRef) -> Option<PyObjectRef> {
        if let Some(idx) = self.slot_of(&name) {
            let slice = unsafe { self.values_slice_mut() };
            let old = slice[idx];
            slice[idx] = value;
            if old != value {
                self.notify_slot_watchers(idx);
            }
            Some(old)
        } else {
            self.names.push(name);
            unsafe { self.push(value) };
            self.slot_watchers.push(Vec::new());
            None
        }
    }

    /// Remove a key from the backing dict storage (PyPy: space.delitem).
    /// This performs a real deletion rather than leaving a tombstone.
    pub fn remove(&mut self, name: &str) -> Option<PyObjectRef> {
        if let Some(idx) = self.slot_of(name) {
            self.notify_slot_watchers(idx);
            self.names.remove(idx);
            let old = unsafe { self.remove_at(idx) };
            self.slot_watchers.remove(idx);
            Some(old)
        } else {
            None
        }
    }

    #[inline]
    pub fn set_slot(&mut self, idx: usize, value: PyObjectRef) -> bool {
        let slice = unsafe { self.values_slice_mut() };
        let Some(slot) = slice.get_mut(idx) else {
            return false;
        };
        let old = *slot;
        *slot = value;
        if old != value {
            self.notify_slot_watchers(idx);
        }
        true
    }

    /// Register a JIT invalidation watcher for a specific slot.
    /// RPython quasiimmut.py:register_loop_token parity: each dict
    /// entry has its own watcher list, so only loops depending on
    /// this slot are invalidated when it changes.
    pub fn register_slot_watcher(
        &mut self,
        slot: usize,
        flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) {
        // Grow slot_watchers if needed (slots added before JIT was active).
        while self.slot_watchers.len() <= slot {
            self.slot_watchers.push(Vec::new());
        }
        self.slot_watchers[slot].push(std::sync::Arc::downgrade(flag));
    }

    /// RPython quasiimmut.py:invalidate parity.
    fn notify_slot_watchers(&mut self, slot: usize) {
        let Some(watchers) = self.slot_watchers.get_mut(slot) else {
            return;
        };
        if watchers.is_empty() {
            return;
        }
        watchers.retain(|w| {
            if let Some(flag) = w.upgrade() {
                flag.store(true, std::sync::atomic::Ordering::Release);
                true
            } else {
                false
            }
        });
    }

    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.names.iter().filter(|n| !n.is_empty())
    }

    /// Remove all entries from the namespace.
    #[inline]
    pub fn clear(&mut self) {
        self.names.clear();
        unsafe { self.reset_values() };
        self.slot_watchers.clear();
    }
}

impl Drop for DictStorage {
    fn drop(&mut self) {
        unsafe { dealloc_list_items_block(self.values) };
    }
}

const TICK_COUNTER_STEP: usize = 100;

#[derive(Debug, Default)]
pub struct WRootFinalizerQueue;

impl WRootFinalizerQueue {
    pub fn finalizer_trigger(&mut self) {}
}

/// Shared execution context for all frames in one interpreter run.
///
/// Holds the builtin dict storage seed. Module-level frames call
/// `fresh_dict_storage()` once to create a leaked globals dict;
/// function calls share the globals pointer without cloning.
#[derive(Clone)]
pub struct ExecutionContext {
    pub space: PyObjectRef,
    pub topframeref: *mut PyFrame,
    pub w_tracefunc: PyObjectRef,
    pub is_tracing: i32,
    pub compiler: PyObjectRef,
    pub profilefunc: PyObjectRef,
    pub w_profilefuncarg: PyObjectRef,
    pub w_profilefuncarg_ref: PyObjectRef,
    pub thread_disappeared: bool,
    pub w_async_exception_type: PyObjectRef,
    pub actionflag: ActionFlag,
    builtins: DictStorage,
    /// Cached dict wrapper over `self.builtins` — pyframe.py:200-204
    /// `space.builtin` returns the same object every call.
    builtin_dict_cache: std::cell::Cell<PyObjectRef>,
    pub check_signal_action: Option<PyObjectRef>,
}

pub type PyExecutionContext = ExecutionContext;

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext {
    #[inline]
    pub fn new() -> Self {
        Self {
            space: pyre_object::PY_NULL,
            topframeref: std::ptr::null_mut(),
            w_tracefunc: pyre_object::PY_NULL,
            is_tracing: 0,
            compiler: pyre_object::PY_NULL,
            profilefunc: pyre_object::PY_NULL,
            w_profilefuncarg: pyre_object::PY_NULL,
            w_profilefuncarg_ref: pyre_object::PY_NULL,
            thread_disappeared: false,
            w_async_exception_type: pyre_object::PY_NULL,
            actionflag: ActionFlag::new(),
            builtins: new_builtin_dict_storage(),
            builtin_dict_cache: std::cell::Cell::new(pyre_object::PY_NULL),
            check_signal_action: None,
        }
    }

    pub fn __init__(&mut self, space: PyObjectRef) {
        self.space = space;
        self.compiler = pyre_object::w_none();
    }

    #[inline]
    pub fn _mark_thread_disappeared(_space: PyObjectRef) {
        let _ = _space;
    }

    #[inline]
    pub fn gettopframe(&self) -> *mut PyFrame {
        self.topframeref
    }

    pub fn gettopframe_nohidden(&self) -> *mut PyFrame {
        let mut frame = self.topframeref;
        while !frame.is_null() {
            // SAFETY: frame pointers are owned by interpreter call stack and can be
            // null-checked before dereference.
            unsafe {
                let current = &*frame;
                if !current.hide() {
                    return frame;
                }
                frame = current.get_f_back();
            }
        }
        frame
    }

    pub fn getnextframe_nohidden(mut frame: *mut PyFrame) -> *mut PyFrame {
        while !frame.is_null() {
            // SAFETY: caller provides a valid frame chain or null.
            unsafe {
                let current = &*frame;
                let next = current.get_f_back();
                if next.is_null() {
                    return std::ptr::null_mut();
                }
                if !(&*next).hide() {
                    return next;
                }
                frame = next;
            }
        }
        frame
    }

    pub fn enter(&mut self, frame: *mut PyFrame) {
        if !self.space.is_null() && self.is_tracing > 0 {
            self._revdb_enter(frame);
        }
        unsafe {
            (*frame).f_backref = self.topframeref;
        }
        self.topframeref = frame;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn leave(&mut self, frame: *mut PyFrame, w_exitvalue: PyObjectRef, got_exception: bool) {
        // pypy/interpreter/executioncontext.py:91-109 leave parity.
        // The Python `try/finally` collapses into linear code here because
        // `_trace` cannot raise — it's a no-op stub.
        if !self.profilefunc.is_null() {
            self._trace(frame, "leaveframe", w_exitvalue, None);
        }
        // `frame_vref = self.topframeref` — vref currently a raw frame
        // pointer; the `vref()` force at the bottom of the if-block is a
        // no-op until `jit.virtual_ref` ports.
        let _frame_vref = self.topframeref;
        unsafe {
            self.topframeref = (*frame).f_backref;
            // pypy/interpreter/executioncontext.py:98-106: if the frame
            // escaped to applevel (or unwinding via exception), the back-
            // chain must also be marked so consumers walking f_back stop
            // treating it as "still on the JIT stack".
            if (*frame).escaped || got_exception {
                let f_back = (*frame).get_f_back();
                if !f_back.is_null() {
                    (*f_back).mark_as_escaped();
                }
                // `frame_vref()` — vref force, no-op until `jit.virtual_ref`
                // is ported.
            }
        }
        // `jit.virtual_ref_finish(frame_vref, frame)` — no-op until
        // `jit.virtual_ref` is ported.
        // PyPy gates this on `self.space.reverse_debugging`; pyre has no
        // equivalent flag, so the (also no-op) call runs unconditionally.
        self._revdb_leave(got_exception);
    }

    pub fn c_call_trace(
        &mut self,
        frame: *mut PyFrame,
        w_func: PyObjectRef,
        args: Option<PyObjectRef>,
    ) {
        let args = args.unwrap_or(pyre_object::PY_NULL);
        self._c_call_return_trace(frame, w_func, args, "c_call");
    }

    pub fn c_return_trace(
        &mut self,
        frame: *mut PyFrame,
        w_func: PyObjectRef,
        args: Option<PyObjectRef>,
    ) {
        let args = args.unwrap_or(pyre_object::PY_NULL);
        self._c_call_return_trace(frame, w_func, args, "c_return");
    }

    pub fn _c_call_return_trace(
        &mut self,
        frame: *mut PyFrame,
        _w_func: PyObjectRef,
        _args: PyObjectRef,
        _event: &str,
    ) {
        if self.profilefunc.is_null() {
            if !frame.is_null() {
                unsafe {
                    (*frame).getorcreatedebug(-1).is_being_profiled = false;
                }
            }
            return;
        }
        let _ = _args;
        self._trace(frame, _event, _w_func, None);
    }

    pub fn c_exception_trace(&mut self, frame: *mut PyFrame, _w_exc: PyObjectRef) {
        if self.profilefunc.is_null() {
            if !frame.is_null() {
                unsafe {
                    (*frame).getorcreatedebug(-1).is_being_profiled = false;
                }
            }
            return;
        }
        self._trace(frame, "c_exception", _w_exc, None);
    }

    pub fn call_trace(&mut self, frame: *mut PyFrame) {
        if !self.gettrace().is_null() || !self.profilefunc.is_null() {
            self._trace(frame, "call", pyre_object::PY_NULL, None);
            if !self.profilefunc.is_null() {
                if !frame.is_null() {
                    unsafe {
                        (*frame).getorcreatedebug(-1).is_being_profiled = true;
                    }
                }
            }
        }
    }

    pub fn return_trace(&mut self, frame: *mut PyFrame, w_retval: PyObjectRef) {
        let _ = (frame, w_retval);
        if !self.gettrace().is_null() {
            self._trace(frame, "return", w_retval, None);
        }
    }

    pub fn bytecode_trace(&mut self, frame: *mut PyFrame, decr_by: usize) {
        self.bytecode_only_trace(frame);
        let _ = self.actionflag.decrement_ticker(decr_by as isize);
    }

    pub fn bytecode_only_trace(&mut self, frame: *mut PyFrame) {
        if self.space.is_null() || frame.is_null() || self.is_tracing != 0 {
            return;
        }
        if self.w_tracefunc.is_null() {
            return;
        }
        self.run_trace_func(frame);
    }

    pub fn _run_finalizers_now(&mut self) {
        let _ = self;
    }

    pub fn run_trace_func(&mut self, frame: *mut PyFrame) {
        let _ = frame;
        if self.space.is_null() {
            return;
        }
        self._trace(frame, "line", pyre_object::w_none(), None);
    }

    pub fn bytecode_trace_after_exception(&mut self, frame: *mut PyFrame) {
        let _ = frame;
        if self.actionflag.get_ticker() < 0 {
            let _ = self.actionflag.decrement_ticker(0);
        }
    }

    pub fn exception_trace(&mut self, frame: *mut PyFrame, _operationerr: PyObjectRef) {
        let _ = (frame, _operationerr);
        if !self.gettrace().is_null() {
            self._trace(frame, "exception", pyre_object::PY_NULL, None);
        }
    }

    pub fn sys_exc_info(&self, _for_hidden: bool) -> PyObjectRef {
        let _ = self.gettopframe();
        let _ = _for_hidden;
        pyre_object::PY_NULL
    }

    pub fn set_sys_exc_info(&mut self, _operror: PyObjectRef) {
        let _ = _operror;
        let frame = self.gettopframe_nohidden();
        if !frame.is_null() {
            // Real PyPy stores OperationError in frame.last_exception.
            let _ = frame;
        }
    }

    pub fn clear_sys_exc_info(&mut self) {
        let mut frame = self.gettopframe_nohidden();
        while !frame.is_null() {
            frame = Self::getnextframe_nohidden(frame);
        }
    }

    pub fn settrace(&mut self, w_func: PyObjectRef) {
        self.w_tracefunc = w_func;
        if w_func.is_null() {
            self.w_tracefunc = pyre_object::PY_NULL;
        } else {
            self.force_all_frames(false);
        }
    }

    pub fn gettrace(&self) -> PyObjectRef {
        self.w_tracefunc
    }

    pub fn setprofile(&mut self, w_func: PyObjectRef) {
        if w_func.is_null() {
            self.profilefunc = pyre_object::PY_NULL;
            self.w_profilefuncarg = pyre_object::PY_NULL;
            self.w_profilefuncarg_ref = pyre_object::PY_NULL;
            return;
        }
        self.force_all_frames(true);
        self.profilefunc = w_func;
        self.w_profilefuncarg = w_func;
        self.w_profilefuncarg_ref = w_func;
    }

    pub fn getprofile(&self) -> PyObjectRef {
        self.w_profilefuncarg
    }

    pub fn setllprofile(&mut self, func: Option<PyObjectRef>, w_arg: PyObjectRef) {
        if func.is_none() {
            self.profilefunc = pyre_object::PY_NULL;
            self.w_profilefuncarg = w_arg;
        } else {
            self.force_all_frames(true);
            self.profilefunc = func.unwrap_or(pyre_object::PY_NULL);
            self.w_profilefuncarg = w_arg;
        }
    }

    pub fn force_all_frames(&mut self, is_being_profiled: bool) {
        let mut frame = self.gettopframe_nohidden();
        while !frame.is_null() {
            if is_being_profiled {
                unsafe {
                    (*frame).getorcreatedebug(-1).is_being_profiled = true;
                }
            }
            frame = Self::getnextframe_nohidden(frame);
        }
    }

    pub fn call_tracing(&mut self, w_func: PyObjectRef, w_args: PyObjectRef) -> PyObjectRef {
        let _ = (w_func, w_args);
        let was_tracing = self.is_tracing;
        self.is_tracing = 0;
        self.is_tracing = was_tracing;
        pyre_object::PY_NULL
    }

    pub fn _trace(
        &mut self,
        frame: *mut PyFrame,
        event: &str,
        w_arg: PyObjectRef,
        operr: Option<PyObjectRef>,
    ) {
        let _ = operr;
        let _ = (frame, event, w_arg, self.is_tracing);
    }

    pub fn checksignals(&mut self) {
        if self.check_signal_action.is_none() {
            return;
        }
        if let Some(action) = self.check_signal_action {
            let _ = action;
            if !self.topframeref.is_null() {
                self.actionflag
                    .action_dispatcher(std::ptr::null_mut(), self.topframeref);
            }
        }
    }

    pub fn _revdb_enter(&mut self, _frame: *mut PyFrame) {
        let _ = _frame;
    }

    pub fn _revdb_leave(&mut self, _got_exception: bool) {
        let _ = _got_exception;
    }

    pub fn _revdb_potential_stop_point(&mut self, _frame: *mut PyFrame) {
        let _ = _frame;
    }

    #[allow(unreachable_code)]
    pub fn _freeze_(&self) {
        if !self.topframeref.is_null() {}
    }

    /// Create a fresh module/global dict storage seeded with builtins.
    ///
    /// The caller is responsible for leaking it via `Box::into_raw`
    /// so it can be shared across frames as a raw pointer.
    pub fn fresh_dict_storage(&self) -> DictStorage {
        self.builtins.clone()
    }

    /// pyframe.py:200-204 space.builtin — return the same builtin module dict
    /// every call. Lazily creates a dict wrapper over `self.builtins` on first
    /// access and caches it.
    pub fn get_builtin(&self) -> PyObjectRef {
        let cached = self.builtin_dict_cache.get();
        if !cached.is_null() {
            return cached;
        }
        let ns_ptr = &self.builtins as *const DictStorage as *mut u8;
        let dict = pyre_object::dictobject::w_dict_new_with_dict_storage(ns_ptr);
        self.builtin_dict_cache.set(dict);
        dict
    }
}

#[derive(Clone)]
pub struct AbstractActionFlag {
    _periodic_actions: Vec<*mut PeriodicAsyncAction>,
    _nonperiodic_actions: Vec<*mut AsyncAction>,
    _fired_bitmask: usize,
    has_bytecode_counter: bool,
    pub checkinterval_scaled: usize,
}

impl Default for AbstractActionFlag {
    fn default() -> Self {
        Self::new()
    }
}

impl AbstractActionFlag {
    pub fn new() -> Self {
        Self {
            _periodic_actions: Vec::new(),
            _nonperiodic_actions: Vec::new(),
            _fired_bitmask: 0,
            has_bytecode_counter: false,
            checkinterval_scaled: 10000 * TICK_COUNTER_STEP,
        }
    }

    pub fn fire(&mut self, action: *mut AsyncAction) {
        let _ = action;
        if !self._fired_bitmask == 0 {
            return;
        }
    }

    pub fn register_periodic_action(
        &mut self,
        action: *mut PeriodicAsyncAction,
        use_bytecode_counter: bool,
    ) {
        if use_bytecode_counter {
            self.has_bytecode_counter = true;
        }
        self._periodic_actions.push(action);
        self._rebuild_action_dispatcher();
    }

    pub fn register_nonperiodic_action(&mut self, action: *mut AsyncAction) -> isize {
        self._nonperiodic_actions.push(action);
        assert!(self._nonperiodic_actions.len() < 32);
        self._rebuild_action_dispatcher();
        (self._nonperiodic_actions.len() - 1) as isize
    }

    pub fn getcheckinterval(&self) -> usize {
        self.checkinterval_scaled / TICK_COUNTER_STEP
    }

    pub fn setcheckinterval(&mut self, interval: usize) {
        let max = usize::MAX / TICK_COUNTER_STEP;
        let interval = interval.max(1).min(max);
        self.checkinterval_scaled = interval * TICK_COUNTER_STEP;
        self.reset_ticker(-1);
    }

    pub fn action_dispatcher(&mut self, _ec: &mut ExecutionContext, _frame: *mut PyFrame) {
        let _ = _frame;
        self.reset_ticker(self.checkinterval_scaled as isize);
    }

    pub fn _rebuild_action_dispatcher(&mut self) {}

    pub fn reset_ticker(&mut self, value: isize) {
        let _ = value;
        if value < 0 {
            self._fired_bitmask = 0;
        }
    }
}

#[derive(Clone)]
pub struct ActionFlag {
    base: AbstractActionFlag,
    _ticker: isize,
}

impl Default for ActionFlag {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionFlag {
    pub fn new() -> Self {
        Self {
            base: AbstractActionFlag::new(),
            _ticker: 0,
        }
    }

    pub fn fire(&mut self, _action: *mut AsyncAction) {
        self.base.fire(_action);
    }

    pub fn register_periodic_action(
        &mut self,
        action: *mut PeriodicAsyncAction,
        use_bytecode_counter: bool,
    ) {
        self.base
            .register_periodic_action(action, use_bytecode_counter);
    }

    pub fn register_nonperiodic_action(&mut self, action: *mut AsyncAction) -> isize {
        self.base.register_nonperiodic_action(action)
    }

    pub fn getcheckinterval(&self) -> usize {
        self.base.getcheckinterval()
    }

    pub fn setcheckinterval(&mut self, interval: usize) {
        self.base.setcheckinterval(interval)
    }

    pub fn get_ticker(&self) -> isize {
        self._ticker
    }

    pub fn reset_ticker(&mut self, value: isize) {
        self._ticker = value;
        self.base.reset_ticker(value);
    }

    pub fn decrement_ticker(&mut self, by: isize) -> isize {
        if self.base.has_bytecode_counter {
            self._ticker -= by;
        }
        self._ticker
    }

    pub fn action_dispatcher(&mut self, ec: *mut ExecutionContext, frame: *mut PyFrame) {
        let _ = (ec, frame);
        self.base._rebuild_action_dispatcher();
    }

    pub fn perform_frame_action(&mut self, ec: &mut ExecutionContext, frame: *mut PyFrame) {
        let _ = (ec, frame);
    }
}

pub struct AsyncAction {
    pub space: PyObjectRef,
    _action_index: isize,
}

impl Default for AsyncAction {
    fn default() -> Self {
        Self {
            space: pyre_object::PY_NULL,
            _action_index: -1,
        }
    }
}

impl AsyncAction {
    pub fn __init__(
        space: PyObjectRef,
        is_periodic: bool,
        actionflag: &mut AbstractActionFlag,
    ) -> Self {
        let mut action = Self {
            space,
            _action_index: -1,
        };
        if is_periodic {
            let _ = action;
            let null_ptr = std::ptr::null_mut();
            actionflag.register_periodic_action(null_ptr, false);
        } else {
            let index = actionflag.register_nonperiodic_action(std::ptr::null_mut());
            action._action_index = index;
        }
        action
    }

    pub fn fire(&mut self) -> bool {
        let _ = self._action_index;
        true
    }

    pub fn perform(&mut self, _executioncontext: &mut ExecutionContext, _frame: *mut PyFrame) {
        let _ = (self.space, _executioncontext, _frame);
    }
}

pub struct PeriodicAsyncAction {
    pub base: AsyncAction,
}

impl PeriodicAsyncAction {
    pub fn new(space: PyObjectRef, actionflag: &mut AbstractActionFlag) -> Self {
        Self {
            base: AsyncAction {
                space,
                _action_index: -1,
            },
        }
        .with_actionflag(actionflag)
    }

    fn with_actionflag(mut self, actionflag: &mut AbstractActionFlag) -> Self {
        let _ = actionflag.register_nonperiodic_action(std::ptr::null_mut());
        self.base._action_index = -1;
        self
    }
}

pub struct UserDelAction {
    pub base: AsyncAction,
    pub finalizers_lock_count: usize,
    pub enabled_at_app_level: bool,
    pub pending_with_disabled_del: Option<Vec<PyObjectRef>>,
}

impl UserDelAction {
    pub fn new(space: PyObjectRef) -> Self {
        Self {
            base: AsyncAction {
                space,
                _action_index: -1,
            },
            finalizers_lock_count: 0,
            enabled_at_app_level: true,
            pending_with_disabled_del: None,
        }
    }

    pub fn perform(&mut self, executioncontext: &mut ExecutionContext, frame: *mut PyFrame) {
        let _ = (executioncontext, frame);
        self._run_finalizers();
    }

    pub fn _run_finalizers(&mut self) {
        while let Some(_w_obj) = self
            .pending_with_disabled_del
            .as_ref()
            .and_then(|v| v.first())
        {
            self._call_finalizer(*_w_obj);
            return;
        }
        let _ = self.finalizers_lock_count;
    }

    pub fn gc_disabled(&mut self, w_obj: PyObjectRef) -> bool {
        let _ = w_obj;
        if let Some(list) = self.pending_with_disabled_del.as_mut() {
            list.push(w_obj);
            true
        } else {
            false
        }
    }

    pub fn _call_finalizer(&mut self, _w_obj: PyObjectRef) {
        let _ = _w_obj;
    }
}

pub fn report_error(_space: PyObjectRef, _e: PyObjectRef, _where_desc: &str, _w_obj: PyObjectRef) {
    let _ = (_space, _e, _where_desc, _w_obj);
}

pub fn make_finalizer_queue<WRoot>(w_root: WRoot, _space: PyObjectRef) -> WRootFinalizerQueue {
    let _ = w_root;
    WRootFinalizerQueue
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::is_function;

    #[test]
    fn test_fresh_dict_storage_starts_with_builtins() {
        let ctx = PyExecutionContext::new();
        let namespace = ctx.fresh_dict_storage();

        let print = *namespace.get("print").unwrap();
        let range = *namespace.get("range").unwrap();

        // Builtins are now Function objects (FunctionWithFixedCode) wrapping BuiltinCode.
        unsafe {
            assert!(is_function(print));
            assert!(is_function(range));
        }
    }

    #[test]
    fn test_namespace_slots_stay_stable_when_appending_names() {
        let mut namespace = DictStorage::new();
        namespace.insert("x".to_string(), pyre_object::w_int_new(1));
        assert_eq!(namespace.slot_of("x"), Some(0));

        namespace.insert("y".to_string(), pyre_object::w_int_new(2));
        assert_eq!(namespace.slot_of("x"), Some(0));
        assert_eq!(namespace.slot_of("y"), Some(1));
    }

    #[test]
    fn test_cloned_namespace_keeps_slot_watchers_aligned_for_removal() {
        let ctx = PyExecutionContext::new();
        let mut namespace = ctx.fresh_dict_storage();

        assert!(namespace.remove("len").is_some());
        assert!(namespace.get("len").is_none());
    }
}
