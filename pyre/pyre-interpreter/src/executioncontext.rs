use pyre_object::PyObjectRef;
use pyre_object::object_array::{
    ItemsBlock, alloc_list_items_block, dealloc_list_items_block, grow_list_items_block,
    items_block_capacity, items_block_items_base,
};
use std::sync::OnceLock;

use crate::{PyFrame, new_builtin_dict_storage};

fn trace_frame_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    let raw = *TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("frame", |ns| {
            crate::dict_storage_store(ns, "__dict__", pyre_object::w_none());
        });
        tp as usize
    });
    raw as PyObjectRef
}

fn wrap_trace_frame(frame: *mut PyFrame) -> PyObjectRef {
    if frame.is_null() {
        return pyre_object::w_none();
    }
    let w_frame = pyre_object::w_instance_new(trace_frame_type());
    unsafe {
        let frame_ref = &mut *frame;
        let w_globals = frame_ref.get_w_globals();
        let w_locals = frame_ref.get_w_locals();
        let w_trace = frame_ref.get_w_f_trace();
        let _ = crate::baseobjspace::setattr(w_frame, "f_code", frame_ref.pycode as PyObjectRef);
        let _ = crate::baseobjspace::setattr(w_frame, "f_back", pyre_object::w_none());
        let _ = crate::baseobjspace::setattr(w_frame, "f_builtins", frame_ref.get_builtin());
        let _ = crate::baseobjspace::setattr(
            w_frame,
            "f_globals",
            if w_globals.is_null() {
                pyre_object::w_none()
            } else {
                pyre_object::dictobject::w_dict_new_with_dict_storage(w_globals as *mut u8)
            },
        );
        let _ = crate::baseobjspace::setattr(
            w_frame,
            "f_locals",
            if w_locals.is_null() {
                pyre_object::w_none()
            } else {
                pyre_object::dictobject::w_dict_new_with_dict_storage(w_locals as *mut u8)
            },
        );
        let _ = crate::baseobjspace::setattr(
            w_frame,
            "f_lineno",
            pyre_object::w_int_new(frame_ref.fget_f_lineno() as i64),
        );
        let _ = crate::baseobjspace::setattr(
            w_frame,
            "f_lasti",
            pyre_object::w_int_new(frame_ref.fget_f_lasti() as i64),
        );
        let _ = crate::baseobjspace::setattr(
            w_frame,
            "f_trace",
            if w_trace.is_null() {
                pyre_object::w_none()
            } else {
                w_trace
            },
        );
    }
    w_frame
}

/// pypy/interpreter/executioncontext.py:10-15 app_profile_call.
///
/// Module-level low-level trampoline used by `setprofile` to bridge
/// `setllprofile` (which stores a function pointer) to the user
/// callable held in `w_profilefuncarg`.  RPython:
///
/// ```python
/// def app_profile_call(space, w_callable, frame, event, w_arg):
///     frame = jit.hint(frame, access_directly=False)
///     space.call_function(w_callable, frame, space.newtext(event), w_arg)
/// ```
///
/// `jit.hint(frame, access_directly=False)` is a JIT-only annotation
/// that demotes the green frame to a regular w_object before the user
/// callable observes it.  Pyre's internal `PyFrame` is not itself a
/// `PyObject`, so the callback sees a frame wrapper carrying the same
/// Python-visible fields instead of a raw struct pointer.
pub fn app_profile_call(
    _space: PyObjectRef,
    w_callable: PyObjectRef,
    frame: *mut PyFrame,
    event: &str,
    w_arg: PyObjectRef,
) -> Result<(), crate::PyError> {
    let frame_obj = wrap_trace_frame(frame);
    let w_event = pyre_object::w_str_new(event);
    crate::call::call_function_impl_result(w_callable, &[frame_obj, w_event, w_arg]).map(|_| ())
}

/// pypy/interpreter/executioncontext.py:320 `self.profilefunc` — the
/// low-level callback stored by `setllprofile`.  In RPython this is a
/// raw function pointer; pyre mirrors the type with a `fn(...)` so the
/// trampoline (`app_profile_call`) is the actual stored value rather
/// than the user callable, matching upstream's two-slot design
/// (`profilefunc` + `w_profilefuncarg`).
pub type ProfileFunc = fn(
    space: PyObjectRef,
    w_callable: PyObjectRef,
    frame: *mut PyFrame,
    event: &str,
    w_arg: PyObjectRef,
) -> Result<(), crate::PyError>;

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
    /// pypy/interpreter/executioncontext.py:320 — function pointer to
    /// the low-level profiling trampoline (e.g. `app_profile_call`).
    /// `None` means profiling is disabled.  The user callable lives in
    /// `w_profilefuncarg`; the trampoline forwards to it.
    pub profilefunc: Option<ProfileFunc>,
    pub w_profilefuncarg: PyObjectRef,
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
            profilefunc: None,
            w_profilefuncarg: pyre_object::PY_NULL,
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
        // pypy/interpreter/executioncontext.py:85-89 enter parity.
        if !self.space.is_null() && self.is_tracing > 0 {
            self._revdb_enter(frame);
        }
        unsafe {
            (*frame).f_backref = self.topframeref;
        }
        // `jit.virtual_ref(frame)` — vref allocation, no-op until
        // jit.virtual_ref is ported.
        self.topframeref = frame;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn leave(
        &mut self,
        frame: *mut PyFrame,
        w_exitvalue: PyObjectRef,
        got_exception: bool,
    ) -> Result<(), crate::PyError> {
        // pypy/interpreter/executioncontext.py:91-109 leave parity.
        // The original wraps `_trace('leaveframe', …)` in try/finally so
        // the topframeref restore and the vref dance always run.  We
        // capture the trace result and propagate it after the cleanup
        // block below.
        let trace_result = if self.profilefunc.is_some() {
            self._trace(frame, "leaveframe", w_exitvalue, None)
        } else {
            Ok(())
        };
        let _frame_vref = self.topframeref;
        unsafe {
            self.topframeref = (*frame).f_backref;
            if (*frame).escaped || got_exception {
                let f_back = (*frame).get_f_back();
                if !f_back.is_null() {
                    (*f_back).mark_as_escaped();
                }
                // `frame_vref()` — vref force, no-op until jit.virtual_ref is ported.
            }
        }
        // `jit.virtual_ref_finish(frame_vref, frame)` — no-op until
        // jit.virtual_ref is ported.
        self._revdb_leave(got_exception);
        trace_result
    }

    pub fn c_call_trace(
        &mut self,
        frame: *mut PyFrame,
        w_func: PyObjectRef,
        args: Option<PyObjectRef>,
    ) -> Result<(), crate::PyError> {
        let args = args.unwrap_or(pyre_object::PY_NULL);
        self._c_call_return_trace(frame, w_func, args, "c_call")
    }

    pub fn c_return_trace(
        &mut self,
        frame: *mut PyFrame,
        w_func: PyObjectRef,
        args: Option<PyObjectRef>,
    ) -> Result<(), crate::PyError> {
        let args = args.unwrap_or(pyre_object::PY_NULL);
        self._c_call_return_trace(frame, w_func, args, "c_return")
    }

    pub fn _c_call_return_trace(
        &mut self,
        frame: *mut PyFrame,
        mut w_func: PyObjectRef,
        args: PyObjectRef,
        event: &str,
    ) -> Result<(), crate::PyError> {
        if self.profilefunc.is_none() {
            if !frame.is_null() {
                unsafe {
                    (*frame).getorcreatedebug(-1).is_being_profiled = false;
                }
            }
            return Ok(());
        }
        // executioncontext.py:125-134 FunctionWithFixedCode method-call
        // rebinding. Pyre represents FunctionWithFixedCode and builtin
        // functions as Function records; builtin-code functions are the
        // fixed-code subset that can appear here.
        unsafe {
            if crate::is_function(w_func)
                && crate::is_builtin_code(crate::function_get_code(w_func) as PyObjectRef)
                && !args.is_null()
            {
                let w_firstarg = if pyre_object::is_tuple(args) {
                    pyre_object::w_tuple_getitem(args, 0)
                } else if pyre_object::is_list(args) {
                    pyre_object::w_list_getitem(args, 0)
                } else {
                    None
                };
                if let Some(w_firstarg) = w_firstarg {
                    if !w_firstarg.is_null() {
                        let w_type =
                            crate::typedef::r#type(w_firstarg).unwrap_or(pyre_object::PY_NULL);
                        w_func = crate::descr_function_get(w_func, w_firstarg, w_type);
                    }
                }
            }
        }
        self._trace(frame, event, w_func, None)
    }

    pub fn c_exception_trace(
        &mut self,
        frame: *mut PyFrame,
        _w_exc: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        if self.profilefunc.is_none() {
            if !frame.is_null() {
                unsafe {
                    (*frame).getorcreatedebug(-1).is_being_profiled = false;
                }
            }
            return Ok(());
        }
        self._trace(frame, "c_exception", _w_exc, None)
    }

    pub fn call_trace(&mut self, frame: *mut PyFrame) -> Result<(), crate::PyError> {
        if !self.gettrace().is_null() || self.profilefunc.is_some() {
            self._trace(frame, "call", pyre_object::w_none(), None)?;
            if self.profilefunc.is_some() {
                if !frame.is_null() {
                    unsafe {
                        (*frame).getorcreatedebug(-1).is_being_profiled = true;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn return_trace(
        &mut self,
        frame: *mut PyFrame,
        w_retval: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        let _ = (frame, w_retval);
        if !self.gettrace().is_null() {
            self._trace(frame, "return", w_retval, None)?;
        }
        Ok(())
    }

    pub fn bytecode_trace(
        &mut self,
        frame: *mut PyFrame,
        decr_by: usize,
    ) -> Result<(), crate::PyError> {
        let trace_result = self.bytecode_only_trace(frame);
        let _ = self.actionflag.decrement_ticker(decr_by as isize);
        trace_result
    }

    pub fn bytecode_only_trace(&mut self, frame: *mut PyFrame) -> Result<(), crate::PyError> {
        if self.space.is_null() || frame.is_null() || self.is_tracing != 0 {
            return Ok(());
        }
        if self.w_tracefunc.is_null() {
            return Ok(());
        }
        self.run_trace_func(frame)
    }

    pub fn _run_finalizers_now(&mut self) {
        let _ = self;
    }

    pub fn run_trace_func(&mut self, frame: *mut PyFrame) -> Result<(), crate::PyError> {
        let _ = frame;
        if self.space.is_null() {
            return Ok(());
        }
        self._trace(frame, "line", pyre_object::w_none(), None)
    }

    pub fn bytecode_trace_after_exception(&mut self, frame: *mut PyFrame) {
        let _ = frame;
        if self.actionflag.get_ticker() < 0 {
            let _ = self.actionflag.decrement_ticker(0);
        }
    }

    /// pypy/interpreter/executioncontext.py:430-433 exception_trace.
    ///
    /// ```python
    /// def exception_trace(self, frame, operationerr):
    ///     if self.w_tracefunc is not None:
    ///         self._trace(frame, 'exception',
    ///                     operationerr.get_w_value(self.space), operationerr)
    /// ```
    ///
    /// PyPy unpacks the `OperationError` inside `_trace` via
    /// `operr.w_type` / `operr.normalize_exception(space)` /
    /// `operr.get_w_traceback(space)` (executioncontext.py:359-363) to
    /// build the `(w_type, w_value, w_traceback)` tuple passed to the
    /// trace callback.  Pyre carries the equivalent fields explicitly
    /// because there is no `OperationError` plumbed through the trace
    /// surface yet — callers supply the raised type, the wrapped value,
    /// and the traceback (the latter two may be `w_none()` while pyre
    /// lacks traceback objects at this layer).
    pub fn exception_trace(
        &mut self,
        frame: *mut PyFrame,
        w_type: PyObjectRef,
        w_value: PyObjectRef,
        w_traceback: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        if !self.gettrace().is_null() {
            self._trace(
                frame,
                "exception",
                w_value,
                Some((w_type, w_value, w_traceback)),
            )?;
        }
        Ok(())
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
        if w_func.is_null() || w_func == pyre_object::w_none() {
            self.w_tracefunc = pyre_object::PY_NULL;
        } else {
            self.force_all_frames(false);
            // executioncontext.py:296-298 — increase the JIT's
            // trace_limit when a tracefunc is installed; tracing
            // generates a ton of extra ops per bytecode.
            crate::call::set_jit_param("trace_limit", 10000);
        }
    }

    pub fn gettrace(&self) -> PyObjectRef {
        self.w_tracefunc
    }

    /// pypy/interpreter/executioncontext.py:303-310 setprofile.
    pub fn setprofile(&mut self, w_func: PyObjectRef) -> Result<(), crate::PyError> {
        if w_func.is_null() || w_func == pyre_object::w_none() {
            self.profilefunc = None;
            self.w_profilefuncarg = pyre_object::PY_NULL;
            Ok(())
        } else {
            self.setllprofile(Some(app_profile_call), w_func)
        }
    }

    /// pypy/interpreter/executioncontext.py:312-313 getprofile.
    pub fn getprofile(&self) -> PyObjectRef {
        self.w_profilefuncarg
    }

    /// pypy/interpreter/executioncontext.py:315-321 setllprofile.
    pub fn setllprofile(
        &mut self,
        func: Option<ProfileFunc>,
        w_arg: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        if func.is_some() {
            // executioncontext.py:317-318 `if w_arg is None: raise
            // ValueError("Cannot call setllprofile with real None")`.
            // The check is against RPython-level None (== null in pyre);
            // Python-level `w_none()` (`space.w_None`) is a valid user
            // argument that flows through unchanged.
            if w_arg.is_null() {
                return Err(crate::PyError::value_error(
                    "Cannot call setllprofile with real None",
                ));
            }
            self.force_all_frames(true);
        }
        self.profilefunc = func;
        self.w_profilefuncarg = w_arg;
        Ok(())
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
        let was_tracing = self.is_tracing;
        self.is_tracing = 0;
        let result = crate::baseobjspace::call(w_func, w_args, None);
        self.is_tracing = was_tracing;
        result
    }

    /// pypy/interpreter/executioncontext.py:346-428 _trace.
    ///
    /// Two-arm dispatch: the tracing arm fires every event for the
    /// frame's `w_f_trace` (or the global `w_tracefunc` on `'call'`),
    /// the profiling arm fires only on `call`/`return`/`c_call`/
    /// `c_return`/`c_exception` and routes through the
    /// `profilefunc` low-level trampoline (`app_profile_call` for
    /// `setprofile`-installed callbacks).
    ///
    /// `operr` carries the `(w_type, w_value, w_traceback)` triple
    /// `executioncontext.py:359-363` reads from the upstream
    /// `OperationError` via `operr.w_type`,
    /// `operr.normalize_exception(space)`, and
    /// `operr.get_w_traceback(space)`.  Pyre passes the three fields
    /// explicitly because no `OperationError` flows through the trace
    /// surface yet; `w_traceback` is typically `w_none()` until pyre
    /// gains traceback objects at this layer.  The structural early
    /// returns, event filtering, and `is_tracing` bookkeeping mirror
    /// upstream.
    pub fn _trace(
        &mut self,
        frame: *mut PyFrame,
        event: &str,
        w_arg: PyObjectRef,
        operr: Option<(PyObjectRef, PyObjectRef, PyObjectRef)>,
    ) -> Result<(), crate::PyError> {
        // executioncontext.py:347 if self.is_tracing or frame.hide():
        if self.is_tracing != 0 {
            return Ok(());
        }
        if frame.is_null() {
            return Ok(());
        }
        if unsafe { (*frame).hide() } {
            return Ok(());
        }

        let space = self.space;

        // executioncontext.py:353-356 Tracing cases
        let w_callback = if event == "call" {
            self.gettrace()
        } else {
            unsafe { (*frame).get_w_f_trace() }
        };

        if !w_callback.is_null() && event != "leaveframe" {
            // executioncontext.py:359-363 normalize_exception + rebuild
            // `w_arg` as `(w_type, w_value, w_traceback)` when an
            // `operr` triple accompanies the trace event.  Caller
            // (`exception_trace`) already supplies the explicit raised
            // type; only fall back to `typedef::r#type(w_value)` when
            // it was passed as null, mirroring the cases where pyre
            // cannot yet preserve `operr.w_type` distinct from the
            // value's runtime type.
            let w_arg = if let Some((w_type, w_value, w_traceback)) = operr {
                let w_type = if w_type.is_null() {
                    crate::typedef::r#type(w_value).unwrap_or_else(pyre_object::w_none)
                } else {
                    w_type
                };
                let w_traceback = if w_traceback.is_null() {
                    pyre_object::w_none()
                } else {
                    w_traceback
                };
                pyre_object::tupleobject::w_tuple_new(vec![w_type, w_value, w_traceback])
            } else {
                w_arg
            };

            let lineno = unsafe { (*frame).get_last_lineno() };
            let init_lineno = if unsafe { (*frame).last_instr } >= 1 {
                lineno
            } else {
                -1
            };
            let had_locals = unsafe {
                let d = (*frame).getorcreatedebug(init_lineno);
                !d.w_locals.is_null()
            };
            if had_locals {
                unsafe { (*frame).fast2locals() };
            }
            let (prev_line_tracing, old_lineno) = unsafe {
                let d = (*frame).getorcreatedebug(init_lineno);
                (d.is_in_line_tracing, d.f_lineno)
            };

            // executioncontext.py:376 self.is_tracing += 1
            self.is_tracing += 1;
            let call_result = (|| {
                unsafe {
                    let d = (*frame).getorcreatedebug(init_lineno);
                    if event == "line" {
                        d.is_in_line_tracing = true;
                    }
                    d.f_lineno = lineno;
                }
                // executioncontext.py:382-385 space.call_function(w_callback, frame, w_event, w_arg)
                let frame_obj = wrap_trace_frame(frame);
                let w_event = pyre_object::w_str_new(event);
                let w_result = crate::call::call_function_impl_result(
                    w_callback,
                    &[frame_obj, w_event, w_arg],
                )?;
                if w_result != pyre_object::w_none() {
                    unsafe {
                        (*frame).getorcreatedebug(init_lineno).w_f_trace = w_result;
                    }
                }
                Ok::<(), crate::PyError>(())
            })();

            if call_result.is_err() {
                self.settrace(pyre_object::w_none());
                unsafe {
                    (*frame).getorcreatedebug(init_lineno).w_f_trace = pyre_object::PY_NULL;
                }
            }

            unsafe {
                let d = (*frame).getorcreatedebug(init_lineno);
                if d.f_lineno == lineno {
                    d.f_lineno = old_lineno;
                }
                d.is_in_line_tracing = prev_line_tracing;
            }
            self.is_tracing -= 1;
            if had_locals {
                unsafe { (*frame).locals2fast(false) };
            }
            // executioncontext.py:392-395 — re-raise the callback's
            // exception after restoring trace bookkeeping. Caller chain
            // (call_trace/return_trace/bytecode_trace/exception_trace
            // → eval_frame_plain) propagates the error up so the
            // tracefunc behaves like an inline raise from the executing
            // frame.
            if let Err(err) = call_result {
                return Err(err);
            }
        }

        // executioncontext.py:404-428 Profile cases
        if let Some(profilefunc) = self.profilefunc {
            // executioncontext.py:406-411 — only call/return/c_call/
            // c_return/c_exception events fire the profile callback.
            let event = if event == "leaveframe" {
                "return"
            } else if event == "call"
                || event == "c_call"
                || event == "c_return"
                || event == "c_exception"
            {
                event
            } else {
                return Ok(());
            };

            // executioncontext.py:416-417 assert self.is_tracing == 0; self.is_tracing += 1
            assert_eq!(self.is_tracing, 0);
            self.is_tracing += 1;
            // executioncontext.py:420-425 self.profilefunc(...), clearing
            // profile slots on exceptions.
            let profile_result = profilefunc(space, self.w_profilefuncarg, frame, event, w_arg);
            if let Err(err) = profile_result {
                // executioncontext.py:421-425 — clear profile slots and
                // re-raise so the caller observes the failure (matches
                // the bare `raise` after the `except:` block).
                self.profilefunc = None;
                self.w_profilefuncarg = pyre_object::PY_NULL;
                self.is_tracing -= 1;
                return Err(err);
            }
            // executioncontext.py:427-428 self.is_tracing -= 1
            self.is_tracing -= 1;
        }
        Ok(())
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
