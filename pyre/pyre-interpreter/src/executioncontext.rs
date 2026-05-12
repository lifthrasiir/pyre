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

/// Copy callback-visible mutations on the trace-frame wrapper back to
/// the live `PyFrame`'s debug data.
///
/// PyPy's `_trace` passes the live frame to the callback (`jit.hint(
/// frame, access_directly=False)`), so `frame.f_trace = local` and
/// `frame.f_lineno = N` from inside the callback land directly on
/// `frame.debug`.  Pyre wraps the frame in a `pyre_object` instance
/// because `PyFrame` is not itself a `PyObject`, so a setattr on the
/// wrapper would otherwise stay isolated.  After the callback returns,
/// read the wrapper's mutated attributes and propagate the changes to
/// the live frame, preserving the user-visible semantics for the
/// common case (debugger setting `frame.f_trace` to a per-frame
/// callback while returning `None`).
///
/// The `_trace` w_result branch still wins (`executioncontext.py:386-
/// 391`): a non-None callback return value overrides whichever value
/// the setattr left behind, matching CPython issue11992 bug-for-bug
/// compatibility.
fn flush_trace_frame_writeback(frame: *mut PyFrame, w_frame: PyObjectRef, init_lineno: isize) {
    if frame.is_null() || w_frame.is_null() {
        return;
    }
    let new_f_trace = match crate::baseobjspace::getattr(w_frame, "f_trace") {
        Ok(v) => v,
        Err(_) => return,
    };
    // pyframe.py:785-791 fset_f_trace:
    //   if space.is_w(w_trace, space.w_None):
    //       self.getorcreatedebug().w_f_trace = None
    //   else:
    //       d = self.getorcreatedebug()
    //       d.w_f_trace = w_trace
    //       d.f_lineno = self.get_last_lineno()
    // The non-None branch also realigns f_lineno to the current
    // bytecode line so trace events fire from the new tracer's
    // perspective immediately.
    unsafe {
        let d = (*frame).getorcreatedebug(init_lineno);
        let is_clear = new_f_trace.is_null() || new_f_trace == pyre_object::w_none();
        if is_clear {
            d.w_f_trace = pyre_object::PY_NULL;
        } else if new_f_trace != d.w_f_trace {
            d.w_f_trace = new_f_trace;
            d.f_lineno = (*frame).get_last_lineno();
        }
    }
    if let Ok(new_f_lineno_obj) = crate::baseobjspace::getattr(w_frame, "f_lineno") {
        if !new_f_lineno_obj.is_null() && unsafe { pyre_object::is_int(new_f_lineno_obj) } {
            let new_lineno = unsafe { pyre_object::w_int_get_value(new_f_lineno_obj) };
            // PRE-EXISTING-ADAPTATION (Task #224): pyframe.py:683-764
            // `PyFrame.fset_f_lineno` is the upstream setter. It runs
            // line-jump validation against the bytecode (block stack
            // unwinding through SETUP_LOOP/SETUP_EXCEPT, code address
            // recomputation via PyCode._signature_addr_to_line, and
            // last_instr realignment). Pyre's wrapper writes only
            // debug.f_lineno because PyFrame is not yet a PyObject
            // and the validator's PyCode internals (try/except
            // boundaries, generator restart guards) are not exposed
            // through the wrapper interface. The full setter port
            // is gated on the PyFrame ↔ PyObject identity epic.
            unsafe {
                let d = (*frame).getorcreatedebug(init_lineno);
                d.f_lineno = new_lineno as isize;
            }
        }
    }
    // pyframe.py:799-806 fset_f_trace_lines / fset_f_trace_opcodes.
    // Both setters apply `space.is_true(w_trace)` to coerce arbitrary
    // truthy values; pyre's `is_true` is the equivalent.
    if let Ok(new_lines_obj) = crate::baseobjspace::getattr(w_frame, "f_trace_lines") {
        if !new_lines_obj.is_null() {
            let new_lines = unsafe { crate::baseobjspace::is_true(new_lines_obj) };
            unsafe {
                (*frame).getorcreatedebug(init_lineno).f_trace_lines = new_lines;
            }
        }
    }
    if let Ok(new_opcodes_obj) = crate::baseobjspace::getattr(w_frame, "f_trace_opcodes") {
        if !new_opcodes_obj.is_null() {
            let new_opcodes = unsafe { crate::baseobjspace::is_true(new_opcodes_obj) };
            unsafe {
                (*frame).getorcreatedebug(init_lineno).f_trace_opcodes = new_opcodes;
            }
        }
    }
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
        // pypy/interpreter/pyframe.py:154 fget_f_back walks the
        // f_backref vref to materialise the parent frame.  Pyre's
        // wrapper has to do the same eagerly because the wrapper is a
        // plain pyre_object instance (no `__getattr__` slot wired to
        // the live struct yet — Task #224).  Recursion depth tracks
        // the live stack depth, which is bounded by Python's
        // recursion limit; the wrappers are short-lived (allocated
        // per callback invocation) so the per-trace overhead scales
        // with stack depth rather than total executed bytecodes.
        let f_back_obj = wrap_trace_frame(frame_ref.get_f_back());
        let _ = crate::baseobjspace::setattr(w_frame, "f_code", frame_ref.pycode as PyObjectRef);
        let _ = crate::baseobjspace::setattr(w_frame, "f_back", f_back_obj);
        // pyframe.py:768-771 fget_f_builtins → self.get_builtin().getdict(space)
        let _ = crate::baseobjspace::setattr(w_frame, "f_builtins", frame_ref.fget_f_builtins());
        // `pyframe.py:766 fget_f_globals` returns `self.w_globals`
        // directly — same identity as `module.__dict__` so trace
        // hooks (`sys.settrace`) observe `frame.f_globals is
        // module.__dict__`.  Pyre routes through
        // `dict_storage_to_dict` which returns the canonical
        // W_DictObject paired with the storage (mirror_target
        // invariant); a fresh `w_dict_new_with_dict_storage`
        // wrapper per trace callback would silently break that
        // identity.  `f_locals` follows the same shape (PyPy
        // `pyframe.py:546 fast2locals` then `self.debugdata.w_locals`).
        let _ = crate::baseobjspace::setattr(
            w_frame,
            "f_globals",
            if w_globals.is_null() {
                pyre_object::w_none()
            } else {
                crate::baseobjspace::dict_storage_to_dict(w_globals as *const DictStorage)
            },
        );
        let _ = crate::baseobjspace::setattr(
            w_frame,
            "f_locals",
            if w_locals.is_null() {
                pyre_object::w_none()
            } else {
                crate::baseobjspace::dict_storage_to_dict(w_locals as *const DictStorage)
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
        // pyframe.py:796-806 fget_f_trace_lines / fget_f_trace_opcodes
        // — exposed as bools.  PyPy stores the live values on the
        // frame's debug data; pyre's wrapper mirrors them so the user
        // callback can read or set them.  flush_trace_frame_writeback
        // copies the wrapper's post-callback values back onto the
        // live frame.
        let _ = crate::baseobjspace::setattr(
            w_frame,
            "f_trace_lines",
            pyre_object::w_bool_from(frame_ref.get_f_trace_lines()),
        );
        let _ = crate::baseobjspace::setattr(
            w_frame,
            "f_trace_opcodes",
            pyre_object::w_bool_from(frame_ref.get_f_trace_opcodes()),
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
    /// Optional W_DictObject to mirror str-keyed writes/deletes back
    /// into.  When non-null, `insert(name, value)` also updates the
    /// W_DictObject's entries Vec via `w_dict_setitem_str_no_proxy`
    /// and `remove(name)` via `w_dict_delitem_str_no_proxy`.  The
    /// no-proxy helpers skip the forward storage-store hook, so
    /// pairing the W_DictObject's `dict_storage_proxy` with this
    /// `mirror_target` produces a non-cyclic bidirectional sync —
    /// the structural stand-in for PyPy's single `W_DictMultiObject`
    /// (`pypy/objspace/std/dictobject.py`) that owns both halves.
    /// The exec/eval globals path uses this to drop the post-exec
    /// drain loop entirely (`pypy/interpreter/pyopcode.py:771-776`
    /// runs the frame on the user dict directly).
    mirror_target: pyre_object::PyObjectRef,
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
            // Cloned storages do not inherit a mirror target — the
            // back-mirror is bound to a specific W_DictObject and
            // copying the storage would have two storages racing for
            // ownership of the same entries Vec.  Callers that need a
            // mirror on the clone re-attach explicitly.
            mirror_target: pyre_object::PY_NULL,
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
            mirror_target: pyre_object::PY_NULL,
        }
    }

    /// Bind a W_DictObject as the back-mirror target for str-keyed
    /// writes and deletes.  See the field doc-comment for the
    /// PyPy-parity rationale.  Pass `pyre_object::PY_NULL` to detach.
    #[inline]
    pub fn set_mirror_target(&mut self, target: pyre_object::PyObjectRef) {
        self.mirror_target = target;
    }

    /// Read the currently bound back-mirror target (`PY_NULL` when
    /// none).  Provided for invariant assertions in tests / callers
    /// that need to detach symmetrically.
    #[inline]
    pub fn mirror_target(&self) -> pyre_object::PyObjectRef {
        self.mirror_target
    }

    /// Reallocate `values` to fit at least `min_cap` entries, copying
    /// the live prefix. Upstream `_ll_list_resize_really`
    /// (rlist.py:262-267) parity.
    unsafe fn grow(&mut self, min_cap: usize) {
        unsafe {
            let current_cap = items_block_capacity(self.values);
            let target_cap = min_cap.max(current_cap.saturating_mul(2).max(4));
            self.values = grow_list_items_block(self.values, target_cap, self.length);
        }
    }

    unsafe fn push(&mut self, value: PyObjectRef) {
        unsafe {
            if self.length == items_block_capacity(self.values) {
                self.grow(self.length + 1);
            }
            let base = items_block_items_base(self.values);
            *base.add(self.length) = value;
            self.length += 1;
        }
    }

    unsafe fn remove_at(&mut self, idx: usize) -> PyObjectRef {
        unsafe {
            assert!(idx < self.length);
            let base = items_block_items_base(self.values);
            let value = *base.add(idx);
            let p = base.add(idx);
            std::ptr::copy(p.add(1), p, self.length - idx - 1);
            self.length -= 1;
            value
        }
    }

    unsafe fn values_slice(&self) -> &[PyObjectRef] {
        unsafe {
            let base = items_block_items_base(self.values);
            std::slice::from_raw_parts(base, self.length)
        }
    }

    unsafe fn values_slice_mut(&mut self) -> &mut [PyObjectRef] {
        unsafe {
            let base = items_block_items_base(self.values);
            std::slice::from_raw_parts_mut(base, self.length)
        }
    }

    /// Replace the entire values backing with a freshly allocated
    /// empty block. Used by `clear()`.
    unsafe fn reset_values(&mut self) {
        unsafe {
            dealloc_list_items_block(self.values);
            self.values = alloc_list_items_block(&[]);
            self.length = 0;
        }
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
        if !self.mirror_target.is_null() {
            unsafe {
                pyre_object::dictobject::w_dict_setitem_str_no_proxy(
                    self.mirror_target,
                    name,
                    value,
                );
            }
        }
        value
    }

    pub fn insert(&mut self, name: String, value: PyObjectRef) -> Option<PyObjectRef> {
        let result = if let Some(idx) = self.slot_of(&name) {
            let slice = unsafe { self.values_slice_mut() };
            let old = slice[idx];
            slice[idx] = value;
            if old != value {
                self.notify_slot_watchers(idx);
            }
            Some(old)
        } else {
            self.names.push(name.clone());
            unsafe { self.push(value) };
            self.slot_watchers.push(Vec::new());
            None
        };
        // Back-mirror to the bound W_DictObject so storage-side writes
        // are visible to Python-level dict operations on the user dict
        // (`exec("g['x']=1", g)` followed by `g['x']` succeeds without
        // a post-exec drain).  The no-proxy variant skips the forward
        // store hook so we don't bounce the same write back into
        // `self`.
        if !self.mirror_target.is_null() {
            unsafe {
                pyre_object::dictobject::w_dict_setitem_str_no_proxy(
                    self.mirror_target,
                    &name,
                    value,
                );
            }
        }
        result
    }

    /// Remove a key from the backing dict storage (PyPy: space.delitem).
    /// This performs a real deletion rather than leaving a tombstone.
    pub fn remove(&mut self, name: &str) -> Option<PyObjectRef> {
        let result = if let Some(idx) = self.slot_of(name) {
            self.notify_slot_watchers(idx);
            self.names.remove(idx);
            let old = unsafe { self.remove_at(idx) };
            self.slot_watchers.remove(idx);
            Some(old)
        } else {
            None
        };
        if !self.mirror_target.is_null() {
            unsafe {
                pyre_object::dictobject::w_dict_delitem_str_no_proxy(self.mirror_target, name);
            }
        }
        result
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

pub const TICK_COUNTER_STEP: usize = 100;

#[derive(Debug, Default)]
pub struct WRootFinalizerQueue;

impl WRootFinalizerQueue {
    pub fn finalizer_trigger(&mut self) {}

    /// `pypy/interpreter/executioncontext.py:640` —
    /// `self.space.finalizer_queue.next_dead()`.
    ///
    /// Returns the next `w_obj` whose finalizer should run, or `None`
    /// when the death queue is empty.
    ///
    /// PRE-EXISTING-ADAPTATION: the real death queue requires the GC
    /// integration epic (Task #224 PyFrame=PyObject) to land a
    /// `WRootFinalizerQueue` instance backed by `rgc.FinalizerQueue`.
    /// Until then this is a constant-`None` stub so callers
    /// (`UserDelAction::_run_finalizers`) can mirror PyPy's loop shape
    /// line-by-line.
    pub fn next_dead(&mut self) -> Option<PyObjectRef> {
        None
    }
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
        // Register the storage ↔ W_DictObject sync hooks before any
        // module / dict allocation observes a missed mirror.  PyPy's
        // single `W_DictMultiObject` owns both views; pyre's split
        // requires the hooks to be live so that early construction
        // (`get_builtin()` calling `w_module_new("builtins", ns)` —
        // which writes `__name__` via `w_dict_setitem_str`) propagates
        // into the storage instead of silently being dropped on the
        // floor.  `Once` makes this idempotent across multiple ECs.
        static HOOKS_INSTALLED: std::sync::Once = std::sync::Once::new();
        HOOKS_INSTALLED.call_once(crate::call::install_dict_storage_hooks);
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

    /// executioncontext.py:113-115 — `c_call_trace(self, frame, w_func, args=None)`.
    pub fn c_call_trace(
        &mut self,
        frame: *mut PyFrame,
        w_func: PyObjectRef,
        args: Option<&crate::argument::Arguments>,
    ) -> Result<(), crate::PyError> {
        self._c_call_return_trace(frame, w_func, args, "c_call")
    }

    /// executioncontext.py:117-119 — `c_return_trace(self, frame, w_func, args=None)`.
    pub fn c_return_trace(
        &mut self,
        frame: *mut PyFrame,
        w_func: PyObjectRef,
        args: Option<&crate::argument::Arguments>,
    ) -> Result<(), crate::PyError> {
        self._c_call_return_trace(frame, w_func, args, "c_return")
    }

    /// executioncontext.py:121-136 — `_c_call_return_trace`.
    pub fn _c_call_return_trace(
        &mut self,
        frame: *mut PyFrame,
        mut w_func: PyObjectRef,
        args: Option<&crate::argument::Arguments>,
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
        // executioncontext.py:128-134 FunctionWithFixedCode method-call
        // rebinding.  PyPy:
        //   if isinstance(w_func, FunctionWithFixedCode) and args is not None:
        //       w_firstarg = args.firstarg()
        //       if w_firstarg is not None:
        //           w_func = descr_function_get(self.space, w_func,
        //                                       w_firstarg, self.space.type(w_firstarg))
        // BuiltinFunction (function.py:786, the module-level
        // builtin sibling) is intentionally excluded — function.py
        // splits the two so that "builtin function binds
        // differently" (no descriptor rebinding).  Pyre's
        // `is_function_with_fixed_code` (FUNCTION_TYPE && !can_change_code)
        // is the line-by-line port of the isinstance check.
        if let Some(args) = args {
            unsafe {
                if crate::is_function_with_fixed_code(w_func) {
                    if let Some(w_firstarg) = args.firstarg() {
                        if !w_firstarg.is_null() {
                            let w_type =
                                crate::typedef::r#type(w_firstarg).unwrap_or(pyre_object::PY_NULL);
                            w_func = crate::descr_function_get(w_func, w_firstarg, w_type);
                        }
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
        // executioncontext.py:158-165 bytecode_trace:
        //   def bytecode_trace(self, frame, decr_by=TICK_COUNTER_STEP):
        //       self.bytecode_only_trace(frame)
        //       actionflag = self.space.actionflag
        //       if actionflag.decrement_ticker(decr_by) < 0:
        //           actionflag.action_dispatcher(self, frame)
        //
        // bytecode_only_trace runs first; if it raises (tracefunc
        // callback exception), the ticker decrement + slow-path
        // action_dispatcher do NOT run.  Use `?` so a tracer error
        // short-circuits before touching actionflag.
        self.bytecode_only_trace(frame)?;
        if self.actionflag.decrement_ticker(decr_by as isize) < 0 {
            let self_ptr = self as *mut ExecutionContext;
            self.actionflag.action_dispatcher(self_ptr, frame);
        }
        Ok(())
    }

    /// pypy/interpreter/executioncontext.py:173-184 `bytecode_only_trace`.
    ///
    /// ```python
    /// def bytecode_only_trace(self, frame):
    ///     if self.space.reverse_debugging:
    ///         self._revdb_potential_stop_point(frame)
    ///     if (frame.get_w_f_trace() is None or self.is_tracing or
    ///         self.gettrace() is None):
    ///         return
    ///     self.run_trace_func(frame)
    /// ```
    ///
    /// reverse_debugging is not implemented in pyre, so the
    /// `_revdb_potential_stop_point` arm reduces to a no-op.  All
    /// three short-circuit conditions stay — including
    /// `frame.get_w_f_trace() is None`, which avoids the trailing
    /// `instr_prev_plus_one` write inside `run_trace_func` when the
    /// per-frame trace is unset (so a later `frame.f_trace = cb`
    /// observes the unmodified instr_prev_plus_one and can fire its
    /// first `line` event correctly).
    pub fn bytecode_only_trace(&mut self, frame: *mut PyFrame) -> Result<(), crate::PyError> {
        if self.space.is_null() || frame.is_null() {
            return Ok(());
        }
        let f_trace_is_none = unsafe { (*frame).get_w_f_trace().is_null() };
        if f_trace_is_none || self.is_tracing != 0 || self.w_tracefunc.is_null() {
            return Ok(());
        }
        self.run_trace_func(frame)
    }

    pub fn _run_finalizers_now(&mut self) {
        let _ = self;
    }

    /// pypy/interpreter/executioncontext.py:185-200 `run_trace_func`.
    ///
    /// ```python
    /// def run_trace_func(self, frame):
    ///     code = frame.pycode
    ///     if frame.last_instr == -1:
    ///         return     # don't trace the SETUP_ANNOTATIONS at the very start
    ///     d = frame.getorcreatedebug()
    ///     if d.is_in_line_tracing or d.f_trace_lines:
    ///         lastline = d.f_lineno
    ///         lineno = code._get_lineno_for_pc_tracing(frame.last_instr)
    ///         if lastline != lineno or frame.last_instr < d.instr_prev_plus_one:
    ///             self._trace(frame, 'line', self.space.w_None)
    ///     if d.f_trace_opcodes:
    ///         self._trace(frame, 'opcode', self.space.w_None)
    ///     d.instr_prev_plus_one = frame.last_instr + 1
    /// ```
    pub fn run_trace_func(&mut self, frame: *mut PyFrame) -> Result<(), crate::PyError> {
        if self.space.is_null() || frame.is_null() {
            return Ok(());
        }
        // executioncontext.py:189-192:
        //   d = frame.getorcreatedebug()
        //   lastline = d.f_lineno
        //   lineno = frame.pycode._get_lineno_for_pc_tracing(frame.last_instr)
        //   d.f_lineno = lineno
        let last_instr = unsafe { (*frame).last_instr };
        let lineno = unsafe { (*frame).get_last_lineno() };
        let (lastline, want_line, want_opcode) = unsafe {
            let d = (*frame).getorcreatedebug(lineno);
            let lastline = d.f_lineno;
            // executioncontext.py:192 d.f_lineno = lineno — PERSISTENT
            // write so subsequent run_trace_func invocations see the
            // current line (not the previous).
            d.f_lineno = lineno;
            // executioncontext.py:193-197:
            //   if d.f_trace_lines and lineno != -1:
            //       if lastline != lineno or frame.last_instr < d.instr_prev_plus_one:
            //           self._trace(frame, 'line', self.space.w_None)
            //   if d.f_trace_opcodes:
            //       self._trace(frame, 'opcode', self.space.w_None)
            let want_line = d.f_trace_lines
                && lineno != -1
                && (lastline != lineno || last_instr < d.instr_prev_plus_one);
            let want_opcode = d.f_trace_opcodes;
            (lastline, want_line, want_opcode)
        };
        let _ = lastline;
        if want_line {
            self._trace(frame, "line", pyre_object::w_none(), None)?;
        }
        if want_opcode {
            self._trace(frame, "opcode", pyre_object::w_none(), None)?;
        }
        // executioncontext.py:200 — record the next-PC sentinel so
        // backward jumps re-fire the line event even when staying on
        // the same source line.
        unsafe {
            (*frame).getorcreatedebug(lineno).instr_prev_plus_one = last_instr + 1;
        }
        Ok(())
    }

    /// pypy/interpreter/executioncontext.py:202-208
    /// `bytecode_trace_after_exception`.
    ///
    /// ```python
    /// def bytecode_trace_after_exception(self, frame):
    ///     "Like bytecode_trace(), but without increasing the ticker."
    ///     actionflag = self.space.actionflag
    ///     self.bytecode_only_trace(frame)
    ///     if actionflag.get_ticker() < 0:
    ///         actionflag.action_dispatcher(self, frame)     # slow path
    /// ```
    ///
    /// pyre's action_dispatcher slow path is not yet wired (the
    /// pre-existing stub only decremented the ticker once); the
    /// `bytecode_only_trace` call mirrors upstream so a tracer error
    /// surfaces through this helper's `Result`.
    pub fn bytecode_trace_after_exception(
        &mut self,
        frame: *mut PyFrame,
    ) -> Result<(), crate::PyError> {
        self.bytecode_only_trace(frame)?;
        if self.actionflag.get_ticker() < 0 {
            // executioncontext.py:207-208 — `if actionflag.get_ticker()
            // < 0: actionflag.action_dispatcher(self, frame)`. Pyre's
            // ActionFlag::action_dispatcher is a stub today (the
            // AsyncAction queue / signal handler dispatch is still a
            // pending port), but the call surface is wired so the
            // line-by-line shape stays correct.
            let self_ptr = self as *mut ExecutionContext;
            self.actionflag.action_dispatcher(self_ptr, frame);
        }
        Ok(())
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
    /// `_trace` consumes the live `OperationError` (executioncontext.py:
    /// 359-363) and mutates it in place via `normalize_exception`.
    /// Pyre's pyopcode call site does not yet hand the live operr to
    /// `exception_trace` — it forwards a `(w_type, w_value, w_traceback)`
    /// triple — so this wrapper fabricates a temporary `OperationError`
    /// whose lifetime spans the `_trace` call.  PyPy's `pyopcode.py:148
    /// ec.exception_trace(self, operr)` passes the live caller-held
    /// `operr`, but the post-call mutation is unobserved in pyre because
    /// the temp falls out of scope here.  Slice 3 of the OperationError
    /// port (caller threads its live operr through) will close that gap.
    pub fn exception_trace(
        &mut self,
        frame: *mut PyFrame,
        w_type: PyObjectRef,
        w_value: PyObjectRef,
        w_traceback: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        if !self.gettrace().is_null() {
            let mut operr = crate::error::OperationError::new(w_type, w_value);
            operr._application_traceback = if w_traceback.is_null() {
                None
            } else {
                Some(w_traceback)
            };
            self._trace(frame, "exception", w_value, Some(&mut operr))?;
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
    /// `operr` carries the live `OperationError` instance that
    /// `executioncontext.py:359-363` reads via `operr.w_type`,
    /// `operr.normalize_exception(space)`, and
    /// `operr.get_w_traceback(space)`.  Pyre's `error::OperationError`
    /// is the line-by-line port of `error.OperationError` (the same
    /// `w_type` / `w_value` / `_application_traceback` shape).
    /// Reading the fields here mirrors PyPy 1:1 — `operr.w_type` for
    /// the type, `get_w_value(space)` for the (possibly-normalized)
    /// value, `_application_traceback` for the traceback (or
    /// `space.w_None` when absent).
    pub fn _trace(
        &mut self,
        frame: *mut PyFrame,
        event: &str,
        w_arg: PyObjectRef,
        operr: Option<&mut crate::error::OperationError>,
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
            // executioncontext.py:359-363:
            //   if operr is not None:
            //       w_value = operr.normalize_exception(space)
            //       w_arg = space.newtuple([operr.w_type, w_value,
            //                               operr.get_w_traceback(space)])
            //
            // PyPy `normalize_exception` mutates the caller's operr in
            // place (error.py:247 `self.w_type = w_type`); after the
            // call `operr.w_type` is the normalized class.  Pyre takes
            // `&mut OperationError` here so the same mutation reaches
            // the caller's instance instead of a throw-away clone.
            let w_arg = if let Some(operr) = operr {
                let w_value = operr.normalize_exception(space)?;
                let w_type = if operr.w_type.is_null() {
                    crate::typedef::r#type(w_value).unwrap_or_else(pyre_object::w_none)
                } else {
                    operr.w_type
                };
                let w_traceback = operr
                    ._application_traceback
                    .unwrap_or_else(pyre_object::w_none);
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
                unsafe { (*frame).fast2locals()? };
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
                let call_result = crate::call::call_function_impl_result(
                    w_callback,
                    &[frame_obj, w_event, w_arg],
                );
                // Mirror in-callback `frame.f_trace = local` /
                // `frame.f_lineno = N` mutations back onto the live
                // PyFrame before processing w_result. PyPy passes the
                // raw frame to the callback, so its setattrs land
                // directly; pyre runs the callback against a wrapper
                // and must propagate explicitly.
                flush_trace_frame_writeback(frame, frame_obj, init_lineno);
                let w_result = call_result?;
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
            // executioncontext.py:401-402 reads `d.w_locals is not None`
            // again inside finally — the callback may have installed or
            // cleared w_locals.  Re-read here instead of caching
            // `had_locals` from before the call.
            let post_had_locals = unsafe {
                let d = (*frame).getorcreatedebug(init_lineno);
                !d.w_locals.is_null()
            };
            if post_had_locals {
                unsafe { (*frame).locals2fast(false)? };
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
    /// PRE-EXISTING-ADAPTATION vs `pypy/interpreter/main.py:43-45`: PyPy
    /// emits `space.setitem(w_globals, '__builtins__', space.builtin)`
    /// PyPy `pyopcode.py:773-774 setdefault('__builtins__', ...)` and
    /// the `Module.__init__` flow seed a freshly-imported module's
    /// dict with only `__builtins__`; reaching `print`/`len`/... is
    /// the LOAD_GLOBAL builtins fallback path `pyopcode.py:918-927
    /// frame.get_builtin().getdictvalue(...)`.  Pyre mirrors this
    /// shape: the new storage starts empty and only the
    /// `__builtins__` Module pointer is seeded.
    ///
    /// The caller is responsible for leaking the result via
    /// `Box::into_raw` so it can be shared across frames as a raw
    /// pointer.
    pub fn fresh_dict_storage(&self) -> DictStorage {
        // Clone the EC's builtins storage so every new frame's globals
        // start with all builtins inlined (`print`, `abs`, ...).  The
        // dedicated `__builtins__` slot is then overwritten with the
        // picked Module identity — `pyopcode.py:773-774` /
        // `moduledef.py:89-109 pick_builtin` look it up via
        // `space.getitem(w_globals, '__builtins__')` and expect a
        // Module (or dict subclass) reference, not a raw storage view.
        //
        // The clone (vs. the simpler "seed only `__builtins__` and let
        // LOAD_GLOBAL fall back to `frame.builtin`") is load-bearing
        // for JIT-compiled trace stability: traces that read globals
        // by name see the entries vector populated up-front, and the
        // shape stays stable across frames so bridges can reconnect to
        // the parent loop.  An empty-globals start triggers per-frame
        // shape divergence and bridge-to-parent reconnect failures
        // that cascade into blackhole interpretation of the whole
        // user loop.
        let mut ns = self.builtins.clone();
        let w_builtin = self.get_builtin();
        if !w_builtin.is_null() {
            crate::dict_storage_store(&mut ns, "__builtins__", w_builtin);
        }
        ns
    }

    /// `pypy/module/__builtin__/moduledef.py:Module.__init__` — `space.builtin`
    /// is a `Module` (not a dict).  Lazily build a `W_ModuleObject` whose
    /// backing storage IS `self.builtins`, so subsequent
    /// `module.getdict(space)` access (`pyframe.py:770 fget_f_builtins`)
    /// surfaces the same storage as a dict view.  The cache field stores
    /// the Module identity so identity-sensitive callers (PyPy
    /// `pick_builtin` `if w_builtin is space.builtin: return space.builtin`)
    /// observe the same object every call.
    pub fn get_builtin(&self) -> PyObjectRef {
        let cached = self.builtin_dict_cache.get();
        if !cached.is_null() {
            return cached;
        }
        let ns_ptr = &self.builtins as *const DictStorage as *mut u8;
        // PyPy `module.py:77 Module.getdict()` parity: single
        // `W_DictMultiObject` per storage.  Reuse the canonical
        // W_DictObject already paired (via `dict_storage_to_dict`'s
        // lazy mirror_target registration) instead of allocating a
        // fresh one through `w_module_new` (which would create a
        // sibling W_DictObject and conflict with any earlier
        // `dict_storage_to_dict` call on this storage).  The cast
        // strips `&self` const; safe because the EC keeps `self.builtins`
        // alive for the program's lifetime and `mirror_target` is the
        // only field touched.
        let canonical = crate::baseobjspace::dict_storage_to_dict(ns_ptr as *const DictStorage);
        // `module.py:24 self.setdictvalue(space, '__name__', w_name)` —
        // a single setitem inside `Module.__init__`.  Pyre routes that
        // seed through `w_module_new_aliasing_dict`, which performs the
        // setitem on the supplied W_DictObject when `name` is non-empty.
        // Issuing a second `w_dict_setitem_str` here would be an extra
        // storage-watcher invalidation with no PyPy counterpart.
        let module = pyre_object::w_module_new_aliasing_dict("builtins", ns_ptr, canonical);
        self.builtin_dict_cache.set(module);
        module
    }

    /// Direct lookup into the builtins storage, bypassing the dict-object
    /// wrapper (whose internal hash table does not see entries inserted
    /// through `install_default_builtins` on the underlying DictStorage).
    /// Used by `LOAD_GLOBAL` / `LOAD_NAME` to reach builtins like `print`
    /// when the frame's `w_globals` lacks the name (pypy/interpreter/
    /// pyopcode.py:558-565 LOAD_GLOBAL builtin fallback).
    pub fn lookup_builtin(&self, name: &str) -> Option<PyObjectRef> {
        crate::dict_storage_get(&self.builtins, name)
    }

    /// Default storage that `pick_builtin` selects when `globals
    /// ['__builtins__']` is absent — equivalent to PyPy's
    /// `space.builtin` fallback in `pick_builtin` (`pypy/module/
    /// __builtin__/moduledef.py:107-109`).
    pub fn builtins_storage_ptr(&self) -> *mut DictStorage {
        &self.builtins as *const DictStorage as *mut DictStorage
    }
}

#[derive(Clone)]
pub struct AbstractActionFlag {
    _periodic_actions: Vec<*mut dyn AsyncActionOps>,
    _nonperiodic_actions: Vec<*mut dyn AsyncActionOps>,
    pub(crate) _fired_bitmask: usize,
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

    /// pypy/interpreter/executioncontext.py:531-556
    /// `_rebuild_action_dispatcher`.
    ///
    /// PRE-EXISTING-ADAPTATION (justified): PyPy rebuilds an inner
    /// closure on every `register_*_action` call so that
    /// `periodic_actions = unrolling_iterable(self._periodic_actions)`
    /// captures the current periodic-action list at codegen time;
    /// RPython's JIT then unrolls the periodic loop at trace time.
    /// `unrolling_iterable` is an RPython-only JIT hint with no
    /// runtime semantic effect — the closure body is identical to a
    /// straightforward `for action in self._periodic_actions: …`.
    ///
    /// Pyre's `action_dispatcher` (`ActionFlagOps` trait default,
    /// `executioncontext.rs:1537`) iterates `_periodic_actions`
    /// directly each call.  Rebuilding the closure would produce no
    /// observable change because (a) majit's loop optimizer subsumes
    /// the unroll role for hot dispatchers and (b) the periodic list
    /// is small (typically 1–3 entries — release-the-GIL,
    /// report-the-signals).  Empty body preserved as the canonical
    /// "rebuild is a no-op in pyre" marker.
    pub fn _rebuild_action_dispatcher(&mut self) {}
}

/// `pypy/interpreter/executioncontext.py:458-585` — `AbstractActionFlag`
/// + `ActionFlag` virtual-dispatch interface.
///
/// PyPy's `AbstractActionFlag.fire` (line 482), `setcheckinterval`
/// (line 522), and `action_dispatcher` (line 531) all call
/// `self.reset_ticker(...)`, which Python resolves through the
/// concrete subclass override (`ActionFlag.reset_ticker` at line 574
/// writes `self._ticker = value`; `SignalActionFlag.reset_ticker`
/// writes a C global).  Rust composition cannot virtual-dispatch
/// from a base-struct inherent method, so this trait carries the
/// virtual surface (`reset_ticker` / `get_ticker` /
/// `decrement_ticker`) plus an accessor pair (`abstract_flag` /
/// `abstract_flag_mut`) that lets the default `fire` /
/// `setcheckinterval` / `action_dispatcher` implementations reach
/// the shared `_periodic_actions` / `_nonperiodic_actions` /
/// `_fired_bitmask` / `checkinterval_scaled` state.  Concrete
/// subclasses (`ActionFlag`) implement only the required methods;
/// the default bodies stay a line-for-line port of upstream.
pub trait ActionFlagOps {
    /// pypy/interpreter/executioncontext.py:574-575 `ActionFlag.reset_ticker`.
    fn reset_ticker(&mut self, value: isize);

    /// pypy/interpreter/executioncontext.py:571-572 `ActionFlag.get_ticker`.
    fn get_ticker(&self) -> isize;

    /// pypy/interpreter/executioncontext.py:577-585 `ActionFlag.decrement_ticker`.
    fn decrement_ticker(&mut self, by: isize) -> isize;

    /// Composition accessor: shared `AbstractActionFlag` state.
    fn abstract_flag(&self) -> &AbstractActionFlag;
    /// Composition accessor: shared `AbstractActionFlag` state.
    fn abstract_flag_mut(&mut self) -> &mut AbstractActionFlag;

    /// pypy/interpreter/executioncontext.py:493-510
    /// `register_periodic_action`.
    ///
    /// ```python
    /// if use_bytecode_counter:
    ///     self._periodic_actions.append(action)
    ///     self.has_bytecode_counter = True
    /// else:
    ///     self._periodic_actions.insert(0, action)
    /// self._rebuild_action_dispatcher()
    /// ```
    ///
    /// PyPy comment (line 503-504): "hack to put the release-the-GIL
    /// one at the end of the list, and the report-the-signals one at
    /// the start of the list."  When `use_bytecode_counter` is False
    /// (signal handling), the action is prepended so it runs first.
    ///
    /// Lives on the trait (PyPy hangs `register_periodic_action` on
    /// `AbstractActionFlag` itself, never overridden by `ActionFlag`)
    /// so `*mut dyn ActionFlagOps` callers — the type pyre uses for
    /// `space.actionflag` to admit a future `SignalActionFlag`
    /// analogue — can reach it without downcasting.
    ///
    /// PyPy line 502 asserts `isinstance(action, PeriodicAsyncAction)`.
    /// Pyre enforces the same constraint at compile time by typing the
    /// argument as `*mut dyn PeriodicAsyncActionOps` — non-periodic
    /// `AsyncAction` subclasses do not impl that subtrait, so passing
    /// one is a type error rather than a runtime assert failure.
    fn register_periodic_action(
        &mut self,
        action: *mut dyn PeriodicAsyncActionOps,
        use_bytecode_counter: bool,
    ) {
        // Trait upcast (stable since Rust 1.86) erases the periodic
        // marker so storage stays as the polymorphic
        // `*mut dyn AsyncActionOps` that `action_dispatcher` reads.
        let action_ops: *mut dyn AsyncActionOps = action;
        let flag = self.abstract_flag_mut();
        if use_bytecode_counter {
            flag._periodic_actions.push(action_ops);
            flag.has_bytecode_counter = true;
        } else {
            flag._periodic_actions.insert(0, action_ops);
        }
        flag._rebuild_action_dispatcher();
    }

    /// pypy/interpreter/executioncontext.py:512-517
    /// `register_nonperiodic_action`.  Trait default for the same
    /// reason as `register_periodic_action` above.
    fn register_nonperiodic_action(&mut self, action: *mut dyn AsyncActionOps) -> isize {
        let flag = self.abstract_flag_mut();
        flag._nonperiodic_actions.push(action);
        assert!(flag._nonperiodic_actions.len() < 32);
        flag._rebuild_action_dispatcher();
        (flag._nonperiodic_actions.len() - 1) as isize
    }

    /// pypy/interpreter/executioncontext.py:519-520 `getcheckinterval`.
    fn getcheckinterval(&self) -> usize {
        self.abstract_flag().checkinterval_scaled / TICK_COUNTER_STEP
    }

    /// pypy/interpreter/executioncontext.py:482-490 `fire`.
    ///
    /// ```python
    /// def fire(self, action):
    ///     "Request for the action to be run before the next opcode."
    ///     assert action._action_index >= 0
    ///     mask = r_uint(1) << action._action_index
    ///     if not self._fired_bitmask & mask:
    ///         self._fired_bitmask |= mask
    ///         # set the ticker to -1 in order to force action_dispatcher()
    ///         # to run at the next possible bytecode
    ///         self.reset_ticker(-1)
    /// ```
    fn fire(&mut self, action: *mut dyn AsyncActionOps) {
        if action.is_null() {
            return;
        }
        let base = unsafe { (*action).async_action() };
        // executioncontext.py:484 — `assert action._action_index >= 0`.
        debug_assert!(
            base._action_index >= 0,
            "periodic actions must not call fire"
        );
        let bitmask = base.bitmask;
        let already_fired = self.abstract_flag()._fired_bitmask & bitmask != 0;
        if !already_fired {
            self.abstract_flag_mut()._fired_bitmask |= bitmask;
            self.reset_ticker(-1);
        }
    }

    /// pypy/interpreter/executioncontext.py:522-529 `setcheckinterval`.
    ///
    /// ```python
    /// def setcheckinterval(self, interval):
    ///     MAX = sys.maxint // TICK_COUNTER_STEP
    ///     if interval < 1:
    ///         interval = 1
    ///     elif interval > MAX:
    ///         interval = MAX
    ///     self.checkinterval_scaled = interval * TICK_COUNTER_STEP
    ///     self.reset_ticker(-1)
    /// ```
    fn setcheckinterval(&mut self, interval: usize) {
        let max = usize::MAX / TICK_COUNTER_STEP;
        let interval = interval.max(1).min(max);
        self.abstract_flag_mut().checkinterval_scaled = interval * TICK_COUNTER_STEP;
        self.reset_ticker(-1);
    }

    /// pypy/interpreter/executioncontext.py:531-562 `action_dispatcher`.
    ///
    /// ```python
    /// def action_dispatcher(ec, frame):
    ///     # periodic actions (first reset the bytecode counter)
    ///     self.reset_ticker(self.checkinterval_scaled)
    ///     for action in periodic_actions:
    ///         action.perform(ec, frame)
    ///     # nonperiodic actions
    ///     if self._fired_bitmask:
    ///         for i in range(len(self._nonperiodic_actions)):
    ///             mask = r_uint(1) << i
    ///             if self._fired_bitmask & mask:
    ///                 action = self._nonperiodic_actions[i]
    ///                 self._fired_bitmask &= ~mask
    ///                 action.perform(ec, frame)
    ///         if self._fired_bitmask:
    ///             self.reset_ticker(-1)
    /// ```
    ///
    /// Pyre's `AsyncAction::perform` is a no-op stub for actions that
    /// have not yet been ported (CheckSignalAction, GcCollectAction,
    /// etc.); the iteration shape matches PyPy so the dispatch surface
    /// lands correctly when those actions are ported individually.
    fn action_dispatcher(&mut self, ec: *mut ExecutionContext, frame: *mut PyFrame) {
        // executioncontext.py:538 — `self.reset_ticker(self.checkinterval_scaled)`.
        let interval = self.abstract_flag().checkinterval_scaled as isize;
        self.reset_ticker(interval);
        // executioncontext.py:539-540 — periodic actions iter.
        // Snapshot pointers before iteration; perform() may register
        // additional actions and mutate `_periodic_actions`.
        let periodic: Vec<*mut dyn AsyncActionOps> = self.abstract_flag()._periodic_actions.clone();
        for action_ptr in periodic {
            if action_ptr.is_null() || ec.is_null() {
                continue;
            }
            unsafe {
                (*action_ptr).perform(&mut *ec, frame);
            }
        }
        // executioncontext.py:543-556 — nonperiodic bit-mask scan.
        // Clear each bit before perform() so a fire() during perform()
        // re-arms cleanly (NB at executioncontext.py:544-549).
        if self.abstract_flag()._fired_bitmask != 0 {
            let nactions = self.abstract_flag()._nonperiodic_actions.len();
            for i in 0..nactions {
                let mask: usize = 1usize << i;
                if self.abstract_flag()._fired_bitmask & mask != 0 {
                    let action_ptr = self.abstract_flag()._nonperiodic_actions[i];
                    self.abstract_flag_mut()._fired_bitmask &= !mask;
                    if !action_ptr.is_null() && !ec.is_null() {
                        unsafe {
                            (*action_ptr).perform(&mut *ec, frame);
                        }
                    }
                }
            }
            // executioncontext.py:560-561 — if a higher-index action
            // re-fired an earlier bit during iteration, force the next
            // bytecode to re-enter dispatch.
            if self.abstract_flag()._fired_bitmask != 0 {
                self.reset_ticker(-1);
            }
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

    pub fn perform_frame_action(&mut self, ec: &mut ExecutionContext, frame: *mut PyFrame) {
        let _ = (ec, frame);
    }
}

/// pypy/interpreter/executioncontext.py:566-585 — `class ActionFlag(AbstractActionFlag)`.
impl ActionFlagOps for ActionFlag {
    fn abstract_flag(&self) -> &AbstractActionFlag {
        &self.base
    }

    fn abstract_flag_mut(&mut self) -> &mut AbstractActionFlag {
        &mut self.base
    }

    /// pypy/interpreter/executioncontext.py:571-572 `get_ticker`.
    fn get_ticker(&self) -> isize {
        self._ticker
    }

    /// pypy/interpreter/executioncontext.py:574-575 `reset_ticker`:
    /// ```python
    /// def reset_ticker(self, value):
    ///     self._ticker = value
    /// ```
    fn reset_ticker(&mut self, value: isize) {
        self._ticker = value;
    }

    /// pypy/interpreter/executioncontext.py:577-585 `decrement_ticker`.
    ///
    /// ```python
    /// def decrement_ticker(self, by):
    ///     value = self._ticker
    ///     if self.has_bytecode_counter:
    ///         if jit.isconstant(by) and by == 0:
    ///             pass
    ///         else:
    ///             value -= by
    ///             self._ticker = value
    ///     return value
    /// ```
    fn decrement_ticker(&mut self, by: isize) -> isize {
        if self.base.has_bytecode_counter {
            self._ticker -= by;
        }
        self._ticker
    }
}

pub struct AsyncAction {
    pub space: PyObjectRef,
    _action_index: isize,
    /// pypy/interpreter/executioncontext.py:600 `self.bitmask = 1 << index`.
    /// Set by `AbstractActionFlag::register_nonperiodic_action` and
    /// consumed by `fire` / `action_dispatcher`'s bit test.
    pub bitmask: usize,
    /// pypy/interpreter/executioncontext.py:603-606 `AsyncAction.fire`
    /// uses `self.space.actionflag` — a constant lookup once the action
    /// is constructed because PyPy's `space.actionflag` is set once at
    /// `pypy/interpreter/baseobjspace.py:447` and never replaced.
    /// Pyre keeps the actionflag on `ExecutionContext` rather than on
    /// `space`, so we cache the back-reference at registration time
    /// (`AsyncAction::new` / `UserDelAction::new` /
    /// `AsyncActionOps::register_periodic_action`).  The pointer is
    /// stable for the process lifetime because pyrex owns the EC for
    /// the entire run.  `fire()` dereferences this slot directly,
    /// matching PyPy's `self.space.actionflag.fire(self)` 1:1 without
    /// the TLS detour.  `null` means the action has not been registered
    /// yet — calling `fire` in that state is a programmer error
    /// (PyPy's docstring at line 605 reads "The action must have been
    /// registered at space initalization time.").
    ///
    /// Stored as `*mut dyn ActionFlagOps` so that `fire()` can dispatch
    /// through the `ActionFlagOps::fire` trait default (which lives on
    /// the trait, mirroring PyPy's polymorphic `actionflag.fire` call)
    /// AND so a future `SignalActionFlag` analogue (PyPy's signal-aware
    /// `AbstractActionFlag` subclass) can be installed without touching
    /// `AsyncAction`.  PyPy types `space.actionflag` as the abstract
    /// base, never as the concrete subclass.
    pub actionflag: *mut dyn ActionFlagOps,
}

impl Default for AsyncAction {
    fn default() -> Self {
        Self {
            space: pyre_object::PY_NULL,
            _action_index: -1,
            bitmask: 0,
            // Fat pointer: null data + ActionFlag vtable.  `fire()` /
            // `register_periodic_action` tests `is_null()` on the data
            // pointer before any deref, so the placeholder vtable never
            // gets reached on an unregistered action.
            actionflag: std::ptr::null_mut::<ActionFlag>(),
        }
    }
}

impl AsyncAction {
    /// pypy/interpreter/executioncontext.py:594-600 `AsyncAction.__init__`.
    ///
    /// ```python
    /// def __init__(self, space):
    ///     self.space = space
    ///     if not isinstance(self, PeriodicAsyncAction):
    ///         self._action_index = self.space.actionflag.register_nonperiodic_action(self)
    ///     else:
    ///         self._action_index = -1
    /// ```
    ///
    /// Returns `Box<Self>` because Rust cannot hand out a stable
    /// pointer to a value before the constructor returns; PyPy's
    /// heap-identity makes `register_nonperiodic_action(self)` safe
    /// in-place.  Caller owns the Box and must keep it alive as long
    /// as the actionflag holds the registered pointer.
    pub fn new(space: PyObjectRef, actionflag: &mut (dyn ActionFlagOps + 'static)) -> Box<Self> {
        let mut action = Box::new(Self {
            space,
            _action_index: -1,
            bitmask: 0,
            actionflag: actionflag as *mut dyn ActionFlagOps,
        });
        let action_ptr: *mut dyn AsyncActionOps = &mut *action;
        let index = actionflag.register_nonperiodic_action(action_ptr);
        action._action_index = index;
        action.bitmask = 1usize << (index as usize);
        action
    }

    /// pypy/interpreter/executioncontext.py:597-600 `else` branch:
    /// `self._action_index = -1`.  Used by PeriodicAsyncAction
    /// subclasses whose `__init__` skips `register_nonperiodic_action`.
    /// The base `AsyncAction` is returned unboxed because
    /// `PeriodicAsyncAction::new` wraps it in a `Box<PeriodicAsyncAction>`
    /// — registration with `register_periodic_action` then captures the
    /// outer Box's stable pointer.
    pub fn new_periodic_base(space: PyObjectRef) -> Self {
        Self {
            space,
            _action_index: -1,
            bitmask: 0,
            // PeriodicAsyncAction subclasses receive their actionflag
            // back-reference at `register_periodic_action` time (the
            // trait default below) rather than at construction —
            // pyjitpl-style mirror of PyPy's two-step `__init__` +
            // `space.actionflag.register_periodic_action(...)` flow.
            //
            // Fat null pointer constructed via ActionFlag upcast — see
            // the `Default` impl above for the same pattern.
            actionflag: std::ptr::null_mut::<ActionFlag>(),
        }
    }
}

/// pypy/interpreter/executioncontext.py:588-609 — `AsyncAction` virtual surface.
///
/// PyPy resolves `action.perform(ec, frame)` through Python class
/// lookup (line 608-609 — base body is "To be overridden.").  Rust
/// composition cannot virtual-dispatch from a base-struct inherent
/// method, so this trait carries the virtual `perform` entry plus an
/// accessor pair (`async_action` / `async_action_mut`) for the shared
/// `_action_index` / `bitmask` / `space` state held on `AsyncAction`.
/// Concrete actions impl the trait; the action lists store
/// `*mut dyn AsyncActionOps` so the dispatcher reaches the override
/// through the embedded vtable, and `fire` / `action_dispatcher` can
/// reach the base bitmask through the accessor.
pub trait AsyncActionOps {
    /// pypy/interpreter/executioncontext.py:608-609 `AsyncAction.perform`:
    /// `def perform(self, executioncontext, frame): "To be overridden."`
    fn perform(&mut self, executioncontext: &mut ExecutionContext, frame: *mut PyFrame);

    /// Composition accessor: shared `AsyncAction` state.
    fn async_action(&self) -> &AsyncAction;
    /// Composition accessor: shared `AsyncAction` state.
    fn async_action_mut(&mut self) -> &mut AsyncAction;

    /// pypy/interpreter/executioncontext.py:602-606 `AsyncAction.fire`:
    ///
    /// ```python
    /// @rgc.no_collect
    /// def fire(self):
    ///     "Request for the action to be run before the next opcode."
    ///     self.space.actionflag.fire(self)
    /// ```
    ///
    /// Pyre stores `actionflag` on `ExecutionContext` rather than on
    /// `space` (PRE-EXISTING-ADAPTATION — pyre's space is an opaque
    /// `PyObjectRef`).  To keep PyPy's `self.space.actionflag.fire(self)`
    /// directness, the actionflag back-reference is cached on the base
    /// `AsyncAction` at registration time (see
    /// `AsyncAction::new` / `UserDelAction::new` / the trait default
    /// `register_periodic_action` above).  `fire()` dereferences that
    /// back-ref unconditionally; the pointer is stable for the process
    /// lifetime because pyrex owns the EC for the entire run.  The
    /// `Self: Sized + 'static` bound captures the concrete subclass
    /// vtable in the `*mut dyn AsyncActionOps` fat pointer so
    /// `actionflag.fire`'s `(*action).async_action()` reads through the
    /// override-aware vtable rather than the bare-base one.
    ///
    /// PyPy's docstring at line 605 reads "The action must have been
    /// registered at space initalization time."  Pyre asserts the same:
    /// `self.actionflag.is_null()` would mean the action skipped both
    /// `register_nonperiodic_action` and `register_periodic_action`,
    /// which is a programmer error.
    fn fire(&mut self)
    where
        Self: Sized + 'static,
    {
        let actionflag = self.async_action().actionflag;
        debug_assert!(
            !actionflag.is_null(),
            "AsyncAction::fire called before registration (executioncontext.py:605)"
        );
        let action_ptr: *mut dyn AsyncActionOps = self;
        // executioncontext.py:606 — `self.space.actionflag.fire(self)`.
        // The trait dispatch reaches the `ActionFlagOps::fire` default
        // body at line 1454 (which encodes executioncontext.py:482-490
        // `AbstractActionFlag.fire`).
        unsafe { (*actionflag).fire(action_ptr) };
    }
}

/// pypy/interpreter/executioncontext.py:588-609 — bare `AsyncAction`.
/// Default `perform` is the "To be overridden." no-op; concrete
/// subclasses override.
impl AsyncActionOps for AsyncAction {
    fn perform(&mut self, _executioncontext: &mut ExecutionContext, _frame: *mut PyFrame) {}

    fn async_action(&self) -> &AsyncAction {
        self
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        self
    }
}

/// pypy/interpreter/executioncontext.py:612-615 `class
/// PeriodicAsyncAction(AsyncAction)`.  Marker subtrait that narrows
/// `space.actionflag.register_periodic_action(self, ...)` so only
/// types descending from `PeriodicAsyncAction` (the trait analogue of
/// PyPy's class hierarchy) can be registered as periodic.  PyPy
/// guards the same constraint at runtime with
/// `assert isinstance(action, PeriodicAsyncAction)` in
/// `executioncontext.py:502`; pyre lifts the assertion to the type
/// system.
pub trait PeriodicAsyncActionOps: AsyncActionOps {
    /// pypy/interpreter/executioncontext.py:594-600 `else` branch
    /// helper: subclasses of `PeriodicAsyncAction` call
    /// `space.actionflag.register_periodic_action(self, ...)` after
    /// `__init__`.  Lifted into this subtrait so the
    /// `*mut dyn PeriodicAsyncActionOps` cast captures the concrete
    /// subclass vtable (which carries the override of `perform`);
    /// `Self: Sized` keeps trait-object dispatch off this method.
    fn register_periodic_action(
        &mut self,
        actionflag: &mut (dyn ActionFlagOps + 'static),
        use_bytecode_counter: bool,
    ) where
        Self: Sized + 'static,
    {
        // executioncontext.py:603-606 — cache the actionflag back-ref
        // on the base AsyncAction so `fire()` can reach it without a
        // TLS lookup.  PyPy's `self.space.actionflag` is a constant
        // lookup once `space.actionflag` is set at
        // `pypy/interpreter/baseobjspace.py:447` and never replaced;
        // storing the resolved pointer here is the same caching the
        // PyPy implementation does implicitly.
        self.async_action_mut().actionflag = actionflag as *mut dyn ActionFlagOps;
        let action_ptr: *mut dyn PeriodicAsyncActionOps = self;
        actionflag.register_periodic_action(action_ptr, use_bytecode_counter);
    }
}

pub struct PeriodicAsyncAction {
    pub base: AsyncAction,
}

impl PeriodicAsyncAction {
    /// pypy/interpreter/executioncontext.py:594-600 `AsyncAction.__init__`
    /// `else` arm (`isinstance(self, PeriodicAsyncAction)`): only sets
    /// `_action_index = -1`, leaves registration to subclass-specific
    /// `space.actionflag.register_periodic_action(self, use_bytecode_counter)`.
    ///
    /// Returns `Box<Self>` because `register_periodic_action` (called
    /// after construction by the subclass via `register`) takes a
    /// stable `*mut dyn AsyncActionOps`.
    pub fn new(space: PyObjectRef) -> Box<Self> {
        Box::new(Self {
            base: AsyncAction::new_periodic_base(space),
        })
    }

    // pypy/interpreter/executioncontext.py:594-600 — PeriodicAsyncAction
    // subclasses call `space.actionflag.register_periodic_action(self,
    // use_bytecode_counter)` after their own `__init__`.  In pyre this
    // is the trait default `AsyncActionOps::register_periodic_action`
    // — it must be reached through the concrete subclass's vtable so
    // the registered fat pointer's `perform` slot points to the
    // override, not `PeriodicAsyncAction::perform` (which is a no-op
    // per `executioncontext.py:612-615`).  Never call
    // `PeriodicAsyncAction(...).register_periodic_action(...)`
    // directly without a concrete subclass.
}

/// pypy/interpreter/executioncontext.py:612-615 — `class
/// PeriodicAsyncAction(AsyncAction)`: empty body, inherits the
/// "To be overridden." `perform` from `AsyncAction`.  Concrete
/// subclasses (CheckSignalAction etc.) override.
impl AsyncActionOps for PeriodicAsyncAction {
    fn perform(&mut self, _executioncontext: &mut ExecutionContext, _frame: *mut PyFrame) {}

    fn async_action(&self) -> &AsyncAction {
        &self.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base
    }
}

/// pypy/interpreter/executioncontext.py:612-615 — `PeriodicAsyncAction`
/// admits `space.actionflag.register_periodic_action(self, ...)`.
impl PeriodicAsyncActionOps for PeriodicAsyncAction {}

pub struct UserDelAction {
    pub base: AsyncAction,
    pub finalizers_lock_count: usize,
    pub enabled_at_app_level: bool,
    pub pending_with_disabled_del: Option<Vec<PyObjectRef>>,
    /// `pypy/interpreter/executioncontext.py:640` —
    /// `self.space.finalizer_queue` access target.
    ///
    /// PRE-EXISTING-ADAPTATION: PyPy reads the queue via `self.space`
    /// (typed `ObjSpace`).  Pyre's `space` is an opaque `PyObjectRef`,
    /// so the queue is held as a UserDelAction field instead.  When the
    /// GC integration epic (Task #224) lands a typed space surface this
    /// can be folded back into `space.finalizer_queue`.
    pub finalizer_queue: WRootFinalizerQueue,
}

impl UserDelAction {
    /// pypy/interpreter/executioncontext.py:618-631 `UserDelAction.__init__`.
    ///
    /// ```python
    /// class UserDelAction(AsyncAction):
    ///     def __init__(self, space):
    ///         AsyncAction.__init__(self, space)
    ///         self.finalizers_lock_count = 0
    ///         self.enabled_at_app_level = True
    ///         self.pending_with_disabled_del = None
    /// ```
    ///
    /// `AsyncAction.__init__` (executioncontext.py:594-600) registers
    /// the live instance as a nonperiodic action because UserDelAction
    /// is NOT a PeriodicAsyncAction subclass.  Pyre boxes the
    /// UserDelAction so registration captures a stable
    /// `*mut dyn AsyncActionOps` whose vtable dispatches into
    /// `UserDelAction::perform` (executioncontext.py:632-633).
    pub fn new(space: PyObjectRef, actionflag: &mut (dyn ActionFlagOps + 'static)) -> Box<Self> {
        let mut action = Box::new(Self {
            base: AsyncAction {
                space,
                _action_index: -1,
                bitmask: 0,
                // executioncontext.py:603-606 — cache actionflag
                // back-ref for `fire()` (mirror of PyPy's
                // `self.space.actionflag` constant lookup).
                actionflag: actionflag as *mut dyn ActionFlagOps,
            },
            finalizers_lock_count: 0,
            enabled_at_app_level: true,
            pending_with_disabled_del: None,
            finalizer_queue: WRootFinalizerQueue,
        });
        let action_ptr: *mut dyn AsyncActionOps = &mut *action;
        let index = actionflag.register_nonperiodic_action(action_ptr);
        action.base._action_index = index;
        action.base.bitmask = 1usize << (index as usize);
        action
    }

    /// `pypy/interpreter/executioncontext.py:636-643`:
    /// ```python
    /// def _run_finalizers(self):
    ///     # called by perform() when we have to "perform" this action,
    ///     # and also directly at the end of gc.collect).
    ///     while True:
    ///         w_obj = self.space.finalizer_queue.next_dead()
    ///         if w_obj is None:
    ///             break
    ///         self._call_finalizer(w_obj)
    /// ```
    ///
    /// `self.space.finalizer_queue` is read via the local
    /// `self.finalizer_queue` field (see struct doc for
    /// PRE-EXISTING-ADAPTATION).
    pub fn _run_finalizers(&mut self) {
        loop {
            let w_obj = self.finalizer_queue.next_dead();
            match w_obj {
                None => break,
                Some(w) => self._call_finalizer(w),
            }
        }
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

/// pypy/interpreter/executioncontext.py:618-633 — `class
/// UserDelAction(AsyncAction)`.  `perform` (line 632-633) calls
/// `self._run_finalizers()`.
impl AsyncActionOps for UserDelAction {
    fn perform(&mut self, _executioncontext: &mut ExecutionContext, _frame: *mut PyFrame) {
        self._run_finalizers();
    }

    fn async_action(&self) -> &AsyncAction {
        &self.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base
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

    #[test]
    fn test_fresh_dict_storage_clones_builtins_with_module_pointer() {
        // Every freshly-created frame's globals start with the EC's
        // builtins inlined (`print`, `abs`, ...), then `__builtins__`
        // is set to the picked Module (so `pick_builtin` /
        // `pyopcode.py:773-774` see a Module reference and
        // `f_builtins` returns the Module's dict).
        let ctx = PyExecutionContext::new();
        let namespace = ctx.fresh_dict_storage();

        let w_builtin = *namespace.get("__builtins__").unwrap();
        unsafe {
            assert!(pyre_object::is_module(w_builtin));
        }
        assert!(namespace.get("print").is_some());
        assert!(namespace.get("range").is_some());
        assert!(namespace.get("__name__").is_none());
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
    fn test_namespace_slot_watchers_align_for_removal() {
        let ctx = PyExecutionContext::new();
        let mut namespace = ctx.fresh_dict_storage();

        assert!(namespace.remove("__builtins__").is_some());
        assert!(namespace.get("__builtins__").is_none());
    }
}
