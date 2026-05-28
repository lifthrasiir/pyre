//! JitState implementation for pyre.
//!
//! `PyreJitState` bridges the interpreter's `PyFrame` with majit's JIT
//! infrastructure. It extracts live values from the frame, restores them
//! after compiled code runs, and provides the meta/sym types for tracing.

use majit_backend::Backend;
use majit_ir::{DescrRef, OpCode, OpRef, Type, Value};
use majit_metainterp::virtualizable::VirtualizableInfo;
use majit_metainterp::{
    BridgeInlineCarrier, JitDriverStaticData, JitState, ReconstructRecipe,
    ResidualVirtualizableSync, TraceAction, TraceCtx,
};

use pyre_interpreter::bytecode::{CodeObject, ComparisonOperator, Instruction};
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::truth_value as objspace_truth_value;
use pyre_object::PyObjectRef;
use pyre_object::boolobject::w_bool_get_value;
use pyre_object::pyobject::{
    FLOAT_TYPE, INT_TYPE, get_instantiate, is_bool, is_float, is_int, py_type_check,
};
use pyre_object::{PY_NULL, w_float_get_value, w_int_get_value, w_int_new};

/// jitcode.py:9-21 / codewriter.py:68: JitCode — compiled bytecode unit.
///
/// MetaInterpStaticData-side wrapper around the shared
/// [`crate::PyJitCode`] payload. The payload `Arc` is the same heap
/// allocation that `pyre_jit::jit::call::CallControl.jitcodes` holds
/// for the same CodeObject — RPython's `MetaInterpStaticData.jitcodes`
/// list and `CallControl.jitcodes` dict reference identical
/// `JitCode` Python objects through Python's refcount semantics, and
/// pyre mirrors that with a shared `Arc`. The wrapper keeps `code`
/// (so `code_for_jitcode_index` can recover the wrapping
/// `PyObjectRef` from a numeric index) and `index` (the SD-local
/// `jitcode.index = len(all_jitcodes)` from codewriter.py:68) on the
/// SD side, since neither is intrinsic to the `PyJitCode` payload.
// SAFETY: JitCode is only written once (during creation) and then
// read-only. The code pointer is stable for the program lifetime.
unsafe impl Sync for JitCode {}

pub(crate) struct JitCode {
    /// Pointer to the Code object (W_CodeObject).
    /// Matches frame.code and getcode(func).
    pub code: *const (),
    /// codewriter.py:68: jitcode.index = len(all_jitcodes).
    pub index: i32,
    /// Shared `PyJitCode` payload. Same `Arc` instance also lives in
    /// `CallControl.jitcodes`. Refinements (e.g. a `merge_point_pc`
    /// rebuild) replace this `Arc` in place so cached `*const JitCode`
    /// pointers see the refreshed payload on the next field access.
    pub payload: std::sync::Arc<crate::PyJitCode>,
}

impl JitCode {
    /// Extract raw CodeObject from the W_CodeObject stored in this JitCode.
    #[inline]
    pub unsafe fn raw_code(&self) -> *const CodeObject {
        unsafe {
            if self.code.is_null() {
                return std::ptr::null();
            }
            pyre_interpreter::w_code_get_ptr(self.code as pyre_object::PyObjectRef)
                as *const CodeObject
        }
    }
}

/// warmspot.py:148-282: MetaInterpStaticData — per-driver compile-time data.
///
/// RPython: created by WarmRunnerDesc, holds jitcodes list populated
/// by codewriter.make_jitcodes(). Accessed as MetaInterp.staticdata.
///
/// pyre: per-thread equivalent (no-GIL runtime). Populated from the
/// authoritative `CodeWriter.make_jitcodes()` result before tracing.
///
/// TODO: the RPython-orthodox
/// `MetaInterpStaticData` lives in
/// `majit_metainterp::pyjitpl::MetaInterpStaticData`.  pyre embeds it
/// as the `canonical` field below and delegates every RPython method
/// (`setup_indirectcalltargets`, `bytecode_for_address`, …) through
/// it, so there is exactly one port of each of those methods.  The
/// pyre-local fields (`jitcodes`, `finish_setup_done`, `op_*`,
/// `liveness_info`) stay here because the surrounding runtime still
/// stores Python-code metadata outside the canonical staticdata.
struct MetaInterpStaticData {
    /// warmspot.py:282: self.metainterp_sd.jitcodes = jitcodes.
    /// Box<JitCode> for address stability across vec growth.
    jitcodes: Vec<Box<JitCode>>,
    /// pyjitpl.py:2264 `self.liveness_info = "".join(asm.all_liveness)` —
    /// frozen snapshot of the assembler's `all_liveness` buffer. In RPython
    /// this is set once at `finish_setup` time; in pyre the assembler is
    /// long-lived and liveness accumulates across lazy JitCode compiles,
    /// so this field is resynced after every `intern_liveness` write.
    ///
    /// Stored as `Arc<[u8]>` so `liveness_info_snapshot()` can hand out
    /// shared read-only slices (`metainterp_sd.liveness_info` parity in
    /// resume.py:1022) without cloning the byte buffer per BH entry.
    liveness_info: std::sync::Arc<[u8]>,
    /// pyjitpl.py:2255 `finish_setup` is per MetaInterpStaticData instance.
    /// `METAINTERP_SD` is thread-local in pyre, so this guard must live on
    /// the thread-local object rather than in a process-global `Once`.
    ///
    /// Unlike RPython, pyre assembles lazily and the writer-side
    /// `Assembler.insns` table grows over time. The guard therefore marks
    /// "initial finish_setup done" only; later snapshots still refresh the
    /// cached opcode ids and liveness bytes in place.
    finish_setup_done: bool,

    // pyjitpl.py:2236-2243 opcode number cache filled by `setup_insns`.
    // RPython stores every field even when the runtime currently does
    // not read them, so the structural parity is preserved. Sentinel
    // `u8::MAX` matches `insns.get('…', -1)` for lookups that happen
    // before `setup_insns` runs (e.g. early fallback paths).
    /// pyjitpl.py:2236 `self.op_live = insns.get('live/', -1)`.
    op_live: u8,
    /// pyjitpl.py:2237 `self.op_goto = insns.get('goto/L', -1)`.
    op_goto: u8,
    /// pyjitpl.py:2238 `self.op_catch_exception = insns.get('catch_exception/L', -1)`.
    op_catch_exception: u8,
    /// pyjitpl.py:2239 `self.op_rvmprof_code = insns.get('rvmprof_code/ii', -1)`.
    op_rvmprof_code: u8,
    /// pyjitpl.py:2240 `self.op_int_return = insns.get('int_return/i', -1)`.
    op_int_return: u8,
    /// pyjitpl.py:2241 `self.op_ref_return = insns.get('ref_return/r', -1)`.
    op_ref_return: u8,
    /// pyjitpl.py:2242 `self.op_float_return = insns.get('float_return/f', -1)`.
    op_float_return: u8,
    /// pyjitpl.py:2243 `self.op_void_return = insns.get('void_return/', -1)`.
    op_void_return: u8,

    /// pyjitpl.py:2190 `class MetaInterpStaticData` — the canonical,
    /// RPython-orthodox port.  Owns `indirectcalltargets`,
    /// `globaldata.indirectcall_dict`, and every RPython staticdata
    /// method (`setup_indirectcalltargets`, `bytecode_for_address`,
    /// `setup_descrs`, `setup_list_of_addr2name`, …).  Reached from
    /// pyre's module-level wrappers (`state::setup_indirectcalltargets`,
    /// `state::bytecode_for_address`) and from future callers that
    /// hold a `&mut MetaInterpStaticData` directly.
    canonical: majit_metainterp::MetaInterpStaticData,
}

#[allow(dead_code)]
impl MetaInterpStaticData {
    fn new() -> Self {
        Self {
            jitcodes: Vec::new(),
            liveness_info: std::sync::Arc::<[u8]>::from(Vec::<u8>::new().into_boxed_slice()),
            finish_setup_done: false,
            op_live: u8::MAX,
            op_goto: u8::MAX,
            op_catch_exception: u8::MAX,
            op_rvmprof_code: u8::MAX,
            op_int_return: u8::MAX,
            op_ref_return: u8::MAX,
            op_float_return: u8::MAX,
            op_void_return: u8::MAX,
            canonical: majit_metainterp::MetaInterpStaticData::new(),
        }
    }

    /// pyjitpl.py:2248-2249 `setup_indirectcalltargets(indirectcalltargets)`.
    /// Thin delegate to the canonical port; exists solely so callers
    /// that hold `&mut state::MetaInterpStaticData` can reach the
    /// method without constructing a reference to the canonical
    /// field themselves.
    fn setup_indirectcalltargets(
        &mut self,
        targets: Vec<std::sync::Arc<majit_metainterp::jitcode::JitCode>>,
    ) {
        self.canonical.setup_indirectcalltargets(targets);
    }

    /// pyjitpl.py:2326-2343 `bytecode_for_address(fnaddress)`.
    /// Thin delegate to the canonical port.
    fn bytecode_for_address(
        &mut self,
        fnaddress: usize,
    ) -> Option<std::sync::Arc<majit_metainterp::jitcode::JitCode>> {
        self.canonical.bytecode_for_address(fnaddress)
    }
}

impl MetaInterpStaticData {
    /// pyjitpl.py:2227-2243 `MetaInterpStaticData.setup_insns(self, insns)`:
    /// copy opcode numbers for the well-known bytecodes out of the
    /// assembler's `insns` dict in the same order as upstream.
    fn setup_insns(&mut self, insns: &majit_ir::VecAssoc<String, u8>) {
        self.op_live = insns.get("live/").copied().unwrap_or(u8::MAX);
        self.op_goto = insns.get("goto/L").copied().unwrap_or(u8::MAX);
        self.op_catch_exception = insns.get("catch_exception/L").copied().unwrap_or(u8::MAX);
        self.op_rvmprof_code = insns.get("rvmprof_code/ii").copied().unwrap_or(u8::MAX);
        self.op_int_return = insns.get("int_return/i").copied().unwrap_or(u8::MAX);
        self.op_ref_return = insns.get("ref_return/r").copied().unwrap_or(u8::MAX);
        self.op_float_return = insns.get("float_return/f").copied().unwrap_or(u8::MAX);
        self.op_void_return = insns.get("void_return/").copied().unwrap_or(u8::MAX);
    }

    /// pyjitpl.py:2255-2264 `finish_setup`: wire the assembler's opcode table
    /// into this staticdata object and snapshot the current `all_liveness`.
    fn finish_setup_if_needed(
        &mut self,
        insns: &majit_ir::VecAssoc<String, u8>,
        all_liveness: Vec<u8>,
    ) {
        let was_done = self.finish_setup_done;
        self.setup_insns(insns);
        self.liveness_info = std::sync::Arc::<[u8]>::from(all_liveness.into_boxed_slice());
        self.finish_setup_done = true;
        // pyjitpl.py:2287-2290 `finish_setup_descrs`: PyPy invokes this
        // immediately after `finish_setup(codewriter)` from
        // `warmspot.py:289`. Pyre's `ensure_finish_setup` is the lazy
        // first-trace gate, so the descr bitstring compaction runs
        // exactly once on the very first JIT entry — the
        // `finish_setup_done` flag above this block keeps subsequent
        // calls idempotent. Bitstring writes land directly on each
        // descr's `ei_index` slot via `effectinfo::compute_bitstrings`
        // (`effectinfo.py:526 descr.ei_index = …`); no process-global
        // side table.
        if !was_done {
            self.canonical.finish_setup_descrs();
        }
    }

    /// pyjitpl.py:2264 `self.liveness_info = "".join(asm.all_liveness)` —
    /// refreshes the staticdata mirror after each writer-side append.
    pub(crate) fn set_liveness_info(&mut self, bytes: Vec<u8>) {
        self.liveness_info = std::sync::Arc::<[u8]>::from(bytes.into_boxed_slice());
    }
}

/// Crate-local helper so sibling modules can refresh
/// `MetaInterpStaticData.liveness_info` without leaking the private
/// type through public API signatures.
pub(crate) fn publish_liveness_info(bytes: Vec<u8>) {
    METAINTERP_SD.with(|r| r.borrow_mut().set_liveness_info(bytes));
}

impl MetaInterpStaticData {
    #[inline]
    fn canonical_code_key_opt(code: *const ()) -> Option<usize> {
        if code.is_null() {
            return None;
        }
        let raw = unsafe { pyre_interpreter::w_code_get_ptr(code as pyre_object::PyObjectRef) };
        if raw.is_null() {
            None
        } else {
            Some(raw as usize)
        }
    }

    /// codewriter.py:67-68 parity — stamp the SD-local `idx` onto the
    /// shared `Arc<majit_metainterp::jitcode::JitCode>` carried by the
    /// payload.  Upstream RPython unconditionally writes
    /// `jitcode.index = index`; a mismatch between an existing
    /// canonical stamp and the SD-local slot is a parity violation, so
    /// rely on `JitCode::set_index`'s own assert (same value: OK,
    /// different value: panic) instead of silently skipping.
    fn stamp_payload_index(idx: i32, payload: &std::sync::Arc<crate::PyJitCode>) {
        payload.jitcode.set_index(idx as usize);
    }

    /// warmspot.py:281-282:
    ///
    /// ```python
    /// jitcodes = self.codewriter.make_jitcodes(verbose=verbose)
    /// self.metainterp_sd.jitcodes = jitcodes
    /// ```
    ///
    /// RPython runs `make_jitcodes()` once before tracing, so existing
    /// frame pointers and jitcode indices never move. Pyre still reaches
    /// this boundary from lazy portal-entry setup, so merge new payloads
    /// into the existing trace-side list instead of replacing the list.
    /// This preserves RPython's stable `metainterp_sd.jitcodes` invariant
    /// for already-captured resume data.
    fn set_jitcodes_from_make_result(&mut self, payloads: Vec<std::sync::Arc<crate::PyJitCode>>) {
        for payload in payloads {
            assert!(
                !payload.w_code.is_null(),
                "make_jitcodes returned a JitCode without W_CodeObject identity"
            );
            assert!(
                !payload.is_skeleton(),
                "make_jitcodes returned an unpopulated JitCode skeleton"
            );
            let raw_key = Self::canonical_code_key_opt(payload.w_code)
                .expect("make_jitcodes returned a non-canonical W_CodeObject");
            let existing_pos = self
                .jitcodes
                .iter()
                .position(|jitcode| unsafe { jitcode.raw_code() as usize } == raw_key);
            if let Some(pos) = existing_pos {
                let index = self.jitcodes[pos].index;
                Self::stamp_payload_index(index, &payload);
                self.jitcodes[pos].code = payload.w_code;
                self.jitcodes[pos].payload = payload;
            } else {
                let index = self.jitcodes.len() as i32;
                Self::stamp_payload_index(index, &payload);
                self.jitcodes.push(Box::new(JitCode {
                    code: payload.w_code,
                    index,
                    payload,
                }));
            }
        }
    }

    fn installed_jitcode_pos_for_raw_key(&self, raw_key: usize) -> Option<usize> {
        self.jitcodes
            .iter()
            .position(|jitcode| unsafe { jitcode.raw_code() as usize } == raw_key)
    }

    fn portal_bridge_payload_for(
        code: *const (),
        raw_key: usize,
    ) -> std::sync::Arc<crate::PyJitCode> {
        let raw_code = raw_key as *const CodeObject;
        let payload = crate::canonical_bridge::install_portal_for(raw_code, code);
        assert!(
            payload.is_portal_bridge(),
            "portal bridge install must produce a portal-bridge PyJitCode"
        );
        payload
    }

    /// Pyre adapter for the PyPy portal model: every user CodeObject runs
    /// through the canonical portal JitCode, but trace/resume readers still
    /// need per-CodeObject frame-shape metadata (`stack_base`,
    /// `depth_at_py_pc`). Build a portal-bridge wrapper that shares the
    /// portal bytecode and carries only that metadata; do not invoke the
    /// codewriter or create a per-CodeObject drained JitCode.
    fn portal_bridge_jitcode_for(&mut self, code: *const ()) -> *const JitCode {
        let raw_key = Self::canonical_code_key_opt(code).unwrap_or_else(|| {
            panic!(
                "portal bridge requested for invalid W_CodeObject {:p}",
                code
            )
        });
        if let Some(pos) = self.installed_jitcode_pos_for_raw_key(raw_key) {
            return &*self.jitcodes[pos] as *const JitCode;
        }
        let payload = Self::portal_bridge_payload_for(code, raw_key);
        let index = self.jitcodes.len() as i32;
        Self::stamp_payload_index(index, &payload);
        let jitcode = Box::new(JitCode {
            code,
            index,
            payload,
        });
        let ptr = &*jitcode as *const JitCode;
        self.jitcodes.push(jitcode);
        ptr
    }

    /// codewriter.py:67-68 / call.py:155-172 adapter: install or return the
    /// trace-side wrapper for `code`. When `supplied` is present it is the
    /// populated PyJitCode Arc from CodeWriter's pending-graph drain; otherwise
    /// this creates the same empty skeleton shape as CallControl.get_jitcode()
    /// before the drain fills it.
    fn jitcode_for(
        &mut self,
        code: *const (),
        supplied: Option<std::sync::Arc<crate::PyJitCode>>,
    ) -> *const JitCode {
        let raw_key = Self::canonical_code_key_opt(code).unwrap_or(0);
        if let Some(pos) = self.installed_jitcode_pos_for_raw_key(raw_key) {
            if let Some(payload) = supplied {
                let index = self.jitcodes[pos].index;
                Self::stamp_payload_index(index, &payload);
                self.jitcodes[pos].code = payload.w_code;
                self.jitcodes[pos].payload = payload;
            }
            return &*self.jitcodes[pos] as *const JitCode;
        }

        let payload = supplied.unwrap_or_else(|| {
            let raw_code = if raw_key == 0 {
                std::ptr::null()
            } else {
                raw_key as *const CodeObject
            };
            std::sync::Arc::new(crate::PyJitCode::skeleton(raw_code, code, None))
        });
        let index = self.jitcodes.len() as i32;
        Self::stamp_payload_index(index, &payload);
        let jitcode = Box::new(JitCode {
            code,
            index,
            payload,
        });
        let ptr = &*jitcode as *const JitCode;
        self.jitcodes.push(jitcode);
        ptr
    }

    /// Return the installed SD entry for a `W_CodeObject`.
    /// RPython's runtime lookup never compiles and never creates a
    /// skeleton here: every entry must already have arrived through
    /// `make_jitcodes()` and `warmspot.py:282`.
    fn compiled_jitcode_lookup(&self, code: *const ()) -> Option<*const JitCode> {
        let key = Self::canonical_code_key_opt(code)?;
        self.jitcodes
            .iter()
            .find(|jitcode| unsafe { jitcode.raw_code() as usize } == key)
            .filter(|jitcode| !jitcode.payload.is_skeleton())
            .map(|jitcode| &**jitcode as *const JitCode)
    }
}

/// RPython assembler.py:234-248 `Assembler._encode_liveness` parity:
/// intern one `[live_i, live_r, live_f]` triple in the assembler's
/// `all_liveness` buffer and return its 2-byte offset.
///
/// Writes land on `AssemblerState` (the writer side); `MetaInterpStaticData`
/// receives a fresh snapshot via the final `METAINTERP_SD.liveness_info`
/// assignment — matching `pyjitpl.py:2264`'s
/// `self.liveness_info = "".join(asm.all_liveness)` after each append.
pub fn intern_liveness(live_i: &[u8], live_r: &[u8], live_f: &[u8]) -> Option<u16> {
    use crate::assembler::ASSEMBLER_STATE;
    use majit_translate::liveness::encode_liveness;

    ensure_finish_setup();

    let snapshot = ASSEMBLER_STATE.with(|r| -> Option<(u16, Vec<u8>)> {
        let mut asm = r.borrow_mut();
        // assembler.py:149 `self.num_liveness_ops += 1` — counted once per
        // `-live-` instruction in `write_insn`, before the dedup lookup
        // inside `_encode_liveness`. The counter measures write-insn call
        // frequency, not unique-entry count.
        asm.num_liveness_ops += 1;
        let key = (live_i.to_vec(), live_r.to_vec(), live_f.to_vec());
        if let Some(&pos) = asm.all_liveness_positions.get(&key) {
            return Some((pos, asm.all_liveness.clone()));
        }
        let pos = asm.all_liveness_length;
        let encoded_i = encode_liveness(live_i);
        let encoded_r = encode_liveness(live_r);
        let encoded_f = encode_liveness(live_f);
        if live_i.len() > u8::MAX as usize
            || live_r.len() > u8::MAX as usize
            || live_f.len() > u8::MAX as usize
            || pos > u16::MAX as usize
        {
            return None;
        }
        let pos_u16 = pos as u16;
        asm.all_liveness_positions.insert(key, pos_u16);
        asm.all_liveness.push(live_i.len() as u8);
        asm.all_liveness.push(live_r.len() as u8);
        asm.all_liveness.push(live_f.len() as u8);
        asm.all_liveness.extend(encoded_i);
        asm.all_liveness.extend(encoded_r);
        asm.all_liveness.extend(encoded_f);
        asm.all_liveness_length = asm.all_liveness.len();
        Some((pos_u16, asm.all_liveness.clone()))
    })?;

    let (pos, all_liveness) = snapshot;
    METAINTERP_SD.with(|r| {
        r.borrow_mut().liveness_info = std::sync::Arc::<[u8]>::from(all_liveness.into_boxed_slice())
    });
    Some(pos)
}

/// RPython resume.py:1022 parity: read-only snapshot of
/// `metainterp_sd.liveness_info` for the ResumeDataDirectReader.
///
/// Returns a shared `Arc<[u8]>` so the caller observes the same packed
/// buffer the assembler published via `intern_liveness` /
/// `publish_liveness_info` without copying its bytes (the previous
/// implementation cloned the underlying `Vec<u8>` per BH entry — RPython
/// upstream simply reads `metainterp_sd.liveness_info` straight off the
/// staticdata object).
pub fn liveness_info_snapshot() -> std::sync::Arc<[u8]> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| std::sync::Arc::clone(&r.borrow().liveness_info))
}

/// pyjitpl.py:2236 parity: expose the staticdata `live/` opcode for callers
/// that need to decode inline liveness offsets from a JitCode.
pub fn op_live() -> u8 {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| r.borrow().op_live)
}

/// blackhole.py:72-74 cached control opcodes consumed by pyre's
/// blackhole-resume adapters. Returns the same `insns.get(..., -1)`
/// values RPython would have stored on `BlackholeInterpBuilder`.
pub fn blackhole_control_opcodes() -> (i32, i32, i32) {
    use crate::assembler::ASSEMBLER_STATE;

    let needs_refresh = METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let missing_live = sd.op_live == u8::MAX;
        let missing_catch_exception = sd.op_catch_exception == u8::MAX;
        let missing_rvmprof_code = sd.op_rvmprof_code == u8::MAX;
        if !sd.finish_setup_done {
            return true;
        }
        if !(missing_live || missing_catch_exception || missing_rvmprof_code) {
            return false;
        }
        ASSEMBLER_STATE.with(|a| {
            let asm = a.borrow();
            (missing_live && asm.insns.contains_key("live/"))
                || (missing_catch_exception && asm.insns.contains_key("catch_exception/L"))
                || (missing_rvmprof_code && asm.insns.contains_key("rvmprof_code/ii"))
        })
    });
    if needs_refresh {
        ensure_finish_setup();
    }
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let decode = |opcode: u8| -> i32 { if opcode == u8::MAX { -1 } else { opcode as i32 } };
        (
            decode(sd.op_live),
            decode(sd.op_catch_exception),
            decode(sd.op_rvmprof_code),
        )
    })
}

/// pyjitpl.py:2248-2249 module-level entry point for
/// `MetaInterpStaticData::setup_indirectcalltargets`.
///
/// RPython sets this at `pyjitpl.py:2262` during
/// `finish_setup(codewriter, optimizer)` by piping
/// `codewriter.assembler.indirectcalltargets` straight through.  pyre's
/// codewriter driver calls this after every `Assembler::assemble`
/// session so the staticdata reflects the assembler's latest target
/// set.  Matching shape: `Vec<Arc<JitCode>>` comes from
/// `Assembler::indirectcalltargets_vec` in `pyre-jit`.
pub fn setup_indirectcalltargets(targets: Vec<std::sync::Arc<majit_metainterp::jitcode::JitCode>>) {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| r.borrow_mut().setup_indirectcalltargets(targets));
}

/// pyjitpl.py:2326-2343 module-level entry point for
/// `MetaInterpStaticData::bytecode_for_address`.
///
/// RPython calls this from `MIFrame.do_residual_or_indirect_call`
/// (`pyjitpl.py:2174-2186`) to check whether a `funcbox.getaddr()`
/// Const corresponds to a known indirect-call target.  pyre's tracer
/// consumer routes through here.
pub fn bytecode_for_address(
    fnaddress: usize,
) -> Option<std::sync::Arc<majit_metainterp::jitcode::JitCode>> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| r.borrow_mut().bytecode_for_address(fnaddress))
}

use std::cell::RefCell;

thread_local! {
    /// warmspot.py:282: MetaInterp.staticdata (per-thread for no-GIL).
    pub(crate) static METAINTERP_SD: RefCell<MetaInterpStaticData> =
        RefCell::new(MetaInterpStaticData::new());
}

/// pyjitpl.py:2255 `MetaInterpStaticData.finish_setup` parity entry point.
///
/// RPython runs `finish_setup` once per `MetaInterpStaticData` object. Pyre's
/// `METAINTERP_SD` is thread-local, so the setup guard also lives on
/// `MetaInterpStaticData`; only the unrelated callback registration below is
/// process-global.
fn ensure_finish_setup() {
    use crate::assembler::ASSEMBLER_STATE;
    use std::sync::Once;
    static FRAME_VALUE_COUNT_INIT: Once = Once::new();
    FRAME_VALUE_COUNT_INIT.call_once(|| {
        majit_ir::resumedata::set_frame_value_count_fn(frame_value_count_at);
    });
    let (insns, all_liveness) = ASSEMBLER_STATE.with(|a| {
        let asm = a.borrow();
        (asm.insns.clone(), asm.all_liveness.clone())
    });
    METAINTERP_SD.with(|r| {
        r.borrow_mut().finish_setup_if_needed(&insns, all_liveness);
    });
}

/// Pyre adapter for portal-bridge experiments. Production trace frames use
/// `jitcode_for()` below, which follows CallControl.get_jitcode + drain.
#[allow(dead_code)]
pub(crate) fn portal_bridge_jitcode_for(code: *const ()) -> *const JitCode {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| r.borrow_mut().portal_bridge_jitcode_for(code))
}

/// pyjitpl.py:74: frame.jitcode — resolve the JitCode for the frame's code
/// object through the writer-side CallControl.get_jitcode path.
pub(crate) fn jitcode_for(code: *const ()) -> *const JitCode {
    ensure_finish_setup();
    if let Some(existing) = METAINTERP_SD.with(|r| r.borrow().compiled_jitcode_lookup(code)) {
        return existing;
    }
    if code.is_null() {
        return METAINTERP_SD.with(|r| r.borrow_mut().jitcode_for(code, None));
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if !raw_code.is_null() {
        let callbacks = crate::callbacks::try_get().unwrap_or_else(|| {
            panic!(
                "CallJitCallbacks not initialized while resolving JitCode for W_CodeObject {:p}",
                code
            )
        });
        (callbacks.ensure_majit_jitcode)(raw_code, code);
        if let Some(existing) = METAINTERP_SD.with(|r| r.borrow().compiled_jitcode_lookup(code)) {
            return existing;
        }
        panic!(
            "ensure_majit_jitcode did not install a populated JitCode for W_CodeObject {:p}",
            code
        );
    }
    METAINTERP_SD.with(|r| r.borrow_mut().jitcode_for(code, None))
}

/// Install one CodeWriter-owned PyJitCode payload into trace-side
/// MetaInterpStaticData. Used by the lazy CallControl.get_jitcode drain path
/// to publish the same Arc that CallControl.jitcodes stores.
pub fn install_jitcode_for(
    code: *const (),
    payload: std::sync::Arc<crate::PyJitCode>,
) -> *const () {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| r.borrow_mut().jitcode_for(code, Some(payload)) as *const ())
}

/// `framework.py root_walker.walk_roots` parity for the persistent
/// `MetaInterpStaticData.jitcodes` list (warmspot.py:282
/// `self.metainterp_sd.jitcodes = jitcodes`).  Each `JitCode.code`
/// (state.rs:40) is a W_CodeObject pointer.
///
/// Intentionally not yet registered as a root walker: W_CodeObject is
/// host-allocated via `Box::into_raw` (pycode.rs), not in the GC heap, so
/// the moving collector never sweeps or relocates it and there is nothing
/// to root.  When code objects become GC-managed this gets wired into
/// `majit_gc::register_extra_root_walker`, at which point `JitCode.code`
/// becomes a `GcRef` slot and `visit` lets a moving collector rewrite it
/// in place.
#[allow(dead_code)]
pub fn walk_jitcode_code_roots(mut visit: impl FnMut(&mut *const ())) {
    METAINTERP_SD.with(|r| {
        for jc in r.borrow_mut().jitcodes.iter_mut() {
            visit(&mut jc.code);
        }
    });
}

/// Install the complete `CodeWriter.make_jitcodes()` result into
/// `MetaInterpStaticData.jitcodes`. Setup-time bulk publish only.
///
/// RPython warmspot.py:281-282 stores the list returned by
/// `codewriter.make_jitcodes()` directly on `metainterp_sd.jitcodes`.
/// This function is the pyre-side analog and is invoked exclusively
/// from `register_portal_jitdriver` after the JitDriver-rooted
/// `make_jitcodes()` drain. Pyre may call that boundary more than once
/// while it still has lazy portal-entry setup, so the implementation
/// merges payloads without moving existing SD entries.
pub fn install_jitcodes(jitcodes: Vec<std::sync::Arc<crate::PyJitCode>>) {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        r.borrow_mut().set_jitcodes_from_make_result(jitcodes);
    });
}

/// Return the SD-local `jitcode.index` for this `W_CodeObject`, ensuring
/// the entry through the same `jitcode_for()` / CallControl path used by
/// trace frame setup.
pub fn ensure_jitcode_index(code: *const ()) -> Option<i32> {
    if code.is_null() {
        return None;
    }
    let jitcode = jitcode_for(code);
    Some(unsafe { (*jitcode).index })
}

/// Return the `JitCode*` for this `W_CodeObject` as an opaque pointer,
/// ensuring the entry through the same path as `ensure_jitcode_index`.
#[doc(hidden)]
pub fn ensure_jitcode_ptr(code: *const ()) -> Option<*const ()> {
    if code.is_null() {
        return None;
    }
    Some(jitcode_for(code) as *const ())
}

#[doc(hidden)]
pub fn frame_locals_cells_stack_array_ref(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    frame_locals_cells_stack_array(ctx, frame)
}

/// Read-only JitCode lookup by CodeObject-wrapper pointer.
///
/// Blackhole/resume paths must not invoke compilation. Returns null
/// if the code was not installed by the setup-time
/// `CodeWriter.make_jitcodes()` result.
///
/// Uses the same canonical raw-code comparison as
/// `MetaInterpStaticData::compiled_jitcode_lookup`; callers still
/// pass the wrapper `w_code`, and the lookup normalizes it internally.
pub(crate) fn jitcode_lookup(code: *const ()) -> *const JitCode {
    ensure_finish_setup();
    METAINTERP_SD
        .with(|r| r.borrow().compiled_jitcode_lookup(code))
        .unwrap_or(std::ptr::null())
}

/// warmspot.py:282 metainterp_sd.jitcodes[jitcode_index]:
/// Resolve jitcode_index (sequential int from snapshot numbering)
/// to the corresponding CodeObject pointer.
pub fn code_for_jitcode_index(jitcode_index: i32) -> Option<*const ()> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let idx = jitcode_index as usize;
        sd.jitcodes.get(idx).map(|jc| jc.code)
    })
}

/// warmspot.py:282 `metainterp_sd.jitcodes[jitcode_index]` helper:
/// resolve the indexed runtime entry to its canonical raw `CodeObject*`.
///
/// Unlike [`code_for_jitcode_index`], this strips the wrapper round-trip
/// up front for callers that only need the graph identity to re-enter
/// `CallControl.jitcodes`.
pub fn raw_code_for_jitcode_index(jitcode_index: i32) -> Option<*const CodeObject> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let idx = jitcode_index as usize;
        sd.jitcodes.get(idx).map(|jc| unsafe { jc.raw_code() })
    })
}

/// Resolve `MetaInterpStaticData.jitcodes[jitcode_index]` to the same
/// PyJitCode payload the trace-side frame used. This keeps blackhole /
/// resume consumers on the RPython single-store path instead of
/// re-looking-up through pyre-jit's CodeWriter side cache, which does not
/// own portal-bridge wrappers.
pub fn pyjitcode_for_jitcode_index(jitcode_index: i32) -> Option<std::sync::Arc<crate::PyJitCode>> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let idx = jitcode_index as usize;
        sd.jitcodes
            .get(idx)
            .map(|jc| std::sync::Arc::clone(&jc.payload))
    })
}

/// Resolve by W_CodeObject wrapper through the trace-side
/// MetaInterpStaticData store. Used by blackhole paths that must see
/// portal-bridge wrappers as well as CodeWriter-drained entries.
pub fn pyjitcode_for_code(code: *const ()) -> Option<std::sync::Arc<crate::PyJitCode>> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let ptr = sd.compiled_jitcode_lookup(code)?;
        Some(std::sync::Arc::clone(unsafe { &(*ptr).payload }))
    })
}

/// `resume.py:1049` `consume_one_section` → `enumerate_vars` parity:
/// return the number of tagged values encoded for a frame at
/// (jitcode_index, pc).
///
/// Upstream `pyjitpl.py:199` / `jitcode.py:82-93`: decode the `-live-`
/// offset from the jitcode byte stream at `jitcode.get_live_vars_info(
/// pc, op_live)`, then read the three-byte `[len_i][len_r][len_f]`
/// header in `all_liveness`. Total live value count = `len_i +
/// len_r + len_f`.
///
/// Fallback: when the jitcode is still a skeleton payload (pc_map
/// empty) or has no backing CodeObject, decode via the pyre-jit-trace
/// `LiveVars` analysis over the Python bytecode. This path is used
/// for inlined callee frames whose majit_jitcode has not been built
/// at trace time.
pub fn frame_value_count_at(jitcode_index: i32, pc: i32) -> usize {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let idx = jitcode_index as usize;
        let jc = match sd.jitcodes.get(idx) {
            Some(jc) => jc,
            None => return 0,
        };
        let payload = &jc.payload;
        let resolved_jit_pc: Option<usize> = payload.resume_jitcode_pc_for(pc as usize);
        if let Some(jit_pc) = resolved_jit_pc {
            let off = payload.jitcode.get_live_vars_info(jit_pc, sd.op_live);
            let all_liveness: &[u8] = &sd.liveness_info;
            if off + 2 < all_liveness.len() {
                let length_i = all_liveness[off] as usize;
                let length_r = all_liveness[off + 1] as usize;
                let length_f = all_liveness[off + 2] as usize;
                return length_i + length_r + length_f;
            }
        }
        // G.4.3 portal-bridge decoder count: read the per-PC depth from
        // the metadata `LiveVars` derivation that
        // `canonical_bridge::install_portal_for` populates (G.4.2).
        // The encoder side (`trace_opcode.rs::get_list_of_active_boxes`)
        // emits exactly `stack_base + depth_at_py_pc[pc]` Ref-typed
        // boxes for portal-bridge frames; this count must agree so the
        // rd_numb cursor advances symmetrically through
        // `_prepare_next_section`.
        //
        // RPython parity: upstream `pyjitpl.py:177
        // get_list_of_active_boxes` and `resume.py:1017-1026
        // _prepare_next_section` share a single packed-liveness
        // definition per (jitcode, pc).  Pyre's portal-bridge wrapper
        // routes both sides through `metadata.depth_at_py_pc` instead
        // of canonical's `all_liveness` (which encodes the dispatch
        // loop's registers, not user PyFrame state) so the same
        // symmetry holds.  All portal-bridge live values are Ref-typed
        // (PyObjectRef stack), so `length_i = length_f = 0` and the
        // total count is a single `length_r`.
        //
        // (Pre-G.4.3a-fix used `payload.nlocals_from_code()` which
        // dropped `ncells`, breaking encoder/decoder count agreement
        // for closure-bearing functions.  `metadata.stack_base` matches
        // upstream's `pyframe.py:111 valuestackdepth = co_nlocals +
        // ncellvars + nfreevars`.)
        if payload.is_portal_bridge() {
            let depth = payload
                .metadata
                .depth_at_py_pc
                .get(pc as usize)
                .copied()
                .unwrap_or(0) as usize;
            return payload.metadata.stack_base + depth;
        }
        // `CallControl.get_jitcode` drain fills pc_map + liveness
        // before any guard capture (pyjitpl.py:199 parity). The
        // out-of-range-pc source is eliminated by threading
        // jitcode_index through `Snapshot::single_frame`, and the
        // guard/resume tests run on the real compile/register path in
        // `pyre-jit`. Unconditional panic — any hit is a bug.
        panic!(
            "frame_value_count_at: fallback hit for jitcode_index={} pc={} \
             (pc_map.len={}, all_liveness.len={}). Phase X-0/X-1 removed \
             all known triggers — further hits are bugs.",
            jitcode_index,
            pc,
            payload.metadata.pc_map.len(),
            sd.liveness_info.len(),
        );
    })
}

/// virtualizable.py:86-98 `read_boxes` parity: assemble the
/// `virtualizable_boxes` layout the tracing-time vable mirror expects
/// and hand it to `TraceCtx::init_virtualizable_boxes`. Used by both
/// the root portal seed (`initialize_sym`) and the bridge entry rebuild
/// (`setup_bridge_sym`). Matches RPython `virtualizable.py:139
/// load_list_of_boxes`, producing `[static_fields..., array_items...,
/// vable_box]` (the trailing `vable_box` is appended by
/// `init_virtualizable_boxes`).
///
/// * `scalar_oprefs` — the NUM_VABLE_SCALARS static field OpRefs in
///   declaration order (last_instr, pycode, valuestackdepth, debugdata,
///   lastblock, w_globals). Excludes both the frame-identity slot and
///   any non-vable extra reds (e.g. `ec`); virtualizable_boxes only
///   carries the vable static fields plus array items.
/// * `array_items` — pre-resolved OpRefs for the heap-side
///   `locals_cells_stack_w` array. Entries past `array_len` are ignored;
///   short lists are padded with a shared const-NULL Ref so the vable
///   mirror covers every interpreter-visible slot (virtualizable.py:109
///   `assert len(boxes) == i + 1`).
/// * `array_len` — the runtime PyFrame's
///   `locals_cells_stack_w.len()`; `init_virtualizable_boxes` stores this
///   as the sole entry in `virtualizable_array_lengths`.
pub(crate) fn seed_virtualizable_boxes(
    ctx: &mut TraceCtx,
    vable_ref: OpRef,
    vable_ref_value: majit_ir::Value,
    scalar_oprefs: &[OpRef],
    array_items: &[OpRef],
    array_len: usize,
    input_values: &[majit_ir::Value],
    heap_ptr: *const u8,
) {
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    let expected_scalars = crate::virtualizable_gen::NUM_VABLE_SCALARS;
    assert_eq!(
        scalar_oprefs.len(),
        expected_scalars,
        "seed_virtualizable_boxes: scalar_oprefs.len() must equal NUM_VABLE_SCALARS",
    );
    let mut input_oprefs: Vec<OpRef> = Vec::with_capacity(expected_scalars + array_len);
    input_oprefs.extend_from_slice(scalar_oprefs);
    let taken = array_items.len().min(array_len);
    input_oprefs.extend_from_slice(&array_items[..taken]);
    if taken < array_len {
        let null_ref = ctx.const_ref(pyre_object::PY_NULL as i64);
        for _ in taken..array_len {
            input_oprefs.push(null_ref);
        }
    }
    let array_lengths = vec![array_len];
    // virtualizable.py:139 load_list_of_boxes parity: the concrete half of
    // virtualizable_boxes is sourced from the caller (heap read for portal
    // entry / resume-data stream for bridge entry), never synthesized here.
    // Callers pass an empty slice to disable the concrete shadow
    // (unit-test / init-before-run path).
    ctx.init_virtualizable_boxes(
        &info,
        vable_ref,
        vable_ref_value,
        &input_oprefs,
        input_values,
        &array_lengths,
    );
    // pyjitpl.py:3446 synchronize_virtualizable parity: cache the live heap
    // pointer on TraceCtx so subsequent vable setfield / setarrayitem calls
    // can mirror their shadow updates into the live virtualizable.  Pass
    // a null pointer to disable (unit-test / init-before-run path).
    ctx.set_virtualizable_heap_ptr(heap_ptr);
}

/// Decode a raw `vinfo.read_all_boxes` entry into the typed Value a
/// virtualizable shadow slot expects.  Local helper that mirrors
/// `majit-metainterp::pyjitpl::heap_value_for` so both seed sites (root
/// portal in `initialize_virtualizable` and bridge entry here) use the
/// same raw-bit → typed-Value rule.
fn value_for_slot(ty: Type, bits: i64) -> majit_ir::Value {
    match ty {
        Type::Int => majit_ir::Value::Int(bits),
        Type::Float => majit_ir::Value::Float(f64::from_bits(bits as u64)),
        Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(bits as usize)),
        Type::Void => majit_ir::Value::Void,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameLivenessRegIndices {
    pub int: Vec<u32>,
    pub ref_: Vec<u32>,
    pub float: Vec<u32>,
}

impl FrameLivenessRegIndices {
    pub fn total_len(&self) -> usize {
        self.int.len() + self.ref_.len() + self.float.len()
    }

    pub fn flattened(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.total_len());
        out.extend_from_slice(&self.int);
        out.extend_from_slice(&self.ref_);
        out.extend_from_slice(&self.float);
        out
    }
}

/// resume.py:1054 consume_boxes(info, boxes_i, boxes_r, boxes_f) parity:
/// return live register indices split by the three liveness banks.
/// RPython writes decoded values through `_callback_i/_callback_r/_callback_f`
/// into `registers_i/r/f[index]`; keeping the banks separate prevents Ref-only
/// semantic-slot remapping from swallowing Int/Float slots.
pub fn frame_liveness_reg_indices_by_bank_at(
    jitcode_index: i32,
    pc: i32,
) -> FrameLivenessRegIndices {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let idx = jitcode_index as usize;
        let Some(jc) = sd.jitcodes.get(idx) else {
            return FrameLivenessRegIndices::default();
        };
        let payload = &jc.payload;
        // G.4.4 portal-bridge fallback (symmetric with frame_value_count_at:803
        // and trace_opcode.rs:642 get_list_of_active_boxes encoder):
        // portal-bridge installs leave `metadata.pc_map` empty and emit a
        // positional Ref-typed box list of length
        // `stack_base + depth_at_py_pc[pc]` covering locals + cells + the
        // live operand stack tail. Color map is identity (no regalloc for
        // portal-bridge), so `reg_idx == slot index` for each box.
        // Without this fallback, `setup_bridge_sym`'s `reg_indices.len() ==
        // frame.values.len()` assert fires on every guard exit out of a
        // portal-bridge trace.
        if payload.is_portal_bridge() {
            let depth = payload
                .metadata
                .depth_at_py_pc
                .get(pc as usize)
                .copied()
                .unwrap_or(0) as usize;
            let total = payload.metadata.stack_base + depth;
            return FrameLivenessRegIndices {
                int: Vec::new(),
                ref_: (0..total as u32).collect(),
                float: Vec::new(),
            };
        }
        let resolved_jit_pc: Option<usize> = payload.resume_jitcode_pc_for(pc as usize);
        let Some(jit_pc) = resolved_jit_pc else {
            return FrameLivenessRegIndices::default();
        };
        let off = payload.jitcode.get_live_vars_info(jit_pc, sd.op_live);
        let all_liveness = &sd.liveness_info;
        if off + 2 >= all_liveness.len() {
            return FrameLivenessRegIndices::default();
        }
        let length_i = all_liveness[off] as u32;
        let length_r = all_liveness[off + 1] as u32;
        let length_f = all_liveness[off + 2] as u32;
        let mut cursor = off + 3;
        use majit_translate::liveness::LivenessIterator;

        fn read_bank(cursor: &mut usize, length: u32, all_liveness: &[u8]) -> Vec<u32> {
            if length == 0 {
                return Vec::new();
            }
            let mut it = LivenessIterator::new(*cursor, length, all_liveness);
            let mut out = Vec::with_capacity(length as usize);
            while let Some(reg_idx) = it.next() {
                out.push(reg_idx);
            }
            *cursor = it.offset;
            out
        }

        let int = read_bank(&mut cursor, length_i, all_liveness);
        let ref_ = read_bank(&mut cursor, length_r, all_liveness);
        let float = read_bank(&mut cursor, length_f, all_liveness);
        FrameLivenessRegIndices { int, ref_, float }
    })
}

pub fn frame_liveness_reg_indices_at(jitcode_index: i32, pc: i32) -> Vec<u32> {
    frame_liveness_reg_indices_by_bank_at(jitcode_index, pc).flattened()
}

/// Test-only fixture seeder for `_with_compiled_trace_jitcode` unit tests.
/// Gated behind `#[cfg(any(test, feature = "test-support"))]` so production
/// builds cannot reach the synthetic register-bank fill — RPython has no
/// counterpart and live holes get filled with `0xfeed` / `0` dummy consts
/// that would corrupt a real trace if invoked outside fixtures.
///
/// `pyjitpl.py:218-225 get_list_of_active_boxes` reads each live register
/// directly from its kind-specific bank (`registers_i[reg]` /
/// `registers_r[reg]` / `registers_f[reg]`); commit
/// `3fdb617f5d1` removed pyre's prior `registers_r_semantic` fallback to
/// match that contract. Production tracers fill the kind banks via
/// `bcd_op` dispatch as the trace is recorded, but unit-test fixtures
/// build a `PyreSym` directly through `from_test_state` and never run the
/// dispatch loop — without seeding, `current_fail_args` returns a vector
/// full of `OpRef::NONE` holes.
///
/// This helper mirrors what production code does:
///   1. `setup_kind_register_banks` to size the three banks +
///      copy-constants per `pyjitpl.py:97-119`.
///   2. Place each `(stack_depth, opref)` at the semantic frame-mirror
///      slot `registers_r[nlocals + depth]`. Production guard capture
///      then materializes the color-indexed Ref bank from the same
///      semantic/vable source before reading liveness.
///   3. Fill any still-`OpRef::NONE` Int/Float bank slot listed in
///      `live_pc`'s bank-split liveness with a typed dummy constant.
///      Ref slots are provided by the caller-provided stack slot map above.
#[cfg(any(test, feature = "test-support"))]
pub fn seed_compiled_trace_jitcode_test_state(
    sym: &mut PyreSym,
    ctx: &mut TraceCtx,
    jitcode_index: i32,
    live_pc: i32,
    stack_slots: &[(usize, OpRef)],
) {
    if !sym.jitcode.is_null() {
        sym.setup_kind_register_banks(ctx);
    }

    for &(depth, opref) in stack_slots {
        // Match production stack writes: keep `registers_r` as a
        // semantic frame mirror. The encoder builds the color-indexed
        // bank from this mirror/vable view at snapshot time.
        let reg_idx = crate::trace_opcode::stack_slot_reg_idx(sym, depth);
        if reg_idx >= sym.registers_r.len() {
            sym.registers_r.resize(reg_idx + 1, OpRef::NONE);
        }
        sym.registers_r[reg_idx] = opref;
    }

    let banks = frame_liveness_reg_indices_by_bank_at(jitcode_index, live_pc);
    for &reg in &banks.int {
        let r = reg as usize;
        if r >= sym.registers_i.len() {
            sym.registers_i.resize(r + 1, OpRef::NONE);
        }
        if sym.registers_i[r].is_none() {
            sym.registers_i[r] = ctx.const_int(0xfeed);
        }
    }
    for &reg in &banks.float {
        let r = reg as usize;
        if r >= sym.registers_f.len() {
            sym.registers_f.resize(r + 1, OpRef::NONE);
        }
        if sym.registers_f[r].is_none() {
            sym.registers_f[r] = ctx.const_float(0);
        }
    }
}

/// Return the post-regalloc Ref-bank color of each Python-semantic stack
/// slot for the registered jitcode at `jitcode_index`. Mirrors the
/// `metadata.stack_slot_color_map` Vec stored on `PyJitCode` (sized
/// `max_stackdepth`), and returns an empty Vec if the index is unknown.
///
/// Used by tests + tooling that need to translate "stack depth `d`" into
/// the post-rename register color the dispatcher would touch — the pinning removal
/// removed the `nlocals + d` identity, so direct slot arithmetic no
/// longer works.
pub fn stack_slot_color_map_at(jitcode_index: i32) -> Vec<u16> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        sd.jitcodes
            .get(jitcode_index as usize)
            .map(|jc| jc.payload.metadata.stack_slot_color_map.clone())
            .unwrap_or_default()
    })
}

/// Return the per-semantic-local Ref-bank color assigned by regalloc for
/// the registered jitcode at `jitcode_index`.  Forward map: local index
/// `i` → post-rename color holding that local's Variable at trace-recorder
/// snapshot time.  Empty vector when the jitcode hasn't been finalized.
///
/// Mirrors `stack_slot_color_map_at`'s contract for the local half of
/// `locals_cells_stack_w`; used by `_with_compiled_trace_jitcode` tests
/// that need to query "what register holds local N" without embedding a
/// specific reg index that would couple to walker vs canonical regalloc
/// strategies.
pub fn local_slot_color_map_at(jitcode_index: i32) -> Vec<u16> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        sd.jitcodes
            .get(jitcode_index as usize)
            .map(|jc| jc.payload.metadata.pyre_color_for_semantic_local.clone())
            .unwrap_or_default()
    })
}

/// Return the post-regalloc Ref-bank colors of the portal red args
/// (`pypy/module/pypyjit/interp_jit.py:67 reds = ['frame', 'ec']`) for
/// the registered jitcode at `jitcode_index`. Both `u16::MAX` for
/// portal-bridge installs that skip per-CodeObject regalloc.
///
/// Used by bridge resume to thread the ec OpRef from
/// `sym.registers_r[portal_ec_reg]` (populated by the liveness-driven
/// `consume_boxes` fill) back into `sym.execution_context`, mirroring
/// `resume.py:1077-1081 _callback_r` which writes the same slot in
/// RPython's BH register bank.
pub fn portal_red_regs_at(jitcode_index: i32) -> (u16, u16) {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        sd.jitcodes
            .get(jitcode_index as usize)
            .map(|jc| {
                (
                    jc.payload.metadata.portal_frame_reg,
                    jc.payload.metadata.portal_ec_reg,
                )
            })
            .unwrap_or((u16::MAX, u16::MAX))
    })
}

/// Map a post-regalloc Ref-bank color back to the semantic
/// `locals_cells_stack_w` slot it denotes at the current PC.
///
/// **Decoder-only**: the encoder
/// (`get_list_of_active_boxes`) reads `registers_r[color]`
/// directly and derives `semantic_idx` inline.  This function is
/// still used by `restore_guard_failure_values` (the decoder) which
/// needs the semantic slot index to call `set_local_at` /
/// `set_stack_at` on the concrete PyFrame.
///
/// After stack-slot pinning removal, stack slots are no longer forced to
/// occupy colors `nlocals + d`; the reverse lookup must consult
/// `metadata.stack_slot_color_map` first, bounded to the LIVE stack
/// prefix at the current PC. Only if no live stack slot owns the color
/// can the color fall back through the local slot -> color map.
pub(crate) fn semantic_ref_slot_for_reg_color(
    nlocals: usize,
    stack_only: usize,
    local_color_map: &[u16],
    stack_color_map: &[u16],
    live_local_indices: &[usize],
    reg: usize,
) -> Option<usize> {
    let live_len = stack_color_map.len().min(stack_only);
    if let Some(stack_idx) = stack_color_map[..live_len]
        .iter()
        .position(|&color| color as usize == reg)
    {
        return Some(nlocals + stack_idx);
    }
    for &local_idx in live_local_indices {
        if local_idx < nlocals
            && local_color_map
                .get(local_idx)
                .is_some_and(|&color| color as usize == reg)
        {
            return Some(local_idx);
        }
    }
    if local_color_map.is_empty() && reg < nlocals {
        return Some(reg);
    }
    None
}

// Sentinel null JitCode for uninitialized PyreSym.
//
// Cannot be `static` because `Arc::new` is not const; use a thread_local
// LazyCell so the initialiser runs once per thread and the resulting
// reference stays valid for the thread's lifetime.
thread_local! {
    static NULL_JITCODE_CELL: std::cell::OnceCell<JitCode> = const { std::cell::OnceCell::new() };
}

fn null_jitcode() -> &'static JitCode {
    NULL_JITCODE_CELL.with(|cell| {
        let r = cell.get_or_init(|| JitCode {
            code: std::ptr::null(),
            index: -1,
            payload: std::sync::Arc::new(crate::PyJitCode::skeleton(
                std::ptr::null(),
                std::ptr::null(),
                None,
            )),
        });
        // SAFETY: per-thread `OnceCell` initialises once; the
        // resulting reference lives for the thread's lifetime.
        unsafe { &*(r as *const JitCode) }
    })
}

/// Traced value — RPython `FrontendOp(position, _resint/_resref/_resfloat)` parity.
///
/// Carries both the symbolic IR reference (OpRef) and the concrete
/// execution value (ConcreteValue). Created by opcode handlers that
/// compute concrete results alongside IR recording.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontendOp {
    pub opref: OpRef,
    pub concrete: ConcreteValue,
}

impl FrontendOp {
    pub fn new(opref: OpRef, concrete: ConcreteValue) -> Self {
        Self { opref, concrete }
    }

    /// `history.py:649-700` `FrontendOp(pos)` parity — the `type='v'`
    /// (void) variant carries only a recorder position with no value
    /// attribute. Pyre folds RPython's class hierarchy
    /// (`IntFrontendOp`/`RefFrontendOp`/`FloatFrontendOp`/bare
    /// `FrontendOp`) into `ConcreteValue` variants, so the void case is
    /// `concrete: Null`.
    pub fn void(opref: OpRef) -> Self {
        Self {
            opref,
            concrete: ConcreteValue::Null,
        }
    }
}

/// Typed concrete value — RPython `FrontendOp._resint/_resref/_resfloat` parity.
///
/// Python bytecode uses untyped locals, so we use a tagged enum instead of
/// RPython's separate `registers_i/r/f` arrays. Each variant corresponds to
/// one of RPython's Box types: `BoxInt`, `BoxPtr`, `BoxFloat`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConcreteValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Ref(PyObjectRef),
    Null,
}

/// Convert a frame slot value to ConcreteValue, preserving null pointers
/// as Ref(PY_NULL) instead of ConcreteValue::Null. Frame slots always
/// contain known values — null means "uninitialized local", not "untracked".
pub(crate) fn concrete_value_from_slot(obj: PyObjectRef) -> ConcreteValue {
    if obj.is_null() {
        return ConcreteValue::Ref(pyre_object::PY_NULL);
    }
    ConcreteValue::from_pyobj(obj)
}

impl ConcreteValue {
    /// Convert from PyObjectRef (unbox if possible).
    /// Null pointers become ConcreteValue::Null ("untracked").
    pub fn from_pyobj(obj: PyObjectRef) -> Self {
        if obj.is_null() {
            return ConcreteValue::Null;
        }
        unsafe {
            if is_bool(obj) {
                ConcreteValue::Bool(w_bool_get_value(obj))
            } else if is_trace_plain_int(obj) {
                ConcreteValue::Int(w_int_get_value(obj))
            } else if is_trace_plain_float(obj) {
                ConcreteValue::Float(w_float_get_value(obj))
            } else {
                ConcreteValue::Ref(obj)
            }
        }
    }

    /// Convert to PyObjectRef (box if needed).
    pub fn to_pyobj(self) -> PyObjectRef {
        match self {
            ConcreteValue::Int(v) => w_int_new(v),
            ConcreteValue::Float(v) => pyre_object::w_float_new(v),
            ConcreteValue::Bool(v) => pyre_object::w_bool_from(v),
            ConcreteValue::Ref(obj) => obj,
            ConcreteValue::Null => PY_NULL,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, ConcreteValue::Null)
    }

    /// RPython box.getint() parity.
    pub fn getint(&self) -> Option<i64> {
        match self {
            ConcreteValue::Int(v) => Some(*v),
            ConcreteValue::Bool(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// RPython box.getfloatstorage() parity.
    pub fn getfloat(&self) -> Option<f64> {
        match self {
            ConcreteValue::Float(v) => Some(*v),
            ConcreteValue::Int(v) => Some(*v as f64),
            ConcreteValue::Bool(v) => Some(*v as i64 as f64),
            _ => None,
        }
    }

    /// RPython box.getref_base() parity.
    pub fn getref(&self) -> PyObjectRef {
        self.to_pyobj()
    }

    /// Convert to majit IR Type.
    pub fn ir_type(&self) -> Type {
        match self {
            ConcreteValue::Int(_) | ConcreteValue::Bool(_) => Type::Int,
            ConcreteValue::Float(_) => Type::Float,
            ConcreteValue::Ref(_) => Type::Ref,
            ConcreteValue::Null => Type::Ref,
        }
    }

    /// Convert to majit IR `Value` for a Ref-typed virtualizable slot, or
    /// `None` when there is no valid heap pointer to record.
    ///
    /// `locals_cells_stack_w` is declared as a W_Root array
    /// (virtualizable.py:86-98), so slots mirror RPython `Box(W_Root)` —
    /// `read_boxes` / `write_boxes` always see real boxed W_Root values.
    /// Pyre's lazy boxing means `wrapint` / `wrapfloat` emit a
    /// `NewWithVtable` OpRef without eagerly allocating a `W_IntObject` /
    /// `W_FloatObject`, so there is no heap pointer to flush back through
    /// `synchronize_virtualizable`. Returning `None` lets callers skip the
    /// concrete half of the shadow and update only the OpRef via
    /// `set_virtualizable_box_at`, preserving whatever valid W_Root the
    /// slot previously held instead of corrupting the PyFrame with an
    /// invalid pointer.
    pub fn to_ir_ref_value(&self) -> Option<majit_ir::Value> {
        match self {
            ConcreteValue::Ref(obj) => Some(majit_ir::Value::Ref(majit_ir::GcRef(*obj as usize))),
            ConcreteValue::Null => Some(majit_ir::Value::Ref(majit_ir::GcRef(0))),
            ConcreteValue::Int(_) | ConcreteValue::Float(_) | ConcreteValue::Bool(_) => None,
        }
    }

    /// Truth value (RPython box.getint() != 0 for goto_if_not).
    pub fn is_truthy(&self) -> bool {
        match self {
            ConcreteValue::Int(v) => *v != 0,
            ConcreteValue::Float(v) => *v != 0.0,
            ConcreteValue::Bool(v) => *v,
            ConcreteValue::Ref(obj) => objspace_truth_value(*obj),
            ConcreteValue::Null => false,
        }
    }
}

#[inline]
unsafe fn is_trace_plain_int(obj: PyObjectRef) -> bool {
    if !unsafe { py_type_check(obj, &INT_TYPE) } {
        return false;
    }
    let int_typeobj = get_instantiate(&INT_TYPE);
    if int_typeobj.is_null() {
        return unsafe { (*obj).w_class.is_null() };
    }
    let w_class = unsafe { (*obj).w_class };
    w_class.is_null() || std::ptr::eq(w_class, int_typeobj)
}

#[inline]
unsafe fn is_trace_plain_float(obj: PyObjectRef) -> bool {
    if !unsafe { py_type_check(obj, &FLOAT_TYPE) } {
        return false;
    }
    let float_typeobj = get_instantiate(&FLOAT_TYPE);
    if float_typeobj.is_null() {
        return unsafe { (*obj).w_class.is_null() };
    }
    let w_class = unsafe { (*obj).w_class };
    w_class.is_null() || std::ptr::eq(w_class, float_typeobj)
}

use pyre_interpreter::DictStorage;

use crate::descr::{
    PY_OBJECT_ARRAY_GC_TYPE_ID, float_floatval_descr, int_intval_descr, make_array_descr_with_type,
    w_float_size_descr, w_int_size_descr,
};
use crate::frame_layout::{
    PYFRAME_DEBUGDATA_OFFSET, PYFRAME_LASTBLOCK_OFFSET, PYFRAME_LOCALS_CELLS_STACK_OFFSET,
    PYFRAME_PYCODE_OFFSET, PYFRAME_VALUESTACKDEPTH_OFFSET, PYFRAME_W_GLOBALS_OBJ_OFFSET,
};
use crate::helpers::emit_box_float_inline;

// Re-export liveness items so downstream `pyre_jit_trace::state::*` keeps working.
pub use crate::liveness::{LiveVars, liveness_for};

/// Interpreter state exposed to the JIT framework.
///
/// Built from `PyFrame` before calling `back_edge`, and synced back
/// after compiled code runs.
/// Heap is the single source of truth (RPython parity).
/// next_instr / valuestackdepth live on the PyFrame heap object
/// and are accessed via read_frame_usize / write_frame_usize.
#[derive(majit_macros::VirtualizableState)]
pub struct PyreJitState {
    #[vable(frame)]
    pub frame: usize,
    /// blackhole.py:337 parity: liveness PC from rd_numb (setposition PC).
    /// When set, `restore_guard_failure_values` uses this instead of
    /// next_instr for liveness lookup — matching RPython's pattern where
    /// `blackholeinterp.setposition(jitcode, pc)` is called before
    /// `consume_one_section`.
    pub resume_pc: Option<usize>,
}

/// Meta information for a trace — describes the shape of the code being traced.
#[derive(Clone, majit_macros::VirtualizableMeta)]
pub struct PyreMeta {
    #[vable(num_locals)]
    pub num_locals: usize,
    pub ns_len: usize,
    #[vable(valuestackdepth)]
    pub valuestackdepth: usize,
    /// Full `locals_cells_stack_w` length on the heap object
    /// (`co_nlocals + ncellvars + nfreevars + co_stacksize`).
    /// This stays separate from `valuestackdepth`, which is the live depth
    /// (`pyframe.py:111`) in the RPython model.
    pub array_capacity: usize,
    /// Temporary staging count for extra portal reds that sit between the
    /// frame red and the expanded virtualizable payload. Root portal traces
    /// now carry `ec` here; guard-resume bridge traces still use 0 until the
    /// resumedata path is migrated to the same contract.
    pub trace_extra_reds: usize,
    pub has_virtualizable: bool,
    #[vable(slot_types)]
    pub slot_types: Vec<Type>,
}

/// Symbolic state during tracing.
///
/// `frame` maps to a live IR `OpRef`. Symbolic frame field tracking
/// (locals, stack, valuestackdepth, next_instr) persists across instructions.
/// Locals and stack are virtualized (carried through JUMP args);
/// only next_instr and valuestackdepth are synced before guards / loop close.
#[derive(Clone, majit_macros::VirtualizableSym)]
pub struct PyreSym {
    /// OpRef for the owning PyFrame pointer.
    #[vable(frame)]
    pub frame: OpRef,
    /// PyPy portal second red: `ec` from `interp_jit.py:67`.
    ///
    /// This is not part of the virtualizable payload; it lives between the
    /// frame red and the virtualizable expansion when running under the
    /// canonical PyPy driver descriptor.
    pub(crate) execution_context: OpRef,
    // ── Persistent symbolic frame field tracking ──
    // The Python stack (`locals_cells_stack_w[nlocals..]`)
    // lives in the tail of `registers_r`. The macro's `collect_stack`
    // emits `registers_r[nlocals..nlocals + stack_only_depth]` so JUMP /
    // GUARD args carry locals followed by stack in one contiguous window.
    // This matches RPython's MIFrame register file (`pyjitpl.py:70-78`),
    // which treats locals and the stack as a single abstract register
    // vector.
    #[vable(local_types)]
    pub(crate) symbolic_local_types: Vec<Type>,
    #[vable(stack_types)]
    pub symbolic_stack_types: Vec<Type>,
    pub pending_next_instr: Option<usize>,
    pub(crate) locals_cells_stack_array_ref: OpRef,
    #[vable(valuestackdepth)]
    pub(crate) valuestackdepth: usize,
    #[vable(nlocals)]
    pub(crate) nlocals: usize,
    /// Bridge-specific override for the locals slice of registers_r.
    /// resume.py:1042 parity: when set, init_symbolic uses these OpRefs
    /// (mapped from RebuiltValue::Box(n) in rebuild_from_resumedata) instead
    /// of the vable_array_base-based layout. This ensures bridge traces see
    /// frame locals as symbolic InputArgs, not concrete values.
    pub(crate) bridge_local_oprefs: Option<Vec<OpRef>>,
    /// Bridge-specific override for symbolic_local_types.
    /// virtualizable.py:44 + interp_jit.py:25-31: locals_cells_stack_w[*]
    /// is a W_Root array → all items are Type::Ref. setup_bridge_sym
    /// populates this with all-Ref; downstream unboxing happens in
    /// opcode handlers via guard_class + getfield_gc_pure_i/_f, not at
    /// the virtualizable slot level.
    pub(crate) bridge_local_types: Option<Vec<Type>>,
    // virtualizable.py:86-93: ALL static fields in declared order.
    // RPython's unroll_static_fields includes every field from
    // _virtualizable_; ALL must be inputarg (not info_only). The `type`
    // tag below mirrors pyframe.py's static-field declared types so the
    // macro mints `OpRef::input_arg_int/ref` matching the RPython
    // `InputArgInt`/`InputArgRef` class for that slot
    // (resoperation.py:719/739).
    #[vable(inputarg, type = int)]
    pub(crate) vable_last_instr: OpRef,
    #[vable(inputarg, type = ref)]
    pub(crate) vable_pycode: OpRef,
    #[vable(inputarg, type = int)]
    pub(crate) vable_valuestackdepth: OpRef,
    #[vable(inputarg, type = ref)]
    pub(crate) vable_debugdata: OpRef,
    #[vable(inputarg, type = ref)]
    pub(crate) vable_lastblock: OpRef,
    #[vable(inputarg, type = ref)]
    pub(crate) vable_w_globals: OpRef,
    #[vable(array_base)]
    pub(crate) vable_array_base: Option<u32>,
    /// True when this frame's `locals_cells_stack_w` array IS the active
    /// virtualizable shadow for the current trace — i.e. when reads /
    /// writes against this frame's locals must consult / mutate
    /// `TraceCtx::virtualizable_boxes` (RPython `metainterp
    /// .virtualizable_boxes`, pyjitpl.py:1230) instead of going through
    /// regular MIFrame `registers_X`.
    ///
    /// Invariant: `is_active_vable_owner == vable_array_base.is_some()`.
    /// The boolean predicate (this field) and the u32 OpRef-offset (the
    /// `Option<u32>` value of `vable_array_base`) are split because their
    /// semantic roles differ: the predicate decides "consult vable
    /// shadow vs. registers", while the offset is only used as a
    /// fallback OpRef synthesizer at `trace_opcode.rs:1248` when the
    /// metainterp-scope shadow is not yet populated. RPython has neither
    /// per-frame state — the codewriter emits `getarrayitem_vable`
    /// opcodes only on the toplevel frame's bytecode. Pyre dispatches
    /// the same `load_local_value` for all frames, so we encode the
    /// "active vable owner" identity per-frame.
    ///
    /// Helpers `become_active_vable_owner` / `clear_active_vable`
    /// maintain the invariant; do not write `vable_array_base` directly
    /// outside of those (or the macro-generated `init_vable_indices`
    /// invoked through `become_active_vable_owner`).
    pub(crate) is_active_vable_owner: bool,
    // ── MIFrame concrete Box tracking (RPython registers_i/r/f parity) ──
    // Concrete Python object values for locals and stack, tracked in
    // parallel with `registers_r`. Each opcode handler updates these
    // alongside the symbolic OpRefs so that guard decisions, branch
    // directions, and call results use internally tracked values
    // instead of reading from an external PyFrame snapshot.
    pub(crate) concrete_locals: Vec<ConcreteValue>,
    pub concrete_stack: Vec<ConcreteValue>,
    /// pyjitpl.py:74: frame.jitcode — JitCode reference.
    /// Provides both .code (CodeObject*) and .index (snapshot encoding).
    pub(crate) jitcode: *const JitCode,
    /// Namespace for global lookups (W_DictObject / W_ModuleDictObject).
    pub(crate) concrete_namespace: PyObjectRef,
    /// Execution context pointer (for creating callee frames).
    pub(crate) concrete_execution_context: *const pyre_interpreter::PyExecutionContext,
    /// Virtualizable object pointer (PyFrame).
    /// RPython MetaInterp stores the virtualizable separately from MIFrame.
    pub(crate) concrete_vable_ptr: *mut u8,
    /// Function-entry traces use typed locals (RPython MIFrame parity).
    pub(crate) is_function_entry_trace: bool,
    /// RPython MetaInterp.last_exc_value (pyjitpl.py:2745): concrete
    /// exception object pending during tracing. Set by execute_ll_raised
    /// (raise_varargs), consumed by handle_possible_exception.
    pub(crate) last_exc_value: pyre_object::PyObjectRef,
    /// RPython MetaInterp.class_of_last_exc_is_const (pyjitpl.py:2754):
    /// True after GUARD_EXCEPTION or GUARD_CLASS on the exception.
    pub(crate) class_of_last_exc_is_const: bool,
    /// RPython MetaInterp.last_exc_box (pyjitpl.py:1696, 3386): symbolic
    /// OpRef for the exception value. Set directly by `opimpl_raise`, or
    /// by handle_possible_exception after GUARD_EXCEPTION, then consumed
    /// by finishframe_exception for stack push.
    pub(crate) last_exc_box: OpRef,
    /// Symbolic mirror of executioncontext.current_exception/sys_exc_info.
    /// Used by PUSH_EXC_INFO / POP_EXCEPT to preserve nested handler state.
    pub(crate) current_exc_value: pyre_object::PyObjectRef,
    pub(crate) current_exc_box: OpRef,
    /// pyjitpl.py:2597 virtualref_boxes: pairs of (jit_virtual, real_vref).
    /// Each pair: (symbolic OpRef, concrete pointer).
    /// resume.py:1093 restores virtual references on guard failure.
    /// Pairs stored flat: [virt_sym, virt_ptr, real_sym, real_ptr, ...].
    pub(crate) virtualref_boxes: Vec<(OpRef, usize)>,
    // ── RPython MIFrame.registers_{i,r,f} port (pyjitpl.py:74-90) ──
    //
    // RPython reference (target shape):
    //   self.registers_i = [history.CONST_NULL] * jitcode.num_regs_and_consts_i()
    //   self.registers_r = [history.CONST_NULL] * jitcode.num_regs_and_consts_r()
    //   self.registers_f = [history.CONST_NULL] * jitcode.num_regs_and_consts_f()
    //
    // Each bank is sized to `num_regs_X + len(constants_X)` and indexed
    // by post-regalloc-color: `[0, num_regs_X)` are register slots
    // initialised to `CONST_NULL`, `[num_regs_X, ...)` are the constant
    // pool entries copied from `jitcode.constants_X`.
    //
    // SSA-authoritative live_r layout:
    //   - `setup_kind_register_banks` sizes all three banks to
    //     `num_regs_and_consts_X` when the owning JitCode is bound.
    //   - `registers_i` / `registers_f` are indexed by post-regalloc
    //     color. The encoder (`get_list_of_active_boxes`) reads them
    //     directly via the bank clone.
    //   - `registers_r` remains pyre's semantic frame mirror for
    //     `locals_cells_stack_w` because stack colors may coalesce with
    //     local colors. The encoder materializes a temporary
    //     color-indexed Ref-bank snapshot before reading liveness.
    pub(crate) registers_i: Vec<OpRef>,
    #[vable(locals)]
    pub(crate) registers_r: Vec<OpRef>,
    pub(crate) registers_f: Vec<OpRef>,
}

#[doc(hidden)]
pub struct TestSymState {
    pub frame: OpRef,
    pub jitcode: *const (),
    pub nlocals: usize,
    pub valuestackdepth: usize,
    pub locals_cells_stack_array_ref: OpRef,
    pub symbolic_local_types: Vec<Type>,
    pub symbolic_stack_types: Vec<Type>,
    pub registers_r: Vec<OpRef>,
    pub concrete_stack: Vec<ConcreteValue>,
    pub concrete_namespace: PyObjectRef,
    pub vable_last_instr: OpRef,
    pub vable_pycode: OpRef,
    pub vable_valuestackdepth: OpRef,
    pub vable_debugdata: OpRef,
    pub vable_lastblock: OpRef,
    pub vable_w_globals: OpRef,
}

/// Trace-time view over the virtualizable `PyFrame`.
///
/// Per-instruction wrapper that borrows persistent symbolic state from
/// `PyreSym` via raw pointer. The symbolic tracking (locals, stack,
/// valuestackdepth, next_instr) lives in PyreSym and survives across
/// instructions; this struct provides the per-instruction context
/// (ctx, fallthrough_pc).
pub struct MIFrame {
    pub(crate) ctx: *mut TraceCtx,
    pub(crate) sym: *mut PyreSym,
    pub(crate) fallthrough_pc: usize,
    /// Concrete PyFrame address for exception table lookup.
    pub(crate) concrete_frame_addr: usize,
    /// RPython pyjitpl.py orgpc parity: the PC at the START of the current
    /// opcode. All guards within one opcode capture this as their resume PC
    /// so that guard failure re-executes the opcode from the beginning.
    pub(crate) orgpc: usize,
    /// RPython `capture_resumedata(resumepc=orgpc)`
    /// Opcode-start snapshot of the unified `registers_r` file used by
    /// guard/resumedata capture for this one opcode. When `None`, guard
    /// capture reads the live register file directly. Shadow-owner snapshots
    /// are semantic frame prefixes sourced from `virtualizable_boxes`;
    /// non-owner snapshots clone the semantic `registers_r` mirror.
    pub(crate) pre_opcode_registers_r: Option<Vec<OpRef>>,
    /// Semantic frame prefix length at opcode start. This is distinct from
    /// `pre_opcode_registers_r.len()` for non-owner traces because the Ref
    /// bank may include post-regalloc color slots above the semantic
    /// locals+stack prefix.
    pub(crate) pre_opcode_semantic_depth: Option<usize>,
    /// PyPy capture_resumedata: parent frame chain for multi-frame guards.
    /// Each entry points at one parent frame plus the resumepc that
    /// should be used when that parent is snapshotted. This stays much
    /// closer to RPython's `self.framestack` than the old flattened
    /// `(fail_args, fail_arg_types, resumepc, jitcode_index)` tuples.
    pub parent_frames: Vec<ResumeFrameState>,
    /// `pyjitpl.py:181-193` `_result_argcode` analogue for non-top-frame
    /// snapshotting. When present, `get_list_of_active_boxes(in_a_call=True)`
    /// overwrites this caller stack slot with a zero/null placeholder before
    /// liveness encoding.
    pub pending_result_stack_idx: Option<usize>,
    pub pending_result_type: Option<Type>,
    pub pending_inline_frame: Option<PendingInlineFrame>,
}

pub(crate) fn instruction_consumes_comparison_truth(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::PopJumpIfFalse { .. } | Instruction::PopJumpIfTrue { .. }
    )
}

pub(crate) fn instruction_is_trivia_between_compare_and_branch(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::ExtendedArg
            | Instruction::Resume { .. }
            | Instruction::Nop
            | Instruction::Cache
            | Instruction::NotTaken
            | Instruction::ToBool
    )
}

pub(crate) fn instruction_needs_pre_opcode_snapshot(instruction: Instruction) -> bool {
    // Only keep the opcode-start snapshot for bytecodes that can emit a
    // guard after mutating the logical stack/register state. A larger
    // "may raise" set still needs GUARD_{NO_,}EXCEPTION handling, but
    // opcodes like GET_ITER / FOR_ITER / GET_LEN / IMPORT_FROM inspect
    // the current stack via peek-at-TOS and do not need a pre-pop
    // resumestate. Unsupported tracer opcodes also stay out of this set
    // until they gain a real guard-producing lowering.
    matches!(
        instruction,
        Instruction::Call { .. }
            | Instruction::CallKw { .. }
            | Instruction::StoreSubscr
            | Instruction::BinaryOp { .. }
            | Instruction::CompareOp { .. }
            | Instruction::UnaryNegative
            | Instruction::UnaryNot
            | Instruction::UnaryInvert
            | Instruction::RaiseVarargs { .. }
    )
}

/// RPython exc=True parity: instructions that correspond to JitCode ops
/// with exc=True. Only external calls and operations that invoke arbitrary
/// Python code need GUARD_NO_EXCEPTION. Arithmetic, comparisons, and
/// local variable access are lowered to primitive IR ops (exc=False) in
/// RPython and protected by type-specific guards instead.
pub(crate) fn instruction_may_raise(instruction: Instruction) -> bool {
    matches!(
        instruction,
        // RPython exc=True: external calls and attribute access that
        // may invoke arbitrary Python code (__getattr__, descriptors).
        // Unsupported tracer opcodes stay out of this set until they gain a
        // real lowering; otherwise the default "not implemented" error is
        // mis-recorded as a traced GUARD_EXCEPTION.
        Instruction::Call { .. }
            | Instruction::CallKw { .. }
            | Instruction::StoreAttr { .. }
            | Instruction::StoreSubscr
            | Instruction::ImportFrom { .. } // RPython raise/reraise are dedicated opimpls, not generic
                                             // exc=True execute_varargs sites. They unwind directly instead
                                             // of going through handle_possible_exception().
    )
}

/// Environment context — currently unused.
pub struct PyreEnv;

/// Descriptor for raw `PyObjectRef` item pointers that already address
/// `items[0]` (no length prefix to skip). Used after an explicit
/// `IntAdd(block, ITEMS_BLOCK_ITEMS_OFFSET)` converts a
/// `*mut ItemsBlock` block-base pointer into the items-base pointer —
/// see `load_namespace_value` in `trace_opcode.rs` for the canonical
/// pattern. `GETARRAYITEM_GC_R(ptr, i)` lands on `ptr + i * item_size`.
///
/// For callers that hold the block-base pointer directly (i.e.
/// `*mut ItemsBlock` for `W_ListObject.items` / tuple backing storage,
/// or `*mut FixedObjectArray` for `PyFrame.locals_cells_stack_w`), use
/// [`pyobject_gcarray_descr`] instead — its
/// `base_size = FIXED_ARRAY_ITEMS_OFFSET` makes the descriptor itself
/// skip the length prefix.
pub(crate) fn pyobject_array_descr() -> DescrRef {
    // `nolength=True` shape (descr.py:359-360): items start at offset 0,
    // no length header — `GETARRAYITEM_GC_R(ptr, i)` lands on
    // `ptr + i * item_size`.  Stable identity carrier
    // `"pyre::pyobject_array_nolength"` so every `pyobject_array_descr()`
    // call returns the same Arc per PyPy `cpu.arraydescrof(ARRAY)`
    // singleton — `gc_cache._cache_array[LLType::Array(path_hash(...))]`
    // canonicalizes across analyzer / runtime consumers.
    crate::descr::make_array_descr_with_full_id(
        0,
        8,
        0,
        None,
        Type::Ref,
        false,
        Some("pyre::pyobject_array_nolength".to_string()),
    )
}

/// Descriptor for RPython `Ptr(GcArray(PyObjectRef))` containers —
/// `[len][items...]` layout where the pointer addresses the length
/// header. Used by virtual array materialization (`NewArray` +
/// `SetarrayitemGc` in `decode_virtual_info`, resume.py:653-670) and
/// by the virtualizable-frame array field (`locals_cells_stack_w` via
/// the autogenerated `frame_locals_cells_stack_descr()` in
/// `virtualizable_gen.rs`). `base_size = FIXED_ARRAY_ITEMS_OFFSET`
/// skips the length prefix so `GETARRAYITEM_GC_R(array_ptr, i)` lands
/// on items[i] directly.
pub(crate) fn pyobject_gcarray_descr() -> DescrRef {
    // type_id = PY_OBJECT_ARRAY_GC_TYPE_ID so the GC tracer walks each
    // item slot as a Ref. Without this, gen_initialize_tid stamps
    // OBJECT_GC_TYPE_ID into the GC header, the tracer treats the
    // allocation as a 16-byte PyObject, and the variable-part items
    // (the inline-emitted PyFrame.locals_cells_stack_w) survive the
    // first young collection without forwarding — manifesting as
    // wrong fib values on dynasm and SIGSEGV on cranelift once the
    // recursion depth fills the nursery (rgc parity:
    // gctypelayout.py:266-291 T_IS_VARSIZE / T_IS_GCARRAY_OF_GCPTR).
    // Length-prefixed shape (descr.py:362): the array_ptr addresses the
    // length header at offset 0, items start at `FIXED_ARRAY_ITEMS_OFFSET`.
    make_array_descr_with_type(
        pyre_object::FIXED_ARRAY_ITEMS_OFFSET,
        8,
        PY_OBJECT_ARRAY_GC_TYPE_ID,
        Some(0),
        Type::Ref,
        false,
    )
}

pub(crate) fn int_array_descr() -> DescrRef {
    // `nolength=True` view: caller already holds `*ItemsBlock`-style
    // pointer to the items region, so no length header.  Stable
    // identity carrier `"pyre::int_array_nolength"` so every
    // `int_array_descr()` call returns the same Arc (PyPy
    // `cpu.arraydescrof` singleton).
    crate::descr::make_array_descr_with_full_id(
        0,
        8,
        0,
        None,
        Type::Int,
        true,
        Some("pyre::int_array_nolength".to_string()),
    )
}

pub(crate) fn float_array_descr() -> DescrRef {
    crate::descr::make_array_descr_with_full_id(
        0,
        8,
        0,
        None,
        Type::Float,
        false,
        Some("pyre::float_array_nolength".to_string()),
    )
}

/// `descr.py SizeDescr` for the host `PyFrame` virtualizable struct.
///
/// All `PyFrame` field descriptors point at this SizeDescr via
/// `FieldDescr.parent_descr` so the optimizer's `ensure_ptr_info_arg0`
/// (`optimizer.py:478-484`) can dispatch the GETFIELD/SETFIELD branch
/// to `InstancePtrInfo` / `StructPtrInfo`. Also handed to
/// `VirtualizableInfo::set_parent_descr` so virtualizable field
/// descriptors share the same parent.
pub fn pyframe_size_descr() -> DescrRef {
    crate::descr::pyframe_size_descr()
}

pub(crate) fn frame_locals_cells_stack_descr() -> DescrRef {
    crate::descr::pyframe_locals_cells_stack_descr()
}

// R3.3: frame_dict_storage_descr retired — frame_get_namespace now
// reads through w_globals_obj → dict_storage_proxy.

pub(crate) fn wrapint(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    crate::helpers::emit_box_int_inline(ctx, value, w_int_size_descr(), int_intval_descr())
}

/// pyjitpl.py:3514 find_biggest_function
pub(crate) fn biggest_inline_trace_key(state: &mut MIFrame) -> Option<u64> {
    state.with_ctx(|_, ctx| ctx.find_biggest_function())
}

pub(crate) fn note_root_trace_too_long(green_key: u64) {
    let (driver, _) = crate::driver::driver_pair();
    let warm_state = driver.meta_interp_mut().warm_state_mut();
    warm_state.trace_next_iteration(green_key);
    warm_state.mark_force_finish_tracing(green_key);
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][trace-too-long] trace_next_iteration + mark_force_finish_tracing key={}",
            green_key
        );
    }
}

pub(crate) fn wrapfloat(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    emit_box_float_inline(ctx, value, w_float_size_descr(), float_floatval_descr())
}

pub(crate) fn ensure_boxed_for_ca(ctx: &mut TraceCtx, state: &MIFrame, value: OpRef) -> OpRef {
    match state.value_type(value) {
        Type::Int => wrapint(ctx, value),
        Type::Float => wrapfloat(ctx, value),
        Type::Ref | Type::Void => value,
    }
}

pub(crate) fn box_value_for_python_helper(
    state: &mut MIFrame,
    ctx: &mut TraceCtx,
    value: OpRef,
) -> OpRef {
    match state.value_type(value) {
        Type::Int => wrapint(ctx, value),
        Type::Float => wrapfloat(ctx, value),
        Type::Ref | Type::Void => value,
    }
}

pub(crate) fn box_args_for_python_helper(
    state: &mut MIFrame,
    ctx: &mut TraceCtx,
    args: &[OpRef],
) -> Vec<OpRef> {
    args.iter()
        .map(|&arg| box_value_for_python_helper(state, ctx, arg))
        .collect()
}

// RPython parity note: pyjitpl.py (tracer) records GETFIELD_GC ops WITHOUT
// any constant folding. Folding happens exclusively in the optimizer's
// `optimize_GETFIELD_GC_I` (heap.py:639-646), which delegates to
// `optimizer.constant_fold(op)` → `_execute_arglist` → `do_getfield_gc_*`.
// pyre's `OptContext::constant_fold` in optimizeopt/mod.rs is the exact
// port of that path — it handles Int/Float/Ref via `execute_nonspec_const`
// dispatched on `field_type()` and `field_size()`.
//
// The previous tracer-level `try_trace_const_pure_int_field` helper was a
// pyre-specific pre-optimization that duplicated (and mistyped) the
// optimizer logic. It has been removed for structural parity with RPython.

pub(crate) fn try_trace_const_boxed_int(
    ctx: &mut TraceCtx,
    value: OpRef,
    concrete_value: PyObjectRef,
) -> Option<OpRef> {
    if ctx.const_value(value) != Some(concrete_value as i64) {
        return None;
    }
    unsafe {
        if is_bool(concrete_value) {
            return Some(ctx.const_int(if w_bool_get_value(concrete_value) {
                1
            } else {
                0
            }));
        }
        if is_trace_plain_int(concrete_value) {
            return Some(ctx.const_int(w_int_get_value(concrete_value)));
        }
    }
    None
}

/// pyjitpl.py:750-758: read container length.
///
/// RPython's `arraylen_gc` reads the GC array header — there is exactly one
/// length per array, so RPython keeps a per-box `heapc_deps[0]` slot. pyre
/// stores list/bytes/tuple lengths as plain struct fields, so the cached
/// value lives in `HeapCache.heap_cache[descr] -> CacheEntry`
/// (heapcache.py:172).  `opimpl_getfield_gc_i` already does that lookup,
/// so this helper is now just a thin alias kept for source-stability
/// with the call sites.
pub(crate) fn trace_arraylen_gc(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) -> OpRef {
    opimpl_getfield_gc_i(ctx, obj, descr)
}

/// pyjitpl.py:744-748 `opimpl_arraylen_gc`. Emits the actual
/// `ArraylenGc` op against the GcArray header (rlist.py:251
/// `len(l.items)` reads the array's length-prefix). Caller is the
/// items_block Ref (output of `opimpl_getfield_gc_r` on
/// `tuple_wrappeditems_descr` / `list_items_descr`); `descr` is the
/// matching array descr (e.g. `pyobject_gcarray_descr`).
///
/// Distinct from `trace_arraylen_gc` above — that helper reads
/// pyre-specific length FIELDS off the host wrapper struct via
/// getfield (TODO: bring to parity) and is reserved for callers
/// that still go through that path (`str_len`, `dict_len`,
/// `list_length`, typed list `int_items.len`).
pub(crate) fn opimpl_arraylen_gc(ctx: &mut TraceCtx, array: OpRef, descr: DescrRef) -> OpRef {
    if let Some(cached) = ctx.heap_cache().arraylen(array) {
        // pyjitpl.py:889-893 `_opimpl_arraylen_gc` cache hit:
        //     if length is not None:
        //         self.metainterp.staticdata.profiler.count_ops(rop.ARRAYLEN_GC, Counters.HEAPCACHED_OPS)
        //         return lengthbox
        ctx.profiler().count_ops(
            OpCode::ArraylenGc,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        return cached;
    }
    let result = ctx.record_op_with_descr(OpCode::ArraylenGc, &[array], descr.clone());
    // Box(value) parity: executor.py:188 do_arraylen_gc returns
    // BoxInt(cpu.bh_arraylen_gc(array, arraydescr)). Stamp the result
    // OpRef with the live length so downstream box_value(result) sees
    // the same value (matches RPython BoxInt(length) carrier).
    if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(array) {
        let struct_ptr = struct_ref.0 as i64;
        if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
            if let Some(live) = ctx.arraylen_sanity_load(struct_ptr, &descr) {
                ctx.set_opref_concrete(result, live);
            }
        }
    }
    ctx.heap_cache_mut().arraylen_now_known(array, result);
    result
}

pub(crate) fn opimpl_getfield_gc_i(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) -> OpRef {
    // pyjitpl.py:opimpl_getfield_gc_i parity: the tracer does NOT fold
    // pure field reads on constant objects. Folding happens in the
    // optimizer (heap.py:optimize_GETFIELD_GC_I → optimizer.constant_fold),
    // which pyre ports in OptContext::execute_nonspec_const with correct
    // type dispatch (Int/Float/Ref). The tracer only records the GC op.
    //
    // heapcache.py: check if this field was already read/written in this trace
    let field_index = descr.index();
    if let Some(cached) = ctx.heapcache_getfield_cached(obj, field_index) {
        // pyjitpl.py:934-945 cache-hit sanity check (int arm). The
        // line-by-line port runs `executor.execute(cpu, mi, opnum,
        // fielddescr, box)` and asserts `resvalue ==
        // upd.currfieldbox.getint()`.  The cached Box's intrinsic
        // value is fetched via `box_value(cached)` — covering const
        // pool, standard-virtualizable shadow, and the frontend
        // object's `value` field (RPython `currfieldbox.getint()`
        // dispatch parity).
        let cached_value = ctx.box_value(cached).unwrap_or(Value::Void);
        let expected_int = match cached_value {
            majit_ir::Value::Int(n) => Some(n),
            _ => None,
        };
        if let Some(cached_int) = expected_int {
            if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj) {
                let struct_ptr = struct_ref.0 as i64;
                if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
                    if let Some(majit_ir::Value::Int(loaded)) =
                        ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Int)
                    {
                        assert_eq!(
                            loaded, cached_int,
                            "_opimpl_getfield_gc_any_pureornot sanity \
                             check (int): loaded {loaded} != cached \
                             {cached_int} (field_index={field_index}, \
                             struct_ptr={struct_ptr:#x})"
                        );
                    }
                }
            }
        }
        // pyjitpl.py:946 cache-hit accounting:
        //   self.metainterp.staticdata.profiler.count_ops(rop.GETFIELD_GC_I, Counters.HEAPCACHED_OPS)
        //   return upd.currfieldbox
        ctx.profiler().count_ops(
            OpCode::GetfieldGcI,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        return cached;
    }
    // pyjitpl.py:1074-1089: quasi-immutable field handling.
    // Record the field as quasi-immut known so subsequent reads skip
    // the QUASIIMMUT_FIELD op. Emit GUARD_NOT_INVALIDATED if needed.
    // NOTE: GuardNotInvalidated is NOT emitted here — it requires
    // PyreSym.generate_guard for proper snapshot/fail_args (pyjitpl.py:1087
    // generate_guard parity). Instead, set a flag on ctx so the caller
    // (PyreSym with_ctx block) can emit it with full resume data.
    if descr.is_quasi_immutable() {
        if ctx.heap_cache().is_quasi_immut_known(obj, field_index) {
            // pyjitpl.py:1077-1080 cache hit:
            //   if heapcache.is_quasi_immut_known(fielddescr, box):
            //       profiler.count_ops(rop.QUASIIMMUT_FIELD, HEAPCACHED_OPS)
            //       return
            ctx.profiler().count_ops(
                OpCode::QuasiimmutField,
                majit_metainterp::counters::HEAPCACHED_OPS,
            );
        } else {
            ctx.heap_cache_mut().quasi_immut_now_known(obj, field_index);
            ctx.record_op_with_descr(OpCode::QuasiimmutField, &[obj], descr.clone());
            if ctx.heap_cache_mut().check_and_clear_guard_not_invalidated() {
                ctx.set_pending_guard_not_invalidated(Some(ctx.last_traced_pc));
            }
        }
    }
    let opcode = if descr.is_always_pure() {
        OpCode::GetfieldGcPureI
    } else {
        OpCode::GetfieldGcI
    };
    let result = ctx.record_op_with_descr(opcode, &[obj], descr.clone());
    // pyjitpl.py:948-949 `resbox = execute_with_descr(...); upd.getfield_now_known(resbox)`.
    // `resbox` carries the loaded value; pair the recorded opref with
    // the live int from `field_sanity_load` so subsequent
    // `box_value(result)` mirrors RPython's executor-returned Box.
    let live_value = if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj) {
        let struct_ptr = struct_ref.0 as i64;
        if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
            ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Int)
                .unwrap_or(majit_ir::Value::Void)
        } else {
            majit_ir::Value::Void
        }
    } else {
        majit_ir::Value::Void
    };
    if !matches!(live_value, majit_ir::Value::Void) {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getfield_now_known(obj, field_index, result);
    result
}

/// pyjitpl.py:874-882 `opimpl_getfield_gc_r`. Same shape as `_i`
/// modulo the rop variant — folding lives in the optimizer
/// (`optimize_GETFIELD_GC_R = optimize_GETFIELD_GC_I` per RPython's
/// alias), so the tracer only records the GC op.
pub(crate) fn opimpl_getfield_gc_r(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) -> OpRef {
    let field_index = descr.index();
    if let Some(cached) = ctx.heapcache_getfield_cached(obj, field_index) {
        // pyjitpl.py:934-945 cache-hit sanity check (ref arm).
        // `box_value(cached)` resolves the upstream
        // `currfieldbox.getref_base()` payload through the full chain
        // (const pool, standard-virtualizable shadow, the frontend
        // object's `value` field).
        let cached_value = ctx.box_value(cached).unwrap_or(Value::Void);
        let expected_ref = match cached_value {
            majit_ir::Value::Ref(r) => Some(r),
            _ => None,
        };
        if let Some(cached_ref) = expected_ref {
            if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj) {
                let struct_ptr = struct_ref.0 as i64;
                if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
                    if let Some(majit_ir::Value::Ref(loaded)) =
                        ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Ref)
                    {
                        assert_eq!(
                            loaded, cached_ref,
                            "_opimpl_getfield_gc_any_pureornot sanity \
                             check (ref): loaded {:#x} != cached {:#x} \
                             (field_index={field_index}, struct_ptr=\
                             {struct_ptr:#x})",
                            loaded.0, cached_ref.0,
                        );
                    }
                }
            }
        }
        // pyjitpl.py:946 — RPython hardcodes `GETFIELD_GC_I` regardless
        // of the rop variant (`_i` / `_r` / `_f`); pyre matches.
        ctx.profiler().count_ops(
            OpCode::GetfieldGcI,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        return cached;
    }
    if descr.is_quasi_immutable() {
        if ctx.heap_cache().is_quasi_immut_known(obj, field_index) {
            // pyjitpl.py:1077-1080 cache hit (see opimpl_getfield_gc_i above).
            ctx.profiler().count_ops(
                OpCode::QuasiimmutField,
                majit_metainterp::counters::HEAPCACHED_OPS,
            );
        } else {
            ctx.heap_cache_mut().quasi_immut_now_known(obj, field_index);
            ctx.record_op_with_descr(OpCode::QuasiimmutField, &[obj], descr.clone());
            if ctx.heap_cache_mut().check_and_clear_guard_not_invalidated() {
                ctx.set_pending_guard_not_invalidated(Some(ctx.last_traced_pc));
            }
        }
    }
    let opcode = if descr.is_always_pure() {
        OpCode::GetfieldGcPureR
    } else {
        OpCode::GetfieldGcR
    };
    let result = ctx.record_op_with_descr(opcode, &[obj], descr.clone());
    // pyjitpl.py:948-949 `resbox = execute_with_descr(...); upd.getfield_now_known(resbox)`.
    // Pair the recorded opref with the live ref so subsequent
    // `box_value(result)` mirrors RPython's executor-returned Box.
    let live_value = if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj) {
        let struct_ptr = struct_ref.0 as i64;
        if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
            ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Ref)
                .unwrap_or(majit_ir::Value::Void)
        } else {
            majit_ir::Value::Void
        }
    } else {
        majit_ir::Value::Void
    };
    if !matches!(live_value, majit_ir::Value::Void) {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getfield_now_known(obj, field_index, result);
    result
}

// Note: pyre does not currently route GetfieldGcF/GetfieldGcPureF through
// state.rs. Float field unboxing goes via the codewriter-generated
// `getfield_gc_f_pureornot` (majit-translate/src/codegen.rs),
// which — matching RPython's pyjitpl.py opimpl_getfield_gc_f — records
// the GC op without folding. The optimizer's `optimize_GETFIELD_GC_F`
// (= `optimize_GETFIELD_GC_I` via RPython's alias) handles folding.

/// Unbox int with proper GuardClass resume data via MIFrame::generate_guard.
pub(crate) fn trace_unbox_int_with_resume(
    frame: &mut MIFrame,
    ctx: &mut TraceCtx,
    obj: OpRef,
    int_type_addr: i64,
) -> OpRef {
    trace_unbox_int_with_resume_descr(
        frame,
        ctx,
        obj,
        int_type_addr,
        crate::descr::int_intval_descr(),
    )
}

pub(crate) fn trace_unbox_int_with_resume_descr(
    frame: &mut MIFrame,
    ctx: &mut TraceCtx,
    obj: OpRef,
    type_addr: i64,
    intval_descr: majit_ir::DescrRef,
) -> OpRef {
    // pyjitpl.py GUARD_CLASS(box, cls): guard takes object box directly,
    // backend loads typeptr at offset 0.
    if !ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.const_int(type_addr);
        frame.generate_guard(ctx, OpCode::GuardClass, &[obj, type_const]);
        ctx.heap_cache_mut().class_now_known(obj, type_addr);
    }
    crate::generated::trace_unbox_int(
        ctx,
        obj,
        type_addr,
        crate::descr::ob_type_descr(),
        intval_descr,
    )
}

/// Unbox a `W_LongObject` (whose BigInt fits in i64) into a raw i64 OpRef.
///
/// Lowering — mirrors `_int_w_unsafe()` (longobject.py:127) under the
/// fits_int precondition (`is_plain_int1` accepts both W_IntObject and
/// fits_int W_LongObject; listobject.py:1957-1958):
///
/// 1. `GUARD_CLASS(obj, LONG_TYPE)` — concrete type observed at trace.
/// 2. `residual_call(jit_w_long_fits_int, obj) -> i64` — runtime
///    fits_int probe. Subsequent trace executions may see a W_LongObject
///    whose BigInt has grown out of i64 range; the residual call captures
///    that observation per execution.
/// 3. `GUARD_TRUE(fits_int_result)` — bridge if the runtime BigInt does
///    not fit (alternatively: deopt back to the interpreter).
/// 4. `residual_call(jit_w_long_toint, obj) -> i64` —
///    `W_LongObject.toint()` (`longobject.py:138`) → `rbigint.toint()`
///    (`rbigint.py:465`, elidable). OverflowError is statically
///    unreachable post-fits-int GUARD_TRUE.
pub(crate) fn trace_unbox_long_with_resume(
    frame: &mut MIFrame,
    ctx: &mut TraceCtx,
    obj: OpRef,
    long_type_addr: i64,
) -> OpRef {
    if !ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.const_int(long_type_addr);
        frame.generate_guard(ctx, OpCode::GuardClass, &[obj, type_const]);
        ctx.heap_cache_mut().class_now_known(obj, long_type_addr);
    }
    let fits_fn =
        ctx.const_int(pyre_object::longobject::jit_w_long_fits_int as *const () as usize as i64);
    let fits_descr = crate::descr::make_jit_w_long_fits_int_calldescr();
    let fits_result = ctx.record_op_with_descr(OpCode::CallI, &[fits_fn, obj], fits_descr);
    frame.generate_guard(ctx, OpCode::GuardTrue, &[fits_result]);
    let unbox_fn =
        ctx.const_int(pyre_object::longobject::jit_w_long_toint as *const () as usize as i64);
    let unbox_descr = crate::descr::make_jit_w_long_toint_calldescr();
    ctx.record_op_with_descr(OpCode::CallI, &[unbox_fn, obj], unbox_descr)
}

/// Unbox float with proper GuardClass resume data via MIFrame::generate_guard.
pub(crate) fn trace_unbox_float_with_resume(
    frame: &mut MIFrame,
    ctx: &mut TraceCtx,
    obj: OpRef,
    float_type_addr: i64,
) -> OpRef {
    if !ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.const_int(float_type_addr);
        frame.generate_guard(ctx, OpCode::GuardClass, &[obj, type_const]);
        ctx.heap_cache_mut().class_now_known(obj, float_type_addr);
    }
    crate::generated::trace_unbox_float(
        ctx,
        obj,
        float_type_addr,
        crate::descr::ob_type_descr(),
        crate::descr::float_floatval_descr(),
    )
}

pub(crate) unsafe fn objspace_compare_ints(
    lhs_obj: PyObjectRef,
    rhs_obj: PyObjectRef,
    op: ComparisonOperator,
) -> bool {
    unsafe {
        let lhs = w_int_get_value(lhs_obj);
        let rhs = w_int_get_value(rhs_obj);
        match op {
            ComparisonOperator::Less => lhs < rhs,
            ComparisonOperator::LessOrEqual => lhs <= rhs,
            ComparisonOperator::Greater => lhs > rhs,
            ComparisonOperator::GreaterOrEqual => lhs >= rhs,
            ComparisonOperator::Equal => lhs == rhs,
            ComparisonOperator::NotEqual => lhs != rhs,
        }
    }
}

/// baseobjspace as_float: coerce int|float → f64.
/// Called only for int/float operands in the tracing fast path.
/// Long operands are handled by residual fallback, not this function.
unsafe fn as_float_for_trace(obj: PyObjectRef) -> f64 {
    unsafe {
        if is_float(obj) {
            w_float_get_value(obj)
        } else if is_int(obj) {
            w_int_get_value(obj) as f64
        } else {
            0.0 // unreachable in trace fast path — long triggers residual
        }
    }
}

/// Compare two numeric values as floats. Handles float_pair (int+float)
/// via as_float coercion matching baseobjspace::float_lt/le/gt/ge/eq/ne.
/// Long operands don't reach here — they trigger residual fallback.
pub(crate) unsafe fn objspace_compare_floats(
    lhs_obj: PyObjectRef,
    rhs_obj: PyObjectRef,
    op: ComparisonOperator,
) -> bool {
    unsafe {
        let lhs = as_float_for_trace(lhs_obj);
        let rhs = as_float_for_trace(rhs_obj);
        match op {
            ComparisonOperator::Less => lhs < rhs,
            ComparisonOperator::LessOrEqual => lhs <= rhs,
            ComparisonOperator::Greater => lhs > rhs,
            ComparisonOperator::GreaterOrEqual => lhs >= rhs,
            ComparisonOperator::Equal => lhs == rhs,
            ComparisonOperator::NotEqual => lhs != rhs,
        }
    }
}

/// virtualizable.py:94 `getattr(virtualizable, fieldname)` parity for
/// the `locals_cells_stack_w` array field. Materialises the array
/// pointer that step 2 (`lst[i]` → `GETARRAYITEM_GC_R`) indexes.
///
/// TODO: the upstream-orthodox emission is
/// `OpCode::GetfieldGcR` because `pyframe_locals_cells_stack_descr`
/// is field 0 of `PYFRAME_DESCR_GROUP` with `field_type = Type::Ref`
/// on a `PYFRAME_GC_TYPE_ID`-typed PyFrame, so the read goes through
/// the GC barrier in RPython's `rclass.py` getfield emission.
///
/// Cranelift backend status (MAJIT_PROBE_GETFIELD_GC_R=1):
///   - `OpCode::GetfieldGcR` lowering exists at
///     `majit-backend-cranelift/src/compiler.rs:10691`.
///   - Direct swap on fib_recursive panics inside
///     `gc_alloc_nursery_shim` with non-unwinding abort.  The
///     post-getfield write-barrier path triggers a nursery
///     allocation (remembered-set slot) which overflows or hits
///     a missing slow-path stub.
///
/// The convergence path is to either (a) implement the missing
/// nursery allocation slow-path for the post-GetfieldGcR remembered
/// set write, or (b) audit why the GC barrier emits a nursery
/// allocation here when dynasm doesn't.  Both are separate
/// cranelift backend work that is not yet done.  Until then the
/// emission stays
/// `GetfieldRawI` and the runtime descr's `field_type = Type::Ref`
/// preserves the optimizer's boxed-pointer view.
pub(crate) fn frame_locals_cells_stack_array(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.record_op_with_descr(
        OpCode::GetfieldRawI,
        &[frame],
        frame_locals_cells_stack_descr(),
    )
}

/// Read from frame's `locals_cells_stack_w` array. Caller passes the
/// `*mut FixedObjectArray` block-base pointer (output of
/// `frame_locals_cells_stack_array`); the
/// [`pyobject_gcarray_descr`] sets
/// `base_size = FIXED_ARRAY_ITEMS_OFFSET` so the descriptor skips the
/// length prefix and `GETARRAYITEM_GC_R(array, i)` lands on
/// `items[i]`.
///
/// Uses GcR (Ref-typed) to match RPython's GETARRAYITEM_GC_R, ensuring
/// the optimizer knows these are boxed pointers.
pub(crate) fn trace_array_getitem_value(ctx: &mut TraceCtx, array: OpRef, index: OpRef) -> OpRef {
    let descr = pyobject_gcarray_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heapcache_getarrayitem(array, index, descr_idx) {
        return cached;
    }
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcR, &[array, index], descr.clone());
    let live_value = array_load_for_cache(ctx, array, index, &descr, majit_ir::Type::Ref);
    if !matches!(live_value, majit_ir::Value::Void) {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getarrayitem_now_known(array, index, descr_idx, result);
    result
}

/// Helper: project (array, index) operands and dispatch
/// `array_sanity_load` to recover the executor-returned payload for
/// the fresh GetarrayitemGc<I|R|F> result.  Mirrors
/// `rpython/jit/metainterp/executor.py:117` `do_getarrayitem_gc_*`
/// which reads `arraybox.getref_base()` / `indexbox.getint()` from
/// the Box directly — pyre's `box_value` exposes the same chain (Const
/// pool / standard-virtualizable shadow / `opref_concrete` stamp) so
/// any operand whose Box.value is known unblocks the sanity load
/// (not gated on the Const-only fast path).
fn array_load_for_cache(
    ctx: &majit_metainterp::TraceCtx,
    array: OpRef,
    index: OpRef,
    descr: &DescrRef,
    kind: majit_ir::Type,
) -> majit_ir::Value {
    let Some(majit_ir::Value::Ref(array_ref)) = ctx.box_value(array) else {
        return majit_ir::Value::Void;
    };
    let array_ptr = array_ref.0 as i64;
    if array_ptr == usize::MAX as i64 || array_ptr == 0 {
        return majit_ir::Value::Void;
    }
    let Some(majit_ir::Value::Int(index_value)) = ctx.box_value(index) else {
        return majit_ir::Value::Void;
    };
    ctx.array_sanity_load(array_ptr, index_value, descr, kind)
        .unwrap_or(majit_ir::Value::Void)
}

/// Read from frame's locals_cells_stack_w — namespace access path.
pub(crate) fn trace_raw_array_getitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = pyobject_array_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heapcache_getarrayitem(array, index, descr_idx) {
        return cached;
    }
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcR, &[array, index], descr.clone());
    let live_value = array_load_for_cache(ctx, array, index, &descr, majit_ir::Type::Ref);
    if !matches!(live_value, majit_ir::Value::Void) {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getarrayitem_now_known(array, index, descr_idx, result);
    result
}

/// `pyjitpl.py:832` `arraybox = opimpl_getfield_gc_r(listbox, itemsdescr)`
/// followed by `getarrayitem_gc(arraybox, idx, arraydescr)`.
///
/// Caller passes the `items_block` Ref (output of
/// `opimpl_getfield_gc_r` against the `items` /  `wrappeditems` field)
/// directly. The `pyobject_gcarray_descr` here is
/// `Ptr(GcArray(OBJECTPTR))` with `base_size = ITEMS_BLOCK_ITEMS_OFFSET`
/// (= length-prefix size), `item_size = 8`, `item_type = Ref`,
/// matching `rpython/rtyper/lltypesystem/rlist.py:84` `GcArray(OBJECTPTR)`.
///
/// Replaces the prior two-step `IntAdd(items_block, OFFSET) +
/// raw-array op` deviation with the upstream single-op shape.
pub(crate) fn trace_items_block_getitem_value(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = pyobject_gcarray_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heapcache_getarrayitem(block, index, descr_idx) {
        return cached;
    }
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcR, &[block, index], descr.clone());
    let live_value = array_load_for_cache(ctx, block, index, &descr, majit_ir::Type::Ref);
    if !matches!(live_value, majit_ir::Value::Void) {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getarrayitem_now_known(block, index, descr_idx, result);
    result
}

/// Companion of [`trace_items_block_getitem_value`] — emits
/// `setarrayitem_gc(block, index, value)` against `pyobject_gcarray_descr`.
pub(crate) fn trace_items_block_setitem_value(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
    value: OpRef,
) {
    let descr = pyobject_gcarray_descr();
    let descr_idx = descr.index();
    ctx.record_op_with_descr(OpCode::SetarrayitemGc, &[block, index, value], descr);
    // pyjitpl.py:980 `upd.setarrayitem(valuebox)` — cache stores the
    // Box identity (`value` OpRef); cache-hit readers resolve the
    // intrinsic value via `box_value(cached)` at hit time.
    ctx.heapcache_setarrayitem(block, index, descr_idx, value);
}

pub(crate) fn trace_raw_int_array_getitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = int_array_descr();
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemRawI, &[array, index], descr.clone());
    // Box(value) parity: executor.py:132 do_getarrayitem_raw_i projects
    // `arraybox.getint()` (raw pointer carried as Int, not the
    // `getref_base()` projection used by executor.py:117 do_get
    // arrayitem_gc_i).  Route the sanity load through the raw helper
    // family so the descriptor path matches `cpu.bh_getarrayitem_raw_i`.
    if let (Some(majit_ir::Value::Int(arr_ptr)), Some(majit_ir::Value::Int(idx))) =
        (ctx.box_value(array), ctx.box_value(index))
    {
        if arr_ptr != 0 {
            if let Some(live) = ctx.raw_array_sanity_load(arr_ptr, idx, &descr, majit_ir::Type::Int)
            {
                ctx.set_opref_concrete(result, live);
            }
        }
    }
    result
}

pub(crate) fn trace_raw_float_array_getitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = float_array_descr();
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemRawF, &[array, index], descr.clone());
    // executor.py:132 do_getarrayitem_raw_f — same Int-carried raw
    // pointer projection as the int variant above.
    if let (Some(majit_ir::Value::Int(arr_ptr)), Some(majit_ir::Value::Int(idx))) =
        (ctx.box_value(array), ctx.box_value(index))
    {
        if arr_ptr != 0 {
            if let Some(live) =
                ctx.raw_array_sanity_load(arr_ptr, idx, &descr, majit_ir::Type::Float)
            {
                ctx.set_opref_concrete(result, live);
            }
        }
    }
    result
}

/// Write to frame's locals_cells_stack_w array.
/// Uses Gc (GC-typed) to match RPython's SETARRAYITEM_GC.
pub(crate) fn trace_raw_array_setitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
    value: OpRef,
) {
    let descr = pyobject_array_descr();
    let descr_idx = descr.index();
    ctx.record_op_with_descr(OpCode::SetarrayitemGc, &[array, index, value], descr);
    // pyjitpl.py:980 `upd.setarrayitem(valuebox)` parity — cache
    // stores the Box identity (`value` OpRef); cache-hit readers
    // resolve the intrinsic value via `box_value(cached)` at hit
    // time.
    ctx.heapcache_setarrayitem(array, index, descr_idx, value);
}

pub(crate) fn trace_raw_int_array_setitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
    value: OpRef,
) {
    ctx.record_op_with_descr(
        OpCode::SetarrayitemRaw,
        &[array, index, value],
        int_array_descr(),
    );
}

pub(crate) fn trace_raw_float_array_setitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
    value: OpRef,
) {
    ctx.record_op_with_descr(
        OpCode::SetarrayitemRaw,
        &[array, index, value],
        float_array_descr(),
    );
}

/// pyframe.py:49 `self.w_globals` — read the canonical dict object
/// from the frame.  Returns a PyObjectRef (W_DictObject or
/// W_ModuleDictObject).  FFI helpers that receive namespace_ptr
/// chase dict_storage_proxy internally.
pub(crate) fn frame_get_globals_obj(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[frame],
        crate::descr::pyframe_w_globals_obj_descr(),
    )
}

/// Read through w_globals_obj → dict_storage_proxy to reach the raw
/// DictStorage* for slot-based namespace reads (celldict quasiimmut
/// path).  Only valid when globals is a W_ModuleDictObject.
/// Read a value from the unified `locals_cells_stack_w` at the given absolute index.
pub fn concrete_stack_value(frame: usize, abs_idx: usize) -> Option<PyObjectRef> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *const pyre_object::FixedObjectArray)
    };
    let arr = unsafe { &*arr_ptr };
    arr.as_slice().get(abs_idx).copied()
}

/// pyframe.py:107-110: `locals_cells_stack_w` length =
/// `co_nlocals + ncellvars + nfreevars + co_stacksize`. Returns the
/// full heap-side array length (matching `virtualizable.py:86-99
/// read_boxes` which iterates `len(lst)` over the full array).
pub(crate) fn concrete_frame_array_len(frame: usize) -> Option<usize> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *const pyre_object::FixedObjectArray)
    };
    if arr_ptr.is_null() {
        return None;
    }
    Some(unsafe { &*arr_ptr }.as_slice().len())
}

/// pyframe.py:111: valuestackdepth = co_nlocals + ncellvars + nfreevars.
/// Returns the stack base index (nlocals + ncells) for the given frame.
/// This is the number of non-stack slots in the unified locals_cells_stack_w
/// array: local variables + cell/free variable slots.
pub(crate) fn concrete_nlocals(frame: usize) -> Option<usize> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let w_code =
        unsafe { *(frame_ptr.add(crate::frame_layout::PYFRAME_PYCODE_OFFSET) as *const *const ()) };
    if w_code.is_null() {
        return None;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    let code = unsafe { &*raw_code };
    let nlocals = code.varnames.len();
    let ncells = pyre_interpreter::pyframe::ncells(code);
    Some(nlocals + ncells)
}

/// Return the absolute valuestackdepth.
pub(crate) fn concrete_stack_depth(frame: usize) -> Option<usize> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    Some(unsafe { *(frame_ptr.add(PYFRAME_VALUESTACKDEPTH_OFFSET) as *const usize) })
}

/// Derive `(num_locals, num_locals + max_stackdepth)` from a `CodeObject`.
///
/// Mirrors the `(callee_nlocals, callee_vsd)` pair the trace-side reads
/// from `driver.get_compiled_meta(callee_key)` for CALL_ASSEMBLER
/// emission (`trace_opcode.rs:4448-4450`). The second value is sized to
/// match `pyframe.rs:1576` (`alloc_fixed_array_with_header(num_locals +
/// num_cells + max_stack, ...)`) — i.e. heap capacity rather than live
/// depth. Used as the fallback shape when no `compiled_meta` exists yet
/// (e.g. tmp_callback target where `compile_tmp_callback` produced a
/// JCT but no compiled-loop metadata).
#[allow(dead_code)]
pub(crate) fn callee_layout_for_call_assembler(
    code: &pyre_interpreter::CodeObject,
) -> (usize, usize) {
    let nlocals = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);
    let stack_only = code.max_stackdepth as usize;
    (nlocals, nlocals + stack_only)
}

/// `celldict.py:42-50 getdictvalue_no_unwrapping` slot lookup against the
/// module dict's `ModuleDictStorage` (the cell store), bypassing the
/// `DictStorage` shadow.  `None` for non-module dicts and after
/// `switch_to_object_strategy`.  Used by the cell fast path to derive the
/// elidable lookup key.
pub(crate) fn module_dict_cell_slot_direct(obj: PyObjectRef, name: &str) -> Option<usize> {
    unsafe { pyre_object::dictmultiobject::module_dict_cell_slot_of(obj, name) }
}

/// `celldict.py:53-54 _getdictvalue_no_unwrapping_pure` — the raw stored
/// value-or-cell at `slot` (not unwrapped), read at trace time to
/// classify the cell (`ObjectMutableCell` → live `GetfieldGcR`, raw value
/// → const fold, `IntMutableCell` → name-based fallback).
pub(crate) fn module_dict_cell_value_direct(obj: PyObjectRef, slot: usize) -> Option<PyObjectRef> {
    unsafe { pyre_object::dictmultiobject::module_dict_cell_at(obj, slot) }
}

/// virtualizable.py:44 + interp_jit.py:25-31 —
/// `locals_cells_stack_w[*]` is declared as a W_Root array, so every
/// item's JIT type is GCREF (Type::Ref). W_IntObject/W_FloatObject are
/// stored as Ref pointers; unboxing happens inside trace opcode handlers
/// via `guard_class + getfield_gc_pure_i` / `_f`, never at the
/// virtualizable slot level.
pub(crate) fn concrete_virtualizable_slot_type(_value: PyObjectRef) -> Type {
    Type::Ref
}

pub(crate) fn looks_like_heap_ref(value: PyObjectRef) -> bool {
    let addr = value as usize;
    let word_align = std::mem::align_of::<usize>() - 1;
    addr >= 0x1_0000 && addr < ((1u64 << 56) as usize) && (addr & word_align) == 0
}

pub(crate) fn extract_concrete_typed_value(slot_type: Type, value: PyObjectRef) -> Value {
    match slot_type {
        Type::Int => {
            if value.is_null() {
                Value::Int(0)
            } else if looks_like_heap_ref(value) && unsafe { is_int(value) } {
                Value::Int(unsafe { w_int_get_value(value) })
            } else {
                Value::Int(value as i64)
            }
        }
        Type::Float => {
            if value.is_null() {
                Value::Float(0.0)
            } else if looks_like_heap_ref(value) && unsafe { is_float(value) } {
                Value::Float(unsafe { pyre_object::floatobject::w_float_get_value(value) })
            } else {
                Value::Float(f64::from_bits(value as u64))
            }
        }
        Type::Ref | Type::Void => Value::Ref(majit_ir::GcRef(value as usize)),
    }
}

pub(crate) fn concrete_slot_types(
    frame: usize,
    num_locals: usize,
    valuestackdepth: usize,
) -> Vec<Type> {
    let stack_only = valuestackdepth.saturating_sub(num_locals);
    let mut types = Vec::with_capacity(num_locals + stack_only);
    for idx in 0..num_locals {
        types.push(
            concrete_stack_value(frame, idx)
                .map(concrete_virtualizable_slot_type)
                .unwrap_or(Type::Ref),
        );
    }
    for stack_idx in 0..stack_only {
        types.push(
            concrete_stack_value(frame, num_locals + stack_idx)
                .map(concrete_virtualizable_slot_type)
                .unwrap_or(Type::Ref),
        );
    }
    types
}

impl PyreMeta {
    /// Heap capacity of `locals_cells_stack_w`, distinct from the live
    /// `valuestackdepth`.
    pub fn array_stack_only_depth(&self) -> usize {
        self.array_capacity.saturating_sub(self.num_locals)
    }
}

pub(crate) fn boxed_slot_i64_for_type(slot_type: Type, raw: i64) -> PyObjectRef {
    match slot_type {
        Type::Int => w_int_new(raw),
        Type::Float => pyre_object::floatobject::w_float_new(f64::from_bits(raw as u64)),
        Type::Ref | Type::Void => raw as PyObjectRef,
    }
}

/// virtualizable.py:136 `lst[j] = reader.load_next_value_of_type(ARRAYITEMTYPE)`:
/// pyre's `locals_cells_stack_w` array item type is GCREF, so every write to
/// a frame slot must produce a boxed PyObjectRef regardless of the label's
/// argument type (a Body-Label Int OpRef still stores a W_IntObject in the
/// array). `slot_type` is retained for call-site symmetry but no longer
/// gates the boxing decision.
pub(crate) fn boxed_slot_value_for_type(_slot_type: Type, value: &Value) -> PyObjectRef {
    match value {
        Value::Int(v) => w_int_new(*v),
        Value::Float(v) => pyre_object::floatobject::w_float_new(*v),
        Value::Ref(r) => r.as_usize() as PyObjectRef,
        Value::Void => PY_NULL,
    }
}

/// virtualizable.py:126/139 parity: box value for frame array slot.
/// Frame array items (locals_cells_stack_w[*]) are declared as GCREF
/// (interp_jit.py:25). The optimizer may unbox ints/floats in fail_args;
/// this function re-boxes them for the frame. Ref values pass through.
pub(crate) fn virtualizable_box_value(value: &Value) -> PyObjectRef {
    match value {
        Value::Ref(r) => r.as_usize() as PyObjectRef,
        Value::Int(v) => w_int_new(*v),
        Value::Float(v) => pyre_object::floatobject::w_float_new(*v),
        Value::Void => PY_NULL,
    }
}

pub(crate) fn fail_arg_opref_for_typed_value(ctx: &mut TraceCtx, value: Value) -> OpRef {
    match value {
        Value::Int(v) => ctx.const_int(v),
        Value::Float(v) => ctx.const_int(v.to_bits() as i64),
        Value::Ref(r) => ctx.const_ref(r.as_usize() as i64),
        Value::Void => ctx.const_ref(PY_NULL as i64),
    }
}

pub(crate) fn frame_callable_arg_types(nargs: usize) -> Vec<Type> {
    let mut types = Vec::with_capacity(2 + nargs);
    types.push(Type::Ref);
    types.push(Type::Ref);
    for _ in 0..nargs {
        types.push(Type::Ref);
    }
    types
}

pub(crate) fn one_arg_callee_frame_helper(
    arg_type: Type,
    is_self_recursive: bool,
) -> (*const (), Vec<Type>) {
    match (is_self_recursive, arg_type) {
        (true, Type::Int) => (
            crate::callbacks::get().jit_create_self_recursive_callee_frame_1_raw_int,
            vec![Type::Ref, Type::Int],
        ),
        (true, _) => (
            crate::callbacks::get().jit_create_self_recursive_callee_frame_1,
            vec![Type::Ref, Type::Ref],
        ),
        (false, Type::Int) => (
            crate::callbacks::get().jit_create_callee_frame_1_raw_int,
            vec![Type::Ref, Type::Ref, Type::Int],
        ),
        (false, _) => (
            crate::callbacks::get().jit_create_callee_frame_1,
            vec![Type::Ref, Type::Ref, Type::Ref],
        ),
    }
}

#[allow(dead_code)]
pub(crate) fn fail_arg_types_for_virtualizable_state(len: usize) -> Vec<Type> {
    let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
    crate::virtualizable_gen::virt_live_value_types(len.saturating_sub(n))
}

/// `pyjitpl.py:98-119 MIFrame.copy_constants` line-by-line port.
///
/// ```python
/// def copy_constants(self, registers, constants, targetindex, ConstClass):
///     num_regs_and_consts = targetindex + len(constants)
///     if registers is None or len(registers) < num_regs_and_consts:
///         registers = [missing] * num_regs_and_consts
///     elif not we_are_translated():
///         for i in range(len(registers)):
///             registers[i] = missing
///     for i in range(len(constants)):
///         registers[targetindex] = ConstClass(constants[i])
///         targetindex += 1
///     return registers
/// ```
///
/// `ConstClass(constants[i])` is supplied as the closure `mint`. The
/// `elif not we_are_translated()` debug wipe is upstream-only
/// (translation-time fill of stale slots with `missing`); pyre always
/// runs the translated path. `MIFrame.setup` can be re-invoked on
/// `free_frames_list` reused frames (`pyjitpl.py:2471`), and each call
/// re-mints `ConstClass(constants[i])` unconditionally — the fill loop
/// here matches that.
///
/// TODOs (pre-main divergences kept for now):
///
/// 1. Grow-in-place via `resize(num_regs_and_consts, NONE)` instead of
///    full replacement `[missing] * num_regs_and_consts` (the
///    `if registers is None or len(registers) < ...` arm at
///    `pyjitpl.py:109`). Convergence is blocked by Adaptation 2 below:
///    callers like `trace_opcode.rs:5466-5476` push callee args into
///    `sym.registers_r[0..args.len()]` BEFORE invoking this helper, and
///    a full replacement would zero those slots. Migrating to
///    full-replace requires reordering the trace_opcode callee setup
///    so the regalloc-color slot writes happen after
///    `setup_kind_register_banks`, plus matching opcode-dispatch reads
///    against the post-color slots rather than sequential
///    `[0..nlocals)`. That is a multi-file restructuring outside the
///    scope of this helper.
///
/// 2. `registers_r` carries both pyre's semantic locals/stack mirror
///    AND the post-regalloc-color Ref bank (see the wider doc on
///    `setup_kind_register_banks` below). RPython's `registers_r` is a
///    pure kind-specific Ref bank — semantic locals go through
///    opcode-dispatch reads of `registers_r[color]`. Pyre's dual-role
///    is what makes Adaptation 1 load-bearing; retiring one entails
///    retiring both.
///
/// 3. Ref constants land via `ctx.const_ref(val)` — a plain Ref
///    constant — instead of upstream's `ConstPtrJitCode` subclass
///    (`history.py:361-365`) that carries an `opencoder_index = -1`
///    cache field. Upstream uses that field to fast-path the trace
///    encoder's pointer dedup at `opencoder.py:583-601
///    _cached_const_ptr` (over its `_refs_dict`). Pyre lacks the
///    per-instance cache: every `ctx.const_ref(val)` mints a fresh
///    constant OpRef (`constant_pool.rs:57-103
///    get_or_insert{,_typed}` — no HashMap-keyed dedup at this
///    layer). The two upstream dedup layers exist independently in
///    pyre:
///      - **Trace encoding** (port of `opencoder.py:583
///        _cached_const_ptr`): `opencoder.rs:1873 _encode_ptr` dedups
///        by address via `_refs_dict: HashMap<u64, u32>`, mirroring
///        upstream's `_refs_dict` lookup. The
///        `ConstPtrJitCode.opencoder_index` cache that bypasses the
///        dict has no pyre counterpart, but the underlying dedup is
///        preserved.
///      - **Resume-data numbering** (port of `resume.py:148-181
///        ResumeDataLoopMemo.large_ints / .refs`):
///        `resume.rs:3460-3490 ResumeDataLoopMemo.large_ints /
///        .refs` memo separately dedups constants when assigning
///        resume numbering tags.
fn copy_constants<F>(registers: &mut Vec<OpRef>, constants: &[i64], targetindex: usize, mut mint: F)
where
    F: FnMut(i64) -> OpRef,
{
    let num_regs_and_consts = targetindex + constants.len();
    if registers.len() < num_regs_and_consts {
        registers.resize(num_regs_and_consts, OpRef::NONE);
    }
    for (i, &val) in constants.iter().enumerate() {
        registers[targetindex + i] = mint(val);
    }
}

impl PyreSym {
    pub(crate) fn new_uninit(frame: OpRef) -> Self {
        Self {
            frame,
            execution_context: OpRef::NONE,
            symbolic_local_types: Vec::new(),
            symbolic_stack_types: Vec::new(),
            pending_next_instr: None,
            locals_cells_stack_array_ref: OpRef::NONE,
            valuestackdepth: 0,
            nlocals: 0,
            bridge_local_oprefs: None,
            bridge_local_types: None,
            vable_last_instr: OpRef::NONE,
            vable_pycode: OpRef::NONE,
            vable_valuestackdepth: OpRef::NONE,
            vable_debugdata: OpRef::NONE,
            vable_lastblock: OpRef::NONE,
            vable_w_globals: OpRef::NONE,
            vable_array_base: None,
            is_active_vable_owner: false,
            concrete_locals: Vec::new(),
            concrete_stack: Vec::new(),
            // jitcode and concrete_namespace initialized below
            jitcode: null_jitcode() as *const JitCode,
            concrete_namespace: std::ptr::null_mut(),
            is_function_entry_trace: false,
            concrete_execution_context: std::ptr::null(),
            concrete_vable_ptr: std::ptr::null_mut(),
            last_exc_value: std::ptr::null_mut(),
            class_of_last_exc_is_const: false,
            last_exc_box: OpRef::NONE,
            current_exc_value: pyre_interpreter::eval::get_current_exception(),
            current_exc_box: OpRef::NONE,
            virtualref_boxes: Vec::new(),
            // RPython pyjitpl.py:74-78 init: registers_X[i] = CONST_NULL for
            // i in num_regs. Sized lazily here — `setup_kind_register_banks`
            // resizes `registers_i` / `registers_f` once the owning JitCode is
            // bound. `registers_r` continues to be driven by the existing
            // semantic-slot logic; the encoder is not yet rewired to
            // per-bank reads for `registers_r`.
            registers_i: Vec::new(),
            registers_r: Vec::new(),
            registers_f: Vec::new(),
        }
    }

    /// `pyjitpl.py:74-90 MIFrame.setup` parity for the per-kind register
    /// files. Sizes `registers_i` and `registers_f` to
    /// `num_regs_and_consts_X`, matching the RPython MIFrame layout where
    /// each bank holds the post-regalloc-color register slots followed by
    /// the constant pool entries. The leading `num_regs_X` slots stay
    /// `OpRef::NONE` (RPython's `CONST_NULL` placeholder); the trailing
    /// `[num_regs_X..num_regs_and_consts_X)` slots are filled with the
    /// constant-pool OpRefs in `pyjitpl.py:97-119 copy_constants` order:
    ///   - `registers_i[num_regs_i + i]` ← `ctx.const_int(constants_i[i])`
    ///   - `registers_r[num_regs_r + i]` ← `ctx.const_ref(constants_r[i])`
    ///   - `registers_f[num_regs_f + i]` ← `ctx.const_float(constants_f[i])`
    /// `TraceCtx::const_int` / `const_ref` / `const_float` all mint a
    /// fresh constant OpRef per call (`constant_pool.rs:57-103
    /// get_or_insert{,_typed}`), matching RPython
    /// `ConstClass(constants[i])`'s fresh-Box allocation
    /// (`history.py:220/261/307`). Per-value dedup, where it exists
    /// upstream, lives in the resume memo
    /// (`resume.rs:3460-3490 ResumeDataLoopMemo.large_ints / .refs`
    /// per `resume.py:148-181`), not in the constant pool. Re-entering
    /// this helper therefore overwrites the trailing slots with freshly
    /// minted OpRefs across all three banks, matching `copy_constants`
    /// overwrite semantics line-by-line.
    ///
    /// `registers_r` carries the unified locals + stack-tail abstract
    /// register file. The SSA-authoritative live_r work extends the resize to
    /// `registers_r` so its size matches the post-regalloc-color shape
    /// `num_regs_and_consts_r` already in use for `registers_i` /
    /// `registers_f`.
    ///
    /// This helper ports the full upstream `pyjitpl.py:74-90
    /// MIFrame.setup` body (resize + `copy_constants`).  Pyre still keeps
    /// `registers_r` as the semantic frame mirror for stack/local writes;
    /// guard capture materializes the post-regalloc-color Ref bank from
    /// that mirror or the virtualizable shadow.  The trailing constant
    /// slots `[num_regs_X, num_regs_and_consts_X)` are filled by this
    /// helper; no production reader consumes them yet (pending the
    /// encoder's constant-bank read path).
    ///
    /// Safe to call when `self.jitcode` points at the thread-local
    /// `null_jitcode()` placeholder — the skeleton's
    /// `num_regs_and_consts_X` values are zero and the constant pools
    /// are empty, which makes both the resize and the constant fill a
    /// no-op.
    pub(crate) fn setup_kind_register_banks(&mut self, ctx: &mut TraceCtx) {
        debug_assert!(!self.jitcode.is_null());
        let (num_regs_i, num_regs_r, num_regs_f, constants_i, constants_r, constants_f) = {
            let jc = unsafe { &*self.jitcode };
            let runtime_jc = &jc.payload.jitcode;
            (
                runtime_jc.num_regs_i() as usize,
                runtime_jc.num_regs_r() as usize,
                runtime_jc.num_regs_f() as usize,
                runtime_jc.constants_i.clone(),
                runtime_jc.constants_r.clone(),
                runtime_jc.constants_f.clone(),
            )
        };
        // pyjitpl.py:98-119 `MIFrame.copy_constants` line-by-line:
        //   num_regs_and_consts = targetindex + len(constants)
        //   if registers is None or len(registers) < num_regs_and_consts:
        //       registers = [missing] * num_regs_and_consts
        //   for i in range(len(constants)):
        //       registers[targetindex] = ConstClass(constants[i])
        //       targetindex += 1
        // The "translation-time-only debug wipe" arm
        // (`elif not we_are_translated()`) has no pyre analogue.
        copy_constants(&mut self.registers_i, &constants_i, num_regs_i, |v| {
            ctx.const_int(v)
        });
        copy_constants(&mut self.registers_r, &constants_r, num_regs_r, |v| {
            ctx.const_ref(v)
        });
        copy_constants(&mut self.registers_f, &constants_f, num_regs_f, |v| {
            ctx.const_float(v)
        });
    }

    /// True when this frame is allowed to mirror writes into the
    /// metainterp-scope `TraceCtx::virtualizable_boxes` cache (RPython
    /// `metainterp.virtualizable_boxes`, pyjitpl.py:1230). Two disjoint
    /// states satisfy the predicate:
    ///
    ///   1. **portal entry** — `is_active_vable_owner == true` after
    ///      `become_active_vable_owner` (which wraps the macro-generated
    ///      `init_vable_indices`). The frame's `locals_cells_stack_w`
    ///      array IS the active virtualizable.
    ///   2. **bridge entry** — `bridge_local_oprefs == Some(...)` after
    ///      `setup_bridge_sym` calls `seed_virtualizable_boxes` to
    ///      repopulate the shadow from resume data. `is_active_vable_owner`
    ///      is cleared (`clear_active_vable`) because the bridge's
    ///      inputarg layout lacks the `[frame, last_instr, pycode,
    ///      valuestackdepth, debugdata, lastblock, w_globals]` scalar
    ///      header that `init_vable_indices` assumes (see
    ///      `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS` for the
    ///      canonical 6-scalar layout — line-by-line PyPy parity with
    ///      `interp_jit.py:25-31`); the frame still owns the shadow
    ///      semantically though.
    ///
    /// Callee inline frames (`inline_function_call` allocates a fresh
    /// `PyreSym::new_uninit`) keep both fields at their defaults and
    /// must NOT mirror into the caller's shadow — their
    /// `nlocals + stack_idx` space is the callee's own, not the
    /// caller's, so writing to `NUM_VABLE_SCALARS + reg_idx` in the
    /// shared `TraceCtx` shadow would corrupt the caller's portal
    /// layout. opencoder.py:718 `_list_of_boxes_virtualizable` treats
    /// `self.virtualizable_boxes` as the single source of truth; this
    /// predicate names the set of syms that are allowed to update it.
    pub(crate) fn owns_virtualizable_shadow(&self) -> bool {
        self.is_active_vable_owner || self.bridge_local_oprefs.is_some()
    }

    /// Promote this frame to active virtualizable owner. Wraps the
    /// macro-generated `init_vable_indices` so the per-frame
    /// `is_active_vable_owner` boolean and the u32 OpRef-offset stay in
    /// lock-step. Call this everywhere `init_vable_indices()` was
    /// previously invoked directly.
    pub(crate) fn become_active_vable_owner(&mut self) {
        self.init_vable_indices(crate::virtualizable_gen::FIRST_VABLE_SCALAR_IDX);
        self.is_active_vable_owner = true;
        debug_assert!(
            self.vable_array_base.is_some(),
            "init_vable_indices must seed vable_array_base = Some(...)"
        );
    }

    /// Demote this frame from active virtualizable owner. Used at bridge
    /// setup (`setup_bridge_sym`) where the bridge's inputarg layout
    /// does not have the `[frame, last_instr, pycode, valuestackdepth,
    /// debugdata, lastblock, w_globals]` scalar header that the
    /// loop-portal `init_vable_indices` assumes (canonical 6-scalar
    /// layout in `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS`);
    /// subsequent reads consult `bridge_local_oprefs` or fall through
    /// to the heap array via `locals_cells_stack_array_ref`.
    pub(crate) fn clear_active_vable(&mut self) {
        self.vable_array_base = None;
        self.is_active_vable_owner = false;
    }

    #[doc(hidden)]
    pub fn from_test_state(state: TestSymState) -> Self {
        let mut sym = Self::new_uninit(state.frame);
        sym.jitcode = state.jitcode as *const JitCode;
        sym.nlocals = state.nlocals;
        sym.valuestackdepth = state.valuestackdepth;
        sym.locals_cells_stack_array_ref = state.locals_cells_stack_array_ref;
        sym.symbolic_local_types = state.symbolic_local_types;
        sym.symbolic_stack_types = state.symbolic_stack_types;
        sym.registers_r = state.registers_r;
        sym.concrete_stack = state.concrete_stack;
        sym.concrete_namespace = state.concrete_namespace;
        sym.vable_last_instr = state.vable_last_instr;
        sym.vable_pycode = state.vable_pycode;
        sym.vable_valuestackdepth = state.vable_valuestackdepth;
        sym.vable_debugdata = state.vable_debugdata;
        sym.vable_lastblock = state.vable_lastblock;
        sym.vable_w_globals = state.vable_w_globals;
        sym
    }

    #[doc(hidden)]
    pub fn set_test_execution_context(&mut self, execution_context: OpRef) {
        self.execution_context = execution_context;
    }

    /// Initialize symbolic tracking state. Called once when the owning
    /// MetaInterpFrame is pushed (trace.rs for root frame). Callee (inline)
    /// frames set symbolic state manually in perform_call
    /// (trace_opcode.rs:3323-3424) and do NOT call this.
    pub(crate) fn init_symbolic(&mut self, ctx: &mut TraceCtx, concrete_frame: usize) {
        self.is_function_entry_trace = ctx.header_pc == 0;
        let nlocals = concrete_nlocals(concrete_frame).unwrap_or(0);
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][init-sym] concrete_frame={:#x} nlocals={} vable_base={:?} header_pc={} func_entry={}",
                concrete_frame,
                nlocals,
                self.vable_array_base,
                ctx.header_pc,
                self.is_function_entry_trace
            );
        }
        let valuestackdepth = concrete_stack_depth(concrete_frame).unwrap_or(nlocals);
        let stack_only_depth = valuestackdepth.saturating_sub(nlocals);
        self.nlocals = nlocals;
        self.locals_cells_stack_array_ref = if self.is_active_vable_owner {
            OpRef::NONE
        } else {
            frame_locals_cells_stack_array(ctx, self.frame)
        };
        // RPython pyjitpl.py:74-78 init analogue for the local-slot view
        // of registers_r. The bridge override / vable inputarg / NONE
        // shape is the per-trace seed; subsequent load_local_value /
        // store_local_value updates the per-color slot directly.
        self.registers_r = if let Some(ref overrides) = self.bridge_local_oprefs {
            // resume.py:1042 parity: bridge trace uses OpRefs derived from
            // rebuild_from_resumedata (Box(n) → bridge InputArg OpRef::from_raw(n)).
            let mut locals = overrides.clone();
            locals.resize(nlocals, OpRef::NONE);
            locals
        } else if let Some(base) = self.vable_array_base {
            (0..nlocals)
                .map(|i| OpRef::input_arg_ref(base + i as u32))
                .collect()
        } else {
            vec![OpRef::NONE; nlocals]
        };
        // RPython resume.py:1042 parity: bridge traces enter with the
        // failing guard's saved boxes, NOT with the loop's full
        // virtualizable inputarg layout. Each `bridge_local_oprefs[i]`
        // points at the bridge inputarg slot the rebuilt frame placed in
        // local i, so the type for local i must come from
        // `inputarg_types[bridge_local_oprefs[i].0]` instead of the
        // loop's `vable_array_base + i` indexing (which only applies to
        // loops where every local sits in a fixed virtualizable slot).
        //
        // Loop / function-entry traces still use vable_array_base because
        // their inputarg list is the full vable layout.
        let inputarg_slot_types = if let Some(ref overrides) = self.bridge_local_oprefs {
            let inputarg_types = ctx.inputarg_types();
            let locals: Vec<Type> = (0..nlocals)
                .map(|i| {
                    overrides
                        .get(i)
                        .and_then(|opref| {
                            if opref.is_none() || opref.is_constant() {
                                None
                            } else {
                                inputarg_types.get(opref.raw() as usize).copied()
                            }
                        })
                        .unwrap_or(Type::Ref)
                })
                .collect();
            // resume.py:1042 consume_boxes + pyjitpl.py:2899 parity:
            // guard-failure resume reconstructs the CURRENT frame state,
            // including the live Python stack. `bridge_local_oprefs`
            // overrides only the locals slice; stack slots still follow
            // the virtualizable Ref contract until setup_bridge_sym
            // overwrites the rebuilt tail with the precise resume-data
            // OpRefs.
            Some((locals, vec![Type::Ref; stack_only_depth]))
        } else {
            self.vable_array_base.map(|base| {
                let inputarg_types = ctx.inputarg_types();
                let locals: Vec<Type> = (0..nlocals)
                    .map(|i| {
                        inputarg_types
                            .get(base as usize + i)
                            .copied()
                            .unwrap_or(Type::Ref)
                    })
                    .collect();
                let stack: Vec<Type> = (0..stack_only_depth)
                    .map(|i| {
                        inputarg_types
                            .get(base as usize + nlocals + i)
                            .copied()
                            .unwrap_or(Type::Ref)
                    })
                    .collect();
                (locals, stack)
            })
        };
        if self.is_function_entry_trace {
            // virtualizable.py:86 read_boxes() parity: all array items
            // are GC pointers → Ref. No pre-unboxing at function entry.
            self.symbolic_local_types = vec![Type::Ref; nlocals];
        } else if let Some(ref overrides) = self.bridge_local_types {
            // virtualizable.py:44: bridge locals keep the Ref contract of
            // the virtualizable array. `bridge_local_types` is always
            // all-Ref (populated by setup_bridge_sym); the override is
            // retained only for length parity with bridge_local_oprefs.
            let mut types = overrides.clone();
            types.resize(nlocals, Type::Ref);
            self.symbolic_local_types = types;
        } else if let Some((ref local_types, _)) = inputarg_slot_types {
            // warmstate.py:73 wrap(): inputargs resolved from the JIT
            // contract retain whatever declared type the input signature
            // records. For virtualizable array inputargs this is Ref.
            self.symbolic_local_types = local_types.clone();
        } else if self.symbolic_local_types.len() != nlocals {
            self.symbolic_local_types = concrete_slot_types(concrete_frame, nlocals, nlocals);
        }
        // Seed the stack portion of `registers_r`
        // directly. `registers_r` is the unified abstract register
        // file — locals occupy `[..nlocals]` and stack slots occupy
        // `[nlocals..nlocals + stack_only_depth]` (RPython
        // `pyjitpl.py:70-78` MIFrame parity).
        let stack_seed: Vec<OpRef> = if let Some(base) = self.vable_array_base {
            let stack_base = base + nlocals as u32;
            (0..stack_only_depth)
                .map(|i| OpRef::input_arg_ref(stack_base + i as u32))
                .collect()
        } else {
            vec![OpRef::NONE; stack_only_depth]
        };
        self.registers_r.extend(stack_seed.iter().copied());
        if let Some((_, ref stack_types)) = inputarg_slot_types {
            self.symbolic_stack_types = stack_types.clone();
        } else if self.symbolic_stack_types.len() != stack_only_depth {
            self.symbolic_stack_types =
                concrete_slot_types(concrete_frame, nlocals, valuestackdepth)
                    .into_iter()
                    .skip(nlocals)
                    .collect();
        }
        self.pending_next_instr = None;
        self.valuestackdepth = valuestackdepth;
        // MIFrame concrete Box tracking: populate concrete value arrays
        // from the concrete frame snapshot (RPython MIFrame.setup_call parity).
        // Use concrete_value_from_slot to distinguish "real null pointer"
        // (Ref(PY_NULL)) from "untracked" (ConcreteValue::Null).
        self.concrete_locals = (0..nlocals)
            .map(|i| {
                let obj = concrete_stack_value(concrete_frame, i).unwrap_or(PY_NULL);
                concrete_value_from_slot(obj)
            })
            .collect();
        self.concrete_stack = (0..stack_only_depth)
            .map(|i| {
                let obj = concrete_stack_value(concrete_frame, nlocals + i).unwrap_or(PY_NULL);
                concrete_value_from_slot(obj)
            })
            .collect();
        // Extract frame metadata pointers for use without concrete_frame
        if concrete_frame != 0 {
            let frame = unsafe { &*(concrete_frame as *const pyre_interpreter::pyframe::PyFrame) };
            self.jitcode = jitcode_for(frame.pycode);
            self.concrete_namespace = frame.w_globals_obj;
            self.concrete_execution_context = frame.execution_context;
            self.concrete_vable_ptr = concrete_frame as *mut u8;
            // pyjitpl.py:74-90 MIFrame.setup parity for the per-kind banks
            // (including pyjitpl.py:97-119 copy_constants).
            self.setup_kind_register_banks(ctx);
        }
        // pyjitpl.py:3458-3462 / virtualizable.py:86-99 read_boxes parity:
        // seed the tracing-time `virtualizable_boxes` cache with the
        // InputArg OpRefs that correspond to the portal's virtualizable
        // layout (scalar fields followed by array items, with the
        // virtualizable identity appended at `boxes[-1]`). Without this
        // seeding, `vable_setarrayitem_indexed` / `vable_getarrayitem_*`
        // fall through to raw SetarrayitemGc / GetarrayitemGc ops, and
        // `close_loop_args_at`'s `set_virtualizable_box_at` mirror
        // becomes a no-op — the very reason `MIFrame::store_local_value`
        // cannot route through the standard vable path today.
        if let Some(base) = self.vable_array_base {
            // virtualizable.py:86-99 read_boxes iterates `len(lst)` — the
            // heap-side `locals_cells_stack_w` length, which is
            // `num_locals + num_cells + max_stack` (pyframe.py:107-110).
            // Match that shape so every interpreter-visible slot has a
            // tracing-time mirror. The live prefix [0, nlocals+stack_only_depth)
            // references the recorder's InputArg stream; reserved stack slots
            // beyond the current stack pointer are NULL on the PyFrame heap
            // (alloc_fixed_array_with_header(..., PY_NULL)) and the helper pads
            // them with a shared const-NULL OpRef.
            let num_vable_scalars = crate::virtualizable_gen::NUM_VABLE_SCALARS;
            let live_prefix = nlocals + stack_only_depth;
            let array_len = concrete_frame_array_len(concrete_frame).unwrap_or(live_prefix);
            // pyjitpl.py:3302 initialize_virtualizable parity: the concrete
            // half of virtualizable_boxes at portal entry comes from a live
            // heap read (vinfo.read_boxes(cpu, virtualizable, 0)). There is
            // no resume-data stream at root-trace start.
            let info = crate::frame_layout::build_pyframe_virtualizable_info();
            // Static fields inputargs at FIRST_VABLE_SCALAR_IDX..+NUM_VABLE_SCALARS.
            // virtualizable_boxes carries vable static fields only — non-vable
            // extra reds (e.g. `ec`) sit between frame and vable scalars in the
            // inputarg space (`pyjitpl.py:2957 redboxes` then `:2964
            // + virtualizable_boxes`) and never enter the shadow. Per-slot
            // type follows `info.static_fields[i].field_type` so the OpRef
            // variant matches RPython's BoxInt/BoxRef inputarg classes.
            let first = crate::virtualizable_gen::FIRST_VABLE_SCALAR_IDX;
            let scalar_oprefs: Vec<OpRef> = (0..num_vable_scalars)
                .map(|i| {
                    let pos = first + i as u32;
                    let tp = info.static_fields[i].field_type;
                    OpRef::input_arg_typed(pos, tp)
                })
                .collect();
            // Array items inputargs at base..base + live_prefix. Items are
            // W_Root (Ref) per `array_item_type = Ref` in virtualizable_gen.
            let array_items: Vec<OpRef> = (0..live_prefix)
                .map(|i| OpRef::input_arg_ref(base + i as u32))
                .collect();
            // SYM_FRAME_IDX is the virtualizable identity (Ref).
            let vable_ref = OpRef::input_arg_ref(crate::virtualizable_gen::SYM_FRAME_IDX);
            let array_lengths = [array_len];
            let (input_values, vable_ref_value) = if concrete_frame != 0 {
                let (static_boxes, array_boxes) =
                    unsafe { info.read_all_boxes(concrete_frame as *const u8, &array_lengths) };
                let mut values = Vec::with_capacity(num_vable_scalars + array_len);
                for (i, bits) in static_boxes.iter().enumerate() {
                    values.push(value_for_slot(info.static_fields[i].field_type, *bits));
                }
                for (a, items) in array_boxes.iter().enumerate() {
                    let item_ty = info.array_fields[a].item_type;
                    for bits in items {
                        values.push(value_for_slot(item_ty, *bits));
                    }
                }
                while values.len() < num_vable_scalars + array_len {
                    values.push(majit_ir::Value::Ref(majit_ir::GcRef::NULL));
                }
                (
                    values,
                    majit_ir::Value::Ref(majit_ir::GcRef(concrete_frame)),
                )
            } else {
                (Vec::new(), majit_ir::Value::Void)
            };
            crate::state::seed_virtualizable_boxes(
                ctx,
                vable_ref,
                vable_ref_value,
                &scalar_oprefs,
                &array_items,
                array_len,
                &input_values,
                concrete_frame as *const u8,
            );
        }
    }

    /// Read a concrete value from the Box arrays using an absolute
    /// unified-array index (0..nlocals = locals, nlocals.. = stack).
    pub(crate) fn concrete_value_at_opt(&self, abs_idx: usize) -> Option<ConcreteValue> {
        if abs_idx < self.nlocals {
            self.concrete_locals.get(abs_idx).copied()
        } else {
            let stack_idx = abs_idx - self.nlocals;
            self.concrete_stack.get(stack_idx).copied()
        }
    }

    /// Read a concrete value from the Box arrays using an absolute
    /// unified-array index (0..nlocals = locals, nlocals.. = stack).
    pub(crate) fn concrete_value_at(&self, abs_idx: usize) -> ConcreteValue {
        self.concrete_value_at_opt(abs_idx)
            .unwrap_or(ConcreteValue::Null)
    }
}

/// pyjitpl.py:1789-1814 opimpl_virtual_ref parity.
/// Creates a concrete JitVirtualRef via virtual_ref_during_tracing(),
/// records VIRTUAL_REF(box, cindex), and pushes
/// [virtualbox, vrefbox] onto virtualref_boxes.
///
/// Called from metainterp push_inline_frame (executioncontext.enter parity).
pub(crate) fn opimpl_virtual_ref(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    virtual_obj: OpRef,
    virtual_obj_ptr: usize,
) -> OpRef {
    // pyjitpl.py:1804: virtual_ref_during_tracing(virtual_obj)
    let vref_info = majit_metainterp::virtualref::VirtualRefInfo::new();
    let vref_ptr = vref_info.virtual_ref_during_tracing(virtual_obj_ptr as *mut u8);
    // pyjitpl.py:1805: cindex = ConstInt(len(virtualref_boxes) // 2)
    let cindex = ctx.const_int((sym.virtualref_boxes.len() / 2) as i64);
    // pyjitpl.py:1806: record VIRTUAL_REF(box, cindex)
    let vref = ctx.record_op(OpCode::VirtualRefR, &[virtual_obj, cindex]);
    // pyjitpl.py:1807: heapcache.new(resbox)
    ctx.heap_cache_mut().new_box(vref);
    // pyjitpl.py:1814: virtualref_boxes += [virtualbox, vrefbox]
    sym.virtualref_boxes.push((virtual_obj, virtual_obj_ptr));
    sym.virtualref_boxes.push((vref, vref_ptr as usize));
    vref
}

/// pyjitpl.py:1819-1831 opimpl_virtual_ref_finish parity.
/// Pops vrefbox and lastbox from virtualref_boxes (LIFO),
/// asserts `box == lastbox`, records VIRTUAL_REF_FINISH if still virtual.
///
/// Called from metainterp finishframe_inline/exception (executioncontext.leave parity).
pub(crate) fn opimpl_virtual_ref_finish(ctx: &mut TraceCtx, sym: &mut PyreSym, virtual_obj: OpRef) {
    if sym.virtualref_boxes.len() < 2 {
        return;
    }
    // pyjitpl.py:1821: vrefbox = virtualref_boxes.pop()
    let (vref_opref, vref_ptr) = sym.virtualref_boxes.pop().unwrap();
    // pyjitpl.py:1822: lastbox = virtualref_boxes.pop()
    let (lastbox_opref, _lastbox_ptr) = sym.virtualref_boxes.pop().unwrap();
    // pyjitpl.py:1823: assert box.getref_base() == lastbox.getref_base()
    debug_assert_eq!(
        virtual_obj, lastbox_opref,
        "opimpl_virtual_ref_finish: leaving frame box != top virtualref box"
    );
    // pyjitpl.py:1831: if is_virtual_ref(vref) → record VIRTUAL_REF_FINISH
    let vref_info = majit_metainterp::virtualref::VirtualRefInfo::new();
    let is_vref = vref_ptr != 0 && unsafe { vref_info.is_virtual_ref(vref_ptr as *const u8) };
    if is_vref {
        // pyjitpl.py:1832: VIRTUAL_REF_FINISH(vrefbox, nullbox)
        let null = ctx.const_ref(0);
        let _ = ctx.record_op(OpCode::VirtualRefFinish, &[vref_opref, null]);
    }
}

impl PyreJitState {
    /// Canonical PyPy portal driver layout from `interp_jit.py:67-74`.
    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_driver_descriptor() -> JitDriverStaticData {
        let mut descriptor = JitDriverStaticData::with_virtualizable(
            vec![
                ("next_instr", Type::Int),
                ("is_being_profiled", Type::Int),
                ("pycode", Type::Ref),
            ],
            vec![("frame", Type::Ref), ("ec", Type::Ref)],
            Some("frame"),
        );
        descriptor.is_recursive = true;
        descriptor
    }

    fn execution_context_as_usize(&self) -> usize {
        let Some(frame_ptr) = self.frame_ptr() else {
            return 0;
        };
        unsafe { (*(frame_ptr as *const PyFrame)).execution_context as usize }
    }

    fn expanded_virtualizable_live_values_with_extra_reds(
        &self,
        meta: &PyreMeta,
        extra_reds: &[Value],
    ) -> Vec<Value> {
        let base = crate::virtualizable_gen::virt_extract_live_values(
            self.frame,
            self.execution_context_as_usize(),
            self.last_instr_as_usize(),
            self.pycode_as_usize(),
            self.valuestackdepth(),
            self.debugdata_as_usize(),
            self.lastblock_as_usize(),
            self.w_globals_as_usize(),
            meta.num_locals,
            meta.valuestackdepth,
            |i| self.local_at(i).unwrap_or(PY_NULL) as usize,
            |i| self.stack_at(i).unwrap_or(PY_NULL) as usize,
        );
        let Some((&frame, tail)) = base.split_first() else {
            return extra_reds.to_vec();
        };
        let mut values = Vec::with_capacity(base.len() + extra_reds.len());
        values.push(frame);
        values.extend_from_slice(extra_reds);
        values.extend_from_slice(tail);
        values
    }

    fn expanded_virtualizable_live_value_types_with_extra_reds(
        meta: &PyreMeta,
        extra_red_types: &[Type],
    ) -> Vec<Type> {
        let base = crate::virtualizable_gen::virt_live_value_types(meta.slot_types.len());
        let Some((&frame_tp, tail)) = base.split_first() else {
            return extra_red_types.to_vec();
        };
        let mut types = Vec::with_capacity(base.len() + extra_red_types.len());
        types.push(frame_tp);
        types.extend_from_slice(extra_red_types);
        types.extend_from_slice(tail);
        types
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_live_values_with_ec(&self, meta: &PyreMeta) -> Vec<Value> {
        let _ = meta;
        vec![
            Value::Ref(majit_ir::GcRef(self.frame)),
            Value::Ref(majit_ir::GcRef(self.execution_context_as_usize())),
        ]
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_live_value_types_with_ec(meta: &PyreMeta) -> Vec<Type> {
        let _ = meta;
        vec![Type::Ref, Type::Ref]
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_collect_jump_args(sym: &PyreSym) -> Vec<OpRef> {
        let base = sym.vable_collect_jump_args();
        let Some((&frame, tail)) = base.split_first() else {
            unreachable!(
                "vable_collect_jump_args returned empty — virtualizable macro \
                 did not emit frame_field push (see majit-macros/.../derive.rs:381)"
            );
        };
        let mut args = Vec::with_capacity(base.len() + 1);
        args.push(frame);
        args.push(sym.execution_context);
        args.extend_from_slice(tail);
        args
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_collect_typed_jump_args(sym: &PyreSym) -> Vec<(OpRef, Type)> {
        let base = sym.vable_collect_typed_jump_args();
        let Some(&(frame, frame_tp)) = base.first() else {
            unreachable!(
                "vable_collect_typed_jump_args returned empty — virtualizable \
                 macro did not emit frame_field push \
                 (see majit-macros/.../derive.rs:389)"
            );
        };
        let mut args = Vec::with_capacity(base.len() + 1);
        args.push((frame, frame_tp));
        args.push((sym.execution_context, Type::Ref));
        args.extend_from_slice(&base[1..]);
        args
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_create_sym(meta: &PyreMeta, _header_pc: usize) -> PyreSym {
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_typed(0, Type::Ref));
        sym.execution_context = OpRef::input_arg_typed(1, Type::Ref);
        // `become_active_vable_owner` calls `init_vable_indices(FIRST_VABLE_SCALAR_IDX)`
        // where `FIRST_VABLE_SCALAR_IDX = 1 + NUM_EXTRA_REDS` already accounts for
        // the `ec` extra red. Do not re-shift the slot indices here — doing so
        // would double-count, leaving e.g. `vable_pycode` at the valuestackdepth
        // slot (Type::Int) instead of the pycode slot (Type::Ref) and tripping
        // the resume-time type assertion in `value_to_static_vable_bits`.
        sym.become_active_vable_owner();
        sym.nlocals = meta.num_locals;
        sym.valuestackdepth = meta.valuestackdepth;
        sym.symbolic_local_types = vec![Type::Ref; meta.num_locals];
        sym.symbolic_stack_types = vec![Type::Ref; meta.vable_stack_only_depth()];
        let stack_only = meta.vable_stack_only_depth();
        sym.concrete_stack = vec![ConcreteValue::Null; stack_only];
        sym
    }

    fn restore_expanded_virtualizable_values_with_extra_reds(
        &mut self,
        meta: &PyreMeta,
        values: &[Value],
        extra_reds: usize,
    ) {
        let Some(frame) = values.first() else {
            return;
        };
        self.frame = value_to_usize(frame);
        if values.len() <= 1 + extra_reds {
            return;
        }

        if meta.has_virtualizable {
            // next_instr is already synced to the PyFrame heap by the
            // compiled code's virtualizable sync before JUMP.
            self.set_valuestackdepth(meta.valuestackdepth);
            let nlocals = self.local_count();
            let stack_only = meta.valuestackdepth.saturating_sub(nlocals);
            let _ = extra_reds; // already folded into NUM_SCALAR_INPUTARGS
            let mut idx = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
            for local_idx in 0..nlocals {
                if let Some(value) = values.get(idx) {
                    let slot_type = meta.slot_types.get(local_idx).copied().unwrap_or(Type::Ref);
                    let _ =
                        self.set_local_at(local_idx, boxed_slot_value_for_type(slot_type, value));
                }
                idx += 1;
            }
            for i in 0..stack_only {
                if let Some(value) = values.get(idx) {
                    let slot_type = meta
                        .slot_types
                        .get(nlocals + i)
                        .copied()
                        .unwrap_or(Type::Ref);
                    let _ = self.set_stack_at(i, boxed_slot_value_for_type(slot_type, value));
                }
                idx += 1;
            }
        } else {
            let nlocals = self.local_count();
            let stack_only_depth = meta.valuestackdepth.saturating_sub(nlocals);
            let mut idx = 1 + extra_reds;
            for local_idx in 0..nlocals {
                let slot_type = meta.slot_types.get(local_idx).copied().unwrap_or(Type::Ref);
                let _ = self.set_local_at(
                    local_idx,
                    boxed_slot_value_for_type(slot_type, &values[idx]),
                );
                idx += 1;
            }
            for i in 0..stack_only_depth {
                let slot_type = meta
                    .slot_types
                    .get(nlocals + i)
                    .copied()
                    .unwrap_or(Type::Ref);
                let _ = self.set_stack_at(i, boxed_slot_value_for_type(slot_type, &values[idx]));
                idx += 1;
            }
            self.set_valuestackdepth(meta.valuestackdepth);
        }
    }

    /// virtualizable.py:126-137 write_from_resume_data_partial parity.
    ///
    /// Restore virtualizable frame state from raw output buffer values.
    /// interp_jit.py:25: locals_cells_stack_w ARRAYITEMTYPE = GCREF.
    /// All raw values are PyObjectRef pointers — write directly.
    pub fn restore_virtualizable_from_raw(&mut self, raw_values: &[i64]) -> bool {
        if raw_values.is_empty() {
            return false;
        }
        let mut idx = crate::virtualizable_gen::virt_restore_scalars_raw(self, raw_values);

        let nlocals = self.local_count();
        let stack_only = self.valuestackdepth().saturating_sub(nlocals);
        for local_idx in 0..nlocals {
            if idx < raw_values.len() {
                let _ = self.set_local_at(local_idx, raw_values[idx] as PyObjectRef);
            }
            idx += 1;
        }
        for stack_idx in 0..stack_only {
            if idx < raw_values.len() {
                let _ = self.set_stack_at(stack_idx, raw_values[idx] as PyObjectRef);
            }
            idx += 1;
        }
        true
    }

    /// Returns true if the optimizer virtualizable mechanism is active.
    fn has_virtualizable_info(&self) -> bool {
        // pyre always uses virtualizable (JitDriverStaticData::with_virtualizable)
        true
    }

    fn frame_ptr(&self) -> Option<*mut u8> {
        (self.frame != 0).then_some(self.frame as *mut u8)
    }

    fn frame_array(&self, offset: usize) -> Option<&pyre_object::FixedObjectArray> {
        let frame_ptr = self.frame_ptr()?;
        let arr_ptr =
            unsafe { *(frame_ptr.add(offset) as *const *const pyre_object::FixedObjectArray) };
        Some(unsafe { &*arr_ptr })
    }

    fn frame_array_mut(&mut self, offset: usize) -> Option<&mut pyre_object::FixedObjectArray> {
        let frame_ptr = self.frame_ptr()?;
        let arr_ptr =
            unsafe { *(frame_ptr.add(offset) as *const *mut pyre_object::FixedObjectArray) };
        Some(unsafe { &mut *arr_ptr })
    }

    fn read_frame_usize(&self, offset: usize) -> Option<usize> {
        let frame_ptr = self.frame_ptr()?;
        Some(unsafe { *(frame_ptr.add(offset) as *const usize) })
    }

    fn write_frame_usize(&mut self, offset: usize, value: usize) -> bool {
        let Some(frame_ptr) = self.frame_ptr() else {
            return false;
        };
        unsafe {
            *(frame_ptr.add(offset) as *mut usize) = value;
        }
        true
    }

    fn locals_cells_stack_array(&self) -> Option<&pyre_object::FixedObjectArray> {
        self.frame_array(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
    }

    fn locals_cells_stack_array_mut(&mut self) -> Option<&mut pyre_object::FixedObjectArray> {
        self.frame_array_mut(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
    }

    fn namespace_len(&self) -> usize {
        let Some(frame_ptr) = self.frame_ptr() else {
            return 0;
        };
        let w_globals_obj = unsafe {
            *(frame_ptr.add(PYFRAME_W_GLOBALS_OBJ_OFFSET) as *const pyre_object::PyObjectRef)
        };
        if w_globals_obj.is_null() {
            return 0;
        }
        // `dictmultiobject.py:107-109 W_DictMultiObject.length`. The common
        // module-dict case reads `ModuleDictStorage` directly (O(1), no
        // `dict_storage_proxy` reconciliation) — this guard is on the
        // per-portal-entry path. Plain dict globals (exec/eval) fall back to
        // the polymorphic strategy length.
        unsafe {
            pyre_object::dictmultiobject::module_dict_storage_len(w_globals_obj)
                .unwrap_or_else(|| pyre_object::dictmultiobject::w_dict_len(w_globals_obj))
        }
    }

    fn restore_single_frame(&mut self, meta: &PyreMeta, values: &[i64]) {
        let Some(&frame) = values.first() else {
            return;
        };
        self.frame = frame as usize;
        if values.len() == 1 {
            return;
        }
        if meta.has_virtualizable {
            self.restore_virtualizable_i64(values);
        } else {
            let nlocals = self.local_count();
            let stack_only = self.valuestackdepth().saturating_sub(nlocals);
            let mut idx = 1;
            for local_idx in 0..nlocals {
                if idx < values.len() {
                    let slot_type = meta.slot_types.get(local_idx).copied().unwrap_or(Type::Ref);
                    let _ = self
                        .set_local_at(local_idx, boxed_slot_i64_for_type(slot_type, values[idx]));
                }
                idx += 1;
            }
            for i in 0..stack_only {
                if idx < values.len() {
                    let slot_type = meta
                        .slot_types
                        .get(nlocals + i)
                        .copied()
                        .unwrap_or(Type::Ref);
                    let _ = self.set_stack_at(i, boxed_slot_i64_for_type(slot_type, values[idx]));
                }
                idx += 1;
            }
        }
    }

    pub fn local_at(&self, idx: usize) -> Option<PyObjectRef> {
        self.locals_cells_stack_array()
            .and_then(|arr| arr.as_slice().get(idx).copied())
    }

    /// Number of local variable slots.
    pub fn local_count(&self) -> usize {
        concrete_nlocals(self.frame).unwrap_or(0)
    }

    pub fn set_local_at(&mut self, idx: usize, value: PyObjectRef) -> bool {
        let Some(arr) = self.locals_cells_stack_array_mut() else {
            return false;
        };
        let Some(slot) = arr.as_mut_slice().get_mut(idx) else {
            return false;
        };
        *slot = value;
        true
    }

    /// Read a stack value at stack-relative index `idx` (0-based from stack bottom).
    pub fn stack_at(&self, idx: usize) -> Option<PyObjectRef> {
        let nlocals = self.local_count();
        self.locals_cells_stack_array()
            .and_then(|arr| arr.as_slice().get(nlocals + idx).copied())
    }

    /// Total capacity of the unified array.
    pub fn array_capacity(&self) -> usize {
        self.locals_cells_stack_array()
            .map(pyre_object::FixedObjectArray::len)
            .unwrap_or(0)
    }

    /// Set a stack value at stack-relative index `idx`.
    pub fn set_stack_at(&mut self, idx: usize, value: PyObjectRef) -> bool {
        let nlocals = self.local_count();
        let Some(arr) = self.locals_cells_stack_array_mut() else {
            return false;
        };
        let Some(slot) = arr.as_mut_slice().get_mut(nlocals + idx) else {
            return false;
        };
        *slot = value;
        true
    }

    // ── Heap accessors: single source of truth (RPython parity) ──
    // RPython's virtualizable IS the heap object — getattr/setattr go
    // directly to the heap.  These accessors do the same via frame_ptr.

    pub fn last_instr_as_usize(&self) -> usize {
        let frame_ptr = self
            .frame_ptr()
            .expect("PyreJitState.frame must point to a valid PyFrame");
        unsafe { (*(frame_ptr as *const PyFrame)).last_instr as usize }
    }

    pub fn set_last_instr(&mut self, value: usize) {
        let frame_ptr = self
            .frame_ptr()
            .expect("PyreJitState.frame must point to a valid PyFrame");
        unsafe {
            (*(frame_ptr as *mut PyFrame)).last_instr = value as isize;
        }
    }

    pub fn next_instr(&self) -> usize {
        let frame_ptr = self
            .frame_ptr()
            .expect("PyreJitState.frame must point to a valid PyFrame");
        unsafe { (&*(frame_ptr as *const PyFrame)).next_instr() }
    }

    pub fn set_next_instr(&mut self, value: usize) {
        let frame_ptr = self
            .frame_ptr()
            .expect("PyreJitState.frame must point to a valid PyFrame");
        unsafe {
            (&mut *(frame_ptr as *mut PyFrame)).set_last_instr_from_next_instr(value);
        }
    }

    pub fn valuestackdepth(&self) -> usize {
        self.read_frame_usize(PYFRAME_VALUESTACKDEPTH_OFFSET)
            .expect("PyreJitState.frame must point to a valid PyFrame")
    }

    pub fn set_valuestackdepth(&mut self, value: usize) {
        assert!(
            self.write_frame_usize(PYFRAME_VALUESTACKDEPTH_OFFSET, value),
            "PyreJitState.frame must point to a valid PyFrame"
        );
    }

    /// Read the code pointer (pycode) from the heap frame.
    pub fn pycode_as_usize(&self) -> usize {
        self.read_frame_usize(PYFRAME_PYCODE_OFFSET)
            .expect("PyreJitState.frame must point to a valid PyFrame")
    }

    /// Read the w_globals_obj pointer from the heap frame.
    pub fn w_globals_as_usize(&self) -> usize {
        self.read_frame_usize(PYFRAME_W_GLOBALS_OBJ_OFFSET)
            .expect("PyreJitState.frame must point to a valid PyFrame")
    }

    /// Read the execution context pointer from the heap frame.
    ///
    /// `interp_jit.py:67 reds = ['frame', 'ec']`: ec is a non-vable red
    /// inputarg in RPython. pyre's PyFrame carries it inline at
    /// `execution_context`, so this accessor derefs the heap; from the
    /// macro-generated layout's perspective ec sits at SYM_EC_IDX between
    /// the frame pointer and the vable scalar block (`pyjitpl.py:2957
    /// redboxes` then `:2964 + virtualizable_boxes`).
    pub fn ec_as_usize(&self) -> usize {
        self.read_frame_usize(crate::frame_layout::PYFRAME_EXECUTION_CONTEXT_OFFSET)
            .expect("PyreJitState.frame must point to a valid PyFrame")
    }

    /// Write the execution context pointer into the heap frame.
    ///
    /// Called by `virt_restore_scalars` when reconstructing red inputargs
    /// from a guard-failure resume vector.
    pub fn set_ec(&mut self, value: usize) {
        assert!(
            self.write_frame_usize(crate::frame_layout::PYFRAME_EXECUTION_CONTEXT_OFFSET, value),
            "PyreJitState.frame must point to a valid PyFrame"
        );
    }

    /// Read the code pointer (pycode) from the heap frame.
    pub fn code_as_usize(&self) -> usize {
        self.pycode_as_usize()
    }

    /// Read the namespace pointer from the heap frame.
    pub fn namespace_as_usize(&self) -> usize {
        self.w_globals_as_usize()
    }

    /// Write the pycode pointer to the heap frame.
    /// virtualizable.py:101-107 write_boxes: ALL static fields written.
    pub fn set_pycode(&mut self, value: usize) {
        assert!(
            self.write_frame_usize(PYFRAME_PYCODE_OFFSET, value),
            "PyreJitState.frame must point to a valid PyFrame"
        );
    }

    /// Write the w_globals_obj pointer to the heap frame.
    pub fn set_w_globals(&mut self, value: usize) {
        assert!(
            self.write_frame_usize(PYFRAME_W_GLOBALS_OBJ_OFFSET, value),
            "PyreJitState.frame must point to a valid PyFrame"
        );
    }

    /// Compatibility wrapper for older callers that still speak in
    /// terms of `code` / `namespace`.
    pub fn set_code(&mut self, value: usize) {
        self.set_pycode(value);
    }

    pub fn set_namespace(&mut self, value: usize) {
        self.set_w_globals(value);
    }

    /// pyframe.py:82 debugdata — read from heap frame.
    pub fn debugdata_as_usize(&self) -> usize {
        self.read_frame_usize(PYFRAME_DEBUGDATA_OFFSET).unwrap_or(0)
    }

    /// pyframe.py:82 debugdata — write to heap frame.
    pub fn set_debugdata(&mut self, value: usize) {
        let _ = self.write_frame_usize(PYFRAME_DEBUGDATA_OFFSET, value);
    }

    /// pyframe.py:86 lastblock — read from heap frame.
    pub fn lastblock_as_usize(&self) -> usize {
        self.read_frame_usize(PYFRAME_LASTBLOCK_OFFSET).unwrap_or(0)
    }

    /// pyframe.py:86 lastblock — write to heap frame.
    pub fn set_lastblock(&mut self, value: usize) {
        let _ = self.write_frame_usize(PYFRAME_LASTBLOCK_OFFSET, value);
    }

    /// Validate that the frame pointer is usable (fields readable, array present).
    fn validate_frame(&self) -> bool {
        self.frame_ptr().is_some()
            && self
                .read_frame_usize(PYFRAME_VALUESTACKDEPTH_OFFSET)
                .is_some()
            && self.locals_cells_stack_array().is_some()
    }

    /// virtualizable.py:126-137 write_from_resume_data_partial parity.
    ///
    /// Restores virtualizable array slots from the fail_args layout:
    ///   [frame, scalars..., active_locals..., active_stack...]
    ///
    /// interp_jit.py:25 declares locals_cells_stack_w as a single
    /// virtualizable array with uniform ARRAYITEMTYPE = GCREF.
    /// resume.py:1408 calls write_from_resume_data_partial which loops
    /// over the array calling reader.load_next_value_of_type(GCREF),
    /// i.e. reader.next_ref() — every slot is a GCREF pointer.
    ///
    /// The raw i64 values here are already PyObjectRef pointers from
    /// the backend's Ref register bank. Write them directly without
    /// per-slot type dispatch.
    fn restore_virtualizable_i64(&mut self, values: &[i64]) {
        let mut idx = crate::virtualizable_gen::virt_restore_scalars_raw(self, values);

        // virtualizable.py:134-137:
        //   for ARRAYITEMTYPE, fieldname in unroll_array_fields:
        //       lst = getattr(virtualizable, fieldname)
        //       for j in range(len(lst)):
        //           lst[j] = reader.load_next_value_of_type(ARRAYITEMTYPE)
        // ARRAYITEMTYPE is always GCREF for locals_cells_stack_w.
        let nlocals = self.local_count();
        for i in 0..nlocals {
            if idx < values.len() {
                let _ = self.set_local_at(i, values[idx] as PyObjectRef);
            }
            idx += 1;
        }

        let stack_only = self.valuestackdepth().saturating_sub(nlocals);
        for i in 0..stack_only {
            if idx < values.len() {
                let _ = self.set_stack_at(i, values[idx] as PyObjectRef);
            }
            idx += 1;
        }
    }

    fn import_virtualizable_state(
        &mut self,
        static_boxes: &[i64],
        array_boxes: &[Vec<i64>],
    ) -> bool {
        // virtualizable.py:126-137 write_from_resume_data_partial parity:
        // write ALL static fields to heap via VirtualizableInfo.
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        if !self.virt_import_static_boxes(&info, static_boxes) {
            return false;
        }

        // virtualizable.py:134-137: write array items to heap.
        // Validate array structure matches VirtualizableInfo.
        if array_boxes.len() != info.array_fields.len() {
            return false;
        }
        let Some(unified) = array_boxes.first() else {
            return info.array_fields.is_empty();
        };
        let Some(frame_arr) = self.locals_cells_stack_array_mut() else {
            return false;
        };
        if frame_arr.len() != unified.len() {
            return false;
        }
        for (dst, &src) in frame_arr.as_mut_slice().iter_mut().zip(unified) {
            *dst = src as PyObjectRef;
        }
        true
    }

    fn export_virtualizable_state(&self) -> (Vec<i64>, Vec<Vec<i64>>) {
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        self.virt_export_all(&info)
    }

    pub fn sync_from_virtualizable(&mut self, info: &VirtualizableInfo) -> bool {
        let _ = info;
        // Heap IS the source of truth. Just validate the frame is usable.
        self.validate_frame()
    }

    pub fn sync_to_virtualizable(&self, info: &VirtualizableInfo) -> bool {
        let Some(frame_ptr) = self.frame_ptr() else {
            return false;
        };
        // Heap is the single source of truth — no state-backed fields to
        // flush.  Only the vable_token needs resetting (virtualizable.py:218
        // force_now: set vable_token to TOKEN_NONE).
        unsafe {
            info.reset_vable_token(frame_ptr);
        }
        true
    }
}

/// resume.py:945-956 getvirtual_ptr parity, trace-time variant.
///
/// `materialize_virtual_from_rd` (eval.rs) does the same job at *runtime*
/// for the blackhole resume path: walks the `RdVirtualInfo` for `vidx`
/// and allocates a real heap object. This function does the same walk
/// at *trace* time, emitting `NEW_WITH_VTABLE` + `SETFIELD_GC` ops into
/// the bridge's trace via `ctx`. Returns the OpRef of the materialized
/// virtual.
///
/// Mirrors RPython's `ResumeDataBoxReader.consume_boxes` →
/// `rd_virtuals[i].allocate(decoder, i)` where `decoder.allocate_with_vtable`
/// is `metainterp.execute_new_with_vtable` (resume.py:1111-1112). The
/// recorded ops appear at the start of the bridge trace, before any
/// python interpreter opcodes are recorded — so when the bridge tracer
/// encounters the first `LOAD_FAST` of a previously-virtual local, it
/// sees the materialized OpRef in `bridge_local_oprefs` instead of
/// falling through to a stale vable-array read.
///
/// resume.py:1143-1188 shared oopspec-call emitter for the four
/// concat/slice materializers. Looks up the call info via
/// `ctx.callinfocollection` and emits a `CALL_R(func, args...)` with the
/// matching calldescr. Panics if `callinfocollection` is not attached to
/// the TraceCtx (should never happen once pyjitpl wiring is complete).
fn emit_stroruni_oopspec_call(
    ctx: &mut majit_metainterp::TraceCtx,
    oopspec: majit_ir::effectinfo::OopSpecIndex,
    args: &[OpRef],
) -> OpRef {
    let cic = ctx
        .callinfocollection
        .as_ref()
        .expect(
            "TraceCtx.callinfocollection missing — bridge-virtual VStr/VUni \
             Concat/Slice materialization requires pyjitpl to populate it \
             (resume.py:1143-1188)",
        )
        .clone();
    let (calldescr, func) = cic.callinfo_for_oopspec(oopspec);
    let calldescr = calldescr.expect("callinfo_for_oopspec missing entry for VStr/VUni oopspec");
    let func_const = ctx.const_int(func as i64);
    let mut call_args = Vec::with_capacity(1 + args.len());
    call_args.push(func_const);
    call_args.extend_from_slice(args);
    ctx.record_op_with_descr(majit_ir::OpCode::CallR, &call_args, calldescr.clone())
}

struct BridgeVirtualCache {
    virtuals_ptr_cache: Vec<Option<OpRef>>,
    virtuals_int_cache: Vec<Option<OpRef>>,
    /// resume.py:882 virtual_ptr_cache concrete parity: raw GcRef
    /// allocation for each virtual, unified with the symbolic OpRef cache
    /// above (RPython's VirtualCache stores both in one object).
    concrete_ptr_cache: Vec<Option<majit_ir::GcRef>>,
    /// resume.py:883 virtual_int_cache concrete parity.  Raw buffers and
    /// slices are INT virtuals, not GC refs.
    concrete_int_cache: Vec<Option<i64>>,
}

impl BridgeVirtualCache {
    fn new(size: usize) -> Self {
        Self {
            virtuals_ptr_cache: vec![None; size],
            virtuals_int_cache: vec![None; size],
            concrete_ptr_cache: vec![None; size],
            concrete_int_cache: vec![None; size],
        }
    }

    fn get_any(&self, i: usize) -> Option<OpRef> {
        self.virtuals_ptr_cache
            .get(i)
            .copied()
            .flatten()
            .or_else(|| self.virtuals_int_cache.get(i).copied().flatten())
    }

    fn set_ptr(&mut self, i: usize, v: OpRef) {
        self.virtuals_ptr_cache[i] = Some(v);
    }

    fn set_int(&mut self, i: usize, v: OpRef) {
        self.virtuals_int_cache[i] = Some(v);
    }

    fn get_concrete_ptr(&self, i: usize) -> Option<majit_ir::GcRef> {
        self.concrete_ptr_cache.get(i).copied().flatten()
    }

    fn set_concrete_ptr(&mut self, i: usize, v: majit_ir::GcRef) {
        if i < self.concrete_ptr_cache.len() {
            self.concrete_ptr_cache[i] = Some(v);
        }
    }

    fn get_concrete_int(&self, i: usize) -> Option<i64> {
        self.concrete_int_cache.get(i).copied().flatten()
    }

    fn set_concrete_int(&mut self, i: usize, v: i64) {
        if i < self.concrete_int_cache.len() {
            self.concrete_int_cache[i] = Some(v);
        }
    }
}

/// resume.py:1245-1264 unified decode_box parity.
/// Returns `(OpRef, Value)`: symbolic OpRef for trace recording, concrete
/// Value for shadow slots / continue_tracing. Replaces the separate
/// `resolve()` + `decode_concrete()` closures so both paths always execute
/// together — no drift between symbolic and concrete materialization.
/// Decode one inlined-callee resume frame
/// (`resume_data.frames[i]`, `i >= 1`) into a [`ReconstructRecipe`] for the
/// multi-frame bridge carrier. Mirrors the per-bank `consume_boxes` decode
/// `setup_bridge_sym` runs for the portal frame (resume.py:1054) but writes a
/// fresh recipe instead of the root `sym`, and emits NO trace IR (the
/// forward-call guard_value / callee-frame helper / no-exception guard belong
/// to a call with a LIVE caller; a reconstructed suspended frame has none).
///
/// The decoded bank vectors are indexed by pyre's semantic register index
/// (Python-bytecode `locals_cells_stack_w` position, not an RPython color),
/// so they align with `write_stack_slot`'s `nlocals + stack_idx`. The
/// assembly into a `PyFrame` + `PyreSym` is deferred to `trace_bytecode`
/// (the carrier drain), where the root concrete frame's globals/EC are
/// available and the rebuilt locals are GC-rooted immediately.
///
/// Returns `None` (→ abort the multi-frame path, fall back to the
/// single-frame bridge) when the callee cannot be faithfully rebuilt:
///   - `pc` is the no-snapshot sentinel (`< 0`),
///   - `jitcode_index` does not resolve to a code object,
///   - the callee has cell or free variables — `locals_cells_stack_w` then
///     carries cell objects whose contents are not in the resume liveness
///     stream, so a dead-frame rebuild would seed null cells and LOAD_DEREF
///     would raise `NameError`,
///   - the liveness enumeration count disagrees with the encoded section.
fn reconstruct_inline_recipe(
    ctx: &mut majit_metainterp::TraceCtx,
    frame: &majit_ir::resumedata::RebuiltFrame,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &majit_metainterp::ResumeDataResult,
    fail_values: &[i64],
    fail_types: &[Type],
    backend: &dyn majit_backend::Backend,
    cache: &mut BridgeVirtualCache,
) -> Option<ReconstructRecipe> {
    if frame.pc < 0 {
        return None;
    }
    let w_code = code_for_jitcode_index(frame.jitcode_index)?;
    if w_code.is_null() {
        return None;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw_code.is_null() {
        return None;
    }
    let code_ref = unsafe { &*raw_code };
    if !code_ref.freevars.is_empty() || pyre_interpreter::pyframe::ncells(code_ref) != 0 {
        return None;
    }
    // pyframe.py:128-132 get_w_globals(): the reconstructed callee frame's
    // globals come from its own pycode (`assemble_bridge_inline_pending`
    // resolves them via `w_code_get_w_globals`). If the callee code never ran
    // under known globals (`w_globals` null), there is no namespace to restore,
    // so abort to the single-frame bridge — the forward inline path declines
    // the same way.
    if unsafe { pyre_interpreter::w_code_get_w_globals(w_code as pyre_object::PyObjectRef) }
        .is_null()
    {
        return None;
    }
    let nlocals = code_ref.varnames.len();

    let reg_indices = frame_liveness_reg_indices_by_bank_at(frame.jitcode_index, frame.pc);
    // resume.py:1054 consume_boxes: the liveness enumeration count must match
    // the encoded frame section exactly.
    if reg_indices.total_len() != frame.values.len() {
        return None;
    }
    // virtualizable.py:86-98: at a bytecode boundary (every resume pc is one)
    // the frame's locals_cells_stack_w is a W_Root array — all live slots are
    // boxed Ref. The encoder confirms this empirically (both reconstructed
    // function_calls frames decode 0 int / 0 float registers). A reconstructed
    // inline callee has no virtualizable array, so an int/float-bank register
    // here would have no boxed-Ref source to seed `registers_r` (the reader
    // always reads `registers_r[k]`, trace_opcode.rs:2128/684). Fall back to
    // the single-frame bridge rather than synthesize an unboxed local.
    if !reg_indices.int.is_empty() || !reg_indices.float.is_empty() {
        return None;
    }

    let mut registers_i: Vec<OpRef> = Vec::new();
    let mut registers_r: Vec<OpRef> = Vec::new();
    let mut registers_f: Vec<OpRef> = Vec::new();
    let mut concrete_r: Vec<majit_ir::Value> = Vec::new();
    let mut value_cursor = 0usize;
    for &reg_idx in &reg_indices.int {
        let (op, _val) = bridge_decode_box(
            ctx,
            &frame.values[value_cursor],
            Type::Int,
            rd_virtuals,
            resume_data,
            fail_values,
            fail_types,
            backend,
            cache,
        );
        let reg_idx = reg_idx as usize;
        if reg_idx >= registers_i.len() {
            registers_i.resize(reg_idx + 1, OpRef::NONE);
        }
        registers_i[reg_idx] = op;
        value_cursor += 1;
    }
    for &reg_idx in &reg_indices.ref_ {
        let (op, val) = bridge_decode_box(
            ctx,
            &frame.values[value_cursor],
            Type::Ref,
            rd_virtuals,
            resume_data,
            fail_values,
            fail_types,
            backend,
            cache,
        );
        let reg_idx = reg_idx as usize;
        if reg_idx >= registers_r.len() {
            registers_r.resize(reg_idx + 1, OpRef::NONE);
            concrete_r.resize(reg_idx + 1, majit_ir::Value::Void);
        }
        registers_r[reg_idx] = op;
        concrete_r[reg_idx] = val;
        value_cursor += 1;
    }
    for &reg_idx in &reg_indices.float {
        let (op, _val) = bridge_decode_box(
            ctx,
            &frame.values[value_cursor],
            Type::Float,
            rd_virtuals,
            resume_data,
            fail_values,
            fail_types,
            backend,
            cache,
        );
        let reg_idx = reg_idx as usize;
        if reg_idx >= registers_f.len() {
            registers_f.resize(reg_idx + 1, OpRef::NONE);
        }
        registers_f[reg_idx] = op;
        value_cursor += 1;
    }

    // pyframe.py:107-110: locals + cells + stack. Cells are gated out above.
    // The semantic `valuestackdepth` is `stack_base() + operand_depth`, where
    // `operand_depth` is the logical stack height the codewriter's forward
    // dataflow computes for this pc (`LiveVars::stack_depth_at`). It is NOT
    // the live-register count: the liveness file can keep dead temp slots
    // live ABOVE the logical stack top (e.g. at a conditional-branch target
    // the not-taken branch's stack colors stay live), so `registers_r.len()`
    // over-counts the stack. The portal frame reads the equivalent figure
    // from the encoded vable `valuestackdepth` scalar (state.rs:6382); an
    // inline frame has no such scalar, so derive it from the bytecode here.
    // Dead-temp register slots above `valuestackdepth` stay in `registers_r`
    // (the reader never accesses them as stack) but are NOT seeded as
    // concrete operands. An unreachable resume pc aborts the multi-frame path.
    let valuestackdepth =
        match crate::liveness::liveness_for(raw_code).stack_depth_at(frame.pc as usize) {
            Some(d) => nlocals + d,
            None => return None,
        };
    if registers_r.len() < valuestackdepth {
        registers_r.resize(valuestackdepth, OpRef::NONE);
        concrete_r.resize(valuestackdepth, majit_ir::Value::Void);
    }

    Some(ReconstructRecipe {
        w_code,
        jitcode_index: frame.jitcode_index,
        pc: frame.pc as usize,
        nlocals,
        valuestackdepth,
        registers_i,
        registers_r,
        registers_f,
        concrete_r,
        nargs: nlocals,
    })
}

fn bridge_decode_box(
    ctx: &mut majit_metainterp::TraceCtx,
    v: &majit_ir::resumedata::RebuiltValue,
    expected_kind: Type,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &majit_metainterp::ResumeDataResult,
    fail_values: &[i64],
    fail_types: &[Type],
    backend: &dyn majit_backend::Backend,
    cache: &mut BridgeVirtualCache,
) -> (OpRef, majit_ir::Value) {
    use majit_ir::resumedata::RebuiltValue;
    let callinfocollection = ctx.callinfocollection.clone();
    match v {
        RebuiltValue::Box(n, tp) => {
            let opref = OpRef::input_arg_typed(*n as u32, *tp);
            let bits = fail_values[*n];
            let effective_tp = fail_types.get(*n).copied().unwrap_or(*tp);
            // resume.py:1264 assert box.type == kind
            assert!(
                effective_tp == expected_kind,
                "bridge_decode_box: Box({n}) type {effective_tp:?} != expected {expected_kind:?}"
            );
            (opref, value_for_slot(effective_tp, bits))
        }
        RebuiltValue::Const(c) => {
            let opref = match c.get_type() {
                Type::Ref => ctx.const_ref(c.getref_base().as_usize() as i64),
                Type::Float => ctx.const_float(c.getfloatstorage()),
                _ => ctx.const_int(c.getint()),
            };
            // resume.py:1264 assert box.type == kind
            assert!(
                c.get_type() == expected_kind,
                "bridge_decode_box: Const type {:?} != expected {:?}",
                c.get_type(),
                expected_kind,
            );
            (opref, value_for_slot(c.get_type(), c.as_raw_i64()))
        }
        RebuiltValue::Virtual(vidx) => {
            let opref = materialize_bridge_virtual(ctx, *vidx, rd_virtuals, resume_data, cache);
            if expected_kind == Type::Int {
                let value = materialize_concrete_virtual_int(
                    *vidx,
                    rd_virtuals,
                    fail_values,
                    resume_data.num_failargs,
                    resume_data.storage.as_ref(),
                    backend,
                    callinfocollection.as_ref(),
                    cache,
                );
                (opref, majit_ir::Value::Int(value))
            } else {
                let gcref = materialize_concrete_virtual_ptr(
                    *vidx,
                    rd_virtuals,
                    fail_values,
                    resume_data.num_failargs,
                    resume_data.storage.as_ref(),
                    backend,
                    callinfocollection.as_ref(),
                    cache,
                );
                (opref, majit_ir::Value::Ref(gcref))
            }
        }
        RebuiltValue::Unassigned => (OpRef::NONE, majit_ir::Value::Void),
    }
}

/// resume.py:1245-1264 decode_box concrete parity for tagged fieldnums.
/// Converts a tagged i16 (from `rd_virtuals[*].fieldnums`) into raw i64
/// bits suitable for backend concrete setters/calls.
fn decode_tagged_concrete(
    tagged: i16,
    expected_kind: Type,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    backend: &dyn majit_backend::Backend,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache,
) -> i64 {
    use majit_ir::resumedata::{
        NULLREF, TAG_CONST_OFFSET, TAGBOX, TAGCONST, TAGINT, TAGVIRTUAL, UNINITIALIZED_TAG, untag,
    };
    if tagged == UNINITIALIZED_TAG {
        return 0;
    }
    let (val, tagbits) = untag(tagged);
    match tagbits {
        TAGBOX => {
            let idx = if val < 0 {
                (val + num_failargs) as usize
            } else {
                val as usize
            };
            // resume.py:1261 fail-loud: direct indexing, not silent fallback
            fail_values[idx]
        }
        TAGINT => val as i64,
        TAGCONST => {
            if tagged == NULLREF {
                return 0;
            }
            let ci = (val - TAG_CONST_OFFSET) as usize;
            // resume.py:1251 fail-loud: direct indexing
            let storage = storage.expect("decode_tagged_concrete: TAGCONST requires storage");
            storage.rd_consts()[ci].as_raw_i64()
        }
        TAGVIRTUAL => {
            if expected_kind == Type::Int {
                materialize_concrete_virtual_int(
                    val as usize,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    backend,
                    callinfocollection,
                    cache,
                )
            } else {
                let gcref = materialize_concrete_virtual_ptr(
                    val as usize,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    backend,
                    callinfocollection,
                    cache,
                );
                gcref.0 as i64
            }
        }
        _ => 0,
    }
}

fn bh_field_descr_from_info(fd: &majit_ir::FieldDescrInfo) -> majit_translate::jitcode::BhDescr {
    majit_translate::jitcode::BhDescr::Field {
        offset: fd.offset,
        field_size: fd.field_size,
        field_type: fd.field_type,
        field_flag: majit_ir::ArrayFlag::from_field_type(fd.field_type),
        is_field_signed: matches!(fd.field_type, Type::Int),
        is_immutable: false,
        is_quasi_immutable: false,
        index_in_parent: fd.index as usize,
        parent: None,
        name: String::new(),
        owner: String::new(),
    }
}

fn bh_size_descr_from_size_descr(
    size_descr: &dyn majit_ir::descr::SizeDescr,
    vtable: usize,
) -> majit_translate::jitcode::BhDescr {
    majit_translate::jitcode::BhDescr::Size {
        size: size_descr.size(),
        // descr.py type_id is the dense GC tid for alloc_nursery_typed,
        // not the cache_key structural identity
        type_id: size_descr.type_id() as u64,
        vtable,
        owner: String::new(),
        all_fielddescrs: majit_translate::jitcode::bh_field_specs_from_size_descr(size_descr),
    }
}

fn bh_array_descr_from_descr(descr: &majit_ir::DescrRef) -> majit_translate::jitcode::BhDescr {
    let ad = descr
        .as_array_descr()
        .expect("resume.py: allocate_array requires ArrayDescr");
    majit_translate::jitcode::BhDescr::from_array_descr(ad)
}

fn decode_tagged_for_kind(
    tagged: i16,
    kind: Type,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    backend: &dyn majit_backend::Backend,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache,
) -> i64 {
    decode_tagged_concrete(
        tagged,
        kind,
        rd_virtuals,
        fail_values,
        num_failargs,
        storage,
        backend,
        callinfocollection,
        cache,
    )
}

fn setfield_concrete_from_tagged(
    backend: &dyn majit_backend::Backend,
    struct_ptr: i64,
    fd: &majit_ir::FieldDescrInfo,
    fieldnum: i16,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache,
) {
    let descr = bh_field_descr_from_info(fd);
    match fd.field_type {
        Type::Ref => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Ref,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            backend.bh_setfield_gc_r(struct_ptr, majit_ir::GcRef(value as usize), &descr);
        }
        Type::Float => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Float,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            backend.bh_setfield_gc_f(struct_ptr, f64::from_bits(value as u64), &descr);
        }
        _ => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Int,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            backend.bh_setfield_gc_i(struct_ptr, value, &descr);
        }
    }
}

fn setarrayitem_concrete_from_tagged(
    backend: &dyn majit_backend::Backend,
    array_ptr: i64,
    index: usize,
    arraydescr: &dyn majit_ir::descr::ArrayDescr,
    bh_descr: &majit_translate::jitcode::BhDescr,
    fieldnum: i16,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache,
) {
    if arraydescr.is_array_of_pointers() {
        let value = decode_tagged_for_kind(
            fieldnum,
            Type::Ref,
            rd_virtuals,
            fail_values,
            num_failargs,
            storage,
            backend,
            callinfocollection,
            cache,
        );
        backend.bh_setarrayitem_gc_r(
            array_ptr,
            index as i64,
            majit_ir::GcRef(value as usize),
            bh_descr,
        );
    } else if arraydescr.is_array_of_floats() {
        let value = decode_tagged_for_kind(
            fieldnum,
            Type::Float,
            rd_virtuals,
            fail_values,
            num_failargs,
            storage,
            backend,
            callinfocollection,
            cache,
        );
        backend.bh_setarrayitem_gc_f(
            array_ptr,
            index as i64,
            f64::from_bits(value as u64),
            bh_descr,
        );
    } else {
        let value = decode_tagged_for_kind(
            fieldnum,
            Type::Int,
            rd_virtuals,
            fail_values,
            num_failargs,
            storage,
            backend,
            callinfocollection,
            cache,
        );
        backend.bh_setarrayitem_gc_i(array_ptr, index as i64, value, bh_descr);
    }
}

/// resume.py:1543-1552 BlackholeResumeDataReader.setinteriorfield_{int,
/// ref,float}: dispatch on the `InteriorFieldDescr.fielddescr` type to
/// `cpu.bh_setinteriorfield_gc_{i,r,f}`.
fn setinteriorfield_concrete_from_tagged(
    backend: &dyn majit_backend::Backend,
    array_ptr: i64,
    index: usize,
    interior_descr: &dyn majit_ir::descr::Descr,
    fieldnum: i16,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache,
) {
    let ifd = interior_descr
        .as_interior_field_descr()
        .expect("VArrayStructInfo: fielddescr is not an InteriorFieldDescr");
    let bh = majit_translate::jitcode::BhDescr::from_interior_field_descr(ifd);
    match ifd.field_descr().field_type() {
        Type::Ref => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Ref,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            backend.bh_setinteriorfield_gc_r(
                array_ptr,
                index as i64,
                majit_ir::GcRef(value as usize),
                &bh,
            );
        }
        Type::Float => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Float,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            backend.bh_setinteriorfield_gc_f(
                array_ptr,
                index as i64,
                f64::from_bits(value as u64),
                &bh,
            );
        }
        _ => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Int,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            backend.bh_setinteriorfield_gc_i(array_ptr, index as i64, value, &bh);
        }
    }
}

fn bh_call_r_for_oopspec(
    backend: &dyn majit_backend::Backend,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    oopspec: majit_ir::OopSpecIndex,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
) -> majit_ir::GcRef {
    let cic = callinfocollection.expect(
        "TraceCtx.callinfocollection missing — concrete VStr/VUni materialization \
         requires resume.py DirectReader funcptr_for_oopspec parity",
    );
    let (calldescr, func) = cic.callinfo_for_oopspec(oopspec);
    let calldescr = calldescr
        .expect("callinfo_for_oopspec missing entry for concrete VStr/VUni materialization");
    let cd = calldescr
        .as_call_descr()
        .expect("VStr/VUni oopspec calldescr must be CallDescr");
    let bh_calldescr = majit_translate::jitcode::BhCallDescr::from_call_descr(cd);
    backend.bh_call_r(func as i64, args_i, args_r, None, &bh_calldescr)
}

/// resume.py:945-956 getvirtual_ptr concrete parity.
/// Lazily allocate a concrete object for virtual index `vidx`, caching in
/// `BridgeVirtualCache.concrete_ptr_cache` so shared/recursive virtuals
/// allocate exactly once.
///
/// resume.py:618-760 polymorphic dispatch by RdVirtualInfo variant.
/// BlackholeResumeDataReader (resume.py:1430-1460) routes each kind
/// through the CPU backend: bh_new_with_vtable, bh_new, bh_new_array,
/// bh_newstr, bh_newunicode, bh_call_i/r.
fn materialize_concrete_virtual_ptr(
    vidx: usize,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    backend: &dyn majit_backend::Backend,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache,
) -> majit_ir::GcRef {
    if let Some(cached) = cache.get_concrete_ptr(vidx) {
        return cached;
    }
    // resume.py:953 assert self.rd_virtuals is not None
    let virtuals = rd_virtuals.expect("materialize_concrete_virtual_ptr: rd_virtuals is None");
    // resume.py:954 self.rd_virtuals[index].allocate(self, index) — direct indexing
    let entry = &virtuals[vidx];
    match entry.as_ref() {
        // resume.py:618-621 VirtualInfo.allocate
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let descr = descr.as_ref().expect("VirtualInfo: descr is None");
            let size_descr = descr.as_size_descr().expect("VirtualInfo: not a SizeDescr");
            let vtable = size_descr.vtable();
            // resume.py:1111 allocate_with_vtable(descr) → cpu.bh_new_with_vtable(descr)
            let bh_descr = bh_size_descr_from_size_descr(size_descr, vtable);
            let ptr = backend.bh_new_with_vtable(&bh_descr);
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            // Pyre adaptation: bh_new_with_vtable writes vtable at
            // vtable_offset but PyObject.w_class needs separate init
            // (pyobject.rs:51). Matches materialize_virtual_object at
            // state.rs:7220.
            if vtable != 0 {
                unsafe {
                    let pyobj = ptr as *mut pyre_object::PyObject;
                    (*pyobj).w_class = pyre_object::pyobject::get_instantiate(
                        &*(vtable as *const pyre_object::pyobject::PyType),
                    );
                }
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            // resume.py:620 cache BEFORE filling fields (circular ref safe)
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py:597-603 setfields — range(len(fielddescrs)), index
            // fieldnums[i]. The len-equality assert (resume.py:606) is in
            // debug_prints, not this allocate path: a short fieldnums raises
            // IndexError here, a longer one is ignored.
            for i in 0..fielddescrs.len() {
                let fd = &fielddescrs[i];
                let fnum = fieldnums[i];
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                setfield_concrete_from_tagged(
                    backend,
                    ptr,
                    fd,
                    fnum,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    callinfocollection,
                    cache,
                );
            }
            gcref
        }
        // resume.py:633-637 VStructInfo.allocate — no vtable
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let descr = typedescr.as_ref().expect("VStructInfo: typedescr is None");
            let size_descr = descr.as_size_descr().expect("VStructInfo: not a SizeDescr");
            // resume.py:1114 allocate_struct(typedescr) → cpu.bh_new(typedescr)
            let bh_descr = bh_size_descr_from_size_descr(size_descr, 0);
            let ptr = backend.bh_new(&bh_descr);
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py:637 setfields — range(len(fielddescrs)), index
            // fieldnums[i]. The len-equality assert (resume.py:606) is in
            // debug_prints, not this allocate path: a short fieldnums raises
            // IndexError here, a longer one is ignored.
            for i in 0..fielddescrs.len() {
                let fd = &fielddescrs[i];
                let fnum = fieldnums[i];
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                setfield_concrete_from_tagged(
                    backend,
                    ptr,
                    fd,
                    fnum,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    callinfocollection,
                    cache,
                );
            }
            gcref
        }
        // resume.py:650-670 VArrayInfo.allocate
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            arraydescr,
            fieldnums,
            ..
        }
        | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            arraydescr,
            fieldnums,
            ..
        } => {
            let length = fieldnums.len();
            let is_clear = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VArrayInfoClear { .. }
            );
            // resume.py:1117 allocate_array(length, arraydescr, clear)
            // → cpu.bh_new_array_clear / bh_new_array
            let descr = arraydescr.as_ref().expect("VArrayInfo: arraydescr is None");
            let ad = descr
                .as_array_descr()
                .expect("VArrayInfo: not an ArrayDescr");
            let bh_descr = bh_array_descr_from_descr(descr);
            let ptr = if is_clear {
                backend.bh_new_array_clear(length as i64, &bh_descr)
            } else {
                backend.bh_new_array(length as i64, &bh_descr)
            };
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py:660-670 setarrayitem per element
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                setarrayitem_concrete_from_tagged(
                    backend,
                    ptr,
                    i,
                    ad,
                    &bh_descr,
                    fnum,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    callinfocollection,
                    cache,
                );
            }
            gcref
        }
        // resume.py:747-760 VArrayStructInfo.allocate
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr,
            fielddescrs,
            size,
            fieldnums,
            ..
        } => {
            // resume.py:749 array = decoder.allocate_array(self.size,
            // self.arraydescr, clear=True) — uses the live `self.arraydescr`
            // directly.
            let array_descr = arraydescr
                .as_ref()
                .expect("VArrayStructInfo: arraydescr is None");
            let bh_array = bh_array_descr_from_descr(array_descr);
            // resume.py:749 clear=True → bh_new_array_clear
            let ptr = backend.bh_new_array_clear(*size as i64, &bh_array);
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py:752-759:
            //   p = 0
            //   for i in range(self.size):
            //       for j in range(len(self.fielddescrs)):
            //           num = self.fieldnums[p]
            //           if not tagged_eq(num, UNINITIALIZED):
            //               decoder.setinteriorfield(i, array, num,
            //                   self.fielddescrs[j])
            //           p += 1
            let num_fields = fielddescrs.len();
            // resume.py:752-759 reads exactly size × len(fielddescrs) entries
            // via self.fieldnums[p] with no length-equality check: a short
            // fieldnums is an out-of-bounds error here (IndexError parity), a
            // longer one leaves its tail unread.
            let mut p = 0;
            for i in 0..*size {
                for j in 0..num_fields {
                    let fnum = fieldnums[p];
                    p += 1;
                    if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                        continue;
                    }
                    setinteriorfield_concrete_from_tagged(
                        backend,
                        ptr,
                        i,
                        fielddescrs[j].as_ref(),
                        fnum,
                        rd_virtuals,
                        fail_values,
                        num_failargs,
                        storage,
                        callinfocollection,
                        cache,
                    );
                }
            }
            gcref
        }
        // resume.py:763-783 VStrPlainInfo.allocate
        majit_ir::RdVirtualInfo::VStrPlainInfo { fieldnums } => {
            let length = fieldnums.len();
            // resume.py:1134 allocate_string(length) → cpu.bh_newstr(length)
            let ptr = backend.bh_newstr(length as i64);
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py:1138 string_setitem → cpu.bh_strsetitem
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                let value = decode_tagged_concrete(
                    fnum,
                    Type::Int,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    backend,
                    callinfocollection,
                    cache,
                );
                backend.bh_strsetitem(ptr, i as i64, value);
            }
            gcref
        }
        // resume.py:820-840 VUniPlainInfo.allocate
        majit_ir::RdVirtualInfo::VUniPlainInfo { fieldnums } => {
            let length = fieldnums.len();
            // resume.py:1162 allocate_unicode(length) → cpu.bh_newunicode(length)
            let ptr = backend.bh_newunicode(length as i64);
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py:1166 unicode_setitem → cpu.bh_unicodesetitem
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                let value = decode_tagged_concrete(
                    fnum,
                    Type::Int,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    backend,
                    callinfocollection,
                    cache,
                );
                backend.bh_unicodesetitem(ptr, i as i64, value);
            }
            gcref
        }
        // resume.py:785-805 VStrConcatInfo / VUniConcatInfo
        majit_ir::RdVirtualInfo::VStrConcatInfo { fieldnums }
        | majit_ir::RdVirtualInfo::VUniConcatInfo { fieldnums } => {
            assert!(fieldnums.len() == 2, "concat: expected 2 fieldnums");
            let str1 = decode_tagged_concrete(
                fieldnums[0],
                Type::Ref,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let str2 = decode_tagged_concrete(
                fieldnums[1],
                Type::Ref,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let oopspec = if matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VStrConcatInfo { .. }
            ) {
                majit_ir::descr::OopSpecIndex::StrConcat
            } else {
                majit_ir::descr::OopSpecIndex::UniConcat
            };
            // resume.py:1462-1497 DirectReader concat_* resolves funcptr
            // via callinfocollection.funcptr_for_oopspec, not a placeholder.
            let gcref = bh_call_r_for_oopspec(
                backend,
                callinfocollection,
                oopspec,
                None,
                Some(&[str1, str2]),
            );
            cache.set_concrete_ptr(vidx, gcref);
            gcref
        }
        // resume.py:805-818 VStrSliceInfo / VUniSliceInfo
        majit_ir::RdVirtualInfo::VStrSliceInfo { fieldnums }
        | majit_ir::RdVirtualInfo::VUniSliceInfo { fieldnums } => {
            assert!(fieldnums.len() == 3, "slice: expected 3 fieldnums");
            let strbox = decode_tagged_concrete(
                fieldnums[0],
                Type::Ref,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let start = decode_tagged_concrete(
                fieldnums[1],
                Type::Int,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let length = decode_tagged_concrete(
                fieldnums[2],
                Type::Int,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let stop = start + length;
            let oopspec = if matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VStrSliceInfo { .. }
            ) {
                majit_ir::descr::OopSpecIndex::StrSlice
            } else {
                majit_ir::descr::OopSpecIndex::UniSlice
            };
            // resume.py:1472-1507 DirectReader slice_* passes stop =
            // start + length to the funcptr resolved from callinfocollection.
            let gcref = bh_call_r_for_oopspec(
                backend,
                callinfocollection,
                oopspec,
                Some(&[start, stop]),
                Some(&[strbox]),
            );
            cache.set_concrete_ptr(vidx, gcref);
            gcref
        }
        // Raw virtuals are INT-kind — should not appear in getvirtual_ptr
        majit_ir::RdVirtualInfo::VRawBufferInfo { .. }
        | majit_ir::RdVirtualInfo::VRawSliceInfo { .. } => {
            panic!(
                "materialize_concrete_virtual_ptr: raw virtual at vidx={vidx} is INT-kind, not PTR"
            )
        }
        // resume.py:954 getvirtual_ptr calls rd_virtuals[index].allocate()
        // directly; a None/Empty slot is never referenced by a TAGVIRTUAL tag
        // in a well-formed resume stream, so reaching it is a corrupt-stream
        // bug, not a NULL fallback (mirrors materialize_concrete_virtual_int
        // and eval.rs materialize_virtual, which already panic on Empty).
        majit_ir::RdVirtualInfo::Empty => panic!(
            "materialize_concrete_virtual_ptr: null rd_virtuals[{vidx}] (resume.py: null rd_virtuals[index])"
        ),
    }
}

/// resume.py:958-967 getvirtual_int concrete parity.  Only raw-buffer
/// virtuals are INT virtuals (`VAbstractRawInfo.kind = INT`).
fn materialize_concrete_virtual_int(
    vidx: usize,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    backend: &dyn majit_backend::Backend,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache,
) -> i64 {
    if let Some(cached) = cache.get_concrete_int(vidx) {
        return cached;
    }
    // resume.py:959 assert self.virtuals_cache is not None (cache is not None by construction)
    // resume.py:962 self.rd_virtuals[index] — direct indexing, not silent fallback
    let virtuals = rd_virtuals.expect("materialize_concrete_virtual_int: rd_virtuals is None");
    let entry = &virtuals[vidx];
    // resume.py:964 assert v.is_about_raw and isinstance(v, VAbstractRawInfo)
    match entry.as_ref() {
        // resume.py:700-709 VRawBufferInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets,
            descrs,
            fieldnums,
        } => {
            // resume.py:1452-1456 BlackholeResumeDataReader.allocate_raw_buffer:
            //   cic = self.callinfocollection
            //   calldescr, _ = cic.callinfo_for_oopspec(OS_RAW_MALLOC_VARSIZE_CHAR)
            //   return self.cpu.bh_call_i(func, [size], None, None, calldescr)
            // No fallback upstream — a missing CIC entry is a metainterp-setup
            // bug, surfaced fail-loud rather than papered over with a synthetic
            // calldescr.
            let cic = callinfocollection.expect("allocate_raw_buffer: callinfocollection is None");
            // resume.py:1455: calldescr, _ = cic.callinfo_for_oopspec(
            //   OS_RAW_MALLOC_VARSIZE_CHAR). callinfo_for_oopspec returns
            // (None, 0) on a missing entry (effectinfo.py:444-447) — no
            // lookup-time check; the calldescr is used directly.
            let (calldescr, _) =
                cic.callinfo_for_oopspec(majit_ir::descr::OopSpecIndex::RawMallocVarsizeChar);
            // resume.py:1456: self.cpu.bh_call_i(func, [size], None, None,
            //   calldescr). A missing entry surfaces here, as the calldescr is
            // consumed by the backend call, not as a separate lookup assertion.
            let descr_ref =
                calldescr.expect("OS_RAW_MALLOC_VARSIZE_CHAR calldescr (callinfocollection)");
            let cd = descr_ref
                .as_call_descr()
                .expect("OS_RAW_MALLOC_VARSIZE_CHAR: not a CallDescr");
            let bh = majit_translate::jitcode::BhCallDescr::from_arg_classes(
                cd.arg_classes(),
                cd.result_class(),
                cd.get_extra_info().clone(),
            );
            let buffer = backend.bh_call_i(*func, Some(&[*size as i64]), None, None, &bh);
            // resume.py:704: decoder.virtuals_cache.set_int(index, buffer)
            cache.set_concrete_int(vidx, buffer);
            // resume.py:705-708 iterate by len(self.offsets) unconditionally
            // (no buffer == 0 guard), indexing self.descrs[i] and
            // self.fieldnums[i] by the same i — a short descrs/fieldnums raises
            // IndexError here (encoder bug), a longer one is ignored. No
            // len-equality assert (VRawBufferInfo has none).
            for i in 0..offsets.len() {
                let off = offsets[i];
                let fnum = fieldnums[i];
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                // resume.py:1543 assert not descr.is_array_of_pointers()
                assert!(
                    descrs[i].item_type != 0,
                    "setrawbuffer_item: pointer arraydescr in raw buffer"
                );
                // resume.py:1545-1552 setrawbuffer_item: the arraydescr drives
                // the store width via the backend's write_{int,float}_at_mem,
                // so the value is stored through bh_raw_store_{f,i} with the
                // real descr — no caller-side field-size truncation.
                let store_descr =
                    majit_translate::jitcode::BhDescr::from_array_descr_info(&descrs[i]);
                if descrs[i].item_type == 2 {
                    let value = decode_tagged_concrete(
                        fnum,
                        Type::Float,
                        rd_virtuals,
                        fail_values,
                        num_failargs,
                        storage,
                        backend,
                        callinfocollection,
                        cache,
                    );
                    // resume.py:1548-1549 bh_raw_store_f(ptr, offset, floatval, descr)
                    backend.bh_raw_store_f(
                        buffer,
                        off as i64,
                        f64::from_bits(value as u64),
                        &store_descr,
                    );
                } else {
                    let value = decode_tagged_concrete(
                        fnum,
                        Type::Int,
                        rd_virtuals,
                        fail_values,
                        num_failargs,
                        storage,
                        backend,
                        callinfocollection,
                        cache,
                    );
                    // resume.py:1550-1552 bh_raw_store_i(ptr, offset, intval, descr)
                    backend.bh_raw_store_i(buffer, off as i64, value, &store_descr);
                }
            }
            buffer
        }
        // resume.py:722-728 VRawSliceInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            assert!(
                fieldnums.len() == 1,
                "VRawSliceInfo must have exactly 1 fieldnum"
            );
            let base = decode_tagged_concrete(
                fieldnums[0],
                Type::Int,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let buffer = base + *offset;
            cache.set_concrete_int(vidx, buffer);
            buffer
        }
        // resume.py:963-964 getvirtual_int asserts is_about_raw
        other => panic!(
            "materialize_concrete_virtual_int: non-raw virtual kind at vidx={vidx}: {other:?}"
        ),
    }
}

fn materialize_bridge_virtual(
    ctx: &mut majit_metainterp::TraceCtx,
    vidx: usize,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &majit_metainterp::ResumeDataResult,
    cache: &mut BridgeVirtualCache,
) -> OpRef {
    use majit_ir::OpCode;
    use majit_ir::resumedata::{TAG_CONST_OFFSET, TAGBOX, TAGCONST, TAGINT, TAGVIRTUAL, untag};

    // resume.py:874-899 VirtualCache: list caches indexed by virtual number
    // (ptr and int banks). This bridge helper is still OpRef-typed, so it
    // probes both banks before allocating.
    if let Some(cached) = cache.get_any(vidx) {
        return cached;
    }

    // resume.py:947 assert self.virtuals_cache is not None — a TAGVIRTUAL in
    // the stream guarantees rd_virtuals is present; None is an encoder bug.
    let virtuals = rd_virtuals.expect("materialize_bridge_virtual: rd_virtuals is None");
    // resume.py:951 self.rd_virtuals[index] — direct indexing, IndexError on
    // an out-of-range virtual number is a bug, not a silent NONE fallback.
    let entry = &virtuals[vidx];

    // resume.py:1556-1564 decode_box parity for fieldnums (i16 tagged).
    fn decode_fieldnum(
        ctx: &mut majit_metainterp::TraceCtx,
        tagged: i16,
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        resume_data: &majit_metainterp::ResumeDataResult,
        cache: &mut BridgeVirtualCache,
    ) -> OpRef {
        // resume.py:1245 `decode_box` dispatches purely on the tag bits;
        // it has no UNINITIALIZED case. The UNINITIALIZED skip lives in
        // the callers (e.g. VArrayStructInfo.allocate, resume.py:629),
        // so this decoder mirrors `decode_box` exactly — an UNINITIALIZED
        // tag reaching here falls into the TAGCONST arm and fails loud on
        // the out-of-range const index, matching upstream's IndexError.
        let (val, tagbits) = untag(tagged);
        match tagbits {
            TAGBOX => {
                // resume.py:1247-1264 decode_box parity:
                //   if num < 0: num += len(liveboxes)
                //   return self.liveboxes[num]
                // The returned Box object carries `box.type` intrinsically
                // (history.py:220). For the bridge tracer, those liveboxes
                // are the bridge's `InputArg{Int,Ref,Float}` slots, so we
                // mint the typed `OpRef::input_arg_typed` variant matching
                // `fail_arg_types[idx]` rather than a bare untyped raw
                // OpRef — variant-aware Eq (resoperation.rs:290) requires
                // the optimizer/heap-cache key to be the same typed variant
                // the bridge inputarg list produces.
                let idx = if val < 0 {
                    val + resume_data.num_failargs
                } else {
                    val
                };
                // resume.py:1261 `box = self.liveboxes[num]` — direct
                // indexing, IndexError on out-of-range. Encoder /
                // decoder asymmetry is a bug, not a silent fallback;
                // mirror the upstream fail-loud contract.
                let tp = *resume_data
                    .fail_arg_types
                    .get(idx as usize)
                    .unwrap_or_else(|| {
                        panic!(
                            "decode_fieldnum TAGBOX out-of-range: idx={} num_failargs={} \
                             fail_arg_types.len()={} (encoder/decoder mismatch — see \
                             resume.py:1245-1264 decode_box)",
                            idx,
                            resume_data.num_failargs,
                            resume_data.fail_arg_types.len()
                        )
                    });
                OpRef::input_arg_typed(idx as u32, tp)
            }
            TAGINT => ctx.const_int(val as i64),
            TAGCONST => {
                // resume.py:1247-1251 decode_box parity:
                //   if tag == TAGCONST:
                //       if tagged_eq(tagged, NULLREF):
                //           box = CONST_NULL
                //       else:
                //           box = self.consts[num - TAG_CONST_OFFSET]
                if tagged == majit_ir::resumedata::NULLREF {
                    return ctx.const_null();
                }
                let ci = (val - TAG_CONST_OFFSET) as usize;
                // resume.py:1251 `box = self.consts[num - TAG_CONST_OFFSET]`
                // — direct indexing, fail-fast on out-of-range (mirrors
                // Python IndexError; never silently substitutes).
                // compile.py:853 `ResumeGuardDescr` storage — read off
                // the shared Arc so the bridge tracer observes the
                // same pool the GC walker updates.
                let storage = resume_data
                    .storage
                    .as_ref()
                    .expect("resume_data.storage missing");
                let c = storage.rd_consts()[ci];
                match c.get_type() {
                    majit_ir::Type::Ref => ctx.const_ref(c.getref_base().as_usize() as i64),
                    majit_ir::Type::Float => ctx.const_float(c.getfloatstorage()),
                    _ => ctx.const_int(c.getint()),
                }
            }
            TAGVIRTUAL => {
                materialize_bridge_virtual(ctx, val as usize, rd_virtuals, resume_data, cache)
            }
            _ => OpRef::NONE,
        }
    }

    // resume.py:612-760 dispatch by virtual kind.
    // RPython: rd_virtuals[index].allocate(self, index) — polymorphic on
    // the AbstractVirtualInfo subclass. Rust equivalent: match on
    // RdVirtualInfo enum variant.

    /// resume.py:591-603 AbstractVirtualStructInfo.setfields helper.
    /// Walks fielddescrs in lock-step with fieldnums, decoding each
    /// fieldnum and emitting SETFIELD_GC.
    fn setfields(
        ctx: &mut majit_metainterp::TraceCtx,
        struct_op: OpRef,
        fielddescrs: &[majit_ir::FieldDescrInfo],
        fieldnums: &[i16],
        parent_descr: majit_ir::DescrRef,
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        resume_data: &majit_metainterp::ResumeDataResult,
        cache: &mut BridgeVirtualCache,
    ) {
        // resume.py:597-603 setfields — range(len(fielddescrs)), index
        // fieldnums[i]. The len-equality assert (resume.py:606) is in
        // debug_prints, not this allocate path: a short fieldnums raises
        // IndexError here, a longer one is ignored.
        for i in 0..fielddescrs.len() {
            let fd_info = &fielddescrs[i];
            let fnum = fieldnums[i];
            if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                continue;
            }
            let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
            if value.is_none() {
                continue;
            }
            // resume.py:597-603 self.setfields → decoder.setfield(struct,
            // fieldnum, fielddescr): reuse the parent SizeDescr's live
            // FieldDescr (canonical immutable / quasi-immutable / ei_index)
            // rather than reconstructing a partial copy. The descr is keyed
            // by index_in_parent (small sequential), not the 268M-hash
            // stable_field_index.
            let field_descr =
                crate::descr::make_field_descr_with_parent(parent_descr.clone(), fd_info.offset);
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[struct_op, value], field_descr.clone());
            // Bridge virtual rematerialisation — `upd.setfield(valuebox)`
            // parity: cache stores the Box identity (`value` OpRef).
            // Cache-hit readers resolve the intrinsic value via
            // `box_value(cached)` at hit time (covering const pool,
            // standard-virtualizable shadow, and the frontend object's
            // `value` field) — non-Const operands whose runtime concrete
            // was stamped at the original record site (or threaded from
            // the parent guard's fail_args via `set_opref_concrete`)
            // surface through that `value` field; unstamped operands
            // return `None` so the downstream sanity check skips.
            ctx.heapcache_setfield_cached(struct_op, fd_info.index, value);
        }
    }

    match entry.as_ref() {
        // resume.py:612-621 VirtualInfo.allocate
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let Some(size_descr) = descr.clone() else {
                return OpRef::NONE;
            };
            // resume.py:619 decoder.allocate_with_vtable(descr=self.descr)
            let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:620 decoder.virtuals_cache.set_ptr(index, struct)
            cache.set_ptr(vidx, new_op);
            // resume.py:621 self.setfields(decoder, struct)
            setfields(
                ctx,
                new_op,
                fielddescrs,
                fieldnums,
                size_descr,
                rd_virtuals,
                resume_data,
                cache,
            );
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VirtualInfo → OpRef::from_raw({})",
                    vidx,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py:628-637 VStructInfo.allocate
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let Some(struct_descr) = typedescr.clone() else {
                return OpRef::NONE;
            };
            // resume.py:635 decoder.allocate_struct(self.typedescr)
            let new_op = ctx.record_op_with_descr(OpCode::New, &[], struct_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:636 decoder.virtuals_cache.set_ptr(index, struct)
            cache.set_ptr(vidx, new_op);
            // resume.py:637 self.setfields(decoder, struct)
            setfields(
                ctx,
                new_op,
                fielddescrs,
                fieldnums,
                struct_descr,
                rd_virtuals,
                resume_data,
                cache,
            );
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VStructInfo → OpRef::from_raw({})",
                    vidx,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py:649-671 AbstractVArrayInfo.allocate (clear=True or False)
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            fieldnums,
            kind,
            arraydescr,
            ..
        }
        | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            fieldnums,
            kind,
            arraydescr,
            ..
        } => {
            let clear = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VArrayInfoClear { .. }
            );
            let kind = *kind;
            let length = fieldnums.len();
            let len_ref = ctx.const_int(length as i64);
            // resume.py:653 decoder.allocate_array(length, arraydescr, self.clear)
            let alloc_opcode = if clear {
                OpCode::NewArrayClear
            } else {
                OpCode::NewArray
            };
            // resume.py:645 AbstractVArrayInfo.__init__ asserts arraydescr is
            // not None; resume.py:652 allocate reads self.arraydescr directly.
            let array_descr = arraydescr.clone().expect("VArrayInfo: arraydescr is None");
            let new_op = ctx.record_op_with_descr(alloc_opcode, &[len_ref], array_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:654 decoder.virtuals_cache.set_ptr(index, array)
            cache.set_ptr(vidx, new_op);
            // resume.py:656-670 element loop: dispatch by arraydescr kind
            // NB. the check for the kind of array elements is moved out of the loop
            let set_opcode = match kind {
                0 => OpCode::SetarrayitemGc, // arraydescr.is_array_of_pointers()
                2 => OpCode::SetarrayitemGc, // arraydescr.is_array_of_floats() — TODO: SetarrayitemRaw/Float
                _ => OpCode::SetarrayitemGc, // int
            };
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                if value.is_none() {
                    continue;
                }
                let idx_ref = ctx.const_int(i as i64);
                // resume.py:660/665/670 setarrayitem_{ref,float,int}
                ctx.record_op_with_descr(
                    set_opcode,
                    &[new_op, idx_ref, value],
                    array_descr.clone(),
                );
            }
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VArrayInfo(clear={}) → OpRef::from_raw({})",
                    vidx,
                    clear,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py:747-760 VArrayStructInfo.allocate
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr,
            fielddescrs,
            size,
            fieldnums,
            ..
        } => {
            let len_ref = ctx.const_int(*size as i64);
            // resume.py:749: array = decoder.allocate_array(self.size,
            // self.arraydescr, clear=True) — uses the live `self.arraydescr`
            // directly.
            let array_descr = arraydescr
                .as_ref()
                .expect("VArrayStructInfo: arraydescr is None");
            let new_op =
                ctx.record_op_with_descr(OpCode::NewArrayClear, &[len_ref], array_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:751: decoder.virtuals_cache.set_ptr(index, array)
            cache.set_ptr(vidx, new_op);
            // resume.py:752-759:
            //   p = 0
            //   for i in range(self.size):
            //       for j in range(len(self.fielddescrs)):
            //           num = self.fieldnums[p]
            //           if not tagged_eq(num, UNINITIALIZED):
            //               decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
            //           p += 1
            let num_fields = fielddescrs.len();
            // resume.py:752-759 reads exactly size × len(fielddescrs) entries
            // via self.fieldnums[p] with no length-equality check: a short
            // fieldnums is an out-of-bounds error here (IndexError parity), a
            // longer one leaves its tail unread.
            let mut p = 0;
            for i in 0..*size {
                for j in 0..num_fields {
                    let fnum = fieldnums[p];
                    p += 1;
                    if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                        continue;
                    }
                    let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                    if value.is_none() {
                        continue;
                    }
                    let idx_ref = ctx.const_int(i as i64);
                    // resume.py:757: decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
                    ctx.record_op_with_descr(
                        OpCode::SetinteriorfieldGc,
                        &[new_op, idx_ref, value],
                        fielddescrs[j].clone(),
                    );
                }
            }
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VArrayStructInfo → OpRef::from_raw({})",
                    vidx,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py:700-709 VRawBufferInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets,
            descrs,
            fieldnums,
        } => {
            // resume.py:703: buffer = decoder.allocate_raw_buffer(self.func, self.size)
            // resume.py:1124-1132: ResumeDataBoxReader.allocate_raw_buffer →
            //   execute_and_record_varargs(rop.CALL_I, [ConstInt(func), ConstInt(size)], calldescr)
            let func_ref = ctx.const_int(*func);
            let size_ref = ctx.const_int(*size as i64);
            // resume.py:1124-1126: calldescr comes from the shared
            // callinfocollection, not a freshly minted synthetic descr. The
            // func is NOT taken from callinfo_for_oopspec (resume.py:1127-1130:
            // several malloc variants share the oopspec), so only the calldescr
            // is read from the CIC and *func from the VRawBufferInfo is kept.
            let cic = ctx
                .callinfocollection
                .as_ref()
                .expect(
                    "TraceCtx.callinfocollection missing — bridge-virtual \
                     VRawBufferInfo materialization requires pyjitpl to populate \
                     it (resume.py:1124-1126)",
                )
                .clone();
            // resume.py:1126: calldescr, _ = cic.callinfo_for_oopspec(
            //   OS_RAW_MALLOC_VARSIZE_CHAR). callinfo_for_oopspec returns
            // (None, 0) on a missing entry (effectinfo.py:444-447) — no
            // lookup-time check; the calldescr is used directly.
            let (calldescr, _) =
                cic.callinfo_for_oopspec(majit_ir::descr::OopSpecIndex::RawMallocVarsizeChar);
            // resume.py:1131-1132: execute_and_record_varargs(CALL_I,
            //   [func, size], calldescr). A missing entry surfaces here, as the
            // calldescr is consumed by the CALL_I op, not as a separate lookup
            // assertion.
            let buffer = ctx.record_op_with_descr(
                OpCode::CallI,
                &[func_ref, size_ref],
                calldescr
                    .cloned()
                    .expect("OS_RAW_MALLOC_VARSIZE_CHAR calldescr (callinfocollection)"),
            );
            // resume.py:704: decoder.virtuals_cache.set_int(index, buffer)
            cache.set_int(vidx, buffer);
            // resume.py:705-708 iterate by len(self.offsets), indexing
            // self.descrs[i] and self.fieldnums[i] by the same i — a short
            // descrs/fieldnums raises IndexError here (encoder bug), a longer
            // one is ignored. No len-equality assert (VRawBufferInfo has none).
            for i in 0..offsets.len() {
                let off = offsets[i];
                let fnum = fieldnums[i];
                // resume.py:701-708 VRawBufferStateInfo.allocate_int passes
                // fieldnums[i] straight to setrawbuffer_item with no
                // UNINITIALIZED skip (unlike VArrayStructInfo) — a raw buffer
                // is fully written by the encoder.
                // resume.py:1232: itembox = self.decode_box(fieldnum, kind).
                // `decode_box` always returns a box (no UNINITIALIZED case),
                // so the store is unconditional, matching setrawbuffer_item.
                let item = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                // resume.py:1225-1234: setrawbuffer_item (direct reader).
                // Dispatches pointer/float/int via arraydescr — all types allowed.
                let di = &descrs[i];
                let tp = match di.item_type {
                    0 => majit_ir::Type::Ref,
                    2 => majit_ir::Type::Float,
                    _ => majit_ir::Type::Int,
                };
                let store_descr = crate::descr::make_array_descr(
                    di.base_size,
                    di.item_size,
                    di.len_offset,
                    tp,
                    di.is_signed,
                );
                let offset_ref = ctx.const_int(off as i64);
                ctx.record_op_with_descr(
                    OpCode::RawStore,
                    &[buffer, offset_ref, item],
                    store_descr,
                );
            }
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VRawBufferInfo(func={:#x}, size={}) → OpRef::from_raw({})",
                    vidx,
                    func,
                    size,
                    buffer.raw(),
                );
            }
            buffer
        }
        // resume.py:722-728 VRawSliceInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            // resume.py:724: assert len(self.fieldnums) == 1
            assert!(
                fieldnums.len() == 1,
                "VRawSliceInfo must have exactly 1 fieldnum"
            );
            // resume.py:725: base_buffer = decoder.decode_int(self.fieldnums[0])
            let base_buffer = decode_fieldnum(ctx, fieldnums[0], rd_virtuals, resume_data, cache);
            // resume.py:726: buffer = decoder.int_add_const(base_buffer, self.offset)
            let offset_ref = ctx.const_int(*offset as i64);
            let buffer = ctx.record_op(OpCode::IntAdd, &[base_buffer, offset_ref]);
            // resume.py:727: decoder.virtuals_cache.set_int(index, buffer)
            cache.set_int(vidx, buffer);
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VRawSliceInfo(offset={}) → OpRef::from_raw({})",
                    vidx,
                    offset,
                    buffer.raw(),
                );
            }
            buffer
        }
        // resume.py:766-775 VStrPlainInfo.allocate / resume.py:820-829
        // VUniPlainInfo.allocate — `ResumeDataBoxReader.allocate_string /
        // allocate_unicode` followed by `string_setitem` / `unicode_setitem`
        // per character.
        //
        //     length = len(self.fieldnums)
        //     string = decoder.allocate_string(length)        # NEWSTR
        //     decoder.virtuals_cache.set_ptr(index, string)
        //     for i in range(length):
        //         charnum = self.fieldnums[i]
        //         if not tagged_eq(charnum, UNINITIALIZED):
        //             decoder.string_setitem(string, i, charnum)  # STRSETITEM
        //     return string
        majit_ir::RdVirtualInfo::VStrPlainInfo { fieldnums }
        | majit_ir::RdVirtualInfo::VUniPlainInfo { fieldnums } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniPlainInfo { .. }
            );
            let length = fieldnums.len();
            let length_ref = ctx.const_int(length as i64);
            let (alloc_opcode, set_opcode) = if is_unicode {
                (OpCode::Newunicode, OpCode::Unicodesetitem)
            } else {
                (OpCode::Newstr, OpCode::Strsetitem)
            };
            // resume.py:769: string = decoder.allocate_string(length)
            let string = ctx.record_op(alloc_opcode, &[length_ref]);
            // resume.py:770: decoder.virtuals_cache.set_ptr(index, string)
            cache.set_ptr(vidx, string);
            // resume.py:771-774: string_setitem for each filled char.
            for (i, &charnum) in fieldnums.iter().enumerate() {
                if charnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                // resume.py:1138-1141 ResumeDataBoxReader.string_setitem:
                //   charbox = self.decode_box(charnum, INT)
                //   execute_and_record(rop.STRSETITEM, string, ConstInt(index), charbox)
                let charbox = decode_fieldnum(ctx, charnum, rd_virtuals, resume_data, cache);
                if charbox.is_none() {
                    continue;
                }
                let idx_ref = ctx.const_int(i as i64);
                ctx.record_op(set_opcode, &[string, idx_ref, charbox]);
            }
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} V{}PlainInfo(length={}) → OpRef::from_raw({})",
                    vidx,
                    if is_unicode { "Uni" } else { "Str" },
                    length,
                    string.raw(),
                );
            }
            string
        }
        // resume.py:785-793 VStrConcatInfo.allocate / resume.py:840-848
        // VUniConcatInfo.allocate:
        //
        //     left, right = self.fieldnums
        //     string = decoder.concat_strings(left, right)   # CALL_R(OS_STR_CONCAT)
        //     decoder.virtuals_cache.set_ptr(index, string)
        //
        // `ResumeDataBoxReader.concat_strings` at resume.py:1143-1149:
        //
        //     cic = self.metainterp.staticdata.callinfocollection
        //     calldescr, func = cic.callinfo_for_oopspec(OS_STR_CONCAT)
        //     str1box = self.decode_box(str1num, REF)
        //     str2box = self.decode_box(str2num, REF)
        //     execute_and_record_varargs(CALL_R, [ConstInt(func), str1box, str2box], calldescr)
        majit_ir::RdVirtualInfo::VStrConcatInfo { fieldnums, .. }
        | majit_ir::RdVirtualInfo::VUniConcatInfo { fieldnums, .. } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniConcatInfo { .. }
            );
            debug_assert_eq!(
                fieldnums.len(),
                2,
                "VStr/VUniConcatInfo must have exactly 2 fieldnums (left, right)"
            );
            let left = decode_fieldnum(ctx, fieldnums[0], rd_virtuals, resume_data, cache);
            let right = decode_fieldnum(ctx, fieldnums[1], rd_virtuals, resume_data, cache);
            let oopspec = if is_unicode {
                majit_ir::effectinfo::OopSpecIndex::UniConcat
            } else {
                majit_ir::effectinfo::OopSpecIndex::StrConcat
            };
            let string = emit_stroruni_oopspec_call(ctx, oopspec, &[left, right]);
            cache.set_ptr(vidx, string);
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} V{}ConcatInfo → OpRef::from_raw({})",
                    vidx,
                    if is_unicode { "Uni" } else { "Str" },
                    string.raw(),
                );
            }
            string
        }
        // resume.py:805-809 VStrSliceInfo.allocate / resume.py:860-864
        // VUniSliceInfo.allocate:
        //
        //     largerstr, start, length = self.fieldnums
        //     string = decoder.slice_string(largerstr, start, length)
        //     decoder.virtuals_cache.set_ptr(index, string)
        //
        // `ResumeDataBoxReader.slice_string` at resume.py:1151-1160 /
        // `slice_unicode` at resume.py:1179-1188:
        //
        //     cic = self.metainterp.staticdata.callinfocollection
        //     calldescr, func = cic.callinfo_for_oopspec(OS_STR_SLICE)
        //     strbox = self.decode_box(strnum, REF)
        //     startbox = self.decode_box(startnum, INT)
        //     lengthbox = self.decode_box(lengthnum, INT)
        //     stopbox = execute_and_record(INT_ADD, startbox, lengthbox)
        //     execute_and_record_varargs(CALL_R,
        //         [ConstInt(func), strbox, startbox, stopbox], calldescr)
        majit_ir::RdVirtualInfo::VStrSliceInfo { fieldnums, .. }
        | majit_ir::RdVirtualInfo::VUniSliceInfo { fieldnums, .. } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniSliceInfo { .. }
            );
            debug_assert_eq!(
                fieldnums.len(),
                3,
                "VStr/VUniSliceInfo must have exactly 3 fieldnums (largerstr, start, length)"
            );
            let largerstr = decode_fieldnum(ctx, fieldnums[0], rd_virtuals, resume_data, cache);
            let start = decode_fieldnum(ctx, fieldnums[1], rd_virtuals, resume_data, cache);
            let length = decode_fieldnum(ctx, fieldnums[2], rd_virtuals, resume_data, cache);
            // resume.py:1157-1158 / :1185-1186: stopbox = INT_ADD(startbox, lengthbox)
            let stop = ctx.record_op(OpCode::IntAdd, &[start, length]);
            let oopspec = if is_unicode {
                majit_ir::effectinfo::OopSpecIndex::UniSlice
            } else {
                majit_ir::effectinfo::OopSpecIndex::StrSlice
            };
            let string = emit_stroruni_oopspec_call(ctx, oopspec, &[largerstr, start, stop]);
            cache.set_ptr(vidx, string);
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} V{}SliceInfo → OpRef::from_raw({})",
                    vidx,
                    if is_unicode { "Uni" } else { "Str" },
                    string.raw(),
                );
            }
            string
        }
        // resume.py:951/954 getvirtual_ptr direct-indexes rd_virtuals[index];
        // the function preamble already documents this as fail-loud ("not a
        // silent NONE fallback"). An Empty hole here means the resume stream
        // tagged a virtual index that was never assigned a real virtual
        // (encoder/decoder asymmetry) — surface it rather than poison the
        // operand chain with OpRef::NONE.
        majit_ir::RdVirtualInfo::Empty => panic!(
            "materialize_bridge_virtual: null rd_virtuals[{vidx}] (resume.py: null rd_virtuals[index])"
        ),
    }
}

impl JitState for PyreJitState {
    type Meta = PyreMeta;
    type Sym = PyreSym;
    type Env = PyreEnv;

    fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {
        let num_locals = self.local_count();
        let vsd = self.valuestackdepth();
        let slot_types = concrete_slot_types(self.frame, num_locals, vsd);
        // The valuestackdepth → heap array capacity flip is not used here
        // because it activates the broken VableExpansion path. The
        // `capacity` reference uses the pre-flip semantics:
        // array_capacity == self.array_capacity().
        let capacity = self.array_capacity();
        PyreMeta {
            num_locals,
            ns_len: self.namespace_len(),
            valuestackdepth: vsd,
            array_capacity: capacity,
            // virtualizable_gen.rs:24-31 wires `extra_reds = { ec: Ref }` per
            // interp_jit.py:67 `reds = ['frame', 'ec']`, so the per-Sym
            // helpers (vable_collect_jump_args, NUM_EXTRA_REDS) already
            // thread the ec slot. The runtime flag here remains 0 because
            // pypy/module/pypyjit/interp_jit.py:67 PyPyJitDriver reds=
            // ['frame', 'ec']. With vable heap-writeback landed
            // (vable heap-writeback active behind the reduced-LABEL
            // gate), the descriptor activation chain is unblocked.
            // Cluster 2 flip — pairs with `driver_descriptor() = Some(...)`
            // below.
            trace_extra_reds: 1,
            has_virtualizable: self.has_virtualizable_info(),
            slot_types,
        }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        self.extract_live_values(_meta)
            .into_iter()
            .map(|value| match value {
                Value::Int(v) => v,
                Value::Float(v) => v.to_bits() as i64,
                Value::Ref(r) => r.as_usize() as i64,
                Value::Void => 0,
            })
            .collect()
    }

    fn extract_live_values(&self, meta: &Self::Meta) -> Vec<Value> {
        if meta.trace_extra_reds == 1 {
            self.pypyjit_live_values_with_ec(meta)
        } else {
            self.expanded_virtualizable_live_values_with_extra_reds(meta, &[])
        }
    }

    // virtualizable.py:86 read_boxes() + warmstate.py:73 wrap() parity:
    // Array items (locals_cells_stack_w) are GC pointers → RefFrontendOp.
    // No pre-unboxing at function entry. Unboxing happens during tracing
    // via guard_class + getfield_gc_i when arithmetic/compare handlers
    // encounter Ref-typed operands.

    fn live_value_types(&self, meta: &Self::Meta) -> Vec<Type> {
        if meta.trace_extra_reds == 1 {
            Self::pypyjit_live_value_types_with_ec(meta)
        } else {
            Self::expanded_virtualizable_live_value_types_with_extra_reds(meta, &[])
        }
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        if _meta.trace_extra_reds == 1 {
            Self::pypyjit_create_sym(_meta, _header_pc)
        } else {
            let mut sym = PyreSym::new_uninit(OpRef::input_arg_typed(0, Type::Ref));
            // `extra_reds = { ec: Ref }` in virtualizable_gen.rs places ec
            // at OpRef::from_raw(1). `init_vable_indices` shifts the vable static
            // field + array_base OpRefs but does not own the extra_red
            // sym storage; initialize it explicitly. NUM_EXTRA_REDS == 1 is
            // a crate-level const-assertion (see `lib.rs`).
            sym.execution_context = OpRef::input_arg_typed(1, Type::Ref);
            sym.become_active_vable_owner();
            sym.nlocals = _meta.num_locals;
            sym.valuestackdepth = _meta.valuestackdepth;
            // virtualizable.py:44 + interp_jit.py:25-31: all locals_cells_stack_w
            // items are W_Root → Type::Ref. Unboxing happens inside trace opcode
            // handlers (guard_class + getfield_gc_pure_i/_f), not at slot setup.
            sym.symbolic_local_types =
                vec![Type::Ref; _meta.num_locals.min(_meta.slot_types.len())];
            sym.symbolic_stack_types =
                vec![Type::Ref; _meta.slot_types.len().saturating_sub(_meta.num_locals)];
            let stack_only = _meta.vable_stack_only_depth();
            sym.concrete_stack = vec![ConcreteValue::Null; stack_only];
            sym
        }
    }

    fn initialize_sym(&self, sym: &mut Self::Sym, _meta: &Self::Meta) {
        // jitdriver.rs:2947-2992 sequencing parity: `start_bridge_tracing`
        // calls `create_sym` → `initialize_sym` → `setup_bridge_sym` BEFORE
        // `trace_bytecode` (which is where `init_symbolic` would otherwise
        // populate `concrete_vable_ptr`). `setup_bridge_sym` reads
        // `sym.concrete_vable_ptr` to size the vable shadow via
        // `concrete_frame_array_len`; without this seed the pointer is
        // null, the helper falls back to `nlocals`, and stack pushes
        // beyond the first array slot panic at `set_virtualizable_entry_at`
        // on recursive benchmarks (fib_recursive). Mirror PyreJitState's
        // live frame pointer so the bridge seed sees the real
        // `locals_cells_stack_w` length.
        sym.concrete_vable_ptr = self.frame as *mut u8;
    }

    fn driver_descriptor(&self, _meta: &Self::Meta) -> Option<JitDriverStaticData> {
        // pypy/module/pypyjit/interp_jit.py:67-70 PyPyJitDriver:
        //   reds = ['frame', 'ec']
        //   greens = ['next_instr', 'is_being_profiled', 'pycode']
        //   virtualizables = ['frame']
        //
        // History (now landed): the atomic flip needed a vable
        // heap-writeback pass that pyre's tracer originally did not
        // emit. Concretely,
        // `patch_new_loop_to_load_virtualizable_fields` (compile.py:425-461)
        // collapses the patched LABEL to `[reds]` and prepends a
        // GETFIELD_GC + GETARRAYITEM_GC preamble. The body then reads
        // every vable static field and locals_cells_stack_w slot from
        // the frame heap object on each iteration. pyre's tracer
        // updates symbolic state (`sym.vable_*`, `sym.registers_r[i]`)
        // but never emits the matching `SetfieldGc(frame, value, descr)`
        // or `SetarrayitemGc(array_ref, idx, value)` — only the
        // freshly-allocated W_IntObject.intval setfields appear in
        // trace dumps. With descriptor=None, this is fine because the
        // closing JUMP carries the post-loop state via expanded inputargs;
        // the patched-LABEL contract makes the heap the source of truth
        // and breaks that invariant.
        //
        // Prerequisites for the flip (all atomic with descriptor=Some):
        //   (a) `trace_extra_reds=1` (build_meta:3970) — emit
        //       `live_values=[frame, ec]` matching descriptor reds. The
        //       legacy `extract_live_values` shape relies on
        //       `descriptor=None`; flipping (a) alone breaks dynasm
        //       nested_loop / fannkuch / nbody (live_values consumers
        //       expect the expanded shape) and panics on
        //       `live_values[index>1]` access without (b).
        //   (b) `initialize_virtualizable` short-live_values gate
        //       (pyjitpl/mod.rs:1744): allow the heap-read branch when
        //       `vable_ptr` is non-null. Required to consume (a)'s
        //       reds-only live_values.
        //   (c1) `pending_frontend_boxes.clone()` (pyjitpl/mod.rs:7338):
        //        the second `compile_bridge` call needs the same stash
        //        as the first; current `take()` empties on the first
        //        call and the second hits `frontend_boxes.len()=0 vs
        //        liveboxes.len()=N`.
        //   (c2) Bridge JUMP arity vs patched LABEL: source emits the
        //        live-window shape (`num_scalars + valuestackdepth`) but
        //        the patched LABEL grows to full vable capacity. Const
        //        padding SSA-forwards `PY_NULL` into outer-loop locals
        //        on cranelift; heap-read padding via
        //        `trace_array_getitem_value` against
        //        `locals_cells_stack_array_ref` is the correct shape.
        //   (d) **Vable heap-writeback infrastructure** — *blocking*. With
        //       (a)+(b)+(c1) applied, cranelift nested_loop dispatches
        //       the bridge 5622+ times with identical inputs `[fr, ec,
        //       500, 2507475000]` because the patched parent reloads
        //       stale state from heap each iteration; the bridge body
        //       has no `SetfieldGc`/`SetarrayitemGc` to advance the
        //       heap. RPython's OptVirtualize emits these via its
        //       `force_at_end_of_preamble` pass; pyre's
        //       `OptVirtualize::force_virtualizable` (virtualize.rs:348)
        //       has the SETFIELD_RAW emission machinery but is not
        //       wired to fire at JUMP. The smallest-delta fix is a
        //       new `gen_writeback_vable_to_heap` helper (modeled on
        //       `trace_ctx.rs:1702 gen_store_back_in_vable` minus the
        //       force-virtualizable bookkeeping) invoked from
        //       `close_loop_args_at`.
        //   (e) dynasm recursive CA frame contract — *blocking*
        //       for dynasm SIGSEGV at fib(24).
        //
        // Status: (a)+(b)+(c1)+(c2) and
        // gen_writeback_vable_to_heap have landed; descriptor is
        // active. (e) dynasm recursive CA frame contract
        // remains as a separate open item driving fib_recursive on
        // dynasm.
        Some(Self::pypyjit_driver_descriptor())
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        // warmstate.py:503-511: RPython enters assembler unconditionally
        // when procedure_token exists. No next_instr check —
        // the compiled code's preamble handles entry from any PC.
        // Shape checks (nlocals, namespace) ensure frame layout matches.
        self.local_count() == meta.num_locals && self.namespace_len() == meta.ns_len
    }

    fn setup_bridge_sym(
        sym: &mut Self::Sym,
        ctx: &mut majit_metainterp::TraceCtx,
        resume_data: &majit_metainterp::ResumeDataResult,
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        fail_values: &[i64],
        fail_types: &[Type],
    ) {
        if resume_data.frames.is_empty() {
            return;
        }

        // virtualizable.py:139 load_list_of_boxes parity: decode each
        // RebuiltValue in the resume stream into a typed Value. The type
        // is the fixed Box kind the encoder recorded at numbering time
        // (fail_arg_types[idx] for Box variants), matching RPython's
        // immutable Box.type invariant. No heap read.

        // resume.py:874-899 VirtualCache parity: single cache indexed by
        // virtual number, holding both symbolic OpRef (for trace ops) and
        // concrete GcRef (for shadow values / continue_tracing). RPython's
        // VirtualCache stores both in one object; pyre unifies them here.
        let mut virtuals_cache = BridgeVirtualCache::new(rd_virtuals.map_or(0, |v| v.len()));
        let (driver, _) = crate::driver::driver_pair();
        let backend = driver.meta_interp().backend();

        // resume.py:1245 decode_box parity — unified via bridge_decode_box.
        // Each call returns (OpRef, Value), eliminating the separate
        // resolve()/decode_concrete() paths so symbolic and concrete
        // materialization are always invoked together.

        let nlocals = sym.nlocals;
        // virtualizable.py:44 + interp_jit.py:25-31: locals_cells_stack_w[*]
        // items are declared Ref (W_Root array). Bridge resume slots stay
        // Ref at the virtualizable contract; any Int/Float unboxing must
        // happen inside trace opcode handlers, not at the inputarg level.
        let bridge_local_types = vec![Type::Ref; nlocals];

        // rebuild_state_after_failure (pyjitpl.py:3400-3437) keeps three
        // streams apart:
        //   virtualizable_boxes ← consume_virtualizable_boxes (vable mirror)
        //   registers_r/_i/_f    ← consume_boxes(f.get_current_position_info(),
        //                                        f.registers_i/_r/_f)
        //   virtualref_boxes     ← consume_virtualref_boxes
        // The majit decoder already splits rd_numb into the same three
        // streams (`resume_data.virtualizable_values`,
        // `resume_data.frames[*].values`, `resume_data.virtualref_values`).
        // This function consumes them in the same order and purpose.
        //
        // Part 1 — virtualizable payload (consume_virtualizable_boxes):
        // decode scalar header into sym.vable_* and capture the array-item
        // prefix for init_virtualizable_boxes below. Matches
        // virtualizable.py:86 `read_boxes` layout
        //   [vable_ptr, static_fields..., array_items...].
        let vvals = &resume_data.virtualizable_values;
        // Resume virtualizable payload mirrors RPython
        // opencoder.py:718-725 + virtualizable.py:139-154:
        //   [vable, vable_static_fields..., array_items...]
        // Non-vable extra reds (e.g. `ec`) are root inputargs, not part of
        // this payload, so the boundary is NUM_VABLE_SCALARS, not
        // NUM_SCALAR_INPUTARGS.
        let first_vable_scalar_idx = 1usize;
        let vable_array_start =
            first_vable_scalar_idx + crate::virtualizable_gen::NUM_VABLE_SCALARS;
        let mut oprefs: Vec<OpRef> = Vec::with_capacity(vvals.len());
        let mut concrete_values: Vec<majit_ir::Value> = Vec::with_capacity(vvals.len());
        // resume.py:1264 `assert box.type == kind`: the vable payload is
        // NOT uniformly Ref — the static fields carry their declared
        // kinds (interp_jit.py:25-31: last_instr/valuestackdepth are Int).
        // `virt_live_value_types` yields the full live layout WITH the
        // extra reds ([frame, <NUM_EXTRA_REDS>, <NUM_VABLE_SCALARS>,
        // array...]); the vvals stream omits the extra reds, so strip them
        // to recover the per-slot kind for each payload position.
        let array_item_count = vvals
            .len()
            .saturating_sub(1 + crate::virtualizable_gen::NUM_VABLE_SCALARS);
        let full_types = crate::virtualizable_gen::virt_live_value_types(array_item_count);
        let nreds = crate::virtualizable_gen::NUM_EXTRA_REDS;
        let mut vvals_types: Vec<Type> = Vec::with_capacity(vvals.len());
        vvals_types.push(full_types.first().copied().unwrap_or(Type::Ref));
        vvals_types.extend_from_slice(&full_types[(1 + nreds).min(full_types.len())..]);
        for (idx, v) in vvals.iter().enumerate() {
            let expected_kind = vvals_types.get(idx).copied().unwrap_or(Type::Ref);
            let (op, val) = bridge_decode_box(
                ctx,
                v,
                expected_kind,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                &mut virtuals_cache,
            );
            oprefs.push(op);
            concrete_values.push(val);
        }
        sym.restore_inputarg_oprefs(&oprefs, first_vable_scalar_idx);
        let vable_ref_value = concrete_values
            .first()
            .copied()
            .unwrap_or(majit_ir::Value::Void);
        let vable_scalar_values: Vec<majit_ir::Value> = concrete_values
            .iter()
            .skip(first_vable_scalar_idx)
            .take(crate::virtualizable_gen::NUM_VABLE_SCALARS)
            .copied()
            .collect();
        let vable_array_items: Vec<OpRef> =
            oprefs.iter().skip(vable_array_start).copied().collect();
        let vable_array_values: Vec<majit_ir::Value> = concrete_values
            .iter()
            .skip(vable_array_start)
            .copied()
            .collect();
        let bridge_valuestackdepth = concrete_values
            // virtualizable_values has no ec red: [vable, last_instr,
            // pycode, valuestackdepth, debugdata, lastblock, w_globals, ...].
            .get(first_vable_scalar_idx + 2)
            .map(value_to_usize)
            .unwrap_or(sym.valuestackdepth)
            .max(nlocals);

        // Part 2 — frame registers (consume_boxes): walk the frame section
        // in liveness enumeration order ([int..., ref..., float...]), keep
        // each bank's register indices separate via
        // `frame_liveness_reg_indices_by_bank_at`, and write each decoded
        // value into the corresponding MIFrame register bank. This mirrors
        // resume.py:1054 `consume_boxes`
        // (`_callback_i/_r/_f(register_index)` writing to
        // `f.registers_i/_r/_f[index]` at the exact slot liveness
        // declared, not at an enumerate-order position). RPython indexes
        // a single `registers_X` vector by abstract register color —
        // there is no `idx < nlocals` decode.
        let frame0 = &resume_data.frames[0];
        let reg_indices =
            crate::state::frame_liveness_reg_indices_by_bank_at(frame0.jitcode_index, frame0.pc);
        let stack_only = bridge_valuestackdepth.saturating_sub(nlocals);
        let bridge_reg_len = nlocals + stack_only;
        let mut bridge_registers_r = vec![OpRef::NONE; bridge_reg_len];
        // RPython parity: after A.1 the guard-recovery path calls
        // `synchronize_virtualizable()` / `write_boxes()`
        // (pyjitpl.py:3430) before `start_bridge_tracing`, so the
        // physical vable image the tracer is about to read is already
        // resume-data-complete. The
        // bridge register file is therefore expected to be fully
        // populated by the liveness-driven zip below; any remaining
        // OpRef::NONE signals a liveness-coverage gap (the tracer keeps a
        // local live past the `-live-` marker) and must be surfaced by
        // the assert rather than papered over with a vable-mirror read.
        assert!(
            reg_indices.total_len() == frame0.values.len(),
            "setup_bridge_sym: reg_indices len={} != frame.values len={} at pc={}",
            reg_indices.total_len(),
            frame0.values.len(),
            frame0.pc,
        );
        let mut value_cursor = 0usize;
        for &reg_idx in &reg_indices.int {
            let value = &frame0.values[value_cursor];
            let (resolved, _) = bridge_decode_box(
                ctx,
                value,
                Type::Int,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                &mut virtuals_cache,
            );
            let reg_idx = reg_idx as usize;
            if reg_idx >= sym.registers_i.len() {
                sym.registers_i.resize(reg_idx + 1, OpRef::NONE);
            }
            sym.registers_i[reg_idx] = resolved;
            value_cursor += 1;
        }
        for &reg_idx in &reg_indices.ref_ {
            let value = &frame0.values[value_cursor];
            let (resolved, _) = bridge_decode_box(
                ctx,
                value,
                Type::Ref,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                &mut virtuals_cache,
            );
            let reg_idx = reg_idx as usize;
            if reg_idx >= bridge_registers_r.len() {
                bridge_registers_r.resize(reg_idx + 1, OpRef::NONE);
            }
            bridge_registers_r[reg_idx] = resolved;
            value_cursor += 1;
        }
        for &reg_idx in &reg_indices.float {
            let value = &frame0.values[value_cursor];
            let (resolved, _) = bridge_decode_box(
                ctx,
                value,
                Type::Float,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                &mut virtuals_cache,
            );
            let reg_idx = reg_idx as usize;
            if reg_idx >= sym.registers_f.len() {
                sym.registers_f.resize(reg_idx + 1, OpRef::NONE);
            }
            sym.registers_f[reg_idx] = resolved;
            value_cursor += 1;
        }
        let semantic_prefix_len = nlocals + stack_only;
        for (idx, slot) in bridge_registers_r
            .iter_mut()
            .enumerate()
            .take(semantic_prefix_len)
        {
            // virtualizable.py:86-98 + pyjitpl.py:3430 synchronize_virtualizable:
            // the frame's `locals_cells_stack_w` array (the vable image) is the
            // authoritative post-guard source for the frame's locals. The
            // per-frame liveness register reconstruction (`frame.values`) is
            // only canonicalized-to-locals at a merge point; at an arbitrary
            // interior resume pc a local slot's jitcode color may hold a dead
            // temp that decodes to a NULL constant (`ConstPtr(GcRef(0))`).
            // Such a dead-temp NULL must not shadow the vable's live local, so
            // for local slots prefer the vable array item over a NONE or
            // null-const register value. Stack slots keep the NONE-only
            // fallback (a real stack value must not be overwritten).
            let is_local = idx < nlocals;
            let slot_is_null_const = matches!(*slot, OpRef::ConstPtr(v) if v.0 == 0);
            let want_vable = slot.is_none() || (is_local && slot_is_null_const);
            if want_vable {
                if let Some(v) = vable_array_items.get(idx).copied() {
                    if !v.is_none() {
                        *slot = v;
                    } else if slot.is_none() {
                        *slot = OpRef::NONE;
                    }
                }
            }
        }
        let bridge_locals: Vec<OpRef> = bridge_registers_r.iter().take(nlocals).copied().collect();

        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-sym] frames[0].values={} reg_indices={:?} \
                 bridge_locals={:?} vable_array_items={:?}",
                frame0.values.len(),
                reg_indices,
                bridge_locals,
                vable_array_items,
            );
        }
        // Override sym.registers_r so subsequent LOAD_FAST sees the
        // bridge inputarg OpRefs, not the parent's vable_array_base+i
        // OpRefs that init_symbolic seeded before setup_bridge_sym ran.
        //
        // pyre's start_bridge_tracing calls initialize_sym() (which runs
        // init_symbolic) BEFORE setup_bridge_sym, so init_symbolic sees
        // bridge_local_oprefs == None and falls into the vable_array_base
        // branch (init_vable_indices hard-codes vable_array_base = 7 for
        // pyre's 7-slot virtualizable header). That branch produces
        // OpRef::from_raw(base+i) values from the PARENT trace's namespace, leaving
        // stale parent OpRefs in registers_r after we set
        // bridge_local_oprefs here.
        //
        // virtualizable.py:44 + interp_jit.py:25-31: array item types are
        // all Ref; RETURN_VALUE / arithmetic paths unbox via
        // `trace_guarded_int_payload` (guard_class + getfield_gc_pure_i),
        // matching the RPython unbox-at-consumer model. The slot-level
        // type override is NOT how RPython avoids the guarded path.
        sym.registers_r = bridge_registers_r.clone();
        sym.symbolic_local_types = {
            let mut types = bridge_local_types.clone();
            types.resize(sym.nlocals, Type::Ref);
            types
        };
        // The bridge inputs do NOT have the 7-slot scalar header that
        // init_vable_indices assumes. Demote this frame from active
        // virtualizable owner so any later LOAD_FAST falling through to
        // the vable_array_base branch uses the heap-array path instead
        // of synthesizing parent OpRefs.
        sym.clear_active_vable();
        // `pypy/module/pypyjit/interp_jit.py:67 reds = ['frame', 'ec']`:
        // ec is a portal red arg, hence a JitCode inputarg present in
        // every `-live-` op's R-bank. Because the codewriter
        // (jit/codewriter.rs:2364 `filter_liveness_in_place`) seeds
        // `portal_ec_reg` into `lv_live`, bridge resume's liveness-
        // driven `consume_boxes` fill at lines 4880-4893 has already
        // written the resolved ec OpRef into
        // `bridge_registers_r[portal_ec_reg]` — the same slot
        // `_callback_r(register_index)` (resume.py:1077-1081) writes
        // in RPython's BH register bank.
        //
        // Thread that slot back into the dedicated `sym.execution_context`
        // field so the symbolic state retains pyre's split shape (ec
        // accessed through a named field rather than by register index).
        // The OpRef value is identical to what RPython's ec inputarg
        // OpRef would be after resume.
        let (_pfr, portal_ec_reg) = crate::state::portal_red_regs_at(frame0.jitcode_index);
        if portal_ec_reg != u16::MAX {
            let slot = portal_ec_reg as usize;
            assert!(
                slot < sym.registers_r.len(),
                "setup_bridge_sym: portal_ec_reg={} out of registers_r range (len={})",
                slot,
                sym.registers_r.len(),
            );
            sym.execution_context = sym.registers_r[slot];
        }
        // pyjitpl.py:3400-3430 rebuild_state_after_failure parity: after
        // a guard failure the tracing-time `virtualizable_boxes` mirror
        // must be rebuilt from the resume data so subsequent vable
        // ops see OpRefs drawn from the bridge's inputarg stream, not
        // the parent loop's vable_array_base+i indices that
        // init_symbolic seeded before setup_bridge_sym ran.
        //
        // Layout mirrors virtualizable.py:86-98 read_boxes():
        //   boxes[0..NUM_SCALARS-1] = scalar fields 1..NUM_SCALARS
        //     (vable_last_instr, vable_pycode, vable_valuestackdepth,
        //      vable_debugdata, vable_lastblock, vable_w_globals)
        //   boxes[NUM_SCALARS-1..NUM_SCALARS-1+array_len] = array items
        //     (bridge_locals followed by reserved stack slots)
        //   boxes[-1] = vable identity (sym.frame)
        // pyframe.py:107-110 `locals_cells_stack_w` length =
        // `nlocals + ncells + max_stack`. Pad beyond the bridge's live
        // local prefix with a shared const-NULL OpRef so every
        // interpreter-visible slot has a tracing-time mirror (matches
        // the portal path above).
        // pyframe.py:107-110 + pyjitpl.py:3437: the virtualizable shape
        // committed at portal-entry time (`nlocals + ncells +
        // max_stackdepth` array slots) does not change at guard-failure
        // resume; `rebuild_state_after_failure` writes the resume blob
        // into the same `virtualizable_boxes` layout the portal seeded.
        // pyre's root portal seeds via `initialize_virtualizable` with
        // exactly that full layout, so the bridge entry must match it
        // here. Earlier `unwrap_or(nlocals)` undersized the shadow when
        // `concrete_frame_array_len` returned None (e.g. when
        // `sym.concrete_vable_ptr` had not yet been bound at
        // setup_bridge_sym time), causing pushes past `nlocals`/the
        // local prefix to panic at `set_virtualizable_entry_at: index N
        // out of range for N slots`. A probe captured the
        // mismatch directly: root portal sized `vable_boxes_len=25`
        // (= 6 + 18 + 1) but a fannkuch bridge fell back to
        // `bridge_array_len=14` → `vable_boxes_len=21`, then pushed
        // `flat_idx=21` and panicked. Fall back to the metadata-derived
        // size — `metadata.stack_base + metadata.stack_slot_color_map
        // .len()` is the same `nlocals + ncells + max_stackdepth` the
        // codewriter committed to and the runtime PyFrame allocates
        // (pyframe.rs:1576).
        let bridge_array_len = concrete_frame_array_len(sym.concrete_vable_ptr as usize)
            .or_else(|| {
                METAINTERP_SD.with(|r| {
                    let sd = r.borrow();
                    sd.jitcodes.get(frame0.jitcode_index as usize).map(|jc| {
                        jc.payload.metadata.stack_base
                            + jc.payload.metadata.stack_slot_color_map.len()
                    })
                })
            })
            .unwrap_or(nlocals);
        let scalar_oprefs = [
            sym.vable_last_instr,
            sym.vable_pycode,
            sym.vable_valuestackdepth,
            sym.vable_debugdata,
            sym.vable_lastblock,
            sym.vable_w_globals,
        ];
        // virtualizable.py:139 load_list_of_boxes parity: both halves of
        // virtualizable_boxes come from the resume-data stream — OpRefs via
        // bridge_decode_box returns (OpRef, Value) pairs.
        // No heap read. The seed helper pads short arrays with const-NULL
        // OpRef; match that here by padding concrete values with
        // Value::Ref(GcRef::NULL) to the same length.
        let mut concrete_values = Vec::with_capacity(vable_scalar_values.len() + bridge_array_len);
        concrete_values.extend_from_slice(&vable_scalar_values);
        let taken_concrete = vable_array_values.len().min(bridge_array_len);
        concrete_values.extend_from_slice(&vable_array_values[..taken_concrete]);
        while concrete_values.len() < vable_scalar_values.len() + bridge_array_len {
            concrete_values.push(majit_ir::Value::Ref(majit_ir::GcRef::NULL));
        }
        crate::state::seed_virtualizable_boxes(
            ctx,
            sym.frame,
            vable_ref_value,
            &scalar_oprefs,
            &vable_array_items,
            bridge_array_len,
            &concrete_values,
            sym.concrete_vable_ptr as *const u8,
        );
        // resume.py:1042-1057 `rebuild_from_resumedata` parity: bridge
        // tracing resumes from the full restored frame state via the
        // vable scalar reads + consume_boxes — `valuestackdepth` is
        // recovered from the decoded virtualizable resume payload
        // (`bridge_valuestackdepth` derived from vable.valuestackdepth), not
        // hardcoded to `nlocals`. Keep the stack tail in the unified
        // register file and expose Ref-typed virtualizable slots to
        // subsequent LOAD_FAST / close_loop_args_at calls.
        sym.symbolic_stack_types = vec![Type::Ref; stack_only];
        sym.valuestackdepth = bridge_valuestackdepth;
        sym.bridge_local_oprefs = Some(bridge_locals);
        sym.bridge_local_types = Some(bridge_local_types);

        // pyjitpl.py:3424 `rebuild_state_after_failure` tail —
        // `consume_virtualref_boxes` (resume.py:1093):
        //   for i in range(0, len(virtualref_boxes), 2):
        //       virtual_box = virtualref_boxes[i]
        //       vref_box    = virtualref_boxes[i + 1]
        //   self.virtualref_boxes += [virtual_box, vref_box]
        // The parent guard already encoded the resumed pair sequence into
        // `rd_numb`'s vref section (resume.py:738-754 `consume_vref_and_vable`
        // → `consume_virtualref_info`), and `rebuild_from_numbering`
        // (majit/majit-ir/src/resumedata.rs:402-416) decoded it into
        // `resume_data.virtualref_values`. Materialize the OpRef + concrete
        // pointer for each pair through the same `resolve` / `decode_concrete`
        // callbacks used for `frames[0].values`, then push the pairs into
        // `sym.virtualref_boxes` so `opimpl_virtual_ref` /
        // `opimpl_virtual_ref_finish` handlers (state.rs:3475-3500) observe
        // the parent's still-open virtualref scope at bridge entry.
        //
        // For traces with no active `virtual_ref`, `virtualref_values` is
        // empty and this loop is a no-op.
        let vref_values = &resume_data.virtualref_values;
        // resume.py:1397 `vrefinfo.continue_tracing(vref, virtual_val)` —
        // restore `JitVirtualRef.forced = virtual` and clear
        // `virtual_token` for every still-active pair. The runtime
        // mutation lives on the live JitVirtualRef heap object, so the
        // call is gated on a non-null concrete `vref_ptr`. Pyre's
        // `VirtualRefInfo::new()` (majit-metainterp/src/virtualref.rs:257)
        // is a cheap struct-only constructor — instantiate once outside
        // the loop instead of per-pair.
        let vrefinfo = majit_metainterp::virtualref::VirtualRefInfo::new();
        // resume.py:1093 `consume_virtualref_boxes` decodes exactly
        // `size * 2` entries; a malformed odd-length stream is a bug.
        assert!(
            vref_values.len() % 2 == 0,
            "virtualref_values must contain an even number of entries (got {})",
            vref_values.len(),
        );
        for pair in vref_values.chunks_exact(2) {
            let (virt_opref, virt_val) = bridge_decode_box(
                ctx,
                &pair[0],
                Type::Ref,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                &mut virtuals_cache,
            );
            let (vref_opref, vref_val) = bridge_decode_box(
                ctx,
                &pair[1],
                Type::Ref,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                &mut virtuals_cache,
            );
            let virt_ptr = value_to_usize(&virt_val);
            let vref_ptr = value_to_usize(&vref_val);
            sym.virtualref_boxes.push((virt_opref, virt_ptr));
            sym.virtualref_boxes.push((vref_opref, vref_ptr));
            // pyjitpl.py:3438 / resume.py:1397: continue_tracing is called
            // unconditionally for every (vref, real_object) pair. The
            // is_virtual_ref(vref) guard (virtualref.py:123) and the
            // `assert real_object` invariant (virtualref.py:125, ported as
            // debug_assert!) both live inside continue_tracing itself
            // (virtualref.rs:419/424) — the outer virt_ptr guard masked
            // exactly the case RPython asserts on, so do not pre-gate here.
            unsafe {
                vrefinfo.continue_tracing(vref_ptr as *mut u8, virt_ptr as *mut u8);
            }
        }

        // pyjitpl.py:3443 `synchronize_virtualizable()` inside
        // `rebuild_state_after_failure` (pyjitpl.py:3454) writes
        // `virtualizable_boxes` to the heap via `write_boxes()`
        // (virtualizable.py:101-113). This is the ONLY call in
        // RPython's bridge-resume path — there is no second call.
        //
        // In pyre, the equivalent write is `sync_virtualizable_after_
        // guard_failure` (eval.rs:5709, `ResumeVableMode::
        // GuardFailureSync`) which runs in the compiled bridge's
        // guard-failure recovery chain BEFORE `setup_bridge_sym`.
        // No additional synchronize call is needed here.

        // Multi-frame bridge. The body above reconstructed
        // the portal (`frames[0]`) into the caller-visible root `sym`. When
        // the guard fired inside inlined callees, `frames[1..]` (OUTERMOST-
        // FIRST) must also be reconstructed and pushed so the framestack
        // matches the inline depth (`rebuild_from_resumedata` resume.py:1049-
        // 1056: every frame via `newframe`+`consume_boxes`, then tracing
        // continues at `framestack[-1]`). Decode each into a lightweight
        // recipe here (while `resume_data` / `virtuals_cache` are in scope);
        // `trace_bytecode` assembles+pushes them right before `interpret()`.
        // Any callee that cannot be faithfully rebuilt aborts the whole
        // multi-frame path back to the single-frame bridge.
        // The outermost (root) frame resumes at its OWN pc once the
        // reconstructed callees return — NOT at the innermost pc that
        // `decode_and_restore_guard_failure` returns as the trace start.
        // Thread `frames[0].pc` so `trace_bytecode` can root the outermost
        // frame at its own pc while the carrier callees resume at their
        // frames[i].pc. A negative (no-snapshot) root pc aborts the
        // multi-frame path.
        if resume_data.frames.len() > 1 && resume_data.frames[0].pc >= 0 {
            let root_pc = resume_data.frames[0].pc as usize;
            let mut recipes: Vec<ReconstructRecipe> =
                Vec::with_capacity(resume_data.frames.len() - 1);
            let mut ok = true;
            for frame in &resume_data.frames[1..] {
                match reconstruct_inline_recipe(
                    ctx,
                    frame,
                    rd_virtuals,
                    resume_data,
                    fail_values,
                    fail_types,
                    backend,
                    &mut virtuals_cache,
                ) {
                    Some(recipe) => recipes.push(recipe),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && !recipes.is_empty() {
                ctx.set_bridge_inline_carrier(BridgeInlineCarrier { root_pc, recipes });
            }
        }
    }

    /// resume.py:1042-1057 rebuild_from_resumedata parity.
    ///
    /// Decodes rd_numb via `majit_ir::resumedata::rebuild_from_numbering`.
    /// Frame box counts come from jitcode liveness (jitcode.position_info)
    /// at the frame's resume pc — the same data the encoder uses via
    /// `get_list_of_active_boxes`.
    fn rebuild_from_resumedata(
        _meta: &mut Self::Meta,
        fail_arg_types: &[Type],
        storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    ) -> Option<majit_metainterp::ResumeDataResult> {
        use majit_ir::resumedata::rebuild_from_numbering;

        // interp_jit.py:67-74 PyPyJitDriver portal reds are `[frame, ec]`;
        // virtualizable boxes are appended later by
        // MetaInterp::initialize_virtualizable(), not carried as reds.
        _meta.trace_extra_reds = 1;
        let storage = storage?;
        let rd_numb = storage.rd_numb.as_slice();
        // resume.py:1071 `self.consts = storage.rd_consts` — borrow
        // the shared pool; `ResumeDataResult` carries the Arc handle
        // so downstream virtual materialization reads the same pool
        // the GC walker updates.
        let rd_consts = storage.rd_consts();

        // resume.py:1049-1055 parity: consume_boxes(f.get_current_position_info())
        // RPython uses jitcode liveness via get_current_position_info; majit
        // routes the same lookup through `frame_value_count_at`.
        let cb = crate::state::frame_value_count_at;
        let (num_failargs, vable_values, vref_values, frames) =
            rebuild_from_numbering(rd_numb, rd_consts, fail_arg_types, Some(&cb));

        if frames.is_empty() {
            return None;
        }

        Some(majit_metainterp::ResumeDataResult {
            frames,
            virtualizable_values: vable_values,
            virtualref_values: vref_values,
            storage: Some(storage.clone()),
            // resume.py:1042 num_failargs from rd_numb header. Used by
            // bridge virtual materialization (resume.py:1556-1564 decode_box
            // negative-index normalization: `num + len(liveboxes)`).
            num_failargs,
            // compile.py:797 ResumeGuardDescr.fail_arg_types — kept so
            // `materialize_bridge_virtual::decode_fieldnum` can mint typed
            // `OpRef::input_arg_typed(idx, fail_arg_types[idx])` per
            // resume.py:1247-1264 (`return self.liveboxes[num]` whose Box
            // carries history.py:220 `box.type`).
            fail_arg_types: fail_arg_types.to_vec(),
        })
    }

    /// pyjitpl.py:2982 get_procedure_token: compute green key for a PC.
    fn green_key_for_pc(&self, pc: usize) -> Option<u64> {
        let frame_ptr = self.frame as *const pyre_interpreter::pyframe::PyFrame;
        if frame_ptr.is_null() {
            return None;
        }
        let code = unsafe { (*frame_ptr).pycode };
        Some(crate::driver::make_green_key(code, pc))
    }

    fn code_ptr(&self) -> usize {
        let frame_ptr = self.frame as *const pyre_interpreter::pyframe::PyFrame;
        if frame_ptr.is_null() {
            return 0;
        }
        unsafe { (*frame_ptr).pycode as usize }
    }

    fn update_meta_for_cut(meta: &mut Self::Meta, _header_pc: usize, original_box_types: &[Type]) {
        // Current pyre cut traces still recover only the expanded array-item
        // count from the merge-point box vector. Preserve the provisional live
        // depth and update just the heap-array capacity until the later
        // red-only/virtualizable-box migration lands.
        // `NUM_SCALAR_INPUTARGS` already counts `NUM_EXTRA_REDS` (frame +
        // extra_reds + vable scalars), so do NOT add `trace_extra_reds`.
        use crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let _ = meta.trace_extra_reds; // staging copy retained for activation gates only
        if original_box_types.len() >= NUM_SCALAR_INPUTARGS {
            let new_capacity = original_box_types
                .len()
                .saturating_sub(NUM_SCALAR_INPUTARGS);
            if meta.slot_types.len() > new_capacity {
                meta.slot_types.truncate(new_capacity);
            } else if meta.slot_types.len() < new_capacity {
                meta.slot_types.resize(new_capacity, Type::Ref);
            }
            meta.array_capacity = new_capacity;
        }
    }

    fn build_meta_from_merge_point(
        provisional: &PyreMeta,
        _header_pc: usize,
        original_boxes: &[majit_metainterp::GreenBox],
    ) -> PyreMeta {
        let original_box_count = original_boxes.len();
        // `NUM_SCALAR_INPUTARGS` already counts `NUM_EXTRA_REDS` (frame +
        // extra_reds + vable scalars), so do NOT add `trace_extra_reds`.
        use crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let _ = provisional.trace_extra_reds; // staging copy only
        // RPython parity: Python locals/stack are always Ref.
        let slot_types = if original_box_count >= NUM_SCALAR_INPUTARGS {
            vec![Type::Ref; original_box_count.saturating_sub(NUM_SCALAR_INPUTARGS)]
        } else {
            Vec::new()
        };
        let array_capacity = if original_box_count >= NUM_SCALAR_INPUTARGS {
            original_box_count.saturating_sub(NUM_SCALAR_INPUTARGS)
        } else {
            provisional.array_capacity
        };
        PyreMeta {
            num_locals: provisional.num_locals,
            ns_len: provisional.ns_len,
            valuestackdepth: provisional.valuestackdepth,
            array_capacity,
            trace_extra_reds: provisional.trace_extra_reds,
            has_virtualizable: provisional.has_virtualizable,
            slot_types,
        }
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        if values.is_empty() {
            return;
        }

        // Multi-frame format: [num_frames, size_0, data_0..., size_1, data_1...]
        // Detect: values[0] is a small number (1-10) = frame count
        // Legacy: values[0] is a large pointer
        let first = values[0];
        if first >= 1 && first <= 10 && values.len() > 2 {
            let _num_frames = first as usize;
            let outer_size = values[1] as usize;
            if outer_size > 0 && 2 + outer_size <= values.len() {
                // Restore outermost frame only.
                // Inner frame guard failure → interpreter re-executes callee call.
                self.restore_single_frame(meta, &values[2..2 + outer_size]);
                return;
            }
        }

        // Legacy single-frame format
        self.restore_single_frame(meta, values);
    }

    fn restore_values(&mut self, meta: &Self::Meta, values: &[Value]) {
        let Some(_frame) = values.first() else {
            return;
        };
        if majit_metainterp::majit_log_enabled() {
            let arg0 = self.local_at(0).and_then(|value| {
                if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) } {
                    return None;
                }
                Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
            });
            eprintln!(
                "[jit][restore_values] before arg0={:?} meta.vsd={} has_vable={} values={:?}",
                arg0, meta.valuestackdepth, meta.has_virtualizable, values
            );
        }
        self.restore_expanded_virtualizable_values_with_extra_reds(
            meta,
            values,
            meta.trace_extra_reds,
        );
        if majit_metainterp::majit_log_enabled() {
            let arg0 = self.local_at(0).and_then(|value| {
                if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) } {
                    return None;
                }
                Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
            });
            eprintln!(
                "[jit][restore_values] after arg0={:?} ni={} vsd={}",
                arg0,
                self.next_instr(),
                self.valuestackdepth()
            );
        }
    }

    fn restore_guard_failure_values(
        &mut self,
        meta: &Self::Meta,
        values: &[Value],
        _exception: &majit_metainterp::blackhole::ExceptionState,
    ) -> bool {
        if !meta.has_virtualizable {
            self.restore_values(meta, values);
            return true;
        }

        let Some(frame) = values.first() else {
            return false;
        };
        self.frame = value_to_usize(frame);
        if values.len() == 1 {
            return self.validate_frame();
        }

        // virtualizable.py:126-137 write_from_resume_data_partial:
        // ALL static fields in unroll_static_fields order.
        if let Some(last_instr) = values
            .get(crate::virtualizable_gen::SYM_LAST_INSTR_IDX as usize)
            .map(value_to_usize)
        {
            self.set_last_instr(last_instr);
        }
        if let Some(code) = values
            .get(crate::virtualizable_gen::SYM_PYCODE_IDX as usize)
            .map(value_to_usize)
        {
            self.set_pycode(code);
        }
        if let Some(vsd) = values
            .get(crate::virtualizable_gen::SYM_VALUESTACKDEPTH_IDX as usize)
            .map(value_to_usize)
        {
            // Sanity check: vsd must not exceed the frame's total capacity
            // (nlocals + stacksize). A bad vsd from stale guard recovery
            // values can corrupt the frame and crash in as_mut_slice.
            let max_vsd = self
                .locals_cells_stack_array()
                .map(|arr| arr.len())
                .unwrap_or(0);
            let safe_vsd = vsd.min(max_vsd);
            self.set_valuestackdepth(safe_vsd);
        }
        if let Some(ns) = values
            .get(crate::virtualizable_gen::SYM_W_GLOBALS_IDX as usize)
            .map(value_to_usize)
        {
            self.set_w_globals(ns);
        }

        let nlocals = self.local_count();
        let stack_only = self.valuestackdepth().saturating_sub(nlocals);
        // resume.py:1077 consume_boxes(info, boxes_i, boxes_r, boxes_f) parity:
        // RPython's consume_boxes uses position_info (liveness at resume PC)
        // to map compact active_boxes back to register indices. Dead registers
        // are skipped in the compact array — only live registers advance the
        // value index.
        //
        // values[3..] = compact active_boxes from get_list_of_active_boxes,
        // which filters by liveness. Use the same liveness table to restore.
        // Two distinct pointers for the same PyFrame.pycode:
        //   - `w_code_ptr`  : Python CodeObject WRAPPER (`PyObjectRef`).
        //                     Key used by `jitcode_for`/`jitcode_lookup`
        //                     (see trace_opcode.rs:3501/:3580 and
        //                     state.rs:1908 — all pass the wrapper).
        //   - `raw_code_ptr`: unwrapped Rust `CodeObject`. Consumed by
        //                     `liveness_for` (the LiveVars cache key).
        let (w_code_ptr, raw_code_ptr) = if self.frame != 0 {
            let w_code = unsafe {
                *((self.frame as *const u8).add(crate::frame_layout::PYFRAME_PYCODE_OFFSET)
                    as *const *const ())
            };
            if !w_code.is_null() {
                let raw = unsafe {
                    pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                        as *const pyre_interpreter::CodeObject
                };
                (w_code, raw)
            } else {
                (std::ptr::null(), std::ptr::null())
            }
        } else {
            (std::ptr::null(), std::ptr::null())
        };
        // resume.py:1383: info = blackholeinterp.get_current_position_info()
        // blackhole.py:337: position was set by setposition(jitcode, pc) where
        // pc comes from rd_numb — the same orgpc used by get_list_of_active_boxes.
        // next_instr = orgpc + 1 + caches, which may have different liveness.
        let live_pc = self.resume_pc.take().unwrap_or_else(|| self.next_instr());
        let mut idx = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        // resume.py:1077 consume_boxes parity — iterate the SAME
        // `all_liveness` BC_LIVE data the encoder used in
        // `trace_opcode.rs::get_list_of_active_boxes`. For every live
        // reg color, reverse-map through the current jitcode's live
        // stack-slot colors before falling back to the local inputarg
        // prefix. This keeps the encoder/decoder contract aligned on
        // jitcode SSA liveness rather than the LiveVars approximation.
        //
        // Falls back to the LiveVars path below only when the frame's
        // code is not (yet) registered or has no pc_map entry — i.e.
        // the same fallback branch trace_opcode.rs:313-334 takes.
        let decoded_via_jitcode: bool = (|| {
            if w_code_ptr.is_null() {
                return false;
            }
            let jc_ptr = jitcode_lookup(w_code_ptr);
            if jc_ptr.is_null() {
                return false;
            }
            let jc = unsafe { &*jc_ptr };
            let payload = &jc.payload;
            // G.4.3 portal-bridge writeback: positional iteration over
            // `0..stack_base + depth_at_py_pc[live_pc]`, paired with the
            // encoder's positional emit in
            // `trace_opcode.rs::get_list_of_active_boxes` and the
            // count check in `frame_value_count_at`.  Skips the canonical
            // `all_liveness` path (whose register indices are for the
            // dispatch loop, not the user PyFrame) and routes each
            // consumed value to `set_local_at` / `set_stack_at` by the
            // same `reg < stack_base` split the canonical path uses
            // below.
            //
            // `metadata.stack_base = nlocals + ncells` (G.3h), the same
            // value `local_count()` derives from `concrete_nlocals(frame)`
            // — but reading from metadata makes the encoder/decoder
            // symmetric source explicit (G.4.3a-fix: pre-fix code mixed
            // `local_count() = stack_base` here with `nlocals_from_code()
            // = varnames.len()` in the count callback, breaking parity
            // for closure-bearing functions).
            if payload.is_portal_bridge() {
                let stack_base = payload.metadata.stack_base;
                let depth = payload
                    .metadata
                    .depth_at_py_pc
                    .get(live_pc)
                    .copied()
                    .unwrap_or(0) as usize;
                let target_count = stack_base + depth;
                for reg in 0..target_count {
                    if let Some(value) = values.get(idx) {
                        let boxed = virtualizable_box_value(value);
                        if reg < stack_base {
                            let _ = self.set_local_at(reg, boxed);
                        } else {
                            let stack_idx = reg - stack_base;
                            if stack_idx < stack_only {
                                let _ = self.set_stack_at(stack_idx, boxed);
                            }
                        }
                    }
                    idx += 1;
                }
                return true;
            }
            let Some(jit_pc) = payload.resume_jitcode_pc_for(live_pc) else {
                return false;
            };
            let all_liveness = liveness_info_snapshot();
            let off = payload.jitcode.get_live_vars_info(jit_pc, op_live());
            if off + 2 >= all_liveness.len() {
                return false;
            }
            let length_i = all_liveness[off] as u32;
            let length_r = all_liveness[off + 1] as u32;
            let length_f = all_liveness[off + 2] as u32;
            let mut cursor = off + 3;
            use majit_translate::liveness::LivenessIterator;
            // Route Ref-bank live-register
            // colors through `metadata.stack_slot_color_map` (forward
            // map: stack slot d → post-rename color). Currently with
            // input-arg pinning the map is `[nlocals, nlocals+1, ...]`
            // so the lookup is currently identity; without the
            // pinning, stack colors may differ from `nlocals + d` and
            // the map is the single source of truth for the
            // `color → stack-slot-index` reverse lookup the heap
            // writeback needs.
            let local_color_map: &[u16] = &payload.metadata.pyre_color_for_semantic_local;
            let stack_color_map: &[u16] = &payload.metadata.stack_slot_color_map;
            let live_local_indices: Vec<usize> = {
                let code_ptr = if !payload.code_ptr.is_null() {
                    payload.code_ptr
                } else {
                    raw_code_ptr
                };
                if code_ptr.is_null() {
                    Vec::new()
                } else {
                    let live_vars = crate::liveness::liveness_for(code_ptr);
                    (0..nlocals)
                        .filter(|&local_idx| live_vars.is_local_live(live_pc, local_idx))
                        .collect()
                }
            };
            for (is_ref_bank, length) in [(false, length_i), (true, length_r), (false, length_f)] {
                if length == 0 {
                    continue;
                }
                let mut it = LivenessIterator::new(cursor, length, &all_liveness);
                while let Some(reg_idx) = it.next() {
                    let reg = reg_idx as usize;
                    if let Some(value) = values.get(idx) {
                        let boxed = virtualizable_box_value(value);
                        if is_ref_bank {
                            if let Some(slot_idx) = semantic_ref_slot_for_reg_color(
                                nlocals,
                                stack_only,
                                local_color_map,
                                stack_color_map,
                                &live_local_indices,
                                reg,
                            ) {
                                if slot_idx < nlocals {
                                    let _ = self.set_local_at(slot_idx, boxed);
                                } else {
                                    let _ = self.set_stack_at(slot_idx - nlocals, boxed);
                                }
                            } else if reg < nlocals
                                && matches!(value, Value::Int(_) | Value::Float(_))
                            {
                                // Runtime resume values are authoritative for
                                // kind.  A stale Ref-bank liveness entry can
                                // still carry an unboxed Int/Float fail arg;
                                // box it into the semantic local without
                                // widening Ref color reverse-mapping for
                                // actual Ref values.
                                let _ = self.set_local_at(reg, boxed);
                            }
                        } else if reg < nlocals {
                            // Int/Float bank register indices are not Ref colors:
                            // do not route them through
                            // semantic_ref_slot_for_reg_color.  For the
                            // current frame-local case, the jitcode liveness
                            // uses the semantic local index; box the raw value
                            // back into PyFrame.locals_cells_stack_w.
                            let _ = self.set_local_at(reg, boxed);
                        }
                    }
                    idx += 1;
                }
                cursor = it.offset;
            }
            true
        })();
        if !decoded_via_jitcode {
            // The out-of-range-pc source that previously reached this
            // branch is eliminated, and the bridge-resume tests run on the
            // real trace-side jitcode registration path
            // (`ensure_jitcode_index`). Unconditional panic — any hit is a
            // bug.
            panic!(
                "bridge resume decode: jitcode path failed — \
                 w_code_ptr={:p} raw_code_ptr={:p} live_pc={} \
                 nlocals={} stack_only={}. Phase X-0/X-1 removed all \
                 known triggers; further hits are bugs.",
                w_code_ptr, raw_code_ptr, live_pc, nlocals, stack_only
            );
        }

        // Clear stale slots beyond valuestackdepth (blackhole fresh frame parity).
        let vsd = self.valuestackdepth();
        if let Some(arr) = self.locals_cells_stack_array_mut() {
            for i in vsd..arr.len() {
                arr[i] = pyre_object::PY_NULL;
            }
        }
        true
    }

    /// resume.py:1077 consume_boxes(info, boxes_i, boxes_r, boxes_f) parity:
    /// Return the type of each slot in the resumed frame section.
    /// In pyre, all frame slots are PyObjectRef (GCREF), so every slot
    /// is Ref. RPython uses typed registers (boxes_i/r/f) but pyre's
    /// virtualizable array is uniformly Ref.
    fn reconstructed_frame_value_types(
        &self,
        meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
    ) -> Option<Vec<Type>> {
        // resume.py:1077: consume_boxes fills boxes_i/boxes_r/boxes_f.
        // pyre frame slots (locals_cells_stack_w) are all GCREF (Ref).
        let nlocals = meta.num_locals;
        let stack_only = self.valuestackdepth().saturating_sub(nlocals);
        // Header [frame_ptr=Ref, ni=Int, code=Ref, vsd=Int, ns=Ref] + all locals/stack as Ref.
        Some(crate::virtualizable_gen::virt_live_value_types(
            nlocals + stack_only,
        ))
    }

    /// resume.py:1049 parity: restore frame register state from decoded values.
    /// resume.py:1077 consume_boxes → _prepare_next_section → enumerate_vars:
    /// each callback_r writes a ref value to the register at the given index.
    /// In pyre, this writes values to the PyFrame's locals/stack via the
    /// virtualizable mechanism (restore_virtualizable_state handles the
    /// full [frame, ni, code, vsd, ns, locals..., stack...] layout).
    fn restore_reconstructed_frame_values(
        &mut self,
        meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
        values: &[Value],
        exception: &majit_metainterp::blackhole::ExceptionState,
    ) -> bool {
        // resume.py:1077 consume_boxes parity: write values to the frame.
        // blackhole.py:337: setposition(jitcode, pc) before consume_one_section —
        // frame_pc from rd_numb is the liveness PC (orgpc).
        self.resume_pc = Some(_frame_pc as usize);
        self.restore_guard_failure_values(meta, values, exception)
    }

    /// blackhole.py:1800 parity: multi-frame support.
    fn supports_multi_frame_restore(&self) -> bool {
        true
    }

    /// blackhole.py:1333 parity: push outer frame for chain.
    /// Multi-frame recovery handled by blackhole chain in call_jit.rs
    /// which receives all frame sections in the typed vector.
    fn push_caller_frame(
        &mut self,
        _meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _values: &[Value],
        _pc: u64,
        _jitcode_index: i32,
    ) -> bool {
        true
    }

    /// blackhole.py:1760 parity: frame transition via chain.
    fn pop_to_caller_frame(&mut self, _meta: &Self::Meta) -> bool {
        false // Blackhole chain handles this directly.
    }

    fn virtualizable_heap_ptr(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<*mut u8> {
        crate::virtualizable_gen::virt_heap_ptr(self, _virtualizable)
    }

    fn virtualizable_array_lengths(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        info: &VirtualizableInfo,
    ) -> Option<Vec<usize>> {
        if info.array_fields.is_empty() {
            return Some(Vec::new());
        }
        // virtualizable.py:86 parity: full array length read from the
        // live heap object (`lst = getattr(virtualizable, fieldname);
        // append(len(lst))`). Upstream has no fallback — if the heap
        // isn't readable here, return `None` so the caller skips.
        if info.can_read_all_array_lengths_from_heap() {
            if let Some(frame_ptr) = self.frame_ptr() {
                let lens = unsafe { info.read_array_lengths_from_heap(frame_ptr) };
                return Some(lens);
            }
        }
        None
    }

    fn sync_virtualizable_before_jit(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        info: &VirtualizableInfo,
    ) -> bool {
        // Heap is source of truth — just validate the frame is usable.
        if !self.validate_frame() {
            return false;
        }
        // virtualizable.py:170 force_token_before_residual_call parity:
        // clear vable_token so the JIT knows the virtualizable is synced.
        if let Some(frame_ptr) = self.frame_ptr() {
            unsafe { info.reset_vable_token(frame_ptr) };
        }
        true
    }

    fn sync_virtualizable_after_jit(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        info: &VirtualizableInfo,
    ) {
        let Some(frame_ptr) = self.frame_ptr() else {
            return;
        };
        // Heap is source of truth — nothing to sync. Just reset token.
        unsafe {
            info.reset_vable_token(frame_ptr);
        }
    }

    fn sync_virtualizable_before_residual_call(&self, ctx: &mut TraceCtx) {
        crate::virtualizable_gen::virt_sync_before_residual(self, ctx)
    }

    fn sync_virtualizable_after_residual_call(
        &self,
        _ctx: &mut TraceCtx,
    ) -> ResidualVirtualizableSync {
        crate::virtualizable_gen::virt_sync_after_residual(self, _ctx)
    }

    fn import_virtualizable_boxes(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
        static_boxes: &[i64],
        array_boxes: &[Vec<i64>],
    ) -> bool {
        self.import_virtualizable_state(static_boxes, array_boxes)
    }

    fn export_virtualizable_boxes(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<(Vec<i64>, Vec<Vec<i64>>)> {
        Some(self.export_virtualizable_state())
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        Self::pypyjit_collect_jump_args(sym)
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        Self::pypyjit_collect_typed_jump_args(sym)
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool {
        let _ = (sym, meta);
        true
    }

    fn validate_close_with_jump_args(
        _sym: &Self::Sym,
        meta: &Self::Meta,
        jump_args: &[OpRef],
    ) -> bool {
        let _ = meta;
        // pyre does not close loops by re-reading symbolic stack state from
        // the trace header.  Instead it materializes explicit jump args from
        // the concrete frame-backed virtualizable state at the merge point:
        //   [frame, next_instr, valuestackdepth, locals..., stack...]
        //
        // Retraces/bridges may start from a transient stackful state
        // (e.g. direct-call trace-through in progress) and still legally
        // jump back to a target loop whose merge-point stack is smaller.
        // RPython closes those traces against the target token contract, not
        // the retrace entry state's stack depth.  So for pyre's explicit
        // jump-arg model, the trace-start `meta.valuestackdepth` is not a
        // sound validator here.
        jump_args.len() >= crate::virtualizable_gen::NUM_SCALAR_INPUTARGS
    }

    /// RPython resume.py: materialize a virtual object from resume data.
    ///
    /// Called during guard failure recovery when the optimizer kept an
    /// object virtual (New + SetfieldGc eliminated). The resume mechanism
    /// reconstructs the object so the interpreter can use it.
    fn materialize_virtual_ref(
        &mut self,
        _meta: &Self::Meta,
        _virtual_index: usize,
        materialized: &majit_metainterp::resume::MaterializedVirtual,
    ) -> Option<majit_ir::GcRef> {
        self.materialize_virtual_ref_from_layout(materialized, &[])
    }

    fn materialize_virtual_ref_with_refs(
        &mut self,
        _meta: &Self::Meta,
        _virtual_index: usize,
        materialized: &majit_metainterp::resume::MaterializedVirtual,
        materialized_refs: &[Option<majit_ir::GcRef>],
    ) -> Option<majit_ir::GcRef> {
        self.materialize_virtual_ref_from_layout(materialized, materialized_refs)
    }
}

impl PyreJitState {
    fn materialize_virtual_ref_from_layout(
        &mut self,
        materialized: &majit_metainterp::resume::MaterializedVirtual,
        materialized_refs: &[Option<majit_ir::GcRef>],
    ) -> Option<majit_ir::GcRef> {
        use majit_metainterp::resume::MaterializedVirtual;

        match materialized {
            // resume.py:618-620 VirtualInfo.allocate — `allocate_with_vtable(descr=self.descr)`
            // followed by `setfields`. descr carries both vtable and obj_size.
            MaterializedVirtual::Obj {
                descr: Some(descr),
                fields,
                ..
            } => materialize_virtual_object(descr, fields, materialized_refs),
            // resume.py:633-636 VStructInfo.allocate — `allocate_struct(self.typedescr)`
            // + `setfields`. No vtable write.
            MaterializedVirtual::Struct {
                descr: Some(descr),
                fields,
                ..
            } => materialize_virtual_struct(descr, fields, materialized_refs),
            MaterializedVirtual::RawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => materialize_virtual_raw_buffer(
                *func,
                *size,
                offsets,
                descrs,
                values,
                materialized_refs,
            ),
            _ => None,
        }
    }
}

/// resume.py:618-621 VirtualInfo.allocate parity — `allocate_with_vtable(descr)`
/// then `setfields`.
///
/// Allocates `descr.size()` bytes aligned to 8, seeds the PyObject header
/// (`ob_type = descr.vtable()`, `w_class = get_instantiate(vtable)`), then
/// replays each traced field using the FieldDescr's byte offset/size/type
/// from the SizeDescr's `all_fielddescrs()` table (resume.py:597-603
/// setfields loop — `decoder.setfield(struct, num, descr)`).
///
/// The vtable source is the descr itself — there is no `type_id` special-casing,
/// so bool (type_id=0, vtable=&BOOL_TYPE), range-iter (type_id=0,
/// vtable=&RANGE_ITER_TYPE) and custom classes all dispatch uniformly.
fn materialize_virtual_object(
    descr: &DescrRef,
    fields: &[(u32, majit_metainterp::resume::MaterializedValue)],
    materialized_refs: &[Option<majit_ir::GcRef>],
) -> Option<majit_ir::GcRef> {
    use pyre_object::pyobject::{
        OB_TYPE_OFFSET, PyObject, PyType, W_CLASS_OFFSET, get_instantiate,
    };

    let size_descr = descr.as_size_descr()?;
    let vtable = size_descr.vtable();
    let obj_size = size_descr.size();
    if vtable == 0 || obj_size < std::mem::size_of::<PyObject>() {
        return None;
    }

    // resume.py:619 allocate_with_vtable — raw heap allocation that
    // matches `Box::leak(Box::new(...))` for pyre's existing W_*Object
    // builders. 8-byte alignment matches the natural alignment of
    // `#[repr(C)]` structs whose first field is an 8-byte pointer.
    let layout = std::alloc::Layout::from_size_align(obj_size, 8).ok()?;
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    if raw.is_null() {
        return None;
    }

    unsafe {
        let ptr = raw as *mut PyObject;
        (*ptr).ob_type = vtable as *const PyType;
        // rclass.py:739-743 set `w_class` from the cached instantiate
        // pointer on the PyType. Tracing may later overwrite this via
        // an explicit `SetfieldGc(w_class)`; the field replay below
        // takes precedence for that case (heaptracker.py:66-style
        // "typeptr" filter does NOT apply to w_class in pyre).
        (*ptr).w_class = get_instantiate(&*(vtable as *const PyType));
    }

    // resume.py:597-603 setfields parity: for each traced field,
    // look up the FieldDescr from the SizeDescr table and write the
    // value at descr.offset() with descr.field_size() bytes.
    for (field_idx, value) in fields {
        let Some(field_descr) = size_descr
            .all_fielddescrs()
            .iter()
            .find(|fd| fd.index() == *field_idx)
        else {
            // Field not in the SizeDescr table — skip (ob_type handled
            // above; anything else is a tracer bug we just log-silently
            // drop rather than corrupt memory).
            continue;
        };
        let offset = field_descr.offset();
        let field_size = field_descr.field_size();
        if offset == OB_TYPE_OFFSET && field_size == std::mem::size_of::<*const PyType>() {
            // ob_type already written from vtable; honour explicit override.
            let concrete = value.resolve_with_refs(materialized_refs)?;
            unsafe {
                (raw.add(offset) as *mut usize).write(concrete as usize);
            }
            continue;
        }
        if offset == W_CLASS_OFFSET && field_size == std::mem::size_of::<*mut PyObject>() {
            let concrete = value.resolve_with_refs(materialized_refs)?;
            unsafe {
                (raw.add(offset) as *mut usize).write(concrete as usize);
            }
            continue;
        }
        let concrete = value.resolve_with_refs(materialized_refs)?;
        unsafe {
            write_field_bytes(raw, offset, field_size, concrete);
        }
    }

    Some(majit_ir::GcRef(raw as usize))
}

/// resume.py:633-636 VStructInfo.allocate parity — `allocate_struct(typedescr)`
/// (no vtable) + `setfields`.
fn materialize_virtual_struct(
    descr: &DescrRef,
    fields: &[(u32, majit_metainterp::resume::MaterializedValue)],
    materialized_refs: &[Option<majit_ir::GcRef>],
) -> Option<majit_ir::GcRef> {
    let size_descr = descr.as_size_descr()?;
    let obj_size = size_descr.size();
    if obj_size == 0 {
        return None;
    }
    let layout = std::alloc::Layout::from_size_align(obj_size, 8).ok()?;
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    if raw.is_null() {
        return None;
    }
    for (field_idx, value) in fields {
        let Some(field_descr) = size_descr
            .all_fielddescrs()
            .iter()
            .find(|fd| fd.index() == *field_idx)
        else {
            continue;
        };
        let offset = field_descr.offset();
        let field_size = field_descr.field_size();
        let concrete = value.resolve_with_refs(materialized_refs)?;
        unsafe {
            write_field_bytes(raw, offset, field_size, concrete);
        }
    }
    Some(majit_ir::GcRef(raw as usize))
}

/// Write an i64 value into a byte-size field at `offset` inside `raw`.
/// Supports 1/2/4/8 byte widths. 8-byte write also handles Ref/Float
/// (Float arrives as `value.to_bits() as i64`; raw bytes reinterpret).
///
/// # Safety
/// Caller guarantees `raw + offset + field_size` is within the object allocation.
unsafe fn write_field_bytes(raw: *mut u8, offset: usize, field_size: usize, value: i64) {
    unsafe {
        let dst = raw.add(offset);
        match field_size {
            1 => (dst as *mut u8).write(value as u8),
            2 => (dst as *mut u16).write(value as u16),
            4 => (dst as *mut u32).write(value as u32),
            8 => (dst as *mut u64).write(value as u64),
            _ => {
                // Fallback: write as raw bytes LE, truncated.
                let bytes = value.to_le_bytes();
                for i in 0..field_size.min(bytes.len()) {
                    dst.add(i).write(bytes[i]);
                }
            }
        }
    }
}

fn materialize_virtual_raw_buffer(
    func: i64,
    size: usize,
    offsets: &[i64],
    descrs: &[majit_ir::ArrayDescrInfo],
    values: &[majit_metainterp::resume::MaterializedValue],
    materialized_refs: &[Option<majit_ir::GcRef>],
) -> Option<majit_ir::GcRef> {
    // resume.py:700-709 VRawBufferInfo.allocate_int iterates len(self.offsets)
    // unconditionally, indexing self.descrs[i]/self.fieldnums[i] by the same i.
    // No len-equality assert (VRawBufferInfo has none); a short descrs/values
    // raises IndexError here (encoder bug), a longer one leaves its tail unread.
    let (driver, _) = crate::driver::driver_pair();
    // resume.py:1452-1456 allocate_raw_buffer:
    //   cic = self.callinfocollection
    //   calldescr, _ = cic.callinfo_for_oopspec(OS_RAW_MALLOC_VARSIZE_CHAR)
    //   return self.cpu.bh_call_i(func, [size], None, None, calldescr)
    // The calldescr comes from the shared callinfocollection, not a freshly
    // minted MOST_GENERAL descr.  func stays the VRawBufferInfo.func (resume.py
    // discards the callinfo's func as `_`).
    let cic = driver
        .meta_interp()
        .callinfocollection()
        .expect(
            "materialize_virtual_raw_buffer: MetaInterp.callinfocollection \
             required for VRawBufferInfo recovery (resume.py:1453)",
        )
        .clone();
    // resume.py:1455: calldescr, _ = cic.callinfo_for_oopspec(...).
    // callinfo_for_oopspec returns (None, 0) on a miss (effectinfo.py:444-447)
    // — no lookup-time guard; the calldescr is consumed directly below.
    let (calldescr, _) =
        cic.callinfo_for_oopspec(majit_ir::descr::OopSpecIndex::RawMallocVarsizeChar);
    let descr_ref = calldescr.expect("OS_RAW_MALLOC_VARSIZE_CHAR calldescr (callinfocollection)");
    let cd = descr_ref
        .as_call_descr()
        .expect("OS_RAW_MALLOC_VARSIZE_CHAR: not a CallDescr");
    let bh_calldescr = majit_translate::jitcode::BhCallDescr::from_arg_classes(
        cd.arg_classes(),
        cd.result_class(),
        cd.get_extra_info().clone(),
    );
    // resume.py:703: buffer = self.cpu.bh_call_i(func, [size], None, None, calldescr)
    let buffer = driver.meta_interp().backend().bh_call_i(
        func,
        Some(&[size as i64]),
        None,
        None,
        &bh_calldescr,
    );

    let backend = driver.meta_interp().backend();
    // resume.py:705-708: per-item bh_raw_store_i/f, run unconditionally (no
    // buffer == 0 guard).
    for i in 0..offsets.len() {
        let concrete = values[i].resolve_with_refs(materialized_refs)?;
        let di = &descrs[i];
        let bh_descr = majit_translate::jitcode::BhDescr::from_array_descr_info(di);
        // resume.py:1544: assert not descr.is_array_of_pointers()
        assert!(
            !bh_descr.is_array_of_pointers(),
            "raw buffer entry must not be pointer type"
        );
        let offset = offsets[i];
        if di.item_type == 2 {
            // resume.py:1545-1547: descr.is_array_of_floats() → bh_raw_store_f
            backend.bh_raw_store_f(buffer, offset, f64::from_bits(concrete as u64), &bh_descr);
        } else {
            // resume.py:1548-1550: else → bh_raw_store_i
            backend.bh_raw_store_i(buffer, offset, concrete, &bh_descr);
        }
    }

    Some(majit_ir::GcRef(buffer as usize))
}

fn value_to_usize(value: &Value) -> usize {
    match value {
        Value::Ref(gc_ref) => gc_ref.0,
        Value::Int(n) => *n as usize,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    //! Most tests in this module are original Rust regressions.
    //!
    //! PyPy/Pyre upstream tests usually assert behavior at the meta-interpreter
    //! level, not at the typed-raw tracer seam exercised here. These tests keep
    //! local descriptor choices, virtualizable restoration, and raw-int/raw-f64
    //! fast paths from drifting silently. Where a test is tied directly to an
    //! upstream parity point, that reference is called out inline.

    use super::*;
    use crate::helpers::TraceHelperAccess;
    use majit_metainterp::JitState;
    use majit_metainterp::resume::{MaterializedValue, MaterializedVirtual};
    use pyre_interpreter::bytecode::{BinaryOperator, CodeObject, ConstantData, Instruction};
    use pyre_interpreter::pyopcode::decode_instruction_at;
    use pyre_interpreter::{
        LocalOpcodeHandler, Mode, OpcodeStepExecutor, PyErrorKind, SharedOpcodeHandler,
        compile_exec, compile_source,
    };
    use pyre_object::OB_TYPE_OFFSET;
    use pyre_object::floatobject::w_float_get_value;
    use pyre_object::listobject::w_list_getitem;
    use pyre_object::pyobject::{INT_TYPE, LIST_TYPE, PyType, is_list};
    use std::cell::{Cell, UnsafeCell};

    thread_local! {
        static TEST_CALLBACKS_INIT: Cell<bool> = const { Cell::new(false) };
        static TEST_JIT_DRIVER: UnsafeCell<crate::driver::JitDriverPair> = UnsafeCell::new({
            let info = crate::frame_layout::build_pyframe_virtualizable_info();
            let mut driver = majit_metainterp::JitDriver::new(1);
            driver.set_virtualizable_info(info.clone());
            driver.meta_interp_mut().num_scalar_inputargs =
                crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
            (driver, info)
        });
    }

    #[test]
    fn concrete_value_preserves_int_subclass_identity() {
        pyre_interpreter::typedef::init_typeobjects();
        let obj = pyre_object::intobject::w_int_new_unique(7);
        unsafe {
            (*obj).w_class = pyre_object::w_none();
        }

        assert_eq!(ConcreteValue::from_pyobj(obj), ConcreteValue::Ref(obj));
    }

    #[test]
    fn concrete_value_still_unboxes_exact_int_and_bool() {
        pyre_interpreter::typedef::init_typeobjects();
        let int_obj = pyre_object::w_int_new(11);
        let true_obj = pyre_object::w_bool_from(true);
        let float_obj = pyre_object::w_float_new(3.14);

        assert_eq!(ConcreteValue::from_pyobj(int_obj), ConcreteValue::Int(11));
        assert_eq!(
            ConcreteValue::from_pyobj(true_obj),
            ConcreteValue::Bool(true)
        );
        assert_eq!(
            ConcreteValue::from_pyobj(float_obj),
            ConcreteValue::Float(3.14)
        );
    }

    #[test]
    fn concrete_value_preserves_float_subclass_identity() {
        pyre_interpreter::typedef::init_typeobjects();
        let obj = pyre_object::w_float_new(2.5);
        unsafe {
            (*obj).w_class = pyre_object::w_none();
        }

        assert_eq!(ConcreteValue::from_pyobj(obj), ConcreteValue::Ref(obj));
    }

    fn ensure_test_callbacks() {
        TEST_CALLBACKS_INIT.with(|init| {
            if init.get() {
                return;
            }
            init.set(true);
            let cb = Box::leak(Box::new(crate::callbacks::CallJitCallbacks {
                callee_frame_helper: |_| None,
                recursive_force_cache_safe: |_| false,
                jit_drop_callee_frame: std::ptr::null(),
                jit_force_callee_frame: std::ptr::null(),
                jit_force_recursive_call_1: std::ptr::null(),
                jit_force_recursive_call_argraw_boxed_1: std::ptr::null(),
                jit_force_self_recursive_call_argraw_boxed_1: std::ptr::null(),
                jit_create_callee_frame_1: std::ptr::null(),
                jit_create_callee_frame_1_raw_int: std::ptr::null(),
                jit_create_self_recursive_callee_frame_1: std::ptr::null(),
                jit_create_self_recursive_callee_frame_1_raw_int: std::ptr::null(),
                driver_pair: || TEST_JIT_DRIVER.with(|cell| cell.get() as *mut u8),
                ensure_majit_jitcode: |_, _| {},
            }));
            crate::callbacks::init(cb);
        });
    }

    /// Install a populated `PyJitCode` for `code_ref` so a subsequent
    /// `jitcode_for(code_ref)` returns a valid SD entry. The trace-side
    /// `MetaInterpStaticData.jitcodes` list is now populated only by
    /// `CodeWriter.make_jitcodes()` (warmspot.py:281-282); tests that
    /// exercise tracer logic directly must still arrive at that
    /// post-`make_jitcodes` state, so we install a minimal assembled
    /// jitcode with a `live/` anchor at offset 0 — every can-raise
    /// CALL_* in the tracer now emits an inline `GuardNoException`
    /// (pyjitpl.py:2082) whose resumedata snapshot routes through
    /// `get_list_of_active_boxes` → `JitCode::get_live_vars_info`
    /// (jitcode.py:80-93), so the test fixture must satisfy that
    /// liveness lookup.
    fn install_test_jitcode(code: &CodeObject, code_ref: *const ()) {
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(code_ref as pyre_object::PyObjectRef)
                as *const CodeObject
        };
        let mut builder = majit_metainterp::JitCodeBuilder::default();
        let live_patch = builder.live_placeholder();
        builder.patch_live_offset(live_patch, 0);
        let mut insns = majit_ir::VecAssoc::new();
        insns.insert(
            "live/".to_string(),
            majit_metainterp::jitcode::insns::BC_LIVE,
        );
        crate::assembler::publish_state(&insns, &[0, 0, 0], 3, 1);
        let mut pyjit = crate::PyJitCode::skeleton(raw_code, code_ref, None);
        pyjit.jitcode = std::sync::Arc::new(builder.finish());
        pyjit.metadata.pc_map.resize(code.instructions.len(), 0);
        METAINTERP_SD.with(|r| {
            r.borrow_mut()
                .set_jitcodes_from_make_result(vec![std::sync::Arc::new(pyjit)]);
        });
    }

    #[test]
    fn semantic_ref_slot_prefers_live_stack_color_reuse() {
        assert_eq!(
            semantic_ref_slot_for_reg_color(2, 1, &[0, 1], &[0], &[0, 1], 0),
            Some(2),
        );
    }

    #[test]
    fn semantic_ref_slot_falls_back_to_local_color_map() {
        assert_eq!(
            semantic_ref_slot_for_reg_color(2, 1, &[4, 1], &[3], &[1], 1),
            Some(1),
        );
    }

    #[test]
    fn semantic_ref_slot_ignores_dead_local_color_reuse() {
        assert_eq!(
            semantic_ref_slot_for_reg_color(2, 0, &[0, 1], &[], &[1], 0),
            None,
        );
    }

    fn empty_meta() -> PyreMeta {
        PyreMeta {
            num_locals: 0,
            ns_len: 0,
            valuestackdepth: 0,
            array_capacity: 0,
            trace_extra_reds: 0,
            slot_types: Vec::new(),
            has_virtualizable: false,
        }
    }

    fn empty_state() -> PyreJitState {
        PyreJitState {
            frame: 0,
            resume_pc: None,
        }
    }

    fn compile_function_body(src: &str) -> CodeObject {
        let module = compile_source(src, Mode::Exec).expect("compile should succeed");
        module
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some((**code).clone()),
                _ => None,
            })
            .expect("source should contain a function body")
    }

    fn contains_instruction(code: &CodeObject, predicate: impl Fn(Instruction) -> bool) -> bool {
        (0..code.instructions.len()).any(|pc| {
            decode_instruction_at(code, pc)
                .map(|(instruction, _)| predicate(instruction))
                .unwrap_or(false)
        })
    }

    #[test]
    fn test_pre_opcode_snapshot_gate_skips_peek_only_and_no_guard_opcodes() {
        let iter_code =
            compile_function_body("def f(xs):\n    for x in xs:\n        return len(xs)\n");
        assert!(contains_instruction(&iter_code, |instruction| {
            matches!(instruction, Instruction::GetIter)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));
        assert!(contains_instruction(&iter_code, |instruction| {
            matches!(instruction, Instruction::ForIter { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let match_code = compile_function_body(
            "def f(xs):\n    match xs:\n        case [a, b]:\n            return 1\n    return 0\n",
        );
        assert!(contains_instruction(&match_code, |instruction| {
            matches!(instruction, Instruction::GetLen)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let import_code =
            compile_function_body("def f(pkg):\n    from pkg import name\n    return name\n");
        assert!(contains_instruction(&import_code, |instruction| {
            matches!(instruction, Instruction::ImportFrom { .. })
                && instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let except_code = compile_function_body(
            "def f():\n    try:\n        raise ValueError('x')\n    except ValueError:\n        return 0\n",
        );
        assert!(contains_instruction(&except_code, |instruction| {
            matches!(instruction, Instruction::CheckExcMatch)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let none_branch_code = compile_function_body(
            "def f(x):\n    if x is not None:\n        return 1\n    return 0\n",
        );
        assert!(contains_instruction(&none_branch_code, |instruction| {
            matches!(instruction, Instruction::PopJumpIfNone { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let not_none_branch_code =
            compile_function_body("def f(x):\n    if x is None:\n        return 1\n    return 0\n");
        assert!(contains_instruction(&not_none_branch_code, |instruction| {
            matches!(instruction, Instruction::PopJumpIfNotNone { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let truth_branch_code =
            compile_function_body("def f(x):\n    if x:\n        return 1\n    return 0\n");
        assert!(contains_instruction(&truth_branch_code, |instruction| {
            matches!(instruction, Instruction::PopJumpIfFalse { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let not_truth_branch_code =
            compile_function_body("def f(x):\n    if not x:\n        return 1\n    return 0\n");
        assert!(contains_instruction(
            &not_truth_branch_code,
            |instruction| {
                matches!(instruction, Instruction::PopJumpIfTrue { .. })
                    && !instruction_needs_pre_opcode_snapshot(instruction)
            }
        ));

        let contains_code = compile_function_body("def f(a, b):\n    return a in b\n");
        assert!(contains_instruction(&contains_code, |instruction| {
            matches!(instruction, Instruction::ContainsOp { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let with_except_code = compile_function_body(
            "def f(cm):\n    try:\n        with cm:\n            return 1\n    except Exception:\n        return 0\n",
        );
        assert!(contains_instruction(&with_except_code, |instruction| {
            matches!(instruction, Instruction::WithExceptStart)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let import_name_code =
            compile_function_body("def f():\n    import math\n    return math\n");
        assert!(contains_instruction(&import_name_code, |instruction| {
            matches!(instruction, Instruction::ImportName { .. })
                && !instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let call_code = compile_function_body("def f(g, x):\n    return g(x)\n");
        let call_kw_instruction = (0..call_code.instructions.len())
            .find_map(|pc| match decode_instruction_at(&call_code, pc) {
                Some((Instruction::Call { argc }, _)) => Some(Instruction::CallKw { argc }),
                _ => None,
            })
            .expect("source should contain a Call instruction to reuse argc shape");
        assert!(instruction_may_raise(call_kw_instruction));
        assert!(instruction_needs_pre_opcode_snapshot(call_kw_instruction));

        assert!(!instruction_may_raise(Instruction::CallFunctionEx));
        assert!(!instruction_needs_pre_opcode_snapshot(
            Instruction::CallFunctionEx
        ));

        let delete_attr_code = compile_function_body("def f(obj):\n    del obj.x\n");
        assert!(contains_instruction(&delete_attr_code, |instruction| {
            matches!(instruction, Instruction::DeleteAttr { .. })
                && !instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let delete_subscr_code = compile_function_body("def f(a, i):\n    del a[i]\n");
        assert!(contains_instruction(&delete_subscr_code, |instruction| {
            matches!(instruction, Instruction::DeleteSubscr)
                && !instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let store_attr_code = compile_function_body("def f(obj, v):\n    obj.x = v\n");
        assert!(contains_instruction(&store_attr_code, |instruction| {
            matches!(instruction, Instruction::StoreAttr { .. })
                && instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let dict_update_code = compile_function_body("def f(a, b):\n    return {**a, **b}\n");
        assert!(contains_instruction(&dict_update_code, |instruction| {
            matches!(instruction, Instruction::DictUpdate { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let binary_slice_code = compile_function_body("def f(a, b, c):\n    return a[b:c]\n");
        assert!(contains_instruction(&binary_slice_code, |instruction| {
            matches!(instruction, Instruction::BinarySlice)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let store_slice_code =
            compile_function_body("def f(a, b, c, v):\n    a[b:c] = v\n    return a\n");
        assert!(contains_instruction(&store_slice_code, |instruction| {
            matches!(instruction, Instruction::StoreSlice)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let list_extend_code = compile_function_body("def f(a, b):\n    return [*a, *b]\n");
        assert!(contains_instruction(&list_extend_code, |instruction| {
            matches!(instruction, Instruction::ListExtend { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let set_update_code = compile_function_body("def f(a, b):\n    return {*a, *b}\n");
        assert!(contains_instruction(&set_update_code, |instruction| {
            matches!(instruction, Instruction::SetUpdate { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let map_add_code = compile_function_body("def f(xs):\n    return {x: x for x in xs}\n");
        assert!(contains_instruction(&map_add_code, |instruction| {
            matches!(instruction, Instruction::MapAdd { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let set_add_code = compile_function_body("def f(xs):\n    return {x for x in xs}\n");
        assert!(contains_instruction(&set_add_code, |instruction| {
            matches!(instruction, Instruction::SetAdd { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let reraise_code = compile_function_body(
            "def f():\n    try:\n        raise ValueError(1)\n    except Exception:\n        raise\n",
        );
        assert!(contains_instruction(&reraise_code, |instruction| {
            matches!(instruction, Instruction::Reraise { .. })
                && !instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));
    }

    #[test]
    fn test_pre_opcode_snapshot_gate_keeps_pop_before_guard_opcodes() {
        let call_code = compile_function_body("def f(g, x):\n    return g(x)\n");
        assert!(contains_instruction(&call_code, |instruction| {
            matches!(instruction, Instruction::Call { .. })
                && instruction_may_raise(instruction)
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let unary_code = compile_function_body("def f(x):\n    return -x\n");
        assert!(contains_instruction(&unary_code, |instruction| {
            matches!(instruction, Instruction::UnaryNegative)
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let store_subscr_code =
            compile_function_body("def f(a, i, v):\n    a[i] = v\n    return a\n");
        assert!(contains_instruction(&store_subscr_code, |instruction| {
            matches!(instruction, Instruction::StoreSubscr)
                && instruction_may_raise(instruction)
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let raise_code = compile_function_body(
            "def f():\n    try:\n        raise ValueError(1)\n    except Exception:\n        raise\n",
        );
        assert!(contains_instruction(&raise_code, |instruction| {
            matches!(instruction, Instruction::RaiseVarargs { .. })
                && !instruction_may_raise(instruction)
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));
    }

    /// Seed `sym` with `slots` on the symbolic value stack (TOS last).
    ///
    /// Mirrors the invariant a real trace would carry into RERAISE:
    /// `pyopcode.py:1361` reads `peekvalue(oparg)` and `:1364` does
    /// `popvalue()`, both of which require the stack to be populated with
    /// the saved lasti (when `oparg != 0`) and the exception object on
    /// top.  Sets `concrete_stack`, `symbolic_stack_types`, `registers_r`,
    /// and `valuestackdepth` in lockstep.
    fn seed_stack(
        ctx: &mut TraceCtx,
        sym: &mut PyreSym,
        slots: &[(OpRef, ConcreteValue, majit_ir::Type)],
    ) {
        sym.nlocals = 0;
        sym.valuestackdepth = slots.len();
        sym.concrete_stack = slots.iter().map(|(_, cv, _)| *cv).collect();
        sym.symbolic_stack_types = slots.iter().map(|(_, _, t)| *t).collect();
        sym.registers_r = slots.iter().map(|(op, _, _)| *op).collect();
        // Tests don't exercise the vable shadow path; pop_value's
        // `is_active_vable_owner` branch stays inactive.
        sym.locals_cells_stack_array_ref = ctx.const_ref(0);
    }

    #[test]
    fn test_reraise_reuses_last_exception_object() {
        let mut ctx = TraceCtx::for_test(1);
        let exc = pyre_interpreter::PyError::runtime_error("boom").to_exc_object();
        let exc_opref = ctx.const_ref(exc as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.last_exc_value = exc;
        sym.last_exc_box = exc_opref;
        // pyopcode.py:1353 — for `RERAISE 0`, stack is `[..., exc]`.
        seed_stack(
            &mut ctx,
            &mut sym,
            &[(exc_opref, ConcreteValue::Ref(exc), majit_ir::Type::Ref)],
        );

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let err = OpcodeStepExecutor::reraise(&mut state, 0).expect_err("reraise should raise");
        assert_eq!(err.to_exc_object(), exc);
        // pyopcode.py:1363 — oparg==0 → reraise_lasti = -1.
        assert_eq!(err.reraise_lasti, -1);
        // pyopcode.py:1376 — RaiseWithExplicitTraceback → attach_tb=False.
        assert!(!err.attach_tb);
    }

    #[test]
    fn test_reraise_nonzero_oparg_threads_saved_lasti() {
        let mut ctx = TraceCtx::for_test(1);
        let exc = pyre_interpreter::PyError::runtime_error("boom").to_exc_object();
        let exc_opref = ctx.const_ref(exc as i64);
        // pyopcode.py:165-170 lasti push synthesizes
        // `space.newint(lasti_value)` — a fresh W_IntObject.
        let saved_lasti_value: i64 = 42;
        let saved_lasti_obj = pyre_object::w_int_new(saved_lasti_value);
        let lasti_opref = ctx.const_ref(saved_lasti_obj as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.last_exc_value = exc;
        sym.last_exc_box = exc_opref;
        // pyopcode.py:1353 — for `RERAISE 1`, stack is `[..., lasti, exc]`.
        seed_stack(
            &mut ctx,
            &mut sym,
            &[
                (
                    lasti_opref,
                    ConcreteValue::Ref(saved_lasti_obj),
                    majit_ir::Type::Ref,
                ),
                (exc_opref, ConcreteValue::Ref(exc), majit_ir::Type::Ref),
            ],
        );

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let err = OpcodeStepExecutor::reraise(&mut state, 1).expect_err("reraise should raise");
        assert_eq!(err.to_exc_object(), exc);
        // pyopcode.py:1361 — `reraise_lasti = self.space.int_w(self.peekvalue(oparg))`.
        assert_eq!(err.reraise_lasti, saved_lasti_value as i32);
        assert!(!err.attach_tb);
    }

    #[test]
    fn test_reraise_nonconst_lasti_signals_abort_to_dispatcher() {
        let mut ctx = TraceCtx::for_test(1);
        let exc = pyre_interpreter::PyError::runtime_error("boom").to_exc_object();
        let exc_opref = ctx.const_ref(exc as i64);
        // Non-Int slot at peek(oparg): an exception object stands in for
        // any non-Int concrete value (a const-int box is the only shape
        // the trace can fold at compile time).
        let non_int_opref = ctx.const_ref(exc as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.last_exc_value = exc;
        sym.last_exc_box = exc_opref;
        seed_stack(
            &mut ctx,
            &mut sym,
            &[
                (non_int_opref, ConcreteValue::Ref(exc), majit_ir::Type::Ref),
                (exc_opref, ConcreteValue::Ref(exc), majit_ir::Type::Ref),
            ],
        );

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let err = OpcodeStepExecutor::reraise(&mut state, 1).expect_err("reraise should raise");
        // Non-Int lasti slot → reraise_lasti < 0; dispatcher detects the
        // combination `oparg != 0 && reraise_lasti < 0` and routes to
        // TraceAction::Abort, letting the interpreter handle the rare
        // non-const case via the concrete-frame fallback.
        assert!(err.reraise_lasti < 0);
    }

    #[test]
    fn test_raise_varargs_zero_reuses_last_exception_object() {
        let mut ctx = TraceCtx::for_test(1);
        let exc = pyre_interpreter::PyError::runtime_error("boom").to_exc_object();
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.last_exc_value = exc;
        sym.last_exc_box = ctx.const_ref(exc as i64);
        pyre_interpreter::eval::set_current_exception(PY_NULL);
        pyre_interpreter::eval::set_current_exception(exc);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let err =
            OpcodeStepExecutor::raise_varargs(&mut state, 0).expect_err("bare raise should raise");
        assert_eq!(err.to_exc_object(), exc);
        pyre_interpreter::eval::set_current_exception(PY_NULL);
    }

    #[test]
    fn test_raise_varargs_seeds_last_exception_box_for_finishframe_exception() {
        let mut ctx = TraceCtx::for_test(1);
        let exc = pyre_interpreter::PyError::runtime_error("boom").to_exc_object();
        let exc_ref = ctx.const_ref(exc as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(exc_ref, ConcreteValue::Ref(exc)),
        )
        .expect("push exception");

        let err = OpcodeStepExecutor::raise_varargs(&mut state, 1)
            .expect_err("explicit raise should raise");
        assert_eq!(state.sym().last_exc_value, exc);
        assert_eq!(state.sym().last_exc_box, exc_ref);
        assert!(state.sym().class_of_last_exc_is_const);
        assert_eq!(err.to_exc_object(), exc);
    }

    #[test]
    fn test_raise_varargs_rejects_non_exception_values_like_interpreter() {
        let mut ctx = TraceCtx::for_test(1);
        let bad = pyre_object::w_int_new(7);
        let bad_ref = ctx.const_ref(bad as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(bad_ref, ConcreteValue::Ref(bad)),
        )
        .expect("push invalid raise value");

        let err = OpcodeStepExecutor::raise_varargs(&mut state, 1)
            .expect_err("invalid raise should fail");
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert_eq!(err.message, "exceptions must derive from BaseException");
        assert_eq!(state.sym().last_exc_value, PY_NULL);
        assert_eq!(state.sym().last_exc_box, OpRef::NONE);
    }

    #[test]
    fn test_raise_varargs_rejects_non_exception_types_like_interpreter() {
        let code = compile_exec("x = int\n").expect("compile failed");
        let mut frame = pyre_interpreter::PyFrame::new(code);
        frame
            .execute_frame(None, None)
            .expect("module body should execute");
        let ty = unsafe {
            (*frame.fget_w_globals())
                .get("x")
                .copied()
                .expect("namespace should contain x")
        };

        let mut ctx = TraceCtx::for_test(1);
        let ty_ref = ctx.const_ref(ty as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(ty_ref, ConcreteValue::Ref(ty)),
        )
        .expect("push non-exception type");

        let err =
            OpcodeStepExecutor::raise_varargs(&mut state, 1).expect_err("raising int should fail");
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert_eq!(err.message, "exceptions must derive from BaseException");
        assert_eq!(state.sym().last_exc_value, PY_NULL);
        assert_eq!(state.sym().last_exc_box, OpRef::NONE);
    }

    #[test]
    fn test_raise_varargs_rejects_builtin_callables_that_are_not_exception_classes() {
        let code = compile_exec("x = len\n").expect("compile failed");
        let mut frame = pyre_interpreter::PyFrame::new(code);
        frame
            .execute_frame(None, None)
            .expect("module body should execute");
        let callable = unsafe {
            (*frame.fget_w_globals())
                .get("x")
                .copied()
                .expect("namespace should contain x")
        };

        let mut ctx = TraceCtx::for_test(1);
        let callable_ref = ctx.const_ref(callable as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: (&mut *frame) as *mut pyre_interpreter::PyFrame as usize,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(callable_ref, ConcreteValue::Ref(callable)),
        )
        .expect("push builtin callable");

        let err =
            OpcodeStepExecutor::raise_varargs(&mut state, 1).expect_err("raising len should fail");
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert_eq!(err.message, "exceptions must derive from BaseException");
        assert_eq!(state.sym().last_exc_value, PY_NULL);
        assert_eq!(state.sym().last_exc_box, OpRef::NONE);
    }

    #[test]
    fn test_raise_varargs_sets_cause_like_interpreter() {
        let mut ctx = TraceCtx::for_test(1);
        let exc = pyre_interpreter::PyError::runtime_error("boom").to_exc_object();
        let cause = pyre_interpreter::PyError::value_error("root").to_exc_object();
        let exc_ref = ctx.const_ref(exc as i64);
        let cause_ref = ctx.const_ref(cause as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(exc_ref, ConcreteValue::Ref(exc)),
        )
        .expect("push exception");
        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(cause_ref, ConcreteValue::Ref(cause)),
        )
        .expect("push cause");

        let err =
            OpcodeStepExecutor::raise_varargs(&mut state, 2).expect_err("raise from should raise");
        assert_eq!(err.to_exc_object(), exc);
        assert_eq!(
            pyre_interpreter::getattr(exc, "__cause__").expect("read cause"),
            cause
        );
    }

    #[test]
    fn test_raise_varargs_rejects_invalid_cause_like_interpreter() {
        let mut ctx = TraceCtx::for_test(1);
        let exc = pyre_interpreter::PyError::runtime_error("boom").to_exc_object();
        let cause = pyre_object::w_int_new(5);
        let exc_ref = ctx.const_ref(exc as i64);
        let cause_ref = ctx.const_ref(cause as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(exc_ref, ConcreteValue::Ref(exc)),
        )
        .expect("push exception");
        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(cause_ref, ConcreteValue::Ref(cause)),
        )
        .expect("push invalid cause");

        let err = OpcodeStepExecutor::raise_varargs(&mut state, 2)
            .expect_err("invalid cause should fail");
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert_eq!(
            err.message,
            "exception cause must be None or derive from BaseException"
        );
        assert_eq!(state.sym().last_exc_value, PY_NULL);
        assert_eq!(state.sym().last_exc_box, OpRef::NONE);
    }

    #[test]
    fn test_trace_code_step_routes_malformed_raise_through_generic_exception_path() {
        let lookup_code = compile_exec("x = int\n").expect("lookup code should compile");
        let mut lookup_frame = pyre_interpreter::PyFrame::new(lookup_code);
        lookup_frame
            .execute_frame(None, None)
            .expect("lookup module should execute");
        let int_type = unsafe {
            (*lookup_frame.fget_w_globals())
                .get("x")
                .copied()
                .expect("globals should contain int")
        };

        let code = compile_exec("try:\n    raise int\nexcept TypeError:\n    pass\n")
            .expect("trace code should compile");
        let raise_pc = (0..code.instructions.len())
            .find(|&pc| {
                matches!(
                    decode_instruction_at(&code, pc),
                    Some((Instruction::RaiseVarargs { .. }, _))
                )
            })
            .expect("test bytecode should contain RAISE_VARARGS");
        // Byte offsets in/out; pyre PC is a code-unit index.
        let (target_bytes, _depth, _lasti) =
            pyre_interpreter::exception_table::lookup_exceptiontable(
                &code.exceptiontable,
                (raise_pc * 2) as u32,
            )
            .expect("raise should be covered by exception table");
        let handler_pc = target_bytes as usize / 2;
        let code_ref =
            pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
                as *const ();
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(code_ref as pyre_object::PyObjectRef)
                as *const CodeObject
        };
        let mut builder = majit_metainterp::JitCodeBuilder::default();
        let live_patch = builder.live_placeholder();
        builder.patch_live_offset(live_patch, 0);
        let mut insns = majit_ir::VecAssoc::new();
        insns.insert(
            "live/".to_string(),
            majit_metainterp::jitcode::insns::BC_LIVE,
        );
        crate::assembler::publish_state(&insns, &[0, 0, 0], 3, 1);
        let mut pyjit = crate::PyJitCode::skeleton(raw_code, code_ref, None);
        pyjit.jitcode = std::sync::Arc::new(builder.finish());
        pyjit.metadata.pc_map.resize(code.instructions.len(), 0);
        METAINTERP_SD.with(|r| {
            r.borrow_mut()
                .set_jitcodes_from_make_result(vec![std::sync::Arc::new(pyjit)]);
        });

        let mut frame = pyre_interpreter::PyFrame::new(code.clone());
        frame.fix_array_ptrs();
        frame.push(int_type);

        let mut ctx = TraceCtx::for_test(1);
        let int_ref = ctx.const_ref(int_type as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = jitcode_for(code_ref);
        sym.nlocals = frame.nlocals();

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: (&mut *frame) as *mut pyre_interpreter::PyFrame as usize,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(int_ref, ConcreteValue::Ref(int_type)),
        )
        .expect("push malformed raise value");

        let action = state.trace_code_step(&code, raise_pc);
        assert!(
            matches!(action, TraceAction::Continue),
            "malformed raise inside a handler should follow the generic exception path"
        );
        assert_eq!(state.sym().pending_next_instr, Some(handler_pc));
        assert_ne!(state.sym().last_exc_value, PY_NULL);
    }

    #[test]
    fn test_push_exc_info_and_pop_except_preserve_symbolic_previous_exception() {
        let code = compile_exec("try:\n    raise ValueError\nexcept Exception:\n    pass\n")
            .expect("compile failed");
        let mut frame = pyre_interpreter::PyFrame::new(code);
        let prev_exc = pyre_interpreter::PyError::value_error("prev").to_exc_object();
        let caught_exc = pyre_interpreter::PyError::runtime_error("caught").to_exc_object();

        let mut ctx = TraceCtx::for_test(1);
        let prev_exc_ref = ctx.const_ref(prev_exc as i64);
        let caught_exc_ref = ctx.const_ref(caught_exc as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.current_exc_value = prev_exc;
        sym.current_exc_box = prev_exc_ref;
        pyre_interpreter::eval::set_current_exception(prev_exc);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: (&mut *frame) as *mut pyre_interpreter::PyFrame as usize,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        frame.push(caught_exc);
        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(caught_exc_ref, ConcreteValue::Ref(caught_exc)),
        )
        .expect("push caught exception");

        OpcodeStepExecutor::push_exc_info(&mut state).expect("push_exc_info should succeed");
        assert_eq!(state.sym().current_exc_value, caught_exc);
        assert_eq!(state.sym().current_exc_box, caught_exc_ref);

        let pushed_exc = <MIFrame as SharedOpcodeHandler>::pop_value(&mut state)
            .expect("caught exception should remain on stack");
        assert_eq!(pushed_exc.opref, caught_exc_ref);
        let restored_prev = <MIFrame as SharedOpcodeHandler>::pop_value(&mut state)
            .expect("previous exception should be underneath the caught exception");
        // push_exc_info emits a residual `trace_get_current_exception_jit`
        // call (pyopcode.py:786 runtime save-restore parity) so the
        // previous-exception slot carries a fresh OpRef from that call.
        // Only the concrete value survives across the save/restore.
        assert_ne!(restored_prev.opref, OpRef::NONE);
        assert_eq!(restored_prev.concrete.to_pyobj(), prev_exc);

        <MIFrame as SharedOpcodeHandler>::push_value(&mut state, restored_prev.clone())
            .expect("restore previous exception for POP_EXCEPT");
        OpcodeStepExecutor::pop_except(&mut state).expect("pop_except should succeed");
        assert_eq!(state.sym().current_exc_value, prev_exc);
        assert_eq!(state.sym().current_exc_box, restored_prev.opref);
        assert_eq!(pyre_interpreter::eval::get_current_exception(), prev_exc);
        pyre_interpreter::eval::set_current_exception(PY_NULL);
    }

    #[test]
    fn test_trace_ob_type_descr_uses_immutable_header_field_descr() {
        let descr = crate::descr::ob_type_descr();
        let field = descr
            .as_field_descr()
            .expect("ob_type descr must be a field descr");
        assert_eq!(field.offset(), OB_TYPE_OFFSET);
        assert_eq!(field.field_type(), Type::Int);
        assert!(descr.is_always_pure());
        assert!(field.is_immutable());
    }

    #[test]
    fn test_pypyjit_driver_descriptor_matches_interp_jit_layout() {
        let descriptor = PyreJitState::pypyjit_driver_descriptor();

        assert!(descriptor.is_recursive);
        assert_eq!(descriptor.num_greens(), 3);
        assert_eq!(descriptor.num_reds(), 2);
        assert_eq!(descriptor.virtualizable.as_deref(), Some("frame"));
        assert_eq!(descriptor.index_of_virtualizable, 0);

        let greens = descriptor.greens();
        assert_eq!(greens[0].name, "next_instr");
        assert_eq!(greens[0].tp, Type::Int);
        assert_eq!(greens[1].name, "is_being_profiled");
        assert_eq!(greens[1].tp, Type::Int);
        assert_eq!(greens[2].name, "pycode");
        assert_eq!(greens[2].tp, Type::Ref);

        let reds = descriptor.reds();
        assert_eq!(reds[0].name, "frame");
        assert_eq!(reds[0].tp, Type::Ref);
        assert_eq!(reds[1].name, "ec");
        assert_eq!(reds[1].tp, Type::Ref);
    }

    #[test]
    fn test_pypyjit_live_values_with_ec_insert_execution_context_after_frame() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = PyFrame::new(code);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut state = empty_state();
        state.frame = frame_ptr;
        let meta = state.build_meta(0, &PyreEnv);
        let with_ec = state.extract_live_values(&meta);

        assert_eq!(with_ec.len(), 2);
        assert_eq!(with_ec[0], Value::Ref(majit_ir::GcRef(frame_ptr)));
        assert_eq!(
            with_ec[1],
            Value::Ref(majit_ir::GcRef(frame.execution_context as usize))
        );
    }

    #[test]
    fn test_pypyjit_live_value_types_with_ec_match_descriptor_red_prefix() {
        let meta = PyreMeta {
            num_locals: 2,
            ns_len: 0,
            valuestackdepth: 3,
            array_capacity: 4,
            trace_extra_reds: 1,
            has_virtualizable: true,
            slot_types: vec![Type::Ref; 4],
        };

        let descriptor = PyreJitState::pypyjit_driver_descriptor();
        let with_ec = <PyreJitState as JitState>::live_value_types(&empty_state(), &meta);

        assert_eq!(
            with_ec,
            vec![descriptor.reds()[0].tp, descriptor.reds()[1].tp]
        );
    }

    #[test]
    #[ignore = "PyreSym::new_uninit hits the Phase X-1 skeleton-panic since the \
                debug-only fallback was removed; needs a populated-jitcode harness."]
    fn test_pypyjit_collect_jump_args_inserts_ec_after_frame() {
        let mut ctx = TraceCtx::for_test(0);
        let frame_ref = ctx.const_ref(0x1000);
        let ec_ref = ctx.const_ref(0x7000);
        let code_ref = ctx.const_ref(0x2000);
        let namespace_ref = ctx.const_ref(0x3000);
        let local0 = ctx.const_ref(0x4000);
        let stack0 = ctx.const_ref(0x5000);
        let stack1 = ctx.const_ref(0x6000);

        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.nlocals = 1;
        sym.valuestackdepth = 3;
        sym.vable_last_instr = ctx.const_int(33);
        sym.vable_pycode = code_ref;
        sym.vable_valuestackdepth = ctx.const_int(3);
        sym.vable_debugdata = ctx.const_ref(0);
        sym.vable_lastblock = ctx.const_ref(0);
        sym.vable_w_globals = namespace_ref;
        sym.execution_context = ec_ref;
        sym.registers_r = vec![local0, stack0, stack1];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack_types = vec![Type::Ref, Type::Ref];

        let base = sym.vable_collect_jump_args();
        let with_ec = PyreJitState::pypyjit_collect_jump_args(&sym);
        let typed_with_ec = PyreJitState::pypyjit_collect_typed_jump_args(&sym);

        assert_eq!(with_ec.len(), base.len() + 1);
        assert_eq!(with_ec[0], frame_ref);
        assert_eq!(with_ec[1], ec_ref);
        assert_eq!(&with_ec[2..], &base[1..]);
        assert_eq!(typed_with_ec[0], (frame_ref, Type::Ref));
        assert_eq!(typed_with_ec[1], (ec_ref, Type::Ref));
    }

    #[test]
    #[ignore = "PyreSym::new_uninit hits the Phase X-1 skeleton-panic since the \
                debug-only fallback was removed; needs a populated-jitcode harness."]
    fn test_pypyjit_create_sym_shifts_virtualizable_indices_after_ec() {
        let meta = PyreMeta {
            num_locals: 2,
            ns_len: 0,
            valuestackdepth: 4,
            array_capacity: 5,
            trace_extra_reds: 1,
            has_virtualizable: true,
            slot_types: vec![Type::Ref; 5],
        };

        let sym = PyreJitState::pypyjit_create_sym(&meta, 0);

        // virtualizable derive `init_vable_indices` mints typed
        // `OpRef::input_arg_*` variants per `#[vable(inputarg, type = ...)]`
        // (state.rs:1428-1438). The expected slot variants must match those
        // exact types so variant-aware OpRef Eq (resoperation.rs:290)
        // compares correctly against later optimizer/heap-cache keys.
        assert_eq!(sym.frame, OpRef::input_arg_ref(0));
        assert_eq!(sym.execution_context, OpRef::input_arg_ref(1));
        assert_eq!(sym.vable_last_instr, OpRef::input_arg_int(2));
        assert_eq!(sym.vable_pycode, OpRef::input_arg_ref(3));
        assert_eq!(sym.vable_valuestackdepth, OpRef::input_arg_int(4));
        assert_eq!(sym.vable_debugdata, OpRef::input_arg_ref(5));
        assert_eq!(sym.vable_lastblock, OpRef::input_arg_ref(6));
        assert_eq!(sym.vable_w_globals, OpRef::input_arg_ref(7));
        assert_eq!(sym.vable_array_base, Some(8));
        assert_eq!(sym.symbolic_local_types.len(), 2);
        assert_eq!(sym.symbolic_stack_types.len(), 2);
    }

    #[test]
    fn test_restore_expanded_virtualizable_values_with_extra_reds_skips_ec_slot() {
        use majit_ir::GcRef;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_interpreter::{ConstantData, compile_exec};

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

        let mut frame = PyFrame::new(code);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut state = PyreJitState {
            frame: frame_ptr,
            resume_pc: None,
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
            slot_types: vec![Type::Ref, Type::Ref, Type::Ref, Type::Ref],
        };
        let values = vec![
            Value::Ref(GcRef(frame_ptr)),                        // frame
            Value::Ref(GcRef(frame.execution_context as usize)), // ec
            Value::Int(8),                                       // last_instr
            Value::Ref(GcRef(frame.pycode as usize)),            // pycode
            Value::Int(4),                                       // valuestackdepth
            Value::Ref(GcRef(0)),                                // debugdata
            Value::Ref(GcRef(0)),                                // lastblock
            Value::Ref(GcRef(0)),                                // w_globals
            Value::Ref(GcRef(w_int_new(1) as usize)),            // local a
            Value::Ref(GcRef(w_int_new(2) as usize)),            // local b
            Value::Ref(GcRef(w_int_new(3) as usize)),            // local c
            Value::Int(7),                                       // local i
        ];

        state.restore_expanded_virtualizable_values_with_extra_reds(&meta, &values, 1);

        assert_eq!(state.valuestackdepth(), 4);
        let restored_i = state.local_at(3).expect("local i should be restored");
        assert!(unsafe { is_int(restored_i) });
        assert_eq!(unsafe { w_int_get_value(restored_i) }, 7);
    }

    #[test]
    #[ignore = "PyreSym::new_uninit hits the Phase X-1 skeleton-panic since the \
                debug-only fallback was removed; needs a populated-jitcode harness."]
    fn test_guard_class_uses_guard_nonnull_class() {
        let mut ctx = TraceCtx::for_test(1);
        // registers_r[i] tracks locals_cells_stack_w[*] — W_Root array, Type::Ref.
        let obj = OpRef::input_arg_ref(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![obj];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        state.with_ctx(|this, ctx| {
            this.guard_class(ctx, obj, &INT_TYPE as *const PyType);
        });

        let tree_loop = ctx.into_tree_loop();
        let op = tree_loop.ops.last().expect("guard op should be present");
        assert_eq!(op.opcode, OpCode::GuardNonnullClass);
        assert_eq!(op.arg(0).to_opref(), obj);
    }

    #[test]
    #[ignore = "PyreSym::new_uninit hits the Phase X-1 skeleton-panic since the \
                debug-only fallback was removed; needs a populated-jitcode harness."]
    fn test_trace_guarded_int_payload_uses_guard_nonnull_class_and_pure_payload() {
        // value_type is read from the recorder's inputarg type (Phase α/β: Box.type
        // intrinsic parity, history.py:220) so the inputarg must be Ref for
        // trace_guarded_int_payload to take the fast path rather than short-circuit.
        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        // value_type is Ref per the comment above; registers_r[0] = inputarg slot 0.
        let int_obj = OpRef::input_arg_ref(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![int_obj];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let _ = state.with_ctx(|this, ctx| this.trace_guarded_int_payload(ctx, int_obj));

        let recorder = ctx.into_recorder();
        let mut saw_guard_nonnull_class = false;
        let mut saw_pure_payload = false;
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            if op.opcode == OpCode::GuardNonnullClass {
                saw_guard_nonnull_class = true;
            }
            if op.opcode == OpCode::GetfieldGcPureI
                && op
                    .getarglist()
                    .iter()
                    .map(|a| a.to_opref())
                    .collect::<Vec<_>>()
                    == vec![int_obj]
            {
                saw_pure_payload = true;
            }
        }
        assert!(
            saw_guard_nonnull_class,
            "int payload fast path should guard object class via GuardNonnullClass"
        );
        assert!(
            saw_pure_payload,
            "int payload fast path should read the immutable payload with GetfieldGcPureI"
        );
    }

    #[test]
    fn test_trace_unbox_int_with_resume_skips_guard_for_constant_object() {
        let mut ctx = TraceCtx::for_test(0);
        let int_obj = ctx.const_ref(w_int_new(7) as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);

        let payload = {
            let mut state = MIFrame {
                ctx: &mut ctx,
                sym: &mut sym,
                fallthrough_pc: 0,
                parent_frames: Vec::new(),
                pending_result_stack_idx: None,
                pending_result_type: None,
                pending_inline_frame: None,
                orgpc: 0,
                concrete_frame_addr: 0,
                pre_opcode_registers_r: None,
                pre_opcode_semantic_depth: None,
            };
            trace_unbox_int_with_resume(
                &mut state,
                &mut ctx,
                int_obj,
                &INT_TYPE as *const PyType as i64,
            )
        };

        let recorder = ctx.into_recorder();
        let payload_op = recorder
            .get_op_by_pos(payload)
            .expect("payload op should be present");
        assert_eq!(payload_op.opcode, OpCode::GetfieldGcPureI);
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            assert_ne!(
                op.opcode,
                OpCode::GuardClass,
                "constant unbox must not emit GUARD_CLASS",
            );
        }
    }

    #[test]
    fn test_load_method_accepts_plain_python_instance_method() {
        let code = compile_exec("class C:\n    def f(self):\n        return self\nc = C()\n")
            .expect("compile failed");
        let mut frame = pyre_interpreter::PyFrame::new(code.clone());
        frame
            .execute_frame(None, None)
            .expect("class body should execute");
        let instance = unsafe {
            (*frame.fget_w_globals())
                .get("c")
                .copied()
                .expect("namespace should contain c")
        };

        install_test_jitcode(&code, frame.pycode);
        let mut ctx = TraceCtx::for_test(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = jitcode_for(frame.pycode);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: (&mut *frame) as *mut pyre_interpreter::PyFrame as usize,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let instance_ref = ctx.const_ref(instance as i64);
        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(instance_ref, ConcreteValue::Ref(instance)),
        )
        .expect("push instance");

        state.load_method("f").expect("load_method should succeed");
        let receiver = <MIFrame as SharedOpcodeHandler>::pop_value(&mut state)
            .expect("receiver should be present");
        let callable = <MIFrame as SharedOpcodeHandler>::pop_value(&mut state)
            .expect("callable should be present");

        assert!(receiver.concrete.to_pyobj().is_null());
        unsafe {
            assert!(pyre_object::is_method(callable.concrete.to_pyobj()));
        }
    }

    #[test]
    fn test_init_symbolic_skips_heap_array_read_for_standard_virtualizable() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = PyFrame::new(code.clone());
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        install_test_jitcode(&code, frame.pycode);
        let mut ctx = TraceCtx::for_test(1);
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        sym.become_active_vable_owner();

        sym.init_symbolic(&mut ctx, frame_ptr);

        assert_eq!(sym.locals_cells_stack_array_ref, OpRef::NONE);
        let recorder = ctx.into_recorder();
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            assert_ne!(
                op.opcode,
                OpCode::GetfieldRawI,
                "standard virtualizable init should not read locals array from heap"
            );
        }
    }

    #[test]
    fn test_materialize_virtual_ref_reconstructs_float_object() {
        let mut state = empty_state();
        let meta = empty_meta();
        let value = 3.25f64;
        let descr = crate::descr::w_float_size_descr();
        let materialized = MaterializedVirtual::Obj {
            descr: Some(descr.clone()),
            type_id: crate::descr::W_FLOAT_GC_TYPE_ID,
            fields: vec![(
                crate::descr::float_floatval_descr().index(),
                MaterializedValue::Value(value.to_bits() as i64),
            )],
        };

        let ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &materialized,
            &[],
        )
        .expect("float virtual should materialize");

        unsafe {
            assert!(is_float(ptr.0 as PyObjectRef));
            assert_eq!(w_float_get_value(ptr.0 as PyObjectRef), value);
        }
    }

    /// type_id=0 case: W_BoolObject has no dedicated GC type id but a
    /// distinct vtable (`&BOOL_TYPE`). The orthodox materializer must
    /// dispatch on `descr.vtable()` rather than `type_id`.
    #[test]
    fn test_materialize_virtual_ref_reconstructs_bool_object() {
        use pyre_object::boolobject::w_bool_get_value;
        let mut state = empty_state();
        let meta = empty_meta();
        let descr = crate::descr::w_bool_size_descr();
        let materialized = MaterializedVirtual::Obj {
            descr: Some(descr.clone()),
            type_id: 0,
            fields: vec![(
                crate::descr::bool_boolval_descr().index(),
                MaterializedValue::Value(1),
            )],
        };

        let ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &materialized,
            &[],
        )
        .expect("bool virtual should materialize");

        unsafe {
            assert!(is_bool(ptr.0 as PyObjectRef));
            assert!(w_bool_get_value(ptr.0 as PyObjectRef));
        }
    }

    /// Second type_id=0 case: W_RangeIterator has three i64 fields.
    /// Verifies generic field replay at different offsets (no
    /// hard-coded PAYLOAD_0/PAYLOAD_1 dispatch).
    #[test]
    fn test_materialize_virtual_ref_reconstructs_range_iterator() {
        use pyre_object::rangeobject::W_RangeIterator;
        let mut state = empty_state();
        let meta = empty_meta();
        let descr = crate::descr::w_range_iter_size_descr();
        let materialized = MaterializedVirtual::Obj {
            descr: Some(descr.clone()),
            type_id: 0,
            fields: vec![
                (
                    crate::descr::range_iter_current_descr().index(),
                    MaterializedValue::Value(7),
                ),
                (
                    crate::descr::range_iter_stop_descr().index(),
                    MaterializedValue::Value(42),
                ),
                (
                    crate::descr::range_iter_step_descr().index(),
                    MaterializedValue::Value(3),
                ),
            ],
        };

        let ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &materialized,
            &[],
        )
        .expect("range-iter virtual should materialize");

        unsafe {
            let iter = &*(ptr.0 as *const W_RangeIterator);
            assert_eq!(iter.current, 7);
            assert_eq!(iter.stop, 42);
            assert_eq!(iter.step, 3);
        }
    }

    // Needs a real raw-buffer allocator (`func` passed as a valid function
    // pointer into `bh_call_i`) and a callinfocollection carrying
    // OS_RAW_MALLOC_VARSIZE_CHAR. The current test supplies `func: 0` with no
    // registered calldescr, so `materialize_virtual_raw_buffer` cannot run the
    // resume.py:1456 bh_call_i; wiring up the backend allocator is a follow-up.
    #[ignore]
    #[test]
    fn test_materialize_virtual_ref_reconstructs_list_from_raw_buffer_ref() {
        ensure_test_callbacks();
        let mut state = empty_state();
        let meta = empty_meta();
        let first = w_int_new(2);
        let second = w_int_new(4);

        let raw_items = MaterializedVirtual::RawBuffer {
            func: 0,
            size: 16,
            offsets: vec![0, 8],
            descrs: vec![
                majit_ir::ArrayDescrInfo {
                    index: 0,
                    base_size: 0,
                    item_size: 8,
                    item_type: 1,
                    is_signed: false,
                    len_offset: None,
                },
                majit_ir::ArrayDescrInfo {
                    index: 0,
                    base_size: 0,
                    item_size: 8,
                    item_type: 1,
                    is_signed: false,
                    len_offset: None,
                },
            ],
            values: vec![
                MaterializedValue::Value(first as i64),
                MaterializedValue::Value(second as i64),
            ],
        };
        let raw_ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &raw_items,
            &[],
        )
        .expect("raw items buffer should materialize");

        let list_virtual = MaterializedVirtual::Obj {
            descr: None,
            type_id: 0,
            fields: vec![
                (
                    crate::descr::ob_type_descr().index(),
                    MaterializedValue::Value(&LIST_TYPE as *const PyType as usize as i64),
                ),
                (
                    crate::descr::list_length_descr().index(),
                    MaterializedValue::Value(2),
                ),
                (
                    crate::descr::list_items_descr().index(),
                    MaterializedValue::VirtualRef(0),
                ),
            ],
        };

        let list_ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            1,
            &list_virtual,
            &[Some(raw_ptr)],
        )
        .expect("list virtual should materialize");

        unsafe {
            assert!(is_list(list_ptr.0 as PyObjectRef));
            assert_eq!(
                w_int_get_value(w_list_getitem(list_ptr.0 as PyObjectRef, 0).unwrap()),
                2
            );
            assert_eq!(
                w_int_get_value(w_list_getitem(list_ptr.0 as PyObjectRef, 1).unwrap()),
                4
            );
        }
    }

    #[test]
    fn test_virtualizable_array_lengths_uses_full_array() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = PyFrame::new(code);
        frame.fix_array_ptrs();
        let full_len = frame.locals_w().len();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut state = empty_state();
        state.frame = frame_ptr;
        state.set_valuestackdepth(2);
        let info = crate::frame_layout::build_pyframe_virtualizable_info();

        // virtualizable.py:86 parity: full array length, not valuestackdepth.
        assert_eq!(
            <PyreJitState as JitState>::virtualizable_array_lengths(
                &state,
                &empty_meta(),
                "frame",
                &info,
            ),
            Some(vec![full_len])
        );
    }

    #[test]
    #[ignore = "PyreSym::new_uninit hits the Phase X-1 skeleton-panic since the \
                debug-only fallback was removed; needs a populated-jitcode harness."]
    fn test_restore_guard_failure_uses_runtime_value_kinds_for_virtualizable_locals() {
        use majit_ir::GcRef;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_interpreter::{ConstantData, compile_exec};

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

        let mut frame = PyFrame::new(code);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut state = PyreJitState {
            frame: frame_ptr,
            resume_pc: None,
        };
        state.set_next_instr(0);
        state.set_valuestackdepth(4);
        let meta = PyreMeta {
            num_locals: 4,
            ns_len: 0,
            valuestackdepth: 4,
            array_capacity: 4,
            trace_extra_reds: 0,
            has_virtualizable: true,
            // Stale trace-entry types: all locals looked boxed refs when the
            // trace started, but guard failure can still carry a raw int in
            // slot `i`.
            slot_types: vec![Type::Ref, Type::Ref, Type::Ref, Type::Ref],
        };
        let values = vec![
            Value::Ref(GcRef(frame_ptr)),             // frame
            Value::Int(8),                            // last_instr
            Value::Ref(GcRef(frame.pycode as usize)), // pycode
            Value::Int(4),                            // valuestackdepth
            Value::Ref(GcRef(0)),                     // debugdata
            Value::Ref(GcRef(0)),                     // lastblock
            Value::Ref(GcRef(0)),                     // w_globals
            Value::Ref(GcRef(w_int_new(1) as usize)), // local a
            Value::Ref(GcRef(w_int_new(2) as usize)), // local b
            Value::Ref(GcRef(w_int_new(3) as usize)), // local c
            Value::Int(7),                            // local i
        ];

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
    #[ignore = "PyreSym::new_uninit hits the Phase X-1 skeleton-panic since the \
                debug-only fallback was removed; needs a populated-jitcode harness."]
    fn test_load_local_checked_value_respects_symbolic_local_type() {
        let run_case = |symbolic_type: Type, name: &str, expected_guard: Option<OpCode>| {
            let mut ctx = TraceCtx::for_test(1);
            // The slot type matches `symbolic_type` (resoperation.py:719/727/739
            // InputArg{Int,Float,Ref}); Void has no inputarg variant in RPython.
            let local = OpRef::input_arg_typed(0, symbolic_type);
            let mut sym = PyreSym::new_uninit(OpRef::NONE);
            sym.registers_r = vec![local];
            sym.symbolic_local_types = vec![symbolic_type];
            sym.nlocals = 1;

            let mut state = MIFrame {
                ctx: &mut ctx,
                sym: &mut sym,
                fallthrough_pc: 0,
                parent_frames: Vec::new(),
                pending_result_stack_idx: None,
                pending_result_type: None,
                pending_inline_frame: None,
                orgpc: 0,
                concrete_frame_addr: 0,
                pre_opcode_registers_r: None,
                pre_opcode_semantic_depth: None,
            };

            let loaded =
                <MIFrame as LocalOpcodeHandler>::load_local_checked_value(&mut state, 0, name)
                    .expect("local should load");
            assert_eq!(loaded.opref, local);

            let tree_loop = ctx.into_tree_loop();
            assert_eq!(tree_loop.ops.last().map(|op| op.opcode), expected_guard);
        };

        run_case(Type::Int, "j", None);
        run_case(Type::Ref, "b", Some(OpCode::GuardNonnull));
    }

    #[test]
    fn test_store_local_value_preserves_ref_slot_without_reboxing() {
        // RPython Box.type parity: `_opimpl_setarrayitem_vable`
        // (pyjitpl.py:1242-1247) writes the value's Ref box directly —
        // it never reboxes at the consumer. `locals_cells_stack_w` is a
        // W_Root array (virtualizable.py:86-98), so the producer side
        // (`push_typed_value` on the operand stack) is responsible for
        // wrapping Int/Float with `wrapint` / `wrapfloat` before the
        // value flows into the stack or local slot. Pin the contract
        // here: storing a pre-wrapped Ref leaves `registers_r[idx]`
        // pointing at the same OpRef with no additional op emitted.
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = PyFrame::new(code);
        frame.locals_w_mut()[0] = w_int_new(41);
        frame.fix_array_ptrs();
        let mut ctx = TraceCtx::for_test(1);
        // Pre-wrapped Ref — this is the shape producers hand us.
        let ref_value = ctx.const_ref(pyre_object::PY_NULL as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![OpRef::NONE];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        state
            .with_ctx(|this, ctx| this.store_local_value(ctx, 0, ref_value, ConcreteValue::Null))
            .expect("store of pre-wrapped Ref should succeed");
        assert_eq!(
            state.sym().registers_r[0],
            ref_value,
            "Ref value must be stored as-is, not reboxed",
        );
        assert_eq!(state.sym().symbolic_local_types[0], Type::Ref);
    }

    #[test]
    fn test_trace_binary_value_boxes_typed_raw_operands_for_python_helper() {
        let code = compile_exec("x = 1.0 ** 2").expect("test code should compile");
        let code_ref =
            pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
                as *const ();
        install_test_jitcode(&code, code_ref);
        let mut ctx = TraceCtx::for_test(2);
        let lhs = OpRef::input_arg_float(0);
        let rhs = OpRef::input_arg_int(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = jitcode_for(code_ref);
        sym.registers_r = vec![lhs, rhs];
        sym.symbolic_local_types = vec![Type::Float, Type::Int];
        sym.nlocals = 2;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let _ = <MIFrame as TraceHelperAccess>::trace_binary_value(
            &mut state,
            lhs,
            rhs,
            BinaryOperator::Power,
        )
        .expect("generic helper call should box raw operands first");

        let recorder = ctx.into_recorder();
        // GuardNoException (pyjitpl.py:2082) follows the residual Call*, so
        // look up the Call* op explicitly rather than via `ops().last()`.
        let call = recorder
            .ops()
            .iter()
            .rev()
            .find(|op| {
                matches!(
                    op.opcode,
                    OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN
                )
            })
            .expect("call op should be present");
        assert_ne!(call.arg(0).to_opref(), lhs);
        assert_ne!(call.arg(1).to_opref(), rhs);
    }

    #[test]
    fn test_trace_known_builtin_call_boxes_typed_raw_args_for_python_helper_boundary() {
        let code = compile_exec("x = abs(1)").expect("test code should compile");
        let code_ref =
            pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
                as *const ();
        install_test_jitcode(&code, code_ref);
        let mut ctx = TraceCtx::for_test(2);
        let callable = OpRef::input_arg_ref(0);
        let arg = OpRef::input_arg_int(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = jitcode_for(code_ref);
        sym.registers_r = vec![callable, arg];
        sym.symbolic_local_types = vec![Type::Ref, Type::Int];
        sym.nlocals = 2;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let _ = state
            .trace_known_builtin_call(callable, &[arg])
            .expect("known builtin helper boundary should box raw int args");

        let recorder = ctx.into_recorder();
        // GuardNoException (pyjitpl.py:2082) follows the residual Call*, so
        // look up the Call* op explicitly rather than via `ops().last()`.
        let call = recorder
            .ops()
            .iter()
            .rev()
            .find(|op| {
                matches!(
                    op.opcode,
                    OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN
                )
            })
            .expect("call op should be present");
        assert_ne!(call.getarglist().last().map(|a| a.to_opref()), Some(arg));
    }

    #[test]
    fn test_compare_value_direct_emits_raw_truth_for_immediate_branch_consumer() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("if 1 < 2:\n    x = 3\n").expect("test code should compile");
        let code_ref =
            pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
                as *const ();
        install_test_jitcode(&code, code_ref);
        let compare_pc = (0..code.instructions.len())
            .find(|&pc| {
                matches!(
                    decode_instruction_at(&code, pc),
                    Some((Instruction::CompareOp { .. }, _))
                )
            })
            .expect("test bytecode should contain COMPARE_OP");
        let branch_pc = ((compare_pc + 1)..code.instructions.len())
            .find(|&pc| {
                decode_instruction_at(&code, pc)
                    .map(|(instruction, _)| instruction_consumes_comparison_truth(instruction))
                    .unwrap_or(false)
            })
            .expect("test bytecode should contain POP_JUMP_IF after COMPARE_OP");

        let mut frame = PyFrame::new(code);
        frame.fix_array_ptrs();
        let _frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(2);
        // Typed `InputArgInt` inputarg slots — `compare_value_direct`
        // routes through `is_int_typed` lookups (history.py:220
        // box.type) and Untyped slots silently fall through to the
        // boxed-bool path under variant-aware Eq.
        let lhs = OpRef::input_arg_int(0);
        let rhs = OpRef::input_arg_int(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.valuestackdepth = 0;
        sym.jitcode = jitcode_for(code_ref);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: branch_pc,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let concrete_lhs = w_int_new(10);
        let concrete_rhs = w_int_new(20);
        let result = state
            .compare_value_direct(
                lhs,
                rhs,
                ComparisonOperator::Less,
                concrete_lhs,
                concrete_rhs,
            )
            .expect("int comparison should trace");

        let recorder = ctx.into_recorder();
        let mut saw_cmp = false;
        let mut saw_bool_call = false;
        let mut saw_bool_unbox = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            if op.opcode == OpCode::IntLt {
                saw_cmp = true;
            }
            if op.opcode == OpCode::CallR {
                saw_bool_call = true;
            }
            if op.opcode == OpCode::GetfieldGcI {
                saw_bool_unbox = true;
            }
        }
        assert!(
            saw_cmp,
            "branch compare should still emit raw int comparison"
        );
        assert_eq!(
            result,
            // IntLt at op pos 2 — `IntOp` mixin (resoperation.py:564),
            // `box.type='i'` (history.py:220).
            OpRef::int_op(2),
            "with two input args, the immediate branch consumer should receive the raw comparison truth"
        );
        assert!(
            !saw_bool_call,
            "immediate branch consumer should not allocate a bool object"
        );
        assert!(
            !saw_bool_unbox,
            "immediate branch consumer should not unbox a transient bool object"
        );
    }

    #[test]
    fn test_compare_value_direct_boxes_bool_when_not_immediately_consumed_by_branch() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let code_ref =
            pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
                as *const ();
        install_test_jitcode(&code, code_ref);
        let mut frame = PyFrame::new(code);
        frame.fix_array_ptrs();
        let _frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(2);
        // Typed `InputArgInt` inputarg slots — `compare_value_direct`
        // routes through `is_int_typed` lookups (history.py:220
        // box.type) and Untyped slots silently fall through to the
        // boxed-bool path under variant-aware Eq.
        let lhs = OpRef::input_arg_int(0);
        let rhs = OpRef::input_arg_int(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.valuestackdepth = 0;
        sym.jitcode = jitcode_for(code_ref);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let concrete_lhs = w_int_new(10);
        let concrete_rhs = w_int_new(20);
        let _ = state
            .compare_value_direct(
                lhs,
                rhs,
                ComparisonOperator::Less,
                concrete_lhs,
                concrete_rhs,
            )
            .expect("non-branch compare should trace");

        let recorder = ctx.into_recorder();
        let mut saw_bool_call = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            if op.opcode == OpCode::CallR {
                saw_bool_call = true;
            }
        }
        assert!(
            saw_bool_call,
            "non-branch compare should continue to materialize a Python bool result"
        );
    }

    #[test]
    fn test_next_instruction_consumes_comparison_truth_skips_extended_arg_trivia() {
        use pyre_interpreter::pyframe::PyFrame;

        ensure_test_callbacks();
        let mut source = String::from("def f(x, y):\n    if x < y:\n");
        for i in 0..400 {
            source.push_str(&format!("        z{i} = {i}\n"));
        }
        source.push_str("    return 0\n");
        source.push_str("f(1, 2)\n");

        let module = compile_exec(&source).expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                pyre_interpreter::ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let compare_pc = (0..code.instructions.len())
            .find(|&pc| {
                matches!(
                    decode_instruction_at(&code, pc),
                    Some((Instruction::CompareOp { .. }, _))
                )
            })
            .expect("test bytecode should contain COMPARE_OP");

        let first_after_compare = decode_instruction_at(&code, compare_pc + 1)
            .map(|(instruction, _)| instruction)
            .expect("bytecode should continue after COMPARE_OP");
        assert!(
            instruction_is_trivia_between_compare_and_branch(first_after_compare),
            "test source should force trivia between COMPARE_OP and POP_JUMP_IF"
        );

        let code_ref =
            pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
                as *const ();
        install_test_jitcode(&code, code_ref);
        let mut frame = PyFrame::new(code.clone());
        frame.fix_array_ptrs();

        let mut ctx = TraceCtx::for_test(2);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.valuestackdepth = 0;
        sym.jitcode = jitcode_for(code_ref);

        let state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: compare_pc + 1,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        assert!(
            state.next_instruction_consumes_comparison_truth(),
            "branch fusion should survive EXTENDED_ARG/other trivia before the branch"
        );
    }

    // test_trace_code_step_preserves_comparison_truth_across_extended_arg_trivia
    // removed: the last_comparison_truth cache no longer exists. Trivia
    // skipping between COMPARE_OP and POP_JUMP_IF* is now verified through
    // the fused-dispatch path (try_fused_compare_goto_if_not), which uses
    // semantic_fallthrough_pc to locate the branch across trivia.

    // Tests for concrete_popped_value, concrete_binary_operands,
    // concrete_store_subscr_operands removed: these stack-based concrete
    // read methods were replaced by direct FrontendOp.concrete parameter passing.

    // test_concrete_branch_truth_reads_last_popped_slot removed:
    // concrete_branch_truth now requires explicit concrete parameter.

    // test_concrete_branch_truth_uses_cached_comparison_truth_without_stack_value
    // and test_truth_value_direct_caches_concrete_truth_for_raw_int_branch_consumer
    // (consolidated on main as test_branch_truth_concrete_cache_paths) removed:
    // all three exercised the last_comparison_truth / last_comparison_concrete_truth
    // cache fields. Those fields were eliminated when the
    // try_fused_compare_goto_if_not dispatcher consumed the fused
    // COMPARE_OP + POP_JUMP_IF* pair directly (pyjitpl.py:541-556 parity).

    // test_close_loop_args_at_target_pc_preserves_virtualizable_stack moved
    // to `pyre-jit` so close_loop_args_at runs with a real compiled jitcode.

    /// `pyjitpl.py:74-90 MIFrame.setup` parity: after
    /// `setup_kind_register_banks` runs, `registers_i` / `registers_r` /
    /// `registers_f` are sized to `num_regs_X + len(constants_X)` and
    /// the trailing `[num_regs_X..)` slots hold the constant-pool
    /// OpRefs from `pyjitpl.py:97-119 copy_constants`
    /// (`ctx.const_int(constants_i[i])` for the int bank, `ctx.const_ref`
    /// for the ref bank, `ctx.const_float` for the float bank). The
    /// leading `[..num_regs_X)` register slots stay `OpRef::NONE` (the
    /// `CONST_NULL`-shaped placeholder).
    #[test]
    fn test_setup_kind_register_banks_sizes_per_num_regs_and_consts() {
        let mut runtime_jc = majit_metainterp::jitcode::JitCode::default();
        runtime_jc.body_mut().c_num_regs_i = 3;
        runtime_jc.body_mut().c_num_regs_r = 4;
        runtime_jc.body_mut().c_num_regs_f = 2;
        runtime_jc.body_mut().constants_i = vec![100, 200];
        runtime_jc.body_mut().constants_r = vec![0xAABB_CCDD_u64 as i64];
        runtime_jc.body_mut().constants_f = vec![3.14_f64.to_bits() as i64];

        let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null(), std::ptr::null(), None);
        pyjit.jitcode = std::sync::Arc::new(runtime_jc);
        let inner_jc = super::JitCode {
            code: std::ptr::null(),
            index: -1,
            payload: std::sync::Arc::new(pyjit),
        };
        let inner_jc_ptr = Box::into_raw(Box::new(inner_jc));

        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = inner_jc_ptr;

        assert_eq!(sym.registers_i.len(), 0);
        assert_eq!(sym.registers_r.len(), 0);
        assert_eq!(sym.registers_f.len(), 0);

        let mut ctx = TraceCtx::for_test(1);
        sym.setup_kind_register_banks(&mut ctx);

        // bank size = num_regs_X + len(constants_X)
        assert_eq!(sym.registers_i.len(), 3 + 2, "registers_i sized to 5");
        assert_eq!(sym.registers_r.len(), 4 + 1, "registers_r sized to 5");
        assert_eq!(sym.registers_f.len(), 2 + 1, "registers_f sized to 3");

        // Leading register slots — CONST_NULL placeholder per
        // pyjitpl.py:86-90.
        for i in 0..3 {
            assert_eq!(sym.registers_i[i], OpRef::NONE, "registers_i[{i}] reg-slot");
        }
        for i in 0..4 {
            assert_eq!(sym.registers_r[i], OpRef::NONE, "registers_r[{i}] reg-slot");
        }
        for i in 0..2 {
            assert_eq!(sym.registers_f[i], OpRef::NONE, "registers_f[{i}] reg-slot");
        }
        // Trailing constant slots — copy_constants populated. The pool
        // dedups by (type, raw value) so each kind's trailing slot
        // resolves to a distinct constant OpRef.
        for i in 0..2 {
            let op = sym.registers_i[3 + i];
            assert_ne!(op, OpRef::NONE, "registers_i[{}] constant slot", 3 + i);
            let val = ctx
                .constants_get_value(op)
                .expect("constants pool resolves trailing int slot");
            assert_eq!(val, majit_ir::Value::Int(100 + 100 * i as i64));
        }
        let op_r = sym.registers_r[4];
        assert_ne!(op_r, OpRef::NONE);
        assert!(matches!(
            ctx.constants_get_value(op_r),
            Some(majit_ir::Value::Ref(_))
        ));
        let op_f = sym.registers_f[2];
        assert_ne!(op_f, OpRef::NONE);
        assert!(matches!(
            ctx.constants_get_value(op_f),
            Some(majit_ir::Value::Float(_))
        ));

        // Calling twice does not shrink the register banks. `history.py:220
        // ConstInt`, `:261 ConstFloat`, `:307 ConstPtr` are all fresh-alloc
        // per construction; `Const.same_constant` (history.py:204) is the
        // upstream value-equality predicate. Assert value-equality across
        // both calls via `constants_get_value` rather than OpRef identity,
        // independent of whether the pool internally dedups.
        let trailing_i_value_before = ctx
            .constants_get_value(sym.registers_i[3])
            .expect("first-call trailing int slot resolves to a constant");
        let trailing_r_value_before = ctx
            .constants_get_value(sym.registers_r[4])
            .expect("first-call trailing ref slot resolves to a constant");
        let trailing_f_value_before = ctx
            .constants_get_value(sym.registers_f[2])
            .expect("first-call trailing float slot resolves to a constant");
        sym.setup_kind_register_banks(&mut ctx);
        assert_eq!(sym.registers_i.len(), 5);
        assert_eq!(sym.registers_r.len(), 5);
        assert_eq!(sym.registers_f.len(), 3);
        assert_eq!(
            ctx.constants_get_value(sym.registers_i[3]),
            Some(trailing_i_value_before),
        );
        assert_eq!(
            ctx.constants_get_value(sym.registers_r[4]),
            Some(trailing_r_value_before),
        );
        assert_eq!(
            ctx.constants_get_value(sym.registers_f[2]),
            Some(trailing_f_value_before),
        );

        // SAFETY: drop the boxed JitCode; sym.jitcode now dangles but goes
        // out of scope at the end of this test.
        unsafe {
            let _ = Box::from_raw(inner_jc_ptr);
        }
    }

    /// `setup_kind_register_banks` is safe to call when `self.jitcode`
    /// points at the thread-local `null_jitcode()` placeholder — the
    /// skeleton's `num_regs_and_consts_X` values are zero and the
    /// constant pools are empty so the resize and the constant fill are
    /// both no-ops.
    #[test]
    fn test_setup_kind_register_banks_is_no_op_for_null_jitcode_placeholder() {
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        // sym.jitcode was initialized by new_uninit to null_jitcode().
        let mut ctx = TraceCtx::for_test(1);
        sym.setup_kind_register_banks(&mut ctx);
        assert_eq!(sym.registers_i.len(), 0);
        assert_eq!(sym.registers_f.len(), 0);
        assert_eq!(sym.registers_r.len(), 0);
    }

    #[test]
    #[ignore = "PyreSym::new_uninit hits the Phase X-1 skeleton-panic since the \
                debug-only fallback was removed; needs a populated-jitcode harness."]
    fn test_setup_bridge_sym_preserves_resumed_stack_tail() {
        use majit_ir::resumedata::{RebuiltFrame, RebuiltValue};
        use majit_metainterp::jitcode::JitCodeBuilder;
        use pyre_interpreter::pyframe::PyFrame;

        ensure_test_callbacks();

        let raw_code = compile_exec("len(x)").expect("test code should compile");
        let mut frame = PyFrame::new(raw_code);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        let code_ref = frame.pycode as *const ();

        let live_off = crate::state::intern_liveness(&[], &[0, 1, 2], &[])
            .expect("bridge liveness must fit in the shared buffer");
        let mut builder = JitCodeBuilder::new();
        let patch = builder.live_placeholder();
        builder.patch_live_offset(patch, live_off);
        let runtime_jitcode = std::sync::Arc::new(builder.finish());
        let pyjit = std::sync::Arc::new(crate::PyJitCode::from_parts(
            runtime_jitcode,
            crate::PyJitCodeMetadata {
                pc_map: vec![0],
                depth_at_py_pc: vec![2],
                portal_frame_reg: 0,
                portal_ec_reg: 0,
                stack_base: 1,
                stack_slot_color_map: Vec::new(),
                pyre_color_for_semantic_local: Vec::new(),
            },
            std::ptr::null(),
            code_ref,
            false,
            None,
        ));
        let jitcode_index = METAINTERP_SD.with(|r| unsafe {
            let ptr = r.borrow_mut().jitcode_for(code_ref, Some(pyjit));
            (*ptr).index
        });

        let input_types = [
            Type::Ref, // frame
            Type::Ref, // ec
            Type::Int, // next_instr
            Type::Ref, // pycode
            Type::Int, // valuestackdepth
            Type::Ref, // debugdata
            Type::Ref, // lastblock
            Type::Ref, // w_globals
            Type::Ref, // local0
            Type::Ref, // stack0
            Type::Ref, // stack1
        ];
        let mut ctx = TraceCtx::for_test_types(&input_types);
        // Slots 0 (frame) and 1 (ec) are both Ref-typed per `input_types`
        // — production `init_vable_indices` mints typed `InputArgRef`
        // variants here (resoperation.py:739, state.rs:1428-1438), so
        // variant-aware Eq (resoperation.rs:290) requires the matching
        // `OpRef::input_arg_ref` shape.
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        sym.frame = OpRef::input_arg_ref(0);
        sym.execution_context = OpRef::input_arg_ref(1);
        sym.nlocals = 1;
        sym.valuestackdepth = 1;
        sym.concrete_vable_ptr = frame_ptr as *mut u8;

        let local0 = w_int_new(41) as i64;
        let stack0 = w_int_new(42) as i64;
        let stack1 = w_int_new(43) as i64;
        let globals = w_int_new(44) as i64;
        let fail_values = [
            frame_ptr as i64,
            0,
            77,
            code_ref as i64,
            3,
            0,
            0,
            globals,
            local0,
            stack0,
            stack1,
        ];
        let fail_types = [
            Type::Ref,
            Type::Ref,
            Type::Int,
            Type::Ref,
            Type::Int,
            Type::Ref,
            Type::Ref,
            Type::Ref,
            Type::Ref,
            Type::Ref,
            Type::Ref,
        ];
        let resume_data = majit_metainterp::ResumeDataResult {
            frames: vec![RebuiltFrame {
                jitcode_index,
                pc: 0,
                values: vec![
                    RebuiltValue::Box(8, Type::Ref),
                    RebuiltValue::Box(9, Type::Ref),
                    RebuiltValue::Box(10, Type::Ref),
                ],
            }],
            virtualizable_values: vec![
                RebuiltValue::Box(0, Type::Ref),
                RebuiltValue::Box(2, Type::Int),
                RebuiltValue::Box(3, Type::Ref),
                RebuiltValue::Box(4, Type::Int),
                RebuiltValue::Box(5, Type::Ref),
                RebuiltValue::Box(6, Type::Ref),
                RebuiltValue::Box(7, Type::Ref),
                RebuiltValue::Box(8, Type::Ref),
                RebuiltValue::Box(9, Type::Ref),
                RebuiltValue::Box(10, Type::Ref),
            ],
            virtualref_values: Vec::new(),
            storage: None,
            num_failargs: fail_values.len() as i32,
            fail_arg_types: fail_types.to_vec(),
        };

        <PyreJitState as majit_metainterp::JitState>::setup_bridge_sym(
            &mut sym,
            &mut ctx,
            &resume_data,
            None,
            &fail_values,
            &fail_types,
        );

        assert_eq!(sym.valuestackdepth, 3);
        // setup_bridge_sym now restores typed `InputArg*` OpRefs from
        // `RebuiltValue::Box(idx, tp)` per state.rs:4647. The resolve
        // closure produces `OpRef::input_arg_typed(idx, tp)`; expectations
        // must match so variant-aware Eq lines up with the resolved
        // bridge inputarg list.
        assert_eq!(
            sym.registers_r,
            vec![
                OpRef::input_arg_ref(8),
                OpRef::input_arg_ref(9),
                OpRef::input_arg_ref(10)
            ]
        );
        assert_eq!(sym.symbolic_local_types, vec![Type::Ref]);
        assert_eq!(sym.symbolic_stack_types, vec![Type::Ref, Type::Ref]);
        assert_eq!(sym.bridge_local_oprefs, Some(vec![OpRef::input_arg_ref(8)]));
    }

    #[test]
    #[ignore = "PyreSym::new_uninit hits the Phase X-1 skeleton-panic since the \
                debug-only fallback was removed; needs a populated-jitcode harness."]
    fn test_close_loop_args_preserves_ec_between_frame_and_virtualizable_header() {
        ensure_test_callbacks();
        let input_types = [
            Type::Ref, // frame
            Type::Ref, // ec
            Type::Int, // next_instr
            Type::Ref, // pycode
            Type::Int, // valuestackdepth
            Type::Ref, // debugdata
            Type::Ref, // lastblock
            Type::Ref, // w_globals
            Type::Ref, // local0
            Type::Ref, // stack0
            Type::Ref, // stack1
        ];
        let mut ctx = TraceCtx::for_test_types(&input_types);

        // The vable static-field types come from `state.rs:1428-1438`
        // `#[vable(inputarg, type = ...)]` annotations: int/ref/int/ref/
        // ref/ref. Mint typed `OpRef::input_arg_*` variants matching
        // those tags so variant-aware Eq (resoperation.rs:290) lines up
        // with what the production `init_vable_indices` produces.
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        sym.execution_context = OpRef::input_arg_ref(1);
        sym.nlocals = 1;
        sym.valuestackdepth = 3;
        sym.vable_last_instr = OpRef::input_arg_int(2);
        sym.vable_pycode = OpRef::input_arg_ref(3);
        sym.vable_valuestackdepth = OpRef::input_arg_int(4);
        sym.vable_debugdata = OpRef::input_arg_ref(5);
        sym.vable_lastblock = OpRef::input_arg_ref(6);
        sym.vable_w_globals = OpRef::input_arg_ref(7);
        // local0 / stack0 / stack1 are Ref-typed per `symbolic_local_types`
        // / `symbolic_stack_types` below — the macro mints the matching
        // `InputArgRef` variant.
        sym.registers_r = vec![
            OpRef::input_arg_ref(8),
            OpRef::input_arg_ref(9),
            OpRef::input_arg_ref(10),
        ];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack_types = vec![Type::Ref, Type::Ref];
        sym.concrete_stack = vec![ConcreteValue::Null, ConcreteValue::Null];

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let jump_args = state.with_ctx(|this, ctx| this.close_loop_args(ctx));

        assert_eq!(jump_args.len(), 11);
        assert_eq!(jump_args[0], OpRef::input_arg_ref(0));
        assert_eq!(jump_args[1], OpRef::input_arg_ref(1));
        assert_eq!(
            &jump_args[8..],
            &[
                OpRef::input_arg_ref(8),
                OpRef::input_arg_ref(9),
                OpRef::input_arg_ref(10)
            ]
        );
        assert_eq!(state.sym().execution_context, OpRef::input_arg_ref(1));
    }

    #[test]
    #[ignore = "PyreSym::new_uninit hits the Phase X-1 skeleton-panic since the \
                debug-only fallback was removed; needs a populated-jitcode harness."]
    fn test_close_loop_args_uses_full_root_virtualizable_array_capacity() {
        use pyre_interpreter::pyframe::PyFrame;

        ensure_test_callbacks();
        let code = compile_exec("len(x)").expect("test code should compile");
        let mut frame = PyFrame::new(code);
        frame.locals_w_mut()[0] = w_int_new(41);
        frame.locals_w_mut()[1] = w_int_new(7);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        let array_len =
            concrete_frame_array_len(frame_ptr).expect("frame should expose locals_cells_stack_w");
        assert!(
            array_len >= 2,
            "test frame must have stack capacity beyond nlocals"
        );

        let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let mut input_types = vec![
            Type::Ref, // frame
            Type::Int, // next_instr
            Type::Ref, // pycode
            Type::Int, // valuestackdepth
            Type::Ref, // debugdata
            Type::Ref, // lastblock
            Type::Ref, // w_globals
        ];
        input_types.extend(std::iter::repeat(Type::Ref).take(array_len));
        let mut ctx = TraceCtx::for_test_types(&input_types);

        // Mint typed `OpRef::input_arg_*` matching each `input_types`
        // slot — production `init_vable_indices` (state.rs:1428-1438
        // `#[vable(inputarg, type = ...)]`) always selects `InputArgInt`
        // / `InputArgRef` (resoperation.py:719/739) per the static-field
        // tag, so variant-aware Eq (resoperation.rs:290) requires the
        // matching variant here too.
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        sym.nlocals = 1;
        sym.valuestackdepth = 1;
        sym.vable_last_instr = OpRef::input_arg_int(1);
        sym.vable_pycode = OpRef::input_arg_ref(2);
        sym.vable_valuestackdepth = OpRef::input_arg_int(3);
        sym.vable_debugdata = OpRef::input_arg_ref(4);
        sym.vable_lastblock = OpRef::input_arg_ref(5);
        sym.vable_w_globals = OpRef::input_arg_ref(6);
        sym.registers_r = vec![OpRef::input_arg_ref(7)];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack_types = Vec::new();
        sym.concrete_vable_ptr = frame_ptr as *mut u8;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: frame_ptr,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let jump_args = state.with_ctx(|this, ctx| this.close_loop_args(ctx));

        assert_eq!(
            jump_args.len(),
            n + array_len,
            "root close_loop_args must carry full virtualizable array capacity"
        );
        assert!(
            jump_args[n..].iter().all(|arg| !arg.is_none()),
            "full-capacity root jump args must materialize missing stack slots"
        );
    }
}

// ── Virtualizable configuration ──────────────────────────────────────
//
// PyPy's `pypy/interpreter/pyframe.py` declares:
//
//     _virtualizable_ = ['locals_stack_w[*]', 'valuestackdepth',
//                         'last_instr', ...]
//
// Our Rust equivalent uses explicit byte offsets instead of name-based
// introspection. The JIT optimizer's Virtualize pass uses this info
// to keep frame fields in CPU registers, eliminating heap accesses
// for LoadFast/StoreFast and stack push/pop during compiled code.
//
// The shared frame layout contract now also lives in `pyre-jit/src/frame_layout.rs`
// so the tracer can compute the same offsets without depending on
// `pyre-interpreter`. Driver registration still happens in `pyre-jit/src/eval.rs`.

/// TODO: deferred `MetaInterp.perform_call`
/// (`rpython/jit/metainterp/pyjitpl.py`) for pyre.  RPython constructs
/// and pushes the callee `MIFrame` directly inside `perform_call`; pyre
/// returns this struct from the trace step so the framestack mutation
/// happens in `MetaInterpreter::push_inline_frame` after the trace
/// handler releases its borrow on `MetaInterpFrame`.  No upstream
/// counterpart.
pub struct PendingInlineFrame {
    pub sym: PyreSym,
    pub concrete_frame: pyre_interpreter::pyframe::PyFrame,
    pub drop_frame_opref: Option<OpRef>,
    pub green_key: u64,
    /// Raw `(code_ptr, target_pc)` greenkey components for element-
    /// wise recursion-depth comparison. `green_key` above is the u64
    /// hash derived from this pair and stays the identity key for
    /// HashMap lookups; `green_key_raw` element-wise equality matches
    /// rpython/jit/metainterp/pyjitpl.py:1396-1401 `for i in
    /// range(len(gk)): if not gk[i].same_constant(greenboxes[i])`
    /// without the hash-collision risk a u64-only comparison carries.
    pub green_key_raw: (usize, usize),
    /// opencoder.py:819-834: accumulated parent frame chain.
    pub parent_frames: Vec<ResumeFrameState>,
    pub nargs: usize,
    pub caller_result_stack_idx: Option<usize>,
    pub caller_result_type: Option<Type>,
}

/// Reify one Ref-bank recipe slot as the boxed `W_Root` pointer that
/// `locals_cells_stack_w` (a W_Root array, virtualizable.py:86-98) must
/// hold. `reconstruct_inline_recipe` gates int/float banks out of multi-
/// frame recipes, so a live slot is always `Value::Ref` (the decoded box,
/// possibly null) and a dead slot is `Value::Void`; the int/float arms only
/// exist defensively (box back to a heap object) and never fire in practice.
fn recipe_slot_to_pyobj(v: majit_ir::Value) -> PyObjectRef {
    match v {
        majit_ir::Value::Ref(gc) => gc.0 as PyObjectRef,
        majit_ir::Value::Void => pyre_object::PY_NULL,
        majit_ir::Value::Int(i) => w_int_new(i),
        majit_ir::Value::Float(f) => pyre_object::w_float_new(f),
    }
}

/// Assemble one decoded inline-callee [`ReconstructRecipe`]
/// into a [`PendingInlineFrame`] (concrete `PyFrame` + symbolic `PyreSym`)
/// for `trace_bytecode` to push onto the bridge framestack.
///
/// Mirrors `build_pending_inline_frame`'s IR-FREE FAST branch
/// (`trace_opcode.rs:6035-6084`): set `jitcode` FIRST (the
/// `setup_kind_register_banks` debug_assert requires a non-null jitcode),
/// then fill the per-bank register files + the lazy-boxing concrete shadow.
/// It emits NO trace IR — the forward FAST branch's guard_value /
/// callee-frame helper / GUARD_NO_EXCEPTION belong to a LIVE caller making a
/// call; a reconstructed suspended frame is simply pushed.
///
/// The callee frame's globals come from its OWN pycode (`recipe.w_code`),
/// matching `pyframe.py:128-132 get_w_globals()` where a frame's globals
/// derive from its promoted pycode rather than the caller; a cross-module
/// inlined callee's LOAD_GLOBAL then resolves against the callee module's
/// namespace. `execution_context` is the per-thread singleton and comes from
/// the bridge's root concrete frame.
///
/// `parent_frames` is the OUTER chain (immediate parent first) the drain
/// builds from the framestack; `push_inline_frame` stamps
/// `parent_frames.first().pending_result_*` with the caller result slot.
pub(crate) fn assemble_bridge_inline_pending(
    ctx: &mut TraceCtx,
    recipe: &ReconstructRecipe,
    execution_context: *const pyre_interpreter::PyExecutionContext,
    parent_frames: Vec<ResumeFrameState>,
) -> PendingInlineFrame {
    use pyre_interpreter::pyframe::PyFrame;

    let nlocals = recipe.nlocals;
    let valuestackdepth = recipe.valuestackdepth;

    // pyframe.py:128-132 get_w_globals(): a frame's globals come from its OWN
    // pycode (`jit.promote(self.pycode).w_globals`), not the caller. Resolve
    // the callee's globals from `recipe.w_code` — the same `pycode.w_globals`
    // the forward inline path reads via `function_get_globals` — so a
    // cross-module inlined callee's LOAD_GLOBAL sees the callee module's
    // namespace. `reconstruct_inline_recipe` aborts the multi-frame path when
    // the callee code has no resolved globals, so `globals` is non-null here.
    let globals = unsafe { pyre_interpreter::w_code_get_w_globals(recipe.w_code as PyObjectRef) };
    let w_globals_obj = if globals.is_null() {
        pyre_object::PY_NULL
    } else {
        pyre_interpreter::baseobjspace::dict_storage_to_dict(globals)
    };

    // resume.py:1042-1057 newframe + reload: build a fresh concrete frame for
    // `recipe.w_code` and seed `locals_cells_stack_w[0..valuestackdepth]` from
    // the decoded boxes. The callee has no cells/freevars (gated in
    // `reconstruct_inline_recipe`), so `closure = PY_NULL` and the array
    // layout is `[locals | stack]` with `stack_base() == nlocals`.
    let mut concrete_frame = PyFrame::new_for_call_with_closure_and_globals_obj(
        recipe.w_code,
        &[],
        globals,
        w_globals_obj,
        execution_context,
        pyre_object::PY_NULL,
    );
    {
        let arr = concrete_frame.locals_w_mut();
        for k in 0..nlocals {
            arr[k] = recipe_slot_to_pyobj(recipe.concrete_r[k]);
        }
    }
    for k in nlocals..valuestackdepth {
        concrete_frame.push(recipe_slot_to_pyobj(recipe.concrete_r[k]));
    }
    // last_instr = recipe.pc - 1 so next_instr() resumes AT recipe.pc.
    concrete_frame.set_last_instr_from_next_instr(recipe.pc);

    // Symbolic side: mirror the FAST branch field-for-field.
    let mut sym = PyreSym::new_uninit(OpRef::NONE);
    sym.nlocals = nlocals;
    sym.valuestackdepth = valuestackdepth;
    // jitcode FIRST — setup_kind_register_banks debug_asserts non-null.
    sym.jitcode = jitcode_for(recipe.w_code);
    // The Ref bank IS the unified locals+stack register file decoded by
    // `reconstruct_inline_recipe`; int/float banks are empty (gated out).
    sym.registers_i = recipe.registers_i.clone();
    sym.registers_r = recipe.registers_r.clone();
    sym.registers_f = recipe.registers_f.clone();
    // locals_cells_stack_w is a W_Root array — every live slot is Ref.
    sym.symbolic_local_types = vec![Type::Ref; nlocals];
    sym.symbolic_stack_types = vec![Type::Ref; valuestackdepth - nlocals];
    // Box-identity shadow: unbox per the FAST branch (`ConcreteValue::
    // from_pyobj`), so a boxed int/float local tracks as Int/Float.
    sym.concrete_locals = (0..nlocals)
        .map(|k| ConcreteValue::from_pyobj(recipe_slot_to_pyobj(recipe.concrete_r[k])))
        .collect();
    sym.concrete_stack = (nlocals..valuestackdepth)
        .map(|k| ConcreteValue::from_pyobj(recipe_slot_to_pyobj(recipe.concrete_r[k])))
        .collect();
    sym.concrete_namespace = w_globals_obj;
    sym.concrete_execution_context = execution_context;
    // pyjitpl.py:74-90 MIFrame.setup: size the per-kind banks + copy_constants.
    // The constant tail lands at `[num_regs_X..]`, beyond the live
    // valuestackdepth prefix (`num_regs_r` is the full Ref register file),
    // so the live slots set above are preserved. `locals_cells_stack_array_
    // ref` stays NONE: dead (NONE) slots are never read — the liveness stream
    // only carries live values, so no heap fall-through is needed (and
    // `s.frame` is NONE, so emitting the array getfield would be invalid).
    sym.setup_kind_register_banks(ctx);

    PendingInlineFrame {
        sym,
        concrete_frame,
        // No virtual_ref for a reconstructed frame: the forward trace's
        // virtual_ref was already finished/encoded; None skips the
        // opimpl_virtual_ref emission in push_inline_frame.
        drop_frame_opref: None,
        // The reconstructed frame represents the same inlined call the
        // forward trace pushed at function entry; match its (code, 0)
        // greenkey identity for recursion-depth + inline-position tracking.
        green_key: crate::driver::make_green_key(recipe.w_code, 0),
        green_key_raw: (recipe.w_code as usize, 0),
        parent_frames,
        nargs: recipe.nargs,
        caller_result_stack_idx: None,
        caller_result_type: Some(Type::Ref),
    }
}

pub enum InlineTraceStepAction {
    Trace(TraceAction),
    PushFrame(PendingInlineFrame),
}

pub fn execute_inline_residual_call(
    frame: &mut pyre_interpreter::pyframe::PyFrame,
    nargs: usize,
) -> Result<(), pyre_interpreter::PyError> {
    let required = nargs + 2; // callable + null/self + args
    if frame.valuestackdepth < frame.stack_base() + required {
        return Err(pyre_interpreter::PyError::type_error(
            "inline residual call stack underflow",
        ));
    }

    let mut args = Vec::with_capacity(nargs);
    for _ in 0..nargs {
        args.push(frame.pop());
    }
    args.reverse();
    let _null_or_self = frame.pop();
    let callable = frame.pop();
    let result = pyre_interpreter::call::call_callable_inline_residual(frame, callable, &args)?;
    frame.push(result);
    Ok(())
}

// inline_trace_and_execute removed — replaced by PyreMetaInterp.interpret()
// which uses a single framestack for both root and inline frames.
// trace_through_callee removed — replaced by build_pending_inline_frame +
// MetaInterp.push_inline_frame (RPython perform_call parity).

/// `pypy/objspace/std/listobject.py:2390 is_plain_int1` parity.
///
/// IntegerListStrategy stores raw i64 and so cannot preserve the
/// W_*Object pointer identity of its elements. PyPy therefore demotes
/// to object strategy on insertion of any value whose exact type is
/// not `W_IntObject` — bools, int subclasses (PyPy `W_IntObjectUser`
/// from `interpreter/typedef.py:205 subcls`), and `W_LongObject` whose
/// value doesn't fit in a machine int.
///
/// Delegates to the single `pyre_object::is_plain_int1` helper that
/// already implements the full upstream predicate including the
/// `w_class != get_instantiate(&INT_TYPE)` int-subclass rejection
/// (`listobject.rs:235`); a previous in-place `py_type_check` shortcut
/// at this site was a deviation that mishandled int subclasses
/// because pyre stores them with `ob_type == &INT_TYPE` and only
/// distinguishes them via `w_class` (`typedef.rs:686 w_int_new_unique`).
pub unsafe fn int_strategy_preserves_identity(value: pyre_object::PyObjectRef) -> bool {
    unsafe { pyre_object::is_plain_int1(value) }
}

#[cfg(test)]
mod indirectcalltargets_tests {
    //! Line-by-line parity tests for `pyjitpl.py:2248-2249` and
    //! `pyjitpl.py:2326-2343`.  Tests exercise the local
    //! `MetaInterpStaticData` methods directly — independent of the
    //! thread-local `METAINTERP_SD` singleton so concurrent callers
    //! (and unrelated tests that use the thread-local) do not alias.
    use super::{METAINTERP_SD, MetaInterpStaticData, raw_code_for_jitcode_index};
    use majit_metainterp::jitcode::{JitCode, JitCodeBuilder};
    use pyre_interpreter::bytecode::CodeObject;
    use std::sync::Arc;

    fn make_jitcode_with_fnaddr(fnaddr: usize) -> Arc<JitCode> {
        let mut jc = JitCodeBuilder::default().finish();
        jc.fnaddr = fnaddr as i64;
        Arc::new(jc)
    }

    /// Freshly-constructed staticdata has no indirect-call targets.
    /// Delegates through `canonical`; matches the behaviour of
    /// `majit_metainterp::MetaInterpStaticData::new()` where the
    /// `MetaInterpGlobalData` inside starts with every lazy cache
    /// (`indirectcall_dict`, `addr2name`) set to `None`.
    #[test]
    fn bytecode_for_address_none_when_empty() {
        let mut sd = MetaInterpStaticData::new();
        assert!(sd.bytecode_for_address(0xdeadbeef).is_none());
    }

    /// `pyjitpl.py:2326-2343` hit path: registered fnaddrs resolve to
    /// their JitCode.
    #[test]
    fn bytecode_for_address_returns_jitcode_when_registered() {
        let mut sd = MetaInterpStaticData::new();
        let j100 = make_jitcode_with_fnaddr(0x100);
        let j200 = make_jitcode_with_fnaddr(0x200);
        let j300 = make_jitcode_with_fnaddr(0x300);
        sd.setup_indirectcalltargets(vec![j100.clone(), j200.clone(), j300.clone()]);

        assert!(Arc::ptr_eq(&sd.bytecode_for_address(0x100).unwrap(), &j100));
        assert!(Arc::ptr_eq(&sd.bytecode_for_address(0x200).unwrap(), &j200));
        assert!(Arc::ptr_eq(&sd.bytecode_for_address(0x300).unwrap(), &j300));
        assert!(sd.bytecode_for_address(0x400).is_none());
    }

    /// `pyjitpl.py:2248-2249` `setup_indirectcalltargets` parity:
    /// every call replaces the targets list and invalidates the lazy
    /// dict so the next lookup rebuilds from the new list.
    #[test]
    fn setup_indirectcalltargets_invalidates_cache() {
        let mut sd = MetaInterpStaticData::new();
        sd.setup_indirectcalltargets(vec![make_jitcode_with_fnaddr(0x100)]);
        assert!(sd.bytecode_for_address(0x100).is_some());
        sd.setup_indirectcalltargets(vec![
            make_jitcode_with_fnaddr(0x200),
            make_jitcode_with_fnaddr(0x300),
        ]);
        assert!(sd.bytecode_for_address(0x100).is_none());
        assert!(sd.bytecode_for_address(0x200).is_some());
        assert!(sd.bytecode_for_address(0x300).is_some());
    }

    fn make_code(source: &str) -> (*const (), *const CodeObject) {
        let raw_code = pyre_interpreter::compile_exec(source).expect("source must compile");
        let code = pyre_interpreter::w_code_new(Box::into_raw(Box::new(raw_code)) as *const ())
            as *const ();
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject
        };
        (code, raw_code)
    }

    fn populated_pyjit(raw_code: *const CodeObject, code: *const ()) -> Arc<crate::PyJitCode> {
        let mut pyjit = crate::PyJitCode::skeleton(raw_code, code, None);
        pyjit.metadata.pc_map.push(0);
        Arc::new(pyjit)
    }

    #[should_panic(expected = "make_jitcodes returned an unpopulated JitCode skeleton")]
    #[test]
    fn install_jitcodes_rejects_skeleton_payload() {
        let mut sd = MetaInterpStaticData::new();
        let (code, raw_code) = make_code("x = 1\n");
        let skeleton = Arc::new(crate::PyJitCode::skeleton(raw_code, code, None));
        sd.set_jitcodes_from_make_result(vec![skeleton]);
    }

    #[test]
    fn compiled_jitcode_lookup_returns_populated_entry() {
        let mut sd = MetaInterpStaticData::new();
        let (code, raw_code) = make_code("x = 1\n");
        sd.set_jitcodes_from_make_result(vec![populated_pyjit(raw_code, code)]);

        let hit = sd
            .compiled_jitcode_lookup(code)
            .expect("populated payload should be installed by make_jitcodes");
        assert!(std::ptr::eq(sd.jitcodes[0].as_ref(), hit));
    }

    #[test]
    fn bulk_publish_replaces_existing_portal_bridge_payload_without_moving_entry() {
        let mut sd = MetaInterpStaticData::new();
        let (code, raw_code) = make_code("x = 1\n");
        let bridge_ptr = sd.portal_bridge_jitcode_for(code);
        let bridge_index = unsafe { (*bridge_ptr).index };

        sd.set_jitcodes_from_make_result(vec![populated_pyjit(raw_code, code)]);

        let hit = sd
            .compiled_jitcode_lookup(code)
            .expect("portal bridge entry should remain installed");
        assert!(std::ptr::eq(hit, bridge_ptr));
        assert_eq!(unsafe { (*hit).index }, bridge_index);
        assert!(!unsafe { &*hit }.payload.is_portal_bridge());
        assert_eq!(unsafe { (*hit).raw_code() }, raw_code);
    }

    #[test]
    fn compiled_jitcode_lookup_scans_by_raw_code_identity() {
        let mut sd = MetaInterpStaticData::new();
        let (code, raw_code) = make_code("x = 1\n");
        sd.set_jitcodes_from_make_result(vec![populated_pyjit(raw_code, code)]);

        let hit = sd
            .compiled_jitcode_lookup(code)
            .expect("wrapper pointer should resolve through raw CodeObject identity");
        assert_eq!(unsafe { (*hit).raw_code() }, raw_code);
    }

    #[test]
    fn raw_code_for_jitcode_index_returns_canonical_graph_pointer() {
        let mut sd = MetaInterpStaticData::new();
        let (code, expected_raw) = make_code("x = 1\n");
        sd.set_jitcodes_from_make_result(vec![populated_pyjit(expected_raw, code)]);
        METAINTERP_SD.with(|slot| {
            *slot.borrow_mut() = sd;
        });

        let hit = raw_code_for_jitcode_index(0).expect("jitcode index 0 must resolve");
        assert_eq!(hit, expected_raw);
    }
}

#[derive(Clone, Copy)]
pub struct ResumeFrameState {
    pub sym: *mut PyreSym,
    pub concrete_frame_addr: usize,
    pub resume_pc: usize,
    /// pyjitpl.py:181-193 `get_list_of_active_boxes(in_a_call=True)`.
    /// Non-top frames clear the caller's pending result slot before
    /// snapshotting liveness so the undefined call result does not leak
    /// stale boxes into guard fail_args.
    pub pending_result_stack_idx: Option<usize>,
    pub pending_result_type: Option<Type>,
}

#[cfg(test)]
mod finish_setup_tests {
    use super::{METAINTERP_SD, MetaInterpStaticData, blackhole_control_opcodes};
    use crate::assembler::publish_state;

    #[test]
    fn finish_setup_refreshes_opcode_cache_after_initial_empty_snapshot() {
        let mut sd = MetaInterpStaticData::new();
        let empty: majit_ir::VecAssoc<String, u8> = majit_ir::VecAssoc::new();
        sd.finish_setup_if_needed(&empty, Vec::new());
        assert_eq!(sd.op_live, u8::MAX);
        assert_eq!(sd.op_goto, u8::MAX);

        let mut insns = majit_ir::VecAssoc::new();
        insns.insert("live/".to_string(), 88u8);
        insns.insert("goto/L".to_string(), 16u8);
        insns.insert("catch_exception/L".to_string(), 89u8);
        insns.insert("rvmprof_code/ii".to_string(), 91u8);
        insns.insert("int_return/i".to_string(), 148u8);
        insns.insert("ref_return/r".to_string(), 76u8);
        insns.insert("float_return/f".to_string(), 149u8);
        insns.insert("void_return/".to_string(), 150u8);

        sd.finish_setup_if_needed(&insns, vec![1, 2, 3]);
        assert_eq!(sd.op_live, 88);
        assert_eq!(sd.op_goto, 16);
        assert_eq!(sd.op_catch_exception, 89);
        assert_eq!(sd.op_rvmprof_code, 91);
        assert_eq!(sd.op_int_return, 148);
        assert_eq!(sd.op_ref_return, 76);
        assert_eq!(sd.op_float_return, 149);
        assert_eq!(sd.op_void_return, 150);
        assert_eq!(&*sd.liveness_info, &[1u8, 2, 3][..]);
        assert!(sd.finish_setup_done);
    }

    #[test]
    fn blackhole_control_opcodes_reflect_finish_setup_cache() {
        let mut sd = MetaInterpStaticData::new();
        let mut insns = majit_ir::VecAssoc::new();
        insns.insert("live/".to_string(), 88u8);
        insns.insert("catch_exception/L".to_string(), 89u8);
        insns.insert("rvmprof_code/ii".to_string(), 91u8);
        sd.finish_setup_if_needed(&insns, Vec::new());
        METAINTERP_SD.with(|slot| {
            *slot.borrow_mut() = sd;
        });

        assert_eq!(blackhole_control_opcodes(), (88, 89, 91));
    }

    #[test]
    fn blackhole_control_opcodes_refresh_after_initial_empty_snapshot() {
        let mut sd = MetaInterpStaticData::new();
        let empty: majit_ir::VecAssoc<String, u8> = majit_ir::VecAssoc::new();
        sd.finish_setup_if_needed(&empty, Vec::new());
        METAINTERP_SD.with(|slot| {
            *slot.borrow_mut() = sd;
        });

        let mut insns = majit_ir::VecAssoc::new();
        insns.insert("live/".to_string(), 88u8);
        insns.insert("catch_exception/L".to_string(), 89u8);
        insns.insert("rvmprof_code/ii".to_string(), 91u8);
        publish_state(&insns, &[], 0, 0);

        assert_eq!(blackhole_control_opcodes(), (88, 89, 91));
    }
}
