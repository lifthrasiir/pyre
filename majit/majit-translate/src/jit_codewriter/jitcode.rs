//! JitCode — assembled bytecode + register/constant pools.
//!
//! RPython equivalent: `rpython/jit/codewriter/jitcode.py` class `JitCode`.
//!
//! In RPython this is a single shared type used by both the codewriter
//! (which writes into it via `Assembler.assemble`) and the metainterp
//! (which reads from it via `BlackholeInterpreter.dispatch_loop` and
//! `MetaInterp.handle_call_assembler`). majit currently has two `JitCode`
//! types — this `codewriter::jitcode::JitCode` (RPython orthodox encoding,
//! `insns` dict, dynamic argcodes) and `metainterp::jitcode::JitCode`
//! (pyre-specific BC_* hardcoded opcode set). Phase D will line-by-line
//! port `BlackholeInterpreter.setup_insns` so the metainterp can consume
//! this type directly, eliminating the fork.

use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// Assembled JitCode — the output of the assembler.
///
/// RPython parity (`rpython/jit/codewriter/jitcode.py:9-43`):
///
/// ```python
/// class JitCode(AbstractDescr):
///     def __init__(self, name, fnaddr=None, calldescr=None, called_from=None):
///         self.name = name
///         self.fnaddr = fnaddr
///         self.calldescr = calldescr
///         self.jitdriver_sd = None
///         self._called_from = called_from
///         self._ssarepr = None
///
///     def setup(self, code='', constants_i=[], constants_r=[], constants_f=[],
///               num_regs_i=255, num_regs_r=255, num_regs_f=255,
///               startpoints=None, alllabels=None, resulttypes=None):
///         self.code = code
///         self.constants_i = constants_i or self._empty_i
///         self.constants_r = constants_r or self._empty_r
///         self.constants_f = constants_f or self._empty_f
///         self.c_num_regs_i = chr(num_regs_i)
///         self.c_num_regs_r = chr(num_regs_r)
///         self.c_num_regs_f = chr(num_regs_f)
///         self._startpoints = startpoints
///         self._alllabels = alllabels
///         self._resulttypes = resulttypes
/// ```
///
/// Field-by-field mapping below preserves the RPython names. Where
/// RPython uses `chr(int)` to pack a 0..255 register count into a single
/// byte we use `u8` directly; the value range is identical.
/// Body of a `JitCode` — populated once by the assembler after
/// `transform_graph_to_jitcode` runs the full codewriter pipeline.
///
/// RPython `jitcode.py:22-42` `JitCode.setup(...)`. RPython mutates the
/// JitCode object in place; pyre groups the late-set fields into a body
/// struct that is committed via `OnceLock::set` so `Arc<JitCode>` shells
/// handed out by `CallControl::get_jitcode` can be filled while shared.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct JitCodeBody {
    /// RPython `jitcode.py:17` `self.calldescr = calldescr`. RPython sets
    /// this at construction because rtyper has resolved the function's
    /// arg/result types upstream; pyre's rtyper-equivalent runs inside
    /// the codewriter pipeline so calldescr is filled here as part of
    /// the body. `transform_graph_to_jitcode` overrides the default with
    /// the assembled `arg_classes`.
    pub calldescr: BhCallDescr,
    /// RPython `jitcode.py:26` `self.code = code` — bytecode bytes.
    pub code: Vec<u8>,
    /// RPython `jitcode.py:32` `self.constants_i`.
    pub constants_i: Vec<i64>,
    /// RPython `jitcode.py:33` `self.constants_r` — GCREF constant pool.
    /// RPython uses `lltype.cast_opaque_ptr(GCREF, ...)`; pyre stores the
    /// raw 64-bit address as `i64` to match the runtime jitcode/blackhole
    /// register file (where `r` registers also flow through `i64`).
    pub constants_r: Vec<i64>,
    /// RPython `jitcode.py:34` `self.constants_f`.
    /// RPython packs the float as `longlong.FLOATSTORAGE` (a 64-bit int
    /// reinterpretation); pyre stores the same bitwise representation as
    /// `i64` so the runtime register file can consume the pool entries
    /// without a re-bitcast.
    pub constants_f: Vec<i64>,
    /// RPython `jitcode.py:37-39` `self.c_num_regs_i = chr(num_regs_i)`.
    /// RPython packs into a single chr (`assert num_regs_i < 256`); pyre
    /// uses `u16` to keep CPython 3.13 codes that legitimately exceed 255
    /// registers per kind reachable.  The codewriter still asserts
    /// `< 256` for now (assembler.rs); widening the field is a parity
    /// preparation for that limit being lifted.
    pub c_num_regs_i: u16,
    /// RPython `jitcode.py:38` `self.c_num_regs_r = chr(num_regs_r)`.
    pub c_num_regs_r: u16,
    /// RPython `jitcode.py:39` `self.c_num_regs_f = chr(num_regs_f)`.
    pub c_num_regs_f: u16,
    /// RPython `jitcode.py:40` `self._startpoints = startpoints` —
    /// debug-only set of bytecode offsets where instructions start.
    /// `setup(..., startpoints=None)` (jitcode.py:24) is the upstream
    /// default; `None` here means "the assembler did not record start
    /// positions for this jitcode" (e.g. hand-built helper jitcodes).
    /// Assembled jitcodes always populate `Some(set)`, even when the
    /// set is empty. `blackhole.py:86 dispatch_loop` consults
    /// `_startpoints is not None` to gate its non-translated `pc in
    /// self._startpoints` assertion.
    pub startpoints: Option<HashSet<usize>>,
    /// RPython `jitcode.py:41` `self._alllabels = alllabels` — debug-only
    /// set of bytecode offsets that are label targets.
    /// `setup(..., alllabels=None)` (jitcode.py:24) is the upstream
    /// default; assembled jitcodes always populate `Some(set)`.
    pub alllabels: Option<HashSet<usize>>,
    /// RPython `jitcode.py:42` `self._resulttypes = resulttypes` —
    /// debug-only map from bytecode offset to result type char.  `None`
    /// is the exact `JitCode.setup(..., resulttypes=None)` sentinel;
    /// assembled jitcodes store `Some(dict)`, even when the dict is empty.
    pub resulttypes: Option<HashMap<usize, char>>,
    /// RPython `jitcode.py:20` `self._ssarepr = None` — debug: the
    /// flattened SSA representation, kept for `dump()` output. Set by
    /// `Assembler.assemble` (assembler.py:49 `jitcode._ssarepr = ssarepr`).
    #[serde(skip)]
    pub _ssarepr: Option<crate::flatten::SSARepr>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JitCode {
    /// RPython `jitcode.py:15` `self.name = name`.
    pub name: String,
    /// RPython `jitcode.py:16` `self.fnaddr = fnaddr`. majit stores the
    /// bound helper trace-call address when the host has supplied one,
    /// otherwise the stable symbolic fallback key; the blackhole-side
    /// inline-call descriptor may still patch its own cached copy from
    /// `all_jitcodes[jitcode.index]`.
    #[serde(default)]
    pub fnaddr: i64,
    /// RPython `jitcode.py:18` `self.jitdriver_sd = None`. `Some(index)`
    /// for portal jitcodes (set by `grab_initial_jitcodes` /
    /// `drain_pending_graphs`). `OnceLock` allows the late single-set
    /// after `Arc<JitCode>` shells have been cloned (e.g. into
    /// `JitDriverStaticData.mainjitcode`). Use `jitdriver_sd()` /
    /// `set_jitdriver_sd()`.
    #[serde(with = "oncelock_usize_serde")]
    pub jitdriver_sd: OnceLock<usize>,
    /// RPython `codewriter.py:68` `jitcode.index = index` — sequential
    /// position in `all_jitcodes[]`. Set once when the codewriter has
    /// finished assembling the jitcode and appended it to the completed
    /// list, matching upstream `CodeWriter.make_jitcodes()`.
    #[serde(with = "oncelock_usize_serde")]
    index: OnceLock<usize>,
    /// RPython `jitcode.py:19` `self._called_from = called_from` — debug:
    /// which call graph first triggered this jitcode's creation. In RPython
    /// this is a graph object; pyre uses an optional CallPath string.
    #[serde(default)]
    pub _called_from: Option<String>,
    /// Body — set once after assembly via `set_body`. Direct field accesses
    /// like `jitcode.code` continue to work via `Deref<Target=JitCodeBody>`.
    #[serde(with = "oncelock_body_serde")]
    body: OnceLock<JitCodeBody>,
}

mod oncelock_usize_serde {
    use std::sync::OnceLock;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(lock: &OnceLock<usize>, ser: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&lock.get().copied(), ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<OnceLock<usize>, D::Error> {
        let opt: Option<usize> = Option::deserialize(de)?;
        let lock = OnceLock::new();
        if let Some(v) = opt {
            let _ = lock.set(v);
        }
        Ok(lock)
    }
}

mod oncelock_body_serde {
    use std::sync::OnceLock;

    use serde::{Deserialize, Deserializer, Serializer};

    use super::JitCodeBody;

    pub fn serialize<S: Serializer>(
        lock: &OnceLock<JitCodeBody>,
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&lock.get(), ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        de: D,
    ) -> Result<OnceLock<JitCodeBody>, D::Error> {
        let opt: Option<JitCodeBody> = Option::deserialize(de)?;
        let lock = OnceLock::new();
        if let Some(v) = opt {
            let _ = lock.set(v);
        }
        Ok(lock)
    }
}

impl JitCode {
    /// RPython `jitcode.py:14-20` `JitCode.__init__(name, fnaddr=None,
    /// calldescr=None, called_from=None)`.
    ///
    /// Constructs a JitCode with name + default-initialized state. The
    /// `setup()` step (RPython `jitcode.py:22-42`) populates `code`,
    /// `constants_*`, `c_num_regs_*`, `startpoints`, etc. via the
    /// assembler.
    ///
    /// `calldescr`, `_called_from`, and `_ssarepr` from RPython are not
    /// fully ported at construction time. `fnaddr` starts as 0 here and is
    /// filled by `CallControl::get_jitcode()` when a graph-backed shell is
    /// allocated.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fnaddr: 0,
            jitdriver_sd: OnceLock::new(),
            index: OnceLock::new(),
            _called_from: None,
            body: OnceLock::new(),
        }
    }

    /// Body accessor — panics if `set_body` has not run.
    ///
    /// RPython does not have an explicit body/header split; pyre groups
    /// late-set fields here so `Arc<JitCode>` shells can be filled while
    /// shared (e.g. when `IndirectCallTargets` already holds clones).
    pub fn body(&self) -> &JitCodeBody {
        self.body
            .get()
            .expect("JitCode body not yet set — call set_body() before reading body fields")
    }

    /// Optional body accessor — returns `None` while the JitCode is still
    /// a shell awaiting assembly.
    pub fn try_body(&self) -> Option<&JitCodeBody> {
        self.body.get()
    }

    /// Mutable accessor for late post-assembly mutation of body fields.
    /// Required by callers (e.g. pyre's `finalize_jitcode`) that fetch
    /// `calldescr` from `CallControl` *after* the assembler has already
    /// committed the body via `set_body`. RPython mutates `JitCode`
    /// fields directly post-`setup()`; pyre routes the mutation through
    /// `OnceLock::get_mut` so the same in-place semantics work on
    /// canonical JitCode shells. Panics if the body has not been
    /// committed yet.
    pub fn body_mut(&mut self) -> &mut JitCodeBody {
        self.body
            .get_mut()
            .expect("JitCode body not yet set — call set_body() before body_mut()")
    }

    /// Commit the body once assembly has produced it. Panics on second
    /// call (RPython equivalent: `JitCode.setup` is also called once per
    /// jitcode lifetime).
    pub fn set_body(&self, body: JitCodeBody) {
        self.body
            .set(body)
            .map_err(|_| ())
            .expect("JitCode body already set");
    }

    /// `Some(idx)` when this jitcode is the portal of jitdriver `idx`.
    /// RPython `jitcode.py:18` `self.jitdriver_sd = None` (overwritten by
    /// `grab_initial_jitcodes` / `drain_pending_graphs`).
    pub fn jitdriver_sd(&self) -> Option<usize> {
        self.jitdriver_sd.get().copied()
    }

    /// RPython `jitcode.index` reader. Panics until the jitcode has been
    /// fully assembled and appended to `all_jitcodes[]`.
    pub fn index(&self) -> usize {
        *self
            .index
            .get()
            .expect("JitCode index not yet set — assemble and append it before reading index")
    }

    /// Optional reader for diagnostics while this JitCode is still only a
    /// shell on `unfinished_graphs`.
    pub fn try_index(&self) -> Option<usize> {
        self.index.get().copied()
    }

    /// RPython `codewriter.py:68 jitcode.index = index` — assigned once,
    /// at the moment the finished jitcode is appended to
    /// `all_jitcodes[]`.  Matches upstream `JitCode` Python-object
    /// identity semantics: a second `set_index` with a *different*
    /// value is a parity violation and panics.  A second `set_index`
    /// with the *same* value is treated as a no-op so concurrent
    /// readers and writers along the codewriter →
    /// `metainterp_sd.jitcodes` boundary can converge on the same
    /// value without forcing every caller to inspect `try_index`
    /// first (this matches the upstream observation that
    /// `jitcode.index = N; jitcode.index = N` is an idempotent write
    /// in Python).
    pub fn set_index(&self, idx: usize) {
        match self.index.set(idx) {
            Ok(()) => {}
            Err(_) => {
                let existing = *self
                    .index
                    .get()
                    .expect("OnceLock::set returned Err but get() is empty");
                assert_eq!(
                    existing, idx,
                    "JitCode index already set to {existing}, cannot reassign to {idx} \
                     — RPython codewriter.py:68 sets it exactly once",
                );
            }
        }
    }

    /// Set `jitdriver_sd` once. Panics on second call.
    pub fn set_jitdriver_sd(&self, idx: usize) {
        self.jitdriver_sd
            .set(idx)
            .expect("JitCode jitdriver_sd already set");
    }

    /// Replace `jitdriver_sd` (or clear it).  Requires `&mut self` so it
    /// cannot race with the `set_jitdriver_sd` interior-mutability path
    /// that production callers use.  Permissive so test fixtures can
    /// cycle a JitCode through several portal/non-portal states without
    /// allocating a fresh `JitCodeBuilder`. `set_jitdriver_sd` (single
    /// shot, `&self`) remains the only supported path in production
    /// because it matches RPython's `call.py:148` "set once at portal
    /// grab time" pattern.
    pub fn replace_jitdriver_sd(&mut self, value: Option<usize>) {
        self.jitdriver_sd = OnceLock::new();
        if let Some(idx) = value {
            let _ = self.jitdriver_sd.set(idx);
        }
    }

    /// RPython `jitcode.py:17` reader. Convenience for callers that
    /// would otherwise write `jitcode.body().calldescr`.
    pub fn calldescr(&self) -> &BhCallDescr {
        &self.body().calldescr
    }
}

/// Allow existing callers to keep `jitcode.code`, `jitcode.constants_i`,
/// `jitcode.startpoints`, etc. through `Deref<Target=JitCodeBody>`.
/// Panics if the body has not been committed yet.
impl Deref for JitCode {
    type Target = JitCodeBody;
    fn deref(&self) -> &JitCodeBody {
        self.body()
    }
}

impl JitCode {
    /// RPython `jitcode.py:114-119` `def dump(self)`:
    ///
    /// ```python
    /// def dump(self):
    ///     if self._ssarepr is None:
    ///         return '<no dump available for %r>' % (self.name,)
    ///     else:
    ///         from rpython.jit.codewriter.format import format_assembler
    ///         return format_assembler(self._ssarepr)
    /// ```
    pub fn dump(&self) -> String {
        match &self._ssarepr {
            None => format!("<no dump available for {:?}>", self.name),
            Some(ssarepr) => format_assembler(ssarepr),
        }
    }

    /// RPython `jitcode.py:47-48` `def num_regs_i(self): return ord(self.c_num_regs_i)`.
    pub fn num_regs_i(&self) -> usize {
        self.c_num_regs_i as usize
    }

    /// RPython `jitcode.py:50-51` `def num_regs_r(self): return ord(self.c_num_regs_r)`.
    pub fn num_regs_r(&self) -> usize {
        self.c_num_regs_r as usize
    }

    /// RPython `jitcode.py:53-54` `def num_regs_f(self): return ord(self.c_num_regs_f)`.
    pub fn num_regs_f(&self) -> usize {
        self.c_num_regs_f as usize
    }

    /// RPython `jitcode.py:56-57` `def num_regs_and_consts_i(self):
    /// return ord(self.c_num_regs_i) + len(self.constants_i)`.
    pub fn num_regs_and_consts_i(&self) -> usize {
        self.num_regs_i() + self.constants_i.len()
    }

    /// RPython `jitcode.py:59-60` `def num_regs_and_consts_r(self):
    /// return ord(self.c_num_regs_r) + len(self.constants_r)`.
    pub fn num_regs_and_consts_r(&self) -> usize {
        self.num_regs_r() + self.constants_r.len()
    }

    /// RPython `jitcode.py:62-63` `def num_regs_and_consts_f(self):
    /// return ord(self.c_num_regs_f) + len(self.constants_f)`.
    pub fn num_regs_and_consts_f(&self) -> usize {
        self.num_regs_f() + self.constants_f.len()
    }

    /// RPython `jitcode.py:102-114` `def follow_jump(self, position)`:
    /// "Assuming that 'position' points just after a bytecode instruction
    /// that ends with a label, follow that label."
    ///
    /// ```python
    /// def follow_jump(self, position):
    ///     code = self.code
    ///     position -= 2
    ///     assert position >= 0
    ///     if not we_are_translated():
    ///         assert position in self._alllabels
    ///     labelvalue = ord(code[position]) | (ord(code[position+1])<<8)
    ///     assert labelvalue < len(code)
    ///     return labelvalue
    /// ```
    ///
    /// pyre is "non-translated" today, so the
    /// `position in self._alllabels` assertion fires unconditionally
    /// — every label-bearing bytecode emit must record its position
    /// in `_alllabels` (RPython `assembler.py:setup_labels`, pyre
    /// `JitCodeBuilder::finish` derives it from the builder's
    /// `labels: Vec<Option<usize>>`).
    pub fn follow_jump(&self, position: usize) -> usize {
        // RPython `:104-105`: `position -= 2; assert position >= 0`.
        // `checked_sub` + `expect` mirrors the non-negativity assert.
        let position = position
            .checked_sub(2)
            .expect("follow_jump: position underflow before 2-byte label slot");
        // RPython `:107-108`: `if not we_are_translated(): assert
        // position in self._alllabels`. PyPy upstream does not gate the
        // assert on `_alllabels is not None` — `pc in None` would raise
        // TypeError; the contract is that any jitcode reaching
        // `follow_jump` was assembled (so `_alllabels = Some(set)`).
        // `debug_assert!` mirrors the non-translated guard — fires in
        // dev/test builds, elided in release just like RPython skips
        // the check post-translation.
        debug_assert!(
            self.alllabels
                .as_ref()
                .expect("follow_jump: _alllabels is None on a non-assembled jitcode")
                .contains(&position),
            "follow_jump: position {position} is not in _alllabels"
        );
        let labelvalue = (self.code[position] as usize) | ((self.code[position + 1] as usize) << 8);
        assert!(labelvalue < self.code.len(), "follow_jump out of range");
        labelvalue
    }

    /// RPython `jitcode.py:82-93` `get_live_vars_info(pc, op_live)`:
    ///
    /// ```python
    /// def get_live_vars_info(self, pc, op_live):
    ///     # either this, or the previous instruction must be -live-
    ///     if not we_are_translated():
    ///         assert pc in self._startpoints
    ///     if ord(self.code[pc]) != op_live:
    ///         pc -= OFFSET_SIZE + 1
    ///         if not we_are_translated():
    ///             assert pc in self._startpoints
    ///         if ord(self.code[pc]) != op_live:
    ///             self._missing_liveness(pc)
    ///     return decode_offset(self.code, pc + 1)
    /// ```
    ///
    /// `op_live` is the runtime opcode byte for `live/` (assigned by the
    /// blackhole interpreter at `setup_insns` time, RPython
    /// `blackhole.py:72`). The result is the offset into the metainterp's
    /// `all_liveness` table.
    pub fn get_live_vars_info(&self, pc: usize, op_live: u8) -> usize {
        // RPython `jitcode.py:85-90`: `if not we_are_translated(): assert
        // pc in self._startpoints`. Pyre is "non-translated" today so the
        // assertion fires in both canonical and runtime jitcodes — the
        // runtime `JitCodeBuilder` populates `startpoints` from each
        // opcode emit position. PyPy does not gate on `_startpoints is
        // not None` here — `pc in None` would raise TypeError; the
        // contract is that any jitcode whose liveness map is consulted
        // was assembled (`_startpoints = Some(set)`).
        debug_assert!(
            self.startpoints
                .as_ref()
                .expect("get_live_vars_info: _startpoints is None on a non-assembled jitcode")
                .contains(&pc),
            "pc not in startpoints",
        );
        let mut pc = pc;
        if self.code[pc] != op_live {
            pc -= crate::liveness::OFFSET_SIZE + 1;
            debug_assert!(
                self.startpoints
                    .as_ref()
                    .expect("get_live_vars_info: _startpoints is None on a non-assembled jitcode")
                    .contains(&pc),
                "pc not in startpoints",
            );
            if self.code[pc] != op_live {
                self.missing_liveness(pc);
            }
        }
        crate::liveness::decode_offset(&self.code, pc + 1)
    }

    /// RPython `jitcode.py:95-100` `_missing_liveness(self, pc)`:
    ///
    /// ```python
    /// def _missing_liveness(self, pc):
    ///     msg = "missing liveness[%d] in %s" % (pc, self.name)
    ///     if we_are_translated():
    ///         print(msg)
    ///         raise AssertionError
    ///     raise MissingLiveness(...)
    /// ```
    fn missing_liveness(&self, pc: usize) -> ! {
        panic!("missing liveness[{pc}] in {}", self.name);
    }
}

// RPython `jitcode.py:121-122` `def __repr__(self): return '<JitCode %r>' % self.name`.
impl std::fmt::Display for JitCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<JitCode {:?}>", self.name)
    }
}

impl Default for JitCode {
    fn default() -> Self {
        // Default placeholders (e.g. `Arc<JitCode>::default()` used by
        // `BlackholeInterpreter::new` before the first `setposition`)
        // need readable zero-size body fields.  Pre-collapse the
        // runtime `JitCode::default()` derived `Default` and
        // therefore returned all-zero numeric fields with empty Vecs;
        // we preserve that observable behaviour by committing an empty
        // `JitCodeBody` upfront so callers like `cleanup_registers`
        // (which reads `num_regs_r()`) keep working without a
        // `setposition` first.
        let jc = Self::new(String::new());
        jc.set_body(JitCodeBody::default());
        jc
    }
}

impl Clone for JitCode {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            fnaddr: self.fnaddr,
            jitdriver_sd: self.jitdriver_sd.clone(),
            index: self.index.clone(),
            _called_from: self._called_from.clone(),
            body: self.body.clone(),
        }
    }
}

/// Identity-keyed handle around `Arc<JitCode>`, mirroring Python set/dict
/// behaviour where `JitCode` instances are deduped by object identity
/// (RPython `IndirectCallTargets.lst` is a list of JitCode objects;
/// `Assembler.indirectcalltargets` is a `set` of those objects keyed by
/// identity).
///
/// Callers use `JitCodeHandle::from(arc)` / `handle.into_inner()` to
/// move between the wrapper and the underlying `Arc<JitCode>`. Display
/// and Deref pass through to the inner JitCode.
#[derive(Debug, Clone)]
pub struct JitCodeHandle(pub std::sync::Arc<JitCode>);

impl JitCodeHandle {
    pub fn new(arc: std::sync::Arc<JitCode>) -> Self {
        Self(arc)
    }

    pub fn into_inner(self) -> std::sync::Arc<JitCode> {
        self.0
    }

    pub fn as_arc(&self) -> &std::sync::Arc<JitCode> {
        &self.0
    }
}

impl PartialEq for JitCodeHandle {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for JitCodeHandle {}

impl std::hash::Hash for JitCodeHandle {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (std::sync::Arc::as_ptr(&self.0) as *const () as usize).hash(state);
    }
}

impl std::ops::Deref for JitCodeHandle {
    type Target = JitCode;
    fn deref(&self) -> &JitCode {
        &self.0
    }
}

impl From<std::sync::Arc<JitCode>> for JitCodeHandle {
    fn from(arc: std::sync::Arc<JitCode>) -> Self {
        Self(arc)
    }
}

mod jitcode_handle_serde {
    use std::sync::Arc;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::{JitCode, JitCodeHandle};

    impl Serialize for JitCodeHandle {
        fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
            (*self.0).serialize(ser)
        }
    }

    impl<'de> Deserialize<'de> for JitCodeHandle {
        fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
            let jc = JitCode::deserialize(de)?;
            Ok(JitCodeHandle(Arc::new(jc)))
        }
    }
}

/// RPython `jitcode.py:146-167` module-level `enumerate_vars(offset,
/// all_liveness, callback_i, callback_r, callback_f, spec)`:
///
/// ```python
/// @specialize.arg(5)
/// def enumerate_vars(offset, all_liveness, callback_i, callback_r, callback_f, spec):
///     length_i = ord(all_liveness[offset])
///     length_r = ord(all_liveness[offset + 1])
///     length_f = ord(all_liveness[offset + 2])
///     offset += 3
///     if length_i:
///         it = LivenessIterator(offset, length_i, all_liveness)
///         for index in it: callback_i(index)
///         offset = it.offset
///     if length_r:
///         it = LivenessIterator(offset, length_r, all_liveness)
///         for index in it: callback_r(index)
///         offset = it.offset
///     if length_f:
///         it = LivenessIterator(offset, length_f, all_liveness)
///         for index in it: callback_f(index)
/// ```
///
/// Reads the `[len_i][len_r][len_f]` header at `offset`, then walks the
/// three packed bitsets (int, ref, float) via `LivenessIterator`, invoking
/// the matching callback for each live register index.
///
/// RPython places this in `rpython/jit/codewriter/jitcode.py` (not in
/// metainterp). majit follows the same module placement.
pub fn enumerate_vars(
    mut offset: usize,
    all_liveness: &[u8],
    mut callback_i: impl FnMut(u32),
    mut callback_r: impl FnMut(u32),
    mut callback_f: impl FnMut(u32),
) {
    use crate::liveness::LivenessIterator;
    // jitcode.py:149-151
    let length_i = all_liveness[offset] as u32;
    let length_r = all_liveness[offset + 1] as u32;
    let length_f = all_liveness[offset + 2] as u32;
    // jitcode.py:152
    offset += 3;
    // jitcode.py:153-157
    if length_i != 0 {
        let mut it = LivenessIterator::new(offset, length_i, all_liveness);
        for index in &mut it {
            callback_i(index);
        }
        offset = it.offset;
    }
    // jitcode.py:158-162
    if length_r != 0 {
        let mut it = LivenessIterator::new(offset, length_r, all_liveness);
        for index in &mut it {
            callback_r(index);
        }
        offset = it.offset;
    }
    // jitcode.py:163-166
    if length_f != 0 {
        let mut it = LivenessIterator::new(offset, length_f, all_liveness);
        for index in &mut it {
            callback_f(index);
        }
    }
}

/// RPython `jitcode.py:127-128` `class MissingLiveness(Exception): pass`.
///
/// Raised by `JitCode::get_live_vars_info` when a `-live-` op is missing
/// at the expected PC. Currently we panic instead of returning a typed
/// error since pyre's blackhole has no exception-based error path yet.
pub struct MissingLiveness {
    pub message: String,
}

/// RPython `jitcode.py:131-143` `class SwitchDictDescr(AbstractDescr)`:
///
/// ```python
/// class SwitchDictDescr(AbstractDescr):
///     "Get a 'dict' attribute mapping integer values to bytecode positions."
///
///     def attach(self, as_dict):
///         self.dict = as_dict
///         self.const_keys_in_order = map(ConstInt, sorted(as_dict.keys()))
///
///     def __repr__(self):
///         dict = getattr(self, 'dict', '?')
///         return '<SwitchDictDescr %s>' % (dict,)
///
///     def _clone_if_mutable(self):
///         raise NotImplementedError
/// ```
///
/// Used by the assembler to encode `switch` ops as a side-table mapping
/// integer values to bytecode positions. Currently a placeholder — pyre
/// has no `switch` op users yet, but the type lives here so the
/// codewriter::jitcode module shape stays parity-aligned with RPython.
#[derive(Debug, Clone, Default)]
pub struct SwitchDictDescr {
    /// RPython `attach`: integer key → bytecode position map.
    pub dict: std::collections::HashMap<i64, usize>,
    /// RPython `attach`: sorted ConstInt keys for replay/serialization.
    pub const_keys_in_order: Vec<i64>,
    /// `True` once `attach` has run, even if the supplied `as_dict` was
    /// empty.  RPython distinguishes the two states via attribute
    /// presence: `getattr(self, 'dict', '?')` returns `'?'` only when
    /// `attach` never set `self.dict`, while an attached empty dict
    /// renders as `{}`.  Pyre's `dict` field is always present (default
    /// `HashMap::new()`), so we carry an explicit flag to keep the
    /// repr distinction intact.
    attached: bool,
}

impl SwitchDictDescr {
    /// RPython `jitcode.py:134-136` `def attach(self, as_dict)`.
    pub fn attach(&mut self, as_dict: std::collections::HashMap<i64, usize>) {
        let mut keys: Vec<i64> = as_dict.keys().copied().collect();
        keys.sort();
        self.const_keys_in_order = keys;
        self.dict = as_dict;
        self.attached = true;
    }
}

impl std::fmt::Display for SwitchDictDescr {
    /// RPython `jitcode.py:138-140`:
    ///
    /// ```python
    /// def __repr__(self):
    ///     dict = getattr(self, 'dict', '?')
    ///     return '<SwitchDictDescr %s>' % (dict,)
    /// ```
    ///
    /// `attach` populates `as_dict` in `_labels` insertion order
    /// (`assembler.py:258-263`), and `_labels` itself is the
    /// post-`switches.sort(key=lambda link: link.llexitcase)` order
    /// from `flatten.py:274`. Iterate `const_keys_in_order` so the
    /// rendered dict matches Python's repr in sorted-key order.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.attached {
            // RPython `getattr(self, 'dict', '?')` falls back to '?'
            // only when `attach` has not run.  An attached empty dict
            // renders as `{}` below, mirroring Python's `repr({})`.
            return write!(f, "<SwitchDictDescr ?>");
        }
        f.write_str("<SwitchDictDescr {")?;
        for (i, key) in self.const_keys_in_order.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            match self.dict.get(key) {
                Some(target) => write!(f, "{key}: {target}")?,
                None => write!(f, "{key}: ?")?,
            }
        }
        f.write_str("}>")
    }
}

/// RPython `history.py:AbstractDescr` — base class for all descriptor
/// objects stored in the assembler's `descrs` list. Read at runtime via
/// 'd'/'j' argcodes in the blackhole interpreter.
///
/// RPython uses a class hierarchy (`FieldDescr`, `ArrayDescr`, `CallDescr`,
/// `JitCode(AbstractDescr)`, `SwitchDictDescr`). pyre uses an enum to
/// represent the same heterogeneous list, shared between the codewriter
/// assembler and the metainterp blackhole.
/// RPython `descr.py:665` `RESULT_ERASED` component of the call-descr cache
/// key. The Rust port still collapses most low-level pointer shapes to
/// `Type::Ref`, but the field is kept explicit so the descriptor table has the
/// same structural slot as upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CallResultErasedKey {
    Void,
    Signed,
    Unsigned,
    SingleFloat,
    Float,
    SignedLongLong,
    GcRef,
    Address,
}

impl CallResultErasedKey {
    pub fn from_ir_type(result_type: majit_ir::value::Type) -> Self {
        Self::from_ir_layout(result_type, result_type == majit_ir::value::Type::Int, 8)
    }

    pub fn from_ir_layout(
        result_type: majit_ir::value::Type,
        result_signed: bool,
        _result_size: usize,
    ) -> Self {
        match result_type {
            majit_ir::value::Type::Void => Self::Void,
            majit_ir::value::Type::Int if result_signed => Self::Signed,
            majit_ir::value::Type::Int => Self::Unsigned,
            majit_ir::value::Type::Ref => Self::GcRef,
            majit_ir::value::Type::Float => Self::Float,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BhCallDescr {
    /// RPython `CallDescr.arg_classes`: one char per non-void FUNC argument.
    /// This is not the assembler `I/R/F` list-marker suffix.
    pub arg_classes: String,
    pub result_type: char,
    /// RPython `descr.py:664` `result_signed`.
    pub result_signed: bool,
    /// RPython `descr.py:662` `symbolic.get_size(RESULT_ERASED, ...)`.
    pub result_size: usize,
    /// RPython `descr.py:665` `RESULT_ERASED`.
    pub result_erased: CallResultErasedKey,
    /// RPython `CallDescr.extrainfo` (`descr.py:453`,
    /// `effectinfo.py:13-263`).
    pub extra_info: majit_ir::descr::EffectInfo,
}

impl BhCallDescr {
    pub fn from_call_descr(cd: &dyn majit_ir::descr::CallDescr) -> Self {
        // RPython `descr.py:456 CallDescr.result_type` is the char
        // 'i'/'r'/'f'/'L'/'S'/'v' itself.  `cd.result_type()` is pyre's
        // coarser IR type, so derive the backend layout from
        // `result_class()` first; SimpleCallDescr preserves specialised
        // 'L'/'S' classes there even though their IR type is Float/Int.
        let result_class = cd.result_class();
        let (_, _, result_erased) = result_type_char_layout_key(result_class);
        let result_signed = cd.is_result_signed();
        let result_size = cd.result_size();
        Self {
            arg_classes: cd.arg_classes(),
            result_type: result_class,
            result_signed,
            result_size,
            result_erased: if result_class == 'i' || result_class == 'r' || result_class == 'f' {
                CallResultErasedKey::from_ir_layout(cd.result_type(), result_signed, result_size)
            } else {
                // Preserve RPython's RESULT_ERASED key for 'S'/'L'/'v'.
                // Keep `result_signed`/`result_size` from the concrete
                // CallDescr above; the layout tuple only supplies the
                // char-specific erased key.
                result_erased
            },
            extra_info: cd.get_extra_info().clone(),
        }
    }

    pub fn from_arg_classes(
        arg_classes: String,
        result_type: char,
        extra_info: majit_ir::descr::EffectInfo,
    ) -> Self {
        let (result_signed, result_size, result_erased) = result_type_char_layout_key(result_type);
        Self {
            arg_classes,
            result_type,
            result_signed,
            result_size,
            result_erased,
            extra_info,
        }
    }

    pub fn from_signature(
        arg_classes: String,
        result_type: majit_ir::value::Type,
        extra_info: majit_ir::descr::EffectInfo,
    ) -> Self {
        let result_size = match result_type {
            majit_ir::value::Type::Int
            | majit_ir::value::Type::Ref
            | majit_ir::value::Type::Float => 8,
            majit_ir::value::Type::Void => 0,
        };
        Self {
            arg_classes,
            result_type: ir_type_to_result_char(result_type),
            result_signed: result_type == majit_ir::value::Type::Int,
            result_size,
            result_erased: CallResultErasedKey::from_ir_layout(
                result_type,
                result_type == majit_ir::value::Type::Int,
                result_size,
            ),
            extra_info,
        }
    }
}

impl Default for BhCallDescr {
    fn default() -> Self {
        Self::from_signature(
            String::new(),
            majit_ir::value::Type::Void,
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BhFieldSpec {
    pub index: u32,
    pub name: String,
    pub offset: usize,
    pub field_size: usize,
    pub field_type: majit_ir::value::Type,
    pub field_flag: majit_ir::descr::ArrayFlag,
    pub is_field_signed: bool,
    pub is_immutable: bool,
    pub is_quasi_immutable: bool,
    pub index_in_parent: usize,
}

impl BhFieldSpec {
    /// Mirror an `Arc<dyn FieldDescr>` into the serializable
    /// `BhFieldSpec` shape so producers outside the codewriter
    /// (e.g. blackhole-allocator dispatch in `pyre-jit`) can build
    /// `BhDescr::Size.all_fielddescrs` matching `descr.py:188
    /// init_size_descr` parity.
    pub fn from_field_descr(fd: &dyn majit_ir::descr::FieldDescr) -> Self {
        let field_flag = if fd.is_pointer_field() {
            majit_ir::descr::ArrayFlag::Unsigned
        } else if fd.is_float_field() {
            majit_ir::descr::ArrayFlag::Float
        } else if fd.field_type() == majit_ir::value::Type::Void {
            majit_ir::descr::ArrayFlag::Void
        } else if fd.is_field_signed() {
            majit_ir::descr::ArrayFlag::Signed
        } else {
            majit_ir::descr::ArrayFlag::Unsigned
        };
        Self {
            index: fd.index(),
            name: fd.field_name().to_string(),
            offset: fd.offset(),
            field_size: fd.field_size(),
            field_type: fd.field_type(),
            field_flag,
            is_field_signed: fd.is_field_signed(),
            is_immutable: fd.is_immutable(),
            is_quasi_immutable: fd.is_quasi_immutable(),
            index_in_parent: fd.index_in_parent(),
        }
    }
}

/// Mirror `SizeDescr.all_fielddescrs` (`descr.py:122-126`) onto a
/// fresh `Vec<BhFieldSpec>`.
pub fn bh_field_specs_from_size_descr(sd: &dyn majit_ir::descr::SizeDescr) -> Vec<BhFieldSpec> {
    sd.all_fielddescrs()
        .iter()
        .map(|fd| BhFieldSpec::from_field_descr(fd.as_ref()))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BhSizeSpec {
    pub size: usize,
    pub type_id: u32,
    pub vtable: usize,
    pub all_fielddescrs: Vec<BhFieldSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BhInteriorFieldSpec {
    pub index: u32,
    pub field: BhFieldSpec,
    pub owner: BhSizeSpec,
}

fn result_type_char_layout_key(result_type: char) -> (bool, usize, CallResultErasedKey) {
    match result_type {
        'i' => (true, 8, CallResultErasedKey::Signed),
        'S' => (false, 4, CallResultErasedKey::SingleFloat),
        'r' => (false, 8, CallResultErasedKey::GcRef),
        'f' => (false, 8, CallResultErasedKey::Float),
        'L' => (false, 8, CallResultErasedKey::SignedLongLong),
        'v' => (false, 0, CallResultErasedKey::Void),
        _ => (false, 0, CallResultErasedKey::Void),
    }
}

fn ir_type_to_result_char(result_type: majit_ir::value::Type) -> char {
    match result_type {
        majit_ir::value::Type::Int => 'i',
        majit_ir::value::Type::Ref => 'r',
        majit_ir::value::Type::Float => 'f',
        majit_ir::value::Type::Void => 'v',
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BhDescr {
    /// Field descriptor: for getfield/setfield.
    /// RPython: `FieldDescr(AbstractDescr)` — carries `offset`, `field_size`.
    /// `name` + `owner` identify the field for runtime offset resolution.
    /// `offset` is populated when known (0 = unresolved placeholder).
    Field {
        offset: usize,
        field_size: usize,
        field_type: majit_ir::value::Type,
        field_flag: majit_ir::descr::ArrayFlag,
        is_field_signed: bool,
        is_immutable: bool,
        is_quasi_immutable: bool,
        index_in_parent: usize,
        parent: Option<BhSizeSpec>,
        name: String,
        owner: String,
    },
    /// Array descriptor: for getarrayitem/setarrayitem/arraylen.
    /// RPython: `ArrayDescr` with `itemsize`, `basesize` attributes.
    /// `itemsize` is populated when known (8 = default placeholder).
    Array {
        base_size: usize,
        itemsize: usize,
        type_id: u32,
        item_type: majit_ir::value::Type,
        is_array_of_pointers: bool,
        is_array_of_structs: bool,
        /// descr.py ArrayDescr.is_item_signed() — FLAG_SIGNED vs FLAG_UNSIGNED.
        is_item_signed: bool,
        /// descr.py:372-375 `arraydescr.all_interiorfielddescrs` for
        /// arrays whose item type is an inline struct.
        interior_fields: Vec<BhInteriorFieldSpec>,
    },
    /// Plain `SizeDescr` (no vtable / NEW_WITH_VTABLE descr).
    ///
    /// `descr.py:120 get_size_descr` + `:188 init_size_descr` populate
    /// the `SizeDescr.all_fielddescrs` and `gc_fielddescrs` lists from
    /// `heaptracker.all_fielddescrs(STRUCT)` at descr-creation time so
    /// downstream consumers (`info.py:180 init_fields`, virtualized
    /// struct fan-out) read the full per-struct layout off the descr.
    /// `owner` carries the upstream `STRUCT._name` so a producer that
    /// only has the size + type_id can re-resolve the layout via
    /// `bh_all_field_specs_for_struct`.
    Size {
        size: usize,
        type_id: u32,
        vtable: usize,
        /// RPython `STRUCT._name` identity (empty when the size descr
        /// is built transiently for `bh_new` / `bh_new_with_vtable`
        /// dispatch and the struct identity is already encoded in the
        /// caller-supplied `DescrRef`).
        owner: String,
        /// `heaptracker.all_fielddescrs(STRUCT)` snapshot; empty when
        /// the size descr is purely transient (no struct context).
        all_fielddescrs: Vec<BhFieldSpec>,
    },
    /// Call descriptor: for residual_call. Carries calling convention.
    /// RPython: `CallDescr`.
    Call { calldescr: BhCallDescr },
    /// JitCode descriptor: for inline_call_*.
    /// RPython: `JitCode(AbstractDescr)` — carries `fnaddr` + `calldescr`.
    /// `jitcode_index` indexes into `all_jitcodes[]` (set by CodeWriter).
    /// `fnaddr` is resolved at runtime from the callee's function address.
    JitCode {
        /// Index into all_jitcodes[]. Used by the blackhole to find the
        /// callee's bytecode for frame-chain push.
        jitcode_index: usize,
        /// Function address for cpu.bh_call_*. Resolved at runtime.
        fnaddr: i64,
        /// CallDescr for cpu.bh_call_* dispatch.
        calldescr: BhCallDescr,
    },
    /// SwitchDictDescr: maps int values to bytecode positions.
    Switch {
        dict: std::collections::HashMap<i64, usize>,
    },
    /// Virtualizable field descriptor: index into VirtualizableInfo.static_fields.
    /// NOT a byte offset — the blackhole resolves it via `vinfo.static_fields[index].offset`.
    VableField { index: usize },
    /// Virtualizable array descriptor: index into VirtualizableInfo.array_fields.
    VableArray { index: usize },
    /// Vtable-method descriptor for `funcptr_from_vtable`.  Carries the
    /// trait + method identity so the runtime (when ported) can resolve
    /// the receiver fat pointer's vtable slot to a function address.
    /// RPython's `op.args[0]` is already a `Ptr(FuncType)` after rtype
    /// (`rpython/jit/codewriter/jtransform.py:546`); Rust `&dyn Trait` is
    /// a fat pointer so the slot lookup must happen at runtime.  No
    /// blackhole/backend consumer ships with this commit — the
    /// descriptor exists so the IR survives serialization.
    VtableMethod {
        trait_root: String,
        method_name: String,
    },
}

impl BhDescr {
    /// Extract byte offset for field/array operations (FieldDescr/ArrayDescr).
    /// Panics on VableField/VableArray — those must use `as_vable_field_index`.
    pub fn as_offset(&self) -> usize {
        match self {
            BhDescr::Field { offset, .. } => *offset,
            BhDescr::Array { itemsize, .. } => *itemsize,
            _ => panic!("BhDescr::as_offset called on {:?}", self),
        }
    }

    pub fn as_size(&self) -> usize {
        match self {
            BhDescr::Size { size, .. } => *size,
            BhDescr::Field { offset, .. } => *offset,
            _ => panic!("BhDescr::as_size called on {:?}", self),
        }
    }

    pub fn get_vtable(&self) -> usize {
        match self {
            BhDescr::Size { vtable, .. } => *vtable,
            _ => 0,
        }
    }

    pub fn get_type_id(&self) -> u32 {
        match self {
            BhDescr::Size { type_id, .. } => *type_id,
            _ => 0,
        }
    }

    pub fn as_itemsize(&self) -> usize {
        match self {
            BhDescr::Array { itemsize, .. } => *itemsize,
            _ => panic!("BhDescr::as_itemsize called on {:?}", self),
        }
    }

    pub fn is_array_of_pointers(&self) -> bool {
        match self {
            BhDescr::Array {
                is_array_of_pointers,
                ..
            } => *is_array_of_pointers,
            _ => false,
        }
    }

    /// descr.py ArrayDescr.is_item_signed() — signed integer items.
    pub fn is_item_signed(&self) -> bool {
        match self {
            BhDescr::Array { is_item_signed, .. } => *is_item_signed,
            _ => false,
        }
    }

    /// Reconstruct BhDescr::Array from serialized ArrayDescrInfo.
    /// Used at resume/materialization boundaries where only the summary is available.
    pub fn from_array_descr_info(info: &majit_ir::ArrayDescrInfo) -> Self {
        BhDescr::Array {
            base_size: info.base_size,
            itemsize: info.item_size,
            type_id: 0,
            item_type: match info.item_type {
                0 => majit_ir::value::Type::Ref,
                2 => majit_ir::value::Type::Float,
                _ => majit_ir::value::Type::Int,
            },
            is_array_of_pointers: info.item_type == 0,
            is_array_of_structs: false,
            is_item_signed: info.is_signed,
            interior_fields: Vec::new(),
        }
    }

    /// ArrayDescr: true when the array items are structs (GC objects).
    /// RPython: `arraydescr.is_array_of_structs()` in blackhole.py:1165.
    pub fn is_array_of_structs(&self) -> bool {
        match self {
            BhDescr::Array {
                is_array_of_structs,
                ..
            } => *is_array_of_structs,
            _ => false,
        }
    }

    /// Get field name (for runtime offset resolution).
    pub fn field_name(&self) -> &str {
        match self {
            BhDescr::Field { name, .. } => name,
            _ => panic!("BhDescr::field_name called on {:?}", self),
        }
    }

    /// Get field owner type name.
    pub fn field_owner(&self) -> &str {
        match self {
            BhDescr::Field { owner, .. } => owner,
            _ => panic!("BhDescr::field_owner called on {:?}", self),
        }
    }

    /// Extract virtualizable field index.
    pub fn as_vable_field_index(&self) -> usize {
        match self {
            BhDescr::VableField { index } => *index,
            _ => panic!("BhDescr::as_vable_field_index called on {:?}", self),
        }
    }

    /// Extract virtualizable array index.
    pub fn as_vable_array_index(&self) -> usize {
        match self {
            BhDescr::VableArray { index } => *index,
            _ => panic!("BhDescr::as_vable_array_index called on {:?}", self),
        }
    }

    /// Extract JitCode index for inline_call.
    pub fn as_jitcode_index(&self) -> usize {
        match self {
            BhDescr::JitCode { jitcode_index, .. } => *jitcode_index,
            _ => panic!("BhDescr::as_jitcode_index called on {:?}", self),
        }
    }

    /// Extract function address for inline_call cpu.bh_call_* fallback.
    pub fn as_jitcode_fnaddr(&self) -> i64 {
        match self {
            BhDescr::JitCode { fnaddr, .. } => *fnaddr,
            _ => 0,
        }
    }

    pub fn as_calldescr(&self) -> &BhCallDescr {
        match self {
            BhDescr::Call { calldescr } => calldescr,
            BhDescr::JitCode { calldescr, .. } => calldescr,
            _ => panic!("BhDescr::as_calldescr called on {:?}", self),
        }
    }

    /// Lookup switch value → position.
    pub fn switch_lookup(&self, value: i64) -> Option<usize> {
        match self {
            BhDescr::Switch { dict } => dict.get(&value).copied(),
            _ => None,
        }
    }
}

/// RPython `format.py:12-80` `format_assembler(ssarepr)`.
///
/// Minimal port: formats each FlatOp in the SSARepr into human-readable
/// text. RPython uses this for debug output and testing.
///
/// ```python
/// def format_assembler(ssarepr):
///     """For testing: format a SSARepr as a multiline string."""
///     ...
///     return buf.getvalue()
/// ```
pub fn format_assembler(ssarepr: &crate::flatten::SSARepr) -> String {
    use crate::flatten::{FlatOp, RegKind, constvalue_kind};
    use std::fmt::Write;

    let kind_char = |v: crate::model::ValueId| -> char {
        match ssarepr.value_kinds.get(&v).copied().unwrap_or(RegKind::Ref) {
            RegKind::Int => 'i',
            RegKind::Ref => 'r',
            RegKind::Float => 'f',
        }
    };
    let kind_name = |v: crate::model::ValueId| -> &'static str {
        match ssarepr.value_kinds.get(&v).copied().unwrap_or(RegKind::Ref) {
            RegKind::Int => "int",
            RegKind::Ref => "ref",
            RegKind::Float => "float",
        }
    };
    let linkarg_kind_char = |arg: &crate::model::LinkArg| -> char {
        match arg {
            crate::model::LinkArg::Value(v) => kind_char(*v),
            crate::model::LinkArg::Const(cv) => constvalue_kind(cv),
        }
    };
    let linkarg_kind_name = |arg: &crate::model::LinkArg| -> &'static str {
        match linkarg_kind_char(arg) {
            'i' => "int",
            'f' => "float",
            _ => "ref",
        }
    };
    let linkarg_repr = |arg: &crate::model::LinkArg| -> String {
        match arg {
            crate::model::LinkArg::Value(v) => format!("%{}{}", kind_char(*v), v.0),
            crate::model::LinkArg::Const(cv) => format!("${cv}"),
        }
    };

    let mut out = String::new();
    writeln!(out, "{}", ssarepr.name).ok();
    for op in &ssarepr.insns {
        match op {
            FlatOp::Label(label) => {
                writeln!(out, "L{}:", label.0).ok();
            }
            FlatOp::Live { live_values } => {
                let regs: Vec<String> = live_values
                    .iter()
                    .map(|v| format!("%{}{}", kind_char(*v), v.0))
                    .collect();
                writeln!(out, "  -live- {}", regs.join(", ")).ok();
            }
            FlatOp::EndOfBlock => {
                writeln!(out, "  ---").ok();
            }
            FlatOp::Unreachable => {
                writeln!(out, "  unreachable").ok();
            }
            FlatOp::Op(space_op) => {
                let result = space_op
                    .result
                    .map(|v| format!(" -> %{}{}", kind_char(v), v.0))
                    .unwrap_or_default();
                writeln!(out, "  {:?}{result}", space_op.kind).ok();
            }
            FlatOp::Jump(label) => {
                writeln!(out, "  goto L{}", label.0).ok();
            }
            FlatOp::CatchException { target } => {
                writeln!(out, "  catch_exception L{}", target.0).ok();
            }
            FlatOp::GotoIfExceptionMismatch { llexitcase, target } => {
                writeln!(
                    out,
                    "  goto_if_exception_mismatch ${:?}, L{}",
                    llexitcase, target.0
                )
                .ok();
            }
            FlatOp::IntBinOpJumpIfOvf {
                op,
                target,
                lhs,
                rhs,
                dst,
            } => {
                let opname = match op {
                    crate::flatten::IntOvfOp::Add => "int_add_jump_if_ovf",
                    crate::flatten::IntOvfOp::Sub => "int_sub_jump_if_ovf",
                    crate::flatten::IntOvfOp::Mul => "int_mul_jump_if_ovf",
                };
                writeln!(
                    out,
                    "  {opname} L{}, %i{}, %i{} -> %i{}",
                    target.0, lhs.0, rhs.0, dst.0
                )
                .ok();
            }
            FlatOp::GotoIfNot { cond, target } => {
                writeln!(
                    out,
                    "  goto_if_not %{}{}, L{}",
                    kind_char(*cond),
                    cond.0,
                    target.0
                )
                .ok();
            }
            FlatOp::Switch { value, targets } => {
                let cases: Vec<String> = targets
                    .iter()
                    .map(|(key, label)| format!("{key}:L{}", label.0))
                    .collect();
                writeln!(
                    out,
                    "  switch %{}{}, <SwitchDictDescr {}>",
                    kind_char(*value),
                    value.0,
                    cases.join(", ")
                )
                .ok();
            }
            // `flatten.py:326-335` kind-prefixed opnames.
            FlatOp::Move { dst, src } => {
                writeln!(
                    out,
                    "  {}_copy {} -> %{}{}",
                    linkarg_kind_name(src),
                    linkarg_repr(src),
                    kind_char(*dst),
                    dst.0
                )
                .ok();
            }
            FlatOp::Push(src) => {
                writeln!(
                    out,
                    "  {}_push %{}{}",
                    kind_name(*src),
                    kind_char(*src),
                    src.0
                )
                .ok();
            }
            FlatOp::Pop(dst) => {
                writeln!(
                    out,
                    "  {}_pop -> %{}{}",
                    kind_name(*dst),
                    kind_char(*dst),
                    dst.0
                )
                .ok();
            }
            FlatOp::LastException { dst } => {
                writeln!(out, "  last_exception -> %i{}", dst.0).ok();
            }
            FlatOp::LastExcValue { dst } => {
                writeln!(out, "  last_exc_value -> %r{}", dst.0).ok();
            }
            FlatOp::Reraise => {
                writeln!(out, "  reraise").ok();
            }
            FlatOp::IntReturn(v) => {
                writeln!(out, "  int_return {}", linkarg_repr(v)).ok();
            }
            FlatOp::RefReturn(v) => {
                writeln!(out, "  ref_return {}", linkarg_repr(v)).ok();
            }
            FlatOp::FloatReturn(v) => {
                writeln!(out, "  float_return {}", linkarg_repr(v)).ok();
            }
            FlatOp::VoidReturn => {
                writeln!(out, "  void_return").ok();
            }
            FlatOp::Raise(v) => {
                writeln!(out, "  raise {}", linkarg_repr(v)).ok();
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switch_dict_descr_unattached_renders_question_mark() {
        // RPython `jitcode.py:138 def __repr__(self): dict =
        // getattr(self, 'dict', '?')` returns `'?'` only when
        // `self.dict` attribute is missing entirely (i.e. `attach`
        // never ran).  Pyre has to track the attach event explicitly
        // because the `dict` field is always present (default empty
        // HashMap); regression-guard the unattached branch so a
        // future refactor cannot silently collapse it back to "empty
        // implies unattached".
        let descr = SwitchDictDescr::default();
        assert_eq!(descr.to_string(), "<SwitchDictDescr ?>");
    }

    #[test]
    fn switch_dict_descr_attached_empty_renders_empty_braces() {
        // RPython `repr({}) == '{}'`, and an attached SwitchDictDescr
        // whose `as_dict` was empty must render the same way to keep
        // debug-output parity with upstream.  Without the
        // `attached: bool` flag this state collapsed into the
        // unattached `'?'` branch.
        let mut descr = SwitchDictDescr::default();
        descr.attach(std::collections::HashMap::new());
        assert_eq!(descr.to_string(), "<SwitchDictDescr {}>");
    }

    #[test]
    fn switch_dict_descr_attached_renders_sorted_dict() {
        let mut descr = SwitchDictDescr::default();
        let mut dict = std::collections::HashMap::new();
        dict.insert(7, 30);
        dict.insert(1, 10);
        dict.insert(3, 20);
        descr.attach(dict);
        assert_eq!(descr.to_string(), "<SwitchDictDescr {1: 10, 3: 20, 7: 30}>");
    }
}
