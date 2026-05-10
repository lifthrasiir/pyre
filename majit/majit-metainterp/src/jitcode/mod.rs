mod assembler;

pub use assembler::{JitCodeBuilder, live_slots_for_state_field_jit};
pub use majit_translate::jitcode::{
    BhCallDescr as CanonicalBhCallDescr, BhDescr as CanonicalBhDescr, BhInteriorFieldSpec,
    JitCode as CanonicalJitCode,
};

// `BC_*` constants and `MAX_HOST_CALL_ARITY` live in
// `majit_translate::insns` (Task #86, slice #86c). The module is
// re-exported here as the canonical access path; in-crate and external
// consumers reach `BC_*` / `MAX_HOST_CALL_ARITY` via
// `jitcode::insns::BC_*`.
pub use majit_translate::insns;

/// Alias for `BC_JUMP`; used in dispatch JitCode loop-close tests
/// (Slice 1.7) and `jitcode_lower::lower_dispatch_body` jump emission.
pub const BC_GOTO: u8 = insns::BC_JUMP;

// `insn_byte` and `wellknown_bh_insns` were moved to
// `majit_translate::insns` in slice #86d (Task #86). Re-exports keep
// internal callers (`jitcode::assembler::JitCodeBuilder`) and external
// callers (`pyre/pyre-jit/src/jit/assembler.rs`) resolving unchanged
// — the import-path sweep is slice #86e.
pub(crate) use majit_translate::insns::insn_byte;

pub use majit_translate::insns::{pyre_extension_insns, wellknown_bh_insns};

/// GC liveness metadata at a specific bytecode PC.
///
/// RPython liveness.py: `[len_i][len_r][len_f][bitset_i][bitset_r][bitset_f]`.
/// Tracks which registers of each type (int/ref/float) are live at a given PC.
///
/// TODO: pyre currently keeps this per-entry form alongside the packed
/// Temporary pyre-side liveness shape used before the codewriter emits
/// RPython `-live-` opcodes directly. Canonical JitCode does not store this.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LivenessInfo {
    pub pc: u16,
    /// Live integer register indices at this PC.
    pub live_i_regs: Vec<u16>,
    /// Live reference register indices at this PC.
    pub live_r_regs: Vec<u16>,
    /// Live float register indices at this PC.
    pub live_f_regs: Vec<u16>,
}

impl LivenessInfo {
    /// Total number of live registers across all typed banks.
    pub fn total_live(&self) -> usize {
        self.live_i_regs.len() + self.live_r_regs.len() + self.live_f_regs.len()
    }
}

/// Re-export of the canonical `enumerate_vars` function so existing
/// metainterp callers can keep using `crate::jitcode::enumerate_vars`.
///
/// RPython places this function in `rpython/jit/codewriter/jitcode.py`,
/// not in metainterp. majit follows the same module placement: the
/// definition lives in `majit_translate::jitcode::enumerate_vars`.
pub use majit_translate::jitcode::enumerate_vars;

// ──────────────────────────────────────────────────────────────────
// Runtime descr pool types — RPython
// `BlackholeInterpBuilder.descrs` / `BlackholeInterpreter.descrs`
// (`blackhole.py:103`, `blackhole.py:288`).
//
// RPython keeps the descr pool on the blackhole interpreter, NOT on
// the JitCode object.  In majit the canonical
// `majit_translate::jitcode::JitCode` mirrors that — it is a
// source-only RPython parity type with no descrs field.  The runtime
// adapter state (descrs pool + call/assembler targets) lives here
// alongside the wrapper `JitCode` defined below, which carries
// `pub exec: JitCodeExecState` as a sibling of the canonical core.
//
// These types are runtime-only — they reference raw `*const ()`
// trampoline addresses and live `Arc<JitCode>` callee handles, neither
// of which has a representation in the codewriter source layer.
// ──────────────────────────────────────────────────────────────────

/// Trace-side function target descriptor for `BC_CALL_*` /
/// `BC_RESIDUAL_CALL_*`.  RPython `blackhole.py:1225-1256` reads the
/// callee function address from an int register (`i` argcode) and the
/// calling convention from a descr (`d` argcode); pyre bundles the
/// trace-side and concrete (non-JIT) function pointers into a single
/// descriptor slot because the runtime emitter wires both pointers
/// through one indirection.
///
/// `effect_info_slot` is the per-target analyzer-result classification
/// (`call.py:282-303 getcalldescr`'s `extraeffect` selection without
/// the graph-based analyzer chain — see
/// [`crate::call_descr::EffectInfoSlot`]).  Callers that have a
/// resolved `JitCallTarget` thread the slot through
/// `make_call_descr_from_target_slot` so the recorded descr carries
/// the right `EffectInfo` instead of the `default_effect_info()`
/// fallback.  The default ([`EffectInfoSlot::CanRaise`]) preserves the
/// pre-G-2 behaviour for every existing construction site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitCallTarget {
    pub trace_ptr: *const (),
    pub concrete_ptr: *const (),
    pub effect_info_slot: crate::call_descr::EffectInfoSlot,
}

impl JitCallTarget {
    pub fn new(trace_ptr: *const (), concrete_ptr: *const ()) -> Self {
        Self {
            trace_ptr,
            concrete_ptr,
            effect_info_slot: crate::call_descr::EffectInfoSlot::CanRaise,
        }
    }

    /// Construct a target with an explicit
    /// [`crate::call_descr::EffectInfoSlot`] classification.  Used by
    /// the macro-time helper registration paths that statically know
    /// the callee's `_canraise` / `_elidable_function_` /
    /// `_jit_loop_invariant_` flags.
    pub fn with_effect_info_slot(
        trace_ptr: *const (),
        concrete_ptr: *const (),
        effect_info_slot: crate::call_descr::EffectInfoSlot,
    ) -> Self {
        Self {
            trace_ptr,
            concrete_ptr,
            effect_info_slot,
        }
    }
}

/// Compiled-loop target for `BC_CALL_ASSEMBLER_*`.  The `token_number`
/// names a `CompiledLoopToken` (RPython `compile.py
/// CompiledLoopToken.number`) that the tracer hands to
/// `ctx.call_assembler_*_typed`; `concrete_ptr` is the pointer the
/// blackhole interpreter calls when the trace bails out before the
/// loop is compiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitCallAssemblerTarget {
    pub token_number: u64,
    pub concrete_ptr: *const (),
}

impl JitCallAssemblerTarget {
    pub fn new(token_number: u64, concrete_ptr: *const ()) -> Self {
        Self {
            token_number,
            concrete_ptr,
        }
    }
}

/// Per-arg kind tag for typed call argument streams.  Mirrors the
/// `i`/`r`/`f` register-bank chars RPython carries in
/// `BlackholeInterpBuilder.descrs` argcode bytes (`blackhole.py:154`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitArgKind {
    Int = 0,
    Ref = 1,
    Float = 2,
}

impl JitArgKind {
    pub fn encode(self) -> u8 {
        self as u8
    }

    pub fn decode(byte: u8) -> Self {
        match byte {
            0 => Self::Int,
            1 => Self::Ref,
            2 => Self::Float,
            other => panic!("unknown jitcode arg kind {other}"),
        }
    }

    /// Map a [`majit_ir::Type`] to its `JitArgKind`.  RPython encodes
    /// the same mapping inline in `_build_allboxes` per
    /// `pyjitpl.py:1969-1989` (`history.INT`/`history.REF`/`history.FLOAT`
    /// chars + `'S'` single-float / `'L'` long-long aliases).  Pyre's
    /// `Type::Void` has no JitArgKind because void calls carry no
    /// argbox.
    pub fn from_type(ty: majit_ir::Type) -> Option<Self> {
        match ty {
            majit_ir::Type::Int => Some(Self::Int),
            majit_ir::Type::Ref => Some(Self::Ref),
            majit_ir::Type::Float => Some(Self::Float),
            majit_ir::Type::Void => None,
        }
    }
}

/// Typed call argument: a register index plus its kind tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitCallArg {
    pub kind: JitArgKind,
    pub reg: u16,
}

impl JitCallArg {
    pub fn int(reg: u16) -> Self {
        Self {
            kind: JitArgKind::Int,
            reg,
        }
    }

    pub fn reference(reg: u16) -> Self {
        Self {
            kind: JitArgKind::Ref,
            reg,
        }
    }

    pub fn float(reg: u16) -> Self {
        Self {
            kind: JitArgKind::Float,
            reg,
        }
    }
}

/// Runtime descriptor entry — heterogeneous pool element indexed by
/// `j` / `d` argcodes at dispatch time.  Equivalent of RPython
/// `self.descrs[idx]` where each entry is an instance of one of the
/// `AbstractDescr` subclasses (`FieldDescr`, `ArrayDescr`, `JitCode`,
/// ...).  RPython uses `isinstance(value, JitCode)` to discriminate at
/// runtime; pyre encodes the same discrimination in the enum tag.
#[derive(Clone, Debug)]
pub enum RuntimeBhDescr {
    /// Ordinary blackhole descriptor for a `d` argcode (`FieldDescr`,
    /// `ArrayDescr`, virtualizable descriptors, ...).  RPython keeps all
    /// of these in `BlackholeInterpBuilder.descrs`; pyre's runtime
    /// `JitCodeBuilder` uses the per-JitCode pool described below.
    Descr(CanonicalBhDescr),
    /// Target JitCode for a `j` argcode (`BC_INLINE_CALL`).  RPython:
    /// `blackhole.py:150-157` — `argtype == 'j' → descrs[idx]` asserted
    /// `isinstance(value, JitCode)`.
    JitCode(std::sync::Arc<JitCode>),
    /// Target function for `BC_CALL_*` / `BC_RESIDUAL_CALL_*`.
    /// RPython `blackhole.py:1225-1256` reads the function address
    /// from an int register (`i` argcode) and the calling convention
    /// from a descr (`d` argcode); pyre keeps the two together in
    /// `JitCallTarget` because the runtime emitter wires trace-side
    /// and blackhole-side function pointers in a single indirection
    /// slot.  Once pyre emits the function address via an int register
    /// this variant can split into the RPython-shaped pair.
    Call(JitCallTarget),
    /// Compiled-assembler target for `BC_CALL_ASSEMBLER_*`.  The
    /// `token_number` identifies a `CompiledLoopToken` (RPython
    /// `compile.py CompiledLoopToken.number`) that the tracer hands
    /// to `ctx.call_assembler_*_typed` so the metainterp can chain
    /// this trace into an already-compiled one.
    AssemblerToken(JitCallAssemblerTarget),
}

impl RuntimeBhDescr {
    /// Extract an ordinary blackhole descriptor for `d` argcodes.
    pub fn as_bh_descr(&self) -> Option<&CanonicalBhDescr> {
        match self {
            Self::Descr(descr) => Some(descr),
            _ => None,
        }
    }

    /// RPython parity: `isinstance(value, JitCode)` assertion at
    /// `blackhole.py:156`.  Returns the callee JitCode for `BC_INLINE_CALL`.
    pub fn as_jitcode(&self) -> Option<&std::sync::Arc<JitCode>> {
        match self {
            Self::JitCode(arc) => Some(arc),
            _ => None,
        }
    }

    /// Extract the `Call` target for `BC_CALL_*` / `BC_RESIDUAL_CALL_*`.
    pub fn as_call(&self) -> Option<&JitCallTarget> {
        match self {
            Self::Call(target) => Some(target),
            _ => None,
        }
    }

    /// Extract the assembler-call target for `BC_CALL_ASSEMBLER_*`.
    pub fn as_assembler_token(&self) -> Option<&JitCallAssemblerTarget> {
        match self {
            Self::AssemblerToken(target) => Some(target),
            _ => None,
        }
    }
}

/// Per-`JitCode` descrs.  Pyre's analog of
/// `BlackholeInterpBuilder.descrs` (`blackhole.py:103`) /
/// `BlackholeInterpreter.descrs` (`blackhole.py:288`).  RPython has a
/// single shared global pool because translation-time JitCodes are
/// produced eagerly; pyre's runtime jitcodes are emitted on demand
/// per-Python-frame and lack a global allocation index, so the pool
/// is per-`JitCode` here as a sibling of the canonical `core`.
#[derive(Clone, Debug, Default)]
pub struct JitCodeExecState {
    /// Descriptor pool — indexed by the 2-byte `j`/`d` argcode operand.
    pub descrs: Vec<RuntimeBhDescr>,
    /// Sidetable mapping the canonical-call `d` argcode descriptor slot
    /// back to pyre's full `JitCallTarget` (`{trace_ptr, concrete_ptr}`).
    /// RPython stores the callable address in the `i` operand and the
    /// signature/effect policy in the `d` operand. Pyre's runtime emitter
    /// still has a trace/concrete pointer split, so this is the minimal
    /// adaptation needed for trace recording while preserving the
    /// RPython-shaped `residual_call_*_v` payload. Keying by descriptor
    /// slot keeps the bridge per callsite; keying by int-const pool slot
    /// would collapse distinct trace targets that share a concrete
    /// pointer.
    pub call_descr_to_call_target: std::collections::HashMap<u16, JitCallTarget>,
    /// Bytecode offset of the `BC_JIT_MERGE_POINT(_C)` opcode byte for
    /// the dispatch JitCode emitted by `lower_dispatch_body`.  `None`
    /// for non-dispatch JitCodes (helpers, sub-arms) — exactly one
    /// `jit_merge_point` op is allowed per dispatch JitCode and the
    /// builder asserts that contract on the second call.
    ///
    /// Captured by `JitCodeBuilder::jit_merge_point` at the
    /// `self.code.len()` immediately before the opcode byte is pushed,
    /// so consumers reading `jit_merge_point_offset` land on the
    /// opcode byte itself (decoded the same way as `frame.next_u8()`
    /// would deliver it).  `register_dispatch_jitcode` reads this
    /// field to validate the green/red list counts against the
    /// declared `JitDriverDescriptor` schema without re-scanning the
    /// bytecode — RPython `blackhole.py:107-156` argcode-based decode
    /// parity, no payload-byte collision risk.
    pub jit_merge_point_offset: Option<usize>,
}

// ──────────────────────────────────────────────────────────────────
// Wrapper `JitCode` — runtime jitcode = canonical core + descr pool.
//
// RPython parity:
//   * `core` is the source-only `rpython/jit/codewriter/jitcode.py`
//     `JitCode` analog (`majit_translate::jitcode::JitCode`).  It
//     holds `name`, `fnaddr`, `jitdriver_sd`, `index`, body
//     (`code`, `constants_*`, `c_num_regs_*`, ...) — exactly the
//     fields RPython's `JitCode` carries.
//   * `exec` mirrors the descr pool RPython keeps on the
//     `BlackholeInterpBuilder` (`blackhole.py:103`).  In RPython the
//     pool is shared globally; pyre keeps it per-jitcode for the lazy
//     emit reasons described above on `JitCodeExecState`.
//
// Existing `jitcode.code`, `jitcode.set_body(...)`, `jitcode.body()`,
// `jitcode.fnaddr` etc. continue to work via `Deref<Target=core>` —
// the wrapper is transparent to read-side callers.  Only writers
// that require `&mut core` need `DerefMut`.
//
// Serde: the wrapper itself is intentionally NOT
// `Serialize`/`Deserialize`.  The build-time bincode embed in
// `pyre-jit-trace::jitcode_runtime` serializes
// `Vec<Arc<majit_translate::jitcode::JitCode>>` (canonical core)
// because build-time jitcodes never carry descrs.  Wrappers are
// constructed at the runtime ingress (where the canonical Arc enters
// dispatch) via `JitCode::from_canonical`.  Per-CodeObject runtime
// jitcodes are produced directly as wrappers by
// `JitCodeBuilder::finish()`.
// ──────────────────────────────────────────────────────────────────

/// Runtime JitCode = canonical RPython parity core + descr pool.
#[derive(Debug)]
pub struct JitCode {
    /// Canonical source-only `JitCode` (RPython
    /// `rpython/jit/codewriter/jitcode.py:9 class JitCode`).
    core: majit_translate::jitcode::JitCode,
    /// Per-jitcode descr pool — pyre's analog of
    /// `BlackholeInterpBuilder.descrs` (RPython
    /// `blackhole.py:103`).  Empty for build-time canonical jitcodes
    /// (descrs resolved through the global `ALL_DESCRS` table); the
    /// `JitCodeBuilder` populates this during runtime per-CodeObject
    /// emission.
    pub exec: JitCodeExecState,
}

// SAFETY: `JitCallTarget` / `JitCallAssemblerTarget` carry `*const ()`
// JIT-emitted code addresses; `RuntimeBhDescr::JitCode` carries
// `Arc<JitCode>` which is itself Send+Sync.  The pool is mutated only
// during `JitCodeBuilder::finish()` (single-threaded) and read
// thereafter; matches RPython's translation-time blackhole-builder
// publication flow.
unsafe impl Send for JitCode {}
unsafe impl Sync for JitCode {}

impl JitCode {
    /// Construct a fresh runtime jitcode wrapping a canonical
    /// `majit_translate::jitcode::JitCode::new(name)` core with an
    /// empty descr pool.  RPython `jitcode.py:14-20`
    /// `JitCode.__init__(name, fnaddr=None, calldescr=None, called_from=None)`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            core: majit_translate::jitcode::JitCode::new(name),
            exec: JitCodeExecState::default(),
        }
    }

    /// `jitcode.py:14 JitCode.name` accessor — proxies to the canonical
    /// source-only core for diagnostic / parity-validator messages.
    pub fn name(&self) -> &str {
        &self.core.name
    }

    /// Wrap a pre-built canonical `JitCode` (e.g. one produced by
    /// `CodeWriter::make_jitcodes()` at build time) with an empty
    /// descr pool.  Build-time jitcodes resolve their `'d'`/`'j'`
    /// argcodes through the global `ALL_DESCRS` table and never
    /// populate `exec.descrs`.
    pub fn from_canonical(core: majit_translate::jitcode::JitCode) -> Self {
        Self {
            core,
            exec: JitCodeExecState::default(),
        }
    }

    /// Borrow the canonical core (e.g. for serialization that
    /// re-serializes only the canonical fields).
    pub fn core(&self) -> &majit_translate::jitcode::JitCode {
        &self.core
    }

    /// Mutable canonical core access for in-place mutation (used by
    /// post-`set_body` `body_mut()` etc.).  RPython mutates `JitCode`
    /// fields directly post-`setup()`; pyre routes the mutation
    /// through this accessor so the wrapper stays transparent.
    pub fn core_mut(&mut self) -> &mut majit_translate::jitcode::JitCode {
        &mut self.core
    }
}

impl Default for JitCode {
    fn default() -> Self {
        Self::from_canonical(majit_translate::jitcode::JitCode::default())
    }
}

impl Clone for JitCode {
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            exec: self.exec.clone(),
        }
    }
}

impl std::ops::Deref for JitCode {
    type Target = majit_translate::jitcode::JitCode;
    fn deref(&self) -> &majit_translate::jitcode::JitCode {
        &self.core
    }
}

impl std::ops::DerefMut for JitCode {
    fn deref_mut(&mut self) -> &mut majit_translate::jitcode::JitCode {
        &mut self.core
    }
}

impl std::fmt::Display for JitCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.core, f)
    }
}

/// Helper preserved from the runtime jitcode era so callers that
/// expected the runtime `JitCode` body fields at the top level keep
/// working through `Deref<Target=JitCodeBody>`.
///
/// `trailing_return_info` is wired here because it depends on the
/// runtime BC_* opcode bytes (`BC_VOID_RETURN`, `BC_INT_RETURN`,
/// `BC_REF_RETURN`, `BC_FLOAT_RETURN`) which are runtime-defined; the
/// canonical jitcode crate does not import them. Provided as a free
/// function so the call sites can keep `jitcode.trailing_return_info()`
/// syntax via the existing trait impl below.
pub trait JitCodeRuntimeExt {
    /// Inspect the trailing typed return opcode of a helper jitcode.
    fn trailing_return_info(&self) -> Option<(JitArgKind, u16)>;
}

impl JitCode {
    /// Resolve `BC_CALL_*` / `BC_RESIDUAL_CALL_*` function-target
    /// descr.  Mirrors RPython `blackhole.py:1225-1256` where the
    /// calling-convention descr travels through `descrs[idx]`; pyre
    /// additionally bundles the trace and concrete fn pointers in
    /// the `Call` variant because the call encoding pre-dates the
    /// RPython-orthodox register-fed function address.
    pub fn call_target(&self, index: usize) -> &JitCallTarget {
        match self.exec.descrs.get(index) {
            Some(RuntimeBhDescr::Call(target)) => target,
            other => {
                panic!("BC_CALL_*/RESIDUAL_CALL_*: descrs[{index}] is not a Call entry: {other:?}",)
            }
        }
    }

    /// Transitional CALL_ASSEMBLER target lookup for the hardcoded
    /// JitCodeBuilder bytecode.  RPython stores the callee loop token
    /// in descriptor data threaded through the shared `descrs` pool;
    /// pyre mirrors the shape via the `AssemblerToken` variant.
    pub fn call_assembler_target(&self, index: usize) -> (u64, *const ()) {
        let target = self
            .exec
            .descrs
            .get(index)
            .and_then(RuntimeBhDescr::as_assembler_token)
            .unwrap_or_else(|| {
                panic!("BC_CALL_ASSEMBLER_*: descrs[{index}] is not an AssemblerToken entry",)
            });
        (target.token_number, target.concrete_ptr)
    }
}

impl JitCodeRuntimeExt for JitCode {
    fn trailing_return_info(&self) -> Option<(JitArgKind, u16)> {
        let body = self.try_body()?;
        let code = &body.code;
        if code.last().copied() == Some(insns::BC_VOID_RETURN) || code.len() < 3 {
            return None;
        }
        let opcode_pos = code.len() - 3;
        let opcode = code[opcode_pos];
        let src = u16::from_le_bytes([code[opcode_pos + 1], code[opcode_pos + 2]]);
        match opcode {
            insns::BC_INT_RETURN => Some((JitArgKind::Int, src)),
            insns::BC_REF_RETURN => Some((JitArgKind::Ref, src)),
            insns::BC_FLOAT_RETURN => Some((JitArgKind::Float, src)),
            _ => None,
        }
    }
}

pub(crate) fn read_u8(code: &[u8], cursor: &mut usize) -> u8 {
    let value = *code.get(*cursor).expect("truncated jitcode");
    *cursor += 1;
    value
}

pub(crate) fn read_u16(code: &[u8], cursor: &mut usize) -> u16 {
    let lo = *code.get(*cursor).expect("truncated jitcode");
    let hi = *code.get(*cursor + 1).expect("truncated jitcode");
    *cursor += 2;
    u16::from_le_bytes([lo, hi])
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_translate::jitcode::{JitCode as BuildJitCode, JitCodeBody as BuildJitCodeBody};

    #[test]
    fn wellknown_bh_insns_stays_canonical_and_avoids_false_call_family_keys() {
        let insns = wellknown_bh_insns();
        assert!(
            !insns.contains_key("jump/L"),
            "wellknown_bh_insns must keep the canonical goto/L spelling",
        );
        assert!(
            !insns.contains_key("conditional_call_ir_v/iiIRd"),
            "helper-side BC_COND_CALL_VOID must not masquerade as \
             canonical conditional_call_ir_v/iiIRd",
        );
        assert!(
            !insns.contains_key("conditional_call_value_ir_r/riIRd>r"),
            "helper-side BC_COND_CALL_VALUE_REF must not masquerade as \
             canonical conditional_call_value_ir_r/riIRd>r",
        );
        assert!(
            !insns.contains_key("record_known_result_r_ir_v/riIRd"),
            "helper-side BC_RECORD_KNOWN_RESULT_REF must not masquerade as \
             canonical record_known_result_r_ir_v/riIRd",
        );
        assert!(
            !insns.contains_key("inline_call_ir_r/dIR>r"),
            "helper-side BC_INLINE_CALL adapter must not masquerade as \
             canonical inline_call_ir_r/dIR>r",
        );
        assert!(
            !insns.contains_key("inline_call_irf_f/dIRF>f"),
            "helper-side BC_INLINE_CALL adapter must not masquerade as \
             canonical inline_call_irf_f/dIRF>f",
        );
        assert_eq!(
            insns.get("getfield_vable_i/rd>i"),
            Some(&super::insns::BC_GETFIELD_VABLE_I)
        );
        assert_eq!(
            insns.get("getfield_vable_r/rd>r"),
            Some(&super::insns::BC_GETFIELD_VABLE_R)
        );
        assert_eq!(
            insns.get("getfield_vable_f/rd>f"),
            Some(&super::insns::BC_GETFIELD_VABLE_F)
        );
        assert_eq!(
            insns.get("setfield_vable_i/rid"),
            Some(&super::insns::BC_SETFIELD_VABLE_I)
        );
        assert_eq!(
            insns.get("setfield_vable_r/rrd"),
            Some(&super::insns::BC_SETFIELD_VABLE_R)
        );
        assert_eq!(
            insns.get("setfield_vable_f/rfd"),
            Some(&super::insns::BC_SETFIELD_VABLE_F)
        );
        assert_eq!(
            insns.get("getarrayitem_vable_i/ridd>i"),
            Some(&super::insns::BC_GETARRAYITEM_VABLE_I)
        );
        assert_eq!(
            insns.get("getarrayitem_vable_r/ridd>r"),
            Some(&super::insns::BC_GETARRAYITEM_VABLE_R)
        );
        assert_eq!(
            insns.get("getarrayitem_vable_f/ridd>f"),
            Some(&super::insns::BC_GETARRAYITEM_VABLE_F)
        );
        assert_eq!(
            insns.get("setarrayitem_vable_i/riidd"),
            Some(&super::insns::BC_SETARRAYITEM_VABLE_I)
        );
        assert_eq!(
            insns.get("setarrayitem_vable_r/rirdd"),
            Some(&super::insns::BC_SETARRAYITEM_VABLE_R)
        );
        assert_eq!(
            insns.get("setarrayitem_vable_f/rifdd"),
            Some(&super::insns::BC_SETARRAYITEM_VABLE_F)
        );
        assert_eq!(
            insns.get("arraylen_vable/rdd>i"),
            Some(&super::insns::BC_ARRAYLEN_VABLE)
        );
        assert_eq!(
            insns.get("hint_force_virtualizable/r"),
            Some(&super::insns::BC_HINT_FORCE_VIRTUALIZABLE)
        );
        // Slice 0 of `pyre-call-family-canonical-migration.md` — canonical
        // residual_call_*_v opcodes reserved for Slice 1 emit migration.
        assert_eq!(
            insns.get("residual_call_r_v/iRd"),
            Some(&super::insns::BC_RESIDUAL_CALL_R_V),
        );
        assert_eq!(
            insns.get("residual_call_ir_v/iIRd"),
            Some(&super::insns::BC_RESIDUAL_CALL_IR_V),
        );
        assert_eq!(
            insns.get("residual_call_irf_v/iIRFd"),
            Some(&super::insns::BC_RESIDUAL_CALL_IRF_V),
        );
    }

    /// Tasks #94c (abort/*) and #94b' (state_*) quarantine the 8 pyre-
    /// only keys arising from Rust adaptations into
    /// `pyre_extension_insns()`. `wellknown_bh_insns()` becomes a strict
    /// subset of RPython's canonical opname universe. `insn_byte` merges
    /// both tables so build-time `write_insn(...)` callers continue to
    /// resolve unchanged.
    #[test]
    fn pyre_extension_insns_quarantines_pyre_only_keys_out_of_wellknown() {
        let wellknown = wellknown_bh_insns();
        let extension = pyre_extension_insns();

        let pairs = [
            // Borrow-checker abort signals.
            ("abort/", insns::BC_ABORT),
            ("abort_permanent/", insns::BC_ABORT_PERMANENT),
            // Proc-macro JIT-machine state addressing.
            ("load_state_field/di", insns::BC_LOAD_STATE_FIELD),
            ("store_state_field/di", insns::BC_STORE_STATE_FIELD),
            ("load_state_array/dii", insns::BC_LOAD_STATE_ARRAY),
            ("store_state_array/dii", insns::BC_STORE_STATE_ARRAY),
            ("load_state_varray/dii", insns::BC_LOAD_STATE_VARRAY),
            ("store_state_varray/dii", insns::BC_STORE_STATE_VARRAY),
            // pyre nested-bytecode inline_call (pyre-only `P` argcode).
            ("inline_call_pyre_nested/P", insns::BC_INLINE_CALL),
        ];

        for (key, expected_byte) in pairs {
            assert!(
                !wellknown.contains_key(key),
                "{key} must be quarantined in pyre_extension_insns(), not \
                 wellknown_bh_insns()",
            );
            assert_eq!(
                extension.get(key),
                Some(&expected_byte),
                "{key} must be present in pyre_extension_insns() with the \
                 fixed BC_* byte",
            );
            assert_eq!(
                majit_translate::insns::insn_byte(key),
                expected_byte,
                "insn_byte must resolve {key} via the merged extension+\
                 wellknown table",
            );
        }
    }

    #[test]
    fn canonical_build_jitcode_sizes_blackhole_register_files_without_conversion() {
        // Extract the upstream-common part of blackhole.py:312 setposition
        // (register sizing + constant copy) and apply it directly to the
        // canonical codewriter JitCode. Dispatch still needs the runtime
        // adapter JitCode for exec.* pools, but the register-file setup no
        // longer needs a build→runtime conversion just to match RPython's
        // `num_regs_* + len(constants_*)` logic.
        //
        // RPython: `blackhole.py:312 setposition` allocates `num_regs_i +
        // len(constants_i)` slots per register file and copies each constant
        // into the tail portion of the file. We verify both — the array
        // sizes and the copied-in constants.
        use crate::blackhole::BlackholeInterpBuilder;

        let body = BuildJitCodeBody {
            code: vec![insns::BC_LIVE, 0x00, 0x00], // live/ with 2-byte offset
            c_num_regs_i: 4,
            c_num_regs_r: 2,
            c_num_regs_f: 1,
            constants_i: vec![100, 200, 300],
            constants_r: vec![
                0xAABB_CCDD_EEFF_0011_u64 as i64,
                0x2233_4455_6677_8899_u64 as i64,
            ],
            constants_f: vec![f64::to_bits(1.25_f64) as i64],
            ..Default::default()
        };
        let bt = BuildJitCode::new("slice2/test");
        bt.set_body(body);

        let mut builder = BlackholeInterpBuilder::new();
        let mut bh = builder.acquire_interp();
        bh.prepare_registers_for_canonical_jitcode(&bt, 0);

        // num_regs_and_consts_i = 4 + 3 = 7; constants occupy [4..7].
        assert_eq!(bh.registers_i.len(), 7);
        assert_eq!(&bh.registers_i[4..7], &[100, 200, 300]);
        // Working regs remain zero-initialised.
        assert_eq!(&bh.registers_i[0..4], &[0, 0, 0, 0]);

        // Refs: u64 bit pattern reinterpreted as i64 by the conversion.
        assert_eq!(bh.registers_r.len(), 4); // 2 regs + 2 constants
        assert_eq!(bh.registers_r[2], 0xAABB_CCDD_EEFF_0011_u64 as i64);
        assert_eq!(bh.registers_r[3], 0x2233_4455_6677_8899_u64 as i64);

        // Floats: f64 bits reinterpreted; round-trip through f64::to_bits
        // must match what BlackholeInterpreter sees.
        assert_eq!(bh.registers_f.len(), 2);
        assert_eq!(bh.registers_f[1], f64::to_bits(1.25_f64) as i64);

        assert_eq!(bh.position, 0);
        assert!(bh.jitcode.code.is_empty());
    }
}
