//! Line-by-line port of `rpython/jit/codewriter/assembler.py`.
//!
//! This module mirrors the high-level `Assembler` flow from RPython:
//! `setup()` → `write_insn()` → `fix_labels()` → `check_result()` →
//! `make_jitcode()`. pyre still targets `majit_metainterp::jitcode::JitCode`
//! and its fixed `BC_*` builder API, so regular-op dispatch lowers the
//! textual `SSARepr` opnames into `JitCodeBuilder` calls instead of growing
//! `self.insns` into a runtime opcode table.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use majit_metainterp::jitcode::{JitCallArg, JitCode, JitCodeBuilder};

use super::flatten::{DescrOperand, Insn, Kind, ListOfKind, Operand, Register, SSARepr, TLabel};

/// `assembler.py:65` `count_regs = dict.fromkeys(KINDS, 0)` override.
#[derive(Debug, Clone, Copy, Default)]
pub struct NumRegs {
    pub int: u16,
    pub ref_: u16,
    pub float: u16,
}

/// `assembler.py:19-32` `class Assembler(object)`.
///
/// Writer-side state, per-instance, matching `Assembler.__init__`
/// (rpython/jit/codewriter/assembler.py:19-32) line-by-line. A fresh
/// `Assembler::new()` produces independent `all_liveness` /
/// `all_liveness_positions` / `num_liveness_ops` /
/// `indirectcalltargets`; upstream does the same — each `CodeWriter`
/// owns one `Assembler` and carries its own dedup state.
///
/// The reader side (`blackhole` / `trace_opcode`, both in the
/// lower `pyre_jit_trace` crate) still needs to read `all_liveness`
/// at runtime. pyre-jit-trace cannot depend on pyre-jit, so we
/// publish the latest `all_liveness` snapshot to
/// `pyre_jit_trace::assembler::ASSEMBLER_STATE` after every write
/// — PRE-EXISTING-ADAPTATION for the crate layering, not a state
/// relocation.
#[derive(Debug, Default)]
pub struct Assembler {
    /// `assembler.py:20` `self.insns = {}`.
    ///
    /// RPython grows this dict in `write_insn()` with dense opcode ids.
    /// pyre still emits majit's fixed runtime bytecodes, so the mirror here
    /// records only the actually-emitted well-known keys with their runtime
    /// opcode byte. That is sufficient for the lazy `finish_setup` cache
    /// refresh in `pyre_jit_trace::state`.
    insns: HashMap<String, u8>,
    /// `assembler.py:29` `self.all_liveness = []`.
    all_liveness: Vec<u8>,
    /// `assembler.py:30` `self.all_liveness_length = 0`.
    all_liveness_length: usize,
    /// `assembler.py:31` `self.all_liveness_positions = {}`.
    all_liveness_positions: HashMap<(Vec<u8>, Vec<u8>, Vec<u8>), u16>,
    /// `assembler.py:32` `self.num_liveness_ops = 0`.
    num_liveness_ops: usize,
    /// `assembler.py:24` `self.indirectcalltargets = set()    # set of JitCodes`.
    /// Accumulated across every `assemble()` call; `pyjitpl.py:2262`
    /// `self.setup_indirectcalltargets(asm.indirectcalltargets)` pipes
    /// this set to `MetaInterpStaticData.setup_indirectcalltargets`.
    pub indirectcalltargets: HashSet<ArcByPtr>,
}

/// Identity-keyed wrapper around `Arc<JitCode>` so `HashSet<ArcByPtr>`
/// matches RPython's `set of JitCodes` (Python set dedupes by object
/// identity, not value equality).  Used as the element type of
/// `Assembler::indirectcalltargets`.
///
/// The stored `JitCode` is still pyre's runtime adapter
/// `majit_metainterp::jitcode::JitCode`; once the runtime path consumes
/// the canonical codewriter `JitCode` directly, this wrapper should move
/// with that type rather than preserving a second object graph.
#[derive(Debug, Clone)]
pub struct ArcByPtr(pub Arc<JitCode>);

impl ArcByPtr {
    pub fn new(jc: Arc<JitCode>) -> Self {
        Self(jc)
    }

    pub fn into_inner(self) -> Arc<JitCode> {
        self.0
    }
}

impl PartialEq for ArcByPtr {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ArcByPtr {}

impl std::hash::Hash for ArcByPtr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.0) as *const () as usize).hash(state);
    }
}

/// Builder-local state created by `setup()` for one `assemble()` call.
///
/// RPython keeps most of these on `self`; pyre keeps them in a per-call
/// struct because the runtime `JitCodeBuilder` is consumed by `finish()`.
struct AssemblyState {
    builder: JitCodeBuilder,
    /// `assembler.py:59` `self.label_positions = {}`.
    label_positions: HashMap<String, usize>,
    /// Builder adapter for `Label/TLabel` name → builder label id.
    /// RPython stores bytecode positions directly in `label_positions`; this
    /// extra vector exists only because `JitCodeBuilder` patches jumps by
    /// symbolic label id rather than by rewriting raw bytes in `fix_labels()`.
    builder_labels: Vec<(String, u16)>,
}

impl Assembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// `assembler.py:29` accessor.
    pub fn all_liveness(&self) -> &[u8] {
        &self.all_liveness
    }

    /// `assembler.py:30` accessor.
    pub fn all_liveness_length(&self) -> usize {
        self.all_liveness_length
    }

    /// Snapshot of the emitted well-known `opname/argcodes` keys.
    pub fn insns_snapshot(&self) -> HashMap<String, u8> {
        self.insns.clone()
    }

    /// `assembler.py:32` accessor.
    pub fn num_liveness_ops(&self) -> usize {
        self.num_liveness_ops
    }

    /// Return the accumulated indirect-call target set as a deduped
    /// `Vec<Arc<JitCode>>`.  Matches the current runtime-adapter shape
    /// expected by `MetaInterpStaticData::setup_indirectcalltargets`;
    /// upstream parity for the dict-building semantics lives below that
    /// storage type, in `pyjitpl`'s fnaddr→jitcode helper.
    pub fn indirectcalltargets_vec(&self) -> Vec<Arc<JitCode>> {
        self.indirectcalltargets
            .iter()
            .map(|a| a.0.clone())
            .collect()
    }

    /// `assembler.py:300-305` `finished(self, callinfocollection)`.
    ///
    /// ```python
    /// def finished(self, callinfocollection):
    ///     # Helper called at the end of assembling.  Registers the extra
    ///     # functions shown in _callinfo_for_oopspec.
    ///     for func in callinfocollection.all_function_addresses_as_int():
    ///         func = int2adr(func)
    ///         self.see_raw_object(func.ptr)
    /// ```
    ///
    /// Called from `codewriter.py:85` after the `enum_pending_graphs`
    /// drain loop completes; registers raw helper-function addresses
    /// into `list_of_addr2name` for `MetaInterpStaticData`'s
    /// debug-symbol map.
    ///
    /// PRE-EXISTING-ADAPTATION: pyre has no `oopspec` system, so the
    /// callinfocollection is empty (see
    /// [`super::call::CallInfoCollection`]) and the iteration is a
    /// no-op. The slot is preserved so `make_jitcodes` calls through
    /// at exactly the same point as `codewriter.py:85`.
    pub fn finished(&mut self, callinfocollection: &super::call::CallInfoCollection) {
        let _ = callinfocollection;
        pyre_jit_trace::assembler::publish_state(
            &self.insns,
            &self.all_liveness,
            self.all_liveness_length,
            self.num_liveness_ops,
        );
    }

    /// `assembler.py:34-54` `assemble(self, ssarepr, jitcode=None, num_regs=None)`.
    pub fn assemble(
        &mut self,
        ssarepr: &mut SSARepr,
        mut builder: JitCodeBuilder,
        num_regs: Option<NumRegs>,
    ) -> JitCode {
        if let Some(nr) = num_regs {
            builder.ensure_i_regs(nr.int);
            builder.ensure_r_regs(nr.ref_);
            builder.ensure_f_regs(nr.float);
        }
        // Freeze num_regs at the regalloc-determined values: subsequent
        // operand emission must not grow the register file (every real
        // register slot is already inside `[0, num_regs_kind)` after
        // regalloc, and `Const*` args inside `ListOfKind` route through
        // the per-kind constants pool — see
        // `expect_int_reg_or_pool` and `dispatch_residual_call` for the
        // pool-byte synthesis that depends on stable `num_regs_kind`).
        builder.freeze_num_regs();
        builder.set_name(ssarepr.name.clone());

        let mut state = AssemblyState {
            builder,
            label_positions: HashMap::new(),
            builder_labels: Vec::new(),
        };

        ssarepr.insns_pos = Some(Vec::with_capacity(ssarepr.insns.len()));
        for insn in &ssarepr.insns {
            ssarepr
                .insns_pos
                .as_mut()
                .expect("insns_pos initialized")
                .push(state.builder.current_pos());
            self.write_insn(&mut state, insn);
        }

        self.fix_labels(&mut state);

        let jitcode = state.builder.finish();
        // `assembler.py:54`: run `check_result()` unconditionally after
        // `fix_labels()` and before `make_jitcode`. Strict RPython parity
        // — an oversized jitcode is a translator-level programming error,
        // so this asserts and aborts rather than degrading to a runtime
        // fallback. Any pyre-only safety net belongs outside this unit.
        self.check_result(&jitcode);
        jitcode
    }

    /// `assembler.py:140-223` `write_insn(insn)`.
    fn write_insn(&mut self, state: &mut AssemblyState, insn: &Insn) {
        match insn {
            Insn::Unreachable => {}
            // PRE-EXISTING-ADAPTATION (see `Insn::PcAnchor` docstring in
            // `flatten.rs`). Anchors carry no bytecode; they exist only
            // so the post-assemble byte-offset table (`ssarepr.insns_pos`)
            // records the position of each Python PC for trace-time
            // dispatch. No emit, no liveness side effect.
            Insn::PcAnchor(_) => {}
            Insn::Label(label) => {
                let label_id = builder_label(state, &label.name);
                state
                    .label_positions
                    .insert(label.name.clone(), state.builder.current_pos());
                state.builder.mark_label(label_id);
            }
            Insn::Op {
                opname,
                args,
                result,
            } => {
                // `assembler.py:143-158` `-live-` branch. `liveness.py:5-12`
                // uses `insn[0] == '-live-'` as the discriminator; pyre
                // matches on `opname` to keep the tuple-shape parity.
                if opname == super::flatten::OPNAME_LIVE {
                    // `assembler.py:149` increments `self.num_liveness_ops`.
                    // pyre's counter lives on `ASSEMBLER_STATE`;
                    // `intern_liveness` below takes care of the `+= 1`, so
                    // there is no local pre-increment here (matching
                    // upstream: the bump happens inside `_encode_liveness`).
                    let live_i = get_liveness_info(args, Kind::Int);
                    let live_r = get_liveness_info(args, Kind::Ref);
                    let live_f = get_liveness_info(args, Kind::Float);
                    let patch_offset = state.builder.live_placeholder();
                    let offset = self.encode_liveness_info(&live_i, &live_r, &live_f);
                    state.builder.patch_live_offset(patch_offset, offset);
                    self.record_insn_key(opname, args, None);
                    return;
                }
                // `assembler.py:208-209`:
                //   elif isinstance(x, IndirectCallTargets):
                //       self.indirectcalltargets.update(x.lst)
                //
                // RPython's `write_insn` iterates every operand and dispatches
                // by Python type.  pyre's `dispatch_op` is keyed on `opname`
                // (PRE-EXISTING-ADAPTATION — see below), so operand iteration
                // is split: `IndirectCallTargets` is collected here before the
                // opcode-specific lowering runs.  The operand is purely
                // metadata for the call — it is not written into the jitcode
                // byte stream (RPython emits nothing for this branch either).
                for arg in args {
                    if let Operand::IndirectCallTargets(targets) = arg {
                        for jc in &targets.lst {
                            self.indirectcalltargets.insert(ArcByPtr::new(jc.clone()));
                        }
                    }
                }
                self.record_insn_key(opname, args, result.as_ref());
                // `assembler.py:197-206` registers descrs inline during op
                // encoding into `Assembler.descrs`. pyre's runtime
                // codewriter has no descr-consuming ops today, so no
                // registration path runs here — once a descr-consuming op
                // lands it must register onto the shared
                // `BlackholeInterpBuilder.descrs` pool (`blackhole.py:
                // 102-103 setup_descrs`), not a per-jitcode duplicate.
                dispatch_op(state, opname, args, result.as_ref());
            }
        }
    }

    /// `assembler.py:250-263` `fix_labels()`.
    ///
    /// PRE-EXISTING-ADAPTATION: pyre's runtime `JitCodeBuilder` patches
    /// jumps by symbolic label id rather than by rewriting raw bytes, so
    /// the label-position loop is folded into the builder. The per-descr
    /// switch patching that upstream runs here is handled through the
    /// shared `BlackholeInterpBuilder.descrs` pool (`blackhole.py:102-103`
    /// `setup_descrs`), not per-jitcode — pyre's runtime codewriter has
    /// no descr-consuming ops yet, so this is currently a no-op.
    fn fix_labels(&mut self, _state: &mut AssemblyState) {}

    /// `assembler.py:265-269` `check_result()`.
    ///
    /// RPython enforces a 256-entry cap per kind because the assembled
    /// bytecode stream treats register-or-constant operands as one-byte
    /// indices. The surrounding pyre pipeline still relies on the same
    /// invariant for liveness and jit-merge-point operand encoding, so we
    /// keep the orthodox translator-time assertion here instead of
    /// widening the limit to match internal `u16` builder fields.
    fn check_result(&self, jitcode: &JitCode) {
        assert!(
            (jitcode.num_regs_i() as usize) + jitcode.constants_i.len() <= 256,
            "too many int registers/constants"
        );
        assert!(
            (jitcode.num_regs_r() as usize) + jitcode.constants_r.len() <= 256,
            "too many ref registers/constants"
        );
        assert!(
            (jitcode.num_regs_f() as usize) + jitcode.constants_f.len() <= 256,
            "too many float registers/constants"
        );
    }

    /// `assembler.py:234-248` `_encode_liveness(...)`.
    ///
    /// Line-by-line port of RPython's `_encode_liveness`:
    ///   - Dedup key is `(live_i, live_r, live_f)` — pyre uses sorted-byte
    ///     slices where RPython uses `frozenset`. Equivalent modulo the
    ///     caller's responsibility to present sorted/dedup'd input, which
    ///     is done by `get_liveness_info` before this call.
    ///   - On miss: append the three u8 length bytes + the encoded
    ///     bitsets, and advance `all_liveness_length`.
    ///   - `num_liveness_ops` bumps once per `-live-` write regardless of
    ///     dedup (`assembler.py:149` does it in `write_insn`, before this
    ///     helper is reached). Matches upstream.
    ///
    /// After the per-instance update we publish the latest buffer to
    /// the `pyre_jit_trace::assembler::ASSEMBLER_STATE` thread-local so
    /// the blackhole reader (a lower-crate consumer that cannot depend
    /// on `pyre_jit`) sees the same bytes. PRE-EXISTING-ADAPTATION for
    /// the crate layering, not a relocation of the RPython state.
    fn encode_liveness_info(&mut self, live_i: &[u8], live_r: &[u8], live_f: &[u8]) -> u16 {
        use majit_translate::liveness::encode_liveness;

        self.num_liveness_ops += 1;
        let key = (live_i.to_vec(), live_r.to_vec(), live_f.to_vec());
        let pos = if let Some(&cached) = self.all_liveness_positions.get(&key) {
            cached
        } else {
            assert!(
                self.all_liveness_length <= u16::MAX as usize,
                "all_liveness offset overflow"
            );
            assert!(
                live_i.len() <= u8::MAX as usize
                    && live_r.len() <= u8::MAX as usize
                    && live_f.len() <= u8::MAX as usize,
                "too many live registers in one slot"
            );
            let pos = self.all_liveness_length as u16;
            self.all_liveness_positions.insert(key, pos);
            self.all_liveness.push(live_i.len() as u8);
            self.all_liveness.push(live_r.len() as u8);
            self.all_liveness.push(live_f.len() as u8);
            self.all_liveness_length += 3;
            for live in [live_i, live_r, live_f] {
                let encoded = encode_liveness(live);
                self.all_liveness.extend_from_slice(&encoded);
                self.all_liveness_length += encoded.len();
            }
            pos
        };
        pyre_jit_trace::assembler::publish_state(
            &self.insns,
            &self.all_liveness,
            self.all_liveness_length,
            self.num_liveness_ops,
        );
        pos
    }

    fn record_insn_key(&mut self, opname: &str, args: &[Operand], result: Option<&Register>) {
        if is_adapter_only_helper_call_family(opname) {
            return;
        }
        let key = if opname == super::flatten::OPNAME_LIVE {
            "live/".to_string()
        } else {
            insn_key(opname, args, result)
        };
        if let Some(&opcode) = WELLKNOWN_BH_INSNS.get(key.as_str()) {
            self.insns.entry(key).or_insert(opcode);
        }
    }
}

static WELLKNOWN_BH_INSNS: LazyLock<HashMap<&'static str, u8>> =
    LazyLock::new(majit_metainterp::jitcode::wellknown_bh_insns);

fn is_adapter_only_helper_call_family(opname: &str) -> bool {
    matches!(
        opname,
        "conditional_call_ir_v"
            | "conditional_call_value_ir_i"
            | "conditional_call_value_ir_r"
            | "record_known_result_i_ir_v"
            | "record_known_result_r_ir_v"
    )
}

fn insn_key(opname: &str, args: &[Operand], result: Option<&Register>) -> String {
    let mut argcodes = String::new();
    let allow_short = use_c_form(opname);
    for arg in args {
        match arg {
            Operand::Register(reg) => argcodes.push(kind_char(reg.kind)),
            Operand::ConstInt(value) => {
                let code = if allow_short && (-128..=127).contains(value) {
                    'c'
                } else {
                    'i'
                };
                argcodes.push(code);
            }
            Operand::ConstRef(_) => argcodes.push('r'),
            Operand::ConstFloat(_) => argcodes.push('f'),
            Operand::TLabel(_) => argcodes.push('L'),
            Operand::ListOfKind(list) => argcodes.push(kind_char(list.kind).to_ascii_uppercase()),
            Operand::Descr(_) => argcodes.push('d'),
            Operand::IndirectCallTargets(_) => {}
        }
    }
    if let Some(result) = result {
        argcodes.push('>');
        argcodes.push(kind_char(result.kind));
    }
    format!("{opname}/{argcodes}")
}

fn kind_char(kind: Kind) -> char {
    match kind {
        Kind::Int => 'i',
        Kind::Ref => 'r',
        Kind::Float => 'f',
    }
}

fn use_c_form(opname: &str) -> bool {
    matches!(
        opname,
        "copystrcontent"
            | "getarrayitem_gc_pure_i"
            | "getarrayitem_gc_pure_r"
            | "getarrayitem_gc_i"
            | "getarrayitem_gc_r"
            | "goto_if_not_int_eq"
            | "goto_if_not_int_ge"
            | "goto_if_not_int_gt"
            | "goto_if_not_int_le"
            | "goto_if_not_int_lt"
            | "goto_if_not_int_ne"
            | "int_add"
            | "int_and"
            | "int_copy"
            | "int_eq"
            | "int_ge"
            | "int_gt"
            | "int_le"
            | "int_lt"
            | "int_ne"
            | "int_return"
            | "int_sub"
            | "jit_merge_point"
            | "new_array"
            | "new_array_clear"
            | "newstr"
            | "setarrayitem_gc_i"
            | "setarrayitem_gc_r"
            | "setfield_gc_i"
            | "strgetitem"
            | "strsetitem"
            | "foobar"
            | "baz"
    )
}

/// Test-only convenience wrapper — creates a throwaway `Assembler` per
/// call. **Do not wire this into production paths.** RPython's
/// `CodeWriter` holds a single `Assembler` whose `all_liveness`,
/// `all_liveness_positions`, and `num_liveness_ops` accumulate across
/// every JitCode compiled in the session (`codewriter.py:19-22` /
/// `pyjitpl.py:2264`). A fresh `Assembler` per call splits the
/// global liveness table and breaks that invariant.
///
/// Production callers must construct one `Assembler` up front and call
/// `assembler.assemble(...)` repeatedly on it.
#[cfg(test)]
fn assemble(ssarepr: &mut SSARepr, builder: JitCodeBuilder, num_regs: Option<NumRegs>) -> JitCode {
    let mut assembler = Assembler::new();
    assembler.assemble(ssarepr, builder, num_regs)
}

fn builder_label(state: &mut AssemblyState, name: &str) -> u16 {
    if let Some((_, label)) = state
        .builder_labels
        .iter()
        .find(|(existing, _)| existing == name)
    {
        return *label;
    }
    let label = state.builder.new_label();
    state.builder_labels.push((name.to_owned(), label));
    label
}

fn get_liveness_info(args: &[Operand], kind: Kind) -> Vec<u8> {
    let mut lives = Vec::new();
    for arg in args {
        if let Operand::Register(Register {
            kind: reg_kind,
            index,
        }) = arg
        {
            if *reg_kind == kind {
                lives.push(u8::try_from(*index).expect("liveness register index exceeds u8"));
            }
        }
    }
    lives.sort_unstable();
    lives.dedup();
    lives
}

fn dispatch_op(
    state: &mut AssemblyState,
    opname: &str,
    args: &[Operand],
    result: Option<&Register>,
) {
    match opname {
        "goto" => {
            let label = expect_tlabel(&args[0]);
            let label_id = builder_label(state, &label.name);
            state.builder.jump(label_id);
        }
        "goto_if_not" | "goto_if_not_int_is_true" => {
            let cond = expect_reg(&args[0], Kind::Int);
            let label = expect_tlabel(&args[1]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_int_is_true(cond, label_id);
        }
        "goto_if_not_int_eq" => {
            let lhs = expect_reg(&args[0], Kind::Int);
            let rhs = expect_reg(&args[1], Kind::Int);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_int_eq(lhs, rhs, label_id);
        }
        "goto_if_not_int_ne" => {
            let lhs = expect_reg(&args[0], Kind::Int);
            let rhs = expect_reg(&args[1], Kind::Int);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_int_ne(lhs, rhs, label_id);
        }
        "goto_if_not_int_lt" => {
            let lhs = expect_reg(&args[0], Kind::Int);
            let rhs = expect_reg(&args[1], Kind::Int);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_int_lt(lhs, rhs, label_id);
        }
        "goto_if_not_int_le" => {
            let lhs = expect_reg(&args[0], Kind::Int);
            let rhs = expect_reg(&args[1], Kind::Int);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_int_le(lhs, rhs, label_id);
        }
        "goto_if_not_int_gt" => {
            let lhs = expect_reg(&args[0], Kind::Int);
            let rhs = expect_reg(&args[1], Kind::Int);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_int_gt(lhs, rhs, label_id);
        }
        "goto_if_not_int_ge" => {
            let lhs = expect_reg(&args[0], Kind::Int);
            let rhs = expect_reg(&args[1], Kind::Int);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_int_ge(lhs, rhs, label_id);
        }
        "goto_if_not_int_is_zero" => {
            let cond = expect_reg(&args[0], Kind::Int);
            let label = expect_tlabel(&args[1]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_int_is_zero(cond, label_id);
        }
        "goto_if_not_float_lt" => {
            let lhs = expect_reg(&args[0], Kind::Float);
            let rhs = expect_reg(&args[1], Kind::Float);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_float_lt(lhs, rhs, label_id);
        }
        "goto_if_not_float_le" => {
            let lhs = expect_reg(&args[0], Kind::Float);
            let rhs = expect_reg(&args[1], Kind::Float);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_float_le(lhs, rhs, label_id);
        }
        "goto_if_not_float_eq" => {
            let lhs = expect_reg(&args[0], Kind::Float);
            let rhs = expect_reg(&args[1], Kind::Float);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_float_eq(lhs, rhs, label_id);
        }
        "goto_if_not_float_ne" => {
            let lhs = expect_reg(&args[0], Kind::Float);
            let rhs = expect_reg(&args[1], Kind::Float);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_float_ne(lhs, rhs, label_id);
        }
        "goto_if_not_float_gt" => {
            let lhs = expect_reg(&args[0], Kind::Float);
            let rhs = expect_reg(&args[1], Kind::Float);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_float_gt(lhs, rhs, label_id);
        }
        "goto_if_not_float_ge" => {
            let lhs = expect_reg(&args[0], Kind::Float);
            let rhs = expect_reg(&args[1], Kind::Float);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_float_ge(lhs, rhs, label_id);
        }
        "goto_if_not_ptr_eq" => {
            let lhs = expect_reg(&args[0], Kind::Ref);
            let rhs = expect_reg(&args[1], Kind::Ref);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_ptr_eq(lhs, rhs, label_id);
        }
        "goto_if_not_ptr_ne" => {
            let lhs = expect_reg(&args[0], Kind::Ref);
            let rhs = expect_reg(&args[1], Kind::Ref);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_ptr_ne(lhs, rhs, label_id);
        }
        "goto_if_not_ptr_iszero" => {
            let cond = expect_reg(&args[0], Kind::Ref);
            let label = expect_tlabel(&args[1]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_ptr_iszero(cond, label_id);
        }
        "goto_if_not_ptr_nonzero" => {
            let cond = expect_reg(&args[0], Kind::Ref);
            let label = expect_tlabel(&args[1]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_ptr_nonzero(cond, label_id);
        }
        "catch_exception" => {
            let label = expect_tlabel(&args[0]);
            let label_id = builder_label(state, &label.name);
            state.builder.catch_exception(label_id);
        }
        "goto_if_exception_mismatch" => {
            let vtable = expect_int_reg_or_pool(state, &args[0]);
            let label = expect_tlabel(&args[1]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_exception_mismatch(vtable, label_id);
        }
        "jit_merge_point" => {
            // RPython jtransform.py:1690-1712 emits the upstream shape:
            // `[jd_index_const, greens_i, greens_r, greens_f, reds_i,
            // reds_r, reds_f]`. blackhole.py:1066
            // `@arguments("self","i","I","R","F","I","R","F")` reads
            // jdindex + the six typed register lists generally; the
            // helper writes the same byte stream regardless of which
            // lists are empty for the current jitdriver.
            let [
                jdindex_arg,
                greens_i_arg,
                greens_r_arg,
                greens_f_arg,
                reds_i_arg,
                reds_r_arg,
                reds_f_arg,
            ] = args
            else {
                panic!(
                    "jit_merge_point: expected 7 upstream-orthodox args, got {}",
                    args.len()
                );
            };
            // jtransform.py:1704 emits `Constant(self.portal_jd.index,
            // lltype.Signed)`, so the assembler-side jdindex flows
            // through as a signed integer. The builder selects the
            // USE_C_FORM short byte vs the constants-pool `i` slot per
            // `assembler.py:99-107` (`-128 <= value <= 127`); no
            // pre-builder narrowing here.
            let jdindex = match jdindex_arg {
                Operand::ConstInt(v) => *v,
                other => {
                    panic!("jit_merge_point expects ConstInt jd_index at arg[0], got {other:?}")
                }
            };
            let greens_i = expect_list_regs_or_pool(state, greens_i_arg, Kind::Int);
            let greens_r = expect_list_regs_or_pool(state, greens_r_arg, Kind::Ref);
            let greens_f = expect_list_regs_or_pool(state, greens_f_arg, Kind::Float);
            let reds_i = expect_list_regs_or_pool(state, reds_i_arg, Kind::Int);
            let reds_r = expect_list_regs_or_pool(state, reds_r_arg, Kind::Ref);
            let reds_f = expect_list_regs_or_pool(state, reds_f_arg, Kind::Float);
            state.builder.jit_merge_point(
                jdindex, &greens_i, &greens_r, &greens_f, &reds_i, &reds_r, &reds_f,
            );
        }
        "loop_header" => {
            // RPython jtransform.py:1714-1718 handle_jit_marker__loop_header
            // emits SpaceOperation('loop_header', [c_index], None) with
            // `Constant(jd.index, lltype.Signed)`. `loop_header` is not
            // in `assembler.py:312-346 USE_C_FORM`, so the builder
            // always emits the constants-pool `i` form regardless of
            // jdindex magnitude — pass the signed value through.
            let jdindex = match &args[0] {
                Operand::ConstInt(v) => *v,
                other => panic!("loop_header expects ConstInt jdindex, got {:?}", other),
            };
            state.builder.loop_header(jdindex);
        }
        "ref_return" => {
            let src = expect_result_or_first_reg(args, result, Kind::Ref);
            state.builder.ref_return(src);
        }
        "raise" => {
            let src = expect_reg(&args[0], Kind::Ref);
            state.builder.emit_raise(src);
        }
        "reraise" => state.builder.emit_reraise(),
        "last_exception" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state.builder.last_exception(dst);
        }
        "last_exc_value" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Ref);
            state.builder.last_exc_value(dst);
        }
        "abort" => state.builder.abort(),
        "abort_permanent" => state.builder.abort_permanent(),
        "unreachable" => state.builder.unreachable(),
        // `flatten.py:333` `self.emitline('%s_copy' % kind, v, "->", w)`.
        // `v` is either a `Register` or a `Constant`. Register source
        // lowers to the primitive `move_{i,r,f}` builder method;
        // Constant source lowers via the builder's constant-pool helper,
        // matching the two argcode variants emitted by
        // `assembler.py:162-174` (`i` for Register, `c` for Constant).
        "int_copy" => match source_operand(args, result) {
            MoveSource::Reg(src) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Int);
                let src = expect_reg(src, Kind::Int);
                state.builder.move_i(dst, src);
            }
            MoveSource::ConstInt(value) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Int);
                state.builder.load_const_i_value(dst, value);
            }
            other => panic!("int_copy expects Register or ConstInt, got {:?}", other),
        },
        "ref_copy" => match source_operand(args, result) {
            MoveSource::Reg(src) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Ref);
                let src = expect_reg(src, Kind::Ref);
                state.builder.move_r(dst, src);
            }
            MoveSource::ConstRef(value) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Ref);
                state.builder.load_const_r_value(dst, value);
            }
            other => panic!("ref_copy expects Register or ConstRef, got {:?}", other),
        },
        "float_copy" => match source_operand(args, result) {
            MoveSource::Reg(src) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Float);
                let src = expect_reg(src, Kind::Float);
                state.builder.move_f(dst, src);
            }
            MoveSource::ConstFloat(value) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Float);
                state.builder.load_const_f_value(dst, value);
            }
            other => panic!("float_copy expects Register or ConstFloat, got {:?}", other),
        },
        // `flatten.py:329` `self.emitline('%s_push' % kind, v)` — cycle-break
        // save into the kind-typed scratch slot.
        "int_push" => {
            let src = expect_reg(&args[0], Kind::Int);
            state.builder.push_i(src);
        }
        "ref_push" => {
            let src = expect_reg(&args[0], Kind::Ref);
            state.builder.push_r(src);
        }
        "float_push" => {
            let src = expect_reg(&args[0], Kind::Float);
            state.builder.push_f(src);
        }
        // `flatten.py:331` `self.emitline('%s_pop' % kind, "->", w)` —
        // cycle-break load from the kind-typed scratch slot.
        "int_pop" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state.builder.pop_i(dst);
        }
        "ref_pop" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Ref);
            state.builder.pop_r(dst);
        }
        "float_pop" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Float);
            state.builder.pop_f(dst);
        }
        // `load_state_*` / `store_state_*` arms removed: pyre-specific
        // opnames with no SSARepr emitter — the pyre value-stack runtime
        // ops are emitted directly through the JitCodeBuilder rather than
        // routed through the SSARepr dispatch table.
        //
        // RPython opname parity (jtransform.py:844-927, blackhole.py:1446-1493):
        // `kind = getkind(...)[0]` in jtransform.py forces single-char
        // suffix everywhere — `getfield_vable_i` / `_r` / `_f` and
        // `setfield_vable_i` / `_r` / `_f`. pyre's JitCodeBuilder methods
        // retain the `vable_*` prefix as a PRE-EXISTING-ADAPTATION so the
        // rename is scoped to the Insn::Op key.  Long-form `_int` /
        // `_ref` / `_float` aliases retired — no SSARepr emitter writes
        // them (codewriter.rs `emit_vable_*` macros use the short form).
        "getfield_vable_i" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state
                .builder
                .vable_getfield_int(dst, expect_small_u16(&args[0]));
        }
        "getfield_vable_r" => {
            // 2-arg shape mirroring `jtransform.py:846-847` getfield:
            // `[v_inst, descr]` with a Ref result.  `args[0]` is the
            // `v_inst` sentinel (`ConstRef(0)`) — pyre's single-frame
            // model leaves the frame implicit at `state.frame`.
            // `field_idx` derives from the trailing
            // `VableStaticField` descr.
            let dst = expect_result_or_first_reg(args, result, Kind::Ref);
            let _v_inst = &args[0];
            let field_idx = expect_descr_vable_static_field(&args[1]);
            state.builder.vable_getfield_ref(dst, field_idx);
        }
        "getfield_vable_f" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Float);
            state
                .builder
                .vable_getfield_float(dst, expect_small_u16(&args[0]));
        }
        "setfield_vable_i" => {
            // 3-arg shape mirroring `jtransform.py:927-928` setfield:
            // `[v_inst, v_value, descr]`.  `args[0]` is the `v_inst`
            // sentinel; see `getfield_vable_r` for rationale.
            // `field_idx` derives from the trailing
            // `VableStaticField` descr.
            let _v_inst = &args[0];
            let value_reg = expect_reg(&args[1], Kind::Int);
            let field_idx = expect_descr_vable_static_field(&args[2]);
            state.builder.vable_setfield_int(field_idx, value_reg);
        }
        "setfield_vable_r" => state
            .builder
            .vable_setfield_ref(expect_small_u16(&args[0]), expect_reg(&args[1], Kind::Ref)),
        "setfield_vable_f" => state.builder.vable_setfield_float(
            expect_small_u16(&args[0]),
            expect_reg(&args[1], Kind::Float),
        ),
        "getarrayitem_vable_i" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state.builder.vable_getarrayitem_int(
                dst,
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
            );
        }
        "getarrayitem_vable_r" => {
            // 4-arg shape mirroring `jtransform.py:1882-1885
            // do_fixed_list_getitem` (vable branch): `[v_base,
            // v_index, arrayfielddescr, arraydescr]` with a Ref
            // result.  See `setarrayitem_vable_r` arm for the
            // descr-pair extraction + lowering rationale.
            let dst = expect_result_or_first_reg(args, result, Kind::Ref);
            let _v_base = &args[0];
            let index_reg = expect_reg(&args[1], Kind::Int);
            let array_idx = expect_descr_vable_array_field(&args[2]);
            let array_descr_idx = expect_descr_vable_array(&args[3]);
            assert_eq!(
                array_idx, array_descr_idx,
                "getarrayitem_vable_r: VableArrayField({}) and VableArray({}) descrs \
                 must reference the same array field",
                array_idx, array_descr_idx,
            );
            state
                .builder
                .vable_getarrayitem_ref(dst, array_idx, index_reg);
        }
        "getarrayitem_vable_f" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Float);
            state.builder.vable_getarrayitem_float(
                dst,
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
            );
        }
        "setarrayitem_vable_i" => state.builder.vable_setarrayitem_int(
            expect_small_u16(&args[0]),
            expect_reg(&args[1], Kind::Int),
            expect_reg(&args[2], Kind::Int),
        ),
        "setarrayitem_vable_r" => {
            // 5-arg shape mirroring `jtransform.py:1898-1906
            // do_fixed_list_setitem` (vable branch): `[v_base,
            // v_index, v_value, arrayfielddescr, arraydescr]`.
            // `args[0]` is the `v_base` sentinel (`ConstRef(0)`) —
            // pyre's single-frame model leaves the frame implicit at
            // `state.frame`; the operand exists for shape parity.
            // `array_idx` derives from the trailing `VableArrayField`
            // descr; `VableArray` is sanity-checked to match.
            // Mirroring `assembler.py:80-138 emit_const`'s role of
            // compressing descr operands into bytecode bytes.
            let _v_base = &args[0];
            let index_reg = expect_reg(&args[1], Kind::Int);
            let array_idx = expect_descr_vable_array_field(&args[3]);
            let array_descr_idx = expect_descr_vable_array(&args[4]);
            assert_eq!(
                array_idx, array_descr_idx,
                "setarrayitem_vable_r: VableArrayField({}) and VableArray({}) descrs \
                 must reference the same array field",
                array_idx, array_descr_idx,
            );
            match &args[2] {
                Operand::Register(r) if r.kind == Kind::Ref => {
                    state
                        .builder
                        .vable_setarrayitem_ref(array_idx, index_reg, r.index);
                }
                // `setarrayitem_vable_r(vable, idx, ConstPtr(value))` —
                // jtransform.py:1898 lowers ConstPtr value operands as
                // SSA constants. The builder side reuses the same
                // BC_SETARRAYITEM_VABLE_R bytecode and points its src
                // u16 at the constants suffix of the unified register
                // space (matching upstream `assembler.py:80-138 emit_const`).
                Operand::ConstRef(value) => {
                    state
                        .builder
                        .vable_setarrayitem_ref_const_value(array_idx, index_reg, *value);
                }
                other => panic!(
                    "setarrayitem_vable_r expects Register(Ref) or ConstRef, got {:?}",
                    other,
                ),
            }
        }
        "setarrayitem_vable_f" => state.builder.vable_setarrayitem_float(
            expect_small_u16(&args[0]),
            expect_reg(&args[1], Kind::Int),
            expect_reg(&args[2], Kind::Float),
        ),
        "arraylen_vable" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state
                .builder
                .vable_arraylen(dst, expect_small_u16(&args[0]));
        }
        "hint_force_virtualizable" => state.builder.vable_force(),
        // Per-OpCode opname dispatch for integer / float primitives —
        // RPython `rpython/jit/metainterp/blackhole.py:459-723` defines
        // one `bhimpl_*` per opname; `assembler.py:162-222` routes each
        // via its own insns key. The SSARepr layer carries the
        // canonical opname so pyre dispatch picks the matching
        // `record_binop_*` / `record_unary_*` helper without a generic
        // OpCode-bearing wrapper insn. No `generic record_*` SSA
        // opname exists; only the canonical RPython names.
        "int_add" | "int_sub" | "int_mul" | "int_floordiv" | "int_mod" | "int_and" | "int_or"
        | "int_xor" | "int_lshift" | "int_rshift" | "int_eq" | "int_ne" | "int_lt" | "int_le"
        | "int_gt" | "int_ge" | "uint_rshift" | "uint_mul_high" | "uint_lt" | "uint_le"
        | "uint_gt" | "uint_ge" => {
            let dst = expect_result_reg(result, Kind::Int, "int binop needs result");
            let opcode = match opname {
                "int_add" => majit_ir::OpCode::IntAdd,
                "int_sub" => majit_ir::OpCode::IntSub,
                "int_mul" => majit_ir::OpCode::IntMul,
                "int_floordiv" => majit_ir::OpCode::IntFloorDiv,
                "int_mod" => majit_ir::OpCode::IntMod,
                "int_and" => majit_ir::OpCode::IntAnd,
                "int_or" => majit_ir::OpCode::IntOr,
                "int_xor" => majit_ir::OpCode::IntXor,
                "int_lshift" => majit_ir::OpCode::IntLshift,
                "int_rshift" => majit_ir::OpCode::IntRshift,
                "int_eq" => majit_ir::OpCode::IntEq,
                "int_ne" => majit_ir::OpCode::IntNe,
                "int_lt" => majit_ir::OpCode::IntLt,
                "int_le" => majit_ir::OpCode::IntLe,
                "int_gt" => majit_ir::OpCode::IntGt,
                "int_ge" => majit_ir::OpCode::IntGe,
                "uint_rshift" => majit_ir::OpCode::UintRshift,
                "uint_mul_high" => majit_ir::OpCode::UintMulHigh,
                "uint_lt" => majit_ir::OpCode::UintLt,
                "uint_le" => majit_ir::OpCode::UintLe,
                "uint_gt" => majit_ir::OpCode::UintGt,
                "uint_ge" => majit_ir::OpCode::UintGe,
                _ => unreachable!(),
            };
            state.builder.record_binop_i(
                dst,
                opcode,
                expect_reg(&args[0], Kind::Int),
                expect_reg(&args[1], Kind::Int),
            );
        }
        "int_neg" | "int_invert" => {
            let dst = expect_result_reg(result, Kind::Int, "int unary needs result");
            let opcode = match opname {
                "int_neg" => majit_ir::OpCode::IntNeg,
                "int_invert" => majit_ir::OpCode::IntInvert,
                _ => unreachable!(),
            };
            state
                .builder
                .record_unary_i(dst, opcode, expect_reg(&args[0], Kind::Int));
        }
        "ptr_eq" | "ptr_ne" | "instance_ptr_eq" | "instance_ptr_ne" => {
            let dst = expect_result_reg(result, Kind::Int, "ptr binop needs result");
            let opcode = match opname {
                "ptr_eq" => majit_ir::OpCode::PtrEq,
                "ptr_ne" => majit_ir::OpCode::PtrNe,
                "instance_ptr_eq" => majit_ir::OpCode::InstancePtrEq,
                "instance_ptr_ne" => majit_ir::OpCode::InstancePtrNe,
                _ => unreachable!(),
            };
            state.builder.record_binop_r(
                dst,
                opcode,
                expect_reg(&args[0], Kind::Ref),
                expect_reg(&args[1], Kind::Ref),
            );
        }
        "ptr_iszero" => {
            let dst = expect_result_reg(result, Kind::Int, "ptr_iszero needs result");
            state
                .builder
                .ptr_iszero(dst, expect_reg(&args[0], Kind::Ref));
        }
        "ptr_nonzero" => {
            let dst = expect_result_reg(result, Kind::Int, "ptr_nonzero needs result");
            state
                .builder
                .ptr_nonzero(dst, expect_reg(&args[0], Kind::Ref));
        }
        "float_add" | "float_sub" | "float_mul" | "float_truediv" => {
            let dst = expect_result_reg(result, Kind::Float, "float binop needs result");
            let opcode = match opname {
                "float_add" => majit_ir::OpCode::FloatAdd,
                "float_sub" => majit_ir::OpCode::FloatSub,
                "float_mul" => majit_ir::OpCode::FloatMul,
                "float_truediv" => majit_ir::OpCode::FloatTrueDiv,
                _ => unreachable!(),
            };
            state.builder.record_binop_f(
                dst,
                opcode,
                expect_reg(&args[0], Kind::Float),
                expect_reg(&args[1], Kind::Float),
            );
        }
        "float_neg" | "float_abs" => {
            let dst = expect_result_reg(result, Kind::Float, "float unary needs result");
            let opcode = match opname {
                "float_neg" => majit_ir::OpCode::FloatNeg,
                "float_abs" => majit_ir::OpCode::FloatAbs,
                _ => unreachable!(),
            };
            state
                .builder
                .record_unary_f(dst, opcode, expect_reg(&args[0], Kind::Float));
        }
        "call_void" | "residual_call_void" => {
            let fn_idx = expect_small_u16(&args[0]);
            let call_args = expect_call_args(&args[1..]);
            state
                .builder
                .residual_call_void_typed_args(fn_idx, &call_args);
        }
        "call_may_force_void" => {
            let fn_idx = expect_small_u16(&args[0]);
            let call_args = expect_call_args(&args[1..]);
            state
                .builder
                .call_may_force_void_typed_args(fn_idx, &call_args);
        }
        "call_release_gil_void" => {
            let fn_idx = expect_small_u16(&args[0]);
            let call_args = expect_call_args(&args[1..]);
            state
                .builder
                .call_release_gil_void_typed_args(fn_idx, &call_args);
        }
        "call_loopinvariant_void" => {
            let fn_idx = expect_small_u16(&args[0]);
            let call_args = expect_call_args(&args[1..]);
            state
                .builder
                .call_loopinvariant_void_typed_args(fn_idx, &call_args);
        }
        "call_assembler_void" => {
            let target_idx = expect_small_u16(&args[0]);
            let call_args = expect_call_args(&args[1..]);
            state
                .builder
                .call_assembler_void_typed_args(target_idx, &call_args);
        }
        "call_int"
        | "call_pure_int"
        | "call_may_force_int"
        | "call_release_gil_int"
        | "call_loopinvariant_int"
        | "call_assembler_int" => {
            let fn_idx = expect_small_u16(&args[0]);
            let dst = expect_result_reg(result, Kind::Int, "int call needs result");
            let call_args = expect_call_args(&args[1..]);
            match opname {
                "call_int" => state.builder.call_int_typed(fn_idx, &call_args, dst),
                "call_pure_int" => state.builder.call_pure_int_typed(fn_idx, &call_args, dst),
                "call_may_force_int" => state
                    .builder
                    .call_may_force_int_typed(fn_idx, &call_args, dst),
                "call_release_gil_int" => state
                    .builder
                    .call_release_gil_int_typed(fn_idx, &call_args, dst),
                "call_loopinvariant_int" => state
                    .builder
                    .call_loopinvariant_int_typed(fn_idx, &call_args, dst),
                _ => state
                    .builder
                    .call_assembler_int_typed(fn_idx, &call_args, dst),
            }
        }
        "call_ref"
        | "call_pure_ref"
        | "call_may_force_ref"
        | "call_loopinvariant_ref"
        | "call_assembler_ref" => {
            let fn_idx = expect_small_u16(&args[0]);
            let dst = expect_result_reg(result, Kind::Ref, "ref call needs result");
            let call_args = expect_call_args(&args[1..]);
            match opname {
                "call_ref" => state.builder.call_ref_typed(fn_idx, &call_args, dst),
                "call_pure_ref" => state.builder.call_pure_ref_typed(fn_idx, &call_args, dst),
                "call_may_force_ref" => state
                    .builder
                    .call_may_force_ref_typed(fn_idx, &call_args, dst),
                // "call_release_gil_ref" intentionally rejected:
                // resoperation.py:1243-1244 (`# no such thing`).
                "call_loopinvariant_ref" => state
                    .builder
                    .call_loopinvariant_ref_typed(fn_idx, &call_args, dst),
                _ => state
                    .builder
                    .call_assembler_ref_typed(fn_idx, &call_args, dst),
            }
        }
        "call_float"
        | "call_pure_float"
        | "call_may_force_float"
        | "call_release_gil_float"
        | "call_loopinvariant_float"
        | "call_assembler_float" => {
            let fn_idx = expect_small_u16(&args[0]);
            let dst = expect_result_reg(result, Kind::Float, "float call needs result");
            let call_args = expect_call_args(&args[1..]);
            match opname {
                "call_float" => state.builder.call_float_typed(fn_idx, &call_args, dst),
                "call_pure_float" => state.builder.call_pure_float_typed(fn_idx, &call_args, dst),
                "call_may_force_float" => state
                    .builder
                    .call_may_force_float_typed(fn_idx, &call_args, dst),
                "call_release_gil_float" => state
                    .builder
                    .call_release_gil_float_typed(fn_idx, &call_args, dst),
                "call_loopinvariant_float" => state
                    .builder
                    .call_loopinvariant_float_typed(fn_idx, &call_args, dst),
                _ => state
                    .builder
                    .call_assembler_float_typed(fn_idx, &call_args, dst),
            }
        }
        "conditional_call_ir_v" => {
            let fn_idx = expect_small_u16(&args[0]);
            let cond_reg = expect_reg(&args[1], Kind::Int);
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .conditional_call_ir_v_typed_args(fn_idx, cond_reg, &call_args);
        }
        "conditional_call_value_ir_i" => {
            let fn_idx = expect_small_u16(&args[0]);
            let value_reg = expect_reg(&args[1], Kind::Int);
            let dst = expect_result_reg(
                result,
                Kind::Int,
                "conditional_call_value_ir_i needs result",
            );
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .conditional_call_value_ir_i_typed_args(fn_idx, value_reg, &call_args, dst);
        }
        "conditional_call_value_ir_r" => {
            let fn_idx = expect_small_u16(&args[0]);
            let value_reg = expect_reg(&args[1], Kind::Ref);
            let dst = expect_result_reg(
                result,
                Kind::Ref,
                "conditional_call_value_ir_r needs result",
            );
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .conditional_call_value_ir_r_typed_args(fn_idx, value_reg, &call_args, dst);
        }
        "record_known_result_i_ir_v" => {
            let fn_idx = expect_small_u16(&args[0]);
            let result_reg = expect_reg(&args[1], Kind::Int);
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .record_known_result_i_ir_v_typed_args(fn_idx, result_reg, &call_args);
        }
        "record_known_result_r_ir_v" => {
            let fn_idx = expect_small_u16(&args[0]);
            let result_reg = expect_reg(&args[1], Kind::Ref);
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .record_known_result_r_ir_v_typed_args(fn_idx, result_reg, &call_args);
        }
        // `rpython/jit/codewriter/jtransform.py:414-435 rewrite_call` shape.
        // 12 opname combinations: residual_call_{r,ir,irf}_{i,r,f,v}. Only
        // the subset the walker currently emits is routed; unused arms
        // panic with a pointer to emit_residual_call.
        opname if opname.starts_with("residual_call_") => {
            dispatch_residual_call(state, opname, args, result);
        }
        other => panic!(
            "assemble(): unimplemented opname {:?} — add a builder mapping in jit/assembler.rs",
            other
        ),
    }
}

/// Phase 3c consumer for the `residual_call_{kinds}_{reskind}` shape
/// emitted by `codewriter::emit_residual_call_shape`.
///
/// SSARepr arg layout (as produced by `emit_residual_call_shape`):
/// ```text
/// [Const(fn_idx),
///  ListOfKind(int,   [...])?,   # present iff 'i' in kinds
///  ListOfKind(ref,   [...])?,   # present iff 'r' in kinds
///  ListOfKind(float, [...])?,   # present iff 'f' in kinds
///  Descr(CallDescrStub{effect_info, arg_kinds})]
/// ```
///
/// pyre's `JitCodeBuilder::call_*_typed` takes a single flat
/// `&[JitCallArg]` in C-function parameter order. `arg_kinds` carries
/// that order; this helper reassembles by pulling one entry at a time
/// from the appropriate per-kind sublist in upstream's `bh_call_*`
/// pool layout.
fn dispatch_residual_call(
    state: &mut AssemblyState,
    opname: &str,
    args: &[Operand],
    result: Option<&Register>,
) {
    // opname suffix = `{kinds}_{reskind}`, kinds ∈ {r, ir, irf}.
    let tail = &opname["residual_call_".len()..];
    let (kinds, reskind_ch) = tail
        .rsplit_once('_')
        .unwrap_or_else(|| panic!("malformed residual_call opname: {opname:?}"));

    let fn_idx = expect_small_u16(&args[0]);

    // Walk `args[1..]`: per-kind ListOfKind sublists come first in
    // (int, ref, float) order per the `kinds` string, then the final
    // Descr operand carrying `CallDescrStub`.
    let mut cursor = 1usize;
    let mut int_list: &[Operand] = &[];
    let mut ref_list: &[Operand] = &[];
    let mut float_list: &[Operand] = &[];
    if kinds.contains('i') {
        match &args[cursor] {
            Operand::ListOfKind(ListOfKind {
                kind: Kind::Int,
                content,
            }) => int_list = content,
            other => {
                panic!("residual_call: expected ListOfKind(int) at index {cursor}, got {other:?}")
            }
        }
        cursor += 1;
    }
    if kinds.contains('r') {
        match &args[cursor] {
            Operand::ListOfKind(ListOfKind {
                kind: Kind::Ref,
                content,
            }) => ref_list = content,
            other => {
                panic!("residual_call: expected ListOfKind(ref) at index {cursor}, got {other:?}")
            }
        }
        cursor += 1;
    }
    if kinds.contains('f') {
        match &args[cursor] {
            Operand::ListOfKind(ListOfKind {
                kind: Kind::Float,
                content,
            }) => float_list = content,
            other => {
                panic!("residual_call: expected ListOfKind(float) at index {cursor}, got {other:?}")
            }
        }
        cursor += 1;
    }
    // CallDescrStub slot.
    let stub = match &args[cursor] {
        Operand::Descr(rc) => match &**rc {
            DescrOperand::CallDescrStub(stub) => stub,
            other => panic!(
                "residual_call: expected CallDescrStub descr at index {cursor}, got {other:?}"
            ),
        },
        other => panic!("residual_call: expected Descr operand at index {cursor}, got {other:?}"),
    };

    // Reassemble flat `&[JitCallArg]` in C-function parameter order.
    // `ListOfKind` slots admit `Variable | Constant`
    // (`rpython/jit/codewriter/jtransform.py:437-445` `make_three_lists`);
    // route `Const*` through the per-kind constant pool to synthesize the
    // `num_regs_kind + pool_idx` pseudo-register byte (mirrors
    // `assembler.py:80-138 emit_const` and the `expect_list_regs_or_pool`
    // helper used by `jit_merge_point` lowering).
    let mut call_args: Vec<JitCallArg> = Vec::with_capacity(stub.arg_kinds.len());
    let mut i_cursor = 0usize;
    let mut r_cursor = 0usize;
    let mut f_cursor = 0usize;
    for &ak in &stub.arg_kinds {
        match ak {
            Kind::Int => {
                let reg = expect_int_reg_or_pool(state, &int_list[i_cursor]);
                call_args.push(JitCallArg::int(reg));
                i_cursor += 1;
            }
            Kind::Ref => {
                let reg = expect_ref_reg_or_pool(state, &ref_list[r_cursor]);
                call_args.push(JitCallArg::reference(reg));
                r_cursor += 1;
            }
            Kind::Float => {
                let reg = expect_float_reg_or_pool(state, &float_list[f_cursor]);
                call_args.push(JitCallArg::float(reg));
                f_cursor += 1;
            }
        }
    }

    use super::flatten::{CallFlavor, ResKind, dispatch_kind_for_effect_info};
    let reskind = match reskind_ch {
        "i" => ResKind::Int,
        "r" => ResKind::Ref,
        "f" => ResKind::Float,
        "v" => ResKind::Void,
        other => panic!("residual_call: unknown reskind {other:?}"),
    };

    // Slice 3 of the EffectInfo wire-up epic: derive the dispatch branch
    // from `stub.effect_info`. Mirrors `pyjitpl.py:1995-2126
    // do_residual_call`'s precedence — the optimizer upstream picks
    // `call_may_force` / `call_release_gil` / `call_loopinvariant` /
    // `call_pure` from the same effectinfo bits, so pyre's pre-baked
    // codewriter branch resolves to the same builder method through the
    // predicates instead of a producer-side enum.
    let dispatch_kind = dispatch_kind_for_effect_info(&stub.effect_info);
    match (dispatch_kind, reskind) {
        (CallFlavor::Plain, ResKind::Int) => {
            let dst = expect_result_reg(result, Kind::Int, "residual_call int needs result");
            state.builder.call_int_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::Plain, ResKind::Ref) => {
            let dst = expect_result_reg(result, Kind::Ref, "residual_call ref needs result");
            state.builder.call_ref_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::Plain, ResKind::Void) => {
            state
                .builder
                .residual_call_void_typed_args(fn_idx, &call_args);
        }
        (CallFlavor::MayForce, ResKind::Int) => {
            let dst = expect_result_reg(
                result,
                Kind::Int,
                "residual_call may_force int needs result",
            );
            state
                .builder
                .call_may_force_int_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::MayForce, ResKind::Ref) => {
            let dst = expect_result_reg(
                result,
                Kind::Ref,
                "residual_call may_force ref needs result",
            );
            state
                .builder
                .call_may_force_ref_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::Plain, ResKind::Float) => {
            let dst = expect_result_reg(result, Kind::Float, "residual_call float needs result");
            state.builder.call_float_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::MayForce, ResKind::Float) => {
            let dst = expect_result_reg(
                result,
                Kind::Float,
                "residual_call may_force float needs result",
            );
            state
                .builder
                .call_may_force_float_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::MayForce, ResKind::Void) => {
            state
                .builder
                .call_may_force_void_typed_args(fn_idx, &call_args);
        }
        (CallFlavor::Pure, ResKind::Int) => {
            let dst = expect_result_reg(result, Kind::Int, "residual_call pure int needs result");
            state.builder.call_pure_int_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::Pure, ResKind::Ref) => {
            let dst = expect_result_reg(result, Kind::Ref, "residual_call pure ref needs result");
            state.builder.call_pure_ref_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::Pure, ResKind::Float) => {
            let dst =
                expect_result_reg(result, Kind::Float, "residual_call pure float needs result");
            state.builder.call_pure_float_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::Pure, ResKind::Void) => {
            panic!("dispatch_residual_call: pure void call is not a valid rewrite_call shape");
        }
        (CallFlavor::ReleaseGil, ResKind::Int) => {
            let dst = expect_result_reg(
                result,
                Kind::Int,
                "residual_call release_gil int needs result",
            );
            state
                .builder
                .call_release_gil_int_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::ReleaseGil, ResKind::Ref) => {
            // RPython `resoperation.py:1238 call_release_gil_for_descr`
            // explicitly skips the `tp == 'r'` arm with the comment
            // `# no such thing` (`:1243-1244`):
            //
            //     # no such thing
            //     #elif tp == 'r':
            //     #    return rop.CALL_RELEASE_GIL_R
            //
            // and `:1462 is_call_release_gil` excludes
            // `CALL_RELEASE_GIL_R` from the predicate.  No upstream
            // dispatcher / optimizer / backend can consume a
            // `CALL_RELEASE_GIL_R` op; emitting one is a NEW-DEVIATION
            // surface that pyre rejects at the codewriter layer.
            // The `OpCode::CallReleaseGilR` enum variant has been
            // removed from `majit-ir` per the upstream comment.
            panic!(
                "dispatch_residual_call: ReleaseGil + ResKind::Ref has no upstream \
                 counterpart (resoperation.py:1243-1244 `# no such thing`); pyre \
                 must not emit CALL_RELEASE_GIL_R via the residual_call shape."
            );
        }
        (CallFlavor::ReleaseGil, ResKind::Float) => {
            let dst = expect_result_reg(
                result,
                Kind::Float,
                "residual_call release_gil float needs result",
            );
            state
                .builder
                .call_release_gil_float_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::ReleaseGil, ResKind::Void) => {
            state
                .builder
                .call_release_gil_void_typed_args(fn_idx, &call_args);
        }
        (CallFlavor::LoopInvariant, ResKind::Int) => {
            let dst = expect_result_reg(
                result,
                Kind::Int,
                "residual_call loopinvariant int needs result",
            );
            state
                .builder
                .call_loopinvariant_int_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::LoopInvariant, ResKind::Ref) => {
            let dst = expect_result_reg(
                result,
                Kind::Ref,
                "residual_call loopinvariant ref needs result",
            );
            state
                .builder
                .call_loopinvariant_ref_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::LoopInvariant, ResKind::Float) => {
            let dst = expect_result_reg(
                result,
                Kind::Float,
                "residual_call loopinvariant float needs result",
            );
            state
                .builder
                .call_loopinvariant_float_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::LoopInvariant, ResKind::Void) => {
            state
                .builder
                .call_loopinvariant_void_typed_args(fn_idx, &call_args);
        }
    }
}

fn expect_result_or_first_reg(args: &[Operand], result: Option<&Register>, kind: Kind) -> u16 {
    match result {
        Some(reg) if reg.kind == kind => reg.index,
        Some(reg) => panic!("expected result register of kind {:?}, got {:?}", kind, reg),
        None => expect_reg(&args[0], kind),
    }
}

fn expect_result_reg(result: Option<&Register>, kind: Kind, msg: &str) -> u16 {
    match result {
        Some(reg) if reg.kind == kind => reg.index,
        Some(reg) => panic!("expected result register of kind {:?}, got {:?}", kind, reg),
        None => panic!("{}", msg),
    }
}

fn expect_reg(op: &Operand, expected: Kind) -> u16 {
    match op {
        Operand::Register(Register { kind, index }) if *kind == expected => *index,
        _ => panic!("expected Register({:?}, _), got {:?}", expected, op),
    }
}

fn expect_tlabel(op: &Operand) -> &TLabel {
    match op {
        Operand::TLabel(label) => label,
        _ => panic!("expected TLabel, got {:?}", op),
    }
}

fn expect_small_u16(op: &Operand) -> u16 {
    match op {
        Operand::ConstInt(value) => u16::try_from(*value).expect("expected u16-sized ConstInt"),
        _ => panic!("expected ConstInt(u16), got {:?}", op),
    }
}

/// `rpython/jit/metainterp/virtualizable.py:73 array_field_descrs[i]`:
/// extract the per-array index `i` carried by an
/// `Operand::Descr(DescrOperand::VableArrayField(i))` operand at the
/// trailing position of `getarrayitem_vable_X` / `setarrayitem_vable_X`
/// / `arraylen_vable` SSA ops (`jtransform.py:1880-1906`,
/// `:1865-1873`). Used by the assembler dispatch to recover the
/// `array_idx:u16` that the existing `vable_*` builder API expects.
fn expect_descr_vable_array_field(op: &Operand) -> u16 {
    match op {
        Operand::Descr(rc) => match &**rc {
            DescrOperand::VableArrayField(idx) => *idx,
            other => panic!("expected DescrOperand::VableArrayField, got {:?}", other),
        },
        _ => panic!("expected Operand::Descr(VableArrayField), got {:?}", op),
    }
}

/// `rpython/jit/metainterp/virtualizable.py:58 array_descrs[i]`:
/// counterpart of `expect_descr_vable_array_field` for the second
/// trailing descr operand (the GcArray layout descr).  Always paired
/// with a `VableArrayField(i)` at the preceding position; the
/// dispatch arms cross-check the indices match.
fn expect_descr_vable_array(op: &Operand) -> u16 {
    match op {
        Operand::Descr(rc) => match &**rc {
            DescrOperand::VableArray(idx) => *idx,
            other => panic!("expected DescrOperand::VableArray, got {:?}", other),
        },
        _ => panic!("expected Operand::Descr(VableArray), got {:?}", op),
    }
}

/// `rpython/jit/metainterp/virtualizable.py:71 static_field_descrs[i]`:
/// extract the per-field index `i` carried by an
/// `Operand::Descr(DescrOperand::VableStaticField(i))` operand at the
/// trailing position of `getfield_vable_<kind>` / `setfield_vable_<kind>`
/// SSA ops (`jtransform.py:846-927`).  Used by the assembler dispatch
/// to recover the `field_idx:u16` that the existing `vable_getfield_*`
/// / `vable_setfield_*` builder API expects.
fn expect_descr_vable_static_field(op: &Operand) -> u16 {
    match op {
        Operand::Descr(rc) => match &**rc {
            DescrOperand::VableStaticField(idx) => *idx,
            other => panic!("expected DescrOperand::VableStaticField, got {:?}", other),
        },
        _ => panic!("expected Operand::Descr(VableStaticField), got {:?}", op),
    }
}

fn expect_int_reg_or_pool(state: &mut AssemblyState, op: &Operand) -> u16 {
    match op {
        Operand::Register(Register {
            kind: Kind::Int,
            index,
        }) => *index,
        Operand::ConstInt(value) => {
            let idx = state.builder.add_const_i(*value);
            state.builder.num_regs_i() + idx
        }
        other => panic!(
            "expected Int register or ConstInt routable through constant pool, got {:?}",
            other
        ),
    }
}

fn expect_ref_reg_or_pool(state: &mut AssemblyState, op: &Operand) -> u16 {
    match op {
        Operand::Register(Register {
            kind: Kind::Ref,
            index,
        }) => *index,
        Operand::ConstRef(value) => {
            let idx = state.builder.add_const_r(*value);
            state.builder.num_regs_r() + idx
        }
        other => panic!(
            "expected Ref register or ConstRef routable through constant pool, got {:?}",
            other
        ),
    }
}

fn expect_float_reg_or_pool(state: &mut AssemblyState, op: &Operand) -> u16 {
    match op {
        Operand::Register(Register {
            kind: Kind::Float,
            index,
        }) => *index,
        Operand::ConstFloat(value) => {
            let idx = state.builder.add_const_f(*value);
            state.builder.num_regs_f() + idx
        }
        other => panic!(
            "expected Float register or ConstFloat routable through constant pool, got {:?}",
            other
        ),
    }
}

fn expect_list_regs(op: &Operand, expected: Kind) -> Vec<u8> {
    match op {
        Operand::ListOfKind(ListOfKind { kind, content }) if *kind == expected => content
            .iter()
            .map(|item| {
                u8::try_from(expect_reg(item, expected)).expect("register index exceeds u8")
            })
            .collect(),
        _ => panic!("expected ListOfKind({:?}), got {:?}", expected, op),
    }
}

/// Like `expect_list_regs`, but also accepts `Operand::ConstInt` /
/// `Operand::ConstRef` entries inside the `ListOfKind` and routes them
/// through the runtime constant pool (`builder.add_const_i` /
/// `builder.add_const_r`), returning the pool-backed register index
/// (`num_regs_kind + pool_idx`).
///
/// Matches upstream `rpython/jit/codewriter/assembler.py:157-175`
/// behavior where the assembler handles both pre-colored `Register`s
/// and raw `Constant`s inside `ListOfKind` by emitting distinct
/// argcodes ('i'/'r'/'f' for Register, 'c' for Constant) and routing
/// Constants through the per-kind constant table at write time.
///
/// Pyre's portal `jit_merge_point` graph path produces the 7-arg
/// upstream-orthodox shape via
/// `codewriter.rs::portal_jit_merge_point_graph_args` +
/// `GraphFlattener` (which emits raw `ConstInt`/`ConstRef` inside
/// `ListOfKind` because pyre's flow-graph constants are not
/// pool-routed upstream), so this helper accepts both already-colored
/// registers and constants inside those lists.
fn expect_list_regs_or_pool(state: &mut AssemblyState, op: &Operand, expected: Kind) -> Vec<u8> {
    let ListOfKind { kind, content } = match op {
        Operand::ListOfKind(list) => list,
        _ => panic!("expected ListOfKind({:?}), got {:?}", expected, op),
    };
    assert_eq!(
        *kind, expected,
        "list kind mismatch: expected {expected:?}, got {:?}",
        kind
    );
    content
        .iter()
        .map(|item| {
            let reg_idx = match item {
                Operand::Register(Register { kind, index }) => {
                    assert_eq!(
                        *kind, expected,
                        "register kind mismatch inside list: expected {expected:?}, got {:?}",
                        kind
                    );
                    *index
                }
                Operand::ConstInt(value) => {
                    assert_eq!(
                        expected,
                        Kind::Int,
                        "ConstInt found inside non-int ListOfKind({expected:?})"
                    );
                    let idx = state.builder.add_const_i(*value);
                    state.builder.num_regs_i() + idx
                }
                Operand::ConstRef(value) => {
                    assert_eq!(
                        expected,
                        Kind::Ref,
                        "ConstRef found inside non-ref ListOfKind({expected:?})"
                    );
                    let idx = state.builder.add_const_r(*value);
                    state.builder.num_regs_r() + idx
                }
                other => panic!(
                    "expected Register/ConstInt/ConstRef inside ListOfKind({expected:?}), \
                     got {other:?}"
                ),
            };
            u8::try_from(reg_idx).expect("register index exceeds u8")
        })
        .collect()
}

fn expect_call_args(args: &[Operand]) -> Vec<JitCallArg> {
    args.iter().map(expect_call_arg).collect()
}

fn expect_call_arg(op: &Operand) -> JitCallArg {
    match op {
        Operand::Register(Register { kind, index }) => match kind {
            Kind::Int => JitCallArg::int(*index),
            Kind::Ref => JitCallArg::reference(*index),
            Kind::Float => JitCallArg::float(*index),
        },
        _ => panic!("expected typed call register, got {:?}", op),
    }
}

/// `flatten.py:333` source operand for `{kind}_copy`. The source may
/// be either a Register (argcode `'i'`/`'r'`/`'f'`) or a Constant
/// (argcode `'c'` in short form). `MoveSource` captures the shape so
/// the dispatch arm can lower to the right `JitCodeBuilder` method.
#[derive(Debug)]
enum MoveSource<'a> {
    Reg(&'a Operand),
    ConstInt(i64),
    ConstRef(i64),
    ConstFloat(i64),
}

fn source_operand<'a>(args: &'a [Operand], result: Option<&Register>) -> MoveSource<'a> {
    let src_operand = if result.is_some() { &args[0] } else { &args[1] };
    match src_operand {
        Operand::Register(_) => MoveSource::Reg(src_operand),
        Operand::ConstInt(v) => MoveSource::ConstInt(*v),
        Operand::ConstRef(v) => MoveSource::ConstRef(*v),
        Operand::ConstFloat(v) => MoveSource::ConstFloat(*v),
        _ => panic!(
            "source_operand: unexpected source operand {:?}",
            src_operand
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::super::flatten::{CallDescrStub, CallFlavor, DescrOperand, Label, SwitchDictDescr};
    use super::*;

    fn r(kind: Kind, index: u16) -> Register {
        Register::new(kind, index)
    }

    #[test]
    fn assemble_empty_ssarepr_produces_valid_jitcode() {
        let mut ssarepr = SSARepr::new("empty");
        let jitcode = assemble(&mut ssarepr, JitCodeBuilder::default(), None);
        assert_eq!(jitcode.name, "empty");
        assert!(jitcode.code.is_empty(), "empty SSARepr -> empty code");
        assert_eq!(ssarepr.insns_pos, Some(Vec::new()));
    }

    #[test]
    fn assemble_records_insn_positions_and_liveness() {
        let mut ssarepr = SSARepr::new("live");
        ssarepr
            .insns
            .push(Insn::live(vec![Operand::Register(r(Kind::Ref, 0))]));
        ssarepr.insns.push(Insn::op(
            "ref_return",
            vec![Operand::Register(r(Kind::Ref, 0))],
        ));

        let mut assembler = Assembler::new();
        let before_ops = assembler.num_liveness_ops();
        let jitcode = assembler.assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        // `num_liveness_ops` and `all_liveness` live on the thread-local
        // `ASSEMBLER_STATE` (the single RPython-`Assembler` analog); other
        // tests running earlier on the same thread may already have
        // bumped the counter, so assert on the delta and the non-empty
        // buffer rather than a raw == 1.
        assert_eq!(assembler.num_liveness_ops() - before_ops, 1);
        assert!(!assembler.all_liveness().is_empty());
        assert_eq!(ssarepr.insns_pos.as_ref().map(Vec::len), Some(2));
        assert!(!jitcode.code.is_empty());
    }

    #[test]
    fn assemble_accumulates_used_wellknown_insns() {
        let mut assembler = Assembler::new();

        let mut first = SSARepr::new("first");
        first
            .insns
            .push(Insn::live(vec![Operand::Register(r(Kind::Ref, 0))]));
        first.insns.push(Insn::op(
            "ref_return",
            vec![Operand::Register(r(Kind::Ref, 0))],
        ));
        assembler.assemble(
            &mut first,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        let mut second = SSARepr::new("second");
        second
            .insns
            .push(Insn::op("goto", vec![Operand::TLabel(TLabel::new("L1"))]));
        second.insns.push(Insn::Label(Label::new("L1")));
        assembler.assemble(&mut second, JitCodeBuilder::default(), None);

        let insns = assembler.insns_snapshot();
        let wellknown = majit_metainterp::jitcode::wellknown_bh_insns();
        assert_eq!(insns.get("live/"), wellknown.get("live/"));
        assert_eq!(insns.get("ref_return/r"), wellknown.get("ref_return/r"));
        assert_eq!(insns.get("goto/L"), wellknown.get("goto/L"));
    }

    #[test]
    fn assemble_accumulates_canonical_last_exc_value_and_jit_merge_point_keys() {
        let mut assembler = Assembler::new();

        let mut ssarepr = SSARepr::new("portal");
        ssarepr.insns.push(Insn::op(
            "jit_merge_point",
            vec![
                Operand::ConstInt(0),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Int,
                    vec![
                        Operand::Register(Register::new(Kind::Int, 0)),
                        Operand::Register(Register::new(Kind::Int, 1)),
                    ],
                )),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Ref,
                    vec![Operand::Register(Register::new(Kind::Ref, 0))],
                )),
                Operand::ListOfKind(ListOfKind::new(Kind::Float, Vec::new())),
                Operand::ListOfKind(ListOfKind::new(Kind::Int, Vec::new())),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Ref,
                    vec![
                        Operand::Register(Register::new(Kind::Ref, 1)),
                        Operand::Register(Register::new(Kind::Ref, 2)),
                    ],
                )),
                Operand::ListOfKind(ListOfKind::new(Kind::Float, Vec::new())),
            ],
        ));
        ssarepr.insns.push(Insn::op_with_result(
            "last_exc_value",
            Vec::new(),
            Register::new(Kind::Ref, 3),
        ));
        assembler.assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 2,
                ref_: 4,
                ..NumRegs::default()
            }),
        );

        let insns = assembler.insns_snapshot();
        let wellknown = majit_metainterp::jitcode::wellknown_bh_insns();
        // The fixture above passes `ConstInt(0)` for jdindex which is in
        // the signed-byte range, so `assembler.py:163,312 USE_C_FORM`
        // selects the `'c'` argcode and the canonical key is
        // `jit_merge_point/cIRFIRF`.
        assert_eq!(
            insns.get("jit_merge_point/cIRFIRF"),
            wellknown.get("jit_merge_point/cIRFIRF")
        );
        assert_eq!(
            insns.get("last_exc_value/>r"),
            wellknown.get("last_exc_value/>r")
        );
    }

    #[test]
    fn assemble_accumulates_canonical_goto_if_exception_mismatch_key() {
        let mut assembler = Assembler::new();

        let mut ssarepr = SSARepr::new("exc_match");
        ssarepr.insns.push(Insn::op(
            "goto_if_exception_mismatch",
            vec![Operand::ConstInt(7), Operand::TLabel(TLabel::new("L1"))],
        ));
        ssarepr.insns.push(Insn::Label(Label::new("L1")));
        ssarepr.insns.push(Insn::op(
            "ref_return",
            vec![Operand::Register(r(Kind::Ref, 0))],
        ));
        assembler.assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        let insns = assembler.insns_snapshot();
        let wellknown = majit_metainterp::jitcode::wellknown_bh_insns();
        assert_eq!(
            insns.get("goto_if_exception_mismatch/iL"),
            wellknown.get("goto_if_exception_mismatch/iL")
        );
    }

    #[test]
    fn assemble_patches_jumps_through_labels() {
        let mut ssarepr = SSARepr::new("jump");
        ssarepr
            .insns
            .push(Insn::op("goto", vec![Operand::TLabel(TLabel::new("L1"))]));
        ssarepr.insns.push(Insn::Label(Label::new("L1")));
        ssarepr.insns.push(Insn::op(
            "ref_return",
            vec![Operand::Register(r(Kind::Ref, 0))],
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        assert_eq!(jitcode.follow_jump(3), 3);
    }

    #[test]
    fn assemble_int_copy_from_constant_uses_copy_opcode_and_constant_pool() {
        let mut ssarepr = SSARepr::new("const_copy");
        ssarepr.insns.push(Insn::op_with_result(
            "int_copy",
            vec![Operand::ConstInt(42)],
            Register::new(Kind::Int, 0),
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 1,
                ..NumRegs::default()
            }),
        );

        let copy_opcode = *majit_metainterp::jitcode::wellknown_bh_insns()
            .get("int_copy/i>i")
            .expect("int_copy must be registered in wellknown insns");
        assert_eq!(jitcode.constants_i, vec![42]);
        assert_eq!(jitcode.code[0], copy_opcode);
        assert_eq!(u16::from_le_bytes([jitcode.code[1], jitcode.code[2]]), 1);
        assert_eq!(u16::from_le_bytes([jitcode.code[3], jitcode.code[4]]), 0);
    }

    #[test]
    fn assemble_ptr_nonzero_uses_canonical_ref_nullity_opcode() {
        let mut ssarepr = SSARepr::new("ptr_nonzero");
        ssarepr.insns.push(Insn::op_with_result(
            "ptr_nonzero",
            vec![Operand::Register(r(Kind::Ref, 0))],
            Register::new(Kind::Int, 0),
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 1,
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        let opcode = *majit_metainterp::jitcode::wellknown_bh_insns()
            .get("ptr_nonzero/r>i")
            .expect("ptr_nonzero must be registered in wellknown insns");
        assert_eq!(jitcode.code[0], opcode);
        assert_eq!(u16::from_le_bytes([jitcode.code[1], jitcode.code[2]]), 0);
        assert_eq!(u16::from_le_bytes([jitcode.code[3], jitcode.code[4]]), 0);
    }

    #[test]
    fn assemble_goto_if_not_ptr_nonzero_uses_canonical_branch_opcode() {
        let mut ssarepr = SSARepr::new("ptr_branch");
        ssarepr.insns.push(Insn::op(
            "goto_if_not_ptr_nonzero",
            vec![
                Operand::Register(r(Kind::Ref, 0)),
                Operand::TLabel(TLabel::new("L1")),
            ],
        ));
        ssarepr.insns.push(Insn::Label(Label::new("L1")));
        // Append a trailing return so `L1` is not at end-of-code.
        // RPython `jitcode.py:108-112 follow_jump` asserts
        // `labelvalue < len(code)` strictly, so the label must not
        // resolve to `len(code)` itself.
        ssarepr.insns.push(Insn::op(
            "ref_return",
            vec![Operand::Register(r(Kind::Ref, 0))],
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        let opcode = *majit_metainterp::jitcode::wellknown_bh_insns()
            .get("goto_if_not_ptr_nonzero/rL")
            .expect("ptr_nonzero branch must be registered in wellknown insns");
        assert_eq!(jitcode.code[0], opcode);
        assert_eq!(jitcode.follow_jump(5), 5);
    }

    #[test]
    fn assemble_conditional_call_value_ir_r_keeps_ref_register_bank() {
        let mut ssarepr = SSARepr::new("cond_call_value_ir_r");
        ssarepr.insns.push(Insn::op_with_result(
            "conditional_call_value_ir_r",
            vec![
                Operand::ConstInt(7),
                Operand::Register(Register::new(Kind::Ref, 1)),
                Operand::Register(Register::new(Kind::Int, 0)),
                Operand::Register(Register::new(Kind::Ref, 2)),
            ],
            Register::new(Kind::Ref, 4),
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 1,
                ref_: 5,
                ..NumRegs::default()
            }),
        );

        assert_eq!(jitcode.num_regs_i(), 1);
        assert_eq!(jitcode.num_regs_r(), 5);
    }

    #[test]
    fn assemble_record_known_result_r_ir_v_keeps_ref_register_bank() {
        let mut ssarepr = SSARepr::new("record_known_result_r_ir_v");
        ssarepr.insns.push(Insn::op(
            "record_known_result_r_ir_v",
            vec![
                Operand::ConstInt(7),
                Operand::Register(Register::new(Kind::Ref, 2)),
                Operand::Register(Register::new(Kind::Int, 0)),
                Operand::Register(Register::new(Kind::Ref, 1)),
            ],
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 1,
                ref_: 3,
                ..NumRegs::default()
            }),
        );

        assert_eq!(jitcode.num_regs_i(), 1);
        assert_eq!(jitcode.num_regs_r(), 3);
    }

    #[test]
    fn assemble_canonical_helper_call_family_does_not_publish_false_insn_keys() {
        let mut assembler = Assembler::new();

        let mut ssarepr = SSARepr::new("canonical_call_family");
        ssarepr.insns.push(Insn::op(
            "conditional_call_ir_v",
            vec![
                Operand::ConstInt(7),
                Operand::Register(Register::new(Kind::Int, 0)),
                Operand::Register(Register::new(Kind::Int, 1)),
                Operand::Register(Register::new(Kind::Ref, 0)),
            ],
        ));
        ssarepr.insns.push(Insn::op_with_result(
            "conditional_call_value_ir_r",
            vec![
                Operand::ConstInt(7),
                Operand::Register(Register::new(Kind::Ref, 1)),
                Operand::Register(Register::new(Kind::Int, 1)),
                Operand::Register(Register::new(Kind::Ref, 0)),
            ],
            Register::new(Kind::Ref, 2),
        ));
        ssarepr.insns.push(Insn::op(
            "record_known_result_r_ir_v",
            vec![
                Operand::ConstInt(7),
                Operand::Register(Register::new(Kind::Ref, 2)),
                Operand::Register(Register::new(Kind::Int, 1)),
                Operand::Register(Register::new(Kind::Ref, 0)),
            ],
        ));
        assembler.assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 2,
                ref_: 3,
                ..NumRegs::default()
            }),
        );

        let insns = assembler.insns_snapshot();
        assert_eq!(
            insns.get("conditional_call_ir_v/iiIRd"),
            None,
            "helper-side conditional_call_ir_v payload is not canonical iiIRd",
        );
        assert_eq!(
            insns.get("conditional_call_value_ir_r/riIRd>r"),
            None,
            "helper-side conditional_call_value_ir_r payload is not canonical riIRd>r",
        );
        assert_eq!(
            insns.get("record_known_result_r_ir_v/riIRd"),
            None,
            "helper-side record_known_result_r_ir_v payload is not canonical riIRd",
        );
    }

    #[test]
    #[should_panic(expected = "unimplemented opname")]
    fn assemble_legacy_conditional_call_alias_is_rejected() {
        let mut ssarepr = SSARepr::new("legacy_conditional_call_alias");
        ssarepr.insns.push(Insn::op(
            "conditional_call_void",
            vec![
                Operand::ConstInt(7),
                Operand::Register(Register::new(Kind::Int, 0)),
                Operand::Register(Register::new(Kind::Int, 1)),
                Operand::Register(Register::new(Kind::Ref, 0)),
            ],
        ));

        let _ = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 2,
                ref_: 1,
                ..NumRegs::default()
            }),
        );
    }

    #[test]
    #[should_panic(expected = "unimplemented opname")]
    fn assemble_legacy_record_known_result_alias_is_rejected() {
        let mut ssarepr = SSARepr::new("legacy_record_known_result_alias");
        ssarepr.insns.push(Insn::op(
            "record_known_result_ref",
            vec![
                Operand::ConstInt(7),
                Operand::Register(Register::new(Kind::Ref, 2)),
                Operand::Register(Register::new(Kind::Int, 0)),
                Operand::Register(Register::new(Kind::Ref, 1)),
            ],
        ));

        let _ = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 1,
                ref_: 3,
                ..NumRegs::default()
            }),
        );
    }

    #[test]
    #[should_panic(expected = "CallFlavor::Pure has no production producer")]
    fn assemble_residual_call_irf_f_supports_pure_float_flavor() {
        let mut ssarepr = SSARepr::new("residual_call_irf_f");
        ssarepr.insns.push(Insn::op_with_result(
            "residual_call_irf_f",
            vec![
                Operand::ConstInt(7),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Int,
                    vec![Operand::Register(Register::new(Kind::Int, 0))],
                )),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Ref,
                    vec![Operand::Register(Register::new(Kind::Ref, 0))],
                )),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Float,
                    vec![Operand::Register(Register::new(Kind::Float, 0))],
                )),
                Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
                    effect_info: super::super::flatten::effect_info_for_call_flavor(
                        CallFlavor::Pure,
                    ),
                    arg_kinds: vec![Kind::Int, Kind::Ref, Kind::Float],
                })),
            ],
            Register::new(Kind::Float, 1),
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 1,
                ref_: 1,
                float: 2,
            }),
        );

        // majit jitcode/mod.rs: `BC_CALL_PURE_FLOAT = 35`.
        assert_eq!(jitcode.code[0], 35);
    }

    #[test]
    fn assemble_residual_call_ir_v_supports_release_gil_void_flavor() {
        let mut ssarepr = SSARepr::new("residual_call_ir_v");
        ssarepr.insns.push(Insn::op(
            "residual_call_ir_v",
            vec![
                Operand::ConstInt(7),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Int,
                    vec![Operand::Register(Register::new(Kind::Int, 0))],
                )),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Ref,
                    vec![Operand::Register(Register::new(Kind::Ref, 0))],
                )),
                Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
                    effect_info: super::super::flatten::effect_info_for_call_flavor(
                        CallFlavor::ReleaseGil,
                    ),
                    arg_kinds: vec![Kind::Int, Kind::Ref],
                })),
            ],
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 1,
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        // majit jitcode/mod.rs: `BC_CALL_RELEASE_GIL_VOID = 45`.
        assert_eq!(jitcode.code[0], 45);
    }

    /// `assembler.py:197-206` parity: a `Descr` operand attached to an op
    /// that does not consume a descr in its emitted bytecode MUST NOT
    /// leak into any descr pool. Registration happens inline at 'd'
    /// argcode emission; pyre's runtime codewriter has no
    /// descr-consuming ops today so every `assemble()` is a no-op for
    /// descr registration. This test keeps the guarantee explicit so a
    /// future descr-consumer addition is forced to ship the
    /// corresponding registration onto `BlackholeInterpBuilder.descrs`
    /// (`blackhole.py:102-103 setup_descrs`), not per-jitcode.
    #[test]
    fn assemble_does_not_register_unconsumed_descr() {
        let mut switch = SwitchDictDescr::new();
        switch.labels.push((4, TLabel::new("L2")));

        let mut ssarepr = SSARepr::new("switch");
        ssarepr.insns.push(Insn::Label(Label::new("L2")));
        // `abort_permanent` does not consume a descr in its bytecode (it
        // emits a single BC_ABORT_PERMANENT byte), so the attached
        // Descr operand is ignored by the assembler.
        ssarepr.insns.push(Insn::op(
            "abort_permanent",
            vec![Operand::descr(DescrOperand::SwitchDict(switch))],
        ));

        // Assemble succeeds without panic — the assertion is that no
        // code path tries to allocate a per-jitcode descr slot.
        let _jitcode = assemble(&mut ssarepr, JitCodeBuilder::default(), None);
    }

    // TODO: once a descr-consuming op (e.g. `switch`, `getfield_gc_d`) is
    // ported into `dispatch_op`, add a positive test that confirms its
    // descr lands on `BlackholeInterpBuilder.descrs` and that
    // `SwitchDictDescr._labels` → `BhDescr::Switch.dict` round-trips via
    // the shared pool at `fix_labels()` time (blackhole.py:102-103).

    /// `assembler.py:208-209` parity:
    /// ```python
    /// elif isinstance(x, IndirectCallTargets):
    ///     self.indirectcalltargets.update(x.lst)
    /// ```
    /// An `Operand::IndirectCallTargets(...)` attached to any op must be
    /// folded into `self.indirectcalltargets` with identity-dedup (RPython
    /// uses a Python `set` which dedupes by object identity, not value).
    fn make_test_jitcode(fnaddr: usize) -> Arc<JitCode> {
        let mut jc = JitCodeBuilder::default().finish();
        jc.fnaddr = fnaddr as i64;
        Arc::new(jc)
    }

    #[test]
    fn assemble_collects_indirectcalltargets_from_ops() {
        use super::super::flatten::IndirectCallTargets;

        let jc_a = make_test_jitcode(0x1000);
        let jc_b = make_test_jitcode(0x2000);

        let mut ssarepr = SSARepr::new("indirect");
        // `abort_permanent` is a zero-descr op in pyre's dispatch table, so
        // it tolerates the attached `IndirectCallTargets` operand without
        // needing a live dispatch arm.  The parity-relevant behaviour —
        // `write_insn` scanning the arg list and updating
        // `self.indirectcalltargets` — does not depend on the op kind.
        ssarepr.insns.push(Insn::op(
            "abort_permanent",
            vec![Operand::IndirectCallTargets(IndirectCallTargets {
                lst: vec![jc_a.clone(), jc_b.clone(), jc_a.clone()],
            })],
        ));

        let mut assembler = Assembler::new();
        assembler.assemble(&mut ssarepr, JitCodeBuilder::default(), None);

        // Identity-dedup: jc_a appears twice in the list, but the set keeps
        // a single entry (Python `set.update` on JitCode identity).
        assert_eq!(assembler.indirectcalltargets.len(), 2);
        let vec = assembler.indirectcalltargets_vec();
        assert_eq!(vec.len(), 2);
        assert!(vec.iter().any(|a| Arc::ptr_eq(a, &jc_a)));
        assert!(vec.iter().any(|a| Arc::ptr_eq(a, &jc_b)));
    }

    #[test]
    fn assemble_accumulates_indirectcalltargets_across_ops() {
        use super::super::flatten::IndirectCallTargets;

        let jc_a = make_test_jitcode(0x1000);
        let jc_b = make_test_jitcode(0x2000);
        let jc_c = make_test_jitcode(0x3000);

        let mut assembler = Assembler::new();

        // First assemble() registers jc_a and jc_b.
        let mut r1 = SSARepr::new("first");
        r1.insns.push(Insn::op(
            "abort_permanent",
            vec![Operand::IndirectCallTargets(IndirectCallTargets {
                lst: vec![jc_a.clone(), jc_b.clone()],
            })],
        ));
        assembler.assemble(&mut r1, JitCodeBuilder::default(), None);

        // Second assemble() registers jc_b (already present) and jc_c.
        let mut r2 = SSARepr::new("second");
        r2.insns.push(Insn::op(
            "abort_permanent",
            vec![Operand::IndirectCallTargets(IndirectCallTargets {
                lst: vec![jc_b.clone(), jc_c.clone()],
            })],
        ));
        assembler.assemble(&mut r2, JitCodeBuilder::default(), None);

        // `assembler.py:24` persists `indirectcalltargets` across every
        // `assemble()` call — the CodeWriter holds a single Assembler and
        // the set accumulates for the whole build (codewriter.py:19-22).
        assert_eq!(assembler.indirectcalltargets.len(), 3);
    }
}
