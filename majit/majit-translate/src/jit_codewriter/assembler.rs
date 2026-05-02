//! Assembler — converts flattened SSARepr into a `JitCode`.
//!
//! RPython equivalent: `rpython/jit/codewriter/assembler.py` class
//! `Assembler`. The `JitCode` struct itself lives in `crate::jitcode`
//! (RPython parity: `rpython/jit/codewriter/jitcode.py`).
//!
//! **Status: partial port.** `write_insn`, `fix_labels`, the
//! `all_liveness` shared table, and the `IndirectCallTargets` sidecar
//! merge for `residual_call` are implemented in the pyre-relevant
//! subset. Descriptor operands are deduplicated through the RPython
//! `_descr_dict` shape before bytecode emission.

use std::collections::HashMap;

use crate::call::CallControl;
use crate::flatten::{FlatOp, IntOvfOp, Label, RegKind, SSARepr};
use crate::flowspace::model::ConstValue;
use crate::jitcode::{BhCallDescr, JitCodeBody};
use crate::model::{LinkArg, ValueId};
use crate::regalloc::RegAllocResult;

/// Assembler — converts SSARepr to JitCode.
///
/// RPython: `assembler.py::Assembler`.
///
/// The assembler maintains state across multiple JitCode assemblies
/// (shared descriptor table, liveness encoding, etc.)
pub struct Assembler {
    /// RPython: Assembler.insns — map {opcode_key: opcode_number}
    insns: HashMap<String, u8>,
    /// RPython: Assembler.descrs — list of descriptors. Inline-call
    /// descriptors keep the callee JitCode object until the final
    /// snapshot, where `jitcode.index` is guaranteed to be assigned.
    descrs: Vec<AssemblerDescr>,
    /// RPython: Assembler._descr_dict — descriptor to descrs[] index.
    ///
    /// This is one of the few legitimate HashMaps in the port: upstream
    /// assembler.py keeps a real dict at lines 26 and 197-203 to deduplicate
    /// AbstractDescr objects before emitting the two-byte 'd' operand.
    descr_dict: HashMap<AssemblerDescrKey, usize>,
    /// RPython: `Assembler.indirectcalltargets` — merged `IndirectCallTargets`
    /// sidecars from every `residual_call` emitted during assembly
    /// (`assembler.py:208-209`).  RPython stores `JitCode` objects; we
    /// store their jitcode indices because codewriter owns the
    /// jitcode-index allocator.
    /// RPython `assembler.py:209` `self.indirectcalltargets.update(x.lst)`:
    /// a `set` of JitCode objects (Python identity dedup). pyre uses
    /// `JitCodeHandle` as the identity-keyed wrapper around
    /// `Arc<JitCode>` so the same shells handed out by
    /// `CallControl::get_jitcode` survive into the metainterp side
    /// without copying.
    pub indirectcalltargets: std::collections::HashSet<crate::jitcode::JitCodeHandle>,
    /// RPython: Assembler.list_of_addr2name — (addr, name) pairs for debugging.
    /// In majit: (target_path, name) string pairs since we don't have raw addresses.
    pub list_of_addr2name: Vec<(String, String)>,
    /// RPython: Assembler._count_jitcodes
    count_jitcodes: usize,
    /// RPython: Assembler._seen_raw_objects — dedup set for see_raw_object.
    seen_raw_objects: std::collections::HashSet<String>,
    /// RPython: Assembler.all_liveness — shared liveness table.
    /// Encoded as bytes: [count_i, count_r, count_f, reg_indices...].
    /// Deduplicated across all JitCodes via all_liveness_positions.
    all_liveness: Vec<u8>,
    /// RPython: Assembler.all_liveness_length (assembler.py:30).
    pub all_liveness_length: usize,
    /// RPython: Assembler.all_liveness_positions — dedup cache.
    /// Maps (live_i set, live_r set, live_f set) → offset in all_liveness.
    all_liveness_positions: HashMap<(Vec<u8>, Vec<u8>, Vec<u8>), usize>,
    /// RPython: Assembler.num_liveness_ops (assembler.py:32).
    pub num_liveness_ops: usize,
    /// Name of the graph currently being assembled, threaded through so
    /// diagnostic panics (e.g. missing regalloc coloring) can cite the
    /// exact function.  RPython tracks this via `self.jitcode.name`
    /// captured at `assembler.py:56 self.setup(ssarepr.name)`.
    current_graph_name: Option<String>,
    /// Pretty-printed FlatOp currently being encoded, only used by
    /// the `MAJIT_COVERAGE_PANIC=1` diagnostic so the missing-ValueId
    /// panic can cite the offending op.
    current_flatop_debug: Option<String>,
    /// Phase C v2 Slice C-3: authoritative ValueId → RegKind table for
    /// the SSARepr currently being assembled.  Set at `assemble`
    /// entry from `ssarepr.value_kinds` and consulted by
    /// `lookup_coloring`, which previously fell back to "search all
    /// three regalloc classes" and silently returned whichever class
    /// happened to hold a coloring for the ValueId.  RPython has no
    /// analogue because `Variable.concretetype` (set once at
    /// `flowmodel.py:Variable.__init__`) is the single source of
    /// truth, and `regalloc.py` only produces a coloring in the kind
    /// matching `getkind(v.concretetype)`.  Pyre's authoritative
    /// table now plays the same role: the kind decided at build time
    /// must match the regalloc class that holds the coloring.
    current_value_kinds: Option<HashMap<ValueId, RegKind>>,
}

impl Assembler {
    /// RPython: `Assembler.__init__()` (assembler.py:21-32).
    pub fn new() -> Self {
        Self {
            insns: HashMap::new(),
            descrs: Vec::new(),
            descr_dict: HashMap::new(),
            indirectcalltargets: std::collections::HashSet::new(),
            list_of_addr2name: Vec::new(),
            count_jitcodes: 0,
            seen_raw_objects: std::collections::HashSet::new(),
            all_liveness: Vec::new(),
            all_liveness_length: 0,
            all_liveness_positions: HashMap::new(),
            num_liveness_ops: 0,
            current_graph_name: None,
            current_flatop_debug: None,
            current_value_kinds: None,
        }
    }

    /// RPython: `Assembler.assemble` descriptor operand path
    /// (`assembler.py:197-207`).
    ///
    /// A descriptor is inserted into `descrs` only once and every later bytecode
    /// operand reuses the same two-byte index from `_descr_dict`.
    fn emit_descr(&mut self, descr: AssemblerDescr) -> usize {
        let key = AssemblerDescrKey::from_descr(&descr);
        if let Some(index) = self.descr_dict.get(&key) {
            return *index;
        }
        let index = self.descrs.len();
        assert!(index <= 0xFFFF, "too many AbstractDescrs!");
        self.descrs.push(descr);
        self.descr_dict.insert(key, index);
        index
    }

    fn emit_ready_descr(&mut self, descr: crate::jitcode::BhDescr) -> usize {
        self.emit_descr(AssemblerDescr::Ready(descr))
    }

    fn emit_pending_jitcode_descr(&mut self, jitcode: crate::jitcode::JitCodeHandle) -> usize {
        self.emit_descr(AssemblerDescr::PendingJitCode { jitcode })
    }

    fn emit_pending_switch_descr(&mut self, cases: Vec<(i64, Label)>) -> usize {
        let index = self.descrs.len();
        assert!(index <= 0xFFFF, "too many AbstractDescrs!");
        // RPython creates a fresh SwitchDictDescr per switch site; do
        // not route through `_descr_dict`, because labels are local to
        // the currently assembled JitCode.
        self.descrs.push(AssemblerDescr::PendingSwitch { cases });
        index
    }

    /// RPython: `Assembler.assemble(ssarepr, jitcode, num_regs)`.
    ///
    /// Takes the SSARepr (flattened instruction sequence) and register
    /// allocation results, and produces a JitCode with encoded bytecode,
    /// constant pools, and register counts.
    ///
    /// RPython assembler.py:34-54.
    /// RPython: `Assembler.assemble(ssarepr, jitcode, num_regs)`.
    ///
    /// RPython codewriter.py:53-56:
    ///   ssarepr = flatten_graph(graph, regallocs)
    ///   compute_liveness(ssarepr)          ← step 3b
    ///   self.assembler.assemble(ssarepr)   ← step 4
    pub fn assemble(
        &mut self,
        ssarepr: &mut SSARepr,
        regallocs: &HashMap<RegKind, RegAllocResult>,
    ) -> JitCodeBody {
        self.assemble_with_callcontrol(ssarepr, regallocs, None)
    }

    pub fn assemble_with_callcontrol(
        &mut self,
        ssarepr: &mut SSARepr,
        regallocs: &HashMap<RegKind, RegAllocResult>,
        callcontrol: Option<&CallControl>,
    ) -> JitCodeBody {
        // RPython codewriter.py:56: compute_liveness(ssarepr)
        // Must run BEFORE assembly so -live- markers carry the full
        // set of alive registers.
        crate::liveness::compute_liveness(ssarepr);
        self.current_graph_name = Some(ssarepr.name.clone());
        // Phase C v2 Slice C-3: snapshot the authoritative ValueId →
        // RegKind table for `lookup_coloring`.  `flatten_with_types`
        // populates `ssarepr.value_kinds` from the resolved type
        // state and augments the canonical exception-block
        // inputargs.  Cloning here keeps lookup_coloring's signature
        // unchanged so the existing `regallocs` parameter still
        // dominates write_insn's read path.
        self.current_value_kinds = Some(ssarepr.value_kinds.clone());

        // Pyre-only diagnostic: under `MAJIT_COVERAGE_AUDIT=1` enumerate
        // every ValueId referenced in `ssarepr.insns` that has no
        // regalloc coloring in any class.  Complements the
        // `MAJIT_COVERAGE_PANIC=1` path (which panics at the first gap
        // hit during `write_insn`) by surfacing the full per-graph gap
        // catalogue in one build.  Upstream RPython has no analogue —
        // the invariant is guaranteed by rtyper's `concretetype`
        // annotation so the lookup cannot miss.
        if std::env::var("MAJIT_COVERAGE_AUDIT").is_ok() {
            self.run_coverage_audit(ssarepr, regallocs);
        }

        let num_regs_i = regallocs.get(&RegKind::Int).map_or(0, |r| r.num_regs);
        let num_regs_r = regallocs.get(&RegKind::Ref).map_or(0, |r| r.num_regs);
        let num_regs_f = regallocs.get(&RegKind::Float).map_or(0, |r| r.num_regs);

        // RPython assembler.py:56-70: self.setup(ssarepr.name)
        let mut state = AssemblyState {
            code: Vec::new(),
            constants_i: Vec::new(),
            constants_r: Vec::new(),
            constants_f: Vec::new(),
            num_regs_i,
            num_regs_r,
            num_regs_f,
            label_positions: HashMap::new(),
            tlabel_fixups: Vec::new(),
            startpoints: std::collections::HashSet::new(),
            alllabels: std::collections::HashSet::new(),
            resulttypes: HashMap::new(),
        };

        // RPython assembler.py:41-44:
        //     ssarepr._insns_pos = []
        //     for insn in ssarepr.insns:
        //         ssarepr._insns_pos.append(len(self.code))
        //         self.write_insn(insn)
        let mut insns_pos = Vec::with_capacity(ssarepr.insns.len());
        // Borrow split: clone the insn vec so we can mutate ssarepr
        // (insns_pos write) without aliasing the borrow used by the
        // write_insn loop.
        let ops = ssarepr.insns.clone();
        let debug_enabled = std::env::var("MAJIT_COVERAGE_PANIC").is_ok();
        for op in &ops {
            insns_pos.push(state.code.len());
            if debug_enabled {
                self.current_flatop_debug = Some(format!("{op:?}"));
            }
            self.write_insn(op, regallocs, &mut state, callcontrol);
        }
        self.current_flatop_debug = None;
        ssarepr.insns_pos = Some(insns_pos);

        // RPython assembler.py:45,250-258: self.fix_labels()
        // Upstream `target = self.label_positions[name]` raises KeyError
        // when the label is missing — never writes a silent 0 target.
        for (label, fixup_pos) in &state.tlabel_fixups {
            // RPython `assembler.py:254 target = self.label_positions[insn[1].name]`
            // — direct dict access, raises KeyError when the TLabel
            // references a label that was never defined. Mirror with
            // a fail-loud panic instead of silently writing 0.
            let target = *state
                .label_positions
                .get(label)
                .unwrap_or_else(|| panic!("undefined TLabel {label:?} at fixup {fixup_pos}"));
            let target_u16 = target as u16;
            // RPython `assembler.py:255 assert 0 <= target <= 0xFFFF`.
            assert!(target <= 0xFFFF, "label target {target} exceeds u16 range");
            // RPython `assembler.py:252-253 assert self.code[pos] == "temp 1"`
            // — the fixup must point to two reserved placeholder
            // bytes still in range.
            assert!(
                fixup_pos + 1 < state.code.len(),
                "tlabel fixup position {fixup_pos} past end of code (len={})",
                state.code.len(),
            );
            state.code[*fixup_pos] = (target_u16 & 0xFF) as u8;
            state.code[*fixup_pos + 1] = (target_u16 >> 8) as u8;
        }
        for descr in &mut self.descrs {
            let AssemblerDescr::PendingSwitch { cases } = descr else {
                continue;
            };
            let dict = cases
                .iter()
                .map(|(key, label)| {
                    // RPython `assembler.py:261 target =
                    // self.label_positions[switchlabel.name]` — KeyError
                    // for a missing switch case label. Same fail-loud
                    // policy as the TLabel fixup loop above.
                    let target = *state.label_positions.get(label).unwrap_or_else(|| {
                        panic!("undefined SwitchDictDescr label {label:?} for key {key}")
                    });
                    (*key, target)
                })
                .collect();
            *descr = AssemblerDescr::Ready(crate::jitcode::BhDescr::Switch { dict });
        }

        // RPython assembler.py:271-281: jitcode.setup(code, ...)
        // Build the body that the codewriter will commit into the
        // pre-allocated `Arc<JitCode>` shell via `set_body`.
        // RPython jitcode.py:36 `assert num_regs_i < 256 and ...`. The
        // assembler limits register pressure via the same invariant.
        assert!(
            num_regs_i < 256 && num_regs_r < 256 && num_regs_f < 256,
            "too many registers (i={num_regs_i} r={num_regs_r} f={num_regs_f})"
        );
        // RPython assembler.py:49 `jitcode._ssarepr = ssarepr`
        let body = JitCodeBody {
            calldescr: BhCallDescr::default(),
            code: state.code,
            constants_i: state.constants_i,
            constants_r: state.constants_r,
            constants_f: state.constants_f,
            c_num_regs_i: num_regs_i as u16,
            c_num_regs_r: num_regs_r as u16,
            c_num_regs_f: num_regs_f as u16,
            // RPython `assembler.py:271-281 make_jitcode(startpoints=
            // self.startpoints, alllabels=self.alllabels, ...)` — the
            // assembler always hands its set, never `None`. Wrap in
            // `Some(...)` so the assembled jitcode parities the upstream
            // shape (helper jitcodes that bypass the assembler keep the
            // `None` default).
            startpoints: Some(state.startpoints),
            alllabels: Some(state.alllabels),
            resulttypes: Some(state.resulttypes),
            _ssarepr: Some(ssarepr.clone()),
        };

        self.count_jitcodes += 1;
        // Phase C v2 Slice C-3: drop the per-graph value_kinds snapshot.
        // A subsequent assemble call will repopulate it from its own
        // SSARepr, so leaking the previous graph's table here would let a
        // stale entry mask a missing kind.
        self.current_value_kinds = None;
        body
    }

    /// RPython: `Assembler.write_insn(insn)` — assembler.py:140-223.
    ///
    /// Encodes a single FlatOp into the bytecode stream. Each instruction
    /// is encoded as: opcode_byte + argument_bytes. The opcode byte is
    /// looked up from `self.insns` using a key of the form
    /// `opname/argcodes` (RPython assembler.py:220).
    fn write_insn(
        &mut self,
        op: &FlatOp,
        regallocs: &HashMap<RegKind, RegAllocResult>,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) {
        match op {
            // RPython assembler.py:143-144: Label → record bytecode position
            FlatOp::Label(label) => {
                state.label_positions.insert(*label, state.code.len());
            }

            // RPython assembler.py:146-158 `Register('-live-', ...)`
            // case in `write_insn`: emit the `live/` opcode followed
            // by the 2-byte offset returned from `_encode_liveness`.
            FlatOp::Live { live_values } => {
                self.num_liveness_ops += 1;
                let key = state.code.len();
                state.startpoints.insert(key);
                // assembler.py:151-156 `live_i, live_r, live_f` —
                // partition live values by kind via the regalloc result.
                let mut live_i = Vec::new();
                let mut live_r = Vec::new();
                let mut live_f = Vec::new();
                for &v in live_values {
                    let (color, kind) = self.lookup_coloring(v, regallocs);
                    match kind {
                        RegKind::Int => live_i.push(color),
                        RegKind::Ref => live_r.push(color),
                        RegKind::Float => live_f.push(color),
                    }
                }
                // assembler.py:236 `key = (frozenset(live_i), …)` — pyre
                // deduplicates with sorted `Vec<u8>` keys; sort each kind
                // so equal sets compare equal under the `Vec<u8>` Eq impl.
                live_i.sort();
                live_r.sort();
                live_f.sort();
                // assembler.py:148 `self.code.append(chr(self.insns['live/']))`
                let opnum = self.get_opnum("live/");
                state.code.push(opnum);
                // assembler.py:158 `self._encode_liveness(live_i, live_r, live_f)`
                // — appends the 2-byte offset into `state.code` after
                // registering or reusing the canonical entry.
                self._encode_liveness(&live_i, &live_r, &live_f, &mut state.code);
            }

            // RPython assembler.py:141-142: '---' → skip
            FlatOp::EndOfBlock => {}

            // RPython `flatten.py:292` `emitline("unreachable")` →
            // single-byte opcode for `bhimpl_unreachable`
            // (`blackhole.py:962-964`). Mirrors the
            // `assembler.py:140-159` general opcode path: a fresh
            // `startposition = len(self.code)` is recorded before the
            // opcode byte goes in so the `_check_no_branch_to_inside_an_op`
            // pass at `assembler.py:283` sees `unreachable/` as a valid
            // start address even though execution never reaches it.
            FlatOp::Unreachable => {
                let opnum = self.get_opnum("unreachable/");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
            }

            // RPython assembler.py:159-223: regular operation
            FlatOp::Op(inner_op) => {
                self.encode_op(inner_op, regallocs, state, callcontrol);
            }

            // RPython flatten.py: 'goto' + TLabel
            FlatOp::Jump(label) => {
                let opnum = self.get_opnum("goto/L");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                // RPython assembler.py:175-179: TLabel → record position + 2 placeholder bytes
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*label, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            FlatOp::CatchException { target } => {
                let opnum = self.get_opnum("catch_exception/L");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            FlatOp::GotoIfExceptionMismatch { llexitcase, target } => {
                // RPython `flatten.py:228-231`:
                //   emitline('goto_if_exception_mismatch',
                //            Constant(link.llexitcase,
                //                     lltype.typeOf(link.llexitcase)),
                //            TLabel(link))
                let opnum = self.get_opnum("goto_if_exception_mismatch/iL");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                let encoded_llexitcase = self.emit_llexitcase(llexitcase, state);
                state.code.push(encoded_llexitcase);
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            // RPython flatten.py:247-267: goto_if_not(cond, TLabel(false_path))
            // Only goto_if_not exists — no goto_if_true in RPython.
            FlatOp::GotoIfNot { cond, target } => {
                let (reg, _) = self.lookup_reg_with_kind(*cond, regallocs);
                let opnum = self.get_opnum("goto_if_not/iL");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
                // RPython assembler.py:175-179: TLabel operand position
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            FlatOp::Switch { value, targets } => {
                let (reg, kind) = self.lookup_reg_with_kind(*value, regallocs);
                assert_eq!(kind, 'i', "switch/id expects an int register");
                let descr_idx = self.emit_pending_switch_descr(targets.clone());
                let opnum = self.get_opnum("switch/id");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
            }

            FlatOp::IntBinOpJumpIfOvf {
                op,
                target,
                lhs,
                rhs,
                dst,
            } => {
                let opname = match op {
                    IntOvfOp::Add => "int_add_jump_if_ovf/Lii>i",
                    IntOvfOp::Sub => "int_sub_jump_if_ovf/Lii>i",
                    IntOvfOp::Mul => "int_mul_jump_if_ovf/Lii>i",
                };
                let opnum = self.get_opnum(opname);
                let (lhs_reg, lhs_kind) = self.lookup_reg_with_kind(*lhs, regallocs);
                let (rhs_reg, rhs_kind) = self.lookup_reg_with_kind(*rhs, regallocs);
                let (dst_reg, dst_kind) = self.lookup_reg_with_kind(*dst, regallocs);
                debug_assert_eq!(lhs_kind, 'i');
                debug_assert_eq!(rhs_kind, 'i');
                debug_assert_eq!(dst_kind, 'i');
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
                state.code.push(lhs_reg);
                state.code.push(rhs_reg);
                state.code.push(dst_reg);
                state.resulttypes.insert(state.code.len(), 'i');
            }

            // RPython flatten.py:333 `self.emitline('%s_copy' % kind,
            // v, "->", w)` — argcodes `i>i` (typed src, result marker,
            // typed dst). The `>` bears no byte in the stream; it only
            // flags the result position in the key so the blackhole
            // wire `int_copy/i>i` (blackhole.rs:5670) finds the handler
            // and `Assembler.resulttypes[pc]` is populated correctly.
            //
            // Upstream's source operand (`v`) can be either a `Register`
            // or a `Constant` (`getcolor` returns the Constant as-is at
            // flatten.py:382-384); in both cases the `assembler.py:164-174`
            // single-byte encoder shares the argcode kind letter and
            // disambiguates register vs constant at decode time via
            // `byte >= count_regs[kind]`.
            FlatOp::Move { dst, src } => {
                let (dst_reg, dst_kind) = self.lookup_reg_with_kind(*dst, regallocs);
                let (src_reg, src_kind) = self.encode_link_arg_source(src, regallocs, state);
                debug_assert_eq!(
                    src_kind, dst_kind,
                    "int/ref/float_copy src and dst must share kind"
                );
                let kind_name = match src_kind {
                    'r' => "ref",
                    'f' => "float",
                    _ => "int",
                };
                let key = format!("{kind_name}_copy/{src_kind}>{src_kind}");
                let opnum = self.get_opnum(&key);
                state.code.push(opnum);
                state.code.push(src_reg);
                state.code.push(dst_reg);
                // RPython `assembler.py:210-212`: when argcodes contain
                // `>`, record the reskind at the current pc. Mirrors
                // `encode_op`'s handling of any op with a result slot.
                state.resulttypes.insert(state.code.len(), src_kind);
            }

            // RPython `flatten.py:329` `self.emitline('%s_push' % kind, v)`.
            // Argcodes: one typed register, no result marker. Blackhole
            // wires under `{kind}_push/{kind}` (see blackhole.rs:5727-5729).
            // Only register-backed cycle breaks reach this path.
            FlatOp::Push(src) => {
                let (src_reg, src_kind) = self.lookup_reg_with_kind(*src, regallocs);
                let kind_name = match src_kind {
                    'r' => "ref",
                    'f' => "float",
                    _ => "int",
                };
                let key = format!("{kind_name}_push/{src_kind}");
                let opnum = self.get_opnum(&key);
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(src_reg);
                // No `>` in argcodes → no resulttypes entry (upstream
                // `assembler.py:210-212` guarded on `'>' in argcodes`).
            }

            // RPython `flatten.py:331` `self.emitline('%s_pop' % kind, "->", w)`.
            // Argcodes: `>{kind}` — result marker then one typed register.
            // Blackhole wires under `{kind}_pop/>{kind}` (see
            // blackhole.rs:5730-5732).
            FlatOp::Pop(dst) => {
                let (dst_reg, dst_kind) = self.lookup_reg_with_kind(*dst, regallocs);
                let kind_name = match dst_kind {
                    'r' => "ref",
                    'f' => "float",
                    _ => "int",
                };
                let key = format!("{kind_name}_pop/>{dst_kind}");
                let opnum = self.get_opnum(&key);
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(dst_reg);
                // `>` present in argcodes → record reskind per upstream
                // `assembler.py:210-212`.
                state.resulttypes.insert(state.code.len(), dst_kind);
            }

            FlatOp::LastException { dst } => {
                let (reg, kind) = self.lookup_reg_with_kind(*dst, regallocs);
                debug_assert_eq!(kind, 'i');
                let opnum = self.get_opnum("last_exception/>i");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
                state.resulttypes.insert(state.code.len(), 'i');
            }

            FlatOp::LastExcValue { dst } => {
                let (reg, kind) = self.lookup_reg_with_kind(*dst, regallocs);
                debug_assert_eq!(kind, 'r');
                let opnum = self.get_opnum("last_exc_value/>r");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
                state.resulttypes.insert(state.code.len(), 'r');
            }

            FlatOp::Reraise => {
                let opnum = self.get_opnum("reraise/");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
            }

            // RPython `flatten.py:131-138` `make_return`.  Blackhole
            // handlers: `blackhole.py:841-863 bhimpl_{int,ref,float,void}_return`.
            // `emit_const_*` returns a byte ≥ `num_regs_{kind}` so the
            // single-byte argcode `i`/`r`/`f` suffices for both register
            // and constant sources (upstream `assembler.py:164-174`).
            FlatOp::IntReturn(v) => {
                let (reg, kind) = self.encode_link_arg_source(v, regallocs, state);
                debug_assert_eq!(kind, 'i');
                let opnum = self.get_opnum("int_return/i");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
            }
            FlatOp::RefReturn(v) => {
                let (reg, kind) = self.encode_link_arg_source(v, regallocs, state);
                debug_assert_eq!(kind, 'r');
                let opnum = self.get_opnum("ref_return/r");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
            }
            FlatOp::FloatReturn(v) => {
                let (reg, kind) = self.encode_link_arg_source(v, regallocs, state);
                debug_assert_eq!(kind, 'f');
                let opnum = self.get_opnum("float_return/f");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
            }
            FlatOp::VoidReturn => {
                let opnum = self.get_opnum("void_return/");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
            }
            // RPython `flatten.py:139-143` `make_return` 2-inputarg case
            // plus the `flatten.py:166-173` overflow reraise.  Both paths
            // funnel through the single `raise/r` opname — upstream's
            // source operand can be either a Variable (the raised
            // exception value) or a Constant (e.g. the standard
            // OverflowError instance), and the single-byte argcode `r`
            // covers both.  Blackhole: `blackhole.py:1000 bhimpl_raise(excvalue)`.
            FlatOp::Raise(v) => {
                let (reg, kind) = self.encode_link_arg_source(v, regallocs, state);
                debug_assert_eq!(kind, 'r');
                let opnum = self.get_opnum("raise/r");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
            }
        }
    }

    /// RPython `assembler.py:234-248` `_encode_liveness(live_i, live_r,
    /// live_f)` — register a `(live_i, live_r, live_f)` triple in the
    /// shared `all_liveness` table (deduplicating against
    /// `all_liveness_positions`) and append the 2-byte offset of the
    /// canonical entry into `code`.
    ///
    /// Mirrors RPython `assembler.py:235`
    /// `key = (frozenset(live_i), frozenset(live_r), frozenset(live_f))`:
    /// the cache key is set-valued, so callers may pass arbitrary-order
    /// or duplicated slices.  Each kind's effective payload is the
    /// sorted, deduplicated set, exactly as `liveness.py:148` `live =
    /// sorted(live)` produces during inner encoding.
    ///
    /// On a cache miss we append three header bytes
    /// (`len(live_i)`, `len(live_r)`, `len(live_f)`) followed by
    /// `encode_liveness` of each kind, exactly mirroring upstream
    /// `assembler.py:241-247` byte order.  The returned offset is
    /// finally written via `liveness::encode_offset` (parity with
    /// `liveness.py:127-131`).
    pub fn _encode_liveness(
        &mut self,
        live_i: &[u8],
        live_r: &[u8],
        live_f: &[u8],
        code: &mut Vec<u8>,
    ) {
        // assembler.py:235 `key = (frozenset(live_i), frozenset(live_r),
        // frozenset(live_f))`.  Pyre normalises by sort+dedup so the
        // `Vec<u8>` cache key matches across equivalent sets.
        let normalize = |slice: &[u8]| -> Vec<u8> {
            let mut v = slice.to_vec();
            v.sort_unstable();
            v.dedup();
            v
        };
        let live_i = normalize(live_i);
        let live_r = normalize(live_r);
        let live_f = normalize(live_f);
        let key = (live_i.clone(), live_r.clone(), live_f.clone());
        let pos = if let Some(&cached) = self.all_liveness_positions.get(&key) {
            cached
        } else {
            let pos = self.all_liveness.len();
            self.all_liveness_positions.insert(key, pos);
            // assembler.py:241 `chr(len(live_i)) + chr(len(live_r)) + chr(len(live_f))`
            self.all_liveness.push(live_i.len() as u8);
            self.all_liveness.push(live_r.len() as u8);
            self.all_liveness.push(live_f.len() as u8);
            // assembler.py:243-247 `for live in live_i, live_r, live_f:
            // liveness = encode_liveness(live); …`
            for live in [live_i.as_slice(), live_r.as_slice(), live_f.as_slice()] {
                let encoded = crate::jit_codewriter::liveness::encode_liveness(live);
                self.all_liveness.extend_from_slice(&encoded);
            }
            self.all_liveness_length = self.all_liveness.len();
            pos
        };
        // assembler.py:248 `encode_offset(pos, self.code)`.
        crate::jit_codewriter::liveness::encode_offset(pos, code);
    }

    /// Encode a [`LinkArg`] source operand for `{kind}_copy` /
    /// `{kind}_push` / `{kind}_return` / `raise`.
    ///
    /// Mirrors RPython `assembler.py:164-174`: registers and constants
    /// share a single-byte argcode per kind, with constants landing at
    /// byte values `>= count_regs[kind]`.  Returns `(byte, kind_char)`.
    fn encode_link_arg_source(
        &mut self,
        arg: &LinkArg,
        regallocs: &HashMap<RegKind, RegAllocResult>,
        state: &mut AssemblyState,
    ) -> (u8, char) {
        match arg {
            LinkArg::Value(v) => self.lookup_reg_with_kind(*v, regallocs),
            LinkArg::Const(cv) => {
                let kind = crate::flatten::constvalue_kind(cv);
                let byte = self.emit_const(cv, kind, state);
                (byte, kind)
            }
        }
    }

    /// RPython assembler.py:159-223: encode one SpaceOperation.
    ///
    /// The encoding for each instruction is:
    /// [opcode_byte][arg1_byte][arg2_byte]...[->][result_byte]
    ///
    /// Where args are:
    /// - Register: 1 byte (index), argcode = kind char ('i','r','f')
    /// - Constant: 1 byte (pool index), argcode = kind char
    /// - TLabel: 2 bytes (u16 LE offset), argcode = 'L'
    /// - ListOfKind: 1 byte (len) + items, argcode = uppercase kind
    /// - Descr: 2 bytes (u16 LE index), argcode = 'd'
    fn encode_op(
        &mut self,
        op: &crate::model::SpaceOperation,
        regallocs: &HashMap<RegKind, RegAllocResult>,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) {
        use crate::model::OpKind;

        let startposition = state.code.len();
        state.code.push(0); // placeholder for opcode byte
        state.startpoints.insert(startposition);

        let mut argcodes = String::new();

        match &op.kind {
            // RPython flatten.py keeps inputargs on Block.inputargs and does
            // not serialize them as bytecode operations.
            OpKind::Input { .. } => {
                panic!("OpKind::Input must be eliminated before assembly");
            }
            // RPython: inline_call → [jitcode, I[...], R[...], F[...]]
            OpKind::InlineCall {
                jitcode,
                args_i,
                args_r,
                args_f,
                result_kind,
                ..
            } => {
                // RPython assembler.py:197-207: jitcode → descrs[index]
                // The JitCode object IS the descriptor for inline_call.
                let descr_idx = self.emit_pending_jitcode_descr(jitcode.clone());
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython jtransform.py:422-431: rewrite_call
                // Only emit the kind sublists that are in 'kinds'.
                let kinds = self.kinds_suffix(args_i, args_r, args_f, *result_kind);
                if kinds.contains('i') {
                    self.emit_list_of_kind(args_i, RegKind::Int, regallocs, state);
                    argcodes.push('I');
                }
                if kinds.contains('r') {
                    self.emit_list_of_kind(args_r, RegKind::Ref, regallocs, state);
                    argcodes.push('R');
                }
                if kinds.contains('f') {
                    self.emit_list_of_kind(args_f, RegKind::Float, regallocs, state);
                    argcodes.push('F');
                }
                // Result — see residual_call note below: derive the
                // key-level `reskind` from regalloc so `_r_i` / `_r_r`
                // match the actual `>X` argcode suffix.
                let result_key_kind = self.emit_call_result_arg(
                    op.result,
                    *result_kind,
                    regallocs,
                    state,
                    &mut argcodes,
                );
                // RPython jtransform.py:434: inline_call_{kinds}_{reskind}
                let key = format!("inline_call_{kinds}_{result_key_kind}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // RPython: recursive_call → [jd_index, G_I, G_R, G_F, R_I, R_R, R_F]
            //
            // `bhimpl_recursive_call_{i,r,f,v}` declares jd_index as
            // `@arguments("self", "i", ...)` (blackhole.py:1101-1132) so
            // the canonical argcode is `i` (register read). `emit_const_i`
            // returns a register-index into the int constant pool; the
            // dispatch side `bh.registers_i[code[p]]` reads the jd_index
            // back out. RPython does not include `recursive_call` in
            // `USE_C_FORM` (assembler.py:312), so the `c` short-const
            // form is not permitted here.
            OpKind::RecursiveCall {
                jd_index,
                greens_i,
                greens_r,
                greens_f,
                reds_i,
                reds_r,
                reds_f,
                result_kind,
            } => {
                let idx = self.emit_const_i(*jd_index as i64, state);
                state.code.push(idx);
                argcodes.push('i');
                // green lists
                self.emit_list_of_kind(greens_i, RegKind::Int, regallocs, state);
                argcodes.push('I');
                self.emit_list_of_kind(greens_r, RegKind::Ref, regallocs, state);
                argcodes.push('R');
                self.emit_list_of_kind(greens_f, RegKind::Float, regallocs, state);
                argcodes.push('F');
                // red lists
                self.emit_list_of_kind(reds_i, RegKind::Int, regallocs, state);
                argcodes.push('I');
                self.emit_list_of_kind(reds_r, RegKind::Ref, regallocs, state);
                argcodes.push('R');
                self.emit_list_of_kind(reds_f, RegKind::Float, regallocs, state);
                argcodes.push('F');
                let result_key_kind = self.emit_call_result_arg(
                    op.result,
                    *result_kind,
                    regallocs,
                    state,
                    &mut argcodes,
                );
                let key = format!("recursive_call_{result_key_kind}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // RPython: residual_call/call_may_force/call_elidable
            // → [funcptr, calldescr, I[...], R[...], F[...]]
            // RPython jtransform.py:414-435: rewrite_call splits args
            // by kind via make_three_lists.
            OpKind::CallResidual {
                funcptr,
                descriptor,
                args_i,
                args_r,
                args_f,
                result_kind,
                ..
            }
            | OpKind::CallMayForce {
                funcptr,
                descriptor,
                args_i,
                args_r,
                args_f,
                result_kind,
                ..
            }
            | OpKind::CallElidable {
                funcptr,
                descriptor,
                args_i,
                args_r,
                args_f,
                result_kind,
                ..
            } => {
                // RPython `assembler.py:208-209`: the sidecar
                // `IndirectCallTargets(lst)` on a `residual_call`
                // merges into the global `Assembler.indirectcalltargets`
                // set so the metainterp can later look up jitcodes by
                // funcptr address during runtime dispatch.  Only
                // `OpKind::CallResidual` carries the sidecar today.
                if let OpKind::CallResidual {
                    indirect_targets: Some(t),
                    ..
                } = &op.kind
                {
                    self.indirectcalltargets.extend(t.lst.iter().cloned());
                }
                let base = match &op.kind {
                    OpKind::CallMayForce { .. } => "call_may_force",
                    OpKind::CallElidable { .. } => "call_elidable",
                    _ => "residual_call",
                };
                // RPython `jtransform.py:422-431` `rewrite_call` emits args
                // by kind (I, R, F) first, then the calldescr, producing
                // keys like `residual_call_ir_r/iIRd>r`. jtransform now
                // materializes direct-call funcptrs as `ConstInt` values,
                // so every post-jtransform call op reaches the assembler
                // as `CallFuncPtr::Value(...)` and encodes the orthodox
                // leading `i` operand.
                match funcptr {
                    crate::model::CallFuncPtr::Value(vid) => {
                        let (reg, kc) = self.lookup_reg_with_kind(*vid, regallocs);
                        state.code.push(reg);
                        argcodes.push(kc);
                    }
                    crate::model::CallFuncPtr::Target(target) => {
                        panic!("call op reached assembler without materialized funcptr: {target}");
                    }
                }
                // RPython `assembler.py:197-203`: resolve the descriptor
                // through `_descr_dict` before writing the two bytes. The
                // bytes are still written AFTER the I/R/F lists to match
                // `jtransform.py:422-431` ordering: `iIRFd` / `iIRd` / `iRd`.
                let calldescr = descriptor.to_bh_calldescr();
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::Call { calldescr });
                // RPython jtransform.py:422-431: kind-separated sublists
                let kinds = self.kinds_suffix(args_i, args_r, args_f, *result_kind);
                if kinds.contains('i') {
                    self.emit_list_of_kind(args_i, RegKind::Int, regallocs, state);
                    argcodes.push('I');
                }
                if kinds.contains('r') {
                    self.emit_list_of_kind(args_r, RegKind::Ref, regallocs, state);
                    argcodes.push('R');
                }
                if kinds.contains('f') {
                    self.emit_list_of_kind(args_f, RegKind::Float, regallocs, state);
                    argcodes.push('F');
                }
                // RPython assembler.py:197-207: descriptor as 2-byte index,
                // emitted last per jtransform.py:422-431 ordering so the
                // blackhole key suffix is `...d>k`.
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // Result
                // RPython `residual_call_r_r` / `residual_call_r_i` /
                // `residual_call_r_v` are *different* bhimpls
                // (`blackhole.py:1225-1231`): the `_r` / `_i` / `_v`
                // suffix encodes the actual result kind. When pyre's
                // rtyper (`translate_legacy::rtyper::resolve_types`)
                // upgrades a call result's concrete type to `Signed`
                // (e.g. via `is_int_arith` backward constraint), the
                // regalloc-assigned register class diverges from the
                // op struct's original `result_kind`. Derive the key
                // name suffix from the regalloc-determined class so
                // `base_{kinds}_{reskind}` stays consistent with the
                // argcode `>X` suffix. If no result, fall back to `v`.
                let result_key_kind = self.emit_call_result_arg(
                    op.result,
                    *result_kind,
                    regallocs,
                    state,
                    &mut argcodes,
                );
                // RPython jtransform.py:434: {base}_{kinds}_{reskind}
                let key = format!("{base}_{kinds}_{result_key_kind}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // RPython `rpython/jit/codewriter/assembler.py:164-174`: ConstInt
            // is NOT a separate op — Constants appear as arguments to other
            // instructions via `emit_const` which returns a pool-region
            // register index (same byte shape as `emit_reg`). Pyre's model
            // forces constants through a standalone materialization op since
            // operands are always `ValueId`; lowering that limitation is
            // multi-session (requires op-level constant operands). Until
            // then, emit as `int_copy/i>i` — canonical register-to-register
            // move — since `emit_const_i` already returns a pool-region
            // register index (`num_regs_i + pool_pos`) and both src and dst
            // are int-kind registers. This eliminates the pyre-only
            // `const_int/c>i` opname and reuses the canonical
            // `bhimpl_int_copy` handler.
            OpKind::ConstInt(val) => {
                let idx = self.emit_const_i(*val, state);
                state.code.push(idx);
                argcodes.push('i');
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("int_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // Float-constant materialization mirrors `ConstInt`: the
            // bit pattern goes through `emit_const_f` (the same pool
            // every `'f'` constant uses) and the resulting pool-region
            // register index is moved into the SSA destination via
            // `float_copy`.
            OpKind::ConstFloat(bits) => {
                let const_value = crate::flowspace::model::ConstValue::Float(*bits);
                let idx = self.emit_const_f(&const_value, state);
                state.code.push(idx);
                argcodes.push('f');
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("float_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // Field/array operations: encode registers + descriptor.
            // RPython assembler.py:197-207: AbstractDescr → 2-byte index.
            // Field operations: register + descriptor.
            // RPython assembler.py:197-207: AbstractDescr → 2-byte index.
            OpKind::FieldRead {
                base,
                field,
                ty,
                pure,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let descr_idx = self.emit_ready_descr(fielddescrof(field, ty, callcontrol));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython `bhimpl_getfield_gc_{i,r,f}` canonical keys key
                // off the RESULT register's kind (`@arguments("cpu", "r",
                // "d", returns="X")`), not the declared field type —
                // declared field `ty` can be pyre-only Void/State/Unknown
                // while the SSA result register is always i/r/f after
                // regalloc. Using the result kind keeps the opname
                // aligned with the `>X` argcode the runtime dispatches on.
                let result_kind = if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                    kc
                } else {
                    'v'
                };
                let mut opname = format!("getfield_gc_{result_kind}");
                if *pure {
                    opname.push_str("_pure");
                }
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            // RPython `rpython/jit/codewriter/jtransform.py:546` emits
            // `int_guard_value(op.args[0])` where `op.args[0]` is already a
            // `Ptr(FuncType)` integer after rtype.  Rust `&dyn Trait` is a
            // fat pointer so the rtyper-equivalent layer
            // (`translator/rtyper/rclass.rs::class_get_method_ptr`) emits
            // `OpKind::VtableMethodPtr(receiver)` with the
            // `(trait_root, method_name)` pair; the assembler encodes it
            // as `vtable_method_ptr/rd>i` carrying a `BhDescr::VtableMethod`.
            // The result register is the integer function address that
            // `int_guard_value` and the subsequent `residual_call_*`
            // consume — backend lowering of the actual vtable slot read
            // is deferred (separate epic).
            OpKind::VtableMethodPtr {
                receiver,
                trait_root,
                method_name,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*receiver, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VtableMethod {
                    trait_root: trait_root.clone(),
                    method_name: method_name.clone(),
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            // RPython `rpython/jit/codewriter/jtransform.py:901-903` emits
            // `record_quasiimmut_field(v_inst, fielddescr, mutatefielddescr)`
            // — a register followed by two descrs.  The blackhole counterpart
            // `bhimpl_record_quasiimmut_field(struct, fielddescr,
            // mutatefielddescr)` (`rpython/jit/metainterp/blackhole.py:1537-1539`)
            // expects argcodes `rdd`.
            OpKind::RecordQuasiImmutField {
                base,
                field,
                mutate_field,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                for fd in [field, mutate_field] {
                    let descr_idx = self.emit_ready_descr(fielddescrof(
                        fd,
                        &crate::model::ValueType::Unknown,
                        callcontrol,
                    ));
                    state.code.push((descr_idx & 0xFF) as u8);
                    state.code.push((descr_idx >> 8) as u8);
                    argcodes.push('d');
                }
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::FieldWrite {
                base,
                value,
                field,
                ty,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, value_kind) = self.lookup_reg_with_kind(*value, regallocs);
                state.code.push(reg);
                argcodes.push(value_kind);
                let descr_idx = self.emit_ready_descr(fielddescrof(field, ty, callcontrol));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython `bhimpl_setfield_gc_{i,r,f}` canonical keys key
                // off the VALUE register's kind (`@arguments("cpu", "r",
                // "X", "d")`), not the declared field type — declared
                // field `ty` can be pyre-only Void/State/Unknown while
                // the SSA value register is always i/r/f after regalloc.
                let opname = format!("setfield_gc_{value_kind}");
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::ArrayRead {
                base,
                index,
                item_ty,
                array_type_id,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind(*index, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let descr_idx =
                    self.emit_ready_descr(arraydescrof(item_ty, array_type_id, callcontrol));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython `bhimpl_getarrayitem_gc_{i,r,f}` keys off the
                // result register's kind — same rationale as getfield_gc_*.
                let result_kind = if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                    kc
                } else {
                    'v'
                };
                let opname = format!("getarrayitem_gc_{result_kind}");
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::ArrayWrite {
                base,
                index,
                value,
                item_ty,
                array_type_id,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind(*index, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, value_kind) = self.lookup_reg_with_kind(*value, regallocs);
                state.code.push(reg);
                argcodes.push(value_kind);
                let descr_idx =
                    self.emit_ready_descr(arraydescrof(item_ty, array_type_id, callcontrol));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython `bhimpl_setarrayitem_gc_{i,r,f}` keys off the
                // value register's kind — same rationale as setfield_gc_*.
                let opname = format!("setarrayitem_gc_{value_kind}");
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            // Vable field/array: encode the base register followed by the
            // field_index descriptor, matching blackhole.py @arguments("r", "d").
            OpKind::VableFieldRead {
                base, field_index, ..
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                // RPython: vable field → VableField descriptor (index, not byte offset).
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VableField {
                    index: *field_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython `bhimpl_getfield_vable_{i,r,f}` canonical keys
                // (blackhole.py:1446-1458) match on the RESULT register
                // kind. See FieldRead above for the Void/State/Unknown
                // rationale — the pyre-only declared ty can be Void
                // while the SSA result register is always i/r/f.
                let result_kind = if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                    kc
                } else {
                    'v'
                };
                let opname = format!("getfield_vable_{result_kind}");
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableFieldWrite {
                base,
                field_index,
                value,
                ..
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, value_kind) = self.lookup_reg_with_kind(*value, regallocs);
                state.code.push(reg);
                argcodes.push(value_kind);
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VableField {
                    index: *field_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython `bhimpl_setfield_vable_{i,r,f}` canonical keys
                // (blackhole.py:1485-1495) match on the VALUE register's
                // kind. Same rationale as setfield_gc_*.
                let opname = format!("setfield_vable_{value_kind}");
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableArrayRead {
                base,
                array_index,
                elem_index,
                item_ty,
                array_itemsize,
                array_is_signed,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind(*elem_index, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                // RPython: two descriptors — fielddescr (vable array field) + arraydescr.
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VableArray {
                    index: *array_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // Second descriptor: arraydescr from VirtualizableInfo.array_descrs.
                let descr_idx2 = self.emit_ready_descr(vable_arraydescrof(
                    item_ty,
                    *array_itemsize,
                    *array_is_signed,
                ));
                state.code.push((descr_idx2 & 0xFF) as u8);
                state.code.push((descr_idx2 >> 8) as u8);
                argcodes.push('d');
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableArrayWrite {
                base,
                array_index,
                elem_index,
                value,
                item_ty,
                array_itemsize,
                array_is_signed,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind(*elem_index, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind(*value, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                // RPython: two descriptors — fielddescr (vable array field) + arraydescr.
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VableArray {
                    index: *array_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // Second descriptor: arraydescr from VirtualizableInfo.array_descrs.
                let descr_idx2 = self.emit_ready_descr(vable_arraydescrof(
                    item_ty,
                    *array_itemsize,
                    *array_is_signed,
                ));
                state.code.push((descr_idx2 & 0xFF) as u8);
                state.code.push((descr_idx2 >> 8) as u8);
                argcodes.push('d');
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableForce { base } => {
                let (reg, kc) = self.lookup_reg_with_kind(*base, regallocs);
                assert_eq!(kc, 'r', "hint_force_virtualizable expects a Ref base");
                state.code.push(reg);
                argcodes.push(kc);
                let opnum = self.get_opnum("hint_force_virtualizable/r");
                state.code[startposition] = opnum;
            }

            // RPython jtransform.py:1714-1718 handle_jit_marker__loop_header
            // emits `SpaceOperation('loop_header', [c_index], None)`; upstream
            // assembler.py encodes that Constant via `emit_const(allow_short=
            // False)` which registers it in `constants_i` and emits a single
            // byte register index (argcodes `i`). The bhimpl signature
            // (`blackhole.py:1062 @arguments("i")`) looks the byte up in
            // `registers_i`. The canonical runtime key is `loop_header/i`
            // (`majit-metainterp/src/jitcode/mod.rs:293`); `emit_const_i`
            // returns `num_regs_i + pool_idx` which the runtime resolves
            // back to the constant via `registers_i[byte]`. Emitting via
            // the generic fallback would push zero operand bytes (because
            // `op_value_refs(LoopHeader)` is empty), misaligning the
            // dispatch cursor.
            OpKind::LoopHeader { jitdriver_index } => {
                let reg_byte = self.emit_const_i(*jitdriver_index as i64, state);
                state.code.push(reg_byte);
                argcodes.push('i');
                let opnum = self.get_opnum("loop_header/i");
                state.code[startposition] = opnum;
            }

            // RPython jtransform.py:1690-1712 handle_jit_marker__jit_merge_point
            // emits `SpaceOperation('jit_merge_point',
            //   [Constant(jdindex), greens_i, greens_r, greens_f,
            //    reds_i, reds_r, reds_f], None)`. Upstream bhimpl signature
            // (`blackhole.py:1066 @arguments("self","i","I","R","F",
            // "I","R","F")`) reads jdindex + six typed register lists, each
            // encoded as `[len:u8][reg:u8 * N]` (assembler.py:181-196 ListOfKind).
            // assembler.py:163 reads `allow_short = (insn[0] in USE_C_FORM)`
            // and assembler.py:312 puts `jit_merge_point` in `USE_C_FORM`,
            // so a jdindex in the signed-byte range is emitted inline as
            // `'c'` (canonical key `jit_merge_point/cIRFIRF`); otherwise
            // the value is constants-pool-promoted and emitted as `'i'`
            // (canonical key `jit_merge_point/iIRFIRF`). Both keys are
            // registered in `wellknown_bh_insns()` mapping to distinct
            // opcodes (`BC_JIT_MERGE_POINT_C` vs `BC_JIT_MERGE_POINT`); the
            // blackhole `@arguments("i", ...)` decoder
            // (`blackhole.py:113-123`) interprets the byte per the runtime
            // argcode.
            OpKind::JitMergePoint {
                jitdriver_index,
                greens_i,
                greens_r,
                greens_f,
                reds_i,
                reds_r,
                reds_f,
            } => {
                let jdindex_value = *jitdriver_index as i64;
                let jdindex_argcode = if (-128..=127).contains(&jdindex_value) {
                    state.code.push(jdindex_value as u8);
                    'c'
                } else {
                    let jdindex_byte = self.emit_const_i(jdindex_value, state);
                    state.code.push(jdindex_byte);
                    'i'
                };
                self.emit_list_of_kind(greens_i, RegKind::Int, regallocs, state);
                self.emit_list_of_kind(greens_r, RegKind::Ref, regallocs, state);
                self.emit_list_of_kind(greens_f, RegKind::Float, regallocs, state);
                self.emit_list_of_kind(reds_i, RegKind::Int, regallocs, state);
                self.emit_list_of_kind(reds_r, RegKind::Ref, regallocs, state);
                self.emit_list_of_kind(reds_f, RegKind::Float, regallocs, state);
                argcodes.push(jdindex_argcode);
                argcodes.push('I');
                argcodes.push('R');
                argcodes.push('F');
                argcodes.push('I');
                argcodes.push('R');
                argcodes.push('F');
                let key = format!("jit_merge_point/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // Default: encode operand registers + result register (no descriptor)
            other => {
                for v in crate::inline::op_value_refs(other) {
                    let (reg, kind_char) = self.lookup_reg_with_kind(v, regallocs);
                    state.code.push(reg);
                    argcodes.push(kind_char);
                }
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kind_char) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kind_char);
                    state.code.push(reg);
                }
                let opname = op_kind_to_opname(other);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
        }

        // RPython assembler.py:217-219: record result type position.
        // If argcodes contains '>', the last char is the result kind,
        // and we record the current code length as the result position.
        if argcodes.contains('>') {
            if let Some(reskind) = argcodes.chars().last() {
                state.resulttypes.insert(state.code.len(), reskind);
            }
        }
    }

    /// Emit a ListOfKind: [u8 count][reg0][reg1]...
    /// RPython assembler.py:181-196.
    fn emit_list_of_kind(
        &self,
        args: &[ValueId],
        _kind: RegKind,
        regallocs: &HashMap<RegKind, RegAllocResult>,
        state: &mut AssemblyState,
    ) {
        state.code.push(args.len().min(255) as u8);
        for &v in args {
            let reg = self.lookup_reg(v, regallocs);
            state.code.push(reg);
        }
    }

    /// RPython `jtransform.py:424-426 rewrite_call`:
    /// ```text
    /// if lst_f or reskind == 'f': kinds = 'irf'
    /// elif lst_i or force_ir: kinds = 'ir'
    /// else: kinds = 'r'
    /// ```
    /// Result float forces `irf` even if no float args (`test_jtransform.py:356`
    /// `if RESTYPE == lltype.Float: with_f = True`). Without this rule a
    /// `&self -> f64` shape (empty `args_i`/`args_f`, single `args_r`,
    /// `result_kind='f'`) would map to a pyre-only `_r_f` handler that
    /// has no RPython `bhimpl_*_r_f` counterpart (`blackhole.py:1224,1278`
    /// only has `_r_{i,r,v}` / `_ir_*` / `_irf_*`).
    fn kinds_suffix(
        &self,
        args_i: &[ValueId],
        _args_r: &[ValueId],
        args_f: &[ValueId],
        result_kind: char,
    ) -> &'static str {
        if !args_f.is_empty() || result_kind == 'f' {
            "irf"
        } else if !args_i.is_empty() {
            "ir"
        } else {
            "r"
        }
    }

    /// RPython `flatten.py` emits the trailing `-> result` only for
    /// non-Void result variables. `jtransform.py:433` still computes
    /// `reskind = getkind(op.result.concretetype)[0]`, so a call op may
    /// carry a Void result placeholder while the bytecode opcode must be
    /// `*_v` with no `>` argcode.
    fn emit_call_result_arg(
        &self,
        result: Option<ValueId>,
        declared_result_kind: char,
        regallocs: &HashMap<RegKind, RegAllocResult>,
        state: &mut AssemblyState,
        argcodes: &mut String,
    ) -> char {
        if declared_result_kind == 'v' {
            return 'v';
        }
        let Some(result) = result else {
            return declared_result_kind;
        };
        argcodes.push('>');
        let (reg, kind_char) = self.lookup_reg_with_kind(result, regallocs);
        argcodes.push(kind_char);
        state.code.push(reg);
        kind_char
    }

    /// RPython: opcode key → opcode number.
    /// RPython assembler.py:220-222: key = opname + '/' + argcodes
    fn get_opnum(&mut self, key: &str) -> u8 {
        let next = self.insns.len() as u8;
        *self.insns.entry(key.to_string()).or_insert(next)
    }

    /// Resolve `(register_index, kind)` for a `ValueId`.
    ///
    /// RPython parity: every `Variable` reaching the assembler has a
    /// `concretetype` and therefore exactly one `(kind, color)` pair
    /// via `getkind()` + `regalloc.py` (`rpython/jit/codewriter/
    /// regalloc.py`).  Phase C v2 Slice C-3 promotes pyre to the same
    /// invariant: `current_value_kinds[v]` (set at `assemble` entry
    /// from `ssarepr.value_kinds`) is the authoritative kind, and the
    /// matching `regallocs[kind].coloring` is the only place we look
    /// for the color.  A miss in either side panics with the full
    /// per-class coverage so future drift surfaces at build time
    /// rather than getting silently masked by the previous "search
    /// all three classes" fallback.
    fn lookup_coloring(
        &self,
        v: ValueId,
        regallocs: &HashMap<RegKind, RegAllocResult>,
    ) -> (u8, RegKind) {
        let value_kinds = self.current_value_kinds.as_ref().unwrap_or_else(|| {
            panic!(
                "lookup_coloring: assemble entry must seed current_value_kinds \
                 before any write_insn lookup (graph={:?}, op={:?})",
                self.current_graph_name, self.current_flatop_debug,
            );
        });
        let class_coverage = || -> Vec<_> {
            [RegKind::Int, RegKind::Ref, RegKind::Float]
                .iter()
                .filter_map(|k| regallocs.get(k).map(|ra| (*k, ra)))
                .map(|(k, ra)| {
                    let min = ra.coloring.keys().map(|v| v.0).min();
                    let max = ra.coloring.keys().map(|v| v.0).max();
                    let contains_v = ra.coloring.contains_key(&v);
                    (k, ra.coloring.len(), min, max, contains_v)
                })
                .collect()
        };
        let expected = value_kinds.get(&v).copied().unwrap_or_else(|| {
            panic!(
                "lookup_coloring: value {v:?} has no value_kinds entry — \
                 type-state did not classify it before regalloc \
                 (graph={:?}, op={:?}, regalloc_coverage={:?})",
                self.current_graph_name,
                self.current_flatop_debug,
                class_coverage(),
            );
        });
        let ra = regallocs.get(&expected).unwrap_or_else(|| {
            panic!(
                "lookup_coloring: regallocs map is missing the expected kind {expected:?} \
                 entry for value {v:?} (graph={:?}, op={:?})",
                self.current_graph_name, self.current_flatop_debug,
            );
        });
        let color = ra.coloring.get(&v).copied().unwrap_or_else(|| {
            panic!(
                "lookup_coloring: value {v:?} declared as {expected:?} by value_kinds \
                 but has no coloring in regallocs[{expected:?}] — \
                 type-state ↔ regalloc kind mismatch \
                 (graph={:?}, op={:?}, regalloc_coverage={:?})",
                self.current_graph_name,
                self.current_flatop_debug,
                class_coverage(),
            );
        });
        (color as u8, expected)
    }

    /// Look up the register index (as u8) for a ValueId.
    fn lookup_reg(&self, v: ValueId, regallocs: &HashMap<RegKind, RegAllocResult>) -> u8 {
        self.lookup_coloring(v, regallocs).0
    }

    /// Look up register index AND kind character for a ValueId.
    /// Returns (register_index, kind_char) where kind_char ∈ {'i','r','f'}.
    fn lookup_reg_with_kind(
        &self,
        v: ValueId,
        regallocs: &HashMap<RegKind, RegAllocResult>,
    ) -> (u8, char) {
        let (color, kind) = self.lookup_coloring(v, regallocs);
        let kind_char = match kind {
            RegKind::Int => 'i',
            RegKind::Ref => 'r',
            RegKind::Float => 'f',
        };
        (color, kind_char)
    }

    /// Eagerly walk every `FlatOp` in `ssarepr.insns` and report every
    /// `ValueId` that lacks a regalloc coloring in any class.
    ///
    /// Pyre-only diagnostic — RPython's `assembler.py` never needs
    /// this because `rtyper` guarantees that every `Variable`'s
    /// `concretetype` produces exactly one `(kind, color)` via
    /// `getkind()` + `regalloc.py`.  Pyre's annotator / rtyper still
    /// has known coverage gaps (tracked as task #71 / #74), and the
    /// `lookup_reg_with_kind` fallback silently emits a `(0, 'r')`
    /// default at write time — which masks how many distinct gaps
    /// exist per graph.  `MAJIT_COVERAGE_PANIC=1` aborts at the first
    /// gap, losing the rest; this walker enumerates them all up
    /// front so the full gap catalogue surfaces in a single build.
    ///
    /// Output goes through `cargo:warning=` so the build script
    /// runner (`build.rs`) surfaces each line to the user.
    fn run_coverage_audit(&self, ssarepr: &SSARepr, regallocs: &HashMap<RegKind, RegAllocResult>) {
        // For each ValueId, track: has a def site (result of some op),
        // count of direct operand uses, count of Live markers mentioning
        // it.  Live-only gaps (no def, no operand use) point at backward
        // liveness leakage; uses-without-def at missing rtyper coverage;
        // def-without-coverage at regalloc class mismatch.
        #[derive(Default)]
        struct ValueSites {
            has_def: bool,
            use_count: usize,
            live_count: usize,
            first_use_tag: Option<&'static str>,
        }

        fn opkind_tag(kind: &crate::model::OpKind) -> &'static str {
            use crate::model::OpKind;
            match kind {
                OpKind::Input { .. } => "Input",
                OpKind::ConstInt(_) => "ConstInt",
                OpKind::ConstFloat(_) => "ConstFloat",
                OpKind::FieldRead { .. } => "FieldRead",
                OpKind::FieldWrite { .. } => "FieldWrite",
                OpKind::ArrayRead { .. } => "ArrayRead",
                OpKind::ArrayWrite { .. } => "ArrayWrite",
                OpKind::InteriorFieldRead { .. } => "InteriorFieldRead",
                OpKind::InteriorFieldWrite { .. } => "InteriorFieldWrite",
                OpKind::Call { .. } => "Call",
                OpKind::GuardTrue { .. } => "GuardTrue",
                OpKind::GuardFalse { .. } => "GuardFalse",
                OpKind::GuardValue { .. } => "GuardValue",
                OpKind::VtableMethodPtr { .. } => "VtableMethodPtr",
                OpKind::IndirectCall { .. } => "IndirectCall",
                OpKind::VableFieldRead { .. } => "VableFieldRead",
                OpKind::VableFieldWrite { .. } => "VableFieldWrite",
                OpKind::VableArrayRead { .. } => "VableArrayRead",
                OpKind::VableArrayWrite { .. } => "VableArrayWrite",
                OpKind::BinOp { .. } => "BinOp",
                OpKind::UnaryOp { .. } => "UnaryOp",
                OpKind::VableForce { .. } => "VableForce",
                OpKind::CallElidable { .. } => "CallElidable",
                OpKind::CallResidual { .. } => "CallResidual",
                OpKind::CallMayForce { .. } => "CallMayForce",
                OpKind::InlineCall { .. } => "InlineCall",
                OpKind::RecursiveCall { .. } => "RecursiveCall",
                OpKind::JitDebug { .. } => "JitDebug",
                OpKind::AssertGreen { .. } => "AssertGreen",
                OpKind::CurrentTraceLength => "CurrentTraceLength",
                OpKind::IsConstant { .. } => "IsConstant",
                OpKind::IsVirtual { .. } => "IsVirtual",
                OpKind::ConditionalCall { .. } => "ConditionalCall",
                OpKind::ConditionalCallValue { .. } => "ConditionalCallValue",
                OpKind::RecordKnownResult { .. } => "RecordKnownResult",
                OpKind::RecordQuasiImmutField { .. } => "RecordQuasiImmutField",
                OpKind::Live => "Live",
                OpKind::JitMergePoint { .. } => "JitMergePoint",
                OpKind::LoopHeader { .. } => "LoopHeader",
                OpKind::Abort { .. } => "Abort",
            }
        }
        let mut sites: std::collections::HashMap<ValueId, ValueSites> =
            std::collections::HashMap::new();

        for op in &ssarepr.insns {
            match op {
                FlatOp::Op(inner) => {
                    let tag = opkind_tag(&inner.kind);
                    if let Some(r) = inner.result {
                        sites.entry(r).or_default().has_def = true;
                    }
                    for v in crate::inline::op_value_refs(&inner.kind) {
                        let s = sites.entry(v).or_default();
                        s.use_count += 1;
                        s.first_use_tag.get_or_insert(tag);
                    }
                }
                FlatOp::GotoIfNot { cond, .. } => {
                    let s = sites.entry(*cond).or_default();
                    s.use_count += 1;
                    s.first_use_tag.get_or_insert("GotoIfNot");
                }
                FlatOp::Switch { value, .. } => {
                    let s = sites.entry(*value).or_default();
                    s.use_count += 1;
                    s.first_use_tag.get_or_insert("Switch");
                }
                FlatOp::IntBinOpJumpIfOvf { lhs, rhs, dst, .. } => {
                    sites.entry(*dst).or_default().has_def = true;
                    for v in [*lhs, *rhs] {
                        let s = sites.entry(v).or_default();
                        s.use_count += 1;
                        s.first_use_tag.get_or_insert("IntBinOpJumpIfOvf");
                    }
                }
                FlatOp::Move { dst, src } => {
                    sites.entry(*dst).or_default().has_def = true;
                    if let Some(v) = src.as_value() {
                        let s = sites.entry(v).or_default();
                        s.use_count += 1;
                        s.first_use_tag.get_or_insert("Move");
                    }
                }
                FlatOp::Push(v) => {
                    let s = sites.entry(*v).or_default();
                    s.use_count += 1;
                    s.first_use_tag.get_or_insert("Push");
                }
                FlatOp::Pop(v) => {
                    sites.entry(*v).or_default().has_def = true;
                }
                FlatOp::LastException { dst } | FlatOp::LastExcValue { dst } => {
                    sites.entry(*dst).or_default().has_def = true;
                }
                FlatOp::IntReturn(a)
                | FlatOp::RefReturn(a)
                | FlatOp::FloatReturn(a)
                | FlatOp::Raise(a) => {
                    if let Some(v) = a.as_value() {
                        let s = sites.entry(v).or_default();
                        s.use_count += 1;
                        s.first_use_tag.get_or_insert("Return/Raise");
                    }
                }
                FlatOp::Live { live_values } => {
                    for v in live_values {
                        sites.entry(*v).or_default().live_count += 1;
                    }
                }
                FlatOp::Label(_)
                | FlatOp::Jump(_)
                | FlatOp::VoidReturn
                | FlatOp::EndOfBlock
                | FlatOp::Unreachable
                | FlatOp::CatchException { .. }
                | FlatOp::GotoIfExceptionMismatch { .. }
                | FlatOp::Reraise => {}
            }
        }

        let mut gaps: Vec<(ValueId, &ValueSites)> = Vec::new();
        for (v, s) in &sites {
            let covered = [RegKind::Int, RegKind::Ref, RegKind::Float]
                .iter()
                .any(|k| {
                    regallocs
                        .get(k)
                        .is_some_and(|ra| ra.coloring.contains_key(v))
                });
            if !covered {
                gaps.push((*v, s));
            }
        }
        gaps.sort_by_key(|(v, _)| v.0);

        if gaps.is_empty() {
            return;
        }

        let class_coverage: Vec<(RegKind, usize)> = [RegKind::Int, RegKind::Ref, RegKind::Float]
            .iter()
            .filter_map(|k| regallocs.get(k).map(|ra| (*k, ra.coloring.len())))
            .collect();
        println!(
            "cargo:warning=[MAJIT_COVERAGE_AUDIT] graph={:?} gaps={} regalloc_coverage={:?}",
            ssarepr.name,
            gaps.len(),
            class_coverage,
        );
        for (v, s) in &gaps {
            let first_use = s.first_use_tag.unwrap_or("<no use>");
            println!(
                "cargo:warning=  - {v:?} def={} uses={} live={} first_use={}",
                s.has_def, s.use_count, s.live_count, first_use,
            );
        }
    }

    /// Look up just the kind for a ValueId.
    fn lookup_kind(&self, v: ValueId, regallocs: &HashMap<RegKind, RegAllocResult>) -> RegKind {
        self.lookup_coloring(v, regallocs).1
    }

    /// RPython assembler.py:80-138: emit_const for integer constants.
    /// Adds to constant pool and returns the index byte.
    fn emit_const(&mut self, value: &ConstValue, kind: char, state: &mut AssemblyState) -> u8 {
        match kind {
            'i' => self.emit_const_i_from_const(value, state),
            'r' => self.emit_const_r(value, state),
            'f' => self.emit_const_f(value, state),
            other => panic!("unknown constant kind {other:?} for {value:?}"),
        }
    }

    fn emit_const_i_from_const(&mut self, value: &ConstValue, state: &mut AssemblyState) -> u8 {
        let value = match value {
            ConstValue::Int(n) => *n,
            ConstValue::Bool(b) => *b as i64,
            ConstValue::SpecTag(tag) => *tag as i64,
            other => panic!("integer-kind constant not supported by emit_const_i: {other:?}"),
        };
        self.emit_const_i(value, state)
    }

    fn emit_const_i(&mut self, value: i64, state: &mut AssemblyState) -> u8 {
        // Check if already in pool
        for (i, &existing) in state.constants_i.iter().enumerate() {
            if existing == value {
                return (state.num_regs_i + i) as u8;
            }
        }
        // Add to pool: index = num_regs + pool_position
        state.constants_i.push(value);
        (state.num_regs_i + state.constants_i.len() - 1) as u8
    }

    fn emit_llexitcase(&mut self, value: &ConstValue, state: &mut AssemblyState) -> u8 {
        match value {
            ConstValue::Int(value) => self.emit_const_i(*value, state),
            ConstValue::HostObject(obj) => self.emit_const_i(obj.identity_id() as i64, state),
            other => {
                panic!("goto_if_exception_mismatch: unsupported llexitcase constant {other:?}")
            }
        }
    }

    fn emit_const_r(&mut self, value: &ConstValue, state: &mut AssemblyState) -> u8 {
        let bits = match value {
            ConstValue::HostObject(obj) => obj.identity_id() as i64,
            other => panic!("raise/r constant pool does not support {other:?}"),
        };
        if let Some(index) = state
            .constants_r
            .iter()
            .position(|&existing| existing == bits)
        {
            return (state.num_regs_r + index) as u8;
        }
        state.constants_r.push(bits);
        (state.num_regs_r + state.constants_r.len() - 1) as u8
    }

    fn emit_const_f(&mut self, value: &ConstValue, state: &mut AssemblyState) -> u8 {
        let bits = match value {
            ConstValue::Float(bits) => *bits as i64,
            other => panic!("float constant pool does not support {other:?}"),
        };
        if let Some(index) = state
            .constants_f
            .iter()
            .position(|&existing| existing == bits)
        {
            return (state.num_regs_f + index) as u8;
        }
        state.constants_f.push(bits);
        (state.num_regs_f + state.constants_f.len() - 1) as u8
    }
}

/// Per-assembly state (RPython: Assembler.setup() fields).
struct AssemblyState {
    code: Vec<u8>,
    constants_i: Vec<i64>,
    constants_r: Vec<i64>,
    constants_f: Vec<i64>,
    num_regs_i: usize,
    num_regs_r: usize,
    num_regs_f: usize,
    label_positions: HashMap<Label, usize>,
    tlabel_fixups: Vec<(Label, usize)>,
    startpoints: std::collections::HashSet<usize>,
    /// RPython assembler.py:176: positions in bytecode where TLabel operands
    /// are written. Used by JitCode.follow_jump() for verification.
    alllabels: std::collections::HashSet<usize>,
    /// RPython assembler.py:217-219: map from bytecode offset (after `->`)
    /// to result kind character. Recorded when encoding result registers.
    resulttypes: HashMap<usize, char>,
}

/// RPython: getkind(v.concretetype)[0] → 'i', 'r', 'f', 'v'.
fn value_type_to_kind(ty: &crate::model::ValueType) -> char {
    use crate::model::ValueType;
    match ty {
        ValueType::Int => 'i',
        ValueType::Ref => 'r',
        ValueType::Float => 'f',
        ValueType::Void | ValueType::State | ValueType::Unknown => 'v',
    }
}

fn value_type_to_itemsize(ty: &crate::model::ValueType) -> usize {
    use crate::model::ValueType;
    match ty {
        ValueType::Int => 8,
        ValueType::Ref => 8,
        ValueType::Float => 8,
        _ => 8,
    }
}

fn value_type_to_ir_type_for_descr(ty: &crate::model::ValueType) -> majit_ir::value::Type {
    match ty {
        crate::model::ValueType::Int => majit_ir::value::Type::Int,
        crate::model::ValueType::Float => majit_ir::value::Type::Float,
        crate::model::ValueType::Void => majit_ir::value::Type::Void,
        _ => majit_ir::value::Type::Ref,
    }
}

fn type_flag_from_str(
    type_str: &str,
) -> (majit_ir::descr::ArrayFlag, majit_ir::value::Type, usize) {
    use majit_ir::descr::ArrayFlag;
    match type_str {
        s if s.starts_with('&')
            || s.starts_with("Box<")
            || s.starts_with("Arc<")
            || s.starts_with("Rc<")
            || s.starts_with("Vec<")
            || s.starts_with("Option<")
            || s == "String" =>
        {
            (ArrayFlag::Pointer, majit_ir::value::Type::Ref, 8)
        }
        "f64" => (ArrayFlag::Float, majit_ir::value::Type::Float, 8),
        "f32" => (ArrayFlag::Float, majit_ir::value::Type::Float, 4),
        "i64" | "isize" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 8),
        "i32" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 4),
        "i16" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 2),
        "i8" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 1),
        "u64" | "usize" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 8),
        "u32" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 4),
        "u16" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 2),
        "u8" | "bool" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 1),
        "()" => (ArrayFlag::Void, majit_ir::value::Type::Void, 0),
        _ => (ArrayFlag::Pointer, majit_ir::value::Type::Ref, 8),
    }
}

fn fallback_field_layout(
    ty: &crate::model::ValueType,
) -> (
    usize,
    majit_ir::value::Type,
    majit_ir::descr::ArrayFlag,
    bool,
) {
    let field_type = value_type_to_ir_type_for_descr(ty);
    let field_size = value_type_to_itemsize(ty);
    let field_flag = majit_ir::descr::ArrayFlag::from_field_type(field_type);
    let is_signed = field_flag == majit_ir::descr::ArrayFlag::Signed;
    (field_size, field_type, field_flag, is_signed)
}

fn bh_field_name(owner: &str, field_name: &str) -> String {
    if owner.is_empty() || field_name.contains('.') {
        field_name.to_string()
    } else {
        format!("{owner}.{field_name}")
    }
}

fn bh_field_spec_from_parts(
    index: u32,
    owner: &str,
    field_name: &str,
    offset: usize,
    field_size: usize,
    field_type: majit_ir::value::Type,
    field_flag: majit_ir::descr::ArrayFlag,
    is_immutable: bool,
    is_quasi_immutable: bool,
    index_in_parent: usize,
) -> crate::jitcode::BhFieldSpec {
    crate::jitcode::BhFieldSpec {
        index,
        name: bh_field_name(owner, field_name),
        offset,
        field_size,
        field_type,
        field_flag,
        is_field_signed: field_flag == majit_ir::descr::ArrayFlag::Signed,
        is_immutable,
        is_quasi_immutable,
        index_in_parent,
    }
}

fn bh_size_spec_from_callcontrol(
    cc: &CallControl,
    owner: &str,
) -> Option<crate::jitcode::BhSizeSpec> {
    if owner.is_empty() {
        return None;
    }
    let size = cc
        .struct_layouts
        .get(owner)
        .map(|layout| layout.size)
        .or_else(|| heuristic_struct_size_for_bh(cc, owner))?;
    Some(crate::jitcode::BhSizeSpec {
        size,
        type_id: 0,
        vtable: 0,
        all_fielddescrs: bh_all_field_specs_for_struct(cc, owner),
    })
}

fn bh_all_field_specs_for_struct(
    cc: &CallControl,
    owner: &str,
) -> Vec<crate::jitcode::BhFieldSpec> {
    let mut specs = Vec::new();
    bh_all_field_specs_for_struct_into(cc, owner, &mut specs);
    specs
}

/// RPython `heaptracker.all_fielddescrs(STRUCT, res=...)` parity port:
/// recursively walks `STRUCT._names`, skipping `Void` / `typeptr` /
/// `c__pad`, and recursing into nested-struct fields so their leaf
/// fielddescrs land in the same flat `res` list with `index_in_parent`
/// matching `heaptracker.get_fielddescr_index_in()` (`heaptracker.py:51`).
///
/// Pyre keeps both a structured `struct_layouts` cache and a textual
/// `struct_field_entries` registry; the layout path doesn't carry the
/// nested-struct type name, so we cross-reference the entries registry
/// to recover the inner owner string before recursing.
fn bh_all_field_specs_for_struct_into(
    cc: &CallControl,
    owner: &str,
    specs: &mut Vec<crate::jitcode::BhFieldSpec>,
) {
    if let Some(layout) = cc.struct_layouts.get(owner) {
        // The textual entries registry carries the inner-struct type name
        // for nested fields; match by field name to recover the owner
        // string when recursing.  Cloned out of `cc` so the immutable
        // borrow does not collide with the recursive call below.
        let entries: Vec<(String, String)> = cc
            .struct_field_entries(owner)
            .map(|fs| fs.to_vec())
            .unwrap_or_default();
        for fl in &layout.fields {
            if fl.field_type == majit_ir::value::Type::Void
                || fl.name == "typeptr"
                || fl.name.starts_with("c__pad")
            {
                continue;
            }
            if fl.flag == majit_ir::descr::ArrayFlag::Struct {
                // `heaptracker.py:68-69 isinstance(FIELD, lltype.Struct):
                //  all_fielddescrs(gccache, FIELD, only_gc, res, get_field_descr)`.
                if let Some(inner_owner) = entries
                    .iter()
                    .find(|(name, _)| name == &fl.name)
                    .map(|(_, ty)| ty.as_str())
                {
                    bh_all_field_specs_for_struct_into(cc, inner_owner, specs);
                }
                continue;
            }
            let index_in_parent = specs.len();
            specs.push(bh_field_spec_from_parts(
                index_in_parent as u32,
                owner,
                &fl.name,
                fl.offset,
                fl.size,
                fl.field_type,
                fl.flag,
                fl.is_immutable(),
                fl.is_quasi_immutable(),
                index_in_parent,
            ));
        }
        return;
    }

    let Some(fields) = cc.struct_field_entries(owner).map(|fs| fs.to_vec()) else {
        return;
    };
    let mut offset = 0usize;
    for (field_name, field_type_str) in &fields {
        let (field_flag, field_type, field_size) = if cc.is_known_struct(field_type_str) {
            (
                majit_ir::descr::ArrayFlag::Struct,
                majit_ir::value::Type::Ref,
                cc.struct_layouts
                    .get(field_type_str.as_str())
                    .map(|layout| layout.size)
                    .unwrap_or(std::mem::size_of::<usize>()),
            )
        } else {
            type_flag_from_str(field_type_str)
        };
        if field_type == majit_ir::value::Type::Void || field_size == 0 {
            continue;
        }
        let align = field_size.min(std::mem::size_of::<usize>());
        offset = (offset + align - 1) & !(align - 1);
        let is_skipped_field = field_name == "typeptr" || field_name.starts_with("c__pad");
        if !is_skipped_field {
            if field_flag == majit_ir::descr::ArrayFlag::Struct {
                // `heaptracker.py:68-69` recursive flatten for nested
                // structs.  `field_type_str` is the inner owner name in
                // this textual path.
                bh_all_field_specs_for_struct_into(cc, field_type_str, specs);
            } else {
                let index_in_parent = specs.len();
                let rank = cc.field_immutability(Some(owner), field_name);
                specs.push(bh_field_spec_from_parts(
                    index_in_parent as u32,
                    owner,
                    field_name,
                    offset,
                    field_size,
                    field_type,
                    field_flag,
                    rank.is_some(),
                    rank.map(|r| r.is_quasi_immutable()).unwrap_or(false),
                    index_in_parent,
                ));
            }
        }
        offset += field_size;
    }
}

fn heuristic_struct_size_for_bh(cc: &CallControl, owner: &str) -> Option<usize> {
    let fields = cc.struct_field_entries(owner)?;
    let mut offset = 0usize;
    let mut max_align = 0usize;
    for (_, field_type_str) in fields {
        let field_size = if cc.is_known_struct(field_type_str) {
            cc.struct_layouts
                .get(field_type_str.as_str())
                .map(|layout| layout.size)
                .unwrap_or(std::mem::size_of::<usize>())
        } else {
            let (_, field_type, size) = type_flag_from_str(field_type_str);
            if field_type == majit_ir::value::Type::Void || size == 0 {
                continue;
            }
            size
        };
        let align = field_size.min(std::mem::size_of::<usize>());
        max_align = max_align.max(align);
        offset = (offset + align - 1) & !(align - 1);
        offset += field_size;
    }
    if offset == 0 {
        return Some(0);
    }
    let align = max_align.max(1);
    Some((offset + align - 1) & !(align - 1))
}

fn fielddescrof(
    field: &crate::model::FieldDescriptor,
    ty: &crate::model::ValueType,
    callcontrol: Option<&CallControl>,
) -> crate::jitcode::BhDescr {
    let (mut offset, mut field_size, mut field_type, mut field_flag, mut is_field_signed) = {
        let (field_size, field_type, field_flag, signed) = fallback_field_layout(ty);
        (0, field_size, field_type, field_flag, signed)
    };
    let mut is_immutable = false;
    let mut is_quasi_immutable = false;
    let mut index_in_parent = 0usize;
    let mut parent = None;

    if let (Some(cc), Some(owner)) = (callcontrol, field.owner_root.as_deref()) {
        parent = bh_size_spec_from_callcontrol(cc, owner);
        if let Some(parent_spec) = parent.as_ref() {
            let full_name = bh_field_name(owner, &field.name);
            if let Some(spec) = parent_spec
                .all_fielddescrs
                .iter()
                .find(|spec| spec.name == full_name)
            {
                offset = spec.offset;
                field_size = spec.field_size;
                field_type = spec.field_type;
                field_flag = spec.field_flag;
                is_field_signed = spec.is_field_signed;
                is_immutable = spec.is_immutable;
                is_quasi_immutable = spec.is_quasi_immutable;
                index_in_parent = spec.index_in_parent;
            }
        }
        if let Some(layout_field) = cc
            .struct_layouts
            .get(owner)
            .and_then(|layout| layout.fields.iter().find(|fl| fl.name == field.name))
        {
            offset = layout_field.offset;
            field_size = layout_field.size;
            field_type = layout_field.field_type;
            field_flag = layout_field.flag;
            is_field_signed = field_flag == majit_ir::descr::ArrayFlag::Signed;
            is_immutable = layout_field.is_immutable();
            is_quasi_immutable = layout_field.is_quasi_immutable();
        } else if let Some((
            computed_offset,
            computed_size,
            computed_type,
            computed_flag,
            computed_signed,
        )) = heuristic_field_layout(cc, owner, &field.name)
        {
            offset = computed_offset;
            field_size = computed_size;
            field_type = computed_type;
            field_flag = computed_flag;
            is_field_signed = computed_signed;
        }

        if let Some(rank) = cc.field_immutability(Some(owner), &field.name) {
            is_immutable = rank.is_immutable();
            is_quasi_immutable = rank.is_quasi_immutable();
        }
    }

    crate::jitcode::BhDescr::Field {
        offset,
        field_size,
        field_type,
        field_flag,
        is_field_signed,
        is_immutable,
        is_quasi_immutable,
        index_in_parent,
        parent,
        name: field.name.clone(),
        owner: field.owner_root.clone().unwrap_or_default(),
    }
}

fn heuristic_field_layout(
    cc: &CallControl,
    owner: &str,
    field_name: &str,
) -> Option<(
    usize,
    usize,
    majit_ir::value::Type,
    majit_ir::descr::ArrayFlag,
    bool,
)> {
    let fields = cc.struct_field_entries(owner)?;
    let mut offset = 0usize;
    for (name, type_str) in fields {
        let (flag, field_type, mut field_size) = if cc.is_known_struct(type_str) {
            (
                majit_ir::descr::ArrayFlag::Struct,
                majit_ir::value::Type::Ref,
                cc.struct_layouts
                    .get(type_str)
                    .map(|layout| layout.size)
                    .unwrap_or(std::mem::size_of::<usize>()),
            )
        } else {
            type_flag_from_str(type_str)
        };
        if field_type == majit_ir::value::Type::Void || field_size == 0 {
            continue;
        }
        let align = field_size.min(std::mem::size_of::<usize>());
        offset = (offset + align - 1) & !(align - 1);
        if name == field_name {
            return Some((
                offset,
                field_size,
                field_type,
                flag,
                flag == majit_ir::descr::ArrayFlag::Signed,
            ));
        }
        field_size = field_size.max(1);
        offset += field_size;
    }
    None
}

fn bh_field_flag_from_descr(fd: &dyn majit_ir::descr::FieldDescr) -> majit_ir::descr::ArrayFlag {
    if fd.is_pointer_field() {
        majit_ir::descr::ArrayFlag::Pointer
    } else if fd.is_float_field() {
        majit_ir::descr::ArrayFlag::Float
    } else if fd.field_type() == majit_ir::value::Type::Void {
        majit_ir::descr::ArrayFlag::Void
    } else if fd.is_field_signed() {
        majit_ir::descr::ArrayFlag::Signed
    } else {
        majit_ir::descr::ArrayFlag::Unsigned
    }
}

fn bh_field_spec_from_descr(fd: &dyn majit_ir::descr::FieldDescr) -> crate::jitcode::BhFieldSpec {
    let field_flag = bh_field_flag_from_descr(fd);
    crate::jitcode::BhFieldSpec {
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

fn bh_size_spec_from_descr(sd: &dyn majit_ir::descr::SizeDescr) -> crate::jitcode::BhSizeSpec {
    crate::jitcode::BhSizeSpec {
        size: sd.size(),
        type_id: sd.type_id(),
        vtable: sd.vtable(),
        all_fielddescrs: sd
            .all_fielddescrs()
            .iter()
            .map(|fd| bh_field_spec_from_descr(fd.as_ref()))
            .collect(),
    }
}

fn bh_interior_field_specs_from_array_descr(
    array_descr: &dyn majit_ir::descr::ArrayDescr,
) -> Vec<crate::jitcode::BhInteriorFieldSpec> {
    array_descr
        .get_all_interiorfielddescrs()
        .unwrap_or(&[])
        .iter()
        .filter_map(|descr| {
            let interior = descr.as_interior_field_descr()?;
            let field = bh_field_spec_from_descr(interior.field_descr());
            let owner = interior
                .field_descr()
                .get_parent_descr()
                .and_then(|parent| parent.as_size_descr().map(bh_size_spec_from_descr))
                .unwrap_or_else(|| crate::jitcode::BhSizeSpec {
                    size: array_descr.item_size(),
                    type_id: 0,
                    vtable: 0,
                    all_fielddescrs: vec![field.clone()],
                });
            Some(crate::jitcode::BhInteriorFieldSpec {
                index: descr.index(),
                field,
                owner,
            })
        })
        .collect()
}

/// jtransform.py:773,802 cpu.arraydescrof(ARRAY) equivalent.
///
/// Determines the full ArrayDescr shape from the array element type.
/// When `array_type_id` is available (e.g. `Vec<i32>` → element `i32`),
/// the result is exact. Fallback uses descr.py:241-254 get_type_flag()
/// semantics: Int → FLAG_SIGNED, Float/Ref → FLAG_UNSIGNED/FLAG_FLOAT.
fn arraydescrof(
    ty: &crate::model::ValueType,
    array_type_id: &Option<String>,
    callcontrol: Option<&CallControl>,
) -> crate::jitcode::BhDescr {
    let ir_type = value_type_to_ir_type_for_descr(ty);
    if let Some(cc) = callcontrol {
        let descr = cc.arraydescrof(0, array_type_id, ir_type);
        let array_descr = descr
            .as_array_descr()
            .expect("CallControl::arraydescrof must return an ArrayDescr");
        return crate::jitcode::BhDescr::Array {
            base_size: array_descr.base_size(),
            itemsize: array_descr.item_size(),
            type_id: array_descr.type_id(),
            item_type: array_descr.item_type(),
            is_array_of_pointers: array_descr.is_array_of_pointers(),
            is_array_of_structs: array_descr.is_array_of_structs(),
            is_item_signed: array_descr.is_item_signed(),
            interior_fields: bh_interior_field_specs_from_array_descr(array_descr),
        };
    }

    // Primary path: extract element type from the array type identity
    // (our equivalent of `ARRAY.OF` in RPython).
    let (flag, item_type, itemsize) = if let Some(elem) = array_type_id
        .as_deref()
        .and_then(extract_element_type_from_str)
    {
        type_flag_from_str(elem.as_str())
    } else {
        match ty {
            crate::model::ValueType::Int => (
                majit_ir::descr::ArrayFlag::Signed,
                majit_ir::value::Type::Int,
                8,
            ),
            crate::model::ValueType::Float => (
                majit_ir::descr::ArrayFlag::Float,
                majit_ir::value::Type::Float,
                8,
            ),
            crate::model::ValueType::Ref => (
                majit_ir::descr::ArrayFlag::Pointer,
                majit_ir::value::Type::Ref,
                8,
            ),
            _ => (
                majit_ir::descr::ArrayFlag::Unsigned,
                majit_ir::value::Type::Int,
                8,
            ),
        }
    };
    crate::jitcode::BhDescr::Array {
        base_size: std::mem::size_of::<usize>(),
        itemsize,
        type_id: 0,
        item_type,
        is_array_of_pointers: flag == majit_ir::descr::ArrayFlag::Pointer,
        is_array_of_structs: flag == majit_ir::descr::ArrayFlag::Struct,
        is_item_signed: flag == majit_ir::descr::ArrayFlag::Signed,
        interior_fields: Vec::new(),
    }
}

fn vable_arraydescrof(
    ty: &crate::model::ValueType,
    itemsize: usize,
    is_item_signed: bool,
) -> crate::jitcode::BhDescr {
    let item_type = value_type_to_ir_type_for_descr(ty);
    crate::jitcode::BhDescr::Array {
        base_size: std::mem::size_of::<usize>(),
        itemsize,
        type_id: 0,
        item_type,
        is_array_of_pointers: matches!(item_type, majit_ir::value::Type::Ref),
        is_array_of_structs: false,
        is_item_signed,
        interior_fields: Vec::new(),
    }
}

fn extract_element_type_from_str(type_str: &str) -> Option<String> {
    let s = type_str.trim();
    if let (Some(start), Some(end)) = (s.find('<'), s.rfind('>')) {
        if start < end {
            return Some(s[start + 1..end].trim().to_string());
        }
    }
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        let elem = if let Some(semi) = inner.find(';') {
            inner[..semi].trim()
        } else {
            inner.trim()
        };
        if !elem.is_empty() {
            return Some(elem.to_string());
        }
    }
    None
}

/// Convert OpKind to an opname string for the assembler's instruction table.
/// RPython: the opname comes from SpaceOperation.opname.
/// Convert OpKind to a typed opname matching RPython's jtransform output.
///
/// RPython jtransform produces fully-qualified names like `getfield_vable_i`,
/// `setfield_gc_r`, `int_add`. The kind suffix comes from the result type
/// or value type of the operation.
fn op_kind_to_opname(kind: &crate::model::OpKind) -> String {
    use crate::model::OpKind;
    match kind {
        OpKind::Input { ty, .. } => format!("input_{}", value_type_to_kind(ty)),
        // RPython: ConstInt is NOT a standalone op; see encode_op comment.
        // Pyre materialises constants as an int_copy from pool-region reg.
        OpKind::ConstInt(_) => "int_copy".into(),
        // Mirrors `ConstInt` — the constant is materialised through the
        // shared `constants_f` pool, then a `float_copy` op moves it into
        // the SSA destination register.
        OpKind::ConstFloat(_) => "float_copy".into(),
        // RPython: getfield_gc_i, getfield_gc_r, getfield_gc_f and `_pure`
        // variants from jtransform.py rewrite_op_getfield().
        OpKind::FieldRead { ty, pure, .. } => {
            let mut opname = format!("getfield_gc_{}", value_type_to_kind(ty));
            if *pure {
                opname.push_str("_pure");
            }
            opname
        }
        OpKind::FieldWrite { ty, .. } => format!("setfield_gc_{}", value_type_to_kind(ty)),
        // RPython: getarrayitem_gc_i etc.
        OpKind::ArrayRead { item_ty, .. } => {
            format!("getarrayitem_gc_{}", value_type_to_kind(item_ty))
        }
        OpKind::ArrayWrite { item_ty, .. } => {
            format!("setarrayitem_gc_{}", value_type_to_kind(item_ty))
        }
        // RPython: getinteriorfield_gc_i etc.
        OpKind::InteriorFieldRead { item_ty, .. } => {
            format!("getinteriorfield_gc_{}", value_type_to_kind(item_ty))
        }
        OpKind::InteriorFieldWrite { item_ty, .. } => {
            format!("setinteriorfield_gc_{}", value_type_to_kind(item_ty))
        }
        OpKind::Call { result_ty, .. } => {
            format!("direct_call_{}", value_type_to_kind(result_ty))
        }
        OpKind::GuardTrue { .. } => "guard_true".into(),
        OpKind::GuardFalse { .. } => "guard_false".into(),
        OpKind::GuardValue { kind_char, .. } => format!("{kind_char}_guard_value"),
        // RPython: getfield_vable_i, getfield_vable_r, getfield_vable_f
        OpKind::VableFieldRead { ty, .. } => {
            format!("getfield_vable_{}", value_type_to_kind(ty))
        }
        OpKind::VableFieldWrite { ty, .. } => {
            format!("setfield_vable_{}", value_type_to_kind(ty))
        }
        // RPython: getarrayitem_vable_i etc.
        OpKind::VableArrayRead { item_ty, .. } => {
            format!("getarrayitem_vable_{}", value_type_to_kind(item_ty))
        }
        OpKind::VableArrayWrite { item_ty, .. } => {
            format!("setarrayitem_vable_{}", value_type_to_kind(item_ty))
        }
        // RPython `blackhole.py:500` canonical opnames for bitwise ints are
        // `int_and` / `int_or` / `int_xor`. pyre's front-end (`front/ast.rs`
        // `binary_op_name`) uses Rust's `syn::BinOp` trait names
        // (`bitand`/`bitor`/`bitxor`) for source faithfulness, so rename them
        // here at the emission boundary instead of duplicating wire entries
        // in the blackhole dispatch table.
        OpKind::BinOp { op, .. } => match op.as_str() {
            "bitand" => "int_and".into(),
            "bitor" => "int_or".into(),
            "bitxor" => "int_xor".into(),
            // RPython `jtransform.py:1243-1255` produces these opnames as-is —
            // do not prefix with `int_`.
            "ptr_eq" | "ptr_ne" => op.clone(),
            // jtransform-rewritten float operands carry the full RPython
            // opname (`float_add` / `float_lt` / etc.) — preserve as-is.
            s if s.starts_with("float_") => op.clone(),
            _ => format!("int_{op}"),
        },
        // RPython `blackhole.py:488-498`: bitwise NOT on i64 is `int_invert`.
        // pyre's front-end uses Rust's `syn::UnOp::Not` spelling `not` for
        // both logical-not and bitwise-not (they share the `!` token at the
        // AST level); canonicalize to `int_invert` at the emission boundary.
        OpKind::UnaryOp { op, .. } => match op.as_str() {
            "not" => "int_invert".into(),
            s if s.starts_with("float_") || s.starts_with("cast_") => op.clone(),
            _ => format!("int_{op}"),
        },
        OpKind::VableForce { .. } => "hint_force_virtualizable".into(),
        // jtransform.py:1731-1743 — jit.* builtin ops
        OpKind::JitDebug { .. } => "jit_debug".into(),
        OpKind::AssertGreen { kind_char, .. } => format!("{kind_char}_assert_green"),
        OpKind::CurrentTraceLength => "current_trace_length".into(),
        OpKind::IsConstant { kind_char, .. } => format!("{kind_char}_isconstant"),
        OpKind::IsVirtual { kind_char, .. } => format!("{kind_char}_isvirtual"),
        OpKind::RecordKnownResult { result_kind, .. } => {
            format!("record_known_result_{result_kind}")
        }
        // jtransform.py:1665-1688 — conditional_call ops
        OpKind::ConditionalCall { .. } => "conditional_call".into(),
        OpKind::ConditionalCallValue { result_kind, .. } => {
            format!("conditional_call_value_{result_kind}")
        }
        OpKind::Live => "live".into(),
        // jtransform.py:1707,1718 — jit_merge_point / loop_header markers.
        OpKind::JitMergePoint { .. } => "jit_merge_point".into(),
        OpKind::LoopHeader { .. } => "loop_header".into(),
        // Call variants are handled by encode_op directly, not here.
        OpKind::CallElidable { .. } => "call_elidable".into(),
        OpKind::CallResidual { .. } => "residual_call".into(),
        OpKind::CallMayForce { .. } => "call_may_force".into(),
        OpKind::InlineCall { .. } => "inline_call".into(),
        OpKind::RecursiveCall { .. } => "recursive_call".into(),
        // RPython: no dedicated opname — the vtable entry becomes the `funcptr`
        // Variable that `int_guard_value` + `residual_call_*` consume.
        OpKind::VtableMethodPtr { .. } => "vtable_method_ptr".into(),
        OpKind::IndirectCall { .. } => "indirect_call".into(),
        // jtransform.py:901-903 — `record_quasiimmut_field(v_inst, descr, descr1)`.
        OpKind::RecordQuasiImmutField { .. } => "record_quasiimmut_field".into(),
        OpKind::Abort { .. } => "abort".into(),
    }
}

// Re-export CallInfoCollection from majit-ir (effectinfo.py::CallInfoCollection).
// majit-ir already has the RPython-parity version with OopSpecIndex keys.
pub use majit_ir::CallInfoCollection;

impl Assembler {
    /// RPython: `Assembler.see_raw_object(value)` (assembler.py:283-298).
    ///
    /// Registers a function/vtable name for debugging.
    /// RPython stores `(addr, name)` pairs; majit stores `(path, name)`.
    pub fn see_raw_object(&mut self, path: &str, name: &str) {
        if self.seen_raw_objects.insert(path.to_string()) {
            self.list_of_addr2name
                .push((path.to_string(), name.to_string()));
        }
    }

    /// RPython: `Assembler.finished(callinfocollection)` (assembler.py:300-305).
    ///
    /// ```python
    /// def finished(self, callinfocollection):
    ///     for func in callinfocollection.all_function_addresses_as_int():
    ///         func = int2adr(func)
    ///         self.see_raw_object(func.ptr)
    /// ```
    ///
    /// RPython's `see_raw_object` extracts `func.ptr._obj._name` to build
    /// `list_of_addr2name`. In majit, names are registered at `add()` time
    /// via `register_func_name()`.
    /// RPython: Assembler.insns — the opcode table. Needed by
    /// BlackholeInterpBuilder::setup_insns() to build the dispatch table.
    pub fn insns(&self) -> &HashMap<String, u8> {
        &self.insns
    }

    /// Register an `(opname/argcodes, opnum)` pair into `self.insns`.
    ///
    /// RPython `assembler.py:222 self.insns[key] = opnum` records every
    /// opcode the assembler emits during `assemble()`.  Pyre's
    /// state-field-JIT macro path skips `assemble()` entirely (the
    /// `JitCodeBuilder` emits BC_* directly), so the canonical entries
    /// — `live/`, `catch_exception/L`, `*_return/*` — are populated
    /// here at install time so `MetaInterpStaticData::setup_insns`
    /// (`pyjitpl.py:2227-2243`) can do the dynamic
    /// `insns.get(name)` lookup instead of a parallel hardcoded
    /// `BC_*` seeding block.
    pub fn register_insn(&mut self, name: &str, opnum: u8) {
        self.insns.insert(name.to_string(), opnum);
    }

    /// RPython `assembler.py:29 self.all_liveness = []` — the shared
    /// liveness byte stream populated by `_encode_liveness`.  Returned
    /// as a contiguous `&[u8]` view so consumers (notably
    /// `MetaInterpStaticData::finish_setup` per `pyjitpl.py:2264`) can
    /// take a snapshot without depending on the dedup cache or
    /// position table.
    pub fn all_liveness(&self) -> &[u8] {
        &self.all_liveness
    }

    /// Snapshot the descriptor table after all jitcodes have been fully
    /// assembled. Pending inline-call descriptors are lowered here to the
    /// final `(jitcode_index, fnaddr, calldescr)` form that runtime
    /// consumers expect.
    pub fn snapshot_descrs(&self) -> Vec<crate::jitcode::BhDescr> {
        self.descrs
            .iter()
            .map(|descr| match descr {
                AssemblerDescr::Ready(descr) => descr.clone(),
                AssemblerDescr::PendingJitCode { jitcode } => crate::jitcode::BhDescr::JitCode {
                    jitcode_index: jitcode.index(),
                    fnaddr: jitcode.fnaddr,
                    calldescr: jitcode.calldescr().clone(),
                },
                AssemblerDescr::PendingSwitch { .. } => {
                    panic!("snapshot_descrs called before switch descriptors were resolved")
                }
            })
            .collect()
    }

    pub fn finished(&mut self, callinfocollection: &CallInfoCollection) {
        for func_addr in callinfocollection.all_function_addresses_as_int() {
            // RPython: see_raw_object(func.ptr)
            // → name = value._obj._name (for FuncType)
            // → self.list_of_addr2name.append((addr, name))
            let name = callinfocollection.func_name(func_addr).unwrap_or("?");
            let addr_key = format!("{func_addr:#x}");
            self.see_raw_object(&addr_key, name);
        }
    }

    /// Number of JitCodes assembled so far.
    pub fn count_jitcodes(&self) -> usize {
        self.count_jitcodes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EffectInfoKey {
    extraeffect: majit_ir::descr::ExtraEffect,
    oopspecindex: majit_ir::descr::OopSpecIndex,
    readonly_descrs_fields: u64,
    write_descrs_fields: u64,
    readonly_descrs_arrays: u64,
    write_descrs_arrays: u64,
    readonly_descrs_interiorfields: u64,
    write_descrs_interiorfields: u64,
    can_invalidate: bool,
    can_collect: bool,
    call_release_gil_target: (u64, i32),
}

impl EffectInfoKey {
    fn from_effect_info(effect: &majit_ir::descr::EffectInfo) -> Self {
        Self {
            extraeffect: effect.extraeffect,
            oopspecindex: effect.oopspecindex,
            readonly_descrs_fields: effect.readonly_descrs_fields,
            write_descrs_fields: effect.write_descrs_fields,
            readonly_descrs_arrays: effect.readonly_descrs_arrays,
            write_descrs_arrays: effect.write_descrs_arrays,
            readonly_descrs_interiorfields: effect.readonly_descrs_interiorfields,
            write_descrs_interiorfields: effect.write_descrs_interiorfields,
            can_invalidate: effect.can_invalidate,
            can_collect: effect.can_collect,
            call_release_gil_target: effect.call_release_gil_target,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AssemblerDescrKey {
    Field {
        offset: usize,
        field_size: usize,
        field_type: majit_ir::value::Type,
        field_flag: majit_ir::descr::ArrayFlag,
        is_field_signed: bool,
        is_immutable: bool,
        is_quasi_immutable: bool,
        index_in_parent: usize,
        parent: Option<crate::jitcode::BhSizeSpec>,
        name: String,
        owner: String,
    },
    Array {
        base_size: usize,
        itemsize: usize,
        type_id: u32,
        item_type: majit_ir::value::Type,
        is_array_of_pointers: bool,
        is_array_of_structs: bool,
        is_item_signed: bool,
        interior_fields: Vec<crate::jitcode::BhInteriorFieldSpec>,
    },
    Size {
        size: usize,
        type_id: u32,
        vtable: usize,
        owner: String,
        all_fielddescrs: Vec<crate::jitcode::BhFieldSpec>,
    },
    Call {
        arg_classes: String,
        result_type: char,
        result_signed: bool,
        result_size: usize,
        result_erased: crate::jitcode::CallResultErasedKey,
        effect: EffectInfoKey,
    },
    /// RPython uses the JitCode object itself as an AbstractDescr for
    /// inline_call. The Rust key is therefore the identity-keyed handle, not
    /// the callsite-local `BhCallDescr`.
    JitCode(crate::jitcode::JitCodeHandle),
    SnapshotJitCode {
        jitcode_index: usize,
        fnaddr: i64,
        arg_classes: String,
        result_type: char,
        result_signed: bool,
        result_size: usize,
        result_erased: crate::jitcode::CallResultErasedKey,
        effect: EffectInfoKey,
    },
    Switch(Vec<(i64, usize)>),
    VableField {
        index: usize,
    },
    VableArray {
        index: usize,
    },
    VtableMethod {
        trait_root: String,
        method_name: String,
    },
}

impl AssemblerDescrKey {
    fn from_descr(descr: &AssemblerDescr) -> Self {
        match descr {
            AssemblerDescr::Ready(descr) => Self::from_ready(descr),
            AssemblerDescr::PendingJitCode { jitcode } => Self::JitCode(jitcode.clone()),
            AssemblerDescr::PendingSwitch { .. } => {
                unreachable!("switch descriptors bypass `_descr_dict`")
            }
        }
    }

    fn from_ready(descr: &crate::jitcode::BhDescr) -> Self {
        match descr {
            crate::jitcode::BhDescr::Field {
                offset,
                field_size,
                field_type,
                field_flag,
                is_field_signed,
                is_immutable,
                is_quasi_immutable,
                index_in_parent,
                parent,
                name,
                owner,
            } => Self::Field {
                offset: *offset,
                field_size: *field_size,
                field_type: *field_type,
                field_flag: *field_flag,
                is_field_signed: *is_field_signed,
                is_immutable: *is_immutable,
                is_quasi_immutable: *is_quasi_immutable,
                index_in_parent: *index_in_parent,
                parent: parent.clone(),
                name: name.clone(),
                owner: owner.clone(),
            },
            crate::jitcode::BhDescr::Array {
                base_size,
                itemsize,
                type_id,
                item_type,
                is_array_of_pointers,
                is_array_of_structs,
                is_item_signed,
                interior_fields,
            } => Self::Array {
                base_size: *base_size,
                itemsize: *itemsize,
                type_id: *type_id,
                item_type: *item_type,
                is_array_of_pointers: *is_array_of_pointers,
                is_array_of_structs: *is_array_of_structs,
                is_item_signed: *is_item_signed,
                interior_fields: interior_fields.clone(),
            },
            crate::jitcode::BhDescr::Size {
                size,
                type_id,
                vtable,
                owner,
                all_fielddescrs,
            } => Self::Size {
                size: *size,
                type_id: *type_id,
                vtable: *vtable,
                owner: owner.clone(),
                all_fielddescrs: all_fielddescrs.clone(),
            },
            crate::jitcode::BhDescr::Call { calldescr } => Self::Call {
                arg_classes: calldescr.arg_classes.clone(),
                result_type: calldescr.result_type,
                result_signed: calldescr.result_signed,
                result_size: calldescr.result_size,
                result_erased: calldescr.result_erased,
                effect: EffectInfoKey::from_effect_info(&calldescr.extra_info),
            },
            crate::jitcode::BhDescr::JitCode {
                jitcode_index,
                fnaddr,
                calldescr,
            } => Self::SnapshotJitCode {
                jitcode_index: *jitcode_index,
                fnaddr: *fnaddr,
                arg_classes: calldescr.arg_classes.clone(),
                result_type: calldescr.result_type,
                result_signed: calldescr.result_signed,
                result_size: calldescr.result_size,
                result_erased: calldescr.result_erased,
                effect: EffectInfoKey::from_effect_info(&calldescr.extra_info),
            },
            crate::jitcode::BhDescr::Switch { dict } => {
                let mut items: Vec<_> = dict.iter().map(|(key, value)| (*key, *value)).collect();
                items.sort_unstable_by_key(|(key, _)| *key);
                Self::Switch(items)
            }
            crate::jitcode::BhDescr::VableField { index } => Self::VableField { index: *index },
            crate::jitcode::BhDescr::VableArray { index } => Self::VableArray { index: *index },
            crate::jitcode::BhDescr::VtableMethod {
                trait_root,
                method_name,
            } => Self::VtableMethod {
                trait_root: trait_root.clone(),
                method_name: method_name.clone(),
            },
        }
    }
}

#[derive(Debug, Clone)]
enum AssemblerDescr {
    Ready(crate::jitcode::BhDescr),
    PendingJitCode {
        jitcode: crate::jitcode::JitCodeHandle,
    },
    PendingSwitch {
        cases: Vec<(i64, Label)>,
    },
}

impl Default for Assembler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{ConstValue, HostObject};
    use crate::model::LinkArg;
    use crate::regalloc;

    fn empty_regallocs() -> HashMap<RegKind, regalloc::RegAllocResult> {
        let mut regallocs = HashMap::new();
        regallocs.insert(
            RegKind::Int,
            regalloc::RegAllocResult {
                coloring: HashMap::new(),
                num_regs: 0,
            },
        );
        regallocs.insert(
            RegKind::Ref,
            regalloc::RegAllocResult {
                coloring: HashMap::new(),
                num_regs: 0,
            },
        );
        regallocs.insert(
            RegKind::Float,
            regalloc::RegAllocResult {
                coloring: HashMap::new(),
                num_regs: 0,
            },
        );
        regallocs
    }

    #[test]
    fn emit_descr_reuses_rpython_descr_dict_index() {
        let mut asm = Assembler::new();

        let first = asm.emit_ready_descr(crate::jitcode::BhDescr::Field {
            offset: 0,
            field_size: 8,
            field_type: majit_ir::value::Type::Ref,
            field_flag: majit_ir::descr::ArrayFlag::Pointer,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: 0,
            parent: None,
            name: "value".into(),
            owner: "Cell".into(),
        });
        let repeated = asm.emit_ready_descr(crate::jitcode::BhDescr::Field {
            offset: 0,
            field_size: 8,
            field_type: majit_ir::value::Type::Ref,
            field_flag: majit_ir::descr::ArrayFlag::Pointer,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: 0,
            parent: None,
            name: "value".into(),
            owner: "Cell".into(),
        });
        let other = asm.emit_ready_descr(crate::jitcode::BhDescr::Field {
            offset: 0,
            field_size: 8,
            field_type: majit_ir::value::Type::Ref,
            field_flag: majit_ir::descr::ArrayFlag::Pointer,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: 1,
            parent: None,
            name: "mutate_value".into(),
            owner: "Cell".into(),
        });

        assert_eq!(first, repeated);
        assert_ne!(first, other);
        assert_eq!(asm.snapshot_descrs().len(), 2);
    }

    #[test]
    fn emit_call_descr_key_uses_full_rpython_calldescr_shape() {
        let mut asm = Assembler::new();
        let effect = majit_ir::descr::EffectInfo::MOST_GENERAL;

        let signed_int =
            crate::jitcode::BhCallDescr::from_arg_classes("i".to_string(), 'i', effect.clone());
        let signed_int_repeat =
            crate::jitcode::BhCallDescr::from_arg_classes("i".to_string(), 'i', effect.clone());
        let single_float =
            crate::jitcode::BhCallDescr::from_arg_classes("i".to_string(), 'S', effect.clone());
        let unsigned_int = crate::jitcode::BhCallDescr {
            arg_classes: "i".to_string(),
            result_type: 'i',
            result_signed: false,
            result_size: 8,
            result_erased: crate::jitcode::CallResultErasedKey::Unsigned,
            extra_info: effect.clone(),
        };
        let raw_address = crate::jitcode::BhCallDescr {
            arg_classes: "i".to_string(),
            result_type: 'i',
            result_signed: false,
            result_size: 8,
            result_erased: crate::jitcode::CallResultErasedKey::Address,
            extra_info: effect,
        };

        assert_eq!(single_float.result_type, 'S');
        assert_eq!(single_float.result_size, 4);
        assert!(!single_float.result_signed);
        assert_eq!(
            single_float.result_erased,
            crate::jitcode::CallResultErasedKey::SingleFloat,
        );

        let signed_idx = asm.emit_ready_descr(crate::jitcode::BhDescr::Call {
            calldescr: signed_int,
        });
        let signed_repeat_idx = asm.emit_ready_descr(crate::jitcode::BhDescr::Call {
            calldescr: signed_int_repeat,
        });
        let single_idx = asm.emit_ready_descr(crate::jitcode::BhDescr::Call {
            calldescr: single_float,
        });
        let unsigned_idx = asm.emit_ready_descr(crate::jitcode::BhDescr::Call {
            calldescr: unsigned_int,
        });
        let address_idx = asm.emit_ready_descr(crate::jitcode::BhDescr::Call {
            calldescr: raw_address,
        });

        assert_eq!(signed_idx, signed_repeat_idx);
        assert_ne!(signed_idx, single_idx);
        assert_ne!(signed_idx, unsigned_idx);
        assert_ne!(unsigned_idx, address_idx);
        assert_eq!(asm.snapshot_descrs().len(), 4);
    }

    #[test]
    fn jit_merge_point_and_loop_header_opnames() {
        // jtransform.py:1707 `op1 = SpaceOperation('jit_merge_point', args, None)`
        let merge = crate::model::OpKind::JitMergePoint {
            jitdriver_index: 0,
            greens_i: vec![],
            greens_r: vec![],
            greens_f: vec![],
            reds_i: vec![],
            reds_r: vec![],
            reds_f: vec![],
        };
        assert_eq!(op_kind_to_opname(&merge), "jit_merge_point");
        // jtransform.py:1718 `SpaceOperation('loop_header', [c_index], None)`
        let header = crate::model::OpKind::LoopHeader { jitdriver_index: 0 };
        assert_eq!(op_kind_to_opname(&header), "loop_header");
    }

    #[test]
    fn assemble_basic() {
        let mut flat = SSARepr {
            name: "test".into(),
            insns: vec![],
            num_values: 0,
            num_blocks: 1,
            value_kinds: HashMap::new(),
            insns_pos: None,
        };

        let regallocs = empty_regallocs();
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);

        assert_eq!(flat.name, "test");
        assert_eq!(body.c_num_regs_i as usize, 0);
        assert_eq!(body.c_num_regs_r as usize, 0);
        assert_eq!(body.c_num_regs_f as usize, 0);
        assert_eq!(asm.count_jitcodes(), 1);
    }

    #[test]
    fn assemble_ref_return_with_host_object_constant() {
        let module = HostObject::new_module("hello");
        let mut flat = SSARepr {
            name: "return_host_object".into(),
            insns: vec![FlatOp::RefReturn(LinkArg::Const(ConstValue::HostObject(
                module.clone(),
            )))],
            num_values: 0,
            num_blocks: 1,
            value_kinds: HashMap::new(),
            insns_pos: None,
        };

        let regallocs = empty_regallocs();
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);

        assert_eq!(body.constants_r, vec![module.identity_id() as i64]);
        assert!(asm.insns.contains_key("ref_return/r"));
    }

    #[test]
    fn assemble_with_registers() {
        use crate::model::{FunctionGraph, OpKind, ValueType};
        // Build graph for regalloc (regalloc operates on graph, not SSARepr)
        let mut graph = FunctionGraph::new("add");
        let entry = graph.startblock;
        let v0 = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "a".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let v1 = graph
            .push_op(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: v0,
                    rhs: v0,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let v2 = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "r".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(v1));

        let mut value_kinds = HashMap::new();
        value_kinds.insert(v0, RegKind::Int);
        value_kinds.insert(v1, RegKind::Int);
        value_kinds.insert(v2, RegKind::Ref);

        let regallocs = regalloc::perform_all_register_allocations(&graph, &value_kinds);
        let mut flat = SSARepr {
            name: "add".into(),
            insns: vec![],
            num_values: 3,
            num_blocks: 1,
            value_kinds,
            insns_pos: None,
        };
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);

        // v0 dies when v1 is defined → they share a register → 1 int reg
        assert_eq!(body.c_num_regs_i as usize, 1);
        assert_eq!(body.c_num_regs_r as usize, 1);
        assert_eq!(body.c_num_regs_f as usize, 0);
    }

    #[test]
    fn assemble_direct_residual_call_encodes_leading_funcptr_operand() {
        use crate::call::CallControl;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FunctionGraph, OpKind, ValueType};
        use crate::translate_legacy::annotator::annrpython::annotate;
        use crate::translate_legacy::rtyper::rtyper::{resolve_rewritten_types, resolve_types};

        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("caller");
        let arg = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "arg".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: crate::model::CallTarget::function_path(["custom_reader"]),
                args: vec![arg],
                result_ty: ValueType::Ref,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let annotations = annotate(&graph);
        let type_state = resolve_types(&graph, &annotations);
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .with_type_state(&type_state);
        let rewritten = transformer.transform(&graph);
        let rewritten_types = resolve_rewritten_types(
            &type_state,
            &rewritten.graph,
            &annotations,
            &rewritten.synth_kinds,
        );
        let value_kinds = crate::jit_codewriter::type_state::build_value_kinds(&rewritten_types);
        let regallocs = regalloc::perform_all_register_allocations(&rewritten.graph, &value_kinds);
        let mut flat =
            crate::flatten::flatten_with_types(&rewritten.graph, &rewritten_types, &regallocs);

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        // RPython jtransform.py:422-431 canonical order:
        // `residual_call_r_r/iRd>r` (funcptr, R-list, descr, >result).
        assert!(
            asm.insns.contains_key("residual_call_r_r/iRd>r"),
            "expected funcptr-first residual_call key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    /// Boundary test (plan Rev 2 Phase E
    /// `no_legacy_funcptr_from_vtable_after_new_pipeline`): after the
    /// rtyper-equivalent `lower_indirect_calls` pass + jtransform +
    /// assemble, no `funcptr_from_vtable/*` opcode key may survive.
    /// The RPython-orthodox replacement is `vtable_method_ptr/rd>i`
    /// emitted by `OpKind::VtableMethodPtr`
    /// (`translator/rtyper/rclass.rs::class_get_method_ptr` ↔
    /// `rpython/rtyper/rclass.py:371-377` + `rpython/rtyper/rpbc.py:1203-1205`).
    /// `OpKind::FuncptrFromVtable` has been deleted from the enum
    /// (Phase D compile-time guarantee); this test adds a runtime
    /// guarantee at the assembled-opname level.
    #[test]
    fn no_legacy_funcptr_from_vtable_after_new_pipeline() {
        use crate::call::CallControl;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{CallTarget, FunctionGraph, OpKind, ValueType};
        use crate::translate_legacy::annotator::annrpython::annotate;
        use crate::translate_legacy::rtyper::rtyper::{resolve_rewritten_types, resolve_types};
        use crate::translator::rtyper::rpbc::lower_indirect_calls;

        fn build_run_impl(name: &str) -> FunctionGraph {
            let mut g = FunctionGraph::new(name);
            g.push_op(
                g.startblock,
                OpKind::Input {
                    name: "self".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
            g.set_return(g.startblock, None);
            g
        }

        let mut cc = CallControl::new();
        cc.register_trait_method("run", Some("Handler"), "A", build_run_impl("A::run"));
        cc.register_trait_method("run", Some("Handler"), "B", build_run_impl("B::run"));
        cc.find_all_graphs_for_tests();

        let mut graph = FunctionGraph::new("outer");
        let receiver = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "handler".into(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        // Per RPython `flatten.py:373-380` `serialize_op` — an op whose
        // result has `getkind(concretetype) == 'void'` does not get a
        // color lookup.  Mirror that here by passing `has_result=false`
        // so no `ValueId` is allocated on the void `run()` return.
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("Handler", "run"),
                args: vec![receiver],
                result_ty: ValueType::Void,
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        let annotations = annotate(&graph);
        let mut type_state = resolve_types(&graph, &annotations);
        lower_indirect_calls(&mut graph, &mut type_state, &cc);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .with_type_state(&type_state);
        let rewritten = transformer.transform(&graph);
        let rewritten_types = resolve_rewritten_types(
            &type_state,
            &rewritten.graph,
            &annotations,
            &rewritten.synth_kinds,
        );
        let value_kinds = crate::jit_codewriter::type_state::build_value_kinds(&rewritten_types);
        let regallocs = regalloc::perform_all_register_allocations(&rewritten.graph, &value_kinds);
        let mut flat =
            crate::flatten::flatten_with_types(&rewritten.graph, &rewritten_types, &regallocs);

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        let legacy_keys: Vec<&String> = asm
            .insns
            .keys()
            .filter(|k| k.starts_with("funcptr_from_vtable"))
            .collect();
        assert!(
            legacy_keys.is_empty(),
            "legacy funcptr_from_vtable opname survived after the new \
             lower_indirect_calls + jtransform + assemble pipeline: {:?}",
            legacy_keys,
        );

        let has_vtable_method_ptr = asm.insns.keys().any(|k| k.starts_with("vtable_method_ptr"));
        assert!(
            has_vtable_method_ptr,
            "expected vtable_method_ptr opname from OpKind::VtableMethodPtr, \
             insns keys: {:?}",
            asm.insns.keys().collect::<Vec<_>>(),
        );
    }

    /// `OpKind::RecordQuasiImmutField` must lower to a single opcode
    /// keyed `record_quasiimmut_field/rdd`, with the field+mutate
    /// FieldDescriptor pair pushed as two `BhDescr::Field` entries — see
    /// `rpython/jit/codewriter/jtransform.py:901-903` and
    /// `rpython/jit/metainterp/blackhole.py:1537-1539`.
    #[test]
    fn assembles_record_quasiimmut_field_with_two_descrs() {
        use crate::call::CallControl;
        use crate::flatten::flatten as flatten_graph;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FieldDescriptor, FunctionGraph, ImmutableRank, OpKind, ValueType};

        let mut cc = CallControl::new();
        cc.immutable_fields_by_struct.insert(
            "Cell".to_string(),
            vec![("value".to_string(), ImmutableRank::QuasiImmutable)],
        );

        let mut graph = FunctionGraph::new("read_cell");
        let base = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "cell".to_string(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        let result = graph
            .push_op(
                graph.startblock,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new("value", Some("Cell".to_string())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result));

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let rewritten = transformer.transform(&graph).graph;

        let mut value_kinds = HashMap::new();
        value_kinds.insert(base, RegKind::Ref);
        value_kinds.insert(result, RegKind::Int);
        let regallocs = regalloc::perform_all_register_allocations(&rewritten, &value_kinds);
        let mut flat = flatten_graph(&rewritten, &regallocs);
        // Slice C-3: seed `SSARepr.value_kinds` with the canonical-
        // exceptblock-augmented map — `flatten_graph` (without type
        // state) leaves it empty, but the Slice C-3 lookup_coloring
        // contract requires the same authoritative table that
        // `perform_all_register_allocations` consumed.
        flat.value_kinds =
            regalloc::augment_value_kinds_with_canonical_exceptblock(&rewritten, &value_kinds);
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        let key_count = asm
            .insns
            .keys()
            .filter(|k| k.starts_with("record_quasiimmut_field/"))
            .count();
        assert_eq!(
            key_count,
            1,
            "expected exactly one record_quasiimmut_field/* key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("record_quasiimmut_field/rdd"),
            "expected key record_quasiimmut_field/rdd, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("getfield_gc_i_pure/rd>i"),
            "expected pure getfield opcode, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        // Two BhDescr::Field entries — for `value` and `mutate_value`.
        let descrs = asm.snapshot_descrs();
        let field_descr_names: Vec<&str> = descrs
            .iter()
            .filter_map(|d| match d {
                crate::jitcode::BhDescr::Field { name, owner, .. } if owner == "Cell" => {
                    Some(name.as_str())
                }
                _ => None,
            })
            .collect();
        assert!(
            field_descr_names.contains(&"value") && field_descr_names.contains(&"mutate_value"),
            "expected Field descrs for `value` + `mutate_value`, got {:?}",
            field_descr_names
        );
    }

    #[test]
    fn assemble_typed_writes_use_canonical_non_v_opnames() {
        use crate::flatten::flatten as flatten_graph;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FieldDescriptor, FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("typed_writes");
        let base = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "obj".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        let index = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "i".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let value = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "v".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::FieldWrite {
                base,
                field: FieldDescriptor::new("x", Some("Point".into())),
                value,
                ty: ValueType::Unknown,
            },
            false,
        );
        graph.push_op(
            graph.startblock,
            OpKind::ArrayWrite {
                base,
                index,
                value,
                item_ty: ValueType::Unknown,
                array_type_id: None,
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        let mut type_state = crate::jit_codewriter::type_state::TypeResolutionState::new();
        type_state
            .concrete_types
            .insert(base, crate::jit_codewriter::type_state::ConcreteType::GcRef);
        type_state.concrete_types.insert(
            index,
            crate::jit_codewriter::type_state::ConcreteType::Signed,
        );
        type_state.concrete_types.insert(
            value,
            crate::jit_codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let rewritten = Transformer::new(&config)
            .with_type_state(&type_state)
            .transform(&graph)
            .graph;
        let value_kinds = crate::jit_codewriter::type_state::build_value_kinds(&type_state);
        let regallocs = regalloc::perform_all_register_allocations(&rewritten, &value_kinds);
        let mut flat = flatten_graph(&rewritten, &regallocs);
        // Slice C-3: seed `SSARepr.value_kinds` with the canonical-
        // exceptblock-augmented map so `lookup_coloring` finds every
        // operand including the synthetic `(etype, evalue)` pair.
        flat.value_kinds =
            regalloc::augment_value_kinds_with_canonical_exceptblock(&rewritten, &value_kinds);

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        assert!(
            asm.insns.contains_key("setfield_gc_i/rid"),
            "expected canonical setfield_gc_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("setarrayitem_gc_i/riid"),
            "expected canonical setarrayitem_gc_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setfield_gc_v/rid"),
            "unexpected setfield_gc_v key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setfield_gc_v/iid"),
            "unexpected setfield_gc_v/iid key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setfield_gc_v/ird"),
            "unexpected setfield_gc_v/ird key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setarrayitem_gc_v/riid"),
            "unexpected setarrayitem_gc_v key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setarrayitem_gc_v/iiid"),
            "unexpected setarrayitem_gc_v/iiid key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn assemble_typed_reads_use_canonical_non_v_opnames() {
        use crate::flatten::flatten as flatten_graph;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FieldDescriptor, FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("typed_reads");
        let base = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "obj".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        let index = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "i".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let field_result = graph
            .push_op(
                graph.startblock,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Unknown,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let array_result = graph
            .push_op(
                graph.startblock,
                OpKind::ArrayRead {
                    base,
                    index,
                    item_ty: ValueType::Unknown,
                    array_type_id: None,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(array_result));

        let mut type_state = crate::jit_codewriter::type_state::TypeResolutionState::new();
        type_state
            .concrete_types
            .insert(base, crate::jit_codewriter::type_state::ConcreteType::GcRef);
        type_state.concrete_types.insert(
            index,
            crate::jit_codewriter::type_state::ConcreteType::Signed,
        );
        type_state.concrete_types.insert(
            field_result,
            crate::jit_codewriter::type_state::ConcreteType::Signed,
        );
        type_state.concrete_types.insert(
            array_result,
            crate::jit_codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let rewritten = Transformer::new(&config)
            .with_type_state(&type_state)
            .transform(&graph)
            .graph;
        let value_kinds = crate::jit_codewriter::type_state::build_value_kinds(&type_state);
        let regallocs = regalloc::perform_all_register_allocations(&rewritten, &value_kinds);
        let mut flat = flatten_graph(&rewritten, &regallocs);
        // Slice C-3: seed `SSARepr.value_kinds` with the canonical-
        // exceptblock-augmented map so `lookup_coloring` finds every
        // operand including the synthetic `(etype, evalue)` pair.
        flat.value_kinds =
            regalloc::augment_value_kinds_with_canonical_exceptblock(&rewritten, &value_kinds);

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        assert!(
            asm.insns.contains_key("getfield_gc_i/rd>i"),
            "expected canonical getfield_gc_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("getarrayitem_gc_i/rid>i"),
            "expected canonical getarrayitem_gc_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getfield_gc_v/rd>i"),
            "unexpected getfield_gc_v/rd>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getfield_gc_v/id>i"),
            "unexpected getfield_gc_v/id>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getarrayitem_gc_v/rid>i"),
            "unexpected getarrayitem_gc_v/rid>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getarrayitem_gc_v/iid>i"),
            "unexpected getarrayitem_gc_v/iid>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getarrayitem_gc_v/ird>i"),
            "unexpected getarrayitem_gc_v/ird>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn assemble_vable_access_uses_explicit_base_argcodes() {
        use crate::flatten::flatten as flatten_graph;
        use crate::model::{FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("vable_access");
        let base = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "frame".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        let index = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "i".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let value = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "v".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let _field_result = graph.push_op(
            graph.startblock,
            OpKind::VableFieldRead {
                base,
                field_index: 0,
                ty: ValueType::Int,
            },
            true,
        );
        graph.push_op(
            graph.startblock,
            OpKind::VableFieldWrite {
                base,
                field_index: 0,
                value,
                ty: ValueType::Int,
            },
            false,
        );
        let array_result = graph
            .push_op(
                graph.startblock,
                OpKind::VableArrayRead {
                    base,
                    array_index: 1,
                    elem_index: index,
                    item_ty: ValueType::Int,
                    array_itemsize: 8,
                    array_is_signed: true,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::VableArrayWrite {
                base,
                array_index: 1,
                elem_index: index,
                value,
                item_ty: ValueType::Int,
                array_itemsize: 8,
                array_is_signed: true,
            },
            false,
        );
        graph.set_return(graph.startblock, Some(array_result));

        // Drive the type pipeline end-to-end so every reachable value
        // receives a concrete class.  `resolve_types` backfills any
        // orphan ValueId (return/except block pseudo-args, link-only
        // values) with `GcRef` so the assembler's
        // `lookup_reg_with_kind` never hits the missing-coloring
        // panic path for this hand-built graph.
        let annotations = crate::translate_legacy::annotator::annrpython::annotate(&graph);
        let type_state =
            crate::translate_legacy::rtyper::rtyper::resolve_types(&graph, &annotations);
        let value_kinds = crate::jit_codewriter::type_state::build_value_kinds(&type_state);
        let regallocs = regalloc::perform_all_register_allocations(&graph, &value_kinds);
        let mut flat = flatten_graph(&graph, &regallocs);
        // Slice C-3: seed `SSARepr.value_kinds` (left empty by
        // `flatten_graph`) so `lookup_coloring` finds the
        // authoritative kind for every operand.
        flat.value_kinds = value_kinds.clone();

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        assert!(
            asm.insns.contains_key("getfield_vable_i/rd>i"),
            "expected canonical getfield_vable_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("setfield_vable_i/rid"),
            "expected canonical setfield_vable_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("getarrayitem_vable_i/ridd>i"),
            "expected canonical getarrayitem_vable_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("setarrayitem_vable_i/riidd"),
            "expected canonical setarrayitem_vable_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getfield_vable_v/d>i"),
            "unexpected getfield_vable_v/d>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setfield_vable_v/id"),
            "unexpected setfield_vable_v/id key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setfield_vable_v/rd"),
            "unexpected setfield_vable_v/rd key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn assemble_skips_input_opnames_after_flatten() {
        use crate::flatten::flatten as flatten_graph;
        use crate::model::{FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("input_free_bytecode");
        let lhs = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let rhs = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let sum = graph
            .push_op(
                graph.startblock,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(sum));

        let mut value_kinds = HashMap::new();
        value_kinds.insert(lhs, RegKind::Int);
        value_kinds.insert(rhs, RegKind::Int);
        value_kinds.insert(sum, RegKind::Int);

        let regallocs = regalloc::perform_all_register_allocations(&graph, &value_kinds);
        let mut flat = flatten_graph(&graph, &regallocs);
        // Slice C-3: `flatten()` (the typed-state-less variant used
        // by this test) leaves `SSARepr.value_kinds` empty.  Seed it
        // with the canonical-exceptblock-augmented map so
        // `lookup_coloring` finds every operand including the
        // synthetic `(etype, evalue)` pair.
        flat.value_kinds =
            regalloc::augment_value_kinds_with_canonical_exceptblock(&graph, &value_kinds);
        assert!(
            !flat.insns.iter().any(|op| matches!(
                op,
                FlatOp::Op(crate::model::SpaceOperation {
                    kind: OpKind::Input { .. },
                    ..
                })
            )),
            "flatten unexpectedly left input ops: {:?}",
            flat.insns
        );

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        assert!(
            !asm.insns.keys().any(|key| key.starts_with("input_")),
            "unexpected input opcode keys: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    #[should_panic(expected = "OpKind::Input must be eliminated before assembly")]
    fn assemble_rejects_input_ops() {
        let mut flat = SSARepr {
            name: "bad_input".into(),
            insns: vec![FlatOp::Op(crate::model::SpaceOperation {
                result: Some(ValueId(0)),
                kind: crate::model::OpKind::Input {
                    name: "x".into(),
                    ty: crate::model::ValueType::Int,
                },
            })],
            num_values: 1,
            num_blocks: 1,
            value_kinds: HashMap::new(),
            insns_pos: None,
        };
        let mut regallocs = HashMap::new();
        regallocs.insert(
            RegKind::Int,
            regalloc::RegAllocResult {
                coloring: HashMap::from([(ValueId(0), 0usize)]),
                num_regs: 1,
            },
        );

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
    }
}
