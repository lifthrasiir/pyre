/// Python bytecode → JitCode compiler.
///
/// RPython: rpython/jit/codewriter/codewriter.py
///
/// RPython's CodeWriter transforms flow graphs → JitCode via a 4-step pipeline:
///   1. jtransform  — rewrite operations
///   2. regalloc    — assign registers
///   3. flatten     — CFG → linear SSARepr
///   4. assemble    — SSARepr → JitCode bytecode
///
/// For pyre, Python bytecodes are already linearized and register-allocated
/// (fast locals = registers, value stack = runtime stack). Steps 1-3 collapse
/// into a single bytecode-to-bytecode translation.
///
/// The Assembler (majit JitCodeBuilder) is RPython's assembler.py equivalent.
use std::cell::{RefCell, UnsafeCell};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use vecset::{VecMap, VecSet};

use pyre_jit_trace::{PyJitCode, PyJitCodeMetadata};

use super::assembler::Assembler;
use super::ssa_emitter::SSAReprEmitter;
use pyre_interpreter::bytecode::{CodeFlags, CodeObject, Instruction, OpArgState};
use pyre_interpreter::runtime_ops::{binary_op_tag, compare_op_tag};

use super::flatten::{
    CallDescrStub, CallFlavor, GraphFlattener, Insn, Kind, Operand, Register, ResKind, SSARepr,
    TLabel, slot_for_call_flavor,
};

// ---------------------------------------------------------------------------
// RPython: codewriter/flatten.py KINDS = ['int', 'ref', 'float']
// ---------------------------------------------------------------------------

/// Python `var_num` → flat index into the `locals_cells_stack_w`
/// virtualizable array.
///
/// PyFrame lays out locals, cells, and the value stack in a single
/// vector; `var_num` from `LOAD_FAST`/`STORE_FAST` is already a direct
/// offset into that vector (no indirection).
///
/// Identity remap of local-index → vable-array slot.
/// `jtransform.py:1877` `do_fixed_list_getitem` / `:1898`
/// `do_fixed_list_setitem` derive the index implicitly from the
/// `_virtualizable_` slot order on the W_Root subclass. Pyre's
/// `PyFrame` lays locals first in `locals_cells_stack_w` so the remap
/// is identity today; the indirection isolates the upstream rewrite
/// step in case the layout ever diverges.
#[inline]
fn local_to_vable_slot(var_num: usize) -> usize {
    var_num
}

/// Re-export of `pyre_jit_trace::pyjitcode::portal_red_pre_regalloc_slots`
/// so the codewriter pipeline shares the same formula with the
/// portal-bridge install path in `canonical_bridge.rs`.  See the
/// definition site for the `interp_jit.py:67 reds = ['frame', 'ec']`
/// rationale.
use pyre_jit_trace::pyjitcode::portal_red_pre_regalloc_slots;

#[inline]
fn entry_arg_slots(code: &CodeObject) -> usize {
    let mut argcount = code.arg_count as usize + code.kwonlyarg_count as usize;
    if code.flags.contains(CodeFlags::VARARGS) {
        argcount += 1;
    }
    if code.flags.contains(CodeFlags::VARKEYWORDS) {
        argcount += 1;
    }
    argcount
}

fn entry_inputargs(code: &CodeObject) -> Vec<super::flow::FlowValue> {
    (0..entry_arg_slots(code))
        .map(|idx| {
            super::flow::Variable::new(super::flow::VariableId(idx as u32), Kind::Ref).into()
        })
        .collect()
}

fn portal_graph_inputvars(code: &CodeObject) -> (super::flow::Variable, super::flow::Variable) {
    let base = entry_arg_slots(code) as u32;
    (
        super::flow::Variable::new(super::flow::VariableId(base), Kind::Ref),
        super::flow::Variable::new(super::flow::VariableId(base + 1), Kind::Ref),
    )
}

/// Which extra red inputargs a per-code jitcode's startblock carries
/// beyond the Python function arguments.
///
/// Every per-code jitcode threads the universal `self` red frame: each
/// opcode-handler graph takes `self` (the interpreter frame) as
/// `inputargs[0]`, and pyre's one-jitcode-per-code-object model carries
/// that as a single red frame inputarg (the "1-red-arg frame shape",
/// `driver_descriptor`).  The portal graph additionally carries the
/// JitDriver's `ec` red (`jitdriver_sd.reds = [frame, ec]`,
/// interp_jit.py:64-69) — `ec` is portal-specific, `frame` is not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FrameInputs {
    /// No frame, no ec — test-only shadow graphs.
    None,
    /// Non-portal callee jitcode: the universal `self` red frame only.
    Frame,
    /// Portal graph: frame + ec.
    Portal,
}

impl FrameInputs {
    fn has_frame(self) -> bool {
        matches!(self, FrameInputs::Frame | FrameInputs::Portal)
    }

    fn has_ec(self) -> bool {
        matches!(self, FrameInputs::Portal)
    }
}

fn graph_entry_inputargs(
    code: &CodeObject,
    frame_inputs: FrameInputs,
) -> Vec<super::flow::FlowValue> {
    let mut inputargs = entry_inputargs(code);
    let (frame, ec) = portal_graph_inputvars(code);
    if frame_inputs.has_frame() {
        inputargs.push(frame.into());
    }
    if frame_inputs.has_ec() {
        inputargs.push(ec.into());
    }
    inputargs
}

fn portal_graph_inputvars_from_startblock(
    graph: &super::flow::FunctionGraph,
) -> (super::flow::Variable, super::flow::Variable) {
    let startblock = graph.startblock.borrow();
    let len = startblock.inputargs.len();
    assert!(
        len >= 2,
        "portal graph startblock missing frame/ec inputargs"
    );
    let frame = match &startblock.inputargs[len - 2] {
        super::flow::FlowValue::Variable(variable) => *variable,
        other => panic!("portal graph frame inputarg must be Variable, got {other:?}"),
    };
    let ec = match &startblock.inputargs[len - 1] {
        super::flow::FlowValue::Variable(variable) => *variable,
        other => panic!("portal graph ec inputarg must be Variable, got {other:?}"),
    };
    (frame, ec)
}

fn flow_value_kind(value: &super::flow::FlowValue) -> Kind {
    match value {
        super::flow::FlowValue::Variable(variable) => variable
            .kind
            .expect("flow graph variable missing kind in jit_merge_point arg"),
        super::flow::FlowValue::Constant(constant) => constant
            .kind
            .expect("flow graph constant missing kind in jit_merge_point arg"),
    }
}

fn make_three_flow_lists(values: &[super::flow::FlowValue]) -> Vec<super::flow::SpaceOperationArg> {
    let mut ints = Vec::new();
    let mut refs = Vec::new();
    let mut floats = Vec::new();
    for value in values {
        match flow_value_kind(value) {
            Kind::Int => ints.push(value.clone()),
            Kind::Ref => refs.push(value.clone()),
            Kind::Float => floats.push(value.clone()),
        }
    }
    vec![
        super::flow::FlowListOfKind::new(Kind::Int, ints).into(),
        super::flow::FlowListOfKind::new(Kind::Ref, refs).into(),
        super::flow::FlowListOfKind::new(Kind::Float, floats).into(),
    ]
}

fn portal_jit_merge_point_graph_args(
    graph: &super::flow::FunctionGraph,
    next_instr: usize,
    pycode_var: super::flow::Variable,
    jitdriver_index: usize,
) -> Vec<super::flow::SpaceOperationArg> {
    let (frame, ec) = portal_graph_inputvars_from_startblock(graph);
    // `pypyjit/interp_jit.py:67-78` PyPyJitDriver greens =
    // `['next_instr', 'is_being_profiled', 'pycode']`.  `pycode` is
    // recovered from the live frame at every merge point via the
    // `getfield_vable_r frame, PYCODE_FIELD_IDX → pycode_var` dual-
    // write emitted immediately upstream in the walker; reference
    // that per-SpaceOp Variable here so the canonical
    // `flatten_graph` driver sees no unresolved `Opaque(Ref)`
    // constants.
    let greens = vec![
        super::flow::Constant::signed(next_instr as i64).into(),
        super::flow::Constant::signed(0).into(),
        pycode_var.into(),
    ];
    let reds = vec![frame.into(), ec.into()];
    let mut args = vec![super::flow::Constant::signed(jitdriver_index as i64).into()];
    args.extend(make_three_flow_lists(&greens));
    args.extend(make_three_flow_lists(&reds));
    args
}

fn frame_blocks_for_offset(code: &CodeObject, next_offset: usize) -> Vec<FrameBlock> {
    if next_offset >= code.instructions.len() {
        return Vec::new();
    }

    // `exception_table::decode_exceptiontable` yields byte offsets; pyre's
    // JIT codewriter tracks instruction-index offsets (`next_offset` is a
    // code-unit index into `code.instructions`), so divide by 2 at the
    // boundary.  Entries are emitted in ascending `start` order so we walk
    // the whole list rather than break early — multiple ranges may cover
    // the same PC (`pypy/interpreter/pycode.py:250-253` last-matching-wins).
    pyre_interpreter::exception_table::decode_exceptiontable(&code.exceptiontable)
        .filter_map(|entry| {
            let start = entry.start as usize / 2;
            let end = entry.end as usize / 2;
            if next_offset >= start && next_offset < end {
                Some(FrameBlock {
                    start_offset: start,
                    end_offset: end,
                    handler_offset: entry.target as usize / 2,
                    stack_depth: entry.depth as u16,
                    push_lasti: entry.lasti,
                })
            } else {
                None
            }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FrameBlock {
    start_offset: usize,
    end_offset: usize,
    handler_offset: usize,
    stack_depth: u16,
    push_lasti: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FrameState {
    /// `rpython/flowspace/framestate.py:20` `locals_w`.
    locals_w: Vec<Option<super::flow::FlowValue>>,
    /// `framestate.py:21` `stack`.
    stack: Vec<super::flow::FlowValue>,
    /// `framestate.py:22` `last_exception`.
    last_exception: Option<(super::flow::FlowValue, super::flow::FlowValue)>,
    /// `framestate.py:23` `blocklist`.
    blocklist: Vec<FrameBlock>,
    /// `framestate.py:24` `next_offset`.
    next_offset: usize,
    /// Graph-level red slots: the universal `self` frame Variable plus,
    /// for portal graphs, the JitDriver `ec` Variable, that flow through
    /// every block.  The first element (frame) is present on every
    /// non-portal callee jitcode (`FrameInputs::Frame`) as well as the
    /// portal (`FrameInputs::Portal`); the second element (`ec`) is
    /// `Some` only on the portal.  Populated on the entry FrameState
    /// (via `entry_frame_state(code, FrameInputs::Frame | Portal)`) and
    /// propagated through block transitions unchanged — these Variables
    /// carry graph-level identity, not per-block SSA names, so `copy()`
    /// passes them through without freshening and `union()` requires
    /// both sides to agree.  Mirrors the red carry-through in
    /// `rpython/jit/metainterp/warmspot.py` where the jitdriver_sd.reds
    /// list names `(jitframe, ec)` that the portal interpreter function
    /// threads through every iteration of the loop.  Participates in
    /// `mergeable()` after the last-exception pair so backedge
    /// `Link.args` produced by `getoutputargs()` stay aligned with the
    /// `startblock.inputargs` appended by `graph_entry_inputargs`.
    portal_extras: Option<(super::flow::FlowValue, Option<super::flow::FlowValue>)>,
}

impl FrameState {
    fn new(
        locals_w: Vec<Option<super::flow::FlowValue>>,
        stack: Vec<super::flow::FlowValue>,
        last_exception: Option<(super::flow::FlowValue, super::flow::FlowValue)>,
        blocklist: Vec<FrameBlock>,
        next_offset: usize,
    ) -> Self {
        Self {
            locals_w,
            stack,
            last_exception,
            blocklist,
            next_offset,
            portal_extras: None,
        }
    }

    /// Seed the graph-level `(frame, Option<ec>)` pair on this state.
    /// Called from `entry_frame_state(code, FrameInputs::Frame | Portal)`
    /// for the startblock state; every state derived from that entry
    /// state via `copy()` or `union()` preserves the same pair.  `ec` is
    /// `Some` only for portal graphs.
    fn with_portal_extras(
        mut self,
        extras: (super::flow::FlowValue, Option<super::flow::FlowValue>),
    ) -> Self {
        self.portal_extras = Some(extras);
        self
    }

    fn mergeable(&self) -> Vec<Option<super::flow::FlowValue>> {
        let mut data = self.locals_w.clone();
        data.extend(self.stack.iter().cloned().map(Some));
        if let Some((w_type, w_value)) = &self.last_exception {
            data.push(Some(w_type.clone()));
            data.push(Some(w_value.clone()));
        } else {
            data.push(Some(super::flow::Constant::none().into()));
            data.push(Some(super::flow::Constant::none().into()));
        }
        if let Some((frame, ec)) = &self.portal_extras {
            data.push(Some(frame.clone()));
            if let Some(ec) = ec {
                data.push(Some(ec.clone()));
            }
        }
        data
    }

    /// return the `mergeable()` position
    /// at which a given Variable appears, or `None` if it is not present.
    ///
    /// `framestate.py:38-43` `mergeable` concatenates `locals_w + stack +
    /// last_exc pair`; the i-th position is a stable per-FrameState slot
    /// identity that `Link.args` / `target.inputargs` correspondence is
    /// built on (see `getoutputargs` above — `link.args[j]` and
    /// `target.inputargs[j]` are both the j-th entry of their respective
    /// mergeable lists filtered for Variables).  The mergeable index
    /// translates to the concrete SSARepr register slot by folding in
    /// `nlocals` / `ncells` / `stack_base`; the pair (mergeable index of
    /// `link.args[j]` in source state, mergeable index of
    /// `target.inputargs[j]` in target state) drives the CFG-level
    /// coalescing that replaces pyre's SSARepr `*_copy` scanner
    /// (`regalloc.rs::coalesce_variables`).
    ///
    /// Match identity is by `VariableId` (Python object identity in
    /// RPython); constants and other FlowValue shapes are ignored.
    fn mergeable_index_of(&self, var: &super::flow::Variable) -> Option<usize> {
        self.mergeable().iter().position(
            |value| matches!(value, Some(super::flow::FlowValue::Variable(v)) if v.id == var.id),
        )
    }

    /// translate a `mergeable()` index
    /// (S1) into the SSARepr register slot that the walker emits for
    /// that FrameState position.
    ///
    /// Pyre's register layout packs fast locals and the operand stack
    /// contiguously as `[locals 0..nlocals][stack nlocals..nlocals+
    /// max_stackdepth]` (see `RegisterLayout::compute`: `stack_base =
    /// nlocals as u16`).  `FrameState.locals_w.len() == nlocals` and
    /// `FrameState.stack` is indexed from `0` at the bottom of the
    /// operand stack, so `mergeable[0..locals_w.len() + stack.len())`
    /// maps identity to the register slot.
    ///
    /// The final two `mergeable()` entries carry the `last_exception`
    /// pair (`framestate.py:23` `last_exception`) — these come from
    /// exception-edge wiring (`rpython/flowspace/flowcontext.py:1259`)
    /// rather than a regular FrameState slot, so they have no register
    /// and the function returns `None`.
    ///
    /// Cell / free variables (`pyframe::ncells`) live in the absolute
    /// virtualizable array between locals and stack, but pyre's
    /// register layout does NOT reserve register slots for them — see
    /// `RegisterLayout::stack_base_absolute = nlocals + ncells` (the
    /// runtime offset) vs `stack_base = nlocals` (the register-space
    /// offset).  Consumers that need the absolute PyFrame slot for a
    /// virtualizable access compute it separately.
    fn mergeable_index_to_slot(&self, merge_idx: usize) -> Option<u16> {
        let regular_len = self.locals_w.len() + self.stack.len();
        if merge_idx < regular_len {
            Some(merge_idx as u16)
        } else {
            None
        }
    }

    /// Convenience composition of S1 + S2: resolve a Variable to its
    /// SSARepr register slot in one call.  Returns `None` if the
    /// Variable does not appear in this FrameState or appears only in
    /// the `last_exception` pair.
    fn variable_slot(&self, var: &super::flow::Variable) -> Option<u16> {
        self.mergeable_index_of(var)
            .and_then(|idx| self.mergeable_index_to_slot(idx))
    }

    fn local_value_at(&self, reg: usize) -> Option<super::flow::FlowValue> {
        self.locals_w.get(reg).and_then(|v| v.clone())
    }

    fn store_local_value(&mut self, reg: usize, value: super::flow::FlowValue) {
        if let Some(slot) = self.locals_w.get_mut(reg) {
            *slot = Some(value);
        }
    }

    fn copy<F>(&self, fresh_variable: &mut F) -> Self
    where
        F: FnMut(Option<Kind>) -> super::flow::Variable,
    {
        Self {
            locals_w: self
                .locals_w
                .iter()
                .map(|value| copy_optional_flow_value(value.as_ref(), fresh_variable))
                .collect(),
            stack: self
                .stack
                .iter()
                .map(|value| copy_flow_value(value, fresh_variable))
                .collect(),
            last_exception: self.last_exception.as_ref().map(|(w_type, w_value)| {
                (
                    copy_flow_value(w_type, fresh_variable),
                    copy_flow_value(w_value, fresh_variable),
                )
            }),
            blocklist: self.blocklist.clone(),
            next_offset: self.next_offset,
            // Portal extras are graph-level identity — same Variables
            // across every FrameState in the graph.  Do not freshen.
            portal_extras: self.portal_extras.clone(),
        }
    }

    fn getvariables(&self) -> Vec<super::flow::FlowValue> {
        self.mergeable()
            .into_iter()
            .flatten()
            .filter(|value| matches!(value, super::flow::FlowValue::Variable(_)))
            .collect()
    }

    fn matches(&self, other: &Self) -> bool {
        assert_eq!(self.blocklist, other.blocklist);
        assert_eq!(self.next_offset, other.next_offset);
        let mergeable = self.mergeable();
        let other_mergeable = other.mergeable();
        if mergeable.len() != other_mergeable.len() {
            return false;
        }
        for (left, right) in mergeable.iter().zip(other_mergeable.iter()) {
            if left == right {
                continue;
            }
            if matches!(
                (left, right),
                (
                    Some(super::flow::FlowValue::Variable(_)),
                    Some(super::flow::FlowValue::Variable(_))
                )
            ) {
                continue;
            }
            return false;
        }
        true
    }

    fn union<F>(&self, other: &Self, fresh_variable: &mut F) -> Option<Self>
    where
        F: FnMut(Option<Kind>) -> super::flow::Variable,
    {
        if self.next_offset != other.next_offset
            || self.locals_w.len() != other.locals_w.len()
            || self.stack.len() != other.stack.len()
        {
            return None;
        }

        let locals_w = self
            .locals_w
            .iter()
            .zip(other.locals_w.iter())
            .map(|(left, right)| union_optional_flow_value(left, right, fresh_variable))
            .collect();
        let stack = self
            .stack
            .iter()
            .zip(other.stack.iter())
            .map(|(left, right)| union_flow_value(left, right, fresh_variable))
            .collect::<Option<Vec<_>>>()?;
        let last_exception = match (&self.last_exception, &other.last_exception) {
            (None, None) => None,
            (Some((left_type, left_value)), Some((right_type, right_value))) => Some((
                union_flow_value(left_type, right_type, fresh_variable)?,
                union_flow_value(left_value, right_value, fresh_variable)?,
            )),
            (Some((left_type, left_value)), None) => Some((
                union_flow_value(
                    left_type,
                    &super::flow::Constant::none().into(),
                    fresh_variable,
                )?,
                union_flow_value(
                    left_value,
                    &super::flow::Constant::none().into(),
                    fresh_variable,
                )?,
            )),
            (None, Some((right_type, right_value))) => Some((
                union_flow_value(
                    &super::flow::Constant::none().into(),
                    right_type,
                    fresh_variable,
                )?,
                union_flow_value(
                    &super::flow::Constant::none().into(),
                    right_value,
                    fresh_variable,
                )?,
            )),
        };
        // Portal extras carry graph-level identity; if the two sides
        // are both seeded they must reference the same `(frame, ec)`
        // Variables, otherwise the graph is malformed.  Both sides of a
        // union belong to the same graph, so they are uniformly seeded
        // (`FrameInputs::Frame`/`Portal`) or uniformly absent
        // (`FrameInputs::None`, test-only graphs).
        let portal_extras = match (&self.portal_extras, &other.portal_extras) {
            (None, None) => None,
            (Some(left), Some(right)) => {
                if left == right {
                    Some(left.clone())
                } else {
                    return None;
                }
            }
            _ => return None,
        };
        let mut merged = Self::new(
            locals_w,
            stack,
            last_exception,
            self.blocklist.clone(),
            self.next_offset,
        );
        merged.portal_extras = portal_extras;
        Some(merged)
    }

    fn getoutputargs(&self, targetstate: &Self) -> Vec<super::flow::FlowValue> {
        let mergeable = self.mergeable();
        let mut result = Vec::new();
        for (index, target_value) in targetstate.mergeable().iter().enumerate() {
            if matches!(target_value, Some(super::flow::FlowValue::Variable(_))) {
                result.push(
                    mergeable[index]
                        .clone()
                        .expect("target variable must correspond to a mergeable source value"),
                );
            }
        }
        result
    }
}

fn copy_optional_flow_value<F>(
    value: Option<&super::flow::FlowValue>,
    fresh_variable: &mut F,
) -> Option<super::flow::FlowValue>
where
    F: FnMut(Option<Kind>) -> super::flow::Variable,
{
    value.map(|value| copy_flow_value(value, fresh_variable))
}

fn copy_flow_value<F>(
    value: &super::flow::FlowValue,
    fresh_variable: &mut F,
) -> super::flow::FlowValue
where
    F: FnMut(Option<Kind>) -> super::flow::Variable,
{
    match value {
        super::flow::FlowValue::Variable(variable) => fresh_variable(variable.kind).into(),
        super::flow::FlowValue::Constant(constant) => constant.clone().into(),
    }
}

fn union_optional_flow_value<F>(
    left: &Option<super::flow::FlowValue>,
    right: &Option<super::flow::FlowValue>,
    fresh_variable: &mut F,
) -> Option<super::flow::FlowValue>
where
    F: FnMut(Option<Kind>) -> super::flow::Variable,
{
    match (left, right) {
        (Some(left), Some(right)) => union_flow_value(left, right, fresh_variable),
        (None, _) | (_, None) => None,
    }
}

fn union_flow_value<F>(
    left: &super::flow::FlowValue,
    right: &super::flow::FlowValue,
    fresh_variable: &mut F,
) -> Option<super::flow::FlowValue>
where
    F: FnMut(Option<Kind>) -> super::flow::Variable,
{
    if left == right {
        return Some(left.clone());
    }
    match (left, right) {
        (super::flow::FlowValue::Variable(left), super::flow::FlowValue::Variable(right)) => {
            Some(fresh_variable(union_kind(left.kind, right.kind)).into())
        }
        (
            super::flow::FlowValue::Variable(variable),
            super::flow::FlowValue::Constant(constant),
        )
        | (
            super::flow::FlowValue::Constant(constant),
            super::flow::FlowValue::Variable(variable),
        ) => Some(fresh_variable(union_kind(variable.kind, constant.kind)).into()),
        (super::flow::FlowValue::Constant(left), super::flow::FlowValue::Constant(right)) => {
            Some(fresh_variable(union_kind(left.kind, right.kind)).into())
        }
    }
}

fn union_kind(left: Option<Kind>, right: Option<Kind>) -> Option<Kind> {
    if left == right { left } else { None }
}

fn entry_frame_state(code: &CodeObject, frame_inputs: FrameInputs) -> FrameState {
    let inputargs = entry_inputargs(code);
    let mut locals_w = vec![None; code.varnames.len()];
    for (index, value) in inputargs.into_iter().enumerate() {
        if index < locals_w.len() {
            locals_w[index] = Some(value);
        }
    }
    let state = FrameState::new(
        locals_w,
        Vec::new(),
        None,
        frame_blocks_for_offset(code, 0),
        0,
    );
    let (frame, ec) = portal_graph_inputvars(code);
    match frame_inputs {
        FrameInputs::None => state,
        FrameInputs::Frame => state.with_portal_extras((frame.into(), None)),
        FrameInputs::Portal => state.with_portal_extras((frame.into(), Some(ec.into()))),
    }
}

#[derive(Debug)]
struct SpamBlock {
    /// `flowcontext.py:40` underlying `Block`.
    block: super::flow::BlockRef,
    /// `flowcontext.py:40` `block.framestate`.
    framestate: Option<FrameState>,
    /// `flowcontext.py:41` `block.dead`.
    dead: bool,
    /// Per-block ssarepr accumulator — pyre-side mirror of upstream
    /// `block.operations` recorded inside `record_block`
    /// (`flowcontext.py:407-416`).  Populated alongside the program-
    /// wide `ssarepr.insns` push so a post-walk `flatten_graph` can
    /// iterate `graph.iterblocks()` and consume the per-block emit
    /// sequence in graph-DFS order, matching
    /// `codewriter.py:53 flatten_graph(graph, regallocs, cpu=...)`.
    per_block_ssarepr: Vec<super::flatten::Insn>,
    /// Length of `per_block_ssarepr` at the moment the multi-pred
    /// trampoline fallthrough fallback first appended `goto + ---`
    /// (`emit_trampoline_for_multi_pred_link`).  Insns beyond this
    /// index are trampoline-tail synthetic, not walker-emitted block
    /// terminators.  Used by `rewrite_direct_terminator_tlabel` to
    /// cap its reverse scan so a sibling link's explicit-jump rewrite
    /// targets the original branch terminator instead of a previously
    /// appended fallthrough `goto TLabel(target)`.
    original_terminator_end: Option<usize>,
}

#[derive(Debug, Clone)]
struct SpamBlockRef(Rc<RefCell<SpamBlock>>);

impl SpamBlockRef {
    fn new(block: super::flow::BlockRef, framestate: Option<FrameState>) -> Self {
        Self(Rc::new(RefCell::new(SpamBlock {
            block,
            framestate,
            dead: false,
            per_block_ssarepr: Vec::new(),
            original_terminator_end: None,
        })))
    }

    fn block(&self) -> super::flow::BlockRef {
        self.0.borrow().block.clone()
    }

    fn framestate(&self) -> Option<FrameState> {
        self.0.borrow().framestate.clone()
    }

    fn set_framestate(&self, framestate: FrameState) {
        self.0.borrow_mut().framestate = Some(framestate);
    }

    fn mark_dead(&self) {
        let mut spam = self.0.borrow_mut();
        spam.dead = true;
        // `flowspace/flowcontext.py:455-457 mergeblock` runs the
        // tuple `block.dead = True; block.operations = ()` in one
        // step.  The empty-operations side carries the "this block
        // contributes no codegen" semantics that `flatten` /
        // `iterblocks` rely on: dead blocks remain enumerable as
        // forwarding stubs (predecessors that already named this
        // block as target follow the single recloseblock link
        // through it), but their `operations` is the empty tuple so
        // serialization yields nothing.  Pyre's per-block SSA
        // accumulator is the moral equivalent of `block.operations`
        // for the walker emit path, so clearing it here matches
        // upstream: dead blocks stay in `all_walker_blocks` and the
        // drain still enumerates them, but they yield no insns.
        spam.per_block_ssarepr.clear();
        // Mirror onto the underlying flow::Block so flatten_graph and
        // any post-walker graph traversal can see the dead status
        // without needing the SpamBlockRef wrapper (matching upstream
        // `flowcontext.py:42 SpamBlock.dead` which is read during
        // `flatten` via `block.dead` access on the flow Block).
        spam.block.borrow_mut().dead = true;
    }

    fn dead(&self) -> bool {
        self.0.borrow().dead
    }

    /// Push an `Insn` into the per-block accumulator.  Mirrors the
    /// per-op `block.operations.append(op)` line that
    /// `flowcontext.py:407-416 record_block` runs inside the recorder
    /// loop.  Walker emit macros call this alongside their program-
    /// wide `ssarepr.insns.push(...)` so the per-block shadow stays
    /// in sync with the production stream.
    fn push_insn(&self, insn: super::flatten::Insn) {
        self.0.borrow_mut().per_block_ssarepr.push(insn);
    }

    /// Snapshot the per-block accumulator — used by verification
    /// probes and by the post-walk flatten driver to drain the
    /// per-block emit sequence in graph-DFS order.
    fn per_block_ssarepr(&self) -> Vec<super::flatten::Insn> {
        self.0.borrow().per_block_ssarepr.clone()
    }

    /// Length of the per-block accumulator without cloning.  Used by
    /// the walker-time PC dispatch tracker to record per-PC `-live-`
    /// marker positions at emit time.
    fn per_block_ssarepr_len(&self) -> usize {
        self.0.borrow().per_block_ssarepr.len()
    }

    /// Move `other`'s per-block accumulator onto the end of `self`'s and clear
    /// `other` (issue #73 `remove_trivial_links` merge bridge).  Returns the
    /// index in `self` where `other`'s insns now begin so callers can re-point
    /// any `-live-` marker offsets recorded against `other`.  `self` and
    /// `other` must be distinct blocks.
    fn absorb_per_block_ssarepr(&self, other: &SpamBlockRef) -> usize {
        let base = self.0.borrow().per_block_ssarepr.len();
        let moved = std::mem::take(&mut other.0.borrow_mut().per_block_ssarepr);
        self.0.borrow_mut().per_block_ssarepr.extend(moved);
        base
    }

    /// Strip a trailing block-boundary `goto TLabel(<label>) + Unreachable`
    /// pair from this block's `per_block_ssarepr`, if the goto targets `label`.
    /// Returns whether a pair was removed.
    ///
    /// Used by the `remove_trivial_links` merge bridge: after absorbing
    /// `target`'s bytecode, the source's OLD boundary goto would otherwise sit
    /// in the middle of the merged stream, before the absorbed opcodes.
    /// `emit_link_renamings_into_block`'s `SourceBeforeTerminator` splice
    /// forward-scans for the FIRST terminator, so the absorbed link's renamings
    /// would land before `target`'s opcodes and read stale slots (codex P1,
    /// PR #127).  Removing the old boundary goto makes `target`'s own
    /// terminator the first one, so renamings splice after its opcodes.
    fn strip_trailing_boundary_goto(&self, label: &str) -> bool {
        let mut b = self.0.borrow_mut();
        let n = b.per_block_ssarepr.len();
        if n < 2 {
            return false;
        }
        let tail_unreachable = matches!(
            b.per_block_ssarepr[n - 1],
            super::flatten::Insn::Unreachable
        );
        let tail_goto = matches!(
            &b.per_block_ssarepr[n - 2],
            super::flatten::Insn::Op { opname, args, .. }
                if opname == "goto"
                    && args.len() == 1
                    && matches!(&args[0], super::flatten::Operand::TLabel(tl) if tl.name == label)
        );
        if tail_unreachable && tail_goto {
            b.per_block_ssarepr.truncate(n - 2);
            // Keep the original-terminator anchor within bounds.
            if let Some(end) = b.original_terminator_end {
                if end > b.per_block_ssarepr.len() {
                    b.original_terminator_end = Some(b.per_block_ssarepr.len());
                }
            }
            true
        } else {
            false
        }
    }
}

impl PartialEq for SpamBlockRef {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SpamBlockRef {}

impl std::hash::Hash for SpamBlockRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

/// Walker emit helper — pushes `insn` into `current_block`'s
/// per-block accumulator.  The program-wide `ssarepr.insns` is
/// populated post-walk via the drain swap at
/// `transform_graph_to_jitcode`'s end (matching `codewriter.py:53
/// flatten_graph(graph, regallocs, cpu)`).  Every walker emit site
/// uniformly routes through this helper — no direct
/// `ssarepr.insns.push` calls remain in production.
fn push_walker_emit(current_block: &SpamBlockRef, insn: super::flatten::Insn) {
    current_block.push_insn(insn);
}

/// `flatten.py:333` `self.emitline('%s_copy' % kind, v, "->", w)` emits a
/// register-to-register move as `ref_copy/r>r`.  Identity copies are
/// dropped — same reg on both sides is a no-op at runtime and at regalloc
/// time, so callers freely route a producer directly into its stack-slot
/// register without inserting a redundant byte.  Walker MUST NOT record a
/// graph-side `ref_copy` op; `insert_renamings` (flatten.py:320) owns
/// emission.
fn emit_ref_copy(current_block: &SpamBlockRef, dst: u16, src: u16) {
    if dst != src {
        let insn = Insn::op_with_result(
            "ref_copy",
            vec![Operand::reg(Kind::Ref, src)],
            Register::new(Kind::Ref, dst),
        );
        push_walker_emit(current_block, insn);
    }
}

fn emit_vable_getfield_ref_walker_only(
    current_block: &SpamBlockRef,
    vable_reg: u16,
    dst: u16,
    field_idx: u16,
) {
    let insn = Insn::op_with_result(
        "getfield_vable_r",
        vec![
            Operand::reg(Kind::Ref, vable_reg),
            Operand::descr_vable_static_field(field_idx),
        ],
        Register::new(Kind::Ref, dst),
    );
    push_walker_emit(current_block, insn);
}

/// #230.M2 measurement helper: bucket an SSARepr insn stream into
/// `(genuine, live, copy, structural)` so the `PYRE_FLATTEN_MEASURE`
/// survey can attribute the canonical-vs-walker length gap to a
/// specific category.  `live` = `-live-` markers (per-PC, retired by
/// #287), `copy` = `*_copy` renamings (walker emits inline where
/// upstream defers them to `insert_renamings`, #289/#290),
/// `structural` = `Label`/`---`, `genuine` = everything else.  Used on
/// both the canonical and walker streams so the comparison is
/// category-to-category.  Measurement-only; no production caller when
/// the gate is unset.
fn bucket_measure_insns(insns: &[super::flatten::Insn]) -> (usize, usize, usize, usize) {
    let (mut genuine, mut live, mut copy, mut structural) = (0usize, 0usize, 0usize, 0usize);
    for insn in insns {
        if insn.is_live() {
            live += 1;
        } else {
            match insn {
                super::flatten::Insn::Label(_) | super::flatten::Insn::Unreachable => {
                    structural += 1
                }
                super::flatten::Insn::Op { opname, .. } if opname.ends_with("_copy") => copy += 1,
                super::flatten::Insn::Op { .. } => genuine += 1,
            }
        }
    }
    (genuine, live, copy, structural)
}

/// #230.M3 measurement helper: tally the `genuine`-category opnames of an
/// SSARepr insn stream — the same classification `bucket_measure_insns`
/// counts as `genuine` (non-`-live-`, non-`*_copy` `Insn::Op`; `Label`/
/// `Unreachable` are not `Op` so they fall out of the `if let`).  Run on
/// both the canonical and walker streams so the per-graph `genuine` count
/// residual #230.M2 left unattributed (int +7, fannkuch +5) is named
/// opname-by-opname.  `BTreeMap` keeps the keys ordered so the logged diff
/// is deterministic.  Measurement-only; no production caller when the gate
/// is unset.
fn genuine_opname_tally(
    insns: &[super::flatten::Insn],
) -> std::collections::BTreeMap<String, usize> {
    let mut tally = std::collections::BTreeMap::new();
    for insn in insns {
        if insn.is_live() {
            continue;
        }
        if let super::flatten::Insn::Op { opname, .. } = insn {
            if opname.ends_with("_copy") {
                continue;
            }
            *tally.entry(opname.clone()).or_insert(0) += 1;
        }
    }
    tally
}

/// #288 build-side liveness resolver: derive, from an SSARepr's OWN
/// `-live-` markers + `pc_first_insn_pos`, the per-Python-PC PRE-MERGE
/// `-live-` marker index that `compute_liveness_with_pc_anchors` consumes
/// — the role today filled by the walker's dense per-PC
/// `walker_tracked_pc_live_indices`.  Resolution is nearest-`-live-`-at-
/// or-before keyed by each PC's first insn position, so a stream with
/// SPARSE markers (canonical: one `-live-` per canraise / before
/// `goto_if_not`+`switch`, not one per PC — see #282/#116) reconstructs
/// the same dense feed.  This is the prerequisite resolver for #287 to
/// stop emitting a per-PC `-live-` while the runtime keeps its dense
/// `pc_map`; #287 will wire it as the `None` branch of
/// `filter_liveness_in_place`.
///
/// Returns one entry per Python PC `0..n_pcs`: `Some(idx)` = the resolved
/// pre-merge marker insn index; `None` = either the PC carries no insn of
/// its own (absent from `pc_first_insn_pos`) or no `-live-` marker
/// precedes its position.  Absent PCs are deliberately NOT forward-filled
/// from the preceding PC: a `jit_merge_point` loop-header PC is not
/// structurally enumerable from the stream (the #286 gap), and masking it
/// behind a neighbour's marker would hide exactly the resume target #286
/// must still resolve separately.
fn derive_pc_live_indices_from_sparse(
    ssarepr: &super::flatten::SSARepr,
    n_pcs: usize,
) -> Vec<Option<usize>> {
    // py_pc -> its own first insn position (sparse; `None` when the PC
    // emitted no op carrying its offset).
    let mut pos_for_pc: Vec<Option<usize>> = vec![None; n_pcs];
    for &(py_pc, pos) in &ssarepr.pc_first_insn_pos {
        if (0..n_pcs as i64).contains(&py_pc) {
            pos_for_pc[py_pc as usize] = Some(pos);
        }
    }
    // `-live-` marker positions in stream order (ascending by construction).
    let live_positions: Vec<usize> = ssarepr
        .insns
        .iter()
        .enumerate()
        .filter_map(|(idx, insn)| insn.is_live().then_some(idx))
        .collect();
    pos_for_pc
        .into_iter()
        .map(|pos| {
            let pos = pos?;
            // Index of the last `-live-` marker at-or-before `pos`.
            live_positions
                .partition_point(|&lp| lp <= pos)
                .checked_sub(1)
                .map(|i| live_positions[i])
        })
        .collect()
}

/// #281 verify: of the runtime's structural resume targets in a canonical
/// SSARepr — branch guards (`goto_if_not` / `goto_if_not_*` / `switch`,
/// each with a LEADING `-live-`, flatten.rs:1868-69 / 1971-72) and
/// can-raise calls (`residual_call_*` with a TRAILING `-live-`,
/// flatten.py:206-217) per #285 — measure how
/// `derive_pc_live_indices_from_sparse` covers them.  Returns
/// `(branch, branch_keyed, branch_fed, canraise, canraise_fed)`:
/// - `branch_keyed`: the resolver entry for the guard's OWN owner PC (the
///   py_pc whose `pc_first_insn_pos` range contains the guard op) equals
///   the guard's leading marker.  This is the strict invariant the runtime
///   branch resume needs — it queries `pc_map[orgpc]` = the guard's own pc
///   (#285: `resume_pc = orgpc` for a non-call guard), so when
///   `branch_keyed < branch` the pc_first_insn_pos keying picks the wrong
///   (earlier) marker for those guards and #281 must re-key the resolver to
///   the guard-op position.
/// - `branch_fed` / `canraise_fed`: the guard/raise marker appears in the
///   fed set (the resolver hands it to SOME pc).  Necessary (the marker
///   reaches the runtime table) but, for branches, weaker than `keyed` (it
///   may be keyed to the fallthrough pc instead of the guard pc).
/// - `(branch - branch_fed) + (canraise - canraise_fed)` = stranded resume
///   markers in NO resolver entry — the (b) safety number.  A stranded
///   marker is a structural resume point the sparse feed cannot surface
///   (e.g. a can-raise whose fallthrough pc is stack-only / `nopc`), so a
///   nonzero count names exactly the resume targets #281 must re-key or
///   #286 must resolve in the runtime.  Can-raise resume keys on
///   `fallthrough_pc` not the call's own pc (#285), so only `fed` is
///   reported for it here; the precise fallthrough-pc keying is #286
///   runtime work.  Measurement-only; no production caller when the gate is
///   unset.
fn resume_target_resolver_coverage(
    ssarepr: &super::flatten::SSARepr,
    live_indices: &[Option<usize>],
) -> (usize, usize, usize, usize, usize) {
    // Owner-pc lookup keyed on stream position: `pc_first_insn_pos` is built
    // first-wins as the stream grows, so its positions are already ascending;
    // re-key by position and sort defensively.  `owner_pc(q)` = the py_pc
    // whose first-insn position is the greatest at-or-before `q`.
    let mut pc_pos: Vec<(usize, usize)> = ssarepr
        .pc_first_insn_pos
        .iter()
        .filter(|&&(pc, _)| pc >= 0)
        .map(|&(pc, pos)| (pos, pc as usize))
        .collect();
    pc_pos.sort_unstable();
    let owner_pc = |q: usize| -> Option<usize> {
        pc_pos
            .partition_point(|&(pos, _)| pos <= q)
            .checked_sub(1)
            .map(|k| pc_pos[k].1)
    };
    // Fed marker set: every marker index the resolver hands to some pc.
    let fed: std::collections::HashSet<usize> = live_indices.iter().filter_map(|&x| x).collect();

    let n_pcs = live_indices.len();
    let (mut branch, mut branch_keyed, mut branch_fed) = (0usize, 0usize, 0usize);
    let (mut canraise, mut canraise_fed) = (0usize, 0usize);
    for (q, insn) in ssarepr.insns.iter().enumerate() {
        let super::flatten::Insn::Op { opname, .. } = insn else {
            continue;
        };
        if opname == "goto_if_not" || opname.starts_with("goto_if_not_") || opname == "switch" {
            // Leading `-live-` immediately precedes the guard.
            let Some(leading) = q.checked_sub(1).filter(|&i| ssarepr.insns[i].is_live()) else {
                continue;
            };
            branch += 1;
            if fed.contains(&leading) {
                branch_fed += 1;
            }
            if let Some(p) = owner_pc(q) {
                if p < n_pcs && live_indices[p] == Some(leading) {
                    branch_keyed += 1;
                }
            }
        } else if opname.starts_with("residual_call") {
            // Can-raise iff a trailing `-live-` immediately follows (the
            // `Insn::Op` carries its result inline, so the marker is q+1);
            // non-raising residual_calls have none.
            if ssarepr.insns.get(q + 1).is_some_and(|n| n.is_live()) {
                canraise += 1;
                if fed.contains(&(q + 1)) {
                    canraise_fed += 1;
                }
            }
        }
    }
    (branch, branch_keyed, branch_fed, canraise, canraise_fed)
}

/// color-budget Step A probe helper: per-Register color-budget violation
/// record `(insn_idx, opname, role, kind_str, color, num_colors)`.
/// Role distinguishes argument operand vs result register vs nested
/// `ListOfKind` element so downstream analysis can correlate the
/// violation with the canonical emit site responsible.
type Phase4ColorBudgetViolation = (usize, String, &'static str, &'static str, u16, u16);

/// The next-block label is recognised as `Insn::Label(L)`, matching
/// upstream's `flatten.py:116 self.emit(Label(block))` block / link /
/// catch-landing labels.  The corresponding `goto TLabel(...)` carries
/// the same name, so a single string-key match suffices.
///
/// **Mutates** each block's `Vec<Insn>` in place to drop the strip
/// tail; appends moved (not cloned) into the output `Vec`.
fn strip_walker_block_boundary_goto(
    blocks: &mut [Vec<super::flatten::Insn>],
) -> Vec<super::flatten::Insn> {
    let total_capacity: usize = blocks.iter().map(|b| b.len()).sum();
    let mut drained: Vec<super::flatten::Insn> = Vec::with_capacity(total_capacity);
    let n = blocks.len();
    for i in 0..n {
        // Scan forward past empty per-block accumulators (dead /
        // superseded blocks whose `mark_dead` cleared their insns —
        // `rpython/flowspace/flowcontext.py:455-457` clears
        // `block.operations` on supersede; pyre mirrors that on
        // `per_block_ssarepr`).  Without this, a PJIF/PJIT parent
        // whose boundary goto targets the supersede newblock's
        // py_pc would fail to strip because the immediate next
        // entry in `all_walker_blocks` is the now-empty dead block,
        // not the supersede newblock that holds the matching
        // `Label(pcN)` first insn.
        // `rpython/jit/codewriter/flatten.py:106-155 make_link` falls
        // through to the IMMEDIATE next emitted block by recursive
        // descent — never hops over an intervening block.  So compare
        // only against the leading-label cluster of the IMMEDIATE next
        // non-empty block.  Empty blocks (dead / superseded SpamBlocks
        // whose `mark_dead` cleared their `per_block_ssarepr` per
        // `flowspace/flowcontext.py:455-457`) emit nothing, so scan
        // past them to find the first block that contributes content.
        // Leading-label CLUSTER (not just `first()`) because a block may
        // carry multiple leading labels
        // (e.g. `Label("block{addr}")` + `Label("pcN")`); the goto may
        // target any one of them.
        let next_label_names: Vec<String> = (i + 1..n)
            .find(|&j| !blocks[j].is_empty())
            .map(|j| {
                blocks[j]
                    .iter()
                    .take_while(|insn| matches!(insn, super::flatten::Insn::Label(_)))
                    .filter_map(|insn| match insn {
                        super::flatten::Insn::Label(l) => Some(l.name.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let block_insns = &mut blocks[i];
        let len = block_insns.len();
        let strip_tail = if len >= 2 {
            match (&block_insns[len - 2], &block_insns[len - 1]) {
                (
                    super::flatten::Insn::Op { opname, args, .. },
                    super::flatten::Insn::Unreachable,
                ) if opname == "goto"
                    && args.len() == 1
                    && matches!(&args[0], Operand::TLabel(target)
                        if next_label_names.iter().any(|n| n == &target.name)) =>
                {
                    2
                }
                _ => 0,
            }
        } else {
            0
        };
        block_insns.truncate(len - strip_tail);
        drained.append(block_insns);
    }
    drained
}

fn fresh_variable_for_state(
    graph: &mut super::flow::FunctionGraph,
    kind: Option<Kind>,
) -> super::flow::Variable {
    match kind {
        Some(kind) => graph.fresh_variable(kind),
        None => graph.fresh_untyped_variable(),
    }
}

/// CFG-level Variable-pair collector — port of
/// `rpython/tool/algo/regalloc.py:79-96
/// RegAllocator.coalesce_variables`.
///
/// Iterates `graph.iterblocks()` → `block.exits` → paired
/// `(link.args[i], link.target.inputargs[i])` (matching upstream's
/// `for i, v in enumerate(link.args): self._try_coalesce(v,
/// link.target.inputargs[i])`).  Returns Variable pairs directly,
/// matching upstream's `_try_coalesce(v1, v2)` Variable-direct
/// shape.  Slot projection (where required by pyre's u16-keyed
/// SSARepr regalloc) happens at the consumer.
///
/// Filter: only Ref-kind pairs are emitted, matching the per-kind
/// gate inside `allocate_registers` (`regalloc.rs:670-677`).  Every
/// `FrameState.mergeable()` position in pyre holds a Ref-kind
/// Variable (locals, stack, last_exc pair), so Int / Float kinds
/// never produce CFG pairs in practice.
///
/// `last_exception` / `last_exc_value` link args are skipped per
/// `flatten.py:336-347 generate_last_exc` — those are emitted
/// separately and don't participate in coalesce.
fn collect_cfg_coalesce_pairs(
    graph: &super::flow::FunctionGraph,
) -> Vec<(super::flow::VariableId, super::flow::VariableId)> {
    let mut pairs: Vec<(super::flow::VariableId, super::flow::VariableId)> = Vec::new();
    for block in graph.iterblocks() {
        let block_borrow = block.borrow();
        for link_ref in &block_borrow.exits {
            let link_borrow = link_ref.borrow();
            let Some(target_ref) = link_borrow.target.clone() else {
                continue;
            };
            let target_borrow = target_ref.borrow();
            if link_borrow.args.len() != target_borrow.inputargs.len() {
                continue;
            }
            for (i, arg) in link_borrow.args.iter().enumerate() {
                let Some(src_value) = arg.as_ref() else {
                    continue;
                };
                let Some(src_variable) = src_value.as_variable() else {
                    continue;
                };
                let Some(dst_variable) = target_borrow.inputargs[i].as_variable() else {
                    continue;
                };
                if Some(src_variable.clone()) == link_borrow.last_exception
                    || Some(src_variable.clone()) == link_borrow.last_exc_value
                {
                    continue;
                }
                // `regalloc.py:99-100 _try_coalesce` predicates on
                // `consider_var(v) and consider_var(w)`; the Ref-kind
                // pass requires both endpoints to be Ref.
                if src_variable.kind != Some(Kind::Ref) || dst_variable.kind != Some(Kind::Ref) {
                    continue;
                }
                // `regalloc.py:98 _try_coalesce` rejects `v is w` before
                // any further work — an inputarg forwarded unchanged
                // through `link.args[i] == target.inputargs[i]` has
                // nothing to coalesce with itself.  Pyre's pre-merge
                // into union-find treats same-VariableId as a no-op,
                // but emitting the pair still costs a `find` lookup
                // per side; drop it at the source to match PyPy's
                // self-pair short-circuit.
                if src_variable.id == dst_variable.id {
                    continue;
                }
                pairs.push((src_variable.id, dst_variable.id));
            }
        }
    }
    pairs
}

/// Walker post-walk `insert_renamings` — port of
/// `rpython/jit/codewriter/flatten.py:306-334`.
///
/// `flatten.py:306-334 insert_renamings` runs after each `make_link`
/// call and emits `<kind>_copy/_push/_pop` ops for every link arg →
/// target inputarg pair whose post-regalloc colors differ.  Pyre's
/// walker emits SSARepr inline per Python PC and never runs
/// `insert_renamings` itself, so links whose link.arg / target.inputarg
/// map to different walker slots have no equivalent walker emission.
///
/// This helper closes the gap by running the same renaming logic
/// post-walk:
///   * Single-exit blocks: splice the link's renamings into the
///     source block's `per_block_ssarepr` BEFORE the terminator
///     (matches `flatten.py:154 self.insert_renamings(link)` which
///     runs BEFORE the recursive `make_bytecode_block(link.target)`).
///   * Multi-exit blocks (POP_JUMP_IF_*, canraise switches) with a
///     unique-predecessor target: splice renamings at the target's
///     entry (`TargetAfterAnchor`).
///   * Multi-exit blocks with a multi-predecessor target: emit a
///     per-link trampoline via
///     [`emit_trampoline_for_multi_pred_link`].  Mirrors upstream's
///     `Label(link) + insert_renamings(link)` per-link emission by
///     either rewriting the source terminator's TLabel onto a
///     synthetic SpamBlock that runs the renamings then jumps to the
///     original target (explicit-jump arm), or appending the
///     renamings + explicit goto to the source's per-block
///     accumulator (fall-through arm).
fn walker_post_walk_insert_renamings(
    graph: &mut super::flow::FunctionGraph,
    walker_slot_for_variable: &[Option<u16>],
    regallocs: &[super::regalloc::GraphAllocationResult; 3],
    all_walker_blocks: &mut Vec<SpamBlockRef>,
    walker_pc_live_marker_pos: &mut [Vec<(SpamBlockRef, usize)>],
) {
    // Variable → walker slot resolver: bridge first, then graph
    // regalloc fallback.  Shared regalloc instance (computed once
    // outside) guarantees same colors as the canonical driver →
    // byte-equivalent renamings.
    let get_color = |variable: &super::flow::Variable| -> u16 {
        if let Some(Some(slot)) = walker_slot_for_variable
            .get(variable.id.0 as usize)
            .copied()
        {
            return slot;
        }
        let kind = variable.kind.unwrap_or(Kind::Ref);
        regallocs[kind.index()]
            .coloring
            .get(&variable.id)
            .copied()
            .unwrap_or(u16::MAX)
    };

    // Reachability DFS from startblock — matches canonical's
    // `generate_ssa_form` recursive descent (flatten.py:103-104).
    // `graph.iterblocks()` may include unreachable / pyre-walker-only
    // blocks (supersede candidates that canonical doesn't visit via
    // `seen_blocks`); processing them would emit ref_copies for
    // Variables canonical never sees.
    let mut reachable: Vec<super::flow::BlockRef> = Vec::new();
    let mut stack: Vec<super::flow::BlockRef> = vec![graph.startblock.clone()];
    while let Some(b) = stack.pop() {
        if reachable.iter().any(|r| r == &b) {
            continue;
        }
        if b.borrow().dead {
            continue;
        }
        reachable.push(b.clone());
        for exit in &b.borrow().exits {
            if let Some(target) = exit.borrow().target.clone() {
                stack.push(target);
            }
        }
    }

    // Pre-compute in-degree per reachable block.  Multi-exit links
    // dispatch on in_degree of the link's target:
    //   * in_degree == 1: splice renamings into target's entry via
    //     `TargetAfterAnchor` (safe since only this edge reaches the
    //     target's entry label).
    //   * in_degree > 1: synthesize a per-link trampoline via
    //     `emit_trampoline_for_multi_pred_link` (mirroring
    //     `flatten.py:175-205` per-link `Label + insert_renamings`
    //     emission).  TargetAfterAnchor would otherwise mix
    //     renamings from multiple incoming edges at the same entry.
    let mut in_degree: Vec<(super::flow::BlockRef, usize)> = Vec::new();
    for b in &reachable {
        for exit in &b.borrow().exits {
            if let Some(t) = exit.borrow().target.clone() {
                if let Some(entry) = in_degree.iter_mut().find(|(r, _)| r == &t) {
                    entry.1 += 1;
                } else {
                    in_degree.push((t, 1));
                }
            }
        }
    }
    let in_degree_of = |b: &super::flow::BlockRef| -> usize {
        in_degree
            .iter()
            .find(|(r, _)| r == b)
            .map(|(_, n)| *n)
            .unwrap_or(0)
    };

    // Single-exit blocks: splice each link's renamings into the
    // source block's `per_block_ssarepr` BEFORE the terminator
    // (matches `flatten.py:154 self.insert_renamings(link)` running
    // before the recursive `make_bytecode_block(link.target)`).
    for block_ref in &reachable {
        if block_ref.borrow().exits.len() != 1 {
            continue;
        }
        let link_ref = block_ref.borrow().exits[0].clone();
        if let Some(shift) = emit_link_renamings_into_block(
            block_ref,
            &link_ref,
            &get_color,
            all_walker_blocks,
            SpliceSite::SourceBeforeTerminator,
        ) {
            shift_walker_pc_tracked_offsets(walker_pc_live_marker_pos, &shift);
        }
    }

    // Multi-exit blocks (POP_JUMP_IF_*, canraise switches): each
    // exit's renamings are per-edge.  Walker's `goto_if_not` / switch
    // ends the source block; canonical emits each link's renamings
    // between `Label(link)` and the target block's body
    // (`flatten.py:175-205 insert_exits`).  Pyre splice strategy:
    //
    //   * unique-predecessor target — splice renamings at target's
    //     entry via `TargetAfterAnchor` (safe because only the link's
    //     edge reaches the target's entry label);
    //   * multi-predecessor target — synthesize a per-link trampoline
    //     via [`emit_trampoline_for_multi_pred_link`].  Explicit-jump
    //     arm rewrites the source terminator's TLabel + appends a
    //     synthetic SpamBlock; fall-through arm appends renamings +
    //     explicit `goto` to the source's per-block accumulator.
    let mut trampoline_counter: usize = 0;
    for block_ref in &reachable {
        let exits = block_ref.borrow().exits.clone();
        if exits.len() < 2 {
            continue;
        }
        for link_ref in exits {
            let Some(target_ref) = link_ref.borrow().target.clone() else {
                continue;
            };
            if in_degree_of(&target_ref) > 1 {
                let outcome = emit_trampoline_for_multi_pred_link(
                    graph,
                    block_ref,
                    &link_ref,
                    &get_color,
                    all_walker_blocks,
                    &mut trampoline_counter,
                );
                match outcome {
                    TrampolineOutcome::Emitted { .. } | TrampolineOutcome::NoPairs => {}
                    TrampolineOutcome::RewriteFailed => {
                        let pairs = collect_distinct_renaming_pairs(&link_ref, &get_color);
                        let target_label = super::flatten::block_label_name(&target_ref);
                        let pair_strs: Vec<String> = pairs
                            .iter()
                            .map(|(src, dst, kind)| format!("{}:{}->{}", kind.as_str(), src, dst))
                            .collect();
                        panic!(
                            "walker_post_walk_insert_renamings: trampoline rewrite \
                             failed for graph={} target_label={} pairs=[{}] — \
                             source terminator carries no TLabel matching the \
                             target's block-identity name nor a SwitchDictDescr \
                             entry pointing at it",
                            graph.name,
                            target_label,
                            pair_strs.join(","),
                        );
                    }
                }
                continue;
            }
            if let Some(shift) = emit_link_renamings_into_block(
                block_ref,
                &link_ref,
                &get_color,
                all_walker_blocks,
                SpliceSite::TargetAfterAnchor,
            ) {
                shift_walker_pc_tracked_offsets(walker_pc_live_marker_pos, &shift);
            }
        }
    }
}

/// Port of `rpython/translator/simplify.py` `eliminate_empty_blocks`.
///
/// `simplify_graph` (`translator.py:55-56`) runs this right after
/// `build_flow`, so the empty forwarding blocks that `mergeblock`
/// supersede leaves behind (`flowcontext.py:455-463`:
/// `block.dead = True; block.operations = (); block.exitswitch = None;
/// recloseblock(Link(outputargs, newblock))`) are collapsed before any
/// later phase sees them.  Upstream redirects every predecessor link
/// THROUGH the forwarder: it rewrites the predecessor's `link.target`
/// to the forwarder's single successor and substitutes the args so the
/// values compose (`link.args = [v.replace(subst) for v in exit.args]`).
/// The inner `while` collapses chained forwarders.
///
/// ```py
/// def eliminate_empty_blocks(graph):
///     for link in list(graph.iterlinks()):
///         while not link.target.operations:
///             block1 = link.target
///             if block1.exitswitch is not None:
///                 break
///             if not block1.exits:
///                 break
///             exit = block1.exits[0]
///             assert block1 is not exit.target
///             subst = dict(zip(block1.inputargs, link.args))
///             link.args = [v.replace(subst) for v in exit.args]
///             link.target = exit.target
/// ```
///
/// Pyre's walker fuses graph-build and flatten and never calls
/// `simplify_graph`, so these forwarders survive and predecessors keep
/// emitting `goto TLabel(block<dead>)`; this pass restores the orthodox
/// collapse so predecessors point straight at the generalization (the
/// byte-stream `TLabel`s are caught by
/// [`rewrite_dead_forwarder_gotos`]).
///
/// Predicate parity: upstream tests `not link.target.operations`.  The
/// walker emits SSARepr into `per_block_ssarepr`, NOT `block.operations`
/// (which stays empty for every walker block), so `operations` cannot be
/// the predicate.  `mark_dead` (the supersede `operations=()` +
/// `recloseblock` step) sets `block.dead`, and supersede is the only
/// producer of the empty-forwarding shape, so `dead` is the faithful
/// pyre predicate.  The `exitswitch.is_none()` / `!exits.is_empty()`
/// guards still mirror upstream exactly.
pub(crate) fn eliminate_empty_blocks(graph: &super::flow::FunctionGraph) {
    for link_ref in graph.iterlinks() {
        loop {
            let Some(block1) = link_ref.borrow().target.clone() else {
                break;
            };
            let exit_link = {
                let b1 = block1.borrow();
                // RPython's predicate is `while not link.target.operations`
                // (stop when the target carries real ops).  `dead` is the
                // pyre proxy (see fn doc); a non-dead target stops the
                // collapse exactly like a non-empty one upstream.
                if !b1.dead {
                    break;
                }
                // A `dead` block is only ever a `mergeblock` supersede
                // forwarder (`flowcontext.py:455-463`): `operations=()`,
                // `exitswitch=None`, recloseblock'd to a single Link.
                // RPython reaches the same shape via `not operations` +
                // `exitswitch is None` (single exit by graph invariant)
                // + the `if not exits: break` returnblock guard.  Assert
                // it so a future change to the supersede shape fails here
                // loudly instead of silently leaving a dangling forwarder
                // goto for the assembler to trip over.
                assert!(
                    b1.exitswitch.is_none() && b1.exits.len() == 1,
                    "eliminate_empty_blocks: dead block {} is not a \
                     single-exit forwarder (exitswitch={:?}, exits={})",
                    super::flatten::block_label_name(&block1),
                    b1.exitswitch,
                    b1.exits.len(),
                );
                b1.exits[0].clone()
            };
            let (exit_args, exit_target) = {
                let exit = exit_link.borrow();
                match exit.target.clone() {
                    Some(target) => (exit.args.clone(), target),
                    None => break,
                }
            };
            assert!(
                block1 != exit_target,
                "eliminate_empty_blocks: the graph contains an empty infinite loop"
            );
            // `subst = dict(zip(block1.inputargs, link.args))`.  RPython
            // builds a dict here (`dict(zip(...))`) and the project's
            // other port of this pass (`majit-translate
            // translator/simplify.rs eliminate_empty_blocks`) uses the
            // same `HashMap<Variable, _>`, so this is a direct port of an
            // RPython dict — not a HashMap added for convenience.  Keep
            // the full `LinkArg` (`Option<FlowValue>`) value so the
            // substitution below can distinguish "inputarg with a
            // concrete actual" from "inputarg with a `None` actual".
            let inputargs = block1.borrow().inputargs.clone();
            let link_args = link_ref.borrow().args.clone();
            let mut subst: HashMap<super::flow::Variable, super::flow::LinkArg> = HashMap::new();
            for (inputarg, arg) in inputargs.iter().zip(link_args.iter()) {
                if let super::flow::FlowValue::Variable(v) = inputarg {
                    subst.insert(*v, arg.clone());
                }
            }
            // `link.args = [v.replace(subst) for v in exit.args]`.
            // RPython's `getoutputargs` yields a concrete value list, so
            // every inputarg has a concrete actual.  Every link feeding a
            // dead forwarder is built by `output_link` -> `Link::new`
            // (all `Some`), so a `None` actual for a substituted inputarg
            // would be a non-orthodox link shape — fail loudly rather
            // than silently emit a reference to the eliminated block's
            // inputarg (a dangling Variable).
            let new_args: Vec<super::flow::LinkArg> = exit_args
                .iter()
                .map(|arg| match arg {
                    Some(super::flow::FlowValue::Variable(v)) => match subst.get(v) {
                        Some(Some(value)) => Some(value.clone()),
                        Some(None) => panic!(
                            "eliminate_empty_blocks: dead forwarder inputarg \
                             {v:?} has no concrete predecessor actual"
                        ),
                        // Not a `block1` inputarg: a free var — leave it
                        // unchanged, matching RPython `replace` which
                        // keeps vars absent from `subst`.
                        None => arg.clone(),
                    },
                    _ => arg.clone(),
                })
                .collect();
            // `link.target = exit.target`; the outer `loop` continues to
            // collapse the new target if it is itself a forwarder.
            let mut link = link_ref.borrow_mut();
            link.args = new_args;
            link.target = Some(exit_target);
        }
    }
}

/// Companion to [`eliminate_empty_blocks`] for pyre's inline-emit
/// walker: rewrite the already-emitted terminator `TLabel`s that name a
/// collapsed dead forwarder so they target the surviving generalization.
///
/// The walker emits `goto` / `goto_if_not_*` / `switch` `TLabel`s by
/// block identity (`block<addr>`) inline DURING the walk — before the
/// `mergeblock` supersede that turns the target into a dead forwarder.
/// `eliminate_empty_blocks` only rewrites graph links, so the byte
/// stream still names the now-empty dead block.  This brings the emitted
/// SSARepr in line with the collapsed graph: each `TLabel(block<dead>)`
/// is retargeted to `block<live>` by following the same forwarder chain
/// `eliminate_empty_blocks` walked, so the renamings the post-walk pass
/// splices for the (collapsed) `predecessor -> generalization` link and
/// the emitted goto agree.
fn rewrite_dead_forwarder_gotos(all_walker_blocks: &[SpamBlockRef]) {
    // Follow a dead forwarder's single-exit chain to its surviving
    // (non-dead) target — mirrors the inner `while` of
    // `eliminate_empty_blocks`.
    fn resolve_forward(start: &super::flow::BlockRef) -> super::flow::BlockRef {
        let mut cur = start.clone();
        loop {
            let next = {
                let b = cur.borrow();
                if !b.dead || b.exitswitch.is_some() || b.exits.is_empty() {
                    None
                } else {
                    b.exits[0].borrow().target.clone()
                }
            };
            match next {
                Some(target) if target != cur => cur = target,
                _ => break,
            }
        }
        cur
    }

    // Dead forwarders number a handful per graph, so a flat
    // `Vec<(from, to)>` with linear lookup is the right shape — no
    // RPython dict stands behind this pyre-only byte-rewrite bridge, so
    // [[feedback-no-hashmap-ever]] applies.
    let mut remap: Vec<(String, String)> = Vec::new();
    for spam in all_walker_blocks {
        let block = spam.block();
        if !block.borrow().dead {
            continue;
        }
        let from = super::flatten::block_label_name(&block);
        let to = super::flatten::block_label_name(&resolve_forward(&block));
        if from != to {
            remap.push((from, to));
        }
    }
    if remap.is_empty() {
        return;
    }

    let retarget = |tl: &mut super::flatten::TLabel| {
        if let Some((_, to)) = remap.iter().find(|(from, _)| *from == tl.name) {
            tl.name = to.clone();
        }
    };
    for spam in all_walker_blocks {
        let mut borrow = spam.0.borrow_mut();
        for insn in borrow.per_block_ssarepr.iter_mut() {
            let super::flatten::Insn::Op { args, .. } = insn else {
                continue;
            };
            for arg in args.iter_mut() {
                match arg {
                    super::flatten::Operand::TLabel(tl) => retarget(tl),
                    super::flatten::Operand::Descr(descr_rc) => {
                        if let super::flatten::DescrOperand::SwitchDict(_) = descr_rc.as_ref() {
                            let descr_mut = std::rc::Rc::make_mut(descr_rc);
                            if let super::flatten::DescrOperand::SwitchDict(sw) = descr_mut {
                                for (_key, tl) in sw.labels.iter_mut() {
                                    retarget(tl);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Issue #73 `remove_trivial_links` merge bridge.
///
/// When `remove_trivial_links` merges `target` into `source`
/// (`source.recloseblock(*target.exits)`), `target` becomes unreachable in
/// `graph.iterblocks()`, but the walker has already emitted `target`'s Python
/// opcodes inline into its `per_block_ssarepr`.  The drain appends unreachable
/// blocks AFTER the DFS-ordered prefix, so without this bridge `target`'s insns
/// would emit at the wrong position behind `source`'s now-dangling
/// `goto TLabel(target)`.
///
/// Relocate each merged `target`'s accumulator to the end of its `source`'s, in
/// the merge (absorption) order, and re-point any per-PC `-live-` markers
/// recorded against `target` to their new `source` offset.  The surviving
/// stream keeps `source`'s block-boundary `goto TLabel(target) + ---`
/// immediately followed by `target`'s `Label` — a runtime-correct (redundant)
/// fall-through branch that the assembler patches normally (`target`'s label is
/// single-entry, so it is marked exactly once and never left unmarked).
fn rewrite_trivial_link_merges(
    merges: &[(super::flow::BlockRef, super::flow::BlockRef)],
    all_walker_blocks: &[SpamBlockRef],
    walker_pc_live_marker_pos: &mut [Vec<(SpamBlockRef, usize)>],
) {
    let find = |b: &super::flow::BlockRef| -> Option<SpamBlockRef> {
        all_walker_blocks
            .iter()
            .find(|spam| &spam.block() == b)
            .cloned()
    };
    for (source, target) in merges {
        let (Some(src_spam), Some(tgt_spam)) = (find(source), find(target)) else {
            continue;
        };
        if src_spam == tgt_spam {
            continue;
        }
        // Drop source's old boundary `goto target + ---` before appending, so
        // target's own terminator is the first terminator in the merged block
        // and the single-exit renaming splice lands after target's opcodes
        // (codex P1, PR #127).
        src_spam.strip_trailing_boundary_goto(&super::flatten::block_label_name(target));
        let base = src_spam.absorb_per_block_ssarepr(&tgt_spam);
        // Re-point `-live-` markers: (tgt_spam, off) -> (src_spam, base + off).
        for pc_entries in walker_pc_live_marker_pos.iter_mut() {
            for entry in pc_entries.iter_mut() {
                if entry.0 == tgt_spam {
                    entry.0 = src_spam.clone();
                    entry.1 += base;
                }
            }
        }
    }
}

/// Outcome of [`emit_trampoline_for_multi_pred_link`].  `Emitted`
/// carries the new trampoline SpamBlock plus the byte length of its
/// renaming body (excluding `Label` / trailing `goto`+`---`) so the
/// caller can update its diagnostic counters.  `NoPairs` means the
/// link's `link.args ↔ target.inputargs` mapping reduces to identity
/// after `get_color` (nothing to copy).  `RewriteFailed` means
/// [`rewrite_source_terminator_for_link`] could not locate a matching
/// `TLabel` — neither a direct `Insn::Op` arg nor a
/// `SwitchDictDescr._labels` entry pointed at the target's block
/// label.  In the explicit-jump path this would mean the source
/// terminator's TLabel was already rewritten by an earlier link
/// (multi-fanout from the same source) or that the source's exit
/// shape doesn't carry a TLabel for this link (fall-through arm —
/// handled in the helper's append fallback below).
enum TrampolineOutcome {
    Emitted { spam: SpamBlockRef, body_len: usize },
    NoPairs,
    RewriteFailed,
}

/// `flatten.py:175-205` per-link `Label(link)` + `insert_renamings`
/// emission ported as a synthetic trampoline SpamBlock for the
/// multi-predecessor case.
///
/// Walker terminators (`goto_if_not`, `goto_if_not_int_is_zero`,
/// `switch`) carry an `Operand::TLabel(block_label_name(target))`
/// pointing directly at the target block's identity label.  When the
/// target has multiple predecessors, upstream emits per-link renamings
/// between `Label(link)` and the target block body; pyre's walker
/// collapses link labels into the target's block label, so per-edge
/// renamings have no place to land.  This helper restores upstream's
/// per-link emission by:
///
/// 1. Collecting `(src_color, dst_color, kind)` pairs for the link
///    (same logic as [`emit_link_renamings_into_block`]).
/// 2. Allocating a unique trampoline label name.
/// 3. Rewriting the source SpamBlock's terminator TLabel from the
///    target's block-identity name to the trampoline name.
/// 4. Building a new SpamBlock holding
///    `Label(<trampoline>) ; <ref_copy/push/pop ops> ;
///     goto TLabel(<original target>) ; ---` and appending it to
///    `all_walker_blocks` (its position is after the DFS-matched
///    prefix because the synthetic flow::Block is not reached by
///    `graph.iterblocks()`).
fn emit_trampoline_for_multi_pred_link<F>(
    graph: &mut super::flow::FunctionGraph,
    source_block: &super::flow::BlockRef,
    link_ref: &super::flow::LinkRef,
    get_color: &F,
    all_walker_blocks: &mut Vec<SpamBlockRef>,
    trampoline_counter: &mut usize,
) -> TrampolineOutcome
where
    F: Fn(&super::flow::Variable) -> u16,
{
    let pairs = collect_distinct_renaming_pairs(link_ref, get_color);
    let body = build_renaming_insns(pairs);
    if body.is_empty() {
        return TrampolineOutcome::NoPairs;
    }
    let target_block = match link_ref.borrow().target.clone() {
        Some(t) => t,
        None => return TrampolineOutcome::NoPairs,
    };

    // Locate the source SpamBlock for the in-place terminator rewrite.
    let Some(source_spam) = all_walker_blocks
        .iter()
        .find(|s| !s.dead() && s.block() == *source_block)
        .cloned()
    else {
        return TrampolineOutcome::RewriteFailed;
    };
    let target_label = super::flatten::block_label_name(&target_block);
    let trampoline_name = format!("epsilon3_link_{}", *trampoline_counter);
    match rewrite_source_terminator_for_link(
        &source_spam,
        link_ref,
        &target_label,
        &trampoline_name,
    ) {
        TerminatorRewrite::Rewritten => {
            *trampoline_counter += 1;
            // Explicit-jump arm: the source's terminator (`goto_if_not`,
            // `goto_if_not_int_is_zero`, `switch`, ...) carried this
            // link's branch target.  Synthesize a new SpamBlock for the
            // trampoline so the rewritten terminator lands at
            // `Label(<trampoline>)`, runs the ref_copies, then jumps to
            // the original target.  The block has no graph reachability,
            // so the post-walk DFS reorder leaves it in append-order at
            // the tail of `all_walker_blocks`.
            let synthetic_block = graph.new_block(Vec::new());
            let trampoline_spam = SpamBlockRef::new(synthetic_block, None);
            trampoline_spam.push_insn(super::flatten::Insn::Label(super::flatten::Label::new(
                trampoline_name,
            )));
            let body_len = body.len();
            for insn in body {
                trampoline_spam.push_insn(insn);
            }
            trampoline_spam.push_insn(super::flatten::Insn::op(
                "goto",
                vec![super::flatten::Operand::TLabel(
                    super::flatten::TLabel::new(target_label),
                )],
            ));
            trampoline_spam.push_insn(super::flatten::Insn::Unreachable);
            all_walker_blocks.push(trampoline_spam.clone());
            return TrampolineOutcome::Emitted {
                spam: trampoline_spam,
                body_len,
            };
        }
        TerminatorRewrite::FallthroughOrDefault => {}
        TerminatorRewrite::Missing => return TrampolineOutcome::RewriteFailed,
    }

    // Fall-through arm fallback: when no terminator TLabel matched the
    // target's block label, the link is the fall-through arm of a
    // multi-exit source (`flatten.py:264 make_link(linktrue)` runs
    // immediately after `goto_if_not` and inlines the renamings before
    // the target block body).  Pyre's walker has no per-link `Label`
    // for the fall-through arm — execution drops past the terminator
    // straight into the next SpamBlock at byte-stream level.  Append
    // the renamings AFTER the source's terminator together with an
    // explicit `goto TLabel(<target>)` + `---` tail so the target is
    // reached deterministically regardless of post-walk DFS reorder.
    // `strip_walker_block_boundary_goto` elides the explicit goto when
    // the immediate next non-empty block opens with the target label.
    let body_len = body.len();
    let mut spam_borrow = source_spam.0.borrow_mut();
    // Codex P1 (PR #89): cap a future explicit-jump rewrite's reverse
    // scan at the pre-append tail so a sibling link to the same target
    // retargets the ORIGINAL branch terminator, not the goto we are
    // about to append here.  Record only on the first append — later
    // appends fall inside the already-tracked trampoline region.
    if spam_borrow.original_terminator_end.is_none() {
        spam_borrow.original_terminator_end = Some(spam_borrow.per_block_ssarepr.len());
    }
    let insns = &mut spam_borrow.per_block_ssarepr;
    for insn in body {
        insns.push(insn);
    }
    insns.push(super::flatten::Insn::op(
        "goto",
        vec![super::flatten::Operand::TLabel(
            super::flatten::TLabel::new(target_label),
        )],
    ));
    insns.push(super::flatten::Insn::Unreachable);
    drop(spam_borrow);
    TrampolineOutcome::Emitted {
        spam: source_spam,
        body_len,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminatorRewrite {
    Rewritten,
    FallthroughOrDefault,
    Missing,
}

/// Rewrite the source SpamBlock's terminator for one concrete
/// flowspace `Link`.  PyPy emits `TLabel(linkfalse)` for bool false
/// branches and `SwitchDictDescr._labels.append((key, TLabel(switch)))`
/// for switch cases (`flatten.py:240-304`); matching only the target
/// block label is not link-specific enough when multiple exits from
/// the same source converge on the same target block.
fn rewrite_source_terminator_for_link(
    source_spam: &SpamBlockRef,
    link_ref: &super::flow::LinkRef,
    target_label: &str,
    to: &str,
) -> TerminatorRewrite {
    let link = link_ref.borrow();
    if is_default_link_exitcase(&link.exitcase) {
        return TerminatorRewrite::FallthroughOrDefault;
    }
    match &link.llexitcase {
        Some(super::flow::FlowValue::Constant(super::flow::Constant {
            value: super::flow::ConstantValue::Bool(false),
            ..
        })) => {
            if rewrite_switch_dict_label_for_key(source_spam, 0, to) {
                TerminatorRewrite::Rewritten
            } else if rewrite_direct_terminator_tlabel(source_spam, target_label, to) {
                TerminatorRewrite::Rewritten
            } else {
                TerminatorRewrite::Missing
            }
        }
        Some(super::flow::FlowValue::Constant(super::flow::Constant {
            value: super::flow::ConstantValue::Bool(true),
            ..
        })) => {
            if rewrite_switch_dict_label_for_key(source_spam, 1, to) {
                TerminatorRewrite::Rewritten
            } else {
                TerminatorRewrite::FallthroughOrDefault
            }
        }
        Some(super::flow::FlowValue::Constant(super::flow::Constant {
            value: super::flow::ConstantValue::Signed(key),
            ..
        })) => {
            if rewrite_switch_dict_label_for_key(source_spam, *key, to) {
                TerminatorRewrite::Rewritten
            } else {
                TerminatorRewrite::Missing
            }
        }
        _ => {
            if rewrite_direct_terminator_tlabel(source_spam, target_label, to) {
                TerminatorRewrite::Rewritten
            } else {
                TerminatorRewrite::Missing
            }
        }
    }
}

fn is_default_link_exitcase(exitcase: &Option<super::flow::FlowValue>) -> bool {
    matches!(
        exitcase,
        Some(super::flow::FlowValue::Constant(super::flow::Constant {
            value: super::flow::ConstantValue::Str(value),
            ..
        })) if value == "default"
    )
}

/// Rewrite a direct branch target (`goto_if_not`, `goto`, exception
/// mismatch branches) from the target block label to the trampoline.
/// Reverse-scans so the terminator is considered before any earlier op.
///
/// Codex P1 (PR #89): when `original_terminator_end` is set, the
/// reverse scan stops there.  The fallthrough fallback in
/// [`emit_trampoline_for_multi_pred_link`] appends `body + goto
/// TLabel(target) + Unreachable` past that anchor; without the cap a
/// sibling link whose explicit-jump rewrite shares the same target
/// would retarget the appended fallthrough goto instead of the
/// original `goto_if_not`/`goto`/exception-mismatch branch terminator.
fn rewrite_direct_terminator_tlabel(source_spam: &SpamBlockRef, from: &str, to: &str) -> bool {
    let mut spam_borrow = source_spam.0.borrow_mut();
    let upper = spam_borrow
        .original_terminator_end
        .unwrap_or(spam_borrow.per_block_ssarepr.len());
    for insn in spam_borrow.per_block_ssarepr[..upper].iter_mut().rev() {
        if let super::flatten::Insn::Op { args, .. } = insn {
            for arg in args.iter_mut() {
                if let super::flatten::Operand::TLabel(tl) = arg {
                    if tl.name == from {
                        tl.name = to.to_string();
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Rewrite a `SwitchDictDescr._labels` entry by its PyPy switch key,
/// not by target label.  The key is the edge identity in
/// `flatten.py:294-298`; using the target label would conflate two
/// different switch cases that jump to the same block with different
/// renamings.  `Rc::make_mut` keeps the mutation local if a descr is
/// shared.
fn rewrite_switch_dict_label_for_key(source_spam: &SpamBlockRef, key: i64, to: &str) -> bool {
    let mut spam_borrow = source_spam.0.borrow_mut();
    for insn in spam_borrow.per_block_ssarepr.iter_mut().rev() {
        if let super::flatten::Insn::Op { args, .. } = insn {
            for arg in args.iter_mut() {
                if let super::flatten::Operand::Descr(descr_rc) = arg {
                    if let super::flatten::DescrOperand::SwitchDict(_) = descr_rc.as_ref() {
                        let descr_mut = std::rc::Rc::make_mut(descr_rc);
                        if let super::flatten::DescrOperand::SwitchDict(sw) = descr_mut {
                            for (label_key, tl) in sw.labels.iter_mut() {
                                if *label_key == key {
                                    tl.name = to.to_string();
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

/// `flatten.py:312-333 insert_renamings` per-kind emission.  Takes
/// the distinct-color `(src, dst, kind)` pairs from
/// [`collect_distinct_renaming_pairs`], groups them per kind in
/// flatten.py:316-318 order, and lowers each kind via
/// `reorder_renaming_list` (cycle-aware `<kind>_push` /
/// `<kind>_pop`).  Returns an empty vec when `pairs` is empty (the
/// caller treats this as "nothing to splice").
fn build_renaming_insns(pairs: Vec<(u16, u16, Kind)>) -> Vec<super::flatten::Insn> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let mut sorted = pairs;
    // `flatten.py:312 lst.sort(key=lambda(v, w): w.index)`.
    sorted.sort_by_key(|(_, dst, _)| *dst);
    // `flatten.py:316-318` group by kind; `[T; 3]` indexed by
    // `Kind::index()` per [[feedback-no-hashmap-ever]].
    let mut renamings: [(Vec<u16>, Vec<u16>); 3] = [
        (Vec::new(), Vec::new()),
        (Vec::new(), Vec::new()),
        (Vec::new(), Vec::new()),
    ];
    for (src, dst, kind) in sorted {
        let bucket = &mut renamings[kind.index()];
        bucket.0.push(src);
        bucket.1.push(dst);
    }
    // `flatten.py:319-333` per-kind emit via `reorder_renaming_list`.
    let mut emitted: Vec<super::flatten::Insn> = Vec::new();
    for &kind in &Kind::ALL {
        let (frm, to) = &renamings[kind.index()];
        if frm.is_empty() {
            continue;
        }
        for (src, dst) in majit_translate::jit_codewriter::flatten::reorder_renaming_list(frm, to) {
            match (src, dst) {
                (Some(src), Some(dst)) => {
                    emitted.push(super::flatten::Insn::op_with_result(
                        format!("{}_copy", kind.as_str()),
                        vec![super::flatten::Operand::reg(kind, src)],
                        super::flatten::Register::new(kind, dst),
                    ));
                }
                (Some(src), None) => {
                    emitted.push(super::flatten::Insn::op(
                        format!("{}_push", kind.as_str()),
                        vec![super::flatten::Operand::reg(kind, src)],
                    ));
                }
                (None, Some(dst)) => {
                    emitted.push(super::flatten::Insn::op_with_result(
                        format!("{}_pop", kind.as_str()),
                        Vec::new(),
                        super::flatten::Register::new(kind, dst),
                    ));
                }
                (None, None) => unreachable!(
                    "reorder_renaming_list never yields (None, None) per majit/flatten.rs"
                ),
            }
        }
    }
    emitted
}

/// `flatten.py:308-311 insert_renamings` pair extraction restricted
/// to collecting distinct-color pairs that would actually emit a
/// `<kind>_copy` / `<kind>_push` / `<kind>_pop`.  Skips
/// `last_exception` / `last_exc_value` (routed through
/// `generate_last_exc`, `flatten.py:336-347`) and pairs whose
/// `src_color == dst_color` (no-op copies).  Skips `is_final &&
/// exits.is_empty()` targets (returnblock/exceptblock) where the
/// walker's RETURN_VALUE / RAISE handlers already emit `*_return` /
/// `raise` referencing the source slot directly.
fn collect_distinct_renaming_pairs<F>(
    link_ref: &super::flow::LinkRef,
    get_color: &F,
) -> Vec<(u16, u16, Kind)>
where
    F: Fn(&super::flow::Variable) -> u16,
{
    let link_borrow = link_ref.borrow();
    let Some(target_ref) = link_borrow.target.clone() else {
        return Vec::new();
    };
    let target_borrow = target_ref.borrow();
    if link_borrow.args.len() != target_borrow.inputargs.len() {
        return Vec::new();
    }
    if target_borrow.is_final && target_borrow.exits.is_empty() {
        return Vec::new();
    }
    let mut pairs = Vec::new();
    for (i, arg) in link_borrow.args.iter().enumerate() {
        let Some(src_value) = arg.as_ref() else {
            continue;
        };
        let Some(src_variable) = src_value.as_variable() else {
            continue;
        };
        let Some(dst_variable) = target_borrow.inputargs[i].as_variable() else {
            continue;
        };
        if Some(src_variable) == link_borrow.last_exception
            || Some(src_variable) == link_borrow.last_exc_value
        {
            continue;
        }
        let src_color = get_color(&src_variable);
        let dst_color = get_color(&dst_variable);
        if src_color != dst_color {
            let kind = dst_variable.kind.unwrap_or(Kind::Ref);
            pairs.push((src_color, dst_color, kind));
        }
    }
    pairs
}

/// Apply a splice's offset shift to the walker-tracked PC anchor /
/// live-marker side-tables.  Called immediately after
/// `emit_link_renamings_into_block` reports a non-empty insertion so
/// any recorded (block_ref, offset) pair whose offset has been pushed
/// forward by the insertion stays consistent with the post-splice
/// `per_block_ssarepr` layout.
fn shift_walker_pc_tracked_offsets(
    walker_pc_live_marker_pos: &mut [Vec<(SpamBlockRef, usize)>],
    shift: &SpliceShift,
) {
    let SpliceShift {
        block_ref,
        insert_pos,
        count,
    } = shift;
    for entries in walker_pc_live_marker_pos.iter_mut() {
        for (block, offset) in entries.iter_mut() {
            if block == block_ref && *offset >= *insert_pos {
                *offset += *count;
            }
        }
    }
}

/// Description of a `Vec::insert` shift produced by
/// [`emit_link_renamings_into_block`]: `count` insns were inserted at
/// `insert_pos` inside `block_ref`'s `per_block_ssarepr`.
struct SpliceShift {
    block_ref: SpamBlockRef,
    insert_pos: usize,
    count: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SpliceSite {
    /// Splice into source block's `per_block_ssarepr` BEFORE the
    /// trailing terminator (goto / *_return).  Single-exit case.
    SourceBeforeTerminator,
    /// Splice into target block's `per_block_ssarepr` AFTER its
    /// leading Label / `-live-` scaffold.  Multi-exit case with
    /// unique-predecessor target.
    TargetAfterAnchor,
}

fn emit_link_renamings_into_block<F>(
    source_block: &super::flow::BlockRef,
    link_ref: &super::flow::LinkRef,
    get_color: &F,
    all_walker_blocks: &[SpamBlockRef],
    site: SpliceSite,
) -> Option<SpliceShift>
where
    F: Fn(&super::flow::Variable) -> u16,
{
    let target_ref = link_ref.borrow().target.clone()?;
    let pairs = collect_distinct_renaming_pairs(link_ref, get_color);
    let emitted = build_renaming_insns(pairs);
    if emitted.is_empty() {
        return None;
    }

    let splice_block = match site {
        SpliceSite::SourceBeforeTerminator => source_block.clone(),
        SpliceSite::TargetAfterAnchor => target_ref,
    };
    let Some(spam) = all_walker_blocks
        .iter()
        .find(|s| !s.dead() && s.block() == splice_block)
    else {
        return None;
    };
    let mut spam_borrow = spam.0.borrow_mut();
    let insns = &mut spam_borrow.per_block_ssarepr;
    let insert_pos = match site {
        SpliceSite::SourceBeforeTerminator => {
            // Forward-scan for the FIRST terminator op and splice
            // immediately before it.  The caller guards on
            // `block_ref.borrow().exits.len() == 1`, so the block has
            // exactly one link and therefore at most one terminator —
            // the first match is the terminator for the link we are
            // emitting renamings for, matching `flatten.py:154
            // insert_renamings(link)` which runs immediately before
            // `make_bytecode_block(link.target)` (whose first emit is
            // `goto Label(block)` when target is in `seen_blocks`).
            //
            // Trailing content after the terminator (next linear PC's
            // `Label + -live-` scaffold accumulated under
            // supersede / fall-through when the same SpamBlock spans
            // multiple Python PCs) stays AFTER the terminator — it is
            // not another terminator, so first-match cannot land
            // there.
            let terminators = [
                "goto",
                "int_return",
                "ref_return",
                "float_return",
                "void_return",
                "raise",
                "reraise",
            ];
            let mut pos = insns.len();
            for (i, insn) in insns.iter().enumerate() {
                if let super::flatten::Insn::Op { opname, .. } = insn {
                    if terminators.contains(&opname.as_str()) {
                        pos = i;
                        break;
                    }
                }
            }
            pos
        }
        SpliceSite::TargetAfterAnchor => {
            // Skip target's leading scaffold (Label / `-live-`).
            // Renamings land between the entry label and the first
            // semantic op so that runtime dispatch into the target
            // lands on the label, falls through the `-live-`
            // placeholder, then through the renamings before any
            // semantic op consumes the target.inputarg registers.
            let mut pos = 0;
            for insn in insns.iter() {
                match insn {
                    super::flatten::Insn::Label(_) => {
                        pos += 1;
                    }
                    super::flatten::Insn::Op { opname, .. } if opname == "-live-" => {
                        pos += 1;
                    }
                    _ => break,
                }
            }
            pos
        }
    };
    let count = emitted.len();
    for (offset, insn) in emitted.into_iter().enumerate() {
        insns.insert(insert_pos + offset, insn);
    }
    drop(spam_borrow);
    Some(SpliceShift {
        block_ref: spam.clone(),
        insert_pos,
        count,
    })
}

fn append_exit(block: &super::flow::BlockRef, link: super::flow::LinkRef) {
    append_exit_tagged(block, link, "append_exit");
}

fn append_exit_tagged(
    block: &super::flow::BlockRef,
    link: super::flow::LinkRef,
    _tag: &'static str,
) {
    link.borrow_mut().prevblock = Some(block.downgrade());
    block.borrow_mut().exits.push(link.clone());
}

fn output_link(
    source_state: &FrameState,
    target_state: &FrameState,
    target: super::flow::BlockRef,
) -> super::flow::LinkRef {
    let outputargs = source_state.getoutputargs(target_state);
    super::flow::Link::new(outputargs, Some(target), None).into_ref()
}

/// Build the `[w_type, w_value]` argument list for a Link targeting
/// `graph.exceptblock`.  Mirrors `flatten.py:161-162` —
/// `assert link.last_exception is not None; assert link.last_exc_value
/// is not None`.  Callers must have seeded `source_state.last_exception`
/// before emitting the link.
fn exceptblock_link_args(source_state: &FrameState) -> Vec<super::flow::FlowValue> {
    match &source_state.last_exception {
        Some((w_type, w_value)) => vec![w_type.clone(), w_value.clone()],
        None => panic!(
            "exceptblock edge requires materialized exception pair \
             (flatten.py:161-162 make_exception_link parity)"
        ),
    }
}

/// Allocate the fresh `(exc_type, exc_value)` Variable pair that
/// represents an exception edge's payload at the graph level.
///
/// vs `rpython/flowspace/flowcontext.py:1250-1261 Raise.nomoreblocks`:
/// RPython's flow analysis sees the Python
/// source form `raise SomeError("msg")` and builds an
/// `OperationException(w_type=Constant(SomeError), w_value=...)` from
/// which `Raise.nomoreblocks` projects `[w_exc.w_type, w_exc.w_value]`
/// as real trace-level values into the exception Link.  Pyre's tracer
/// is one level lower: the stack carries a SINGLE Ref value (the
/// exception instance, written into the exception slot
/// `stack_base + site.stack_depth` at the `raise` opcode), and the
/// exception type is extracted at runtime inside the `raise` opcode's
/// backend implementation (`ssa_emitter.rs emit_raise` + blackhole
/// handler).
/// There is no graph-level Variable that stands for "the type of the
/// raised value" because pyre's graph emission is driven by bytecode,
/// not by `raise`-statement source.  Synthesizing fresh Variables here
/// matches `flowcontext.py:133-143 guessexception` — the same
/// mechanism upstream itself uses on implicit exception edges, where
/// type/value are also not statically knowable.
///
/// The fresh pair is carried on the Link as BOTH `link.args` AND
/// `link.extravars` (see `exception_edge_extravars`), so the upstream
/// `flatten.py:163-164 make_exception_link` check `link.args ==
/// [link.last_exception, link.last_exc_value]` matches and the
/// pass-through `raise` / `reraise` emission path fires.  The payload
/// is structurally synthetic at the graph layer and becomes concrete
/// only when the backend `raise`/`reraise` opcode populates the
/// JitFrame's exception slots from the exception slot at runtime.
fn exception_edge_vars(
    graph: &mut super::flow::FunctionGraph,
) -> (super::flow::Variable, super::flow::Variable) {
    // `last_exception/>i` (Kind::Int) + `last_exc_value/>r` (Kind::Ref)
    // per `assembler.py:220`.  Matches the walker emit at
    // `codewriter.rs:4491` / `:4509` and the fixture etype/evalue kinds
    // at `flatten.rs:5495-5496` / `:7649-7650`.  Untyped Variables
    // crash the canonical SSARepr build's `regalloc_color` when an
    // exception edge propagates them into a colored slot
    // (raise_catch_loop reproducer).
    (
        graph.fresh_variable(Kind::Int),
        graph.fresh_variable(Kind::Ref),
    )
}

fn exception_landing_state(
    graph: &mut super::flow::FunctionGraph,
    source_state: &FrameState,
) -> FrameState {
    let (exc_type, exc_value) = exception_edge_vars(graph);
    let mut state = source_state.clone();
    state.last_exception = Some((exc_type.into(), exc_value.into()));
    state
}

/// `flowcontext.py:635-636` computes `w_type = op.type(w_value).eval(self)`
/// before `Raise.nomoreblocks` projects the explicit raise edge to
/// `[w_exc.w_type, w_exc.w_value]`.
///
/// pyre's production bytecode still emits a single `raise/r` opcode and
/// derives the exception type at runtime, but the shadow graph can still
/// mirror the upstream shape exactly: record a graph-level `type`
/// operation whose result becomes `link.args[0]`, and carry the actual
/// raised value as `link.args[1]`.
fn explicit_raise_exception_pair(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    raised_value: super::flow::FlowValue,
    offset: i64,
) -> (super::flow::FlowValue, super::flow::FlowValue) {
    let exc_type = emit_graph_op_with_result(
        graph,
        block,
        "type",
        vec![raised_value.clone().into()],
        Kind::Ref,
        offset,
    );
    (exc_type.into(), raised_value)
}

fn explicit_raise_state(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    source_state: &FrameState,
    raised_value: super::flow::FlowValue,
    offset: i64,
) -> FrameState {
    let mut state = source_state.clone();
    state.last_exception = Some(explicit_raise_exception_pair(
        graph,
        block,
        raised_value,
        offset,
    ));
    state
}

/// Extract the `(etype, evalue)` Variable pair from the edge state
/// produced by `explicit_raise_state` / `exception_landing_state`.
/// Mirrors the pattern used by `flowcontext.py:141-143` where the
/// freshly-created `last_exc` / `last_exc_value` Variables are both
/// placed into `link.args` AND attached to `link.extravars(...)`
/// (`model.py:127-129 Link.extravars`) — so downstream passes that
/// check `link.args == [link.last_exception, link.last_exc_value]`
/// (`flatten.py:163-164 make_exception_link`) can identify the edge
/// as a pass-through of the exception pair.
fn exception_edge_extravars(
    edge_state: &FrameState,
) -> (super::flow::Variable, super::flow::Variable) {
    let (w_type, w_value) = edge_state
        .last_exception
        .as_ref()
        .expect("exception edge state missing last_exception pair");
    let as_variable = |value: &super::flow::FlowValue| match value {
        super::flow::FlowValue::Variable(v) => *v,
        super::flow::FlowValue::Constant(_) => panic!(
            "exception edge last_exception carries Constant; extravars \
             expects Variables (flowcontext.py:130-134 guessexception)"
        ),
    };
    (as_variable(w_type), as_variable(w_value))
}

fn update_catch_landing_state(
    graph: &mut super::flow::FunctionGraph,
    target: &SpamBlockRef,
    edge_state: &FrameState,
) {
    // `flowcontext.py:130-139 guessexception` separates `vars`
    // (link.args, the link's `extravars` — `[last_exc,
    // last_exc_value]`) from `vars2` (target.inputargs, fresh
    // Variables on the EggBlock).  Both single-source and
    // multi-source paths must allocate fresh inputargs so the
    // landing block's `inputargs` Variable IDs are disjoint from
    // the link's outgoing args.  `copy()` is the upstream-canonical
    // shape — `framestate.py:80 copy(rename)` re-renames every
    // FlowValue Variable through the closure.
    let new_state = if let Some(existing) = target.framestate() {
        let mut fresh = |kind| fresh_variable_for_state(graph, kind);
        existing.union(edge_state, &mut fresh)
    } else {
        let mut fresh = |kind| fresh_variable_for_state(graph, kind);
        Some(edge_state.copy(&mut fresh))
    };
    // `flowcontext.py:139` `egg = EggBlock(vars2, block, case)` — the
    // catch landing's inputargs receive the exception edge's incoming
    // values.  Single-source case (above) keeps the alias today;
    // pyre's union path (multi-source raise sites flowing into the
    // same handler) merges via `existing.union(&candidate)` and the
    // inputargs become the union state's Variables.  Mirrors the
    // `target.inputargs = state.getvariables()` pattern at
    // `make_next_block` (codewriter.rs:932) and `initialize_spam_block`
    // (codewriter.rs:913).  When `union` returns `None` (incompatible
    // states), framestate and inputargs both stay at the existing
    // values.
    if let Some(state) = new_state {
        target.block().borrow_mut().inputargs = state.getvariables();
        target.set_framestate(state);
    }
}

fn handler_entry_state_from_catch_site(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    landing_state: &FrameState,
    site: &ExceptionCatchSite,
) -> FrameState {
    let mut state = landing_state.clone();
    sync_stack_state(graph, &mut state, site.stack_depth);
    if site.push_lasti {
        state.stack.push(fresh_ref_value(graph));
    }
    let exc_value = state
        .last_exception
        .as_ref()
        .map(|(_w_type, w_value)| w_value.clone())
        .unwrap_or_else(|| fresh_ref_value(graph));
    state.stack.push(exc_value);
    state.next_offset = site.handler_py_pc;
    state.blocklist = frame_blocks_for_offset(code, site.handler_py_pc);
    state
}

fn handler_entry_state_from_catch_sites(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    catch_sites: &[ExceptionCatchSite],
    handler_py_pc: usize,
) -> Option<FrameState> {
    let mut merged: Option<FrameState> = None;
    for site in catch_sites {
        if site.handler_py_pc != handler_py_pc {
            continue;
        }
        let Some(landing_state) = site.landing.framestate() else {
            continue;
        };
        let candidate = handler_entry_state_from_catch_site(code, graph, &landing_state, site);
        merged = Some(match merged {
            None => candidate,
            Some(existing) => {
                let mut fresh = |kind| fresh_variable_for_state(graph, kind);
                existing.union(&candidate, &mut fresh).unwrap_or(candidate)
            }
        });
    }
    merged
}

fn initialize_spam_block(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    target: &SpamBlockRef,
    source_state: &FrameState,
    next_offset: usize,
) -> FrameState {
    if let Some(state) = target.framestate() {
        return state;
    }

    let mut fresh = |kind| fresh_variable_for_state(graph, kind);
    let mut target_state = source_state.copy(&mut fresh);
    target_state.blocklist = frame_blocks_for_offset(code, next_offset);
    target_state.next_offset = next_offset;
    target.block().borrow_mut().inputargs = target_state.getvariables();
    target.set_framestate(target_state.clone());
    target_state
}

fn make_next_block(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    currentblock: &SpamBlockRef,
    currentstate: &FrameState,
    next_offset: usize,
    pendingblocks: &mut VecDeque<SpamBlockRef>,
    all_walker_blocks: &mut Vec<SpamBlockRef>,
) -> SpamBlockRef {
    let mut fresh = |kind| fresh_variable_for_state(graph, kind);
    let mut newstate = currentstate.copy(&mut fresh);
    newstate.blocklist = frame_blocks_for_offset(code, next_offset);
    newstate.next_offset = next_offset;
    let newblock = SpamBlockRef::new(graph.new_block(Vec::new()), Some(newstate.clone()));
    // Track every walker-created block in walker-visit order so the
    // post-walk drain can iterate per-block accumulators in the same
    // order their emits reached the program-wide `ssarepr.insns`.
    all_walker_blocks.push(newblock.clone());
    newblock.block().borrow_mut().inputargs = newstate.getvariables();
    append_exit(
        &currentblock.block(),
        output_link(currentstate, &newstate, newblock.block()),
    );
    // flowcontext.py:472 `self.pendingblocks.append(newblock)`.
    pendingblocks.push_back(newblock.clone());
    newblock
}

fn mergeblock(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    joinpoints: &mut VecMap<usize, Vec<SpamBlockRef>>,
    currentblock: &SpamBlockRef,
    currentstate: &FrameState,
    next_offset: usize,
    pendingblocks: &mut VecDeque<SpamBlockRef>,
    all_walker_blocks: &mut Vec<SpamBlockRef>,
) -> SpamBlockRef {
    // `flowcontext.py:426 candidates = self.joinpoints.setdefault(
    // next_offset, [])` — sparse-by-PC dict in upstream.  VecMap
    // preserves the sparse semantics (only PCs that actually carry
    // joinpoint candidates allocate an entry).
    let candidates = joinpoints.entry(next_offset).or_insert_with(Vec::new);
    for index in 0..candidates.len() {
        let block = candidates[index].clone();
        let block_state = block
            .framestate()
            .expect("joinpoint candidate must carry a FrameState");
        let mut fresh = |kind| fresh_variable_for_state(graph, kind);
        let Some(mut newstate) = block_state.union(currentstate, &mut fresh) else {
            continue;
        };
        if newstate.matches(&block_state) {
            append_exit(
                &currentblock.block(),
                output_link(currentstate, &newstate, block.block()),
            );
            // Pyre-only head-of-list promotion.  Upstream
            // `flowcontext.py:438-441` returns the matched block
            // directly; the surrounding pendingblocks queue carries
            // block objects so a PC-keyed joinpoint lookup never
            // happens.  Pyre's walker is PC-sequential and reads
            // "active block at PC N" through `joinpoints.get(&py_pc).
            // and_then(|blocks| blocks.iter().find(|b| !b.dead()))`.
            // The loop above allows `continue` on union-None, so a
            // match can land at `index > 0`; without this reorder the
            // next joinpoint lookup at `next_offset` would return a
            // sibling candidate instead of the one we just linked
            // into, and the walker would emit subsequent ops against
            // a different block's FrameState.  The supersede branch
            // and the fresh-path `candidates.insert(0, ...)` already
            // preserve the head-of-list invariant; the match branch
            // does the same.
            if index != 0 {
                candidates.remove(index);
                candidates.insert(0, block.clone());
            }
            return block;
        }

        for (name, value) in code.varnames.iter().zip(newstate.locals_w.iter_mut()) {
            if let Some(super::flow::FlowValue::Variable(variable)) = value.as_mut() {
                variable.rename(name);
            }
        }
        let newblock = SpamBlockRef::new(graph.new_block(Vec::new()), Some(newstate.clone()));
        // Record the supersede-newblock in walker-visit order so the
        // post-walk drain enumerates it.
        all_walker_blocks.push(newblock.clone());
        newblock.block().borrow_mut().inputargs = newstate.getvariables();
        append_exit(
            &currentblock.block(),
            output_link(currentstate, &newstate, newblock.block()),
        );

        // flowcontext.py:455-463 supersede.  The line-by-line port:
        //
        //     block.dead = True
        //     block.operations = ()
        //     block.exitswitch = None
        //     block.recloseblock(Link(outputargs, newblock))
        //     candidates.remove(block)
        //     candidates.insert(0, newblock)
        //     self.pendingblocks.append(newblock)
        //
        // Matches upstream: the supersede newblock IS
        // re-walked under widened inputargs.  The dead block's
        // walker accumulator is cleared by `mark_dead`, so the
        // drain skips it and only the newblock's bytes reach the
        // final SSARepr.
        block.mark_dead();
        block.block().borrow_mut().operations.clear();
        block.block().borrow_mut().exitswitch = None;
        let supersede_link = output_link(&block_state, &newstate, newblock.block());
        block.block().recloseblock(vec![supersede_link]);

        candidates.remove(index);
        candidates.insert(0, newblock.clone());
        // flowcontext.py:463 `self.pendingblocks.append(newblock)`.
        pendingblocks.push_back(newblock.clone());
        let _ = newstate;
        return newblock;
    }

    let newblock = make_next_block(
        code,
        graph,
        currentblock,
        currentstate,
        next_offset,
        pendingblocks,
        all_walker_blocks,
    );
    candidates.insert(0, newblock.clone());
    newblock
}

/// Rust `FlowValue` is statically kinded (Int/Ref/Float) and requires
/// `Kind::Ref` at construction. RPython `Variable()`
/// (`flowspace/model.py`) is unkinded — flowgraph variables carry no
/// type at construction; the annotator infers types in a later pass.
/// The 1-line wrapper exists only because pyre's `Kind::Ref` parameter
/// would otherwise repeat at every call site.
fn fresh_ref_value(graph: &mut super::flow::FunctionGraph) -> super::flow::FlowValue {
    graph.fresh_variable(Kind::Ref).into()
}

fn pop_ref_or_fresh(
    state: &mut FrameState,
    graph: &mut super::flow::FunctionGraph,
) -> super::flow::FlowValue {
    state.stack.pop().unwrap_or_else(|| fresh_ref_value(graph))
}

fn push_fresh_ref(state: &mut FrameState, graph: &mut super::flow::FunctionGraph) {
    state.stack.push(fresh_ref_value(graph));
}

fn pop_and_decr_depth(state: &mut FrameState, depth: &mut u16) {
    let _ = state.stack.pop();
    *depth = depth.saturating_sub(1);
}

fn null_stack_sentinel() -> super::flow::FlowValue {
    // CPython's PUSH_NULL / LOAD_GLOBAL(push_null) stack marker.  The
    // runtime side emits `PY_NULL = 0` via `emit_pushvalue_ref_const!`;
    // the symbolic side carries a `Constant(None, Ref)` so any graph
    // SpaceOp that observes the sentinel (e.g. CALL after a
    // LOAD_ATTR(method) push when the walker can't statically prove
    // the slot is unused) lowers cleanly via `flatten_constant_operand`
    // (`(None, Ref) → ConstRef(0)`) without an `Opaque` detour.
    super::flow::Constant::none().into()
}

fn pyobject_const_ref_value(w_obj: pyre_object::PyObjectRef) -> super::flow::FlowValue {
    super::flow::Constant::new(
        super::flow::ConstantValue::Signed(w_obj as i64),
        Some(Kind::Ref),
    )
    .into()
}

fn duplicate_shadow_tos(
    graph: &mut super::flow::FunctionGraph,
    state: &mut FrameState,
) -> super::flow::FlowValue {
    // CPython/PyPy stack DUP/COPY semantics preserve the exact top-of-stack
    // value identity.  When the walker's shadow stack is temporarily out of
    // sync, fall back to a fresh Ref variable instead of panicking so the
    // compile can continue, but keep the normal path as a clone of TOS.
    let duplicated = state
        .stack
        .last()
        .cloned()
        .unwrap_or_else(|| fresh_ref_value(graph));
    state.stack.push(duplicated.clone());
    duplicated
}

/// Transitional dual-write.  `rpython/jit/codewriter/codewriter.py:44-67`
/// runs `perform_register_allocation(graph) → flatten_graph(graph) →
/// compute_liveness(ssarepr) → assemble(ssarepr)`.  Upstream has **one**
/// IR stream — the flow graph — which `flatten_graph` lowers into an
/// `SSARepr`.
///
/// Pyre historically emitted `SSARepr` directly from the trace recorder
/// and skipped the flow-graph stage.  Pyre reintroduces the graph
/// (so CFG-level `regalloc.py:79-96 coalesce_variables` can run), but the
/// SSARepr emission has not yet been replaced with a `flatten_graph` pass
/// .  Until it is, each opcode handler must populate both
/// streams — the SSARepr byte stream that backend/blackhole consume, and
/// the graph that `RegAllocator` consumes.
///
/// Delete this helper once the SSARepr stream is
/// generated from the graph by `flatten_graph`.
fn record_graph_op(
    block: &super::flow::BlockRef,
    opname: impl Into<String>,
    args: Vec<super::flow::SpaceOperationArg>,
    result: Option<super::flow::FlowValue>,
    offset: i64,
) -> super::flow::SpaceOperation {
    let op = super::flow::SpaceOperation::new(opname, args, result, offset);
    super::flow::push_op(block, op.clone());
    op
}

/// Production-flip bridge: pair a graph dual-write's result
/// `Variable` with the SSARepr slot the walker assigned in the same
/// emit.  No-op when `var` is `None` (residual_call with `ResKind::Void`,
/// non-portal CodeWriter, etc.).
///
/// **Overwrite semantics**: when the same `Variable` is paired across
/// multiple emit sites (e.g. an inputarg `Variable` initially paired
/// with its parameter slot, then re-paired with a stack slot when a
/// `LOAD_FAST` push lands the same `Variable` on the stack), the most
/// recent pairing wins.  `walker_slot_for_variable[v.id]` therefore
/// reflects "the slot this Variable currently lives in" at the walk
/// point of the last `pair_walker_slot` call — not "the Variable's
/// home slot".  Use [`pair_walker_slot_if_absent`] when the caller
/// wants first-write-wins instead.
fn pair_walker_slot(
    table: &mut Vec<Option<u16>>,
    var: Option<super::flow::Variable>,
    walker_slot: u16,
) {
    if let Some(v) = var {
        let idx = v.id.0 as usize;
        if table.len() <= idx {
            table.resize(idx + 1, None);
        }
        table[idx] = Some(walker_slot);
    }
}

/// First-wins variant of `pair_walker_slot`.  Used by the post-walk
/// FrameState pairing pass that re-iterates every block's mergeable to
/// seed inputarg Variables; per-PC emit sites have already paired the
/// Variables they directly produce, and that pairing reflects the
/// register slot the walker actually wrote into.  A FrameState-derived
/// slot for the same Variable may differ when the Variable flows through
/// a stack push/pop that lands it at a non-canonical slot temporarily —
/// the per-PC pairing must take precedence so canonical's `get_register`
/// resolves to the slot the walker emit chose.
fn pair_walker_slot_if_absent(
    table: &mut Vec<Option<u16>>,
    var: Option<super::flow::Variable>,
    walker_slot: u16,
) {
    if let Some(v) = var {
        let idx = v.id.0 as usize;
        if table.len() <= idx {
            table.resize(idx + 1, None);
        }
        if table[idx].is_none() {
            table[idx] = Some(walker_slot);
        }
    }
}

/// Build the 5-arg `setarrayitem_vable_r` arg vector matching
/// `rpython/jit/codewriter/jtransform.py:1898-1906 do_fixed_list_setitem`
/// (vable branch): `[v_base, v_index, v_value, arrayfielddescr,
/// arraydescr]`. `v_base` is the portal frame Variable produced by
/// `portal_graph_inputvars(code).0` — matching jtransform.py:840 where
/// the JIT driver's red `frame` arg is threaded into every vable op
/// from the start. The trailing two operands are the
/// `vable_array_field_descr` / `vable_array_descr` singletons from
/// `majit_ir::descr` (matching `virtualizable.py:73,58` 1:1 — Arc
/// identity is preserved across calls so `flatten_descr_by_ptr`
/// resolves them via `Arc::ptr_eq`).
///
/// Pyre's PyFrame has a single virtualizable array
/// (`locals_cells_stack_w`) so the array index is hardcoded to 0
/// today; the type signature is shaped to allow multi-array
/// virtualizables.
fn vable_setarrayitem_ref_graph_args(
    v_base: super::flow::SpaceOperationArg,
    v_idx: super::flow::SpaceOperationArg,
    v_value: super::flow::SpaceOperationArg,
) -> Vec<super::flow::SpaceOperationArg> {
    vec![
        v_base,
        v_idx,
        v_value,
        majit_ir::descr::vable_array_field_descr(0).into(),
        majit_ir::descr::vable_array_descr(0).into(),
    ]
}

/// Build the 4-arg `getarrayitem_vable_r` arg vector matching
/// `rpython/jit/codewriter/jtransform.py:1882-1885 do_fixed_list_getitem`
/// (vable branch): `[v_base, v_index, arrayfielddescr, arraydescr]`.
/// Counterpart of `vable_setarrayitem_ref_graph_args` for the read
/// side; the result Variable is supplied by the caller to
/// `emit_graph_op_with_result`. `v_base` is the portal frame Variable
/// from `portal_graph_inputvars(code).0` per jtransform.py:840.
fn vable_getarrayitem_ref_graph_args(
    v_base: super::flow::SpaceOperationArg,
    v_idx: super::flow::SpaceOperationArg,
) -> Vec<super::flow::SpaceOperationArg> {
    vec![
        v_base,
        v_idx,
        majit_ir::descr::vable_array_field_descr(0).into(),
        majit_ir::descr::vable_array_descr(0).into(),
    ]
}

/// Build the 3-arg `setfield_vable_i` arg vector matching
/// `rpython/jit/codewriter/jtransform.py:927-928` setfield (vable
/// branch): `[v_inst, v_value, descr]`.  `v_inst` is the portal frame
/// Variable from `portal_graph_inputvars(code).0` per
/// jtransform.py:840 (the JIT driver's red `frame` arg threaded into
/// every vable op). The trailing `vable_static_field_descr(idx)`
/// singleton mirrors `virtualizable.py:71 static_field_descrs[idx]`.
fn vable_setfield_int_graph_args(
    v_inst: super::flow::SpaceOperationArg,
    v_value: super::flow::SpaceOperationArg,
    field_idx: u16,
) -> Vec<super::flow::SpaceOperationArg> {
    vec![
        v_inst,
        v_value,
        majit_ir::descr::vable_static_field_descr(field_idx).into(),
    ]
}

/// Build the 2-arg `getfield_vable_r` arg vector matching
/// `rpython/jit/codewriter/jtransform.py:846-847` getfield (vable
/// branch): `[v_inst, descr]`.  `v_inst` is the portal frame Variable
/// from `portal_graph_inputvars(code).0` per jtransform.py:840 (the
/// JIT driver's red `frame` arg threaded into every vable op). The
/// trailing `vable_static_field_descr(idx)` singleton mirrors
/// `virtualizable.py:71 static_field_descrs[idx]`.
fn vable_getfield_ref_graph_args(
    v_inst: super::flow::SpaceOperationArg,
    field_idx: u16,
) -> Vec<super::flow::SpaceOperationArg> {
    vec![
        v_inst,
        majit_ir::descr::vable_static_field_descr(field_idx).into(),
    ]
}

/// Emit a graph-side `residual_call_{kinds}_{reskind}` SpaceOperation
/// mirroring the SSA shape produced by [`emit_residual_call_shape`].
///
/// Args follow the same kinds-string selection logic as the SSA emit
/// (`codewriter.rs:2545-2587`):
///   - opname suffix `kinds` ∈ `{"r", "ir", "irf"}` chosen by which arg
///     kinds are present + whether `reskind == ResKind::Float`;
///   - argv `[Const(fn_idx), ListI?, ListR?, ListF?, Descr(stub)]`,
///     each `ListX` present iff that letter appears in `kinds`;
///   - the trailing descr is an interned `Arc<CallDescrStub>` from
///     [`super::flatten::intern_call_descr_stub`] .
///
/// Returns the fresh result `Variable` if `reskind != Void`; callers
/// at sites where the symbolic value is already provided by an
/// `emit_frontend_*` HLOp discard it via `let _ = ...` (matching the
/// `emit_vable_getfield_ref!` graph dual-write pattern).
///
/// Mirrors `rpython/jit/codewriter/jtransform.py:414-435 rewrite_call`
/// SpaceOperation construction.
fn record_residual_call_graph_op(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    fn_idx: u16,
    flavor: CallFlavor,
    args_i: Vec<super::flow::FlowValue>,
    args_r: Vec<super::flow::FlowValue>,
    args_f: Vec<super::flow::FlowValue>,
    arg_kinds: Vec<Kind>,
    reskind: ResKind,
    offset: i64,
) -> Option<super::flow::Variable> {
    use super::flow::{FlowListOfKind, SpaceOperationArg};
    let kinds: &str = if !args_f.is_empty() || reskind == ResKind::Float {
        "irf"
    } else if !args_i.is_empty() {
        "ir"
    } else {
        "r"
    };
    let opname = format!("residual_call_{kinds}_{}", reskind.as_char());
    let mut op_args: Vec<SpaceOperationArg> = Vec::with_capacity(5);
    op_args.push(super::flow::Constant::signed(fn_idx as i64).into());
    if kinds.contains('i') {
        op_args.push(SpaceOperationArg::ListOfKind(FlowListOfKind::new(
            Kind::Int,
            args_i,
        )));
    }
    if kinds.contains('r') {
        op_args.push(SpaceOperationArg::ListOfKind(FlowListOfKind::new(
            Kind::Ref,
            args_r,
        )));
    }
    if kinds.contains('f') {
        op_args.push(SpaceOperationArg::ListOfKind(FlowListOfKind::new(
            Kind::Float,
            args_f,
        )));
    }
    let effect_info = super::flatten::effect_info_for_call_flavor(flavor);
    let can_raise = effect_info.check_can_raise(false);
    op_args.push(
        super::flatten::intern_call_descr_stub(effect_info, arg_kinds, reskind.to_kind()).into(),
    );

    let result_var = match reskind.to_kind() {
        Some(result_kind) => {
            let result = graph.fresh_variable(result_kind);
            record_graph_op(block, opname, op_args, Some(result.into()), offset);
            Some(result)
        }
        None => {
            record_graph_op(block, opname, op_args, None, offset);
            None
        }
    };
    // `jtransform.py:311-313` / `handle_residual_call`: a residual_call
    // whose calldescr can raise is immediately followed by a trailing
    // `-live-` so the liveness pass records the registers alive at the
    // implicit GUARD_NO_EXCEPTION.  `flatten.py:206-217` recognises an
    // actually-raising block by scanning this trailing `-live-`.
    if can_raise {
        record_graph_op(block, super::flatten::OPNAME_LIVE, Vec::new(), None, offset);
    }
    result_var
}

/// Emit a void-result `SpaceOperation` into `block` and return it.
/// Matches the call-marker / control-flow emission path in
/// `rpython/jit/codewriter/jtransform.py:1690-1723` where markers like
/// `jit_merge_point` and `loop_header` are produced with no `result`
/// and immediately fed into `GraphFlattener.serialize_op`.
///
/// Walker-rewrite entrypoint: the void counterpart
/// of `emit_graph_op_with_result`.  Callers that need the recorded
/// `SpaceOperation` (e.g. to immediately flatten it into the SSARepr via
/// `GraphFlattener::serialize_op`) use the returned value; callers
/// that only need the side-effect can ignore it.
fn emit_graph_op_void(
    block: &super::flow::BlockRef,
    opname: impl Into<String>,
    args: Vec<super::flow::SpaceOperationArg>,
    offset: i64,
) -> super::flow::SpaceOperation {
    record_graph_op(block, opname, args, None, offset)
}

/// Emit a value-producing `SpaceOperation` into `block`, allocating a
/// fresh `Variable` of `result_kind` to hold the result.  Mirrors the
/// upstream pattern in `rpython/flowspace/flowcontext.py:135-139`:
///
/// ```python
/// w_result = Variable()
/// spaceop = SpaceOperation(name, args_w, w_result)
/// self.recorder.append(spaceop)
/// ```
///
/// Walker-rewrite entrypoint: a single place that
/// packages the fresh-Variable → `record_graph_op` → return-Variable
/// pattern so the walker's per-opcode handlers can record
/// value-producing graph operations without inlining `graph.fresh_variable`
/// + `record_graph_op` at every call site.  Once every value-producing op
/// records through this path the production pipeline can source `SSARepr`
/// from `flatten_graph(graph, ...)` per `codewriter.py:44-67`.
fn emit_graph_op_with_result(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    opname: impl Into<String>,
    args: Vec<super::flow::SpaceOperationArg>,
    result_kind: Kind,
    offset: i64,
) -> super::flow::Variable {
    let result = graph.fresh_variable(result_kind);
    record_graph_op(block, opname, args, Some(result.into()), offset);
    result
}

fn emit_frontend_neg(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    operand: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    // flowcontext.py:192 + unaryoperation(): UNARY_NEGATIVE records
    // `op.neg(w_1).eval(self)` at the frontend graph level.  Keep the
    // graph semantic here and leave the current helper-call lowering in
    // SSARepr as a backend adaptation until frontend-lowering lands.
    emit_graph_op_with_result(graph, block, "neg", vec![operand.into()], Kind::Ref, offset)
}

fn emit_frontend_newlist(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    items: Vec<super::flow::FlowValue>,
    offset: i64,
) -> super::flow::Variable {
    // flowcontext.py:1168-1171 BUILD_LIST -> `op.newlist(*items).eval(self)`.
    // Preserve the frontend semantic op in the graph; the current
    // build_list helper call remains a pyre backend adaptation only.
    emit_graph_op_with_result(
        graph,
        block,
        "newlist",
        items.into_iter().map(Into::into).collect(),
        Kind::Ref,
        offset,
    )
}

fn emit_frontend_newtuple(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    items: Vec<super::flow::FlowValue>,
    offset: i64,
) -> super::flow::Variable {
    // flowcontext.py BUILD_TUPLE -> `op.newtuple(*items).eval(self)`.
    // Preserve the frontend semantic op in the graph; the build_tuple
    // helper call remains a pyre backend adaptation only.
    emit_graph_op_with_result(
        graph,
        block,
        "newtuple",
        items.into_iter().map(Into::into).collect(),
        Kind::Ref,
        offset,
    )
}

fn emit_frontend_newslice(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    start: super::flow::FlowValue,
    stop: super::flow::FlowValue,
    step: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    // flowcontext.py:1154-1161 BUILD_SLICE -> `op.newslice(w_start,
    // w_end, w_step).eval(self)`. Preserve all three operands in the
    // shadow graph so graph-side analysis sees the same dependency shape
    // as RPython/PyPy, even while the bytecode-level SSA stream still uses
    // the pyre helper-call adaptation below.
    emit_graph_op_with_result(
        graph,
        block,
        "newslice",
        vec![start.into(), stop.into(), step.into()],
        Kind::Ref,
        offset,
    )
}

fn emit_frontend_buildslice_shadow_graph(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    argc: pyre_interpreter::bytecode::BuildSliceArgCount,
    start: super::flow::FlowValue,
    stop: super::flow::FlowValue,
    step: Option<super::flow::FlowValue>,
    offset: i64,
) -> super::flow::Variable {
    use pyre_interpreter::bytecode::BuildSliceArgCount;
    let step = match argc {
        BuildSliceArgCount::Two => {
            debug_assert!(step.is_none(), "BUILD_SLICE argc=2 must synthesize None");
            super::flow::Constant::none().into()
        }
        BuildSliceArgCount::Three => step.expect("BUILD_SLICE argc=3 must preserve explicit step"),
    };
    emit_frontend_newslice(graph, block, start, stop, step, offset)
}

fn emit_frontend_setitem(
    _graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    obj: super::flow::FlowValue,
    key: super::flow::FlowValue,
    value: super::flow::FlowValue,
    offset: i64,
) {
    // flowcontext.py:1146-1149 STORE_SUBSCR ->
    // `op.setitem(w_obj, w_subscr, w_newvalue).eval(self)`.
    // Upstream `HLOperation.__init__` (operation.py:66) unconditionally
    // creates a result Variable that rtyper later rewrites to void.
    // pyre has no rtyper, so the op is emitted directly without a
    // result slot; `flatten_space_operation`'s `result == None` branch
    // consumes it identically to what rtyper would produce.
    record_graph_op(
        block,
        "setitem",
        vec![obj.into(), key.into(), value.into()],
        None,
        offset,
    );
}

fn emit_frontend_setattr(
    _graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    obj: super::flow::FlowValue,
    attr_name: super::flow::FlowValue,
    value: super::flow::FlowValue,
    offset: i64,
) {
    // flowcontext.py:1031-1036 STORE_ATTR ->
    // `op.setattr(w_obj, w_attributename, w_newvalue).eval(self)`.
    // See `emit_frontend_setitem` for the void-result rationale.
    record_graph_op(
        block,
        "setattr",
        vec![obj.into(), attr_name.into(), value.into()],
        None,
        offset,
    );
}

fn emit_frontend_getattr(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    obj: super::flow::FlowValue,
    attr_name: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    // flowcontext.py:862-867 LOAD_ATTR ->
    // `op.getattr(w_obj, w_attributename).eval(self)`.
    emit_graph_op_with_result(
        graph,
        block,
        "getattr",
        vec![obj.into(), attr_name.into()],
        Kind::Ref,
        offset,
    )
}

fn emit_frontend_simple_call(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    callable: super::flow::FlowValue,
    args: Vec<super::flow::FlowValue>,
    offset: i64,
) -> super::flow::Variable {
    let mut op_args = Vec::with_capacity(args.len() + 1);
    op_args.push(callable.into());
    op_args.extend(args.into_iter().map(Into::into));
    emit_graph_op_with_result(graph, block, "simple_call", op_args, Kind::Ref, offset)
}

fn emit_frontend_bool(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    operand: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    // flowcontext.py:756-763 POP_JUMP_IF_* branches on
    // `guessbool(op.bool(w_value).eval(self))`. Keep the frontend
    // `bool` operation in the graph and leave the current `truth_fn`
    // SSA helper call as a backend adaptation.  pyre represents
    // `lltype.Bool` in control-flow positions as `Kind::Int`, matching
    // the existing `goto_if_not` / `goto_if_not_int_is_zero` SSA ops.
    //
    // Upstream's `op.bool(w_value).eval(self)` produces a single
    // `lltype.Bool` Variable that flows into the
    // block's exitswitch AND is reused as the `goto_if_not` input by
    // `flatten.py:240-267`. pyre keeps two parallel value chains: the
    // graph-side Variable returned here (consumed only by the front-end
    // exitswitch) and a separate SSA scratch produced by an
    // immediately-following `emit_residual_call(truth_fn_idx, ...,
    // ResKind::Int, Some(scratch_truth))` (consumed by `goto_if_not`).
    // The duplication exists because `FunctionGraph` Variables and
    // `SSARepr` registers still live in two regalloc colorings even
    // though the dual emitter has already been collapsed into the
    // dual emitter into a single walker-local `SSARepr`.
    emit_graph_op_with_result(
        graph,
        block,
        "bool",
        vec![operand.into()],
        Kind::Int,
        offset,
    )
}

fn binary_opname(op: pyre_interpreter::bytecode::BinaryOperator) -> &'static str {
    use pyre_interpreter::bytecode::BinaryOperator as B;

    match op {
        B::Add => "add",
        B::Subtract => "sub",
        B::Multiply => "mul",
        B::FloorDivide => "floordiv",
        B::Remainder => "mod",
        B::TrueDivide => "truediv",
        B::Subscr => "getitem",
        B::Power => "pow",
        B::Lshift => "lshift",
        B::Rshift => "rshift",
        B::And => "and_",
        B::Or => "or_",
        B::Xor => "xor",
        B::InplaceAdd => "inplace_add",
        B::InplaceSubtract => "inplace_sub",
        B::InplaceMultiply => "inplace_mul",
        B::InplaceFloorDivide => "inplace_floordiv",
        B::InplaceRemainder => "inplace_mod",
        B::InplaceTrueDivide => "inplace_truediv",
        B::InplacePower => "inplace_pow",
        B::InplaceLshift => "inplace_lshift",
        B::InplaceRshift => "inplace_rshift",
        B::InplaceAnd => "inplace_and",
        B::InplaceOr => "inplace_or",
        B::InplaceXor => "inplace_xor",
        other => panic!("unsupported BinaryOperator in frontend graph: {other:?}"),
    }
}

fn emit_frontend_binary(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    op: pyre_interpreter::bytecode::BinaryOperator,
    lhs: super::flow::FlowValue,
    rhs: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    emit_graph_op_with_result(
        graph,
        block,
        binary_opname(op),
        vec![lhs.into(), rhs.into()],
        Kind::Ref,
        offset,
    )
}

fn compare_opname(op: pyre_interpreter::bytecode::ComparisonOperator) -> &'static str {
    use pyre_interpreter::bytecode::ComparisonOperator as C;

    match op {
        C::Less => "lt",
        C::LessOrEqual => "le",
        C::Equal => "eq",
        C::NotEqual => "ne",
        C::Greater => "gt",
        C::GreaterOrEqual => "ge",
    }
}

fn emit_frontend_compare(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    op: pyre_interpreter::bytecode::ComparisonOperator,
    lhs: super::flow::FlowValue,
    rhs: super::flow::FlowValue,
    offset: i64,
) -> super::flow::Variable {
    emit_graph_op_with_result(
        graph,
        block,
        compare_opname(op),
        vec![lhs.into(), rhs.into()],
        Kind::Ref,
        offset,
    )
}

fn frontend_load_const_flow_value(code: &CodeObject, idx: usize) -> super::flow::FlowValue {
    // `flowcontext.py:841-843 LOAD_CONST`: fetch the pre-wrapped constant
    // and push that value.  Pyre's CodeObject stores RustPython
    // `ConstantData`, so route through the same materializer used by the
    // blackhole helper and carry the resulting W_Root as a Ref constant in
    // the shadow graph.  The walker still emits `load_const_fn`; this is
    // the graph-side RPython shape.
    pyobject_const_ref_value(pyre_interpreter::pyframe::load_const_from_code(code, idx))
}

fn frontend_global_flow_value(w_code: *const (), name: &str) -> Option<super::flow::FlowValue> {
    // `flowcontext.py:845-858 find_global` resolves globals during flow
    // analysis and pushes a Constant.  Do the same when the current
    // W_CodeObject exposes its globals/builtins; callers fall back to a
    // fresh Ref only when pyre cannot reproduce the static lookup.
    let w_code = w_code as pyre_object::PyObjectRef;
    if w_code.is_null() {
        return None;
    }
    let w_globals = unsafe { pyre_interpreter::w_code_get_w_globals(w_code) };
    if w_globals.is_null() {
        return None;
    }
    let globals = unsafe { &*w_globals };
    if let Some(w_value) = pyre_interpreter::dict_storage_get(globals, name) {
        return Some(pyobject_const_ref_value(w_value));
    }
    let Some(w_builtin) = pyre_interpreter::dict_storage_get(globals, "__builtins__") else {
        return None;
    };
    let lookup_obj = if unsafe { pyre_object::is_module(w_builtin) } {
        unsafe { pyre_object::w_module_get_w_dict(w_builtin) }
    } else if unsafe { pyre_object::is_dict(w_builtin) } {
        w_builtin
    } else {
        return None;
    };
    if lookup_obj.is_null() {
        return None;
    }
    pyre_interpreter::baseobjspace::finditem_str(lookup_obj, name)
        .ok()
        .flatten()
        .map(pyobject_const_ref_value)
}

fn set_last_bool_exitcase(block: &super::flow::BlockRef, branch_taken: bool) {
    let link = block
        .borrow()
        .exits
        .last()
        .cloned()
        .expect("boolean branch must append a Link before setting exitcase");
    let case: super::flow::FlowValue = super::flow::Constant::bool(branch_taken).into();
    let mut link_borrow = link.borrow_mut();
    link_borrow.exitcase = Some(case.clone());
    link_borrow.llexitcase = Some(case);
}

fn sync_stack_state(graph: &mut super::flow::FunctionGraph, state: &mut FrameState, depth: u16) {
    while state.stack.len() > depth as usize {
        state.stack.pop();
    }
    while state.stack.len() < depth as usize {
        state.stack.push(fresh_ref_value(graph));
    }
}

fn new_shadow_graph_with_portal_inputs(
    code: &CodeObject,
    frame_inputs: FrameInputs,
) -> super::flow::FunctionGraph {
    let start_inputargs = graph_entry_inputargs(code, frame_inputs);
    let return_var = Some(super::flow::Variable::new(
        super::flow::VariableId(start_inputargs.len() as u32),
        Kind::Ref,
    ));
    super::flow::FunctionGraph::new(
        code.obj_name.to_string(),
        super::flow::Block::shared(start_inputargs),
        return_var,
    )
}

fn new_shadow_graph(code: &CodeObject) -> super::flow::FunctionGraph {
    new_shadow_graph_with_portal_inputs(code, FrameInputs::None)
}

fn attach_catch_exception_edge(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    target: &SpamBlockRef,
    source_state: &FrameState,
) -> super::flow::LinkRef {
    // `flowcontext.py:148-149 guessexception` sets
    // `block.exitswitch = c_last_exception` before the link is
    // attached.  Run the source-block side first so that the link
    // construction below sees a stable target/source pair.
    {
        let mut block_mut = block.borrow_mut();
        block_mut.exitswitch = Some(super::flow::ExitSwitch::Value(
            super::flow::c_last_exception().into(),
        ));
    }

    // `flowcontext.py:130-134 guessexception` synthesises the
    // `(last_exception, last_exc_value)` Variable pair for this
    // edge.  `exception_landing_state` clones `source_state` and
    // sets `last_exception` to the fresh pair, so the same
    // Variables can be threaded into BOTH `link.args` (via
    // `getoutputargs` below) AND `link.extravars`.
    let edge_state = exception_landing_state(graph, source_state);

    // Update the landing block's framestate / inputargs from the
    // edge state.  Note: RPython models each
    // raise site with its own `EggBlock(vars2, block, case)`
    // (`flowcontext.py:138`), with `vars2 = [Variable(),
    // Variable()]` per case — the egg's body is responsible for
    // any subsequent frame-state restoration.  Pyre coalesces
    // every raise site flowing into the same handler PC into a
    // single catch landing block, so the landing's inputargs are
    // the union of all incoming edge states (pyre-only).  The
    // arity invariant below is satisfied either way because
    // `getoutputargs` walks `target_state.mergeable()` — the
    // same mergeable layout as `target.inputargs`.
    update_catch_landing_state(graph, target, &edge_state);

    // `model.py:114-116 Link.__init__` enforces
    // `len(args) == len(target.inputargs)`.  Build `link.args` via
    // `FrameState::getoutputargs(target_state)` so each link arg
    // aligns with the corresponding target inputarg by mergeable
    // position.  This restores the RPython invariant that the
    // previous `Link::new(Vec::new(), …)` then-mutate flow
    // bypassed (the `Link::new` arity assert ran before
    // `update_catch_landing_state` populated `target.inputargs`).
    let target_state = target
        .framestate()
        .expect("catch landing must have a framestate after update_catch_landing_state");
    let link_args = edge_state.getoutputargs(&target_state);

    // `model.py:127-129 Link.extravars` carries the source-side
    // `(last_exception, last_exc_value)` pair so
    // `flatten.py:340-347` can identify the exception edge and
    // emit `last_exception` / `last_exc_value` SSA renamings at
    // link entry.  The pair is the SAME (exc_type, exc_value)
    // Variables as `edge_state.last_exception`, so they appear in
    // BOTH `link.args` (via `getoutputargs` at the
    // `last_exception` mergeable position) AND `link.extravars`
    // — matching `flowcontext.py:141-143`.
    let (exc_type, exc_value) = exception_edge_extravars(&edge_state);
    let mut link = super::flow::Link::new(link_args, Some(target.block()), None);
    link.extravars(Some(exc_type), Some(exc_value));
    let link = link.into_ref();
    let _ = source_state;
    append_exit(block, link.clone());
    link
}

fn restore_canraise_exit_order(block: &super::flow::BlockRef) {
    let mut block_mut = block.borrow_mut();
    if block_mut.exits.len() < 2 {
        return;
    }

    let first_normal = block_mut.exits.iter().position(|link| {
        let link = link.borrow();
        link.exitcase.is_none()
            && link.llexitcase.is_none()
            && link.last_exception.is_none()
            && link.last_exc_value.is_none()
    });
    let Some(first_normal) = first_normal else {
        return;
    };

    let mut ordered = Vec::with_capacity(block_mut.exits.len());
    ordered.push(block_mut.exits[first_normal].clone());
    for (index, link) in block_mut.exits.iter().enumerate() {
        if index == first_normal {
            continue;
        }
        let is_exception_edge = {
            let link = link.borrow();
            link.last_exception.is_some() || link.last_exc_value.is_some()
        };
        if is_exception_edge {
            ordered.push(link.clone());
        }
    }

    // Structural adaptation for pyre's PC-sequential walker:
    // RPython `flowcontext.py:130-156 guessexception` closes a
    // canraise block with exactly one normal edge followed by
    // exception edges.  Pyre may transiently append an extra normal
    // fallthrough while forcing the next-PC boundary after
    // `emit_catch_exception!`.  That duplicate has no
    // `Link.extravars`, so `flatten.py:223-238` would treat it as an
    // exception link and trip `make_exception_link`'s
    // `last_exception` assertion.  Keep the upstream shape at the
    // graph boundary: first normal edge, then only seeded exception
    // edges.
    if ordered.len() >= 2 {
        block_mut.exits = ordered;
    }
}

// `PyJitCode` and `PyJitCodeMetadata` live in `pyre_jit_trace::pyjitcode`
// so both the codewriter (here) and the trace/blackhole runtime can hold
// the same `Arc<PyJitCode>` instances.

#[derive(Clone)]
struct ExceptionCatchSite {
    landing_label: u16,
    handler_py_pc: usize,
    stack_depth: u16,
    push_lasti: bool,
    lasti_py_pc: usize,
    landing: SpamBlockRef,
}

/// RPython: per-graph output of `perform_register_allocation` over the
/// three register kinds (codewriter.py:46-48). pyre's regalloc is
/// trivial — fast locals occupy the bottom of the ref register file
/// and the value stack stacks above them — so the "allocation" reduces
/// to a handful of constant offsets derived from `code.varnames` /
/// `code.max_stackdepth`. `RegisterLayout::compute` runs the same
/// arithmetic the inline section of `transform_graph_to_jitcode` used
/// to do directly; its only purpose is to give the layout a name and
/// pull the calculation out of the 1400-line dispatch loop.
#[derive(Clone, Copy, Debug)]
struct RegisterLayout {
    /// `code.varnames.len()` — number of fast locals.
    nlocals: usize,
    /// Absolute index where the operand stack begins in
    /// `PyFrame.locals_cells_stack_w` — `nlocals + pyframe::ncells(code)`.
    stack_base_absolute: usize,
    /// Compile-time depth bound from `code.max_stackdepth` (= CPython
    /// `co_stacksize`). Used directly without clamping so the per-CodeObject
    /// `stack_slot_color_map` length matches the runtime PyFrame allocation
    /// `nlocals + ncells + max_stackdepth` (`pyframe.rs:1576`).
    ///
    /// NOTE: this is the FRAME-LENGTH bound, not the regalloc PIN bound.
    /// `ExternalInputs::max_stack_depth` (regalloc.rs:603) takes
    /// `max_stack_depth_observed = max(depth_at_pc)` instead — only the
    /// live prefix is forced into identity colors by `enforce_input_args`.
    /// Tail entries `d >= max_stack_depth_observed` get identity colors
    /// only by virtue of never appearing in any SSA op (regalloc skips
    /// them, fallthrough to pre-rename pass-through). See
    /// `pyjitcode.rs::stack_slot_color_map` "Color invariant" docstring.
    max_stackdepth: usize,
    /// Ref register index where the operand stack begins
    /// (`stack_base = nlocals` since locals occupy the first registers).
    stack_base: u16,
}

impl RegisterLayout {
    /// Pure arithmetic over `code` — no allocation, no side effects.
    /// Mirrors the constant block at the top of
    /// `transform_graph_to_jitcode`.
    fn compute(code: &CodeObject) -> Self {
        let nlocals = code.varnames.len();
        let stack_base_absolute = nlocals + pyre_interpreter::pyframe::ncells(code);
        let max_stackdepth = code.max_stackdepth as usize;
        let stack_base = nlocals as u16;
        Self {
            nlocals,
            stack_base_absolute,
            max_stackdepth,
            stack_base,
        }
    }
}

/// Per-helper `(idx, flavor)` pair returned by
/// `register_helper_fn_pointers`.  RPython's `getcalldescr`
/// (`call.py:282-330`) derives the analogous information from a chain
/// of graph analyzers (`raise_analyzer`, `readwrite_analyzer`,
/// `collect_analyzer`, `virtualizable_analyzer`,
/// `quasiimmut_analyzer`, `randomeffects_analyzer`); pyre lacks the
/// flow-graph + rtyper infrastructure those analyzers need.  Until the
/// analyzer port lands, we mirror RPython's `effects is top_set`
/// fallback (`effectinfo.py:285`) at helper granularity: each helper
/// carries a hand-classified `CallFlavor` that the codewriter consults
/// instead of hardcoding `Plain` / `MayForce` per emit site.
#[derive(Clone, Copy, Debug)]
struct HelperHandle {
    idx: u16,
    flavor: CallFlavor,
}

/// Indices + per-helper `CallFlavor` returned by
/// `assembler.add_fn_ptr` for every blackhole helper fn pointer the
/// dispatch loop references. Mirrors the slot shape of RPython's
/// `_callinfo_for_oopspec`-derived index table — the helpers are
/// interned in a fixed order so the dispatch handlers can capture the
/// indices once and reuse them across emit sites.
///
/// Note: the order matches the historical
/// inline sequence (`call_fn`, then the per-opcode helpers, then the
/// per-arity `call_fn_n`). Changing the order would shift every
/// `assembler.add_fn_ptr` index — RPython's `assembler.see_raw_object`
/// path has the same constraint.
#[derive(Clone, Copy, Debug)]
struct FnPtrIndices {
    call_fn: HelperHandle,
    load_global_fn: HelperHandle,
    compare_fn: HelperHandle,
    binary_op_fn: HelperHandle,
    box_int_fn: HelperHandle,
    truth_fn: HelperHandle,
    load_const_fn: HelperHandle,
    store_subscr_fn: HelperHandle,
    build_list_fn: HelperHandle,
    build_tuple_fn: HelperHandle,
    unpack_sequence_fn: HelperHandle,
    unpack_item_fn: HelperHandle,
    build_slice_fn: HelperHandle,
    normalize_raise_varargs_fn: HelperHandle,
    call_fn_0: HelperHandle,
    call_fn_2: HelperHandle,
    call_fn_3: HelperHandle,
    call_fn_4: HelperHandle,
    call_fn_5: HelperHandle,
    call_fn_6: HelperHandle,
    call_fn_7: HelperHandle,
    call_fn_8: HelperHandle,
    get_current_exception_fn: HelperHandle,
    set_current_exception_fn: HelperHandle,
}

/// Register every blackhole helper fn pointer with the assembler in
/// the canonical order. Returns the per-helper handle table used by
/// the dispatch loop.
///
/// CallFlavor classification per helper (RPython parity:
/// `call.py:282-330` + `effectinfo.py:13-52`):
///
/// * `get_current_exception_fn` / `set_current_exception_fn`: TLS
///   read/write of `CURRENT_EXCEPTION`; never raises.  RPython's
///   `EF_CANNOT_RAISE` (`call.py:303 getcalldescr`'s `else` branch) →
///   `PlainCannotRaise`.
/// * `compare_fn` / `binary_op_fn` / `store_subscr_fn` / `call_fn` /
///   `call_fn_n` / `truth_fn` / `normalize_raise_varargs_fn`:
///   dispatch into user Python code (`__add__` / `__eq__` /
///   `__setitem__` / arbitrary callable / `__bool__` /
///   exception-class `__init__`); arbitrary user code that observes
///   virtualizables.  Matches `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
///   (`effectinfo.py:23`) → `MayForce`.
/// * `load_global_fn` / `build_list_fn` / `build_slice_fn`: namespace dict lookup +
///   list allocation; can raise (`NameError` / `MemoryError`) but do
///   not force virtuals — `EF_CAN_RAISE` → `Plain`.
/// * `box_int_fn` / `load_const_fn`: kept on `Plain` until the
///   upstream `@jit.elidable_promote` decorator is wired
///   (`rpython/rlib/jit.py:180`) and pyre's constant storage shape
///   matches PyPy's pre-wrapped `co_consts_w`
///   (`pypy/interpreter/pyopcode.py:516` vs
///   `pyre-interpreter/src/pyframe.rs:1748-1768`).  Hand-pure
///   classification without those prerequisites would be a
///   TODO.
fn register_helper_fn_pointers(
    assembler: &mut SSAReprEmitter,
    cpu: &super::cpu::Cpu,
) -> FnPtrIndices {
    // RPython: CallControl manages fn addresses; assembler.finished()
    // writes them into callinfocollection. pyre adds them inline so
    // each handler can capture the index it needs.
    //
    // `bind` registers a helper fn pointer with its per-callee
    // [`majit_metainterp::EffectInfoSlot`] (`call.py:282-303
    // getcalldescr` parity, see [`slot_for_call_flavor`]) so the runtime
    // [`majit_metainterp::JitCallTarget`] descriptor carries the
    // matching `extraeffect`.  The dispatcher then threads
    // `target.effect_info_slot` through
    // `make_call_descr_from_target_slot` (`call_descr.rs:390`) so the
    // recorded trace descr's `EffectInfo` matches the producer's
    // hand-classified flavor.
    let bind = |assembler: &mut SSAReprEmitter, ptr: *const (), flavor: CallFlavor| {
        // `MayForce` / `ReleaseGil` are dispatched via the
        // `call_may_force_*` / `call_release_gil_*` paths whose EI is
        // resolved inline by the const factory at
        // `jitcode/assembler.rs::emit_canonical_call_typed_via_target*`
        // (saturated bitsets, `(1, 0)` release-gil sentinel —
        // `effectinfo.py:249`).  The `JitCallTarget.effect_info_slot`
        // is unread for those families, so we register without a slot;
        // routing them through `slot_for_call_flavor` would trip its
        // `jtransform.py:1677` assert.  Every other flavor goes through
        // the slot path so the runtime descriptor carries the matching
        // `extraeffect`.
        let idx = match flavor {
            CallFlavor::MayForce | CallFlavor::ReleaseGil => assembler.add_fn_ptr(ptr),
            _ => assembler.add_fn_ptr_with_slot(ptr, slot_for_call_flavor(flavor)),
        };
        HelperHandle { idx, flavor }
    };
    let call_fn = bind(assembler, cpu.call_fn as *const (), CallFlavor::MayForce);
    // `bh_load_global_fn` mirrors pyopcode.py `_load_global`: globals
    // lookup, then the current frame's picked builtins module, then
    // `NameError` synthesis on miss.
    // Matches `EF_CAN_RAISE` (`call.py:301` `elif self._canraise(op)`):
    // can raise but does not force virtuals.
    let load_global_fn = bind(
        assembler,
        cpu.load_global_fn as *const (),
        CallFlavor::Plain,
    );
    let compare_fn = bind(assembler, cpu.compare_fn as *const (), CallFlavor::MayForce);
    let binary_op_fn = bind(
        assembler,
        cpu.binary_op_fn as *const (),
        CallFlavor::MayForce,
    );
    // `pypy/objspace/std/intobject.py:wrap_int` is NOT decorated
    // with `@jit.elidable_promote` (the upstream decorator lives
    // at `rpython/rlib/jit.py:180`; the local PyPy intobject
    // declaration omits it).  Without an analyzer or a matching
    // upstream decorator, hand-classifying `box_int_fn` as
    // `Pure*` would be a TODO; stay on `Plain` until the
    // decorator path or a per-callee analyzer landing produces
    // an upstream-cited pure flavor.
    let box_int_fn = bind(assembler, cpu.box_int_fn as *const (), CallFlavor::Plain);
    // `bh_truth_fn` delegates to `opcode_ops::truth_value(obj)`,
    // which invokes Python `__bool__` and may run arbitrary user
    // code that observes (and therefore forces) virtualizables.
    // Matches `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
    // (`effectinfo.py:23`) → `MayForce` per Slice α-2.
    let truth_fn = bind(assembler, cpu.truth_fn as *const (), CallFlavor::MayForce);
    // PyPy's `LOAD_CONST` reads pre-wrapped `co_consts_w`
    // (`pypy/interpreter/pyopcode.py:516`); pyre's
    // `pyre_interpreter::pyframe::load_const_from_code`
    // (`pyre-interpreter/src/pyframe.rs:1748-1768`)
    // re-materializes int / float / str / bool constants on every
    // call, so the helper is NOT observably idempotent. Stay on
    // `Plain` until the constant-storage shape converges to the
    // PyPy pre-wrapped representation.
    let load_const_fn = bind(assembler, cpu.load_const_fn as *const (), CallFlavor::Plain);
    let store_subscr_fn = bind(
        assembler,
        cpu.store_subscr_fn as *const (),
        CallFlavor::MayForce,
    );
    // `bh_build_list_fn` is allocation-only — `build_list_from_refs`
    // wraps the supplied PyObjectRefs; items are pre-existing heap
    // refs, no user `__init__` invocation (`call_jit.rs:3452-3464`).
    // Matches `EF_CAN_RAISE` (allocation can `MemoryError`) without
    // virtual-force.
    let build_list_fn = bind(assembler, cpu.build_list_fn as *const (), CallFlavor::Plain);
    // `pypy/interpreter/pyopcode.py:1463-1472 BUILD_SLICE` calls
    // `space.newslice(w_start, w_end, w_step)`.  Pyre mirrors that with
    // a flat allocation helper; no user code runs, but allocation can fail.
    let build_slice_fn = bind(
        assembler,
        cpu.build_slice_fn as *const (),
        CallFlavor::Plain,
    );
    // `bh_normalize_raise_varargs_with_frame` walks the exception class /
    // value pair and instantiates user `__init__` — arbitrary
    // user code that may observe virtualizables.  Matches
    // `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` (`effectinfo.py:23`)
    // → `MayForce` per Slice α-2.
    let normalize_raise_varargs_fn = bind(
        assembler,
        cpu.normalize_raise_varargs_fn as *const (),
        CallFlavor::MayForce,
    );
    // Per-arity call helpers (appended AFTER existing fn_ptrs to preserve indices).
    let call_fn_0 = bind(assembler, cpu.call_fn_0 as *const (), CallFlavor::MayForce);
    let call_fn_2 = bind(assembler, cpu.call_fn_2 as *const (), CallFlavor::MayForce);
    let call_fn_3 = bind(assembler, cpu.call_fn_3 as *const (), CallFlavor::MayForce);
    let call_fn_4 = bind(assembler, cpu.call_fn_4 as *const (), CallFlavor::MayForce);
    let call_fn_5 = bind(assembler, cpu.call_fn_5 as *const (), CallFlavor::MayForce);
    let call_fn_6 = bind(assembler, cpu.call_fn_6 as *const (), CallFlavor::MayForce);
    let call_fn_7 = bind(assembler, cpu.call_fn_7 as *const (), CallFlavor::MayForce);
    let call_fn_8 = bind(assembler, cpu.call_fn_8 as *const (), CallFlavor::MayForce);
    // TLS read of `CURRENT_EXCEPTION`; cannot raise; touches no GC
    // heap (TLS slot is not tracked by `force_from_effectinfo`'s
    // field/array bitstring caches). Maps to PyPy's analyzer output
    // for a flat helper: `extraeffect=EF_CANNOT_RAISE` + every six
    // raw set `frozenset()` + `can_collect=False`
    // (`effectinfo.py:281-283`-equivalent).
    let get_current_exception_fn = bind(
        assembler,
        cpu.get_current_exception_fn as *const (),
        CallFlavor::PlainCannotRaiseNoHeap,
    );
    // TLS write; void return; cannot raise; touches no GC heap.
    let set_current_exception_fn = bind(
        assembler,
        cpu.set_current_exception_fn as *const (),
        CallFlavor::PlainCannotRaiseNoHeap,
    );
    // `bh_build_tuple_fn` is allocation-only, same classification as
    // `bh_build_list_fn` (allocation can `MemoryError`, no virtual-force).
    // Bound after the existing fn_ptrs to preserve their indices.
    let build_tuple_fn = bind(
        assembler,
        cpu.build_tuple_fn as *const (),
        CallFlavor::Plain,
    );
    // `bh_unpack_sequence_fn` validates the length and allocates the item
    // tuple (can raise ValueError/TypeError); `bh_unpack_item_fn` indexes
    // that validated tuple. Both can raise → `CallFlavor::Plain`.
    let unpack_sequence_fn = bind(
        assembler,
        cpu.unpack_sequence_fn as *const (),
        CallFlavor::Plain,
    );
    let unpack_item_fn = bind(
        assembler,
        cpu.unpack_item_fn as *const (),
        CallFlavor::Plain,
    );
    FnPtrIndices {
        call_fn,
        load_global_fn,
        compare_fn,
        binary_op_fn,
        box_int_fn,
        truth_fn,
        load_const_fn,
        store_subscr_fn,
        build_list_fn,
        build_tuple_fn,
        unpack_sequence_fn,
        unpack_item_fn,
        build_slice_fn,
        normalize_raise_varargs_fn,
        call_fn_0,
        call_fn_2,
        call_fn_3,
        call_fn_4,
        call_fn_5,
        call_fn_6,
        call_fn_7,
        call_fn_8,
        get_current_exception_fn,
        set_current_exception_fn,
    }
}

/// RPython: `liveness.py:19-80` `compute_liveness(ssarepr)` —
/// backward dataflow over the populated `SSARepr` that fills each
/// `-live-` marker with the set of registers alive across it.
///
/// The dataflow runs on the post-regalloc `SSARepr` via the upstream
/// `liveness::compute_liveness`, including `remove_repeated_live`.
/// Pyre's per-PC `Label("pcN")` markers survive that rewrite unchanged,
/// so the follow-up filter rescans label-delimited ranges in the FINAL
/// `SSARepr` to find each Python PC's `-live-` marker instead of caching
/// pre-rewrite insn indices.
///
/// After the dataflow, pyre rewrites each `-live-` marker so the
/// args are split into live_i / live_r / live_f sequences, mirroring
/// upstream `assembler.py:150-152`
/// (`get_liveness_info(insn[1:], 'int'/'ref'/'float')`). The
/// tracer (`trace_opcode.rs:670`) and the blackhole bridge-resume
/// (`call_jit.rs:870-887`) read the three banks in order via
/// `LivenessIterator`, so the post-rename `-live-` marker is the
/// sole source.
///
/// `live_i` and `live_f` are emitted line-by-line parity with
/// upstream — RPython's `liveness.py:19-76 compute_liveness`
/// produces one SSA-driven alive set as the sole authority, and
/// `assembler.py:150-152` only splits that set by kind.
///
/// Note: `live_r` carries an extra LV∩SSA
/// `retain` step on top of the SSA bank — see the inline comment
/// in the loop body below.  Removing it requires extending the
/// encoder + symbolic-state pair to track scratch Ref colors,
/// matching `pyjitpl.py:218-225` line by line.
///
/// Unreachable PCs still get emptied in place via the bytecode
/// `LiveVars` analysis. The direct-dispatch walker emits one
/// `Label("pcN")` + `-live-` pair per Python PC, including dead
/// bytecodes that never execute, whereas upstream RPython only
/// flattens reachable flow-graph blocks.
fn filter_liveness_in_place(
    ssarepr: &mut super::flatten::SSARepr,
    code: &CodeObject,
    depth_at_pc: &[u16],
    local_color_map: &[u16],
    stack_slot_color_map: &[u16],
    portal_frame_reg: u16,
    portal_ec_reg: u16,
    walker_tracked_pc_live_indices: Option<&[usize]>,
) -> Vec<usize> {
    use super::flatten::{Kind as SsaKind, Operand as SsaOperand};
    // Walker-tracked positions are required: the post-merge
    // `live_markers` vector is built by translating each walker-
    // recorded per-PC `-live-` position through the
    // `remove_repeated_live` remap.  The walker's
    // `walker_pc_live_marker_pos` side-table is the authoritative
    // source since label retirement retired the per-PC `Insn::Label`
    // emission.
    let walker_tracked = walker_tracked_pc_live_indices
        .filter(|walker_tracked| walker_tracked.len() == code.instructions.len())
        .expect(
            "filter_liveness_in_place: walker_tracked_pc_live_indices must be Some with one \
             entry per Python PC since label retirement retired per-PC label emission",
        );
    // Run `compute_liveness` + `remove_repeated_live` and resolve each
    // Python PC's `-live-` marker to its POST-merge SSARepr index.
    // `liveness.rs`'s public API (`compute_liveness`,
    // `remove_repeated_live`) matches upstream `liveness.py`
    // exactly — adjacent walker `-live-` markers may fold; the
    // tolerant filter below handles PCs that resolve to a shared
    // marker by emitting the UNION of per-PC narrowed sets.
    let live_markers = super::liveness::compute_liveness_with_pc_anchors(ssarepr, walker_tracked);
    let live_vars = pyre_jit_trace::state::liveness_for(code as *const _);
    let nlocals = code.varnames.len();
    let live_markers_out = live_markers.clone();
    assert!(
        local_color_map.len() >= nlocals,
        "local_color_map is shorter than nlocals: {} < {}",
        local_color_map.len(),
        nlocals
    );

    // Snapshot original marker contents BEFORE any mutation so that
    // when multiple Python PCs share a single post-merge `-live-`
    // marker (possible once `remove_repeated_live` folds adjacent
    // markers without protection), each PC's narrowing pass reads
    // the SSA union — not a previously-narrowed set.
    let original_markers: Vec<Vec<SsaOperand>> = live_markers
        .iter()
        .map(|&idx| {
            ssarepr
                .insns
                .get(idx)
                .and_then(|i| i.live_args())
                .map(|args| args.to_vec())
                .unwrap_or_default()
        })
        .collect();

    // Group `(py_pc, insn_idx)` pairs by `insn_idx` so a shared
    // marker accumulates the UNION of per-PC narrowed sets (resume
    // from any sharing PC reads a conservative superset, which is
    // safe — preserving more registers than strictly needed never
    // causes incorrect resume).
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (py_pc, &insn_idx) in live_markers.iter().enumerate() {
        if let Some(entry) = groups.iter_mut().find(|(idx, _)| *idx == insn_idx) {
            entry.1.push(py_pc);
        } else {
            groups.push((insn_idx, vec![py_pc]));
        }
    }

    for (insn_idx, py_pcs) in groups {
        // Original snapshot is the same for every PC in the group
        // (they all point at the same marker).
        let original = &original_markers[py_pcs[0]];
        let non_register: Vec<SsaOperand> = original
            .iter()
            .filter(|op| !matches!(op, SsaOperand::Register(_)))
            .cloned()
            .collect();

        let mut any_reachable = false;
        let mut union_i: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
        let mut union_r: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
        let mut union_f: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();

        for &py_pc in &py_pcs {
            if !live_vars.is_reachable(py_pc) {
                continue;
            }
            any_reachable = true;

            // Per-PC SSA-live decomposition over the snapshot.
            // `compute_liveness` emits Registers sorted by `(kind,
            // index)`, so seen-set dedup keeps the encounter order
            // stable across runs.
            //
            // liveness.py:67-75 `compute_liveness` adds every Register to
            // the alive set; assembler.py:150-152 splits the `-live-`
            // args into live_i / live_r / live_f by kind via
            // `get_liveness_info(insn[1:], 'int'/'ref'/'float')`.
            let mut seen_r: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
            let mut seen_i: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
            let mut seen_f: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
            let mut pc_live_r: Vec<u16> = Vec::new();
            let mut pc_live_i: Vec<u16> = Vec::new();
            let mut pc_live_f: Vec<u16> = Vec::new();
            for op in original.iter() {
                let SsaOperand::Register(reg) = op else {
                    continue;
                };
                match reg.kind {
                    SsaKind::Ref => {
                        if seen_r.insert(reg.index) {
                            pc_live_r.push(reg.index);
                        }
                    }
                    SsaKind::Int => {
                        if seen_i.insert(reg.index) {
                            pc_live_i.push(reg.index);
                        }
                    }
                    SsaKind::Float => {
                        if seen_f.insert(reg.index) {
                            pc_live_f.push(reg.index);
                        }
                    }
                }
            }

            // Note: LV∩SSA retain narrows the Ref bank to post-rename
            // colors that correspond to LV-live Python locals or live
            // stack slots at this PC.  Scratch registers (temporaries
            // SSA-live but with no Python-frame slot) remain
            // `OpRef::NONE` in `registers_r` because no trace-time
            // writer populates them.  Removing this retain requires
            // either (a) populating scratch colors during tracing
            // (graph regalloc) or (b) the encoder tolerating
            // NONE for non-frame live registers.
            let depth = depth_at_pc[py_pc] as usize;
            let live_stack_colors: std::collections::BTreeSet<u16> =
                stack_slot_color_map.iter().copied().take(depth).collect();
            let lv_live: std::collections::BTreeSet<u16> = {
                let mut s: std::collections::BTreeSet<u16> = (0..nlocals)
                    .filter(|&idx| live_vars.is_local_live(py_pc, idx))
                    .map(|idx| local_color_map[idx])
                    .collect();
                s.extend(live_stack_colors.iter().copied());
                // Portal red args (`pypy/module/pypyjit/interp_jit.py:67
                // reds = ['frame', 'ec']`) reach `live_r` through the
                // RPython force-alive mechanism (`liveness.py:11-12`):
                // `emit_live_placeholder!` emits every PC's `-live-` op
                // with explicit Register args for `portal_frame_reg` /
                // `portal_ec_reg`.  Gate the portal colors past the
                // LV∩SSA retain so the retain does not drop the
                // RPython-tracked live registers.  Portal-bridge
                // installs sentinel-skip (`u16::MAX`).
                if portal_frame_reg != u16::MAX {
                    s.insert(portal_frame_reg);
                }
                if portal_ec_reg != u16::MAX {
                    s.insert(portal_ec_reg);
                }
                s
            };
            pc_live_r.retain(|idx| lv_live.contains(idx));

            union_i.extend(pc_live_i);
            union_r.extend(pc_live_r);
            union_f.extend(pc_live_f);
        }

        let existing = match ssarepr.insns.get_mut(insn_idx) {
            Some(insn) if insn.is_live() => insn.live_args_mut().unwrap(),
            Some(other) => panic!(
                "filter_liveness_in_place: expected -live- marker at index {insn_idx}, got \
                 {other:?}"
            ),
            None => panic!(
                "filter_liveness_in_place: insn index {insn_idx} out of range (len {})",
                ssarepr.insns.len()
            ),
        };
        existing.clear();
        if any_reachable {
            for &idx in &union_i {
                existing.push(SsaOperand::Register(super::flatten::Register::new(
                    SsaKind::Int,
                    idx,
                )));
            }
            for &idx in &union_r {
                existing.push(SsaOperand::Register(super::flatten::Register::new(
                    SsaKind::Ref,
                    idx,
                )));
            }
            for &idx in &union_f {
                existing.push(SsaOperand::Register(super::flatten::Register::new(
                    SsaKind::Float,
                    idx,
                )));
            }
        }
        existing.extend(non_register);
    }
    live_markers_out
}

/// Decode `code.exceptiontable` into the structures the dispatch loop
/// consumes:
/// - `catch_for_pc[py_pc]` — `Some(landing_label)` for every PC that
///   falls inside an exception range, mapping to the landing label
///   the dispatch loop will branch to on raise.
/// - `catch_sites` — one entry per active range, holding the handler
///   PC, the saved stack depth, and the `push_lasti` flag. The
///   dispatch loop emits a landing block per entry at the end.
/// - `handler_depth_at[handler_pc]` — the stack depth Python sets on
///   exception-handler entry (`entry.depth + 1` plus another `+1`
///   when `push_lasti`); used by the dispatch loop to fix
///   `current_depth` at the handler's first instruction.
///
/// Note: RPython has no analog because RPython
/// flow graphs already carry exception-handling links; pyre's input
/// is raw CPython bytecode + the packed exception table, so this
/// preprocessing step is pyre-specific.
fn decode_exception_catch_sites(
    assembler: &mut SSAReprEmitter,
    graph: &mut super::flow::FunctionGraph,
    code: &CodeObject,
    num_instrs: usize,
) -> (Vec<Option<u16>>, Vec<ExceptionCatchSite>, Vec<Option<u16>>) {
    // `decode_exceptiontable` yields byte offsets; codewriter operates in
    // instruction-index units.  Convert at the boundary.
    let exception_entries: Vec<_> =
        pyre_interpreter::exception_table::decode_exceptiontable(&code.exceptiontable)
            .map(|e| {
                (
                    e.start as usize / 2,
                    e.end as usize / 2,
                    e.target as usize / 2,
                    e.depth as u16,
                    e.lasti,
                )
            })
            .collect();
    let mut catch_for_pc: Vec<Option<u16>> = vec![None; num_instrs];
    let mut catch_sites: Vec<ExceptionCatchSite> = Vec::new();
    for py_pc in 0..num_instrs {
        // `pypy/interpreter/pycode.py:250-253` last-matching-wins: walk the
        // entries in encoding order, keep the *last* match for this PC.
        // Multiple ranges may cover one PC (nested try/finally/with), and
        // CPython's emission order puts the innermost (most-specific) entry
        // last.  Earlier pyre used `.find(...)` which returned the first
        // match — divergence from PyPy in nested cases.
        let mut chosen: Option<&(usize, usize, usize, u16, bool)> = None;
        for entry in &exception_entries {
            let (start, end, _target, _depth, _lasti) = *entry;
            if py_pc >= start && py_pc < end {
                chosen = Some(entry);
            } else if start > py_pc {
                break;
            }
        }
        let Some(&(_start, _end, handler_py_pc, depth, push_lasti)) = chosen else {
            continue;
        };
        if handler_py_pc >= num_instrs {
            continue;
        }
        let landing_label = assembler.new_label();
        catch_for_pc[py_pc] = Some(landing_label);
        let landing = SpamBlockRef::new(graph.new_block(Vec::new()), None);
        catch_sites.push(ExceptionCatchSite {
            landing_label,
            handler_py_pc,
            stack_depth: depth,
            push_lasti,
            lasti_py_pc: py_pc,
            landing,
        });
    }
    let mut handler_depth_at: Vec<Option<u16>> = vec![None; num_instrs];
    for (_start, _end, target, depth, lasti) in &exception_entries {
        if *target < num_instrs {
            let extra = if *lasti { 1u16 } else { 0 };
            handler_depth_at[*target] = Some(*depth + extra + 1);
        }
    }
    (catch_for_pc, catch_sites, handler_depth_at)
}

// Note: the legacy `liveness_regs_to_u8_sorted` helper that returned
// `Option<Vec<u8>>` to flag the 256-register cap is gone. The cap is
// now enforced by `majit_translate::liveness::encode_liveness`'s
// `assert!(char_ < 256)` (RPython `liveness.py:147-166` parity), and
// the post-pass register allocator
// (`super::regalloc::allocate_registers`) compresses the indices so
// the cap fires only on pathological functions whose `nlocals` alone
// exceeds 256 — the same condition that crashes the RPython
// translator.

// ---------------------------------------------------------------------------
// RPython: codewriter/codewriter.py — class CodeWriter
// ---------------------------------------------------------------------------

/// Compiles Python CodeObjects into JitCode for blackhole execution.
///
/// RPython: `rpython/jit/codewriter/codewriter.py::CodeWriter`.
/// `codewriter.py:20-23` stores `self.assembler = Assembler()` and
/// `self.callcontrol = CallControl(cpu, jitdrivers_sd)` once on the
/// CodeWriter and reuses them across every `transform_graph_to_jitcode`
/// call so `all_liveness` / `num_liveness_ops` and the `jitcodes` dict
/// accumulate over the whole translator session.
///
/// pyre mirrors that ownership via a per-thread singleton: the process
/// holds a single `CodeWriter` instance (one per thread) reachable via
/// [`CodeWriter::instance`], matching `warmspot.py:245`
/// `codewriter = CodeWriter(cpu, [jd])`. The owned `Assembler` lives on
/// a `RefCell<Assembler>` field so `transform_graph_to_jitcode` can
/// still mutate it under the immutable-by-default singleton borrow.
pub struct CodeWriter {
    /// `codewriter.py:22` `self.assembler = Assembler()`.
    ///
    /// Single Assembler instance shared across every `transform_graph_to_jitcode`
    /// call on this CodeWriter. `all_liveness` / `all_liveness_positions` /
    /// `num_liveness_ops` accumulate here just like the upstream object.
    assembler: RefCell<Assembler>,
    /// RPython: `self.callcontrol = CallControl(cpu, jitdrivers_sd)`
    /// (codewriter.py:23). Owned in a `UnsafeCell` so `&CodeWriter` can
    /// mint `&mut CallControl` through [`Self::callcontrol`] — matches
    /// the legacy `JITCODE_CACHE` interior-mutability contract.
    callcontrol: UnsafeCell<super::call::CallControl>,
    /// RPython: `gc_ll_descr.gc_cache._cache_call`
    /// (`backend/llsupport/descr.py:14 GcCache.__init__` +
    /// `:665-673 get_call_descr`).  RPython's call descr cache is a
    /// per-`GcCache` instance dict keyed by `(arg_classes, result_type,
    /// result_signed, RESULT_ERASED, extrainfo)` and reached via
    /// `cpu.gc_ll_descr.gc_cache`.  pyre owns the cache here so each
    /// `CodeWriter` instance carries its own cache, mirroring the
    /// per-instance ownership upstream.  See
    /// [`super::flatten::intern_call_descr_stub`] for the `Option<Kind>`
    /// → `result_type` mapping.
    call_descr_stub_cache:
        Mutex<HashMap<(majit_ir::EffectInfo, Vec<Kind>, Option<Kind>), Arc<CallDescrStub>>>,
}

impl CodeWriter {
    /// RPython: `CodeWriter.__init__(cpu, jitdrivers_sd)` (codewriter.py:20-23).
    ///
    /// The cpu helpers are fixed module-level functions in
    /// `crate::call_jit`; `callcontrol` is a field so
    /// `writer.callcontrol()` matches `self.callcontrol` in upstream.
    pub fn new() -> Self {
        // codewriter.py:21-23 `self.cpu = cpu; self.assembler = Assembler();
        //   self.callcontrol = CallControl(cpu, jitdrivers_sd)`.
        // pyre owns the single `Cpu` on `CallControl`; `CodeWriter::cpu()`
        // returns a borrow back out so the upstream attribute access
        // pattern (`self.cpu`) still works.
        let cpu = super::cpu::Cpu::new();
        Self {
            assembler: RefCell::new(Assembler::new()),
            callcontrol: UnsafeCell::new(super::call::CallControl::new(cpu, Vec::new())),
            call_descr_stub_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Intern a [`CallDescrStub`] by `(effect_info, arg_kinds,
    /// result_kind)` into this CodeWriter's instance cache and return
    /// the shared `Arc` upcast to `majit_ir::DescrRef`.  Mirrors
    /// `gc_ll_descr.gc_cache.get_call_descr(arg_classes, result_type,
    /// result_signed, RESULT_ERASED, extrainfo)` upstream.  See
    /// [`super::flatten::intern_call_descr_stub`] (free-function
    /// forwarder used by graph-side recorders) and
    /// `call_descr_stub_cache` field docstring for the parity mapping.
    pub fn intern_call_descr_stub(
        &self,
        effect_info: majit_ir::EffectInfo,
        arg_kinds: Vec<Kind>,
        result_kind: Option<Kind>,
    ) -> majit_ir::DescrRef {
        // `descr.py:665` keys `get_call_descr` on
        // `(arg_classes, result_type, result_signed, RESULT_ERASED, extrainfo)`
        // and `descr.py:670-674` writes `result_type` onto the constructed
        // CallDescr. Pyre mirrors the redundancy: the cache key carries
        // `result_kind` AND the stored stub carries it too, so the
        // assembler can cross-validate the descr's `result_kind` against
        // the opname-tail-derived `ResKind` at dispatch time
        // (`assembler.rs:1370`).
        let key = (effect_info, arg_kinds, result_kind);
        let mut cache = self.call_descr_stub_cache.lock().unwrap();
        let arc = cache.entry(key.clone()).or_insert_with(|| {
            Arc::new(CallDescrStub {
                effect_info: key.0.clone(),
                arg_kinds: key.1.clone(),
                result_kind: key.2,
            })
        });
        arc.clone() as Arc<dyn majit_ir::Descr>
    }

    /// `codewriter.py:21` `self.cpu = cpu`.
    ///
    /// Convenience accessor — pyre owns the single `Cpu` on
    /// `CallControl` (call.py:27 `self.cpu = cpu`); upstream both
    /// attributes point at the same object.
    pub fn cpu(&self) -> &super::cpu::Cpu {
        &self.callcontrol().cpu
    }

    /// RPython: `CodeWriter.setup_vrefinfo(self, vrefinfo)`
    /// (codewriter.py:91-94).
    ///
    /// ```python
    /// def setup_vrefinfo(self, vrefinfo):
    ///     # must be called at most once
    ///     assert self.callcontrol.virtualref_info is None
    ///     self.callcontrol.virtualref_info = vrefinfo
    /// ```
    ///
    /// Note: pyre has no `virtualref` machinery
    /// (no `@jit.virtual_ref`, no `vref_info` lookup); the slot is
    /// preserved so future warmspot wiring can call through with the
    /// same name.
    pub fn setup_vrefinfo(&self, vrefinfo: ()) {
        // codewriter.py:93 `assert self.callcontrol.virtualref_info is None`.
        assert!(self.callcontrol().virtualref_info.is_none());
        // codewriter.py:94 `self.callcontrol.virtualref_info = vrefinfo`.
        self.callcontrol().virtualref_info = Some(vrefinfo);
    }

    /// RPython: `CodeWriter.setup_jitdriver(self, jitdriver_sd)`
    /// (codewriter.py:96-99).
    ///
    /// ```python
    /// def setup_jitdriver(self, jitdriver_sd):
    ///     # Must be called once per jitdriver.
    ///     self.callcontrol.jitdrivers_sd.append(jitdriver_sd)
    /// ```
    ///
    /// Note: RPython appends unconditionally because
    /// each `@jit_callback` decoration calls `setup_jitdriver` exactly
    /// once at translation time. pyre's portal discovery is lazy and
    /// fires on every JIT entry, so the same `portal_graph` would be
    /// pushed repeatedly without the `find` guard below — `jitdrivers_sd`
    /// would grow linearly with JIT entries instead of staying bounded
    /// by the number of unique portals. The dedup updates the existing
    /// jd's `merge_point_pc` so the refinement hint propagates into
    /// the next `grab_initial_jitcodes` pass.
    pub fn setup_jitdriver(&self, jitdriver_sd: super::call::JitDriverStaticData) {
        let jitdriver_sd = jitdriver_sd.canonicalized();
        let cc = self.callcontrol();
        if let Some(existing) = cc
            .jitdrivers_sd
            .iter_mut()
            .find(|j| j.portal_graph == jitdriver_sd.portal_graph)
        {
            if jitdriver_sd.merge_point_pc.is_some() {
                existing.merge_point_pc = jitdriver_sd.merge_point_pc;
            }
            return;
        }
        // codewriter.py:99 `self.callcontrol.jitdrivers_sd.append(jitdriver_sd)`.
        cc.jitdrivers_sd.push(jitdriver_sd);
    }

    /// RPython: `self.callcontrol` (codewriter.py:23).
    ///
    /// Returns a mutable reference to the owned `CallControl`. Safe under
    /// the same invariant as the legacy `JITCODE_CACHE` thread_local: the
    /// caller must not re-enter `callcontrol()` while the returned borrow
    /// is live.
    #[allow(clippy::mut_from_ref)]
    pub fn callcontrol(&self) -> &mut super::call::CallControl {
        // SAFETY: `CodeWriter` is only accessed via `instance()` which
        // returns a thread-local reference; all callers execute on the
        // owning thread.
        unsafe { &mut *self.callcontrol.get() }
    }

    /// Access the process-wide single `CodeWriter` — analog of the
    /// single `codewriter` owned by `warmspot.py:245-281` for the
    /// lifetime of the JIT.
    ///
    /// Implemented as a per-thread singleton: pyre's JIT currently runs
    /// one interpreter per thread and function pointers in `Self` are
    /// `Sync`, so a thread-local provides the RPython "one CodeWriter
    /// per warmspot" invariant without a global lock.
    pub fn instance() -> &'static CodeWriter {
        thread_local! {
            static INSTANCE: CodeWriter = CodeWriter::new();
        }
        INSTANCE.with(|cw| unsafe { &*(cw as *const CodeWriter) })
    }

    /// Transform a Python CodeObject into a JitCode.
    ///
    /// RPython: CodeWriter.transform_graph_to_jitcode(graph, jitcode, verbose, index)
    ///
    /// Python bytecodes serve as the "graph". Since they are already linear
    /// and register-allocated, jtransform/regalloc/flatten are identity
    /// transforms. We go directly to assembly.
    pub fn transform_graph_to_jitcode(
        &self,
        code: &CodeObject,
        w_code: *const (),
        merge_point_pc: Option<usize>,
    ) -> PyJitCode {
        // jtransform.py:840 — the portal `frame` (and `ec`) red args are
        // threaded into every vable op from the start. Compute the graph
        // Variables once at function entry so all vable graph-shadow
        // emit sites can reference the same `frame_var`/`ec_var` pair
        // (instead of substituting a `Constant::none()` sentinel).
        let (frame_var, ec_var) = portal_graph_inputvars(code);
        // RPython codewriter.py:46-48 `regallocs[kind] = perform_register_allocation(graph, kind)`.
        // pyre's regalloc is trivial — fast locals occupy the bottom of
        // the ref register file and the operand stack stacks above
        // them — so the "allocation" reduces to a `RegisterLayout`
        // computed from `code.varnames` / `code.max_stackdepth`.
        let layout = RegisterLayout::compute(code);
        let RegisterLayout {
            nlocals,
            stack_base_absolute,
            max_stackdepth,
            stack_base,
        } = layout;
        // Per-arm fresh int scratch slots.
        // Each opcode handler arm that needs a transient int-typed
        // register calls `ssarepr.fresh_var(Kind::Int, scratch_int_base)`
        // (flatten.rs:`SSARepr::fresh_var`) to claim a unique pre-regalloc
        // slot above `scratch_int_base`. Non-overlapping arm Variables
        // coalesce into the same post-coloring color via the chordal
        // allocator (`regalloc::allocate_registers`); overlapping ranges
        // get distinct colors. The single SSARepr counter replaces the
        // earlier `next_scratch_int_slot` local — fresh_var is now the
        // sole int-bank scratch source.
        let scratch_int_base: u16 = 1;
        // `interp_jit.py:64` portal red `(frame, ec)` registers — pre-regalloc
        // placeholder slots in the conflated Ref index space. Their final
        // post-regalloc colors are looked up from `alloc_result.rename` after
        // `apply_rename` runs (see below). Slot `+10` was the dedicated
        // `null_ref_reg` PY_NULL holder before null_ref_reg retirement retired it; the
        // portal red regs keep their numerical positions so layout-sensitive
        // tests stay stable.
        let (portal_frame_reg, portal_ec_reg) =
            portal_red_pre_regalloc_slots(nlocals, max_stackdepth);
        // Per-arm fresh ref scratch slots.  Each opcode handler arm that needs
        // a transient ref-typed register allocates one or more fresh slots
        // from this counter instead of sharing the historical
        // `obj_tmp0`/`obj_tmp1` fixed slots.  Non-overlapping handler-arm
        // live ranges let the chordal coloring in
        // `regalloc::allocate_registers` (`regalloc.py:8-15`) coalesce
        // distinct allocations into the same post-coloring color, while
        // killing the cross-arm conflated Variable that previously caused
        // scratch slots to appear "alive" at unrelated `-live-` markers.
        let scratch_ref_base: u16 = portal_ec_reg + 1;
        // Note: literal field indices crystallised at
        // the codewriter call site. RPython looks up the index dynamically
        // through `VABLEINFO.static_field_descrs` since each backend may
        // reorder fields. Pyre's `_virtualizable_` order matches PyPy
        // `interp_jit.py:25-31` line by line:
        // [last_instr, pycode, valuestackdepth, debugdata, lastblock,
        // w_globals], so the literals match
        // `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS`.
        const VABLE_CODE_FIELD_IDX: u16 = 1;
        const VABLE_VALUESTACKDEPTH_FIELD_IDX: u16 = 2;
        const VABLE_NAMESPACE_FIELD_IDX: u16 = 5;

        // regalloc.py: compile-time stack depth counter — tracks which
        // stack register (stack_base + depth) is the current TOS.
        // `current_depth` is kept synchronised with
        // `current_state.stack.len()` (pyframe.py `valuestack_w`):
        // every emit_pushvalue_ref! / popvalue_ref! callsite and every
        // direct +=/-= maintains `current_state.stack` alongside the
        // depth bump, so a consumer that needs the FlowValue at
        // `stack_base + depth - 1` reads `current_state.stack.last()`.
        let mut current_depth: u16 = 0;

        // RPython: self.assembler = Assembler() + JitCode(graph.name, ...)
        // (rpython/jit/codewriter/jitcode.py:14-15 takes name as the first
        // __init__ argument; majit's JitCodeBuilder::set_name mirrors that).
        let mut assembler = SSAReprEmitter::new();
        assembler.set_name(code.obj_name.to_string());
        // Grow an `SSARepr` per-block via `push_walker_emit`; at drain
        // time the per-block accumulators concatenate into
        // `ssarepr.insns` which feeds
        // `jit::assembler::Assembler::assemble`.
        let mut ssarepr = SSARepr::new(code.obj_name.to_string());

        // Walker slot bridge: each entry maps a graph `Variable.id`
        // to the SSARepr slot the walker assigned when emitting the
        // dual-write.  `walker_post_walk_insert_renamings`
        // consults it so post-walk `insert_renamings` emits `<kind>_copy`
        // ops against the same walker slots the surrounding inline emits
        // wrote, matching `flatten.py:306-334` color-resolution semantics
        // under shared regalloc.  Synthetic graph-only Variables (no
        // walker counterpart) leave their entry as `None` and fall back
        // to graph regalloc.
        let mut walker_slot_for_variable: Vec<Option<u16>> = Vec::new();
        macro_rules! pin {
            ($var:expr, $slot:expr $(,)?) => {
                pair_walker_slot(&mut walker_slot_for_variable, $var, $slot)
            };
        }
        macro_rules! pin_if_absent {
            ($var:expr, $slot:expr $(,)?) => {
                pair_walker_slot_if_absent(&mut walker_slot_for_variable, $var, $slot)
            };
        }
        // Seed the bridge with the portal `(frame, ec)` red Variables —
        // upstream `jtransform.py:840` threads `frame` (and `ec`) as the
        // leading red args of every vable op; pyre's walker reads them
        // out of `portal_frame_reg` / `portal_ec_reg` rather than allocating
        // them per-op, so canonical needs the same fixed slots.
        pin!(Some(frame_var), portal_frame_reg);
        pin!(Some(ec_var), portal_ec_reg);
        // Function args occupy graph startblock inputargs `VariableId(0..nargs)`
        // and live in walker register slots `0..nargs` (the fast-local
        // bank base).  Without seeding them, ref_return on `arg0` falls
        // through to graph regalloc and gets a different colour than the
        // walker's `Operand::reg(Kind::Ref, 0)` emit.
        for idx in 0..entry_arg_slots(code) as u16 {
            pin!(
                Some(super::flow::Variable::new(
                    super::flow::VariableId(idx as u32),
                    Kind::Ref,
                )),
                idx,
            );
        }

        // RPython regalloc.py: keep kind-separated register files.
        // Soft minimums; `touch_reg` auto-grows the files as the dispatch
        // loop emits writes against fresh per-arm scratch slots
        // (`ssarepr.fresh_var(Kind::{Ref,Int}, scratch_*_base)`).
        assembler.ensure_r_regs(portal_ec_reg + 1);
        assembler.ensure_i_regs(scratch_int_base);

        // Register helper fn pointers in the canonical order; the
        // returned struct names every index so the dispatch handlers
        // below can reference them by field instead of an opaque local.
        let FnPtrIndices {
            call_fn:
                HelperHandle {
                    idx: call_fn_idx,
                    flavor: _call_fn_flavor,
                },
            load_global_fn:
                HelperHandle {
                    idx: load_global_fn_idx,
                    flavor: _load_global_fn_flavor,
                },
            compare_fn:
                HelperHandle {
                    idx: compare_fn_idx,
                    flavor: _compare_fn_flavor,
                },
            binary_op_fn:
                HelperHandle {
                    idx: binary_op_fn_idx,
                    flavor: _binary_op_fn_flavor,
                },
            box_int_fn:
                HelperHandle {
                    idx: box_int_fn_idx,
                    flavor: _box_int_fn_flavor,
                },
            truth_fn:
                HelperHandle {
                    idx: truth_fn_idx,
                    flavor: _truth_fn_flavor,
                },
            load_const_fn:
                HelperHandle {
                    idx: load_const_fn_idx,
                    flavor: _load_const_fn_flavor,
                },
            store_subscr_fn:
                HelperHandle {
                    idx: store_subscr_fn_idx,
                    flavor: _store_subscr_fn_flavor,
                },
            build_list_fn:
                HelperHandle {
                    idx: build_list_fn_idx,
                    flavor: _build_list_fn_flavor,
                },
            build_tuple_fn:
                HelperHandle {
                    idx: build_tuple_fn_idx,
                    flavor: _build_tuple_fn_flavor,
                },
            unpack_sequence_fn:
                HelperHandle {
                    idx: unpack_sequence_fn_idx,
                    flavor: _unpack_sequence_fn_flavor,
                },
            unpack_item_fn:
                HelperHandle {
                    idx: unpack_item_fn_idx,
                    flavor: _unpack_item_fn_flavor,
                },
            build_slice_fn:
                HelperHandle {
                    idx: build_slice_fn_idx,
                    flavor: _build_slice_fn_flavor,
                },
            normalize_raise_varargs_fn:
                HelperHandle {
                    idx: normalize_raise_varargs_fn_idx,
                    flavor: _normalize_raise_varargs_fn_flavor,
                },
            call_fn_0:
                HelperHandle {
                    idx: call_fn_0_idx,
                    flavor: _call_fn_0_flavor,
                },
            call_fn_2:
                HelperHandle {
                    idx: call_fn_2_idx,
                    flavor: _call_fn_2_flavor,
                },
            call_fn_3:
                HelperHandle {
                    idx: call_fn_3_idx,
                    flavor: _call_fn_3_flavor,
                },
            call_fn_4:
                HelperHandle {
                    idx: call_fn_4_idx,
                    flavor: _call_fn_4_flavor,
                },
            call_fn_5:
                HelperHandle {
                    idx: call_fn_5_idx,
                    flavor: _call_fn_5_flavor,
                },
            call_fn_6:
                HelperHandle {
                    idx: call_fn_6_idx,
                    flavor: _call_fn_6_flavor,
                },
            call_fn_7:
                HelperHandle {
                    idx: call_fn_7_idx,
                    flavor: _call_fn_7_flavor,
                },
            call_fn_8:
                HelperHandle {
                    idx: call_fn_8_idx,
                    flavor: _call_fn_8_flavor,
                },
            get_current_exception_fn:
                HelperHandle {
                    idx: get_current_exception_fn_idx,
                    flavor: _get_current_exception_fn_flavor,
                },
            set_current_exception_fn:
                HelperHandle {
                    idx: set_current_exception_fn_idx,
                    flavor: _set_current_exception_fn_flavor,
                },
        } = register_helper_fn_pointers(&mut assembler, self.cpu());

        // codewriter.py:37 `portal_jd = self.callcontrol.jitdriver_sd_from_portal_graph(graph)`
        // — RPython looks up portal-ness in the registry that
        // `setup_jitdriver` populates. pyre matches that: a code is a
        // portal iff it is in `CallControl.jitdrivers_sd`. The portal
        // path (`register_portal_jitdriver`) registers before the drain
        // runs `transform_graph_to_jitcode`, so the lookup must happen
        // before creating the shadow graph / entry FrameState below.
        // `merge_point_pc` is only a pyre refinement hint; `None` still
        // means "registered portal whose exact merge PC is not known yet",
        // not "non-portal".
        let portal_jd_index = self
            .callcontrol()
            .jitdriver_sd_from_portal_graph(code as *const CodeObject);
        let is_portal = portal_jd_index.is_some();
        // Every per-code jitcode threads the universal `self` red frame;
        // the portal additionally threads `ec` (`jitdriver_sd.reds =
        // [frame, ec]`).  Non-portal callees carry frame only.
        let frame_inputs = if is_portal {
            FrameInputs::Portal
        } else {
            FrameInputs::Frame
        };

        // Populate `cpu.lowering_ctx` with the four retired-family fn
        // indices so the canonical `flatten.rs::flatten_graph(graph,
        // regallocs, _include_all_exc_links, cpu)` driver can dispatch
        // pre-rtype HLOps (`add`/`lt`/`bool`/`setitem`) to the matching
        // `residual_call_*` Insn shape (`flatten.rs::
        // try_flatten_retired_family_hlop_to_insn`).  Upstream's
        // `flatten_graph` doesn't take a `lowering_ctx` because its
        // rtyper pre-rewrites these HLOps; pyre's lowering happens at
        // flatten time so the dispatcher needs the indices.
        if let Ok(mut guard) = self.cpu().lowering_ctx.write() {
            *guard = Some(super::flatten::LoweringContext {
                binary_op_fn_idx,
                compare_op_fn_idx: compare_fn_idx,
                truth_fn_idx,
                store_subscr_fn_idx,
                build_list_fn_idx,
                // `[u16; 9]` indexed by nargs (0..=8) per
                // [[feedback-no-hashmap-ever]].  `call_fn_idx` (nargs=1)
                // is the unsuffixed binding from line 3153; the suffixed
                // 0/2..=8 fill the surrounding slots.
                call_fn_idx_by_nargs: [
                    call_fn_0_idx,
                    call_fn_idx,
                    call_fn_2_idx,
                    call_fn_3_idx,
                    call_fn_4_idx,
                    call_fn_5_idx,
                    call_fn_6_idx,
                    call_fn_7_idx,
                    call_fn_8_idx,
                ],
                // Both portal and non-portal graphs now include
                // `frame_var` as a real startblock inputarg
                // (`graph_entry_inputargs(code, FrameInputs::Frame |
                // Portal)`), so `frame_var.id` (= `entry_arg_slots`) no
                // longer collides with the synthesised `return_var.id`
                // (= `start_inputargs.len()` = `entry_arg_slots + 1`).
                // `portal_frame_var` nonetheless stays `None` for
                // non-portal in this slice: it drives
                // `lower_simple_call_hlop_to_insn` (flatten.rs), and the
                // non-portal call/LOAD_GLOBAL readers still source the
                // frame from the shared `portal_frame_reg` slot.  Routing
                // them to the per-callee `frame_var` is a later slice.
                portal_frame_var: if is_portal { Some(frame_var) } else { None },
            });
        }

        // RPython flatten.py: pre-create labels for each block.
        // Python bytecodes are linear, so each instruction index gets a label.
        let num_instrs = code.instructions.len();
        let mut labels: Vec<u16> = Vec::with_capacity(num_instrs);
        for _ in 0..num_instrs {
            labels.push(assembler.new_label());
        }

        // shadow `FunctionGraph` alongside `ssarepr`.
        //
        // RPython's flow space keeps `framestate` on each `SpamBlock`
        // (`flowcontext.py:38-44`) and derives `Link.args ↔
        // target.inputargs` from `FrameState.getoutputargs()`. Pyre's
        // walker is still single-pass over Python bytecode, but the
        // shadow graph now carries the same per-block `FrameState`
        // object instead of a topology-only `BlockRef`.
        //
        // Every per-code jitcode carries the universal `self` red frame
        // as a startblock inputarg — upstream every opcode-handler graph
        // takes `self` (the interpreter frame) as `inputargs[0]`, and
        // pyre's one-jitcode-per-code-object model carries that single
        // red frame for both portal and non-portal callees.  The portal
        // additionally carries the JitDriver `ec` red (`jitdriver_sd
        // .reds = [frame, ec]`, warmspot.py).  `frame_inputs` selects:
        // `Portal` = frame + ec, `Frame` = frame only (non-portal
        // callee).  Both are appended to `startblock.inputargs` via
        // `graph_entry_inputargs(code, frame_inputs)` AND to `FrameState`
        // via `entry_frame_state(code, frame_inputs)`;
        // `FrameState.portal_extras` carries those Variables through
        // every block transition so `getoutputargs()` on any backedge
        // produces link args aligned with the appended startblock slots.
        let mut graph = new_shadow_graph_with_portal_inputs(code, frame_inputs);
        let (catch_for_pc, catch_sites, handler_depth_at) =
            decode_exception_catch_sites(&mut assembler, &mut graph, code, num_instrs);
        // `flowcontext.py:293 self.joinpoints = {}` — sparse-by-PC dict
        // keyed by `next_offset`, populated via `setdefault(...)` per
        // `flowcontext.py:426`.  Vec-of-Vec value preserves the candidate
        // list semantics for supersede where multiple SpamBlocks at the
        // same PC are tracked head-of-list.
        let mut joinpoints: VecMap<usize, Vec<SpamBlockRef>> = VecMap::new();
        let start_state = entry_frame_state(code, frame_inputs);
        // Collect every walker-created block in walker-visit order so the
        // post-walk drain can iterate per-block accumulators in the same
        // order their pushes reached `ssarepr.insns`.  Each block's
        // accumulator receives emits contiguously between the block's
        // first `emit_mark_label_pc!` and its terminator.
        let mut all_walker_blocks: Vec<SpamBlockRef> = Vec::new();
        if num_instrs > 0 {
            let start_block =
                SpamBlockRef::new(graph.startblock.clone(), Some(start_state.clone()));
            all_walker_blocks.push(start_block.clone());
            joinpoints.insert(0, vec![start_block]);
        }
        // Walker-time PC dispatch tracker.  Records every per-PC
        // `-live-` marker as `(SpamBlockRef, offset_in_block)` so a
        // drain-time resolver can pick the FIRST entry whose block
        // contributes non-empty content to the final SSARepr
        // (supersede / `mark_dead` clears the per-block accumulator,
        // so the canonical answer comes from the first entry in a
        // *live* block).  Vec-of-Vec mirrors `flowcontext.py:42
        // SpamBlock` where multiple SpamBlocks can record into the
        // same py_pc through joinpoint candidates
        // (`flowcontext.py:426 candidates =
        // self.joinpoints.setdefault(...)`).  Populated at walker
        // emit time so the runtime can resolve `pc_map` from
        // walker-tracked positions, without per-PC `Insn::Label`
        // anchors in the final SSARepr (label retirement).
        let mut walker_pc_live_marker_pos: Vec<Vec<(SpamBlockRef, usize)>> =
            vec![Vec::new(); num_instrs];
        // Catch landings live on `ExceptionCatchSite::landing` (decode-
        // time SpamBlockRef::new with framestate=None) and are NOT pushed
        // to `all_walker_blocks` at creation — they're queued at emission
        // time inside `emit_mark_label_catch_landing!` so the drain order
        // reflects walker emission order (catch landings emit AFTER the
        // main walker loop completes per `codewriter.rs::6907+`).
        // The walker emits into `current_block`; `emit_mark_label_pc!` and
        // `emit_mark_label_catch_landing!` reassign it as the walker enters
        // each block. Initialised to the first PC block so the
        // `Label("pcN")` / live_placeholder / jit_merge_point
        // emissions that precede the first `emit_mark_label_pc!`
        // belong to it.
        let mut current_block: SpamBlockRef = joinpoints
            .get(&0)
            .and_then(|blocks| blocks.first().cloned())
            .unwrap_or_else(|| {
                let synthetic =
                    SpamBlockRef::new(graph.startblock.clone(), Some(start_state.clone()));
                all_walker_blocks.push(synthetic.clone());
                synthetic
            });
        // Per-block contiguous walker — `emit_mark_label_pc!` sets
        // `block_switch_pending = true` at block transitions instead
        // of switching `current_block` inline; the inner for-loop
        // checks the flag after each per-PC emit and breaks, yielding
        // to the outer `while let Some(pending_block) =
        // pendingblocks.pop_front()` which picks up the queued new
        // block in the next iteration.  Mirrors upstream's
        // `flowcontext.py:407-416 record_block` shape where each
        // block is processed contiguously without mid-iteration
        // re-entry.  Correctness relies on the explicit `goto
        // Label("pcN")` + `Unreachable` pair emitted on the yield
        // path, aligning with `flatten.py:177-258 insert_exits`.
        let mut block_switch_pending: bool = false;
        let mut current_state = current_block
            .framestate()
            .unwrap_or_else(|| start_state.clone());
        // Tracks whether the current block still needs an implicit
        // fallthrough `Link` on the next `emit_mark_label_pc!`. Reset
        // to `true` at every block entry; terminator macros that fully
        // close the block (`emit_goto!` / `emit_ref_return!` /
        // `emit_raise!` / `emit_reraise!` / `emit_abort_permanent!`)
        // clear it. Terminators that leave fallthrough open —
        // `emit_goto_if_not!`, `emit_goto_if_not_int_is_zero!`,
        // `emit_catch_exception!` — keep it set. Mirrors RPython
        // `flatten.py:240-267` where a conditional / exception exit
        // always coexists with the straight-line successor on
        // `Block.exits`.
        let mut needs_fallthrough: bool = true;
        // pending_bool_fallthrough_case retired: PopJumpIfFalse now mirrors
        // PopJumpIfTrue by attaching both Bool exit links at the branch
        // point via explicit mergeblock + set_last_bool_exitcase pairs.
        // The deferred case-application across emit_mark_label_pc no
        // longer exists.

        // rpython/flowspace/flowcontext.py:399-405 `build_flow` parity:
        // `pendingblocks = deque([startblock])` + `while pendingblocks:
        // block = pendingblocks.popleft(); record_block(block)`.
        // Queue element is the block itself (flowcontext.py:401); the
        // framestate (`block.framestate.next_offset`) is read on pop
        // (flowcontext.py:408-409).
        //
        // Declared here so `emit_mark_label_pc!`, `emit_goto!`,
        // `emit_goto_if_not!`, `emit_goto_if_not_int_is_zero!` (macro
        // definitions below) resolve it at expansion — macro_rules
        // hygiene requires captured identifiers be in scope at the
        // macro DEFINITION site.
        let mut pendingblocks: VecDeque<SpamBlockRef> = VecDeque::new();
        // Upstream `build_flow` relies on `block.dead` alone
        // (`flowcontext.py:404 if not block.dead: record_block(block)`).
        // Pyre matches this: supersede may re-walk a PC under widened
        // framestate, producing duplicate `-live-` markers.
        // `walker_pc_live_marker_pos` uses first-live-wins resolution
        // so the runtime canonical bytes are the dead block's emit;
        // the supersede newblock's re-walk emit is unreachable
        // through the resolved `pc_map`.

        // interp_jit.py:118 `pypyjitdriver.can_enter_jit(...)` is called in
        // `jump_absolute` (`jumpto < next_instr` branch), i.e. at each
        // Python backward jump.  jtransform.py:1714-1723
        // `handle_jit_marker__can_enter_jit = handle_jit_marker__loop_header`
        // lowers each one to a `loop_header` jitcode op.  Pyre has no
        // `jump_absolute` Python wrapper — the equivalent is to pre-scan
        // `JumpBackward` opcodes and record their targets; each target PC
        // becomes a `loop_header` site.
        let loop_header_pcs = find_loop_header_pcs(code);
        // Pre-scanned set of every block-entry PC.  Used
        // by emit_mark_label_pc to force a block boundary (call
        // mergeblock to close current_block + create/match a fresh
        // SpamBlock for py_pc) when the walker reaches a branch
        // target sequentially, instead of letting current_block span
        // the boundary.  Mirrors upstream's `flowcontext.py:425-435
        // set_branch` which creates `joinpoints[py_pc]` candidates at
        // every branch destination — pyre's pre-scan front-loads the
        // same set so the per-block walker iteration matches per-block
        // record_block emission.
        let branch_target_pcs = find_branch_target_pcs(code);

        // RPython: flatten_graph() walks blocks and emits instruction tuples.
        // RPython: assembler.assemble(ssarepr, jitcode, num_regs) emits bytecodes.
        // For pyre, we combine both steps: walk Python bytecodes and emit
        // JitCode bytecodes directly.
        let mut arg_state = OpArgState::default();
        // liveness.py parity: record stack depth at each Python PC for
        // precise liveness generation. Stack registers stack_base..stack_base+depth
        // are live at each PC.
        let mut depth_at_pc: Vec<u16> = vec![0; num_instrs];
        // RPython parity: every backward jump goes through dispatch() →
        // jit_merge_point(). `merge_point_pc` is still threaded in from
        // bound_reached as the trace-entry refinement hint, but portal
        // jitcode emission must not restrict merge-point bytecodes to that
        // single PC: PyPy's dispatch loop reaches a portal merge point for
        // every bytecode dispatch, and nested Python loops rely on the
        // blackhole CRN at those inner headers to compile and target their
        // own loops instead of growing giant bridges.

        // pyframe.py:379-417 pushvalue/popvalue_maybe_none parity:
        // Each push/pop writes self.valuestackdepth = depth ± 1.
        // jtransform.py:923-928 lowers this to setfield_vable_i.
        // This macro emits the equivalent BC_SETFIELD_VABLE_I after
        // every current_depth mutation so the frame's valuestackdepth
        // stays in sync at every guard/call point — matching RPython's
        // per-push/per-pop semantics.
        //
        // Records a matching graph op pair (constant-source `int_copy`
        // producing a fresh Int Variable + `setfield_vable_i`
        // consuming it) alongside the SSA emission.  The SSA side
        // mirrors that shape via `ssarepr.fresh_var(Kind::Int, ...)`
        // so the canonical and walker streams agree on the VSD-sync
        // scratch's liverange.
        macro_rules! emit_vsd {
            ($depth:expr, $py_pc:expr) => {
                if is_portal {
                    let depth_value = (stack_base_absolute + $depth as usize) as i64;
                    // Graph-side shadow: produce a fresh Int Variable
                    // from a constant-source `int_copy` op and consume it
                    // in a matching `setfield_vable_i` op. Mirrors jtransform.py:844 +
                    // jtransform.py:925 so graph regalloc observes the
                    // liverange of the VSD-sync scratch.
                    //
                    // Record the graph offset as the
                    // walker's current py_pc.  Walker emits emit_vsd!
                    // INLINE during each PC's handler; canonical
                    // make_bytecode_block (flatten.rs:2258-2303) sorts ops
                    // by op.offset for per-PC label interleaving, and
                    // offset=-1 routes synthetic ops to current_pc (the
                    // last anchored PC), landing them in the wrong PC's
                    // region when emit_vsd ran mid-stream.  Recording
                    // py_pc anchors the synthetic to its actual owning PC.
                    let v_depth: super::flow::FlowValue =
                        super::flow::Constant::signed(depth_value).into();
                    record_graph_op(
                        &current_block.block(),
                        "setfield_vable_i",
                        vable_setfield_int_graph_args(
                            frame_var.into(),
                            v_depth.into(),
                            VABLE_VALUESTACKDEPTH_FIELD_IDX,
                        ),
                        None,
                        ($py_pc) as i64,
                    );
                    emit_vable_setfield_int_const!(
                        portal_frame_reg,
                        VABLE_VALUESTACKDEPTH_FIELD_IDX,
                        depth_value
                    );
                }
            };
        }

        // Post-emit bookkeeping for a stack-pushing handler: append the
        // produced FlowValue to the symbolic stack, bump `current_depth`,
        // and emit the VSD sync.  Mirrors the inline triplet that every
        // residual_call / HLOp-result emit runs after writing into the
        // dst slot.
        macro_rules! push_and_bump {
            ($value:expr, $py_pc:expr $(,)?) => {{
                current_state.stack.push($value);
                current_depth += 1;
                emit_vsd!(current_depth, $py_pc);
            }};
        }

        // Record a residual_call SpaceOperation on the current block.
        // Captures the two boilerplate arguments
        // (`&mut graph, &current_block.block()`) implicitly; positional
        // tail matches `record_residual_call_graph_op`.
        macro_rules! residual_call {
            (
                $fn_idx:expr,
                $flavor:expr,
                $args_i:expr,
                $args_r:expr,
                $args_f:expr,
                $arg_kinds:expr,
                $reskind:expr,
                $offset:expr $(,)?
            ) => {
                record_residual_call_graph_op(
                    &mut graph,
                    &current_block.block(),
                    $fn_idx,
                    $flavor,
                    $args_i,
                    $args_r,
                    $args_f,
                    $arg_kinds,
                    $reskind,
                    $offset,
                )
            };
        }

        // Note: the `BC_ABORT_PERMANENT` runtime
        // bytecode does not appear in `rpython/jit/codewriter/` or
        // `rpython/jit/metainterp/`. RPython refuses to build jitcode for
        // bytecodes it cannot translate (the translator surfaces the
        // failure at build time); pyre must always produce runnable
        // jitcode because bytecode translation is lazy at runtime. We
        // therefore keep the runtime-side adaptation (assembler emits
        // `BC_ABORT_PERMANENT` so the blackhole interpreter falls back to
        // CPython evaluation) but never surface the pyre-only opname into
        // the RPython-parity SSARepr layer — `flatten.py:106` uses plain
        // `Label` for loop headers and `assembler.py:159` does not encode
        // unsupported bytecodes as named opnames.

        // RPython parity:
        // `flatten.py:344` `self.emitline("last_exception", "->",
        // self.getcolor(w))` — `assembler.py:220` turns it into
        // `last_exception/>i`. Loads the thread-local exception class
        // pointer into a Signed register. Canonical
        // `generate_last_exc` emits this immediately before
        // `last_exc_value` at every exception link landing whose
        // `link.args` mentions `link.last_exception`.
        macro_rules! emit_last_exception {
            ($dst:expr) => {{
                let dst = $dst;
                let insn = Insn::op_with_result(
                    "last_exception",
                    Vec::new(),
                    Register::new(Kind::Int, dst),
                );
                push_walker_emit(&current_block, insn);
            }};
        }

        // RPython parity:
        // `flatten.py:347` `self.emitline("last_exc_value", "->",
        // self.getcolor(w))` — `assembler.py:220` turns it into
        // `last_exc_value/>r`. pyre emits this once per catch site to
        // load the thread-local exception into the handler's input
        // register.
        macro_rules! emit_last_exc_value {
            ($dst:expr) => {{
                let dst = $dst;
                let insn = Insn::op_with_result(
                    "last_exc_value",
                    Vec::new(),
                    Register::new(Kind::Ref, dst),
                );
                push_walker_emit(&current_block, insn);
            }};
        }

        // Note: the `BC_JUMP_TARGET` runtime opcode
        // does not appear in `rpython/jit/codewriter/`. RPython marks
        // loop-header block entries with a plain `Insn::Label` and lets
        // the blackhole's dispatch loop recognise them via the label
        // position; pyre emits a dedicated `BC_JUMP_TARGET` opcode so the
        // runtime inner-loop can cheaply identify back-edge targets
        // without consulting a label table. The runtime-side adaptation
        // stays (assembler dispatch at `assembler.rs:367-372`) but the
        // pyre-only opname is not surfaced into the RPython-parity
        // SSARepr layer — `flatten.py:106` uses plain `Label` for loop
        // headers.

        // Dual emission for `int_copy` / `ref_copy` /
        // `float_copy` with a Constant source. RPython parity:
        // `flatten.py:333` `self.emitline('%s_copy' % kind, v, "->", w)`
        // — `v` is resolved via `getcolor(v)` which returns either a
        // `Register` or an unchanged `Constant` (see `flatten.py:382-384`).
        // The `assembler.py:140-222` dispatch handles both: the Register
        // source emits an `int_copy/i>i` entry, and the Constant source
        // emits an `int_copy/c>i` entry (argcode `'c'` for a compact
        // Constant — `>` is the result marker per
        // `assembler.py:210-212`). pyre's legacy `load_const_{i,r,f}_value`
        // emits the same runtime bytes under pyre-only `load_const_*`
        // opnames; the SSARepr now carries the RPython-parity `*_copy`
        // name with a ConstInt/ConstRef/ConstFloat source operand.
        // Per-opname integer / float primitives — `int_add`, `int_sub`,
        // `int_mul`, `int_{floordiv,mod,and,or,xor,lshift,rshift}`,
        // `int_{eq,ne,lt,le,gt,ge}`, `int_neg`, `int_invert`,
        // `uint_{rshift,mul_high,lt,le,gt,ge}`, `float_{add,sub,mul,
        // truediv,neg,abs}` — flow through canonical RPython opnames
        // and the matching `record_binop_*` / `record_unary_*` arms in
        // `assembler.rs`. The build-time pyre codewriter currently
        // emits `BINARY_OP` / `COMPARE_OP` via polymorphic residual
        // calls because pyre can't prove static operand types from the
        // bytecode alone; the canonical per-opname handlers handle
        // emissions that come from #[jit_interp]-lowered macros.

        // Call family intentionally has NO dual-emit.
        //
        // `rpython/jit/codewriter/jtransform.py:414-435` `rewrite_call()`
        // emits `residual_call_{kinds}_{reskind}` with
        // `[fnptr_constant, ListOfKind(int)?, ListOfKind(ref),
        //   ListOfKind(float)?, calldescr]`. pyre's runtime ABI uses
        // a caller-order Register list plus a u16 helper-table index and
        // encodes `may_force` in the target bytecode — none of which
        // fit into RPython's SSA tuple shape. Reviewer guidance:
        //   "codewriter는 원본 RPython tuple 을 만들고, pyre 적응은
        //    assembler 가 해야 한다."
        // Rather than baking the pyre shape into the SSA (which would
        // ossify a pyre-only SSA vocabulary), we keep the call handlers
        // on the direct builder path until assembler.rs grows exact
        // `residual_call_*` dispatch that can reconstruct the pyre
        // caller-order list. See `B6_CODEWRITER_PIPELINE_PLAN.md`.

        // Every site that used
        // to invoke `assembler.ref_return(src)` now also appends an
        // `Insn::Op { opname: "ref_return", args: [Register(Ref, src)] }`
        // to the SSARepr so the future `Assembler::assemble` path
        // (`assembler.rs::dispatch_op:374`) can reproduce the same byte at
        // the switchover. The direct builder call stays until the
        // switchover runs so the emitted JitCode remains bit-identical.
        // RPython parity: `rpython/jit/codewriter/jtransform.py` emits
        // `op_ref_return(v)` via `rewrite_op_jit_return` for the portal
        // return path; `assembler.py:221` turns that into the `ref_return/r`
        // bytecode key.
        macro_rules! emit_ref_return {
            ($src:expr, $retval:expr) => {{
                let src = $src;
                let retval = $retval;
                let insn = Insn::op("ref_return", vec![Operand::reg(Kind::Ref, src)]);
                push_walker_emit(&current_block, insn);
                // `rpython/jit/codewriter/flatten.py:144-146`: terminators
                // emit `('---',)` so the backward liveness pass clears its
                // alive set.
                push_walker_emit(&current_block, Insn::Unreachable);
                // attach the return edge to
                // `graph.returnblock` (`model.py:18`). The return value
                // now comes from the symbolic `FrameState` stack,
                // matching `flatten.py:130-139` `make_return(args)`.
                let link =
                    super::flow::Link::new(vec![retval], Some(graph.returnblock.clone()), None)
                        .into_ref();
                append_exit(&current_block.block(), link);
                needs_fallthrough = false;
            }};
        }

        // RPython parity:
        // `flatten.py:161` `self.emitline('goto', TLabel(link.target))` —
        // `assembler.py:220` turns the op into `goto/L`. Pyre labels are
        // integer indices into `labels[]`, one per Python PC; the
        // `TLabel` carries the synthetic name `pc{target_py_pc}` so the
        // dispatch (`assembler.rs::dispatch_op:345`) can resolve
        // it against `builder_label`.
        macro_rules! emit_goto {
            ($target_py_pc:expr) => {{
                let target_py_pc = $target_py_pc;
                // mergeblock support: mergeblock first to learn the target
                // SpamBlock, then emit `goto` against its block-identity
                // `TLabel("block{addr}")`.  Mirrors upstream
                // `flatten.py:161 self.emitline('goto',
                // TLabel(link.target))` where `link.target` is a Block
                // identity, not a PC.  The per-PC `Label("pcN")` at the
                // target's first PC remains a valid branch target until
                // per-PC labels are retired.
                let target_block = mergeblock(
                    code,
                    &mut graph,
                    &mut joinpoints,
                    &current_block,
                    &{
                        let mut branch_state = current_state.clone();
                        branch_state.next_offset = target_py_pc;
                        branch_state.blocklist = frame_blocks_for_offset(code, target_py_pc);
                        branch_state
                    },
                    target_py_pc,
                    &mut pendingblocks,
                    &mut all_walker_blocks,
                );
                let insn = Insn::op(
                    "goto",
                    vec![Operand::TLabel(super::flatten::block_tlabel(
                        &target_block.block(),
                    ))],
                );
                push_walker_emit(&current_block, insn);
                // `rpython/jit/codewriter/flatten.py:111-112`: an
                // unconditional goto implicitly ends a block so the
                // liveness pass (`liveness.py:68-69`) can reset the alive
                // set.
                push_walker_emit(&current_block, Insn::Unreachable);
                needs_fallthrough = false;
            }};
        }

        // The opname is
        // a pyre-only runtime construct (`BC_ABORT_PERMANENT`) with no
        // counterpart in `rpython/jit/codewriter/*` or
        // `rpython/jit/metainterp/*` — pyre uses it to short-circuit the
        // translation of unsupported bytecode handlers and permanent
        // guard-fail edges, which upstream sidesteps via
        // `rpython/jit/metainterp/policy.py`-driven whitelisting. Because
        // the opname *already* surfaces at the runtime layer, the
        // single-SSARepr requirement forces it through the walker-local
        // `ssarepr` too; the alternative — a hybrid "some ops go through
        // SSARepr, some don't" dispatch — defeats the purpose of the
        // switchover. `dispatch_op` in `assembler.rs:510` already routes
        // `"abort_permanent"` to the builder, so the external push is
        // an exact mirror of the pre-existing internal behavior.
        macro_rules! emit_abort_permanent {
            () => {{
                let insn = Insn::op("abort_permanent", Vec::new());
                push_walker_emit(&current_block, insn);
                // Graph-side dual-write so the canonical `flatten_graph`
                // driver sees the same `abort_permanent` SpaceOp via
                // passthrough.  Without this dual-write, canonical
                // would omit the runtime bail-out marker that pyre's
                // walker emits inline — production-flip-unsafe (the
                // compiled trace would continue past unsupported
                // opcodes instead of falling back to the interpreter).
                // `abort_permanent` is pyre-specific (no upstream
                // RPython counterpart); use `offset = -1` matching
                // `emit_vsd!`'s synthetic-op convention since
                // `abort_permanent` is an emission-time bail-out
                // marker, not tied to a single Python bytecode PC.
                record_graph_op(
                    &current_block.block(),
                    "abort_permanent",
                    Vec::new(),
                    None,
                    -1,
                );
                // pyre-only dead-end: the block has no successor in
                // the shadow graph. Leaving `needs_fallthrough = false`
                // blocks the auto-fallthrough at the next
                // `emit_mark_label_pc!`.
                needs_fallthrough = false;
            }};
        }

        // RPython parity:
        // `flatten.py` emits `self.emitline("raise", self.getcolor(args[1]))`
        // inside the exception-link handler; `assembler.py:220` turns it
        // into `raise/r`. pyre's single `emit_raise(exc_reg)` call site
        // (RAISE_VARARGS with argc >= 1) corresponds to the same edge.
        // `$try_catch_adjacency` — when `true`, this raise is an
        // explicit `RAISE_VARARGS argc >= 1` op and may be covered by
        // a CPython 3.13 exception-table range. In that case, emit a
        // `catch_exception/L<L_handlers>` *immediately byte-adjacent*
        // to `raise/r` so blackhole's `handle_exception_in_frame`
        // (`blackhole.py:396-408`) can find the catch dispatch right
        // after `bh.position` advances past the raise. This mirrors
        // `flatten.py:194-209` canraise-arm shape.
        //
        // When `false` (the RERAISE call site), keep the legacy
        // `Raise.nomoreblocks` shape: link to `graph.exceptblock` and
        // skip catch_exception adjacency. RERAISE already
        // semantically *is* the exception-propagation op, and its
        // surrounding stack/last_exception shape differs from the
        // explicit-raise shape that the catch-landing union path
        // expects.
        macro_rules! emit_raise {
            ($src:expr, $evalue:expr, $offset:expr, $try_catch_adjacency:expr) => {{
                let src = $src;
                let evalue_fv: super::flow::FlowValue = $evalue;
                let offset = $offset;
                let try_catch_adjacency: bool = $try_catch_adjacency;
                let insn = Insn::op("raise", vec![Operand::reg(Kind::Ref, src)]);
                push_walker_emit(&current_block, insn);
                let py_pc_for_catch = offset as usize;
                let catch_label_opt = if try_catch_adjacency {
                    catch_for_pc.get(py_pc_for_catch).copied().flatten()
                } else {
                    None
                };
                if let Some(catch_label) = catch_label_opt {
                    // Raise inside a try/except range: RPython
                    // canraise-arm shape. The block's exception edge
                    // goes to the catch landing, not `graph.exceptblock`.
                    // `emit_catch_exception!` both pushes the
                    // `catch_exception/L<catch_landing_{label}>` insn
                    // AND calls `attach_catch_exception_edge` (the
                    // exception Link onto `current_block.exits`).
                    emit_catch_exception!(catch_label);
                } else {
                    // `flowcontext.py:1246-1261 Raise.nomoreblocks` shape:
                    //   link = Link([w_exc.w_type, w_exc.w_value],
                    //               ctx.graph.exceptblock)
                    // `w_exc.w_value` is the actual trace-level FlowValue
                    // of the raised exception instance; `w_exc.w_type`
                    // upstream is a statically-known Constant because flow
                    // analysis sees the `raise SomeError(...)` source form.
                    //
                    // pyre still emits a single runtime `raise/r`, but the
                    // shadow graph can mirror `flowcontext.py:635-636`
                    // exactly by recording `w_type = type(w_value)` and
                    // routing that result through the explicit raise edge.
                    // Like upstream `Raise.nomoreblocks`, this edge does
                    // NOT use `link.extravars`.
                    let edge_state = explicit_raise_state(
                        &mut graph,
                        &current_block.block(),
                        &current_state,
                        evalue_fv,
                        offset,
                    );
                    let link = super::flow::Link::new(
                        exceptblock_link_args(&edge_state),
                        Some(graph.exceptblock.clone()),
                        None,
                    );
                    let link = link.into_ref();
                    let _ = edge_state;
                    append_exit(&current_block.block(), link);
                }
                needs_fallthrough = false;
            }};
        }

        // RPython parity:
        // `flatten.py` emits the zero-arg `self.emitline("reraise")` for
        // the re-raise edge; `assembler.py:220` turns it into
        // `reraise/`. pyre emits this for RAISE_VARARGS with argc == 0.
        macro_rules! emit_reraise {
            () => {{
                let insn = Insn::op("reraise", Vec::new());
                push_walker_emit(&current_block, insn);
                // same edge as `emit_raise!` — the
                // re-raise opname shares the `Block.exits` topology
                // (`flatten.py` emits the two as alternative codings
                // of the same exception exit).
                //
                // `reraise` preserves the current handler exception in
                // `FrameState.last_exception` (framestate.py:22).
                // Upstream `rpython/jit/codewriter/flatten.py:161-162`
                // `make_exception_link` asserts
                //     assert link.last_exception is not None
                //     assert link.last_exc_value is not None
                // before emitting `reraise`, so reaching this macro
                // with `current_state.last_exception == None` is a
                // structural bug in the caller rather than a normal
                // path. Fail loudly instead of quietly constructing
                // a sentinel-filled exit link.
                let (etype, evalue) = exception_edge_extravars(&current_state);
                let mut link = super::flow::Link::new(
                    exceptblock_link_args(&current_state),
                    Some(graph.exceptblock.clone()),
                    None,
                );
                // `flowcontext.py:141-143` `guessexception` / `model.py:
                // 127-129 Link.extravars`: pass the exception pair as
                // both `link.args` and `link.extravars` so the
                // downstream `flatten.py:163-174 make_exception_link`
                // check `link.args == [link.last_exception,
                // link.last_exc_value]` matches and emits `reraise`.
                link.extravars(Some(etype), Some(evalue));
                let link = link.into_ref();
                append_exit(&current_block.block(), link);
                needs_fallthrough = false;
            }};
        }

        // RPython parity:
        // `flatten.py` emits `self.emitline('catch_exception',
        // TLabel(block.exits[0]))` when a block has an exception edge;
        // `assembler.py:220` turns it into `catch_exception/L`. pyre
        // emits this after each Python PC that has an exception handler.
        // The catch landing block lives after the main loop
        // (`mark_label(site.landing_label)`), so the `TLabel` carries
        // `catch_landing_{landing_label}` — distinct from the
        // `pc{py_pc}` naming used for PC-indexed labels.
        macro_rules! emit_catch_exception {
            ($catch_label:expr) => {{
                let catch_label = $catch_label;
                let insn = Insn::op(
                    "catch_exception",
                    vec![Operand::TLabel(TLabel::new(format!(
                        "catch_landing_{}",
                        catch_label
                    )))],
                );
                push_walker_emit(&current_block, insn);
                // attach the exception edge to the
                // current PC's block. In RPython this is the
                // `Constant(last_exception)` exit added by
                // `flatten.py` when the block `canraise`; the matching
                // normal-control-flow Link (fallthrough / goto) is
                // added by its own emit macro so the two edges coexist
                // on `Block.exits`.
                let landing = catch_sites
                    .iter()
                    .find(|s| s.landing_label == catch_label)
                    .expect("catch_sites entry for catch_label")
                    .landing
                    .clone();
                attach_catch_exception_edge(
                    &mut graph,
                    &current_block.block(),
                    &landing,
                    &current_state,
                );
            }};
        }

        // Dual emission for block `Label`. RPython parity:
        // `flatten.py:180` `self.emitline(Label(block))` marks block
        // entry; `assembler.py:157-158` records the label position in
        // `self.label_positions`. pyre marks a label at every Python PC
        // (`mark_label(labels[py_pc])`) and at each catch landing
        // block's entry. The two naming schemes (`pc{py_pc}` vs
        // `catch_landing_{u16}`) match the TLabel schemes used by
        // `emit_goto!` and `emit_catch_exception!`.
        macro_rules! emit_mark_label_pc {
            ($py_pc:expr) => {{
                let py_pc = $py_pc;
                // NOTE: the program-wide `ssarepr.insns` Label push is
                // DEFERRED until the switch check below.  When the
                // gate is on and a switch is detected, both ssarepr
                // and per_block_ssarepr stay un-pushed at this PC —
                // the new block's outer iter will emit its own Label
                // at PC=py_pc.  When no switch (gate off or same
                // block), push Label to ssarepr + per_block.
                // if the previous block still needs
                // a fallthrough edge AND we're not already standing in
                // the block for `py_pc`, attach one before switching
                // `current_block`.
                //
                // The `current_state.next_offset != py_pc` guard skips
                // the self-loop edge that would otherwise land at the
                // very first PC of every walker-pop iteration: each
                // pop sets `current_block = pending_block` whose
                // `current_state.next_offset == start_pc`, and the
                // first iteration of the inner `for py_pc in
                // start_pc..` would call `mergeblock(currentblock=
                // pending_block, py_pc=start_pc)` — a no-op transition
                // whose only side-effect is to `append_exit`
                // a `pending_block → pending_block` self-loop, leaving
                // every empty pending block with two outgoing edges
                // (the self-loop + the next PC's fallthrough) and no
                // exitswitch.  RPython's `flowcontext.py:407-475` walks
                // per-block, never invoking the joinpoint-merge path
                // when "entering" a block — pyre's PC-sequential walker
                // is the adaptation, but the join check belongs only on
                // PC transitions, not on PC entry.
                // Force a block boundary when the walker
                // reaches a pre-scanned branch target PC sequentially.
                // Without this, current_block would continue past the
                // boundary via arm 3's self-registration, and a later
                // back-edge to py_pc would create an orphan via
                // make_next_block (next_offset mismatch in mergeblock's
                // union loop).  The block-start case is excluded by
                // checking that current_block.framestate.next_offset !=
                // py_pc — we are not yet at start, so we need to close
                // current_block at the boundary.
                // `force_branch_boundary` is a pyre-walker adaptation to
                // create an explicit block transition when the PC-
                // sequential iterator reaches a branch target without
                // an intervening branch (mirroring upstream's per-block
                // walker boundary at branch entries).  It must NOT
                // fire when `current_block` has already been closed by
                // an explicit branch terminator (POP_JUMP_IF_*,
                // emit_goto*, etc.), because those branches already
                // appended both the linkfalse and linktrue exits with
                // proper exitcase stamps — forcing another
                // `mergeblock(currentblock=L1, target=fallthrough_pc)`
                // here would append a stray `(None,None)` exit on top
                // of the explicit bool branches, producing the 3-exit
                // `[Bool(false), Bool(true), (None,None)]` shape that
                // trips `flatten.py:275-296 insert_switch_exits`.
                //
                // Skipping the force when `current_block.exits` is
                // non-empty falls into the `else if let Some(target) =
                // joinpoints.get(&py_pc)` arm, which correctly switches
                // to the joinpoint candidate POP_JUMP_IF_FALSE's
                // mergeblock just created at `fallthrough_pc`.
                // `rpython/flowspace/flowcontext.py:130-156
                // guessexception` closes the canraise block at the
                // op that just attached the exception edge, so the
                // next PC's bytecode lands in a fresh egg.  Pyre's
                // `emit_catch_exception!` only attaches the exception
                // edge (sets `exitswitch = LastException`) and leaves
                // the canraise block "half-closed" — the walker keeps
                // emitting subsequent ops into the same block, and a
                // later POP_JUMP_IF overwrites `exitswitch` with its
                // bool var, leaving the orphan exception edge in
                // `exits[0]` and tripping `flatten.py:275-296
                // insert_switch_exits` ("switch link requires
                // Signed/Bool llexitcase").  Force a block boundary
                // here so the next PC's ops emit into a fresh egg.
                let canraise_pending = matches!(
                    current_block.block().borrow().exitswitch,
                    Some(super::flow::ExitSwitch::Value(super::flow::FlowValue::Constant(
                        ref c,
                    ))) if matches!(c.value, super::flow::ConstantValue::Atom(super::flow::Atom::LastException))
                );
                let force_branch_boundary = needs_fallthrough
                    && current_block
                        .framestate()
                        .map_or(true, |fs| fs.next_offset != py_pc)
                    && (canraise_pending
                        || (branch_target_pcs.contains(&py_pc)
                            && current_block.block().borrow().exits.is_empty()));
                let new_block = if needs_fallthrough
                    && (current_state.next_offset != py_pc || force_branch_boundary)
                {
                    let merged = mergeblock(
                        code,
                        &mut graph,
                        &mut joinpoints,
                        &current_block,
                        &current_state,
                        py_pc,
                        &mut pendingblocks,
                        &mut all_walker_blocks,
                    );
                    if canraise_pending {
                        restore_canraise_exit_order(&current_block.block());
                    }
                    merged
                } else if let Some(target) = joinpoints
                    .get(&py_pc)
                    .and_then(|blocks| blocks.iter().find(|b| !b.dead()))
                    .cloned()
                {
                    // Branch arrival / catch landing / earlier sequential
                    // walker step at this PC already registered a live
                    // block.  RPython equivalent: the `set_branch` /
                    // `mergeblock` that targeted `py_pc` populated the
                    // joinpoint candidate list (`flowcontext.py:426
                    // candidates = self.joinpoints.setdefault(...)`).
                    //
                    // Per-block contiguous walker: when the joinpoint
                    // target differs from `current_block`, queue the
                    // target to `pendingblocks` (mergeblock-path
                    // queuing is already done by mergeblock itself;
                    // joinpoint match doesn't push automatically).
                    //
                    // Gate the re-push on `target.exits.is_empty()`
                    // matching upstream `flowcontext.py:407-475
                    // record_block`: a block is added to pendingblocks
                    // **exactly once** (initial seed + supersede), and
                    // once popped + walked the walker iterates per-PC
                    // until a terminator closes the block.  Pyre's
                    // emit_mark_label_pc! joinpoint-arrival arm can
                    // hit the same block again via a later sequential
                    // PC iteration; re-pushing a block that's already
                    // been fully walked (exits non-empty) would cause
                    // the outer while-let to pop and re-walk it,
                    // emitting RETURN_VALUE / ref_return etc. a second
                    // time and appending a duplicate return link.
                    // Empty exits == not yet processed → push.
                    if target != current_block
                        && !pendingblocks.iter().any(|b| b == &target)
                        && target.block().borrow().exits.is_empty()
                    {
                        pendingblocks.push_front(target.clone());
                    }
                    target
                } else if !current_block.dead() {
                    // Natural fall-through within current_block — Phase
                    // A.1+A.2 cover branch targets via boundary force.
                    // When `current_block` was closed by a previous
                    // terminator emit (`emit_goto!`, `emit_ref_return!`,
                    // `emit_raise!`, `emit_reraise!`, POP_JUMP_IF) and
                    // no joinpoint exists at `py_pc`, this arm still
                    // returns `current_block` so the per-PC `-live-`
                    // marker is emitted for pyre's per-PC dispatch
                    // invariant (`walker_pc_live_marker_pos`).  The
                    // `block_closed_by_terminator` gate after
                    // `emit_live_placeholder!()` then suppresses op
                    // dispatch into the closed block — mirroring
                    // `rpython/flowspace/flowcontext.py:407-475` which
                    // raises `StopFlowing` from `closeblock` and pops
                    // the next block.
                    current_block.clone()
                } else {
                    // `current_block` already closed and no joinpoint
                    // candidate exists — RPython has no equivalent
                    // because its per-block walker (`flowcontext.py:
                    // 407-475`) cannot re-enter PC iteration with a
                    // dead current block: every walker pop installs a
                    // fresh live SpamBlock from `pendingblocks`.  Pyre's
                    // PC-sequential walker drove the prior synthesise-
                    // fresh-block adaptation here, but with the W-1 fix
                    // every sequential PC keeps `current_block` alive
                    // and every branch arrival registers a joinpoint
                    // candidate, so this arm should be unreachable.
                    // Fail-loud per RPython invariant.
                    panic!(
                        "emit_mark_label_pc!(py_pc={}): no live current_block \
                         and no joinpoint candidate — invariant violation",
                        py_pc,
                    );
                };
                // Yield-on-switch: when `new_block` differs from
                // `current_block`, set the `block_switch_pending`
                // flag and SKIP the inline switch (the new block has
                // been queued to `pendingblocks` above; the outer
                // walker loop will pop it and process its emit
                // sequence contiguously).  The inner for-loop body
                // checks `block_switch_pending` after each per-PC
                // emit and breaks, yielding control.
                if new_block != current_block {
                    // Yield without pushing Label — the new block's
                    // outer iter at start_pc=py_pc will emit its
                    // own Label via its own `emit_mark_label_pc!(
                    // py_pc)` call (which will see no-switch since
                    // joinpoints[py_pc] now points at new_block ==
                    // current_block at that point).
                    //
                    // Emit `goto Label("pcN")` + `Unreachable` into
                    // the previous block's per-block accumulator
                    // (mirrors `flatten.py:177-258 insert_exits`)
                    // before yielding, so the per-block stream
                    // routes via explicit goto rather than implicit
                    // fallthrough to whichever block lands next in
                    // walker-pop order.
                    if needs_fallthrough {
                        // mergeblock support: emit goto against the new
                        // block's identity TLabel, matching upstream
                        // `flatten.py:161` `TLabel(link.target)` where
                        // `link.target` is a Block, not a PC.
                        let goto_insn = Insn::op(
                            "goto",
                            vec![Operand::TLabel(super::flatten::block_tlabel(
                                &new_block.block(),
                            ))],
                        );
                        push_walker_emit(&current_block, goto_insn);
                        push_walker_emit(&current_block, Insn::Unreachable);
                    }
                    block_switch_pending = true;
                } else {
                    // No switch — same block continues at py_pc.
                    // Pyre's runtime `pc_map` is sourced from the
                    // `walker_pc_live_marker_pos` side-table at drain
                    // time (see `walker_tracked_pc_live_indices`), so
                    // we emit only one `Label(block)` per FunctionGraph
                    // block matching `flatten.py:116`.
                    needs_fallthrough = true;
                }
            }};
        }
        macro_rules! emit_mark_label_catch_landing {
            ($landing_label:expr) => {{
                let landing_label = $landing_label;
                // switch the shadow graph's
                // `current_block` into the pre-allocated catch-landing
                // block. Matches `flatten.py:180` `Label(block)` being the
                // block-entry marker in RPython. Catch landings are
                // reached via `catch_exception` edges rather than
                // fallthrough, so no implicit Link is inserted here —
                // reset `needs_fallthrough` for the landing block's
                // own emission sequence.
                current_block = catch_sites
                    .iter()
                    .find(|s| s.landing_label == landing_label)
                    .expect("catch_sites entry for landing_label")
                    .landing
                    .clone();
                if let Some(state) = current_block.framestate() {
                    current_state = state;
                }
                needs_fallthrough = true;
                // Emission-order tracking: push the catch landing
                // block to `all_walker_blocks` AT FIRST EMIT (not at
                // creation) so the drain order reflects walker
                // emission order — catch landings emit after the
                // main walker loop, so creation-order tracking would
                // misalign with ssarepr.insns ordering.  Guard against
                // double-push: a single catch landing may be entered
                // multiple times if multiple catch sites share a
                // landing label (unusual but possible per the
                // `catch_sites` dedup).
                if !all_walker_blocks.iter().any(|b| b == &current_block) {
                    all_walker_blocks.push(current_block.clone());
                }
                // Per-block accumulator entry Label — drain swap
                // (line ~7319) reproduces it into ssarepr.insns in
                // walker-block-creation order.
                push_walker_emit(
                    &current_block,
                    Insn::Label(super::flatten::Label::new(format!(
                        "catch_landing_{}",
                        landing_label
                    ))),
                );
            }};
        }

        // RPython `-live-` placement is *not* per-PC: `jtransform.py`
        // emits `SpaceOperation('-live-', [], None)` graph-side only at
        // raising / virtualizable / inline-call decision points (e.g.
        // `jtransform.py:469-471 handle_residual_call`,
        // `jtransform.py:481 handle_regular_call`,
        // `jtransform.py:845` before `getfield_vable_<kind>`); flatten
        // serialises those graph ops via `serialize_op` and additionally
        // emits SSA-only `-live-` at branch / raise / switch boundaries
        // (`flatten.py:142, 259, 285, 303`) — those four SSA-only sites
        // are mirrored line-for-line by pyre's renderer-side
        // `flatten_graph` (`super::flatten::FlattenGraph::insert_exits` /
        // `make_return` at `flatten.rs:1000, 1139, 1208, 1228`).
        //
        // pyre's walker, by contrast, runs 1:1 against the Python
        // bytecode and emits `-live-` at every PC entry to seed the
        // post-regalloc `all_liveness` table (`assembler.py:146-158`).
        // That per-PC emission is a walker-shape adaptation, not an
        // orthodox graph emission, and intentionally has no
        // `record_graph_op` companion — recording it graph-side at
        // every PC would create a `-live-` cluster the upstream graph
        // never holds.
        macro_rules! emit_live_placeholder {
            () => {{
                // RPython force-alive mechanism (`liveness.py:11-12`):
                //
                //   You can also force extra variables to be alive by putting
                //   them as args of the '-live-' operation in the first place.
                //
                // Use it to keep the portal red args (`pypy/module/pypyjit/
                // interp_jit.py:67 reds = ['frame', 'ec']`) alive across every
                // PC.
                let mut force_alive: Vec<Operand> = Vec::new();
                if portal_frame_reg != u16::MAX {
                    force_alive.push(Operand::Register(Register::new(
                        Kind::Ref,
                        portal_frame_reg,
                    )));
                }
                if portal_ec_reg != u16::MAX {
                    force_alive.push(Operand::Register(Register::new(Kind::Ref, portal_ec_reg)));
                }
                push_walker_emit(&current_block, Insn::live(force_alive));
            }};
        }

        // flatten.py:240-260 boolean exitswitch emission. When the bool is a
        // plain variable (truth_fn result), flatten emits `goto_if_not <v> L`
        // (alias of bhimpl_goto_if_not_int_is_true per blackhole.py:913).
        // Both POP_JUMP_IF_FALSE and POP_JUMP_IF_TRUE use that generic Bool
        // exitswitch form; the polarity difference is encoded by which edge is
        // arranged as `linkfalse`, not by changing the opcode.
        macro_rules! emit_goto_if_not {
            ($cond:expr, $py_pc:expr) => {{
                let cond = $cond;
                let py_pc = $py_pc;
                // `flatten.py:259` — emit bare `-live-` (empty
                // force_alive) IMMEDIATELY before the guard.  Canonical
                // (`flatten.rs:1888`) does the same; the bare marker
                // seeds the guard's liveness snapshot for blackhole
                // reconstruction.  The per-PC `-live-` with portal red
                // args lives at the START of each PC's emission, not
                // here.
                push_walker_emit(&current_block, Insn::live(Vec::new()));
                // mergeblock support: mergeblock first to learn target block
                // identity, then emit the guard with block-identity
                // `TLabel`.  `flatten.py:240-267` linkfalse mergeblock.
                let target_block = mergeblock(
                    code,
                    &mut graph,
                    &mut joinpoints,
                    &current_block,
                    &{
                        let mut branch_state = current_state.clone();
                        branch_state.next_offset = py_pc;
                        branch_state.blocklist = frame_blocks_for_offset(code, py_pc);
                        branch_state
                    },
                    py_pc,
                    &mut pendingblocks,
                    &mut all_walker_blocks,
                );
                let insn = Insn::op(
                    "goto_if_not",
                    vec![
                        Operand::reg(Kind::Int, cond),
                        Operand::TLabel(super::flatten::block_tlabel(&target_block.block())),
                    ],
                );
                push_walker_emit(&current_block, insn);
            }};
        }
        macro_rules! emit_goto_if_not_int_is_zero {
            ($cond:expr, $py_pc:expr) => {{
                let cond = $cond;
                let py_pc = $py_pc;
                // `flatten.py:259` bare `-live-` precedes the guard.
                push_walker_emit(&current_block, Insn::live(Vec::new()));
                // mergeblock support: mergeblock first to learn target block,
                // then emit guard with block-identity `TLabel`.
                // `flatten.py:247` `goto_if_not_int_is_zero` shape is
                // identical to `goto_if_not` save for the opname.
                let target_block = mergeblock(
                    code,
                    &mut graph,
                    &mut joinpoints,
                    &current_block,
                    &{
                        let mut branch_state = current_state.clone();
                        branch_state.next_offset = py_pc;
                        branch_state.blocklist = frame_blocks_for_offset(code, py_pc);
                        branch_state
                    },
                    py_pc,
                    &mut pendingblocks,
                    &mut all_walker_blocks,
                );
                let insn = Insn::op(
                    "goto_if_not_int_is_zero",
                    vec![
                        Operand::reg(Kind::Int, cond),
                        Operand::TLabel(super::flatten::block_tlabel(&target_block.block())),
                    ],
                );
                push_walker_emit(&current_block, insn);
            }};
        }

        // RPython-orthodox vable scalar field shapes
        // (`getfield_vable_<kind>` / `setfield_vable_<kind>`). Upstream
        // `jtransform.py:846-847` emits `getfield_vable_<kind>` with
        // **2 args + result**: `[v_inst, descr]` → `op.result`;
        // `jtransform.py:927-928` emits `setfield_vable_<kind>` with
        // **3 args**: `[v_inst, v_value, descr]`. Pyre matches that shape
        // end-to-end across all three layers:
        //
        // - **GRAPH layer** (`record_graph_op("setfield_vable_i", …)`):
        //   `vable_setfield_int_graph_args(v_inst, v_value, idx)` carries
        //   the portal frame Variable (`portal_graph_inputvars(code).0`,
        //   per `jtransform.py:840`) as `v_inst`, threaded by the call
        //   sites via `frame_var.into()` (Stage 3 Issue 2.3 —
        //   graph-shadow `v_inst/v_base` parity landed).
        // - **SSARepr layer** (`emit_vable_setfield_int!` /
        //   `emit_vable_getfield_ref!`): setfield = `[reg(Ref, frame),
        //   reg(Int, src), descr_vable_static_field(idx)]`; getfield =
        //   `[reg(Ref, frame), descr_vable_static_field(idx)]` with a
        //   Ref result.  `flatten_arg` lowers graph-side
        //   `SpaceOperationArg::Descr` to the matching `Operand::Descr`
        //   via `flatten_descr_by_ptr` (Arc::ptr_eq against the
        //   singleton) — same shape end-to-end.
        // - **Assembler dispatch** lowers that exact `[r, d]` / `[r, X, d]`
        //   shape to the canonical `JitCodeBuilder::*_with_base` emitters:
        //   one-byte vable/value registers plus a two-byte descriptor-pool
        //   index, matching `assembler.py:80-138`.
        //
        // Graph-side shadow for `getfield_vable_r` intentionally
        // absent: jtransform.py:919-922 `do_fixed_list_getitem`
        // lowers `getfield_vable_r` to a fresh Variable result that
        // subsequent ops consume as an input. Pyre does not yet
        // thread that result through downstream graph ops, so emitting
        // an unused
        // Variable here would introduce a dangling shadow with no
        // upstream backing. The graph mirror returns once a real
        // consumer exists.
        //
        // The remaining vable scalar variants (`getfield_vable_i/f`,
        // `setfield_vable_r/f`) have assembler dispatch arms but no
        // production emit site today; those arms already require the same
        // canonical `[v_inst, ... descr]` operand shape.
        macro_rules! emit_vable_getfield_ref {
            ($vable_reg:expr, $dst:expr, $field_idx:expr) => {{
                let vable_reg: u16 = $vable_reg;
                let dst = $dst;
                let field_idx: u16 = $field_idx;
                // `jtransform.py:846-847` getfield: `[v_inst, descr]` → result.
                // `args[0]` is the vable register holding the live frame
                // pointer — `portal_frame_reg` is filled by
                // `BlackholeInterpreter::fill_portal_registers` at portal
                // entry and encoded by the assembler as the canonical
                // leading `r` operand.
                emit_vable_getfield_ref_walker_only(&current_block, vable_reg, dst, field_idx);
                // Graph dual-write threads `frame_var.into()`.  As of
                // Epic #245 Slice 1 `frame_var` is a startblock inputarg
                // for both portal and non-portal graphs
                // (`graph_entry_inputargs(code, frame_inputs)`), so the
                // Variable always has a producer.  The dual-write stays
                // portal-gated here because the non-portal vable readers
                // still source the frame from the shared
                // `portal_frame_reg` slot rather than the per-callee
                // `frame_var`; rewiring them is a later slice.  Returns
                // `Option<Variable>` so callsites that need the graph
                // identity for downstream dual-writes can thread the
                // same Variable; non-portal callees skip the graph emit
                // and return `None`.
                if is_portal {
                    let result = emit_graph_op_with_result(
                        &mut graph,
                        &current_block.block(),
                        "getfield_vable_r",
                        vable_getfield_ref_graph_args(frame_var.into(), field_idx),
                        Kind::Ref,
                        -1,
                    );
                    pin!(Some(result), dst);
                    Some(result)
                } else {
                    None
                }
            }};
        }
        macro_rules! emit_vable_setfield_int {
            ($vable_reg:expr, $field_idx:expr, $src:expr) => {{
                let vable_reg: u16 = $vable_reg;
                let field_idx: u16 = $field_idx;
                let src = $src;
                // `jtransform.py:927-928` setfield: `[v_inst, v_value, descr]`.
                // `args[0]` is the vable register — see
                // `emit_vable_getfield_ref!` for rationale.
                let insn = Insn::op(
                    "setfield_vable_i",
                    vec![
                        Operand::reg(Kind::Ref, vable_reg),
                        Operand::reg(Kind::Int, src),
                        Operand::descr_vable_static_field(field_idx),
                    ],
                );
                push_walker_emit(&current_block, insn);
            }};
        }
        macro_rules! emit_vable_setfield_int_const {
            ($vable_reg:expr, $field_idx:expr, $value:expr) => {{
                let vable_reg: u16 = $vable_reg;
                let field_idx: u16 = $field_idx;
                let value: i64 = $value;
                // ConstInt-source setfield_vable_i: assembler dispatch
                // (assembler.rs:907-911) routes `Operand::ConstInt` to
                // `vable_setfield_int_const_value_with_base`.  Matches
                // upstream's flatten output for jtransform.py:927-928
                // when the value is a folded ConstInt.
                let insn = Insn::op(
                    "setfield_vable_i",
                    vec![
                        Operand::reg(Kind::Ref, vable_reg),
                        Operand::ConstInt(value),
                        Operand::descr_vable_static_field(field_idx),
                    ],
                );
                push_walker_emit(&current_block, insn);
            }};
        }
        // RPython-orthodox vable arrayitem shapes for
        // `setarrayitem_vable_r` and
        // `getarrayitem_vable_r`.  Upstream
        // `jtransform.py:1880-1885 do_fixed_list_getitem` emits
        // `getarrayitem_vable_X` with **4 args**: `[v_base, v_index,
        // arrayfielddescr, arraydescr]`; `jtransform.py:1898-1906
        // do_fixed_list_setitem` emits `setarrayitem_vable_X` with
        // **5 args**: `[v_base, v_index, v_value, arrayfielddescr,
        // arraydescr]`.  Pyre matches that shape end-to-end across all
        // three layers:
        //
        // - **GRAPH layer** (`record_graph_op("setarrayitem_vable_r",
        //   …)`): `vable_setarrayitem_ref_graph_args(v_base, v_idx,
        //   v_value)` carries the portal frame Variable
        //   (`portal_graph_inputvars(code).0`, per `jtransform.py:840`)
        //   as `v_base`, threaded by the call sites via
        //   `frame_var.into()` (Stage 3 Issue 2.3 — graph-shadow
        //   `v_base/v_inst` parity landed).  When the value being stored
        //   is a true `ConstPtr` lifted to the bytecode const-pool,
        //   `v_value` is recorded as a `Constant::none()` placeholder
        //   (the live SSA register is patched at bytecode-finish time
        //   via `vable_setarrayitem_ref_const_value_with_base`); the
        //   bytecode shape stays identical.  The two trailing descrs
        //   are singleton `Arc<dyn Descr>`s in `majit_ir::descr`
        //   mirroring `rpython/jit/metainterp/virtualizable.py:73,58`.
        // - **SSARepr layer** (`emit_vable_setarrayitem_ref!` /
        //   `emit_vable_setarrayitem_ref_const!`):
        //   `[reg(Ref, frame), reg(Int, idx), reg(Ref, src) |
        //   ConstRef(value), descr_vable_array_field(idx),
        //   descr_vable_array(idx)]`.  `flatten_arg` lowers
        //   graph-side `SpaceOperationArg::Descr` to the matching
        //   `Operand::Descr` via `flatten_descr_by_ptr` (Arc::ptr_eq
        //   against the singletons) — same shape end-to-end.
        // - **Assembler dispatch** extracts and validates the two trailing
        //   descrs, then emits canonical `[r, i, d, d, >r]` /
        //   `[r, i, r, d, d]` bytecode through `JitCodeBuilder::*_with_base`.
        //
        // The remaining vable op family variants
        // (`getarrayitem_vable_i/f`, `setarrayitem_vable_i/f`,
        // `arraylen_vable`) have assembler dispatch arms but no
        // production emit site today (pyre's `PyFrame
        // .locals_cells_stack_w` carries Ref items only and its
        // length is constant).  The arms already require the canonical
        // `[v_base, ... arrayfielddescr, arraydescr]` operand shape.
        macro_rules! emit_vable_getarrayitem_ref {
            ($vable_reg:expr, $dst:expr, $field_idx:expr, $index:expr) => {{
                let vable_reg: u16 = $vable_reg;
                let dst = $dst;
                let field_idx: u16 = $field_idx;
                let index = $index;
                // `jtransform.py:1882-1885 do_fixed_list_getitem` (vable
                // branch): `[v_base, v_index, arrayfielddescr,
                // arraydescr]` with a Ref result.  See
                // `emit_vable_setarrayitem_ref!` for v_base register
                // rationale and the descr-pair parity citations.
                let insn = Insn::op_with_result(
                    "getarrayitem_vable_r",
                    vec![
                        Operand::reg(Kind::Ref, vable_reg),
                        Operand::reg(Kind::Int, index),
                        Operand::descr_vable_array_field(field_idx),
                        Operand::descr_vable_array(field_idx),
                    ],
                    Register::new(Kind::Ref, dst),
                );
                push_walker_emit(&current_block, insn);
            }};
        }
        macro_rules! emit_vable_setarrayitem_ref {
            ($vable_reg:expr, $field_idx:expr, $index:expr, $src:expr) => {{
                let vable_reg: u16 = $vable_reg;
                let field_idx: u16 = $field_idx;
                let index = $index;
                let src = $src;
                // `jtransform.py:1898-1906 do_fixed_list_setitem` (vable
                // branch): `[v_base, v_index, v_value, arrayfielddescr,
                // arraydescr]`. `args[0]` is the vable register holding
                // the live frame pointer (`portal_frame_reg` filled by
                // `fill_portal_registers`).  Trailing two descrs are
                // `array_field_descrs[i]` / `array_descrs[i]` per
                // `rpython/jit/metainterp/virtualizable.py:73,58`.
                let insn = Insn::op(
                    "setarrayitem_vable_r",
                    vec![
                        Operand::reg(Kind::Ref, vable_reg),
                        Operand::reg(Kind::Int, index),
                        Operand::reg(Kind::Ref, src),
                        Operand::descr_vable_array_field(field_idx),
                        Operand::descr_vable_array(field_idx),
                    ],
                );
                push_walker_emit(&current_block, insn);
            }};
        }

        // `setarrayitem_vable_r(vable, idx, ConstPtr(value))` — the
        // ConstPtr-source variant produced by jtransform.py:1898 when
        // the value operand is a Const. Carries `Operand::ConstRef`
        // through to the assembler dispatch, which routes it to
        // `JitCodeBuilder::vable_setarrayitem_ref_const_value_with_base`.
        // No separate bytecode: the canonical `setarrayitem_vable_r/rirdd`
        // form can address const sources through the unified ref register
        // space after `const_patches_u8` resolution.
        macro_rules! emit_vable_setarrayitem_ref_const {
            ($vable_reg:expr, $field_idx:expr, $index:expr, $value:expr) => {{
                let vable_reg: u16 = $vable_reg;
                let field_idx: u16 = $field_idx;
                let index = $index;
                let value: i64 = $value;
                // ConstPtr-source variant of the 5-arg SSA shape (see
                // `emit_vable_setarrayitem_ref!` for the parity
                // citation). The third operand carries
                // `Operand::ConstRef(value)` instead of a register.
                let insn = Insn::op(
                    "setarrayitem_vable_r",
                    vec![
                        Operand::reg(Kind::Ref, vable_reg),
                        Operand::reg(Kind::Int, index),
                        Operand::ConstRef(value),
                        Operand::descr_vable_array_field(field_idx),
                        Operand::descr_vable_array(field_idx),
                    ],
                );
                push_walker_emit(&current_block, insn);
            }};
        }

        // `setarrayitem_vable_r(vable, ConstInt(idx), value_reg)` —
        // ConstInt-INDEX variant matching upstream's `jtransform.py:1898`
        // shape when the index is folded to a Const.  Assembler dispatch
        // routes to `vable_setarrayitem_ref_const_idx_with_base`.
        macro_rules! emit_vable_setarrayitem_ref_const_idx {
            ($vable_reg:expr, $field_idx:expr, $index_value:expr, $src:expr) => {{
                let vable_reg: u16 = $vable_reg;
                let field_idx: u16 = $field_idx;
                let index_value: i64 = $index_value;
                let src = $src;
                let insn = Insn::op(
                    "setarrayitem_vable_r",
                    vec![
                        Operand::reg(Kind::Ref, vable_reg),
                        Operand::ConstInt(index_value),
                        Operand::reg(Kind::Ref, src),
                        Operand::descr_vable_array_field(field_idx),
                        Operand::descr_vable_array(field_idx),
                    ],
                );
                push_walker_emit(&current_block, insn);
            }};
        }

        // `setarrayitem_vable_r(vable, ConstInt(idx), ConstRef(value))`
        // — both index and value as constants.  Assembler routes to
        // `vable_setarrayitem_ref_const_idx_const_value_with_base`.
        macro_rules! emit_vable_setarrayitem_ref_const_idx_const_value {
            ($vable_reg:expr, $field_idx:expr, $index_value:expr, $src_value:expr) => {{
                let vable_reg: u16 = $vable_reg;
                let field_idx: u16 = $field_idx;
                let index_value: i64 = $index_value;
                let src_value: i64 = $src_value;
                let insn = Insn::op(
                    "setarrayitem_vable_r",
                    vec![
                        Operand::reg(Kind::Ref, vable_reg),
                        Operand::ConstInt(index_value),
                        Operand::ConstRef(src_value),
                        Operand::descr_vable_array_field(field_idx),
                        Operand::descr_vable_array(field_idx),
                    ],
                );
                push_walker_emit(&current_block, insn);
            }};
        }

        // `getarrayitem_vable_r(vable, ConstInt(idx)) → dst` — ConstInt-
        // INDEX variant matching upstream's `jtransform.py:1882-1885`
        // shape when the index is folded.  Assembler dispatch routes to
        // `vable_getarrayitem_ref_const_idx_with_base`.
        macro_rules! emit_vable_getarrayitem_ref_const_idx {
            ($vable_reg:expr, $dst:expr, $field_idx:expr, $index_value:expr) => {{
                let vable_reg: u16 = $vable_reg;
                let dst = $dst;
                let field_idx: u16 = $field_idx;
                let index_value: i64 = $index_value;
                let insn = Insn::op_with_result(
                    "getarrayitem_vable_r",
                    vec![
                        Operand::reg(Kind::Ref, vable_reg),
                        Operand::ConstInt(index_value),
                        Operand::descr_vable_array_field(field_idx),
                        Operand::descr_vable_array(field_idx),
                    ],
                    Register::new(Kind::Ref, dst),
                );
                push_walker_emit(&current_block, insn);
            }};
        }

        // pyframe.py:378-381 `pushvalue` lowers to
        // `setarrayitem_vable_r(locals_cells_stack_w, depth, w_object)`
        // + `setfield_vable_i(valuestackdepth, depth + 1)` via
        // jtransform.py:1898 `do_fixed_list_setitem` +
        // jtransform.py:920-928. RPython's optimizer folds the per-push
        // `setarrayitem_vable_r` via OptVirtualize so that the compiled
        // trace pays only the final force-vable cost; pyre's
        // OptVirtualize does not yet fold these at the same grain, so
        // the emission is load-bearing for shadow parity with
        // `list_of_boxes_virtualizable` + BH `virtualizable_boxes`
        // reconstruction and the per-push cost is recovered only as the
        // optimizer port progresses.
        macro_rules! emit_pushvalue_ref {
            ($depth:ident, $src:expr, $src_value:expr, $py_pc:expr) => {{
                let src_reg = $src;
                let src_value: super::flow::FlowValue = $src_value;
                let pushvalue_ref_py_pc: i64 = ($py_pc) as i64;
                emit_ref_copy(&current_block, stack_base + $depth, src_reg);
                if is_portal {
                    let depth_value = (stack_base_absolute + $depth as usize) as i64;
                    // `pyframe.py:389 pushvalue` lowers to
                    // `setarrayitem_vable_r(locals_cells_stack_w,
                    // depth, w_object)` via `jtransform.py:1898
                    // do_fixed_list_setitem` (vable branch).  The
                    // index operand goes directly as a Constant —
                    // upstream's vable branch threads the depth as a
                    // ConstInt arg to setarrayitem_vable_r, no
                    // intermediate `int_copy(ConstInt(depth)) → Var`
                    // SpaceOp.
                    let v_idx: super::flow::FlowValue =
                        super::flow::Constant::signed(depth_value).into();
                    record_graph_op(
                        &current_block.block(),
                        "setarrayitem_vable_r",
                        vable_setarrayitem_ref_graph_args(
                            frame_var.into(),
                            v_idx.into(),
                            src_value.into(),
                        ),
                        None,
                        pushvalue_ref_py_pc,
                    );
                    emit_vable_setarrayitem_ref_const_idx!(
                        portal_frame_reg,
                        0_u16,
                        depth_value,
                        src_reg
                    );
                }
                $depth += 1;
                emit_vsd!($depth, pushvalue_ref_py_pc);
            }};
        }

        // null_ref_reg retirement — null_ref_reg → ConstRef(PY_NULL) migration.
        // pyframe.py:389 `pushvalue(w_object)` lowers, when w_object is a
        // compile-time `ConstPtr.NULL`, to `setarrayitem_vable_r(
        // locals_cells_stack_w, depth, ConstPtr(NULL))` via
        // jtransform.py:1898. Pyre's bytecode does not yet expose a
        // const-source variant of `setarrayitem_vable_r`, so we lazily
        // materialize the constant into the caller-supplied scratch ref
        // register and emit the regular reg-source path. The graph
        // shadow's third operand is the canonical null ref constant —
        // `Constant::none()` (`ConstantValue::None` + `Kind::Ref`),
        // matching pyframe.py:411 (`None`) and assembler.py:109's null
        // ref handling. flatten.rs:1163 lowers it to `ConstRef(0)`,
        // which is the same sentinel `PY_NULL` (a null pointer) that
        // the SSA emit writes via `emit_vable_setarrayitem_ref_const`.
        // All current callers pass `PY_NULL`; the parameter is retained
        // for surface symmetry with `emit_pushvalue_ref!`.
        macro_rules! emit_pushvalue_ref_const {
            ($depth:ident, $value:expr, $py_pc:expr) => {{
                let value: i64 = $value;
                let pushvalue_const_py_pc: i64 = ($py_pc) as i64;
                debug_assert_eq!(
                    value,
                    pyre_object::PY_NULL as i64,
                    "emit_pushvalue_ref_const: only PY_NULL is supported today; \
                     graph shadow uses Constant::none() per assembler.py:109",
                );
                if is_portal {
                    let depth_value = (stack_base_absolute + $depth as usize) as i64;
                    let v_idx: super::flow::FlowValue =
                        super::flow::Constant::signed(depth_value).into();
                    record_graph_op(
                        &current_block.block(),
                        "setarrayitem_vable_r",
                        vable_setarrayitem_ref_graph_args(
                            frame_var.into(),
                            v_idx.into(),
                            super::flow::Constant::none().into(),
                        ),
                        None,
                        pushvalue_const_py_pc,
                    );
                    emit_vable_setarrayitem_ref_const_idx_const_value!(
                        portal_frame_reg,
                        0_u16,
                        depth_value,
                        value
                    );
                } else {
                    // Non-portal frames have no vable mirror; the runtime
                    // expects the pushed PY_NULL to be visible in the
                    // stack slot register for any downstream consumer.
                    // `flatten.py:333-334` parity: ref_copy with ConstRef
                    // source.  Walker MUST NOT record a graph-side
                    // `ref_copy` op (matching `emit_ref_copy`).
                    let const_copy_insn = Insn::op_with_result(
                        "ref_copy",
                        vec![Operand::ConstRef(value)],
                        Register::new(Kind::Ref, stack_base + $depth),
                    );
                    push_walker_emit(&current_block, const_copy_insn);
                }
                $depth += 1;
                emit_vsd!($depth, pushvalue_const_py_pc);
            }};
        }

        // pyframe.py:411-417 `popvalue_maybe_none` lowers to
        // `setarrayitem_vable_r(locals_cells_stack_w, depth, ConstPtr.NULL)`
        // + `setfield_vable_i(valuestackdepth, depth)` via
        // jtransform.py:1898 / :927. The SSA op carries `ConstRef(0)`
        // as the value operand — at assembler time the dispatch routes
        // it to `vable_setarrayitem_ref_const_value`, which reuses the
        // existing `BC_SETARRAYITEM_VABLE_R` bytecode with its src u16
        // patched to the constants suffix of the unified register
        // space. Single bytecode op per pop, matching upstream's
        // `iric` argcode lowering. The popped SSA register stays
        // available for downstream uses. The graph shadow's third
        // operand is `Constant::none()` (`ConstantValue::None` +
        // `Kind::Ref`), the canonical null ref representation upstream
        // uses for stack-slot clears (pyframe.py:411 `None`,
        // assembler.py:109 null ref handling). flatten.rs:1163 lowers
        // it to `ConstRef(0)`.
        macro_rules! emit_popvalue_ref {
            ($depth:ident, $py_pc:expr) => {{
                // Do not change this to a plain `$depth -= 1` until the
                // portal stack-depth model is fully aligned with PyPy's
                // assert-on-underflow behavior.  The direct parity change
                // makes `synth/comprehensions` crash on both dynasm and
                // cranelift (`python3 pyre/check.py --synthetic-only
                // --synthetic-pattern comprehensions.py`).
                let popvalue_ref_py_pc: i64 = ($py_pc) as i64;
                $depth = $depth.saturating_sub(1);
                let popped_reg = stack_base + $depth;
                if is_portal {
                    let depth_value = (stack_base_absolute + $depth as usize) as i64;
                    let v_idx: super::flow::FlowValue =
                        super::flow::Constant::signed(depth_value).into();
                    record_graph_op(
                        &current_block.block(),
                        "setarrayitem_vable_r",
                        vable_setarrayitem_ref_graph_args(
                            frame_var.into(),
                            v_idx.into(),
                            super::flow::Constant::none().into(),
                        ),
                        None,
                        popvalue_ref_py_pc,
                    );
                    emit_vable_setarrayitem_ref_const_idx_const_value!(
                        portal_frame_reg,
                        0_u16,
                        depth_value,
                        pyre_object::PY_NULL as i64
                    );
                }
                emit_vsd!($depth, popvalue_ref_py_pc);
                popped_reg
            }};
        }

        // pyopcode.py:500-507 LOAD_FAST + pyframe.py:378-381 pushvalue.
        // Portal case lowers the local read to `getarrayitem_vable_r`
        // (jtransform.py:1877 `do_fixed_list_getitem`). Both the load
        // and the subsequent pushvalue's `setarrayitem_vable_r` mirror
        // (jtransform.py:1898) are emitted here so the shadow
        // `locals_cells_stack_w` slot mirrors the value loaded into the
        // stack-side SSA register.
        macro_rules! emit_load_fast_ref {
            ($depth:ident, $reg:expr, $py_pc:expr) => {{
                let reg = $reg;
                let load_fast_py_pc: i64 = ($py_pc) as i64;
                if is_portal {
                    let local_slot = local_to_vable_slot(reg as usize) as i64;
                    let stack_slot = (stack_base_absolute + $depth as usize) as i64;
                    emit_vable_getarrayitem_ref_const_idx!(
                        portal_frame_reg,
                        stack_base + $depth,
                        0_u16,
                        local_slot
                    );
                    // Graph-side dual-write of BOTH halves of the
                    // LOAD_FAST lowering:
                    //   - local read: jtransform.py:1877
                    //     `do_fixed_list_getitem` lowers
                    //     `locals_cells_stack_w[local_slot]` to
                    //     `getarrayitem_vable_r(_, ConstInt(local_slot))`,
                    //     producing a Ref result that is the loaded
                    //     local value.
                    //   - stack write: jtransform.py:1898
                    //     `do_fixed_list_setitem` lowers the subsequent
                    //     `pushvalue(loaded)` to
                    //     `setarrayitem_vable_r(_, ConstInt(stack_slot),
                    //     loaded)`.
                    // The result of the read feeds the source of the
                    // write — a single fresh Ref Variable threads
                    // through both ops.
                    //
                    // `current_state.locals_w[reg]` is left UNCHANGED:
                    // pyopcode.py:500-507 LOAD_FAST is a stack push,
                    // not a local-binding mutation.  The pre-existing
                    // Variable in `locals_w[reg]` continues to identify
                    // the local slot for subsequent reads (matching
                    // RPython, where `getarrayitem_vable_r` reads do
                    // not feed back into `vable_array_vars`).
                    let v_local_idx: super::flow::FlowValue =
                        super::flow::Constant::signed(local_slot).into();
                    let v_loaded = emit_graph_op_with_result(
                        &mut graph,
                        &current_block.block(),
                        "getarrayitem_vable_r",
                        vable_getarrayitem_ref_graph_args(frame_var.into(), v_local_idx.into()),
                        Kind::Ref,
                        load_fast_py_pc,
                    );
                    pin!(Some(v_loaded), stack_base + $depth);
                    let loaded: super::flow::FlowValue = v_loaded.into();
                    let v_stack_idx: super::flow::FlowValue =
                        super::flow::Constant::signed(stack_slot).into();
                    record_graph_op(
                        &current_block.block(),
                        "setarrayitem_vable_r",
                        vable_setarrayitem_ref_graph_args(
                            frame_var.into(),
                            v_stack_idx.into(),
                            loaded.clone().into(),
                        ),
                        None,
                        load_fast_py_pc,
                    );
                    emit_vable_setarrayitem_ref_const_idx!(
                        portal_frame_reg,
                        0_u16,
                        stack_slot,
                        stack_base + $depth
                    );
                    current_state.stack.push(loaded);
                    $depth += 1;
                    emit_vsd!($depth, load_fast_py_pc);
                } else {
                    let loaded = current_state
                        .local_value_at(reg as usize)
                        .unwrap_or_else(|| fresh_ref_value(&mut graph));
                    current_state.stack.push(loaded.clone());
                    emit_pushvalue_ref!($depth, reg, loaded, load_fast_py_pc);
                }
            }};
        }

        // `pypy/interpreter/pyframe.py:?? STORE_FAST` lowers
        // `self.locals_cells_stack_w[varindex] = w_newvalue` to a
        // single `setarrayitem_vable_r` via `jtransform.py:1898
        // do_fixed_list_setitem` (vable branch).  Upstream's local
        // model is the virtualizable array — there is NO separate
        // local register that mirrors the array slot.  LOAD_FAST
        // reads via `getarrayitem_vable_r` from the same array
        // (`jtransform.py:1877 do_fixed_list_getitem`), so the
        // mirror is redundant on portal frames where push/pop
        // routes through the array.
        //
        // Non-portal frames have no vable; the inline `ref_copy`
        // is their only stack-maintenance mechanism and LOAD_FAST
        // (`codewriter.rs:5094-5101 else branch`) reads
        // Reg(Ref, reg) directly.  Keep the mirror for those.
        macro_rules! emit_store_local_with_mirror {
            ($reg:expr, $stored_reg:expr) => {{
                let reg = $reg;
                let stored_reg = $stored_reg;
                if is_portal {
                    emit_vable_setarrayitem_ref_const_idx!(
                        portal_frame_reg,
                        0_u16,
                        local_to_vable_slot(reg as usize) as i64,
                        stored_reg
                    );
                } else {
                    emit_ref_copy(&current_block, reg, stored_reg);
                }
            }};
        }

        // Seed the outer walker queue.  Matches
        // `flowcontext.py:401` `pendingblocks = deque([startblock])`.
        pendingblocks.push_back(current_block.clone());

        // flowcontext.py:402-405 `while self.pendingblocks: block =
        // self.pendingblocks.popleft(); if not block.dead: self.record_block(block)`.
        //
        // Outer loop: outer loop wraps main drain + catch landings
        // emit so handler-entry blocks queued by
        // `emit_goto!(handler_py_pc)` in catch landings get drained by
        // a second main-drain pass.  Without this, handler-entry blocks
        // would be orphans with 0 ops + 0 exits + framestate-wide
        // inputargs, tripping `make_return`'s 1-or-2-arg invariant in
        // canonical `flatten_graph` (`flatten.py:107-109`).
        let mut catch_landings_processed = false;
        loop {
            while let Some(pending_block) = pendingblocks.pop_front() {
                if pending_block.dead() {
                    continue;
                }
                let pending_state = pending_block
                    .framestate()
                    .expect("pending block must carry a FrameState (flowcontext.py:408)");
                let start_pc = pending_state.next_offset;
                // Mirrors upstream's `flowcontext.py:404 if not
                // block.dead: record_block(block)` identity-only check.
                // Supersede re-walks under widened framestate may
                // produce duplicate `-live-` markers for a Python PC;
                // the walker-tracked `walker_pc_live_marker_pos`
                // first-live-wins resolver picks the original dead
                // block's marker as canonical for `pc_map`.
                current_block = pending_block;
                current_state = pending_state;
                current_depth = current_state.stack.len() as u16;
                needs_fallthrough = true;
                // Per-block walker: reset switch flag at the start of
                // every new block iteration so a previous block's
                // queued switch doesn't bleed into this one.
                block_switch_pending = false;
                // Block-entry `Label(block)` per `flatten.py:116
                // self.emitline(Label(block))`.  Emitted at the moment
                // a freshly-popped block becomes `current_block`,
                // mirroring upstream's recursive `make_bytecode_block`
                // top.  Skipped when the per-block accumulator already
                // contains content (mergeblock candidate joins emit
                // into the block before the pop in some flows).
                if current_block.per_block_ssarepr_len() == 0 {
                    push_walker_emit(
                        &current_block,
                        super::flatten::Insn::Label(super::flatten::Label::new(
                            super::flatten::block_label_name(&current_block.block()),
                        )),
                    );
                }
                // Note — upstream `flowcontext.py:407-416`
                // drives per-block op accumulation via `while True:
                // handle_bytecode(...)` until a terminator, then
                // `record_block` assigns `block.operations` from the
                // recorder.  Pyre iterates PCs linearly because the walker
                // emits directly into program-wide `ssarepr.insns`.
                for py_pc in start_pc..num_instrs {
                    // Exception handler entry: Python resets stack depth to the
                    // handler's specified depth and arrives only from
                    // `catch_exception` edges, not from sequential fallthrough.
                    if handler_depth_at.get(py_pc).map_or(false, |v| v.is_some()) {
                        // When reached sequentially from a prior PC
                        // (start_pc != py_pc), break.  Handler PCs are
                        // reached only via exception edges in upstream
                        // RPython (`flowcontext.py:130-156
                        // guessexception`); pyre's analogous catch
                        // landings `emit_goto!(handler_py_pc)` create
                        // the handler-entry block, which the outer-loop
                        // second drain pass walks when start_pc ==
                        // handler_py_pc.
                        if start_pc != py_pc {
                            break;
                        }
                        if let Some(handler_state) = handler_entry_state_from_catch_sites(
                            code,
                            &mut graph,
                            &catch_sites,
                            py_pc,
                        ) {
                            current_depth = handler_state.stack.len() as u16;
                            current_state = handler_state;
                            needs_fallthrough = false;
                        } else if let Some(handler_depth) =
                            handler_depth_at.get(py_pc).copied().flatten()
                        {
                            current_depth = handler_depth;
                            // Bare handler entry without a recorded
                            // FrameState: only the depth is known, so
                            // overwrite the whole symbolic stack with
                            // null sentinels.  `resize` would preserve
                            // any prefix carried over from the predecessor
                            // block, leaking stale symbolic values into
                            // the handler entry; a consumer reading
                            // `current_state.stack[i]` here sees the
                            // sentinel and falls back to the raw
                            // `stack_base + i` register.
                            current_state.stack =
                                vec![null_stack_sentinel(); handler_depth as usize];
                        }
                    }
                    // RPython flatten.py: Label(block) at block entry
                    emit_mark_label_pc!(py_pc);
                    // Yield-on-switch: if `emit_mark_label_pc!` detected
                    // a block boundary at this PC and queued the new
                    // block to `pendingblocks`, break the inner loop
                    // and let the outer walker pop the new block in its
                    // own iteration.  The new block's outer iter then
                    // re-enters at PC=py_pc and the same
                    // `emit_mark_label_pc!` resolves to the new
                    // current_block.
                    if block_switch_pending {
                        break;
                    }
                    // The walker emits one block-identity `Label(block)`
                    // at block entry (`flatten.py:116` parity) and
                    // tracks each Python PC's `-live-` marker position
                    // in `walker_pc_live_marker_pos` for `pc_map`
                    // population at finalize time.
                    depth_at_pc[py_pc] = current_depth;

                    // jtransform.py:1708-1712 emits [op3, op1, op2]:
                    //   op3 = -live- (for inlined short preambles)
                    //   op1 = jit_merge_point
                    //   op2 = -live- (for do_recursive_call / guard resume)
                    // The per-PC emit_live_placeholder!() after this block
                    // serves as op2; op3 is emitted inside the block below.
                    // `walker_pc_live_marker_pos[py_pc]` accumulates both
                    // positions; the post-walk resolver in `resolve_walker_pc`
                    // picks the FIRST per-PC entry whose block contributes
                    // non-empty content to the final SSARepr, so blackhole
                    // guard-failure resume lands consistently.
                    if loop_header_pcs.contains(&py_pc) {
                        // jtransform.py:1710-1711 op3: -live- before
                        // jit_merge_point, "for inlined short preambles".
                        emit_live_placeholder!();
                        if is_portal {
                            let jdindex = portal_jd_index
                                .expect("portal jit_merge_point requires a registered jitdriver");
                            let scratch_pycode_reg =
                                ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                            let pycode_var = emit_vable_getfield_ref!(
                                portal_frame_reg,
                                scratch_pycode_reg,
                                VABLE_CODE_FIELD_IDX
                            )
                            .expect(
                                "portal jit_merge_point requires is_portal=true; \
                             emit_vable_getfield_ref! must return a per-SpaceOp \
                             Variable for the `pycode` green arg",
                            );
                            let graph_args = portal_jit_merge_point_graph_args(
                                &graph, py_pc, pycode_var, jdindex,
                            );
                            let graph_op = emit_graph_op_void(
                                &current_block.block(),
                                "jit_merge_point",
                                graph_args,
                                py_pc as i64,
                            );
                            let pre_len = ssarepr.insns.len();
                            // Build a Ref-only regallocs that maps the
                            // 3 portal Variables (frame / ec / pycode)
                            // to their pre-assigned register indices.
                            // The walker emits jit_merge_point inline
                            // outside the canonical graph regalloc pass,
                            // so no `regallocs[]` entry exists for them
                            // — this site assembles one ad hoc.
                            let mut portal_ref_coloring = std::collections::HashMap::new();
                            portal_ref_coloring.insert(frame_var.id, portal_frame_reg);
                            portal_ref_coloring.insert(ec_var.id, portal_ec_reg);
                            portal_ref_coloring.insert(pycode_var.id, scratch_pycode_reg);
                            let mut portal_regallocs = [
                                super::regalloc::GraphAllocationResult {
                                    coloring: std::collections::HashMap::new(),
                                    num_colors: 0,
                                },
                                super::regalloc::GraphAllocationResult {
                                    coloring: portal_ref_coloring,
                                    num_colors: 3,
                                },
                                super::regalloc::GraphAllocationResult {
                                    coloring: std::collections::HashMap::new(),
                                    num_colors: 0,
                                },
                            ];
                            GraphFlattener::new(&graph, &mut portal_regallocs, &mut ssarepr)
                                .serialize_op(&graph_op);
                            for insn in ssarepr.insns[pre_len..].iter().cloned() {
                                current_block.push_insn(insn);
                            }
                        }
                    }

                    emit_live_placeholder!();
                    // Record the per-PC `-live-` marker position at
                    // walker emit time.  `emit_live_placeholder!()`
                    // pushed the `-live-` as the last insn in
                    // `current_block.per_block_ssarepr`.  Record EVERY
                    // emit (not just first) — `mark_dead` later may
                    // clear the recorded block's accumulator, so the
                    // drain-time resolver picks the first entry whose
                    // block contributes non-empty content to the
                    // final SSARepr.
                    let offset = current_block.per_block_ssarepr_len() - 1;
                    walker_pc_live_marker_pos[py_pc].push((current_block.clone(), offset));

                    // Dead-code dispatch gate: `current_block` has already
                    // been closed by a previous terminator emit (`emit_goto!`,
                    // `emit_ref_return!`, `emit_raise!`, `emit_reraise!`,
                    // POP_JUMP_IF) that appended a normal-flow / bool /
                    // raise / return exit, and no joinpoint exists at
                    // `py_pc`.  The per-PC `-live-` has been emitted to
                    // satisfy pyre's per-PC dispatch invariant
                    // (`walker_pc_live_marker_pos`), but dispatching the
                    // op would append more SpaceOps and potentially more
                    // exits (orphan `(None,None)` link) to the closed
                    // block.  Upstream `flowcontext.py:407-475` never
                    // reaches dead-code PCs because `StopFlowing` from
                    // `closeblock` pops the next pending block; pyre's
                    // PC-sequential walker scans through but skips dispatch.
                    //
                    // The `exitswitch=LastException` exclusion preserves
                    // dispatch after `attach_catch_exception_edge` — the
                    // canraise op attaches its exception edge but the walker
                    // is expected to continue processing the canraise op's
                    // normal-flow result and subsequent ops.  `emit_abort_
                    // permanent!` is excluded by checking `exits.is_empty()`
                    // — it inserts a runtime abort marker without closing
                    // the block, so subsequent ops still need dispatch for
                    // stack-depth parity (e.g. `Instruction::LoadName` +
                    // `Instruction::StoreName` patterns at module scope).
                    let block_closed_by_terminator = {
                        let block_rc = current_block.block();
                        let block = block_rc.borrow();
                        !block.exits.is_empty()
                            && !matches!(
                                block.exitswitch,
                                Some(super::flow::ExitSwitch::Value(super::flow::FlowValue::Constant(
                                    ref c,
                                ))) if matches!(
                                    c.value,
                                    super::flow::ConstantValue::Atom(super::flow::Atom::LastException)
                                )
                            )
                    };
                    if block_closed_by_terminator {
                        current_state.next_offset = py_pc + 1;
                        current_state.blocklist =
                            frame_blocks_for_offset(code, current_state.next_offset);
                        continue;
                    }

                    let code_unit = code.instructions[py_pc];
                    let (instruction, op_arg) = arg_state.get(code_unit);

                    // pyframe.py:379-417 pushvalue/popvalue_maybe_none parity:
                    // RPython's push/pop each write `self.valuestackdepth = depth +/- 1`.
                    // On the JIT, these map to per-push `setfield_vable_i`. pyre's
                    // codewriter stores stack values in typed registers rather than
                    // the `locals_cells_stack_w` array, so we cannot emit a vable
                    // setitem for each push. As the coarsest RPython-compatible
                    // approximation we flush `valuestackdepth` once at opcode entry,
                    // reflecting the pre-opcode stack depth — which is what the
                    // interpreter (eval.rs:92 `target_depth = frame.nlocals() +
                    // frame.ncells() + entry.depth`) uses when an exception handler
                    // unwinds the frame.
                    //
                    // RPython interp_jit.py keeps `next_instr` as a green portal
                    // argument and updates `last_instr` in the interpreter loop; it
                    // does not lower a per-bytecode virtualizable write here. pyre's
                    // portal entry / guard-resume paths already restore
                    // `frame.next_instr`, and the interpreter updates `last_instr`
                    // once execution returns there. Emitting `py_pc + 1` here only
                    // grows the int constant pool linearly with function size and
                    // trips assembler.py's 256-entry cap.
                    // pyframe.py:379-417: valuestackdepth is written per-push/per-pop
                    // via setfield_vable_i (jtransform.py:923-928), NOT once at opcode
                    // entry. The per-push/per-pop emit_vsd! calls below mirror that.
                    // (The old single-entry flush is removed.)

                    // RPython jtransform.py: rewrite_operation() dispatches per opname.
                    // Each match arm is the pyre equivalent of rewrite_op_*.
                    match instruction {
                        Instruction::Resume { .. }
                        | Instruction::Nop
                        | Instruction::Cache
                        | Instruction::NotTaken
                        | Instruction::ExtendedArg => {
                            // RPython: no-op operations produce no jitcode output
                        }

                        // jtransform.py:1877 do_fixed_list_getitem vable case:
                        // portal locals are virtualizable array items — emit
                        // vable_getarrayitem_ref so the optimizer folds the read
                        // against virtualizable_boxes and the blackhole pulls the
                        // live frame value into stack_base+current_depth on
                        // resume. Non-portal frames keep ref_copy (no virtualizable
                        // in scope).
                        Instruction::LoadFast { var_num }
                        | Instruction::LoadFastBorrow { var_num } => {
                            let reg = var_num.get(op_arg).as_usize() as u16;
                            emit_load_fast_ref!(current_depth, reg, py_pc);
                        }

                        // jtransform.py:1898 do_fixed_list_setitem vable case:
                        // Portal frames treat `locals_cells_stack_w` as the sole
                        // storage for locals — setarrayitem_vable_r writes from
                        // the value-stack slot directly, so no register-per-local
                        // shadow exists. Non-portal frames keep ref_copy (no vable
                        // in scope).
                        Instruction::StoreFast { var_num } => {
                            let reg = var_num.get(op_arg).as_usize() as u16;
                            let stored_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let stored = pop_ref_or_fresh(&mut current_state, &mut graph);
                            if is_portal {
                                // Graph dual-write of jtransform.py:1898
                                // `do_fixed_list_setitem` — STORE_FAST →
                                // `setarrayitem_vable_r(locals_cells_stack_w,
                                // local_slot, w_value)`.  `frame_var` is a
                                // startblock inputarg for both portal and
                                // non-portal graphs (Epic #245 Slice 1);
                                // the dual-write stays portal-gated until
                                // the non-portal readers move off the
                                // shared `portal_frame_reg` slot.
                                let local_slot = local_to_vable_slot(reg as usize) as i64;
                                let v_idx: super::flow::FlowValue =
                                    super::flow::Constant::signed(local_slot).into();
                                record_graph_op(
                                    &current_block.block(),
                                    "setarrayitem_vable_r",
                                    vable_setarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        v_idx.into(),
                                        stored.clone().into(),
                                    ),
                                    None,
                                    -1,
                                );
                            }
                            emit_store_local_with_mirror!(reg, stored_reg);
                            current_state.store_local_value(reg as usize, stored);
                        }

                        Instruction::LoadSmallInt { i } => {
                            let val = i.get(op_arg) as u32 as i64;
                            // Call writes result directly
                            // to the target stack slot. Safe because the only
                            // call input is a literal constant (no Ref conflict)
                            // and no post-call op reads from that stack slot
                            // before the next opcode's frontend push.
                            // `make_three_lists` (jtransform.py:437-445) admits
                            // `Variable | Constant` directly, so the constant
                            // reaches `expect_list_regs_or_pool`
                            // (assembler.rs:1736-1784) without a scratch register.
                            // box_int_fn factor
                            // refactor.  The prior `emit_residual_call(
                            // box_int_fn_idx, ...)` is replaced by a single
                            // direct push of
                            // `build_box_int_fn_residual_call_ir_r_insn`,
                            // which produces the same `residual_call_ir_r(
                            // ConstInt(fn_idx), ListI([ConstInt(val)]),
                            // ListR([]), Descr) → Reg(dst)` Insn shape
                            // `emit_residual_call_shape` would have
                            // produced (empty `ListR` per RPython
                            // jtransform.py:425 `kinds = 'ir'` whenever
                            // `lst_i` is non-empty).  Helper hardcodes
                            // `CallFlavor::Plain` matching the production
                            // source at codewriter.rs:2202.  Graph
                            // dual-write below is retained.
                            push_walker_emit(
                                &current_block,
                                super::flatten::build_box_int_fn_residual_call_ir_r_insn(
                                    box_int_fn_idx,
                                    val,
                                    stack_base + current_depth,
                                ),
                            );
                            // Graph-side `residual_call_ir_r` for
                            // `box_int_fn(val:Int) → Ref`.  RPython parity:
                            // `flowcontext.py:135-139 self.recorder.append`
                            // produces a fresh result Variable for every
                            // residual_call, and the consumer (here, the
                            // value-stack push) reads that Variable directly
                            // — no separate fresh Ref placeholder.  Thread
                            // the call result Variable into the symbolic
                            // stack so the def-use chain matches the
                            // upstream "call result is the downstream value"
                            // shape.
                            // Walker-orthodoxy: graph residual_call
                            // dual-write fires unconditionally.  `box_int_fn`
                            // takes only a literal Int as input, so no
                            // frame_var or other portal-only Variable is
                            // threaded — the graph op is well-formed for
                            // every CodeWriter regardless of is_portal.
                            let boxed = residual_call!(
                                box_int_fn_idx,
                                CallFlavor::Plain,
                                vec![super::flow::Constant::signed(val).into()],
                                vec![],
                                vec![],
                                vec![Kind::Int],
                                ResKind::Ref,
                                py_pc as i64,
                            );
                            pin!(boxed, stack_base + current_depth);
                            let stack_value = boxed
                                .map(super::flow::FlowValue::from)
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            push_and_bump!(stack_value, py_pc);
                        }

                        Instruction::LoadConst { consti } => {
                            let idx = consti.get(op_arg).as_usize();
                            let dst_slot = stack_base + current_depth;
                            // pyframe.py:509-510 `getcode(): hint(self.pycode,
                            // promote=True)`: the JIT treats `frame.pycode` as
                            // promote-to-constant on every trace.  In a portal
                            // entry the promotion happens at the merge point
                            // via the vable getfield; in a non-portal callee
                            // the pycode is the callee's own `W_Code` pointer,
                            // already a compile-time constant.  Skip the
                            // portal vable getfield in the non-portal branch
                            // and feed `w_code` as `ConstRef` directly — the
                            // portal `portal_frame_reg` would otherwise alias
                            // the caller's frame and read the wrong pycode.
                            if is_portal {
                                emit_vable_getfield_ref_walker_only(
                                    &current_block,
                                    portal_frame_reg,
                                    dst_slot,
                                    VABLE_CODE_FIELD_IDX,
                                );
                            }
                            if is_portal {
                                push_walker_emit(
                                    &current_block,
                                    super::flatten::build_load_const_fn_residual_call_ir_r_insn(
                                        load_const_fn_idx,
                                        idx as i64,
                                        dst_slot,
                                        dst_slot,
                                    ),
                                );
                            } else {
                                push_walker_emit(
                                    &current_block,
                                    super::flatten::build_load_const_fn_residual_call_ir_r_insn_with_const_pycode(
                                        load_const_fn_idx,
                                        idx as i64,
                                        w_code as i64,
                                        dst_slot,
                                    ),
                                );
                            }
                            // Graph-side RPython parity: `flowcontext.py:841-843`
                            // resolves LOAD_CONST to a Constant and pushes it.
                            // Do not record the pyre runtime helper as a
                            // SpaceOperation; that helper is walker/backend
                            // adaptation only.
                            let value = frontend_load_const_flow_value(code, idx);
                            push_and_bump!(value, py_pc);
                        }

                        // CPython super-instructions LOAD_FAST_LOAD_FAST /
                        // LOAD_FAST_BORROW_LOAD_FAST_BORROW decompose to two plain
                        // LOAD_FAST reads. Keep the portal virtualizable lowering
                        // identical to plain LoadFast: every local read goes
                        // through getarrayitem_vable_r so blackhole resume can
                        // reload dead-at-resume locals on demand, as RPython does
                        // via jtransform.py:1877 do_fixed_list_getitem.
                        Instruction::LoadFastBorrowLoadFastBorrow { var_nums }
                        | Instruction::LoadFastLoadFast { var_nums } => {
                            let pair = var_nums.get(op_arg);
                            let reg_a = u32::from(pair.idx_1()) as u16;
                            let reg_b = u32::from(pair.idx_2()) as u16;
                            emit_load_fast_ref!(current_depth, reg_a, py_pc);
                            emit_load_fast_ref!(current_depth, reg_b, py_pc);
                        }

                        // Super-instruction STORE_FAST; LOAD_FAST: pop TOS into
                        // idx_1 (store), then push idx_2 (load). Net depth 0.
                        // Portal: store via setarrayitem_vable_r, load via
                        // getarrayitem_vable_r. Non-portal: ref_copy for both halves.
                        Instruction::StoreFastLoadFast { var_nums } => {
                            let pair = var_nums.get(op_arg);
                            let store_reg = u32::from(pair.idx_1()) as u16;
                            let load_reg = u32::from(pair.idx_2()) as u16;
                            let stored_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let stored = pop_ref_or_fresh(&mut current_state, &mut graph);
                            if is_portal {
                                // STORE_FAST half graph dual-write
                                // (jtransform.py:1898 `do_fixed_list_setitem`).
                                let store_slot = local_to_vable_slot(store_reg as usize) as i64;
                                let v_store_idx: super::flow::FlowValue =
                                    super::flow::Constant::signed(store_slot).into();
                                record_graph_op(
                                    &current_block.block(),
                                    "setarrayitem_vable_r",
                                    vable_setarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        v_store_idx.into(),
                                        stored.clone().into(),
                                    ),
                                    None,
                                    -1,
                                );
                            }
                            // STORE_FAST half: same dual-write as Instruction::StoreFast.
                            // Non-portal popvalue places `stored_reg` at
                            // `stack_base + current_depth` post-decrement, so the
                            // macro's `ref_copy(store_reg, stored_reg)` is
                            // equivalent to the prior explicit
                            // `ref_copy(store_reg, stack_base + current_depth)`.
                            emit_store_local_with_mirror!(store_reg, stored_reg);
                            if is_portal {
                                let load_slot = local_to_vable_slot(load_reg as usize) as i64;
                                let stack_slot =
                                    (stack_base_absolute + current_depth as usize) as i64;
                                // LOAD_FAST half: read local, then pyframe.py:378-381
                                // pushvalue parity — mirror to the value-stack slot.
                                emit_vable_getarrayitem_ref_const_idx!(
                                    portal_frame_reg,
                                    stack_base + current_depth,
                                    0_u16,
                                    load_slot
                                );
                                // CPython 3.13 super-instruction semantics: STORE
                                // is observable to the immediately-following LOAD
                                // when store_reg == load_reg. Apply the locals_w
                                // update before recording the graph LOAD half so
                                // any prior Variable on `store_reg` is replaced
                                // with `stored` first.
                                current_state.store_local_value(store_reg as usize, stored);
                                // Graph-side dual-write of BOTH halves of the
                                // LOAD half lowering — symmetric to
                                // `emit_load_fast_ref!` (codewriter.rs:3833+):
                                //   - local read: jtransform.py:1877
                                //     `do_fixed_list_getitem` lowers
                                //     `locals_cells_stack_w[load_slot]` to
                                //     `getarrayitem_vable_r(_, ConstInt(load_slot))`,
                                //     producing a Ref result.  Every read in SSA
                                //     form produces a fresh Variable; when
                                //     load_reg == store_reg the optimizer is
                                //     responsible for CSE'ing the read back to
                                //     `stored`.
                                //   - stack write: jtransform.py:1898
                                //     `do_fixed_list_setitem` lowers the
                                //     subsequent `pushvalue(loaded)` to
                                //     `setarrayitem_vable_r(_, ConstInt(stack_slot),
                                //     loaded)`.
                                //
                                // `current_state.locals_w[load_reg]` is left
                                // UNCHANGED — the LOAD half of
                                // StoreFastLoadFast is a stack push, not a
                                // local-binding mutation.  When
                                // load_reg == store_reg the just-set
                                // `Some(stored)` from the STORE half
                                // remains as the slot's Variable
                                // (matching pyopcode.py super-instruction
                                // semantics).
                                let v_load_idx: super::flow::FlowValue =
                                    super::flow::Constant::signed(load_slot).into();
                                let v_loaded = emit_graph_op_with_result(
                                    &mut graph,
                                    &current_block.block(),
                                    "getarrayitem_vable_r",
                                    vable_getarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        v_load_idx.into(),
                                    ),
                                    Kind::Ref,
                                    -1,
                                );
                                pin!(Some(v_loaded), stack_base + current_depth);
                                let loaded: super::flow::FlowValue = v_loaded.into();
                                let v_stack_idx: super::flow::FlowValue =
                                    super::flow::Constant::signed(stack_slot).into();
                                record_graph_op(
                                    &current_block.block(),
                                    "setarrayitem_vable_r",
                                    vable_setarrayitem_ref_graph_args(
                                        frame_var.into(),
                                        v_stack_idx.into(),
                                        loaded.clone().into(),
                                    ),
                                    None,
                                    -1,
                                );
                                emit_vable_setarrayitem_ref_const_idx!(
                                    portal_frame_reg,
                                    0_u16,
                                    stack_slot,
                                    stack_base + current_depth
                                );
                                push_and_bump!(loaded, py_pc);
                            } else {
                                current_state.store_local_value(store_reg as usize, stored);
                                let loaded = current_state
                                    .local_value_at(load_reg as usize)
                                    .unwrap_or_else(|| fresh_ref_value(&mut graph));
                                current_state.stack.push(loaded.clone());
                                emit_pushvalue_ref!(current_depth, load_reg, loaded, py_pc);
                            }
                        }

                        // STORE_SUBSCR: stack [value, obj, key] → obj[key] = value
                        Instruction::StoreSubscr => {
                            // Pass stack slots directly as call args,
                            // retiring obj_tmp0/obj_tmp1/arg_regs_start staging.
                            // Inputs are read by the backend ABI into call regs
                            // before the call executes; no write-back conflicts
                            // because ResKind::Void.
                            current_depth -= 1;
                            emit_vsd!(current_depth, py_pc);
                            let key_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let key_reg = stack_base + current_depth;
                            current_depth -= 1;
                            emit_vsd!(current_depth, py_pc);
                            let obj_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let obj_reg = stack_base + current_depth;
                            current_depth -= 1;
                            emit_vsd!(current_depth, py_pc);
                            let stored_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let value_reg = stack_base + current_depth;
                            emit_frontend_setitem(
                                &mut graph,
                                &current_block.block(),
                                obj_value,
                                key_value,
                                stored_value,
                                py_pc as i64,
                            );
                            // SETITEM family retirement: emit the lowered
                            // `residual_call_r_v` Insn directly here via the
                            // `(Ref, Ref, Ref) → Void` shape constructor.
                            // Graph carries only the void
                            // `setitem(obj, key, value)` HLOp from
                            // `emit_frontend_setitem` above.
                            push_walker_emit(
                                &current_block,
                                super::flatten::build_store_subscr_fn_residual_call_r_v_insn(
                                    store_subscr_fn_idx,
                                    obj_reg,
                                    key_reg,
                                    value_reg,
                                ),
                            );
                        }

                        Instruction::PopTop => {
                            let _ = emit_popvalue_ref!(current_depth, py_pc);
                            let _ = current_state.stack.pop();
                            // flowcontext.py:891 `self.popvalue()`; regalloc.py:
                            // discard = just decrement depth, no bytecode.
                        }

                        Instruction::PushNull => {
                            current_state.stack.push(null_stack_sentinel());
                            emit_pushvalue_ref_const!(
                                current_depth,
                                pyre_object::PY_NULL as i64,
                                py_pc
                            );
                        }

                        // jtransform.py: rewrite_op_int_add etc.
                        //
                        // Call reads stack slots DIRECTLY rather than copying through
                        // obj_tmp0/obj_tmp1 temps. This keeps the call's argument
                        // registers inside the trace-tracked range (`registers_r`
                        // + `symbolic_stack`), so guards fired across the op (e.g.
                        // `GUARD_NOT_FORCED_2` after a helper call) capture the
                        // lhs/rhs values in fail_args. See
                        // `memory/pyre_trace_temp_reg_tracking_gap_2026_04_19.md`.
                        Instruction::BinaryOp { op } => {
                            let op_kind = op.get(op_arg);
                            let op_val = binary_op_tag(op_kind)
                                .expect("unsupported binary op tag in jitcode lowering")
                                as i64;
                            // Pop rhs (blackhole will see vsd reflect this pop).
                            let rhs_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let rhs_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // Pop lhs.
                            let lhs_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let lhs_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_frontend_binary(
                                &mut graph,
                                &current_block.block(),
                                op_kind,
                                lhs_value,
                                rhs_value,
                                py_pc as i64,
                            );
                            pin!(Some(result_value), stack_base + current_depth);
                            // BINARY_OP family retirement: emit the lowered
                            // `residual_call_ir_r` Insn directly here via
                            // `build_binary_op_residual_call_ir_r_insn`.
                            // Graph carries only the `add(lhs, rhs)` HLOp
                            // recorded by `emit_frontend_binary` above; the
                            // helper produces the same Insn bytes the
                            // post-walker `flatten_graph(graph, regallocs)`
                            // dispatcher would emit from that HLOp.
                            push_walker_emit(
                                &current_block,
                                super::flatten::build_binary_op_residual_call_ir_r_insn(
                                    binary_op_fn_idx,
                                    op_val,
                                    lhs_reg,
                                    rhs_reg,
                                    stack_base + current_depth,
                                ),
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // jtransform.py: rewrite_op_int_lt, optimize_goto_if_not
                        Instruction::CompareOp { opname } => {
                            // Same stack-direct pattern as BinaryOp — see its comment.
                            let op_kind = opname.get(op_arg);
                            let op_val = compare_op_tag(op_kind);
                            let rhs_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let rhs_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let lhs_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let lhs_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_frontend_compare(
                                &mut graph,
                                &current_block.block(),
                                op_kind,
                                lhs_value,
                                rhs_value,
                                py_pc as i64,
                            );
                            pin!(Some(result_value), stack_base + current_depth);
                            // COMPARE_OP family retirement: same closure as
                            // BinaryOp above.  Graph carries only the
                            // `lt(lhs, rhs)` (or sibling) HLOp from
                            // `emit_frontend_compare`; the SSARepr Insn is
                            // built here by the helper whose output shape
                            // matches `lower_compare_op_hlop_to_insn`.
                            push_walker_emit(
                                &current_block,
                                super::flatten::build_compare_op_residual_call_ir_r_insn(
                                    compare_fn_idx,
                                    op_val,
                                    lhs_reg,
                                    rhs_reg,
                                    stack_base + current_depth,
                                ),
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // flatten.py:240-260 + blackhole.py:865-869. truth_fn returns
                        // a bool-as-int; emit plain `goto_if_not <bool> L` — the
                        // unfused form flatten.py takes when the exitswitch is a
                        // plain variable (not a tuple of a foldable comparison op).
                        // bhimpl_goto_if_not takes the target when `a == 0`.
                        Instruction::PopJumpIfFalse { delta } => {
                            let target_py_pc = jump_target_forward(
                                code,
                                num_instrs,
                                py_pc + 1,
                                delta.get(op_arg).as_usize(),
                            );
                            // truth_fn reads cond directly from the popped
                            // stack slot; `popvalue_ref` leaves the value at
                            // `stack_base + current_depth` (the slot below the new
                            // TOS), so there is no staging copy — mirrors upstream
                            // flatten.py:240-260 which feeds the Variable straight to
                            // `goto_if_not`.
                            let cond_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let cond_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            if let super::flow::FlowValue::Variable(v) = &cond_value {
                                pin!(Some(*v), cond_reg);
                            }
                            let bool_value = emit_frontend_bool(
                                &mut graph,
                                &current_block.block(),
                                cond_value,
                                py_pc as i64,
                            );
                            // flowcontext.py:756-763 `block.exitswitch = w_cond`.
                            current_block.block().borrow_mut().exitswitch =
                                Some(super::flow::ExitSwitch::Value(bool_value.into()));
                            let scratch_truth = ssarepr.fresh_var(Kind::Int, scratch_int_base).0;
                            pin!(Some(bool_value), scratch_truth);
                            // BOOL family retirement: emit the lowered
                            // `residual_call_r_i` Insn directly here via the
                            // `(Ref) → Int` shape constructor.  Graph carries
                            // only the `bool(cond_value)` HLOp from
                            // `emit_frontend_bool` above.
                            push_walker_emit(
                                &current_block,
                                super::flatten::build_truth_fn_residual_call_r_i_insn(
                                    truth_fn_idx,
                                    cond_reg,
                                    scratch_truth,
                                ),
                            );
                            // POP_JUMP_IF_FALSE jumps when cond is false; the
                            // bool=true path falls through to PC+1.  Mirror
                            // POP_JUMP_IF_TRUE by attaching BOTH the linkfalse
                            // (to target) and the linktrue (to PC+1) at this
                            // emit point so the closed Bool-exitswitch block
                            // carries both Bool exit cases.  Without the
                            // explicit linktrue mergeblock here, the walker's
                            // PC-sequential continuation into PC+1 reuses the
                            // same `current_block` (no new exit created via
                            // `emit_mark_label_pc!`'s joinpoint arm), leaving
                            // the linktrue link missing and the dispatcher
                            // falling through to the switch path with the
                            // surviving exit's None llexitcase per
                            // `flatten.py:275`.
                            let fallthrough_py_pc = py_pc + 1;
                            if target_py_pc < num_instrs && fallthrough_py_pc < num_instrs {
                                // `rpython/jit/codewriter/flatten.py:240-267
                                // insert_exits` (Bool 2-exit arm) emits
                                // `make_link(linktrue)` BEFORE
                                // `make_link(linkfalse)` so the TRUE arm
                                // body is physically next after
                                // `goto_if_not`, leaving `linkfalse` to
                                // reach its `Label(linkfalse)` via
                                // explicit dispatch.  Walker must match
                                // this order so the boundary
                                // `goto pc{TRUE_arm}` emitted by the next
                                // PC's `emit_mark_label_pc!` strips
                                // cleanly against the immediately-following
                                // TRUE arm block (see
                                // `strip_walker_block_boundary_goto`).
                                // Mergeblock the fallthrough (TRUE arm)
                                // FIRST so it pops first from
                                // `pendingblocks`; `emit_goto_if_not!`
                                // then appends the FALSE link.  Canonical
                                // `flatten.rs:1875-1880 insert_exits`
                                // normalises the [TRUE, FALSE] exits
                                // ordering via the llexitcase swap.
                                mergeblock(
                                    code,
                                    &mut graph,
                                    &mut joinpoints,
                                    &current_block,
                                    &{
                                        let mut branch_state = current_state.clone();
                                        branch_state.next_offset = fallthrough_py_pc;
                                        branch_state.blocklist =
                                            frame_blocks_for_offset(code, fallthrough_py_pc);
                                        branch_state
                                    },
                                    fallthrough_py_pc,
                                    &mut pendingblocks,
                                    &mut all_walker_blocks,
                                );
                                set_last_bool_exitcase(&current_block.block(), true);
                                emit_goto_if_not!(scratch_truth, target_py_pc);
                                set_last_bool_exitcase(&current_block.block(), false);
                            }
                        }

                        // flowcontext.py:761-763 POP_JUMP_IF_TRUE still branches on
                        // `guessbool(op.bool(w_value).eval(self))`, so upstream
                        // flatten.py handles it as the same generic Bool exitswitch
                        // shape as POP_JUMP_IF_FALSE. The polarity difference is only
                        // in the link ordering: jump target = True path, fallthrough =
                        // False path.
                        Instruction::PopJumpIfTrue { delta } => {
                            let target_py_pc = jump_target_forward(
                                code,
                                num_instrs,
                                py_pc + 1,
                                delta.get(op_arg).as_usize(),
                            );
                            // See PopJumpIfFalse — no obj_tmp0 staging
                            // needed; the residual call reads the popped stack slot.
                            let cond_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let cond_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            if let super::flow::FlowValue::Variable(v) = &cond_value {
                                pin!(Some(*v), cond_reg);
                            }
                            let bool_value = emit_frontend_bool(
                                &mut graph,
                                &current_block.block(),
                                cond_value,
                                py_pc as i64,
                            );
                            // flowcontext.py:756-763 `block.exitswitch = w_cond`.
                            current_block.block().borrow_mut().exitswitch =
                                Some(super::flow::ExitSwitch::Value(bool_value.into()));
                            let scratch_truth = ssarepr.fresh_var(Kind::Int, scratch_int_base).0;
                            pin!(Some(bool_value), scratch_truth);
                            // BOOL family
                            // retirement (sibling of the PopJumpIfFalse
                            // closure above) — same `(Ref) → Int` shape
                            // helper, same probe coverage.
                            push_walker_emit(
                                &current_block,
                                super::flatten::build_truth_fn_residual_call_r_i_insn(
                                    truth_fn_idx,
                                    cond_reg,
                                    scratch_truth,
                                ),
                            );
                            // `flatten.py:244-267` for a Bool exitswitch always
                            // emits generic `goto_if_not cond, TLabel(linkfalse)`
                            // + inline `make_link(linktrue)`.
                            // `linkfalse.llexitcase == False`, so for
                            // POP_JUMP_IF_TRUE the False link is the fallthrough
                            // (PC+1) and the True link is the jump target.  The
                            // specialised `goto_if_not_<opname>` form is reserved
                            // for tuple exitswitches produced by
                            // `jtransform.optimize_goto_if_not` (comparisons plus
                            // zero/nonzero-style predicates), not generic Bool
                            // exitswitches like this truthiness branch.
                            let fallthrough_py_pc = py_pc + 1;
                            if target_py_pc < num_instrs && fallthrough_py_pc < num_instrs {
                                // `rpython/jit/codewriter/flatten.py:240-267
                                // insert_exits` (Bool 2-exit arm) emits
                                // `make_link(linktrue)` BEFORE
                                // `make_link(linkfalse)`.  For POP_JUMP_IF_TRUE
                                // linktrue = target_py_pc (jump on TRUE), so
                                // target_block must physically follow parent
                                // immediately in walker emission order to match
                                // canonical's inline `make_link(linktrue)`
                                // body.  Mergeblock the target FIRST so it
                                // pops first from `pendingblocks`, then
                                // `emit_goto_if_not!` appends the FALSE link
                                // and queues fallthrough second.  Unlike
                                // POP_JUMP_IF_FALSE (whose linktrue IS the
                                // PC-sequential next block), POP_JUMP_IF_TRUE's
                                // linktrue is the jump target — the PC-
                                // sequential walker's auto-switch at PC+1
                                // would inject a `goto pc{PC+1}` boundary
                                // routing through `emit_mark_label_pc!` that
                                // `strip_walker_block_boundary_goto` cannot
                                // elide (next block in walker order is
                                // target_block, not fallthrough_block).
                                // Setting `needs_fallthrough = false` here
                                // suppresses that boundary injection — the
                                // exits are already laid out via the two
                                // mergeblocks above; no implicit fallthrough
                                // link is needed.
                                mergeblock(
                                    code,
                                    &mut graph,
                                    &mut joinpoints,
                                    &current_block,
                                    &{
                                        let mut branch_state = current_state.clone();
                                        branch_state.next_offset = target_py_pc;
                                        branch_state.blocklist =
                                            frame_blocks_for_offset(code, target_py_pc);
                                        branch_state
                                    },
                                    target_py_pc,
                                    &mut pendingblocks,
                                    &mut all_walker_blocks,
                                );
                                set_last_bool_exitcase(&current_block.block(), true);
                                emit_goto_if_not!(scratch_truth, fallthrough_py_pc);
                                set_last_bool_exitcase(&current_block.block(), false);
                                needs_fallthrough = false;
                            }
                        }

                        // RPython flatten.py: goto Label
                        Instruction::JumpForward { delta } => {
                            let target_py_pc = jump_target_forward(
                                code,
                                num_instrs,
                                py_pc + 1,
                                delta.get(op_arg).as_usize(),
                            );
                            if target_py_pc < num_instrs {
                                emit_goto!(target_py_pc);
                            }
                        }

                        instr @ Instruction::JumpBackward { .. } => {
                            if let Some(target_py_pc) =
                                backward_jump_target(code, py_pc, instr, op_arg)
                            {
                                if target_py_pc < num_instrs {
                                    emit_goto!(target_py_pc);
                                }
                            }
                        }

                        // flatten.py: int_return / ref_return
                        Instruction::ReturnValue => {
                            let retval_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let retval = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // ref_return reads from the stack slot
                            // directly — the obj_tmp0 staging was redundant since
                            // this is the terminating op of the block.
                            emit_ref_return!(retval_reg, retval);
                        }

                        // RPython jtransform.py: rewrite_op_direct_call (residual)
                        Instruction::LoadGlobal { namei } => {
                            let raw_namei = namei.get(op_arg) as usize as i64;
                            // `flowcontext.py:856-859` resolves globals during
                            // flow analysis and pushes the resolved Constant via
                            // `pushvalue(w_value)` — there is NO
                            // `SpaceOperation('load_global', ...)` at the graph
                            // level. Keep that graph shape; the residual helper
                            // below is walker/backend adaptation only.
                            let scratch_ns = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                            // Walker-side getfield_vable_r for w_globals
                            // (field 5) — the namespace for the lookup.  Do not
                            // dual-write into the graph: RPython flowspace's
                            // LOAD_GLOBAL path has already folded the value to a
                            // Constant.  `w_globals` stays a register read: its
                            // traced GetfieldVableR form is what the optimizer
                            // matches in the known-result cache, so a ConstRef
                            // would break that fold (see
                            // `compile_jitcode_for_callee` doc block).
                            emit_vable_getfield_ref_walker_only(
                                &current_block,
                                portal_frame_reg,
                                scratch_ns,
                                VABLE_NAMESPACE_FIELD_IDX,
                            );
                            // Write the load_global result directly to the
                            // stack slot it will occupy after the push (and
                            // after the optional NULL push for the
                            // `raw_namei & 1` LOAD_GLOBAL(push_null) variant).
                            // The trailing `emit_pushvalue_ref!` then sees
                            // `src == dst` and elides its `ref_copy` per the
                            // identity-elide guard in `emit_ref_copy`,
                            // matching upstream RPython where pushvalue is
                            // symbolic and the residual_call writes directly
                            // to the consumer slot.
                            let null_offset: u16 = if raw_namei & 1 != 0 { 1 } else { 0 };
                            let loaded_dst_reg = stack_base + current_depth + null_offset;
                            // pyframe.py:509-510 `getcode(): hint(self.pycode,
                            // promote=True)`: `frame.pycode` (field 1) is a
                            // promote-to-constant on every trace.  A portal
                            // entry promotes at the merge point via the vable
                            // getfield; a non-portal callee's pycode is the
                            // callee's own `W_Code` pointer, already a
                            // compile-time constant.  In a chained blackhole
                            // resume `portal_frame_reg` aliases the OUTERMOST
                            // (caller) frame, so reading pycode off it would
                            // pick up the caller's `names` table — the
                            // function_calls miscompile.  Mirror the
                            // `Instruction::LoadConst` arm: portal keeps the
                            // vable getfield + register operand, non-portal
                            // feeds `w_code` directly as `Operand::ConstRef`.
                            if is_portal {
                                let scratch_code = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                                emit_vable_getfield_ref_walker_only(
                                    &current_block,
                                    portal_frame_reg,
                                    scratch_code,
                                    VABLE_CODE_FIELD_IDX,
                                );
                                push_walker_emit(
                                    &current_block,
                                    super::flatten::build_load_global_fn_residual_call_ir_r_insn(
                                        load_global_fn_idx,
                                        raw_namei,
                                        scratch_ns,
                                        scratch_code,
                                        portal_frame_reg,
                                        loaded_dst_reg,
                                    ),
                                );
                            } else {
                                push_walker_emit(
                                    &current_block,
                                    super::flatten::build_load_global_fn_residual_call_ir_r_insn_with_const_pycode(
                                        load_global_fn_idx,
                                        raw_namei,
                                        scratch_ns,
                                        w_code as i64,
                                        portal_frame_reg,
                                        loaded_dst_reg,
                                    ),
                                );
                            }
                            let name_idx = raw_namei as usize >> 1;
                            let result_value = code
                                .names
                                .get(name_idx)
                                .and_then(|name| frontend_global_flow_value(w_code, name.as_ref()))
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            if let super::flow::FlowValue::Variable(v) = &result_value {
                                pin!(Some(*v), loaded_dst_reg);
                            }
                            // LOAD_GLOBAL with (namei >> 1) & 1: push NULL first.
                            // const-source pushvalue writes the constant directly to
                            // the stack TOS register and (in portal case) to the
                            // vable slot via setarrayitem_vable_r_const, leaving
                            // the scratch regs untouched for the trailing
                            // `emit_pushvalue_ref!(loaded_dst_reg)`.
                            if raw_namei & 1 != 0 {
                                current_state.stack.push(null_stack_sentinel());
                                emit_pushvalue_ref_const!(
                                    current_depth,
                                    pyre_object::PY_NULL as i64,
                                    py_pc
                                );
                            }
                            current_state.stack.push(result_value.clone());
                            // `loaded_dst_reg == stack_base + current_depth` here
                            // (computed before the optional NULL push that bumps
                            // current_depth by `null_offset`), so the trailing
                            // `emit_ref_copy(stack_base + current_depth, loaded_dst_reg)`
                            // inside `emit_pushvalue_ref!` is the identity copy
                            // elided by `emit_ref_copy`'s `dst != src` guard.
                            emit_pushvalue_ref!(current_depth, loaded_dst_reg, result_value, py_pc);
                        }

                        // RPython jtransform.py: rewrite_op_direct_call →
                        // call_may_force / residual_call
                        //
                        // RPython blackhole.py: bhimpl_recursive_call_i calls
                        // portal_runner directly, bypassing JIT entry.
                        // Here we pop args and callable from the stack into
                        // registers, then call the helper with explicit args.
                        //
                        // shared_opcode.rs:56 opcode_call parity:
                        // Stack layout before CALL(argc):
                        //   [callable, null_or_self, arg0, ..., arg(argc-1)]
                        // Pop in reverse: args, null_or_self, callable.
                        Instruction::Call { argc } => {
                            let nargs = argc.get(op_arg) as usize;
                            let mut arg_regs_rev: Vec<u16> = Vec::with_capacity(nargs);
                            let mut graph_arg_values_rev = Vec::with_capacity(nargs);
                            for _ in 0..nargs {
                                let arg_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let arg_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                if let super::flow::FlowValue::Variable(v) = &arg_value {
                                    pin!(Some(*v), arg_reg);
                                }
                                arg_regs_rev.push(arg_reg);
                                graph_arg_values_rev.push(arg_value);
                            }
                            // Args were popped in reverse stack order; reverse to
                            // match the call site's positional order (arg0 first).
                            let arg_regs: Vec<u16> = arg_regs_rev.iter().rev().copied().collect();
                            let arg_values: Vec<super::flow::FlowValue> =
                                graph_arg_values_rev.iter().rev().cloned().collect();
                            let callable_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let callable_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let _ = emit_popvalue_ref!(current_depth, py_pc); // NULL (discard)
                            let _null_or_self = pop_ref_or_fresh(&mut current_state, &mut graph);

                            // RPython: bhimpl_recursive_call_i(jdindex, greens, reds)
                            // call_fn(frame, callable, arg0, ...) → result
                            // Parent frame is passed explicitly as the leading
                            // ref operand; no thread-local indirection.
                            // The flatten.rs helper consumes `callable_reg` and
                            // `&arg_regs` directly; no intermediate Vec needed.
                            // Select the correct arity-specific call helper.
                            // RPython blackhole.py: call_int_function transmutes
                            // to the correct arity. Each nargs needs a matching
                            // extern "C" fn with that many i64 parameters.
                            // nargs > 8 → abort_permanent (no matching helper).
                            let call_result_value = if nargs > 8 {
                                fresh_ref_value(&mut graph)
                            } else {
                                // Graph-side `simple_call(callable, args...)`
                                // carries only the RPython rewrite_call shape
                                // (jtransform.py:414 — no hidden frame arg).
                                // `lower_simple_call_hlop_to_insn` prepends
                                // `ctx.portal_frame_reg` to the ListR at flatten
                                // time so the lowered Insn matches the inline
                                // walker emit at codewriter.rs:6784-6788.
                                let graph_call_args: Vec<super::flow::FlowValue> =
                                    graph_arg_values_rev.iter().rev().cloned().collect();
                                let result = emit_frontend_simple_call(
                                    &mut graph,
                                    &current_block.block(),
                                    callable_value.clone(),
                                    graph_call_args,
                                    py_pc as i64,
                                );
                                pin!(Some(result), stack_base + current_depth);
                                result.into()
                            };
                            if nargs > 8 {
                                emit_abort_permanent!();
                            } else {
                                let fn_idx = match nargs {
                                    0 => call_fn_0_idx,
                                    1 => call_fn_idx,
                                    2 => call_fn_2_idx,
                                    3 => call_fn_3_idx,
                                    4 => call_fn_4_idx,
                                    5 => call_fn_5_idx,
                                    6 => call_fn_6_idx,
                                    7 => call_fn_7_idx,
                                    _ => call_fn_8_idx,
                                };
                                // CALL family
                                // factor refactor.  The prior
                                // `emit_residual_call(call_fn_N_idx, ...)`
                                // call is replaced by a single direct push
                                // of `build_call_fn_residual_call_r_r_insn`,
                                // which produces the same `residual_call_r_r(
                                // ConstInt(fn_idx), ListR([Reg(frame),
                                // Reg(callable), Reg(arg0), ...,
                                // Reg(arg_{N-1})]), Descr) → Reg(dst)`
                                // Insn shape
                                // `emit_residual_call_shape` would have
                                // produced (no leading `ListI` because
                                // `args_i` is empty for all-Ref call_args).
                                // CALL has no frontend HLOp with the same
                                // shape (the graph carries `simple_call`
                                // pre-rtype HLOp recorded by
                                // `emit_frontend_simple_call`); the matching
                                // graph dual-write below is retained, feeding
                                // the eventual `flatten_graph(graph, regallocs)`
                                // path.  Helper hardcodes
                                // `CallFlavor::MayForce` matching the
                                // production source at codewriter.rs:2175
                                // and 2238-2245 (every `call_fn_N` is
                                // bound MayForce).
                                // Map (FlowValue, register) → Operand: Constant
                                // null sentinels lower to ConstRef(0) (matches
                                // canonical's `lower_simple_call_hlop_to_insn`
                                // routing Constant args through `lower_constant`,
                                // which routes `(None, Ref)` to `ConstRef(0)` per
                                // `flatten_constant_operand`).  Variable values
                                // emit Register operands as before.  Walker prior
                                // to this lowering always emitted Registers and
                                // produced byte-divergence at the CALL site for
                                // graphs whose `simple_call` carried a Constant
                                // null arg (e.g. list_reverse, list_pop_append
                                // after LOAD_ATTR(method) pushes the sentinel).
                                let to_operand =
                                |value: &super::flow::FlowValue, reg: u16| -> super::flatten::Operand {
                                    if let super::flow::FlowValue::Constant(c) = value {
                                        if matches!(c.value, super::flow::ConstantValue::None) {
                                            return super::flatten::Operand::ConstRef(0);
                                        }
                                    }
                                    super::flatten::Operand::Register(
                                        super::flatten::Register::new(
                                            super::flatten::Kind::Ref,
                                            reg,
                                        ),
                                    )
                                };
                                let mut ref_operands: Vec<super::flatten::Operand> =
                                    Vec::with_capacity(2 + arg_regs.len());
                                ref_operands.push(super::flatten::Operand::Register(
                                    super::flatten::Register::new(
                                        super::flatten::Kind::Ref,
                                        portal_frame_reg,
                                    ),
                                ));
                                ref_operands.push(to_operand(&callable_value, callable_reg));
                                for (value, reg) in arg_values.iter().zip(arg_regs.iter()) {
                                    ref_operands.push(to_operand(value, *reg));
                                }
                                push_walker_emit(
                                    &current_block,
                                    super::flatten::build_residual_call_r_r_insn_from_operands(
                                        fn_idx,
                                        ref_operands,
                                        super::flatten::CallFlavor::MayForce,
                                        super::flatten::Register::new(
                                            super::flatten::Kind::Ref,
                                            stack_base + current_depth,
                                        ),
                                    ),
                                );
                                // Graph-side residual_call_r_r dual-write
                                // retired — replaced by canonical driver's
                                // `lower_simple_call_hlop_to_insn` arm
                                // which lowers the `simple_call` HLOp
                                // recorded above into the same
                                // `residual_call_r_r` Insn shape.  Matches
                                // upstream RPython where `simple_call` IS the
                                // graph form and `residual_call_r_r` only
                                // appears post-flatten.
                                let _ = (callable_value, graph_arg_values_rev);
                            }
                            push_and_bump!(call_result_value, py_pc);
                        }

                        // Python 3.13: ToBool converts TOS to bool before branch.
                        // No-op in JitCode: the value is already truthy/falsy and
                        // the following PopJumpIfFalse guards on it.
                        Instruction::ToBool => {}

                        // RPython bhimpl_int_neg: -obj via binary_op(0, obj, NB_SUBTRACT)
                        Instruction::UnaryNegative => {
                            let operand_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let operand_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let operand_value_for_dual = operand_value.clone();
                            let negated = emit_frontend_neg(
                                &mut graph,
                                &current_block.block(),
                                operand_value,
                                py_pc as i64,
                            );
                            let subtract_tag =
                                binary_op_tag(pyre_interpreter::bytecode::BinaryOperator::Subtract)
                                    .expect("subtract must have a jit binary-op tag");
                            let scratch_zero = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                            // box_int_fn factor
                            // refactor (UnaryNegative site).  See
                            // LoadSmallInt site for the shared rationale.
                            push_walker_emit(
                                &current_block,
                                super::flatten::build_box_int_fn_residual_call_ir_r_insn(
                                    box_int_fn_idx,
                                    0,
                                    scratch_zero,
                                ),
                            );
                            // UnaryNegative
                            // binary_op_fn factor refactor.  The prior
                            // `emit_residual_call(binary_op_fn_idx, ...)`
                            // is replaced by a single direct push of the
                            // existing `build_binary_op_residual_call_ir_r_insn`
                            // helper — no new
                            // flatten.rs code is needed because the shape
                            // matches BINARY_OP exactly: `(zero:Ref,
                            // operand:Ref, sub_tag:Int) → Ref` MayForce.
                            // Graph dual-write below is unchanged.
                            push_walker_emit(
                                &current_block,
                                super::flatten::build_binary_op_residual_call_ir_r_insn(
                                    binary_op_fn_idx,
                                    subtract_tag,
                                    scratch_zero,
                                    operand_reg,
                                    stack_base + current_depth,
                                ),
                            );
                            // Walker-orthodoxy: graph dual-writes for
                            // the UnaryNegative `box_int_fn(0)` + `binary_op_fn(
                            // zero, operand, subtract_tag)` pair fire
                            // unconditionally (no `is_portal` gating).  Both
                            // residual_calls operate on values that are
                            // available regardless of portal status —
                            // `operand_value_for_dual` is the cloned popped
                            // FlowValue (always present), and the `0:Int`
                            // constant has no source dependency.  Mirrors
                            // upstream `jtransform.py rewrite_op_int_neg`
                            // (`0 - x`) which records both ops on the graph
                            // for EVERY function, not just the portal.
                            let zero_graph_var = residual_call!(
                                box_int_fn_idx,
                                CallFlavor::Plain,
                                vec![super::flow::Constant::signed(0).into()],
                                vec![],
                                vec![],
                                vec![Kind::Int],
                                ResKind::Ref,
                                py_pc as i64,
                            );
                            pin!(zero_graph_var, scratch_zero);
                            pin!(Some(negated), stack_base + current_depth);
                            if let Some(zero_var) = &zero_graph_var {
                                let binary_result = residual_call!(
                                    binary_op_fn_idx,
                                    CallFlavor::MayForce,
                                    vec![super::flow::Constant::signed(subtract_tag as i64).into()],
                                    vec![zero_var.clone().into(), operand_value_for_dual.into()],
                                    vec![],
                                    vec![Kind::Ref, Kind::Ref, Kind::Int],
                                    ResKind::Ref,
                                    py_pc as i64,
                                );
                                pin!(binary_result, stack_base + current_depth);
                            }
                            push_and_bump!(negated.into(), py_pc);
                        }

                        // JumpBackwardNoInterrupt reuses `backward_jump_target`:
                        // the encoding differs from JumpBackward (no skip_caches
                        // on the next-PC base) but the helper routes each variant
                        // to its correct arithmetic so pre-scan and emit stay in
                        // lockstep.  interp_jit.py:103 + jtransform.py:1714.
                        instr @ Instruction::JumpBackwardNoInterrupt { .. } => {
                            if let Some(target_py_pc) =
                                backward_jump_target(code, py_pc, instr, op_arg)
                            {
                                if target_py_pc < num_instrs {
                                    emit_goto!(target_py_pc);
                                }
                            }
                        }

                        // flowcontext.py:1168 BUILD_LIST -> `op.newlist(*items).eval(self)`
                        // consumes all `itemcount` items and returns the list.
                        // pyre's `build_list_fn` helper accepts the small fixed
                        // arities this bytecode lowering can pass directly; larger
                        // lists fall back to `abort_permanent` + interpreter —
                        // silently dropping items was the prior behaviour and would
                        // have produced wrong list contents at runtime.
                        Instruction::BuildList { count } => {
                            let argc = count.get(op_arg) as usize;
                            if argc > 3 {
                                for _ in 0..argc {
                                    let _ = emit_popvalue_ref!(current_depth, py_pc);
                                    let _ = pop_ref_or_fresh(&mut current_state, &mut graph);
                                }
                                emit_abort_permanent!();
                                push_fresh_ref(&mut current_state, &mut graph);
                                current_depth += 1;
                                emit_vsd!(current_depth, py_pc);
                                continue;
                            }
                            let mut arg_regs_rev: Vec<u16> = Vec::with_capacity(argc);
                            let mut item_values_rev = Vec::with_capacity(argc);
                            for _ in 0..argc {
                                let item_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let item_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                if let super::flow::FlowValue::Variable(v) = &item_value {
                                    pin!(Some(*v), item_reg);
                                }
                                arg_regs_rev.push(item_reg);
                                item_values_rev.push(item_value);
                            }
                            let arg_regs: Vec<u16> = arg_regs_rev.iter().rev().copied().collect();
                            // build_list_fn(argc, item0, item1, item2) → list. The C ABI is
                            // `extern "C" fn(i64, i64, i64, i64)`; the helper dispatches
                            // internally by `argc`, so unused item slots may be any
                            // bit pattern. Encode unused slots as `ConstInt(0)` —
                            // routed through the int constants pool, matches
                            // upstream `make_three_lists` Constant admit
                            // (jtransform.py:437-445). Used item slots stay
                            // Ref-typed so they read from `registers_r`.
                            let result_value = emit_frontend_newlist(
                                &mut graph,
                                &current_block.block(),
                                item_values_rev.into_iter().rev().collect(),
                                py_pc as i64,
                            );
                            pin!(Some(result_value), stack_base + current_depth);
                            // BuildList factor
                            // refactor.  The prior `emit_residual_call(
                            // build_list_fn_idx, ...)` is replaced by a
                            // single direct push of
                            // `build_build_list_fn_residual_call_ir_r_insn`.
                            // The helper internally pads unused item slots
                            // with `ConstInt(0)` matching the prior inline
                            // dummy logic, and produces the same `residual_
                            // call_ir_r(ConstInt(fn_idx), ListI([argc, ...
                            // dummies]), ListR([... regs]), Descr)` shape
                            // `emit_residual_call_shape` would have
                            // produced.  No graph dual-write exists for
                            // build_list_fn (only the `newlist` frontend
                            // HLOp recorded above).  Helper hardcodes
                            // `CallFlavor::Plain` matching the production
                            // source at codewriter.rs:2226.
                            push_walker_emit(
                                &current_block,
                                super::flatten::build_build_list_fn_residual_call_ir_r_insn(
                                    build_list_fn_idx,
                                    argc,
                                    &arg_regs,
                                    stack_base + current_depth,
                                ),
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // pyopcode.py:1463 BUILD_SLICE:
                        //   if numargs == 3: w_step = popvalue()
                        //   elif numargs == 2: w_step = space.w_None
                        //   w_end = popvalue(); w_start = popvalue()
                        //   pushvalue(space.newslice(w_start, w_end, w_step))
                        Instruction::BuildSlice { argc } => {
                            use pyre_interpreter::bytecode::BuildSliceArgCount;
                            let argc = argc.get(op_arg);
                            let raw_argc = match argc {
                                BuildSliceArgCount::Two => 2usize,
                                BuildSliceArgCount::Three => 3usize,
                            };
                            let step_info = if raw_argc == 3 {
                                let reg = emit_popvalue_ref!(current_depth, py_pc);
                                let step_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                Some((reg, step_value))
                            } else {
                                None
                            };
                            let stop_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let stop_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let start_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let start_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let step_reg = step_info.as_ref().map(|(reg, _)| *reg);
                            let result_value = emit_frontend_buildslice_shadow_graph(
                                &mut graph,
                                &current_block.block(),
                                argc,
                                start_value,
                                stop_value,
                                step_info.map(|(_, value)| value),
                                py_pc as i64,
                            );
                            pin!(Some(result_value), stack_base + current_depth);
                            push_walker_emit(
                                &current_block,
                                super::flatten::build_build_slice_fn_residual_call_ir_r_insn(
                                    build_slice_fn_idx,
                                    raw_argc,
                                    start_reg,
                                    stop_reg,
                                    step_reg,
                                    stack_base + current_depth,
                                ),
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // pyopcode.py:690 RAISE_VARARGS: argc=0 reraise,
                        // argc=1 normalize+raise, argc=2 pop cause + normalize+raise.
                        // `normalize_raise_varargs_fn` residual performs the
                        // exception_is_valid_obj_as_class_w instantiation and
                        // set_cause attachment at runtime so the shadow graph's
                        // exception edge always carries a normalized instance.
                        Instruction::RaiseVarargs { argc } => {
                            let n = argc.get(op_arg) as i64;
                            if n >= 1 {
                                // argc==2: pop cause operand (top of stack) first.
                                // Capture the cause FlowValue alongside the
                                // Operand so the graph dual-write below can
                                // record the upstream-orthodox call shape.
                                let (cause, cause_fv): (
                                    super::flatten::Operand,
                                    super::flow::FlowValue,
                                ) = if n >= 2 {
                                    let cause_fv = pop_ref_or_fresh(&mut current_state, &mut graph);
                                    let cause_reg = emit_popvalue_ref!(current_depth, py_pc);
                                    (
                                        super::flatten::Operand::Register(
                                            super::flatten::Register::new(
                                                super::flatten::Kind::Ref,
                                                cause_reg,
                                            ),
                                        ),
                                        cause_fv,
                                    )
                                } else {
                                    // null_ref_reg retirement: PY_NULL flows directly through
                                    // the residual_call's ListOfKind(Ref) as a
                                    // raw constant — make_three_lists
                                    // (jtransform.py:437-445) admits Constant in
                                    // any slot, and the assembler's
                                    // dispatch_residual_call routes ConstRef
                                    // through the ref constants pool
                                    // (assembler.rs:1709-1724
                                    // expect_ref_reg_or_pool).
                                    // `Constant::none()` lowers to
                                    // `Operand::ConstRef(0)` per
                                    // `flatten_constant_operand`'s
                                    // `(ConstantValue::None, Some(Kind::Ref))`
                                    // arm, matching `PY_NULL as i64 = 0`
                                    // (std::ptr::null_mut()) for the inline
                                    // emit above.
                                    (
                                        super::flatten::Operand::ConstRef(
                                            pyre_object::PY_NULL as i64,
                                        ),
                                        super::flow::Constant::none().into(),
                                    )
                                };
                                // Drop the pre-normalization exception operand from
                                // the shadow stack. The residual call below may
                                // rewrite `raise SomeExcClass` into a fresh
                                // instance, so the exception edge must carry a
                                // NEW FlowValue representing the normalized result.
                                let exc_fv = pop_ref_or_fresh(&mut current_state, &mut graph);
                                let exc_reg = emit_popvalue_ref!(current_depth, py_pc);
                                // pyopcode.py:711 `exception_is_valid_obj_as_class_w`
                                // normalization + `set_cause` attachment.  Call ABI
                                // reads inputs before writing the result; the
                                // popped stack slot is the natural result
                                // destination, so the call writes the normalized
                                // exception directly into `exc_reg` and
                                // `emit_raise!` reads the same register as its
                                // source. Pattern matches the
                                // Call/UnaryNegative/CheckExcMatch input-side.
                                // RaiseVarargs
                                // normalize_raise_varargs_fn factor
                                // refactor.  The prior `emit_residual_call(
                                // normalize_raise_varargs_fn_idx, ...)` is
                                // replaced by a single direct push of
                                // `build_normalize_raise_varargs_fn_residual_call_r_r_insn`,
                                // which produces the same `residual_call_r_r(
                                // ConstInt(fn_idx), ListR([Reg(frame),
                                // Reg(exc), cause]), Descr) → Reg(exc)`
                                // Insn shape.
                                // Helper hardcodes `CallFlavor::MayForce`
                                // matching the production source at
                                // codewriter.rs:2235.  The polymorphic
                                // `cause` Operand (Reg or ConstRef) is
                                // built inline above.
                                push_walker_emit(&current_block,
                                super::flatten::build_normalize_raise_varargs_fn_residual_call_r_r_insn(
                                    normalize_raise_varargs_fn_idx,
                                    portal_frame_reg,
                                    exc_reg,
                                    cause,
                                    exc_reg,
                                ),
                            );
                                // Graph-side `residual_call_r_r` dual-write so the
                                // canonical `flatten_graph` driver sees the same
                                // op via passthrough.  `normalize_raise_varargs_fn`
                                // takes `(frame:Ref, exc:Ref, cause:Ref) → Ref` MayForce.
                                let normalized_var = residual_call!(
                                    normalize_raise_varargs_fn_idx,
                                    CallFlavor::MayForce,
                                    vec![],
                                    vec![frame_var.into(), exc_fv.into(), cause_fv.into()],
                                    vec![],
                                    vec![Kind::Ref, Kind::Ref, Kind::Ref],
                                    ResKind::Ref,
                                    py_pc as i64,
                                );
                                pin!(normalized_var, exc_reg);
                                let normalized_exc_fv = normalized_var
                                    .map(super::flow::FlowValue::from)
                                    .unwrap_or_else(|| fresh_ref_value(&mut graph));
                                // RAISE_VARARGS argc>=1: explicit
                                // `raise X` source form. When inside
                                // a try/except range, `catch_for_pc`
                                // is consulted to emit
                                // `catch_exception/L` adjacent to
                                // `raise/r`.
                                emit_raise!(exc_reg, normalized_exc_fv, py_pc as i64, true);
                            } else {
                                // reraise: re-raise exception_last_value
                                emit_reraise!();
                            }
                        }

                        Instruction::PushExcInfo => {
                            // eval.rs:1220-1229 / pyopcode.py:786 parity:
                            //   exc  = pop()
                            //   prev = CURRENT_EXCEPTION
                            //   CURRENT_EXCEPTION = exc
                            //   push(prev)
                            //   push(exc)
                            //
                            // Emit two residual helper calls so the traced code
                            // reads/writes the same per-thread exception slot as
                            // the interpreter; pushing `None` for `prev` breaks
                            // nested exception state (pyopcode.py:786 saves the
                            // previous sys_exc_info so `POP_EXCEPT` can restore it).
                            let exc_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let exc_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // pyopcode.py:786 keeps `exc` in a local after
                            // `popvalue()`.  Mirror that with a scratch register:
                            // the following `push(prev)` writes to the popped
                            // stack slot, so reusing `exc_reg` for the trailing
                            // `push(exc)` would read back `prev` instead of the
                            // caught exception.
                            let scratch_exc = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                            emit_ref_copy(&current_block, scratch_exc, exc_reg);
                            let scratch_prev = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                            // get_current_exception / set_current_exception are TLS read/write —
                            // EF_CANNOT_RAISE per `effectinfo.py:19` (matching call.py:296
                            // getcalldescr's analyzer outcome for non-raising helpers).
                            // PushExcInfo
                            // get/set_current_exception factor refactor.
                            // Both helpers are PlainCannotRaise (TLS
                            // read/write only).  `get_current_exception`
                            // is 0-arg `() → Ref`; `set_current_exception`
                            // is 1-arg `(exc:Ref) → Void`.  Graph
                            // dual-writes below remain unchanged.
                            push_walker_emit(
                            &current_block,
                            super::flatten::build_get_current_exception_fn_residual_call_r_r_insn(
                                get_current_exception_fn_idx,
                                scratch_prev,
                            ),
                        );
                            push_walker_emit(
                            &current_block,
                            super::flatten::build_set_current_exception_fn_residual_call_r_v_insn(
                                set_current_exception_fn_idx,
                                scratch_exc,
                            ),
                        );
                            // graph dual-writes for
                            // both PushExcInfo emits.  get_current_exception
                            // takes no args (shape residual_call_r_r with empty
                            // ListR); set_current_exception is `(exc:Ref)→Void`
                            // (shape residual_call_r_v).
                            // Walker-orthodoxy: TLS-only helpers,
                            // no frame_var threading.  Match helper
                            // bind-site flavors at codewriter.rs:2207-2217
                            // — both current-exception helpers are TLS
                            // read/write only and statically prove "no GC
                            // heap touched", binding `PlainCannotRaiseNoHeap`
                            // for the analyzer-equivalent `EF_CANNOT_RAISE
                            // + empty raw frozensets + can_collect=false`
                            // shape (`effectinfo.py:281-283`).
                            let prev_var = residual_call!(
                                get_current_exception_fn_idx,
                                CallFlavor::PlainCannotRaiseNoHeap,
                                vec![],
                                vec![],
                                vec![],
                                vec![],
                                ResKind::Ref,
                                py_pc as i64,
                            );
                            pin!(prev_var, scratch_prev);
                            let _ = residual_call!(
                                set_current_exception_fn_idx,
                                CallFlavor::PlainCannotRaiseNoHeap,
                                vec![],
                                vec![exc_value.clone()],
                                vec![],
                                vec![Kind::Ref],
                                ResKind::Void,
                                py_pc as i64,
                            );
                            // `emit_pushvalue_ref!` writes each value into its
                            // positional stack slot (`stack_base + depth`) but
                            // does not pin the `FlowValue` Variable to that slot,
                            // so graph regalloc colours the pushed value
                            // arbitrarily.  For the exception value that flows
                            // from PUSH_EXC_INFO into the handler's CHECK_EXC_MATCH
                            // across a block boundary, the colour/slot mismatch
                            // makes `walker_post_walk_insert_renamings` emit a
                            // parallel copy that clobbers the slot the exc
                            // actually lives in (raise_catch: exc lost before
                            // CHECK_EXC_MATCH -> NULL operand).  Pin both pushed
                            // values to the slot the walker wrote, mirroring the
                            // catch-landing `last_exc_value` pin, so
                            // `get_color` agrees with the runtime register.
                            let prev_slot = stack_base + current_depth;
                            let prev_value = fresh_ref_value(&mut graph);
                            current_state.stack.push(prev_value.clone());
                            emit_pushvalue_ref!(
                                current_depth,
                                scratch_prev,
                                prev_value.clone(),
                                py_pc
                            );
                            pin!(prev_value.as_variable(), prev_slot);
                            let exc_handler_slot = stack_base + current_depth;
                            current_state.stack.push(exc_value.clone());
                            emit_pushvalue_ref!(
                                current_depth,
                                scratch_exc,
                                exc_value.clone(),
                                py_pc
                            );
                            pin!(exc_value.as_variable(), exc_handler_slot);
                        }

                        Instruction::CheckExcMatch => {
                            // CPython 3.14: pop match type, peek exception, push
                            // bool result. Net stack effect is zero.
                            //
                            // Runtime check = `isinstance(exc, match_type)` via
                            // compare_fn with ISINSTANCE_OP (tag 10). No
                            // flowspace-level shortcut — upstream
                            // flowcontext.py:591 folds `cmp_exc_match` at analysis
                            // time, but pyre's shadow graph cannot observe the
                            // runtime exception type; the residual helper owns
                            // the check.
                            let match_type_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let match_type_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let exc_reg = stack_base + current_depth - 1;
                            // Peek (don't pop) the exception value for the graph
                            // dual-write — net stack effect is zero (pop match
                            // type, peek exception, push bool result).
                            let exc_value = current_state
                                .stack
                                .last()
                                .cloned()
                                .unwrap_or_else(|| fresh_ref_value(&mut graph).into());
                            let scratch_match = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                            // CheckExcMatch
                            // compare_fn factor refactor.  `compare_fn` is
                            // the same helper used by COMPARE_OP — the
                            // call shape `(exc:Ref, match_type:Ref, op_val:
                            // Int) → Ref` MayForce is identical to the
                            // BINARY_OP/COMPARE_OP `_ir_r` family.  CheckExcMatch
                            // passes `op_val = 10` (ISINSTANCE_OP from
                            // `runtime_ops::compare_op_tag`'s table) directly
                            // rather than mapping through `compare_op_tag`.
                            // Reusing `build_compare_op_residual_call_ir_r_insn`
                            // matches the semantic shape — the helper
                            // accepts any `op_val: i64` and the dual-write
                            // already records the same `compare_fn(...,
                            // ISINSTANCE_OP:Int) → Ref` `residual_call_ir_r`
                            // shape (codewriter.rs:6219-6232).
                            push_walker_emit(
                                &current_block,
                                super::flatten::build_compare_op_residual_call_ir_r_insn(
                                    compare_fn_idx,
                                    10,
                                    exc_reg,
                                    match_type_reg,
                                    scratch_match,
                                ),
                            );
                            // Walker-orthodoxy: compare_fn(exc,
                            // match_type, ISINSTANCE_OP:Int) → Ref shape
                            // residual_call_ir_r.  No frame_var threading.
                            let cmp_result = residual_call!(
                                compare_fn_idx,
                                CallFlavor::MayForce,
                                vec![super::flow::Constant::signed(10).into()],
                                vec![exc_value, match_type_value],
                                vec![],
                                vec![Kind::Ref, Kind::Ref, Kind::Int],
                                ResKind::Ref,
                                py_pc as i64,
                            );
                            pin!(cmp_result, scratch_match);
                            let result_value = fresh_ref_value(&mut graph);
                            current_state.stack.push(result_value.clone());
                            emit_pushvalue_ref!(current_depth, scratch_match, result_value, py_pc);
                        }

                        Instruction::PopExcept => {
                            // eval.rs:1243-1249 / pyopcode.py:778 parity:
                            //   prev = pop()
                            //   CURRENT_EXCEPTION = prev
                            //
                            // Previously the arm just popped and left TLS stale,
                            // which silently broke nested `except` blocks: after
                            // `POP_EXCEPT` the outer handler's exception must be
                            // reinstated as the "current" one so a bare `raise`
                            // re-propagates it.
                            let prev_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let prev_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // set_current_exception is a TLS write — EF_CANNOT_RAISE.
                            // PopExcept
                            // set_current_exception factor refactor.
                            // PlainCannotRaise TLS write `(prev:Ref) → Void`.
                            // Graph dual-write below unchanged.
                            push_walker_emit(
                            &current_block,
                            super::flatten::build_set_current_exception_fn_residual_call_r_v_insn(
                                set_current_exception_fn_idx,
                                prev_reg,
                            ),
                        );
                            // Walker-orthodoxy: set_current_exception
                            // `(prev:Ref)→Void` shape residual_call_r_v.
                            // TLS write, no GC heap touched,
                            // `PlainCannotRaiseNoHeap` (`effectinfo.py:281-283`
                            // analyzer output).
                            let _ = residual_call!(
                                set_current_exception_fn_idx,
                                CallFlavor::PlainCannotRaiseNoHeap,
                                vec![],
                                vec![prev_value],
                                vec![],
                                vec![Kind::Ref],
                                ResKind::Void,
                                py_pc as i64,
                            );
                            current_state.last_exception = None;
                        }

                        Instruction::Reraise { depth } => {
                            // CPython 3.x RERAISE:
                            //   oparg=1: TOS is lasti (int), TOS-1 is exception.
                            //            Pop lasti, then pop exception and raise.
                            //            Cleanup-handler shape generated by
                            //            `Python/compile.c` for try/except.
                            //   oparg=0: TOS is exception. Pop and raise.
                            // Both shapes end with the exception value popped
                            // and re-raised via the exception edge.  Pyre's
                            // `emit_raise!` emits the `raise/r` insn and a
                            // graph link to `exceptblock` so the block has a
                            // proper terminator (mirrors `flatten.py:189
                            // make_exception_link` exception-edge shape).
                            // `emit_popvalue_ref!` decrements `current_depth`
                            // internally, so the explicit saturating_sub the
                            // main-branch version added is redundant here.
                            let oparg_count = depth.get(op_arg) as usize;
                            if oparg_count >= 1 {
                                // Discard lasti.
                                let _ = emit_popvalue_ref!(current_depth, py_pc);
                                let _ = current_state.stack.pop();
                            }
                            let exc_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let exc_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            // RERAISE: pyre-only deviation from RPython
                            // (which has no RERAISE bytecode — its
                            // `Reraise.nomoreblocks` calls reraise
                            // directly into the exception link).
                            // Suppress catch_exception adjacency:
                            // RERAISE's source FrameState has had
                            // POP_EXCEPT mutate the stack, which
                            // makes the catch-landing union path
                            // (`attach_catch_exception_edge` →
                            // `getoutputargs_with_positions`)
                            // mismatched against the explicit-raise
                            // shape that earlier raise sites already
                            // populated into the same landing.
                            emit_raise!(exc_reg, exc_value, py_pc as i64, false);
                        }

                        Instruction::WithExceptStart => {
                            // CPython 3.14: `WITH_EXCEPT_START` leaves the existing
                            // stack entries intact and pushes the exit-function
                            // result on top. Preserve the net `+1` stack effect in
                            // the shadow graph and fall back to the interpreter for
                            // the actual helper call semantics.
                            emit_abort_permanent!();
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_vsd!(current_depth, py_pc);
                        }

                        Instruction::Copy { i } => {
                            let d = i.get(op_arg) as usize;
                            if d == 1 {
                                let duplicated =
                                    duplicate_shadow_tos(&mut graph, &mut current_state);
                                emit_pushvalue_ref!(
                                    current_depth,
                                    stack_base + current_depth - 1,
                                    duplicated,
                                    py_pc
                                );
                            } else {
                                // CPython COPY n (n>1): pushes PEEK(n) where
                                // PEEK(1)=TOS.  Source value is
                                // `stack[len-n]`; source register is
                                // `stack_base + current_depth - n`.  Generated
                                // by `Python/compile.c` for try-except cleanup
                                // handlers (L9 reraise shape) to duplicate the
                                // pushed exception before POP_EXCEPT clears
                                // the saved state.  `emit_pushvalue_ref!`
                                // increments `current_depth` internally;
                                // explicit `current_depth += 1` is unnecessary.
                                let stack_idx = current_state.stack.len().saturating_sub(d);
                                let src_value = current_state
                                    .stack
                                    .get(stack_idx)
                                    .cloned()
                                    .unwrap_or_else(|| fresh_ref_value(&mut graph));
                                current_state.stack.push(src_value.clone());
                                let src_reg = stack_base + current_depth.saturating_sub(d as u16);
                                emit_pushvalue_ref!(current_depth, src_reg, src_value, py_pc);
                            }
                        }

                        // Stack-effect-aware abort_permanent for unsupported ops.
                        // current_depth must track interpreter parity so that
                        // subsequent CALL handlers don't underflow.
                        Instruction::LoadName { .. } => {
                            // flowcontext.py:859 LOAD_NAME = LOAD_GLOBAL.
                            // RPython resolves the name to a Constant during flow
                            // analysis; pyre cannot fold module namespace lookups at
                            // codewriter time, so do not invent a graph op here.
                            // The abort_permanent above means the loaded value is
                            // never observed at runtime, but downstream ops in the
                            // same block (e.g. the bool() arg of POP_JUMP_IF_FALSE)
                            // pop from current_state.stack and would otherwise
                            // hit `fresh_ref_value` — producing an orphan Variable
                            // with no graph SpaceOp producer that breaks canonical
                            // SSARepr build's make_dependencies (regalloc.py:26-77
                            // adds depgraph nodes only for op.result + inputargs).
                            // Push `Constant::none()` so downstream pops resolve
                            // to a constant, not a fresh Variable.
                            current_state
                                .stack
                                .push(super::flow::Constant::none().into());
                            emit_abort_permanent!();
                            current_depth += 1;
                            emit_vsd!(current_depth, py_pc);
                        }
                        Instruction::StoreName { .. } | Instruction::StoreGlobal { .. } => {
                            // flowcontext.py marks STORE_NAME unsupported, but the
                            // stack effect still consumes one value. STORE_GLOBAL
                            // follows the same shape in flowcontext.py:884-890.
                            emit_abort_permanent!();
                            pop_and_decr_depth(&mut current_state, &mut current_depth);
                            emit_vsd!(current_depth, py_pc);
                        }
                        Instruction::MakeFunction { .. } => {
                            // Pops code object (TOS), pushes function. Net: 0.
                            // Replace shadow value so SET_FUNCTION_ATTRIBUTE sees func.
                            // RustPython: (1 pushed, 1 popped).
                            let _ = current_state.stack.pop();
                            push_fresh_ref(&mut current_state, &mut graph);
                            emit_abort_permanent!();
                        }
                        Instruction::StoreAttr { namei } => {
                            let name_idx = namei.get(op_arg) as usize;
                            let attr_name =
                                super::flow::Constant::string(code.names[name_idx].as_str());
                            current_depth = current_depth.saturating_sub(1);
                            emit_vsd!(current_depth, py_pc);
                            let obj_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            current_depth = current_depth.saturating_sub(1);
                            emit_vsd!(current_depth, py_pc);
                            let stored_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            emit_frontend_setattr(
                                &mut graph,
                                &current_block.block(),
                                obj_value,
                                attr_name.into(),
                                stored_value,
                                py_pc as i64,
                            );
                            emit_abort_permanent!();
                        }
                        Instruction::LoadAttr { namei } => {
                            // PyPy assemble.py gives LOAD_ATTR a net-0 stack effect.
                            // pyre's CPython-3.13 method form pushes an extra
                            // null/self sentinel, so keep current_depth in sync.
                            let attr = namei.get(op_arg);
                            let name_idx = attr.name_idx() as usize;
                            let attr_name =
                                super::flow::Constant::string(code.names[name_idx].as_str());
                            let obj_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let result_value = emit_frontend_getattr(
                                &mut graph,
                                &current_block.block(),
                                obj_value,
                                attr_name.into(),
                                py_pc as i64,
                            );
                            emit_abort_permanent!();
                            current_state.stack.push(result_value.into());
                            if attr.is_method() {
                                push_and_bump!(null_stack_sentinel(), py_pc);
                            }
                        }

                        // CPython 3.13 superinstruction: STORE_FAST_STORE_FAST.
                        // jtransform.py:1898 — each local write → setarrayitem_vable_r
                        // in portal, ref_copy in non-portal. Mirrors plain StoreFast.
                        Instruction::StoreFastStoreFast { var_nums } => {
                            let pair = var_nums.get(op_arg);
                            let reg_a = u32::from(pair.idx_1()) as u16;
                            let reg_b = u32::from(pair.idx_2()) as u16;
                            for reg in [reg_a, reg_b] {
                                let stored_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let stored = pop_ref_or_fresh(&mut current_state, &mut graph);
                                if is_portal {
                                    // Graph-side dual-write — same shape as
                                    // the StoreFast handler.  SSA emission
                                    // is delegated to
                                    // `emit_store_local_with_mirror!` below.
                                    let local_slot = local_to_vable_slot(reg as usize) as i64;
                                    let v_idx: super::flow::FlowValue =
                                        super::flow::Constant::signed(local_slot).into();
                                    record_graph_op(
                                        &current_block.block(),
                                        "setarrayitem_vable_r",
                                        vable_setarrayitem_ref_graph_args(
                                            frame_var.into(),
                                            v_idx.into(),
                                            stored.clone().into(),
                                        ),
                                        None,
                                        -1,
                                    );
                                }
                                emit_store_local_with_mirror!(reg, stored_reg);
                                current_state.store_local_value(reg as usize, stored);
                            }
                        }

                        // CPython 3.13 UNPACK_SEQUENCE: pop 1 (seq), push `count`.
                        // Emit abort_permanent (no getitem helper yet) but
                        // adjust current_depth so subsequent instructions don't
                        // underflow.
                        Instruction::UnpackSequence { count } => {
                            let n = count.get(op_arg) as usize;
                            // Pop the sequence; preserve it in a stable scratch reg
                            // because the item pushes below reuse the stack slots.
                            let seq_reg = emit_popvalue_ref!(current_depth, py_pc);
                            let _seq_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                            let seq_scratch = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                            emit_ref_copy(&current_block, seq_scratch, seq_reg);
                            // unpack_sequence_fn(n, seq) → tuple of exactly n items;
                            // raises ValueError/TypeError on length mismatch or a
                            // non-sequence, matching opcode_unpack_sequence.
                            let tuple_scratch = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                            push_walker_emit(
                                &current_block,
                                super::flatten::build_one_int_one_ref_fn_residual_call_ir_r_insn(
                                    unpack_sequence_fn_idx,
                                    n as i64,
                                    seq_scratch,
                                    tuple_scratch,
                                ),
                            );
                            // Push the items in reverse so the stack top is item[0]
                            // (opcode_unpack_sequence pushes `items.into_iter().rev()`).
                            for k in (0..n).rev() {
                                let item_dst = stack_base + current_depth;
                                push_walker_emit(
                                    &current_block,
                                    super::flatten::build_one_int_one_ref_fn_residual_call_ir_r_insn(
                                        unpack_item_fn_idx,
                                        k as i64,
                                        tuple_scratch,
                                        item_dst,
                                    ),
                                );
                                let item_value = fresh_ref_value(&mut graph);
                                if let super::flow::FlowValue::Variable(v) = &item_value {
                                    pin!(Some(*v), item_dst);
                                }
                                push_and_bump!(item_value, py_pc);
                            }
                        }

                        // CPython 3.13 iterator protocol — emit abort_permanent
                        // with correct depth tracking so subsequent instructions
                        // don't underflow.
                        Instruction::GetIter => {
                            // Pop iterable, push iterator. Net: 0. Replace shadow value.
                            // pypy/interpreter/pyopcode.py:1281.
                            let _ = current_state.stack.pop();
                            push_fresh_ref(&mut current_state, &mut graph);
                            emit_abort_permanent!();
                        }

                        Instruction::ForIter { .. } => {
                            // push next item: net +1
                            emit_abort_permanent!();
                            current_depth += 1;
                            emit_vsd!(current_depth, py_pc);
                        }

                        Instruction::EndFor => {
                            // Pyre's end_for() is a no-op (pyopcode.rs:999). Net: 0.
                            // The actual pop is handled by the subsequent PopIter (-1).
                            emit_abort_permanent!();
                        }

                        Instruction::PopIter => {
                            // pop iterator: net -1
                            current_depth = current_depth.saturating_sub(1);
                            emit_vsd!(current_depth, py_pc);
                        }

                        // BinarySlice: obj[start:stop] — pops 3 (stop, start, obj), pushes 1 (result).
                        // Net stack effect: -2.
                        // pyopcode.py BINARY_SLICE / eval.rs:2857-2935.
                        Instruction::BinarySlice => {
                            for _ in 0..3 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // ContainsOp: item in container — pops 2, pushes 1 (bool).
                        // Net stack effect: -1.
                        // pyopcode.py CONTAINS_OP / eval.rs:1784-1798.
                        Instruction::ContainsOp { .. } => {
                            for _ in 0..2 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // CallKw: like Call but with extra kwnames tuple.
                        // Pops: kwnames + argc args + null_or_self + callable = argc + 3.
                        // Pushes: result. Net stack effect: -(argc + 2).
                        // pyopcode.py CALL_FUNCTION_KW / CALL_KW / eval.rs:2570-2726.
                        Instruction::CallKw { argc } => {
                            let nargs = argc.get(op_arg) as usize;
                            // Pop kwnames + nargs args + null_or_self + callable.
                            for _ in 0..nargs + 3 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // Swap: swap TOS with TOS[i]. No net stack effect.
                        // pyopcode.py SWAP / eval.rs:1029-1034.
                        Instruction::Swap { i } => {
                            let depth = i.get(op_arg) as usize;
                            let stack_len = current_state.stack.len();
                            if depth > 0 && depth <= stack_len {
                                current_state.stack.swap(stack_len - 1, stack_len - depth);
                            }
                            emit_abort_permanent!();
                        }

                        // LoadFastAndClear: push local, clear it. Net: +1.
                        // pyopcode.py LOAD_FAST_AND_CLEAR / eval.rs:2052-2058.
                        Instruction::LoadFastAndClear { var_num } => {
                            let idx = var_num.get(op_arg).as_usize();
                            let value = current_state
                                .local_value_at(idx)
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            if idx < current_state.locals_w.len() {
                                current_state.locals_w[idx] = None;
                            }
                            current_state.stack.push(value);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // ListAppend(i): peek list at stack[i], pop value. Net: -1.
                        // shared_opcode.rs opcode_list_append.
                        Instruction::ListAppend { .. } => {
                            pop_and_decr_depth(&mut current_state, &mut current_depth);
                            emit_abort_permanent!();
                        }

                        // BuildMap(count): pop 2*count key-value pairs, push dict. Net: -(2*count - 1).
                        // shared_opcode.rs opcode_build_map.
                        Instruction::BuildMap { count } => {
                            let n = count.get(op_arg) as usize;
                            for _ in 0..n * 2 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // MapAdd(i): peek dict at stack[i], pop value + key. Net: -2.
                        // eval.rs map_add.
                        Instruction::MapAdd { .. } => {
                            for _ in 0..2 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            emit_abort_permanent!();
                        }

                        // ── Remaining instructions: stack-effect-only accounting ──
                        // Each arm adjusts current_depth / current_state.stack to
                        // match the interpreter's stack effect, then aborts so the
                        // codewriter's mergeblock/pendingblocks converge.

                        // IsOp: pops 2, pushes 1 bool. Net: -1.
                        Instruction::IsOp { .. } => {
                            for _ in 0..2 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // BuildTuple(count): pops count items, pushes 1 tuple. Net: -(count-1).
                        Instruction::BuildTuple { count } => {
                            let argc = count.get(op_arg) as usize;
                            if argc > 3 {
                                for _ in 0..argc {
                                    let _ = emit_popvalue_ref!(current_depth, py_pc);
                                    let _ = pop_ref_or_fresh(&mut current_state, &mut graph);
                                }
                                emit_abort_permanent!();
                                push_fresh_ref(&mut current_state, &mut graph);
                                current_depth += 1;
                                emit_vsd!(current_depth, py_pc);
                                continue;
                            }
                            let mut arg_regs_rev: Vec<u16> = Vec::with_capacity(argc);
                            let mut item_values_rev = Vec::with_capacity(argc);
                            for _ in 0..argc {
                                let item_reg = emit_popvalue_ref!(current_depth, py_pc);
                                let item_value = pop_ref_or_fresh(&mut current_state, &mut graph);
                                if let super::flow::FlowValue::Variable(v) = &item_value {
                                    pin!(Some(*v), item_reg);
                                }
                                arg_regs_rev.push(item_reg);
                                item_values_rev.push(item_value);
                            }
                            let arg_regs: Vec<u16> = arg_regs_rev.iter().rev().copied().collect();
                            // build_tuple_fn(argc, item0, item1, item2) → tuple. Same
                            // residual_call_ir_r shape as build_list_fn (the IR builder
                            // is fn-index agnostic); unused item slots padded with
                            // ConstInt(0).
                            let result_value = emit_frontend_newtuple(
                                &mut graph,
                                &current_block.block(),
                                item_values_rev.into_iter().rev().collect(),
                                py_pc as i64,
                            );
                            pin!(Some(result_value), stack_base + current_depth);
                            push_walker_emit(
                                &current_block,
                                super::flatten::build_build_list_fn_residual_call_ir_r_insn(
                                    build_tuple_fn_idx,
                                    argc,
                                    &arg_regs,
                                    stack_base + current_depth,
                                ),
                            );
                            push_and_bump!(result_value.into(), py_pc);
                        }

                        // BuildSet(count): pops count items, pushes 1 set. Net: -(count-1).
                        Instruction::BuildSet { count } => {
                            let n = count.get(op_arg) as usize;
                            for _ in 0..n {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // BuildString(count): pops count strings, pushes 1. Net: -(count-1).
                        Instruction::BuildString { count } => {
                            let n = count.get(op_arg) as usize;
                            for _ in 0..n {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // CallFunctionEx: pops callable+null+args+kwargs_or_null (4), pushes 1. Net: -3.
                        Instruction::CallFunctionEx => {
                            for _ in 0..4 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // DeleteSubscr: pops 2 (key, obj). Net: -2.
                        Instruction::DeleteSubscr => {
                            for _ in 0..2 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            emit_abort_permanent!();
                        }

                        // DeleteAttr: pops 1 (obj). Net: -1.
                        Instruction::DeleteAttr { .. } => {
                            pop_and_decr_depth(&mut current_state, &mut current_depth);
                            emit_abort_permanent!();
                        }

                        // PopJumpIfNone / PopJumpIfNotNone: pops 1. Net: -1.
                        Instruction::PopJumpIfNone { .. }
                        | Instruction::PopJumpIfNotNone { .. } => {
                            pop_and_decr_depth(&mut current_state, &mut current_depth);
                            emit_abort_permanent!();
                        }

                        // SetAdd(i): peek set, pop value. Net: -1.
                        Instruction::SetAdd { .. } => {
                            pop_and_decr_depth(&mut current_state, &mut current_depth);
                            emit_abort_permanent!();
                        }

                        // ListExtend(i): peek list, pop iterable. Net: -1.
                        Instruction::ListExtend { .. } => {
                            pop_and_decr_depth(&mut current_state, &mut current_depth);
                            emit_abort_permanent!();
                        }

                        // SetUpdate(i): peek set, pop iterable. Net: -1.
                        Instruction::SetUpdate { .. } => {
                            pop_and_decr_depth(&mut current_state, &mut current_depth);
                            emit_abort_permanent!();
                        }

                        // DictUpdate(i) / DictMerge(i): peek dict, pop source. Net: -1.
                        Instruction::DictUpdate { .. } | Instruction::DictMerge { .. } => {
                            pop_and_decr_depth(&mut current_state, &mut current_depth);
                            emit_abort_permanent!();
                        }

                        // SetFunctionAttribute: pops func (TOS), pops attr (TOS1),
                        // pushes same func back. Net: -1. Preserve func identity.
                        // eval.rs:1907-1908: func = pop(), attr = pop().
                        Instruction::SetFunctionAttribute { .. } => {
                            let func = pop_ref_or_fresh(&mut current_state, &mut graph);
                            current_depth = current_depth.saturating_sub(1);
                            let _ = current_state.stack.pop(); // attr
                            current_depth = current_depth.saturating_sub(1);
                            current_state.stack.push(func);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // EndSend: pops result (TOS), pops iter (TOS1), pushes result back.
                        // Net: -1. Preserve result identity. eval.rs:2305-2309.
                        Instruction::EndSend => {
                            let result = pop_ref_or_fresh(&mut current_state, &mut graph);
                            current_depth = current_depth.saturating_sub(1);
                            let _ = current_state.stack.pop(); // iter
                            current_depth = current_depth.saturating_sub(1);
                            current_state.stack.push(result);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // ImportName: pops 2 (level, fromlist), pushes 1 module. Net: -1.
                        Instruction::ImportName { .. } => {
                            for _ in 0..2 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // ImportFrom: peek module, push attr. Net: +1.
                        Instruction::ImportFrom { .. } => {
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // StoreSlice: pops 4 (stop, start, obj, value). Net: -4.
                        Instruction::StoreSlice => {
                            for _ in 0..4 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            emit_abort_permanent!();
                        }

                        // FormatWithSpec: pops 2 (spec, value), pushes 1 string. Net: -1.
                        Instruction::FormatWithSpec => {
                            for _ in 0..2 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // LoadSuperAttr: pops 3 (super, cls, self).
                        // is_method=false → pushes 1 (result). Net: -2.
                        // is_method=true  → pushes 2 (func, self_or_null). Net: -1.
                        // pyopcode.rs:1926-1932, eval.rs:2331-2360.
                        Instruction::LoadSuperAttr { .. } => {
                            let is_method = (u32::from(op_arg) & 1) != 0;
                            for _ in 0..3 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            if is_method {
                                push_fresh_ref(&mut current_state, &mut graph);
                                current_depth += 1;
                            }
                            emit_abort_permanent!();
                        }

                        // UnpackEx: pops 1, pushes before+1+after items. Net: before+after.
                        Instruction::UnpackEx { counts } => {
                            let args = counts.get(op_arg);
                            let before = args.before as usize;
                            let after = args.after as usize;
                            pop_and_decr_depth(&mut current_state, &mut current_depth);
                            for _ in 0..before + 1 + after {
                                push_fresh_ref(&mut current_state, &mut graph);
                                current_depth += 1;
                            }
                            emit_abort_permanent!();
                        }

                        // BuildInterpolation: conditionally pops format_spec when (oparg & 1) != 0,
                        // then pops 2 (value, expression_str) via build_tuple, pushes 1.
                        // No spec: pops 2, pushes 1. Net: -1.
                        // With spec: pops 3, pushes 1. Net: -2.
                        // pyopcode.rs:1798-1806.
                        Instruction::BuildInterpolation { format } => {
                            let has_format_spec = (u32::from(format.get(op_arg)) & 1) != 0;
                            if has_format_spec {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            for _ in 0..2 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // BuildTemplate: pops 2, pushes 1. Net: -1.
                        Instruction::BuildTemplate => {
                            for _ in 0..2 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // CallIntrinsic1: pops 1, pushes 1 (result may differ). Net: 0.
                        Instruction::CallIntrinsic1 { .. } => {
                            let _ = current_state.stack.pop();
                            push_fresh_ref(&mut current_state, &mut graph);
                            emit_abort_permanent!();
                        }

                        // CallIntrinsic2: variant-dependent stack effect.
                        // SetFunctionTypeParams: pops type_params (TOS), leaves func. Net: -1.
                        // Other variants: general pop 2, push 1. Net: -1.
                        // pyopcode.rs:1302-1316.
                        Instruction::CallIntrinsic2 { func } => {
                            use pyre_interpreter::bytecode::IntrinsicFunction2;
                            match func.get(op_arg) {
                                IntrinsicFunction2::SetFunctionTypeParams => {
                                    let _ = current_state.stack.pop(); // type_params only
                                    current_depth = current_depth.saturating_sub(1);
                                }
                                _ => {
                                    for _ in 0..2 {
                                        pop_and_decr_depth(&mut current_state, &mut current_depth);
                                    }
                                    push_fresh_ref(&mut current_state, &mut graph);
                                    current_depth += 1;
                                }
                            }
                            emit_abort_permanent!();
                        }

                        // GetLen: peeks obj, pushes len. Net: +1.
                        Instruction::GetLen => {
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // LoadSpecial: pops 1 (obj), pushes 2 (callable, self_or_null). Net: +1.
                        // pyopcode.rs:2059 delegates to load_method; eval.rs:2365 pops 1 pushes 2.
                        Instruction::LoadSpecial { .. } => {
                            pop_and_decr_depth(&mut current_state, &mut current_depth);
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // LoadFromDictOrGlobals: pops 1 (dict), pushes 1 (result). Net: 0.
                        // Replace shadow value. eval.rs:2028.
                        Instruction::LoadFromDictOrGlobals { .. } => {
                            let _ = current_state.stack.pop();
                            push_fresh_ref(&mut current_state, &mut graph);
                            emit_abort_permanent!();
                        }

                        // LoadFromDictOrDeref: structural adaptation — CPython pops dict,
                        // pushes result (net 0). Pyre's trait default raises before stack
                        // mutation (pyopcode.rs:1247), so this models the intended CPython
                        // shape, not current pyre runtime behavior.
                        Instruction::LoadFromDictOrDeref { .. } => {
                            let _ = current_state.stack.pop();
                            push_fresh_ref(&mut current_state, &mut graph);
                            emit_abort_permanent!();
                        }

                        // Loads that push +1.
                        Instruction::LoadDeref { .. }
                        | Instruction::LoadFastCheck { .. }
                        | Instruction::LoadCommonConstant { .. }
                        | Instruction::LoadLocals
                        | Instruction::LoadBuildClass => {
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // Pops 1, pushes 1 (net 0). Replace shadow value.
                        Instruction::ConvertValue { .. }
                        | Instruction::FormatSimple
                        | Instruction::UnaryNot
                        | Instruction::UnaryInvert
                        | Instruction::GetYieldFromIter => {
                            let _ = current_state.stack.pop();
                            push_fresh_ref(&mut current_state, &mut graph);
                            emit_abort_permanent!();
                        }

                        // Structural adaptation: async opcodes. Pyre's dispatcher
                        // errors immediately (pyopcode.rs:2027) without stack mutation.
                        // Stack effects model intended CPython shape for convergence.
                        Instruction::GetAiter | Instruction::GetAwaitable { .. } => {
                            let _ = current_state.stack.pop();
                            push_fresh_ref(&mut current_state, &mut graph);
                            emit_abort_permanent!();
                        }

                        // StoreDeref: pops 1 value. Net: -1.
                        Instruction::StoreDeref { .. } => {
                            pop_and_decr_depth(&mut current_state, &mut current_depth);
                            emit_abort_permanent!();
                        }

                        // Instructions that don't touch the operand stack (locals/cells only).
                        Instruction::DeleteFast { .. }
                        | Instruction::DeleteDeref { .. }
                        | Instruction::DeleteGlobal { .. }
                        | Instruction::DeleteName { .. }
                        | Instruction::CopyFreeVars { .. }
                        | Instruction::MakeCell { .. }
                        | Instruction::SetupAnnotations => {
                            emit_abort_permanent!();
                        }

                        // ExitInitCheck: no-op in pyre (pyopcode.rs:2069). Net: 0.
                        // RustPython pops the __init__ return value, but pyre's
                        // dispatch is a plain Ok(StepResult::Continue).
                        Instruction::ExitInitCheck => {
                            emit_abort_permanent!();
                        }

                        // StoreName pops 1 value from the stack.
                        // (This is separate from the above because pyopcode.rs pops.)

                        // YieldValue: pops yielded value, pushes placeholder back. Net: 0.
                        // Replace shadow value. rpython/flowspace/flowcontext.py:721,
                        // liveness.rs:569, assemble.py:1543.
                        Instruction::YieldValue { .. } => {
                            let _ = current_state.stack.pop();
                            push_fresh_ref(&mut current_state, &mut graph);
                            emit_abort_permanent!();
                        }

                        // ReturnGenerator: pushes 1. Net: +1.
                        Instruction::ReturnGenerator => {
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // Send: pops sent value, peeks iter, pushes next result. Net: 0.
                        // Replace shadow value.
                        Instruction::Send { .. } => {
                            let _ = current_state.stack.pop();
                            push_fresh_ref(&mut current_state, &mut graph);
                            emit_abort_permanent!();
                        }

                        // Structural adaptation: async opcodes below. Pyre's dispatcher
                        // errors immediately (pyopcode.rs:2027) without stack mutation.
                        // Stack effects model intended CPython shape for convergence.

                        // GetAnext: pushes 1. Net: +1.
                        Instruction::GetAnext => {
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // EndAsyncFor: pops 2. Net: -2.
                        // CPython 3.12/3.13 semantics; PyPy pops 3 (w_exc, w_prev, aiter)
                        // on the StopAsyncIteration path (assemble.py:1578). Structural
                        // adaptation: pyre targets CPython opcode shape here.
                        Instruction::EndAsyncFor => {
                            for _ in 0..2 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            emit_abort_permanent!();
                        }

                        // CleanupThrow: pops 3, pushes 1. Net: -2.
                        Instruction::CleanupThrow => {
                            for _ in 0..3 {
                                pop_and_decr_depth(&mut current_state, &mut current_depth);
                            }
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // MatchSequence: peeks TOS (subject), pushes bool. Net: +1.
                        // assemble.py:1614, liveness.rs:601.
                        Instruction::MatchSequence => {
                            push_fresh_ref(&mut current_state, &mut graph);
                            current_depth += 1;
                            emit_abort_permanent!();
                        }

                        // Catch-all: unknown instruction.
                        _other => {
                            emit_abort_permanent!();
                        }
                    }
                    sync_stack_state(&mut graph, &mut current_state, current_depth);
                    current_state.next_offset = py_pc + 1;
                    current_state.blocklist =
                        frame_blocks_for_offset(code, current_state.next_offset);
                    if let Some(catch_label) = catch_for_pc[py_pc] {
                        // RPython `flowcontext.py:130-156 guessexception`
                        // attaches an exception edge only to **canraise**
                        // ops — control-flow opcodes (POP_JUMP_IF_*,
                        // JUMP_*, RETURN_*, RAISE_*, RERAISE) do not
                        // canraise and close the block with their own
                        // explicit exits.  Pyre's catch_for_pc map covers
                        // every PC inside a Python exception_table range
                        // (a coarser over-approximation), so without this
                        // gate `emit_catch_exception!` would append a
                        // stray catch link to a block whose exits the
                        // just-emitted opcode already closed.
                        let block_already_closed = !current_block.block().borrow().exits.is_empty();
                        if !block_already_closed {
                            emit_catch_exception!(catch_label);
                        }
                    }
                }
            } // end inner while-let pendingblocks

            // Outer loop outer loop logic.  After main drain
            // exhausts pendingblocks, catch landings emit (below) runs
            // ONCE (gated on `catch_landings_processed`).  `emit_goto!`
            // in catch landings queues handler-entry blocks onto
            // pendingblocks; the next outer-loop iteration's inner
            // while-let drains them so they don't end up as orphan
            // empty-exits blocks with framestate-wide inputargs.
            if catch_landings_processed {
                break;
            }
            catch_landings_processed = true;

            for site in &catch_sites {
                emit_mark_label_catch_landing!(site.landing_label);
                // `emit_mark_label_catch_landing!` reassigns
                // `current_block` to the pre-allocated catch landing
                // block on every iteration, so subsequent graph emits
                // in this loop body land in a block reachable from
                // `graph.iterblocks()`.  Lock the invariant in debug
                // builds: the exception unwind PY_NULL graph dual-
                // write below and any future catch-landing dual-write
                // rely on this targeting being intact.
                debug_assert_eq!(
                    current_block, site.landing,
                    "catch_landing block-targeting invariant violated: \
                 current_block != site.landing for landing_label {} after \
                 emit_mark_label_catch_landing!",
                    site.landing_label,
                );
                // eval.rs:150-168 handle_exception parity:
                // the handler edge enters with the protected prefix of the
                // value stack preserved, then `push_lasti` (if any), then the
                // exception value. `emit_goto!(handler_py_pc)` snapshots
                // `current_state`, so mirror the same stack shape here before
                // linking the landing block to the handler block.
                sync_stack_state(&mut graph, &mut current_state, site.stack_depth);
                if site.push_lasti {
                    push_fresh_ref(&mut current_state, &mut graph);
                }
                // `flatten.py:336-352 generate_last_exc` emits the
                // `last_exc_value` SSARepr op at flatten time only — there
                // is no graph SpaceOperation counterpart, the Variable
                // flows through `link.last_exc_value` and is materialised
                // by the flatten-time emission.  Allocate a fresh Ref
                // Variable to carry the catch-landing's exception value
                // through `current_state.stack` and the subsequent vable
                // push, matching the variable-lifecycle shape upstream
                // produces — without recording a graph `last_exc_value`
                // SpaceOp the flatten driver would have to filter.
                let exc_value: super::flow::FlowValue = fresh_ref_value(&mut graph);
                current_state.stack.push(exc_value.clone());
                // pyframe.py:503-510 + eval.rs:155-158 `dropvaluesuntil` parity:
                //
                //     while frame.valuestackdepth > target_depth:
                //         frame.pop()          # locals_cells_stack_w[d] = None
                //
                // Python 3.11+ exception-table dispatch pops each value-stack
                // slot above the handler's declared depth and clears it to
                // `None` before pushing lasti / the exception value. Without
                // this step the vable array keeps stale refs at the popped
                // slots, which GC tracing and blackhole resume will read back.
                //
                // The raising PC is `site.lasti_py_pc`; its entry depth
                // (`depth_at_pc[site.lasti_py_pc]`) is the upper bound on
                // runtime valuestackdepth at any guard-firing point within
                // that PC's emitted IR, because every sub-op's guard runs
                // after its `emit_vsd!` and the peak depth within a pc equals
                // `depth_at_pc[pc]` for all raise-capable opcodes (BINARY_OP,
                // CALL, etc. enter with their args already on the stack).
                let raising_depth = depth_at_pc
                    .get(site.lasti_py_pc)
                    .copied()
                    .unwrap_or(site.stack_depth);
                if is_portal {
                    let mut unwind_depth = raising_depth;
                    while unwind_depth > site.stack_depth {
                        unwind_depth -= 1;
                        let depth_value = (stack_base_absolute + unwind_depth as usize) as i64;
                        // Graph-side dual-write — same shape as
                        // `emit_pushvalue_ref_const!` at codewriter.rs:3576-3603.
                        // The unwind PY_NULL is `Constant::none()` per
                        // assembler.py:109 ConstPtr.NULL.
                        let v_idx: super::flow::FlowValue =
                            super::flow::Constant::signed(depth_value).into();
                        record_graph_op(
                            &current_block.block(),
                            "setarrayitem_vable_r",
                            vable_setarrayitem_ref_graph_args(
                                frame_var.into(),
                                v_idx.into(),
                                super::flow::Constant::none().into(),
                            ),
                            None,
                            -1,
                        );
                        emit_vable_setarrayitem_ref_const_idx_const_value!(
                            portal_frame_reg,
                            0_u16,
                            depth_value,
                            pyre_object::PY_NULL as i64
                        );
                    }
                }
                // pyframe.py:378-387 `pushvalue` semantics — each push writes
                // `locals_cells_stack_w[depth]` AND bumps `valuestackdepth`.
                // jtransform.py:1898 `do_fixed_list_setitem` lowers the array
                // write to `setarrayitem_vable_r`; jtransform.py:920-928
                // lowers the `valuestackdepth` write to `setfield_vable_i`.
                // Without this mirror, the handler's first opcode (and any
                // compiled-trace re-entry via ContinueRunningNormally) reads
                // stale vable state because only the SSA stack slot was
                // populated.
                let mut exc_slot = stack_base + site.stack_depth;
                let mut depth: u16 = site.stack_depth;
                if site.push_lasti {
                    // box_int writes the lasti result directly to
                    // the exception slot, retiring obj_tmp0 → exc_slot copy.
                    // box_int_fn factor refactor
                    // (exception lasti site).  See LoadSmallInt site for
                    // the shared rationale.
                    push_walker_emit(
                        &current_block,
                        super::flatten::build_box_int_fn_residual_call_ir_r_insn(
                            box_int_fn_idx,
                            site.lasti_py_pc as i64,
                            exc_slot,
                        ),
                    );
                    // Graph-side `residual_call_ir_r` for
                    // `box_int_fn(lasti:Int) → Ref` followed by the
                    // matching `setarrayitem_vable_r(frame, lasti_depth,
                    // boxed_lasti)` — the call result Variable feeds the
                    // vable-array write so the def-use chain matches the
                    // upstream "call result is the consumer's input" shape
                    // (`flowcontext.py:135-139` recorder pattern,
                    // `jtransform.py:1898 do_fixed_list_setitem` for the
                    // array write).
                    if is_portal {
                        let boxed_lasti = residual_call!(
                            box_int_fn_idx,
                            CallFlavor::Plain,
                            vec![super::flow::Constant::signed(site.lasti_py_pc as i64).into()],
                            vec![],
                            vec![],
                            vec![Kind::Int],
                            ResKind::Ref,
                            -1,
                        );
                        pin!(boxed_lasti, exc_slot);
                        if let Some(boxed_var) = boxed_lasti {
                            let lasti_depth_value = (stack_base_absolute + depth as usize) as i64;
                            let v_lasti_idx: super::flow::FlowValue =
                                super::flow::Constant::signed(lasti_depth_value).into();
                            record_graph_op(
                                &current_block.block(),
                                "setarrayitem_vable_r",
                                vable_setarrayitem_ref_graph_args(
                                    frame_var.into(),
                                    v_lasti_idx.into(),
                                    boxed_var.into(),
                                ),
                                None,
                                -1,
                            );
                        }
                        emit_vable_setarrayitem_ref_const_idx!(
                            portal_frame_reg,
                            0_u16,
                            (stack_base_absolute + depth as usize) as i64,
                            exc_slot
                        );
                    }
                    depth += 1;
                    emit_vsd!(depth, site.handler_py_pc);
                    exc_slot += 1;
                }
                // `flatten.py:336-347 generate_last_exc` emits
                // `last_exception` immediately before `last_exc_value` at
                // every exception link landing where
                // `link.last_exception` is in `link.args`.  pyre's walker
                // synthesises both Variables (exception_edge_vars,
                // codewriter.rs:944-951), so both must land in the
                // SSARepr.  Use a fresh Int scratch slot — pyre's
                // catch-handler bytecode does not currently read the
                // exception-class register (per-kind PyType makes
                // type-discrimination implicit), so the write is
                // structural parity rather than a live consumer.
                let exc_type_slot = ssarepr.fresh_var(Kind::Int, scratch_int_base).0;
                emit_last_exception!(exc_type_slot);
                emit_last_exc_value!(exc_slot);
                // `flatten.py:336-347 generate_last_exc` writes
                // `last_exc_value` straight into `getcolor(handler_inputarg)`,
                // and `insert_renamings` (flatten.py:311) excludes the exc
                // value from the copy list.  pyre's walker materialises the
                // exception into the positional `exc_slot` above; that slot
                // equals `getcolor(handler_inputarg)` (both are
                // `stack_base + depth`).  But `exc_value` reaches graph
                // regalloc uncoloured-to-slot and gets an arbitrary colour,
                // so `walker_post_walk_insert_renamings` emits a spurious
                // `ref_copy[get_color(exc_value) -> get_color(handler_inputarg)]`
                // that reads the never-written colour and overwrites the exc
                // at the catch-landing -> handler boundary — leaving the
                // handler's CHECK_EXC_MATCH / PUSH_EXC_INFO with a NULL
                // exception.  Pin `exc_value` to `exc_slot` so
                // `get_color(exc_value)` matches the slot `emit_last_exc_value!`
                // wrote; the renaming then collapses to a `src == dst` no-op
                // (skipped) or a correct `exc_slot -> handler_color` copy.
                pin!(exc_value.as_variable(), exc_slot);
                if is_portal {
                    let depth_value = (stack_base_absolute + depth as usize) as i64;
                    // pyframe.py:378-387 `pushvalue` semantics — graph
                    // dual-write of the stack mirror.
                    let v_idx: super::flow::FlowValue =
                        super::flow::Constant::signed(depth_value).into();
                    record_graph_op(
                        &current_block.block(),
                        "setarrayitem_vable_r",
                        vable_setarrayitem_ref_graph_args(
                            frame_var.into(),
                            v_idx.into(),
                            exc_value.clone().into(),
                        ),
                        None,
                        -1,
                    );
                    emit_vable_setarrayitem_ref_const_idx!(
                        portal_frame_reg,
                        0_u16,
                        depth_value,
                        exc_slot
                    );
                }
                // CATCH-LANDING dual-write follow-up.
                // RPython parity: `pypy/interpreter/pyopcode.py` exception
                // handler entry pushes the lasti box (`push_lasti` arm) and
                // the captured exc_value onto the value stack; both writes
                // lower through `jtransform.py:1898 do_fixed_list_setitem`
                // to `setarrayitem_vable_r` in the upstream graph.
                //
                // Push exc_value graph dual-write — LANDED above:
                // `last_exc_value -> v_exc_value` produces a fresh Ref
                // Variable per catch entry (flatten.py:336-347
                // `generate_last_exc`), and the subsequent
                // `setarrayitem_vable_r(_, ConstInt(stack_base+depth),
                // v_exc_value)` consumes it.
                //
                // Push lasti graph dual-write — LANDED above for portal
                // catch landings: `box_int(lasti_py_pc)` records the
                // `residual_call_ir_r` producer and the following
                // `setarrayitem_vable_r(_, ConstInt(stack_base+depth),
                // boxed_lasti)` consumes it.  Non-portal catch landings have
                // no virtualizable frame graph inputarg, so they keep the
                // walker-only stack write.
                //
                // Block-targeting is handled by
                // `emit_mark_label_catch_landing!`, which runs at the
                // head of every iteration and reassigns `current_block`
                // to the pre-allocated catch landing block.  The
                // invariant is locked in via `debug_assert_eq!` at the
                // head of the loop body.
                //
                depth += 1;
                emit_vsd!(depth, site.handler_py_pc);
                emit_goto!(site.handler_py_pc);
            }
        } // end outer drain loop

        // RPython flatten.py parity: every code path ends with an explicit
        // return/raise/goto/unreachable. No end-of-code sentinel needed —
        // falling off the end is unreachable if all bytecodes are covered.

        // pyre-only PyJitCode.has_abort: a "this jitcode cannot be
        // blackhole-dispatched, pipe straight to the interpreter" flag.
        // RPython has no such flag (rpython/jit/codewriter/jitcode.py:14
        // — no abort tracking on JitCode). Upstream's `Assembler.abort()`
        // (assembler.py:177-181, bhimpl_abort) emits BC_ABORT so the
        // blackhole raises SwitchToBlackhole(ABORT_ESCAPE) at runtime;
        // `abort_permanent()` is a different pyre-only bytecode we emit
        // for genuinely unsupported Python opcodes, and its execution
        // path already raises/aborts correctly from the blackhole. We
        // keep has_abort narrowly scoped to `abort()` emissions (matches
        // the JitCodeBuilder flag shape) so the flag's meaning doesn't
        // drift into "assembler overflow" or "abort_permanent present"
        // — both of which the assembler/blackhole already handle without
        // a front-end gate.
        let has_abort = assembler.has_abort_flag();

        // `simplify_graph` (`translator.py:55-56`) parity: collapse the
        // empty forwarding blocks `mergeblock` supersede left behind
        // (`flowcontext.py:455-463`) before regalloc coalescing and the
        // drain read the graph.  Runs BEFORE `collect_cfg_coalesce_pairs`
        // so the coalesce pairs (and therefore the colors the renamings
        // use) reflect the collapsed `predecessor -> generalization`
        // links; `rewrite_dead_forwarder_gotos` then retargets the
        // walker's inline-emitted `TLabel`s to match.
        //
        // The full orthodox pass list lives in
        // [`super::simplify::simplify_graph`] / `all_passes` (issue #112).
        // Here we run the **walker-safe subset** of the orthodox pass list,
        // each either keeping every SpamBlock's `per_block_ssarepr` in the
        // block-by-block drain, or reconciling the inline byte stream with a
        // bridge, in the upstream `all_passes` relative order:
        //
        //   eliminate_empty_blocks   (+ rewrite_dead_forwarder_gotos bridge;
        //                             dead blocks have cleared per_block_ssarepr)
        //   constfold_exitswitch     (no-op today — the walker folds constant
        //                             branch conditions before emitting an
        //                             exitswitch — but kept for all_passes
        //                             parity and to fold any that do appear)
        //   remove_trivial_links     (merges single-entry/exit chains; the
        //                             merge bridge below relocates each merged
        //                             target's per_block_ssarepr into source)
        //
        // EMPIRICALLY VALIDATED (#73): this subset passes the full check.py
        // gate (dynasm 39/39, correctness + 8-benchmark performance).
        //
        // The remaining active `all_passes` entries are deliberately NOT wired
        // here (all pay off only once codegen is driven from the graph, #73):
        //
        //   - `transform_dead_op_vars` and `remove_identical_vars_SSA` both
        //     drop a graph inputarg + its predecessor link arg (one PRUNES dead
        //     ones, the other DEDUPS a duplicate phi and renames it to its
        //     twin).  But the walker's real bytecode uses live in each
        //     SpamBlock's `per_block_ssarepr` and read the dropped variable's
        //     walker SLOT, which the graph rename does not touch.  Since
        //     `walker_post_walk_insert_renamings` only sees the pruned graph it
        //     emits no copy into that slot, so the inline insns can read a
        //     stale slot — a lost / wrong edge value (codex review P1 x2,
        //     PR #127).  They need a walker-side liveness/renaming bridge, or
        //     the graph-driven drain, before they can be wired.
        //   - `ssa_to_ssi` runs CORRECTLY on walker graphs (the
        //     `backendopt_ssa.rs` stop-at-startblock adaptation), but is
        //     overhead-only: it threads values the walker already routes
        //     through its register/slot model, adding redundant coalesce pairs
        //     / `ref_copy` that measurably regress exception-heavy code
        //     (raise_catch ~4.2x -> ~10.9x).
        //
        // The other `all_passes` entries are structural no-ops on today's
        // empty-`operations` walker blocks (see their classification in
        // `simplify.rs`).
        eliminate_empty_blocks(&graph);
        super::simplify::constfold_exitswitch(&graph);
        // Port-boundary guard: after the collapse, no link reachable from
        // the startblock may still target a dead forwarder.  RPython has
        // no equivalent check (its graph is simplified before flatten),
        // but pyre's `walker_post_walk_insert_renamings` reachability DFS
        // skips dead targets, so a surviving reachable link -> dead edge
        // would silently drop that edge's renamings while
        // `rewrite_dead_forwarder_gotos` still retargets the byte goto —
        // the exact renaming/goto divergence this guard prevents.  Fail loud
        // instead.
        for link_ref in graph.iterlinks() {
            if let Some(target) = link_ref.borrow().target.clone() {
                assert!(
                    !target.borrow().dead,
                    "eliminate_empty_blocks: reachable link still targets \
                     dead forwarder {}; collapse is incomplete",
                    super::flatten::block_label_name(&target),
                );
            }
        }
        // Byte-stream companion to `eliminate_empty_blocks`: retarget the
        // walker's inline-emitted gotos that still name a collapsed dead
        // forwarder to the surviving generalization.  MUST run before the
        // merge bridge below: for a `source -> dead_forwarder -> target`
        // shape `remove_trivial_links` records the merge `(source, target)`,
        // but the source block's boundary terminator still reads
        // `goto TLabel(dead_forwarder)` until this rewrite — so the merge
        // strip (which looks for `block_label_name(target)`) would miss it
        // and leave the stale terminator in front of the absorbed target
        // opcodes, letting the single-exit renaming splice land before
        // them (codex P2, PR #130).
        rewrite_dead_forwarder_gotos(&all_walker_blocks);
        // remove_trivial_links merges single-entry/single-exit chains; the
        // merge bridge relocates each merged target's inline per_block_ssarepr
        // into its surviving source so the drain keeps the opcodes (issue #73).
        let trivial_link_merges = super::simplify::remove_trivial_links_recording(&graph);
        rewrite_trivial_link_merges(
            &trivial_link_merges,
            &all_walker_blocks,
            &mut walker_pc_live_marker_pos,
        );

        // Drain per-block accumulators into ssarepr.insns in
        // walker-block-creation order.  Mirrors `codewriter.py:53
        // flatten_graph(graph, regallocs, cpu)` shape — block-by-block
        // emit, no PC interleaving.
        //
        // Peel-off optimisation: at every block-switch boundary the
        // walker emits a defensive `goto TLabel(block) + Unreachable`
        // pair (the eventual drain order is not known at yield time
        // since `pendingblocks` is mixed push_front / push_back).  This
        // pass strips the pair when the next block actually opens with
        // a `Label` matching the goto target — turning a runtime no-op
        // branch into implicit fall-through.  Upstream `flatten.py:106-155
        // make_link` skips the goto outright via recursive descent +
        // `seen_blocks` (`flatten.py:110-113`); pyre's two-phase
        // emit-then-strip approach converges to the same byte stream.
        //
        // Seed `walker_slot_for_variable` with block inputarg slots
        // BEFORE `walker_post_walk_insert_renamings` reads it.  The same
        // pairing pass also runs downstream (idempotent via
        // `pair_walker_slot_if_absent`).
        for spam in &all_walker_blocks {
            let Some(state) = spam.framestate() else {
                continue;
            };
            for (idx, value) in state.mergeable().iter().enumerate() {
                if let Some(super::flow::FlowValue::Variable(v)) = value {
                    if let Some(slot) = state.mergeable_index_to_slot(idx) {
                        pin_if_absent!(Some(v.clone()), slot);
                    }
                }
            }
        }
        // Compute graph regallocs ONCE pre-drain so walker
        // insert_renamings and any downstream consumer share identical
        // colors (HashMap iteration non-determinism between two
        // separate regalloc calls would otherwise diverge bridge-
        // fallback Variables' colors).
        //
        // PyPy `regalloc.py` runs the CFG coalesce sweep BEFORE
        // `flatten.py:154 insert_renamings` mutates the graph.  Collect
        // once here so both the canonical pass and the SSARepr-side
        // `allocate_registers` observe the same pre-renaming graph.  The
        // walker scratch↔inputarg pins (#248) used to be chained in
        // alongside these CFG pairs, but P4SUB proved them subsumed:
        // `cfg_variable_pairs` alone reproduces the combined Ref coloring
        // (per-Variable + num_colors) on every production graph, so they
        // are dropped (#238).
        //
        // PRE-EXISTING-ADAPTATION (parity regression vs PyPy): these CFG
        // `link.args ↔ target.inputargs` pairs are pre-merged into the
        // union-find BEFORE `make_dependencies`.  PyPy `regalloc.py:79-96
        // coalesce_variables` + `:98-112 _try_coalesce` coalesce post-
        // `make_dependencies` with the `v0 not in dg.neighbours[w0]`
        // interference check; the pre-merge bypasses that check and
        // silently merges pairs PyPy would reject.  Honouring the check
        // measurably regresses cranelift (fib_recursive/raise_catch/
        // fannkuch TIMEOUT) — see #260.  Root cause
        // is walker's color-aggressive emit diverging from canonical's
        // interference-respecting coloring; `walker_post_walk_insert_
        // renamings` emits bridge ref_copy ops, pending retirement of
        // `walker_slot_for_variable` (#238).
        let cfg_variable_pairs = collect_cfg_coalesce_pairs(&graph);
        let mut graph_regallocs = super::regalloc::perform_register_allocation_all_kinds_with_pairs(
            &graph,
            &cfg_variable_pairs,
        );
        super::regalloc::enforce_input_args(&graph, &mut graph_regallocs);
        // Default-off measurement: run the post-walk canonical driver on
        // the production graph (`flatten.py:63-70`) and record its insn
        // category buckets for the post-drain DIFF survey below.  This is
        // the first production-reachable caller of `flatten_graph`; until
        // now only `*_for_test` and `#[cfg(test)]` reached it.  Survey, not
        // abort: graph shapes the canonical driver can't yet flatten
        // panic inside `flatten_graph` (the Phase-1 SCAFFOLD panics —
        // uncolored exception-edge vars, non-portal `simple_call` frame),
        // so catch the unwind and record `None`, letting one
        // `PYRE_FLATTEN_MEASURE=1` pass survey every production graph
        // instead of aborting on the first bad one.  `flatten_graph` only
        // reads `graph`/`cpu` (no `.borrow_mut`, `next_variable_id`
        // untouched), so discarding the result leaves the post-measurement
        // production path byte-identical; gate unset => skipped entirely.
        // `(genuine, live, copy, structural)` buckets paired with the
        // canonical driver's genuine-category opname tally (#230.M3) and
        // the #288 sparse-liveness coverage report
        // `(reachable, resolved, absent, uncovered)`, or `None` when the
        // driver panicked.  All three derive from the single canonical
        // `flatten_graph` stream so the per-graph genuine residual can be
        // named and the sparse `-live-` resolver verified at the log site.
        let measure_canonical: Option<(
            (usize, usize, usize, usize),
            std::collections::BTreeMap<String, usize>,
            (usize, usize, usize, usize),
            (usize, usize, usize, usize, usize),
        )> = if std::env::var("PYRE_FLATTEN_MEASURE").is_ok() {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut measure_regallocs = graph_regallocs.clone();
                let canonical = super::flatten::flatten_graph(
                    &graph,
                    &mut measure_regallocs,
                    true,
                    Some(self.cpu()),
                );
                // #288: resolve each reachable Python PC to its
                // nearest-at-or-before `-live-` marker from the canonical
                // stream's OWN sparse markers + `pc_first_insn_pos`, then
                // bucket the reachable PCs into resolved / absent (the PC
                // emits no graph op — a stack-only instruction — so it has
                // no `pc_first_insn_pos` entry) / uncovered (the PC emits a
                // graph op but no `-live-` marker precedes it).
                let live_indices = derive_pc_live_indices_from_sparse(&canonical, num_instrs);
                let live_vars = pyre_jit_trace::state::liveness_for(code as *const _);
                let mut present = vec![false; num_instrs];
                for &(pc, _) in &canonical.pc_first_insn_pos {
                    if (0..num_instrs as i64).contains(&pc) {
                        present[pc as usize] = true;
                    }
                }
                let (mut reachable, mut resolved, mut absent, mut uncovered) = (0, 0, 0, 0);
                for pc in 0..num_instrs {
                    if !live_vars.is_reachable(pc) {
                        continue;
                    }
                    reachable += 1;
                    if live_indices[pc].is_some() {
                        resolved += 1;
                    } else if !present[pc] {
                        absent += 1;
                    } else {
                        uncovered += 1;
                    }
                }
                // #281 verify: restrict the sparse-resolver coverage check
                // to the runtime's structural resume targets (branch guards
                // + can-raise calls, #285) and report how the resolver feeds
                // / keys each.
                let resume = resume_target_resolver_coverage(&canonical, &live_indices);
                (
                    bucket_measure_insns(&canonical.insns),
                    genuine_opname_tally(&canonical.insns),
                    (reachable, resolved, absent, uncovered),
                    resume,
                )
            }))
            .ok()
        } else {
            None
        };
        // Walker-tracked per-PC `-live-` marker positions exposed to
        // the post-drain `pc_map` computation.  Populated inside the
        // drain block below; consumed by `filter_liveness_in_place`
        // (translated through the `remove_repeated_live` remap) as the
        // sole source for `pc_map`.
        let mut walker_tracked_pc_live_indices_out: Option<Vec<usize>> = None;
        {
            // Walker post-walk insert_renamings.
            // Run BEFORE the per-block drain so the splice positions
            // land in the per_block_ssarepr accumulators.  Mirrors
            // `flatten.py:154 self.insert_renamings(link)` for the
            // simple unconditional single-exit case (loop back-edges,
            // straight-line forward jumps).  Multi-exit blocks
            // (POP_JUMP_IF_*, canraise) require per-link positional
            // injection between source-block terminator and each
            // target-block Label, which is not yet implemented here.
            walker_post_walk_insert_renamings(
                &mut graph,
                &walker_slot_for_variable,
                &graph_regallocs,
                &mut all_walker_blocks,
                &mut walker_pc_live_marker_pos,
            );
            // Reorder all_walker_blocks per `graph.iterblocks()` DFS
            // pre-order so the drain matches canonical `make_bytecode_block`
            // (`flatten.py:107-156`) emission order.  Canonical recurses
            // into each link's target immediately after emitting the
            // source's body; iterblocks (`flowspace/model.py:55-77`)
            // produces the same pre-order via explicit reversed-stack
            // DFS.  Walker emits per-PC into pendingblocks queue order
            // which diverges from DFS — block-boundary `goto pcX` ops
            // whose target isn't the immediately-next walker block
            // survive `strip_walker_block_boundary_goto` and produce
            // walker_unmatched against canonical.  Post-walk reorder
            // by DFS aligns the drain so the strip catches them.
            //
            // Dead supersede blocks have empty `per_block_ssarepr`
            // (cleared by `mark_dead`), so their position in the order
            // doesn't contribute insns.  Synthetic walker blocks
            // (catch-landings, blocks not reachable in graph DFS)
            // append in their original creation order after the
            // DFS-matched prefix.
            let dfs_blocks = graph.iterblocks();
            let mut reordered: Vec<SpamBlockRef> = Vec::with_capacity(all_walker_blocks.len());
            let mut placed: Vec<bool> = vec![false; all_walker_blocks.len()];
            for gb in &dfs_blocks {
                for (idx, spam) in all_walker_blocks.iter().enumerate() {
                    if !placed[idx] && spam.block() == *gb {
                        reordered.push(spam.clone());
                        placed[idx] = true;
                    }
                }
            }
            for (idx, spam) in all_walker_blocks.iter().enumerate() {
                if !placed[idx] {
                    reordered.push(spam.clone());
                }
            }
            let mut blocks: Vec<Vec<super::flatten::Insn>> = reordered
                .iter()
                .map(|block| block.per_block_ssarepr())
                .collect();
            // Resolve walker-tracked PC live-marker positions to
            // absolute SSARepr indices using POST-STRIP prefix lengths.
            // Captured BEFORE strip mutates `blocks` so per-block
            // offsets stay aligned — strip only truncates the trailing
            // `goto + Unreachable` pair from a block tail, so the
            // `-live-` marker positions (which never live in the tail)
            // are invariant.  Picks the FIRST entry per PC whose block
            // contributes non-empty content to the final SSARepr.
            let walker_tracked_pc_indices: Option<Vec<usize>> = {
                // Compute each block's post-strip length without invoking
                // `strip_walker_block_boundary_goto` directly: that helper
                // moves block insns into its return value (via
                // `Vec::append`), which would zero-out the source blocks
                // and corrupt the resolver's offset math.  Mirror the
                // helper's strip-detection rule (goto+Unreachable tail
                // whose target matches a leading label in any subsequent
                // block) but only measure the resulting length here.
                let post_strip_lens: Vec<usize> = (0..blocks.len())
                    .map(|i| {
                        // Mirror `strip_walker_block_boundary_goto`'s
                        // IMMEDIATE-next-non-empty-block rule per
                        // `flatten.py:106-155 make_link` recursive
                        // descent — fall-through never hops over an
                        // intervening non-empty block.
                        let next_label_names: Vec<String> = (i + 1..blocks.len())
                            .find(|&j| !blocks[j].is_empty())
                            .map(|j| {
                                blocks[j]
                                    .iter()
                                    .take_while(|insn| {
                                        matches!(insn, super::flatten::Insn::Label(_))
                                    })
                                    .filter_map(|insn| match insn {
                                        super::flatten::Insn::Label(l) => Some(l.name.clone()),
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        let len = blocks[i].len();
                        let strip_tail = if len >= 2 {
                            match (&blocks[i][len - 2], &blocks[i][len - 1]) {
                                (
                                    super::flatten::Insn::Op { opname, args, .. },
                                    super::flatten::Insn::Unreachable,
                                ) if opname == "goto"
                                    && args.len() == 1
                                    && matches!(&args[0], Operand::TLabel(target)
                                        if next_label_names.iter().any(|n| n == &target.name)) =>
                                {
                                    2
                                }
                                _ => 0,
                            }
                        } else {
                            0
                        };
                        len - strip_tail
                    })
                    .collect();
                let mut post_strip_block_starts: Vec<usize> = Vec::with_capacity(blocks.len());
                let mut running = 0usize;
                for &len in &post_strip_lens {
                    post_strip_block_starts.push(running);
                    running += len;
                }
                let resolve_walker_pc =
                    |records: &Vec<Vec<(SpamBlockRef, usize)>>| -> Option<Vec<usize>> {
                        let mut translated: Vec<usize> = Vec::with_capacity(records.len());
                        for py_pc_entries in records {
                            let resolved = py_pc_entries.iter().find_map(|(block_ref, offset)| {
                                let block_pos = reordered.iter().position(|b| b == block_ref)?;
                                if post_strip_lens[block_pos] == 0 {
                                    return None;
                                }
                                if *offset >= post_strip_lens[block_pos] {
                                    return None;
                                }
                                Some(post_strip_block_starts[block_pos] + offset)
                            });
                            match resolved {
                                Some(idx) => translated.push(idx),
                                None => return None,
                            }
                        }
                        Some(translated)
                    };
                resolve_walker_pc(&walker_pc_live_marker_pos)
            };
            ssarepr.insns = strip_walker_block_boundary_goto(&mut blocks);
            // Walker-tracked per-PC `-live-` positions are the sole
            // source of truth for `pc_map`; downstream consumers
            // (`filter_liveness_in_place`, `pc_map`) translate them
            // through the `remove_repeated_live` remap.
            walker_tracked_pc_live_indices_out = walker_tracked_pc_indices;
            // #230.M2 DIFF bucketing survey: under PYRE_FLATTEN_MEASURE,
            // bucket both the canonical driver's insn stream (captured
            // pre-drain above) and the walker's drained `ssarepr.insns`
            // into `(genuine, live, copy, struct)` and log them
            // category-to-category, so the ~40% excess (M1: canonical is
            // consistently ~60-65% of walker length) is attributed to a
            // specific retirement.  `live` = per-PC `-live-` markers
            // (retired by #287), `copy` = `*_copy` renamings the walker
            // emits inline where upstream defers them to
            // `insert_renamings` (#289/#290), `struct` = `Label`/`---`,
            // `genuine` = everything else.  When walker `genuine` equals
            // canonical `genuine` the entire gap is `live`+`copy` (both
            // already-tracked retirements) and the flip is purely a
            // filtering problem; a residual walker `genuine` excess is a
            // NEW-DEVIATION needing its own retirement.  `Some` when the
            // canonical driver completed, `None` when it panicked
            // (Phase-1 SCAFFOLD shape).  Log-only — never affects codegen.
            if let Some((
                (cg, cl, cc, cs),
                canonical_genuine,
                (lreach, lres, labs, lunc),
                (rb, rbk, rbf, rc, rcf),
            )) = measure_canonical
            {
                let (wg, wl, wc, ws) = bucket_measure_insns(&ssarepr.insns);
                eprintln!(
                    "[flatten-measure] {} canon[g={} l={} c={} s={} tot={}] walker[g={} l={} c={} s={} tot={}] {} {}",
                    ssarepr.name,
                    cg,
                    cl,
                    cc,
                    cs,
                    cg + cl + cc + cs,
                    wg,
                    wl,
                    wc,
                    ws,
                    wg + wl + wc + ws,
                    if wg == cg {
                        "GENUINE_MATCH"
                    } else {
                        "GENUINE_DIFF"
                    },
                    if cg + cl + cc + cs == wg + wl + wc + ws {
                        "MATCH"
                    } else {
                        "DIFF"
                    },
                );
                // #230.M3: name the residual `genuine` deviation
                // opname-by-opname.  M2 collapses `genuine` to a single
                // count (e.g. int +7, fannkuch +5) but does not say which
                // opnames the walker over- or under-emits.  Diff the two
                // per-opname tallies and log each opname whose walker count
                // differs from canonical, so the residual NEW-DEVIATION has
                // a name before #287/#281 close the `-live-`/`copy` bulk.
                let walker_genuine = genuine_opname_tally(&ssarepr.insns);
                let mut opnames: Vec<&String> = canonical_genuine
                    .keys()
                    .chain(walker_genuine.keys())
                    .collect();
                opnames.sort();
                opnames.dedup();
                let diffs: Vec<String> = opnames
                    .into_iter()
                    .filter_map(|opname| {
                        let c = canonical_genuine.get(opname).copied().unwrap_or(0);
                        let w = walker_genuine.get(opname).copied().unwrap_or(0);
                        (c != w).then(|| {
                            format!("{}:w{}/c{}({:+})", opname, w, c, w as isize - c as isize)
                        })
                    })
                    .collect();
                if diffs.is_empty() {
                    eprintln!("[flatten-measure-genuine] {} MATCH", ssarepr.name);
                } else {
                    eprintln!(
                        "[flatten-measure-genuine] {} DIFF {}",
                        ssarepr.name,
                        diffs.join(" "),
                    );
                }
                // #288: survey the canonical sparse `-live-` resolver's
                // coverage over reachable instruction PCs.  `resolved` =
                // PC emits a graph op (has a `pc_first_insn_pos` entry)
                // AND a `-live-` marker precedes it.  `nopc` = PC emits no
                // graph op (stack-only instruction: LOAD_FAST/POP_TOP/…),
                // so it has no `pc_first_insn_pos` entry — only a resume
                // target if the runtime deopts into a stack-only PC, which
                // #281 must confirm it never does.  `unmarked` = PC emits
                // a graph op but no `-live-` precedes it (≈ the entry PC
                // before the first marker).  Survey-only, like the other
                // `[flatten-measure]` lines; no PASS/GAP verdict because
                // the meaningful gate (resume-target coverage, not
                // all-PC) is #281's, which restricts to deopt targets.
                eprintln!(
                    "[flatten-measure-liveness] {} reachable={} resolved={} nopc={} unmarked={}",
                    ssarepr.name, lreach, lres, labs, lunc,
                );
                // #281 verify: resume-target-restricted resolver coverage.
                // `branch`/`canraise` = structural resume targets (#285);
                // `keyed` = branch guards whose OWN owner-pc resolver entry
                // is the guard's leading marker (the strict runtime
                // `pc_map[orgpc]` invariant); `fed` = markers the resolver
                // hands to some pc; `stranded` = resume markers in NO entry
                // (the (b) nopc-fallthrough safety number).  Survey-only,
                // verdict-less like the other `[flatten-measure]` lines.
                let stranded = (rb - rbf) + (rc - rcf);
                eprintln!(
                    "[flatten-measure-resume] {} branch={} keyed={} branch_fed={} canraise={} canraise_fed={} stranded={}",
                    ssarepr.name, rb, rbk, rbf, rc, rcf, stranded,
                );
            } else if std::env::var("PYRE_FLATTEN_MEASURE").is_ok() {
                eprintln!("[flatten-measure] {} canonical=PANIC", ssarepr.name);
            }
        }

        // codewriter.py:45-47 `for kind in KINDS:
        //   regallocs[kind] = perform_register_allocation(graph, kind)`
        //
        // RPython runs regalloc on the CFG before flatten emits the
        // SSARepr (`codewriter.py:44` vs `:53-56`). Regalloc uses
        // `block.operations` + `link.args` for interference; `-live-`
        // markers don't exist yet. pyre dispatches directly to the
        // SSARepr — at regalloc time the `-live-` markers are present
        // but still hold empty args (`filter_liveness_in_place` runs
        // post-rename), so the allocator's generic `Insn::Op` walk is
        // a no-op on them, matching the upstream pre-liveness ordering.
        let inputs = super::regalloc::ExternalInputs {
            portal_frame_reg,
            portal_ec_reg,
            // Portal frames carry a virtualizable + ec red argument
            // pair (interp_jit.py:64-69). Non-portal callees pass red
            // args via the call assembler edge; the dispatch loop
            // does not pre-load them into Ref registers.
            portal_inputs: portal_frame_reg != u16::MAX,
        };
        // `flow.rs` now models `Block.operations` as upstream
        // `SpaceOperation`, not flattened `Insn`. The direct-dispatch
        // walker still emits only SSA/flatten-level data, so the shadow
        // graph remains topology-only until a pre-regalloc Variable
        // environment is introduced.

        // `regalloc.py:79-96 coalesce_variables` CFG-level coalescing:
        // preserve every Link's `(link.args[i], link.target.inputargs[i])`
        // pair for the SSARepr-side allocator too.  Graph-side regalloc
        // uses the Variable-keyed `cfg_variable_pairs`; the walker allocator
        // is slot-keyed, so project through `walker_slot_for_variable`.
        // Without this, `walker_post_walk_insert_renamings` can skip a copy
        // because graph colors already match while SSARepr regalloc later
        // assigns different colors to the same edge.
        let cfg_coalesce_pairs: Vec<(u16, u16)> = cfg_variable_pairs
            .iter()
            .filter_map(|(src, dst)| {
                let s = walker_slot_for_variable
                    .get(src.0 as usize)
                    .copied()
                    .flatten()?;
                let d = walker_slot_for_variable
                    .get(dst.0 as usize)
                    .copied()
                    .flatten()?;
                Some((s, d))
            })
            .collect();
        let alloc_result = super::regalloc::allocate_registers(
            &ssarepr,
            code.varnames.len(),
            inputs,
            &cfg_coalesce_pairs,
        );
        // Run graph-side
        // `perform_register_allocation_all_kinds` +
        // `enforce_input_args` post-walker on every
        // production graph, matching upstream `codewriter.py:44-46`'s
        // pre-flatten regalloc step.  The result is not consumed yet —
        // The `(Kind, slot) → color` bridge map that
        // the 5 SSA-side `alloc_result` consumers below currently rely
        // on; the source of truth will be swapped later.  Running the
        // graph-side allocator unconditionally first surfaces any
        // graph topology that the allocator can't process before any
        // downstream code reads from it.
        //
        // `graph_regallocs` is computed earlier
        // (pre-drain) so walker insert_renamings shares the same
        // colors as canonical's pass below.  The duplicate compute is
        // retired; we reuse the shared instance.

        super::regalloc::apply_rename(&mut ssarepr, &alloc_result.rename);

        // `flatten.py:88-100` `enforce_input_args` may rotate the
        // portal `(frame, ec)` inputargs into new colors. Keep the
        // pyre-side metadata aligned with the post-regalloc SSA/JitCode
        // slots the assembler will actually emit; the blackhole fill
        // path must write the colored portal registers, not the
        // pre-color layout placeholders.
        let portal_frame_reg =
            super::regalloc::rename_lookup(&alloc_result.rename, Kind::Ref, portal_frame_reg);
        let portal_ec_reg =
            super::regalloc::rename_lookup(&alloc_result.rename, Kind::Ref, portal_ec_reg);

        // Graph-side register allocation: record each
        // Python-semantic stack slot's post-regalloc color. With the
        // input-arg pinning removed (regalloc.rs `enforce_input_args`
        // no longer rotates stack slots), the chordal coloring may
        // coalesce disjointly-live stack slots into the same color,
        // so this map is the only authoritative slot → color
        // translation for runtime decoders.
        //
        // Length is `code.max_stackdepth` (= CPython `co_stacksize`) so
        // the map covers every stack slot the runtime PyFrame allocates
        // (`pyframe.rs:1576` `alloc_fixed_array_with_header(num_locals +
        // num_cells + max_stack, ...)`). Using `max(depth_at_pc)` would
        // fall short of `co_stacksize` on programs whose JIT-traced PCs
        // do not reach the static peak; the bridge fallback at
        // `state.rs::setup_bridge_sym` (`stack_base + color_map.len()`)
        // requires the full PyFrame length, so this width is the
        // contract `pyjitcode.rs:97-110` already documents.
        let stack_map_len = max_stackdepth as u16;
        let mut stack_slot_color_map: Vec<u16> = Vec::with_capacity(stack_map_len as usize);
        for d in 0..stack_map_len {
            let pre = stack_base + d;
            let post = super::regalloc::rename_lookup(&alloc_result.rename, Kind::Ref, pre);
            stack_slot_color_map.push(post);
        }
        // SSA-authoritative live_r: record each Python-semantic
        // local slot's post-regalloc color.  The encoder
        // (`get_list_of_active_boxes`) derives `semantic_idx` from
        // `color_idx < nlocals → identity`.
        //
        // Today `enforce_input_args` (regalloc.rs:524-563, flatten.py:88
        // -100 parity) pins each local-i inputarg color to identity
        // (`color = i`), so this map is `[0, 1, ..., nlocals-1]` for
        // every populated jitcode.  When `enforce_input_args` pinning
        // is relaxed, the encoder will read this map to
        // derive the semantic local index from a non-identity color.
        let mut pyre_color_for_semantic_local: Vec<u16> = Vec::with_capacity(nlocals);
        for i in 0..nlocals as u16 {
            let post = super::regalloc::rename_lookup(&alloc_result.rename, Kind::Ref, i);
            pyre_color_for_semantic_local.push(post);
        }
        // After step C the chordal coloring is free to coalesce
        // disjointly-live stack slots into the same color, so the full
        // map may legitimately repeat colors (e.g. `[1, 1, 2, 3, 4, 0,
        // 5]`). The runtime decoder bounds its `iter().position()`
        // lookup to the slots that are LIVE at the resume PC
        // (`stack_only` in `state.rs::write_from_resume_data_partial`),
        // and chordal coloring guarantees uniqueness within any
        // simultaneously-live subset.

        // codewriter.py:55-56 parity: `compute_liveness(ssarepr)` runs
        // AFTER regalloc + flatten, so the live-register indices the
        // pass writes into each `-live-` marker are already the
        // post-rename colors. `filter_liveness_in_place` then splits
        // them into live_i/live_r/live_f per assembler.py:150-152.
        let post_remove_live_indices = filter_liveness_in_place(
            &mut ssarepr,
            code,
            &depth_at_pc,
            &pyre_color_for_semantic_local,
            &stack_slot_color_map,
            portal_frame_reg,
            portal_ec_reg,
            walker_tracked_pc_live_indices_out.as_deref(),
        );
        // Runtime entry/liveness lookups expect the byte offset of the
        // surviving `-live-` marker for each Python PC
        // (`jitcode.get_live_vars_info` first checks `code[pc] ==
        // op_live`).  The walker-tracked `walker_pc_live_marker_pos`
        // side-table — translated through the `remove_repeated_live`
        // remap by `filter_liveness_in_place` — is the sole source of
        // per-PC `-live-` positions; assert one entry per Python PC.
        assert_eq!(
            post_remove_live_indices.len(),
            num_instrs,
            "filter_liveness_in_place must return one entry per Python PC; \
             walker-tracked side-table was unavailable"
        );
        let pc_map = post_remove_live_indices.clone();

        // codewriter.py:62-67 num_regs[kind] = max(coloring)+1
        // (or 0 if coloring is empty). Pass through to the Assembler
        // step so `JitCode.num_regs_*` reflect the post-regalloc
        // ceiling rather than the pre-regalloc PyFrame-slot range.
        let num_regs = super::assembler::NumRegs {
            int: alloc_result.num_regs[super::flatten::Kind::Int.index()],
            ref_: alloc_result.num_regs[super::flatten::Kind::Ref.index()],
            float: alloc_result.num_regs[super::flatten::Kind::Float.index()],
        };

        // codewriter.py:67-72 step 4 — assemble the SSARepr into an
        // owned JitCode, translate pc_map insn indices to byte offsets,
        // and stamp the per-graph metadata. See `Self::finalize_jitcode`.
        self.finalize_jitcode(
            assembler,
            ssarepr,
            code,
            w_code,
            pc_map,
            depth_at_pc,
            portal_frame_reg,
            portal_ec_reg,
            has_abort,
            merge_point_pc,
            num_regs,
            stack_slot_color_map,
            pyre_color_for_semantic_local,
        )
    }

    /// RPython: `codewriter.py:62-72` step 4 — produce the
    /// owned `JitCode` from the populated `SSARepr` and stamp the
    /// per-graph metadata.
    ///
    /// ```python
    /// num_regs = {kind: ... for kind in KINDS}
    /// self.assembler.assemble(ssarepr, jitcode, num_regs)
    /// jitcode.index = index
    /// ```
    ///
    /// pyre's combined step:
    ///   - `SSAReprEmitter::finish_with_positions` runs the
    ///     `assembler.py:assemble` analog through the shared
    ///     `self.assembler`, returning the owned `JitCode` plus the
    ///     translated `pc_map` byte offsets.
    ///   - jitdriver_sd / calldescr / fnaddr are stamped onto the
    ///     `JitCode` (call.py:148, call.py:174-187, call.py:167).
    ///   - `PyJitCodeMetadata` is bundled with the ref-count-stable
    ///     `Arc<JitCode>` plus the pyre-only `has_abort` /
    ///     `merge_point_pc` fields into the returned `PyJitCode`.
    fn finalize_jitcode(
        &self,
        mut assembler: SSAReprEmitter,
        ssarepr: SSARepr,
        code: &CodeObject,
        w_code: *const (),
        pc_map: Vec<usize>,
        depth_at_pc: Vec<u16>,
        portal_frame_reg: u16,
        portal_ec_reg: u16,
        has_abort: bool,
        merge_point_pc: Option<usize>,
        num_regs: super::assembler::NumRegs,
        stack_slot_color_map: Vec<u16>,
        pyre_color_for_semantic_local: Vec<u16>,
    ) -> PyJitCode {
        // call.py:167-169 — `(fnaddr, calldescr) = get_jitcode_calldescr(graph);
        // jitcode = JitCode(name, fnaddr, calldescr)`.  Stage the values
        // before assembly so `JitCodeBuilder::finish()` can stamp them
        // alongside the body in a single object construction step,
        // matching the upstream constructor order.  See
        // [`super::call::CallControl::get_jitcode_calldescr`] for the
        // Note rationale of the constant return value.
        let (fnaddr, calldescr) = self
            .callcontrol()
            .get_jitcode_calldescr(code as *const CodeObject);
        assembler.set_fnaddr_and_calldescr(fnaddr, calldescr);

        // pc_map[py_pc] currently holds SSARepr insn indices (returned by
        // SSAReprEmitter::current_pos()). Translate them to JitCode byte
        // offsets via ssarepr.insns_pos, populated during
        // Assembler::assemble (assembler.py:41-44). Runtime readers
        // (get_live_vars_info, resume dispatch) expect byte offsets.
        //
        // `codewriter.py:67` `self.assembler.assemble(ssarepr, jitcode, num_regs)`
        // parity: borrow the CodeWriter's single Assembler so
        // `all_liveness` / `num_liveness_ops` continue to accumulate
        // across every jitcode compiled on this thread.
        let (jitcode, pc_map_bytes) = {
            let mut asm = self.assembler.borrow_mut();
            assembler.finish_with_positions_from(&mut *asm, ssarepr, &pc_map, num_regs)
        };

        // call.py:148 `jd.mainjitcode.jitdriver_sd = jd`. RPython mutates
        // the shell returned by `grab_initial_jitcodes`; pyre still
        // builds the populated `JitCode` as the final codewriter step, so
        // stamp the exact jdindex while constructing that populated object.
        // Non-portals keep the JitCode constructor default of `None`.
        if let Some(idx) = self
            .callcontrol()
            .jitdriver_sd_from_portal_graph(code as *const CodeObject)
        {
            // OnceLock semantics: only the first portal grab sets the
            // index. RPython sets it once at call.py:148 then leaves it
            // for the lifetime of the jitcode.
            if jitcode.jitdriver_sd().is_none() {
                jitcode.set_jitdriver_sd(idx);
            }
        }
        // Per-code stack base in `locals_cells_stack_w`. RPython's JitCode
        // does not carry PyFrame layout data; keep it in PyJitCodeMetadata
        // and attach it to BlackholeInterpreter setup when pyre needs it.
        let frame_stack_base = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);

        let metadata = PyJitCodeMetadata {
            pc_map: pc_map_bytes,
            depth_at_py_pc: depth_at_pc,
            portal_frame_reg,
            portal_ec_reg,
            stack_base: frame_stack_base,
            stack_slot_color_map,
            pyre_color_for_semantic_local,
        };

        PyJitCode::from_parts(
            std::sync::Arc::new(jitcode),
            metadata,
            code as *const CodeObject,
            w_code,
            has_abort,
            merge_point_pc,
        )
    }

    /// RPython: `CodeWriter.make_jitcodes(verbose)` (codewriter.py:74-89).
    ///
    /// ```python
    /// def make_jitcodes(self, verbose=False):
    ///     log.info("making JitCodes...")
    ///     self.callcontrol.grab_initial_jitcodes()
    ///     count = 0
    ///     all_jitcodes = []
    ///     for graph, jitcode in self.callcontrol.enum_pending_graphs():
    ///         self.transform_graph_to_jitcode(graph, jitcode, verbose, len(all_jitcodes))
    ///         all_jitcodes.append(jitcode)
    ///         count += 1
    ///         if not count % 500:
    ///             log.info("Produced %d jitcodes" % count)
    ///     self.assembler.finished(self.callcontrol.callinfocollection)
    ///     log.info("There are %d JitCode instances." % count)
    ///     log.info("There are %d -live- ops. Size of liveness is %s bytes" % (
    ///         self.assembler.num_liveness_ops, self.assembler.all_liveness_length))
    ///     return all_jitcodes
    /// ```
    ///
    /// Each freshly-compiled `PyJitCode` is `Arc`-wrapped before being
    /// inserted into `CallControl.jitcodes`; callers publish the whole
    /// returned list into trace-side `MetaInterpStaticData.jitcodes`,
    /// so both stores reference one allocation — the Rust analog of
    /// RPython's two stores referencing the same Python `JitCode` via
    /// refcount semantics.
    ///
    /// `grab_initial_jitcodes` reads its seed list from
    /// [`super::call::CallControl::jitdrivers_sd`]; callers register
    /// portals with [`Self::setup_jitdriver`] before invoking this
    /// method (matching codewriter.py:74 — `setup_jitdriver` followed
    /// by `make_jitcodes` is the upstream order).
    pub fn make_jitcodes(&self) -> Vec<std::sync::Arc<PyJitCode>> {
        // codewriter.py:75 `log.info("making JitCodes...")` — pyre has no
        // codewriter.py log channel, intentionally elided.

        // codewriter.py:76 `self.callcontrol.grab_initial_jitcodes()`.
        self.callcontrol().grab_initial_jitcodes();
        // codewriter.py:79-84 drain + per-jitcode assemble.
        let all_jitcodes = self.drain_unfinished_graphs();
        // call.py:148 `jd.mainjitcode.jitdriver_sd = jd` — assign
        // jdindex to each portal's populated `PyJitCode` AFTER the
        // drain so we use the actual position in
        // `CallControl.jitdrivers_sd` instead of a hardcoded `Some(0)`.
        self.assign_portal_jitdriver_indices();
        // codewriter.py:86-88 final log lines — elided.
        // codewriter.py:89 `return all_jitcodes`.
        all_jitcodes
    }

    /// Drain `CallControl.unfinished_graphs`.
    ///
    /// RPython's `make_jitcodes` (codewriter.py:79-85) drains the queue
    /// once and then calls `assembler.finished()`. Pyre runs the same
    /// drain from `make_jitcodes` so each batch ends with
    /// `assembler.finished()`
    /// and the matching `setup_indirectcalltargets(asm.indirectcalltargets)`
    /// publish, matching `codewriter.py:85` plus `pyjitpl.py:2262`.
    pub(crate) fn drain_unfinished_graphs(&self) -> Vec<std::sync::Arc<PyJitCode>> {
        let mut all_jitcodes: Vec<std::sync::Arc<PyJitCode>> = Vec::new();
        // codewriter.py:79 `for graph, jitcode in enum_pending_graphs():`.
        loop {
            let popped = self.callcontrol().enum_pending_graphs();
            let Some(code_ptr) = popped else {
                break;
            };
            let (w_code, merge_point_pc) = self
                .callcontrol()
                .queued_graph_inputs(code_ptr)
                .expect("queued graph must still have a cached skeleton");
            // codewriter.py:80 `self.transform_graph_to_jitcode(graph,
            //                     jitcode, verbose, len(all_jitcodes))`.
            //
            // Note: `transform_graph_to_jitcode`
            // still returns a fresh `PyJitCode`, but `publish_jitcode`
            // replaces the cached skeleton's payload in place. That
            // matches RPython's "same JitCode object is filled later"
            // identity flow even after other stores cloned the Arc.
            let pyjitcode =
                self.transform_graph_to_jitcode(unsafe { &*code_ptr }, w_code, merge_point_pc);
            let key = code_ptr as usize;
            let pyjitcode = self.callcontrol().publish_jitcode(key, pyjitcode);
            // codewriter.py:81 `all_jitcodes.append(jitcode)`.
            all_jitcodes.push(pyjitcode);
        }
        // codewriter.py:85 `self.assembler.finished(self.callcontrol.callinfocollection)`.
        self.assembler
            .borrow_mut()
            .finished(&self.callcontrol().callinfocollection);
        self.publish_indirectcalltargets();
        all_jitcodes
    }

    /// `pyjitpl.py:2262`
    /// `self.setup_indirectcalltargets(asm.indirectcalltargets)`.
    ///
    /// RPython wires the codewriter's accumulated assembler set into
    /// `MetaInterpStaticData` during `finish_setup(codewriter, optimizer)`.
    /// pyre publishes the same accumulated set after each drain batch so the
    /// trace-side staticdata stays aligned with the writer's current
    /// `Assembler.indirectcalltargets`.
    fn publish_indirectcalltargets(&self) {
        let targets = self.assembler.borrow().indirectcalltargets_vec();
        pyre_jit_trace::state::setup_indirectcalltargets(targets);
    }

    /// call.py:147-148 follow-up after the drain. `grab_initial_jitcodes`
    /// already binds `jd.mainjitcode = self.get_jitcode(jd.portal_graph)`;
    /// the drain fills that same Arc in place, so this method only refreshes
    /// each jd from `CallControl.jitcodes[portal_graph]`. `finalize_jitcode` stamps
    /// the exact `jitdriver_sd` on the populated runtime `JitCode`; because
    /// `publish_jitcode` replaces the payload in place, this no longer needs
    /// to repair identity after the drain.
    fn assign_portal_jitdriver_indices(&self) {
        let cc = self.callcontrol();
        // Snapshot the (key, jdindex) pairs first so the borrow on
        // `cc.jitdrivers_sd` is released before we mutate `cc.jitcodes`.
        let assignments: Vec<(usize, usize)> = cc
            .jitdrivers_sd
            .iter()
            .enumerate()
            .map(|(idx, jd)| (super::call::CallControl::jitcode_key(jd.portal_graph), idx))
            .collect();
        for (key, idx) in assignments {
            let arc_clone = cc.jitcodes.get(&key).map(std::sync::Arc::clone);
            if let Some(clone) = arc_clone {
                cc.jitdrivers_sd[idx].mainjitcode = Some(clone);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Jump target calculation (RPython: flatten.py link following)
// ---------------------------------------------------------------------------

/// Forward jump target: skip_caches(next_instr) + delta.
/// Must match pyre-interpreter/pyopcode.rs:jump_target_forward.
fn jump_target_forward(
    code: &CodeObject,
    num_instrs: usize,
    next_instr: usize,
    delta: usize,
) -> usize {
    let target = skip_caches(code, next_instr) + delta;
    target.min(num_instrs)
}

/// Single-source-of-truth backward-jump target calculation used by both
/// the loop-header pre-scan (pypy/module/pypyjit/interp_jit.py:103) and
/// the emitter (jtransform.py:1714 `handle_jit_marker__loop_header`).
///
/// Returns the target PC for `JumpBackward` (with `skip_caches` on the
/// next-PC base) and `JumpBackwardNoInterrupt` (direct `py_pc + 1 - delta`
/// arithmetic to match the interpreter's dispatch in pyopcode.rs).
/// Returns `None` for any non-backward-jump opcode.
fn backward_jump_target(
    code: &CodeObject,
    py_pc: usize,
    instr: Instruction,
    op_arg: pyre_interpreter::OpArg,
) -> Option<usize> {
    match instr {
        Instruction::JumpBackward { delta } => {
            Some(skip_caches(code, py_pc + 1).saturating_sub(delta.get(op_arg).as_usize()))
        }
        Instruction::JumpBackwardNoInterrupt { delta } => {
            Some((py_pc + 1).saturating_sub(delta.get(op_arg).as_usize()))
        }
        _ => None,
    }
}

/// Match pyre-interpreter/pyopcode.rs:skip_caches.
fn skip_caches(code: &CodeObject, mut pos: usize) -> usize {
    let mut state = OpArgState::default();
    while pos < code.instructions.len() {
        let (instruction, _) = state.get(code.instructions[pos]);
        if matches!(instruction, Instruction::Cache) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

// ---------------------------------------------------------------------------
// JitCode cache — RPython: `CallControl.get_jitcode` (call.py:155-172).
// The cache + `unfinished_graphs` queue live on `super::call::CallControl`;
// `CallControl::get_jitcode` is the canonical entry point.
// ---------------------------------------------------------------------------

/// Portal entry path: `setup_jitdriver` followed by `make_jitcodes` —
/// the warmspot order at codewriter.py:74-99. RPython runs this once
/// per `@jit_callback` decoration; pyre's portal discovery is lazy,
/// so this adapter fires per JIT entry. `setup_jitdriver` dedups by
/// `portal_graph` so `jitdrivers_sd` stays bounded by the number of
/// unique portals (see [`CodeWriter::setup_jitdriver`] for the
/// Note rationale).
///
/// `make_jitcodes` is then the canonical RPython no-arg call: it
/// pulls its seed list from `CallControl.jitdrivers_sd` and runs
/// `grab_initial_jitcodes` → drain → `assembler.finished()` →
/// `assign_portal_jitdriver_indices`. The resulting list is published
/// whole to trace-side `MetaInterpStaticData`, matching
/// `warmspot.py:281-282`. Runtime trace-side lookup must observe this
/// installed result; it must not compile missing callees lazily.
pub fn register_portal_jitdriver(
    code: &pyre_interpreter::CodeObject,
    w_code: *const (),
    merge_point_pc: Option<usize>,
) {
    let writer = CodeWriter::instance();
    // codewriter.py:96-99 `setup_jitdriver(jd)` — register the
    // portal so `grab_initial_jitcodes` finds it.
    writer.setup_jitdriver(super::call::JitDriverStaticData {
        portal_graph: code as *const pyre_interpreter::CodeObject,
        w_code,
        merge_point_pc,
        // call.py:147 LHS — initial `None` matches RPython's
        // `jd.mainjitcode = None` before `grab_initial_jitcodes`
        // fires; `grab_initial_jitcodes` itself stores the
        // `Arc<PyJitCode>` from `get_jitcode(jd.portal_graph)`.
        mainjitcode: None,
    });
    // codewriter.py:74 `make_jitcodes()` — drain everything pending.
    let jitcodes = writer.make_jitcodes();
    // RPython warmspot.py:281-282 stores the complete
    // `make_jitcodes()` result on MetaInterpStaticData before tracing
    // can observe it. Pyre keeps trace-side SD in a separate crate
    // keyed by W_CodeObject, so install the whole just-drained list at
    // this codewriter boundary. A missing portal entry after the drain
    // is an impossible postcondition and must fail loudly.
    if !jitcodes.is_empty() {
        pyre_jit_trace::state::install_jitcodes(jitcodes);
    }
    let portal_jitcode = writer
        .callcontrol()
        .find_compiled_jitcode_arc(code as *const pyre_interpreter::CodeObject)
        .expect("make_jitcodes must populate the registered portal jitcode");
    assert_eq!(
        portal_jitcode.w_code, w_code,
        "registered portal jitcode must preserve the W_CodeObject identity"
    );
}

/// Callee compile path: `CallControl.get_jitcode(graph)` followed by the
/// pending-graph drain. This is the trace-time adapter for the RPython flow in
/// `jtransform`: regular calls ask `cc.callcontrol.get_jitcode(callee_graph)`,
/// which inserts the callee into `CallControl.jitcodes` and queues it on
/// `unfinished_graphs`; the surrounding `make_jitcodes` drain then fills the
/// same JitCode object.
///
/// # Non-portal callee status (TODO, partial fix in flight)
///
/// Drained graphs whose `code` is NOT registered as a portal go through the
/// same `transform_graph_to_jitcode` body. `is_portal=false` already gates
/// the graph-side dual-writes (vable getfield/setfield → `frame_var` would
/// be an undefined inputarg), and per-helper walker emit sites are being
/// migrated to feed compile-time-known operands directly per
/// `pyframe.py:509-510 getcode(): hint(self.pycode, promote=True)`.
///
/// Per-helper migration status:
///   * `bh_load_const_fn(w_code_ptr, consti)` — **migrated**.  Walker emit
///     in the `Instruction::LoadConst` arm splits on `is_portal`: portal
///     keeps the `getfield_vable_r(portal_frame_reg, code_field)` +
///     register-operand call shape; non-portal skips the vable getfield
///     and emits `build_load_const_fn_residual_call_ir_r_insn_with_const_pycode`
///     with the callee's own `w_code` pointer as `Operand::ConstRef`.
///   * `bh_load_global_fn(namespace_ptr, w_code_ptr, frame_ptr, namei)` —
///     **pycode operand migrated**.  Walker emit in the
///     `Instruction::LoadGlobal` arm splits on `is_portal`: portal keeps
///     the `getfield_vable_r(portal_frame_reg, code_field)` + register
///     operand; non-portal skips that vable getfield and emits
///     `build_load_global_fn_residual_call_ir_r_insn_with_const_pycode`
///     with the callee's own `w_code` pointer as `Operand::ConstRef`.
///     This fixes the function_calls miscompile: in a chained blackhole
///     resume `portal_frame_reg` aliases the OUTERMOST (caller) frame, so
///     `getfield_vable_r(portal_frame_reg, pycode)` returned the caller's
///     `W_Code`, and an inlined callee's `LOAD_GLOBAL` then indexed the
///     caller's `names` table (resolving the wrong global, e.g. calling
///     an int).  Same basis as the migrated LoadConst pycode ConstRef
///     (`pyframe.py:509-510 getcode(): hint(self.pycode, promote=True)`)
///     and safe for the same reason (see "Why LoadConst ConstRef works"
///     below): pycode is a promoted constant whose value equals the
///     pointer `getfield_vable_r` would read, so the QuasiimmutField key
///     matches.
///
///     The `ns` (namespace) and `frame` operands stay register-form and
///     are NOT migrated.  `pyframe.py:49 self.w_globals = w_globals` at
///     frame construction initializes the namespace from
///     `pycode.w_globals`, so its VALUE is derivable from the promoted
///     pycode, but the trace walker reads `frame.w_globals` as a
///     register-form operand (via `GetfieldVableR`) so the optimizer can
///     match it in the known-result cache — a ConstRef would break that
///     fold.  A trial wire of the dormant
///     `build_load_global_fn_residual_call_ir_r_insn_with_all_consts`
///     (all-three ConstRef: namespace from `W_CodeObject.w_globals`,
///     pycode = callee `w_code`, frame = `ConstRef(0)`) was attempted and
///     reverted — **root cause found and CLOSED**.  `_with_all_consts`
///     is doubly unsound: the null frame skips the `get_builtin()`
///     fallback required by `pyopcode.py:957` (breaks `print`/`len`),
///     and non-portal jitcode bytecode IS walked by the trace recorder
///     (`jitcode_dispatch.rs` walker) during callee inlining — its
///     `read_ref_var_list` (jitcode_dispatch.rs:1029) reads
///     `ctx.registers_r[reg]` for each ref operand, so register-form
///     operands produce TRACED variables (results of `GetfieldVableR`
///     IR ops) that participate in the optimizer's
///     `RecordKnownResult`/`QuasiimmutField` fold, while ConstRef
///     operands produce RAW CONSTANT arguments that cannot match the
///     known-result cache, emitting wrong code (mismatched fold →
///     wrong result patched into the portal's resume snapshot →
///     fastlocal corruption on guard failure).
///     **Why LoadConst ConstRef works**: `load_const_fn(pycode, idx)`
///     has pycode as its sole ref arg; the ConstRef value IS the same
///     pointer the walker would read from `getfield_vable_r`, so the
///     QuasiimmutField key matches — and the same holds for the migrated
///     LoadGlobal pycode operand.  **Why LoadGlobal `ns` ConstRef fails**:
///     `load_global_fn(ns, code, frame, namei)` has ns as a ref arg
///     whose traced form (GetfieldVableR result) differs from the raw
///     constant form in the optimizer's value-identity tracking.
///     **Resolution**: migrate only the fold-safe pycode operand to
///     ConstRef; `ns`/`frame` keep the register-form emit
///     (getfield_vable_r + Register operands) so the trace walker
///     produces proper traced variables.  A cross-module inlined
///     `LOAD_GLOBAL` is therefore still latently wrong (ns reads the
///     portal frame's globals); function_calls is single-module so this
///     does not surface.  Fully retiring the register-form `ns`/`frame`
///     reads requires separating the walker path from the blackhole
///     path — a structural prerequisite not yet in place.
///   * `bh_call_fn` / `bh_call_fn_N(frame_ptr, callable, args...)` —
///     `frame_ptr` non-null asserted at runtime entry; used for
///     `bh_call_self_recursive_portal`, `set_last_exec_ctx`, and
///     `call_user_function_plain(parent_frame, ...)`.  Pyre's
///     `bh_call_fn_*` ABI threads `parent_frame` as the first ref argument
///     — RPython equivalent has no such argument.  Non-portal callees do
///     not have a self/caller frame on the graph as a startblock
///     inputarg, so the migration requires either an entry-arg restructure
///     or a ConstRef-null + adapter-side guard.  Not yet migrated.
///   * `bh_normalize_raise_varargs_with_frame(frame_ptr, exc, cause)` —
///     `frame_ptr` non-null asserted; pins
///     `(*parent_frame_ptr).execution_context` for the normalization
///     path.  Same migration shape as `bh_call_fn_*`.  Not yet migrated.
///
/// Today the un-migrated emit sites are latent for non-portal callees:
/// production tracing runs `MIFrame::execute_opcode_step` (the trait
/// dispatch path), not the codewriter-emitted jitcode bytecode.
/// Shadow-walker validation (`pyre-jit-trace/src/shadow_walker.rs`,
/// env-gated `MAJIT_SHADOW_WALKER`) and `inline_call_*` sub-jitcode
/// recursion (`pyre-jit-trace/src/jitcode_dispatch.rs`) are symbolic —
/// they record IR ops rather than invoke `bh_*_fn` runtime helpers.
///
/// Orthodox convergence target: RPython compiles `LOAD_CONST` etc. into
/// inline JitCode ops via `pypy/module/pypyjit/interp_jit.py` +
/// `pypy/interpreter/pyopcode.py` opcode_implementations, not via
/// `residual_call_*` to a Rust-side helper. The pyre helper-fn pattern is
/// a translation shortcut; collapsing it onto the inline JitCode-op shape
/// remains the longer-term endgame. The per-helper ConstRef pycode
/// migration here is the conservative first step that recovers the
/// `hint(promote=True)` semantics for the part of the helper ABI that
/// upstream actually treats as a JIT-time constant.
pub fn compile_jitcode_for_callee(
    code: &pyre_interpreter::CodeObject,
    w_code: *const (),
) -> Vec<std::sync::Arc<PyJitCode>> {
    let writer = CodeWriter::instance();
    // call.py:155-172 `get_jitcode(graph)` — insert skeleton if missing and
    // queue the graph for the drain.
    let _ = writer.callcontrol().get_jitcode(code, w_code, None);
    // codewriter.py:79-85 — drain the queued graph(s), then assembler.finished.
    writer.drain_unfinished_graphs()
}

/// Ensure the writer-owned `PyJitCode` for `w_code` exists and publish the
/// same Arc into trace-side `MetaInterpStaticData.jitcodes`.
///
/// This is the pyre boundary corresponding to RPython's setup-time
/// `make_jitcodes()` handoff: the writer owns `CallControl.jitcodes`, and the
/// trace-side staticdata stores the same populated JitCode objects.
///
/// `call.py:155-172 CallControl.get_jitcode(graph)` keys its dictionary on
/// graph identity; pyre's "graph identity" is the `*const CodeObject`
/// pointer.  The trace recorder already has that pointer in hand when it
/// calls into the writer, so consume it directly and assert it agrees with
/// `w_code`'s embedded code pointer instead of re-deriving the pointer.
pub fn ensure_trace_jitcode_for_w_code(
    raw_code: *const pyre_interpreter::CodeObject,
    w_code: *const (),
) -> Option<std::sync::Arc<PyJitCode>> {
    let pyjit = compile_jitcode_via_raw_code(raw_code, w_code)?;
    pyre_jit_trace::state::install_jitcode_for(w_code, std::sync::Arc::clone(&pyjit));
    Some(pyjit)
}

fn compile_jitcode_via_raw_code(
    raw_code: *const pyre_interpreter::CodeObject,
    w_code: *const (),
) -> Option<std::sync::Arc<PyJitCode>> {
    if raw_code.is_null() || w_code.is_null() {
        return None;
    }
    assert_eq!(
        unsafe {
            pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject
        },
        raw_code,
        "ensure_trace_jitcode_for_w_code: w_code's embedded code pointer must match raw_code",
    );
    let code = unsafe { &*raw_code };
    if let Some(existing) = CodeWriter::instance()
        .callcontrol()
        .find_compiled_jitcode_arc(code as *const _)
    {
        return Some(existing);
    }

    let drained = compile_jitcode_for_callee(code, w_code);
    if !drained.is_empty() {
        pyre_jit_trace::state::install_jitcodes(drained);
    }
    CodeWriter::instance()
        .callcontrol()
        .find_compiled_jitcode_arc(code as *const _)
}

/// Scan `code` for JUMP_BACKWARD targets — the PCs where
/// `transform_graph_to_jitcode` would emit `BC_JUMP_TARGET` and where
/// `jit_merge_point` is evaluated.
///
/// RPython parity: corresponds to `jtransform.py:1714-1718`
/// `handle_jit_marker__loop_header`, which walks the flow graph looking
/// for `jit_marker('loop_header', ...)` operations. pyre's "graph" is
/// raw Python bytecode, so the equivalent scan looks for
/// `JUMP_BACKWARD` instructions and resolves their target PCs.
///
/// Used by `transform_graph_to_jitcode` to decide where loop markers
/// belong. Portal classification itself comes from
/// `CallControl.jitdrivers_sd`, matching codewriter.py:37.
pub fn find_loop_header_pcs(code: &pyre_interpreter::CodeObject) -> VecSet<usize> {
    let num_instrs = code.instructions.len();
    let mut loop_header_pcs: VecSet<usize> = VecSet::new();
    let mut scan_state = OpArgState::default();
    for scan_pc in 0..num_instrs {
        let (scan_instr, scan_arg) = scan_state.get(code.instructions[scan_pc]);
        if let Some(target) = backward_jump_target(code, scan_pc, scan_instr, scan_arg) {
            if target < num_instrs {
                loop_header_pcs.insert(target);
            }
        }
    }
    loop_header_pcs
}

/// All PCs that are block-entry points: PC 0, every forward jump
/// target, every backward jump target, and every exception handler
/// entry.  Mirrors upstream's set of `joinpoints` keys after
/// `flowcontext.py:425-435 set_branch` has fired for every branch
/// instruction (the `mergeblock` candidates list at PC X is
/// non-empty iff PC X is an entry from at least one branch / fall-
/// through / catch edge).  Pre-scanning lets the per-block walker
/// pre-allocate one SpamBlock per boundary at walker entry, retiring
/// the per-PC self-registration that pyre's PC-sequential walker
/// previously needed at every emit_mark_label_pc fall-through arm.
///
/// PC 0 is always included (entry block).  Catch landings come from
/// `code.exception_table` via `frame_blocks_for_offset` consumers;
/// callers thread those in if they need them in the same set, but
/// this scan focuses on the bytecode-derived branch destinations
/// where upstream's `mergeblock` would create candidates.
pub fn find_branch_target_pcs(code: &pyre_interpreter::CodeObject) -> VecSet<usize> {
    let num_instrs = code.instructions.len();
    let mut targets: VecSet<usize> = VecSet::new();
    if num_instrs > 0 {
        targets.insert(0);
    }
    let mut scan_state = OpArgState::default();
    for scan_pc in 0..num_instrs {
        let (scan_instr, scan_arg) = scan_state.get(code.instructions[scan_pc]);
        // Backward jumps reuse the canonical helper.
        if let Some(target) = backward_jump_target(code, scan_pc, scan_instr, scan_arg) {
            if target < num_instrs {
                targets.insert(target);
            }
        }
        // Forward conditional / unconditional jumps.  Targets compute
        // via `jump_target_forward(code, num_instrs, py_pc + 1, delta)`
        // matching the walker's PopJumpIfFalse / PopJumpIfTrue /
        // JumpForward arms.
        let forward_delta = match scan_instr {
            Instruction::PopJumpIfFalse { delta }
            | Instruction::PopJumpIfTrue { delta }
            | Instruction::JumpForward { delta }
            | Instruction::ForIter { delta } => Some(delta.get(scan_arg).as_usize()),
            _ => None,
        };
        if let Some(delta) = forward_delta {
            let target = jump_target_forward(code, num_instrs, scan_pc + 1, delta);
            if target < num_instrs {
                targets.insert(target);
            }
            // The fallthrough PC after a conditional branch is also a
            // boundary (the linktrue / linkfalse fallthrough side).
            // Unconditional JumpForward has no fallthrough but the
            // PC after the next instruction may be reached from
            // elsewhere; including it does not create false boundaries
            // because the subsequent backward-jump scan / next-iteration
            // branch will repopulate it as needed.
            let fallthrough = scan_pc + 1;
            if matches!(
                scan_instr,
                Instruction::PopJumpIfFalse { .. }
                    | Instruction::PopJumpIfTrue { .. }
                    | Instruction::ForIter { .. }
            ) && fallthrough < num_instrs
            {
                targets.insert(fallthrough);
            }
        }
        // Terminator-after pcs are block entries: PCs immediately
        // following ReturnValue / RaiseVarargs / Reraise are reachable
        // only from elsewhere (not from sequential fallthrough), so
        // they are real block entries.  Mirrors upstream's block
        // boundary at every terminator's `next_offset` candidate
        // (`flowcontext.py:407-475 record_block` exits via terminator;
        // the pendingblocks queue picks up the next block).
        if matches!(
            scan_instr,
            Instruction::ReturnValue
                | Instruction::RaiseVarargs { .. }
                | Instruction::Reraise { .. }
        ) {
            let next_pc = scan_pc + 1;
            if next_pc < num_instrs {
                targets.insert(next_pc);
            }
        }
    }
    targets
}

// `liveness_regs_to_u8_sorted` tests removed alongside the helper.
// The 256-register cap is now enforced inside `encode_liveness` and
// covered by `majit_translate::liveness::encode_liveness*` tests.

#[cfg(test)]
mod tests {
    use super::*;
    use super::{
        FrameState, SpamBlockRef, attach_catch_exception_edge, entry_arg_slots, entry_frame_state,
        entry_inputargs, mergeblock, new_shadow_graph,
    };
    use crate::jit::assembler::ArcByPtr;
    use crate::jit::flatten::{Insn, Kind, Operand, Register, SSARepr};
    use crate::jit::flow::{
        Block, Constant, ExitSwitch, FlowValue, FunctionGraph, Link, SpaceOperationArg, Variable,
        VariableId, c_last_exception,
    };
    use pyre_interpreter::bytecode::{CodeObject, ConstantData};
    use pyre_interpreter::compile_exec;
    use std::sync::Arc;

    /// Tail-strip pass folds a `goto + Unreachable` tail into the
    /// following block when the goto's target name matches a leading
    /// `Insn::Label` in that block.
    #[test]
    fn strip_walker_block_boundary_goto_folds_matching_label_tails() {
        use super::super::flatten::{Insn, Label as FlatLabel, Operand, TLabel};

        let target_name = "block_target_7".to_string();
        let goto = Insn::op(
            "goto",
            vec![Operand::TLabel(TLabel::new(target_name.clone()))],
        );
        let block_a = vec![Insn::live(Vec::new()), goto, Insn::Unreachable];
        let block_b = vec![
            Insn::Label(FlatLabel::new(target_name.clone())),
            Insn::live(Vec::new()),
        ];
        let mut blocks = vec![block_a, block_b];

        let drained = super::strip_walker_block_boundary_goto(&mut blocks);

        assert_eq!(
            drained.len(),
            3,
            "expected [live, label, live] after strip, got {drained:?}",
        );
        assert!(matches!(drained[0], Insn::Op { ref opname, .. } if opname == "-live-"));
        assert!(matches!(&drained[1], Insn::Label(l) if l.name == target_name));
        assert!(matches!(drained[2], Insn::Op { ref opname, .. } if opname == "-live-"));

        // The strip should NOT fire when the goto target doesn't
        // match the next block's label.
        let other_name = "block_other_99".to_string();
        let goto_to_other = Insn::op("goto", vec![Operand::TLabel(TLabel::new(other_name))]);
        let block_a = vec![Insn::live(Vec::new()), goto_to_other, Insn::Unreachable];
        let block_b = vec![Insn::Label(FlatLabel::new(target_name))];
        let mut blocks = vec![block_a, block_b];
        let drained = super::strip_walker_block_boundary_goto(&mut blocks);
        assert_eq!(
            drained.len(),
            4,
            "goto/--- must remain when target != next block's label",
        );
    }

    /// `rpython/jit/codewriter/flatten.py:106-155 make_link` falls
    /// through only when the IMMEDIATE next emitted block is the
    /// link's target; it never hops over an intervening non-empty
    /// block.  Lock in that no-hop semantics: with A -> [unrelated B]
    /// -> C and A's goto targeting C's label, the goto must remain
    /// (not strip).  Empty intervening blocks (dead supersede) DO
    /// get skipped because `mark_dead` clears their accumulator
    /// per `flowspace/flowcontext.py:455-457`.
    #[test]
    fn strip_walker_block_boundary_goto_does_not_hop_over_intervening_block() {
        use super::super::flatten::{Insn, Label as FlatLabel, Operand, TLabel};

        let target_name = "block_target_c".to_string();
        let intervening_name = "block_intervening_b".to_string();
        let goto_c = Insn::op(
            "goto",
            vec![Operand::TLabel(TLabel::new(target_name.clone()))],
        );
        let block_a = vec![Insn::live(Vec::new()), goto_c, Insn::Unreachable];
        let block_b = vec![
            Insn::Label(FlatLabel::new(intervening_name)),
            Insn::live(Vec::new()),
        ];
        let block_c = vec![
            Insn::Label(FlatLabel::new(target_name)),
            Insn::live(Vec::new()),
        ];
        let mut blocks = vec![block_a, block_b, block_c];
        let drained = super::strip_walker_block_boundary_goto(&mut blocks);
        assert_eq!(
            drained.len(),
            7,
            "strip must NOT hop over intervening non-empty block B to match C's label, got {drained:?}",
        );
        assert!(
            matches!(&drained[1], Insn::Op { opname, .. } if opname == "goto"),
            "goto must remain after non-strip drain, got {:?}",
            drained[1],
        );
        assert!(matches!(drained[2], Insn::Unreachable));
    }

    /// Empty intervening blocks (dead supersede whose `mark_dead`
    /// cleared `per_block_ssarepr` per `flowspace/flowcontext.py:455-457`)
    /// must be skipped when finding the immediate next emitted block —
    /// they contribute no insns to the final stream, so RPython's
    /// recursive descent effectively sees the next non-empty block as
    /// the fall-through target.
    #[test]
    fn strip_walker_block_boundary_goto_skips_empty_intervening_blocks() {
        use super::super::flatten::{Insn, Label as FlatLabel, Operand, TLabel};

        let target_name = "block_target_c".to_string();
        let goto_c = Insn::op(
            "goto",
            vec![Operand::TLabel(TLabel::new(target_name.clone()))],
        );
        let block_a = vec![Insn::live(Vec::new()), goto_c, Insn::Unreachable];
        let block_b: Vec<Insn> = Vec::new();
        let block_c = vec![
            Insn::Label(FlatLabel::new(target_name)),
            Insn::live(Vec::new()),
        ];
        let mut blocks = vec![block_a, block_b, block_c];
        let drained = super::strip_walker_block_boundary_goto(&mut blocks);
        assert_eq!(
            drained.len(),
            3,
            "strip must fold through empty B to match C, got {drained:?}",
        );
        assert!(matches!(&drained[0], Insn::Op { opname, .. } if opname == "-live-"));
        assert!(matches!(&drained[1], Insn::Label(_)));
        assert!(matches!(&drained[2], Insn::Op { opname, .. } if opname == "-live-"));
    }

    fn make_runtime_jitcode_with_fnaddr(fnaddr: usize) -> Arc<majit_metainterp::jitcode::JitCode> {
        let mut jitcode = majit_metainterp::jitcode::JitCodeBuilder::default().finish();
        jitcode.fnaddr = fnaddr as i64;
        Arc::new(jitcode)
    }

    fn first_nested_function_code(source: &str) -> CodeObject {
        let module = compile_exec(source).expect("compile failed");
        module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } => Some((**code).clone()),
                _ => None,
            })
            .expect("expected nested function code object")
    }

    fn fresh_variable_factory(start: u32) -> impl FnMut(Option<Kind>) -> Variable {
        let mut next_id = start;
        move |kind| {
            let variable = Variable {
                id: VariableId(next_id),
                kind,
            };
            next_id += 1;
            variable
        }
    }

    fn sample_framestate() -> FrameState {
        FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Constant::none().into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            0,
        )
    }

    #[test]
    fn exceptblock_link_args_uses_framestate_exception_pair() {
        let exc_type = Variable::new(VariableId(10), Kind::Ref);
        let exc_value = Variable::new(VariableId(11), Kind::Ref);
        let state = FrameState::new(
            vec![Some(Variable::new(VariableId(0), Kind::Ref).into())],
            Vec::new(),
            Some((exc_type.into(), exc_value.into())),
            Vec::new(),
            0,
        );

        assert_eq!(
            exceptblock_link_args(&state),
            vec![exc_type.into(), exc_value.into()],
        );
    }

    #[test]
    #[should_panic(expected = "exceptblock edge requires materialized exception pair")]
    fn exceptblock_link_args_rejects_missing_exception_pair() {
        let state = sample_framestate();
        let _ = exceptblock_link_args(&state);
    }

    #[test]
    fn explicit_raise_state_records_type_of_raised_value() {
        let code = first_nested_function_code("def f(a):\n    return a\n");
        let mut graph = new_shadow_graph(&code);
        let block = graph.startblock.clone();
        let handler_exc_type = Variable::new(VariableId(20), Kind::Ref);
        let handler_exc_value = Variable::new(VariableId(21), Kind::Ref);
        let raised_value = Variable::new(VariableId(22), Kind::Ref);
        let state = FrameState::new(
            vec![Some(Variable::new(VariableId(0), Kind::Ref).into())],
            Vec::new(),
            Some((handler_exc_type.into(), handler_exc_value.into())),
            Vec::new(),
            0,
        );

        let raised = explicit_raise_state(&mut graph, &block, &state, raised_value.into(), 123);
        let Some((FlowValue::Variable(new_type), FlowValue::Variable(new_value))) =
            raised.last_exception
        else {
            panic!("explicit raise should materialize fresh exception vars");
        };
        assert_ne!(new_type.id, handler_exc_type.id);
        assert_eq!(new_value.id, raised_value.id);

        let block_borrow = block.borrow();
        let Some(op) = block_borrow.operations.last() else {
            panic!("explicit raise should record a graph operation");
        };
        assert_eq!(op.opname, "type");
        assert_eq!(op.offset, 123);
        assert_eq!(op.args, vec![SpaceOperationArg::from(raised_value)]);
        assert_eq!(op.result, Some(new_type.into()));
    }

    #[test]
    fn null_stack_sentinel_is_none_ref_constant() {
        let value = null_stack_sentinel();
        let FlowValue::Constant(constant) = value else {
            panic!("null stack sentinel must be a Constant");
        };
        assert_eq!(constant.kind, Some(Kind::Ref));
        assert!(matches!(
            constant.value,
            crate::jit::flow::ConstantValue::None
        ));
    }

    #[test]
    fn duplicate_shadow_tos_clones_existing_top_value() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("dup_shadow_tos", start, None);
        let top = Variable::new(VariableId(9), Kind::Ref);
        let mut state = FrameState::new(Vec::new(), vec![top.into()], None, Vec::new(), 0);

        let duplicated = duplicate_shadow_tos(&mut graph, &mut state);

        assert_eq!(duplicated, top.into());
        assert_eq!(state.stack, vec![top.into(), top.into()]);
    }

    #[test]
    fn duplicate_shadow_tos_falls_back_to_fresh_ref_when_stack_is_empty() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("dup_shadow_tos_empty", start, None);
        let mut state = FrameState::new(Vec::new(), Vec::new(), None, Vec::new(), 0);

        let duplicated = duplicate_shadow_tos(&mut graph, &mut state);

        let FlowValue::Variable(variable) = duplicated else {
            panic!("empty-stack duplication fallback must synthesize a Variable");
        };
        assert_eq!(variable.kind, Some(Kind::Ref));
        assert_eq!(state.stack, vec![duplicated]);
    }

    #[test]
    fn emit_frontend_neg_records_flowspace_style_unary_op() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("neg", start.clone(), None);
        let operand = Variable::new(VariableId(12), Kind::Ref);

        let result = emit_frontend_neg(&mut graph, &start, operand.into(), 33);

        let block = start.borrow();
        let op = block.operations.last().expect("neg op should be recorded");
        assert_eq!(op.opname, "neg");
        assert_eq!(op.offset, 33);
        assert_eq!(op.args, vec![operand.into()]);
        assert_eq!(op.result, Some(result.into()));
    }

    #[test]
    fn emit_frontend_newlist_records_all_items() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("newlist", start.clone(), None);
        let item0 = Variable::new(VariableId(20), Kind::Ref);
        let item1 = Constant::signed(7);
        let item2 = Variable::new(VariableId(21), Kind::Ref);

        let result = emit_frontend_newlist(
            &mut graph,
            &start,
            vec![item0.into(), item1.clone().into(), item2.into()],
            44,
        );

        let block = start.borrow();
        let op = block
            .operations
            .last()
            .expect("newlist op should be recorded");
        assert_eq!(op.opname, "newlist");
        assert_eq!(op.offset, 44);
        assert_eq!(op.args, vec![item0.into(), item1.into(), item2.into()]);
        assert_eq!(op.result, Some(result.into()));
    }

    #[test]
    fn emit_frontend_newslice_records_three_ref_operands() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("newslice", start.clone(), None);
        let w_start = Variable::new(VariableId(22), Kind::Ref);
        let w_stop = Variable::new(VariableId(23), Kind::Ref);
        let w_step = Variable::new(VariableId(24), Kind::Ref);

        let result = emit_frontend_newslice(
            &mut graph,
            &start,
            w_start.into(),
            w_stop.into(),
            w_step.into(),
            46,
        );

        let block = start.borrow();
        let op = block
            .operations
            .last()
            .expect("newslice op should be recorded");
        assert_eq!(op.opname, "newslice");
        assert_eq!(op.offset, 46);
        assert_eq!(op.args, vec![w_start.into(), w_stop.into(), w_step.into()]);
        assert_eq!(op.result, Some(result.into()));
        assert_eq!(result.kind, Some(Kind::Ref));
    }

    #[test]
    fn emit_frontend_buildslice_shadow_graph_two_arg_synthesizes_none_step() {
        use pyre_interpreter::bytecode::BuildSliceArgCount;
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("build_slice_two", start.clone(), None);
        let w_start = Variable::new(VariableId(25), Kind::Ref);
        let w_stop = Variable::new(VariableId(26), Kind::Ref);

        let result = emit_frontend_buildslice_shadow_graph(
            &mut graph,
            &start,
            BuildSliceArgCount::Two,
            w_start.into(),
            w_stop.into(),
            None,
            47,
        );

        let block = start.borrow();
        let op = block
            .operations
            .last()
            .expect("BUILD_SLICE argc=2 should record newslice");
        assert_eq!(op.opname, "newslice");
        assert_eq!(op.offset, 47);
        assert_eq!(
            op.args,
            vec![w_start.into(), w_stop.into(), Constant::none().into()],
        );
        assert_eq!(op.result, Some(result.into()));
        assert_eq!(result.kind, Some(Kind::Ref));
    }

    #[test]
    fn emit_frontend_buildslice_shadow_graph_three_arg_preserves_step() {
        use pyre_interpreter::bytecode::BuildSliceArgCount;
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("build_slice_three", start.clone(), None);
        let w_start = Variable::new(VariableId(27), Kind::Ref);
        let w_stop = Variable::new(VariableId(28), Kind::Ref);
        let w_step = Variable::new(VariableId(29), Kind::Ref);

        let result = emit_frontend_buildslice_shadow_graph(
            &mut graph,
            &start,
            BuildSliceArgCount::Three,
            w_start.into(),
            w_stop.into(),
            Some(w_step.into()),
            48,
        );

        let block = start.borrow();
        let op = block
            .operations
            .last()
            .expect("BUILD_SLICE argc=3 should record newslice");
        assert_eq!(op.opname, "newslice");
        assert_eq!(op.offset, 48);
        assert_eq!(op.args, vec![w_start.into(), w_stop.into(), w_step.into()]);
        assert_eq!(op.result, Some(result.into()));
        assert_eq!(result.kind, Some(Kind::Ref));
    }

    #[test]
    fn emit_frontend_setitem_records_flowspace_style_store_subscr() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("setitem", start.clone(), None);
        let obj = Variable::new(VariableId(30), Kind::Ref);
        let key = Constant::signed(2);
        let value = Variable::new(VariableId(31), Kind::Ref);

        emit_frontend_setitem(
            &mut graph,
            &start,
            obj.into(),
            key.clone().into(),
            value.into(),
            55,
        );

        let block = start.borrow();
        let op = block
            .operations
            .last()
            .expect("setitem op should be recorded");
        assert_eq!(op.opname, "setitem");
        assert_eq!(op.offset, 55);
        assert_eq!(op.args, vec![obj.into(), key.into(), value.into()]);
        assert_eq!(op.result, None);
    }

    #[test]
    fn emit_frontend_setattr_records_flowspace_style_store_attr() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("setattr", start.clone(), None);
        let obj = Variable::new(VariableId(32), Kind::Ref);
        let name = Constant::string("field");
        let value = Variable::new(VariableId(33), Kind::Ref);

        emit_frontend_setattr(
            &mut graph,
            &start,
            obj.into(),
            name.clone().into(),
            value.into(),
            56,
        );

        let block = start.borrow();
        let op = block
            .operations
            .last()
            .expect("setattr op should be recorded");
        assert_eq!(op.opname, "setattr");
        assert_eq!(op.offset, 56);
        assert_eq!(op.args, vec![obj.into(), name.into(), value.into()]);
        assert_eq!(op.result, None);
    }

    #[test]
    fn emit_frontend_getattr_records_flowspace_style_load_attr() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("getattr", start.clone(), None);
        let obj = Variable::new(VariableId(34), Kind::Ref);
        let name = Constant::string("field");

        let result = emit_frontend_getattr(&mut graph, &start, obj.into(), name.clone().into(), 57);

        let block = start.borrow();
        let op = block
            .operations
            .last()
            .expect("getattr op should be recorded");
        assert_eq!(op.opname, "getattr");
        assert_eq!(op.offset, 57);
        assert_eq!(op.args, vec![obj.into(), name.into()]);
        assert_eq!(op.result, Some(result.into()));
    }

    #[test]
    fn emit_frontend_binary_uses_flowspace_operator_name() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("binary", start.clone(), None);
        let lhs = Variable::new(VariableId(40), Kind::Ref);
        let rhs = Variable::new(VariableId(41), Kind::Ref);

        let result = emit_frontend_binary(
            &mut graph,
            &start,
            pyre_interpreter::bytecode::BinaryOperator::InplaceAdd,
            lhs.into(),
            rhs.into(),
            66,
        );

        let block = start.borrow();
        let op = block
            .operations
            .last()
            .expect("binary op should be recorded");
        assert_eq!(op.opname, "inplace_add");
        assert_eq!(op.offset, 66);
        assert_eq!(op.args, vec![lhs.into(), rhs.into()]);
        assert_eq!(op.result, Some(result.into()));
    }

    #[test]
    fn emit_frontend_compare_uses_flowspace_compare_name() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("compare", start.clone(), None);
        let lhs = Variable::new(VariableId(50), Kind::Ref);
        let rhs = Variable::new(VariableId(51), Kind::Ref);

        let result = emit_frontend_compare(
            &mut graph,
            &start,
            pyre_interpreter::bytecode::ComparisonOperator::LessOrEqual,
            lhs.into(),
            rhs.into(),
            77,
        );

        let block = start.borrow();
        let op = block
            .operations
            .last()
            .expect("compare op should be recorded");
        assert_eq!(op.opname, "le");
        assert_eq!(op.offset, 77);
        assert_eq!(op.args, vec![lhs.into(), rhs.into()]);
        assert_eq!(op.result, Some(result.into()));
    }

    #[test]
    fn emit_frontend_bool_records_flowspace_truth_op() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("bool", start.clone(), None);
        let operand = Variable::new(VariableId(52), Kind::Ref);

        let result = emit_frontend_bool(&mut graph, &start, operand.into(), 78);

        assert_eq!(result.kind, Some(Kind::Int));
        let block = start.borrow();
        let op = block.operations.last().expect("bool op should be recorded");
        assert_eq!(op.opname, "bool");
        assert_eq!(op.offset, 78);
        assert_eq!(op.args, vec![operand.into()]);
        assert_eq!(op.result, Some(result.into()));
    }

    #[test]
    fn set_last_bool_exitcase_updates_latest_link() {
        let start = Block::shared(Vec::new());
        let target = Block::shared(Vec::new());
        let link = Link::new(Vec::new(), Some(target), None).into_ref();
        start.closeblock(vec![link.clone()]);

        set_last_bool_exitcase(&start, true);

        let link_borrow = link.borrow();
        assert_eq!(link_borrow.exitcase, Some(Constant::bool(true).into()));
        assert_eq!(link_borrow.llexitcase, Some(Constant::bool(true).into()));
    }

    #[test]
    fn frontend_load_const_flow_value_returns_ref_constant() {
        let code = compile_exec("x = 'hello'\n").expect("compile failed");
        let idx = code
            .constants
            .iter()
            .position(|constant| matches!(constant, ConstantData::Str { .. }))
            .expect("string constant");

        let value = frontend_load_const_flow_value(&code, idx);

        match value {
            FlowValue::Constant(c) => {
                assert_eq!(c.kind, Some(Kind::Ref));
                assert!(
                    matches!(c.value, super::super::flow::ConstantValue::Signed(ptr) if ptr != 0)
                );
            }
            other => panic!("LOAD_CONST graph value must be a Ref constant, got {other:?}"),
        }
    }

    #[test]
    fn frontend_global_flow_value_reads_w_code_globals_as_constant() {
        let code = compile_exec("x\n").expect("compile failed");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let mut globals = Box::new(pyre_interpreter::DictStorage::new());
        let w_value = pyre_object::intobject::w_int_new(42);
        pyre_interpreter::dict_storage_store(globals.as_mut(), "x", w_value);
        let globals_ptr = Box::into_raw(globals);
        unsafe {
            pyre_interpreter::w_code_set_w_globals(w_code, globals_ptr);
        }

        let value = frontend_global_flow_value(w_code as *const (), "x").expect("global constant");

        assert_eq!(value, pyobject_const_ref_value(w_value));
    }

    #[test]
    fn emit_frontend_simple_call_records_callable_then_args() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("simple_call", start.clone(), None);
        let callable = Variable::new(VariableId(60), Kind::Ref);
        let arg0 = Variable::new(VariableId(61), Kind::Ref);
        let arg1 = Constant::signed(9);

        let result = emit_frontend_simple_call(
            &mut graph,
            &start,
            callable.into(),
            vec![arg0.into(), arg1.clone().into()],
            88,
        );

        let block = start.borrow();
        let op = block
            .operations
            .last()
            .expect("simple_call op should be recorded");
        assert_eq!(op.opname, "simple_call");
        assert_eq!(op.offset, 88);
        assert_eq!(op.args, vec![callable.into(), arg0.into(), arg1.into()]);
        assert_eq!(op.result, Some(result.into()));
    }

    /// Regression: `FrameState::mergeable_index_of` locates
    /// a Variable by its `VariableId` across locals / stack / last-exc
    /// positions and returns `None` for non-existent ids or non-Variable
    /// FlowValues.  Mirrors `framestate.py:38-43` `mergeable()` layout.
    #[test]
    fn mergeable_index_of_finds_variables_across_locals_stack_and_last_exc() {
        let v_local = Variable::new(VariableId(0), Kind::Ref);
        let v_stack = Variable::new(VariableId(1), Kind::Int);
        let v_exc_type = Variable::new(VariableId(2), Kind::Int);
        let v_exc_value = Variable::new(VariableId(3), Kind::Ref);
        let state = FrameState::new(
            vec![Some(v_local.into()), Some(Constant::none().into())],
            vec![v_stack.into()],
            Some((v_exc_type.into(), v_exc_value.into())),
            Vec::new(),
            0,
        );

        // Local at mergeable[0]; Constant at [1] has no Variable id.
        assert_eq!(state.mergeable_index_of(&v_local), Some(0));
        // Stack pushed after locals_w: len(locals_w) == 2, so stack[0] is at [2].
        assert_eq!(state.mergeable_index_of(&v_stack), Some(2));
        // last_exception pair sits at the end.
        assert_eq!(state.mergeable_index_of(&v_exc_type), Some(3));
        assert_eq!(state.mergeable_index_of(&v_exc_value), Some(4));
        // Unknown VariableId returns None.
        let v_absent = Variable::new(VariableId(99), Kind::Ref);
        assert_eq!(state.mergeable_index_of(&v_absent), None);
    }

    /// Regression: `FrameState::mergeable_index_to_slot`
    /// is identity in the regular `[0, locals_w.len() + stack.len())`
    /// range and returns `None` for the `last_exception` pair.
    #[test]
    fn mergeable_index_to_slot_is_identity_in_regular_range() {
        let v_local = Variable::new(VariableId(0), Kind::Ref);
        let v_stack = Variable::new(VariableId(1), Kind::Int);
        let v_exc_type = Variable::new(VariableId(2), Kind::Int);
        let v_exc_value = Variable::new(VariableId(3), Kind::Ref);
        let state = FrameState::new(
            vec![Some(v_local.into()), Some(Constant::none().into())],
            vec![v_stack.into()],
            Some((v_exc_type.into(), v_exc_value.into())),
            Vec::new(),
            0,
        );

        // Regular range: identity. locals_w.len() + stack.len() = 2 + 1 = 3.
        assert_eq!(state.mergeable_index_to_slot(0), Some(0));
        assert_eq!(state.mergeable_index_to_slot(1), Some(1));
        assert_eq!(state.mergeable_index_to_slot(2), Some(2));
        // last_exception pair at mergeable[3..5) has no register slot.
        assert_eq!(state.mergeable_index_to_slot(3), None);
        assert_eq!(state.mergeable_index_to_slot(4), None);
        // Anything beyond mergeable: also None.
        assert_eq!(state.mergeable_index_to_slot(100), None);
    }

    /// Regression: `variable_slot` composes the two lookups so
    /// a Variable resolves directly to its register slot.  last_exc
    /// Variables resolve to `None` even though they DO appear in
    /// `mergeable()`.
    #[test]
    fn variable_slot_resolves_locals_and_stack_but_not_last_exc() {
        let v_local = Variable::new(VariableId(0), Kind::Ref);
        let v_stack = Variable::new(VariableId(1), Kind::Int);
        let v_exc_type = Variable::new(VariableId(2), Kind::Int);
        let v_exc_value = Variable::new(VariableId(3), Kind::Ref);
        let state = FrameState::new(
            vec![Some(v_local.into())],
            vec![v_stack.into()],
            Some((v_exc_type.into(), v_exc_value.into())),
            Vec::new(),
            0,
        );

        // Local at mergeable[0] → slot 0.  Stack at mergeable[1] → slot 1.
        assert_eq!(state.variable_slot(&v_local), Some(0));
        assert_eq!(state.variable_slot(&v_stack), Some(1));
        // last_exception variables: present in mergeable but no slot.
        assert_eq!(state.variable_slot(&v_exc_type), None);
        assert_eq!(state.variable_slot(&v_exc_value), None);
        // Absent variable: None.
        let v_absent = Variable::new(VariableId(99), Kind::Ref);
        assert_eq!(state.variable_slot(&v_absent), None);
    }

    #[test]
    fn filter_liveness_drops_non_lv_live_colors_from_live_r() {
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let live_vars = pyre_jit_trace::state::liveness_for(&code as *const _);
        let reachable_pc = (0..code.instructions.len())
            .find(|&py_pc| live_vars.is_reachable(py_pc))
            .expect("compiled code must have a reachable pc");

        // One `-live-` marker per Python PC, recorded in walker-
        // tracked indices.  filter_liveness_in_place protects each
        // recorded position from the `remove_repeated_live` merge.
        let mut ssarepr = SSARepr::new("t");
        let mut walker_tracked_pc_live_indices: Vec<usize> =
            Vec::with_capacity(code.instructions.len());
        for _py_pc in 0..code.instructions.len() {
            walker_tracked_pc_live_indices.push(ssarepr.insns.len());
            ssarepr.insns.push(Insn::live(vec![
                Operand::Register(Register::new(Kind::Ref, 0)),
                Operand::Register(Register::new(Kind::Ref, 7)),
                Operand::Register(Register::new(Kind::Int, 3)),
            ]));
        }

        let depth_at_pc: Vec<u16> = vec![0; code.instructions.len()];
        let local_color_map: Vec<u16> = (0..code.varnames.len() as u16).collect();
        let stack_slot_color_map: Vec<u16> = Vec::new();
        let post_remove_live_indices = filter_liveness_in_place(
            &mut ssarepr,
            &code,
            &depth_at_pc,
            &local_color_map,
            &stack_slot_color_map,
            u16::MAX,
            u16::MAX,
            Some(&walker_tracked_pc_live_indices),
        );

        let live_idx = post_remove_live_indices[reachable_pc];
        let live_args = ssarepr.insns[live_idx]
            .live_args()
            .expect("reachable pc must keep a -live- marker");
        let refs: std::collections::BTreeSet<u16> = live_args
            .iter()
            .filter_map(|op| match op {
                Operand::Register(reg) if reg.kind == Kind::Ref => Some(reg.index),
                _ => None,
            })
            .collect();
        assert!(
            !refs.contains(&7),
            "scratch-stand-in color 7 must be dropped by LV∩SSA retain",
        );
        let ints: std::collections::BTreeSet<u16> = live_args
            .iter()
            .filter_map(|op| match op {
                Operand::Register(reg) if reg.kind == Kind::Int => Some(reg.index),
                _ => None,
            })
            .collect();
        assert_eq!(
            ints,
            std::collections::BTreeSet::from([3]),
            "Int bank must be untouched by the Ref-only filter",
        );
    }

    #[test]
    fn publish_indirectcalltargets_updates_trace_staticdata() {
        let writer = CodeWriter::new();
        let j100 = make_runtime_jitcode_with_fnaddr(0x100);
        let j200 = make_runtime_jitcode_with_fnaddr(0x200);

        {
            let mut assembler = writer.assembler.borrow_mut();
            assembler
                .indirectcalltargets
                .insert(ArcByPtr::new(j100.clone()));
            assembler
                .indirectcalltargets
                .insert(ArcByPtr::new(j200.clone()));
        }

        writer.publish_indirectcalltargets();

        let hit_100 = pyre_jit_trace::state::bytecode_for_address(0x100)
            .expect("fnaddr 0x100 must be published to trace staticdata");
        let hit_200 = pyre_jit_trace::state::bytecode_for_address(0x200)
            .expect("fnaddr 0x200 must be published to trace staticdata");
        assert!(Arc::ptr_eq(&hit_100, &j100));
        assert!(Arc::ptr_eq(&hit_200, &j200));
        assert!(pyre_jit_trace::state::bytecode_for_address(0x300).is_none());
    }

    #[test]
    fn get_jitcode_queues_canonical_raw_graph_only() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let code_ref = unsafe { &*raw_code };

        let _ = writer
            .callcontrol()
            .get_jitcode(code_ref, w_code as *const (), Some(11));

        let queued = writer
            .callcontrol()
            .enum_pending_graphs()
            .expect("fresh jitcode must queue one graph");
        let (queued_w_code, queued_merge_point_pc) = writer
            .callcontrol()
            .queued_graph_inputs(raw_code)
            .expect("queued graph must still have a cached skeleton");

        assert_eq!(queued, raw_code);
        assert_eq!(queued_w_code, w_code as *const ());
        assert_eq!(queued_merge_point_pc, Some(11));
    }

    #[test]
    fn drain_unfinished_graphs_preserves_unique_pyjitcode_identity() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let code_ref = unsafe { &*raw_code };
        let key = raw_code as usize;

        let _ = writer
            .callcontrol()
            .get_jitcode(code_ref, w_code as *const (), None);
        let skeleton_ptr = {
            let slot = writer
                .callcontrol()
                .jitcodes
                .get(&key)
                .expect("skeleton jitcode must be cached");
            Arc::as_ptr(slot)
        };
        let skeleton_runtime_ptr = {
            let slot = writer
                .callcontrol()
                .jitcodes
                .get(&key)
                .expect("skeleton jitcode must be cached");
            Arc::as_ptr(&slot.jitcode)
        };
        let descriptor_view = {
            let slot = writer
                .callcontrol()
                .jitcodes
                .get(&key)
                .expect("skeleton jitcode must be cached");
            slot.jitcode.clone()
        };
        assert!(
            descriptor_view.code.is_empty(),
            "setup-time descriptor clone starts from the unassembled shell"
        );

        let all_jitcodes = writer.drain_unfinished_graphs();
        let populated_ptr = {
            let slot = writer
                .callcontrol()
                .jitcodes
                .get(&key)
                .expect("populated jitcode must remain cached");
            Arc::as_ptr(slot)
        };
        let populated_runtime_ptr = {
            let slot = writer
                .callcontrol()
                .jitcodes
                .get(&key)
                .expect("populated jitcode must remain cached");
            Arc::as_ptr(&slot.jitcode)
        };

        let all_ptrs: Vec<*const PyJitCode> =
            all_jitcodes.iter().map(std::sync::Arc::as_ptr).collect();
        assert_eq!(all_ptrs, vec![populated_ptr]);
        assert_eq!(
            populated_ptr, skeleton_ptr,
            "unique skeleton Arc should be filled in place"
        );
        assert_eq!(
            populated_runtime_ptr, skeleton_runtime_ptr,
            "runtime JitCode allocation must also be filled in place so inline_call descrs keep parity with RPython"
        );
        assert_eq!(
            Arc::as_ptr(&descriptor_view),
            skeleton_runtime_ptr,
            "setup-time inline_call descriptors must keep pointing at the same runtime JitCode shell"
        );
        assert!(
            !descriptor_view.code.is_empty(),
            "filling the runtime JitCode shell in place must update pre-existing descriptor clones"
        );
        let pyjit = writer.callcontrol().find_jitcode(raw_code).unwrap();
        assert!(
            !pyjit.metadata.pc_map.is_empty(),
            "drain must populate bytecode metadata on the existing entry"
        );
        assert_eq!(pyjit.w_code, w_code as *const ());
    }

    #[test]
    fn entry_arg_slots_counts_kwonly_varargs_and_varkeywords() {
        let code =
            first_nested_function_code("def f(a, b, *args, c, d, **kw):\n    return a + b\n");

        assert_eq!(entry_arg_slots(&code), 6);
    }

    #[test]
    fn new_shadow_graph_uses_entry_inputargs_as_startblock_shape() {
        let code =
            first_nested_function_code("def f(a, b, *args, c, d, **kw):\n    return a + b\n");

        let expected_inputargs = entry_inputargs(&code);
        let graph = new_shadow_graph(&code);
        let startblock = graph.startblock.borrow();
        let returnblock = graph.returnblock.borrow();

        assert_eq!(graph.name, "f");
        assert_eq!(startblock.inputargs, expected_inputargs);
        assert_eq!(startblock.inputargs.len(), 6);
        for (idx, value) in startblock.inputargs.iter().enumerate() {
            match value {
                FlowValue::Variable(variable) => {
                    assert_eq!(variable.id, VariableId(idx as u32));
                    assert_eq!(variable.kind, Some(Kind::Ref));
                }
                other => panic!("expected variable inputarg, got {other:?}"),
            }
        }

        assert_eq!(returnblock.inputargs.len(), 1);
        match &returnblock.inputargs[0] {
            FlowValue::Variable(variable) => {
                assert_eq!(variable.id, VariableId(6));
                assert_eq!(variable.kind, Some(Kind::Ref));
            }
            other => panic!("expected variable return arg, got {other:?}"),
        }
    }

    #[test]
    fn graph_entry_inputargs_append_portal_frame_and_ec() {
        let code = first_nested_function_code(
            "def f(a):\n    while a:\n        a = a - 1\n    return a\n",
        );

        let inputargs = graph_entry_inputargs(&code, FrameInputs::Portal);
        let arg_slots = entry_arg_slots(&code);
        assert_eq!(inputargs.len(), arg_slots + 2);
        match &inputargs[arg_slots] {
            FlowValue::Variable(variable) => {
                assert_eq!(*variable, portal_graph_inputvars(&code).0);
                assert_eq!(variable.kind, Some(Kind::Ref));
            }
            other => panic!("expected portal frame variable, got {other:?}"),
        }
        match &inputargs[arg_slots + 1] {
            FlowValue::Variable(variable) => {
                assert_eq!(*variable, portal_graph_inputvars(&code).1);
                assert_eq!(variable.kind, Some(Kind::Ref));
            }
            other => panic!("expected portal ec variable, got {other:?}"),
        }
    }

    #[test]
    fn portal_shadow_graph_reserves_return_var_after_frame_and_ec() {
        let code = first_nested_function_code(
            "def f(a):\n    while a:\n        a = a - 1\n    return a\n",
        );

        let graph = new_shadow_graph_with_portal_inputs(&code, FrameInputs::Portal);
        let startblock = graph.startblock.borrow();
        let returnblock = graph.returnblock.borrow();

        assert_eq!(
            startblock.inputargs,
            graph_entry_inputargs(&code, FrameInputs::Portal)
        );
        match &returnblock.inputargs[0] {
            FlowValue::Variable(variable) => {
                assert_eq!(
                    variable.id,
                    VariableId(graph_entry_inputargs(&code, FrameInputs::Portal).len() as u32)
                );
                assert_eq!(variable.kind, Some(Kind::Ref));
            }
            other => panic!("expected variable return arg, got {other:?}"),
        }
    }

    #[test]
    fn nonportal_shadow_graph_appends_frame_inputarg_only() {
        // Epic #245 Slice 1 keystone: a non-portal callee jitcode carries
        // the universal `self` red frame as a real startblock inputarg
        // (`FrameInputs::Frame`) — one extra slot, NOT the portal's
        // `(frame, ec)` pair — and `return_var` shifts past it so it no
        // longer aliases `frame_var` (the prior collision that forced
        // `LoweringContext::portal_frame_var = None`).
        let code = first_nested_function_code(
            "def f(a):\n    while a:\n        a = a - 1\n    return a\n",
        );
        let arg_slots = entry_arg_slots(&code);
        let (frame_var, _ec_var) = portal_graph_inputvars(&code);

        // graph_entry_inputargs appends frame only (no ec).
        let inputargs = graph_entry_inputargs(&code, FrameInputs::Frame);
        assert_eq!(inputargs.len(), arg_slots + 1);
        match &inputargs[arg_slots] {
            FlowValue::Variable(variable) => {
                assert_eq!(*variable, frame_var);
                assert_eq!(variable.kind, Some(Kind::Ref));
            }
            other => panic!("expected frame variable, got {other:?}"),
        }

        // return_var moves to `arg_slots + 1`; the `frame_var.id ==
        // return_var.id` collision is dissolved.
        let graph = new_shadow_graph_with_portal_inputs(&code, FrameInputs::Frame);
        let startblock = graph.startblock.borrow();
        let returnblock = graph.returnblock.borrow();
        assert_eq!(startblock.inputargs, inputargs);
        match &returnblock.inputargs[0] {
            FlowValue::Variable(variable) => {
                assert_eq!(variable.id, VariableId((arg_slots + 1) as u32));
                assert_ne!(variable.id, frame_var.id);
            }
            other => panic!("expected variable return arg, got {other:?}"),
        }

        // entry_frame_state threads frame through portal_extras (ec
        // absent), so getvariables() = entry locals + [frame].
        let state = entry_frame_state(&code, FrameInputs::Frame);
        let mut expected = entry_inputargs(&code);
        expected.push(frame_var.into());
        assert_eq!(state.getvariables(), expected);
    }

    #[test]
    fn portal_jit_merge_point_graph_args_match_upstream_shape() {
        let code = first_nested_function_code(
            "def f(a):\n    while a:\n        a = a - 1\n    return a\n",
        );
        let graph = new_shadow_graph_with_portal_inputs(&code, FrameInputs::Portal);
        let pycode_var = Variable::new(VariableId(9999), Kind::Ref);
        let args = portal_jit_merge_point_graph_args(&graph, 17, pycode_var, 7);

        assert_eq!(args.len(), 7);
        match &args[0] {
            SpaceOperationArg::Value(FlowValue::Constant(constant)) => {
                assert_eq!(constant, &Constant::signed(7));
            }
            other => panic!("expected jdindex constant, got {other:?}"),
        }
        match &args[1] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Int);
                assert_eq!(
                    list.content,
                    vec![Constant::signed(17).into(), Constant::signed(0).into()]
                );
            }
            other => panic!("expected greens int list, got {other:?}"),
        }
        match &args[2] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Ref);
                assert_eq!(list.content.len(), 1);
                match &list.content[0] {
                    FlowValue::Variable(variable) => {
                        assert_eq!(variable.id, pycode_var.id);
                        assert_eq!(variable.kind, Some(Kind::Ref));
                    }
                    other => panic!("expected pycode ref variable, got {other:?}"),
                }
            }
            other => panic!("expected greens ref list, got {other:?}"),
        }
        match &args[3] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Float);
                assert!(list.content.is_empty());
            }
            other => panic!("expected empty greens float list, got {other:?}"),
        }
        match &args[4] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Int);
                assert!(list.content.is_empty());
            }
            other => panic!("expected empty reds int list, got {other:?}"),
        }
        match &args[5] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Ref);
                assert_eq!(
                    list.content,
                    vec![
                        portal_graph_inputvars(&code).0.into(),
                        portal_graph_inputvars(&code).1.into(),
                    ]
                );
            }
            other => panic!("expected reds ref list, got {other:?}"),
        }
        match &args[6] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Float);
                assert!(list.content.is_empty());
            }
            other => panic!("expected empty reds float list, got {other:?}"),
        }
    }

    #[test]
    fn entry_frame_state_matches_pygraph_locals_shape() {
        let code =
            first_nested_function_code("def f(a, b, *args, c, d, **kw):\n    return a + b\n");
        let state = entry_frame_state(&code, FrameInputs::None);

        assert_eq!(state.locals_w.len(), code.varnames.len());
        assert_eq!(state.getvariables(), entry_inputargs(&code));
        assert!(state.stack.is_empty());
        assert!(state.last_exception.is_none());
    }

    #[test]
    fn frame_blocks_follow_exception_table_ranges() {
        let code = first_nested_function_code(
            "def f(a):\n    try:\n        return a\n    except Exception:\n        return 0\n",
        );
        // `exception_table::decode_exceptiontable` yields byte offsets;
        // codewriter operates in code-unit indices (offset/2).
        let entries: Vec<_> =
            pyre_interpreter::exception_table::decode_exceptiontable(&code.exceptiontable)
                .collect();
        assert!(!entries.is_empty());

        let first = &entries[0];
        let first_start_units = first.start as usize / 2;
        let blocks = frame_blocks_for_offset(&code, first_start_units);

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].start_offset, first_start_units);
        assert_eq!(blocks[0].end_offset, first.end as usize / 2);
        assert_eq!(blocks[0].handler_offset, first.target as usize / 2);
        assert_eq!(blocks[0].stack_depth, first.depth as u16);
        assert_eq!(blocks[0].push_lasti, first.lasti);
    }

    #[test]
    fn framestate_copy_refreshes_variables() {
        let state = sample_framestate();
        let mut fresh = fresh_variable_factory(10);
        let copied = state.copy(&mut fresh);

        assert!(state.matches(&copied));
        assert_ne!(state, copied);
        assert_eq!(copied.locals_w[1], Some(Constant::none().into()));
    }

    #[test]
    #[should_panic]
    fn framestate_matches_asserts_on_different_next_offset() {
        let left = sample_framestate();
        let right = FrameState::new(
            left.locals_w.clone(),
            left.stack.clone(),
            left.last_exception.clone(),
            left.blocklist.clone(),
            1,
        );

        let _ = left.matches(&right);
    }

    #[test]
    fn framestate_union_generalizes_different_constants() {
        let state1 = sample_framestate();
        let state2 = FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Constant::signed(42).into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let mut fresh = fresh_variable_factory(20);
        let union = state1
            .union(&state2, &mut fresh)
            .expect("union should succeed");

        match union.locals_w[1].as_ref() {
            Some(FlowValue::Variable(variable)) => assert_eq!(variable.id, VariableId(20)),
            other => panic!("expected generalized variable, got {other:?}"),
        }
    }

    #[test]
    fn framestate_union_matches_more_general_variable_state() {
        let state1 = sample_framestate();
        let state2 = FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Variable::new(VariableId(5), Kind::Ref).into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let mut fresh = fresh_variable_factory(20);
        let union = state1
            .union(&state2, &mut fresh)
            .expect("union should succeed");

        assert!(union.matches(&state2));
    }

    #[test]
    fn framestate_getoutputargs_follows_target_variables() {
        let state1 = sample_framestate();
        let state2 = FrameState::new(
            vec![
                Some(Variable::new(VariableId(1), Kind::Ref).into()),
                Some(Variable::new(VariableId(2), Kind::Ref).into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );

        assert_eq!(
            state1.getoutputargs(&state2),
            vec![
                Variable::new(VariableId(0), Kind::Ref).into(),
                Constant::none().into(),
            ]
        );
    }

    #[test]
    fn callcontrol_compiled_lookup_ignores_skeleton_pyjitcode() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let code_ref = unsafe { &*raw_code };

        let _ = writer
            .callcontrol()
            .get_jitcode(code_ref, w_code as *const (), None);
        assert!(writer.callcontrol().find_jitcode_arc(raw_code).is_some());
        assert!(
            writer
                .callcontrol()
                .find_compiled_jitcode_arc(raw_code)
                .is_none(),
            "fresh shells must not be treated as populated jitcodes"
        );

        writer.drain_unfinished_graphs();
        assert!(
            writer
                .callcontrol()
                .find_compiled_jitcode_arc(raw_code)
                .is_some(),
            "drained jitcodes must become visible through the compiled-only lookup"
        );
    }

    #[test]
    fn grab_initial_jitcodes_binds_mainjitcode_immediately() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 4\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            w_code: w_code as *const (),
            merge_point_pc: None,
            mainjitcode: None,
        });

        writer.callcontrol().grab_initial_jitcodes();

        let cached = writer
            .callcontrol()
            .find_jitcode_arc(code_ptr)
            .expect("grab_initial_jitcodes must insert a jitcode shell");
        let mainjitcode = writer.callcontrol().jitdrivers_sd[0]
            .mainjitcode
            .as_ref()
            .expect("grab_initial_jitcodes must bind jd.mainjitcode");
        assert!(
            Arc::ptr_eq(mainjitcode, &cached),
            "call.py:147 requires jd.mainjitcode to be the get_jitcode return object"
        );
        assert!(
            mainjitcode.is_skeleton(),
            "call.py:147 binds the shell before codewriter.py:80 fills it"
        );
    }

    #[test]
    fn make_jitcodes_fills_mainjitcode_payload_in_place() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 5\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            w_code: w_code as *const (),
            merge_point_pc: None,
            mainjitcode: None,
        });

        writer.callcontrol().grab_initial_jitcodes();
        let shell = writer.callcontrol().jitdrivers_sd[0]
            .mainjitcode
            .as_ref()
            .expect("grab_initial_jitcodes must bind jd.mainjitcode")
            .clone();
        let shell_ptr = Arc::as_ptr(&shell);
        let cached_shell = writer
            .callcontrol()
            .jitcodes
            .get(&(code_ptr as usize))
            .expect("grab_initial_jitcodes must cache the same shell");
        assert!(
            Arc::ptr_eq(&shell, cached_shell),
            "call.py:147-170 binds jd.mainjitcode to self.jitcodes[portal_graph]"
        );
        assert!(
            shell.is_skeleton(),
            "test must start from the call.py:147 shell"
        );

        writer.make_jitcodes();

        let cached = writer
            .callcontrol()
            .find_compiled_jitcode_arc(code_ptr)
            .expect("make_jitcodes must populate the portal jitcode");
        let mainjitcode = writer.callcontrol().jitdrivers_sd[0]
            .mainjitcode
            .as_ref()
            .expect("make_jitcodes must leave jd.mainjitcode bound");
        assert_eq!(
            Arc::as_ptr(mainjitcode),
            shell_ptr,
            "codewriter.py:80 must fill the call.py:147 shell in place"
        );
        assert!(
            Arc::ptr_eq(mainjitcode, &cached),
            "jd.mainjitcode and CallControl.jitcodes[portal_graph] must share the populated Arc"
        );
        assert_eq!(
            mainjitcode.jitcode.jitdriver_sd(),
            Some(0),
            "call.py:148 requires the portal jitcode to carry its jd index"
        );
    }

    #[test]
    fn portal_without_merge_point_hint_still_allocates_portal_inputs() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 1\nwhile x:\n    x = 0\n")
            .expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        assert_eq!(
            entry_arg_slots(unsafe { &*code_ptr }),
            0,
            "regression fixture must expose missing portal frame/ec inputargs"
        );
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: code_ptr,
            w_code: w_code as *const (),
            merge_point_pc: None,
            mainjitcode: None,
        });

        writer.make_jitcodes();

        let pyjit = writer
            .callcontrol()
            .find_compiled_jitcode_arc(code_ptr)
            .expect("make_jitcodes must populate the registered portal");
        assert_eq!(pyjit.jitcode.jitdriver_sd(), Some(0));
        assert!(
            pyjit.jitcode.exec.jit_merge_point_offset.is_some(),
            "registered portal with no hint should still emit portal jit_merge_point bytecode"
        );
    }

    #[test]
    fn attach_catch_exception_edge_marks_block_as_canraise() {
        let code = first_nested_function_code("def f():\n    return 1\n");
        let mut graph = new_shadow_graph(&code);
        let catch_block = graph.new_block(Vec::new());
        let catch_ref = SpamBlockRef::new(catch_block.clone(), None);
        let source_state = FrameState::new(Vec::new(), Vec::new(), None, Vec::new(), 0);
        let startblock_ref = graph.startblock.clone();

        let link =
            attach_catch_exception_edge(&mut graph, &startblock_ref, &catch_ref, &source_state);
        let startblock = graph.startblock.borrow();

        assert_eq!(
            startblock.exitswitch,
            Some(ExitSwitch::Value(c_last_exception().into()))
        );
        assert_eq!(startblock.exits.len(), 1);
        assert_eq!(startblock.exits[0], link);

        let link_borrow = startblock.exits[0].borrow();
        assert_eq!(link_borrow.target, Some(catch_block));
        assert_eq!(link_borrow.exitcase, None);
        assert!(link_borrow.last_exception.is_some());
        assert!(link_borrow.last_exc_value.is_some());
    }

    #[test]
    fn attach_catch_exception_edge_materializes_exception_state_and_extravars() {
        let code = first_nested_function_code("def f(a):\n    return a\n");
        let mut graph = new_shadow_graph(&code);
        let catch_block = graph.new_block(Vec::new());
        let catch_ref = SpamBlockRef::new(catch_block.clone(), None);
        let source_state = FrameState::new(
            vec![Some(Variable::new(VariableId(0), Kind::Ref).into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let startblock_ref = graph.startblock.clone();

        let link =
            attach_catch_exception_edge(&mut graph, &startblock_ref, &catch_ref, &source_state);

        let link_borrow = link.borrow();
        assert!(link_borrow.last_exception.is_some());
        assert!(link_borrow.last_exc_value.is_some());
        drop(link_borrow);

        let catch_state = catch_ref
            .framestate()
            .expect("catch landing should acquire a FrameState");
        assert!(catch_state.last_exception.is_some());
    }

    #[test]
    fn attach_catch_exception_edge_populates_target_inputargs() {
        let code = first_nested_function_code("def f(a):\n    return a\n");
        let mut graph = new_shadow_graph(&code);
        let catch_block = graph.new_block(Vec::new());
        let catch_ref = SpamBlockRef::new(catch_block.clone(), None);
        let local = Variable::new(VariableId(0), Kind::Ref);
        let source_state =
            FrameState::new(vec![Some(local.into())], Vec::new(), None, Vec::new(), 0);
        let startblock_ref = graph.startblock.clone();

        assert!(
            catch_block.borrow().inputargs.is_empty(),
            "catch landing block starts with no inputargs"
        );

        attach_catch_exception_edge(&mut graph, &startblock_ref, &catch_ref, &source_state);

        let inputargs = catch_block.borrow().inputargs.clone();
        assert_eq!(
            inputargs.len(),
            3,
            "expected 1 local + 2 exception Variables, got {:?}",
            inputargs
        );
        assert!(
            inputargs
                .iter()
                .all(|v| matches!(v, FlowValue::Variable(_))),
            "all catch landing inputargs must be Variables, got {:?}",
            inputargs
        );

        // model.py:114-116 Link.__init__ invariant:
        // `len(args) == len(target.inputargs)`.  Pyre's previous
        // `Link::new(Vec::new(), …)` + then-mutate flow bypassed
        // this assert (the assert ran while target.inputargs was
        // still empty, then update_catch_landing_state populated
        // it after the fact).  The regression test pins the
        // post-fix arity match.
        let startblock = graph.startblock.borrow();
        let link_borrow = startblock.exits[0].borrow();
        assert_eq!(
            link_borrow.args.len(),
            inputargs.len(),
            "Link.__init__ invariant: len(args) == len(target.inputargs)",
        );
    }

    #[test]
    fn mergeblock_reuses_matching_joinpoint() {
        let code = first_nested_function_code("def f(a):\n    return a\n");
        let mut graph = new_shadow_graph(&code);
        let current_state = FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Constant::none().into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            1,
        );
        let current_block =
            SpamBlockRef::new(graph.startblock.clone(), Some(current_state.clone()));
        let target_state = FrameState::new(
            current_state.locals_w.clone(),
            current_state.stack.clone(),
            current_state.last_exception.clone(),
            Vec::new(),
            1,
        );
        let target_block = SpamBlockRef::new(
            graph.new_block(target_state.getvariables()),
            Some(target_state),
        );
        let mut joinpoints: VecMap<usize, Vec<SpamBlockRef>> = VecMap::new();
        joinpoints.insert(1, vec![target_block.clone()]);
        let mut pendingblocks: VecDeque<SpamBlockRef> = VecDeque::new();
        let mut all_walker_blocks: Vec<SpamBlockRef> = Vec::new();

        let merged = mergeblock(
            &code,
            &mut graph,
            &mut joinpoints,
            &current_block,
            &current_state,
            1,
            &mut pendingblocks,
            &mut all_walker_blocks,
        );

        assert_eq!(merged, target_block);
        assert_eq!(
            joinpoints.get(&1).and_then(|b| b.first()),
            Some(&target_block)
        );
        // flowcontext.py:438-441 match-success returns without touching
        // pendingblocks.
        assert!(
            pendingblocks.is_empty(),
            "match-success path must not push to pendingblocks",
        );
        let exits = current_block.block().borrow().exits.clone();
        assert_eq!(exits.len(), 1);
        let link = exits[0].borrow();
        assert_eq!(link.target, Some(target_block.block()));
        assert_eq!(
            link.args,
            vec![Some(Variable::new(VariableId(0), Kind::Ref).into())]
        );
    }

    #[test]
    fn mergeblock_generalizes_existing_joinpoint() {
        let code = first_nested_function_code("def f(a):\n    return a\n");
        let mut graph = new_shadow_graph(&code);
        let source_state = FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Constant::signed(7).into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            2,
        );
        let current_block = SpamBlockRef::new(graph.startblock.clone(), Some(source_state.clone()));
        let existing_state = FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Constant::none().into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            2,
        );
        let existing_block = SpamBlockRef::new(
            graph.new_block(existing_state.getvariables()),
            Some(existing_state),
        );
        let mut joinpoints: VecMap<usize, Vec<SpamBlockRef>> = VecMap::new();
        joinpoints.insert(2, vec![existing_block.clone()]);
        let mut pendingblocks: VecDeque<SpamBlockRef> = VecDeque::new();
        let mut all_walker_blocks: Vec<SpamBlockRef> = Vec::new();

        let merged = mergeblock(
            &code,
            &mut graph,
            &mut joinpoints,
            &current_block,
            &source_state,
            2,
            &mut pendingblocks,
            &mut all_walker_blocks,
        );

        assert_ne!(merged, existing_block);
        assert!(existing_block.dead());
        // flowcontext.py:463 `self.pendingblocks.append(newblock)` parity.
        assert_eq!(
            pendingblocks.len(),
            1,
            "supersede path must push the widened block to pendingblocks"
        );
        assert_eq!(pendingblocks[0], merged);
        assert_eq!(
            pendingblocks[0]
                .framestate()
                .and_then(|s| Some(s.next_offset)),
            Some(2),
            "pending block's framestate.next_offset must carry the merge PC"
        );
        assert_eq!(joinpoints.get(&2).and_then(|b| b.first()), Some(&merged));
        let merged_state = merged
            .framestate()
            .expect("merged block should keep framestate");
        match merged_state.locals_w[1].as_ref() {
            Some(FlowValue::Variable(_)) => {}
            other => panic!("expected generalized variable, got {other:?}"),
        }
        match merged_state.locals_w[0].as_ref() {
            Some(FlowValue::Variable(variable)) => assert!(variable.name().starts_with("a_")),
            other => panic!("expected renamed local variable, got {other:?}"),
        }
        let existing_ref = existing_block.block();
        let existing_borrow = existing_ref.borrow();
        assert_eq!(existing_borrow.exits.len(), 1);
        let forwarded = existing_borrow.exits[0].borrow();
        assert_eq!(forwarded.target, Some(merged.block()));
    }

    #[test]
    fn setup_jitdriver_dedups_by_runtime_code_identity() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };

        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: &code as *const _,
            w_code: w_code as *const (),
            merge_point_pc: None,
            mainjitcode: None,
        });
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: raw_code,
            w_code: w_code as *const (),
            merge_point_pc: Some(7),
            mainjitcode: None,
        });

        let skeleton_ptr = {
            let cc = writer.callcontrol();
            assert_eq!(cc.jitdrivers_sd.len(), 1);
            assert_eq!(cc.jitdrivers_sd[0].portal_graph, raw_code);
            assert_eq!(cc.jitdrivers_sd[0].merge_point_pc, Some(7));
            assert_eq!(cc.jitdriver_sd_from_portal_graph(raw_code), Some(0));

            cc.grab_initial_jitcodes();
            let cached = cc
                .jitcodes
                .get(&(raw_code as usize))
                .expect("grab_initial_jitcodes inserts portal jitcode skeleton");
            let mainjitcode = cc.jitdrivers_sd[0]
                .mainjitcode
                .as_ref()
                .expect("call.py:147 binds jd.mainjitcode immediately");
            assert!(
                std::sync::Arc::ptr_eq(cached, mainjitcode),
                "jd.mainjitcode must share the same PyJitCode Arc as CallControl.jitcodes"
            );
            assert_eq!(
                mainjitcode.jitcode.jitdriver_sd(),
                Some(0),
                "call.py:148 stamps jd.mainjitcode.jitdriver_sd immediately"
            );
            std::sync::Arc::as_ptr(cached)
        };

        let all_jitcodes = writer.drain_unfinished_graphs();
        let cc = writer.callcontrol();
        let cached = cc
            .jitcodes
            .get(&(raw_code as usize))
            .expect("drain keeps the portal jitcode cached");
        let mainjitcode = cc.jitdrivers_sd[0]
            .mainjitcode
            .as_ref()
            .expect("drain rebinds jd.mainjitcode to the populated portal");
        let all_ptrs: Vec<*const PyJitCode> =
            all_jitcodes.iter().map(std::sync::Arc::as_ptr).collect();
        assert_eq!(all_ptrs, vec![std::sync::Arc::as_ptr(cached)]);
        assert_eq!(
            std::sync::Arc::as_ptr(cached),
            skeleton_ptr,
            "portal skeleton Arc should be filled in place despite jd.mainjitcode"
        );
        assert!(
            std::sync::Arc::ptr_eq(cached, mainjitcode),
            "populated jd.mainjitcode must remain the same Arc as CallControl.jitcodes"
        );
        assert!(mainjitcode.is_populated());
        assert_eq!(mainjitcode.jitcode.jitdriver_sd(), Some(0));
    }

    /// Bool branches must be selected by their `llexitcase`, not by
    /// target block label.  If true and false both target the same
    /// block, only the false link owns the `goto_if_not` TLabel; the
    /// true link is PyPy's fallthrough arm (`flatten.py:260-267`).
    #[test]
    fn rewrite_source_terminator_for_link_keeps_bool_true_fallthrough() {
        use super::super::flatten::{Insn, Operand, TLabel};

        let target_name = "block_target_42".to_string();
        let goto_if_not = Insn::op(
            "goto_if_not",
            vec![
                Operand::reg(Kind::Int, 5),
                Operand::TLabel(TLabel::new(target_name.clone())),
            ],
        );
        let source = super::SpamBlockRef::new(super::super::flow::Block::shared(Vec::new()), None);
        source.push_insn(Insn::live(Vec::new()));
        source.push_insn(goto_if_not);

        let target = super::super::flow::Block::shared(Vec::new());
        let true_link = super::super::flow::Link::new(
            Vec::new(),
            Some(target.clone()),
            Some(super::super::flow::Constant::bool(true).into()),
        )
        .with_llexitcase(super::super::flow::Constant::bool(true).into())
        .into_ref();
        let false_link = super::super::flow::Link::new(
            Vec::new(),
            Some(target),
            Some(super::super::flow::Constant::bool(false).into()),
        )
        .with_llexitcase(super::super::flow::Constant::bool(false).into())
        .into_ref();

        let true_result = super::rewrite_source_terminator_for_link(
            &source,
            &true_link,
            &target_name,
            "epsilon3_link_true",
        );
        assert_eq!(true_result, super::TerminatorRewrite::FallthroughOrDefault);
        assert!(
            matches!(&source.per_block_ssarepr()[1],
                Insn::Op { args, .. }
                    if args.iter().any(|a| matches!(a, Operand::TLabel(t) if t.name == target_name))),
            "true-link fallthrough must not steal the false branch label",
        );

        let false_result = super::rewrite_source_terminator_for_link(
            &source,
            &false_link,
            &target_name,
            "epsilon3_link_false",
        );
        assert_eq!(false_result, super::TerminatorRewrite::Rewritten);
        assert!(
            matches!(&source.per_block_ssarepr()[1],
                Insn::Op { args, .. }
                    if args.iter().any(|a| matches!(a, Operand::TLabel(t) if t.name == "epsilon3_link_false"))),
            "false link must rewrite the explicit goto_if_not label",
        );
    }

    /// Switch branches must be selected by `(key, TLabel(link))`.
    /// Two different switch cases may jump to the same target block
    /// with different renamings, so target-label matching alone can
    /// attach the trampoline to the wrong case.
    #[test]
    fn rewrite_source_terminator_for_link_rewrites_switch_by_key() {
        use super::super::flatten::{DescrOperand, Insn, Operand, SwitchDictDescr, TLabel};
        use std::rc::Rc;

        let target_name = "block_switch_target_7".to_string();
        let trampoline_name = "epsilon3_link_case3".to_string();

        let mut sw = SwitchDictDescr::new();
        sw.labels.push((1, TLabel::new(target_name.clone())));
        sw.labels.push((3, TLabel::new(target_name.clone())));

        let switch_op = Insn::op(
            "switch",
            vec![
                Operand::reg(Kind::Int, 11),
                Operand::Descr(Rc::new(DescrOperand::SwitchDict(sw))),
            ],
        );
        let source = super::SpamBlockRef::new(super::super::flow::Block::shared(Vec::new()), None);
        source.push_insn(switch_op);

        let target = super::super::flow::Block::shared(Vec::new());
        let link_case3 = super::super::flow::Link::new(
            Vec::new(),
            Some(target),
            Some(super::super::flow::Constant::signed(3).into()),
        )
        .with_llexitcase(super::super::flow::Constant::signed(3).into())
        .into_ref();

        let rewrote = super::rewrite_source_terminator_for_link(
            &source,
            &link_case3,
            &target_name,
            &trampoline_name,
        );
        assert_eq!(rewrote, super::TerminatorRewrite::Rewritten);

        let insns = source.per_block_ssarepr();
        let Insn::Op { args, .. } = &insns[0] else {
            panic!("expected single switch op");
        };
        let descr = args.iter().find_map(|a| match a {
            Operand::Descr(d) => Some(d.clone()),
            _ => None,
        });
        let descr = descr.expect("switch op must carry a Descr arg");
        let DescrOperand::SwitchDict(sw) = descr.as_ref() else {
            panic!("expected SwitchDict descr");
        };
        let rewritten = sw
            .labels
            .iter()
            .find(|(k, _)| *k == 3)
            .map(|(_, tl)| tl.name.clone());
        assert_eq!(rewritten.as_deref(), Some(trampoline_name.as_str()));
        let untouched_case1 = sw
            .labels
            .iter()
            .find(|(k, _)| *k == 1)
            .map(|(_, tl)| tl.name.clone());
        assert_eq!(untouched_case1.as_deref(), Some(target_name.as_str()));
    }
}
