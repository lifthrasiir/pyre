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

fn graph_entry_inputargs(code: &CodeObject, portal_inputs: bool) -> Vec<super::flow::FlowValue> {
    let mut inputargs = entry_inputargs(code);
    if portal_inputs {
        let (frame, ec) = portal_graph_inputvars(code);
        inputargs.push(frame.into());
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
    w_code: *const (),
    jitdriver_index: usize,
) -> Vec<super::flow::SpaceOperationArg> {
    let (frame, ec) = portal_graph_inputvars_from_startblock(graph);
    let greens = vec![
        super::flow::Constant::signed(next_instr as i64).into(),
        super::flow::Constant::signed(0).into(),
        super::flow::Constant::opaque(format!("pycode@{w_code:p}"), Some(Kind::Ref)).into(),
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
    /// Graph-level portal red slots: the `(frame, ec)` Variables that
    /// flow through every block of a portal graph.  Populated on the
    /// entry FrameState (via `entry_frame_state(code, portal_inputs=
    /// true)`) and propagated through block transitions unchanged —
    /// portal Variables carry graph-level identity, not per-block SSA
    /// names, so `copy()` passes them through without freshening and
    /// `union()` requires both sides to agree.  Mirrors the red
    /// carry-through in `rpython/jit/metainterp/warmspot.py` where the
    /// jitdriver_sd.reds list names `(jitframe, ec)` that the portal
    /// interpreter function threads through every iteration of the
    /// loop.  Participates in `mergeable()` after the last-exception
    /// pair so backedge `Link.args` produced by `getoutputargs()` stay
    /// aligned with the portal `startblock.inputargs` appended by
    /// `graph_entry_inputargs(code, portal_inputs=true)`.
    portal_extras: Option<(super::flow::FlowValue, super::flow::FlowValue)>,
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

    /// Seed the graph-level portal `(frame, ec)` pair on this state.
    /// Called from `entry_frame_state(code, portal_inputs=true)` for
    /// the startblock state of portal graphs; every state derived from
    /// that entry state via `copy()` or `union()` preserves the same
    /// pair.
    fn with_portal_extras(
        mut self,
        extras: (super::flow::FlowValue, super::flow::FlowValue),
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
            data.push(Some(ec.clone()));
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
    /// mergeable lists filtered for Variables).  Subsequent slices (S2)
    /// translate this mergeable index to the concrete SSARepr register
    /// slot by folding in `nlocals` / `ncells` / `stack_base`.  S3 uses
    /// the pair (mergeable index of `link.args[j]` in source state,
    /// mergeable index of `target.inputargs[j]` in target state) to
    /// drive `coalesce_by_links()`, the CFG-level replacement for pyre's
    /// current SSARepr `*_copy` scanner (`regalloc.rs::coalesce_variables`).
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
        // are both portal-seeded they must reference the same Variables,
        // otherwise the graph is malformed.  Non-portal graphs never
        // populate them.
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
        self.getoutputargs_with_positions(targetstate).0
    }

    fn getoutputargs_with_positions(
        &self,
        targetstate: &Self,
    ) -> (
        Vec<super::flow::FlowValue>,
        Vec<super::flow::LinkArgPosition>,
    ) {
        let mergeable = self.mergeable();
        let mut result = Vec::new();
        let mut positions = Vec::new();
        for (index, target_value) in targetstate.mergeable().iter().enumerate() {
            if matches!(target_value, Some(super::flow::FlowValue::Variable(_))) {
                result.push(
                    mergeable[index]
                        .clone()
                        .expect("target variable must correspond to a mergeable source value"),
                );
                positions.push(super::flow::LinkArgPosition {
                    source_mergeable_index: Some(index),
                    target_mergeable_index: Some(index),
                });
            }
        }
        (result, positions)
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

fn entry_frame_state(code: &CodeObject, portal_inputs: bool) -> FrameState {
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
    if portal_inputs {
        let (frame, ec) = portal_graph_inputvars(code);
        state.with_portal_extras((frame.into(), ec.into()))
    } else {
        state
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
    /// Task #227 per-block ssarepr accumulator — pyre-side mirror of
    /// upstream `block.operations` recorded inside `record_block`
    /// (`flowcontext.py:407-416`).  Populated alongside the program-
    /// wide `ssarepr.insns` push so a future post-walk `flatten_graph`
    /// can iterate `graph.iterblocks()` and consume the per-block
    /// emit sequence in graph-DFS order, matching
    /// `codewriter.py:53 flatten_graph(graph, regallocs, cpu=...)`.
    /// While the walker still drives production, this shadow only
    /// records label-equivalent block entries; once Task #227.2 wires
    /// every `emit_*!` macro through `push_insn` the shadow becomes
    /// the authoritative source consumed by the post-walk flatten.
    per_block_ssarepr: Vec<super::flatten::Insn>,
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
    /// in sync until production flips to consume it (Task #227.3).
    fn push_insn(&self, insn: super::flatten::Insn) {
        self.0.borrow_mut().per_block_ssarepr.push(insn);
    }

    /// Snapshot the per-block accumulator — used by Task #227.2
    /// verification probes and by the post-walk flatten driver to
    /// drain the per-block emit sequence in graph-DFS order.
    fn per_block_ssarepr(&self) -> Vec<super::flatten::Insn> {
        self.0.borrow().per_block_ssarepr.clone()
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

/// Task #227.4 walker emit helper — pushes `insn` into
/// `current_block`'s per-block accumulator.  The program-wide
/// `ssarepr.insns` is populated post-walk via the drain swap at
/// `transform_graph_to_jitcode`'s end (matching `codewriter.py:53
/// flatten_graph(graph, regallocs, cpu)`).  Every walker emit site
/// uniformly routes through this helper now — no direct
/// `ssarepr.insns.push` calls remain in production.
fn push_walker_emit(current_block: &SpamBlockRef, insn: super::flatten::Insn) {
    current_block.push_insn(insn);
}

/// Drain per-block accumulators into a single contiguous `Insn`
/// stream, stripping the defensive walker-emitted `goto pcN; ---`
/// pair when the next block opens with that label (block boundary
/// fall-through). RPython `flatten.py:106-155 make_link` falls through
/// to the next block by recursive descent and never materialises the
/// pair; pyre's walker emits both at block-switch boundaries (the
/// drain order isn't known at yield time since `pendingblocks` is
/// mixed push_front / push_back) so this pass undoes the materialisation
/// when the layout makes it redundant.
///
/// The next-block label is recognised in two shapes:
/// * `Insn::Label(L)` — upstream-orthodox block / link / catch-landing
///   labels (`flatten.py:116 self.emit(Label(block))`).
/// * `Insn::PcAnchor { py_pc }` — pyre's per-PC anchor introduced
///   when the transitional `Label("pcN")` shape was retired. The
///   matching `goto TLabel("pcN")` carries the name produced by
///   `pc_label_name(py_pc)`; both shapes resolve to the same string
///   key, so the strip recogniser unifies them.
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
        let next_label_name: Option<String> = blocks
            .get(i + 1)
            .and_then(|next| next.first())
            .and_then(|first| match first {
                super::flatten::Insn::Label(l) => Some(l.name.clone()),
                super::flatten::Insn::PcAnchor { py_pc } => {
                    Some(super::flatten::pc_label_name(*py_pc))
                }
                _ => None,
            });
        let block_insns = &mut blocks[i];
        let len = block_insns.len();
        let strip_tail = if len >= 2 {
            match (
                &block_insns[len - 2],
                &block_insns[len - 1],
                next_label_name.as_deref(),
            ) {
                (
                    super::flatten::Insn::Op { opname, args, .. },
                    super::flatten::Insn::Unreachable,
                    Some(next_name),
                ) if opname == "goto"
                    && args.len() == 1
                    && matches!(&args[0], Operand::TLabel(target) if target.name == next_name) =>
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

fn append_exit(block: &super::flow::BlockRef, link: super::flow::LinkRef) {
    link.borrow_mut().prevblock = Some(block.downgrade());
    block.borrow_mut().exits.push(link);
}

/// atomically append `link` to `block.exits` and
/// snapshot `source_state` into `link_exit_states` so later passes
/// (`collect_link_slot_pairs`) can resolve the source-side register
/// slots at this link.
///
/// RPython parity: there is no direct counterpart — RPython's
/// `coalesce_variables` runs inline with Variable-keyed UnionFind
/// over `graph.iterblocks()`, so no per-link state capture is
/// needed.  pyre's regalloc runs after-the-fact on a u16-indexed
/// SSARepr (`regalloc.rs` docstring, lines 26-36 PRE-EXISTING-
/// ADAPTATION), so the collector needs the source FrameState to
/// translate Variables back to slots.  The snapshot is the minimal
/// bridging data — one FrameState per link, cloned at emission time
/// (the walker discards its `currentstate` after the terminator
/// finishes so a clone is the only way to preserve it).
fn append_exit_with_state(
    block: &super::flow::BlockRef,
    link: super::flow::LinkRef,
    source_state: &FrameState,
    link_exit_states: &mut HashMap<super::flow::LinkRef, FrameState>,
) {
    link_exit_states.insert(link.clone(), source_state.clone());
    append_exit(block, link);
}

fn output_link(
    source_state: &FrameState,
    target_state: &FrameState,
    target: super::flow::BlockRef,
) -> super::flow::LinkRef {
    let (outputargs, arg_positions) = source_state.getoutputargs_with_positions(target_state);
    super::flow::Link::new(outputargs, Some(target), None)
        .with_arg_positions(arg_positions)
        .into_ref()
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
    (
        graph.fresh_untyped_variable(),
        graph.fresh_untyped_variable(),
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
    catch_landing_blocks: &HashMap<u16, SpamBlockRef>,
    handler_py_pc: usize,
) -> Option<FrameState> {
    let mut merged: Option<FrameState> = None;
    for site in catch_sites {
        if site.handler_py_pc != handler_py_pc {
            continue;
        }
        let landing_state = catch_landing_blocks
            .get(&site.landing_label)
            .and_then(|block| block.framestate())?;
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
    link_exit_states: &mut HashMap<super::flow::LinkRef, FrameState>,
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
    append_exit_with_state(
        &currentblock.block(),
        output_link(currentstate, &newstate, newblock.block()),
        currentstate,
        link_exit_states,
    );
    // flowcontext.py:472 `self.pendingblocks.append(newblock)`.
    pendingblocks.push_back(newblock.clone());
    newblock
}

fn mergeblock(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    joinpoints: &mut HashMap<usize, Vec<SpamBlockRef>>,
    currentblock: &SpamBlockRef,
    currentstate: &FrameState,
    next_offset: usize,
    link_exit_states: &mut HashMap<super::flow::LinkRef, FrameState>,
    pendingblocks: &mut VecDeque<SpamBlockRef>,
    all_walker_blocks: &mut Vec<SpamBlockRef>,
) -> SpamBlockRef {
    let candidates = joinpoints.entry(next_offset).or_default();
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
            append_exit_with_state(
                &currentblock.block(),
                output_link(currentstate, &newstate, block.block()),
                currentstate,
                link_exit_states,
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
            // does the same.  Retires when the Task #227 walker
            // restructure replaces PC sequencing with a pendingblocks-driven
            // walker.
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
        // Task #227.3 SpamBlockRef enumeration — record the
        // supersede-newblock in walker-visit order.
        all_walker_blocks.push(newblock.clone());
        newblock.block().borrow_mut().inputargs = newstate.getvariables();
        append_exit_with_state(
            &currentblock.block(),
            output_link(currentstate, &newstate, newblock.block()),
            currentstate,
            link_exit_states,
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
        // Phase A.4 matches upstream: the supersede newblock IS
        // re-walked under widened inputargs.  Its emit appends a
        // duplicate `Label("pcN")` + `-live-` pair to ssarepr.insns,
        // but `pc_anchor_positions` (first-wins) and
        // `live_marker_indices_by_pc` (first-takes) ensure the
        // original dead block's bytes remain the runtime canonical
        // emission for `pcN`.  The supersede re-walk's bytes are
        // unreachable via pcN labels.
        block.mark_dead();
        block.block().borrow_mut().operations.clear();
        block.block().borrow_mut().exitswitch = None;
        let supersede_link = output_link(&block_state, &newstate, newblock.block());
        link_exit_states.insert(supersede_link.clone(), block_state.clone());
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
        link_exit_states,
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

fn null_stack_sentinel() -> super::flow::FlowValue {
    // CPython's PUSH_NULL / LOAD_GLOBAL(push_null) stack marker is a
    // frontend-only calling-convention sentinel, not a user-visible
    // Python object and not an RPython flow-graph value.  Keep it out of
    // graph operations and use it only to preserve the shadow stack's
    // arity until CALL discards the slot.
    super::flow::Constant::opaque("push_null", Some(Kind::Ref)).into()
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

/// Step 6 transitional dual-write.  `rpython/jit/codewriter/codewriter.py:44-67`
/// runs `perform_register_allocation(graph) → flatten_graph(graph) →
/// compute_liveness(ssarepr) → assemble(ssarepr)`.  Upstream has **one**
/// IR stream — the flow graph — which `flatten_graph` lowers into an
/// `SSARepr`.
///
/// Pyre historically emitted `SSARepr` directly from the trace recorder
/// and skipped the flow-graph stage.  Step 6A reintroduces the graph
/// (so CFG-level `regalloc.py:79-96 coalesce_variables` can run), but the
/// SSARepr emission has not yet been replaced with a `flatten_graph` pass
/// (Task #214).  Until it is, each opcode handler must populate both
/// streams — the SSARepr byte stream that backend/blackhole consume, and
/// the graph that `FlowGraphRegAllocator` consumes.
///
/// Delete this helper once Task #214 lands and the SSARepr stream is
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
///     [`super::flatten::intern_call_descr_stub`] (Task #41 plumbing).
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
    op_args.push(
        super::flatten::intern_call_descr_stub(
            super::flatten::effect_info_for_call_flavor(flavor),
            arg_kinds,
            reskind.to_kind(),
        )
        .into(),
    );

    match reskind.to_kind() {
        Some(result_kind) => {
            let result = graph.fresh_variable(result_kind);
            record_graph_op(block, opname, op_args, Some(result.into()), offset);
            Some(result)
        }
        None => {
            record_graph_op(block, opname, op_args, None, offset);
            None
        }
    }
}

/// Emit a void-result `SpaceOperation` into `block` and return it.
/// Matches the call-marker / control-flow emission path in
/// `rpython/jit/codewriter/jtransform.py:1690-1723` where markers like
/// `jit_merge_point` and `loop_header` are produced with no `result`
/// and immediately fed into `GraphFlattener.serialize_op`.
///
/// Phase 1 walker-rewrite entrypoint (Task #224): the void counterpart
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
/// Phase 1 walker-rewrite entrypoint (Task #224): a single place that
/// packages the fresh-Variable → `record_graph_op` → return-Variable
/// pattern so the walker's per-opcode handlers can record
/// value-producing graph operations without inlining `graph.fresh_variable`
/// + `record_graph_op` at every call site.  Future sessions migrate
/// individual emit sites to call this helper instead of emitting directly
/// to `SSARepr`; when every value-producing op records through this path
/// the production pipeline can flip to `flatten_graph(graph, ...)` per
/// `codewriter.py:44-67`.
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
    // though Phase 3c (commit `bc0d6a06c4`) has already collapsed the
    // dual emitter into a single walker-local `SSARepr`.
    //
    // Convergence path: Task #229 (TmpVarEnv) replaces the SSA-side
    // `scratch_truth` slot with a `fresh_var(Kind::Int)` graph Variable so
    // the same Variable drives both the front-end exitswitch and the
    // flatten-emitted `goto_if_not`. Once that lands, lower `bool` as a
    // residual_call to `truth_fn` in the same pass that lowers other
    // graph ops to assembler Insns and drop the second emit at the
    // call sites below.
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

/// Pyre's `ConstantData` enum is richer than RPython's flat
/// `Constant(value)` — it carries variant-typed payloads
/// (None/Boolean/Integer/Str/...) that not all map cleanly into a
/// flowgraph `Constant`. Returns `None` for variants the shadow graph
/// cannot represent (the caller falls back to `fresh_ref_value`).
/// `flowcontext.py:838-843` (`LOAD_CONST` → `getconstant_w()` +
/// `pushvalue`) has no analogous variant filter because RPython
/// constants are uniform Python objects.
fn frontend_constant_flow_value(
    constant: &pyre_interpreter::bytecode::ConstantData,
) -> Option<super::flow::FlowValue> {
    // Keep every representable frontend constant in the shadow graph
    // instead of degrading immediately to a fresh Variable.
    match constant {
        pyre_interpreter::bytecode::ConstantData::None => {
            Some(super::flow::Constant::none().into())
        }
        pyre_interpreter::bytecode::ConstantData::Boolean { value } => {
            Some(super::flow::Constant::bool(*value).into())
        }
        pyre_interpreter::bytecode::ConstantData::Integer { value } => {
            use num_traits::ToPrimitive;
            value
                .to_i64()
                .map(|value| super::flow::Constant::signed(value).into())
        }
        pyre_interpreter::bytecode::ConstantData::Str { value } => Some(
            super::flow::Constant::string(value.as_str().expect("non-UTF-8 string constant"))
                .into(),
        ),
        _ => None,
    }
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
    portal_inputs: bool,
) -> super::flow::FunctionGraph {
    let start_inputargs = graph_entry_inputargs(code, portal_inputs);
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
    new_shadow_graph_with_portal_inputs(code, false)
}

fn attach_catch_exception_edge(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    target: &SpamBlockRef,
    source_state: &FrameState,
    link_exit_states: &mut HashMap<super::flow::LinkRef, FrameState>,
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
    // `getoutputargs_with_positions` below) AND `link.extravars`.
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
    // `getoutputargs_with_positions` walks `target_state.mergeable()`
    // — the same mergeable layout as `target.inputargs`.
    update_catch_landing_state(graph, target, &edge_state);

    // `model.py:114-116 Link.__init__` enforces
    // `len(args) == len(target.inputargs)`.  Build `link.args` via
    // `FrameState::getoutputargs_with_positions(target_state)` so
    // each link arg aligns with the corresponding target inputarg
    // by mergeable position.  This restores the RPython invariant
    // that the previous `Link::new(Vec::new(), …)` then-mutate
    // flow bypassed (the `Link::new` arity assert ran before
    // `update_catch_landing_state` populated `target.inputargs`).
    let target_state = target
        .framestate()
        .expect("catch landing must have a framestate after update_catch_landing_state");
    let (link_args, arg_positions) = edge_state.getoutputargs_with_positions(&target_state);

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
    let mut link = super::flow::Link::new(link_args, Some(target.block()), None)
        .with_arg_positions(arg_positions);
    link.extravars(Some(exc_type), Some(exc_value));
    let link = link.into_ref();
    append_exit_with_state(block, link.clone(), source_state, link_exit_states);
    link
}

/// collect `BlockRef → FrameState` entries from the
/// walker's in-flight block catalogues.  Pure function, no side effects.
///
/// After Phase P2c the walker maintains two `SpamBlockRef` containers:
///   - `joinpoints[py_pc]`          — every merged / superseded / fresh
///     candidate for each Python PC.
///   - `catch_landing_blocks[label]` — pre-allocated catch-landing
///     entries.
///
/// Catch-landing `SpamBlockRef`s are constructed with `framestate =
/// None` (`SpamBlockRef::new(..., None)`), so they are naturally
/// skipped here.  Same for `FunctionGraph::returnblock` /
/// `exceptblock` — those are canonical blocks that never flow through
/// a `SpamBlockRef`.
///
/// Consumer: S4 feeds this map plus the graph into
/// `collect_link_slot_pairs` to produce per-link coalesce pairs in
/// the production walker path.
fn collect_block_states(
    joinpoints: &HashMap<usize, Vec<SpamBlockRef>>,
    catch_landing_blocks: &HashMap<u16, SpamBlockRef>,
) -> HashMap<super::flow::BlockRef, FrameState> {
    let mut map = HashMap::new();
    let mut absorb = |entry: &SpamBlockRef| {
        if let Some(state) = entry.framestate() {
            map.insert(entry.block(), state);
        }
    };
    for candidates in joinpoints.values() {
        for entry in candidates {
            absorb(entry);
        }
    }
    for entry in catch_landing_blocks.values() {
        absorb(entry);
    }
    map
}

/// CFG-level collection of
/// `(source_slot, target_slot)` coalesce pairs.  Pure function, no
/// side effects.
///
/// Walks `graph.iterblocks()` → each block's exits.  For each Link:
///   1. Source state = `link_exit_states[link]` — the walker's
///      `currentstate` snapshot captured at terminator emission time
///      (`flowcontext.py:1237,1268-1280`).  This is the source
///      block's EXIT state, not its ENTRY state, because fresh
///      Variables produced by mid-block operations live in
///      `currentstate.locals_w` / `currentstate.stack` but never in
///      the source block's stored ENTRY FrameState.
///   2. Target state = `block_entry_states[link.target]` — the target
///      block's ENTRY FrameState set up by `mergeblock` /
///      `initialize_spam_block` (its mergeable positions correspond
///      directly to `target.inputargs`).
///   3. Links with no source EXIT entry or no target ENTRY entry
///      (catch landings, `returnblock`, `exceptblock`) contribute no
///      pairs.
///   4. For each `link.args[j]` with preserved
///      `Link.arg_positions[j]`:
///        - `source_mergeable_index` comes from the source
///          `FrameState.getoutputargs()` walk at edge-construction
///          time; `target_mergeable_index` records the target-side
///          mergeable entry that produced `target.inputargs[j]`.
///        - Skip non-Variable source args, matching
///          `regalloc.py:99-101` `if isinstance(v, Variable)`.
///        - Resolve source / target slots independently with
///          `FrameState::mergeable_index_to_slot(...)`; either side
///          may return `None` for the `last_exception` pair.
///        - Push `(source_slot, target_slot)`.  For ordinary
///          jump/merge edges these are usually equal because
///          `framestate.py:getoutputargs()` uses the same mergeable
///          index on both sides, but pyre now reads the recorded
///          per-link positions instead of re-deriving them from whole
///          FrameState scans.
///
/// Upstream reference: `rpython/tool/algo/regalloc.py:79-96`
/// `RegAllocator.coalesce_variables` iterates `graph.iterblocks()` →
/// `block.exits` → `zip(link.args, link.target.inputargs)` and unions
/// each Variable pair via `_try_coalesce`.  RPython has no FrameState
/// indirection — Variables carry their own UnionFind identity.
/// pyre's regalloc is u16-register-keyed (Note;
/// see `regalloc.rs:26-36`), so this helper projects Variables back
/// onto slots through the per-link mergeable positions preserved when
/// the Link was created.
///
/// Why positional, not Variable-keyed: pyre's walker can reuse one
/// Variable across multiple mergeable positions simultaneously — e.g.
/// `LoadFast` at `codewriter.rs:2413-2414` pushes the local's own
/// Variable onto the stack, so that Variable lives at slot `x` (in
/// `locals_w`) AND at slot `stack_base + depth` (in `stack`) in the
/// same FrameState.  A Variable → single slot map would be ambiguous;
/// the per-link mergeable indices preserved from
/// `FrameState::getoutputargs_with_positions` keep the exact source /
/// target positions.
fn collect_link_slot_pairs(
    graph: &super::flow::FunctionGraph,
    block_entry_states: &HashMap<super::flow::BlockRef, FrameState>,
    link_exit_states: &HashMap<super::flow::LinkRef, FrameState>,
) -> Vec<(u16, u16)> {
    let mut pairs = Vec::new();
    for block in graph.iterblocks() {
        let block_borrow = block.borrow();
        for link in &block_borrow.exits {
            let Some(source_state) = link_exit_states.get(link) else {
                continue;
            };
            let link_borrow = link.borrow();
            let Some(target) = link_borrow.target.clone() else {
                continue;
            };
            let Some(target_state) = block_entry_states.get(&target) else {
                continue;
            };
            for (arg, positions) in link_borrow
                .args
                .iter()
                .zip(link_borrow.arg_positions.iter())
            {
                let Some(super::flow::FlowValue::Variable(_)) = arg.as_ref() else {
                    continue;
                };
                let Some(source_idx) = positions.source_mergeable_index else {
                    continue;
                };
                let Some(target_idx) = positions.target_mergeable_index else {
                    continue;
                };
                let Some(source_slot) = source_state.mergeable_index_to_slot(source_idx) else {
                    continue;
                };
                let Some(target_slot) = target_state.mergeable_index_to_slot(target_idx) else {
                    continue;
                };
                pairs.push((source_slot, target_slot));
            }
        }
    }
    pairs
}

// `PyJitCode` and `PyJitCodeMetadata` live in `pyre_jit_trace::pyjitcode`
// so both the codewriter (here) and the trace/blackhole runtime can hold
// the same `Arc<PyJitCode>` instances.

#[derive(Clone, Copy)]
struct ExceptionCatchSite {
    landing_label: u16,
    handler_py_pc: usize,
    stack_depth: u16,
    push_lasti: bool,
    lasti_py_pc: usize,
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
///   NEW-DEVIATION.
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
    // `Pure*` would be a NEW-DEVIATION; stay on `Plain` until the
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
    // `bh_normalize_raise_varargs_fn` walks the exception class /
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
use super::flatten::label_pc_index;

fn pc_anchor_positions(ssarepr: &super::flatten::SSARepr, num_pcs: usize) -> Vec<usize> {
    // Per-PC anchor positions are resolved from `Insn::PcAnchor { py_pc }`
    // entries.
    //
    // Pyre vs RPython label keying (structural difference): RPython uses
    // `Label(block)` per SpamBlock object (`flatten.py:116
    // self.emitline(Label(block))`), so two SpamBlocks reaching the same
    // Python PC carry distinct Label values. Pyre keys labels by Python
    // PC (`pc{N}` via `pc_label_name`), so two alive SpamBlocks both
    // joining at PC N emit `Insn::PcAnchor { py_pc: N }` independently —
    // both can survive R5's dead-block filter when neither is superseded.
    //
    // The runtime resolves `pc{N}` to the FIRST anchor in the drained
    // stream (first-wins below). Subsequent alive duplicates belong to
    // sibling joinpoint blocks that are reachable from the first anchor
    // via fall-through inside the walker's PC-sequential emission, so
    // the first anchor is the canonical entry point per pyre's runtime
    // dispatch contract. Eliminating duplicates entirely requires
    // switching pyre's label key from `pc{N}` to block-identity (the
    // Task #227 walker restructure), at which point this first-wins
    // semantics retires.
    let mut positions = vec![usize::MAX; num_pcs];
    for (insn_idx, insn) in ssarepr.insns.iter().enumerate() {
        if let Some(py_pc) = label_pc_index(insn) {
            assert!(
                py_pc < num_pcs,
                "pc_anchor_positions: Label pc{py_pc} out of range {num_pcs}"
            );
            if positions[py_pc] == usize::MAX {
                positions[py_pc] = insn_idx;
            }
        }
    }
    for py_pc in 0..num_pcs {
        assert_ne!(
            positions[py_pc],
            usize::MAX,
            "pc_anchor_positions: missing Label('pc{py_pc}') for py_pc {py_pc}"
        );
    }
    positions
}

fn live_marker_indices_by_pc(ssarepr: &super::flatten::SSARepr, num_pcs: usize) -> Vec<usize> {
    // Map py_pc → anchor insn_idx via `Insn::PcAnchor { py_pc }` entries.
    // Reuses `pc_anchor_positions`'s first-wins semantics; the documented
    // pyre-vs-RPython label-keying difference there applies here too.
    // Position-indexed `Vec<usize>` matches the no-HashMap invariant
    // (memory `feedback-no-hashmap-ever`).
    let positions = pc_anchor_positions(ssarepr, num_pcs);
    // Sort (anchor_idx, py_pc) by anchor_idx to compute scan boundaries
    // for the per-anchor `-live-` lookup.
    let mut anchors: Vec<(usize, usize)> = positions
        .iter()
        .enumerate()
        .map(|(py_pc, &insn_idx)| (insn_idx, py_pc))
        .collect();
    anchors.sort_by_key(|(insn_idx, _)| *insn_idx);
    let mut live_indices = vec![usize::MAX; num_pcs];
    for (anchor_pos, (anchor_idx, py_pc)) in anchors.iter().enumerate() {
        let end = anchors
            .get(anchor_pos + 1)
            .map(|(next_idx, _)| *next_idx)
            .unwrap_or(ssarepr.insns.len());
        // Take the FIRST -live- marker per anchor pair.  RPython emits
        // one per block-entry (`flatten.py:107`); pyre's PC-sequential
        // walker can emit multiple `-live-` markers in the scan range
        // when sibling joinpoint blocks share a py_pc (same structural
        // root as `pc_anchor_positions`'s first-wins).  The first
        // marker matches the runtime's actual liveness window via the
        // first-wins anchor; subsequent duplicates belong to sibling
        // blocks reachable through fall-through after the first anchor.
        let mut live_idx: Option<usize> = None;
        for insn_idx in (anchor_idx + 1)..end {
            if ssarepr.insns[insn_idx].is_live() {
                if live_idx.is_none() {
                    live_idx = Some(insn_idx);
                }
            }
        }
        live_indices[*py_pc] = live_idx.unwrap_or_else(|| {
            panic!(
                "live_marker_indices_by_pc: missing -live- marker for py_pc {} in range {}..{}",
                py_pc,
                anchor_idx + 1,
                end
            )
        });
    }
    live_indices
}

fn filter_liveness_in_place(
    ssarepr: &mut super::flatten::SSARepr,
    code: &CodeObject,
    depth_at_pc: &[u16],
    local_color_map: &[u16],
    stack_slot_color_map: &[u16],
    portal_frame_reg: u16,
    portal_ec_reg: u16,
) {
    use super::flatten::{Kind as SsaKind, Operand as SsaOperand};
    super::liveness::compute_liveness(ssarepr);
    let live_vars = pyre_jit_trace::state::liveness_for(code as *const _);
    let nlocals = code.varnames.len();
    let live_markers = live_marker_indices_by_pc(ssarepr, code.instructions.len());
    for (py_pc, insn_idx) in live_markers.into_iter().enumerate() {
        let existing = match ssarepr.insns.get_mut(insn_idx) {
            Some(insn) if insn.is_live() => insn.live_args_mut().unwrap(),
            Some(other) => panic!(
                "filter_liveness_in_place: expected -live- marker at index {insn_idx}, got {other:?}"
            ),
            None => panic!(
                "filter_liveness_in_place: insn index {insn_idx} out of range (len {})",
                ssarepr.insns.len()
            ),
        };
        // Preserve non-Register operands (TLabel) exactly as RPython
        // `liveness.py:52` keeps them alongside the `alive` set.
        let mut non_register: Vec<SsaOperand> = Vec::new();
        for op in existing.iter() {
            if !matches!(op, SsaOperand::Register(_)) {
                non_register.push(op.clone());
            }
        }

        if !live_vars.is_reachable(py_pc) {
            existing.clear();
            existing.extend(non_register);
            continue;
        }

        let depth = depth_at_pc[py_pc] as usize;
        let live_stack_colors: std::collections::BTreeSet<u16> =
            stack_slot_color_map.iter().copied().take(depth).collect();
        let mut seen_r: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
        let mut seen_i: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
        let mut seen_f: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
        let mut live_r: Vec<u16> = Vec::new();
        let mut live_i: Vec<u16> = Vec::new();
        let mut live_f: Vec<u16> = Vec::new();
        // liveness.py:67-75 `compute_liveness` adds every Register to
        // the alive set; assembler.py:150-152 splits the `-live-` args
        // into live_i / live_r / live_f by kind via
        // `get_liveness_info(insn[1:], 'int'/'ref'/'float')`. All three
        // banks mirror that shape — every alive Register of the kind is
        // pushed in encounter order with seen-set deduplication.
        for op in existing.iter() {
            let SsaOperand::Register(reg) = op else {
                continue;
            };
            match reg.kind {
                SsaKind::Ref => {
                    if seen_r.insert(reg.index) {
                        live_r.push(reg.index);
                    }
                }
                SsaKind::Int => {
                    if seen_i.insert(reg.index) {
                        live_i.push(reg.index);
                    }
                }
                SsaKind::Float => {
                    if seen_f.insert(reg.index) {
                        live_f.push(reg.index);
                    }
                }
            }
        }
        // Note: LV∩SSA retain narrows the Ref bank
        // to the post-rename colors that correspond to LV-live Python
        // locals or live stack slots at this PC.  After the slice
        // 3b-2/3b-3 flip the encoder reads `registers_r[color]`
        // directly, but scratch registers (temporaries that are SSA-
        // live but have no Python-frame slot) remain `OpRef::NONE` in
        // `registers_r` because no trace-time writer populates them.
        // Removing this retain requires either (a) populating scratch
        // colors during tracing (Task #158 graph regalloc) or (b) the
        // encoder tolerating NONE for non-frame live registers.
        //
        // `MAJIT_PHASE06_DROP_LV=1` skips the retain, exposing the
        // RPython-orthodox SSA-only `live_r` so probe-A logs in
        // `consume_one_section` (`call_jit.rs::resume_in_blackhole`)
        // can capture what BH writes per color when the bank widens.
        // Default off — bench / production keep the retain. Removed
        // once Task #158 graph regalloc lands a separate scratch
        // color space; until then this env-var is the only path back
        // to RPython form.
        assert!(
            local_color_map.len() >= nlocals,
            "local_color_map is shorter than nlocals: {} < {}",
            local_color_map.len(),
            nlocals
        );
        let lv_live: std::collections::BTreeSet<u16> = {
            let mut s: std::collections::BTreeSet<u16> = (0..nlocals)
                .filter(|&idx| live_vars.is_local_live(py_pc, idx))
                .map(|idx| local_color_map[idx])
                .collect();
            s.extend(live_stack_colors.iter().copied());
            // Portal red args (`pypy/module/pypyjit/interp_jit.py:67
            // reds = ['frame', 'ec']`) reach `live_r` through the RPython
            // force-alive mechanism (`liveness.py:11-12`): the
            // `emit_live_placeholder!` macro at codewriter.rs:3773 emits
            // every PC's `-live-` op with explicit Register args for
            // `portal_frame_reg` / `portal_ec_reg`, and `compute_liveness`
            // (`liveness.rs:101-107` line-by-line port of
            // `liveness.py:46-48`) adds those Register args to the
            // backward-propagating `alive` set, leaving them in `existing`
            // as live Register operands by the time this filter runs.
            //
            // The LV∩SSA `retain` below is a pyre adaptation for scratch
            // colors; gate the portal colors past it explicitly so the
            // retain does not drop the RPython-tracked live registers.
            // Portal-bridge installs sentinel-skip (`u16::MAX`).
            if portal_frame_reg != u16::MAX {
                s.insert(portal_frame_reg);
            }
            if portal_ec_reg != u16::MAX {
                s.insert(portal_ec_reg);
            }
            s
        };
        if std::env::var_os("MAJIT_PHASE06_DROP_LV").is_none() {
            live_r.retain(|idx| lv_live.contains(idx));
        }

        existing.clear();
        for &idx in &live_i {
            existing.push(SsaOperand::Register(super::flatten::Register::new(
                SsaKind::Int,
                idx,
            )));
        }
        for &idx in &live_r {
            existing.push(SsaOperand::Register(super::flatten::Register::new(
                SsaKind::Ref,
                idx,
            )));
        }
        for &idx in &live_f {
            existing.push(SsaOperand::Register(super::flatten::Register::new(
                SsaKind::Float,
                idx,
            )));
        }
        existing.extend(non_register);
    }
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
    code: &CodeObject,
    num_instrs: usize,
) -> (
    Vec<Option<u16>>,
    Vec<ExceptionCatchSite>,
    std::collections::HashMap<usize, u16>,
) {
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
        catch_sites.push(ExceptionCatchSite {
            landing_label,
            handler_py_pc,
            stack_depth: depth,
            push_lasti,
            lasti_py_pc: py_pc,
        });
    }
    let handler_depth_at: std::collections::HashMap<usize, u16> = exception_entries
        .iter()
        .map(|(_start, _end, target, depth, lasti)| {
            let extra = if *lasti { 1u16 } else { 0 };
            (*target, *depth + extra + 1)
        })
        .collect();
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
    /// Phase A: the cpu helpers are fixed module-level functions in
    /// `crate::call_jit`; Phase D.2 wired `callcontrol` as a field so
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
        // Per-arm fresh int scratch slots — Phase 2 Commit 2.2b
        // (Tasks #158/#159/#122 plan, plan staged-sauteeing-koala).
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
        // `null_ref_reg` PY_NULL holder before Tier 4 Epic A retired it; the
        // portal red regs keep their numerical positions so layout-sensitive
        // tests stay stable.
        let (portal_frame_reg, portal_ec_reg) =
            portal_red_pre_regalloc_slots(nlocals, max_stackdepth);
        // Per-arm fresh ref scratch slots — Phase 2 Commit 2.2 first slice
        // (Tasks #158/#159/#122 plan).  Each opcode handler arm that needs
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
        let mut current_depth: u16 = 0;

        // RPython: self.assembler = Assembler() + JitCode(graph.name, ...)
        // (rpython/jit/codewriter/jitcode.py:14-15 takes name as the first
        // __init__ argument; majit's JitCodeBuilder::set_name mirrors that).
        let mut assembler = SSAReprEmitter::new();
        assembler.set_name(code.obj_name.to_string());
        // grow an `SSARepr` alongside the direct
        // `JitCodeBuilder` calls. Currently only a handful of handlers
        // (`ref_return` below) dual-emit an `Insn::Op`; the remaining
        // bytecode handlers still route through the builder only. When
        // every handler has been converted, `ssarepr` becomes the
        // authoritative input to `jit::assembler::Assembler::assemble`
        // (Phase 3c switchover) and the direct builder calls disappear.
        // See `pyre/pyre-jit/src/jit/B6_CODEWRITER_PIPELINE_PLAN.md`.
        let mut ssarepr = SSARepr::new(code.obj_name.to_string());

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
            });
        }

        // RPython flatten.py: pre-create labels for each block.
        // Python bytecodes are linear, so each instruction index gets a label.
        let num_instrs = code.instructions.len();
        let mut labels: Vec<u16> = Vec::with_capacity(num_instrs);
        for _ in 0..num_instrs {
            labels.push(assembler.new_label());
        }

        let (catch_for_pc, catch_sites, handler_depth_at) =
            decode_exception_catch_sites(&mut assembler, code, num_instrs);

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

        // shadow `FunctionGraph` alongside `ssarepr`.
        //
        // RPython's flow space keeps `framestate` on each `SpamBlock`
        // (`flowcontext.py:38-44`) and derives `Link.args ↔
        // target.inputargs` from `FrameState.getoutputargs()`. Pyre's
        // walker is still single-pass over Python bytecode, but the
        // shadow graph now carries the same per-block `FrameState`
        // object instead of a topology-only `BlockRef`.
        //
        // Portal graphs (registered in `jitdrivers_sd`, per
        // codewriter.py:37) carry two extra red inputs —
        // `(frame, ec)` — appended to both `startblock.inputargs` via
        // `graph_entry_inputargs(code, portal_inputs=true)` AND to
        // `FrameState` via `entry_frame_state(code, portal_inputs=
        // true)`.  `FrameState.portal_extras` carries those Variables
        // through every block transition so `getoutputargs()` on any
        // backedge produces link args aligned with the appended
        // startblock slots.  Non-portal graphs populate neither side
        // and behave exactly as before.
        // `rpython/jit/codewriter/codewriter.py:37 portal_jd = self
        // .callcontrol.jitdriver_sd_from_portal_graph(graph)` —
        // upstream copies each source graph's actual `inputargs` and
        // routes the portal-only extras through transformation, not
        // by appending synthetic `(frame, ec)` to every non-portal
        // graph.  Pyre matches by gating the portal-input append on
        // `is_portal` (the prior unconditional shortcut introduced
        // upstream non-orthodoxy by adding unused inputargs to non-
        // portal graphs).
        let mut graph = new_shadow_graph_with_portal_inputs(code, is_portal);
        let mut joinpoints: HashMap<usize, Vec<SpamBlockRef>> = HashMap::new();
        // snapshot the walker's `currentstate` at
        // every terminator emission so `collect_link_slot_pairs` can
        // translate link-arg Variables to SSARepr register slots via
        // the positional walk.  RPython does not need this map because
        // `regalloc.py:79-96` unions Variables directly via UnionFind;
        // pyre's u16-keyed regalloc (regalloc.rs:26-36 PRE-EXISTING-
        // ADAPTATION) reads the source state per-link to project back
        // onto slots.  Keyed on `LinkRef` (Rc-pointer identity).
        let mut link_exit_states: HashMap<super::flow::LinkRef, FrameState> = HashMap::new();
        let start_state = entry_frame_state(code, is_portal);
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
        let mut catch_landing_blocks: HashMap<u16, SpamBlockRef> =
            HashMap::with_capacity(catch_sites.len());
        for site in &catch_sites {
            let landing = SpamBlockRef::new(graph.new_block(Vec::new()), None);
            // NOTE: catch landings are NOT pushed to `all_walker_blocks`
            // at creation — they're queued at emission time inside
            // `emit_mark_label_catch_landing!` so the drain order
            // reflects walker emission order (catch landings emit
            // AFTER the main walker loop completes per
            // `codewriter.rs::6907+`).  Creation-order tracking would
            // place them before the walker-created blocks in the
            // drain, misaligning with the post-main-loop emission
            // sequence.
            catch_landing_blocks.insert(site.landing_label, landing);
        }
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
        // Task #227.5 per-block contiguous walker — `emit_mark_label_pc!`
        // sets `block_switch_pending = true` at block transitions
        // instead of switching `current_block` inline; the inner
        // for-loop checks the flag after each per-PC emit and breaks,
        // yielding to the outer `while let Some(pending_block) =
        // pendingblocks.pop_front()` which picks up the queued new
        // block in the next iteration.  Mirrors upstream's
        // `flowcontext.py:407-416 record_block` shape where each
        // block is processed contiguously without mid-iteration
        // re-entry.  Correctness relies on the explicit `goto
        // Label("pcN")` + `Unreachable` pair emitted on the yield
        // path (Phase 4 alignment with `flatten.py:177-258
        // insert_exits`).
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
        // Pyre matched this in Phase A.4: the per-PC `emitted_pc_starts`
        // skip is retired; supersede may re-walk a PC under widened
        // framestate, producing duplicate `Label("pcN")` + `-live-`
        // pairs.  Both `pc_anchor_positions` and
        // `live_marker_indices_by_pc` use first-wins / first-takes
        // semantics so the runtime canonical bytes are the dead
        // block's emit; the supersede newblock's re-walk emit is
        // unreachable via pcN labels.

        // interp_jit.py:118 `pypyjitdriver.can_enter_jit(...)` is called in
        // `jump_absolute` (`jumpto < next_instr` branch), i.e. at each
        // Python backward jump.  jtransform.py:1714-1723
        // `handle_jit_marker__can_enter_jit = handle_jit_marker__loop_header`
        // lowers each one to a `loop_header` jitcode op.  Pyre has no
        // `jump_absolute` Python wrapper — the equivalent is to pre-scan
        // `JumpBackward` opcodes and record their targets; each target PC
        // becomes a `loop_header` site.
        let loop_header_pcs = find_loop_header_pcs(code);
        // Phase A.1/A.2: pre-scanned set of every block-entry PC.  Used
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
        // Task #229 Session 1 slice: record a matching graph op pair
        // (constant-source `int_copy` producing a fresh Int Variable +
        // `setfield_vable_i` consuming it) alongside the SSA emission.
        // The SSA side now mirrors that shape via
        // `ssarepr.fresh_var(Kind::Int, ...)`,
        // lifting `graph_num` toward `ssa_num` as Task #227 Phase 4
        // prepares to flip `flatten_graph(graph, regallocs)` as the
        // source of truth.
        macro_rules! emit_vsd {
            ($depth:expr) => {
                if is_portal {
                    let depth_value = (stack_base_absolute + $depth as usize) as i64;
                    // Graph-side shadow: produce a fresh Int Variable
                    // from a constant-source `int_copy` op and consume it
                    // in a matching `setfield_vable_i` op. Mirrors jtransform.py:844 +
                    // jtransform.py:925 so graph regalloc observes the
                    // liverange of the VSD-sync scratch.
                    // Graph offsets for these synthetic shadow ops use -1
                    // — they're emission-time bookkeeping, not tied to a
                    // Python bytecode PC. `SpaceOperation.offset` is
                    // advisory in regalloc (`regalloc.rs::make_dependencies`
                    // doesn't read it); -1 simply distinguishes them from
                    // real py_pc-anchored ops.
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
                        -1,
                    );
                    emit_vable_setfield_int_const!(
                        portal_frame_reg,
                        VABLE_VALUESTACKDEPTH_FIELD_IDX,
                        depth_value
                    );
                }
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

        macro_rules! emit_load_const_i {
            ($dst:expr, $value:expr $(,)?) => {{
                let dst = $dst;
                let value: i64 = $value;
                let insn = Insn::op_with_result(
                    "int_copy",
                    vec![Operand::ConstInt(value)],
                    Register::new(Kind::Int, dst),
                );
                push_walker_emit(&current_block, insn);
            }};
        }

        // Every site that used
        // to invoke `assembler.ref_return(src)` now also appends an
        // `Insn::Op { opname: "ref_return", args: [Register(Ref, src)] }`
        // to the SSARepr so the future `Assembler::assemble` path
        // (`assembler.rs::dispatch_op:374`) can reproduce the same byte at
        // the Phase 3c switchover. The direct builder call stays until the
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
                // snapshot the EXIT FrameState.
                append_exit_with_state(
                    &current_block.block(),
                    link,
                    &current_state,
                    &mut link_exit_states,
                );
                needs_fallthrough = false;
            }};
        }

        // RPython parity:
        // `flatten.py:161` `self.emitline('goto', TLabel(link.target))` —
        // `assembler.py:220` turns the op into `goto/L`. Pyre labels are
        // integer indices into `labels[]`, one per Python PC; the
        // `TLabel` carries the synthetic name `pc{target_py_pc}` so the
        // Phase 3c dispatch (`assembler.rs::dispatch_op:345`) can resolve
        // it against `builder_label`.
        macro_rules! emit_goto {
            ($target_py_pc:expr) => {{
                let target_py_pc = $target_py_pc;
                let insn = Insn::op(
                    "goto",
                    vec![Operand::TLabel(super::flatten::pc_tlabel(target_py_pc))],
                );
                push_walker_emit(&current_block, insn);
                // `rpython/jit/codewriter/flatten.py:111-112`: an
                // unconditional goto implicitly ends a block so the
                // liveness pass (`liveness.py:68-69`) can reset the alive
                // set.
                push_walker_emit(&current_block, Insn::Unreachable);
                // `flatten.py:161` `self.emitline('goto',
                // TLabel(link.target))` is the serialised view of the
                // same edge.  Phase A.4 retired the back-edge skip
                // (was Phase 4.C); back-edge mergeblock now creates a
                // re-walked supersede block, which the relaxed
                // `live_marker_indices_by_pc` first-takes semantics
                // tolerates.
                mergeblock(
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
                    &mut link_exit_states,
                    &mut pendingblocks,
                    &mut all_walker_blocks,
                );
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
        // the opname *already* surfaces at the runtime layer, Phase 3c's
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
        macro_rules! emit_raise {
            ($src:expr, $evalue:expr, $offset:expr) => {{
                let src = $src;
                let evalue_fv: super::flow::FlowValue = $evalue;
                let offset = $offset;
                let insn = Insn::op("raise", vec![Operand::reg(Kind::Ref, src)]);
                push_walker_emit(&current_block, insn);
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
                append_exit_with_state(
                    &current_block.block(),
                    link,
                    &edge_state,
                    &mut link_exit_states,
                );
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
                // snapshot the EXIT state (same
                // reasoning as `emit_raise!`).
                append_exit_with_state(
                    &current_block.block(),
                    link,
                    &current_state,
                    &mut link_exit_states,
                );
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
                attach_catch_exception_edge(
                    &mut graph,
                    &current_block.block(),
                    &catch_landing_blocks[&catch_label],
                    &current_state,
                    &mut link_exit_states,
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
                // whose only side-effect is to `append_exit_with_state`
                // a `pending_block → pending_block` self-loop, leaving
                // every empty pending block with two outgoing edges
                // (the self-loop + the next PC's fallthrough) and no
                // exitswitch.  RPython's `flowcontext.py:407-475` walks
                // per-block, never invoking the joinpoint-merge path
                // when "entering" a block — pyre's PC-sequential walker
                // is the adaptation, but the join check belongs only on
                // PC transitions, not on PC entry.
                // Phase A.2: force a block boundary when the walker
                // reaches a pre-scanned branch target PC sequentially.
                // Without this, current_block would continue past the
                // boundary via arm 3's self-registration, and a later
                // back-edge to py_pc would create an orphan via
                // make_next_block (next_offset mismatch in mergeblock's
                // union loop).  The block-start case is excluded by
                // checking that current_block.framestate.next_offset !=
                // py_pc — we are not yet at start, so we need to close
                // current_block at the boundary.
                let force_branch_boundary = needs_fallthrough
                    && branch_target_pcs.contains(&py_pc)
                    && current_block
                        .framestate()
                        .map_or(true, |fs| fs.next_offset != py_pc);
                let new_block = if needs_fallthrough
                    && (current_state.next_offset != py_pc || force_branch_boundary)
                {
                    mergeblock(
                        code,
                        &mut graph,
                        &mut joinpoints,
                        &current_block,
                        &current_state,
                        py_pc,
                        &mut link_exit_states,
                        &mut pendingblocks,
                        &mut all_walker_blocks,
                    )
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
                    // Task #227.5 per-block contiguous walker: when
                    // the gate is on AND the joinpoint target differs
                    // from `current_block`, queue the target to
                    // `pendingblocks` (mergeblock-path queuing is
                    // already done by mergeblock itself; joinpoint
                    // match doesn't push automatically).
                    if target != current_block && !pendingblocks.iter().any(|b| b == &target) {
                        pendingblocks.push_front(target.clone());
                    }
                    target
                } else if !current_block.dead() {
                    // Natural fall-through within current_block — Phase
                    // A.1+A.2 cover branch targets via boundary force.
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
                    // Fail-loud per RPython invariant; a follow-up
                    // slice deletes the arm once the bench / lib suite
                    // confirms.
                    panic!(
                        "emit_mark_label_pc!(py_pc={}): no live current_block \
                         and no joinpoint candidate — invariant violation",
                        py_pc,
                    );
                };
                // Task #227.5 yield-on-switch: when the gate is on and
                // `new_block` differs from `current_block`, set the
                // `block_switch_pending` flag and SKIP the inline
                // switch (the new block has been queued to
                // `pendingblocks` above; the outer walker loop will
                // pop it and process its emit sequence
                // contiguously).  The inner for-loop body checks
                // `block_switch_pending` after each per-PC emit and
                // breaks, yielding control.  Default (gate off):
                // switch inline as before, preserving the PC-
                // sequential walker's behaviour.
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
                        let goto_insn = Insn::op(
                            "goto",
                            vec![Operand::TLabel(super::flatten::pc_tlabel(py_pc))],
                        );
                        push_walker_emit(&current_block, goto_insn);
                        push_walker_emit(&current_block, Insn::Unreachable);
                    }
                    block_switch_pending = true;
                } else {
                    // No switch — same block continues at py_pc.  Just
                    // mark the Label so post-walk dispatch resolves
                    // `pcN` to this position.  No goto / fallthrough
                    // bookkeeping needed because `current_block`
                    // doesn't change.
                    needs_fallthrough = true;
                    push_walker_emit(&current_block, Insn::pc_anchor(py_pc));
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
                current_block = catch_landing_blocks[&landing_label].clone();
                if let Some(state) = current_block.framestate() {
                    current_state = state;
                }
                needs_fallthrough = true;
                // Task #227.5 emission-order tracking: push the catch
                // landing block to `all_walker_blocks` AT FIRST EMIT
                // (not at creation) so the drain order reflects
                // walker emission order — catch landings emit after
                // the main walker loop per `codewriter.rs::6907+`,
                // so creation-order tracking would misalign with
                // ssarepr.insns ordering.  Guard against double-push:
                // a single catch landing may be entered multiple
                // times if multiple catch sites share a landing
                // label (unusual but possible per the catch_sites
                // dedup at codewriter.rs:catch_sites).
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
                let insn = Insn::op(
                    "goto_if_not",
                    vec![
                        Operand::reg(Kind::Int, cond),
                        Operand::TLabel(super::flatten::pc_tlabel(py_pc)),
                    ],
                );
                push_walker_emit(&current_block, insn);
                // `flatten.py:240-267` linkfalse mergeblock.  Phase A.4
                // retired the back-edge skip so this fires on every
                // emit, matching upstream's `set_branch` semantic.
                mergeblock(
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
                    &mut link_exit_states,
                    &mut pendingblocks,
                    &mut all_walker_blocks,
                );
            }};
        }
        macro_rules! emit_goto_if_not_int_is_zero {
            ($cond:expr, $py_pc:expr) => {{
                let cond = $cond;
                let py_pc = $py_pc;
                let insn = Insn::op(
                    "goto_if_not_int_is_zero",
                    vec![
                        Operand::reg(Kind::Int, cond),
                        Operand::TLabel(super::flatten::pc_tlabel(py_pc)),
                    ],
                );
                push_walker_emit(&current_block, insn);
                // same as `emit_goto_if_not!` — the
                // specialised `int_is_zero` form is the pyre-port of
                // `flatten.py:247` `goto_if_not_int_is_zero`; Link
                // shape is identical.
                mergeblock(
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
                    &mut link_exit_states,
                    &mut pendingblocks,
                    &mut all_walker_blocks,
                );
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
        // thread that result through downstream graph ops (Phase A6 —
        // `emit_residual_call` arg shadow), so emitting an unused
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
                let insn = Insn::op_with_result(
                    "getfield_vable_r",
                    vec![
                        Operand::reg(Kind::Ref, vable_reg),
                        Operand::descr_vable_static_field(field_idx),
                    ],
                    Register::new(Kind::Ref, dst),
                );
                push_walker_emit(&current_block, insn);
                // Graph dual-write threads `frame_var.into()` which is
                // only a startblock inputarg when `is_portal` (per
                // `graph_entry_inputargs(code, is_portal)`).  Non-portal
                // graphs would record an op reading a Variable that has
                // no producer, violating upstream's well-formedness; gate
                // accordingly.  Returns `Option<Variable>` so callsites
                // that need the graph identity for downstream
                // dual-writes can thread the same Variable; non-portal
                // callees skip the graph emit and return `None`.
                if is_portal {
                    Some(emit_graph_op_with_result(
                        &mut graph,
                        &current_block.block(),
                        "getfield_vable_r",
                        vable_getfield_ref_graph_args(frame_var.into(), field_idx),
                        Kind::Ref,
                        -1,
                    ))
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
        // RPython-orthodox vable arrayitem shapes (Slices 1+2+3+4
        // fully landed for `setarrayitem_vable_r` and
        // `getarrayitem_vable_r`).  Upstream
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

        // RPython parity: `flatten.py:333`
        // `self.emitline('%s_copy' % kind, v, "->", w)` emits the
        // register-to-register move as `ref_copy` when `kind == 'ref'`;
        // `assembler.py:220` turns it into the bytecode key
        // `ref_copy/r>r`. The SSARepr arg list follows the upstream
        // `(src, '->', dst)` shape via `op_with_result`.
        //
        // RPython generates `ref_copy` ONLY at flatten.py:320 during
        // link renaming (`GraphFlattener::insert_renamings`), never as
        // a flow graph SpaceOperation.  Walker MUST NOT record a
        // graph-side `ref_copy` op.
        macro_rules! emit_ref_copy {
            ($dst:expr, $src:expr) => {{
                let dst = $dst;
                let src = $src;
                let insn = Insn::op_with_result(
                    "ref_copy",
                    vec![Operand::reg(Kind::Ref, src)],
                    Register::new(Kind::Ref, dst),
                );
                push_walker_emit(&current_block, insn);
            }};
        }

        // `flatten.py:333-334` parity for `ref_copy` with a ConstRef source.
        // Used when opcode semantics push a real `None`, not the internal
        // CALL `NULL` sentinel.  Same graph-side prohibition as
        // `emit_ref_copy!`.
        macro_rules! emit_ref_const_copy {
            ($dst:expr, $value:expr) => {{
                let dst = $dst;
                let value = $value;
                let insn = Insn::op_with_result(
                    "ref_copy",
                    vec![Operand::ConstRef(value)],
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
            ($depth:ident, $src:expr, $src_value:expr) => {{
                let src_reg = $src;
                let src_value: super::flow::FlowValue = $src_value;
                emit_ref_copy!(stack_base + $depth, src_reg);
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
                        -1,
                    );
                    emit_vable_setarrayitem_ref_const_idx!(
                        portal_frame_reg,
                        0_u16,
                        depth_value,
                        src_reg
                    );
                }
                $depth += 1;
                emit_vsd!($depth);
            }};
        }

        // Tier 4 Epic A — null_ref_reg → ConstRef(PY_NULL) migration.
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
            ($depth:ident, $value:expr) => {{
                let value: i64 = $value;
                debug_assert_eq!(
                    value,
                    pyre_object::PY_NULL as i64,
                    "emit_pushvalue_ref_const: only PY_NULL is supported today; \
                     graph shadow uses Constant::none() per assembler.py:109",
                );
                emit_ref_const_copy!(stack_base + $depth, value);
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
                        -1,
                    );
                    emit_vable_setarrayitem_ref_const_idx_const_value!(
                        portal_frame_reg,
                        0_u16,
                        depth_value,
                        value
                    );
                }
                $depth += 1;
                emit_vsd!($depth);
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
            ($depth:ident) => {{
                // Do not change this to a plain `$depth -= 1` until the
                // portal stack-depth model is fully aligned with PyPy's
                // assert-on-underflow behavior.  The direct parity change
                // makes `synth/comprehensions` crash on both dynasm and
                // cranelift (`python3 pyre/check.py --synthetic-only
                // --synthetic-pattern comprehensions.py`).
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
                        -1,
                    );
                    emit_vable_setarrayitem_ref_const_idx_const_value!(
                        portal_frame_reg,
                        0_u16,
                        depth_value,
                        pyre_object::PY_NULL as i64
                    );
                }
                emit_vsd!($depth);
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
            ($depth:ident, $reg:expr) => {{
                let reg = $reg;
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
                        -1,
                    );
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
                        stack_base + $depth
                    );
                    current_state.stack.push(loaded);
                    $depth += 1;
                    emit_vsd!($depth);
                } else {
                    let loaded = current_state
                        .locals_w
                        .get(reg as usize)
                        .and_then(|value| value.clone())
                        .unwrap_or_else(|| fresh_ref_value(&mut graph));
                    current_state.stack.push(loaded.clone());
                    emit_pushvalue_ref!($depth, reg, loaded);
                }
            }};
        }

        // jtransform.py:1898 `do_fixed_list_setitem` vable case +
        // post-store reg_N mirror. STORE_FAST and its super-inst
        // relatives (StoreFastLoadFast, StoreFastStoreFast) all
        // perform the same dual-write pair: when the frame is
        // portal-virtualizable, write `stored_reg` into the vable
        // array slot for the local; in every case, `ref_copy` it
        // into reg_N so super-inst consumers reading reg_N directly
        // (LoadFastLoadFast / LoadFastBorrowLoadFastBorrow) see the
        // post-store value. The reg==vable invariant established
        // here is the foundation for the LFLF vable flip — see
        // memo super_inst_candidate1_probe_scope_2026_04_23.
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
                }
                emit_ref_copy!(reg, stored_reg);
            }};
        }

        // Seed the outer walker queue.  Matches
        // `flowcontext.py:401` `pendingblocks = deque([startblock])`.
        pendingblocks.push_back(current_block.clone());

        // flowcontext.py:402-405 `while self.pendingblocks: block =
        // self.pendingblocks.popleft(); if not block.dead: self.record_block(block)`.
        while let Some(pending_block) = pendingblocks.pop_front() {
            if pending_block.dead() {
                continue;
            }
            let pending_state = pending_block
                .framestate()
                .expect("pending block must carry a FrameState (flowcontext.py:408)");
            let start_pc = pending_state.next_offset;
            // Phase A.4 mirrors upstream's `flowcontext.py:404 if not
            // block.dead: record_block(block)` identity-only check.
            // Supersede re-walks under widened framestate produce
            // duplicate `Label("pcN")` + `-live-` pairs in ssarepr;
            // `pc_anchor_positions` (first-wins) and
            // `live_marker_indices_by_pc` (first-takes) keep the
            // original dead block's bytes canonical for `pcN`
            // dispatch.
            current_block = pending_block;
            current_state = pending_state;
            current_depth = current_state.stack.len() as u16;
            needs_fallthrough = true;
            // Task #227.5 per-block walker: reset switch flag at the
            // start of every new block iteration so a previous
            // block's queued switch doesn't bleed into this one.
            block_switch_pending = false;
            // Note — upstream `flowcontext.py:407-416`
            // drives per-block op accumulation via `while True:
            // handle_bytecode(...)` until a terminator, then
            // `record_block` assigns `block.operations` from the
            // recorder.  Pyre iterates PCs linearly because the walker
            // emits directly into program-wide `ssarepr.insns`.
            // Convergence: Task #227 Phase 4 + Task #212 (per-block
            // `record_block` + post-walk `flatten_graph(graph,
            // regallocs)` per `codewriter.py:44-67`).
            for py_pc in start_pc..num_instrs {
                // Exception handler entry: Python resets stack depth to the
                // handler's specified depth and arrives only from
                // `catch_exception` edges, not from sequential fallthrough.
                if handler_depth_at.contains_key(&py_pc) {
                    if let Some(handler_state) = handler_entry_state_from_catch_sites(
                        code,
                        &mut graph,
                        &catch_sites,
                        &catch_landing_blocks,
                        py_pc,
                    ) {
                        current_depth = handler_state.stack.len() as u16;
                        current_state = handler_state;
                        needs_fallthrough = false;
                    } else if let Some(&handler_depth) = handler_depth_at.get(&py_pc) {
                        current_depth = handler_depth;
                    }
                }
                // RPython flatten.py: Label(block) at block entry
                emit_mark_label_pc!(py_pc);
                // Task #227.5 yield-on-switch: if `emit_mark_label_pc!`
                // detected a block boundary at this PC and queued the
                // new block to `pendingblocks`, break the inner loop
                // and let the outer walker pop the new block in its
                // own iteration.  The new block's outer iter then
                // re-enters at PC=py_pc and the same
                // `emit_mark_label_pc!` resolves to the new
                // current_block.
                if block_switch_pending {
                    break;
                }
                // Task #227.5 item 6: PcAnchor walker emit retired.
                // The `Label("pc{N}")` that `emit_mark_label_pc!`
                // emitted is the per-PC anchor for
                // `pc_anchor_positions` / `live_marker_indices_by_pc`
                // (Label-based fallback).  `remove_repeated_live` is
                // updated to break its merge run on per-PC Labels so
                // each PC keeps its own `-live-` marker.  Aligns
                // with upstream RPython which has no per-PC anchor
                // concept — Labels are the boundary markers.
                depth_at_pc[py_pc] = current_depth;
                emit_live_placeholder!();

                if loop_header_pcs.contains(&py_pc) {
                    if is_portal {
                        // interp_jit.py:64 portal contract:
                        //   greens = ['next_instr', 'is_being_profiled', 'pycode']
                        //   reds = ['frame', 'ec']
                        //
                        // Graph side: record the upstream-matched 7-arg
                        // SpaceOperation per
                        // `jtransform.py:1690-1712 handle_jit_marker__jit_merge_point`.
                        // The graph carries the full
                        // `[jd_index, 3 green ListOfKinds, 3 red ListOfKinds]`
                        // shape, and `GraphFlattener::serialize_op`
                        // lowers that same shape into SSARepr — the byte
                        // side is no longer pyre's old 3-list shorthand.
                        // Assembler / blackhole / backend (`assembler.rs:712`)
                        // assert the canonical 7-arg form on the way out.
                        //
                        // DEVIATION (β.2 migration target — see plan
                        // `~/.claude/plans/inline-call-portal-migration.md`
                        // + memory `inline_call_portal_beta2_audit_2026_05_03.md`):
                        //
                        // pycode is currently carried as an `Opaque(Ref)`
                        // Constant at the graph layer, then lowered by the
                        // `lower_constant` callback below to `Operand::ConstRef`
                        // which routes through `builder.add_const_r` and bakes
                        // the user CodeObject pointer into the per-CodeObject
                        // jitcode's constants_r pool. This is one of the
                        // primary sources of the `drained.r > portal_canonical.r`
                        // divergence measured in
                        // `inline_call_portal_b5_probe_a_2026_05_03.md`
                        // (nbody drained.r=163 vs portal canonical.r=0).
                        //
                        // RPython orthodox: pycode is a green argument to
                        // `portal_runner` (`pypyjit/interp_jit.py:67-78
                        // PyPyJitDriver greens=['next_instr','is_being_profiled',
                        // 'pycode']`), present at runtime in a calling-convention
                        // register. To migrate, replace the `Constant::opaque(…)`
                        // in `portal_jit_merge_point_graph_args` (line 152) with
                        // a Variable produced by emitting `getfield_vable_r
                        // frame, PYCODE_FIELD_IDX → pycode_var` immediately
                        // before `serialize_op`. The `lower_variable`
                        // closure (line 4361) gains a third arm mapping
                        // `pycode_var.id` to the dst register of the new
                        // getfield. Empirical Probe A re-run after the change
                        // should show `drained.r` reduced by 1 per jit_merge_point
                        // emission (~10-30 for nbody).
                        //
                        // Existing pattern reusable: the `emit_vable_getfield_ref!`
                        // macro at line 3728 already emits the
                        // `getfield_vable_r [vable_reg, descr_vable_static_field(idx)]
                        // → Register(Ref, dst)` shape that this migration needs.
                        // PYFRAME_VABLE_FIELDS in `virtualizable_spec.rs` enumerates
                        // the field indices (pycode is field 1 per
                        // `interp_jit.py:25-31`).
                        let jdindex = portal_jd_index
                            .expect("portal jit_merge_point requires a registered jitdriver");
                        // β.2.1 (plan inline-call-portal-migration.md): instead
                        // of baking `w_code` into the runtime constants_r pool
                        // via `Operand::ConstRef`, load `frame.pycode` at
                        // runtime into a fresh scratch register and reference
                        // that register from `jit_merge_point`'s pycode green
                        // arg. RPython orthodox parity per
                        // `pypyjit/interp_jit.py:67 reds=['frame','ec']` +
                        // `interp_jit.py:25 _virtualizable_=['..., 'pycode',
                        // ...]`: pycode is recovered from the live frame at
                        // every merge point, so the trace key is the runtime
                        // value rather than a build-time constant.
                        let scratch_pycode_reg = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                        emit_vable_getfield_ref!(
                            portal_frame_reg,
                            scratch_pycode_reg,
                            VABLE_CODE_FIELD_IDX
                        );
                        let graph_args = portal_jit_merge_point_graph_args(
                            &graph,
                            py_pc,
                            w_code as *const (),
                            jdindex,
                        );
                        let graph_op = emit_graph_op_void(
                            &current_block.block(),
                            "jit_merge_point",
                            graph_args,
                            py_pc as i64,
                        );
                        // Task #227.3 capture-before/after-len so the
                        // GraphFlattener-mediated push lands in both
                        // `ssarepr.insns` AND `current_block`'s per-
                        // block accumulator.
                        let pre_len = ssarepr.insns.len();
                        GraphFlattener::new_with_constant_lowering(
                            &mut ssarepr,
                            |v: super::flow::Variable| {
                                if v.id == frame_var.id {
                                    Register::new(Kind::Ref, portal_frame_reg)
                                } else if v.id == ec_var.id {
                                    Register::new(Kind::Ref, portal_ec_reg)
                                } else {
                                    panic!(
                                        "portal jit_merge_point: unexpected graph Variable {v:?} \
                                     (only portal frame/ec expected)"
                                    )
                                }
                            },
                            |c: &super::flow::Constant| match (&c.value, c.kind) {
                                (super::flow::ConstantValue::Signed(value), Some(Kind::Int)) => {
                                    Operand::ConstInt(*value)
                                }
                                (super::flow::ConstantValue::Opaque(_), Some(Kind::Ref)) => {
                                    // β.2.1: pycode green arg references the
                                    // scratch register pre-loaded with
                                    // `frame.pycode` via `emit_vable_getfield_ref!`
                                    // above. RPython orthodox parity
                                    // (`pypyjit/interp_jit.py:67-78` PyPyJitDriver
                                    // `greens=[..., 'pycode']`): pycode is read
                                    // from the live frame at every merge point,
                                    // not baked into the per-CodeObject
                                    // constants_r pool.
                                    Operand::reg(Kind::Ref, scratch_pycode_reg)
                                }
                                other => {
                                    panic!("portal jit_merge_point: unexpected Constant {other:?}")
                                }
                            },
                        )
                        .serialize_op(&graph_op);
                        // Task #227.3 mirror GraphFlattener-mediated
                        // jit_merge_point emit into per-block accumulator.
                        for insn in ssarepr.insns[pre_len..].iter().cloned() {
                            current_block.push_insn(insn);
                        }
                    }
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
                    Instruction::LoadFast { var_num } | Instruction::LoadFastBorrow { var_num } => {
                        let reg = var_num.get(op_arg).as_usize() as u16;
                        emit_load_fast_ref!(current_depth, reg);
                    }

                    // jtransform.py:1898 do_fixed_list_setitem vable case:
                    // Portal frames treat `locals_cells_stack_w` as the sole
                    // storage for locals — setarrayitem_vable_r writes from
                    // the value-stack slot directly, so no register-per-local
                    // shadow exists. Non-portal frames keep ref_copy (no vable
                    // in scope).
                    Instruction::StoreFast { var_num } => {
                        let reg = var_num.get(op_arg).as_usize() as u16;
                        let stored_reg = emit_popvalue_ref!(current_depth);
                        let stored = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        if is_portal {
                            // Graph dual-write of jtransform.py:1898
                            // `do_fixed_list_setitem` — STORE_FAST →
                            // `setarrayitem_vable_r(locals_cells_stack_w,
                            // local_slot, w_value)`.  `frame_var` is a
                            // startblock inputarg only when `is_portal`.
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
                        if let Some(slot) = current_state.locals_w.get_mut(reg as usize) {
                            *slot = Some(stored);
                        }
                    }

                    Instruction::LoadSmallInt { i } => {
                        let val = i.get(op_arg) as u32 as i64;
                        // A-slice 1 (Task #224): call writes result directly
                        // to the target stack slot. Safe because the only
                        // call input is a literal constant (no Ref conflict)
                        // and no post-call op reads from that stack slot
                        // before the next opcode's frontend push.
                        // `make_three_lists` (jtransform.py:437-445) admits
                        // `Variable | Constant` directly, so the constant
                        // reaches `expect_list_regs_or_pool`
                        // (assembler.rs:1736-1784) without a scratch register.
                        // Task #48 micro-slice 10: box_int_fn factor
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
                        // dual-write below is NOT retired in this
                        // slice — incremental factor refactor only.
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
                        // Phase 4 walker-orthodoxy: graph residual_call
                        // dual-write fires unconditionally.  `box_int_fn`
                        // takes only a literal Int as input, so no
                        // frame_var or other portal-only Variable is
                        // threaded — the graph op is well-formed for
                        // every CodeWriter regardless of is_portal.
                        let boxed = record_residual_call_graph_op(
                            &mut graph,
                            &current_block.block(),
                            box_int_fn_idx,
                            CallFlavor::Plain,
                            vec![super::flow::Constant::signed(val).into()],
                            vec![],
                            vec![],
                            vec![Kind::Int],
                            ResKind::Ref,
                            py_pc as i64,
                        );
                        let stack_value = boxed
                            .map(super::flow::FlowValue::from)
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        current_state.stack.push(stack_value);
                        current_depth += 1;
                        emit_vsd!(current_depth);
                    }

                    Instruction::LoadConst { consti } => {
                        let idx = consti.get(op_arg).as_usize();
                        let dst_slot = stack_base + current_depth;
                        // jtransform.py: getfield_vable_r for pycode (field 1)
                        // — write straight to the target stack slot. The slot
                        // is the next push destination (currently free); the
                        // call below reads it as input and overwrites it with
                        // the load_const result. SSA-wise: write1 (getfield)
                        // → read (call input) → write2 (call result) — same
                        // input-output share pattern as Sessions 1-3.
                        // Portal vable sync at this slot relies on the next
                        // opcode's pushvalue (LoadConst's existing A-slice 2
                        // elision documented at LoadGlobal's caveat).
                        let pycode_graph_var = emit_vable_getfield_ref!(
                            portal_frame_reg,
                            dst_slot,
                            VABLE_CODE_FIELD_IDX
                        );
                        // Task #48 micro-slice 7: LoadConst factor
                        // refactor.  The prior `emit_residual_call(
                        // load_const_fn_idx, ...)` call is replaced by
                        // a single direct push of
                        // `build_load_const_fn_residual_call_ir_r_insn`,
                        // which produces the same `residual_call_ir_r(
                        // ConstInt(fn_idx), ListI([ConstInt(idx)]),
                        // ListR([Reg(pycode)]), Descr) → Reg(dst)` Insn
                        // shape `emit_residual_call_shape` would have
                        // produced.  LoadConst has no frontend HLOp
                        // (no `lower_load_const_hlop_to_insn` arm), so
                        // the matching graph dual-write below is NOT
                        // retired in this slice — this is incremental
                        // factor refactor only, prepping the future
                        // `flatten_graph(graph, regallocs)` migration.
                        // The helper hardcodes `CallFlavor::Plain`
                        // matching the production source at
                        // codewriter.rs:2215, so `load_const_fn_flavor`
                        // is no longer threaded into the SSARepr emit.
                        push_walker_emit(
                            &current_block,
                            super::flatten::build_load_const_fn_residual_call_ir_r_insn(
                                load_const_fn_idx,
                                idx as i64,
                                dst_slot,
                                dst_slot,
                            ),
                        );
                        // Graph-side `residual_call_ir_r` for
                        // `load_const_fn(pycode:Ref, idx:Int) → Ref`.
                        // RPython `flowcontext.py:135-139` keeps the
                        // residual_call result as the consumer's input
                        // (no separate fresh placeholder); the call is
                        // recorded only when the symbolic stack is
                        // about to consume its result Variable.
                        // `frontend_constant_flow_value` recognises a
                        // small set of constants (e.g. `None`) directly
                        // as Ref-kind FlowValues — for those, the
                        // upstream optimizer does not record a
                        // residual_call at all (jtransform.py inlines
                        // the constant), so skip the graph dual-write
                        // to avoid an orphan call result.
                        let raw_value = code
                            .constants
                            .get(idx)
                            .and_then(frontend_constant_flow_value);
                        let recognised_ref = match &raw_value {
                            Some(super::flow::FlowValue::Constant(c))
                                if c.kind == Some(Kind::Ref) =>
                            {
                                Some(super::flow::FlowValue::Constant(c.clone()))
                            }
                            Some(super::flow::FlowValue::Variable(v))
                                if v.kind == Some(Kind::Ref) =>
                            {
                                Some(super::flow::FlowValue::Variable(v.clone()))
                            }
                            _ => None,
                        };
                        let value = if let Some(constant_value) = recognised_ref {
                            constant_value
                        } else if let Some(pycode_var) = pycode_graph_var {
                            let loaded = record_residual_call_graph_op(
                                &mut graph,
                                &current_block.block(),
                                load_const_fn_idx,
                                CallFlavor::Plain,
                                vec![super::flow::Constant::signed(idx as i64).into()],
                                vec![pycode_var.into()],
                                vec![],
                                vec![Kind::Ref, Kind::Int],
                                ResKind::Ref,
                                py_pc as i64,
                            );
                            loaded
                                .map(super::flow::FlowValue::from)
                                .unwrap_or_else(|| fresh_ref_value(&mut graph))
                        } else {
                            // is_portal=false: no graph dual-write at all.
                            fresh_ref_value(&mut graph)
                        };
                        current_state.stack.push(value);
                        current_depth += 1;
                        emit_vsd!(current_depth);
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
                        emit_load_fast_ref!(current_depth, reg_a);
                        emit_load_fast_ref!(current_depth, reg_b);
                    }

                    // Super-instruction STORE_FAST; LOAD_FAST: pop TOS into
                    // idx_1 (store), then push idx_2 (load). Net depth 0.
                    // Portal: store via setarrayitem_vable_r, load via
                    // getarrayitem_vable_r. Non-portal: ref_copy for both halves.
                    Instruction::StoreFastLoadFast { var_nums } => {
                        let pair = var_nums.get(op_arg);
                        let store_reg = u32::from(pair.idx_1()) as u16;
                        let load_reg = u32::from(pair.idx_2()) as u16;
                        let stored_reg = emit_popvalue_ref!(current_depth);
                        let stored = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
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
                            let stack_slot = (stack_base_absolute + current_depth as usize) as i64;
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
                            if let Some(slot) = current_state.locals_w.get_mut(store_reg as usize) {
                                *slot = Some(stored);
                            }
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
                            current_state.stack.push(loaded);
                            current_depth += 1;
                            emit_vsd!(current_depth);
                        } else {
                            if let Some(slot) = current_state.locals_w.get_mut(store_reg as usize) {
                                *slot = Some(stored);
                            }
                            let loaded = current_state
                                .locals_w
                                .get(load_reg as usize)
                                .and_then(|value| value.clone())
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            current_state.stack.push(loaded.clone());
                            emit_pushvalue_ref!(current_depth, load_reg, loaded);
                        }
                    }

                    // STORE_SUBSCR: stack [value, obj, key] → obj[key] = value
                    Instruction::StoreSubscr => {
                        // A-slice 4: pass stack slots directly as call args,
                        // retiring obj_tmp0/obj_tmp1/arg_regs_start staging.
                        // Inputs are read by the backend ABI into call regs
                        // before the call executes; no write-back conflicts
                        // because ResKind::Void.
                        current_depth -= 1;
                        emit_vsd!(current_depth);
                        let key_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        let key_reg = stack_base + current_depth;
                        current_depth -= 1;
                        emit_vsd!(current_depth);
                        let obj_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        let obj_reg = stack_base + current_depth;
                        current_depth -= 1;
                        emit_vsd!(current_depth);
                        let stored_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
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
                        let _ = emit_popvalue_ref!(current_depth);
                        let _ = current_state.stack.pop();
                        // flowcontext.py:891 `self.popvalue()`; regalloc.py:
                        // discard = just decrement depth, no bytecode.
                    }

                    Instruction::PushNull => {
                        current_state.stack.push(null_stack_sentinel());
                        emit_pushvalue_ref_const!(current_depth, pyre_object::PY_NULL as i64);
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
                        let rhs_reg = emit_popvalue_ref!(current_depth);
                        let rhs_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        // Pop lhs.
                        let lhs_reg = emit_popvalue_ref!(current_depth);
                        let lhs_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        let result_value = emit_frontend_binary(
                            &mut graph,
                            &current_block.block(),
                            op_kind,
                            lhs_value,
                            rhs_value,
                            py_pc as i64,
                        );
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
                        current_state.stack.push(result_value.into());
                        current_depth += 1;
                        emit_vsd!(current_depth);
                    }

                    // jtransform.py: rewrite_op_int_lt, optimize_goto_if_not
                    Instruction::CompareOp { opname } => {
                        // Same stack-direct pattern as BinaryOp — see its comment.
                        let op_kind = opname.get(op_arg);
                        let op_val = compare_op_tag(op_kind);
                        let rhs_reg = emit_popvalue_ref!(current_depth);
                        let rhs_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        let lhs_reg = emit_popvalue_ref!(current_depth);
                        let lhs_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        let result_value = emit_frontend_compare(
                            &mut graph,
                            &current_block.block(),
                            op_kind,
                            lhs_value,
                            rhs_value,
                            py_pc as i64,
                        );
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
                        current_state.stack.push(result_value.into());
                        current_depth += 1;
                        emit_vsd!(current_depth);
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
                        // A-slice 7: truth_fn reads cond directly from the popped
                        // stack slot; `popvalue_ref` leaves the value at
                        // `stack_base + current_depth` (the slot below the new
                        // TOS), so there is no staging copy — mirrors upstream
                        // flatten.py:240-260 which feeds the Variable straight to
                        // `goto_if_not`.
                        let cond_reg = emit_popvalue_ref!(current_depth);
                        let cond_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
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
                            emit_goto_if_not!(scratch_truth, target_py_pc);
                            // Phase A.4 retired the back-edge gate;
                            // emit_goto_if_not's mergeblock now always
                            // appends a fresh linkfalse, so case stamp
                            // always applies.
                            set_last_bool_exitcase(&current_block.block(), false);
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
                                &mut link_exit_states,
                                &mut pendingblocks,
                                &mut all_walker_blocks,
                            );
                            set_last_bool_exitcase(&current_block.block(), true);
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
                        // A-slice 7: see PopJumpIfFalse — no obj_tmp0 staging
                        // needed; the residual call reads the popped stack slot.
                        let cond_reg = emit_popvalue_ref!(current_depth);
                        let cond_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
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
                        // Task #48 micro-slice 5: BOOL family
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
                            // POP_JUMP_IF_TRUE: emit_goto_if_not(fallthrough)
                            // always targets PC+1 (forward, never emitted),
                            // Phase A.4 retired the back-edge gate;
                            // emit_goto's mergeblock now always appends
                            // linktrue, so case stamp always applies.
                            emit_goto_if_not!(scratch_truth, fallthrough_py_pc);
                            set_last_bool_exitcase(&current_block.block(), false);
                            emit_goto!(target_py_pc);
                            set_last_bool_exitcase(&current_block.block(), true);
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
                        if let Some(target_py_pc) = backward_jump_target(code, py_pc, instr, op_arg)
                        {
                            if target_py_pc < num_instrs {
                                emit_goto!(target_py_pc);
                            }
                        }
                    }

                    // flatten.py: int_return / ref_return
                    Instruction::ReturnValue => {
                        let retval_reg = emit_popvalue_ref!(current_depth);
                        let retval = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        // A-slice 3: ref_return reads from the stack slot
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
                        // level. Pyre cannot fold runtime globals at compile
                        // time, so the shadow stack receives the runtime
                        // residual_call's result Variable (bound below).
                        let scratch_ns = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                        let scratch_code = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                        // jtransform.py: getfield_vable_r for w_globals (field 3)
                        // and pycode (field 1) — namespace for lookup, code for names.
                        let ns_graph_var = emit_vable_getfield_ref!(
                            portal_frame_reg,
                            scratch_ns,
                            VABLE_NAMESPACE_FIELD_IDX
                        );
                        let code_graph_var = emit_vable_getfield_ref!(
                            portal_frame_reg,
                            scratch_code,
                            VABLE_CODE_FIELD_IDX
                        );
                        // Task #48 micro-slice 8: LoadGlobal factor
                        // refactor.  The prior `emit_residual_call(
                        // load_global_fn_idx, ...)` call is replaced by
                        // a single direct push of
                        // `build_load_global_fn_residual_call_ir_r_insn`,
                        // which produces the matching `residual_call_ir_r(
                        // ConstInt(fn_idx), ListI([ConstInt(namei)]),
                        // ListR([Reg(ns), Reg(code), Reg(frame)]), Descr)
                        // → Reg(scratch_ns)` Insn shape
                        // `emit_residual_call_shape` would have
                        // produced.  LoadGlobal has no frontend HLOp
                        // (no `lower_load_global_hlop_to_insn` arm);
                        // the matching graph dual-write below is NOT
                        // retired in this slice — incremental factor
                        // refactor only, prepping the future
                        // `flatten_graph(graph, regallocs)` migration.
                        // Helper hardcodes `CallFlavor::Plain` matching
                        // the production source at codewriter.rs:2184.
                        push_walker_emit(
                            &current_block,
                            super::flatten::build_load_global_fn_residual_call_ir_r_insn(
                                load_global_fn_idx,
                                raw_namei,
                                scratch_ns,
                                scratch_code,
                                portal_frame_reg,
                                scratch_ns,
                            ),
                        );
                        // Task #46 micro-slice 6: graph-side residual_call
                        // dual-write for load_global_fn(ns:Ref, code:Ref,
                        // frame:Ref, namei:Int) → Ref.  ns and code
                        // Variables come from the preceding
                        // emit_vable_getfield_ref! graph dual-writes; frame
                        // is the portal red variable, matching PyPy's
                        // `_load_global(self, ...)` receiver.
                        // Match helper bind-site flavor at
                        // codewriter.rs:2186 (`load_global_fn`
                        // is `EF_CAN_RAISE`, not virtual-forcing)
                        // — graph dual-write must agree with the
                        // SSA helper so any future
                        // `flatten_graph(graph, regallocs)`
                        // migration sees a single classification.
                        // RPython `flowcontext.py:135-139` keeps the
                        // residual_call result as the consumer's input
                        // (no separate fresh placeholder).
                        let loaded = if let (Some(ns_var), Some(code_var)) =
                            (ns_graph_var, code_graph_var)
                        {
                            record_residual_call_graph_op(
                                &mut graph,
                                &current_block.block(),
                                load_global_fn_idx,
                                CallFlavor::Plain,
                                vec![super::flow::Constant::signed(raw_namei).into()],
                                vec![ns_var.into(), code_var.into(), frame_var.into()],
                                vec![],
                                vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Int],
                                ResKind::Ref,
                                py_pc as i64,
                            )
                        } else {
                            None
                        };
                        let result_value = loaded
                            .map(super::flow::FlowValue::from)
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        // LOAD_GLOBAL with (namei >> 1) & 1: push NULL first.
                        // const-source pushvalue writes the constant directly to
                        // the stack TOS register and (in portal case) to the
                        // vable slot via setarrayitem_vable_r_const, leaving
                        // the scratch regs untouched for the trailing
                        // `emit_pushvalue_ref!(scratch_ns)`.
                        if raw_namei & 1 != 0 {
                            current_state.stack.push(null_stack_sentinel());
                            emit_pushvalue_ref_const!(current_depth, pyre_object::PY_NULL as i64);
                        }
                        current_state.stack.push(result_value.clone());
                        emit_pushvalue_ref!(current_depth, scratch_ns, result_value);
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
                        let arg_regs: Vec<u16> = (0..nargs)
                            .map(|_| ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0)
                            .collect();
                        let mut graph_arg_values_rev = Vec::with_capacity(nargs);
                        for i in (0..nargs).rev() {
                            let arg_reg = emit_popvalue_ref!(current_depth);
                            let arg_value = current_state
                                .stack
                                .pop()
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            emit_ref_copy!(arg_regs[i], arg_reg);
                            graph_arg_values_rev.push(arg_value);
                        }
                        let callable_reg = emit_popvalue_ref!(current_depth);
                        let callable_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        let _ = emit_popvalue_ref!(current_depth); // NULL (discard)
                        let _null_or_self = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));

                        // RPython: bhimpl_recursive_call_i(jdindex, greens, reds)
                        // call_fn(callable, arg0, ...) → result
                        // Parent frame accessed via BH_VABLE_PTR thread-local.
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
                            let graph_call_args: Vec<_> =
                                graph_arg_values_rev.iter().rev().cloned().collect();
                            emit_frontend_simple_call(
                                &mut graph,
                                &current_block.block(),
                                callable_value.clone(),
                                graph_call_args,
                                py_pc as i64,
                            )
                            .into()
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
                            // Task #48 micro-slice 9: CALL family
                            // factor refactor.  The prior
                            // `emit_residual_call(call_fn_N_idx, ...)`
                            // call is replaced by a single direct push
                            // of `build_call_fn_residual_call_r_r_insn`,
                            // which produces the same `residual_call_r_r(
                            // ConstInt(fn_idx), ListR([Reg(callable),
                            // Reg(arg0), ..., Reg(arg_{N-1})]), Descr) →
                            // Reg(dst)` Insn shape
                            // `emit_residual_call_shape` would have
                            // produced (no leading `ListI` because
                            // `args_i` is empty for all-Ref call_args).
                            // CALL has no frontend HLOp with the same
                            // shape (the graph carries `simple_call`
                            // pre-rtype HLOp recorded by
                            // `emit_frontend_simple_call`); the matching
                            // graph dual-write below is NOT retired in
                            // this slice — incremental factor refactor
                            // only, prepping the future
                            // `flatten_graph(graph, regallocs)`
                            // migration.  Helper hardcodes
                            // `CallFlavor::MayForce` matching the
                            // production source at codewriter.rs:2175
                            // and 2238-2245 (every `call_fn_N` is
                            // bound MayForce).
                            push_walker_emit(
                                &current_block,
                                super::flatten::build_call_fn_residual_call_r_r_insn(
                                    fn_idx,
                                    callable_reg,
                                    &arg_regs,
                                    stack_base + current_depth,
                                ),
                            );
                            // Task #46 micro-slice 5: graph-side residual_call
                            // dual-write — call_fn_N signature
                            // `(ref, ref, ..., ref) -> ref` (nargs+1 refs,
                            // callable + nargs).  All-ref args make the
                            // shape `residual_call_r_r`.
                            //
                            // Phase 4 walker-orthodoxy: dual-write fires
                            // regardless of `is_portal`.  Upstream RPython
                            // builds full flow graphs for every function;
                            // pyre's prior is_portal gate dropped graph
                            // coverage for inner-helper CodeWriters.
                            let mut graph_args_r: Vec<super::flow::FlowValue> =
                                Vec::with_capacity(nargs + 1);
                            graph_args_r.push(callable_value);
                            graph_args_r.extend(graph_arg_values_rev.into_iter().rev());
                            let _ = record_residual_call_graph_op(
                                &mut graph,
                                &current_block.block(),
                                fn_idx,
                                CallFlavor::MayForce,
                                vec![],
                                graph_args_r,
                                vec![],
                                vec![Kind::Ref; nargs + 1],
                                ResKind::Ref,
                                py_pc as i64,
                            );
                        }
                        current_state.stack.push(call_result_value);
                        current_depth += 1;
                        emit_vsd!(current_depth);
                    }

                    // Python 3.13: ToBool converts TOS to bool before branch.
                    // No-op in JitCode: the value is already truthy/falsy and
                    // the following PopJumpIfFalse guards on it.
                    Instruction::ToBool => {}

                    // RPython bhimpl_int_neg: -obj via binary_op(0, obj, NB_SUBTRACT)
                    Instruction::UnaryNegative => {
                        let operand_reg = emit_popvalue_ref!(current_depth);
                        let operand_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
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
                        // Task #48 micro-slice 10: box_int_fn factor
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
                        // Task #48 micro-slice 11: UnaryNegative
                        // binary_op_fn factor refactor.  The prior
                        // `emit_residual_call(binary_op_fn_idx, ...)`
                        // is replaced by a single direct push of the
                        // existing `build_binary_op_residual_call_ir_r_insn`
                        // helper introduced in micro-slice 3 — no new
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
                        // Phase 3 walker-orthodoxy: graph dual-writes for
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
                        let zero_graph_var = record_residual_call_graph_op(
                            &mut graph,
                            &current_block.block(),
                            box_int_fn_idx,
                            CallFlavor::Plain,
                            vec![super::flow::Constant::signed(0).into()],
                            vec![],
                            vec![],
                            vec![Kind::Int],
                            ResKind::Ref,
                            py_pc as i64,
                        );
                        if let Some(zero_var) = &zero_graph_var {
                            let binary_result = record_residual_call_graph_op(
                                &mut graph,
                                &current_block.block(),
                                binary_op_fn_idx,
                                CallFlavor::MayForce,
                                vec![super::flow::Constant::signed(subtract_tag as i64).into()],
                                vec![zero_var.clone().into(), operand_value_for_dual.into()],
                                vec![],
                                vec![Kind::Ref, Kind::Ref, Kind::Int],
                                ResKind::Ref,
                                py_pc as i64,
                            );
                            let _ = &binary_result;
                        }
                        current_state.stack.push(negated.into());
                        current_depth += 1;
                        emit_vsd!(current_depth);
                    }

                    // JumpBackwardNoInterrupt reuses `backward_jump_target`:
                    // the encoding differs from JumpBackward (no skip_caches
                    // on the next-PC base) but the helper routes each variant
                    // to its correct arithmetic so pre-scan and emit stay in
                    // lockstep.  interp_jit.py:103 + jtransform.py:1714.
                    instr @ Instruction::JumpBackwardNoInterrupt { .. } => {
                        if let Some(target_py_pc) = backward_jump_target(code, py_pc, instr, op_arg)
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
                                let _ = emit_popvalue_ref!(current_depth);
                                let _ = current_state
                                    .stack
                                    .pop()
                                    .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            }
                            emit_abort_permanent!();
                            current_state.stack.push(fresh_ref_value(&mut graph));
                            current_depth += 1;
                            emit_vsd!(current_depth);
                            continue;
                        }
                        let arg_regs: Vec<u16> = (0..argc)
                            .map(|_| ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0)
                            .collect();
                        let mut item_values_rev = Vec::with_capacity(argc);
                        for i in (0..argc).rev() {
                            let item_reg = emit_popvalue_ref!(current_depth);
                            let item_value = current_state
                                .stack
                                .pop()
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            emit_ref_copy!(arg_regs[i], item_reg);
                            item_values_rev.push(item_value);
                        }
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
                        // Task #48 micro-slice 13: BuildList factor
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
                        current_state.stack.push(result_value.into());
                        current_depth += 1;
                        emit_vsd!(current_depth);
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
                            let reg = emit_popvalue_ref!(current_depth);
                            let step_value = current_state
                                .stack
                                .pop()
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            Some((reg, step_value))
                        } else {
                            None
                        };
                        let stop_reg = emit_popvalue_ref!(current_depth);
                        let stop_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        let start_reg = emit_popvalue_ref!(current_depth);
                        let start_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
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
                        current_state.stack.push(result_value.into());
                        current_depth += 1;
                        emit_vsd!(current_depth);
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
                            // The cause FlowValue is discarded — the exception
                            // edge in the shadow graph carries the exception
                            // value, not the cause.
                            let cause = if n >= 2 {
                                let _cause_fv = current_state
                                    .stack
                                    .pop()
                                    .unwrap_or_else(|| fresh_ref_value(&mut graph));
                                let cause_reg = emit_popvalue_ref!(current_depth);
                                super::flatten::Operand::Register(super::flatten::Register::new(
                                    super::flatten::Kind::Ref,
                                    cause_reg,
                                ))
                            } else {
                                // Tier 4 Epic A: PY_NULL flows directly through
                                // the residual_call's ListOfKind(Ref) as a
                                // raw constant — make_three_lists
                                // (jtransform.py:437-445) admits Constant in
                                // any slot, and the assembler's
                                // dispatch_residual_call routes ConstRef
                                // through the ref constants pool
                                // (assembler.rs:1709-1724
                                // expect_ref_reg_or_pool).
                                super::flatten::Operand::ConstRef(pyre_object::PY_NULL as i64)
                            };
                            // Drop the pre-normalization exception operand from
                            // the shadow stack. The residual call below may
                            // rewrite `raise SomeExcClass` into a fresh
                            // instance, so the exception edge must carry a
                            // NEW FlowValue representing the normalized result.
                            let _ = current_state
                                .stack
                                .pop()
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            let exc_reg = emit_popvalue_ref!(current_depth);
                            // pyopcode.py:711 `exception_is_valid_obj_as_class_w`
                            // normalization + `set_cause` attachment.  Call ABI
                            // reads inputs before writing the result; the
                            // popped stack slot is the natural result
                            // destination, so the call writes the normalized
                            // exception directly into `exc_reg` and
                            // `emit_raise!` reads the same register as its
                            // source. Pattern matches Sessions 1-3 retirements
                            // (Call/UnaryNegative/CheckExcMatch input-side).
                            // Task #48 micro-slice 14: RaiseVarargs
                            // normalize_raise_varargs_fn factor
                            // refactor.  The prior `emit_residual_call(
                            // normalize_raise_varargs_fn_idx, ...)` is
                            // replaced by a single direct push of
                            // `build_normalize_raise_varargs_fn_residual_call_r_r_insn`,
                            // which produces the same `residual_call_r_r(
                            // ConstInt(fn_idx), ListR([Reg(exc),
                            // cause]), Descr) → Reg(exc)` Insn shape.
                            // Helper hardcodes `CallFlavor::MayForce`
                            // matching the production source at
                            // codewriter.rs:2235.  The polymorphic
                            // `cause` Operand (Reg or ConstRef) is
                            // built inline above.  No graph dual-write
                            // exists for normalize_raise_varargs_fn.
                            push_walker_emit(&current_block,
                                super::flatten::build_normalize_raise_varargs_fn_residual_call_r_r_insn(
                                    normalize_raise_varargs_fn_idx,
                                    exc_reg,
                                    cause,
                                    exc_reg,
                                ),
                            );
                            let normalized_exc_fv = fresh_ref_value(&mut graph);
                            emit_raise!(exc_reg, normalized_exc_fv, py_pc as i64);
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
                        let exc_reg = emit_popvalue_ref!(current_depth);
                        let exc_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        // pyopcode.py:786 keeps `exc` in a local after
                        // `popvalue()`.  Mirror that with a scratch register:
                        // the following `push(prev)` writes to the popped
                        // stack slot, so reusing `exc_reg` for the trailing
                        // `push(exc)` would read back `prev` instead of the
                        // caught exception.
                        let scratch_exc = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                        emit_ref_copy!(scratch_exc, exc_reg);
                        let scratch_prev = ssarepr.fresh_var(Kind::Ref, scratch_ref_base).0;
                        // get_current_exception / set_current_exception are TLS read/write —
                        // EF_CANNOT_RAISE per `effectinfo.py:19` (matching call.py:296
                        // getcalldescr's analyzer outcome for non-raising helpers).
                        // Task #48 micro-slice 15: PushExcInfo
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
                        // Task #46 micro-slice 7: graph dual-writes for
                        // both PushExcInfo emits.  get_current_exception
                        // takes no args (shape residual_call_r_r with empty
                        // ListR); set_current_exception is `(exc:Ref)→Void`
                        // (shape residual_call_r_v).
                        // Phase 4 walker-orthodoxy: TLS-only helpers,
                        // no frame_var threading.  Match helper
                        // bind-site flavors at codewriter.rs:2207-2217
                        // — both current-exception helpers are TLS
                        // read/write only and statically prove "no GC
                        // heap touched", binding `PlainCannotRaiseNoHeap`
                        // for the analyzer-equivalent `EF_CANNOT_RAISE
                        // + empty raw frozensets + can_collect=false`
                        // shape (`effectinfo.py:281-283`).
                        let _ = record_residual_call_graph_op(
                            &mut graph,
                            &current_block.block(),
                            get_current_exception_fn_idx,
                            CallFlavor::PlainCannotRaiseNoHeap,
                            vec![],
                            vec![],
                            vec![],
                            vec![],
                            ResKind::Ref,
                            py_pc as i64,
                        );
                        let _ = record_residual_call_graph_op(
                            &mut graph,
                            &current_block.block(),
                            set_current_exception_fn_idx,
                            CallFlavor::PlainCannotRaiseNoHeap,
                            vec![],
                            vec![exc_value.clone()],
                            vec![],
                            vec![Kind::Ref],
                            ResKind::Void,
                            py_pc as i64,
                        );
                        let prev_value = fresh_ref_value(&mut graph);
                        current_state.stack.push(prev_value.clone());
                        emit_pushvalue_ref!(current_depth, scratch_prev, prev_value);
                        current_state.stack.push(exc_value.clone());
                        emit_pushvalue_ref!(current_depth, scratch_exc, exc_value);
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
                        let match_type_reg = emit_popvalue_ref!(current_depth);
                        let match_type_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
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
                        // Task #48 micro-slice 12: CheckExcMatch
                        // compare_fn factor refactor.  `compare_fn` is
                        // the same helper used by COMPARE_OP — the
                        // call shape `(exc:Ref, match_type:Ref, op_val:
                        // Int) → Ref` MayForce is identical to slice 4's
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
                        // Phase 4 walker-orthodoxy: compare_fn(exc,
                        // match_type, ISINSTANCE_OP:Int) → Ref shape
                        // residual_call_ir_r.  No frame_var threading.
                        let _ = record_residual_call_graph_op(
                            &mut graph,
                            &current_block.block(),
                            compare_fn_idx,
                            CallFlavor::MayForce,
                            vec![super::flow::Constant::signed(10).into()],
                            vec![exc_value, match_type_value],
                            vec![],
                            vec![Kind::Ref, Kind::Ref, Kind::Int],
                            ResKind::Ref,
                            py_pc as i64,
                        );
                        let result_value = fresh_ref_value(&mut graph);
                        current_state.stack.push(result_value.clone());
                        emit_pushvalue_ref!(current_depth, scratch_match, result_value);
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
                        let prev_reg = emit_popvalue_ref!(current_depth);
                        let prev_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        // set_current_exception is a TLS write — EF_CANNOT_RAISE.
                        // Task #48 micro-slice 15: PopExcept
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
                        // Phase 4 walker-orthodoxy: set_current_exception
                        // `(prev:Ref)→Void` shape residual_call_r_v.
                        // TLS write, no GC heap touched,
                        // `PlainCannotRaiseNoHeap` (`effectinfo.py:281-283`
                        // analyzer output).
                        let _ = record_residual_call_graph_op(
                            &mut graph,
                            &current_block.block(),
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

                    Instruction::Reraise { .. } => {
                        // Pops the exception. Net: -1.
                        // pypy/interpreter/pyopcode.py:1364, assemble.py:1608.
                        let _ = current_state.stack.pop();
                        current_depth = current_depth.saturating_sub(1);
                        emit_abort_permanent!();
                    }

                    Instruction::WithExceptStart => {
                        // CPython 3.14: `WITH_EXCEPT_START` leaves the existing
                        // stack entries intact and pushes the exit-function
                        // result on top. Preserve the net `+1` stack effect in
                        // the shadow graph and fall back to the interpreter for
                        // the actual helper call semantics.
                        emit_abort_permanent!();
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_vsd!(current_depth);
                    }

                    Instruction::Copy { i } => {
                        let d = i.get(op_arg) as usize;
                        if d == 1 {
                            let duplicated = duplicate_shadow_tos(&mut graph, &mut current_state);
                            emit_pushvalue_ref!(
                                current_depth,
                                stack_base + current_depth - 1,
                                duplicated
                            );
                        } else {
                            // COPY(d>1): duplicates stack[d] onto TOS. Net: +1.
                            // assemble.py:1482.
                            let stack_len = current_state.stack.len();
                            let duplicated = if d > 0 && d <= stack_len {
                                current_state.stack[stack_len - d].clone()
                            } else {
                                fresh_ref_value(&mut graph)
                            };
                            current_state.stack.push(duplicated);
                            current_depth += 1;
                            emit_abort_permanent!();
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
                        emit_abort_permanent!();
                        current_depth += 1;
                        emit_vsd!(current_depth);
                    }
                    Instruction::StoreName { .. } | Instruction::StoreGlobal { .. } => {
                        // flowcontext.py marks STORE_NAME unsupported, but the
                        // stack effect still consumes one value. STORE_GLOBAL
                        // follows the same shape in flowcontext.py:884-890.
                        emit_abort_permanent!();
                        let _ = current_state.stack.pop();
                        current_depth = current_depth.saturating_sub(1);
                        emit_vsd!(current_depth);
                    }
                    Instruction::MakeFunction { .. } => {
                        // Pops code object (TOS), pushes function. Net: 0.
                        // Replace shadow value so SET_FUNCTION_ATTRIBUTE sees func.
                        // RustPython: (1 pushed, 1 popped).
                        let _ = current_state.stack.pop();
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        emit_abort_permanent!();
                    }
                    Instruction::StoreAttr { namei } => {
                        let name_idx = namei.get(op_arg) as usize;
                        let attr_name =
                            super::flow::Constant::string(code.names[name_idx].as_str());
                        current_depth = current_depth.saturating_sub(1);
                        emit_vsd!(current_depth);
                        let obj_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        current_depth = current_depth.saturating_sub(1);
                        emit_vsd!(current_depth);
                        let stored_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
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
                        let obj_value = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
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
                            current_state.stack.push(null_stack_sentinel());
                            current_depth += 1;
                            emit_vsd!(current_depth);
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
                            let stored_reg = emit_popvalue_ref!(current_depth);
                            let stored = current_state
                                .stack
                                .pop()
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
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
                            if let Some(slot) = current_state.locals_w.get_mut(reg as usize) {
                                *slot = Some(stored);
                            }
                        }
                    }

                    // CPython 3.13 UNPACK_SEQUENCE: pop 1 (seq), push `count`.
                    // Emit abort_permanent (no getitem helper yet) but
                    // adjust current_depth so subsequent instructions don't
                    // underflow.
                    Instruction::UnpackSequence { count } => {
                        let n = count.get(op_arg) as usize;
                        // Pop iterable, push n unpacked items.
                        // pypy/interpreter/pyopcode.py:872.
                        let _ = current_state.stack.pop();
                        current_depth = current_depth.saturating_sub(1);
                        for _ in 0..n {
                            current_state.stack.push(fresh_ref_value(&mut graph));
                            current_depth += 1;
                        }
                        emit_abort_permanent!();
                    }

                    // CPython 3.13 iterator protocol — emit abort_permanent
                    // with correct depth tracking so subsequent instructions
                    // don't underflow.
                    Instruction::GetIter => {
                        // Pop iterable, push iterator. Net: 0. Replace shadow value.
                        // pypy/interpreter/pyopcode.py:1281.
                        let _ = current_state.stack.pop();
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        emit_abort_permanent!();
                    }

                    Instruction::ForIter { .. } => {
                        // push next item: net +1
                        emit_abort_permanent!();
                        current_depth += 1;
                        emit_vsd!(current_depth);
                    }

                    Instruction::EndFor => {
                        // Pyre's end_for() is a no-op (pyopcode.rs:999). Net: 0.
                        // The actual pop is handled by the subsequent PopIter (-1).
                        emit_abort_permanent!();
                    }

                    Instruction::PopIter => {
                        // pop iterator: net -1
                        current_depth = current_depth.saturating_sub(1);
                        emit_vsd!(current_depth);
                    }

                    // BinarySlice: obj[start:stop] — pops 3 (stop, start, obj), pushes 1 (result).
                    // Net stack effect: -2.
                    // pyopcode.py BINARY_SLICE / eval.rs:2857-2935.
                    Instruction::BinarySlice => {
                        for _ in 0..3 {
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // ContainsOp: item in container — pops 2, pushes 1 (bool).
                    // Net stack effect: -1.
                    // pyopcode.py CONTAINS_OP / eval.rs:1784-1798.
                    Instruction::ContainsOp { .. } => {
                        for _ in 0..2 {
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        current_state.stack.push(fresh_ref_value(&mut graph));
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
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        current_state.stack.push(fresh_ref_value(&mut graph));
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
                        let value = if idx < current_state.locals_w.len() {
                            current_state.locals_w[idx]
                                .clone()
                                .unwrap_or_else(|| fresh_ref_value(&mut graph))
                        } else {
                            fresh_ref_value(&mut graph)
                        };
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
                        let _ = current_state.stack.pop();
                        current_depth = current_depth.saturating_sub(1);
                        emit_abort_permanent!();
                    }

                    // BuildMap(count): pop 2*count key-value pairs, push dict. Net: -(2*count - 1).
                    // shared_opcode.rs opcode_build_map.
                    Instruction::BuildMap { count } => {
                        let n = count.get(op_arg) as usize;
                        for _ in 0..n * 2 {
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // MapAdd(i): peek dict at stack[i], pop value + key. Net: -2.
                    // eval.rs map_add.
                    Instruction::MapAdd { .. } => {
                        for _ in 0..2 {
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
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
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // BuildTuple(count): pops count items, pushes 1 tuple. Net: -(count-1).
                    Instruction::BuildTuple { count } => {
                        let n = count.get(op_arg) as usize;
                        for _ in 0..n {
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // BuildSet(count): pops count items, pushes 1 set. Net: -(count-1).
                    Instruction::BuildSet { count } => {
                        let n = count.get(op_arg) as usize;
                        for _ in 0..n {
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // BuildString(count): pops count strings, pushes 1. Net: -(count-1).
                    Instruction::BuildString { count } => {
                        let n = count.get(op_arg) as usize;
                        for _ in 0..n {
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // CallFunctionEx: pops callable+null+args+kwargs_or_null (4), pushes 1. Net: -3.
                    Instruction::CallFunctionEx => {
                        for _ in 0..4 {
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // DeleteSubscr: pops 2 (key, obj). Net: -2.
                    Instruction::DeleteSubscr => {
                        for _ in 0..2 {
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        emit_abort_permanent!();
                    }

                    // DeleteAttr: pops 1 (obj). Net: -1.
                    Instruction::DeleteAttr { .. } => {
                        let _ = current_state.stack.pop();
                        current_depth = current_depth.saturating_sub(1);
                        emit_abort_permanent!();
                    }

                    // PopJumpIfNone / PopJumpIfNotNone: pops 1. Net: -1.
                    Instruction::PopJumpIfNone { .. } | Instruction::PopJumpIfNotNone { .. } => {
                        let _ = current_state.stack.pop();
                        current_depth = current_depth.saturating_sub(1);
                        emit_abort_permanent!();
                    }

                    // SetAdd(i): peek set, pop value. Net: -1.
                    Instruction::SetAdd { .. } => {
                        let _ = current_state.stack.pop();
                        current_depth = current_depth.saturating_sub(1);
                        emit_abort_permanent!();
                    }

                    // ListExtend(i): peek list, pop iterable. Net: -1.
                    Instruction::ListExtend { .. } => {
                        let _ = current_state.stack.pop();
                        current_depth = current_depth.saturating_sub(1);
                        emit_abort_permanent!();
                    }

                    // SetUpdate(i): peek set, pop iterable. Net: -1.
                    Instruction::SetUpdate { .. } => {
                        let _ = current_state.stack.pop();
                        current_depth = current_depth.saturating_sub(1);
                        emit_abort_permanent!();
                    }

                    // DictUpdate(i) / DictMerge(i): peek dict, pop source. Net: -1.
                    Instruction::DictUpdate { .. } | Instruction::DictMerge { .. } => {
                        let _ = current_state.stack.pop();
                        current_depth = current_depth.saturating_sub(1);
                        emit_abort_permanent!();
                    }

                    // SetFunctionAttribute: pops func (TOS), pops attr (TOS1),
                    // pushes same func back. Net: -1. Preserve func identity.
                    // eval.rs:1907-1908: func = pop(), attr = pop().
                    Instruction::SetFunctionAttribute { .. } => {
                        let func = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
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
                        let result = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
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
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // ImportFrom: peek module, push attr. Net: +1.
                    Instruction::ImportFrom { .. } => {
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // StoreSlice: pops 4 (stop, start, obj, value). Net: -4.
                    Instruction::StoreSlice => {
                        for _ in 0..4 {
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        emit_abort_permanent!();
                    }

                    // FormatWithSpec: pops 2 (spec, value), pushes 1 string. Net: -1.
                    Instruction::FormatWithSpec => {
                        for _ in 0..2 {
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        current_state.stack.push(fresh_ref_value(&mut graph));
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
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        if is_method {
                            current_state.stack.push(fresh_ref_value(&mut graph));
                            current_depth += 1;
                        }
                        emit_abort_permanent!();
                    }

                    // UnpackEx: pops 1, pushes before+1+after items. Net: before+after.
                    Instruction::UnpackEx { counts } => {
                        let args = counts.get(op_arg);
                        let before = args.before as usize;
                        let after = args.after as usize;
                        let _ = current_state.stack.pop();
                        current_depth = current_depth.saturating_sub(1);
                        for _ in 0..before + 1 + after {
                            current_state.stack.push(fresh_ref_value(&mut graph));
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
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        for _ in 0..2 {
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // BuildTemplate: pops 2, pushes 1. Net: -1.
                    Instruction::BuildTemplate => {
                        for _ in 0..2 {
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // CallIntrinsic1: pops 1, pushes 1 (result may differ). Net: 0.
                    Instruction::CallIntrinsic1 { .. } => {
                        let _ = current_state.stack.pop();
                        current_state.stack.push(fresh_ref_value(&mut graph));
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
                                    let _ = current_state.stack.pop();
                                    current_depth = current_depth.saturating_sub(1);
                                }
                                current_state.stack.push(fresh_ref_value(&mut graph));
                                current_depth += 1;
                            }
                        }
                        emit_abort_permanent!();
                    }

                    // GetLen: peeks obj, pushes len. Net: +1.
                    Instruction::GetLen => {
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // LoadSpecial: pops 1 (obj), pushes 2 (callable, self_or_null). Net: +1.
                    // pyopcode.rs:2059 delegates to load_method; eval.rs:2365 pops 1 pushes 2.
                    Instruction::LoadSpecial { .. } => {
                        let _ = current_state.stack.pop();
                        current_depth = current_depth.saturating_sub(1);
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // LoadFromDictOrGlobals: pops 1 (dict), pushes 1 (result). Net: 0.
                    // Replace shadow value. eval.rs:2028.
                    Instruction::LoadFromDictOrGlobals { .. } => {
                        let _ = current_state.stack.pop();
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        emit_abort_permanent!();
                    }

                    // LoadFromDictOrDeref: structural adaptation — CPython pops dict,
                    // pushes result (net 0). Pyre's trait default raises before stack
                    // mutation (pyopcode.rs:1247), so this models the intended CPython
                    // shape, not current pyre runtime behavior.
                    Instruction::LoadFromDictOrDeref { .. } => {
                        let _ = current_state.stack.pop();
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        emit_abort_permanent!();
                    }

                    // Loads that push +1.
                    Instruction::LoadDeref { .. }
                    | Instruction::LoadFastCheck { .. }
                    | Instruction::LoadCommonConstant { .. }
                    | Instruction::LoadLocals
                    | Instruction::LoadBuildClass => {
                        current_state.stack.push(fresh_ref_value(&mut graph));
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
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        emit_abort_permanent!();
                    }

                    // Structural adaptation: async opcodes. Pyre's dispatcher
                    // errors immediately (pyopcode.rs:2027) without stack mutation.
                    // Stack effects model intended CPython shape for convergence.
                    Instruction::GetAiter | Instruction::GetAwaitable { .. } => {
                        let _ = current_state.stack.pop();
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        emit_abort_permanent!();
                    }

                    // StoreDeref: pops 1 value. Net: -1.
                    Instruction::StoreDeref { .. } => {
                        let _ = current_state.stack.pop();
                        current_depth = current_depth.saturating_sub(1);
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
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        emit_abort_permanent!();
                    }

                    // ReturnGenerator: pushes 1. Net: +1.
                    Instruction::ReturnGenerator => {
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // Send: pops sent value, peeks iter, pushes next result. Net: 0.
                    // Replace shadow value.
                    Instruction::Send { .. } => {
                        let _ = current_state.stack.pop();
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        emit_abort_permanent!();
                    }

                    // Structural adaptation: async opcodes below. Pyre's dispatcher
                    // errors immediately (pyopcode.rs:2027) without stack mutation.
                    // Stack effects model intended CPython shape for convergence.

                    // GetAnext: pushes 1. Net: +1.
                    Instruction::GetAnext => {
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // EndAsyncFor: pops 2. Net: -2.
                    // CPython 3.12/3.13 semantics; PyPy pops 3 (w_exc, w_prev, aiter)
                    // on the StopAsyncIteration path (assemble.py:1578). Structural
                    // adaptation: pyre targets CPython opcode shape here.
                    Instruction::EndAsyncFor => {
                        for _ in 0..2 {
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        emit_abort_permanent!();
                    }

                    // CleanupThrow: pops 3, pushes 1. Net: -2.
                    Instruction::CleanupThrow => {
                        for _ in 0..3 {
                            let _ = current_state.stack.pop();
                            current_depth = current_depth.saturating_sub(1);
                        }
                        current_state.stack.push(fresh_ref_value(&mut graph));
                        current_depth += 1;
                        emit_abort_permanent!();
                    }

                    // MatchSequence: peeks TOS (subject), pushes bool. Net: +1.
                    // assemble.py:1614, liveness.rs:601.
                    Instruction::MatchSequence => {
                        current_state.stack.push(fresh_ref_value(&mut graph));
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
                current_state.blocklist = frame_blocks_for_offset(code, current_state.next_offset);
                if let Some(catch_label) = catch_for_pc[py_pc] {
                    emit_catch_exception!(catch_label);
                }
            }
        } // end while-let pendingblocks

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

        for site in catch_sites {
            emit_mark_label_catch_landing!(site.landing_label);
            // `emit_mark_label_catch_landing!` (codewriter.rs:3318)
            // reassigns `current_block` to the pre-allocated catch
            // landing block on every iteration, so subsequent graph
            // emits in this loop body land in a block reachable from
            // `graph.iterblocks()`.  Lock the invariant in debug
            // builds — Session 17's exception unwind PY_NULL graph
            // dual-write (codewriter.rs:5481-5491) and any future
            // catch-landing dual-write rely on this targeting being
            // intact.
            debug_assert_eq!(
                current_block, catch_landing_blocks[&site.landing_label],
                "catch_landing block-targeting invariant violated: \
                 current_block != catch_landing_blocks[{}] after \
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
                current_state.stack.push(fresh_ref_value(&mut graph));
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
                // A-slice 6: box_int writes the lasti result directly to
                // the exception slot, retiring obj_tmp0 → exc_slot copy.
                // Task #48 micro-slice 10: box_int_fn factor refactor
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
                    let boxed_lasti = record_residual_call_graph_op(
                        &mut graph,
                        &current_block.block(),
                        box_int_fn_idx,
                        CallFlavor::Plain,
                        vec![super::flow::Constant::signed(site.lasti_py_pc as i64).into()],
                        vec![],
                        vec![],
                        vec![Kind::Int],
                        ResKind::Ref,
                        -1,
                    );
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
                emit_vsd!(depth);
                exc_slot += 1;
            }
            emit_last_exc_value!(exc_slot);
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
            // CATCH-LANDING dual-write follow-up (Task #227).
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
            // Push lasti graph dual-write — STILL DEFERRED:
            //   - `box_int(lasti_py_pc)` lowers to a `residual_call_*`
            //     shape that pyre's graph layer does not yet record
            //     (the `residual_call_*` family has zero graph
            //     coverage as of Session 17 — `flatten_arg` panics on
            //     `SpaceOperationArg::Descr`, and per-call-shape
            //     variant routing for `residual_call_ir_r` / `_r_r` /
            //     `_r_v` / `_r_i` is absent).  Adding the
            //     `setarrayitem_vable_r` dual-write alone would
            //     introduce an orphan def-use chain (def: nothing,
            //     use: setarrayitem) that breaks RPython's
            //     "every Variable has exactly one def" invariant.
            //
            // Block-targeting (was Session 17 blocker #1) is CLOSED:
            // `emit_mark_label_catch_landing!` (codewriter.rs:3318)
            // runs at the head of every iteration and reassigns
            // `current_block` to the pre-allocated catch landing
            // block.  The invariant is locked in via
            // `debug_assert_eq!` at the head of the loop body.
            //
            // The push_lasti dual-write joins when graph coverage for
            // `residual_call_*` (via `flatten_arg` Descr handling +
            // per-shape variant routing) lands.
            depth += 1;
            emit_vsd!(depth);
            emit_goto!(site.handler_py_pc);
        }

        // Drain per-block accumulators into ssarepr.insns in
        // walker-block-creation order.  Mirrors `codewriter.py:53
        // flatten_graph(graph, regallocs, cpu)` shape — block-by-block
        // emit, no PC interleaving.  Walker-creation order is canonical
        // because Phase A.4's first-wins `pc_anchor_positions` semantics
        // depend on the dead block's emit landing before its supersede
        // newblock's re-walked duplicate.
        //
        // Peel-off optimisation: at every block-switch boundary the
        // walker emits a defensive `goto Label("pcN") + Unreachable`
        // pair (the eventual drain order is not known at yield time
        // since `pendingblocks` is mixed push_front / push_back).  This
        // pass strips the pair when the next block actually opens with
        // `Label("pcN")` for the SAME N — turning a runtime no-op
        // branch into implicit fall-through.  Upstream `flatten.py:
        // 106-155 make_link` skips the goto outright via recursive
        // descent + `seen_blocks` (`flatten.py:110-113`); convergence
        // arrives when production flips to post-walk `flatten_graph`.
        {
            // RPython parity: `iterblocks` (`flowspace/model.py:55-77`)
            // enumerates every reachable block including dead ones;
            // the "no codegen" semantics ride on `block.operations`
            // being the empty tuple after `mergeblock` cleared it
            // (`flowcontext.py:455-457`).  Pyre mirrors that in
            // `SpamBlockRef::mark_dead` by clearing
            // `per_block_ssarepr` alongside the `dead` flag, so the
            // unfiltered drain below yields zero insns from dead
            // blocks while still preserving their position in walker-
            // creation order (relevant for the supersede newblock's
            // first-wins `pc_anchor_positions` lookup since the dead
            // block's now-empty entry no longer competes).
            let mut blocks: Vec<Vec<super::flatten::Insn>> = all_walker_blocks
                .iter()
                .map(|block| block.per_block_ssarepr())
                .collect();
            ssarepr.insns = strip_walker_block_boundary_goto(&mut blocks);
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

        // compute CFG-level link coalesce pairs
        // (`regalloc.py:79-96` projected onto pyre's u16 slot space)
        // and feed them into `allocate_registers` alongside the
        // existing SSARepr `*_copy` scanner.  Consumers (this call):
        //   - `collect_block_states(joinpoints, catch_landing_blocks)`
        //     → target ENTRY FrameStates per BlockRef.
        //   - `link_exit_states` — populated by the walker at every
        //     `append_exit_with_state` site (S4a).
        //   - `collect_link_slot_pairs(graph, block_entry_states,
        //      link_exit_states)` → `(src_slot, dst_slot)` pairs.
        //
        // In pyre's positional-aligned architecture the pairs are
        // always `(slot, slot)` with `slot == slot`, so
        // `try_coalesce` is a runtime no-op — but wiring the call
        // preserves the exact iteration shape of
        // `regalloc.py:79-96`.  Intra-block `*_copy` coalescing stays
        // in `RegAllocator::coalesce_variables` (orthogonal source).
        let block_entry_states = collect_block_states(&joinpoints, &catch_landing_blocks);
        let cfg_coalesce_pairs =
            collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        let alloc_result = super::regalloc::allocate_registers(
            &ssarepr,
            code.varnames.len(),
            inputs,
            &cfg_coalesce_pairs,
        );
        // Phase 1 pilot (Task #224): run graph-based
        // `perform_graph_register_allocation_all_kinds` in parallel
        // for instrumentation.  Upstream `codewriter.py:44-46` runs
        // regalloc on the CFG before flatten emits the SSARepr; pyre
        // still drives regalloc off the SSARepr, and the two allocators
        // see DIFFERENT Variable sets: graph regalloc sees the shadow
        // graph (semantic ops + tmp_claim shadows + frame-state-threaded

        super::regalloc::apply_rename(&mut ssarepr, &alloc_result.rename);

        // `flatten.py:88-100` `enforce_input_args` may rotate the
        // portal `(frame, ec)` inputargs into new colors. Keep the
        // pyre-side metadata aligned with the post-regalloc SSA/JitCode
        // slots the assembler will actually emit; the blackhole fill
        // path must write the colored portal registers, not the
        // pre-color layout placeholders.
        let portal_frame_reg = alloc_result
            .rename
            .get(&(Kind::Ref, portal_frame_reg))
            .copied()
            .unwrap_or(portal_frame_reg);
        let portal_ec_reg = alloc_result
            .rename
            .get(&(Kind::Ref, portal_ec_reg))
            .copied()
            .unwrap_or(portal_ec_reg);

        // Phase 2 commit 2.1 (Tasks #158/#159/#122 epic, plan
        // `~/.claude/plans/staged-sauteeing-koala.md`): record each
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
            let post = alloc_result
                .rename
                .get(&(Kind::Ref, pre))
                .copied()
                .unwrap_or(pre);
            stack_slot_color_map.push(post);
        }
        // SSA-authoritative live_r slice 3a: record each Python-semantic
        // local slot's post-regalloc color.  The encoder
        // (`get_list_of_active_boxes`) derives `semantic_idx` from
        // `color_idx < nlocals → identity` after the slice 3b-2 flip.
        //
        // Today `enforce_input_args` (regalloc.rs:524-563, flatten.py:88
        // -100 parity) pins each local-i inputarg color to identity
        // (`color = i`), so this map is `[0, 1, ..., nlocals-1]` for
        // every populated jitcode.  When `enforce_input_args` pinning
        // is relaxed (Task #158), the encoder will read this map to
        // derive the semantic local index from a non-identity color.
        let mut pyre_color_for_semantic_local: Vec<u16> = Vec::with_capacity(nlocals);
        for i in 0..nlocals as u16 {
            let post = alloc_result
                .rename
                .get(&(Kind::Ref, i))
                .copied()
                .unwrap_or(i);
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
        filter_liveness_in_place(
            &mut ssarepr,
            code,
            &depth_at_pc,
            &pyre_color_for_semantic_local,
            &stack_slot_color_map,
            portal_frame_reg,
            portal_ec_reg,
        );
        // Runtime entry/liveness lookups expect the byte offset of the
        // surviving `-live-` marker for each Python PC
        // (`jitcode.get_live_vars_info` first checks `code[pc] ==
        // op_live`). `remove_repeated_live` may move that marker away
        // from the per-PC `Label("pcN")` anchor, so record the final
        // per-PC live-marker positions here instead of the label indices.
        let pc_map = live_marker_indices_by_pc(&ssarepr, num_instrs);

        // codewriter.py:62-67 num_regs[kind] = max(coloring)+1
        // (or 0 if coloring is empty). Pass through to the Assembler
        // step so `JitCode.num_regs_*` reflect the post-regalloc
        // ceiling rather than the pre-regalloc PyFrame-slot range.
        let num_regs = super::assembler::NumRegs {
            int: alloc_result
                .num_regs
                .get(&super::flatten::Kind::Int)
                .copied()
                .unwrap_or(0),
            ref_: alloc_result
                .num_regs
                .get(&super::flatten::Kind::Ref)
                .copied()
                .unwrap_or(0),
            float: alloc_result
                .num_regs
                .get(&super::flatten::Kind::Float)
                .copied()
                .unwrap_or(0),
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
pub fn find_loop_header_pcs(
    code: &pyre_interpreter::CodeObject,
) -> std::collections::HashSet<usize> {
    let num_instrs = code.instructions.len();
    let mut loop_header_pcs: std::collections::HashSet<usize> = std::collections::HashSet::new();
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
pub fn find_branch_target_pcs(
    code: &pyre_interpreter::CodeObject,
) -> std::collections::HashSet<usize> {
    let num_instrs = code.instructions.len();
    let mut targets: std::collections::HashSet<usize> = std::collections::HashSet::new();
    targets.insert(0);
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
        FrameState, SpamBlockRef, attach_catch_exception_edge, collect_block_states,
        collect_link_slot_pairs, entry_arg_slots, entry_frame_state, entry_inputargs, mergeblock,
        new_shadow_graph,
    };
    use crate::jit::assembler::ArcByPtr;
    use crate::jit::flatten::{Insn, Kind, Label as FlatLabel, Operand, Register, SSARepr};
    use crate::jit::flow::{
        Block, Constant, ExitSwitch, FlowValue, FunctionGraph, Link, LinkArgPosition, LinkRef,
        SpaceOperationArg, Variable, VariableId, c_last_exception,
    };
    use pyre_interpreter::bytecode::{CodeObject, ConstantData};
    use pyre_interpreter::compile_exec;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// R4 regression test: tail-strip pass must recognise the
    /// `Insn::PcAnchor { py_pc }` shape the walker emits at PC block
    /// boundaries (commit 0b016d9bde retired the transitional
    /// `Insn::Label("pcN")` shape, but the strip recogniser still
    /// only matched `Insn::Label`, so the defensive
    /// `goto pcN; ---` pair was left in the drained stream — extra
    /// control-flow RPython's recursive `make_link` would never
    /// emit per `flatten.py:106-155`).
    #[test]
    fn strip_walker_block_boundary_goto_strips_against_pc_anchor() {
        use super::super::flatten::{Insn, Operand, pc_label_name, pc_tlabel};

        let goto = Insn::op("goto", vec![Operand::TLabel(pc_tlabel(7))]);
        let trailing_unreachable = Insn::Unreachable;
        let block_a = vec![
            Insn::live(Vec::new()),
            goto.clone(),
            trailing_unreachable.clone(),
        ];
        let block_b = vec![Insn::pc_anchor(7), Insn::live(Vec::new())];
        let mut blocks = vec![block_a, block_b];

        let drained = super::strip_walker_block_boundary_goto(&mut blocks);

        // Block A's goto + Unreachable were stripped; block B's
        // PcAnchor opens the fall-through directly.
        assert_eq!(
            drained.len(),
            3,
            "expected [live, PcAnchor, live] after strip, got {drained:?}",
        );
        assert!(matches!(drained[0], Insn::Op { ref opname, .. } if opname == "-live-"));
        assert!(matches!(drained[1], Insn::PcAnchor { py_pc: 7 },));
        assert!(matches!(drained[2], Insn::Op { ref opname, .. } if opname == "-live-"));

        // The strip should NOT fire when the goto target doesn't
        // match the next block's anchor.
        let goto_to_99 = Insn::op("goto", vec![Operand::TLabel(pc_tlabel(99))]);
        let block_a = vec![Insn::live(Vec::new()), goto_to_99, Insn::Unreachable];
        let block_b = vec![Insn::pc_anchor(7)];
        let mut blocks = vec![block_a, block_b];
        let drained = super::strip_walker_block_boundary_goto(&mut blocks);
        assert_eq!(
            drained.len(),
            4,
            "goto/--- must remain when target != next block's PcAnchor",
        );

        // Sanity for the upstream-orthodox `Insn::Label` shape — strip
        // still works (R4 doesn't regress the Label case).
        use super::super::flatten::Label as FlatLabel;
        let label_name = pc_label_name(11);
        let goto = Insn::op("goto", vec![Operand::TLabel(pc_tlabel(11))]);
        let block_a = vec![Insn::live(Vec::new()), goto, Insn::Unreachable];
        let block_b = vec![
            Insn::Label(FlatLabel::new(label_name)),
            Insn::live(Vec::new()),
        ];
        let mut blocks = vec![block_a, block_b];
        let drained = super::strip_walker_block_boundary_goto(&mut blocks);
        assert_eq!(drained.len(), 3, "Insn::Label strip path must also fire");
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
    fn null_stack_sentinel_is_opaque_ref_constant() {
        let value = null_stack_sentinel();
        let FlowValue::Constant(constant) = value else {
            panic!("null stack sentinel must be a Constant");
        };
        assert_eq!(constant.kind, Some(Kind::Ref));
        assert!(matches!(
            constant.value,
            crate::jit::flow::ConstantValue::Opaque(_)
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
    fn frontend_constant_flow_value_preserves_string_constants() {
        let constant = ConstantData::Str {
            value: "hello".to_owned().into(),
        };

        let value = frontend_constant_flow_value(&constant);

        assert_eq!(value, Some(Constant::string("hello").into()));
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

    fn identity_arg_positions(count: usize) -> Vec<LinkArgPosition> {
        (0..count)
            .map(|index| LinkArgPosition {
                source_mergeable_index: Some(index),
                target_mergeable_index: Some(index),
            })
            .collect()
    }

    /// Step 6A slice S1 regression: `FrameState::mergeable_index_of` locates
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

    /// Step 6A slice S2 regression: `FrameState::mergeable_index_to_slot`
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

    /// Step 6A slice S2 regression: `variable_slot` composes S1 + S2 so
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

    /// Helper: build a `link_exit_states` map from `(LinkRef,
    /// FrameState)` pairs.  Production walker will populate this by
    /// cloning `currentstate` at each `append_exit` call
    /// (`flowcontext.py:1237,1268-1280`).
    fn link_exit_states_from(pairs: Vec<(LinkRef, FrameState)>) -> HashMap<LinkRef, FrameState> {
        let mut map = HashMap::new();
        for (link, state) in pairs {
            map.insert(link, state);
        }
        map
    }

    /// Step 6A slice S3 regression: `collect_link_slot_pairs` emits a
    /// trivially-equal slot pair at every mergeable position where
    /// both source (EXIT state) and target (ENTRY state) hold a
    /// Variable.  The pairs are positional by `getoutputargs`
    /// construction (`codewriter.rs:333-346`); see S3c docstring.
    #[test]
    fn collect_link_slot_pairs_emits_positional_pairs_for_variable_links() {
        let start_arg = Variable::new(VariableId(0), Kind::Ref);
        let start_arg2 = Variable::new(VariableId(1), Kind::Ref);
        let mid_arg = Variable::new(VariableId(2), Kind::Ref);
        let mid_arg2 = Variable::new(VariableId(3), Kind::Ref);
        let mut graph = FunctionGraph::new(
            "coalesce",
            Block::shared(vec![start_arg.into(), start_arg2.into()]),
            None,
        );
        let mid = graph.new_block(vec![mid_arg.into(), mid_arg2.into()]);
        let link = Link::new(
            vec![start_arg.into(), start_arg2.into()],
            Some(mid.clone()),
            None,
        )
        .with_arg_positions(identity_arg_positions(2))
        .into_ref();
        graph.startblock.closeblock(vec![link.clone()]);

        let start_state = FrameState::new(
            vec![Some(start_arg.into()), Some(start_arg2.into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let mid_state = FrameState::new(
            vec![Some(mid_arg.into()), Some(mid_arg2.into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let mut block_entry_states = HashMap::new();
        block_entry_states.insert(graph.startblock.clone(), start_state.clone());
        block_entry_states.insert(mid.clone(), mid_state);
        let link_exit_states = link_exit_states_from(vec![(link, start_state)]);

        let pairs = collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        assert_eq!(pairs, vec![(0, 0), (1, 1)]);
    }

    /// Step 6A slice S3 regression: Constant link args do not
    /// contribute a pair (source mergeable at that position is a
    /// Constant, not a Variable).  Mirrors `flatten.py:355-363`
    /// `flatten_list` + `regalloc.py:99-101` `if isinstance(v,
    /// Variable)` — Constants pass through unchanged.
    #[test]
    fn collect_link_slot_pairs_skips_constant_link_args() {
        let start_arg = Variable::new(VariableId(0), Kind::Ref);
        let next_arg = Variable::new(VariableId(1), Kind::Ref);
        let mut graph =
            FunctionGraph::new("with_const", Block::shared(vec![start_arg.into()]), None);
        let next = graph.new_block(vec![next_arg.into()]);
        let link = Link::new(vec![Constant::signed(42).into()], Some(next.clone()), None)
            .with_arg_positions(identity_arg_positions(1))
            .into_ref();
        graph.startblock.closeblock(vec![link.clone()]);

        // Source EXIT state has a Constant at position 0 (matching
        // the Constant-carrying link arg) — e.g. a parameter with a
        // default Constant flowing through a branch.  Target ENTRY
        // state still has a Variable.
        let start_exit = FrameState::new(
            vec![Some(Constant::signed(42).into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let next_state =
            FrameState::new(vec![Some(next_arg.into())], Vec::new(), None, Vec::new(), 0);
        let mut block_entry_states = HashMap::new();
        block_entry_states.insert(
            graph.startblock.clone(),
            FrameState::new(
                vec![Some(start_arg.into())],
                Vec::new(),
                None,
                Vec::new(),
                0,
            ),
        );
        block_entry_states.insert(next.clone(), next_state);
        let link_exit_states = link_exit_states_from(vec![(link, start_exit)]);

        let pairs = collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        assert!(
            pairs.is_empty(),
            "constant link args contribute no coalesce pairs"
        );
    }

    /// Step 6A slice S3 regression: a Link whose target has no
    /// attached FrameState (catch landings, returnblock, exceptblock)
    /// contributes no pairs.  Covers the
    /// `block_entry_states.get(&target)` early-exit branch.
    #[test]
    fn collect_link_slot_pairs_skips_missing_target_framestate() {
        let start_arg = Variable::new(VariableId(0), Kind::Ref);
        let next_arg = Variable::new(VariableId(1), Kind::Ref);
        let mut graph = FunctionGraph::new(
            "missing_target",
            Block::shared(vec![start_arg.into()]),
            None,
        );
        let next = graph.new_block(vec![next_arg.into()]);
        let link = Link::new(vec![start_arg.into()], Some(next.clone()), None)
            .with_arg_positions(identity_arg_positions(1))
            .into_ref();
        graph.startblock.closeblock(vec![link.clone()]);

        let start_state = FrameState::new(
            vec![Some(start_arg.into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let mut block_entry_states = HashMap::new();
        block_entry_states.insert(graph.startblock.clone(), start_state.clone());
        // Deliberately do NOT insert `next` — mimics catch landing block.
        let link_exit_states = link_exit_states_from(vec![(link, start_state)]);

        let pairs = collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        assert!(pairs.is_empty());
    }

    /// Step 6A slice S3c regression: a Link whose source EXIT state
    /// replaced the ENTRY-state Variable with a freshly-allocated
    /// mid-block Variable still emits the correct slot pair.
    /// Previously the helper consulted only the source block's ENTRY
    /// state and missed the fresh Variable.  S3c supplies the source
    /// state via `link_exit_states`; the positional walk ignores
    /// identity and looks only at whether each mergeable position is
    /// a Variable on both sides.
    ///
    /// Scenario:
    ///  - Source ENTRY locals_w = [v_entry] at mergeable position 0.
    ///  - Walker STORE_FAST overwrites locals_w[0] with v_exit; at
    ///    terminator time currentstate.locals_w[0] == v_exit.
    ///  - Link.args = [v_exit].  Target ENTRY locals_w = [v_target].
    ///
    /// Expected: one (0, 0) coalesce pair via link_exit_states[link].
    /// See Task #222.
    #[test]
    fn collect_link_slot_pairs_finds_variable_via_link_exit_state() {
        let v_entry = Variable::new(VariableId(0), Kind::Ref);
        let v_exit = Variable::new(VariableId(1), Kind::Ref);
        let v_target = Variable::new(VariableId(2), Kind::Ref);
        let mut graph = FunctionGraph::new("exit_state", Block::shared(vec![v_entry.into()]), None);
        let target = graph.new_block(vec![v_target.into()]);
        let link = Link::new(vec![v_exit.into()], Some(target.clone()), None)
            .with_arg_positions(identity_arg_positions(1))
            .into_ref();
        graph.startblock.closeblock(vec![link.clone()]);

        let start_entry =
            FrameState::new(vec![Some(v_entry.into())], Vec::new(), None, Vec::new(), 0);
        let start_exit =
            FrameState::new(vec![Some(v_exit.into())], Vec::new(), None, Vec::new(), 0);
        let target_entry =
            FrameState::new(vec![Some(v_target.into())], Vec::new(), None, Vec::new(), 0);

        let mut block_entry_states = HashMap::new();
        block_entry_states.insert(graph.startblock.clone(), start_entry);
        block_entry_states.insert(target.clone(), target_entry);
        let link_exit_states = link_exit_states_from(vec![(link, start_exit)]);

        let pairs = collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        assert_eq!(
            pairs,
            vec![(0, 0)],
            "EXIT-state Variable must not prevent pair emission",
        );
    }

    /// Step 6A slice S3c regression: a Link with no
    /// `link_exit_states` entry contributes no pairs.  Production
    /// walker MUST populate the EXIT snapshot for every link it
    /// emits; a missing entry (un-wired path or test that skipped it)
    /// skips rather than panicking to keep the helper robust during
    /// staged integration.
    #[test]
    fn collect_link_slot_pairs_skips_links_without_exit_state() {
        let start_arg = Variable::new(VariableId(0), Kind::Ref);
        let next_arg = Variable::new(VariableId(1), Kind::Ref);
        let mut graph = FunctionGraph::new(
            "missing_exit_state",
            Block::shared(vec![start_arg.into()]),
            None,
        );
        let next = graph.new_block(vec![next_arg.into()]);
        let link = Link::new(vec![start_arg.into()], Some(next.clone()), None)
            .with_arg_positions(identity_arg_positions(1))
            .into_ref();
        graph.startblock.closeblock(vec![link]);

        let start_state = FrameState::new(
            vec![Some(start_arg.into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let next_state =
            FrameState::new(vec![Some(next_arg.into())], Vec::new(), None, Vec::new(), 0);
        let mut block_entry_states = HashMap::new();
        block_entry_states.insert(graph.startblock.clone(), start_state);
        block_entry_states.insert(next.clone(), next_state);
        // Deliberately empty: no source EXIT state available.
        let link_exit_states = HashMap::new();

        let pairs = collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        assert!(pairs.is_empty());
    }

    /// Step 6A slice S3c regression: `LoadFast`-style aliasing where
    /// the same Variable lives at two mergeable positions
    /// simultaneously (`codewriter.rs:2413-2414` pushes the local's
    /// own Variable onto the stack).  The positional walk must emit
    /// one pair per mergeable position, not one pair per Variable,
    /// so both (0, 0) for the local slot and (1, 1) for the stack
    /// slot fire.  Proves the helper is not vulnerable to
    /// Variable-collision ambiguity.
    #[test]
    fn collect_link_slot_pairs_handles_variable_aliased_across_slots() {
        let v_local = Variable::new(VariableId(0), Kind::Ref);
        let v_next_local = Variable::new(VariableId(1), Kind::Ref);
        let v_next_stack = Variable::new(VariableId(2), Kind::Ref);
        let mut graph = FunctionGraph::new("aliased", Block::shared(vec![v_local.into()]), None);
        // target inputargs == mergeable Variables in locals_w + stack
        let next = graph.new_block(vec![v_next_local.into(), v_next_stack.into()]);
        // Link carries v_local twice — once for locals_w[0], once for stack[0].
        let link = Link::new(
            vec![v_local.into(), v_local.into()],
            Some(next.clone()),
            None,
        )
        .with_arg_positions(identity_arg_positions(2))
        .into_ref();
        graph.startblock.closeblock(vec![link.clone()]);

        // Source EXIT state: locals_w[0] AND stack[0] both hold v_local.
        let start_exit = FrameState::new(
            vec![Some(v_local.into())],
            vec![v_local.into()],
            None,
            Vec::new(),
            0,
        );
        let next_entry = FrameState::new(
            vec![Some(v_next_local.into())],
            vec![v_next_stack.into()],
            None,
            Vec::new(),
            0,
        );
        let mut block_entry_states = HashMap::new();
        block_entry_states.insert(
            graph.startblock.clone(),
            FrameState::new(
                vec![Some(v_local.into())],
                vec![v_local.into()],
                None,
                Vec::new(),
                0,
            ),
        );
        block_entry_states.insert(next.clone(), next_entry);
        let link_exit_states = link_exit_states_from(vec![(link, start_exit)]);

        let pairs = collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        assert_eq!(
            pairs,
            vec![(0, 0), (1, 1)],
            "positional walk must emit one pair per mergeable slot, not per Variable",
        );
    }

    /// Step 6A slice S3b regression: `collect_block_states` absorbs
    /// the walker's SpamBlockRef containers, skipping entries whose
    /// FrameState is `None` (catch landings), deduplicating blocks
    /// that appear in multiple containers.
    #[test]
    fn collect_block_states_walks_all_walker_containers() {
        let mut graph = FunctionGraph::new("s3b", Block::shared(Vec::new()), None);
        let block_a = graph.new_block(Vec::new());
        let block_b = graph.new_block(Vec::new());
        let block_landing = graph.new_block(Vec::new());

        let state_a = FrameState::new(
            vec![Some(Variable::new(VariableId(0), Kind::Ref).into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let state_b = FrameState::new(
            vec![Some(Variable::new(VariableId(1), Kind::Ref).into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );

        let a_ref = SpamBlockRef::new(block_a.clone(), Some(state_a.clone()));
        let b_ref = SpamBlockRef::new(block_b.clone(), Some(state_b.clone()));
        let landing_ref = SpamBlockRef::new(block_landing.clone(), None);

        let mut joinpoints: HashMap<usize, Vec<SpamBlockRef>> = HashMap::new();
        joinpoints.insert(0, vec![a_ref.clone()]);
        joinpoints.insert(2, vec![b_ref.clone()]);
        let mut catch_landing_blocks: HashMap<u16, SpamBlockRef> = HashMap::new();
        // Catch landings have framestate = None and MUST be skipped.
        catch_landing_blocks.insert(7, landing_ref);

        let map = collect_block_states(&joinpoints, &catch_landing_blocks);

        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&block_a), Some(&state_a));
        assert_eq!(map.get(&block_b), Some(&state_b));
        assert!(
            !map.contains_key(&block_landing),
            "catch-landing block with None framestate must not appear in the map"
        );
    }

    /// Step 6A slice S3b + S3 end-to-end: when the
    /// `block_entry_states` map is built from the walker helpers
    /// (`collect_block_states`), `collect_link_slot_pairs` still
    /// yields the same positional pair as the hand-built variant.
    /// S3c revision: caller also supplies a `link_exit_states` map —
    /// here populated with the source block's ENTRY state because the
    /// fabricated graph has no mid-block ops.
    #[test]
    fn collect_block_states_feeds_collect_link_slot_pairs() {
        let start_arg = Variable::new(VariableId(0), Kind::Ref);
        let next_arg = Variable::new(VariableId(1), Kind::Ref);
        let mut graph = FunctionGraph::new("s3b_e2e", Block::shared(vec![start_arg.into()]), None);
        let next = graph.new_block(vec![next_arg.into()]);
        let link = Link::new(vec![start_arg.into()], Some(next.clone()), None)
            .with_arg_positions(identity_arg_positions(1))
            .into_ref();
        graph.startblock.closeblock(vec![link.clone()]);

        let start_state = FrameState::new(
            vec![Some(start_arg.into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let next_state =
            FrameState::new(vec![Some(next_arg.into())], Vec::new(), None, Vec::new(), 0);

        let mut joinpoints: HashMap<usize, Vec<SpamBlockRef>> = HashMap::new();
        joinpoints.insert(
            0,
            vec![SpamBlockRef::new(
                graph.startblock.clone(),
                Some(start_state.clone()),
            )],
        );
        joinpoints.insert(
            1,
            vec![SpamBlockRef::new(next.clone(), Some(next_state.clone()))],
        );
        let catch_landing_blocks = HashMap::new();

        let block_entry_states = collect_block_states(&joinpoints, &catch_landing_blocks);
        let link_exit_states = link_exit_states_from(vec![(link, start_state)]);
        let pairs = collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        assert_eq!(pairs, vec![(0, 0)]);
    }

    #[test]
    fn pc_anchor_and_live_marker_rescan_follow_final_ssarepr_order() {
        let mut ssarepr = SSARepr::new("t");
        ssarepr.insns.push(Insn::pc_anchor(0));
        ssarepr
            .insns
            .push(Insn::live(vec![Operand::Register(Register::new(
                Kind::Ref,
                0,
            ))]));
        // PcAnchor is a merge boundary in `remove_repeated_live`, so
        // each PC keeps its own `-live-` marker without cross-PC
        // merge-then-reorder.  The anchor scan resolves each PC anchor
        // at its PcAnchor position; the live marker for each PC stays
        // at its pre-merge position.
        ssarepr.insns.push(Insn::pc_anchor(1));
        ssarepr
            .insns
            .push(Insn::live(vec![Operand::Register(Register::new(
                Kind::Ref,
                1,
            ))]));

        crate::jit::liveness::remove_repeated_live(&mut ssarepr);

        // Anchor positions: PcAnchor(0)@0, PcAnchor(1)@2.
        assert_eq!(pc_anchor_positions(&ssarepr, 2), vec![0, 2]);
        // Per-PC `-live-` boundaries: live(R0) at index 1,
        // live(R1) at index 3.
        assert_eq!(live_marker_indices_by_pc(&ssarepr, 2), vec![1, 3]);
    }

    #[test]
    fn filter_liveness_drops_non_lv_live_colors_from_live_r() {
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let live_vars = pyre_jit_trace::state::liveness_for(&code as *const _);
        let reachable_pc = (0..code.instructions.len())
            .find(|&py_pc| live_vars.is_reachable(py_pc))
            .expect("compiled code must have a reachable pc");

        let mut ssarepr = SSARepr::new("t");
        for py_pc in 0..code.instructions.len() {
            ssarepr.insns.push(Insn::pc_anchor(py_pc));
            ssarepr.insns.push(Insn::live(vec![
                Operand::Register(Register::new(Kind::Ref, 0)),
                Operand::Register(Register::new(Kind::Ref, 7)),
                Operand::Register(Register::new(Kind::Int, 3)),
            ]));
        }

        let depth_at_pc: Vec<u16> = vec![0; code.instructions.len()];
        let local_color_map: Vec<u16> = (0..code.varnames.len() as u16).collect();
        let stack_slot_color_map: Vec<u16> = Vec::new();
        filter_liveness_in_place(
            &mut ssarepr,
            &code,
            &depth_at_pc,
            &local_color_map,
            &stack_slot_color_map,
            u16::MAX,
            u16::MAX,
        );

        let live_idx = live_marker_indices_by_pc(&ssarepr, code.instructions.len())[reachable_pc];
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

        let inputargs = graph_entry_inputargs(&code, true);
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

        let graph = new_shadow_graph_with_portal_inputs(&code, true);
        let startblock = graph.startblock.borrow();
        let returnblock = graph.returnblock.borrow();

        assert_eq!(startblock.inputargs, graph_entry_inputargs(&code, true));
        match &returnblock.inputargs[0] {
            FlowValue::Variable(variable) => {
                assert_eq!(
                    variable.id,
                    VariableId(graph_entry_inputargs(&code, true).len() as u32)
                );
                assert_eq!(variable.kind, Some(Kind::Ref));
            }
            other => panic!("expected variable return arg, got {other:?}"),
        }
    }

    #[test]
    fn portal_jit_merge_point_graph_args_match_upstream_shape() {
        let code = first_nested_function_code(
            "def f(a):\n    while a:\n        a = a - 1\n    return a\n",
        );
        let w_code = pyre_interpreter::box_code_constant(&code);
        let graph = new_shadow_graph_with_portal_inputs(&code, true);
        let args = portal_jit_merge_point_graph_args(&graph, 17, w_code as *const (), 7);

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
                    FlowValue::Constant(constant) => {
                        assert_eq!(constant.kind, Some(Kind::Ref));
                    }
                    other => panic!("expected pycode ref constant, got {other:?}"),
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
        let state = entry_frame_state(&code, false);

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
        assert_eq!(
            state1.getoutputargs_with_positions(&state2).1,
            identity_arg_positions(2),
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
        let mut link_exit_states: HashMap<LinkRef, FrameState> = HashMap::new();
        let source_state = FrameState::new(Vec::new(), Vec::new(), None, Vec::new(), 0);
        let startblock_ref = graph.startblock.clone();

        let link = attach_catch_exception_edge(
            &mut graph,
            &startblock_ref,
            &catch_ref,
            &source_state,
            &mut link_exit_states,
        );
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
        let mut link_exit_states: HashMap<LinkRef, FrameState> = HashMap::new();
        let source_state = FrameState::new(
            vec![Some(Variable::new(VariableId(0), Kind::Ref).into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let startblock_ref = graph.startblock.clone();

        let link = attach_catch_exception_edge(
            &mut graph,
            &startblock_ref,
            &catch_ref,
            &source_state,
            &mut link_exit_states,
        );

        let link_borrow = link.borrow();
        assert!(link_borrow.last_exception.is_some());
        assert!(link_borrow.last_exc_value.is_some());
        drop(link_borrow);

        let catch_state = catch_ref
            .framestate()
            .expect("catch landing should acquire a FrameState");
        assert!(catch_state.last_exception.is_some());
        assert_eq!(link_exit_states.get(&link), Some(&source_state));
    }

    #[test]
    fn attach_catch_exception_edge_populates_target_inputargs() {
        let code = first_nested_function_code("def f(a):\n    return a\n");
        let mut graph = new_shadow_graph(&code);
        let catch_block = graph.new_block(Vec::new());
        let catch_ref = SpamBlockRef::new(catch_block.clone(), None);
        let mut link_exit_states: HashMap<LinkRef, FrameState> = HashMap::new();
        let local = Variable::new(VariableId(0), Kind::Ref);
        let source_state =
            FrameState::new(vec![Some(local.into())], Vec::new(), None, Vec::new(), 0);
        let startblock_ref = graph.startblock.clone();

        assert!(
            catch_block.borrow().inputargs.is_empty(),
            "catch landing block starts with no inputargs"
        );

        attach_catch_exception_edge(
            &mut graph,
            &startblock_ref,
            &catch_ref,
            &source_state,
            &mut link_exit_states,
        );

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
        let mut joinpoints: HashMap<usize, Vec<SpamBlockRef>> = HashMap::new();
        joinpoints.insert(1, vec![target_block.clone()]);
        let mut link_exit_states: HashMap<LinkRef, FrameState> = HashMap::new();
        let mut pendingblocks: VecDeque<SpamBlockRef> = VecDeque::new();
        let mut all_walker_blocks: Vec<SpamBlockRef> = Vec::new();

        let merged = mergeblock(
            &code,
            &mut graph,
            &mut joinpoints,
            &current_block,
            &current_state,
            1,
            &mut link_exit_states,
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
        let mut joinpoints: HashMap<usize, Vec<SpamBlockRef>> = HashMap::new();
        joinpoints.insert(2, vec![existing_block.clone()]);
        let mut link_exit_states: HashMap<LinkRef, FrameState> = HashMap::new();
        let mut pendingblocks: VecDeque<SpamBlockRef> = VecDeque::new();
        let mut all_walker_blocks: Vec<SpamBlockRef> = Vec::new();

        let merged = mergeblock(
            &code,
            &mut graph,
            &mut joinpoints,
            &current_block,
            &source_state,
            2,
            &mut link_exit_states,
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
}
