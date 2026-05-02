//! Flatten pass: CFG → linear instruction sequence.
//!
//! RPython equivalent: `jit/codewriter/flatten.py` flatten_graph().
//!
//! Converts a multi-block FunctionGraph into a linear sequence of
//! FlatOps with Labels and Jumps. This is the last graph pass
//! before register allocation and JitCode assembly.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::flowspace::model::ConstValue;
use crate::model::{
    BlockId, ExitCase, ExitSwitch, FunctionGraph, Link, LinkArg, SpaceOperation, ValueId,
};
use crate::regalloc::RegAllocResult;

/// A label in the flattened instruction stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Label(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntOvfOp {
    Add,
    Sub,
    Mul,
}

/// A flattened instruction (post-CFG).
///
/// RPython equivalent: SSARepr instruction tuples from flatten.py.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlatOp {
    /// Label definition (target for jumps).
    Label(Label),
    /// Semantic op (from the graph).
    Op(SpaceOperation),
    /// Unconditional jump to label.
    /// RPython: `('goto', TLabel(target))`.
    Jump(Label),
    /// Conditional jump: if cond is false (zero), jump to label.
    /// RPython: `('goto_if_not', cond, TLabel(false_path))`.
    /// There is NO goto_if_true — RPython only uses goto_if_not.
    /// The true path is always the fallthrough.
    GotoIfNot { cond: ValueId, target: Label },
    /// RPython `flatten.py:278-308` integer switch:
    /// `('switch', value, SwitchDictDescr)` after a preceding `-live-`.
    /// The default path is the fall-through after the switch op; each
    /// `(key, label)` entry jumps to the corresponding case landing pad.
    Switch {
        value: ValueId,
        targets: Vec<(i64, Label)>,
    },
    /// RPython `flatten.py:190-197`
    /// `int_add_jump_if_ovf` / `int_sub_jump_if_ovf` / `int_mul_jump_if_ovf`.
    IntBinOpJumpIfOvf {
        op: IntOvfOp,
        target: Label,
        lhs: ValueId,
        rhs: ValueId,
        dst: ValueId,
    },
    /// Exception setup for a can-raise block.
    /// RPython: `('catch_exception', TLabel(normal_link))`.
    CatchException { target: Label },
    /// RPython `flatten.py:228-231`
    /// `('goto_if_exception_mismatch', Constant(link.llexitcase, lltype.typeOf(link.llexitcase)), TLabel(link))`.
    /// The link-side `llexitcase` is preserved as the full RPython-style
    /// Constant and encoded by the assembler according to backend needs.
    GotoIfExceptionMismatch {
        llexitcase: ConstValue,
        target: Label,
    },
    /// Copy value (for Phi-node resolution: Link.args → target.inputargs).
    ///
    /// RPython `flatten.py:333` `self.emitline('%s_copy' % kind, v, "->", w)`.
    /// Upstream `getcolor(v)` returns `v` as-is for `Constant`
    /// (flatten.py:382-384), so `src` can be either a `Variable`-backed
    /// `ValueId` or a `Constant` literal — carried here as
    /// [`LinkArg::Value`] / [`LinkArg::Const`] respectively.
    Move { dst: ValueId, src: LinkArg },
    /// Save a value into the per-kind tmpreg, to break a cycle in a
    /// link renaming. Always paired with a later `Pop`.
    ///
    /// RPython `flatten.py:329` `self.emitline('%s_push' % kind, v)`.
    /// Blackhole handler: `blackhole.py:661-669` `bhimpl_{int,ref,float}_push`.
    /// Only register sources participate in cycle breaking, so `Push`
    /// stays `ValueId`-typed even though [`Move`] can copy constants.
    Push(ValueId),
    /// Restore a value from the per-kind tmpreg into `dst`, completing
    /// a cycle break started by a prior `Push`.
    ///
    /// RPython `flatten.py:331` `self.emitline('%s_pop' % kind, "->", w)`.
    /// Blackhole handler: `blackhole.py:671-679` `bhimpl_{int,ref,float}_pop`.
    Pop(ValueId),
    /// RPython: `('last_exception', '->', result)`.
    LastException { dst: ValueId },
    /// RPython: `('last_exc_value', '->', result)`.
    LastExcValue { dst: ValueId },
    /// Liveness marker — expanded by `compute_liveness()` to include
    /// all values alive at this point.
    ///
    /// RPython: `-live-` operation. Inserted by jtransform after calls
    /// that may need guard resumption (call_may_force, residual_call,
    /// inline_call, recursive_call). The liveness pass expands the
    /// `live_values` set to include all registers alive at this point.
    Live {
        /// Values known to be live (forced by jtransform).
        /// `compute_liveness()` expands this set.
        live_values: Vec<ValueId>,
    },
    /// Re-raise the current exception.
    /// RPython: `('reraise',)`.
    Reraise,
    /// RPython `flatten.py:130-138` `make_return`:
    ///   `{kind}_return` with a single arg when the final block returns
    ///   a non-void value.  Blackhole: `blackhole.py:841-857` sets
    ///   `_return_type = kind` and raises `LeaveFrame`.
    IntReturn(LinkArg),
    /// RPython `flatten.py:137` `ref_return` — blackhole at
    /// `blackhole.py:847-851`.
    RefReturn(LinkArg),
    /// RPython `flatten.py:137` `float_return` — blackhole at
    /// `blackhole.py:853-857`.
    FloatReturn(LinkArg),
    /// RPython `flatten.py:136` `void_return` — blackhole at
    /// `blackhole.py:859-863`.
    VoidReturn,
    /// RPython `flatten.py:139-143` `make_return` with a 2-inputarg
    /// final block: emit `raise` on the `evalue` (second inputarg).
    /// Blackhole: `blackhole.py:1000 bhimpl_raise(excvalue)`.
    Raise(LinkArg),
    /// RPython `flatten.py:146` / `:238` / `:293` `emitline('---')`.
    /// End-of-block marker placed after every terminator (return /
    /// raise / reraise / unreachable / goto-back-to-seen-block).
    /// Resets the alive set in liveness analysis (`liveness.py:55-57`)
    /// and is skipped during bytecode emission
    /// (`assembler.py:141-142`).
    EndOfBlock,
    /// RPython `flatten.py:292` `emitline("unreachable")` and
    /// `blackhole.py:962-964` `bhimpl_unreachable()`.
    ///
    /// Emitted after a switch with no `default` exit so that an
    /// unmatched switch value lands on a real opcode that raises
    /// `AssertionError`. Distinct from [`FlatOp::EndOfBlock`] which
    /// is a placement separator and emits no bytecode.
    Unreachable,
}

/// Register kind for a value (RPython regalloc).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegKind {
    Int,
    Ref,
    Float,
}

/// Result of the flatten pass.
#[derive(Debug, Clone)]
pub struct SSARepr {
    pub name: String,
    pub insns: Vec<FlatOp>,
    /// Total number of values used (for register allocation).
    pub num_values: usize,
    /// Number of basic blocks in the source graph.
    pub num_blocks: usize,
    /// Value kinds inferred from the type resolution pass.
    pub value_kinds: std::collections::HashMap<ValueId, RegKind>,
    /// flatten.py / assembler.py `ssarepr._insns_pos` — byte position
    /// of each instruction in the final bytecode, populated by the
    /// assembler.  `format.py:57-60` uses it to prefix every line with
    /// the position when set.  `None` when the SSARepr has not yet been
    /// assembled, matching upstream's `if ssarepr._insns_pos:` guard.
    pub insns_pos: Option<Vec<usize>>,
}

fn is_bool_branch(block: &crate::model::Block) -> bool {
    if !matches!(block.exitswitch, Some(ExitSwitch::Value(_))) || block.exits.len() != 2 {
        return false;
    }
    // RPython `flatten.py:244-246` accepts both orderings: `linkfalse,
    // linktrue = block.exits` followed by an `if linkfalse.llexitcase
    // == True: linkfalse, linktrue = linktrue, linkfalse` swap.
    let cases = (
        bool_llexitcase(&block.exits[0]),
        bool_llexitcase(&block.exits[1]),
    );
    matches!(cases, (Some(false), Some(true)) | (Some(true), Some(false)),)
}

fn bool_llexitcase(link: &Link) -> Option<bool> {
    match link.llexitcase.as_ref() {
        Some(ConstValue::Bool(value)) => return Some(*value),
        Some(_) => return None,
        None => {}
    }
    // Structural adaptation for direct semantic graphs that have not
    // gone through the Rust rtyper analogue yet.  RPython's codewriter
    // path reads `link.llexitcase` here.
    match &link.exitcase {
        Some(ExitCase::Bool(value)) => Some(*value),
        Some(ExitCase::Const(ConstValue::Bool(value))) => Some(*value),
        _ => None,
    }
}

fn is_default_exitcase(exitcase: &Option<ExitCase>) -> bool {
    matches!(
        exitcase,
        Some(ExitCase::Const(value)) if value.string_eq("default")
    )
}

fn signed_switch_key(value: &ConstValue) -> Option<i64> {
    match value {
        ConstValue::Int(value) => Some(*value),
        ConstValue::Bool(value) => Some(i64::from(*value)),
        _ => None,
    }
}

fn switch_llexitcase_key(link: &Link) -> Option<i64> {
    if let Some(llexitcase) = &link.llexitcase {
        return signed_switch_key(llexitcase);
    }
    // Structural adaptation for the Rust front-end's semantic graphs:
    // RPython's rtyper has already materialized `link.llexitcase`
    // before `flatten.py:274,296-298`, but direct front-end tests and
    // narrow graph builders can still carry the low-level integer only
    // in `exitcase`.
    match &link.exitcase {
        Some(ExitCase::Const(value)) => signed_switch_key(value),
        _ => None,
    }
}

/// Flatten a FunctionGraph into a linear instruction sequence.
///
/// RPython equivalent: `flatten_graph(graph, regallocs)` from flatten.py.
///
/// `regallocs` is the per-kind register-allocation result produced by
/// the preceding `perform_all_register_allocations` pass. Upstream's
/// `insert_renamings` reads it via `getcolor(v)` to decide cycle-break
/// on the assigned color, not on the pre-regalloc ValueId identity.
///
/// Block ordering: entry first, then BFS order. Back-edges (loops)
/// become jumps to earlier labels.
pub fn flatten(graph: &FunctionGraph, regallocs: &HashMap<RegKind, RegAllocResult>) -> SSARepr {
    let mut ops = Vec::new();
    let mut block_labels: std::collections::HashMap<BlockId, Label> =
        std::collections::HashMap::new();
    let mut next_label = 0usize;

    // Assign labels to all blocks
    let order = block_order(graph);
    for &bid in &order {
        block_labels.insert(bid, Label(next_label));
        next_label += 1;
    }

    // Emit instructions in block order
    for &bid in &order {
        let block = graph.block(bid);
        let label = block_labels[&bid];

        // Label
        ops.push(FlatOp::Label(label));

        // Ops
        for op in &block.operations {
            if matches!(&op.kind, crate::model::OpKind::Input { .. }) {
                continue;
            } else if matches!(&op.kind, crate::model::OpKind::Live) {
                // RPython: -live- op becomes FlatOp::Live marker
                ops.push(FlatOp::Live {
                    live_values: Vec::new(),
                });
            } else {
                ops.push(FlatOp::Op(op.clone()));
            }
        }

        // RPython flatten.py:177-278 `insert_exits()`.
        if block.exits.len() == 1 {
            make_link(graph, &mut ops, &block_labels, &block.exits[0], regallocs);
        } else if block.canraise() {
            debug_assert_eq!(block.exits[0].exitcase, None);
            if let Some(ovf_op) = block
                .operations
                .last()
                .and_then(|op| overflow_jump_op(op, Label(next_label)))
            {
                let ovf_landing = Label(next_label);
                next_label += 1;
                let last_flat_op = ops.pop();
                debug_assert!(matches!(last_flat_op, Some(FlatOp::Op(_))));
                debug_assert!(block.exits.len() == 2 || block.exits.len() == 3);
                ops.push(ovf_op);
                make_link(graph, &mut ops, &block_labels, &block.exits[0], regallocs);
                ops.push(FlatOp::Label(ovf_landing));
                make_exception_link(
                    graph,
                    &mut ops,
                    &block_labels,
                    &block.exits[1],
                    regallocs,
                    true,
                );
                if block.exits.len() == 3 {
                    debug_assert!(block.exits[2].catches_all_exceptions());
                    make_exception_link(
                        graph,
                        &mut ops,
                        &block_labels,
                        &block.exits[2],
                        regallocs,
                        false,
                    );
                }
            } else {
                // RPython flatten.py:205-218: walk the operations tail backward
                // past `-live-` markers to find the real raising op.  If the last
                // op is NOT `-live-` (RPython's `index == -1` case), the call at
                // the tail did not declare `can_raise` and this block cannot
                // actually raise — emit only the normal link and move on.
                let last_is_live = matches!(
                    block.operations.last().map(|op| &op.kind),
                    Some(crate::model::OpKind::Live)
                );
                if !last_is_live {
                    make_link(graph, &mut ops, &block_labels, &block.exits[0], regallocs);
                } else {
                    ops.push(FlatOp::CatchException {
                        target: Label(next_label),
                    });
                    let normal_landing = Label(next_label);
                    next_label += 1;
                    make_link(graph, &mut ops, &block_labels, &block.exits[0], regallocs);
                    ops.push(FlatOp::Label(normal_landing));
                    let mut catches_all = false;
                    for link in &block.exits[1..] {
                        if link.catches_all_exceptions() {
                            make_exception_link(
                                graph,
                                &mut ops,
                                &block_labels,
                                link,
                                regallocs,
                                false,
                            );
                            catches_all = true;
                            break;
                        }
                        let mismatch_landing = Label(next_label);
                        next_label += 1;
                        let llexitcase = link
                            .llexitcase
                            .clone()
                            .expect("typed exception links need llexitcase for parity");
                        ops.push(FlatOp::GotoIfExceptionMismatch {
                            llexitcase,
                            target: mismatch_landing,
                        });
                        make_exception_link(graph, &mut ops, &block_labels, link, regallocs, false);
                        ops.push(FlatOp::Label(mismatch_landing));
                    }
                    if !catches_all {
                        ops.push(FlatOp::Reraise);
                        // RPython `flatten.py:238` `emitline('---')` after
                        // `reraise_xxx` — close the block in liveness.
                        ops.push(FlatOp::EndOfBlock);
                    }
                }
            }
        } else if is_bool_branch(block) {
            let cond = match block.exitswitch {
                Some(ExitSwitch::Value(cond)) => cond,
                _ => unreachable!(),
            };
            // RPython `flatten.py:244-246`:
            //   linkfalse, linktrue = block.exits
            //   if linkfalse.llexitcase == True:
            //       linkfalse, linktrue = linktrue, linkfalse
            let (linkfalse, linktrue) = if bool_llexitcase(&block.exits[0]) == Some(true) {
                (&block.exits[1], &block.exits[0])
            } else {
                (&block.exits[0], &block.exits[1])
            };

            // RPython flatten.py:259: -live- before goto_if_not.
            ops.push(FlatOp::Live {
                live_values: Vec::new(),
            });
            // Fresh TLabel for the false-path landing pad (distinct from
            // `Label(linkfalse.target)`), matching `TLabel(linkfalse)`.
            let false_landing = Label(next_label);
            next_label += 1;
            // RPython flatten.py:260: goto_if_not(cond, TLabel(linkfalse)).
            ops.push(FlatOp::GotoIfNot {
                cond,
                target: false_landing,
            });
            // RPython flatten.py:264: true path (fallthrough) — make_link(linktrue).
            make_link(graph, &mut ops, &block_labels, linktrue, regallocs);
            // RPython flatten.py:266-267: false path — Label(linkfalse)
            // + make_link(linkfalse).
            ops.push(FlatOp::Label(false_landing));
            make_link(graph, &mut ops, &block_labels, linkfalse, regallocs);
        } else if block.exits.len() >= 2 && matches!(block.exitswitch, Some(ExitSwitch::Value(_))) {
            let cond = match block.exitswitch {
                Some(ExitSwitch::Value(cond)) => cond,
                _ => unreachable!(),
            };
            let kind = value_kind(cond, regallocs);
            assert_eq!(kind, 'i', "switch exitswitch must be int");
            let default_link = block
                .exits
                .last()
                .filter(|link| is_default_exitcase(&link.exitcase));
            let mut switches: Vec<(i64, &Link)> = Vec::new();
            for link in &block.exits {
                if !is_default_exitcase(&link.exitcase) {
                    let key = switch_llexitcase_key(link).unwrap_or_else(|| {
                        panic!(
                            "unsupported switch llexitcase {:?} (exitcase {:?}) in block {:?}",
                            link.llexitcase, link.exitcase, block.id
                        )
                    });
                    switches.push((key, link));
                }
            }
            switches.sort_by_key(|(key, _)| *key);
            let mut targets = Vec::with_capacity(switches.len());
            for (key, _) in &switches {
                let landing = Label(next_label);
                next_label += 1;
                targets.push((*key, landing));
            }

            // RPython `flatten.py:302`: `-live-` before `switch` so the
            // guard_value path can reconstruct state.
            ops.push(FlatOp::Live {
                live_values: Vec::new(),
            });
            ops.push(FlatOp::Switch {
                value: cond,
                targets: targets.clone(),
            });

            if let Some(link) = default_link {
                make_link(graph, &mut ops, &block_labels, link, regallocs);
            } else {
                // RPython `flatten.py:292-293`:
                //     self.emitline("unreachable")
                //     self.emitline("---")
                // The real `unreachable` opcode raises `AssertionError`
                // at runtime if the switch value matches no key; the
                // `---` resets liveness past this point.
                ops.push(FlatOp::Unreachable);
                ops.push(FlatOp::EndOfBlock);
            }

            for ((_, link), (_, landing)) in switches.into_iter().zip(targets.into_iter()) {
                ops.push(FlatOp::Label(landing));
                ops.push(FlatOp::Live {
                    live_values: Vec::new(),
                });
                make_link(graph, &mut ops, &block_labels, link, regallocs);
            }
        } else if block.exits.is_empty() {
            // RPython `flatten.py:106-109` `make_bytecode_block`:
            //   if block.exits == (): self.make_return(block.inputargs)
            // Final block — emit the matching return from the block's
            // own inputargs.
            let final_args: Vec<LinkArg> =
                block.inputargs.iter().copied().map(LinkArg::from).collect();
            make_return(graph, &mut ops, regallocs, &final_args);
        } else {
            // Upstream `flatten.py:177-278 insert_exits` only handles the
            // four Block.exits shapes above (single goto, can-raise
            // multi-exit, 2-way bool branch, empty final block).  A block
            // whose exits don't match any of those has been produced by
            // an unsupported front-end construct.
            panic!(
                "unsupported block.exits shape: {} exits, exitswitch = {:?}",
                block.exits.len(),
                block.exitswitch,
            );
        }
    }

    // Count total values
    let mut max_value = 0usize;
    for block in &graph.blocks {
        for &arg in &block.inputargs {
            max_value = max_value.max(arg.0 + 1);
        }
        for op in &block.operations {
            if let Some(ValueId(v)) = op.result {
                max_value = max_value.max(v + 1);
            }
        }
    }
    for op in &ops {
        match op {
            FlatOp::Op(_) => {}
            FlatOp::Move {
                dst: ValueId(d),
                src,
            } => {
                max_value = max_value.max(*d + 1);
                if let Some(ValueId(s)) = src.as_value() {
                    max_value = max_value.max(s + 1);
                }
            }
            FlatOp::Push(ValueId(v)) => {
                max_value = max_value.max(*v + 1);
            }
            FlatOp::Pop(ValueId(v)) => {
                max_value = max_value.max(*v + 1);
            }
            FlatOp::GotoIfNot {
                cond: ValueId(c), ..
            }
            | FlatOp::Switch {
                value: ValueId(c), ..
            } => {
                max_value = max_value.max(*c + 1);
            }
            FlatOp::IntBinOpJumpIfOvf {
                lhs: ValueId(lhs),
                rhs: ValueId(rhs),
                dst: ValueId(dst),
                ..
            } => {
                max_value = max_value.max(*lhs + 1);
                max_value = max_value.max(*rhs + 1);
                max_value = max_value.max(*dst + 1);
            }
            FlatOp::LastException { dst: ValueId(d) }
            | FlatOp::LastExcValue { dst: ValueId(d) } => {
                max_value = max_value.max(*d + 1);
            }
            FlatOp::IntReturn(v)
            | FlatOp::RefReturn(v)
            | FlatOp::FloatReturn(v)
            | FlatOp::Raise(v) => {
                if let Some(ValueId(v)) = v.as_value() {
                    max_value = max_value.max(v + 1);
                }
            }
            _ => {}
        }
    }

    SSARepr {
        name: graph.name.clone(),
        insns: ops,
        num_values: max_value,
        num_blocks: graph.blocks.len(),
        value_kinds: std::collections::HashMap::new(),
        insns_pos: None,
    }
}

/// Flatten with type information from rtype pass.
///
/// Like `flatten()` but populates `value_kinds` from the TypeResolutionState.
pub fn flatten_with_types(
    graph: &FunctionGraph,
    types: &crate::jit_codewriter::type_state::TypeResolutionState,
    regallocs: &HashMap<RegKind, RegAllocResult>,
) -> SSARepr {
    let mut result = flatten(graph, regallocs);
    result.value_kinds = crate::jit_codewriter::type_state::build_value_kinds(types);
    // Seed canonical `exceptblock.inputargs` kinds if the rtyper pass
    // missed them — same contract as
    // `perform_all_register_allocations`.
    let except_args = &graph.block(graph.exceptblock).inputargs;
    if except_args.len() == 2 {
        result
            .value_kinds
            .entry(except_args[0])
            .or_insert(RegKind::Int);
        result
            .value_kinds
            .entry(except_args[1])
            .or_insert(RegKind::Ref);
    }
    result
}

/// Compute block ordering (entry first, then BFS).
fn block_order(graph: &FunctionGraph) -> Vec<BlockId> {
    let mut order = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();

    queue.push_back(graph.startblock);
    visited.insert(graph.startblock);

    while let Some(bid) = queue.pop_front() {
        order.push(bid);
        let block = graph.block(bid);
        for succ in successors(block) {
            if visited.insert(succ) {
                queue.push_back(succ);
            }
        }
    }

    // Add any unreachable blocks (shouldn't happen in well-formed graphs)
    for block in &graph.blocks {
        if !visited.contains(&block.id) {
            order.push(block.id);
        }
    }

    order
}

/// RPython `flatten.py:148-155` `make_link(link, handling_ovf)`.
///
/// The "target is a final block" optimization collapses
/// `goto + label + make_return` into a direct `make_return(link.args)`.
/// Fallback: `insert_renamings(link) + goto(TLabel(link.target))` —
/// `insert_renamings` tail-emits `generate_last_exc` as part of its
/// own contract.
fn make_link(
    graph: &FunctionGraph,
    ops: &mut Vec<FlatOp>,
    block_labels: &std::collections::HashMap<BlockId, Label>,
    link: &Link,
    regallocs: &HashMap<RegKind, RegAllocResult>,
) {
    let target_block = graph.block(link.target);
    // RPython `flatten.py:149-153`:
    //   if (link.target.exits == ()
    //       and link.last_exception not in link.args
    //       and link.last_exc_value not in link.args):
    //       self.make_return(link.args); return
    // Skip the renaming + goto and emit the final return inline.
    if target_block.exits.is_empty() {
        let carries_exception_args = link
            .last_exception
            .as_ref()
            .is_some_and(|arg| link.args.contains(arg))
            || link
                .last_exc_value
                .as_ref()
                .is_some_and(|arg| link.args.contains(arg));
        if !carries_exception_args {
            let _ = target_block;
            make_return(graph, ops, regallocs, &link.args);
            return;
        }
    }
    insert_renamings(link, &target_block.inputargs, regallocs, ops);
    ops.push(FlatOp::Jump(block_labels[&link.target]));
}

/// RPython `flatten.py:336-347` `generate_last_exc(link, inputargs)`.
///
/// Emitted by `insert_renamings` at the tail of its rename work.
/// Walks `zip(link.args, inputargs)` and, for every position where the
/// link-side arg is one of the exception Variables, writes the
/// matching `last_exception` / `last_exc_value` op directly into the
/// target inputarg's register.
fn generate_last_exc(link: &Link, target_inputargs: &[ValueId], ops: &mut Vec<FlatOp>) {
    if link.last_exception.is_none() && link.last_exc_value.is_none() {
        return;
    }
    for (v, w) in link.args.iter().zip(target_inputargs.iter()) {
        if Some(v) == link.last_exception.as_ref() {
            ops.push(FlatOp::LastException { dst: *w });
        }
    }
    for (v, w) in link.args.iter().zip(target_inputargs.iter()) {
        if Some(v) == link.last_exc_value.as_ref() {
            ops.push(FlatOp::LastExcValue { dst: *w });
        }
    }
}

/// RPython `flatten.py:130-146` `make_return(args)`.
///
/// Emits the matching `{kind}_return` / `void_return` / `raise` + `---`
/// (`Unreachable`) pair from the final-block inputargs (or, when called
/// from the `make_link` optimization, directly from the link's args).
fn make_return(
    graph: &FunctionGraph,
    ops: &mut Vec<FlatOp>,
    regallocs: &HashMap<RegKind, RegAllocResult>,
    args: &[LinkArg],
) {
    let arg_kind = |arg: &LinkArg, fallback: Option<ValueId>, context: &str| match arg {
        LinkArg::Value(value) => value_kind(*value, regallocs),
        LinkArg::Const(_) => fallback
            .map(|value| value_kind(value, regallocs))
            .unwrap_or_else(|| panic!("{context}: missing target inputarg kind for Constant")),
    };
    match args.len() {
        1 => {
            // `flatten.py:131-138` return-from-function.
            let kind = arg_kind(
                &args[0],
                graph.block(graph.returnblock).inputargs.first().copied(),
                "make_return",
            );
            match kind {
                'v' => ops.push(FlatOp::VoidReturn),
                'i' => ops.push(FlatOp::IntReturn(args[0].clone())),
                'r' => ops.push(FlatOp::RefReturn(args[0].clone())),
                'f' => ops.push(FlatOp::FloatReturn(args[0].clone())),
                _ => unreachable!("unexpected kind {kind} for return value"),
            }
        }
        2 => {
            // `flatten.py:139-143` raise-from-function.  The final
            // exceptblock inputargs are ValueIds in majit, i.e. the
            // direct analogue of upstream Variables here, so mirror the
            // upstream `-live-` hack before `raise`.
            ops.push(FlatOp::Live {
                live_values: Vec::new(),
            });
            let _ = arg_kind(
                &args[1],
                graph.block(graph.exceptblock).inputargs.get(1).copied(),
                "make_return",
            );
            ops.push(FlatOp::Raise(args[1].clone()));
        }
        0 => {
            // RPython reaches `make_return` only for exits == () blocks,
            // whose inputargs are always `[return_var]` or
            // `[etype, evalue]`.  An empty args list corresponds to a
            // declared-void final block without a rtyper-supplied
            // Void Variable — the pyre adaptation emits `void_return`
            // and drops the redundant argument.
            ops.push(FlatOp::VoidReturn);
        }
        other => panic!("make_return: unexpected final-block inputarg count {other}"),
    }
    // RPython `flatten.py:146` `emitline('---')`.
    ops.push(FlatOp::EndOfBlock);
}

fn value_kind(value: ValueId, regallocs: &HashMap<RegKind, RegAllocResult>) -> char {
    for (kind, ra) in regallocs {
        if ra.coloring.contains_key(&value) {
            return match kind {
                RegKind::Int => 'i',
                RegKind::Ref => 'r',
                RegKind::Float => 'f',
            };
        }
    }
    // `lltype.Void` is not assigned a color by regalloc
    // (`flatten.py:325 if v.concretetype is not lltype.Void`).
    'v'
}

/// Kind of a [`LinkArg`] for opname selection — upstream `assembler.py:168-170`
/// `getkind(x.concretetype)` for `Constant`, `x.kind` for `Register`.
///
/// Returns `'i'` / `'r'` / `'f'` / `'v'` matching RPython `KINDS`.
#[allow(dead_code)]
pub(crate) fn linkarg_kind(arg: &LinkArg, regallocs: &HashMap<RegKind, RegAllocResult>) -> char {
    match arg {
        LinkArg::Value(v) => value_kind(*v, regallocs),
        LinkArg::Const(cv) => constvalue_kind(cv),
    }
}

/// RPython `rpython/rtyper/lltypesystem/lltype.py` + `rpython/jit/codewriter/support.py`
/// `getkind` — map a Constant's concretetype to a `KINDS` char.
///
/// Pyre's [`ConstValue`] carries the effective lltype by variant: `Int`
/// is `lltype.Signed`, `Float` is `lltype.Float`, `HostObject`/`None`/`Str`
/// are gcref-bearing (kind `'r'`), `SpecTag` is a Signed wrapper.
pub(crate) fn constvalue_kind(cv: &ConstValue) -> char {
    match cv {
        ConstValue::Int(_)
        | ConstValue::Bool(_)
        | ConstValue::SpecTag(_)
        | ConstValue::LLAddress(_) => 'i',
        ConstValue::Float(_) => 'f',
        ConstValue::None
        | ConstValue::ByteStr(_)
        | ConstValue::UniStr(_)
        | ConstValue::HostObject(_)
        | ConstValue::Tuple(_)
        | ConstValue::List(_)
        | ConstValue::Dict(_)
        | ConstValue::Graphs(_)
        | ConstValue::LowLevelType(_)
        | ConstValue::Code(_)
        | ConstValue::LLPtr(_)
        | ConstValue::Function(_)
        | ConstValue::Atom(_)
        | ConstValue::Placeholder => 'r',
    }
}

fn overflow_jump_op(op: &SpaceOperation, target: Label) -> Option<FlatOp> {
    let (name, lhs, rhs) = match &op.kind {
        crate::model::OpKind::BinOp { op, lhs, rhs, .. } => (op.as_str(), *lhs, *rhs),
        _ => return None,
    };
    let opcode = match name {
        "add_ovf" => IntOvfOp::Add,
        "sub_ovf" => IntOvfOp::Sub,
        "mul_ovf" => IntOvfOp::Mul,
        _ => return None,
    };
    let dst = op
        .result
        .expect("overflow-checked arithmetic op needs a result for flatten parity");
    Some(FlatOp::IntBinOpJumpIfOvf {
        op: opcode,
        target,
        lhs,
        rhs,
        dst,
    })
}

fn overflow_error_instance() -> ConstValue {
    ConstValue::HostObject(
        crate::flowspace::model::HOST_ENV
            .lookup_standard_exception_instance("OverflowError")
            .expect("HOST_ENV missing standard OverflowError instance"),
    )
}

/// RPython `flatten.py:157-175` `make_exception_link(link, handling_ovf)`.
///
/// Special-cases the bare reraise link (target has no ops and
/// `link.args == [link.last_exception, link.last_exc_value]`) into a
/// `reraise` + `---` pair; otherwise delegates to `make_link`, which
/// handles the `last_exception` / `last_exc_value` emission through
/// `generate_last_exc` at the tail of `insert_renamings` — the exception
/// Variables are written directly into the target inputarg's register,
/// not into the prevblock-side Variable.
fn make_exception_link(
    graph: &FunctionGraph,
    ops: &mut Vec<FlatOp>,
    block_labels: &std::collections::HashMap<BlockId, Label>,
    link: &Link,
    regallocs: &HashMap<RegKind, RegAllocResult>,
    handling_ovf: bool,
) {
    debug_assert!(link.last_exception.is_some());
    debug_assert!(link.last_exc_value.is_some());
    let target = graph.block(link.target);
    // `flatten.py:160-168` keys this collapse only on the empty-target
    // shape plus the exact `[last_exception, last_exc_value]` args.
    // Preserve that structure literally here; any typed handler that
    // would otherwise match must be fixed at graph-construction time to
    // carry a non-empty body, as upstream flow graphs do.
    if target.operations.is_empty()
        && link.args
            == vec![
                link.last_exception.clone().unwrap(),
                link.last_exc_value.clone().unwrap(),
            ]
    {
        if handling_ovf {
            ops.push(FlatOp::Raise(LinkArg::Const(overflow_error_instance())));
        } else {
            ops.push(FlatOp::Reraise);
        }
        // RPython `flatten.py:238` `emitline('---')` after raise/reraise.
        ops.push(FlatOp::EndOfBlock);
        return;
    }
    make_link(graph, ops, block_labels, link, regallocs);
}

/// Get successor block IDs from orthodox block exits.
///
/// RPython `flowspace/model.py:66-76 FunctionGraph.iterblocks` derives
/// the successor set from `Block.exits` directly; final blocks
/// (returnblock / exceptblock with `exits == ()`) have no successors.
fn successors(block: &crate::model::Block) -> Vec<BlockId> {
    block.exits.iter().map(|link| link.target).collect()
}

/// `flatten.py:306-334` `def insert_renamings(self, link)`.
///
/// Emits the ordered series of `%s_copy` / `%s_push` / `%s_pop` ops
/// that resolve a link's argument-to-inputarg renaming, breaking any
/// cycles via `reorder_renaming_list`. Mirrors upstream's structure
/// line-by-line:
///
/// 1. Build `lst = [(src_color, dst_color)]` filtering extravars.
/// 2. `lst.sort(key=lambda (v, w): w.index)` — global sort by dst color
///    (`flatten.py:312`).
/// 3. Skip identity entries (`flatten.py:314 if v == w: continue`) and
///    group by `w.kind` (`flatten.py:316-318`).
/// 4. For each kind in `KINDS = ['int', 'ref', 'float']` order, run
///    `reorder_renaming_list` on that kind's `(frm, to)` pair and
///    emit `{kind}_push` / `{kind}_pop` / `{kind}_copy`
///    (`flatten.py:319-333`).
/// 5. Tail-emit `generate_last_exc` (`flatten.py:334`).
///
/// pyre's `FlatOp::Move` / `Push` / `Pop` do not carry an explicit
/// `kind` field — the assembler (`assembler.rs::write_insn`) looks
/// each ValueId's kind up through `value_kinds` and emits the
/// matching `{kind}_copy` / `{kind}_push` / `{kind}_pop` opname.
/// The kind-grouping loop therefore affects emission ORDER (upstream's
/// `int_copy`s all come before `ref_copy`s, which come before
/// `float_copy`s) rather than opname selection, and the cycle-break is
/// computed per kind so that a Push in one kind's bank cannot pair
/// with a Pop in another kind's bank.
///
/// `regallocs` supplies `getcolor(v)` per upstream — cycle detection
/// operates on colors, not ValueIds, so it remains correct whether
/// coalescing merged ValueIds into one color or split them across
/// separate colors.
///
/// Upstream:
/// ```py
/// def insert_renamings(self, link):
///     renamings = {}
///     lst = [(self.getcolor(v), self.getcolor(link.target.inputargs[i]))
///            for i, v in enumerate(link.args)
///            if v.concretetype is not lltype.Void and
///               v not in (link.last_exception, link.last_exc_value)]
///     lst.sort(key=lambda(v, w): w.index)
///     for v, w in lst:
///         if v == w:
///             continue
///         frm, to = renamings.setdefault(w.kind, ([], []))
///         frm.append(v)
///         to.append(w)
///     for kind in KINDS:
///         if kind in renamings:
///             frm, to = renamings[kind]
///             result = reorder_renaming_list(frm, to)
///             for v, w in result:
///                 if w is None:
///                     self.emitline('%s_push' % kind, v)
///                 elif v is None:
///                     self.emitline('%s_pop' % kind, "->", w)
///                 else:
///                     self.emitline('%s_copy' % kind, v, "->", w)
///     self.generate_last_exc(link, link.target.inputargs)
/// ```
///
/// The `link` + `target_inputargs` signature mirrors upstream's
/// `def insert_renamings(self, link)` — `target_inputargs` is
/// `link.target.inputargs` threaded through pyre's non-`self` call
/// shape.  The filter drops link args that are `link.last_exception` /
/// `link.last_exc_value` (upstream `flatten.py:310-311` `v not in
/// (link.last_exception, link.last_exc_value)`); those values are
/// instead placed by the `generate_last_exc` tail emission at
/// `flatten.py:334`.
pub fn insert_renamings(
    link: &Link,
    target_inputargs: &[ValueId],
    regallocs: &HashMap<RegKind, RegAllocResult>,
    ops: &mut Vec<FlatOp>,
) {
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    enum RenameItem {
        Color(usize),
        Const(ConstValue),
    }

    // Resolve each ValueId to its regalloc-assigned `(kind, color)`,
    // mirroring upstream's `self.getcolor(v)` + `self.getcolor(w)`.
    // Returning the kind alongside the color lets the `flatten.py:317`
    // `renamings.setdefault(w.kind, ...)` grouping key come from the
    // coloring rather than a separate `value_kinds` lookup.
    let get_kind_color = |v: ValueId| -> Option<(RegKind, usize)> {
        for (&kind, ra) in regallocs {
            if let Some(&c) = ra.coloring.get(&v) {
                return Some((kind, c));
            }
        }
        None
    };

    // Upstream `flatten.py:308` requires `len(link.args) ==
    // len(link.target.inputargs)` — the helper assumes equality and
    // indexes both sides pairwise. The graph builder must provide
    // well-formed links; any mismatch is a front-end bug.
    assert_eq!(
        link.args.len(),
        target_inputargs.len(),
        "insert_renamings: link.args and target.inputargs must have equal length"
    );

    // Each entry captures one raw (src, dst) pair with its dst `(kind,
    // color)` alongside the original ValueIds — the ValueIds are
    // needed to reconstruct `FlatOp::Move/Push/Pop` after
    // `reorder_renaming_list` has shuffled the group.
    struct Entry {
        src: RenameItem,
        src_vid: Option<ValueId>,
        dst_color: usize,
        dst_kind: RegKind,
        dst_vid: ValueId,
    }
    let mut lst: Vec<Entry> = Vec::with_capacity(link.args.len());
    for (v, w) in link.args.iter().zip(target_inputargs.iter()) {
        // `flatten.py:310-311` skip.
        if Some(v) == link.last_exception.as_ref() || Some(v) == link.last_exc_value.as_ref() {
            continue;
        }
        let Some((dst_kind, dst_color)) = get_kind_color(*w) else {
            continue;
        };
        let (src, src_vid) = match v {
            LinkArg::Value(value) => {
                let Some((_, v_color)) = get_kind_color(*value) else {
                    continue;
                };
                // `flatten.py:314 if v == w: continue` — after regalloc,
                // equal colors mean the renaming is the identity.
                if v_color == dst_color {
                    continue;
                }
                (RenameItem::Color(v_color), Some(*value))
            }
            LinkArg::Const(value) => (RenameItem::Const(value.clone()), None),
        };
        lst.push(Entry {
            src,
            src_vid,
            dst_color,
            dst_kind,
            dst_vid: *w,
        });
    }

    // `flatten.py:312` `lst.sort(key=lambda (v, w): w.index)` — global
    // stable sort by destination color.  Within a kind group (post
    // split), the sort order is preserved — so each kind's `frm` /
    // `to` list is internally sorted by dst color.
    lst.sort_by_key(|e| e.dst_color);

    // `flatten.py:316-318` `renamings.setdefault(w.kind, ([], []))`
    // group by destination kind, preserving the sort order from above.
    let mut by_kind: HashMap<RegKind, Vec<Entry>> = HashMap::new();
    for entry in lst {
        by_kind.entry(entry.dst_kind).or_default().push(entry);
    }

    // `flatten.py:319-333` iterate kinds in `KINDS = ['int', 'ref',
    // 'float']` order so upstream's emission sequence (`int_copy` runs,
    // then `ref_copy` runs, then `float_copy` runs) is reproduced.
    for kind in [RegKind::Int, RegKind::Ref, RegKind::Float] {
        let Some(entries) = by_kind.remove(&kind) else {
            continue;
        };
        let frm: Vec<RenameItem> = entries.iter().map(|e| e.src.clone()).collect();
        let to: Vec<RenameItem> = entries
            .iter()
            .map(|e| RenameItem::Color(e.dst_color))
            .collect();
        let result = reorder_renaming_list(&frm, &to);

        // Map a color back to a representative ValueId in the current
        // kind group so `FlatOp::Move/Push/Pop` carry the original
        // identity.  Cycle-break keeps src/dst colors inside this group,
        // so the search is bounded by `entries`.
        let find_src_vid = |c: usize| -> ValueId {
            entries
                .iter()
                .find_map(|e| match e.src {
                    RenameItem::Color(sc) if sc == c => e.src_vid,
                    _ => None,
                })
                .unwrap_or_else(|| panic!("reorder_renaming_list missing source for color {c}"))
        };
        let find_dst_vid = |c: usize| -> ValueId {
            entries
                .iter()
                .find_map(|e| (e.dst_color == c).then_some(e.dst_vid))
                .unwrap_or_else(|| {
                    panic!("reorder_renaming_list missing destination for color {c}")
                })
        };

        for (v, w) in result {
            match (v, w) {
                // `if w is None: self.emitline('%s_push' % kind, v)`.
                (Some(RenameItem::Color(src_c)), None) => {
                    ops.push(FlatOp::Push(find_src_vid(src_c)))
                }
                // `elif v is None: self.emitline('%s_pop' % kind, "->", w)`.
                (None, Some(RenameItem::Color(dst_c))) => {
                    ops.push(FlatOp::Pop(find_dst_vid(dst_c)))
                }
                // `else: self.emitline('%s_copy' % kind, v, "->", w)`.
                (Some(RenameItem::Color(src_c)), Some(RenameItem::Color(dst_c))) => {
                    ops.push(FlatOp::Move {
                        src: LinkArg::Value(find_src_vid(src_c)),
                        dst: find_dst_vid(dst_c),
                    })
                }
                (Some(RenameItem::Const(value)), Some(RenameItem::Color(dst_c))) => {
                    ops.push(FlatOp::Move {
                        src: LinkArg::Const(value),
                        dst: find_dst_vid(dst_c),
                    })
                }
                (Some(RenameItem::Const(_)), None) => {
                    unreachable!("constant renaming sources cannot participate in cycles")
                }
                // Renaming destinations come exclusively from
                // `link.target.inputargs` (upstream `flatten.py:309`),
                // which are always colored Variables.
                (None, Some(RenameItem::Const(_)))
                | (Some(RenameItem::Color(_)), Some(RenameItem::Const(_)))
                | (Some(RenameItem::Const(_)), Some(RenameItem::Const(_))) => {
                    unreachable!("renaming destinations are always colored inputargs")
                }
                (None, None) => unreachable!("reorder_renaming_list never yields (None, None)"),
            }
        }
    }

    // RPython `flatten.py:334` `self.generate_last_exc(link, link.target.inputargs)`.
    generate_last_exc(link, target_inputargs, ops);
}

/// `flatten.py:395-414` `def reorder_renaming_list(frm, to):`.
///
/// Line-by-line port. Given two equal-length sequences `frm[i] -> to[i]`,
/// return an ordered list of `(src, dst)` pairs so that each move runs
/// after every read of its `dst` register has happened. Cycles are
/// broken by a `(src, None)` save and `(None, dst)` load pair:
///
/// ```py
/// def reorder_renaming_list(frm, to):
///     result = []
///     pending_indices = range(len(to))
///     while pending_indices:
///         not_read = dict.fromkeys([frm[i] for i in pending_indices])
///         still_pending_indices = []
///         for i in pending_indices:
///             if to[i] not in not_read:
///                 result.append((frm[i], to[i]))
///             else:
///                 still_pending_indices.append(i)
///         if len(pending_indices) == len(still_pending_indices):
///             # no progress -- there is a cycle
///             assert None not in not_read
///             result.append((frm[pending_indices[0]], None))
///             frm[pending_indices[0]] = None
///             continue
///         pending_indices = still_pending_indices
///     return result
/// ```
///
/// Each `(src, dst)` entry maps to one `%s_copy src -> dst` operation
/// emitted by `insert_renamings`; `(src, None)` maps to `%s_push src`
/// and `(None, dst)` maps to `%s_pop -> dst` (flatten.py:326-335).
///
/// `T: Eq + Clone + Hash` so the algorithm works for any register
/// representation — RPython uses `Register` objects keyed by identity,
/// we'll typically instantiate with `Register`, `u16` color indices,
/// or a mixed color/constant enum.
pub fn reorder_renaming_list<T>(frm: &[T], to: &[T]) -> Vec<(Option<T>, Option<T>)>
where
    T: Eq + Clone + std::hash::Hash,
{
    // Mutable copy so the `frm[pending_indices[0]] = None` cycle-break
    // write has a home. In Rust we use `Option<T>` in the working
    // buffer; `None` is the "register already saved on the stack"
    // marker, matching RPython's `frm[...] = None`.
    let mut frm: Vec<Option<T>> = frm.iter().cloned().map(Some).collect();
    let to: Vec<T> = to.to_vec();
    assert_eq!(frm.len(), to.len(), "frm and to must have equal length");

    let mut result: Vec<(Option<T>, Option<T>)> = Vec::new();
    // `pending_indices = range(len(to))`.
    let mut pending_indices: Vec<usize> = (0..to.len()).collect();

    // `while pending_indices:`.
    while !pending_indices.is_empty() {
        // `not_read = dict.fromkeys([frm[i] for i in pending_indices])`.
        // RPython builds a dict keyed on `frm[i]`; `None` entries mean
        // "already saved via push", which `to[i] not in not_read` checks
        // against.
        let not_read: std::collections::HashSet<Option<T>> =
            pending_indices.iter().map(|&i| frm[i].clone()).collect();
        let mut still_pending_indices: Vec<usize> = Vec::new();
        // `for i in pending_indices:`.
        for &i in &pending_indices {
            // `if to[i] not in not_read`.
            if !not_read.contains(&Some(to[i].clone())) {
                // `result.append((frm[i], to[i]))`.
                result.push((frm[i].clone(), Some(to[i].clone())));
            } else {
                // `still_pending_indices.append(i)`.
                still_pending_indices.push(i);
            }
        }
        // `if len(pending_indices) == len(still_pending_indices):`.
        if pending_indices.len() == still_pending_indices.len() {
            // `assert None not in not_read`.
            debug_assert!(
                !not_read.contains(&None),
                "reorder_renaming_list: duplicate cycle break"
            );
            // `result.append((frm[pending_indices[0]], None))`.
            let head = pending_indices[0];
            result.push((frm[head].clone(), None));
            // `frm[pending_indices[0]] = None`.
            frm[head] = None;
            continue;
        }
        pending_indices = still_pending_indices;
    }

    // After the main loop finishes, every `(src, None)` push needs a
    // matching `(None, dst)` pop at the tail of its cycle. RPython's
    // loop emits the pop naturally when the cycle's final read slot
    // becomes safe — but because `frm[head] = None` is an in-place
    // rewrite, the next iteration sees `frm[...] = None` and emits
    // `(None, to[...])` directly as part of `(frm[i], to[i])` above.
    // No separate pop stage needed.
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::ConstValue;
    use crate::model::{ExitCase, FunctionGraph, OpKind, SpaceOperation, exception_exitcase};

    /// Test helper — build a `regallocs` map that assigns each
    /// `ValueId(n)` the color `n` in `RegKind::Int`. This turns the
    /// color-based `insert_renamings` cycle-break into pure ValueId
    /// identity, matching the pre-regalloc reasoning used by the
    /// `insert_renamings_*` unit tests below.
    fn identity_regallocs(
        max_id: usize,
    ) -> std::collections::HashMap<RegKind, crate::regalloc::RegAllocResult> {
        let coloring: std::collections::HashMap<ValueId, usize> =
            (0..=max_id).map(|n| (ValueId(n), n)).collect();
        let num_regs = max_id + 1;
        let mut m = std::collections::HashMap::new();
        m.insert(
            RegKind::Int,
            crate::regalloc::RegAllocResult { coloring, num_regs },
        );
        m
    }

    #[test]
    fn flatten_single_block() {
        let mut graph = FunctionGraph::new("simple");
        let entry = graph.startblock;
        let v = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();
        graph.set_return(entry, Some(v));

        let flat = flatten(&graph, &identity_regallocs(8));
        assert_eq!(flat.name, "simple");
        // Label + ConstInt op = 2 flat ops
        assert!(flat.insns.len() >= 2);
        assert!(matches!(flat.insns[0], FlatOp::Label(Label(0))));
    }

    #[test]
    fn flatten_if_else_produces_jumps() {
        let mut graph = FunctionGraph::new("branch");
        let entry = graph.startblock;
        let cond = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let then_block = graph.create_block();
        let else_block = graph.create_block();
        let merge = graph.create_block();

        graph.set_branch(entry, cond, then_block, vec![], else_block, vec![]);
        graph.set_goto(then_block, merge, vec![]);
        graph.set_goto(else_block, merge, vec![]);
        graph.set_return(merge, None);

        let flat = flatten(&graph, &identity_regallocs(8));
        // Should have labels + jumps
        let has_jump = flat
            .insns
            .iter()
            .any(|op| matches!(op, FlatOp::Jump(_) | FlatOp::GotoIfNot { .. }));
        assert!(has_jump, "flattened if/else should have jumps");
        // Should have 4 labels (one per block)
        let label_count = flat
            .insns
            .iter()
            .filter(|op| matches!(op, FlatOp::Label(_)))
            .count();
        // 4 block labels + 1 false-path label from Branch (RPython goto_if_not convention)
        assert!(
            label_count >= 4,
            "should have at least 4 labels, got {label_count}"
        );
    }

    #[test]
    fn flatten_bool_branch_orders_by_llexitcase() {
        let mut graph = FunctionGraph::new("branch_llexitcase");
        let entry = graph.startblock;
        let cond = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let true_block = graph.create_block();
        let false_block = graph.create_block();
        graph.set_return(true_block, None);
        graph.set_return(false_block, None);
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::Value(cond)),
            vec![
                Link::new(Vec::new(), true_block, Some(ExitCase::Bool(false)))
                    .with_llexitcase(ConstValue::Bool(true)),
                Link::new(Vec::new(), false_block, Some(ExitCase::Bool(true)))
                    .with_llexitcase(ConstValue::Bool(false)),
            ],
        );

        let flat = flatten(&graph, &identity_regallocs(8));
        let goto_idx = flat
            .insns
            .iter()
            .position(|op| matches!(op, FlatOp::GotoIfNot { .. }))
            .expect("bool branch must emit goto_if_not");
        assert!(
            matches!(flat.insns.get(goto_idx + 1), Some(FlatOp::Jump(Label(1)))),
            "true fallthrough path must follow link.llexitcase, not flow-level exitcase: {:?}",
            flat.insns
        );
    }

    #[test]
    fn flatten_integer_switch_emits_switch_op() {
        let mut graph = FunctionGraph::new("switch");
        let entry = graph.startblock;
        let cond = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let case0 = graph.create_block();
        let case1 = graph.create_block();
        let default = graph.create_block();
        graph.set_return(case0, None);
        graph.set_return(case1, None);
        graph.set_return(default, None);
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::Value(cond)),
            vec![
                Link::new_mixed(Vec::new(), case0, Some(ExitCase::Const(ConstValue::Int(0))))
                    .with_llexitcase(ConstValue::Int(0)),
                Link::new_mixed(Vec::new(), case1, Some(ExitCase::Const(ConstValue::Int(1))))
                    .with_llexitcase(ConstValue::Int(1)),
                Link::new_mixed(
                    Vec::new(),
                    default,
                    Some(ExitCase::Const(ConstValue::byte_str("default"))),
                ),
            ],
        );

        let flat = flatten(&graph, &identity_regallocs(8));
        assert!(
            flat.insns.iter().any(|op| matches!(
                op,
                FlatOp::Switch { value, targets }
                    if *value == cond && targets.iter().map(|(key, _)| *key).collect::<Vec<_>>() == vec![0, 1]
            )),
            "integer switch should emit a Switch op: {:?}",
            flat.insns
        );
    }

    #[test]
    fn flatten_integer_switch_sorts_and_keys_by_llexitcase() {
        let mut graph = FunctionGraph::new("switch_llexitcase");
        let entry = graph.startblock;
        let cond = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let case_lowlevel_2 = graph.create_block();
        let case_lowlevel_1 = graph.create_block();
        graph.set_return(case_lowlevel_2, None);
        graph.set_return(case_lowlevel_1, None);
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::Value(cond)),
            vec![
                Link::new_mixed(
                    Vec::new(),
                    case_lowlevel_2,
                    Some(ExitCase::Const(ConstValue::Int(10))),
                )
                .with_llexitcase(ConstValue::Int(2)),
                Link::new_mixed(
                    Vec::new(),
                    case_lowlevel_1,
                    Some(ExitCase::Const(ConstValue::Int(20))),
                )
                .with_llexitcase(ConstValue::Int(1)),
            ],
        );

        let flat = flatten(&graph, &identity_regallocs(8));
        assert!(
            flat.insns.iter().any(|op| matches!(
                op,
                FlatOp::Switch { targets, .. }
                    if targets.iter().map(|(key, _)| *key).collect::<Vec<_>>() == vec![1, 2]
            )),
            "switch keys must come from sorted link.llexitcase, not flow-level exitcase: {:?}",
            flat.insns
        );
    }

    #[test]
    fn flatten_integer_zero_one_switch_is_not_bool_branch() {
        let mut graph = FunctionGraph::new("switch_zero_one");
        let entry = graph.startblock;
        let cond = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let case0 = graph.create_block();
        let case1 = graph.create_block();
        graph.set_return(case0, None);
        graph.set_return(case1, None);
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::Value(cond)),
            vec![
                Link::new_mixed(Vec::new(), case0, Some(ExitCase::Const(ConstValue::Int(0))))
                    .with_llexitcase(ConstValue::Int(0)),
                Link::new_mixed(Vec::new(), case1, Some(ExitCase::Const(ConstValue::Int(1))))
                    .with_llexitcase(ConstValue::Int(1)),
            ],
        );

        let flat = flatten(&graph, &identity_regallocs(8));
        assert!(
            flat.insns
                .iter()
                .any(|op| matches!(op, FlatOp::Switch { .. })),
            "integer 0/1 switch must stay a switch/id, not collapse to goto_if_not: {:?}",
            flat.insns
        );
        assert!(
            !flat
                .insns
                .iter()
                .any(|op| matches!(op, FlatOp::GotoIfNot { .. })),
            "integer 0/1 switch must not use the bool-branch lowering: {:?}",
            flat.insns
        );
    }

    #[test]
    fn flatten_while_loop_has_back_edge() {
        let mut graph = FunctionGraph::new("loop");
        let entry = graph.startblock;
        let header = graph.create_block();
        let body = graph.create_block();
        let exit = graph.create_block();

        graph.set_goto(entry, header, vec![]);
        let cond = graph.push_op(header, OpKind::ConstInt(1), true).unwrap();
        graph.set_branch(header, cond, body, vec![], exit, vec![]);
        graph.set_goto(body, header, vec![]);
        graph.set_return(exit, None);

        let flat = flatten(&graph, &identity_regallocs(8));
        // Body should jump back to header label
        let jumps: Vec<_> = flat
            .insns
            .iter()
            .filter(|op| matches!(op, FlatOp::Jump(_)))
            .collect();
        assert!(
            jumps.len() >= 2,
            "loop should have >=2 jumps (entry→header, body→header)"
        );
    }

    #[test]
    fn flatten_phi_produces_move_ops() {
        // When a Goto carries Link args to a target with inputargs and
        // the target is NOT a final block (so the RPython `flatten.py:148-155`
        // make_return-inline optimization does not fire), flatten must
        // emit a `{kind}_copy` Move op for Phi resolution.
        let mut graph = FunctionGraph::new("phi");
        let entry = graph.startblock;
        let val = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();

        let (target, phi_args) = graph.create_block_with_args(1);
        let phi = phi_args[0];

        // target → returnblock, so target is NOT a final block
        // (`target.exits.is_empty()` is false once a Goto to returnblock
        // is installed).
        graph.set_return(target, Some(phi));

        graph.set_goto(entry, target, vec![val]);

        let flat = flatten(&graph, &identity_regallocs(8));
        let moves: Vec<_> = flat
            .insns
            .iter()
            .filter(|op| matches!(op, FlatOp::Move { .. }))
            .collect();
        assert_eq!(moves.len(), 1, "should have 1 Move for Phi resolution");
    }

    #[test]
    fn flatten_skips_input_ops() {
        let mut graph = FunctionGraph::new("inputs");
        let entry = graph.startblock;
        let input = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "a".into(),
                    ty: crate::model::ValueType::Int,
                },
                true,
            )
            .unwrap();
        let value = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();
        let sum = graph
            .push_op(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: input,
                    rhs: value,
                    result_ty: crate::model::ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(sum));

        let flat = flatten(&graph, &identity_regallocs(8));
        assert!(
            !flat.insns.iter().any(|op| matches!(
                op,
                FlatOp::Op(SpaceOperation {
                    kind: OpKind::Input { .. },
                    ..
                })
            )),
            "flatten must not serialize input ops: {:?}",
            flat.insns
        );
        assert!(
            flat.num_values >= 3,
            "input ValueIds must still contribute to num_values"
        );
    }

    #[test]
    fn flatten_call_with_exception_emits_catch_and_reraise() {
        let mut graph = FunctionGraph::new("canraise");
        let entry = graph.startblock;
        let call_result = graph.push_op(entry, OpKind::ConstInt(7), true).unwrap();
        graph.push_op(entry, OpKind::Live, false);
        let continuation = graph.create_block();
        let phi = graph.alloc_value();
        graph.block_mut(continuation).inputargs.push(phi);
        graph.set_return(continuation, Some(phi));

        let (exc_block, last_exception, last_exc_value) = graph.exceptblock_args();
        graph.set_goto(entry, continuation, vec![call_result]);
        graph.set_control_flow_metadata(
            entry,
            Some(crate::model::ExitSwitch::LastException),
            vec![
                crate::model::Link::new(vec![call_result], continuation, None),
                crate::model::Link::new(
                    vec![last_exception, last_exc_value],
                    exc_block,
                    Some(exception_exitcase()),
                )
                .extravars(
                    Some(LinkArg::from(last_exception)),
                    Some(LinkArg::from(last_exc_value)),
                ),
            ],
        );

        let flat = flatten(&graph, &identity_regallocs(16));
        assert!(
            flat.insns
                .iter()
                .any(|op| matches!(op, FlatOp::CatchException { .. })),
            "canraise block must flatten to catch_exception"
        );
        assert!(
            flat.insns.iter().any(|op| matches!(op, FlatOp::Reraise)),
            "shared exception block should re-raise in flattened form"
        );
    }

    #[test]
    fn flatten_typed_exception_links_emit_mismatch_and_last_exc_value() {
        let mut graph = FunctionGraph::new("typed_canraise");
        let entry = graph.startblock;
        let call_result = graph.push_op(entry, OpKind::ConstInt(7), true).unwrap();
        graph.push_op(entry, OpKind::Live, false);

        let handler = graph.create_block();
        let handler_exc_type = graph.alloc_value();
        let handler_exc_value = graph.alloc_value();
        graph.block_mut(handler).inputargs.push(handler_exc_type);
        graph.block_mut(handler).inputargs.push(handler_exc_value);
        // Upstream invariant: typed catch handlers are not empty blocks.
        // Keep one op in the handler so the bare-reraise collapse remains
        // reserved for the empty exception block shape from flatten.py.
        graph.push_op(handler, OpKind::Live, false);
        graph.set_goto(handler, graph.returnblock, vec![handler_exc_value]);

        let (exc_block, last_exception, last_exc_value) = graph.exceptblock_args();
        let value_error = ConstValue::builtin("ValueError");
        graph.set_goto(entry, graph.returnblock, vec![call_result]);
        graph.set_control_flow_metadata(
            entry,
            Some(crate::model::ExitSwitch::LastException),
            vec![
                crate::model::Link::new(vec![call_result], graph.returnblock, None),
                crate::model::Link::new_mixed(
                    vec![
                        LinkArg::from(value_error.clone()),
                        LinkArg::from(last_exc_value),
                    ],
                    handler,
                    Some(ExitCase::Const(value_error.clone())),
                )
                .with_llexitcase(ConstValue::Int(123))
                .extravars(
                    Some(LinkArg::from(value_error)),
                    Some(LinkArg::from(last_exc_value)),
                ),
                crate::model::Link::new(
                    vec![last_exception, last_exc_value],
                    exc_block,
                    Some(exception_exitcase()),
                )
                .extravars(
                    Some(LinkArg::from(last_exception)),
                    Some(LinkArg::from(last_exc_value)),
                ),
            ],
        );

        let flat = flatten(&graph, &identity_regallocs(16));
        assert!(
            flat.insns.iter().any(|op| matches!(
                op,
                FlatOp::GotoIfExceptionMismatch {
                    llexitcase: ConstValue::Int(123),
                    ..
                }
            )),
            "typed exception link should emit goto_if_exception_mismatch"
        );
        assert!(
            flat.insns
                .iter()
                .any(|op| matches!(op, FlatOp::LastException { dst } if *dst == handler_exc_type)),
            "typed exception link should materialize last_exception at target inputarg"
        );
        // RPython `flatten.py:336-347 generate_last_exc` writes the
        // exception value into the TARGET inputarg's register, not the
        // prevblock-side `link.last_exc_value` Variable.
        assert!(
            flat.insns
                .iter()
                .any(|op| matches!(op, FlatOp::LastExcValue { dst } if *dst == handler_exc_value)),
            "typed exception link should materialize last_exc_value at target inputarg"
        );
    }

    #[test]
    fn flatten_final_exceptblock_emits_live_before_raise() {
        let mut graph = FunctionGraph::new("final_exceptblock");
        let entry = graph.startblock;
        let (exc_block, last_exception, last_exc_value) = graph.exceptblock_args();
        graph.set_goto(entry, exc_block, vec![last_exception, last_exc_value]);

        let flat = flatten(&graph, &identity_regallocs(16));
        let raise_idx = flat
            .insns
            .iter()
            .position(|op| matches!(op, FlatOp::Raise(LinkArg::Value(v)) if *v == last_exc_value))
            .expect("final exceptblock should flatten to raise");
        assert!(
            matches!(
                flat.insns.get(raise_idx.saturating_sub(1)),
                Some(FlatOp::Live { .. })
            ),
            "final exceptblock should emit -live- before raise"
        );
        assert!(
            matches!(flat.insns.get(raise_idx + 1), Some(FlatOp::EndOfBlock)),
            "raise should still terminate with ---"
        );
    }

    #[test]
    fn flatten_final_return_accepts_constant_link_arg() {
        let mut graph = FunctionGraph::new("final_const_return");
        let entry = graph.startblock;
        graph.set_control_flow_metadata(
            entry,
            None,
            vec![Link::new_mixed(
                vec![LinkArg::Const(ConstValue::Int(42))],
                graph.returnblock,
                None,
            )],
        );

        let flat = flatten(&graph, &identity_regallocs(16));
        assert!(
            flat.insns
                .iter()
                .any(|op| matches!(op, FlatOp::IntReturn(LinkArg::Const(ConstValue::Int(42))))),
            "final return should preserve Constant link args"
        );
    }

    #[test]
    fn flatten_int_add_ovf_uses_jump_if_ovf() {
        let mut graph = FunctionGraph::new("add_ovf");
        let entry = graph.startblock;
        let lhs = graph.push_op(entry, OpKind::ConstInt(7), true).unwrap();
        let rhs = graph.push_op(entry, OpKind::ConstInt(2), true).unwrap();
        graph.push_op(entry, OpKind::Live, false);
        let sum = graph
            .push_op(
                entry,
                OpKind::BinOp {
                    op: "add_ovf".into(),
                    lhs,
                    rhs,
                    result_ty: crate::model::ValueType::Int,
                },
                true,
            )
            .unwrap();

        let handler = graph.create_block();
        let handler_exc_type = graph.alloc_value();
        let handler_exc_value = graph.alloc_value();
        graph.block_mut(handler).inputargs.push(handler_exc_type);
        graph.block_mut(handler).inputargs.push(handler_exc_value);
        let forty_two = graph.push_op(handler, OpKind::ConstInt(42), true).unwrap();
        graph.set_return(handler, Some(forty_two));

        let (_, _, last_exc_value) = graph.exceptblock_args();
        let overflow_error = ConstValue::builtin("OverflowError");
        graph.set_control_flow_metadata(
            entry,
            Some(crate::model::ExitSwitch::LastException),
            vec![
                crate::model::Link::new(vec![sum], graph.returnblock, None),
                crate::model::Link::new_mixed(
                    vec![
                        LinkArg::from(overflow_error.clone()),
                        LinkArg::from(last_exc_value),
                    ],
                    handler,
                    Some(ExitCase::Const(overflow_error.clone())),
                )
                .extravars(
                    Some(LinkArg::from(overflow_error)),
                    Some(LinkArg::from(last_exc_value)),
                ),
            ],
        );

        let flat = flatten(&graph, &identity_regallocs(16));
        assert!(
            flat.insns.iter().any(|op| matches!(
                op,
                FlatOp::IntBinOpJumpIfOvf {
                    op: IntOvfOp::Add,
                    lhs: l,
                    rhs: r,
                    dst,
                    ..
                } if *l == lhs && *r == rhs && *dst == sum
            )),
            "ovf arithmetic should flatten to int_add_jump_if_ovf"
        );
        assert!(
            !flat
                .insns
                .iter()
                .any(|op| matches!(op, FlatOp::CatchException { .. })),
            "ovf-specialized path should bypass generic catch_exception lowering"
        );
    }

    #[test]
    fn flatten_ovf_reraise_emits_constant_raise() {
        let mut graph = FunctionGraph::new("ovf_reraise");
        let entry = graph.startblock;
        let lhs = graph.push_op(entry, OpKind::ConstInt(7), true).unwrap();
        let rhs = graph.push_op(entry, OpKind::ConstInt(2), true).unwrap();
        graph.push_op(entry, OpKind::Live, false);
        let sum = graph
            .push_op(
                entry,
                OpKind::BinOp {
                    op: "add_ovf".into(),
                    lhs,
                    rhs,
                    result_ty: crate::model::ValueType::Int,
                },
                true,
            )
            .unwrap();

        let (exc_block, _, last_exc_value) = graph.exceptblock_args();
        let overflow_error = ConstValue::builtin("OverflowError");
        graph.set_control_flow_metadata(
            entry,
            Some(crate::model::ExitSwitch::LastException),
            vec![
                crate::model::Link::new(vec![sum], graph.returnblock, None),
                crate::model::Link::new_mixed(
                    vec![
                        LinkArg::from(overflow_error.clone()),
                        LinkArg::from(last_exc_value),
                    ],
                    exc_block,
                    Some(ExitCase::Const(overflow_error.clone())),
                )
                .extravars(
                    Some(LinkArg::from(overflow_error)),
                    Some(LinkArg::from(last_exc_value)),
                ),
            ],
        );

        let flat = flatten(&graph, &identity_regallocs(16));
        let standard_overflow = crate::flowspace::model::HOST_ENV
            .lookup_standard_exception_instance("OverflowError")
            .expect("missing standard OverflowError instance");
        assert!(
            flat.insns.iter().any(|op| matches!(
                op,
                FlatOp::Raise(LinkArg::Const(ConstValue::HostObject(obj))) if *obj == standard_overflow
            )),
            "overflow direct reraises should emit raise Constant(OverflowError-instance)"
        );
        assert!(
            !flat.insns.iter().any(|op| matches!(op, FlatOp::Reraise)),
            "overflow direct reraises should not use generic reraise"
        );
    }

    // `rpython/jit/codewriter/test/test_flatten.py:115-128` `test_reorder_renaming_list`.
    #[test]
    fn reorder_renaming_list_empty() {
        let result: Vec<(Option<i32>, Option<i32>)> = reorder_renaming_list::<i32>(&[], &[]);
        assert_eq!(result, Vec::<(Option<i32>, Option<i32>)>::new());
    }

    #[test]
    fn reorder_renaming_list_all_independent() {
        // No overlap between frm and to → identity order.
        let result = reorder_renaming_list(&[1, 2, 3], &[4, 5, 6]);
        assert_eq!(
            result,
            vec![(Some(1), Some(4)), (Some(2), Some(5)), (Some(3), Some(6)),]
        );
    }

    #[test]
    fn reorder_renaming_list_chain() {
        // 4→1, 5→2, 1→3, 2→4. Safe order: do (1→3) and (2→4) first
        // (their destinations aren't read later), then (4→1) and
        // (5→2). RPython expected: [(1,3), (4,1), (2,4), (5,2)].
        let result = reorder_renaming_list(&[4, 5, 1, 2], &[1, 2, 3, 4]);
        assert_eq!(
            result,
            vec![
                (Some(1), Some(3)),
                (Some(4), Some(1)),
                (Some(2), Some(4)),
                (Some(5), Some(2)),
            ]
        );
    }

    #[test]
    fn reorder_renaming_list_swap_cycle() {
        // 1↔2 is a cycle of length 2. Save 1 with push, do 2→1,
        // then pop→2. RPython expected: [(1,None), (2,1), (None,2)].
        let result = reorder_renaming_list(&[1, 2], &[2, 1]);
        assert_eq!(
            result,
            vec![(Some(1), None), (Some(2), Some(1)), (None, Some(2))]
        );
    }

    #[test]
    fn reorder_renaming_list_long_chain_and_two_cycles() {
        // Chain + two independent cycles: (7→8) safe;
        // (4→1, 3→2, 1→3, 2→4) is a 4-cycle; (6→5, 5→6) is a 2-cycle.
        let result = reorder_renaming_list(&[4, 3, 6, 1, 2, 5, 7], &[1, 2, 5, 3, 4, 6, 8]);
        assert_eq!(
            result,
            vec![
                (Some(7), Some(8)),
                (Some(4), None),
                (Some(2), Some(4)),
                (Some(3), Some(2)),
                (Some(1), Some(3)),
                (None, Some(1)),
                (Some(6), None),
                (Some(5), Some(6)),
                (None, Some(5)),
            ]
        );
    }

    // `rpython/jit/codewriter/test/test_flatten.py` exercises
    // `insert_renamings` indirectly via whole-graph tests; majit covers
    // the standalone helper below.  Each case constructs a minimal
    // `Link` with no `extravars` so the `flatten.py:310-311` exception
    // filter doesn't fire; identity coloring (`ValueId(n) → color n`)
    // keeps the cycle-break reasoning at ValueId level for readability.
    fn plain_link(args: &[ValueId]) -> Link {
        Link::new(args.to_vec(), BlockId(0), None)
    }

    #[test]
    fn insert_renamings_emits_nothing_for_identity() {
        // `for i, v in enumerate(link.args): if v == w: continue`.
        let regallocs = identity_regallocs(8);
        let mut ops: Vec<FlatOp> = Vec::new();
        let args = [ValueId(0), ValueId(1), ValueId(2)];
        let link = plain_link(&args);
        insert_renamings(&link, &args, &regallocs, &mut ops);
        assert_eq!(ops, Vec::<FlatOp>::new());
    }

    #[test]
    fn insert_renamings_emits_move_for_acyclic_rename() {
        // Simple `%0 -> %1` phi resolution.
        let regallocs = identity_regallocs(8);
        let mut ops: Vec<FlatOp> = Vec::new();
        let link = plain_link(&[ValueId(0)]);
        insert_renamings(&link, &[ValueId(1)], &regallocs, &mut ops);
        assert_eq!(
            ops,
            vec![FlatOp::Move {
                dst: ValueId(1),
                src: LinkArg::Value(ValueId(0)),
            }]
        );
    }

    #[test]
    fn insert_renamings_emits_move_for_constant_source() {
        let regallocs = identity_regallocs(8);
        let mut ops: Vec<FlatOp> = Vec::new();
        let link = Link::new_mixed(vec![LinkArg::Const(ConstValue::Int(7))], BlockId(0), None);
        insert_renamings(&link, &[ValueId(1)], &regallocs, &mut ops);
        assert_eq!(
            ops,
            vec![FlatOp::Move {
                dst: ValueId(1),
                src: LinkArg::Const(ConstValue::Int(7)),
            }]
        );
    }

    #[test]
    fn insert_renamings_breaks_swap_cycle_with_push_pop() {
        // Swap `%0 <-> %1` — the raw `(src,dst)` pairs are `(0,1)` and
        // `(1,0)`.  `flatten.py:312` `lst.sort(key=lambda (v,w): w.index)`
        // reorders by dst color → `(1,0)` comes first, then `(0,1)`.
        // `reorder_renaming_list([1,0], [0,1])` then picks the first
        // pending entry for the cycle break → push ValueId(1);
        // copy ValueId(0)->ValueId(1); pop ValueId(0).
        let regallocs = identity_regallocs(8);
        let mut ops: Vec<FlatOp> = Vec::new();
        let link = plain_link(&[ValueId(0), ValueId(1)]);
        insert_renamings(&link, &[ValueId(1), ValueId(0)], &regallocs, &mut ops);
        assert_eq!(
            ops,
            vec![
                FlatOp::Push(ValueId(1)),
                FlatOp::Move {
                    dst: ValueId(1),
                    src: LinkArg::Value(ValueId(0)),
                },
                FlatOp::Pop(ValueId(0)),
            ]
        );
    }

    /// Two ValueIds that regalloc coalesced to the same color must NOT
    /// emit a Move — upstream `flatten.py:314` `if v == w: continue`
    /// tests color identity, not ValueId identity. This is the key
    /// difference between the pre- and post-regalloc insert_renamings.
    #[test]
    fn insert_renamings_skips_coalesced_same_color() {
        // ValueId(0) and ValueId(1) both colored 7 — the rename is a
        // no-op even though the ValueIds differ.
        let mut coloring: std::collections::HashMap<ValueId, usize> =
            std::collections::HashMap::new();
        coloring.insert(ValueId(0), 7);
        coloring.insert(ValueId(1), 7);
        let mut regallocs = std::collections::HashMap::new();
        regallocs.insert(
            RegKind::Int,
            crate::regalloc::RegAllocResult {
                coloring,
                num_regs: 8,
            },
        );
        let mut ops: Vec<FlatOp> = Vec::new();
        let link = plain_link(&[ValueId(0)]);
        insert_renamings(&link, &[ValueId(1)], &regallocs, &mut ops);
        assert_eq!(ops, Vec::<FlatOp>::new());
    }

    /// Four distinct ValueIds colored so that the color-level rename is
    /// a 2-cycle swap. Pre-regalloc cycle-break (ValueId identity) would
    /// not see any cycle — two unrelated moves — and emit only Moves,
    /// corrupting the color bank. Post-regalloc cycle-break (colors)
    /// emits Push/Move/Pop just like the pure 2-cycle test above.
    #[test]
    fn insert_renamings_detects_cycle_at_color_level() {
        // Coloring:
        //   v0 → c0, v2 → c1 (link.args)
        //   v1 → c1, v3 → c0 (target.inputargs)
        // Color pairs: (c0 → c1), (c1 → c0) — a swap cycle on colors.
        let mut coloring: std::collections::HashMap<ValueId, usize> =
            std::collections::HashMap::new();
        coloring.insert(ValueId(0), 0);
        coloring.insert(ValueId(1), 1);
        coloring.insert(ValueId(2), 1);
        coloring.insert(ValueId(3), 0);
        let mut regallocs = std::collections::HashMap::new();
        regallocs.insert(
            RegKind::Int,
            crate::regalloc::RegAllocResult {
                coloring,
                num_regs: 2,
            },
        );
        let mut ops: Vec<FlatOp> = Vec::new();
        let link = plain_link(&[ValueId(0), ValueId(2)]);
        insert_renamings(&link, &[ValueId(1), ValueId(3)], &regallocs, &mut ops);
        // Must emit push/copy/pop — NOT two naive Moves.
        assert!(
            ops.iter().any(|o| matches!(o, FlatOp::Push(_))),
            "color-level 2-cycle must emit a Push, got {:?}",
            ops
        );
        assert!(
            ops.iter().any(|o| matches!(o, FlatOp::Pop(_))),
            "color-level 2-cycle must emit a Pop, got {:?}",
            ops
        );
    }

    /// `flatten.py:312` `lst.sort(key=lambda (v,w): w.index)` + per-kind
    /// grouping emits each kind's renamings contiguously, sorted by dst
    /// color within the group.  With mixed Int + Ref link args, the
    /// output shows all Int-bank Moves before the Ref-bank Move even
    /// when the link args interleave the kinds.
    #[test]
    fn insert_renamings_groups_by_kind_and_sorts_by_dst() {
        let mut int_coloring: std::collections::HashMap<ValueId, usize> =
            std::collections::HashMap::new();
        int_coloring.insert(ValueId(0), 0);
        int_coloring.insert(ValueId(1), 3);
        int_coloring.insert(ValueId(2), 1);
        int_coloring.insert(ValueId(3), 2);
        let mut ref_coloring: std::collections::HashMap<ValueId, usize> =
            std::collections::HashMap::new();
        ref_coloring.insert(ValueId(10), 0);
        ref_coloring.insert(ValueId(11), 5);
        let mut regallocs = std::collections::HashMap::new();
        regallocs.insert(
            RegKind::Int,
            crate::regalloc::RegAllocResult {
                coloring: int_coloring,
                num_regs: 4,
            },
        );
        regallocs.insert(
            RegKind::Ref,
            crate::regalloc::RegAllocResult {
                coloring: ref_coloring,
                num_regs: 6,
            },
        );
        // Link args interleave Ref (10->11) with two Int renamings, one
        // going to a higher dst color (v0->v1 i.e. 0->3) and one going
        // to a lower dst color (v2->v3 i.e. 1->2).  Upstream sorts by
        // dst color (Int 2 before Int 3) and emits kinds in
        // `['int', 'ref', 'float']` order:
        //
        //   Move v2 -> v3  (Int, dst color 2)
        //   Move v0 -> v1  (Int, dst color 3)
        //   Move v10 -> v11 (Ref, dst color 5)
        let mut ops: Vec<FlatOp> = Vec::new();
        let link = plain_link(&[ValueId(0), ValueId(10), ValueId(2)]);
        insert_renamings(
            &link,
            &[ValueId(1), ValueId(11), ValueId(3)],
            &regallocs,
            &mut ops,
        );
        assert_eq!(
            ops,
            vec![
                // Int group, dst color 2 (v2 -> v3)
                FlatOp::Move {
                    dst: ValueId(3),
                    src: LinkArg::Value(ValueId(2)),
                },
                // Int group, dst color 3 (v0 -> v1)
                FlatOp::Move {
                    dst: ValueId(1),
                    src: LinkArg::Value(ValueId(0)),
                },
                // Ref group, dst color 5 (v10 -> v11)
                FlatOp::Move {
                    dst: ValueId(11),
                    src: LinkArg::Value(ValueId(10)),
                },
            ]
        );
    }
}
