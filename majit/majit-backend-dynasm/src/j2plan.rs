//! Experimental j2-style planning layer for the dynasm backend.
//!
//! It gives the existing dynasm backend a small, architecture-neutral
//! lowering target plus reverse liveness information. Regalloc dispatches
//! the lowered operations first; legacy opcode dispatch remains only as a
//! guard rail for a missing plan entry.
#![allow(dead_code)]

use std::fmt;

use majit_ir::{InputArg, Op, OpCode, OpRef};

/// A lowered dynasm-backend operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LirOp {
    Label {
        args: Vec<OpRef>,
    },
    Jump {
        args: Vec<OpRef>,
    },
    Finish {
        args: Vec<OpRef>,
    },
    IntBin {
        kind: IntBinKind,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
    },
    IntUnary {
        kind: IntUnaryKind,
        dst: OpRef,
        arg: OpRef,
    },
    IntCmp {
        kind: IntCmpKind,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
    },
    Guard {
        kind: GuardKind,
        args: Vec<OpRef>,
        fail_args: Vec<OpRef>,
    },
    Load {
        kind: LoadKind,
        dst: OpRef,
        base: OpRef,
        offset: Option<OpRef>,
        index: Option<OpRef>,
        scale: Option<OpRef>,
        size: Option<OpRef>,
    },
    Store {
        kind: StoreKind,
        base: OpRef,
        offset: Option<OpRef>,
        index: Option<OpRef>,
        scale: Option<OpRef>,
        value: OpRef,
        size: Option<OpRef>,
    },
    Call {
        opcode: OpCode,
        dst: Option<OpRef>,
        args: Vec<OpRef>,
    },
    Opcode {
        opcode: OpCode,
        dst: Option<OpRef>,
        args: Vec<OpRef>,
        fail_args: Vec<OpRef>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntBinKind {
    Add,
    AddOvf,
    Sub,
    SubOvf,
    Mul,
    MulOvf,
    And,
    Or,
    Xor,
    LShift,
    RShift,
    URShift,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntUnaryKind {
    Neg,
    Invert,
    IsTrue,
    IsZero,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntCmpKind {
    SignedLt,
    SignedLe,
    SignedGt,
    SignedGe,
    Eq,
    Ne,
    UnsignedLt,
    UnsignedLe,
    UnsignedGt,
    UnsignedGe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GuardKind {
    True,
    False,
    Value,
    Class,
    GcType,
    IsObject,
    Subclass,
    NonNull,
    IsNull,
    NonNullClass,
    NoException,
    Exception,
    NoOverflow,
    Overflow,
    NotInvalidated,
    FutureCondition,
    NotForced,
    AlwaysFails,
    Other(OpCode),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoadKind {
    Gc,
    GcIndexed,
    Raw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StoreKind {
    Gc,
    GcIndexed,
    Raw,
}

/// Per-op reverse liveness computed from the lowered plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LivePoint {
    /// Operation index in the original trace.
    pub op_index: usize,
    /// Values that must be available before this operation executes.
    pub live_in: Vec<OpRef>,
}

/// A j2-style plan for a single trace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TracePlan {
    pub inputargs: Vec<OpRef>,
    pub ops: Vec<LirOp>,
    pub live_points: Vec<LivePoint>,
    pub deopt_spill_points: Vec<DeoptSpillPoint>,
    pub max_live: usize,
    pub lowered_ops: usize,
    pub fallback_ops: usize,
}

/// Guard fail args that are only needed on the deopt path at this point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeoptSpillPoint {
    pub op_index: usize,
    pub args: Vec<OpRef>,
}

impl TracePlan {
    pub(crate) fn build(inputargs: &[InputArg], ops: &[Op]) -> Self {
        let lowered: Vec<LirOp> = ops.iter().map(lower_op).collect();
        let live_points = compute_live_points(&lowered);
        let deopt_spill_points = compute_deopt_spill_points(&lowered);
        let max_live = live_points
            .iter()
            .map(|point| point.live_in.len())
            .max()
            .unwrap_or(0);
        let fallback_ops = 0;

        TracePlan {
            inputargs: inputargs.iter().map(|arg| arg.opref()).collect::<Vec<_>>(),
            lowered_ops: lowered.len() - fallback_ops,
            fallback_ops,
            ops: lowered,
            live_points,
            deopt_spill_points,
            max_live,
        }
    }

    pub(crate) fn summary(&self) -> TracePlanSummary<'_> {
        TracePlanSummary(self)
    }

    pub(crate) fn deopt_spill_args_by_index(&self, len: usize) -> Vec<Vec<OpRef>> {
        let mut by_index = vec![Vec::new(); len];
        for point in &self.deopt_spill_points {
            if point.op_index < by_index.len() {
                by_index[point.op_index] = point.args.clone();
            }
        }
        by_index
    }
}

pub(crate) struct TracePlanSummary<'a>(&'a TracePlan);

impl fmt::Display for TracePlanSummary<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let plan = self.0;
        write!(
            f,
            "ops={} lowered={} fallback={} max_live={} deopt_spills={}",
            plan.ops.len(),
            plan.lowered_ops,
            plan.fallback_ops,
            plan.max_live,
            plan.deopt_spill_points.len()
        )
    }
}

fn lower_op(op: &Op) -> LirOp {
    match op.opcode {
        OpCode::Label => LirOp::Label {
            args: op.args.to_vec(),
        },
        OpCode::Jump => LirOp::Jump {
            args: op.args.to_vec(),
        },
        OpCode::Finish => LirOp::Finish {
            args: op.args.to_vec(),
        },
        OpCode::IntAdd
        | OpCode::IntAddOvf
        | OpCode::IntSub
        | OpCode::IntSubOvf
        | OpCode::IntMul
        | OpCode::IntMulOvf
        | OpCode::IntAnd
        | OpCode::IntOr
        | OpCode::IntXor
        | OpCode::IntLshift
        | OpCode::IntRshift
        | OpCode::UintRshift
            if op.args.len() >= 2 =>
        {
            LirOp::IntBin {
                kind: int_bin_kind(op.opcode),
                dst: op.pos,
                lhs: op.args[0],
                rhs: op.args[1],
            }
        }
        OpCode::IntNeg | OpCode::IntInvert | OpCode::IntIsTrue | OpCode::IntIsZero
            if !op.args.is_empty() =>
        {
            LirOp::IntUnary {
                kind: int_unary_kind(op.opcode),
                dst: op.pos,
                arg: op.args[0],
            }
        }
        OpCode::IntLt
        | OpCode::IntLe
        | OpCode::IntGt
        | OpCode::IntGe
        | OpCode::IntEq
        | OpCode::IntNe
        | OpCode::UintLt
        | OpCode::UintLe
        | OpCode::UintGt
        | OpCode::UintGe
        | OpCode::PtrEq
        | OpCode::PtrNe
        | OpCode::InstancePtrEq
        | OpCode::InstancePtrNe
            if op.args.len() >= 2 =>
        {
            LirOp::IntCmp {
                kind: int_cmp_kind(op.opcode),
                dst: op.pos,
                lhs: op.args[0],
                rhs: op.args[1],
            }
        }
        OpCode::GcLoadI | OpCode::GcLoadR | OpCode::GcLoadF if op.args.len() >= 2 => LirOp::Load {
            kind: LoadKind::Gc,
            dst: op.pos,
            base: op.args[0],
            offset: Some(op.args[1]),
            index: None,
            scale: None,
            size: op.args.get(2).copied(),
        },
        OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF
            if op.args.len() >= 5 =>
        {
            LirOp::Load {
                kind: LoadKind::GcIndexed,
                dst: op.pos,
                base: op.args[0],
                offset: Some(op.args[3]),
                index: Some(op.args[1]),
                scale: Some(op.args[2]),
                size: Some(op.args[4]),
            }
        }
        OpCode::RawLoadI | OpCode::RawLoadF if op.args.len() >= 2 => LirOp::Load {
            kind: LoadKind::Raw,
            dst: op.pos,
            base: op.args[0],
            offset: Some(op.args[1]),
            index: None,
            scale: None,
            size: None,
        },
        OpCode::GcStore if op.args.len() >= 4 => LirOp::Store {
            kind: StoreKind::Gc,
            base: op.args[0],
            offset: Some(op.args[1]),
            index: None,
            scale: None,
            value: op.args[2],
            size: Some(op.args[3]),
        },
        OpCode::GcStoreIndexed if op.args.len() >= 6 => LirOp::Store {
            kind: StoreKind::GcIndexed,
            base: op.args[0],
            offset: Some(op.args[4]),
            index: Some(op.args[1]),
            scale: Some(op.args[3]),
            value: op.args[2],
            size: Some(op.args[5]),
        },
        OpCode::RawStore if op.args.len() >= 3 => LirOp::Store {
            kind: StoreKind::Raw,
            base: op.args[0],
            offset: Some(op.args[1]),
            index: None,
            scale: None,
            value: op.args[2],
            size: None,
        },
        opcode if opcode.is_guard() => LirOp::Guard {
            kind: guard_kind(opcode),
            args: op.args.to_vec(),
            fail_args: op.fail_args.as_deref().unwrap_or(&[]).to_vec(),
        },
        opcode if opcode.is_call() => LirOp::Call {
            opcode,
            dst: result_ref(op),
            args: op.args.to_vec(),
        },
        opcode => LirOp::Opcode {
            opcode,
            dst: result_ref(op),
            args: op.args.to_vec(),
            fail_args: op.fail_args.as_deref().unwrap_or(&[]).to_vec(),
        },
    }
}

fn compute_live_points(ops: &[LirOp]) -> Vec<LivePoint> {
    let mut live = Vec::new();
    let mut points = Vec::with_capacity(ops.len());

    for (op_index, op) in ops.iter().enumerate().rev() {
        if let Some(dst) = op.def() {
            remove_ref(&mut live, dst);
        }
        op.add_fail_uses(&mut live);
        op.add_uses(&mut live);
        points.push(LivePoint {
            op_index,
            live_in: live.clone(),
        });
    }

    points.reverse();
    points
}

fn compute_deopt_spill_points(ops: &[LirOp]) -> Vec<DeoptSpillPoint> {
    let mut fast_live_after = Vec::new();
    let mut points = Vec::new();

    for (op_index, op) in ops.iter().enumerate().rev() {
        if let LirOp::Guard { fail_args, .. } = op {
            let mut args = Vec::new();
            for &arg in fail_args {
                if !fast_live_after.contains(&arg) {
                    add_ref(&mut args, arg);
                }
            }
            if !args.is_empty() {
                points.push(DeoptSpillPoint { op_index, args });
            }
        }

        if let Some(dst) = op.def() {
            remove_ref(&mut fast_live_after, dst);
        }
        op.add_uses(&mut fast_live_after);
    }

    points.reverse();
    points
}

fn int_bin_kind(opcode: OpCode) -> IntBinKind {
    match opcode {
        OpCode::IntAdd => IntBinKind::Add,
        OpCode::IntAddOvf => IntBinKind::AddOvf,
        OpCode::IntSub => IntBinKind::Sub,
        OpCode::IntSubOvf => IntBinKind::SubOvf,
        OpCode::IntMul => IntBinKind::Mul,
        OpCode::IntMulOvf => IntBinKind::MulOvf,
        OpCode::IntAnd => IntBinKind::And,
        OpCode::IntOr => IntBinKind::Or,
        OpCode::IntXor => IntBinKind::Xor,
        OpCode::IntLshift => IntBinKind::LShift,
        OpCode::IntRshift => IntBinKind::RShift,
        OpCode::UintRshift => IntBinKind::URShift,
        _ => unreachable!("not an integer binary opcode: {:?}", opcode),
    }
}

fn int_unary_kind(opcode: OpCode) -> IntUnaryKind {
    match opcode {
        OpCode::IntNeg => IntUnaryKind::Neg,
        OpCode::IntInvert => IntUnaryKind::Invert,
        OpCode::IntIsTrue => IntUnaryKind::IsTrue,
        OpCode::IntIsZero => IntUnaryKind::IsZero,
        _ => unreachable!("not an integer unary opcode: {:?}", opcode),
    }
}

fn int_cmp_kind(opcode: OpCode) -> IntCmpKind {
    match opcode {
        OpCode::IntLt => IntCmpKind::SignedLt,
        OpCode::IntLe => IntCmpKind::SignedLe,
        OpCode::IntGt => IntCmpKind::SignedGt,
        OpCode::IntGe => IntCmpKind::SignedGe,
        OpCode::IntEq | OpCode::PtrEq | OpCode::InstancePtrEq => IntCmpKind::Eq,
        OpCode::IntNe | OpCode::PtrNe | OpCode::InstancePtrNe => IntCmpKind::Ne,
        OpCode::UintLt => IntCmpKind::UnsignedLt,
        OpCode::UintLe => IntCmpKind::UnsignedLe,
        OpCode::UintGt => IntCmpKind::UnsignedGt,
        OpCode::UintGe => IntCmpKind::UnsignedGe,
        _ => unreachable!("not an integer comparison opcode: {:?}", opcode),
    }
}

fn guard_kind(opcode: OpCode) -> GuardKind {
    match opcode {
        OpCode::GuardTrue | OpCode::VecGuardTrue => GuardKind::True,
        OpCode::GuardFalse | OpCode::VecGuardFalse => GuardKind::False,
        OpCode::GuardValue => GuardKind::Value,
        OpCode::GuardClass => GuardKind::Class,
        OpCode::GuardGcType => GuardKind::GcType,
        OpCode::GuardIsObject => GuardKind::IsObject,
        OpCode::GuardSubclass => GuardKind::Subclass,
        OpCode::GuardNonnull => GuardKind::NonNull,
        OpCode::GuardIsnull => GuardKind::IsNull,
        OpCode::GuardNonnullClass => GuardKind::NonNullClass,
        OpCode::GuardNoException => GuardKind::NoException,
        OpCode::GuardException => GuardKind::Exception,
        OpCode::GuardNoOverflow => GuardKind::NoOverflow,
        OpCode::GuardOverflow => GuardKind::Overflow,
        OpCode::GuardNotInvalidated => GuardKind::NotInvalidated,
        OpCode::GuardFutureCondition => GuardKind::FutureCondition,
        OpCode::GuardNotForced | OpCode::GuardNotForced2 => GuardKind::NotForced,
        OpCode::GuardAlwaysFails => GuardKind::AlwaysFails,
        _ => GuardKind::Other(opcode),
    }
}

fn result_ref(op: &Op) -> Option<OpRef> {
    if op.pos.is_none() || op.opcode.result_type().is_void() {
        None
    } else {
        Some(op.pos)
    }
}

fn filtered_refs(args: &[OpRef]) -> Vec<OpRef> {
    args.iter()
        .copied()
        .filter(|arg| !arg.is_none() && !arg.is_constant())
        .collect()
}

fn add_ref(live: &mut Vec<OpRef>, opref: OpRef) {
    if opref.is_none() || opref.is_constant() {
        return;
    }
    if !live.contains(&opref) {
        live.push(opref);
    }
}

fn remove_ref(live: &mut Vec<OpRef>, opref: OpRef) {
    if let Some(pos) = live.iter().position(|candidate| *candidate == opref) {
        live.swap_remove(pos);
    }
}

trait TypeIsVoid {
    fn is_void(self) -> bool;
}

impl TypeIsVoid for majit_ir::Type {
    fn is_void(self) -> bool {
        matches!(self, majit_ir::Type::Void)
    }
}

impl LirOp {
    fn def(&self) -> Option<OpRef> {
        match self {
            LirOp::IntBin { dst, .. }
            | LirOp::IntUnary { dst, .. }
            | LirOp::IntCmp { dst, .. }
            | LirOp::Load { dst, .. } => Some(*dst),
            LirOp::Call { dst, .. } | LirOp::Opcode { dst, .. } => *dst,
            LirOp::Label { .. }
            | LirOp::Jump { .. }
            | LirOp::Finish { .. }
            | LirOp::Guard { .. }
            | LirOp::Store { .. } => None,
        }
    }

    fn add_uses(&self, live: &mut Vec<OpRef>) {
        match self {
            LirOp::Label { args } | LirOp::Jump { args } | LirOp::Finish { args } => {
                add_refs(live, args);
            }
            LirOp::IntBin { lhs, rhs, .. } | LirOp::IntCmp { lhs, rhs, .. } => {
                add_ref(live, *lhs);
                add_ref(live, *rhs);
            }
            LirOp::IntUnary { arg, .. } => add_ref(live, *arg),
            LirOp::Guard { args, .. } => add_refs(live, args),
            LirOp::Load {
                base,
                offset,
                index,
                scale,
                size,
                ..
            } => {
                add_ref(live, *base);
                if let Some(offset) = offset {
                    add_ref(live, *offset);
                }
                if let Some(index) = index {
                    add_ref(live, *index);
                }
                if let Some(scale) = scale {
                    add_ref(live, *scale);
                }
                if let Some(size) = size {
                    add_ref(live, *size);
                }
            }
            LirOp::Store {
                base,
                offset,
                index,
                scale,
                value,
                size,
                ..
            } => {
                add_ref(live, *base);
                if let Some(offset) = offset {
                    add_ref(live, *offset);
                }
                if let Some(index) = index {
                    add_ref(live, *index);
                }
                if let Some(scale) = scale {
                    add_ref(live, *scale);
                }
                add_ref(live, *value);
                if let Some(size) = size {
                    add_ref(live, *size);
                }
            }
            LirOp::Call { args, .. } | LirOp::Opcode { args, .. } => {
                add_refs(live, args);
            }
        }
    }

    fn add_fail_uses(&self, live: &mut Vec<OpRef>) {
        match self {
            LirOp::Guard { fail_args, .. } | LirOp::Opcode { fail_args, .. } => {
                add_refs(live, fail_args);
            }
            _ => {}
        }
    }
}

fn add_refs(live: &mut Vec<OpRef>, args: &[OpRef]) {
    for &arg in args {
        add_ref(live, arg);
    }
}

#[cfg(test)]
mod tests {
    use majit_ir::{InputArg, Op, OpCode, OpRef, Type};

    use super::{GuardKind, IntBinKind, IntCmpKind, LirOp, TracePlan};

    #[test]
    fn lowers_simple_integer_loop_shape() {
        let i0 = OpRef(0);
        let c1 = OpRef::from_const(1);
        let c10 = OpRef::from_const(10);

        let mut label = Op::new(OpCode::Label, &[i0]);
        label.pos = OpRef(10);

        let mut add = Op::new(OpCode::IntAdd, &[i0, c1]);
        add.pos = OpRef(1);

        let mut lt = Op::new(OpCode::IntLt, &[OpRef(1), c10]);
        lt.pos = OpRef(2);

        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(2)]);
        guard.pos = OpRef(3);
        guard.fail_args = Some(vec![OpRef(1)].into());

        let mut jump = Op::new(OpCode::Jump, &[OpRef(1)]);
        jump.pos = OpRef(4);

        let plan = TracePlan::build(
            &[InputArg {
                tp: Type::Int,
                index: 0,
            }],
            &[label, add, lt, guard, jump],
        );

        assert_eq!(plan.fallback_ops, 0);
        assert!(plan.deopt_spill_points.is_empty());
        assert!(matches!(
            plan.ops[1],
            LirOp::IntBin {
                kind: IntBinKind::Add,
                ..
            }
        ));
        assert!(matches!(
            plan.ops[2],
            LirOp::IntCmp {
                kind: IntCmpKind::SignedLt,
                ..
            }
        ));
        assert!(matches!(
            plan.ops[3],
            LirOp::Guard {
                kind: GuardKind::True,
                ..
            }
        ));
    }

    #[test]
    fn reverse_liveness_keeps_guard_fail_args_available() {
        let i0 = OpRef(0);
        let c1 = OpRef::from_const(1);

        let mut add = Op::new(OpCode::IntAdd, &[i0, c1]);
        add.pos = OpRef(1);

        let mut is_true = Op::new(OpCode::IntIsTrue, &[OpRef(1)]);
        is_true.pos = OpRef(2);

        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(2)]);
        guard.pos = OpRef(3);
        guard.fail_args = Some(vec![OpRef(1)].into());

        let mut finish = Op::new(OpCode::Finish, &[]);
        finish.pos = OpRef(4);

        let plan = TracePlan::build(
            &[InputArg {
                tp: Type::Int,
                index: 0,
            }],
            &[add, is_true, guard, finish],
        );

        let guard_live = &plan.live_points[2].live_in;
        assert!(guard_live.contains(&OpRef(1)));
        assert!(guard_live.contains(&OpRef(2)));

        assert_eq!(
            plan.deopt_spill_points,
            vec![super::DeoptSpillPoint {
                op_index: 2,
                args: vec![OpRef(1)]
            }]
        );

        let add_live = &plan.live_points[0].live_in;
        assert!(add_live.contains(&i0));
        assert!(!add_live.contains(&c1));
    }

    #[test]
    fn deopt_spill_point_keeps_jump_args_on_fast_path() {
        let i0 = OpRef(0);
        let c1 = OpRef::from_const(1);

        let mut add = Op::new(OpCode::IntAdd, &[i0, c1]);
        add.pos = OpRef(1);

        let mut is_true = Op::new(OpCode::IntIsTrue, &[OpRef(1)]);
        is_true.pos = OpRef(2);

        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(2)]);
        guard.pos = OpRef(3);
        guard.fail_args = Some(vec![OpRef(1)].into());

        let mut jump = Op::new(OpCode::Jump, &[OpRef(1)]);
        jump.pos = OpRef(4);

        let plan = TracePlan::build(
            &[InputArg {
                tp: Type::Int,
                index: 0,
            }],
            &[add, is_true, guard, jump],
        );

        assert!(plan.deopt_spill_points.is_empty());
    }

    #[test]
    fn lowers_indexed_memory_operands_by_role() {
        let base = OpRef(0);
        let index = OpRef(1);
        let value = OpRef(2);
        let scale = OpRef::from_const(1);
        let offset = OpRef::from_const(16);
        let size = OpRef::from_const(8);

        let mut load = Op::new(OpCode::GcLoadIndexedI, &[base, index, scale, offset, size]);
        load.pos = OpRef(3);
        let store = Op::new(
            OpCode::GcStoreIndexed,
            &[base, index, value, scale, offset, size],
        );

        let plan = TracePlan::build(
            &[
                InputArg {
                    tp: Type::Ref,
                    index: base.0,
                },
                InputArg {
                    tp: Type::Int,
                    index: index.0,
                },
                InputArg {
                    tp: Type::Int,
                    index: value.0,
                },
            ],
            &[load, store],
        );

        assert!(matches!(
            plan.ops[0],
            LirOp::Load {
                base: b,
                index: Some(i),
                scale: Some(s),
                offset: Some(o),
                size: Some(z),
                ..
            } if b == base && i == index && s == scale && o == offset && z == size
        ));
        assert!(matches!(
            plan.ops[1],
            LirOp::Store {
                base: b,
                index: Some(i),
                scale: Some(s),
                offset: Some(o),
                value: v,
                size: Some(z),
                ..
            } if b == base && i == index && s == scale && o == offset && v == value && z == size
        ));
    }

    #[test]
    fn lowers_misc_opcode_without_fallback() {
        let i0 = OpRef(0);
        let mut same_as = Op::new(OpCode::SameAsI, &[i0]);
        same_as.pos = OpRef(1);
        let debug = Op::new(OpCode::JitDebug, &[]);

        let plan = TracePlan::build(
            &[InputArg {
                tp: Type::Int,
                index: i0.0,
            }],
            &[same_as, debug],
        );

        assert_eq!(plan.fallback_ops, 0);
        assert!(matches!(
            plan.ops[0],
            LirOp::Opcode {
                opcode: OpCode::SameAsI,
                dst: Some(OpRef(1)),
                ..
            }
        ));
        assert!(matches!(
            plan.ops[1],
            LirOp::Opcode {
                opcode: OpCode::JitDebug,
                ..
            }
        ));
    }

    #[test]
    fn lowers_remaining_guard_kinds_without_fallback() {
        let i0 = OpRef(0);
        let mut is_object = Op::new(OpCode::GuardIsObject, &[i0]);
        is_object.pos = OpRef(1);
        is_object.fail_args = Some(vec![i0].into());

        let mut future = Op::new(OpCode::GuardFutureCondition, &[]);
        future.pos = OpRef(2);
        future.fail_args = Some(vec![i0].into());

        let plan = TracePlan::build(
            &[InputArg {
                tp: Type::Ref,
                index: i0.0,
            }],
            &[is_object, future],
        );

        assert_eq!(plan.fallback_ops, 0);
        assert!(matches!(
            plan.ops[0],
            LirOp::Guard {
                kind: GuardKind::IsObject,
                ..
            }
        ));
        assert!(matches!(
            plan.ops[1],
            LirOp::Guard {
                kind: GuardKind::FutureCondition,
                ..
            }
        ));
    }
}
