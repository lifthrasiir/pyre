use std::borrow::Cow;
use std::collections::HashMap;

use crate::resoperation::{Op, OpRef};
use crate::value::{InputArg, Type};

/// O(1) `OpRef → Type` lookup over a trace's inputargs + ops + constant pool.
///
/// rpython/jit/metainterp/history.py:220 `ConstInt.type = INT`,
/// rpython/jit/metainterp/resoperation.py:567 `IntOp.type = 'i'` —
/// RPython reads `box.type` directly from the Box object. Pyre's
/// `OpRef(u32)` wrapper carries no intrinsic type, so backend boundaries
/// (compile.rs guard metadata, regalloc, assembler) need an explicit
/// index. Single source of truth: `op.type_`, `inputarg.tp`,
/// caller-supplied `constant_types` (resume.py ResumeDataLoopMemo).
///
/// `inputarg_index` and `op_index` are stored as `Cow` so callers may
/// either let `new` build them eagerly or share pre-built indexes that
/// outlive the trace (e.g. `RegAlloc<'a>`).
pub struct OpTypeIndex<'a> {
    inputargs: &'a [InputArg],
    ops: &'a [Op],
    constant_types: &'a HashMap<u32, Type>,
    inputarg_index: Cow<'a, HashMap<u32, usize>>,
    op_index: Cow<'a, HashMap<u32, usize>>,
}

impl<'a> OpTypeIndex<'a> {
    pub fn new(
        inputargs: &'a [InputArg],
        ops: &'a [Op],
        constant_types: &'a HashMap<u32, Type>,
    ) -> Self {
        let inputarg_index: HashMap<u32, usize> = inputargs
            .iter()
            .enumerate()
            .map(|(idx, arg)| (arg.index, idx))
            .collect();
        let op_index = Self::build_op_index(ops);
        Self {
            inputargs,
            ops,
            constant_types,
            inputarg_index: Cow::Owned(inputarg_index),
            op_index: Cow::Owned(op_index),
        }
    }

    /// Construct from pre-built indexes (e.g. owned by `RegAlloc<'a>`).
    /// Saves the O(n) HashMap rebuild on each call.
    pub fn from_parts(
        inputargs: &'a [InputArg],
        ops: &'a [Op],
        constant_types: &'a HashMap<u32, Type>,
        inputarg_index: &'a HashMap<u32, usize>,
        op_index: &'a HashMap<u32, usize>,
    ) -> Self {
        Self {
            inputargs,
            ops,
            constant_types,
            inputarg_index: Cow::Borrowed(inputarg_index),
            op_index: Cow::Borrowed(op_index),
        }
    }

    /// Re-export of inputarg index for callers that hold their own
    /// `OpTypeIndex` and need to pre-populate it on cache rebuild.
    pub fn build_inputarg_index(inputargs: &[InputArg]) -> HashMap<u32, usize> {
        inputargs
            .iter()
            .enumerate()
            .map(|(idx, arg)| (arg.index, idx))
            .collect()
    }

    /// Re-export of op index for the same purpose as
    /// `build_inputarg_index`. Filters out Void-typed ops because RPython's
    /// `box.type` only exists on Box-bearing ops; a Void op is not a Box
    /// and must never shadow an inputarg slot when flat-`OpRef(u32)`
    /// positions collide.
    ///
    /// RPython Box identity gives a one-to-one map from a Box object to
    /// its producing ResOperation; collapsing two Box-bearing ops onto
    /// the same `OpRef(u32)` would silently keep the later op only and
    /// rewrite Box.type for any earlier reader. Assert the invariant on
    /// build so a violation surfaces here instead of as a wrong-type
    /// guard fail much further along.
    pub fn build_op_index(ops: &[Op]) -> HashMap<u32, usize> {
        let mut map: HashMap<u32, usize> = HashMap::new();
        for (idx, op) in ops.iter().enumerate() {
            if op.pos.is_none() || op.type_ == Type::Void {
                continue;
            }
            if let Some(&prev_idx) = map.get(&op.pos.0) {
                debug_assert!(
                    false,
                    "OpTypeIndex: OpRef({}) bound to ops[{}] {:?} and ops[{}] {:?} — Box identity broken",
                    op.pos.0, prev_idx, ops[prev_idx].opcode, idx, op.opcode,
                );
            }
            map.insert(op.pos.0, idx);
        }
        map
    }

    /// Lookup priority for a fully defined value: real constants
    /// (`ConstInt.type`/`ConstPtr.type`/`ConstFloat.type`) first, then ops
    /// (`opclasses[opnum].type` — resoperation.py:1693), then inputargs
    /// (`InputArgInt/Ref/Float.type` — history.py:220). Returns `None`
    /// for `OpRef::NONE`, unresolvable refs, or `Type::Void`.
    ///
    /// This is the post-definition view.  Callers that stand at a specific
    /// trace position must use `opref_type_at`, which preserves the
    /// RPython Box-identity rule for flat `OpRef(u32)` collisions by using
    /// the inputarg type until the colliding op result has actually been
    /// defined.
    ///
    /// `constant_types` may still contain compatibility seeds for non-constant
    /// refs at some call sites, but RPython's `box.type` only takes the Const
    /// path for actual Const boxes.  Guard the table by `OpRef::is_constant()`
    /// so such seeds cannot shadow op/inputarg typing.
    pub fn opref_type(&self, opref: OpRef) -> Option<Type> {
        self.opref_type_at_or_after(opref, None)
    }

    /// Position-sensitive `box.type` lookup.
    ///
    /// RPython never has to ask whether an integer id names an InputArg or a
    /// later ResOperation: those are different Box objects.  pyre's flat
    /// `OpRef(u32)` namespace can collide, so callers that are walking the
    /// trace must pass their current operation index.  Before the producing
    /// op is reached, the inputarg Box is the only live Box with that id; at
    /// or after the producing op, the ResOperation's `.type` is the live
    /// Box type.
    pub fn opref_type_at(&self, opref: OpRef, op_index: usize) -> Option<Type> {
        self.opref_type_at_or_after(opref, Some(op_index))
    }

    fn opref_type_at_or_after(&self, opref: OpRef, op_index: Option<usize>) -> Option<Type> {
        if opref.is_none() {
            return None;
        }
        if opref.is_constant() {
            // PyPy `ConstInt.type=='i'` / `ConstFloat.type=='f'` /
            // `ConstPtr.type=='r'` are object attributes, always set
            // (history.py:220).  When `constant_types` is unseeded
            // (synthetic / empty-map callers), fall back to Int — the
            // statically-most-common Const flavour — so the helper still
            // satisfies the "Const always has a type" invariant rather
            // than returning None and diverging from PyPy.
            return Some(
                self.constant_types
                    .get(&opref.0)
                    .copied()
                    .unwrap_or(Type::Int),
            );
        }
        if let Some(&idx) = self.op_index.get(&opref.0) {
            let tp = self.ops[idx].type_;
            if tp == Type::Void {
                return None;
            }
            if op_index.map_or(true, |at| idx <= at) {
                return Some(tp);
            }
        }
        if let Some(&idx) = self.inputarg_index.get(&opref.0) {
            return Some(self.inputargs[idx].tp);
        }
        if let Some(&idx) = self.op_index.get(&opref.0) {
            let tp = self.ops[idx].type_;
            if tp != Type::Void {
                return Some(tp);
            }
        }
        None
    }

    /// Direct `OpRef → &Op` lookup; returns `None` for constants,
    /// inputargs, or `OpRef::NONE`.
    pub fn op_at(&self, opref: OpRef) -> Option<&Op> {
        let idx = *self.op_index.get(&opref.0)?;
        Some(&self.ops[idx])
    }

    /// Inputarg-only type lookup; returns `None` if `opref` does not
    /// reference an inputarg. Used by callers that need to roll back to
    /// an OpRef's pre-redefinition type when an op later overwrote the
    /// same OpRef with a different type.
    pub fn inputarg_type(&self, opref: OpRef) -> Option<Type> {
        let idx = *self.inputarg_index.get(&opref.0)?;
        Some(self.inputargs[idx].tp)
    }
}
