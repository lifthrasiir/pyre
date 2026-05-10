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
/// `inputarg_pos` and `op_pos` are stored as `Cow` so callers may
/// either let `new` build them eagerly or share pre-built indexes that
/// outlive the trace (e.g. `RegAlloc<'a>`).
pub struct OpTypeIndex<'a> {
    inputargs: &'a [InputArg],
    ops: &'a [Op],
    constant_types: &'a HashMap<u32, Type>,
    /// `inputarg_pos[raw] = slice index in inputargs`, sentinel
    /// [`NO_POS`] for unset slots. `arg.index` raw uniqueness is
    /// enforced at build time, mirroring RPython's backend uniqueness
    /// assertion (x86/assembler.py:516-518 + aarch64/assembler.py:54-56
    /// `assert len(set(inputargs)) == len(inputargs)`).
    inputarg_pos: Cow<'a, [u32]>,
    /// `op_pos[raw] = slice index in ops`, sentinel [`NO_POS`] for
    /// unset slots and Void/None ops. `op.pos.raw()` raw uniqueness is
    /// enforced at build time per RPython Box identity (Box `is`
    /// semantics in `rpython/jit/metainterp/resoperation.py:38`).
    op_pos: Cow<'a, [u32]>,
}

/// Sentinel for "no entry at this raw u32 slot" in `inputarg_pos` /
/// `op_pos` arrays. Production raw u32 values come from monotonic
/// counters; the constant-pool side sets `CONST_BIT = 1 << 31` and is
/// gated out before reaching these arrays, so raw values land well
/// below `u32::MAX`.
pub const NO_POS: u32 = u32::MAX;

impl<'a> OpTypeIndex<'a> {
    pub fn new(
        inputargs: &'a [InputArg],
        ops: &'a [Op],
        constant_types: &'a HashMap<u32, Type>,
    ) -> Self {
        let inputarg_pos = Self::build_inputarg_pos(inputargs);
        let op_pos = Self::build_op_pos(ops);
        Self {
            inputargs,
            ops,
            constant_types,
            inputarg_pos: Cow::Owned(inputarg_pos),
            op_pos: Cow::Owned(op_pos),
        }
    }

    /// Construct from pre-built indexes (e.g. owned by `RegAlloc<'a>`).
    /// O(1) — borrows slices instead of rebuilding the position arrays.
    pub fn from_parts(
        inputargs: &'a [InputArg],
        ops: &'a [Op],
        constant_types: &'a HashMap<u32, Type>,
        inputarg_pos: &'a [u32],
        op_pos: &'a [u32],
    ) -> Self {
        Self {
            inputargs,
            ops,
            constant_types,
            inputarg_pos: Cow::Borrowed(inputarg_pos),
            op_pos: Cow::Borrowed(op_pos),
        }
    }

    /// Build `inputarg_pos` indexed by `arg.index`.
    ///
    /// RPython's backend enforces `assert len(set(inputargs)) == len(inputargs)`
    /// at loop/bridge entry (x86/assembler.py:516-518 +
    /// aarch64/assembler.py:54-56). Raw u32 uniqueness is the dual
    /// invariant in pyre's flat OpRef namespace: an `InputArgInt(7)` +
    /// `InputArgRef(7)` collision would silently keep only the later
    /// one in any variant-blind reader. Hard-panic on raw collision so
    /// the violation surfaces here rather than as a wrong-type guard
    /// fail much further along.
    pub fn build_inputarg_pos(inputargs: &[InputArg]) -> Vec<u32> {
        if inputargs.is_empty() {
            return Vec::new();
        }
        let max_raw = inputargs.iter().map(|a| a.index).max().unwrap_or(0);
        let mut pos: Vec<u32> = vec![NO_POS; max_raw as usize + 1];
        for (idx, arg) in inputargs.iter().enumerate() {
            let r = arg.index as usize;
            if pos[r] != NO_POS {
                panic!(
                    "OpTypeIndex: raw inputarg index {} bound to inputargs[{}] {:?} and inputargs[{}] {:?} — backend uniqueness violated",
                    arg.index, pos[r], inputargs[pos[r] as usize].tp, idx, arg.tp,
                );
            }
            pos[r] = idx as u32;
        }
        pos
    }

    /// Build `op_pos` indexed by `op.pos.raw()`. Filters out Void-typed
    /// ops because RPython's `box.type` only exists on Box-bearing ops;
    /// a Void op is not a Box and must never shadow an inputarg slot.
    ///
    /// RPython Box identity gives a one-to-one map from a Box object to
    /// its producing ResOperation; two Box-bearing ops sharing the same
    /// raw OpRef payload (e.g. `IntOp(7)` + `RefOp(7)`) is a Box-identity
    /// violation even though the typed variants disambiguate the type
    /// tag, because pyre's backend boundary keys by raw u32 and would
    /// silently keep only the later op. Hard-panic on raw collision so
    /// the violation surfaces here.
    pub fn build_op_pos(ops: &[Op]) -> Vec<u32> {
        let max_raw = ops
            .iter()
            .filter(|op| !op.pos.is_none() && op.type_ != Type::Void)
            .map(|op| op.pos.raw())
            .max();
        let Some(max_raw) = max_raw else {
            return Vec::new();
        };
        let mut pos: Vec<u32> = vec![NO_POS; max_raw as usize + 1];
        for (idx, op) in ops.iter().enumerate() {
            if op.pos.is_none() || op.type_ == Type::Void {
                continue;
            }
            let r = op.pos.raw() as usize;
            if pos[r] != NO_POS {
                panic!(
                    "OpTypeIndex: raw {} bound to ops[{}] {:?} and ops[{}] {:?} — Box identity broken",
                    op.pos.raw(),
                    pos[r],
                    ops[pos[r] as usize].opcode,
                    idx,
                    op.opcode,
                );
            }
            pos[r] = idx as u32;
        }
        pos
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
        // history.py:182 / resoperation.py:29: `box.type` lives on the Box
        // object itself; pyre's typed OpRef variants (`Const{Int,Float,Ptr}`
        // / `InputArg{Int,Float,Ref}` / `{Int,Float,Ref,Void}Op`) carry the
        // matching type tag intrinsically. Trust the variant first; the
        // side indexes are reserved for the transitional `Untyped` variant.
        if let Some(tp) = opref.ty() {
            return (tp != Type::Void).then_some(tp);
        }
        if opref.is_constant() {
            // history.py:220/261/307: `Const*` boxes pin `box.type` at
            // construction.  Typed `OpRef::ConstInt/ConstFloat/ConstPtr`
            // variants short-circuit at the `opref.ty()` arm above, so
            // this branch only fires for `Untyped(x | CONST_BIT)` —
            // legacy `OpRef::from_const` / `OpRef::from_raw` reconstructs
            // produced before producer-side typing was complete. The
            // canonical seeder (`ConstantPool::get_or_insert*`,
            // constant_pool.rs:173) already guarantees that any constant
            // present in a pool's `constants` map is also present in its
            // `constant_types` map, so the only way to land here with a
            // miss is for a caller to have built `OpTypeIndex` with a
            // partial `constant_types` snapshot. Panic loudly so the
            // misseed surfaces at the helper boundary instead of
            // silently mis-typing the Const as Int.
            return Some(
                self.constant_types
                    .get(&opref.raw())
                    .copied()
                    .unwrap_or_else(|| {
                        panic!(
                            "OpTypeIndex: untyped Const OpRef raw={} missing from constant_types — \
                             every Const must have its type seeded (history.py:220/261/307); \
                             producer must migrate to typed `OpRef::const_{{int,float,ref}}` \
                             factories or seed `constant_types` in lockstep with the constant pool",
                            opref.raw(),
                        )
                    }),
            );
        }
        // raw uniqueness panic at build time means there is at most one
        // ops[]/inputargs[] entry per raw u32, so there is no
        // typed-vs-raw asymmetry to walk; a single positional lookup
        // suffices.
        let raw = opref.raw() as usize;
        if let Some(idx) = op_pos_lookup(&self.op_pos, raw) {
            let tp = self.ops[idx].type_;
            if tp == Type::Void {
                return None;
            }
            if op_index.map_or(true, |at| idx <= at) {
                return Some(tp);
            }
        }
        if let Some(idx) = op_pos_lookup(&self.inputarg_pos, raw) {
            return Some(self.inputargs[idx].tp);
        }
        if let Some(idx) = op_pos_lookup(&self.op_pos, raw) {
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
        let idx = op_pos_lookup(&self.op_pos, opref.raw() as usize)?;
        Some(&self.ops[idx])
    }

    /// Inputarg-only type lookup; returns `None` if `opref` does not
    /// reference an inputarg. Used by callers that need to roll back to
    /// an OpRef's pre-redefinition type when an op later overwrote the
    /// same OpRef with a different type.
    pub fn inputarg_type(&self, opref: OpRef) -> Option<Type> {
        let idx = op_pos_lookup(&self.inputarg_pos, opref.raw() as usize)?;
        Some(self.inputargs[idx].tp)
    }
}

/// Look up `raw` in a position array (`inputarg_pos` / `op_pos`).
/// Returns `Some(idx)` for a populated slot, `None` for an out-of-range
/// raw or a sentinel slot.
#[inline]
fn op_pos_lookup(pos: &[u32], raw: usize) -> Option<usize> {
    let entry = *pos.get(raw)?;
    if entry == NO_POS {
        return None;
    }
    Some(entry as usize)
}
