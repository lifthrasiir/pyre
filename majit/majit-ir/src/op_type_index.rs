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
    inputarg_index: Cow<'a, HashMap<OpRef, usize>>,
    op_index: Cow<'a, HashMap<OpRef, usize>>,
    /// Raw-keyed fallback for legacy `OpRef::from_raw()` (`Untyped(_)`)
    /// readers. Pre-Phase-3 storage was `HashMap<u32, usize>` and a raw
    /// reader matched any same-raw entry regardless of variant tag; the
    /// variant-aware typed maps reject that match. Until every
    /// `from_raw` callsite is migrated to a typed factory, the raw maps
    /// keep `Untyped(n)` lookups finding typed entries at raw n.
    raw_inputarg_index: HashMap<u32, usize>,
    raw_op_index: HashMap<u32, usize>,
}

impl<'a> OpTypeIndex<'a> {
    pub fn new(
        inputargs: &'a [InputArg],
        ops: &'a [Op],
        constant_types: &'a HashMap<u32, Type>,
    ) -> Self {
        let inputarg_index = Self::build_inputarg_index(inputargs);
        let op_index = Self::build_op_index(ops);
        let raw_inputarg_index = Self::build_raw_inputarg_index(inputargs);
        let raw_op_index = Self::build_raw_op_index(ops);
        Self {
            inputargs,
            ops,
            constant_types,
            inputarg_index: Cow::Owned(inputarg_index),
            op_index: Cow::Owned(op_index),
            raw_inputarg_index,
            raw_op_index,
        }
    }

    /// Construct from pre-built indexes (e.g. owned by `RegAlloc<'a>`).
    /// Saves the O(n) HashMap rebuild on each call.
    pub fn from_parts(
        inputargs: &'a [InputArg],
        ops: &'a [Op],
        constant_types: &'a HashMap<u32, Type>,
        inputarg_index: &'a HashMap<OpRef, usize>,
        op_index: &'a HashMap<OpRef, usize>,
    ) -> Self {
        let raw_inputarg_index = Self::build_raw_inputarg_index(inputargs);
        let raw_op_index = Self::build_raw_op_index(ops);
        Self {
            inputargs,
            ops,
            constant_types,
            inputarg_index: Cow::Borrowed(inputarg_index),
            op_index: Cow::Borrowed(op_index),
            raw_inputarg_index,
            raw_op_index,
        }
    }

    /// Build the raw inputarg index in iteration order. RPython's
    /// backend enforces `assert len(set(inputargs)) == len(inputargs)`
    /// at loop/bridge entry (x86/assembler.py:516-518 +
    /// aarch64/assembler.py:54-56). The raw u32 boundary is the
    /// variant-blind dual of that assertion: pyre's regalloc/assembler
    /// raw fallback path keys by raw u32, so an `InputArgInt(7)` +
    /// `InputArgRef(7)` collision would silently keep only the later
    /// one. Hard-panic on raw collision so the violation surfaces here
    /// rather than as a wrong-type guard fail much further along,
    /// symmetric to `build_op_index`.
    pub fn build_raw_inputarg_index(inputargs: &[InputArg]) -> HashMap<u32, usize> {
        let mut map: HashMap<u32, usize> = HashMap::with_capacity(inputargs.len());
        for (idx, arg) in inputargs.iter().enumerate() {
            if let Some(&prev_idx) = map.get(&arg.index) {
                panic!(
                    "OpTypeIndex: raw inputarg index {} bound to inputargs[{}] {:?} and inputargs[{}] {:?} — backend uniqueness violated",
                    arg.index, prev_idx, inputargs[prev_idx].tp, idx, arg.tp,
                );
            }
            map.insert(arg.index, idx);
        }
        map
    }

    /// Build the raw op index in iteration order.  `build_op_index`
    /// hard-panics on raw collision (RPython Box identity), so this
    /// table always agrees with the typed map; the iteration-order
    /// build keeps the result deterministic across runs.
    pub fn build_raw_op_index(ops: &[Op]) -> HashMap<u32, usize> {
        let mut map = HashMap::with_capacity(ops.len());
        for (idx, op) in ops.iter().enumerate() {
            if op.pos.is_none() || op.type_ == Type::Void {
                continue;
            }
            map.insert(op.pos.raw(), idx);
        }
        map
    }

    /// Re-export of inputarg index for callers that hold their own
    /// `OpTypeIndex` and need to pre-populate it on cache rebuild.
    /// Keys are typed `OpRef::input_arg_*(arg.index)` per `arg.tp`,
    /// matching the variant a caller would synthesize via
    /// `InputArg::opref()`.
    pub fn build_inputarg_index(inputargs: &[InputArg]) -> HashMap<OpRef, usize> {
        inputargs
            .iter()
            .enumerate()
            .map(|(idx, arg)| (OpRef::input_arg_typed(arg.index, arg.tp), idx))
            .collect()
    }

    /// Re-export of op index for the same purpose as
    /// `build_inputarg_index`. Filters out Void-typed ops because RPython's
    /// `box.type` only exists on Box-bearing ops; a Void op is not a Box
    /// and must never shadow an inputarg slot when flat-`OpRef(u32)`
    /// positions collide.
    ///
    /// RPython Box identity gives a one-to-one map from a Box object to
    /// its producing ResOperation; two Box-bearing ops sharing the same
    /// raw OpRef payload (e.g. `IntOp(7)` + `RefOp(7)`) is a Box-identity
    /// violation even though the typed variants disambiguate the type
    /// tag, because pyre's backend boundary (regalloc/assembler/raw
    /// fallback path) keys by raw u32 and would silently keep only the
    /// later op. Hard-panic on raw collision (release as well as debug)
    /// so a violation surfaces here instead of as a wrong-type guard
    /// fail much further along.
    pub fn build_op_index(ops: &[Op]) -> HashMap<OpRef, usize> {
        let mut map: HashMap<OpRef, usize> = HashMap::new();
        let mut raw_seen: HashMap<u32, usize> = HashMap::new();
        for (idx, op) in ops.iter().enumerate() {
            if op.pos.is_none() || op.type_ == Type::Void {
                continue;
            }
            let raw = op.pos.raw();
            if let Some(&prev_idx) = raw_seen.get(&raw) {
                panic!(
                    "OpTypeIndex: raw {} bound to ops[{}] {:?} and ops[{}] {:?} — Box identity broken",
                    raw, prev_idx, ops[prev_idx].opcode, idx, op.opcode,
                );
            }
            raw_seen.insert(raw, idx);
            map.insert(op.pos, idx);
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
            // construction. Typed `OpRef::ConstInt/ConstFloat/ConstPtr`
            // variants short-circuit at the `opref.ty()` arm above, so
            // this branch only fires for `Untyped(x | CONST_BIT)` —
            // legacy `OpRef::from_const` / `OpRef::from_raw` reconstructs
            // produced before producer-side typing was complete.
            //
            // PRE-EXISTING-ADAPTATION: cranelift backend has callsites
            // that build `OpTypeIndex` with a partial `constant_types`
            // snapshot, so a strict miss-panic here triggers caught
            // unwinds that silently mis-execute (`fib_recursive`,
            // `nested_loop`, `fannkuch` regress on cranelift). Fall back
            // to `Type::Int` — the statically-most-common Const flavour —
            // so the helper still satisfies the "Const always has a
            // type" invariant rather than returning None.  Closing this
            // requires every legacy `from_const` / `from_raw` const
            // producer to be migrated to typed factories (#171) AND each
            // cranelift caller to seed the full pool.
            return Some(
                self.constant_types
                    .get(&opref.raw())
                    .copied()
                    .unwrap_or(Type::Int),
            );
        }
        // Variant-aware primary lookup: typed Box identity (an
        // `IntOp(n)` is not the same Box as a `RefOp(n)` even when the
        // raw payloads coincide).
        if let Some(&idx) = self.op_index.get(&opref) {
            let tp = self.ops[idx].type_;
            if tp == Type::Void {
                return None;
            }
            if op_index.map_or(true, |at| idx <= at) {
                return Some(tp);
            }
        }
        if let Some(&idx) = self.inputarg_index.get(&opref) {
            return Some(self.inputargs[idx].tp);
        }
        if let Some(&idx) = self.op_index.get(&opref) {
            let tp = self.ops[idx].type_;
            if tp != Type::Void {
                return Some(tp);
            }
        }
        // Variant-blind raw fallback: legacy `OpRef::from_raw(n)` lands
        // on `Untyped(n)`. Pre-Phase-3 raw-keyed storage matched that
        // against any typed entry at raw n; preserve that until every
        // `from_raw` reader is migrated to a typed factory.
        let raw = opref.raw();
        if let Some(&idx) = self.raw_op_index.get(&raw) {
            let tp = self.ops[idx].type_;
            if tp == Type::Void {
                return None;
            }
            if op_index.map_or(true, |at| idx <= at) {
                return Some(tp);
            }
        }
        if let Some(&idx) = self.raw_inputarg_index.get(&raw) {
            return Some(self.inputargs[idx].tp);
        }
        if let Some(&idx) = self.raw_op_index.get(&raw) {
            let tp = self.ops[idx].type_;
            if tp != Type::Void {
                return Some(tp);
            }
        }
        None
    }

    /// Direct `OpRef → &Op` lookup; returns `None` for constants,
    /// inputargs, or `OpRef::NONE`.  Variant-aware typed key first;
    /// falls back to raw u32 so legacy `OpRef::from_raw()` `Untyped(_)`
    /// readers (e.g. `compile.rs:1381` exit-arg ForceToken filter) keep
    /// resolving against typed Box-bearing ops at the same raw payload.
    pub fn op_at(&self, opref: OpRef) -> Option<&Op> {
        if let Some(&idx) = self.op_index.get(&opref) {
            return Some(&self.ops[idx]);
        }
        let idx = *self.raw_op_index.get(&opref.raw())?;
        Some(&self.ops[idx])
    }

    /// Inputarg-only type lookup; returns `None` if `opref` does not
    /// reference an inputarg. Used by callers that need to roll back to
    /// an OpRef's pre-redefinition type when an op later overwrote the
    /// same OpRef with a different type.
    ///
    /// **Variant-blind by design.** The collision-detection use case
    /// (`majit-backend-cranelift` `build_type_overrides`) asks "does an
    /// inputarg slot exist at this raw position regardless of variant
    /// tag?" — an op's `.pos` typed as `IntOp(n)` should still surface
    /// the inputarg at slot n even when the inputarg's typed key is
    /// `InputArgInt(n)`. Raw lookup mirrors RPython's flat box-id
    /// namespace where the question is purely positional.
    pub fn inputarg_type(&self, opref: OpRef) -> Option<Type> {
        let idx = *self.raw_inputarg_index.get(&opref.raw())?;
        Some(self.inputargs[idx].tp)
    }
}
