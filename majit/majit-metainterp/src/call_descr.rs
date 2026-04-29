use std::sync::Arc;

use majit_ir::{CallDescr, DescrRef, EffectInfo, ExtraEffect, OopSpecIndex, Type, VableExpansion};

/// Generic CallDescr for function call operations.
///
/// Stores per-call-site EffectInfo, matching RPython's
/// `effectinfo_from_writeanalyze` (call.py:320).
#[derive(Debug)]
struct MetaCallDescr {
    arg_types: Vec<Type>,
    result_type: Type,
    effect_info: EffectInfo,
}

#[derive(Debug)]
struct MetaCallAssemblerDescr {
    arg_types: Vec<Type>,
    result_type: Type,
    target_token: u64,
    vable_expansion: Option<VableExpansion>,
    /// rewrite.py:684 `jd.index_of_virtualizable`: index of the
    /// virtualizable argument inside the callee's red-arg list.
    virtualizable_arg_index: Option<usize>,
}

impl majit_ir::Descr for MetaCallDescr {
    fn index(&self) -> u32 {
        u32::MAX
    }
    fn as_call_descr(&self) -> Option<&dyn CallDescr> {
        Some(self)
    }
}

impl CallDescr for MetaCallDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }
    fn result_type(&self) -> Type {
        self.result_type
    }
    fn result_size(&self) -> usize {
        0
    }
    fn get_extra_info(&self) -> &EffectInfo {
        &self.effect_info
    }
}

impl majit_ir::Descr for MetaCallAssemblerDescr {
    fn index(&self) -> u32 {
        u32::MAX
    }
    fn as_call_descr(&self) -> Option<&dyn CallDescr> {
        Some(self)
    }
    fn as_loop_token_descr(&self) -> Option<&dyn majit_ir::descr::LoopTokenDescr> {
        Some(self)
    }
}

impl CallDescr for MetaCallAssemblerDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }
    fn result_type(&self) -> Type {
        self.result_type
    }
    fn result_size(&self) -> usize {
        8
    }
    fn call_target_token(&self) -> Option<u64> {
        Some(self.target_token)
    }
    fn call_virtualizable_index(&self) -> Option<usize> {
        self.virtualizable_arg_index
    }
    fn get_extra_info(&self) -> &EffectInfo {
        static INFO: EffectInfo = EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None);
        &INFO
    }
    fn vable_expansion(&self) -> Option<&VableExpansion> {
        self.vable_expansion.as_ref()
    }
}

impl majit_ir::descr::LoopTokenDescr for MetaCallAssemblerDescr {
    fn loop_token_number(&self) -> u64 {
        self.target_token
    }

    fn call_virtualizable_index(&self) -> Option<usize> {
        self.virtualizable_arg_index
    }
}

/// Default EffectInfo for call descriptors that lack per-call-site
/// analysis.
///
/// Upstream `effectinfo_from_writeanalyze` (effectinfo.py:285-298)
/// returns `EF_RANDOM_EFFECTS` (≡ `EffectInfo.MOST_GENERAL`,
/// effectinfo.py:271-273) for any callee whose write-analyzer reports
/// `top_set`. Pyre lacks the analyzer for the residual helpers majit
/// emits today, so the line-by-line match would be `MOST_GENERAL`.
///
/// Two practical caveats keep the default at `EF_CAN_RAISE` with all
/// read/write bitsets full instead:
///
/// 1. `MOST_GENERAL` triggers `OptHeap.call_has_random_effects` which
///    takes the `force_all_lazy_sets + clean_caches` branch. That path
///    correctly flushes the lazy_set described in the comment for
///    `make_call_descr` below — but it also invalidates non-lazy
///    field/array caches and resets `seen_guard_not_invalidated`,
///    which over-zeroes heap state across helper calls in tight loops
///    (visible as 1.5x perf drops on `fib_loop` / `inline_helper`).
/// 2. `MOST_GENERAL` makes `check_forces_virtual_or_virtualizable()`
///    true and the walker tags the call `can_raise = true`, inserting
///    a `GUARD_NO_EXCEPTION` after every helper call. That's a
///    correctness no-op for helpers that never raise but still bloats
///    the trace.
///
/// `EF_CAN_RAISE` with all-ones field/array bitsets (`u64::MAX`)
/// is the parity-equivalent middle ground: `force_from_effectinfo`
/// (heap.py:540-560) iterates per cached descr index and sees both
/// readonly and write bits set, so every cached lazy_set / field
/// gets flushed exactly the same way as the conservative branch —
/// without resetting `seen_guard_not_invalidated` or routing through
/// `clean_caches`. The bitsets cap at u64 (descr_idx < 64 in
/// `effectinfo.rs`); descr indices ≥ 64 still slip through, the same
/// blind spot upstream papered over with frozenset bitstrings before
/// the bitstring rewrite. PRE-EXISTING-ADAPTATION: the bitset width
/// upgrade is a separate slice from the EffectInfo port.
const DEFAULT_EFFECT_INFO: EffectInfo = EffectInfo {
    extraeffect: ExtraEffect::CanRaise,
    oopspecindex: OopSpecIndex::None,
    readonly_descrs_fields: u64::MAX,
    write_descrs_fields: u64::MAX,
    readonly_descrs_arrays: u64::MAX,
    write_descrs_arrays: u64::MAX,
    readonly_descrs_interiorfields: u64::MAX,
    write_descrs_interiorfields: u64::MAX,
    can_invalidate: false,
    can_collect: true,
    single_write_descr_array: None,
    extradescrs: None,
    call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
};

/// `EF_ELIDABLE_CAN_RAISE` (effectinfo.py:21). Pure calls do not need
/// the conservative flush — `effectinfo_from_writeanalyze` (effectinfo.py:
/// 169-181) clears `_write_descrs_*` for elidable extraeffects. With
/// the bitsets at zero this becomes "no writes" inside
/// `force_from_effectinfo`, matching upstream.
const ELIDABLE_EFFECT_INFO: EffectInfo =
    EffectInfo::const_new(ExtraEffect::ElidableCanRaise, OopSpecIndex::None);

/// `EF_LOOPINVARIANT` (effectinfo.py:18). Same write-mask treatment as
/// elidable; the trace optimizer recognises the opcode and skips cache
/// invalidation regardless of the bitsets.
const LOOPINVARIANT_EFFECT_INFO: EffectInfo =
    EffectInfo::const_new(ExtraEffect::LoopInvariant, OopSpecIndex::None);

/// Pick the upstream-equivalent default effect for an opcode whose
/// callee has not been write-analyzed.
///
/// `pyjitpl.py:1991-1995 do_residual_or_indirect_call` selects between
/// CALL / CALL_PURE / CALL_LOOPINVARIANT / CALL_MAY_FORCE based on
/// `descr.get_extra_info().extraeffect`. Pyre baked the choice into the
/// opcode at codewriter time, so reverse the mapping here so the descr
/// the optimizer reads carries the matching effect class.
pub fn default_effect_for_opcode(opcode: majit_ir::OpCode) -> EffectInfo {
    if opcode.is_call_pure() {
        ELIDABLE_EFFECT_INFO
    } else if opcode.is_call_loopinvariant() {
        LOOPINVARIANT_EFFECT_INFO
    } else {
        DEFAULT_EFFECT_INFO
    }
}

/// Create a CallDescr with the given argument types and result type.
pub fn make_call_descr(arg_types: &[Type], result_type: Type) -> DescrRef {
    make_call_descr_with_effect(arg_types, result_type, DEFAULT_EFFECT_INFO)
}

/// Create a CallDescr whose effect info matches the call opcode family.
pub fn make_call_descr_for_opcode(
    opcode: majit_ir::OpCode,
    arg_types: &[Type],
    result_type: Type,
) -> DescrRef {
    make_call_descr_with_effect(arg_types, result_type, default_effect_for_opcode(opcode))
}

/// call.py:320 `effectinfo_from_writeanalyze` parity. Create a
/// CallDescr with explicit per-call-site EffectInfo.
pub fn make_call_descr_with_effect(
    arg_types: &[Type],
    result_type: Type,
    effect_info: EffectInfo,
) -> DescrRef {
    Arc::new(MetaCallDescr {
        arg_types: arg_types.to_vec(),
        result_type,
        effect_info,
    })
}

/// Create a CallDescr for CALL_MAY_FORCE_* operations.
///
/// RPython treats these as may-raise calls guarded by GUARD_NOT_FORCED, not as
/// generic cannot-raise helpers.
pub fn make_call_may_force_descr(arg_types: &[Type], result_type: Type) -> DescrRef {
    #[derive(Debug)]
    struct MetaCallMayForceDescr {
        arg_types: Vec<Type>,
        result_type: Type,
    }

    impl majit_ir::Descr for MetaCallMayForceDescr {
        fn index(&self) -> u32 {
            u32::MAX
        }
        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for MetaCallMayForceDescr {
        fn arg_types(&self) -> &[Type] {
            &self.arg_types
        }
        fn result_type(&self) -> Type {
            self.result_type
        }
        fn result_size(&self) -> usize {
            0
        }
        fn get_extra_info(&self) -> &EffectInfo {
            // CALL_MAY_FORCE pairs with `GUARD_NOT_FORCED`; the
            // optimizer postpones the call (heap.rs:2722-2747) and
            // flushes lazy sets at the guard via
            // `force_lazy_sets_for_guard` (heap.rs:2770). That's the
            // single flush that mirrors RPython's same code path, so
            // there is no need to also fire `force_from_effectinfo`
            // at the call site itself — leave the bitsets empty.
            // `EF_CAN_RAISE` keeps the optimizer from flagging the
            // call as elidable / loopinvariant.
            static INFO: EffectInfo =
                EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None);
            &INFO
        }
    }

    Arc::new(MetaCallMayForceDescr {
        arg_types: arg_types.to_vec(),
        result_type,
    })
}

/// Create a CallDescr for `CALL_ASSEMBLER_*` with the given target token.
pub fn make_call_assembler_descr(
    target_token: u64,
    arg_types: &[Type],
    result_type: Type,
    virtualizable_arg_index: Option<usize>,
) -> DescrRef {
    Arc::new(MetaCallAssemblerDescr {
        arg_types: arg_types.to_vec(),
        result_type,
        target_token,
        vable_expansion: None,
        virtualizable_arg_index,
    })
}

/// rewrite.py:665-695 handle_call_assembler: create a CallDescr that carries
/// virtualizable expansion info. The backend reads fields from the frame
/// reference to populate the callee's full inputarg jitframe layout.
pub fn make_call_assembler_descr_with_vable(
    target_token: u64,
    arg_types: &[Type],
    result_type: Type,
    expansion: VableExpansion,
) -> DescrRef {
    Arc::new(MetaCallAssemblerDescr {
        arg_types: arg_types.to_vec(),
        result_type,
        target_token,
        vable_expansion: Some(expansion),
        virtualizable_arg_index: None,
    })
}
