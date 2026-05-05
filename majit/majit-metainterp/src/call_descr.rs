use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use majit_backend::JitCellToken;
use majit_ir::{CallDescr, DescrRef, EffectInfo, ExtraEffect, OopSpecIndex, Type, VableExpansion};

/// Generic CallDescr for function call operations.
///
/// Stores per-call-site EffectInfo, matching RPython's
/// `effectinfo_from_writeanalyze` (call.py:320).
#[derive(Debug)]
struct MetaCallDescr {
    heapcache_index: u32,
    arg_types: Vec<Type>,
    result_type: Type,
    effect_info: EffectInfo,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EffectInfoKey {
    extraeffect: ExtraEffect,
    oopspecindex: OopSpecIndex,
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
    fn from_effect_info(effect_info: &EffectInfo) -> Self {
        Self {
            extraeffect: effect_info.extraeffect,
            oopspecindex: effect_info.oopspecindex,
            readonly_descrs_fields: effect_info.readonly_descrs_fields,
            write_descrs_fields: effect_info.write_descrs_fields,
            readonly_descrs_arrays: effect_info.readonly_descrs_arrays,
            write_descrs_arrays: effect_info.write_descrs_arrays,
            readonly_descrs_interiorfields: effect_info.readonly_descrs_interiorfields,
            write_descrs_interiorfields: effect_info.write_descrs_interiorfields,
            can_invalidate: effect_info.can_invalidate,
            can_collect: effect_info.can_collect,
            call_release_gil_target: effect_info.call_release_gil_target,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CallDescrKey {
    arg_types: Vec<Type>,
    result_type: Type,
    effect_info: EffectInfoKey,
}

static CALL_DESCR_CACHE: OnceLock<Mutex<HashMap<CallDescrKey, DescrRef>>> = OnceLock::new();
static NEXT_CALL_DESCR_HEAPCACHE_INDEX: AtomicU32 = AtomicU32::new(1_000_000_000);

/// `compile.py:187 isinstance(descr, JitCellToken)` parity.
///
/// RPython's `op.getdescr()` for a `CALL_ASSEMBLER_*` op IS a `JitCellToken`
/// — `record_loop_or_bridge` reads `descr.number` directly and calls
/// `original.record_jump_to(descr)` without any indirection. majit cannot
/// inherit-from the trait, but it preserves the *identity* contract by
/// owning an `Arc<JitCellToken>` here. Callers (`direct_assembler_call`,
/// `compile_tmp_callback`) clone the same Arc that the warm cell /
/// `CompiledEntry::token` /` MemoryManager.alive_loops` already hold, so the
/// keepalive walker's downcast recovers the production token's strong
/// reference rather than a number-recovered side-table lookup.
#[derive(Debug)]
struct MetaCallAssemblerDescr {
    arg_types: Vec<Type>,
    result_type: Type,
    target_token: Arc<JitCellToken>,
    vable_expansion: Option<VableExpansion>,
}

impl majit_ir::Descr for MetaCallDescr {
    fn index(&self) -> u32 {
        self.heapcache_index
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
        Some(self.target_token.number)
    }
    fn call_virtualizable_index(&self) -> Option<usize> {
        self.target_token.virtualizable_arg_index
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
        self.target_token.number
    }

    fn call_virtualizable_index(&self) -> Option<usize> {
        self.target_token.virtualizable_arg_index
    }

    fn token_handle_any(&self) -> Option<&dyn std::any::Any> {
        Some(&self.target_token)
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
pub const DEFAULT_EFFECT_INFO: EffectInfo = EffectInfo {
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

/// `EF_CANNOT_RAISE` (effectinfo.py:19). Selected by `call.py:303
/// getcalldescr`'s `else` branch (non-elidable callee whose
/// `_canraise(op) == False`).  `pyjitpl.py:2111-2115 do_residual_call`
/// reads `exc = effectinfo.check_can_raise()` (effectinfo.py:236) which
/// is false for `extraeffect == 2`, so the canonical walker omits the
/// trailing `GUARD_NO_EXCEPTION`.
///
/// Same `read/write_descrs_*` saturation as [`DEFAULT_EFFECT_INFO`] —
/// pyre's analyzer port (Task #64) is the upstream replacement; in the
/// meantime the conservative full-bitset preserves `force_from_effectinfo`'s
/// per-cached-descr flush behaviour for callees that mutate heap state
/// but never raise.
pub const CANNOT_RAISE_EFFECT_INFO: EffectInfo = EffectInfo {
    extraeffect: ExtraEffect::CannotRaise,
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

/// `EF_ELIDABLE_CANNOT_RAISE` (effectinfo.py:17). Selected by
/// `call.py:299 getcalldescr` when `_canraise(op) == False` for an
/// elidable callee — `pyjitpl.py:2126 do_residual_call` records
/// `CALL_PURE_*` without the trailing `GUARD_NO_EXCEPTION` because
/// `effectinfo.check_can_raise()` (`effectinfo.py:232`) is false for
/// `extraeffect == 0`.
pub const ELIDABLE_CANNOT_RAISE_EFFECT_INFO: EffectInfo =
    EffectInfo::const_new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None);

/// `EF_ELIDABLE_OR_MEMORYERROR` (effectinfo.py:20). Selected by
/// `call.py:295 getcalldescr` when `_canraise(op) == "mem"` — i.e.
/// the elidable callee's only failure mode is `MemoryError`. Same
/// dispatch as `EF_ELIDABLE_CAN_RAISE` (`check_can_raise()` is true
/// for extraeffect ≥ 3) but distinguishes memory-only raises for the
/// optimizer.
pub const ELIDABLE_OR_MEMERROR_EFFECT_INFO: EffectInfo =
    EffectInfo::const_new(ExtraEffect::ElidableOrMemoryError, OopSpecIndex::None);

/// `EF_ELIDABLE_CAN_RAISE` (effectinfo.py:21). Pure calls do not need
/// the conservative flush — `effectinfo_from_writeanalyze` (effectinfo.py:
/// 169-181) clears `_write_descrs_*` for elidable extraeffects. With
/// the bitsets at zero this becomes "no writes" inside
/// `force_from_effectinfo`, matching upstream.
pub const ELIDABLE_EFFECT_INFO: EffectInfo =
    EffectInfo::const_new(ExtraEffect::ElidableCanRaise, OopSpecIndex::None);

/// `EF_LOOPINVARIANT` (effectinfo.py:18). Same write-mask treatment as
/// elidable; the trace optimizer recognises the opcode and skips cache
/// invalidation regardless of the bitsets.
pub const LOOPINVARIANT_EFFECT_INFO: EffectInfo =
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
    // effectinfo.py:144-146: `if tgt_func: key += (object(),)  # don't
    // care about caching in this case` — release-gil targets bypass the
    // EffectInfo._cache because each one carries a unique target
    // function pointer that the deduplicator should not collapse.
    // Mirror by short-circuiting the cache lookup/insert when the
    // release-gil target is non-null.
    if effect_info.call_release_gil_target.0 != 0 {
        return Arc::new(MetaCallDescr {
            heapcache_index: NEXT_CALL_DESCR_HEAPCACHE_INDEX.fetch_add(1, Ordering::Relaxed),
            arg_types: arg_types.to_vec(),
            result_type,
            effect_info,
        });
    }

    let key = CallDescrKey {
        arg_types: arg_types.to_vec(),
        result_type,
        effect_info: EffectInfoKey::from_effect_info(&effect_info),
    };

    // descr.py:22 `GcCache._cache_call`: call descriptors are cached
    // structurally, so repeated construction of the same call shape
    // yields the same descr identity.  The HashMap is that RPython
    // descriptor cache, not a side table for per-box optimizer state.
    let cache = CALL_DESCR_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().unwrap();
    if let Some(descr) = cache.get(&key) {
        return descr.clone();
    }
    let descr: DescrRef = Arc::new(MetaCallDescr {
        heapcache_index: NEXT_CALL_DESCR_HEAPCACHE_INDEX.fetch_add(1, Ordering::Relaxed),
        arg_types: arg_types.to_vec(),
        result_type,
        effect_info,
    });
    cache.insert(key, descr.clone());
    descr
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

/// `compile.py:187 isinstance(descr, JitCellToken)` parity factory.
///
/// Create a `CALL_ASSEMBLER_*` descr that owns the same `Arc<JitCellToken>`
/// as the production warm cell / `CompiledEntry::token` / `alive_loops`.
/// `direct_assembler_call` (`pyjitpl.py:3589-3609`) is the canonical caller —
/// it threads the cell's compiled token through, so `record_loop_or_bridge`'s
/// keepalive walker downcasts the descr and pushes that same Arc into
/// `original.keepalive_tokens`, matching `compile.py:187 record_jump_to(descr)`.
pub fn make_call_assembler_descr(
    target_token: Arc<JitCellToken>,
    arg_types: &[Type],
    result_type: Type,
) -> DescrRef {
    Arc::new(MetaCallAssemblerDescr {
        arg_types: arg_types.to_vec(),
        result_type,
        target_token,
        vable_expansion: None,
    })
}

/// Number-only factory for callers that have not yet been threaded an
/// `Arc<JitCellToken>` (jitcode dispatch in `dispatch.rs`, test fixtures).
///
/// Synthesises a fresh stand-alone `Arc<JitCellToken>` with the requested
/// `target_number` so the descr keeps the same shape as the identity-preserving
/// path. Identity is **not** preserved — the keepalive walker recovers the
/// real Arc via `jitcell_token_by_number(target_number)` for these descrs
/// (`pyjitpl/mod.rs:record_loop_or_bridge` Arc-fallback inside the
/// CALL_ASSEMBLER branch). Sites transitioning to
/// `make_call_assembler_descr` once the Arc is available upstream remove
/// the lookup.
pub fn make_call_assembler_descr_by_number(
    target_number: u64,
    arg_types: &[Type],
    result_type: Type,
    virtualizable_arg_index: Option<usize>,
) -> DescrRef {
    let mut tok = JitCellToken::new(target_number);
    tok.virtualizable_arg_index = virtualizable_arg_index;
    make_call_assembler_descr(Arc::new(tok), arg_types, result_type)
}

/// rewrite.py:665-695 handle_call_assembler: create a CallDescr that carries
/// virtualizable expansion info. The backend reads fields from the frame
/// reference to populate the callee's full inputarg jitframe layout.
pub fn make_call_assembler_descr_with_vable(
    target_token: Arc<JitCellToken>,
    arg_types: &[Type],
    result_type: Type,
    expansion: VableExpansion,
) -> DescrRef {
    Arc::new(MetaCallAssemblerDescr {
        arg_types: arg_types.to_vec(),
        result_type,
        target_token,
        vable_expansion: Some(expansion),
    })
}

/// Number-only sibling of `make_call_assembler_descr_with_vable` for transitional
/// callers (jitcode dispatch). See `make_call_assembler_descr_by_number`.
pub fn make_call_assembler_descr_with_vable_by_number(
    target_number: u64,
    arg_types: &[Type],
    result_type: Type,
    expansion: VableExpansion,
) -> DescrRef {
    let tok = Arc::new(JitCellToken::new(target_number));
    make_call_assembler_descr_with_vable(tok, arg_types, result_type, expansion)
}
