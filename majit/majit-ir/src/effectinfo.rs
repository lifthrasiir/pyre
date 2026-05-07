//! Side effect classification for calls.
//!
//! Translated from `rpython/jit/codewriter/effectinfo.py`.
//!
//! PRE-EXISTING-ADAPTATION: the upstream module path is
//! `rpython/jit/codewriter/effectinfo.py`, but in pyre this lives in
//! `majit-ir` because `LLType::for_call` (descr.rs:46) holds
//! `extraeffect`/`oopspecindex` indices and `EffectInfo` is constructed
//! from RPython call analysis. Putting it in `majit-translate` would
//! create a circular crate dependency (`majit-ir` ↔ `majit-translate`).
//! The name and contents otherwise mirror upstream line-by-line.

use crate::descr::DescrRef;
use serde::{Deserialize, Serialize};

/// effectinfo.py:9-10
#[derive(Debug, Clone)]
pub struct UnsupportedFieldExc;

impl std::fmt::Display for UnsupportedFieldExc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("UnsupportedFieldExc")
    }
}

impl std::error::Error for UnsupportedFieldExc {}

/// effectinfo.py:266-269: frozenset_or_none(x)
///
/// `frozenset(x)` if `x is not None`, else `None`. Pyre uses bitsets in
/// place of frozensets so this helper just unwraps `Option<Vec<T>>` into
/// the equivalent. PRE-EXISTING-ADAPTATION (frozensets vs bitstrings).
pub fn frozenset_or_none<T>(x: Option<Vec<T>>) -> Option<Vec<T>> {
    x
}

/// effectinfo.py:380-390: consider_struct(TYPE, fieldname)
///
/// In RPython this filters out `lltype.Void` fields and non-`GcStruct`
/// types, plus the `OBJECT.typeptr` special case. Pyre lacks `lltype`
/// metadata at this layer, so the predicate is intentionally permissive
/// (the codewriter has already filtered out non-GC fields earlier).
/// PRE-EXISTING-ADAPTATION.
pub fn consider_struct(_type_name: &str, _fieldname: &str) -> bool {
    true
}

/// effectinfo.py:392-397: consider_array(ARRAY)
///
/// Same caveat as `consider_struct`. PRE-EXISTING-ADAPTATION.
pub fn consider_array(_array_name: &str) -> bool {
    true
}

/// Side effect classification for calls.
///
/// effectinfo.py:13-263: `class EffectInfo`. The seven `EF_*` constants
/// and the `OS_*` family live as associated constants on this struct
/// plus the [`ExtraEffect`] / [`OopSpecIndex`] enums for type-safe match.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectInfo {
    pub extraeffect: ExtraEffect,
    pub oopspecindex: OopSpecIndex,
    /// effectinfo.py: bitstring_readonly_descrs_fields
    pub readonly_descrs_fields: u64,
    /// effectinfo.py: bitstring_write_descrs_fields
    pub write_descrs_fields: u64,
    /// effectinfo.py: bitstring_readonly_descrs_arrays
    pub readonly_descrs_arrays: u64,
    /// effectinfo.py: bitstring_write_descrs_arrays
    pub write_descrs_arrays: u64,
    /// effectinfo.py: bitstring_readonly_descrs_interiorfields
    /// Bitset of interior field descriptor indices that may be read.
    /// effectinfo.py:327-340: interiorfield reads also set array read bits.
    pub readonly_descrs_interiorfields: u64,
    /// effectinfo.py: bitstring_write_descrs_interiorfields
    pub write_descrs_interiorfields: u64,
    /// effectinfo.py: can_invalidate
    pub can_invalidate: bool,
    /// effectinfo.py:194: can_collect — whether this call can trigger GC collection.
    /// RPython: set by collect_analyzer.analyze(op, self.seen_gc).
    pub can_collect: bool,
    /// effectinfo.py:201-206: single_write_descr_array
    #[serde(skip)]
    pub single_write_descr_array: Option<DescrRef>,
    /// effectinfo.py:196: extradescrs — extra descriptors carried by oopspec helpers
    /// (LIBFFI, ARRAYCOPY/ARRAYMOVE etc.).
    #[serde(skip)]
    pub extradescrs: Option<Vec<DescrRef>>,
    /// effectinfo.py:114, 197: call_release_gil_target = (target_fn_addr, save_err)
    /// `_NO_CALL_RELEASE_GIL_TARGET = (llmemory.NULL, 0)` by default.
    pub call_release_gil_target: (u64, i32),
}

/// Manual PartialEq: single_write_descr_array is excluded (like RPython's
/// cache key which also excludes it — effectinfo.py:155-164).
impl PartialEq for EffectInfo {
    fn eq(&self, other: &Self) -> bool {
        self.extraeffect == other.extraeffect
            && self.oopspecindex == other.oopspecindex
            && self.readonly_descrs_fields == other.readonly_descrs_fields
            && self.write_descrs_fields == other.write_descrs_fields
            && self.readonly_descrs_arrays == other.readonly_descrs_arrays
            && self.write_descrs_arrays == other.write_descrs_arrays
            && self.readonly_descrs_interiorfields == other.readonly_descrs_interiorfields
            && self.write_descrs_interiorfields == other.write_descrs_interiorfields
            && self.can_invalidate == other.can_invalidate
            && self.can_collect == other.can_collect
            && self.call_release_gil_target == other.call_release_gil_target
    }
}

impl Eq for EffectInfo {}

/// Manual Hash matching the PartialEq subset (skips
/// `single_write_descr_array` and `extradescrs` for the same reason
/// `effectinfo.py:155-164` excludes them from the cache key).
impl std::hash::Hash for EffectInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.extraeffect.hash(state);
        self.oopspecindex.hash(state);
        self.readonly_descrs_fields.hash(state);
        self.write_descrs_fields.hash(state);
        self.readonly_descrs_arrays.hash(state);
        self.write_descrs_arrays.hash(state);
        self.readonly_descrs_interiorfields.hash(state);
        self.write_descrs_interiorfields.hash(state);
        self.can_invalidate.hash(state);
        self.can_collect.hash(state);
        self.call_release_gil_target.hash(state);
    }
}

impl Default for EffectInfo {
    fn default() -> Self {
        EffectInfo {
            extraeffect: ExtraEffect::CanRaise,
            oopspecindex: OopSpecIndex::None,
            readonly_descrs_fields: 0,
            write_descrs_fields: 0,
            readonly_descrs_arrays: 0,
            write_descrs_arrays: 0,
            readonly_descrs_interiorfields: 0,
            write_descrs_interiorfields: 0,
            single_write_descr_array: None,
            extradescrs: None,
            // RPython effectinfo.py:125: can_collect=True default
            can_invalidate: false,
            can_collect: true,
            // effectinfo.py:114, 123: call_release_gil_target=_NO_CALL_RELEASE_GIL_TARGET
            call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
        }
    }
}

/// effectinfo.py:17-24: `EF_*` extraeffect constants.
///
/// The Rust enum is a PRE-EXISTING-ADAPTATION over RPython's bare
/// integer constants for type safety. The `repr(u8)` discriminants are
/// the upstream values verbatim so ordering comparisons (`>`, `>=`)
/// behave identically to RPython.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ExtraEffect {
    /// effectinfo.py:17 `EF_ELIDABLE_CANNOT_RAISE = 0`
    ElidableCannotRaise = 0,
    /// effectinfo.py:18 `EF_LOOPINVARIANT = 1`
    LoopInvariant = 1,
    /// effectinfo.py:19 `EF_CANNOT_RAISE = 2`
    CannotRaise = 2,
    /// effectinfo.py:20 `EF_ELIDABLE_OR_MEMORYERROR = 3`
    ElidableOrMemoryError = 3,
    /// effectinfo.py:21 `EF_ELIDABLE_CAN_RAISE = 4`
    ElidableCanRaise = 4,
    /// effectinfo.py:22 `EF_CAN_RAISE = 5`
    CanRaise = 5,
    /// effectinfo.py:23 `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE = 6`
    ForcesVirtualOrVirtualizable = 6,
    /// effectinfo.py:24 `EF_RANDOM_EFFECTS = 7`
    RandomEffects = 7,
}

/// effectinfo.py:27-105: `OS_*` oopspec index constants.
///
/// Rust enum + `repr(u16)` for type safety; values mirror upstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u16)]
pub enum OopSpecIndex {
    None = 0,
    Arraycopy = 1,
    Str2Unicode = 2,
    ShrinkArray = 3,
    DictLookup = 4,
    ThreadlocalrefGet = 5,
    NotInTrace = 8,
    Arraymove = 9,
    IntPyDiv = 12,
    IntUdiv = 13,
    IntPyMod = 14,
    IntUmod = 15,
    StrConcat = 22,
    StrSlice = 23,
    StrEqual = 24,
    StreqSliceChecknull = 25,
    StreqSliceNonnull = 26,
    StreqSliceChar = 27,
    StreqNonnull = 28,
    StreqNonnullChar = 29,
    StreqChecknullChar = 30,
    StreqLengthok = 31,
    StrCmp = 32,
    UniConcat = 42,
    UniSlice = 43,
    UniEqual = 44,
    UnieqSliceChecknull = 45,
    UnieqSliceNonnull = 46,
    UnieqSliceChar = 47,
    UnieqNonnull = 48,
    UnieqNonnullChar = 49,
    UnieqChecknullChar = 50,
    UnieqLengthok = 51,
    UniCmp = 52,
    LibffiCall = 62,
    LlongInvert = 69,
    LlongAdd = 70,
    LlongSub = 71,
    LlongMul = 72,
    LlongLt = 73,
    LlongLe = 74,
    LlongEq = 75,
    LlongNe = 76,
    LlongGt = 77,
    LlongGe = 78,
    LlongAnd = 79,
    LlongOr = 80,
    LlongLshift = 81,
    LlongRshift = 82,
    LlongXor = 83,
    LlongFromInt = 84,
    LlongToInt = 85,
    LlongFromFloat = 86,
    LlongToFloat = 87,
    LlongUlt = 88,
    LlongUle = 89,
    LlongUgt = 90,
    LlongUge = 91,
    LlongUrshift = 92,
    LlongFromUint = 93,
    LlongUToFloat = 94,
    MathSqrt = 100,
    MathReadTimestamp = 101,
    RawMallocVarsizeChar = 110,
    RawFree = 111,
    StrCopyToRaw = 112,
    UniCopyToRaw = 113,
    JitForceVirtual = 120,
    JitForceVirtualizable = 121,
}

impl EffectInfo {
    pub fn check_is_elidable(&self) -> bool {
        self.extraeffect == ExtraEffect::ElidableCanRaise
            || self.extraeffect == ExtraEffect::ElidableOrMemoryError
            || self.extraeffect == ExtraEffect::ElidableCannotRaise
    }

    pub fn check_can_invalidate(&self) -> bool {
        self.can_invalidate
    }

    pub fn check_can_collect(&self) -> bool {
        self.can_collect
    }

    pub fn check_forces_virtual_or_virtualizable(&self) -> bool {
        self.extraeffect >= ExtraEffect::ForcesVirtualOrVirtualizable
    }

    /// `effectinfo.py:252-253` `has_random_effects(self)`:
    ///
    /// ```python
    /// def has_random_effects(self):
    ///     return self.extraeffect >= self.EF_RANDOM_EFFECTS
    /// ```
    ///
    /// Uses `>=` (not `==`).  `EF_RANDOM_EFFECTS = 7` is the highest
    /// ExtraEffect numeric value today, so the two operators pick the
    /// same set; the structural parity matters because future upstream
    /// additions that slot a value after `EF_RANDOM_EFFECTS` would need
    /// this predicate to remain inclusive.
    pub fn has_random_effects(&self) -> bool {
        self.extraeffect >= ExtraEffect::RandomEffects
    }

    /// Whether the oopspec identifies a special-cased operation.
    pub fn has_oopspec(&self) -> bool {
        self.oopspecindex != OopSpecIndex::None
    }

    /// effectinfo.py:232-236: check_can_raise(ignore_memoryerror)
    pub fn check_can_raise(&self, ignore_memoryerror: bool) -> bool {
        if ignore_memoryerror {
            self.extraeffect > ExtraEffect::ElidableOrMemoryError
        } else {
            self.extraeffect > ExtraEffect::CannotRaise
        }
    }

    /// effectinfo.py:255-257: is_call_release_gil()
    /// `tgt_func, tgt_saveerr = self.call_release_gil_target; return bool(tgt_func)`
    pub fn is_call_release_gil(&self) -> bool {
        let (tgt_func, _tgt_saveerr) = self.call_release_gil_target;
        tgt_func != 0
    }

    /// Const-compatible constructor for static initialization.
    pub const fn const_new(extraeffect: ExtraEffect, oopspecindex: OopSpecIndex) -> Self {
        EffectInfo {
            extraeffect,
            oopspecindex,
            readonly_descrs_fields: 0,
            write_descrs_fields: 0,
            readonly_descrs_arrays: 0,
            write_descrs_arrays: 0,
            readonly_descrs_interiorfields: 0,
            write_descrs_interiorfields: 0,
            single_write_descr_array: None,
            extradescrs: None,
            can_invalidate: false,
            can_collect: true,
            call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
        }
    }

    /// Create a new EffectInfo with the given effect and oopspec.
    pub fn new(extraeffect: ExtraEffect, oopspecindex: OopSpecIndex) -> Self {
        EffectInfo {
            extraeffect,
            oopspecindex,
            ..Default::default()
        }
    }

    /// effectinfo.py:64: _OS_offset_uni = OS_UNI_CONCAT - OS_STR_CONCAT
    pub const _OS_OFFSET_UNI: u16 = OopSpecIndex::UniConcat as u16 - OopSpecIndex::StrConcat as u16;

    /// effectinfo.py:114: _NO_CALL_RELEASE_GIL_TARGET = (llmemory.NULL, 0)
    pub const _NO_CALL_RELEASE_GIL_TARGET: (u64, i32) = (0, 0);

    /// effectinfo.py:108-112: _OS_CANRAISE — oopspecindex values that can raise
    /// even when extraeffect <= EF_CANNOT_RAISE.
    pub fn _is_os_canraise(idx: OopSpecIndex) -> bool {
        matches!(
            idx,
            OopSpecIndex::None
                | OopSpecIndex::Str2Unicode
                | OopSpecIndex::LibffiCall
                | OopSpecIndex::RawMallocVarsizeChar
                | OopSpecIndex::JitForceVirtual
                | OopSpecIndex::ShrinkArray
                | OopSpecIndex::DictLookup
                | OopSpecIndex::NotInTrace
        )
    }

    /// effectinfo.py:271-273: MOST_GENERAL
    /// `EffectInfo(None, None, None, None, None, None, EF_RANDOM_EFFECTS, can_invalidate=True)`.
    ///
    /// RPython encodes the `descrs=None` arguments as "any descr can
    /// be touched" — `effectinfo.py:149-155` keeps the bitsets as
    /// `None` for `EF_RANDOM_EFFECTS` and the consumer reads `None`
    /// as a wildcard.  Pyre's bare-`u64` bitset has no `None` slot,
    /// so the parity-equivalent representation is `u64::MAX`
    /// (`force_from_effectinfo` flushes every bit-set descr index).
    pub const MOST_GENERAL: EffectInfo = EffectInfo {
        extraeffect: ExtraEffect::RandomEffects,
        oopspecindex: OopSpecIndex::None,
        readonly_descrs_fields: u64::MAX,
        write_descrs_fields: u64::MAX,
        readonly_descrs_arrays: u64::MAX,
        write_descrs_arrays: u64::MAX,
        readonly_descrs_interiorfields: u64::MAX,
        write_descrs_interiorfields: u64::MAX,
        single_write_descr_array: None,
        extradescrs: None,
        can_invalidate: true,
        can_collect: true,
        call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
    };

    // ── Bitstring check methods (effectinfo.py:211-230 parity) ──
    //
    // PyPy `effectinfo.py:211-230` does NOT short-circuit on
    // `EF_RANDOM_EFFECTS`: each `check_*` is a one-liner
    // `bitstring.bitcheck(self.bitstring_*, ei_index)`.  The
    // `EF_RANDOM_EFFECTS` case is handled at construction time —
    // `effectinfo.py:149-156` keeps the `_readonly_*`/`_write_*` sets as
    // `None` and `compute_bitstrings` (`effectinfo.py:484-489`) sets the
    // bitstring fields to `None` too.  The contract is that callers must
    // gate via `has_random_effects()` first (heap.py:460) so the
    // `None`-bitstring case is never queried.
    //
    // Pyre `MOST_GENERAL` (line 356) populates every bitset with
    // `u64::MAX` so the bitcheck reports every descr_idx < 64 as
    // touched, matching the wildcard contract for the bounded index
    // range pyre supports today.  The optimizer caller
    // (`heap.rs:2589 call_has_random_effects`) gates the same way as
    // PyPy, so `descr_idx >= 64` is unreachable in practice — the bare
    // bitcheck is sufficient parity, no `random_effects()`
    // short-circuit needed.

    /// effectinfo.py:211-213: check_readonly_descr_field(fielddescr)
    pub fn check_readonly_descr_field(&self, descr_idx: u32) -> bool {
        descr_idx < 64 && (self.readonly_descrs_fields & (1u64 << descr_idx)) != 0
    }

    /// effectinfo.py:214-216: check_write_descr_field(fielddescr)
    pub fn check_write_descr_field(&self, descr_idx: u32) -> bool {
        descr_idx < 64 && (self.write_descrs_fields & (1u64 << descr_idx)) != 0
    }

    /// effectinfo.py:217-219: check_readonly_descr_array(arraydescr)
    pub fn check_readonly_descr_array(&self, descr_idx: u32) -> bool {
        descr_idx < 64 && (self.readonly_descrs_arrays & (1u64 << descr_idx)) != 0
    }

    /// effectinfo.py:220-222: check_write_descr_array(arraydescr)
    pub fn check_write_descr_array(&self, descr_idx: u32) -> bool {
        descr_idx < 64 && (self.write_descrs_arrays & (1u64 << descr_idx)) != 0
    }

    /// effectinfo.py:223-226: check_readonly_descr_interiorfield (NOTE: not used so far)
    pub fn check_readonly_descr_interiorfield(&self, descr_idx: u32) -> bool {
        descr_idx < 64 && (self.readonly_descrs_interiorfields & (1u64 << descr_idx)) != 0
    }

    /// effectinfo.py:227-230: check_write_descr_interiorfield (NOTE: not used so far)
    pub fn check_write_descr_interiorfield(&self, descr_idx: u32) -> bool {
        descr_idx < 64 && (self.write_descrs_interiorfields & (1u64 << descr_idx)) != 0
    }

    /// effectinfo.py:201-206: set single_write_descr_array.
    ///
    /// Builder: attaches the actual array DescrRef for ARRAYCOPY/ARRAYMOVE
    /// unrolling. RPython sets this in `EffectInfo.__new__()` when
    /// `_write_descrs_arrays` has exactly one element.
    pub fn with_single_write_descr_array(mut self, descr: DescrRef) -> Self {
        self.single_write_descr_array = Some(descr);
        self
    }

    /// effectinfo.py:201-206: auto-set single_write_descr_array.
    pub fn set_single_write_descr_array(&mut self, descr: DescrRef) {
        let w = self.write_descrs_arrays;
        if w != 0 && w.is_power_of_two() {
            self.single_write_descr_array = Some(descr);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// effectinfo.py:401-418: Three analyzer classes derived from
// `BoolGraphAnalyzer`. PRE-EXISTING-ADAPTATION: pyre's `CallControl`
// already inspects Rust source operations directly, so the analyzers
// are kept as marker types whose `analyze_simple_operation` mirrors
// upstream's discriminator (op-name match) for documentation parity.
// ════════════════════════════════════════════════════════════════════════

/// effectinfo.py:401-404: VirtualizableAnalyzer
pub struct VirtualizableAnalyzer;

impl VirtualizableAnalyzer {
    /// effectinfo.py:402-404: analyze_simple_operation(op, graphinfo)
    pub fn analyze_simple_operation(opname: &str) -> bool {
        opname == "jit_force_virtualizable" || opname == "jit_force_virtual"
    }
}

/// effectinfo.py:406-408: QuasiImmutAnalyzer
pub struct QuasiImmutAnalyzer;

impl QuasiImmutAnalyzer {
    /// effectinfo.py:407-408: analyze_simple_operation(op, graphinfo)
    pub fn analyze_simple_operation(opname: &str) -> bool {
        opname == "jit_force_quasi_immutable"
    }
}

/// effectinfo.py:410-418: RandomEffectsAnalyzer
pub struct RandomEffectsAnalyzer;

impl RandomEffectsAnalyzer {
    /// effectinfo.py:411-415: analyze_external_call(funcobj)
    /// Returns true when an external call is annotated
    /// `random_effects_on_gcobjs`. Pyre has no analogous attribute on
    /// Rust function items, so this is a stub that always returns
    /// false; CallControl marks `RandomEffects` explicitly when needed.
    pub fn analyze_external_call(_random_effects_on_gcobjs: bool) -> bool {
        _random_effects_on_gcobjs
    }

    /// effectinfo.py:417-418: analyze_simple_operation(op, graphinfo)
    pub fn analyze_simple_operation(_opname: &str) -> bool {
        false
    }
}

// ════════════════════════════════════════════════════════════════════════
// effectinfo.py:422-461: CallInfoCollection
// ════════════════════════════════════════════════════════════════════════

/// effectinfo.py:422: `class CallInfoCollection(object)`.
///
/// Maps oopspec indices to `(calldescr, func_as_int)` pairs. Used to
/// look up the implementation of special-cased operations (arraycopy,
/// string ops, etc.).
#[derive(Debug, Clone, Default)]
pub struct CallInfoCollection {
    /// effectinfo.py:425: `_callinfo_for_oopspec` — `{oopspecindex: (calldescr, func_as_int)}`
    entries: std::collections::HashMap<OopSpecIndex, (DescrRef, u64)>,
    /// majit extension: func_as_int → function name.
    /// RPython derives names from `func.ptr._obj._name` at `see_raw_object` time.
    /// Since majit has no function pointers, we store the name at `add()` time.
    func_names: std::collections::HashMap<u64, String>,
}

impl CallInfoCollection {
    pub fn new() -> Self {
        Self::default()
    }

    /// effectinfo.py:430-431: add(oopspecindex, calldescr, func_as_int)
    pub fn add(&mut self, oopspec: OopSpecIndex, calldescr: DescrRef, func_addr: u64) {
        self.entries.insert(oopspec, (calldescr, func_addr));
    }

    /// Register the name for a function address.
    /// RPython: the name is `func.ptr._obj._name`, extracted by `see_raw_object`.
    /// In majit, we must store it explicitly since we have no pointer linkage.
    pub fn register_func_name(&mut self, func_addr: u64, name: String) {
        self.func_names.insert(func_addr, name);
    }

    /// effectinfo.py:433-434: has_oopspec(oopspecindex)
    pub fn has_oopspec(&self, oopspec: OopSpecIndex) -> bool {
        self.entries.contains_key(&oopspec)
    }

    /// effectinfo.py:439-447: callinfo_for_oopspec(oopspecindex)
    /// Returns (calldescr, func_as_int) for the oopspec, or `None` on miss.
    /// (RPython returns `(None, 0)` on miss; Rust uses `Option`.)
    pub fn callinfo_for_oopspec(&self, oopspec: OopSpecIndex) -> Option<&(DescrRef, u64)> {
        self.entries.get(&oopspec)
    }

    /// effectinfo.py:436-437: all_function_addresses_as_int()
    pub fn all_function_addresses_as_int(&self) -> Vec<u64> {
        self.entries.values().map(|(_, addr)| *addr).collect()
    }

    /// Look up function name by address.
    /// RPython: `see_raw_object(func.ptr)` derives name from `func.ptr._obj._name`.
    pub fn func_name(&self, addr: u64) -> Option<&str> {
        self.func_names.get(&addr).map(String::as_str)
    }
}
