/// JIT IR operations, faithfully translated from rpython/jit/metainterp/resoperation.py.
///
/// Operations with multiple result types (e.g., SAME_AS/1/ifr) are expanded
/// into type-suffixed variants (SameAsI, SameAsR, SameAsF).
///
/// Naming convention: CamelCase variant name, with type suffix I/R/F/N where applicable.
use smallvec::SmallVec;

use crate::descr::DescrRef;
use crate::value::{Const, GcRef, Type};

/// Index into an operation list, used as a reference to an operation's
/// result. Variant-tagged enum mirroring RPython's `AbstractValue` class
/// hierarchy (resoperation.py:29 + history.py:182).
///
/// Each typed variant carries the same raw u32 encoding used by the
/// pre-Phase-3 tuple struct (`CONST_BIT` set for `Const*`, plain `pos`
/// for `InputArg*` / `*Op`). The `Untyped` variant is a transitional
/// state produced by `from_raw()` when reconstructing an OpRef from a
/// u32 of unknown type (HashMap key, ConstantPool index, etc.); Phase 4
/// audits each `from_raw` call site and specializes it to a typed
/// factory.
///
/// `PartialEq` / `Eq` / `Hash` include the enum variant, not just `.raw()`.
/// This keeps the disjoint RPython Box classes disjoint even when Pyre's
/// flat encoding reuses the same raw position across InputArg / ResOp /
/// Const namespaces.
#[derive(Clone, Copy, Debug)]
pub enum OpRef {
    /// Sentinel for missing/absent reference; `OpRef::NONE` aliases this.
    /// RPython has no equivalent — missing values are Python `None`.
    None,
    /// history.py:220 `ConstInt` — `type = 'i'`. Payload: raw u32 with
    /// `CONST_BIT` set; low bits are the constant pool index.
    ConstInt(u32),
    /// history.py:261 `ConstFloat` — `type = 'f'`.
    ConstFloat(u32),
    /// history.py:307 `ConstPtr` — `type = 'r'`.
    ConstPtr(u32),
    /// resoperation.py:719 `InputArgInt` — `type = 'i'`. Payload: input
    /// arg slot position.
    InputArgInt(u32),
    /// resoperation.py:727 `InputArgFloat` — `type = 'f'`.
    InputArgFloat(u32),
    /// resoperation.py:739 `InputArgRef` — `type = 'r'`.
    InputArgRef(u32),
    /// `AbstractResOp` + `IntOp` mixin — `type = 'i'`. Payload: op
    /// result OpRef position.
    IntOp(u32),
    /// `AbstractResOp` + `FloatOp` mixin — `type = 'f'`.
    FloatOp(u32),
    /// `AbstractResOp` + `RefOp` mixin — `type = 'r'`.
    RefOp(u32),
    /// `AbstractResOp` default — `type = 'v'` (resoperation.py:260).
    /// Void-result ops (SETFIELD_GC, GUARD_*, JUMP, …) carry no result
    /// type but still occupy an op position.
    VoidOp(u32),
    /// Transitional — `from_raw()` lands here when reconstructing from
    /// a u32 of unknown type. Phase 4 audits each call site and
    /// specializes to a typed variant.
    Untyped(u32),
}

impl OpRef {
    pub const NONE: OpRef = OpRef::None;
    /// High bit distinguishes constant-namespace OpRefs from operation OpRefs.
    /// opencoder.py: TAGINT/TAGCONSTPTR/TAGCONSTOTHER/TAGBOX use 2-bit tags;
    /// here a single high bit suffices (op vs const).
    const CONST_BIT: u32 = 1 << 31;
    /// Top of the u32 range reserved for sentinel OpRefs used by the
    /// regalloc backend (temp register placeholders); kept disjoint from
    /// the constant namespace so `is_constant()` returns false for them.
    const SENTINEL_BASE: u32 = u32::MAX - 16;

    pub fn is_none(self) -> bool {
        matches!(self, Self::None | Self::Untyped(u32::MAX))
    }

    /// Extract the raw u32 payload. For `None` returns `u32::MAX` to
    /// preserve pre-Phase-3 round-trip semantics.
    #[inline]
    pub const fn raw(self) -> u32 {
        match self {
            Self::None => u32::MAX,
            Self::ConstInt(x)
            | Self::ConstFloat(x)
            | Self::ConstPtr(x)
            | Self::InputArgInt(x)
            | Self::InputArgFloat(x)
            | Self::InputArgRef(x)
            | Self::IntOp(x)
            | Self::FloatOp(x)
            | Self::RefOp(x)
            | Self::VoidOp(x)
            | Self::Untyped(x) => x,
        }
    }

    /// Mirrors RPython `AbstractValue.type` — the type embedded in the
    /// variant tag. `Untyped` returns `None`; Phase 4 audits and
    /// specializes those sites.
    pub fn ty(self) -> Option<Type> {
        match self {
            Self::None | Self::Untyped(_) => None,
            Self::ConstInt(_) | Self::InputArgInt(_) | Self::IntOp(_) => Some(Type::Int),
            Self::ConstFloat(_) | Self::InputArgFloat(_) | Self::FloatOp(_) => Some(Type::Float),
            Self::ConstPtr(_) | Self::InputArgRef(_) | Self::RefOp(_) => Some(Type::Ref),
            Self::VoidOp(_) => Some(Type::Void),
        }
    }

    /// Create an OpRef in the constant namespace from a zero-based index.
    /// Type-erased — prefer `const_int` / `const_float` / `const_ptr`
    /// where the type is known.
    pub fn from_const(index: u32) -> OpRef {
        debug_assert!(
            index & Self::CONST_BIT == 0,
            "const index too large: {}",
            index
        );
        let raw = index | Self::CONST_BIT;
        debug_assert!(
            raw < Self::SENTINEL_BASE,
            "const index collides with regalloc sentinel range: {}",
            index
        );
        OpRef::Untyped(raw)
    }

    /// Extract the zero-based constant index (masks off high bit).
    pub fn const_index(self) -> u32 {
        debug_assert!(self.is_constant());
        self.raw() & !Self::CONST_BIT
    }

    /// resoperation.py: is_constant() — Const subclass check.
    pub fn is_constant(self) -> bool {
        match self {
            Self::ConstInt(_) | Self::ConstFloat(_) | Self::ConstPtr(_) => true,
            Self::Untyped(x) if x & Self::CONST_BIT != 0 && x < Self::SENTINEL_BASE => true,
            _ => false,
        }
    }

    /// Bit-helper variant of `is_constant()` for callers that hold a raw
    /// u32 from an index-keyed pool (constant pool key, opencoder tag,
    /// etc.) and only need to test the constant-namespace bit.  Avoids
    /// the `OpRef::from_raw(raw).is_constant()` round-trip which would
    /// land on `OpRef::Untyped(_)` and is being retired (see plan
    /// `untyped-opref-retirement-epic.md` Slice P3).
    ///
    /// Stays raw u32 because the underlying pool (`HashMap<u32, V>`) is
    /// genuinely index-keyed (Slice P3 category E — keep raw u32).
    pub const fn raw_is_constant(raw: u32) -> bool {
        raw & Self::CONST_BIT != 0 && raw < Self::SENTINEL_BASE
    }

    /// Bit-helper variant of `const_index()` for callers that hold a raw
    /// u32 known to be a constant-namespace key.  See `raw_is_constant`
    /// for context.
    pub const fn raw_const_index(raw: u32) -> u32 {
        debug_assert!(Self::raw_is_constant(raw));
        raw & !Self::CONST_BIT
    }

    // ── Typed constructors mirroring RPython AbstractValue variants ──
    //
    // Each factory produces the matching enum variant carrying the
    // pre-Phase-3 raw u32 encoding. These are the canonical OpRef
    // construction entry points; `from_raw` is reserved for type-erased
    // reconstruction during the Phase 3→4 migration window.

    /// history.py:220 `ConstInt` — `type = 'i'`. Index points into the
    /// integer constant pool.
    pub const fn const_int(idx: u32) -> OpRef {
        OpRef::ConstInt(idx | Self::CONST_BIT)
    }

    /// history.py:261 `ConstFloat` — `type = 'f'`.
    pub const fn const_float(idx: u32) -> OpRef {
        OpRef::ConstFloat(idx | Self::CONST_BIT)
    }

    /// history.py:307 `ConstPtr` — `type = 'r'`.
    pub const fn const_ptr(idx: u32) -> OpRef {
        OpRef::ConstPtr(idx | Self::CONST_BIT)
    }

    /// resoperation.py:719 `InputArgInt` — `type = 'i'`.
    pub const fn input_arg_int(pos: u32) -> OpRef {
        OpRef::InputArgInt(pos)
    }

    /// resoperation.py:727 `InputArgFloat` — `type = 'f'`.
    pub const fn input_arg_float(pos: u32) -> OpRef {
        OpRef::InputArgFloat(pos)
    }

    /// resoperation.py:739 `InputArgRef` — `type = 'r'`.
    pub const fn input_arg_ref(pos: u32) -> OpRef {
        OpRef::InputArgRef(pos)
    }

    /// `AbstractResOp` + `IntOp` mixin — `type = 'i'`.
    pub const fn int_op(pos: u32) -> OpRef {
        OpRef::IntOp(pos)
    }

    /// `AbstractResOp` + `FloatOp` mixin — `type = 'f'`.
    pub const fn float_op(pos: u32) -> OpRef {
        OpRef::FloatOp(pos)
    }

    /// `AbstractResOp` + `RefOp` mixin — `type = 'r'`.
    pub const fn ref_op(pos: u32) -> OpRef {
        OpRef::RefOp(pos)
    }

    /// `AbstractResOp` default — `type = 'v'` (resoperation.py:260).
    pub const fn void_op(pos: u32) -> OpRef {
        OpRef::VoidOp(pos)
    }

    /// Allocate a typed `Const*` OpRef from a zero-based pool index. The
    /// type tag picks the matching variant (history.py:220/261/307).
    /// `Type::Void` is rejected — RPython has no Const-Void class.
    pub fn const_typed(idx: u32, tp: Type) -> OpRef {
        match tp {
            Type::Int => OpRef::const_int(idx),
            Type::Float => OpRef::const_float(idx),
            Type::Ref => OpRef::const_ptr(idx),
            Type::Void => panic!("Void constants are not supported (no ConstVoid class upstream)"),
        }
    }

    /// Allocate a typed `InputArg*` OpRef from a position. The type tag
    /// picks the matching variant (resoperation.py:719/727/739).
    /// `Type::Void` is rejected — RPython has no Void inputarg class.
    pub fn input_arg_typed(pos: u32, tp: Type) -> OpRef {
        match tp {
            Type::Int => OpRef::input_arg_int(pos),
            Type::Float => OpRef::input_arg_float(pos),
            Type::Ref => OpRef::input_arg_ref(pos),
            Type::Void => panic!("Void input args are not supported"),
        }
    }

    /// Allocate a typed `*Op` OpRef from a position. The type tag picks
    /// the matching mixin variant (resoperation.py:564-638).
    /// `Type::Void` lands on `VoidOp` — `AbstractResOp.type = 'v'`
    /// (resoperation.py:260), the default for ops with no result-type
    /// mixin.
    pub const fn op_typed(pos: u32, tp: Type) -> OpRef {
        match tp {
            Type::Int => OpRef::int_op(pos),
            Type::Float => OpRef::float_op(pos),
            Type::Ref => OpRef::ref_op(pos),
            Type::Void => OpRef::void_op(pos),
        }
    }

    /// Type-erased reconstruction from a previously-stored raw u32.
    /// Used when an OpRef was serialized into a u32-keyed container
    /// (HashMap key, ConstantPool index, etc.) and must be reconstructed
    /// without a known type. Phase 4 audits each call site and
    /// specializes to a typed factory; until then `Untyped(_)` carries
    /// the value.
    pub const fn from_raw(raw: u32) -> OpRef {
        if raw == u32::MAX {
            OpRef::None
        } else {
            OpRef::Untyped(raw)
        }
    }

    /// Re-encode this OpRef's variant with a fresh raw payload while
    /// preserving the type tag. Used by post-optimization remaps that
    /// renumber positions but keep RPython's `box.type` attached
    /// (history.py:802 record_same_as parity, where the remapped Box
    /// inherits the source Box's `.type`).
    ///
    /// `None` round-trips to `None` regardless of `new_raw`.
    pub const fn with_raw(self, new_raw: u32) -> OpRef {
        match self {
            Self::None => Self::None,
            Self::ConstInt(_) => Self::ConstInt(new_raw),
            Self::ConstFloat(_) => Self::ConstFloat(new_raw),
            Self::ConstPtr(_) => Self::ConstPtr(new_raw),
            Self::InputArgInt(_) => Self::InputArgInt(new_raw),
            Self::InputArgFloat(_) => Self::InputArgFloat(new_raw),
            Self::InputArgRef(_) => Self::InputArgRef(new_raw),
            Self::IntOp(_) => Self::IntOp(new_raw),
            Self::FloatOp(_) => Self::FloatOp(new_raw),
            Self::RefOp(_) => Self::RefOp(new_raw),
            Self::VoidOp(_) => Self::VoidOp(new_raw),
            Self::Untyped(_) => Self::Untyped(new_raw),
        }
    }
}

/// `PartialEq` / `Hash` are variant-aware: the enum discriminant
/// participates so `ConstInt(x) != ConstFloat(x) != ConstPtr(x) !=
/// Untyped(x)` even when raw payloads coincide. Mirrors RPython's
/// `AbstractValue.same_box` (resoperation.py:38 `self is other`) and
/// `ConstInt.same_constant` (history.py:244) which reject across the
/// disjoint Const / InputArg / ResOp sub-hierarchies.
///
/// Hand-written rather than `#[derive]` so that any "none" OpRef
/// (`Self::None` and `Self::Untyped(u32::MAX)` — the legacy sentinel
/// produced by `OpRef::from_raw(u32::MAX)`) collapses onto a single
/// equivalence class. RPython has no separate identity for missing
/// values; pre-Phase-3 raw-only equality also collapsed these. Without
/// this special case `OpRef::None != OpRef::from_raw(u32::MAX)` would
/// split the absent-value sentinel into two non-interchangeable keys.
impl PartialEq for OpRef {
    fn eq(&self, other: &Self) -> bool {
        if self.is_none() {
            return other.is_none();
        }
        if other.is_none() {
            return false;
        }
        std::mem::discriminant(self) == std::mem::discriminant(other) && self.raw() == other.raw()
    }
}

impl Eq for OpRef {}

impl std::hash::Hash for OpRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if self.is_none() {
            // Single bucket for every absent-value representation so
            // `OpRef::None` and `Untyped(u32::MAX)` hash identically.
            std::mem::discriminant(&OpRef::None).hash(state);
            u32::MAX.hash(state);
            return;
        }
        std::mem::discriminant(self).hash(state);
        self.raw().hash(state);
    }
}

/// AbstractValue parity: rpython/jit/metainterp/resoperation.py:29
/// + history.py:182.
///
/// RPython's `AbstractValue` is the root of the value hierarchy that
/// carries `type` ('i' / 'r' / 'f') as a class-level constant. The
/// concrete subclasses split into three families:
///
/// 1. **`Const` family** (history.py:220 `ConstInt`, history.py:261
///    `ConstFloat`, history.py:307 `ConstPtr`).
/// 2. **`AbstractInputArg` family** (resoperation.py:719 `InputArgInt`,
///    resoperation.py:727 `InputArgFloat`, resoperation.py:739
///    `InputArgRef`).
/// 3. **`AbstractResOp`** (resoperation.py:250) mixed with one of the
///    `IntOp` / `FloatOp` / `RefOp` mixins (resoperation.py:564-638) —
///    every concrete ResOp subclass picks up its `type` attribute via
///    one of these three mixins.
///
/// In pyre, [`OpRef`] carries only a numeric handle (CONST_BIT-encoded
/// for constants, `pos` for inputargs and op results) and the type lives
/// in side tables (`OptContext::value_types`, the constant pool, op
/// result types). `AbstractValue` is the Rust analogue of an
/// instantiated RPython value: `type` is intrinsic to the variant.
///
/// The `None` variant is a Rust adaptation for missing/sentinel
/// references; in RPython missing values are Python `None` or absent
/// attributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AbstractValue {
    None,
    /// history.py:220 `ConstInt` — `type = 'i'`. Payload: index into
    /// the integer constant pool.
    ConstInt(u32),
    /// history.py:261 `ConstFloat` — `type = 'f'`.
    ConstFloat(u32),
    /// history.py:307 `ConstPtr` — `type = 'r'`.
    ConstPtr(u32),
    /// resoperation.py:719 `InputArgInt` — `type = 'i'`. Payload: input
    /// arg slot position.
    InputArgInt(u32),
    /// resoperation.py:727 `InputArgFloat` — `type = 'f'`.
    InputArgFloat(u32),
    /// resoperation.py:739 `InputArgRef` — `type = 'r'`.
    InputArgRef(u32),
    /// `AbstractResOp` + `IntOp` mixin — `type = 'i'`. Payload: op
    /// result OpRef position.
    IntOp(u32),
    /// `AbstractResOp` + `FloatOp` mixin — `type = 'f'`.
    FloatOp(u32),
    /// `AbstractResOp` + `RefOp` mixin — `type = 'r'`.
    RefOp(u32),
    /// `AbstractResOp` default — `type = 'v'` (resoperation.py:260).
    /// Void-result ops (SETFIELD_GC, GUARD_*, JUMP, …).
    VoidOp(u32),
}

impl AbstractValue {
    /// Mirrors RPython `AbstractValue.type` access.
    pub fn ty(self) -> Option<Type> {
        match self {
            Self::None => None,
            Self::ConstInt(_) | Self::InputArgInt(_) | Self::IntOp(_) => Some(Type::Int),
            Self::ConstFloat(_) | Self::InputArgFloat(_) | Self::FloatOp(_) => Some(Type::Float),
            Self::ConstPtr(_) | Self::InputArgRef(_) | Self::RefOp(_) => Some(Type::Ref),
            Self::VoidOp(_) => Some(Type::Void),
        }
    }

    /// Mirrors RPython `isinstance(value, Const)`.
    pub fn is_constant(self) -> bool {
        matches!(
            self,
            Self::ConstInt(_) | Self::ConstFloat(_) | Self::ConstPtr(_)
        )
    }

    /// Mirrors RPython `isinstance(value, AbstractInputArg)`.
    pub fn is_input_arg(self) -> bool {
        matches!(
            self,
            Self::InputArgInt(_) | Self::InputArgFloat(_) | Self::InputArgRef(_)
        )
    }

    /// Mirrors RPython `isinstance(value, AbstractResOp)`.
    pub fn is_res_op(self) -> bool {
        matches!(
            self,
            Self::IntOp(_) | Self::FloatOp(_) | Self::RefOp(_) | Self::VoidOp(_)
        )
    }

    /// Returns the variant payload (constant index, input arg position,
    /// or op result position). `None` variant returns `None`.
    pub fn raw(self) -> Option<u32> {
        match self {
            Self::None => None,
            Self::ConstInt(x)
            | Self::ConstFloat(x)
            | Self::ConstPtr(x)
            | Self::InputArgInt(x)
            | Self::InputArgFloat(x)
            | Self::InputArgRef(x)
            | Self::IntOp(x)
            | Self::FloatOp(x)
            | Self::RefOp(x)
            | Self::VoidOp(x) => Some(x),
        }
    }
}

/// resume.py:576-860: virtual object serialization for rd_virtuals.
///
/// Each variant corresponds to a concrete virtual type in RPython's
/// resume.py:591-593 AbstractVirtualStructInfo.fielddescrs parity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldDescrInfo {
    pub index: u32,
    pub offset: usize,
    pub field_type: Type,
    pub field_size: usize,
}

/// Serializable snapshot of an ArrayDescr.
///
/// RPython's resume.py:692 VRawBufferInfo carries live ArrayDescr objects,
/// but we cannot put `Arc<dyn Descr>` in the IR serialization boundary.
/// This captures the fields needed by `_descrs_are_compatible()` (rawbuffer.py:83)
/// and `setrawbuffer_item()` dispatch (resume.py:1543).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArrayDescrInfo {
    /// Descriptor registry index.
    pub index: u32,
    /// descr.py:273 ArrayDescr.basesize.
    pub base_size: usize,
    /// descr.py:274 ArrayDescr.itemsize.
    pub item_size: usize,
    /// Item type: 0=ref, 1=int, 2=float.
    pub item_type: u8,
    /// descr.py:241-254 FLAG_SIGNED.
    pub is_signed: bool,
}

/// AbstractVirtualInfo hierarchy (VirtualInfo, VStructInfo, VArrayInfo, etc.).
#[derive(Clone, Debug)]
pub enum RdVirtualInfo {
    /// resume.py:612 VirtualInfo(descr, fielddescrs).
    VirtualInfo {
        /// resume.py:615 self.descr — live SizeDescr reference.
        descr: Option<crate::DescrRef>,
        /// descr.tid — GC type identifier for allocation dispatch.
        type_id: u32,
        descr_index: u32,
        known_class: Option<i64>,
        fielddescrs: Vec<FieldDescrInfo>,
        fieldnums: Vec<i16>,
        descr_size: usize,
    },
    /// resume.py:628 VStructInfo(typedescr, fielddescrs).
    VStructInfo {
        /// resume.py:631 self.typedescr — live SizeDescr reference.
        typedescr: Option<crate::DescrRef>,
        /// typedescr.tid — GC type identifier (cached for serialization).
        type_id: u32,
        descr_index: u32,
        fielddescrs: Vec<FieldDescrInfo>,
        fieldnums: Vec<i16>,
        descr_size: usize,
    },
    /// resume.py:680: VArrayInfoClear (clear=True)
    VArrayInfoClear {
        /// resume.py:646 self.arraydescr — live ArrayDescr reference.
        arraydescr: Option<crate::DescrRef>,
        descr_index: u32,
        /// resume.py:656: arraydescr element kind (ref/int/float).
        kind: u8, // 0=ref, 1=int, 2=float (ArrayDescr.flag parity)
        fieldnums: Vec<i16>,
    },
    /// resume.py:683: VArrayInfoNotClear (clear=False)
    VArrayInfoNotClear {
        /// resume.py:646 self.arraydescr — live ArrayDescr reference.
        arraydescr: Option<crate::DescrRef>,
        descr_index: u32,
        /// resume.py:656: arraydescr element kind (ref/int/float).
        kind: u8, // 0=ref, 1=int, 2=float (ArrayDescr.flag parity)
        fieldnums: Vec<i16>,
    },
    /// resume.py:736: VArrayStructInfo
    VArrayStructInfo {
        /// resume.py:739 self.arraydescr — live ArrayDescr reference.
        arraydescr: Option<crate::DescrRef>,
        descr_index: u32,
        size: usize,
        /// resume.py:740: self.fielddescrs — live InteriorFieldDescr objects.
        fielddescrs: Vec<crate::DescrRef>,
        /// resume.py VArrayStructInfo.fielddescrs — per-field descriptor indices.
        fielddescr_indices: Vec<u32>,
        /// resume.py:757: fielddescrs[j].is_pointer_field/is_float_field dispatch.
        /// Per-field type within each element: 0=ref, 1=int, 2=float.
        field_types: Vec<u8>,
        /// descr.py:273 ArrayDescr.basesize — fixed header before array items.
        base_size: usize,
        /// llmodel.py:648: arraydescr.itemsize — bytes per struct element.
        item_size: usize,
        /// llmodel.py:649: fielddescr.offset — per-field byte offset within struct.
        field_offsets: Vec<usize>,
        /// llmodel.py:649: fielddescr.field_size — per-field byte width.
        field_sizes: Vec<usize>,
        fieldnums: Vec<i16>,
    },
    /// resume.py:692: VRawBufferInfo(func, size, offsets, descrs)
    VRawBufferInfo {
        /// resume.py:695: self.func — raw malloc function pointer.
        func: i64,
        size: usize,
        /// resume.py:696: self.offsets — byte offsets of stored values.
        offsets: Vec<usize>,
        /// resume.py:697: self.descrs — per-entry ArrayDescr snapshots.
        /// RPython carries live ArrayDescr objects; we carry serializable snapshots.
        descrs: Vec<ArrayDescrInfo>,
        fieldnums: Vec<i16>,
    },
    /// resume.py:717: VRawSliceInfo
    VRawSliceInfo {
        offset: usize,
        fieldnums: Vec<i16>,
    },
    /// resume.py:763 `VStrPlainInfo` — virtual byte-string built from
    /// character fieldnums. `fieldnums` length = string length.
    VStrPlainInfo {
        fieldnums: Vec<i16>,
    },
    /// resume.py:781 `VStrConcatInfo` — virtual concatenation of two
    /// strings. `fieldnums = [left, right]`.
    VStrConcatInfo {
        fieldnums: Vec<i16>,
    },
    /// resume.py:801 `VStrSliceInfo` — virtual slice of a larger string.
    /// `fieldnums = [largerstr, start, length]`.
    VStrSliceInfo {
        fieldnums: Vec<i16>,
    },
    /// resume.py:817 `VUniPlainInfo` — unicode counterpart of VStrPlain.
    VUniPlainInfo {
        fieldnums: Vec<i16>,
    },
    /// resume.py:836 `VUniConcatInfo` — unicode counterpart of VStrConcat.
    VUniConcatInfo {
        fieldnums: Vec<i16>,
    },
    /// resume.py:856 `VUniSliceInfo` — unicode counterpart of VStrSlice.
    VUniSliceInfo {
        fieldnums: Vec<i16>,
    },
    Empty,
}

// PartialEq/Eq: compare by data fields, skip descr/typedescr (Arc<dyn Descr>).
impl PartialEq for RdVirtualInfo {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::VirtualInfo {
                    type_id: a0,
                    descr_index: a1,
                    known_class: a2,
                    fielddescrs: a3,
                    fieldnums: a4,
                    descr_size: a5,
                    ..
                },
                Self::VirtualInfo {
                    type_id: b0,
                    descr_index: b1,
                    known_class: b2,
                    fielddescrs: b3,
                    fieldnums: b4,
                    descr_size: b5,
                    ..
                },
            ) => a0 == b0 && a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4 && a5 == b5,
            (
                Self::VStructInfo {
                    type_id: a1,
                    descr_index: a2,
                    fielddescrs: a3,
                    fieldnums: a4,
                    descr_size: a5,
                    ..
                },
                Self::VStructInfo {
                    type_id: b1,
                    descr_index: b2,
                    fielddescrs: b3,
                    fieldnums: b4,
                    descr_size: b5,
                    ..
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4 && a5 == b5,
            (
                Self::VArrayInfoClear {
                    arraydescr: _,
                    descr_index: a1,
                    kind: a2,
                    fieldnums: a3,
                },
                Self::VArrayInfoClear {
                    arraydescr: _,
                    descr_index: b1,
                    kind: b2,
                    fieldnums: b3,
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3,
            (
                Self::VArrayInfoNotClear {
                    arraydescr: _,
                    descr_index: a1,
                    kind: a2,
                    fieldnums: a3,
                },
                Self::VArrayInfoNotClear {
                    arraydescr: _,
                    descr_index: b1,
                    kind: b2,
                    fieldnums: b3,
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3,
            (
                Self::VArrayStructInfo {
                    arraydescr: _,
                    descr_index: a1,
                    size: a2,
                    fielddescrs: _,
                    fielddescr_indices: a3,
                    field_types: a4,
                    base_size: a4b,
                    item_size: a5,
                    field_offsets: a6,
                    field_sizes: a7,
                    fieldnums: a8,
                },
                Self::VArrayStructInfo {
                    arraydescr: _,
                    descr_index: b1,
                    size: b2,
                    fielddescrs: _,
                    fielddescr_indices: b3,
                    field_types: b4,
                    base_size: b4b,
                    item_size: b5,
                    field_offsets: b6,
                    field_sizes: b7,
                    fieldnums: b8,
                },
            ) => {
                a1 == b1
                    && a2 == b2
                    && a3 == b3
                    && a4 == b4
                    && a4b == b4b
                    && a5 == b5
                    && a6 == b6
                    && a7 == b7
                    && a8 == b8
            }
            (
                Self::VRawBufferInfo {
                    func: a0,
                    size: a1,
                    offsets: a2,
                    descrs: a3,
                    fieldnums: a4,
                },
                Self::VRawBufferInfo {
                    func: b0,
                    size: b1,
                    offsets: b2,
                    descrs: b3,
                    fieldnums: b4,
                },
            ) => a0 == b0 && a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4,
            (
                Self::VRawSliceInfo {
                    offset: a1,
                    fieldnums: a2,
                },
                Self::VRawSliceInfo {
                    offset: b1,
                    fieldnums: b2,
                },
            ) => a1 == b1 && a2 == b2,
            (Self::VStrPlainInfo { fieldnums: a }, Self::VStrPlainInfo { fieldnums: b }) => a == b,
            (Self::VStrConcatInfo { fieldnums: a }, Self::VStrConcatInfo { fieldnums: b }) => {
                a == b
            }
            (Self::VStrSliceInfo { fieldnums: a }, Self::VStrSliceInfo { fieldnums: b }) => a == b,
            (Self::VUniPlainInfo { fieldnums: a }, Self::VUniPlainInfo { fieldnums: b }) => a == b,
            (Self::VUniConcatInfo { fieldnums: a }, Self::VUniConcatInfo { fieldnums: b }) => {
                a == b
            }
            (Self::VUniSliceInfo { fieldnums: a }, Self::VUniSliceInfo { fieldnums: b }) => a == b,
            (Self::Empty, Self::Empty) => true,
            _ => false,
        }
    }
}
impl Eq for RdVirtualInfo {}

impl RdVirtualInfo {
    /// resume.py:584-585 `AbstractVirtualInfo.set_content` stores `fieldnums`
    /// onto every concrete vinfo. This accessor exposes that per-variant
    /// field for `equals` / caching.
    pub fn fieldnums(&self) -> Option<&[i16]> {
        match self {
            Self::VirtualInfo { fieldnums, .. }
            | Self::VStructInfo { fieldnums, .. }
            | Self::VArrayInfoClear { fieldnums, .. }
            | Self::VArrayInfoNotClear { fieldnums, .. }
            | Self::VArrayStructInfo { fieldnums, .. }
            | Self::VRawBufferInfo { fieldnums, .. }
            | Self::VRawSliceInfo { fieldnums, .. }
            | Self::VStrPlainInfo { fieldnums }
            | Self::VStrConcatInfo { fieldnums }
            | Self::VStrSliceInfo { fieldnums }
            | Self::VUniPlainInfo { fieldnums }
            | Self::VUniConcatInfo { fieldnums }
            | Self::VUniSliceInfo { fieldnums } => Some(fieldnums),
            Self::Empty => None,
        }
    }

    /// resume.py:581-582 `AbstractVirtualInfo.equals(fieldnums)`:
    ///
    /// ```python
    /// def equals(self, fieldnums):
    ///     return tagged_list_eq(self.fieldnums, fieldnums)
    /// ```
    ///
    /// Used by `ResumeDataVirtualAdder.make_virtual_info` (resume.py:310)
    /// to decide whether a cached `_cached_vinfo` can be reused verbatim.
    pub fn equals(&self, other_fieldnums: &[i16]) -> bool {
        self.fieldnums().is_some_and(|fns| fns == other_fieldnums)
    }

    /// resume.py:584-585 `AbstractVirtualInfo.set_content(fieldnums)`:
    ///
    /// ```python
    /// def set_content(self, fieldnums):
    ///     self.fieldnums = fieldnums
    /// ```
    ///
    /// Called by `ResumeDataVirtualAdder.make_virtual_info` (resume.py:313)
    /// after `info.visitor_dispatch_virtual_type(self)` produced a fresh
    /// vinfo — the visitor constructs the variant without fieldnums,
    /// and this method stores the caller-supplied `fieldnums` onto it
    /// before caching.
    pub fn set_content(&mut self, new_fieldnums: Vec<i16>) {
        match self {
            Self::VirtualInfo { fieldnums, .. }
            | Self::VStructInfo { fieldnums, .. }
            | Self::VArrayInfoClear { fieldnums, .. }
            | Self::VArrayInfoNotClear { fieldnums, .. }
            | Self::VArrayStructInfo { fieldnums, .. }
            | Self::VRawBufferInfo { fieldnums, .. }
            | Self::VRawSliceInfo { fieldnums, .. }
            | Self::VStrPlainInfo { fieldnums }
            | Self::VStrConcatInfo { fieldnums }
            | Self::VStrSliceInfo { fieldnums }
            | Self::VUniPlainInfo { fieldnums }
            | Self::VUniConcatInfo { fieldnums }
            | Self::VUniSliceInfo { fieldnums } => *fieldnums = new_fieldnums,
            Self::Empty => {}
        }
    }
}

/// resume.py:87-92 PENDINGFIELDSTRUCT parity: a deferred
/// SETFIELD_GC/SETARRAYITEM_GC where the stored value is virtual.
/// Encoded into the guard's resume data and replayed on guard failure
/// after virtual materialization.
///
/// Fields mirror PENDINGFIELDSTRUCT (lldescr / num / fieldnum / itemindex).
/// `target` / `value` are pyre-only (SSA position before resume numbering);
/// `descr_index` is a stable handle for serialization paths
/// (`PendingFieldLayoutSummary`, `ExitPendingFieldLayout`) that travel
/// through layers without an `Arc<dyn Descr>` slot.
#[derive(Clone, Debug)]
pub struct GuardPendingFieldEntry {
    /// resume.py:88 `lldescr`: the field/array descriptor itself. Carries
    /// `field_offset` / `field_size` / `field_type` via the trait, so the
    /// consumer never needs a precomputed cache.
    pub descr: Option<DescrRef>,
    /// Stable u32 handle for serialization sinks
    /// (`ExitPendingFieldLayout`, `PendingFieldLayoutSummary`).
    /// Equals `descr.as_ref().map_or(0, |d| d.index())` when both are set.
    pub descr_index: u32,
    /// resume.py:91 `itemindex` — for SETARRAYITEM_GC the constant array
    /// index, -1 for SETFIELD_GC.
    pub item_index: i32,
    /// OpRef of the target struct/array (compile-time SSA position,
    /// pyre-only — RPython resolves this via Box identity).
    pub target: OpRef,
    /// OpRef of the value being stored (compile-time SSA position).
    pub value: OpRef,
    /// resume.py:89 `num` — tagged target (TAGBOX/TAGCONST/TAGVIRTUAL).
    /// Set by store_final_boxes_in_guard when resume numbering is available.
    pub target_tagged: i16,
    /// resume.py:90 `fieldnum` — tagged value (TAGBOX/TAGCONST/TAGVIRTUAL).
    pub value_tagged: i16,
}

/// resume.py:419-426 — virtual object field info discovered by
/// `visitor_walk_recursive` inside `finish()`.
#[derive(Debug, Clone)]
pub struct VirtualFieldsInfo {
    /// Type descriptor for the virtual object.
    pub descr: Option<DescrRef>,
    /// Known class pointer (for NewWithVtable).
    pub known_class: Option<GcRef>,
    /// Field OpRefs (after get_box_replacement). Order matches the
    /// virtual's field descriptor list.
    pub field_oprefs: Vec<OpRef>,
}

/// resume.py:192-226 parity: box environment for _number_boxes.
///
/// Abstracts the operations RPython performs on boxes during snapshot
/// numbering. Used by ResumeDataLoopMemo.number() to tag each box.
pub trait BoxEnv {
    /// resume.py:202 — box.get_box_replacement()
    fn get_box_replacement(&self, opref: OpRef) -> OpRef;
    /// resume.py:204 — isinstance(box, Const)
    fn is_const(&self, opref: OpRef) -> bool;
    /// Constant value + type. Only valid when is_const returns true.
    fn get_const(&self, opref: OpRef) -> (i64, Type);
    /// resume.py:211,214 — box.type
    fn get_type(&self, opref: OpRef) -> Type;
    /// resume.py:212-213 — getptrinfo(box) is not None and info.is_virtual()
    fn is_virtual_ref(&self, opref: OpRef) -> bool;
    /// resume.py:215-216 — getrawptrinfo(box) is not None and info.is_virtual()
    fn is_virtual_raw(&self, opref: OpRef) -> bool;
    /// resume.py:419-426 — getptrinfo(box).visitor_walk_recursive(box, self)
    ///
    /// Returns virtual field info for the given OpRef if it is a virtual
    /// object. Called by `finish()` to discover virtual fields inline,
    /// matching RPython's callback-based `visitor_walk_recursive` pattern.
    /// Default returns None (no virtual info available).
    fn get_virtual_fields(&self, _opref: OpRef) -> Option<VirtualFieldsInfo> {
        None
    }
    /// bridgeopt.py:79-80: getptrinfo(box).get_known_class(cpu) is not None.
    /// Returns true if the optimizer knows the class of the given OpRef.
    fn has_known_class(&self, _opref: OpRef) -> bool {
        false
    }
    /// resume.py:307-315 make_virtual_info(info, fieldnums) parity.
    ///
    /// Creates an `RdVirtualInfo` for a virtual OpRef with given fieldnums.
    /// Dispatches on the virtual type (Virtual, VStruct, VArray, etc.)
    /// to produce the correct variant — matching RPython's
    /// `info.visitor_dispatch_virtual_type(self)` + `vinfo.set_content(fieldnums)`.
    fn make_virtual_info(
        &self,
        _opref: OpRef,
        _fieldnums: Vec<i16>,
    ) -> Option<std::rc::Rc<RdVirtualInfo>> {
        None
    }
    /// resume.py:504-505 `if vinfo.fieldnums is not fieldnums: memo.nvreused += 1`.
    ///
    /// Returns true when `make_virtual_info()` would reuse an already-cached
    /// virtual info object for the given `(opref, fieldnums)` instead of
    /// allocating a fresh one.
    fn virtual_info_would_be_reused(&self, _opref: OpRef, _fieldnums: &[i16]) -> bool {
        false
    }
}

/// A single IR operation.
#[derive(Clone, Debug)]
pub struct Op {
    pub opcode: OpCode,
    pub args: SmallVec<[OpRef; 3]>,
    pub descr: Option<DescrRef>,
    /// Index of this op in the trace (set by the trace builder).
    pub pos: OpRef,
    /// resoperation.py:1693 `opclasses[opnum].type` parity (Box.type intrinsic).
    /// Mirrors RPython's `op.type` class attribute set by `optypes[opnum]`
    /// (`resoperation.py:1597`). Populated at construction from
    /// `opcode.result_type()`. Replaces side-table `value_types: HashMap<u32, Type>`.
    pub type_: Type,
    /// For guard ops: values to store in the dead frame on guard failure.
    /// Mirrors rpython/jit/metainterp/resoperation.py getfailargs/setfailargs.
    /// If None, the backend falls back to storing input args.
    pub fail_args: Option<SmallVec<[OpRef; 3]>>,
    /// Types of fail_args, set by the optimizer from constant_types.
    /// When present, the backend uses these instead of inferring types.
    pub fail_arg_types: Option<Vec<Type>>,
    /// resoperation.py: GuardResOp.rd_resume_position — index of the
    /// guard in the trace for resume data lookup. Set by unroll when
    /// creating extra guards from short preamble / virtual state.
    /// -1 means unset.
    pub rd_resume_position: i32,
    /// resoperation.py:156-200: VectorizationInfo — per-op vector metadata.
    /// Set by the vectorizer to track SIMD lane count, byte size, signedness.
    pub vecinfo: Option<Box<VectorizationInfo>>,
}

/// resoperation.py:156-200: Per-op vector metadata for the vectorizer.
/// Tracks how a scalar op maps to SIMD lanes.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorizationInfo {
    /// 'i' for integer, 'f' for float, '\0' for unset
    pub datatype: char,
    /// Byte size per element (-1 = machine word)
    pub bytesize: i8,
    /// Whether the values are signed
    pub signed: bool,
    /// Number of SIMD lanes (-1 = unset)
    pub count: i16,
}

impl VectorizationInfo {
    /// resoperation.py:156-162: default values
    pub fn new() -> Self {
        VectorizationInfo {
            datatype: '\0',
            bytesize: -1,
            signed: true,
            count: -1,
        }
    }

    /// resoperation.py:214-230: setinfo — normalize bytesize by datatype.
    pub fn setinfo(&mut self, datatype: char, bytesize: i8, signed: bool) {
        self.datatype = datatype;
        self.bytesize = if bytesize == -1 {
            match datatype {
                'i' => Self::INT_WORD,
                'f' => Self::FLOAT_WORD,
                'r' => Self::INT_WORD,
                'v' => 0,
                'V' => Self::INT_WORD,
                _ => Self::INT_WORD, // safe default
            }
        } else {
            bytesize
        };
        self.signed = signed;
    }

    /// resoperation.py:219-222: getbytesize
    pub fn getbytesize(&self) -> usize {
        if self.bytesize == -1 {
            Self::INT_WORD as usize
        } else {
            self.bytesize as usize
        }
    }

    /// Machine word sizes (64-bit platform).
    const INT_WORD: i8 = 8;
    const FLOAT_WORD: i8 = 8;

    /// resoperation.py:224-227: getcount
    pub fn getcount(&self) -> usize {
        if self.count == -1 {
            1
        } else {
            self.count as usize
        }
    }
}

impl Op {
    pub fn new(opcode: OpCode, args: &[OpRef]) -> Self {
        Op {
            opcode,
            args: SmallVec::from_slice(args),
            descr: None,
            pos: OpRef::NONE,
            type_: opcode.result_type(),
            fail_args: None,
            fail_arg_types: None,
            rd_resume_position: -1,
            vecinfo: None,
        }
    }

    pub fn with_descr(opcode: OpCode, args: &[OpRef], descr: DescrRef) -> Self {
        Op {
            opcode,
            args: SmallVec::from_slice(args),
            descr: Some(descr),
            pos: OpRef::NONE,
            type_: opcode.result_type(),
            fail_args: None,
            fail_arg_types: None,
            rd_resume_position: -1,
            vecinfo: None,
        }
    }

    pub fn arg(&self, idx: usize) -> OpRef {
        self.args[idx]
    }

    pub fn num_args(&self) -> usize {
        self.args.len()
    }

    pub fn result_type(&self) -> Type {
        self.type_
    }

    /// resoperation.py:323-334 AbstractResOp.copy_and_change +
    /// resoperation.py:498-503 GuardResOp.copy_and_change parity.
    ///
    /// "shallow copy: the returned operation is meant to be used in place
    /// of self". For guard ops, copies fail_args AND rd_resume_position.
    /// `fail_arg_types` is the only resume-related cache still on `Op`;
    /// `rd_numb / rd_consts / rd_virtuals / rd_pendingfields` live on
    /// the descr (compile.py:855 `_attrs_`) and follow `descr` automatically
    /// when the same DescrRef is reused.
    ///
    /// `args=None` → reuse self.args (matches getarglist_copy()).
    /// `descr=None` → reuse self.descr.
    pub fn copy_and_change(
        &self,
        opcode: OpCode,
        args: Option<&[OpRef]>,
        descr: Option<Option<DescrRef>>,
    ) -> Op {
        let new_args: SmallVec<[OpRef; 3]> = match args {
            Some(a) => SmallVec::from_slice(a),
            None => self.args.clone(),
        };
        let new_descr = match descr {
            Some(d) => d,
            None => self.descr.clone(),
        };
        let mut newop = Op {
            opcode,
            args: new_args,
            descr: new_descr,
            pos: self.pos,
            type_: opcode.result_type(),
            fail_args: None,
            fail_arg_types: None,
            rd_resume_position: -1,
            // resoperation.py:511-518 VectorOp/VectorGuardOp.copy_and_change
            // copy datatype/bytesize/signed/count from the source.  pyre
            // collapses VectorOp/VectorGuardOp into Op, so the same copy
            // happens unconditionally — None for scalar ops, Some(_) for
            // vector ops which is what RPython's Vector* subclasses do.
            vecinfo: self.vecinfo.clone(),
        };
        // resoperation.py:498-503 GuardResOp.copy_and_change:
        //   newop.setfailargs(self.getfailargs())
        //   newop.rd_resume_position = self.rd_resume_position
        // The check is on opcode.is_guard() because in RPython this lives
        // on the GuardResOp class hierarchy.  rd_* live on the descr
        // (compile.py:855 `_attrs_`); the descr Arc was already copied
        // above, so newop reads the same payload through descr.fail_descr().
        if opcode.is_guard() || self.opcode.is_guard() {
            newop.fail_args = self.fail_args.clone();
            newop.fail_arg_types = self.fail_arg_types.clone();
            newop.rd_resume_position = self.rd_resume_position;
        }
        newop
    }

    /// `compile.py:849 ResumeGuardCopiedDescr.get_resumestorage(): return prev`
    /// parity. Reads `rd_numb` from `op.descr` — `ResumeGuardCopiedDescr`
    /// chases `prev` automatically.
    pub fn resolved_rd_numb(&self) -> Option<&[u8]> {
        self.descr.as_ref()?.as_fail_descr()?.rd_numb()
    }

    /// Same as `resolved_rd_numb` but for the `rd_consts` const pool.
    pub fn resolved_rd_consts(&self) -> Option<&[Const]> {
        self.descr.as_ref()?.as_fail_descr()?.rd_consts()
    }

    /// Same as `resolved_rd_numb` but for the `rd_virtuals` table.
    pub fn resolved_rd_virtuals(&self) -> Option<&[std::rc::Rc<RdVirtualInfo>]> {
        self.descr.as_ref()?.as_fail_descr()?.rd_virtuals()
    }

    /// Same as `resolved_rd_numb` but for the `rd_pendingfields` table.
    pub fn resolved_rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.descr.as_ref()?.as_fail_descr()?.rd_pendingfields()
    }
    /// compile.py: ResumeGuardDescr.store_final_boxes(guard_op, boxes, metainterp_sd)
    ///   guard_op.setfailargs(boxes)
    /// compile.py:874-876 store_final_boxes
    pub fn store_final_boxes(&mut self, boxes: Vec<OpRef>) {
        // optimizer.py:745-749: check no duplicates (debug only)
        #[cfg(debug_assertions)]
        {
            let mut seen = std::collections::HashSet::new();
            for &b in &boxes {
                if !b.is_none() {
                    debug_assert!(seen.insert(b.raw()), "duplicate box in fail_args: {:?}", b);
                }
            }
        }
        self.fail_args = Some(boxes.into());
    }
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.opcode.is_guard() {
            write!(f, "{:?}(", self.opcode)?;
            for (i, arg) in self.args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "v{}", arg.raw())?;
            }
            write!(f, ")")?;
            if let Some(ref fa) = self.fail_args {
                write!(f, " [")?;
                for (i, arg) in fa.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "v{}", arg.raw())?;
                }
                write!(f, "]")?;
            }
            Ok(())
        } else if self.result_type() != Type::Void {
            write!(f, "v{} = {:?}(", self.pos.raw(), self.opcode)?;
            for (i, arg) in self.args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "v{}", arg.raw())?;
            }
            write!(f, ")")
        } else {
            write!(f, "{:?}(", self.opcode)?;
            for (i, arg) in self.args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "v{}", arg.raw())?;
            }
            write!(f, ")")
        }
    }
}

/// Format a trace (list of ops) with optional constants for debugging.
pub fn format_trace(ops: &[Op], constants: &std::collections::HashMap<u32, i64>) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for op in ops {
        // Replace known constants in display
        write!(out, "  ").unwrap();
        if op.opcode.is_guard() {
            write!(out, "{:?}(", op.opcode).unwrap();
        } else if op.type_ != Type::Void {
            write!(out, "v{} = {:?}(", op.pos.raw(), op.opcode).unwrap();
        } else {
            write!(out, "{:?}(", op.opcode).unwrap();
        }
        for (i, arg) in op.args.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            if let Some(&val) = constants.get(&arg.raw()) {
                write!(out, "{val}").unwrap();
            } else {
                write!(out, "v{}", arg.raw()).unwrap();
            }
        }
        write!(out, ")").unwrap();
        // Render descriptor if present (parity with RPython's logger repr_of_descr)
        if let Some(ref descr) = op.descr {
            let repr = descr.repr();
            if !repr.is_empty() {
                write!(out, " descr=<{repr}>").unwrap();
            }
        }
        if let Some(ref fa) = op.fail_args {
            write!(out, " [").unwrap();
            for (i, arg) in fa.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                if let Some(&val) = constants.get(&arg.raw()) {
                    write!(out, "{val}").unwrap();
                } else {
                    write!(out, "v{}", arg.raw()).unwrap();
                }
            }
            write!(out, "]").unwrap();
        }
        writeln!(out).unwrap();
    }
    out
}

/// All JIT IR opcodes.
///
/// Faithfully mirrors rpython/jit/metainterp/resoperation.py `_oplist`.
/// Operations that produce typed results are expanded with suffixes:
///   _I (int), _R (ref/pointer), _F (float), _N (void/none)
///
/// Boundary markers (e.g., _GUARD_FIRST) are not included as enum variants;
/// instead, classification is done via methods on OpCode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum OpCode {
    // ── Final ──
    Jump = 0,
    Finish,

    Label,

    // ── Guards (foldable) ──
    GuardTrue,
    GuardFalse,
    VecGuardTrue,
    VecGuardFalse,
    GuardValue,
    GuardClass,
    GuardNonnull,
    GuardIsnull,
    GuardNonnullClass,
    GuardGcType,
    GuardIsObject,
    GuardSubclass,
    // ── Guards (non-foldable) ──
    GuardNoException,
    GuardException,
    GuardNoOverflow,
    GuardOverflow,
    GuardNotForced,
    GuardNotForced2,
    GuardNotInvalidated,
    GuardFutureCondition,
    GuardAlwaysFails,

    // ── Always pure: integer arithmetic ──
    IntAdd,
    IntSub,
    IntMul,
    UintMulHigh,
    IntFloorDiv,
    IntMod,
    IntAnd,
    IntOr,
    IntXor,
    IntRshift,
    IntLshift,
    UintRshift,
    IntSignext,

    // ── Always pure: float arithmetic ──
    FloatAdd,
    FloatSub,
    FloatMul,
    FloatTrueDiv,
    FloatFloorDiv,
    FloatMod,
    FloatNeg,
    FloatAbs,

    // ── Always pure: casts ──
    CastFloatToInt,
    CastIntToFloat,
    CastFloatToSinglefloat,
    CastSinglefloatToFloat,
    ConvertFloatBytesToLonglong,
    ConvertLonglongBytesToFloat,

    // ── Always pure: vector arithmetic ──
    VecIntAdd,
    VecIntSub,
    VecIntMul,
    VecIntAnd,
    VecIntOr,
    VecIntXor,
    VecFloatAdd,
    VecFloatSub,
    VecFloatMul,
    VecFloatTrueDiv,
    VecFloatNeg,
    VecFloatAbs,

    // ── Always pure: vector comparisons / casts ──
    VecFloatEq,
    VecFloatNe,
    VecFloatXor,
    VecIntIsTrue,
    VecIntNe,
    VecIntEq,
    VecIntSignext,
    VecCastFloatToSinglefloat,
    VecCastSinglefloatToFloat,
    VecCastFloatToInt,
    VecCastIntToFloat,

    // ── Always pure: vector pack/unpack ──
    VecI,
    VecF,
    VecUnpackI,
    VecUnpackF,
    VecPackI,
    VecPackF,
    VecExpandI,
    VecExpandF,

    // ── Always pure: integer comparisons ──
    IntLt,
    IntLe,
    IntEq,
    IntNe,
    IntGt,
    IntGe,
    UintLt,
    UintLe,
    UintGt,
    UintGe,

    // ── Always pure: float comparisons ──
    FloatLt,
    FloatLe,
    FloatEq,
    FloatNe,
    FloatGt,
    FloatGe,

    // ── Always pure: unary int ──
    IntIsZero,
    IntIsTrue,
    IntNeg,
    IntInvert,
    IntForceGeZero,
    IntBetween,

    // ── Always pure: identity / cast ──
    SameAsI,
    SameAsR,
    SameAsF,
    CastPtrToInt,
    CastIntToPtr,
    CastOpaquePtr,

    // ── Always pure: pointer comparisons ──
    PtrEq,
    PtrNe,
    InstancePtrEq,
    InstancePtrNe,
    NurseryPtrIncrement,

    // ── Always pure: array/string length, getitem ──
    ArraylenGc,
    Strlen,
    Strgetitem,
    GetarrayitemGcPureI,
    GetarrayitemGcPureR,
    GetarrayitemGcPureF,
    Unicodelen,
    Unicodegetitem,

    // ── Always pure: backend-specific loads ──
    LoadFromGcTable,
    LoadEffectiveAddress,

    // ── Thread-local reference ──
    ThreadlocalrefGet,

    // ── No side effect (but not always pure) ──
    GcLoadI,
    GcLoadR,
    GcLoadF,
    GcLoadIndexedI,
    GcLoadIndexedR,
    GcLoadIndexedF,

    // ── Raw loads ──
    GetarrayitemGcI,
    GetarrayitemGcR,
    GetarrayitemGcF,
    GetarrayitemRawI,
    GetarrayitemRawR,
    GetarrayitemRawF,
    RawLoadI,
    RawLoadF,
    VecLoadI,
    VecLoadF,

    // ── No side effect: field/interior access ──
    GetinteriorfieldGcI,
    GetinteriorfieldGcR,
    GetinteriorfieldGcF,
    GetfieldGcI,
    GetfieldGcR,
    GetfieldGcF,
    GetfieldRawI,
    GetfieldRawR,
    GetfieldRawF,

    // ── No side effect: pure field access (immutable) ──
    GetfieldGcPureI,
    GetfieldGcPureR,
    GetfieldGcPureF,

    // ── Allocation ──
    New,
    NewWithVtable,
    NewArray,
    NewArrayClear,
    Newstr,
    Newunicode,

    // ── No side effect: misc ──
    ForceToken,
    VirtualRefI,
    VirtualRefR,
    Strhash,
    Unicodehash,

    // ── Side effects: GC stores ──
    GcStore,
    GcStoreIndexed,

    // ── Side effects: misc ──
    IncrementDebugCounter,

    // ── Raw stores ──
    SetarrayitemGc,
    SetarrayitemRaw,
    RawStore,
    VecStore,

    // ── Side effects: field/interior stores ──
    SetinteriorfieldGc,
    SetinteriorfieldRaw,
    SetfieldGc,
    ZeroArray,
    SetfieldRaw,
    Strsetitem,
    Unicodesetitem,

    // ── GC write barriers ──
    CondCallGcWb,
    CondCallGcWbArray,

    // ── Debug ──
    DebugMergePoint,
    EnterPortalFrame,
    LeavePortalFrame,
    JitDebug,

    // ── Testing only ──
    ForceSpill,

    // ── Misc side effects ──
    VirtualRefFinish,
    Copystrcontent,
    Copyunicodecontent,
    QuasiimmutField,
    AssertNotNone,
    RecordExactClass,
    RecordExactValueR,
    RecordExactValueI,
    Keepalive,
    SaveException,
    SaveExcClass,
    RestoreException,

    // ── Calls (can raise) ──
    CallI,
    CallR,
    CallF,
    CallN,
    CondCallN,
    CondCallValueI,
    CondCallValueR,
    CallAssemblerI,
    CallAssemblerR,
    CallAssemblerF,
    CallAssemblerN,
    CallMayForceI,
    CallMayForceR,
    CallMayForceF,
    CallMayForceN,
    CallLoopinvariantI,
    CallLoopinvariantR,
    CallLoopinvariantF,
    CallLoopinvariantN,
    CallReleaseGilI,
    // CallReleaseGilR intentionally absent: resoperation.py:1243-1244
    // (`# no such thing`) excludes CALL_RELEASE_GIL_R from the upstream
    // opcode table.
    CallReleaseGilF,
    CallReleaseGilN,
    CallPureI,
    CallPureR,
    CallPureF,
    CallPureN,
    CheckMemoryError,
    CallMallocNursery,
    CallMallocNurseryVarsize,
    CallMallocNurseryVarsizeFrame,
    RecordKnownResult,

    // ── Overflow ──
    IntAddOvf,
    IntSubOvf,
    IntMulOvf,
}

// ── Boundary constants for category classification ──
// These correspond to the _FIRST/_LAST markers in resoperation.py.

const FINAL_FIRST: u16 = OpCode::Jump as u16;
const FINAL_LAST: u16 = OpCode::Finish as u16;

const GUARD_FIRST: u16 = OpCode::GuardTrue as u16;
const GUARD_FOLDABLE_FIRST: u16 = OpCode::GuardTrue as u16;
const GUARD_FOLDABLE_LAST: u16 = OpCode::GuardSubclass as u16;
const GUARD_LAST: u16 = OpCode::GuardAlwaysFails as u16;

const ALWAYS_PURE_FIRST: u16 = OpCode::IntAdd as u16;
const ALWAYS_PURE_LAST: u16 = OpCode::LoadEffectiveAddress as u16;

const NOSIDEEFFECT_FIRST: u16 = OpCode::IntAdd as u16; // same as ALWAYS_PURE_FIRST
const NOSIDEEFFECT_LAST: u16 = OpCode::Unicodehash as u16;

const MALLOC_FIRST: u16 = OpCode::New as u16;
const MALLOC_LAST: u16 = OpCode::Newunicode as u16;

const RAW_LOAD_FIRST: u16 = OpCode::GetarrayitemGcI as u16;
const RAW_LOAD_LAST: u16 = OpCode::VecLoadF as u16;

const RAW_STORE_FIRST: u16 = OpCode::SetarrayitemGc as u16;
const RAW_STORE_LAST: u16 = OpCode::VecStore as u16;

const JIT_DEBUG_FIRST: u16 = OpCode::DebugMergePoint as u16;
const JIT_DEBUG_LAST: u16 = OpCode::JitDebug as u16;

const CALL_FIRST: u16 = OpCode::CallI as u16;
const CALL_LAST: u16 = OpCode::RecordKnownResult as u16;

const CANRAISE_FIRST: u16 = OpCode::CallI as u16;
const CANRAISE_LAST: u16 = OpCode::IntMulOvf as u16;

const OVF_FIRST: u16 = OpCode::IntAddOvf as u16;
const OVF_LAST: u16 = OpCode::IntMulOvf as u16;

impl OpCode {
    pub fn as_u16(self) -> u16 {
        self as u16
    }

    /// Iterate over all defined OpCode variants (0..OPCODE_COUNT).
    pub fn all() -> impl Iterator<Item = OpCode> {
        (0..OPCODE_COUNT as u16).map(|i| unsafe { std::mem::transmute::<u16, OpCode>(i) })
    }

    /// Safe reverse of `as_u16` — bounds-checked conversion used by the
    /// byte-stream `ByteTraceIter` (opencoder.py:362-406 `next()` reads the
    /// opnum byte and looks it up in the `OP_*` registry; in RPython this
    /// is the `opnum` → class-table mapping). Returns `None` when `n`
    /// lies outside the defined `0..OPCODE_COUNT` range.
    pub fn from_u16(n: u16) -> Option<OpCode> {
        if (n as usize) < OPCODE_COUNT {
            // SAFETY: `OpCode` is `#[repr(u16)]` with contiguous variants
            // in `0..OPCODE_COUNT` — any value in that range is a
            // well-defined enum bit pattern (the same rationale used by
            // `OpCode::all` above).
            Some(unsafe { std::mem::transmute::<u16, OpCode>(n) })
        } else {
            None
        }
    }

    // ── Category classification (mirrors rop.is_* static methods) ──

    pub fn is_final(self) -> bool {
        let n = self.as_u16();
        FINAL_FIRST <= n && n <= FINAL_LAST
    }

    pub fn is_guard(self) -> bool {
        let n = self.as_u16();
        GUARD_FIRST <= n && n <= GUARD_LAST
    }

    pub fn is_foldable_guard(self) -> bool {
        let n = self.as_u16();
        GUARD_FOLDABLE_FIRST <= n && n <= GUARD_FOLDABLE_LAST
    }

    pub fn is_always_pure(self) -> bool {
        let n = self.as_u16();
        (ALWAYS_PURE_FIRST <= n && n <= ALWAYS_PURE_LAST)
            || matches!(
                self,
                OpCode::GetfieldGcPureI | OpCode::GetfieldGcPureR | OpCode::GetfieldGcPureF
            )
    }

    pub fn has_no_side_effect(self) -> bool {
        let n = self.as_u16();
        (NOSIDEEFFECT_FIRST <= n && n <= NOSIDEEFFECT_LAST)
            || matches!(
                self,
                OpCode::GetfieldGcPureI | OpCode::GetfieldGcPureR | OpCode::GetfieldGcPureF
            )
    }

    pub fn is_malloc(self) -> bool {
        let n = self.as_u16();
        MALLOC_FIRST <= n && n <= MALLOC_LAST
    }

    pub fn is_call(self) -> bool {
        let n = self.as_u16();
        CALL_FIRST <= n && n <= CALL_LAST
    }

    pub fn can_raise(self) -> bool {
        let n = self.as_u16();
        CANRAISE_FIRST <= n && n <= CANRAISE_LAST
    }

    pub fn can_malloc(self) -> bool {
        self.is_call() || self.is_malloc()
    }

    pub fn is_ovf(self) -> bool {
        let n = self.as_u16();
        OVF_FIRST <= n && n <= OVF_LAST
    }

    pub fn is_raw_load(self) -> bool {
        let n = self.as_u16();
        RAW_LOAD_FIRST < n && n < RAW_LOAD_LAST
    }

    pub fn is_raw_store(self) -> bool {
        let n = self.as_u16();
        RAW_STORE_FIRST < n && n < RAW_STORE_LAST
    }

    pub fn is_jit_debug(self) -> bool {
        let n = self.as_u16();
        JIT_DEBUG_FIRST <= n && n <= JIT_DEBUG_LAST
    }

    pub fn is_comparison(self) -> bool {
        self.is_always_pure() && self.returns_bool()
    }

    pub fn is_guard_exception(self) -> bool {
        matches!(self, OpCode::GuardException | OpCode::GuardNoException)
    }

    pub fn is_guard_overflow(self) -> bool {
        matches!(self, OpCode::GuardOverflow | OpCode::GuardNoOverflow)
    }

    pub fn is_same_as(self) -> bool {
        matches!(self, OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF)
    }

    pub fn is_getfield(self) -> bool {
        matches!(
            self,
            OpCode::GetfieldGcI
                | OpCode::GetfieldGcR
                | OpCode::GetfieldGcF
                | OpCode::GetfieldGcPureI
                | OpCode::GetfieldGcPureR
                | OpCode::GetfieldGcPureF
        )
    }

    pub fn is_getarrayitem(self) -> bool {
        matches!(
            self,
            OpCode::GetarrayitemGcI
                | OpCode::GetarrayitemGcR
                | OpCode::GetarrayitemGcF
                | OpCode::GetarrayitemGcPureI
                | OpCode::GetarrayitemGcPureR
                | OpCode::GetarrayitemGcPureF
        )
    }

    pub fn is_setarrayitem(self) -> bool {
        matches!(self, OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw)
    }

    pub fn is_setfield(self) -> bool {
        matches!(self, OpCode::SetfieldGc | OpCode::SetfieldRaw)
    }

    pub fn is_getinteriorfield(self) -> bool {
        matches!(
            self,
            OpCode::GetinteriorfieldGcI | OpCode::GetinteriorfieldGcR | OpCode::GetinteriorfieldGcF
        )
    }

    pub fn is_setinteriorfield(self) -> bool {
        matches!(self, OpCode::SetinteriorfieldGc)
    }

    pub fn is_plain_call(self) -> bool {
        matches!(
            self,
            OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN
        )
    }

    pub fn is_call_assembler(self) -> bool {
        matches!(
            self,
            OpCode::CallAssemblerI
                | OpCode::CallAssemblerR
                | OpCode::CallAssemblerF
                | OpCode::CallAssemblerN
        )
    }

    pub fn is_call_may_force(self) -> bool {
        matches!(
            self,
            OpCode::CallMayForceI
                | OpCode::CallMayForceR
                | OpCode::CallMayForceF
                | OpCode::CallMayForceN
        )
    }

    pub fn is_call_pure(self) -> bool {
        matches!(
            self,
            OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF | OpCode::CallPureN
        )
    }

    pub fn is_call_release_gil(self) -> bool {
        // resoperation.py:1238-1248 call_release_gil_for_descr maps
        // 'i'/'f'/'v' only; 'r' is `# no such thing`.
        matches!(
            self,
            OpCode::CallReleaseGilI | OpCode::CallReleaseGilF | OpCode::CallReleaseGilN
        )
    }

    pub fn is_call_loopinvariant(self) -> bool {
        matches!(
            self,
            OpCode::CallLoopinvariantI
                | OpCode::CallLoopinvariantR
                | OpCode::CallLoopinvariantF
                | OpCode::CallLoopinvariantN
        )
    }

    pub fn is_cond_call_value(self) -> bool {
        matches!(self, OpCode::CondCallValueI | OpCode::CondCallValueR)
    }

    pub fn is_label(self) -> bool {
        matches!(self, OpCode::Label)
    }

    pub fn is_vector_arithmetic(self) -> bool {
        matches!(
            self,
            OpCode::VecIntAdd
                | OpCode::VecIntSub
                | OpCode::VecIntMul
                | OpCode::VecIntAnd
                | OpCode::VecIntOr
                | OpCode::VecIntXor
                | OpCode::VecFloatAdd
                | OpCode::VecFloatSub
                | OpCode::VecFloatMul
                | OpCode::VecFloatTrueDiv
                | OpCode::VecFloatNeg
                | OpCode::VecFloatAbs
        )
    }

    /// Expected number of arguments, or None for variadic.
    pub fn arity(self) -> Option<u8> {
        OPARITY[self.as_u16() as usize]
    }

    /// Whether this operation takes a descriptor.
    pub fn has_descr(self) -> bool {
        OPWITHDESCR[self.as_u16() as usize]
    }

    /// Whether this operation produces a boolean result.
    pub fn returns_bool(self) -> bool {
        OPBOOL[self.as_u16() as usize]
    }

    /// Result type of this operation.
    pub fn result_type(self) -> Type {
        OPRESTYPE[self.as_u16() as usize]
    }

    /// Name of this operation (for debugging).
    pub fn name(self) -> &'static str {
        OPNAME[self.as_u16() as usize]
    }
}

// ── Typed dispatch helpers (mirrors rop.*_for_descr) ──

impl OpCode {
    pub fn call_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallI,
            Type::Ref => OpCode::CallR,
            Type::Float => OpCode::CallF,
            Type::Void => OpCode::CallN,
        }
    }

    pub fn call_pure_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallPureI,
            Type::Ref => OpCode::CallPureR,
            Type::Float => OpCode::CallPureF,
            Type::Void => OpCode::CallPureN,
        }
    }

    pub fn call_may_force_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallMayForceI,
            Type::Ref => OpCode::CallMayForceR,
            Type::Float => OpCode::CallMayForceF,
            Type::Void => OpCode::CallMayForceN,
        }
    }

    pub fn call_assembler_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallAssemblerI,
            Type::Ref => OpCode::CallAssemblerR,
            Type::Float => OpCode::CallAssemblerF,
            Type::Void => OpCode::CallAssemblerN,
        }
    }

    pub fn call_loopinvariant_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallLoopinvariantI,
            Type::Ref => OpCode::CallLoopinvariantR,
            Type::Float => OpCode::CallLoopinvariantF,
            Type::Void => OpCode::CallLoopinvariantN,
        }
    }

    /// Mirrors `resoperation.py:1238-1248 call_release_gil_for_descr`:
    /// the `'r'` arm is explicitly commented out as `# no such thing`,
    /// so a `Type::Ref` result-typed release-gil callee has no upstream
    /// opcode mapping.  Panic rather than returning `CallReleaseGilR`,
    /// which has no producer in upstream and would record an IR op the
    /// optimizer/backend cannot consume.
    pub fn call_release_gil_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::CallReleaseGilI,
            Type::Ref => panic!(
                "call_release_gil_for_type: Type::Ref has no upstream counterpart \
                 (resoperation.py:1243-1244 `# no such thing`); CALL_RELEASE_GIL_R \
                 has no producer in RPython"
            ),
            Type::Float => OpCode::CallReleaseGilF,
            Type::Void => OpCode::CallReleaseGilN,
        }
    }

    pub fn same_as_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Int => OpCode::SameAsI,
            Type::Ref => OpCode::SameAsR,
            Type::Float => OpCode::SameAsF,
            Type::Void => unreachable!("same_as has no void variant"),
        }
    }

    pub fn getfield_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Ref => OpCode::GetfieldGcR,
            Type::Float => OpCode::GetfieldGcF,
            _ => OpCode::GetfieldGcI,
        }
    }

    pub fn getarrayitem_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Ref => OpCode::GetarrayitemGcR,
            Type::Float => OpCode::GetarrayitemGcF,
            _ => OpCode::GetarrayitemGcI,
        }
    }

    pub fn getfield_raw_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Ref => OpCode::GetfieldRawR,
            Type::Float => OpCode::GetfieldRawF,
            _ => OpCode::GetfieldRawI,
        }
    }

    pub fn getarrayitem_raw_for_type(tp: Type) -> OpCode {
        match tp {
            Type::Ref => OpCode::GetarrayitemRawR,
            Type::Float => OpCode::GetarrayitemRawF,
            _ => OpCode::GetarrayitemRawI,
        }
    }
}

// ── Boolean inverse/reflex tables (from resoperation.py) ──

impl OpCode {
    /// Returns the boolean inverse of a comparison, e.g. INT_EQ -> INT_NE.
    pub fn bool_inverse(self) -> Option<OpCode> {
        match self {
            OpCode::IntEq => Some(OpCode::IntNe),
            OpCode::IntNe => Some(OpCode::IntEq),
            OpCode::IntLt => Some(OpCode::IntGe),
            OpCode::IntGe => Some(OpCode::IntLt),
            OpCode::IntGt => Some(OpCode::IntLe),
            OpCode::IntLe => Some(OpCode::IntGt),
            OpCode::UintLt => Some(OpCode::UintGe),
            OpCode::UintGe => Some(OpCode::UintLt),
            OpCode::UintGt => Some(OpCode::UintLe),
            OpCode::UintLe => Some(OpCode::UintGt),
            OpCode::FloatEq => Some(OpCode::FloatNe),
            OpCode::FloatNe => Some(OpCode::FloatEq),
            OpCode::FloatLt => Some(OpCode::FloatGe),
            OpCode::FloatGe => Some(OpCode::FloatLt),
            OpCode::FloatGt => Some(OpCode::FloatLe),
            OpCode::FloatLe => Some(OpCode::FloatGt),
            OpCode::PtrEq => Some(OpCode::PtrNe),
            OpCode::PtrNe => Some(OpCode::PtrEq),
            _ => None,
        }
    }

    /// Returns the reflexive form of a comparison (swap operands),
    /// e.g. INT_LT -> INT_GT.
    pub fn bool_reflex(self) -> Option<OpCode> {
        match self {
            OpCode::IntEq => Some(OpCode::IntEq),
            OpCode::IntNe => Some(OpCode::IntNe),
            OpCode::IntLt => Some(OpCode::IntGt),
            OpCode::IntGe => Some(OpCode::IntLe),
            OpCode::IntGt => Some(OpCode::IntLt),
            OpCode::IntLe => Some(OpCode::IntGe),
            OpCode::UintLt => Some(OpCode::UintGt),
            OpCode::UintGe => Some(OpCode::UintLe),
            OpCode::UintGt => Some(OpCode::UintLt),
            OpCode::UintLe => Some(OpCode::UintGe),
            OpCode::FloatEq => Some(OpCode::FloatEq),
            OpCode::FloatNe => Some(OpCode::FloatNe),
            OpCode::FloatLt => Some(OpCode::FloatGt),
            OpCode::FloatGe => Some(OpCode::FloatLe),
            OpCode::FloatGt => Some(OpCode::FloatLt),
            OpCode::FloatLe => Some(OpCode::FloatGe),
            OpCode::PtrEq => Some(OpCode::PtrEq),
            OpCode::PtrNe => Some(OpCode::PtrNe),
            _ => None,
        }
    }

    /// Maps a scalar op to its vector equivalent, e.g. INT_ADD -> VEC_INT_ADD.
    pub fn to_vector(self) -> Option<OpCode> {
        match self {
            OpCode::IntAdd => Some(OpCode::VecIntAdd),
            OpCode::IntSub => Some(OpCode::VecIntSub),
            OpCode::IntMul => Some(OpCode::VecIntMul),
            OpCode::IntAnd => Some(OpCode::VecIntAnd),
            OpCode::IntOr => Some(OpCode::VecIntOr),
            OpCode::IntXor => Some(OpCode::VecIntXor),
            OpCode::FloatAdd => Some(OpCode::VecFloatAdd),
            OpCode::FloatSub => Some(OpCode::VecFloatSub),
            OpCode::FloatMul => Some(OpCode::VecFloatMul),
            OpCode::FloatTrueDiv => Some(OpCode::VecFloatTrueDiv),
            OpCode::FloatAbs => Some(OpCode::VecFloatAbs),
            OpCode::FloatNeg => Some(OpCode::VecFloatNeg),
            OpCode::FloatEq => Some(OpCode::VecFloatEq),
            OpCode::FloatNe => Some(OpCode::VecFloatNe),
            OpCode::IntIsTrue => Some(OpCode::VecIntIsTrue),
            OpCode::IntEq => Some(OpCode::VecIntEq),
            OpCode::IntNe => Some(OpCode::VecIntNe),
            OpCode::IntSignext => Some(OpCode::VecIntSignext),
            OpCode::CastFloatToSinglefloat => Some(OpCode::VecCastFloatToSinglefloat),
            OpCode::CastSinglefloatToFloat => Some(OpCode::VecCastSinglefloatToFloat),
            OpCode::CastIntToFloat => Some(OpCode::VecCastIntToFloat),
            OpCode::CastFloatToInt => Some(OpCode::VecCastFloatToInt),
            OpCode::GuardTrue => Some(OpCode::VecGuardTrue),
            OpCode::GuardFalse => Some(OpCode::VecGuardFalse),
            _ => None,
        }
    }

    /// The non-overflow version of an overflow op, e.g. INT_ADD_OVF -> INT_ADD.
    pub fn without_overflow(self) -> Option<OpCode> {
        match self {
            OpCode::IntAddOvf => Some(OpCode::IntAdd),
            OpCode::IntSubOvf => Some(OpCode::IntSub),
            OpCode::IntMulOvf => Some(OpCode::IntMul),
            _ => None,
        }
    }

    /// Whether this opcode accesses memory (load/store).
    pub fn is_memory_access(self) -> bool {
        matches!(
            self,
            // Typed getfield
            OpCode::GetfieldGcI
                | OpCode::GetfieldGcR
                | OpCode::GetfieldGcF
                | OpCode::GetfieldRawI
                | OpCode::GetfieldRawR
                | OpCode::GetfieldRawF
                | OpCode::GetfieldGcPureI
                | OpCode::GetfieldGcPureR
                | OpCode::GetfieldGcPureF
                // Untyped setfield
                | OpCode::SetfieldGc
                | OpCode::SetfieldRaw
                // Typed getarrayitem
                | OpCode::GetarrayitemGcI
                | OpCode::GetarrayitemGcR
                | OpCode::GetarrayitemGcF
                | OpCode::GetarrayitemGcPureI
                | OpCode::GetarrayitemGcPureR
                | OpCode::GetarrayitemGcPureF
                | OpCode::GetarrayitemRawI
                | OpCode::GetarrayitemRawR
                | OpCode::GetarrayitemRawF
                // Untyped setarrayitem
                | OpCode::SetarrayitemGc
                | OpCode::SetarrayitemRaw
                // Raw load/store
                | OpCode::RawLoadI
                | OpCode::RawLoadF
                | OpCode::RawStore
                // GC load (typed)
                | OpCode::GcLoadI
                | OpCode::GcLoadR
                | OpCode::GcLoadF
                | OpCode::GcLoadIndexedI
                | OpCode::GcLoadIndexedR
                | OpCode::GcLoadIndexedF
                // GC store (untyped)
                | OpCode::GcStore
                | OpCode::GcStoreIndexed
        )
    }

    /// dependency.py:207-208: loads_from_complex_object
    /// (ALWAYS_PURE_LAST <= opnum < MALLOC_FIRST in RPython)
    pub fn is_complex_load(self) -> bool {
        matches!(
            self,
            OpCode::GetarrayitemGcI
                | OpCode::GetarrayitemGcR
                | OpCode::GetarrayitemGcF
                | OpCode::GetarrayitemGcPureI
                | OpCode::GetarrayitemGcPureR
                | OpCode::GetarrayitemGcPureF
                | OpCode::GetarrayitemRawI
                | OpCode::GetarrayitemRawF
                | OpCode::RawLoadI
                | OpCode::RawLoadF
                | OpCode::GetfieldGcI
                | OpCode::GetfieldGcR
                | OpCode::GetfieldGcF
                | OpCode::GetfieldRawI
                | OpCode::GetfieldRawR
                | OpCode::GetfieldRawF
                | OpCode::GetinteriorfieldGcI
                | OpCode::GetinteriorfieldGcF
                | OpCode::GetinteriorfieldGcR
        )
    }

    /// dependency.py:210-211: modifies_complex_object
    /// (SETARRAYITEM_GC <= opnum <= UNICODESETITEM)
    pub fn is_complex_modify(self) -> bool {
        matches!(
            self,
            OpCode::SetarrayitemGc
                | OpCode::SetarrayitemRaw
                | OpCode::RawStore
                | OpCode::SetinteriorfieldGc
                | OpCode::SetinteriorfieldRaw
                | OpCode::SetfieldGc
                | OpCode::SetfieldRaw
                | OpCode::ZeroArray
                | OpCode::Strsetitem
                | OpCode::Unicodesetitem
        )
    }
}

// ── Metadata tables ──
// These are generated to match the setup() function in resoperation.py.
// Format: arity (None = variadic), has_descr, returns_bool, result_type, name.

macro_rules! opcode_count {
    () => {
        OpCode::IntMulOvf as usize + 1
    };
}

/// Number of defined opcodes.
pub const OPCODE_COUNT: usize = opcode_count!();

// We use include! or manual arrays. For now, manual tables.
// These tables are indexed by OpCode as u16.

/// Arity: Some(n) for fixed arity, None for variadic.
static OPARITY: [Option<u8>; OPCODE_COUNT] = {
    let mut t = [None; OPCODE_COUNT];
    use OpCode::*;
    // Variadic ops (arity = *)
    // Jump, Finish, Label, DebugMergePoint, JitDebug, Escape*, all Calls, CondCall*, RecordKnownResult
    // are variadic -> None (already default)

    // Fixed arity ops
    macro_rules! set {
        ($op:ident, $a:expr) => {
            t[$op as usize] = Some($a);
        };
    }
    // Guards
    set!(GuardTrue, 1);
    set!(GuardFalse, 1);
    set!(VecGuardTrue, 1);
    set!(VecGuardFalse, 1);
    set!(GuardValue, 2);
    set!(GuardClass, 2);
    set!(GuardNonnull, 1);
    set!(GuardIsnull, 1);
    set!(GuardNonnullClass, 2);
    set!(GuardGcType, 2);
    set!(GuardIsObject, 1);
    set!(GuardSubclass, 2);
    set!(GuardNoException, 0);
    set!(GuardException, 1);
    set!(GuardNoOverflow, 0);
    set!(GuardOverflow, 0);
    set!(GuardNotForced, 0);
    set!(GuardNotForced2, 0);
    set!(GuardNotInvalidated, 0);
    set!(GuardFutureCondition, 0);
    set!(GuardAlwaysFails, 0);
    // Arithmetic (binary)
    set!(IntAdd, 2);
    set!(IntSub, 2);
    set!(IntMul, 2);
    set!(UintMulHigh, 2);
    set!(IntFloorDiv, 2);
    set!(IntMod, 2);
    set!(IntAnd, 2);
    set!(IntOr, 2);
    set!(IntXor, 2);
    set!(IntRshift, 2);
    set!(IntLshift, 2);
    set!(UintRshift, 2);
    set!(IntSignext, 2);
    set!(FloatAdd, 2);
    set!(FloatSub, 2);
    set!(FloatMul, 2);
    set!(FloatTrueDiv, 2);
    set!(FloatFloorDiv, 2);
    set!(FloatMod, 2);
    set!(FloatNeg, 1);
    set!(FloatAbs, 1);
    // Casts (unary)
    set!(CastFloatToInt, 1);
    set!(CastIntToFloat, 1);
    set!(CastFloatToSinglefloat, 1);
    set!(CastSinglefloatToFloat, 1);
    set!(ConvertFloatBytesToLonglong, 1);
    set!(ConvertLonglongBytesToFloat, 1);
    // Vector arithmetic (binary/unary)
    set!(VecIntAdd, 2);
    set!(VecIntSub, 2);
    set!(VecIntMul, 2);
    set!(VecIntAnd, 2);
    set!(VecIntOr, 2);
    set!(VecIntXor, 2);
    set!(VecFloatAdd, 2);
    set!(VecFloatSub, 2);
    set!(VecFloatMul, 2);
    set!(VecFloatTrueDiv, 2);
    set!(VecFloatNeg, 1);
    set!(VecFloatAbs, 1);
    set!(VecFloatEq, 2);
    set!(VecFloatNe, 2);
    set!(VecFloatXor, 2);
    set!(VecIntIsTrue, 1);
    set!(VecIntNe, 2);
    set!(VecIntEq, 2);
    set!(VecIntSignext, 2);
    set!(VecCastFloatToSinglefloat, 1);
    set!(VecCastSinglefloatToFloat, 1);
    set!(VecCastFloatToInt, 1);
    set!(VecCastIntToFloat, 1);
    set!(VecI, 0);
    set!(VecF, 0);
    set!(VecUnpackI, 3);
    set!(VecUnpackF, 3);
    set!(VecPackI, 4);
    set!(VecPackF, 4);
    set!(VecExpandI, 1);
    set!(VecExpandF, 1);
    // Comparisons
    set!(IntLt, 2);
    set!(IntLe, 2);
    set!(IntEq, 2);
    set!(IntNe, 2);
    set!(IntGt, 2);
    set!(IntGe, 2);
    set!(UintLt, 2);
    set!(UintLe, 2);
    set!(UintGt, 2);
    set!(UintGe, 2);
    set!(FloatLt, 2);
    set!(FloatLe, 2);
    set!(FloatEq, 2);
    set!(FloatNe, 2);
    set!(FloatGt, 2);
    set!(FloatGe, 2);
    // Unary int
    set!(IntIsZero, 1);
    set!(IntIsTrue, 1);
    set!(IntNeg, 1);
    set!(IntInvert, 1);
    set!(IntForceGeZero, 1);
    set!(IntBetween, 3);
    // Identity/cast
    set!(SameAsI, 1);
    set!(SameAsR, 1);
    set!(SameAsF, 1);
    set!(CastPtrToInt, 1);
    set!(CastIntToPtr, 1);
    set!(CastOpaquePtr, 1);
    // Pointer comparisons
    set!(PtrEq, 2);
    set!(PtrNe, 2);
    set!(InstancePtrEq, 2);
    set!(InstancePtrNe, 2);
    set!(NurseryPtrIncrement, 2);
    // Array/string length
    set!(ArraylenGc, 1);
    set!(Strlen, 1);
    set!(Strgetitem, 2);
    set!(GetarrayitemGcPureI, 2);
    set!(GetarrayitemGcPureR, 2);
    set!(GetarrayitemGcPureF, 2);
    set!(Unicodelen, 1);
    set!(Unicodegetitem, 2);
    set!(LoadFromGcTable, 1);
    set!(LoadEffectiveAddress, 4);
    // Thread-local
    set!(ThreadlocalrefGet, 0);
    // GC load
    set!(GcLoadI, 3);
    set!(GcLoadR, 3);
    set!(GcLoadF, 3);
    set!(GcLoadIndexedI, 5);
    set!(GcLoadIndexedR, 5);
    set!(GcLoadIndexedF, 5);
    // Array/field get
    set!(GetarrayitemGcI, 2);
    set!(GetarrayitemGcR, 2);
    set!(GetarrayitemGcF, 2);
    set!(GetarrayitemRawI, 2);
    set!(GetarrayitemRawR, 2);
    set!(GetarrayitemRawF, 2);
    set!(RawLoadI, 2);
    set!(RawLoadF, 2);
    set!(VecLoadI, 4);
    set!(VecLoadF, 4);
    set!(GetinteriorfieldGcI, 2);
    set!(GetinteriorfieldGcR, 2);
    set!(GetinteriorfieldGcF, 2);
    set!(GetfieldGcI, 1);
    set!(GetfieldGcR, 1);
    set!(GetfieldGcF, 1);
    set!(GetfieldRawI, 1);
    set!(GetfieldRawR, 1);
    set!(GetfieldRawF, 1);
    set!(GetfieldGcPureI, 1);
    set!(GetfieldGcPureR, 1);
    set!(GetfieldGcPureF, 1);
    // Allocation
    set!(New, 0);
    set!(NewWithVtable, 0);
    set!(NewArray, 1);
    set!(NewArrayClear, 1);
    set!(Newstr, 1);
    set!(Newunicode, 1);
    // Misc no-side-effect
    set!(ForceToken, 0);
    set!(VirtualRefI, 2);
    set!(VirtualRefR, 2);
    set!(Strhash, 1);
    set!(Unicodehash, 1);
    // GC store
    set!(GcStore, 4);
    set!(GcStoreIndexed, 6);
    set!(IncrementDebugCounter, 1);
    // Array/field set
    set!(SetarrayitemGc, 3);
    set!(SetarrayitemRaw, 3);
    set!(RawStore, 3);
    set!(VecStore, 5);
    set!(SetinteriorfieldGc, 3);
    set!(SetinteriorfieldRaw, 3);
    set!(SetfieldGc, 2);
    set!(ZeroArray, 5);
    set!(SetfieldRaw, 2);
    set!(Strsetitem, 3);
    set!(Unicodesetitem, 3);
    // GC write barriers
    set!(CondCallGcWb, 1);
    set!(CondCallGcWbArray, 2);
    // Debug (variadic) - already None
    // Portal frames
    set!(EnterPortalFrame, 2);
    set!(LeavePortalFrame, 1);
    // Misc
    set!(ForceSpill, 1);
    set!(VirtualRefFinish, 2);
    set!(Copystrcontent, 5);
    set!(Copyunicodecontent, 5);
    set!(QuasiimmutField, 1);
    set!(AssertNotNone, 1);
    set!(RecordExactClass, 2);
    set!(RecordExactValueR, 2);
    set!(RecordExactValueI, 2);
    set!(Keepalive, 1);
    set!(SaveException, 0);
    set!(SaveExcClass, 0);
    set!(RestoreException, 2);
    // Calls: all variadic (None) - default
    set!(CheckMemoryError, 1);
    set!(CallMallocNursery, 1);
    set!(CallMallocNurseryVarsizeFrame, 1);
    // Overflow
    set!(IntAddOvf, 2);
    set!(IntSubOvf, 2);
    set!(IntMulOvf, 2);
    t
};

/// Whether the operation takes a descriptor.
static OPWITHDESCR: [bool; OPCODE_COUNT] = {
    let mut t = [false; OPCODE_COUNT];
    use OpCode::*;
    macro_rules! set {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = true;)+
        };
    }
    set!(
        Jump,
        Finish,
        Label,
        // Guards
        GuardTrue,
        GuardFalse,
        VecGuardTrue,
        VecGuardFalse,
        GuardValue,
        GuardClass,
        GuardNonnull,
        GuardIsnull,
        GuardNonnullClass,
        GuardGcType,
        GuardIsObject,
        GuardSubclass,
        GuardNoException,
        GuardException,
        GuardNoOverflow,
        GuardOverflow,
        GuardNotForced,
        GuardNotForced2,
        GuardNotInvalidated,
        GuardFutureCondition,
        GuardAlwaysFails,
        // Array/field access
        ArraylenGc,
        GetarrayitemGcPureI,
        GetarrayitemGcPureR,
        GetarrayitemGcPureF,
        GetarrayitemGcI,
        GetarrayitemGcR,
        GetarrayitemGcF,
        GetarrayitemRawI,
        GetarrayitemRawR,
        GetarrayitemRawF,
        RawLoadI,
        RawLoadF,
        VecLoadI,
        VecLoadF,
        GetinteriorfieldGcI,
        GetinteriorfieldGcR,
        GetinteriorfieldGcF,
        GetfieldGcI,
        GetfieldGcR,
        GetfieldGcF,
        GetfieldRawI,
        GetfieldRawR,
        GetfieldRawF,
        GetfieldGcPureI,
        GetfieldGcPureR,
        GetfieldGcPureF,
        // Allocation
        New,
        NewWithVtable,
        NewArray,
        NewArrayClear,
        // Stores
        GcStore,
        GcStoreIndexed,
        SetarrayitemGc,
        SetarrayitemRaw,
        RawStore,
        VecStore,
        SetinteriorfieldGc,
        SetinteriorfieldRaw,
        SetfieldGc,
        ZeroArray,
        SetfieldRaw,
        // GC barriers
        CondCallGcWb,
        CondCallGcWbArray,
        // Misc
        QuasiimmutField,
        // Calls
        CallI,
        CallR,
        CallF,
        CallN,
        CondCallN,
        CondCallValueI,
        CondCallValueR,
        CallAssemblerI,
        CallAssemblerR,
        CallAssemblerF,
        CallAssemblerN,
        CallMayForceI,
        CallMayForceR,
        CallMayForceF,
        CallMayForceN,
        CallLoopinvariantI,
        CallLoopinvariantR,
        CallLoopinvariantF,
        CallLoopinvariantN,
        CallReleaseGilI,
        CallReleaseGilF,
        CallReleaseGilN,
        CallPureI,
        CallPureR,
        CallPureF,
        CallPureN,
        CallMallocNurseryVarsize,
        ThreadlocalrefGet,
        RecordKnownResult
    );
    t
};

/// Whether the operation returns a boolean result.
static OPBOOL: [bool; OPCODE_COUNT] = {
    let mut t = [false; OPCODE_COUNT];
    use OpCode::*;
    macro_rules! set {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = true;)+
        };
    }
    set!(
        IntLt,
        IntLe,
        IntEq,
        IntNe,
        IntGt,
        IntGe,
        UintLt,
        UintLe,
        UintGt,
        UintGe,
        FloatLt,
        FloatLe,
        FloatEq,
        FloatNe,
        FloatGt,
        FloatGe,
        IntIsZero,
        IntIsTrue,
        IntBetween,
        PtrEq,
        PtrNe,
        InstancePtrEq,
        InstancePtrNe,
        VecFloatEq,
        VecFloatNe,
        VecIntIsTrue,
        VecIntNe,
        VecIntEq
    );
    t
};

/// Result type of each operation.
static OPRESTYPE: [Type; OPCODE_COUNT] = {
    let mut t = [Type::Void; OPCODE_COUNT];
    use OpCode::*;

    macro_rules! int {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = Type::Int;)+
        };
    }
    macro_rules! float {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = Type::Float;)+
        };
    }
    macro_rules! ref_ {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = Type::Ref;)+
        };
    }

    int!(
        IntAdd,
        IntSub,
        IntMul,
        UintMulHigh,
        IntFloorDiv,
        IntMod,
        IntAnd,
        IntOr,
        IntXor,
        IntRshift,
        IntLshift,
        UintRshift,
        IntSignext,
        CastFloatToInt,
        CastFloatToSinglefloat,
        ConvertFloatBytesToLonglong,
        // Vector int
        VecIntAdd,
        VecIntSub,
        VecIntMul,
        VecIntAnd,
        VecIntOr,
        VecIntXor,
        VecFloatEq,
        VecFloatNe,
        VecIntIsTrue,
        VecIntNe,
        VecIntEq,
        VecIntSignext,
        VecCastFloatToSinglefloat,
        VecCastFloatToInt,
        // Comparisons (all return int)
        IntLt,
        IntLe,
        IntEq,
        IntNe,
        IntGt,
        IntGe,
        UintLt,
        UintLe,
        UintGt,
        UintGe,
        FloatLt,
        FloatLe,
        FloatEq,
        FloatNe,
        FloatGt,
        FloatGe,
        IntIsZero,
        IntIsTrue,
        IntNeg,
        IntInvert,
        IntForceGeZero,
        IntBetween,
        SameAsI,
        CastPtrToInt,
        PtrEq,
        PtrNe,
        InstancePtrEq,
        InstancePtrNe,
        ArraylenGc,
        Strlen,
        Strgetitem,
        GetarrayitemGcPureI,
        Unicodelen,
        Unicodegetitem,
        LoadEffectiveAddress,
        GcLoadI,
        GcLoadIndexedI,
        GetarrayitemGcI,
        GetarrayitemRawI,
        RawLoadI,
        GetinteriorfieldGcI,
        GetfieldGcI,
        GetfieldRawI,
        GetfieldGcPureI,
        Strhash,
        Unicodehash,
        CondCallValueI,
        CallI,
        CallPureI,
        CallMayForceI,
        CallAssemblerI,
        CallLoopinvariantI,
        CallReleaseGilI,
        SaveExcClass,
        RecordExactValueI,
        IntAddOvf,
        IntSubOvf,
        IntMulOvf
    );

    float!(
        FloatAdd,
        FloatSub,
        FloatMul,
        FloatTrueDiv,
        FloatFloorDiv,
        FloatMod,
        FloatNeg,
        FloatAbs,
        CastIntToFloat,
        CastSinglefloatToFloat,
        ConvertLonglongBytesToFloat,
        VecFloatAdd,
        VecFloatSub,
        VecFloatMul,
        VecFloatTrueDiv,
        VecFloatNeg,
        VecFloatAbs,
        VecFloatXor,
        VecCastSinglefloatToFloat,
        VecCastIntToFloat,
        SameAsF,
        GetarrayitemGcPureF,
        GcLoadF,
        GcLoadIndexedF,
        GetarrayitemGcF,
        GetarrayitemRawF,
        RawLoadF,
        GetinteriorfieldGcF,
        GetfieldGcF,
        GetfieldRawF,
        GetfieldGcPureF,
        CallF,
        CallPureF,
        CallMayForceF,
        CallAssemblerF,
        CallLoopinvariantF,
        CallReleaseGilF
    );

    ref_!(
        CastIntToPtr,
        CastOpaquePtr,
        SameAsR,
        NurseryPtrIncrement,
        GetarrayitemGcPureR,
        LoadFromGcTable,
        GcLoadR,
        GcLoadIndexedR,
        GetarrayitemGcR,
        GetarrayitemRawR,
        GetinteriorfieldGcR,
        GetfieldGcR,
        GetfieldRawR,
        GetfieldGcPureR,
        New,
        NewWithVtable,
        NewArray,
        NewArrayClear,
        Newstr,
        Newunicode,
        ForceToken,
        VirtualRefR,
        GuardException,
        CondCallValueR,
        CallR,
        CallPureR,
        CallMayForceR,
        CallAssemblerR,
        CallLoopinvariantR,
        ThreadlocalrefGet,
        CallMallocNursery,
        CallMallocNurseryVarsize,
        CallMallocNurseryVarsizeFrame,
        SaveException
    );

    // VecI/VecF, VecUnpack*, VecPack*, VecExpand* can be either int or float
    // depending on usage. Default to int for I variants, float for F variants.
    int!(VecI, VecUnpackI, VecPackI, VecExpandI, VecLoadI);
    float!(VecF, VecUnpackF, VecPackF, VecExpandF, VecLoadF);
    int!(VirtualRefI);
    t
};

/// Operation names for debugging.
static OPNAME: [&str; OPCODE_COUNT] = {
    let mut t = [""; OPCODE_COUNT];
    use OpCode::*;
    macro_rules! name {
        ($($op:ident),+ $(,)?) => {
            $(t[$op as usize] = stringify!($op);)+
        };
    }
    name!(
        Jump,
        Finish,
        Label,
        GuardTrue,
        GuardFalse,
        VecGuardTrue,
        VecGuardFalse,
        GuardValue,
        GuardClass,
        GuardNonnull,
        GuardIsnull,
        GuardNonnullClass,
        GuardGcType,
        GuardIsObject,
        GuardSubclass,
        GuardNoException,
        GuardException,
        GuardNoOverflow,
        GuardOverflow,
        GuardNotForced,
        GuardNotForced2,
        GuardNotInvalidated,
        GuardFutureCondition,
        GuardAlwaysFails,
        IntAdd,
        IntSub,
        IntMul,
        UintMulHigh,
        IntFloorDiv,
        IntMod,
        IntAnd,
        IntOr,
        IntXor,
        IntRshift,
        IntLshift,
        UintRshift,
        IntSignext,
        FloatAdd,
        FloatSub,
        FloatMul,
        FloatTrueDiv,
        FloatFloorDiv,
        FloatMod,
        FloatNeg,
        FloatAbs,
        CastFloatToInt,
        CastIntToFloat,
        CastFloatToSinglefloat,
        CastSinglefloatToFloat,
        ConvertFloatBytesToLonglong,
        ConvertLonglongBytesToFloat,
        VecIntAdd,
        VecIntSub,
        VecIntMul,
        VecIntAnd,
        VecIntOr,
        VecIntXor,
        VecFloatAdd,
        VecFloatSub,
        VecFloatMul,
        VecFloatTrueDiv,
        VecFloatNeg,
        VecFloatAbs,
        VecFloatEq,
        VecFloatNe,
        VecFloatXor,
        VecIntIsTrue,
        VecIntNe,
        VecIntEq,
        VecIntSignext,
        VecCastFloatToSinglefloat,
        VecCastSinglefloatToFloat,
        VecCastFloatToInt,
        VecCastIntToFloat,
        VecI,
        VecF,
        VecUnpackI,
        VecUnpackF,
        VecPackI,
        VecPackF,
        VecExpandI,
        VecExpandF,
        IntLt,
        IntLe,
        IntEq,
        IntNe,
        IntGt,
        IntGe,
        UintLt,
        UintLe,
        UintGt,
        UintGe,
        FloatLt,
        FloatLe,
        FloatEq,
        FloatNe,
        FloatGt,
        FloatGe,
        IntIsZero,
        IntIsTrue,
        IntNeg,
        IntInvert,
        IntForceGeZero,
        IntBetween,
        SameAsI,
        SameAsR,
        SameAsF,
        CastPtrToInt,
        CastIntToPtr,
        CastOpaquePtr,
        PtrEq,
        PtrNe,
        InstancePtrEq,
        InstancePtrNe,
        NurseryPtrIncrement,
        ArraylenGc,
        Strlen,
        Strgetitem,
        GetarrayitemGcPureI,
        GetarrayitemGcPureR,
        GetarrayitemGcPureF,
        Unicodelen,
        Unicodegetitem,
        LoadFromGcTable,
        LoadEffectiveAddress,
        ThreadlocalrefGet,
        GcLoadI,
        GcLoadR,
        GcLoadF,
        GcLoadIndexedI,
        GcLoadIndexedR,
        GcLoadIndexedF,
        GetarrayitemGcI,
        GetarrayitemGcR,
        GetarrayitemGcF,
        GetarrayitemRawI,
        GetarrayitemRawR,
        GetarrayitemRawF,
        RawLoadI,
        RawLoadF,
        VecLoadI,
        VecLoadF,
        GetinteriorfieldGcI,
        GetinteriorfieldGcR,
        GetinteriorfieldGcF,
        GetfieldGcI,
        GetfieldGcR,
        GetfieldGcF,
        GetfieldRawI,
        GetfieldRawR,
        GetfieldRawF,
        GetfieldGcPureI,
        GetfieldGcPureR,
        GetfieldGcPureF,
        New,
        NewWithVtable,
        NewArray,
        NewArrayClear,
        Newstr,
        Newunicode,
        ForceToken,
        VirtualRefI,
        VirtualRefR,
        Strhash,
        Unicodehash,
        GcStore,
        GcStoreIndexed,
        IncrementDebugCounter,
        SetarrayitemGc,
        SetarrayitemRaw,
        RawStore,
        VecStore,
        SetinteriorfieldGc,
        SetinteriorfieldRaw,
        SetfieldGc,
        ZeroArray,
        SetfieldRaw,
        Strsetitem,
        Unicodesetitem,
        CondCallGcWb,
        CondCallGcWbArray,
        DebugMergePoint,
        EnterPortalFrame,
        LeavePortalFrame,
        JitDebug,
        ForceSpill,
        VirtualRefFinish,
        Copystrcontent,
        Copyunicodecontent,
        QuasiimmutField,
        AssertNotNone,
        RecordExactClass,
        RecordExactValueR,
        RecordExactValueI,
        Keepalive,
        SaveException,
        SaveExcClass,
        RestoreException,
        CallI,
        CallR,
        CallF,
        CallN,
        CondCallN,
        CondCallValueI,
        CondCallValueR,
        CallAssemblerI,
        CallAssemblerR,
        CallAssemblerF,
        CallAssemblerN,
        CallMayForceI,
        CallMayForceR,
        CallMayForceF,
        CallMayForceN,
        CallLoopinvariantI,
        CallLoopinvariantR,
        CallLoopinvariantF,
        CallLoopinvariantN,
        CallReleaseGilI,
        CallReleaseGilF,
        CallReleaseGilN,
        CallPureI,
        CallPureR,
        CallPureF,
        CallPureN,
        CheckMemoryError,
        CallMallocNursery,
        CallMallocNurseryVarsize,
        CallMallocNurseryVarsizeFrame,
        RecordKnownResult,
        IntAddOvf,
        IntSubOvf,
        IntMulOvf
    );
    t
};

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! op {
        ($($field:tt)*) => {{
            let mut __op = Op {
                $($field)*
                type_: Type::Void,
                vecinfo: None,
            };
            __op.type_ = __op.opcode.result_type();
            __op
        }};
    }

    /// Iterate over all defined OpCode variants.
    fn all_opcodes() -> impl Iterator<Item = OpCode> {
        (0..OPCODE_COUNT as u16).map(|i| unsafe { std::mem::transmute::<u16, OpCode>(i) })
    }

    // ══════════════════════════════════════════════════════════════════
    // Resoperation parity tests
    // Ported from rpython/jit/metainterp/test/test_resoperation.py
    // ══════════════════════════════════════════════════════════════════

    // ── OpRef Eq/Hash sentinel coverage ──

    #[test]
    fn opref_none_collapses_with_untyped_max() {
        // Pre-Phase-3 raw-only equality treated `OpRef::from_raw(u32::MAX)`
        // and the explicit `OpRef::None` as the same absent-value
        // sentinel. RPython has no separate identity for missing
        // values, so variant-aware Eq/Hash must keep that collapse.
        let none = OpRef::NONE;
        let untyped_max = OpRef::from_raw(u32::MAX);
        assert!(none.is_none());
        assert!(untyped_max.is_none());
        assert_eq!(none, untyped_max);
        assert_eq!(untyped_max, none);

        let mut h1 = std::collections::hash_map::DefaultHasher::new();
        let mut h2 = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&none, &mut h1);
        std::hash::Hash::hash(&untyped_max, &mut h2);
        assert_eq!(
            std::hash::Hasher::finish(&h1),
            std::hash::Hasher::finish(&h2)
        );

        let mut set = std::collections::HashSet::new();
        set.insert(none);
        assert!(set.contains(&untyped_max));
    }

    #[test]
    fn opref_typed_variants_disjoint_from_none() {
        // The is_none() short-circuit must not bleed into typed
        // variants — only `OpRef::None` and `Untyped(u32::MAX)`
        // collapse onto the absent-value bucket.
        let none = OpRef::NONE;
        let typed = OpRef::int_op(0);
        assert_ne!(none, typed);
        assert_ne!(typed, none);
        assert_ne!(OpRef::int_op(0), OpRef::ref_op(0));
    }

    // ── Metadata table coverage ──

    #[test]
    fn test_every_opcode_has_name() {
        for op in all_opcodes() {
            let name = op.name();
            assert!(
                !name.is_empty(),
                "OpCode {:?} (u16={}) has empty name",
                op,
                op.as_u16()
            );
        }
    }

    /// `from_u16` is the left-inverse of `as_u16` over the defined
    /// `0..OPCODE_COUNT` range, and returns `None` everywhere outside it.
    #[test]
    fn test_opcode_from_u16_roundtrip() {
        for op in all_opcodes() {
            let n = op.as_u16();
            assert_eq!(OpCode::from_u16(n), Some(op));
        }
        assert_eq!(OpCode::from_u16(OPCODE_COUNT as u16), None);
        assert_eq!(OpCode::from_u16(u16::MAX), None);
    }

    #[test]
    fn test_every_opcode_has_result_type() {
        for op in all_opcodes() {
            let _tp = op.result_type();
        }
    }

    #[test]
    fn test_every_opcode_has_arity_entry() {
        for op in all_opcodes() {
            let _arity = op.arity();
        }
    }

    #[test]
    fn test_every_opcode_has_descr_entry() {
        for op in all_opcodes() {
            let _has_descr = op.has_descr();
        }
    }

    #[test]
    fn test_every_opcode_has_bool_entry() {
        for op in all_opcodes() {
            let _returns_bool = op.returns_bool();
        }
    }

    // ── Arity: nullary / unary / binary / variadic ──

    #[test]
    fn test_arity_nullary() {
        let nullary_ops = [
            OpCode::New,
            OpCode::NewWithVtable,
            OpCode::ForceToken,
            OpCode::GuardNoException,
            OpCode::GuardNoOverflow,
            OpCode::GuardOverflow,
            OpCode::GuardNotForced,
            OpCode::GuardNotForced2,
            OpCode::GuardNotInvalidated,
            OpCode::GuardFutureCondition,
            OpCode::GuardAlwaysFails,
            OpCode::VecI,
            OpCode::VecF,
            OpCode::ThreadlocalrefGet,
            OpCode::SaveException,
            OpCode::SaveExcClass,
        ];
        for op in &nullary_ops {
            assert_eq!(op.arity(), Some(0), "{:?} should have arity 0", op);
        }
    }

    #[test]
    fn test_arity_unary() {
        let unary_ops = [
            OpCode::GuardTrue,
            OpCode::GuardFalse,
            OpCode::GuardNonnull,
            OpCode::GuardIsnull,
            OpCode::GuardIsObject,
            OpCode::GuardException,
            OpCode::FloatNeg,
            OpCode::FloatAbs,
            OpCode::CastFloatToInt,
            OpCode::CastIntToFloat,
            OpCode::IntIsZero,
            OpCode::IntIsTrue,
            OpCode::IntNeg,
            OpCode::IntInvert,
            OpCode::IntForceGeZero,
            OpCode::SameAsI,
            OpCode::SameAsR,
            OpCode::SameAsF,
            OpCode::CastPtrToInt,
            OpCode::CastIntToPtr,
            OpCode::CastOpaquePtr,
            OpCode::ArraylenGc,
            OpCode::Strlen,
            OpCode::Unicodelen,
            OpCode::GetfieldGcI,
            OpCode::GetfieldGcR,
            OpCode::GetfieldGcF,
            OpCode::GetfieldRawI,
            OpCode::GetfieldRawR,
            OpCode::GetfieldRawF,
            OpCode::GetfieldGcPureI,
            OpCode::GetfieldGcPureR,
            OpCode::GetfieldGcPureF,
            OpCode::NewArray,
            OpCode::NewArrayClear,
            OpCode::Newstr,
            OpCode::Newunicode,
            OpCode::Strhash,
            OpCode::Unicodehash,
            OpCode::CheckMemoryError,
            OpCode::ForceSpill,
            OpCode::QuasiimmutField,
            OpCode::AssertNotNone,
            OpCode::Keepalive,
            OpCode::CondCallGcWb,
            OpCode::LoadFromGcTable,
            OpCode::IncrementDebugCounter,
            OpCode::LeavePortalFrame,
            OpCode::CallMallocNursery,
            OpCode::CallMallocNurseryVarsizeFrame,
        ];
        for op in &unary_ops {
            assert_eq!(op.arity(), Some(1), "{:?} should have arity 1", op);
        }
    }

    #[test]
    fn test_arity_binary() {
        let binary_ops = [
            OpCode::IntAdd,
            OpCode::IntSub,
            OpCode::IntMul,
            OpCode::UintMulHigh,
            OpCode::IntFloorDiv,
            OpCode::IntMod,
            OpCode::IntAnd,
            OpCode::IntOr,
            OpCode::IntXor,
            OpCode::IntRshift,
            OpCode::IntLshift,
            OpCode::UintRshift,
            OpCode::IntSignext,
            OpCode::FloatAdd,
            OpCode::FloatSub,
            OpCode::FloatMul,
            OpCode::FloatTrueDiv,
            OpCode::FloatFloorDiv,
            OpCode::FloatMod,
            OpCode::IntLt,
            OpCode::IntLe,
            OpCode::IntEq,
            OpCode::IntNe,
            OpCode::IntGt,
            OpCode::IntGe,
            OpCode::UintLt,
            OpCode::UintLe,
            OpCode::UintGt,
            OpCode::UintGe,
            OpCode::FloatLt,
            OpCode::FloatLe,
            OpCode::FloatEq,
            OpCode::FloatNe,
            OpCode::FloatGt,
            OpCode::FloatGe,
            OpCode::PtrEq,
            OpCode::PtrNe,
            OpCode::InstancePtrEq,
            OpCode::InstancePtrNe,
            OpCode::NurseryPtrIncrement,
            OpCode::Strgetitem,
            OpCode::Unicodegetitem,
            OpCode::GuardValue,
            OpCode::GuardClass,
            OpCode::GuardNonnullClass,
            OpCode::GuardGcType,
            OpCode::GuardSubclass,
            OpCode::SetfieldGc,
            OpCode::SetfieldRaw,
            OpCode::CondCallGcWbArray,
            OpCode::VirtualRefI,
            OpCode::VirtualRefR,
            OpCode::VirtualRefFinish,
            OpCode::RecordExactClass,
            OpCode::RecordExactValueR,
            OpCode::RecordExactValueI,
            OpCode::EnterPortalFrame,
            OpCode::RestoreException,
            OpCode::RawLoadI,
            OpCode::RawLoadF,
            OpCode::GetarrayitemGcI,
            OpCode::GetarrayitemGcR,
            OpCode::GetarrayitemGcF,
            OpCode::GetarrayitemGcPureI,
            OpCode::GetarrayitemGcPureR,
            OpCode::GetarrayitemGcPureF,
            OpCode::GetarrayitemRawI,
            OpCode::GetarrayitemRawR,
            OpCode::GetarrayitemRawF,
            OpCode::GetinteriorfieldGcI,
            OpCode::GetinteriorfieldGcR,
            OpCode::GetinteriorfieldGcF,
            OpCode::IntAddOvf,
            OpCode::IntSubOvf,
            OpCode::IntMulOvf,
        ];
        for op in &binary_ops {
            assert_eq!(op.arity(), Some(2), "{:?} should have arity 2", op);
        }
    }

    #[test]
    fn test_arity_variadic() {
        let variadic_ops = [
            OpCode::Jump,
            OpCode::Finish,
            OpCode::Label,
            OpCode::DebugMergePoint,
            OpCode::JitDebug,
            OpCode::CallI,
            OpCode::CallR,
            OpCode::CallF,
            OpCode::CallN,
            OpCode::CondCallN,
            OpCode::CondCallValueI,
            OpCode::CondCallValueR,
            OpCode::CallAssemblerI,
            OpCode::CallAssemblerR,
            OpCode::CallAssemblerF,
            OpCode::CallAssemblerN,
            OpCode::CallMayForceI,
            OpCode::CallMayForceR,
            OpCode::CallMayForceF,
            OpCode::CallMayForceN,
            OpCode::CallLoopinvariantI,
            OpCode::CallLoopinvariantR,
            OpCode::CallLoopinvariantF,
            OpCode::CallLoopinvariantN,
            OpCode::CallReleaseGilI,
            OpCode::CallReleaseGilF,
            OpCode::CallReleaseGilN,
            OpCode::CallPureI,
            OpCode::CallPureR,
            OpCode::CallPureF,
            OpCode::CallPureN,
            OpCode::CallMallocNurseryVarsize,
            OpCode::RecordKnownResult,
        ];
        for op in &variadic_ops {
            assert_eq!(op.arity(), None, "{:?} should be variadic (arity=None)", op);
        }
    }

    // ── Result type exhaustive checks ──

    #[test]
    fn test_int_result_types() {
        let int_ops = [
            OpCode::IntAdd,
            OpCode::IntSub,
            OpCode::IntMul,
            OpCode::IntFloorDiv,
            OpCode::IntMod,
            OpCode::IntAnd,
            OpCode::IntOr,
            OpCode::IntXor,
            OpCode::IntRshift,
            OpCode::IntLshift,
            OpCode::UintRshift,
            OpCode::IntSignext,
            OpCode::CastFloatToInt,
            OpCode::IntLt,
            OpCode::IntLe,
            OpCode::IntEq,
            OpCode::IntNe,
            OpCode::IntGt,
            OpCode::IntGe,
            OpCode::IntIsZero,
            OpCode::IntIsTrue,
            OpCode::IntNeg,
            OpCode::IntInvert,
            OpCode::IntForceGeZero,
            OpCode::SameAsI,
            OpCode::CastPtrToInt,
            OpCode::PtrEq,
            OpCode::PtrNe,
            OpCode::IntAddOvf,
            OpCode::IntSubOvf,
            OpCode::IntMulOvf,
            OpCode::GetfieldGcI,
            OpCode::GetfieldRawI,
            OpCode::GetfieldGcPureI,
            OpCode::GetarrayitemGcI,
            OpCode::GetarrayitemRawI,
            OpCode::GetarrayitemGcPureI,
            OpCode::CallI,
            OpCode::CallPureI,
            OpCode::CallMayForceI,
            OpCode::CallAssemblerI,
            OpCode::CallLoopinvariantI,
            OpCode::CallReleaseGilI,
            OpCode::SaveExcClass,
        ];
        for op in &int_ops {
            assert_eq!(op.result_type(), Type::Int, "{:?} should return Int", op);
        }
    }

    #[test]
    fn test_float_result_types() {
        let float_ops = [
            OpCode::FloatAdd,
            OpCode::FloatSub,
            OpCode::FloatMul,
            OpCode::FloatTrueDiv,
            OpCode::FloatFloorDiv,
            OpCode::FloatMod,
            OpCode::FloatNeg,
            OpCode::FloatAbs,
            OpCode::CastIntToFloat,
            OpCode::CastSinglefloatToFloat,
            OpCode::SameAsF,
            OpCode::GetfieldGcF,
            OpCode::GetfieldRawF,
            OpCode::GetfieldGcPureF,
            OpCode::GetarrayitemGcF,
            OpCode::GetarrayitemRawF,
            OpCode::GetarrayitemGcPureF,
            OpCode::CallF,
            OpCode::CallPureF,
            OpCode::CallMayForceF,
            OpCode::CallAssemblerF,
            OpCode::CallLoopinvariantF,
            OpCode::CallReleaseGilF,
        ];
        for op in &float_ops {
            assert_eq!(
                op.result_type(),
                Type::Float,
                "{:?} should return Float",
                op
            );
        }
    }

    #[test]
    fn test_ref_result_types() {
        let ref_ops = [
            OpCode::CastIntToPtr,
            OpCode::CastOpaquePtr,
            OpCode::SameAsR,
            OpCode::NurseryPtrIncrement,
            OpCode::LoadFromGcTable,
            OpCode::New,
            OpCode::NewWithVtable,
            OpCode::NewArray,
            OpCode::NewArrayClear,
            OpCode::Newstr,
            OpCode::Newunicode,
            OpCode::ForceToken,
            OpCode::VirtualRefR,
            OpCode::GuardException,
            OpCode::GetfieldGcR,
            OpCode::GetfieldRawR,
            OpCode::GetfieldGcPureR,
            OpCode::GetarrayitemGcR,
            OpCode::GetarrayitemRawR,
            OpCode::GetarrayitemGcPureR,
            OpCode::CallR,
            OpCode::CallPureR,
            OpCode::CallMayForceR,
            OpCode::CallAssemblerR,
            OpCode::CallLoopinvariantR,
            OpCode::CondCallValueR,
            OpCode::ThreadlocalrefGet,
            OpCode::CallMallocNursery,
            OpCode::CallMallocNurseryVarsize,
            OpCode::CallMallocNurseryVarsizeFrame,
            OpCode::SaveException,
        ];
        for op in &ref_ops {
            assert_eq!(op.result_type(), Type::Ref, "{:?} should return Ref", op);
        }
    }

    #[test]
    fn test_void_result_types() {
        let void_ops = [
            OpCode::Jump,
            OpCode::Finish,
            OpCode::Label,
            OpCode::SetfieldGc,
            OpCode::SetfieldRaw,
            OpCode::SetarrayitemGc,
            OpCode::SetarrayitemRaw,
            OpCode::SetinteriorfieldGc,
            OpCode::SetinteriorfieldRaw,
            OpCode::RawStore,
            OpCode::GcStore,
            OpCode::GcStoreIndexed,
            OpCode::Strsetitem,
            OpCode::Unicodesetitem,
            OpCode::CondCallGcWb,
            OpCode::CondCallGcWbArray,
            OpCode::DebugMergePoint,
            OpCode::EnterPortalFrame,
            OpCode::LeavePortalFrame,
            OpCode::JitDebug,
            OpCode::CallN,
            OpCode::CondCallN,
            OpCode::CallAssemblerN,
            OpCode::CallMayForceN,
            OpCode::CallLoopinvariantN,
            OpCode::CallReleaseGilN,
            OpCode::CallPureN,
            OpCode::ForceSpill,
            OpCode::VirtualRefFinish,
            OpCode::Copystrcontent,
            OpCode::Copyunicodecontent,
            OpCode::QuasiimmutField,
            OpCode::AssertNotNone,
            OpCode::RecordExactClass,
            OpCode::Keepalive,
            OpCode::RestoreException,
            OpCode::ZeroArray,
            OpCode::VecStore,
            OpCode::IncrementDebugCounter,
        ];
        for op in &void_ops {
            assert_eq!(op.result_type(), Type::Void, "{:?} should return Void", op);
        }
    }

    // ── Classification methods ──

    #[test]
    fn test_category_classification() {
        assert!(OpCode::Jump.is_final());
        assert!(OpCode::Finish.is_final());
        assert!(!OpCode::Label.is_final());

        assert!(OpCode::GuardTrue.is_guard());
        assert!(OpCode::GuardAlwaysFails.is_guard());
        assert!(!OpCode::IntAdd.is_guard());

        assert!(OpCode::IntAdd.is_always_pure());
        assert!(OpCode::FloatMul.is_always_pure());
        assert!(OpCode::GetfieldGcPureI.is_always_pure());
        assert!(!OpCode::SetfieldGc.is_always_pure());

        assert!(OpCode::IntAddOvf.is_ovf());
        assert!(!OpCode::IntAdd.is_ovf());

        assert!(OpCode::CallI.is_call());
        assert!(OpCode::CallPureN.is_call());
        assert!(!OpCode::IntAdd.is_call());
    }

    #[test]
    fn test_guard_classification_exhaustive() {
        let all_guards: Vec<OpCode> = all_opcodes().filter(|op| op.is_guard()).collect();
        assert!(
            all_guards.len() >= 20,
            "expected at least 20 guard ops, got {}",
            all_guards.len()
        );
        let expected_guards = [
            OpCode::GuardTrue,
            OpCode::GuardFalse,
            OpCode::VecGuardTrue,
            OpCode::VecGuardFalse,
            OpCode::GuardValue,
            OpCode::GuardClass,
            OpCode::GuardNonnull,
            OpCode::GuardIsnull,
            OpCode::GuardNonnullClass,
            OpCode::GuardGcType,
            OpCode::GuardIsObject,
            OpCode::GuardSubclass,
            OpCode::GuardNoException,
            OpCode::GuardException,
            OpCode::GuardNoOverflow,
            OpCode::GuardOverflow,
            OpCode::GuardNotForced,
            OpCode::GuardNotForced2,
            OpCode::GuardNotInvalidated,
            OpCode::GuardFutureCondition,
            OpCode::GuardAlwaysFails,
        ];
        for op in &expected_guards {
            assert!(op.is_guard(), "{:?} should be a guard", op);
        }
    }

    #[test]
    fn test_foldable_guard_subset() {
        let foldable_guards = [
            OpCode::GuardTrue,
            OpCode::GuardFalse,
            OpCode::VecGuardTrue,
            OpCode::VecGuardFalse,
            OpCode::GuardValue,
            OpCode::GuardClass,
            OpCode::GuardNonnull,
            OpCode::GuardIsnull,
            OpCode::GuardNonnullClass,
            OpCode::GuardGcType,
            OpCode::GuardIsObject,
            OpCode::GuardSubclass,
        ];
        for op in &foldable_guards {
            assert!(op.is_foldable_guard(), "{:?} should be foldable", op);
            assert!(
                op.is_guard(),
                "foldable guard {:?} must also be a guard",
                op
            );
        }
        let non_foldable = [
            OpCode::GuardNoException,
            OpCode::GuardNotForced,
            OpCode::GuardNotInvalidated,
            OpCode::GuardAlwaysFails,
        ];
        for op in &non_foldable {
            assert!(!op.is_foldable_guard(), "{:?} should NOT be foldable", op);
            assert!(op.is_guard(), "{:?} should still be a guard", op);
        }
    }

    #[test]
    fn test_pure_ops_no_side_effect() {
        for op in all_opcodes() {
            if op.is_always_pure() {
                assert!(
                    op.has_no_side_effect(),
                    "{:?} is pure but does not claim no_side_effect",
                    op
                );
            }
        }
    }

    #[test]
    fn test_no_side_effect_superset_of_pure() {
        let extra_nosideeffect = [
            OpCode::GcLoadI,
            OpCode::GcLoadR,
            OpCode::GcLoadF,
            OpCode::GetarrayitemGcI,
            OpCode::GetarrayitemGcR,
            OpCode::GetarrayitemGcF,
            OpCode::GetfieldGcI,
            OpCode::GetfieldGcR,
            OpCode::GetfieldGcF,
            OpCode::New,
            OpCode::NewWithVtable,
            OpCode::NewArray,
            OpCode::ForceToken,
            OpCode::Strhash,
            OpCode::Unicodehash,
        ];
        for op in &extra_nosideeffect {
            assert!(
                op.has_no_side_effect(),
                "{:?} should have no_side_effect",
                op
            );
        }
    }

    #[test]
    fn test_can_malloc() {
        assert!(OpCode::New.can_malloc());
        assert!(OpCode::NewWithVtable.can_malloc());
        assert!(OpCode::NewArray.can_malloc());
        assert!(OpCode::CallN.can_malloc());
        assert!(OpCode::CallI.can_malloc());
        assert!(OpCode::CallMayForceI.can_malloc());
        assert!(!OpCode::IntAdd.can_malloc());
        assert!(!OpCode::GuardTrue.can_malloc());
    }

    #[test]
    fn test_is_comparison() {
        let comparisons = [
            OpCode::IntLt,
            OpCode::IntLe,
            OpCode::IntEq,
            OpCode::IntNe,
            OpCode::IntGt,
            OpCode::IntGe,
            OpCode::UintLt,
            OpCode::UintLe,
            OpCode::UintGt,
            OpCode::UintGe,
            OpCode::FloatLt,
            OpCode::FloatLe,
            OpCode::FloatEq,
            OpCode::FloatNe,
            OpCode::FloatGt,
            OpCode::FloatGe,
            OpCode::PtrEq,
            OpCode::PtrNe,
            OpCode::InstancePtrEq,
            OpCode::InstancePtrNe,
            OpCode::IntIsZero,
            OpCode::IntIsTrue,
            OpCode::IntBetween,
        ];
        for op in &comparisons {
            assert!(op.is_comparison(), "{:?} should be a comparison", op);
            assert!(op.is_always_pure(), "comparison {:?} must be pure", op);
            assert!(op.returns_bool(), "comparison {:?} must return bool", op);
        }
        assert!(!OpCode::IntAdd.is_comparison());
        assert!(!OpCode::FloatAdd.is_comparison());
    }

    #[test]
    fn test_guard_exception_classification() {
        assert!(OpCode::GuardException.is_guard_exception());
        assert!(OpCode::GuardNoException.is_guard_exception());
        assert!(!OpCode::GuardTrue.is_guard_exception());
    }

    #[test]
    fn test_guard_overflow_classification() {
        assert!(OpCode::GuardOverflow.is_guard_overflow());
        assert!(OpCode::GuardNoOverflow.is_guard_overflow());
        assert!(!OpCode::GuardTrue.is_guard_overflow());
    }

    #[test]
    fn test_call_subcategories() {
        for op in all_opcodes() {
            if op.is_plain_call()
                || op.is_call_assembler()
                || op.is_call_may_force()
                || op.is_call_pure()
                || op.is_call_release_gil()
                || op.is_call_loopinvariant()
                || op.is_cond_call_value()
            {
                assert!(
                    op.is_call(),
                    "{:?} is a call subcategory but not is_call()",
                    op
                );
            }
        }
    }

    #[test]
    fn test_is_same_as() {
        assert!(OpCode::SameAsI.is_same_as());
        assert!(OpCode::SameAsR.is_same_as());
        assert!(OpCode::SameAsF.is_same_as());
        assert!(!OpCode::IntAdd.is_same_as());
    }

    // ── Typed dispatch ──

    #[test]
    fn test_call_for_type() {
        assert_eq!(OpCode::call_for_type(Type::Int), OpCode::CallI);
        assert_eq!(OpCode::call_for_type(Type::Ref), OpCode::CallR);
        assert_eq!(OpCode::call_for_type(Type::Float), OpCode::CallF);
        assert_eq!(OpCode::call_for_type(Type::Void), OpCode::CallN);
    }

    #[test]
    fn test_call_pure_for_type() {
        assert_eq!(OpCode::call_pure_for_type(Type::Int), OpCode::CallPureI);
        assert_eq!(OpCode::call_pure_for_type(Type::Float), OpCode::CallPureF);
    }

    #[test]
    fn test_same_as_for_type() {
        assert_eq!(OpCode::same_as_for_type(Type::Int), OpCode::SameAsI);
        assert_eq!(OpCode::same_as_for_type(Type::Ref), OpCode::SameAsR);
        assert_eq!(OpCode::same_as_for_type(Type::Float), OpCode::SameAsF);
    }

    #[test]
    fn test_getfield_for_type() {
        assert_eq!(OpCode::getfield_for_type(Type::Int), OpCode::GetfieldGcI);
        assert_eq!(OpCode::getfield_for_type(Type::Ref), OpCode::GetfieldGcR);
        assert_eq!(OpCode::getfield_for_type(Type::Float), OpCode::GetfieldGcF);
    }

    // ── bool_inverse / bool_reflex ──

    #[test]
    fn test_bool_inverse() {
        assert_eq!(OpCode::IntEq.bool_inverse(), Some(OpCode::IntNe));
        assert_eq!(OpCode::IntNe.bool_inverse(), Some(OpCode::IntEq));
        assert_eq!(OpCode::IntLt.bool_inverse(), Some(OpCode::IntGe));
        assert_eq!(OpCode::IntGe.bool_inverse(), Some(OpCode::IntLt));
        assert_eq!(OpCode::IntGt.bool_inverse(), Some(OpCode::IntLe));
        assert_eq!(OpCode::IntLe.bool_inverse(), Some(OpCode::IntGt));
        assert_eq!(OpCode::FloatEq.bool_inverse(), Some(OpCode::FloatNe));
        assert_eq!(OpCode::FloatLt.bool_inverse(), Some(OpCode::FloatGe));
        assert_eq!(OpCode::UintLt.bool_inverse(), Some(OpCode::UintGe));
        assert_eq!(OpCode::PtrEq.bool_inverse(), Some(OpCode::PtrNe));
        assert_eq!(OpCode::IntAdd.bool_inverse(), None);
    }

    #[test]
    fn test_bool_inverse_is_involution() {
        for op in all_opcodes() {
            if let Some(inv) = op.bool_inverse() {
                assert_eq!(
                    inv.bool_inverse(),
                    Some(op),
                    "bool_inverse should be an involution for {:?}",
                    op
                );
            }
        }
    }

    #[test]
    fn test_bool_reflex() {
        assert_eq!(OpCode::IntLt.bool_reflex(), Some(OpCode::IntGt));
        assert_eq!(OpCode::IntGt.bool_reflex(), Some(OpCode::IntLt));
        assert_eq!(OpCode::IntEq.bool_reflex(), Some(OpCode::IntEq));
        assert_eq!(OpCode::IntNe.bool_reflex(), Some(OpCode::IntNe));
        assert_eq!(OpCode::FloatLt.bool_reflex(), Some(OpCode::FloatGt));
        assert_eq!(OpCode::PtrEq.bool_reflex(), Some(OpCode::PtrEq));
        assert_eq!(OpCode::IntAdd.bool_reflex(), None);
    }

    #[test]
    fn test_bool_reflex_is_involution() {
        for op in all_opcodes() {
            if let Some(refl) = op.bool_reflex() {
                assert_eq!(
                    refl.bool_reflex(),
                    Some(op),
                    "bool_reflex should be an involution for {:?}",
                    op
                );
            }
        }
    }

    // ── without_overflow / to_vector ──

    #[test]
    fn test_without_overflow() {
        assert_eq!(OpCode::IntAddOvf.without_overflow(), Some(OpCode::IntAdd));
        assert_eq!(OpCode::IntSubOvf.without_overflow(), Some(OpCode::IntSub));
        assert_eq!(OpCode::IntMulOvf.without_overflow(), Some(OpCode::IntMul));
        assert_eq!(OpCode::IntAdd.without_overflow(), None);
    }

    #[test]
    fn test_to_vector() {
        assert_eq!(OpCode::IntAdd.to_vector(), Some(OpCode::VecIntAdd));
        assert_eq!(OpCode::FloatAdd.to_vector(), Some(OpCode::VecFloatAdd));
        assert_eq!(OpCode::GuardTrue.to_vector(), Some(OpCode::VecGuardTrue));
        assert_eq!(OpCode::SetfieldGc.to_vector(), None);
    }

    // ── Name table ──

    #[test]
    fn test_opname_matches_debug_name() {
        for op in all_opcodes() {
            let name = op.name();
            let debug = format!("{:?}", op);
            assert_eq!(name, debug, "name() and Debug should match for {:?}", op);
        }
    }

    #[test]
    fn test_specific_opnames() {
        assert_eq!(OpCode::IntAdd.name(), "IntAdd");
        assert_eq!(OpCode::GuardTrue.name(), "GuardTrue");
        assert_eq!(OpCode::CallI.name(), "CallI");
        assert_eq!(OpCode::Jump.name(), "Jump");
        assert_eq!(OpCode::Finish.name(), "Finish");
        assert_eq!(OpCode::New.name(), "New");
        assert_eq!(OpCode::SetfieldGc.name(), "SetfieldGc");
    }

    // ── Op construction ──

    #[test]
    fn test_op_new() {
        let op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]);
        assert_eq!(op.opcode, OpCode::IntAdd);
        assert_eq!(op.args.len(), 2);
        assert_eq!(op.args[0], OpRef::from_raw(0));
        assert_eq!(op.args[1], OpRef::from_raw(1));
        assert!(op.descr.is_none());
        assert!(op.fail_args.is_none());
        assert_eq!(op.result_type(), Type::Int);
        assert_eq!(op.num_args(), 2);
    }

    #[test]
    fn test_op_getarg() {
        let op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(10), OpRef::from_raw(20)]);
        assert_eq!(op.arg(0), OpRef::from_raw(10));
        assert_eq!(op.arg(1), OpRef::from_raw(20));
    }

    // ── Descriptor requirements ──

    #[test]
    fn test_guards_have_descr() {
        for op in all_opcodes() {
            if op.is_guard() {
                assert!(op.has_descr(), "guard {:?} should have has_descr=true", op);
            }
        }
    }

    #[test]
    fn test_calls_have_descr() {
        // All call subcategories (plain calls, call_assembler, call_may_force,
        // call_pure, call_release_gil, call_loopinvariant, cond_call_value)
        // must have descriptors. Backend helpers like CheckMemoryError and
        // CallMallocNursery* are in the call range but don't need descriptors.
        for op in all_opcodes() {
            if op.is_plain_call()
                || op.is_call_assembler()
                || op.is_call_may_force()
                || op.is_call_pure()
                || op.is_call_release_gil()
                || op.is_call_loopinvariant()
                || op.is_cond_call_value()
            {
                assert!(op.has_descr(), "call {:?} should have has_descr=true", op);
            }
        }
    }

    // ── ovf alignment ──

    #[test]
    fn test_ovf_to_non_ovf_alignment() {
        let add_ovf_offset = OpCode::IntAddOvf as u16 - OVF_FIRST;
        let add_offset = OpCode::IntAdd as u16 - ALWAYS_PURE_FIRST;
        assert_eq!(add_ovf_offset, add_offset);

        let sub_ovf_offset = OpCode::IntSubOvf as u16 - OVF_FIRST;
        let sub_offset = OpCode::IntSub as u16 - ALWAYS_PURE_FIRST;
        assert_eq!(sub_ovf_offset, sub_offset);

        let mul_ovf_offset = OpCode::IntMulOvf as u16 - OVF_FIRST;
        let mul_offset = OpCode::IntMul as u16 - ALWAYS_PURE_FIRST;
        assert_eq!(mul_ovf_offset, mul_offset);
    }

    // ── is_getfield / is_getarrayitem / is_memory_access ──

    #[test]
    fn test_is_getfield() {
        assert!(OpCode::GetfieldGcI.is_getfield());
        assert!(OpCode::GetfieldGcR.is_getfield());
        assert!(OpCode::GetfieldGcF.is_getfield());
        assert!(OpCode::GetfieldGcPureI.is_getfield());
        assert!(OpCode::GetfieldGcPureR.is_getfield());
        assert!(OpCode::GetfieldGcPureF.is_getfield());
        assert!(!OpCode::GetfieldRawI.is_getfield());
        assert!(!OpCode::IntAdd.is_getfield());
    }

    #[test]
    fn test_is_getarrayitem() {
        assert!(OpCode::GetarrayitemGcI.is_getarrayitem());
        assert!(OpCode::GetarrayitemGcPureI.is_getarrayitem());
        assert!(!OpCode::IntAdd.is_getarrayitem());
    }

    #[test]
    fn test_memory_access_includes_fields_and_arrays() {
        let memory_ops = [
            OpCode::GetfieldGcI,
            OpCode::SetfieldGc,
            OpCode::GetarrayitemGcI,
            OpCode::SetarrayitemGc,
            OpCode::RawLoadI,
            OpCode::RawStore,
            OpCode::GcLoadI,
            OpCode::GcStore,
        ];
        for op in &memory_ops {
            assert!(op.is_memory_access(), "{:?} should be memory access", op);
        }
        assert!(!OpCode::IntAdd.is_memory_access());
        assert!(!OpCode::CallI.is_memory_access());
    }

    // ── can_raise ──

    #[test]
    fn test_can_raise() {
        assert!(OpCode::CallI.can_raise());
        assert!(OpCode::CallMayForceN.can_raise());
        assert!(OpCode::IntAddOvf.can_raise());
        assert!(OpCode::IntSubOvf.can_raise());
        assert!(OpCode::IntMulOvf.can_raise());
        assert!(!OpCode::IntAdd.can_raise());
        assert!(!OpCode::GuardTrue.can_raise());
        assert!(!OpCode::New.can_raise());
    }

    // ── is_label / is_jit_debug / is_malloc / is_vector_arithmetic ──

    #[test]
    fn test_is_label() {
        assert!(OpCode::Label.is_label());
        assert!(!OpCode::Jump.is_label());
    }

    #[test]
    fn test_is_jit_debug() {
        assert!(OpCode::DebugMergePoint.is_jit_debug());
        assert!(OpCode::EnterPortalFrame.is_jit_debug());
        assert!(OpCode::LeavePortalFrame.is_jit_debug());
        assert!(OpCode::JitDebug.is_jit_debug());
        assert!(!OpCode::IntAdd.is_jit_debug());
    }

    #[test]
    fn test_is_malloc() {
        let malloc_ops = [
            OpCode::New,
            OpCode::NewWithVtable,
            OpCode::NewArray,
            OpCode::NewArrayClear,
            OpCode::Newstr,
            OpCode::Newunicode,
        ];
        for op in &malloc_ops {
            assert!(op.is_malloc(), "{:?} should be malloc", op);
        }
        assert!(!OpCode::IntAdd.is_malloc());
        assert!(!OpCode::CallI.is_malloc());
    }

    #[test]
    fn test_is_vector_arithmetic() {
        let vec_arith = [
            OpCode::VecIntAdd,
            OpCode::VecIntSub,
            OpCode::VecIntMul,
            OpCode::VecFloatAdd,
            OpCode::VecFloatMul,
            OpCode::VecFloatNeg,
            OpCode::VecFloatAbs,
        ];
        for op in &vec_arith {
            assert!(
                op.is_vector_arithmetic(),
                "{:?} should be vec arithmetic",
                op
            );
        }
        assert!(!OpCode::IntAdd.is_vector_arithmetic());
    }

    // ── Consistency invariants ──

    #[test]
    fn test_guard_and_call_disjoint() {
        for op in all_opcodes() {
            assert!(
                !(op.is_guard() && op.is_call()),
                "{:?} is both guard and call",
                op
            );
        }
    }

    #[test]
    fn test_final_and_guard_disjoint() {
        for op in all_opcodes() {
            assert!(
                !(op.is_final() && op.is_guard()),
                "{:?} is both final and guard",
                op
            );
        }
    }

    #[test]
    fn test_guards_not_pure() {
        for op in all_opcodes() {
            if op.is_guard() {
                assert!(
                    !op.is_always_pure(),
                    "{:?} is a guard and should not be always_pure",
                    op
                );
            }
        }
    }

    // ── Logger parity tests (rpython/jit/metainterp/test/test_logger.py) ──

    #[test]
    fn test_format_trace_readable_output() {
        let ops = vec![
            op! {
                opcode: OpCode::IntAdd,
                args: smallvec::smallvec![OpRef::from_raw(1), OpRef::from_raw(2)],
                descr: None,
                pos: OpRef::from_raw(3),
                fail_args: None,

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::IntAdd,
                args: smallvec::smallvec![OpRef::from_raw(3), OpRef::from_raw(10_000)],
                descr: None,
                pos: OpRef::from_raw(4),
                fail_args: None,

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::Jump,
                args: smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(4), OpRef::from_raw(3)],
                descr: None,
                pos: OpRef::NONE,
                fail_args: None,


                fail_arg_types: None,
                rd_resume_position: -1,
            },
        ];
        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 3);
        let output = format_trace(&ops, &constants);
        assert!(output.contains("v3 = IntAdd(v1, v2)"));
        assert!(output.contains("v4 = IntAdd(v3, 3)"));
        assert!(output.contains("Jump(v0, v4, v3)"));
    }

    #[test]
    fn test_op_display_int_result() {
        let op = op! {
            opcode: OpCode::IntAdd,
            args: smallvec::smallvec![OpRef::from_raw(1), OpRef::from_raw(2)],
            descr: None,
            pos: OpRef::from_raw(6),
            fail_args: None,

            fail_arg_types: None,
            rd_resume_position: -1,
        };
        let s = format!("{op}");
        assert_eq!(s, "v6 = IntAdd(v1, v2)");
    }

    #[test]
    fn test_op_display_void() {
        let op = op! {
            opcode: OpCode::SetfieldGc,
            args: smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(1)],
            descr: None,
            pos: OpRef::NONE,
            fail_args: None,

            fail_arg_types: None,
            rd_resume_position: -1,
        };
        let s = format!("{op}");
        assert_eq!(s, "SetfieldGc(v0, v1)");
    }

    #[test]
    fn test_op_display_guard_with_fail_args() {
        let op = op! {
            opcode: OpCode::GuardTrue,
            args: smallvec::smallvec![OpRef::from_raw(0)],
            descr: None,
            pos: OpRef::NONE,
            fail_args: Some(smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(1)]),


            fail_arg_types: None,
            rd_resume_position: -1,
        };
        let s = format!("{op}");
        assert_eq!(s, "GuardTrue(v0) [v0, v1]");
    }

    #[test]
    fn test_op_display_guard_without_fail_args() {
        let op = op! {
            opcode: OpCode::GuardTrue,
            args: smallvec::smallvec![OpRef::from_raw(0)],
            descr: None,
            pos: OpRef::NONE,
            fail_args: None,

            fail_arg_types: None,
            rd_resume_position: -1,
        };
        let s = format!("{op}");
        assert_eq!(s, "GuardTrue(v0)");
    }

    #[test]
    fn test_format_trace_constants_rendered_with_values() {
        let ops = vec![op! {
            opcode: OpCode::IntAdd,
            args: smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(10_000)],
            descr: None,
            pos: OpRef::from_raw(1),
            fail_args: None,

            fail_arg_types: None,
            rd_resume_position: -1,
        }];
        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 42);
        let output = format_trace(&ops, &constants);
        assert!(output.contains("v1 = IntAdd(v0, 42)"));
        assert!(!output.contains("v10000"));
    }

    #[test]
    fn test_format_trace_guards_show_fail_args() {
        let ops = vec![
            op! {
                opcode: OpCode::IntAdd,
                args: smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(10_000)],
                descr: None,
                pos: OpRef::from_raw(1),
                fail_args: None,

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::GuardTrue,
                args: smallvec::smallvec![OpRef::from_raw(0)],
                descr: None,
                pos: OpRef::NONE,
                fail_args: Some(smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(1)]),

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::Finish,
                args: smallvec::smallvec![OpRef::from_raw(1)],
                descr: None,
                pos: OpRef::NONE,
                fail_args: None,


                fail_arg_types: None,
                rd_resume_position: -1,
            },
        ];
        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 1);
        let output = format_trace(&ops, &constants);
        assert!(output.contains("GuardTrue(v0) [v0, v1]"));
    }

    #[test]
    fn test_format_trace_constants_in_fail_args() {
        let ops = vec![op! {
            opcode: OpCode::GuardTrue,
            args: smallvec::smallvec![OpRef::from_raw(0)],
            descr: None,
            pos: OpRef::NONE,
            fail_args: Some(smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(10_000)]),


            fail_arg_types: None,
            rd_resume_position: -1,
        }];
        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 99);
        let output = format_trace(&ops, &constants);
        assert!(output.contains("GuardTrue(v0) [v0, 99]"));
    }

    #[test]
    fn test_format_trace_empty() {
        let ops: Vec<Op> = vec![];
        let constants = std::collections::HashMap::new();
        let output = format_trace(&ops, &constants);
        assert!(output.is_empty());
    }

    // ── Extended logger parity tests (rpython/jit/metainterp/test/test_logger.py) ──

    #[test]
    fn test_format_trace_full_loop_label_to_jump() {
        // Parity with test_simple: a full loop trace from Label to Jump
        // should format each op on its own line with readable names and args.
        let ops = vec![
            op! {
                opcode: OpCode::Label,
                args: smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)],
                descr: None,
                pos: OpRef::NONE,
                fail_args: None,

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::IntAdd,
                args: smallvec::smallvec![OpRef::from_raw(1), OpRef::from_raw(2)],
                descr: None,
                pos: OpRef::from_raw(3),
                fail_args: None,

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::IntAdd,
                args: smallvec::smallvec![OpRef::from_raw(3), OpRef::from_raw(10_000)],
                descr: None,
                pos: OpRef::from_raw(4),
                fail_args: None,

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::Jump,
                args: smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(4), OpRef::from_raw(3)],
                descr: None,
                pos: OpRef::NONE,
                fail_args: None,


                fail_arg_types: None,
                rd_resume_position: -1,
            },
        ];
        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 3);
        let output = format_trace(&ops, &constants);
        // Label opens, Jump closes
        assert!(output.contains("Label(v0, v1, v2)"));
        assert!(output.contains("v3 = IntAdd(v1, v2)"));
        assert!(output.contains("v4 = IntAdd(v3, 3)"));
        assert!(output.contains("Jump(v0, v4, v3)"));
        // Each line is indented with 2 spaces
        for line in output.lines() {
            assert!(
                line.starts_with("  "),
                "each line should be indented: {line}"
            );
        }
    }

    #[test]
    fn test_format_trace_bridge_guard_to_finish() {
        // Parity with test_guard: a bridge trace starts with ops and ends with Finish.
        let ops = vec![
            op! {
                opcode: OpCode::IntSub,
                args: smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(10_000)],
                descr: None,
                pos: OpRef::from_raw(1),
                fail_args: None,

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::IntGt,
                args: smallvec::smallvec![OpRef::from_raw(1), OpRef::from_raw(10_001)],
                descr: None,
                pos: OpRef::from_raw(2),
                fail_args: None,

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::GuardTrue,
                args: smallvec::smallvec![OpRef::from_raw(2)],
                descr: None,
                pos: OpRef::NONE,
                fail_args: Some(smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(1)]),

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::Finish,
                args: smallvec::smallvec![OpRef::from_raw(1)],
                descr: None,
                pos: OpRef::NONE,
                fail_args: None,


                fail_arg_types: None,
                rd_resume_position: -1,
            },
        ];
        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 1);
        constants.insert(10_001, 0);
        let output = format_trace(&ops, &constants);
        assert!(output.contains("v1 = IntSub(v0, 1)"));
        assert!(output.contains("v2 = IntGt(v1, 0)"));
        assert!(output.contains("GuardTrue(v2) [v0, v1]"));
        assert!(output.contains("Finish(v1)"));
    }

    #[test]
    fn test_format_trace_descr_repr_in_output() {
        // Parity with test_descr: descriptors are rendered in the output
        // via repr_of_descr.
        use crate::descr::{DebugMergePointDescr, DebugMergePointInfo};
        let descr: crate::DescrRef = std::sync::Arc::new(DebugMergePointDescr::new(
            DebugMergePointInfo::new("testdriver", "bytecode ADD at 5", 5, 0),
        ));
        let ops = vec![op! {
            opcode: OpCode::DebugMergePoint,
            args: smallvec::smallvec![],
            descr: Some(descr),
            pos: OpRef::NONE,
            fail_args: None,

            fail_arg_types: None,
            rd_resume_position: -1,
        }];
        let constants = std::collections::HashMap::new();
        let output = format_trace(&ops, &constants);
        assert!(
            output.contains("descr=<"),
            "output should contain 'descr=<': {output}"
        );
        assert!(
            output.contains("testdriver"),
            "descr repr should contain driver name: {output}"
        );
        assert!(
            output.contains("bytecode ADD at 5"),
            "descr repr should contain source repr: {output}"
        );
    }

    #[test]
    fn test_format_trace_complex_with_guards_and_constants() {
        // Parity with test_guard: complex trace with mixed ops, guards, constants,
        // and fail_args all render correctly and can be round-tripped.
        let ops = vec![
            op! {
                opcode: OpCode::Label,
                args: smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(1)],
                descr: None,
                pos: OpRef::NONE,
                fail_args: None,

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::IntAdd,
                args: smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(1)],
                descr: None,
                pos: OpRef::from_raw(2),
                fail_args: None,

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::IntLt,
                args: smallvec::smallvec![OpRef::from_raw(2), OpRef::from_raw(10_000)],
                descr: None,
                pos: OpRef::from_raw(3),
                fail_args: None,

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::GuardTrue,
                args: smallvec::smallvec![OpRef::from_raw(3)],
                descr: None,
                pos: OpRef::NONE,
                fail_args: Some(smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(2)]),

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::IntSub,
                args: smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(10_001)],
                descr: None,
                pos: OpRef::from_raw(4),
                fail_args: None,

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::Jump,
                args: smallvec::smallvec![OpRef::from_raw(4), OpRef::from_raw(2)],
                descr: None,
                pos: OpRef::NONE,
                fail_args: None,


                fail_arg_types: None,
                rd_resume_position: -1,
            },
        ];
        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 100);
        constants.insert(10_001, 1);
        let output = format_trace(&ops, &constants);

        // Verify every op is present
        assert!(output.contains("Label(v0, v1)"));
        assert!(output.contains("v2 = IntAdd(v0, v1)"));
        assert!(output.contains("v3 = IntLt(v2, 100)"));
        assert!(output.contains("GuardTrue(v3) [v0, v2]"));
        assert!(output.contains("v4 = IntSub(v0, 1)"));
        assert!(output.contains("Jump(v4, v2)"));

        // Verify line count (6 ops = 6 lines)
        assert_eq!(output.lines().count(), 6);
    }

    #[test]
    fn test_format_trace_multiple_guards_with_different_fail_args() {
        // Multiple guards in a single trace, each with distinct fail_args.
        let ops = vec![
            op! {
                opcode: OpCode::GuardTrue,
                args: smallvec::smallvec![OpRef::from_raw(0)],
                descr: None,
                pos: OpRef::NONE,
                fail_args: Some(smallvec::smallvec![OpRef::from_raw(0)]),

                fail_arg_types: None,
                rd_resume_position: -1,
            },
            op! {
                opcode: OpCode::GuardFalse,
                args: smallvec::smallvec![OpRef::from_raw(1)],
                descr: None,
                pos: OpRef::NONE,
                fail_args: Some(smallvec::smallvec![OpRef::from_raw(0), OpRef::from_raw(1), OpRef::from_raw(2)]),


                fail_arg_types: None,
                rd_resume_position: -1,
            },
        ];
        let constants = std::collections::HashMap::new();
        let output = format_trace(&ops, &constants);
        assert!(output.contains("GuardTrue(v0) [v0]"));
        assert!(output.contains("GuardFalse(v1) [v0, v1, v2]"));
    }

    #[test]
    fn test_is_setarrayitem() {
        assert!(OpCode::SetarrayitemGc.is_setarrayitem());
        assert!(OpCode::SetarrayitemRaw.is_setarrayitem());
        assert!(!OpCode::GetarrayitemGcI.is_setarrayitem());
        assert!(!OpCode::IntAdd.is_setarrayitem());
    }

    #[test]
    fn test_is_setfield() {
        assert!(OpCode::SetfieldGc.is_setfield());
        assert!(OpCode::SetfieldRaw.is_setfield());
        assert!(!OpCode::GetfieldGcI.is_setfield());
    }

    #[test]
    fn test_is_getinteriorfield() {
        assert!(OpCode::GetinteriorfieldGcI.is_getinteriorfield());
        assert!(OpCode::GetinteriorfieldGcR.is_getinteriorfield());
        assert!(OpCode::GetinteriorfieldGcF.is_getinteriorfield());
        assert!(!OpCode::SetinteriorfieldGc.is_getinteriorfield());
    }

    #[test]
    fn test_is_setinteriorfield() {
        assert!(OpCode::SetinteriorfieldGc.is_setinteriorfield());
        assert!(!OpCode::GetinteriorfieldGcI.is_setinteriorfield());
    }

    // ══════════════════════════════════════════════════════════════════
    // AbstractValue parity tests
    // Mirror rpython/jit/metainterp/test/test_history.py and
    // rpython/jit/metainterp/test/test_resoperation.py for the
    // AbstractValue / Const / InputArg / ResOp class hierarchy.
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_abstract_value_ty_const() {
        assert_eq!(AbstractValue::ConstInt(0).ty(), Some(Type::Int));
        assert_eq!(AbstractValue::ConstFloat(0).ty(), Some(Type::Float));
        assert_eq!(AbstractValue::ConstPtr(0).ty(), Some(Type::Ref));
    }

    #[test]
    fn test_abstract_value_ty_input_arg() {
        assert_eq!(AbstractValue::InputArgInt(0).ty(), Some(Type::Int));
        assert_eq!(AbstractValue::InputArgFloat(0).ty(), Some(Type::Float));
        assert_eq!(AbstractValue::InputArgRef(0).ty(), Some(Type::Ref));
    }

    #[test]
    fn test_abstract_value_ty_res_op() {
        assert_eq!(AbstractValue::IntOp(0).ty(), Some(Type::Int));
        assert_eq!(AbstractValue::FloatOp(0).ty(), Some(Type::Float));
        assert_eq!(AbstractValue::RefOp(0).ty(), Some(Type::Ref));
        assert_eq!(AbstractValue::VoidOp(0).ty(), Some(Type::Void));
    }

    #[test]
    fn test_abstract_value_ty_none() {
        assert_eq!(AbstractValue::None.ty(), None);
    }

    #[test]
    fn test_abstract_value_is_constant() {
        assert!(AbstractValue::ConstInt(7).is_constant());
        assert!(AbstractValue::ConstFloat(7).is_constant());
        assert!(AbstractValue::ConstPtr(7).is_constant());
        assert!(!AbstractValue::InputArgInt(7).is_constant());
        assert!(!AbstractValue::IntOp(7).is_constant());
        assert!(!AbstractValue::None.is_constant());
    }

    #[test]
    fn test_abstract_value_is_input_arg() {
        assert!(AbstractValue::InputArgInt(7).is_input_arg());
        assert!(AbstractValue::InputArgFloat(7).is_input_arg());
        assert!(AbstractValue::InputArgRef(7).is_input_arg());
        assert!(!AbstractValue::ConstInt(7).is_input_arg());
        assert!(!AbstractValue::IntOp(7).is_input_arg());
        assert!(!AbstractValue::None.is_input_arg());
    }

    #[test]
    fn test_abstract_value_is_res_op() {
        assert!(AbstractValue::IntOp(7).is_res_op());
        assert!(AbstractValue::FloatOp(7).is_res_op());
        assert!(AbstractValue::RefOp(7).is_res_op());
        assert!(!AbstractValue::ConstInt(7).is_res_op());
        assert!(!AbstractValue::InputArgInt(7).is_res_op());
        assert!(!AbstractValue::None.is_res_op());
    }

    #[test]
    fn test_abstract_value_raw() {
        assert_eq!(AbstractValue::ConstInt(11).raw(), Some(11));
        assert_eq!(AbstractValue::InputArgRef(22).raw(), Some(22));
        assert_eq!(AbstractValue::IntOp(33).raw(), Some(33));
        assert_eq!(AbstractValue::None.raw(), None);
    }

    #[test]
    fn test_abstract_value_disjoint_categories() {
        // RPython parity: Const / AbstractInputArg / AbstractResOp
        // are disjoint sub-hierarchies under AbstractValue.
        let const_iv = AbstractValue::ConstInt(0);
        let input_iv = AbstractValue::InputArgInt(0);
        let res_iv = AbstractValue::IntOp(0);
        assert!(const_iv.is_constant() && !const_iv.is_input_arg() && !const_iv.is_res_op());
        assert!(!input_iv.is_constant() && input_iv.is_input_arg() && !input_iv.is_res_op());
        assert!(!res_iv.is_constant() && !res_iv.is_input_arg() && res_iv.is_res_op());
    }

    // ── Typed OpRef constructors (Phase 2A) ──

    #[test]
    fn typed_const_constructors_share_raw_with_legacy_but_keep_variant_distinct() {
        // history.py:244 `ConstInt.same_constant` rejects `ConstFloat` /
        // `ConstPtr` — Const sub-classes are disjoint identities even
        // at matching raw payloads.
        for idx in [0u32, 1, 7, 100, 0x0FFF_FFFF] {
            assert_eq!(OpRef::const_int(idx).raw(), OpRef::from_const(idx).raw());
            assert_eq!(OpRef::const_float(idx).raw(), OpRef::from_const(idx).raw());
            assert_eq!(OpRef::const_ptr(idx).raw(), OpRef::from_const(idx).raw());
            assert_ne!(OpRef::const_int(idx), OpRef::from_const(idx));
            assert_ne!(OpRef::const_float(idx), OpRef::from_const(idx));
            assert_ne!(OpRef::const_ptr(idx), OpRef::from_const(idx));
            assert_ne!(OpRef::const_int(idx), OpRef::const_float(idx));
            assert_ne!(OpRef::const_int(idx), OpRef::const_ptr(idx));
            assert_ne!(OpRef::const_float(idx), OpRef::const_ptr(idx));
        }
    }

    #[test]
    fn typed_input_arg_constructors_share_raw_with_legacy_but_keep_variant_distinct() {
        // resoperation.py:719/727/739 `InputArg{Int,Float,Ref}` are
        // disjoint Box classes; the enum discriminant rejects
        // cross-variant identity even at matching raw payloads.
        for pos in [0u32, 1, 7, 100] {
            assert_eq!(OpRef::input_arg_int(pos).raw(), OpRef::from_raw(pos).raw());
            assert_eq!(
                OpRef::input_arg_float(pos).raw(),
                OpRef::from_raw(pos).raw()
            );
            assert_eq!(OpRef::input_arg_ref(pos).raw(), OpRef::from_raw(pos).raw());
            assert_ne!(OpRef::input_arg_int(pos), OpRef::from_raw(pos));
            assert_ne!(OpRef::input_arg_float(pos), OpRef::from_raw(pos));
            assert_ne!(OpRef::input_arg_ref(pos), OpRef::from_raw(pos));
            assert_ne!(OpRef::input_arg_int(pos), OpRef::input_arg_float(pos));
            assert_ne!(OpRef::input_arg_int(pos), OpRef::input_arg_ref(pos));
            assert_ne!(OpRef::input_arg_float(pos), OpRef::input_arg_ref(pos));
        }
    }

    #[test]
    fn typed_op_result_constructors_share_raw_with_legacy_but_keep_variant_distinct() {
        // resoperation.py:564-638 `IntOp` / `FloatOp` / `RefOp` mixins:
        // each ResOp's `.type` is fixed by the mixin class, so the typed
        // factories must not collapse onto type-erased `from_raw`.
        for pos in [0u32, 1, 7, 100, 1_000_000] {
            assert_eq!(OpRef::int_op(pos).raw(), OpRef::from_raw(pos).raw());
            assert_eq!(OpRef::float_op(pos).raw(), OpRef::from_raw(pos).raw());
            assert_eq!(OpRef::ref_op(pos).raw(), OpRef::from_raw(pos).raw());
            assert_ne!(OpRef::int_op(pos), OpRef::from_raw(pos));
            assert_ne!(OpRef::float_op(pos), OpRef::from_raw(pos));
            assert_ne!(OpRef::ref_op(pos), OpRef::from_raw(pos));
            assert_ne!(OpRef::int_op(pos), OpRef::float_op(pos));
            assert_ne!(OpRef::int_op(pos), OpRef::ref_op(pos));
            assert_ne!(OpRef::float_op(pos), OpRef::ref_op(pos));
        }
    }

    #[test]
    fn test_typed_constructors_classification() {
        // is_constant() distinguishes Const family from the rest.
        assert!(OpRef::const_int(0).is_constant());
        assert!(OpRef::const_float(0).is_constant());
        assert!(OpRef::const_ptr(0).is_constant());
        assert!(!OpRef::input_arg_int(0).is_constant());
        assert!(!OpRef::input_arg_float(0).is_constant());
        assert!(!OpRef::input_arg_ref(0).is_constant());
        assert!(!OpRef::int_op(0).is_constant());
        assert!(!OpRef::float_op(0).is_constant());
        assert!(!OpRef::ref_op(0).is_constant());
    }

    #[test]
    fn test_typed_const_round_trip_via_const_index() {
        for idx in [0u32, 1, 7, 100, 0x0FFF_FFFF] {
            assert_eq!(OpRef::const_int(idx).const_index(), idx);
            assert_eq!(OpRef::const_float(idx).const_index(), idx);
            assert_eq!(OpRef::const_ptr(idx).const_index(), idx);
        }
    }

    /// Locks in the bit-helper byte-equivalence with the OpRef-construction
    /// path so callers in index-keyed pools (Slice P3 cat E) can replace
    /// `OpRef::from_raw(raw).is_constant()` / `.const_index()` with the
    /// raw helpers without semantic drift.
    #[test]
    fn test_raw_is_constant_matches_opref_path() {
        for idx in [0u32, 1, 7, 100, 0x0FFF_FFFF] {
            let raw = OpRef::const_int(idx).raw();
            assert!(OpRef::raw_is_constant(raw));
            assert_eq!(OpRef::raw_const_index(raw), idx);
            assert_eq!(
                OpRef::raw_const_index(raw),
                OpRef::const_int(idx).const_index()
            );
        }
        // Non-constant raw values: op positions, inputarg positions, plain numbers.
        for raw in [0u32, 1, 7, 100, 0x0FFF_FFFF] {
            assert!(!OpRef::raw_is_constant(raw));
        }
        // Sentinel range stays out of the constant namespace.
        assert!(!OpRef::raw_is_constant(u32::MAX));
        assert!(!OpRef::raw_is_constant(u32::MAX - 1));
    }
}
