//! RPython `rpython/rtyper/lltypesystem/lltype.py`.
#![allow(non_camel_case_types, non_snake_case)]
//!
//! Currently ports two surfaces:
//! * Function-pointer surface consumed by `translator/simplify.py:get_graph`:
//!   [`_ptr`], [`_func`], [`FuncType`], [`functionptr`], [`getfunctionptr`],
//!   [`_getconcretetype`].
//! * [`LowLevelType`] primitive enum consumed by `rmodel.py`'s
//!   `Repr.lowleveltype` attribute and by `inputconst(reqtype, value)`.
//!   The Rust adaptation collapses upstream's class hierarchy
//!   (`LowLevelType` → `Primitive` / `Number` / `Ptr` / `Struct` / `Array`
//!   at `lltype.py:98,642,665,721,...`) into an enum so `Repr`
//!   implementations can pattern-match on kind without Rust trait-object
//!   downcasts. The three variants currently populated (`Void`, `Bool`,
//!   `Signed`, `Float`, `Char`, `UniChar`, `Unsigned`, `SingleFloat`,
//!   `LongFloat`, `SignedLongLong`, `UnsignedLongLong`, `Ptr`) cover every
//!   type used by `rpbc.py FunctionRepr` / `rclass.py InstanceRepr` /
//!   `FunctionReprBase.call` — additional container kinds (`Struct`,
//!   `Array`, `ForwardReference`) land with the commit that consumes
//!   them.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use crate::annotator::model::KnownType;
use crate::flowspace::model::{ConcretetypePlaceholder, ConstValue, GraphKey, GraphRef, Hlvalue};
use crate::translator::rtyper::error::TyperError;

thread_local! {
    /// RPython `_ptr._become()` mutates the pointer object in place so
    /// every alias starts resolving to the real target. Rust `_ptr`
    /// values are plain cloned structs, so the narrow annlowlevel port
    /// records the post-`_become` target by pointer identity here and
    /// routes read-side operations through the redirect.
    static PTR_BECOME_TARGETS: RefCell<HashMap<u64, _ptr>> = RefCell::new(HashMap::new());
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DelayedPointer;

/// RPython `frozendict` (`lltype.py:90-95`): a dict whose hash is
/// computed from sorted items. This keeps `_flds` / `_adtmeths` /
/// `_hints` order-insensitive for equality and hashing while `_names`
/// carries the explicit field order, as in upstream `Struct.__init__`.
#[derive(Clone, Debug)]
pub struct FrozenDict<V> {
    items: Vec<(String, V)>,
}

impl<V> FrozenDict<V> {
    pub fn new(items: Vec<(String, V)>) -> Self {
        let mut seen: Vec<String> = Vec::with_capacity(items.len());
        for (key, _) in &items {
            if seen.iter().any(|existing| existing == key) {
                panic!("frozendict duplicate key {:?}", key);
            }
            seen.push(key.clone());
        }
        FrozenDict { items }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        self.items
            .iter()
            .find(|(existing, _)| existing == key)
            .map(|(_, value)| value)
    }

    pub fn first(&self) -> Option<(&String, &V)> {
        self.items.first().map(|(key, value)| (key, value))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &V)> {
        self.items.iter().map(|(key, value)| (key, value))
    }

    pub fn to_vec(&self) -> Vec<(String, V)>
    where
        V: Clone,
    {
        self.items.clone()
    }
}

impl<V> From<Vec<(String, V)>> for FrozenDict<V> {
    fn from(value: Vec<(String, V)>) -> Self {
        FrozenDict::new(value)
    }
}

impl<'a, V> IntoIterator for &'a FrozenDict<V> {
    type Item = (&'a String, &'a V);
    type IntoIter =
        std::iter::Map<std::slice::Iter<'a, (String, V)>, fn(&(String, V)) -> (&String, &V)>;

    fn into_iter(self) -> Self::IntoIter {
        fn as_refs<V>((key, value): &(String, V)) -> (&String, &V) {
            (key, value)
        }
        self.items.iter().map(as_refs::<V>)
    }
}

impl<V: PartialEq> PartialEq for FrozenDict<V> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self.items.iter().all(|(key, value)| {
                other
                    .get(key)
                    .is_some_and(|other_value| other_value == value)
            })
    }
}

impl<V: Eq> Eq for FrozenDict<V> {}

impl<V: Hash> Hash for FrozenDict<V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut items: Vec<(&String, &V)> = self.iter().collect();
        items.sort_by(|(left_key, _), (right_key, _)| left_key.cmp(right_key));
        items.len().hash(state);
        for (key, value) in items {
            key.hash(state);
            value.hash(state);
        }
    }
}

/// RPython `lltype.Void` singleton surface. Pure re-export of the enum
/// variant so `rmodel.rs` / `rpbc.rs` / `rclass.rs` ports can mirror
/// upstream `from rpython.rtyper.lltypesystem.lltype import Void` reads.
pub const VOID: LowLevelType = LowLevelType::Void;
/// RPython `lltype.Bool` singleton surface.
pub const BOOL: LowLevelType = LowLevelType::Bool;
/// RPython `lltype.Signed` singleton surface.
pub const SIGNED: LowLevelType = LowLevelType::Signed;
/// RPython `lltype.Float` singleton surface.
pub const FLOAT: LowLevelType = LowLevelType::Float;
/// RPython `lltype.Char` singleton surface.
pub const CHAR: LowLevelType = LowLevelType::Char;
/// RPython `lltype.UniChar` singleton surface.
pub const UNICHAR: LowLevelType = LowLevelType::UniChar;

impl LowLevelType {
    /// RPython `isinstance(self, Primitive)` — true for the scalar
    /// types Number / Bool / Float / Char / UniChar / Void. `Ptr` is
    /// NOT a Primitive in upstream's class hierarchy.
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            LowLevelType::Void
                | LowLevelType::Signed
                | LowLevelType::Unsigned
                | LowLevelType::SignedLongLong
                | LowLevelType::SignedLongLongLong
                | LowLevelType::UnsignedLongLong
                | LowLevelType::UnsignedLongLongLong
                | LowLevelType::Bool
                | LowLevelType::Float
                | LowLevelType::SingleFloat
                | LowLevelType::LongFloat
                | LowLevelType::Char
                | LowLevelType::UniChar
                | LowLevelType::Address
        )
    }

    /// RPython `LowLevelType._contains_value(value)` — used by
    /// `Repr.convert_const` (`rmodel.py:122`) and by `inputconst`
    /// (`rmodel.py:390`) as the "does this low-level type admit this
    /// Python value as a prebuilt constant" check.
    ///
    /// Upstream dispatches through each subclass's `_enforce` /
    /// `_contains_value` implementation; the Rust port pattern-matches
    /// on variant + [`ConstValue`]. Returns `true` if `value` is a
    /// valid constant of kind `self`. Unsupported variants (e.g. rich
    /// container wrappers outside the covered set) conservatively
    /// accept [`ConstValue::Placeholder`] and reject everything else,
    /// matching upstream's TyperError raising surface downstream.
    pub fn contains_value(&self, value: &ConstValue) -> bool {
        // Upstream special-cases `Placeholder` (used by normalizecalls
        // sentinel `description.NODEFAULT`) as a universally acceptable
        // constant while its holder recomputes the real type. Mirror
        // that tolerance so the normalizecalls rewrite branch does not
        // trip convert_const validation during mid-pipeline rewrites.
        if matches!(value, ConstValue::Placeholder) {
            return true;
        }
        match self {
            // upstream `lltype.py:194-197` — `Void._contains_value`
            // returns True for *any* value, not only None. Repr.py's
            // convert_const path relies on this to allow Void slots to
            // carry arbitrary Python sentinels during constant folding.
            LowLevelType::Void => true,
            // upstream `Bool = Primitive("Bool", False)`.
            LowLevelType::Bool => matches!(value, ConstValue::Bool(_)),
            // upstream `Signed` / `Unsigned` / `SignedLongLong` /
            // `SignedLongLongLong` / `UnsignedLongLong` /
            // `UnsignedLongLongLong` all accept Python `int` (with range
            // checking upstream; pyre's `ConstValue::Int` is already i64
            // so the only check left is category match).
            LowLevelType::Signed
            | LowLevelType::Unsigned
            | LowLevelType::SignedLongLong
            | LowLevelType::SignedLongLongLong
            | LowLevelType::UnsignedLongLong
            | LowLevelType::UnsignedLongLongLong => matches!(value, ConstValue::Int(_)),
            // upstream `Float` / `SingleFloat` / `LongFloat` accept
            // Python `float`.
            LowLevelType::Float | LowLevelType::SingleFloat | LowLevelType::LongFloat => {
                matches!(value, ConstValue::Float(_))
            }
            LowLevelType::Char => matches!(value, ConstValue::ByteStr(s) if s.len() == 1),
            LowLevelType::UniChar => {
                matches!(value, ConstValue::UniStr(s) if s.chars().count() == 1)
            }
            // Address is a Primitive with `_defl = NULL`. Concrete
            // values are llmemory.fakeaddress instances; pyre carries
            // them as `ConstValue::LLAddress`.
            LowLevelType::Address => matches!(value, ConstValue::LLAddress(_)),
            // upstream `Ptr(FuncType)` accepts `_ptr` instances — pyre's
            // `ConstValue::LLPtr` is the direct equivalent.
            LowLevelType::Ptr(_) => matches!(value, ConstValue::LLPtr(_)),
            LowLevelType::Struct(_)
            | LowLevelType::Array(_)
            | LowLevelType::FixedSizeArray(_)
            | LowLevelType::Opaque(_)
            | LowLevelType::Func(_)
            | LowLevelType::ForwardReference(_)
            | LowLevelType::InteriorPtr(_) => false,
        }
    }

    /// RPython `LowLevelType._short_name` (`lltype.py:172-173` default,
    /// `lltype.py:563-566` FuncType override, `lltype.py:748` Ptr
    /// override). Used by Repr's diagnostic messages (`rmodel.py:30,123`).
    /// Primitive types fall back to their class name; FuncType composes
    /// args/result recursively; Ptr prefixes with `"Ptr "` (not `"* "`,
    /// which is `Ptr.__str__`).
    pub fn short_name(&self) -> String {
        match self {
            LowLevelType::Void => "Void".to_string(),
            LowLevelType::Bool => "Bool".to_string(),
            LowLevelType::Signed => "Signed".to_string(),
            LowLevelType::Unsigned => "Unsigned".to_string(),
            LowLevelType::SignedLongLong => "SignedLongLong".to_string(),
            LowLevelType::SignedLongLongLong => "SignedLongLongLong".to_string(),
            LowLevelType::UnsignedLongLong => "UnsignedLongLong".to_string(),
            LowLevelType::UnsignedLongLongLong => "UnsignedLongLongLong".to_string(),
            LowLevelType::Float => "Float".to_string(),
            LowLevelType::SingleFloat => "SingleFloat".to_string(),
            LowLevelType::LongFloat => "LongFloat".to_string(),
            LowLevelType::Char => "Char".to_string(),
            LowLevelType::UniChar => "UniChar".to_string(),
            LowLevelType::Address => "Address".to_string(),
            LowLevelType::Struct(t) => t._short_name(),
            LowLevelType::Array(t) => t._short_name(),
            LowLevelType::FixedSizeArray(t) => t._short_name(),
            LowLevelType::Opaque(t) => t.tag.clone(),
            LowLevelType::Func(t) => t._short_name(),
            LowLevelType::ForwardReference(t) => t
                .resolved()
                .map_or_else(|| "ForwardReference".to_string(), |real| real.short_name()),
            // upstream `Ptr._short_name = "Ptr %s" % (self.TO._short_name(),)`
            // (`lltype.py:748`). Ptr.__str__ is `"* %s"` — a different
            // method, not used here.
            LowLevelType::Ptr(ptr) => format!("Ptr {}", ptr._to_short_name()),
            // upstream `_InteriorPtr._short_name` is not directly
            // defined — falls back to `LowLevelType._short_name` =
            // `str(self)` = class name. Match with `"InteriorPtr"`.
            LowLevelType::InteriorPtr(_) => "InteriorPtr".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AttributeError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GcKind {
    Raw,
    Gc,
    Prebuilt,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LowLevelValue {
    Void,
    Signed(i64),
    Unsigned(u64),
    Bool(bool),
    Float(u64),
    SingleFloat(u32),
    LongFloat(u64),
    Char(char),
    UniChar(char),
    /// RPython `llmemory.Address` values — `fakeaddress(ptr)` / `NULL`.
    /// The NULL sentinel is [`_address::Null`]; richer `fakeaddress`
    /// bodies will extend [`_address`] as llmemory wrappers land.
    Address(_address),
    Struct(Box<_struct>),
    Array(Box<_array>),
    Opaque(Box<_opaque>),
    Ptr(Box<_ptr>),
    InteriorPtr(Box<_interior_ptr>),
}

/// RPython `_fakeaddress(self, ptr)` (`llmemory.py:441-570`).
///
/// Upstream carries the underlying `_ptr` (or `None` for NULL) and
/// exposes `_cast_to_ptr` / `_cast_to_int` / dereference helpers.
/// Pyre carries the two arms consumers actually hit today — the NULL
/// sentinel + a `Fake(_ptr)` wrapper produced by
/// [`MultipleUnrelatedFrozenPBCRepr.convert_pbc`] via
/// `llmemory.fakeaddress(pbcptr)`. Cast / dereference helpers extend
/// `_address` later as further consumers land.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum _address {
    /// `NULL = fakeaddress(None)` (`llmemory.py:649`).
    Null,
    /// `fakeaddress(ptr)` — wraps a live `_ptr` so it can flow
    /// through `Address`-typed slots (e.g. `MUFrozenPBCRepr` whose
    /// lowleveltype is `llmemory.Address`). Upstream stores this on
    /// the dict-by-identity `converted_pbc_cache`; pyre keys the same
    /// cache by `DescKey` and stores the [`_address::Fake`] payload
    /// directly.
    ///
    /// `_address` no longer derives `Hash` because `_ptr` does not
    /// have a stable structural hash; the `Fake` arm carries a Box
    /// to keep the variant fixed-size.
    Fake(Box<_ptr>),
}

#[derive(Clone, Debug)]
pub enum LowLevelType {
    Void,
    Signed,
    Unsigned,
    SignedLongLong,
    SignedLongLongLong,
    UnsignedLongLong,
    UnsignedLongLongLong,
    Bool,
    Float,
    SingleFloat,
    LongFloat,
    Char,
    UniChar,
    /// RPython `Address = lltype.Primitive("Address", NULL)`
    /// (`llmemory.py:650`). Represents the primitive address type used
    /// by `MultipleUnrelatedFrozenPBCRepr.lowleveltype` and `adr_eq` /
    /// `adr_ne` operations. Values are [`LowLevelValue::Address`].
    Address,
    Func(Box<FuncType>),
    Struct(Box<StructType>),
    Array(Box<ArrayType>),
    FixedSizeArray(Box<FixedSizeArrayType>),
    Opaque(Box<OpaqueType>),
    ForwardReference(Box<ForwardReference>),
    Ptr(Box<Ptr>),
    InteriorPtr(Box<InteriorPtr>),
}

impl PartialEq for LowLevelType {
    fn eq(&self, other: &Self) -> bool {
        // Cycle short-circuit: two ForwardReferences sharing the
        // same Arc compare equal without descending into resolved
        // types (closes `OBJECT_VTABLE → instantiate → OBJECTPTR →
        // OBJECT → CLASSTYPE → OBJECT_VTABLE`).
        if let (LowLevelType::ForwardReference(left_fwd), LowLevelType::ForwardReference(right_fwd)) =
            (self, other)
            && Arc::ptr_eq(&left_fwd.target, &right_fwd.target)
        {
            return true;
        }
        // Saferecursive guard — when one side is a ForwardReference
        // and we're already comparing it elsewhere on the stack
        // (cycle through the resolved type), short-circuit to `true`.
        // Mirrors RPython `saferecursive(safe_equal, True)` from
        // `lltype.py:74-95` — re-entering the same comparison means
        // the outer call hasn't returned False, so the optimistic
        // cycle assumption is "equal". The re-entry case happens for
        // `Struct == ForwardReference` asymmetric pairs where the
        // Struct contains a Ptr looping back to the same fwd.
        if let LowLevelType::ForwardReference(forward_ref) = self
            && let Some(real) = forward_ref.resolved()
        {
            let id = Arc::as_ptr(&forward_ref.target) as *const _ as usize;
            if FORWARD_REF_EQ_GUARD.with(|g| g.borrow().contains(&id)) {
                return true;
            }
            FORWARD_REF_EQ_GUARD.with(|g| g.borrow_mut().insert(id));
            let r = real == *other;
            FORWARD_REF_EQ_GUARD.with(|g| g.borrow_mut().remove(&id));
            return r;
        }
        if let LowLevelType::ForwardReference(forward_ref) = other
            && let Some(real) = forward_ref.resolved()
        {
            let id = Arc::as_ptr(&forward_ref.target) as *const _ as usize;
            if FORWARD_REF_EQ_GUARD.with(|g| g.borrow().contains(&id)) {
                return true;
            }
            FORWARD_REF_EQ_GUARD.with(|g| g.borrow_mut().insert(id));
            let r = *self == real;
            FORWARD_REF_EQ_GUARD.with(|g| g.borrow_mut().remove(&id));
            return r;
        }
        match (self, other) {
            (LowLevelType::Void, LowLevelType::Void)
            | (LowLevelType::Signed, LowLevelType::Signed)
            | (LowLevelType::Unsigned, LowLevelType::Unsigned)
            | (LowLevelType::SignedLongLong, LowLevelType::SignedLongLong)
            | (LowLevelType::SignedLongLongLong, LowLevelType::SignedLongLongLong)
            | (LowLevelType::UnsignedLongLong, LowLevelType::UnsignedLongLong)
            | (LowLevelType::UnsignedLongLongLong, LowLevelType::UnsignedLongLongLong)
            | (LowLevelType::Bool, LowLevelType::Bool)
            | (LowLevelType::Float, LowLevelType::Float)
            | (LowLevelType::SingleFloat, LowLevelType::SingleFloat)
            | (LowLevelType::LongFloat, LowLevelType::LongFloat)
            | (LowLevelType::Char, LowLevelType::Char)
            | (LowLevelType::UniChar, LowLevelType::UniChar)
            | (LowLevelType::Address, LowLevelType::Address) => true,
            (LowLevelType::Func(left), LowLevelType::Func(right)) => left == right,
            (LowLevelType::Struct(left), LowLevelType::Struct(right)) => left == right,
            (LowLevelType::Array(left), LowLevelType::Array(right)) => left == right,
            (LowLevelType::FixedSizeArray(left), LowLevelType::FixedSizeArray(right)) => {
                left == right
            }
            (LowLevelType::Opaque(left), LowLevelType::Opaque(right)) => left == right,
            (LowLevelType::ForwardReference(left), LowLevelType::ForwardReference(right)) => {
                left == right
            }
            (LowLevelType::Ptr(left), LowLevelType::Ptr(right)) => left == right,
            (LowLevelType::InteriorPtr(left), LowLevelType::InteriorPtr(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for LowLevelType {}

impl Hash for LowLevelType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            LowLevelType::Void => 0_u8.hash(state),
            LowLevelType::Signed => 1_u8.hash(state),
            LowLevelType::Unsigned => 2_u8.hash(state),
            LowLevelType::SignedLongLong => 3_u8.hash(state),
            LowLevelType::SignedLongLongLong => 4_u8.hash(state),
            LowLevelType::UnsignedLongLong => 5_u8.hash(state),
            LowLevelType::UnsignedLongLongLong => 6_u8.hash(state),
            LowLevelType::Bool => 7_u8.hash(state),
            LowLevelType::Float => 8_u8.hash(state),
            LowLevelType::SingleFloat => 9_u8.hash(state),
            LowLevelType::LongFloat => 10_u8.hash(state),
            LowLevelType::Char => 11_u8.hash(state),
            LowLevelType::UniChar => 12_u8.hash(state),
            LowLevelType::Func(t) => {
                13_u8.hash(state);
                t.hash(state);
            }
            LowLevelType::Struct(t) => {
                14_u8.hash(state);
                t.hash(state);
            }
            LowLevelType::Array(t) => {
                15_u8.hash(state);
                t.hash(state);
            }
            LowLevelType::FixedSizeArray(t) => {
                16_u8.hash(state);
                t.hash(state);
            }
            LowLevelType::Opaque(t) => {
                17_u8.hash(state);
                t.hash(state);
            }
            LowLevelType::ForwardReference(t) => {
                // Saferecursive cycle guard — when the resolved type
                // contains a Ptr that loops back to this same
                // ForwardReference (e.g. `OBJECT_VTABLE.instantiate
                // → Ptr(FuncType([], OBJECTPTR)) → OBJECT.typeptr →
                // CLASSTYPE → OBJECT_VTABLE`), hashing `real`
                // unconditionally recurses forever. Mirrors RPython
                // `saferecursive(get_hash, 0)` (lltype.py:136) — on
                // re-entry hash a constant 0 so deterministic across
                // runs and identity-independent.
                let id = Arc::as_ptr(&t.target) as *const _ as usize;
                let already = FORWARD_REF_HASH_GUARD.with(|g| g.borrow().contains(&id));
                if already {
                    0_u8.hash(state);
                    return;
                }
                if let Some(real) = t.resolved() {
                    FORWARD_REF_HASH_GUARD.with(|g| g.borrow_mut().insert(id));
                    real.hash(state);
                    FORWARD_REF_HASH_GUARD.with(|g| g.borrow_mut().remove(&id));
                } else {
                    18_u8.hash(state);
                    t.hash(state);
                }
            }
            LowLevelType::Ptr(t) => {
                19_u8.hash(state);
                t.hash(state);
            }
            LowLevelType::InteriorPtr(t) => {
                20_u8.hash(state);
                t.hash(state);
            }
            LowLevelType::Address => 21_u8.hash(state),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FuncType {
    pub args: Vec<ConcretetypePlaceholder>,
    pub result: ConcretetypePlaceholder,
}

/// RPython `Struct`/`GcStruct` (`lltype.py:258-380`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StructType {
    pub _name: String,
    pub _flds: FrozenDict<ConcretetypePlaceholder>,
    pub _names: Vec<String>,
    pub _adtmeths: FrozenDict<ConstValue>,
    pub _hints: FrozenDict<ConstValue>,
    pub _arrayfld: Option<String>,
    pub _gckind: GcKind,
    /// RPython `RttiStruct._runtime_type_info` (`lltype.py:382-389`).
    /// `None` for plain `Struct`/`GcStruct` without rtti; populated by
    /// `StructType::gc_rtti` (or a later `_install_extras(rtti=True)`
    /// port) with a freshly-minted opaque whose identity distinguishes
    /// two structurally-equal `GcStruct(..., rtti=True)` builds — the
    /// same distinction upstream Python makes via per-instance
    /// `_runtime_type_info` attrs.
    pub _runtime_type_info: Option<Box<_opaque>>,
}

/// RPython `Array`/`GcArray` (`lltype.py:420-489`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArrayType {
    pub OF: ConcretetypePlaceholder,
    pub _hints: FrozenDict<ConstValue>,
    pub _gckind: GcKind,
}

/// RPython `FixedSizeArray` (`lltype.py:491-540`) — structurally a
/// `Struct` with fields `item0..itemN-1`. The Rust port keeps `OF` and
/// `length` direct so lookups match array-indexing semantics without
/// walking a `StructType._flds` list.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FixedSizeArrayType {
    pub OF: ConcretetypePlaceholder,
    pub length: usize,
    pub _hints: FrozenDict<ConstValue>,
    pub _gckind: GcKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OpaqueType {
    pub tag: String,
    pub _gckind: GcKind,
}

/// Flattens upstream's three sibling classes
/// `ForwardReference(_gckind='raw')`, `GcForwardReference(_gckind='gc')`,
/// and `FuncForwardReference(_gckind='prebuilt')` (`lltype.py:615-635`)
/// into one struct with an explicit `_gckind` field. Upstream `become()`
/// rebinds `self.__class__` / `self.__dict__` to the realcontainertype
/// (`lltype.py:624-625`) so every pointer to the same `ForwardReference`
/// observes the resolved type. Rust cannot re-tag enum variants in place,
/// so clones share a mutable target cell and pointer-op sites unwrap it.
#[derive(Clone, Debug)]
pub struct ForwardReference {
    pub _gckind: GcKind,
    target: Arc<Mutex<Option<LowLevelType>>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum InteriorOffset {
    Field(String),
    Index(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InteriorPtr {
    pub PARENTTYPE: Box<LowLevelType>,
    pub TO: Box<LowLevelType>,
    pub offsets: Vec<InteriorOffset>,
}

#[derive(Clone, Debug)]
pub enum PtrTarget {
    Func(FuncType),
    Struct(StructType),
    Array(ArrayType),
    FixedSizeArray(FixedSizeArrayType),
    Opaque(OpaqueType),
    ForwardReference(ForwardReference),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ptr {
    pub TO: PtrTarget,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LowLevelPointerType {
    Ptr(Ptr),
    InteriorPtr(InteriorPtr),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LowLevelAdtMember {
    Value(ConstValue),
    Method {
        ll_ptrtype: LowLevelPointerType,
        func: ConstValue,
    },
}

impl From<Ptr> for LowLevelType {
    fn from(value: Ptr) -> Self {
        LowLevelType::Ptr(Box::new(value))
    }
}

impl From<InteriorPtr> for LowLevelType {
    fn from(value: InteriorPtr) -> Self {
        LowLevelType::InteriorPtr(Box::new(value))
    }
}

impl From<LowLevelPointerType> for LowLevelType {
    fn from(value: LowLevelPointerType) -> Self {
        match value {
            LowLevelPointerType::Ptr(ptr) => LowLevelType::Ptr(Box::new(ptr)),
            LowLevelPointerType::InteriorPtr(ptr) => LowLevelType::InteriorPtr(Box::new(ptr)),
        }
    }
}

impl From<PtrTarget> for LowLevelType {
    fn from(value: PtrTarget) -> Self {
        match value {
            PtrTarget::Func(func) => LowLevelType::Func(Box::new(func)),
            PtrTarget::Struct(t) => LowLevelType::Struct(Box::new(t)),
            PtrTarget::Array(t) => LowLevelType::Array(Box::new(t)),
            PtrTarget::FixedSizeArray(t) => LowLevelType::FixedSizeArray(Box::new(t)),
            PtrTarget::Opaque(t) => LowLevelType::Opaque(Box::new(t)),
            PtrTarget::ForwardReference(t) => LowLevelType::ForwardReference(Box::new(t)),
        }
    }
}

impl PartialEq for PtrTarget {
    fn eq(&self, other: &Self) -> bool {
        LowLevelType::from(self.clone()) == LowLevelType::from(other.clone())
    }
}

impl Eq for PtrTarget {}

impl Hash for PtrTarget {
    fn hash<H: Hasher>(&self, state: &mut H) {
        LowLevelType::from(self.clone()).hash(state);
    }
}

#[derive(Clone, Debug)]
pub struct _func {
    pub TYPE: FuncType,
    pub _name: String,
    pub graph: Option<usize>,
    pub _callable: Option<String>,
    pub attrs: HashMap<String, ConstValue>,
}

impl _func {
    pub fn new(
        TYPE: FuncType,
        _name: String,
        graph: Option<usize>,
        _callable: Option<String>,
        attrs: HashMap<String, ConstValue>,
    ) -> Self {
        _func {
            TYPE,
            _name,
            graph,
            _callable,
            attrs,
        }
    }
}

/// `_struct._fields` is wrapped in `Arc<Mutex<...>>` so a `_struct`
/// `#[derive(Clone)]` shares its field storage with its clones.
/// Mirrors RPython's pointer-by-identity semantics on Python objects:
/// `iprebuiltinstances[value] = result; init(value, result)` (rclass.py
/// :799-800) requires that mutations to `result` mid-init are visible
/// to recursive `convert_const(value)` cache hits.
///
/// Without the Arc-share, `_ptr.clone()` deep-copies `_obj0` →
/// `_struct.clone()` → `_fields.clone()` (owned `Vec`), so two clones
/// of the same prebuilt instance carry independent field storage and
/// intra-class circular prebuilt instances (`a.b.back == a`) wire
/// `back` to a stale snapshot rather than the in-progress `a`. See
/// task #157 — this is the structural pre-requisite for
/// `InstanceRepr::convert_const_exact` byte-for-byte parity with
/// `rclass.py:794-802`.
///
/// `Arc<Mutex<>>` (rather than `Rc<RefCell<>>`) because
/// `LowLevelValue` carriers (e.g. `static LazyLock<LowLevelType>` for
/// `STR`/`UNICODE`) traverse the Send/Sync boundary; `_struct` itself
/// must therefore be Send+Sync. Single-threaded RPython has no need
/// for the lock; pyre's lock acquisitions are uncontended in practice
/// (the tracer/codewriter share the per-process repr cache without
/// concurrent mutation of any single `_struct`).
#[derive(Clone, Debug)]
pub struct _struct {
    pub _identity: usize,
    pub TYPE: StructType,
    pub _fields: Arc<Mutex<Vec<(String, LowLevelValue)>>>,
}

/// `_array.items` is wrapped in `Arc<Mutex<...>>` for the same shared-
/// storage discipline as `_struct._fields` — see the `_struct` doc
/// above for the parity rationale (task #157).
#[derive(Clone, Debug)]
pub struct _array {
    pub _identity: usize,
    pub TYPE: ArrayContainer,
    pub items: Arc<Mutex<Vec<LowLevelValue>>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ArrayContainer {
    Array(ArrayType),
    FixedSizeArray(FixedSizeArrayType),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum _ptr_obj {
    Func(_func),
    Struct(_struct),
    Array(_array),
    Opaque(_opaque),
}

#[derive(Clone, Debug)]
pub struct _opaque {
    pub _identity: usize,
    pub TYPE: OpaqueType,
    /// Optional human-readable name — `_opaque(TYPE, _name=name, **attrs)`
    /// kwarg upstream (used by `opaqueptr` to mint named containers such
    /// as `GCSTRUCT._runtime_type_info`). Plain `_opaque(TYPE)` defaults
    /// to the upstream `"?"` marker.
    pub _name: Option<String>,
    /// Upstream `opaqueptr(..., about=self)` stores the source type on
    /// the opaque itself (`lltype.py:387-389`).
    pub about: Option<LowLevelType>,
    /// Upstream `_attach_runtime_type_info_funcptr()` stores validated
    /// helper pointers directly on the RTTI opaque (`lltype.py:405-415`).
    pub query_funcptr: Option<Box<_ptr>>,
    pub destructor_funcptr: Option<Box<_ptr>>,
}

#[derive(Clone, Debug)]
pub struct _interior_ptr {
    pub _T: LowLevelType,
    pub _parent: LowLevelValue,
    pub _offsets: Vec<InteriorOffset>,
}

impl PartialEq for _interior_ptr {
    fn eq(&self, other: &Self) -> bool {
        if self._TYPE() != other._TYPE() {
            panic!("comparing {:?} and {:?}", self._TYPE(), other._TYPE());
        }
        self._obj() == other._obj()
    }
}

impl Eq for _interior_ptr {}

impl _func {
    /// RPython `_abstract_ptr.__call__` / `_func` arg validation
    /// (`lltype.py:1349-1385`). Arity mismatch and per-arg type
    /// mismatch raise `TypeError`; a None `_callable` raises
    /// `RuntimeError`. The Rust port models `_callable` as an optional
    /// name string rather than a real closure, so the return value is
    /// the result type's default — matching the `_container_example`
    /// closure `def ex(*args): return self.RESULT._defl()`.
    pub fn call(&self, args: &[LowLevelValue]) -> LowLevelValue {
        if args.len() != self.TYPE.args.len() {
            panic!(
                "calling {:?} with wrong argument number: got {}, expected {}",
                self.TYPE,
                args.len(),
                self.TYPE.args.len(),
            );
        }
        for (i, (arg, expected)) in args.iter().zip(self.TYPE.args.iter()).enumerate() {
            let got = typeOf_value(arg);
            if got == *expected {
                continue;
            }
            // upstream lltype.py:1357-1376 special cases:
            // - Void arg: `ARG == Void` accepts any value.
            // - Pointer arg expected, None actual: accepted.
            // - ContainerType expected, Ptr(ContainerType) actual:
            //   accepted (backends unwrap).
            if matches!(expected, LowLevelType::Void) {
                continue;
            }
            if matches!(expected, LowLevelType::Ptr(_)) && matches!(arg, LowLevelValue::Void) {
                continue;
            }
            if expected.is_container_type()
                && let LowLevelValue::Ptr(ptr) = arg
            {
                let expected_ptr = LowLevelType::Ptr(Box::new(Ptr {
                    TO: match expected {
                        LowLevelType::Func(t) => PtrTarget::Func((**t).clone()),
                        LowLevelType::Struct(t) => PtrTarget::Struct((**t).clone()),
                        LowLevelType::Array(t) => PtrTarget::Array((**t).clone()),
                        LowLevelType::FixedSizeArray(t) => PtrTarget::FixedSizeArray((**t).clone()),
                        LowLevelType::Opaque(t) => PtrTarget::Opaque((**t).clone()),
                        LowLevelType::ForwardReference(t) => {
                            PtrTarget::ForwardReference((**t).clone())
                        }
                        _ => unreachable!(),
                    },
                }));
                if expected_ptr == LowLevelType::Ptr(Box::new(ptr._TYPE.clone())) {
                    continue;
                }
            }
            panic!(
                "calling {:?} with wrong argument type at index {}: expected {:?}, got {:?}",
                self.TYPE, i, expected, got
            );
        }
        if self._callable.is_none() {
            panic!("calling undefined function {:?}", self._name);
        }
        self.TYPE.result._defl()
    }
}

/// Upstream `_container` inherits Python `object`'s identity-based
/// `__eq__` / `__hash__` (`lltype.py:1634-1649`); two `_struct`
/// instances compare equal iff they are the same Python object
/// (`id(self) == id(other)`). Rust `#[derive(Clone)]` on `_ptr._obj0`
/// duplicates container bodies at different addresses, so address-based
/// identity cannot survive a clone. We assign each container a stable
/// `_identity: usize` at `_container_example()` time and key `PartialEq`
/// / `Hash` / `_identityhash` off it — preserving the `is`-comparison
/// semantics `lltype.py:1387-1391` (`_ptr._identityhash`) relies on.
fn fresh_low_level_container_identity() -> usize {
    static NEXT_LOW_LEVEL_CONTAINER_ID: AtomicUsize = AtomicUsize::new(1);
    NEXT_LOW_LEVEL_CONTAINER_ID.fetch_add(1, Ordering::Relaxed)
}

fn fresh_low_level_pointer_identity() -> u64 {
    static NEXT_LOW_LEVEL_POINTER_ID: AtomicUsize = AtomicUsize::new(1);
    NEXT_LOW_LEVEL_POINTER_ID.fetch_add(1, Ordering::Relaxed) as u64
}

impl PartialEq for _struct {
    fn eq(&self, other: &Self) -> bool {
        self._identity == other._identity
    }
}

impl Eq for _struct {}

impl Hash for _struct {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self._identity.hash(state);
    }
}

impl PartialEq for _array {
    fn eq(&self, other: &Self) -> bool {
        self._identity == other._identity
    }
}

impl Eq for _array {}

impl Hash for _array {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self._identity.hash(state);
    }
}

impl PartialEq for _opaque {
    fn eq(&self, other: &Self) -> bool {
        self._identity == other._identity
    }
}

impl Eq for _opaque {}

impl Hash for _opaque {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self._identity.hash(state);
    }
}

impl _struct {
    /// `_struct._fields` is an `Arc<Mutex<...>>`-shared cell so the
    /// returned value must be cloned out of the lock guard. Cheap for
    /// the primitive variants of `LowLevelValue`; `Struct`/`Array`
    /// variants Clone the inner `_struct`/`_array` (which themselves
    /// share their inner `_fields`/`items` via `Arc::clone` per the
    /// shared-storage discipline).
    pub fn _getattr(&self, field_name: &str) -> Option<LowLevelValue> {
        self._fields
            .lock()
            .unwrap()
            .iter()
            .find(|(name, _)| name == field_name)
            .map(|(_, value)| value.clone())
    }

    pub fn _setattr(&self, field_name: &str, value: LowLevelValue) -> bool {
        let mut fields = self._fields.lock().unwrap();
        let Some((_, slot)) = fields.iter_mut().find(|(name, _)| name == field_name) else {
            return false;
        };
        *slot = value;
        true
    }

    /// Recursively descend through nested Struct fields named by `path`
    /// and `_setattr(field, value)` at the bottom. Returns `false` when
    /// any path step is not a Struct field. Used by
    /// [`_ptr::setattr_at_path`] to mutate vtable substructs in place
    /// without going through `_ptr.getattr` (which returns a detached
    /// copy for non-Gc Struct fields).
    pub fn _setattr_descending(&self, path: &[&str], field: &str, value: LowLevelValue) -> bool {
        if path.is_empty() {
            return self._setattr(field, value);
        }
        // Pull the inner Struct out of the lock guard before recursing
        // so the outer guard is released — the inner `_struct` carries
        // its own `Arc<Mutex<_fields>>` cell, and `LowLevelValue::Struct`
        // Clone shares the same Arc storage by `Arc::clone`.
        let inner = {
            let fields = self._fields.lock().unwrap();
            let Some((_, slot)) = fields.iter().find(|(name, _)| name == path[0]) else {
                return false;
            };
            let LowLevelValue::Struct(inner) = slot else {
                return false;
            };
            (**inner).clone()
        };
        inner._setattr_descending(&path[1..], field, value)
    }
}

impl _array {
    pub fn getlength(&self) -> usize {
        self.items.lock().unwrap().len()
    }

    pub fn getbounds(&self) -> (usize, usize) {
        (0, self.items.lock().unwrap().len())
    }

    /// Cloned out of the lock guard — same rationale as `_struct::_getattr`.
    pub fn getitem(&self, index: usize) -> Option<LowLevelValue> {
        self.items.lock().unwrap().get(index).cloned()
    }

    pub fn setitem(&self, index: usize, value: LowLevelValue) -> bool {
        let mut items = self.items.lock().unwrap();
        let Some(slot) = items.get_mut(index) else {
            return false;
        };
        *slot = value;
        true
    }
}

#[allow(dead_code)]
fn ptr_from_parent_type(parent_type: &LowLevelType) -> Ptr {
    match parent_type {
        LowLevelType::Struct(t) => Ptr {
            TO: PtrTarget::Struct((**t).clone()),
        },
        LowLevelType::Array(t) => Ptr {
            TO: PtrTarget::Array((**t).clone()),
        },
        LowLevelType::FixedSizeArray(t) => Ptr {
            TO: PtrTarget::FixedSizeArray((**t).clone()),
        },
        LowLevelType::Opaque(t) => Ptr {
            TO: PtrTarget::Opaque((**t).clone()),
        },
        LowLevelType::ForwardReference(t) => Ptr {
            TO: PtrTarget::ForwardReference((**t).clone()),
        },
        other => panic!("InteriorPtr parent must be a container type, got {other:?}"),
    }
}

/// RPython `_func.__eq__` (`lltype.py:2121-2123`) overrides `_container`'s
/// identity default with structural comparison on `self.__dict__ ==
/// other.__dict__`. Two `_func`s from separate `functionptr()` /
/// `_container_example()` calls that share `_TYPE`/`_name`/`_callable`/
/// `graph` compare equal — this is how `build_concrete_calltable`'s
/// `lookup` merges rows pointing at the same graph.
impl PartialEq for _func {
    fn eq(&self, other: &Self) -> bool {
        self.TYPE == other.TYPE
            && self._name == other._name
            && self._callable == other._callable
            && self.graph == other.graph
            && self.attrs == other.attrs
    }
}

impl Eq for _func {}

impl Hash for _func {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.TYPE.hash(state);
        self._name.hash(state);
        self._callable.hash(state);
        self.graph.hash(state);
        let mut attrs: Vec<_> = self.attrs.iter().collect();
        attrs.sort_by(|(a, _), (b, _)| a.cmp(b));
        for (key, value) in attrs {
            key.hash(state);
            value.hash(state);
        }
    }
}

#[derive(Clone, Debug)]
pub struct _ptr {
    pub _identity: u64,
    pub _TYPE: Ptr,
    pub _solid: bool,
    pub _obj0: Result<Option<_ptr_obj>, DelayedPointer>,
    /// RPython `self._set_weak(False)` from `_ptr.__init__`
    /// (lltype.py:1410-1413). `_become` asserts `not self._weak`
    /// (lltype.py:1416-1418); reserved for the weak-ptr variants
    /// (`weakref_create` family) when those land.
    pub _weak: bool,
}

/// RPython `_abstract_ptr.__eq__` (`lltype.py:1185-1195`). Order:
/// 1. `_TYPE` mismatch → `TypeError` (Rust panic).
/// 2. Either side `DelayedPointer` → fall back to pointer identity.
/// 3. Otherwise compare `_obj0` values directly — null-null is equal,
///    null-nonnull is unequal, and non-null values use `_ptr_obj` eq
///    (which respects `_container` identity for Struct/Array/Opaque
///    and `_func.__eq__`'s structural dict comparison for Func).
impl PartialEq for _ptr {
    fn eq(&self, other: &Self) -> bool {
        let self_resolved = self._resolved_ptr();
        let other_resolved = other._resolved_ptr();
        if self_resolved._TYPE != other_resolved._TYPE {
            panic!(
                "comparing {:?} and {:?}",
                self_resolved._TYPE, other_resolved._TYPE
            );
        }
        match (&self_resolved._obj0, &other_resolved._obj0) {
            (Ok(a), Ok(b)) => a == b,
            _ => self._identity == other._identity,
        }
    }
}

impl Eq for _ptr {}

impl _ptr {
    pub fn new(_TYPE: Ptr, _obj0: Result<Option<_ptr_obj>, DelayedPointer>) -> Self {
        Self::new_with_solid(_TYPE, _obj0, false)
    }

    pub fn new_with_solid(
        _TYPE: Ptr,
        _obj0: Result<Option<_ptr_obj>, DelayedPointer>,
        _solid: bool,
    ) -> Self {
        _ptr {
            _identity: fresh_low_level_pointer_identity(),
            _TYPE,
            _solid,
            _obj0,
            _weak: false,
        }
    }

    pub fn _hashable_identity(&self) -> u64 {
        self._identity
    }

    pub fn _togckind(&self) -> GcKind {
        self._TYPE._gckind()
    }

    pub fn _needsgc(&self) -> bool {
        self._TYPE._needsgc()
    }

    fn _resolved_ptr(&self) -> _ptr {
        let mut current = self.clone();
        loop {
            let next = PTR_BECOME_TARGETS
                .with(|targets| targets.borrow().get(&current._identity).cloned());
            match next {
                Some(target) => current = target,
                None => return current,
            }
        }
    }

    /// RPython `_ptr._become(self, other)` (`lltype.py:1416-1419`).
    ///
    /// ```python
    /// def _become(self, other):
    ///     assert self._TYPE == other._TYPE
    ///     assert not self._weak
    ///     self._setobj(other._obj, other._solid)
    /// ```
    ///
    /// The Python object mutates in place. Rust `_ptr` values are copied
    /// by clone, so this records the resolved target by `_identity` and
    /// read-side operations consult the redirect table.
    pub fn _become(&self, other: &_ptr) {
        assert_eq!(
            self._TYPE, other._TYPE,
            "_ptr._become: type mismatch (self={:?}, other={:?})",
            self._TYPE, other._TYPE,
        );
        assert!(!self._weak, "_ptr._become: cannot reassign a weak pointer",);
        let resolved = other._resolved_ptr();
        PTR_BECOME_TARGETS.with(|targets| {
            targets.borrow_mut().insert(self._identity, resolved);
        });
    }

    /// RPython `_abstract_ptr._getobj` (`lltype.py:1226-1240`). Returns
    /// the underlying container (non-null required by callers that
    /// dereference); delayed pointers surface as `Err(DelayedPointer)`.
    /// Null pointers panic at the dereference site — upstream raises
    /// `AttributeError` on `self._obj.getattr(...)` implicitly, so
    /// either outcome is a programmer error.
    pub fn _obj(&self) -> Result<_ptr_obj, DelayedPointer> {
        let resolved = self._resolved_ptr();
        match resolved._obj0 {
            Ok(Some(obj)) => Ok(obj),
            Ok(None) => panic!("null low-level pointer has no underlying object"),
            Err(_) => Err(DelayedPointer),
        }
    }

    /// RPython `_abstract_ptr._same_obj` (`lltype.py:1200-1201`).
    /// Compares `_obj` values directly (handles null-null equality)
    /// and surfaces `DelayedPointer` for either-side delayed.
    pub fn _same_obj(&self, other: &_ptr) -> Result<bool, DelayedPointer> {
        let self_resolved = self._resolved_ptr();
        let other_resolved = other._resolved_ptr();
        match (&self_resolved._obj0, &other_resolved._obj0) {
            (Ok(a), Ok(b)) => Ok(a == b),
            _ => Err(DelayedPointer),
        }
    }

    pub fn nonzero(&self) -> bool {
        match &self._resolved_ptr()._obj0 {
            Ok(Some(_)) => true,
            Ok(None) => false,
            Err(DelayedPointer) => true,
        }
    }

    /// RPython `_ptr._cast_to(self, PTRTYPE)` (`lltype.py:1421-1451`).
    ///
    /// Walks the struct super-chain to produce a re-typed pointer that
    /// aliases the same underlying allocation. Down-casts (positive
    /// `castable` depth) walk via the first field name (`super`);
    /// up-casts via `_parentstructure()`.
    ///
    /// Pyre-port scope: down-casts and identity casts are supported.
    /// Null casts return `Ptr(target)._defl()`. Up-casts surface a
    /// structured error pending a `_parent_link` port — exception
    /// classes only ever down-cast for `ll_cast_to_object`, which is
    /// the immediate consumer.
    pub fn _cast_to(&self, ptrtype: &Ptr) -> Result<_ptr, String> {
        let down_or_up = castable(ptrtype, &self._TYPE)?;
        if down_or_up == 0 {
            return Ok(self.clone());
        }
        // upstream: `if not self: return PTRTYPE._defl()`.
        if !self.nonzero() {
            return Ok(ptrtype._defl());
        }
        if down_or_up < 0 {
            return Err(format!(
                "_ptr._cast_to: up-cast (depth={down_or_up}) requires \
                 _parentstructure() — not yet ported",
            ));
        }
        // upstream: `while down_or_up: p = getattr(p, typeOf(p).TO._names[0])`.
        // The first field of a Struct in our port is the leading entry
        // of `_flds`; for the instance/class hierarchy that is
        // `("super", parent_struct)`. Walk by repeated `_obj.getattr`
        // re-pointing through the chain.
        let mut current = self.clone();
        let mut steps = down_or_up;
        while steps > 0 {
            let to_struct = match &current._TYPE.TO {
                PtrTarget::Struct(s) => s.clone(),
                PtrTarget::ForwardReference(fwd) => match fwd.resolved() {
                    Some(LowLevelType::Struct(s)) => *s,
                    _ => {
                        return Err(format!(
                            "_ptr._cast_to: ForwardReference {fwd:?} did not \
                             resolve to a Struct"
                        ));
                    }
                },
                _ => {
                    return Err(format!(
                        "_ptr._cast_to: cast walk requires Struct target, got {:?}",
                        current._TYPE
                    ));
                }
            };
            let first_name = to_struct._names.first().ok_or_else(|| {
                format!(
                    "_ptr._cast_to: struct {:?} has no fields to walk",
                    to_struct._name
                )
            })?;
            let lv = current.getattr(first_name)?;
            let LowLevelValue::Ptr(next) = lv else {
                return Err(format!(
                    "_ptr._cast_to: vtable.{first_name} did not yield a Ptr value"
                ));
            };
            current = *next;
            steps -= 1;
        }
        // Re-wrap with the target Ptr type while preserving the
        // underlying object and solid bit. Upstream `_ptr(PTRTYPE,
        // p._obj, solid=self._solid)`.
        let obj0 = current._obj0.clone();
        Ok(_ptr::new_with_solid(ptrtype.clone(), obj0, self._solid))
    }

    pub fn _identityhash(&self) -> i64 {
        assert_eq!(self._TYPE._gckind(), GcKind::Gc);
        assert!(self.nonzero());
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self._obj()
            .expect("gc identityhash requires a concrete underlying object")
            .hash(&mut hasher);
        hasher.finish() as i64
    }

    pub fn call(&self, args: &[LowLevelValue]) -> LowLevelValue {
        match self
            ._obj()
            .expect("low-level function pointer example must expose an underlying object")
        {
            _ptr_obj::Func(func) => func.call(args),
            _ptr_obj::Struct(_) => panic!("{:?} instance is not callable", self._TYPE),
            _ptr_obj::Array(_) => panic!("{:?} instance is not callable", self._TYPE),
            _ptr_obj::Opaque(_) => panic!("{:?} instance is not callable", self._TYPE),
        }
    }

    /// RPython `_abstract_ptr._fixedlength` (`lltype.py:1331-1336`).
    /// Always calls `len(self)` first, which raises `TypeError` for
    /// non-array pointers (so `_fixedlength` cannot silently succeed
    /// on a struct/function/opaque pointer). Returns `Ok(Some(n))` for
    /// FixedSizeArray and `Ok(None)` for plain `Array`.
    pub fn _fixedlength(&self) -> Result<Option<i64>, String> {
        let length = self.len()?;
        let is_fixed_size = match &self._TYPE.TO {
            PtrTarget::FixedSizeArray(_) => true,
            PtrTarget::ForwardReference(forward_ref) => matches!(
                forward_ref.resolved(),
                Some(LowLevelType::FixedSizeArray(_))
            ),
            _ => false,
        };
        Ok(if is_fixed_size {
            Some(length as i64)
        } else {
            None
        })
    }

    fn _as_parent_value(&self) -> LowLevelValue {
        match self
            ._obj()
            .expect("low-level pointer example must expose an underlying object")
        {
            _ptr_obj::Func(_) => panic!("function pointer has no container parent"),
            _ptr_obj::Struct(obj) => LowLevelValue::Struct(Box::new(obj.clone())),
            _ptr_obj::Array(obj) => LowLevelValue::Array(Box::new(obj.clone())),
            _ptr_obj::Opaque(obj) => LowLevelValue::Opaque(Box::new(obj.clone())),
        }
    }

    /// Delegate to [`Ptr::_interior_ptr_type_with_index`] — the method
    /// is defined on the pointer *type* upstream; the value wrapper is
    /// kept here so existing `_ptr`-side call sites stay ergonomic.
    pub fn _interior_ptr_type_with_index(&self, to: &LowLevelType) -> StructType {
        self._TYPE._interior_ptr_type_with_index(to)
    }

    pub fn _expose(&self, offset: InteriorOffset, value: LowLevelValue) -> LowLevelValue {
        let wrap_as_interior_ptr = self._TYPE._gckind() == GcKind::Gc;
        match value {
            LowLevelValue::Struct(obj) => {
                if wrap_as_interior_ptr && obj.TYPE._gckind == GcKind::Raw {
                    LowLevelValue::InteriorPtr(Box::new(_interior_ptr {
                        _T: LowLevelType::Struct(Box::new(obj.TYPE.clone())),
                        _parent: self._as_parent_value(),
                        _offsets: vec![offset],
                    }))
                } else {
                    LowLevelValue::Ptr(Box::new(_ptr::new_with_solid(
                        Ptr {
                            TO: PtrTarget::Struct(obj.TYPE.clone()),
                        },
                        Ok(Some(_ptr_obj::Struct(*obj))),
                        self._solid,
                    )))
                }
            }
            LowLevelValue::Array(obj) => {
                let target = match &obj.TYPE {
                    ArrayContainer::Array(array_t) => PtrTarget::Array(array_t.clone()),
                    ArrayContainer::FixedSizeArray(array_t) => {
                        PtrTarget::FixedSizeArray(array_t.clone())
                    }
                };
                if wrap_as_interior_ptr && target._gckind() == GcKind::Raw {
                    LowLevelValue::InteriorPtr(Box::new(_interior_ptr {
                        _T: LowLevelType::from(target.clone()),
                        _parent: self._as_parent_value(),
                        _offsets: vec![offset],
                    }))
                } else {
                    LowLevelValue::Ptr(Box::new(_ptr::new_with_solid(
                        Ptr { TO: target },
                        Ok(Some(_ptr_obj::Array(*obj))),
                        self._solid,
                    )))
                }
            }
            LowLevelValue::Opaque(obj) => LowLevelValue::Ptr(Box::new(_ptr::new_with_solid(
                Ptr {
                    TO: PtrTarget::Opaque(obj.TYPE.clone()),
                },
                Ok(Some(_ptr_obj::Opaque(*obj))),
                self._solid,
            ))),
            other => other,
        }
    }

    pub fn _lookup_adtmeth(&self, member_name: &str) -> Result<LowLevelAdtMember, AttributeError> {
        match &self._TYPE.TO {
            PtrTarget::Struct(struct_t) => {
                struct_t._lookup_adtmeth(member_name, &LowLevelPointerType::Ptr(self._TYPE.clone()))
            }
            PtrTarget::ForwardReference(forward_ref) => {
                let Some(LowLevelType::Struct(struct_t)) = forward_ref.resolved() else {
                    return Err(AttributeError);
                };
                struct_t._lookup_adtmeth(member_name, &LowLevelPointerType::Ptr(self._TYPE.clone()))
            }
            _ => Err(AttributeError),
        }
    }

    pub fn getattr(&self, field_name: &str) -> Result<LowLevelValue, String> {
        if (matches!(&self._TYPE.TO, PtrTarget::Struct(_))
            || matches!(
                &self._TYPE.TO,
                PtrTarget::ForwardReference(forward_ref)
                    if matches!(forward_ref.resolved(), Some(LowLevelType::Struct(_)))
            ))
            && let _ptr_obj::Struct(obj) = self._obj().map_err(|_| {
                format!(
                    "delayed pointer {:?} has no field {:?}",
                    self._TYPE, field_name
                )
            })?
            && let Some(value) = obj._getattr(field_name)
        {
            return Ok(self._expose(InteriorOffset::Field(field_name.to_owned()), value.clone()));
        }
        match self._lookup_adtmeth(field_name) {
            Ok(_) => {
                panic!("_ptr.__getattr__ adtmethod path must be handled by SomePtr.getattr first")
            }
            Err(AttributeError) => Err(self._TYPE._nofield(field_name)),
        }
    }

    pub fn getitem(&self, index: usize) -> Result<LowLevelValue, String> {
        if matches!(
            &self._TYPE.TO,
            PtrTarget::Array(_) | PtrTarget::FixedSizeArray(_)
        ) || matches!(
            &self._TYPE.TO,
            PtrTarget::ForwardReference(forward_ref)
                if matches!(
                    forward_ref.resolved(),
                    Some(LowLevelType::Array(_) | LowLevelType::FixedSizeArray(_))
                )
        ) {
            let _ptr_obj::Array(obj) = self
                ._obj()
                .map_err(|_| format!("delayed pointer {:?} is not an array", self._TYPE))?
            else {
                panic!("array pointer must expose an array object");
            };
            if let Some(value) = obj.getitem(index) {
                return Ok(self._expose(InteriorOffset::Index(index), value.clone()));
            }
            return Err(format!("array index out of bounds: {index}"));
        }
        Err(format!("{:?} instance is not an array", self._TYPE))
    }

    pub fn setattr(&mut self, field_name: &str, val: LowLevelValue) -> Result<(), String> {
        let struct_t = match &self._TYPE.TO {
            PtrTarget::Struct(struct_t) => Some(struct_t.clone()),
            PtrTarget::ForwardReference(forward_ref) => match forward_ref.resolved() {
                Some(LowLevelType::Struct(struct_t)) => Some(*struct_t),
                _ => None,
            },
            _ => None,
        };
        if let Some(struct_t) = &struct_t
            && let Some(field_t) = struct_t.getattr_field_type(field_name)
        {
            let got_t = typeOf_value(&val);
            if field_t != got_t {
                return Err(format!(
                    "{} instance field {:?}:\nexpects {:?}\n    got {:?}",
                    struct_t._name, field_name, field_t, got_t
                ));
            }
            let obj = self
                ._obj0
                .as_mut()
                .map_err(|_| {
                    format!(
                        "delayed pointer {:?} has no field {:?}",
                        self._TYPE, field_name
                    )
                })?
                .as_mut()
                .expect("non-null struct pointer must expose an underlying object");
            let _ptr_obj::Struct(obj) = obj else {
                panic!("struct pointer must expose a struct object");
            };
            if !obj._setattr(field_name, val) {
                panic!("struct object must expose declared field {field_name:?}");
            }
            return Ok(());
        }
        Err(self._TYPE._nofield(field_name))
    }

    /// Variant of [`Self::setattr`] that descends through nested Struct
    /// fields named by `path` before writing `field`. The pointer
    /// continues to alias the same underlying allocation (unlike
    /// `_ptr.getattr("super")` which returns a detached copy of the
    /// substruct). Required by `ClassRepr.init_vtable` to write
    /// `subclassrange_min/max` / `rtti` on the OBJECT_VTABLE substruct
    /// nested inside a per-class vtable allocation.
    pub fn setattr_at_path(
        &mut self,
        path: &[&str],
        field: &str,
        val: LowLevelValue,
    ) -> Result<(), String> {
        let obj = self
            ._obj0
            .as_mut()
            .map_err(|_| {
                format!(
                    "delayed pointer {:?} has no nested field {:?}",
                    self._TYPE, field
                )
            })?
            .as_mut()
            .expect("non-null struct pointer must expose an underlying object");
        let _ptr_obj::Struct(obj) = obj else {
            return Err(format!(
                "setattr_at_path: pointer {:?} does not target a Struct",
                self._TYPE
            ));
        };
        if !obj._setattr_descending(path, field, val) {
            return Err(format!(
                "setattr_at_path: nested write at path={:?} field={:?} failed",
                path, field
            ));
        }
        Ok(())
    }

    pub fn setitem(&mut self, index: usize, val: LowLevelValue) -> Result<(), String> {
        let expected = match &self._TYPE.TO {
            PtrTarget::Array(array_t) => Some(array_t.OF.clone()),
            PtrTarget::FixedSizeArray(array_t) => Some(array_t.OF.clone()),
            PtrTarget::ForwardReference(forward_ref) => match forward_ref.resolved() {
                Some(LowLevelType::Array(array_t)) => Some(array_t.OF.clone()),
                Some(LowLevelType::FixedSizeArray(array_t)) => Some(array_t.OF.clone()),
                _ => None,
            },
            _ => None,
        };
        let Some(expected) = expected else {
            return Err(format!("{:?} instance is not an array", self._TYPE));
        };
        let got_t = typeOf_value(&val);
        if expected != got_t {
            return Err(format!(
                "{:?} items:\nexpect {:?}\n   got {:?}",
                self._TYPE, expected, got_t
            ));
        }
        let _ptr_obj::Array(obj) = self
            ._obj0
            .as_mut()
            .map_err(|_| format!("delayed pointer {:?} is not an array", self._TYPE))?
            .as_mut()
            .expect("non-null array pointer must expose an underlying object")
        else {
            panic!("array pointer must expose an array object");
        };
        if !obj.setitem(index, val) {
            return Err(format!("array index out of bounds: {index}"));
        }
        Ok(())
    }

    pub fn len(&self) -> Result<usize, String> {
        if (matches!(
            &self._TYPE.TO,
            PtrTarget::Array(_) | PtrTarget::FixedSizeArray(_)
        ) || matches!(
            &self._TYPE.TO,
            PtrTarget::ForwardReference(forward_ref)
                if matches!(
                    forward_ref.resolved(),
                    Some(LowLevelType::Array(_) | LowLevelType::FixedSizeArray(_))
                )
        )) && let _ptr_obj::Array(obj) = self
            ._obj()
            .map_err(|_| format!("delayed pointer {:?} has no length", self._TYPE))?
        {
            return Ok(obj.getlength());
        }
        Err(format!("{:?} instance has no length attribute", self._TYPE))
    }
}

impl _interior_ptr {
    pub fn _togckind(&self) -> GcKind {
        self._T._gckind()
    }

    pub fn nonzero(&self) -> bool {
        panic!("do not test an interior pointer for nullity")
    }

    pub fn _same_obj(&self, other: &_interior_ptr) -> bool {
        self._obj() == other._obj()
    }

    pub fn _obj(&self) -> LowLevelValue {
        let mut ob = self._parent.clone();
        for offset in &self._offsets {
            ob = match (offset, ob) {
                (InteriorOffset::Field(name), LowLevelValue::Struct(obj)) => obj
                    ._getattr(name)
                    .unwrap_or_else(|| panic!("interior ptr field {name:?} missing")),
                (InteriorOffset::Index(index), LowLevelValue::Array(obj)) => obj
                    .getitem(*index)
                    .unwrap_or_else(|| panic!("interior ptr index path missing item {index}")),
                (offset, other) => panic!("invalid interior ptr offset {offset:?} on {other:?}"),
            };
        }
        ob
    }

    pub fn _TYPE(&self) -> InteriorPtr {
        InteriorPtr {
            PARENTTYPE: Box::new(typeOf_value(&self._parent)),
            TO: Box::new(self._T.clone()),
            offsets: self._offsets.clone(),
        }
    }

    pub fn _lookup_adtmeth(&self, member_name: &str) -> Result<LowLevelAdtMember, AttributeError> {
        let LowLevelValue::Struct(obj) = self._obj() else {
            return Err(AttributeError);
        };
        obj.TYPE
            ._lookup_adtmeth(member_name, &LowLevelPointerType::InteriorPtr(self._TYPE()))
    }

    /// RPython `_abstract_ptr._fixedlength` (`lltype.py:1331-1336`),
    /// inherited by `_interior_ptr`. Always runs `len()` first so
    /// non-array interior pointers surface the upstream `TypeError`.
    pub fn _fixedlength(&self) -> Result<Option<i64>, String> {
        let length = self.len()?;
        match self._obj() {
            LowLevelValue::Array(obj) => match obj.TYPE {
                ArrayContainer::FixedSizeArray(_) => Ok(Some(length as i64)),
                ArrayContainer::Array(_) => Ok(None),
            },
            _ => Ok(None),
        }
    }

    pub fn _expose(&self, offset: InteriorOffset, value: LowLevelValue) -> LowLevelValue {
        let typ = typeOf_value(&value);
        match typ {
            LowLevelType::Struct(_)
            | LowLevelType::Array(_)
            | LowLevelType::FixedSizeArray(_)
            | LowLevelType::Opaque(_) => {
                assert_eq!(typ._gckind(), GcKind::Raw);
                let mut offsets = self._offsets.clone();
                offsets.push(offset);
                LowLevelValue::InteriorPtr(Box::new(_interior_ptr {
                    _T: typ,
                    _parent: self._parent.clone(),
                    _offsets: offsets,
                }))
            }
            _ => value,
        }
    }

    pub fn getattr(&self, field_name: &str) -> Result<LowLevelValue, String> {
        let LowLevelValue::Struct(obj) = self._obj() else {
            return Err(format!("{:?} has no field {:?}", self._TYPE(), field_name));
        };
        let Some(value) = obj._getattr(field_name) else {
            return Err(format!("{:?} has no field {:?}", self._TYPE(), field_name));
        };
        Ok(self._expose(InteriorOffset::Field(field_name.to_owned()), value.clone()))
    }

    pub fn setitem(&mut self, index: usize, val: LowLevelValue) -> Result<(), String> {
        let LowLevelValue::Array(obj) = self._obj() else {
            return Err(format!("{:?} instance is not an array", self._TYPE()));
        };
        let expected = match obj.TYPE {
            ArrayContainer::Array(ref array_t) => array_t.OF.clone(),
            ArrayContainer::FixedSizeArray(ref array_t) => array_t.OF.clone(),
        };
        let got_t = typeOf_value(&val);
        if expected != got_t {
            return Err(format!(
                "{:?} items:\nexpect {:?}\n   got {:?}",
                self._TYPE(),
                expected,
                got_t
            ));
        }
        let LowLevelValue::Array(obj) = self._resolve_mut()? else {
            unreachable!();
        };
        if !obj.setitem(index, val) {
            return Err(format!("array index out of bounds: {index}"));
        }
        Ok(())
    }

    pub fn getitem(&self, index: usize) -> Result<LowLevelValue, String> {
        let LowLevelValue::Array(obj) = self._obj() else {
            return Err(format!("{:?} instance is not an array", self._TYPE()));
        };
        let Some(value) = obj.getitem(index) else {
            return Err(format!("array index out of bounds: {index}"));
        };
        Ok(self._expose(InteriorOffset::Index(index), value.clone()))
    }

    pub fn setattr(&mut self, field_name: &str, val: LowLevelValue) -> Result<(), String> {
        let LowLevelValue::Struct(obj) = self._obj() else {
            return Err(format!("{:?} has no field {:?}", self._TYPE(), field_name));
        };
        let Some(field_t) = obj.TYPE.getattr_field_type(field_name) else {
            return Err(format!("{:?} has no field {:?}", self._TYPE(), field_name));
        };
        let got_t = typeOf_value(&val);
        if field_t != got_t {
            return Err(format!(
                "{:?} field {:?}:\nexpects {:?}\n    got {:?}",
                self._TYPE(),
                field_name,
                field_t,
                got_t
            ));
        }
        let LowLevelValue::Struct(obj) = self._resolve_mut()? else {
            unreachable!();
        };
        if !obj._setattr(field_name, val) {
            return Err(format!("{:?} has no field {:?}", self._TYPE(), field_name));
        }
        Ok(())
    }

    pub fn len(&self) -> Result<usize, String> {
        let LowLevelValue::Array(obj) = self._obj() else {
            return Err(format!(
                "{:?} instance has no length attribute",
                self._TYPE()
            ));
        };
        Ok(obj.getlength())
    }

    pub fn call(&self, _args: &[LowLevelValue]) -> LowLevelValue {
        panic!("{:?} instance is not callable", self._TYPE())
    }

    /// Walks `_parent` by `_offsets` and returns the terminal element
    /// as an owned `LowLevelValue`. The shared-storage discipline of
    /// `_struct._fields` and `_array.items` (`Arc<Mutex<...>>`, see
    /// task #157) means owned clones still alias the underlying
    /// container, so subsequent `_setattr` / `setitem` calls on the
    /// returned owned value mutate the same backing storage as the
    /// original `_parent` chain.
    fn _resolve_mut(&self) -> Result<LowLevelValue, String> {
        fn descend(
            current: LowLevelValue,
            offsets: &[InteriorOffset],
        ) -> Result<LowLevelValue, String> {
            if offsets.is_empty() {
                return Ok(current);
            }
            match (&offsets[0], current) {
                (InteriorOffset::Field(name), LowLevelValue::Struct(obj)) => {
                    let next = obj
                        ._getattr(name)
                        .ok_or_else(|| format!("interior ptr field {name:?} missing"))?;
                    descend(next, &offsets[1..])
                }
                (InteriorOffset::Index(index), LowLevelValue::Array(obj)) => {
                    let next = obj
                        .getitem(*index)
                        .ok_or_else(|| format!("array index out of bounds: {index}"))?;
                    descend(next, &offsets[1..])
                }
                (offset, other) => Err(format!(
                    "invalid interior ptr offset {offset:?} on {other:?}"
                )),
            }
        }
        descend(self._parent.clone(), &self._offsets)
    }
}

impl LowLevelType {
    pub fn _defl(&self) -> LowLevelValue {
        match self {
            LowLevelType::Void => LowLevelValue::Void,
            LowLevelType::Signed => LowLevelValue::Signed(0),
            LowLevelType::Unsigned => LowLevelValue::Unsigned(0),
            LowLevelType::SignedLongLong => LowLevelValue::Signed(0),
            LowLevelType::SignedLongLongLong => LowLevelValue::Signed(0),
            LowLevelType::UnsignedLongLong => LowLevelValue::Unsigned(0),
            LowLevelType::UnsignedLongLongLong => LowLevelValue::Unsigned(0),
            LowLevelType::Bool => LowLevelValue::Bool(false),
            LowLevelType::Float => LowLevelValue::Float(0.0f64.to_bits()),
            LowLevelType::SingleFloat => LowLevelValue::SingleFloat(0.0f32.to_bits()),
            LowLevelType::LongFloat => LowLevelValue::LongFloat(0.0f64.to_bits()),
            LowLevelType::Char => LowLevelValue::Char('\0'),
            LowLevelType::UniChar => LowLevelValue::UniChar('\0'),
            // upstream `Address = Primitive("Address", NULL)` with
            // `NULL = fakeaddress(None)` — `_defl` returns the NULL
            // sentinel.
            LowLevelType::Address => LowLevelValue::Address(_address::Null),
            LowLevelType::Func(_) => {
                panic!("FuncType has no standalone low-level value default")
            }
            LowLevelType::Struct(struct_t) => {
                LowLevelValue::Struct(Box::new(struct_t._container_example()))
            }
            LowLevelType::Array(array_t) => {
                LowLevelValue::Array(Box::new(array_t._container_example()))
            }
            LowLevelType::FixedSizeArray(array_t) => {
                LowLevelValue::Array(Box::new(array_t._container_example()))
            }
            LowLevelType::Opaque(opaque_t) => {
                LowLevelValue::Opaque(Box::new(opaque_t._container_example()))
            }
            LowLevelType::ForwardReference(fwd) => match fwd.resolved() {
                // Once a ForwardReference has `become` its real type, its
                // `_defl()` must mirror the resolved container — upstream
                // mutates `fwd.__class__` / `fwd.__dict__` at `become` so the
                // `_defl` lookup dispatches through the real type directly
                // (lltype.py:624-625). Pyre carries the `ForwardReference`
                // wrapper, so resolve explicitly here.
                Some(resolved) => resolved._defl(),
                None => panic!("ForwardReference must be resolved before _defl()"),
            },
            LowLevelType::Ptr(ptr) => LowLevelValue::Ptr(Box::new(ptr._defl())),
            LowLevelType::InteriorPtr(ptr) => LowLevelValue::InteriorPtr(Box::new(ptr._example())),
        }
    }
}

pub fn typeOf(ptr: &_ptr) -> Ptr {
    ptr._TYPE.clone()
}

pub fn identityhash(p: &_ptr) -> i64 {
    assert!(p.nonzero());
    p._identityhash()
}

pub fn typeOf_value(value: &LowLevelValue) -> ConcretetypePlaceholder {
    match value {
        LowLevelValue::Void => LowLevelType::Void,
        LowLevelValue::Signed(_) => LowLevelType::Signed,
        LowLevelValue::Unsigned(_) => LowLevelType::Unsigned,
        LowLevelValue::Bool(_) => LowLevelType::Bool,
        LowLevelValue::Float(_) => LowLevelType::Float,
        LowLevelValue::SingleFloat(_) => LowLevelType::SingleFloat,
        LowLevelValue::LongFloat(_) => LowLevelType::LongFloat,
        LowLevelValue::Char(_) => LowLevelType::Char,
        LowLevelValue::UniChar(_) => LowLevelType::UniChar,
        LowLevelValue::Address(_) => LowLevelType::Address,
        LowLevelValue::Struct(obj) => LowLevelType::Struct(Box::new(obj.TYPE.clone())),
        LowLevelValue::Array(obj) => match &obj.TYPE {
            ArrayContainer::Array(array_t) => LowLevelType::Array(Box::new(array_t.clone())),
            ArrayContainer::FixedSizeArray(array_t) => {
                LowLevelType::FixedSizeArray(Box::new(array_t.clone()))
            }
        },
        LowLevelValue::Opaque(obj) => LowLevelType::Opaque(Box::new(obj.TYPE.clone())),
        LowLevelValue::Ptr(ptr) => LowLevelType::Ptr(Box::new(typeOf(ptr))),
        LowLevelValue::InteriorPtr(ptr) => LowLevelType::InteriorPtr(Box::new(ptr._TYPE())),
    }
}

impl FuncType {
    pub fn _container_example(&self) -> _func {
        // upstream `FuncType._container_example` (lltype.py:568-571)
        // constructs `_func(self, _callable=ex)` where `ex` is a fresh
        // closure returning `self.RESULT._defl()`. Rust stores
        // `_callable` as an Option<String>; the marker `"<example>"`
        // keeps `_func.call` from tripping the "undefined function"
        // raise while the arg validation still runs.
        _func::new(
            self.clone(),
            "<example>".into(),
            None,
            Some("<example>".into()),
            HashMap::new(),
        )
    }

    /// RPython `FuncType._short_name` (`lltype.py:563-566`) — composes
    /// args/result short-names into `"Func(arg1, arg2, ...)->result"`.
    /// `saferecursive` guard is approximated by not recursing through
    /// `LowLevelType::Func` (returns bare `"Func(...)"`); the full
    /// TLS-backed recursion guard isn't needed until a downstream caller
    /// builds mutually-recursive FuncTypes.
    pub fn _short_name(&self) -> String {
        let args: Vec<String> = self.args.iter().map(LowLevelType::short_name).collect();
        format!("Func({})->{}", args.join(", "), self.result.short_name())
    }
}

impl StructType {
    pub fn new(name: &str, fields: Vec<(String, ConcretetypePlaceholder)>) -> Self {
        Self::_build(name, fields, GcKind::Raw, vec![], vec![])
    }

    pub fn gc(name: &str, fields: Vec<(String, ConcretetypePlaceholder)>) -> Self {
        Self::_build(name, fields, GcKind::Gc, vec![], vec![])
    }

    pub fn with_adtmeths(
        name: &str,
        fields: Vec<(String, ConcretetypePlaceholder)>,
        adtmeths: Vec<(String, ConstValue)>,
    ) -> Self {
        Self::_build(name, fields, GcKind::Raw, adtmeths, vec![])
    }

    /// Raw `Struct(name, *fields, hints={...})`. Upstream
    /// `rpython/rtyper/lltypesystem/lltype.py:258-294 Struct.__init__`
    /// forwards the `hints` kwarg through `_install_extras`.
    pub fn with_hints(
        name: &str,
        fields: Vec<(String, ConcretetypePlaceholder)>,
        hints: Vec<(String, ConstValue)>,
    ) -> Self {
        Self::_build(name, fields, GcKind::Raw, vec![], hints)
    }

    /// `GcStruct(name, *fields, hints={...})`. Same as
    /// [`StructType::gc`] plus the `hints` dict upstream passes through
    /// `Struct.__init__` kwargs.
    pub fn gc_with_hints(
        name: &str,
        fields: Vec<(String, ConcretetypePlaceholder)>,
        hints: Vec<(String, ConstValue)>,
    ) -> Self {
        Self::_build(name, fields, GcKind::Gc, vec![], hints)
    }

    /// `GcStruct(name, *fields, rtti=True)` — upstream
    /// `RttiStruct._install_extras(rtti=True)` (`lltype.py:385-389`) mints
    /// an `_opaque` of type `RuntimeTypeInfo` with `_name` = the struct
    /// name and stores it under `_runtime_type_info`. `getRuntimeTypeInfo`
    /// / `attachRuntimeTypeInfo` later surface it wrapped in
    /// `Ptr(RuntimeTypeInfo)`. Must be called on a fresh struct-type —
    /// the opaque's `_identity` is per-call and therefore per struct.
    pub fn gc_rtti(name: &str, fields: Vec<(String, ConcretetypePlaceholder)>) -> Self {
        Self::gc_rtti_with_hints(name, fields, vec![])
    }

    /// `GcStruct(name, *fields, hints=hints, rtti=True)` — upstream
    /// funnels both options through `Struct.__init__(**kwds)` so they
    /// compose freely (`lltype.py:261-294`). Used e.g. for
    /// `OBJECT = GcStruct('object', ('typeptr', CLASSTYPE),
    /// hints={...}, rtti=True)` (`rclass.py:162-165`).
    pub fn gc_rtti_with_hints(
        name: &str,
        fields: Vec<(String, ConcretetypePlaceholder)>,
        hints: Vec<(String, ConstValue)>,
    ) -> Self {
        let mut result = Self::_build(name, fields, GcKind::Gc, vec![], hints);
        let rtti_ptr = opaqueptr_with_attrs(
            RUNTIME_TYPE_INFO.clone(),
            &result._name,
            Some(LowLevelType::Struct(Box::new(result.clone()))),
        )
        .expect("opaqueptr(RuntimeTypeInfo, ...) must succeed for gc_rtti()");
        let Ok(Some(_ptr_obj::Opaque(rtti_opaque))) = rtti_ptr._obj0 else {
            panic!("opaqueptr(RuntimeTypeInfo, ...) must yield an opaque container");
        };
        result._runtime_type_info = Some(Box::new(rtti_opaque));
        result
    }

    /// Unified constructor mirroring RPython `Struct.__init__` /
    /// `Struct._install_extras` (`lltype.py:261-294, 208-210`). Enforces:
    /// * field names must not begin with `_` (`NameError` upstream);
    /// * repeated field names are rejected (`TypeError` upstream);
    /// * a non-raw container type can only be inlined as the first
    ///   field of a struct with matching `_gckind`;
    /// * the last field is allowed to be varsize (recorded in
    ///   `_arrayfld`).
    fn _build(
        name: &str,
        fields: Vec<(String, ConcretetypePlaceholder)>,
        gckind: GcKind,
        adtmeths: Vec<(String, ConstValue)>,
        hints: Vec<(String, ConstValue)>,
    ) -> Self {
        let mut seen: Vec<String> = Vec::with_capacity(fields.len());
        let first_name = fields.first().map(|(n, _)| n.clone());
        for (i, (fname, ftyp)) in fields.iter().enumerate() {
            if fname.starts_with('_') {
                panic!(
                    "{}: field name {:?} should not start with an underscore",
                    name, fname
                );
            }
            if seen.iter().any(|existing| existing == fname) {
                panic!("{}: repeated field name", name);
            }
            seen.push(fname.clone());
            if ftyp.is_container_type() {
                let child_gc = ftyp._gckind();
                if child_gc != GcKind::Raw {
                    let is_first = i == 0;
                    let same_gc = child_gc == gckind;
                    if !is_first || !same_gc {
                        panic!(
                            "{}: cannot inline {:?} container {:?}",
                            name, child_gc, ftyp
                        );
                    }
                }
            }
        }
        let _ = first_name;
        let _arrayfld = fields.last().and_then(|(n, t)| {
            if t._is_varsize() {
                Some(n.clone())
            } else {
                None
            }
        });
        let names = fields.iter().map(|(n, _)| n.clone()).collect();
        let result = StructType {
            _name: name.into(),
            _flds: FrozenDict::new(fields),
            _names: names,
            _adtmeths: FrozenDict::new(adtmeths),
            _hints: FrozenDict::new(hints),
            _arrayfld,
            _gckind: gckind,
            _runtime_type_info: None,
        };
        let parent = LowLevelType::Struct(Box::new(result.clone()));
        for (i, (_, typ)) in result._flds.iter().enumerate() {
            typ._note_inlined_into(&parent, i == 0, i + 1 == result._flds.len())
                .unwrap_or_else(|err| panic!("{err}"));
        }
        result
    }

    /// RPython `Struct._is_varsize` (`lltype.py:320-321`) — a struct is
    /// varsize iff its last field is a varsize container recorded in
    /// `_arrayfld`.
    pub fn _is_varsize(&self) -> bool {
        self._arrayfld.is_some()
    }

    /// RPython `Struct._short_name` (`lltype.py:358-359`) —
    /// `"<class_name> <struct_name>"` where `class_name` is `Struct` or
    /// `GcStruct` depending on `_gckind`.
    pub fn _short_name(&self) -> String {
        let kind = match self._gckind {
            GcKind::Gc => "GcStruct",
            _ => "Struct",
        };
        format!("{} {}", kind, self._name)
    }

    /// RPython `Struct._first_struct` (`lltype.py:296-303`). Returns the
    /// leading field name and type iff it is a struct of matching
    /// `_gckind`; used by rtyper to walk gc-inlined struct chains.
    pub fn _first_struct(&self) -> Option<(String, &StructType)> {
        let first_name = self._names.first()?;
        let first_type = self._flds.get(first_name)?;
        let LowLevelType::Struct(first_struct) = first_type else {
            return None;
        };
        if self._gckind != first_struct._gckind {
            return None;
        }
        Some((first_name.clone(), first_struct.as_ref()))
    }

    /// Owned variant of [`Self::_first_struct`] that resolves a
    /// leading `ForwardReference` field by cloning out the resolved
    /// struct body. Required by `castdepth` to walk the inheritance
    /// chain produced by `InstanceRepr._setup_repr` (where the
    /// immediate parent is wrapped in `ForwardReference` to support
    /// `_become`-style late resolution).
    pub fn _first_struct_owned(&self) -> Option<(String, StructType)> {
        let first_name = self._names.first()?.clone();
        let first_type = self._flds.get(&first_name)?.clone();
        let first_struct: StructType = match first_type {
            LowLevelType::Struct(boxed) => *boxed,
            LowLevelType::ForwardReference(fwd) => match fwd.resolved()? {
                LowLevelType::Struct(boxed) => *boxed,
                _ => return None,
            },
            _ => return None,
        };
        if self._gckind != first_struct._gckind {
            return None;
        }
        Some((first_name, first_struct))
    }

    /// RPython `Struct._is_atomic` (`lltype.py:314-318`). All fields must
    /// themselves be atomic (primitive or opaque-with-no-gc-children).
    pub fn _is_atomic(&self) -> bool {
        self._flds.iter().all(|(_, typ)| typ._is_atomic())
    }

    /// RPython `Struct._names_without_voids` (`lltype.py:333-334`).
    /// Returns field names whose type is not `Void`.
    pub fn _names_without_voids(&self) -> Vec<String> {
        self._names
            .iter()
            .filter(|name| {
                self._flds
                    .get(name)
                    .is_some_and(|typ| !matches!(typ, LowLevelType::Void))
            })
            .cloned()
            .collect()
    }

    /// RPython `Struct._note_inlined_into` (`lltype.py:305-312`). Checks
    /// structural rules at the parent container's inlining call site.
    /// Varsize structs cannot be inlined; gc structs can only be
    /// inlined as the first field of another gc struct.
    pub fn _note_inlined_into(&self, parent: &LowLevelType, first: bool) -> Result<(), String> {
        if self._arrayfld.is_some() {
            return Err("cannot inline a var-sized struct inside another container".to_string());
        }
        if self._gckind == GcKind::Gc {
            let parent_is_gc_struct = matches!(
                parent,
                LowLevelType::Struct(parent_t) if parent_t._gckind == GcKind::Gc
            );
            if !first || !parent_is_gc_struct {
                return Err(
                    "a GcStruct can only be inlined as the first field of another GcStruct"
                        .to_string(),
                );
            }
        }
        Ok(())
    }

    /// RPython `Struct._immutable_field(field)` (`lltype.py:372-380`).
    /// Returns `true` iff the struct carries `'immutable'` in `_hints`
    /// or the field is named in `_hints['immutable_fields']`.
    pub fn _immutable_field(&self, field: &str) -> bool {
        if matches!(self._hints.get("immutable"), Some(ConstValue::Bool(true))) {
            return true;
        }
        if let Some(ConstValue::Dict(fields)) = self._hints.get("immutable_fields")
            && fields.contains_key(&ConstValue::byte_str(field))
        {
            return true;
        }
        false
    }

    pub fn _container_example(&self) -> _struct {
        let fields: Vec<(String, LowLevelValue)> = self
            ._names
            .iter()
            .map(|name| {
                let typ = self
                    ._flds
                    .get(name)
                    .expect("StructType._names entry must exist in _flds");
                (name.clone(), typ._defl())
            })
            .collect();
        _struct {
            _identity: fresh_low_level_container_identity(),
            TYPE: self.clone(),
            _fields: Arc::new(Mutex::new(fields)),
        }
    }

    pub fn getattr_field_type(&self, field_name: &str) -> Option<ConcretetypePlaceholder> {
        self._flds.get(field_name).cloned()
    }

    pub fn _lookup_adtmeth(
        &self,
        member_name: &str,
        ll_ptrtype: &LowLevelPointerType,
    ) -> Result<LowLevelAdtMember, AttributeError> {
        let Some(adtmember) = self._adtmeths.get(member_name) else {
            return Err(AttributeError);
        };
        match adtmember {
            ConstValue::HostObject(_) => Ok(LowLevelAdtMember::Method {
                ll_ptrtype: ll_ptrtype.clone(),
                func: adtmember.clone(),
            }),
            _ => Ok(LowLevelAdtMember::Value(adtmember.clone())),
        }
    }

    pub fn _nofield(&self, name: &str) -> String {
        format!("struct {} has no field {:?}", self._name, name)
    }
}

impl ArrayType {
    pub fn new(of: ConcretetypePlaceholder) -> Self {
        Self::_build(of, GcKind::Raw, vec![])
    }

    pub fn gc(of: ConcretetypePlaceholder) -> Self {
        Self::_build(of, GcKind::Gc, vec![])
    }

    /// Raw `Array(OF, hints={...})`. Upstream
    /// `rpython/rtyper/lltypesystem/lltype.py:428-439 Array.__init__`
    /// + `_install_extras` forwards the `hints` kwarg to
    /// `self._hints`. Used by
    /// `SmallFunctionSetPBCRepr._setup_repr` (rpbc.py:416-418) which
    /// builds `Array(self.pointer_repr.lowleveltype,
    ///                hints={'nolength': True, 'immutable': True,
    ///                       'static_immutable': True})`.
    pub fn with_hints(of: ConcretetypePlaceholder, hints: Vec<(String, ConstValue)>) -> Self {
        Self::_build(of, GcKind::Raw, hints)
    }

    /// `GcArray(OF, hints={...})`. Same as [`ArrayType::gc`] plus a
    /// `hints` dict mirrored from upstream `_install_extras`.
    pub fn gc_with_hints(of: ConcretetypePlaceholder, hints: Vec<(String, ConstValue)>) -> Self {
        Self::_build(of, GcKind::Gc, hints)
    }

    /// Unified constructor mirroring `Array.__init__` / `_install_extras`
    /// (`lltype.py:428-439`). Rejects any non-raw container as the item
    /// type (`lltype.py:434-436`) — a gc array-of-gc-containers would
    /// double-manage lifetimes.
    fn _build(
        of: ConcretetypePlaceholder,
        gckind: GcKind,
        hints: Vec<(String, ConstValue)>,
    ) -> Self {
        if of.is_container_type() && of._gckind() != GcKind::Raw {
            panic!(
                "cannot have a {:?} container as array item type",
                of._gckind()
            );
        }
        let result = ArrayType {
            OF: of,
            _hints: FrozenDict::new(hints),
            _gckind: gckind,
        };
        let parent = LowLevelType::Array(Box::new(result.clone()));
        result
            .OF
            ._note_inlined_into(&parent, false, false)
            .unwrap_or_else(|err| panic!("{err}"));
        result
    }

    pub fn _container_example(&self) -> _array {
        _array {
            _identity: fresh_low_level_container_identity(),
            TYPE: ArrayContainer::Array(self.clone()),
            items: Arc::new(Mutex::new(vec![self.OF._defl()])),
        }
    }

    /// RPython `Array._is_atomic` (`lltype.py:450-451`) — an array is
    /// atomic iff its item type is atomic.
    pub fn _is_atomic(&self) -> bool {
        self.OF._is_atomic()
    }

    /// RPython `Array._short_name` (`lltype.py:475-480`). Includes
    /// `_gckind`-derived class prefix and the item short name.
    pub fn _short_name(&self) -> String {
        let kind = match self._gckind {
            GcKind::Gc => "GcArray",
            _ => "Array",
        };
        format!("{} {}", kind, self.OF.short_name())
    }

    /// RPython `Array._note_inlined_into` (`lltype.py:441-448`). Arrays
    /// only inline as the *last* field of a `Struct`; gc arrays cannot
    /// inline at all; no-length arrays cannot inline inside a GcStruct.
    pub fn _note_inlined_into(&self, parent: &LowLevelType, last: bool) -> Result<(), String> {
        let parent_is_struct = matches!(parent, LowLevelType::Struct(_));
        if !last || !parent_is_struct {
            return Err(
                "cannot inline an array in another container unless as the last field of a structure"
                    .to_string(),
            );
        }
        if self._gckind == GcKind::Gc {
            return Err("cannot inline a GC array inside a structure".to_string());
        }
        let parent_is_gc_struct = matches!(
            parent,
            LowLevelType::Struct(parent_t) if parent_t._gckind == GcKind::Gc
        );
        let has_nolength = matches!(self._hints.get("nolength"), Some(ConstValue::Bool(true)));
        if parent_is_gc_struct && has_nolength {
            return Err("cannot inline a no-length array inside a GcStruct".to_string());
        }
        Ok(())
    }

    /// RPython `Array._immutable_field` (`lltype.py:485-486`). Returns
    /// the `immutable` hint flag.
    pub fn _immutable_field(&self) -> bool {
        matches!(self._hints.get("immutable"), Some(ConstValue::Bool(true)))
    }
}

impl FixedSizeArrayType {
    pub fn new(of: ConcretetypePlaceholder, length: usize) -> Self {
        Self::_build(of, length, GcKind::Raw, vec![])
    }

    /// Unified constructor mirroring `FixedSizeArray.__init__`
    /// (`lltype.py:508-521`) — same item-type restrictions as `Array`
    /// apply (`lltype.py:518-520`).
    fn _build(
        of: ConcretetypePlaceholder,
        length: usize,
        gckind: GcKind,
        hints: Vec<(String, ConstValue)>,
    ) -> Self {
        if of.is_container_type() && of._gckind() != GcKind::Raw {
            panic!(
                "cannot have a {:?} container as array item type",
                of._gckind()
            );
        }
        let result = FixedSizeArrayType {
            OF: of,
            length,
            _hints: FrozenDict::new(hints),
            _gckind: gckind,
        };
        let parent = LowLevelType::FixedSizeArray(Box::new(result.clone()));
        result
            .OF
            ._note_inlined_into(&parent, false, false)
            .unwrap_or_else(|err| panic!("{err}"));
        result
    }

    pub fn _container_example(&self) -> _array {
        let items: Vec<LowLevelValue> = (0..self.length).map(|_| self.OF._defl()).collect();
        _array {
            _identity: fresh_low_level_container_identity(),
            TYPE: ArrayContainer::FixedSizeArray(self.clone()),
            items: Arc::new(Mutex::new(items)),
        }
    }

    /// RPython `FixedSizeArray._short_name` (`lltype.py:532-536`).
    pub fn _short_name(&self) -> String {
        format!("FixedSizeArray {} {}", self.length, self.OF.short_name())
    }

    /// RPython `FixedSizeArray._is_atomic` — inherited from Struct;
    /// because every `item%d` field shares `OF`, the test reduces to
    /// `OF._is_atomic()`.
    pub fn _is_atomic(&self) -> bool {
        self.OF._is_atomic()
    }
}

impl OpaqueType {
    pub fn new(tag: &str) -> Self {
        OpaqueType {
            tag: tag.into(),
            _gckind: GcKind::Raw,
        }
    }

    pub fn gc(tag: &str) -> Self {
        OpaqueType {
            tag: tag.into(),
            _gckind: GcKind::Gc,
        }
    }

    pub fn _container_example(&self) -> _opaque {
        _opaque {
            _identity: fresh_low_level_container_identity(),
            TYPE: self.clone(),
            _name: Some("?".to_string()),
            about: None,
            query_funcptr: None,
            destructor_funcptr: None,
        }
    }

    /// RPython `OpaqueType._note_inlined_into` (`lltype.py:592-596`).
    /// Raw opaque values may be inlined; gc opaque values may not.
    pub fn _note_inlined_into(&self, parent: &LowLevelType) -> Result<(), String> {
        if self._gckind == GcKind::Gc {
            return Err(format!(
                "{:?} cannot be inlined in {:?}",
                self._short_name(),
                parent.short_name()
            ));
        }
        Ok(())
    }

    pub fn _short_name(&self) -> String {
        match self._gckind {
            GcKind::Gc => format!("{} (gcopaque)", self.tag),
            _ => format!("{} (opaque)", self.tag),
        }
    }
}

/// RPython `RuntimeTypeInfo = OpaqueType("RuntimeTypeInfo")` (lltype.py:607).
///
/// Module-level singleton opaque type token used as the target of
/// `Ptr(RuntimeTypeInfo)` for GC-tracked instances. R3 helpers
/// (`attachRuntimeTypeInfo`, `getRuntimeTypeInfo`, `ClassRepr::fill_vtable_root`'s
/// `rtti` slot) consume this as the well-known opaque type.
pub static RUNTIME_TYPE_INFO: LazyLock<LowLevelType> =
    LazyLock::new(|| LowLevelType::Opaque(Box::new(OpaqueType::new("RuntimeTypeInfo"))));

fn new_opaque_container(TYPE: OpaqueType, name: &str, about: Option<LowLevelType>) -> _opaque {
    _opaque {
        _identity: fresh_low_level_container_identity(),
        TYPE,
        _name: Some(name.to_string()),
        about,
        query_funcptr: None,
        destructor_funcptr: None,
    }
}

fn expect_rtti_struct(T: &LowLevelType) -> Result<&StructType, String> {
    match T {
        LowLevelType::Struct(struct_t) if struct_t._gckind == GcKind::Gc => Ok(struct_t.as_ref()),
        _ => Err(format!("expected a RttiStruct: {}", T.short_name())),
    }
}

fn expect_rtti_struct_mut(T: &mut LowLevelType) -> Result<&mut StructType, String> {
    let short_name = T.short_name();
    match T {
        LowLevelType::Struct(struct_t) if struct_t._gckind == GcKind::Gc => Ok(struct_t.as_mut()),
        _ => Err(format!("expected a RttiStruct: {}", short_name)),
    }
}

fn attach_runtime_type_info_missing_error(struct_t: &StructType) -> String {
    format!(
        "attachRuntimeTypeInfo: {} must have been built with the rtti=True argument",
        struct_t._short_name()
    )
}

fn castdepth(outside: &StructType, inside: &StructType) -> i32 {
    if outside == inside {
        return 0;
    }
    let mut dwn = 0;
    // Walk an owned chain so we can resolve `ForwardReference` parents
    // produced by `_setup_repr` for the inheritance hierarchy.
    let mut current_owned = outside.clone();
    loop {
        let Some((_, first_type)) = current_owned._first_struct_owned() else {
            break;
        };
        dwn += 1;
        if &first_type == inside {
            return dwn;
        }
        current_owned = first_type;
    }
    -1
}

/// RPython `castable(PTRTYPE, CURTYPE)` (`lltype.py:944-961`). Returns
/// the signed depth distance between two `Ptr` types: positive when
/// `PTRTYPE` is reached by walking inward (through `_first_struct`),
/// negative when reached outward (parent), zero when identical. Errors
/// when the types are not castable (gc-status mismatch / non-Struct
/// targets / no chain).
pub fn castable(ptrtype: &Ptr, curtype: &Ptr) -> Result<i32, String> {
    castable_ptr_types(ptrtype, curtype)
}

fn castable_ptr_types(ptrtype: &Ptr, curtype: &Ptr) -> Result<i32, String> {
    if curtype._gckind() != ptrtype._gckind() {
        return Err(format!(
            "cast_pointer() cannot change the gc status: {:?} to {:?}",
            curtype, ptrtype
        ));
    }
    if curtype == ptrtype {
        return Ok(0);
    }
    let (PtrTarget::Struct(curstruc), PtrTarget::Struct(ptrstruc)) = (&curtype.TO, &ptrtype.TO)
    else {
        return Err(format!(
            "invalid cast between {:?} and {:?}",
            curtype, ptrtype
        ));
    };
    let d = castdepth(curstruc, ptrstruc);
    if d >= 0 {
        return Ok(d);
    }
    let u = castdepth(ptrstruc, curstruc);
    if u == -1 {
        return Err(format!(
            "invalid cast between {:?} and {:?}",
            curtype, ptrtype
        ));
    }
    Ok(-u)
}

fn validate_rtti_helper_ptr(
    funcptr: &_ptr,
    gcstruct: &StructType,
    result_type: &LowLevelType,
    error_label: &str,
) -> Result<(), String> {
    let T = typeOf(funcptr);
    let PtrTarget::Func(func_t) = &T.TO else {
        return Err(format!(
            "expected a {} function implementation, got: {:?}",
            error_label, funcptr
        ));
    };
    let expected_self_ptr =
        Ptr::from_container_type(LowLevelType::Struct(Box::new(gcstruct.clone())))
            .expect("Ptr(GcStruct) must be constructible for RTTI helper validation");
    let arg_ok = func_t.args.len() == 1
        && matches!(
            func_t.args.first(),
            Some(LowLevelType::Ptr(arg_ptr))
                if matches!(castable_ptr_types(arg_ptr, &expected_self_ptr), Ok(depth) if depth >= 0)
        );
    if !(arg_ok && func_t.result == *result_type) {
        return Err(format!(
            "expected a {} function implementation, got: {:?}",
            error_label, funcptr
        ));
    }
    Ok(())
}

impl ForwardReference {
    pub fn new() -> Self {
        ForwardReference {
            _gckind: GcKind::Raw,
            target: Arc::new(Mutex::new(None)),
        }
    }

    pub fn gc() -> Self {
        ForwardReference {
            _gckind: GcKind::Gc,
            target: Arc::new(Mutex::new(None)),
        }
    }

    pub fn prebuilt() -> Self {
        ForwardReference {
            _gckind: GcKind::Prebuilt,
            target: Arc::new(Mutex::new(None)),
        }
    }

    pub fn r#become(&self, realcontainertype: LowLevelType) -> Result<(), String> {
        if !realcontainertype.is_container_type() {
            return Err(format!(
                "ForwardReference can only be to a container, not {realcontainertype:?}"
            ));
        }
        if realcontainertype._gckind() != self._gckind {
            return Err(format!(
                "become() gives conflicting gckind, use the correct XxForwardReference"
            ));
        }
        *self.target.lock().unwrap() = Some(realcontainertype);
        Ok(())
    }

    pub fn resolved(&self) -> Option<LowLevelType> {
        self.target.lock().unwrap().clone()
    }
}

impl PartialEq for ForwardReference {
    fn eq(&self, other: &Self) -> bool {
        // Identity short-circuit: clones of the same `ForwardReference`
        // share the `Arc<Mutex<_>>` target, so pointer equality on the
        // Arc proves equivalence without recursing into the resolved
        // type — required to break cycles like `OBJECT_VTABLE →
        // OBJECTPTR → OBJECT → CLASSTYPE → OBJECT_VTABLE` (the
        // `instantiate` funcptr field closes the cycle and structural
        // recursion through `resolved == resolved` would not terminate).
        if Arc::ptr_eq(&self.target, &other.target) {
            return true;
        }
        match (self.resolved(), other.resolved()) {
            (Some(left), Some(right)) => {
                let id = Arc::as_ptr(&self.target) as *const _ as usize;
                if FORWARD_REF_EQ_GUARD.with(|g| g.borrow().contains(&id)) {
                    // Already comparing this fwd elsewhere on the
                    // stack — short-circuit to `true` per RPython
                    // `saferecursive(safe_equal, True)` (lltype.py:74).
                    return true;
                }
                FORWARD_REF_EQ_GUARD.with(|g| g.borrow_mut().insert(id));
                let r = left == right;
                FORWARD_REF_EQ_GUARD.with(|g| g.borrow_mut().remove(&id));
                r
            }
            (Some(_), None) | (None, Some(_)) => false,
            (None, None) => self._gckind == other._gckind,
        }
    }
}

impl Eq for ForwardReference {}

thread_local! {
    /// TLS-based saferecursive guard. Tracks the set of
    /// `ForwardReference` Arc pointers currently being compared / hashed
    /// on this thread's stack, so re-entry on the same `Arc` short-
    /// circuits to identity comparison / identity hashing instead of
    /// recursing forever through a cyclic type graph.
    static FORWARD_REF_EQ_GUARD: std::cell::RefCell<std::collections::HashSet<usize>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
    static FORWARD_REF_HASH_GUARD: std::cell::RefCell<std::collections::HashSet<usize>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

impl Hash for ForwardReference {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let id = Arc::as_ptr(&self.target) as *const _ as usize;
        let already = FORWARD_REF_HASH_GUARD.with(|g| g.borrow().contains(&id));
        if already {
            // Re-entry on the same fwd — hash a constant 0 per
            // RPython `saferecursive(get_hash, 0)` (lltype.py:136).
            // Identity hashing here would make hash values depend
            // on Arc address, breaking deterministic equality
            // contract for structurally-equal cyclic types.
            0_u8.hash(state);
            return;
        }
        match self.resolved() {
            Some(real) => {
                FORWARD_REF_HASH_GUARD.with(|g| g.borrow_mut().insert(id));
                real.hash(state);
                FORWARD_REF_HASH_GUARD.with(|g| g.borrow_mut().remove(&id));
            }
            None => panic!("ForwardReference object is not hashable"),
        }
    }
}

impl LowLevelType {
    pub fn is_container_type(&self) -> bool {
        matches!(
            self,
            LowLevelType::Func(_)
                | LowLevelType::Struct(_)
                | LowLevelType::Array(_)
                | LowLevelType::FixedSizeArray(_)
                | LowLevelType::Opaque(_)
                | LowLevelType::ForwardReference(_)
        )
    }

    /// RPython `LowLevelType._is_varsize` (`lltype.py:191-192`). Default
    /// is `False`; only `Array` and `Struct` with a trailing varsize
    /// field report `True`.
    pub fn _is_varsize(&self) -> bool {
        match self {
            LowLevelType::Array(_) => true,
            LowLevelType::Struct(t) => t._is_varsize(),
            LowLevelType::ForwardReference(t) => t.resolved().is_some_and(|t| t._is_varsize()),
            _ => false,
        }
    }

    /// RPython `LowLevelType._is_atomic` (`lltype.py:188-189`). Default
    /// is `False`; primitives report `True` and `Struct` is atomic iff
    /// all fields are atomic.
    pub fn _is_atomic(&self) -> bool {
        match self {
            LowLevelType::Void
            | LowLevelType::Bool
            | LowLevelType::Signed
            | LowLevelType::Unsigned
            | LowLevelType::SignedLongLong
            | LowLevelType::SignedLongLongLong
            | LowLevelType::UnsignedLongLong
            | LowLevelType::UnsignedLongLongLong
            | LowLevelType::Float
            | LowLevelType::SingleFloat
            | LowLevelType::LongFloat
            | LowLevelType::Char
            | LowLevelType::UniChar
            | LowLevelType::Address => true,
            LowLevelType::Struct(t) => t._is_atomic(),
            _ => false,
        }
    }

    /// RPython `LowLevelType._note_inlined_into` / container overrides
    /// (`lltype.py:185-206, 305-312, 441-448, 592-596`).
    pub fn _note_inlined_into(
        &self,
        parent: &LowLevelType,
        first: bool,
        last: bool,
    ) -> Result<(), String> {
        match self {
            LowLevelType::Void
            | LowLevelType::Signed
            | LowLevelType::Unsigned
            | LowLevelType::SignedLongLong
            | LowLevelType::SignedLongLongLong
            | LowLevelType::UnsignedLongLong
            | LowLevelType::UnsignedLongLongLong
            | LowLevelType::Bool
            | LowLevelType::Float
            | LowLevelType::SingleFloat
            | LowLevelType::LongFloat
            | LowLevelType::Char
            | LowLevelType::UniChar
            | LowLevelType::Address
            | LowLevelType::Ptr(_)
            | LowLevelType::InteriorPtr(_) => Ok(()),
            LowLevelType::Struct(t) => t._note_inlined_into(parent, first),
            LowLevelType::FixedSizeArray(t) => {
                if t._gckind == GcKind::Gc {
                    let parent_is_gc_struct = matches!(
                        parent,
                        LowLevelType::Struct(parent_t) if parent_t._gckind == GcKind::Gc
                    );
                    if !first || !parent_is_gc_struct {
                        return Err(
                            "a GcStruct can only be inlined as the first field of another GcStruct"
                                .to_string(),
                        );
                    }
                }
                Ok(())
            }
            LowLevelType::Array(t) => t._note_inlined_into(parent, last),
            LowLevelType::Opaque(t) => t._note_inlined_into(parent),
            LowLevelType::Func(_) => Err(format!(
                "FuncType cannot be inlined in {:?}",
                parent.short_name()
            )),
            LowLevelType::ForwardReference(t) => match t.resolved() {
                Some(real) => real._note_inlined_into(parent, first, last),
                None => Err(format!(
                    "ForwardReference cannot be inlined in {:?}",
                    parent.short_name()
                )),
            },
        }
    }
}

impl Ptr {
    pub fn _defl(&self) -> _ptr {
        _ptr::new(self.clone(), Ok(None))
    }

    /// RPython `Ptr.__new__(cls, TO)` validation + construction
    /// (`lltype.py:725-739`). Takes a container-kind `LowLevelType`
    /// variant and packs it into the matching `PtrTarget` arm. The
    /// upstream `WeakValueDictionary` cache is omitted — Rust
    /// `PartialEq` derives structural equality that covers the identity
    /// role of the cache.
    pub fn from_container_type(T: LowLevelType) -> Result<Self, String> {
        let TO = match T {
            LowLevelType::Func(t) => PtrTarget::Func(*t),
            LowLevelType::Struct(t) => PtrTarget::Struct(*t),
            LowLevelType::Array(t) => PtrTarget::Array(*t),
            LowLevelType::FixedSizeArray(t) => PtrTarget::FixedSizeArray(*t),
            LowLevelType::Opaque(t) => PtrTarget::Opaque(*t),
            LowLevelType::ForwardReference(t) => PtrTarget::ForwardReference(*t),
            other => {
                return Err(format!(
                    "can only point to a Container type, not to {other:?}"
                ));
            }
        };
        Ok(Ptr { TO })
    }

    /// Short-name of the pointer's target container type. Called by
    /// `LowLevelType::short_name` for `"Ptr %s"` formatting (upstream
    /// `lltype.py:748`).
    pub fn _to_short_name(&self) -> String {
        match &self.TO {
            PtrTarget::Func(t) => t._short_name(),
            PtrTarget::Struct(t) => t._short_name(),
            PtrTarget::Array(t) => t._short_name(),
            PtrTarget::FixedSizeArray(t) => t._short_name(),
            PtrTarget::Opaque(t) => t.tag.clone(),
            PtrTarget::ForwardReference(t) => t
                .resolved()
                .map_or_else(|| "ForwardReference".to_string(), |real| real.short_name()),
        }
    }

    pub fn _example(&self) -> _ptr {
        _ptr::new_with_solid(
            self.clone(),
            Ok(Some(match &self.TO {
                PtrTarget::Func(func_t) => _ptr_obj::Func(func_t._container_example()),
                PtrTarget::Struct(struct_t) => _ptr_obj::Struct(struct_t._container_example()),
                PtrTarget::Array(array_t) => _ptr_obj::Array(array_t._container_example()),
                PtrTarget::FixedSizeArray(array_t) => _ptr_obj::Array(array_t._container_example()),
                PtrTarget::Opaque(opaque_t) => _ptr_obj::Opaque(opaque_t._container_example()),
                PtrTarget::ForwardReference(forward_ref) => {
                    match forward_ref
                        .resolved()
                        .expect("ForwardReference must be resolved before _example()")
                    {
                        LowLevelType::Func(func_t) => _ptr_obj::Func(func_t._container_example()),
                        LowLevelType::Struct(struct_t) => {
                            _ptr_obj::Struct(struct_t._container_example())
                        }
                        LowLevelType::Array(array_t) => {
                            _ptr_obj::Array(array_t._container_example())
                        }
                        LowLevelType::FixedSizeArray(array_t) => {
                            _ptr_obj::Array(array_t._container_example())
                        }
                        LowLevelType::Opaque(opaque_t) => {
                            _ptr_obj::Opaque(opaque_t._container_example())
                        }
                        LowLevelType::ForwardReference(_) => {
                            panic!("ForwardReference target must resolve to a concrete container")
                        }
                        LowLevelType::Ptr(_)
                        | LowLevelType::InteriorPtr(_)
                        | LowLevelType::Void
                        | LowLevelType::Signed
                        | LowLevelType::Unsigned
                        | LowLevelType::SignedLongLong
                        | LowLevelType::SignedLongLongLong
                        | LowLevelType::UnsignedLongLong
                        | LowLevelType::UnsignedLongLongLong
                        | LowLevelType::Bool
                        | LowLevelType::Float
                        | LowLevelType::SingleFloat
                        | LowLevelType::LongFloat
                        | LowLevelType::Char
                        | LowLevelType::UniChar
                        | LowLevelType::Address => {
                            panic!("ForwardReference target must resolve to a container type")
                        }
                    }
                }
            })),
            true,
        )
    }

    pub fn _nofield(&self, name: &str) -> String {
        match &self.TO {
            PtrTarget::Struct(struct_t) => struct_t._nofield(name),
            PtrTarget::Func(_)
            | PtrTarget::Array(_)
            | PtrTarget::FixedSizeArray(_)
            | PtrTarget::Opaque(_)
            | PtrTarget::ForwardReference(_) => {
                format!("{:?} instance has no field {:?}", self, name)
            }
        }
    }

    pub fn _gckind(&self) -> GcKind {
        self.TO._gckind()
    }

    pub fn _needsgc(&self) -> bool {
        !matches!(self.TO._gckind(), GcKind::Raw | GcKind::Prebuilt)
    }

    /// RPython `Ptr._interior_ptr_type_with_index(TO)` (`lltype.py:769-778`).
    /// Builds the GcStruct that represents an interior-pointer-with-index
    /// into a gc array. Invariants:
    /// * `self.TO._gckind` must be `gc` (upstream asserts this);
    /// * the resulting struct is a GcStruct, not a raw Struct;
    /// * the `interior_ptr_type` hint flags the struct as synthetic;
    /// * when `TO` is a Struct, its `_adtmeths` are copied.
    pub fn _interior_ptr_type_with_index(&self, to: &LowLevelType) -> StructType {
        assert_eq!(
            self.TO._gckind(),
            GcKind::Gc,
            "interior_ptr_type_with_index requires gc parent, got {:?}",
            self.TO._gckind()
        );
        let adtmeths = match to {
            LowLevelType::Struct(struct_t) => struct_t._adtmeths.to_vec(),
            _ => vec![],
        };
        let hints = vec![("interior_ptr_type".into(), ConstValue::Bool(true))];
        StructType::_build(
            "Interior",
            vec![
                ("ptr".into(), LowLevelType::Ptr(Box::new(self.clone()))),
                ("index".into(), LowLevelType::Signed),
            ],
            GcKind::Gc,
            adtmeths,
            hints,
        )
    }
}

/// RPython `nullptr(T)` (`lltype.py:2347-2348`):
///
/// ```python
/// def nullptr(T):
///     return Ptr(T)._defl()
/// ```
///
/// Returns a null pointer value whose `Ptr` type wraps the given container
/// `T`. Upstream uses `_ptr(self, None)` directly in `Ptr._defl`; the Rust
/// port follows the same two-step `Ptr::from_container_type(T)?._defl()`.
pub fn nullptr(T: LowLevelType) -> Result<_ptr, String> {
    Ok(Ptr::from_container_type(T)?._defl())
}

/// RPython `cast_pointer(PTRTYPE, ptr)` (`lltype.py:964-968`):
///
/// ```python
/// def cast_pointer(PTRTYPE, ptr):
///     CURTYPE = typeOf(ptr)
///     if not isinstance(CURTYPE, Ptr) or not isinstance(PTRTYPE, Ptr):
///         raise TypeError("can only cast pointers to other pointers")
///     return ptr._cast_to(PTRTYPE)
/// ```
///
/// Ports the entry point used by `ll_cast_to_object` (`rclass.py:1126-1127`)
/// and other repr-side static-cast call sites. The actual chain walk
/// lives on [`_ptr::_cast_to`].
pub fn cast_pointer(ptrtype: &Ptr, ptr: &_ptr) -> Result<_ptr, String> {
    ptr._cast_to(ptrtype)
}

/// RPython `getRuntimeTypeInfo(GCSTRUCT)` (`lltype.py:2391-2397`):
///
/// ```python
/// def getRuntimeTypeInfo(GCSTRUCT):
///     if not isinstance(GCSTRUCT, RttiStruct):
///         raise TypeError(...)
///     if GCSTRUCT._runtime_type_info is None:
///         raise ValueError("no attached runtime type info for GcStruct %s" % GCSTRUCT._name)
///     return _ptr(Ptr(RuntimeTypeInfo), GCSTRUCT._runtime_type_info)
/// ```
///
/// Returns a `Ptr(RuntimeTypeInfo)` pointing at the struct's pre-attached
/// opaque. Errors if the struct was built without `gc_rtti` (no opaque)
/// or if a non-Struct `LowLevelType` is passed. Both conditions match
/// upstream's `TypeError` / `ValueError`.
pub fn getRuntimeTypeInfo(T: &LowLevelType) -> Result<_ptr, String> {
    let struct_t = expect_rtti_struct(T)?;
    let Some(rtti_opaque) = &struct_t._runtime_type_info else {
        return Err(format!(
            "no attached runtime type info for GcStruct {}",
            struct_t._name
        ));
    };
    let ptr_t = Ptr::from_container_type(RUNTIME_TYPE_INFO.clone())?;
    Ok(_ptr::new(
        ptr_t,
        Ok(Some(_ptr_obj::Opaque((**rtti_opaque).clone()))),
    ))
}

/// RPython `attachRuntimeTypeInfo(GCSTRUCT, funcptr=None, destrptr=None)`
/// (`lltype.py:2385-2389`):
///
/// ```python
/// def attachRuntimeTypeInfo(GCSTRUCT, funcptr=None, destrptr=None):
///     if not isinstance(GCSTRUCT, RttiStruct):
///         raise TypeError(...)
///     GCSTRUCT._attach_runtime_type_info_funcptr(funcptr, destrptr)
///     return _ptr(Ptr(RuntimeTypeInfo), GCSTRUCT._runtime_type_info)
/// ```
///
/// Rust keeps the no-extra-args wrapper for existing call sites and
/// exposes the full mutable port in
/// [`attachRuntimeTypeInfo_with_ptrs`], which stores helper pointers on
/// the `_runtime_type_info` opaque just like upstream.
pub fn attachRuntimeTypeInfo(T: &LowLevelType) -> Result<_ptr, String> {
    let struct_t = expect_rtti_struct(T)?;
    let Some(rtti_opaque) = &struct_t._runtime_type_info else {
        return Err(attach_runtime_type_info_missing_error(struct_t));
    };
    let ptr_t = Ptr::from_container_type(RUNTIME_TYPE_INFO.clone())?;
    Ok(_ptr::new(
        ptr_t,
        Ok(Some(_ptr_obj::Opaque((**rtti_opaque).clone()))),
    ))
}

pub fn attachRuntimeTypeInfo_with_ptrs(
    T: &mut LowLevelType,
    funcptr: Option<_ptr>,
    destrptr: Option<_ptr>,
) -> Result<_ptr, String> {
    let struct_t = expect_rtti_struct_mut(T)?;
    let struct_snapshot = struct_t.clone();
    if struct_t._runtime_type_info.is_none() {
        return Err(attach_runtime_type_info_missing_error(&struct_snapshot));
    }
    let runtime_type_info_ptr = LowLevelType::Ptr(Box::new(Ptr::from_container_type(
        RUNTIME_TYPE_INFO.clone(),
    )?));
    if let Some(funcptr) = funcptr.as_ref() {
        validate_rtti_helper_ptr(
            funcptr,
            &struct_snapshot,
            &runtime_type_info_ptr,
            "runtime type info",
        )?;
    }
    if let Some(destrptr) = destrptr.as_ref() {
        validate_rtti_helper_ptr(
            destrptr,
            &struct_snapshot,
            &LowLevelType::Void,
            "destructor",
        )?;
    }
    let rtti_opaque = struct_t._runtime_type_info.as_mut().expect("checked above");
    if let Some(funcptr) = funcptr {
        rtti_opaque.query_funcptr = Some(Box::new(funcptr));
    }
    if let Some(destrptr) = destrptr {
        rtti_opaque.destructor_funcptr = Some(Box::new(destrptr));
    }
    let ptr_t = Ptr::from_container_type(RUNTIME_TYPE_INFO.clone())?;
    Ok(_ptr::new(
        ptr_t,
        Ok(Some(_ptr_obj::Opaque((**rtti_opaque).clone()))),
    ))
}

/// RPython `opaqueptr(TYPE, name, **attrs)` (`lltype.py:2357-2361`):
///
/// ```python
/// def opaqueptr(TYPE, name, **attrs):
///     if not isinstance(TYPE, OpaqueType):
///         raise TypeError("opaqueptr() for OpaqueTypes only")
///     o = _opaque(TYPE, _name=name, **attrs)
///     return _ptr(Ptr(TYPE), o, solid=True)
/// ```
///
/// Mints a pointer to a freshly-constructed `_opaque` container stamped
/// with the given human-readable `name`. Used by
/// `RttiStruct._install_extras(rtti=True)` to register the struct's
/// `_runtime_type_info` opaque (`lltype.py:385-389`). The helper keeps
/// the public two-arg surface for existing Rust callers and routes the
/// actual construction through a private attrs-aware helper.
pub fn opaqueptr(TYPE: LowLevelType, name: &str) -> Result<_ptr, String> {
    opaqueptr_with_attrs(TYPE, name, None)
}

fn opaqueptr_with_attrs(
    TYPE: LowLevelType,
    name: &str,
    about: Option<LowLevelType>,
) -> Result<_ptr, String> {
    let LowLevelType::Opaque(opaque_t) = &TYPE else {
        return Err(format!("opaqueptr() for OpaqueTypes only, got {TYPE:?}"));
    };
    let obj = new_opaque_container((**opaque_t).clone(), name, about);
    let ptr_t = Ptr::from_container_type(TYPE)?;
    Ok(_ptr::new_with_solid(
        ptr_t,
        Ok(Some(_ptr_obj::Opaque(obj))),
        true,
    ))
}

/// Allocation flavor for `malloc(T, ..., flavor=...)`, mirroring upstream
/// `lltype.py:2192-2216` string kwarg (`'gc'` | `'raw'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MallocFlavor {
    Gc,
    Raw,
}

/// RPython `lltype.malloc(T, n=None, flavor='gc', immortal=False, ...)`
/// (`lltype.py:2192-2216`).
///
/// ```python
/// def malloc(T, n=None, flavor='gc', immortal=False, zero=False, ...):
///     ...
///     if isinstance(T, Struct):
///         o = _struct(T, n, initialization=initialization)
///     elif isinstance(T, Array):
///         o = _array(T, n, initialization=initialization)
///     elif isinstance(T, OpaqueType):
///         assert n is None
///         o = _opaque(T, initialization=initialization)
///     ...
///     return _ptr(Ptr(T), o, solid)
/// ```
///
/// This port covers the subset R3/RString needs: `flavor='gc'` or
/// `'raw'` × `immortal=True/False` × Struct / Array / OpaqueType,
/// including varsize Structs whose trailing field is an `Array`.
/// `zero`, `track_allocation`, `add_memory_pressure`, and
/// `nonmovable` upstream kwargs are still not accepted. The `solid`
/// kwarg threaded through upstream `_ptr(Ptr(T), o, solid)` is
/// tracked on `_ptr._solid` (upstream lltype.py:2221 `immortal or
/// flavor == 'raw'`).
pub fn malloc(
    T: LowLevelType,
    n: Option<usize>,
    flavor: MallocFlavor,
    immortal: bool,
) -> Result<_ptr, String> {
    let obj: _ptr_obj = match &T {
        LowLevelType::Struct(struct_t) => {
            if struct_t._arrayfld.is_some() {
                let n = n.ok_or_else(|| {
                    format!("malloc: varsize struct {:?} requires n", struct_t._name)
                })?;
                let fields = struct_t
                    ._names
                    .iter()
                    .map(|name| {
                        let typ = struct_t
                            ._flds
                            .get(name)
                            .expect("StructType._names entry must exist in _flds");
                        if Some(name) == struct_t._arrayfld.as_ref() {
                            let LowLevelType::Array(array_t) = typ else {
                                panic!("StructType._arrayfld must name an Array field");
                            };
                            let inner_items: Vec<LowLevelValue> =
                                (0..n).map(|_| array_t.OF._defl()).collect();
                            let arr = _array {
                                _identity: fresh_low_level_container_identity(),
                                TYPE: ArrayContainer::Array((**array_t).clone()),
                                items: Arc::new(Mutex::new(inner_items)),
                            };
                            (name.clone(), LowLevelValue::Array(Box::new(arr)))
                        } else {
                            (name.clone(), typ._defl())
                        }
                    })
                    .collect();
                _ptr_obj::Struct(_struct {
                    _identity: fresh_low_level_container_identity(),
                    TYPE: (**struct_t).clone(),
                    _fields: Arc::new(Mutex::new(fields)),
                })
            } else {
                if n.is_some() {
                    return Err(format!(
                        "malloc: non-varsize struct {:?} got n={n:?}",
                        struct_t._name
                    ));
                }
                _ptr_obj::Struct(struct_t._container_example())
            }
        }
        LowLevelType::Array(array_t) => {
            let items: Vec<LowLevelValue> = match n {
                Some(n) => (0..n).map(|_| array_t.OF._defl()).collect(),
                None => array_t._container_example().items.lock().unwrap().clone(),
            };
            _ptr_obj::Array(_array {
                _identity: fresh_low_level_container_identity(),
                TYPE: ArrayContainer::Array((**array_t).clone()),
                items: Arc::new(Mutex::new(items)),
            })
        }
        LowLevelType::Opaque(opaque_t) => {
            if n.is_some() {
                return Err("malloc: OpaqueType does not accept n".to_string());
            }
            _ptr_obj::Opaque(opaque_t._container_example())
        }
        other => {
            return Err(format!("malloc: unmallocable type {other:?}"));
        }
    };
    // upstream lltype.py:2211-2212 — gc flavor on non-gc, non-immortal
    // struct/array/opaque is rejected. We only enforce it when !immortal
    // because the init_vtable call site is immortal and can target
    // either a gc or non-gc prototype container.
    if flavor == MallocFlavor::Gc && !immortal && T._gckind() != GcKind::Gc {
        return Err(format!(
            "gc flavor malloc of a non-GC non-immortal structure {:?}",
            T.short_name()
        ));
    }
    let solid = immortal || flavor == MallocFlavor::Raw;
    let ptr_t = Ptr::from_container_type(T)?;
    Ok(_ptr::new_with_solid(ptr_t, Ok(Some(obj)), solid))
}

impl InteriorPtr {
    pub fn _example(&self) -> _interior_ptr {
        let parent = match &*self.PARENTTYPE {
            LowLevelType::Struct(struct_t) => {
                LowLevelValue::Struct(Box::new(struct_t._container_example()))
            }
            LowLevelType::Array(array_t) => {
                LowLevelValue::Array(Box::new(array_t._container_example()))
            }
            LowLevelType::FixedSizeArray(array_t) => {
                LowLevelValue::Array(Box::new(array_t._container_example()))
            }
            LowLevelType::Opaque(opaque_t) => {
                LowLevelValue::Opaque(Box::new(opaque_t._container_example()))
            }
            other => panic!("InteriorPtr parent must be container, got {other:?}"),
        };
        _interior_ptr {
            _T: (*self.TO).clone(),
            _parent: parent,
            _offsets: self.offsets.clone(),
        }
    }
}

pub fn functionptr(
    TYPE: FuncType,
    name: &str,
    graph: Option<usize>,
    _callable: Option<String>,
) -> _ptr {
    functionptr_with_attrs(TYPE, name, graph, _callable, HashMap::new())
}

fn functionptr_with_attrs(
    TYPE: FuncType,
    name: &str,
    graph: Option<usize>,
    _callable: Option<String>,
    attrs: HashMap<String, ConstValue>,
) -> _ptr {
    _ptr::new(
        Ptr {
            TO: PtrTarget::Func(TYPE.clone()),
        },
        Ok(Some(_ptr_obj::Func(_func::new(
            TYPE,
            name.to_string(),
            graph,
            _callable,
            attrs,
        )))),
    )
}

pub fn build_number(_name: Option<()>, knowntype: KnownType) -> LowLevelType {
    match knowntype {
        KnownType::Int => LowLevelType::Signed,
        KnownType::Ruint => LowLevelType::Unsigned,
        KnownType::LongLong => LowLevelType::SignedLongLong,
        KnownType::LongLongLong => LowLevelType::SignedLongLongLong,
        KnownType::ULongLong => LowLevelType::UnsignedLongLong,
        KnownType::ULongLongLong => LowLevelType::UnsignedLongLongLong,
        other => panic!("lltype.build_number() does not support knowntype {other}"),
    }
}

impl LowLevelType {
    pub fn _gckind(&self) -> GcKind {
        match self {
            LowLevelType::Func(_) => GcKind::Raw,
            LowLevelType::Struct(t) => t._gckind,
            LowLevelType::Array(t) => t._gckind,
            LowLevelType::FixedSizeArray(t) => t._gckind,
            LowLevelType::Opaque(t) => t._gckind,
            LowLevelType::ForwardReference(t) => t._gckind,
            LowLevelType::Ptr(t) => t._gckind(),
            LowLevelType::InteriorPtr(t) => t.TO._gckind(),
            other => panic!("{other:?} is not a container type"),
        }
    }
}

impl PtrTarget {
    pub fn _gckind(&self) -> GcKind {
        match self {
            PtrTarget::Func(_) => GcKind::Raw,
            PtrTarget::Struct(t) => t._gckind,
            PtrTarget::Array(t) => t._gckind,
            PtrTarget::FixedSizeArray(t) => t._gckind,
            PtrTarget::Opaque(t) => t._gckind,
            PtrTarget::ForwardReference(t) => t._gckind,
        }
    }
}

pub fn _getconcretetype(v: &Hlvalue) -> Result<ConcretetypePlaceholder, TyperError> {
    // RPython `rtyper.py:570-571 getcallable.getconcretetype` does not
    // invent a fallback type: the callable path obtains
    // `self.bindingrepr(v).lowleveltype`. This helper mirrors the raw
    // `v.concretetype` lookup used by lltype-level tests and fails when
    // the rtyper has not assigned a concrete type yet.
    let concretetype = match v {
        Hlvalue::Variable(v) => v.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    };
    concretetype.ok_or_else(|| TyperError::message("missing concretetype for getfunctionptr"))
}

pub fn getfunctionptr<F>(graph: &GraphRef, getconcretetype: F) -> Result<_ptr, TyperError>
where
    F: Fn(&Hlvalue) -> Result<ConcretetypePlaceholder, TyperError>,
{
    let graph_b = graph.borrow();
    let mut llinputs = Vec::new();
    for arg in graph_b.getargs() {
        llinputs.push(getconcretetype(&arg)?);
    }
    let lloutput = getconcretetype(&graph_b.getreturnvar())?;
    let ft = FuncType {
        args: llinputs,
        result: lloutput,
    };
    let mut name = graph_b.name.clone();
    let mut callable = graph_b.func.as_ref().map(|func| func.name.clone());
    let mut attrs = HashMap::new();
    if let Some(func) = &graph_b.func {
        attrs = func._llfnobjattrs_.clone();
        if let Some(forced_name) = attrs.remove("_name").and_then(ConstValue::into_text) {
            name = forced_name;
        }
        if let Some(forced_callable) = attrs.remove("_callable") {
            callable = Some(match forced_callable {
                ConstValue::ByteStr(name) => {
                    String::from_utf8(name).unwrap_or_else(|err| format!("{:?}", err.into_bytes()))
                }
                ConstValue::UniStr(name) => name,
                ConstValue::Function(func) => func.name,
                ConstValue::HostObject(obj) => obj.qualname().to_string(),
                other => format!("{other:?}"),
            });
        }
    }
    drop(graph_b);
    Ok(functionptr_with_attrs(
        ft,
        &name,
        Some(GraphKey::of(graph).as_usize()),
        callable,
        attrs,
    ))
}

/// RPython `class SomePtr(SomeObject)` (lltype.py:1520-1528).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomePtr {
    pub base: crate::annotator::model::SomeObjectBase,
    pub ll_ptrtype: Ptr,
}

impl SomePtr {
    pub fn new(ll_ptrtype: Ptr) -> Self {
        SomePtr {
            base: crate::annotator::model::SomeObjectBase::new(KnownType::LlPtr, true),
            ll_ptrtype,
        }
    }
}

impl crate::annotator::model::SomeObjectTrait for SomePtr {
    fn knowntype(&self) -> KnownType {
        KnownType::LlPtr
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{Block, Constant, FunctionGraph, GraphFunc, Variable};
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn runtime_type_info_singleton_is_raw_opaque_with_upstream_tag() {
        let LowLevelType::Opaque(inner) = &*RUNTIME_TYPE_INFO else {
            panic!("RUNTIME_TYPE_INFO must resolve to LowLevelType::Opaque");
        };
        assert_eq!(inner.tag, "RuntimeTypeInfo");
        assert_eq!(inner._gckind, GcKind::Raw);
    }

    #[test]
    fn runtime_type_info_is_stable_across_lookups() {
        let first: *const LowLevelType = &*RUNTIME_TYPE_INFO;
        let second: *const LowLevelType = &*RUNTIME_TYPE_INFO;
        assert_eq!(first, second);
    }

    #[test]
    fn ptr_from_container_type_packs_struct_into_ptr_target_struct() {
        let s = StructType::_build(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
            GcKind::Gc,
            vec![],
            vec![],
        );
        let T = LowLevelType::Struct(Box::new(s));
        let p = Ptr::from_container_type(T).unwrap();
        assert!(matches!(p.TO, PtrTarget::Struct(_)));
    }

    #[test]
    fn ptr_from_container_type_rejects_primitive_types() {
        let err = Ptr::from_container_type(LowLevelType::Signed).unwrap_err();
        assert!(
            err.contains("can only point to a Container type"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn nullptr_of_runtime_type_info_is_null_ptr_to_opaque_target() {
        let p = nullptr(RUNTIME_TYPE_INFO.clone()).unwrap();
        assert!(matches!(p._TYPE.TO, PtrTarget::Opaque(_)));
        assert!(matches!(&p._obj0, Ok(None)));
    }

    #[test]
    fn nullptr_rejects_non_container_type() {
        let err = nullptr(LowLevelType::Float).unwrap_err();
        assert!(
            err.contains("can only point to a Container type"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn cast_pointer_identity_returns_same_pointer_value() {
        let s = StructType::_build(
            "vtable",
            vec![("super".into(), LowLevelType::Signed)],
            GcKind::Gc,
            vec![],
            vec![],
        );
        let T = LowLevelType::Struct(Box::new(s));
        let p = malloc(T.clone(), None, MallocFlavor::Gc, true).unwrap();
        let ptr_T = Ptr::from_container_type(T).unwrap();
        let q = cast_pointer(&ptr_T, &p).unwrap();
        // Identity cast preserves both _identity (same allocation) and
        // _TYPE.
        assert_eq!(p._hashable_identity(), q._hashable_identity());
    }

    #[test]
    fn cast_pointer_downcast_walks_first_field_super_chain() {
        // Build a parent gc struct (rtti=true by `gc_rtti`) and a
        // sub-struct whose first field is `("super", parent)`. cast
        // from sub→parent should yield a pointer whose _TYPE matches
        // parent and which aliases the same allocation.
        let parent = StructType::gc_rtti("parent", vec![("typeptr".into(), LowLevelType::Signed)]);
        let parent_T = LowLevelType::Struct(Box::new(parent.clone()));
        let sub = StructType::gc_rtti(
            "sub",
            vec![
                ("super".into(), parent_T.clone()),
                ("data".into(), LowLevelType::Signed),
            ],
        );
        let sub_T = LowLevelType::Struct(Box::new(sub));
        let sub_ptr = malloc(sub_T, None, MallocFlavor::Gc, true).unwrap();

        let parent_ptr_T = Ptr::from_container_type(parent_T).unwrap();
        let cast = cast_pointer(&parent_ptr_T, &sub_ptr).unwrap();
        // The cast result's `_TYPE` is `Ptr(parent)`.
        let PtrTarget::Struct(s) = &cast._TYPE.TO else {
            panic!("cast result must be Ptr(Struct)");
        };
        assert_eq!(s._name, "parent");
    }

    #[test]
    fn cast_pointer_null_yields_null_of_target_type() {
        let parent = StructType::gc_rtti("parent_n", vec![]);
        let parent_T = LowLevelType::Struct(Box::new(parent));
        let parent_ptr_T = Ptr::from_container_type(parent_T.clone()).unwrap();
        let null = nullptr(parent_T).unwrap();
        let cast = cast_pointer(&parent_ptr_T, &null).unwrap();
        assert!(!cast.nonzero(), "null cast must remain null");
    }

    #[test]
    fn malloc_immortal_gc_struct_produces_live_struct_container() {
        let s = StructType::_build(
            "vtable",
            vec![("super".into(), LowLevelType::Signed)],
            GcKind::Gc,
            vec![],
            vec![],
        );
        let T = LowLevelType::Struct(Box::new(s));
        let p = malloc(T, None, MallocFlavor::Gc, true).unwrap();
        let Ok(Some(_ptr_obj::Struct(inner))) = &p._obj0 else {
            panic!("malloc(Struct, immortal=true) must produce Struct container");
        };
        assert_eq!(inner.TYPE._name, "vtable");
        assert!(inner._getattr("super").is_some());
    }

    #[test]
    fn malloc_immortal_gc_opaque_allocates_opaque_container() {
        let T = RUNTIME_TYPE_INFO.clone();
        let p = malloc(T, None, MallocFlavor::Gc, true).unwrap();
        assert!(matches!(&p._obj0, Ok(Some(_ptr_obj::Opaque(_)))));
        assert!(p._solid);
    }

    #[test]
    fn malloc_rejects_non_container_type() {
        let err = malloc(LowLevelType::Signed, None, MallocFlavor::Gc, true).unwrap_err();
        assert!(err.contains("unmallocable type"), "unexpected error: {err}");
    }

    #[test]
    fn malloc_varsize_struct_initialises_trailing_array_to_requested_length() {
        let s = StructType::_build(
            "S",
            vec![
                ("x".into(), LowLevelType::Signed),
                (
                    "items".into(),
                    LowLevelType::Array(Box::new(ArrayType::new(LowLevelType::Char))),
                ),
            ],
            GcKind::Gc,
            vec![],
            vec![],
        );
        let p = malloc(
            LowLevelType::Struct(Box::new(s)),
            Some(3),
            MallocFlavor::Gc,
            true,
        )
        .unwrap();
        let Some(_ptr_obj::Struct(s)) = p._obj0.as_ref().unwrap().as_ref() else {
            panic!("malloc should return a live Struct pointer");
        };
        let items_value = s
            ._getattr("items")
            .expect("varsize Struct must expose 'items' field");
        let LowLevelValue::Array(items) = items_value else {
            panic!("varsize array field should be initialised as an Array");
        };
        assert_eq!(items.getlength(), 3);
    }

    #[test]
    fn malloc_rejects_gc_flavor_non_immortal_on_non_gc_struct() {
        let s = StructType::_build(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
            GcKind::Raw,
            vec![],
            vec![],
        );
        let err = malloc(
            LowLevelType::Struct(Box::new(s)),
            None,
            MallocFlavor::Gc,
            false,
        )
        .unwrap_err();
        assert!(
            err.contains("gc flavor malloc of a non-GC"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn opaqueptr_mints_named_opaque_container_and_wraps_in_ptr() {
        let p = opaqueptr(RUNTIME_TYPE_INFO.clone(), "ExceptionFoo").unwrap();
        let Ok(Some(_ptr_obj::Opaque(inner))) = &p._obj0 else {
            panic!("opaqueptr must produce an Opaque container");
        };
        assert_eq!(inner._name.as_deref(), Some("ExceptionFoo"));
        assert!(inner.about.is_none());
        assert_eq!(inner.TYPE.tag, "RuntimeTypeInfo");
        assert!(matches!(p._TYPE.TO, PtrTarget::Opaque(_)));
        assert!(p._solid);
    }

    #[test]
    fn opaqueptr_rejects_non_opaque_type() {
        let err = opaqueptr(LowLevelType::Signed, "x").unwrap_err();
        assert!(
            err.contains("opaqueptr() for OpaqueTypes only"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn gc_rtti_struct_has_runtime_type_info_opaque_named_after_struct() {
        let s = StructType::gc_rtti("ExceptionFoo", vec![("msg".into(), LowLevelType::Signed)]);
        let rtti = s
            ._runtime_type_info
            .as_ref()
            .expect("gc_rtti must attach _runtime_type_info");
        assert_eq!(rtti.TYPE.tag, "RuntimeTypeInfo");
        assert_eq!(rtti._name.as_deref(), Some("ExceptionFoo"));
        let Some(LowLevelType::Struct(about)) = rtti.about.as_ref() else {
            panic!("gc_rtti must record about=self on the opaque");
        };
        assert_eq!(about._name, "ExceptionFoo");
    }

    #[test]
    fn gc_struct_without_rtti_leaves_runtime_type_info_none() {
        let s = StructType::gc("PlainStruct", vec![("x".into(), LowLevelType::Signed)]);
        assert!(s._runtime_type_info.is_none());
    }

    #[test]
    fn get_runtime_type_info_returns_ptr_to_attached_opaque() {
        let s = StructType::gc_rtti("ExceptionBar", vec![("msg".into(), LowLevelType::Signed)]);
        let T = LowLevelType::Struct(Box::new(s));
        let p = getRuntimeTypeInfo(&T).unwrap();
        assert!(matches!(p._TYPE.TO, PtrTarget::Opaque(_)));
        let Ok(Some(_ptr_obj::Opaque(inner))) = &p._obj0 else {
            panic!("getRuntimeTypeInfo must produce an Opaque container");
        };
        assert_eq!(inner._name.as_deref(), Some("ExceptionBar"));
    }

    #[test]
    fn get_runtime_type_info_errors_when_struct_lacks_rtti() {
        let s = StructType::gc("NoRttiStruct", vec![("x".into(), LowLevelType::Signed)]);
        let err = getRuntimeTypeInfo(&LowLevelType::Struct(Box::new(s))).unwrap_err();
        assert!(
            err.contains("no attached runtime type info"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn get_runtime_type_info_rejects_raw_structs() {
        let s = StructType::new("RawStruct", vec![("x".into(), LowLevelType::Signed)]);
        let err = getRuntimeTypeInfo(&LowLevelType::Struct(Box::new(s))).unwrap_err();
        assert!(
            err.contains("expected a RttiStruct"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn get_runtime_type_info_errors_on_non_struct_type() {
        let err = getRuntimeTypeInfo(&LowLevelType::Signed).unwrap_err();
        assert!(
            err.contains("expected a RttiStruct"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn attach_runtime_type_info_returns_same_opaque_as_get() {
        let s = StructType::gc_rtti("AttachedStruct", vec![("x".into(), LowLevelType::Signed)]);
        let T = LowLevelType::Struct(Box::new(s));
        let from_attach = attachRuntimeTypeInfo(&T).unwrap();
        let from_get = getRuntimeTypeInfo(&T).unwrap();
        let (Ok(Some(_ptr_obj::Opaque(a))), Ok(Some(_ptr_obj::Opaque(b)))) =
            (&from_attach._obj0, &from_get._obj0)
        else {
            panic!("both helpers must produce Opaque containers");
        };
        assert_eq!(a._identity, b._identity);
        assert_eq!(a._name, b._name);
    }

    #[test]
    fn attach_runtime_type_info_errors_when_gc_struct_lacks_rtti() {
        let s = StructType::gc("AttachedStruct", vec![("x".into(), LowLevelType::Signed)]);
        let T = LowLevelType::Struct(Box::new(s));
        let err = attachRuntimeTypeInfo(&T).unwrap_err();
        assert!(
            err.contains("must have been built with the rtti=True argument"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn attach_runtime_type_info_with_ptrs_stores_query_and_destructor() {
        let mut T = LowLevelType::Struct(Box::new(StructType::gc_rtti(
            "AttachedStruct",
            vec![("x".into(), LowLevelType::Signed)],
        )));
        let self_ptr_t = Ptr::from_container_type(T.clone()).unwrap();
        let rtti_ptr_t = Ptr::from_container_type(RUNTIME_TYPE_INFO.clone()).unwrap();
        let query = functionptr(
            FuncType {
                args: vec![LowLevelType::Ptr(Box::new(self_ptr_t.clone()))],
                result: LowLevelType::Ptr(Box::new(rtti_ptr_t)),
            },
            "query",
            None,
            Some("<query>".into()),
        );
        let destr = functionptr(
            FuncType {
                args: vec![LowLevelType::Ptr(Box::new(self_ptr_t))],
                result: LowLevelType::Void,
            },
            "destr",
            None,
            Some("<destr>".into()),
        );
        let attached =
            attachRuntimeTypeInfo_with_ptrs(&mut T, Some(query.clone()), Some(destr.clone()))
                .unwrap();
        let LowLevelType::Struct(struct_t) = &T else {
            panic!("attachRuntimeTypeInfo_with_ptrs must keep T as a struct");
        };
        let rtti = struct_t
            ._runtime_type_info
            .as_ref()
            .expect("rtti opaque must still be present");
        assert_eq!(
            rtti.query_funcptr
                .as_ref()
                .map(|ptr| ptr._hashable_identity()),
            Some(query._hashable_identity())
        );
        assert_eq!(
            rtti.destructor_funcptr
                .as_ref()
                .map(|ptr| ptr._hashable_identity()),
            Some(destr._hashable_identity())
        );
        let Ok(Some(_ptr_obj::Opaque(attached_rtti))) = &attached._obj0 else {
            panic!("attachRuntimeTypeInfo_with_ptrs must return the RTTI opaque pointer");
        };
        assert!(attached_rtti.query_funcptr.is_some());
        assert!(attached_rtti.destructor_funcptr.is_some());
    }

    #[test]
    fn functionptr_keeps_graph_on_funcobj() {
        let start = Rc::new(RefCell::new(Block::new(vec![])));
        let mut ret = Variable::new();
        ret.set_concretetype(Some(LowLevelType::Void));
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "f",
            start,
            Hlvalue::Variable(ret),
        )));
        let ptr = getfunctionptr(&graph, _getconcretetype).unwrap();
        let funcobj = ptr._obj().unwrap();
        let _ptr_obj::Func(funcobj) = funcobj else {
            panic!("functionptr must expose a function object");
        };
        assert_eq!(funcobj.graph, Some(GraphKey::of(&graph).as_usize()));
    }

    #[test]
    fn getfunctionptr_calls_getconcretetype_for_args_and_result() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static CALLS: AtomicUsize = AtomicUsize::new(0);

        fn counting_getconcretetype(v: &Hlvalue) -> Result<ConcretetypePlaceholder, TyperError> {
            let _ = v;
            CALLS.fetch_add(1, Ordering::Relaxed);
            Ok(LowLevelType::Void)
        }

        let mut a0 = Variable::new();
        a0.set_concretetype(Some(LowLevelType::Signed));
        let mut a1 = Variable::new();
        a1.set_concretetype(Some(LowLevelType::Bool));
        let mut ret = Variable::new();
        ret.set_concretetype(Some(LowLevelType::Void));
        let start = Rc::new(RefCell::new(Block::new(vec![
            Hlvalue::Variable(a0),
            Hlvalue::Variable(a1),
        ])));
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "f",
            start,
            Hlvalue::Variable(ret),
        )));
        CALLS.store(0, Ordering::Relaxed);

        let _ = getfunctionptr(&graph, counting_getconcretetype).unwrap();

        assert_eq!(CALLS.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn getfunctionptr_rejects_missing_concretetype_instead_of_void_fallback() {
        let arg = Hlvalue::Variable(crate::flowspace::model::Variable::new());
        let start = Rc::new(RefCell::new(Block::new(vec![arg])));
        let graph = Rc::new(RefCell::new(FunctionGraph::new("f", start)));

        assert!(getfunctionptr(&graph, _getconcretetype).is_err());
    }

    #[test]
    fn getfunctionptr_copies_llfnobjattrs_from_graph_func() {
        let start = Rc::new(RefCell::new(Block::new(vec![])));
        let mut ret = Variable::new();
        ret.set_concretetype(Some(LowLevelType::Void));
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "graph_name",
            start,
            Hlvalue::Variable(ret),
        )));
        let mut func = GraphFunc::new("py_func", Constant::new(ConstValue::Dict(HashMap::new())));
        func._llfnobjattrs_
            .insert("_name".to_string(), ConstValue::byte_str("forced_name"));
        func._llfnobjattrs_.insert(
            "_callable".to_string(),
            ConstValue::byte_str("forced_callable"),
        );
        func._llfnobjattrs_
            .insert("extra".to_string(), ConstValue::Int(7));
        graph.borrow_mut().func = Some(func);

        let ptr = getfunctionptr(&graph, _getconcretetype).unwrap();
        let _ptr_obj::Func(funcobj) = ptr._obj().unwrap() else {
            panic!("functionptr must expose a function object");
        };
        assert_eq!(funcobj._name, "forced_name");
        assert_eq!(funcobj._callable.as_deref(), Some("forced_callable"));
        assert_eq!(funcobj.attrs.get("extra"), Some(&ConstValue::Int(7)));
    }

    #[test]
    fn lowleveltype_primitive_contains_value_matches_upstream_enforce() {
        use crate::flowspace::model::ConstValue;

        // RPython `LowLevelType._contains_value` (`lltype.py:194-197`):
        // Void admits any value as a valid constant. Non-Void primitives
        // still dispatch through isCompatibleType.
        assert!(LowLevelType::Void.contains_value(&ConstValue::None));
        assert!(LowLevelType::Void.contains_value(&ConstValue::Bool(true)));
        assert!(LowLevelType::Void.contains_value(&ConstValue::Int(7)));

        // RPython `Bool` Primitive accepts bools.
        assert!(LowLevelType::Bool.contains_value(&ConstValue::Bool(false)));
        assert!(LowLevelType::Bool.contains_value(&ConstValue::Bool(true)));
        assert!(!LowLevelType::Bool.contains_value(&ConstValue::Int(0)));

        // RPython `Signed` / `Unsigned` / `SignedLongLong` /
        // `UnsignedLongLong` accept Python int.
        assert!(LowLevelType::Signed.contains_value(&ConstValue::Int(42)));
        assert!(LowLevelType::Unsigned.contains_value(&ConstValue::Int(42)));
        assert!(LowLevelType::SignedLongLong.contains_value(&ConstValue::Int(42)));
        assert!(!LowLevelType::Signed.contains_value(&ConstValue::Bool(true)));

        // RPython `Float` / `SingleFloat` / `LongFloat` accept Python float.
        // pyre stores floats as u64 bit-patterns in ConstValue::Float.
        assert!(LowLevelType::Float.contains_value(&ConstValue::Float(0)));
        assert!(LowLevelType::SingleFloat.contains_value(&ConstValue::Float(0)));
        assert!(LowLevelType::LongFloat.contains_value(&ConstValue::Float(0)));
        assert!(!LowLevelType::Float.contains_value(&ConstValue::Int(0)));
    }

    #[test]
    fn lowleveltype_char_unichar_contains_value_enforces_type_and_single_char_length() {
        use crate::flowspace::model::ConstValue;
        assert!(LowLevelType::Char.contains_value(&ConstValue::byte_str(b"a")));
        assert!(!LowLevelType::Char.contains_value(&ConstValue::byte_str(b"ab")));
        assert!(!LowLevelType::Char.contains_value(&ConstValue::byte_str(b"")));
        assert!(!LowLevelType::Char.contains_value(&ConstValue::uni_str("a")));

        assert!(LowLevelType::UniChar.contains_value(&ConstValue::uni_str("a")));
        assert!(LowLevelType::UniChar.contains_value(&ConstValue::uni_str("π")));
        assert!(!LowLevelType::UniChar.contains_value(&ConstValue::uni_str("πi")));
        assert!(!LowLevelType::UniChar.contains_value(&ConstValue::uni_str("")));
        assert!(!LowLevelType::UniChar.contains_value(&ConstValue::byte_str(b"a")));

        // Non-string variants stay rejected.
        assert!(!LowLevelType::Char.contains_value(&ConstValue::Int(0)));
        assert!(!LowLevelType::UniChar.contains_value(&ConstValue::Bool(false)));
    }

    #[test]
    fn lowleveltype_placeholder_value_is_universally_accepted() {
        use crate::flowspace::model::ConstValue;

        // Placeholder sentinel (`description.NODEFAULT` upstream) must
        // pass `_contains_value` so the normalizecalls rewrite branch
        // can stash it as a transient row-level padding without
        // tripping convert_const validation. See rmodel.rs's
        // inputconst port for the load-bearing use.
        for lltype in [
            LowLevelType::Void,
            LowLevelType::Bool,
            LowLevelType::Signed,
            LowLevelType::Float,
            LowLevelType::Char,
            LowLevelType::UniChar,
            LowLevelType::Ptr(Box::new(Ptr {
                TO: PtrTarget::Func(FuncType {
                    args: vec![],
                    result: LowLevelType::Void,
                }),
            })),
        ] {
            assert!(
                lltype.contains_value(&ConstValue::Placeholder),
                "Placeholder must be universally acceptable (lltype={lltype:?})"
            );
        }
    }

    #[test]
    fn lowleveltype_primitive_short_name_matches_upstream_class_name() {
        // rmodel.py:30 `<%s %s>` formatter and rmodel.py:33
        // `compact_repr` both consume `lowleveltype._short_name()`
        // (Primitive) or `lowleveltype.__name__` (Ptr). Lock in the
        // upstream strings.
        assert_eq!(LowLevelType::Void.short_name(), "Void");
        assert_eq!(LowLevelType::Bool.short_name(), "Bool");
        assert_eq!(LowLevelType::Signed.short_name(), "Signed");
        assert_eq!(LowLevelType::Unsigned.short_name(), "Unsigned");
        assert_eq!(LowLevelType::SignedLongLong.short_name(), "SignedLongLong");
        assert_eq!(
            LowLevelType::UnsignedLongLong.short_name(),
            "UnsignedLongLong"
        );
        assert_eq!(LowLevelType::Float.short_name(), "Float");
        assert_eq!(LowLevelType::SingleFloat.short_name(), "SingleFloat");
        assert_eq!(LowLevelType::LongFloat.short_name(), "LongFloat");
        assert_eq!(LowLevelType::Char.short_name(), "Char");
        assert_eq!(LowLevelType::UniChar.short_name(), "UniChar");
        assert_eq!(LowLevelType::Address.short_name(), "Address");
    }

    #[test]
    fn lowleveltype_address_is_primitive_atomic_and_equal_to_self() {
        // rpython/rtyper/lltypesystem/llmemory.py:650 —
        // `Address = lltype.Primitive("Address", NULL)`. Address must
        // behave as an atomic primitive (is_primitive / _is_atomic) and
        // its default value is the NULL sentinel.
        assert!(LowLevelType::Address.is_primitive());
        assert!(LowLevelType::Address._is_atomic());
        assert_eq!(LowLevelType::Address, LowLevelType::Address);
        assert_ne!(LowLevelType::Address, LowLevelType::Signed);
    }

    #[test]
    fn lowleveltype_address_defl_returns_null_sentinel() {
        // upstream `NULL = fakeaddress(None)`; `Address._defl()` returns
        // NULL. Pyre models the NULL sentinel as `_address::Null` until
        // richer fakeaddress bodies land.
        match LowLevelType::Address._defl() {
            LowLevelValue::Address(_address::Null) => {}
            other => panic!("expected NULL address, got {other:?}"),
        }
    }

    #[test]
    fn typeof_value_address_round_trips_to_address_lowleveltype() {
        let v = LowLevelValue::Address(_address::Null);
        assert_eq!(typeOf_value(&v), LowLevelType::Address);
    }

    #[test]
    fn lowleveltype_address_contains_value_accepts_fakeaddress_constants() {
        use crate::flowspace::model::ConstValue;
        assert!(!LowLevelType::Address.contains_value(&ConstValue::Int(0)));
        assert!(!LowLevelType::Address.contains_value(&ConstValue::None));
        assert!(
            LowLevelType::Address.contains_value(&ConstValue::LLAddress(_address::Null)),
            "Address must accept llmemory.NULL fakeaddress"
        );
        assert!(LowLevelType::Address.contains_value(&ConstValue::Placeholder));
    }

    #[test]
    fn lowleveltype_ptr_short_name_follows_upstream_prefix() {
        // upstream `Ptr._short_name` (`lltype.py:748`):
        // `'Ptr %s' % self.TO._short_name()`. Note this is distinct
        // from `Ptr.__str__` which uses `"* %s"` prefix.
        let ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![LowLevelType::Void, LowLevelType::Void],
                result: LowLevelType::Void,
            }),
        }));
        assert_eq!(ptr.short_name(), "Ptr Func(Void, Void)->Void");
    }

    #[test]
    fn lowleveltype_module_constants_match_variant_singletons() {
        // `VOID` / `BOOL` / `SIGNED` / `FLOAT` / `CHAR` / `UNICHAR` are
        // the pyre re-exports of upstream's `lltype.Void` / `lltype.Bool`
        // / ... (which are instance singletons). Lock the identity so
        // downstream `use lltype::{Void, Bool, ...}` imports see a
        // stable match source.
        assert_eq!(VOID, LowLevelType::Void);
        assert_eq!(BOOL, LowLevelType::Bool);
        assert_eq!(SIGNED, LowLevelType::Signed);
        assert_eq!(FLOAT, LowLevelType::Float);
        assert_eq!(CHAR, LowLevelType::Char);
        assert_eq!(UNICHAR, LowLevelType::UniChar);
    }

    #[test]
    fn delayed_pointer_raises_on_obj_access() {
        let ptr = _ptr::new(
            Ptr {
                TO: PtrTarget::Func(FuncType {
                    args: vec![],
                    result: LowLevelType::Void,
                }),
            },
            Err(DelayedPointer),
        );
        assert_eq!(ptr._obj(), Err(DelayedPointer));
    }

    #[test]
    fn delayed_pointer_equality_requires_same_ptr_instance() {
        let delayed1 = _ptr::new(
            Ptr {
                TO: PtrTarget::Struct(StructType::new(
                    "S",
                    vec![("x".into(), LowLevelType::Signed)],
                )),
            },
            Err(DelayedPointer),
        );
        let delayed2 = _ptr::new(delayed1._TYPE.clone(), Err(DelayedPointer));
        assert!(delayed1 == delayed1);
        assert!(delayed1 != delayed2);
    }

    #[test]
    #[should_panic(expected = "comparing")]
    fn ptr_equality_rejects_different_pointer_types() {
        let left = Ptr {
            TO: PtrTarget::Struct(StructType::new(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        }
        ._example();
        let right = Ptr {
            TO: PtrTarget::Struct(StructType::new(
                "T",
                vec![("y".into(), LowLevelType::Signed)],
            )),
        }
        ._example();
        let _ = left == right;
    }

    #[test]
    fn ptr_needsgc_tracks_target_gckind() {
        let raw_ptr = Ptr {
            TO: PtrTarget::Struct(StructType::new(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        };
        assert!(!raw_ptr._needsgc());

        let gc_ptr = Ptr {
            TO: PtrTarget::Struct(StructType::gc(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        };
        assert!(gc_ptr._needsgc());
    }

    #[test]
    fn gc_ptr_identityhash_tracks_underlying_object_identity() {
        let ptr1 = Ptr {
            TO: PtrTarget::Struct(StructType::gc(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        }
        ._example();
        let ptr2 = _ptr {
            _identity: ptr1._identity,
            _TYPE: ptr1._TYPE.clone(),
            _solid: ptr1._solid,
            _obj0: ptr1._obj0.clone(),
            _weak: ptr1._weak,
        };
        let ptr3 = Ptr {
            TO: PtrTarget::Struct(StructType::gc(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        }
        ._example();
        assert_eq!(identityhash(&ptr1), identityhash(&ptr2));
        assert_ne!(identityhash(&ptr1), identityhash(&ptr3));
    }

    #[test]
    #[should_panic]
    fn raw_ptr_identityhash_rejects_non_gc_pointer() {
        let ptr = Ptr {
            TO: PtrTarget::Struct(StructType::new(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        }
        ._example();
        let _ = identityhash(&ptr);
    }

    #[test]
    fn ptr_same_obj_requires_same_underlying_object() {
        let ptr1 = Ptr {
            TO: PtrTarget::Struct(StructType::new(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        }
        ._example();
        let ptr2 = Ptr {
            TO: PtrTarget::Struct(StructType::new(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        }
        ._example();
        assert!(!ptr1._same_obj(&ptr2).unwrap());
        assert!(ptr1._same_obj(&ptr1).unwrap());
    }

    #[test]
    fn interior_ptr_same_obj_compares_current_target_object() {
        let iptr1 = _interior_ptr {
            _T: LowLevelType::Signed,
            _parent: LowLevelValue::Struct(Box::new(
                StructType::new("S", vec![("x".into(), LowLevelType::Signed)])._container_example(),
            )),
            _offsets: vec![InteriorOffset::Field("x".into())],
        };
        let iptr2 = iptr1.clone();
        assert!(iptr1._same_obj(&iptr2));
    }

    #[test]
    fn interior_ptr_equality_matches_current_target_object() {
        let iptr1 = _interior_ptr {
            _T: LowLevelType::Signed,
            _parent: LowLevelValue::Struct(Box::new(
                StructType::new("S", vec![("x".into(), LowLevelType::Signed)])._container_example(),
            )),
            _offsets: vec![InteriorOffset::Field("x".into())],
        };
        let iptr2 = iptr1.clone();
        assert!(iptr1 == iptr2);
    }

    #[test]
    #[should_panic(expected = "comparing")]
    fn interior_ptr_equality_rejects_different_pointer_types() {
        let left = _interior_ptr {
            _T: LowLevelType::Signed,
            _parent: LowLevelValue::Struct(Box::new(
                StructType::new("S", vec![("x".into(), LowLevelType::Signed)])._container_example(),
            )),
            _offsets: vec![InteriorOffset::Field("x".into())],
        };
        let right = _interior_ptr {
            _T: LowLevelType::Signed,
            _parent: LowLevelValue::Struct(Box::new(
                StructType::new("S", vec![("y".into(), LowLevelType::Signed)])._container_example(),
            )),
            _offsets: vec![InteriorOffset::Field("y".into())],
        };
        let _ = left == right;
    }

    #[test]
    fn forward_reference_become_rejects_non_container() {
        let mut forward_ref = ForwardReference::new();
        let err = forward_ref
            .r#become(LowLevelType::Signed)
            .expect_err("ForwardReference must reject non-container targets");
        assert!(err.contains("can only be to a container"));
    }

    #[test]
    fn forward_reference_become_rejects_conflicting_gckind() {
        let mut forward_ref = ForwardReference::gc();
        let err = forward_ref
            .r#become(LowLevelType::Struct(Box::new(StructType::new(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            ))))
            .expect_err("GcForwardReference must reject raw targets");
        assert!(err.contains("conflicting gckind"));
    }

    #[test]
    fn forward_reference_become_allows_resolved_struct_example() {
        let mut forward_ref = ForwardReference::new();
        forward_ref
            .r#become(LowLevelType::Struct(Box::new(StructType::new(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            ))))
            .unwrap();
        let ptr = Ptr {
            TO: PtrTarget::ForwardReference(forward_ref),
        }
        ._example();
        assert_eq!(ptr.getattr("x").unwrap(), LowLevelValue::Signed(0));
    }

    #[test]
    fn forward_reference_clones_observe_become() {
        // lltype.py:624-625 mutates the ForwardReference object in place;
        // all aliases must observe the resolved container.
        let mut forward_ref = ForwardReference::new();
        let alias = forward_ref.clone();
        forward_ref
            .r#become(LowLevelType::Struct(Box::new(StructType::new(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            ))))
            .unwrap();

        let ptr = Ptr {
            TO: PtrTarget::ForwardReference(alias),
        }
        ._example();
        assert_eq!(ptr.getattr("x").unwrap(), LowLevelValue::Signed(0));
    }

    #[test]
    fn resolved_forward_reference_compares_as_real_container() {
        // lltype.py:624-625 rebinds __class__/__dict__; after become(),
        // the forward reference participates in LowLevelType equality/hash
        // as the real container type.
        let real = LowLevelType::Struct(Box::new(StructType::new(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
        )));
        let mut forward_ref = ForwardReference::new();
        forward_ref.r#become(real.clone()).unwrap();
        let resolved = LowLevelType::ForwardReference(Box::new(forward_ref));

        assert_eq!(resolved, real);

        let mut left = std::collections::hash_map::DefaultHasher::new();
        resolved.hash(&mut left);
        let mut right = std::collections::hash_map::DefaultHasher::new();
        real.hash(&mut right);
        assert_eq!(left.finish(), right.finish());
    }

    #[test]
    fn cyclic_forward_reference_hash_uses_zero_on_re_entry_so_isomorphic_cycles_match() {
        // Companion to the eq cycle test: hash and eq must agree
        // (Eq+Hash contract). Two distinct ForwardReferences with
        // self-referential isomorphic Struct shapes compare equal,
        // therefore must hash equal. RPython `saferecursive(get_hash,
        // 0)` (lltype.py:136) yields 0 on re-entry; hashing the Arc
        // identity instead would diverge per allocation.
        let fwd_a = ForwardReference::gc();
        let s_a = StructType::gc(
            "S",
            vec![(
                "next".into(),
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::ForwardReference(fwd_a.clone()),
                })),
            )],
        );
        fwd_a.r#become(LowLevelType::Struct(Box::new(s_a))).unwrap();

        let fwd_b = ForwardReference::gc();
        let s_b = StructType::gc(
            "S",
            vec![(
                "next".into(),
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::ForwardReference(fwd_b.clone()),
                })),
            )],
        );
        fwd_b.r#become(LowLevelType::Struct(Box::new(s_b))).unwrap();

        assert!(!Arc::ptr_eq(&fwd_a.target, &fwd_b.target));
        let lhs = LowLevelType::ForwardReference(Box::new(fwd_a));
        let rhs = LowLevelType::ForwardReference(Box::new(fwd_b));

        let mut left_hasher = std::collections::hash_map::DefaultHasher::new();
        lhs.hash(&mut left_hasher);
        let mut right_hasher = std::collections::hash_map::DefaultHasher::new();
        rhs.hash(&mut right_hasher);
        assert_eq!(left_hasher.finish(), right_hasher.finish());
    }

    #[test]
    fn cyclic_forward_reference_equality_short_circuits_to_true_on_re_entry() {
        // Two distinct ForwardReferences fwd_a and fwd_b, each
        // resolved to a Struct containing a Ptr looping back to
        // itself. Comparing them recursively re-enters the same
        // ForwardReference comparison; lltype.py:74
        // `saferecursive(safe_equal, True)` short-circuits to True
        // on re-entry so structurally identical cyclic types compare
        // equal. Returning False there would propagate up through
        // the Struct field comparison and report unequal.
        let fwd_a = ForwardReference::gc();
        let s_a = StructType::gc(
            "S",
            vec![(
                "next".into(),
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::ForwardReference(fwd_a.clone()),
                })),
            )],
        );
        fwd_a.r#become(LowLevelType::Struct(Box::new(s_a))).unwrap();

        let fwd_b = ForwardReference::gc();
        let s_b = StructType::gc(
            "S",
            vec![(
                "next".into(),
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::ForwardReference(fwd_b.clone()),
                })),
            )],
        );
        fwd_b.r#become(LowLevelType::Struct(Box::new(s_b))).unwrap();

        assert!(!Arc::ptr_eq(&fwd_a.target, &fwd_b.target));
        let lhs = LowLevelType::ForwardReference(Box::new(fwd_a));
        let rhs = LowLevelType::ForwardReference(Box::new(fwd_b));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn ptr_to_resolved_forward_reference_compares_as_ptr_to_real_container() {
        let real_struct = StructType::new("S", vec![("x".into(), LowLevelType::Signed)]);
        let mut forward_ref = ForwardReference::new();
        forward_ref
            .r#become(LowLevelType::Struct(Box::new(real_struct.clone())))
            .unwrap();

        let through_forward_ref = Ptr {
            TO: PtrTarget::ForwardReference(forward_ref),
        };
        let direct = Ptr {
            TO: PtrTarget::Struct(real_struct),
        };

        assert_eq!(through_forward_ref, direct);

        let mut left = std::collections::hash_map::DefaultHasher::new();
        through_forward_ref.hash(&mut left);
        let mut right = std::collections::hash_map::DefaultHasher::new();
        direct.hash(&mut right);
        assert_eq!(left.finish(), right.finish());
    }

    #[test]
    #[should_panic(expected = "ForwardReference object is not hashable")]
    fn unresolved_forward_reference_hash_panics() {
        // lltype.py:627-628 — unresolved ForwardReference.__hash__ raises.
        let forward_ref = ForwardReference::new();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        forward_ref.hash(&mut hasher);
    }

    #[test]
    #[should_panic(expected = "_ptr._become: type mismatch")]
    fn ptr_become_panics_on_type_mismatch() {
        // lltype.py:1416 — `assert self._TYPE == other._TYPE` in `_become`.
        let self_ptr = Ptr {
            TO: PtrTarget::Struct(StructType::gc(
                "A",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        }
        ._example();
        let other_ptr = Ptr {
            TO: PtrTarget::Struct(StructType::gc(
                "B",
                vec![("y".into(), LowLevelType::Signed)],
            )),
        }
        ._example();
        self_ptr._become(&other_ptr);
    }

    #[test]
    #[should_panic(expected = "_ptr._become: cannot reassign a weak pointer")]
    fn ptr_become_panics_when_self_is_weak() {
        // lltype.py:1417 — `assert not self._weak` in `_become`.
        let mut self_ptr = Ptr {
            TO: PtrTarget::Struct(StructType::gc(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        }
        ._example();
        self_ptr._weak = true;
        let other_ptr = Ptr {
            TO: PtrTarget::Struct(StructType::gc(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        }
        ._example();
        self_ptr._become(&other_ptr);
    }

    #[test]
    fn struct_pointer_example_exposes_field_defaults() {
        let ptr_t = Ptr {
            TO: PtrTarget::Struct(StructType::new(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        };
        let ptr = ptr_t._example();
        assert_eq!(ptr.getattr("x").unwrap(), LowLevelValue::Signed(0));
    }

    #[test]
    fn struct_pointer_lookup_adtmeth_binds_to_same_ptrtype() {
        let ptr_t = Ptr {
            TO: PtrTarget::Struct(StructType::with_adtmeths(
                "S",
                vec![],
                vec![(
                    "f".into(),
                    ConstValue::HostObject(crate::flowspace::model::HostObject::new_user_function(
                        crate::flowspace::model::GraphFunc::new(
                            "f",
                            crate::flowspace::model::Constant::new(ConstValue::Dict(
                                Default::default(),
                            )),
                        ),
                    )),
                )],
            )),
        };
        let ptr = ptr_t._example();
        let LowLevelAdtMember::Method { ll_ptrtype, .. } = ptr._lookup_adtmeth("f").unwrap() else {
            panic!("expected bound adt method");
        };
        assert_eq!(ll_ptrtype, LowLevelPointerType::Ptr(ptr_t));
    }

    #[test]
    fn struct_pointer_getattr_exposes_inlined_struct_field_as_pointer() {
        let inner = StructType::new("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let outer = Ptr {
            TO: PtrTarget::Struct(StructType::new(
                "Outer",
                vec![("x".into(), LowLevelType::Struct(Box::new(inner.clone())))],
            )),
        };
        let value = outer._example().getattr("x").unwrap();
        let LowLevelValue::Ptr(inner_ptr) = value else {
            panic!("expected nested struct field to be exposed as a pointer");
        };
        assert_eq!(inner_ptr._TYPE.TO, PtrTarget::Struct(inner));
    }

    #[test]
    fn gc_struct_pointer_getattr_exposes_raw_struct_field_as_interior_ptr() {
        let inner = StructType::new("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let outer = Ptr {
            TO: PtrTarget::Struct(StructType::gc(
                "Outer",
                vec![("x".into(), LowLevelType::Struct(Box::new(inner.clone())))],
            )),
        };
        let value = outer._example().getattr("x").unwrap();
        let LowLevelValue::InteriorPtr(inner_ptr) = value else {
            panic!("expected raw field of gc struct to be exposed as an interior ptr");
        };
        assert_eq!(
            inner_ptr._TYPE(),
            InteriorPtr {
                PARENTTYPE: Box::new(LowLevelType::Struct(Box::new(StructType::gc(
                    "Outer",
                    vec![("x".into(), LowLevelType::Struct(Box::new(inner.clone())))],
                ),))),
                TO: Box::new(LowLevelType::Struct(Box::new(inner))),
                offsets: vec![InteriorOffset::Field("x".into())],
            }
        );
    }

    #[test]
    fn array_pointer_len_and_getitem_follow_array_surface() {
        let ptr_t = Ptr {
            TO: PtrTarget::Array(ArrayType::new(LowLevelType::Signed)),
        };
        let ptr = ptr_t._example();
        assert_eq!(ptr.len().unwrap(), 1);
        assert_eq!(ptr.getitem(0).unwrap(), LowLevelValue::Signed(0));
        assert_eq!(ptr._fixedlength(), Ok(None));
    }

    #[test]
    fn gc_array_pointer_getitem_exposes_raw_struct_item_as_interior_ptr() {
        let ptr_t = Ptr {
            TO: PtrTarget::Array(ArrayType::gc(LowLevelType::Struct(Box::new(
                StructType::new("Item", vec![("x".into(), LowLevelType::Signed)]),
            )))),
        };
        let value = ptr_t._example().getitem(0).unwrap();
        let LowLevelValue::InteriorPtr(iptr) = value else {
            panic!("expected gc array item to be exposed as an interior ptr");
        };
        assert_eq!(iptr._offsets, vec![InteriorOffset::Index(0)]);
    }

    #[test]
    fn gc_struct_pointer_getattr_exposes_opaque_field_as_ptr_not_interior_ptr() {
        let outer = Ptr {
            TO: PtrTarget::Struct(StructType::gc(
                "Outer",
                vec![(
                    "x".into(),
                    LowLevelType::Opaque(Box::new(OpaqueType::new("OpaqueX"))),
                )],
            )),
        };
        let value = outer._example().getattr("x").unwrap();
        let LowLevelValue::Ptr(opaque_ptr) = value else {
            panic!("expected opaque field of gc struct to stay a plain ptr");
        };
        assert_eq!(
            opaque_ptr._TYPE.TO,
            PtrTarget::Opaque(OpaqueType::new("OpaqueX"))
        );
    }

    #[test]
    fn fixedsize_array_pointer_reports_fixed_length() {
        let ptr_t = Ptr {
            TO: PtrTarget::FixedSizeArray(FixedSizeArrayType::new(LowLevelType::Signed, 3)),
        };
        let ptr = ptr_t._example();
        assert_eq!(ptr.len().unwrap(), 3);
        assert_eq!(ptr._fixedlength(), Ok(Some(3)));
    }

    #[test]
    fn array_pointer_setitem_checks_item_type() {
        let mut ptr = Ptr {
            TO: PtrTarget::Array(ArrayType::new(LowLevelType::Signed)),
        }
        ._example();
        ptr.setitem(0, LowLevelValue::Signed(7)).unwrap();
        assert_eq!(ptr.getitem(0).unwrap(), LowLevelValue::Signed(7));
        let err = ptr
            .setitem(0, LowLevelValue::Bool(false))
            .expect_err("array setitem must reject mismatched item types");
        assert!(err.contains("expect"));
    }

    #[test]
    fn fixedsize_array_pointer_getitem_reports_out_of_bounds() {
        let ptr = Ptr {
            TO: PtrTarget::FixedSizeArray(FixedSizeArrayType::new(LowLevelType::Signed, 0)),
        }
        ._example();
        let err = ptr
            .getitem(0)
            .expect_err("zero-length fixed-size array must reject index 0");
        assert!(err.contains("out of bounds"));
    }

    #[test]
    fn interior_ptr_obj_uses_actual_index_offset() {
        let mut parent = StructType::new(
            "S",
            vec![(
                "arr".into(),
                LowLevelType::FixedSizeArray(Box::new(FixedSizeArrayType::new(
                    LowLevelType::Signed,
                    3,
                ))),
            )],
        )
        ._container_example();
        // The `_struct._fields` cell is shared via `Arc<Mutex<>>` so
        // modifying `arr.items` propagates back to `parent._fields`.
        let arr_value = parent
            ._getattr("arr")
            .expect("parent must expose 'arr' field");
        let LowLevelValue::Array(arr) = arr_value else {
            panic!("expected array field");
        };
        arr.setitem(1, LowLevelValue::Signed(7));
        let iptr = _interior_ptr {
            _T: LowLevelType::Signed,
            _parent: LowLevelValue::Struct(Box::new(parent)),
            _offsets: vec![
                InteriorOffset::Field("arr".into()),
                InteriorOffset::Index(1),
            ],
        };
        assert_eq!(iptr._obj(), LowLevelValue::Signed(7));
    }

    #[test]
    fn interior_ptr_setitem_updates_parent_storage() {
        let mut iptr = _interior_ptr {
            _T: LowLevelType::Array(Box::new(ArrayType::new(LowLevelType::Signed))),
            _parent: LowLevelValue::Struct(Box::new(
                StructType::new(
                    "S",
                    vec![(
                        "arr".into(),
                        LowLevelType::Array(Box::new(ArrayType::new(LowLevelType::Signed))),
                    )],
                )
                ._container_example(),
            )),
            _offsets: vec![InteriorOffset::Field("arr".into())],
        };
        iptr.setitem(0, LowLevelValue::Signed(9)).unwrap();
        let LowLevelValue::Array(arr) = iptr._obj() else {
            panic!("expected array after interior setitem");
        };
        assert_eq!(arr.getitem(0), Some(LowLevelValue::Signed(9)));
    }

    #[test]
    fn interior_ptr_exposes_opaque_child_as_interior_ptr() {
        let iptr = _interior_ptr {
            _T: LowLevelType::Struct(Box::new(StructType::new(
                "S",
                vec![(
                    "x".into(),
                    LowLevelType::Opaque(Box::new(OpaqueType::new("OpaqueX"))),
                )],
            ))),
            _parent: LowLevelValue::Struct(Box::new(
                StructType::new(
                    "S",
                    vec![(
                        "x".into(),
                        LowLevelType::Opaque(Box::new(OpaqueType::new("OpaqueX"))),
                    )],
                )
                ._container_example(),
            )),
            _offsets: vec![],
        };
        let value = iptr.getattr("x").unwrap();
        let LowLevelValue::InteriorPtr(opaque_iptr) = value else {
            panic!("expected interior ptr to keep opaque child as interior ptr");
        };
        assert_eq!(
            opaque_iptr._TYPE(),
            InteriorPtr {
                PARENTTYPE: Box::new(LowLevelType::Struct(Box::new(StructType::new(
                    "S",
                    vec![(
                        "x".into(),
                        LowLevelType::Opaque(Box::new(OpaqueType::new("OpaqueX"))),
                    )],
                )))),
                TO: Box::new(LowLevelType::Opaque(Box::new(OpaqueType::new("OpaqueX")))),
                offsets: vec![InteriorOffset::Field("x".into())],
            }
        );
    }

    #[test]
    #[should_panic(expected = "do not test an interior pointer for nullity")]
    fn interior_ptr_nonzero_panics() {
        let iptr = _interior_ptr {
            _T: LowLevelType::Signed,
            _parent: LowLevelValue::Struct(Box::new(
                StructType::new("S", vec![("x".into(), LowLevelType::Signed)])._container_example(),
            )),
            _offsets: vec![InteriorOffset::Field("x".into())],
        };
        let _ = iptr.nonzero();
    }

    #[test]
    #[should_panic(expected = "field name")]
    fn struct_new_rejects_underscore_prefix_field() {
        // lltype.py:267-269 — NameError on leading underscore.
        let _ = StructType::new("S", vec![("_hidden".into(), LowLevelType::Signed)]);
    }

    #[test]
    #[should_panic(expected = "repeated field name")]
    fn struct_new_rejects_repeated_field_name() {
        // lltype.py:271-272 — TypeError on repeated field.
        let _ = StructType::new(
            "S",
            vec![
                ("x".into(), LowLevelType::Signed),
                ("x".into(), LowLevelType::Bool),
            ],
        );
    }

    #[test]
    #[should_panic(expected = "cannot inline")]
    fn struct_new_rejects_gc_container_inlined_past_first_field() {
        // lltype.py:274-279 — a gc container can only be inlined as the
        // first field of a struct with matching gckind.
        let gc_inner = StructType::gc("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let _ = StructType::gc(
            "Outer",
            vec![
                ("first".into(), LowLevelType::Signed),
                ("bad".into(), LowLevelType::Struct(Box::new(gc_inner))),
            ],
        );
    }

    #[test]
    fn struct_new_allows_gc_first_field_of_matching_gckind() {
        // lltype.py:275-276 — ok to inline XxContainer as first field of
        // XxStruct when _gckinds match.
        let gc_inner = StructType::gc("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let outer = StructType::gc(
            "Outer",
            vec![
                (
                    "head".into(),
                    LowLevelType::Struct(Box::new(gc_inner.clone())),
                ),
                ("x".into(), LowLevelType::Signed),
            ],
        );
        assert_eq!(outer._names, vec!["head".to_string(), "x".to_string()]);
    }

    #[test]
    fn struct_short_name_prefixes_with_kind() {
        // lltype.py:358-359.
        let raw = StructType::new("S", vec![("x".into(), LowLevelType::Signed)]);
        assert_eq!(raw._short_name(), "Struct S");
        let gc = StructType::gc("S", vec![("x".into(), LowLevelType::Signed)]);
        assert_eq!(gc._short_name(), "GcStruct S");
    }

    #[test]
    fn struct_is_atomic_walks_fields() {
        // lltype.py:314-318.
        let plain = StructType::new(
            "S",
            vec![
                ("a".into(), LowLevelType::Signed),
                ("b".into(), LowLevelType::Bool),
            ],
        );
        assert!(plain._is_atomic());
        let with_opaque = StructType::new(
            "S",
            vec![(
                "o".into(),
                LowLevelType::Opaque(Box::new(OpaqueType::new("T"))),
            )],
        );
        assert!(!with_opaque._is_atomic());
    }

    #[test]
    fn struct_names_without_voids_filters_voids() {
        // lltype.py:333-334.
        let s = StructType::new(
            "S",
            vec![
                ("keep".into(), LowLevelType::Signed),
                ("drop".into(), LowLevelType::Void),
            ],
        );
        assert_eq!(s._names_without_voids(), vec!["keep".to_string()]);
    }

    #[test]
    fn frozendict_fields_are_order_insensitive_for_extras() {
        // lltype.py:90-95, 208-210 — _adtmeths/_hints are frozendict,
        // so dict item order must not affect type equality or hash.
        let left = StructType::_build(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
            GcKind::Raw,
            vec![
                ("a".into(), ConstValue::Bool(true)),
                ("b".into(), ConstValue::Bool(false)),
            ],
            vec![
                ("immutable".into(), ConstValue::Bool(true)),
                ("render_as_void".into(), ConstValue::Bool(false)),
            ],
        );
        let right = StructType::_build(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
            GcKind::Raw,
            vec![
                ("b".into(), ConstValue::Bool(false)),
                ("a".into(), ConstValue::Bool(true)),
            ],
            vec![
                ("render_as_void".into(), ConstValue::Bool(false)),
                ("immutable".into(), ConstValue::Bool(true)),
            ],
        );

        assert_eq!(left, right);

        let mut left_hash = std::collections::hash_map::DefaultHasher::new();
        left.hash(&mut left_hash);
        let mut right_hash = std::collections::hash_map::DefaultHasher::new();
        right.hash(&mut right_hash);
        assert_eq!(left_hash.finish(), right_hash.finish());
    }

    #[test]
    fn struct_first_struct_matches_leading_gc_child() {
        // lltype.py:296-303.
        let inner = StructType::gc("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let outer = StructType::gc(
            "Outer",
            vec![
                ("head".into(), LowLevelType::Struct(Box::new(inner.clone()))),
                ("x".into(), LowLevelType::Signed),
            ],
        );
        let (name, child) = outer._first_struct().expect("leading gc struct expected");
        assert_eq!(name, "head");
        assert_eq!(child._name, "Inner");
    }

    #[test]
    fn struct_is_varsize_tracks_trailing_array_field() {
        // lltype.py:288-292 — trailing Array field sets _arrayfld;
        // FixedSizeArray does not.
        let varsize = StructType::new(
            "Vs",
            vec![
                ("len".into(), LowLevelType::Signed),
                (
                    "items".into(),
                    LowLevelType::Array(Box::new(ArrayType::new(LowLevelType::Signed))),
                ),
            ],
        );
        assert!(varsize._is_varsize());
        assert_eq!(varsize._arrayfld.as_deref(), Some("items"));

        let fixed = StructType::new(
            "Fs",
            vec![(
                "items".into(),
                LowLevelType::FixedSizeArray(Box::new(FixedSizeArrayType::new(
                    LowLevelType::Signed,
                    4,
                ))),
            )],
        );
        assert!(!fixed._is_varsize());
        assert_eq!(fixed._arrayfld, None);
    }

    #[test]
    fn struct_note_inlined_into_rejects_gc_past_first_field() {
        // lltype.py:305-312 — _note_inlined_into guard.
        let gc_child = StructType::gc("Child", vec![("x".into(), LowLevelType::Signed)]);
        let gc_parent = LowLevelType::Struct(Box::new(StructType::gc("Parent", vec![])));
        assert!(gc_child._note_inlined_into(&gc_parent, true).is_ok());
        let err = gc_child
            ._note_inlined_into(&gc_parent, false)
            .expect_err("gc struct inlined past first field must be rejected");
        assert!(err.contains("GcStruct can only be inlined"));
    }

    #[test]
    #[should_panic(expected = "last field")]
    fn struct_new_rejects_array_in_non_last_field() {
        // lltype.py:281-288 calls Array._note_inlined_into for each field.
        let _ = StructType::new(
            "S",
            vec![
                (
                    "items".into(),
                    LowLevelType::Array(Box::new(ArrayType::new(LowLevelType::Signed))),
                ),
                ("tail".into(), LowLevelType::Signed),
            ],
        );
    }

    #[test]
    #[should_panic(expected = "cannot be inlined")]
    fn struct_new_rejects_gc_opaque_even_as_first_gc_field() {
        // lltype.py:592-596 — GcOpaqueType is never inlineable.
        let _ = StructType::gc(
            "S",
            vec![(
                "opaque".into(),
                LowLevelType::Opaque(Box::new(OpaqueType::gc("O"))),
            )],
        );
    }

    #[test]
    #[should_panic(expected = "cannot have")]
    fn array_new_rejects_gc_container_item() {
        // lltype.py:434-436 — Array cannot have a gc container item.
        let gc_inner = StructType::gc("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let _ = ArrayType::new(LowLevelType::Struct(Box::new(gc_inner)));
    }

    #[test]
    #[should_panic(expected = "last field")]
    fn array_new_rejects_raw_array_item() {
        // lltype.py:437 calls OF._note_inlined_into(self, first=False, last=False).
        let inner = ArrayType::new(LowLevelType::Signed);
        let _ = ArrayType::new(LowLevelType::Array(Box::new(inner)));
    }

    #[test]
    #[should_panic(expected = "cannot have")]
    fn fixedsize_array_new_rejects_gc_container_item() {
        // lltype.py:518-520 — same restriction on FixedSizeArray.
        let gc_inner = StructType::gc("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let _ = FixedSizeArrayType::new(LowLevelType::Struct(Box::new(gc_inner)), 4);
    }

    #[test]
    #[should_panic(expected = "last field")]
    fn fixedsize_array_new_rejects_raw_array_item() {
        // lltype.py:521 applies the same inline-position check to OF.
        let inner = ArrayType::new(LowLevelType::Signed);
        let _ = FixedSizeArrayType::new(LowLevelType::Array(Box::new(inner)), 4);
    }

    #[test]
    fn array_new_allows_raw_struct_item() {
        // lltype.py:428-432 — raw container items are fine.
        let raw_inner = StructType::new("Inner", vec![("y".into(), LowLevelType::Signed)]);
        let arr = ArrayType::new(LowLevelType::Struct(Box::new(raw_inner)));
        assert_eq!(arr._gckind, GcKind::Raw);
    }

    #[test]
    fn array_short_name_prefixes_with_kind() {
        // lltype.py:475-480 — Array/GcArray _short_name.
        assert_eq!(
            ArrayType::new(LowLevelType::Signed)._short_name(),
            "Array Signed"
        );
        assert_eq!(
            ArrayType::gc(LowLevelType::Signed)._short_name(),
            "GcArray Signed"
        );
    }

    #[test]
    fn fixedsize_array_short_name_carries_length_and_item() {
        // lltype.py:532-536.
        assert_eq!(
            FixedSizeArrayType::new(LowLevelType::Signed, 3)._short_name(),
            "FixedSizeArray 3 Signed"
        );
    }

    #[test]
    fn array_is_atomic_walks_item_type() {
        assert!(ArrayType::new(LowLevelType::Signed)._is_atomic());
        let with_opaque = ArrayType::new(LowLevelType::Opaque(Box::new(OpaqueType::new("T"))));
        assert!(!with_opaque._is_atomic());
    }

    #[test]
    fn array_note_inlined_into_requires_last_struct_slot() {
        // lltype.py:441-448 — last field of a Struct only.
        let arr = ArrayType::new(LowLevelType::Signed);
        let parent_struct = LowLevelType::Struct(Box::new(StructType::new("S", vec![])));
        assert!(arr._note_inlined_into(&parent_struct, true).is_ok());
        let err = arr
            ._note_inlined_into(&parent_struct, false)
            .expect_err("array inlined in non-last position must be rejected");
        assert!(err.contains("last field"));
    }

    #[test]
    fn array_note_inlined_into_rejects_gc_array() {
        // lltype.py:445-446 — gc arrays never inline.
        let gc_arr = ArrayType::gc(LowLevelType::Signed);
        let parent_struct = LowLevelType::Struct(Box::new(StructType::new("S", vec![])));
        let err = gc_arr
            ._note_inlined_into(&parent_struct, true)
            .expect_err("gc array must not inline");
        assert!(err.contains("GC array"));
    }

    #[test]
    fn null_ptr_equality_handles_both_sides_null() {
        // lltype.py:1185-1195 — pointer equality should handle null-null
        // (equal) and null-nonnull (unequal) without panicking at _obj().
        let ptr_t = Ptr {
            TO: PtrTarget::Struct(StructType::new(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        };
        let null_a = _ptr::new(ptr_t.clone(), Ok(None));
        let null_b = _ptr::new(ptr_t.clone(), Ok(None));
        assert!(
            null_a == null_b,
            "two null pointers of same type compare equal"
        );
        let nonnull = ptr_t._example();
        assert!(null_a != nonnull, "null vs nonnull must be unequal");
        assert!(null_a._same_obj(&null_b).unwrap());
        assert!(!null_a._same_obj(&nonnull).unwrap());
    }

    #[test]
    #[should_panic(expected = "wrong argument number")]
    fn func_call_rejects_wrong_arity() {
        // lltype.py:1352-1354.
        let f = _func::new(
            FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Signed,
            },
            "f".into(),
            None,
            Some("impl".into()),
            HashMap::new(),
        );
        let _ = f.call(&[]);
    }

    #[test]
    #[should_panic(expected = "wrong argument type")]
    fn func_call_rejects_wrong_arg_type() {
        // lltype.py:1355-1380 — per-arg type mismatch raises TypeError.
        let f = _func::new(
            FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Signed,
            },
            "f".into(),
            None,
            Some("impl".into()),
            HashMap::new(),
        );
        let _ = f.call(&[LowLevelValue::Bool(true)]);
    }

    #[test]
    #[should_panic(expected = "undefined function")]
    fn func_call_rejects_none_callable() {
        // lltype.py:1381-1383 — RuntimeError on None _callable.
        let f = _func::new(
            FuncType {
                args: vec![],
                result: LowLevelType::Signed,
            },
            "f".into(),
            None,
            None,
            HashMap::new(),
        );
        let _ = f.call(&[]);
    }

    #[test]
    fn func_call_accepts_matching_args() {
        let f = _func::new(
            FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Bool,
            },
            "f".into(),
            None,
            Some("impl".into()),
            HashMap::new(),
        );
        let result = f.call(&[LowLevelValue::Signed(42)]);
        assert_eq!(result, LowLevelValue::Bool(false));
    }

    #[test]
    fn ptr_fixedlength_rejects_non_array_ptr() {
        // lltype.py:1331-1336 — _fixedlength calls len() first, which
        // raises TypeError for non-array pointers. Rust port surfaces
        // this as Err(String).
        let struct_ptr = Ptr {
            TO: PtrTarget::Struct(StructType::new(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        }
        ._example();
        let err = struct_ptr
            ._fixedlength()
            .expect_err("_fixedlength on struct ptr must error");
        assert!(err.contains("no length"));
    }

    #[test]
    fn interior_ptr_type_with_index_is_gcstruct_with_interior_hint() {
        // lltype.py:769-778 — result is GcStruct, not raw Struct, and
        // carries the interior_ptr_type hint.
        let gc_parent_ptr = Ptr {
            TO: PtrTarget::Array(ArrayType::gc(LowLevelType::Struct(Box::new(
                StructType::new("Item", vec![("x".into(), LowLevelType::Signed)]),
            )))),
        };
        let interior_struct =
            gc_parent_ptr._interior_ptr_type_with_index(&LowLevelType::Struct(Box::new(
                StructType::new("Item", vec![("x".into(), LowLevelType::Signed)]),
            )));
        assert_eq!(
            interior_struct._gckind,
            GcKind::Gc,
            "interior_ptr_type_with_index must produce GcStruct"
        );
        assert!(
            interior_struct
                ._hints
                .get("interior_ptr_type")
                .is_some_and(|v| matches!(v, ConstValue::Bool(true))),
            "interior_ptr_type hint must be present"
        );
        assert_eq!(
            interior_struct._names,
            vec!["ptr".to_string(), "index".to_string()]
        );
    }

    #[test]
    #[should_panic(expected = "requires gc parent")]
    fn interior_ptr_type_with_index_rejects_raw_parent() {
        // lltype.py:770 — `assert self.TO._gckind == 'gc'`.
        let raw_ptr = Ptr {
            TO: PtrTarget::Struct(StructType::new(
                "S",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        };
        let _ = raw_ptr._interior_ptr_type_with_index(&LowLevelType::Signed);
    }

    #[test]
    fn interior_ptr_type_with_index_copies_struct_adtmeths() {
        // lltype.py:771-774 — when TO is a Struct, _adtmeths propagates.
        let meth_name = "adt_probe".to_string();
        let adtmeths = vec![(meth_name.clone(), ConstValue::Bool(true))];
        let to_struct = StructType::with_adtmeths(
            "Item",
            vec![("x".into(), LowLevelType::Signed)],
            adtmeths.clone(),
        );
        let gc_parent_ptr = Ptr {
            TO: PtrTarget::Array(ArrayType::gc(LowLevelType::Struct(Box::new(
                to_struct.clone(),
            )))),
        };
        let interior =
            gc_parent_ptr._interior_ptr_type_with_index(&LowLevelType::Struct(Box::new(to_struct)));
        assert_eq!(interior._adtmeths, FrozenDict::new(adtmeths));
    }
}
