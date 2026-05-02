/// Value types and constants for the JIT IR.
///
/// Translated from rpython/jit/metainterp/history.py.
use serde::{Deserialize, Serialize};

/// The type of a value in the JIT IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Type {
    /// Machine-word signed integer (i64).
    Int,
    /// GC-managed reference (pointer).
    Ref,
    /// IEEE 754 double-precision float.
    Float,
    /// No value (void).
    Void,
}

impl Type {
    pub fn from_char(c: char) -> Self {
        match c {
            'i' => Type::Int,
            'r' | 'p' => Type::Ref,
            'f' => Type::Float,
            'v' | 'n' => Type::Void,
            _ => panic!("unknown type char: {c}"),
        }
    }

    pub fn to_char(self) -> char {
        match self {
            Type::Int => 'i',
            Type::Ref => 'r',
            Type::Float => 'f',
            Type::Void => 'v',
        }
    }
}

/// An opaque GC-managed reference.
///
/// In the actual runtime this wraps a pointer to a GC-managed object.
/// During tracing/optimization it may be a tagged value.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GcRef(pub usize);

impl GcRef {
    pub const NULL: GcRef = GcRef(0);

    pub fn is_null(self) -> bool {
        self.0 == 0
    }

    pub fn as_usize(self) -> usize {
        self.0
    }
}

/// A concrete runtime value.
///
/// PartialEq, Eq, and Hash all use f64::to_bits() for Float values,
/// matching RPython history.py:282-294 where ConstFloat._get_hash_()
/// and same_constant() are both bitwise: 0.0 ≠ -0.0, NaN == NaN
/// (same bits).
#[derive(Clone, Copy, Debug)]
pub enum Value {
    Int(i64),
    Float(f64),
    Ref(GcRef),
    Void,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            // history.py:292: longlong.extract_bits(self.value) == longlong.extract_bits(other.value)
            (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
            (Value::Ref(a), Value::Ref(b)) => a.0 == b.0,
            (Value::Void, Value::Void) => true,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Value::Int(v) => v.hash(state),
            // history.py:283: longlong.gethash(self.value) — bitwise
            Value::Float(v) => v.to_bits().hash(state),
            Value::Ref(r) => r.0.hash(state),
            Value::Void => {}
        }
    }
}

impl Value {
    pub fn get_type(&self) -> Type {
        match self {
            Value::Int(_) => Type::Int,
            Value::Float(_) => Type::Float,
            Value::Ref(_) => Type::Ref,
            Value::Void => Type::Void,
        }
    }

    pub fn as_int(&self) -> i64 {
        match self {
            Value::Int(v) => *v,
            _ => panic!("expected Int, got {:?}", self),
        }
    }

    pub fn as_float(&self) -> f64 {
        match self {
            Value::Float(v) => *v,
            _ => panic!("expected Float, got {:?}", self),
        }
    }

    pub fn as_ref(&self) -> GcRef {
        match self {
            Value::Ref(v) => *v,
            _ => panic!("expected Ref, got {:?}", self),
        }
    }
}

/// A constant value known at trace time.
///
/// Mirrors rpython/jit/metainterp/resoperation.py Const* classes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Const {
    Int(i64),
    Float(f64),
    Ref(GcRef),
}

impl Const {
    pub fn get_type(&self) -> Type {
        match self {
            Const::Int(_) => Type::Int,
            Const::Float(_) => Type::Float,
            Const::Ref(_) => Type::Ref,
        }
    }

    pub fn to_value(self) -> Value {
        match self {
            Const::Int(v) => Value::Int(v),
            Const::Float(v) => Value::Float(v),
            Const::Ref(v) => Value::Ref(v),
        }
    }

    /// history.py:225 ConstInt.getint — unsigned/signed integer value.
    /// Method belongs to ConstInt upstream; Rust single-enum collapses
    /// all variants, so assert variant at call time instead of
    /// silently cross-casting.
    pub fn getint(&self) -> i64 {
        match self {
            Const::Int(v) => *v,
            other => panic!("Const::getint on non-Int variant: {other:?}"),
        }
    }

    /// history.py:316 ConstPtr.getref_base — raw GC pointer value.
    pub fn getref_base(&self) -> GcRef {
        match self {
            Const::Ref(v) => *v,
            other => panic!("Const::getref_base on non-Ref variant: {other:?}"),
        }
    }

    /// history.py:265 ConstFloat.getfloatstorage — i64 bit-pattern.
    pub fn getfloatstorage(&self) -> i64 {
        match self {
            Const::Float(v) => v.to_bits() as i64,
            other => panic!("Const::getfloatstorage on non-Float variant: {other:?}"),
        }
    }

    /// Raw i64 projection. For Int it's the value, for Ref the pointer bits,
    /// for Float the bit-pattern of the `f64`. Matches the encoded
    /// `rd_consts[idx].0` layout.
    pub fn as_raw_i64(&self) -> i64 {
        match self {
            Const::Int(v) => *v,
            Const::Ref(GcRef(v)) => *v as i64,
            Const::Float(v) => v.to_bits() as i64,
        }
    }

    /// Reconstruct a Const from the encoded `(raw_i64, Type)` pair.
    pub fn from_raw_i64(raw: i64, tp: Type) -> Self {
        match tp {
            Type::Int => Const::Int(raw),
            Type::Ref => Const::Ref(GcRef(raw as usize)),
            Type::Float => Const::Float(f64::from_bits(raw as u64)),
            Type::Void => Const::Int(raw),
        }
    }
}

/// An input argument to a loop or bridge.
///
/// Mirrors rpython/jit/metainterp/resoperation.py InputArg* classes.
#[derive(Clone, Debug, PartialEq)]
pub struct InputArg {
    pub tp: Type,
    /// Index in the inputargs list.
    pub index: u32,
}

impl InputArg {
    pub fn new_int(index: u32) -> Self {
        InputArg {
            tp: Type::Int,
            index,
        }
    }

    pub fn new_ref(index: u32) -> Self {
        InputArg {
            tp: Type::Ref,
            index,
        }
    }

    pub fn new_float(index: u32) -> Self {
        InputArg {
            tp: Type::Float,
            index,
        }
    }

    pub fn from_type(tp: Type, index: u32) -> Self {
        InputArg { tp, index }
    }
}

/// Limit on the number of fail arguments per guard.
///
/// From history.py: FAILARGS_LIMIT = 1000
pub const FAILARGS_LIMIT: usize = 1000;

/// Classification of a variable as green (compile-time constant) or red (runtime).
///
/// Mirrors RPython's `JitDriver(greens=[...], reds=[...])` distinction:
/// - **Green** variables identify the program point (loop header).
///   They are fixed for a given compiled trace and encoded as constants in the IR.
/// - **Red** variables carry runtime state across loop iterations.
///   They become the trace's `InputArg`s.
///
/// During tracing, `promote` (GUARD_VALUE) converts a red value to green
/// by asserting it equals a specific constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VarKind {
    /// Compile-time constant: contributes to the green key identity.
    /// Changing a green variable means a different loop / compilation unit.
    Green,
    /// Runtime variable: carried as InputArg in the trace.
    Red,
}

/// A descriptor for a JitDriver variable (green or red).
///
/// Mirrors RPython's JitDriver parameter lists. Each variable has
/// a name (for debugging), a type, and a kind (green/red).
#[derive(Clone, Debug, PartialEq)]
pub struct JitDriverVar {
    /// Variable name (e.g., "pc", "stack", "sp").
    pub name: String,
    /// Type of this variable.
    pub tp: Type,
    /// Whether this is a green (constant) or red (runtime) variable.
    pub kind: VarKind,
}

impl JitDriverVar {
    pub fn green(name: impl Into<String>, tp: Type) -> Self {
        JitDriverVar {
            name: name.into(),
            tp,
            kind: VarKind::Green,
        }
    }

    pub fn red(name: impl Into<String>, tp: Type) -> Self {
        JitDriverVar {
            name: name.into(),
            tp,
            kind: VarKind::Red,
        }
    }
}

/// warmstate.py:108-112 equal_whatever / :115-128 hash_whatever take a
/// TYPE parameter that can be primitive, generic Ptr, or specifically a
/// Ptr to rstr.STR / rstr.UNICODE. The IR-level [`Type`] only carries the
/// kind (i/r/f/v); this enum extends it with the STR/UNICODE subtypes so
/// green key comparisons match RPython 1:1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GreenType {
    /// lltype.Signed / Unsigned / Bool / Char / primitive.
    Int,
    /// lltype.Float.
    Float,
    /// lltype.Void.
    Void,
    /// Generic GC Ptr — identityhash / pointer equality.
    Ref,
    /// Ptr to rstr.STR — ll_streq / ll_strhash.
    Str,
    /// Ptr to rstr.UNICODE — ll_streq / ll_strhash.
    Unicode,
}

impl From<Type> for GreenType {
    fn from(t: Type) -> Self {
        match t {
            Type::Int => GreenType::Int,
            Type::Ref => GreenType::Ref,
            Type::Float => GreenType::Float,
            Type::Void => GreenType::Void,
        }
    }
}

/// Resolver types for ll_streq / ll_strhash — pluggable because each
/// frontend (pyre, RPython) has its own rstr.STR / UNICODE layout. The
/// helper remains pure at the majit-ir layer; frontends register their
/// resolvers at startup via [`set_str_resolver`] / [`set_unicode_resolver`].
pub type StrEqFn = fn(i64, i64) -> bool;
pub type StrHashFn = fn(i64) -> u64;

static STR_EQ: std::sync::OnceLock<StrEqFn> = std::sync::OnceLock::new();
static STR_HASH: std::sync::OnceLock<StrHashFn> = std::sync::OnceLock::new();
static UNICODE_EQ: std::sync::OnceLock<StrEqFn> = std::sync::OnceLock::new();
static UNICODE_HASH: std::sync::OnceLock<StrHashFn> = std::sync::OnceLock::new();

pub fn set_str_resolver(eq: StrEqFn, hash: StrHashFn) {
    let _ = STR_EQ.set(eq);
    let _ = STR_HASH.set(hash);
}

pub fn set_unicode_resolver(eq: StrEqFn, hash: StrHashFn) {
    let _ = UNICODE_EQ.set(eq);
    let _ = UNICODE_HASH.set(hash);
}

/// warmstate.py:108-112 equal_whatever(TYPE, x, y)
///
/// Port of RPython's lltype dispatch:
/// - Ptr to STR / UNICODE → rstr.LLHelpers.ll_streq
/// - everything else → `x == y` (with Float using bitwise f64 equality)
pub fn equal_whatever(tp: GreenType, x: i64, y: i64) -> bool {
    match tp {
        GreenType::Str => STR_EQ.get().map(|f| f(x, y)).unwrap_or_else(|| x == y),
        GreenType::Unicode => UNICODE_EQ.get().map(|f| f(x, y)).unwrap_or_else(|| x == y),
        GreenType::Float => {
            let a = f64::from_bits(x as u64);
            let b = f64::from_bits(y as u64);
            a == b
        }
        // Int, Ref, Void: x == y (integer / pointer equality)
        GreenType::Int | GreenType::Ref | GreenType::Void => x == y,
    }
}

/// warmstate.py:115-128 hash_whatever(TYPE, x)
///
/// - Ptr to STR / UNICODE → rstr.LLHelpers.ll_strhash
/// - generic GC Ptr → identityhash (or 0 for null)
/// - primitive → rffi.cast(Signed, x)
pub fn hash_whatever(tp: GreenType, value: i64) -> u64 {
    match tp {
        GreenType::Str => STR_HASH
            .get()
            .map(|f| f(value))
            .unwrap_or_else(|| value as u64),
        GreenType::Unicode => UNICODE_HASH
            .get()
            .map(|f| f(value))
            .unwrap_or_else(|| value as u64),
        GreenType::Ref => {
            // identityhash(x) or 0
            if value != 0 { value as u64 } else { 0 }
        }
        GreenType::Float => {
            // rffi.cast(Signed, x) — truncate float to integer
            let float_val = f64::from_bits(value as u64);
            (float_val as i64) as u64
        }
        // Int, Void: rffi.cast(Signed, x) — the value itself
        GreenType::Int | GreenType::Void => value as u64,
    }
}

/// Structured green key — represents the exact values and types of all
/// green variables at a particular program point.
///
/// warmstate.py:564-565 green_args_name_spec — pairs each green arg with
/// its TYPE. comparekey uses equal_whatever(TYPE, ...) and get_uhash uses
/// hash_whatever(TYPE, ...) per RPython.
#[derive(Clone, Debug, Default)]
pub struct GreenKey {
    /// Values of all green variables, in declaration order.
    pub values: Vec<i64>,
    /// warmstate.py:564 — per-entry TYPE. Drives hash_whatever/equal_whatever.
    /// `GreenType` (not IR `Type`) so `Ptr to rstr.STR/UNICODE` stays distinct
    /// from generic Ref and is dispatched through ll_streq / ll_strhash.
    pub types: Vec<GreenType>,
}

impl PartialEq for GreenKey {
    /// warmstate.py:575-582 JitCell.comparekey(*greenargs2)
    ///
    /// RPython's comparekey iterates green_args_name_spec (fixed per JitCell
    /// class), comparing each stored attr with the incoming greenarg using
    /// equal_whatever(TYPE, stored, incoming). Both the type spec and the
    /// values must match for equality.
    fn eq(&self, other: &Self) -> bool {
        if self.values.len() != other.values.len() {
            return false;
        }
        if self.types != other.types {
            return false;
        }
        for i in 0..self.values.len() {
            if !equal_whatever(self.types[i], self.values[i], other.values[i]) {
                return false;
            }
        }
        true
    }
}

impl Eq for GreenKey {}

impl std::fmt::Display for GreenKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GreenKey({:?})", self.values)
    }
}

impl std::hash::Hash for GreenKey {
    /// warmstate.py:584-593 JitCell.get_uhash(*greenargs)
    ///
    /// Delegates to get_uhash() so that HashMap<GreenKey, _> uses the same
    /// hash as jitcounter bucket lookup.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.get_uhash());
    }
}

impl GreenKey {
    /// Create an all-Int green key (most common case: PC-based keys).
    pub fn new(values: Vec<i64>) -> Self {
        let types = vec![GreenType::Int; values.len()];
        GreenKey { values, types }
    }

    /// warmstate.py:564-565 — typed green key. Accepts either IR-level
    /// [`Type`] (via `From<Type>`) or the richer [`GreenType`].
    pub fn with_types<T: Into<GreenType> + Copy>(values: Vec<i64>, types: Vec<T>) -> Self {
        debug_assert_eq!(values.len(), types.len());
        let types = types.into_iter().map(Into::into).collect();
        GreenKey { values, types }
    }

    /// Single Int green key.
    pub fn single(value: i64) -> Self {
        GreenKey {
            values: vec![value],
            types: vec![GreenType::Int],
        }
    }

    /// warmstate.py:584-593 JitCell.get_uhash(*greenargs)
    ///
    /// Exact port of RPython's hash algorithm:
    ///     x = r_uint(-1888132534)
    ///     for _, TYPE in green_args_name_spec:
    ///         y = r_uint(hash_whatever(TYPE, item))
    ///         x = (x ^ y) * r_uint(1405695061)
    ///     return x
    pub fn get_uhash(&self) -> u64 {
        let mut x: u64 = (-1888132534_i64) as u64;
        for i in 0..self.values.len() {
            let tp = self.types.get(i).copied().unwrap_or(GreenType::Int);
            let y = hash_whatever(tp, self.values[i]);
            x = (x ^ y).wrapping_mul(1405695061);
        }
        x
    }

    /// Alias for get_uhash.
    pub fn hash_u64(&self) -> u64 {
        self.get_uhash()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_roundtrip() {
        for tp in [Type::Int, Type::Ref, Type::Float, Type::Void] {
            assert_eq!(Type::from_char(tp.to_char()), tp);
        }
    }

    #[test]
    fn test_value_types() {
        assert_eq!(Value::Int(42).get_type(), Type::Int);
        assert_eq!(Value::Float(3.14).get_type(), Type::Float);
        assert_eq!(Value::Ref(GcRef::NULL).get_type(), Type::Ref);
        assert_eq!(Value::Void.get_type(), Type::Void);
    }

    #[test]
    fn test_gcref_null() {
        assert!(GcRef::NULL.is_null());
        assert!(!GcRef(0x1234).is_null());
    }

    #[test]
    fn test_var_kind() {
        let green = JitDriverVar::green("pc", Type::Int);
        assert_eq!(green.kind, VarKind::Green);
        assert_eq!(green.name, "pc");

        let red = JitDriverVar::red("stack", Type::Ref);
        assert_eq!(red.kind, VarKind::Red);
        assert_eq!(red.name, "stack");
    }

    #[test]
    fn test_green_key_hash() {
        let k1 = GreenKey::single(42);
        let k2 = GreenKey::single(42);
        let k3 = GreenKey::single(43);

        assert_eq!(k1.hash_u64(), k2.hash_u64());
        assert_ne!(k1.hash_u64(), k3.hash_u64());
    }

    #[test]
    fn test_green_key_multi() {
        let k1 = GreenKey::new(vec![10, 20, 30]);
        let k2 = GreenKey::new(vec![10, 20, 30]);
        let k3 = GreenKey::new(vec![10, 20, 31]);

        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
        assert_eq!(k1.hash_u64(), k2.hash_u64());
    }

    /// warmstate.py:108-112 — STR/UNICODE greens use ll_streq, not
    /// identity. With a resolver wired, equal_whatever should treat two
    /// distinct pointers with identical content as equal.
    #[test]
    fn test_green_type_str_dispatch_without_resolver() {
        // Without a resolver the helpers fall back to pointer identity.
        assert!(equal_whatever(GreenType::Str, 0x1000, 0x1000));
        assert!(!equal_whatever(GreenType::Str, 0x1000, 0x1001));
        assert_eq!(hash_whatever(GreenType::Str, 0x1000), 0x1000u64);
    }

    #[test]
    fn test_green_type_from_type() {
        assert_eq!(GreenType::from(Type::Int), GreenType::Int);
        assert_eq!(GreenType::from(Type::Ref), GreenType::Ref);
        assert_eq!(GreenType::from(Type::Float), GreenType::Float);
        assert_eq!(GreenType::from(Type::Void), GreenType::Void);
    }
}
