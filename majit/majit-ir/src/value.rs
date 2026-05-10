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

    /// Returns the OpRef referencing this input arg's slot.
    ///
    /// RPython's `InputArg*` Box object IS its own reference — there is
    /// no separate handle/slot distinction. Pyre encodes the inputarg
    /// position into the low bits of an `OpRef`; this helper centralises
    /// that construction so call sites do not reach for the raw `.index`
    /// field directly.
    pub fn opref(&self) -> crate::resoperation::OpRef {
        crate::resoperation::OpRef::input_arg_typed(self.index, self.tp)
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
///
/// `warmspot.py:663 jd._green_args_spec = [TYPE for ARG, TYPE in
/// jd._JIT_ENTER_FUNCTYPE.ARGS[:jd.num_green_args]]` carries the full
/// lltype TYPE per green — including `Ptr(rstr.STR)` / `Ptr(rstr.UNICODE)`.
/// The IR-level [`Type`] (i/r/f/v) collapses those to `Ref`, so greens
/// also carry an optional [`GreenType`] preserving the STR/UNICODE
/// subtype.  Reds keep `green_type = None` (no equivalent of
/// `equal_whatever` for runtime args).
#[derive(Clone, Debug, PartialEq)]
pub struct JitDriverVar {
    /// Variable name (e.g., "pc", "stack", "sp").
    pub name: String,
    /// Type of this variable.
    pub tp: Type,
    /// Whether this is a green (constant) or red (runtime) variable.
    pub kind: VarKind,
    /// Per-green lltype subtype (`STR` / `UNICODE` / canonical
    /// `Int`/`Float`/`Ref`/`Void`).  `None` for reds and for greens
    /// constructed via [`Self::green`] without an explicit subtype —
    /// the canonical fall-back is `GreenType::from(self.tp)`.
    pub green_type: Option<GreenType>,
}

impl JitDriverVar {
    /// Construct a green var with the canonical [`GreenType`] derived
    /// from `tp` (Int/Ref/Float/Void).  Use [`Self::green_with_type`]
    /// when the upstream lltype is `Ptr(rstr.STR)` / `Ptr(rstr.UNICODE)`
    /// so `green_args_spec` reports the correct
    /// `equal_whatever`/`hash_whatever` dispatch type.
    pub fn green(name: impl Into<String>, tp: Type) -> Self {
        JitDriverVar {
            name: name.into(),
            tp,
            kind: VarKind::Green,
            green_type: None,
        }
    }

    /// Construct a green var with an explicit [`GreenType`] subtype —
    /// `warmspot.py:663` parity for STR/UNICODE Ptr greens that
    /// `Type::Ref` cannot distinguish on its own.
    pub fn green_with_type(name: impl Into<String>, tp: Type, green_type: GreenType) -> Self {
        JitDriverVar {
            name: name.into(),
            tp,
            kind: VarKind::Green,
            green_type: Some(green_type),
        }
    }

    pub fn red(name: impl Into<String>, tp: Type) -> Self {
        JitDriverVar {
            name: name.into(),
            tp,
            kind: VarKind::Red,
            green_type: None,
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

/// Project a [`GreenType`] back onto the IR-level [`Type`].  STR /
/// UNICODE collapse to `Type::Ref` (their lltype is `lltype.Ptr` —
/// `warmstate.py:109` `isinstance(TYPE, lltype.Ptr)` precondition).
pub fn green_type_to_ir(g: GreenType) -> Type {
    match g {
        GreenType::Int => Type::Int,
        GreenType::Float => Type::Float,
        GreenType::Void => Type::Void,
        GreenType::Ref | GreenType::Str | GreenType::Unicode => Type::Ref,
    }
}

/// Resolver function pointer types for `GreenType::Str` /
/// `GreenType::Unicode` content equality and hashing.
///
/// Pluggable because each frontend has its own `rstr.STR` / `rstr.UNICODE`
/// layout — RPython holds gc-typed pointers, pyre holds `*const &'static str`
/// cast through `usize` (per-occurrence `Box::leak`).  The IR layer keeps
/// no assumption about that ABI; frontends register their decoders via
/// [`set_str_resolver`] / [`set_unicode_resolver`] at startup, mirroring
/// RPython's `warmstate.py:108-128` indirection through
/// `rstr.LLHelpers.ll_streq` / `ll_strhash` (which is itself a function
/// pointer the frontend produces for its own STR/UNICODE layout).
pub type StrEqFn = fn(i64, i64) -> bool;
pub type StrHashFn = fn(i64) -> u64;

static STR_EQ: std::sync::OnceLock<StrEqFn> = std::sync::OnceLock::new();
static STR_HASH: std::sync::OnceLock<StrHashFn> = std::sync::OnceLock::new();
static UNICODE_EQ: std::sync::OnceLock<StrEqFn> = std::sync::OnceLock::new();
static UNICODE_HASH: std::sync::OnceLock<StrHashFn> = std::sync::OnceLock::new();

/// Frontend-registered Str green resolver.  First call wins; subsequent
/// calls are silently ignored to mirror `OnceLock::set`'s init-once
/// contract.  Frontends register at JitDriver startup before any
/// trace records a Str green.
pub fn set_str_resolver(eq: StrEqFn, hash: StrHashFn) {
    let _ = STR_EQ.set(eq);
    let _ = STR_HASH.set(hash);
}

/// Frontend-registered Unicode green resolver.  Same init-once contract
/// as [`set_str_resolver`].
pub fn set_unicode_resolver(eq: StrEqFn, hash: StrHashFn) {
    let _ = UNICODE_EQ.set(eq);
    let _ = UNICODE_HASH.set(hash);
}

/// Pyre canonical `ll_streq` analog for `GreenType::Str` / `GreenType::Unicode`.
///
/// Public so consumers can register it at startup via
/// [`set_str_resolver`] / [`set_unicode_resolver`] — pyre's macro
/// (`majit-macros::jit_interp::emit_green_repr`) emits the `*const
/// &'static str` slot ABI (per-occurrence `Box::leak` of a `Box<&'static str>`,
/// the pyre analog to RPython's GC-allocated `rstr.STR*`).  Decoding
/// the slot to compare by content is the pyre half of the contract;
/// RPython's half is in `rstr.py:604 ll_streq`.  Other frontends with
/// different STR layouts register their own resolver pair.
///
/// Stable-slot helper for STR/UNICODE green emission.
///
/// Materialises a `*const &'static str` slot at a stable address by
/// leaking a `Box<&'static str>` — pyre's per-occurrence analog of
/// RPython's GC-allocated `rstr.STR*` (`rstr.py:25-30`).  Each call
/// allocates a fresh slot; the GreenKey HashMap content-de-dupes via
/// `default_str_eq` / `default_str_hash` so semantically every
/// merge-point hit collapses to a single cache entry, but the leaked
/// slots themselves grow unboundedly with hit count for long-running
/// programs.
///
/// This shape is a Rust-side adaptation: `&str` literals don't have
/// stable backing-storage addresses by default, and RPython's
/// `rstr.STR*` keeps the data + len at a GC-allocated address that is
/// stable for the JitCell's lifetime.  RPython does NOT maintain a
/// global string-intern side table — `JitCell.greenargs[i]` holds the
/// rstr pointer and the GC frees it when the cell dies.  An earlier
/// pyre revision added a process-global `HashMap<Box<str>, usize>`
/// here to bound the leak, but a global intern is non-orthodox: it
/// has no RPython precedent and never frees, so the bound it provides
/// only delays the unbounded growth (distinct STR/UNICODE contents
/// still grow without bound over a process lifetime).  The structural
/// fix — reshape `GreenKey::values` from `Vec<i64>` to a typed enum
/// carrying `Box<str>` for str/unicode greens, with the macro
/// emitting a temporary that the JitCell cache promotes on insertion
/// — is a multi-session refactor and is intentionally deferred.
/// Functional behavior matches RPython (content-keyed compare/hash);
/// only the lifetime / allocation profile differs.
pub fn make_str_slot(s: &str) -> i64 {
    let owned: Box<str> = Box::from(s);
    let owned_static: &'static str = Box::leak(owned);
    let slot: &'static &'static str = Box::leak(Box::new(owned_static));
    slot as *const &'static str as usize as i64
}

pub fn default_str_eq(a: i64, b: i64) -> bool {
    if a == b {
        // `if s1 == s2: return True` (`rstr.py:604`) — handles both
        // pointer-equal and `(0, 0)` (both null).
        return true;
    }
    if a == 0 || b == 0 {
        // `if not s1 or not s2: return False` (`rstr.py:606`) — exactly
        // one side is null; null vs non-null can never match.  Without
        // this guard the deref below would dereference a null pointer.
        return false;
    }
    // SAFETY: pyre canonical green ABI — both `i64`s are non-null
    // `*const &'static str` pointing to leaked slots with `'static`
    // lifetime per `emit_green_repr`'s `Box::leak` path.
    let (sa, sb) = unsafe {
        (
            *(a as usize as *const &'static str),
            *(b as usize as *const &'static str),
        )
    };
    sa == sb
}

/// `rpython/rlib/objectmodel.py:596 _hash_string` parity over a byte
/// stream — the modified Fowler-Noll-Vo (FNV) variant that
/// CPython 2.7 (and RPython by inheritance) uses for string hashes:
///
/// ```text
///     length = len(s)
///     if length == 0: return -1
///     x = ord(s[0]) << 7
///     for i in 0..length: x = intmask((1000003 * x) ^ ord(s[i]))
///     x ^= length
///     return intmask(x)
/// ```
///
/// `intmask` truncates to the machine word; on a 64-bit target this
/// matches `i64::wrapping_mul` / `i64::wrapping_xor` exactly, so the
/// value is bit-identical to a 64-bit RPython build.
fn rpython_hash_bytes(bytes: &[u8]) -> i64 {
    let length = bytes.len();
    if length == 0 {
        return -1;
    }
    let mut x: i64 = (bytes[0] as i64) << 7;
    for &b in bytes {
        x = 1000003i64.wrapping_mul(x) ^ (b as i64);
    }
    x ^= length as i64;
    x
}

/// `rpython/rlib/objectmodel.py:596 _hash_string` parity over a
/// codepoint stream — companion to [`rpython_hash_bytes`] for
/// `rstr.UNICODE` (whose `chars` field is an array of codepoint
/// values, not bytes).  Iterates the `&str` codepoint-by-codepoint
/// so multi-byte UTF-8 sequences hash by their decoded value rather
/// than the byte representation.
fn rpython_hash_codepoints(s: &str) -> i64 {
    let length = s.chars().count();
    if length == 0 {
        return -1;
    }
    let first = s.chars().next().unwrap() as i64;
    let mut x: i64 = first << 7;
    for c in s.chars() {
        x = 1000003i64.wrapping_mul(x) ^ (c as i64);
    }
    x ^= length as i64;
    x
}

/// `rpython/rtyper/lltypesystem/rstr.py:405 _ll_strhash` zero-substitute.
///
/// RPython treats `0` as the "hash not yet computed" sentinel for
/// `rstr.STR.hash` / `rstr.UNICODE.hash` and substitutes a fixed
/// non-zero replacement (`29872897`) when the FNV result is zero.
/// pyre mirrors that contract for both `Str` and `Unicode` greens —
/// stable bucket assignment requires a deterministic non-zero hash.
fn rpython_zero_substitute(x: i64) -> i64 {
    if x == 0 { 29872897 } else { x }
}

/// Pyre canonical `ll_strhash` analog for `GreenType::Str` (Py2 byte
/// strings).  Public so consumers can register it via
/// [`set_str_resolver`].  Decodes the slot ABI (same as
/// [`default_str_eq`]) and hashes the underlying `&str` byte-by-byte
/// through [`rpython_hash_bytes`] — `rstr.STR.chars` is a sequence of
/// single-byte values, so byte iteration matches the upstream
/// `ord(s[i])` loop bit-for-bit.
pub fn default_str_hash(a: i64) -> u64 {
    // `if not s: return 0` (`rstr.py:407`) — null slot hashes to 0,
    // matching RPython's `_ll_strhash` precondition guard.  Without
    // this the deref would dereference a null pointer for unset
    // green slots.
    if a == 0 {
        return 0;
    }
    // SAFETY: non-null canonical slot ABI as [`default_str_eq`].
    let s = unsafe { *(a as usize as *const &'static str) };
    rpython_zero_substitute(rpython_hash_bytes(s.as_bytes())) as u64
}

/// Pyre canonical `ll_strhash` analog for `GreenType::Unicode`
/// (Py2 unicode strings — codepoint-indexed).  Public so consumers
/// can register it via [`set_unicode_resolver`].  Decodes the slot
/// ABI and hashes codepoint-by-codepoint through
/// [`rpython_hash_codepoints`] — `rstr.UNICODE.chars` is a sequence
/// of decoded codepoint values, matching `&str.chars()` iteration.
pub fn default_unicode_hash(a: i64) -> u64 {
    // `if not s: return 0` (`rstr.py:407`) — null slot hashes to 0,
    // matching `_ll_strhash`'s precondition guard.
    if a == 0 {
        return 0;
    }
    // SAFETY: non-null canonical slot ABI as [`default_str_eq`].
    let s = unsafe { *(a as usize as *const &'static str) };
    rpython_zero_substitute(rpython_hash_codepoints(s)) as u64
}

/// warmstate.py:108-112 equal_whatever(TYPE, x, y)
///
/// Port of RPython's lltype dispatch:
/// - Ptr to STR / UNICODE → rstr.LLHelpers.ll_streq
/// - everything else → `x == y` (with Float using bitwise f64 equality)
///
/// STR / UNICODE indirect through frontend-registered resolvers
/// ([`set_str_resolver`] / [`set_unicode_resolver`]).  Each frontend
/// owns its own `rstr.STR` / `rstr.UNICODE` decoder; PyPy compiles a
/// type-specialised `equal_whatever(STR, ..)` whose body is the
/// frontend's `ll_streq` (`@specialize.arg(0)` at `warmstate.py:107`).
/// Pyre's runtime equivalent: a registered resolver MUST exist before
/// any STR/UNICODE green key is compared.  An unregistered call
/// panics rather than silently falling back to bitwise equality —
/// PyPy never returns `x == y` for an STR green, and a silent
/// fallback masks frontend-init bugs (e.g., calling
/// `equal_whatever(GreenType::Str, ..)` before
/// `install_jit_call_bridge` runs).
pub fn equal_whatever(tp: GreenType, x: i64, y: i64) -> bool {
    match tp {
        GreenType::Str => STR_EQ.get().expect(
            "equal_whatever(GreenType::Str, ..): no Str resolver \
                 registered — frontend must call \
                 `set_str_resolver(eq, hash)` at startup before any \
                 STR green key is compared (warmstate.py:108-111 \
                 ll_streq parity)",
        )(x, y),
        GreenType::Unicode => UNICODE_EQ.get().expect(
            "equal_whatever(GreenType::Unicode, ..): no Unicode \
                 resolver registered — frontend must call \
                 `set_unicode_resolver(eq, hash)` at startup before \
                 any UNICODE green key is compared (warmstate.py:\
                 108-111 ll_streq parity)",
        )(x, y),
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
///
/// STR / UNICODE indirect through frontend-registered resolvers
/// ([`set_str_resolver`] / [`set_unicode_resolver`]).  PyPy compiles
/// a type-specialised `hash_whatever(STR, ..)` whose body is the
/// frontend's `ll_strhash` (`@specialize.arg(0)` at `warmstate.py:114`).
/// Pyre's runtime equivalent: a registered resolver MUST exist before
/// any STR/UNICODE green key is hashed.  An unregistered call panics
/// rather than silently returning the slot-bits-as-u64 — PyPy never
/// hashes an STR green by raw pointer bits, and a silent fallback
/// masks frontend-init bugs (e.g., calling
/// `hash_whatever(GreenType::Str, ..)` before
/// `install_jit_call_bridge` runs).
pub fn hash_whatever(tp: GreenType, value: i64) -> u64 {
    match tp {
        GreenType::Str => STR_HASH.get().expect(
            "hash_whatever(GreenType::Str, ..): no Str resolver \
                 registered — frontend must call \
                 `set_str_resolver(eq, hash)` at startup before any \
                 STR green key is hashed (warmstate.py:115-121 \
                 ll_strhash parity)",
        )(value),
        GreenType::Unicode => UNICODE_HASH.get().expect(
            "hash_whatever(GreenType::Unicode, ..): no Unicode \
                 resolver registered — frontend must call \
                 `set_unicode_resolver(eq, hash)` at startup before \
                 any UNICODE green key is hashed (warmstate.py:\
                 115-121 ll_strhash parity)",
        )(value),
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

/// Macro-emitted bridge used by `#[jit_interp]` to build a typed
/// `GreenKey` from heterogeneous green expressions.
///
/// Returns the `(i64 bit-representation, GreenType)` pair so the caller
/// emits both vectors in lockstep with a single move of the green value.
/// Integer / bool greens widen as Int (`warmstate.py:566 hash_whatever`
/// equal_int / hash_int);  float greens carry their bit-pattern under
/// Float (equal_float / hash_float);  reference greens widen via raw
/// pointer bits under Ref (Ptr identity — `equal_ptr` compares the bit
/// pattern that `hash_ptr` keyed on).
pub trait GreenAsI64 {
    fn __green_repr(self) -> (i64, GreenType);
}

macro_rules! impl_green_as_i64_int {
    ($($ty:ty),*) => {
        $(
            impl GreenAsI64 for $ty {
                #[inline(always)]
                fn __green_repr(self) -> (i64, GreenType) {
                    (self as i64, GreenType::Int)
                }
            }
        )*
    };
}

impl_green_as_i64_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, bool);

impl GreenAsI64 for f32 {
    #[inline(always)]
    fn __green_repr(self) -> (i64, GreenType) {
        // Promote f32 → f64 before extracting the i64 bit pattern.
        // [`equal_whatever`] / [`hash_whatever`] interpret the stored
        // bits as an `f64` (`value.rs:378 f64::from_bits(x as u64)`);
        // storing the bare `f32::to_bits()` (a u32 in the low 32 bits
        // of an i64) would round-trip to a subnormal f64 instead of
        // the float value the caller intended.  PyPy / RPython's
        // Float type is always f64 (`lltype.Float`); promoting f32
        // matches that single-width contract.
        ((self as f64).to_bits() as i64, GreenType::Float)
    }
}

impl GreenAsI64 for f64 {
    #[inline(always)]
    fn __green_repr(self) -> (i64, GreenType) {
        (self.to_bits() as i64, GreenType::Float)
    }
}

impl<T: ?Sized> GreenAsI64 for &T {
    #[inline(always)]
    fn __green_repr(self) -> (i64, GreenType) {
        (
            self as *const T as *const () as usize as i64,
            GreenType::Ref,
        )
    }
}

impl<T: ?Sized> GreenAsI64 for &mut T {
    #[inline(always)]
    fn __green_repr(self) -> (i64, GreenType) {
        (
            self as *const T as *const () as usize as i64,
            GreenType::Ref,
        )
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
    fn green_repr_returns_per_type_green_type() {
        assert_eq!(7i64.__green_repr(), (7i64, GreenType::Int));
        assert_eq!(7u32.__green_repr(), (7i64, GreenType::Int));
        assert_eq!(true.__green_repr(), (1i64, GreenType::Int));

        let (fv, ft) = 3.14f64.__green_repr();
        assert_eq!(fv, 3.14f64.to_bits() as i64);
        assert_eq!(ft, GreenType::Float);

        let s: &'static str = "abc";
        let (rv, rt) = s.__green_repr();
        assert_eq!(rv, s as *const str as *const () as usize as i64);
        assert_eq!(rt, GreenType::Ref);
    }

    #[test]
    fn green_repr_f32_promotes_to_f64_for_consistent_hash() {
        // `equal_whatever(Float, x, y)` interprets bits as `f64`
        // (`value.rs:378`).  An f32 green that stored only its 32-bit
        // pattern would round-trip to a subnormal f64; promoting to
        // f64 first keeps the i64 representation consistent with the
        // f64 green carrying the same numeric value.
        let f32_val: f32 = 1.5;
        let f64_val: f64 = 1.5;
        let (f32_bits, _) = f32_val.__green_repr();
        let (f64_bits, _) = f64_val.__green_repr();
        assert_eq!(f32_bits, f64_bits);

        // Round-trip via `equal_whatever(Float, _, _)` confirms both
        // evaluate to the same f64 value (1.5 == 1.5).
        assert!(equal_whatever(GreenType::Float, f32_bits, f64_bits));
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

    #[test]
    fn test_green_type_from_type() {
        assert_eq!(GreenType::from(Type::Int), GreenType::Int);
        assert_eq!(GreenType::from(Type::Ref), GreenType::Ref);
        assert_eq!(GreenType::from(Type::Float), GreenType::Float);
        assert_eq!(GreenType::from(Type::Void), GreenType::Void);
    }
}
