//! Epic H — Rust mirror of RPython's `AbstractValue` object identity.
//!
//! Direct port of the Python object identity hierarchy formed by
//! `rpython/jit/metainterp/resoperation.py:29 AbstractValue` together with
//! `AbstractResOpOrInputArg` / `AbstractResOp` / `AbstractInputArg` /
//! `Const*` (`history.py:182`), expressed as `Rc<Box>`.
//!
//! Callers are introduced from H-2 onward. The H-1 commit is type-only —
//! it coexists with the existing `OpRef(u32)` code and is a functional
//! no-op with zero callers.
//!
//! # Design decisions
//!
//! - The `forwarded` slot is a `RefCell<Forwarded>`. `Cell` is not used
//!   because `Forwarded` carries `OpInfo` / `BoxRef`, neither of which is
//!   `Copy`. Helpers terminate the borrow scope immediately after reading.
//! - `BoxRef`'s `Eq` / `Hash` use `Rc::ptr_eq` / `Rc::as_ptr` — equivalent
//!   to RPython's use of object identity as a dict key.
//! - When `Forwarded::Box(BoxRef)` carries a BoxRef whose kind is
//!   `BoxKind::Const(...)`, that mirrors RPython's
//!   `box.set_forwarded(constbox)`. We do not introduce a separate `Const`
//!   variant: RPython stores everything in a single `_forwarded` slot.

use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use majit_ir::{Type, Value};

use crate::optimizeopt::info::OpInfo;

/// `AbstractValue` mirror — unified representation of RPython's
/// op/inputarg/const objects.
pub struct Box {
    /// `resoperation.py:233-243 AbstractResOpOrInputArg._forwarded`.
    ///
    /// Const boxes also carry the slot, but in RPython `Const` is not a
    /// subclass of `AbstractResOpOrInputArg`, so its `_forwarded` is
    /// always `None`. Rust unifies the layout into a single struct
    /// shape while preserving the same invariant.
    pub forwarded: RefCell<Forwarded>,

    /// `resoperation.py:260 type` (`'i'` / `'r'` / `'f'` / `'v'`).
    /// Absorbs the frontend semantic portion that majit currently spreads
    /// across `value_types` / `inputarg_types` /
    /// `constant_types_for_numbering`.
    pub type_: Type,

    /// Rust enum mirror of RPython's subclass hierarchy.
    pub kind: BoxKind,
}

/// Enum mirror of the PyPy class hierarchy.
pub enum BoxKind {
    /// `resoperation.py:250 AbstractResOp` — the operation object itself
    /// is the result identity. `position` is a pyre-only field with no
    /// RPython counterpart on `AbstractResOpOrInputArg` (which carries
    /// only `_forwarded`); it stores the index where the box lands in the
    /// pool so the chain walker can reconstruct an `OpRef::op_typed(pos,
    /// type)` when advancing through `Forwarded::Box(target)`. Set by
    /// `opencoder.py Trace.record()` (and the recorder's `record_op_*`
    /// family in pyre) at construction; updated by `optimizer.rs:2783`
    /// const-pool compaction via `BoxRef::set_position`. RPython has no
    /// equivalent on `AbstractResOp` itself — it threads
    /// `position_and_flags` through `FrontendOp` (resoperation.py:233).
    ResOp { position: std::cell::Cell<u32> },

    /// `resoperation.py:699 AbstractInputArg`.
    /// `position` mirrors `AbstractInputArg.position`
    /// (resoperation.py:699) — `Optional` because pyre constructs
    /// inputargs without an assigned slot in some test fixtures.
    InputArg { position: Option<u32> },

    /// `history.py:220 ConstInt` / `:261 ConstFloat` / `:307 ConstPtr`.
    /// `const_index` is a pyre-only field carrying the
    /// `OpRef::Const{Int,Float,Ptr}.const_index()` so the chain walker
    /// can reconstruct a constant-namespace OpRef when advancing past
    /// `Forwarded::Box(const_box)` written by `replace_op(_, const_target)`.
    /// `None` for `BoxRef::new_const(value)` (no index in scope —
    /// `make_constant` used to pick this path before slice 18 split it
    /// off; remaining `None` callers are test fixtures).
    Const {
        value: Value,
        const_index: Option<u32>,
    },
}

/// Variant of the `_forwarded` slot.
///
/// RPython's `_forwarded` is `None | another AbstractResOpOrInputArg |
/// AbstractInfo`. Const forwarding is one case of "another box", so we
/// represent it as `Box(BoxRef)` carrying a `BoxKind::Const(...)`.
#[derive(Debug)]
pub enum Forwarded {
    None,

    /// Forwarding to another `AbstractResOpOrInputArg` or `Const`.
    Box(BoxRef),

    /// `optimizeopt/info.py:17 AbstractInfo (is_info_class = True)` family —
    /// `PtrInfo`, `IntBound`, `FloatConstInfo`, `EmptyInfo`, etc. The
    /// vector optimizer's `VectorizationInfo` also fits inside this
    /// variant.
    Info(OpInfo),
}

/// `Rc<Box>` newtype.
///
/// `Eq` / `Hash` are pointer identity for ALL variants — mirrors PyPy's
/// `is`-based default `__eq__` on `AbstractValue` (covers `ResOp` /
/// `InputArg` / `Const`). PyPy's value-based comparison for constants is
/// the explicit `same_box` / `same_constant` method (`history.py:204`),
/// not `__eq__`. Callers that need value comparison on constants must use
/// `BoxRef::same_constant`.
pub struct BoxRef(Rc<Box>);

impl BoxRef {
    /// New `AbstractResOp` Box.
    ///
    /// `position` is a pyre-only field with no RPython counterpart on
    /// `AbstractResOpOrInputArg` (which carries only `_forwarded`).
    /// Stores the index this box occupies in the pool (== raw value of
    /// the matching OpRef). Construction-time assignment matches PyPy's
    /// `Trace.record()`.
    pub fn new_resop(type_: Type, position: u32) -> Self {
        Self(Rc::new(Box {
            forwarded: RefCell::new(Forwarded::None),
            type_,
            kind: BoxKind::ResOp {
                position: std::cell::Cell::new(position),
            },
        }))
    }

    /// New `AbstractInputArg` Box.
    pub fn new_inputarg(type_: Type, position: Option<u32>) -> Self {
        Self(Rc::new(Box {
            forwarded: RefCell::new(Forwarded::None),
            type_,
            kind: BoxKind::InputArg { position },
        }))
    }

    /// New `Const*` Box. `type_` is inferred from `value`.
    /// No `const_index` — used by callers without a const-namespace
    /// OpRef in scope (test fixtures, default constructors).
    pub fn new_const(value: Value) -> Self {
        let type_ = value.get_type();
        Self(Rc::new(Box {
            forwarded: RefCell::new(Forwarded::None),
            type_,
            kind: BoxKind::Const {
                value,
                const_index: None,
            },
        }))
    }

    /// New `Const*` Box carrying a `const_index`. Used by
    /// `replace_op(_, const_target)` so the chain walker can reconstruct
    /// `OpRef::Const{Int,Float,Ptr}(const_index)` when advancing past
    /// `Forwarded::Box(const_box)`.
    pub fn new_const_with_index(value: Value, const_index: u32) -> Self {
        let type_ = value.get_type();
        Self(Rc::new(Box {
            forwarded: RefCell::new(Forwarded::None),
            type_,
            kind: BoxKind::Const {
                value,
                const_index: Some(const_index),
            },
        }))
    }

    /// Extract the `const_index` field for chain-walker reconstruction.
    /// Returns `None` for non-Const boxes and for Consts created via
    /// `new_const` (no index in scope).
    pub fn const_index(&self) -> Option<u32> {
        match &self.0.kind {
            BoxKind::Const { const_index, .. } => *const_index,
            _ => None,
        }
    }

    pub fn type_(&self) -> Type {
        self.0.type_
    }

    /// `resoperation.py:47 is_constant`.
    pub fn is_constant(&self) -> bool {
        matches!(self.0.kind, BoxKind::Const { .. })
    }

    pub fn is_inputarg(&self) -> bool {
        matches!(self.0.kind, BoxKind::InputArg { .. })
    }

    pub fn is_resop(&self) -> bool {
        matches!(self.0.kind, BoxKind::ResOp { .. })
    }

    /// `resoperation.py:233 AbstractResOpOrInputArg._pos` accessor.
    ///
    /// Returns the index where this box resides in the pool for
    /// `ResOp` / `InputArg` (which are the two PyPy classes that own
    /// `_pos`). `Const` has no canonical position and returns `None`.
    pub fn position(&self) -> Option<u32> {
        match &self.0.kind {
            BoxKind::ResOp { position } => Some(position.get()),
            BoxKind::InputArg { position } => *position,
            BoxKind::Const { .. } => None,
        }
    }

    /// Update the ResOp position field. Used by `optimizer.rs:2783`
    /// const-pool compaction to keep `BoxKind::ResOp { position }` aligned
    /// with the new dense op-position range so the chain walker's BoxRef
    /// reconstruction (`OpRef::op_typed(target.position(), tp)`) returns
    /// post-compact positions.
    ///
    /// No-op for `InputArg` / `Const` (their positions are not subject
    /// to compaction).
    pub fn set_position(&self, new_pos: u32) {
        if let BoxKind::ResOp { position } = &self.0.kind {
            position.set(new_pos);
        }
    }

    /// Extract the constant value. Mirrors `history.py:233 ConstInt.getint`
    /// and the equivalent accessors on the other Const subclasses.
    pub fn const_value(&self) -> Option<Value> {
        match self.0.kind {
            BoxKind::Const { value, .. } => Some(value),
            _ => None,
        }
    }

    /// Extract `AbstractInputArg.position`.
    pub fn inputarg_position(&self) -> Option<u32> {
        match self.0.kind {
            BoxKind::InputArg { position } => position,
            _ => None,
        }
    }

    /// `resoperation.py:50 get_forwarded`.
    pub fn get_forwarded(&self) -> Ref<'_, Forwarded> {
        self.0.forwarded.borrow()
    }

    /// `resoperation.py:53 set_forwarded(forwarded_to)` — Box variant.
    pub fn set_forwarded_box(&self, target: BoxRef) {
        // `assert forwarded_to is not self` (resoperation.py:241).
        debug_assert!(!Rc::ptr_eq(&self.0, &target.0));
        // RPython AbstractValue invariant: `Const` is not a subclass of
        // `AbstractResOpOrInputArg` (history.py:182), so `set_forwarded`
        // is undefined on Const objects (resoperation.py:50 default
        // raises). The Rust port unifies the layout into a single struct
        // shape; this assertion preserves the invariant. PyPy raises
        // unconditionally, so the check is always-on (not `debug_assert!`).
        assert!(
            !matches!(self.0.kind, BoxKind::Const { .. }),
            "set_forwarded_box on Const violates RPython AbstractValue \
             invariant (Const has no _forwarded slot)"
        );
        *self.0.forwarded.borrow_mut() = Forwarded::Box(target);
    }

    /// `resoperation.py:53 set_forwarded(forwarded_to)` — Info variant.
    pub fn set_forwarded_info(&self, info: OpInfo) {
        // PyPy `AbstractValue.set_forwarded` raises unconditionally on
        // `Const`; mirror that with an always-on assert (not
        // `debug_assert!`) so release builds preserve the invariant.
        assert!(
            !matches!(self.0.kind, BoxKind::Const { .. }),
            "set_forwarded_info on Const violates RPython AbstractValue \
             invariant (Const has no _forwarded slot)"
        );
        *self.0.forwarded.borrow_mut() = Forwarded::Info(info);
    }

    /// `_forwarded = None` (used during transition / phase reset).
    pub fn clear_forwarded(&self) {
        // Const has no _forwarded slot to reset; clearing is a no-op for
        // Const but should not be called on it. Allow clear (idempotent
        // None) for transitional safety while migration progresses.
        if matches!(self.0.kind, BoxKind::Const { .. }) {
            return;
        }
        *self.0.forwarded.borrow_mut() = Forwarded::None;
    }

    /// `resoperation.py:57-68 get_box_replacement(not_const=False)`.
    ///
    /// Walk the `_forwarded` chain, returning the box one step before the
    /// chain hits `None`, `Info`, or (`not_const=true && next.is_constant()`).
    pub fn get_box_replacement(&self, not_const: bool) -> BoxRef {
        let mut cur = self.clone();
        loop {
            // Drop the borrow scope immediately. While a
            // `Ref<'_, Forwarded>` is alive we cannot move `cur`, so we
            // snapshot the decision and release the borrow before
            // advancing.
            enum Step {
                Stop,
                Advance(BoxRef),
            }
            let step = match &*cur.0.forwarded.borrow() {
                Forwarded::None | Forwarded::Info(_) => Step::Stop,
                Forwarded::Box(b) => {
                    if not_const && b.is_constant() {
                        Step::Stop
                    } else {
                        Step::Advance(b.clone())
                    }
                }
            };
            match step {
                Step::Stop => return cur,
                Step::Advance(next) => cur = next,
            }
        }
    }

    /// `optimizer.py:99-113 getptrinfo` BoxRef-native reader.
    ///
    /// When `_forwarded` is `Info(OpInfo::Ptr(_))`, return the inner
    /// `PtrInfo` as `Ref<'_, PtrInfo>`. All other states (`None`,
    /// `Box(_)`, other `OpInfo` variants) return `None`.
    ///
    /// Does not walk the chain — the caller is responsible for advancing
    /// to the terminal BoxRef (e.g. via
    /// `OptContext::get_box_replacement_box`) before calling. This mirrors
    /// reading `box.get_forwarded()` directly in RPython.
    pub fn ptr_info(&self) -> Option<Ref<'_, crate::optimizeopt::info::PtrInfo>> {
        Ref::filter_map(self.0.forwarded.borrow(), |f| match f {
            Forwarded::Info(info) => info.get_ptr_info(),
            _ => None,
        })
        .ok()
    }

    /// `optimizer.py:99-113 getintbound` BoxRef-native reader.
    ///
    /// When `_forwarded` is `Info(OpInfo::IntBound(_))`, return the inner
    /// `IntBound` as `Ref<'_, IntBound>`. Other states return `None`.
    /// Same caller-walks-the-chain contract as `ptr_info`.
    pub fn int_bound(&self) -> Option<Ref<'_, crate::optimizeopt::intutils::IntBound>> {
        Ref::filter_map(self.0.forwarded.borrow(), |f| match f {
            Forwarded::Info(info) => info.get_int_bound(),
            _ => None,
        })
        .ok()
    }

    /// Mutable counterpart of `ptr_info`.
    ///
    /// PyPy's `optimizer.py:99-113` mutates the `PtrInfo` returned from
    /// `box.get_forwarded()` in place — Python objects are reference types,
    /// so any `info.<method>(...)` call on the returned object mutates the
    /// `_forwarded` slot's contents directly. The Rust mirror exposes that
    /// through a `RefMut<'_, PtrInfo>` that aliases the inner `RefCell`.
    ///
    /// Holds the `RefCell` borrow for the lifetime of the returned guard;
    /// callers must drop the guard before any other access to
    /// `self.0.forwarded`.
    pub fn ptr_info_mut(&self) -> Option<RefMut<'_, crate::optimizeopt::info::PtrInfo>> {
        RefMut::filter_map(self.0.forwarded.borrow_mut(), |f| match f {
            Forwarded::Info(info) => info.get_ptr_info_mut(),
            _ => None,
        })
        .ok()
    }

    /// Mutable counterpart of `int_bound`. Same contract as `ptr_info_mut`.
    pub fn int_bound_mut(&self) -> Option<RefMut<'_, crate::optimizeopt::intutils::IntBound>> {
        RefMut::filter_map(self.0.forwarded.borrow_mut(), |f| match f {
            Forwarded::Info(info) => info.get_int_bound_mut(),
            _ => None,
        })
        .ok()
    }

    /// `Rc::as_ptr` raw pointer — for debug / logging only.
    pub fn as_ptr(&self) -> *const Box {
        Rc::as_ptr(&self.0)
    }

    pub fn strong_count(&self) -> usize {
        Rc::strong_count(&self.0)
    }
}

impl Clone for BoxRef {
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}

impl PartialEq for BoxRef {
    fn eq(&self, other: &Self) -> bool {
        // PyPy `__eq__` on `AbstractValue` defaults to Python `is` for all
        // box subclasses (ResOp / InputArg / Const). Value equality on
        // constants is the explicit `same_constant` method, not `==`.
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for BoxRef {}

impl std::hash::Hash for BoxRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (Rc::as_ptr(&self.0) as usize).hash(state);
    }
}

impl std::fmt::Debug for BoxRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.0.kind {
            BoxKind::ResOp { .. } => "ResOp",
            BoxKind::InputArg { .. } => "InputArg",
            BoxKind::Const { .. } => "Const",
        };
        write!(
            f,
            "BoxRef@{:p}({:?},{})",
            Rc::as_ptr(&self.0),
            self.0.type_,
            kind
        )
    }
}

/// Encapsulated `BoxRef` storage for `OptContext` (Codex plan step 1).
///
/// Wraps `Vec<BoxRef>` as a newtype so consumers index by `OpRef` raw
/// position. `BoxRef._forwarded` is the authoritative PyPy-style storage;
/// `BoxPool` only maps pyre's flat `OpRef` indices to those Box identities.
/// `Deref` / `DerefMut` to `Vec<BoxRef>` keeps existing call sites working
/// while the wrapper remains a thin container.
#[derive(Clone, Debug, Default)]
pub struct BoxPool {
    inner: Vec<BoxRef>,
}

impl BoxPool {
    pub fn new() -> Self {
        Self::default()
    }
}

impl std::ops::Deref for BoxPool {
    type Target = Vec<BoxRef>;
    fn deref(&self) -> &Vec<BoxRef> {
        &self.inner
    }
}

impl std::ops::DerefMut for BoxPool {
    fn deref_mut(&mut self) -> &mut Vec<BoxRef> {
        &mut self.inner
    }
}

impl From<Vec<BoxRef>> for BoxPool {
    fn from(inner: Vec<BoxRef>) -> Self {
        Self { inner }
    }
}

impl From<BoxPool> for Vec<BoxRef> {
    fn from(pool: BoxPool) -> Self {
        pool.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{Type, Value};

    #[test]
    fn box_ref_identity_is_pointer_equality() {
        let a = BoxRef::new_resop(Type::Int, 0);
        let cloned = a.clone();
        let other = BoxRef::new_resop(Type::Int, 1);
        assert_eq!(a, cloned);
        assert_ne!(a, other);
    }

    #[test]
    fn forwarded_chain_walk_returns_terminal() {
        // a -> b -> c (all resop), all forwarded = None at c.
        let a = BoxRef::new_resop(Type::Int, 0);
        let b = BoxRef::new_resop(Type::Int, 1);
        let c = BoxRef::new_resop(Type::Int, 2);
        a.set_forwarded_box(b.clone());
        b.set_forwarded_box(c.clone());
        assert_eq!(a.get_box_replacement(false), c);
        assert_eq!(b.get_box_replacement(false), c);
        assert_eq!(c.get_box_replacement(false), c);
    }

    #[test]
    fn forwarded_chain_stops_at_info() {
        // a -> b, then b._forwarded = Info(...).
        let a = BoxRef::new_resop(Type::Int, 0);
        let b = BoxRef::new_resop(Type::Int, 1);
        a.set_forwarded_box(b.clone());
        b.set_forwarded_info(OpInfo::Unknown);
        // Walker reaches b, sees Info, returns b.
        assert_eq!(a.get_box_replacement(false), b);
    }

    #[test]
    fn forwarded_chain_not_const_stops_before_const() {
        // a -> b (resop) -> c (const).
        let a = BoxRef::new_resop(Type::Int, 0);
        let b = BoxRef::new_resop(Type::Int, 1);
        let c = BoxRef::new_const(Value::Int(42));
        a.set_forwarded_box(b.clone());
        b.set_forwarded_box(c.clone());

        // not_const=true: stop at b (const detection BEFORE descending).
        assert_eq!(a.get_box_replacement(true), b);
        // not_const=false: descend into const.
        assert_eq!(a.get_box_replacement(false), c);
    }

    #[test]
    fn const_box_kind_and_type() {
        let i = BoxRef::new_const(Value::Int(7));
        assert!(i.is_constant());
        assert_eq!(i.const_value(), Some(Value::Int(7)));
        assert_eq!(i.type_(), Type::Int);

        let f = BoxRef::new_const(Value::Float(1.5));
        assert!(f.is_constant());
        assert_eq!(f.type_(), Type::Float);
    }

    /// PyPy `__eq__` defaults to `is` on `AbstractValue` for ALL box
    /// subclasses (ResOp / InputArg / Const). Value comparison on constants
    /// is the explicit `same_box` / `same_constant` method
    /// (`history.py:204`), not `==`. `BoxRef::eq` therefore stays pointer
    /// identity (`Rc::ptr_eq`) across every variant.
    #[test]
    fn boxref_eq_is_pointer_identity_for_every_variant() {
        use std::collections::HashSet;

        // Const: two fresh allocations of the same value compare unequal —
        // identity (Rc pointer) differs. PyPy's value comparison is the
        // explicit `same_constant` path, not `__eq__`.
        let a = BoxRef::new_const(Value::Int(42));
        let b = BoxRef::new_const(Value::Int(42));
        assert_ne!(Rc::as_ptr(&a.0), Rc::as_ptr(&b.0));
        assert_ne!(a, b);
        // Clone preserves identity (Rc::clone shares the allocation).
        assert_eq!(a, a.clone());

        // HashSet keys by pointer identity for Const.
        let mut set: HashSet<BoxRef> = HashSet::new();
        set.insert(a.clone());
        assert!(set.contains(&a));
        assert!(!set.contains(&b));

        // ResOp / InputArg: pointer identity (RPython `is` parity).
        let r1 = BoxRef::new_resop(Type::Int, 0);
        let r2 = BoxRef::new_resop(Type::Int, 0);
        assert_ne!(r1, r2);
        assert_eq!(r1, r1.clone());
    }

    #[test]
    fn inputarg_position_preserved() {
        let arg = BoxRef::new_inputarg(Type::Ref, Some(3));
        assert!(arg.is_inputarg());
        assert_eq!(arg.inputarg_position(), Some(3));
        assert_eq!(arg.type_(), Type::Ref);
    }

    #[test]
    fn clear_forwarded_resets_slot() {
        let a = BoxRef::new_resop(Type::Int, 0);
        let b = BoxRef::new_resop(Type::Int, 0);
        a.set_forwarded_box(b.clone());
        a.clear_forwarded();
        assert_eq!(a.get_box_replacement(false), a);
        assert!(matches!(*a.get_forwarded(), Forwarded::None));
    }

    #[test]
    fn boxref_used_as_hashmap_key() {
        use std::collections::HashMap;
        let a = BoxRef::new_resop(Type::Int, 0);
        let b = BoxRef::new_resop(Type::Int, 0);
        let mut m: HashMap<BoxRef, i32> = HashMap::new();
        m.insert(a.clone(), 1);
        m.insert(b.clone(), 2);
        assert_eq!(m.get(&a), Some(&1));
        assert_eq!(m.get(&b), Some(&2));
        // Clone shares the allocation, so it hashes to the same key.
        assert_eq!(m.get(&a.clone()), Some(&1));
    }

    #[test]
    fn resop_position_round_trips() {
        // `BoxKind::ResOp { position }` mirrors `AbstractResOpOrInputArg._pos`.
        // Construction-time assignment must round-trip through `position()`.
        let r = BoxRef::new_resop(Type::Int, 42);
        assert_eq!(r.position(), Some(42));
    }

    #[test]
    fn position_returns_inputarg_position_when_set() {
        let arg = BoxRef::new_inputarg(Type::Ref, Some(7));
        assert_eq!(arg.position(), Some(7));
        let unset = BoxRef::new_inputarg(Type::Int, None);
        assert_eq!(unset.position(), None);
    }

    #[test]
    fn position_is_none_for_const() {
        let c = BoxRef::new_const(Value::Int(5));
        assert_eq!(c.position(), None);
    }

    #[test]
    #[should_panic]
    fn set_forwarded_to_self_panics_in_debug() {
        let a = BoxRef::new_resop(Type::Int, 0);
        a.set_forwarded_box(a.clone());
    }

    // H-3.2c slice 2: BoxRef-native ptr_info / int_bound readers.
    // RPython parity: optimizer.py:99-113 getptrinfo / getintbound is the
    // BoxRef-direct read path. The contract is that the caller has
    // already walked the chain to the terminal box before calling.

    #[test]
    fn ptr_info_returns_inner_when_forwarded_is_ptr_info() {
        use crate::optimizeopt::info::PtrInfo;
        let a = BoxRef::new_resop(Type::Ref, 0);
        a.set_forwarded_info(OpInfo::Ptr(PtrInfo::nonnull()));
        let pi = a.ptr_info().expect("ptr_info should return Some");
        assert!(pi.is_nonnull());
    }

    #[test]
    fn ptr_info_returns_none_for_unset_box() {
        let a = BoxRef::new_resop(Type::Ref, 0);
        assert!(a.ptr_info().is_none());
    }

    #[test]
    fn ptr_info_returns_none_for_box_forwarded() {
        // Chain walk is the caller's responsibility, so when `_forwarded`
        // is `Forwarded::Box(_)` (i.e. a box, not info), `ptr_info()` must
        // return None.
        let a = BoxRef::new_resop(Type::Ref, 0);
        let b = BoxRef::new_resop(Type::Ref, 0);
        a.set_forwarded_box(b.clone());
        assert!(a.ptr_info().is_none());
    }

    #[test]
    fn ptr_info_returns_none_for_intbound_forwarded() {
        // `_forwarded` carries OpInfo::IntBound; ptr_info() must reject it.
        use crate::optimizeopt::intutils::IntBound;
        let a = BoxRef::new_resop(Type::Int, 0);
        a.set_forwarded_info(OpInfo::IntBound(IntBound::from_constant(7)));
        assert!(a.ptr_info().is_none());
    }

    #[test]
    fn int_bound_returns_inner_when_forwarded_is_intbound() {
        use crate::optimizeopt::intutils::IntBound;
        let a = BoxRef::new_resop(Type::Int, 0);
        a.set_forwarded_info(OpInfo::IntBound(IntBound::from_constant(42)));
        let ib = a.int_bound().expect("int_bound should return Some");
        assert!(ib.is_constant());
        assert_eq!(ib.get_constant_int(), 42);
    }

    #[test]
    fn int_bound_returns_none_for_unset_box() {
        let a = BoxRef::new_resop(Type::Int, 0);
        assert!(a.int_bound().is_none());
    }

    #[test]
    fn int_bound_returns_none_for_box_forwarded() {
        let a = BoxRef::new_resop(Type::Int, 0);
        let b = BoxRef::new_resop(Type::Int, 0);
        a.set_forwarded_box(b.clone());
        assert!(a.int_bound().is_none());
    }

    #[test]
    fn int_bound_returns_none_for_ptrinfo_forwarded() {
        use crate::optimizeopt::info::PtrInfo;
        let a = BoxRef::new_resop(Type::Ref, 0);
        a.set_forwarded_info(OpInfo::Ptr(PtrInfo::nonnull()));
        assert!(a.int_bound().is_none());
    }

    #[test]
    fn ptr_info_mut_mutates_inner_in_place() {
        use crate::optimizeopt::info::PtrInfo;
        let a = BoxRef::new_resop(Type::Ref, 0);
        a.set_forwarded_info(OpInfo::Ptr(PtrInfo::nonnull()));
        {
            let mut pi = a.ptr_info_mut().expect("ptr_info_mut should return Some");
            pi.set_last_guard_pos(7);
        }
        let pi = a
            .ptr_info()
            .expect("ptr_info should return Some after mutation");
        assert_eq!(pi.last_guard_pos(), Some(7));
    }

    #[test]
    fn ptr_info_mut_returns_none_for_intbound_forwarded() {
        use crate::optimizeopt::intutils::IntBound;
        let a = BoxRef::new_resop(Type::Int, 0);
        a.set_forwarded_info(OpInfo::IntBound(IntBound::from_constant(7)));
        assert!(a.ptr_info_mut().is_none());
    }

    #[test]
    fn int_bound_mut_mutates_inner_in_place() {
        use crate::optimizeopt::intutils::IntBound;
        let a = BoxRef::new_resop(Type::Int, 0);
        a.set_forwarded_info(OpInfo::IntBound(IntBound::unbounded()));
        {
            let mut ib = a.int_bound_mut().expect("int_bound_mut should return Some");
            let _ = ib.make_eq_const(99);
        }
        let ib = a
            .int_bound()
            .expect("int_bound should return Some after mutation");
        assert!(ib.is_constant());
        assert_eq!(ib.get_constant_int(), 99);
    }

    #[test]
    fn int_bound_mut_returns_none_for_ptrinfo_forwarded() {
        use crate::optimizeopt::info::PtrInfo;
        let a = BoxRef::new_resop(Type::Ref, 0);
        a.set_forwarded_info(OpInfo::Ptr(PtrInfo::nonnull()));
        assert!(a.int_bound_mut().is_none());
    }
}
