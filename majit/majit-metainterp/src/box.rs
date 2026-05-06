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

use std::cell::{Ref, RefCell};
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
    /// is the result identity. The `Op` struct mapping is attached via
    /// an adapter in H-6.1.
    ResOp,

    /// `resoperation.py:699 AbstractInputArg`.
    /// `position` mirrors `opencoder.py:710 AbstractInputArg.get_position`.
    InputArg { position: Option<u32> },

    /// `history.py:220 ConstInt` / `:261 ConstFloat` / `:307 ConstPtr`.
    Const(Value),

    /// pyre-only transition variant — no direct RPython counterpart.
    /// Temporary representation until opencoder's byte-trace position
    /// information is absorbed into the InputArg/ResOp fields themselves.
    /// **Retired at H-7 prerequisite 8** — if it remains afterwards the
    /// epic is incomplete.
    FrontendOp { position_and_flags: u64 },
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
/// `Eq` / `Hash` are pointer identity. `Rc::clone` shares the allocation,
/// so it is a stable dict key.
pub struct BoxRef(Rc<Box>);

impl BoxRef {
    /// New `AbstractResOp` Box.
    pub fn new_resop(type_: Type) -> Self {
        Self(Rc::new(Box {
            forwarded: RefCell::new(Forwarded::None),
            type_,
            kind: BoxKind::ResOp,
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
    pub fn new_const(value: Value) -> Self {
        let type_ = value.get_type();
        Self(Rc::new(Box {
            forwarded: RefCell::new(Forwarded::None),
            type_,
            kind: BoxKind::Const(value),
        }))
    }

    /// pyre transition variant — used only until H-7.
    pub fn new_frontend_op(type_: Type, position_and_flags: u64) -> Self {
        Self(Rc::new(Box {
            forwarded: RefCell::new(Forwarded::None),
            type_,
            kind: BoxKind::FrontendOp { position_and_flags },
        }))
    }

    pub fn type_(&self) -> Type {
        self.0.type_
    }

    /// `resoperation.py:47 is_constant`.
    pub fn is_constant(&self) -> bool {
        matches!(self.0.kind, BoxKind::Const(_))
    }

    pub fn is_inputarg(&self) -> bool {
        matches!(self.0.kind, BoxKind::InputArg { .. })
    }

    pub fn is_resop(&self) -> bool {
        matches!(self.0.kind, BoxKind::ResOp)
    }

    /// Extract the constant value. Mirrors `history.py:233 ConstInt.getint`
    /// and the equivalent accessors on the other Const subclasses.
    pub fn const_value(&self) -> Option<Value> {
        match self.0.kind {
            BoxKind::Const(v) => Some(v),
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
            !matches!(self.0.kind, BoxKind::Const(_)),
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
            !matches!(self.0.kind, BoxKind::Const(_)),
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
        if matches!(self.0.kind, BoxKind::Const(_)) {
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
            BoxKind::ResOp => "ResOp",
            BoxKind::InputArg { .. } => "InputArg",
            BoxKind::Const(_) => "Const",
            BoxKind::FrontendOp { .. } => "FrontendOp",
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

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{Type, Value};

    #[test]
    fn box_ref_identity_is_pointer_equality() {
        let a = BoxRef::new_resop(Type::Int);
        let cloned = a.clone();
        let other = BoxRef::new_resop(Type::Int);
        assert_eq!(a, cloned);
        assert_ne!(a, other);
    }

    #[test]
    fn forwarded_chain_walk_returns_terminal() {
        // a -> b -> c (all resop), all forwarded = None at c.
        let a = BoxRef::new_resop(Type::Int);
        let b = BoxRef::new_resop(Type::Int);
        let c = BoxRef::new_resop(Type::Int);
        a.set_forwarded_box(b.clone());
        b.set_forwarded_box(c.clone());
        assert_eq!(a.get_box_replacement(false), c);
        assert_eq!(b.get_box_replacement(false), c);
        assert_eq!(c.get_box_replacement(false), c);
    }

    #[test]
    fn forwarded_chain_stops_at_info() {
        // a -> b, then b._forwarded = Info(...).
        let a = BoxRef::new_resop(Type::Int);
        let b = BoxRef::new_resop(Type::Int);
        a.set_forwarded_box(b.clone());
        b.set_forwarded_info(OpInfo::Unknown);
        // Walker reaches b, sees Info, returns b.
        assert_eq!(a.get_box_replacement(false), b);
    }

    #[test]
    fn forwarded_chain_not_const_stops_before_const() {
        // a -> b (resop) -> c (const).
        let a = BoxRef::new_resop(Type::Int);
        let b = BoxRef::new_resop(Type::Int);
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

    #[test]
    fn inputarg_position_preserved() {
        let arg = BoxRef::new_inputarg(Type::Ref, Some(3));
        assert!(arg.is_inputarg());
        assert_eq!(arg.inputarg_position(), Some(3));
        assert_eq!(arg.type_(), Type::Ref);
    }

    #[test]
    fn clear_forwarded_resets_slot() {
        let a = BoxRef::new_resop(Type::Int);
        let b = BoxRef::new_resop(Type::Int);
        a.set_forwarded_box(b.clone());
        a.clear_forwarded();
        assert_eq!(a.get_box_replacement(false), a);
        assert!(matches!(*a.get_forwarded(), Forwarded::None));
    }

    #[test]
    fn boxref_used_as_hashmap_key() {
        use std::collections::HashMap;
        let a = BoxRef::new_resop(Type::Int);
        let b = BoxRef::new_resop(Type::Int);
        let mut m: HashMap<BoxRef, i32> = HashMap::new();
        m.insert(a.clone(), 1);
        m.insert(b.clone(), 2);
        assert_eq!(m.get(&a), Some(&1));
        assert_eq!(m.get(&b), Some(&2));
        // Clone shares the allocation, so it hashes to the same key.
        assert_eq!(m.get(&a.clone()), Some(&1));
    }

    #[test]
    fn frontend_op_transition_variant() {
        let op = BoxRef::new_frontend_op(Type::Int, 0xdeadbeef);
        assert!(!op.is_resop());
        assert!(!op.is_inputarg());
        assert!(!op.is_constant());
        if let BoxKind::FrontendOp { position_and_flags } = &op.0.kind {
            assert_eq!(*position_and_flags, 0xdeadbeef);
        } else {
            panic!("expected FrontendOp kind");
        }
    }

    #[test]
    #[should_panic]
    fn set_forwarded_to_self_panics_in_debug() {
        let a = BoxRef::new_resop(Type::Int);
        a.set_forwarded_box(a.clone());
    }

    // H-3.2c slice 2: BoxRef-native ptr_info / int_bound readers.
    // RPython parity: optimizer.py:99-113 getptrinfo / getintbound is the
    // BoxRef-direct read path. The contract is that the caller has
    // already walked the chain to the terminal box before calling.

    #[test]
    fn ptr_info_returns_inner_when_forwarded_is_ptr_info() {
        use crate::optimizeopt::info::PtrInfo;
        let a = BoxRef::new_resop(Type::Ref);
        a.set_forwarded_info(OpInfo::Ptr(PtrInfo::nonnull()));
        let pi = a.ptr_info().expect("ptr_info should return Some");
        assert!(pi.is_nonnull());
    }

    #[test]
    fn ptr_info_returns_none_for_unset_box() {
        let a = BoxRef::new_resop(Type::Ref);
        assert!(a.ptr_info().is_none());
    }

    #[test]
    fn ptr_info_returns_none_for_box_forwarded() {
        // Chain walk is the caller's responsibility, so when `_forwarded`
        // is `Forwarded::Box(_)` (i.e. a box, not info), `ptr_info()` must
        // return None.
        let a = BoxRef::new_resop(Type::Ref);
        let b = BoxRef::new_resop(Type::Ref);
        a.set_forwarded_box(b.clone());
        assert!(a.ptr_info().is_none());
    }

    #[test]
    fn ptr_info_returns_none_for_intbound_forwarded() {
        // `_forwarded` carries OpInfo::IntBound; ptr_info() must reject it.
        use crate::optimizeopt::intutils::IntBound;
        let a = BoxRef::new_resop(Type::Int);
        a.set_forwarded_info(OpInfo::IntBound(IntBound::from_constant(7)));
        assert!(a.ptr_info().is_none());
    }

    #[test]
    fn int_bound_returns_inner_when_forwarded_is_intbound() {
        use crate::optimizeopt::intutils::IntBound;
        let a = BoxRef::new_resop(Type::Int);
        a.set_forwarded_info(OpInfo::IntBound(IntBound::from_constant(42)));
        let ib = a.int_bound().expect("int_bound should return Some");
        assert!(ib.is_constant());
        assert_eq!(ib.get_constant_int(), 42);
    }

    #[test]
    fn int_bound_returns_none_for_unset_box() {
        let a = BoxRef::new_resop(Type::Int);
        assert!(a.int_bound().is_none());
    }

    #[test]
    fn int_bound_returns_none_for_box_forwarded() {
        let a = BoxRef::new_resop(Type::Int);
        let b = BoxRef::new_resop(Type::Int);
        a.set_forwarded_box(b.clone());
        assert!(a.int_bound().is_none());
    }

    #[test]
    fn int_bound_returns_none_for_ptrinfo_forwarded() {
        use crate::optimizeopt::info::PtrInfo;
        let a = BoxRef::new_resop(Type::Ref);
        a.set_forwarded_info(OpInfo::Ptr(PtrInfo::nonnull()));
        assert!(a.int_bound().is_none());
    }
}
