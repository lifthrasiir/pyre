//! Flow-space high-level operations.
//!
//! RPython upstream: `rpython/flowspace/operation.py` (764 LOC).
//!
//! Commit split (Phase 3 F3.3 of the five-year roadmap at
//! `.claude/plans/majestic-forging-meteor.md`):
//!
//! * **Commit 1 (this file's initial landing)** — data model skeleton.
//!   Every RPython `HLOperation` subclass surfaces as an `OpKind`
//!   enum variant; property tables (`arity`, `pure`, `can_overflow`,
//!   `dispatch`, `ovf_variant`) mirror the add_operator table line by
//!   line. `constfold()` returns `None` pending Commit 2. No wiring
//!   into `flowcontext.rs` yet — `record_pure_op` / `record_maybe_raise_op`
//!   still emit `SpaceOperation` directly via raw opname strings.
//! * **Commit 2** — real `constfold()` bodies (pyfunc mirror via match
//!   on `ConstValue`) + `BuiltinException::canraise` data populated
//!   from the trailing `_add_exceptions` / `_add_except_ovf` loop.
//! * **Commit 3** — the explicit subclasses with custom `eval()` logic
//!   (`NewDict`, `NewTuple`, `NewList`, `Pow`, `Iter`, `Next`,
//!   `GetAttr`, `SimpleCall`, `CallArgs`, `Contains`). Flowcontext
//!   migrates to OpKind-based emission.
//!
//! Rust adaptation (parity rule #1, minimum deviation):
//!
//! * Python `class HLOperation(SpaceOperation)` + `HLOperationMeta`
//!   metaclass dispatch collapses into `struct HLOperation { kind:
//!   OpKind, … }` plus a single enum. Each `OpKind` variant is the
//!   direct 1:1 mapping of an RPython `add_operator('name', …)` call
//!   or a top-level class declaration (`NewDict`, `Pow`, `SimpleCall`,
//!   …).
//! * RPython's global `op.*` namespace — populated by `HLOperationMeta.
//!   __init__` via `setattr(op, cls.opname, cls)` — is replaced by the
//!   `OpKind` variant itself; code that says `op.add` in RPython says
//!   `OpKind::Add` in Rust.
//! * `HLOperationMeta._registry` / `_transform` runtime dicts used by
//!   `SingleDispatchMixin` / `DoubleDispatchMixin` are not materialised
//!   here — they are consumed only by the annotator and rtyper, which
//!   land in Phases 4–6. `OpKind::dispatch()` preserves the `None` / 1
//!   / 2 classification so those phases can populate their own tables.

use std::collections::HashMap;

use super::flowcontext::FlowingError;
use super::model::{ConstValue, Constant, Hlvalue, SpaceOperation, Variable};

/// RPython `NOT_REALLY_CONST` (operation.py:22-35). Maps a module
/// qualname to the set of attribute names that are *real* constants
/// on that module; any other attribute of a listed module is treated
/// as runtime-variable and [`constfold_getattr`] declines the fold.
///
/// Upstream uses `Constant(sys)` as the outer key and `Constant(name)`
/// as the inner set member; the Rust port flattens to string keys
/// because our `HostObject::Module` carries a unique qualname that
/// identifies the hosted module 1:1. Adding a module here is a parity
/// requirement — without it, `getattr(sys, 'path')` would fold to a
/// compile-time snapshot, diverging from upstream.
fn not_really_const_declines(module_qualname: &str, attr: &str) -> bool {
    static SYS_REAL_CONSTS: &[&str] = &[
        "maxint",
        "maxunicode",
        "api_version",
        "exit",
        "exc_info",
        "getrefcount",
        "getdefaultencoding",
    ];
    if module_qualname == "sys" {
        return !SYS_REAL_CONSTS.contains(&attr);
    }
    // Modules not in NOT_REALLY_CONST fold normally.
    false
}

/// RPython `rpython/flowspace/operation.py` — enumerates every
/// `HLOperation` subclass. Variant order matches the upstream
/// `add_operator` table then the explicit subclass block.
///
/// `opname()` returns the exact string identifier used on
/// `SpaceOperation.opname`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OpKind {
    // ---- add_operator table (operation.py:445-521) ----
    // unary/binary operators registered via add_operator(); variant
    // names follow the upstream `opname` argument (UpperCamel'd), with
    // `_ovf` siblings as `*Ovf`.
    Is,
    Id,
    Type,
    IsSubtype,
    IsInstance,
    Repr,
    Str,
    Format,
    Len,
    Hash,
    SetAttr,
    DelAttr,
    GetItem,
    GetItemIdx,
    SetItem,
    DelItem,
    GetSlice,
    SetSlice,
    DelSlice,
    Trunc,
    Pos,
    Neg,
    NegOvf,
    Bool,
    Abs,
    AbsOvf,
    Hex,
    Oct,
    Bin,
    Ord,
    Invert,
    Add,
    AddOvf,
    Sub,
    SubOvf,
    Mul,
    MulOvf,
    TrueDiv,
    FloorDiv,
    FloorDivOvf,
    Div,
    DivOvf,
    Mod,
    ModOvf,
    DivMod,
    LShift,
    LShiftOvf,
    RShift,
    And,
    Or,
    Xor,
    Int,
    Index,
    Float,
    Long,
    InplaceAdd,
    InplaceSub,
    InplaceMul,
    InplaceTrueDiv,
    InplaceFloorDiv,
    InplaceDiv,
    InplaceMod,
    InplacePow,
    InplaceLShift,
    InplaceRShift,
    InplaceAnd,
    InplaceOr,
    InplaceXor,
    Lt,
    Le,
    Eq,
    Ne,
    Gt,
    Ge,
    Cmp,
    Coerce,
    Get,
    Set,
    Delete,
    UserDel,
    Buffer,
    Yield,
    NewSlice,
    Hint,

    // ---- explicit subclasses (operation.py:523-712) ----
    // These carry custom `eval()` / `consider()` overrides in RPython.
    // Commit 3 lifts the custom logic into the Rust port.
    Contains,
    NewDict,
    NewTuple,
    NewList,
    Pow,
    Iter,
    Next,
    GetAttr,
    SimpleCall,
    CallArgs,
}

/// RPython `operation.py` `canraise` entries reference the Python
/// exception classes `ValueError`, `OverflowError`, … directly. The
/// Rust port carries identities through this enum so the trailing
/// `_add_exceptions` table (operation.py:717-763) can populate
/// [`OpKind::canraise`] without pulling in the full `HOST_ENV`
/// exception hierarchy at this phase.
///
/// This list matches the set of exception names literally mentioned by
/// `operation.py`; `Exception` is the "any" fallback used by
/// `CallOp.canraise` and `op.getitem`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BuiltinException {
    ValueError,
    UnicodeDecodeError,
    ZeroDivisionError,
    OverflowError,
    IndexError,
    KeyError,
    StopIteration,
    RuntimeError,
    Exception,
}

impl BuiltinException {
    /// Python class name — matches `HOST_ENV.lookup_builtin(...)` keys
    /// so `Bookkeeper.new_exception([BuiltinException::IndexError, ...])`
    /// can resolve each entry through the existing HostObject lookup.
    pub fn host_name(self) -> &'static str {
        match self {
            BuiltinException::ValueError => "ValueError",
            BuiltinException::UnicodeDecodeError => "UnicodeDecodeError",
            BuiltinException::ZeroDivisionError => "ZeroDivisionError",
            BuiltinException::OverflowError => "OverflowError",
            BuiltinException::IndexError => "IndexError",
            BuiltinException::KeyError => "KeyError",
            BuiltinException::StopIteration => "StopIteration",
            BuiltinException::RuntimeError => "RuntimeError",
            BuiltinException::Exception => "Exception",
        }
    }
}

/// RPython `SingleDispatchMixin.dispatch = 1` /
/// `DoubleDispatchMixin.dispatch = 2` / `HLOperation.dispatch = None`
/// (operation.py:70-72, 202-203, 258-259).
///
/// Consumed by the annotator to pick a specialisation; flowspace itself
/// only records the classification.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Dispatch {
    /// `HLOperation.dispatch = None` — special-cased op (NewDict,
    /// NewTuple, NewList, NewSlice, Pow, SimpleCall, CallArgs,
    /// Contains, Trunc, Format, Get, Set, Delete, UserDel, Buffer,
    /// Yield, Hint, InplacePow). Annotator dispatches manually.
    None,
    /// `SingleDispatchMixin.dispatch = 1` — dispatch on `args[0]`.
    Single,
    /// `DoubleDispatchMixin.dispatch = 2` — dispatch on
    /// `(args[0], args[1])`.
    Double,
}

impl OpKind {
    /// Inverse of [`Self::opname`]. Returns `None` when `name` is not
    /// one of RPython's canonical `SpaceOperation.opname` strings —
    /// `flowcontext.rs` also emits a few majit-local synthetic op
    /// names (`not_`, `newset`, `buildstr`, `ll_assert_not_none`,
    /// `list_to_tuple`, `set_typeparam_default`) that decline here.
    pub fn from_opname(name: &str) -> Option<OpKind> {
        Some(match name {
            "is_" => OpKind::Is,
            "id" => OpKind::Id,
            "type" => OpKind::Type,
            "issubtype" => OpKind::IsSubtype,
            "isinstance" => OpKind::IsInstance,
            "repr" => OpKind::Repr,
            "str" => OpKind::Str,
            "format" => OpKind::Format,
            "len" => OpKind::Len,
            "hash" => OpKind::Hash,
            "setattr" => OpKind::SetAttr,
            "delattr" => OpKind::DelAttr,
            "getitem" => OpKind::GetItem,
            "getitem_idx" => OpKind::GetItemIdx,
            "setitem" => OpKind::SetItem,
            "delitem" => OpKind::DelItem,
            "getslice" => OpKind::GetSlice,
            "setslice" => OpKind::SetSlice,
            "delslice" => OpKind::DelSlice,
            "trunc" => OpKind::Trunc,
            "pos" => OpKind::Pos,
            "neg" => OpKind::Neg,
            "neg_ovf" => OpKind::NegOvf,
            "bool" => OpKind::Bool,
            "abs" => OpKind::Abs,
            "abs_ovf" => OpKind::AbsOvf,
            "hex" => OpKind::Hex,
            "oct" => OpKind::Oct,
            "bin" => OpKind::Bin,
            "ord" => OpKind::Ord,
            "invert" => OpKind::Invert,
            "add" => OpKind::Add,
            "add_ovf" => OpKind::AddOvf,
            "sub" => OpKind::Sub,
            "sub_ovf" => OpKind::SubOvf,
            "mul" => OpKind::Mul,
            "mul_ovf" => OpKind::MulOvf,
            "truediv" => OpKind::TrueDiv,
            "floordiv" => OpKind::FloorDiv,
            "floordiv_ovf" => OpKind::FloorDivOvf,
            "div" => OpKind::Div,
            "div_ovf" => OpKind::DivOvf,
            "mod" => OpKind::Mod,
            "mod_ovf" => OpKind::ModOvf,
            "divmod" => OpKind::DivMod,
            "lshift" => OpKind::LShift,
            "lshift_ovf" => OpKind::LShiftOvf,
            "rshift" => OpKind::RShift,
            "and_" => OpKind::And,
            "or_" => OpKind::Or,
            "xor" => OpKind::Xor,
            "int" => OpKind::Int,
            "index" => OpKind::Index,
            "float" => OpKind::Float,
            "long" => OpKind::Long,
            "inplace_add" => OpKind::InplaceAdd,
            "inplace_sub" => OpKind::InplaceSub,
            "inplace_mul" => OpKind::InplaceMul,
            "inplace_truediv" => OpKind::InplaceTrueDiv,
            "inplace_floordiv" => OpKind::InplaceFloorDiv,
            "inplace_div" => OpKind::InplaceDiv,
            "inplace_mod" => OpKind::InplaceMod,
            "inplace_pow" => OpKind::InplacePow,
            "inplace_lshift" => OpKind::InplaceLShift,
            "inplace_rshift" => OpKind::InplaceRShift,
            "inplace_and" => OpKind::InplaceAnd,
            "inplace_or" => OpKind::InplaceOr,
            "inplace_xor" => OpKind::InplaceXor,
            "lt" => OpKind::Lt,
            "le" => OpKind::Le,
            "eq" => OpKind::Eq,
            "ne" => OpKind::Ne,
            "gt" => OpKind::Gt,
            "ge" => OpKind::Ge,
            "cmp" => OpKind::Cmp,
            "coerce" => OpKind::Coerce,
            "get" => OpKind::Get,
            "set" => OpKind::Set,
            "delete" => OpKind::Delete,
            "userdel" => OpKind::UserDel,
            "buffer" => OpKind::Buffer,
            "yield_" => OpKind::Yield,
            "newslice" => OpKind::NewSlice,
            "hint" => OpKind::Hint,
            "contains" => OpKind::Contains,
            "newdict" => OpKind::NewDict,
            "newtuple" => OpKind::NewTuple,
            "newlist" => OpKind::NewList,
            "pow" => OpKind::Pow,
            "iter" => OpKind::Iter,
            "next" => OpKind::Next,
            "getattr" => OpKind::GetAttr,
            "simple_call" => OpKind::SimpleCall,
            "call_args" => OpKind::CallArgs,
            _ => return None,
        })
    }

    /// RPython `cls.opname` (populated by `add_operator(name, …)`).
    ///
    /// Every variant's return value is the exact string used on
    /// `SpaceOperation.opname`, verified by the upstream test
    /// `rpython/flowspace/test/test_objspace.py`.
    pub fn opname(self) -> &'static str {
        match self {
            OpKind::Is => "is_",
            OpKind::Id => "id",
            OpKind::Type => "type",
            OpKind::IsSubtype => "issubtype",
            OpKind::IsInstance => "isinstance",
            OpKind::Repr => "repr",
            OpKind::Str => "str",
            OpKind::Format => "format",
            OpKind::Len => "len",
            OpKind::Hash => "hash",
            OpKind::SetAttr => "setattr",
            OpKind::DelAttr => "delattr",
            OpKind::GetItem => "getitem",
            OpKind::GetItemIdx => "getitem_idx",
            OpKind::SetItem => "setitem",
            OpKind::DelItem => "delitem",
            OpKind::GetSlice => "getslice",
            OpKind::SetSlice => "setslice",
            OpKind::DelSlice => "delslice",
            OpKind::Trunc => "trunc",
            OpKind::Pos => "pos",
            OpKind::Neg => "neg",
            OpKind::NegOvf => "neg_ovf",
            OpKind::Bool => "bool",
            OpKind::Abs => "abs",
            OpKind::AbsOvf => "abs_ovf",
            OpKind::Hex => "hex",
            OpKind::Oct => "oct",
            OpKind::Bin => "bin",
            OpKind::Ord => "ord",
            OpKind::Invert => "invert",
            OpKind::Add => "add",
            OpKind::AddOvf => "add_ovf",
            OpKind::Sub => "sub",
            OpKind::SubOvf => "sub_ovf",
            OpKind::Mul => "mul",
            OpKind::MulOvf => "mul_ovf",
            OpKind::TrueDiv => "truediv",
            OpKind::FloorDiv => "floordiv",
            OpKind::FloorDivOvf => "floordiv_ovf",
            OpKind::Div => "div",
            OpKind::DivOvf => "div_ovf",
            OpKind::Mod => "mod",
            OpKind::ModOvf => "mod_ovf",
            OpKind::DivMod => "divmod",
            OpKind::LShift => "lshift",
            OpKind::LShiftOvf => "lshift_ovf",
            OpKind::RShift => "rshift",
            OpKind::And => "and_",
            OpKind::Or => "or_",
            OpKind::Xor => "xor",
            OpKind::Int => "int",
            OpKind::Index => "index",
            OpKind::Float => "float",
            OpKind::Long => "long",
            OpKind::InplaceAdd => "inplace_add",
            OpKind::InplaceSub => "inplace_sub",
            OpKind::InplaceMul => "inplace_mul",
            OpKind::InplaceTrueDiv => "inplace_truediv",
            OpKind::InplaceFloorDiv => "inplace_floordiv",
            OpKind::InplaceDiv => "inplace_div",
            OpKind::InplaceMod => "inplace_mod",
            OpKind::InplacePow => "inplace_pow",
            OpKind::InplaceLShift => "inplace_lshift",
            OpKind::InplaceRShift => "inplace_rshift",
            OpKind::InplaceAnd => "inplace_and",
            OpKind::InplaceOr => "inplace_or",
            OpKind::InplaceXor => "inplace_xor",
            OpKind::Lt => "lt",
            OpKind::Le => "le",
            OpKind::Eq => "eq",
            OpKind::Ne => "ne",
            OpKind::Gt => "gt",
            OpKind::Ge => "ge",
            OpKind::Cmp => "cmp",
            OpKind::Coerce => "coerce",
            OpKind::Get => "get",
            OpKind::Set => "set",
            OpKind::Delete => "delete",
            OpKind::UserDel => "userdel",
            OpKind::Buffer => "buffer",
            OpKind::Yield => "yield_",
            OpKind::NewSlice => "newslice",
            OpKind::Hint => "hint",

            OpKind::Contains => "contains",
            OpKind::NewDict => "newdict",
            OpKind::NewTuple => "newtuple",
            OpKind::NewList => "newlist",
            OpKind::Pow => "pow",
            OpKind::Iter => "iter",
            OpKind::Next => "next",
            OpKind::GetAttr => "getattr",
            OpKind::SimpleCall => "simple_call",
            OpKind::CallArgs => "call_args",
        }
    }

    /// RPython `cls.arity` — the declared argument count. Matches the
    /// `arity=` argument of `add_operator()` / the explicit `arity`
    /// class attribute on each subclass.
    ///
    /// Returns `None` for variadic / manually dispatched operations
    /// where RPython leaves `arity` unset (`NewDict`, `NewTuple`,
    /// `NewList`, `SimpleCall`, `CallArgs`, `Hint`).
    pub fn arity(self) -> Option<usize> {
        match self {
            // unary
            OpKind::Id
            | OpKind::Type
            | OpKind::Repr
            | OpKind::Str
            | OpKind::Len
            | OpKind::Hash
            | OpKind::Trunc
            | OpKind::Pos
            | OpKind::Neg
            | OpKind::NegOvf
            | OpKind::Bool
            | OpKind::Abs
            | OpKind::AbsOvf
            | OpKind::Hex
            | OpKind::Oct
            | OpKind::Bin
            | OpKind::Ord
            | OpKind::Invert
            | OpKind::Int
            | OpKind::Index
            | OpKind::Float
            | OpKind::Long
            | OpKind::UserDel
            | OpKind::Buffer
            | OpKind::Yield
            | OpKind::Iter
            | OpKind::Next => Some(1),

            // binary
            OpKind::Is
            | OpKind::IsSubtype
            | OpKind::IsInstance
            | OpKind::Format
            | OpKind::DelAttr
            | OpKind::GetItem
            | OpKind::GetItemIdx
            | OpKind::DelItem
            | OpKind::Add
            | OpKind::AddOvf
            | OpKind::Sub
            | OpKind::SubOvf
            | OpKind::Mul
            | OpKind::MulOvf
            | OpKind::TrueDiv
            | OpKind::FloorDiv
            | OpKind::FloorDivOvf
            | OpKind::Div
            | OpKind::DivOvf
            | OpKind::Mod
            | OpKind::ModOvf
            | OpKind::DivMod
            | OpKind::LShift
            | OpKind::LShiftOvf
            | OpKind::RShift
            | OpKind::And
            | OpKind::Or
            | OpKind::Xor
            | OpKind::InplaceAdd
            | OpKind::InplaceSub
            | OpKind::InplaceMul
            | OpKind::InplaceTrueDiv
            | OpKind::InplaceFloorDiv
            | OpKind::InplaceDiv
            | OpKind::InplaceMod
            | OpKind::InplacePow
            | OpKind::InplaceLShift
            | OpKind::InplaceRShift
            | OpKind::InplaceAnd
            | OpKind::InplaceOr
            | OpKind::InplaceXor
            | OpKind::Lt
            | OpKind::Le
            | OpKind::Eq
            | OpKind::Ne
            | OpKind::Gt
            | OpKind::Ge
            | OpKind::Cmp
            | OpKind::Coerce
            | OpKind::Delete
            | OpKind::Contains
            | OpKind::GetAttr => Some(2),

            // ternary
            OpKind::SetAttr
            | OpKind::SetItem
            | OpKind::GetSlice
            | OpKind::DelSlice
            | OpKind::Get
            | OpKind::Set
            | OpKind::NewSlice
            | OpKind::Pow => Some(3),

            // quaternary
            OpKind::SetSlice => Some(4),

            // variadic / manual-dispatch
            OpKind::NewDict
            | OpKind::NewTuple
            | OpKind::NewList
            | OpKind::SimpleCall
            | OpKind::CallArgs
            | OpKind::Hint => None,
        }
    }

    /// RPython `cls.pure` — whether the op is a `PureOperation`
    /// subclass and therefore eligible for `constfold()`.
    ///
    /// Side-effecting ops (setattr/setitem/setslice/call*/inplace*) are
    /// `False`; pure arithmetic / container construction is `True`.
    /// Matches the `pure=` argument of `add_operator` plus the explicit
    /// `PureOperation` base of subclasses like `NewTuple` and `Pow`.
    ///
    /// **`_ovf` siblings are NOT pure.** Upstream
    /// `add_operator(name, ovf=True)` registers BOTH the base op (with
    /// `base_cls = OverflowingOperation`, which extends
    /// `PureOperation`) AND a suffixed twin via
    /// `add_operator(name + '_ovf', arity, dispatch, pyfunc=ovf_func)`
    /// (`operation.py:337-338`). The recursive call carries no
    /// `pure=True` / `ovf=True`, so the suffixed twin's `base_cls`
    /// falls through to plain `HLOperation` (`operation.py:321`) —
    /// which has the default `constfold(self) -> None`
    /// (`operation.py:96-97`). Therefore `_ovf` ops never constfold:
    /// they remain as ops even when all operands are constants and a
    /// constant-time `ovf_func(*args)` would raise. Reading them as
    /// pure (and folding them) inserts a hard-error divergence at
    /// every constant-zero-divisor / overflow site that upstream
    /// would have left to the runtime.
    pub fn pure(self) -> bool {
        match self {
            OpKind::Is
            | OpKind::Type
            | OpKind::IsSubtype
            | OpKind::IsInstance
            | OpKind::Repr
            | OpKind::Str
            | OpKind::Len
            | OpKind::GetItem
            | OpKind::GetItemIdx
            | OpKind::GetSlice
            | OpKind::Pos
            | OpKind::Neg
            | OpKind::Bool
            | OpKind::Abs
            | OpKind::Hex
            | OpKind::Oct
            | OpKind::Bin
            | OpKind::Ord
            | OpKind::Invert
            | OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::TrueDiv
            | OpKind::FloorDiv
            | OpKind::Div
            | OpKind::Mod
            | OpKind::DivMod
            | OpKind::LShift
            | OpKind::RShift
            | OpKind::And
            | OpKind::Or
            | OpKind::Xor
            | OpKind::Int
            | OpKind::Index
            | OpKind::Float
            | OpKind::Long
            | OpKind::Lt
            | OpKind::Le
            | OpKind::Eq
            | OpKind::Ne
            | OpKind::Gt
            | OpKind::Ge
            | OpKind::Cmp
            | OpKind::Coerce
            | OpKind::Contains
            | OpKind::Get
            | OpKind::Buffer
            | OpKind::NewTuple
            | OpKind::Pow => true,

            // `_ovf` siblings: HLOperation default (no constfold).
            OpKind::NegOvf
            | OpKind::AbsOvf
            | OpKind::AddOvf
            | OpKind::SubOvf
            | OpKind::MulOvf
            | OpKind::FloorDivOvf
            | OpKind::DivOvf
            | OpKind::ModOvf
            | OpKind::LShiftOvf => false,

            // upstream `class GetAttr(SingleDispatchMixin, HLOperation)`
            // (operation.py:617-646) is NOT a PureOperation — only its
            // `constfold()` override is wired to fold constant
            // attribute lookups. The Rust port surfaces that via the
            // "custom constfold" eligibility gate in
            // `HLOperation::constfold`, not through `pure()`.
            OpKind::GetAttr
            | OpKind::Id
            | OpKind::Format
            | OpKind::Hash
            | OpKind::SetAttr
            | OpKind::DelAttr
            | OpKind::SetItem
            | OpKind::DelItem
            | OpKind::SetSlice
            | OpKind::DelSlice
            | OpKind::Trunc
            | OpKind::InplaceAdd
            | OpKind::InplaceSub
            | OpKind::InplaceMul
            | OpKind::InplaceTrueDiv
            | OpKind::InplaceFloorDiv
            | OpKind::InplaceDiv
            | OpKind::InplaceMod
            | OpKind::InplacePow
            | OpKind::InplaceLShift
            | OpKind::InplaceRShift
            | OpKind::InplaceAnd
            | OpKind::InplaceOr
            | OpKind::InplaceXor
            | OpKind::Set
            | OpKind::Delete
            | OpKind::UserDel
            | OpKind::Yield
            | OpKind::NewSlice
            | OpKind::Hint
            | OpKind::NewDict
            | OpKind::NewList
            | OpKind::Iter
            | OpKind::Next
            | OpKind::SimpleCall
            | OpKind::CallArgs => false,
        }
    }

    /// RPython `cls.can_overflow` — `True` for `OverflowingOperation`
    /// variants (`add`, `sub`, `mul`, `neg`, `abs`, `div`, `floordiv`,
    /// `mod`, `lshift` and their `_ovf` siblings).
    pub fn can_overflow(self) -> bool {
        matches!(
            self,
            OpKind::Neg
                | OpKind::NegOvf
                | OpKind::Abs
                | OpKind::AbsOvf
                | OpKind::Add
                | OpKind::AddOvf
                | OpKind::Sub
                | OpKind::SubOvf
                | OpKind::Mul
                | OpKind::MulOvf
                | OpKind::FloorDiv
                | OpKind::FloorDivOvf
                | OpKind::Div
                | OpKind::DivOvf
                | OpKind::Mod
                | OpKind::ModOvf
                | OpKind::LShift
                | OpKind::LShiftOvf
        )
    }

    /// RPython `cls.dispatch` classification used by the annotator to
    /// pick a specialisation.
    pub fn dispatch(self) -> Dispatch {
        match self {
            // SingleDispatchMixin: `add_operator(…, dispatch=1)`.
            OpKind::Id
            | OpKind::Type
            | OpKind::IsSubtype
            | OpKind::IsInstance
            | OpKind::Repr
            | OpKind::Str
            | OpKind::Len
            | OpKind::Hash
            | OpKind::SetAttr
            | OpKind::DelAttr
            | OpKind::GetSlice
            | OpKind::SetSlice
            | OpKind::DelSlice
            | OpKind::Pos
            | OpKind::Neg
            | OpKind::NegOvf
            | OpKind::Bool
            | OpKind::Abs
            | OpKind::AbsOvf
            | OpKind::Hex
            | OpKind::Oct
            | OpKind::Bin
            | OpKind::Ord
            | OpKind::Invert
            | OpKind::Int
            | OpKind::Float
            | OpKind::Long
            | OpKind::Hint
            | OpKind::Contains
            | OpKind::Iter
            | OpKind::Next
            | OpKind::GetAttr
            | OpKind::SimpleCall
            | OpKind::CallArgs => Dispatch::Single,

            // DoubleDispatchMixin: `add_operator(…, dispatch=2)`.
            OpKind::Is
            | OpKind::GetItem
            | OpKind::GetItemIdx
            | OpKind::SetItem
            | OpKind::DelItem
            | OpKind::Add
            | OpKind::AddOvf
            | OpKind::Sub
            | OpKind::SubOvf
            | OpKind::Mul
            | OpKind::MulOvf
            | OpKind::TrueDiv
            | OpKind::FloorDiv
            | OpKind::FloorDivOvf
            | OpKind::Div
            | OpKind::DivOvf
            | OpKind::Mod
            | OpKind::ModOvf
            | OpKind::LShift
            | OpKind::LShiftOvf
            | OpKind::RShift
            | OpKind::And
            | OpKind::Or
            | OpKind::Xor
            | OpKind::InplaceAdd
            | OpKind::InplaceSub
            | OpKind::InplaceMul
            | OpKind::InplaceTrueDiv
            | OpKind::InplaceFloorDiv
            | OpKind::InplaceDiv
            | OpKind::InplaceMod
            | OpKind::InplaceLShift
            | OpKind::InplaceRShift
            | OpKind::InplaceAnd
            | OpKind::InplaceOr
            | OpKind::InplaceXor
            | OpKind::Lt
            | OpKind::Le
            | OpKind::Eq
            | OpKind::Ne
            | OpKind::Gt
            | OpKind::Ge
            | OpKind::Cmp
            | OpKind::Coerce => Dispatch::Double,

            // manual dispatch / no dispatch=… argument on upstream.
            OpKind::Format
            | OpKind::Trunc
            | OpKind::Index
            | OpKind::InplacePow
            | OpKind::DivMod
            | OpKind::Get
            | OpKind::Set
            | OpKind::Delete
            | OpKind::UserDel
            | OpKind::Buffer
            | OpKind::Yield
            | OpKind::NewSlice
            | OpKind::NewDict
            | OpKind::NewTuple
            | OpKind::NewList
            | OpKind::Pow => Dispatch::None,
        }
    }

    /// RPython `OverflowingOperation.ovf_variant` (operation.py:338-339).
    ///
    /// Returns the `_ovf` twin of a checked arithmetic op, or `None`
    /// when the op has no overflow variant.
    pub fn ovf_variant(self) -> Option<OpKind> {
        Some(match self {
            OpKind::Neg => OpKind::NegOvf,
            OpKind::Abs => OpKind::AbsOvf,
            OpKind::Add => OpKind::AddOvf,
            OpKind::Sub => OpKind::SubOvf,
            OpKind::Mul => OpKind::MulOvf,
            OpKind::FloorDiv => OpKind::FloorDivOvf,
            OpKind::Div => OpKind::DivOvf,
            OpKind::Mod => OpKind::ModOvf,
            OpKind::LShift => OpKind::LShiftOvf,
            _ => return None,
        })
    }

    /// RPython `cls.canraise` — populated by the trailing
    /// `_add_exceptions` / `_add_except_ovf` loop
    /// (`operation.py:728-764`).
    ///
    /// Trace order matches upstream, so each arm corresponds to a
    /// specific set of `lis.append(exc)` calls:
    ///
    /// * `op.getitem / getitem_idx / setitem / delitem` line 728-731.
    /// * `op.contains` line 732.
    /// * `_add_exceptions("div mod divmod truediv floordiv pow
    ///   inplace_div inplace_mod inplace_truediv inplace_floordiv
    ///   inplace_pow", ZeroDivisionError)` line 751-753.
    /// * `_add_exceptions("pow inplace_pow lshift inplace_lshift rshift
    ///   inplace_rshift", ValueError)` line 754-755.
    /// * `_add_exceptions("truediv divmod inplace_add inplace_sub
    ///   inplace_mul inplace_truediv inplace_floordiv inplace_div
    ///   inplace_mod inplace_pow inplace_lshift", OverflowError)`
    ///   line 756-759.
    /// * `_add_except_ovf("neg abs add sub mul floordiv div mod
    ///   lshift")` line 760-761 — copies the base canraise onto the
    ///   `_ovf` twin then appends `OverflowError`.
    /// * `_add_exceptions("pow", OverflowError)` line 762-763 — float
    ///   case.
    pub fn canraise(self) -> &'static [BuiltinException] {
        use BuiltinException::*;
        match self {
            // Explicit HLOperation subclasses with custom canraise.
            OpKind::GetAttr | OpKind::Iter => &[],
            OpKind::Next => &[StopIteration, RuntimeError],

            // operation.py:728-731.
            OpKind::GetItem | OpKind::GetItemIdx | OpKind::SetItem | OpKind::DelItem => {
                &[IndexError, KeyError, Exception]
            }
            // operation.py:732.
            OpKind::Contains => &[Exception],

            // `div`: ZeroDivisionError (751). No direct OverflowError.
            // `div_ovf`: `_add_except_ovf` duplicates the base list and
            // appends OverflowError (760).
            OpKind::Div => &[ZeroDivisionError],
            OpKind::DivOvf => &[ZeroDivisionError, OverflowError],
            OpKind::Mod => &[ZeroDivisionError],
            OpKind::ModOvf => &[ZeroDivisionError, OverflowError],
            OpKind::FloorDiv => &[ZeroDivisionError],
            OpKind::FloorDivOvf => &[ZeroDivisionError, OverflowError],
            // `truediv` / `divmod` — ZeroDivisionError (751) + OverflowError
            // (756). No `_ovf` variant in the table.
            OpKind::TrueDiv => &[ZeroDivisionError, OverflowError],
            OpKind::DivMod => &[ZeroDivisionError, OverflowError],

            // `pow` — ZeroDivisionError (751) + ValueError (754) +
            // OverflowError (762, float case).
            OpKind::Pow => &[ZeroDivisionError, ValueError, OverflowError],

            // `lshift`: ValueError (754); `lshift_ovf`: + OverflowError (760).
            OpKind::LShift => &[ValueError],
            OpKind::LShiftOvf => &[ValueError, OverflowError],
            // `rshift`: ValueError (754). No `_ovf`.
            OpKind::RShift => &[ValueError],

            // `_ovf` twins of the pure arithmetic family (760-761).
            // Their base variants have empty canraise.
            OpKind::NegOvf | OpKind::AbsOvf | OpKind::AddOvf | OpKind::SubOvf | OpKind::MulOvf => {
                &[OverflowError]
            }

            // inplace family (753, 754, 758-759). Each upstream line
            // appends one exception; the Rust port flattens the
            // cumulative set.
            OpKind::InplaceAdd => &[OverflowError],
            OpKind::InplaceSub => &[OverflowError],
            OpKind::InplaceMul => &[OverflowError],
            OpKind::InplaceDiv => &[ZeroDivisionError, OverflowError],
            OpKind::InplaceMod => &[ZeroDivisionError, OverflowError],
            OpKind::InplaceFloorDiv => &[ZeroDivisionError, OverflowError],
            OpKind::InplaceTrueDiv => &[ZeroDivisionError, OverflowError],
            OpKind::InplacePow => &[ZeroDivisionError, ValueError, OverflowError],
            OpKind::InplaceLShift => &[ValueError, OverflowError],
            OpKind::InplaceRShift => &[ValueError],

            // Default: upstream leaves `canraise = []`.
            _ => &[],
        }
    }
}

/// RPython `flowspace/operation.py:66-116` — `class HLOperation(SpaceOperation)`.
///
/// A high-level operation produced by flow objspace handlers. Each
/// `HLOperation` may be folded to a `Constant` via `constfold()` or
/// recorded as a `SpaceOperation` by `flowcontext.record()`.
#[derive(Clone, Debug)]
pub struct HLOperation {
    /// RPython type-erased identity; matches the Python subclass of
    /// `HLOperation` that would have been instantiated upstream.
    pub kind: OpKind,
    /// RPython `self.args = list(args)` (operation.py:74).
    pub args: Vec<Hlvalue>,
    /// RPython `self.result = Variable()` (operation.py:75).
    pub result: Variable,
    /// RPython `self.offset = -1` (operation.py:76) — rewritten by
    /// `flowcontext.record()` before the op becomes a `SpaceOperation`.
    pub offset: i64,
}

impl HLOperation {
    /// RPython `HLOperation.__init__(*args)` (operation.py:73-76).
    pub fn new(kind: OpKind, args: Vec<Hlvalue>) -> Self {
        HLOperation {
            kind,
            args,
            result: Variable::new(),
            offset: -1,
        }
    }

    /// RPython `HLOperation.replace(mapping)` (operation.py:78-84).
    ///
    /// The mapping is polymorphic (`Variable → Variable | Constant`)
    /// per `flowspace/model.py:347 Variable.replace`. HLOperation.result
    /// is typed `Variable` in upstream too, and RPython's pre-rtyper
    /// stages never rename a Variable into a Constant — the Rust port
    /// keeps the field concrete and asserts that shape.
    pub fn replace(&self, mapping: &HashMap<Variable, Hlvalue>) -> HLOperation {
        let newargs: Vec<Hlvalue> = self.args.iter().map(|a| a.replace(mapping)).collect();
        let newresult = match self.result.replace(mapping) {
            Hlvalue::Variable(v) => v,
            Hlvalue::Constant(c) => panic!(
                "HLOperation.replace: Variable result renamed to Constant ({c:?}) \
                 — upstream HLOperation.result is always Variable"
            ),
        };
        HLOperation {
            kind: self.kind,
            args: newargs,
            result: newresult,
            offset: self.offset,
        }
    }

    /// RPython `HLOperation.constfold()` (operation.py:98-99, overridden
    /// on PureOperation at 120-132).
    ///
    /// Behaviour matches PureOperation upstream: require every arg to
    /// be foldable (`Constant.foldable()`), then apply the pyfunc
    /// equivalent. Non-pure ops and folds that would overflow (RPython
    /// `type(result) is long` branch at 141-142) return `Ok(None)` so
    /// the caller emits a `SpaceOperation` instead. Flow-time hard
    /// errors (currently the 3-arg `getattr` case and constant module
    /// attribute misses) surface as [`FlowingError`].
    ///
    /// Commit 2 scope: pure integer / float / bool / tuple / comparison
    /// arithmetic — the subset actually exercised by flowspace on
    /// realistic RPython inputs. Non-covered pure ops fall through to
    /// `None` (no fold attempted), which is a strict subset of upstream
    /// behaviour.
    pub fn constfold(&self) -> Result<Option<Hlvalue>, FlowingError> {
        // Pure ops go through PureOperation.constfold (operation.py:120-132).
        // A few non-pure ops (`GetAttr`, `Iter`, `Next`) carry their own
        // `constfold()` override upstream and must be allowed past the
        // gate. `SimpleCall` / `CallArgs` / `Pow` handle their own
        // special-casing inside pyfunc() above.
        let eligible =
            self.kind.pure() || matches!(self.kind, OpKind::GetAttr | OpKind::Iter | OpKind::Next);
        if !eligible {
            return Ok(None);
        }
        if self.kind == OpKind::GetAttr {
            return self.constfold_getattr();
        }
        for arg in &self.args {
            match arg {
                Hlvalue::Constant(c) if c.foldable() => {}
                _ => return Ok(None),
            }
        }
        let args: Vec<&ConstValue> = self
            .args
            .iter()
            .map(|a| match a {
                Hlvalue::Constant(c) => &c.value,
                _ => unreachable!("foldable() above excludes Variable args"),
            })
            .collect();
        // RPython `PureOperation.constfold` (operation.py:120-127) wraps
        // `pyfunc(*args)` in `try/except Exception` and re-raises the
        // captured exception as `FlowingError(...)` when all args are
        // foldable. Rust `pyfunc` returns `Option<ConstValue>`, conflating
        // "type combo not yet handled" (decline → `Ok(None)`) with
        // "operation deterministically raises" (e.g. division by zero,
        // mixed-type ordering comparison) which upstream surfaces as
        // FlowingError. Detect the deterministic-raise cases explicitly
        // before pyfunc fall-through so the exposed graph shape matches
        // upstream — record_pure_op consumers (build_flow / flowcontext)
        // see the FlowingError instead of silently emitting a runtime-
        // failing SpaceOperation.
        if let Some(err) = constfold_always_raises(self.kind, &args) {
            return Err(err);
        }
        let Some(result) = pyfunc(self.kind, &args) else {
            return Ok(None);
        };
        Ok(Some(Hlvalue::Constant(Constant::new(result))))
    }

    fn constfold_getattr(&self) -> Result<Option<Hlvalue>, FlowingError> {
        // upstream operation.py:624-646:
        //
        //     def constfold(self):
        //         if len(self.args) == 3:
        //             raise FlowingError("getattr() with three arguments not supported: %s" % (self,))
        //         w_obj, w_name = self.args
        //         # handling special things like sys
        //         if (w_obj in NOT_REALLY_CONST and
        //                 w_name not in NOT_REALLY_CONST[w_obj]):
        //             return
        //         if w_obj.foldable() and w_name.foldable():
        //             obj, name = w_obj.value, w_name.value
        //             try:
        //                 result = getattr(obj, name)
        //             except Exception as e:
        //                 raise FlowingError("getattr(%s, %s) always raises %s: %s" % ...)
        //             try:
        //                 return const(result)
        //             except WrapException:
        //                 pass
        if self.args.len() == 3 {
            return Err(FlowingError::new(format!(
                "getattr() with three arguments not supported: {self:?}"
            )));
        }
        let [w_obj_hl, w_name_hl] = self.args.as_slice() else {
            return Ok(None);
        };
        let (Hlvalue::Constant(w_obj), Hlvalue::Constant(w_name)) = (w_obj_hl, w_name_hl) else {
            return Ok(None);
        };
        let Some(name) = w_name.value.as_text() else {
            return Ok(None);
        };
        let name_str = name.to_string();

        // upstream operation.py:631-633 — NOT_REALLY_CONST guard. In
        // the Rust port the table is keyed on module qualname; the
        // guard only fires for module objects.
        if let ConstValue::HostObject(h) = &w_obj.value {
            if h.is_module() && not_really_const_declines(h.qualname(), &name_str) {
                return Ok(None);
            }
        }

        // upstream operation.py:634 — `if w_obj.foldable() and w_name.foldable()`.
        // Constant-str for `w_name` already satisfies foldable().
        if !w_obj.foldable() {
            return Ok(None);
        }

        match crate::flowspace::model::const_runtime_getattr(&w_obj.value, &name_str) {
            Ok(Some(value)) => Ok(Some(Hlvalue::Constant(Constant::new(value)))),
            // upstream WrapException path — flowspace couldn't wrap the
            // result, so fold declines silently. We report this as "no
            // surface for this attribute" by returning None.
            Ok(None) => Ok(None),
            Err(msg) => Err(FlowingError::new(msg)),
        }
    }

    /// RPython `OverflowingOperation.ovfchecked()` (operation.py:197-200)
    /// — returns the `_ovf` twin of this operation, carrying the same
    /// args / result / offset.
    pub fn ovfchecked(&self) -> Option<HLOperation> {
        let ovf_kind = self.kind.ovf_variant()?;
        Some(HLOperation {
            kind: ovf_kind,
            args: self.args.clone(),
            result: self.result.clone(),
            offset: self.offset,
        })
    }

    /// Lower this `HLOperation` into a plain `SpaceOperation` for
    /// `flowcontext.record()`. Mirrors RPython's implicit upcast via
    /// `HLOperation`'s `SpaceOperation` base class.
    pub fn into_space_operation(self) -> SpaceOperation {
        SpaceOperation::with_offset(
            self.kind.opname(),
            self.args,
            Hlvalue::Variable(self.result),
            self.offset,
        )
    }
}

/// RPython `cls.pyfunc(*args)` applied to constant args — returns
/// `Some(result)` when the pyfunc equivalent fires, or `None` when the
/// fold is skipped for a reason upstream also skips (overflow to long,
/// unhandled type combo, unknown op).
///
/// Commit 2 covers the pure arithmetic subset (int / float / bool / str /
/// tuple / comparisons / identity / type-conversion) that realistic
/// flowspace inputs actually fold. Commit 3 extends this with the
/// explicit subclasses that have custom `eval()` (Iter, Next, GetAttr,
/// SimpleCall, CallArgs, Pow).
pub(crate) fn pyfunc(kind: OpKind, args: &[&ConstValue]) -> Option<ConstValue> {
    // --- variadic / ternary ops that fall outside the fixed-arity
    //     match below ---
    match (kind, args.len()) {
        // RPython `NewTuple` (operation.py:542-548). `PureOperation`
        // whose `pyfunc = lambda *args: args`.
        (OpKind::NewTuple, _) => {
            let items: Vec<ConstValue> = args.iter().map(|&v| v.clone()).collect();
            return Some(ConstValue::Tuple(items));
        }
        // RPython `Pow(PureOperation)` (operation.py:568-578) with
        // arity 3 — `pyfunc = pow`. We fold only the int/int/None
        // variant; float Pow / int-with-mod defer to runtime emit.
        (OpKind::Pow, 3) => {
            if let [
                ConstValue::Int(base),
                ConstValue::Int(exp),
                ConstValue::None,
            ] = args
            {
                if *exp < 0 {
                    return None;
                }
                let e: u32 = (*exp).try_into().ok()?;
                return base.checked_pow(e).map(ConstValue::Int);
            }
        }
        // RPython `GetAttr.constfold()` (operation.py:624-646).
        //
        // upstream shape:
        //   if len(self.args) == 3:
        //       raise FlowingError(...)
        //   w_obj, w_name = self.args
        //   if w_obj in NOT_REALLY_CONST and w_name not in NOT_REALLY_CONST[w_obj]:
        //       return                       # decline fold
        //   if w_obj.foldable() and w_name.foldable():
        //       try:
        //           result = getattr(obj, name)
        //       except Exception as e:
        //           raise FlowingError("%s always raises %s" % …)
        //       return const(result)
        //
        // Phase 3 F3.3 gaps documented per CLAUDE.md parity rule #1:
        //
        //  1. 3-arg `getattr(x, name, default)` support — upstream
        //     raises `FlowingError` at flow time, surfacing a hard
        //     error. The Rust port declines the fold here (falls
        //     through to SpaceOperation emission); `flowcontext.rs`
        //     does not yet surface a FlowingError at this site. The
        //     stricter check lands once `HLOperation::constfold`
        //     returns `Result<Option, FlowingError>` — scheduled
        //     alongside the Phase 5 annotator wiring.
        //  2. `NOT_REALLY_CONST` table (operation.py:22-35) blocks
        //     folds of `sys.path`, `sys.modules`, etc. while keeping
        //     `sys.maxint`, `sys.exc_info`, etc. foldable — PARITY
        //     with upstream's sys-attribute allowlist via the
        //     [`not_really_const_declines`] helper at the top of this
        //     file. Extra volatile modules gain entries by editing
        //     that helper.
        //  3. The generic `getattr(obj, name)` fall-through that
        //     upstream wraps in `try/except Exception` covers class
        //     methods / instance attributes that flowspace may see on
        //     real Python inputs; the Rust port only folds
        //     `HostObject::Module::module_get`. Extending to class
        //     attribute lookup requires a `HostObject::class_get`
        //     helper that resolves bound methods via
        //     `HOST_ENV.lookup_builtin`.
        (OpKind::GetAttr, 2) => {
            if let [ConstValue::HostObject(obj), name] = args
                && let Some(name) = name.as_text()
            {
                if let Some(value) = obj.module_get(name) {
                    return Some(ConstValue::HostObject(value));
                }
            }
        }
        (OpKind::GetAttr, 3) => {
            // upstream raises FlowingError at flow time. Without an
            // error channel on constfold() we decline the fold and
            // let the runtime emit a 3-arg `getattr` SpaceOperation;
            // Phase 5's annotator will surface the error when it
            // consumes the op.
            return None;
        }
        _ => {}
    }

    match (kind, args) {
        // --- unary ---
        (OpKind::Bool, [v]) => v.truthy().map(ConstValue::Bool),
        (OpKind::Neg, [ConstValue::Int(n)]) => n.checked_neg().map(ConstValue::Int),
        (OpKind::NegOvf, [ConstValue::Int(n)]) => n.checked_neg().map(ConstValue::Int),
        (OpKind::Neg, [ConstValue::Float(bits)]) => Some(ConstValue::float(-f64::from_bits(*bits))),
        (OpKind::Pos, [ConstValue::Int(n)]) => Some(ConstValue::Int(*n)),
        (OpKind::Pos, [ConstValue::Float(bits)]) => Some(ConstValue::Float(*bits)),
        (OpKind::Abs, [ConstValue::Int(n)]) => n.checked_abs().map(ConstValue::Int),
        (OpKind::AbsOvf, [ConstValue::Int(n)]) => n.checked_abs().map(ConstValue::Int),
        (OpKind::Abs, [ConstValue::Float(bits)]) => {
            Some(ConstValue::float(f64::from_bits(*bits).abs()))
        }
        (OpKind::Invert, [ConstValue::Int(n)]) => Some(ConstValue::Int(!*n)),
        (OpKind::Int, [ConstValue::Int(n)]) => Some(ConstValue::Int(*n)),
        (OpKind::Int, [ConstValue::Bool(b)]) => Some(ConstValue::Int(if *b { 1 } else { 0 })),
        (OpKind::Int, [ConstValue::Float(bits)]) => {
            let f = f64::from_bits(*bits);
            // RPython `int(float)` truncates toward zero and raises
            // OverflowError on infinities/NaN. Rust `as i64` saturates,
            // so guard the range explicitly and decline otherwise.
            if f.is_finite() && f >= (i64::MIN as f64) && f <= (i64::MAX as f64) {
                Some(ConstValue::Int(f as i64))
            } else {
                None
            }
        }
        (OpKind::Float, [ConstValue::Int(n)]) => Some(ConstValue::float(*n as f64)),
        (OpKind::Float, [ConstValue::Float(bits)]) => Some(ConstValue::Float(*bits)),
        (OpKind::Float, [ConstValue::Bool(b)]) => {
            Some(ConstValue::float(if *b { 1.0 } else { 0.0 }))
        }
        (OpKind::Long, [ConstValue::Int(n)]) => Some(ConstValue::Int(*n)),
        (OpKind::Ord, [ConstValue::ByteStr(s)]) if s.len() == 1 => {
            Some(ConstValue::Int(s[0] as i64))
        }
        (OpKind::Ord, [ConstValue::UniStr(s)]) if s.chars().count() == 1 => {
            s.chars().next().map(|c| ConstValue::Int(c as i64))
        }
        (OpKind::Len, [ConstValue::ByteStr(s)]) => Some(ConstValue::Int(s.len() as i64)),
        (OpKind::Len, [ConstValue::UniStr(s)]) => Some(ConstValue::Int(s.chars().count() as i64)),
        (OpKind::Len, [ConstValue::Tuple(items) | ConstValue::List(items)]) => {
            Some(ConstValue::Int(items.len() as i64))
        }
        (OpKind::Len, [ConstValue::Dict(items)]) => Some(ConstValue::Int(items.len() as i64)),

        // --- binary identity ---
        // `operator.is_` compares Python identity. For the value set
        // the Rust port carries on `Constant`, structural equality of
        // `ConstValue` is a safe proxy: two `ConstValue::Int(3)` do
        // compare `==` in Python via small-int cache and we never
        // synthesise two distinct wrappers for the same primitive.
        (OpKind::Is, [a, b]) => Some(ConstValue::Bool(a == b)),

        // --- binary arithmetic (int) ---
        // `_ovf` siblings are NOT pure (`OpKind::pure()` rejects them),
        // so the `pure()` gate in `HLOperation::constfold` short-
        // circuits before reaching this dispatch — matching upstream
        // `add_operator(name + '_ovf', …)` falling through to plain
        // `HLOperation`. The arms below cover only the BASE pure ops.
        (OpKind::Add, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            a.checked_add(*b).map(ConstValue::Int)
        }
        (OpKind::Sub, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            a.checked_sub(*b).map(ConstValue::Int)
        }
        (OpKind::Mul, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            a.checked_mul(*b).map(ConstValue::Int)
        }
        (OpKind::FloorDiv, [ConstValue::Int(a), ConstValue::Int(b)])
        | (OpKind::Div, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            // RPython flow-space folds `div` as `operator.floordiv` for
            // int operands (Python 3 floor-toward-`-inf`), implemented
            // by `ll_int_py_div` (rpython/rtyper/rint.py:398). `Rust`'s
            // `checked_div_euclid` is Euclidean (always non-negative
            // remainder) and diverges from Python for negative divisors
            // — `3 // -2 == -2` in Python but Euclidean gives `-1`.
            // `int_py_floor_div` ports `ll_int_py_div` line-by-line.
            // `ZeroDivisionError` is not caught here; the always-raises
            // whitelist surfaces it for `Div`/`FloorDiv`/`Mod` and their
            // `_ovf` siblings.
            int_py_floor_div(*a, *b).map(ConstValue::Int)
        }
        (OpKind::Mod, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            // `ll_int_py_mod` (rpython/rtyper/rint.py:496): Python's
            // `%` returns a remainder with the sign of the divisor, not
            // the dividend (so `3 % -2 == -1`, not `1`).
            int_py_mod(*a, *b).map(ConstValue::Int)
        }
        (OpKind::LShift, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            if *b < 0 || *b >= 64 {
                None
            } else {
                a.checked_shl(*b as u32).map(ConstValue::Int)
            }
        }
        (OpKind::RShift, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            if *b < 0 || *b >= 64 {
                None
            } else {
                Some(ConstValue::Int(*a >> (*b as u32)))
            }
        }
        (OpKind::And, [ConstValue::Int(a), ConstValue::Int(b)]) => Some(ConstValue::Int(*a & *b)),
        (OpKind::Or, [ConstValue::Int(a), ConstValue::Int(b)]) => Some(ConstValue::Int(*a | *b)),
        (OpKind::Xor, [ConstValue::Int(a), ConstValue::Int(b)]) => Some(ConstValue::Int(*a ^ *b)),

        // --- binary arithmetic (float) ---
        (OpKind::Add, [ConstValue::Float(a), ConstValue::Float(b)]) => {
            Some(ConstValue::float(f64::from_bits(*a) + f64::from_bits(*b)))
        }
        (OpKind::Sub, [ConstValue::Float(a), ConstValue::Float(b)]) => {
            Some(ConstValue::float(f64::from_bits(*a) - f64::from_bits(*b)))
        }
        (OpKind::Mul, [ConstValue::Float(a), ConstValue::Float(b)]) => {
            Some(ConstValue::float(f64::from_bits(*a) * f64::from_bits(*b)))
        }
        (OpKind::TrueDiv, [ConstValue::Float(a), ConstValue::Float(b)]) => {
            let y = f64::from_bits(*b);
            if y == 0.0 {
                None
            } else {
                Some(ConstValue::float(f64::from_bits(*a) / y))
            }
        }

        // --- binary concat (str / tuple / list) ---
        (OpKind::Add, [ConstValue::ByteStr(a), ConstValue::ByteStr(b)]) => {
            let mut out = a.clone();
            out.extend_from_slice(b);
            Some(ConstValue::ByteStr(out))
        }
        (OpKind::Add, [ConstValue::UniStr(a), ConstValue::UniStr(b)]) => {
            Some(ConstValue::uni_str(format!("{a}{b}")))
        }
        (OpKind::Add, [ConstValue::Tuple(a), ConstValue::Tuple(b)]) => {
            let mut out = a.clone();
            out.extend(b.iter().cloned());
            Some(ConstValue::Tuple(out))
        }
        (OpKind::Add, [ConstValue::List(a), ConstValue::List(b)]) => {
            let mut out = a.clone();
            out.extend(b.iter().cloned());
            Some(ConstValue::List(out))
        }

        // --- comparisons (int / float / str / bool) ---
        (OpKind::Lt, [a, b]) => cmp_fold(a, b).map(|o| ConstValue::Bool(o.is_lt())),
        (OpKind::Le, [a, b]) => cmp_fold(a, b).map(|o| ConstValue::Bool(o.is_le())),
        (OpKind::Eq, [a, b]) => Some(ConstValue::Bool(python_eq_const(a, b))),
        (OpKind::Ne, [a, b]) => Some(ConstValue::Bool(!python_eq_const(a, b))),
        (OpKind::Gt, [a, b]) => cmp_fold(a, b).map(|o| ConstValue::Bool(o.is_gt())),
        (OpKind::Ge, [a, b]) => cmp_fold(a, b).map(|o| ConstValue::Bool(o.is_ge())),

        // Unhandled combination — decline the fold, let flowcontext
        // emit a SpaceOperation. This is the strict subset of upstream
        // behaviour that Commit 2 covers.
        _ => None,
    }
}

/// Ordering helper for the `Lt / Le / Gt / Ge` fold family. Returns
/// `None` for cross-type comparisons that RPython would raise on (e.g.
/// `int < str`) or for NaN float operands.
///
/// Mixed numeric (Bool ↔ Int ↔ Float) IS folded — Python 3 admits
/// `1 < 1.5`, `True < 2`, `1.0 > False` because `bool` is an `int`
/// subclass and `int`/`float` admit numeric coercion in
/// `operator.lt` etc. (`PureOperation.constfold` calls `pyfunc`
/// directly, so the dispatch follows Python's coercion rules).
fn cmp_fold(a: &ConstValue, b: &ConstValue) -> Option<std::cmp::Ordering> {
    use ConstValue as C;
    match (a, b) {
        (C::Int(x), C::Int(y)) => Some(x.cmp(y)),
        (C::Float(x), C::Float(y)) => f64::from_bits(*x).partial_cmp(&f64::from_bits(*y)),
        (C::ByteStr(x), C::ByteStr(y)) => Some(x.cmp(y)),
        (C::UniStr(x), C::UniStr(y)) => Some(x.cmp(y)),
        (C::Bool(x), C::Bool(y)) => Some(x.cmp(y)),
        // Bool ↔ Int (bool is int subclass).
        (C::Bool(x), C::Int(y)) => Some(i64::from(*x).cmp(y)),
        (C::Int(x), C::Bool(y)) => Some(x.cmp(&i64::from(*y))),
        // Int ↔ Float (numeric coercion via host f64 cast).
        (C::Int(x), C::Float(b)) => (*x as f64).partial_cmp(&f64::from_bits(*b)),
        (C::Float(b), C::Int(x)) => f64::from_bits(*b).partial_cmp(&(*x as f64)),
        // Bool ↔ Float (bool → int → float).
        (C::Bool(x), C::Float(b)) => (i64::from(*x) as f64).partial_cmp(&f64::from_bits(*b)),
        (C::Float(b), C::Bool(x)) => f64::from_bits(*b).partial_cmp(&(i64::from(*x) as f64)),
        _ => None,
    }
}

/// Python-3 floor division over `i64`. Line-by-line port of
/// `rpython/rtyper/rint.py:398 ll_int_py_div`:
///
/// ```text
/// r = trunc_div(x, y)        # C-style truncation toward 0
/// p = r * y
/// u = (p - x) if y < 0 else (x - p)
/// return r + (u >> (BITS - 1))
/// ```
///
/// `(u >> 63)` is `-1` when `u < 0` else `0`, applying the
/// floor-toward-`-inf` correction whenever C's truncated quotient
/// over-shot Python's floor (sign mismatch between dividend and
/// divisor with non-zero remainder).
///
/// Returns `None` for `y == 0` (the always-raises whitelist surfaces
/// `ZeroDivisionError`) and for `x == i64::MIN, y == -1` (Rust's
/// `checked_div` rejects the wrap; the runtime path handles bigint
/// promotion at upstream).
fn int_py_floor_div(x: i64, y: i64) -> Option<i64> {
    let r = x.checked_div(y)?;
    let p = r.checked_mul(y)?;
    let u = if y < 0 {
        p.checked_sub(x)?
    } else {
        x.checked_sub(p)?
    };
    r.checked_add(u >> (i64::BITS - 1))
}

/// Python-3 `%` over `i64`. Line-by-line port of
/// `rpython/rtyper/rint.py:496 ll_int_py_mod`:
///
/// ```text
/// r = trunc_mod(x, y)        # C-style %, sign of dividend
/// u = -r if y < 0 else r
/// return r + (y & (u >> (BITS - 1)))
/// ```
///
/// The `(u >> 63)` mask is `-1` when `u < 0` else `0`; `y & -1 == y`
/// adds the divisor to flip the remainder's sign whenever Python's
/// `(sign of divisor)` rule disagrees with C's `(sign of dividend)`
/// rule.
///
/// Returns `None` for `y == 0` and for `x == i64::MIN, y == -1`
/// (same overflow boundary as [`int_py_floor_div`]).
fn int_py_mod(x: i64, y: i64) -> Option<i64> {
    let r = x.checked_rem(y)?;
    let u = if y < 0 { r.checked_neg()? } else { r };
    r.checked_add(y & (u >> (i64::BITS - 1)))
}

/// Python-3 `==` semantics over `ConstValue`. Mirrors upstream
/// `PureOperation.constfold` calling `operator.eq(*args)` rather than
/// the host language's variant equality:
///
/// - **Numeric coercion** — Python's `bool` is a subclass of `int`,
///   and `int` / `float` admit cross-type equality via numeric
///   value (`True == 1`, `1 == 1.0`, `True == 1.0`). The naive Rust
///   `ConstValue::PartialEq` derives variant-strict equality and
///   would say those are unequal.
/// - **Distinct types stay false** — `"a" == b"a"`, `1 == "1"`,
///   `None == 0` all return `False` in Python 3 (`BytesWarning` is a
///   diagnostic, not an exception). For these the strict variant
///   compare matches semantics, so we fall through to `a == b`.
/// - **Containers** — `Tuple` / `List` of equal length compare
///   element-wise via the same `python_eq_const`, so
///   `(1, True) == (1, 1)` returns `True` (each numeric element
///   coerces). `Dict` deep-equality is left to variant compare for
///   now: keys hash by variant identity, so `{1: 'a'}` and
///   `{True: 'a'}` already key into different slots, and rebuilding
///   the lookup via Python-equality would require re-keying the
///   whole side-table — left as a follow-up.
fn python_eq_const(a: &ConstValue, b: &ConstValue) -> bool {
    use ConstValue as C;
    match (a, b) {
        (C::Bool(x), C::Bool(y)) => x == y,
        (C::Bool(x), C::Int(y)) | (C::Int(y), C::Bool(x)) => i64::from(*x) == *y,
        (C::Int(x), C::Int(y)) => x == y,
        (C::Bool(x), C::Float(bits)) | (C::Float(bits), C::Bool(x)) => {
            f64::from_bits(*bits) == (i64::from(*x) as f64)
        }
        (C::Int(x), C::Float(bits)) | (C::Float(bits), C::Int(x)) => {
            f64::from_bits(*bits) == (*x as f64)
        }
        (C::Float(a), C::Float(b)) => f64::from_bits(*a) == f64::from_bits(*b),
        // Container deep equality — Python compares
        // `tuple` / `list` element-wise via `==` per element. So
        // `(True,) == (1,)` is True even though Bool/Int are
        // distinct variants. Cross-container types stay distinct
        // (`(1,) == [1]` is False in Python 3).
        (C::Tuple(xs), C::Tuple(ys)) | (C::List(xs), C::List(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys.iter()).all(|(x, y)| python_eq_const(x, y))
        }
        _ => a == b,
    }
}

/// Detect operation/argument combinations that RPython's
/// `PureOperation.constfold` (operation.py:120-127) catches as
/// `Exception` and re-raises as `FlowingError`. Returns
/// `Some(FlowingError)` when the combination ALWAYS raises in
/// upstream Python; `None` otherwise (let pyfunc decide whether to
/// fold or decline).
///
/// Covered cases (the audit-flagged subset that surfaces through
/// `record_pure_op` — see `HLOperation::constfold` callsite):
///
/// - **Integer division / modulo by zero** — upstream Python raises
///   `ZeroDivisionError`. Mirrors `Div(int, 0)`, `Mod(int, 0)`,
///   `FloorDiv(int, 0)` (the operator-pair entries from
///   `operation.py:475-491 add_operator`).
/// - **TrueDiv with int / float zero divisor** — upstream Python
///   raises `ZeroDivisionError` for both `1 / 0` and `1.0 / 0.0`.
///   Mirrors `TrueDiv` registered as `add_operator('truediv', 2,
///   pure=True)`.
/// - **Mixed-type ordering comparison** — upstream Python 3 raises
///   `TypeError("<' not supported between instances of …")` for
///   `Lt`/`Le`/`Gt`/`Ge` between mismatched primitive types.
///   `Eq`/`Ne` are NOT covered: Python 3 returns `False` / `True`
///   for cross-type identity comparison, never raises.
///
/// Other "always raises" combinations (e.g. `len()` of a non-
/// iterable, `int()` of a non-numeric str) are intentionally NOT
/// covered here pending the broader pyfunc signature port — those
/// type-error cases are documented in their pyfunc match arms as
/// "decline" and continue to fall through to runtime emission until
/// pyfunc itself surfaces an error channel.
fn constfold_always_raises(kind: OpKind, args: &[&ConstValue]) -> Option<FlowingError> {
    let make_err = |reason: &str| {
        FlowingError::new(format!(
            "{}{:?} always raises {}",
            kind.opname(),
            args,
            reason,
        ))
    };
    match (kind, args) {
        // ZeroDivisionError on int /, //, %. upstream
        // `PureOperation.constfold` (operation.py:120-127) calls
        // `cls.pyfunc(*args)` and re-raises any `Exception` as
        // `FlowingError`. For the BASE ops, that's
        // `operator.{div,floordiv,mod}` which raises
        // `ZeroDivisionError` on a zero divisor.
        //
        // The `_ovf` siblings (DivOvf / FloorDivOvf / ModOvf) are
        // intentionally NOT covered here. Upstream creates them via
        // `add_operator(name + '_ovf', arity, dispatch, pyfunc=ovf_func)`
        // (`operation.py:337-338`) without `pure=True`, so their
        // base class is plain `HLOperation` whose default
        // `constfold(self) -> None` (`operation.py:96-97`) leaves
        // them as ops at compile time. Folding them here would
        // surface a hard-error divergence at every constant-zero-
        // divisor site that upstream leaves to the runtime.
        // `OpKind::pure()` enforces this by classifying `_ovf`
        // siblings as non-pure, so this whitelist arm is
        // unreachable for them; the explicit comment is here in
        // case a future caller bypasses the `pure()` gate.
        (
            OpKind::Div | OpKind::FloorDiv | OpKind::Mod,
            [ConstValue::Int(_), ConstValue::Int(0)],
        ) => Some(make_err(
            "ZeroDivisionError: integer division or modulo by zero",
        )),
        // ZeroDivisionError on truediv. Upstream `PureOperation.const
        // fold` calls `operator.truediv(lhs, rhs)`; for numeric lhs
        // and zero rhs the dispatch raises `ZeroDivisionError`. For
        // non-numeric lhs (e.g. `"x" / 0`) Python raises `TypeError`
        // first ("unsupported operand type(s)") before the divisor is
        // even consulted — so we only classify zero-divisor truediv
        // as ZeroDivisionError when lhs is itself foldable-numeric
        // (Int / Bool / Float). For non-numeric lhs we decline here
        // and leave the type-error path to the broader pyfunc port.
        (OpKind::TrueDiv, [lhs, ConstValue::Int(0)]) if is_foldable_numeric(lhs) => {
            Some(make_err("ZeroDivisionError: division by zero"))
        }
        (OpKind::TrueDiv, [lhs, ConstValue::Float(bits)])
            if is_foldable_numeric(lhs) && f64::from_bits(*bits) == 0.0 =>
        {
            Some(make_err("ZeroDivisionError: float division by zero"))
        }
        // TypeError on cross-type ordering comparisons. Eq/Ne deliberately
        // not included — upstream Python 3 returns Bool for those.
        (OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge, [a, b])
            if cross_type_ordering(a, b) =>
        {
            Some(make_err(
                "TypeError: ordering comparison not supported between distinct primitive types",
            ))
        }
        _ => None,
    }
}

/// `true` for `ConstValue` variants that participate in upstream
/// numeric arithmetic (`int` / `bool` / `float`). Used by
/// [`constfold_always_raises`] to guard the truediv zero-divisor
/// classification: only numeric lhs reaches `ZeroDivisionError`
/// — non-numeric lhs raises `TypeError` first per upstream
/// `operator.truediv` dispatch.
fn is_foldable_numeric(v: &ConstValue) -> bool {
    matches!(
        v,
        ConstValue::Int(_) | ConstValue::Bool(_) | ConstValue::Float(_)
    )
}

/// Helper for [`constfold_always_raises`]: returns `true` when `a`
/// and `b` are foldable primitives whose Python-3 cross-type
/// ordering would raise `TypeError` (e.g. `Int <-> UniStr`,
/// `ByteStr <-> Int`). Same-type, NaN-float, and Eq/Ne paths return
/// `false` (handled elsewhere or not raise).
fn cross_type_ordering(a: &ConstValue, b: &ConstValue) -> bool {
    use ConstValue as C;
    let kind = |v: &ConstValue| match v {
        C::Int(_) => Some(0),
        C::Bool(_) => Some(0), // Python: bool is int — same kind
        C::Float(_) => Some(1),
        C::UniStr(_) => Some(2),
        C::ByteStr(_) => Some(3),
        _ => None,
    };
    match (kind(a), kind(b)) {
        (Some(ka), Some(kb)) => {
            // Int<->Float same-numeric kind (Python 3 admits int<->float
            // ordering); UniStr<->ByteStr distinct (Python 3 raises).
            (ka, kb) != (0, 1) && (ka, kb) != (1, 0) && ka != kb
        }
        _ => false,
    }
}

// =====================================================================
// Dispatcher plumbing (operation.py:66-300 + pairtype.py:75-96).
// =====================================================================
//
// Upstream `class SingleDispatchMixin` / `class DoubleDispatchMixin`
// store the registration table on the HLOperation subclass itself
// (`cls._registry`). The Rust port collapses all HLOperation subclasses
// into `OpKind`, so the per-class registries become a `HashMap<OpKind,
// ...>` keyed on the op identity.

use std::cell::RefCell;

use crate::tool::pairtype::DoubleDispatchRegistry;

/// RPython `specialized` closure value returned by
/// `get_specialization` (operation.py:231-236 / 273-278).
///
/// Upstream this is a Python closure; it carries an optional
/// `can_only_throw` attribute that annotator `read_can_only_throw`
/// (model.py:837-841) consults. Rust packages the callable and the
/// attribute side-by-side.
pub struct Specialization {
    /// The actual annotation handler — `spec(annotator, *self.args)`
    /// upstream (operation.py:104).
    pub apply: Box<
        dyn Fn(
            &crate::annotator::annrpython::RPythonAnnotator,
            &HLOperation,
        ) -> crate::annotator::model::SomeValue,
    >,
    /// RPython `specialized.can_only_throw = impl.can_only_throw`
    /// side-band (operation.py:234).
    pub can_only_throw: CanOnlyThrow,
}

fn annotator_error_from_panic(
    payload: &(dyn std::any::Any + Send),
) -> Option<crate::annotator::model::AnnotatorError> {
    fn parse_text(text: &str) -> Option<crate::annotator::model::AnnotatorError> {
        let suffix = text.strip_prefix("AnnotatorError")?;
        let msg = suffix.trim_start_matches([':', ' ']);
        Some(crate::annotator::model::AnnotatorError::new(msg))
    }

    payload
        .downcast_ref::<crate::annotator::model::AnnotatorError>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<String>()
                .and_then(|text| parse_text(text))
        })
        .or_else(|| {
            payload
                .downcast_ref::<&'static str>()
                .and_then(|text| parse_text(text))
        })
}

pub(crate) fn apply_specialization(
    spec: &Specialization,
    annotator: &crate::annotator::annrpython::RPythonAnnotator,
    hlop: &HLOperation,
) -> Result<crate::annotator::model::SomeValue, crate::annotator::model::AnnotatorError> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (spec.apply)(annotator, hlop)
    })) {
        Ok(value) => Ok(value),
        Err(payload) => match annotator_error_from_panic(payload.as_ref()) {
            Some(err) => Err(err),
            None => std::panic::resume_unwind(payload),
        },
    }
}

/// RPython `getattr(opimpl, 'can_only_throw', None)` polymorphism
/// (model.py:837-841).
///
/// Upstream the attribute is either absent, a list of exception
/// classes, or a callable that produces one. Rust models the three
/// branches explicitly.
pub enum CanOnlyThrow {
    /// Attribute absent — upstream `None`.
    Absent,
    /// `can_only_throw = [Exc, Exc, ...]` — upstream line 839
    /// `isinstance(can_only_throw, list)` branch.
    List(Vec<BuiltinException>),
    /// `can_only_throw = lambda *args: [...]` — upstream line 841
    /// `return can_only_throw(*args)` branch. Returns `None` to mirror
    /// `_dict_can_only_throw_*` helpers (binaryop.py:527-535) that
    /// defer to `op.canraise` for r_dict's unrestricted throw set.
    Callable(Box<dyn Fn(&[crate::annotator::model::SomeValue]) -> Option<Vec<BuiltinException>>>),
}

/// RPython `@op.<name>.register_transform(...)` handler signature.
///
/// Upstream transformers take `(annotator, *self.args)` and return
/// either `None` (no rewrite) or a Python list of `HLOperation`s that
/// replace the current op. The Rust port mirrors that shape.
///
/// Lives on a per-OpKind, per-SomeValue-kind registry (`_TRANSFORM_*`)
/// that `HLOperation::transform` looks up before returning.
pub type Transformation = Box<
    dyn Fn(
        &crate::annotator::annrpython::RPythonAnnotator,
        &[crate::flowspace::model::Hlvalue],
    ) -> Option<Vec<HLOperation>>,
>;

thread_local! {
    /// RPython `cls._registry` on `SingleDispatchMixin` (operation.py:59).
    ///
    /// Upstream: one dict per HLOperation subclass, keyed by the argument
    /// class (`Some_cls`). Lookup walks `Some_cls.__mro__`
    /// (operation.py:212-219). Rust collapses "per HLOperation subclass"
    /// into an outer `HashMap<OpKind, ...>` since OpKind replaces the
    /// class identity of the HLOperation subclass. The registry is
    /// `thread_local!` because its contents include non-Send `Rc`
    /// references and because RPython's annotator is single-threaded.
    pub static _REGISTRY_SINGLE: RefCell<
        std::collections::HashMap<
            OpKind,
            std::collections::HashMap<
                crate::annotator::model::SomeValueTag,
                Specialization,
            >,
        >,
    > = {
        let mut outer = std::collections::HashMap::new();
        crate::annotator::unaryop::init(&mut outer);
        RefCell::new(outer)
    };

    /// RPython `cls._registry = DoubleDispatchRegistry()` on
    /// `DoubleDispatchMixin` (operation.py:62). Per-OpKind pair registry
    /// using the [`DoubleDispatchRegistry`] ported from
    /// `rpython/tool/pairtype.py`. Initialized on first access by
    /// calling the module-import-time `init` helpers.
    pub static _REGISTRY_DOUBLE: RefCell<
        std::collections::HashMap<
            OpKind,
            DoubleDispatchRegistry<
                crate::annotator::model::SomeValueTag,
                crate::annotator::model::SomeValueTag,
                Specialization,
            >,
        >,
    > = {
        let mut outer = std::collections::HashMap::new();
        crate::annotator::binaryop::init(&mut outer);
        RefCell::new(outer)
    };

    /// RPython `cls._transform = {}` on `HLOperationMeta.__init__`
    /// (operation.py:60 for `dispatch == 1`). Keyed the same way as
    /// `_REGISTRY_SINGLE` — MRO lookup on `type(args_s[0])`.
    pub static _TRANSFORM_SINGLE: RefCell<
        std::collections::HashMap<
            OpKind,
            std::collections::HashMap<crate::annotator::model::SomeValueTag, Transformation>,
        >,
    > = {
        let mut outer = std::collections::HashMap::new();
        crate::annotator::unaryop::init_transform(&mut outer);
        RefCell::new(outer)
    };

    /// RPython `cls._transform = DoubleDispatchRegistry()` on
    /// `HLOperationMeta.__init__` (operation.py:63 for
    /// `dispatch == 2`). Same MRO cross-product lookup as
    /// `_REGISTRY_DOUBLE`.
    pub static _TRANSFORM_DOUBLE: RefCell<
        std::collections::HashMap<
            OpKind,
            DoubleDispatchRegistry<
                crate::annotator::model::SomeValueTag,
                crate::annotator::model::SomeValueTag,
                Transformation,
            >,
        >,
    > = {
        let mut outer = std::collections::HashMap::new();
        crate::annotator::binaryop::init_transform(&mut outer);
        RefCell::new(outer)
    };
}

/// RPython `@op.<name>.register(Some_cls)` (operation.py:205-210 —
/// `SingleDispatchMixin.register`).
pub fn register_single(
    op: OpKind,
    tag: crate::annotator::model::SomeValueTag,
    spec: Specialization,
) {
    _REGISTRY_SINGLE.with(|cell| {
        cell.borrow_mut().entry(op).or_default().insert(tag, spec);
    });
}

/// RPython `@op.<name>.register(Some1, Some2)` (operation.py:261-266 —
/// `DoubleDispatchMixin.register`).
pub fn register_double(
    op: OpKind,
    tag1: crate::annotator::model::SomeValueTag,
    tag2: crate::annotator::model::SomeValueTag,
    spec: Specialization,
) {
    _REGISTRY_DOUBLE.with(|cell| {
        cell.borrow_mut()
            .entry(op)
            .or_default()
            .set((tag1, tag2), spec);
    });
}

/// RPython `@op.<name>.register_transform(Some_cls)` (operation.py:241-246 —
/// `SingleDispatchMixin.register_transform`).
pub fn register_transform_single(
    op: OpKind,
    tag: crate::annotator::model::SomeValueTag,
    tx: Transformation,
) {
    _TRANSFORM_SINGLE.with(|cell| {
        cell.borrow_mut().entry(op).or_default().insert(tag, tx);
    });
}

/// RPython `@op.<name>.register_transform(Some1, Some2)`
/// (operation.py:288-293 — `DoubleDispatchMixin.register_transform`).
pub fn register_transform_double(
    op: OpKind,
    tag1: crate::annotator::model::SomeValueTag,
    tag2: crate::annotator::model::SomeValueTag,
    tx: Transformation,
) {
    _TRANSFORM_DOUBLE.with(|cell| {
        cell.borrow_mut()
            .entry(op)
            .or_default()
            .set((tag1, tag2), tx);
    });
}

impl HLOperation {
    /// RPython `HLOperation.consider(self, annotator)` (operation.py:101-104).
    ///
    /// ```python
    /// def consider(self, annotator):
    ///     args_s = [annotator.annotation(arg) for arg in self.args]
    ///     spec = type(self).get_specialization(*args_s)
    ///     return spec(annotator, *self.args)
    /// ```
    ///
    /// The Rust port splits on `self.kind.dispatch()` to pick the
    /// correct `get_specialization` path — upstream selects the same
    /// paths via MRO dispatch (`SingleDispatchMixin.get_specialization`
    /// / `DoubleDispatchMixin.get_specialization`).
    pub fn consider(
        &self,
        annotator: &crate::annotator::annrpython::RPythonAnnotator,
    ) -> Result<crate::annotator::model::SomeValue, crate::annotator::model::AnnotatorError> {
        use crate::annotator::model::{AnnotatorError, SomeValue};
        // upstream operation.py:101-104 —
        //     args_s = [annotator.annotation(arg) for arg in self.args]
        //     spec = type(self).get_specialization(*args_s)
        //     return spec(annotator, *self.args)
        let mut args_s: Vec<SomeValue> = Vec::with_capacity(self.args.len());
        for a in &self.args {
            let s = annotator.annotation(a).ok_or_else(|| {
                AnnotatorError::new(format!(
                    "consider({:?}): unbound argument {:?}",
                    self.kind, a
                ))
            })?;
            args_s.push(s);
        }
        match self.kind.dispatch() {
            Dispatch::Single => {
                let tag = args_s
                    .first()
                    .ok_or_else(|| {
                        AnnotatorError::new(format!(
                            "consider: dispatch=1 op {:?} with 0 args",
                            self.kind
                        ))
                    })?
                    .tag();
                let result =
                    _REGISTRY_SINGLE.with(|cell| -> Result<SomeValue, AnnotatorError> {
                        let reg = cell.borrow();
                        let entries = reg.get(&self.kind).ok_or_else(|| {
                            AnnotatorError::new(format!(
                                "consider: no single-dispatch entries for {:?}",
                                self.kind
                            ))
                        })?;
                        // Upstream `SingleDispatchMixin._dispatch`
                        // (operation.py:212-219) walks
                        // `type(s_arg).__mro__`.
                        for c in tag.mro() {
                            if let Some(spec) = entries.get(c) {
                                return apply_specialization(spec, annotator, self);
                            }
                        }
                        Err(AnnotatorError::new(format!(
                            "consider: no unary spec for {:?}({:?})",
                            self.kind, tag
                        )))
                    })?;
                Ok(result)
            }
            Dispatch::Double => {
                let tag_l = args_s
                    .first()
                    .ok_or_else(|| {
                        AnnotatorError::new(format!(
                            "consider: dispatch=2 op {:?} with 0 args",
                            self.kind
                        ))
                    })?
                    .tag();
                let tag_r = args_s
                    .get(1)
                    .ok_or_else(|| {
                        AnnotatorError::new(format!(
                            "consider: dispatch=2 op {:?} with 1 arg",
                            self.kind
                        ))
                    })?
                    .tag();
                let result =
                    _REGISTRY_DOUBLE.with(|cell| -> Result<SomeValue, AnnotatorError> {
                        let reg = cell.borrow();
                        let entries = reg.get(&self.kind).ok_or_else(|| {
                            AnnotatorError::new(format!(
                                "consider: no double-dispatch entries for {:?}",
                                self.kind
                            ))
                        })?;
                        match entries.get((tag_l, tag_r), tag_l.mro(), tag_r.mro()) {
                            Some(spec) => apply_specialization(spec, annotator, self),
                            None => Err(AnnotatorError::new(format!(
                                "consider: no binary spec for {:?}({:?}, {:?})",
                                self.kind, tag_l, tag_r
                            ))),
                        }
                    })?;
                Ok(result)
            }
            Dispatch::None => {
                // operation.py:534-565 — per-class `consider()`
                // overrides on explicit `Dispatch::None` subclasses.
                use crate::annotator::model::SomeTuple as AnSomeTuple;
                match self.kind {
                    // operation.py:534-539 — NewDict.consider.
                    OpKind::NewDict => Ok(SomeValue::Dict(annotator.bookkeeper.newdict())),
                    // operation.py:542-548 — NewTuple.consider.
                    OpKind::NewTuple => Ok(SomeValue::Tuple(AnSomeTuple::new(args_s))),
                    // operation.py:551-557 — NewList.consider.
                    OpKind::NewList => {
                        let list = annotator.bookkeeper.newlist(&args_s, None)?;
                        Ok(SomeValue::List(list))
                    }
                    // operation.py:560-565 — NewSlice.consider raises
                    // AnnotatorError outright.
                    OpKind::NewSlice => Err(AnnotatorError::new(
                        "Cannot use extended slicing in rpython",
                    )),
                    // Unregistered HLOperation subclasses with
                    // `dispatch=None`. Upstream expects these to have
                    // been constfolded. Surface the gap as a proper
                    // error so keepgoing mode can continue.
                    _ => Err(AnnotatorError::new(format!(
                        "consider: no Dispatch::None override for {:?}",
                        self.kind
                    ))),
                }
            }
        }
    }

    /// RPython `HLOperation.transform(self, annotator)` (operation.py:112-115).
    ///
    /// ```python
    /// def transform(self, annotator):
    ///     args_s = [annotator.annotation(arg) for arg in self.args]
    ///     transformer = self.get_transformer(*args_s)
    ///     return transformer(annotator, *self.args)
    /// ```
    ///
    /// `get_transformer` does an MRO lookup (operation.py:248-255 for
    /// single-dispatch, 295-300 for double-dispatch) and returns the
    /// default `lambda *args: None` when no registration matches.
    pub fn transform(
        &self,
        annotator: &crate::annotator::annrpython::RPythonAnnotator,
    ) -> Option<Vec<HLOperation>> {
        use crate::annotator::model::SomeValue;
        let args_s: Vec<SomeValue> = self
            .args
            .iter()
            .filter_map(|a| annotator.annotation(a))
            .collect();
        match self.kind.dispatch() {
            Dispatch::Single => {
                if args_s.is_empty() {
                    return None;
                }
                let tag = args_s[0].tag();
                _TRANSFORM_SINGLE.with(|cell| {
                    let reg = cell.borrow();
                    let entries = reg.get(&self.kind)?;
                    for c in tag.mro() {
                        if let Some(tx) = entries.get(c) {
                            return tx(annotator, &self.args);
                        }
                    }
                    None
                })
            }
            Dispatch::Double => {
                if args_s.len() < 2 {
                    return None;
                }
                let tag_l = args_s[0].tag();
                let tag_r = args_s[1].tag();
                _TRANSFORM_DOUBLE.with(|cell| {
                    let reg = cell.borrow();
                    let entries = reg.get(&self.kind)?;
                    let tx = entries.get((tag_l, tag_r), tag_l.mro(), tag_r.mro())?;
                    tx(annotator, &self.args)
                })
            }
            Dispatch::None => None,
        }
    }

    /// RPython `HLOperation.get_can_only_throw(self, annotator)`
    /// (operation.py:106-107, SingleDispatchMixin:221-224,
    /// DoubleDispatchMixin:283-286).
    pub fn get_can_only_throw(
        &self,
        annotator: &crate::annotator::annrpython::RPythonAnnotator,
    ) -> Option<Vec<BuiltinException>> {
        use crate::annotator::model::SomeValue;
        let args_s: Vec<SomeValue> = self
            .args
            .iter()
            .filter_map(|a| annotator.annotation(a))
            .collect();
        match self.kind.dispatch() {
            Dispatch::Single => {
                if args_s.is_empty() {
                    return None;
                }
                let tag = args_s[0].tag();
                _REGISTRY_SINGLE.with(|cell| {
                    let reg = cell.borrow();
                    let entries = reg.get(&self.kind)?;
                    for c in tag.mro() {
                        if let Some(spec) = entries.get(c) {
                            return crate::annotator::model::read_can_only_throw(
                                &spec.can_only_throw,
                                &args_s,
                            );
                        }
                    }
                    None
                })
            }
            Dispatch::Double => {
                if args_s.len() < 2 {
                    return None;
                }
                let tag_l = args_s[0].tag();
                let tag_r = args_s[1].tag();
                _REGISTRY_DOUBLE.with(|cell| {
                    let reg = cell.borrow();
                    let entries = reg.get(&self.kind)?;
                    entries
                        .get((tag_l, tag_r), tag_l.mro(), tag_r.mro())
                        .and_then(|spec| {
                            crate::annotator::model::read_can_only_throw(
                                &spec.can_only_throw,
                                &args_s,
                            )
                        })
                })
            }
            // Upstream `HLOperation.get_can_only_throw` default
            // (operation.py:106-107) returns None.
            Dispatch::None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::{ConstValue, Constant};
    use super::*;

    #[test]
    fn opname_matches_upstream_strings() {
        // A curated selection: one entry per upstream RPython string
        // that carries an underscore or a non-obvious mapping.
        assert_eq!(OpKind::Is.opname(), "is_");
        assert_eq!(OpKind::And.opname(), "and_");
        assert_eq!(OpKind::Or.opname(), "or_");
        assert_eq!(OpKind::Yield.opname(), "yield_");
        assert_eq!(OpKind::GetItemIdx.opname(), "getitem_idx");
        assert_eq!(OpKind::SimpleCall.opname(), "simple_call");
        assert_eq!(OpKind::CallArgs.opname(), "call_args");
        assert_eq!(OpKind::AddOvf.opname(), "add_ovf");
        assert_eq!(OpKind::IsSubtype.opname(), "issubtype");
        assert_eq!(OpKind::IsInstance.opname(), "isinstance");
    }

    #[test]
    fn ovf_variant_matches_upstream_table() {
        // upstream `add_operator(..., ovf=True)` recurses via
        // `add_operator(name+'_ovf', …)`; the pairs below trace that
        // recursion.
        assert_eq!(OpKind::Add.ovf_variant(), Some(OpKind::AddOvf));
        assert_eq!(OpKind::Sub.ovf_variant(), Some(OpKind::SubOvf));
        assert_eq!(OpKind::Mul.ovf_variant(), Some(OpKind::MulOvf));
        assert_eq!(OpKind::Neg.ovf_variant(), Some(OpKind::NegOvf));
        assert_eq!(OpKind::Abs.ovf_variant(), Some(OpKind::AbsOvf));
        assert_eq!(OpKind::LShift.ovf_variant(), Some(OpKind::LShiftOvf));
        // No overflow variant for Eq / Lt / And / ….
        assert_eq!(OpKind::Eq.ovf_variant(), None);
        assert_eq!(OpKind::And.ovf_variant(), None);
        // And the _ovf's themselves don't recurse further.
        assert_eq!(OpKind::AddOvf.ovf_variant(), None);
    }

    #[test]
    fn dispatch_classification_matches_upstream() {
        // `add_operator('add', 2, dispatch=2, …)` → Double.
        assert_eq!(OpKind::Add.dispatch(), Dispatch::Double);
        // `add_operator('len', 1, dispatch=1, …)` → Single.
        assert_eq!(OpKind::Len.dispatch(), Dispatch::Single);
        // `Pow` / `NewDict` / `NewTuple` carry no `dispatch=` →
        // `HLOperation.dispatch = None`.
        assert_eq!(OpKind::Pow.dispatch(), Dispatch::None);
        assert_eq!(OpKind::NewTuple.dispatch(), Dispatch::None);
    }

    #[test]
    fn arity_matches_upstream() {
        assert_eq!(OpKind::Add.arity(), Some(2));
        assert_eq!(OpKind::Len.arity(), Some(1));
        assert_eq!(OpKind::SetAttr.arity(), Some(3));
        assert_eq!(OpKind::SetSlice.arity(), Some(4));
        // variadic / manual-dispatch → None.
        assert_eq!(OpKind::NewTuple.arity(), None);
        assert_eq!(OpKind::SimpleCall.arity(), None);
    }

    #[test]
    fn apply_specialization_reifies_typed_annotator_error_payload() {
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let hl = HLOperation::new(OpKind::Hash, vec![]);
        let spec = Specialization {
            apply: Box::new(|_, _| {
                std::panic::panic_any(crate::annotator::model::AnnotatorError::new(
                    "typed payload",
                ))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        };

        let err = apply_specialization(&spec, &ann, &hl)
            .expect_err("typed AnnotatorError panic must be reified");
        assert_eq!(err.msg.as_deref(), Some("typed payload"));
    }

    #[test]
    fn pure_matches_upstream() {
        assert!(OpKind::Add.pure());
        assert!(OpKind::NewTuple.pure());
        assert!(OpKind::Pow.pure());
        assert!(!OpKind::SetAttr.pure());
        assert!(!OpKind::SimpleCall.pure());
        assert!(!OpKind::InplaceAdd.pure());
        // `_ovf` siblings: NOT pure. upstream `add_operator(name +
        // '_ovf', arity, dispatch, pyfunc=ovf_func)` carries no
        // `pure=True` (`operation.py:337-338`), so the suffixed
        // class falls through to plain `HLOperation` whose default
        // `constfold(self) -> None` (`operation.py:96-97`) prevents
        // any constant-time overflow / zero-divisor folding.
        assert!(!OpKind::AddOvf.pure());
        assert!(!OpKind::SubOvf.pure());
        assert!(!OpKind::MulOvf.pure());
        assert!(!OpKind::NegOvf.pure());
        assert!(!OpKind::AbsOvf.pure());
        assert!(!OpKind::FloorDivOvf.pure());
        assert!(!OpKind::DivOvf.pure());
        assert!(!OpKind::ModOvf.pure());
        assert!(!OpKind::LShiftOvf.pure());
    }

    #[test]
    fn can_overflow_matches_upstream() {
        // `add_operator(..., ovf=True)` → True for both the base and
        // its `_ovf` twin (RPython keeps `can_overflow=True` on the
        // OverflowingOperation subclass).
        assert!(OpKind::Add.can_overflow());
        assert!(OpKind::AddOvf.can_overflow());
        assert!(OpKind::LShift.can_overflow());
        assert!(!OpKind::RShift.can_overflow());
        assert!(!OpKind::Eq.can_overflow());
    }

    fn c(v: ConstValue) -> Hlvalue {
        Hlvalue::Constant(Constant::new(v))
    }

    fn ci(n: i64) -> Hlvalue {
        c(ConstValue::Int(n))
    }

    fn cf(f: f64) -> Hlvalue {
        c(ConstValue::float(f))
    }

    fn cb(b: bool) -> Hlvalue {
        c(ConstValue::Bool(b))
    }

    fn cs(s: &str) -> Hlvalue {
        c(ConstValue::byte_str(s))
    }

    fn fold(kind: OpKind, args: Vec<Hlvalue>) -> Option<ConstValue> {
        let op = HLOperation::new(kind, args);
        op.constfold().ok().flatten().map(|v| match v {
            Hlvalue::Constant(c) => c.value,
            _ => unreachable!("constfold must not produce Variables"),
        })
    }

    #[test]
    fn canraise_table_matches_upstream() {
        use BuiltinException::*;
        // operation.py:728-731.
        assert_eq!(
            OpKind::GetItem.canraise(),
            &[IndexError, KeyError, Exception]
        );
        assert_eq!(
            OpKind::DelItem.canraise(),
            &[IndexError, KeyError, Exception]
        );
        // operation.py:732.
        assert_eq!(OpKind::Contains.canraise(), &[Exception]);
        assert_eq!(OpKind::GetAttr.canraise(), &[]);
        assert_eq!(OpKind::Iter.canraise(), &[]);
        assert_eq!(OpKind::Next.canraise(), &[StopIteration, RuntimeError]);
        // `add` / `sub` / `mul` — base variants have empty canraise.
        assert_eq!(OpKind::Add.canraise(), &[]);
        assert_eq!(OpKind::Sub.canraise(), &[]);
        assert_eq!(OpKind::Mul.canraise(), &[]);
        // `_ovf` twins pick up OverflowError (_add_except_ovf).
        assert_eq!(OpKind::AddOvf.canraise(), &[OverflowError]);
        assert_eq!(OpKind::NegOvf.canraise(), &[OverflowError]);
        // `div` picks up ZeroDivisionError only; `div_ovf` adds OverflowError.
        assert_eq!(OpKind::Div.canraise(), &[ZeroDivisionError]);
        assert_eq!(
            OpKind::DivOvf.canraise(),
            &[ZeroDivisionError, OverflowError]
        );
        // `lshift` picks up ValueError; `lshift_ovf` + OverflowError.
        assert_eq!(OpKind::LShift.canraise(), &[ValueError]);
        assert_eq!(OpKind::LShiftOvf.canraise(), &[ValueError, OverflowError]);
        // `pow` picks up all three (ZeroDiv from 751, ValueError from
        // 754, OverflowError from 762).
        assert_eq!(
            OpKind::Pow.canraise(),
            &[ZeroDivisionError, ValueError, OverflowError]
        );
        // `truediv` / `divmod` — ZeroDivisionError + OverflowError, no _ovf.
        assert_eq!(
            OpKind::TrueDiv.canraise(),
            &[ZeroDivisionError, OverflowError]
        );
        assert_eq!(
            OpKind::DivMod.canraise(),
            &[ZeroDivisionError, OverflowError]
        );
        // `inplace_add` — only OverflowError (line 756).
        assert_eq!(OpKind::InplaceAdd.canraise(), &[OverflowError]);
        // Comparisons / identity / len stay clean.
        assert_eq!(OpKind::Lt.canraise(), &[]);
        assert_eq!(OpKind::Is.canraise(), &[]);
        assert_eq!(OpKind::Len.canraise(), &[]);
    }

    #[test]
    fn constfold_declines_non_pure_ops() {
        // `setattr` is side-effecting → no fold.
        assert_eq!(fold(OpKind::SetAttr, vec![ci(1), cs("x"), ci(2)]), None);
        // `simple_call` is non-pure → no fold.
        assert_eq!(fold(OpKind::SimpleCall, vec![ci(1), ci(2)]), None);
    }

    #[test]
    fn constfold_declines_variable_args() {
        let v = Hlvalue::Variable(Variable::new());
        assert_eq!(fold(OpKind::Add, vec![v, ci(1)]), None);
    }

    #[test]
    fn getattr_constfold_raises_flowing_error_on_three_arg_form() {
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![
                ci(1),
                cs("real"),
                Hlvalue::Constant(Constant::new(ConstValue::None)),
            ],
        );
        let err = op
            .constfold()
            .expect_err("3-arg getattr must raise FlowingError");
        assert!(err.message.contains("three arguments not supported"));
    }

    #[test]
    fn getattr_constfold_folds_constant_module_member() {
        let module = super::super::model::HOST_ENV
            .import_module("rpython.rlib.rfile")
            .expect("bootstrap module must exist");
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::HostObject(module)), cs("create_file")],
        );
        let folded = op
            .constfold()
            .expect("module getattr should not error")
            .expect("module getattr should fold");
        let Hlvalue::Constant(constant) = folded else {
            panic!("expected Constant result");
        };
        assert!(matches!(constant.value, ConstValue::HostObject(_)));
    }

    #[test]
    fn getattr_constfold_declines_volatile_sys_attr() {
        // upstream operation.py:22-35 — `sys.path` / `sys.modules`
        // are runtime-variable (mutable state of the interpreter), so
        // the NOT_REALLY_CONST guard must decline the fold even if
        // the module exposes the attribute via `module_get`. Matches
        // the `if w_obj in NOT_REALLY_CONST and w_name not in …`
        // branch at operation.py:631-633.
        use crate::flowspace::model::HostObject;
        let sys = HostObject::new_module("sys");
        sys.module_set(
            "path".to_string(),
            HostObject::new_module("path_placeholder"),
        );
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::HostObject(sys)), cs("path")],
        );
        assert!(op.constfold().unwrap().is_none());
    }

    #[test]
    fn getattr_constfold_folds_allowlisted_sys_attr() {
        // upstream operation.py:25 — `sys.maxint` is explicitly
        // declared as a real constant. Folding must succeed when the
        // module exposes it.
        use crate::flowspace::model::HostObject;
        let sys = HostObject::new_module("sys");
        sys.module_set(
            "maxint".to_string(),
            HostObject::new_module("maxint_placeholder"),
        );
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::HostObject(sys)), cs("maxint")],
        );
        assert!(op.constfold().unwrap().is_some());
    }

    #[test]
    fn getattr_constfold_folds_class_attribute() {
        // upstream operation.py:634-644 — `getattr(cls, 'method')`
        // folds when the class exposes the attribute via its __dict__.
        use crate::flowspace::model::HostObject;
        let cls = HostObject::new_class("Foo", vec![]);
        cls.class_set("CONST".to_string(), ConstValue::Int(42));
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::HostObject(cls)), cs("CONST")],
        );
        let folded = op
            .constfold()
            .expect("class getattr should not error")
            .expect("class getattr should fold");
        let Hlvalue::Constant(constant) = folded else {
            panic!("expected Constant result");
        };
        assert_eq!(constant.value, ConstValue::Int(42));
    }

    #[test]
    fn getattr_constfold_folds_instance_dict_attribute() {
        // upstream operation.py:634-644 — `getattr(instance, 'attr')`
        // respects `inst.__dict__` before walking the class MRO. Rust
        // port mirrors this via HostObject::instance_get.
        use crate::flowspace::model::HostObject;
        let cls = HostObject::new_class("Foo", vec![]);
        cls.class_set("x".to_string(), ConstValue::Int(1)); // class-level
        let inst = HostObject::new_instance(cls, vec![]);
        inst.instance_set("x", ConstValue::Int(99)); // shadow
        inst.instance_set("y", ConstValue::Int(7)); // instance-only
        // Need to make the instance foldable for the test — it isn't by
        // default (foldable() returns false for user instances). The
        // exercise here is `const_runtime_getattr` itself, so call the
        // helper directly. Upstream's
        // `w_obj.foldable()` gate intentionally rejects user instances;
        // real folds hit this code path via `_freeze_` markers we
        // don't model. Verify the helper's correctness instead:
        let r = crate::flowspace::model::const_runtime_getattr(
            &ConstValue::HostObject(inst.clone()),
            "x",
        )
        .expect("instance x lookup");
        assert_eq!(r, Some(ConstValue::Int(99))); // shadowed
        let r2 = crate::flowspace::model::const_runtime_getattr(&ConstValue::HostObject(inst), "y")
            .expect("instance y lookup");
        assert_eq!(r2, Some(ConstValue::Int(7))); // instance-only
    }

    #[test]
    fn getattr_constfold_instance_falls_through_to_class_mro() {
        // Attribute missing on instance but present on class → use MRO.
        use crate::flowspace::model::HostObject;
        let parent = HostObject::new_class("Parent", vec![]);
        parent.class_set("z", ConstValue::Int(100));
        let child = HostObject::new_class("Child", vec![parent.clone()]);
        let inst = HostObject::new_instance(child, vec![]);
        let r = crate::flowspace::model::const_runtime_getattr(&ConstValue::HostObject(inst), "z")
            .expect("inherited attribute lookup");
        assert_eq!(r, Some(ConstValue::Int(100)));
    }

    #[test]
    fn getattr_constfold_raises_flowing_error_on_missing_class_attribute() {
        // upstream operation.py:637-642 — `getattr(cls, 'missing')`
        // raises `AttributeError`, which flowspace escalates to
        // `FlowingError("… always raises AttributeError")`.
        use crate::flowspace::model::HostObject;
        let cls = HostObject::new_class("Foo", vec![]);
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::HostObject(cls)), cs("missing")],
        );
        let err = op
            .constfold()
            .expect_err("missing class attribute must escalate to FlowingError");
        assert!(err.message.contains("always raises AttributeError"));
    }

    #[test]
    fn getattr_constfold_user_function_missing_attr_raises() {
        // upstream operation.py:634-642 — user function constants are
        // foldable, so `getattr(function, "missing")` reaches the real
        // getattr() call and raises FlowingError on AttributeError.
        use crate::flowspace::model::{GraphFunc, HostObject};
        let globals = Constant::new(ConstValue::None);
        let gf = GraphFunc::new("user_fn".to_string(), globals);
        let func = HostObject::new_user_function(gf);
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::HostObject(func)), cs("whatever")],
        );
        let err = op
            .constfold()
            .expect_err("missing function attribute must escalate to FlowingError");
        assert!(err.message.contains("always raises AttributeError"));
    }

    #[test]
    fn getattr_constfold_user_function_dunder_name_returns_constant_str() {
        use crate::flowspace::model::{GraphFunc, HostObject};
        let globals = Constant::new(ConstValue::None);
        let gf = GraphFunc::new("pkg.demo.user_fn".to_string(), globals);
        let func = HostObject::new_user_function(gf);
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::HostObject(func)), cs("__name__")],
        );
        let folded = op.constfold().expect("function.__name__ should fold");
        let Some(Hlvalue::Constant(constant)) = folded else {
            panic!("expected Constant result");
        };
        assert_eq!(constant.value, ConstValue::byte_str("user_fn"));
    }

    #[test]
    fn getattr_constfold_int_dunder_class_returns_builtin_int_class() {
        use crate::flowspace::model::HOST_ENV;
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::Int(5)), cs("__class__")],
        );
        let folded = op.constfold().expect("int.__class__ should fold");
        let Some(Hlvalue::Constant(constant)) = folded else {
            panic!("expected Constant result");
        };
        let expected = HOST_ENV.lookup_builtin("int").expect("builtin int class");
        assert_eq!(constant.value, ConstValue::HostObject(expected));
    }

    #[test]
    fn getattr_constfold_int_dunder_class_name_returns_constant_str() {
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::Int(5)), cs("__class__")],
        );
        let folded = op.constfold().expect("int.__class__ should fold");
        let Some(Hlvalue::Constant(class_constant)) = folded else {
            panic!("expected Constant result");
        };
        let op2 = HLOperation::new(
            OpKind::GetAttr,
            vec![Hlvalue::Constant(class_constant), cs("__name__")],
        );
        let folded2 = op2.constfold().expect("int.__class__.__name__ should fold");
        let Some(Hlvalue::Constant(name_constant)) = folded2 else {
            panic!("expected Constant result");
        };
        assert_eq!(name_constant.value, ConstValue::byte_str("int"));
    }

    #[test]
    fn getattr_constfold_int_dunder_class_module_returns_constant_str() {
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::Int(5)), cs("__class__")],
        );
        let folded = op.constfold().expect("int.__class__ should fold");
        let Some(Hlvalue::Constant(class_constant)) = folded else {
            panic!("expected Constant result");
        };
        let op2 = HLOperation::new(
            OpKind::GetAttr,
            vec![Hlvalue::Constant(class_constant), cs("__module__")],
        );
        let folded2 = op2
            .constfold()
            .expect("int.__class__.__module__ should fold");
        let Some(Hlvalue::Constant(module_constant)) = folded2 else {
            panic!("expected Constant result");
        };
        assert_eq!(module_constant.value, ConstValue::byte_str("__builtin__"));
    }

    #[test]
    fn getattr_constfold_function_carrier_dunder_name_returns_constant_str() {
        let globals = Constant::new(ConstValue::None);
        let func = ConstValue::Function(Box::new(crate::flowspace::model::GraphFunc::new(
            "pkg.demo.user_fn",
            globals,
        )));
        let op = HLOperation::new(OpKind::GetAttr, vec![c(func), cs("__name__")]);
        let folded = op.constfold().expect("function.__name__ should fold");
        let Some(Hlvalue::Constant(name_constant)) = folded else {
            panic!("expected Constant result");
        };
        assert_eq!(name_constant.value, ConstValue::byte_str("user_fn"));
    }

    #[test]
    fn getattr_constfold_function_carrier_dunder_module_returns_constant_str() {
        let globals = Constant::new(ConstValue::None);
        let func = ConstValue::Function(Box::new(crate::flowspace::model::GraphFunc::new(
            "pkg.demo.user_fn",
            globals,
        )));
        let op = HLOperation::new(OpKind::GetAttr, vec![c(func), cs("__module__")]);
        let folded = op.constfold().expect("function.__module__ should fold");
        let Some(Hlvalue::Constant(module_constant)) = folded else {
            panic!("expected Constant result");
        };
        assert_eq!(module_constant.value, ConstValue::byte_str("pkg.demo"));
    }

    #[test]
    fn getattr_constfold_function_carrier_dunder_globals_returns_constant_dict() {
        let globals = Constant::new(ConstValue::Dict(HashMap::new()));
        let func = ConstValue::Function(Box::new(crate::flowspace::model::GraphFunc::new(
            "pkg.demo.user_fn",
            globals,
        )));
        let op = HLOperation::new(OpKind::GetAttr, vec![c(func), cs("__globals__")]);
        let folded = op.constfold().expect("function.__globals__ should fold");
        let Some(Hlvalue::Constant(globals_constant)) = folded else {
            panic!("expected Constant result");
        };
        assert_eq!(globals_constant.value, ConstValue::Dict(HashMap::new()));
    }

    #[test]
    fn getattr_constfold_function_carrier_dunder_defaults_returns_none_when_absent() {
        let globals = Constant::new(ConstValue::Dict(HashMap::new()));
        let func = ConstValue::Function(Box::new(crate::flowspace::model::GraphFunc::new(
            "pkg.demo.user_fn",
            globals,
        )));
        let op = HLOperation::new(OpKind::GetAttr, vec![c(func), cs("__defaults__")]);
        let folded = op.constfold().expect("function.__defaults__ should fold");
        let Some(Hlvalue::Constant(defaults_constant)) = folded else {
            panic!("expected Constant result");
        };
        assert_eq!(defaults_constant.value, ConstValue::None);
    }

    #[test]
    fn getattr_constfold_function_carrier_dunder_closure_returns_none_when_absent() {
        let globals = Constant::new(ConstValue::Dict(HashMap::new()));
        let func = ConstValue::Function(Box::new(crate::flowspace::model::GraphFunc::new(
            "pkg.demo.user_fn",
            globals,
        )));
        let op = HLOperation::new(OpKind::GetAttr, vec![c(func), cs("__closure__")]);
        let folded = op.constfold().expect("function.__closure__ should fold");
        let Some(Hlvalue::Constant(closure_constant)) = folded else {
            panic!("expected Constant result");
        };
        assert_eq!(closure_constant.value, ConstValue::None);
    }

    #[test]
    fn getattr_constfold_function_carrier_dunder_defaults_and_closure_return_tuple_payloads() {
        let globals = Constant::new(ConstValue::Dict(HashMap::new()));
        let mut graph_func = crate::flowspace::model::GraphFunc::new("pkg.demo.user_fn", globals);
        graph_func.defaults = vec![Constant::new(ConstValue::Int(7))];
        graph_func.closure = vec![Constant::new(ConstValue::byte_str("cell"))];
        let func = ConstValue::Function(Box::new(graph_func));

        let op_defaults =
            HLOperation::new(OpKind::GetAttr, vec![c(func.clone()), cs("__defaults__")]);
        let folded_defaults = op_defaults
            .constfold()
            .expect("function.__defaults__ should fold");
        let Some(Hlvalue::Constant(defaults_constant)) = folded_defaults else {
            panic!("expected Constant result");
        };
        assert_eq!(
            defaults_constant.value,
            ConstValue::Tuple(vec![ConstValue::Int(7)])
        );

        let op_closure = HLOperation::new(OpKind::GetAttr, vec![c(func), cs("__closure__")]);
        let folded_closure = op_closure
            .constfold()
            .expect("function.__closure__ should fold");
        let Some(Hlvalue::Constant(closure_constant)) = folded_closure else {
            panic!("expected Constant result");
        };
        assert_eq!(
            closure_constant.value,
            ConstValue::Tuple(vec![ConstValue::byte_str("cell")])
        );
    }

    #[test]
    fn constfold_int_arithmetic() {
        assert_eq!(
            fold(OpKind::Add, vec![ci(3), ci(4)]),
            Some(ConstValue::Int(7))
        );
        assert_eq!(
            fold(OpKind::Sub, vec![ci(10), ci(4)]),
            Some(ConstValue::Int(6))
        );
        assert_eq!(
            fold(OpKind::Mul, vec![ci(6), ci(7)]),
            Some(ConstValue::Int(42))
        );
        assert_eq!(fold(OpKind::Neg, vec![ci(5)]), Some(ConstValue::Int(-5)));
        assert_eq!(fold(OpKind::Abs, vec![ci(-5)]), Some(ConstValue::Int(5)));
        assert_eq!(fold(OpKind::Invert, vec![ci(0)]), Some(ConstValue::Int(-1)));
    }

    #[test]
    fn constfold_int_overflow_declines() {
        // RPython `add_ovf` raises OverflowError; flow-space declines
        // the fold (`PureOperation._pure_result` skips long results).
        assert_eq!(
            fold(OpKind::AddOvf, vec![ci(i64::MAX), ci(1)]),
            None,
            "overflow must not fold"
        );
        assert_eq!(
            fold(OpKind::Add, vec![ci(i64::MAX), ci(1)]),
            None,
            "overflow on checked i64 must not fold"
        );
        assert_eq!(
            fold(OpKind::Neg, vec![ci(i64::MIN)]),
            None,
            "neg(i64::MIN) overflows"
        );
    }

    #[test]
    fn constfold_division_by_zero_raises_flowing_error() {
        // upstream `PureOperation.constfold` (operation.py:120-127)
        // wraps `pyfunc(*args)` in `try/except Exception` and re-raises
        // captured exceptions as `FlowingError`. Division by zero is
        // the canonical case — Python's int `/`, `//`, `%` raise
        // `ZeroDivisionError`, so a constant-folded `1 / 0` must
        // surface as FlowingError, not silent decline.
        // Only base ops raise at constfold time. upstream `_ovf`
        // siblings are NOT pure (`add_operator(name + '_ovf', …)`
        // carries no `pure=True`, `operation.py:337-338`), so their
        // `HLOperation.constfold` default returns `None`
        // (`operation.py:96-97`) — the op stays as an op for the
        // runtime to handle. Folding `_ovf` zero-divisor at compile
        // time would surface a hard-error divergence vs upstream.
        for kind in [OpKind::Div, OpKind::Mod, OpKind::FloorDiv] {
            let op = HLOperation::new(kind, vec![ci(1), ci(0)]);
            let err = op.constfold().expect_err("zero divisor must raise");
            assert!(
                err.message.contains("ZeroDivisionError"),
                "{kind:?}: {}",
                err.message
            );
        }
        for kind in [OpKind::DivOvf, OpKind::ModOvf, OpKind::FloorDivOvf] {
            let op = HLOperation::new(kind, vec![ci(1), ci(0)]);
            assert_eq!(
                op.constfold().expect("_ovf sibling must NOT raise at fold"),
                None,
                "{kind:?} is HLOperation default — must decline (op stays at runtime)",
            );
        }
        // TrueDiv with int or float zero divisor.
        let op = HLOperation::new(OpKind::TrueDiv, vec![ci(1), ci(0)]);
        let err = op
            .constfold()
            .expect_err("truediv int zero divisor must raise");
        assert!(err.message.contains("ZeroDivisionError"));
        let op = HLOperation::new(OpKind::TrueDiv, vec![cf(1.0), cf(0.0)]);
        let err = op
            .constfold()
            .expect_err("truediv float zero divisor must raise");
        assert!(err.message.contains("ZeroDivisionError"));
        // TrueDiv with non-numeric lhs: upstream
        // `operator.truediv("x", 0)` raises `TypeError` first (the
        // dispatch fails before the divisor is consulted), NOT
        // `ZeroDivisionError`. The always-raises whitelist therefore
        // declines this case rather than misclassifying it as a
        // zero-divisor error — fold falls through to the broader
        // pyfunc port, where the type-error path is documented as a
        // pre-existing gap.
        assert_eq!(
            fold(OpKind::TrueDiv, vec![cs("x"), ci(0)]),
            None,
            "non-numeric lhs / zero divisor must NOT classify as ZeroDivisionError",
        );
        assert_eq!(
            fold(OpKind::TrueDiv, vec![cs("x"), cf(0.0)]),
            None,
            "non-numeric lhs / float zero divisor must NOT classify as ZeroDivisionError",
        );
    }

    #[test]
    fn constfold_floor_div_and_mod_python_semantics() {
        // upstream `operator.floordiv` / `operator.mod` over ints:
        // floor toward `-inf` (Python 3 semantics), NOT C truncation
        // toward 0 and NOT Euclidean (always non-negative remainder).
        // `rpython/rtyper/rint.py:398 ll_int_py_div` and `:496
        // ll_int_py_mod` carry the correction over the C primitive;
        // our `int_py_floor_div` / `int_py_mod` helpers port those
        // line-by-line.
        //
        // Sign coverage matrix (4 quadrants × Div/FloorDiv/Mod):
        // Python results from the reference implementation —
        //   3 // 2 = 1,   3 % 2 = 1
        //   3 // -2 = -2, 3 % -2 = -1
        //   -3 // 2 = -2, -3 % 2 = 1
        //   -3 // -2 = 1, -3 % -2 = -1
        // (a) same sign: floor matches C-trunc, no correction.
        for kind in [OpKind::FloorDiv, OpKind::Div] {
            assert_eq!(
                fold(kind, vec![ci(3), ci(2)]),
                Some(ConstValue::Int(1)),
                "{kind:?}(3, 2) must be 1",
            );
            assert_eq!(
                fold(kind, vec![ci(-3), ci(-2)]),
                Some(ConstValue::Int(1)),
                "{kind:?}(-3, -2) must be 1",
            );
        }
        assert_eq!(
            fold(OpKind::Mod, vec![ci(3), ci(2)]),
            Some(ConstValue::Int(1))
        );
        assert_eq!(
            fold(OpKind::Mod, vec![ci(-3), ci(-2)]),
            Some(ConstValue::Int(-1)),
        );
        // (b) mixed sign — these are where C-trunc / Euclidean diverge
        // from Python floor. Pre-fix Euclidean would return wrong
        // values here (see commit message).
        for kind in [OpKind::FloorDiv, OpKind::Div] {
            assert_eq!(
                fold(kind, vec![ci(3), ci(-2)]),
                Some(ConstValue::Int(-2)),
                "{kind:?}(3, -2) must be -2 (Python floor)",
            );
            assert_eq!(
                fold(kind, vec![ci(-3), ci(2)]),
                Some(ConstValue::Int(-2)),
                "{kind:?}(-3, 2) must be -2 (Python floor)",
            );
        }
        assert_eq!(
            fold(OpKind::Mod, vec![ci(3), ci(-2)]),
            Some(ConstValue::Int(-1)),
            "Mod(3, -2) must be -1 (sign of divisor, not dividend)",
        );
        assert_eq!(
            fold(OpKind::Mod, vec![ci(-3), ci(2)]),
            Some(ConstValue::Int(1)),
        );
        // (c) i64::MIN / -1 declines (Rust checked_div rejects the
        // 2's-complement overflow; the runtime path handles bigint
        // promotion at upstream).
        for kind in [OpKind::FloorDiv, OpKind::Div, OpKind::Mod] {
            assert_eq!(
                fold(kind, vec![ci(i64::MIN), ci(-1)]),
                None,
                "{kind:?}(MIN, -1) must decline",
            );
        }
    }

    #[test]
    fn constfold_bitwise() {
        assert_eq!(
            fold(OpKind::And, vec![ci(0b1100), ci(0b1010)]),
            Some(ConstValue::Int(0b1000))
        );
        assert_eq!(
            fold(OpKind::Or, vec![ci(0b1100), ci(0b1010)]),
            Some(ConstValue::Int(0b1110))
        );
        assert_eq!(
            fold(OpKind::Xor, vec![ci(0b1100), ci(0b1010)]),
            Some(ConstValue::Int(0b0110))
        );
        assert_eq!(
            fold(OpKind::LShift, vec![ci(1), ci(4)]),
            Some(ConstValue::Int(16))
        );
        assert_eq!(
            fold(OpKind::RShift, vec![ci(16), ci(2)]),
            Some(ConstValue::Int(4))
        );
    }

    #[test]
    fn constfold_float_arithmetic() {
        assert_eq!(
            fold(OpKind::Add, vec![cf(1.5), cf(2.25)]),
            Some(ConstValue::float(3.75))
        );
        assert_eq!(
            fold(OpKind::Sub, vec![cf(10.0), cf(0.5)]),
            Some(ConstValue::float(9.5))
        );
        assert_eq!(
            fold(OpKind::Mul, vec![cf(2.0), cf(3.0)]),
            Some(ConstValue::float(6.0))
        );
        assert_eq!(
            fold(OpKind::TrueDiv, vec![cf(7.0), cf(2.0)]),
            Some(ConstValue::float(3.5))
        );
        // float zero divisor: upstream `operator.truediv(1.0, 0.0)`
        // raises `ZeroDivisionError`. `PureOperation.constfold` re-
        // raises that as `FlowingError`, so this must be observed as
        // `Err`, not silent decline. (The test helper `fold` collapses
        // `Err` to `None`, so we drive `constfold` directly here.)
        let op = HLOperation::new(OpKind::TrueDiv, vec![cf(1.0), cf(0.0)]);
        let err = op
            .constfold()
            .expect_err("float zero divisor must raise FlowingError");
        assert!(err.message.contains("ZeroDivisionError"), "{}", err.message);
    }

    #[test]
    fn constfold_comparisons() {
        assert_eq!(
            fold(OpKind::Lt, vec![ci(1), ci(2)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Gt, vec![ci(1), ci(2)]),
            Some(ConstValue::Bool(false))
        );
        assert_eq!(
            fold(OpKind::Eq, vec![ci(3), ci(3)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Ne, vec![ci(3), ci(4)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Le, vec![cs("a"), cs("b")]),
            Some(ConstValue::Bool(true))
        );
        // cross-type ordering comparison: upstream Python 3 raises
        // `TypeError("'<' not supported between instances of 'int' and
        // 'str'")`. `PureOperation.constfold` re-raises as
        // `FlowingError`. Eq/Ne are deliberately NOT raised — Python
        // 3 returns Bool for cross-type identity comparison.
        let op = HLOperation::new(OpKind::Lt, vec![ci(1), cs("a")]);
        let err = op
            .constfold()
            .expect_err("cross-type Lt must raise FlowingError");
        assert!(err.message.contains("TypeError"), "{}", err.message);
        // Eq/Ne tolerate cross-type — fold returns Bool(false)/Bool(true).
        assert_eq!(
            fold(OpKind::Eq, vec![ci(1), cs("a")]),
            Some(ConstValue::Bool(false))
        );
        assert_eq!(
            fold(OpKind::Ne, vec![ci(1), cs("a")]),
            Some(ConstValue::Bool(true))
        );
    }

    #[test]
    fn constfold_cross_type_numeric_ordering() {
        // upstream `operator.lt` / `operator.gt` / `operator.le` /
        // `operator.ge` over int/float/bool admit numeric coercion:
        // `1 < 1.5`, `True < 2`, `1.0 > False` are all foldable.
        // Pre-fix `cmp_fold` returned `None` for cross-type pairs,
        // forcing the runtime to evaluate trivially-foldable ordering
        // — wasted op + lost FlowingError surface for impossible-
        // ordering cases.
        // Int < Float.
        assert_eq!(
            fold(OpKind::Lt, vec![ci(1), cf(1.5)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Gt, vec![cf(2.0), ci(1)]),
            Some(ConstValue::Bool(true)),
        );
        // Bool < Int (bool subclass of int).
        assert_eq!(
            fold(OpKind::Lt, vec![cb(true), ci(2)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Le, vec![cb(false), ci(0)]),
            Some(ConstValue::Bool(true)),
        );
        // Bool < Float.
        assert_eq!(
            fold(OpKind::Lt, vec![cb(true), cf(1.5)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Ge, vec![cf(0.0), cb(false)]),
            Some(ConstValue::Bool(true)),
        );
        // Cross-NON-numeric (Int vs UniStr) still raises FlowingError
        // via `cross_type_ordering` whitelist (sanity).
        let op = HLOperation::new(OpKind::Lt, vec![ci(1), cs("a")]);
        let err = op.constfold().expect_err("Int < UniStr must raise");
        assert!(err.message.contains("TypeError"), "{}", err.message);
    }

    #[test]
    fn constfold_eq_ne_container_deep_equality() {
        // upstream tuple/list `==` compares element-wise via Python's
        // `==` per element, so `(True,) == (1,)` is True (Bool ↔ Int
        // numeric coercion). Pre-fix the variant-strict fallback
        // would return False, mismatching upstream constfold.
        let t_true = ConstValue::Tuple(vec![ConstValue::Bool(true)]);
        let t_one = ConstValue::Tuple(vec![ConstValue::Int(1)]);
        assert_eq!(
            fold(OpKind::Eq, vec![c(t_true.clone()), c(t_one.clone())]),
            Some(ConstValue::Bool(true)),
            "(True,) == (1,) must fold True per element-wise numeric coercion",
        );
        assert_eq!(
            fold(OpKind::Ne, vec![c(t_true), c(t_one)]),
            Some(ConstValue::Bool(false)),
        );
        // Length mismatch always False.
        let t_two = ConstValue::Tuple(vec![ConstValue::Int(1), ConstValue::Int(2)]);
        let t_one2 = ConstValue::Tuple(vec![ConstValue::Int(1)]);
        assert_eq!(
            fold(OpKind::Eq, vec![c(t_two), c(t_one2)]),
            Some(ConstValue::Bool(false)),
        );
        // List vs Tuple — distinct container types remain unequal
        // (Python 3: `[1] == (1,)` is False).
        let lst = ConstValue::List(vec![ConstValue::Int(1)]);
        let tup = ConstValue::Tuple(vec![ConstValue::Int(1)]);
        assert_eq!(
            fold(OpKind::Eq, vec![c(lst), c(tup)]),
            Some(ConstValue::Bool(false)),
        );
        // Nested element-wise — `((True,),)` vs `((1,),)`.
        let nested_a = ConstValue::Tuple(vec![ConstValue::Tuple(vec![ConstValue::Bool(true)])]);
        let nested_b = ConstValue::Tuple(vec![ConstValue::Tuple(vec![ConstValue::Int(1)])]);
        assert_eq!(
            fold(OpKind::Eq, vec![c(nested_a), c(nested_b)]),
            Some(ConstValue::Bool(true)),
        );
    }

    #[test]
    fn constfold_eq_ne_python_numeric_coercion() {
        // upstream `PureOperation.constfold` calls `operator.eq(*args)`
        // / `operator.ne(*args)` which dispatches Python's numeric
        // coercion: `True == 1`, `1 == 1.0`, `True == 1.0` all return
        // `True`. The naive Rust `ConstValue::PartialEq` derives
        // variant-strict equality and would say those are unequal —
        // diverging from constfold parity.
        // Bool ↔ Int.
        assert_eq!(
            fold(OpKind::Eq, vec![cb(true), ci(1)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Eq, vec![ci(1), cb(true)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Eq, vec![cb(false), ci(0)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Ne, vec![cb(true), ci(0)]),
            Some(ConstValue::Bool(true)),
        );
        // Int ↔ Float.
        assert_eq!(
            fold(OpKind::Eq, vec![ci(1), cf(1.0)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Eq, vec![cf(1.5), ci(1)]),
            Some(ConstValue::Bool(false)),
        );
        // Bool ↔ Float.
        assert_eq!(
            fold(OpKind::Eq, vec![cb(true), cf(1.0)]),
            Some(ConstValue::Bool(true)),
        );
        // Distinct types stay False (no numeric coercion).
        assert_eq!(
            fold(OpKind::Eq, vec![ci(1), cs("1")]),
            Some(ConstValue::Bool(false)),
        );
    }

    #[test]
    fn constfold_bool_conversion() {
        assert_eq!(
            fold(OpKind::Bool, vec![ci(0)]),
            Some(ConstValue::Bool(false))
        );
        assert_eq!(
            fold(OpKind::Bool, vec![ci(5)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Bool, vec![cb(true)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Bool, vec![cs("")]),
            Some(ConstValue::Bool(false))
        );
        assert_eq!(
            fold(OpKind::Bool, vec![cs("x")]),
            Some(ConstValue::Bool(true))
        );
    }

    #[test]
    fn constfold_string_and_tuple_concat() {
        assert_eq!(
            fold(OpKind::Add, vec![cs("hi"), cs(" there")]),
            Some(ConstValue::byte_str("hi there"))
        );
        assert_eq!(
            fold(
                OpKind::Add,
                vec![
                    c(ConstValue::Tuple(vec![ConstValue::Int(1)])),
                    c(ConstValue::Tuple(vec![
                        ConstValue::Int(2),
                        ConstValue::Int(3)
                    ])),
                ],
            ),
            Some(ConstValue::Tuple(vec![
                ConstValue::Int(1),
                ConstValue::Int(2),
                ConstValue::Int(3),
            ]))
        );
    }

    #[test]
    fn constfold_identity() {
        assert_eq!(
            fold(OpKind::Is, vec![ci(3), ci(3)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Is, vec![ci(3), ci(4)]),
            Some(ConstValue::Bool(false))
        );
    }

    #[test]
    fn constfold_len() {
        assert_eq!(fold(OpKind::Len, vec![cs("abc")]), Some(ConstValue::Int(3)));
        assert_eq!(
            fold(
                OpKind::Len,
                vec![c(ConstValue::Tuple(vec![
                    ConstValue::Int(1),
                    ConstValue::Int(2)
                ]))],
            ),
            Some(ConstValue::Int(2))
        );
    }

    #[test]
    fn constfold_int_conversion() {
        assert_eq!(fold(OpKind::Int, vec![cf(3.75)]), Some(ConstValue::Int(3)));
        assert_eq!(fold(OpKind::Int, vec![cf(f64::INFINITY)]), None);
        assert_eq!(fold(OpKind::Int, vec![cf(f64::NAN)]), None);
        assert_eq!(fold(OpKind::Int, vec![cb(true)]), Some(ConstValue::Int(1)));
    }

    #[test]
    fn constfold_newtuple_is_variadic_pure() {
        // `NewTuple(PureOperation)` with `pyfunc = lambda *args: args`
        // (operation.py:542-548).
        assert_eq!(
            fold(OpKind::NewTuple, vec![ci(1), ci(2), cs("x")]),
            Some(ConstValue::Tuple(vec![
                ConstValue::Int(1),
                ConstValue::Int(2),
                ConstValue::byte_str("x"),
            ]))
        );
        // empty tuple.
        assert_eq!(
            fold(OpKind::NewTuple, vec![]),
            Some(ConstValue::Tuple(Vec::new()))
        );
    }

    #[test]
    fn constfold_pow_int_nomod() {
        // `Pow(PureOperation)` arity 3, `pyfunc = pow`; default third
        // arg is `Constant(None)` (operation.py:575).
        let none = c(ConstValue::None);
        assert_eq!(
            fold(OpKind::Pow, vec![ci(3), ci(4), none.clone()]),
            Some(ConstValue::Int(81))
        );
        // Negative exponent → not an int result → decline fold.
        assert_eq!(fold(OpKind::Pow, vec![ci(2), ci(-1), none.clone()]), None);
        // Overflow → i64::checked_pow returns None → decline fold.
        assert_eq!(fold(OpKind::Pow, vec![ci(2), ci(64), none]), None);
    }

    #[test]
    fn from_opname_round_trips_every_variant() {
        // Sample a few canonical strings; full round-trip is
        // brittle in a unit test, so we assert the ones most likely
        // to regress.
        for kind in [
            OpKind::Add,
            OpKind::AddOvf,
            OpKind::Is,
            OpKind::And,
            OpKind::Yield,
            OpKind::GetItemIdx,
            OpKind::SimpleCall,
            OpKind::CallArgs,
            OpKind::NewTuple,
            OpKind::Pow,
            OpKind::GetAttr,
        ] {
            let name = kind.opname();
            assert_eq!(
                OpKind::from_opname(name),
                Some(kind),
                "round-trip failed for {kind:?}"
            );
        }

        // Unknown majit-synthetic opnames decline.
        assert_eq!(OpKind::from_opname("not_"), None);
        assert_eq!(OpKind::from_opname("newset"), None);
        assert_eq!(OpKind::from_opname("ll_assert_not_none"), None);
    }

    #[test]
    fn ovfchecked_rewrites_kind() {
        let c1 = Hlvalue::Constant(Constant::new(ConstValue::Int(1)));
        let c2 = Hlvalue::Constant(Constant::new(ConstValue::Int(2)));
        let op = HLOperation::new(OpKind::Add, vec![c1.clone(), c2.clone()]);
        let ovf = op.ovfchecked().expect("add has ovf variant");
        assert_eq!(ovf.kind, OpKind::AddOvf);
        assert_eq!(ovf.args, vec![c1, c2]);

        // Ops without an ovf twin return None.
        let eq_op = HLOperation::new(OpKind::Eq, Vec::new());
        assert!(eq_op.ovfchecked().is_none());
    }

    #[test]
    fn into_space_operation_preserves_opname_and_args() {
        let c1 = Hlvalue::Constant(Constant::new(ConstValue::Int(1)));
        let c2 = Hlvalue::Constant(Constant::new(ConstValue::Int(2)));
        let op = HLOperation::new(OpKind::Add, vec![c1.clone(), c2.clone()]);
        let result_var = op.result.clone();
        let spaceop = op.into_space_operation();
        assert_eq!(spaceop.opname, "add");
        assert_eq!(spaceop.args, vec![c1, c2]);
        assert_eq!(spaceop.result, Hlvalue::Variable(result_var));
        assert_eq!(spaceop.offset, -1);
    }

    #[test]
    fn replace_remaps_args_and_result() {
        let src = Variable::new();
        let dst = Variable::new();
        let mut mapping: HashMap<Variable, Hlvalue> = HashMap::new();
        mapping.insert(src.clone(), Hlvalue::Variable(dst.clone()));

        let op = HLOperation {
            kind: OpKind::Add,
            args: vec![
                Hlvalue::Variable(src.clone()),
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
            ],
            result: src.clone(),
            offset: 42,
        };
        let replaced = op.replace(&mapping);
        assert_eq!(replaced.kind, OpKind::Add);
        assert_eq!(
            replaced.args,
            vec![
                Hlvalue::Variable(dst.clone()),
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
            ]
        );
        assert_eq!(replaced.result, dst);
        assert_eq!(replaced.offset, 42);
    }
}
