//! rtyper-side `class __extend__(pairtype(R_A, R_B))` double-dispatch glue.
//!
//! ## NOT a port of `rpython/tool/pairtype.py`
//!
//! Upstream's `tool/pairtype.py` (134 LOC — `pair()` / `pairtype()` /
//! `pairmro()` / `DoubleDispatchRegistry` / `extendabletype` metaclass)
//! is ported at [`crate::tool::pairtype`]. This file instead centralises
//! the `class __extend__(pairtype(R_A, R_B))` extension blocks that
//! upstream scatters across every `r*.py` file (rnone.py:46-64,
//! rbool.py:49-84, rfloat.py:75-135, rint.py:200-665, rptr.py, ...).
//! Python's metaclass machinery makes that distribution invisible —
//! each `class __extend__` block silently binds to the pair class
//! produced by `pairtype(R_A, R_B)`. Rust has no metaclass, so the
//! binding surface must be explicit; this module provides the
//! centralization point and enumerates the (R_A, R_B) arms that
//! upstream's metaclass resolves at import time.
//!
//! ## Pieces
//!
//! 1. [`ReprClassId`] — explicit enum tag that stands in for upstream's
//!    `type(repr)` identity. The [`Repr::repr_class_id`] default
//!    returns [`ReprClassId::Repr`] (the wildcard base used by
//!    `pairtype(Repr, X)` / `pairtype(X, Repr)` extension blocks —
//!    e.g. rnone.py:46,56 or rmodel.py:298); every concrete `Repr`
//!    overrides to return its specific variant.
//!
//! 2. [`ReprClassId::mro`] + [`pair_mro`] — per-class MRO and its
//!    cross-product, mirroring upstream `pairtype.py:65-73 pairmro`.
//!    MRO chains for `BoolRepr → IntegerRepr → FloatRepr → Repr` etc.
//!    match upstream Python class inheritance, so pairtype entries
//!    resolve through the same walk order.
//!
//! 3. `pair_convert_from_to` + `pair_rtype_*` central dispatchers —
//!    the per-(R_A, R_B) match arms that upstream metaclass lookup
//!    resolves implicitly. Each arm delegates to a concrete
//!    helper in the matching `r*.rs` module. Adding a new
//!    `class __extend__(pairtype(R_X, R_Y))` block upstream means
//!    adding an arm here + a helper in `r_x.rs` or `r_y.rs`.
//!
//! ## Why an enum instead of `TypeId`
//!
//! Using `std::any::TypeId` via an `Any` supertrait on `Repr` would
//! avoid the enum, but the enum surfaces the mapping explicitly —
//! every new `Repr` port must add an arm here and update the match
//! tables in concrete dispatchers. That explicitness catches missing
//! pairtype wiring at compile time, whereas TypeId-keyed HashMaps only
//! fail at runtime.

use crate::flowspace::model::{ConstValue, Hlvalue};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, inputconst_from_lltype};
use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult, HighLevelOp, LowLevelOpList};

/// Explicit tag used by [`Repr::repr_class_id`] to drive pairtype
/// lookup.
///
/// Variants mirror each concrete `class ...Repr(Repr)` in
/// `rpython/rtyper/`. The wildcard [`ReprClassId::Repr`] corresponds
/// to `pairtype(Repr, X)` / `pairtype(X, Repr)` blocks — upstream's
/// base-class catch-all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReprClassId {
    /// Catch-all matching upstream's `class __extend__(pairtype(Repr,
    /// X))` — any unregistered concrete Repr falls back to this tag.
    Repr,
    /// `rmodel.py:353 VoidRepr`.
    VoidRepr,
    /// `rmodel.py:365 SimplePointerRepr`.
    SimplePointerRepr,
    /// `rnone.py:10 NoneRepr`.
    NoneRepr,
    /// `rbool.py:10 BoolRepr(IntegerRepr)` — MRO includes
    /// [`ReprClassId::IntegerRepr`] + [`ReprClassId::FloatRepr`] so
    /// integer / float pairtype entries resolve transitively, matching
    /// upstream's inheritance chain.
    BoolRepr,
    /// `rint.py:18 IntegerRepr(FloatRepr)` — MRO includes
    /// [`ReprClassId::FloatRepr`].
    IntegerRepr,
    /// `rfloat.py:11 FloatRepr`.
    FloatRepr,
    /// `rfloat.py:150 SingleFloatRepr`.
    SingleFloatRepr,
    /// `rfloat.py:166 LongFloatRepr`.
    LongFloatRepr,
    /// `rptr.py:27 PtrRepr`.
    PtrRepr,
    /// `rptr.py:220 InteriorPtrRepr`.
    InteriorPtrRepr,
    /// `rptr.py:195 LLADTMethRepr`.
    LLADTMethRepr,
    /// `rpbc.py:315 FunctionRepr`.
    FunctionRepr,
    /// `rpbc.py:224 FunctionsPBCRepr`.
    FunctionsPBCRepr,
    /// `rpbc.py:393 SmallFunctionSetPBCRepr`.
    SmallFunctionSetPBCRepr,
    /// `rbuiltin.py:67 BuiltinFunctionRepr`.
    BuiltinFunctionRepr,
    /// `rbuiltin.py:113 BuiltinMethodRepr`.
    BuiltinMethodRepr,
    /// `rpbc.py:635 SingleFrozenPBCRepr`.
    SingleFrozenPBCRepr,
    /// `rpbc.py:675 MultipleUnrelatedFrozenPBCRepr`.
    MultipleUnrelatedFrozenPBCRepr,
    /// `rpbc.py:728 MultipleFrozenPBCRepr`.
    MultipleFrozenPBCRepr,
    /// `rpbc.py:844 MethodOfFrozenPBCRepr`.
    MethodOfFrozenPBCRepr,
    /// `rpbc.py:1126 MethodsPBCRepr`.
    MethodsPBCRepr,
    /// `rpbc.py:920 ClassesPBCRepr`.
    ClassesPBCRepr,
    /// `rtuple.py:129 TupleRepr`.
    TupleRepr,
    /// `rstr.py:483 AbstractCharRepr` (`CharRepr` lltypesystem
    /// realisation, `lowleveltype = Char`).
    CharRepr,
    /// `rstr.py:758 AbstractUniCharRepr` (`UniCharRepr` lltypesystem
    /// realisation, `lowleveltype = UniChar`).
    UniCharRepr,
    /// `rstr.py:95 AbstractStringRepr` (`lltypesystem.rstr.StringRepr`
    /// realisation, `lowleveltype = Ptr(STR)`).
    StringRepr,
    /// `rstr.py:450 AbstractUnicodeRepr` (`lltypesystem.rstr.UnicodeRepr`
    /// realisation, `lowleveltype = Ptr(UNICODE)`).
    UnicodeRepr,
    /// Abstract base shared by `StringRepr` and `UnicodeRepr`.
    AbstractStringRepr,
}

impl ReprClassId {
    /// Upstream MRO for this repr class. Used by [`pair_mro`] to build
    /// the double-dispatch resolution order.
    ///
    /// Each slice starts with `self` and ends with
    /// [`ReprClassId::Repr`] (the ultimate base). Intermediate entries
    /// mirror upstream's Python `__mro__`:
    ///
    /// | concrete | upstream chain |
    /// |---|---|
    /// | `BoolRepr` | `BoolRepr → IntegerRepr → FloatRepr → Repr` |
    /// | `IntegerRepr` | `IntegerRepr → FloatRepr → Repr` |
    /// | any other | `Self → Repr` |
    pub fn mro(self) -> &'static [ReprClassId] {
        use ReprClassId::*;
        match self {
            Repr => &[Repr],
            VoidRepr => &[VoidRepr, Repr],
            SimplePointerRepr => &[SimplePointerRepr, Repr],
            NoneRepr => &[NoneRepr, Repr],
            BoolRepr => &[BoolRepr, IntegerRepr, FloatRepr, Repr],
            IntegerRepr => &[IntegerRepr, FloatRepr, Repr],
            FloatRepr => &[FloatRepr, Repr],
            SingleFloatRepr => &[SingleFloatRepr, Repr],
            LongFloatRepr => &[LongFloatRepr, Repr],
            PtrRepr => &[PtrRepr, Repr],
            InteriorPtrRepr => &[InteriorPtrRepr, Repr],
            LLADTMethRepr => &[LLADTMethRepr, Repr],
            FunctionRepr => &[FunctionRepr, Repr],
            FunctionsPBCRepr => &[FunctionsPBCRepr, Repr],
            SmallFunctionSetPBCRepr => &[SmallFunctionSetPBCRepr, Repr],
            BuiltinFunctionRepr => &[BuiltinFunctionRepr, Repr],
            BuiltinMethodRepr => &[BuiltinMethodRepr, Repr],
            SingleFrozenPBCRepr => &[SingleFrozenPBCRepr, Repr],
            MultipleUnrelatedFrozenPBCRepr => &[MultipleUnrelatedFrozenPBCRepr, Repr],
            MultipleFrozenPBCRepr => &[MultipleFrozenPBCRepr, Repr],
            MethodOfFrozenPBCRepr => &[MethodOfFrozenPBCRepr, Repr],
            MethodsPBCRepr => &[MethodsPBCRepr, Repr],
            ClassesPBCRepr => &[ClassesPBCRepr, Repr],
            TupleRepr => &[TupleRepr, Repr],
            CharRepr => &[CharRepr, Repr],
            UniCharRepr => &[UniCharRepr, Repr],
            StringRepr => &[StringRepr, AbstractStringRepr, Repr],
            UnicodeRepr => &[UnicodeRepr, AbstractStringRepr, Repr],
            AbstractStringRepr => &[AbstractStringRepr, Repr],
        }
    }
}

/// RPython `pairmro(cls1, cls2)` (`pairtype.py:65-73`) — Repr-side wrapper.
///
/// Defers to [`crate::tool::pairtype::pairmro`], which already ports
/// upstream's cross-MRO iterator. This wrapper binds it to
/// [`ReprClassId::mro`] so the `rtyper` pairtype dispatchers can read
/// a pre-materialised `Vec<(ReprClassId, ReprClassId)>` without
/// threading lifetime-bound slices through match arms.
pub fn pair_mro(c1: ReprClassId, c2: ReprClassId) -> Vec<(ReprClassId, ReprClassId)> {
    crate::tool::pairtype::pairmro(c1.mro(), c2.mro()).collect()
}

/// RPython `pair(a, b).convert_from_to(v, llops)` (`pairtype.py:46-49`
/// + per-module `class __extend__(pairtype(R_A, R_B))` blocks).
///
/// Returns:
/// - `Ok(Some(v))` — conversion emitted one or more llops and produced
///   the converted value.
/// - `Ok(None)` — upstream `return NotImplemented` (pairtype.py walks
///   further up the MRO; exhausted MRO returns None).
/// - `Err(TyperError)` — pair handler raised.
///
/// The Rust adaptation bundles both "handler doesn't exist" and
/// "handler returned NotImplemented" into `Ok(None)`; the `convertvar`
/// caller surfaces the final TyperError.
pub fn pair_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    let c1 = r_from.repr_class_id();
    let c2 = r_to.repr_class_id();
    for (b1, b2) in pair_mro(c1, c2) {
        if let Some(result) = dispatch_convert_from_to(b1, b2, r_from, r_to, v, llops)? {
            return Ok(Some(result));
        }
    }
    Ok(None)
}

/// Single-step `(b1, b2)` dispatcher. Each arm corresponds to one
/// `class __extend__(pairtype(R_A, R_B)): def convert_from_to(...)`
/// block in upstream.
///
/// Returns `Ok(None)` both for "no registration at (b1, b2)" and for
/// "registration returned NotImplemented" — the caller keeps walking
/// pair_mro until a `Some` surfaces.
fn dispatch_convert_from_to(
    b1: ReprClassId,
    b2: ReprClassId,
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    use ReprClassId::*;
    match (b1, b2) {
        // rptr.py:120-124 — same pointer low-level type is identity.
        (PtrRepr, PtrRepr) => same_lowleveltype_convert_from_to(r_from, r_to, v),
        // rptr.py:213-218 and rptr.py:331-336 — matching pointer /
        // interior-pointer ADT-method low-level surfaces are identity
        // conversions.
        (PtrRepr, LLADTMethRepr) | (InteriorPtrRepr, LLADTMethRepr) => {
            same_lowleveltype_convert_from_to(r_from, r_to, v)
        }
        // rptr.py:338-343 — InteriorPtrRepr -> InteriorPtrRepr is stricter
        // than lowleveltype equality: upstream checks `__dict__` equality.
        (InteriorPtrRepr, InteriorPtrRepr) => {
            same_interior_ptr_dict_convert_from_to(r_from, r_to, v)
        }
        // rbool.py:49-84 — bool participates in IntegerRepr's MRO but
        // carries explicit primitive casts for the common Bool edges.
        (BoolRepr, FloatRepr) => {
            super::rbool::pair_bool_float_convert_from_to(r_from, r_to, v, llops)
        }
        (FloatRepr, BoolRepr) => {
            super::rbool::pair_float_bool_convert_from_to(r_from, r_to, v, llops)
        }
        (BoolRepr, IntegerRepr) => {
            super::rbool::pair_bool_integer_convert_from_to(r_from, r_to, v, llops)
        }
        (IntegerRepr, BoolRepr) => {
            super::rbool::pair_integer_bool_convert_from_to(r_from, r_to, v, llops)
        }
        // rint.py:202-213,645-675 — primitive numeric casts.
        (IntegerRepr, IntegerRepr) => {
            super::rint::pair_integer_integer_convert_from_to(r_from, r_to, v, llops)
        }
        (IntegerRepr, FloatRepr) => {
            super::rint::pair_integer_float_convert_from_to(r_from, r_to, v, llops)
        }
        (FloatRepr, IntegerRepr) => {
            super::rint::pair_float_integer_convert_from_to(r_from, r_to, v, llops)
        }
        // rbuiltin.py:144-151 — pairtype(BuiltinMethodRepr,
        // BuiltinMethodRepr).convert_from_to: only converts between
        // same-methodname reprs, delegating the receiver lowering via
        // llops.convertvar(v, r_from.self_repr, r_to.self_repr).
        (BuiltinMethodRepr, BuiltinMethodRepr) => {
            super::rbuiltin::pair_builtin_method_convert_from_to(r_from, r_to, v, llops)
        }
        // rpbc.py:373-375 — pairtype(FunctionRepr,
        // FunctionRepr).convert_from_to: upstream `return v`. Both
        // ends are Void-typed so the Variable can be passed through
        // unmodified — no need to emit any operation.
        (FunctionRepr, FunctionRepr) => Ok(Some(v.clone())),
        // rpbc.py:381-383 — pairtype(FunctionsPBCRepr,
        // FunctionRepr).convert_from_to: upstream
        // `return inputconst(Void, None)`. FunctionRepr is Void-typed,
        // so the conversion collapses to a Void None constant
        // regardless of the FunctionsPBCRepr source variable.
        (FunctionsPBCRepr, FunctionRepr) => Ok(Some(Hlvalue::Constant(inputconst_from_lltype(
            &LowLevelType::Void,
            &ConstValue::None,
        )?))),
        // rpbc.py:377-379 — pairtype(FunctionRepr,
        // FunctionsPBCRepr).convert_from_to: upstream
        // `return inputconst(r_fpbc2, r_fpbc1.s_pbc.const)`.
        //   FunctionRepr always carries a single-FunctionDesc PBC whose
        //   `s_pbc.const` is the host-side function pyobj. Re-input the
        //   constant against r_fpbc2's lowleveltype, which routes
        //   through `FunctionsPBCRepr.convert_const(pyobj)`.
        (FunctionRepr, FunctionsPBCRepr) => {
            super::rpbc::pair_function_repr_functions_pbc_convert_from_to(r_from, r_to, v, llops)
        }
        // rpbc.py:385-390 — pairtype(FunctionsPBCRepr,
        // FunctionsPBCRepr).convert_from_to: upstream identity if
        // `r_fpbc1.lowleveltype == r_fpbc2.lowleveltype`, else
        // `NotImplemented`. The rtyper distinguishes different spec-func
        // Struct pointer types even within the same Repr subclass.
        (FunctionsPBCRepr, FunctionsPBCRepr) => same_lowleveltype_convert_from_to(r_from, r_to, v),
        // rpbc.py:517-519 — pairtype(SmallFunctionSetPBCRepr,
        // FunctionRepr).convert_from_to: `return inputconst(Void, None)`.
        //   FunctionRepr is Void-typed; the Char source variable is
        //   discarded.
        (SmallFunctionSetPBCRepr, FunctionRepr) => Ok(Some(Hlvalue::Constant(
            inputconst_from_lltype(&LowLevelType::Void, &ConstValue::None)?,
        ))),
        // rpbc.py:521-526 — pairtype(SmallFunctionSetPBCRepr,
        // FunctionsPBCRepr).convert_from_to:
        //   `assert v.concretetype is Char;
        //    v_int = llops.genop('cast_char_to_int', [v], resulttype=Signed);
        //    return llops.genop('getarrayitem',
        //                        [r_set.c_pointer_table, v_int],
        //                        resulttype=r_ptr.lowleveltype)`.
        (SmallFunctionSetPBCRepr, FunctionsPBCRepr) => {
            super::rpbc::pair_small_function_set_functions_pbc_convert_from_to(
                r_from, r_to, v, llops,
            )
        }
        // rpbc.py:548-551 — pairtype(FunctionRepr,
        // SmallFunctionSetPBCRepr).convert_from_to: pulls the unique
        // FunctionDesc out of `r_ptr.s_pbc.descriptions` and returns
        // `inputconst(Char, r_set.convert_desc(desc))`.
        (FunctionRepr, SmallFunctionSetPBCRepr) => {
            super::rpbc::pair_function_repr_small_function_set_convert_from_to(
                r_from, r_to, v, llops,
            )
        }
        // rpbc.py:553-556 — pairtype(FunctionsPBCRepr,
        // SmallFunctionSetPBCRepr).convert_from_to:
        //   `ll_compress = compression_function(r_set);
        //    return llops.gendirectcall(ll_compress, v)`.
        (FunctionsPBCRepr, SmallFunctionSetPBCRepr) => {
            super::rpbc::pair_functions_pbc_small_function_set_convert_from_to(
                r_from, r_to, v, llops,
            )
        }
        // rpbc.py:597-607 — pairtype(SmallFunctionSetPBCRepr,
        // SmallFunctionSetPBCRepr).convert_from_to:
        //   `c_table = conversion_table(r_from, r_to);
        //    if c_table:
        //        v_int = genop('cast_char_to_int', [v], Signed)
        //        return genop('getarrayitem', [c_table, v_int], Char)
        //    else:
        //        return v  # identity`.
        (SmallFunctionSetPBCRepr, SmallFunctionSetPBCRepr) => {
            super::rpbc::pair_small_function_set_small_function_set_convert_from_to(
                r_from, r_to, v, llops,
            )
        }
        // rtuple.py:340-353 — `pairtype(TupleRepr, TupleRepr).convert_from_to`.
        // Same-arity tuple-to-tuple: identity if lltypes match, else
        // per-item getitem_internal + convertvar + newtuple.
        // Different-arity returns NotImplemented.
        (TupleRepr, TupleRepr) => {
            super::rtuple::pair_tuple_tuple_convert_from_to(r_from, r_to, v, llops)
        }
        // rstr.py:805-814 — Char→String and UniChar→Unicode conversion
        // routes through `ll_chr2str`, allocating a one-character
        // STR/UNICODE before downstream string operations reuse the
        // AbstractStringRepr machinery.
        (CharRepr, StringRepr) | (UniCharRepr, UnicodeRepr) => {
            super::rstr::pair_charlike_string_convert_from_to(r_from, r_to, v, llops)
        }
        // rstr.py:815-820 — String→Char conversion extracts
        // `ll_stritem_nonneg(s, 0)`.
        (StringRepr, CharRepr) => {
            super::rstr::pair_string_char_convert_from_to(r_from, r_to, v, llops)
        }
        // rmodel.py:361-363 — `pairtype(Repr, VoidRepr).convert_from_to`
        // returns `inputconst(Void, None)`.
        (Repr, VoidRepr) => Ok(Some(Hlvalue::Constant(inputconst_from_lltype(
            &LowLevelType::Void,
            &ConstValue::None,
        )?))),
        // rnone.py:46-49 — `pairtype(Repr, NoneRepr).convert_from_to`
        // returns `inputconst(Void, None)`.
        (Repr, NoneRepr) => super::rnone::pair_any_none_convert_from_to(r_from, r_to, v, llops),
        // rnone.py:56-59 — `pairtype(NoneRepr, Repr).convert_from_to`
        // returns `inputconst(r_to, None)`.
        (NoneRepr, Repr) => super::rnone::pair_none_any_convert_from_to(r_from, r_to, v, llops),
        _ => Ok(None),
    }
}

fn same_lowleveltype_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
) -> Result<Option<Hlvalue>, TyperError> {
    if r_from.lowleveltype() == r_to.lowleveltype() {
        return Ok(Some(v.clone()));
    }
    Ok(None)
}

fn same_interior_ptr_dict_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
) -> Result<Option<Hlvalue>, TyperError> {
    let from_key = r_from.interior_ptr_repr_dict_key();
    if from_key.is_some() && from_key == r_to.interior_ptr_repr_dict_key() {
        return Ok(Some(v.clone()));
    }
    Ok(None)
}

type PairRTypeDispatch = Result<Option<Option<Hlvalue>>, TyperError>;

fn committed(result: RTypeResult) -> PairRTypeDispatch {
    result.map(Some)
}

fn pair_rtype_op(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp, opname: &str) -> RTypeResult {
    let c1 = r1.repr_class_id();
    let c2 = r2.repr_class_id();
    for (b1, b2) in pair_mro(c1, c2) {
        if let Some(result) = dispatch_rtype_op(b1, b2, r1, r2, hop, opname)? {
            return Ok(result);
        }
    }
    Err(TyperError::missing_rtype_operation(format!(
        "pair(rtype_{opname}) not implemented for ({:?}, {:?})",
        c1, c2
    )))
}

fn dispatch_rtype_op(
    b1: ReprClassId,
    b2: ReprClassId,
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &HighLevelOp,
    opname: &str,
) -> PairRTypeDispatch {
    use ReprClassId::*;
    match (b1, b2, opname) {
        // rptr.py:126-159 / 301-329 — array and interior-array indexing.
        (PtrRepr, IntegerRepr, "getitem") | (InteriorPtrRepr, IntegerRepr, "getitem") => {
            committed(r1.rtype_getitem(hop))
        }
        // rtuple.py:264-273 — `pairtype(TupleRepr, IntegerRepr).rtype_getitem`.
        (TupleRepr, IntegerRepr, "getitem") => {
            committed(super::rtuple::pair_tuple_int_rtype_getitem(r1, hop))
        }
        // rstr.py:614-632 — `pair(AbstractStringRepr, IntegerRepr).rtype_getitem`.
        // Pyre handles the nonneg + checkidx=False branch; other
        // branches (negative-index ll_stritem, checkidx
        // ll_stritem_*_checked) surface a TyperError until those
        // helpers land.
        (StringRepr, IntegerRepr, "getitem") => {
            committed(super::rstr::pair_string_int_rtype_getitem(hop))
        }
        (UnicodeRepr, IntegerRepr, "getitem") => {
            committed(super::rstr::pair_unicode_int_rtype_getitem(hop))
        }
        // rstr.py:634-635 — `rtype_getitem_idx` defers to
        // `rtype_getitem(hop, checkidx=True)`. The checked path raises
        // IndexError from inside ll_stritem_checked /
        // ll_stritem_nonneg_checked.
        (StringRepr, IntegerRepr, "getitem_idx") => {
            committed(super::rstr::pair_string_int_rtype_getitem_idx(hop))
        }
        (UnicodeRepr, IntegerRepr, "getitem_idx") => {
            committed(super::rstr::pair_unicode_int_rtype_getitem_idx(hop))
        }
        // rstr.py:723-730 — `pairtype(AbstractCharRepr, IntegerRepr)`
        // and `pairtype(AbstractUniCharRepr, IntegerRepr)` rtype_getitem
        // share the same body: constant-0 pass-through, otherwise
        // Char→Str/UniChar→Unicode via `ll_chr2str` and reuse
        // AbstractStringRepr+IntegerRepr.
        (CharRepr, IntegerRepr, "getitem") => {
            committed(super::rstr::pair_char_int_rtype_getitem(hop))
        }
        (UniCharRepr, IntegerRepr, "getitem") => {
            committed(super::rstr::pair_unichar_int_rtype_getitem(hop))
        }
        (CharRepr, IntegerRepr, "getitem_idx") => {
            committed(super::rstr::pair_char_int_rtype_getitem_idx(hop))
        }
        (UniCharRepr, IntegerRepr, "getitem_idx") => {
            committed(super::rstr::pair_unichar_int_rtype_getitem_idx(hop))
        }
        // rstr.py:651-659 — `pair(AbstractStringRepr, AbstractStringRepr).rtype_add`
        // (and `rtype_inplace_add`) lower to
        // `direct_call(ll_strconcat, v_s1, v_s2)` against the
        // helper graph synthesised by
        // `build_ll_strconcat_helper_graph` (malloc_varsize +
        // chars-array copy loop).
        (StringRepr, StringRepr, "add") | (StringRepr, StringRepr, "inplace_add") => {
            committed(super::rstr::pair_string_string_rtype_add(hop))
        }
        (UnicodeRepr, UnicodeRepr, "add") | (UnicodeRepr, UnicodeRepr, "inplace_add") => {
            committed(super::rstr::pair_unicode_unicode_rtype_add(hop))
        }
        (AbstractStringRepr, AbstractStringRepr, "add")
        | (AbstractStringRepr, AbstractStringRepr, "inplace_add") => {
            committed(super::rstr::pair_abstract_string_rtype_add(r1, r2, hop))
        }
        // rtuple.py:319-327 — `pairtype(TupleRepr, TupleRepr).rtype_add`
        // (and its `rtype_inplace_add` alias) concatenates two tuples
        // by per-position getfield + newtuple_cached.
        (TupleRepr, TupleRepr, "add") | (TupleRepr, TupleRepr, "inplace_add") => {
            committed(super::rtuple::pair_tuple_tuple_rtype_add(r1, r2, hop))
        }
        // rtuple.py:329-334 — `pairtype(TupleRepr, TupleRepr).rtype_eq`
        // dispatches to the per-shape `ll_eq` helper synthesised by
        // `gen_eq_function`.
        (TupleRepr, TupleRepr, "eq") => {
            committed(super::rtuple::pair_tuple_tuple_rtype_eq(r1, r2, hop))
        }
        // rtuple.py:336-338 — `rtype_ne = rtype_eq + bool_not`.
        (TupleRepr, TupleRepr, "ne") => {
            committed(super::rtuple::pair_tuple_tuple_rtype_ne(r1, r2, hop))
        }
        // rtuple.py:292-315 — `pairtype(TupleRepr, Repr).rtype_contains`
        // dispatches to a constant-tuple membership test using the
        // synthesised per-type `ll_equal` helper.
        (TupleRepr, _, "contains") => {
            committed(super::rtuple::pair_tuple_repr_rtype_contains(r1, r2, hop))
        }
        (PtrRepr, IntegerRepr, "setitem") | (InteriorPtrRepr, IntegerRepr, "setitem") => {
            committed(r1.rtype_setitem(hop))
        }

        // rptr.py:165-184 — pointer comparison accepts any repr on the
        // other side and coerces both args to the pointer repr.
        (PtrRepr, Repr, "eq") => committed(r1.rtype_eq(hop)),
        (PtrRepr, Repr, "ne") => committed(r1.rtype_ne(hop)),
        (Repr, PtrRepr, "eq") => committed(r2.rtype_eq(hop)),
        (Repr, PtrRepr, "ne") => committed(r2.rtype_ne(hop)),

        // rint.py:217-310 — IntegerRepr/IntegerRepr pair arithmetic and
        // comparisons. `truediv` is intentionally absent here: upstream
        // delegates it to FloatRepr through the MRO.
        (IntegerRepr, IntegerRepr, "add") => committed(super::rint::rtype_template(hop, "add")),
        (IntegerRepr, IntegerRepr, "add_ovf") => committed(super::rint::rtype_add_ovf(hop)),
        (IntegerRepr, IntegerRepr, "sub") => committed(super::rint::rtype_template(hop, "sub")),
        (IntegerRepr, IntegerRepr, "sub_ovf") => {
            committed(super::rint::rtype_template(hop, "sub_ovf"))
        }
        (IntegerRepr, IntegerRepr, "mul") => committed(super::rint::rtype_template(hop, "mul")),
        (IntegerRepr, IntegerRepr, "mul_ovf") => {
            committed(super::rint::rtype_template(hop, "mul_ovf"))
        }
        (IntegerRepr, IntegerRepr, "floordiv") | (IntegerRepr, IntegerRepr, "div") => committed(
            super::rint::rtype_call_helper(hop, "py_div".to_string(), &["ZeroDivisionError"]),
        ),
        (IntegerRepr, IntegerRepr, "floordiv_ovf") | (IntegerRepr, IntegerRepr, "div_ovf") => {
            committed(super::rint::rtype_call_helper(
                hop,
                "py_div_ovf".to_string(),
                &["ZeroDivisionError"],
            ))
        }
        (IntegerRepr, IntegerRepr, "mod") => committed(super::rint::rtype_call_helper(
            hop,
            "py_mod".to_string(),
            &["ZeroDivisionError"],
        )),
        (IntegerRepr, IntegerRepr, "mod_ovf") => committed(super::rint::rtype_call_helper(
            hop,
            "py_mod_ovf".to_string(),
            &["ZeroDivisionError"],
        )),
        (IntegerRepr, IntegerRepr, "xor") => committed(super::rint::rtype_template(hop, "xor")),
        (IntegerRepr, IntegerRepr, "and") => committed(super::rint::rtype_template(hop, "and")),
        (IntegerRepr, IntegerRepr, "or") => committed(super::rint::rtype_template(hop, "or")),
        (IntegerRepr, IntegerRepr, "lshift") => {
            committed(super::rint::rtype_template(hop, "lshift"))
        }
        (IntegerRepr, IntegerRepr, "lshift_ovf") => committed(super::rint::rtype_call_helper(
            hop,
            "lshift_ovf".to_string(),
            &[],
        )),
        (IntegerRepr, IntegerRepr, "rshift") => {
            committed(super::rint::rtype_template(hop, "rshift"))
        }
        (IntegerRepr, IntegerRepr, "eq") => {
            committed(super::rint::rtype_compare_template(hop, "eq"))
        }
        (IntegerRepr, IntegerRepr, "ne") => {
            committed(super::rint::rtype_compare_template(hop, "ne"))
        }
        (IntegerRepr, IntegerRepr, "lt") => {
            committed(super::rint::rtype_compare_template(hop, "lt"))
        }
        (IntegerRepr, IntegerRepr, "le") => {
            committed(super::rint::rtype_compare_template(hop, "le"))
        }
        (IntegerRepr, IntegerRepr, "gt") => {
            committed(super::rint::rtype_compare_template(hop, "gt"))
        }
        (IntegerRepr, IntegerRepr, "ge") => {
            committed(super::rint::rtype_compare_template(hop, "ge"))
        }
        (IntegerRepr, IntegerRepr, "is_") => {
            committed(super::rint::rtype_compare_template(hop, "eq"))
        }

        // rfloat.py:75-135 — FloatRepr/FloatRepr pair arithmetic and
        // comparisons. IntegerRepr inherits FloatRepr upstream, so mixed
        // int/float arithmetic reaches these arms via `pair_mro`.
        (FloatRepr, FloatRepr, "add") => committed(super::rfloat::rtype_template(hop, "add")),
        (FloatRepr, FloatRepr, "sub") => committed(super::rfloat::rtype_template(hop, "sub")),
        (FloatRepr, FloatRepr, "mul") => committed(super::rfloat::rtype_template(hop, "mul")),
        (FloatRepr, FloatRepr, "truediv") | (FloatRepr, FloatRepr, "div") => {
            committed(super::rfloat::rtype_template(hop, "truediv"))
        }
        (FloatRepr, FloatRepr, "eq") => committed(super::rfloat::rtype_compare_template(hop, "eq")),
        (FloatRepr, FloatRepr, "ne") => committed(super::rfloat::rtype_compare_template(hop, "ne")),
        (FloatRepr, FloatRepr, "lt") => committed(super::rfloat::rtype_compare_template(hop, "lt")),
        (FloatRepr, FloatRepr, "le") => committed(super::rfloat::rtype_compare_template(hop, "le")),
        (FloatRepr, FloatRepr, "gt") => committed(super::rfloat::rtype_compare_template(hop, "gt")),
        (FloatRepr, FloatRepr, "ge") => committed(super::rfloat::rtype_compare_template(hop, "ge")),
        (FloatRepr, FloatRepr, "is_") => {
            committed(super::rfloat::rtype_compare_template(hop, "eq"))
        }

        // rstr.py:740-746 — pairtype(AbstractCharRepr, AbstractCharRepr)
        // dispatches all six compare ops to `char_<func>` lloperations.
        (CharRepr, CharRepr, "eq") => {
            committed(super::rstr::pair_char_char_rtype_compare(hop, "eq"))
        }
        (CharRepr, CharRepr, "ne") => {
            committed(super::rstr::pair_char_char_rtype_compare(hop, "ne"))
        }
        (CharRepr, CharRepr, "lt") => {
            committed(super::rstr::pair_char_char_rtype_compare(hop, "lt"))
        }
        (CharRepr, CharRepr, "le") => {
            committed(super::rstr::pair_char_char_rtype_compare(hop, "le"))
        }
        (CharRepr, CharRepr, "gt") => {
            committed(super::rstr::pair_char_char_rtype_compare(hop, "gt"))
        }
        (CharRepr, CharRepr, "ge") => {
            committed(super::rstr::pair_char_char_rtype_compare(hop, "ge"))
        }

        // rstr.py:778-784 — pairtype(AbstractUniCharRepr, AbstractUniCharRepr).
        // eq/ne dispatch to `unichar_<func>`; lt/le/gt/ge cast both
        // arms via `cast_unichar_to_int` and dispatch to `int_<func>`.
        (UniCharRepr, UniCharRepr, "eq") => committed(
            super::rstr::pair_unichar_unichar_rtype_compare_eqne(hop, "eq"),
        ),
        (UniCharRepr, UniCharRepr, "ne") => committed(
            super::rstr::pair_unichar_unichar_rtype_compare_eqne(hop, "ne"),
        ),
        (UniCharRepr, UniCharRepr, "lt") => committed(
            super::rstr::pair_unichar_unichar_rtype_compare_ord(hop, "lt"),
        ),
        (UniCharRepr, UniCharRepr, "le") => committed(
            super::rstr::pair_unichar_unichar_rtype_compare_ord(hop, "le"),
        ),
        (UniCharRepr, UniCharRepr, "gt") => committed(
            super::rstr::pair_unichar_unichar_rtype_compare_ord(hop, "gt"),
        ),
        (UniCharRepr, UniCharRepr, "ge") => committed(
            super::rstr::pair_unichar_unichar_rtype_compare_ord(hop, "ge"),
        ),

        // rstr.py:651-692 — pairtype(AbstractStringRepr, AbstractStringRepr).
        // eq routes through ll_streq, ne wraps ll_streq with bool_not,
        // lt/le/gt/ge route through ll_strcmp + int_<func>(diff, 0).
        (StringRepr, StringRepr, "eq") => {
            committed(super::rstr::pair_string_string_rtype_compare(hop, "eq"))
        }
        (StringRepr, StringRepr, "ne") => {
            committed(super::rstr::pair_string_string_rtype_compare(hop, "ne"))
        }
        (StringRepr, StringRepr, "lt") => {
            committed(super::rstr::pair_string_string_rtype_compare(hop, "lt"))
        }
        (StringRepr, StringRepr, "le") => {
            committed(super::rstr::pair_string_string_rtype_compare(hop, "le"))
        }
        (StringRepr, StringRepr, "gt") => {
            committed(super::rstr::pair_string_string_rtype_compare(hop, "gt"))
        }
        (StringRepr, StringRepr, "ge") => {
            committed(super::rstr::pair_string_string_rtype_compare(hop, "ge"))
        }

        // rstr.py:651-692 inherited via AbstractUnicodeRepr(AbstractStringRepr) —
        // pairtype(AbstractUnicodeRepr, AbstractUnicodeRepr) routes through
        // the same body modulo the helper-graph identity (ll_unicode_eq /
        // ll_unicode_cmp).
        (UnicodeRepr, UnicodeRepr, "eq") => {
            committed(super::rstr::pair_unicode_unicode_rtype_compare(hop, "eq"))
        }
        (UnicodeRepr, UnicodeRepr, "ne") => {
            committed(super::rstr::pair_unicode_unicode_rtype_compare(hop, "ne"))
        }
        (UnicodeRepr, UnicodeRepr, "lt") => {
            committed(super::rstr::pair_unicode_unicode_rtype_compare(hop, "lt"))
        }
        (UnicodeRepr, UnicodeRepr, "le") => {
            committed(super::rstr::pair_unicode_unicode_rtype_compare(hop, "le"))
        }
        (UnicodeRepr, UnicodeRepr, "gt") => {
            committed(super::rstr::pair_unicode_unicode_rtype_compare(hop, "gt"))
        }
        (UnicodeRepr, UnicodeRepr, "ge") => {
            committed(super::rstr::pair_unicode_unicode_rtype_compare(hop, "ge"))
        }

        _ => Ok(None),
    }
}

pub fn pair_rtype_getitem(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "getitem")
}

pub fn pair_rtype_setitem(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "setitem")
}

pub fn pair_rtype_delitem(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "delitem")
}

pub fn pair_rtype_add(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "add")
}

pub fn pair_rtype_add_ovf(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "add_ovf")
}

pub fn pair_rtype_sub(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "sub")
}

pub fn pair_rtype_sub_ovf(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "sub_ovf")
}

pub fn pair_rtype_mul(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "mul")
}

pub fn pair_rtype_mul_ovf(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "mul_ovf")
}

pub fn pair_rtype_truediv(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "truediv")
}

pub fn pair_rtype_floordiv(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "floordiv")
}

pub fn pair_rtype_floordiv_ovf(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "floordiv_ovf")
}

pub fn pair_rtype_div(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "div")
}

pub fn pair_rtype_div_ovf(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "div_ovf")
}

pub fn pair_rtype_mod(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "mod")
}

pub fn pair_rtype_mod_ovf(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "mod_ovf")
}

pub fn pair_rtype_xor(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "xor")
}

pub fn pair_rtype_and_(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "and")
}

pub fn pair_rtype_or_(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "or")
}

pub fn pair_rtype_lshift(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "lshift")
}

pub fn pair_rtype_lshift_ovf(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "lshift_ovf")
}

pub fn pair_rtype_rshift(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "rshift")
}

pub fn pair_rtype_cmp(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "cmp")
}

pub fn pair_rtype_coerce(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "coerce")
}

pub fn pair_rtype_contains(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "contains")
}

pub fn pair_rtype_eq(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "eq")
}

pub fn pair_rtype_ne(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "ne")
}

pub fn pair_rtype_lt(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "lt")
}

pub fn pair_rtype_le(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "le")
}

pub fn pair_rtype_gt(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "gt")
}

pub fn pair_rtype_ge(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "ge")
}

/// RPython `pair(r1, r2).rtype_is_(hop)` (pairtype.py + rnone.py:51-54,
/// 61-64 + rfloat.py:110 `rtype_is_ = rtype_eq`).
///
/// Equivalent of `translate_op_is` consulting the pair table. `pos`
/// defaults to 0 upstream for the first-arg side and 1 for the
/// second-arg side; the caller picks based on which pair matched.
pub fn pair_rtype_is_(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    let c1 = r1.repr_class_id();
    let c2 = r2.repr_class_id();
    for (b1, b2) in pair_mro(c1, c2) {
        if let Some(result) = dispatch_rtype_is_(b1, b2, r1, r2, hop)? {
            return Ok(Some(result));
        }
        if let Some(result) = dispatch_rtype_op(b1, b2, r1, r2, hop, "is_")? {
            return Ok(result);
        }
    }
    Err(TyperError::missing_rtype_operation(format!(
        "pair(rtype_is_) not implemented for ({:?}, {:?})",
        c1, c2
    )))
}

fn dispatch_rtype_is_(
    b1: ReprClassId,
    b2: ReprClassId,
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &HighLevelOp,
) -> Result<Option<Hlvalue>, TyperError> {
    use ReprClassId::*;
    match (b1, b2) {
        // rnone.py:51-54 — `pairtype(Repr, NoneRepr).rtype_is_` calls
        // `rtype_is_None(robj1, rnone2, hop, pos=0)`.
        (Repr, NoneRepr) => super::rnone::pair_any_none_rtype_is_(r1, r2, hop),
        // rnone.py:61-64 — `pairtype(NoneRepr, Repr).rtype_is_` calls
        // `rtype_is_None(robj2, rnone1, hop, pos=1)`.
        (NoneRepr, Repr) => super::rnone::pair_none_any_rtype_is_(r1, r2, hop),
        // rpbc.py:713-725 — `pairtype(MultipleUnrelatedFrozenPBCRepr,
        // MultipleUnrelatedFrozenPBCRepr).rtype_is_` emits adr_eq.
        // The (MU, Single) / (Single, MU) arms in the same upstream
        // block depend on a Single->MU convert_from_to that has not
        // landed yet; those shapes still fall through to the generic
        // (Repr, Repr) dispatcher below.
        (MultipleUnrelatedFrozenPBCRepr, MultipleUnrelatedFrozenPBCRepr) => {
            super::rpbc::pair_mu_mu_rtype_is_(r1, r2, hop).map(Some)
        }
        // rpbc.py:558-571 — `pairtype(FunctionReprBase,
        // FunctionReprBase).rtype_is_`: union the two SomePBCs,
        // resolve the merged repr, convert both sides to it, then
        // emit `char_eq` (Char-typed Small set) or `ptr_eq` (Ptr-typed
        // Functions / Function set). Pyre's FunctionReprBase analogues
        // are `FunctionRepr`, `FunctionsPBCRepr`, and
        // `SmallFunctionSetPBCRepr` — all three pairwise.
        (FunctionRepr, FunctionRepr)
        | (FunctionRepr, FunctionsPBCRepr)
        | (FunctionRepr, SmallFunctionSetPBCRepr)
        | (FunctionsPBCRepr, FunctionRepr)
        | (FunctionsPBCRepr, FunctionsPBCRepr)
        | (FunctionsPBCRepr, SmallFunctionSetPBCRepr)
        | (SmallFunctionSetPBCRepr, FunctionRepr)
        | (SmallFunctionSetPBCRepr, FunctionsPBCRepr)
        | (SmallFunctionSetPBCRepr, SmallFunctionSetPBCRepr) => {
            super::rpbc::pair_function_repr_base_rtype_is_(r1, r2, hop).map(Some)
        }
        // rtuple.py:355-356 — `pairtype(TupleRepr, TupleRepr).rtype_is_`
        // raises `TyperError("cannot compare tuples with 'is'")`. The
        // identity check is structurally meaningless on a tuple value,
        // and the rtyper rejects it eagerly so callers cannot route a
        // tuple through ptr_eq via the generic `(Repr, Repr)` arm below.
        (TupleRepr, TupleRepr) => super::rtuple::pair_tuple_tuple_rtype_is_(r1, r2, hop),
        // rmodel.py:300-318 — generic identity comparison for pointer
        // low-level values, with Void adopting the opposite repr.
        (Repr, Repr) => pair_repr_repr_rtype_is_(r1, r2, hop).map(Some),
        _ => Ok(None),
    }
}

fn pair_repr_repr_rtype_is_(
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &HighLevelOp,
) -> Result<Hlvalue, TyperError> {
    if let Some(value) = hop
        .s_result
        .borrow()
        .as_ref()
        .and_then(|s| s.const_().cloned())
    {
        return Ok(Hlvalue::Constant(inputconst_from_lltype(
            &LowLevelType::Bool,
            &value,
        )?));
    }

    let roriginal1 = r1;
    let roriginal2 = r2;
    let mut robj1 = r1;
    let mut robj2 = r2;
    if matches!(robj1.lowleveltype(), LowLevelType::Void) {
        robj1 = robj2;
    } else if matches!(robj2.lowleveltype(), LowLevelType::Void) {
        robj2 = robj1;
    }

    if !matches!(robj1.lowleveltype(), LowLevelType::Ptr(_))
        || !matches!(robj2.lowleveltype(), LowLevelType::Ptr(_))
    {
        return Err(TyperError::message(format!(
            "is of instances of the non-pointers: {}, {}",
            roriginal1.repr_string(),
            roriginal2.repr_string()
        )));
    }
    if robj1.lowleveltype() != robj2.lowleveltype() {
        return Err(TyperError::message(format!(
            "is of instances of different pointer types: {}, {}",
            roriginal1.repr_string(),
            roriginal2.repr_string()
        )));
    }

    let v_list = hop.inputargs(vec![ConvertedTo::Repr(robj1), ConvertedTo::Repr(robj2)])?;
    Ok(hop
        .genop("ptr_eq", v_list, GenopResult::LLType(LowLevelType::Bool))
        .expect("ptr_eq with Bool result returns a value"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::rtyper::rbool::bool_repr;
    use crate::translator::rtyper::rfloat::float_repr;
    use crate::translator::rtyper::rint::signed_repr;
    use crate::translator::rtyper::rnone::none_repr;
    use std::sync::Arc;

    #[test]
    fn repr_class_id_of_concrete_reprs_returns_their_variant() {
        // Exercises `Repr::repr_class_id` override on each landed Repr
        // — any new Repr port must add its arm here + the matching MRO
        // entry in [`ReprClassId::mro`].
        use ReprClassId::*;
        assert_eq!((*none_repr()).repr_class_id(), NoneRepr);
        assert_eq!((*bool_repr()).repr_class_id(), BoolRepr);
        assert_eq!((*signed_repr()).repr_class_id(), IntegerRepr);
        assert_eq!((*float_repr()).repr_class_id(), FloatRepr);
    }

    #[test]
    fn pair_mro_order_matches_upstream_python() {
        // pairtype.py:65-73 — outer loop over cls2.__mro__, inner over
        // cls1.__mro__. For (BoolRepr, FloatRepr):
        //  mro1 = [BoolRepr, IntegerRepr, FloatRepr, Repr]
        //  mro2 = [FloatRepr, Repr]
        //  yielded pairs:
        //    (BoolRepr, FloatRepr), (IntegerRepr, FloatRepr),
        //    (FloatRepr, FloatRepr), (Repr, FloatRepr),
        //    (BoolRepr, Repr), (IntegerRepr, Repr), (FloatRepr, Repr),
        //    (Repr, Repr).
        use ReprClassId::*;
        let mro = pair_mro(BoolRepr, FloatRepr);
        assert_eq!(
            mro,
            vec![
                (BoolRepr, FloatRepr),
                (IntegerRepr, FloatRepr),
                (FloatRepr, FloatRepr),
                (Repr, FloatRepr),
                (BoolRepr, Repr),
                (IntegerRepr, Repr),
                (FloatRepr, Repr),
                (Repr, Repr),
            ]
        );
    }

    #[test]
    fn pair_mro_simple_cases_end_with_repr_repr() {
        // For pair(NoneRepr, IntegerRepr) the MRO terminates at
        // (Repr, Repr) after exhausting both chains.
        use ReprClassId::*;
        let mro = pair_mro(NoneRepr, IntegerRepr);
        assert_eq!(mro.first().copied(), Some((NoneRepr, IntegerRepr)));
        assert_eq!(mro.last().copied(), Some((Repr, Repr)));
    }

    #[test]
    fn convert_from_to_any_to_none_produces_void_constant() {
        // rnone.py:46-49 — `pairtype(Repr, NoneRepr).convert_from_to`
        // returns `inputconst(Void, None)`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);

        let r_from = signed_repr();
        let r_to = none_repr();
        let dummy_v = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(42),
            LowLevelType::Signed,
        ));
        let converted = pair_convert_from_to(r_from.as_ref(), r_to.as_ref(), &dummy_v, &mut llops)
            .expect("pair_convert_from_to should succeed")
            .expect("convert should not return NotImplemented");
        let Hlvalue::Constant(c) = converted else {
            panic!("expected Constant result");
        };
        assert_eq!(c.value, ConstValue::None);
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Void));
    }

    #[test]
    fn convert_from_to_same_repr_falls_through_to_caller_identity() {
        // Pairtype dispatch returns `Ok(None)` when no MRO entry
        // registers the combination — e.g. (FloatRepr, FloatRepr) has
        // no convert_from_to registration. The `convertvar` caller
        // short-circuits that to identity (same-repr) before even
        // calling the dispatcher, so this test directly exercises the
        // "no handler" NotImplemented path.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);

        let r = float_repr();
        let dummy_v = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Float(1.0_f64.to_bits()),
            LowLevelType::Float,
        ));
        let result = pair_convert_from_to(r.as_ref(), r.as_ref(), &dummy_v, &mut llops)
            .expect("pair_convert_from_to should not error");
        assert!(
            result.is_none(),
            "no (FloatRepr, FloatRepr) convert handler — MRO exhausts to None"
        );
    }

    #[test]
    fn convert_from_to_integer_integer_emits_rint_cast() {
        // rint.py:202-213 — Signed -> Unsigned emits
        // `cast_int_to_uint` through pairtype(IntegerRepr, IntegerRepr).
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rint::unsigned_repr;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);

        let r_from = signed_repr();
        let r_to = unsigned_repr();
        let dummy_v = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(1),
            LowLevelType::Signed,
        ));
        let converted = pair_convert_from_to(r_from.as_ref(), r_to.as_ref(), &dummy_v, &mut llops)
            .expect("pair_convert_from_to should succeed")
            .expect("integer conversion should commit");
        assert!(matches!(converted, Hlvalue::Variable(_)));
        assert_eq!(llops.ops.len(), 1);
        assert_eq!(llops.ops[0].opname, "cast_int_to_uint");
    }

    #[test]
    fn convert_from_to_ptr_lladtmeth_uses_same_lowleveltype_like_rptr() {
        // rptr.py:213-218 — pairtype(PtrRepr, LLADTMethRepr) converts by
        // identity when the low-level pointer type matches.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::translator::rtyper::llannotation::SomeLLADTMeth;
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelPointerType, Ptr, PtrTarget, StructType,
        };
        use crate::translator::rtyper::rmodel::{LLADTMethRepr, PtrRepr};
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let ptr = Ptr {
            TO: PtrTarget::Struct(StructType::gc("S", vec![])),
        };
        let r_from = PtrRepr::new(ptr.clone());
        let adtmeth = SomeLLADTMeth::new(
            LowLevelPointerType::Ptr(ptr.clone()),
            ConstValue::Function(Box::new(crate::flowspace::model::GraphFunc::new(
                "meth",
                Constant::new(ConstValue::None),
            ))),
        );
        let r_to = LLADTMethRepr::new(&adtmeth);
        let dummy_v = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            r_from.lowleveltype().clone(),
        ));

        let converted = pair_convert_from_to(&r_from, &r_to, &dummy_v, &mut llops)
            .expect("pair_convert_from_to should succeed")
            .expect("matching PtrRepr -> LLADTMethRepr should commit");
        assert_eq!(converted, dummy_v);
    }

    #[test]
    fn convert_from_to_interiorptr_interiorptr_requires_dict_equality_like_rptr() {
        // rptr.py:338-343 — lowleveltype equality is not enough here;
        // upstream checks `r_from.__dict__ == r_to.__dict__`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::translator::rtyper::lltypesystem::lltype::{
            InteriorOffset, InteriorPtr, LowLevelType, StructType,
        };
        use crate::translator::rtyper::rmodel::InteriorPtrRepr;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let parent = LowLevelType::Struct(Box::new(StructType::gc(
            "S",
            vec![
                ("a".to_string(), LowLevelType::Signed),
                ("b".to_string(), LowLevelType::Signed),
            ],
        )));
        let target = LowLevelType::Signed;
        let ptr_a = InteriorPtr {
            PARENTTYPE: Box::new(parent.clone()),
            TO: Box::new(target.clone()),
            offsets: vec![InteriorOffset::Field("a".to_string())],
        };
        let ptr_b = InteriorPtr {
            PARENTTYPE: Box::new(parent),
            TO: Box::new(target),
            offsets: vec![InteriorOffset::Field("b".to_string())],
        };
        let r_from = InteriorPtrRepr::new(ptr_a.clone());
        let r_same = InteriorPtrRepr::new(ptr_a);
        let r_different = InteriorPtrRepr::new(ptr_b);
        assert_eq!(r_from.lowleveltype(), r_different.lowleveltype());
        let dummy_v = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            r_from.lowleveltype().clone(),
        ));

        let converted = pair_convert_from_to(&r_from, &r_same, &dummy_v, &mut llops)
            .expect("same dict conversion should not error")
            .expect("same dict conversion should commit");
        assert_eq!(converted, dummy_v);
        let rejected = pair_convert_from_to(&r_from, &r_different, &dummy_v, &mut llops)
            .expect("different dict conversion should not error");
        assert!(rejected.is_none());
    }

    #[test]
    fn pair_rtype_add_integer_integer_emits_int_add() {
        // rint.py:217-218 + rint.py:314-341.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue, Constant, SpaceOperation, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "add".to_string(),
                vec![
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(2),
                        LowLevelType::Signed,
                    )),
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(3),
                        LowLevelType::Signed,
                    )),
                ],
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops.clone(),
        );
        let r_signed = signed_repr();
        let r_signed_dyn: Arc<dyn Repr> = r_signed.clone();
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_signed_dyn.clone()), Some(r_signed_dyn.clone())]);
        *hop.s_result.borrow_mut() = Some(SomeValue::Integer(SomeInteger::new(false, false)));
        *hop.r_result.borrow_mut() = Some(r_signed_dyn);

        let result = pair_rtype_add(r_signed.as_ref(), r_signed.as_ref(), &hop)
            .expect("int add should rtype")
            .expect("int add should return a value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "int_add");
    }

    #[test]
    fn pair_rtype_add_ovf_integer_integer_emits_int_add_ovf() {
        // rint.py:221-230 + rint.py:330-335 — overflow add goes through
        // `_rtype_template`, marks the llop as the raising point, and
        // emits `int_add_ovf` when neither argument is proven non-negative.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue, Constant, SpaceOperation, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "add_ovf".to_string(),
                vec![
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(2),
                        LowLevelType::Signed,
                    )),
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(-3),
                        LowLevelType::Signed,
                    )),
                ],
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops.clone(),
        );
        let r_signed = signed_repr();
        let r_signed_dyn: Arc<dyn Repr> = r_signed.clone();
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_signed_dyn.clone()), Some(r_signed_dyn.clone())]);
        *hop.s_result.borrow_mut() = Some(SomeValue::Integer(SomeInteger::new(false, false)));
        *hop.r_result.borrow_mut() = Some(r_signed_dyn);

        let result = pair_rtype_add_ovf(r_signed.as_ref(), r_signed.as_ref(), &hop)
            .expect("int add_ovf should rtype")
            .expect("int add_ovf should return a value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert!(ops._called_exception_is_here_or_cannot_occur);
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "int_add_ovf");
    }

    #[test]
    fn pair_rtype_add_ovf_nonneg_arg_emits_int_add_nonneg_ovf() {
        // rint.py:223-229 — for signed-int results, a proven non-negative
        // operand selects `add_nonneg_ovf`; if it is the first operand,
        // the copied hop swaps the first two args before template lowering.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue, Constant, SpaceOperation, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "add_ovf".to_string(),
                vec![
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(4),
                        LowLevelType::Signed,
                    )),
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(-5),
                        LowLevelType::Signed,
                    )),
                ],
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops.clone(),
        );
        let r_signed = signed_repr();
        let r_signed_dyn: Arc<dyn Repr> = r_signed.clone();
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Integer(SomeInteger::new(true, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_signed_dyn.clone()), Some(r_signed_dyn.clone())]);
        *hop.s_result.borrow_mut() = Some(SomeValue::Integer(SomeInteger::new(false, false)));
        *hop.r_result.borrow_mut() = Some(r_signed_dyn);

        let result = pair_rtype_add_ovf(r_signed.as_ref(), r_signed.as_ref(), &hop)
            .expect("int add_ovf should rtype")
            .expect("int add_ovf should return a value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert!(ops._called_exception_is_here_or_cannot_occur);
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "int_add_nonneg_ovf");
    }

    #[test]
    fn pair_rtype_div_integer_integer_emits_ll_int_py_div_direct_call() {
        // rint.py:246-256 + rint.py:344-387 — integer `div` is an alias
        // for floordiv and must lower through `_rtype_call_helper`, not
        // through FloatRepr's `float_truediv` MRO fallback.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue, Constant, SpaceOperation, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::{_ptr_obj, LowLevelType};
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "div".to_string(),
                vec![
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(7),
                        LowLevelType::Signed,
                    )),
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(3),
                        LowLevelType::Signed,
                    )),
                ],
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops.clone(),
        );
        let r_signed = signed_repr();
        let r_signed_dyn: Arc<dyn Repr> = r_signed.clone();
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_signed_dyn.clone()), Some(r_signed_dyn.clone())]);
        *hop.s_result.borrow_mut() = Some(SomeValue::Integer(SomeInteger::new(false, false)));
        *hop.r_result.borrow_mut() = Some(r_signed_dyn);

        let result = pair_rtype_div(r_signed.as_ref(), r_signed.as_ref(), &hop)
            .expect("int div should rtype")
            .expect("int div should return a value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert!(ops._called_exception_is_here_or_cannot_occur);
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert_eq!(ops.ops[0].args.len(), 3);
        let Hlvalue::Constant(c_func) = &ops.ops[0].args[0] else {
            panic!("direct_call first arg must be the function constant");
        };
        let ConstValue::LLPtr(ptr) = &c_func.value else {
            panic!("direct_call first arg must be a low-level function pointer");
        };
        let _ptr_obj::Func(func) = ptr._obj().expect("function pointer must be concrete") else {
            panic!("direct_call first arg must point to a function");
        };
        assert_eq!(func._name, "ll_int_py_div");
    }

    #[test]
    fn pair_rtype_add_integer_float_reaches_float_pair_via_mro() {
        // IntegerRepr inherits FloatRepr upstream. For `(IntegerRepr,
        // FloatRepr)` there is no integer-pair `rtype_add`; pair_mro
        // reaches `(FloatRepr, FloatRepr)`, coercing the integer arg to
        // Float before emitting `float_add`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeFloat, SomeInteger, SomeValue};
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let mut v_int = Variable::new();
        v_int.set_concretetype(Some(LowLevelType::Signed));
        let mut v_float = Variable::new();
        v_float.set_concretetype(Some(LowLevelType::Float));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "add".to_string(),
                vec![Hlvalue::Variable(v_int), Hlvalue::Variable(v_float)],
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops.clone(),
        );
        let r_signed = signed_repr();
        let r_float = float_repr();
        let r_signed_dyn: Arc<dyn Repr> = r_signed.clone();
        let r_float_dyn: Arc<dyn Repr> = r_float.clone();
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::Float(SomeFloat::new()),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_signed_dyn), Some(r_float_dyn.clone())]);
        *hop.s_result.borrow_mut() = Some(SomeValue::Float(SomeFloat::new()));
        *hop.r_result.borrow_mut() = Some(r_float_dyn);

        let result = pair_rtype_add(r_signed.as_ref(), r_float.as_ref(), &hop)
            .expect("mixed add should rtype")
            .expect("mixed add should return a value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 2);
        assert_eq!(ops.ops[0].opname, "cast_int_to_float");
        assert_eq!(ops.ops[1].opname, "float_add");
    }

    #[test]
    fn pair_function_function_convert_from_to_is_identity_and_emits_no_ops() {
        // rpbc.py:373-375 — pairtype(FunctionRepr,
        // FunctionRepr).convert_from_to is upstream `return v`. Both
        // reprs are Void-typed; the conversion should be a pass-through.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::annotator::description::{DescEntry, FunctionDesc};
        use crate::annotator::model::SomePBC;
        use crate::flowspace::argument::Signature;
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rpbc::FunctionRepr;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::cell::RefCell as StdRefCell;
        use std::rc::Rc;

        fn f_entry(bk: &Rc<Bookkeeper>, name: &str) -> DescEntry {
            DescEntry::Function(Rc::new(StdRefCell::new(FunctionDesc::new(
                bk.clone(),
                None,
                name,
                Signature::new(vec![], None, None),
                None,
                None,
            ))))
        }

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);

        let r_from: Arc<dyn Repr> = Arc::new(
            FunctionRepr::new(
                &rtyper,
                SomePBC::new(vec![f_entry(&ann.bookkeeper, "f")], false),
            )
            .unwrap(),
        );
        let r_to: Arc<dyn Repr> = Arc::new(
            FunctionRepr::new(
                &rtyper,
                SomePBC::new(vec![f_entry(&ann.bookkeeper, "g")], false),
            )
            .unwrap(),
        );

        let input_var = Variable::new();
        input_var.set_concretetype(Some(
            crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void,
        ));
        let v_in = Hlvalue::Variable(input_var);

        let converted =
            pair_convert_from_to(r_from.as_ref(), r_to.as_ref(), &v_in, &mut llops).unwrap();
        assert_eq!(converted, Some(v_in));
        assert!(
            llops.ops.is_empty(),
            "identity conversion should not emit ops"
        );
    }

    #[test]
    fn pair_functions_pbc_to_function_returns_void_none_constant() {
        // rpbc.py:381-383 — FunctionsPBCRepr → FunctionRepr is
        // `inputconst(Void, None)`, irrespective of the source variable.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::annotator::description::{DescEntry, FunctionDesc, GraphCacheKey};
        use crate::annotator::model::{SomeInteger, SomePBC, SomeValue};
        use crate::flowspace::argument::Signature;
        use crate::flowspace::model::{ConstValue, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rpbc::{FunctionRepr, FunctionsPBCRepr};
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::cell::RefCell as StdRefCell;
        use std::rc::Rc;

        fn f_entry(bk: &Rc<Bookkeeper>, name: &str) -> DescEntry {
            DescEntry::Function(Rc::new(StdRefCell::new(FunctionDesc::new(
                bk.clone(),
                None,
                name,
                Signature::new(vec!["x".to_string()], None, None),
                None,
                None,
            ))))
        }

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);

        let fd_f = match f_entry(&ann.bookkeeper, "f") {
            DescEntry::Function(rc) => rc,
            _ => unreachable!(),
        };
        let fd_g = match f_entry(&ann.bookkeeper, "g") {
            DescEntry::Function(rc) => rc,
            _ => unreachable!(),
        };
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow().cache.borrow_mut().insert(
                GraphCacheKey::None,
                crate::translator::rtyper::rpbc::tests::make_pygraph(name),
            );
        }
        let args = crate::annotator::argument::ArgumentsForTranslation::new(
            vec![SomeValue::Integer(SomeInteger::default())],
            None,
            None,
        );
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &args,
            &SomeValue::Impossible,
            None,
        )
        .unwrap();

        let s_from = SomePBC::new(
            vec![DescEntry::Function(fd_f), DescEntry::Function(fd_g.clone())],
            false,
        );
        let r_from: Arc<dyn Repr> = Arc::new(FunctionsPBCRepr::new(&rtyper, s_from).unwrap());
        let r_to: Arc<dyn Repr> = Arc::new(
            FunctionRepr::new(
                &rtyper,
                SomePBC::new(vec![DescEntry::Function(fd_g)], false),
            )
            .unwrap(),
        );

        let input_var = Variable::new();
        input_var.set_concretetype(Some(LowLevelType::Void));
        let v_in = Hlvalue::Variable(input_var);
        let converted =
            pair_convert_from_to(r_from.as_ref(), r_to.as_ref(), &v_in, &mut llops).unwrap();
        match converted {
            Some(Hlvalue::Constant(c)) => {
                assert!(matches!(c.value, ConstValue::None));
                assert_eq!(c.concretetype, Some(LowLevelType::Void));
            }
            other => panic!("expected Void None constant, got {other:?}"),
        }
    }

    #[test]
    fn pair_functions_pbc_identity_passes_through_when_lowleveltypes_match() {
        // rpbc.py:385-390 — two FunctionsPBCRepr reprs with matching
        // lowleveltype collapse to `return v`. Re-using the same
        // FunctionsPBCRepr on both ends is the simplest way to trigger
        // the identity arm.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::annotator::description::{DescEntry, FunctionDesc, GraphCacheKey};
        use crate::annotator::model::{SomeInteger, SomePBC, SomeValue};
        use crate::flowspace::argument::Signature;
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rpbc::FunctionsPBCRepr;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::cell::RefCell as StdRefCell;
        use std::rc::Rc;

        fn f_entry(bk: &Rc<Bookkeeper>, name: &str) -> DescEntry {
            DescEntry::Function(Rc::new(StdRefCell::new(FunctionDesc::new(
                bk.clone(),
                None,
                name,
                Signature::new(vec!["x".to_string()], None, None),
                None,
                None,
            ))))
        }
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);

        let fd_f = match f_entry(&ann.bookkeeper, "f") {
            DescEntry::Function(rc) => rc,
            _ => unreachable!(),
        };
        let fd_g = match f_entry(&ann.bookkeeper, "g") {
            DescEntry::Function(rc) => rc,
            _ => unreachable!(),
        };
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow().cache.borrow_mut().insert(
                GraphCacheKey::None,
                crate::translator::rtyper::rpbc::tests::make_pygraph(name),
            );
        }
        let args = crate::annotator::argument::ArgumentsForTranslation::new(
            vec![SomeValue::Integer(SomeInteger::default())],
            None,
            None,
        );
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &args,
            &SomeValue::Impossible,
            None,
        )
        .unwrap();

        let s_pbc = SomePBC::new(
            vec![DescEntry::Function(fd_f), DescEntry::Function(fd_g)],
            false,
        );
        let r: Arc<dyn Repr> = Arc::new(FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap());

        let input_var = Variable::new();
        input_var.set_concretetype(Some(r.lowleveltype().clone()));
        let v_in = Hlvalue::Variable(input_var);
        let converted = pair_convert_from_to(r.as_ref(), r.as_ref(), &v_in, &mut llops).unwrap();
        assert_eq!(converted, Some(v_in));
        assert!(llops.ops.is_empty());
    }
}
