//! RPython `rpython/rtyper/lltypesystem/rstr.py` — minimal slice for
//! the `Ptr(STR)` / `Ptr(UNICODE)` low-level types that
//! `OBJECT_VTABLE.name` (rclass.py:171) and
//! `ClassesPBCRepr.rtype_getattr('__name__')` (rpbc.py:975-981)
//! consume, plus the `UNICODE` data shape that the eventual
//! `UnicodeRepr` port will dereference.
//!
//! Upstream rstr.py weighs ~3000 LOC and houses every operation that
//! rtypes Python's `str` / `unicode` types (`StringRepr`,
//! `UnicodeRepr`, `LLHelpers`, `ll_strconcat`, ...). Pyre lands only
//! the data shape this commit needs:
//!
//! ```python
//! STR = GcForwardReference()
//! UNICODE = GcForwardReference()
//! ...
//! STR.become(GcStruct('rpy_string',
//!                     ('hash', Signed),
//!                     ('chars', Array(Char, hints={'immutable': True,
//!                                     'extra_item_after_alloc': 1})),
//!                     adtmeths={...},
//!                     hints={'remove_hash': True}))
//! UNICODE.become(GcStruct('rpy_unicode',
//!                         ('hash', Signed),
//!                         ('chars', Array(UniChar, hints={'immutable': True})),
//!                         adtmeths={...},
//!                         hints={'remove_hash': True}))
//! ```
//!
//! The full repr port (`AbstractStringRepr`/`AbstractUnicodeRepr`
//! methods, `LLHelpers.ll_*` helpers, comparison, formatting,
//! adtmeths) is deferred to its own follow-on slices — see
//! `~/.claude/projects/.../memory/item3_abstractstringrepr_epic_plan.md`.
//!
//! ## Why a separate module today
//!
//! Both `OBJECT_VTABLE.name` (rclass.py) and
//! `ClassesPBCRepr.rtype_getattr('__name__')` (rpbc.py) want
//! `Ptr(STR)`. Defining `STR` / `UNICODE` here keeps the dependency
//! direction `rclass / rpbc → rstr → lltype` clean and gives a
//! natural home for the eventual full `StringRepr` / `UnicodeRepr`
//! port.

use std::sync::LazyLock;

use crate::flowspace::model::{
    Block, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, _ptr_obj, ArrayType, ForwardReference, LowLevelType, LowLevelValue, MallocFlavor, Ptr,
    PtrTarget, StructType, malloc,
};
use crate::translator::rtyper::rtyper::{
    constant_with_lltype, exception_args, helper_pygraph_from_graph, variable_with_lltype,
};

/// RPython `STR = GcForwardReference()` resolved via
/// `STR.become(GcStruct('rpy_string', ('hash', Signed), ('chars',
/// Array(Char, ...)), hints={'remove_hash': True}))`
/// (rstr.py:32 + rstr.py:1226-1237).
///
/// Pyre lands the data shape — the `adtmeths` table (mallocstr,
/// emptystrfun, copy_string_contents, ll_strhash, ll_length, ll_find,
/// ll_rfind) is deferred. `OBJECT_VTABLE.name`'s `Ptr(STR)` only
/// needs the structural shape today; concrete consumers that emit
/// `getfield` / dereferences will lower through generic `_ptr.getattr`
/// against the resolved Struct.
pub static STR: LazyLock<LowLevelType> = LazyLock::new(|| {
    // Pyre deviation: upstream `Array(Char, hints={'immutable': True,
    // 'extra_item_after_alloc': 1})` carries hints that drive
    // RPython's array layout / immutability tracking. The Rust port
    // omits those hints today — `ArrayType::new` does not accept a
    // hints kwarg, and the structural shape (Array of Char) is the
    // only piece consumed by `ClassesPBCRepr.rtype_getattr('__name__')`
    // and `OBJECT_VTABLE.name`. Adding a `with_hints` constructor is
    // a follow-up alongside the full StringRepr port.
    let chars = ArrayType::new(LowLevelType::Char);
    let body = StructType::gc_with_hints(
        "rpy_string",
        vec![
            ("hash".into(), LowLevelType::Signed),
            ("chars".into(), LowLevelType::Array(Box::new(chars))),
        ],
        vec![(
            "remove_hash".into(),
            crate::flowspace::model::ConstValue::Bool(true),
        )],
    );
    let fwd = ForwardReference::gc();
    fwd.r#become(LowLevelType::Struct(Box::new(body)))
        .expect("STR.become should succeed");
    LowLevelType::ForwardReference(Box::new(fwd))
});

/// RPython `Ptr(STR)` — surfaces as the lltype of
/// `OBJECT_VTABLE.name` (rclass.py:171) and the result type of
/// `genop('getfield', [vcls, 'name'], resulttype=Ptr(rstr.STR))` in
/// `ClassesPBCRepr.rtype_getattr('__name__')` (rpbc.py:980-981).
pub static STRPTR: LazyLock<LowLevelType> = LazyLock::new(|| {
    let str_t = STR.clone();
    let LowLevelType::ForwardReference(fwd) = str_t else {
        panic!("STR must be a ForwardReference");
    };
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::ForwardReference(*fwd),
    }))
});

/// RPython `UNICODE = GcForwardReference()` resolved via
/// `UNICODE.become(GcStruct('rpy_unicode', ('hash', Signed), ('chars',
/// Array(UniChar, hints={'immutable': True})), hints={'remove_hash':
/// True}))` (rstr.py:33 + rstr.py:1238-1246).
///
/// Mirror of [`STR`] with the `chars` element type swapped from
/// `Char` to `UniChar`. The `extra_item_after_alloc` hint that STR
/// carries (for the trailing NUL slot) is omitted upstream for
/// UNICODE — only `immutable` is set on the array. The Rust port
/// elides both today (see [`STR`] docstring); structural shape is
/// what the eventual `UnicodeRepr` port consumes.
pub static UNICODE: LazyLock<LowLevelType> = LazyLock::new(|| {
    let chars = ArrayType::new(LowLevelType::UniChar);
    let body = StructType::gc_with_hints(
        "rpy_unicode",
        vec![
            ("hash".into(), LowLevelType::Signed),
            ("chars".into(), LowLevelType::Array(Box::new(chars))),
        ],
        vec![(
            "remove_hash".into(),
            crate::flowspace::model::ConstValue::Bool(true),
        )],
    );
    let fwd = ForwardReference::gc();
    fwd.r#become(LowLevelType::Struct(Box::new(body)))
        .expect("UNICODE.become should succeed");
    LowLevelType::ForwardReference(Box::new(fwd))
});

/// RPython `Ptr(UNICODE)` — surfaces as the lltype of
/// `UnicodeRepr.lowleveltype` (rstr.py:248) and is the result type of
/// every `gendirectcall(self.ll.ll_*unicode*, ...)` inside
/// `AbstractUnicodeRepr` methods.
pub static UNICODEPTR: LazyLock<LowLevelType> = LazyLock::new(|| {
    let unicode_t = UNICODE.clone();
    let LowLevelType::ForwardReference(fwd) = unicode_t else {
        panic!("UNICODE must be a ForwardReference");
    };
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::ForwardReference(*fwd),
    }))
});

/// RPython `alloc_array_name(name)` (rclass.py:187-188):
///
/// ```python
/// def alloc_array_name(name):
///     return rstr.string_repr.convert_const(name)
/// ```
///
/// Materialises an immortal `_ptr` to a `STR` GcStruct carrying the
/// class name. This is the narrow `StringRepr.convert_const(str)` slice
/// needed by `OBJECT_VTABLE.name`: allocate `STR` with a chars array of
/// `name.len()`, fill the byte chars, and initialise `hash` to 0. The
/// full `StringRepr` hash precomputation and adtmeth table stay in the
/// future full rstr port.
pub fn alloc_array_name(name: &str) -> Result<_ptr, String> {
    let str_body = match STR.clone() {
        LowLevelType::ForwardReference(fwd) => fwd
            .resolved()
            .ok_or_else(|| "alloc_array_name: STR ForwardReference is not resolved".to_string())?,
        other => other,
    };
    let byte_len = name.len();
    let mut ptr = malloc(str_body, Some(byte_len), MallocFlavor::Gc, true)?;
    ptr.setattr("hash", LowLevelValue::Signed(0))?;
    let Some(obj) = ptr
        ._obj0
        .as_mut()
        .map_err(|_| "alloc_array_name: delayed STR pointer cannot be initialised".to_string())?
    else {
        return Err("alloc_array_name: malloc returned null STR pointer".to_string());
    };
    let _ptr_obj::Struct(s) = obj else {
        return Err("alloc_array_name: STR pointer does not target a Struct".to_string());
    };
    let chars_value = s
        ._getattr("chars")
        .ok_or_else(|| "alloc_array_name: STR struct has no chars field".to_string())?;
    let LowLevelValue::Array(chars) = chars_value else {
        return Err("alloc_array_name: STR struct chars field is not an Array".to_string());
    };
    for (i, byte) in name.bytes().enumerate() {
        chars.setitem(i, LowLevelValue::Char(byte as char));
    }
    Ok(ptr)
}

/// `unicode_repr.convert_const(value)` mirror of [`alloc_array_name`]
/// for the `UNICODE` GcStruct. Materialises an immortal `_ptr` to a
/// `UNICODE` carrying the Unicode codepoint sequence of `value` and
/// initialises `hash` to 0. The full `UnicodeRepr.convert_const` and
/// hash precomputation stay in the future full rstr port.
pub fn alloc_array_unicode(value: &str) -> Result<_ptr, String> {
    let unicode_body = match UNICODE.clone() {
        LowLevelType::ForwardReference(fwd) => fwd.resolved().ok_or_else(|| {
            "alloc_array_unicode: UNICODE ForwardReference is not resolved".to_string()
        })?,
        other => other,
    };
    let codepoint_count = value.chars().count();
    let mut ptr = malloc(unicode_body, Some(codepoint_count), MallocFlavor::Gc, true)?;
    ptr.setattr("hash", LowLevelValue::Signed(0))?;
    let Some(obj) = ptr._obj0.as_mut().map_err(|_| {
        "alloc_array_unicode: delayed UNICODE pointer cannot be initialised".to_string()
    })?
    else {
        return Err("alloc_array_unicode: malloc returned null UNICODE pointer".to_string());
    };
    let _ptr_obj::Struct(s) = obj else {
        return Err("alloc_array_unicode: UNICODE pointer does not target a Struct".to_string());
    };
    let chars_value = s
        ._getattr("chars")
        .ok_or_else(|| "alloc_array_unicode: UNICODE struct has no chars field".to_string())?;
    let LowLevelValue::Array(chars) = chars_value else {
        return Err("alloc_array_unicode: UNICODE struct chars field is not an Array".to_string());
    };
    for (i, ch) in value.chars().enumerate() {
        chars.setitem(i, LowLevelValue::UniChar(ch));
    }
    Ok(ptr)
}

// ____________________________________________________________
// LLHelpers — `lltypesystem/rstr.py:307` `class LLHelpers(AbstractLLHelpers)`.
//
// Each helper is synthesised as a single-block low-level graph the
// rtyper inserts via `gendirectcall`. The helpers correspond to
// upstream `LLHelpers.ll_*` static methods.

/// Build the `getsubstruct('chars')` + `getarraysize` op pair shared
/// by `ll_strlen` and `ll_unilen`. The `chars` field is an inline
/// `Array(Char)` / `Array(UniChar)` inside the `STR`/`UNICODE`
/// GcStruct, so RPython emits `getsubstruct` (returning an interior
/// `Ptr(Array(...))`) rather than `getfield`. Compare
/// `rmodel.rs:1290-1306`'s `is_container_type` branch for the same
/// dispatch on the rtyper side.
fn emit_chars_length_ops(
    startblock: &crate::flowspace::model::BlockRef,
    arg: Hlvalue,
    chars_array_ptr_lltype: LowLevelType,
) -> Hlvalue {
    let v_chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![
            arg,
            constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
        ],
        Hlvalue::Variable(v_chars.clone()),
    ));
    let v_len = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(v_chars)],
        Hlvalue::Variable(v_len.clone()),
    ));
    Hlvalue::Variable(v_len)
}

/// Build the chars Array's `Ptr(Array(...))` lltype for a resolved
/// `STR`/`UNICODE` `ForwardReference`. The result wraps the field's
/// inline `Array(Char|UniChar)` in `Ptr` so the synthesised
/// `getsubstruct` op's result variable carries a valid
/// `SomePtr`-shaped annotation (upstream `lltype.py:1530+`'s
/// `SomePtr.__init__` rejects bare container types — see
/// `llannotation.rs:159-167`).
fn chars_array_ptr_lltype_from_strptr(
    ptr_lltype: &LowLevelType,
) -> Result<LowLevelType, TyperError> {
    let LowLevelType::Ptr(ptr) = ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_strlen helper expects Ptr(STR/UNICODE), got {ptr_lltype:?}"
        )));
    };
    let body = match &ptr.TO {
        PtrTarget::ForwardReference(fwd) => fwd
            .resolved()
            .ok_or_else(|| TyperError::message("STR/UNICODE ForwardReference not resolved"))?,
        PtrTarget::Struct(s) => LowLevelType::Struct(Box::new(s.clone())),
        other => {
            return Err(TyperError::message(format!(
                "ll_strlen helper Ptr.TO must target STR/UNICODE struct, got {other:?}"
            )));
        }
    };
    let LowLevelType::Struct(body) = body else {
        return Err(TyperError::message(
            "STR/UNICODE ForwardReference must resolve to Struct",
        ));
    };
    let chars_field = body
        ._flds
        .get("chars")
        .cloned()
        .ok_or_else(|| TyperError::message("STR/UNICODE struct has no chars field"))?;
    let LowLevelType::Array(arr) = chars_field else {
        return Err(TyperError::message(
            "STR/UNICODE chars field must be Array(Char|UniChar)",
        ));
    };
    Ok(LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(*arr),
    })))
}

/// Resolve the `Ptr(STR/UNICODE)` wrapper to the concrete
/// `LowLevelType::Struct(body)` payload that names the GcStruct.
/// `malloc_varsize` SpaceOperation's first arg requires this concrete
/// struct lltype (mirrors upstream `cTEMP = inputconst(Void, TEMPBUF)`
/// at rstr.py:1129 where `TEMPBUF` is the resolved GcStruct itself,
/// not its Ptr wrapper).
fn struct_lltype_from_strptr(ptr_lltype: &LowLevelType) -> Result<LowLevelType, TyperError> {
    let LowLevelType::Ptr(ptr) = ptr_lltype else {
        return Err(TyperError::message(format!(
            "struct_lltype_from_strptr expects Ptr(STR/UNICODE), got {ptr_lltype:?}"
        )));
    };
    let body = match &ptr.TO {
        PtrTarget::ForwardReference(fwd) => fwd
            .resolved()
            .ok_or_else(|| TyperError::message("STR/UNICODE ForwardReference not resolved"))?,
        PtrTarget::Struct(s) => LowLevelType::Struct(Box::new(s.clone())),
        other => {
            return Err(TyperError::message(format!(
                "struct_lltype_from_strptr Ptr.TO must target STR/UNICODE struct, got {other:?}"
            )));
        }
    };
    if !matches!(body, LowLevelType::Struct(_)) {
        return Err(TyperError::message(
            "STR/UNICODE ForwardReference must resolve to Struct",
        ));
    }
    Ok(body)
}

/// Synthesise the helper graph for `LLHelpers.ll_strlen` /
/// `LLHelpers.ll_length` (`lltypesystem/rstr.py:351-352, 417-418`):
///
/// ```python
/// @staticmethod
/// def ll_strlen(s):
///     return len(s.chars)
/// ```
///
/// Single-block graph:
/// `getsubstruct(s, 'chars') -> Ptr(Array); getarraysize(...) -> Signed`.
/// `name` is the helper identity (`"ll_strlen"` / `"ll_unilen"`)
/// and `ptr_lltype` selects between `Ptr(STR)` and `Ptr(UNICODE)`.
pub(crate) fn build_ll_strlen_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let arg = variable_with_lltype("s", ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_len = emit_chars_length_ops(&startblock, Hlvalue::Variable(arg), chars_array_ptr_lltype);
    startblock.closeblock(vec![
        Link::new(vec![v_len], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise `LLHelpers.ll_chr2str` (`lltypesystem/rstr.py:363-369`):
///
/// ```python
/// def ll_chr2str(ch):
///     if typeOf(ch) is Char:
///         malloc = mallocstr
///     else:
///         malloc = mallocunicode
///     s = malloc(1)
///     s.chars[0] = ch
///     return s
/// ```
///
/// Pyre specialises the upstream `typeOf(ch)` branch at helper-build
/// time via `(ptr_lltype, elem_lltype)`: `Char -> Ptr(STR)` or
/// `UniChar -> Ptr(UNICODE)`. The graph is a single block:
/// `malloc_varsize(STR|UNICODE, gc, 1)`, `getsubstruct(..., 'chars')`,
/// `setarrayitem(chars, 0, ch)`, return the allocated string.
pub(crate) fn build_ll_chr2str_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    elem_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_chr2str helper expects Ptr(Array(Char|UniChar)), got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_chr2str helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    if arr.OF != elem_lltype {
        return Err(TyperError::message(format!(
            "ll_chr2str helper element mismatch: chars array stores {:?}, argument is {elem_lltype:?}",
            arr.OF
        )));
    }
    if !matches!(elem_lltype, LowLevelType::Char | LowLevelType::UniChar) {
        return Err(TyperError::message(format!(
            "ll_chr2str helper unsupported char element type {elem_lltype:?}"
        )));
    }

    let struct_lltype = struct_lltype_from_strptr(&ptr_lltype)?;
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let ch = variable_with_lltype("ch", elem_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(ch.clone())]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let newstr = variable_with_lltype("s", ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc_varsize",
        vec![
            lowlevel_type_const(struct_lltype),
            gc_flavor_const()?,
            signed_const(1),
        ],
        Hlvalue::Variable(newstr.clone()),
    ));

    let newchars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(newstr.clone()), chars_field_const()],
        Hlvalue::Variable(newchars.clone()),
    ));

    let void_set = variable_with_lltype("set0", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(newchars),
            signed_const(0),
            Hlvalue::Variable(ch),
        ],
        Hlvalue::Variable(void_set),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(newstr)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ch".to_string()],
        func,
    ))
}

/// Synthesise `LLHelpers.ll_str2unicode` (`lltypesystem/rstr.py:375-383`):
///
/// ```python
/// def ll_str2unicode(str):
///     lgt = len(str.chars)
///     s = mallocunicode(lgt)
///     for i in range(lgt):
///         if ord(str.chars[i]) > 127:
///             raise UnicodeDecodeError
///         s.chars[i] = cast_primitive(UniChar, str.chars[i])
///     return s
/// ```
///
/// 4-block CFG plus returnblock and exceptblock:
/// - **start**: read source length, allocate UNICODE with
///   `malloc_varsize`, then enter the loop at `i = 0`.
/// - **loop_cond**: `i < length`; False returns the allocated unicode.
/// - **check_char**: load `str.chars[i]`, cast to int, raise
///   `UnicodeDecodeError` when the codepoint is > 127.
/// - **store_char**: cast the ASCII code to UniChar, store it, increment
///   `i`, and jump back to `loop_cond`.
pub(crate) fn build_ll_str2unicode_helper_graph(name: &str) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let src_chars_ptr_lltype = chars_array_ptr_lltype_from_strptr(&STRPTR)?;
    let dst_chars_ptr_lltype = chars_array_ptr_lltype_from_strptr(&UNICODEPTR)?;
    let unicode_struct_lltype = struct_lltype_from_strptr(&UNICODEPTR)?;

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);
    let exc_args = exception_args("UnicodeDecodeError")?;

    let str_arg = variable_with_lltype("str", STRPTR.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(str_arg.clone())]);
    let return_var = variable_with_lltype("result", UNICODEPTR.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let src_chars_for_cond = variable_with_lltype("chars", src_chars_ptr_lltype.clone());
    let dst_uni_for_cond = variable_with_lltype("s", UNICODEPTR.clone());
    let dst_chars_for_cond = variable_with_lltype("newchars", dst_chars_ptr_lltype.clone());
    let length_for_cond = variable_with_lltype("lgt", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(src_chars_for_cond.clone()),
        Hlvalue::Variable(dst_uni_for_cond.clone()),
        Hlvalue::Variable(dst_chars_for_cond.clone()),
        Hlvalue::Variable(length_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
    ]);

    let src_chars_for_check = variable_with_lltype("chars", src_chars_ptr_lltype.clone());
    let dst_uni_for_check = variable_with_lltype("s", UNICODEPTR.clone());
    let dst_chars_for_check = variable_with_lltype("newchars", dst_chars_ptr_lltype.clone());
    let length_for_check = variable_with_lltype("lgt", LowLevelType::Signed);
    let i_for_check = variable_with_lltype("i", LowLevelType::Signed);
    let block_check_char = Block::shared(vec![
        Hlvalue::Variable(src_chars_for_check.clone()),
        Hlvalue::Variable(dst_uni_for_check.clone()),
        Hlvalue::Variable(dst_chars_for_check.clone()),
        Hlvalue::Variable(length_for_check.clone()),
        Hlvalue::Variable(i_for_check.clone()),
    ]);

    let src_chars_for_store = variable_with_lltype("chars", src_chars_ptr_lltype.clone());
    let dst_uni_for_store = variable_with_lltype("s", UNICODEPTR.clone());
    let dst_chars_for_store = variable_with_lltype("newchars", dst_chars_ptr_lltype.clone());
    let length_for_store = variable_with_lltype("lgt", LowLevelType::Signed);
    let i_for_store = variable_with_lltype("i", LowLevelType::Signed);
    let c_int_for_store = variable_with_lltype("c_int", LowLevelType::Signed);
    let block_store_char = Block::shared(vec![
        Hlvalue::Variable(src_chars_for_store.clone()),
        Hlvalue::Variable(dst_uni_for_store.clone()),
        Hlvalue::Variable(dst_chars_for_store.clone()),
        Hlvalue::Variable(length_for_store.clone()),
        Hlvalue::Variable(i_for_store.clone()),
        Hlvalue::Variable(c_int_for_store.clone()),
    ]);

    // ---- start: source chars/length, mallocunicode(length), dest chars.
    let src_chars = variable_with_lltype("chars", src_chars_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(str_arg), chars_field_const()],
        Hlvalue::Variable(src_chars.clone()),
    ));
    let length = variable_with_lltype("lgt", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(src_chars.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let newuni = variable_with_lltype("s", UNICODEPTR.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc_varsize",
        vec![
            lowlevel_type_const(unicode_struct_lltype),
            gc_flavor_const()?,
            Hlvalue::Variable(length.clone()),
        ],
        Hlvalue::Variable(newuni.clone()),
    ));
    let newchars = variable_with_lltype("newchars", dst_chars_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(newuni.clone()), chars_field_const()],
        Hlvalue::Variable(newchars.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_chars),
                Hlvalue::Variable(newuni),
                Hlvalue::Variable(newchars),
                Hlvalue::Variable(length),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ---- loop_cond: i < lgt.
    let keep_going = variable_with_lltype("keep_going", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(length_for_cond.clone()),
            ],
            Hlvalue::Variable(keep_going.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(keep_going));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_chars_for_cond),
                Hlvalue::Variable(dst_uni_for_cond.clone()),
                Hlvalue::Variable(dst_chars_for_cond),
                Hlvalue::Variable(length_for_cond),
                Hlvalue::Variable(i_for_cond),
            ],
            Some(block_check_char.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(dst_uni_for_cond)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- check_char: ord(str.chars[i]) > 127 raises UnicodeDecodeError.
    let c = variable_with_lltype("c", LowLevelType::Char);
    block_check_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(src_chars_for_check.clone()),
                Hlvalue::Variable(i_for_check.clone()),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    let c_int = variable_with_lltype("c_int", LowLevelType::Signed);
    block_check_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "cast_char_to_int",
            vec![Hlvalue::Variable(c)],
            Hlvalue::Variable(c_int.clone()),
        ));
    let too_large = variable_with_lltype("too_large", LowLevelType::Bool);
    block_check_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_gt",
            vec![Hlvalue::Variable(c_int.clone()), signed_const(127)],
            Hlvalue::Variable(too_large.clone()),
        ));
    block_check_char.borrow_mut().exitswitch = Some(Hlvalue::Variable(too_large));
    block_check_char.closeblock(vec![
        Link::new(exc_args, Some(graph.exceptblock.clone()), Some(bool_true())).into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(src_chars_for_check),
                Hlvalue::Variable(dst_uni_for_check),
                Hlvalue::Variable(dst_chars_for_check),
                Hlvalue::Variable(length_for_check),
                Hlvalue::Variable(i_for_check),
                Hlvalue::Variable(c_int),
            ],
            Some(block_store_char.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- store_char: cast ASCII code to UniChar, store, increment.
    let uc = variable_with_lltype("uc", LowLevelType::UniChar);
    block_store_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "cast_int_to_unichar",
            vec![Hlvalue::Variable(c_int_for_store)],
            Hlvalue::Variable(uc.clone()),
        ));
    let void_set = variable_with_lltype("set", LowLevelType::Void);
    block_store_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(dst_chars_for_store.clone()),
                Hlvalue::Variable(i_for_store.clone()),
                Hlvalue::Variable(uc),
            ],
            Hlvalue::Variable(void_set),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_store_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_store), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_store_char.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_chars_for_store),
                Hlvalue::Variable(dst_uni_for_store),
                Hlvalue::Variable(dst_chars_for_store),
                Hlvalue::Variable(length_for_store),
                Hlvalue::Variable(i_next),
            ],
            Some(block_loop_cond),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["str".to_string()],
        func,
    ))
}

/// Synthesise the lltypesystem half of
/// `AbstractUniCharRepr.ll_str(ch) -> str(unicode(ch))`
/// (`rtyper/rstr.py:560-562`).
///
/// Upstream goes through the default unicode-to-byte-string encoding
/// path. For a single `UniChar`, that is an ASCII check followed by a
/// one-byte `STR` allocation; non-ASCII codepoints raise
/// `UnicodeEncodeError`.
pub(crate) fn build_ll_unichr2str_helper_graph(name: &str) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let dst_chars_ptr_lltype = chars_array_ptr_lltype_from_strptr(&STRPTR)?;
    let str_struct_lltype = struct_lltype_from_strptr(&STRPTR)?;

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);
    let exc_args = exception_args("UnicodeEncodeError")?;

    let ch = variable_with_lltype("ch", LowLevelType::UniChar);
    let startblock = Block::shared(vec![Hlvalue::Variable(ch.clone())]);
    let return_var = variable_with_lltype("result", STRPTR.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let code_for_alloc = variable_with_lltype("code", LowLevelType::Signed);
    let block_alloc = Block::shared(vec![Hlvalue::Variable(code_for_alloc.clone())]);

    let code = variable_with_lltype("code", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_unichar_to_int",
        vec![Hlvalue::Variable(ch)],
        Hlvalue::Variable(code.clone()),
    ));
    let too_large = variable_with_lltype("too_large", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_gt",
        vec![Hlvalue::Variable(code.clone()), signed_const(127)],
        Hlvalue::Variable(too_large.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(too_large));
    startblock.closeblock(vec![
        Link::new(exc_args, Some(graph.exceptblock.clone()), Some(bool_true())).into_ref(),
        Link::new(
            vec![Hlvalue::Variable(code)],
            Some(block_alloc.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let byte = variable_with_lltype("byte", LowLevelType::Char);
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "cast_int_to_char",
            vec![Hlvalue::Variable(code_for_alloc)],
            Hlvalue::Variable(byte.clone()),
        ));
    let result = variable_with_lltype("result", STRPTR.clone());
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "malloc_varsize",
            vec![
                lowlevel_type_const(str_struct_lltype),
                gc_flavor_const()?,
                signed_const(1),
            ],
            Hlvalue::Variable(result.clone()),
        ));
    let chars = variable_with_lltype("chars", dst_chars_ptr_lltype);
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(result.clone()), chars_field_const()],
            Hlvalue::Variable(chars.clone()),
        ));
    let void_set = variable_with_lltype("set", LowLevelType::Void);
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(chars),
                signed_const(0),
                Hlvalue::Variable(byte),
            ],
            Hlvalue::Variable(void_set),
        ));
    block_alloc.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ch".to_string()],
        func,
    ))
}

/// Synthesise `LLHelpers.ll_int` (`lltypesystem/rstr.py:1057-1110`):
///
/// ```python
/// def ll_int(s, base):
///     if not 2 <= base <= 36:
///         raise ValueError
///     chars = s.chars
///     strlen = len(chars)
///     i = 0
///     while i < strlen and ord(chars[i]) == ord(' '):
///         i += 1
///     if not i < strlen:
///         raise ValueError
///     sign = 1
///     if ord(chars[i]) == ord('-'):
///         sign = -1
///         i += 1
///     elif ord(chars[i]) == ord('+'):
///         i += 1
///     while i < strlen and ord(chars[i]) == ord(' '):
///         i += 1
///     val = 0
///     oldpos = i
///     while i < strlen:
///         ...
///     if i == oldpos:
///         raise ValueError
///     while i < strlen and ord(chars[i]) == ord(' '):
///         i += 1
///     if not i == strlen:
///         raise ValueError
///     return sign * val
/// ```
///
/// The graph is intentionally verbose: each Python short-circuit and
/// `break` point is a separate block so the control-flow shape remains
/// close to the upstream loop-and-branch structure.
pub(crate) fn build_ll_int_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_int helper expects Ptr(Array(Char|UniChar)), got {chars_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_int helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let char_lltype = arr.OF.clone();
    let cast_op = match char_lltype {
        LowLevelType::Char => "cast_char_to_int",
        LowLevelType::UniChar => "cast_unichar_to_int",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_int helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);
    let exc_args = exception_args("ValueError")?;

    let s_arg = variable_with_lltype("s", ptr_lltype.clone());
    let base_arg = variable_with_lltype("base", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s_arg.clone()),
        Hlvalue::Variable(base_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let mk_block = |spec: Vec<(&str, LowLevelType)>| {
        Block::shared(
            spec.into_iter()
                .map(|(name, lltype)| Hlvalue::Variable(variable_with_lltype(name, lltype)))
                .collect(),
        )
    };

    let block_base_hi = mk_block(vec![
        ("s", ptr_lltype.clone()),
        ("base", LowLevelType::Signed),
    ]);
    let block_init = mk_block(vec![
        ("s", ptr_lltype.clone()),
        ("base", LowLevelType::Signed),
    ]);
    let block_leading_cond = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
    ]);
    let block_leading_char = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
    ]);
    let block_leading_inc = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
    ]);
    let block_sign_minus = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
    ]);
    let block_sign_plus = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_sign_minus_inc = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
    ]);
    let block_sign_plus_inc = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
    ]);
    let block_after_sign_cond = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
    ]);
    let block_after_sign_char = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
    ]);
    let block_after_sign_inc = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
    ]);
    let block_digit_cond = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
    ]);
    let block_digit_load = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
    ]);
    let block_lower_hi = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_lower_digit = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_upper_lo = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_upper_hi = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_upper_digit = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_num_lo = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_num_hi = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_num_digit = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("c", LowLevelType::Signed),
    ]);
    let block_digit_base = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("digit", LowLevelType::Signed),
    ]);
    let block_digit_accum = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("base", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
        ("digit", LowLevelType::Signed),
    ]);
    let block_empty_check = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
        ("oldpos", LowLevelType::Signed),
    ]);
    let block_trailing_cond = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
    ]);
    let block_trailing_char = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
    ]);
    let block_trailing_inc = mk_block(vec![
        ("chars", chars_ptr_lltype.clone()),
        ("strlen", LowLevelType::Signed),
        ("i", LowLevelType::Signed),
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
    ]);
    let block_return_value = mk_block(vec![
        ("sign", LowLevelType::Signed),
        ("val", LowLevelType::Signed),
    ]);

    let args = |block: &crate::flowspace::model::BlockRef| block.borrow().inputargs.clone();
    let var = |block: &crate::flowspace::model::BlockRef, idx: usize| {
        let Hlvalue::Variable(v) = block.borrow().inputargs[idx].clone() else {
            unreachable!("helper graph block inputargs are variables")
        };
        v
    };

    // start: lower base bound.
    let base_ge2 = variable_with_lltype("base_ge2", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![Hlvalue::Variable(base_arg.clone()), signed_const(2)],
        Hlvalue::Variable(base_ge2.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(base_ge2));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s_arg), Hlvalue::Variable(base_arg)],
            Some(block_base_hi.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let s = var(&block_base_hi, 0);
    let base = var(&block_base_hi, 1);
    let base_le36 = variable_with_lltype("base_le36", LowLevelType::Bool);
    block_base_hi
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_le",
            vec![Hlvalue::Variable(base.clone()), signed_const(36)],
            Hlvalue::Variable(base_le36.clone()),
        ));
    block_base_hi.borrow_mut().exitswitch = Some(Hlvalue::Variable(base_le36));
    block_base_hi.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s), Hlvalue::Variable(base)],
            Some(block_init.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let s = var(&block_init, 0);
    let base = var(&block_init, 1);
    let chars = variable_with_lltype("chars", chars_ptr_lltype.clone());
    block_init.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let strlen = variable_with_lltype("strlen", LowLevelType::Signed);
    block_init.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars.clone())],
        Hlvalue::Variable(strlen.clone()),
    ));
    block_init.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                signed_const(0),
            ],
            Some(block_leading_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let chars = var(&block_leading_cond, 0);
    let strlen = var(&block_leading_cond, 1);
    let base = var(&block_leading_cond, 2);
    let i = var(&block_leading_cond, 3);
    let has_char = variable_with_lltype("has_char", LowLevelType::Bool);
    block_leading_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(strlen.clone()),
            ],
            Hlvalue::Variable(has_char.clone()),
        ));
    block_leading_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(has_char));
    block_leading_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i),
            ],
            Some(block_leading_char.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_leading_char, 0);
    let strlen = var(&block_leading_char, 1);
    let base = var(&block_leading_char, 2);
    let i = var(&block_leading_char, 3);
    let ch = variable_with_lltype("ch", char_lltype.clone());
    block_leading_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Hlvalue::Variable(ch.clone()),
        ));
    let c = variable_with_lltype("c", LowLevelType::Signed);
    block_leading_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(ch)],
            Hlvalue::Variable(c.clone()),
        ));
    let is_space = variable_with_lltype("is_space", LowLevelType::Bool);
    block_leading_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(c), signed_const(32)],
            Hlvalue::Variable(is_space.clone()),
        ));
    block_leading_char.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_space));
    block_leading_char.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(strlen.clone()),
                Hlvalue::Variable(base.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Some(block_leading_inc.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i),
            ],
            Some(block_sign_minus.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_leading_inc, 0);
    let strlen = var(&block_leading_inc, 1);
    let base = var(&block_leading_inc, 2);
    let i = var(&block_leading_inc, 3);
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_leading_inc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_leading_inc.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i_next),
            ],
            Some(block_leading_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let chars = var(&block_sign_minus, 0);
    let strlen = var(&block_sign_minus, 1);
    let base = var(&block_sign_minus, 2);
    let i = var(&block_sign_minus, 3);
    let ch = variable_with_lltype("ch", char_lltype.clone());
    block_sign_minus
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Hlvalue::Variable(ch.clone()),
        ));
    let c = variable_with_lltype("c", LowLevelType::Signed);
    block_sign_minus
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(ch)],
            Hlvalue::Variable(c.clone()),
        ));
    let is_minus = variable_with_lltype("is_minus", LowLevelType::Bool);
    block_sign_minus
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(c.clone()), signed_const(45)],
            Hlvalue::Variable(is_minus.clone()),
        ));
    block_sign_minus.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_minus));
    block_sign_minus.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(strlen.clone()),
                Hlvalue::Variable(base.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Some(block_sign_minus_inc.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i),
                Hlvalue::Variable(c),
            ],
            Some(block_sign_plus.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_sign_plus, 0);
    let strlen = var(&block_sign_plus, 1);
    let base = var(&block_sign_plus, 2);
    let i = var(&block_sign_plus, 3);
    let c = var(&block_sign_plus, 4);
    let is_plus = variable_with_lltype("is_plus", LowLevelType::Bool);
    block_sign_plus
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(c), signed_const(43)],
            Hlvalue::Variable(is_plus.clone()),
        ));
    block_sign_plus.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_plus));
    block_sign_plus.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(strlen.clone()),
                Hlvalue::Variable(base.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Some(block_sign_plus_inc.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i),
                signed_const(1),
            ],
            Some(block_after_sign_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    for (block_inc, sign_value) in [
        (block_sign_minus_inc.clone(), -1),
        (block_sign_plus_inc.clone(), 1),
    ] {
        let chars = var(&block_inc, 0);
        let strlen = var(&block_inc, 1);
        let base = var(&block_inc, 2);
        let i = var(&block_inc, 3);
        let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
        block_inc.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
        block_inc.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(chars),
                    Hlvalue::Variable(strlen),
                    Hlvalue::Variable(base),
                    Hlvalue::Variable(i_next),
                    signed_const(sign_value),
                ],
                Some(block_after_sign_cond.clone()),
                None,
            )
            .into_ref(),
        ]);
    }

    let chars = var(&block_after_sign_cond, 0);
    let strlen = var(&block_after_sign_cond, 1);
    let base = var(&block_after_sign_cond, 2);
    let i = var(&block_after_sign_cond, 3);
    let sign = var(&block_after_sign_cond, 4);
    let has_char = variable_with_lltype("has_char", LowLevelType::Bool);
    block_after_sign_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(strlen.clone()),
            ],
            Hlvalue::Variable(has_char.clone()),
        ));
    block_after_sign_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(has_char));
    block_after_sign_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(strlen.clone()),
                Hlvalue::Variable(base.clone()),
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(sign.clone()),
            ],
            Some(block_after_sign_char.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(sign),
                signed_const(0),
                Hlvalue::Variable(i),
            ],
            Some(block_digit_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_after_sign_char, 0);
    let strlen = var(&block_after_sign_char, 1);
    let base = var(&block_after_sign_char, 2);
    let i = var(&block_after_sign_char, 3);
    let sign = var(&block_after_sign_char, 4);
    let ch = variable_with_lltype("ch", char_lltype.clone());
    block_after_sign_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Hlvalue::Variable(ch.clone()),
        ));
    let c = variable_with_lltype("c", LowLevelType::Signed);
    block_after_sign_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(ch)],
            Hlvalue::Variable(c.clone()),
        ));
    let is_space = variable_with_lltype("is_space", LowLevelType::Bool);
    block_after_sign_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(c), signed_const(32)],
            Hlvalue::Variable(is_space.clone()),
        ));
    block_after_sign_char.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_space));
    block_after_sign_char.closeblock(vec![
        Link::new(
            args(&block_after_sign_char),
            Some(block_after_sign_inc.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(sign),
                signed_const(0),
                Hlvalue::Variable(i),
            ],
            Some(block_digit_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_after_sign_inc, 0);
    let strlen = var(&block_after_sign_inc, 1);
    let base = var(&block_after_sign_inc, 2);
    let i = var(&block_after_sign_inc, 3);
    let sign = var(&block_after_sign_inc, 4);
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_after_sign_inc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_after_sign_inc.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i_next),
                Hlvalue::Variable(sign),
            ],
            Some(block_after_sign_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let chars = var(&block_digit_cond, 0);
    let strlen = var(&block_digit_cond, 1);
    let _base = var(&block_digit_cond, 2);
    let i = var(&block_digit_cond, 3);
    let sign = var(&block_digit_cond, 4);
    let val = var(&block_digit_cond, 5);
    let oldpos = var(&block_digit_cond, 6);
    let has_char = variable_with_lltype("has_char", LowLevelType::Bool);
    block_digit_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(strlen.clone()),
            ],
            Hlvalue::Variable(has_char.clone()),
        ));
    block_digit_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(has_char));
    block_digit_cond.closeblock(vec![
        Link::new(
            args(&block_digit_cond),
            Some(block_digit_load.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(i),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
                Hlvalue::Variable(oldpos),
            ],
            Some(block_empty_check.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_digit_load, 0);
    let strlen = var(&block_digit_load, 1);
    let base = var(&block_digit_load, 2);
    let i = var(&block_digit_load, 3);
    let sign = var(&block_digit_load, 4);
    let val = var(&block_digit_load, 5);
    let oldpos = var(&block_digit_load, 6);
    let ch = variable_with_lltype("ch", char_lltype.clone());
    block_digit_load
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Hlvalue::Variable(ch.clone()),
        ));
    let c = variable_with_lltype("c", LowLevelType::Signed);
    block_digit_load
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(ch)],
            Hlvalue::Variable(c.clone()),
        ));
    let ge_a = variable_with_lltype("ge_a", LowLevelType::Bool);
    block_digit_load
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ge",
            vec![Hlvalue::Variable(c.clone()), signed_const(97)],
            Hlvalue::Variable(ge_a.clone()),
        ));
    block_digit_load.borrow_mut().exitswitch = Some(Hlvalue::Variable(ge_a));
    block_digit_load.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(strlen.clone()),
                Hlvalue::Variable(base.clone()),
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(sign.clone()),
                Hlvalue::Variable(val.clone()),
                Hlvalue::Variable(oldpos.clone()),
                Hlvalue::Variable(c.clone()),
            ],
            Some(block_lower_hi.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
                Hlvalue::Variable(oldpos),
                Hlvalue::Variable(c),
            ],
            Some(block_upper_lo.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let close_digit_test =
        |block: &crate::flowspace::model::BlockRef,
         op: &str,
         rhs: i64,
         true_target: crate::flowspace::model::BlockRef,
         false_target: crate::flowspace::model::BlockRef| {
            let c = var(block, 7);
            let ok = variable_with_lltype("ok", LowLevelType::Bool);
            block.borrow_mut().operations.push(SpaceOperation::new(
                op,
                vec![Hlvalue::Variable(c), signed_const(rhs)],
                Hlvalue::Variable(ok.clone()),
            ));
            block.borrow_mut().exitswitch = Some(Hlvalue::Variable(ok));
            block.closeblock(vec![
                Link::new(args(block), Some(true_target), Some(bool_true())).into_ref(),
                Link::new(args(block), Some(false_target), Some(bool_false())).into_ref(),
            ]);
        };

    close_digit_test(
        &block_lower_hi,
        "int_le",
        122,
        block_lower_digit.clone(),
        block_upper_lo.clone(),
    );
    close_digit_test(
        &block_upper_hi,
        "int_le",
        90,
        block_upper_digit.clone(),
        block_num_lo.clone(),
    );

    let chars = var(&block_num_hi, 0);
    let strlen = var(&block_num_hi, 1);
    let i = var(&block_num_hi, 3);
    let sign = var(&block_num_hi, 4);
    let val = var(&block_num_hi, 5);
    let oldpos = var(&block_num_hi, 6);
    let c = var(&block_num_hi, 7);
    let le_9 = variable_with_lltype("le_9", LowLevelType::Bool);
    block_num_hi
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_le",
            vec![Hlvalue::Variable(c), signed_const(57)],
            Hlvalue::Variable(le_9.clone()),
        ));
    block_num_hi.borrow_mut().exitswitch = Some(Hlvalue::Variable(le_9));
    block_num_hi.closeblock(vec![
        Link::new(
            args(&block_num_hi),
            Some(block_num_digit.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(i),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
                Hlvalue::Variable(oldpos),
            ],
            Some(block_empty_check.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let close_digit_make = |block: &crate::flowspace::model::BlockRef,
                            sub: i64,
                            add: i64,
                            target: crate::flowspace::model::BlockRef| {
        let c = var(block, 7);
        let digit_base = variable_with_lltype("digit_base", LowLevelType::Signed);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Variable(c), signed_const(sub)],
            Hlvalue::Variable(digit_base.clone()),
        ));
        let digit = variable_with_lltype("digit", LowLevelType::Signed);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(digit_base), signed_const(add)],
            Hlvalue::Variable(digit.clone()),
        ));
        let mut target_args = args(block);
        target_args.pop();
        target_args.push(Hlvalue::Variable(digit));
        block.closeblock(vec![Link::new(target_args, Some(target), None).into_ref()]);
    };

    close_digit_make(&block_lower_digit, 97, 10, block_digit_base.clone());
    close_digit_make(&block_upper_digit, 65, 10, block_digit_base.clone());
    close_digit_make(&block_num_digit, 48, 0, block_digit_base.clone());

    let chars = var(&block_upper_lo, 0);
    let strlen = var(&block_upper_lo, 1);
    let _base = var(&block_upper_lo, 2);
    let i = var(&block_upper_lo, 3);
    let sign = var(&block_upper_lo, 4);
    let val = var(&block_upper_lo, 5);
    let oldpos = var(&block_upper_lo, 6);
    let c = var(&block_upper_lo, 7);
    let ge_a = variable_with_lltype("ge_A", LowLevelType::Bool);
    block_upper_lo
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ge",
            vec![Hlvalue::Variable(c.clone()), signed_const(65)],
            Hlvalue::Variable(ge_a.clone()),
        ));
    block_upper_lo.borrow_mut().exitswitch = Some(Hlvalue::Variable(ge_a));
    block_upper_lo.closeblock(vec![
        Link::new(
            args(&block_upper_lo),
            Some(block_upper_hi.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(i),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
                Hlvalue::Variable(oldpos),
            ],
            Some(block_empty_check.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_num_lo, 0);
    let strlen = var(&block_num_lo, 1);
    let _base = var(&block_num_lo, 2);
    let i = var(&block_num_lo, 3);
    let sign = var(&block_num_lo, 4);
    let val = var(&block_num_lo, 5);
    let oldpos = var(&block_num_lo, 6);
    let c = var(&block_num_lo, 7);
    let ge_0 = variable_with_lltype("ge_0", LowLevelType::Bool);
    block_num_lo
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ge",
            vec![Hlvalue::Variable(c), signed_const(48)],
            Hlvalue::Variable(ge_0.clone()),
        ));
    block_num_lo.borrow_mut().exitswitch = Some(Hlvalue::Variable(ge_0));
    block_num_lo.closeblock(vec![
        Link::new(
            args(&block_num_lo),
            Some(block_num_hi.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(i),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
                Hlvalue::Variable(oldpos),
            ],
            Some(block_empty_check.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let base = var(&block_digit_base, 2);
    let digit = var(&block_digit_base, 7);
    let too_big = variable_with_lltype("too_big", LowLevelType::Bool);
    block_digit_base
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ge",
            vec![Hlvalue::Variable(digit.clone()), Hlvalue::Variable(base)],
            Hlvalue::Variable(too_big.clone()),
        ));
    block_digit_base.borrow_mut().exitswitch = Some(Hlvalue::Variable(too_big));
    block_digit_base.closeblock(vec![
        Link::new(
            vec![
                args(&block_digit_base)[0].clone(),
                args(&block_digit_base)[1].clone(),
                args(&block_digit_base)[3].clone(),
                args(&block_digit_base)[4].clone(),
                args(&block_digit_base)[5].clone(),
                args(&block_digit_base)[6].clone(),
            ],
            Some(block_empty_check.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            args(&block_digit_base),
            Some(block_digit_accum.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_digit_accum, 0);
    let strlen = var(&block_digit_accum, 1);
    let base = var(&block_digit_accum, 2);
    let i = var(&block_digit_accum, 3);
    let sign = var(&block_digit_accum, 4);
    let val = var(&block_digit_accum, 5);
    let oldpos = var(&block_digit_accum, 6);
    let digit = var(&block_digit_accum, 7);
    let val_mul = variable_with_lltype("val_mul", LowLevelType::Signed);
    block_digit_accum
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_mul",
            vec![Hlvalue::Variable(val), Hlvalue::Variable(base.clone())],
            Hlvalue::Variable(val_mul.clone()),
        ));
    let val_next = variable_with_lltype("val_next", LowLevelType::Signed);
    block_digit_accum
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(val_mul), Hlvalue::Variable(digit)],
            Hlvalue::Variable(val_next.clone()),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_digit_accum
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_digit_accum.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(base),
                Hlvalue::Variable(i_next),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val_next),
                Hlvalue::Variable(oldpos),
            ],
            Some(block_digit_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let chars = var(&block_empty_check, 0);
    let strlen = var(&block_empty_check, 1);
    let i = var(&block_empty_check, 2);
    let sign = var(&block_empty_check, 3);
    let val = var(&block_empty_check, 4);
    let oldpos = var(&block_empty_check, 5);
    let no_digits = variable_with_lltype("no_digits", LowLevelType::Bool);
    block_empty_check
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(i.clone()), Hlvalue::Variable(oldpos)],
            Hlvalue::Variable(no_digits.clone()),
        ));
    block_empty_check.borrow_mut().exitswitch = Some(Hlvalue::Variable(no_digits));
    block_empty_check.closeblock(vec![
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(i),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
            ],
            Some(block_trailing_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let _chars = var(&block_trailing_cond, 0);
    let strlen = var(&block_trailing_cond, 1);
    let i = var(&block_trailing_cond, 2);
    let sign = var(&block_trailing_cond, 3);
    let val = var(&block_trailing_cond, 4);
    let has_trailing = variable_with_lltype("has_trailing", LowLevelType::Bool);
    block_trailing_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(strlen.clone()),
            ],
            Hlvalue::Variable(has_trailing.clone()),
        ));
    block_trailing_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(has_trailing));
    block_trailing_cond.closeblock(vec![
        Link::new(
            args(&block_trailing_cond),
            Some(block_trailing_char.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(sign), Hlvalue::Variable(val)],
            Some(block_return_value.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_trailing_char, 0);
    let strlen = var(&block_trailing_char, 1);
    let i = var(&block_trailing_char, 2);
    let sign = var(&block_trailing_char, 3);
    let val = var(&block_trailing_char, 4);
    let ch = variable_with_lltype("ch", char_lltype);
    block_trailing_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Hlvalue::Variable(ch.clone()),
        ));
    let c = variable_with_lltype("c", LowLevelType::Signed);
    block_trailing_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(ch)],
            Hlvalue::Variable(c.clone()),
        ));
    let is_space = variable_with_lltype("is_space", LowLevelType::Bool);
    block_trailing_char
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(c), signed_const(32)],
            Hlvalue::Variable(is_space.clone()),
        ));
    block_trailing_char.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_space));
    block_trailing_char.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(i),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
            ],
            Some(block_trailing_inc.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let chars = var(&block_trailing_inc, 0);
    let strlen = var(&block_trailing_inc, 1);
    let i = var(&block_trailing_inc, 2);
    let sign = var(&block_trailing_inc, 3);
    let val = var(&block_trailing_inc, 4);
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_trailing_inc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_trailing_inc.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(strlen),
                Hlvalue::Variable(i_next),
                Hlvalue::Variable(sign),
                Hlvalue::Variable(val),
            ],
            Some(block_trailing_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let sign = var(&block_return_value, 0);
    let val = var(&block_return_value, 1);
    let result = variable_with_lltype("result", LowLevelType::Signed);
    block_return_value
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_mul",
            vec![Hlvalue::Variable(sign), Hlvalue::Variable(val)],
            Hlvalue::Variable(result.clone()),
        ));
    block_return_value.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string(), "base".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_str_is_true`
/// (`rtyper/rstr.py:944-947`):
///
/// ```python
/// @classmethod
/// def ll_str_is_true(cls, s):
///     # check if a string is True, allowing for None
///     return bool(s) and cls.ll_strlen(s) != 0
/// ```
///
/// 2-block CFG that mirrors the source-level short-circuit `and`:
/// - **start**: `v_nz = ptr_nonzero(s)`. Branches on `v_nz` —
///   True → `block_check_len` (link arg `[s]`); False → returnblock
///   carrying `Bool(false)`.
/// - **block_check_len**: `getsubstruct(s, 'chars')` +
///   `getarraysize(...)` + `int_ne(len, 0)`; link to returnblock with
///   the comparison result.
///
/// `name` is the helper identity (`"ll_str_is_true"` /
/// `"ll_unicode_is_true"`) and `ptr_lltype` selects between
/// `Ptr(STR)` and `Ptr(UNICODE)`. The chars-array `Ptr` lltype is
/// derived from the resolved STR/UNICODE struct via
/// [`chars_array_ptr_lltype_from_strptr`], so the synthesised
/// `getsubstruct` op carries a valid `SomePtr`-shaped annotation
/// (mirrors `build_ll_strlen_helper_graph`).
pub(crate) fn build_ll_str_is_true_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let arg = variable_with_lltype("s", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_zero = || constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed);

    // block_check_len inputarg: same `s` ptr passed forward through the
    // True branch link.
    let s_for_len = variable_with_lltype("s", ptr_lltype);
    let block_check_len = Block::shared(vec![Hlvalue::Variable(s_for_len.clone())]);

    // ---- start: ptr_nonzero(s); branch on the result.
    let v_nz = variable_with_lltype("v_nz", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_nonzero",
        vec![Hlvalue::Variable(arg.clone())],
        Hlvalue::Variable(v_nz.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_nz));
    let start_true_link = Link::new(
        vec![Hlvalue::Variable(arg)],
        Some(block_check_len.clone()),
        Some(bool_true()),
    )
    .into_ref();
    let start_false_link = Link::new(
        vec![bool_false()],
        Some(graph.returnblock.clone()),
        Some(bool_false()),
    )
    .into_ref();
    startblock.closeblock(vec![start_true_link, start_false_link]);

    // ---- block_check_len: getsubstruct('chars') + getarraysize +
    // int_ne(len, 0); link to returnblock with the comparison result.
    let v_len = emit_chars_length_ops(
        &block_check_len,
        Hlvalue::Variable(s_for_len),
        chars_array_ptr_lltype,
    );
    let v_result = variable_with_lltype("result", LowLevelType::Bool);
    block_check_len
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ne",
            vec![v_len, signed_zero()],
            Hlvalue::Variable(v_result.clone()),
        ));
    block_check_len.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_streq`
/// (`rtyper/lltypesystem/rstr.py:604-620`):
///
/// ```python
/// @staticmethod
/// def ll_streq(s1, s2):
///     if s1 == s2:       # also if both are NULLs
///         return True
///     if not s1 or not s2:
///         return False
///     len1 = len(s1.chars)
///     len2 = len(s2.chars)
///     if len1 != len2:
///         return False
///     j = 0
///     chars1 = s1.chars
///     chars2 = s2.chars
///     while j < len1:
///         if chars1[j] != chars2[j]:
///             return False
///         j += 1
///     return True
/// ```
///
/// 6-block CFG plus the returnblock:
/// - **start**: `eq = ptr_eq(s1, s2)`. True → returnblock(`Bool(true)`),
///   False → `block_null_check_s1`.
/// - **block_null_check_s1**: `nz1 = ptr_nonzero(s1)`. False →
///   returnblock(`Bool(false)`), True → `block_null_check_s2`.
/// - **block_null_check_s2**: `nz2 = ptr_nonzero(s2)`. False →
///   returnblock(`Bool(false)`), True → `block_compare_lens`.
/// - **block_compare_lens**: `chars1 = getsubstruct(s1, 'chars')`,
///   `len1 = getarraysize(chars1)`, `chars2 = getsubstruct(s2,
///   'chars')`, `len2 = getarraysize(chars2)`, `lens_eq = int_eq(len1,
///   len2)`. True → `block_loop_cond` with `j = 0`, False →
///   returnblock(`Bool(false)`).
/// - **block_loop_cond**: `lt = int_lt(j, len1)`. True →
///   `block_loop_body`, False → returnblock(`Bool(true)`)
///   (loop exhausted with all chars matching).
/// - **block_loop_body**: `c1 = getarrayitem(chars1, j)`, `c2 =
///   getarrayitem(chars2, j)`, `chars_eq = char_eq/unichar_eq(c1,
///   c2)`, `j_next = int_add(j, 1)`. True (chars match) →
///   `block_loop_cond` with `j_next`, False → returnblock(`Bool(false)`).
///
/// Polymorphic over `Ptr(STR)` / `Ptr(UNICODE)` via the
/// chars-array Ptr lltype derivation reused from
/// [`chars_array_ptr_lltype_from_strptr`]. The element comparison op
/// (`char_eq` / `unichar_eq`) is selected from the chars-array
/// element type.
pub(crate) fn build_ll_streq_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    // Element comparison op depends on the chars Array element type.
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_streq helper expects Ptr(Array(Char|UniChar)), got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_streq helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let char_eq_op = match elem_lltype {
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_streq helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Inputargs.
    let s1 = variable_with_lltype("s1", ptr_lltype.clone());
    let s2 = variable_with_lltype("s2", ptr_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s1.clone()),
        Hlvalue::Variable(s2.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Pre-create downstream blocks so that `closeblock` calls below
    // can supply already-existing targets. Each block carries its own
    // local copy of the inputargs that flow forward.
    let s1_for_null = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_null = variable_with_lltype("s2", ptr_lltype.clone());
    let block_null_check_s1 = Block::shared(vec![
        Hlvalue::Variable(s1_for_null.clone()),
        Hlvalue::Variable(s2_for_null.clone()),
    ]);

    let s1_for_null2 = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_null2 = variable_with_lltype("s2", ptr_lltype.clone());
    let block_null_check_s2 = Block::shared(vec![
        Hlvalue::Variable(s1_for_null2.clone()),
        Hlvalue::Variable(s2_for_null2.clone()),
    ]);

    let s1_for_lens = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_lens = variable_with_lltype("s2", ptr_lltype.clone());
    let block_compare_lens = Block::shared(vec![
        Hlvalue::Variable(s1_for_lens.clone()),
        Hlvalue::Variable(s2_for_lens.clone()),
    ]);

    let chars1_for_cond = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_cond = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len1_for_cond = variable_with_lltype("len1", LowLevelType::Signed);
    let j_for_cond = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars1_for_cond.clone()),
        Hlvalue::Variable(chars2_for_cond.clone()),
        Hlvalue::Variable(len1_for_cond.clone()),
        Hlvalue::Variable(j_for_cond.clone()),
    ]);

    let chars1_for_body = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_body = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len1_for_body = variable_with_lltype("len1", LowLevelType::Signed);
    let j_for_body = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars1_for_body.clone()),
        Hlvalue::Variable(chars2_for_body.clone()),
        Hlvalue::Variable(len1_for_body.clone()),
        Hlvalue::Variable(j_for_body.clone()),
    ]);

    // ---- start: ptr_eq(s1, s2); branch on result.
    let v_eq = variable_with_lltype("eq", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_eq",
        vec![Hlvalue::Variable(s1.clone()), Hlvalue::Variable(s2.clone())],
        Hlvalue::Variable(v_eq.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_eq));
    startblock.closeblock(vec![
        Link::new(
            vec![bool_true()],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(s1), Hlvalue::Variable(s2)],
            Some(block_null_check_s1.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_null_check_s1: ptr_nonzero(s1).
    let nz1 = variable_with_lltype("nz1", LowLevelType::Bool);
    block_null_check_s1
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "ptr_nonzero",
            vec![Hlvalue::Variable(s1_for_null.clone())],
            Hlvalue::Variable(nz1.clone()),
        ));
    block_null_check_s1.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz1));
    block_null_check_s1.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s1_for_null),
                Hlvalue::Variable(s2_for_null),
            ],
            Some(block_null_check_s2.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_null_check_s2: ptr_nonzero(s2).
    let nz2 = variable_with_lltype("nz2", LowLevelType::Bool);
    block_null_check_s2
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "ptr_nonzero",
            vec![Hlvalue::Variable(s2_for_null2.clone())],
            Hlvalue::Variable(nz2.clone()),
        ));
    block_null_check_s2.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz2));
    block_null_check_s2.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s1_for_null2),
                Hlvalue::Variable(s2_for_null2),
            ],
            Some(block_compare_lens.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_compare_lens: extract chars + len for both sides,
    // then int_eq(len1, len2).
    let chars1 = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    block_compare_lens
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(s1_for_lens.clone()), chars_field_const()],
            Hlvalue::Variable(chars1.clone()),
        ));
    let len1 = variable_with_lltype("len1", LowLevelType::Signed);
    block_compare_lens
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(chars1.clone())],
            Hlvalue::Variable(len1.clone()),
        ));
    let chars2 = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    block_compare_lens
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(s2_for_lens.clone()), chars_field_const()],
            Hlvalue::Variable(chars2.clone()),
        ));
    let len2 = variable_with_lltype("len2", LowLevelType::Signed);
    block_compare_lens
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(chars2.clone())],
            Hlvalue::Variable(len2.clone()),
        ));
    let lens_eq = variable_with_lltype("lens_eq", LowLevelType::Bool);
    block_compare_lens
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_eq",
            vec![Hlvalue::Variable(len1.clone()), Hlvalue::Variable(len2)],
            Hlvalue::Variable(lens_eq.clone()),
        ));
    block_compare_lens.borrow_mut().exitswitch = Some(Hlvalue::Variable(lens_eq));
    block_compare_lens.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1),
                Hlvalue::Variable(chars2),
                Hlvalue::Variable(len1),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_loop_cond: int_lt(j, len1); branch.
    let lt = variable_with_lltype("lt", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(j_for_cond.clone()),
                Hlvalue::Variable(len1_for_cond.clone()),
            ],
            Hlvalue::Variable(lt.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_cond),
                Hlvalue::Variable(chars2_for_cond),
                Hlvalue::Variable(len1_for_cond),
                Hlvalue::Variable(j_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_true()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_loop_body: getarrayitem on both, char_eq, int_add(j, 1).
    let c1 = variable_with_lltype("c1", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars1_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
            ],
            Hlvalue::Variable(c1.clone()),
        ));
    let c2 = variable_with_lltype("c2", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars2_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
            ],
            Hlvalue::Variable(c2.clone()),
        ));
    let chars_eq = variable_with_lltype("chars_eq", LowLevelType::Bool);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            char_eq_op,
            vec![Hlvalue::Variable(c1), Hlvalue::Variable(c2)],
            Hlvalue::Variable(chars_eq.clone()),
        ));
    let j_next = variable_with_lltype("j_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(j_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(j_next.clone()),
        ));
    block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(chars_eq));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_body),
                Hlvalue::Variable(chars2_for_body),
                Hlvalue::Variable(len1_for_body),
                Hlvalue::Variable(j_next),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s1".to_string(), "s2".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_strcmp`
/// (`rtyper/lltypesystem/rstr.py:579-599`):
///
/// ```python
/// @staticmethod
/// def ll_strcmp(s1, s2):
///     if not s1 and not s2:
///         return True
///     if not s1 or not s2:
///         return False
///     chars1 = s1.chars
///     chars2 = s2.chars
///     len1 = len(chars1)
///     len2 = len(chars2)
///
///     if len1 < len2:
///         cmplen = len1
///     else:
///         cmplen = len2
///     i = 0
///     while i < cmplen:
///         diff = ord(chars1[i]) - ord(chars2[i])
///         if diff != 0:
///             return diff
///         i += 1
///     return len1 - len2
/// ```
///
/// Return type is `Signed`: RPython coerces the `True`/`False` literals
/// in the NULL-pair branches to `Signed(1)` / `Signed(0)` so the four
/// returns (1 / 0 / diff / len1-len2) all land in a single `Signed`
/// inputarg on the returnblock.
///
/// 7-block CFG plus the returnblock:
/// - **start**: `nz1 = ptr_nonzero(s1)`. True → `block_s1_nonnull`,
///   False → `block_s1_null`.
/// - **block_s1_null**: `nz2 = ptr_nonzero(s2)`. True (s2 non-NULL,
///   s1 NULL) → returnblock(`Signed(0)`); False (both NULL) →
///   returnblock(`Signed(1)`).
/// - **block_s1_nonnull**: `nz2 = ptr_nonzero(s2)`. True (both
///   non-NULL) → `block_both_nonnull`; False (s1 non-NULL, s2 NULL) →
///   returnblock(`Signed(0)`).
/// - **block_both_nonnull**: `chars1 = getsubstruct(s1, 'chars')`,
///   `len1 = getarraysize(chars1)`, `chars2 = getsubstruct(s2,
///   'chars')`, `len2 = getarraysize(chars2)`, `lens_lt = int_lt(len1,
///   len2)`. True → `block_loop_cond` with `cmplen = len1`, False →
///   `block_loop_cond` with `cmplen = len2`. Both branches initialise
///   `i = 0`.
/// - **block_loop_cond**: `lt = int_lt(i, cmplen)`. True →
///   `block_loop_body`, False → `block_return_len_diff`.
/// - **block_loop_body**: `c1 = getarrayitem(chars1, i)`, `c2 =
///   getarrayitem(chars2, i)`, `c1_int = cast_*_to_int(c1)`, `c2_int
///   = cast_*_to_int(c2)`, `diff = int_sub(c1_int, c2_int)`,
///   `has_diff = int_ne(diff, 0)`, `i_next = int_add(i, 1)`. True
///   (chars differ) → returnblock(`diff`), False (chars match) →
///   `block_loop_cond` with `i_next`.
/// - **block_return_len_diff**: `len_diff = int_sub(len1, len2)`;
///   unconditional link to returnblock.
///
/// Polymorphic over `Ptr(STR)` / `Ptr(UNICODE)` via the chars-array
/// Ptr lltype derivation reused from
/// [`chars_array_ptr_lltype_from_strptr`]. The cast op
/// (`cast_char_to_int` / `cast_unichar_to_int`) is selected from the
/// chars-array element type — mirrors `AbstractStringRepr` /
/// `AbstractUnicodeRepr` `ord(chars[i])` lowering.
pub(crate) fn build_ll_strcmp_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_strcmp helper expects Ptr(Array(Char|UniChar)), got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_strcmp helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let cast_op = match elem_lltype {
        LowLevelType::Char => "cast_char_to_int",
        LowLevelType::UniChar => "cast_unichar_to_int",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_strcmp helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Inputargs.
    let s1 = variable_with_lltype("s1", ptr_lltype.clone());
    let s2 = variable_with_lltype("s2", ptr_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s1.clone()),
        Hlvalue::Variable(s2.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // ---- Pre-create downstream blocks.
    // block_s1_null: s1 IS NULL on entry; only s2 is needed.
    let s2_for_null_pair = variable_with_lltype("s2", ptr_lltype.clone());
    let block_s1_null = Block::shared(vec![Hlvalue::Variable(s2_for_null_pair.clone())]);

    // block_s1_nonnull: s1 IS non-NULL; both pointers carry forward.
    let s1_for_s2check = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_s2check = variable_with_lltype("s2", ptr_lltype.clone());
    let block_s1_nonnull = Block::shared(vec![
        Hlvalue::Variable(s1_for_s2check.clone()),
        Hlvalue::Variable(s2_for_s2check.clone()),
    ]);

    // block_both_nonnull: both pointers are live and non-NULL.
    let s1_for_lens = variable_with_lltype("s1", ptr_lltype.clone());
    let s2_for_lens = variable_with_lltype("s2", ptr_lltype.clone());
    let block_both_nonnull = Block::shared(vec![
        Hlvalue::Variable(s1_for_lens.clone()),
        Hlvalue::Variable(s2_for_lens.clone()),
    ]);

    // block_loop_cond: carries (chars1, chars2, len1, len2, cmplen, i).
    // len1/len2 propagate through the loop to feed
    // `block_return_len_diff` once the loop exits without finding a
    // mismatch.
    let chars1_for_cond = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_cond = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len1_for_cond = variable_with_lltype("len1", LowLevelType::Signed);
    let len2_for_cond = variable_with_lltype("len2", LowLevelType::Signed);
    let cmplen_for_cond = variable_with_lltype("cmplen", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars1_for_cond.clone()),
        Hlvalue::Variable(chars2_for_cond.clone()),
        Hlvalue::Variable(len1_for_cond.clone()),
        Hlvalue::Variable(len2_for_cond.clone()),
        Hlvalue::Variable(cmplen_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
    ]);

    // block_loop_body: same six-tuple as loop_cond.
    let chars1_for_body = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_body = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len1_for_body = variable_with_lltype("len1", LowLevelType::Signed);
    let len2_for_body = variable_with_lltype("len2", LowLevelType::Signed);
    let cmplen_for_body = variable_with_lltype("cmplen", LowLevelType::Signed);
    let i_for_body = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars1_for_body.clone()),
        Hlvalue::Variable(chars2_for_body.clone()),
        Hlvalue::Variable(len1_for_body.clone()),
        Hlvalue::Variable(len2_for_body.clone()),
        Hlvalue::Variable(cmplen_for_body.clone()),
        Hlvalue::Variable(i_for_body.clone()),
    ]);

    // block_return_len_diff: takes (len1, len2), computes int_sub.
    let len1_for_lendiff = variable_with_lltype("len1", LowLevelType::Signed);
    let len2_for_lendiff = variable_with_lltype("len2", LowLevelType::Signed);
    let block_return_len_diff = Block::shared(vec![
        Hlvalue::Variable(len1_for_lendiff.clone()),
        Hlvalue::Variable(len2_for_lendiff.clone()),
    ]);

    // ---- start: ptr_nonzero(s1); branch.
    let nz1 = variable_with_lltype("nz1", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_nonzero",
        vec![Hlvalue::Variable(s1.clone())],
        Hlvalue::Variable(nz1.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz1));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s1.clone()), Hlvalue::Variable(s2.clone())],
            Some(block_s1_nonnull.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(s2)],
            Some(block_s1_null.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_s1_null: s1 known NULL; ptr_nonzero(s2) decides
    // both-NULL (False → return 1) vs only-s1-NULL (True → return 0).
    let nz2_in_null = variable_with_lltype("nz2", LowLevelType::Bool);
    block_s1_null
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "ptr_nonzero",
            vec![Hlvalue::Variable(s2_for_null_pair)],
            Hlvalue::Variable(nz2_in_null.clone()),
        ));
    block_s1_null.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz2_in_null));
    block_s1_null.closeblock(vec![
        Link::new(
            vec![signed_const(0)],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![signed_const(1)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_s1_nonnull: ptr_nonzero(s2). True (both non-NULL) →
    // continue; False (only s2 NULL) → return 0.
    let nz2_in_nn = variable_with_lltype("nz2", LowLevelType::Bool);
    block_s1_nonnull
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "ptr_nonzero",
            vec![Hlvalue::Variable(s2_for_s2check.clone())],
            Hlvalue::Variable(nz2_in_nn.clone()),
        ));
    block_s1_nonnull.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz2_in_nn));
    block_s1_nonnull.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s1_for_s2check),
                Hlvalue::Variable(s2_for_s2check),
            ],
            Some(block_both_nonnull.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![signed_const(0)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_both_nonnull: extract chars/len for both, branch on
    // int_lt(len1, len2) to compute cmplen = min(len1, len2).
    let chars1 = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    block_both_nonnull
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(s1_for_lens.clone()), chars_field_const()],
            Hlvalue::Variable(chars1.clone()),
        ));
    let len1 = variable_with_lltype("len1", LowLevelType::Signed);
    block_both_nonnull
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(chars1.clone())],
            Hlvalue::Variable(len1.clone()),
        ));
    let chars2 = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    block_both_nonnull
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(s2_for_lens), chars_field_const()],
            Hlvalue::Variable(chars2.clone()),
        ));
    let len2 = variable_with_lltype("len2", LowLevelType::Signed);
    block_both_nonnull
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(chars2.clone())],
            Hlvalue::Variable(len2.clone()),
        ));
    let lens_lt = variable_with_lltype("lens_lt", LowLevelType::Bool);
    block_both_nonnull
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(len1.clone()),
                Hlvalue::Variable(len2.clone()),
            ],
            Hlvalue::Variable(lens_lt.clone()),
        ));
    block_both_nonnull.borrow_mut().exitswitch = Some(Hlvalue::Variable(lens_lt));
    block_both_nonnull.closeblock(vec![
        // True (len1 < len2): cmplen = len1.
        Link::new(
            vec![
                Hlvalue::Variable(chars1.clone()),
                Hlvalue::Variable(chars2.clone()),
                Hlvalue::Variable(len1.clone()),
                Hlvalue::Variable(len2.clone()),
                Hlvalue::Variable(len1.clone()),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False (len1 >= len2): cmplen = len2.
        Link::new(
            vec![
                Hlvalue::Variable(chars1),
                Hlvalue::Variable(chars2),
                Hlvalue::Variable(len1),
                Hlvalue::Variable(len2.clone()),
                Hlvalue::Variable(len2),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_loop_cond: int_lt(i, cmplen); branch.
    let lt = variable_with_lltype("lt", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(cmplen_for_cond.clone()),
            ],
            Hlvalue::Variable(lt.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_cond),
                Hlvalue::Variable(chars2_for_cond),
                Hlvalue::Variable(len1_for_cond.clone()),
                Hlvalue::Variable(len2_for_cond.clone()),
                Hlvalue::Variable(cmplen_for_cond),
                Hlvalue::Variable(i_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(len1_for_cond),
                Hlvalue::Variable(len2_for_cond),
            ],
            Some(block_return_len_diff.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_loop_body: getarrayitem×2, cast×2, int_sub diff,
    // int_ne, int_add i_next; branch on has_diff.
    let c1 = variable_with_lltype("c1", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars1_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
            ],
            Hlvalue::Variable(c1.clone()),
        ));
    let c2 = variable_with_lltype("c2", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars2_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
            ],
            Hlvalue::Variable(c2.clone()),
        ));
    let c1_int = variable_with_lltype("c1_int", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(c1)],
            Hlvalue::Variable(c1_int.clone()),
        ));
    let c2_int = variable_with_lltype("c2_int", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(c2)],
            Hlvalue::Variable(c2_int.clone()),
        ));
    let diff = variable_with_lltype("diff", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Variable(c1_int), Hlvalue::Variable(c2_int)],
            Hlvalue::Variable(diff.clone()),
        ));
    let has_diff = variable_with_lltype("has_diff", LowLevelType::Bool);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ne",
            vec![Hlvalue::Variable(diff.clone()), signed_const(0)],
            Hlvalue::Variable(has_diff.clone()),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(has_diff));
    block_loop_body.closeblock(vec![
        // True (diff != 0): return diff.
        Link::new(
            vec![Hlvalue::Variable(diff)],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False (chars match): cycle to loop_cond with i_next.
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_body),
                Hlvalue::Variable(chars2_for_body),
                Hlvalue::Variable(len1_for_body),
                Hlvalue::Variable(len2_for_body),
                Hlvalue::Variable(cmplen_for_body),
                Hlvalue::Variable(i_next),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_return_len_diff: int_sub(len1, len2); link to
    // returnblock.
    let len_diff = variable_with_lltype("len_diff", LowLevelType::Signed);
    block_return_len_diff
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_sub",
            vec![
                Hlvalue::Variable(len1_for_lendiff),
                Hlvalue::Variable(len2_for_lendiff),
            ],
            Hlvalue::Variable(len_diff.clone()),
        ));
    block_return_len_diff.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(len_diff)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s1".to_string(), "s2".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_startswith`
/// (`rtyper/lltypesystem/rstr.py:622-637`):
///
/// ```python
/// @staticmethod
/// @jit.elidable
/// def ll_startswith(s1, s2):
///     len1 = len(s1.chars)
///     len2 = len(s2.chars)
///     if len1 < len2:
///         return False
///     j = 0
///     chars1 = s1.chars
///     chars2 = s2.chars
///     while j < len2:
///         if chars1[j] != chars2[j]:
///             return False
///         j += 1
///
///     return True
/// ```
///
/// 3-block CFG plus the returnblock:
/// - **start**: `chars1 = getsubstruct(s1, 'chars')`, `len1 =
///   getarraysize(chars1)`, `chars2 = getsubstruct(s2, 'chars')`,
///   `len2 = getarraysize(chars2)`, `lens_lt = int_lt(len1, len2)`.
///   True → returnblock(`Bool(false)`) (s1 too short to start with
///   s2); False → `block_loop_cond` with `j = 0`.
/// - **block_loop_cond**: `lt = int_lt(j, len2)`. True →
///   `block_loop_body`; False → returnblock(`Bool(true)`) (loop
///   exhausted, all common chars matched).
/// - **block_loop_body**: `c1 = getarrayitem(chars1, j)`, `c2 =
///   getarrayitem(chars2, j)`, `chars_eq = char_eq/unichar_eq(c1,
///   c2)`, `j_next = int_add(j, 1)`. True (chars match) →
///   `block_loop_cond` with `j_next`; False → returnblock(`Bool(false)`).
///
/// Polymorphic over `Ptr(STR)` / `Ptr(UNICODE)` via the chars-array
/// Ptr lltype derivation reused from
/// [`chars_array_ptr_lltype_from_strptr`]. The element comparison op
/// (`char_eq` / `unichar_eq`) is selected from the chars-array
/// element type (mirrors [`build_ll_streq_helper_graph`]).
pub(crate) fn build_ll_startswith_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_startswith helper expects Ptr(Array(Char|UniChar)), got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_startswith helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let char_eq_op = match elem_lltype {
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_startswith helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Inputargs.
    let s1 = variable_with_lltype("s1", ptr_lltype.clone());
    let s2 = variable_with_lltype("s2", ptr_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s1.clone()),
        Hlvalue::Variable(s2.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Pre-create downstream blocks. loop_cond carries forward (chars1,
    // chars2, len2, j); loop_body shares the same shape so the True
    // branch back-edge from loop_body cycles to loop_cond.
    let chars1_for_cond = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_cond = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len2_for_cond = variable_with_lltype("len2", LowLevelType::Signed);
    let j_for_cond = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars1_for_cond.clone()),
        Hlvalue::Variable(chars2_for_cond.clone()),
        Hlvalue::Variable(len2_for_cond.clone()),
        Hlvalue::Variable(j_for_cond.clone()),
    ]);

    let chars1_for_body = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_body = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len2_for_body = variable_with_lltype("len2", LowLevelType::Signed);
    let j_for_body = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars1_for_body.clone()),
        Hlvalue::Variable(chars2_for_body.clone()),
        Hlvalue::Variable(len2_for_body.clone()),
        Hlvalue::Variable(j_for_body.clone()),
    ]);

    // ---- start: extract chars/len for both, branch on int_lt(len1, len2).
    let chars1 = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s1.clone()), chars_field_const()],
        Hlvalue::Variable(chars1.clone()),
    ));
    let len1 = variable_with_lltype("len1", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars1.clone())],
        Hlvalue::Variable(len1.clone()),
    ));
    let chars2 = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s2.clone()), chars_field_const()],
        Hlvalue::Variable(chars2.clone()),
    ));
    let len2 = variable_with_lltype("len2", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars2.clone())],
        Hlvalue::Variable(len2.clone()),
    ));
    let lens_lt = variable_with_lltype("lens_lt", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(len1), Hlvalue::Variable(len2.clone())],
        Hlvalue::Variable(lens_lt.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(lens_lt));
    startblock.closeblock(vec![
        // True (len1 < len2): s1 too short.
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False (len1 >= len2): enter loop with j = 0.
        Link::new(
            vec![
                Hlvalue::Variable(chars1),
                Hlvalue::Variable(chars2),
                Hlvalue::Variable(len2),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_loop_cond: int_lt(j, len2); branch.
    let lt = variable_with_lltype("lt", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(j_for_cond.clone()),
                Hlvalue::Variable(len2_for_cond.clone()),
            ],
            Hlvalue::Variable(lt.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_cond),
                Hlvalue::Variable(chars2_for_cond),
                Hlvalue::Variable(len2_for_cond),
                Hlvalue::Variable(j_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_true()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_loop_body: getarrayitem×2, char_eq/unichar_eq,
    // int_add(j, 1); branch on chars_eq.
    let c1 = variable_with_lltype("c1", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars1_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
            ],
            Hlvalue::Variable(c1.clone()),
        ));
    let c2 = variable_with_lltype("c2", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars2_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
            ],
            Hlvalue::Variable(c2.clone()),
        ));
    let chars_eq = variable_with_lltype("chars_eq", LowLevelType::Bool);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            char_eq_op,
            vec![Hlvalue::Variable(c1), Hlvalue::Variable(c2)],
            Hlvalue::Variable(chars_eq.clone()),
        ));
    let j_next = variable_with_lltype("j_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(j_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(j_next.clone()),
        ));
    block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(chars_eq));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_body),
                Hlvalue::Variable(chars2_for_body),
                Hlvalue::Variable(len2_for_body),
                Hlvalue::Variable(j_next),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s1".to_string(), "s2".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_endswith`
/// (`rtyper/lltypesystem/rstr.py:645-661`):
///
/// ```python
/// @staticmethod
/// @jit.elidable
/// def ll_endswith(s1, s2):
///     len1 = len(s1.chars)
///     len2 = len(s2.chars)
///     if len1 < len2:
///         return False
///     j = 0
///     chars1 = s1.chars
///     chars2 = s2.chars
///     offset = len1 - len2
///     while j < len2:
///         if chars1[offset + j] != chars2[j]:
///             return False
///         j += 1
///
///     return True
/// ```
///
/// 3-block CFG plus the returnblock — mirror of
/// [`build_ll_startswith_helper_graph`] with the addition of a
/// constant `offset = len1 - len2` carried through the loop and an
/// `int_add(offset, j) -> idx` indirection in `block_loop_body`:
/// - **start**: `chars1 = getsubstruct(s1, 'chars')`, `len1 =
///   getarraysize(chars1)`, `chars2 = getsubstruct(s2, 'chars')`,
///   `len2 = getarraysize(chars2)`, `offset = int_sub(len1, len2)`,
///   `lens_lt = int_lt(len1, len2)`. True → returnblock(`Bool(false)`);
///   False → `block_loop_cond` with `j = 0`.
/// - **block_loop_cond**: `lt = int_lt(j, len2)`. True →
///   `block_loop_body`; False → returnblock(`Bool(true)`).
/// - **block_loop_body**: `idx = int_add(offset, j)`, `c1 =
///   getarrayitem(chars1, idx)`, `c2 = getarrayitem(chars2, j)`,
///   `chars_eq = char_eq/unichar_eq(c1, c2)`, `j_next = int_add(j,
///   1)`. True (chars match) → `block_loop_cond` with `j_next`,
///   False → returnblock(`Bool(false)`).
///
/// Polymorphic Ptr(STR)/Ptr(UNICODE); element op selected from chars
/// Array element type (mirrors [`build_ll_streq_helper_graph`] /
/// [`build_ll_startswith_helper_graph`]).
pub(crate) fn build_ll_endswith_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_endswith helper expects Ptr(Array(Char|UniChar)), got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_endswith helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let char_eq_op = match elem_lltype {
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_endswith helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s1 = variable_with_lltype("s1", ptr_lltype.clone());
    let s2 = variable_with_lltype("s2", ptr_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s1.clone()),
        Hlvalue::Variable(s2.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Pre-create downstream blocks. loop_cond / loop_body carry
    // (chars1, chars2, len2, offset, j).
    let chars1_for_cond = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_cond = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len2_for_cond = variable_with_lltype("len2", LowLevelType::Signed);
    let offset_for_cond = variable_with_lltype("offset", LowLevelType::Signed);
    let j_for_cond = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars1_for_cond.clone()),
        Hlvalue::Variable(chars2_for_cond.clone()),
        Hlvalue::Variable(len2_for_cond.clone()),
        Hlvalue::Variable(offset_for_cond.clone()),
        Hlvalue::Variable(j_for_cond.clone()),
    ]);

    let chars1_for_body = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_body = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len2_for_body = variable_with_lltype("len2", LowLevelType::Signed);
    let offset_for_body = variable_with_lltype("offset", LowLevelType::Signed);
    let j_for_body = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars1_for_body.clone()),
        Hlvalue::Variable(chars2_for_body.clone()),
        Hlvalue::Variable(len2_for_body.clone()),
        Hlvalue::Variable(offset_for_body.clone()),
        Hlvalue::Variable(j_for_body.clone()),
    ]);

    // ---- start: chars1/len1 + chars2/len2 + offset (int_sub) +
    // length comparison branch.
    let chars1 = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s1.clone()), chars_field_const()],
        Hlvalue::Variable(chars1.clone()),
    ));
    let len1 = variable_with_lltype("len1", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars1.clone())],
        Hlvalue::Variable(len1.clone()),
    ));
    let chars2 = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s2.clone()), chars_field_const()],
        Hlvalue::Variable(chars2.clone()),
    ));
    let len2 = variable_with_lltype("len2", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars2.clone())],
        Hlvalue::Variable(len2.clone()),
    ));
    let offset = variable_with_lltype("offset", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![
            Hlvalue::Variable(len1.clone()),
            Hlvalue::Variable(len2.clone()),
        ],
        Hlvalue::Variable(offset.clone()),
    ));
    let lens_lt = variable_with_lltype("lens_lt", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(len1), Hlvalue::Variable(len2.clone())],
        Hlvalue::Variable(lens_lt.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(lens_lt));
    startblock.closeblock(vec![
        // True (len1 < len2): s1 too short.
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False: enter loop with offset = len1 - len2 and j = 0.
        Link::new(
            vec![
                Hlvalue::Variable(chars1),
                Hlvalue::Variable(chars2),
                Hlvalue::Variable(len2),
                Hlvalue::Variable(offset),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_loop_cond: int_lt(j, len2); branch.
    let lt = variable_with_lltype("lt", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(j_for_cond.clone()),
                Hlvalue::Variable(len2_for_cond.clone()),
            ],
            Hlvalue::Variable(lt.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_cond),
                Hlvalue::Variable(chars2_for_cond),
                Hlvalue::Variable(len2_for_cond),
                Hlvalue::Variable(offset_for_cond),
                Hlvalue::Variable(j_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_true()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_loop_body: int_add(offset, j) -> idx; getarrayitem
    // (chars1, idx); getarrayitem(chars2, j); char_eq/unichar_eq;
    // int_add(j, 1); branch on chars_eq.
    let idx = variable_with_lltype("idx", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(offset_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
            ],
            Hlvalue::Variable(idx.clone()),
        ));
    let c1 = variable_with_lltype("c1", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars1_for_body.clone()),
                Hlvalue::Variable(idx),
            ],
            Hlvalue::Variable(c1.clone()),
        ));
    let c2 = variable_with_lltype("c2", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars2_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
            ],
            Hlvalue::Variable(c2.clone()),
        ));
    let chars_eq = variable_with_lltype("chars_eq", LowLevelType::Bool);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            char_eq_op,
            vec![Hlvalue::Variable(c1), Hlvalue::Variable(c2)],
            Hlvalue::Variable(chars_eq.clone()),
        ));
    let j_next = variable_with_lltype("j_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(j_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(j_next.clone()),
        ));
    block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(chars_eq));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars1_for_body),
                Hlvalue::Variable(chars2_for_body),
                Hlvalue::Variable(len2_for_body),
                Hlvalue::Variable(offset_for_body),
                Hlvalue::Variable(j_next),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s1".to_string(), "s2".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_startswith_char`
/// (`rtyper/lltypesystem/rstr.py:639-643`):
///
/// ```python
/// @staticmethod
/// def ll_startswith_char(s, ch):
///     if not len(s.chars):
///         return False
///     return s.chars[0] == ch
/// ```
///
/// 2-block CFG plus the returnblock:
/// - **start**: `chars = getsubstruct(s, 'chars')`, `length =
///   getarraysize(chars)`, `is_empty = int_eq(length, 0)`. True →
///   returnblock(`Bool(false)`); False → `block_compare` with
///   `chars`.
/// - **block_compare**: `c0 = getarrayitem(chars, 0)`, `eq =
///   char_eq/unichar_eq(c0, ch)`. Unconditional link to returnblock
///   with `eq`.
///
/// `ch` is the helper's second inputarg with lltype derived from the
/// chars Array element type (Char for STR, UniChar for UNICODE).
pub(crate) fn build_ll_startswith_char_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    build_ll_startsendswith_char_helper_graph(name, ptr_lltype, /* is_endswith */ false)
}

/// Synthesise the helper graph for `LLHelpers.ll_endswith_char`
/// (`rtyper/lltypesystem/rstr.py:663-667`):
///
/// ```python
/// @staticmethod
/// def ll_endswith_char(s, ch):
///     if not len(s.chars):
///         return False
///     return s.chars[len(s.chars) - 1] == ch
/// ```
///
/// Mirror of [`build_ll_startswith_char_helper_graph`] differing only
/// in the index used to read the comparison char: instead of
/// `chars[0]`, the body emits `idx = int_sub(length, 1)` then
/// `chars[idx]`. CFG and length-empty fast path are identical.
pub(crate) fn build_ll_endswith_char_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    build_ll_startsendswith_char_helper_graph(name, ptr_lltype, /* is_endswith */ true)
}

/// Shared body for [`build_ll_startswith_char_helper_graph`] and
/// [`build_ll_endswith_char_helper_graph`]. The two helpers differ
/// only in the index expression for the chars-array read:
/// startswith uses `Constant(0)` directly; endswith inserts an
/// `idx = int_sub(length, 1)` op and reads `chars[idx]`.
fn build_ll_startsendswith_char_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    is_endswith: bool,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_starts/endswith_char helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_starts/endswith_char helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let char_eq_op = match elem_lltype {
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_starts/endswith_char helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let ch = variable_with_lltype("ch", elem_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(ch.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // block_compare carries forward (chars, length, ch); length is
    // only consumed by the endswith variant for `int_sub(length, 1)`.
    let chars_for_compare = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let length_for_compare = variable_with_lltype("length", LowLevelType::Signed);
    let ch_for_compare = variable_with_lltype("ch", elem_lltype.clone());
    let block_compare = Block::shared(vec![
        Hlvalue::Variable(chars_for_compare.clone()),
        Hlvalue::Variable(length_for_compare.clone()),
        Hlvalue::Variable(ch_for_compare.clone()),
    ]);

    // ---- start: getsubstruct + getarraysize + int_eq(length, 0).
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s.clone()), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let is_empty = variable_with_lltype("is_empty", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![Hlvalue::Variable(length.clone()), signed_const(0)],
        Hlvalue::Variable(is_empty.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_empty));
    startblock.closeblock(vec![
        // True (length == 0): empty string can't start/end with ch.
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False: compare chars[idx] against ch.
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(length),
                Hlvalue::Variable(ch),
            ],
            Some(block_compare.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_compare: read chars[idx], emit char_eq/unichar_eq,
    // unconditional link to returnblock with the result.
    let idx_value = if is_endswith {
        let idx = variable_with_lltype("idx", LowLevelType::Signed);
        block_compare
            .borrow_mut()
            .operations
            .push(SpaceOperation::new(
                "int_sub",
                vec![Hlvalue::Variable(length_for_compare), signed_const(1)],
                Hlvalue::Variable(idx.clone()),
            ));
        Hlvalue::Variable(idx)
    } else {
        signed_const(0)
    };
    let c = variable_with_lltype("c", elem_lltype.clone());
    block_compare
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![Hlvalue::Variable(chars_for_compare), idx_value],
            Hlvalue::Variable(c.clone()),
        ));
    let eq = variable_with_lltype("eq", LowLevelType::Bool);
    block_compare
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            char_eq_op,
            vec![Hlvalue::Variable(c), Hlvalue::Variable(ch_for_compare)],
            Hlvalue::Variable(eq.clone()),
        ));
    block_compare.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(eq)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string(), "ch".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `_hash_string`
/// (`rpython/rlib/objectmodel.py:596-618`):
///
/// ```python
/// @specialize.ll()
/// def _hash_string(s):
///     length = len(s)
///     if length == 0:
///         return -1
///     x = ord(s[0]) << 7
///     i = 0
///     while i < length:
///         x = intmask((1000003*x) ^ ord(s[i]))
///         i += 1
///     x ^= length
///     return intmask(x)
/// ```
///
/// `s` is a chars Array (`Ptr(Array(Char|UniChar))`), not a STR/UNICODE
/// pointer — the caller (`ll_hash_string`, objectmodel.py:620-621)
/// supplies `ll_s.chars` via `getsubstruct`. `intmask` is a no-op
/// at lltype level (lltype `Signed` arithmetic already wraps), so the
/// synthesizer drops the call.
///
/// 5-block CFG plus the returnblock:
/// - **start**: `length = getarraysize(chars)`, `is_empty =
///   int_eq(length, 0)`. True → returnblock(`Signed(-1)`); False →
///   `block_init` with `(chars, length)`.
/// - **block_init**: `c0 = getarrayitem(chars, 0)`, `c0_int =
///   cast_*_to_int(c0)`, `x = int_lshift(c0_int, 7)`. Unconditional
///   link to `block_loop_cond` with `(chars, length, x, 0)`.
/// - **block_loop_cond**: `lt = int_lt(i, length)`. True →
///   `block_loop_body`; False → `block_finalize` with `(length, x)`.
/// - **block_loop_body**: `x_mul = int_mul(1000003, x)`, `ci =
///   getarrayitem(chars, i)`, `ci_int = cast_*_to_int(ci)`, `x_new =
///   int_xor(x_mul, ci_int)`, `i_next = int_add(i, 1)`. Unconditional
///   link back to `block_loop_cond` with `(chars, length, x_new,
///   i_next)`.
/// - **block_finalize**: `x_final = int_xor(x, length)`. Link to
///   returnblock with `x_final`.
///
/// Polymorphic over `Ptr(Array(Char))` / `Ptr(Array(UniChar))` — the
/// cast op (`cast_char_to_int` / `cast_unichar_to_int`) and the
/// `ord`-emit element type are selected from the array's `OF`. Use
/// [`build_ll_strlen_helper_graph::chars_array_ptr_lltype_from_strptr`]
/// callers wrap this with the STR/UNICODE pointer.
pub(crate) fn build_hash_string_helper_graph(
    name: &str,
    chars_array_ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "_hash_string helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "_hash_string helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let cast_op = match elem_lltype {
        LowLevelType::Char => "cast_char_to_int",
        LowLevelType::UniChar => "cast_unichar_to_int",
        ref other => {
            return Err(TyperError::message(format!(
                "_hash_string helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);

    let chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(chars.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Pre-create downstream blocks. block_init / loop_cond / loop_body
    // carry forward (chars, length [, x [, i]]); block_finalize takes
    // only (length, x).
    let chars_for_init = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let length_for_init = variable_with_lltype("length", LowLevelType::Signed);
    let block_init = Block::shared(vec![
        Hlvalue::Variable(chars_for_init.clone()),
        Hlvalue::Variable(length_for_init.clone()),
    ]);

    let chars_for_cond = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let length_for_cond = variable_with_lltype("length", LowLevelType::Signed);
    let x_for_cond = variable_with_lltype("x", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars_for_cond.clone()),
        Hlvalue::Variable(length_for_cond.clone()),
        Hlvalue::Variable(x_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
    ]);

    let chars_for_body = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let length_for_body = variable_with_lltype("length", LowLevelType::Signed);
    let x_for_body = variable_with_lltype("x", LowLevelType::Signed);
    let i_for_body = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars_for_body.clone()),
        Hlvalue::Variable(length_for_body.clone()),
        Hlvalue::Variable(x_for_body.clone()),
        Hlvalue::Variable(i_for_body.clone()),
    ]);

    let length_for_finalize = variable_with_lltype("length", LowLevelType::Signed);
    let x_for_finalize = variable_with_lltype("x", LowLevelType::Signed);
    let block_finalize = Block::shared(vec![
        Hlvalue::Variable(length_for_finalize.clone()),
        Hlvalue::Variable(x_for_finalize.clone()),
    ]);

    // ---- start: getarraysize + int_eq(length, 0).
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let is_empty = variable_with_lltype("is_empty", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![Hlvalue::Variable(length.clone()), signed_const(0)],
        Hlvalue::Variable(is_empty.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_empty));
    startblock.closeblock(vec![
        Link::new(
            vec![signed_const(-1)],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(chars), Hlvalue::Variable(length)],
            Some(block_init.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_init: chars[0] cast + lshift(7) -> x.
    let c0 = variable_with_lltype("c0", elem_lltype.clone());
    block_init.borrow_mut().operations.push(SpaceOperation::new(
        "getarrayitem",
        vec![Hlvalue::Variable(chars_for_init.clone()), signed_const(0)],
        Hlvalue::Variable(c0.clone()),
    ));
    let c0_int = variable_with_lltype("c0_int", LowLevelType::Signed);
    block_init.borrow_mut().operations.push(SpaceOperation::new(
        cast_op,
        vec![Hlvalue::Variable(c0)],
        Hlvalue::Variable(c0_int.clone()),
    ));
    let x = variable_with_lltype("x", LowLevelType::Signed);
    block_init.borrow_mut().operations.push(SpaceOperation::new(
        "int_lshift",
        vec![Hlvalue::Variable(c0_int), signed_const(7)],
        Hlvalue::Variable(x.clone()),
    ));
    block_init.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_init),
                Hlvalue::Variable(length_for_init),
                Hlvalue::Variable(x),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ---- block_loop_cond: int_lt(i, length); branch.
    let lt = variable_with_lltype("lt", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(length_for_cond.clone()),
            ],
            Hlvalue::Variable(lt.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_cond),
                Hlvalue::Variable(length_for_cond.clone()),
                Hlvalue::Variable(x_for_cond.clone()),
                Hlvalue::Variable(i_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(length_for_cond),
                Hlvalue::Variable(x_for_cond),
            ],
            Some(block_finalize.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_loop_body: x_mul = int_mul(1000003, x); ci =
    // getarrayitem(chars, i); ci_int = cast(ci); x_new = int_xor(x_mul,
    // ci_int); i_next = int_add(i, 1).
    let x_mul = variable_with_lltype("x_mul", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_mul",
            vec![signed_const(1000003), Hlvalue::Variable(x_for_body.clone())],
            Hlvalue::Variable(x_mul.clone()),
        ));
    let ci = variable_with_lltype("ci", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
            ],
            Hlvalue::Variable(ci.clone()),
        ));
    let ci_int = variable_with_lltype("ci_int", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(ci)],
            Hlvalue::Variable(ci_int.clone()),
        ));
    let x_new = variable_with_lltype("x_new", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_xor",
            vec![Hlvalue::Variable(x_mul), Hlvalue::Variable(ci_int)],
            Hlvalue::Variable(x_new.clone()),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_body),
                Hlvalue::Variable(length_for_body),
                Hlvalue::Variable(x_new),
                Hlvalue::Variable(i_next),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ---- block_finalize: int_xor(x, length); return.
    let x_final = variable_with_lltype("x_final", LowLevelType::Signed);
    block_finalize
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_xor",
            vec![
                Hlvalue::Variable(x_for_finalize),
                Hlvalue::Variable(length_for_finalize),
            ],
            Hlvalue::Variable(x_final.clone()),
        ));
    block_finalize.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(x_final)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["chars".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `ll_hash_string`
/// (`rpython/rlib/objectmodel.py:620-621`):
///
/// ```python
/// def ll_hash_string(ll_s):
///     return _hash_string(ll_s.chars)
/// ```
///
/// Single-block wrapper around [`build_hash_string_helper_graph`]:
/// `chars = getsubstruct(s, 'chars')`, `x = direct_call(_hash_string,
/// chars)`, return `x`.
///
/// `inner_helper_name` (e.g. `_hash_string` for STR pair,
/// `_hash_unicode_string` for UNICODE pair) is the cache key for the
/// inner FNV synthesizer; sub-helper graph is registered against that
/// name during this builder. The funcptr for the `direct_call` op is
/// derived via [`crate::translator::rtyper::rtyper::RPythonTyper::getcallable`]
/// once the inner helper materialises.
#[allow(dead_code)]
pub(crate) fn build_ll_hash_string_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    inner_helper_name: &str,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Materialise (or retrieve cached) `_hash_string` sub-helper.
    let inner_name_owned = inner_helper_name.to_string();
    let chars_for_inner_builder = chars_array_ptr_lltype.clone();
    let inner_helper = rtyper.lowlevel_helper_function_with_builder(
        inner_helper_name.to_string(),
        vec![chars_array_ptr_lltype.clone()],
        LowLevelType::Signed,
        move |_rtyper, _args, _result| {
            build_hash_string_helper_graph(&inner_name_owned, chars_for_inner_builder)
        },
    )?;
    let c_inner_funcptr = sub_helper_funcptr_constant(rtyper, &inner_helper)?;

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s.clone()), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let x = variable_with_lltype("x", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![Hlvalue::Constant(c_inner_funcptr), Hlvalue::Variable(chars)],
        Hlvalue::Variable(x.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(x)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers._ll_strhash`
/// (`rtyper/lltypesystem/rstr.py:402-414`):
///
/// ```python
/// @staticmethod
/// @dont_inline
/// @jit.dont_look_inside
/// def _ll_strhash(s):
///     # unlike CPython, there is no reason to avoid to return -1
///     # but our malloc initializes the memory to zero, so we use zero as the
///     # special non-computed-yet value.  Also, jit.conditional_call_elidable
///     # always checks for zero, for now.
///     x = ll_hash_string(s)
///     if x == 0:
///         x = 29872897
///     s.hash = x
///     return x
/// ```
///
/// 2-block CFG plus the returnblock:
/// - **start**: `chars = getsubstruct(s, 'chars')`, `x = direct_call
///   (_hash_string, chars)`, `is_zero = int_eq(x, 0)`. True →
///   `block_set_hash` with `(s, Signed(29872897))`; False →
///   `block_set_hash` with `(s, x)`.
/// - **block_set_hash**: `setfield(s, 'hash', x_use)`. Unconditional
///   link to returnblock with `x_use`.
///
/// Inlines the `getsubstruct + direct_call(_hash_string)` pair instead
/// of routing through a separate `ll_hash_string` helper to avoid a
/// second cross-helper indirection. Result is identical: `_hash_string`
/// is registered once in the helper cache and shared with anyone else
/// who synthesises the same `(name, args, result)` tuple.
pub(crate) fn build_ll_strhash_internal_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    inner_helper_name: &str,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);
    let hash_field_const =
        || constant_with_lltype(ConstValue::byte_str("hash"), LowLevelType::Void);

    // Materialise (or retrieve cached) inner `_hash_string` sub-helper
    // and derive its funcptr Constant for the `direct_call` op.
    let inner_name_owned = inner_helper_name.to_string();
    let chars_for_inner_builder = chars_array_ptr_lltype.clone();
    let inner_helper = rtyper.lowlevel_helper_function_with_builder(
        inner_helper_name.to_string(),
        vec![chars_array_ptr_lltype.clone()],
        LowLevelType::Signed,
        move |_rtyper, _args, _result| {
            build_hash_string_helper_graph(&inner_name_owned, chars_for_inner_builder)
        },
    )?;
    let c_inner_funcptr = sub_helper_funcptr_constant(rtyper, &inner_helper)?;

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // block_set_hash: takes (s, x_use); writes hash + returns x_use.
    let s_for_set = variable_with_lltype("s", ptr_lltype.clone());
    let x_for_set = variable_with_lltype("x_use", LowLevelType::Signed);
    let block_set_hash = Block::shared(vec![
        Hlvalue::Variable(s_for_set.clone()),
        Hlvalue::Variable(x_for_set.clone()),
    ]);

    // ---- start: chars + direct_call + int_eq(x, 0); branch.
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s.clone()), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let x = variable_with_lltype("x", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![Hlvalue::Constant(c_inner_funcptr), Hlvalue::Variable(chars)],
        Hlvalue::Variable(x.clone()),
    ));
    let is_zero = variable_with_lltype("is_zero", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![Hlvalue::Variable(x.clone()), signed_const(0)],
        Hlvalue::Variable(is_zero.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_zero));
    startblock.closeblock(vec![
        // True (x == 0): substitute zero-fixup constant.
        Link::new(
            vec![Hlvalue::Variable(s.clone()), signed_const(29872897)],
            Some(block_set_hash.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False: forward the computed x.
        Link::new(
            vec![Hlvalue::Variable(s), Hlvalue::Variable(x)],
            Some(block_set_hash.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_set_hash: setfield(s, 'hash', x_use); return x_use.
    let void_var = variable_with_lltype("__set_hash_void", LowLevelType::Void);
    block_set_hash
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(s_for_set),
                hash_field_const(),
                Hlvalue::Variable(x_for_set.clone()),
            ],
            Hlvalue::Variable(void_var),
        ));
    block_set_hash.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(x_for_set)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_strhash`
/// (`rtyper/lltypesystem/rstr.py:394-400`):
///
/// ```python
/// @staticmethod
/// def ll_strhash(s):
///     if s:
///         return jit.conditional_call_elidable(s.hash,
///                                              LLHelpers._ll_strhash, s)
///     else:
///         return 0
/// ```
///
/// 2-block CFG plus the returnblock:
/// - **start**: `nz = ptr_nonzero(s)`. True → `block_lookup`;
///   False → returnblock(`Signed(0)`).
/// - **block_lookup**: `h = getfield(s, 'hash')`, `result =
///   jit_conditional_call_value(h, _ll_strhash_funcptr, s)`. Link to
///   returnblock with `result`. The `jit_conditional_call_value`
///   lloperation returns `h` if `h != 0`, otherwise calls
///   `_ll_strhash(s)` and caches the result back into `s.hash`
///   (rstr.py:396-398 semantics).
///
/// Sub-helper chain: `ll_strhash` -> `_ll_strhash` -> `_hash_string`.
/// All three are registered in the helper-cache and shared across
/// callers via the `(name, args, result)` tuple.
pub(crate) fn build_ll_strhash_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    internal_helper_name: &str,
    inner_helper_name: &str,
) -> Result<PyGraph, TyperError> {
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let hash_field_const =
        || constant_with_lltype(ConstValue::byte_str("hash"), LowLevelType::Void);

    // Materialise (or retrieve cached) `_ll_strhash` sub-helper.
    // `_ll_strhash` itself recursively materialises `_hash_string`.
    let internal_name_owned = internal_helper_name.to_string();
    let inner_name_for_internal = inner_helper_name.to_string();
    let ptr_for_internal = ptr_lltype.clone();
    let internal_helper = rtyper.lowlevel_helper_function_with_builder(
        internal_helper_name.to_string(),
        vec![ptr_lltype.clone()],
        LowLevelType::Signed,
        move |rtyper_inner, _args, _result| {
            build_ll_strhash_internal_helper_graph(
                rtyper_inner,
                &internal_name_owned,
                ptr_for_internal,
                &inner_name_for_internal,
            )
        },
    )?;
    let c_internal_funcptr = sub_helper_funcptr_constant(rtyper, &internal_helper)?;

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // block_lookup: takes (s); reads cached hash + conditional call.
    let s_for_lookup = variable_with_lltype("s", ptr_lltype.clone());
    let block_lookup = Block::shared(vec![Hlvalue::Variable(s_for_lookup.clone())]);

    // ---- start: ptr_nonzero(s); branch.
    let nz = variable_with_lltype("nz", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_nonzero",
        vec![Hlvalue::Variable(s.clone())],
        Hlvalue::Variable(nz.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s)],
            Some(block_lookup.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![signed_const(0)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_lookup: getfield(s, 'hash') + jit_conditional_call_value.
    let h = variable_with_lltype("h", LowLevelType::Signed);
    block_lookup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(s_for_lookup.clone()), hash_field_const()],
            Hlvalue::Variable(h.clone()),
        ));
    let result = variable_with_lltype("result", LowLevelType::Signed);
    block_lookup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "jit_conditional_call_value",
            vec![
                Hlvalue::Variable(h),
                Hlvalue::Constant(c_internal_funcptr),
                Hlvalue::Variable(s_for_lookup),
            ],
            Hlvalue::Variable(result.clone()),
        ));
    block_lookup.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_find_char`
/// (`rtyper/lltypesystem/rstr.py:670-680`):
///
/// ```python
/// @staticmethod
/// @jit.elidable
/// @signature(types.any(), types.any(), types.int(), types.int(), returns=types.int())
/// def ll_find_char(s, ch, start, end):
///     i = start
///     if end > len(s.chars):
///         end = len(s.chars)
///     while i < end:
///         if s.chars[i] == ch:
///             return i
///         i += 1
///     return -1
/// ```
///
/// 3-block CFG plus the returnblock:
/// - **start**: extract `chars`, `length`. Branch on
///   `int_gt(end, length)`: True → `block_loop_cond` with
///   `(chars, ch, length, start)`; False → `block_loop_cond` with
///   `(chars, ch, end, start)`.
/// - **block_loop_cond**: `int_lt(i, end_clamped)`. True →
///   `block_loop_body`; False → returnblock(`Signed(-1)`).
/// - **block_loop_body**: `c = getarrayitem(chars, i)`,
///   `eq = char_eq/unichar_eq(c, ch)`, `i_next = int_add(i, 1)`.
///   Branch on `eq`: True → returnblock(`i`); False →
///   `block_loop_cond` with `(chars, ch, end_clamped, i_next)`.
///
/// Polymorphic Ptr(STR)/Ptr(UNICODE); element op chosen from chars
/// Array element type. Helper is dead code today; dispatch is
/// `rtype_method_find` follow-up.
pub(crate) fn build_ll_find_char_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    build_ll_findlike_char_helper_graph(name, ptr_lltype, FindLikeFlavor::Forward)
}

/// Synthesise the helper graph for `LLHelpers.ll_rfind_char`
/// (`rtyper/lltypesystem/rstr.py:682-693`):
///
/// ```python
/// @staticmethod
/// @jit.elidable
/// @signature(types.any(), types.any(), types.int(), types.int(), returns=types.int())
/// def ll_rfind_char(s, ch, start, end):
///     if end > len(s.chars):
///         end = len(s.chars)
///     i = end
///     while i > start:
///         i -= 1
///         if s.chars[i] == ch:
///             return i
///     return -1
/// ```
///
/// Mirror of [`build_ll_find_char_helper_graph`] with the loop
/// running backwards: `i` initialises to `end_clamped` instead of
/// `start`, the loop test is `int_gt(i, start)`, and the body
/// pre-decrements `i` (`i = int_sub(i, 1)`) before reading
/// `chars[i]`.
pub(crate) fn build_ll_rfind_char_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    build_ll_findlike_char_helper_graph(name, ptr_lltype, FindLikeFlavor::Reverse)
}

/// Discriminator for [`build_ll_findlike_char_helper_graph`]: forward
/// (`ll_find_char`) vs reverse (`ll_rfind_char`).
#[derive(Clone, Copy)]
enum FindLikeFlavor {
    Forward,
    Reverse,
}

/// Shared body for [`build_ll_find_char_helper_graph`] and
/// [`build_ll_rfind_char_helper_graph`]. The two helpers differ in
/// the loop's iteration direction and the condition / index
/// expression; otherwise CFG shape, end-clamp pre-step, and end-of-
/// loop sentinel match.
fn build_ll_findlike_char_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    flavor: FindLikeFlavor,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_find/rfind_char helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_find/rfind_char helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let char_eq_op = match elem_lltype {
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_find/rfind_char helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Inputargs: s, ch, start, end.
    let s = variable_with_lltype("s", ptr_lltype.clone());
    let ch = variable_with_lltype("ch", elem_lltype.clone());
    let start_arg = variable_with_lltype("start", LowLevelType::Signed);
    let end_arg = variable_with_lltype("end", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(ch.clone()),
        Hlvalue::Variable(start_arg.clone()),
        Hlvalue::Variable(end_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Loop carries (chars, ch, end_clamped_or_start_bound, i). For
    // forward the third slot is `end_clamped`; for reverse it is
    // `start_bound` (the i > start_bound test). We name the slot
    // `bound` to capture either flavour.
    let chars_for_cond = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let ch_for_cond = variable_with_lltype("ch", elem_lltype.clone());
    let bound_for_cond = variable_with_lltype("bound", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars_for_cond.clone()),
        Hlvalue::Variable(ch_for_cond.clone()),
        Hlvalue::Variable(bound_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
    ]);

    let chars_for_body = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let ch_for_body = variable_with_lltype("ch", elem_lltype.clone());
    let bound_for_body = variable_with_lltype("bound", LowLevelType::Signed);
    let i_for_body = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars_for_body.clone()),
        Hlvalue::Variable(ch_for_body.clone()),
        Hlvalue::Variable(bound_for_body.clone()),
        Hlvalue::Variable(i_for_body.clone()),
    ]);

    // ---- start: getsubstruct + getarraysize + int_gt(end, length).
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let end_too_big = variable_with_lltype("end_too_big", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_gt",
        vec![
            Hlvalue::Variable(end_arg.clone()),
            Hlvalue::Variable(length.clone()),
        ],
        Hlvalue::Variable(end_too_big.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(end_too_big));
    // Forward: bound = end_clamped, i = start.
    // Reverse: bound = start (loop while i > bound), i = end_clamped.
    let (initial_i_true, initial_i_false, initial_bound_true, initial_bound_false) = match flavor {
        FindLikeFlavor::Forward => (
            // True: end > length, end := length, i := start, bound := length.
            Hlvalue::Variable(start_arg.clone()),
            Hlvalue::Variable(start_arg),
            Hlvalue::Variable(length.clone()),
            Hlvalue::Variable(end_arg.clone()),
        ),
        FindLikeFlavor::Reverse => (
            // True: end > length, i := length (end_clamped), bound := start.
            // False: i := end, bound := start.
            Hlvalue::Variable(length.clone()),
            Hlvalue::Variable(end_arg.clone()),
            Hlvalue::Variable(start_arg.clone()),
            Hlvalue::Variable(start_arg),
        ),
    };
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(ch.clone()),
                initial_bound_true,
                initial_i_true,
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(ch),
                initial_bound_false,
                initial_i_false,
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_loop_cond: forward int_lt(i, bound), reverse int_gt(i, bound).
    let cond = variable_with_lltype("cond", LowLevelType::Bool);
    let cond_op = match flavor {
        FindLikeFlavor::Forward => "int_lt",
        FindLikeFlavor::Reverse => "int_gt",
    };
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            cond_op,
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(bound_for_cond.clone()),
            ],
            Hlvalue::Variable(cond.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_cond),
                Hlvalue::Variable(ch_for_cond),
                Hlvalue::Variable(bound_for_cond),
                Hlvalue::Variable(i_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![signed_const(-1)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_loop_body: shape depends on direction.
    // Forward: c = getarrayitem(chars, i); eq = ...; i_next = int_add(i, 1);
    //   branch on eq: True -> return i; False -> loop_cond(i_next).
    // Reverse: i_next = int_sub(i, 1); c = getarrayitem(chars, i_next);
    //   eq = ...; branch on eq: True -> return i_next; False ->
    //   loop_cond(i_next).
    let (read_idx, idx_var_for_return, idx_for_loop) = match flavor {
        FindLikeFlavor::Forward => {
            // Forward: read at i, return i (current), continue with i_next.
            let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
            let c = variable_with_lltype("c", elem_lltype.clone());
            block_loop_body
                .borrow_mut()
                .operations
                .push(SpaceOperation::new(
                    "getarrayitem",
                    vec![
                        Hlvalue::Variable(chars_for_body.clone()),
                        Hlvalue::Variable(i_for_body.clone()),
                    ],
                    Hlvalue::Variable(c.clone()),
                ));
            let eq = variable_with_lltype("eq", LowLevelType::Bool);
            block_loop_body
                .borrow_mut()
                .operations
                .push(SpaceOperation::new(
                    char_eq_op,
                    vec![Hlvalue::Variable(c), Hlvalue::Variable(ch_for_body.clone())],
                    Hlvalue::Variable(eq.clone()),
                ));
            block_loop_body
                .borrow_mut()
                .operations
                .push(SpaceOperation::new(
                    "int_add",
                    vec![Hlvalue::Variable(i_for_body.clone()), signed_const(1)],
                    Hlvalue::Variable(i_next.clone()),
                ));
            block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(eq));
            (i_for_body.clone(), i_for_body.clone(), i_next)
        }
        FindLikeFlavor::Reverse => {
            // Reverse: pre-decrement, read at i_dec, return i_dec on match.
            let i_dec = variable_with_lltype("i_dec", LowLevelType::Signed);
            block_loop_body
                .borrow_mut()
                .operations
                .push(SpaceOperation::new(
                    "int_sub",
                    vec![Hlvalue::Variable(i_for_body.clone()), signed_const(1)],
                    Hlvalue::Variable(i_dec.clone()),
                ));
            let c = variable_with_lltype("c", elem_lltype.clone());
            block_loop_body
                .borrow_mut()
                .operations
                .push(SpaceOperation::new(
                    "getarrayitem",
                    vec![
                        Hlvalue::Variable(chars_for_body.clone()),
                        Hlvalue::Variable(i_dec.clone()),
                    ],
                    Hlvalue::Variable(c.clone()),
                ));
            let eq = variable_with_lltype("eq", LowLevelType::Bool);
            block_loop_body
                .borrow_mut()
                .operations
                .push(SpaceOperation::new(
                    char_eq_op,
                    vec![Hlvalue::Variable(c), Hlvalue::Variable(ch_for_body.clone())],
                    Hlvalue::Variable(eq.clone()),
                ));
            block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(eq));
            (i_dec.clone(), i_dec.clone(), i_dec)
        }
    };
    let _ = read_idx;
    block_loop_body.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(idx_var_for_return)],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_body),
                Hlvalue::Variable(ch_for_body),
                Hlvalue::Variable(bound_for_body),
                Hlvalue::Variable(idx_for_loop),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "s".to_string(),
            "ch".to_string(),
            "start".to_string(),
            "end".to_string(),
        ],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_count_char`
/// (`rtyper/lltypesystem/rstr.py:695-706`):
///
/// ```python
/// @staticmethod
/// @jit.elidable
/// def ll_count_char(s, ch, start, end):
///     count = 0
///     i = start
///     if end > len(s.chars):
///         end = len(s.chars)
///     while i < end:
///         if s.chars[i] == ch:
///             count += 1
///         i += 1
///     return count
/// ```
///
/// 3-block CFG plus the returnblock. Same shape as
/// [`build_ll_find_char_helper_graph`] with a `count` accumulator
/// carried through the loop:
/// - **start**: getsubstruct + getarraysize + int_gt(end, length).
///   True → `block_loop_cond` with `(chars, ch, length, start, 0)`;
///   False → `block_loop_cond` with `(chars, ch, end, start, 0)`.
/// - **block_loop_cond**: int_lt(i, end_clamped). True →
///   `block_loop_body`; False → returnblock(`count`).
/// - **block_loop_body**: getarrayitem + char_eq + int_add(i, 1)
///   `i_next` + int_add(count, 1) `count_inc`. Branch on `eq`:
///   True → loop_cond with `count_inc`; False → loop_cond with
///   `count`.
pub(crate) fn build_ll_count_char_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_count_char helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_count_char helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    let char_eq_op = match elem_lltype {
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_count_char helper unsupported chars element type {other:?}"
            )));
        }
    };

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let ch = variable_with_lltype("ch", elem_lltype.clone());
    let start_arg = variable_with_lltype("start", LowLevelType::Signed);
    let end_arg = variable_with_lltype("end", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(ch.clone()),
        Hlvalue::Variable(start_arg.clone()),
        Hlvalue::Variable(end_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Loop carries (chars, ch, end_clamped, i, count).
    let chars_for_cond = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let ch_for_cond = variable_with_lltype("ch", elem_lltype.clone());
    let end_for_cond = variable_with_lltype("end_clamped", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let count_for_cond = variable_with_lltype("count", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars_for_cond.clone()),
        Hlvalue::Variable(ch_for_cond.clone()),
        Hlvalue::Variable(end_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
        Hlvalue::Variable(count_for_cond.clone()),
    ]);

    let chars_for_body = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let ch_for_body = variable_with_lltype("ch", elem_lltype.clone());
    let end_for_body = variable_with_lltype("end_clamped", LowLevelType::Signed);
    let i_for_body = variable_with_lltype("i", LowLevelType::Signed);
    let count_for_body = variable_with_lltype("count", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars_for_body.clone()),
        Hlvalue::Variable(ch_for_body.clone()),
        Hlvalue::Variable(end_for_body.clone()),
        Hlvalue::Variable(i_for_body.clone()),
        Hlvalue::Variable(count_for_body.clone()),
    ]);

    // ---- start: getsubstruct + getarraysize + int_gt(end, length).
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let end_too_big = variable_with_lltype("end_too_big", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_gt",
        vec![
            Hlvalue::Variable(end_arg.clone()),
            Hlvalue::Variable(length.clone()),
        ],
        Hlvalue::Variable(end_too_big.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(end_too_big));
    startblock.closeblock(vec![
        // True (end > length): clamp end := length.
        Link::new(
            vec![
                Hlvalue::Variable(chars.clone()),
                Hlvalue::Variable(ch.clone()),
                Hlvalue::Variable(length),
                Hlvalue::Variable(start_arg.clone()),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False: keep end.
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(ch),
                Hlvalue::Variable(end_arg),
                Hlvalue::Variable(start_arg),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_loop_cond: int_lt(i, end_clamped); branch.
    let lt = variable_with_lltype("lt", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(end_for_cond.clone()),
            ],
            Hlvalue::Variable(lt.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_cond),
                Hlvalue::Variable(ch_for_cond),
                Hlvalue::Variable(end_for_cond),
                Hlvalue::Variable(i_for_cond),
                Hlvalue::Variable(count_for_cond.clone()),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(count_for_cond)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_loop_body: getarrayitem + char_eq + int_add(i, 1) +
    // int_add(count, 1); branch on eq.
    let c = variable_with_lltype("c", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    let eq = variable_with_lltype("eq", LowLevelType::Bool);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            char_eq_op,
            vec![Hlvalue::Variable(c), Hlvalue::Variable(ch_for_body.clone())],
            Hlvalue::Variable(eq.clone()),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    let count_inc = variable_with_lltype("count_inc", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(count_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(count_inc.clone()),
        ));
    block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(eq));
    block_loop_body.closeblock(vec![
        // True (match): cycle to loop_cond with count_inc.
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_body.clone()),
                Hlvalue::Variable(ch_for_body.clone()),
                Hlvalue::Variable(end_for_body.clone()),
                Hlvalue::Variable(i_next.clone()),
                Hlvalue::Variable(count_inc),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False (no match): cycle to loop_cond with count unchanged.
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_body),
                Hlvalue::Variable(ch_for_body),
                Hlvalue::Variable(end_for_body),
                Hlvalue::Variable(i_next),
                Hlvalue::Variable(count_for_body),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "s".to_string(),
            "ch".to_string(),
            "start".to_string(),
            "end".to_string(),
        ],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_stritem_nonneg`
/// (`rtyper/lltypesystem/rstr.py:354-360`):
///
/// ```python
/// @staticmethod
/// @signature(types.any(), types.int(), returns=types.any())
/// def ll_stritem_nonneg(s, i):
///     chars = s.chars
///     ll_assert(i >= 0, "negative str getitem index")
///     ll_assert(i < len(chars), "str getitem index out of bound")
///     return chars[i]
/// ```
///
/// Single-block CFG: `chars = getsubstruct(s, 'chars')`, `c =
/// getarrayitem(chars, i)`, link to returnblock with `c`. The
/// `ll_assert` calls are debug-only annotations in upstream and
/// are not lowered to lloperations (`rffi.py` no-op definition for
/// production translation).
///
/// Polymorphic Ptr(STR)/Ptr(UNICODE); return lltype is the chars
/// Array element type (Char for STR, UniChar for UNICODE).
pub(crate) fn build_ll_stritem_nonneg_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_stritem_nonneg helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_stritem_nonneg helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    if !matches!(elem_lltype, LowLevelType::Char | LowLevelType::UniChar) {
        return Err(TyperError::message(format!(
            "ll_stritem_nonneg helper unsupported chars element type {elem_lltype:?}"
        )));
    }

    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let i = variable_with_lltype("i", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(i.clone()),
    ]);
    let return_var = variable_with_lltype("result", elem_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let c = variable_with_lltype("c", elem_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarrayitem",
        vec![Hlvalue::Variable(chars), Hlvalue::Variable(i)],
        Hlvalue::Variable(c.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string(), "i".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `AbstractLLHelpers.ll_stritem`
/// (`rtyper/rstr.py:955-959`):
///
/// ```python
/// @classmethod
/// def ll_stritem(cls, s, i):
///     if i < 0:
///         i += cls.ll_strlen(s)
///     return cls.ll_stritem_nonneg(s, i)
/// ```
///
/// 3-block CFG plus the returnblock:
/// - **start**: `is_neg = int_lt(i, 0)`. True → `block_neg_fix(s,
///   i)`; False → `block_dispatch(s, i)` (direct).
/// - **block_neg_fix**: `chars = getsubstruct(s, 'chars')`,
///   `length = getarraysize(chars)`, `i_fix = int_add(i, length)`.
///   Unconditional link to `block_dispatch(s, i_fix)`.
/// - **block_dispatch**: `c = direct_call(ll_stritem_nonneg, s,
///   i_eff)`. Link to returnblock with `c`.
///
/// `length` is computed inline via getsubstruct/getarraysize rather
/// than a `ll_strlen` sub-helper — inline emit matches the same shape
/// the RPython compiler would produce after inlining the `ll_strlen`
/// 1-line wrapper.
///
/// `inner_helper_name` is the cache key for the `ll_stritem_nonneg`
/// sub-helper (`ll_stritem_nonneg` for STR pair,
/// `ll_unicode_stritem_nonneg` for UNICODE pair).
pub(crate) fn build_ll_stritem_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    inner_helper_name: &str,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_stritem helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_stritem helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    if !matches!(elem_lltype, LowLevelType::Char | LowLevelType::UniChar) {
        return Err(TyperError::message(format!(
            "ll_stritem helper unsupported chars element type {elem_lltype:?}"
        )));
    }

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Materialise (or retrieve cached) ll_stritem_nonneg sub-helper.
    let inner_name_owned = inner_helper_name.to_string();
    let ptr_for_inner = ptr_lltype.clone();
    let inner_helper = rtyper.lowlevel_helper_function_with_builder(
        inner_helper_name.to_string(),
        vec![ptr_lltype.clone(), LowLevelType::Signed],
        elem_lltype.clone(),
        move |_rtyper, _args, _result| {
            build_ll_stritem_nonneg_helper_graph(&inner_name_owned, ptr_for_inner)
        },
    )?;
    let c_inner_funcptr = sub_helper_funcptr_constant(rtyper, &inner_helper)?;

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let i = variable_with_lltype("i", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(i.clone()),
    ]);
    let return_var = variable_with_lltype("result", elem_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // block_neg_fix: takes (s, i); computes i + length.
    let s_for_fix = variable_with_lltype("s", ptr_lltype.clone());
    let i_for_fix = variable_with_lltype("i", LowLevelType::Signed);
    let block_neg_fix = Block::shared(vec![
        Hlvalue::Variable(s_for_fix.clone()),
        Hlvalue::Variable(i_for_fix.clone()),
    ]);

    // block_dispatch: takes (s, i_eff); direct_calls ll_stritem_nonneg.
    let s_for_dispatch = variable_with_lltype("s", ptr_lltype.clone());
    let i_for_dispatch = variable_with_lltype("i_eff", LowLevelType::Signed);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(s_for_dispatch.clone()),
        Hlvalue::Variable(i_for_dispatch.clone()),
    ]);

    // ---- start: int_lt(i, 0); branch.
    let is_neg = variable_with_lltype("is_neg", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(i.clone()), signed_const(0)],
        Hlvalue::Variable(is_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_neg));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s.clone()), Hlvalue::Variable(i.clone())],
            Some(block_neg_fix.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(s), Hlvalue::Variable(i)],
            Some(block_dispatch.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_neg_fix: chars + length + i_fix = i + length.
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    block_neg_fix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(s_for_fix.clone()), chars_field_const()],
            Hlvalue::Variable(chars.clone()),
        ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    block_neg_fix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(chars)],
            Hlvalue::Variable(length.clone()),
        ));
    let i_fix = variable_with_lltype("i_fix", LowLevelType::Signed);
    block_neg_fix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_fix), Hlvalue::Variable(length)],
            Hlvalue::Variable(i_fix.clone()),
        ));
    block_neg_fix.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s_for_fix), Hlvalue::Variable(i_fix)],
            Some(block_dispatch.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ---- block_dispatch: direct_call(ll_stritem_nonneg, s, i_eff).
    let c = variable_with_lltype("c", elem_lltype);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_inner_funcptr),
                Hlvalue::Variable(s_for_dispatch),
                Hlvalue::Variable(i_for_dispatch),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    block_dispatch.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string(), "i".to_string()],
        func,
    ))
}

/// Synthesise a `ll_isdigit/isalpha/isalnum/isspace/isupper/islower`-
/// style `string-isxxx` helper graph that loops over chars and
/// `direct_call`s a per-char predicate sub-helper. RPython
/// `AbstractStringRepr.rtype_method_isdigit` etc. (`rstr.py:253-269`)
/// ultimately lower to `ll_isdigit(s)` / `ll_isalpha(s)` etc., which
/// in turn iterate `s.chars` and call the matching `ll_char_isxxx`
/// per-char predicate.
///
/// 3-block CFG plus the returnblock:
/// - **start**: `chars = getsubstruct(s, 'chars')`, `length =
///   getarraysize(chars)`, `is_empty = int_eq(length, 0)`. True →
///   returnblock(`Bool(false)`); False → `block_loop_cond` with
///   `(chars, length, 0)`.
/// - **block_loop_cond**: `lt = int_lt(i, length)`. True →
///   `block_loop_body`; False → returnblock(`Bool(true)`).
/// - **block_loop_body**: `c = getarrayitem(chars, i)`,
///   `is_d = direct_call(per-char-predicate, c)`,
///   `i_next = int_add(i, 1)`. Branch on `is_d`: True →
///   `block_loop_cond` with `i_next`; False → returnblock(`Bool(false)`).
///
/// `inner_helper_name` / `build_inner_helper` are the cache key and
/// builder for the per-char predicate (e.g. `ll_char_isdigit` /
/// `build_ll_char_isdigit_helper_graph`). The helper expects the
/// sub-helper signature to be `(elem) -> Bool`.
pub(crate) fn build_ll_string_isxxx_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    inner_helper_name: &str,
    build_inner_helper: impl FnOnce(&str) -> Result<PyGraph, TyperError> + 'static,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_string_isxxx helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_string_isxxx helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    if !matches!(elem_lltype, LowLevelType::Char | LowLevelType::UniChar) {
        return Err(TyperError::message(format!(
            "ll_string_isxxx helper unsupported chars element type {elem_lltype:?}"
        )));
    }

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Materialise (or retrieve cached) per-char predicate sub-helper.
    let inner_name_owned = inner_helper_name.to_string();
    let inner_helper = rtyper.lowlevel_helper_function_with_builder(
        inner_helper_name.to_string(),
        vec![elem_lltype.clone()],
        LowLevelType::Bool,
        move |_rtyper, _args, _result| build_inner_helper(&inner_name_owned),
    )?;
    let c_inner_funcptr = sub_helper_funcptr_constant(rtyper, &inner_helper)?;

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let chars_for_cond = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let length_for_cond = variable_with_lltype("length", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars_for_cond.clone()),
        Hlvalue::Variable(length_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
    ]);

    let chars_for_body = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
    let length_for_body = variable_with_lltype("length", LowLevelType::Signed);
    let i_for_body = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars_for_body.clone()),
        Hlvalue::Variable(length_for_body.clone()),
        Hlvalue::Variable(i_for_body.clone()),
    ]);

    // ---- start: getsubstruct + getarraysize + int_eq(length, 0).
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let is_empty = variable_with_lltype("is_empty", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![Hlvalue::Variable(length.clone()), signed_const(0)],
        Hlvalue::Variable(is_empty.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_empty));
    startblock.closeblock(vec![
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(length),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_loop_cond: int_lt(i, length).
    let lt = variable_with_lltype("lt", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(length_for_cond.clone()),
            ],
            Hlvalue::Variable(lt.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_cond),
                Hlvalue::Variable(length_for_cond),
                Hlvalue::Variable(i_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_true()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_loop_body: c + direct_call(predicate, c) + int_add.
    let c = variable_with_lltype("c", elem_lltype);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    let is_d = variable_with_lltype("is_d", LowLevelType::Bool);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Constant(c_inner_funcptr), Hlvalue::Variable(c)],
            Hlvalue::Variable(is_d.clone()),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_body.clone()), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_d));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_body),
                Hlvalue::Variable(length_for_body),
                Hlvalue::Variable(i_next),
            ],
            Some(block_loop_cond.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![bool_false()],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise `LLHelpers.ll_upper` / `LLHelpers.ll_lower`
/// (`lltypesystem/rstr.py:511-535`):
///
/// ```python
/// def ll_upper(s):
///     s_chars = s.chars
///     s_len = len(s_chars)
///     if s_len == 0:
///         return s.empty()
///     i = 0
///     result = mallocstr(s_len)
///     while i < s_len:
///         result.chars[i] = LLHelpers.ll_upper_char(s_chars[i])
///         i += 1
///     return result
/// ```
///
/// The helper is intentionally `Ptr(STR)`-only. Upstream uses
/// `mallocstr` specifically so calling it for unicode explodes; the
/// Rust dispatcher mirrors that by never routing UnicodeRepr here, and
/// the builder rejects non-Char arrays.
///
/// The empty branch returns the incoming empty string object. Upstream
/// returns `s.empty()` (the canonical empty string), but avoiding a new
/// allocation preserves the important low-level invariant of the fast
/// path until the full string adtmeth singleton is wired.
pub(crate) fn build_ll_string_casefold_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    inner_helper_name: &str,
    build_inner_helper: impl FnOnce(&str) -> Result<PyGraph, TyperError> + 'static,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_string_casefold helper expects Ptr(Array(Char)), got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_string_casefold helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    if arr.OF != LowLevelType::Char {
        return Err(TyperError::message(format!(
            "ll_string_casefold helper only supports Char arrays, got {:?}",
            arr.OF
        )));
    }
    let struct_lltype = struct_lltype_from_strptr(&ptr_lltype)?;

    let inner_name_owned = inner_helper_name.to_string();
    let inner_helper = rtyper.lowlevel_helper_function_with_builder(
        inner_helper_name.to_string(),
        vec![LowLevelType::Char],
        LowLevelType::Char,
        move |_rtyper, _args, _result| build_inner_helper(&inner_name_owned),
    )?;
    let c_inner_funcptr = sub_helper_funcptr_constant(rtyper, &inner_helper)?;

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let s_for_empty = variable_with_lltype("s", ptr_lltype.clone());
    let block_empty_return = Block::shared(vec![Hlvalue::Variable(s_for_empty.clone())]);

    let chars_for_alloc = variable_with_lltype("s_chars", chars_array_ptr_lltype.clone());
    let len_for_alloc = variable_with_lltype("s_len", LowLevelType::Signed);
    let block_alloc = Block::shared(vec![
        Hlvalue::Variable(chars_for_alloc.clone()),
        Hlvalue::Variable(len_for_alloc.clone()),
    ]);

    let chars_for_cond = variable_with_lltype("s_chars", chars_array_ptr_lltype.clone());
    let newstr_for_cond = variable_with_lltype("result", ptr_lltype.clone());
    let newchars_for_cond = variable_with_lltype("newchars", chars_array_ptr_lltype.clone());
    let len_for_cond = variable_with_lltype("s_len", LowLevelType::Signed);
    let i_for_cond = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(chars_for_cond.clone()),
        Hlvalue::Variable(newstr_for_cond.clone()),
        Hlvalue::Variable(newchars_for_cond.clone()),
        Hlvalue::Variable(len_for_cond.clone()),
        Hlvalue::Variable(i_for_cond.clone()),
    ]);

    let chars_for_body = variable_with_lltype("s_chars", chars_array_ptr_lltype.clone());
    let newstr_for_body = variable_with_lltype("result", ptr_lltype.clone());
    let newchars_for_body = variable_with_lltype("newchars", chars_array_ptr_lltype.clone());
    let len_for_body = variable_with_lltype("s_len", LowLevelType::Signed);
    let i_for_body = variable_with_lltype("i", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(chars_for_body.clone()),
        Hlvalue::Variable(newstr_for_body.clone()),
        Hlvalue::Variable(newchars_for_body.clone()),
        Hlvalue::Variable(len_for_body.clone()),
        Hlvalue::Variable(i_for_body.clone()),
    ]);

    // ---- start: s_chars, s_len, empty fast-path branch.
    let chars = variable_with_lltype("s_chars", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s.clone()), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let s_len = variable_with_lltype("s_len", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars.clone())],
        Hlvalue::Variable(s_len.clone()),
    ));
    let is_empty = variable_with_lltype("is_empty", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![Hlvalue::Variable(s_len.clone()), signed_const(0)],
        Hlvalue::Variable(is_empty.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_empty));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s.clone())],
            Some(block_empty_return.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(chars), Hlvalue::Variable(s_len)],
            Some(block_alloc.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- empty fast path: no allocation, matching `s.empty()`'s
    // allocation-free contract for already-empty input.
    block_empty_return.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(s_for_empty)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ---- non-empty allocation: result = mallocstr(s_len).
    let newstr = variable_with_lltype("result", ptr_lltype.clone());
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "malloc_varsize",
            vec![
                lowlevel_type_const(struct_lltype),
                gc_flavor_const()?,
                Hlvalue::Variable(len_for_alloc.clone()),
            ],
            Hlvalue::Variable(newstr.clone()),
        ));
    let newchars = variable_with_lltype("newchars", chars_array_ptr_lltype.clone());
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(newstr.clone()), chars_field_const()],
            Hlvalue::Variable(newchars.clone()),
        ));
    block_alloc.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_alloc),
                Hlvalue::Variable(newstr),
                Hlvalue::Variable(newchars),
                Hlvalue::Variable(len_for_alloc),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ---- loop_cond: i < s_len.
    let keep_going = variable_with_lltype("keep_going", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_cond.clone()),
                Hlvalue::Variable(len_for_cond.clone()),
            ],
            Hlvalue::Variable(keep_going.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(keep_going));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_cond),
                Hlvalue::Variable(newstr_for_cond.clone()),
                Hlvalue::Variable(newchars_for_cond),
                Hlvalue::Variable(len_for_cond),
                Hlvalue::Variable(i_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(newstr_for_cond)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- loop_body: result.chars[i] = ll_upper/lower_char(s_chars[i]).
    let c = variable_with_lltype("c", LowLevelType::Char);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    let c_folded = variable_with_lltype("c_folded", LowLevelType::Char);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Constant(c_inner_funcptr), Hlvalue::Variable(c)],
            Hlvalue::Variable(c_folded.clone()),
        ));
    let set_void = variable_with_lltype("set", LowLevelType::Void);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(newchars_for_body.clone()),
                Hlvalue::Variable(i_for_body.clone()),
                Hlvalue::Variable(c_folded),
            ],
            Hlvalue::Variable(set_void),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_body), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(chars_for_body),
                Hlvalue::Variable(newstr_for_body),
                Hlvalue::Variable(newchars_for_body),
                Hlvalue::Variable(len_for_body),
                Hlvalue::Variable(i_next),
            ],
            Some(block_loop_cond),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise `LLHelpers.ll_replace_chr_chr`
/// (`lltypesystem/rstr.py:1032-1045`):
///
/// ```python
/// def ll_replace_chr_chr(s, c1, c2):
///     length = len(s.chars)
///     newstr = s.malloc(length)
///     src = s.chars
///     dst = newstr.chars
///     j = 0
///     while j < length:
///         c = src[j]
///         if c == c1:
///             c = c2
///         dst[j] = c
///         j += 1
///     return newstr
/// ```
pub(crate) fn build_ll_replace_chr_chr_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    elem_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_replace_chr_chr helper expects Ptr(Array(Char|UniChar)), got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_replace_chr_chr helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    if arr.OF != elem_lltype {
        return Err(TyperError::message(format!(
            "ll_replace_chr_chr helper element mismatch: chars array stores {:?}, argument is {elem_lltype:?}",
            arr.OF
        )));
    }
    let eq_op = match elem_lltype {
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        ref other => {
            return Err(TyperError::message(format!(
                "ll_replace_chr_chr helper unsupported char element type {other:?}"
            )));
        }
    };
    let struct_lltype = struct_lltype_from_strptr(&ptr_lltype)?;

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let c1 = variable_with_lltype("c1", elem_lltype.clone());
    let c2 = variable_with_lltype("c2", elem_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(c1.clone()),
        Hlvalue::Variable(c2.clone()),
    ]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let src_for_cond = variable_with_lltype("src", chars_array_ptr_lltype.clone());
    let dst_for_cond = variable_with_lltype("dst", chars_array_ptr_lltype.clone());
    let newstr_for_cond = variable_with_lltype("newstr", ptr_lltype.clone());
    let length_for_cond = variable_with_lltype("length", LowLevelType::Signed);
    let c1_for_cond = variable_with_lltype("c1", elem_lltype.clone());
    let c2_for_cond = variable_with_lltype("c2", elem_lltype.clone());
    let j_for_cond = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(src_for_cond.clone()),
        Hlvalue::Variable(dst_for_cond.clone()),
        Hlvalue::Variable(newstr_for_cond.clone()),
        Hlvalue::Variable(length_for_cond.clone()),
        Hlvalue::Variable(c1_for_cond.clone()),
        Hlvalue::Variable(c2_for_cond.clone()),
        Hlvalue::Variable(j_for_cond.clone()),
    ]);

    let src_for_body = variable_with_lltype("src", chars_array_ptr_lltype.clone());
    let dst_for_body = variable_with_lltype("dst", chars_array_ptr_lltype.clone());
    let newstr_for_body = variable_with_lltype("newstr", ptr_lltype.clone());
    let length_for_body = variable_with_lltype("length", LowLevelType::Signed);
    let c1_for_body = variable_with_lltype("c1", elem_lltype.clone());
    let c2_for_body = variable_with_lltype("c2", elem_lltype.clone());
    let j_for_body = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(src_for_body.clone()),
        Hlvalue::Variable(dst_for_body.clone()),
        Hlvalue::Variable(newstr_for_body.clone()),
        Hlvalue::Variable(length_for_body.clone()),
        Hlvalue::Variable(c1_for_body.clone()),
        Hlvalue::Variable(c2_for_body.clone()),
        Hlvalue::Variable(j_for_body.clone()),
    ]);

    let src_for_store = variable_with_lltype("src", chars_array_ptr_lltype.clone());
    let dst_for_store = variable_with_lltype("dst", chars_array_ptr_lltype.clone());
    let newstr_for_store = variable_with_lltype("newstr", ptr_lltype.clone());
    let length_for_store = variable_with_lltype("length", LowLevelType::Signed);
    let c1_for_store = variable_with_lltype("c1", elem_lltype.clone());
    let c2_for_store = variable_with_lltype("c2", elem_lltype.clone());
    let j_for_store = variable_with_lltype("j", LowLevelType::Signed);
    let c_for_store = variable_with_lltype("c", elem_lltype.clone());
    let block_store = Block::shared(vec![
        Hlvalue::Variable(src_for_store.clone()),
        Hlvalue::Variable(dst_for_store.clone()),
        Hlvalue::Variable(newstr_for_store.clone()),
        Hlvalue::Variable(length_for_store.clone()),
        Hlvalue::Variable(c1_for_store.clone()),
        Hlvalue::Variable(c2_for_store.clone()),
        Hlvalue::Variable(j_for_store.clone()),
        Hlvalue::Variable(c_for_store.clone()),
    ]);

    // ---- start: source chars, length, destination allocation/chars.
    let src = variable_with_lltype("src", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s), chars_field_const()],
        Hlvalue::Variable(src.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(src.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let newstr = variable_with_lltype("newstr", ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc_varsize",
        vec![
            lowlevel_type_const(struct_lltype),
            gc_flavor_const()?,
            Hlvalue::Variable(length.clone()),
        ],
        Hlvalue::Variable(newstr.clone()),
    ));
    let dst = variable_with_lltype("dst", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(newstr.clone()), chars_field_const()],
        Hlvalue::Variable(dst.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src),
                Hlvalue::Variable(dst),
                Hlvalue::Variable(newstr),
                Hlvalue::Variable(length),
                Hlvalue::Variable(c1),
                Hlvalue::Variable(c2),
                signed_const(0),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ---- loop_cond: j < length.
    let keep_going = variable_with_lltype("keep_going", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(j_for_cond.clone()),
                Hlvalue::Variable(length_for_cond.clone()),
            ],
            Hlvalue::Variable(keep_going.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(keep_going));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_for_cond),
                Hlvalue::Variable(dst_for_cond),
                Hlvalue::Variable(newstr_for_cond.clone()),
                Hlvalue::Variable(length_for_cond),
                Hlvalue::Variable(c1_for_cond),
                Hlvalue::Variable(c2_for_cond),
                Hlvalue::Variable(j_for_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(newstr_for_cond)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- loop_body: c = src[j]; if c == c1: c = c2.
    let c = variable_with_lltype("c", elem_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(src_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    let replace = variable_with_lltype("replace", LowLevelType::Bool);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            eq_op,
            vec![
                Hlvalue::Variable(c.clone()),
                Hlvalue::Variable(c1_for_body.clone()),
            ],
            Hlvalue::Variable(replace.clone()),
        ));
    block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(replace));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_for_body.clone()),
                Hlvalue::Variable(dst_for_body.clone()),
                Hlvalue::Variable(newstr_for_body.clone()),
                Hlvalue::Variable(length_for_body.clone()),
                Hlvalue::Variable(c1_for_body.clone()),
                Hlvalue::Variable(c2_for_body.clone()),
                Hlvalue::Variable(j_for_body.clone()),
                Hlvalue::Variable(c2_for_body.clone()),
            ],
            Some(block_store.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(src_for_body),
                Hlvalue::Variable(dst_for_body),
                Hlvalue::Variable(newstr_for_body),
                Hlvalue::Variable(length_for_body),
                Hlvalue::Variable(c1_for_body),
                Hlvalue::Variable(c2_for_body),
                Hlvalue::Variable(j_for_body),
                Hlvalue::Variable(c),
            ],
            Some(block_store.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- store: dst[j] = c; j += 1.
    let set_void = variable_with_lltype("set", LowLevelType::Void);
    block_store
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(dst_for_store.clone()),
                Hlvalue::Variable(j_for_store.clone()),
                Hlvalue::Variable(c_for_store),
            ],
            Hlvalue::Variable(set_void),
        ));
    let j_next = variable_with_lltype("j_next", LowLevelType::Signed);
    block_store
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(j_for_store), signed_const(1)],
            Hlvalue::Variable(j_next.clone()),
        ));
    block_store.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_for_store),
                Hlvalue::Variable(dst_for_store),
                Hlvalue::Variable(newstr_for_store),
                Hlvalue::Variable(length_for_store),
                Hlvalue::Variable(c1_for_store),
                Hlvalue::Variable(c2_for_store),
                Hlvalue::Variable(j_next),
            ],
            Some(block_loop_cond),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string(), "c1".to_string(), "c2".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `AbstractLLHelpers.ll_stritem_nonneg_checked`
/// (`rtyper/rstr.py:950-953`):
///
/// ```python
/// @classmethod
/// def ll_stritem_nonneg_checked(cls, s, i):
///     if i >= cls.ll_strlen(s):
///         raise IndexError
///     return cls.ll_stritem_nonneg(s, i)
/// ```
///
/// 2-block CFG plus the returnblock and graph.exceptblock:
/// - **start**: `chars = getsubstruct(s, 'chars')`,
///   `length = getarraysize(chars)`, `oob = int_ge(i, length)`. True →
///   `graph.exceptblock` with `[exc_cls, exc_inst]` for IndexError;
///   False → `block_dispatch(s, i)`.
/// - **block_dispatch**: `c = direct_call(ll_stritem_nonneg, s, i)`.
///   Link to returnblock with `c`.
///
/// `inner_helper_name` is the cache key for the `ll_stritem_nonneg`
/// sub-helper (`ll_stritem_nonneg` for STR pair,
/// `ll_unicode_stritem_nonneg` for UNICODE pair).
pub(crate) fn build_ll_stritem_nonneg_checked_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    inner_helper_name: &str,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_stritem_nonneg_checked helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_stritem_nonneg_checked helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    if !matches!(elem_lltype, LowLevelType::Char | LowLevelType::UniChar) {
        return Err(TyperError::message(format!(
            "ll_stritem_nonneg_checked helper unsupported chars element type {elem_lltype:?}"
        )));
    }

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Materialise (or retrieve cached) ll_stritem_nonneg sub-helper.
    let inner_name_owned = inner_helper_name.to_string();
    let ptr_for_inner = ptr_lltype.clone();
    let inner_helper = rtyper.lowlevel_helper_function_with_builder(
        inner_helper_name.to_string(),
        vec![ptr_lltype.clone(), LowLevelType::Signed],
        elem_lltype.clone(),
        move |_rtyper, _args, _result| {
            build_ll_stritem_nonneg_helper_graph(&inner_name_owned, ptr_for_inner)
        },
    )?;
    let c_inner_funcptr = sub_helper_funcptr_constant(rtyper, &inner_helper)?;

    let exc_args = exception_args("IndexError")?;

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let i = variable_with_lltype("i", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(i.clone()),
    ]);
    let return_var = variable_with_lltype("result", elem_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // block_dispatch carries (s, i) and direct_calls ll_stritem_nonneg.
    let s_for_dispatch = variable_with_lltype("s", ptr_lltype.clone());
    let i_for_dispatch = variable_with_lltype("i", LowLevelType::Signed);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(s_for_dispatch.clone()),
        Hlvalue::Variable(i_for_dispatch.clone()),
    ]);

    // ---- start: getsubstruct + getarraysize + int_ge(i, length); branch.
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s.clone()), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars)],
        Hlvalue::Variable(length.clone()),
    ));
    let oob = variable_with_lltype("oob", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![Hlvalue::Variable(i.clone()), Hlvalue::Variable(length)],
        Hlvalue::Variable(oob.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(oob));
    startblock.closeblock(vec![
        // True (i >= length): raise IndexError.
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        // False (in-bounds): forward to block_dispatch.
        Link::new(
            vec![Hlvalue::Variable(s), Hlvalue::Variable(i)],
            Some(block_dispatch.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_dispatch: direct_call(ll_stritem_nonneg, s, i).
    let c = variable_with_lltype("c", elem_lltype);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_inner_funcptr),
                Hlvalue::Variable(s_for_dispatch),
                Hlvalue::Variable(i_for_dispatch),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    block_dispatch.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string(), "i".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `AbstractLLHelpers.ll_stritem_checked`
/// (`rtyper/rstr.py:962-967`):
///
/// ```python
/// @classmethod
/// def ll_stritem_checked(cls, s, i):
///     length = cls.ll_strlen(s)
///     if i < 0:
///         i += length
///     if i >= length or i < 0:
///         raise IndexError
///     return cls.ll_stritem_nonneg(s, i)
/// ```
///
/// 5-block CFG plus the returnblock and graph.exceptblock. The
/// `i >= length or i < 0` short-circuit lowers to two sequential
/// conditional blocks (`block_check_high` then `block_check_low`),
/// each with its own raise edge to graph.exceptblock — matching
/// flowspace's `or` lowering.
///
/// - **start**: `chars = getsubstruct`, `length = getarraysize`,
///   `is_neg = int_lt(i, 0)`. True → `block_neg_fix(s, length, i)`;
///   False → `block_check_high(s, length, i)`.
/// - **block_neg_fix**: `i_fix = int_add(i, length)`. Forward
///   `block_check_high(s, length, i_fix)`.
/// - **block_check_high**: `too_high = int_ge(i, length)`. True →
///   raise IndexError; False → `block_check_low(s, i)`.
/// - **block_check_low**: `too_low = int_lt(i, 0)`. True → raise;
///   False → `block_dispatch(s, i)`.
/// - **block_dispatch**: `c = direct_call(ll_stritem_nonneg, s, i)`.
///   Forward returnblock with `c`.
///
/// `inner_helper_name` is the cache key for the `ll_stritem_nonneg`
/// sub-helper.
pub(crate) fn build_ll_stritem_checked_helper_graph(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    inner_helper_name: &str,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_stritem_checked helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_stritem_checked helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    if !matches!(elem_lltype, LowLevelType::Char | LowLevelType::UniChar) {
        return Err(TyperError::message(format!(
            "ll_stritem_checked helper unsupported chars element type {elem_lltype:?}"
        )));
    }

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    // Materialise (or retrieve cached) ll_stritem_nonneg sub-helper.
    let inner_name_owned = inner_helper_name.to_string();
    let ptr_for_inner = ptr_lltype.clone();
    let inner_helper = rtyper.lowlevel_helper_function_with_builder(
        inner_helper_name.to_string(),
        vec![ptr_lltype.clone(), LowLevelType::Signed],
        elem_lltype.clone(),
        move |_rtyper, _args, _result| {
            build_ll_stritem_nonneg_helper_graph(&inner_name_owned, ptr_for_inner)
        },
    )?;
    let c_inner_funcptr = sub_helper_funcptr_constant(rtyper, &inner_helper)?;

    let exc_args = exception_args("IndexError")?;

    let s = variable_with_lltype("s", ptr_lltype.clone());
    let i = variable_with_lltype("i", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s.clone()),
        Hlvalue::Variable(i.clone()),
    ]);
    let return_var = variable_with_lltype("result", elem_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // block_neg_fix: takes (s, length, i); computes i_fix = i + length.
    let s_for_fix = variable_with_lltype("s", ptr_lltype.clone());
    let length_for_fix = variable_with_lltype("length", LowLevelType::Signed);
    let i_for_fix = variable_with_lltype("i", LowLevelType::Signed);
    let block_neg_fix = Block::shared(vec![
        Hlvalue::Variable(s_for_fix.clone()),
        Hlvalue::Variable(length_for_fix.clone()),
        Hlvalue::Variable(i_for_fix.clone()),
    ]);

    // block_check_high: takes (s, length, i); int_ge(i, length).
    let s_for_high = variable_with_lltype("s", ptr_lltype.clone());
    let length_for_high = variable_with_lltype("length", LowLevelType::Signed);
    let i_for_high = variable_with_lltype("i", LowLevelType::Signed);
    let block_check_high = Block::shared(vec![
        Hlvalue::Variable(s_for_high.clone()),
        Hlvalue::Variable(length_for_high.clone()),
        Hlvalue::Variable(i_for_high.clone()),
    ]);

    // block_check_low: takes (s, i); int_lt(i, 0).
    let s_for_low = variable_with_lltype("s", ptr_lltype.clone());
    let i_for_low = variable_with_lltype("i", LowLevelType::Signed);
    let block_check_low = Block::shared(vec![
        Hlvalue::Variable(s_for_low.clone()),
        Hlvalue::Variable(i_for_low.clone()),
    ]);

    // block_dispatch: takes (s, i); direct_calls ll_stritem_nonneg.
    let s_for_dispatch = variable_with_lltype("s", ptr_lltype.clone());
    let i_for_dispatch = variable_with_lltype("i", LowLevelType::Signed);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(s_for_dispatch.clone()),
        Hlvalue::Variable(i_for_dispatch.clone()),
    ]);

    // ---- start: getsubstruct + getarraysize + int_lt(i, 0); branch.
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s.clone()), chars_field_const()],
        Hlvalue::Variable(chars.clone()),
    ));
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars)],
        Hlvalue::Variable(length.clone()),
    ));
    let is_neg = variable_with_lltype("is_neg", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(i.clone()), signed_const(0)],
        Hlvalue::Variable(is_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_neg));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s.clone()),
                Hlvalue::Variable(length.clone()),
                Hlvalue::Variable(i.clone()),
            ],
            Some(block_neg_fix.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(s),
                Hlvalue::Variable(length),
                Hlvalue::Variable(i),
            ],
            Some(block_check_high.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_neg_fix: i_fix = int_add(i, length).
    let i_fix = variable_with_lltype("i_fix", LowLevelType::Signed);
    block_neg_fix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(i_for_fix),
                Hlvalue::Variable(length_for_fix.clone()),
            ],
            Hlvalue::Variable(i_fix.clone()),
        ));
    block_neg_fix.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s_for_fix),
                Hlvalue::Variable(length_for_fix),
                Hlvalue::Variable(i_fix),
            ],
            Some(block_check_high.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ---- block_check_high: too_high = int_ge(i, length); branch.
    let too_high = variable_with_lltype("too_high", LowLevelType::Bool);
    block_check_high
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ge",
            vec![
                Hlvalue::Variable(i_for_high.clone()),
                Hlvalue::Variable(length_for_high),
            ],
            Hlvalue::Variable(too_high.clone()),
        ));
    block_check_high.borrow_mut().exitswitch = Some(Hlvalue::Variable(too_high));
    block_check_high.closeblock(vec![
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(s_for_high), Hlvalue::Variable(i_for_high)],
            Some(block_check_low.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_check_low: too_low = int_lt(i, 0); branch.
    let too_low = variable_with_lltype("too_low", LowLevelType::Bool);
    block_check_low
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![Hlvalue::Variable(i_for_low.clone()), signed_const(0)],
            Hlvalue::Variable(too_low.clone()),
        ));
    block_check_low.borrow_mut().exitswitch = Some(Hlvalue::Variable(too_low));
    block_check_low.closeblock(vec![
        Link::new(exc_args, Some(graph.exceptblock.clone()), Some(bool_true())).into_ref(),
        Link::new(
            vec![Hlvalue::Variable(s_for_low), Hlvalue::Variable(i_for_low)],
            Some(block_dispatch.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_dispatch: direct_call(ll_stritem_nonneg, s, i).
    let c = variable_with_lltype("c", elem_lltype);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_inner_funcptr),
                Hlvalue::Variable(s_for_dispatch),
                Hlvalue::Variable(i_for_dispatch),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    block_dispatch.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string(), "i".to_string()],
        func,
    ))
}

/// Synthesise the helper graph for `LLHelpers.ll_strconcat`
/// (`rtyper/lltypesystem/rstr.py:425-444`):
///
/// ```python
/// @staticmethod
/// @jit.elidable
/// @jit.oopspec('stroruni.concat(s1, s2)')
/// def ll_strconcat(s1, s2):
///     len1 = s1.length()
///     len2 = s2.length()
///     newstr = s2.malloc(len1 + len2)
///     newstr.copy_contents_from_str(s1, newstr, 0, 0, len1)
///     newstr.copy_contents_from_str(s2, newstr, 0, len1, len2)
///     return newstr
/// ```
///
/// Pyre lowers `s.malloc(N)` to a `malloc_varsize` SpaceOperation
/// (rstr.py:1131 emit pattern: `[cTEMP, cflags, size]`) and inlines
/// the two `copy_contents_from_str` calls as explicit per-char copy
/// loops over `s.chars` / `newstr.chars` — the upstream
/// `copy_string_contents` itself is a `mh.copy_string_contents`
/// adtmeth wrapper that, after rtyping, decomposes into the same
/// `getarrayitem` / `setarrayitem` loop shape.
///
/// 5-block-plus-returnblock CFG:
/// - **start**: extract chars1/len1/chars2/len2, compute
///   `total = int_add(len1, len2)`, allocate `newstr` via
///   `malloc_varsize(struct_lltype, gc_flavor, total)`, fetch
///   `newchars = getsubstruct(newstr, 'chars')`. Forward into
///   `block_copy1_cond(newstr, newchars, chars1, chars2, len1, len2, 0)`.
/// - **block_copy1_cond**: `cond = int_lt(i, len1)`. True →
///   `block_copy1_body`; False → `block_copy2_cond(..., 0)`.
/// - **block_copy1_body**: `c = getarrayitem(chars1, i)`,
///   `setarrayitem(newchars, i, c)`, `i_next = int_add(i, 1)`. Cycle
///   back into `block_copy1_cond` with `i_next`.
/// - **block_copy2_cond**: `cond = int_lt(j, len2)`. True →
///   `block_copy2_body`; False → returnblock(newstr).
/// - **block_copy2_body**: `c = getarrayitem(chars2, j)`,
///   `dst_idx = int_add(len1, j)`, `setarrayitem(newchars, dst_idx,
///   c)`, `j_next = int_add(j, 1)`. Cycle back into
///   `block_copy2_cond` with `j_next`.
///
/// Polymorphic over `Ptr(STR)` / `Ptr(UNICODE)`. The chars-array
/// element type drives the lltype of the per-loop `c` temporary
/// (Char or UniChar); `setarrayitem` does not require a per-element-
/// type op selection (unlike `char_eq` vs `unichar_eq`) since it
/// emits a single op name.
pub(crate) fn build_ll_strconcat_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let LowLevelType::Ptr(chars_ptr) = &chars_array_ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_strconcat helper expects Ptr(Array(Char|UniChar)), \
             got {chars_array_ptr_lltype:?}"
        )));
    };
    let PtrTarget::Array(arr) = &chars_ptr.TO else {
        return Err(TyperError::message(format!(
            "ll_strconcat helper Ptr.TO must be Array, got {:?}",
            chars_ptr.TO
        )));
    };
    let elem_lltype = arr.OF.clone();
    if !matches!(elem_lltype, LowLevelType::Char | LowLevelType::UniChar) {
        return Err(TyperError::message(format!(
            "ll_strconcat helper unsupported chars element type {elem_lltype:?}"
        )));
    }

    let struct_lltype = struct_lltype_from_strptr(&ptr_lltype)?;

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let chars_field_const =
        || constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void);

    let s1 = variable_with_lltype("s1", ptr_lltype.clone());
    let s2 = variable_with_lltype("s2", ptr_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(s1.clone()),
        Hlvalue::Variable(s2.clone()),
    ]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // ---- Block declarations: copy1_cond / copy1_body / copy2_cond / copy2_body.
    //
    // copy1 carries (newstr, newchars, chars1, chars2, len1, len2, i).
    let newstr_for_c1c = variable_with_lltype("newstr", ptr_lltype.clone());
    let newchars_for_c1c = variable_with_lltype("newchars", chars_array_ptr_lltype.clone());
    let chars1_for_c1c = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_c1c = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len1_for_c1c = variable_with_lltype("len1", LowLevelType::Signed);
    let len2_for_c1c = variable_with_lltype("len2", LowLevelType::Signed);
    let i_for_c1c = variable_with_lltype("i", LowLevelType::Signed);
    let block_copy1_cond = Block::shared(vec![
        Hlvalue::Variable(newstr_for_c1c.clone()),
        Hlvalue::Variable(newchars_for_c1c.clone()),
        Hlvalue::Variable(chars1_for_c1c.clone()),
        Hlvalue::Variable(chars2_for_c1c.clone()),
        Hlvalue::Variable(len1_for_c1c.clone()),
        Hlvalue::Variable(len2_for_c1c.clone()),
        Hlvalue::Variable(i_for_c1c.clone()),
    ]);

    let newstr_for_c1b = variable_with_lltype("newstr", ptr_lltype.clone());
    let newchars_for_c1b = variable_with_lltype("newchars", chars_array_ptr_lltype.clone());
    let chars1_for_c1b = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    let chars2_for_c1b = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len1_for_c1b = variable_with_lltype("len1", LowLevelType::Signed);
    let len2_for_c1b = variable_with_lltype("len2", LowLevelType::Signed);
    let i_for_c1b = variable_with_lltype("i", LowLevelType::Signed);
    let block_copy1_body = Block::shared(vec![
        Hlvalue::Variable(newstr_for_c1b.clone()),
        Hlvalue::Variable(newchars_for_c1b.clone()),
        Hlvalue::Variable(chars1_for_c1b.clone()),
        Hlvalue::Variable(chars2_for_c1b.clone()),
        Hlvalue::Variable(len1_for_c1b.clone()),
        Hlvalue::Variable(len2_for_c1b.clone()),
        Hlvalue::Variable(i_for_c1b.clone()),
    ]);

    // copy2 carries (newstr, newchars, chars2, len1, len2, j) — chars1
    // is no longer needed once copy1 finishes.
    let newstr_for_c2c = variable_with_lltype("newstr", ptr_lltype.clone());
    let newchars_for_c2c = variable_with_lltype("newchars", chars_array_ptr_lltype.clone());
    let chars2_for_c2c = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len1_for_c2c = variable_with_lltype("len1", LowLevelType::Signed);
    let len2_for_c2c = variable_with_lltype("len2", LowLevelType::Signed);
    let j_for_c2c = variable_with_lltype("j", LowLevelType::Signed);
    let block_copy2_cond = Block::shared(vec![
        Hlvalue::Variable(newstr_for_c2c.clone()),
        Hlvalue::Variable(newchars_for_c2c.clone()),
        Hlvalue::Variable(chars2_for_c2c.clone()),
        Hlvalue::Variable(len1_for_c2c.clone()),
        Hlvalue::Variable(len2_for_c2c.clone()),
        Hlvalue::Variable(j_for_c2c.clone()),
    ]);

    let newstr_for_c2b = variable_with_lltype("newstr", ptr_lltype.clone());
    let newchars_for_c2b = variable_with_lltype("newchars", chars_array_ptr_lltype.clone());
    let chars2_for_c2b = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    let len1_for_c2b = variable_with_lltype("len1", LowLevelType::Signed);
    let len2_for_c2b = variable_with_lltype("len2", LowLevelType::Signed);
    let j_for_c2b = variable_with_lltype("j", LowLevelType::Signed);
    let block_copy2_body = Block::shared(vec![
        Hlvalue::Variable(newstr_for_c2b.clone()),
        Hlvalue::Variable(newchars_for_c2b.clone()),
        Hlvalue::Variable(chars2_for_c2b.clone()),
        Hlvalue::Variable(len1_for_c2b.clone()),
        Hlvalue::Variable(len2_for_c2b.clone()),
        Hlvalue::Variable(j_for_c2b.clone()),
    ]);

    // ---- start: extract chars + lengths, malloc_varsize, get newchars.
    let chars1 = variable_with_lltype("chars1", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s1), chars_field_const()],
        Hlvalue::Variable(chars1.clone()),
    ));
    let len1 = variable_with_lltype("len1", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars1.clone())],
        Hlvalue::Variable(len1.clone()),
    ));
    let chars2 = variable_with_lltype("chars2", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(s2), chars_field_const()],
        Hlvalue::Variable(chars2.clone()),
    ));
    let len2 = variable_with_lltype("len2", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars2.clone())],
        Hlvalue::Variable(len2.clone()),
    ));
    let total = variable_with_lltype("total", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![
            Hlvalue::Variable(len1.clone()),
            Hlvalue::Variable(len2.clone()),
        ],
        Hlvalue::Variable(total.clone()),
    ));
    let newstr = variable_with_lltype("newstr", ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc_varsize",
        vec![
            lowlevel_type_const(struct_lltype),
            gc_flavor_const()?,
            Hlvalue::Variable(total),
        ],
        Hlvalue::Variable(newstr.clone()),
    ));
    let newchars = variable_with_lltype("newchars", chars_array_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![Hlvalue::Variable(newstr.clone()), chars_field_const()],
        Hlvalue::Variable(newchars.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(newstr),
                Hlvalue::Variable(newchars),
                Hlvalue::Variable(chars1),
                Hlvalue::Variable(chars2),
                Hlvalue::Variable(len1),
                Hlvalue::Variable(len2),
                signed_const(0),
            ],
            Some(block_copy1_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ---- block_copy1_cond: int_lt(i, len1).
    let cond1 = variable_with_lltype("cond1", LowLevelType::Bool);
    block_copy1_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_for_c1c.clone()),
                Hlvalue::Variable(len1_for_c1c.clone()),
            ],
            Hlvalue::Variable(cond1.clone()),
        ));
    block_copy1_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond1));
    block_copy1_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(newstr_for_c1c.clone()),
                Hlvalue::Variable(newchars_for_c1c.clone()),
                Hlvalue::Variable(chars1_for_c1c.clone()),
                Hlvalue::Variable(chars2_for_c1c.clone()),
                Hlvalue::Variable(len1_for_c1c.clone()),
                Hlvalue::Variable(len2_for_c1c.clone()),
                Hlvalue::Variable(i_for_c1c.clone()),
            ],
            Some(block_copy1_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(newstr_for_c1c),
                Hlvalue::Variable(newchars_for_c1c),
                Hlvalue::Variable(chars2_for_c1c),
                Hlvalue::Variable(len1_for_c1c),
                Hlvalue::Variable(len2_for_c1c),
                signed_const(0),
            ],
            Some(block_copy2_cond.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_copy1_body: c = getarrayitem(chars1, i); setarrayitem;
    //      i_next = int_add(i, 1); cycle back.
    let c1 = variable_with_lltype("c", elem_lltype.clone());
    block_copy1_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars1_for_c1b.clone()),
                Hlvalue::Variable(i_for_c1b.clone()),
            ],
            Hlvalue::Variable(c1.clone()),
        ));
    let void_set1 = variable_with_lltype("set1", LowLevelType::Void);
    block_copy1_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(newchars_for_c1b.clone()),
                Hlvalue::Variable(i_for_c1b.clone()),
                Hlvalue::Variable(c1),
            ],
            Hlvalue::Variable(void_set1),
        ));
    let i_next = variable_with_lltype("i_next", LowLevelType::Signed);
    block_copy1_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_for_c1b), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    block_copy1_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(newstr_for_c1b),
                Hlvalue::Variable(newchars_for_c1b),
                Hlvalue::Variable(chars1_for_c1b),
                Hlvalue::Variable(chars2_for_c1b),
                Hlvalue::Variable(len1_for_c1b),
                Hlvalue::Variable(len2_for_c1b),
                Hlvalue::Variable(i_next),
            ],
            Some(block_copy1_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ---- block_copy2_cond: int_lt(j, len2).
    let cond2 = variable_with_lltype("cond2", LowLevelType::Bool);
    block_copy2_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(j_for_c2c.clone()),
                Hlvalue::Variable(len2_for_c2c.clone()),
            ],
            Hlvalue::Variable(cond2.clone()),
        ));
    block_copy2_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond2));
    block_copy2_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(newstr_for_c2c.clone()),
                Hlvalue::Variable(newchars_for_c2c.clone()),
                Hlvalue::Variable(chars2_for_c2c.clone()),
                Hlvalue::Variable(len1_for_c2c.clone()),
                Hlvalue::Variable(len2_for_c2c.clone()),
                Hlvalue::Variable(j_for_c2c.clone()),
            ],
            Some(block_copy2_body.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(newstr_for_c2c)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ---- block_copy2_body: c = getarrayitem(chars2, j);
    //      dst_idx = int_add(len1, j); setarrayitem(newchars, dst_idx, c);
    //      j_next = int_add(j, 1); cycle back.
    let c2 = variable_with_lltype("c", elem_lltype);
    block_copy2_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(chars2_for_c2b.clone()),
                Hlvalue::Variable(j_for_c2b.clone()),
            ],
            Hlvalue::Variable(c2.clone()),
        ));
    let dst_idx = variable_with_lltype("dst_idx", LowLevelType::Signed);
    block_copy2_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(len1_for_c2b.clone()),
                Hlvalue::Variable(j_for_c2b.clone()),
            ],
            Hlvalue::Variable(dst_idx.clone()),
        ));
    let void_set2 = variable_with_lltype("set2", LowLevelType::Void);
    block_copy2_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(newchars_for_c2b.clone()),
                Hlvalue::Variable(dst_idx),
                Hlvalue::Variable(c2),
            ],
            Hlvalue::Variable(void_set2),
        ));
    let j_next = variable_with_lltype("j_next", LowLevelType::Signed);
    block_copy2_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(j_for_c2b), signed_const(1)],
            Hlvalue::Variable(j_next.clone()),
        ));
    block_copy2_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(newstr_for_c2b),
                Hlvalue::Variable(newchars_for_c2b),
                Hlvalue::Variable(chars2_for_c2b),
                Hlvalue::Variable(len1_for_c2b),
                Hlvalue::Variable(len2_for_c2b),
                Hlvalue::Variable(j_next),
            ],
            Some(block_copy2_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s1".to_string(), "s2".to_string()],
        func,
    ))
}

/// Helper: derive a funcptr `Constant` (lltype `Ptr(FuncType)`) from
/// a sub-helper `LowLevelFunction` so it can be used as the first arg
/// of a `direct_call` SpaceOperation. Mirrors the funcptr derivation
/// inside
/// [`crate::translator::rtyper::rtyper::LowLevelOpList::gendirectcall`]
/// but exposed for builder closures that emit `direct_call` ops
/// directly into a synthesised `PyGraph`.
fn sub_helper_funcptr_constant(
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
    sub_helper: &crate::translator::rtyper::rtyper::LowLevelFunction,
) -> Result<Constant, TyperError> {
    let sub_graph = sub_helper.graph.as_ref().ok_or_else(|| {
        TyperError::message(format!(
            "sub-helper {} has no annotated helper graph",
            sub_helper.name
        ))
    })?;
    let func_ptr = rtyper.getcallable(sub_graph)?;
    let func_ptr_type = LowLevelType::Ptr(Box::new(func_ptr._TYPE.clone()));
    crate::translator::rtyper::rmodel::inputconst_from_lltype(
        &func_ptr_type,
        &ConstValue::LLPtr(Box::new(func_ptr)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `STR` resolves to a GcStruct named `rpy_string` with the two
    /// upstream fields `(hash: Signed, chars: Array(Char, ...))` and
    /// the `remove_hash` hint that upstream attaches at
    /// rstr.py:1237.
    #[test]
    fn str_resolves_to_gcstruct_rpy_string_with_hash_and_chars() {
        let resolved = match STR.clone() {
            LowLevelType::ForwardReference(fwd) => {
                fwd.resolved().expect("STR ForwardReference must resolve")
            }
            other => panic!("STR must be ForwardReference, got {other:?}"),
        };
        let LowLevelType::Struct(body) = resolved else {
            panic!("STR resolves to Struct");
        };
        assert_eq!(body._name, "rpy_string");
        assert!(matches!(body._flds.get("hash"), Some(LowLevelType::Signed)));
        assert!(matches!(
            body._flds.get("chars"),
            Some(LowLevelType::Array(_))
        ));
        // upstream `hints={'remove_hash': True}`.
        assert!(
            body._hints.get("remove_hash").is_some(),
            "remove_hash hint must be present on STR"
        );
    }

    /// `STRPTR` is `Ptr(STR)` — the lltype that `OBJECT_VTABLE.name`
    /// uses and that `ClassesPBCRepr.rtype_getattr('__name__')` would
    /// surface as the resulttype of its `getfield` op.
    #[test]
    fn strptr_is_ptr_to_str_forwardreference() {
        let LowLevelType::Ptr(ptr) = STRPTR.clone() else {
            panic!("STRPTR must be Ptr");
        };
        assert!(matches!(&ptr.TO, PtrTarget::ForwardReference(_)));
    }

    #[test]
    fn alloc_array_name_returns_live_str_with_chars() {
        let p = alloc_array_name("Foo").expect("alloc_array_name");
        assert!(p.nonzero());
        let LowLevelValue::Signed(0) = p.getattr("hash").unwrap() else {
            panic!("hash field must be Signed(0)");
        };
        let Some(obj) = p._obj0.as_ref().unwrap().as_ref() else {
            panic!("STR pointer must be live");
        };
        let _ptr_obj::Struct(s) = obj else {
            panic!("STR pointer must target a Struct");
        };
        let chars_value = s._getattr("chars").expect("chars field must be present");
        let LowLevelValue::Array(chars) = chars_value else {
            panic!("chars field must be an Array");
        };
        assert_eq!(chars.getlength(), 3);
        assert_eq!(chars.getitem(0), Some(LowLevelValue::Char('F')));
        assert_eq!(chars.getitem(1), Some(LowLevelValue::Char('o')));
        assert_eq!(chars.getitem(2), Some(LowLevelValue::Char('o')));
    }

    /// `UNICODE` resolves to a GcStruct named `rpy_unicode` with
    /// the two upstream fields `(hash: Signed, chars: Array(UniChar,
    /// ...))` and the `remove_hash` hint that upstream attaches at
    /// rstr.py:1246. Mirror of
    /// [`str_resolves_to_gcstruct_rpy_string_with_hash_and_chars`]
    /// — verifies the `Array` element type is `UniChar` (not `Char`)
    /// so that an `Ptr(UNICODE).chars[i]` dereference lowers to a
    /// codepoint, not a byte.
    #[test]
    fn unicode_resolves_to_gcstruct_rpy_unicode_with_hash_and_unichar_chars() {
        let resolved = match UNICODE.clone() {
            LowLevelType::ForwardReference(fwd) => fwd
                .resolved()
                .expect("UNICODE ForwardReference must resolve"),
            other => panic!("UNICODE must be ForwardReference, got {other:?}"),
        };
        let LowLevelType::Struct(body) = resolved else {
            panic!("UNICODE resolves to Struct");
        };
        assert_eq!(body._name, "rpy_unicode");
        assert!(matches!(body._flds.get("hash"), Some(LowLevelType::Signed)));
        let Some(LowLevelType::Array(arr)) = body._flds.get("chars") else {
            panic!("chars field must be Array");
        };
        assert_eq!(arr.OF, LowLevelType::UniChar);
        assert!(
            body._hints.get("remove_hash").is_some(),
            "remove_hash hint must be present on UNICODE"
        );
    }

    /// `UNICODEPTR` is `Ptr(UNICODE)` — the lltype that
    /// `UnicodeRepr.lowleveltype` (rstr.py:248) carries.
    #[test]
    fn unicodeptr_is_ptr_to_unicode_forwardreference() {
        let LowLevelType::Ptr(ptr) = UNICODEPTR.clone() else {
            panic!("UNICODEPTR must be Ptr");
        };
        assert!(matches!(&ptr.TO, PtrTarget::ForwardReference(_)));
    }

    /// Round-trip a non-ASCII Unicode string through
    /// `alloc_array_unicode` and verify each codepoint lands as a
    /// distinct `UniChar` cell — confirms that the chars Array
    /// element type is wide enough for the full Unicode range, not
    /// truncated to a single byte.
    #[test]
    fn alloc_array_unicode_returns_live_unicode_with_unichar_codepoints() {
        let p = alloc_array_unicode("αβγ").expect("alloc_array_unicode");
        assert!(p.nonzero());
        let LowLevelValue::Signed(0) = p.getattr("hash").unwrap() else {
            panic!("hash field must be Signed(0)");
        };
        let Some(obj) = p._obj0.as_ref().unwrap().as_ref() else {
            panic!("UNICODE pointer must be live");
        };
        let _ptr_obj::Struct(s) = obj else {
            panic!("UNICODE pointer must target a Struct");
        };
        let chars_value = s._getattr("chars").expect("chars field must be present");
        let LowLevelValue::Array(chars) = chars_value else {
            panic!("chars field must be an Array");
        };
        assert_eq!(chars.getlength(), 3);
        assert_eq!(chars.getitem(0), Some(LowLevelValue::UniChar('α')));
        assert_eq!(chars.getitem(1), Some(LowLevelValue::UniChar('β')));
        assert_eq!(chars.getitem(2), Some(LowLevelValue::UniChar('γ')));
    }

    /// `LLHelpers.ll_chr2str` allocates a length-1 STR and stores the
    /// incoming Char at index 0. The Rust helper graph mirrors that as
    /// `malloc_varsize` + `getsubstruct('chars')` + `setarrayitem`.
    #[test]
    fn build_ll_chr2str_synthesizes_malloc_varsize_and_sets_char_zero() {
        let helper =
            build_ll_chr2str_helper_graph("ll_chr2str", STRPTR.clone(), LowLevelType::Char)
                .expect("build_ll_chr2str_helper_graph");
        let inner = helper.graph.borrow();
        let sb = inner.startblock.borrow();
        let opnames: Vec<&str> = sb.operations.iter().map(|op| op.opname.as_str()).collect();
        assert_eq!(
            opnames,
            vec!["malloc_varsize", "getsubstruct", "setarrayitem"]
        );

        let malloc_op = &sb.operations[0];
        assert_eq!(malloc_op.args.len(), 3);
        assert!(matches!(malloc_op.args[0], Hlvalue::Constant(_)));
        assert!(matches!(malloc_op.args[1], Hlvalue::Constant(_)));
        assert!(matches!(
            malloc_op.args[2],
            Hlvalue::Constant(Constant {
                value: ConstValue::Int(1),
                ..
            })
        ));

        let set_op = &sb.operations[2];
        assert_eq!(set_op.opname, "setarrayitem");
        assert!(matches!(
            set_op.args[1],
            Hlvalue::Constant(Constant {
                value: ConstValue::Int(0),
                ..
            })
        ));
    }

    /// The same upstream helper branches on `typeOf(ch)`: UniChar uses
    /// `mallocunicode(1)` and stores a UniChar into `UNICODE.chars[0]`.
    #[test]
    fn build_ll_chr2str_accepts_unichar_to_unicode_shape() {
        let helper = build_ll_chr2str_helper_graph(
            "ll_unichr2unicode",
            UNICODEPTR.clone(),
            LowLevelType::UniChar,
        )
        .expect("build_ll_chr2str_helper_graph for UniChar");
        let inner = helper.graph.borrow();
        let sb = inner.startblock.borrow();
        let opnames: Vec<&str> = sb.operations.iter().map(|op| op.opname.as_str()).collect();
        assert_eq!(
            opnames,
            vec!["malloc_varsize", "getsubstruct", "setarrayitem"]
        );
    }

    /// `ll_str2unicode` allocates UNICODE with the source length, then
    /// loops over bytes. The check block raises UnicodeDecodeError on
    /// non-ASCII input before storing through `cast_int_to_unichar`.
    #[test]
    fn build_ll_str2unicode_synthesizes_ascii_checked_copy_loop() {
        let helper =
            build_ll_str2unicode_helper_graph("ll_str2unicode").expect("ll_str2unicode helper");
        let inner = helper.graph.borrow();

        let block_loop_cond = {
            let sb = inner.startblock.borrow();
            let opnames: Vec<&str> = sb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(
                opnames,
                vec![
                    "getsubstruct",
                    "getarraysize",
                    "malloc_varsize",
                    "getsubstruct"
                ]
            );
            let malloc_op = &sb.operations[2];
            assert_eq!(malloc_op.opname, "malloc_varsize");
            assert!(matches!(malloc_op.args[2], Hlvalue::Variable(_)));
            sb.exits[0].borrow().target.as_ref().unwrap().clone()
        };

        let block_check_char = {
            let cb = block_loop_cond.borrow();
            let opnames: Vec<&str> = cb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(opnames, vec!["int_lt"]);
            cb.exits
                .iter()
                .find_map(|link| {
                    let link_borrow = link.borrow();
                    let is_true = matches!(
                        link_borrow.exitcase,
                        Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                    );
                    is_true.then(|| link_borrow.target.as_ref().unwrap().clone())
                })
                .expect("loop true branch")
        };

        let block_store_char = {
            let check = block_check_char.borrow();
            let opnames: Vec<&str> = check
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(opnames, vec!["getarrayitem", "cast_char_to_int", "int_gt"]);
            let mut saw_raise = false;
            let mut store_target = None;
            for link in &check.exits {
                let link_borrow = link.borrow();
                let target = link_borrow.target.as_ref().unwrap();
                let is_true = matches!(
                    link_borrow.exitcase,
                    Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                );
                if is_true {
                    assert!(std::rc::Rc::ptr_eq(target, &inner.exceptblock));
                    saw_raise = true;
                } else {
                    store_target = Some(target.clone());
                }
            }
            assert!(saw_raise, "non-ASCII branch must raise UnicodeDecodeError");
            store_target.expect("ASCII false branch")
        };

        let store = block_store_char.borrow();
        let opnames: Vec<&str> = store
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            opnames,
            vec!["cast_int_to_unichar", "setarrayitem", "int_add"]
        );
    }

    fn collect_reachable_blocks(
        start: &crate::flowspace::model::BlockRef,
    ) -> Vec<crate::flowspace::model::BlockRef> {
        let mut seen = Vec::new();
        let mut stack = vec![start.clone()];
        while let Some(block) = stack.pop() {
            if seen.iter().any(|known| std::rc::Rc::ptr_eq(known, &block)) {
                continue;
            }
            for link in &block.borrow().exits {
                if let Some(target) = link.borrow().target.as_ref() {
                    stack.push(target.clone());
                }
            }
            seen.push(block);
        }
        seen
    }

    /// `LLHelpers.ll_int` keeps PyPy's base parser shape: base range
    /// guard, leading/sign/trailing space loops, digit classification,
    /// base cutoff, ValueError exits, and final `sign * val`.
    #[test]
    fn build_ll_int_synthesizes_base_checked_string_parser_cfg() {
        let helper = build_ll_int_helper_graph("ll_int", STRPTR.clone()).expect("ll_int helper");
        let inner = helper.graph.borrow();

        {
            let startblock = inner.startblock.borrow();
            let start_ops: Vec<&str> = startblock
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(start_ops, vec!["int_ge"]);
        }

        let blocks = collect_reachable_blocks(&inner.startblock);
        for block in &blocks {
            for link in &block.borrow().exits {
                let link = link.borrow();
                if let Some(target) = link.target.as_ref() {
                    assert_eq!(
                        link.args.len(),
                        target.borrow().inputargs.len(),
                        "link arity must match target block arity"
                    );
                }
            }
        }

        let all_ops: Vec<String> = blocks
            .iter()
            .flat_map(|block| {
                block
                    .borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect::<Vec<_>>()
            })
            .collect();
        for opname in [
            "getsubstruct",
            "getarraysize",
            "cast_char_to_int",
            "int_ge",
            "int_le",
            "int_lt",
            "int_eq",
            "int_sub",
            "int_mul",
            "int_add",
        ] {
            assert!(
                all_ops.iter().any(|op| op == opname),
                "ll_int graph must contain {opname}"
            );
        }

        let raises_value_error = blocks.iter().any(|block| {
            block.borrow().exits.iter().any(|link| {
                let link = link.borrow();
                link.target
                    .as_ref()
                    .is_some_and(|target| std::rc::Rc::ptr_eq(target, &inner.exceptblock))
            })
        });
        assert!(
            raises_value_error,
            "invalid parse paths must raise ValueError"
        );
    }

    /// Unicode inherits the same parser, swapping Char loads for UniChar
    /// loads and `cast_unichar_to_int`.
    #[test]
    fn build_ll_int_uses_unichar_cast_for_unicode() {
        let helper = build_ll_int_helper_graph("ll_unicode_int", UNICODEPTR.clone())
            .expect("unicode ll_int helper");
        let inner = helper.graph.borrow();
        let blocks = collect_reachable_blocks(&inner.startblock);
        let all_ops: Vec<String> = blocks
            .iter()
            .flat_map(|block| {
                block
                    .borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(all_ops.iter().any(|op| op == "cast_unichar_to_int"));
        assert!(!all_ops.iter().any(|op| op == "cast_char_to_int"));
    }

    /// `ll_strlen` synthesised against `Ptr(STR)` produces a single
    /// startblock with `getsubstruct('chars')` then `getarraysize`,
    /// and returns `Signed`. Mirrors upstream
    /// `lltypesystem/rstr.py:351-352`:
    /// `def ll_strlen(s): return len(s.chars)`.
    /// `getsubstruct` (not `getfield`) is the correct lltype op for
    /// the inline composite `chars` field — see `rmodel.rs:1290-1306`
    /// `is_container_type` branch.
    #[test]
    fn build_ll_strlen_emits_getsubstruct_chars_then_getarraysize() {
        let helper = build_ll_strlen_helper_graph("ll_strlen", STRPTR.clone())
            .expect("build_ll_strlen_helper_graph");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["getsubstruct", "getarraysize"]);
        // `getsubstruct` first arg is the input `s`, second is the
        // void field-name constant `"chars"`.
        let getsubstruct_args = &startblock.operations[0].args;
        assert_eq!(getsubstruct_args.len(), 2);
        let Hlvalue::Constant(c) = &getsubstruct_args[1] else {
            panic!("getsubstruct's second arg must be a Constant");
        };
        assert!(matches!(c.value, ConstValue::ByteStr(ref b) if b == b"chars"));
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
    }

    /// `ll_unilen` is the `Ptr(UNICODE)` mirror of `ll_strlen` — same
    /// op sequence, just a different input pointer lltype. The chars
    /// Array element lltype derived from the struct body must be
    /// `UniChar`, not `Char`. The `getsubstruct` result is wrapped as
    /// `Ptr(Array(UniChar))` so it carries a valid `SomePtr`-shaped
    /// annotation.
    #[test]
    fn build_ll_unilen_emits_same_op_sequence_with_unichar_chars_lltype() {
        let helper = build_ll_strlen_helper_graph("ll_unilen", UNICODEPTR.clone())
            .expect("build_ll_strlen_helper_graph for UNICODE");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["getsubstruct", "getarraysize"]);
        let getsubstruct_result = &startblock.operations[0].result;
        let Hlvalue::Variable(v) = getsubstruct_result else {
            panic!("getsubstruct result must be a Variable");
        };
        let chars_lltype = v
            .concretetype
            .borrow()
            .clone()
            .expect("chars var must have concretetype");
        let LowLevelType::Ptr(ptr) = chars_lltype else {
            panic!("chars var must be Ptr");
        };
        let PtrTarget::Array(arr) = &ptr.TO else {
            panic!("chars Ptr.TO must be Array");
        };
        assert_eq!(arr.OF, LowLevelType::UniChar);
    }

    /// The synthesised helper graph carries the helper-identity
    /// `name` on both its FunctionGraph and `func` slot, and exposes
    /// the input slot via the startblock's input variable. Required so
    /// `helper_pygraph_from_graph`'s caller cache dedupes by
    /// `(name, args)` rather than by graph identity.
    #[test]
    fn build_ll_strlen_helper_carries_name_and_input_s_variable() {
        let helper = build_ll_strlen_helper_graph("ll_strlen", STRPTR.clone()).unwrap();
        assert_eq!(helper.func.name, "ll_strlen");
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "ll_strlen");
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 1);
        let Hlvalue::Variable(v) = &startblock.inputargs[0] else {
            panic!("startblock input must be a Variable");
        };
        assert!(
            v.name().starts_with('s'),
            "input variable name = {:?}",
            v.name()
        );
    }

    /// Passing a non-Ptr lltype should be rejected — upstream's
    /// `len(s.chars)` is only meaningful when `s: Ptr(STR/UNICODE)`.
    /// Guards a future caller from accidentally requesting a
    /// `ll_strlen` against e.g. `LowLevelType::Char`.
    #[test]
    fn build_ll_strlen_rejects_non_ptr_input_lltype() {
        let err = build_ll_strlen_helper_graph("ll_strlen", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(STR/UNICODE)"));
    }

    /// `ll_str_is_true` synthesised against `Ptr(STR)` produces a
    /// 2-block CFG: startblock with `ptr_nonzero(s)` + boolean
    /// exitswitch (True → block_check_len, False → returnblock with
    /// `Bool(false)`), and a check-length block emitting
    /// `getsubstruct('chars')` + `getarraysize` + `int_ne(len, 0)`.
    /// Mirrors upstream `rstr.py:944-947`:
    /// `def ll_str_is_true(cls, s): return bool(s) and cls.ll_strlen(s) != 0`.
    #[test]
    fn build_ll_str_is_true_synthesizes_2_block_short_circuit_cfg() {
        let helper = build_ll_str_is_true_helper_graph("ll_str_is_true", STRPTR.clone())
            .expect("build_ll_str_is_true_helper_graph");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["ptr_nonzero"]);
        assert!(startblock.exitswitch.is_some());
        assert_eq!(startblock.exits.len(), 2);

        // False branch: direct return with Bool(false) link arg.
        let false_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("False exit link present");
        let false_first_arg = false_link
            .borrow()
            .args
            .first()
            .and_then(|opt| opt.as_ref())
            .cloned()
            .expect("False link first arg present");
        assert!(matches!(
            false_first_arg,
            Hlvalue::Constant(c) if c.value == ConstValue::Bool(false)
        ));

        // True branch: forwards `s` ptr to block_check_len.
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link present");
        let target = true_link.borrow().target.as_ref().unwrap().clone();
        let target_block = target.borrow();
        let target_ops: Vec<&str> = target_block
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(target_ops, vec!["getsubstruct", "getarraysize", "int_ne"]);
    }

    /// `ll_unicode_is_true` is the `Ptr(UNICODE)` mirror of
    /// `ll_str_is_true` — same op sequence, but the chars-array `Ptr`
    /// derived in `block_check_len` carries `Array(UniChar)` rather
    /// than `Array(Char)` so that downstream length comparison
    /// dispatches against the unicode struct shape.
    #[test]
    fn build_ll_unicode_is_true_uses_unichar_chars_lltype() {
        let helper = build_ll_str_is_true_helper_graph("ll_unicode_is_true", UNICODEPTR.clone())
            .expect("build_ll_str_is_true_helper_graph for UNICODE");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link present");
        let target = true_link.borrow().target.as_ref().unwrap().clone();
        let target_block = target.borrow();
        let getsubstruct_result = &target_block.operations[0].result;
        let Hlvalue::Variable(v) = getsubstruct_result else {
            panic!("getsubstruct result must be a Variable");
        };
        let chars_lltype = v
            .concretetype
            .borrow()
            .clone()
            .expect("chars var must have concretetype");
        let LowLevelType::Ptr(ptr) = chars_lltype else {
            panic!("chars var must be Ptr");
        };
        let PtrTarget::Array(arr) = &ptr.TO else {
            panic!("chars Ptr.TO must be Array");
        };
        assert_eq!(arr.OF, LowLevelType::UniChar);
    }

    /// The synthesised helper graph carries the helper-identity `name`
    /// on both its FunctionGraph and `func` slot, returns Bool, and
    /// exposes the input `s` via the startblock's input variable.
    /// Mirror of `build_ll_strlen_helper_carries_name_and_input_s_variable`.
    #[test]
    fn build_ll_str_is_true_carries_name_input_s_and_bool_return() {
        let helper = build_ll_str_is_true_helper_graph("ll_str_is_true", STRPTR.clone()).unwrap();
        assert_eq!(helper.func.name, "ll_str_is_true");
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "ll_str_is_true");
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 1);
        let Hlvalue::Variable(v) = &startblock.inputargs[0] else {
            panic!("startblock input must be a Variable");
        };
        assert!(
            v.name().starts_with('s'),
            "input variable name = {:?}",
            v.name()
        );
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Bool),
            "ll_str_is_true must return Bool"
        );
    }

    /// Passing a non-Ptr lltype should be rejected — propagated from
    /// the shared `chars_array_ptr_lltype_from_strptr` guard.
    #[test]
    fn build_ll_str_is_true_rejects_non_ptr_input_lltype() {
        let err = build_ll_str_is_true_helper_graph("ll_str_is_true", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(STR/UNICODE)"));
    }

    /// `ll_streq` synthesised against `Ptr(STR)` produces the expected
    /// 6-block-plus-returnblock CFG. Each block carries the operations
    /// listed in the synthesizer docstring; the loop edge from
    /// `block_loop_body` back to `block_loop_cond` realises the
    /// upstream `while j < len1` cycle.
    #[test]
    fn build_ll_streq_synthesizes_short_circuit_loop_cfg_for_str() {
        let helper = build_ll_streq_helper_graph("ll_streq", STRPTR.clone())
            .expect("build_ll_streq_helper_graph for STR");
        let inner = helper.graph.borrow();

        // start: ptr_eq.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["ptr_eq"]);
        assert_eq!(startblock.exits.len(), 2);

        // start False branch → block_null_check_s1 (ptr_nonzero).
        let start_false = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("start False exit");
        let null_check_s1 = start_false.borrow().target.as_ref().unwrap().clone();
        let null_check_s1_borrow = null_check_s1.borrow();
        let null_s1_ops: Vec<&str> = null_check_s1_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(null_s1_ops, vec!["ptr_nonzero"]);

        // null_check_s1 True branch → block_null_check_s2 (ptr_nonzero).
        let s1_true = null_check_s1_borrow
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("null_check_s1 True exit");
        let null_check_s2 = s1_true.borrow().target.as_ref().unwrap().clone();
        let null_check_s2_borrow = null_check_s2.borrow();
        let null_s2_ops: Vec<&str> = null_check_s2_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(null_s2_ops, vec!["ptr_nonzero"]);

        // null_check_s2 True branch → block_compare_lens.
        let s2_true = null_check_s2_borrow
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("null_check_s2 True exit");
        let compare_lens = s2_true.borrow().target.as_ref().unwrap().clone();
        let compare_lens_borrow = compare_lens.borrow();
        let lens_ops: Vec<&str> = compare_lens_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            lens_ops,
            vec![
                "getsubstruct",
                "getarraysize",
                "getsubstruct",
                "getarraysize",
                "int_eq",
            ]
        );

        // compare_lens True branch → block_loop_cond (int_lt).
        let lens_true = compare_lens_borrow
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("compare_lens True exit");
        let loop_cond = lens_true.borrow().target.as_ref().unwrap().clone();
        let loop_cond_borrow = loop_cond.borrow();
        let cond_ops: Vec<&str> = loop_cond_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(cond_ops, vec!["int_lt"]);

        // loop_cond True branch → block_loop_body (getarrayitem,
        // getarrayitem, char_eq, int_add).
        let cond_true = loop_cond_borrow
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("loop_cond True exit");
        let loop_body = cond_true.borrow().target.as_ref().unwrap().clone();
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec!["getarrayitem", "getarrayitem", "char_eq", "int_add"]
        );

        // loop_body True branch loops back to block_loop_cond.
        let body_true = loop_body_borrow
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("loop_body True exit");
        let body_true_target = body_true.borrow().target.as_ref().unwrap().clone();
        // Identity: same Rc as loop_cond.
        assert!(std::rc::Rc::ptr_eq(&body_true_target, &loop_cond));
    }

    /// `ll_streq` against `Ptr(UNICODE)` differs only in the chars
    /// element comparison op — `unichar_eq` instead of `char_eq`. The
    /// body block's third op should be `unichar_eq`.
    #[test]
    fn build_ll_streq_uses_unichar_eq_op_for_unicode() {
        let helper = build_ll_streq_helper_graph("ll_unicode_eq", UNICODEPTR.clone())
            .expect("build_ll_streq_helper_graph for UNICODE");
        let inner = helper.graph.borrow();

        // Walk: start False → null_check_s1 True → null_check_s2 True
        // → compare_lens True → loop_cond True → loop_body.
        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }
        let null_check_s1 = pick_exit_target(&inner.startblock, false);
        let null_check_s2 = pick_exit_target(&null_check_s1, true);
        let compare_lens = pick_exit_target(&null_check_s2, true);
        let loop_cond = pick_exit_target(&compare_lens, true);
        let loop_body = pick_exit_target(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec!["getarrayitem", "getarrayitem", "unichar_eq", "int_add"]
        );
    }

    /// The synthesised `ll_streq` graph carries the helper-identity
    /// `name` on both `FunctionGraph.name` and `func.name`, exposes
    /// the two `s1` / `s2` input slots via the startblock's
    /// inputargs, and returns Bool. Mirrors the existing
    /// `build_ll_strlen_helper_carries_name_and_input_s_variable`
    /// covenant so callers (`gen_eq_function`) can dedupe by
    /// `(name, args)`.
    #[test]
    fn build_ll_streq_carries_name_two_inputs_and_bool_return() {
        let helper = build_ll_streq_helper_graph("ll_streq", STRPTR.clone()).unwrap();
        assert_eq!(helper.func.name, "ll_streq");
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "ll_streq");
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 2);
        for (i, expected) in ["s1", "s2"].iter().enumerate() {
            let Hlvalue::Variable(v) = &startblock.inputargs[i] else {
                panic!("startblock input {i} must be a Variable");
            };
            assert!(
                v.name().starts_with(expected),
                "input {i} variable name = {:?}, expected prefix {expected}",
                v.name()
            );
        }
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Bool),
            "ll_streq must return Bool"
        );
    }

    /// Passing a non-Ptr lltype should be rejected — propagated from
    /// the shared `chars_array_ptr_lltype_from_strptr` guard.
    #[test]
    fn build_ll_streq_rejects_non_ptr_input_lltype() {
        let err = build_ll_streq_helper_graph("ll_streq", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(STR/UNICODE)"));
    }

    /// `ll_strcmp` synthesised against `Ptr(STR)` produces the expected
    /// 7-block-plus-returnblock CFG. Walks: start False → block_s1_null
    /// (ptr_nonzero); start True → block_s1_nonnull (ptr_nonzero);
    /// block_s1_nonnull True → block_both_nonnull
    /// (getsubstruct/getarraysize × 2 + int_lt); block_both_nonnull True
    /// → block_loop_cond (int_lt); block_loop_cond True → block_loop_body
    /// (getarrayitem × 2 + cast_char_to_int × 2 + int_sub + int_ne +
    /// int_add); block_loop_cond False → block_return_len_diff (int_sub);
    /// block_loop_body False (chars match) cycles back to block_loop_cond.
    #[test]
    fn build_ll_strcmp_synthesizes_min_loop_cfg_with_back_edge_for_str() {
        let helper = build_ll_strcmp_helper_graph("ll_strcmp", STRPTR.clone())
            .expect("build_ll_strcmp_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        // start: ptr_nonzero(s1).
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["ptr_nonzero"]);
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);

        // start False → block_s1_null (ptr_nonzero).
        let s1_null = pick_exit_target(&inner.startblock, false);
        let s1_null_borrow = s1_null.borrow();
        let s1_null_ops: Vec<&str> = s1_null_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(s1_null_ops, vec!["ptr_nonzero"]);
        drop(s1_null_borrow);

        // start True → block_s1_nonnull (ptr_nonzero).
        let s1_nn = pick_exit_target(&inner.startblock, true);
        let s1_nn_borrow = s1_nn.borrow();
        let s1_nn_ops: Vec<&str> = s1_nn_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(s1_nn_ops, vec!["ptr_nonzero"]);
        drop(s1_nn_borrow);

        // block_s1_nonnull True → block_both_nonnull (getsubstruct,
        // getarraysize, getsubstruct, getarraysize, int_lt).
        let both_nn = pick_exit_target(&s1_nn, true);
        let both_nn_borrow = both_nn.borrow();
        let both_nn_ops: Vec<&str> = both_nn_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            both_nn_ops,
            vec![
                "getsubstruct",
                "getarraysize",
                "getsubstruct",
                "getarraysize",
                "int_lt",
            ]
        );
        drop(both_nn_borrow);

        // block_both_nonnull True → block_loop_cond (int_lt).
        let loop_cond = pick_exit_target(&both_nn, true);
        let loop_cond_borrow = loop_cond.borrow();
        let cond_ops: Vec<&str> = loop_cond_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(cond_ops, vec!["int_lt"]);
        drop(loop_cond_borrow);

        // block_loop_cond True → block_loop_body (getarrayitem×2,
        // cast_char_to_int×2, int_sub, int_ne, int_add).
        let loop_body = pick_exit_target(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec![
                "getarrayitem",
                "getarrayitem",
                "cast_char_to_int",
                "cast_char_to_int",
                "int_sub",
                "int_ne",
                "int_add",
            ]
        );
        drop(loop_body_borrow);

        // block_loop_body False (chars match) cycles to block_loop_cond.
        let body_false_target = pick_exit_target(&loop_body, false);
        assert!(std::rc::Rc::ptr_eq(&body_false_target, &loop_cond));

        // block_loop_cond False → block_return_len_diff (int_sub).
        let len_diff = pick_exit_target(&loop_cond, false);
        let len_diff_borrow = len_diff.borrow();
        let len_diff_ops: Vec<&str> = len_diff_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(len_diff_ops, vec!["int_sub"]);
    }

    /// `ll_strcmp` against `Ptr(UNICODE)` differs only in the chars
    /// element `ord` op — `cast_unichar_to_int` instead of
    /// `cast_char_to_int`. The body block's third + fourth ops should
    /// be `cast_unichar_to_int`.
    #[test]
    fn build_ll_strcmp_uses_cast_unichar_to_int_op_for_unicode() {
        let helper = build_ll_strcmp_helper_graph("ll_unicode_cmp", UNICODEPTR.clone())
            .expect("build_ll_strcmp_helper_graph for UNICODE");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        // start True → s1_nonnull True → both_nonnull True → loop_cond
        // True → loop_body.
        let s1_nn = pick_exit_target(&inner.startblock, true);
        let both_nn = pick_exit_target(&s1_nn, true);
        let loop_cond = pick_exit_target(&both_nn, true);
        let loop_body = pick_exit_target(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec![
                "getarrayitem",
                "getarrayitem",
                "cast_unichar_to_int",
                "cast_unichar_to_int",
                "int_sub",
                "int_ne",
                "int_add",
            ]
        );
    }

    /// The synthesised `ll_strcmp` graph carries the helper-identity
    /// `name` on both `FunctionGraph.name` and `func.name`, exposes
    /// the two `s1` / `s2` input slots via the startblock's
    /// inputargs, and returns `Signed` (since the diff / len-diff
    /// branches return raw integers and the NULL-pair branches feed
    /// `Signed(1)` / `Signed(0)` into the same slot).
    #[test]
    fn build_ll_strcmp_carries_name_two_inputs_and_signed_return() {
        let helper = build_ll_strcmp_helper_graph("ll_strcmp", STRPTR.clone()).unwrap();
        assert_eq!(helper.func.name, "ll_strcmp");
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "ll_strcmp");
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 2);
        for (i, expected) in ["s1", "s2"].iter().enumerate() {
            let Hlvalue::Variable(v) = &startblock.inputargs[i] else {
                panic!("startblock input {i} must be a Variable");
            };
            assert!(
                v.name().starts_with(expected),
                "input {i} variable name = {:?}, expected prefix {expected}",
                v.name()
            );
        }
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Signed),
            "ll_strcmp must return Signed"
        );
    }

    /// Passing a non-Ptr lltype should be rejected — propagated from
    /// the shared `chars_array_ptr_lltype_from_strptr` guard.
    #[test]
    fn build_ll_strcmp_rejects_non_ptr_input_lltype() {
        let err = build_ll_strcmp_helper_graph("ll_strcmp", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(STR/UNICODE)"));
    }

    /// `ll_startswith` synthesised against `Ptr(STR)` produces the
    /// expected 3-block-plus-returnblock CFG. start (getsubstruct ×2,
    /// getarraysize ×2, int_lt) → block_loop_cond (int_lt) →
    /// block_loop_body (getarrayitem ×2, char_eq, int_add). Loop body
    /// True branch cycles back to loop_cond.
    #[test]
    fn build_ll_startswith_synthesizes_length_check_then_loop_cfg_for_str() {
        let helper = build_ll_startswith_helper_graph("ll_startswith", STRPTR.clone())
            .expect("build_ll_startswith_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        // start: 4 chars/len ops + int_lt.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            start_ops,
            vec![
                "getsubstruct",
                "getarraysize",
                "getsubstruct",
                "getarraysize",
                "int_lt",
            ]
        );
        drop(startblock);

        // start False (len1 >= len2) → block_loop_cond (int_lt).
        let loop_cond = pick_exit_target(&inner.startblock, false);
        let loop_cond_borrow = loop_cond.borrow();
        let cond_ops: Vec<&str> = loop_cond_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(cond_ops, vec!["int_lt"]);
        drop(loop_cond_borrow);

        // block_loop_cond True → block_loop_body.
        let loop_body = pick_exit_target(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec!["getarrayitem", "getarrayitem", "char_eq", "int_add"]
        );
        drop(loop_body_borrow);

        // block_loop_body True (chars match) cycles back to loop_cond.
        let body_true_target = pick_exit_target(&loop_body, true);
        assert!(std::rc::Rc::ptr_eq(&body_true_target, &loop_cond));
    }

    /// `ll_startswith` against `Ptr(UNICODE)` differs only in the
    /// element comparison op — `unichar_eq` instead of `char_eq`.
    #[test]
    fn build_ll_startswith_uses_unichar_eq_op_for_unicode() {
        let helper = build_ll_startswith_helper_graph("ll_unicode_startswith", UNICODEPTR.clone())
            .expect("build_ll_startswith_helper_graph for UNICODE");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        // start False → loop_cond True → loop_body.
        let loop_cond = pick_exit_target(&inner.startblock, false);
        let loop_body = pick_exit_target(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec!["getarrayitem", "getarrayitem", "unichar_eq", "int_add"]
        );
    }

    /// The synthesised `ll_startswith` graph carries the helper-identity
    /// `name` on both `FunctionGraph.name` and `func.name`, exposes the
    /// two `s1` / `s2` input slots, and returns Bool. Mirrors the
    /// existing `build_ll_streq_carries_name_two_inputs_and_bool_return`
    /// covenant.
    #[test]
    fn build_ll_startswith_carries_name_two_inputs_and_bool_return() {
        let helper = build_ll_startswith_helper_graph("ll_startswith", STRPTR.clone()).unwrap();
        assert_eq!(helper.func.name, "ll_startswith");
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "ll_startswith");
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 2);
        for (i, expected) in ["s1", "s2"].iter().enumerate() {
            let Hlvalue::Variable(v) = &startblock.inputargs[i] else {
                panic!("startblock input {i} must be a Variable");
            };
            assert!(
                v.name().starts_with(expected),
                "input {i} variable name = {:?}, expected prefix {expected}",
                v.name()
            );
        }
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Bool),
            "ll_startswith must return Bool"
        );
    }

    /// Passing a non-Ptr lltype should be rejected — propagated from
    /// the shared `chars_array_ptr_lltype_from_strptr` guard.
    #[test]
    fn build_ll_startswith_rejects_non_ptr_input_lltype() {
        let err = build_ll_startswith_helper_graph("ll_startswith", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(STR/UNICODE)"));
    }

    /// `ll_endswith` synthesised against `Ptr(STR)` produces the
    /// expected 3-block-plus-returnblock CFG. start (getsubstruct ×2,
    /// getarraysize ×2, int_sub offset, int_lt) → block_loop_cond
    /// (int_lt) → block_loop_body (int_add idx, getarrayitem ×2,
    /// char_eq, int_add j_next). Loop body True branch cycles back to
    /// loop_cond.
    #[test]
    fn build_ll_endswith_synthesizes_offset_loop_cfg_for_str() {
        let helper = build_ll_endswith_helper_graph("ll_endswith", STRPTR.clone())
            .expect("build_ll_endswith_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            start_ops,
            vec![
                "getsubstruct",
                "getarraysize",
                "getsubstruct",
                "getarraysize",
                "int_sub",
                "int_lt",
            ]
        );
        drop(startblock);

        let loop_cond = pick_exit_target(&inner.startblock, false);
        let loop_cond_borrow = loop_cond.borrow();
        let cond_ops: Vec<&str> = loop_cond_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(cond_ops, vec!["int_lt"]);
        drop(loop_cond_borrow);

        let loop_body = pick_exit_target(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec![
                "int_add",
                "getarrayitem",
                "getarrayitem",
                "char_eq",
                "int_add",
            ]
        );
        drop(loop_body_borrow);

        // Loop body True branch (chars match) cycles back to loop_cond.
        let body_true_target = pick_exit_target(&loop_body, true);
        assert!(std::rc::Rc::ptr_eq(&body_true_target, &loop_cond));
    }

    /// `ll_endswith` against `Ptr(UNICODE)` differs only in the
    /// element comparison op — `unichar_eq` instead of `char_eq`.
    #[test]
    fn build_ll_endswith_uses_unichar_eq_op_for_unicode() {
        let helper = build_ll_endswith_helper_graph("ll_unicode_endswith", UNICODEPTR.clone())
            .expect("build_ll_endswith_helper_graph for UNICODE");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let loop_cond = pick_exit_target(&inner.startblock, false);
        let loop_body = pick_exit_target(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec![
                "int_add",
                "getarrayitem",
                "getarrayitem",
                "unichar_eq",
                "int_add",
            ]
        );
    }

    /// The synthesised `ll_endswith` graph carries the helper-identity
    /// `name`, exposes the two `s1` / `s2` input slots, and returns Bool.
    #[test]
    fn build_ll_endswith_carries_name_two_inputs_and_bool_return() {
        let helper = build_ll_endswith_helper_graph("ll_endswith", STRPTR.clone()).unwrap();
        assert_eq!(helper.func.name, "ll_endswith");
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "ll_endswith");
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 2);
        for (i, expected) in ["s1", "s2"].iter().enumerate() {
            let Hlvalue::Variable(v) = &startblock.inputargs[i] else {
                panic!("startblock input {i} must be a Variable");
            };
            assert!(
                v.name().starts_with(expected),
                "input {i} variable name = {:?}, expected prefix {expected}",
                v.name()
            );
        }
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Bool),
            "ll_endswith must return Bool"
        );
    }

    /// Passing a non-Ptr lltype should be rejected — propagated from
    /// the shared `chars_array_ptr_lltype_from_strptr` guard.
    #[test]
    fn build_ll_endswith_rejects_non_ptr_input_lltype() {
        let err = build_ll_endswith_helper_graph("ll_endswith", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(STR/UNICODE)"));
    }

    /// `ll_startswith_char` synthesised against `Ptr(STR)` produces
    /// the expected 2-block-plus-returnblock CFG. start
    /// (getsubstruct, getarraysize, int_eq) → block_compare
    /// (getarrayitem at index 0, char_eq).
    #[test]
    fn build_ll_startswith_char_synthesizes_empty_check_then_compare_cfg_for_str() {
        let helper = build_ll_startswith_char_helper_graph("ll_startswith_char", STRPTR.clone())
            .expect("build_ll_startswith_char_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getsubstruct", "getarraysize", "int_eq"]);
        drop(startblock);

        let compare = pick_exit_target(&inner.startblock, false);
        let compare_borrow = compare.borrow();
        let compare_ops: Vec<&str> = compare_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        // No int_sub — startswith reads chars[Constant(0)] directly.
        assert_eq!(compare_ops, vec!["getarrayitem", "char_eq"]);
    }

    /// `ll_endswith_char` against `Ptr(STR)` inserts an additional
    /// `int_sub(length, 1) -> idx` op before the `getarrayitem`.
    #[test]
    fn build_ll_endswith_char_synthesizes_int_sub_idx_for_str() {
        let helper = build_ll_endswith_char_helper_graph("ll_endswith_char", STRPTR.clone())
            .expect("build_ll_endswith_char_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let compare = pick_exit_target(&inner.startblock, false);
        let compare_borrow = compare.borrow();
        let compare_ops: Vec<&str> = compare_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(compare_ops, vec!["int_sub", "getarrayitem", "char_eq"]);
    }

    /// UNICODE variants substitute `unichar_eq` for the comparison op.
    #[test]
    fn build_ll_startswith_endswith_char_use_unichar_eq_op_for_unicode() {
        for (name, builder) in [
            (
                "ll_unicode_startswith_char",
                build_ll_startswith_char_helper_graph
                    as fn(&str, LowLevelType) -> Result<PyGraph, TyperError>,
            ),
            (
                "ll_unicode_endswith_char",
                build_ll_endswith_char_helper_graph
                    as fn(&str, LowLevelType) -> Result<PyGraph, TyperError>,
            ),
        ] {
            let helper = builder(name, UNICODEPTR.clone()).expect(name);
            let inner = helper.graph.borrow();
            let block_borrow = inner.startblock.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
                .expect("compare branch");
            let compare = link.borrow().target.as_ref().unwrap().clone();
            let compare_borrow = compare.borrow();
            let opnames: Vec<&str> = compare_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert!(
                opnames.iter().any(|&op| op == "unichar_eq"),
                "{name}: expected 'unichar_eq', got {opnames:?}"
            );
        }
    }

    /// Both helpers carry the helper-identity name + two inputargs
    /// (s + ch) and return Bool.
    #[test]
    fn build_ll_startswith_endswith_char_carry_name_two_inputs_and_bool_return() {
        for (name, builder) in [
            (
                "ll_startswith_char",
                build_ll_startswith_char_helper_graph
                    as fn(&str, LowLevelType) -> Result<PyGraph, TyperError>,
            ),
            (
                "ll_endswith_char",
                build_ll_endswith_char_helper_graph
                    as fn(&str, LowLevelType) -> Result<PyGraph, TyperError>,
            ),
        ] {
            let helper = builder(name, STRPTR.clone()).unwrap();
            assert_eq!(helper.func.name, name);
            let inner = helper.graph.borrow();
            assert_eq!(inner.name, name);
            let startblock = inner.startblock.borrow();
            assert_eq!(startblock.inputargs.len(), 2);
            let Hlvalue::Variable(v_s) = &startblock.inputargs[0] else {
                panic!("startblock input 0 must be Variable");
            };
            assert!(v_s.name().starts_with("s"));
            let Hlvalue::Variable(v_ch) = &startblock.inputargs[1] else {
                panic!("startblock input 1 must be Variable");
            };
            assert!(v_ch.name().starts_with("ch"));
            let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
                panic!("returnblock inputarg must be Variable");
            };
            assert_eq!(
                ret.concretetype.borrow().clone(),
                Some(LowLevelType::Bool),
                "{name}: must return Bool"
            );
        }
    }

    /// `_hash_string` synthesised against `Ptr(Array(Char))` produces
    /// the expected 5-block-plus-returnblock CFG: start (getarraysize +
    /// int_eq) → block_init (getarrayitem 0 + cast + int_lshift) →
    /// block_loop_cond (int_lt) → block_loop_body (int_mul +
    /// getarrayitem + cast + int_xor + int_add) → block_finalize
    /// (int_xor). Loop body always cycles back to loop_cond.
    #[test]
    fn build_hash_string_synthesizes_fnv_loop_cfg_for_char_array() {
        let chars_array_ptr = chars_array_ptr_lltype_from_strptr(&STRPTR.clone())
            .expect("chars_array_ptr_lltype_from_strptr");
        let helper = build_hash_string_helper_graph("_hash_string", chars_array_ptr)
            .expect("build_hash_string_helper_graph for STR chars Array");
        let inner = helper.graph.borrow();

        fn pick_exit_target_bool(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }
        fn first_exit_target(
            block: &crate::flowspace::model::BlockRef,
        ) -> crate::flowspace::model::BlockRef {
            let block_borrow = block.borrow();
            let link = block_borrow.exits.first().expect("at least one exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        // start: getarraysize + int_eq.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getarraysize", "int_eq"]);
        drop(startblock);

        // start False -> block_init.
        let init = pick_exit_target_bool(&inner.startblock, false);
        let init_borrow = init.borrow();
        let init_ops: Vec<&str> = init_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            init_ops,
            vec!["getarrayitem", "cast_char_to_int", "int_lshift"]
        );
        drop(init_borrow);

        // init -> block_loop_cond (single unconditional exit).
        let loop_cond = first_exit_target(&init);
        let loop_cond_borrow = loop_cond.borrow();
        let cond_ops: Vec<&str> = loop_cond_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(cond_ops, vec!["int_lt"]);
        drop(loop_cond_borrow);

        // loop_cond True -> block_loop_body.
        let loop_body = pick_exit_target_bool(&loop_cond, true);
        let loop_body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = loop_body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec![
                "int_mul",
                "getarrayitem",
                "cast_char_to_int",
                "int_xor",
                "int_add",
            ]
        );
        drop(loop_body_borrow);

        // loop_body cycles back to loop_cond.
        let body_target = first_exit_target(&loop_body);
        assert!(std::rc::Rc::ptr_eq(&body_target, &loop_cond));

        // loop_cond False -> block_finalize (int_xor).
        let finalize = pick_exit_target_bool(&loop_cond, false);
        let finalize_borrow = finalize.borrow();
        let finalize_ops: Vec<&str> = finalize_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(finalize_ops, vec!["int_xor"]);
    }

    /// UNICODE chars Array routes through `cast_unichar_to_int` for
    /// both `ord` sites (init's chars[0] and the loop body).
    #[test]
    fn build_hash_string_uses_cast_unichar_to_int_for_unicode_array() {
        let chars_array_ptr = chars_array_ptr_lltype_from_strptr(&UNICODEPTR.clone())
            .expect("chars_array_ptr_lltype_from_strptr UNICODE");
        let helper = build_hash_string_helper_graph("_hash_unicode_string", chars_array_ptr)
            .expect("build_hash_string_helper_graph UNICODE");
        let inner = helper.graph.borrow();

        let block_borrow = inner.startblock.borrow();
        let init_link = block_borrow
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("init branch");
        let init = init_link.borrow().target.as_ref().unwrap().clone();
        let init_borrow = init.borrow();
        let init_ops: Vec<&str> = init_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(init_ops[1], "cast_unichar_to_int");
    }

    /// Helper-identity covenant: name appears on FunctionGraph.name +
    /// func.name; chars input slot named `chars`; returns Signed.
    #[test]
    fn build_hash_string_carries_name_chars_input_and_signed_return() {
        let chars_array_ptr = chars_array_ptr_lltype_from_strptr(&STRPTR.clone())
            .expect("chars_array_ptr_lltype_from_strptr");
        let helper = build_hash_string_helper_graph("_hash_string", chars_array_ptr).unwrap();
        assert_eq!(helper.func.name, "_hash_string");
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "_hash_string");
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 1);
        let Hlvalue::Variable(v_chars) = &startblock.inputargs[0] else {
            panic!("startblock input must be Variable");
        };
        assert!(v_chars.name().starts_with("chars"));
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Signed),
            "_hash_string must return Signed"
        );
    }

    /// Non-Ptr / non-Array input must error at the boundary.
    #[test]
    fn build_hash_string_rejects_non_ptr_array_input_lltype() {
        let err = build_hash_string_helper_graph("_hash_string", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(Array"));
    }

    /// `ll_hash_string` synthesised against `Ptr(STR)` produces a
    /// single-block helper that lowers to `getsubstruct(s, 'chars') +
    /// direct_call(_hash_string, chars)`, then unconditionally links
    /// to returnblock with `x`. The `direct_call` first arg is a
    /// funcptr Constant (LLPtr) whose target is the inner
    /// `_hash_string` helper graph.
    #[test]
    fn build_ll_hash_string_emits_subhelper_direct_call_for_str() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_hash_string_helper_graph(
            &rtyper,
            "ll_hash_string",
            STRPTR.clone(),
            "_hash_string",
        )
        .expect("build_ll_hash_string_helper_graph for STR");
        let inner = helper.graph.borrow();

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getsubstruct", "direct_call"]);

        // direct_call's first arg must be a funcptr Constant (LLPtr).
        let direct_call_op = &startblock.operations[1];
        let Hlvalue::Constant(c_func) = &direct_call_op.args[0] else {
            panic!("direct_call arg[0] must be a Constant funcptr");
        };
        let dbg = format!("{:?}", c_func.value);
        assert!(dbg.contains("LLPtr"), "expected LLPtr funcptr, got {dbg}");
    }

    /// `_ll_strhash` synthesised against `Ptr(STR)` produces a
    /// 2-block-plus-returnblock CFG: start (getsubstruct + direct_call
    /// + int_eq) branches on `is_zero`; both branches converge on
    /// `block_set_hash` (setfield). The True branch carries the
    /// zero-fixup constant 29872897.
    #[test]
    fn build_ll_strhash_internal_emits_zero_fixup_then_setfield_for_str() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_strhash_internal_helper_graph(
            &rtyper,
            "_ll_strhash",
            STRPTR.clone(),
            "_hash_string",
        )
        .expect("build_ll_strhash_internal_helper_graph for STR");
        let inner = helper.graph.borrow();

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getsubstruct", "direct_call", "int_eq"]);

        // True branch (is_zero == true): link arg[1] is Signed(29872897).
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit");
        let link_borrow = true_link.borrow();
        let Some(Hlvalue::Constant(c_const)) = &link_borrow.args[1] else {
            panic!(
                "True-branch link arg[1] must be a Constant, got {:?}",
                link_borrow.args[1]
            );
        };
        assert!(
            matches!(c_const.value, ConstValue::Int(29872897)),
            "expected Signed(29872897), got {:?}",
            c_const.value
        );
        drop(link_borrow);

        // Both branches converge on block_set_hash.
        let true_target = true_link.borrow().target.as_ref().unwrap().clone();
        let false_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("False exit");
        let false_target = false_link.borrow().target.as_ref().unwrap().clone();
        assert!(
            std::rc::Rc::ptr_eq(&true_target, &false_target),
            "both branches must converge on the same block_set_hash"
        );
        drop(startblock);

        // block_set_hash: ops = setfield only.
        let set_hash_borrow = true_target.borrow();
        let set_hash_ops: Vec<&str> = set_hash_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(set_hash_ops, vec!["setfield"]);
    }

    /// `ll_stritem_nonneg` synthesised against `Ptr(STR)` produces a
    /// single-block CFG: getsubstruct + getarrayitem; link to
    /// returnblock with the char. Returns `Char` for STR, `UniChar`
    /// for UNICODE.
    #[test]
    fn build_ll_stritem_nonneg_synthesizes_single_block_cfg_for_str() {
        let helper = build_ll_stritem_nonneg_helper_graph("ll_stritem_nonneg", STRPTR.clone())
            .expect("build_ll_stritem_nonneg_helper_graph for STR");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getsubstruct", "getarrayitem"]);
        // Single unconditional exit to returnblock.
        assert_eq!(startblock.exits.len(), 1);
    }

    /// UNICODE variant returns `UniChar` (chars Array element type).
    #[test]
    fn build_ll_stritem_nonneg_returns_unichar_for_unicode() {
        let helper =
            build_ll_stritem_nonneg_helper_graph("ll_unicode_stritem_nonneg", UNICODEPTR.clone())
                .expect("build_ll_stritem_nonneg_helper_graph UNICODE");
        let inner = helper.graph.borrow();
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::UniChar),
        );
    }

    /// `ll_stritem` synthesised against `Ptr(STR)` produces a
    /// 3-block-plus-returnblock CFG: start (int_lt(i, 0)) branches on
    /// `is_neg`. True → block_neg_fix (getsubstruct + getarraysize +
    /// int_add(i, length)). False → block_dispatch (direct_call
    /// ll_stritem_nonneg). block_neg_fix unconditionally links to
    /// block_dispatch with i_fix.
    #[test]
    fn build_ll_stritem_synthesizes_neg_fix_then_direct_call_for_str() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_stritem_helper_graph(
            &rtyper,
            "ll_stritem",
            STRPTR.clone(),
            "ll_stritem_nonneg",
        )
        .expect("build_ll_stritem_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["int_lt"]);
        drop(startblock);

        let neg_fix = pick_exit_target(&inner.startblock, true);
        {
            let nf_borrow = neg_fix.borrow();
            let nf_ops: Vec<&str> = nf_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(nf_ops, vec!["getsubstruct", "getarraysize", "int_add"]);
        }

        // Both start branches converge on block_dispatch.
        let dispatch_via_false = pick_exit_target(&inner.startblock, false);
        let dispatch_via_neg = {
            // neg_fix has a single unconditional exit.
            let nf_borrow = neg_fix.borrow();
            let link = nf_borrow.exits.first().expect("neg_fix exit").clone();
            link.borrow().target.as_ref().unwrap().clone()
        };
        assert!(std::rc::Rc::ptr_eq(&dispatch_via_false, &dispatch_via_neg));

        // dispatch: direct_call ll_stritem_nonneg.
        let dispatch_borrow = dispatch_via_false.borrow();
        let dispatch_ops: Vec<&str> = dispatch_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(dispatch_ops, vec!["direct_call"]);
        let Hlvalue::Constant(c_func) = &dispatch_borrow.operations[0].args[0] else {
            panic!("direct_call arg[0] must be Constant funcptr");
        };
        assert!(format!("{:?}", c_func.value).contains("LLPtr"));
    }

    /// Helper-identity covenant: name + (s, i) input slots; returns
    /// Char for STR.
    #[test]
    fn build_ll_stritem_nonneg_carries_name_inputs_and_char_return_for_str() {
        let helper =
            build_ll_stritem_nonneg_helper_graph("ll_stritem_nonneg", STRPTR.clone()).unwrap();
        assert_eq!(helper.func.name, "ll_stritem_nonneg");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 2);
        let Hlvalue::Variable(v_s) = &startblock.inputargs[0] else {
            panic!("input 0 must be Variable");
        };
        assert!(v_s.name().starts_with("s"));
        let Hlvalue::Variable(v_i) = &startblock.inputargs[1] else {
            panic!("input 1 must be Variable");
        };
        assert!(v_i.name().starts_with("i"));
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Char));
    }

    /// `ll_stritem_nonneg_checked` synthesises a 2-block-plus-return-
    /// plus-exceptblock CFG: start emits getsubstruct + getarraysize +
    /// int_ge(i, length), with the True branch closing a Link to
    /// graph.exceptblock (raise IndexError) and the False branch
    /// forwarding to block_dispatch which direct_calls
    /// ll_stritem_nonneg.
    #[test]
    fn build_ll_stritem_nonneg_checked_synthesizes_oob_branch_and_dispatch_for_str() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_stritem_nonneg_checked_helper_graph(
            &rtyper,
            "ll_stritem_nonneg_checked",
            STRPTR.clone(),
            "ll_stritem_nonneg",
        )
        .expect("build_ll_stritem_nonneg_checked_helper_graph for STR");
        let inner = helper.graph.borrow();

        // start: getsubstruct + getarraysize + int_ge(i, length).
        {
            let sb = inner.startblock.borrow();
            let ops: Vec<&str> = sb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(
                ops,
                vec!["getsubstruct", "getarraysize", "int_ge"],
                "start emits chars + length + bound check"
            );
        }

        // start.exits[True] should target graph.exceptblock (raise).
        // start.exits[False] should target block_dispatch (direct_call).
        let start_borrow = inner.startblock.borrow();
        let mut saw_raise = false;
        let mut dispatch_block: Option<crate::flowspace::model::BlockRef> = None;
        for link in &start_borrow.exits {
            let lb = link.borrow();
            let target = lb.target.as_ref().expect("link.target");
            let exitcase_is_true = matches!(
                lb.exitcase,
                Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
            );
            if std::rc::Rc::ptr_eq(target, &inner.exceptblock) {
                assert!(
                    exitcase_is_true,
                    "raise edge must be the True branch of int_ge(oob)"
                );
                saw_raise = true;
            } else if !exitcase_is_true {
                dispatch_block = Some(target.clone());
            }
        }
        drop(start_borrow);
        assert!(saw_raise, "missing raise edge to exceptblock");
        let dispatch = dispatch_block.expect("missing False edge to block_dispatch");

        // block_dispatch: direct_call(ll_stritem_nonneg, s, i).
        let db = dispatch.borrow();
        let ops: Vec<&str> = db.operations.iter().map(|op| op.opname.as_str()).collect();
        assert_eq!(ops, vec!["direct_call"]);
        let Hlvalue::Constant(c_func) = &db.operations[0].args[0] else {
            panic!("direct_call arg[0] must be Constant funcptr");
        };
        assert!(format!("{:?}", c_func.value).contains("LLPtr"));
    }

    /// `ll_stritem_checked` synthesises a 5-block-plus-return-plus-
    /// exceptblock CFG: start branches on int_lt(i, 0) into block_neg_fix
    /// (which int_adds length and forwards to block_check_high) or
    /// directly into block_check_high. block_check_high (int_ge) raises
    /// or forwards to block_check_low. block_check_low (int_lt) raises
    /// or forwards to block_dispatch (direct_call). Two raise edges
    /// reach graph.exceptblock — mirroring the upstream
    /// `if i >= length or i < 0` short-circuit.
    #[test]
    fn build_ll_stritem_checked_synthesizes_neg_fix_two_bound_checks_for_str() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_stritem_checked_helper_graph(
            &rtyper,
            "ll_stritem_checked",
            STRPTR.clone(),
            "ll_stritem_nonneg",
        )
        .expect("build_ll_stritem_checked_helper_graph for STR");
        let inner = helper.graph.borrow();

        // start: getsubstruct + getarraysize + int_lt(i, 0).
        {
            let sb = inner.startblock.borrow();
            let ops: Vec<&str> = sb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(ops, vec!["getsubstruct", "getarraysize", "int_lt"]);
        }

        // start has 2 exits: True → block_neg_fix; False → block_check_high.
        let (block_neg_fix, block_check_high_via_false) = {
            let sb = inner.startblock.borrow();
            assert_eq!(sb.exits.len(), 2);
            let mut neg_fix: Option<crate::flowspace::model::BlockRef> = None;
            let mut check_high: Option<crate::flowspace::model::BlockRef> = None;
            for link in &sb.exits {
                let lb = link.borrow();
                let exitcase_is_true = matches!(
                    lb.exitcase,
                    Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                );
                let target = lb.target.as_ref().unwrap().clone();
                if exitcase_is_true {
                    neg_fix = Some(target);
                } else {
                    check_high = Some(target);
                }
            }
            (neg_fix.unwrap(), check_high.unwrap())
        };

        // block_neg_fix: int_add only; unconditional exit to block_check_high.
        {
            let nb = block_neg_fix.borrow();
            let ops: Vec<&str> = nb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(ops, vec!["int_add"]);
            assert_eq!(nb.exits.len(), 1);
        }

        // Both starts converge on the same block_check_high.
        let block_check_high_via_neg = {
            let nb = block_neg_fix.borrow();
            nb.exits[0].borrow().target.as_ref().unwrap().clone()
        };
        assert!(std::rc::Rc::ptr_eq(
            &block_check_high_via_false,
            &block_check_high_via_neg,
        ));

        // block_check_high: int_ge; True → exceptblock, False → block_check_low.
        let block_check_low = {
            let cb = block_check_high_via_false.borrow();
            let ops: Vec<&str> = cb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(ops, vec!["int_ge"]);
            assert_eq!(cb.exits.len(), 2);
            let mut next: Option<crate::flowspace::model::BlockRef> = None;
            let mut saw_raise = false;
            for link in &cb.exits {
                let lb = link.borrow();
                let exitcase_is_true = matches!(
                    lb.exitcase,
                    Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                );
                let target = lb.target.as_ref().unwrap().clone();
                if exitcase_is_true {
                    assert!(
                        std::rc::Rc::ptr_eq(&target, &inner.exceptblock),
                        "block_check_high True branch must raise"
                    );
                    saw_raise = true;
                } else {
                    next = Some(target);
                }
            }
            assert!(saw_raise);
            next.unwrap()
        };

        // block_check_low: int_lt(i, 0); True → exceptblock, False → block_dispatch.
        let block_dispatch = {
            let cb = block_check_low.borrow();
            let ops: Vec<&str> = cb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(ops, vec!["int_lt"]);
            assert_eq!(cb.exits.len(), 2);
            let mut next: Option<crate::flowspace::model::BlockRef> = None;
            let mut saw_raise = false;
            for link in &cb.exits {
                let lb = link.borrow();
                let exitcase_is_true = matches!(
                    lb.exitcase,
                    Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                );
                let target = lb.target.as_ref().unwrap().clone();
                if exitcase_is_true {
                    assert!(
                        std::rc::Rc::ptr_eq(&target, &inner.exceptblock),
                        "block_check_low True branch must raise"
                    );
                    saw_raise = true;
                } else {
                    next = Some(target);
                }
            }
            assert!(saw_raise);
            next.unwrap()
        };

        // block_dispatch: direct_call(ll_stritem_nonneg, s, i).
        {
            let db = block_dispatch.borrow();
            let ops: Vec<&str> = db.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(ops, vec!["direct_call"]);
        }
    }

    /// `ll_strconcat` synthesises a 5-block-plus-returnblock CFG:
    /// start emits getsubstruct + getarraysize × 2 + int_add (total)
    /// + **malloc_varsize** + getsubstruct (newchars), then enters
    /// block_copy1_cond which loops int_lt + body (getarrayitem +
    /// setarrayitem + int_add); on cond=False forwards into
    /// block_copy2_cond with j=0 which loops int_lt + body
    /// (getarrayitem + int_add(len1, j) for the destination index +
    /// setarrayitem + int_add). False branch of copy2_cond returns
    /// newstr.
    ///
    /// First emission of `malloc_varsize` SpaceOperation from inside
    /// an rstr helper graph: arg shape is `[Constant(struct_lltype),
    /// Constant({'flavor': 'gc'}), length_var]` matching upstream
    /// rstr.py:1131.
    #[test]
    fn build_ll_strconcat_synthesizes_malloc_varsize_and_two_copy_loops_for_str() {
        let helper = build_ll_strconcat_helper_graph("ll_strconcat", STRPTR.clone())
            .expect("build_ll_strconcat_helper_graph for STR");
        let inner = helper.graph.borrow();

        // start: 6 ops (getsubstruct, getarraysize, getsubstruct,
        // getarraysize, int_add, malloc_varsize, getsubstruct).
        let start_ops = {
            let sb = inner.startblock.borrow();
            let names: Vec<String> = sb.operations.iter().map(|op| op.opname.clone()).collect();
            names
        };
        assert_eq!(
            start_ops,
            vec![
                "getsubstruct",
                "getarraysize",
                "getsubstruct",
                "getarraysize",
                "int_add",
                "malloc_varsize",
                "getsubstruct",
            ],
            "start emits chars1/len1/chars2/len2/total + malloc_varsize + newchars"
        );

        // malloc_varsize op (5th op, 0-indexed) takes 3 args:
        // [Constant(LowLevelType::Struct), Constant(Dict), Variable(total)].
        let mv_op = {
            let sb = inner.startblock.borrow();
            sb.operations[5].clone()
        };
        assert_eq!(mv_op.opname, "malloc_varsize");
        assert_eq!(mv_op.args.len(), 3);
        let Hlvalue::Constant(struct_const) = &mv_op.args[0] else {
            panic!("malloc_varsize arg[0] must be Constant carrying struct lltype");
        };
        assert!(matches!(struct_const.value, ConstValue::LowLevelType(_)));
        let Hlvalue::Constant(flavor_const) = &mv_op.args[1] else {
            panic!("malloc_varsize arg[1] must be Constant carrying flags dict");
        };
        assert!(matches!(flavor_const.value, ConstValue::Dict(_)));
        assert!(matches!(mv_op.args[2], Hlvalue::Variable(_)));

        // start has a single unconditional exit into block_copy1_cond
        // (carrying newstr/newchars/chars1/chars2/len1/len2/0).
        let block_copy1_cond = {
            let sb = inner.startblock.borrow();
            assert_eq!(sb.exits.len(), 1, "start has a single unconditional exit");
            let link = sb.exits[0].clone();
            link.borrow().target.as_ref().unwrap().clone()
        };

        // block_copy1_cond: int_lt(i, len1); 2 exits.
        let (block_copy1_body, block_copy2_cond) = {
            let cb = block_copy1_cond.borrow();
            let names: Vec<&str> = cb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(names, vec!["int_lt"]);
            assert_eq!(cb.exits.len(), 2);
            let mut body: Option<crate::flowspace::model::BlockRef> = None;
            let mut next: Option<crate::flowspace::model::BlockRef> = None;
            for link in &cb.exits {
                let lb = link.borrow();
                let exitcase_is_true = matches!(
                    lb.exitcase,
                    Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                );
                let target = lb.target.as_ref().unwrap().clone();
                if exitcase_is_true {
                    body = Some(target);
                } else {
                    next = Some(target);
                }
            }
            (body.unwrap(), next.unwrap())
        };

        // block_copy1_body: getarrayitem + setarrayitem + int_add;
        // unconditional cycle back into block_copy1_cond.
        {
            let bb = block_copy1_body.borrow();
            let names: Vec<&str> = bb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(names, vec!["getarrayitem", "setarrayitem", "int_add"]);
            assert_eq!(bb.exits.len(), 1);
            let cycle_target = bb.exits[0].borrow().target.as_ref().unwrap().clone();
            assert!(
                std::rc::Rc::ptr_eq(&cycle_target, &block_copy1_cond),
                "copy1_body must cycle back into copy1_cond"
            );
        }

        // block_copy2_cond: int_lt(j, len2); 2 exits, False → returnblock.
        let block_copy2_body = {
            let cb = block_copy2_cond.borrow();
            let names: Vec<&str> = cb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(names, vec!["int_lt"]);
            assert_eq!(cb.exits.len(), 2);
            let mut body: Option<crate::flowspace::model::BlockRef> = None;
            let mut return_target_seen = false;
            for link in &cb.exits {
                let lb = link.borrow();
                let exitcase_is_true = matches!(
                    lb.exitcase,
                    Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                );
                let target = lb.target.as_ref().unwrap().clone();
                if exitcase_is_true {
                    body = Some(target);
                } else {
                    assert!(
                        std::rc::Rc::ptr_eq(&target, &inner.returnblock),
                        "copy2_cond False branch must return"
                    );
                    return_target_seen = true;
                }
            }
            assert!(return_target_seen);
            body.unwrap()
        };

        // block_copy2_body: getarrayitem + int_add(len1, j) for dst_idx
        // + setarrayitem + int_add(j, 1); cycle back into copy2_cond.
        {
            let bb = block_copy2_body.borrow();
            let names: Vec<&str> = bb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(
                names,
                vec!["getarrayitem", "int_add", "setarrayitem", "int_add"]
            );
            assert_eq!(bb.exits.len(), 1);
            let cycle_target = bb.exits[0].borrow().target.as_ref().unwrap().clone();
            assert!(std::rc::Rc::ptr_eq(&cycle_target, &block_copy2_cond));
        }
    }

    /// `ll_unicode_concat` synthesised against `Ptr(UNICODE)` produces
    /// the same CFG shape as `ll_strconcat` modulo elem type. Locks
    /// in that the malloc_varsize struct const for UNICODE differs
    /// from the STR one (otherwise the helper-cache families would
    /// alias).
    #[test]
    fn build_ll_strconcat_synthesizes_unicode_struct_const_in_malloc_varsize() {
        let str_helper =
            build_ll_strconcat_helper_graph("ll_strconcat", STRPTR.clone()).expect("STR strconcat");
        let uni_helper = build_ll_strconcat_helper_graph("ll_unicode_concat", UNICODEPTR.clone())
            .expect("UNICODE strconcat");
        let str_inner = str_helper.graph.borrow();
        let uni_inner = uni_helper.graph.borrow();

        let str_struct_const = {
            let sb = str_inner.startblock.borrow();
            let mv = &sb.operations[5];
            assert_eq!(mv.opname, "malloc_varsize");
            mv.args[0].clone()
        };
        let uni_struct_const = {
            let sb = uni_inner.startblock.borrow();
            let mv = &sb.operations[5];
            assert_eq!(mv.opname, "malloc_varsize");
            mv.args[0].clone()
        };
        assert!(
            format!("{str_struct_const:?}") != format!("{uni_struct_const:?}"),
            "STR and UNICODE malloc_varsize struct constants must differ"
        );
    }

    /// `ll_count_char` synthesised against `Ptr(STR)` produces a
    /// 3-block-plus-returnblock CFG: same end-clamp + loop pattern
    /// as `ll_find_char` plus a count accumulator. body emits both
    /// `int_add(i, 1)` and `int_add(count, 1)`; the True branch
    /// carries `count_inc` and the False branch carries the original
    /// `count` back into loop_cond.
    #[test]
    fn build_ll_count_char_synthesizes_count_accumulator_loop_cfg_for_str() {
        let helper = build_ll_count_char_helper_graph("ll_count_char", STRPTR.clone())
            .expect("build_ll_count_char_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getsubstruct", "getarraysize", "int_gt"]);
        drop(startblock);

        let loop_cond = pick_exit_target(&inner.startblock, false);
        {
            let lc_borrow = loop_cond.borrow();
            let cond_ops: Vec<&str> = lc_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(cond_ops, vec!["int_lt"]);
        }

        let loop_body = pick_exit_target(&loop_cond, true);
        {
            let lb_borrow = loop_body.borrow();
            let body_ops: Vec<&str> = lb_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(
                body_ops,
                vec!["getarrayitem", "char_eq", "int_add", "int_add"]
            );
        }

        // body True (match) cycles back to loop_cond.
        let body_true = pick_exit_target(&loop_body, true);
        assert!(std::rc::Rc::ptr_eq(&body_true, &loop_cond));
        // body False (no match) also cycles back.
        let body_false = pick_exit_target(&loop_body, false);
        assert!(std::rc::Rc::ptr_eq(&body_false, &loop_cond));
    }

    /// UNICODE variant uses `unichar_eq`.
    #[test]
    fn build_ll_count_char_uses_unichar_eq_for_unicode() {
        let helper = build_ll_count_char_helper_graph("ll_unicode_count_char", UNICODEPTR.clone())
            .expect("build_ll_count_char_helper_graph UNICODE");
        let inner = helper.graph.borrow();
        let loop_cond = {
            let sb_borrow = inner.startblock.borrow();
            let link = sb_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
                .expect("loop_cond branch")
                .clone();
            link.borrow().target.as_ref().unwrap().clone()
        };
        let loop_body = {
            let lc_borrow = loop_cond.borrow();
            let link = lc_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
                .expect("loop_body branch")
                .clone();
            link.borrow().target.as_ref().unwrap().clone()
        };
        let lb_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = lb_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert!(body_ops.iter().any(|&op| op == "unichar_eq"));
    }

    /// `ll_strhash` synthesised against `Ptr(STR)` produces a
    /// 2-block-plus-returnblock CFG. start (ptr_nonzero) branches on
    /// `nz`: True → block_lookup, False → returnblock(0). block_lookup
    /// emits `getfield + jit_conditional_call_value`.
    #[test]
    fn build_ll_strhash_emits_null_guard_then_conditional_call_value_for_str() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_strhash_helper_graph(
            &rtyper,
            "ll_strhash",
            STRPTR.clone(),
            "_ll_strhash",
            "_hash_string",
        )
        .expect("build_ll_strhash_helper_graph for STR");
        let inner = helper.graph.borrow();

        // start: ptr_nonzero.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["ptr_nonzero"]);

        // False branch -> returnblock with Signed(0).
        let false_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("False exit");
        let link_borrow = false_link.borrow();
        let Some(Hlvalue::Constant(c_zero)) = &link_borrow.args[0] else {
            panic!("False-branch link arg[0] must be Constant");
        };
        assert!(matches!(c_zero.value, ConstValue::Int(0)));
        drop(link_borrow);

        // True branch -> block_lookup with getfield + jit_conditional_call_value.
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit");
        let lookup = true_link.borrow().target.as_ref().unwrap().clone();
        drop(startblock);
        let lookup_borrow = lookup.borrow();
        let lookup_ops: Vec<&str> = lookup_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(lookup_ops, vec!["getfield", "jit_conditional_call_value"]);

        // jit_conditional_call_value has 3 args: cond (Variable h),
        // funcptr Constant, s (Variable).
        let cond_op = &lookup_borrow.operations[1];
        assert_eq!(cond_op.args.len(), 3);
        let Hlvalue::Constant(c_func) = &cond_op.args[1] else {
            panic!("jit_conditional_call_value arg[1] must be funcptr Constant");
        };
        assert!(format!("{:?}", c_func.value).contains("LLPtr"));
    }

    /// `ll_find_char` synthesised against `Ptr(STR)` produces the
    /// expected 3-block-plus-returnblock CFG. start (getsubstruct +
    /// getarraysize + int_gt) → block_loop_cond (int_lt) →
    /// block_loop_body (getarrayitem + char_eq + int_add). Body True
    /// branch returns i; False cycles back to loop_cond.
    #[test]
    fn build_ll_find_char_synthesizes_forward_loop_cfg_for_str() {
        let helper = build_ll_find_char_helper_graph("ll_find_char", STRPTR.clone())
            .expect("build_ll_find_char_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getsubstruct", "getarraysize", "int_gt"]);
        drop(startblock);

        // start False branch: end_clamped == end (original).
        let loop_cond_via_false = pick_exit_target(&inner.startblock, false);
        {
            let lc_borrow = loop_cond_via_false.borrow();
            let cond_ops: Vec<&str> = lc_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(cond_ops, vec!["int_lt"]);
        }

        // start True branch also goes to loop_cond (different bound init).
        let loop_cond_via_true = pick_exit_target(&inner.startblock, true);
        assert!(std::rc::Rc::ptr_eq(
            &loop_cond_via_false,
            &loop_cond_via_true
        ));

        let loop_body = pick_exit_target(&loop_cond_via_false, true);
        {
            let lb_borrow = loop_body.borrow();
            let body_ops: Vec<&str> = lb_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(body_ops, vec!["getarrayitem", "char_eq", "int_add"]);
        }

        // body False (no match) cycles back to loop_cond.
        let body_false = pick_exit_target(&loop_body, false);
        assert!(std::rc::Rc::ptr_eq(&body_false, &loop_cond_via_false));
    }

    /// `ll_rfind_char` against `Ptr(STR)` reverses the loop direction:
    /// loop_cond uses `int_gt(i, start_bound)` instead of `int_lt`,
    /// and loop_body pre-decrements i via `int_sub` before reading
    /// `chars[i_dec]`.
    #[test]
    fn build_ll_rfind_char_synthesizes_reverse_loop_cfg_for_str() {
        let helper = build_ll_rfind_char_helper_graph("ll_rfind_char", STRPTR.clone())
            .expect("build_ll_rfind_char_helper_graph for STR");
        let inner = helper.graph.borrow();

        fn pick_exit_target(
            block: &crate::flowspace::model::BlockRef,
            want_true: bool,
        ) -> crate::flowspace::model::BlockRef {
            let target_value = ConstValue::Bool(want_true);
            let block_borrow = block.borrow();
            let link = block_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == target_value))
                .expect("matching exit");
            link.borrow().target.as_ref().unwrap().clone()
        }

        let loop_cond = pick_exit_target(&inner.startblock, false);
        {
            let lc_borrow = loop_cond.borrow();
            let cond_ops: Vec<&str> = lc_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(cond_ops, vec!["int_gt"]);
        }

        let loop_body = pick_exit_target(&loop_cond, true);
        {
            let lb_borrow = loop_body.borrow();
            let body_ops: Vec<&str> = lb_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert_eq!(body_ops, vec!["int_sub", "getarrayitem", "char_eq"]);
        }
    }

    /// UNICODE variants substitute `unichar_eq` for the comparison op
    /// in both find/rfind body blocks.
    #[test]
    fn build_ll_find_rfind_char_use_unichar_eq_for_unicode() {
        for (name, builder) in [
            (
                "ll_unicode_find_char",
                build_ll_find_char_helper_graph
                    as fn(&str, LowLevelType) -> Result<PyGraph, TyperError>,
            ),
            (
                "ll_unicode_rfind_char",
                build_ll_rfind_char_helper_graph
                    as fn(&str, LowLevelType) -> Result<PyGraph, TyperError>,
            ),
        ] {
            let helper = builder(name, UNICODEPTR.clone()).expect(name);
            let inner = helper.graph.borrow();
            let loop_cond = {
                let sb_borrow = inner.startblock.borrow();
                let link = sb_borrow
                    .exits
                    .iter()
                    .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
                    .expect("loop_cond branch")
                    .clone();
                let target = link.borrow().target.as_ref().unwrap().clone();
                target
            };
            let loop_body = {
                let lc_borrow = loop_cond.borrow();
                let link = lc_borrow
                    .exits
                    .iter()
                    .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
                    .expect("loop_body branch")
                    .clone();
                let target = link.borrow().target.as_ref().unwrap().clone();
                target
            };
            let lb_borrow = loop_body.borrow();
            let body_ops: Vec<&str> = lb_borrow
                .operations
                .iter()
                .map(|op| op.opname.as_str())
                .collect();
            assert!(
                body_ops.iter().any(|&op| op == "unichar_eq"),
                "{name}: expected 'unichar_eq', got {body_ops:?}"
            );
        }
    }

    /// UNICODE variants route the inner `_hash_string` sub-helper
    /// against the Unicode chars Array. The inner helper's body uses
    /// `cast_unichar_to_int` (verified indirectly via the inner
    /// helper's helper-cache key — distinct lltype implies distinct
    /// graph).
    #[test]
    fn build_ll_hash_string_routes_through_unicode_inner_helper() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_hash_string_helper_graph(
            &rtyper,
            "ll_unicode_hash_string",
            UNICODEPTR.clone(),
            "_hash_unicode_string",
        )
        .expect("build_ll_hash_string_helper_graph for UNICODE");
        let inner = helper.graph.borrow();

        // direct_call[0] funcptr Constant — debug-format must mention LLPtr.
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.operations[1].opname, "direct_call");
        let Hlvalue::Constant(c_func) = &startblock.operations[1].args[0] else {
            panic!("direct_call arg[0] must be Constant funcptr");
        };
        assert!(format!("{:?}", c_func.value).contains("LLPtr"));
    }

    fn build_identity_char_helper_graph(name: &str) -> Result<PyGraph, TyperError> {
        let ch = variable_with_lltype("ch", LowLevelType::Char);
        let startblock = Block::shared(vec![Hlvalue::Variable(ch.clone())]);
        let return_var = variable_with_lltype("result", LowLevelType::Char);
        let mut graph = FunctionGraph::with_return_var(
            name.to_string(),
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(ch)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let func = GraphFunc::new(
            name.to_string(),
            Constant::new(ConstValue::Dict(Default::default())),
        );
        graph.func = Some(func.clone());
        Ok(helper_pygraph_from_graph(
            graph,
            vec!["ch".to_string()],
            func,
        ))
    }

    /// `ll_upper/ll_lower` graph shape: extract chars/length, keep the
    /// explicit empty branch, allocate result with malloc_varsize, loop
    /// over source chars, call the per-char helper, and store into the
    /// result chars array.
    #[test]
    fn build_ll_string_casefold_synthesizes_malloc_loop_and_char_helper_call() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let helper = build_ll_string_casefold_helper_graph(
            &rtyper,
            "ll_upper",
            STRPTR.clone(),
            "ll_upper_char",
            build_identity_char_helper_graph,
        )
        .expect("build ll_upper");
        let inner = helper.graph.borrow();

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getsubstruct", "getarraysize", "int_eq"]);

        let alloc = {
            let link = startblock
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
                .expect("non-empty allocation branch")
                .clone();
            link.borrow().target.as_ref().unwrap().clone()
        };
        drop(startblock);

        let alloc_borrow = alloc.borrow();
        let alloc_ops: Vec<&str> = alloc_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(alloc_ops, vec!["malloc_varsize", "getsubstruct"]);
        let loop_cond = alloc_borrow.exits[0]
            .borrow()
            .target
            .as_ref()
            .unwrap()
            .clone();
        drop(alloc_borrow);

        let cond_borrow = loop_cond.borrow();
        let body = {
            let link = cond_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
                .expect("loop body branch")
                .clone();
            link.borrow().target.as_ref().unwrap().clone()
        };
        drop(cond_borrow);

        let body_borrow = body.borrow();
        let body_ops: Vec<&str> = body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            body_ops,
            vec!["getarrayitem", "direct_call", "setarrayitem", "int_add"]
        );
        let Hlvalue::Constant(c_func) = &body_borrow.operations[1].args[0] else {
            panic!("direct_call arg[0] must be Constant funcptr");
        };
        assert!(format!("{:?}", c_func.value).contains("LLPtr"));
    }

    /// Unicode must not accidentally share the byte-string casefold
    /// helper. PyPy's `ll_upper/lower` use `mallocstr` specifically to
    /// explode for unicode; the Rust builder rejects UniChar arrays.
    #[test]
    fn build_ll_string_casefold_rejects_unicode_ptr() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let err = build_ll_string_casefold_helper_graph(
            &rtyper,
            "ll_unicode_upper",
            UNICODEPTR.clone(),
            "ll_upper_char",
            build_identity_char_helper_graph,
        )
        .expect_err("unicode casefold helper must be rejected");
        assert!(format!("{err:?}").contains("only supports Char arrays"));
    }

    /// `ll_replace_chr_chr` emits the upstream malloc + replacement
    /// loop shape, with an equality branch before storing to the new
    /// string.
    #[test]
    fn build_ll_replace_chr_chr_synthesizes_malloc_replace_loop_for_str() {
        let helper = build_ll_replace_chr_chr_helper_graph(
            "ll_replace_chr_chr",
            STRPTR.clone(),
            LowLevelType::Char,
        )
        .expect("build ll_replace_chr_chr");
        let inner = helper.graph.borrow();

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            start_ops,
            vec![
                "getsubstruct",
                "getarraysize",
                "malloc_varsize",
                "getsubstruct"
            ]
        );
        let loop_cond = startblock.exits[0]
            .borrow()
            .target
            .as_ref()
            .unwrap()
            .clone();
        drop(startblock);

        let cond_borrow = loop_cond.borrow();
        assert_eq!(cond_borrow.operations[0].opname, "int_lt");
        let loop_body = {
            let link = cond_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
                .expect("loop body branch")
                .clone();
            link.borrow().target.as_ref().unwrap().clone()
        };
        drop(cond_borrow);

        let body_borrow = loop_body.borrow();
        let body_ops: Vec<&str> = body_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(body_ops, vec!["getarrayitem", "char_eq"]);
        let store = body_borrow.exits[0]
            .borrow()
            .target
            .as_ref()
            .unwrap()
            .clone();
        drop(body_borrow);

        let store_borrow = store.borrow();
        let store_ops: Vec<&str> = store_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(store_ops, vec!["setarrayitem", "int_add"]);
    }

    /// Unicode uses the same LLHelpers body but must compare UniChar
    /// elements with `unichar_eq`.
    #[test]
    fn build_ll_replace_chr_chr_uses_unichar_eq_for_unicode() {
        let helper = build_ll_replace_chr_chr_helper_graph(
            "ll_unicode_replace_chr_chr",
            UNICODEPTR.clone(),
            LowLevelType::UniChar,
        )
        .expect("build unicode replace");
        let inner = helper.graph.borrow();
        let loop_cond = inner.startblock.borrow().exits[0]
            .borrow()
            .target
            .as_ref()
            .unwrap()
            .clone();
        let loop_body = {
            let cond_borrow = loop_cond.borrow();
            let link = cond_borrow
                .exits
                .iter()
                .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
                .expect("loop body branch")
                .clone();
            link.borrow().target.as_ref().unwrap().clone()
        };
        let body_borrow = loop_body.borrow();
        assert_eq!(body_borrow.operations[1].opname, "unichar_eq");
    }
}
