//! RPython `rpython/rtyper/rstr.py` — Char / UniChar Repr (minimal slice).
//!
//! Upstream rstr.py is ~1500 LOC covering `AbstractCharRepr`,
//! `AbstractUniCharRepr`, `AbstractStringRepr`, `AbstractUnicodeRepr`,
//! plus their lltypesystem / ootypesystem realisations and the dense
//! pairtype dispatch surface (eq/ne/lt/le/gt/ge/add/mul/contains/in
//! /str/repr/encode/decode/...). Pyre lands the **minimal slice**
//! required to unblock tuple eq/hash with Char/UniChar items:
//!
//! | upstream | Rust mirror |
//! |---|---|
//! | `class AbstractCharRepr` (`rstr.py:483-541`) | [`CharRepr`] |
//! | `class AbstractUniCharRepr` (`rstr.py:758-775`) | [`UniCharRepr`] |
//! | `AbstractCharRepr.get_ll_eq_function` (`rstr.py:496-497`) | [`Repr::get_ll_eq_function`] impl |
//! | `AbstractCharRepr.get_ll_hash_function` (`rstr.py:499-500`) | [`Repr::get_ll_hash_function`] impl + [`build_ll_char_hash_helper_graph`] |
//! | `AbstractUniCharRepr.get_ll_hash_function` (`rstr.py:767-768`) | [`Repr::get_ll_hash_function`] impl + [`build_ll_unichar_hash_helper_graph`] |
//! | `ll_char_hash` / `ll_unichar_hash` (`rstr.py:937-942`) | helper graph: `cast_char_to_int(ch)` / `cast_unichar_to_int(ch)` |
//! | `SomeChar.rtyper_makerepr` / `SomeUnicodeCodePoint.rtyper_makerepr` (`rstr.py:589-598`) | wired in [`super::rmodel::rtyper_makerepr`] |
//!
//! ## Deferred to follow-up commits
//!
//! * `rtype_chr` / `rtype_unichr` / `rtype_int` / `rtype_float`
//!   (rstr.py:516-541, 772-784) — char-side conversions that lower
//!   non-trivial usages and have no caller on the tuple eq/hash path.
//! * `AbstractStringRepr` / `AbstractUnicodeRepr` method bodies
//!   (`rtype_len`, `rtype_bool`, `rtype_method_*`, pairtype `rtype_eq`
//!   / `rtype_add` / `rtype_getitem`, `rtype_int` / `rtype_float`,
//!   etc., rstr.py:119-449 + 651-737). The struct skeletons and
//!   module-global singletons land here today; method bodies arrive
//!   slice-by-slice — see the epic plan in
//!   `~/.claude/projects/.../memory/item3_abstractstringrepr_epic_plan.md`.
//! * `LLHelpers.ll_*` helper graphs (`ll_str2int` / `ll_str2float`,
//!   `ll_startswith` / `ll_endswith` / `ll_find` / `ll_strip` /
//!   `ll_lower` / `ll_upper` / `ll_split` / `ll_join` / `ll_replace`)
//!   live in `lltypesystem/rstr.rs` and land slice-by-slice.

use std::sync::Arc;
use std::sync::OnceLock;

use crate::annotator::model::{SomeObjectTrait, SomeValue};
use crate::flowspace::model::{
    Block, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::lltypesystem::rstr::{
    STRPTR, UNICODEPTR, build_ll_chr2str_helper_graph, build_ll_count_char_helper_graph,
    build_ll_endswith_char_helper_graph, build_ll_endswith_helper_graph,
    build_ll_find_char_helper_graph, build_ll_int_helper_graph,
    build_ll_replace_chr_chr_helper_graph, build_ll_rfind_char_helper_graph,
    build_ll_startswith_char_helper_graph, build_ll_startswith_helper_graph,
    build_ll_str_is_true_helper_graph, build_ll_str2unicode_helper_graph,
    build_ll_strcmp_helper_graph, build_ll_strconcat_helper_graph, build_ll_streq_helper_graph,
    build_ll_strhash_helper_graph, build_ll_string_casefold_helper_graph,
    build_ll_string_isxxx_helper_graph, build_ll_stritem_checked_helper_graph,
    build_ll_stritem_helper_graph, build_ll_stritem_nonneg_checked_helper_graph,
    build_ll_stritem_nonneg_helper_graph, build_ll_strlen_helper_graph,
    build_ll_unichr2str_helper_graph,
};
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{
    ConvertedTo, GenopResult, HighLevelOp, LowLevelFunction, LowLevelOpList, RPythonTyper,
    constant_with_lltype, helper_pygraph_from_graph, variable_with_lltype,
};

// ____________________________________________________________
// StringRepr / UnicodeRepr — `rpython/rtyper/lltypesystem/rstr.py:229`
// + `:247`. The upstream class hierarchy splits over two files:
// `rstr.py:95` `class AbstractStringRepr(Repr)` carries the abstract
// method surface (`rtype_len`, `rtype_bool`, `rtype_method_startswith`
// / `endswith` / `find` / `count` / `strip` / `lower` / `upper` /
// `split` / `join` / `replace` / `format`, `rtype_int` / `rtype_float`,
// pairtype `rtype_eq` / `rtype_add` / `rtype_getitem`); the
// lltypesystem subclasses bind `lowleveltype = Ptr(STR)` / `Ptr(UNICODE)`
// and dispatch to `LLHelpers.ll_*` graphs.
//
// Pyre lands the **struct skeleton + module-global singletons** today
// (`Slice 3` of the Item 3 epic — `item3_abstractstringrepr_epic_plan.md`).
// Method bodies arrive in slices 4-12 alongside the `LLHelpers.ll_*`
// helper graphs in `lltypesystem/rstr.rs`. Until then the trait
// methods inherit `Repr`'s default `rtype_*` impls, which surface
// `MissingRTypeOperation` errors with the upstream method name.

/// RPython `class StringRepr(BaseLLStringRepr, AbstractStringRepr)`
/// (`lltypesystem/rstr.py:229-238`):
///
/// ```python
/// class StringRepr(BaseLLStringRepr, AbstractStringRepr):
///     lowleveltype = Ptr(STR)
///     basetype = str
///     base = Char
///     CACHE = CONST_STR_CACHE
/// ```
///
/// The `basetype` / `base` / `CACHE` attributes only matter for
/// `BaseLLStringRepr.convert_const` (`lltypesystem/rstr.py:191-206`)
/// which lands in a follow-up slice. Today the struct just carries
/// `lowleveltype = Ptr(STR)` so [`super::rmodel::rtyper_makerepr`]
/// can return the singleton when `SomeString` shows up. Per-method
/// `rtype_*` calls fall through `Repr`'s default `MissingRTypeOperation`
/// stubs until each slice 4-12 method body lands.
#[derive(Debug)]
pub struct StringRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl StringRepr {
    pub fn new() -> Self {
        StringRepr {
            state: ReprState::new(),
            lltype: STRPTR.clone(),
        }
    }

    /// RPython `StringRepr.char_repr = char_repr` (`lltypesystem/rstr.py:1268`)
    /// class-level attribute. Pyre exposes the link as a method that
    /// returns the module-global [`char_repr`] singleton.
    pub fn char_repr(&self) -> Arc<CharRepr> {
        char_repr()
    }
}

impl Default for StringRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for StringRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "StringRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::StringRepr
    }

    /// RPython `AbstractStringRepr.rtype_len(self, hop)` (`rstr.py:119-122`).
    fn rtype_len(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_abstract_string_len(self, hop, "ll_strlen", STRPTR.clone())
    }

    /// RPython `AbstractStringRepr.rtype_bool(self, hop)`
    /// (`rstr.py:124-132`).
    fn rtype_bool(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_abstract_string_bool(self, hop, "ll_str_is_true", STRPTR.clone())
    }

    /// RPython `AbstractStringRepr.rtype_unicode` (`rstr.py:385-396`):
    /// non-constant byte strings lower through `LLHelpers.ll_str2unicode`.
    fn rtype_unicode(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_abstract_string_unicode(self, hop)
    }

    /// RPython `AbstractStringRepr.rtype_int` (`rstr.py:371-383`):
    /// dispatch to `LLHelpers.ll_int(s, base)`.
    fn rtype_int(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_abstract_string_int(self, hop, "ll_int", STRPTR.clone())
    }

    /// RPython `AbstractStringRepr.rtype_method_*` dispatch table
    /// (`rstr.py:134-449`). Pyre lands methods slice-by-slice; today
    /// `startswith` (rstr.py:134-145) and `endswith` (rstr.py:147-158)
    /// lower.
    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        match method_name {
            "startswith" => rtype_abstract_string_method_startswith(
                self,
                hop,
                "ll_startswith",
                "ll_startswith_char",
                STRPTR.clone(),
                LowLevelType::Char,
            ),
            "endswith" => rtype_abstract_string_method_endswith(
                self,
                hop,
                "ll_endswith",
                "ll_endswith_char",
                STRPTR.clone(),
                LowLevelType::Char,
            ),
            "find" => rtype_abstract_string_method_findlike_char(
                self,
                hop,
                "ll_find_char",
                FindLikeCharKind::Find,
                STRPTR.clone(),
                LowLevelType::Char,
            ),
            "rfind" => rtype_abstract_string_method_findlike_char(
                self,
                hop,
                "ll_rfind_char",
                FindLikeCharKind::RFind,
                STRPTR.clone(),
                LowLevelType::Char,
            ),
            "count" => rtype_abstract_string_method_findlike_char(
                self,
                hop,
                "ll_count_char",
                FindLikeCharKind::Count,
                STRPTR.clone(),
                LowLevelType::Char,
            ),
            "isdigit" => rtype_abstract_string_method_isdigit(
                self,
                hop,
                "ll_isdigit",
                "ll_char_isdigit",
                STRPTR.clone(),
                LowLevelType::Char,
                "cast_char_to_int",
            ),
            "isalpha" => rtype_abstract_string_method_isxxx(
                self,
                hop,
                "ll_isalpha",
                "ll_char_isalpha",
                STRPTR.clone(),
                LowLevelType::Char,
                "cast_char_to_int",
                StringIsxxxPredicate::OrConditions(ISALPHA_CONDITIONS),
            ),
            "isalnum" => rtype_abstract_string_method_isxxx(
                self,
                hop,
                "ll_isalnum",
                "ll_char_isalnum",
                STRPTR.clone(),
                LowLevelType::Char,
                "cast_char_to_int",
                StringIsxxxPredicate::OrConditions(ISALNUM_CONDITIONS),
            ),
            "isspace" => rtype_abstract_string_method_isxxx(
                self,
                hop,
                "ll_isspace",
                "ll_char_isspace",
                STRPTR.clone(),
                LowLevelType::Char,
                "cast_char_to_int",
                StringIsxxxPredicate::OrConditions(ISSPACE_CONDITIONS),
            ),
            "isupper" => rtype_abstract_string_method_isxxx(
                self,
                hop,
                "ll_isupper",
                "ll_char_isupper",
                STRPTR.clone(),
                LowLevelType::Char,
                "cast_char_to_int",
                StringIsxxxPredicate::InRange(65, 90),
            ),
            "islower" => rtype_abstract_string_method_isxxx(
                self,
                hop,
                "ll_islower",
                "ll_char_islower",
                STRPTR.clone(),
                LowLevelType::Char,
                "cast_char_to_int",
                StringIsxxxPredicate::InRange(97, 122),
            ),
            "upper" => rtype_abstract_string_method_casefold(
                self,
                hop,
                "ll_upper",
                "ll_upper_char",
                97,
                122,
                -32,
            ),
            "lower" => rtype_abstract_string_method_casefold(
                self,
                hop,
                "ll_lower",
                "ll_lower_char",
                65,
                90,
                32,
            ),
            "replace" => rtype_abstract_string_method_replace(
                self,
                hop,
                "ll_replace_chr_chr",
                STRPTR.clone(),
                LowLevelType::Char,
            ),
            _ => Err(TyperError::message(format!(
                "missing StringRepr.rtype_method_{method_name}"
            ))),
        }
    }

    /// RPython `AbstractStringRepr.get_ll_eq_function` (`rstr.py:110-111`):
    ///
    /// ```python
    /// def get_ll_eq_function(self):
    ///     return self.ll.ll_streq
    /// ```
    fn get_ll_eq_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        let name = "ll_streq".to_string();
        let ptr_for_builder = STRPTR.clone();
        rtyper
            .lowlevel_helper_function_with_builder(
                name.clone(),
                vec![STRPTR.clone(), STRPTR.clone()],
                LowLevelType::Bool,
                move |_rtyper, _args, _result| build_ll_streq_helper_graph(&name, ptr_for_builder),
            )
            .map(Some)
    }

    /// RPython `AbstractStringRepr.get_ll_hash_function` (`rstr.py:113-114`):
    ///
    /// ```python
    /// def get_ll_hash_function(self):
    ///     return self.ll.ll_strhash
    /// ```
    fn get_ll_hash_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        rtyper
            .lowlevel_helper_function_with_builder(
                "ll_strhash".to_string(),
                vec![STRPTR.clone()],
                LowLevelType::Signed,
                move |rtyper_inner, _args, _result| {
                    build_ll_strhash_helper_graph(
                        rtyper_inner,
                        "ll_strhash",
                        STRPTR.clone(),
                        "_ll_strhash",
                        "_hash_string",
                    )
                },
            )
            .map(Some)
    }
}

/// RPython `class UnicodeRepr(BaseLLStringRepr, AbstractUnicodeRepr)`
/// (`lltypesystem/rstr.py:247-256`):
///
/// ```python
/// class UnicodeRepr(BaseLLStringRepr, AbstractUnicodeRepr):
///     lowleveltype = Ptr(UNICODE)
///     basetype = basestring
///     base = UniChar
///     CACHE = CONST_UNICODE_CACHE
/// ```
///
/// Mirror of [`StringRepr`] swapping `Ptr(STR)` → `Ptr(UNICODE)` and
/// the char-side backlink to `unichar_repr`.
#[derive(Debug)]
pub struct UnicodeRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl UnicodeRepr {
    pub fn new() -> Self {
        UnicodeRepr {
            state: ReprState::new(),
            lltype: UNICODEPTR.clone(),
        }
    }

    /// RPython `UnicodeRepr.char_repr = unichar_repr`
    /// (`lltypesystem/rstr.py:1266`).
    pub fn char_repr(&self) -> Arc<UniCharRepr> {
        unichar_repr()
    }
}

impl Default for UnicodeRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for UnicodeRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "UnicodeRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::UnicodeRepr
    }

    /// RPython `AbstractUnicodeRepr.rtype_len` inherits the
    /// `AbstractStringRepr.rtype_len` body (`rstr.py:119-122`); the
    /// only delta is the helper-graph identity (`ll_unilen`) and the
    /// pointer lltype (`Ptr(UNICODE)`).
    fn rtype_len(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_abstract_string_len(self, hop, "ll_unilen", UNICODEPTR.clone())
    }

    /// RPython `AbstractUnicodeRepr.rtype_bool` inherits
    /// `AbstractStringRepr.rtype_bool` (`rstr.py:124-132`); only the
    /// `is_true` helper identity (`ll_unicode_is_true`) and the
    /// pointer lltype differ.
    fn rtype_bool(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_abstract_string_bool(self, hop, "ll_unicode_is_true", UNICODEPTR.clone())
    }

    /// RPython `AbstractStringRepr.rtype_unicode` identity branch for
    /// values whose repr already equals the result UnicodeRepr.
    fn rtype_unicode(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_abstract_string_unicode(self, hop)
    }

    /// `AbstractStringRepr.rtype_int`, inherited by UnicodeRepr with
    /// the unicode-specialised helper graph.
    fn rtype_int(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_abstract_string_int(self, hop, "ll_unicode_int", UNICODEPTR.clone())
    }

    /// RPython `AbstractUnicodeRepr.rtype_method_*` dispatch table
    /// (`rstr.py:134-449` inherited via `AbstractUnicodeRepr(AbstractStringRepr)`).
    /// Pyre lands methods slice-by-slice; today `startswith` and
    /// `endswith` lower with helper-graph identities
    /// `ll_unicode_startswith` / `ll_unicode_endswith` (Ptr(UNICODE)).
    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        match method_name {
            "startswith" => rtype_abstract_string_method_startswith(
                self,
                hop,
                "ll_unicode_startswith",
                "ll_unicode_startswith_char",
                UNICODEPTR.clone(),
                LowLevelType::UniChar,
            ),
            "endswith" => rtype_abstract_string_method_endswith(
                self,
                hop,
                "ll_unicode_endswith",
                "ll_unicode_endswith_char",
                UNICODEPTR.clone(),
                LowLevelType::UniChar,
            ),
            "find" => rtype_abstract_string_method_findlike_char(
                self,
                hop,
                "ll_unicode_find_char",
                FindLikeCharKind::Find,
                UNICODEPTR.clone(),
                LowLevelType::UniChar,
            ),
            "rfind" => rtype_abstract_string_method_findlike_char(
                self,
                hop,
                "ll_unicode_rfind_char",
                FindLikeCharKind::RFind,
                UNICODEPTR.clone(),
                LowLevelType::UniChar,
            ),
            "count" => rtype_abstract_string_method_findlike_char(
                self,
                hop,
                "ll_unicode_count_char",
                FindLikeCharKind::Count,
                UNICODEPTR.clone(),
                LowLevelType::UniChar,
            ),
            "isdigit" => rtype_abstract_string_method_isdigit(
                self,
                hop,
                "ll_unicode_isdigit",
                "ll_unichar_isdigit",
                UNICODEPTR.clone(),
                LowLevelType::UniChar,
                "cast_unichar_to_int",
            ),
            "isalpha" => rtype_abstract_string_method_isxxx(
                self,
                hop,
                "ll_unicode_isalpha",
                "ll_unichar_isalpha",
                UNICODEPTR.clone(),
                LowLevelType::UniChar,
                "cast_unichar_to_int",
                StringIsxxxPredicate::OrConditions(ISALPHA_CONDITIONS),
            ),
            "isalnum" => rtype_abstract_string_method_isxxx(
                self,
                hop,
                "ll_unicode_isalnum",
                "ll_unichar_isalnum",
                UNICODEPTR.clone(),
                LowLevelType::UniChar,
                "cast_unichar_to_int",
                StringIsxxxPredicate::OrConditions(ISALNUM_CONDITIONS),
            ),
            "isspace" => rtype_abstract_string_method_isxxx(
                self,
                hop,
                "ll_unicode_isspace",
                "ll_unichar_isspace",
                UNICODEPTR.clone(),
                LowLevelType::UniChar,
                "cast_unichar_to_int",
                StringIsxxxPredicate::OrConditions(ISSPACE_CONDITIONS),
            ),
            "isupper" => rtype_abstract_string_method_isxxx(
                self,
                hop,
                "ll_unicode_isupper",
                "ll_unichar_isupper",
                UNICODEPTR.clone(),
                LowLevelType::UniChar,
                "cast_unichar_to_int",
                StringIsxxxPredicate::InRange(65, 90),
            ),
            "islower" => rtype_abstract_string_method_isxxx(
                self,
                hop,
                "ll_unicode_islower",
                "ll_unichar_islower",
                UNICODEPTR.clone(),
                LowLevelType::UniChar,
                "cast_unichar_to_int",
                StringIsxxxPredicate::InRange(97, 122),
            ),
            // AbstractUnicodeRepr overrides these to raise TyperError
            // instead of inheriting AbstractStringRepr's byte-string
            // casefold helpers (rstr.py:452-457).
            "upper" => Err(TyperError::message("Cannot do toupper on unicode string")),
            "lower" => Err(TyperError::message("Cannot do tolower on unicode string")),
            "replace" => rtype_abstract_string_method_replace(
                self,
                hop,
                "ll_unicode_replace_chr_chr",
                UNICODEPTR.clone(),
                LowLevelType::UniChar,
            ),
            _ => Err(TyperError::message(format!(
                "missing UnicodeRepr.rtype_method_{method_name}"
            ))),
        }
    }

    /// Mirror of `StringRepr::get_ll_eq_function` for the Unicode pair —
    /// helper-cache key `ll_unicode_eq` (Ptr(UNICODE)).
    fn get_ll_eq_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        let name = "ll_unicode_eq".to_string();
        let ptr_for_builder = UNICODEPTR.clone();
        rtyper
            .lowlevel_helper_function_with_builder(
                name.clone(),
                vec![UNICODEPTR.clone(), UNICODEPTR.clone()],
                LowLevelType::Bool,
                move |_rtyper, _args, _result| build_ll_streq_helper_graph(&name, ptr_for_builder),
            )
            .map(Some)
    }

    /// Mirror of `StringRepr::get_ll_hash_function` for the Unicode
    /// pair — helper-cache key `ll_unicode_hash` (Ptr(UNICODE)).
    fn get_ll_hash_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        rtyper
            .lowlevel_helper_function_with_builder(
                "ll_unicode_hash".to_string(),
                vec![UNICODEPTR.clone()],
                LowLevelType::Signed,
                move |rtyper_inner, _args, _result| {
                    build_ll_strhash_helper_graph(
                        rtyper_inner,
                        "ll_unicode_hash",
                        UNICODEPTR.clone(),
                        "_ll_unicode_strhash",
                        "_hash_unicode_string",
                    )
                },
            )
            .map(Some)
    }
}

/// RPython `string_repr = StringRepr()` (`lltypesystem/rstr.py:1255`)
/// module-global. Pyre mirrors the upstream singleton via [`OnceLock`]
/// so every `SomeString.rtyper_makerepr(rtyper)` call returns the same
/// `Arc`.
pub fn string_repr() -> Arc<StringRepr> {
    static REPR: OnceLock<Arc<StringRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(StringRepr::new())).clone()
}

/// RPython `unicode_repr = UnicodeRepr()` (`lltypesystem/rstr.py:1260`)
/// module-global.
pub fn unicode_repr() -> Arc<UnicodeRepr> {
    static REPR: OnceLock<Arc<UnicodeRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(UnicodeRepr::new())).clone()
}

/// RPython `AbstractStringRepr.rtype_len(self, hop)` (`rstr.py:119-122`):
///
/// ```python
/// def rtype_len(self, hop):
///     string_repr = self.repr
///     v_str, = hop.inputargs(string_repr)
///     return hop.gendirectcall(self.ll.ll_strlen, v_str)
/// ```
///
/// Shared body for the StringRepr / UnicodeRepr `rtype_len` impls —
/// both reprs carry the same `len(s.chars)` lowering, only the
/// pointer lltype (`Ptr(STR)` vs `Ptr(UNICODE)`) and helper-graph
/// identity differ.
fn rtype_abstract_string_len(
    self_repr: &dyn Repr,
    hop: &HighLevelOp,
    helper_name: &str,
    ptr_lltype: LowLevelType,
) -> RTypeResult {
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(self_repr)])?;
    let helper_name_owned = helper_name.to_string();
    let ptr_for_builder = ptr_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        helper_name_owned.clone(),
        vec![ptr_lltype],
        LowLevelType::Signed,
        move |_rtyper, _args, _result| {
            build_ll_strlen_helper_graph(&helper_name_owned, ptr_for_builder)
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// RPython `AbstractStringRepr.rtype_bool(self, hop)` (`rstr.py:124-132`):
///
/// ```python
/// def rtype_bool(self, hop):
///     s_str = hop.args_s[0]
///     if s_str.can_be_None:
///         string_repr = hop.args_r[0].repr
///         v_str, = hop.inputargs(string_repr)
///         return hop.gendirectcall(self.ll.ll_str_is_true, v_str)
///     else:
///         # defaults to checking the length
///         return super(AbstractStringRepr, self).rtype_bool(hop)
/// ```
///
/// The `super().rtype_bool` fallback is the `Repr.rtype_bool` default
/// at `rmodel.py:199-207` — `int_is_true(self.rtype_len(hop))`. Rust
/// trait defaults can't be invoked from an override, so the fallback
/// is replicated inline. The non-None path emits the
/// `ll_str_is_true`/`ll_unicode_is_true` helper graph.
fn rtype_abstract_string_bool(
    self_repr: &dyn Repr,
    hop: &HighLevelOp,
    is_true_helper_name: &str,
    ptr_lltype: LowLevelType,
) -> RTypeResult {
    let can_be_none = match hop.args_s.borrow().first() {
        Some(SomeValue::String(s)) => s.inner.can_be_none,
        Some(SomeValue::UnicodeString(s)) => s.inner.can_be_none,
        _ => false,
    };
    if can_be_none {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self_repr)])?;
        let helper_name_owned = is_true_helper_name.to_string();
        let ptr_for_builder = ptr_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            helper_name_owned.clone(),
            vec![ptr_lltype],
            LowLevelType::Bool,
            move |_rtyper, _args, _result| {
                build_ll_str_is_true_helper_graph(&helper_name_owned, ptr_for_builder)
            },
        )?;
        hop.gendirectcall(&helper, vlist)
    } else {
        // Replicates `Repr.rtype_bool` (`rmodel.py:199-207`):
        // `gendirectcall(rtype_len)` then wrap with `int_is_true`.
        let v_len = self_repr.rtype_len(hop)?.ok_or_else(|| {
            TyperError::message(format!(
                "rtype_bool({}) returned no length value",
                self_repr.repr_string()
            ))
        })?;
        Ok(hop.genop(
            "int_is_true",
            vec![v_len],
            GenopResult::LLType(LowLevelType::Bool),
        ))
    }
}

fn lowlevel_str2unicode_function(rtyper: &RPythonTyper) -> Result<LowLevelFunction, TyperError> {
    rtyper.lowlevel_helper_function_with_builder(
        "ll_str2unicode".to_string(),
        vec![STRPTR.clone()],
        UNICODEPTR.clone(),
        move |_rtyper, _args, _result| build_ll_str2unicode_helper_graph("ll_str2unicode"),
    )
}

fn lowlevel_int_function(
    rtyper: &RPythonTyper,
    helper_name: &str,
    ptr_lltype: LowLevelType,
) -> Result<LowLevelFunction, TyperError> {
    let helper_name_owned = helper_name.to_string();
    let ptr_for_builder = ptr_lltype.clone();
    rtyper.lowlevel_helper_function_with_builder(
        helper_name_owned.clone(),
        vec![ptr_lltype, LowLevelType::Signed],
        LowLevelType::Signed,
        move |_rtyper, _args, _result| {
            build_ll_int_helper_graph(&helper_name_owned, ptr_for_builder.clone())
        },
    )
}

/// RPython `AbstractStringRepr.rtype_int` (`rstr.py:371-383`).
/// The helper itself implements the upstream base parser
/// (`LLHelpers.ll_int(s, base)`) including base range validation,
/// leading/trailing spaces, sign handling, and ValueError exits.
fn rtype_abstract_string_int(
    self_repr: &dyn Repr,
    hop: &HighLevelOp,
    helper_name: &str,
    ptr_lltype: LowLevelType,
) -> RTypeResult {
    hop.has_implicit_exception("ValueError");
    let helper = lowlevel_int_function(hop.rtyper.as_ref(), helper_name, ptr_lltype)?;
    match hop.nb_args() {
        1 => {
            let v_str = hop.inputarg(ConvertedTo::Repr(self_repr), 0)?;
            let c_base = Hlvalue::Constant(HighLevelOp::inputconst(
                ConvertedTo::LowLevelType(&LowLevelType::Signed),
                &ConstValue::Int(10),
            )?);
            hop.exception_is_here()?;
            hop.gendirectcall(&helper, vec![v_str, c_base])
        }
        2 => {
            let r_base = hop
                .args_r
                .borrow()
                .get(1)
                .cloned()
                .flatten()
                .ok_or_else(|| TyperError::message("rtype_int: args_r[1] missing"))?;
            if r_base.lowleveltype() != &LowLevelType::Signed {
                return Err(TyperError::message("base needs to be an int"));
            }
            let vlist = hop.inputargs(vec![
                ConvertedTo::Repr(self_repr),
                ConvertedTo::LowLevelType(&LowLevelType::Signed),
            ])?;
            hop.exception_is_here()?;
            hop.gendirectcall(&helper, vlist)
        }
        other => Err(TyperError::message(format!(
            "rtype_int: expected 1 or 2 args, got {other}"
        ))),
    }
}

/// RPython `AbstractStringRepr.rtype_unicode` (`rstr.py:385-396`).
/// The same body is inherited by UnicodeRepr: constants are returned as
/// constants, already-unicode values are identity, byte strings call
/// `ll_str2unicode` and can raise `UnicodeDecodeError`.
fn rtype_abstract_string_unicode(self_repr: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    if let Some(result_const) = hop
        .s_result
        .borrow()
        .as_ref()
        .and_then(|s| s.const_().cloned())
    {
        hop.exception_cannot_occur()?;
        let r_result = hop
            .r_result
            .borrow()
            .clone()
            .ok_or_else(|| TyperError::message("rtype_unicode: r_result missing"))?;
        return HighLevelOp::inputconst(ConvertedTo::Repr(r_result.as_ref()), &result_const)
            .map(Hlvalue::Constant)
            .map(Some);
    }

    let v_str = hop.inputarg(ConvertedTo::Repr(self_repr), 0)?;
    let r_result = hop
        .r_result
        .borrow()
        .clone()
        .ok_or_else(|| TyperError::message("rtype_unicode: r_result missing"))?;
    if self_repr.lowleveltype() == r_result.lowleveltype() {
        hop.exception_cannot_occur()?;
        return Ok(Some(v_str));
    }

    hop.exception_is_here()?;
    let helper = lowlevel_str2unicode_function(hop.rtyper.as_ref())?;
    hop.gendirectcall(&helper, vec![v_str])
}

fn lowlevel_chr2str_function(
    rtyper: &RPythonTyper,
    helper_name: &str,
    ptr_lltype: LowLevelType,
    elem_lltype: LowLevelType,
) -> Result<LowLevelFunction, TyperError> {
    let helper_name_owned = helper_name.to_string();
    let ptr_for_builder = ptr_lltype.clone();
    let elem_for_builder = elem_lltype.clone();
    rtyper.lowlevel_helper_function_with_builder(
        helper_name_owned.clone(),
        vec![elem_lltype],
        ptr_lltype,
        move |_rtyper, _args, _result| {
            build_ll_chr2str_helper_graph(&helper_name_owned, ptr_for_builder, elem_for_builder)
        },
    )
}

fn lowlevel_unichr2str_function(rtyper: &RPythonTyper) -> Result<LowLevelFunction, TyperError> {
    rtyper.lowlevel_helper_function_with_builder(
        "ll_unichr2str".to_string(),
        vec![LowLevelType::UniChar],
        STRPTR.clone(),
        move |_rtyper, _args, _result| build_ll_unichr2str_helper_graph("ll_unichr2str"),
    )
}

/// RPython `AbstractCharRepr.ll_str(ch) -> ll_chr2str(ch)` and the
/// UniChar→Unicode half of `pairtype(AbstractUniCharRepr,
/// AbstractUnicodeRepr).convert_from_to` (`rstr.py:554-562, 805-814`).
fn rtype_charlike_chr2str(
    hop: &HighLevelOp,
    helper_name: &str,
    ptr_lltype: LowLevelType,
    elem_lltype: LowLevelType,
) -> RTypeResult {
    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&elem_lltype)])?;
    hop.exception_cannot_occur()?;
    let helper =
        lowlevel_chr2str_function(hop.rtyper.as_ref(), helper_name, ptr_lltype, elem_lltype)?;
    hop.gendirectcall(&helper, vlist)
}

/// RPython `pairtype(AbstractCharRepr, AbstractStringRepr)` and
/// `pairtype(AbstractUniCharRepr, AbstractUnicodeRepr).convert_from_to`
/// (`rstr.py:805-814`): char-to-one-character-string conversion.
pub fn pair_charlike_string_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    let (helper_name, ptr_lltype, elem_lltype) =
        match (r_from.repr_class_id(), r_to.repr_class_id()) {
            (super::pairtype::ReprClassId::CharRepr, super::pairtype::ReprClassId::StringRepr) => {
                ("ll_chr2str", STRPTR.clone(), LowLevelType::Char)
            }
            (
                super::pairtype::ReprClassId::UniCharRepr,
                super::pairtype::ReprClassId::UnicodeRepr,
            ) => (
                "ll_unichr2unicode",
                UNICODEPTR.clone(),
                LowLevelType::UniChar,
            ),
            _ => return Ok(None),
        };

    let helper = lowlevel_chr2str_function(
        llops
            .require_rtyper("pair_char_string_convert_from_to")?
            .as_ref(),
        helper_name,
        ptr_lltype,
        elem_lltype,
    )?;
    let result = llops.gendirectcall(&helper, vec![v.clone()])?;
    Ok(result.map(Hlvalue::Variable))
}

/// RPython `pairtype(AbstractStringRepr, AbstractCharRepr).convert_from_to`
/// (`rstr.py:815-820`): `string_repr -> char_repr` extracts index 0
/// through `ll_stritem_nonneg`.
pub fn pair_string_char_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    if r_from.repr_class_id() != super::pairtype::ReprClassId::StringRepr
        || r_to.repr_class_id() != super::pairtype::ReprClassId::CharRepr
    {
        return Ok(None);
    }
    let helper = llops
        .require_rtyper("pair_string_char_convert_from_to")?
        .lowlevel_helper_function_with_builder(
            "ll_stritem_nonneg".to_string(),
            vec![STRPTR.clone(), LowLevelType::Signed],
            LowLevelType::Char,
            move |_rtyper, _args, _result| {
                build_ll_stritem_nonneg_helper_graph("ll_stritem_nonneg", STRPTR.clone())
            },
        )?;
    let zero = Hlvalue::Constant(HighLevelOp::inputconst(
        &LowLevelType::Signed,
        &ConstValue::Int(0),
    )?);
    let result = llops.gendirectcall(&helper, vec![v.clone(), zero])?;
    Ok(result.map(Hlvalue::Variable))
}

// ____________________________________________________________
// CharRepr — `rstr.py:483-541` (lltypesystem-bound `AbstractCharRepr`).

/// RPython `class AbstractCharRepr(AbstractStringRepr, AbstractCharRepr_)`
/// (`rstr.py:483-541`) — the lltypesystem `CharRepr` carries `lowleveltype = Char`.
///
/// Pyre lands a single concrete `CharRepr` since the
/// abstract/lltypesystem split is an upstream artefact of supporting
/// both `lltypesystem` and `ootypesystem`; pyre only targets
/// lltypesystem.
#[derive(Debug)]
pub struct CharRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl CharRepr {
    pub fn new() -> Self {
        CharRepr {
            state: ReprState::new(),
            lltype: LowLevelType::Char,
        }
    }

    /// RPython `CharRepr.char_repr = char_repr`
    /// (`lltypesystem/rstr.py:1267`) class-level attribute — char-side
    /// backlink so the shared `BaseCharReprMixin._rtype_method_isxxx`
    /// helper (`rstr.py:516-520`) can read `hop.args_r[0].char_repr`.
    pub fn char_repr(&self) -> Arc<CharRepr> {
        char_repr()
    }

    /// RPython `class CharRepr(AbstractCharRepr, StringRepr)`
    /// (`lltypesystem/rstr.py:291-292`) — `CharRepr` MRO inherits
    /// `StringRepr.repr = string_repr` (`lltypesystem/rstr.py:1262`).
    /// Method form so callers (`rstr.py:120` `string_repr = self.repr`)
    /// reach the parent string repr through `char_repr.repr()`.
    pub fn repr(&self) -> Arc<StringRepr> {
        string_repr()
    }
}

impl Default for CharRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for CharRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "CharRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::CharRepr
    }

    /// RPython `BaseCharReprMixin.convert_const(self, value)`
    /// (`rstr.py:491-494`):
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if not isinstance(value, str) or len(value) != 1:
    ///         raise TyperError("not a character: %r" % (value,))
    ///     return value
    /// ```
    ///
    /// Pyre maps `Char` lltype to a one-byte
    /// [`ConstValue::ByteStr`]. Unicode constants are rejected here.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        match value {
            ConstValue::ByteStr(s) if s.len() == 1 => Ok(Constant::with_concretetype(
                ConstValue::ByteStr(s.clone()),
                LowLevelType::Char,
            )),
            other => Err(TyperError::message(format!("not a character: {other:?}"))),
        }
    }

    /// RPython `AbstractCharRepr.get_ll_eq_function` (`rstr.py:496-497`):
    ///
    /// ```python
    /// def get_ll_eq_function(self):
    ///     return None
    /// ```
    ///
    /// Returning `None` instructs callers (`gen_eq_function` /
    /// `rtype_contains`) to fall back to the primitive `char_eq`
    /// inline op via `eq_funcs[i] or operator.eq`.
    fn get_ll_eq_function(
        &self,
        _rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        Ok(None)
    }

    /// RPython `AbstractCharRepr.get_ll_hash_function` (`rstr.py:499-500`):
    ///
    /// ```python
    /// def get_ll_hash_function(self):
    ///     return self.ll.ll_char_hash
    ///
    /// def ll_char_hash(ch):
    ///     return ord(ch)
    /// ```
    ///
    /// Synthesizes the `ll_char_hash(ch) -> Signed` helper graph.
    /// Body: single block, `cast_char_to_int(ch) -> hashed` then close
    /// to returnblock.
    fn get_ll_hash_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        let name = "ll_char_hash".to_string();
        rtyper
            .lowlevel_helper_function_with_builder(
                name.clone(),
                vec![LowLevelType::Char],
                LowLevelType::Signed,
                move |_rtyper, _args, _result| build_ll_char_hash_helper_graph(&name),
            )
            .map(Some)
    }

    /// RPython `BaseCharReprMixin.rtype_len(_, hop)` (`rstr.py:504-505`):
    /// `return hop.inputconst(Signed, 1)`. Single chars always carry
    /// length 1.
    fn rtype_len(&self, _hop: &HighLevelOp) -> RTypeResult {
        let c = HighLevelOp::inputconst(&LowLevelType::Signed, &ConstValue::Int(1))?;
        Ok(Some(Hlvalue::Constant(c)))
    }

    /// RPython `BaseCharReprMixin.rtype_bool(_, hop)` (`rstr.py:507-509`):
    /// `assert not hop.args_s[0].can_be_None; return hop.inputconst(Bool, True)`.
    /// Pyre's CharRepr has lltype `Char` (not nullable in the lltype
    /// sense — `NoneRepr` would be a separate static type), so the
    /// `can_be_None` assert is structurally satisfied.
    fn rtype_bool(&self, _hop: &HighLevelOp) -> RTypeResult {
        let c = HighLevelOp::inputconst(&LowLevelType::Bool, &ConstValue::Bool(true))?;
        Ok(Some(Hlvalue::Constant(c)))
    }

    /// RPython `AbstractCharRepr.ll_str(ch)` (`rstr.py:554-555`) plus
    /// the builtin `str(ch)` lowering path: allocate a one-character
    /// `Ptr(STR)` via `LLHelpers.ll_chr2str`.
    fn rtype_str(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_charlike_chr2str(hop, "ll_chr2str", STRPTR.clone(), LowLevelType::Char)
    }

    /// RPython `BaseCharReprMixin.rtype_ord(_, hop)` (`rstr.py:511-514`):
    ///
    /// ```python
    /// def rtype_ord(_, hop):
    ///     repr = hop.args_r[0].char_repr
    ///     vlist = hop.inputargs(repr)
    ///     return hop.genop('cast_char_to_int', vlist, resulttype=Signed)
    /// ```
    fn rtype_ord(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Char)])?;
        Ok(hop.genop(
            "cast_char_to_int",
            vlist,
            GenopResult::LLType(LowLevelType::Signed),
        ))
    }

    /// RPython `BaseCharReprMixin._rtype_method_isxxx(_, llfn, hop)`
    /// dispatch (`rstr.py:516-538`) — routes by method name to the
    /// per-predicate `ll_char_*` helper graph.
    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        // Simple inrange predicates (rstr.py:891-922).
        if let Some((llfn_name, lo, hi)) = match method_name {
            "isdigit" => Some(("ll_char_isdigit", 48, 57)),
            "isupper" => Some(("ll_char_isupper", 65, 90)),
            "islower" => Some(("ll_char_islower", 97, 122)),
            _ => None,
        } {
            return char_predicate_inrange_method(
                hop,
                llfn_name.to_string(),
                LowLevelType::Char,
                "cast_char_to_int",
                lo,
                hi,
            );
        }

        // OR-of-conditions predicates (rstr.py:886-912).
        if let Some((llfn_name, conditions)) = match method_name {
            "isspace" => Some(("ll_char_isspace", ISSPACE_CONDITIONS)),
            "isalpha" => Some(("ll_char_isalpha", ISALPHA_CONDITIONS)),
            "isalnum" => Some(("ll_char_isalnum", ISALNUM_CONDITIONS)),
            _ => None,
        } {
            return char_predicate_or_of_conditions_method(
                hop,
                llfn_name.to_string(),
                LowLevelType::Char,
                "cast_char_to_int",
                conditions,
            );
        }

        // ASCII case-folding (rstr.py:542-552 + 925-934). lower: 'A'..='Z'
        // → +32; upper: 'a'..='z' → -32. UniCharRepr does NOT define
        // these — Unicode case-folding semantics are out of scope.
        if let Some((llfn_name, lo, hi, offset)) = match method_name {
            "lower" => Some(("ll_lower_char", 65, 90, 32)),
            "upper" => Some(("ll_upper_char", 97, 122, -32)),
            _ => None,
        } {
            return char_case_fold_method(hop, llfn_name.to_string(), lo, hi, offset);
        }

        Err(TyperError::message(format!(
            "missing CharRepr.rtype_method_{method_name}"
        )))
    }
}

/// RPython `char_repr = CharRepr()` (`rstr.py:1009`) module-global.
pub fn char_repr() -> Arc<CharRepr> {
    static REPR: OnceLock<Arc<CharRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(CharRepr::new())).clone()
}

/// Synthesizes the `ll_char_hash(ch)` helper graph (`rstr.py:937-938`):
/// single block, `cast_char_to_int(ch) -> hashed` then return.
pub(crate) fn build_ll_char_hash_helper_graph(name: &str) -> Result<PyGraph, TyperError> {
    build_ll_charlike_hash_helper_graph(name, LowLevelType::Char, "cast_char_to_int")
}

// ____________________________________________________________
// UniCharRepr — `rstr.py:758-775` (lltypesystem-bound `AbstractUniCharRepr`).

/// RPython `class AbstractUniCharRepr(AbstractUnicodeRepr,
/// AbstractCharRepr_)` (`rstr.py:758-775`) — lltypesystem `UniCharRepr`
/// carries `lowleveltype = UniChar`.
#[derive(Debug)]
pub struct UniCharRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl UniCharRepr {
    /// RPython `UniCharRepr.char_repr = unichar_repr`
    /// (`lltypesystem/rstr.py:1265`) — UniCharRepr's char-side backlink
    /// is itself; mirrors `CharRepr.char_repr = char_repr`
    /// (`lltypesystem/rstr.py:1267`).
    pub fn char_repr(&self) -> Arc<UniCharRepr> {
        unichar_repr()
    }

    /// RPython `class UniCharRepr(AbstractUniCharRepr, UnicodeRepr)`
    /// (`lltypesystem/rstr.py:294-295`) — `UniCharRepr` MRO inherits
    /// `UniCharRepr.repr = unicode_repr` (`lltypesystem/rstr.py:1264`).
    pub fn repr(&self) -> Arc<UnicodeRepr> {
        unicode_repr()
    }

    pub fn new() -> Self {
        UniCharRepr {
            state: ReprState::new(),
            lltype: LowLevelType::UniChar,
        }
    }
}

impl Default for UniCharRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for UniCharRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "UniCharRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::UniCharRepr
    }

    /// RPython `AbstractUniCharRepr.convert_const` (`rstr.py:759-766`):
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if isinstance(value, str):
    ///         value = unicode(value)
    ///     if not isinstance(value, unicode) or len(value) != 1:
    ///         raise TyperError("not a unicode character: %r" % (value,))
    ///     return value
    /// ```
    ///
    /// Python2's `unicode(value)` for a `str` runs the default codec
    /// (ASCII) — non-ASCII bytes raise `UnicodeDecodeError`. Pyre
    /// mirrors that: a one-byte ASCII [`ConstValue::ByteStr`] is
    /// promoted to a [`ConstValue::UniStr`] holding the same scalar
    /// (codepoint < 0x80, so byte value == codepoint), while non-ASCII
    /// or multi-byte byte strings are rejected. Native `UniStr` of
    /// length 1 passes through unchanged.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        match value {
            ConstValue::UniStr(s) if s.chars().count() == 1 => Ok(Constant::with_concretetype(
                ConstValue::UniStr(s.clone()),
                LowLevelType::UniChar,
            )),
            ConstValue::ByteStr(b) if b.len() == 1 && b[0] < 0x80 => {
                Ok(Constant::with_concretetype(
                    ConstValue::UniStr((b[0] as char).to_string()),
                    LowLevelType::UniChar,
                ))
            }
            other => Err(TyperError::message(format!(
                "not a unicode character: {other:?}"
            ))),
        }
    }

    /// RPython `AbstractUniCharRepr.get_ll_eq_function` (`rstr.py:764-765`):
    /// `return None` — callers fall back to primitive `unichar_eq`.
    fn get_ll_eq_function(
        &self,
        _rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        Ok(None)
    }

    /// RPython `AbstractUniCharRepr.get_ll_hash_function` (`rstr.py:767-768`):
    ///
    /// ```python
    /// def get_ll_hash_function(self):
    ///     return self.ll.ll_unichar_hash
    ///
    /// def ll_unichar_hash(ch):
    ///     return ord(ch)
    /// ```
    fn get_ll_hash_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        let name = "ll_unichar_hash".to_string();
        rtyper
            .lowlevel_helper_function_with_builder(
                name.clone(),
                vec![LowLevelType::UniChar],
                LowLevelType::Signed,
                move |_rtyper, _args, _result| build_ll_unichar_hash_helper_graph(&name),
            )
            .map(Some)
    }

    /// `BaseCharReprMixin.rtype_len` (`rstr.py:504-505`).
    fn rtype_len(&self, _hop: &HighLevelOp) -> RTypeResult {
        let c = HighLevelOp::inputconst(&LowLevelType::Signed, &ConstValue::Int(1))?;
        Ok(Some(Hlvalue::Constant(c)))
    }

    /// `BaseCharReprMixin.rtype_bool` (`rstr.py:507-509`).
    fn rtype_bool(&self, _hop: &HighLevelOp) -> RTypeResult {
        let c = HighLevelOp::inputconst(&LowLevelType::Bool, &ConstValue::Bool(true))?;
        Ok(Some(Hlvalue::Constant(c)))
    }

    /// RPython `AbstractUniCharRepr.ll_unicode(ch)` (`rstr.py:564-565`):
    /// `return unicode(ch)`. In lltypesystem this is the same
    /// one-character allocation shape as `ll_chr2str`, specialised to
    /// `UniChar -> Ptr(UNICODE)`.
    fn rtype_unicode(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_charlike_chr2str(
            hop,
            "ll_unichr2unicode",
            UNICODEPTR.clone(),
            LowLevelType::UniChar,
        )
    }

    /// RPython `AbstractUniCharRepr.ll_str(ch)` (`rstr.py:560-562`):
    /// `str(unicode(ch))`. The helper implements the single-codepoint
    /// ASCII encode path and raises `UnicodeEncodeError` for
    /// non-ASCII, matching the upstream unicode-to-str conversion
    /// surface for this narrowed input.
    fn rtype_str(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::UniChar)])?;
        hop.exception_is_here()?;
        let helper = lowlevel_unichr2str_function(hop.rtyper.as_ref())?;
        hop.gendirectcall(&helper, vlist)
    }

    /// `BaseCharReprMixin.rtype_ord` (`rstr.py:772-775`) — UniChar variant
    /// uses the `cast_unichar_to_int` op.
    fn rtype_ord(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::UniChar)])?;
        Ok(hop.genop(
            "cast_unichar_to_int",
            vlist,
            GenopResult::LLType(LowLevelType::Signed),
        ))
    }

    /// RPython `BaseCharReprMixin._rtype_method_isxxx(_, llfn, hop)`
    /// dispatch (`rstr.py:516-538`) — UniCharRepr inherits the same
    /// mixin, so the predicate routes go through `cast_unichar_to_int`
    /// per-predicate `ll_unichar_*` helpers.
    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        if let Some((llfn_name, lo, hi)) = match method_name {
            "isdigit" => Some(("ll_unichar_isdigit", 48, 57)),
            "isupper" => Some(("ll_unichar_isupper", 65, 90)),
            "islower" => Some(("ll_unichar_islower", 97, 122)),
            _ => None,
        } {
            return char_predicate_inrange_method(
                hop,
                llfn_name.to_string(),
                LowLevelType::UniChar,
                "cast_unichar_to_int",
                lo,
                hi,
            );
        }

        if let Some((llfn_name, conditions)) = match method_name {
            "isspace" => Some(("ll_unichar_isspace", ISSPACE_CONDITIONS)),
            "isalpha" => Some(("ll_unichar_isalpha", ISALPHA_CONDITIONS)),
            "isalnum" => Some(("ll_unichar_isalnum", ISALNUM_CONDITIONS)),
            _ => None,
        } {
            return char_predicate_or_of_conditions_method(
                hop,
                llfn_name.to_string(),
                LowLevelType::UniChar,
                "cast_unichar_to_int",
                conditions,
            );
        }

        Err(TyperError::message(format!(
            "missing UniCharRepr.rtype_method_{method_name}"
        )))
    }
}

/// RPython `unichar_repr = UniCharRepr()` (`rstr.py:1010`) module-global.
pub fn unichar_repr() -> Arc<UniCharRepr> {
    static REPR: OnceLock<Arc<UniCharRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(UniCharRepr::new())).clone()
}

/// Synthesizes the `ll_unichar_hash(ch)` helper graph (`rstr.py:941-942`):
/// single block, `cast_unichar_to_int(ch) -> hashed` then return.
pub(crate) fn build_ll_unichar_hash_helper_graph(name: &str) -> Result<PyGraph, TyperError> {
    build_ll_charlike_hash_helper_graph(name, LowLevelType::UniChar, "cast_unichar_to_int")
}

// ____________________________________________________________
// pairtype(AbstractCharRepr, AbstractCharRepr) — `rstr.py:740-746`.
// Six comparison ops (eq/ne/lt/le/gt/ge) all dispatch to the
// per-name lloperation `char_<func>`.

/// RPython `_rtype_compare_template(hop, func)` (`rstr.py:750-753`):
///
/// ```python
/// def _rtype_compare_template(hop, func):
///     vlist = hop.inputargs(char_repr, char_repr)
///     return hop.genop('char_' + func, vlist, resulttype=Bool)
/// ```
pub fn pair_char_char_rtype_compare(hop: &HighLevelOp, func: &str) -> RTypeResult {
    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::Char),
        ConvertedTo::LowLevelType(&LowLevelType::Char),
    ])?;
    let opname = format!("char_{func}");
    Ok(hop.genop(&opname, vlist, GenopResult::LLType(LowLevelType::Bool)))
}

// ____________________________________________________________
// pairtype(AbstractUniCharRepr, AbstractUniCharRepr) — `rstr.py:778-784`.
// `rtype_eq` / `rtype_ne` use the lloperations `unichar_eq` /
// `unichar_ne`; `rtype_lt|le|gt|ge` cast both args through
// `cast_unichar_to_int` and dispatch to `int_<func>`.

/// RPython `_rtype_unchr_compare_template(hop, func)` (`rstr.py:789-792`):
///
/// ```python
/// def _rtype_unchr_compare_template(hop, func):
///     vlist = hop.inputargs(unichar_repr, unichar_repr)
///     return hop.genop('unichar_' + func, vlist, resulttype=Bool)
/// ```
pub fn pair_unichar_unichar_rtype_compare_eqne(hop: &HighLevelOp, func: &str) -> RTypeResult {
    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::UniChar),
        ConvertedTo::LowLevelType(&LowLevelType::UniChar),
    ])?;
    let opname = format!("unichar_{func}");
    Ok(hop.genop(&opname, vlist, GenopResult::LLType(LowLevelType::Bool)))
}

/// RPython `_rtype_unchr_compare_template_ord(hop, func)` (`rstr.py:794-800`):
///
/// ```python
/// def _rtype_unchr_compare_template_ord(hop, func):
///     vlist = hop.inputargs(*hop.args_r)
///     vlist2 = []
///     for v in vlist:
///         v = hop.genop('cast_unichar_to_int', [v], resulttype=lltype.Signed)
///         vlist2.append(v)
///     return hop.genop('int_' + func, vlist2, resulttype=Bool)
/// ```
pub fn pair_unichar_unichar_rtype_compare_ord(hop: &HighLevelOp, func: &str) -> RTypeResult {
    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::UniChar),
        ConvertedTo::LowLevelType(&LowLevelType::UniChar),
    ])?;
    let mut vlist2 = Vec::with_capacity(vlist.len());
    for v in vlist {
        let casted = hop
            .genop(
                "cast_unichar_to_int",
                vec![v],
                GenopResult::LLType(LowLevelType::Signed),
            )
            .ok_or_else(|| {
                TyperError::message("cast_unichar_to_int genop did not produce a value")
            })?;
        vlist2.push(casted);
    }
    let opname = format!("int_{func}");
    Ok(hop.genop(&opname, vlist2, GenopResult::LLType(LowLevelType::Bool)))
}

// ____________________________________________________________
// pairtype(AbstractStringRepr, AbstractStringRepr) and
// pairtype(AbstractUnicodeRepr, AbstractUnicodeRepr) — `rstr.py:651-702`.
// Six compare ops (eq/ne/lt/le/gt/ge) all flow through the same body
// modulo the helper-graph identity (`ll_streq` / `ll_strcmp` for
// String, `ll_unicode_eq` / `ll_unicode_cmp` for Unicode).

/// RPython pair compare body shared between
/// `pairtype(AbstractStringRepr, AbstractStringRepr)` and
/// `pairtype(AbstractUnicodeRepr, AbstractUnicodeRepr)` (`rstr.py:661-692`):
///
/// ```python
/// def rtype_eq((r_str1, r_str2), hop):
///     v_str1, v_str2 = hop.inputargs(r_str1.repr, r_str2.repr)
///     return hop.gendirectcall(r_str1.ll.ll_streq, v_str1, v_str2)
///
/// def rtype_ne((r_str1, r_str2), hop):
///     v_str1, v_str2 = hop.inputargs(r_str1.repr, r_str2.repr)
///     vres = hop.gendirectcall(r_str1.ll.ll_streq, v_str1, v_str2)
///     return hop.genop('bool_not', [vres], resulttype=Bool)
///
/// def rtype_lt((r_str1, r_str2), hop):
///     v_str1, v_str2 = hop.inputargs(r_str1.repr, r_str2.repr)
///     vres = hop.gendirectcall(r_str1.ll.ll_strcmp, v_str1, v_str2)
///     return hop.genop('int_lt', [vres, hop.inputconst(Signed, 0)],
///                      resulttype=Bool)
/// # rtype_le / rtype_gt / rtype_ge mirror rtype_lt with their respective
/// # int_<func> ops.
/// ```
///
/// `func` is one of `eq`, `ne`, `lt`, `le`, `gt`, `ge`. `ptr_lltype`
/// is `Ptr(STR)` for String pair, `Ptr(UNICODE)` for Unicode pair.
/// `eq_helper_name` / `cmp_helper_name` are the cache keys for the
/// `ll_streq` / `ll_strcmp` synthesizer registrations (`ll_streq`
/// `ll_strcmp` for String, `ll_unicode_eq` `ll_unicode_cmp` for
/// Unicode), so the two pair surfaces stay distinct in the helper
/// cache even though both lower to structurally identical CFGs
/// (parametrised by `ptr_lltype`).
fn pair_abstract_string_rtype_compare(
    hop: &HighLevelOp,
    func: &str,
    ptr_lltype: LowLevelType,
    eq_helper_name: &str,
    cmp_helper_name: &str,
) -> RTypeResult {
    let r0_arc = hop
        .args_r
        .borrow()
        .first()
        .cloned()
        .flatten()
        .ok_or_else(|| TyperError::message("pair compare: args_r[0] missing"))?;
    let r1_arc = hop
        .args_r
        .borrow()
        .get(1)
        .cloned()
        .flatten()
        .ok_or_else(|| TyperError::message("pair compare: args_r[1] missing"))?;
    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(r0_arc.as_ref()),
        ConvertedTo::Repr(r1_arc.as_ref()),
    ])?;

    match func {
        "eq" => {
            let v_eq = call_ll_streq_helper(hop, vlist, ptr_lltype, eq_helper_name)?;
            Ok(Some(v_eq))
        }
        "ne" => {
            let v_eq = call_ll_streq_helper(hop, vlist, ptr_lltype, eq_helper_name)?;
            Ok(hop.genop(
                "bool_not",
                vec![v_eq],
                GenopResult::LLType(LowLevelType::Bool),
            ))
        }
        "lt" | "le" | "gt" | "ge" => {
            let v_diff = call_ll_strcmp_helper(hop, vlist, ptr_lltype, cmp_helper_name)?;
            let zero = constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed);
            let opname = format!("int_{func}");
            Ok(hop.genop(
                &opname,
                vec![v_diff, zero],
                GenopResult::LLType(LowLevelType::Bool),
            ))
        }
        _ => Err(TyperError::message(format!(
            "pair_abstract_string_rtype_compare unsupported func '{func}'"
        ))),
    }
}

fn call_ll_streq_helper(
    hop: &HighLevelOp,
    vlist: Vec<Hlvalue>,
    ptr_lltype: LowLevelType,
    helper_name: &str,
) -> Result<Hlvalue, TyperError> {
    let helper_name_owned = helper_name.to_string();
    let ptr_for_builder = ptr_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        helper_name_owned.clone(),
        vec![ptr_lltype.clone(), ptr_lltype],
        LowLevelType::Bool,
        move |_rtyper, _args, _result| {
            build_ll_streq_helper_graph(&helper_name_owned, ptr_for_builder)
        },
    )?;
    hop.gendirectcall(&helper, vlist)?
        .ok_or_else(|| TyperError::message("ll_streq gendirectcall produced no value"))
}

fn call_ll_strcmp_helper(
    hop: &HighLevelOp,
    vlist: Vec<Hlvalue>,
    ptr_lltype: LowLevelType,
    helper_name: &str,
) -> Result<Hlvalue, TyperError> {
    let helper_name_owned = helper_name.to_string();
    let ptr_for_builder = ptr_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        helper_name_owned.clone(),
        vec![ptr_lltype.clone(), ptr_lltype],
        LowLevelType::Signed,
        move |_rtyper, _args, _result| {
            build_ll_strcmp_helper_graph(&helper_name_owned, ptr_for_builder)
        },
    )?;
    hop.gendirectcall(&helper, vlist)?
        .ok_or_else(|| TyperError::message("ll_strcmp gendirectcall produced no value"))
}

/// RPython `pairtype(AbstractStringRepr, AbstractStringRepr)` compare
/// dispatch — String pair, all six ops route through `ll_streq` /
/// `ll_strcmp` (rstr.py:651-692). Mirror of
/// [`pair_unichar_unichar_rtype_compare_ord`] for the StringRepr
/// surface.
pub fn pair_string_string_rtype_compare(hop: &HighLevelOp, func: &str) -> RTypeResult {
    pair_abstract_string_rtype_compare(hop, func, STRPTR.clone(), "ll_streq", "ll_strcmp")
}

/// RPython `pairtype(AbstractUnicodeRepr, AbstractUnicodeRepr)`
/// compare dispatch — Unicode pair, all six ops route through
/// `ll_unicode_eq` / `ll_unicode_cmp` (rstr.py:651-692 inherited via
/// `AbstractUnicodeRepr(AbstractStringRepr)`). Helper-graph identity
/// is distinct from the String pair so the two pairs stay separate
/// entries in the helper cache.
pub fn pair_unicode_unicode_rtype_compare(hop: &HighLevelOp, func: &str) -> RTypeResult {
    pair_abstract_string_rtype_compare(
        hop,
        func,
        UNICODEPTR.clone(),
        "ll_unicode_eq",
        "ll_unicode_cmp",
    )
}

/// RPython `pair(AbstractStringRepr, AbstractStringRepr).rtype_add`
/// (`rstr.py:651-659`):
///
/// ```python
/// def rtype_add((r_str1, r_str2), hop):
///     str1_repr = r_str1.repr
///     str2_repr = r_str2.repr
///     if hop.s_result.is_constant():
///         return hop.inputconst(str1_repr, hop.s_result.const)
///     v_str1, v_str2 = hop.inputargs(str1_repr, str2_repr)
///     return hop.gendirectcall(r_str1.ll.ll_strconcat, v_str1, v_str2)
/// rtype_inplace_add = rtype_add
/// ```
///
/// Pyre mirrors the constant-result fast path first, then lowers the
/// non-constant case to `direct_call(ll_strconcat, v_s1, v_s2)`.
/// Helper-cache identity diverges per pair (`ll_strconcat` for STR +
/// STR, `ll_unicode_concat` for UNICODE + UNICODE) so the two helper
/// families do not collide.
fn pair_same_string_rtype_add(
    hop: &HighLevelOp,
    self_repr: &dyn Repr,
    helper_name: &str,
    ptr_lltype: LowLevelType,
) -> RTypeResult {
    if let Some(s_result) = hop.s_result.borrow().as_ref()
        && s_result.is_constant()
    {
        let value = s_result
            .const_()
            .ok_or_else(|| TyperError::message("rtype_add: constant s_result missing const"))?;
        return HighLevelOp::inputconst(ConvertedTo::Repr(self_repr), value)
            .map(Hlvalue::Constant)
            .map(Some);
    }

    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(self_repr),
        ConvertedTo::Repr(self_repr),
    ])?;

    let helper_name_owned = helper_name.to_string();
    let ptr_for_builder = ptr_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        helper_name_owned.clone(),
        vec![ptr_lltype.clone(), ptr_lltype.clone()],
        ptr_lltype,
        move |_rtyper, _args, _result| {
            build_ll_strconcat_helper_graph(&helper_name_owned, ptr_for_builder)
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// Abstract pairtype entry used when dispatch reaches
/// `pair(AbstractStringRepr, AbstractStringRepr)` through the repr MRO.
pub fn pair_abstract_string_rtype_add(
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &HighLevelOp,
) -> RTypeResult {
    let helper_name = match r1.repr_class_id() {
        super::pairtype::ReprClassId::StringRepr => "ll_strconcat",
        super::pairtype::ReprClassId::UnicodeRepr => "ll_unicode_concat",
        other => {
            return Err(TyperError::message(format!(
                "rtype_add: expected string-family left repr, got {other:?}"
            )));
        }
    };
    let ptr_lltype = r1.lowleveltype().clone();
    if r2.lowleveltype() != &ptr_lltype {
        return Err(TyperError::message(format!(
            "rtype_add: mixed string-family concat requires conversion before lowering: left={:?}, right={:?}",
            r1.lowleveltype(),
            r2.lowleveltype()
        )));
    }
    pair_same_string_rtype_add(hop, r1, helper_name, ptr_lltype)
}

/// `pair(StringRepr, StringRepr).rtype_add` — STR surface.
pub fn pair_string_string_rtype_add(hop: &HighLevelOp) -> RTypeResult {
    pair_same_string_rtype_add(hop, string_repr().as_ref(), "ll_strconcat", STRPTR.clone())
}

/// `pair(UnicodeRepr, UnicodeRepr).rtype_add` — UNICODE surface.
pub fn pair_unicode_unicode_rtype_add(hop: &HighLevelOp) -> RTypeResult {
    pair_same_string_rtype_add(
        hop,
        unicode_repr().as_ref(),
        "ll_unicode_concat",
        UNICODEPTR.clone(),
    )
}

/// Per-pair helper-name bundle for
/// [`pair_abstract_string_int_rtype_getitem`]. Mirrors the
/// `r_str.ll.ll_stritem*` lookup table at upstream `rstr.py:619-627`:
/// the four helpers differ along (checkidx × nonneg) axes, and the
/// String/Unicode pair surfaces choose between two distinct
/// helper-cache families (`ll_stritem*` vs `ll_unicode_stritem*`).
struct StritemHelperNames {
    nonneg: &'static str,
    neg: &'static str,
    nonneg_checked: &'static str,
    neg_checked: &'static str,
}

/// RPython `pair(AbstractStringRepr, IntegerRepr).rtype_getitem`
/// (`rstr.py:614-632`):
///
/// ```python
/// def rtype_getitem((r_str, r_int), hop, checkidx=False):
///     string_repr = r_str.repr
///     v_str, v_index = hop.inputargs(string_repr, Signed)
///     if checkidx:
///         if hop.args_s[1].nonneg:
///             llfn = r_str.ll.ll_stritem_nonneg_checked
///         else:
///             llfn = r_str.ll.ll_stritem_checked
///     else:
///         if hop.args_s[1].nonneg:
///             llfn = r_str.ll.ll_stritem_nonneg
///         else:
///             llfn = r_str.ll.ll_stritem
///     if checkidx:
///         hop.exception_is_here()
///     else:
///         hop.exception_cannot_occur()
///     return hop.gendirectcall(llfn, v_str, v_index)
/// ```
///
/// All four (checkidx × nonneg) combinations are wired:
/// - `checkidx=False, nonneg=True`  → `ll_stritem_nonneg`
/// - `checkidx=False, nonneg=False` → `ll_stritem` (inlines neg-fix +
///   sub-helper direct_call to `ll_stritem_nonneg`)
/// - `checkidx=True,  nonneg=True`  → `ll_stritem_nonneg_checked`
///   (raise IndexError when `i >= length`)
/// - `checkidx=True,  nonneg=False` → `ll_stritem_checked`
///   (full neg-fix + bound-check, raise IndexError on out-of-range)
fn pair_abstract_string_int_rtype_getitem(
    hop: &HighLevelOp,
    self_repr: &dyn Repr,
    helper_names: StritemHelperNames,
    ptr_lltype: LowLevelType,
    elem_lltype: LowLevelType,
    checkidx: bool,
) -> RTypeResult {
    use crate::annotator::model::SomeValue;

    let s1 = hop
        .args_s
        .borrow()
        .get(1)
        .cloned()
        .ok_or_else(|| TyperError::message("rtype_getitem: args_s[1] missing"))?;
    let nonneg = match &s1 {
        SomeValue::Integer(i) => i.nonneg,
        other => {
            return Err(TyperError::message(format!(
                "rtype_getitem: args_s[1] must be SomeInteger, got {other:?}"
            )));
        }
    };

    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(self_repr),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
    ])?;

    // rstr.py:629-632 — `hop.exception_is_here()` for the checkidx
    // branch (which raises IndexError from inside
    // ll_stritem*_checked); `hop.exception_cannot_occur()` otherwise.
    if checkidx {
        hop.exception_is_here()?;
    } else {
        hop.exception_cannot_occur()?;
    }

    let helper = match (checkidx, nonneg) {
        (false, true) => {
            let helper_name_owned = helper_names.nonneg.to_string();
            let ptr_for_builder = ptr_lltype.clone();
            hop.rtyper.lowlevel_helper_function_with_builder(
                helper_name_owned.clone(),
                vec![ptr_lltype, LowLevelType::Signed],
                elem_lltype,
                move |_rtyper, _args, _result| {
                    build_ll_stritem_nonneg_helper_graph(&helper_name_owned, ptr_for_builder)
                },
            )?
        }
        (false, false) => {
            let neg_name_owned = helper_names.neg.to_string();
            let nonneg_name_owned = helper_names.nonneg.to_string();
            let ptr_for_builder = ptr_lltype.clone();
            hop.rtyper.lowlevel_helper_function_with_builder(
                neg_name_owned.clone(),
                vec![ptr_lltype, LowLevelType::Signed],
                elem_lltype,
                move |rtyper_inner, _args, _result| {
                    build_ll_stritem_helper_graph(
                        rtyper_inner,
                        &neg_name_owned,
                        ptr_for_builder,
                        &nonneg_name_owned,
                    )
                },
            )?
        }
        (true, true) => {
            let checked_name_owned = helper_names.nonneg_checked.to_string();
            let nonneg_name_owned = helper_names.nonneg.to_string();
            let ptr_for_builder = ptr_lltype.clone();
            hop.rtyper.lowlevel_helper_function_with_builder(
                checked_name_owned.clone(),
                vec![ptr_lltype, LowLevelType::Signed],
                elem_lltype,
                move |rtyper_inner, _args, _result| {
                    build_ll_stritem_nonneg_checked_helper_graph(
                        rtyper_inner,
                        &checked_name_owned,
                        ptr_for_builder,
                        &nonneg_name_owned,
                    )
                },
            )?
        }
        (true, false) => {
            let checked_name_owned = helper_names.neg_checked.to_string();
            let nonneg_name_owned = helper_names.nonneg.to_string();
            let ptr_for_builder = ptr_lltype.clone();
            hop.rtyper.lowlevel_helper_function_with_builder(
                checked_name_owned.clone(),
                vec![ptr_lltype, LowLevelType::Signed],
                elem_lltype,
                move |rtyper_inner, _args, _result| {
                    build_ll_stritem_checked_helper_graph(
                        rtyper_inner,
                        &checked_name_owned,
                        ptr_for_builder,
                        &nonneg_name_owned,
                    )
                },
            )?
        }
    };
    hop.gendirectcall(&helper, vlist)
}

/// Helper-name bundles for the STR and UNICODE pair surfaces. Distinct
/// caches keep the two helper families from colliding in the helper
/// registry (`ll_stritem` vs `ll_unicode_stritem`, etc.).
const STR_STRITEM_HELPER_NAMES: StritemHelperNames = StritemHelperNames {
    nonneg: "ll_stritem_nonneg",
    neg: "ll_stritem",
    nonneg_checked: "ll_stritem_nonneg_checked",
    neg_checked: "ll_stritem_checked",
};

const UNICODE_STRITEM_HELPER_NAMES: StritemHelperNames = StritemHelperNames {
    nonneg: "ll_unicode_stritem_nonneg",
    neg: "ll_unicode_stritem",
    nonneg_checked: "ll_unicode_stritem_nonneg_checked",
    neg_checked: "ll_unicode_stritem_checked",
};

/// `pair(StringRepr, IntegerRepr).rtype_getitem` — STR surface.
pub fn pair_string_int_rtype_getitem(hop: &HighLevelOp) -> RTypeResult {
    pair_abstract_string_int_rtype_getitem(
        hop,
        string_repr().as_ref(),
        STR_STRITEM_HELPER_NAMES,
        STRPTR.clone(),
        LowLevelType::Char,
        /* checkidx */ false,
    )
}

/// `pair(StringRepr, IntegerRepr).rtype_getitem_idx` — STR surface,
/// rstr.py:634 dispatches via `pair(r_str, r_int).rtype_getitem(hop,
/// checkidx=True)`.
pub fn pair_string_int_rtype_getitem_idx(hop: &HighLevelOp) -> RTypeResult {
    pair_abstract_string_int_rtype_getitem(
        hop,
        string_repr().as_ref(),
        STR_STRITEM_HELPER_NAMES,
        STRPTR.clone(),
        LowLevelType::Char,
        /* checkidx */ true,
    )
}

/// `pair(UnicodeRepr, IntegerRepr).rtype_getitem` — UNICODE surface.
pub fn pair_unicode_int_rtype_getitem(hop: &HighLevelOp) -> RTypeResult {
    pair_abstract_string_int_rtype_getitem(
        hop,
        unicode_repr().as_ref(),
        UNICODE_STRITEM_HELPER_NAMES,
        UNICODEPTR.clone(),
        LowLevelType::UniChar,
        /* checkidx */ false,
    )
}

/// `pair(UnicodeRepr, IntegerRepr).rtype_getitem_idx` — UNICODE surface.
pub fn pair_unicode_int_rtype_getitem_idx(hop: &HighLevelOp) -> RTypeResult {
    pair_abstract_string_int_rtype_getitem(
        hop,
        unicode_repr().as_ref(),
        UNICODE_STRITEM_HELPER_NAMES,
        UNICODEPTR.clone(),
        LowLevelType::UniChar,
        /* checkidx */ true,
    )
}

/// RPython `pair(AbstractCharRepr|AbstractUniCharRepr, IntegerRepr).rtype_getitem`
/// (`rstr.py:723-730`):
///
/// ```python
/// def rtype_getitem((r_char, r_int), hop, checkidx=False):
///     if hop.args_s[1].is_constant() and hop.args_s[1].const == 0:
///         hop.exception_cannot_occur()
///         return hop.inputarg(hop.r_result, arg=0)
///     # do the slow thing (super would be cool)
///     paircls = pairtype(AbstractStringRepr, IntegerRepr)
///     return paircls.rtype_getitem(pair(r_char, r_int), hop, checkidx)
/// ```
///
/// The constant-0 branch returns the char itself via
/// `hop.inputarg(r_result, 0)`. Other indexes fall through exactly as
/// upstream: convert the char to a one-character STR/UNICODE via
/// `ll_chr2str`, then reuse the AbstractStringRepr+IntegerRepr getitem
/// dispatcher (including the checked `getitem_idx` variant).
fn pair_abstract_char_int_rtype_getitem(hop: &HighLevelOp, checkidx: bool) -> RTypeResult {
    let s1 = hop
        .args_s
        .borrow()
        .get(1)
        .cloned()
        .ok_or_else(|| TyperError::message("rtype_getitem (Char, Int): args_s[1] missing"))?;

    if matches!(s1.const_(), Some(ConstValue::Int(0))) {
        // rstr.py:725 — `hop.exception_cannot_occur()` precedes the inputarg.
        hop.exception_cannot_occur()?;

        // rstr.py:726 — `return hop.inputarg(hop.r_result, arg=0)` —
        // identity-coerce the char arg into the result repr (CharRepr or
        // UniCharRepr). The result repr equals args_r[0] for the constant-0
        // case since `c[0] == c`.
        let r_result =
            hop.r_result.borrow().clone().ok_or_else(|| {
                TyperError::message("rtype_getitem (Char, Int): r_result missing")
            })?;
        return hop
            .inputarg(ConvertedTo::Repr(r_result.as_ref()), 0)
            .map(Some);
    }

    let r_char = hop
        .args_r
        .borrow()
        .first()
        .cloned()
        .flatten()
        .ok_or_else(|| TyperError::message("rtype_getitem (Char, Int): args_r[0] missing"))?;
    match r_char.repr_class_id() {
        super::pairtype::ReprClassId::CharRepr => pair_abstract_string_int_rtype_getitem(
            hop,
            string_repr().as_ref(),
            STR_STRITEM_HELPER_NAMES,
            STRPTR.clone(),
            LowLevelType::Char,
            checkidx,
        ),
        super::pairtype::ReprClassId::UniCharRepr => pair_abstract_string_int_rtype_getitem(
            hop,
            unicode_repr().as_ref(),
            UNICODE_STRITEM_HELPER_NAMES,
            UNICODEPTR.clone(),
            LowLevelType::UniChar,
            checkidx,
        ),
        other => Err(TyperError::message(format!(
            "rtype_getitem (Char, Int): unsupported char repr {other:?}"
        ))),
    }
}

/// `pair(CharRepr, IntegerRepr).rtype_getitem` — Char surface.
pub fn pair_char_int_rtype_getitem(hop: &HighLevelOp) -> RTypeResult {
    pair_abstract_char_int_rtype_getitem(hop, false)
}

/// `pair(CharRepr, IntegerRepr).rtype_getitem_idx` — checked Char surface.
pub fn pair_char_int_rtype_getitem_idx(hop: &HighLevelOp) -> RTypeResult {
    pair_abstract_char_int_rtype_getitem(hop, true)
}

/// `pair(UniCharRepr, IntegerRepr).rtype_getitem` — UniChar surface.
pub fn pair_unichar_int_rtype_getitem(hop: &HighLevelOp) -> RTypeResult {
    pair_abstract_char_int_rtype_getitem(hop, false)
}

/// `pair(UniCharRepr, IntegerRepr).rtype_getitem_idx` — checked UniChar surface.
pub fn pair_unichar_int_rtype_getitem_idx(hop: &HighLevelOp) -> RTypeResult {
    pair_abstract_char_int_rtype_getitem(hop, true)
}

/// RPython `AbstractStringRepr.rtype_method_isdigit(self, hop)`
/// (`rstr.py:253-257`):
///
/// ```python
/// def rtype_method_isdigit(self, hop):
///     string_repr = hop.args_r[0].repr
///     [v_str] = hop.inputargs(string_repr)
///     hop.exception_cannot_occur()
///     return hop.gendirectcall(self.ll.ll_isdigit, v_str)
/// ```
///
/// Pyre lowers `ll_isdigit` to a chars-loop helper that
/// `direct_call`s the matching per-char `ll_char_isdigit` /
/// `ll_unichar_isdigit` predicate (rstr.py:891-922 inrange shape,
/// `48..=57`).
fn rtype_abstract_string_method_isdigit(
    self_repr: &dyn Repr,
    hop: &HighLevelOp,
    helper_name: &str,
    inner_predicate_name: &str,
    ptr_lltype: LowLevelType,
    elem_lltype: LowLevelType,
    cast_op: &'static str,
) -> RTypeResult {
    rtype_abstract_string_method_isxxx(
        self_repr,
        hop,
        helper_name,
        inner_predicate_name,
        ptr_lltype,
        elem_lltype,
        cast_op,
        StringIsxxxPredicate::InRange(48, 57),
    )
}

#[derive(Clone, Copy)]
enum StringIsxxxPredicate {
    InRange(i64, i64),
    OrConditions(&'static [CharCondition]),
}

fn rtype_abstract_string_method_isxxx(
    self_repr: &dyn Repr,
    hop: &HighLevelOp,
    helper_name: &str,
    inner_predicate_name: &str,
    ptr_lltype: LowLevelType,
    elem_lltype: LowLevelType,
    cast_op: &'static str,
    predicate: StringIsxxxPredicate,
) -> RTypeResult {
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(self_repr)])?;
    // rstr.py:256 — hop.exception_cannot_occur() before gendirectcall.
    hop.exception_cannot_occur()?;

    let helper_name_owned = helper_name.to_string();
    let inner_name_owned = inner_predicate_name.to_string();
    let ptr_for_builder = ptr_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        helper_name_owned.clone(),
        vec![ptr_lltype],
        LowLevelType::Bool,
        move |rtyper_inner, _args, _result| {
            // Inner helper closure consumes once at sub-helper miss.
            let inner_for_predicate = inner_name_owned.clone();
            let elem_for_predicate = elem_lltype.clone();
            build_ll_string_isxxx_helper_graph(
                rtyper_inner,
                &helper_name_owned,
                ptr_for_builder,
                &inner_name_owned,
                move |_inner_name| match predicate {
                    StringIsxxxPredicate::InRange(lo, hi) => {
                        build_ll_charlike_predicate_inrange_helper_graph(
                            &inner_for_predicate,
                            elem_for_predicate,
                            cast_op,
                            lo,
                            hi,
                        )
                    }
                    StringIsxxxPredicate::OrConditions(conditions) => {
                        build_ll_charlike_or_of_conditions_helper_graph(
                            &inner_for_predicate,
                            elem_for_predicate,
                            cast_op,
                            conditions,
                        )
                    }
                },
            )
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// RPython `AbstractStringRepr.rtype_method_upper/lower`
/// (`rstr.py:241-251`): byte strings dispatch to
/// `LLHelpers.ll_upper/lower`; UnicodeRepr overrides these methods and
/// never reaches this helper.
fn rtype_abstract_string_method_casefold(
    self_repr: &dyn Repr,
    hop: &HighLevelOp,
    helper_name: &str,
    inner_helper_name: &str,
    lo: i64,
    hi: i64,
    offset: i64,
) -> RTypeResult {
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(self_repr)])?;
    // rstr.py:244/250 — hop.exception_cannot_occur() before gendirectcall.
    hop.exception_cannot_occur()?;

    let helper_name_owned = helper_name.to_string();
    let inner_name_owned = inner_helper_name.to_string();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        helper_name_owned.clone(),
        vec![STRPTR.clone()],
        STRPTR.clone(),
        move |rtyper_inner, _args, _result| {
            let inner_for_builder = inner_name_owned.clone();
            build_ll_string_casefold_helper_graph(
                rtyper_inner,
                &helper_name_owned,
                STRPTR.clone(),
                &inner_name_owned,
                move |_inner_name| {
                    build_ll_char_case_fold_helper_graph(&inner_for_builder, lo, hi, offset)
                },
            )
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// RPython `AbstractStringRepr.rtype_method_replace`
/// (`rstr.py:363-369`): currently only the `(str-or-unicode, char,
/// char)` form is supported upstream here; non-char arguments raise
/// `TyperError('replace only works for char args')`.
fn rtype_abstract_string_method_replace(
    self_repr: &dyn Repr,
    hop: &HighLevelOp,
    helper_name: &str,
    ptr_lltype: LowLevelType,
    elem_lltype: LowLevelType,
) -> RTypeResult {
    use super::pairtype::ReprClassId;

    let expected = match elem_lltype {
        LowLevelType::Char => ReprClassId::CharRepr,
        LowLevelType::UniChar => ReprClassId::UniCharRepr,
        ref other => {
            return Err(TyperError::message(format!(
                "rtype_method_replace unsupported char element type {other:?}"
            )));
        }
    };
    for idx in [1usize, 2usize] {
        let r_arg = hop
            .args_r
            .borrow()
            .get(idx)
            .cloned()
            .flatten()
            .ok_or_else(|| {
                TyperError::message(format!("rtype_method_replace: args_r[{idx}] missing"))
            })?;
        if r_arg.repr_class_id() != expected {
            return Err(TyperError::message("replace only works for char args"));
        }
    }

    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(self_repr),
        ConvertedTo::LowLevelType(&elem_lltype),
        ConvertedTo::LowLevelType(&elem_lltype),
    ])?;
    // rstr.py:368 — hop.exception_cannot_occur() before gendirectcall.
    hop.exception_cannot_occur()?;

    let helper_name_owned = helper_name.to_string();
    let ptr_for_builder = ptr_lltype.clone();
    let elem_for_builder = elem_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        helper_name_owned.clone(),
        vec![ptr_lltype.clone(), elem_lltype.clone(), elem_lltype],
        ptr_lltype,
        move |_rtyper, _args, _result| {
            build_ll_replace_chr_chr_helper_graph(
                &helper_name_owned,
                ptr_for_builder,
                elem_for_builder,
            )
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

// ____________________________________________________________
// AbstractStringRepr / AbstractUnicodeRepr method dispatch
// (`rstr.py:134-449`). Pyre lands the methods slice-by-slice; today
// only `startswith` (rstr.py:134-145) lowers in the
// `args_r[1]` is-a-String/Unicode branch. The Char-side branch
// (`ll_startswith_char` for char-keyed startswith) is deferred since
// its synthesizer has not landed yet.

/// RPython `AbstractStringRepr.rtype_method_startswith(self, hop)`
/// (`rstr.py:134-145`):
///
/// ```python
/// def rtype_method_startswith(self, hop):
///     str1_repr = hop.args_r[0].repr
///     str2_repr = hop.args_r[1]
///     v_str = hop.inputarg(str1_repr, arg=0)
///     if str2_repr == str2_repr.char_repr:
///         v_value = hop.inputarg(str2_repr.char_repr, arg=1)
///         fn = self.ll.ll_startswith_char
///     else:
///         v_value = hop.inputarg(str2_repr, arg=1)
///         fn = self.ll.ll_startswith
///     hop.exception_cannot_occur()
///     return hop.gendirectcall(fn, v_str, v_value)
/// ```
///
/// Pyre slice today: only the String/String (or Unicode/Unicode)
/// branch — `args_r[1].repr_class_id()` must equal `self_repr`'s
/// id. The Char-side branch (`ll_startswith_char`) is a follow-up.
fn rtype_abstract_string_method_startswith(
    self_repr: &dyn Repr,
    hop: &HighLevelOp,
    helper_name: &str,
    char_helper_name: &str,
    ptr_lltype: LowLevelType,
    elem_lltype: LowLevelType,
) -> RTypeResult {
    use super::pairtype::ReprClassId;

    let r1_arc = hop
        .args_r
        .borrow()
        .get(1)
        .cloned()
        .flatten()
        .ok_or_else(|| TyperError::message("rtype_method_startswith: args_r[1] missing"))?;
    // Char-side branch (`ll_startswith_char`, rstr.py:138-141): when
    // args_r[1] is a CharRepr/UniCharRepr, gendirectcall the
    // single-char helper instead of `ll_startswith`.
    //
    // `elem_lltype` is the chars-Array element type of the self repr
    // (Char for STR, UniChar for UNICODE). Both the helper signature
    // (`vec![ptr_lltype, elem_lltype]`) and the second-arg conversion
    // (`ConvertedTo::LowLevelType(&elem_lltype)`) draw from this single
    // source so the dispatcher and `build_ll_startswith_char_helper_graph`
    // agree on the `ch` lltype. A mismatched pair (e.g.
    // `str.startswith(unichar)`) surfaces immediately at
    // `hop.inputargs(...)` rather than later as a helper-cache or
    // gendirectcall mismatch.
    if matches!(
        r1_arc.repr_class_id(),
        ReprClassId::CharRepr | ReprClassId::UniCharRepr
    ) {
        let vlist = hop.inputargs(vec![
            ConvertedTo::Repr(self_repr),
            ConvertedTo::LowLevelType(&elem_lltype),
        ])?;
        // rstr.py:144 — hop.exception_cannot_occur() before gendirectcall.
        hop.exception_cannot_occur()?;
        let char_name_owned = char_helper_name.to_string();
        let ptr_for_builder = ptr_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            char_name_owned.clone(),
            vec![ptr_lltype, elem_lltype],
            LowLevelType::Bool,
            move |_rtyper, _args, _result| {
                build_ll_startswith_char_helper_graph(&char_name_owned, ptr_for_builder)
            },
        )?;
        return hop.gendirectcall(&helper, vlist);
    }

    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(self_repr),
        ConvertedTo::Repr(r1_arc.as_ref()),
    ])?;
    // rstr.py:144 — hop.exception_cannot_occur() before gendirectcall.
    hop.exception_cannot_occur()?;

    let helper_name_owned = helper_name.to_string();
    let ptr_for_builder = ptr_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        helper_name_owned.clone(),
        vec![ptr_lltype.clone(), ptr_lltype],
        LowLevelType::Bool,
        move |_rtyper, _args, _result| {
            build_ll_startswith_helper_graph(&helper_name_owned, ptr_for_builder)
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// RPython `AbstractStringRepr.rtype_method_endswith(self, hop)`
/// (`rstr.py:147-158`):
///
/// ```python
/// def rtype_method_endswith(self, hop):
///     str1_repr = hop.args_r[0].repr
///     str2_repr = hop.args_r[1]
///     v_str = hop.inputarg(str1_repr, arg=0)
///     if str2_repr == str2_repr.char_repr:
///         v_value = hop.inputarg(str2_repr.char_repr, arg=1)
///         fn = self.ll.ll_endswith_char
///     else:
///         v_value = hop.inputarg(str2_repr, arg=1)
///         fn = self.ll.ll_endswith
///     hop.exception_cannot_occur()
///     return hop.gendirectcall(fn, v_str, v_value)
/// ```
///
/// Mirror of [`rtype_abstract_string_method_startswith`] — only the
/// helper-graph builder (`build_ll_endswith_helper_graph`) and the
/// char-side helper name (`ll_endswith_char`) differ. The Char-side
/// branch is deferred just like startswith.
fn rtype_abstract_string_method_endswith(
    self_repr: &dyn Repr,
    hop: &HighLevelOp,
    helper_name: &str,
    char_helper_name: &str,
    ptr_lltype: LowLevelType,
    elem_lltype: LowLevelType,
) -> RTypeResult {
    use super::pairtype::ReprClassId;

    let r1_arc = hop
        .args_r
        .borrow()
        .get(1)
        .cloned()
        .flatten()
        .ok_or_else(|| TyperError::message("rtype_method_endswith: args_r[1] missing"))?;
    // See `rtype_abstract_string_method_startswith` for the rationale on
    // sourcing `elem_lltype` from the self repr (ptr_lltype.chars elem)
    // rather than args_r[1].
    if matches!(
        r1_arc.repr_class_id(),
        ReprClassId::CharRepr | ReprClassId::UniCharRepr
    ) {
        let vlist = hop.inputargs(vec![
            ConvertedTo::Repr(self_repr),
            ConvertedTo::LowLevelType(&elem_lltype),
        ])?;
        // rstr.py:157 — hop.exception_cannot_occur() before gendirectcall.
        hop.exception_cannot_occur()?;
        let char_name_owned = char_helper_name.to_string();
        let ptr_for_builder = ptr_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            char_name_owned.clone(),
            vec![ptr_lltype, elem_lltype],
            LowLevelType::Bool,
            move |_rtyper, _args, _result| {
                build_ll_endswith_char_helper_graph(&char_name_owned, ptr_for_builder)
            },
        )?;
        return hop.gendirectcall(&helper, vlist);
    }

    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(self_repr),
        ConvertedTo::Repr(r1_arc.as_ref()),
    ])?;
    // rstr.py:157 — hop.exception_cannot_occur() before gendirectcall.
    hop.exception_cannot_occur()?;

    let helper_name_owned = helper_name.to_string();
    let ptr_for_builder = ptr_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        helper_name_owned.clone(),
        vec![ptr_lltype.clone(), ptr_lltype],
        LowLevelType::Bool,
        move |_rtyper, _args, _result| {
            build_ll_endswith_helper_graph(&helper_name_owned, ptr_for_builder)
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

#[derive(Clone, Copy)]
enum FindLikeCharKind {
    Find,
    RFind,
    Count,
}

/// Char-needle slice of RPython `rtype_method_find` /
/// `rtype_method_count` (`rstr.py:160-189`, `:191-207`).
///
/// Full PyPy parity also accepts string-family substring needles and
/// routes to `ll_find` / `ll_rfind` / `ll_count`. The current helper
/// graph set only contains the single-char helpers; non-char needles
/// therefore fail closed instead of pretending the substring branch was
/// ported.
fn rtype_abstract_string_method_findlike_char(
    self_repr: &dyn Repr,
    hop: &HighLevelOp,
    helper_name: &str,
    kind: FindLikeCharKind,
    ptr_lltype: LowLevelType,
    elem_lltype: LowLevelType,
) -> RTypeResult {
    use super::pairtype::ReprClassId;

    let method_name = match kind {
        FindLikeCharKind::Find => "find",
        FindLikeCharKind::RFind => "rfind",
        FindLikeCharKind::Count => "count",
    };
    let r1_arc = hop
        .args_r
        .borrow()
        .get(1)
        .cloned()
        .flatten()
        .ok_or_else(|| {
            TyperError::message(format!("rtype_method_{method_name}: args_r[1] missing"))
        })?;
    if !matches!(
        r1_arc.repr_class_id(),
        ReprClassId::CharRepr | ReprClassId::UniCharRepr
    ) {
        return Err(TyperError::message(format!(
            "rtype_method_{method_name}: substring helper is not ported yet"
        )));
    }
    let nb_args = hop.nb_args();
    if !(2..=4).contains(&nb_args) {
        return Err(TyperError::message(format!(
            "rtype_method_{method_name}: expected 2 to 4 arguments, got {nb_args}"
        )));
    }

    let v_str = hop.inputarg(ConvertedTo::Repr(self_repr), 0)?;
    let v_ch = hop.inputarg(ConvertedTo::LowLevelType(&elem_lltype), 1)?;
    let v_start = if nb_args >= 3 {
        hop.inputarg(ConvertedTo::LowLevelType(&LowLevelType::Signed), 2)?
    } else {
        constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed)
    };
    let v_end = if nb_args >= 4 {
        hop.inputarg(ConvertedTo::LowLevelType(&LowLevelType::Signed), 3)?
    } else {
        let strlen_name = if ptr_lltype == STRPTR.clone() {
            "ll_strlen"
        } else {
            "ll_unicode_strlen"
        };
        let strlen_name_owned = strlen_name.to_string();
        let ptr_for_builder = ptr_lltype.clone();
        let strlen = hop.rtyper.lowlevel_helper_function_with_builder(
            strlen_name_owned.clone(),
            vec![ptr_lltype.clone()],
            LowLevelType::Signed,
            move |_rtyper, _args, _result| {
                build_ll_strlen_helper_graph(&strlen_name_owned, ptr_for_builder)
            },
        )?;
        hop.gendirectcall(&strlen, vec![v_str.clone()])?
            .ok_or_else(|| TyperError::message("ll_strlen returned Void"))?
    };

    hop.exception_cannot_occur()?;
    let helper_name_owned = helper_name.to_string();
    let ptr_for_builder = ptr_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        helper_name_owned.clone(),
        vec![
            ptr_lltype.clone(),
            elem_lltype,
            LowLevelType::Signed,
            LowLevelType::Signed,
        ],
        LowLevelType::Signed,
        move |_rtyper, _args, _result| match kind {
            FindLikeCharKind::Find => {
                build_ll_find_char_helper_graph(&helper_name_owned, ptr_for_builder)
            }
            FindLikeCharKind::RFind => {
                build_ll_rfind_char_helper_graph(&helper_name_owned, ptr_for_builder)
            }
            FindLikeCharKind::Count => {
                build_ll_count_char_helper_graph(&helper_name_owned, ptr_for_builder)
            }
        },
    )?;
    hop.gendirectcall(&helper, vec![v_str, v_ch, v_start, v_end])
}

// ____________________________________________________________
// Shared single-cast hash helper synthesizer — used by both CharRepr
// and UniCharRepr since their helper graphs are structurally identical
// modulo the cast op + arg lltype.

/// Shared `rtype_method_<predicate>` wrapper for the inrange-pattern
/// helpers (`isdigit/isupper/islower`). RPython
/// `BaseCharReprMixin._rtype_method_isxxx(_, llfn, hop)`
/// (`rstr.py:516-520`):
///
/// ```python
/// def _rtype_method_isxxx(_, llfn, hop):
///     repr = hop.args_r[0].char_repr
///     vlist = hop.inputargs(repr)
///     hop.exception_cannot_occur()
///     return hop.gendirectcall(llfn, vlist[0])
/// ```
fn char_predicate_inrange_method(
    hop: &HighLevelOp,
    llfn_name: String,
    arg_lltype: LowLevelType,
    cast_op: &'static str,
    lo: i64,
    hi: i64,
) -> RTypeResult {
    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&arg_lltype)])?;
    hop.exception_cannot_occur()?;
    let helper_name = llfn_name.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        llfn_name,
        vec![arg_lltype.clone()],
        LowLevelType::Bool,
        move |_rtyper, _args, _result| {
            build_ll_charlike_predicate_inrange_helper_graph(
                &helper_name,
                arg_lltype.clone(),
                cast_op,
                lo,
                hi,
            )
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// Shared `rtype_method_<predicate>` wrapper for `isspace/isalpha/
/// isalnum` — same call-site shape as `char_predicate_inrange_method`
/// but the helper synthesizer chains an OR of conditions (see
/// [`build_ll_charlike_or_of_conditions_helper_graph`]).
fn char_predicate_or_of_conditions_method(
    hop: &HighLevelOp,
    llfn_name: String,
    arg_lltype: LowLevelType,
    cast_op: &'static str,
    conditions: &'static [CharCondition],
) -> RTypeResult {
    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&arg_lltype)])?;
    hop.exception_cannot_occur()?;
    let helper_name = llfn_name.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        llfn_name,
        vec![arg_lltype.clone()],
        LowLevelType::Bool,
        move |_rtyper, _args, _result| {
            build_ll_charlike_or_of_conditions_helper_graph(
                &helper_name,
                arg_lltype.clone(),
                cast_op,
                conditions,
            )
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// Shared `rtype_method_<predicate>` wrapper for `lower/upper`. Mirrors
/// `AbstractCharRepr.rtype_method_lower/upper` (`rstr.py:542-552`):
///
/// ```python
/// def rtype_method_lower(self, hop):
///     char_repr = hop.args_r[0].char_repr
///     v_chr, = hop.inputargs(char_repr)
///     hop.exception_cannot_occur()
///     return hop.gendirectcall(self.ll.ll_lower_char, v_chr)
/// ```
fn char_case_fold_method(
    hop: &HighLevelOp,
    llfn_name: String,
    lo: i64,
    hi: i64,
    offset: i64,
) -> RTypeResult {
    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Char)])?;
    hop.exception_cannot_occur()?;
    let helper_name = llfn_name.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        llfn_name,
        vec![LowLevelType::Char],
        LowLevelType::Char,
        move |_rtyper, _args, _result| {
            build_ll_char_case_fold_helper_graph(&helper_name, lo, hi, offset)
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

// rstr.py:886-912 — predicate condition tables.
// `ll_char_isspace`: `c == 32 or 9 <= c <= 13`.
const ISSPACE_CONDITIONS: &[CharCondition] =
    &[CharCondition::Eq(32), CharCondition::InRange(9, 13)];
// `ll_char_isalpha`: `c >= 97 ? c <= 122 : 65 <= c <= 90`.
// Equivalent OR form: `97 <= c <= 122 or 65 <= c <= 90`.
const ISALPHA_CONDITIONS: &[CharCondition] = &[
    CharCondition::InRange(97, 122),
    CharCondition::InRange(65, 90),
];
// `ll_char_isalnum`: digit or upper-alpha or lower-alpha (rstr.py:903-912 nested form).
const ISALNUM_CONDITIONS: &[CharCondition] = &[
    CharCondition::InRange(48, 57),
    CharCondition::InRange(65, 90),
    CharCondition::InRange(97, 122),
];

/// One condition in an OR-of-predicates check (`ll_char_isspace`,
/// `ll_char_isalpha`, `ll_char_isalnum`, ...).
///
/// - `Eq(n)` — `ord(ch) == n`. One check block (1 `int_eq`).
/// - `InRange(lo, hi)` — `lo <= ord(ch) <= hi`. Two check blocks
///   (`int_ge` then `int_le`).
#[derive(Clone, Copy, Debug)]
enum CharCondition {
    Eq(i64),
    InRange(i64, i64),
}

/// Synthesizes a `ll_<charlike>_<predicate>(ch) -> Bool` helper graph
/// whose body is a left-to-right short-circuit `c0 OR c1 OR ...` over
/// per-condition checks against `cast_<arg>_to_int(ch)`. Mirrors
/// RPython source-level short-circuit `or` (`rstr.py:886-912`).
///
/// CFG layout:
/// - **start**: `c = cast(ch)`, then evaluate `predicates[0]`
///   (`int_eq` or `int_ge`).
/// - For each `predicates[i]`, on True branch return `Bool(true)`;
///   on False branch fall through to `predicates[i+1]`'s entry
///   block (taking `c` as link arg) or, for the last predicate,
///   return `Bool(false)`.
/// - `InRange(lo, hi)` predicates have a secondary check block
///   `block_check_hi` for the `int_le(c, hi)` test.
fn build_ll_charlike_or_of_conditions_helper_graph(
    name: &str,
    arg_lltype: LowLevelType,
    cast_op: &'static str,
    predicates: &[CharCondition],
) -> Result<PyGraph, TyperError> {
    assert!(
        !predicates.is_empty(),
        "or-of-predicates helper requires at least one condition"
    );

    let arg = variable_with_lltype("ch", arg_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);

    // Compute `c = cast(ch)` once at the entry; carry it as a link
    // arg through to subsequent check blocks.
    let c0 = variable_with_lltype("c", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        cast_op,
        vec![Hlvalue::Variable(arg)],
        Hlvalue::Variable(c0.clone()),
    ));

    // Pre-create entry blocks for predicates[1..N-1] so each falls
    // through to the next on False.
    let next_entry_blocks: Vec<(crate::flowspace::model::BlockRef, _)> = (1..predicates.len())
        .map(|_| {
            let c_in = variable_with_lltype("c", LowLevelType::Signed);
            (Block::shared(vec![Hlvalue::Variable(c_in.clone())]), c_in)
        })
        .collect();

    let mut current_block = startblock.clone();
    let mut current_c = c0;
    for (i, pred) in predicates.iter().enumerate() {
        let last = i + 1 == predicates.len();
        let false_target = if last {
            graph.returnblock.clone()
        } else {
            next_entry_blocks[i].0.clone()
        };
        let false_link_args = || -> Vec<Hlvalue> {
            if last {
                vec![bool_false()]
            } else {
                vec![Hlvalue::Variable(current_c.clone())]
            }
        };

        match pred {
            CharCondition::Eq(n) => {
                let eq_var = variable_with_lltype("eq", LowLevelType::Bool);
                current_block
                    .borrow_mut()
                    .operations
                    .push(SpaceOperation::new(
                        "int_eq",
                        vec![Hlvalue::Variable(current_c.clone()), signed_const(*n)],
                        Hlvalue::Variable(eq_var.clone()),
                    ));
                current_block.borrow_mut().exitswitch = Some(Hlvalue::Variable(eq_var));
                let true_link = Link::new(
                    vec![bool_true()],
                    Some(graph.returnblock.clone()),
                    Some(bool_true()),
                )
                .into_ref();
                let false_link =
                    Link::new(false_link_args(), Some(false_target), Some(bool_false())).into_ref();
                current_block.closeblock(vec![true_link, false_link]);
            }
            CharCondition::InRange(lo, hi) => {
                let ge_var = variable_with_lltype("ge", LowLevelType::Bool);
                current_block
                    .borrow_mut()
                    .operations
                    .push(SpaceOperation::new(
                        "int_ge",
                        vec![Hlvalue::Variable(current_c.clone()), signed_const(*lo)],
                        Hlvalue::Variable(ge_var.clone()),
                    ));
                current_block.borrow_mut().exitswitch = Some(Hlvalue::Variable(ge_var));

                let c_for_hi = variable_with_lltype("c", LowLevelType::Signed);
                let block_check_hi = Block::shared(vec![Hlvalue::Variable(c_for_hi.clone())]);
                let ge_true_link = Link::new(
                    vec![Hlvalue::Variable(current_c.clone())],
                    Some(block_check_hi.clone()),
                    Some(bool_true()),
                )
                .into_ref();
                let ge_false_link = Link::new(
                    false_link_args(),
                    Some(false_target.clone()),
                    Some(bool_false()),
                )
                .into_ref();
                current_block.closeblock(vec![ge_true_link, ge_false_link]);

                let le_var = variable_with_lltype("le", LowLevelType::Bool);
                block_check_hi
                    .borrow_mut()
                    .operations
                    .push(SpaceOperation::new(
                        "int_le",
                        vec![Hlvalue::Variable(c_for_hi), signed_const(*hi)],
                        Hlvalue::Variable(le_var.clone()),
                    ));
                block_check_hi.borrow_mut().exitswitch = Some(Hlvalue::Variable(le_var));
                let le_true_link = Link::new(
                    vec![bool_true()],
                    Some(graph.returnblock.clone()),
                    Some(bool_true()),
                )
                .into_ref();
                let le_false_link =
                    Link::new(false_link_args(), Some(false_target), Some(bool_false())).into_ref();
                block_check_hi.closeblock(vec![le_true_link, le_false_link]);
            }
        }

        if !last {
            current_block = next_entry_blocks[i].0.clone();
            current_c = next_entry_blocks[i].1.clone();
        }
    }

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

/// Synthesizes the `ll_<charlike>_<predicate>(ch) -> Bool` helper graph
/// for predicates of the form `lo <= ord(ch) <= hi` (RPython
/// `rstr.py:891-922` `ll_char_isdigit/isupper/islower`).
///
/// 3-block CFG (mirrors RPython source-level short-circuit `and`):
/// - **start**: `c = cast_<arg>_to_int(ch); ge = int_ge(c, lo)`. Branches
///   on `ge`: True → `block_check_hi`, False → `returnblock` (carrying
///   `Bool(false)` constant via the link).
/// - **block_check_hi**: `le = int_le(c, hi)`. Closes to `returnblock`
///   carrying `le` as the function result.
fn build_ll_charlike_predicate_inrange_helper_graph(
    name: &str,
    arg_lltype: LowLevelType,
    cast_op: &'static str,
    lo: i64,
    hi: i64,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("ch", arg_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let c_for_hi = variable_with_lltype("c", LowLevelType::Signed);
    let block_check_hi = Block::shared(vec![Hlvalue::Variable(c_for_hi.clone())]);

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);

    // ---- start block: cast then compare against `lo`.
    let c = variable_with_lltype("c", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        cast_op,
        vec![Hlvalue::Variable(arg)],
        Hlvalue::Variable(c.clone()),
    ));
    let ge = variable_with_lltype("ge", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![Hlvalue::Variable(c.clone()), signed_const(lo)],
        Hlvalue::Variable(ge.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(ge));
    let start_true_link = Link::new(
        vec![Hlvalue::Variable(c)],
        Some(block_check_hi.clone()),
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

    // ---- block_check_hi: compare against `hi`, return result.
    let le = variable_with_lltype("le", LowLevelType::Bool);
    block_check_hi
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_le",
            vec![Hlvalue::Variable(c_for_hi), signed_const(hi)],
            Hlvalue::Variable(le.clone()),
        ));
    block_check_hi.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(le)],
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

/// Synthesizes the `ll_lower_char(ch) -> Char` /
/// `ll_upper_char(ch) -> Char` helper graph (RPython `rstr.py:925-934`):
///
/// ```python
/// def ll_lower_char(ch):
///     if 'A' <= ch <= 'Z':
///         ch = chr(ord(ch) + 32)
///     return ch
/// ```
///
/// 4-block CFG:
/// - **start**: `c = cast_char_to_int(ch); ge = int_ge(c, lo)`. True →
///   `block_check_hi` (link args `[ch, c]`); False → returnblock with
///   original `ch`.
/// - **block_check_hi**: `le = int_le(c, hi)`. True → `block_offset`
///   (link arg `[c]`); False → returnblock with original `ch`.
/// - **block_offset**: `c2 = int_add(c, offset); ch2 = cast_int_to_char(c2)`.
///   Link to returnblock with `ch2`.
fn build_ll_char_case_fold_helper_graph(
    name: &str,
    lo: i64,
    hi: i64,
    offset: i64,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("ch", LowLevelType::Char);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Char);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);

    // block_check_hi inputargs: ch_in (carries through unchanged), c_in.
    let ch_for_hi = variable_with_lltype("ch", LowLevelType::Char);
    let c_for_hi = variable_with_lltype("c", LowLevelType::Signed);
    let block_check_hi = Block::shared(vec![
        Hlvalue::Variable(ch_for_hi.clone()),
        Hlvalue::Variable(c_for_hi.clone()),
    ]);

    // block_offset inputargs: c (already in range; offset applied here).
    let c_for_offset = variable_with_lltype("c", LowLevelType::Signed);
    let block_offset = Block::shared(vec![Hlvalue::Variable(c_for_offset.clone())]);

    // ---- start: cast + range-lo check.
    let c = variable_with_lltype("c", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_char_to_int",
        vec![Hlvalue::Variable(arg.clone())],
        Hlvalue::Variable(c.clone()),
    ));
    let ge = variable_with_lltype("ge", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![Hlvalue::Variable(c.clone()), signed_const(lo)],
        Hlvalue::Variable(ge.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(ge));
    let start_true_link = Link::new(
        vec![Hlvalue::Variable(arg.clone()), Hlvalue::Variable(c)],
        Some(block_check_hi.clone()),
        Some(bool_true()),
    )
    .into_ref();
    let start_false_link = Link::new(
        vec![Hlvalue::Variable(arg)],
        Some(graph.returnblock.clone()),
        Some(bool_false()),
    )
    .into_ref();
    startblock.closeblock(vec![start_true_link, start_false_link]);

    // ---- block_check_hi: range-hi check.
    let le = variable_with_lltype("le", LowLevelType::Bool);
    block_check_hi
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_le",
            vec![Hlvalue::Variable(c_for_hi.clone()), signed_const(hi)],
            Hlvalue::Variable(le.clone()),
        ));
    block_check_hi.borrow_mut().exitswitch = Some(Hlvalue::Variable(le));
    let hi_true_link = Link::new(
        vec![Hlvalue::Variable(c_for_hi)],
        Some(block_offset.clone()),
        Some(bool_true()),
    )
    .into_ref();
    let hi_false_link = Link::new(
        vec![Hlvalue::Variable(ch_for_hi)],
        Some(graph.returnblock.clone()),
        Some(bool_false()),
    )
    .into_ref();
    block_check_hi.closeblock(vec![hi_true_link, hi_false_link]);

    // ---- block_offset: int_add(c, offset); cast_int_to_char.
    let c2 = variable_with_lltype("c2", LowLevelType::Signed);
    block_offset
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(c_for_offset), signed_const(offset)],
            Hlvalue::Variable(c2.clone()),
        ));
    let ch2 = variable_with_lltype("ch2", LowLevelType::Char);
    block_offset
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "cast_int_to_char",
            vec![Hlvalue::Variable(c2)],
            Hlvalue::Variable(ch2.clone()),
        ));
    block_offset.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(ch2)],
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

fn build_ll_charlike_hash_helper_graph(
    name: &str,
    arg_lltype: LowLevelType,
    cast_op: &'static str,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("ch", arg_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let hashed = variable_with_lltype("hashed", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        cast_op,
        vec![Hlvalue::Variable(arg)],
        Hlvalue::Variable(hashed.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(hashed)],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;

    /// rstr.py:496-500 — `CharRepr.get_ll_eq_function` returns None;
    /// `get_ll_hash_function` returns `ll_char_hash` which casts to int.
    #[test]
    fn char_repr_get_ll_hash_function_emits_cast_char_to_int() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r = char_repr();

        assert!(r.get_ll_eq_function(&rtyper).unwrap().is_none());

        let llfn = r
            .get_ll_hash_function(&rtyper)
            .unwrap()
            .expect("Some helper");
        assert_eq!(llfn.name, "ll_char_hash");
        assert_eq!(llfn.args, vec![LowLevelType::Char]);
        assert_eq!(llfn.result, LowLevelType::Signed);

        let graph = llfn.graph.as_ref().unwrap();
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["cast_char_to_int"]);
    }

    /// rstr.py:764-768 — `UniCharRepr.get_ll_eq_function` returns None;
    /// `get_ll_hash_function` returns `ll_unichar_hash` which casts to int.
    #[test]
    fn unichar_repr_get_ll_hash_function_emits_cast_unichar_to_int() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r = unichar_repr();

        assert!(r.get_ll_eq_function(&rtyper).unwrap().is_none());

        let llfn = r
            .get_ll_hash_function(&rtyper)
            .unwrap()
            .expect("Some helper");
        assert_eq!(llfn.name, "ll_unichar_hash");
        assert_eq!(llfn.args, vec![LowLevelType::UniChar]);
        assert_eq!(llfn.result, LowLevelType::Signed);

        let graph = llfn.graph.as_ref().unwrap();
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["cast_unichar_to_int"]);
    }

    /// rstr.py:110-114 — `AbstractStringRepr.get_ll_eq_function`
    /// returns `ll_streq`, `get_ll_hash_function` returns `ll_strhash`.
    /// StringRepr inherits both. The helper-cache materialises both
    /// graphs against `Ptr(STR)`.
    #[test]
    fn string_repr_get_ll_eq_function_returns_ll_streq_helper() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r = string_repr();

        let llfn = r
            .get_ll_eq_function(&rtyper)
            .unwrap()
            .expect("StringRepr ll_streq helper");
        assert_eq!(llfn.name, "ll_streq");
        assert_eq!(llfn.args, vec![STRPTR.clone(), STRPTR.clone()]);
        assert_eq!(llfn.result, LowLevelType::Bool);
    }

    #[test]
    fn string_repr_get_ll_hash_function_returns_ll_strhash_helper() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r = string_repr();

        let llfn = r
            .get_ll_hash_function(&rtyper)
            .unwrap()
            .expect("StringRepr ll_strhash helper");
        assert_eq!(llfn.name, "ll_strhash");
        assert_eq!(llfn.args, vec![STRPTR.clone()]);
        assert_eq!(llfn.result, LowLevelType::Signed);

        // Sanity: helper graph emits the NULL guard + getfield +
        // jit_conditional_call_value pattern.
        let graph = llfn.graph.as_ref().unwrap();
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["ptr_nonzero"]);
    }

    /// Mirror for UnicodeRepr — distinct helper-cache identities
    /// `ll_unicode_eq` / `ll_unicode_hash` (Ptr(UNICODE)).
    #[test]
    fn unicode_repr_get_ll_eq_and_hash_function_use_unicode_helper_identities() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r = unicode_repr();

        let eq_fn = r
            .get_ll_eq_function(&rtyper)
            .unwrap()
            .expect("UnicodeRepr ll_unicode_eq helper");
        assert_eq!(eq_fn.name, "ll_unicode_eq");
        assert_eq!(eq_fn.args, vec![UNICODEPTR.clone(), UNICODEPTR.clone()]);
        assert_eq!(eq_fn.result, LowLevelType::Bool);

        let hash_fn = r
            .get_ll_hash_function(&rtyper)
            .unwrap()
            .expect("UnicodeRepr ll_unicode_hash helper");
        assert_eq!(hash_fn.name, "ll_unicode_hash");
        assert_eq!(hash_fn.args, vec![UNICODEPTR.clone()]);
        assert_eq!(hash_fn.result, LowLevelType::Signed);
    }

    /// rstr.py:491-494 / 759-762 — `convert_const` accepts only the
    /// matching byte/unicode one-character constant.
    #[test]
    fn char_repr_convert_const_accepts_single_char_only() {
        let r = char_repr();
        let c = r.convert_const(&ConstValue::byte_str(b"a")).unwrap();
        assert_eq!(c.value, ConstValue::byte_str(b"a"));
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Char));

        let err = r.convert_const(&ConstValue::byte_str(b"ab")).unwrap_err();
        assert!(err.to_string().contains("not a character"));

        let err = r.convert_const(&ConstValue::Int(1)).unwrap_err();
        assert!(err.to_string().contains("not a character"));
    }

    #[test]
    fn char_convert_const_rejects_unistr() {
        let r = char_repr();
        let err = r.convert_const(&ConstValue::uni_str("a")).unwrap_err();
        assert!(err.to_string().contains("not a character"));
    }

    #[test]
    fn unichar_repr_convert_const_accepts_single_unicode_only() {
        let r = unichar_repr();
        let c = r.convert_const(&ConstValue::uni_str("π")).unwrap();
        assert_eq!(c.value, ConstValue::uni_str("π"));
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::UniChar));

        let err = r.convert_const(&ConstValue::uni_str("πi")).unwrap_err();
        assert!(err.to_string().contains("not a unicode character"));
    }

    /// RPython `AbstractUniCharRepr.convert_const` (`rstr.py:759-766`)
    /// promotes a Python2 `str` of length 1 to `unicode(value)` —
    /// implicit ASCII decode. Pyre mirrors that for an ASCII-byte
    /// `ConstValue::ByteStr` of length 1.
    #[test]
    fn unichar_convert_const_promotes_ascii_byte_str_to_unistr() {
        let r = unichar_repr();
        let c = r.convert_const(&ConstValue::byte_str(b"a")).unwrap();
        assert_eq!(c.value, ConstValue::uni_str("a"));
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::UniChar));
    }

    /// Non-ASCII bytes raise `UnicodeDecodeError` under Python2's
    /// default codec — pyre rejects the `ByteStr` outright.
    #[test]
    fn unichar_convert_const_rejects_non_ascii_byte_str() {
        let r = unichar_repr();
        let err = r.convert_const(&ConstValue::byte_str(&[0xff])).unwrap_err();
        assert!(err.to_string().contains("not a unicode character"));
    }

    /// `len(value) != 1` filter still applies to byte input — RPython
    /// promotes first then length-checks; pyre short-circuits on
    /// length since ASCII bytes preserve length 1:1 in unicode.
    #[test]
    fn unichar_convert_const_rejects_multibyte_byte_str() {
        let r = unichar_repr();
        let err = r.convert_const(&ConstValue::byte_str(b"ab")).unwrap_err();
        assert!(err.to_string().contains("not a unicode character"));
    }

    /// `char_repr` / `unichar_repr` are module-global singletons.
    #[test]
    fn char_and_unichar_repr_singletons_dedupe() {
        let a = char_repr();
        let b = char_repr();
        assert!(Arc::ptr_eq(&a, &b));
        let u1 = unichar_repr();
        let u2 = unichar_repr();
        assert!(Arc::ptr_eq(&u1, &u2));
    }

    /// `lltypesystem/rstr.py:1255` — `string_repr = StringRepr()`
    /// module-global. Lowleveltype is `Ptr(STR)`; `class_name` /
    /// `repr_class_id` mirror the `StringRepr` upstream class.
    #[test]
    fn string_repr_singleton_lowleveltype_is_strptr() {
        let r = string_repr();
        assert_eq!(r.lowleveltype(), &*STRPTR);
        assert_eq!(r.class_name(), "StringRepr");
        assert_eq!(
            r.repr_class_id(),
            super::super::pairtype::ReprClassId::StringRepr
        );
        let r2 = string_repr();
        assert!(Arc::ptr_eq(&r, &r2));
    }

    /// `lltypesystem/rstr.py:1260` — `unicode_repr = UnicodeRepr()`
    /// module-global with `Ptr(UNICODE)` lowleveltype.
    #[test]
    fn unicode_repr_singleton_lowleveltype_is_unicodeptr() {
        let r = unicode_repr();
        assert_eq!(r.lowleveltype(), &*UNICODEPTR);
        assert_eq!(r.class_name(), "UnicodeRepr");
        assert_eq!(
            r.repr_class_id(),
            super::super::pairtype::ReprClassId::UnicodeRepr
        );
        let r2 = unicode_repr();
        assert!(Arc::ptr_eq(&r, &r2));
    }

    /// `lltypesystem/rstr.py:1267` — `CharRepr.char_repr = char_repr`;
    /// `lltypesystem/rstr.py:1268` — `StringRepr.char_repr = char_repr`
    /// — both class-level attributes alias to the same singleton.
    #[test]
    fn string_repr_and_char_repr_char_repr_method_returns_char_repr() {
        let s = string_repr();
        let c_via_string = s.char_repr();
        assert!(Arc::ptr_eq(&c_via_string, &char_repr()));

        let c = char_repr();
        let c_via_char = c.char_repr();
        assert!(Arc::ptr_eq(&c_via_char, &char_repr()));
    }

    /// `lltypesystem/rstr.py:1265` — `UniCharRepr.char_repr = unichar_repr`;
    /// `lltypesystem/rstr.py:1266` — `UnicodeRepr.char_repr = unichar_repr`.
    #[test]
    fn unicode_repr_and_unichar_repr_char_repr_method_returns_unichar_repr() {
        let u = unicode_repr();
        let c_via_unicode = u.char_repr();
        assert!(Arc::ptr_eq(&c_via_unicode, &unichar_repr()));

        let uc = unichar_repr();
        let c_via_unichar = uc.char_repr();
        assert!(Arc::ptr_eq(&c_via_unichar, &unichar_repr()));
    }

    /// `lltypesystem/rstr.py:1262` — `StringRepr.repr = string_repr`;
    /// `class CharRepr(AbstractCharRepr, StringRepr)` (`:291-292`)
    /// inherits the attribute. Pyre exposes both via `.repr()`.
    #[test]
    fn char_repr_repr_method_returns_string_repr() {
        let c = char_repr();
        let s_via_char = c.repr();
        assert!(Arc::ptr_eq(&s_via_char, &string_repr()));
    }

    /// `lltypesystem/rstr.py:1264` — `UniCharRepr.repr = unicode_repr`;
    /// `class UniCharRepr(AbstractUniCharRepr, UnicodeRepr)` (`:294-295`)
    /// inherits the attribute.
    #[test]
    fn unichar_repr_repr_method_returns_unicode_repr() {
        let uc = unichar_repr();
        let u_via_unichar = uc.repr();
        assert!(Arc::ptr_eq(&u_via_unichar, &unicode_repr()));
    }

    fn build_pair_compare_hop(
        rtyper: std::rc::Rc<RPythonTyper>,
        llops: std::rc::Rc<std::cell::RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>>,
        opname: &str,
        lltype: LowLevelType,
        repr: Arc<dyn Repr>,
        s_each: crate::annotator::model::SomeValue,
    ) -> HighLevelOp {
        use crate::flowspace::model::Variable;
        let v_left = Variable::new();
        v_left.set_concretetype(Some(lltype.clone()));
        let v_right = Variable::new();
        v_right.set_concretetype(Some(lltype.clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Bool));
        let hop = HighLevelOp::new(
            rtyper,
            SpaceOperation::new(
                opname.to_string(),
                vec![Hlvalue::Variable(v_left), Hlvalue::Variable(v_right)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops,
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([s_each.clone(), s_each]);
        hop.args_r
            .borrow_mut()
            .extend([Some(repr.clone()), Some(repr)]);
        hop
    }

    /// rstr.py:740-746 + 750-753 — `pairtype(AbstractCharRepr,
    /// AbstractCharRepr).rtype_<func>` emits `char_<func>` for each
    /// of the six compare operations.
    #[test]
    fn pair_char_char_rtype_compare_emits_char_func_op() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for func in ["eq", "ne", "lt", "le", "gt", "ge"] {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let hop = build_pair_compare_hop(
                rtyper.clone(),
                llops.clone(),
                func,
                LowLevelType::Char,
                char_repr() as Arc<dyn Repr>,
                crate::annotator::model::SomeValue::Char(crate::annotator::model::SomeChar::new(
                    false,
                )),
            );
            let result = pair_char_char_rtype_compare(&hop, func)
                .unwrap_or_else(|err| panic!("char {func}: {err:?}"));
            assert!(matches!(result, Some(Hlvalue::Variable(_))));
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 1, "char {func}: one llop");
            assert_eq!(ops.ops[0].opname, format!("char_{func}"));
        }
    }

    /// rstr.py:778-780 + 789-792 — `pairtype(AbstractUniCharRepr,
    /// AbstractUniCharRepr).rtype_eq` / `rtype_ne` emit `unichar_eq`
    /// / `unichar_ne` directly.
    #[test]
    fn pair_unichar_unichar_rtype_eqne_emits_unichar_func_op() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for func in ["eq", "ne"] {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let hop = build_pair_compare_hop(
                rtyper.clone(),
                llops.clone(),
                func,
                LowLevelType::UniChar,
                unichar_repr() as Arc<dyn Repr>,
                crate::annotator::model::SomeValue::UnicodeCodePoint(
                    crate::annotator::model::SomeUnicodeCodePoint::new(false),
                ),
            );
            let result = pair_unichar_unichar_rtype_compare_eqne(&hop, func)
                .unwrap_or_else(|err| panic!("unichar {func}: {err:?}"));
            assert!(matches!(result, Some(Hlvalue::Variable(_))));
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 1, "unichar {func}: one llop");
            assert_eq!(ops.ops[0].opname, format!("unichar_{func}"));
        }
    }

    /// rstr.py:781-784 + 794-800 — `pairtype(AbstractUniCharRepr,
    /// AbstractUniCharRepr).rtype_lt|le|gt|ge` cast both args via
    /// `cast_unichar_to_int` then dispatch to `int_<func>`.
    #[test]
    fn pair_unichar_unichar_rtype_ord_casts_then_int_func_op() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for func in ["lt", "le", "gt", "ge"] {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let hop = build_pair_compare_hop(
                rtyper.clone(),
                llops.clone(),
                func,
                LowLevelType::UniChar,
                unichar_repr() as Arc<dyn Repr>,
                crate::annotator::model::SomeValue::UnicodeCodePoint(
                    crate::annotator::model::SomeUnicodeCodePoint::new(false),
                ),
            );
            let result = pair_unichar_unichar_rtype_compare_ord(&hop, func)
                .unwrap_or_else(|err| panic!("unichar ord {func}: {err:?}"));
            assert!(matches!(result, Some(Hlvalue::Variable(_))));
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 3, "unichar ord {func}: three llops");
            assert_eq!(ops.ops[0].opname, "cast_unichar_to_int");
            assert_eq!(ops.ops[1].opname, "cast_unichar_to_int");
            assert_eq!(ops.ops[2].opname, format!("int_{func}"));
        }
    }

    /// rstr.py:661-663 — `pairtype(AbstractStringRepr,
    /// AbstractStringRepr).rtype_eq` lowers to a single `direct_call`
    /// against the `ll_streq` helper graph (Bool result).
    #[test]
    fn pair_string_string_rtype_eq_emits_direct_call_to_ll_streq() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let hop = build_pair_compare_hop(
            rtyper.clone(),
            llops.clone(),
            "eq",
            STRPTR.clone(),
            string_repr() as Arc<dyn Repr>,
            crate::annotator::model::SomeValue::String(crate::annotator::model::SomeString::new(
                false, false,
            )),
        );
        let result = pair_string_string_rtype_compare(&hop, "eq")
            .unwrap_or_else(|err| panic!("pair eq: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1, "string eq: one direct_call");
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(dbg.contains("ll_streq"), "expected 'll_streq' in {dbg}");
    }

    /// rstr.py:665-668 — `pairtype(AbstractStringRepr,
    /// AbstractStringRepr).rtype_ne` emits `direct_call(ll_streq)`
    /// then wraps with `bool_not`.
    #[test]
    fn pair_string_string_rtype_ne_emits_ll_streq_then_bool_not() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let hop = build_pair_compare_hop(
            rtyper.clone(),
            llops.clone(),
            "ne",
            STRPTR.clone(),
            string_repr() as Arc<dyn Repr>,
            crate::annotator::model::SomeValue::String(crate::annotator::model::SomeString::new(
                false, false,
            )),
        );
        let result = pair_string_string_rtype_compare(&hop, "ne")
            .unwrap_or_else(|err| panic!("pair ne: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 2, "string ne: direct_call + bool_not");
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(dbg.contains("ll_streq"), "expected 'll_streq' in {dbg}");
        assert_eq!(ops.ops[1].opname, "bool_not");
    }

    /// rstr.py:670-692 — `pairtype(AbstractStringRepr,
    /// AbstractStringRepr).rtype_<lt|le|gt|ge>` emits
    /// `direct_call(ll_strcmp)` then `int_<func>(diff, Signed(0))`.
    /// Walk all four ord ops in one loop.
    #[test]
    fn pair_string_string_rtype_ord_emits_ll_strcmp_then_int_func_against_zero() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for func in ["lt", "le", "gt", "ge"] {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let hop = build_pair_compare_hop(
                rtyper.clone(),
                llops.clone(),
                func,
                STRPTR.clone(),
                string_repr() as Arc<dyn Repr>,
                crate::annotator::model::SomeValue::String(
                    crate::annotator::model::SomeString::new(false, false),
                ),
            );
            let result = pair_string_string_rtype_compare(&hop, func)
                .unwrap_or_else(|err| panic!("string ord {func}: {err:?}"));
            assert!(matches!(result, Some(Hlvalue::Variable(_))));
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 2, "string {func}: direct_call + int_{func}");
            assert_eq!(ops.ops[0].opname, "direct_call");
            let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
                panic!("expected Constant funcptr");
            };
            let dbg = format!("{:?}", c.value);
            assert!(dbg.contains("ll_strcmp"), "expected 'll_strcmp' in {dbg}");
            assert_eq!(ops.ops[1].opname, format!("int_{func}"));
            // int_<func> 's second arg is Signed(0).
            let Hlvalue::Constant(c0) = &ops.ops[1].args[1] else {
                panic!("expected Constant zero arg");
            };
            assert!(matches!(
                c0.value,
                crate::flowspace::model::ConstValue::Int(0)
            ));
        }
    }

    /// rstr.py:651-692 inherited via
    /// `AbstractUnicodeRepr(AbstractStringRepr)` —
    /// `pairtype(AbstractUnicodeRepr, AbstractUnicodeRepr).rtype_eq`
    /// routes through `ll_unicode_eq`. Distinct from `ll_streq` so the
    /// helper-cache key separates the two pairs.
    #[test]
    fn pair_unicode_unicode_rtype_eq_emits_direct_call_to_ll_unicode_eq() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let hop = build_pair_compare_hop(
            rtyper.clone(),
            llops.clone(),
            "eq",
            UNICODEPTR.clone(),
            unicode_repr() as Arc<dyn Repr>,
            crate::annotator::model::SomeValue::UnicodeString(
                crate::annotator::model::SomeUnicodeString::new(false, false),
            ),
        );
        let result = pair_unicode_unicode_rtype_compare(&hop, "eq")
            .unwrap_or_else(|err| panic!("pair unicode eq: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_unicode_eq"),
            "expected 'll_unicode_eq' in {dbg}"
        );
    }

    /// Mirror of `pair_string_string_rtype_ord_*` for UnicodeRepr —
    /// emits `direct_call(ll_unicode_cmp)` then `int_<func>(diff, 0)`.
    #[test]
    fn pair_unicode_unicode_rtype_ord_emits_ll_unicode_cmp_then_int_func_against_zero() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for func in ["lt", "le", "gt", "ge"] {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let hop = build_pair_compare_hop(
                rtyper.clone(),
                llops.clone(),
                func,
                UNICODEPTR.clone(),
                unicode_repr() as Arc<dyn Repr>,
                crate::annotator::model::SomeValue::UnicodeString(
                    crate::annotator::model::SomeUnicodeString::new(false, false),
                ),
            );
            let result = pair_unicode_unicode_rtype_compare(&hop, func)
                .unwrap_or_else(|err| panic!("unicode ord {func}: {err:?}"));
            assert!(matches!(result, Some(Hlvalue::Variable(_))));
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 2);
            assert_eq!(ops.ops[0].opname, "direct_call");
            let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
                panic!("expected Constant funcptr");
            };
            let dbg = format!("{:?}", c.value);
            assert!(
                dbg.contains("ll_unicode_cmp"),
                "expected 'll_unicode_cmp' in {dbg}"
            );
            assert_eq!(ops.ops[1].opname, format!("int_{func}"));
        }
    }

    /// rstr.py:134-145 — `AbstractStringRepr.rtype_method_startswith`
    /// (String/String branch) lowers to a single `direct_call` against
    /// the `ll_startswith` helper graph (Bool result).
    #[test]
    fn string_repr_rtype_method_startswith_emits_direct_call_to_ll_startswith() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let hop = build_pair_compare_hop(
            rtyper.clone(),
            llops.clone(),
            "startswith",
            STRPTR.clone(),
            string_repr() as Arc<dyn Repr>,
            crate::annotator::model::SomeValue::String(crate::annotator::model::SomeString::new(
                false, false,
            )),
        );
        let result = string_repr()
            .rtype_method("startswith", &hop)
            .unwrap_or_else(|err| panic!("rtype_method startswith: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        // rstr.py:144 — `hop.exception_cannot_occur()` precedes the
        // gendirectcall.
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_method_startswith must call hop.exception_cannot_occur()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_startswith"),
            "expected 'll_startswith' in {dbg}"
        );
    }

    /// Mirror for `UnicodeRepr.rtype_method_startswith` — distinct
    /// helper-cache identity (`ll_unicode_startswith`).
    #[test]
    fn unicode_repr_rtype_method_startswith_emits_direct_call_to_ll_unicode_startswith() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let hop = build_pair_compare_hop(
            rtyper.clone(),
            llops.clone(),
            "startswith",
            UNICODEPTR.clone(),
            unicode_repr() as Arc<dyn Repr>,
            crate::annotator::model::SomeValue::UnicodeString(
                crate::annotator::model::SomeUnicodeString::new(false, false),
            ),
        );
        let result = unicode_repr()
            .rtype_method("startswith", &hop)
            .unwrap_or_else(|err| panic!("rtype_method startswith: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_unicode_startswith"),
            "expected 'll_unicode_startswith' in {dbg}"
        );
    }

    /// Char-side branch of `rtype_method_startswith` (`ll_startswith_char`,
    /// `rstr.py:138-141`) — when `args_r[1]` is a CharRepr, the
    /// dispatcher gendirectcalls the single-char helper instead of
    /// `ll_startswith`.
    #[test]
    fn string_repr_rtype_method_startswith_dispatches_to_ll_startswith_char_for_char_arg1() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_str = Variable::new();
        v_str.set_concretetype(Some(STRPTR.clone()));
        let v_ch = Variable::new();
        v_ch.set_concretetype(Some(LowLevelType::Char));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Bool));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "startswith".to_string(),
                vec![Hlvalue::Variable(v_str), Hlvalue::Variable(v_ch)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            crate::annotator::model::SomeValue::String(crate::annotator::model::SomeString::new(
                false, false,
            )),
            crate::annotator::model::SomeValue::Char(crate::annotator::model::SomeChar::new(false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(string_repr() as Arc<dyn Repr>),
            Some(char_repr() as Arc<dyn Repr>),
        ]);

        let result = string_repr()
            .rtype_method("startswith", &hop)
            .unwrap_or_else(|err| panic!("rtype_method startswith char-side: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_startswith_char"),
            "expected 'll_startswith_char' in {dbg}"
        );
    }

    /// Char-side branch of `rtype_method_endswith` (`ll_endswith_char`,
    /// `rstr.py:151-153`) — analogous to the startswith char-side
    /// branch, dispatcher routes through `ll_endswith_char`.
    #[test]
    fn string_repr_rtype_method_endswith_dispatches_to_ll_endswith_char_for_char_arg1() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_str = Variable::new();
        v_str.set_concretetype(Some(STRPTR.clone()));
        let v_ch = Variable::new();
        v_ch.set_concretetype(Some(LowLevelType::Char));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Bool));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "endswith".to_string(),
                vec![Hlvalue::Variable(v_str), Hlvalue::Variable(v_ch)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            crate::annotator::model::SomeValue::String(crate::annotator::model::SomeString::new(
                false, false,
            )),
            crate::annotator::model::SomeValue::Char(crate::annotator::model::SomeChar::new(false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(string_repr() as Arc<dyn Repr>),
            Some(char_repr() as Arc<dyn Repr>),
        ]);

        let result = string_repr()
            .rtype_method("endswith", &hop)
            .unwrap_or_else(|err| panic!("rtype_method endswith char-side: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_endswith_char"),
            "expected 'll_endswith_char' in {dbg}"
        );
    }

    /// rstr.py:160-207 — char-needle `find` / `rfind` / `count`
    /// dispatch to their single-char helpers with explicit start/end.
    #[test]
    fn string_repr_rtype_method_findlike_dispatches_to_char_helpers_for_char_arg1() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rint::signed_repr;
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for (method, expected_helper) in [
            ("find", "ll_find_char"),
            ("rfind", "ll_rfind_char"),
            ("count", "ll_count_char"),
        ] {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let v_str = Variable::new();
            v_str.set_concretetype(Some(STRPTR.clone()));
            let v_ch = Variable::new();
            v_ch.set_concretetype(Some(LowLevelType::Char));
            let v_start = Variable::new();
            v_start.set_concretetype(Some(LowLevelType::Signed));
            let v_end = Variable::new();
            v_end.set_concretetype(Some(LowLevelType::Signed));
            let v_result = Variable::new();
            v_result.set_concretetype(Some(LowLevelType::Signed));
            let hop = HighLevelOp::new(
                rtyper.clone(),
                SpaceOperation::new(
                    method.to_string(),
                    vec![
                        Hlvalue::Variable(v_str),
                        Hlvalue::Variable(v_ch),
                        Hlvalue::Variable(v_start),
                        Hlvalue::Variable(v_end),
                    ],
                    Hlvalue::Variable(v_result),
                ),
                Vec::new(),
                llops.clone(),
            );
            hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
            hop.args_s.borrow_mut().extend([
                crate::annotator::model::SomeValue::String(
                    crate::annotator::model::SomeString::new(false, false),
                ),
                crate::annotator::model::SomeValue::Char(crate::annotator::model::SomeChar::new(
                    false,
                )),
                crate::annotator::model::SomeValue::Integer(
                    crate::annotator::model::SomeInteger::new(false, false),
                ),
                crate::annotator::model::SomeValue::Integer(
                    crate::annotator::model::SomeInteger::new(false, false),
                ),
            ]);
            hop.args_r.borrow_mut().extend([
                Some(string_repr() as Arc<dyn Repr>),
                Some(char_repr() as Arc<dyn Repr>),
                Some(signed_repr() as Arc<dyn Repr>),
                Some(signed_repr() as Arc<dyn Repr>),
            ]);

            let result = string_repr()
                .rtype_method(method, &hop)
                .unwrap_or_else(|err| panic!("rtype_method {method} char-side: {err:?}"));
            assert!(matches!(result, Some(Hlvalue::Variable(_))));
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 1, "{method}: one direct_call");
            assert_eq!(ops.ops[0].opname, "direct_call");
            assert!(
                ops._called_exception_is_here_or_cannot_occur,
                "{method}: must call hop.exception_cannot_occur()"
            );
            let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
                panic!("expected Constant funcptr");
            };
            let dbg = format!("{:?}", c.value);
            assert!(
                dbg.contains(expected_helper),
                "{method}: expected '{expected_helper}' in {dbg}"
            );
        }
    }

    /// Audit fix #2 regression coverage — Unicode/UniChar matched-pair
    /// char-branch dispatch must use UniChar as the helper signature
    /// elem type so the dispatcher and `build_ll_startswith_char_helper_graph`
    /// (which derives elem from `ptr_lltype.chars`) agree.
    #[test]
    fn unicode_repr_rtype_method_startswith_dispatches_to_ll_unicode_startswith_char_for_unichar_arg1()
     {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_uni = Variable::new();
        v_uni.set_concretetype(Some(UNICODEPTR.clone()));
        let v_uch = Variable::new();
        v_uch.set_concretetype(Some(LowLevelType::UniChar));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Bool));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "startswith".to_string(),
                vec![Hlvalue::Variable(v_uni), Hlvalue::Variable(v_uch)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            crate::annotator::model::SomeValue::UnicodeString(
                crate::annotator::model::SomeUnicodeString::new(false, false),
            ),
            crate::annotator::model::SomeValue::UnicodeCodePoint(
                crate::annotator::model::SomeUnicodeCodePoint::new(false),
            ),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(unicode_repr() as Arc<dyn Repr>),
            Some(unichar_repr() as Arc<dyn Repr>),
        ]);

        let result = unicode_repr()
            .rtype_method("startswith", &hop)
            .unwrap_or_else(|err| panic!("rtype_method startswith unichar-side: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_unicode_startswith_char"),
            "expected 'll_unicode_startswith_char' in {dbg}"
        );
    }

    /// rstr.py:147-158 — `AbstractStringRepr.rtype_method_endswith`
    /// (String/String branch) lowers to a single `direct_call` against
    /// the `ll_endswith` helper graph (Bool result).
    #[test]
    fn string_repr_rtype_method_endswith_emits_direct_call_to_ll_endswith() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let hop = build_pair_compare_hop(
            rtyper.clone(),
            llops.clone(),
            "endswith",
            STRPTR.clone(),
            string_repr() as Arc<dyn Repr>,
            crate::annotator::model::SomeValue::String(crate::annotator::model::SomeString::new(
                false, false,
            )),
        );
        let result = string_repr()
            .rtype_method("endswith", &hop)
            .unwrap_or_else(|err| panic!("rtype_method endswith: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        // rstr.py:157 — `hop.exception_cannot_occur()` precedes the
        // gendirectcall.
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_method_endswith must call hop.exception_cannot_occur()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_endswith"),
            "expected 'll_endswith' in {dbg}"
        );
    }

    /// rstr.py:614-632 — `pair(StringRepr, IntegerRepr).rtype_getitem`
    /// nonneg branch lowers to a single `direct_call` against the
    /// `ll_stritem_nonneg` helper graph (returns Char).
    #[test]
    fn pair_string_int_rtype_getitem_nonneg_emits_direct_call_to_ll_stritem_nonneg() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_str = Variable::new();
        v_str.set_concretetype(Some(STRPTR.clone()));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Char));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_str), Hlvalue::Variable(v_idx)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            crate::annotator::model::SomeValue::String(crate::annotator::model::SomeString::new(
                false, false,
            )),
            crate::annotator::model::SomeValue::Integer(crate::annotator::model::SomeInteger::new(
                /* nonneg */ true, false,
            )),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(string_repr() as Arc<dyn Repr>),
            Some(crate::translator::rtyper::rint::signed_repr() as Arc<dyn Repr>),
        ]);

        let result = pair_string_int_rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("pair getitem nonneg: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        // rstr.py:628 (checkidx=False path) — `hop.exception_cannot_occur()`
        // precedes the gendirectcall.
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "pair_abstract_string_int_rtype_getitem (checkidx=False) must call \
             hop.exception_cannot_occur()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_stritem_nonneg"),
            "expected 'll_stritem_nonneg' in {dbg}"
        );
    }

    /// rstr.py:614-632 — neg-index branch (`args_s[1].nonneg = false`)
    /// dispatches to `ll_stritem` (which inlines neg-fix + direct_call
    /// ll_stritem_nonneg sub-helper).
    #[test]
    fn pair_string_int_rtype_getitem_dispatches_to_ll_stritem_for_neg_arg() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_str = Variable::new();
        v_str.set_concretetype(Some(STRPTR.clone()));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Char));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_str), Hlvalue::Variable(v_idx)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            crate::annotator::model::SomeValue::String(crate::annotator::model::SomeString::new(
                false, false,
            )),
            crate::annotator::model::SomeValue::Integer(crate::annotator::model::SomeInteger::new(
                /* nonneg */ false, false,
            )),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(string_repr() as Arc<dyn Repr>),
            Some(crate::translator::rtyper::rint::signed_repr() as Arc<dyn Repr>),
        ]);

        let result = pair_string_int_rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("pair getitem neg-index: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(dbg.contains("ll_stritem"), "expected 'll_stritem' in {dbg}");
        // Crucially the funcptr must NOT be `ll_stritem_nonneg` — neg
        // dispatch goes through ll_stritem first.
        assert!(
            !dbg.contains("ll_stritem_nonneg"),
            "expected ll_stritem (not ll_stritem_nonneg) for neg-index path; \
             got {dbg}"
        );
    }

    /// rstr.py:614-635 — `pair(StringRepr, IntegerRepr).rtype_getitem_idx`
    /// dispatches to `rtype_getitem(hop, checkidx=True)`. With
    /// `args_s[1].nonneg=True` the helper is `ll_stritem_nonneg_checked`
    /// (raises IndexError on `i >= length`).
    #[test]
    fn pair_string_int_rtype_getitem_idx_nonneg_dispatches_to_ll_stritem_nonneg_checked() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_str = Variable::new();
        v_str.set_concretetype(Some(STRPTR.clone()));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Char));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem_idx".to_string(),
                vec![Hlvalue::Variable(v_str), Hlvalue::Variable(v_idx)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            crate::annotator::model::SomeValue::String(crate::annotator::model::SomeString::new(
                false, false,
            )),
            crate::annotator::model::SomeValue::Integer(crate::annotator::model::SomeInteger::new(
                /* nonneg */ true, false,
            )),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(string_repr() as Arc<dyn Repr>),
            Some(crate::translator::rtyper::rint::signed_repr() as Arc<dyn Repr>),
        ]);

        let result = pair_string_int_rtype_getitem_idx(&hop)
            .unwrap_or_else(|err| panic!("pair getitem_idx nonneg: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        // rstr.py:629-630 — checkidx=True path uses
        // `hop.exception_is_here()`, which still sets the
        // `_called_exception_is_here_or_cannot_occur` flag.
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "checkidx=True path must call hop.exception_is_here()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_stritem_nonneg_checked"),
            "expected 'll_stritem_nonneg_checked' in {dbg}"
        );
    }

    /// rstr.py:614-635 — `rtype_getitem_idx` with `nonneg=False`
    /// dispatches to `ll_stritem_checked` (full neg-fix + bound-check;
    /// raises IndexError when `i >= length` or `i < 0` after fix-up).
    #[test]
    fn pair_string_int_rtype_getitem_idx_neg_dispatches_to_ll_stritem_checked() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_str = Variable::new();
        v_str.set_concretetype(Some(STRPTR.clone()));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Char));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem_idx".to_string(),
                vec![Hlvalue::Variable(v_str), Hlvalue::Variable(v_idx)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            crate::annotator::model::SomeValue::String(crate::annotator::model::SomeString::new(
                false, false,
            )),
            crate::annotator::model::SomeValue::Integer(crate::annotator::model::SomeInteger::new(
                /* nonneg */ false, false,
            )),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(string_repr() as Arc<dyn Repr>),
            Some(crate::translator::rtyper::rint::signed_repr() as Arc<dyn Repr>),
        ]);

        let result = pair_string_int_rtype_getitem_idx(&hop)
            .unwrap_or_else(|err| panic!("pair getitem_idx neg-index: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "checkidx=True path must call hop.exception_is_here()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_stritem_checked"),
            "expected 'll_stritem_checked' in {dbg}"
        );
        // Must NOT be `ll_stritem_nonneg_checked` — neg-index path
        // routes through `ll_stritem_checked`.
        assert!(
            !dbg.contains("ll_stritem_nonneg_checked"),
            "expected ll_stritem_checked (not ll_stritem_nonneg_checked) for neg-index path; \
             got {dbg}"
        );
    }

    /// rstr.py:651-659 — `pair(StringRepr, StringRepr).rtype_add`
    /// emits `direct_call(ll_strconcat, v_s1, v_s2)` returning Ptr(STR).
    #[test]
    fn pair_string_string_rtype_add_emits_direct_call_to_ll_strconcat() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_s1 = Variable::new();
        v_s1.set_concretetype(Some(STRPTR.clone()));
        let v_s2 = Variable::new();
        v_s2.set_concretetype(Some(STRPTR.clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(STRPTR.clone()));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "add".to_string(),
                vec![Hlvalue::Variable(v_s1), Hlvalue::Variable(v_s2)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        let s_str = crate::annotator::model::SomeValue::String(
            crate::annotator::model::SomeString::new(false, false),
        );
        hop.args_s.borrow_mut().extend([s_str.clone(), s_str]);
        hop.args_r.borrow_mut().extend([
            Some(string_repr() as Arc<dyn Repr>),
            Some(string_repr() as Arc<dyn Repr>),
        ]);

        let result =
            pair_string_string_rtype_add(&hop).unwrap_or_else(|err| panic!("pair add: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1, "string add: one direct_call");
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_strconcat"),
            "expected 'll_strconcat' in {dbg}"
        );
    }

    /// rstr.py:651-659 (UnicodeRepr inherited) — UNICODE pair surface
    /// uses distinct helper-cache identity `ll_unicode_concat`.
    #[test]
    fn pair_unicode_unicode_rtype_add_emits_direct_call_to_ll_unicode_concat() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_s1 = Variable::new();
        v_s1.set_concretetype(Some(UNICODEPTR.clone()));
        let v_s2 = Variable::new();
        v_s2.set_concretetype(Some(UNICODEPTR.clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(UNICODEPTR.clone()));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "add".to_string(),
                vec![Hlvalue::Variable(v_s1), Hlvalue::Variable(v_s2)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        let s_uni = crate::annotator::model::SomeValue::UnicodeString(
            crate::annotator::model::SomeUnicodeString::new(false, false),
        );
        hop.args_s.borrow_mut().extend([s_uni.clone(), s_uni]);
        hop.args_r.borrow_mut().extend([
            Some(unicode_repr() as Arc<dyn Repr>),
            Some(unicode_repr() as Arc<dyn Repr>),
        ]);

        let result = pair_unicode_unicode_rtype_add(&hop)
            .unwrap_or_else(|err| panic!("pair unicode add: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_unicode_concat"),
            "expected 'll_unicode_concat' in {dbg}"
        );
        // Helper-cache identity must NOT alias with the STR family.
        assert!(
            !dbg.contains("ll_strconcat"),
            "UNICODE add must use ll_unicode_concat, not ll_strconcat: {dbg}"
        );
    }

    /// rstr.py:253-269 plus Slice 9 isxxx batch —
    /// `AbstractStringRepr.rtype_method_isxxx` lowers to
    /// `direct_call(ll_isxxx, v_str)`. The helper graph itself
    /// sub-helper-direct-calls the per-char predicate.
    #[test]
    fn string_repr_rtype_method_isxxx_emit_direct_call_to_string_predicate_helpers() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for (method, expected_helper) in [
            ("isdigit", "ll_isdigit"),
            ("isalpha", "ll_isalpha"),
            ("isalnum", "ll_isalnum"),
            ("isspace", "ll_isspace"),
            ("isupper", "ll_isupper"),
            ("islower", "ll_islower"),
        ] {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let s_str = crate::annotator::model::SomeValue::String(
                crate::annotator::model::SomeString::new(false, false),
            );
            let hop = build_string_unary_hop(
                rtyper.clone(),
                llops.clone(),
                method,
                STRPTR.clone(),
                LowLevelType::Bool,
                string_repr() as Arc<dyn Repr>,
                s_str,
            );
            let result = string_repr()
                .rtype_method(method, &hop)
                .unwrap_or_else(|err| panic!("rtype_method {method}: {err:?}"));
            assert!(matches!(result, Some(Hlvalue::Variable(_))));
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 1, "{method}: one direct_call");
            assert_eq!(ops.ops[0].opname, "direct_call");
            assert!(
                ops._called_exception_is_here_or_cannot_occur,
                "{method}: must call hop.exception_cannot_occur()"
            );
            let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
                panic!("expected Constant funcptr");
            };
            let dbg = format!("{:?}", c.value);
            assert!(
                dbg.contains(expected_helper),
                "{method}: expected '{expected_helper}' in {dbg}"
            );
        }
    }

    /// rstr.py:241-251 — `str.upper/lower` dispatch to
    /// `LLHelpers.ll_upper/lower` and cannot raise.
    #[test]
    fn string_repr_rtype_method_casefold_emit_direct_call_to_string_helpers() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for (method, expected_helper) in [("upper", "ll_upper"), ("lower", "ll_lower")] {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let s_str = crate::annotator::model::SomeValue::String(
                crate::annotator::model::SomeString::new(false, false),
            );
            let hop = build_string_unary_hop(
                rtyper.clone(),
                llops.clone(),
                method,
                STRPTR.clone(),
                STRPTR.clone(),
                string_repr() as Arc<dyn Repr>,
                s_str,
            );
            let result = string_repr()
                .rtype_method(method, &hop)
                .unwrap_or_else(|err| panic!("rtype_method {method}: {err:?}"));
            assert!(matches!(result, Some(Hlvalue::Variable(_))));
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 1, "{method}: one direct_call");
            assert_eq!(ops.ops[0].opname, "direct_call");
            assert!(
                ops._called_exception_is_here_or_cannot_occur,
                "{method}: must call hop.exception_cannot_occur()"
            );
            let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
                panic!("expected Constant funcptr");
            };
            let dbg = format!("{:?}", c.value);
            assert!(
                dbg.contains(expected_helper),
                "{method}: expected '{expected_helper}' in {dbg}"
            );
        }
    }

    /// Mirror for UnicodeRepr — distinct helper-cache identities
    /// (`ll_unicode_isxxx`).
    #[test]
    fn unicode_repr_rtype_method_isxxx_emit_direct_call_to_unicode_predicate_helpers() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for (method, expected_helper) in [
            ("isdigit", "ll_unicode_isdigit"),
            ("isalpha", "ll_unicode_isalpha"),
            ("isalnum", "ll_unicode_isalnum"),
            ("isspace", "ll_unicode_isspace"),
            ("isupper", "ll_unicode_isupper"),
            ("islower", "ll_unicode_islower"),
        ] {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let s_uni = crate::annotator::model::SomeValue::UnicodeString(
                crate::annotator::model::SomeUnicodeString::new(false, false),
            );
            let hop = build_string_unary_hop(
                rtyper.clone(),
                llops.clone(),
                method,
                UNICODEPTR.clone(),
                LowLevelType::Bool,
                unicode_repr() as Arc<dyn Repr>,
                s_uni,
            );
            let result = unicode_repr()
                .rtype_method(method, &hop)
                .unwrap_or_else(|err| panic!("rtype_method {method}: {err:?}"));
            assert!(matches!(result, Some(Hlvalue::Variable(_))));
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 1, "{method}: one direct_call");
            assert_eq!(ops.ops[0].opname, "direct_call");
            assert!(
                ops._called_exception_is_here_or_cannot_occur,
                "{method}: must call hop.exception_cannot_occur()"
            );
            let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
                panic!("expected Constant funcptr");
            };
            let dbg = format!("{:?}", c.value);
            assert!(
                dbg.contains(expected_helper),
                "{method}: expected '{expected_helper}' in {dbg}"
            );
        }
    }

    /// rstr.py:452-457 — UnicodeRepr intentionally rejects
    /// `upper/lower` instead of using byte-string casefolding.
    #[test]
    fn unicode_repr_rtype_method_casefold_raises_typererror() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for (method, expected_message) in [
            ("upper", "Cannot do toupper on unicode string"),
            ("lower", "Cannot do tolower on unicode string"),
        ] {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let s_uni = crate::annotator::model::SomeValue::UnicodeString(
                crate::annotator::model::SomeUnicodeString::new(false, false),
            );
            let hop = build_string_unary_hop(
                rtyper.clone(),
                llops,
                method,
                UNICODEPTR.clone(),
                UNICODEPTR.clone(),
                unicode_repr() as Arc<dyn Repr>,
                s_uni,
            );
            let err = unicode_repr()
                .rtype_method(method, &hop)
                .expect_err("unicode upper/lower must be rejected");
            assert!(
                format!("{err:?}").contains(expected_message),
                "{method}: got {err:?}"
            );
        }
    }

    /// rstr.py:363-369 — `replace` accepts only char/unichar needle
    /// and replacement args, then dispatches to `ll_replace_chr_chr`.
    #[test]
    fn string_unicode_rtype_method_replace_emit_direct_call_to_chr_chr_helpers() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        struct Case<'a> {
            method_repr: Arc<dyn Repr>,
            ptr_lltype: LowLevelType,
            elem_lltype: LowLevelType,
            s_str: crate::annotator::model::SomeValue,
            s_char: crate::annotator::model::SomeValue,
            char_repr: Arc<dyn Repr>,
            expected_helper: &'a str,
        }

        let cases = [
            Case {
                method_repr: string_repr() as Arc<dyn Repr>,
                ptr_lltype: STRPTR.clone(),
                elem_lltype: LowLevelType::Char,
                s_str: crate::annotator::model::SomeValue::String(
                    crate::annotator::model::SomeString::new(false, false),
                ),
                s_char: crate::annotator::model::SomeValue::Char(
                    crate::annotator::model::SomeChar::new(false),
                ),
                char_repr: char_repr() as Arc<dyn Repr>,
                expected_helper: "ll_replace_chr_chr",
            },
            Case {
                method_repr: unicode_repr() as Arc<dyn Repr>,
                ptr_lltype: UNICODEPTR.clone(),
                elem_lltype: LowLevelType::UniChar,
                s_str: crate::annotator::model::SomeValue::UnicodeString(
                    crate::annotator::model::SomeUnicodeString::new(false, false),
                ),
                s_char: crate::annotator::model::SomeValue::UnicodeCodePoint(
                    crate::annotator::model::SomeUnicodeCodePoint::new(false),
                ),
                char_repr: unichar_repr() as Arc<dyn Repr>,
                expected_helper: "ll_unicode_replace_chr_chr",
            },
        ];

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for case in cases {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let v_str = Variable::new();
            v_str.set_concretetype(Some(case.ptr_lltype.clone()));
            let v_c1 = Variable::new();
            v_c1.set_concretetype(Some(case.elem_lltype.clone()));
            let v_c2 = Variable::new();
            v_c2.set_concretetype(Some(case.elem_lltype.clone()));
            let v_result = Variable::new();
            v_result.set_concretetype(Some(case.ptr_lltype.clone()));
            let hop = HighLevelOp::new(
                rtyper.clone(),
                SpaceOperation::new(
                    "replace".to_string(),
                    vec![
                        Hlvalue::Variable(v_str),
                        Hlvalue::Variable(v_c1),
                        Hlvalue::Variable(v_c2),
                    ],
                    Hlvalue::Variable(v_result),
                ),
                Vec::new(),
                llops.clone(),
            );
            hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
            hop.args_s
                .borrow_mut()
                .extend([case.s_str, case.s_char.clone(), case.s_char]);
            hop.args_r.borrow_mut().extend([
                Some(case.method_repr.clone()),
                Some(case.char_repr.clone()),
                Some(case.char_repr),
            ]);

            let result = case
                .method_repr
                .rtype_method("replace", &hop)
                .unwrap_or_else(|err| panic!("rtype_method replace: {err:?}"));
            assert!(matches!(result, Some(Hlvalue::Variable(_))));
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 1);
            assert_eq!(ops.ops[0].opname, "direct_call");
            assert!(
                ops._called_exception_is_here_or_cannot_occur,
                "replace must call hop.exception_cannot_occur()"
            );
            let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
                panic!("expected Constant funcptr");
            };
            let dbg = format!("{:?}", c.value);
            assert!(
                dbg.contains(case.expected_helper),
                "expected '{}' in {dbg}",
                case.expected_helper
            );
        }
    }

    /// Mirror for UnicodeRepr — distinct helper-cache identity
    /// `ll_unicode_stritem_nonneg`, returns UniChar.
    #[test]
    fn pair_unicode_int_rtype_getitem_nonneg_emits_direct_call_to_ll_unicode_stritem_nonneg() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_str = Variable::new();
        v_str.set_concretetype(Some(UNICODEPTR.clone()));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::UniChar));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_str), Hlvalue::Variable(v_idx)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            crate::annotator::model::SomeValue::UnicodeString(
                crate::annotator::model::SomeUnicodeString::new(false, false),
            ),
            crate::annotator::model::SomeValue::Integer(crate::annotator::model::SomeInteger::new(
                /* nonneg */ true, false,
            )),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(unicode_repr() as Arc<dyn Repr>),
            Some(crate::translator::rtyper::rint::signed_repr() as Arc<dyn Repr>),
        ]);

        let result = pair_unicode_int_rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("pair unicode getitem: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_unicode_stritem_nonneg"),
            "expected 'll_unicode_stritem_nonneg' in {dbg}"
        );
    }

    /// Mirror for `UnicodeRepr.rtype_method_endswith` — distinct
    /// helper-cache identity (`ll_unicode_endswith`).
    #[test]
    fn unicode_repr_rtype_method_endswith_emits_direct_call_to_ll_unicode_endswith() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let hop = build_pair_compare_hop(
            rtyper.clone(),
            llops.clone(),
            "endswith",
            UNICODEPTR.clone(),
            unicode_repr() as Arc<dyn Repr>,
            crate::annotator::model::SomeValue::UnicodeString(
                crate::annotator::model::SomeUnicodeString::new(false, false),
            ),
        );
        let result = unicode_repr()
            .rtype_method("endswith", &hop)
            .unwrap_or_else(|err| panic!("rtype_method endswith: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_unicode_endswith"),
            "expected 'll_unicode_endswith' in {dbg}"
        );
    }

    /// rstr.py:891-922 ll_char_isdigit / ll_char_isupper /
    /// ll_char_islower bodies are all `lo <= ord(ch) <= hi`. Pyre
    /// synthesizes the same shape as a 3-block CFG: cast → int_ge →
    /// branch (False fallthrough to false-return / True to
    /// block_check_hi → int_le → return).
    #[test]
    fn build_ll_charlike_predicate_inrange_helper_graph_synthesizes_3_block_cfg() {
        let graph = build_ll_charlike_predicate_inrange_helper_graph(
            "ll_char_isdigit",
            LowLevelType::Char,
            "cast_char_to_int",
            48,
            57,
        )
        .expect("synthesize");
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["cast_char_to_int", "int_ge"]);
        assert!(startblock.exitswitch.is_some());
        assert_eq!(startblock.exits.len(), 2);

        // True branch leads to a block that runs int_le; False branch
        // leads directly to the returnblock with a Bool(false) link arg.
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link present");
        let true_target = true_link.borrow().target.as_ref().unwrap().clone();
        let target_block = true_target.borrow();
        let target_ops: Vec<&str> = target_block
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(target_ops, vec!["int_le"]);
    }

    /// rstr.py:886-912 — `ll_char_isspace` body is `c == 32 or 9 <= c
    /// <= 13`. Pyre synthesizes a 4-named-block CFG (start +
    /// inrange_check_hi + next-condition-entry + returnblock).
    #[test]
    fn build_ll_charlike_or_of_conditions_helper_graph_synthesizes_isspace_chain() {
        let graph = build_ll_charlike_or_of_conditions_helper_graph(
            "ll_char_isspace",
            LowLevelType::Char,
            "cast_char_to_int",
            ISSPACE_CONDITIONS,
        )
        .expect("synthesize");
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        // start: cast → int_eq(c, 32) → branch on int_eq result.
        assert_eq!(start_ops, vec!["cast_char_to_int", "int_eq"]);
        assert_eq!(startblock.exits.len(), 2);

        // True branch is direct return Bool(true).
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link present");
        let true_link_borrow = true_link.borrow();
        let true_first_arg = true_link_borrow
            .args
            .first()
            .and_then(|opt| opt.as_ref())
            .expect("True link first arg present");
        assert!(matches!(
            true_first_arg,
            Hlvalue::Constant(c) if c.value == ConstValue::Bool(true)
        ));
        drop(true_link_borrow);

        // False branch falls through to the InRange(9, 13) entry block,
        // which emits int_ge(c, 9).
        let false_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("False exit link present");
        let inrange_entry = false_link.borrow().target.as_ref().unwrap().clone();
        let entry_borrow = inrange_entry.borrow();
        let entry_ops: Vec<&str> = entry_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(entry_ops, vec!["int_ge"]);
    }

    /// rstr.py:925-934 — `ll_lower_char(ch)` body is the conditional
    /// ASCII offset `if 'A' <= ch <= 'Z': ch = chr(ord(ch) + 32)`.
    /// Pyre synthesizes a 4-block CFG: start (cast + ge check),
    /// block_check_hi (le check), block_offset (int_add +
    /// cast_int_to_char), returnblock.
    #[test]
    fn build_ll_char_case_fold_helper_graph_synthesizes_4_block_cfg() {
        let graph =
            build_ll_char_case_fold_helper_graph("ll_lower_char", 65, 90, 32).expect("synthesize");
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["cast_char_to_int", "int_ge"]);
        assert_eq!(startblock.exits.len(), 2);

        // True branch carries [ch, c] to block_check_hi (which runs int_le).
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link");
        let check_hi = true_link.borrow().target.as_ref().unwrap().clone();
        let check_hi_borrow = check_hi.borrow();
        let check_hi_ops: Vec<&str> = check_hi_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(check_hi_ops, vec!["int_le"]);

        // block_check_hi True branch leads to block_offset (int_add + cast_int_to_char).
        let hi_true = check_hi_borrow
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("hi True exit link");
        let offset_block = hi_true.borrow().target.as_ref().unwrap().clone();
        let offset_borrow = offset_block.borrow();
        let offset_ops: Vec<&str> = offset_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(offset_ops, vec!["int_add", "cast_int_to_char"]);
    }

    /// rstr.py:516-538 — CharRepr/UniCharRepr inherit
    /// BaseCharReprMixin._rtype_method_isxxx; pyre's `rtype_method`
    /// dispatch routes isdigit/isupper/islower through
    /// `char_predicate_inrange_method`, which emits a single
    /// `direct_call` to the per-arg-type helper graph.
    #[test]
    fn char_unichar_rtype_method_predicates_emit_direct_call_to_per_predicate_helper() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        struct Case<'a> {
            method: &'a str,
            r: Arc<dyn Repr>,
            arg_lltype: LowLevelType,
            s_each: crate::annotator::model::SomeValue,
            expected_helper: &'a str,
        }
        let s_char = || {
            crate::annotator::model::SomeValue::Char(crate::annotator::model::SomeChar::new(false))
        };
        let s_unichar = || {
            crate::annotator::model::SomeValue::UnicodeCodePoint(
                crate::annotator::model::SomeUnicodeCodePoint::new(false),
            )
        };
        let cases = [
            Case {
                method: "isdigit",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_char_isdigit",
            },
            Case {
                method: "isupper",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_char_isupper",
            },
            Case {
                method: "islower",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_char_islower",
            },
            Case {
                method: "isspace",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_char_isspace",
            },
            Case {
                method: "isalpha",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_char_isalpha",
            },
            Case {
                method: "isalnum",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_char_isalnum",
            },
            Case {
                method: "isdigit",
                r: unichar_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::UniChar,
                s_each: s_unichar(),
                expected_helper: "ll_unichar_isdigit",
            },
            Case {
                method: "isspace",
                r: unichar_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::UniChar,
                s_each: s_unichar(),
                expected_helper: "ll_unichar_isspace",
            },
            Case {
                method: "lower",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_lower_char",
            },
            Case {
                method: "upper",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_upper_char",
            },
        ];

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for case in cases {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let v_arg = Variable::new();
            v_arg.set_concretetype(Some(case.arg_lltype.clone()));
            let v_result = Variable::new();
            v_result.set_concretetype(Some(LowLevelType::Bool));
            let hop = HighLevelOp::new(
                rtyper.clone(),
                SpaceOperation::new(
                    case.method.to_string(),
                    vec![Hlvalue::Variable(v_arg)],
                    Hlvalue::Variable(v_result),
                ),
                Vec::new(),
                llops.clone(),
            );
            hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
            hop.args_s.borrow_mut().push(case.s_each.clone());
            hop.args_r.borrow_mut().push(Some(case.r.clone()));

            let result = case
                .r
                .rtype_method(case.method, &hop)
                .unwrap_or_else(|err| panic!("{} {}: {err:?}", case.method, case.expected_helper));
            assert!(matches!(result, Some(Hlvalue::Variable(_))));
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 1, "{}: one direct_call", case.method);
            assert_eq!(ops.ops[0].opname, "direct_call");
            // First arg of direct_call is the funcptr Constant whose
            // payload identifies the helper graph by name.
            let funcptr_arg = &ops.ops[0].args[0];
            let Hlvalue::Constant(c) = funcptr_arg else {
                panic!("expected Constant funcptr, got {funcptr_arg:?}");
            };
            let dbg = format!("{:?}", c.value);
            assert!(
                dbg.contains(case.expected_helper),
                "{}: expected funcptr '{}' in {dbg}",
                case.method,
                case.expected_helper
            );
        }
    }

    /// Build a single-arg HighLevelOp (`v_arg = OP(v_str)`) annotated
    /// with the supplied `s_arg` SomeValue and the supplied repr.
    /// `result_lltype` is the expected lowleveltype of the result
    /// variable (Signed for `len`, Bool for `bool`).
    fn build_string_unary_hop(
        rtyper: std::rc::Rc<RPythonTyper>,
        llops: std::rc::Rc<std::cell::RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>>,
        opname: &str,
        arg_lltype: LowLevelType,
        result_lltype: LowLevelType,
        repr: Arc<dyn Repr>,
        s_arg: crate::annotator::model::SomeValue,
    ) -> HighLevelOp {
        use crate::flowspace::model::Variable;
        let v_arg = Variable::new();
        v_arg.set_concretetype(Some(arg_lltype));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(result_lltype));
        let hop = HighLevelOp::new(
            rtyper,
            SpaceOperation::new(
                opname.to_string(),
                vec![Hlvalue::Variable(v_arg)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops,
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().push(s_arg);
        hop.args_r.borrow_mut().push(Some(repr));
        hop
    }

    /// rstr.py:119-122 — `AbstractStringRepr.rtype_len` lowers to a
    /// single `direct_call` against the `ll_strlen` helper graph
    /// (Ptr(STR) → Signed). Mirrors upstream `gendirectcall(self.ll.ll_strlen, v_str)`.
    #[test]
    fn string_repr_rtype_len_emits_direct_call_to_ll_strlen() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let s_str = crate::annotator::model::SomeValue::String(
            crate::annotator::model::SomeString::new(false, false),
        );
        let hop = build_string_unary_hop(
            rtyper.clone(),
            llops.clone(),
            "len",
            STRPTR.clone(),
            LowLevelType::Signed,
            string_repr() as Arc<dyn Repr>,
            s_str,
        );

        let result = string_repr()
            .rtype_len(&hop)
            .unwrap_or_else(|err| panic!("rtype_len: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1, "rtype_len: one direct_call");
        assert_eq!(ops.ops[0].opname, "direct_call");
        let funcptr_arg = &ops.ops[0].args[0];
        let Hlvalue::Constant(c) = funcptr_arg else {
            panic!("expected Constant funcptr, got {funcptr_arg:?}");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_strlen"),
            "expected funcptr 'll_strlen' in {dbg}"
        );
    }

    /// rstr.py:119-122 mirror for UnicodeRepr — same shape, different
    /// helper identity (`ll_unilen`) and pointer lltype (Ptr(UNICODE)).
    #[test]
    fn unicode_repr_rtype_len_emits_direct_call_to_ll_unilen() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let s_uni = crate::annotator::model::SomeValue::UnicodeString(
            crate::annotator::model::SomeUnicodeString::new(false, false),
        );
        let hop = build_string_unary_hop(
            rtyper.clone(),
            llops.clone(),
            "len",
            UNICODEPTR.clone(),
            LowLevelType::Signed,
            unicode_repr() as Arc<dyn Repr>,
            s_uni,
        );

        let result = unicode_repr()
            .rtype_len(&hop)
            .unwrap_or_else(|err| panic!("rtype_len: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1, "rtype_len: one direct_call");
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_unilen"),
            "expected funcptr 'll_unilen' in {dbg}"
        );
    }

    /// rstr.py:124-132 — `AbstractStringRepr.rtype_bool` with
    /// `can_be_None == True` emits `gendirectcall(ll_str_is_true)`.
    #[test]
    fn string_repr_rtype_bool_when_can_be_none_emits_direct_call_to_ll_str_is_true() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let s_str = crate::annotator::model::SomeValue::String(
            crate::annotator::model::SomeString::new(true, false),
        );
        let hop = build_string_unary_hop(
            rtyper.clone(),
            llops.clone(),
            "bool",
            STRPTR.clone(),
            LowLevelType::Bool,
            string_repr() as Arc<dyn Repr>,
            s_str,
        );

        let result = string_repr()
            .rtype_bool(&hop)
            .unwrap_or_else(|err| panic!("rtype_bool: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1, "rtype_bool can_be_None: one direct_call");
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_str_is_true"),
            "expected funcptr 'll_str_is_true' in {dbg}"
        );
    }

    /// rstr.py:124-132 mirror for UnicodeRepr — `can_be_None == True`
    /// path emits `gendirectcall(ll_unicode_is_true)`.
    #[test]
    fn unicode_repr_rtype_bool_when_can_be_none_emits_direct_call_to_ll_unicode_is_true() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let s_uni = crate::annotator::model::SomeValue::UnicodeString(
            crate::annotator::model::SomeUnicodeString::new(true, false),
        );
        let hop = build_string_unary_hop(
            rtyper.clone(),
            llops.clone(),
            "bool",
            UNICODEPTR.clone(),
            LowLevelType::Bool,
            unicode_repr() as Arc<dyn Repr>,
            s_uni,
        );

        let result = unicode_repr()
            .rtype_bool(&hop)
            .unwrap_or_else(|err| panic!("rtype_bool: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_unicode_is_true"),
            "expected funcptr 'll_unicode_is_true' in {dbg}"
        );
    }

    /// rstr.py:124-132 + rmodel.py:199-207 — when `can_be_None ==
    /// False`, `rtype_bool` falls through to the `Repr.rtype_bool`
    /// default which calls `self.rtype_len(hop)` and wraps the result
    /// with `int_is_true`. Two llops emitted: `direct_call(ll_strlen)`
    /// then `int_is_true(v_len)`.
    #[test]
    fn string_repr_rtype_bool_when_not_can_be_none_falls_back_to_int_is_true_of_length() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let s_str = crate::annotator::model::SomeValue::String(
            crate::annotator::model::SomeString::new(false, false),
        );
        let hop = build_string_unary_hop(
            rtyper.clone(),
            llops.clone(),
            "bool",
            STRPTR.clone(),
            LowLevelType::Bool,
            string_repr() as Arc<dyn Repr>,
            s_str,
        );

        let result = string_repr()
            .rtype_bool(&hop)
            .unwrap_or_else(|err| panic!("rtype_bool: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        let opnames: Vec<&str> = ops.ops.iter().map(|op| op.opname.as_str()).collect();
        assert_eq!(opnames, vec!["direct_call", "int_is_true"]);
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_strlen"),
            "expected funcptr 'll_strlen' in {dbg}"
        );
    }

    /// rstr.py:124-132 + rmodel.py:199-207 mirror for UnicodeRepr —
    /// same shape but the underlying length helper is `ll_unilen`.
    #[test]
    fn unicode_repr_rtype_bool_when_not_can_be_none_falls_back_to_int_is_true_of_length() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let s_uni = crate::annotator::model::SomeValue::UnicodeString(
            crate::annotator::model::SomeUnicodeString::new(false, false),
        );
        let hop = build_string_unary_hop(
            rtyper.clone(),
            llops.clone(),
            "bool",
            UNICODEPTR.clone(),
            LowLevelType::Bool,
            unicode_repr() as Arc<dyn Repr>,
            s_uni,
        );

        let result = unicode_repr()
            .rtype_bool(&hop)
            .unwrap_or_else(|err| panic!("rtype_bool: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        let opnames: Vec<&str> = ops.ops.iter().map(|op| op.opname.as_str()).collect();
        assert_eq!(opnames, vec!["direct_call", "int_is_true"]);
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_unilen"),
            "expected funcptr 'll_unilen' in {dbg}"
        );
    }

    /// rstr.py:385-396 — `unicode(s)` for a byte string emits
    /// `direct_call(ll_str2unicode, s)` and marks the operation as
    /// exception-capable because non-ASCII bytes raise
    /// UnicodeDecodeError inside the helper graph.
    #[test]
    fn string_repr_rtype_unicode_emits_direct_call_to_ll_str2unicode() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let hop = build_string_unary_hop(
            rtyper.clone(),
            llops.clone(),
            "unicode",
            STRPTR.clone(),
            UNICODEPTR.clone(),
            string_repr() as Arc<dyn Repr>,
            crate::annotator::model::SomeValue::String(crate::annotator::model::SomeString::new(
                false, false,
            )),
        );
        *hop.r_result.borrow_mut() = Some(unicode_repr() as Arc<dyn Repr>);

        let result = string_repr()
            .rtype_unicode(&hop)
            .unwrap_or_else(|err| panic!("StringRepr.rtype_unicode: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_str2unicode"),
            "expected ll_str2unicode in {dbg}"
        );
        assert!(ops._called_exception_is_here_or_cannot_occur);
    }

    /// Same upstream method, identity branch: if the source repr already
    /// matches `hop.r_result`, no helper call is emitted.
    #[test]
    fn unicode_repr_rtype_unicode_returns_identity_without_direct_call() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let hop = build_string_unary_hop(
            rtyper.clone(),
            llops.clone(),
            "unicode",
            UNICODEPTR.clone(),
            UNICODEPTR.clone(),
            unicode_repr() as Arc<dyn Repr>,
            crate::annotator::model::SomeValue::UnicodeString(
                crate::annotator::model::SomeUnicodeString::new(false, false),
            ),
        );
        *hop.r_result.borrow_mut() = Some(unicode_repr() as Arc<dyn Repr>);

        let result = unicode_repr()
            .rtype_unicode(&hop)
            .unwrap_or_else(|err| panic!("UnicodeRepr.rtype_unicode: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 0, "unicode identity path emits no llops");
        assert!(ops._called_exception_is_here_or_cannot_occur);
    }

    /// rstr.py:371-383 — `int(s)` records ValueError and calls
    /// `LLHelpers.ll_int(s, 10)`.
    #[test]
    fn string_repr_rtype_int_default_base_emits_direct_call_to_ll_int() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rint::signed_repr;
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_arg = Variable::new();
        v_arg.set_concretetype(Some(STRPTR.clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper,
            SpaceOperation::new(
                "int".to_string(),
                vec![Hlvalue::Variable(v_arg)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::String(
                crate::annotator::model::SomeString::new(false, false),
            ));
        hop.args_r
            .borrow_mut()
            .push(Some(string_repr() as Arc<dyn Repr>));
        *hop.r_result.borrow_mut() = Some(signed_repr() as Arc<dyn Repr>);

        let result = string_repr()
            .rtype_int(&hop)
            .unwrap_or_else(|err| panic!("StringRepr.rtype_int: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c_func) = &ops.ops[0].args[0] else {
            panic!("expected direct_call funcptr constant");
        };
        assert!(format!("{:?}", c_func.value).contains("ll_int"));
        assert!(matches!(
            ops.ops[0].args[2],
            Hlvalue::Constant(Constant {
                value: ConstValue::Int(10),
                ..
            })
        ));
        assert!(ops._called_exception_is_here_or_cannot_occur);
    }

    /// Two-argument `int(u, base)` is inherited by UnicodeRepr and
    /// routes through the unicode-specialised `ll_unicode_int` helper.
    #[test]
    fn unicode_repr_rtype_int_with_base_emits_direct_call_to_ll_unicode_int() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rint::signed_repr;
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_str = Variable::new();
        v_str.set_concretetype(Some(UNICODEPTR.clone()));
        let v_base = Variable::new();
        v_base.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper,
            SpaceOperation::new(
                "int".to_string(),
                vec![Hlvalue::Variable(v_str), Hlvalue::Variable(v_base)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::UnicodeString(
                crate::annotator::model::SomeUnicodeString::new(false, false),
            ));
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Integer(
                crate::annotator::model::SomeInteger::new(false, false),
            ));
        hop.args_r
            .borrow_mut()
            .push(Some(unicode_repr() as Arc<dyn Repr>));
        hop.args_r
            .borrow_mut()
            .push(Some(signed_repr() as Arc<dyn Repr>));
        *hop.r_result.borrow_mut() = Some(signed_repr() as Arc<dyn Repr>);

        let result = unicode_repr()
            .rtype_int(&hop)
            .unwrap_or_else(|err| panic!("UnicodeRepr.rtype_int: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c_func) = &ops.ops[0].args[0] else {
            panic!("expected direct_call funcptr constant");
        };
        assert!(format!("{:?}", c_func.value).contains("ll_unicode_int"));
        assert!(ops._called_exception_is_here_or_cannot_occur);
    }

    /// rstr.py:554-555 — `AbstractCharRepr.ll_str(ch)` delegates to
    /// `LLHelpers.ll_chr2str`, so `str(ch)` emits one direct_call to the
    /// one-character STR allocator helper.
    #[test]
    fn char_repr_rtype_str_emits_direct_call_to_ll_chr2str() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let hop = build_string_unary_hop(
            rtyper.clone(),
            llops.clone(),
            "str",
            LowLevelType::Char,
            STRPTR.clone(),
            char_repr() as Arc<dyn Repr>,
            crate::annotator::model::SomeValue::Char(crate::annotator::model::SomeChar::new(false)),
        );

        let result = char_repr()
            .rtype_str(&hop)
            .unwrap_or_else(|err| panic!("CharRepr.rtype_str: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(dbg.contains("ll_chr2str"), "expected ll_chr2str in {dbg}");
        assert!(ops._called_exception_is_here_or_cannot_occur);
    }

    /// rstr.py:564-565 — `AbstractUniCharRepr.ll_unicode(ch)` returns
    /// `unicode(ch)`, represented here by the UniChar-specialised
    /// one-character UNICODE allocator helper.
    #[test]
    fn unichar_repr_rtype_unicode_emits_direct_call_to_ll_unichr2unicode() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let hop = build_string_unary_hop(
            rtyper.clone(),
            llops.clone(),
            "unicode",
            LowLevelType::UniChar,
            UNICODEPTR.clone(),
            unichar_repr() as Arc<dyn Repr>,
            crate::annotator::model::SomeValue::UnicodeCodePoint(
                crate::annotator::model::SomeUnicodeCodePoint::new(false),
            ),
        );

        let result = unichar_repr()
            .rtype_unicode(&hop)
            .unwrap_or_else(|err| panic!("UniCharRepr.rtype_unicode: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_unichr2unicode"),
            "expected ll_unichr2unicode in {dbg}"
        );
        assert!(ops._called_exception_is_here_or_cannot_occur);
    }

    /// rstr.py:723-726 — `pairtype(AbstractCharRepr, IntegerRepr).rtype_getitem`
    /// constant-0 path returns the char itself via
    /// `hop.inputarg(hop.r_result, arg=0)`. No SpaceOperation is
    /// emitted; the result is identity-coerced from arg[0].
    #[test]
    fn pair_char_int_rtype_getitem_constant_zero_returns_arg0_identity() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_ch = Variable::new();
        v_ch.set_concretetype(Some(LowLevelType::Char));
        let v_idx = Hlvalue::Constant(Constant::new(ConstValue::Int(0)));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Char));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_ch), v_idx],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            crate::annotator::model::SomeValue::Char(crate::annotator::model::SomeChar::new(false)),
            {
                let mut s_idx = crate::annotator::model::SomeInteger::new(true, false);
                s_idx.base.const_box = Some(Constant::new(ConstValue::Int(0)));
                crate::annotator::model::SomeValue::Integer(s_idx)
            },
        ]);
        hop.args_r.borrow_mut().extend([
            Some(char_repr() as Arc<dyn Repr>),
            Some(crate::translator::rtyper::rint::signed_repr() as Arc<dyn Repr>),
        ]);
        *hop.r_result.borrow_mut() = Some(char_repr() as Arc<dyn Repr>);

        let result = pair_char_int_rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("pair (Char, Int) getitem [0]: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        // Constant-0 path emits no SpaceOperation — only the identity
        // inputarg coercion runs (which is a no-op for matched reprs).
        assert_eq!(ops.ops.len(), 0, "constant-0 path should emit zero ops");
        // rstr.py:725 — `hop.exception_cannot_occur()` precedes the
        // inputarg.
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "pair_char_int_rtype_getitem must call hop.exception_cannot_occur()"
        );
    }

    /// rstr.py:723-726 — UniCharRepr+Int constant-0 mirror.
    #[test]
    fn pair_unichar_int_rtype_getitem_constant_zero_returns_arg0_identity() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_uch = Variable::new();
        v_uch.set_concretetype(Some(LowLevelType::UniChar));
        let v_idx = Hlvalue::Constant(Constant::new(ConstValue::Int(0)));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::UniChar));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_uch), v_idx],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            crate::annotator::model::SomeValue::UnicodeCodePoint(
                crate::annotator::model::SomeUnicodeCodePoint::new(false),
            ),
            {
                let mut s_idx = crate::annotator::model::SomeInteger::new(true, false);
                s_idx.base.const_box = Some(Constant::new(ConstValue::Int(0)));
                crate::annotator::model::SomeValue::Integer(s_idx)
            },
        ]);
        hop.args_r.borrow_mut().extend([
            Some(unichar_repr() as Arc<dyn Repr>),
            Some(crate::translator::rtyper::rint::signed_repr() as Arc<dyn Repr>),
        ]);
        *hop.r_result.borrow_mut() = Some(unichar_repr() as Arc<dyn Repr>);

        let result = pair_unichar_int_rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("pair (UniChar, Int) getitem [0]: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 0, "constant-0 path should emit zero ops");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "pair_unichar_int_rtype_getitem must call hop.exception_cannot_occur()"
        );
    }

    /// rstr.py:728-730 — non-zero index falls through to the
    /// AbstractStringRepr+IntegerRepr branch via `ll_chr2str`; the
    /// converted one-character STR then reuses `ll_stritem_nonneg`.
    #[test]
    fn pair_char_int_rtype_getitem_non_zero_index_falls_through_via_ll_chr2str() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_ch = Variable::new();
        v_ch.set_concretetype(Some(LowLevelType::Char));
        let v_idx = Hlvalue::Constant(Constant::new(ConstValue::Int(1)));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Char));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_ch), v_idx],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            crate::annotator::model::SomeValue::Char(crate::annotator::model::SomeChar::new(false)),
            {
                let mut s_idx = crate::annotator::model::SomeInteger::new(true, false);
                s_idx.base.const_box = Some(Constant::new(ConstValue::Int(1)));
                crate::annotator::model::SomeValue::Integer(s_idx)
            },
        ]);
        hop.args_r.borrow_mut().extend([
            Some(char_repr() as Arc<dyn Repr>),
            Some(crate::translator::rtyper::rint::signed_repr() as Arc<dyn Repr>),
        ]);
        *hop.r_result.borrow_mut() = Some(char_repr() as Arc<dyn Repr>);

        let result = pair_char_int_rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("non-zero index fallback via ll_chr2str: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));

        let ops = llops.borrow();
        let opnames: Vec<&str> = ops.ops.iter().map(|op| op.opname.as_str()).collect();
        assert_eq!(opnames, vec!["direct_call", "direct_call"]);
        let Hlvalue::Constant(c_cast) = &ops.ops[0].args[0] else {
            panic!("expected ll_chr2str funcptr");
        };
        let cast_dbg = format!("{:?}", c_cast.value);
        assert!(
            cast_dbg.contains("ll_chr2str"),
            "expected first direct_call to ll_chr2str in {cast_dbg}"
        );
        let Hlvalue::Constant(c_item) = &ops.ops[1].args[0] else {
            panic!("expected ll_stritem_nonneg funcptr");
        };
        let item_dbg = format!("{:?}", c_item.value);
        assert!(
            item_dbg.contains("ll_stritem_nonneg"),
            "expected second direct_call to ll_stritem_nonneg in {item_dbg}"
        );
        assert!(ops._called_exception_is_here_or_cannot_occur);
    }
}
