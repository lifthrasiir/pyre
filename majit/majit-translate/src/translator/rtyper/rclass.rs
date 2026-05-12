//! RPython `rpython/rtyper/rclass.py`.
//!
//! Current port scope:
//!
//! * Module-level type constants `OBJECT_VTABLE`, `CLASSTYPE`, `OBJECT`,
//!   `OBJECTPTR`, `NONGCOBJECT`, `NONGCOBJECTPTR` (rclass.py:160-180).
//!   Backed by `LazyLock<LowLevelType>` to preserve the upstream
//!   module-level singleton semantics (module import = first Lazy deref).
//! * `RootClassRepr` (rclass.py:420-437), `ClassRepr` (rclass.py:191-284),
//!   `InstanceRepr` (rclass.py:467-558) — `__init__` for all three and
//!   the `_setup_repr` vtable-super / object_type-super chains landed,
//!   including readonly/non-readonly attr iteration, `prepare_method`,
//!   `extra_access_sets`, and field reordering by `attr_reverse_size`.
//! * `getclassrepr` / `getinstancerepr` / `buildinstancerepr`
//!   (rclass.py:67-119) — both `classdef is None` and
//!   `classdef != None` arms dispatch. `getclassrepr` caches through
//!   upstream's `classdef.repr` slot, and `getinstancerepr` reads
//!   `_alloc_flavor_` through `classdesc.get_param(...)`.
//!
//! Deferred:
//!
//! * `_check_for_immutable_hints` (rclass.py:560-581),
//!   `special_memory_pressure` + `mutate_*` quasi-immutable fields
//!   (rclass.py:534-546). Phase R2-D, gated on `classdesc.get_param`
//!   / bookkeeper `memory_pressure_types`.
//! * `ClassRepr.init_vtable` + `fill_vtable_root` + `setup_vtable`
//!   (rclass.py:296-418) — needs `lltype.malloc(immortal=True)`,
//!   `attachRuntimeTypeInfo`, `RuntimeTypeInfo`. Phase R3.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};

/// RPython `flags={}` keyword argument threaded through
/// `InstanceRepr.{getfield,setfield}` (rclass.py:987 / :1002) and the
/// `hook_access_field` / `hook_setfield` virtualizable hooks
/// (rclass.py:712-715). Default empty maps to upstream's `{}` literal;
/// `_jit_virtualizable_2_` instance reprs read keys like `'access_directly'`
/// and `'fresh_virtualizable'` from this dict.
pub type Flags = HashMap<String, ConstValue>;
use std::rc::{Rc, Weak};
use std::sync::{Arc, LazyLock};

use crate::annotator::classdesc::ClassDef;
use crate::annotator::description::{ClassDefKey, DescEntry};
use crate::annotator::model::{DescKind, SomePBC, SomeValue};
use crate::flowspace::model::{ConstValue, Constant, Hlvalue, HostObject, Variable};
use crate::jit_codewriter::type_state::{ConcreteType, TypeResolutionState};
use crate::model::{BlockId, FunctionGraph, OpKind, SpaceOperation, ValueId};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    self, _ptr, ForwardReference, LowLevelType, Ptr, PtrTarget, RUNTIME_TYPE_INFO, StructType,
};
use crate::translator::rtyper::pairtype::ReprClassId;
use crate::translator::rtyper::rmodel::{DescOrConst, RTypeResult, Repr, ReprState, mangle};
use crate::translator::rtyper::rtyper::{GenopResult, HighLevelOp, LowLevelOpList, RPythonTyper};

// ---------------------------------------------------------------------
// VtableMethodPtr helper (carried over from the pre-R1 rclass.rs scaffold).
// ---------------------------------------------------------------------

/// Insert a `VtableMethodPtr` op at `(block_id, op_index)` and return the
/// produced funcptr ValueId. Updates `type_state` so downstream passes
/// (`build_value_kinds`, regalloc, flatten) see the funcptr as integer
/// kind — matching `int_guard_value(op.args[0])` in
/// `jtransform.py:546`.
///
/// RPython equivalent: `ClassRepr.getclsfield(vcls, attr, llops)`
/// (`rclass.py:371-377`), which appends `cast_pointer + getfield(vtable,
/// mangled_name)` to `llops`. The full `getclsfield` is ported on
/// [`ClassRepr`] (`rclass.rs:1096`) and on the rooted [`RootClassRepr`]
/// (`rclass.rs:1828`); this freestanding helper stays as a
/// PRE-EXISTING-ADAPTATION bridge for the pyre IR representation of
/// vtable method slots (see the `OpKind::VtableMethodPtr` comment
/// block in `model.rs`).
pub fn class_get_method_ptr(
    graph: &mut FunctionGraph,
    type_state: &mut TypeResolutionState,
    block_id: BlockId,
    op_index: usize,
    receiver: ValueId,
    trait_root: String,
    method_name: String,
) -> ValueId {
    let funcptr = graph.alloc_value();
    let op = SpaceOperation {
        result: Some(funcptr),
        kind: OpKind::VtableMethodPtr {
            receiver,
            trait_root,
            method_name,
        },
    };
    graph.blocks[block_id.0].operations.insert(op_index, op);
    type_state
        .concrete_types
        .insert(funcptr, ConcreteType::Signed);
    funcptr
}

// ---------------------------------------------------------------------
// rclass.py:25-65 — FieldListAccessor / ImmutableRanking / IR_* constants.
// ---------------------------------------------------------------------

/// RPython `class ImmutableRanking` (`rclass.py:45-54`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImmutableRanking {
    pub name: &'static str,
    pub is_immutable: bool,
}

/// RPython `IR_MUTABLE = ImmutableRanking('mutable', False)` (rclass.py:56).
pub const IR_MUTABLE: ImmutableRanking = ImmutableRanking {
    name: "mutable",
    is_immutable: false,
};

/// RPython `IR_IMMUTABLE = ImmutableRanking('immutable', True)` (rclass.py:57).
pub const IR_IMMUTABLE: ImmutableRanking = ImmutableRanking {
    name: "immutable",
    is_immutable: true,
};

// IR_IMMUTABLE_ARRAY / IR_QUASIIMMUTABLE / IR_QUASIIMMUTABLE_ARRAY defer to
// Phase R2 when `ClassRepr._setup_repr` starts consuming field rankings.

// ---------------------------------------------------------------------
// rclass.py:160-180 — OBJECT_VTABLE / OBJECT / NONGCOBJECT module constants.
// ---------------------------------------------------------------------

/// Internal aggregate that materialises the four interdependent
/// module-level types (`OBJECT_VTABLE` / `CLASSTYPE` / `OBJECT` /
/// `OBJECTPTR`) in a single `LazyLock` body, mirroring upstream's
/// mutable-ForwardReference + post-hoc `become()` ordering
/// (rclass.py:160-174).
///
/// Cycle topology:
///
/// ```text
/// OBJECT_VTABLE.instantiate  →  Ptr(FuncType([], OBJECTPTR))
///                                                   ↓
///                                                 OBJECTPTR
///                                                   ↓
///                                                 OBJECT
///                                                   ↓
///                                          OBJECT.typeptr = CLASSTYPE
///                                                   ↓
///                                              CLASSTYPE = Ptr(OBJECT_VTABLE)
///                                                   ↺
/// ```
///
/// Resolved by:
/// 1. Mint an unresolved `vtable_fwd: ForwardReference`.
/// 2. Build `CLASSTYPE = Ptr(ForwardReference(vtable_fwd.clone()))`
///    — clones share `Arc<Mutex<Option<...>>>` so any later
///    `become()` propagates.
/// 3. Build `OBJECT` and `OBJECTPTR` referencing CLASSTYPE.
/// 4. Build the full `object_vtable` Struct (with all 5 fields
///    including `instantiate: Ptr(FuncType([], OBJECTPTR))`).
/// 5. `vtable_fwd.become(struct_value)` — every clone (including
///    the one embedded in CLASSTYPE) sees the resolution via the
///    shared `Arc`.
struct ObjectFamilyTypes {
    object_vtable: LowLevelType,
    classtype: LowLevelType,
    object: LowLevelType,
    objectptr: LowLevelType,
}

static OBJECT_FAMILY: LazyLock<ObjectFamilyTypes> = LazyLock::new(|| {
    use crate::translator::rtyper::lltypesystem::lltype::FuncType;

    // Step 1 — mint the unresolved ForwardReference.
    let vtable_fwd = ForwardReference::new();

    // Step 2 — CLASSTYPE = Ptr(OBJECT_VTABLE) via a Ptr wrapping a
    // clone of `vtable_fwd`. Clones share the resolved state via
    // `Arc<Mutex<_>>`, so step 5 propagates here.
    let classtype = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::ForwardReference(vtable_fwd.clone()),
    }));

    // Step 3a — OBJECT = GcStruct('object', ('typeptr', CLASSTYPE),
    // rtti=True). Same body as before, but built from the local
    // `classtype` rather than the static singleton.
    let object = LowLevelType::Struct(Box::new(StructType::gc_rtti_with_hints(
        "object",
        vec![("typeptr".into(), classtype.clone())],
        vec![
            ("immutable".into(), ConstValue::Bool(true)),
            ("shouldntbenull".into(), ConstValue::Bool(true)),
            ("typeptr".into(), ConstValue::Bool(true)),
        ],
    )));

    // Step 3b — OBJECTPTR = Ptr(OBJECT).
    let LowLevelType::Struct(object_body) = object.clone() else {
        unreachable!("OBJECT must be a Struct");
    };
    let objectptr = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(*object_body),
    }));

    // Step 4 — build the full object_vtable Struct, including the
    // `instantiate: Ptr(FuncType([], OBJECTPTR))` field that closes
    // the cycle.
    let rtti_ptr_type = LowLevelType::Ptr(Box::new(
        Ptr::from_container_type(RUNTIME_TYPE_INFO.clone()).expect(
            "Ptr(RuntimeTypeInfo) must be constructible from the RUNTIME_TYPE_INFO singleton",
        ),
    ));
    let name_ptr_type = crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone();
    // upstream rclass.py:172 — `Ptr(FuncType([], OBJECTPTR))`. The
    // funcptr has zero args and returns OBJECTPTR.
    let instantiate_funcptr_type = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Func(FuncType {
            args: Vec::new(),
            result: objectptr.clone(),
        }),
    }));
    let body = StructType::with_hints(
        "object_vtable",
        vec![
            ("subclassrange_min".into(), LowLevelType::Signed),
            ("subclassrange_max".into(), LowLevelType::Signed),
            ("rtti".into(), rtti_ptr_type),
            ("name".into(), name_ptr_type),
            ("instantiate".into(), instantiate_funcptr_type),
        ],
        vec![
            ("immutable".into(), ConstValue::Bool(true)),
            ("static_immutable".into(), ConstValue::Bool(true)),
        ],
    );

    // Step 5 — resolve.
    vtable_fwd
        .r#become(LowLevelType::Struct(Box::new(body)))
        .expect("OBJECT_VTABLE.become should succeed");
    let object_vtable = LowLevelType::ForwardReference(Box::new(vtable_fwd));

    ObjectFamilyTypes {
        object_vtable,
        classtype,
        object,
        objectptr,
    }
});

/// RPython module-level `OBJECT_VTABLE = lltype.ForwardReference()` resolved
/// via `.become(Struct('object_vtable', ...))` (rclass.py:160, 167-174).
///
/// All 5 head fields land — `subclassrange_min`/`max`, `rtti`, `name`,
/// and `instantiate: Ptr(FuncType([], OBJECTPTR))`. The instantiate
/// funcptr field closes the `OBJECT_VTABLE → OBJECTPTR → OBJECT →
/// CLASSTYPE → OBJECT_VTABLE` cycle, broken by `OBJECT_FAMILY`'s
/// LazyLock body which builds an unresolved ForwardReference first,
/// constructs the four types in dependency order, then `become()`s
/// the ForwardReference to the full vtable Struct.
pub static OBJECT_VTABLE: LazyLock<LowLevelType> =
    LazyLock::new(|| OBJECT_FAMILY.object_vtable.clone());

/// RPython `CLASSTYPE = Ptr(OBJECT_VTABLE)` (rclass.py:161).
pub static CLASSTYPE: LazyLock<LowLevelType> = LazyLock::new(|| OBJECT_FAMILY.classtype.clone());

/// RPython `OBJECT = GcStruct('object', ('typeptr', CLASSTYPE),
/// hints={...}, rtti=True)` (rclass.py:162-165). `rtti=True` funnels
/// through `RttiStruct._install_extras` and mints a
/// `RuntimeTypeInfo` opaque stored on `_runtime_type_info`, so
/// `getRuntimeTypeInfo(OBJECT)` succeeds once R3 consumers
/// (`fill_vtable_root`) land.
pub static OBJECT: LazyLock<LowLevelType> = LazyLock::new(|| OBJECT_FAMILY.object.clone());

/// RPython `OBJECTPTR = Ptr(OBJECT)` (rclass.py:166).
pub static OBJECTPTR: LazyLock<LowLevelType> = LazyLock::new(|| OBJECT_FAMILY.objectptr.clone());

/// RPython `NONGCOBJECT = Struct('nongcobject', ('typeptr', CLASSTYPE))`
/// (rclass.py:176).
pub static NONGCOBJECT: LazyLock<LowLevelType> = LazyLock::new(|| {
    LowLevelType::Struct(Box::new(StructType::new(
        "nongcobject",
        vec![("typeptr".into(), CLASSTYPE.clone())],
    )))
});

/// RPython `NONGCOBJECTPTR = Ptr(NONGCOBJECT)` (rclass.py:177).
pub static NONGCOBJECTPTR: LazyLock<LowLevelType> = LazyLock::new(|| {
    let LowLevelType::Struct(body) = NONGCOBJECT.clone() else {
        panic!("NONGCOBJECT must be a Struct");
    };
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(*body),
    }))
});

/// RPython `Flavor` as stored in `getgcflavor(classdef)` return values /
/// `default_flavor` kwargs. Upstream uses strings `'gc'` / `'raw'` /
/// `'stack'` (rclass.py:180 `LLFLAVOR = {'gc':'gc','raw':'raw','stack':'raw'}`).
/// Rust uses a `Copy + Hash + Eq` enum so the `instance_reprs` cache key
/// `(Option<ClassDefKey>, Flavor)` stays pointer-identity friendly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Flavor {
    Gc,
    Raw,
}

impl Flavor {
    /// RPython `LLFLAVOR[flavor]` (rclass.py:180). `'stack'` folds to
    /// `'raw'` upstream; Rust surfaces it via [`Flavor::from_str_strict`]
    /// when parity of that arm matters.
    pub fn llflavor(self) -> &'static str {
        match self {
            Flavor::Gc => "gc",
            Flavor::Raw => "raw",
        }
    }

    fn from_alloc_flavor(value: &str) -> Result<Self, TyperError> {
        match value {
            "gc" => Ok(Flavor::Gc),
            "raw" | "stack" => Ok(Flavor::Raw),
            other => Err(TyperError::message(format!(
                "unsupported _alloc_flavor_ {other:?} in rclass.getgcflavor"
            ))),
        }
    }
}

fn const_truthy(value: &ConstValue) -> bool {
    match value {
        ConstValue::None => false,
        ConstValue::Placeholder => false,
        ConstValue::Bool(flag) => *flag,
        ConstValue::Int(value) => *value != 0,
        ConstValue::Float(bits) => f64::from_bits(*bits) != 0.0,
        ConstValue::ByteStr(text) => !text.is_empty(),
        ConstValue::UniStr(text) => !text.is_empty(),
        ConstValue::Tuple(items) => !items.is_empty(),
        ConstValue::List(items) => !items.is_empty(),
        ConstValue::Dict(items) => !items.is_empty(),
        ConstValue::Atom(_)
        | ConstValue::Code(_)
        | ConstValue::Function(_)
        | ConstValue::Graphs(_)
        | ConstValue::LowLevelType(_)
        | ConstValue::LLPtr(_)
        | ConstValue::LLAddress(_)
        | ConstValue::SpecTag(_)
        | ConstValue::HostObject(_) => true,
    }
}

fn host_is_unboxed_value_subclass(pyobj: &HostObject) -> bool {
    pyobj.mro().is_some_and(|mro| {
        mro.iter().any(|cls| {
            let qualname = cls.qualname();
            qualname == "UnboxedValue" || qualname.ends_with(".UnboxedValue")
        })
    })
}

fn classdesc_get_param(
    classdef: &Rc<RefCell<ClassDef>>,
    name: &str,
    default: ConstValue,
    inherit: bool,
) -> ConstValue {
    let mut cdesc = Some(classdef.borrow().classdesc.clone());
    while let Some(cdesc_rc) = cdesc {
        let (value, basedesc) = {
            let cdesc_ref = cdesc_rc.borrow();
            (cdesc_ref.pyobj.class_get(name), cdesc_ref.basedesc.clone())
        };
        if let Some(value) = value {
            return value;
        }
        if !inherit {
            break;
        }
        cdesc = basedesc;
    }
    default
}

pub(super) fn getgcflavor(classdef: &Rc<RefCell<ClassDef>>) -> Result<Flavor, TyperError> {
    let alloc_flavor =
        classdesc_get_param(classdef, "_alloc_flavor_", ConstValue::byte_str("gc"), true);
    let Some(alloc_flavor) = alloc_flavor.as_text() else {
        return Err(TyperError::message(
            "classdesc.get_param('_alloc_flavor_') must return a string",
        ));
    };
    Flavor::from_alloc_flavor(&alloc_flavor)
}

fn lowleveltype_size_hint(lltype: &LowLevelType) -> Option<usize> {
    match lltype {
        LowLevelType::Void => None,
        LowLevelType::Bool | LowLevelType::Char => Some(1),
        LowLevelType::UniChar | LowLevelType::SingleFloat => Some(4),
        LowLevelType::Signed
        | LowLevelType::Unsigned
        | LowLevelType::SignedLongLong
        | LowLevelType::UnsignedLongLong
        | LowLevelType::Float
        | LowLevelType::LongFloat
        | LowLevelType::Address
        | LowLevelType::Ptr(_)
        | LowLevelType::InteriorPtr(_) => Some(std::mem::size_of::<usize>()),
        LowLevelType::SignedLongLongLong | LowLevelType::UnsignedLongLongLong => Some(16),
        LowLevelType::Struct(_)
        | LowLevelType::Array(_)
        | LowLevelType::FixedSizeArray(_)
        | LowLevelType::Func(_)
        | LowLevelType::Opaque(_) => None,
        LowLevelType::ForwardReference(fwd) => {
            fwd.resolved().as_ref().and_then(lowleveltype_size_hint)
        }
    }
}

fn attr_reverse_size((_, lltype): &(String, LowLevelType)) -> Option<i64> {
    lowleveltype_size_hint(lltype).map(|size| -(size as i64))
}

fn sort_llfields(llfields: &mut Vec<(String, LowLevelType)>) {
    llfields.sort_by(|left, right| left.0.cmp(&right.0));
    llfields.sort_by_key(attr_reverse_size);
}

fn class_attr_family_key(
    access_set: &Rc<RefCell<crate::annotator::description::ClassAttrFamily>>,
) -> usize {
    Rc::as_ptr(access_set) as usize
}

/// RPython `OBJECT_BY_FLAVOR[LLFLAVOR[gcflavor]]` (rclass.py:179-180,
/// consumed at rclass.py:472). Returns the underlying `LowLevelType` for
/// the root `object_type` of an `InstanceRepr` with `classdef is None`.
pub fn object_by_flavor(flavor: Flavor) -> LowLevelType {
    match flavor {
        Flavor::Gc => OBJECT.clone(),
        Flavor::Raw => NONGCOBJECT.clone(),
    }
}

/// Adapter from a converted [`Constant`] (produced by
/// [`crate::translator::rtyper::rmodel::Repr::convert_const`] /
/// [`crate::translator::rtyper::rmodel::Repr::convert_desc_or_const`])
/// to the
/// [`crate::translator::rtyper::lltypesystem::lltype::LowLevelValue`]
/// shape that [`_ptr::setattr`](crate::translator::rtyper::lltypesystem::lltype::_ptr::setattr)
/// consumes for vtable slot writes (rclass.py:321 `setattr(vtable,
/// mangled_name, llvalue)`).
///
/// Upstream Python performs no explicit conversion — the `Constant`
/// object itself is stored through the live lltype container's
/// `__setattr__`. The Rust port has separate `LowLevelValue` and
/// `ConstValue` enums, so this helper picks the variant that matches
/// the target field type carried on `c.concretetype`.
///
/// Scope: covers the concrete shapes needed by `setup_vtable`'s
/// vtable-field writes — `Void`, `Signed`, `Bool`, `Float`, and `Ptr`.
/// `Unsigned` / `Char` / `UniChar` / `SingleFloat` / `LongFloat` etc.
/// surface as `TyperError` until their repr ports land.
pub(crate) fn constant_to_lowlevel_value(
    c: &crate::flowspace::model::Constant,
) -> Result<lltype::LowLevelValue, TyperError> {
    let target = c.concretetype.as_ref().ok_or_else(|| {
        TyperError::message(format!(
            "constant_to_lowlevel_value: constant {:?} lacks concretetype",
            c.value
        ))
    })?;
    match (target, &c.value) {
        (LowLevelType::Void, _) => Ok(lltype::LowLevelValue::Void),
        (LowLevelType::Signed, ConstValue::Int(n)) => Ok(lltype::LowLevelValue::Signed(*n)),
        (LowLevelType::Signed, ConstValue::Bool(b)) => {
            Ok(lltype::LowLevelValue::Signed(i64::from(*b)))
        }
        (LowLevelType::Bool, ConstValue::Bool(b)) => Ok(lltype::LowLevelValue::Bool(*b)),
        (LowLevelType::Float, ConstValue::Float(bits)) => Ok(lltype::LowLevelValue::Float(*bits)),
        (LowLevelType::Address, ConstValue::LLAddress(addr)) => {
            Ok(lltype::LowLevelValue::Address(addr.clone()))
        }
        (LowLevelType::Ptr(_), ConstValue::LLPtr(ptr)) => {
            Ok(lltype::LowLevelValue::Ptr(ptr.clone()))
        }
        _ => Err(TyperError::message(format!(
            "constant_to_lowlevel_value: unsupported ConstValue→LowLevelValue for target {target:?} and value {:?}",
            c.value
        ))),
    }
}

// ---------------------------------------------------------------------
// rclass.py:191-418 — ClassRepr (classdef != None flavour).
// ---------------------------------------------------------------------

/// RPython `class ClassRepr(Repr)` (rclass.py:191-418).
///
/// Readonly class attrs populate `clsfields` / `allmethods`, and
/// `extra_access_sets` populate `pbcfields`, matching the attr walk in
/// rclass.py:252-271. Method-valued attrs flow through
/// `prepare_method()` before repr lookup so method-descriptor PBCs are
/// rewritten to function PBCs like upstream.
#[derive(Debug)]
pub struct ClassRepr {
    /// RPython `self.rtyper = rtyper` (rclass.py:193). Weak because
    /// `ClassDef.repr` holds the strong repr cache entry, and the
    /// back-edge mirrors R1's `Weak<RPythonAnnotator>` on
    /// `RPythonTyper.annotator`.
    rtyper: Weak<RPythonTyper>,
    /// RPython `self.classdef = classdef` (rclass.py:194). Always
    /// non-None for `ClassRepr`; `classdef is None` routes through
    /// [`RootClassRepr`] per upstream's `class
    /// RootClassRepr(ClassRepr)` override at rclass.py:420.
    classdef: Rc<RefCell<ClassDef>>,
    /// RPython `self.vtable_type = lltype.ForwardReference()`
    /// (rclass.py:195). Stored as `LowLevelType::ForwardReference`; the
    /// inner `Arc<Mutex<Option<LowLevelType>>>` target is shared with
    /// [`ClassRepr::lowleveltype`]'s Ptr target so resolving one also
    /// resolves the other.
    vtable_type: LowLevelType,
    /// RPython `self.lowleveltype = Ptr(self.vtable_type)`
    /// (rclass.py:196).
    lowleveltype: LowLevelType,
    /// RPython `self.clsfields = clsfields` (rclass.py:281). `{name:
    /// (mangled_name, r)}` dict holding the class-level attribute
    /// reprs, populated by `_setup_repr`.
    clsfields: RefCell<HashMap<String, (String, Arc<dyn Repr>)>>,
    /// RPython `self.pbcfields = pbcfields` (rclass.py:282).
    /// Upstream keys by `(access_set, attr)`; Rust stores the
    /// access-set identity pointer alongside `attr`.
    pbcfields: RefCell<HashMap<(usize, String), (String, Arc<dyn Repr>)>>,
    /// RPython `self.allmethods = allmethods` (rclass.py:283).
    /// `{name: True}` set-as-dict upstream; Rust keeps the dict shape
    /// to remain line-by-line portable even though `HashSet<String>`
    /// would be idiomatic.
    allmethods: RefCell<HashMap<String, bool>>,
    /// RPython `self.rbase = getclassrepr(...)` (rclass.py:273).
    rbase: RefCell<Option<ClassReprArc>>,
    /// RPython `self.vtable = None` (rclass.py:284). Lazily filled by
    /// [`Self::init_vtable`] and read by [`Self::getvtable`]. Stores the
    /// solid vtable pointer (`malloc(vtable_type, immortal=True)`).
    vtable: RefCell<Option<_ptr>>,
    /// RPython `Repr._initialized` state machine.
    state: ReprState,
}

impl ClassRepr {
    /// RPython `ClassRepr.__init__(self, rtyper, classdef)`
    /// (rclass.py:192-196).
    ///
    /// Allocates a fresh `ForwardReference` for the vtable container and
    /// stores `Ptr(self.vtable_type)` as the Repr's low-level type. The
    /// `ForwardReference`'s `target: Arc<Mutex<_>>` is cloned into the
    /// `Ptr.TO` variant so both fields observe the same resolution once
    /// `_setup_repr` lands the Struct body.
    pub fn new(rtyper: &Rc<RPythonTyper>, classdef: &Rc<RefCell<ClassDef>>) -> Self {
        let fwd = ForwardReference::new();
        let fwd_for_ptr = fwd.clone();
        let vtable_type = LowLevelType::ForwardReference(Box::new(fwd));
        let lowleveltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::ForwardReference(fwd_for_ptr),
        }));
        ClassRepr {
            rtyper: Rc::downgrade(rtyper),
            classdef: classdef.clone(),
            vtable_type,
            lowleveltype,
            clsfields: RefCell::new(HashMap::new()),
            pbcfields: RefCell::new(HashMap::new()),
            allmethods: RefCell::new(HashMap::new()),
            rbase: RefCell::new(None),
            vtable: RefCell::new(None),
            state: ReprState::new(),
        }
    }

    /// RPython `ClassRepr.classdef` (rclass.py:194).
    pub fn classdef(&self) -> Rc<RefCell<ClassDef>> {
        self.classdef.clone()
    }

    /// RPython `ClassRepr.vtable_type` (rclass.py:195). Exposed so
    /// `ClassRepr._setup_repr` (on a child class) can borrow the parent
    /// vtable_type when building its `('super', rbase.vtable_type)`
    /// Struct entry.
    pub fn vtable_type(&self) -> &LowLevelType {
        &self.vtable_type
    }

    /// Read-only view of the class-level `clsfields` dict populated by
    /// `_setup_repr`. RPython `self.clsfields` (rclass.py:281).
    pub fn clsfields(&self) -> std::cell::Ref<'_, HashMap<String, (String, Arc<dyn Repr>)>> {
        self.clsfields.borrow()
    }

    /// Read-only view of the class-level `allmethods` dict. RPython
    /// `self.allmethods` (rclass.py:283).
    pub fn allmethods(&self) -> std::cell::Ref<'_, HashMap<String, bool>> {
        self.allmethods.borrow()
    }

    fn prepare_method(&self, s_value: &SomeValue) -> Result<Option<SomeValue>, TyperError> {
        let SomeValue::PBC(s_pbc) = s_value else {
            return Ok(None);
        };
        let kind = s_pbc
            .get_kind()
            .map_err(|err| TyperError::message(err.to_string()))?;
        if kind != DescKind::Method {
            return Ok(None);
        }
        let s_unbound = ClassDef::lookup_filter(&self.classdef, s_pbc, None, &BTreeMap::new())
            .map_err(|err| TyperError::message(err.to_string()))?;
        let SomeValue::PBC(s_unbound) = s_unbound else {
            return Err(TyperError::message(
                "ClassRepr.prepare_method expected lookup_filter() to return SomePBC",
            ));
        };
        let funcdescs: Vec<DescEntry> = s_unbound
            .descriptions
            .values()
            .map(|entry| {
                let method = entry.as_method().ok_or_else(|| {
                    TyperError::message(
                        "ClassRepr.prepare_method expected MethodDesc entries after lookup_filter",
                    )
                })?;
                Ok(DescEntry::Function(method.borrow().funcdesc.clone()))
            })
            .collect::<Result<_, TyperError>>()?;
        // rclass.py:236 — `return annmodel.SomePBC(funcdescs)`. The
        // upstream constructor uses the `can_be_None=False` default; do
        // not propagate `s_unbound.can_be_none` (would alter the
        // nullable-method-PBC repr/key shape).
        Ok(Some(SomeValue::PBC(SomePBC::new(funcdescs, false))))
    }

    /// RPython `ClassRepr.setup_vtable(self, vtable, r_parentcls)`
    /// (rclass.py:307-336).
    ///
    /// ```python
    /// def setup_vtable(self, vtable, r_parentcls):
    ///     def assign(mangled_name, value):
    ///         if value is None:
    ///             llvalue = r.special_uninitialized_value()
    ///             if llvalue is None:
    ///                 return
    ///         else:
    ///             if (isinstance(value, Constant) and
    ///                     isinstance(value.value, staticmethod)):
    ///                 value = Constant(value.value.__get__(42))
    ///             llvalue = r.convert_desc_or_const(value)
    ///         setattr(vtable, mangled_name, llvalue)
    ///
    ///     for fldname in r_parentcls.clsfields:
    ///         mangled_name, r = r_parentcls.clsfields[fldname]
    ///         if r.lowleveltype is Void:
    ///             continue
    ///         value = self.classdef.classdesc.read_attribute(fldname, None)
    ///         assign(mangled_name, value)
    ///     for (access_set, attr), (mangled_name, r) in r_parentcls.pbcfields.items():
    ///         if self.classdef.classdesc not in access_set.descs:
    ///             continue
    ///         if r.lowleveltype is Void:
    ///             continue
    ///         attrvalue = self.classdef.classdesc.read_attribute(attr, None)
    ///         assign(mangled_name, attrvalue)
    /// ```
    ///
    /// `vtable` points at the vtable sub-struct for the level
    /// `r_parentcls` describes (rclass.py:298-304 walks the super chain
    /// via [`Self::init_vtable`]). Upstream calls `setattr` on the
    /// live `_struct` container; the Rust port routes through
    /// [`_ptr::setattr`] after mapping the converted
    /// [`crate::flowspace::model::Constant`] to a
    /// [`lltype::LowLevelValue`] via [`constant_to_lowlevel_value`].
    pub fn setup_vtable(
        &self,
        vtable: &mut _ptr,
        r_parentcls: &Arc<ClassRepr>,
    ) -> Result<(), TyperError> {
        // Backward-compatible shim — leaf-level write at empty path.
        // [`Self::init_vtable`] uses [`Self::setup_vtable_at_path`]
        // directly to address nested levels.
        self.setup_vtable_at_path(vtable, &[], r_parentcls)
    }

    /// Path-aware variant of [`Self::setup_vtable`] used by
    /// [`Self::init_vtable`] to mutate nested vtable substructs.
    /// `path` is a sequence of `"super"` segments naming the level to
    /// write — empty for the leaf vtable, `["super"]` for the parent
    /// level, and so on. Each `assign` call routes through
    /// [`_ptr::setattr_at_path`] so writes alias the original allocation
    /// instead of being lost on a detached substruct copy returned by
    /// `_ptr.getattr("super")`.
    pub fn setup_vtable_at_path(
        &self,
        vtable: &mut _ptr,
        path: &[&str],
        r_parentcls: &Arc<ClassRepr>,
    ) -> Result<(), TyperError> {
        let classdesc = self.classdef.borrow().classdesc.clone();
        let classdesc_key = crate::annotator::description::DescKey::from_rc(&classdesc);

        // upstream: `for fldname in r_parentcls.clsfields`.
        let clsfields_snapshot: Vec<(String, (String, Arc<dyn Repr>))> = r_parentcls
            .clsfields
            .borrow()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for (fldname, (mangled_name, r)) in &clsfields_snapshot {
            if matches!(r.lowleveltype(), LowLevelType::Void) {
                continue;
            }
            let value = crate::annotator::classdesc::ClassDesc::read_attribute(&classdesc, fldname);
            self.setup_vtable_assign(vtable, path, mangled_name, r, value)?;
        }

        // upstream: extra PBC attributes —
        // `for (access_set, attr), (mangled_name, r) in r_parentcls.pbcfields.items()`.
        let pbcfields_snapshot: Vec<((usize, String), (String, Arc<dyn Repr>))> = r_parentcls
            .pbcfields
            .borrow()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for ((access_set_id, attr), (mangled_name, r)) in &pbcfields_snapshot {
            // upstream: `if self.classdef.classdesc not in access_set.descs: continue`.
            let Some(access_set) = r_parentcls
                .classdef
                .borrow()
                .extra_access_sets
                .values()
                .find_map(|(family, _, _)| {
                    (class_attr_family_key(family) == *access_set_id).then(|| family.clone())
                })
            else {
                continue;
            };
            if !access_set.borrow().descs.contains_key(&classdesc_key) {
                continue;
            }
            if matches!(r.lowleveltype(), LowLevelType::Void) {
                continue;
            }
            let attrvalue =
                crate::annotator::classdesc::ClassDesc::read_attribute(&classdesc, attr);
            self.setup_vtable_assign(vtable, path, mangled_name, r, attrvalue)?;
        }
        Ok(())
    }

    /// Helper that mirrors the nested `def assign(...)` closure in
    /// upstream `ClassRepr.setup_vtable` (rclass.py:310-321). Routes
    /// through [`_ptr::setattr_at_path`] so substruct writes propagate
    /// back to the parent allocation.
    fn setup_vtable_assign(
        &self,
        vtable: &mut _ptr,
        path: &[&str],
        mangled_name: &str,
        r: &Arc<dyn Repr>,
        value: Option<crate::annotator::classdesc::ClassDictEntry>,
    ) -> Result<(), TyperError> {
        let llvalue = match value {
            None => {
                // upstream: `llvalue = r.special_uninitialized_value()`
                // + `if llvalue is None: return`.
                let Some(raw) = r.special_uninitialized_value() else {
                    return Ok(());
                };
                let constant = crate::flowspace::model::Constant::with_concretetype(
                    raw,
                    r.lowleveltype().clone(),
                );
                constant_to_lowlevel_value(&constant)?
            }
            Some(entry) => {
                // upstream: staticmethod unwrap — `if isinstance(value,
                // Constant) and isinstance(value.value, staticmethod):
                // value = Constant(value.value.__get__(42))`.
                let entry = match entry {
                    crate::annotator::classdesc::ClassDictEntry::Constant(c) => {
                        let c = match &c.value {
                            ConstValue::HostObject(host) if host.is_staticmethod() => {
                                let Some(func) = host.staticmethod_func().cloned() else {
                                    return Err(TyperError::message(
                                        "setup_vtable_assign: staticmethod without underlying func",
                                    ));
                                };
                                crate::flowspace::model::Constant::new(ConstValue::HostObject(func))
                            }
                            _ => c,
                        };
                        DescOrConst::Const(c)
                    }
                    crate::annotator::classdesc::ClassDictEntry::Desc(d) => DescOrConst::Desc(d),
                };
                let converted = r.convert_desc_or_const(&entry)?;
                constant_to_lowlevel_value(&converted)?
            }
        };
        if path.is_empty() {
            vtable
                .setattr(mangled_name, llvalue)
                .map_err(TyperError::message)?;
        } else {
            vtable
                .setattr_at_path(path, mangled_name, llvalue)
                .map_err(TyperError::message)?;
        }
        Ok(())
    }

    /// RPython `ClassRepr.init_vtable(self)` (rclass.py:296-305).
    ///
    /// ```python
    /// def init_vtable(self):
    ///     """Create the actual vtable"""
    ///     self.vtable = malloc(self.vtable_type, immortal=True)
    ///     vtable_part = self.vtable
    ///     r_parentcls = self
    ///     while r_parentcls.classdef is not None:
    ///         self.setup_vtable(vtable_part, r_parentcls)
    ///         vtable_part = vtable_part.super
    ///         r_parentcls = r_parentcls.rbase
    ///     self.fill_vtable_root(vtable_part)
    /// ```
    ///
    /// Walks the super-chain from `self` toward the root, invoking
    /// `setup_vtable` for each ClassRepr level and finishing with
    /// `fill_vtable_root` on the root vtable slice (the OBJECT_VTABLE
    /// portion). Idempotent via the `vtable: Option<_ptr>` cache —
    /// callers go through [`Self::getvtable`] instead of calling this
    /// directly.
    pub fn init_vtable(self: &Arc<Self>) -> Result<(), TyperError> {
        // upstream: `self.vtable = malloc(self.vtable_type, immortal=True)`.
        let vtable_lltype = match self.vtable_type.clone() {
            LowLevelType::ForwardReference(fwd) => fwd.resolved().ok_or_else(|| {
                TyperError::message(
                    "ClassRepr.init_vtable: vtable_type ForwardReference not resolved \
                     (call setup() first)",
                )
            })?,
            other => other,
        };
        let mut vtable_root = crate::translator::rtyper::lltypesystem::lltype::malloc(
            vtable_lltype,
            None,
            crate::translator::rtyper::lltypesystem::lltype::MallocFlavor::Gc,
            true,
        )
        .map_err(TyperError::message)?;

        // Walk the super-chain via increasing `path`, calling
        // `setup_vtable_at_path(vtable_root, path, r_parentcls)` at
        // each level. Upstream's `vtable_part = vtable_part.super`
        // re-uses the same allocation by reference; the Rust port
        // models that with a path-tracked address into the root
        // allocation (see [`_ptr::setattr_at_path`]).
        let mut path: Vec<&str> = Vec::new();
        let mut r_parentcls: ClassReprArc = ClassReprArc::Inst(self.clone());
        loop {
            let ClassReprArc::Inst(parent_inst) = &r_parentcls else {
                break;
            };
            // Receiver is `self` (the leaf class) — values come from
            // `self.classdef.classdesc` while the field shape comes from
            // `r_parentcls.clsfields`. Upstream rclass.py:302 reads
            // `self.setup_vtable(vtable_part, r_parentcls)`.
            self.setup_vtable_at_path(&mut vtable_root, &path, parent_inst)?;
            // upstream: `vtable_part = vtable_part.super`.
            path.push("super");
            // upstream: `r_parentcls = r_parentcls.rbase`.
            let next = parent_inst.rbase.borrow().clone().ok_or_else(|| {
                TyperError::message("ClassRepr.init_vtable: rbase missing — call setup() first")
            })?;
            r_parentcls = next;
        }
        // upstream: `self.fill_vtable_root(vtable_part)` — at the
        // OBJECT_VTABLE level, addressed by the accumulated `path`.
        self.fill_vtable_root_at_path(&mut vtable_root, &path)?;

        // Cache and return.
        *self.vtable.borrow_mut() = Some(vtable_root);
        Ok(())
    }

    /// RPython `ClassRepr.fill_vtable_root(self, vtable)` (rclass.py:338-360).
    ///
    /// ```python
    /// def fill_vtable_root(self, vtable):
    ///     if self.classdef is not None:
    ///         vtable.subclassrange_min = self.classdef.minid
    ///         vtable.subclassrange_max = self.classdef.maxid
    ///     else:
    ///         vtable.subclassrange_min = 0
    ///         vtable.subclassrange_max = sys.maxint
    ///     rinstance = getinstancerepr(self.rtyper, self.classdef)
    ///     rinstance.setup()
    ///     if rinstance.gcflavor == 'gc':
    ///         vtable.rtti = getRuntimeTypeInfo(rinstance.object_type)
    ///     ...
    ///     vtable.name = alloc_array_name(name)
    ///     if hasattr(self.classdef, 'my_instantiate_graph'):
    ///         vtable.instantiate = self.rtyper.getcallable(graph)
    /// ```
    ///
    /// Pyre-port deviations:
    /// - `instantiate` slot: deferred; OBJECT_VTABLE omits the
    ///   `instantiate: Ptr(FuncType([], OBJECTPTR))` field, and
    ///   `my_instantiate_graph` is only attached after
    ///   `normalizecalls.create_instantiate_functions` runs.
    pub fn fill_vtable_root(&self, vtable: &mut _ptr) -> Result<(), TyperError> {
        // Backward-compatible shim — leaf-level write at empty path.
        self.fill_vtable_root_at_path(vtable, &[])
    }

    /// Path-aware variant of [`Self::fill_vtable_root`] used by
    /// [`Self::init_vtable`] to address the OBJECT_VTABLE substruct
    /// nested inside the per-class vtable allocation. `path` is the
    /// chain of `"super"` segments leading from the leaf vtable to
    /// the OBJECT_VTABLE level.
    pub fn fill_vtable_root_at_path(
        &self,
        vtable: &mut _ptr,
        path: &[&str],
    ) -> Result<(), TyperError> {
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassRepr.fill_vtable_root: RPythonTyper weak ref expired")
        })?;
        let classdef = self.classdef.clone();
        // upstream: subclassrange_min / subclassrange_max.
        let (minid, maxid) = {
            let cd = classdef.borrow();
            let minid = cd.minid.ok_or_else(|| {
                TyperError::message(
                    "ClassRepr.fill_vtable_root: classdef.minid not set — \
                     run normalizecalls.assign_inheritance_ids first",
                )
            })?;
            let maxid = cd.maxid.ok_or_else(|| {
                TyperError::message(
                    "ClassRepr.fill_vtable_root: classdef.maxid not set — \
                     run normalizecalls.assign_inheritance_ids first",
                )
            })?;
            (minid, maxid)
        };
        let setattr_path =
            |vtable: &mut _ptr, name: &str, val: lltype::LowLevelValue| -> Result<(), TyperError> {
                if path.is_empty() {
                    vtable.setattr(name, val).map_err(TyperError::message)
                } else {
                    vtable
                        .setattr_at_path(path, name, val)
                        .map_err(TyperError::message)
                }
            };
        setattr_path(
            vtable,
            "subclassrange_min",
            lltype::LowLevelValue::Signed(minid),
        )?;
        setattr_path(
            vtable,
            "subclassrange_max",
            lltype::LowLevelValue::Signed(maxid),
        )?;

        // upstream: `rinstance = getinstancerepr(self.rtyper, self.classdef);
        //           rinstance.setup();
        //           if rinstance.gcflavor == 'gc':
        //               vtable.rtti = getRuntimeTypeInfo(rinstance.object_type)`.
        let rinstance = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc)?;
        Repr::setup(rinstance.as_ref() as &dyn Repr)?;
        if rinstance.gcflavor() == Flavor::Gc {
            // `object_type` for non-None classdefs is a
            // ForwardReference that `_setup_repr.r#become` resolves to
            // a `Struct`. `getRuntimeTypeInfo` only accepts `Struct`
            // directly, so resolve before passing.
            let object_lltype = match rinstance.object_type().clone() {
                LowLevelType::ForwardReference(fwd) => fwd.resolved().ok_or_else(|| {
                    TyperError::message(
                        "ClassRepr.fill_vtable_root: InstanceRepr.object_type \
                         ForwardReference not resolved",
                    )
                })?,
                other => other,
            };
            let rtti =
                crate::translator::rtyper::lltypesystem::lltype::getRuntimeTypeInfo(&object_lltype)
                    .map_err(TyperError::message)?;
            setattr_path(vtable, "rtti", lltype::LowLevelValue::Ptr(Box::new(rtti)))?;
        }
        // upstream rclass.py:351-355 — root uses "object"; non-root
        // classdefs use `shortname`, not the fully-qualified name.
        let class_shortname = classdef.borrow().shortname.clone();
        let name_ptr =
            crate::translator::rtyper::lltypesystem::rstr::alloc_array_name(&class_shortname)
                .map_err(TyperError::message)?;
        setattr_path(
            vtable,
            "name",
            lltype::LowLevelValue::Ptr(Box::new(name_ptr)),
        )?;
        // `vtable.instantiate = ...` — deferred
        // (LazyLock cycle on Ptr(FuncType([], OBJECTPTR));
        // normalizecalls.create_instantiate_functions also pending).
        Ok(())
    }

    /// RPython `ClassRepr.getvtable(self)` (rclass.py:286-290).
    ///
    /// ```python
    /// def getvtable(self):
    ///     if self.vtable is None:
    ///         self.init_vtable()
    ///     return cast_vtable_to_typeptr(self.vtable)
    /// ```
    pub fn getvtable(self: &Arc<Self>) -> Result<_ptr, TyperError> {
        if self.vtable.borrow().is_none() {
            self.init_vtable()?;
        }
        let vtable = self
            .vtable
            .borrow()
            .clone()
            .expect("init_vtable post-condition: vtable is Some");
        Ok(cast_vtable_to_typeptr(vtable))
    }

    /// RPython `ClassRepr.getruntime(self, expected_type)` (rclass.py:292-294).
    ///
    /// ```python
    /// def getruntime(self, expected_type):
    ///     assert expected_type == CLASSTYPE
    ///     return self.getvtable()
    /// ```
    pub fn getruntime(self: &Arc<Self>, expected_type: &LowLevelType) -> Result<_ptr, TyperError> {
        if expected_type != &CLASSTYPE.clone() {
            return Err(TyperError::message(format!(
                "ClassRepr.getruntime: expected CLASSTYPE, got {expected_type:?}"
            )));
        }
        self.getvtable()
    }

    /// RPython `ClassRepr.fromtypeptr(self, vcls, llops)`
    /// (rclass.py:362-369).
    ///
    /// ```python
    /// def fromtypeptr(self, vcls, llops):
    ///     self.setup()
    ///     castable(self.lowleveltype, vcls.concretetype)  # sanity check
    ///     return llops.genop('cast_pointer', [vcls],
    ///                        resulttype=self.lowleveltype)
    /// ```
    ///
    /// The upstream `castable` sanity check is deferred — `genop` will
    /// surface a TypeError on a malformed cast at lower codegen, and
    /// our static `castable` helper has not landed.
    pub fn fromtypeptr(
        self: &Arc<Self>,
        vcls: Hlvalue,
        llops: &mut LowLevelOpList,
    ) -> Result<Variable, TyperError> {
        Repr::setup(self.as_ref() as &dyn Repr)?;
        Ok(llops
            .genop(
                "cast_pointer",
                vec![vcls],
                GenopResult::LLType(self.lowleveltype.clone()),
            )
            .expect("cast_pointer with non-Void resulttype yields a Variable"))
    }

    /// RPython `ClassRepr.getclsfield(self, vcls, attr, llops)`
    /// (rclass.py:371-381).
    ///
    /// ```python
    /// def getclsfield(self, vcls, attr, llops):
    ///     if attr in self.clsfields:
    ///         mangled_name, r = self.clsfields[attr]
    ///         v_vtable = self.fromtypeptr(vcls, llops)
    ///         cname = inputconst(Void, mangled_name)
    ///         return llops.genop('getfield', [v_vtable, cname], resulttype=r)
    ///     else:
    ///         if self.classdef is None:
    ///             raise MissingRTypeAttribute(attr)
    ///         return self.rbase.getclsfield(vcls, attr, llops)
    /// ```
    pub fn getclsfield(
        self: &Arc<Self>,
        vcls: Hlvalue,
        attr: &str,
        llops: &mut LowLevelOpList,
    ) -> Result<Variable, TyperError> {
        if let Some((mangled_name, r)) = self.clsfields.borrow().get(attr).cloned() {
            let v_vtable = self.fromtypeptr(vcls, llops)?;
            let cname =
                Constant::with_concretetype(ConstValue::byte_str(mangled_name), LowLevelType::Void);
            return Ok(llops
                .genop(
                    "getfield",
                    vec![Hlvalue::Variable(v_vtable), Hlvalue::Constant(cname)],
                    GenopResult::LLType(r.lowleveltype().clone()),
                )
                .expect("getfield with non-Void result yields a Variable"));
        }
        // upstream: `if self.classdef is None: raise MissingRTypeAttribute(attr)`.
        // ClassRepr always has classdef != None, so route to rbase.
        let rbase = self.rbase.borrow().clone().ok_or_else(|| {
            TyperError::message("ClassRepr.getclsfield: rbase missing — call setup() first")
        })?;
        match rbase {
            ClassReprArc::Inst(inst) => inst.getclsfield(vcls, attr, llops),
            ClassReprArc::Root(root) => root.getclsfield(vcls, attr, llops),
        }
    }

    /// RPython `ClassRepr.setclsfield(self, vcls, attr, vvalue, llops)`
    /// (rclass.py:383-393).
    ///
    /// ```python
    /// def setclsfield(self, vcls, attr, vvalue, llops):
    ///     if attr in self.clsfields:
    ///         mangled_name, r = self.clsfields[attr]
    ///         v_vtable = self.fromtypeptr(vcls, llops)
    ///         cname = inputconst(Void, mangled_name)
    ///         llops.genop('setfield', [v_vtable, cname, vvalue])
    ///     else:
    ///         if self.classdef is None:
    ///             raise MissingRTypeAttribute(attr)
    ///         self.rbase.setclsfield(vcls, attr, vvalue, llops)
    /// ```
    pub fn setclsfield(
        self: &Arc<Self>,
        vcls: Hlvalue,
        attr: &str,
        vvalue: Hlvalue,
        llops: &mut LowLevelOpList,
    ) -> Result<(), TyperError> {
        if let Some((mangled_name, _r)) = self.clsfields.borrow().get(attr).cloned() {
            let v_vtable = self.fromtypeptr(vcls, llops)?;
            let cname =
                Constant::with_concretetype(ConstValue::byte_str(mangled_name), LowLevelType::Void);
            llops.genop(
                "setfield",
                vec![
                    Hlvalue::Variable(v_vtable),
                    Hlvalue::Constant(cname),
                    vvalue,
                ],
                GenopResult::Void,
            );
            return Ok(());
        }
        let rbase = self.rbase.borrow().clone().ok_or_else(|| {
            TyperError::message("ClassRepr.setclsfield: rbase missing — call setup() first")
        })?;
        match rbase {
            ClassReprArc::Inst(inst) => inst.setclsfield(vcls, attr, vvalue, llops),
            ClassReprArc::Root(root) => root.setclsfield(vcls, attr, vvalue, llops),
        }
    }

    /// RPython `ClassRepr.getpbcfield(self, vcls, access_set, attr,
    /// llops)` (rclass.py:395-401).
    ///
    /// ```python
    /// def getpbcfield(self, vcls, access_set, attr, llops):
    ///     if (access_set, attr) not in self.pbcfields:
    ///         raise TyperError("internal error: missing PBC field")
    ///     mangled_name, r = self.pbcfields[access_set, attr]
    ///     v_vtable = self.fromtypeptr(vcls, llops)
    ///     cname = inputconst(Void, mangled_name)
    ///     return llops.genop('getfield', [v_vtable, cname], resulttype=r)
    /// ```
    ///
    /// The Rust port keys `pbcfields` on `(access_set_id: usize, attr:
    /// String)` so callers pass the access-set pointer-identity hash
    /// (via [`class_attr_family_key`]).
    pub fn getpbcfield(
        self: &Arc<Self>,
        vcls: Hlvalue,
        access_set_id: usize,
        attr: &str,
        llops: &mut LowLevelOpList,
    ) -> Result<Variable, TyperError> {
        let (mangled_name, r) = self
            .pbcfields
            .borrow()
            .get(&(access_set_id, attr.to_string()))
            .cloned()
            .ok_or_else(|| {
                TyperError::message(format!(
                    "ClassRepr.getpbcfield: internal error: missing PBC field \
                     (access_set={access_set_id}, attr={attr:?})"
                ))
            })?;
        let v_vtable = self.fromtypeptr(vcls, llops)?;
        let cname =
            Constant::with_concretetype(ConstValue::byte_str(mangled_name), LowLevelType::Void);
        Ok(llops
            .genop(
                "getfield",
                vec![Hlvalue::Variable(v_vtable), Hlvalue::Constant(cname)],
                GenopResult::LLType(r.lowleveltype().clone()),
            )
            .expect("getfield with non-Void result yields a Variable"))
    }

    /// RPython `ClassRepr.rtype_issubtype(self, hop)` (rclass.py:403-414).
    ///
    /// ```python
    /// def rtype_issubtype(self, hop):
    ///     class_repr = get_type_repr(self.rtyper)
    ///     v_cls1, v_cls2 = hop.inputargs(class_repr, class_repr)
    ///     if isinstance(v_cls2, Constant):
    ///         cls2 = v_cls2.value
    ///         minid = hop.inputconst(Signed, cls2.subclassrange_min)
    ///         maxid = hop.inputconst(Signed, cls2.subclassrange_max)
    ///         return hop.gendirectcall(ll_issubclass_const, v_cls1, minid, maxid)
    ///     else:
    ///         v_cls1, v_cls2 = hop.inputargs(class_repr, class_repr)
    ///         return hop.gendirectcall(ll_issubclass, v_cls1, v_cls2)
    /// ```
    ///
    /// Both branches dispatch to a low-level helper graph minted via
    /// [`RPythonTyper::lowlevel_helper_function`]: `ll_issubclass`
    /// (rclass.py:1133-1137) for the variable case and
    /// `ll_issubclass_const` (rclass.py:1139-1140) for the constant
    /// case. The helper bodies emit a single `int_between` op against
    /// `cls.subclassrange_min` / `cls.subclassrange_max`.
    /// Inherent method body — receiver is `&self` so the
    /// [`Repr::rtype_issubtype`] trait override can forward to it
    /// without needing the `Arc<Self>` smart pointer (no upstream
    /// caller depends on the Arc handle here; only `self.rtyper.upgrade()`
    /// is touched).
    pub fn rtype_issubtype(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> RTypeResult {
        rtype_issubtype_helper(&self.rtyper, hop, "ClassRepr.rtype_issubtype")
    }
}

/// Shared body of `AbstractClassRepr.rtype_issubtype` (rclass.py:403-414).
///
/// Upstream both `ClassRepr` and `RootClassRepr` inherit this method
/// from `AbstractClassRepr`. The Rust port has them as separate
/// structs, so the body is hoisted into a free helper that both
/// `Repr::rtype_issubtype` overrides forward to. The only
/// per-receiver state the body uses is the `rtyper` weak ref, so the
/// helper takes that explicitly.
fn rtype_issubtype_helper(
    rtyper_weak: &Weak<RPythonTyper>,
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    caller: &'static str,
) -> RTypeResult {
    use crate::translator::rtyper::rtyper::ConvertedTo;
    let rtyper = rtyper_weak
        .upgrade()
        .ok_or_else(|| TyperError::message(format!("{caller}: rtyper weak ref expired")))?;
    let class_repr = get_type_repr(&rtyper)?;
    let class_repr_dyn: &dyn Repr = class_repr.as_ref();
    let v_args = hop.inputargs(vec![
        ConvertedTo::Repr(class_repr_dyn),
        ConvertedTo::Repr(class_repr_dyn),
    ])?;
    let v_cls1 = v_args[0].clone();
    let v_cls2 = v_args[1].clone();

    // upstream `if isinstance(v_cls2, Constant)`. The constant
    // branch unpacks `cls2.value.subclassrange_{min,max}` so we
    // need an `_ptr` carrying `subclassrange_min/max` Signed
    // fields (the OBJECT_VTABLE struct, after fill_vtable_root).
    if let Hlvalue::Constant(c) = &v_cls2 {
        let ConstValue::LLPtr(c_ptr) = &c.value else {
            return Err(TyperError::message(format!(
                "{caller}: constant cls2 must carry an _ptr, got {:?}",
                c.value
            )));
        };
        let min_val = c_ptr
            .getattr("subclassrange_min")
            .map_err(TyperError::message)?;
        let max_val = c_ptr
            .getattr("subclassrange_max")
            .map_err(TyperError::message)?;
        let lltype::LowLevelValue::Signed(min_n) = min_val else {
            return Err(TyperError::message(format!(
                "{caller}: subclassrange_min not Signed, got {min_val:?}"
            )));
        };
        let lltype::LowLevelValue::Signed(max_n) = max_val else {
            return Err(TyperError::message(format!(
                "{caller}: subclassrange_max not Signed, got {max_val:?}"
            )));
        };
        let c_min = Constant::with_concretetype(ConstValue::Int(min_n), LowLevelType::Signed);
        let c_max = Constant::with_concretetype(ConstValue::Int(max_n), LowLevelType::Signed);
        // upstream: `gendirectcall(ll_issubclass_const, v_cls1,
        //   minid, maxid)`.
        let helper = rtyper.lowlevel_helper_function(
            "ll_issubclass_const",
            vec![
                CLASSTYPE.clone(),
                LowLevelType::Signed,
                LowLevelType::Signed,
            ],
            LowLevelType::Bool,
        )?;
        return hop.gendirectcall(
            &helper,
            vec![v_cls1, Hlvalue::Constant(c_min), Hlvalue::Constant(c_max)],
        );
    }

    // upstream variable case: `gendirectcall(ll_issubclass, v_cls1, v_cls2)`.
    let helper = rtyper.lowlevel_helper_function(
        "ll_issubclass",
        vec![CLASSTYPE.clone(), CLASSTYPE.clone()],
        LowLevelType::Bool,
    )?;
    hop.gendirectcall(&helper, vec![v_cls1, v_cls2])
}

/// RPython `cast_vtable_to_typeptr(vtable)` (rclass.py:182-185).
///
/// ```python
/// def cast_vtable_to_typeptr(vtable):
///     while typeOf(vtable).TO != OBJECT_VTABLE:
///         vtable = vtable.super
///     return vtable
/// ```
///
/// Walks the per-class vtable Struct's `super` chain inward until the
/// pointee type matches `OBJECT_VTABLE` (the root vtable Struct). At
/// the OBJECT_VTABLE level the upstream `vtable.super` access would
/// raise `AttributeError`, so the walk terminates by struct-type
/// equality before that point.
pub fn cast_vtable_to_typeptr(mut vtable: _ptr) -> _ptr {
    let object_vtable = OBJECT_VTABLE.clone();
    loop {
        let to_lltype = LowLevelType::from(vtable._TYPE.TO.clone());
        if to_lltype == object_vtable {
            return vtable;
        }
        // upstream: `vtable = vtable.super`. The Rust port follows
        // `_ptr.getattr("super")` which yields LowLevelValue::Ptr for
        // raw substructs (per `_ptr._expose`).
        let super_lv = vtable
            .getattr("super")
            .expect("vtable.super: per-class vtable must have a 'super' field");
        let crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Ptr(super_ptr) =
            super_lv
        else {
            panic!(
                "cast_vtable_to_typeptr: vtable.super did not yield a Ptr value (got {:?})",
                super_lv
            );
        };
        vtable = *super_ptr;
    }
}

impl Repr for ClassRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "ClassRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::Repr
    }

    /// `RPythonTyper.translate_op_issubtype` (rtyper.py:498-500)
    /// dispatches `r.rtype_issubtype(hop)`. Without this override the
    /// trait default would surface `MissingRTypeOperation`. Forwards
    /// to the inherent [`ClassRepr::rtype_issubtype`] which mirrors
    /// upstream rclass.py:403-414.
    fn rtype_issubtype(&self, hop: &crate::translator::rtyper::rtyper::HighLevelOp) -> RTypeResult {
        ClassRepr::rtype_issubtype(self, hop)
    }

    /// RPython `ClassRepr.convert_desc(self, desc)` (rclass.py:212-220).
    ///
    /// ```python
    /// def convert_desc(self, desc):
    ///     subclassdef = desc.getuniqueclassdef()
    ///     if self.classdef is not None:
    ///         if self.classdef.commonbase(subclassdef) != self.classdef:
    ///             raise TyperError("not a subclass of %r: %r" % (
    ///                 self.classdef.name, desc))
    ///     r_subclass = getclassrepr(self.rtyper, subclassdef)
    ///     return r_subclass.getruntime(self.lowleveltype)
    /// ```
    fn convert_desc(
        &self,
        desc: &crate::annotator::description::DescEntry,
    ) -> Result<Constant, TyperError> {
        let crate::annotator::description::DescEntry::Class(class_rc) = desc else {
            return Err(TyperError::message(format!(
                "ClassRepr.convert_desc: expected ClassDesc, got {desc:?}"
            )));
        };
        let subclassdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc)
            .map_err(|e| TyperError::message(e.to_string()))?;
        // upstream `self.classdef.commonbase(subclassdef) != self.classdef`
        // — verify the candidate is a subclass of the receiver classdef.
        let common =
            crate::annotator::classdesc::ClassDef::commonbase(&self.classdef, &subclassdef);
        let same_class = match &common {
            Some(c) => Rc::ptr_eq(c, &self.classdef),
            None => false,
        };
        if !same_class {
            return Err(TyperError::message(format!(
                "ClassRepr.convert_desc: {:?} is not a subclass of {:?}",
                subclassdef.borrow().name,
                self.classdef.borrow().name
            )));
        }
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassRepr.convert_desc: rtyper weak ref expired")
        })?;
        let r_subclass = getclassrepr_arc(&rtyper, Some(&subclassdef))?;
        let vtable_ptr = r_subclass.getruntime(&self.lowleveltype)?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(vtable_ptr)),
            self.lowleveltype.clone(),
        ))
    }

    /// RPython `ClassRepr.convert_const(self, value)` (rclass.py:222-226).
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if not isinstance(value, (type, types.ClassType)):
    ///         raise TyperError("not a class: %r" % (value,))
    ///     bk = self.rtyper.annotator.bookkeeper
    ///     return self.convert_desc(bk.getdesc(value))
    /// ```
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let ConstValue::HostObject(host) = value else {
            return Err(TyperError::message(format!(
                "ClassRepr.convert_const: not a class: {value:?}"
            )));
        };
        if !host.is_class() {
            return Err(TyperError::message(format!(
                "ClassRepr.convert_const: not a class: {host:?}"
            )));
        }
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassRepr.convert_const: rtyper weak ref expired")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("ClassRepr.convert_const: annotator weak ref dropped")
        })?;
        let desc = annotator
            .bookkeeper
            .getdesc(host)
            .map_err(|e| TyperError::message(e.to_string()))?;
        self.convert_desc(&desc)
    }

    /// RPython `ClassRepr._setup_repr` (rclass.py:242-284).
    fn _setup_repr(&self) -> Result<(), TyperError> {
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassRepr._setup_repr: RPythonTyper weak ref expired")
        })?;
        let (name, basedef) = {
            let cd = self.classdef.borrow();
            (cd.name.clone(), cd.basedef.clone())
        };

        // rclass.py:247-249 — `clsfields = {}; pbcfields = {};
        // allmethods = {}`. Populate local copies first (upstream's
        // "don't store mutable objects on self before they are fully
        // built" rule at rclass.py:243-245).
        let mut clsfields: HashMap<String, (String, Arc<dyn Repr>)> = HashMap::new();
        let mut pbcfields: HashMap<(usize, String), (String, Arc<dyn Repr>)> = HashMap::new();
        let mut allmethods: HashMap<String, bool> = HashMap::new();
        let mut llfields: Vec<(String, LowLevelType)> = Vec::new();

        // rclass.py:252-262 — attrs iteration with readonly filter.
        let attrs_sorted = {
            let cd = self.classdef.borrow();
            cd.attrs
                .iter()
                .map(|(k, v)| (k.clone(), (v.s_value.clone(), v.readonly)))
                .collect::<BTreeMap<String, (SomeValue, bool)>>()
        };
        for (attr_name, (s_value, readonly)) in &attrs_sorted {
            if !readonly {
                continue;
            }
            let mut s_value = s_value.clone();
            if let Some(s_unboundmethod) = self.prepare_method(&s_value)? {
                allmethods.insert(attr_name.clone(), true);
                s_value = s_unboundmethod;
            }
            let r = rtyper.getrepr(&s_value)?;
            let mangled_name = format!("cls_{attr_name}");
            clsfields.insert(attr_name.clone(), (mangled_name.clone(), r.clone()));
            llfields.push((mangled_name, r.lowleveltype().clone()));
        }
        for (access_set, attr_name, counter) in
            self.classdef.borrow().extra_access_sets.values().cloned()
        {
            let r = rtyper.getrepr(&access_set.borrow().s_value)?;
            let mangled_name = mangle(&format!("pbc{counter}"), &attr_name);
            pbcfields.insert(
                (class_attr_family_key(&access_set), attr_name.clone()),
                (mangled_name.clone(), r.clone()),
            );
            llfields.push((mangled_name, r.lowleveltype().clone()));
        }
        sort_llfields(&mut llfields);

        // rclass.py:273 — `self.rbase = getclassrepr(self.rtyper,
        // self.classdef.basedef)`.
        let rbase = getclassrepr_arc(&rtyper, basedef.as_ref())?;
        // rclass.py:274 — `self.rbase.setup()`.
        Repr::setup(rbase.as_repr().as_ref())?;

        // rclass.py:275-278 — `kwds = {'hints': {...}}; vtable_type =
        // Struct('%s_vtable' % name, ('super', rbase.vtable_type),
        // *llfields, **kwds)`.
        let super_field_type = rbase.vtable_type().clone();
        let mut fields = Vec::with_capacity(1 + llfields.len());
        fields.push(("super".into(), super_field_type));
        fields.extend(llfields);
        let vtable_body = StructType::with_hints(
            &format!("{name}_vtable"),
            fields,
            vec![
                ("immutable".into(), ConstValue::Bool(true)),
                ("static_immutable".into(), ConstValue::Bool(true)),
            ],
        );
        // rclass.py:279 — `self.vtable_type.become(vtable_type)`. The
        // inner `Arc<Mutex<_>>` propagates to `self.lowleveltype`'s Ptr
        // target via the clone in `new()`.
        let LowLevelType::ForwardReference(fwd) = &self.vtable_type else {
            return Err(TyperError::message(
                "ClassRepr.vtable_type must be LowLevelType::ForwardReference",
            ));
        };
        fwd.r#become(LowLevelType::Struct(Box::new(vtable_body)))
            .map_err(TyperError::message)?;

        // rclass.py:280 — `allmethods.update(self.rbase.allmethods)`.
        // Merge whichever parent methods upstream already recorded.
        if let ClassReprArc::Inst(parent_inst) = &rbase {
            for (method_name, _) in parent_inst.allmethods.borrow().iter() {
                allmethods.insert(method_name.clone(), true);
            }
        }

        // rclass.py:281-284 — publish the populated dicts onto `self`.
        *self.clsfields.borrow_mut() = clsfields;
        *self.pbcfields.borrow_mut() = pbcfields;
        *self.allmethods.borrow_mut() = allmethods;
        *self.rbase.borrow_mut() = Some(rbase);
        // `self.vtable = None` is represented by the absence of any R3
        // vtable slot on this struct.
        Ok(())
    }
}

/// Typed return for [`getclassrepr_arc`]: the internal recursion in
/// [`ClassRepr::_setup_repr`] needs direct access to `rbase.vtable_type`
/// and the Repr `setup()` entry-point, both of which are awkward to
/// reach through the public `Arc<dyn Repr>`. The enum branches mirror
/// upstream's class hierarchy (`ClassRepr` vs its `RootClassRepr`
/// subclass at rclass.py:420) without requiring runtime type reflection.
#[derive(Clone, Debug)]
pub enum ClassReprArc {
    Root(Arc<RootClassRepr>),
    Inst(Arc<ClassRepr>),
}

impl ClassReprArc {
    /// RPython `rbase.vtable_type` reader (rclass.py:277). Returns the
    /// `LowLevelType` that `ClassRepr._setup_repr` plugs into the
    /// `('super', ...)` field of the child vtable Struct.
    pub fn vtable_type(&self) -> &LowLevelType {
        match self {
            Self::Root(r) => r.vtable_type(),
            Self::Inst(r) => r.vtable_type(),
        }
    }

    /// Upcast to `Arc<dyn Repr>` so callers can invoke `Repr::setup` or
    /// hand the value to code that only needs the Repr surface (e.g.
    /// [`ExceptionData::r_exception_type`]).
    pub fn as_repr(&self) -> Arc<dyn Repr> {
        match self {
            Self::Root(r) => r.clone() as Arc<dyn Repr>,
            Self::Inst(r) => r.clone() as Arc<dyn Repr>,
        }
    }

    /// RPython `ClassRepr.getruntime(self, expected_type)` polymorphic
    /// entry. Both branches assert `expected_type == CLASSTYPE` and
    /// return `getvtable()`.
    pub fn getruntime(&self, expected_type: &LowLevelType) -> Result<_ptr, TyperError> {
        match self {
            Self::Root(r) => r.getruntime(expected_type),
            Self::Inst(r) => r.getruntime(expected_type),
        }
    }
}

// ---------------------------------------------------------------------
// rclass.py:420-438 — RootClassRepr (classdef = None flavour).
// ---------------------------------------------------------------------

/// RPython `class RootClassRepr(ClassRepr)` (rclass.py:420-437).
///
/// Upstream inherits all `ClassRepr` methods and overrides `__init__` /
/// `_setup_repr` / `init_vtable`. The Rust port keeps `RootClassRepr` as
/// a sibling struct sharing the [`ClassReprArc`] dispatch enum because
/// Rust has no structural inheritance. [`RootClassRepr::_setup_repr`] is
/// a no-op (clsfields/pbcfields/allmethods/vtable all start empty) until
/// R2-B / R3 add their storage.
#[derive(Debug)]
pub struct RootClassRepr {
    /// RPython `self.rtyper = rtyper` (rclass.py:425). Weak backref —
    /// `init_vtable` calls `getinstancerepr(rtyper, None)` for the
    /// `rtti` slot of the root vtable.
    rtyper: Weak<RPythonTyper>,
    /// RPython `self.classdef = None` (rclass.py:422).
    classdef: Option<ClassDefKey>,
    /// RPython `self.vtable_type = OBJECT_VTABLE` (rclass.py:426).
    vtable_type: LowLevelType,
    /// RPython `self.lowleveltype = Ptr(self.vtable_type)` (rclass.py:427).
    lowleveltype: LowLevelType,
    /// RPython `self.vtable = None` (rclass.py:433). Lazily filled by
    /// [`Self::init_vtable`].
    vtable: RefCell<Option<_ptr>>,
    /// RPython `Repr._initialized` state machine.
    state: ReprState,
}

impl RootClassRepr {
    /// RPython `RootClassRepr.__init__(self, rtyper)` (rclass.py:424-427).
    pub fn new(rtyper: &Rc<RPythonTyper>) -> Self {
        RootClassRepr {
            rtyper: Rc::downgrade(rtyper),
            classdef: None,
            vtable_type: OBJECT_VTABLE.clone(),
            lowleveltype: CLASSTYPE.clone(),
            vtable: RefCell::new(None),
            state: ReprState::new(),
        }
    }

    /// RPython `RootClassRepr.classdef` (`classdef = None`, rclass.py:422).
    pub fn classdef(&self) -> Option<ClassDefKey> {
        self.classdef
    }

    /// RPython `RootClassRepr.vtable_type` (rclass.py:426).
    pub fn vtable_type(&self) -> &LowLevelType {
        &self.vtable_type
    }

    /// RPython `RootClassRepr.init_vtable(self)` (rclass.py:435-437).
    ///
    /// ```python
    /// def init_vtable(self):
    ///     self.vtable = malloc(self.vtable_type, immortal=True)
    ///     self.fill_vtable_root(self.vtable)
    /// ```
    ///
    /// The root has no `super` chain, so allocation + a single
    /// `fill_vtable_root` finalises the OBJECT_VTABLE slot. Idempotent
    /// via the `vtable: Option<_ptr>` cache.
    pub fn init_vtable(&self) -> Result<(), TyperError> {
        // upstream malloc rejects ForwardReference, so resolve the
        // OBJECT_VTABLE forward-ref before passing.
        let vtable_lltype = match self.vtable_type.clone() {
            LowLevelType::ForwardReference(fwd) => fwd.resolved().ok_or_else(|| {
                TyperError::message(
                    "RootClassRepr.init_vtable: OBJECT_VTABLE ForwardReference not resolved",
                )
            })?,
            other => other,
        };
        let mut vtable = crate::translator::rtyper::lltypesystem::lltype::malloc(
            vtable_lltype,
            None,
            crate::translator::rtyper::lltypesystem::lltype::MallocFlavor::Gc,
            true,
        )
        .map_err(TyperError::message)?;
        self.fill_vtable_root(&mut vtable)?;
        *self.vtable.borrow_mut() = Some(vtable);
        Ok(())
    }

    /// RPython `ClassRepr.fill_vtable_root(self, vtable)` (rclass.py:338-360)
    /// adapted for the `classdef is None` arm.
    ///
    /// Upstream uses `0` / `sys.maxint` for the root subclassrange. The
    /// Rust port uses `i64::MAX` for the upper bound — RPython's
    /// `sys.maxint` is the platform `intptr_t.max`; on 64-bit hosts
    /// (the only target pyre supports today) that is `i64::MAX`. The
    /// `rtti` slot is populated via `getinstancerepr(rtyper, None)`
    /// + `getRuntimeTypeInfo(rinstance.object_type)`. The `name` slot
    /// is the upstream `"object"` string; `instantiate` is deferred
    /// until the `Ptr(FuncType([], OBJECTPTR))` cycle is resolved.
    pub fn fill_vtable_root(&self, vtable: &mut _ptr) -> Result<(), TyperError> {
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("RootClassRepr.fill_vtable_root: RPythonTyper weak ref expired")
        })?;
        // upstream: classdef is None → `subclassrange_min = 0,
        // subclassrange_max = sys.maxint`.
        vtable
            .setattr(
                "subclassrange_min",
                crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Signed(0),
            )
            .map_err(TyperError::message)?;
        vtable
            .setattr(
                "subclassrange_max",
                crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Signed(i64::MAX),
            )
            .map_err(TyperError::message)?;
        // upstream: `rinstance = getinstancerepr(self.rtyper,
        // self.classdef); rinstance.setup(); if rinstance.gcflavor ==
        // 'gc': vtable.rtti = getRuntimeTypeInfo(rinstance.object_type)`.
        let rinstance = getinstancerepr(&rtyper, None, Flavor::Gc)?;
        Repr::setup(rinstance.as_ref() as &dyn Repr)?;
        if rinstance.gcflavor() == Flavor::Gc {
            let rtti = crate::translator::rtyper::lltypesystem::lltype::getRuntimeTypeInfo(
                rinstance.object_type(),
            )
            .map_err(TyperError::message)?;
            vtable
                .setattr(
                    "rtti",
                    crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Ptr(Box::new(
                        rtti,
                    )),
                )
                .map_err(TyperError::message)?;
        }
        let name_ptr = crate::translator::rtyper::lltypesystem::rstr::alloc_array_name("object")
            .map_err(TyperError::message)?;
        vtable
            .setattr(
                "name",
                crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Ptr(Box::new(
                    name_ptr,
                )),
            )
            .map_err(TyperError::message)?;
        // `vtable.instantiate` deferred — see `ClassRepr::fill_vtable_root`.
        Ok(())
    }

    /// RPython `ClassRepr.getvtable(self)` (rclass.py:286-290) for the
    /// rootclass.
    pub fn getvtable(&self) -> Result<_ptr, TyperError> {
        if self.vtable.borrow().is_none() {
            self.init_vtable()?;
        }
        let vtable = self
            .vtable
            .borrow()
            .clone()
            .expect("init_vtable post-condition: vtable is Some");
        // The root vtable IS the OBJECT_VTABLE level, so
        // `cast_vtable_to_typeptr` is a no-op (the `while typeOf(vtable).TO
        // != OBJECT_VTABLE` loop terminates immediately).
        Ok(cast_vtable_to_typeptr(vtable))
    }

    /// RPython `ClassRepr.getruntime(self, expected_type)` (rclass.py:292-294)
    /// for the rootclass.
    pub fn getruntime(&self, expected_type: &LowLevelType) -> Result<_ptr, TyperError> {
        if expected_type != &CLASSTYPE.clone() {
            return Err(TyperError::message(format!(
                "RootClassRepr.getruntime: expected CLASSTYPE, got {expected_type:?}"
            )));
        }
        self.getvtable()
    }

    /// RPython `ClassRepr.fromtypeptr(self, vcls, llops)` (rclass.py:362-369)
    /// for the rootclass — `lowleveltype` is `CLASSTYPE` and the cast
    /// reduces to a no-op-style `cast_pointer` that confirms the type.
    pub fn fromtypeptr(
        self: &Arc<Self>,
        vcls: Hlvalue,
        llops: &mut LowLevelOpList,
    ) -> Result<Variable, TyperError> {
        Repr::setup(self.as_ref() as &dyn Repr)?;
        Ok(llops
            .genop(
                "cast_pointer",
                vec![vcls],
                GenopResult::LLType(self.lowleveltype.clone()),
            )
            .expect("cast_pointer with non-Void resulttype yields a Variable"))
    }

    /// RPython `ClassRepr.getclsfield` terminal arm (rclass.py:378-381):
    /// `raise MissingRTypeAttribute(attr)` when the rootclass has no
    /// matching clsfield. Pyre surfaces a structured `TyperError`.
    pub fn getclsfield(
        self: &Arc<Self>,
        _vcls: Hlvalue,
        attr: &str,
        _llops: &mut LowLevelOpList,
    ) -> Result<Variable, TyperError> {
        Err(TyperError::message(format!(
            "RootClassRepr.getclsfield: MissingRTypeAttribute({attr:?}) — \
             attribute not found in any clsfields"
        )))
    }

    /// RPython `ClassRepr.setclsfield` terminal arm (rclass.py:390-393):
    /// `raise MissingRTypeAttribute(attr)`.
    pub fn setclsfield(
        self: &Arc<Self>,
        _vcls: Hlvalue,
        attr: &str,
        _vvalue: Hlvalue,
        _llops: &mut LowLevelOpList,
    ) -> Result<(), TyperError> {
        Err(TyperError::message(format!(
            "RootClassRepr.setclsfield: MissingRTypeAttribute({attr:?}) — \
             attribute not found in any clsfields"
        )))
    }
}

impl Repr for RootClassRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        // Upstream `ClassRepr.__repr__` (rclass.py:198-203) emits
        // `"<ClassRepr for object>"`; the pyre tag follows the class name
        // matching the `.__class__.__name__` of RootClassRepr.
        "RootClassRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::Repr
    }

    /// `RPythonTyper.translate_op_issubtype` (rtyper.py:498-500)
    /// dispatches `r.rtype_issubtype(hop)`. Upstream's
    /// `RootClassRepr` inherits the body from `AbstractClassRepr`
    /// (rclass.py:403-414); the Rust port forwards through the shared
    /// [`rtype_issubtype_helper`] free fn so both ClassRepr and
    /// RootClassRepr surface the same op.
    fn rtype_issubtype(&self, hop: &crate::translator::rtyper::rtyper::HighLevelOp) -> RTypeResult {
        rtype_issubtype_helper(&self.rtyper, hop, "RootClassRepr.rtype_issubtype")
    }

    /// RPython `RootClassRepr._setup_repr(self)` (rclass.py:429-433).
    ///
    /// ```python
    /// def _setup_repr(self):
    ///     self.clsfields = {}
    ///     self.pbcfields = {}
    ///     self.allmethods = {}
    ///     self.vtable = None
    /// ```
    ///
    /// All four assignments are currently no-ops because Phase R1 holds
    /// no storage for them — the fields only exist upstream to keep
    /// `ClassRepr` method dispatch consistent, and the methods that read
    /// them (`getvtable`, `getclsfield`, `get_field`) land with Phase R2.
    fn _setup_repr(&self) -> Result<(), TyperError> {
        Ok(())
    }

    /// RPython `ClassRepr.convert_desc(self, desc)` (rclass.py:212-220),
    /// inherited by `RootClassRepr`. With `classdef is None` the
    /// commonbase check is skipped (upstream `if self.classdef is not
    /// None: ...`).
    fn convert_desc(
        &self,
        desc: &crate::annotator::description::DescEntry,
    ) -> Result<Constant, TyperError> {
        let crate::annotator::description::DescEntry::Class(class_rc) = desc else {
            return Err(TyperError::message(format!(
                "RootClassRepr.convert_desc: expected ClassDesc, got {desc:?}"
            )));
        };
        let subclassdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc)
            .map_err(|e| TyperError::message(e.to_string()))?;
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("RootClassRepr.convert_desc: rtyper weak ref expired")
        })?;
        let r_subclass = getclassrepr_arc(&rtyper, Some(&subclassdef))?;
        let vtable_ptr = r_subclass.getruntime(&self.lowleveltype)?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(vtable_ptr)),
            self.lowleveltype.clone(),
        ))
    }

    /// RPython `ClassRepr.convert_const(self, value)` (rclass.py:222-226),
    /// inherited by `RootClassRepr`.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let ConstValue::HostObject(host) = value else {
            return Err(TyperError::message(format!(
                "RootClassRepr.convert_const: not a class: {value:?}"
            )));
        };
        if !host.is_class() {
            return Err(TyperError::message(format!(
                "RootClassRepr.convert_const: not a class: {host:?}"
            )));
        }
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("RootClassRepr.convert_const: rtyper weak ref expired")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("RootClassRepr.convert_const: annotator weak ref dropped")
        })?;
        let desc = annotator
            .bookkeeper
            .getdesc(host)
            .map_err(|e| TyperError::message(e.to_string()))?;
        self.convert_desc(&desc)
    }
}

// ---------------------------------------------------------------------
// rclass.py:467-558 — InstanceRepr (classdef=None + classdef!=None
// scaffolding branches).
// ---------------------------------------------------------------------

/// RPython `class InstanceRepr(Repr)` (rclass.py:467-558).
///
/// R2-C brings in the attrs iteration for `readonly = False` instance
/// attributes: `fields` / `allinstancefields` populate via
/// `rtyper.getrepr(attrdef.s_value)` + `mangled_name = 'inst_' + name`
/// (rclass.py:503-509), and the `object_type` Struct body includes the
/// `(mangled_name, r.lowleveltype)` pairs alongside `('super',
/// rbase.object_type)`. The `special_memory_pressure` /
/// `mutate_<name>` quasi-immutable fields (rclass.py:534-546), the
/// `_check_for_immutable_hints` branch (rclass.py:560-581), and the
/// `rtti=True` kwarg (rclass.py:531-532 `attachRuntimeTypeInfo` side
/// effect) remain R2-D / R3 scope.
#[derive(Debug)]
pub struct InstanceRepr {
    /// RPython `self.rtyper = rtyper` (rclass.py:469).
    rtyper: Weak<RPythonTyper>,
    /// RPython `self.classdef` (rclass.py:470). `None` mirrors
    /// upstream's sentinel for the root instance that carries
    /// `object_type = OBJECT_BY_FLAVOR[...]` directly.
    classdef: Option<Rc<RefCell<ClassDef>>>,
    /// RPython `self.object_type = OBJECT_BY_FLAVOR[LLFLAVOR[gcflavor]]`
    /// when classdef is None (rclass.py:472), otherwise a fresh
    /// `ForwardReference` (rclass.py:474-475). The FR target resolves
    /// during `_setup_repr` once the `MkStruct('name', ('super',
    /// rbase.object_type), ...)` body is built (rclass.py:548-554).
    object_type: LowLevelType,
    /// RPython `self.lowleveltype = Ptr(self.object_type)` (rclass.py:477).
    lowleveltype: LowLevelType,
    /// RPython `self.gcflavor` (rclass.py:478).
    gcflavor: Flavor,
    /// RPython `self.rclass = getclassrepr(...)` (rclass.py:494).
    rclass: RefCell<Option<ClassReprArc>>,
    /// RPython `self.fields = fields` (rclass.py:557). Instance-level
    /// attributes keyed by unmangled attr name; populated by
    /// `_setup_repr`.
    fields: RefCell<HashMap<String, (String, Arc<dyn Repr>)>>,
    /// RPython `self.rbase = getinstancerepr(...)` (rclass.py:517).
    rbase: RefCell<Option<Arc<InstanceRepr>>>,
    /// RPython `self.allinstancefields = allinstancefields`
    /// (rclass.py:558). Accumulates rbase.allinstancefields ∪ fields.
    allinstancefields: RefCell<HashMap<String, (String, Arc<dyn Repr>)>>,
    /// RPython `self.immutable_field_set = set()` (rclass.py:493) for
    /// classdef=None, overwritten by `_check_for_immutable_hints` for
    /// classdef!=None (rclass.py:576). R2-C keeps it as an empty set —
    /// the immutable-hint derivation lands in R2-D.
    #[allow(dead_code)]
    immutable_field_set: RefCell<HashSet<String>>,
    /// RPython `self._reusable_prebuilt_instance` (rclass.py:807).
    /// Lazy-initialised by [`Self::get_reusable_prebuilt_instance`] —
    /// stays `None` until the first call.
    reusable_prebuilt_instance: RefCell<Option<_ptr>>,
    /// RPython `self.iprebuiltinstances = identity_dict()` (rclass.py:482).
    /// Value-keyed identity cache populated by
    /// [`Self::convert_const_exact`]. The HostObject is the upstream
    /// `value` argument; `Hash`/`Eq` use Arc identity matching
    /// upstream's `identity_dict`. Stays empty until the first
    /// exact-match `convert_const` reaches the cache.
    iprebuiltinstances: RefCell<HashMap<HostObject, _ptr>>,
    /// RPython `Repr._initialized` state machine.
    state: ReprState,
}

impl InstanceRepr {
    /// RPython `InstanceRepr.__init__(self, rtyper, classdef,
    /// gcflavor='gc')` (rclass.py:468-478).
    ///
    /// Covers both branches: `classdef is None` (rclass.py:471-472) and
    /// `classdef is not None` (rclass.py:473-475). In the latter case a
    /// fresh `ForwardReference` / `GcForwardReference` is allocated with
    /// its `Arc<Mutex<_>>` target shared into the `lowleveltype` Ptr, so
    /// [`InstanceRepr::_setup_repr`] can resolve it once.
    pub fn new(
        rtyper: &Rc<RPythonTyper>,
        classdef: Option<&Rc<RefCell<ClassDef>>>,
        gcflavor: Flavor,
    ) -> Self {
        let (object_type, lowleveltype) = build_instance_types(classdef, gcflavor);
        InstanceRepr {
            rtyper: Rc::downgrade(rtyper),
            classdef: classdef.cloned(),
            object_type,
            lowleveltype,
            gcflavor,
            rclass: RefCell::new(None),
            fields: RefCell::new(HashMap::new()),
            rbase: RefCell::new(None),
            allinstancefields: RefCell::new(HashMap::new()),
            immutable_field_set: RefCell::new(HashSet::new()),
            reusable_prebuilt_instance: RefCell::new(None),
            iprebuiltinstances: RefCell::new(HashMap::new()),
            state: ReprState::new(),
        }
    }

    /// Convenience constructor for the classdef=None root instance
    /// (rclass.py:471-472). Keeps the live `rtyper` backref so
    /// `_setup_repr` can populate the root `__class__` field exactly
    /// like upstream.
    pub fn new_rootinstance(rtyper: &Rc<RPythonTyper>, gcflavor: Flavor) -> Self {
        Self::new(rtyper, None, gcflavor)
    }

    /// RPython `InstanceRepr.classdef` (rclass.py:470). Returns
    /// `None` for root instances, `Some(Rc<RefCell<ClassDef>>)` otherwise.
    pub fn classdef(&self) -> Option<Rc<RefCell<ClassDef>>> {
        self.classdef.clone()
    }

    /// RPython `InstanceRepr.object_type` (rclass.py:472,475).
    pub fn object_type(&self) -> &LowLevelType {
        &self.object_type
    }

    /// RPython `InstanceRepr.gcflavor` (rclass.py:478).
    pub fn gcflavor(&self) -> Flavor {
        self.gcflavor
    }

    /// Read-only view of the instance-level `fields` dict populated by
    /// `_setup_repr`. RPython `self.fields` (rclass.py:557).
    pub fn fields(&self) -> std::cell::Ref<'_, HashMap<String, (String, Arc<dyn Repr>)>> {
        self.fields.borrow()
    }

    /// Read-only view of the `allinstancefields` dict. RPython
    /// `self.allinstancefields` (rclass.py:558).
    pub fn allinstancefields(
        &self,
    ) -> std::cell::Ref<'_, HashMap<String, (String, Arc<dyn Repr>)>> {
        self.allinstancefields.borrow()
    }

    /// RPython `InstanceRepr.rclass` (rclass.py:494) — the
    /// [`ClassRepr`] / [`RootClassRepr`] handle for this instance's
    /// class. Populated lazily by [`InstanceRepr::_setup_repr`]; callers
    /// must `Repr::setup` first or be invoked from a context where
    /// setup has run.
    pub fn rclass(&self) -> Option<ClassReprArc> {
        self.rclass.borrow().clone()
    }

    /// RPython `InstanceRepr.getfieldrepr(self, attr)` (rclass.py:977-985):
    ///
    /// ```python
    /// def getfieldrepr(self, attr):
    ///     """Return the repr used for the given attribute."""
    ///     if attr in self.fields:
    ///         mangled_name, r = self.fields[attr]
    ///         return r
    ///     else:
    ///         if self.classdef is None:
    ///             raise MissingRTypeAttribute(attr)
    ///         return self.rbase.getfieldrepr(attr)
    /// ```
    pub fn getfieldrepr(self: &Arc<Self>, attr: &str) -> Result<Arc<dyn Repr>, TyperError> {
        if let Some((_mangled, r)) = self.fields.borrow().get(attr).cloned() {
            return Ok(r);
        }
        // upstream: `if self.classdef is None: raise MissingRTypeAttribute(attr)`.
        if self.classdef.is_none() {
            return Err(TyperError::message(format!(
                "InstanceRepr.getfieldrepr: MissingRTypeAttribute({attr:?})"
            )));
        }
        let rbase = self.rbase.borrow().clone().ok_or_else(|| {
            TyperError::message("InstanceRepr.getfieldrepr: rbase missing — call setup() first")
        })?;
        rbase.getfieldrepr(attr)
    }

    /// RPython `InstanceRepr.hook_access_field(self, vinst, cname,
    /// llops, flags)` (rclass.py:712-713):
    ///
    /// ```python
    /// def hook_access_field(self, vinst, cname, llops, flags):
    ///     pass # for virtualizables; see rvirtualizable.py
    /// ```
    ///
    /// Default no-op. Subclasses (e.g. virtualizable instance reprs)
    /// override to emit `promote_virtualizable` op before the getfield.
    pub fn hook_access_field(
        &self,
        _vinst: &Hlvalue,
        _cname: &Hlvalue,
        _llops: &mut LowLevelOpList,
        _flags: &Flags,
    ) {
        // upstream: `pass`.
    }

    /// RPython `InstanceRepr.getfield(self, vinst, attr, llops,
    /// force_cast=False, flags={})` (rclass.py:987-1000):
    ///
    /// ```python
    /// def getfield(self, vinst, attr, llops, force_cast=False, flags={}):
    ///     """Read the given attribute (or __class__ for the type) of 'vinst'."""
    ///     if attr in self.fields:
    ///         mangled_name, r = self.fields[attr]
    ///         cname = inputconst(Void, mangled_name)
    ///         if force_cast:
    ///             vinst = llops.genop('cast_pointer', [vinst], resulttype=self)
    ///         self.hook_access_field(vinst, cname, llops, flags)
    ///         return llops.genop('getfield', [vinst, cname], resulttype=r)
    ///     else:
    ///         if self.classdef is None:
    ///             raise MissingRTypeAttribute(attr)
    ///         return self.rbase.getfield(vinst, attr, llops, force_cast=True,
    ///                                    flags=flags)
    /// ```
    pub fn getfield(
        self: &Arc<Self>,
        vinst: Hlvalue,
        attr: &str,
        llops: &mut LowLevelOpList,
        force_cast: bool,
        flags: &Flags,
    ) -> Result<Variable, TyperError> {
        if let Some((mangled_name, r)) = self.fields.borrow().get(attr).cloned() {
            // upstream: `cname = inputconst(Void, mangled_name)`.
            let cname = Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::byte_str(mangled_name.clone()),
                LowLevelType::Void,
            ));
            // upstream: `if force_cast: vinst = llops.genop('cast_pointer',
            //                                              [vinst], resulttype=self)`.
            let vinst = if force_cast {
                let v_cast = llops
                    .genop(
                        "cast_pointer",
                        vec![vinst],
                        GenopResult::LLType(self.lowleveltype.clone()),
                    )
                    .expect("cast_pointer with non-Void resulttype yields a Variable");
                Hlvalue::Variable(v_cast)
            } else {
                vinst
            };
            // upstream: `self.hook_access_field(vinst, cname, llops, flags)`.
            self.hook_access_field(&vinst, &cname, llops, flags);
            // upstream: `return llops.genop('getfield', [vinst, cname],
            //                              resulttype=r)`.
            return Ok(llops
                .genop(
                    "getfield",
                    vec![vinst, cname],
                    GenopResult::LLType(r.lowleveltype().clone()),
                )
                .expect("getfield with non-Void resulttype yields a Variable"));
        }
        // upstream: `if self.classdef is None: raise MissingRTypeAttribute(attr)`.
        if self.classdef.is_none() {
            return Err(TyperError::message(format!(
                "InstanceRepr.getfield: MissingRTypeAttribute({attr:?})"
            )));
        }
        let rbase = self.rbase.borrow().clone().ok_or_else(|| {
            TyperError::message("InstanceRepr.getfield: rbase missing — call setup() first")
        })?;
        // upstream: `return self.rbase.getfield(vinst, attr, llops,
        //                                      force_cast=True, flags=flags)`.
        rbase.getfield(vinst, attr, llops, true, flags)
    }

    /// RPython `InstanceRepr.null_instance(self)` (rclass.py:938-939):
    ///
    /// ```python
    /// def null_instance(self):
    ///     return nullptr(self.object_type)
    /// ```
    ///
    /// Returns a null `_ptr` to the same container as `self.lowleveltype`,
    /// suitable for embedding into `Constant(value=LLPtr(...),
    /// concretetype=self.lowleveltype)` (the upstream `convert_const`
    /// `value is None` branch).
    pub fn null_instance(&self) -> Result<_ptr, TyperError> {
        let object_lltype = match self.object_type.clone() {
            LowLevelType::ForwardReference(fwd) => fwd.resolved().ok_or_else(|| {
                TyperError::message(
                    "InstanceRepr.null_instance: object_type ForwardReference not \
                     resolved (call setup() first)",
                )
            })?,
            other => other,
        };
        lltype::nullptr(object_lltype).map_err(TyperError::message)
    }

    /// RPython `InstanceRepr.upcast(self, result)` (rclass.py:941-942):
    ///
    /// ```python
    /// def upcast(self, result):
    ///     return cast_pointer(self.lowleveltype, result)
    /// ```
    ///
    /// Used by `convert_const` when delegating to a subclass `InstanceRepr`
    /// (rclass.py:788-790): the subclass produces a pointer typed at the
    /// subclass's `lowleveltype`; this helper casts it back up to the
    /// owning `InstanceRepr`'s `lowleveltype`.
    pub fn upcast(&self, result: &_ptr) -> Result<_ptr, TyperError> {
        let LowLevelType::Ptr(ptrtype) = &self.lowleveltype else {
            return Err(TyperError::message(format!(
                "InstanceRepr.upcast: self.lowleveltype is not a Ptr: {:?}",
                self.lowleveltype
            )));
        };
        lltype::cast_pointer(ptrtype, result).map_err(TyperError::message)
    }

    /// RPython `InstanceRepr.create_instance(self)` (rclass.py:944-945):
    ///
    /// ```python
    /// def create_instance(self):
    ///     return malloc(self.object_type, flavor=self.gcflavor, immortal=True)
    /// ```
    ///
    /// Allocates a fresh instance container in the appropriate gc flavor
    /// with `immortal=True` so it survives the prebuilt-data lifetime.
    /// The `object_type` ForwardReference (for non-root) must already be
    /// resolved by `_setup_repr`; the caller is responsible for
    /// `Repr::setup` first.
    pub fn create_instance(&self) -> Result<_ptr, TyperError> {
        let object_lltype = match self.object_type.clone() {
            LowLevelType::ForwardReference(fwd) => fwd.resolved().ok_or_else(|| {
                TyperError::message(
                    "InstanceRepr.create_instance: object_type ForwardReference not \
                     resolved (call setup() first)",
                )
            })?,
            other => other,
        };
        let flavor = match self.gcflavor {
            Flavor::Gc => lltype::MallocFlavor::Gc,
            Flavor::Raw => lltype::MallocFlavor::Raw,
        };
        lltype::malloc(object_lltype, None, flavor, true).map_err(TyperError::message)
    }

    /// RPython `InstanceRepr.initialize_prebuilt_data(self, value,
    /// classdef, result)` (rclass.py:947-975).
    ///
    /// Recursive helper that walks the `rbase` chain (each level
    /// addressed via the per-step `path` of `"super"` segments,
    /// matching the `setattr_at_path` semantics introduced for vtable
    /// init) and writes each level's instance fields. Reaching the
    /// root (`classdef is None`) writes the `typeptr` slot to point at
    /// the leaf-class's vtable via `getclassrepr(rtyper, classdef)`.
    ///
    /// Pyre-port adaptations:
    /// - `value` is the live HostObject (or `None` for the upstream
    ///   `Ellipsis` sentinel meaning "use defaults"). The Ellipsis
    ///   path skips the live `instance_get` probe and falls through
    ///   straight to `read_attribute` → `_defl`.
    /// - Upstream's `try: getattr(value, name) except AttributeError:`
    ///   is split into an explicit `host.instance_get(name)` lookup
    ///   on the per-instance `__dict__` followed by the same
    ///   `read_attribute(name, None)` → `_defl` cascade upstream uses
    ///   inside the except branch — see the body block at
    ///   `for (name, ...) in &fields_snapshot` below for the full
    ///   try/except mirror.
    pub fn initialize_prebuilt_data(
        &self,
        _value: Option<&HostObject>,
        classdef: Option<&Rc<RefCell<ClassDef>>>,
        result: &mut _ptr,
        path: &[&str],
    ) -> Result<(), TyperError> {
        if let Some(_self_classdef) = self.classdef.as_ref() {
            // upstream rclass.py:949 — recurse into rbase first.
            let rbase = self.rbase.borrow().clone().ok_or_else(|| {
                TyperError::message(
                    "InstanceRepr.initialize_prebuilt_data: rbase missing — \
                     call setup() first",
                )
            })?;
            let mut next_path = Vec::with_capacity(path.len() + 1);
            next_path.extend_from_slice(path);
            next_path.push("super");
            rbase.initialize_prebuilt_data(_value, classdef, result, &next_path)?;
            // upstream rclass.py:951-971 — per-level fields:
            //   for name, (mangled_name, r) in self.fields.items():
            //       if r.lowleveltype is Void:
            //           llattrvalue = None
            //       else:
            //           try:
            //               attrvalue = getattr(value, name)
            //           except AttributeError:
            //               attrvalue = self.classdef.classdesc.read_attribute(
            //                   name, None)
            //               if attrvalue is None:
            //                   llattrvalue = r.lowleveltype._defl()
            //               else:
            //                   llattrvalue = r.convert_desc_or_const(attrvalue)
            //           else:
            //               llattrvalue = r.convert_const(attrvalue)
            //       setattr(result, mangled_name, llattrvalue)
            //
            // Pyre splits the upstream try/except on `getattr` into an
            // explicit `instance_get` (live `__dict__`) → fall through
            // to `read_attribute` (class-level dict) → fall through to
            // `_defl` cascade. The Ellipsis-sentinel path
            // (`get_reusable_prebuilt_instance` with `_value = None`)
            // skips the `instance_get` lookup entirely and goes
            // straight to `read_attribute` / `_defl`.
            let fields_snapshot: Vec<(String, (String, Arc<dyn Repr>))> = self
                .fields
                .borrow()
                .iter()
                .map(|(name, (mangled, r))| (name.clone(), (mangled.clone(), r.clone())))
                .collect();
            if !fields_snapshot.is_empty() {
                let classdef_b = _self_classdef.borrow();
                let classdesc = classdef_b.classdesc.clone();
                drop(classdef_b);
                for (name, (mangled_name, r)) in &fields_snapshot {
                    let llattrvalue = if matches!(r.lowleveltype(), LowLevelType::Void) {
                        // upstream `if r.lowleveltype is Void: llattrvalue = None` —
                        // pyre's `_ptr.setattr` with a Void-tagged value
                        // is modeled as `LowLevelValue::Void`.
                        lltype::LowLevelValue::Void
                    } else {
                        // upstream try/except: `getattr(value, name)` ⇒
                        // pyre's `host.instance_get(name)`. AttributeError
                        // ⇒ `read_attribute(name, None)` ⇒ `_defl`.
                        let probe = _value.and_then(|host| host.instance_get(name));
                        if let Some(attrvalue) = probe {
                            let const_for_field =
                                (r.as_ref() as &dyn Repr).convert_const(&attrvalue)?;
                            constant_to_lowlevel_value(&const_for_field)?
                        } else {
                            // upstream rclass.py:959-968:
                            //     attrvalue = self.classdef.classdesc.read_attribute(
                            //         name, None)
                            //     if attrvalue is None:
                            //         llattrvalue = r.lowleveltype._defl()
                            //     else:
                            //         llattrvalue = r.convert_desc_or_const(attrvalue)
                            //
                            // `ClassDictEntry::Constant(c)` ↔ upstream
                            // `Constant(value)` (classdesc.py:601,634), and
                            // `ClassDictEntry::Desc(d)` ↔ upstream
                            // `funcdesc` stored verbatim (classdesc.py:612).
                            // Both arms project cleanly into `DescOrConst`,
                            // matching upstream `convert_desc_or_const`
                            // (rmodel.py:111-118).
                            match super::super::super::annotator::classdesc::ClassDesc::read_attribute(
                                &classdesc, name,
                            ) {
                                Some(class_entry) => {
                                    let desc_or_const = match class_entry {
                                        crate::annotator::classdesc::ClassDictEntry::Constant(c) => {
                                            super::rmodel::DescOrConst::Const(c)
                                        }
                                        crate::annotator::classdesc::ClassDictEntry::Desc(d) => {
                                            super::rmodel::DescOrConst::Desc(d)
                                        }
                                    };
                                    let const_for_field =
                                        (r.as_ref() as &dyn Repr).convert_desc_or_const(&desc_or_const)?;
                                    constant_to_lowlevel_value(&const_for_field)?
                                }
                                None => r.lowleveltype()._defl(),
                            }
                        }
                    };
                    if path.is_empty() {
                        result
                            .setattr(mangled_name, llattrvalue)
                            .map_err(TyperError::message)?;
                    } else {
                        result
                            .setattr_at_path(path, mangled_name, llattrvalue)
                            .map_err(TyperError::message)?;
                    }
                }
            }
        } else {
            // upstream rclass.py:973-975 — root sets typeptr.
            // `result.typeptr = rclass.getvtable()`.
            let rtyper = self.rtyper.upgrade().ok_or_else(|| {
                TyperError::message(
                    "InstanceRepr.initialize_prebuilt_data: rtyper weak ref expired",
                )
            })?;
            let rclass = getclassrepr_arc(&rtyper, classdef)?;
            // The vtable is already cast to typeptr (CLASSTYPE) by
            // `cast_vtable_to_typeptr` inside `getvtable`.
            let vtable_ptr = match &rclass {
                ClassReprArc::Inst(inst) => inst.getvtable()?,
                ClassReprArc::Root(root) => root.getvtable()?,
            };
            let val = lltype::LowLevelValue::Ptr(Box::new(vtable_ptr));
            if path.is_empty() {
                result
                    .setattr("typeptr", val)
                    .map_err(TyperError::message)?;
            } else {
                result
                    .setattr_at_path(path, "typeptr", val)
                    .map_err(TyperError::message)?;
            }
        }
        Ok(())
    }

    /// RPython `InstanceRepr.convert_const_exact(self, value)`
    /// (rclass.py:794-802):
    ///
    /// ```python
    /// def convert_const_exact(self, value):
    ///     try:
    ///         return self.iprebuiltinstances[value]
    ///     except KeyError:
    ///         self.setup()
    ///         result = self.create_instance()
    ///         self.iprebuiltinstances[value] = result
    ///         self.initialize_prebuilt_instance(value, self.classdef, result)
    ///         return result
    /// ```
    ///
    /// `iprebuiltinstances` is the value-keyed identity cache
    /// (rclass.py:482); upstream uses `identity_dict()` which is
    /// pointer-keyed dict semantics — pyre keys on
    /// [`HostObject`]'s Arc identity (Hash + Eq via `Arc::ptr_eq`).
    /// `initialize_prebuilt_instance` is a thin wrapper around
    /// `initialize_prebuilt_data` (the recursion shield via
    /// `_initialize_data_flattenrec` is folded into pyre's recursive
    /// call structure since the `flattenrec` mechanism is upstream-only
    /// and operates as a no-op for non-recursive
    /// `initialize_prebuilt_data` graphs).
    pub fn convert_const_exact(
        self: &Arc<Self>,
        host_obj: &HostObject,
    ) -> Result<Constant, TyperError> {
        // upstream `try: return self.iprebuiltinstances[value]`.
        if let Some(cached) = self.iprebuiltinstances.borrow().get(host_obj).cloned() {
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(cached)),
                self.lowleveltype.clone(),
            ));
        }
        // upstream `self.setup()`. Repr::setup is idempotent.
        Repr::setup(self.as_ref() as &dyn Repr)?;
        // upstream rclass.py:799-800:
        //     result = self.create_instance()
        //     self.iprebuiltinstances[value] = result
        //     self.initialize_prebuilt_data(value, self.classdef, result)
        //
        // RPython aliases `result` between the cache slot and the
        // init target via Python object identity, so mutations made
        // by `initialize_prebuilt_data` are visible through
        // `iprebuiltinstances[value]` for recursive
        // `convert_const(value)` calls during init (`a.b.back == a`
        // intra-class circular prebuilt instances).
        //
        // Pyre clones `result` into the cache; both the cached clone
        // and the init target (`local`, the moved original) share
        // `_struct._fields` / `_array.items` via `Arc<Mutex<...>>`
        // (lltype.rs:711-748, task #157), preserving the same
        // mid-init aliasing semantics. The cache `borrow_mut` is
        // dropped before init runs so a recursive `convert_const_exact`
        // cache probe at the top of this function can re-acquire
        // `borrow()` without panicking on overlapping borrows.
        let initial = self.create_instance()?;
        self.iprebuiltinstances
            .borrow_mut()
            .insert(host_obj.clone(), initial.clone());
        let mut local = initial;
        self.initialize_prebuilt_data(Some(host_obj), self.classdef.as_ref(), &mut local, &[])?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(local)),
            self.lowleveltype.clone(),
        ))
    }

    /// RPython `InstanceRepr.get_reusable_prebuilt_instance(self)`
    /// (rclass.py:804-813):
    ///
    /// ```python
    /// def get_reusable_prebuilt_instance(self):
    ///     try:
    ///         return self._reusable_prebuilt_instance
    ///     except AttributeError:
    ///         self.setup()
    ///         result = self.create_instance()
    ///         self._reusable_prebuilt_instance = result
    ///         self.initialize_prebuilt_data(Ellipsis, self.classdef, result)
    ///         return result
    /// ```
    ///
    /// Lazy-cached `_ptr` to a single dummy instance, used by
    /// `ExceptionData.get_standard_ll_exc_instance` to materialise the
    /// canonical instance for a given exception class. Multiple calls
    /// return the same `_ptr` value (same `_identity`).
    pub fn get_reusable_prebuilt_instance(self: &Arc<Self>) -> Result<_ptr, TyperError> {
        if let Some(cached) = self.reusable_prebuilt_instance.borrow().clone() {
            return Ok(cached);
        }
        // upstream: `self.setup()`. Repr::setup is idempotent.
        Repr::setup(self.as_ref() as &dyn Repr)?;
        // upstream rclass.py:810-813:
        //     result = self.create_instance()
        //     self._reusable_prebuilt_instance = result
        //     self.initialize_prebuilt_data(Ellipsis, self.classdef, result)
        //     return result
        //
        // Pyre clones `result` into the cache; the cached clone and the
        // init target (`local`, the moved original) share
        // `_struct._fields` / `_array.items` via `Arc<Mutex<...>>`
        // (lltype.rs:711-748, task #157), so mutations applied by
        // `initialize_prebuilt_data` are visible through
        // `_reusable_prebuilt_instance` for the no-recursion case
        // (`get_standard_ll_exc_instance`). The cache `borrow_mut` is
        // dropped before init runs.
        let result = self.create_instance()?;
        *self.reusable_prebuilt_instance.borrow_mut() = Some(result.clone());
        let mut local = result;
        self.initialize_prebuilt_data(None, self.classdef.as_ref(), &mut local, &[])?;
        Ok(local)
    }
}

/// Build the `(object_type, lowleveltype)` pair for an InstanceRepr
/// per rclass.py:471-477.
fn build_instance_types(
    classdef: Option<&Rc<RefCell<ClassDef>>>,
    gcflavor: Flavor,
) -> (LowLevelType, LowLevelType) {
    if classdef.is_some() {
        // rclass.py:474-475 — `ForwardRef = lltype.FORWARDREF_BY_FLAVOR[
        // LLFLAVOR[gcflavor]]; self.object_type = ForwardRef()`.
        let fwd = match gcflavor {
            Flavor::Gc => ForwardReference::gc(),
            Flavor::Raw => ForwardReference::new(),
        };
        let fwd_for_ptr = fwd.clone();
        let object_type = LowLevelType::ForwardReference(Box::new(fwd));
        let lowleveltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::ForwardReference(fwd_for_ptr),
        }));
        (object_type, lowleveltype)
    } else {
        // rclass.py:471-472 — classdef is None path.
        let object_type = object_by_flavor(gcflavor);
        let lowleveltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: match &object_type {
                LowLevelType::Struct(body) => PtrTarget::Struct((**body).clone()),
                other => panic!("object_by_flavor returned non-Struct {other:?}"),
            },
        }));
        (object_type, lowleveltype)
    }
}

impl Repr for InstanceRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "InstanceRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::Repr
    }

    /// RPython `InstanceRepr.rtype_getattr(self, hop)` (rclass.py:838-857):
    ///
    /// ```python
    /// def rtype_getattr(self, hop):
    ///     if hop.s_result.is_constant():
    ///         return hop.inputconst(hop.r_result, hop.s_result.const)
    ///     attr = hop.args_s[1].const
    ///     vinst, vattr = hop.inputargs(self, Void)
    ///     if attr == '__class__' and hop.r_result.lowleveltype is Void:
    ///         # special case for when the result of '.__class__' is a constant
    ///         [desc] = hop.s_result.descriptions
    ///         return hop.inputconst(Void, desc.pyobj)
    ///     if attr in self.allinstancefields:
    ///         return self.getfield(vinst, attr, hop.llops,
    ///                              flags=hop.args_s[0].flags)
    ///     elif attr in self.rclass.allmethods:
    ///         # special case for methods: represented as their 'self' only
    ///         # (see MethodsPBCRepr)
    ///         return hop.r_result.get_method_from_instance(self, vinst,
    ///                                                      hop.llops)
    ///     else:
    ///         vcls = self.getfield(vinst, '__class__', hop.llops)
    ///         return self.rclass.getclsfield(vcls, attr, hop.llops)
    /// ```
    ///
    /// Step 6 (`allmethods` method dispatch) routes through
    /// `MethodsPBCRepr.get_method_from_instance` (rpbc.py:1173-1176)
    /// which is also unported.  For `classdef=None` (the only
    /// `InstanceRepr` flavour reaching the rtyper today, per
    /// `valuetype_to_someshell(Ref) -> SomeInstance(classdef=None)` at
    /// `flowspace_adapter.rs:1638-1659`), the Step 6 branch is
    /// unreachable: `RootClassRepr._setup_repr` (rclass.rs:1902) keeps
    /// `allmethods = {}`.
    fn rtype_getattr(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::annotator::model::{SomeObjectTrait, SomeValue};
        use crate::flowspace::model::Hlvalue;
        use crate::translator::rtyper::rtyper::ConvertedTo;

        // upstream rclass.py:839-840: `if hop.s_result.is_constant():
        //     return hop.inputconst(hop.r_result, hop.s_result.const)`.
        let s_result_clone = hop.s_result.borrow().clone();
        if let Some(s_result) = &s_result_clone {
            if s_result.is_constant() {
                let r_result = hop.r_result.borrow().clone().ok_or_else(|| {
                    TyperError::message("InstanceRepr.rtype_getattr: r_result missing")
                })?;
                let const_val = s_result
                    .const_()
                    .cloned()
                    .expect("s_result.is_constant() implies const_() is Some");
                return HighLevelOp::inputconst(ConvertedTo::Repr(r_result.as_ref()), &const_val)
                    .map(Hlvalue::Constant)
                    .map(Some);
            }
        }

        // upstream: `attr = hop.args_s[1].const` (rclass.py:841).
        let attr = {
            let args_s = hop.args_s.borrow();
            let s_attr = args_s.get(1).ok_or_else(|| {
                TyperError::message("InstanceRepr.rtype_getattr: hop.args_s[1] missing")
            })?;
            s_attr
                .const_()
                .and_then(ConstValue::as_pystr)
                .ok_or_else(|| {
                    TyperError::message(
                        "InstanceRepr.rtype_getattr: attribute name must be a constant string",
                    )
                })?
                .to_string()
        };

        // upstream: `vinst, vattr = hop.inputargs(self, Void)` (rclass.py:842).
        let void = LowLevelType::Void;
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self), ConvertedTo::from(&void)])?;
        let vinst = vlist[0].clone();

        // upstream rclass.py:843-846:
        //     if attr == '__class__' and hop.r_result.lowleveltype is Void:
        //         # special case for when the result of '.__class__' is a constant
        //         [desc] = hop.s_result.descriptions
        //         return hop.inputconst(Void, desc.pyobj)
        let r_result_for_classattr =
            hop.r_result.borrow().clone().ok_or_else(|| {
                TyperError::message("InstanceRepr.rtype_getattr: r_result missing")
            })?;
        if attr == "__class__"
            && matches!(r_result_for_classattr.lowleveltype(), &LowLevelType::Void)
        {
            let pbc = match s_result_clone.as_ref() {
                Some(SomeValue::PBC(pbc)) => pbc.clone(),
                _ => {
                    return Err(TyperError::message(
                        "InstanceRepr.rtype_getattr: __class__ Void-typed result expects \
                         SomePBC s_result (rclass.py:843-846)",
                    ));
                }
            };
            if pbc.descriptions.len() != 1 {
                return Err(TyperError::message(
                    "InstanceRepr.rtype_getattr: __class__ Void-typed result expects a \
                     singleton SomePBC.descriptions (rclass.py:845)",
                ));
            }
            let entry = pbc.descriptions.values().next().unwrap();
            let pyobj = entry.pyobj().ok_or_else(|| {
                TyperError::message(
                    "InstanceRepr.rtype_getattr: __class__ description missing pyobj \
                     (rclass.py:846)",
                )
            })?;
            return HighLevelOp::inputconst(
                ConvertedTo::from(&LowLevelType::Void),
                &ConstValue::HostObject(pyobj),
            )
            .map(Hlvalue::Constant)
            .map(Some);
        }

        // Recover `Arc<Self>` from the rtyper repr cache so we can call
        // the `&Arc<Self>`-receiver helpers (`getfield`).  Same pattern
        // as `convert_const` (rclass.rs:2666) which routes back through
        // `getinstancerepr`.
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("InstanceRepr.rtype_getattr: rtyper weak ref expired")
        })?;
        let self_arc = getinstancerepr(&rtyper, self.classdef.as_ref(), self.gcflavor)?;

        // upstream: `if attr in self.allinstancefields:`
        //          `return self.getfield(vinst, attr, hop.llops,
        //                                flags=hop.args_s[0].flags)`
        // (rclass.py:847-849).
        let has_field = self.allinstancefields.borrow().contains_key(&attr);
        if has_field {
            let flags = {
                let args_s = hop.args_s.borrow();
                let s_inst = args_s.get(0).ok_or_else(|| {
                    TyperError::message("InstanceRepr.rtype_getattr: hop.args_s[0] missing")
                })?;
                match s_inst {
                    SomeValue::Instance(inst) => inst
                        .flags
                        .iter()
                        .map(|(k, v)| (k.clone(), ConstValue::Bool(*v)))
                        .collect::<Flags>(),
                    _ => Flags::default(),
                }
            };
            let mut llops = hop.llops.borrow_mut();
            let var = self_arc.getfield(vinst, &attr, &mut llops, false, &flags)?;
            return Ok(Some(Hlvalue::Variable(var)));
        }

        // upstream: `elif attr in self.rclass.allmethods:`
        //          `return hop.r_result.get_method_from_instance(self, vinst, hop.llops)`
        // (rclass.py:850-854).
        let rclass = self.rclass.borrow().clone().ok_or_else(|| {
            TyperError::message("InstanceRepr.rtype_getattr: rclass missing — call setup() first")
        })?;
        let has_method = match &rclass {
            ClassReprArc::Inst(c) => c.allmethods().contains_key(&attr),
            // `RootClassRepr.allmethods` is always empty (rclass.rs:1902).
            ClassReprArc::Root(_) => false,
        };
        if has_method {
            // upstream rclass.py:850-854:
            //     elif attr in self.rclass.allmethods:
            //         # special case for methods: represented as their
            //         # 'self' only (see MethodsPBCRepr)
            //         return hop.r_result.get_method_from_instance(
            //             self, vinst, hop.llops)
            //
            // `hop.r_result` upstream is the `MethodsPBCRepr` for the
            // dispatched method.  Pyre stores it as `Arc<dyn Repr>`;
            // downcast through `Repr: Any` (rmodel.rs:332) to reach the
            // concrete `get_method_from_instance` (rpbc.rs:6363+).
            use crate::translator::rtyper::rpbc::MethodsPBCRepr;
            let r_result_arc = hop.r_result.borrow().clone().ok_or_else(|| {
                TyperError::message("InstanceRepr.rtype_getattr: r_result missing on method branch")
            })?;
            let methods_repr = (r_result_arc.as_ref() as &dyn std::any::Any)
                .downcast_ref::<MethodsPBCRepr>()
                .ok_or_else(|| {
                    TyperError::message(format!(
                        "InstanceRepr.rtype_getattr: method '{attr}' dispatch expects \
                         MethodsPBCRepr r_result, got {}",
                        r_result_arc.class_name()
                    ))
                })?;
            let mut llops = hop.llops.borrow_mut();
            let v_method = methods_repr.get_method_from_instance(
                self_arc.as_ref() as &dyn Repr,
                vinst.clone(),
                &mut llops,
            )?;
            return Ok(Some(v_method));
        }

        // upstream: `else:`
        //          `vcls = self.getfield(vinst, '__class__', hop.llops)`
        //          `return self.rclass.getclsfield(vcls, attr, hop.llops)`
        // (rclass.py:855-857).
        let mut llops = hop.llops.borrow_mut();
        let flags = Flags::default();
        let v_cls = self_arc.getfield(vinst, "__class__", &mut llops, false, &flags)?;
        let var = match &rclass {
            ClassReprArc::Inst(c) => c.getclsfield(Hlvalue::Variable(v_cls), &attr, &mut llops)?,
            ClassReprArc::Root(r) => r.getclsfield(Hlvalue::Variable(v_cls), &attr, &mut llops)?,
        };
        Ok(Some(Hlvalue::Variable(var)))
    }

    /// RPython `InstanceRepr.convert_const(self, value)` (rclass.py:772-792):
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if value is None:
    ///         return self.null_instance()
    ///     if isinstance(value, types.MethodType):
    ///         value = value.im_self   # bound method -> instance
    ///     bk = self.rtyper.annotator.bookkeeper
    ///     try:
    ///         classdef = bk.getuniqueclassdef(value.__class__)
    ///     except KeyError:
    ///         raise TyperError("no classdef: %r" % (value.__class__,))
    ///     if classdef != self.classdef:
    ///         if classdef.commonbase(self.classdef) != self.classdef:
    ///             raise TyperError("not an instance of %r: %r" % (
    ///                 self.classdef.name, value))
    ///         rinstance = getinstancerepr(self.rtyper, classdef)
    ///         result = rinstance.convert_const(value)
    ///         return self.upcast(result)
    ///     # common case
    ///     return self.convert_const_exact(value)
    /// ```
    ///
    /// The `classdef == self.classdef` exact-match arm dispatches into
    /// [`InstanceRepr::convert_const_exact`] (rclass.py:794-802) via
    /// [`getinstancerepr`], which recovers the owning `Arc<Self>` from
    /// the rtyper's repr cache so the borrow-cell-backed
    /// `iprebuiltinstances` and `initialize_prebuilt_data` (rclass.py:
    /// 947-975, including the class-level `read_attribute` /
    /// `convert_desc_or_const` branch ported as Task #149) can run.
    /// The Variable-arm field-init is in
    /// [`InstanceRepr::initialize_prebuilt_data`].
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        use crate::annotator::classdesc::ClassDef;

        // upstream: `if value is None: return self.null_instance()`.
        if matches!(value, ConstValue::None) {
            let null = self.null_instance()?;
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(null)),
                self.lowleveltype.clone(),
            ));
        }

        // upstream: `isinstance(value, types.MethodType)` —
        //   bound method redirect to its `im_self`.
        let host_obj = match value {
            ConstValue::HostObject(h) => h.clone(),
            other => {
                return Err(TyperError::message(format!(
                    "InstanceRepr.convert_const: expected HostObject or None, \
                     got {other:?}"
                )));
            }
        };
        let host_obj = if host_obj.is_bound_method() {
            host_obj.bound_method_self().cloned().ok_or_else(|| {
                TyperError::message("InstanceRepr.convert_const: bound method has no self_obj")
            })?
        } else {
            host_obj
        };

        // upstream: `classdef = bk.getuniqueclassdef(value.__class__)`.
        let host_cls = host_obj.class_of().ok_or_else(|| {
            TyperError::message(format!(
                "InstanceRepr.convert_const: no class_of() for {:?}",
                host_obj.qualname()
            ))
        })?;
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("InstanceRepr.convert_const: rtyper weak ref dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("InstanceRepr.convert_const: annotator weak ref dropped")
        })?;
        let classdef = annotator
            .bookkeeper
            .getuniqueclassdef(&host_cls)
            .map_err(|e| {
                TyperError::message(format!("no classdef: {} ({})", host_cls.qualname(), e))
            })?;

        // upstream: `if classdef != self.classdef`.
        let exact_match = match self.classdef.as_ref() {
            Some(self_cd) => Rc::ptr_eq(self_cd, &classdef),
            None => false,
        };
        if !exact_match {
            // upstream: `if classdef.commonbase(self.classdef) !=
            //              self.classdef: raise TyperError(...)`.
            //   Subclass-relationship check; self must be a base of the
            //   value's class.
            let self_cd = self.classdef.as_ref().ok_or_else(|| {
                TyperError::message(format!(
                    "InstanceRepr.convert_const: cannot delegate via root \
                     InstanceRepr (classdef=None) to {:?}",
                    classdef.borrow().name
                ))
            })?;
            let common = ClassDef::commonbase(&classdef, self_cd).ok_or_else(|| {
                TyperError::message(format!(
                    "not an instance of {:?}: {:?}",
                    self_cd.borrow().name,
                    host_obj.qualname()
                ))
            })?;
            if !Rc::ptr_eq(&common, self_cd) {
                return Err(TyperError::message(format!(
                    "not an instance of {:?}: {:?}",
                    self_cd.borrow().name,
                    host_obj.qualname()
                )));
            }
            // upstream: `rinstance = getinstancerepr(rtyper, classdef);
            //            result = rinstance.convert_const(value);
            //            return self.upcast(result)`.
            let rinstance = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc)?;
            let sub_result = (rinstance.as_ref() as &dyn Repr).convert_const(value)?;
            let ConstValue::LLPtr(sub_ptr) = &sub_result.value else {
                return Err(TyperError::message(format!(
                    "InstanceRepr.convert_const: subclass result is not LLPtr: \
                     {:?}",
                    sub_result.value
                )));
            };
            let upcast_ptr = self.upcast(sub_ptr)?;
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(upcast_ptr)),
                self.lowleveltype.clone(),
            ));
        }

        // upstream: `return self.convert_const_exact(value)`.
        //
        // The trait method only carries `&self`, but
        // [`convert_const_exact`] needs `&Arc<Self>` for the cache /
        // setup() entry. Look ourselves back up via
        // [`getinstancerepr`] — pyre's repr cache is keyed by
        // `(classdef, flavor)`, which gives the same shared `Arc`
        // back. The lookup is a no-op O(1) HashMap probe (the entry
        // is guaranteed present since `self` exists).
        let arc_self = getinstancerepr(&rtyper, self.classdef.as_ref(), self.gcflavor)?;
        arc_self.convert_const_exact(&host_obj)
    }

    /// RPython `InstanceRepr._setup_repr` (rclass.py:487-558).
    fn _setup_repr(&self) -> Result<(), TyperError> {
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("InstanceRepr._setup_repr: RPythonTyper weak ref expired")
        })?;

        // rclass.py:494 — `self.rclass = getclassrepr(self.rtyper,
        // self.classdef)`.
        let rclass = getclassrepr_arc(&rtyper, self.classdef.as_ref())?;
        *self.rclass.borrow_mut() = Some(rclass);

        // rclass.py:495-496 — `fields = {}; allinstancefields = {}`.
        // Build local dicts first (rclass.py:488-491 "don't store
        // mutable objects on self before they are fully built").
        let mut fields: HashMap<String, (String, Arc<dyn Repr>)> = HashMap::new();
        let mut allinstancefields: HashMap<String, (String, Arc<dyn Repr>)> = HashMap::new();
        let Some(classdef_rc) = self.classdef.as_ref() else {
            fields.insert(
                "__class__".to_string(),
                ("typeptr".to_string(), get_type_repr(rtyper.as_ref())?),
            );
            allinstancefields.extend(fields.clone());
            *self.fields.borrow_mut() = fields;
            *self.allinstancefields.borrow_mut() = allinstancefields;
            return Ok(());
        };
        let mut myllfields: Vec<(String, LowLevelType)> = Vec::new();

        // rclass.py:501-509 — attrs iteration with readonly filter.
        let attrs_sorted = {
            let cd = classdef_rc.borrow();
            cd.attrs
                .iter()
                .map(|(k, v)| (k.clone(), (v.s_value.clone(), v.readonly)))
                .collect::<BTreeMap<String, (SomeValue, bool)>>()
        };
        for (attr_name, (s_value, readonly)) in &attrs_sorted {
            if *readonly {
                continue;
            }
            let r = rtyper.getrepr(s_value)?;
            let mangled_name = format!("inst_{attr_name}");
            fields.insert(attr_name.clone(), (mangled_name.clone(), r.clone()));
            myllfields.push((mangled_name, r.lowleveltype().clone()));
        }
        sort_llfields(&mut myllfields);

        // rclass.py:517-519 — `self.rbase = getinstancerepr(
        // self.rtyper, self.classdef.basedef, self.gcflavor);
        // self.rbase.setup()`.
        let basedef = classdef_rc.borrow().basedef.clone();
        let rbase = getinstancerepr(&rtyper, basedef.as_ref(), self.gcflavor)?;
        Repr::setup(rbase.as_ref())?;
        *self.rbase.borrow_mut() = Some(rbase.clone());

        // rclass.py:548-554 — `object_type = MkStruct(classdef.name,
        // ('super', rbase.object_type), hints=hints, adtmeths=adtmeths,
        // *llfields, **kwds)` + `self.object_type.become(object_type)`.
        // Gc-flavor also passes `rtti=True` (rclass.py:531-532) so the
        // struct carries an `_runtime_type_info` opaque consumable by
        // `fill_vtable_root` via `getRuntimeTypeInfo`. Immutable /
        // special_memory_pressure hints stay unported (R2-D).
        let name = classdef_rc.borrow().name.clone();
        let mut struct_fields = Vec::with_capacity(1 + myllfields.len());
        struct_fields.push(("super".into(), rbase.object_type().clone()));
        struct_fields.extend(myllfields);
        let body = match self.gcflavor {
            Flavor::Gc => StructType::gc_rtti(&name, struct_fields),
            Flavor::Raw => StructType::with_hints(&name, struct_fields, vec![]),
        };
        let LowLevelType::ForwardReference(fwd) = &self.object_type else {
            return Err(TyperError::message(
                "InstanceRepr.object_type must be LowLevelType::ForwardReference \
                 when classdef is not None",
            ));
        };
        fwd.r#become(LowLevelType::Struct(Box::new(body)))
            .map_err(TyperError::message)?;

        // rclass.py:555-558 — `allinstancefields.update(
        // self.rbase.allinstancefields); allinstancefields.update(
        // fields); self.fields = fields;
        // self.allinstancefields = allinstancefields`.
        for (k, v) in rbase.allinstancefields.borrow().iter() {
            allinstancefields.insert(k.clone(), v.clone());
        }
        for (k, v) in fields.iter() {
            allinstancefields.insert(k.clone(), v.clone());
        }
        *self.fields.borrow_mut() = fields;
        *self.allinstancefields.borrow_mut() = allinstancefields;
        Ok(())
    }
}

// ---------------------------------------------------------------------
// rclass.py:67-88, 91-119, 439-440 — module-level accessors.
// ---------------------------------------------------------------------

/// RPython `get_type_repr(rtyper)` (`rclass.py:439-440`).
///
/// ```python
/// def get_type_repr(rtyper):
///     return rtyper.rootclass_repr
/// ```
pub fn get_type_repr(rtyper: &RPythonTyper) -> Result<Arc<dyn Repr>, TyperError> {
    rtyper
        .rootclass_repr
        .borrow()
        .clone()
        .map(|r| r as Arc<dyn Repr>)
        .ok_or_else(|| {
            TyperError::message(
                "rtyper.rootclass_repr is not set — call \
                 RPythonTyper::initialize_exceptiondata() after construction",
            )
        })
}

/// RPython `getclassrepr(rtyper, classdef)` (`rclass.py:67-74`).
///
/// ```python
/// def getclassrepr(rtyper, classdef):
///     if classdef is None:
///         return rtyper.rootclass_repr
///     result = classdef.repr
///     if result is None:
///         result = classdef.repr = ClassRepr(rtyper, classdef)
///         rtyper.add_pendingsetup(result)
///     return result
/// ```
///
/// The `classdef != None` arm caches through `ClassDef.repr`, matching
/// upstream's per-classdef storage.
pub fn getclassrepr(
    rtyper: &Rc<RPythonTyper>,
    classdef: Option<&Rc<RefCell<ClassDef>>>,
) -> Result<Arc<dyn Repr>, TyperError> {
    Ok(getclassrepr_arc(rtyper, classdef)?.as_repr())
}

/// Internal-recursion companion to [`getclassrepr`]: returns the
/// typed [`ClassReprArc`] enum so [`ClassRepr::_setup_repr`] can read
/// `rbase.vtable_type` directly without upcasting through
/// `Arc<dyn Repr>`.
pub(crate) fn getclassrepr_arc(
    rtyper: &Rc<RPythonTyper>,
    classdef: Option<&Rc<RefCell<ClassDef>>>,
) -> Result<ClassReprArc, TyperError> {
    let Some(classdef_rc) = classdef else {
        // rclass.py:68-69 — `if classdef is None: return rtyper.rootclass_repr`.
        let root = rtyper.rootclass_repr.borrow().clone().ok_or_else(|| {
            TyperError::message(
                "rtyper.rootclass_repr is not set — call \
                 RPythonTyper::initialize_exceptiondata() after construction",
            )
        })?;
        return Ok(ClassReprArc::Root(root));
    };
    // rclass.py:70-73 — `classdef.repr` cache, fresh-build path.
    if let Some(existing) = classdef_rc.borrow().repr.clone() {
        return Ok(ClassReprArc::Inst(existing));
    }
    let repr = Arc::new(ClassRepr::new(rtyper, classdef_rc));
    classdef_rc.borrow_mut().repr = Some(repr.clone());
    rtyper.add_pendingsetup(repr.clone() as Arc<dyn Repr>);
    Ok(ClassReprArc::Inst(repr))
}

/// RPython `getinstancerepr(rtyper, classdef, default_flavor='gc')`
/// (`rclass.py:76-88`).
///
/// ```python
/// def getinstancerepr(rtyper, classdef, default_flavor='gc'):
///     if classdef is None:
///         flavor = default_flavor
///     else:
///         flavor = getgcflavor(classdef)
///     try:
///         result = rtyper.instance_reprs[classdef, flavor]
///     except KeyError:
///         result = buildinstancerepr(rtyper, classdef, gcflavor=flavor)
///         rtyper.instance_reprs[classdef, flavor] = result
///         rtyper.add_pendingsetup(result)
///     return result
/// ```
///
/// RPython `externalvsinternal(rtyper, item_repr, gcref=False)`
/// (rmodel.py:417-429).
///
/// ```python
/// def externalvsinternal(rtyper, item_repr, gcref=False):
///     ...
///     if (gcref and isinstance(item_repr.lowleveltype, Ptr) and
///             item_repr.lowleveltype.TO._gckind == 'gc'):
///         return item_repr, rgcref.GCRefRepr.make(item_repr, rtyper.gcrefreprcache)
///     if (isinstance(item_repr, rclass.InstanceRepr) and
///         getattr(item_repr, 'gcflavor', 'gc') == 'gc'):
///         return item_repr, rclass.getinstancerepr(rtyper, None)
///     else:
///         return item_repr, item_repr
/// ```
///
/// Returns `(external_repr, internal_repr)`. Used by container reprs
/// (TupleRepr / ListRepr / DictRepr) to store GC `InstanceRepr`s as
/// the root `InstanceRepr` (gcflavor='gc', classdef=None) internally
/// while preserving the concrete external repr at the surface.
///
/// Pyre placement deviation: upstream lives in `rmodel.py` and lazy-
/// imports `rclass`; the Rust port flips that — InstanceRepr lives in
/// `rclass.rs` and `rmodel.rs` cannot import `rclass` without
/// introducing a module-level cycle. Callers in `rtuple.rs` /
/// follow-on `rlist.rs` import this directly from `rclass`.
///
/// `gcref=True` arm (rmodel.py:422-424) is deferred until `rgcref` is
/// ported — the GCRef wrapping is dead code today (no caller passes
/// `gcref=True` from the minimal slice we have).
pub fn externalvsinternal(
    rtyper: &Rc<RPythonTyper>,
    item_repr: Arc<dyn Repr>,
) -> Result<(Arc<dyn Repr>, Arc<dyn Repr>), TyperError> {
    let any_r: &dyn std::any::Any = item_repr.as_ref();
    if let Some(inst) = any_r.downcast_ref::<InstanceRepr>() {
        if inst.gcflavor() == Flavor::Gc {
            let internal = getinstancerepr(rtyper, None, Flavor::Gc)?;
            let external = item_repr.clone();
            return Ok((external, internal as Arc<dyn Repr>));
        }
    }
    let external = item_repr.clone();
    Ok((external, item_repr))
}

pub fn getinstancerepr(
    rtyper: &Rc<RPythonTyper>,
    classdef: Option<&Rc<RefCell<ClassDef>>>,
    default_flavor: Flavor,
) -> Result<Arc<InstanceRepr>, TyperError> {
    let flavor = match classdef {
        None => default_flavor,
        Some(classdef_rc) => getgcflavor(classdef_rc)?,
    };
    let classdef_key = classdef.map(ClassDefKey::from_classdef);
    let key = (classdef_key, flavor);
    if let Some(existing) = rtyper.instance_reprs.borrow().get(&key) {
        return Ok(existing.clone());
    }
    let result = Arc::new(buildinstancerepr(rtyper, classdef, flavor)?);
    rtyper
        .instance_reprs
        .borrow_mut()
        .insert(key, result.clone());
    rtyper.add_pendingsetup(result.clone());
    Ok(result)
}

/// RPython `buildinstancerepr(rtyper, classdef, gcflavor='gc')`
/// (`rclass.py:91-119`).
///
pub fn buildinstancerepr(
    rtyper: &Rc<RPythonTyper>,
    classdef: Option<&Rc<RefCell<ClassDef>>>,
    gcflavor: Flavor,
) -> Result<InstanceRepr, TyperError> {
    // rclass.py:94-103 — `classdef is None` short-circuits `unboxed =
    // []`, `virtualizable = False`; otherwise scan all subdefs for
    // `UnboxedValue`-derived classes and read `_virtualizable_`. The
    // `usetagging` gate is `len(unboxed) != 0 and
    // config.translation.taggedpointers`, so when taggedpointers is off
    // the presence of an `UnboxedValue` subclass falls through to a
    // regular `InstanceRepr`.
    let (unboxed, virtualizable) = match classdef {
        None => (Vec::new(), false),
        Some(classdef_rc) => {
            let unboxed: Vec<Rc<RefCell<ClassDef>>> = ClassDef::getallsubdefs(classdef_rc)
                .into_iter()
                .filter(|subdef| {
                    host_is_unboxed_value_subclass(&subdef.borrow().classdesc.borrow().pyobj)
                })
                .collect();
            let virtualizable = const_truthy(&classdesc_get_param(
                classdef_rc,
                "_virtualizable_",
                ConstValue::Bool(false),
                true,
            ));
            (unboxed, virtualizable)
        }
    };
    // `rtyper.annotator` is weak because the annotator owns a strong
    // `Rc<RPythonTyper>` (rtyper.py `annotator.translator.rtyper` +
    // `rtyper.annotator` form a Python-side cycle; Rust breaks it by
    // keeping this edge weak). Tests that construct an RPythonTyper from
    // a non-retained `Rc<RPythonAnnotator>` observe the weak drop here.
    // Upstream `config.translation.taggedpointers` defaults to `False`,
    // so on upgrade failure we use the default — matching upstream
    // semantics for any caller that has not explicitly opted in to
    // tagged pointers.
    let taggedpointers = rtyper
        .annotator
        .upgrade()
        .map(|ann| ann.translator.config.translation.taggedpointers)
        .unwrap_or(false);
    let usetagging = !unboxed.is_empty() && taggedpointers;

    // rclass.py:105-108 — virtualizable path.
    if virtualizable {
        assert!(
            unboxed.is_empty(),
            "_virtualizable_ class must not have UnboxedValue subclasses"
        );
        assert_eq!(gcflavor, Flavor::Gc, "_virtualizable_ requires gc flavor");
        return Err(TyperError::message(
            "buildinstancerepr: VirtualizableInstanceRepr parity requires \
             rpython/rtyper/rvirtualizable.py (not yet ported)",
        ));
    }
    // rclass.py:109-117 — tagged-pointer path.
    if usetagging {
        if unboxed.len() != 1 {
            return Err(TyperError::message(format!(
                "{:?} has several UnboxedValue subclasses",
                classdef.map(|cd| cd.borrow().name.clone())
            )));
        }
        assert_eq!(
            gcflavor,
            Flavor::Gc,
            "UnboxedValue tagging requires gc flavor"
        );
        return Err(TyperError::message(
            "buildinstancerepr: TaggedInstanceRepr parity requires \
             rpython/rtyper/lltypesystem/rtagged.py (not yet ported)",
        ));
    }
    // rclass.py:118-119 — default `InstanceRepr`. An `UnboxedValue`
    // subclass without `config.translation.taggedpointers` falls through
    // here upstream too, so this arm owns that parity case.
    match classdef {
        None => Ok(InstanceRepr::new_rootinstance(rtyper, gcflavor)),
        Some(classdef_rc) => Ok(InstanceRepr::new(rtyper, Some(classdef_rc), gcflavor)),
    }
}

/// Cache key shape `(Option<ClassDefKey>, Flavor)` used by
/// [`RPythonTyper::instance_reprs`]. Exposed so consumers outside this
/// module can build matching queries without importing `Flavor` + the
/// field directly.
pub type InstanceReprKey = (Option<ClassDefKey>, Flavor);

// ---------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::rtyper::lltypesystem::lltype::GcKind;

    #[test]
    fn object_vtable_resolves_to_struct_with_subclassrange_fields() {
        let LowLevelType::ForwardReference(fwd) = OBJECT_VTABLE.clone() else {
            panic!("OBJECT_VTABLE must be ForwardReference");
        };
        let resolved = fwd.resolved().expect("OBJECT_VTABLE should be resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!("OBJECT_VTABLE must resolve to a Struct");
        };
        assert_eq!(body._name, "object_vtable");
        assert!(body._flds.get("subclassrange_min").is_some());
        assert!(body._flds.get("subclassrange_max").is_some());
        assert_eq!(body._hints.get("immutable"), Some(&ConstValue::Bool(true)));
        assert_eq!(
            body._hints.get("static_immutable"),
            Some(&ConstValue::Bool(true))
        );
    }

    /// `OBJECT_VTABLE.name: Ptr(rstr.STR)` (rclass.py:171). The `name`
    /// field is consumed by `ClassesPBCRepr.rtype_getattr('__name__')`
    /// (rpbc.py:980-981), which emits a `getfield` op against this
    /// slot. The lltype of the field must be `Ptr(STR)` resolving to
    /// the `rpy_string` GcStruct.
    /// `OBJECT_VTABLE.instantiate: Ptr(FuncType([], OBJECTPTR))`
    /// (rclass.py:172). Closes the cycle `OBJECT_VTABLE → OBJECTPTR
    /// → OBJECT → CLASSTYPE → OBJECT_VTABLE`. The Rust port resolves
    /// it via the `OBJECT_FAMILY` LazyLock that builds an unresolved
    /// ForwardReference first, constructs the four types, then
    /// `become()`s the ForwardReference to the full vtable Struct
    /// (mirroring upstream's mutable-ForwardReference + post-hoc
    /// `become` ordering).
    #[test]
    fn object_vtable_carries_instantiate_funcptr_returning_objectptr() {
        let LowLevelType::ForwardReference(fwd) = OBJECT_VTABLE.clone() else {
            panic!("OBJECT_VTABLE must be ForwardReference");
        };
        let LowLevelType::Struct(body) = fwd.resolved().expect("resolved") else {
            panic!("OBJECT_VTABLE must resolve to a Struct");
        };
        let instantiate_type = body
            ._flds
            .get("instantiate")
            .expect("instantiate field must be present (rclass.py:172)");
        let LowLevelType::Ptr(funcptr) = instantiate_type else {
            panic!("instantiate must be a Ptr, got {instantiate_type:?}");
        };
        let PtrTarget::Func(funcsig) = &funcptr.TO else {
            panic!(
                "instantiate's Ptr target must be Func, got {:?}",
                funcptr.TO
            );
        };
        assert!(
            funcsig.args.is_empty(),
            "instantiate funcptr has zero args, got {} args",
            funcsig.args.len()
        );
        // Result must be OBJECTPTR.
        assert_eq!(
            &funcsig.result, &*OBJECTPTR,
            "instantiate funcptr result must be OBJECTPTR"
        );
    }

    #[test]
    fn object_vtable_carries_name_ptr_to_str() {
        let LowLevelType::ForwardReference(fwd) = OBJECT_VTABLE.clone() else {
            panic!();
        };
        let LowLevelType::Struct(body) = fwd.resolved().expect("resolved") else {
            panic!();
        };
        let name_type = body
            ._flds
            .get("name")
            .expect("name field must be present (rclass.py:171)");
        let LowLevelType::Ptr(ptr) = name_type else {
            panic!("name must be a Ptr, got {name_type:?}");
        };
        // STR is a ForwardReference resolving to rpy_string.
        let PtrTarget::ForwardReference(fwd) = &ptr.TO else {
            panic!(
                "name's Ptr target must be ForwardReference (STR), got {:?}",
                ptr.TO
            );
        };
        let LowLevelType::Struct(str_body) = fwd.resolved().expect("STR must resolve") else {
            panic!("STR must resolve to a Struct");
        };
        assert_eq!(str_body._name, "rpy_string");
    }

    #[test]
    fn object_vtable_carries_rtti_ptr_to_runtime_type_info() {
        let LowLevelType::ForwardReference(fwd) = OBJECT_VTABLE.clone() else {
            panic!();
        };
        let LowLevelType::Struct(body) = fwd.resolved().expect("resolved") else {
            panic!();
        };
        let rtti_type = body
            ._flds
            .get("rtti")
            .expect("rtti field must be present (rclass.py:171)");
        let LowLevelType::Ptr(ptr) = rtti_type else {
            panic!("rtti must be a Ptr, got {rtti_type:?}");
        };
        let PtrTarget::Opaque(opaque) = &ptr.TO else {
            panic!("rtti must point to an opaque, got {:?}", ptr.TO);
        };
        assert_eq!(opaque.tag, "RuntimeTypeInfo");
    }

    #[test]
    fn classtype_is_ptr_to_object_vtable_forward_reference() {
        let LowLevelType::Ptr(ptr) = CLASSTYPE.clone() else {
            panic!("CLASSTYPE must be Ptr");
        };
        assert!(matches!(ptr.TO, PtrTarget::ForwardReference(_)));
    }

    #[test]
    fn object_is_gcstruct_with_typeptr_field_and_hints() {
        let LowLevelType::Struct(body) = OBJECT.clone() else {
            panic!("OBJECT must be Struct");
        };
        assert_eq!(body._name, "object");
        assert_eq!(body._gckind, GcKind::Gc);
        assert!(body._flds.get("typeptr").is_some());
        assert_eq!(body._hints.get("immutable"), Some(&ConstValue::Bool(true)));
        assert_eq!(
            body._hints.get("shouldntbenull"),
            Some(&ConstValue::Bool(true))
        );
        assert_eq!(body._hints.get("typeptr"), Some(&ConstValue::Bool(true)));
    }

    #[test]
    fn object_carries_rtti_opaque_under_runtime_type_info() {
        // rclass.py:162-165 sets `rtti=True` on OBJECT so
        // `getRuntimeTypeInfo(OBJECT)` resolves. R3 consumers
        // (`fill_vtable_root`) rely on this.
        let LowLevelType::Struct(body) = OBJECT.clone() else {
            panic!("OBJECT must be Struct");
        };
        let opaque = body
            ._runtime_type_info
            .as_ref()
            .expect("OBJECT must carry an _runtime_type_info opaque");
        assert_eq!(opaque._name.as_deref(), Some("object"));
    }

    #[test]
    fn nongcobject_is_raw_struct_with_typeptr_field() {
        let LowLevelType::Struct(body) = NONGCOBJECT.clone() else {
            panic!("NONGCOBJECT must be Struct");
        };
        assert_eq!(body._name, "nongcobject");
        assert_eq!(body._gckind, GcKind::Raw);
    }

    #[test]
    fn object_by_flavor_returns_gc_and_raw_forms() {
        assert_eq!(object_by_flavor(Flavor::Gc), OBJECT.clone());
        assert_eq!(object_by_flavor(Flavor::Raw), NONGCOBJECT.clone());
    }

    #[test]
    fn root_class_repr_lowleveltype_is_classtype() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let repr = RootClassRepr::new(&rtyper);
        assert_eq!(repr.lowleveltype(), &CLASSTYPE.clone());
        assert_eq!(repr.vtable_type(), &OBJECT_VTABLE.clone());
        assert_eq!(repr.classdef(), None);
    }

    #[test]
    fn root_class_repr_setup_is_noop_idempotent() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let repr = RootClassRepr::new(&rtyper);
        Repr::setup(&repr).expect("first setup");
        Repr::setup(&repr).expect("second setup re-enters Finished branch");
    }

    #[test]
    fn root_class_repr_init_vtable_writes_subclassrange_and_rtti() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let repr = rtyper
            .rootclass_repr
            .borrow()
            .clone()
            .expect("rootclass_repr after initialize");
        let vtable = repr.getvtable().expect("getvtable");
        let min = vtable.getattr("subclassrange_min").unwrap();
        let max = vtable.getattr("subclassrange_max").unwrap();
        assert_eq!(min, lltype::LowLevelValue::Signed(0));
        assert_eq!(max, lltype::LowLevelValue::Signed(i64::MAX));
        // rtti slot is populated when gcflavor == Gc.
        let rtti = vtable.getattr("rtti").unwrap();
        let lltype::LowLevelValue::Ptr(rtti_ptr) = rtti else {
            panic!("rtti must be a Ptr value");
        };
        assert!(
            rtti_ptr.nonzero(),
            "rtti pointer must be live for Gc flavor"
        );
    }

    #[test]
    fn class_repr_init_vtable_writes_subclassrange_from_classdef_minmax_id() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");

        let host = HostObject::new_class("C", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(class_rc) = &entry else {
            unreachable!();
        };
        let classdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc).unwrap();
        {
            let mut cd = classdef.borrow_mut();
            cd.minid = Some(7);
            cd.maxid = Some(11);
        }
        let r = getclassrepr_arc(&rtyper, Some(&classdef)).unwrap();
        Repr::setup(&*r.as_repr()).expect("setup classrepr");

        let ClassReprArc::Inst(inst_repr) = r else {
            panic!("expected ClassReprArc::Inst");
        };
        let vtable = inst_repr.getvtable().expect("getvtable");
        // The per-class vtable Struct is wrapped — `cast_vtable_to_typeptr`
        // already walked `super` down to the OBJECT_VTABLE level, so the
        // returned pointer carries the root subclassrange writes from
        // `fill_vtable_root`.
        let min = vtable.getattr("subclassrange_min").unwrap();
        let max = vtable.getattr("subclassrange_max").unwrap();
        assert_eq!(min, lltype::LowLevelValue::Signed(7));
        assert_eq!(max, lltype::LowLevelValue::Signed(11));
    }

    #[test]
    fn cast_vtable_to_typeptr_walks_super_chain_to_object_vtable() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");

        let host = HostObject::new_class("C", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(class_rc) = &entry else {
            unreachable!();
        };
        let classdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc).unwrap();
        {
            let mut cd = classdef.borrow_mut();
            cd.minid = Some(2);
            cd.maxid = Some(3);
        }
        let r = getclassrepr_arc(&rtyper, Some(&classdef)).unwrap();
        Repr::setup(&*r.as_repr()).expect("setup classrepr");
        let ClassReprArc::Inst(inst_repr) = r else {
            unreachable!();
        };
        // Internal pre-cast vtable still points at the per-class
        // struct (`C_vtable`); after `getvtable`'s
        // `cast_vtable_to_typeptr` it points at `OBJECT_VTABLE`.
        inst_repr.init_vtable().unwrap();
        let raw = inst_repr.vtable.borrow().clone().unwrap();
        let raw_to = LowLevelType::from(raw._TYPE.TO.clone());
        assert_ne!(
            raw_to,
            OBJECT_VTABLE.clone(),
            "raw vtable points at C_vtable"
        );

        let cast = cast_vtable_to_typeptr(raw);
        let cast_to = LowLevelType::from(cast._TYPE.TO.clone());
        assert_eq!(
            cast_to,
            OBJECT_VTABLE.clone(),
            "cast walked to OBJECT_VTABLE"
        );
    }

    #[test]
    fn root_class_repr_convert_desc_returns_vtable_constant() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        // ClassRepr.convert_desc upstream is the conversion entry for
        // SomeType-typed values (which route through `get_type_repr →
        // rtyper.rootclass_repr`). The receiver is therefore the root
        // ClassRepr whose `lowleveltype == CLASSTYPE`, satisfying the
        // `getruntime` assert. Non-root ClassRepr.convert_desc is
        // exercised only via the inherited shape but never reached
        // with a mismatched `expected_type`.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let host = HostObject::new_class("C", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(class_rc) = &entry else {
            unreachable!();
        };
        let classdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc).unwrap();
        classdef.borrow_mut().minid = Some(5);
        classdef.borrow_mut().maxid = Some(5);
        // Pre-setup the subclass ClassRepr so its vtable_type
        // ForwardReference is resolved when `getruntime → init_vtable`
        // runs from `convert_desc`. Real-world flow has this happen via
        // `RPythonTyper.specialize` → `_setup_repr` chain.
        let r_sub = getclassrepr_arc(&rtyper, Some(&classdef)).unwrap();
        Repr::setup(&*r_sub.as_repr()).unwrap();

        let root = rtyper.rootclass_repr.borrow().clone().unwrap();
        let out = root.convert_desc(&entry).expect("convert_desc(C)");
        let ConstValue::LLPtr(p) = &out.value else {
            panic!("expected LLPtr, got {:?}", out.value);
        };
        assert!(p.nonzero());
        assert_eq!(out.concretetype.as_ref(), Some(&CLASSTYPE.clone()));
    }

    #[test]
    fn root_class_repr_convert_const_routes_through_bookkeeper() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let host = HostObject::new_class("D", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(class_rc) = &entry else {
            unreachable!();
        };
        let classdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc).unwrap();
        classdef.borrow_mut().minid = Some(13);
        classdef.borrow_mut().maxid = Some(13);
        let r_sub = getclassrepr_arc(&rtyper, Some(&classdef)).unwrap();
        Repr::setup(&*r_sub.as_repr()).unwrap();
        let root = rtyper.rootclass_repr.borrow().clone().unwrap();
        let out = Repr::convert_const(root.as_ref(), &ConstValue::HostObject(host))
            .expect("convert_const HostObject(class)");
        let ConstValue::LLPtr(p) = &out.value else {
            panic!("expected LLPtr, got {:?}", out.value);
        };
        assert!(p.nonzero());

        // Non-class HostObject is rejected upfront.
        let module_host = HostObject::new_module("not_a_class_mod");
        let err =
            Repr::convert_const(root.as_ref(), &ConstValue::HostObject(module_host)).unwrap_err();
        assert!(err.to_string().contains("not a class"));
    }

    #[test]
    fn class_repr_convert_desc_rejects_unrelated_class() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        // Two unrelated classes (no common ancestor in HostObject mro
        // beyond `object`). The receiver `inst_a` rejects converting
        // for desc(B) because B is not a subclass of A.
        let host_a = HostObject::new_class("A", vec![]);
        let host_b = HostObject::new_class("B", vec![]);
        let entry_a = ann.bookkeeper.getdesc(&host_a).unwrap();
        let entry_b = ann.bookkeeper.getdesc(&host_b).unwrap();
        let crate::annotator::description::DescEntry::Class(rc_a) = &entry_a else {
            unreachable!();
        };
        let cd_a = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc_a).unwrap();
        cd_a.borrow_mut().minid = Some(1);
        cd_a.borrow_mut().maxid = Some(1);

        let r_a = getclassrepr_arc(&rtyper, Some(&cd_a)).unwrap();
        Repr::setup(&*r_a.as_repr()).unwrap();
        let ClassReprArc::Inst(inst_a) = r_a else {
            unreachable!();
        };
        // Subclass check fires before the (assert-violating) getruntime
        // call, so this exercises just the commonbase-rejection branch.
        let err = inst_a.convert_desc(&entry_b).unwrap_err();
        assert!(err.to_string().contains("not a subclass"), "{err}");
    }

    #[test]
    fn class_repr_fromtypeptr_emits_cast_pointer_with_self_lowleveltype() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let host = HostObject::new_class("E", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(rc) = &entry else {
            unreachable!();
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        cd.borrow_mut().minid = Some(1);
        cd.borrow_mut().maxid = Some(1);
        let r = getclassrepr_arc(&rtyper, Some(&cd)).unwrap();
        Repr::setup(&*r.as_repr()).unwrap();
        let ClassReprArc::Inst(inst) = r else {
            unreachable!();
        };

        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let mut vcls = Variable::new();
        vcls.set_concretetype(Some(CLASSTYPE.clone()));
        let v = inst
            .fromtypeptr(Hlvalue::Variable(vcls), &mut llops)
            .expect("fromtypeptr");
        assert_eq!(llops.ops.len(), 1);
        assert_eq!(llops.ops[0].opname, "cast_pointer");
        assert_eq!(v.concretetype().as_ref(), Some(inst.lowleveltype()));
    }

    #[test]
    fn class_repr_getclsfield_traverses_rbase_until_root_then_errors() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let host = HostObject::new_class("F", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(rc) = &entry else {
            unreachable!();
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        cd.borrow_mut().minid = Some(1);
        cd.borrow_mut().maxid = Some(1);
        let r = getclassrepr_arc(&rtyper, Some(&cd)).unwrap();
        Repr::setup(&*r.as_repr()).unwrap();
        let ClassReprArc::Inst(inst) = r else {
            unreachable!();
        };

        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let mut vcls = Variable::new();
        vcls.set_concretetype(Some(CLASSTYPE.clone()));
        // No clsfield named "missing_attr" anywhere in the chain →
        // routes through F → rbase (RootClassRepr) which terminates
        // with a MissingRTypeAttribute-style structured error.
        let err = inst
            .getclsfield(Hlvalue::Variable(vcls), "missing_attr", &mut llops)
            .unwrap_err();
        assert!(err.to_string().contains("MissingRTypeAttribute"));
    }

    #[test]
    fn class_repr_getpbcfield_internal_error_when_field_unknown() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let host = HostObject::new_class("G", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(rc) = &entry else {
            unreachable!();
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        cd.borrow_mut().minid = Some(1);
        cd.borrow_mut().maxid = Some(1);
        let r = getclassrepr_arc(&rtyper, Some(&cd)).unwrap();
        Repr::setup(&*r.as_repr()).unwrap();
        let ClassReprArc::Inst(inst) = r else {
            unreachable!();
        };

        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let mut vcls = Variable::new();
        vcls.set_concretetype(Some(CLASSTYPE.clone()));
        let err = inst
            .getpbcfield(Hlvalue::Variable(vcls), 0, "missing_pbc_attr", &mut llops)
            .unwrap_err();
        assert!(err.to_string().contains("missing PBC field"));
    }

    #[test]
    fn class_repr_rtype_issubtype_const_branch_emits_direct_call_to_ll_issubclass_const() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{HostObject, SpaceOperation};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::HighLevelOp;
        use std::cell::RefCell as StdRef;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let host = HostObject::new_class("Z", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(rc) = &entry else {
            unreachable!();
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        cd.borrow_mut().minid = Some(2);
        cd.borrow_mut().maxid = Some(2);
        let r = getclassrepr_arc(&rtyper, Some(&cd)).unwrap();
        Repr::setup(&*r.as_repr()).unwrap();
        let ClassReprArc::Inst(inst) = r else {
            unreachable!();
        };

        // Build a HighLevelOp where v_cls2 is a HostObject-typed
        // class Constant. `inputargs(class_repr)` routes through
        // RootClassRepr.convert_const → convert_desc → getruntime,
        // producing a CLASSTYPE-typed Constant whose value is the
        // vtable `_ptr`. The const-branch probe then reads
        // `subclassrange_min/max` off that vtable.
        let mut v_cls1 = Variable::new();
        v_cls1.set_concretetype(Some(CLASSTYPE.clone()));
        let v_cls1_h = Hlvalue::Variable(v_cls1);

        let v_cls2_h = Hlvalue::Constant(Constant::new(ConstValue::HostObject(host)));

        let result_var = Variable::new();
        result_var.set_concretetype(Some(LowLevelType::Bool));
        let spaceop = SpaceOperation::new(
            OpKind::IsSubtype.opname(),
            vec![v_cls1_h.clone(), v_cls2_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_cls1_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        let class_repr_arc: Arc<dyn Repr> =
            rtyper.rootclass_repr.borrow().clone().unwrap() as Arc<dyn Repr>;
        hop.args_r.borrow_mut().push(Some(class_repr_arc.clone()));
        hop.args_v.borrow_mut().push(v_cls2_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(class_repr_arc));

        let _ = inst.rtype_issubtype(&hop).expect("rtype_issubtype");
        let ops = hop.llops.borrow();
        let dc = ops
            .ops
            .iter()
            .find(|op| op.opname == "direct_call")
            .expect("direct_call to ll_issubclass_const helper expected");
        // direct_call args: [funcptr, v_cls1, c_min, c_max]
        assert_eq!(dc.args.len(), 4);
        let Hlvalue::Constant(c_min) = &dc.args[2] else {
            panic!("c_min must be a Constant");
        };
        let Hlvalue::Constant(c_max) = &dc.args[3] else {
            panic!("c_max must be a Constant");
        };
        assert_eq!(c_min.value, ConstValue::Int(2));
        assert_eq!(c_max.value, ConstValue::Int(2));
    }

    #[test]
    fn class_repr_rtype_issubtype_var_branch_emits_direct_call_to_ll_issubclass() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::HighLevelOp;
        use std::cell::RefCell as StdRef;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let root = rtyper.rootclass_repr.borrow().clone().unwrap();
        // Use rootclass to keep the example simple — the variable
        // branch doesn't need a subclass classdef.

        let mut v_cls1 = Variable::new();
        v_cls1.set_concretetype(Some(CLASSTYPE.clone()));
        let mut v_cls2 = Variable::new();
        v_cls2.set_concretetype(Some(CLASSTYPE.clone()));
        let v_cls1_h = Hlvalue::Variable(v_cls1);
        let v_cls2_h = Hlvalue::Variable(v_cls2);
        let result_var = Variable::new();
        result_var.set_concretetype(Some(LowLevelType::Bool));
        let spaceop = SpaceOperation::new(
            OpKind::IsSubtype.opname(),
            vec![v_cls1_h.clone(), v_cls2_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_cls1_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        let class_repr_arc: Arc<dyn Repr> = root.clone() as Arc<dyn Repr>;
        hop.args_r.borrow_mut().push(Some(class_repr_arc.clone()));
        hop.args_v.borrow_mut().push(v_cls2_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(class_repr_arc));

        // Build a synthetic ClassRepr instance just so we can invoke
        // rtype_issubtype. The receiver only needs `rtyper` for
        // helper-function lookup; classdef.minid/maxid are not read in
        // the variable branch.
        let host = crate::flowspace::model::HostObject::new_class("V", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(rc) = &entry else {
            unreachable!();
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        cd.borrow_mut().minid = Some(0);
        cd.borrow_mut().maxid = Some(0);
        let r = getclassrepr_arc(&rtyper, Some(&cd)).unwrap();
        Repr::setup(&*r.as_repr()).unwrap();
        let ClassReprArc::Inst(inst) = r else {
            unreachable!();
        };

        let _ = inst.rtype_issubtype(&hop).expect("rtype_issubtype");
        let ops = hop.llops.borrow();
        let dc = ops
            .ops
            .iter()
            .find(|op| op.opname == "direct_call")
            .expect("direct_call expected");
        // funcptr + 2 args.
        assert_eq!(dc.args.len(), 3);
    }

    /// `RPythonTyper::translate_operation("issubtype")` dispatches
    /// `r.rtype_issubtype(hop)` via the `Repr` trait. Without the
    /// trait override on `RootClassRepr` (which is what `SomeType`
    /// values rtype to) this would fall to the trait default
    /// `missing_rtype_operation`. The test calls
    /// `translate_operation` end-to-end to pin the trait dispatch
    /// path — not just the inherent method.
    #[test]
    fn translate_operation_issubtype_routes_through_root_class_repr_trait_override() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::HighLevelOp;
        use std::cell::RefCell as StdRef;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let root = rtyper.rootclass_repr.borrow().clone().unwrap();

        let mut v_cls1 = Variable::new();
        v_cls1.set_concretetype(Some(CLASSTYPE.clone()));
        let mut v_cls2 = Variable::new();
        v_cls2.set_concretetype(Some(CLASSTYPE.clone()));
        let v_cls1_h = Hlvalue::Variable(v_cls1);
        let v_cls2_h = Hlvalue::Variable(v_cls2);
        let result_var = Variable::new();
        result_var.set_concretetype(Some(LowLevelType::Bool));
        let spaceop = SpaceOperation::new(
            OpKind::IsSubtype.opname(),
            vec![v_cls1_h.clone(), v_cls2_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        let class_repr_arc: Arc<dyn Repr> = root as Arc<dyn Repr>;
        hop.args_v.borrow_mut().push(v_cls1_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(class_repr_arc.clone()));
        hop.args_v.borrow_mut().push(v_cls2_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(class_repr_arc));

        rtyper
            .translate_operation(&hop)
            .expect("translate_operation issubtype must dispatch through RootClassRepr");
        let ops = hop.llops.borrow();
        assert!(
            ops.ops.iter().any(|op| op.opname == "direct_call"),
            "translate_operation issubtype must emit a direct_call (ll_issubclass), \
             got {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>()
        );
    }

    #[test]
    fn root_class_repr_getruntime_returns_vtable() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let repr = rtyper.rootclass_repr.borrow().clone().unwrap();
        let v1 = repr.getruntime(&CLASSTYPE.clone()).unwrap();
        let v2 = repr.getruntime(&CLASSTYPE.clone()).unwrap();
        // Lazy cache: two calls return the same _ptr identity.
        assert_eq!(v1._hashable_identity(), v2._hashable_identity());
        let err = repr.getruntime(&LowLevelType::Signed).unwrap_err();
        assert!(err.to_string().contains("expected CLASSTYPE"));
    }

    #[test]
    fn instance_repr_none_classdef_carries_gc_object_type() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let repr = InstanceRepr::new_rootinstance(&rtyper, Flavor::Gc);
        assert!(repr.classdef().is_none());
        assert_eq!(repr.gcflavor(), Flavor::Gc);
        assert_eq!(repr.object_type(), &OBJECT.clone());
        let LowLevelType::Ptr(ptr) = repr.lowleveltype().clone() else {
            panic!("InstanceRepr.lowleveltype must be Ptr");
        };
        assert!(matches!(ptr.TO, PtrTarget::Struct(_)));
    }

    #[test]
    fn instance_repr_none_classdef_setup_ok() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let repr = InstanceRepr::new_rootinstance(&rtyper, Flavor::Gc);
        Repr::setup(&repr).expect("setup classdef=None path");
    }

    // -----------------------------------------------------------------
    // R2-A — ClassRepr scaffold + getclassrepr(classdef != None).
    // -----------------------------------------------------------------

    fn fresh_rtyper() -> Rc<RPythonTyper> {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        rtyper
    }

    #[test]
    fn instance_repr_none_classdef_populates___class___field() {
        let rtyper = fresh_rtyper();
        let repr = getinstancerepr(&rtyper, None, Flavor::Gc).expect("root instance repr");
        Repr::setup(repr.as_ref()).expect("setup root instance repr");

        let fields = repr.fields();
        let (mangled, field_repr) = fields
            .get("__class__")
            .expect("root instance must expose __class__");
        assert_eq!(mangled, "typeptr");
        assert_eq!(field_repr.lowleveltype(), &CLASSTYPE.clone());
        assert!(repr.allinstancefields().contains_key("__class__"));
    }

    #[test]
    fn instance_repr_get_reusable_prebuilt_instance_caches_and_writes_typeptr() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");

        let host = HostObject::new_class("ExcClass", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(rc) = &entry else {
            unreachable!();
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        cd.borrow_mut().minid = Some(11);
        cd.borrow_mut().maxid = Some(11);

        let inst = getinstancerepr(&rtyper, Some(&cd), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(inst.as_ref() as &dyn Repr).expect("setup InstanceRepr");
        // Setup the matching ClassRepr too so its vtable_type
        // ForwardReference is resolved when `initialize_prebuilt_data`
        // walks down to root and calls `getvtable`.
        let rclass_arc = getclassrepr_arc(&rtyper, Some(&cd)).unwrap();
        Repr::setup(&*rclass_arc.as_repr()).unwrap();

        let p1 = inst.get_reusable_prebuilt_instance().expect("first call");
        let p2 = inst.get_reusable_prebuilt_instance().expect("second call");
        // upstream `try: return self._reusable_prebuilt_instance`
        // returns the cached identity on subsequent calls.
        assert_eq!(p1._hashable_identity(), p2._hashable_identity());
        assert!(p1.nonzero(), "prebuilt instance must be live");

        // The recursive initialize_prebuilt_data walked down to the
        // OBJECT-level substruct via `setattr_at_path(["super"],
        // "typeptr", ...)` so the root super substruct now carries the
        // vtable pointer. Verify by fetching the parent allocation's
        // `super._fields` directly.
        let _ptr_obj_actual = p1
            ._obj()
            .expect("prebuilt _ptr must expose underlying object");
        let lltype::_ptr_obj::Struct(outer) = _ptr_obj_actual else {
            panic!("prebuilt instance must be a Struct");
        };
        let super_field = outer
            ._getattr("super")
            .expect("ExcClass instance has 'super' field");
        let lltype::LowLevelValue::Struct(super_struct) = super_field else {
            panic!("super field must be a Struct value");
        };
        let typeptr_val = super_struct
            ._getattr("typeptr")
            .expect("OBJECT.typeptr field present");
        let lltype::LowLevelValue::Ptr(typeptr) = typeptr_val else {
            panic!("typeptr must be a Ptr value");
        };
        assert!(typeptr.nonzero(), "typeptr must point at the class vtable");
    }

    #[test]
    fn instance_repr_create_instance_allocates_immortal_struct() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let host = HostObject::new_class("Bare", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(rc) = &entry else {
            unreachable!();
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        let inst = getinstancerepr(&rtyper, Some(&cd), Flavor::Gc).unwrap();
        Repr::setup(inst.as_ref() as &dyn Repr).unwrap();
        let p = inst.create_instance().expect("create_instance");
        assert!(p.nonzero());
    }

    #[test]
    fn instance_repr_convert_const_none_returns_null_instance_constant() {
        // rclass.py:773-774 — `if value is None: return self.null_instance()`.
        // null_instance returns `nullptr(self.object_type)` wrapped as
        // ConstValue::LLPtr; concretetype tracks `self.lowleveltype`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("init exceptiondata");
        let host = HostObject::new_class("Bare2", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let DescEntry::Class(rc) = &entry else {
            unreachable!()
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        let inst = getinstancerepr(&rtyper, Some(&cd), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(inst.as_ref() as &dyn Repr).expect("setup InstanceRepr");

        let c = (inst.as_ref() as &dyn Repr)
            .convert_const(&ConstValue::None)
            .expect("convert_const(None)");
        assert_eq!(c.concretetype.as_ref(), Some(inst.lowleveltype()));
        let ConstValue::LLPtr(ptr) = &c.value else {
            panic!("convert_const(None) must produce LLPtr, got {:?}", c.value);
        };
        assert!(!ptr.nonzero(), "null_instance must be a null pointer");
    }

    #[test]
    fn instance_repr_convert_const_exact_caches_prebuilt_instance() {
        // rclass.py:794-802 — `convert_const_exact` populates
        // `iprebuiltinstances[value]` on first call. Re-issuing
        // `convert_const(value)` returns the same `_ptr` from the cache.
        // For a class with no instance attrs (`fields` empty), the
        // initialize_prebuilt_data branch only writes the root
        // `typeptr`, so the cache identity is observable through the
        // resulting `LLPtr` `_identity` equality.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("init exceptiondata");
        let host = HostObject::new_class("PrebuiltCls", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let DescEntry::Class(rc) = &entry else {
            unreachable!()
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        let inst = getinstancerepr(&rtyper, Some(&cd), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(inst.as_ref() as &dyn Repr).expect("setup InstanceRepr");
        // Drain pendingsetup so the associated ClassRepr (vtable_type
        // ForwardReference) is also resolved before
        // initialize_prebuilt_data writes typeptr.
        rtyper.call_all_setups().expect("call_all_setups");
        // ClassRepr.init_vtable wants `classdef.minid`, populated by
        // normalizecalls.assign_inheritance_ids.
        crate::translator::rtyper::normalizecalls::assign_inheritance_ids(&ann);

        // Build a prebuilt host instance so convert_const reaches the
        // exact-match arm.
        let prebuilt = HostObject::new_instance(host.clone(), vec![]);
        let value = ConstValue::HostObject(prebuilt.clone());
        let c1 = (inst.as_ref() as &dyn Repr)
            .convert_const(&value)
            .expect("convert_const(prebuilt)#1");
        let c2 = (inst.as_ref() as &dyn Repr)
            .convert_const(&value)
            .expect("convert_const(prebuilt)#2");
        let (ConstValue::LLPtr(p1), ConstValue::LLPtr(p2)) = (&c1.value, &c2.value) else {
            panic!(
                "convert_const must produce LLPtr, got {:?} / {:?}",
                c1.value, c2.value
            );
        };
        // upstream identity: `iprebuiltinstances[value]` returns the
        // same ptr each call. Pyre `_ptr` equality is by `_identity`
        // which derives from the underlying object identity.
        assert_eq!(
            p1._hashable_identity(),
            p2._hashable_identity(),
            "iprebuiltinstances cache must return the same _ptr identity"
        );
        // The cache holds exactly one entry for this prebuilt.
        assert_eq!(inst.iprebuiltinstances.borrow().len(), 1);
    }

    #[test]
    fn instance_repr_convert_const_non_hostobject_rejected() {
        // rclass.py:778-781 (`bk.getuniqueclassdef(value.__class__)`) —
        // pyre's `getuniqueclassdef` only accepts HostObject inputs, so
        // the convert_const port rejects non-HostObject ConstValues
        // before reaching the bookkeeper.
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.RejectShape", None);
        let inst = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(inst.as_ref() as &dyn Repr).expect("setup InstanceRepr");
        let err = (inst.as_ref() as &dyn Repr)
            .convert_const(&ConstValue::Int(7))
            .unwrap_err();
        assert!(
            err.to_string().contains("expected HostObject or None"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn classrepr_new_stores_classdef_and_unresolved_vtable_forward_reference() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let repr = ClassRepr::new(&rtyper, &classdef);

        // rclass.py:194 — `self.classdef = classdef`.
        assert!(Rc::ptr_eq(&repr.classdef(), &classdef));

        // rclass.py:195 — vtable_type is a fresh ForwardReference (no
        // become() yet).
        let LowLevelType::ForwardReference(fwd) = repr.vtable_type() else {
            panic!("ClassRepr.vtable_type must be ForwardReference before setup");
        };
        assert!(
            fwd.resolved().is_none(),
            "vtable_type should not be resolved until _setup_repr runs"
        );

        // rclass.py:196 — `lowleveltype = Ptr(self.vtable_type)`.
        let LowLevelType::Ptr(ptr) = repr.lowleveltype() else {
            panic!("ClassRepr.lowleveltype must be Ptr");
        };
        assert!(matches!(ptr.TO, PtrTarget::ForwardReference(_)));
    }

    #[test]
    fn classrepr_setup_resolves_vtable_struct_with_super_pointing_at_root_vtable() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let repr_arc =
            match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr_arc(classdef)") {
                ClassReprArc::Inst(r) => r,
                _ => panic!("classdef != None should return ClassReprArc::Inst"),
            };

        Repr::setup(repr_arc.as_ref()).expect("ClassRepr._setup_repr");

        // rclass.py:276-279 — vtable_type resolves to `<name>_vtable`
        // Struct with a single ('super', rbase.vtable_type) field.
        let LowLevelType::ForwardReference(fwd) = repr_arc.vtable_type() else {
            panic!("vtable_type must be ForwardReference");
        };
        let resolved = fwd.resolved().expect("vtable_type must be resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!("vtable_type must resolve to Struct");
        };
        assert_eq!(body._name, "pkg.C_vtable");
        let super_field = body
            ._flds
            .get("super")
            .expect("vtable Struct must carry a 'super' field");

        // rbase for classdef with no basedef = RootClassRepr, whose
        // vtable_type is OBJECT_VTABLE (via rclass.py:69 + rclass.py:426).
        assert_eq!(*super_field, OBJECT_VTABLE.clone());
        assert_eq!(body._hints.get("immutable"), Some(&ConstValue::Bool(true)));
        assert_eq!(
            body._hints.get("static_immutable"),
            Some(&ConstValue::Bool(true))
        );
    }

    #[test]
    fn getclassrepr_caches_second_lookup_returns_same_arc() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let first = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("first") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        let second = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("second") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        assert!(
            Arc::ptr_eq(&first, &second),
            "getclassrepr must cache by ClassDefKey identity"
        );
    }

    #[test]
    fn classrepr_basedef_recursion_threads_parent_vtable_type_into_super_field() {
        let rtyper = fresh_rtyper();
        let base = ClassDef::new_standalone("pkg.Base", None);
        let child = ClassDef::new_standalone("pkg.Child", Some(&base));

        // Force both reprs to exist so we can compare their vtable_types.
        let base_arc = match getclassrepr_arc(&rtyper, Some(&base)).expect("base") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        let child_arc = match getclassrepr_arc(&rtyper, Some(&child)).expect("child") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };

        // Setup child: internally triggers base.setup() via rclass.py:274.
        Repr::setup(child_arc.as_ref()).expect("child setup");

        // Both reprs must now be FINISHED.
        let resolved_base = match base_arc.vtable_type() {
            LowLevelType::ForwardReference(fwd) => fwd
                .resolved()
                .expect("base vtable_type must be resolved via child setup"),
            _ => panic!(),
        };
        let resolved_child = match child_arc.vtable_type() {
            LowLevelType::ForwardReference(fwd) => {
                fwd.resolved().expect("child vtable_type must be resolved")
            }
            _ => panic!(),
        };

        // Child's 'super' field type must equal base's vtable_type
        // (resolved LowLevelType::ForwardReference that compares equal
        // via inner Arc<Mutex> target).
        let LowLevelType::Struct(child_body) = resolved_child else {
            panic!("child vtable must be Struct");
        };
        let super_field = child_body
            ._flds
            .get("super")
            .expect("child vtable must have 'super'");
        // rclass.py:277 passes `rbase.vtable_type` (a ForwardReference
        // LowLevelType wrapper around the same Arc<Mutex> that the
        // base repr holds). We stored a clone in the 'super' slot, so
        // both should resolve to the same inner Struct body.
        let LowLevelType::ForwardReference(super_fwd) = super_field else {
            panic!("'super' field type must be ForwardReference to base vtable");
        };
        assert_eq!(
            super_fwd
                .resolved()
                .expect("super field's ForwardReference resolved"),
            resolved_base,
        );
    }

    #[test]
    fn getclassrepr_arc_none_classdef_returns_rootclass_repr() {
        let rtyper = fresh_rtyper();
        let arc = getclassrepr_arc(&rtyper, None).expect("None classdef");
        match arc {
            ClassReprArc::Root(r) => assert_eq!(r.classdef(), None),
            _ => panic!("None classdef must route to Root"),
        }
    }

    #[test]
    fn getclassrepr_without_rootclass_repr_returns_typererror_for_none() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        // No initialize_exceptiondata — rootclass_repr stays None.
        let err =
            getclassrepr_arc(&rtyper, None).expect_err("should surface missing rootclass_repr");
        assert!(
            err.to_string().contains("initialize_exceptiondata"),
            "error must point at initialize_exceptiondata, got: {err}",
        );
    }

    #[test]
    fn classrepr_prepare_method_rewrites_method_pbc_to_function_pbc() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::annotator::classdesc::ClassDesc;
        use crate::annotator::description::{FunctionDesc, MethodDesc};
        use crate::flowspace::argument::Signature;

        let rtyper = fresh_rtyper();
        let bk = Rc::new(Bookkeeper::new());
        let pyobj = HostObject::new_class("pkg.C", vec![]);
        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            pyobj,
            "pkg.C".into(),
        )));
        let classdef = ClassDef::new(&bk, &desc);
        bk.register_classdef(classdef.clone());
        let funcdesc = Rc::new(RefCell::new(FunctionDesc::new(
            bk.clone(),
            None,
            "pkg.C.method",
            Signature::new(vec!["self".to_string()], None, None),
            None,
            None,
        )));
        let method = Rc::new(RefCell::new(MethodDesc::new(
            bk,
            funcdesc.clone(),
            ClassDefKey::from_classdef(&classdef),
            None,
            "method",
            BTreeMap::new(),
        )));
        let s_value = SomeValue::PBC(SomePBC::new(vec![DescEntry::Method(method)], true));

        let repr = ClassRepr::new(&rtyper, &classdef);
        let prepared = repr
            .prepare_method(&s_value)
            .expect("prepare_method")
            .expect("method SomePBC should be rewritten");
        let SomeValue::PBC(prepared_pbc) = prepared else {
            panic!("prepare_method must keep a SomePBC");
        };
        assert_eq!(
            prepared_pbc.get_kind().expect("PBC kind"),
            DescKind::Function
        );
        assert!(
            !prepared_pbc.can_be_none,
            "prepare_method must drop nullable method-PBC shape like upstream"
        );
        assert_eq!(prepared_pbc.descriptions.len(), 1);
        assert!(
            prepared_pbc
                .descriptions
                .values()
                .next()
                .expect("rewritten PBC entry")
                .is_function()
        );
    }

    #[test]
    fn classrepr_setup_materializes_extra_access_sets() {
        use crate::annotator::description::{ClassAttrFamily, DescKey};

        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let access_set = Rc::new(RefCell::new(ClassAttrFamily::new(DescKey::from_rc(
            &classdef.borrow().classdesc,
        ))));
        access_set.borrow_mut().s_value = SomeValue::Bool(crate::annotator::model::SomeBool::new());
        let access_key = class_attr_family_key(&access_set);
        classdef
            .borrow_mut()
            .extra_access_sets
            .insert(access_key, (access_set, "x".to_string(), 0));

        let repr = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        Repr::setup(repr.as_ref()).expect("extra_access_sets setup");
        let pbcfields = repr.pbcfields.borrow();
        assert!(pbcfields.contains_key(&(access_key, "x".to_string())));
    }

    // -----------------------------------------------------------------
    // R2-B — InstanceRepr classdef != None scaffolding.
    // -----------------------------------------------------------------

    #[test]
    fn instance_repr_new_with_classdef_stores_forward_reference_object_type() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let repr = InstanceRepr::new(&rtyper, Some(&classdef), Flavor::Gc);

        // rclass.py:470 — classdef field.
        let stored = repr.classdef().expect("classdef must be Some");
        assert!(Rc::ptr_eq(&stored, &classdef));

        // rclass.py:474-475 — object_type is a fresh GcForwardReference.
        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!("object_type must be ForwardReference when classdef != None");
        };
        assert!(fwd.resolved().is_none());

        // rclass.py:477 — lowleveltype is Ptr(self.object_type).
        let LowLevelType::Ptr(ptr) = repr.lowleveltype() else {
            panic!("lowleveltype must be Ptr");
        };
        assert!(matches!(ptr.TO, PtrTarget::ForwardReference(_)));
    }

    #[test]
    fn buildinstancerepr_unboxedvalue_without_taggedpointers_falls_back_to_plain_instance_repr() {
        use crate::annotator::annrpython::RPythonAnnotator;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let classdef = ClassDef::new_standalone("pkg.UnboxedValue", None);

        let repr =
            buildinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("plain InstanceRepr");

        let stored = repr.classdef().expect("classdef must be Some");
        assert!(Rc::ptr_eq(&stored, &classdef));
        Repr::setup(&repr).expect("plain InstanceRepr setup");
    }

    #[test]
    fn instance_repr_setup_resolves_object_type_to_struct_with_super_pointing_at_root_object() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc)
            .expect("getinstancerepr(classdef)");

        Repr::setup(repr.as_ref()).expect("InstanceRepr._setup_repr");

        // rclass.py:548-554 — object_type resolves to Struct('pkg.C',
        // ('super', rbase.object_type)) where rbase is the root
        // InstanceRepr's OBJECT (classdef=None).
        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!("object_type must still be ForwardReference wrapper");
        };
        let resolved = fwd.resolved().expect("object_type must be resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!("object_type must resolve to Struct");
        };
        assert_eq!(body._name, "pkg.C");
        assert_eq!(body._gckind, GcKind::Gc);
        let super_field = body
            ._flds
            .get("super")
            .expect("object_type Struct must carry 'super' field");
        assert_eq!(*super_field, OBJECT.clone());
    }

    #[test]
    fn instance_repr_gc_setup_attaches_runtime_type_info_opaque_on_body_struct() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.RttiC", None);
        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc)
            .expect("getinstancerepr(classdef)");

        Repr::setup(repr.as_ref()).expect("InstanceRepr._setup_repr");

        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!("object_type must be ForwardReference wrapper");
        };
        let resolved = fwd.resolved().expect("object_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!("object_type must resolve to Struct");
        };
        // rclass.py:531-532 — gc-flavor instance reprs pass rtti=True
        // so `getRuntimeTypeInfo(body)` succeeds after setup.
        let rtti = body
            ._runtime_type_info
            .as_ref()
            .expect("gc-flavor instance body must carry _runtime_type_info");
        assert_eq!(rtti.TYPE.tag, "RuntimeTypeInfo");
        assert_eq!(rtti._name.as_deref(), Some("pkg.RttiC"));
    }

    #[test]
    fn instance_repr_raw_setup_does_not_attach_runtime_type_info_opaque() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.RawC", None);
        classdef
            .borrow()
            .classdesc
            .borrow()
            .pyobj
            .class_set("_alloc_flavor_", ConstValue::byte_str("raw"));
        let repr =
            getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("getinstancerepr raw");
        assert_eq!(repr.gcflavor(), Flavor::Raw);
        Repr::setup(repr.as_ref()).expect("setup raw");
        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!();
        };
        let resolved = fwd.resolved().expect("raw resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!();
        };
        assert!(body._runtime_type_info.is_none());
    }

    #[test]
    fn getinstancerepr_caches_by_classdef_identity() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let first = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("first");
        let second = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("second");
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn getinstancerepr_reads_alloc_flavor_from_classdesc() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.RawC", None);
        classdef
            .borrow()
            .classdesc
            .borrow()
            .pyobj
            .class_set("_alloc_flavor_", ConstValue::byte_str("raw"));

        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("raw repr");
        assert_eq!(repr.gcflavor(), Flavor::Raw);
    }

    #[test]
    fn instance_repr_basedef_recursion_threads_parent_object_type_into_super_field() {
        let rtyper = fresh_rtyper();
        let base = ClassDef::new_standalone("pkg.Base", None);
        let child = ClassDef::new_standalone("pkg.Child", Some(&base));
        let base_repr = getinstancerepr(&rtyper, Some(&base), Flavor::Gc).expect("base");
        let child_repr = getinstancerepr(&rtyper, Some(&child), Flavor::Gc).expect("child");

        Repr::setup(child_repr.as_ref()).expect("child setup");

        // Both reprs must now be FINISHED.
        let LowLevelType::ForwardReference(child_fwd) = child_repr.object_type() else {
            panic!();
        };
        let child_resolved = child_fwd.resolved().expect("child resolved");
        let LowLevelType::ForwardReference(base_fwd) = base_repr.object_type() else {
            panic!();
        };
        let base_resolved = base_fwd.resolved().expect("base resolved via child");

        let LowLevelType::Struct(child_body) = child_resolved else {
            panic!();
        };
        let super_field = child_body
            ._flds
            .get("super")
            .expect("child object_type must have 'super'");
        let LowLevelType::ForwardReference(super_fwd) = super_field else {
            panic!("child 'super' field must be a ForwardReference pointing at base");
        };
        assert_eq!(
            super_fwd.resolved().expect("super field resolved"),
            base_resolved
        );
    }

    // -----------------------------------------------------------------
    // R2-C — attrs iteration on ClassRepr + InstanceRepr.
    // -----------------------------------------------------------------

    fn attach_attr(
        classdef: &Rc<RefCell<ClassDef>>,
        name: &str,
        s_value: SomeValue,
        readonly: bool,
    ) {
        use crate::annotator::classdesc::Attribute;
        let mut attr = Attribute::new(name);
        attr.s_value = s_value;
        attr.readonly = readonly;
        classdef.borrow_mut().attrs.insert(name.to_string(), attr);
    }

    #[test]
    fn classrepr_setup_populates_clsfields_for_readonly_attrs() {
        use crate::annotator::model::SomeInteger;
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        attach_attr(
            &classdef,
            "LIMIT",
            SomeValue::Integer(SomeInteger::new(false, false)),
            true,
        );

        let repr = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        Repr::setup(repr.as_ref()).expect("setup");

        let clsfields = repr.clsfields();
        let (mangled, field_repr) = clsfields
            .get("LIMIT")
            .expect("readonly attr LIMIT should land in clsfields");
        assert_eq!(mangled, "cls_LIMIT");
        // rclass.py:262 — llfields append uses r.lowleveltype; Integer's
        // Repr lowleveltype is lltype.Signed.
        assert_eq!(field_repr.lowleveltype(), &LowLevelType::Signed);

        // rclass.py:275-278 — the vtable Struct body must carry the
        // `(mangled_name, r.lowleveltype)` pair after 'super'.
        let LowLevelType::ForwardReference(fwd) = repr.vtable_type() else {
            panic!();
        };
        let resolved = fwd.resolved().expect("vtable_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!();
        };
        let limit_field = body
            ._flds
            .get("cls_LIMIT")
            .expect("vtable must carry cls_LIMIT field");
        assert_eq!(*limit_field, LowLevelType::Signed);
    }

    #[test]
    fn classrepr_setup_skips_non_readonly_attrs() {
        use crate::annotator::model::SomeInteger;
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        attach_attr(
            &classdef,
            "x",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false, // instance attr, not class attr
        );

        let repr = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        Repr::setup(repr.as_ref()).expect("setup");

        assert!(
            repr.clsfields().is_empty(),
            "non-readonly attrs must not leak into clsfields"
        );
    }

    #[test]
    fn classrepr_sorts_vtable_fields_by_reverse_size_after_name_order() {
        use crate::annotator::model::{SomeBool, SomeInteger};

        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        attach_attr(&classdef, "a_bool", SomeValue::Bool(SomeBool::new()), true);
        attach_attr(
            &classdef,
            "z_int",
            SomeValue::Integer(SomeInteger::new(false, false)),
            true,
        );

        let repr = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        Repr::setup(repr.as_ref()).expect("setup");

        let LowLevelType::ForwardReference(fwd) = repr.vtable_type() else {
            panic!();
        };
        let resolved = fwd.resolved().expect("vtable_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!();
        };
        assert_eq!(body._names[0], "super");
        assert_eq!(body._names[1], "cls_z_int");
        assert_eq!(body._names[2], "cls_a_bool");
    }

    #[test]
    fn instance_repr_setup_populates_fields_for_non_readonly_attrs() {
        use crate::annotator::model::SomeInteger;
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        attach_attr(
            &classdef,
            "x",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
        );

        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(repr.as_ref()).expect("setup");

        let fields = repr.fields();
        let (mangled, field_repr) = fields.get("x").expect("instance attr x in fields");
        assert_eq!(mangled, "inst_x");
        assert_eq!(field_repr.lowleveltype(), &LowLevelType::Signed);

        // rclass.py:548-554 — object_type Struct body has inst_x pair.
        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!();
        };
        let resolved = fwd.resolved().expect("object_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!();
        };
        let x_field = body
            ._flds
            .get("inst_x")
            .expect("object_type must carry inst_x field");
        assert_eq!(*x_field, LowLevelType::Signed);
    }

    #[test]
    fn instance_repr_allinstancefields_unions_parent_chain() {
        use crate::annotator::model::SomeInteger;
        let rtyper = fresh_rtyper();
        let base = ClassDef::new_standalone("pkg.Base", None);
        attach_attr(
            &base,
            "shared",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
        );
        let child = ClassDef::new_standalone("pkg.Child", Some(&base));
        attach_attr(
            &child,
            "extra",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
        );

        let child_repr =
            getinstancerepr(&rtyper, Some(&child), Flavor::Gc).expect("getinstancerepr child");
        Repr::setup(child_repr.as_ref()).expect("child setup");

        let all = child_repr.allinstancefields();
        assert!(
            all.contains_key("shared"),
            "parent 'shared' attr must propagate into child.allinstancefields"
        );
        assert!(
            all.contains_key("extra"),
            "child's own 'extra' attr must appear in allinstancefields"
        );

        // Fields-proper stays minimal (only child's own).
        let fields = child_repr.fields();
        assert!(!fields.contains_key("shared"));
        assert!(fields.contains_key("extra"));
    }

    #[test]
    fn instance_repr_sorts_fields_by_reverse_size_after_name_order() {
        use crate::annotator::model::{SomeBool, SomeInteger};

        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        attach_attr(&classdef, "a_bool", SomeValue::Bool(SomeBool::new()), false);
        attach_attr(
            &classdef,
            "z_int",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
        );

        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(repr.as_ref()).expect("setup");

        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!();
        };
        let resolved = fwd.resolved().expect("object_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!();
        };
        assert_eq!(body._names[0], "super");
        assert_eq!(body._names[1], "inst_z_int");
        assert_eq!(body._names[2], "inst_a_bool");
    }

    // ---------------------------------------------------------------
    // R3 — `ClassRepr::setup_vtable` + constant→lowlevel adapter.
    // ---------------------------------------------------------------

    #[test]
    fn constant_to_lowlevel_value_unwraps_int_bool_float_ptr_and_void() {
        use crate::flowspace::model::Constant;
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelValue, MallocFlavor, malloc,
        };

        let signed = Constant::with_concretetype(ConstValue::Int(7), LowLevelType::Signed);
        assert_eq!(
            constant_to_lowlevel_value(&signed).unwrap(),
            LowLevelValue::Signed(7)
        );

        // rint.convert_const(Bool) → Int(b as i64); adapter routes through Signed.
        let signed_from_bool =
            Constant::with_concretetype(ConstValue::Bool(true), LowLevelType::Signed);
        assert_eq!(
            constant_to_lowlevel_value(&signed_from_bool).unwrap(),
            LowLevelValue::Signed(1)
        );

        let bool_const = Constant::with_concretetype(ConstValue::Bool(false), LowLevelType::Bool);
        assert_eq!(
            constant_to_lowlevel_value(&bool_const).unwrap(),
            LowLevelValue::Bool(false)
        );

        let float_const = Constant::with_concretetype(
            ConstValue::Float(0x4020_0000_0000_0000),
            LowLevelType::Float,
        );
        assert_eq!(
            constant_to_lowlevel_value(&float_const).unwrap(),
            LowLevelValue::Float(0x4020_0000_0000_0000)
        );

        let void_const = Constant::with_concretetype(ConstValue::None, LowLevelType::Void);
        assert_eq!(
            constant_to_lowlevel_value(&void_const).unwrap(),
            LowLevelValue::Void
        );

        // LLPtr case — forge a _ptr via malloc and verify the adapter returns
        // a Ptr variant carrying the exact same _ptr identity.
        let struct_t = LowLevelType::Struct(Box::new(StructType::gc(
            "pkg.P",
            vec![("x".into(), LowLevelType::Signed)],
        )));
        let ptr = malloc(struct_t.clone(), None, MallocFlavor::Gc, true).unwrap();
        let ptr_t = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(StructType::gc(
                "pkg.P",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        }));
        let ptr_const =
            Constant::with_concretetype(ConstValue::LLPtr(Box::new(ptr.clone())), ptr_t.clone());
        let llvalue = constant_to_lowlevel_value(&ptr_const).unwrap();
        match llvalue {
            LowLevelValue::Ptr(got) => assert_eq!(got._identity, ptr._identity),
            other => panic!("expected Ptr, got {other:?}"),
        }
    }

    #[test]
    fn constant_to_lowlevel_value_rejects_mismatched_shape() {
        use crate::flowspace::model::Constant;

        let bad = Constant::with_concretetype(ConstValue::Int(3), LowLevelType::Bool);
        let err = constant_to_lowlevel_value(&bad).unwrap_err().to_string();
        assert!(
            err.contains("unsupported ConstValue→LowLevelValue"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn setup_vtable_writes_readonly_int_attr_into_vtable_field() {
        use crate::annotator::classdesc::ClassDictEntry;
        use crate::annotator::model::SomeInteger;
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelValue, MallocFlavor, malloc,
        };

        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        // Seed readonly class attr on ClassDef so _setup_repr populates
        // clsfields with ("cls_LIMIT", IntegerRepr).
        attach_attr(
            &classdef,
            "LIMIT",
            SomeValue::Integer(SomeInteger::new(false, false)),
            true,
        );
        // Seed the classdesc's dictionary so read_attribute("LIMIT")
        // returns Constant(Int(7)) — the concrete value upstream resolves
        // via self.classdef.classdesc.read_attribute(fldname, None).
        {
            let cdesc = classdef.borrow().classdesc.clone();
            cdesc
                .borrow_mut()
                .classdict
                .insert("LIMIT".into(), ClassDictEntry::constant(ConstValue::Int(7)));
        }

        let repr_arc = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr") {
            ClassReprArc::Inst(r) => r,
            _ => panic!("classdef != None routes to Inst"),
        };
        Repr::setup(repr_arc.as_ref()).expect("_setup_repr");

        // Allocate the vtable container — malloc produces an immortal
        // gc-Struct _ptr whose type matches `repr.lowleveltype` after
        // ForwardReference resolution.
        let LowLevelType::ForwardReference(fwd) = repr_arc.vtable_type() else {
            panic!("vtable_type must be ForwardReference");
        };
        let resolved = fwd.resolved().expect("resolved vtable_type");
        let mut vtable = malloc(resolved, None, MallocFlavor::Gc, true).expect("malloc vtable");

        repr_arc
            .setup_vtable(&mut vtable, &repr_arc)
            .expect("setup_vtable");

        // rclass.py:321 — `setattr(vtable, mangled_name, llvalue)` is
        // observable via the struct's getattr.
        let value = vtable.getattr("cls_LIMIT").expect("cls_LIMIT stored");
        assert_eq!(value, LowLevelValue::Signed(7));
    }

    /// `init_vtable` walks the super-chain calling
    /// `self.setup_vtable(vtable_part, r_parentcls)` (rclass.py:302) —
    /// receiver is the leaf class, argument is the parent providing the
    /// field shape. The values therefore come from the leaf's classdesc
    /// (via `read_attribute` MRO walk), not from the parent's. A child
    /// that overrides a parent class attr must see the child's value at
    /// the parent's vtable slot.
    #[test]
    fn init_vtable_uses_leaf_classdesc_for_parent_level_field_values() {
        use crate::annotator::classdesc::ClassDictEntry;
        use crate::annotator::model::SomeInteger;
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelValue;

        let rtyper = fresh_rtyper();

        // Parent P: declares readonly class attr `LIMIT` (so it lands in
        // P.clsfields after _setup_repr) with value 7 in classdict.
        let parent = ClassDef::new_standalone("pkg.P", None);
        attach_attr(
            &parent,
            "LIMIT",
            SomeValue::Integer(SomeInteger::new(false, false)),
            true,
        );
        {
            let cdesc = parent.borrow().classdesc.clone();
            cdesc
                .borrow_mut()
                .classdict
                .insert("LIMIT".into(), ClassDictEntry::constant(ConstValue::Int(7)));
        }

        // `fill_vtable_root` reads `self.classdef.{minid,maxid}` —
        // assign_inheritance_ids would normally populate these. Provide
        // distinct ranges for parent/child so the test does not depend
        // on that pass.
        {
            let mut p = parent.borrow_mut();
            p.minid = Some(2);
            p.maxid = Some(9);
        }

        // Child C extends P with override `LIMIT = 42` in C's classdict.
        // C does NOT redeclare the attribute so its own clsfields stay
        // empty — the only `cls_LIMIT` slot lives at the parent (super)
        // level of the vtable struct.
        let child = ClassDef::new_standalone("pkg.C", Some(&parent));
        {
            let mut c = child.borrow_mut();
            c.minid = Some(3);
            c.maxid = Some(8);
        }
        {
            let cdesc = child.borrow().classdesc.clone();
            cdesc.borrow_mut().classdict.insert(
                "LIMIT".into(),
                ClassDictEntry::constant(ConstValue::Int(42)),
            );
        }

        // Pre-setup the parent repr so its rbase + clsfields are
        // populated before C's setup walks the chain.
        let parent_repr = match getclassrepr_arc(&rtyper, Some(&parent)).expect("getclassrepr P") {
            ClassReprArc::Inst(r) => r,
            _ => panic!("classdef != None routes to Inst"),
        };
        Repr::setup(parent_repr.as_ref()).expect("_setup_repr P");

        let child_repr = match getclassrepr_arc(&rtyper, Some(&child)).expect("getclassrepr C") {
            ClassReprArc::Inst(r) => r,
            _ => panic!("classdef != None routes to Inst"),
        };
        Repr::setup(child_repr.as_ref()).expect("_setup_repr C");

        // Drive `init_vtable` on the leaf (child). Without the
        // leaf-receiver fix the parent-level slot would be filled from
        // P's classdesc (value 7) instead of C's override (42).
        child_repr.init_vtable().expect("init_vtable C");

        let raw = child_repr
            .vtable
            .borrow()
            .clone()
            .expect("init_vtable post-condition: vtable is Some");
        // C_vtable.super → P_vtable; P_vtable carries `cls_LIMIT`.
        // `_ptr.getattr` exposes substructs as a `Ptr` aliasing the
        // parent allocation, so chain `.getattr` calls drill down.
        let parent_slice = raw.getattr("super").expect("super substruct");
        let LowLevelValue::Ptr(parent_ptr) = parent_slice else {
            panic!("expected substruct ptr, got {parent_slice:?}");
        };
        let limit = parent_ptr.getattr("cls_LIMIT").expect("cls_LIMIT");
        assert_eq!(
            limit,
            LowLevelValue::Signed(42),
            "leaf-class override must win — value should come from C.classdesc"
        );
    }
}
