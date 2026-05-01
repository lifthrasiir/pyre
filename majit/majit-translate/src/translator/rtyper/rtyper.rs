//! RPython `rpython/rtyper/rtyper.py` — `RPythonTyper`, `HighLevelOp`,
//! `LowLevelOpList`.
//!
//! Mirrors upstream module layout: all three types live in the same
//! file (`rtyper.py:47 RPythonTyper`, `rtyper.py:617 HighLevelOp`,
//! `rtyper.py:783 LowLevelOpList`) because the `specialize_block →
//! highlevelops → translate_hl_to_ll` dispatch loop ties them
//! together.
//!
//! Current scope ports the core `HighLevelOp` carrier methods
//! (`setup`, `copy`, `dispatch`, `inputarg`, `inputargs`,
//! `exception_is_here`, `exception_cannot_occur`, `r_s_pop`,
//! `v_s_insertfirstarg`, `swap_fst_snd_args`) and the `LowLevelOpList`
//! buffer methods (`append`, `convertvar`, `genop`) so concrete Repr
//! ports can plug into the same surface as upstream.
//!
//! Deferred methods are documented inline with their upstream line
//! reference.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use crate::annotator::annrpython::RPythonAnnotator;
use crate::annotator::bookkeeper::PositionKey;
use crate::annotator::model::{SomeObjectTrait, SomeValue};
use crate::flowspace::argument::Signature;
use crate::flowspace::model::{
    Block, BlockKey, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc,
    GraphKey, GraphRef, HOST_ENV, Hlvalue, Link, LinkRef, SpaceOperation, Variable,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::annlowlevel::{LowLevelAnnotatorPolicy, annotate_lowlevel_helper};
use crate::translator::rtyper::error::{TyperError, TyperWhere};
use crate::translator::rtyper::llannotation::lltype_to_annotation;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, LowLevelType, LowLevelValue, PtrTarget, getfunctionptr,
};
use crate::translator::rtyper::rclass::{
    CLASSTYPE, Flavor, InstanceRepr, InstanceReprKey, OBJECTPTR, RootClassRepr, getinstancerepr,
};
use crate::translator::rtyper::rmodel::{
    RTypeResult, Repr, ReprKey, inputconst, inputconst_from_lltype, rtyper_makekey, rtyper_makerepr,
};
use crate::translator::rtyper::rpbc::LLCallTable;
use crate::translator::unsimplify::insert_empty_block;

/// RPython `class RPythonTyper(object)` (rtyper.py:42+).
///
/// The full constructor state lands incrementally as the rtyper port
/// progresses. For `simplify.py` parity we need the annotator link and
/// the `already_seen` dict populated by `specialize_more_blocks()`.
/// RPython `class ExceptionData` (`rpython/rtyper/exceptiondata.py:11-26`).
///
/// Upstream's `__init__` obtains `r_type = rtyper.rootclass_repr` and
/// `r_instance = getinstancerepr(rtyper, None)`, then freezes their
/// lltypes into the last two fields. The Rust port exposes the same
/// four-field surface; populating it requires the `rootclass_repr` /
/// `getinstancerepr(rtyper, None)` port which has not landed, so until
/// then the `RPythonTyper.exceptiondata` slot stays `None` and callers
/// receive a structured `TyperError` pointing at that dependency.
#[derive(Clone, Debug)]
pub struct ExceptionData {
    /// RPython `self.r_exception_type = rtyper.rootclass_repr` — the
    /// class repr used for every exception vtable pointer.
    pub r_exception_type: Arc<dyn Repr>,
    /// RPython `self.r_exception_value = getinstancerepr(rtyper, None)`
    /// — the instance repr shared by every exception value.
    pub r_exception_value: Arc<dyn Repr>,
    /// RPython `self.lltype_of_exception_type = r_type.lowleveltype`.
    pub lltype_of_exception_type: LowLevelType,
    /// RPython `self.lltype_of_exception_value = r_instance.lowleveltype`.
    pub lltype_of_exception_value: LowLevelType,
    /// RPython `self.fn_exception_match` assigned by
    /// `ExceptionData.make_helpers()`.
    pub fn_exception_match: RefCell<Option<LowLevelFunction>>,
    /// RPython `self.fn_type_of_exc_inst` assigned by
    /// `ExceptionData.make_helpers()`.
    pub fn_type_of_exc_inst: RefCell<Option<LowLevelFunction>>,
}

impl ExceptionData {
    /// RPython `ExceptionData.make_helpers(self, rtyper)`
    /// (`exceptiondata.py:47-50`).
    pub fn make_helpers(&self, rtyper: &RPythonTyper) -> Result<(), TyperError> {
        *self.fn_exception_match.borrow_mut() = Some(self.make_exception_matcher(rtyper)?);
        *self.fn_type_of_exc_inst.borrow_mut() = Some(self.make_type_of_exc_inst(rtyper)?);
        Ok(())
    }

    /// RPython `ExceptionData.make_exception_matcher(self, rtyper)`
    /// (`exceptiondata.py:52-56`).
    fn make_exception_matcher(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<LowLevelFunction, TyperError> {
        rtyper.lowlevel_helper_function(
            "ll_issubclass",
            vec![
                self.lltype_of_exception_type.clone(),
                self.lltype_of_exception_type.clone(),
            ],
            LowLevelType::Bool,
        )
    }

    /// RPython `ExceptionData.make_type_of_exc_inst(self, rtyper)`
    /// (`exceptiondata.py:58-62`).
    fn make_type_of_exc_inst(&self, rtyper: &RPythonTyper) -> Result<LowLevelFunction, TyperError> {
        rtyper.lowlevel_helper_function(
            "ll_type",
            vec![self.lltype_of_exception_value.clone()],
            self.lltype_of_exception_type.clone(),
        )
    }

    /// RPython `ExceptionData.get_standard_ll_exc_instance(self, rtyper,
    /// clsdef)` (`exceptiondata.py:34-38`).
    ///
    /// ```python
    /// def get_standard_ll_exc_instance(self, rtyper, clsdef):
    ///     r_inst = getinstancerepr(rtyper, clsdef)
    ///     example = r_inst.get_reusable_prebuilt_instance()
    ///     example = ll_cast_to_object(example)
    ///     return example
    /// ```
    ///
    /// `ll_cast_to_object` upstream is `cast_pointer(OBJECTPTR, obj)`
    /// (rclass.py:1126-1127); the Rust port emits the static cast via
    /// [`lltype::cast_pointer`]. The call chain depends on the leaf
    /// `InstanceRepr` and its `ClassRepr` having had `_setup_repr` run
    /// (so the vtable_type/object_type ForwardReferences are
    /// resolved); upstream achieves this via the
    /// `RPythonTyper.specialize` pipeline.
    pub fn get_standard_ll_exc_instance(
        &self,
        rtyper: &RPythonTyper,
        clsdef: Option<Rc<RefCell<crate::annotator::classdesc::ClassDef>>>,
    ) -> Result<Constant, TyperError> {
        let rtyper_rc = rtyper.self_rc()?;
        // upstream: `r_inst = getinstancerepr(rtyper, clsdef)`.
        let r_inst = getinstancerepr(&rtyper_rc, clsdef.as_ref(), Flavor::Gc)?;
        // upstream: `example = r_inst.get_reusable_prebuilt_instance()`.
        let example = r_inst.get_reusable_prebuilt_instance()?;
        // upstream: `example = ll_cast_to_object(example)` =
        // `cast_pointer(OBJECTPTR, example)`.
        let LowLevelType::Ptr(objectptr) = OBJECTPTR.clone() else {
            return Err(TyperError::message(
                "ExceptionData.get_standard_ll_exc_instance: OBJECTPTR \
                 is not a Ptr — internal error",
            ));
        };
        let cast = crate::translator::rtyper::lltypesystem::lltype::cast_pointer(
            objectptr.as_ref(),
            &example,
        )
        .map_err(TyperError::message)?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(cast)),
            OBJECTPTR.clone(),
        ))
    }

    /// RPython `ExceptionData.get_standard_ll_exc_instance_by_class(self,
    /// exceptionclass)` (`exceptiondata.py:40-45`).
    ///
    /// ```python
    /// def get_standard_ll_exc_instance_by_class(self, exceptionclass):
    ///     if exceptionclass not in self.standardexceptions:
    ///         raise UnknownException(exceptionclass)
    ///     clsdef = self.rtyper.annotator.bookkeeper.getuniqueclassdef(
    ///         exceptionclass)
    ///     return self.get_standard_ll_exc_instance(self.rtyper, clsdef)
    /// ```
    ///
    /// Validates that the supplied `exceptionclass` (HostObject of a
    /// class) is one of the standard exceptions, then defers to
    /// [`Self::get_standard_ll_exc_instance`] for the actual instance
    /// materialisation. Membership matches upstream
    /// `exceptionclass in self.standardexceptions` — identity
    /// comparison on the live class object (Pyre `HostObject`'s
    /// `PartialEq` is `Arc::ptr_eq`).
    pub fn get_standard_ll_exc_instance_by_class(
        &self,
        rtyper: &RPythonTyper,
        exceptionclass: &crate::flowspace::model::HostObject,
    ) -> Result<Constant, TyperError> {
        // upstream: `if exceptionclass not in self.standardexceptions:
        //     raise UnknownException(exceptionclass)`.
        let known = crate::annotator::exception::standard_exception_classes()
            .into_iter()
            .any(|cls| &cls == exceptionclass);
        if !known {
            return Err(TyperError::message(format!(
                "ExceptionData.get_standard_ll_exc_instance_by_class: \
                 {} is not a standard exception (UnknownException)",
                exceptionclass.qualname()
            )));
        }
        // upstream: `clsdef = self.rtyper.annotator.bookkeeper.getuniqueclassdef(exceptionclass)`.
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message(
                "ExceptionData.get_standard_ll_exc_instance_by_class: annotator dropped",
            )
        })?;
        let classdefs =
            crate::annotator::exception::standard_exception_classdefs(&annotator.bookkeeper)
                .map_err(|e| TyperError::message(e.to_string()))?;
        let clsdef = classdefs
            .into_iter()
            .find(|cd| {
                let cd_borrow = cd.borrow();
                let cdesc = cd_borrow.classdesc.borrow();
                &cdesc.pyobj == exceptionclass
            })
            .ok_or_else(|| {
                TyperError::message(format!(
                    "ExceptionData.get_standard_ll_exc_instance_by_class: \
                     classdef for {} not in bookkeeper cache",
                    exceptionclass.qualname()
                ))
            })?;
        self.get_standard_ll_exc_instance(rtyper, Some(clsdef))
    }

    /// RPython `ExceptionData.generate_exception_match(self, oplist,
    /// var_etype, const_etype)` (`exceptiondata.py:64-80`).
    ///
    /// ```python
    /// def generate_exception_match(self, oplist, var_etype, const_etype):
    ///     # generate the content of rclass.ll_issubclass(_const)
    ///     llops = LowLevelOpList(None)
    ///     field = llops.genop(
    ///         'getfield',
    ///         [var_etype, llops.genvoidconst('subclassrange_min')],
    ///         lltype.Signed)
    ///     res = llops.genop(
    ///         'int_between', [
    ///             llops.genconst(const_etype.value.subclassrange_min),
    ///             field,
    ///             llops.genconst(const_etype.value.subclassrange_max),
    ///         ],
    ///         lltype.Bool)
    ///     oplist.extend(llops)
    ///     return res
    /// ```
    ///
    /// `oplist` upstream is a Python `block.operations` list which the
    /// helper extends in place. The Rust port mirrors that contract by
    /// taking `&mut Vec<SpaceOperation>` so callers (notably
    /// `backendopt/inline.py:366` and the ExceptionData consumers in
    /// rtyper.py exception branches) can extend the live block buffer
    /// without reaching into a `LowLevelOpList`. The fresh
    /// `LowLevelOpList(None)` mirrors upstream — no parent graph is
    /// recorded for the helper ops, so `record_extra_call` stays inert.
    pub fn generate_exception_match(
        &self,
        rtyper: Rc<RPythonTyper>,
        oplist: &mut Vec<SpaceOperation>,
        var_etype: Hlvalue,
        const_etype: Hlvalue,
    ) -> Result<Hlvalue, TyperError> {
        // upstream `const_etype.value.subclassrange_{min,max}` —
        // `const_etype` is a `Constant(_ptr, Ptr(OBJECT_VTABLE))` and
        // `.value` is the underlying `_ptr` to a vtable struct.
        let Hlvalue::Constant(const_etype_c) = &const_etype else {
            return Err(TyperError::message(
                "generate_exception_match: const_etype must be a Constant",
            ));
        };
        let ConstValue::LLPtr(c_ptr) = &const_etype_c.value else {
            return Err(TyperError::message(format!(
                "generate_exception_match: const_etype.value must be _ptr, got {:?}",
                const_etype_c.value,
            )));
        };
        let min_val = c_ptr
            .getattr("subclassrange_min")
            .map_err(TyperError::message)?;
        let max_val = c_ptr
            .getattr("subclassrange_max")
            .map_err(TyperError::message)?;
        let LowLevelValue::Signed(min_n) = min_val else {
            return Err(TyperError::message(format!(
                "generate_exception_match: subclassrange_min must be Signed, got {min_val:?}",
            )));
        };
        let LowLevelValue::Signed(max_n) = max_val else {
            return Err(TyperError::message(format!(
                "generate_exception_match: subclassrange_max must be Signed, got {max_val:?}",
            )));
        };

        // upstream `LowLevelOpList(None)` — a temporary buffer with no
        // originalblock, accumulating helper ops before the
        // `oplist.extend(llops)` flush.
        let mut llops = LowLevelOpList::new(rtyper, None);

        // upstream `field = llops.genop('getfield',
        //     [var_etype, llops.genvoidconst('subclassrange_min')], Signed)`.
        let void_field = Constant::with_concretetype(
            ConstValue::byte_str("subclassrange_min"),
            LowLevelType::Void,
        );
        let v_field = llops
            .genop(
                "getfield",
                vec![var_etype, Hlvalue::Constant(void_field)],
                GenopResult::LLType(LowLevelType::Signed),
            )
            .expect("getfield with Signed result returns a value");

        // upstream `res = llops.genop('int_between', [genconst(min),
        // field, genconst(max)], Bool)`.
        let c_min = Constant::with_concretetype(ConstValue::Int(min_n), LowLevelType::Signed);
        let c_max = Constant::with_concretetype(ConstValue::Int(max_n), LowLevelType::Signed);
        let v_res = llops
            .genop(
                "int_between",
                vec![
                    Hlvalue::Constant(c_min),
                    Hlvalue::Variable(v_field),
                    Hlvalue::Constant(c_max),
                ],
                GenopResult::LLType(LowLevelType::Bool),
            )
            .expect("int_between with Bool result returns a value");

        // upstream `oplist.extend(llops)` — flush helper ops onto the
        // caller-provided block.operations buffer.
        oplist.extend(llops.ops);
        Ok(Hlvalue::Variable(v_res))
    }
}

/// RPython `rtyper.pbc_reprs` dict key (`rpbc.py:621-630`). Upstream
/// uses the Python object `'unrelated'` string sentinel for
/// `MultipleUnrelatedFrozenPBCRepr` and the `access_set` identity for
/// `MultipleFrozenPBCRepr`. Pyre collapses both into this enum — the
/// `Access` variant keys on `FrozenAttrFamily` pointer identity (via
/// `Rc::as_ptr as usize`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PbcReprKey {
    /// `rtyper.pbc_reprs['unrelated']` (rpbc.py:621).
    Unrelated,
    /// `rtyper.pbc_reprs[access_set]` (rpbc.py:627) — keyed by the
    /// `FrozenAttrFamily` pointer identity. Reserved for
    /// MultipleFrozenPBCRepr caching when that repr lands; currently
    /// unused.
    Access(usize),
}

pub struct RPythonTyper {
    /// RPython `self.annotator`.
    ///
    /// Rust stores this edge weakly to avoid the
    /// `annotator -> translator -> rtyper -> annotator` cycle that
    /// Python's GC handles upstream.
    pub annotator: Weak<RPythonAnnotator>,
    /// RPython `self.rootclass_repr = RootClassRepr(self)` assigned at
    /// `__init__` line 57 (rtyper.py:57). The `.setup()` call at
    /// `__init__` line 58 is replayed inside
    /// [`RPythonTyper::initialize_exceptiondata`].
    ///
    /// `RefCell<Option<_>>` instead of eagerly initialising in
    /// [`RPythonTyper::new`] is a PRE-EXISTING-ADAPTATION (option 1A
    /// from the porting plan): `RootClassRepr::new` does not need
    /// `self`, but `ExceptionData::new` consumes the populated
    /// `rootclass_repr` at rtyper.py:71, and that write lands after
    /// the rtyper is wrapped in `Rc<Self>` so we cannot inline it into
    /// `new()`. The `__init__` invariant that `rootclass_repr` is
    /// `Some` on completion is still honoured by callers invoking
    /// `initialize_exceptiondata` once at construction-time.
    pub rootclass_repr: RefCell<Option<Arc<RootClassRepr>>>,
    /// RPython `self.instance_reprs = {}` assigned at `__init__` line 59
    /// (rtyper.py:59). `None` classdef mirrors upstream Python's ability
    /// to use `None` as a dict key (option 3A from the porting plan).
    pub instance_reprs: RefCell<HashMap<InstanceReprKey, Arc<InstanceRepr>>>,
    /// RPython `self.exceptiondata = ExceptionData(self)` assigned in
    /// `__init__` (rtyper.py:71). Stays `None` until
    /// [`RPythonTyper::initialize_exceptiondata`] runs.
    pub exceptiondata: RefCell<Option<Rc<ExceptionData>>>,
    /// Self-weak backref populated by
    /// [`RPythonTyper::initialize_exceptiondata`]. Exists so repr
    /// dispatchers that only receive `&RPythonTyper` (notably
    /// [`rmodel::rtyper_makerepr`] arms that route to
    /// [`rclass::getinstancerepr`]) can upgrade back into
    /// `Rc<RPythonTyper>` without plumbing a new parameter through
    /// every `getrepr` call site. PRE-EXISTING-ADAPTATION; upstream
    /// Python sidesteps this by storing `rtyper` on every Repr's
    /// `self.rtyper` directly.
    self_weak: RefCell<Weak<Self>>,
    /// RPython `self.already_seen = {}` assigned in `specialize()`
    /// (rtyper.py:186). Membership is queried by `simplify.py`.
    pub already_seen: RefCell<HashMap<BlockKey, bool>>,
    /// RPython `self.concrete_calltables = {}` assigned in `__init__`
    /// (rtyper.py:57).
    pub concrete_calltables: RefCell<HashMap<usize, (LLCallTable, usize)>>,
    /// RPython `self.reprs = {}` (`rtyper.py:54`) — cache keyed by
    /// `s_obj.rtyper_makekey()`. Pyre stores `Option<Arc<dyn Repr>>`
    /// because upstream pre-inserts `None` before calling
    /// `rtyper_makerepr` to detect recursive `getrepr()` (rtyper.py:156).
    pub reprs: RefCell<HashMap<ReprKey, Option<Arc<dyn Repr>>>>,
    /// RPython `self._reprs_must_call_setup = []` (`rtyper.py:55`).
    pub reprs_must_call_setup: RefCell<Vec<Arc<dyn Repr>>>,
    /// RPython `self._seen_reprs_must_call_setup = {}` (`rtyper.py:56`).
    ///
    /// Pyre stores pointer-identity fingerprints so Arc-equal Reprs
    /// are deduped without requiring `Hash + Eq` on the trait object.
    pub seen_reprs_must_call_setup: RefCell<Vec<*const ()>>,
    /// RPython `self.primitive_to_repr = {}` (`rtyper.py:53`).
    pub primitive_to_repr: RefCell<HashMap<LowLevelType, Arc<dyn Repr>>>,
    /// RPython `self.pbc_reprs = {}` (`rtyper.py:58`, populated by
    /// `rpbc.getFrozenPBCRepr` at rpbc.py:621-630). Keyed by
    /// [`PbcReprKey`] — singleton cache for
    /// [`crate::translator::rtyper::rpbc::MultipleUnrelatedFrozenPBCRepr`]
    /// (and future `MultipleFrozenPBCRepr`).
    pub pbc_reprs: RefCell<HashMap<PbcReprKey, Arc<dyn Repr>>>,
    /// RPython `self.annmixlevel = None` (rtyper.py:204) lazily filled
    /// by [`RPythonTyper::getannmixlevel`] (rtyper.py:191-196). Reset to
    /// `None` at the start of every `specialize_more_blocks()` pass so
    /// helper-annotator state from a prior pass is not reused.
    pub annmixlevel:
        RefCell<Option<Rc<crate::translator::rtyper::annlowlevel::MixLevelHelperAnnotator>>>,
    /// RPython low-level helper graphs are cached by
    /// `FunctionDesc.cachedgraph()` under `annlowlevel.py`'s
    /// `LowLevelAnnotatorPolicy.lowlevelspecialize`. The Rust port has
    /// no host Python helper function object to hang that cache from, so
    /// the rtyper owns the equivalent graph cache for `ll_*` helpers.
    lowlevel_helper_graphs: RefCell<HashMap<LowLevelHelperKey, Rc<PyGraph>>>,
    /// RPython `self.lowlevel_ann_policy = LowLevelAnnotatorPolicy(self)`
    /// (`rtyper.py:53`). Populated by
    /// [`RPythonTyper::initialize_exceptiondata`] after `self_weak` is
    /// set so the policy's `Weak<RPythonTyper>` backref is non-empty,
    /// matching upstream's `LowLevelAnnotatorPolicy(self)` argument.
    /// Cached on the typer because `annotate_helper` (`rtyper.py:600`)
    /// passes the same instance on every call.
    pub lowlevel_ann_policy: RefCell<Option<LowLevelAnnotatorPolicy>>,
}

/// RPython rtyper.py:456-458 constant-result agreement predicate.
///
/// Upstream `assert` passes when `resultvar.value == hop.s_result.const
/// or (math.isnan(resultvar.value) and math.isnan(hop.s_result.const))`.
/// Extracted as a testable helper because the assert lives deep inside
/// `translate_hl_to_ll` after dispatch.
pub fn constant_result_values_agree(rv: &ConstValue, s_const: &ConstValue) -> bool {
    if rv == s_const {
        return true;
    }
    match (rv, s_const) {
        (ConstValue::Float(a), ConstValue::Float(b)) => {
            f64::from_bits(*a).is_nan() && f64::from_bits(*b).is_nan()
        }
        _ => false,
    }
}

/// RPython `RPythonTyper.annotate_helper(..., argtypes)` accepts a
/// mixed list: each item is either already an `annmodel.SomeObject` or
/// a low-level type that must pass through `lltype_to_annotation`.
/// Rust carries that Python union explicitly.
#[derive(Clone, Debug)]
pub enum AnnotateHelperArg {
    Annotation(SomeValue),
    LlType(LowLevelType),
}

impl From<SomeValue> for AnnotateHelperArg {
    fn from(value: SomeValue) -> Self {
        AnnotateHelperArg::Annotation(value)
    }
}

impl From<LowLevelType> for AnnotateHelperArg {
    fn from(value: LowLevelType) -> Self {
        AnnotateHelperArg::LlType(value)
    }
}

fn annotate_helper_arg_to_annotation(s: AnnotateHelperArg) -> SomeValue {
    match s {
        AnnotateHelperArg::Annotation(s) => s,
        AnnotateHelperArg::LlType(t) => lltype_to_annotation(t),
    }
}

impl RPythonTyper {
    /// RPython `RPythonTyper.__init__(self, annotator, backend=...)`.
    ///
    /// Only the fields required by current simplify parity are seeded
    /// here; additional constructor state lands with the full rtyper
    /// port.
    pub fn new(annotator: &Rc<RPythonAnnotator>) -> Self {
        RPythonTyper {
            annotator: Rc::downgrade(annotator),
            rootclass_repr: RefCell::new(None),
            instance_reprs: RefCell::new(HashMap::new()),
            exceptiondata: RefCell::new(None),
            self_weak: RefCell::new(Weak::new()),
            already_seen: RefCell::new(HashMap::new()),
            concrete_calltables: RefCell::new(HashMap::new()),
            reprs: RefCell::new(HashMap::new()),
            reprs_must_call_setup: RefCell::new(Vec::new()),
            seen_reprs_must_call_setup: RefCell::new(Vec::new()),
            primitive_to_repr: RefCell::new(HashMap::new()),
            pbc_reprs: RefCell::new(HashMap::new()),
            annmixlevel: RefCell::new(None),
            lowlevel_helper_graphs: RefCell::new(HashMap::new()),
            // RPython `self.lowlevel_ann_policy = LowLevelAnnotatorPolicy(self)`
            // (`rtyper.py:53`) cannot run here because the typer is not
            // yet wrapped in `Rc`. Populated by
            // `initialize_exceptiondata` once `self_weak` is set.
            lowlevel_ann_policy: RefCell::new(None),
        }
    }

    /// RPython inlined segment of `RPythonTyper.__init__` (rtyper.py:57-58,71):
    ///
    /// ```python
    /// self.rootclass_repr = RootClassRepr(self)
    /// self.rootclass_repr.setup()
    /// ...
    /// self.exceptiondata = ExceptionData(self)
    /// ```
    ///
    /// Split off from [`RPythonTyper::new`] because `ExceptionData::new`
    /// consumes the populated `self.rootclass_repr` / `self.instance_reprs`
    /// state — that read has to happen after the typer is already
    /// reachable via `Rc`, which `new()` cannot observe. The dual-phase
    /// shape is a PRE-EXISTING-ADAPTATION documented on the field
    /// declarations; callers must invoke this exactly once after
    /// `new()` so the `__init__`-complete invariant (`rootclass_repr`
    /// and `exceptiondata` are both `Some`) is restored.
    pub fn initialize_exceptiondata(self: &Rc<Self>) -> Result<(), TyperError> {
        // Populate the self-weak backref first so any downstream
        // `rtyper_makerepr` arm that needs `Rc<Self>` (e.g.
        // `SomeInstance.rtyper_makerepr -> getinstancerepr`) sees a
        // live handle even before `rootclass_repr` is installed.
        *self.self_weak.borrow_mut() = Rc::downgrade(self);
        // rtyper.py:53 — `self.lowlevel_ann_policy =
        // LowLevelAnnotatorPolicy(self)`. The policy stores a
        // `Weak<RPythonTyper>` so it is safe to construct now that
        // `self_weak` is populated and to reuse on every
        // `annotate_helper` call (`rtyper.py:600`).
        *self.lowlevel_ann_policy.borrow_mut() = Some(LowLevelAnnotatorPolicy::new(Some(self)));
        // rtyper.py:57 — `self.rootclass_repr = RootClassRepr(self)`.
        let root = Arc::new(RootClassRepr::new(self));
        // rtyper.py:58 — `self.rootclass_repr.setup()`.
        Repr::setup(root.as_ref())?;
        *self.rootclass_repr.borrow_mut() = Some(root.clone());
        // rtyper.py:71 — `self.exceptiondata = ExceptionData(self)`.
        //
        // Inlining exceptiondata.py:17-26 because `ExceptionData::new`
        // upstream takes `rtyper` and directly reads back the state
        // this method just installed. We keep that single-shot body
        // here rather than add a `Rc<RPythonTyper>` ctor parameter to
        // `ExceptionData`.
        let r_type: Arc<dyn Repr> = root.clone();
        let r_instance = getinstancerepr(self, None, Flavor::Gc)?;
        // exceptiondata.py:20-21 — `r_type.setup(); r_instance.setup()`.
        Repr::setup(r_type.as_ref())?;
        Repr::setup(r_instance.as_ref() as &dyn Repr)?;
        let r_instance_as_repr: Arc<dyn Repr> = r_instance.clone();
        let lltype_of_exception_type = r_type.lowleveltype().clone();
        let lltype_of_exception_value = r_instance_as_repr.lowleveltype().clone();
        let ed = ExceptionData {
            r_exception_type: r_type,
            r_exception_value: r_instance_as_repr,
            lltype_of_exception_type,
            lltype_of_exception_value,
            fn_exception_match: RefCell::new(None),
            fn_type_of_exc_inst: RefCell::new(None),
        };
        *self.exceptiondata.borrow_mut() = Some(Rc::new(ed));
        Ok(())
    }

    /// RPython `RPythonTyper.getconfig(self)` (rtyper.py).
    ///
    /// ```python
    /// def getconfig(self):
    ///     return self.annotator.translator.config
    /// ```
    ///
    /// Returns `None` when the weak-ref to the annotator cannot be
    /// upgraded (e.g. in unit tests that drop the annotator after
    /// constructing the rtyper). Call sites that treat a missing
    /// config as "use defaults" should `.unwrap_or_default()` or
    /// mirror upstream's option defaults explicitly.
    pub fn getconfig(&self) -> Option<crate::translator::translator::TranslationConfig> {
        self.annotator
            .upgrade()
            .map(|ann| ann.translator.config.clone())
    }

    /// Upgrade the stored self-weak (populated by
    /// [`RPythonTyper::initialize_exceptiondata`]) back into
    /// `Rc<Self>`. Surfaces a structured `TyperError` when the typer
    /// was never wrapped in `Rc` or `initialize_exceptiondata` never
    /// ran — both paths violate the invariant that R5 repr dispatch
    /// requires.
    pub fn self_rc(&self) -> Result<Rc<Self>, TyperError> {
        self.self_weak.borrow().upgrade().ok_or_else(|| {
            TyperError::message(
                "RPythonTyper self-weak not set — call \
                 initialize_exceptiondata() on an Rc<RPythonTyper> \
                 before dispatching SomeInstance / SomeException / \
                 SomeType reprs",
            )
        })
    }

    /// Read access to the `ExceptionData` surface. Returns a structured
    /// `TyperError` when [`RPythonTyper::initialize_exceptiondata`] has
    /// not yet been invoked for this typer instance.
    pub fn exceptiondata(&self) -> Result<Rc<ExceptionData>, TyperError> {
        self.exceptiondata.borrow().clone().ok_or_else(|| {
            TyperError::message(
                "ExceptionData not initialised — call \
                 RPythonTyper::initialize_exceptiondata() after construction",
            )
        })
    }

    /// RPython `ExceptionData.finish(rtyper)`
    /// (`rpython/rtyper/exceptiondata.py:28-32`):
    ///
    /// ```python
    /// def finish(self, rtyper):
    ///     bk = rtyper.annotator.bookkeeper
    ///     for cls in self.standardexceptions:
    ///         classdef = bk.getuniqueclassdef(cls)
    ///         getclassrepr(rtyper, classdef).setup()
    /// ```
    ///
    pub fn finish_exceptiondata(self: &Rc<Self>) -> Result<(), TyperError> {
        let _ = self.exceptiondata()?;
        let annotator = self
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("RPythonTyper.annotator weak reference dropped"))?;
        let classdefs =
            crate::annotator::exception::standard_exception_classdefs(&annotator.bookkeeper)
                .map_err(|err| TyperError::message(err.to_string()))?;
        for classdef in classdefs {
            let repr = crate::translator::rtyper::rclass::getclassrepr(self, Some(&classdef))?;
            Repr::setup(repr.as_ref())?;
        }
        Ok(())
    }

    /// RPython `RPythonTyper.getprimitiverepr(self, lltype)`
    /// (`rtyper.py:85-93`).
    pub fn getprimitiverepr(&self, lltype: &LowLevelType) -> Result<Arc<dyn Repr>, TyperError> {
        if let Some(repr) = self.primitive_to_repr.borrow().get(lltype) {
            return Ok(repr.clone());
        }
        if !is_primitive_lowleveltype(lltype) {
            return Err(TyperError::message(format!(
                "There is no primitive repr for {:?}",
                lltype
            )));
        }
        let repr = self.getrepr(&lltype_to_annotation(lltype.clone()))?;
        self.primitive_to_repr
            .borrow_mut()
            .insert(lltype.clone(), repr.clone());
        Ok(repr)
    }

    /// Test/debug helper mirroring `self.already_seen[block] = True`
    /// in `specialize_more_blocks()` (rtyper.py:225).
    pub fn mark_already_seen(&self, block: &BlockRef) {
        self.already_seen
            .borrow_mut()
            .insert(BlockKey::of(block), true);
    }

    /// RPython `RPythonTyper.getcallable(self, graph)` (rtyper.py:569-584).
    ///
    /// ```python
    /// def getcallable(self, graph):
    ///     def getconcretetype(v):
    ///         return self.bindingrepr(v).lowleveltype
    ///     if self.annotator.translator.config.translation.sandbox:
    ///         try:
    ///             name = graph.func._sandbox_external_name
    ///         except AttributeError:
    ///             pass
    ///         else:
    ///             args_s = [v.annotation for v in graph.getargs()]
    ///             s_result = graph.getreturnvar().annotation
    ///             sandboxed = make_sandbox_trampoline(name, args_s, s_result)
    ///             return self.getannmixlevel().delayedfunction(
    ///                     sandboxed, args_s, s_result)
    ///
    ///     return getfunctionptr(graph, getconcretetype)
    /// ```
    pub fn getcallable(&self, graph: &Rc<PyGraph>) -> Result<_ptr, TyperError> {
        // rtyper.py:572 — `if self.annotator.translator.config.translation.sandbox:`.
        let sandbox = self
            .annotator
            .upgrade()
            .map(|ann| ann.translator.config.translation.sandbox)
            .unwrap_or(false);
        if sandbox {
            // rtyper.py:573-575 — `try: name = graph.func._sandbox_external_name
            // except AttributeError: pass`.
            let sandbox_name = graph.func._sandbox_external_name.clone();
            if let Some(name) = sandbox_name {
                // rtyper.py:577-582 — sandbox trampoline path.
                // `make_sandbox_trampoline` lives in `rsandbox.py`
                // and is currently unported (see
                // `annotator/policy.rs:42` — same gate). Surface a
                // structured TaskError-equivalent so the missing
                // dep is visible at the right call site.
                return Err(TyperError::message(format!(
                    "rtyper.py:577-582 getcallable sandbox path: \
                     make_sandbox_trampoline({name}) is unported \
                     (rsandbox.py). Convergence path: port \
                     rsandbox.make_sandbox_trampoline alongside \
                     LowLevelAnnotatorPolicy sandbox_external lookup; \
                     then route through self.getannmixlevel().delayedfunction(...)."
                )));
            }
        }
        // rtyper.py:584 — `return getfunctionptr(graph, getconcretetype)`.
        getfunctionptr(&graph.graph, |v| {
            self.bindingrepr(v).map(|r| r.lowleveltype().clone())
        })
    }

    /// RPython `RPythonTyper.annotate_helper(self, ll_function, argtypes)`
    /// (`rtyper.py:586-603`).
    pub fn annotate_helper(
        &self,
        ll_function: &crate::flowspace::model::HostObject,
        argtypes: Vec<AnnotateHelperArg>,
    ) -> Result<Rc<PyGraph>, TyperError> {
        let annotator = self
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("RPythonTyper.annotator weak reference dropped"))?;
        let mut args_s = Vec::new();
        for s in argtypes {
            // rtyper.py:590-594:
            //   assume 's' is a low-level type, unless it is already
            //   an annotation.
            args_s.push(annotate_helper_arg_to_annotation(s));
        }
        let mut function = ll_function;

        // rtyper.py:595-599 — bound method hack:
        // prepend `im_self`'s immutable annotation and annotate `im_func`.
        if ll_function.is_bound_method() {
            let self_obj = ll_function.bound_method_self().ok_or_else(|| {
                TyperError::message("RPythonTyper.annotate_helper: bound method missing im_self")
            })?;
            let s_self = annotator
                .bookkeeper
                .immutablevalue(&ConstValue::HostObject(self_obj.clone()))
                .map_err(|e| TyperError::message(e.to_string()))?;
            args_s.insert(0, s_self);
            function = ll_function.bound_method_func().ok_or_else(|| {
                TyperError::message("RPythonTyper.annotate_helper: bound method missing im_func")
            })?;
        }

        // rtyper.py:600 — `policy=self.lowlevel_ann_policy`.
        // The same instance is reused on every call; `__init__`
        // (`rtyper.py:53`) stores it once. Surface the missing-init
        // case as a structured `TyperError` to mirror
        // [`RPythonTyper::self_rc`].
        let policy = self.lowlevel_ann_policy.borrow().clone().ok_or_else(|| {
            TyperError::message(
                "RPythonTyper.lowlevel_ann_policy not set — call \
                 initialize_exceptiondata() on an Rc<RPythonTyper> \
                 before invoking annotate_helper",
            )
        })?;
        annotate_lowlevel_helper(&annotator, function, args_s, Some(policy))
    }

    /// RPython `RPythonTyper.annotate_helper_fn(self, ll_function,
    /// argtypes)` (`rtyper.py:605-610`).
    pub fn annotate_helper_fn(
        &self,
        ll_function: &crate::flowspace::model::HostObject,
        argtypes: Vec<AnnotateHelperArg>,
    ) -> Result<_ptr, TyperError> {
        let graph = self.annotate_helper(ll_function, argtypes)?;
        self.getcallable(&graph)
    }

    /// RPython `annlowlevel.annotate_lowlevel_helper()` +
    /// `FunctionDesc.cachedgraph()` for low-level helper functions used
    /// by `LowLevelOpList.gendirectcall`.
    pub fn lowlevel_helper_function(
        &self,
        name: impl Into<String>,
        args: Vec<LowLevelType>,
        result: LowLevelType,
    ) -> Result<LowLevelFunction, TyperError> {
        let name = name.into();
        let name_for_builder = name.clone();
        self.lowlevel_helper_function_with_builder(
            name,
            args,
            result,
            move |rtyper, args, result| {
                lowlevel_helper_graph(rtyper, &name_for_builder, args, result)
            },
        )
    }

    /// RPython `annlowlevel.annotate_lowlevel_helper()` for synthesized
    /// helper functions whose body is built dynamically per call shape.
    ///
    /// Upstream (e.g. `rtuple.gen_eq_function`) builds a Python source
    /// helper closure per shape, hands it to
    /// `annlowlevel.annotate_lowlevel_helper`, and `cachedgraph` then
    /// produces the FunctionGraph. Pyre cannot annotate Python source
    /// at this stage of the port — instead the caller passes a
    /// `builder` closure that emits a `PyGraph` directly. The closure
    /// is invoked **only on cache miss**; the produced graph is
    /// memoised under `(name, args, result)` and registered with the
    /// translator's graph list (matching the bookkeeping path of the
    /// hardcoded `lowlevel_helper_function`).
    ///
    /// Callers must encode the shape into `name` to avoid collisions —
    /// e.g. `gen_eq_function([r_int, r_int])` uses
    /// `"ll_tuple_eq_signed_signed"`.
    pub fn lowlevel_helper_function_with_builder<F>(
        &self,
        name: impl Into<String>,
        args: Vec<LowLevelType>,
        result: LowLevelType,
        builder: F,
    ) -> Result<LowLevelFunction, TyperError>
    where
        F: FnOnce(&RPythonTyper, &[LowLevelType], &LowLevelType) -> Result<PyGraph, TyperError>,
    {
        let name = name.into();
        let key = LowLevelHelperKey {
            name: name.clone(),
            args: args.clone(),
            result: result.clone(),
        };
        if let Some(graph) = self.lowlevel_helper_graphs.borrow().get(&key).cloned() {
            return Ok(LowLevelFunction::from_pygraph(name, args, result, graph));
        }

        let graph = Rc::new(builder(self, &args, &result)?);
        self.lowlevel_helper_graphs
            .borrow_mut()
            .insert(key, graph.clone());

        let annotator = self
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("RPythonTyper.annotator weak reference dropped"))?;
        annotator
            .translator
            .graphs
            .borrow_mut()
            .push(graph.graph.clone());
        Ok(LowLevelFunction::from_pygraph(name, args, result, graph))
    }

    /// RPython `RPythonTyper.annotation(self, var)` (rtyper.py:166-168).
    ///
    /// ```python
    /// def annotation(self, var):
    ///     s_obj = self.annotator.annotation(var)
    ///     return s_obj
    /// ```
    pub fn annotation(&self, var: &Hlvalue) -> Option<SomeValue> {
        self.annotator.upgrade().and_then(|ann| ann.annotation(var))
    }

    /// RPython `RPythonTyper.binding(self, var)` (rtyper.py:170-172).
    ///
    /// ```python
    /// def binding(self, var):
    ///     s_obj = self.annotator.binding(var)
    ///     return s_obj
    /// ```
    ///
    pub fn binding(&self, var: &Hlvalue) -> SomeValue {
        let ann = self
            .annotator
            .upgrade()
            .expect("RPythonTyper.annotator weak reference dropped");
        ann.binding(var)
    }

    /// RPython `RPythonTyper.add_pendingsetup(self, repr)`
    /// (rtyper.py:105-111).
    ///
    /// ```python
    /// def add_pendingsetup(self, repr):
    ///     assert isinstance(repr, Repr)
    ///     if repr in self._seen_reprs_must_call_setup:
    ///         return
    ///     self._reprs_must_call_setup.append(repr)
    ///     self._seen_reprs_must_call_setup[repr] = True
    /// ```
    pub fn add_pendingsetup(&self, repr: Arc<dyn Repr>) {
        let fingerprint = Arc::as_ptr(&repr) as *const ();
        let mut seen = self.seen_reprs_must_call_setup.borrow_mut();
        if seen.contains(&fingerprint) {
            return;
        }
        seen.push(fingerprint);
        self.reprs_must_call_setup.borrow_mut().push(repr);
    }

    /// RPython `RPythonTyper.getrepr(self, s_obj)` (rtyper.py:149-164).
    ///
    /// ```python
    /// def getrepr(self, s_obj):
    ///     key = s_obj.rtyper_makekey()
    ///     assert key[0] is s_obj.__class__
    ///     try:
    ///         result = self.reprs[key]
    ///     except KeyError:
    ///         self.reprs[key] = None
    ///         result = s_obj.rtyper_makerepr(self)
    ///         assert not isinstance(result.lowleveltype, ContainerType), ...
    ///         self.reprs[key] = result
    ///         self.add_pendingsetup(result)
    ///     assert result is not None     # recursive getrepr()!
    ///     return result
    /// ```
    pub fn getrepr(&self, s_obj: &SomeValue) -> Result<Arc<dyn Repr>, TyperError> {
        let key = rtyper_makekey(s_obj);
        // Fast-path hit: entry present and Some → return clone.
        if let Some(slot) = self.reprs.borrow().get(&key) {
            let result = slot
                .as_ref()
                .expect("recursive getrepr() should not observe None sentinel");
            return Ok(result.clone());
        }
        // First time seeing this key: upstream pre-inserts None as the
        // recursion sentinel, then materialises the Repr.
        self.reprs.borrow_mut().insert(key.clone(), None);
        let result = rtyper_makerepr(s_obj, self)?;
        self.reprs.borrow_mut().insert(key, Some(result.clone()));
        self.add_pendingsetup(result.clone());
        Ok(result)
    }

    /// RPython `RPythonTyper.bindingrepr(self, var)` (rtyper.py:174-175).
    ///
    /// ```python
    /// def bindingrepr(self, var):
    ///     return self.getrepr(self.binding(var))
    /// ```
    pub fn bindingrepr(&self, var: &Hlvalue) -> Result<Arc<dyn Repr>, TyperError> {
        let s_obj = self.binding(var);
        self.getrepr(&s_obj)
    }

    /// RPython `RPythonTyper.setconcretetype(self, v)` (rtyper.py:258-260).
    ///
    /// ```python
    /// def setconcretetype(self, v):
    ///     assert isinstance(v, Variable)
    ///     v.concretetype = self.bindingrepr(v).lowleveltype
    /// ```
    ///
    /// The upstream `isinstance(v, Variable)` assertion is enforced by
    /// the `&Variable` signature. Mutation propagates through
    /// `Variable.concretetype`'s `Rc<RefCell>` to every clone of the
    /// same Variable identity.
    pub fn setconcretetype(&self, var: &Variable) -> Result<(), TyperError> {
        let repr = self.bindingrepr(&Hlvalue::Variable(var.clone()))?;
        var.set_concretetype(Some(repr.lowleveltype().clone()));
        Ok(())
    }

    /// RPython `RPythonTyper.make_new_lloplist(self, block)`
    /// (rtyper.py:280-281).
    ///
    /// ```python
    /// def make_new_lloplist(self, block):
    ///     return LowLevelOpList(self, block)
    /// ```
    ///
    /// `self: &Rc<Self>` is required because `LowLevelOpList` stores
    /// `Rc<RPythonTyper>` — upstream Python carries `self` by
    /// reference semantics, Rust needs the shared-ownership handle
    /// explicit.
    pub fn make_new_lloplist(self: &Rc<Self>, block: BlockRef) -> LowLevelOpList {
        LowLevelOpList::new(Rc::clone(self), Some(block))
    }

    /// RPython `RPythonTyper.highlevelops(self, block, llops)`
    /// (rtyper.py:422-432).
    ///
    /// ```python
    /// def highlevelops(self, block, llops):
    ///     if block.operations:
    ///         for op in block.operations[:-1]:
    ///             yield HighLevelOp(self, op, [], llops)
    ///         if block.canraise:
    ///             exclinks = block.exits[1:]
    ///         else:
    ///             exclinks = []
    ///         yield HighLevelOp(self, block.operations[-1], exclinks, llops)
    /// ```
    ///
    /// Upstream is a generator; Rust returns a Vec since the caller
    /// (`specialize_block`) iterates it fully.
    pub fn highlevelops(
        self: &Rc<Self>,
        block: &BlockRef,
        llops: Rc<RefCell<LowLevelOpList>>,
    ) -> Vec<HighLevelOp> {
        let mut result = Vec::new();
        let b = block.borrow();
        if b.operations.is_empty() {
            return result;
        }
        let last_idx = b.operations.len() - 1;
        for op in &b.operations[..last_idx] {
            result.push(HighLevelOp::new(
                Rc::clone(self),
                op.clone(),
                Vec::new(),
                Rc::clone(&llops),
            ));
        }
        let exclinks: Vec<LinkRef> = if b.canraise() {
            b.exits.iter().skip(1).cloned().collect()
        } else {
            Vec::new()
        };
        result.push(HighLevelOp::new(
            Rc::clone(self),
            b.operations[last_idx].clone(),
            exclinks,
            llops,
        ));
        result
    }

    /// Like [`Self::highlevelops`] but also stamps each emitted hop's
    /// [`HighLevelOp::position_key`] with the graph + block + op_index
    /// triple. Required for production callers (`specialize_block`)
    /// because downstream `find_row` / `desc.specialize` lookups rely
    /// on the position key — `highlevelops` alone leaves `position_key
    /// = None`, which is fine for tests but causes
    /// `desc.specialize(_, None)` to fall back to the bookkeeper's
    /// (cleared) ambient position and pick the wrong calltable row.
    pub fn highlevelops_with_graph(
        self: &Rc<Self>,
        graph: &GraphRef,
        block: &BlockRef,
        llops: Rc<RefCell<LowLevelOpList>>,
    ) -> Vec<HighLevelOp> {
        let hops = self.highlevelops(block, llops);
        for (idx, hop) in hops.iter().enumerate() {
            hop.set_position_key(Some(PositionKey::from_refs(graph, block, idx)));
        }
        hops
    }

    /// RPython `RPythonTyper.translate_hl_to_ll(self, hop, varmapping)`
    /// (rtyper.py:434-481).
    ///
    /// ```python
    /// def translate_hl_to_ll(self, hop, varmapping):
    ///     resultvar = hop.dispatch()
    ///     if hop.exceptionlinks and hop.llops.llop_raising_exceptions is None:
    ///         raise TyperError("the graph catches %s, but the rtyper did not "
    ///                          "take exceptions into account "
    ///                          "(exception_is_here() not called)" % (
    ///             [link.exitcase.__name__ for link in hop.exceptionlinks],))
    ///     if resultvar is None:
    ///         self.translate_no_return_value(hop)
    ///     else:
    ///         assert isinstance(resultvar, (Variable, Constant))
    ///         op = hop.spaceop
    ///         if hop.s_result.is_constant():
    ///             ...assertion that matches resultvar.value == hop.s_result.const...
    ///         resulttype = resultvar.concretetype
    ///         op.result.concretetype = hop.r_result.lowleveltype
    ///         if op.result.concretetype != resulttype:
    ///             raise TyperError(...)
    ///         if (isinstance(resultvar, Variable) and
    ///             resultvar.annotation is None and
    ///             resultvar not in varmapping):
    ///             varmapping[resultvar] = op.result
    ///         elif resultvar is op.result:
    ///             assert varmapping[resultvar] is resultvar
    ///         else:
    ///             hop.llops.append(SpaceOperation('same_as', [resultvar],
    ///                                             op.result))
    /// ```
    ///
    pub fn translate_hl_to_ll(
        self: &Rc<Self>,
        hop: &HighLevelOp,
        varmapping: &mut HashMap<Variable, Hlvalue>,
    ) -> Result<(), TyperError> {
        let resultvar = hop.dispatch()?;

        // rtyper.py:437-441 exception-catch audit.
        if !hop.exceptionlinks.is_empty() && hop.llops.borrow().llop_raising_exceptions.is_none() {
            return Err(TyperError::message(
                "the graph catches exceptions, but the rtyper did not take \
                 exceptions into account (exception_is_here() not called)",
            ));
        }

        let Some(resultvar) = resultvar else {
            // rtyper.py:442-444 None-return delegates to
            // `translate_no_return_value`.
            return self.translate_no_return_value(hop);
        };

        // rtyper.py:446 `isinstance(resultvar, (Variable, Constant))` is
        // structural under `Hlvalue`.
        let r_result = hop
            .r_result
            .borrow()
            .as_ref()
            .map(Arc::clone)
            .ok_or_else(|| {
                TyperError::message(
                    "HighLevelOp.r_result not populated; setup() must run \
                     before translate_hl_to_ll (rtyper.py:632)",
                )
            })?;
        let expected = r_result.lowleveltype().clone();

        // rtyper.py:451-458 constant-result assert: if the annotator
        // declares the result constant and the repr is a non-Void
        // Primitive, then a Constant resultvar must equal s_result.const
        // (or both sides be NaN).
        let s_result = hop.s_result.borrow().clone();
        let s_result_is_constant = s_result.as_ref().map(|s| s.is_constant()).unwrap_or(false);
        if s_result_is_constant {
            if let Hlvalue::Constant(rc) = &resultvar {
                if expected.is_primitive() && !matches!(expected, LowLevelType::Void) {
                    let s_const = s_result
                        .as_ref()
                        .and_then(|s| s.const_().cloned())
                        .expect("s_result.is_constant() implies const_() is Some");
                    assert!(
                        constant_result_values_agree(&rc.value, &s_const),
                        "translate_hl_to_ll: Constant result mismatch \
                         — resultvar.value = {:?}, s_result.const = {:?} \
                         (rtyper.py:456-458)",
                        rc.value,
                        s_const,
                    );
                }
            }
        }

        // rtyper.py:460 resulttype = resultvar.concretetype.
        let resulttype = match &resultvar {
            Hlvalue::Variable(v) => v.concretetype(),
            Hlvalue::Constant(c) => c.concretetype.clone(),
        };

        // rtyper.py:461 `op.result.concretetype = hop.r_result.lowleveltype`.
        // Variable op.result: mutate in place (reference-semantic Rc<RefCell>
        // propagates to every clone). Constant op.result: construct a fresh
        // typed Constant mirror and thread it through `varmapping` /
        // `same_as` so downstream ops see the typed value.
        let op_result: Hlvalue = match &hop.spaceop.result {
            Hlvalue::Variable(v) => {
                v.set_concretetype(Some(expected.clone()));
                Hlvalue::Variable(v.clone())
            }
            Hlvalue::Constant(c) => Hlvalue::Constant(Constant {
                id: c.id,
                value: c.value.clone(),
                concretetype: Some(expected.clone()),
            }),
        };

        // rtyper.py:462-468 type consistency check. Upstream compares
        // `op.result.concretetype` (just written to `expected`) against
        // `resulttype`; equivalent to `expected != resulttype`.
        if resulttype.as_ref() != Some(&expected) {
            return Err(TyperError::message(format!(
                "inconsistent type for the result of '{op_name}':\n\
                 rtype_{op_name} returned {returned:?} \
                 but annotator/repr expected {expected:?}",
                op_name = hop.spaceop.opname,
                returned = resulttype,
                expected = expected,
            )));
        }

        // rtyper.py:470-481 renaming / same_as insertion. `varmapping`
        // values are polymorphic (Variable or Constant) so the fresh-
        // variable branch stores `op_result` directly.
        let resultvar_is_op_result = match (&resultvar, &op_result) {
            (Hlvalue::Variable(a), Hlvalue::Variable(b)) => a == b,
            _ => false,
        };
        match &resultvar {
            Hlvalue::Variable(v)
                if v.annotation.borrow().is_none() && !varmapping.contains_key(v) =>
            {
                // rtyper.py:470-474 fresh Variable: rename to op.result.
                varmapping.insert(v.clone(), op_result);
            }
            Hlvalue::Variable(_) if resultvar_is_op_result => {
                // rtyper.py:475-477 — same Variable handed back. Upstream
                // asserts `varmapping[resultvar] is resultvar`; we elide the
                // assert because it only fires once specialize_block has
                // populated varmapping, which happens after translate_hl_to_ll
                // returns.
            }
            _ => {
                // rtyper.py:478-481 renaming unsafe: emit same_as.
                let mut llops = hop.llops.borrow_mut();
                llops.append(SpaceOperation::new("same_as", vec![resultvar], op_result));
            }
        }
        Ok(())
    }

    /// RPython `RPythonTyper.translate_no_return_value(self, hop)`
    /// (rtyper.py:483-488).
    ///
    /// ```python
    /// def translate_no_return_value(self, hop):
    ///     op = hop.spaceop
    ///     if hop.s_result != annmodel.s_ImpossibleValue:
    ///         raise TyperError("the annotator doesn't agree that '%s' "
    ///                          "has no return value" % op.opname)
    ///     op.result.concretetype = Void
    /// ```
    ///
    pub fn translate_no_return_value(&self, hop: &HighLevelOp) -> Result<(), TyperError> {
        let is_impossible = matches!(hop.s_result.borrow().as_ref(), Some(SomeValue::Impossible));
        if !is_impossible {
            return Err(TyperError::message(format!(
                "the annotator doesn't agree that '{}' has no return value",
                hop.spaceop.opname,
            )));
        }
        // rtyper.py:488 `op.result.concretetype = Void` — unconditional
        // write regardless of whether op.result is Variable or Constant.
        // Variable: Rc<RefCell<Option<LowLevelType>>> propagates through
        // every clone of the same identity. Constant: upstream's Python
        // would set the attribute on the Constant instance; Rust's
        // immutable Constant forces us to construct a typed mirror. The
        // mirror is materialised so the translate shape matches upstream
        // even though the immediate caller (translate_hl_to_ll) discards
        // the None-resultvar path without threading it further.
        let _typed_op_result: Hlvalue = match &hop.spaceop.result {
            Hlvalue::Variable(v) => {
                v.set_concretetype(Some(LowLevelType::Void));
                Hlvalue::Variable(v.clone())
            }
            Hlvalue::Constant(c) => Hlvalue::Constant(Constant {
                id: c.id,
                value: c.value.clone(),
                concretetype: Some(LowLevelType::Void),
            }),
        };
        Ok(())
    }

    /// RPython `RPythonTyper.gottypererror(self, exc, block, position)`
    /// (rtyper.py:490-493).
    ///
    /// ```python
    /// def gottypererror(self, exc, block, position):
    ///     """Record information about the location of a TyperError"""
    ///     graph = self.annotator.annotated.get(block)
    ///     exc.where = (graph, block, position)
    /// ```
    ///
    /// `annotator.annotated: HashMap<BlockKey, Option<GraphRef>>` encodes
    /// upstream's `dict.get(block)` trichotomy:
    /// - key absent → `"<no graph>"` (upstream returns `None`)
    /// - present as `None` → `"<False>"` (upstream's `False` sentinel for
    ///   graphs scheduled but not yet attached)
    /// - present as `Some(graph)` → `graph.name`
    ///
    /// `TyperWhere.stage` holds the graph slot; the naming is a
    /// PRE-EXISTING-DEVIATION that predates this port. Block and
    /// position are serialized via `Debug` so the TyperError can cross
    /// the unwind boundary without borrowing into the block graph.
    ///
    /// Dead on landing — the callers (`specialize_block`,
    /// `insert_link_conversions`) port in 5c/5d. Unit-tested in
    /// isolation.
    pub fn gottypererror<P: std::fmt::Debug + ?Sized>(
        &self,
        exc: TyperError,
        block: &BlockRef,
        position: &P,
    ) -> TyperError {
        let stage = {
            let ann = self
                .annotator
                .upgrade()
                .expect("RPythonTyper.annotator weak reference dropped");
            let annotated = ann.annotated.borrow();
            match annotated.get(&BlockKey::of(block)) {
                None => "<no graph>".to_string(),
                Some(None) => "<False>".to_string(),
                Some(Some(g)) => g.borrow().name.clone(),
            }
        };
        let block_repr = format!("{:?}", block.borrow());
        let op_repr = format!("{position:?}");
        exc.with_where(TyperWhere {
            stage,
            block: block_repr,
            op: op_repr,
        })
    }

    /// RPython `RPythonTyper._convert_link(self, block, link)`
    /// (rtyper.py:353-376).
    ///
    /// ```python
    /// def _convert_link(self, block, link):
    ///     if link.exitcase is not None and link.exitcase != 'default':
    ///         if isinstance(block.exitswitch, Variable):
    ///             r_case = self.bindingrepr(block.exitswitch)
    ///         else:
    ///             assert block.canraise
    ///             r_case = rclass.get_type_repr(self)
    ///         link.llexitcase = r_case.convert_const(link.exitcase)
    ///     else:
    ///         link.llexitcase = None
    ///
    ///     a = link.last_exception
    ///     if isinstance(a, Variable):
    ///         a.concretetype = self.exceptiondata.lltype_of_exception_type
    ///     elif isinstance(a, Constant):
    ///         link.last_exception = inputconst(
    ///             self.exceptiondata.r_exception_type, a.value)
    ///
    ///     a = link.last_exc_value
    ///     if isinstance(a, Variable):
    ///         a.concretetype = self.exceptiondata.lltype_of_exception_value
    ///     elif isinstance(a, Constant):
    ///         link.last_exc_value = inputconst(
    ///             self.exceptiondata.r_exception_value, a.value)
    /// ```
    ///
    /// Non-exception path is fully ported; exception-carrying paths
    /// (`canraise` → `rclass.get_type_repr`, Variable/Constant
    /// `last_exception` / `last_exc_value`) delegate to
    /// [`RPythonTyper::exceptiondata`] and [`rclass::get_type_repr`] and
    /// surface a structured `TyperError` until `exceptiondata.py`
    /// initialisation lands.
    pub fn _convert_link(&self, block: &BlockRef, link: &LinkRef) -> Result<(), TyperError> {
        // First clause: link.exitcase → link.llexitcase.
        let exitcase = link.borrow().exitcase.clone();
        let needs_convert = match exitcase.as_ref() {
            None => false,
            Some(Hlvalue::Constant(c)) => !c.value.string_eq("default"),
            Some(Hlvalue::Variable(_)) => {
                // Upstream Python's `exitcase != 'default'` compares
                // concrete values; a Variable exitcase is not an
                // expected upstream shape.
                return Err(TyperError::message(
                    "_convert_link: Variable exitcase is not expected",
                ));
            }
        };
        if needs_convert {
            let exitswitch = block.borrow().exitswitch.clone();
            let r_case = match exitswitch.as_ref() {
                Some(Hlvalue::Variable(_)) => {
                    let ev = exitswitch.as_ref().unwrap();
                    self.bindingrepr(ev)?
                }
                _ => {
                    // Upstream: `assert block.canraise; r_case = rclass.get_type_repr(self)`.
                    assert!(
                        block.borrow().canraise(),
                        "_convert_link: non-Variable exitswitch must be a canraise block",
                    );
                    crate::translator::rtyper::rclass::get_type_repr(self)?
                }
            };
            // Upstream hands `link.exitcase` — a concrete Python value —
            // directly to `convert_const`. In Rust the concrete value
            // lives inside `Hlvalue::Constant(Constant { value, .. })`;
            // thread `&value` through.
            let cv = match exitcase.as_ref().expect("needs_convert implies Some") {
                Hlvalue::Constant(c) => &c.value,
                Hlvalue::Variable(_) => unreachable!("Variable exitcase rejected above"),
            };
            let converted = r_case.convert_const(cv)?;
            link.borrow_mut().llexitcase = Some(Hlvalue::Constant(converted));
        } else {
            link.borrow_mut().llexitcase = None;
        }

        // Second clause: link.last_exception.
        let last_exception = link.borrow().last_exception.clone();
        match last_exception {
            None => {}
            Some(Hlvalue::Variable(v)) => {
                let ed = self.exceptiondata()?;
                v.set_concretetype(Some(ed.lltype_of_exception_type.clone()));
            }
            Some(Hlvalue::Constant(c)) => {
                let ed = self.exceptiondata()?;
                let typed = inputconst(ed.r_exception_type.as_ref(), &c.value)?;
                link.borrow_mut().last_exception = Some(Hlvalue::Constant(typed));
            }
        }

        // Third clause: link.last_exc_value. Same shape as above.
        let last_exc_value = link.borrow().last_exc_value.clone();
        match last_exc_value {
            None => {}
            Some(Hlvalue::Variable(v)) => {
                let ed = self.exceptiondata()?;
                v.set_concretetype(Some(ed.lltype_of_exception_value.clone()));
            }
            Some(Hlvalue::Constant(c)) => {
                let ed = self.exceptiondata()?;
                let typed = inputconst(ed.r_exception_value.as_ref(), &c.value)?;
                link.borrow_mut().last_exc_value = Some(Hlvalue::Constant(typed));
            }
        }

        Ok(())
    }

    /// RPython `RPythonTyper.insert_link_conversions(self, block, skip=0)`
    /// (rtyper.py:378-420).
    ///
    /// ```python
    /// def insert_link_conversions(self, block, skip=0):
    ///     can_insert_here = block.exitswitch is None and len(block.exits) == 1
    ///     for link in block.exits[skip:]:
    ///         self._convert_link(block, link)
    ///         inputargs_reprs = self.setup_block_entry(link.target)
    ///         newops = self.make_new_lloplist(block)
    ///         newlinkargs = {}
    ///         for i in range(len(link.args)):
    ///             a1 = link.args[i]
    ///             r_a2 = inputargs_reprs[i]
    ///             if isinstance(a1, Constant):
    ///                 link.args[i] = inputconst(r_a2, a1.value)
    ///                 continue
    ///             if a1 is link.last_exception:
    ///                 r_a1 = self.exceptiondata.r_exception_type
    ///             elif a1 is link.last_exc_value:
    ///                 r_a1 = self.exceptiondata.r_exception_value
    ///             else:
    ///                 r_a1 = self.bindingrepr(a1)
    ///             if r_a1 == r_a2:
    ///                 continue
    ///             try:
    ///                 new_a1 = newops.convertvar(a1, r_a1, r_a2)
    ///             except TyperError as e:
    ///                 self.gottypererror(e, block, link)
    ///                 raise
    ///             if new_a1 != a1:
    ///                 newlinkargs[i] = new_a1
    ///
    ///         if newops:
    ///             if can_insert_here:
    ///                 block.operations.extend(newops)
    ///             else:
    ///                 newblock = insert_empty_block(link, newops=newops)
    ///                 link = newblock.exits[0]
    ///         for i, new_a1 in newlinkargs.items():
    ///             link.args[i] = new_a1
    /// ```
    ///
    /// Simple-case only: exception-aware branches (`last_exception` /
    /// `last_exc_value` reprs from `ExceptionData`) are gated out via
    /// `_convert_link` and `setup_block_entry`, whose deferred errors
    /// bubble back through `gottypererror`. Multi-exit blocks are still
    /// handled when none of their exits need ExceptionData reprs.
    pub fn insert_link_conversions(
        self: &Rc<Self>,
        block: &BlockRef,
        skip: usize,
    ) -> Result<(), TyperError> {
        let can_insert_here = {
            let b = block.borrow();
            b.exitswitch.is_none() && b.exits.len() == 1
        };
        let exits: Vec<LinkRef> = {
            let b = block.borrow();
            b.exits.iter().skip(skip).cloned().collect()
        };
        for link in exits {
            // rtyper.py:382-383 — `_convert_link` and `setup_block_entry`
            // errors propagate uncaught; only `newops.convertvar` is
            // wrapped with `gottypererror` (rtyper.py:400-403).
            self._convert_link(block, &link)?;

            let target = link
                .borrow()
                .target
                .clone()
                .expect("insert_link_conversions: link.target must be set");
            let inputargs_reprs = self.setup_block_entry(&target)?;

            let mut newops = self.make_new_lloplist(block.clone());
            let mut newlinkargs: HashMap<usize, Hlvalue> = HashMap::new();
            let link_args = link.borrow().args.clone();
            for i in 0..link_args.len() {
                // Upstream `link.args[i]` is always a Hlvalue; pyre's
                // `Option<Hlvalue>` carries transient `None` for merge
                // links, which are not valid input to the rtyper phase.
                let a1 = match &link_args[i] {
                    Some(v) => v.clone(),
                    None => {
                        return Err(TyperError::message(
                            "insert_link_conversions: transient None link.args slot reached rtyper",
                        ));
                    }
                };
                let r_a2 = &inputargs_reprs[i];

                if let Hlvalue::Constant(c) = &a1 {
                    let typed = inputconst(r_a2.as_ref(), &c.value)?;
                    link.borrow_mut().args[i] = Some(Hlvalue::Constant(typed));
                    continue;
                }

                // Variable branch. rtyper.py:392-397 — `a1 is
                // link.last_exception` / `is link.last_exc_value` pick
                // up the ExceptionData reprs directly instead of going
                // through `bindingrepr`.
                let a1_is_last_exception = matches!(
                    (&a1, link.borrow().last_exception.as_ref()),
                    (Hlvalue::Variable(va), Some(Hlvalue::Variable(vb))) if va == vb,
                );
                let a1_is_last_exc_value = matches!(
                    (&a1, link.borrow().last_exc_value.as_ref()),
                    (Hlvalue::Variable(va), Some(Hlvalue::Variable(vb))) if va == vb,
                );
                let r_a1 = if a1_is_last_exception {
                    self.exceptiondata()?.r_exception_type.clone()
                } else if a1_is_last_exc_value {
                    self.exceptiondata()?.r_exception_value.clone()
                } else {
                    self.bindingrepr(&a1)?
                };
                if Arc::ptr_eq(&r_a1, r_a2) {
                    continue;
                }
                let new_a1 = match newops.convertvar(a1.clone(), r_a1.as_ref(), r_a2.as_ref()) {
                    Ok(v) => v,
                    Err(e) => {
                        let snapshot = link.borrow();
                        return Err(self.gottypererror(e, block, &*snapshot));
                    }
                };
                if new_a1 != a1 {
                    newlinkargs.insert(i, new_a1);
                }
            }

            let newops_vec = std::mem::take(&mut newops.ops);
            let target_link = if !newops_vec.is_empty() {
                if can_insert_here {
                    block.borrow_mut().operations.extend(newops_vec);
                    link.clone()
                } else {
                    let newblock = insert_empty_block(&link, newops_vec);
                    // upstream: `link = newblock.exits[0]`
                    newblock.borrow().exits[0].clone()
                }
            } else {
                link.clone()
            };
            for (i, new_a1) in newlinkargs {
                target_link.borrow_mut().args[i] = Some(new_a1);
            }
        }
        Ok(())
    }

    /// RPython `RPythonTyper.specialize_block(self, block)`
    /// (rtyper.py:283-351).
    ///
    /// Block-local driver: concretetype the return var on first visit
    /// per graph, type-up the block's inputargs via `setup_block_entry`,
    /// dispatch each `HighLevelOp` through `translate_hl_to_ll`, replace
    /// `block.operations` with the generated LowLevelOpList,
    /// optionally split around an exception-raising llop, and finally
    /// reconcile link-arg reprs via `insert_link_conversions`. Any
    /// `TyperError` raised along the way is annotated via
    /// `gottypererror` with the (graph, block, position) trio before
    /// propagating.
    ///
    /// `annotator.annotated[block]` is expected to resolve — upstream
    /// raises `KeyError` when missing and pyre panics for the same
    /// reason. A `Some(None)` inner value (the `False` sentinel from
    /// annrpython) skips the fixed_graphs update to match upstream's
    /// defensive behavior when `graph.getreturnvar()` would crash.
    pub fn specialize_block(self: &Rc<Self>, block: &BlockRef) -> Result<(), TyperError> {
        let graph: GraphRef = {
            let ann = self
                .annotator
                .upgrade()
                .expect("RPythonTyper.annotator weak reference dropped");
            let annotated = ann.annotated.borrow();
            annotated
                .get(&BlockKey::of(block))
                .cloned()
                .expect("specialize_block: block missing from annotator.annotated")
                .expect(
                    "specialize_block: annotator.annotated[block] is False sentinel — \
                     RPython would raise AttributeError on graph.getreturnvar()",
                )
        };

        {
            let ann = self
                .annotator
                .upgrade()
                .expect("RPythonTyper.annotator weak reference dropped");
            let gkey = GraphKey::of(&graph);
            let is_new = !ann.fixed_graphs.borrow().contains_key(&gkey);
            if is_new {
                ann.fixed_graphs.borrow_mut().insert(gkey, graph.clone());
                if let Hlvalue::Variable(v) = graph.borrow().getreturnvar() {
                    self.setconcretetype(&v)?;
                }
            }
        }

        if let Err(e) = self.setup_block_entry(block) {
            return Err(self.gottypererror(e, block, &"block-entry"));
        }

        // rtyper.py:300 — upstream's `block.operations == ()` distinguishes
        // the tuple-sentinel final block (return/except) from an empty
        // list in a regular block. Empty regular blocks must still flow
        // through `insert_link_conversions`.
        if block.borrow().is_final_block() {
            return Ok(());
        }

        let newops = Rc::new(RefCell::new(self.make_new_lloplist(block.clone())));
        let mut varmapping: HashMap<Variable, Hlvalue> = HashMap::new();
        for v in block.borrow().getvariables() {
            varmapping.insert(v.clone(), Hlvalue::Variable(v));
        }

        let hops = self.highlevelops_with_graph(&graph, block, newops.clone());
        for hop in hops {
            if let Err(e) = hop.setup() {
                return Err(self.gottypererror(e, block, &hop.spaceop));
            }
            if let Err(e) = self.translate_hl_to_ll(&hop, &mut varmapping) {
                return Err(self.gottypererror(e, block, &hop.spaceop));
            }
        }

        // `block.operations[:] = newops` — replace in place, matching
        // upstream's slice assignment.
        let new_ops_vec = std::mem::take(&mut newops.borrow_mut().ops);
        block.borrow_mut().operations = new_ops_vec;
        block.borrow_mut().renamevariables(&varmapping);

        // extrablock handling (rtyper.py:318-341): if the llop that
        // raises exceptions isn't the last one, either strip the
        // exception exits (`"removed"` sentinel) or split the block.
        let pos = newops.borrow().llop_raising_exceptions.clone();
        let ops_len = block.borrow().operations.len();
        let last = ops_len.saturating_sub(1);
        let mut extrablock: Option<BlockRef> = None;
        match pos {
            None => {}
            Some(LlopRaisingExceptions::Index(i)) if i == last => {}
            Some(LlopRaisingExceptions::Removed) => {
                let first_exit = block.borrow().exits[0].clone();
                let mut b = block.borrow_mut();
                b.exitswitch = None;
                b.exits = vec![first_exit];
            }
            Some(LlopRaisingExceptions::Index(i)) => {
                assert!(
                    block.borrow().canraise(),
                    "specialize_block: non-last raising llop requires canraise block",
                );
                let noexclink = block.borrow().exits[0].clone();
                assert!(
                    noexclink.borrow().exitcase.is_none(),
                    "specialize_block: noexclink must have no exitcase",
                );
                assert!(i < last);
                // extraops = block.operations[i+1:]; del block.operations[i+1:]
                let extraops: Vec<SpaceOperation> = block.borrow_mut().operations.split_off(i + 1);
                extrablock = Some(insert_empty_block(&noexclink, extraops));
            }
        }

        if extrablock.is_none() {
            self.insert_link_conversions(block, 0)?;
        } else {
            // skip the extrablock as a link target, handle it as source
            // below.
            self.insert_link_conversions(block, 1)?;
            self.insert_link_conversions(extrablock.as_ref().unwrap(), 0)?;
        }

        Ok(())
    }

    /// RPython `RPythonTyper.specialize(self, dont_simplify_again=False)`
    /// (rtyper.py:177-189).
    ///
    /// ```python
    /// def specialize(self, dont_simplify_again=False):
    ///     if not dont_simplify_again:
    ///         self.annotator.simplify()
    ///     self.exceptiondata.finish(self)
    ///     self.already_seen = {}
    ///     self.specialize_more_blocks()
    ///     self.exceptiondata.make_helpers(self)
    ///     self.specialize_more_blocks()   # for the helpers just made
    /// ```
    ///
    /// Scope deviation: `exceptiondata.make_helpers(self)` is blocked
    /// on R4 (`MixLevelHelperAnnotator` + `ll_issubclass` / `ll_type`);
    /// documented on the module header of
    /// `translator/rtyper/rclass.rs`. Until that lands the call site
    /// below is a no-op and the second `specialize_more_blocks()` pass
    /// is redundant (no new helper graphs are generated). Both lines
    /// remain in code shape so the future port is a swap-in of the
    /// make_helpers body rather than a call-site audit.
    pub fn specialize(self: &Rc<Self>, dont_simplify_again: bool) -> Result<(), TyperError> {
        // rtyper.py:180-181 — optional annotator.simplify pass.
        if !dont_simplify_again {
            let ann = self.annotator.upgrade().ok_or_else(|| {
                TyperError::message(
                    "RPythonTyper.specialize: RPythonAnnotator weak reference dropped",
                )
            })?;
            ann.simplify(None, None);
        }
        // rtyper.py:182 — `self.exceptiondata.finish(self)`.
        self.finish_exceptiondata()?;
        // rtyper.py:186 — `self.already_seen = {}`. Upstream resets the
        // specialize visitation set so a second pass can retrace the
        // newly-created helper blocks; pyre mirrors this with a
        // RefCell-scoped clear.
        self.already_seen.borrow_mut().clear();
        // rtyper.py:187 — first `specialize_more_blocks()` pass.
        self.specialize_more_blocks()?;
        // rtyper.py:188 — `self.exceptiondata.make_helpers(self)`.
        self.exceptiondata()?.make_helpers(self)?;
        // rtyper.py:189 — second `specialize_more_blocks()` pass for
        // the helpers just made.
        self.specialize_more_blocks()?;
        Ok(())
    }

    /// RPython `RPythonTyper.specialize_more_blocks(self)`
    /// (rtyper.py:198-241).
    ///
    /// Fixed-point loop: call all pending repr setups, collect every
    /// annotated block not yet marked in `already_seen`, specialize
    /// each, mark it seen, repeat until nothing new is pending.
    ///
    /// Scope deviations from upstream:
    /// - No `seed`-based shuffle (pyre lacks the translator-level seed
    ///   knob; block ordering falls out of HashMap iteration).
    /// - No progress-bar log events.
    /// - `BlockKey → BlockRef` materialisation reads from
    ///   `annotator.all_blocks`, the Rust-side reverse index that keeps
    ///   Python's "dict keys ARE Block objects" iteration shape.
    pub fn specialize_more_blocks(self: &Rc<Self>) -> Result<(), TyperError> {
        // upstream rtyper.py:204 — `self.annmixlevel = None`. Resets
        // the helper-annotator cache at every pass so each pass that
        // needs `getannmixlevel` constructs a fresh
        // `MixLevelHelperAnnotator` whose `pending` queue is drained on
        // the matching `finish` call at the bottom of this method.
        *self.annmixlevel.borrow_mut() = None;
        loop {
            self.call_all_setups()?;
            let pending: Vec<BlockRef> = {
                let ann = self
                    .annotator
                    .upgrade()
                    .expect("RPythonTyper.annotator weak reference dropped");
                let annotated = ann.annotated.borrow();
                let all_blocks = ann.all_blocks.borrow();
                let already_seen = self.already_seen.borrow();
                annotated
                    .keys()
                    .filter(|k| !already_seen.contains_key(*k))
                    .filter_map(|k| all_blocks.get(k).cloned())
                    .collect()
            };
            if pending.is_empty() {
                break;
            }
            for block in pending {
                self.specialize_block(&block)?;
                self.mark_already_seen(&block);
            }
        }
        // upstream rtyper.py:238-241 — `annmixlevel = self.annmixlevel;
        // del self.annmixlevel; if annmixlevel is not None:
        // annmixlevel.finish()`. The take-then-finish drains any helper
        // graphs queued via `getannmixlevel().getgraph(...)` during this
        // pass. Skipped silently when no caller invoked `getannmixlevel`.
        let helper = self.annmixlevel.borrow_mut().take();
        if let Some(helper) = helper {
            helper.finish()?;
        }
        Ok(())
    }

    /// RPython `RPythonTyper.getannmixlevel(self)` (rtyper.py:191-196).
    ///
    /// ```python
    /// def getannmixlevel(self):
    ///     if self.annmixlevel is not None:
    ///         return self.annmixlevel
    ///     from rpython.rtyper.annlowlevel import MixLevelHelperAnnotator
    ///     self.annmixlevel = MixLevelHelperAnnotator(self)
    ///     return self.annmixlevel
    /// ```
    ///
    /// Lazy-singleton getter for the per-pass
    /// [`MixLevelHelperAnnotator`]. The cache is cleared at every
    /// [`Self::specialize_more_blocks`] entry, so callers within the
    /// same pass share one helper instance and `finish_annotate` in
    /// `specialize` (rtyper.py:188) drains a deterministic queue.
    pub fn getannmixlevel(
        self: &Rc<Self>,
    ) -> Rc<crate::translator::rtyper::annlowlevel::MixLevelHelperAnnotator> {
        if let Some(annmixlevel) = self.annmixlevel.borrow().clone() {
            return annmixlevel;
        }
        let helper =
            Rc::new(crate::translator::rtyper::annlowlevel::MixLevelHelperAnnotator::new(self));
        *self.annmixlevel.borrow_mut() = Some(helper.clone());
        helper
    }

    /// RPython `RPythonTyper.setup_block_entry(self, block)`
    /// (rtyper.py:262-278).
    ///
    /// ```python
    /// def setup_block_entry(self, block):
    ///     if block.operations == () and len(block.inputargs) == 2:
    ///         # special case for exception blocks
    ///         v1, v2 = block.inputargs
    ///         v1.concretetype = self.exceptiondata.lltype_of_exception_type
    ///         v2.concretetype = self.exceptiondata.lltype_of_exception_value
    ///         return [self.exceptiondata.r_exception_type,
    ///                 self.exceptiondata.r_exception_value]
    ///     else:
    ///         result = []
    ///         for a in block.inputargs:
    ///             r = self.bindingrepr(a)
    ///             a.concretetype = r.lowleveltype
    ///             result.append(r)
    ///         return result
    /// ```
    ///
    /// The exception-block branch reads `ExceptionData`; callers hit a
    /// structured `TyperError` through [`RPythonTyper::exceptiondata`]
    /// until `exceptiondata.py:16` initialisation lands.
    pub fn setup_block_entry(&self, block: &BlockRef) -> Result<Vec<Arc<dyn Repr>>, TyperError> {
        let is_exception_block = {
            let b = block.borrow();
            b.is_final_block() && b.inputargs.len() == 2
        };
        if is_exception_block {
            let ed = self.exceptiondata()?;
            let b = block.borrow();
            // upstream: `v1, v2 = block.inputargs`
            // `v1.concretetype = self.exceptiondata.lltype_of_exception_type`
            // `v2.concretetype = self.exceptiondata.lltype_of_exception_value`
            // `return [self.exceptiondata.r_exception_type,
            //          self.exceptiondata.r_exception_value]`
            let Hlvalue::Variable(v1) = &b.inputargs[0] else {
                return Err(TyperError::message(
                    "setup_block_entry: exception block inputarg[0] must be Variable",
                ));
            };
            let Hlvalue::Variable(v2) = &b.inputargs[1] else {
                return Err(TyperError::message(
                    "setup_block_entry: exception block inputarg[1] must be Variable",
                ));
            };
            v1.set_concretetype(Some(ed.lltype_of_exception_type.clone()));
            v2.set_concretetype(Some(ed.lltype_of_exception_value.clone()));
            return Ok(vec![
                ed.r_exception_type.clone(),
                ed.r_exception_value.clone(),
            ]);
        }
        let mut result: Vec<Arc<dyn Repr>> = Vec::new();
        let b = block.borrow();
        for a in b.inputargs.iter() {
            let r = self.bindingrepr(a)?;
            match a {
                Hlvalue::Variable(v) => {
                    // `Variable.concretetype` is `Rc<RefCell<...>>`; the
                    // write propagates through every clone of the same
                    // Variable identity.
                    v.set_concretetype(Some(r.lowleveltype().clone()));
                }
                Hlvalue::Constant(_) => {
                    return Err(TyperError::message(
                        "Block.inputargs contained Constant; \
                         upstream Python raises AttributeError on .concretetype",
                    ));
                }
            }
            result.push(r);
        }
        Ok(result)
    }

    /// RPython `RPythonTyper.call_all_setups(self)` (rtyper.py:243-256).
    ///
    /// ```python
    /// def call_all_setups(self):
    ///     must_setup_more = []
    ///     delayed = []
    ///     while self._reprs_must_call_setup:
    ///         r = self._reprs_must_call_setup.pop()
    ///         if r.is_setup_delayed():
    ///             delayed.append(r)
    ///         else:
    ///             r.setup()
    ///             must_setup_more.append(r)
    ///     for r in must_setup_more:
    ///         r.setup_final()
    ///     self._reprs_must_call_setup.extend(delayed)
    /// ```
    pub fn call_all_setups(&self) -> Result<(), TyperError> {
        let mut must_setup_more: Vec<Arc<dyn Repr>> = Vec::new();
        let mut delayed: Vec<Arc<dyn Repr>> = Vec::new();
        loop {
            let r = self.reprs_must_call_setup.borrow_mut().pop();
            let Some(r) = r else {
                break;
            };
            if r.is_setup_delayed() {
                delayed.push(r);
            } else {
                r.setup()?;
                must_setup_more.push(r);
            }
        }
        for r in &must_setup_more {
            r.setup_final()?;
        }
        self.reprs_must_call_setup.borrow_mut().extend(delayed);
        Ok(())
    }

    /// RPython `HighLevelOp.dispatch` target (`rtyper.py:648-653`).
    pub fn translate_operation(&self, hop: &HighLevelOp) -> RTypeResult {
        match hop.spaceop.opname.as_str() {
            // rtyper.py:497-518 registers `translate_op_<opname>` for
            // unary/binary operations. Unary ops route to `Repr.rtype_*`;
            // binary ops route through the explicit pairtype dispatcher,
            // matching `pair(r_arg1, r_arg2).rtype_*`.
            "getattr" => self.translate_unary_operation(hop, |r, hop| r.rtype_getattr(hop)),
            "setattr" => self.translate_unary_operation(hop, |r, hop| r.rtype_setattr(hop)),
            "len" => self.translate_unary_operation(hop, |r, hop| r.rtype_len(hop)),
            "bool" => self.translate_unary_operation(hop, |r, hop| r.rtype_bool(hop)),
            "simple_call" => self.translate_unary_operation(hop, |r, hop| r.rtype_simple_call(hop)),
            "neg" => self.translate_unary_operation(hop, |r, hop| r.rtype_neg(hop)),
            "neg_ovf" => self.translate_unary_operation(hop, |r, hop| r.rtype_neg_ovf(hop)),
            "pos" => self.translate_unary_operation(hop, |r, hop| r.rtype_pos(hop)),
            "abs" => self.translate_unary_operation(hop, |r, hop| r.rtype_abs(hop)),
            "abs_ovf" => self.translate_unary_operation(hop, |r, hop| r.rtype_abs_ovf(hop)),
            "int" => self.translate_unary_operation(hop, |r, hop| r.rtype_int(hop)),
            "float" => self.translate_unary_operation(hop, |r, hop| r.rtype_float(hop)),
            "invert" => self.translate_unary_operation(hop, |r, hop| r.rtype_invert(hop)),
            "bin" => self.translate_unary_operation(hop, |r, hop| r.rtype_bin(hop)),
            "call_args" => self.translate_unary_operation(hop, |r, hop| r.rtype_call_args(hop)),
            "delattr" => self.translate_unary_operation(hop, |r, hop| r.rtype_delattr(hop)),
            "delslice" => self.translate_unary_operation(hop, |r, hop| r.rtype_delslice(hop)),
            "getslice" => self.translate_unary_operation(hop, |r, hop| r.rtype_getslice(hop)),
            "hash" => self.translate_unary_operation(hop, |r, hop| r.rtype_hash(hop)),
            "hex" => self.translate_unary_operation(hop, |r, hop| r.rtype_hex(hop)),
            "hint" => self.translate_unary_operation(hop, |r, hop| r.rtype_hint(hop)),
            "id" => self.translate_unary_operation(hop, |r, hop| r.rtype_id(hop)),
            "isinstance" => self.translate_unary_operation(hop, |r, hop| r.rtype_isinstance(hop)),
            "issubtype" => self.translate_unary_operation(hop, |r, hop| r.rtype_issubtype(hop)),
            "iter" => self.translate_unary_operation(hop, |r, hop| r.rtype_iter(hop)),
            "long" => self.translate_unary_operation(hop, |r, hop| r.rtype_long(hop)),
            "next" => self.translate_unary_operation(hop, |r, hop| r.rtype_next(hop)),
            "oct" => self.translate_unary_operation(hop, |r, hop| r.rtype_oct(hop)),
            "ord" => self.translate_unary_operation(hop, |r, hop| r.rtype_ord(hop)),
            "repr" => self.translate_unary_operation(hop, |r, hop| r.rtype_repr(hop)),
            "setslice" => self.translate_unary_operation(hop, |r, hop| r.rtype_setslice(hop)),
            "str" => self.translate_unary_operation(hop, |r, hop| r.rtype_str(hop)),
            "type" => self.translate_unary_operation(hop, |r, hop| r.rtype_type(hop)),
            "getitem" | "getitem_idx" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_getitem)
            }
            "setitem" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_setitem),
            "delitem" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_delitem),
            "is_" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_is_),
            "eq" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_eq),
            "ne" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_ne),
            "lt" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_lt),
            "le" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_le),
            "gt" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_gt),
            "ge" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_ge),
            "add" | "inplace_add" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_add)
            }
            "add_ovf" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_add_ovf),
            "sub" | "inplace_sub" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_sub)
            }
            "sub_ovf" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_sub_ovf),
            "mul" | "inplace_mul" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_mul)
            }
            "mul_ovf" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_mul_ovf),
            "truediv" | "inplace_truediv" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_truediv)
            }
            "floordiv" | "inplace_floordiv" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_floordiv)
            }
            "floordiv_ovf" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_floordiv_ovf)
            }
            "div" | "inplace_div" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_div)
            }
            "div_ovf" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_div_ovf),
            "mod" | "inplace_mod" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_mod)
            }
            "mod_ovf" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_mod_ovf),
            "xor" | "inplace_xor" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_xor)
            }
            "and_" | "inplace_and" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_and_)
            }
            "or_" | "inplace_or" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_or_)
            }
            "lshift" | "inplace_lshift" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_lshift)
            }
            "lshift_ovf" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_lshift_ovf)
            }
            "rshift" | "inplace_rshift" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_rshift)
            }
            "cmp" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_cmp),
            "coerce" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_coerce),
            // rtyper.py:547-549 — `translate_op_newtuple` calls the
            // free function `rtuple.rtype_newtuple(hop)` which routes
            // to `TupleRepr._rtype_newtuple`. No per-Repr dispatch.
            "newtuple" => super::rtuple::TupleRepr::rtype_newtuple(hop),
            // rtuple.py:292-315 — `pairtype(TupleRepr, Repr).rtype_contains`.
            "contains" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_contains),
            _ => self.default_translate_operation(hop),
        }
    }

    fn translate_unary_operation<F>(&self, hop: &HighLevelOp, method: F) -> RTypeResult
    where
        F: Fn(&dyn Repr, &HighLevelOp) -> RTypeResult,
    {
        let r_arg1 = self.hop_arg_repr(hop, 0)?;
        method(r_arg1.as_ref(), hop)
    }

    fn translate_pair_operation(
        &self,
        hop: &HighLevelOp,
        method: fn(&dyn Repr, &dyn Repr, &HighLevelOp) -> RTypeResult,
    ) -> RTypeResult {
        let r_arg1 = self.hop_arg_repr(hop, 0)?;
        let r_arg2 = self.hop_arg_repr(hop, 1)?;
        method(r_arg1.as_ref(), r_arg2.as_ref(), hop)
    }

    fn hop_arg_repr(&self, hop: &HighLevelOp, index: usize) -> Result<Arc<dyn Repr>, TyperError> {
        let args_r = hop.args_r.borrow();
        let r = args_r.get(index).ok_or_else(|| {
            TyperError::message(format!(
                "translate_op_{} missing argument repr {}",
                hop.spaceop.opname, index
            ))
        })?;
        r.clone().ok_or_else(|| {
            TyperError::message(format!(
                "translate_op_{} argument repr {} is not set",
                hop.spaceop.opname, index
            ))
        })
    }

    /// RPython `RPythonTyper.default_translate_operation`
    /// (`rtyper.py:557-558`).
    pub fn default_translate_operation(&self, hop: &HighLevelOp) -> RTypeResult {
        Err(TyperError::message(format!(
            "unimplemented operation: '{}'",
            hop.spaceop.opname
        )))
    }
}

fn is_primitive_lowleveltype(lltype: &LowLevelType) -> bool {
    matches!(
        lltype,
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
    )
}

// ____________________________________________________________
// HighLevelOp — `rtyper.py:617-779`.

/// RPython `HighLevelOp.inputarg(converted_to, arg)` accepts either a
/// `Repr` instance or a primitive low-level type. Rust makes that overload
/// explicit.
pub enum ConvertedTo<'a> {
    Repr(&'a dyn Repr),
    LowLevelType(&'a LowLevelType),
}

impl<'a, T: Repr> From<&'a T> for ConvertedTo<'a> {
    fn from(value: &'a T) -> Self {
        ConvertedTo::Repr(value)
    }
}

impl<'a> From<&'a dyn Repr> for ConvertedTo<'a> {
    fn from(value: &'a dyn Repr) -> Self {
        ConvertedTo::Repr(value)
    }
}

impl<'a> From<&'a Arc<dyn Repr>> for ConvertedTo<'a> {
    fn from(value: &'a Arc<dyn Repr>) -> Self {
        ConvertedTo::Repr(value.as_ref())
    }
}

impl<'a> From<&'a LowLevelType> for ConvertedTo<'a> {
    fn from(value: &'a LowLevelType) -> Self {
        ConvertedTo::LowLevelType(value)
    }
}

enum ResolvedConvertedTo<'a> {
    Borrowed(&'a dyn Repr),
    Owned(Arc<dyn Repr>),
}

impl<'a> ResolvedConvertedTo<'a> {
    fn as_repr(&self) -> &dyn Repr {
        match self {
            ResolvedConvertedTo::Borrowed(repr) => *repr,
            ResolvedConvertedTo::Owned(repr) => repr.as_ref(),
        }
    }
}

/// RPython `class HighLevelOp(object)` (rtyper.py:617-779).
///
/// The per-operation carrier passed to every `translate_op_*` +
/// `Repr.rtype_*` method during `specialize_block`. Fields populated
/// in two stages:
///
/// * Construction (`HighLevelOp.__init__`, rtyper.py:619-623) — stores
///   the rtyper/spaceop/exceptionlinks/llops handles.
/// * `setup()` (rtyper.py:625-633) — materialises `args_v`, `args_s`,
///   `s_result`, `args_r`, `r_result` by querying the annotator and
///   the rtyper. Calls can still surface `MissingRTypeOperation` until
///   each concrete `SomeValue.rtyper_makerepr` arm is ported.
pub struct HighLevelOp {
    /// RPython `self.rtyper = rtyper` (rtyper.py:620).
    pub rtyper: Rc<RPythonTyper>,
    /// RPython `self.spaceop = spaceop` (rtyper.py:621).
    pub spaceop: SpaceOperation,
    /// RPython `self.exceptionlinks = exceptionlinks` (rtyper.py:622).
    /// Set of exceptional successor links collected by
    /// `highlevelops(...)` when a block raises
    /// (`rtyper.py:428-431`).
    pub exceptionlinks: Vec<LinkRef>,
    /// RPython `self.llops = llops` (rtyper.py:623) — shared mutable
    /// low-level op buffer across all hops of the block. Pyre wraps
    /// in `Rc<RefCell<_>>` to mirror Python's by-reference sharing.
    pub llops: Rc<RefCell<LowLevelOpList>>,

    // Fields populated by `setup()` — upstream initialises these
    // lazily (rtyper.py:625-633). Pyre mirrors with `RefCell<Option>`
    // so concrete Reprs can be filled in order without forcing a
    // one-shot `setup()` contract on the Rust side.
    /// RPython `self.args_v = list(spaceop.args)` (rtyper.py:628).
    pub args_v: RefCell<Vec<Hlvalue>>,
    /// RPython `self.args_s = [rtyper.binding(a) for a in spaceop.args]`
    /// (rtyper.py:629).
    pub args_s: RefCell<Vec<SomeValue>>,
    /// RPython `self.s_result = rtyper.binding(spaceop.result)`
    /// (rtyper.py:630).
    pub s_result: RefCell<Option<SomeValue>>,
    /// RPython `self.args_r = [rtyper.getrepr(s_a) for s_a in
    /// self.args_s]` (rtyper.py:631). `None` entries mean the
    /// corresponding Repr has not been materialised yet (upstream
    /// requires all to be filled before `dispatch()` runs).
    pub args_r: RefCell<Vec<Option<Arc<dyn Repr>>>>,
    /// RPython `self.r_result = rtyper.getrepr(self.s_result)`
    /// (rtyper.py:632).
    pub r_result: RefCell<Option<Arc<dyn Repr>>>,
    /// Position key for this op site (graph + block + op_index identity).
    ///
    /// Upstream `find_row(bookkeeper, descs, args, op)` (rpbc.py:204,
    /// description.py:54-59) passes `hop.spaceop` to
    /// `desc.specialize(inputs, op)`; specialize then derives a
    /// position key from the op's enclosing context. Pyre cannot recover
    /// graph/block identity from a bare [`SpaceOperation`], so the
    /// callsite that constructs the [`HighLevelOp`] populates this field
    /// directly via [`Self::set_position_key`]. `None` is the
    /// test-fixture default; production [`RPythonTyper::specialize_block`]
    /// fills it via [`RPythonTyper::highlevelops_with_graph`].
    pub position_key: RefCell<Option<PositionKey>>,
}

impl HighLevelOp {
    /// RPython `HighLevelOp.__init__(self, rtyper, spaceop,
    /// exceptionlinks, llops)` (rtyper.py:619-623).
    pub fn new(
        rtyper: Rc<RPythonTyper>,
        spaceop: SpaceOperation,
        exceptionlinks: Vec<LinkRef>,
        llops: Rc<RefCell<LowLevelOpList>>,
    ) -> Self {
        HighLevelOp {
            rtyper,
            spaceop,
            exceptionlinks,
            llops,
            args_v: RefCell::new(Vec::new()),
            args_s: RefCell::new(Vec::new()),
            s_result: RefCell::new(None),
            args_r: RefCell::new(Vec::new()),
            r_result: RefCell::new(None),
            position_key: RefCell::new(None),
        }
    }

    /// Stamp the (graph, block, op_index) position key onto this hop.
    /// Mirrors upstream's reliance on `hop.spaceop` to recover the
    /// op-site identity for `desc.specialize(inputs, op)` lookups.
    pub fn set_position_key(&self, position_key: Option<PositionKey>) {
        *self.position_key.borrow_mut() = position_key;
    }

    /// RPython `HighLevelOp.nb_args` property (rtyper.py:636-637).
    pub fn nb_args(&self) -> usize {
        self.args_v.borrow().len()
    }

    /// RPython `HighLevelOp.copy(self)` (`rtyper.py:639-646`).
    pub fn copy(&self) -> Self {
        let result = HighLevelOp::new(
            self.rtyper.clone(),
            self.spaceop.clone(),
            self.exceptionlinks.clone(),
            self.llops.clone(),
        );
        *result.args_v.borrow_mut() = self.args_v.borrow().clone();
        *result.args_s.borrow_mut() = self.args_s.borrow().clone();
        *result.s_result.borrow_mut() = self.s_result.borrow().clone();
        *result.args_r.borrow_mut() = self.args_r.borrow().clone();
        *result.r_result.borrow_mut() = self.r_result.borrow().clone();
        *result.position_key.borrow_mut() = self.position_key.borrow().clone();
        result
    }

    /// RPython `HighLevelOp.dispatch(self)` (`rtyper.py:648-653`).
    pub fn dispatch(&self) -> RTypeResult {
        self.rtyper.translate_operation(self)
    }

    /// RPython `HighLevelOp.setup(self)` (rtyper.py:625-633).
    ///
    /// ```python
    /// def setup(self):
    ///     rtyper = self.rtyper
    ///     spaceop = self.spaceop
    ///     self.args_v   = list(spaceop.args)
    ///     self.args_s   = [rtyper.binding(a) for a in spaceop.args]
    ///     self.s_result = rtyper.binding(spaceop.result)
    ///     self.args_r   = [rtyper.getrepr(s_a) for s_a in self.args_s]
    ///     self.r_result = rtyper.getrepr(self.s_result)
    ///     rtyper.call_all_setups()
    /// ```
    ///
    /// `getrepr` arms for non-Impossible SomeValue variants currently
    /// surface `MissingRTypeOperation` — cascading port work fills
    /// them in (rint.rs, rfloat.rs, rpbc.rs, ...). Callers that hit
    /// the error surface know exactly which upstream module to land
    /// next.
    pub fn setup(&self) -> Result<(), TyperError> {
        let rtyper = &self.rtyper;
        *self.args_v.borrow_mut() = self.spaceop.args.clone();
        let args_s: Vec<SomeValue> = self
            .spaceop
            .args
            .iter()
            .map(|a| rtyper.binding(a))
            .collect();
        *self.s_result.borrow_mut() = Some(rtyper.binding(&self.spaceop.result));
        let mut args_r: Vec<Option<Arc<dyn Repr>>> = Vec::with_capacity(args_s.len());
        for s_a in &args_s {
            args_r.push(Some(rtyper.getrepr(s_a)?));
        }
        *self.args_r.borrow_mut() = args_r;
        let s_result_clone = self.s_result.borrow().clone();
        if let Some(s_r) = s_result_clone {
            *self.r_result.borrow_mut() = Some(rtyper.getrepr(&s_r)?);
        }
        *self.args_s.borrow_mut() = args_s;
        rtyper.call_all_setups()?;
        Ok(())
    }

    fn resolve_converted_to<'a>(
        &self,
        converted_to: ConvertedTo<'a>,
    ) -> Result<ResolvedConvertedTo<'a>, TyperError> {
        match converted_to {
            ConvertedTo::Repr(repr) => Ok(ResolvedConvertedTo::Borrowed(repr)),
            ConvertedTo::LowLevelType(lltype) => Ok(ResolvedConvertedTo::Owned(
                self.rtyper.getprimitiverepr(lltype)?,
            )),
        }
    }

    /// RPython `HighLevelOp.inputarg(self, converted_to, arg)`
    /// (`rtyper.py:655-673`).
    pub fn inputarg<'a>(
        &self,
        converted_to: impl Into<ConvertedTo<'a>>,
        arg: usize,
    ) -> Result<Hlvalue, TyperError> {
        let converted_to = self.resolve_converted_to(converted_to.into())?;
        let v = self.args_v.borrow()[arg].clone();
        if let Hlvalue::Constant(c) = &v {
            return inputconst(converted_to.as_repr(), &c.value).map(Hlvalue::Constant);
        }

        let s_binding = self.args_s.borrow()[arg].clone();
        if let Some(value) = s_binding.const_() {
            return inputconst(converted_to.as_repr(), value).map(Hlvalue::Constant);
        }

        let r_binding = self.args_r.borrow()[arg]
            .clone()
            .ok_or_else(|| TyperError::message("HighLevelOp.inputarg missing source repr"))?;
        self.llops
            .borrow_mut()
            .convertvar(v, r_binding.as_ref(), converted_to.as_repr())
    }

    /// RPython `HighLevelOp.inputconst = staticmethod(inputconst)`
    /// (`rtyper.py:675`).
    pub fn inputconst<'a>(
        converted_to: impl Into<ConvertedTo<'a>>,
        value: &ConstValue,
    ) -> Result<Constant, TyperError> {
        match converted_to.into() {
            ConvertedTo::Repr(repr) => inputconst(repr, value),
            ConvertedTo::LowLevelType(lltype) => inputconst_from_lltype(lltype, value),
        }
    }

    /// RPython `HighLevelOp.inputargs(self, *converted_to)`
    /// (`rtyper.py:677-685`).
    pub fn inputargs<'a>(
        &self,
        converted_to: Vec<ConvertedTo<'a>>,
    ) -> Result<Vec<Hlvalue>, TyperError> {
        if converted_to.len() != self.nb_args() {
            return Err(TyperError::message(format!(
                "operation argument count mismatch:\n'{}' has {} arguments, rtyper wants {}",
                self.spaceop.opname,
                self.nb_args(),
                converted_to.len()
            )));
        }
        let mut vars = Vec::with_capacity(converted_to.len());
        for (i, converted_to) in converted_to.into_iter().enumerate() {
            vars.push(self.inputarg(converted_to, i)?);
        }
        Ok(vars)
    }

    /// RPython `HighLevelOp.genop(self, opname, args_v, resulttype=None)`
    /// (`rtyper.py:687-688`).
    pub fn genop(
        &self,
        opname: &str,
        args_v: Vec<Hlvalue>,
        resulttype: GenopResult,
    ) -> Option<Hlvalue> {
        self.llops
            .borrow_mut()
            .genop(opname, args_v, resulttype)
            .map(Hlvalue::Variable)
    }

    /// RPython `HighLevelOp.gendirectcall(self, ll_function, *args_v)`
    /// (`rtyper.py:690-691`).
    pub fn gendirectcall(
        &self,
        ll_function: &LowLevelFunction,
        args_v: Vec<Hlvalue>,
    ) -> Result<Option<Hlvalue>, TyperError> {
        Ok(self
            .llops
            .borrow_mut()
            .gendirectcall(ll_function, args_v)?
            .map(Hlvalue::Variable))
    }

    /// RPython `HighLevelOp.has_implicit_exception(self, exc_cls)`
    /// (rtyper.py:713-729).
    ///
    pub fn has_implicit_exception(&self, exc_cls_name: &str) -> bool {
        let mut llops = self.llops.borrow_mut();
        if llops.llop_raising_exceptions.is_some() {
            panic!("already generated the llop that raises the exception");
        }
        if self.exceptionlinks.is_empty() {
            return false;
        }
        let exc_cls = HOST_ENV
            .lookup_exception_class(exc_cls_name)
            .unwrap_or_else(|| panic!("unknown exception class {exc_cls_name}"));
        let checked = llops
            .implicit_exceptions_checked
            .get_or_insert_with(Vec::new);
        let mut result = false;
        for link in &self.exceptionlinks {
            let exitcase = link.borrow().exitcase.clone();
            let Some(Hlvalue::Constant(Constant {
                value: ConstValue::HostObject(exit_cls),
                ..
            })) = exitcase
            else {
                continue;
            };
            if exc_cls.is_subclass_of(&exit_cls) {
                checked.push(exit_cls.qualname().to_string());
                result = true;
            }
        }
        result
    }

    /// RPython `HighLevelOp.exception_is_here(self)` (rtyper.py:731-745).
    ///
    /// Stores the index of the current llop as the "raising" llop.
    pub fn exception_is_here(&self) -> Result<(), TyperError> {
        let mut llops = self.llops.borrow_mut();
        llops._called_exception_is_here_or_cannot_occur = true;
        if llops.llop_raising_exceptions.is_some() {
            return Err(TyperError::message(
                "cannot catch an exception at more than one llop",
            ));
        }
        if self.exceptionlinks.is_empty() {
            return Ok(()); // rtyper.py:735-736
        }
        // upstream sanity check rtyper.py:737-744 — when
        // `has_implicit_exception` was called at least once on this
        // hop (`implicit_exceptions_checked` is Some, possibly
        // empty), every `exceptionlink.exitcase` must appear in the
        // checked list. Otherwise the graph catches an exception the
        // rtyper did not explicitly handle.
        if let Some(checked) = &llops.implicit_exceptions_checked {
            for link in &self.exceptionlinks {
                let exitcase = link.borrow().exitcase.clone();
                let qualname = match exitcase {
                    Some(Hlvalue::Constant(Constant {
                        value: ConstValue::HostObject(cls),
                        ..
                    })) => cls.qualname().to_string(),
                    _ => continue,
                };
                if !checked.contains(&qualname) {
                    return Err(TyperError::message(format!(
                        "the graph catches {qualname}, but the rtyper did not \
                         explicitely handle it"
                    )));
                }
            }
        }
        llops.llop_raising_exceptions = Some(LlopRaisingExceptions::Index(llops.ops.len()));
        Ok(())
    }

    /// RPython `HighLevelOp.exception_cannot_occur(self)`
    /// (rtyper.py:747-753).
    pub fn exception_cannot_occur(&self) -> Result<(), TyperError> {
        let mut llops = self.llops.borrow_mut();
        llops._called_exception_is_here_or_cannot_occur = true;
        if llops.llop_raising_exceptions.is_some() {
            return Err(TyperError::message(
                "cannot catch an exception at more than one llop",
            ));
        }
        if self.exceptionlinks.is_empty() {
            return Ok(());
        }
        llops.llop_raising_exceptions = Some(LlopRaisingExceptions::Removed);
        Ok(())
    }

    /// RPython `HighLevelOp.r_s_pop(self, index=-1)` (rtyper.py:693-696).
    ///
    /// ```python
    /// def r_s_pop(self, index=-1):
    ///     "Return and discard the argument with index position."
    ///     self.args_v.pop(index)
    ///     return self.args_r.pop(index), self.args_s.pop(index)
    /// ```
    pub fn r_s_pop(&self, index: Option<usize>) -> (Option<Arc<dyn Repr>>, SomeValue) {
        let mut v = self.args_v.borrow_mut();
        let mut r = self.args_r.borrow_mut();
        let mut s = self.args_s.borrow_mut();
        let i = index.unwrap_or_else(|| v.len().saturating_sub(1));
        v.remove(i);
        (r.remove(i), s.remove(i))
    }

    /// RPython `HighLevelOp.r_s_popfirstarg(self)` (rtyper.py:698-700).
    pub fn r_s_popfirstarg(&self) -> (Option<Arc<dyn Repr>>, SomeValue) {
        self.r_s_pop(Some(0))
    }

    /// RPython `HighLevelOp.swap_fst_snd_args(self)` (rtyper.py:708-711).
    pub fn swap_fst_snd_args(&self) {
        let mut v = self.args_v.borrow_mut();
        let mut s = self.args_s.borrow_mut();
        let mut r = self.args_r.borrow_mut();
        v.swap(0, 1);
        s.swap(0, 1);
        r.swap(0, 1);
    }

    /// RPython `HighLevelOp.v_s_insertfirstarg(self, v_newfirstarg,
    /// s_newfirstarg)` (rtyper.py:702-706).
    pub fn v_s_insertfirstarg(
        &self,
        v_newfirstarg: Hlvalue,
        s_newfirstarg: SomeValue,
    ) -> Result<(), TyperError> {
        let r_newfirstarg = self.rtyper.getrepr(&s_newfirstarg)?;
        self.args_v.borrow_mut().insert(0, v_newfirstarg);
        self.args_s.borrow_mut().insert(0, s_newfirstarg);
        self.args_r.borrow_mut().insert(0, Some(r_newfirstarg));
        Ok(())
    }
}

// ____________________________________________________________
// LowLevelOpList — `rtyper.py:783-871+`.

/// RPython `class LowLevelOpList(list)` (rtyper.py:783-809) — mutable
/// buffer of `SpaceOperation`s built during specialize_block.
///
/// Upstream subclasses `list`; pyre keeps the list in `ops` and
/// exposes Vec-style operations explicitly.
pub struct LowLevelOpList {
    /// RPython `self.rtyper = rtyper` (rtyper.py:794). Upstream's
    /// `__init__(self, rtyper=None, ...)` accepts `rtyper=None` so
    /// callers that only need `genop` (e.g.
    /// `removeassert.py:72 LowLevelOpList()`) can construct without
    /// the typer. The Rust port mirrors that with `Option<...>`;
    /// methods that actually read the typer (`gendirectcall`,
    /// `record_extra_call`, `record_extra_call_by_graph_id`) surface
    /// a structured `TyperError` when this is `None`.
    pub rtyper: Option<Rc<RPythonTyper>>,
    /// RPython `self.originalblock = originalblock` (rtyper.py:795).
    pub originalblock: Option<BlockRef>,
    /// RPython `LowLevelOpList.llop_raising_exceptions = None` class
    /// attribute (rtyper.py:790), set by
    /// `HighLevelOp.exception_is_here` / `exception_cannot_occur`.
    pub llop_raising_exceptions: Option<LlopRaisingExceptions>,
    /// RPython `LowLevelOpList.implicit_exceptions_checked = None`
    /// class attribute (rtyper.py:791), managed by
    /// `HighLevelOp.has_implicit_exception`.
    pub implicit_exceptions_checked: Option<Vec<String>>,
    /// Tracks whether the hop has run `exception_is_here` /
    /// `exception_cannot_occur` at least once — used by
    /// `rtyper.py:732,748` bookkeeping.
    pub _called_exception_is_here_or_cannot_occur: bool,
    /// The SpaceOperation buffer itself (upstream stores via
    /// `list.__init__`).
    pub ops: Vec<SpaceOperation>,
}

/// Rust carrier for RPython's `ll_function` argument to
/// `LowLevelOpList.gendirectcall`.
///
/// Upstream receives the actual Python helper function, annotates it with
/// the argument annotations, then obtains a callable graph. The Rust port
/// requires that graph to be present before emitting `direct_call`; a
/// signature-only helper surfaces `MissingRTypeOperation` instead of
/// fabricating a function pointer that upstream would not have produced.
#[derive(Clone, Debug)]
pub struct LowLevelFunction {
    pub name: String,
    pub args: Vec<LowLevelType>,
    pub result: LowLevelType,
    pub graph: Option<Rc<PyGraph>>,
}

impl LowLevelFunction {
    pub fn new(name: impl Into<String>, args: Vec<LowLevelType>, result: LowLevelType) -> Self {
        LowLevelFunction {
            name: name.into(),
            args,
            result,
            graph: None,
        }
    }

    pub fn from_pygraph(
        name: impl Into<String>,
        args: Vec<LowLevelType>,
        result: LowLevelType,
        graph: Rc<PyGraph>,
    ) -> Self {
        LowLevelFunction {
            name: name.into(),
            args,
            result,
            graph: Some(graph),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct LowLevelHelperKey {
    name: String,
    args: Vec<LowLevelType>,
    result: LowLevelType,
}

fn lowlevel_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
) -> Result<PyGraph, TyperError> {
    match name {
        "ll_check_chr" => lowlevel_range_check_helper_graph(name, args, result, 255),
        "ll_check_unichr" => lowlevel_range_check_helper_graph(name, args, result, 0x10ffff),
        "ll_int_abs_ovf" => lowlevel_int_min_unary_ovf_helper_graph(name, args, result, "int_abs"),
        "ll_int_neg_ovf" => lowlevel_int_min_unary_ovf_helper_graph(name, args, result, "int_neg"),
        "ll_int_lshift_ovf" => lowlevel_int_lshift_ovf_helper_graph(name, args, result),
        "ll_int_py_div" => lowlevel_int_py_div_helper_graph(name, args, result, false),
        "ll_int_py_div_nonnegargs" => lowlevel_int_py_div_helper_graph(name, args, result, true),
        "ll_int_py_mod" => lowlevel_int_py_mod_helper_graph(name, args, result, false),
        "ll_int_py_mod_nonnegargs" => lowlevel_int_py_mod_helper_graph(name, args, result, true),
        "ll_int_py_div_ovf" => {
            lowlevel_overflow_check_wrapper_graph(rtyper, name, args, result, "ll_int_py_div")
        }
        "ll_int_py_mod_ovf" => {
            lowlevel_overflow_check_wrapper_graph(rtyper, name, args, result, "ll_int_py_mod")
        }
        "ll_int_py_div_zer" => {
            lowlevel_zero_check_wrapper_graph(rtyper, name, args, result, "ll_int_py_div")
        }
        "ll_int_py_mod_zer" => {
            lowlevel_zero_check_wrapper_graph(rtyper, name, args, result, "ll_int_py_mod")
        }
        "ll_int_py_div_ovf_zer" => {
            lowlevel_zero_check_wrapper_graph(rtyper, name, args, result, "ll_int_py_div_ovf")
        }
        "ll_int_py_mod_ovf_zer" => {
            lowlevel_zero_check_wrapper_graph(rtyper, name, args, result, "ll_int_py_mod_ovf")
        }
        "ll_uint_py_div" => lowlevel_simple_binary_helper_graph(
            name,
            args,
            result,
            LowLevelType::Unsigned,
            "uint_floordiv",
        ),
        "ll_uint_py_mod" => lowlevel_simple_binary_helper_graph(
            name,
            args,
            result,
            LowLevelType::Unsigned,
            "uint_mod",
        ),
        "ll_uint_py_div_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_uint_py_div",
            LowLevelType::Unsigned,
            "uint_eq",
        ),
        "ll_uint_py_mod_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_uint_py_mod",
            LowLevelType::Unsigned,
            "uint_eq",
        ),
        "ll_ullong_py_div" => lowlevel_simple_binary_helper_graph(
            name,
            args,
            result,
            LowLevelType::UnsignedLongLong,
            "ullong_floordiv",
        ),
        "ll_ullong_py_mod" => lowlevel_simple_binary_helper_graph(
            name,
            args,
            result,
            LowLevelType::UnsignedLongLong,
            "ullong_mod",
        ),
        "ll_ullong_py_div_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_ullong_py_div",
            LowLevelType::UnsignedLongLong,
            "ullong_eq",
        ),
        "ll_ullong_py_mod_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_ullong_py_mod",
            LowLevelType::UnsignedLongLong,
            "ullong_eq",
        ),
        "ll_ulllong_py_div" => lowlevel_simple_binary_helper_graph(
            name,
            args,
            result,
            LowLevelType::UnsignedLongLongLong,
            "ulllong_floordiv",
        ),
        "ll_ulllong_py_mod" => lowlevel_simple_binary_helper_graph(
            name,
            args,
            result,
            LowLevelType::UnsignedLongLongLong,
            "ulllong_mod",
        ),
        "ll_ulllong_py_div_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_ulllong_py_div",
            LowLevelType::UnsignedLongLongLong,
            "ulllong_eq",
        ),
        "ll_ulllong_py_mod_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_ulllong_py_mod",
            LowLevelType::UnsignedLongLongLong,
            "ulllong_eq",
        ),
        "ll_llong_py_div" => lowlevel_signed_wide_py_div_helper_graph(
            name,
            args,
            result,
            LowLevelType::SignedLongLong,
            "llong",
            63,
        ),
        "ll_llong_py_mod" => lowlevel_signed_wide_py_mod_helper_graph(
            name,
            args,
            result,
            LowLevelType::SignedLongLong,
            "llong",
            63,
        ),
        "ll_llong_py_div_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_llong_py_div",
            LowLevelType::SignedLongLong,
            "llong_eq",
        ),
        "ll_llong_py_mod_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_llong_py_mod",
            LowLevelType::SignedLongLong,
            "llong_eq",
        ),
        "ll_lllong_py_div" => lowlevel_signed_wide_py_div_helper_graph(
            name,
            args,
            result,
            LowLevelType::SignedLongLongLong,
            "lllong",
            127,
        ),
        "ll_lllong_py_mod" => lowlevel_signed_wide_py_mod_helper_graph(
            name,
            args,
            result,
            LowLevelType::SignedLongLongLong,
            "lllong",
            127,
        ),
        "ll_lllong_py_div_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_lllong_py_div",
            LowLevelType::SignedLongLongLong,
            "lllong_eq",
        ),
        "ll_lllong_py_mod_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_lllong_py_mod",
            LowLevelType::SignedLongLongLong,
            "lllong_eq",
        ),
        "ll_issubclass" => lowlevel_issubclass_helper_graph(name, args, result),
        "ll_type" => lowlevel_type_helper_graph(name, args, result),
        _ => Ok(synthetic_lowlevel_helper_graph(name, args, result)),
    }
}

pub(crate) fn variable_with_lltype(name: &str, lltype: LowLevelType) -> Variable {
    let mut var = Variable::named(name);
    var.set_concretetype(Some(lltype.clone()));
    var.annotation
        .replace(Some(Rc::new(lltype_to_annotation(lltype))));
    var
}

pub(crate) fn constant_with_lltype(value: ConstValue, lltype: LowLevelType) -> Hlvalue {
    Hlvalue::Constant(Constant::with_concretetype(value, lltype))
}

pub(crate) fn helper_pygraph_from_graph(
    graph: FunctionGraph,
    argnames: Vec<String>,
    func: GraphFunc,
) -> PyGraph {
    PyGraph {
        graph: Rc::new(RefCell::new(graph)),
        func,
        signature: RefCell::new(Signature::new(argnames, None, None)),
        defaults: RefCell::new(Some(Vec::new())),
        access_directly: Cell::new(false),
    }
}

pub(crate) fn void_field_const(name: &str) -> Hlvalue {
    constant_with_lltype(ConstValue::byte_str(name), LowLevelType::Void)
}

fn lowlevel_issubclass_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
) -> Result<PyGraph, TyperError> {
    if args != [CLASSTYPE.clone(), CLASSTYPE.clone()] || result != &LowLevelType::Bool {
        return Err(TyperError::message(format!(
            "{name} expects (CLASSTYPE, CLASSTYPE) -> Bool, got ({args:?}) -> {result:?}"
        )));
    }

    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let subcls = variable_with_lltype("arg0", CLASSTYPE.clone());
    let cls = variable_with_lltype("arg1", CLASSTYPE.clone());
    let cls_min = variable_with_lltype("cls_min", LowLevelType::Signed);
    let subcls_min = variable_with_lltype("subcls_min", LowLevelType::Signed);
    let cls_max = variable_with_lltype("cls_max", LowLevelType::Signed);
    let is_subclass = variable_with_lltype("result", LowLevelType::Bool);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);

    let startblock = Block::shared(vec![
        Hlvalue::Variable(subcls.clone()),
        Hlvalue::Variable(cls.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(cls.clone()),
            void_field_const("subclassrange_min"),
        ],
        Hlvalue::Variable(cls_min.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(subcls.clone()),
            void_field_const("subclassrange_min"),
        ],
        Hlvalue::Variable(subcls_min.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(cls),
            void_field_const("subclassrange_max"),
        ],
        Hlvalue::Variable(cls_max.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_between",
        vec![
            Hlvalue::Variable(cls_min),
            Hlvalue::Variable(subcls_min),
            Hlvalue::Variable(cls_max),
        ],
        Hlvalue::Variable(is_subclass.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(is_subclass)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_type_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
) -> Result<PyGraph, TyperError> {
    if args != [OBJECTPTR.clone()] || result != &CLASSTYPE.clone() {
        return Err(TyperError::message(format!(
            "{name} expects (OBJECTPTR) -> CLASSTYPE, got ({args:?}) -> {result:?}"
        )));
    }

    let argnames = vec!["arg0".to_string()];
    let excinst = variable_with_lltype("arg0", OBJECTPTR.clone());
    let obj = variable_with_lltype("obj", OBJECTPTR.clone());
    let typeptr = variable_with_lltype("result", CLASSTYPE.clone());
    let return_var = variable_with_lltype("result", CLASSTYPE.clone());

    let startblock = Block::shared(vec![Hlvalue::Variable(excinst.clone())]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_pointer",
        vec![Hlvalue::Variable(excinst)],
        Hlvalue::Variable(obj.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(obj), void_field_const("typeptr")],
        Hlvalue::Variable(typeptr.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(typeptr)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

pub(crate) fn exception_args(exc_name: &str) -> Result<Vec<Hlvalue>, TyperError> {
    let exc_cls = HOST_ENV
        .lookup_exception_class(exc_name)
        .or_else(|| {
            crate::annotator::exception::standard_exception_classes()
                .into_iter()
                .find(|cls| cls.qualname() == exc_name)
        })
        .ok_or_else(|| TyperError::message(format!("missing host {exc_name} exception class")))?;
    let exc_instance = HOST_ENV
        .lookup_standard_exception_instance(exc_name)
        .or_else(|| exc_cls.reusable_prebuilt_instance())
        .ok_or_else(|| {
            TyperError::message(format!("missing host {exc_name} exception instance"))
        })?;
    Ok(vec![
        Hlvalue::Constant(Constant::new(ConstValue::HostObject(exc_cls))),
        Hlvalue::Constant(Constant::new(ConstValue::HostObject(exc_instance))),
    ])
}

fn lowlevel_range_check_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    upper_bound: i64,
) -> Result<PyGraph, TyperError> {
    if args != [LowLevelType::Signed] || result != &LowLevelType::Void {
        return Err(TyperError::message(format!(
            "{name} expects (Signed) -> Void, got ({args:?}) -> {result:?}"
        )));
    }

    // RPython rtyper/rint.py:
    //   if 0 <= n <= <bound>: return
    //   raise ValueError
    let argnames = vec!["arg0".to_string()];
    let n0 = variable_with_lltype("arg0", LowLevelType::Signed);
    let n1 = variable_with_lltype("n", LowLevelType::Signed);
    let ge0 = variable_with_lltype("ge0", LowLevelType::Bool);
    let lemax = variable_with_lltype("lemax", LowLevelType::Bool);
    let return_var = variable_with_lltype("result", LowLevelType::Void);

    let startblock = Block::shared(vec![Hlvalue::Variable(n0.clone())]);
    let check_upper = Block::shared(vec![Hlvalue::Variable(n1.clone())]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let exc_args = exception_args("ValueError")?;

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![
            Hlvalue::Variable(n0.clone()),
            constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
        ],
        Hlvalue::Variable(ge0.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(ge0.clone()));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(n0)],
            Some(check_upper.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    check_upper
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_le",
            vec![
                Hlvalue::Variable(n1),
                constant_with_lltype(ConstValue::Int(upper_bound), LowLevelType::Signed),
            ],
            Hlvalue::Variable(lemax.clone()),
        ));
    check_upper.borrow_mut().exitswitch = Some(Hlvalue::Variable(lemax));
    check_upper.closeblock(vec![
        Link::new(
            vec![constant_with_lltype(ConstValue::None, LowLevelType::Void)],
            Some(graph.returnblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_int_min_unary_ovf_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    opname: &str,
) -> Result<PyGraph, TyperError> {
    if args != [LowLevelType::Signed] || result != &LowLevelType::Signed {
        return Err(TyperError::message(format!(
            "{name} expects (Signed) -> Signed, got ({args:?}) -> {result:?}"
        )));
    }

    // RPython rtyper/rint.py:
    //   if x == INT_MIN: raise OverflowError
    //   return -x       # ll_int_neg_ovf
    //   return abs(x)   # ll_int_abs_ovf
    let argnames = vec!["arg0".to_string()];
    let x0 = variable_with_lltype("arg0", LowLevelType::Signed);
    let x1 = variable_with_lltype("x", LowLevelType::Signed);
    let is_min = variable_with_lltype("is_min", LowLevelType::Bool);
    let op_result = variable_with_lltype("result", LowLevelType::Signed);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);

    let startblock = Block::shared(vec![Hlvalue::Variable(x0.clone())]);
    let compute = Block::shared(vec![Hlvalue::Variable(x1.clone())]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let exc_args = exception_args("OverflowError")?;

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![
            Hlvalue::Variable(x0.clone()),
            constant_with_lltype(ConstValue::Int(i64::MIN), LowLevelType::Signed),
        ],
        Hlvalue::Variable(is_min.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_min));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(x0)],
            Some(compute.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    compute.borrow_mut().operations.push(SpaceOperation::new(
        opname,
        vec![Hlvalue::Variable(x1)],
        Hlvalue::Variable(op_result.clone()),
    ));
    compute.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(op_result)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_int_lshift_ovf_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
) -> Result<PyGraph, TyperError> {
    if args != [LowLevelType::Signed, LowLevelType::Signed] || result != &LowLevelType::Signed {
        return Err(TyperError::message(format!(
            "{name} expects (Signed, Signed) -> Signed, got ({args:?}) -> {result:?}"
        )));
    }

    // RPython rtyper/rint.py:
    //   result = x << y
    //   if (result >> y) != x: raise OverflowError
    //   return result
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x = variable_with_lltype("arg0", LowLevelType::Signed);
    let y = variable_with_lltype("arg1", LowLevelType::Signed);
    let shifted = variable_with_lltype("result", LowLevelType::Signed);
    let shifted_back = variable_with_lltype("shifted_back", LowLevelType::Signed);
    let overflowed = variable_with_lltype("overflowed", LowLevelType::Bool);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);

    let startblock = Block::shared(vec![
        Hlvalue::Variable(x.clone()),
        Hlvalue::Variable(y.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let exc_args = exception_args("OverflowError")?;

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lshift",
        vec![Hlvalue::Variable(x.clone()), Hlvalue::Variable(y.clone())],
        Hlvalue::Variable(shifted.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_rshift",
        vec![
            Hlvalue::Variable(shifted.clone()),
            Hlvalue::Variable(y.clone()),
        ],
        Hlvalue::Variable(shifted_back.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ne",
        vec![Hlvalue::Variable(shifted_back), Hlvalue::Variable(x)],
        Hlvalue::Variable(overflowed.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(overflowed));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(shifted)],
            Some(graph.returnblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_int_py_div_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    nonnegargs: bool,
) -> Result<PyGraph, TyperError> {
    if args != [LowLevelType::Signed, LowLevelType::Signed] || result != &LowLevelType::Signed {
        return Err(TyperError::message(format!(
            "{name} expects (Signed, Signed) -> Signed, got ({args:?}) -> {result:?}"
        )));
    }

    // RPython rtyper/rint.py ll_int_py_div:
    //   r = llop.int_floordiv(Signed, x, y)
    //   p = r * y
    //   if y < 0: u = p - x
    //   else:     u = x - p
    //   return r + (u >> INT_BITS_1)
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x = variable_with_lltype("arg0", LowLevelType::Signed);
    let y = variable_with_lltype("arg1", LowLevelType::Signed);
    let r = variable_with_lltype("r", LowLevelType::Signed);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(x.clone()),
        Hlvalue::Variable(y.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_floordiv",
        vec![Hlvalue::Variable(x.clone()), Hlvalue::Variable(y.clone())],
        Hlvalue::Variable(r.clone()),
    ));

    if nonnegargs {
        let ok = variable_with_lltype("ok", LowLevelType::Bool);
        let debug_result = variable_with_lltype("debug_result", LowLevelType::Void);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_ge",
            vec![
                Hlvalue::Variable(r.clone()),
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
            ],
            Hlvalue::Variable(ok.clone()),
        ));
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "debug_assert",
            vec![
                Hlvalue::Variable(ok),
                constant_with_lltype(
                    ConstValue::byte_str("int_py_div_nonnegargs(): one arg is negative"),
                    LowLevelType::Void,
                ),
            ],
            Hlvalue::Variable(debug_result),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(r)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
    } else {
        let p = variable_with_lltype("p", LowLevelType::Signed);
        let y_is_neg = variable_with_lltype("y_is_neg", LowLevelType::Bool);
        let p_neg = variable_with_lltype("p", LowLevelType::Signed);
        let x_neg = variable_with_lltype("x", LowLevelType::Signed);
        let r_neg = variable_with_lltype("r", LowLevelType::Signed);
        let x_pos = variable_with_lltype("x", LowLevelType::Signed);
        let p_pos = variable_with_lltype("p", LowLevelType::Signed);
        let r_pos = variable_with_lltype("r", LowLevelType::Signed);
        let u_neg = variable_with_lltype("u", LowLevelType::Signed);
        let u_pos = variable_with_lltype("u", LowLevelType::Signed);
        let r_join = variable_with_lltype("r", LowLevelType::Signed);
        let u_join = variable_with_lltype("u", LowLevelType::Signed);
        let shifted = variable_with_lltype("shifted", LowLevelType::Signed);
        let div_result = variable_with_lltype("result", LowLevelType::Signed);

        let neg_block = Block::shared(vec![
            Hlvalue::Variable(p_neg.clone()),
            Hlvalue::Variable(x_neg.clone()),
            Hlvalue::Variable(r_neg.clone()),
        ]);
        let pos_block = Block::shared(vec![
            Hlvalue::Variable(x_pos.clone()),
            Hlvalue::Variable(p_pos.clone()),
            Hlvalue::Variable(r_pos.clone()),
        ]);
        let join = Block::shared(vec![
            Hlvalue::Variable(r_join.clone()),
            Hlvalue::Variable(u_join.clone()),
        ]);

        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_mul",
            vec![Hlvalue::Variable(r.clone()), Hlvalue::Variable(y.clone())],
            Hlvalue::Variable(p.clone()),
        ));
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(y),
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
            ],
            Hlvalue::Variable(y_is_neg.clone()),
        ));
        startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(y_is_neg));
        startblock.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(p.clone()),
                    Hlvalue::Variable(x.clone()),
                    Hlvalue::Variable(r.clone()),
                ],
                Some(neg_block.clone()),
                Some(constant_with_lltype(
                    ConstValue::Bool(true),
                    LowLevelType::Bool,
                )),
            )
            .into_ref(),
            Link::new(
                vec![
                    Hlvalue::Variable(x),
                    Hlvalue::Variable(p),
                    Hlvalue::Variable(r),
                ],
                Some(pos_block.clone()),
                Some(constant_with_lltype(
                    ConstValue::Bool(false),
                    LowLevelType::Bool,
                )),
            )
            .into_ref(),
        ]);

        neg_block.borrow_mut().operations.push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Variable(p_neg), Hlvalue::Variable(x_neg)],
            Hlvalue::Variable(u_neg.clone()),
        ));
        neg_block.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(r_neg), Hlvalue::Variable(u_neg)],
                Some(join.clone()),
                None,
            )
            .into_ref(),
        ]);

        pos_block.borrow_mut().operations.push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Variable(x_pos), Hlvalue::Variable(p_pos)],
            Hlvalue::Variable(u_pos.clone()),
        ));
        pos_block.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(r_pos), Hlvalue::Variable(u_pos)],
                Some(join.clone()),
                None,
            )
            .into_ref(),
        ]);

        join.borrow_mut().operations.push(SpaceOperation::new(
            "int_rshift",
            vec![
                Hlvalue::Variable(u_join),
                constant_with_lltype(ConstValue::Int(63), LowLevelType::Signed),
            ],
            Hlvalue::Variable(shifted.clone()),
        ));
        join.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(r_join), Hlvalue::Variable(shifted)],
            Hlvalue::Variable(div_result.clone()),
        ));
        join.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(div_result)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
    }

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_int_py_mod_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    nonnegargs: bool,
) -> Result<PyGraph, TyperError> {
    if args != [LowLevelType::Signed, LowLevelType::Signed] || result != &LowLevelType::Signed {
        return Err(TyperError::message(format!(
            "{name} expects (Signed, Signed) -> Signed, got ({args:?}) -> {result:?}"
        )));
    }

    // RPython rtyper/rint.py ll_int_py_mod:
    //   r = llop.int_mod(Signed, x, y)
    //   if y < 0: u = -r
    //   else:     u = r
    //   return r + (y & (u >> INT_BITS_1))
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x = variable_with_lltype("arg0", LowLevelType::Signed);
    let y = variable_with_lltype("arg1", LowLevelType::Signed);
    let r = variable_with_lltype("r", LowLevelType::Signed);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(x.clone()),
        Hlvalue::Variable(y.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_mod",
        vec![Hlvalue::Variable(x), Hlvalue::Variable(y.clone())],
        Hlvalue::Variable(r.clone()),
    ));

    if nonnegargs {
        let ok = variable_with_lltype("ok", LowLevelType::Bool);
        let debug_result = variable_with_lltype("debug_result", LowLevelType::Void);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_ge",
            vec![
                Hlvalue::Variable(r.clone()),
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
            ],
            Hlvalue::Variable(ok.clone()),
        ));
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "debug_assert",
            vec![
                Hlvalue::Variable(ok),
                constant_with_lltype(
                    ConstValue::byte_str("int_py_mod_nonnegargs(): one arg is negative"),
                    LowLevelType::Void,
                ),
            ],
            Hlvalue::Variable(debug_result),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(r)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
    } else {
        let y_is_neg = variable_with_lltype("y_is_neg", LowLevelType::Bool);
        let r_neg = variable_with_lltype("r", LowLevelType::Signed);
        let y_neg = variable_with_lltype("y", LowLevelType::Signed);
        let r_pos = variable_with_lltype("r", LowLevelType::Signed);
        let y_pos = variable_with_lltype("y", LowLevelType::Signed);
        let u_neg = variable_with_lltype("u", LowLevelType::Signed);
        let r_join = variable_with_lltype("r", LowLevelType::Signed);
        let y_join = variable_with_lltype("y", LowLevelType::Signed);
        let u_join = variable_with_lltype("u", LowLevelType::Signed);
        let shifted = variable_with_lltype("shifted", LowLevelType::Signed);
        let masked = variable_with_lltype("masked", LowLevelType::Signed);
        let mod_result = variable_with_lltype("result", LowLevelType::Signed);

        let neg_block = Block::shared(vec![
            Hlvalue::Variable(r_neg.clone()),
            Hlvalue::Variable(y_neg.clone()),
        ]);
        let pos_block = Block::shared(vec![
            Hlvalue::Variable(r_pos.clone()),
            Hlvalue::Variable(y_pos.clone()),
        ]);
        let join = Block::shared(vec![
            Hlvalue::Variable(r_join.clone()),
            Hlvalue::Variable(y_join.clone()),
            Hlvalue::Variable(u_join.clone()),
        ]);

        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(y.clone()),
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
            ],
            Hlvalue::Variable(y_is_neg.clone()),
        ));
        startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(y_is_neg));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(r.clone()), Hlvalue::Variable(y.clone())],
                Some(neg_block.clone()),
                Some(constant_with_lltype(
                    ConstValue::Bool(true),
                    LowLevelType::Bool,
                )),
            )
            .into_ref(),
            Link::new(
                vec![Hlvalue::Variable(r.clone()), Hlvalue::Variable(y.clone())],
                Some(pos_block.clone()),
                Some(constant_with_lltype(
                    ConstValue::Bool(false),
                    LowLevelType::Bool,
                )),
            )
            .into_ref(),
        ]);

        neg_block.borrow_mut().operations.push(SpaceOperation::new(
            "int_neg",
            vec![Hlvalue::Variable(r_neg.clone())],
            Hlvalue::Variable(u_neg.clone()),
        ));
        neg_block.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(r_neg),
                    Hlvalue::Variable(y_neg),
                    Hlvalue::Variable(u_neg),
                ],
                Some(join.clone()),
                None,
            )
            .into_ref(),
        ]);

        pos_block.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(r_pos.clone()),
                    Hlvalue::Variable(y_pos),
                    Hlvalue::Variable(r_pos),
                ],
                Some(join.clone()),
                None,
            )
            .into_ref(),
        ]);

        join.borrow_mut().operations.push(SpaceOperation::new(
            "int_rshift",
            vec![
                Hlvalue::Variable(u_join),
                constant_with_lltype(ConstValue::Int(63), LowLevelType::Signed),
            ],
            Hlvalue::Variable(shifted.clone()),
        ));
        join.borrow_mut().operations.push(SpaceOperation::new(
            "int_and",
            vec![Hlvalue::Variable(y_join), Hlvalue::Variable(shifted)],
            Hlvalue::Variable(masked.clone()),
        ));
        join.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(r_join), Hlvalue::Variable(masked)],
            Hlvalue::Variable(mod_result.clone()),
        ));
        join.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(mod_result)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
    }

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_zero_check_wrapper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    callee_name: &str,
) -> Result<PyGraph, TyperError> {
    if args != [LowLevelType::Signed, LowLevelType::Signed] || result != &LowLevelType::Signed {
        return Err(TyperError::message(format!(
            "{name} expects (Signed, Signed) -> Signed, got ({args:?}) -> {result:?}"
        )));
    }

    // RPython rtyper/rint.py:
    //   if y == 0: raise ZeroDivisionError
    //   return ll_int_py_div(x, y) / ll_int_py_mod(x, y)
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x0 = variable_with_lltype("arg0", LowLevelType::Signed);
    let y0 = variable_with_lltype("arg1", LowLevelType::Signed);
    let x1 = variable_with_lltype("x", LowLevelType::Signed);
    let y1 = variable_with_lltype("y", LowLevelType::Signed);
    let is_zero = variable_with_lltype("is_zero", LowLevelType::Bool);
    let call_result = variable_with_lltype("result", LowLevelType::Signed);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);

    let startblock = Block::shared(vec![
        Hlvalue::Variable(x0.clone()),
        Hlvalue::Variable(y0.clone()),
    ]);
    let callblock = Block::shared(vec![
        Hlvalue::Variable(x1.clone()),
        Hlvalue::Variable(y1.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let exc_args = exception_args("ZeroDivisionError")?;

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![
            Hlvalue::Variable(y0.clone()),
            constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
        ],
        Hlvalue::Variable(is_zero.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_zero));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(x0), Hlvalue::Variable(y0)],
            Some(callblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    let callee = rtyper.lowlevel_helper_function(
        callee_name,
        vec![LowLevelType::Signed, LowLevelType::Signed],
        LowLevelType::Signed,
    )?;
    callblock
        .borrow_mut()
        .operations
        .push(direct_call_operation(
            rtyper,
            &callee,
            vec![Hlvalue::Variable(x1), Hlvalue::Variable(y1)],
            call_result.clone(),
        )?);
    callblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(call_result)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_signed_wide_py_div_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    lltype: LowLevelType,
    opprefix: &str,
    bits_minus_one: i64,
) -> Result<PyGraph, TyperError> {
    if args != [lltype.clone(), lltype.clone()] || result != &lltype {
        return Err(TyperError::message(format!(
            "{name} expects ({0}, {0}) -> {0}, got ({args:?}) -> {result:?}",
            lltype.short_name()
        )));
    }

    // RPython rtyper/rint.py ll_llong_py_div / ll_lllong_py_div:
    //   r = llop.<prefix>_floordiv(TYPE, x, y)
    //   p = r * y
    //   if y < 0: u = p - x
    //   else:     u = x - p
    //   return r + (u >> <BITS_1>)
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x = variable_with_lltype("arg0", lltype.clone());
    let y = variable_with_lltype("arg1", lltype.clone());
    let r = variable_with_lltype("r", lltype.clone());
    let p = variable_with_lltype("p", lltype.clone());
    let y_is_neg = variable_with_lltype("y_is_neg", LowLevelType::Bool);
    let p_neg = variable_with_lltype("p", lltype.clone());
    let x_neg = variable_with_lltype("x", lltype.clone());
    let r_neg = variable_with_lltype("r", lltype.clone());
    let x_pos = variable_with_lltype("x", lltype.clone());
    let p_pos = variable_with_lltype("p", lltype.clone());
    let r_pos = variable_with_lltype("r", lltype.clone());
    let u_neg = variable_with_lltype("u", lltype.clone());
    let u_pos = variable_with_lltype("u", lltype.clone());
    let r_join = variable_with_lltype("r", lltype.clone());
    let u_join = variable_with_lltype("u", lltype.clone());
    let shifted = variable_with_lltype("shifted", lltype.clone());
    let div_result = variable_with_lltype("result", lltype.clone());
    let return_var = variable_with_lltype("result", lltype.clone());

    let startblock = Block::shared(vec![
        Hlvalue::Variable(x.clone()),
        Hlvalue::Variable(y.clone()),
    ]);
    let neg_block = Block::shared(vec![
        Hlvalue::Variable(p_neg.clone()),
        Hlvalue::Variable(x_neg.clone()),
        Hlvalue::Variable(r_neg.clone()),
    ]);
    let pos_block = Block::shared(vec![
        Hlvalue::Variable(x_pos.clone()),
        Hlvalue::Variable(p_pos.clone()),
        Hlvalue::Variable(r_pos.clone()),
    ]);
    let join = Block::shared(vec![
        Hlvalue::Variable(r_join.clone()),
        Hlvalue::Variable(u_join.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_floordiv"),
        vec![Hlvalue::Variable(x.clone()), Hlvalue::Variable(y.clone())],
        Hlvalue::Variable(r.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_mul"),
        vec![Hlvalue::Variable(r.clone()), Hlvalue::Variable(y.clone())],
        Hlvalue::Variable(p.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_lt"),
        vec![
            Hlvalue::Variable(y),
            constant_with_lltype(ConstValue::Int(0), lltype.clone()),
        ],
        Hlvalue::Variable(y_is_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(y_is_neg));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(p.clone()),
                Hlvalue::Variable(x.clone()),
                Hlvalue::Variable(r.clone()),
            ],
            Some(neg_block.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(x),
                Hlvalue::Variable(p),
                Hlvalue::Variable(r),
            ],
            Some(pos_block.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    neg_block.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_sub"),
        vec![Hlvalue::Variable(p_neg), Hlvalue::Variable(x_neg)],
        Hlvalue::Variable(u_neg.clone()),
    ));
    neg_block.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(r_neg), Hlvalue::Variable(u_neg)],
            Some(join.clone()),
            None,
        )
        .into_ref(),
    ]);

    pos_block.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_sub"),
        vec![Hlvalue::Variable(x_pos), Hlvalue::Variable(p_pos)],
        Hlvalue::Variable(u_pos.clone()),
    ));
    pos_block.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(r_pos), Hlvalue::Variable(u_pos)],
            Some(join.clone()),
            None,
        )
        .into_ref(),
    ]);

    join.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_rshift"),
        vec![
            Hlvalue::Variable(u_join),
            constant_with_lltype(ConstValue::Int(bits_minus_one), LowLevelType::Signed),
        ],
        Hlvalue::Variable(shifted.clone()),
    ));
    join.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_add"),
        vec![Hlvalue::Variable(r_join), Hlvalue::Variable(shifted)],
        Hlvalue::Variable(div_result.clone()),
    ));
    join.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(div_result)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_signed_wide_py_mod_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    lltype: LowLevelType,
    opprefix: &str,
    bits_minus_one: i64,
) -> Result<PyGraph, TyperError> {
    if args != [lltype.clone(), lltype.clone()] || result != &lltype {
        return Err(TyperError::message(format!(
            "{name} expects ({0}, {0}) -> {0}, got ({args:?}) -> {result:?}",
            lltype.short_name()
        )));
    }

    // RPython rtyper/rint.py ll_llong_py_mod / ll_lllong_py_mod:
    //   r = llop.<prefix>_mod(TYPE, x, y)
    //   if y < 0: u = -r
    //   else:     u = r
    //   return r + (y & (u >> <BITS_1>))
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x = variable_with_lltype("arg0", lltype.clone());
    let y = variable_with_lltype("arg1", lltype.clone());
    let r = variable_with_lltype("r", lltype.clone());
    let y_is_neg = variable_with_lltype("y_is_neg", LowLevelType::Bool);
    let r_neg = variable_with_lltype("r", lltype.clone());
    let y_neg = variable_with_lltype("y", lltype.clone());
    let r_pos = variable_with_lltype("r", lltype.clone());
    let y_pos = variable_with_lltype("y", lltype.clone());
    let u_neg = variable_with_lltype("u", lltype.clone());
    let r_join = variable_with_lltype("r", lltype.clone());
    let y_join = variable_with_lltype("y", lltype.clone());
    let u_join = variable_with_lltype("u", lltype.clone());
    let shifted = variable_with_lltype("shifted", lltype.clone());
    let masked = variable_with_lltype("masked", lltype.clone());
    let mod_result = variable_with_lltype("result", lltype.clone());
    let return_var = variable_with_lltype("result", lltype.clone());

    let startblock = Block::shared(vec![
        Hlvalue::Variable(x.clone()),
        Hlvalue::Variable(y.clone()),
    ]);
    let neg_block = Block::shared(vec![
        Hlvalue::Variable(r_neg.clone()),
        Hlvalue::Variable(y_neg.clone()),
    ]);
    let pos_block = Block::shared(vec![
        Hlvalue::Variable(r_pos.clone()),
        Hlvalue::Variable(y_pos.clone()),
    ]);
    let join = Block::shared(vec![
        Hlvalue::Variable(r_join.clone()),
        Hlvalue::Variable(y_join.clone()),
        Hlvalue::Variable(u_join.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_mod"),
        vec![Hlvalue::Variable(x), Hlvalue::Variable(y.clone())],
        Hlvalue::Variable(r.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_lt"),
        vec![
            Hlvalue::Variable(y.clone()),
            constant_with_lltype(ConstValue::Int(0), lltype.clone()),
        ],
        Hlvalue::Variable(y_is_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(y_is_neg));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(r.clone()), Hlvalue::Variable(y.clone())],
            Some(neg_block.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(r.clone()), Hlvalue::Variable(y)],
            Some(pos_block.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    neg_block.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_neg"),
        vec![Hlvalue::Variable(r_neg.clone())],
        Hlvalue::Variable(u_neg.clone()),
    ));
    neg_block.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(r_neg),
                Hlvalue::Variable(y_neg),
                Hlvalue::Variable(u_neg),
            ],
            Some(join.clone()),
            None,
        )
        .into_ref(),
    ]);

    pos_block.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(r_pos.clone()),
                Hlvalue::Variable(y_pos),
                Hlvalue::Variable(r_pos),
            ],
            Some(join.clone()),
            None,
        )
        .into_ref(),
    ]);

    join.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_rshift"),
        vec![
            Hlvalue::Variable(u_join),
            constant_with_lltype(ConstValue::Int(bits_minus_one), LowLevelType::Signed),
        ],
        Hlvalue::Variable(shifted.clone()),
    ));
    join.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_and"),
        vec![Hlvalue::Variable(y_join), Hlvalue::Variable(shifted)],
        Hlvalue::Variable(masked.clone()),
    ));
    join.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_add"),
        vec![Hlvalue::Variable(r_join), Hlvalue::Variable(masked)],
        Hlvalue::Variable(mod_result.clone()),
    ));
    join.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(mod_result)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_simple_binary_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    lltype: LowLevelType,
    opname: &str,
) -> Result<PyGraph, TyperError> {
    if args != [lltype.clone(), lltype.clone()] || result != &lltype {
        return Err(TyperError::message(format!(
            "{name} expects ({0}, {0}) -> {0}, got ({args:?}) -> {result:?}",
            lltype.short_name()
        )));
    }

    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x = variable_with_lltype("arg0", lltype.clone());
    let y = variable_with_lltype("arg1", lltype.clone());
    let op_result = variable_with_lltype("result", lltype.clone());
    let return_var = variable_with_lltype("result", lltype);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(x.clone()),
        Hlvalue::Variable(y.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        opname,
        vec![Hlvalue::Variable(x), Hlvalue::Variable(y)],
        Hlvalue::Variable(op_result.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(op_result)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_typed_zero_check_wrapper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    callee_name: &str,
    lltype: LowLevelType,
    eq_opname: &str,
) -> Result<PyGraph, TyperError> {
    if args != [lltype.clone(), lltype.clone()] || result != &lltype {
        return Err(TyperError::message(format!(
            "{name} expects ({0}, {0}) -> {0}, got ({args:?}) -> {result:?}",
            lltype.short_name()
        )));
    }

    // RPython rtyper/rint.py unsigned helpers:
    //   if y == 0: raise ZeroDivisionError
    //   return ll_uint_py_div(x, y) / ll_uint_py_mod(x, y)
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x0 = variable_with_lltype("arg0", lltype.clone());
    let y0 = variable_with_lltype("arg1", lltype.clone());
    let x1 = variable_with_lltype("x", lltype.clone());
    let y1 = variable_with_lltype("y", lltype.clone());
    let is_zero = variable_with_lltype("is_zero", LowLevelType::Bool);
    let call_result = variable_with_lltype("result", lltype.clone());
    let return_var = variable_with_lltype("result", lltype.clone());

    let startblock = Block::shared(vec![
        Hlvalue::Variable(x0.clone()),
        Hlvalue::Variable(y0.clone()),
    ]);
    let callblock = Block::shared(vec![
        Hlvalue::Variable(x1.clone()),
        Hlvalue::Variable(y1.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let exc_args = exception_args("ZeroDivisionError")?;

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        eq_opname,
        vec![
            Hlvalue::Variable(y0.clone()),
            constant_with_lltype(ConstValue::Int(0), lltype.clone()),
        ],
        Hlvalue::Variable(is_zero.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_zero));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(x0), Hlvalue::Variable(y0)],
            Some(callblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    let callee = rtyper.lowlevel_helper_function(
        callee_name,
        vec![lltype.clone(), lltype.clone()],
        lltype,
    )?;
    callblock
        .borrow_mut()
        .operations
        .push(direct_call_operation(
            rtyper,
            &callee,
            vec![Hlvalue::Variable(x1), Hlvalue::Variable(y1)],
            call_result.clone(),
        )?);
    callblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(call_result)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_overflow_check_wrapper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    callee_name: &str,
) -> Result<PyGraph, TyperError> {
    if args != [LowLevelType::Signed, LowLevelType::Signed] || result != &LowLevelType::Signed {
        return Err(TyperError::message(format!(
            "{name} expects (Signed, Signed) -> Signed, got ({args:?}) -> {result:?}"
        )));
    }

    // RPython rtyper/rint.py:
    //   if (x == -sys.maxint - 1) & (y == -1): raise OverflowError
    //   return ll_int_py_div(x, y) / ll_int_py_mod(x, y)
    //
    // Keep the bitwise `&` shape: RPython intentionally does not
    // short-circuit here so the JIT can see a single combined guard.
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x0 = variable_with_lltype("arg0", LowLevelType::Signed);
    let y0 = variable_with_lltype("arg1", LowLevelType::Signed);
    let x1 = variable_with_lltype("x", LowLevelType::Signed);
    let y1 = variable_with_lltype("y", LowLevelType::Signed);
    let x_is_min = variable_with_lltype("x_is_min", LowLevelType::Bool);
    let y_is_minus_one = variable_with_lltype("y_is_minus_one", LowLevelType::Bool);
    let overflowed = variable_with_lltype("overflowed", LowLevelType::Bool);
    let call_result = variable_with_lltype("result", LowLevelType::Signed);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);

    let startblock = Block::shared(vec![
        Hlvalue::Variable(x0.clone()),
        Hlvalue::Variable(y0.clone()),
    ]);
    let callblock = Block::shared(vec![
        Hlvalue::Variable(x1.clone()),
        Hlvalue::Variable(y1.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let exc_args = exception_args("OverflowError")?;

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![
            Hlvalue::Variable(x0.clone()),
            constant_with_lltype(ConstValue::Int(i64::MIN), LowLevelType::Signed),
        ],
        Hlvalue::Variable(x_is_min.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![
            Hlvalue::Variable(y0.clone()),
            constant_with_lltype(ConstValue::Int(-1), LowLevelType::Signed),
        ],
        Hlvalue::Variable(y_is_minus_one.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_and",
        vec![
            Hlvalue::Variable(x_is_min),
            Hlvalue::Variable(y_is_minus_one),
        ],
        Hlvalue::Variable(overflowed.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(overflowed));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(x0), Hlvalue::Variable(y0)],
            Some(callblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    let callee = rtyper.lowlevel_helper_function(
        callee_name,
        vec![LowLevelType::Signed, LowLevelType::Signed],
        LowLevelType::Signed,
    )?;
    callblock
        .borrow_mut()
        .operations
        .push(direct_call_operation(
            rtyper,
            &callee,
            vec![Hlvalue::Variable(x1), Hlvalue::Variable(y1)],
            call_result.clone(),
        )?);
    callblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(call_result)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn direct_call_operation(
    rtyper: &RPythonTyper,
    ll_function: &LowLevelFunction,
    args_v: Vec<Hlvalue>,
    result: Variable,
) -> Result<SpaceOperation, TyperError> {
    let graph = ll_function.graph.as_ref().ok_or_else(|| {
        TyperError::missing_rtype_operation(format!(
            "low-level helper {} has no annotated helper graph",
            ll_function.name
        ))
    })?;
    let func_ptr = rtyper.getcallable(graph)?;
    let PtrTarget::Func(func_type) = &func_ptr._TYPE.TO else {
        return Err(TyperError::message(format!(
            "direct_call({}) callable is not a function pointer",
            ll_function.name
        )));
    };
    if func_type.args != ll_function.args || func_type.result != ll_function.result {
        return Err(TyperError::message(format!(
            "direct_call({}) graph type mismatch: got {:?} -> {:?}, expected {:?} -> {:?}",
            ll_function.name,
            func_type.args,
            func_type.result,
            ll_function.args,
            ll_function.result
        )));
    }
    let func_ptr_type = LowLevelType::Ptr(Box::new(func_ptr._TYPE.clone()));
    let c_func = inputconst_from_lltype(&func_ptr_type, &ConstValue::LLPtr(Box::new(func_ptr)))
        .expect("functionptr constant must match Ptr(FuncType)");
    let mut call_args = Vec::with_capacity(args_v.len() + 1);
    call_args.push(Hlvalue::Constant(c_func));
    call_args.extend(args_v);
    Ok(SpaceOperation::new(
        "direct_call",
        call_args,
        Hlvalue::Variable(result),
    ))
}

fn synthetic_lowlevel_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
) -> PyGraph {
    let argnames: Vec<String> = (0..args.len()).map(|i| format!("arg{i}")).collect();
    let inputargs: Vec<Hlvalue> = argnames
        .iter()
        .zip(args.iter())
        .map(|(argname, lltype)| Hlvalue::Variable(variable_with_lltype(argname, lltype.clone())))
        .collect();
    let startblock = Block::shared(inputargs);
    let return_var = variable_with_lltype("result", result.clone());
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock, Hlvalue::Variable(return_var));
    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    helper_pygraph_from_graph(graph, argnames, func)
}

fn hlvalue_concretetype(value: &Hlvalue) -> Option<LowLevelType> {
    match value {
        Hlvalue::Variable(v) => v.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    }
}

impl LowLevelOpList {
    /// RPython `LowLevelOpList.__init__(self, rtyper=None,
    /// originalblock=None)` (rtyper.py:793-795).
    pub fn new(rtyper: Rc<RPythonTyper>, originalblock: Option<BlockRef>) -> Self {
        LowLevelOpList {
            rtyper: Some(rtyper),
            originalblock,
            llop_raising_exceptions: None,
            implicit_exceptions_checked: None,
            _called_exception_is_here_or_cannot_occur: false,
            ops: Vec::new(),
        }
    }

    /// RPython `LowLevelOpList()` (rtyper.py:793-795 with default
    /// `rtyper=None`). Used by callers that only ever invoke
    /// `genop` and friends — e.g. `removeassert.py:72`. Methods
    /// that actually require the typer (`gendirectcall`,
    /// `record_extra_call`, `record_extra_call_by_graph_id`)
    /// surface a structured `TyperError` when called against an
    /// instance built this way.
    pub fn without_rtyper(originalblock: Option<BlockRef>) -> Self {
        LowLevelOpList {
            rtyper: None,
            originalblock,
            llop_raising_exceptions: None,
            implicit_exceptions_checked: None,
            _called_exception_is_here_or_cannot_occur: false,
            ops: Vec::new(),
        }
    }

    /// Internal helper for methods that *require* the typer. Mirrors
    /// the upstream call sites that would have crashed with
    /// `AttributeError: 'NoneType' object has no attribute ...` if
    /// they ran against `LowLevelOpList(rtyper=None)`. Pyre converts
    /// that to a structured `TyperError`.
    pub fn require_rtyper(&self, fn_name: &str) -> Result<&Rc<RPythonTyper>, TyperError> {
        self.rtyper.as_ref().ok_or_else(|| {
            TyperError::message(format!(
                "LowLevelOpList::{fn_name} requires an rtyper but the \
                 list was constructed via without_rtyper()"
            ))
        })
    }

    /// PyPy parity proxy for upstream's `llops.gendirectcall(ll_function,
    /// *args)` shape. Upstream callers (e.g. `rstr.py:812,820 convert_from_to`)
    /// hand `gendirectcall` a host Python helper function and let the
    /// `LowLevelOpList` consult its own `self.rtyper` internally to
    /// annotate and resolve the helper graph. Pyre cannot annotate Python
    /// source at this stage of the port (the host helper objects do not
    /// exist), so the helper graph is built explicitly through a
    /// `builder` closure.
    ///
    /// Routing the build through `LowLevelOpList` instead of asking
    /// callers to dance through `llops.require_rtyper(...)?
    /// .lowlevel_helper_function_with_builder(...)` keeps the upstream
    /// caller-side shape intact: `pair_*` conversions and other
    /// `convert_from_to` ports never touch `llops.rtyper` directly.
    pub fn lowlevel_helper_function_with_builder<F>(
        &self,
        name: impl Into<String>,
        args: Vec<LowLevelType>,
        result: LowLevelType,
        builder: F,
    ) -> Result<LowLevelFunction, TyperError>
    where
        F: FnOnce(&RPythonTyper, &[LowLevelType], &LowLevelType) -> Result<PyGraph, TyperError>,
    {
        let rtyper = self.require_rtyper("lowlevel_helper_function_with_builder")?;
        rtyper.lowlevel_helper_function_with_builder(name, args, result, builder)
    }

    /// RPython `LowLevelOpList.hasparentgraph(self)` (rtyper.py:800-801).
    pub fn hasparentgraph(&self) -> bool {
        self.originalblock.is_some()
    }

    /// RPython `LowLevelOpList.record_extra_call(self, graph)`
    /// (`rtyper.py:803-808`).
    pub fn record_extra_call(&self, callee_graph: &GraphRef) -> Result<(), TyperError> {
        if !self.hasparentgraph() {
            return Ok(());
        }
        let rtyper = self.require_rtyper("record_extra_call")?;
        let annotator = rtyper
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("RPythonTyper.annotator weak reference dropped"))?;
        let Some(parent_block) = &self.originalblock else {
            return Ok(());
        };
        if let Some(Some(caller_graph)) = annotator
            .annotated
            .borrow()
            .get(&BlockKey::of(parent_block))
            .cloned()
        {
            annotator.translator.update_call_graph(
                &caller_graph,
                callee_graph,
                (BlockKey::of(parent_block), self.ops.len()),
            );
        }
        Ok(())
    }

    /// Rust bridge for RPython `record_extra_call(vlist[0].value.graph)`
    /// in `rptr.py:100-101`. Function pointers store graph identity in
    /// `lltype._func.graph`; the live graph object is in
    /// `translator.graphs`.
    pub fn record_extra_call_by_graph_id(&self, graph_id: usize) -> Result<(), TyperError> {
        let rtyper = self.require_rtyper("record_extra_call_by_graph_id")?;
        let annotator = rtyper
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("RPythonTyper.annotator weak reference dropped"))?;
        let callee_graph = annotator
            .translator
            .graphs
            .borrow()
            .iter()
            .find(|graph| crate::flowspace::model::GraphKey::of(graph).as_usize() == graph_id)
            .cloned()
            .ok_or_else(|| {
                TyperError::message(format!(
                    "record_extra_call could not find graph id {graph_id}"
                ))
            })?;
        self.record_extra_call(&callee_graph)
    }

    /// RPython `LowLevelOpList.append(self, op)` via `list.append` —
    /// push a `SpaceOperation` onto the buffer.
    pub fn append(&mut self, op: SpaceOperation) {
        self.ops.push(op);
    }

    /// RPython `LowLevelOpList.__len__` via list base.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// RPython `LowLevelOpList.__bool__` via list base.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// RPython `LowLevelOpList.convertvar(self, orig_v, r_from, r_to)`
    /// (`rtyper.py:810-823`).
    ///
    /// Upstream's `pairtype(Repr, Repr).convert_from_to` walks the
    /// `pair_mro` until it finds a registered handler (or exhausts and
    /// raises). Pyre bridges through
    /// [`super::pairtype::pair_convert_from_to`] for the cross-class
    /// lookup and preserves the short-circuit "same repr is identity"
    /// path (upstream `rmodel.py:300`).
    pub fn convertvar(
        &mut self,
        orig_v: Hlvalue,
        r_from: &dyn Repr,
        r_to: &dyn Repr,
    ) -> Result<Hlvalue, TyperError> {
        let same_repr = std::ptr::eq(
            r_from as *const dyn Repr as *const (),
            r_to as *const dyn Repr as *const (),
        );
        if same_repr {
            return Ok(orig_v);
        }
        // Cross-class conversions route through the pairtype
        // dispatcher, which walks `pair_mro(r_from, r_to)` and calls
        // the matching `class __extend__(pairtype(R_A, R_B))` helper.
        if let Some(converted) = super::pairtype::pair_convert_from_to(r_from, r_to, &orig_v, self)?
        {
            if hlvalue_concretetype(&converted).as_ref() != Some(r_to.lowleveltype()) {
                return Err(TyperError::message(format!(
                    "bug in conversion from {} to {}: returned a {:?}",
                    r_from.repr_string(),
                    r_to.repr_string(),
                    hlvalue_concretetype(&converted)
                )));
            }
            return Ok(converted);
        }
        Err(TyperError::message(format!(
            "don't know how to convert from {} to {}",
            r_from.repr_string(),
            r_to.repr_string()
        )))
    }

    /// RPython `LowLevelOpList.genop(self, opname, args_v,
    /// resulttype=None)` (rtyper.py:825-843).
    ///
    /// ```python
    /// def genop(self, opname, args_v, resulttype=None):
    ///     try:
    ///         for v in args_v:
    ///             v.concretetype
    ///     except AttributeError:
    ///         raise AssertionError("wrong level!  you must call hop.inputargs()"
    ///                              " and pass its result to genop(),"
    ///                              " never hop.args_v directly.")
    ///     vresult = Variable()
    ///     self.append(SpaceOperation(opname, args_v, vresult))
    ///     if resulttype is None:
    ///         vresult.concretetype = Void
    ///         return None
    ///     else:
    ///         if isinstance(resulttype, Repr):
    ///             resulttype = resulttype.lowleveltype
    ///         assert isinstance(resulttype, LowLevelType)
    ///         vresult.concretetype = resulttype
    ///         return vresult
    /// ```
    pub fn genop(
        &mut self,
        opname: &str,
        args_v: Vec<Hlvalue>,
        resulttype: GenopResult,
    ) -> Option<Variable> {
        for v in &args_v {
            // upstream asserts each arg already carries a concretetype
            // (rtyper.py:826-832). Pyre stores `Option<LowLevelType>`
            // on Variable/Constant; None means "not rtyped yet".
            match v {
                Hlvalue::Variable(var) => {
                    assert!(
                        var.concretetype().is_some(),
                        "wrong level! you must call hop.inputargs() and pass \
                         its result to genop(), never hop.args_v directly."
                    );
                }
                Hlvalue::Constant(c) => {
                    assert!(
                        c.concretetype.is_some(),
                        "wrong level! Constant used as genop arg has no concretetype."
                    );
                }
            }
        }
        let mut vresult = Variable::new();
        match resulttype {
            GenopResult::Void => {
                vresult.set_concretetype(Some(LowLevelType::Void));
                self.ops.push(SpaceOperation::new(
                    opname.to_string(),
                    args_v,
                    Hlvalue::Variable(vresult),
                ));
                None
            }
            GenopResult::LLType(lltype) => {
                vresult.set_concretetype(Some(lltype));
                let vresult_h = Hlvalue::Variable(vresult.clone());
                self.ops
                    .push(SpaceOperation::new(opname.to_string(), args_v, vresult_h));
                Some(vresult)
            }
            GenopResult::Repr(repr) => {
                vresult.set_concretetype(Some(repr.lowleveltype().clone()));
                let vresult_h = Hlvalue::Variable(vresult.clone());
                self.ops
                    .push(SpaceOperation::new(opname.to_string(), args_v, vresult_h));
                Some(vresult)
            }
        }
    }

    /// RPython `LowLevelOpList.gendirectcall(self, ll_function, *args_v)`
    /// (`rtyper.py:845-882`).
    pub fn gendirectcall(
        &mut self,
        ll_function: &LowLevelFunction,
        args_v: Vec<Hlvalue>,
    ) -> Result<Option<Variable>, TyperError> {
        for (i, (arg, expected)) in args_v.iter().zip(ll_function.args.iter()).enumerate() {
            let got = hlvalue_concretetype(arg).ok_or_else(|| {
                TyperError::message(format!(
                    "gendirectcall argument {i} to {} is missing concretetype",
                    ll_function.name
                ))
            })?;
            if &got != expected {
                return Err(TyperError::message(format!(
                    "gendirectcall argument {i} to {} has type {:?}, expected {:?}",
                    ll_function.name, got, expected
                )));
            }
        }
        if args_v.len() != ll_function.args.len() {
            return Err(TyperError::message(format!(
                "gendirectcall({}) argument count mismatch: got {}, expected {}",
                ll_function.name,
                args_v.len(),
                ll_function.args.len()
            )));
        }

        let graph = ll_function.graph.as_ref().ok_or_else(|| {
            TyperError::missing_rtype_operation(format!(
                "low-level helper {} has no annotated helper graph",
                ll_function.name
            ))
        })?;
        self.record_extra_call(&graph.graph)?;

        let rtyper = self.require_rtyper("gendirectcall")?;
        let func_ptr = rtyper.getcallable(graph)?;
        let PtrTarget::Func(func_type) = &func_ptr._TYPE.TO else {
            return Err(TyperError::message(format!(
                "gendirectcall({}) callable is not a function pointer",
                ll_function.name
            )));
        };
        if func_type.args != ll_function.args || func_type.result != ll_function.result {
            return Err(TyperError::message(format!(
                "gendirectcall({}) graph type mismatch: got {:?} -> {:?}, expected {:?} -> {:?}",
                ll_function.name,
                func_type.args,
                func_type.result,
                ll_function.args,
                ll_function.result
            )));
        }
        let graph_result = func_type.result.clone();
        let func_ptr_type = LowLevelType::Ptr(Box::new(func_ptr._TYPE.clone()));
        let c_func = inputconst_from_lltype(&func_ptr_type, &ConstValue::LLPtr(Box::new(func_ptr)))
            .expect("functionptr constant must match Ptr(FuncType)");
        let mut call_args = Vec::with_capacity(args_v.len() + 1);
        call_args.push(Hlvalue::Constant(c_func));
        call_args.extend(args_v);
        Ok(self.genop("direct_call", call_args, GenopResult::LLType(graph_result)))
    }
}

/// RPython `resulttype=None | LowLevelType | Repr` overload type in
/// `LowLevelOpList.genop` (rtyper.py:825-843). Rust needs an explicit
/// enum because the Python overload uses isinstance().
pub enum GenopResult {
    /// `resulttype=None` — upstream emits `vresult.concretetype =
    /// Void` and returns `None`.
    Void,
    /// `resulttype=LowLevelType` — upstream asserts
    /// `isinstance(resulttype, LowLevelType)` and uses it directly.
    LLType(LowLevelType),
    /// `resulttype=Repr` — upstream extracts `.lowleveltype`.
    Repr(Arc<dyn Repr>),
}

/// RPython `self.llop_raising_exceptions` sentinel states (rtyper.py:745,753).
///
/// ```python
/// self.llops.llop_raising_exceptions = len(self.llops)  # :745
/// self.llops.llop_raising_exceptions = "removed"         # :753
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlopRaisingExceptions {
    /// Index into `LowLevelOpList.ops` of the raising llop.
    Index(usize),
    /// rtyper.py:326 sentinel: the exception cannot actually occur,
    /// all exception links should be removed.
    Removed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::model::{SomeBool, SomeInteger, SomePtr, SomeValue};
    use crate::flowspace::argument::Signature;
    use crate::flowspace::bytecode::HostCode;
    use crate::flowspace::model::{BlockKey, ConstValue, Constant, GraphFunc};
    use crate::translator::rtyper::rmodel::{PtrRepr, VoidRepr};

    #[test]
    fn new_rtyper_starts_with_empty_already_seen() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        assert!(rtyper.already_seen.borrow().is_empty());
        assert!(rtyper.concrete_calltables.borrow().is_empty());
        assert!(rtyper.primitive_to_repr.borrow().is_empty());
        assert!(rtyper.rootclass_repr.borrow().is_none());
        assert!(rtyper.instance_reprs.borrow().is_empty());
    }

    #[test]
    fn annotate_helper_arg_matches_rtyper_py_someobject_or_lltype_branch() {
        // rtyper.py:590-594 keeps already-annotation arguments as-is
        // and only maps low-level types through lltype_to_annotation.
        let already_annotation = SomeValue::Bool(SomeBool::new());
        assert!(matches!(
            annotate_helper_arg_to_annotation(already_annotation.into()),
            SomeValue::Bool(_)
        ));
        assert!(matches!(
            annotate_helper_arg_to_annotation(LowLevelType::Signed.into()),
            SomeValue::Integer(_)
        ));
    }

    #[test]
    fn initialize_exceptiondata_populates_rootclass_repr_and_instance_reprs() {
        use crate::translator::rtyper::rclass::{Flavor, OBJECT, OBJECTPTR};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");

        // rtyper.py:57-58 — rootclass_repr set + setup.
        let root = rtyper
            .rootclass_repr
            .borrow()
            .clone()
            .expect("rootclass_repr should be Some after initialize");
        assert_eq!(
            root.lowleveltype(),
            &crate::translator::rtyper::rclass::CLASSTYPE.clone()
        );

        // rtyper.py:59 / rclass.py:76-88 — instance_reprs cache now
        // contains a `(None, Flavor::Gc)` entry.
        let cache = rtyper.instance_reprs.borrow();
        let inst = cache
            .get(&(None, Flavor::Gc))
            .expect("(None, Gc) entry should be present");
        assert!(inst.classdef().is_none());
        assert_eq!(inst.gcflavor(), Flavor::Gc);
        assert_eq!(inst.object_type(), &OBJECT.clone());
        drop(cache);

        // rtyper.py:71 / exceptiondata.py:22-25 — ExceptionData fields.
        let ed = rtyper.exceptiondata().expect("exceptiondata");
        assert_eq!(
            ed.lltype_of_exception_type,
            crate::translator::rtyper::rclass::CLASSTYPE.clone()
        );
        assert_eq!(ed.lltype_of_exception_value, OBJECTPTR.clone());
    }

    #[test]
    fn exceptiondata_without_initialize_returns_structured_typererror() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let err = rtyper
            .exceptiondata()
            .expect_err("exceptiondata before initialize_exceptiondata should fail");
        assert!(format!("{err:?}").contains("initialize_exceptiondata"));
    }

    #[test]
    fn finish_exceptiondata_sets_up_class_reprs_for_every_standard_exception() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        rtyper
            .finish_exceptiondata()
            .expect("finish_exceptiondata must succeed after initialize");

        let classdefs =
            crate::annotator::exception::standard_exception_classdefs(&ann.bookkeeper).unwrap();
        assert!(
            classdefs
                .iter()
                .all(|classdef| classdef.borrow().repr.is_some())
        );
    }

    #[test]
    fn finish_exceptiondata_without_initialize_surfaces_typererror() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let err = rtyper
            .finish_exceptiondata()
            .expect_err("finish without initialize must error");
        assert!(err.to_string().contains("ExceptionData not initialised"));
    }

    /// Allocate a vtable container and pre-populate the
    /// `subclassrange_min` / `subclassrange_max` head fields. Mirrors the
    /// upstream `init_vtable` writes (`rclass.py:296-330`) at the granularity
    /// `generate_exception_match` consumes — no rclass machinery needed.
    fn make_const_etype(min: i64, max: i64) -> Hlvalue {
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelValue, MallocFlavor, malloc,
        };
        // OBJECT_VTABLE is a `ForwardReference(Struct(...))` for
        // self-reference parity (rclass.py:159); `malloc` only accepts
        // container types, so resolve through the forward-ref before
        // allocating.
        let vtable_lltype = match crate::translator::rtyper::rclass::OBJECT_VTABLE.clone() {
            LowLevelType::ForwardReference(fwd) => fwd
                .resolved()
                .expect("OBJECT_VTABLE forward-reference must be resolved"),
            other => other,
        };
        let mut vtable_ptr =
            malloc(vtable_lltype, None, MallocFlavor::Raw, true).expect("malloc OBJECT_VTABLE");
        vtable_ptr
            .setattr("subclassrange_min", LowLevelValue::Signed(min))
            .expect("setattr subclassrange_min");
        vtable_ptr
            .setattr("subclassrange_max", LowLevelValue::Signed(max))
            .expect("setattr subclassrange_max");
        let const_etype = Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(vtable_ptr)),
            crate::translator::rtyper::rclass::CLASSTYPE.clone(),
        );
        Hlvalue::Constant(const_etype)
    }

    #[test]
    fn getannmixlevel_returns_cached_instance_within_one_pass() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        // Initially the cache is empty.
        assert!(rtyper.annmixlevel.borrow().is_none());

        let first = rtyper.getannmixlevel();
        let second = rtyper.getannmixlevel();
        // Two consecutive calls within the same pass must reuse the
        // same `MixLevelHelperAnnotator` instance (upstream identity
        // check via `self.annmixlevel is not None`).
        assert!(Rc::ptr_eq(&first, &second));
        assert!(rtyper.annmixlevel.borrow().is_some());
    }

    #[test]
    fn specialize_more_blocks_resets_annmixlevel_per_pass() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");

        // Seed the cache so we can observe the reset at pass entry.
        let _seeded = rtyper.getannmixlevel();
        assert!(rtyper.annmixlevel.borrow().is_some());

        // No annotated blocks → loop terminates immediately, but the
        // annmixlevel reset and post-loop finish-take still execute.
        rtyper
            .specialize_more_blocks()
            .expect("specialize_more_blocks on empty annotated");
        assert!(
            rtyper.annmixlevel.borrow().is_none(),
            "specialize_more_blocks must clear annmixlevel after finish",
        );
    }

    #[test]
    fn get_standard_ll_exc_instance_returns_objectptr_constant_for_root_classdef() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let ed = rtyper.exceptiondata().expect("exceptiondata");

        // None classdef → root InstanceRepr; create_instance allocates
        // an OBJECT struct (no super chain), and ll_cast_to_object is
        // identity for OBJECTPTR.
        let out = ed
            .get_standard_ll_exc_instance(&rtyper, None)
            .expect("get_standard_ll_exc_instance(None) on root");
        let crate::flowspace::model::ConstValue::LLPtr(p) = &out.value else {
            panic!("expected LLPtr, got {:?}", out.value);
        };
        assert!(p.nonzero(), "prebuilt root instance must be live");
        assert_eq!(
            out.concretetype.as_ref(),
            Some(&crate::translator::rtyper::rclass::OBJECTPTR.clone())
        );
    }

    #[test]
    fn get_standard_ll_exc_instance_by_class_unknown_raises_typererror() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let ed = rtyper.exceptiondata().expect("exceptiondata");
        let bogus = crate::flowspace::model::HostObject::new_class("NotAStandardException", vec![]);
        let err = ed
            .get_standard_ll_exc_instance_by_class(&rtyper, &bogus)
            .expect_err("unknown class must error (UnknownException parity)");
        assert!(err.to_string().contains("UnknownException"), "{err}");
    }

    #[test]
    fn get_standard_ll_exc_instance_by_class_known_returns_live_objectptr() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        rtyper.finish_exceptiondata().expect("finish_exceptiondata");
        // `fill_vtable_root` requires `classdef.minid/maxid`, which
        // `assign_inheritance_ids` populates. Upstream
        // `RPythonTyper.specialize` runs that pass via
        // `perform_normalizations` before any vtable materialisation.
        crate::translator::rtyper::normalizecalls::assign_inheritance_ids(&ann);
        let ed = rtyper.exceptiondata().expect("exceptiondata");

        // OverflowError is a known standard exception (verified by
        // standard_exception_classes_have_expected_names test). The
        // by-class wrapper resolves the classdef and forwards to the
        // live get_standard_ll_exc_instance body. Look up the
        // canonical HostObject from the standard set so identity
        // matches what `initialize_exceptiondata` registered.
        let overflow = crate::annotator::exception::standard_exception_classes()
            .into_iter()
            .find(|cls| cls.qualname() == "OverflowError")
            .expect("OverflowError must be a standard exception");
        let out = ed
            .get_standard_ll_exc_instance_by_class(&rtyper, &overflow)
            .expect("standard exception class must produce a live prebuilt instance");
        let crate::flowspace::model::ConstValue::LLPtr(p) = &out.value else {
            panic!("expected LLPtr, got {:?}", out.value);
        };
        assert!(p.nonzero());
    }

    #[test]
    fn generate_exception_match_emits_getfield_then_int_between() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let ed = rtyper.exceptiondata().expect("exceptiondata");

        let mut var_etype = Variable::new();
        var_etype.set_concretetype(Some(crate::translator::rtyper::rclass::CLASSTYPE.clone()));
        let var_etype_h = Hlvalue::Variable(var_etype);

        let const_etype = make_const_etype(7, 11);

        let mut oplist: Vec<SpaceOperation> = Vec::new();
        let res = ed
            .generate_exception_match(rtyper.clone(), &mut oplist, var_etype_h, const_etype)
            .expect("generate_exception_match");

        // upstream emits exactly two ops: getfield + int_between.
        assert_eq!(oplist.len(), 2, "expected two helper ops, got {oplist:?}");
        assert_eq!(oplist[0].opname, "getfield");
        assert_eq!(oplist[1].opname, "int_between");

        // getfield arg[1] is `genvoidconst('subclassrange_min')`.
        let Hlvalue::Constant(field_const) = &oplist[0].args[1] else {
            panic!("getfield arg[1] must be a Constant");
        };
        assert_eq!(field_const.value, ConstValue::byte_str("subclassrange_min"));
        assert_eq!(field_const.concretetype.as_ref(), Some(&LowLevelType::Void));

        // int_between args[0]/args[2] are `genconst(min)` / `genconst(max)`.
        let Hlvalue::Constant(c_min) = &oplist[1].args[0] else {
            panic!("int_between arg[0] must be a Constant");
        };
        let Hlvalue::Constant(c_max) = &oplist[1].args[2] else {
            panic!("int_between arg[2] must be a Constant");
        };
        assert_eq!(c_min.value, ConstValue::Int(7));
        assert_eq!(c_max.value, ConstValue::Int(11));
        assert_eq!(c_min.concretetype.as_ref(), Some(&LowLevelType::Signed));
        assert_eq!(c_max.concretetype.as_ref(), Some(&LowLevelType::Signed));

        // Returned res variable carries Bool concretetype.
        let Hlvalue::Variable(res_var) = res else {
            panic!("generate_exception_match must return a Variable");
        };
        assert_eq!(res_var.concretetype().as_ref(), Some(&LowLevelType::Bool));
    }

    #[test]
    fn generate_exception_match_rejects_non_constant_etype() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let ed = rtyper.exceptiondata().expect("exceptiondata");

        let mut var_etype = Variable::new();
        var_etype.set_concretetype(Some(crate::translator::rtyper::rclass::CLASSTYPE.clone()));
        let var_etype_h = Hlvalue::Variable(var_etype);
        // Pass a Variable as `const_etype` to violate the upstream
        // contract that it must carry a concrete `_ptr`.
        let mut bad_const = Variable::new();
        bad_const.set_concretetype(Some(crate::translator::rtyper::rclass::CLASSTYPE.clone()));
        let bad_const_h = Hlvalue::Variable(bad_const);

        let mut oplist: Vec<SpaceOperation> = Vec::new();
        let err = ed
            .generate_exception_match(rtyper.clone(), &mut oplist, var_etype_h, bad_const_h)
            .expect_err("non-constant const_etype must error");
        assert!(
            err.to_string().contains("const_etype must be a Constant"),
            "{err}",
        );
        assert!(oplist.is_empty(), "no ops should leak when input invalid");
    }

    #[test]
    fn mark_already_seen_records_block_key() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let block = ann.translator.entry_point_graph.borrow().clone();
        assert!(block.is_none());

        let graph = crate::flowspace::model::FunctionGraph::new(
            "g",
            std::rc::Rc::new(std::cell::RefCell::new(
                crate::flowspace::model::Block::new(vec![]),
            )),
        );
        let startblock = graph.startblock.clone();
        rtyper.mark_already_seen(&startblock);
        assert!(
            rtyper
                .already_seen
                .borrow()
                .contains_key(&BlockKey::of(&startblock))
        );
    }

    #[test]
    fn getcallable_returns_functionptr_for_graph() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let code = HostCode {
            id: HostCode::fresh_identity(),
            co_name: "f".into(),
            co_filename: "<test>".into(),
            co_firstlineno: 1,
            co_nlocals: 1,
            co_argcount: 1,
            co_stacksize: 0,
            co_flags: 0,
            co_code: rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
            co_varnames: vec!["x".into()],
            co_freevars: Vec::new(),
            co_cellvars: Vec::new(),
            consts: Vec::new(),
            names: Vec::new(),
            co_lnotab: Vec::new(),
            exceptiontable: Vec::new().into_boxed_slice(),
            signature: Signature::new(vec!["x".into()], None, None),
        };
        let graph = Rc::new(PyGraph::new(
            GraphFunc::new("f", Constant::new(ConstValue::Dict(Default::default()))),
            &code,
        ));
        {
            use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
            let graph_borrow = graph.graph.borrow();
            for arg in graph_borrow.startblock.borrow_mut().inputargs.iter_mut() {
                if let crate::flowspace::model::Hlvalue::Variable(v) = arg {
                    v.set_concretetype(Some(LowLevelType::Signed));
                    v.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
                }
            }
            for arg in graph_borrow.returnblock.borrow_mut().inputargs.iter_mut() {
                if let crate::flowspace::model::Hlvalue::Variable(v) = arg {
                    v.set_concretetype(Some(LowLevelType::Void));
                    v.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
                }
            }
        }

        let llfn = rtyper.getcallable(&graph).unwrap();
        use crate::translator::rtyper::lltypesystem::lltype::{_ptr_obj, PtrTarget};
        let PtrTarget::Func(func_t) = &llfn._TYPE.TO else {
            panic!("expected Func ptr, got {:?}", llfn._TYPE.TO);
        };
        assert_eq!(func_t.args.len(), 1);
        let _ptr_obj::Func(func_obj) = llfn._obj().unwrap() else {
            panic!("expected _ptr_obj::Func");
        };
        assert_eq!(func_obj.graph.is_some(), true);
    }

    #[test]
    fn getcallable_sandbox_with_external_name_surfaces_unported_trampoline() {
        // rtyper.py:572-582 — sandbox path requires
        // `make_sandbox_trampoline` (rsandbox.py), which is unported
        // in pyre. When `translation.sandbox` is enabled and the
        // graph carries a `_sandbox_external_name`, the path must
        // surface a structured TyperError citing the missing dep
        // rather than silently falling through to `getfunctionptr`.
        let mut translator = crate::translator::translator::TranslationContext::new();
        translator.config.translation.sandbox = true;
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let code = HostCode {
            id: HostCode::fresh_identity(),
            co_name: "f".into(),
            co_filename: "<test>".into(),
            co_firstlineno: 1,
            co_nlocals: 1,
            co_argcount: 1,
            co_stacksize: 0,
            co_flags: 0,
            co_code: rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
            co_varnames: vec!["x".into()],
            co_freevars: Vec::new(),
            co_cellvars: Vec::new(),
            consts: Vec::new(),
            names: Vec::new(),
            co_lnotab: Vec::new(),
            exceptiontable: Vec::new().into_boxed_slice(),
            signature: Signature::new(vec!["x".into()], None, None),
        };
        let mut func = GraphFunc::new("ext", Constant::new(ConstValue::Dict(Default::default())));
        func._sandbox_external_name = Some("ext".to_string());
        let graph = Rc::new(PyGraph::new(func, &code));
        let err = rtyper
            .getcallable(&graph)
            .expect_err("sandbox path must surface unported trampoline");
        assert!(
            err.to_string().contains("make_sandbox_trampoline"),
            "expected sandbox-trampoline TaskError, got {:?}",
            err.to_string()
        );
    }

    #[test]
    fn getcallable_sandbox_without_external_name_falls_through_to_getfunctionptr() {
        // rtyper.py:573-575 — `try:
        // graph.func._sandbox_external_name; except AttributeError:
        // pass`. With sandbox enabled but no external name set, the
        // try block raises AttributeError and the function falls
        // through to `getfunctionptr` per upstream.
        let mut translator = crate::translator::translator::TranslationContext::new();
        translator.config.translation.sandbox = true;
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let code = HostCode {
            id: HostCode::fresh_identity(),
            co_name: "f".into(),
            co_filename: "<test>".into(),
            co_firstlineno: 1,
            co_nlocals: 1,
            co_argcount: 1,
            co_stacksize: 0,
            co_flags: 0,
            co_code: rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
            co_varnames: vec!["x".into()],
            co_freevars: Vec::new(),
            co_cellvars: Vec::new(),
            consts: Vec::new(),
            names: Vec::new(),
            co_lnotab: Vec::new(),
            exceptiontable: Vec::new().into_boxed_slice(),
            signature: Signature::new(vec!["x".into()], None, None),
        };
        let graph = Rc::new(PyGraph::new(
            GraphFunc::new("f", Constant::new(ConstValue::Dict(Default::default()))),
            &code,
        ));
        {
            use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
            let graph_borrow = graph.graph.borrow();
            for arg in graph_borrow.startblock.borrow_mut().inputargs.iter_mut() {
                if let crate::flowspace::model::Hlvalue::Variable(v) = arg {
                    v.set_concretetype(Some(LowLevelType::Signed));
                    v.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
                }
            }
            for arg in graph_borrow.returnblock.borrow_mut().inputargs.iter_mut() {
                if let crate::flowspace::model::Hlvalue::Variable(v) = arg {
                    v.set_concretetype(Some(LowLevelType::Void));
                    v.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
                }
            }
        }
        rtyper
            .getcallable(&graph)
            .expect("sandbox without external name must fall through");
    }

    fn make_rtyper_rc() -> Rc<RPythonTyper> {
        let ann = RPythonAnnotator::new(None, None, None, false);
        Rc::new(RPythonTyper::new(&ann))
    }

    fn make_live_rtyper() -> (Rc<RPythonAnnotator>, Rc<RPythonTyper>) {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        (ann, rtyper)
    }

    fn empty_spaceop(opname: &str) -> SpaceOperation {
        SpaceOperation::new(
            opname.to_string(),
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        )
    }

    #[test]
    fn highlevelop_constructor_stores_fields_without_running_setup() {
        // rtyper.py:619-623: `__init__` only records references; the
        // args_v / args_s / args_r / s_result / r_result slots remain
        // empty until `setup()` runs (which pyre defers pending the
        // Repr-dispatch chain).
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            empty_spaceop("simple_call"),
            Vec::new(),
            llops,
        );
        assert_eq!(hop.spaceop.opname, "simple_call");
        assert_eq!(hop.nb_args(), 0);
        assert!(hop.exceptionlinks.is_empty());
        assert!(hop.args_v.borrow().is_empty());
        assert!(hop.args_s.borrow().is_empty());
        assert!(hop.args_r.borrow().is_empty());
        assert!(hop.s_result.borrow().is_none());
        assert!(hop.r_result.borrow().is_none());
    }

    #[test]
    fn highlevelop_copy_clones_lists_and_shares_llops() {
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), Vec::new(), llops.clone());
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(None);

        let copied = hop.copy();
        copied.args_v.borrow_mut().clear();
        assert_eq!(hop.args_v.borrow().len(), 1);
        assert!(Rc::ptr_eq(&copied.llops, &llops));
    }

    #[test]
    fn highlevelop_dispatch_routes_to_default_translate_operation() {
        let (_ann, rtyper) = make_live_rtyper();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            empty_spaceop("not_ported_yet"),
            Vec::new(),
            llops,
        );
        let err = hop.dispatch().unwrap_err();
        assert!(err.to_string().contains("unimplemented operation"));
        assert!(err.to_string().contains("not_ported_yet"));
    }

    #[test]
    fn highlevelop_dispatch_routes_binary_op_to_first_repr_like_rtyper_registeroperations() {
        use crate::translator::rtyper::lltypesystem::lltype::{ArrayType, Ptr, PtrTarget};

        let (_ann, rtyper) = make_live_rtyper();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let ptr = Ptr {
            TO: PtrTarget::Array(ArrayType::new(LowLevelType::Signed)),
        };
        let mut v_array = Variable::new();
        v_array.set_concretetype(Some(LowLevelType::Ptr(Box::new(ptr.clone()))));
        let v_index = Constant::with_concretetype(ConstValue::Int(1), LowLevelType::Signed);
        let mut result = Variable::new();
        result
            .annotation
            .replace(Some(Rc::new(SomeValue::Integer(SomeInteger::new(
                false, false,
            )))));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_array), Hlvalue::Constant(v_index)],
                Hlvalue::Variable(result),
            ),
            Vec::new(),
            llops.clone(),
        );
        let r_ptr: Arc<dyn Repr> = Arc::new(PtrRepr::new(ptr.clone()));
        let r_signed = rtyper.getprimitiverepr(&LowLevelType::Signed).unwrap();
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Ptr(SomePtr::new(ptr)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_ptr), Some(r_signed.clone())]);
        *hop.r_result.borrow_mut() = Some(r_signed);

        let out = hop.dispatch().unwrap().unwrap();
        assert!(matches!(out, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "getarrayitem");
    }

    #[test]
    fn highlevelop_inputconst_accepts_repr_and_lowleveltype() {
        let repr = VoidRepr::new();
        let c = HighLevelOp::inputconst(&repr, &ConstValue::Int(7)).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Void));

        let c = HighLevelOp::inputconst(&LowLevelType::Signed, &ConstValue::Int(7)).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Signed));
    }

    #[test]
    fn highlevelop_inputarg_constant_uses_requested_repr() {
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), Vec::new(), llops);
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Constant(Constant::new(ConstValue::Int(7))));

        let repr = VoidRepr::new();
        let out = hop.inputarg(&repr, 0).unwrap();
        let Hlvalue::Constant(c) = out else {
            panic!("inputarg on Constant should return a Constant");
        };
        assert_eq!(c.value, ConstValue::Int(7));
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
    }

    #[test]
    fn highlevelop_inputarg_variable_uses_convertvar() {
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), Vec::new(), llops);
        let repr: Arc<dyn Repr> = Arc::new(VoidRepr::new());
        let mut var = Variable::new();
        var.set_concretetype(Some(LowLevelType::Void));
        let arg = Hlvalue::Variable(var.clone());
        hop.args_v.borrow_mut().push(arg.clone());
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(repr.clone()));

        let out = hop.inputarg(&repr, 0).unwrap();
        assert_eq!(out, arg);
    }

    #[test]
    fn highlevelop_inputargs_checks_arity() {
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), Vec::new(), llops);
        let repr = VoidRepr::new();
        let err = hop.inputargs(vec![ConvertedTo::from(&repr)]).unwrap_err();
        assert!(
            err.to_string()
                .contains("operation argument count mismatch")
        );
    }

    #[test]
    fn highlevelop_r_s_pop_removes_trailing_element() {
        // rtyper.py:693-696: pop from args_v + args_r + args_s in lockstep.
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), Vec::new(), llops);
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::default()));
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(None);
        hop.args_r.borrow_mut().push(None);
        let (_r, s) = hop.r_s_pop(None);
        assert!(matches!(s, SomeValue::Impossible));
        assert_eq!(hop.args_v.borrow().len(), 1);
    }

    #[test]
    fn highlevelop_swap_fst_snd_args_swaps_all_three_parallel_vecs() {
        // rtyper.py:708-711.
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), Vec::new(), llops);
        let a = Variable::new();
        let b = Variable::new();
        hop.args_v.borrow_mut().push(Hlvalue::Variable(a.clone()));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(b.clone()));
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::default()));
        hop.args_r.borrow_mut().push(None);
        hop.args_r.borrow_mut().push(None);
        hop.swap_fst_snd_args();
        let v = hop.args_v.borrow();
        match (&v[0], &v[1]) {
            (Hlvalue::Variable(first), Hlvalue::Variable(second)) => {
                // Compare by Variable equality (upstream uses `is`);
                // pyre's Variable `PartialEq` matches on identity
                // through the `id` UID allocator.
                assert_eq!(first, &b);
                assert_eq!(second, &a);
            }
            _ => panic!("expected Variable entries"),
        }
        assert!(matches!(hop.args_s.borrow()[0], SomeValue::Integer(_)));
    }

    #[test]
    fn lowlevelop_append_and_len_track_ops_buffer() {
        let rtyper = make_rtyper_rc();
        let mut llops = LowLevelOpList::new(rtyper, None);
        assert!(llops.is_empty());
        llops.append(SpaceOperation::new(
            "direct_call".to_string(),
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        ));
        assert_eq!(llops.len(), 1);
        assert_eq!(llops.ops[0].opname, "direct_call");
    }

    #[test]
    fn lowlevelop_convertvar_reuses_value_for_same_repr() {
        let rtyper = make_rtyper_rc();
        let mut llops = LowLevelOpList::new(rtyper, None);
        let repr: Arc<dyn Repr> = Arc::new(VoidRepr::new());
        let value = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ));
        let out = llops
            .convertvar(value.clone(), repr.as_ref(), repr.as_ref())
            .unwrap();
        assert_eq!(out, value);
    }

    #[test]
    fn lowlevelop_convertvar_to_void_returns_void_constant() {
        let rtyper = make_rtyper_rc();
        let mut llops = LowLevelOpList::new(rtyper, None);
        let from: Arc<dyn Repr> = Arc::new(VoidRepr::new());
        let to: Arc<dyn Repr> = Arc::new(VoidRepr::new());
        let value = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ));
        let out = llops.convertvar(value, from.as_ref(), to.as_ref()).unwrap();
        assert_eq!(
            out,
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::None,
                LowLevelType::Void
            ))
        );
    }

    #[test]
    fn lowlevelop_genop_void_emits_void_concretetype() {
        // rtyper.py:835-837: resulttype=None → Void + return None.
        let rtyper = make_rtyper_rc();
        let mut llops = LowLevelOpList::new(rtyper, None);
        let result = llops.genop("nop", Vec::new(), GenopResult::Void);
        assert!(result.is_none());
        assert_eq!(llops.ops.len(), 1);
        if let Hlvalue::Variable(v) = &llops.ops[0].result {
            assert_eq!(v.concretetype(), Some(LowLevelType::Void));
        } else {
            panic!("expected Variable result");
        }
    }

    #[test]
    fn lowlevelop_genop_with_lltype_result_sets_concretetype() {
        // rtyper.py:839-843: resulttype=LowLevelType → Variable
        // carries that type.
        let rtyper = make_rtyper_rc();
        let mut llops = LowLevelOpList::new(rtyper, None);
        // Build a typed input arg so the concretetype assertion passes.
        let mut input = Variable::new();
        input.set_concretetype(Some(LowLevelType::Signed));
        let result = llops
            .genop(
                "int_add",
                vec![Hlvalue::Variable(input)],
                GenopResult::LLType(LowLevelType::Signed),
            )
            .expect("Signed resulttype must produce a Variable");
        assert_eq!(result.concretetype(), Some(LowLevelType::Signed));
    }

    #[test]
    fn lowlevelop_genop_with_repr_extracts_lowleveltype() {
        // rtyper.py:839-843: resulttype=Repr → upstream does
        // `resulttype = resulttype.lowleveltype`.
        let rtyper = make_rtyper_rc();
        let mut llops = LowLevelOpList::new(rtyper, None);
        let result = llops
            .genop(
                "nop",
                Vec::new(),
                GenopResult::Repr(Arc::new(VoidRepr::new())),
            )
            .expect("VoidRepr produces a Variable with Void concretetype");
        assert_eq!(result.concretetype(), Some(LowLevelType::Void));
    }

    #[test]
    #[should_panic(expected = "wrong level!")]
    fn lowlevelop_genop_rejects_args_without_concretetype() {
        // rtyper.py:826-832: `hop.args_v directly` mistake must
        // assertion-fail.
        let rtyper = make_rtyper_rc();
        let mut llops = LowLevelOpList::new(rtyper, None);
        let untyped = Variable::new();
        assert!(untyped.concretetype().is_none());
        let _ = llops.genop(
            "int_add",
            vec![Hlvalue::Variable(untyped)],
            GenopResult::LLType(LowLevelType::Signed),
        );
    }

    #[test]
    fn lowlevelop_without_rtyper_supports_genop() {
        // rtyper.py:793-795 — `LowLevelOpList(rtyper=None)` is the
        // shape `removeassert.py:72` uses. The Rust port mirrors
        // that with `LowLevelOpList::without_rtyper`; methods that
        // do not require the typer (`genop`, `append`, `len`,
        // `is_empty`) work normally.
        let mut llops = LowLevelOpList::without_rtyper(None);
        let mut input = Variable::new();
        input.set_concretetype(Some(LowLevelType::Bool));
        let result = llops
            .genop(
                "bool_not",
                vec![Hlvalue::Variable(input)],
                GenopResult::LLType(LowLevelType::Bool),
            )
            .expect("genop without rtyper must succeed");
        assert_eq!(result.concretetype(), Some(LowLevelType::Bool));
        assert_eq!(llops.ops.len(), 1);
        assert_eq!(llops.ops[0].opname, "bool_not");
    }

    #[test]
    fn lowlevelop_without_rtyper_record_extra_call_surfaces_typererror() {
        // Methods that require the typer must surface a structured
        // TyperError when called on a `without_rtyper()` instance,
        // matching upstream's implicit `AttributeError` on
        // `None.something(...)`.
        let block = Rc::new(RefCell::new(crate::flowspace::model::Block::new(vec![])));
        let llops = LowLevelOpList::without_rtyper(Some(block));
        let err = llops
            .record_extra_call_by_graph_id(0)
            .expect_err("record_extra_call_by_graph_id without rtyper must surface TyperError");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("requires an rtyper") || msg.contains("without_rtyper"),
            "expected without_rtyper guard message, got {msg:?}"
        );
    }

    #[test]
    fn exception_cannot_occur_sets_removed_sentinel() {
        // rtyper.py:747-753.
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        // Build a dummy exceptionlink so the path executes (upstream
        // early-returns when exceptionlinks is empty).
        let exitblock = Rc::new(RefCell::new(crate::flowspace::model::Block::new(vec![])));
        let link = Rc::new(RefCell::new(crate::flowspace::model::Link::new(
            vec![],
            Some(exitblock),
            None,
        )));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), vec![link], llops.clone());
        hop.exception_cannot_occur().unwrap();
        assert_eq!(
            llops.borrow().llop_raising_exceptions,
            Some(LlopRaisingExceptions::Removed)
        );
    }

    /// rtyper.py:737-744 — when `has_implicit_exception` was called
    /// at least once on this hop (`implicit_exceptions_checked` is
    /// `Some`, possibly empty), `exception_is_here` must verify that
    /// every `exceptionlink.exitcase` was checked. If a catch arm is
    /// not handled, raise `TyperError("the graph catches X, but the
    /// rtyper did not explicitely handle it")`.
    #[test]
    fn exception_is_here_rejects_unchecked_exitcase() {
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        // Two exceptionlinks: IndexError (will be checked) +
        // ValueError (NOT checked) — sanity check must reject.
        let exitblock1 = Rc::new(RefCell::new(crate::flowspace::model::Block::new(vec![])));
        let exitblock2 = Rc::new(RefCell::new(crate::flowspace::model::Block::new(vec![])));
        let cls_index = HOST_ENV.lookup_exception_class("IndexError").unwrap();
        let cls_value = HOST_ENV.lookup_exception_class("ValueError").unwrap();
        let link_index = Rc::new(RefCell::new(crate::flowspace::model::Link::new(
            vec![],
            Some(exitblock1),
            Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                cls_index,
            )))),
        )));
        let link_value = Rc::new(RefCell::new(crate::flowspace::model::Link::new(
            vec![],
            Some(exitblock2),
            Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                cls_value,
            )))),
        )));
        let hop = HighLevelOp::new(
            rtyper,
            empty_spaceop("nop"),
            vec![link_index, link_value],
            llops,
        );
        // Check IndexError only.
        assert!(hop.has_implicit_exception("IndexError"));
        let err = hop
            .exception_is_here()
            .expect_err("ValueError unchecked must error");
        assert!(
            format!("{err}").contains("ValueError"),
            "expected ValueError-naming error, got: {err}"
        );
    }

    /// rtyper.py:737-744 — sanity check passes when every
    /// exceptionlink.exitcase was visited by has_implicit_exception.
    #[test]
    fn exception_is_here_accepts_all_checked_exitcases() {
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let exitblock = Rc::new(RefCell::new(crate::flowspace::model::Block::new(vec![])));
        let cls_index = HOST_ENV.lookup_exception_class("IndexError").unwrap();
        let link = Rc::new(RefCell::new(crate::flowspace::model::Link::new(
            vec![],
            Some(exitblock),
            Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                cls_index,
            )))),
        )));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), vec![link], llops.clone());
        assert!(hop.has_implicit_exception("IndexError"));
        hop.exception_is_here()
            .expect("all exitcases checked → exception_is_here OK");
        assert!(matches!(
            llops.borrow().llop_raising_exceptions,
            Some(LlopRaisingExceptions::Index(_))
        ));
    }

    #[test]
    fn has_implicit_exception_records_matching_exception_link() {
        // rtyper.py:713-729: issubclass(exc_cls, link.exitcase) records
        // every matching exception link and returns True.
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let exitblock = Rc::new(RefCell::new(crate::flowspace::model::Block::new(vec![])));
        let catch_exception = HOST_ENV
            .lookup_exception_class("Exception")
            .expect("Exception class");
        let link = Rc::new(RefCell::new(crate::flowspace::model::Link::new(
            vec![],
            Some(exitblock),
            Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                catch_exception,
            )))),
        )));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), vec![link], llops.clone());

        assert!(hop.has_implicit_exception("ValueError"));
        assert_eq!(
            llops.borrow().implicit_exceptions_checked,
            Some(vec!["Exception".to_string()])
        );
    }

    #[test]
    fn lowlevel_helper_function_builds_rint_check_range_graphs() {
        // rint.py:629-638: ll_check_chr / ll_check_unichr are real
        // helper bodies, not signature-only synthetic graphs.
        let (_ann, rtyper) = make_live_rtyper();
        for (name, upper_bound) in [("ll_check_chr", 255), ("ll_check_unichr", 0x10ffff)] {
            let llfn = rtyper
                .lowlevel_helper_function(name, vec![LowLevelType::Signed], LowLevelType::Void)
                .unwrap();
            let pygraph = llfn.graph.expect("helper graph");
            let graph = pygraph.graph.borrow();
            let startblock = graph.startblock.clone();
            let returnblock = graph.returnblock.clone();
            let exceptblock = graph.exceptblock.clone();
            drop(graph);

            let start = startblock.borrow();
            assert_eq!(start.operations.len(), 1);
            assert_eq!(start.operations[0].opname, "int_ge");
            assert_eq!(start.exits.len(), 2);
            let true_link = start
                .exits
                .iter()
                .find(|link| bool_exitcase(link) == Some(true))
                .expect("lower-bound true link");
            let check_upper = true_link
                .borrow()
                .target
                .as_ref()
                .expect("upper-bound block")
                .clone();
            drop(start);

            let upper = check_upper.borrow();
            assert_eq!(upper.operations.len(), 1);
            assert_eq!(upper.operations[0].opname, "int_le");
            assert!(matches!(
                &upper.operations[0].args[1],
                Hlvalue::Constant(Constant {
                    value: ConstValue::Int(n),
                    concretetype: Some(LowLevelType::Signed),
                    ..
                }) if *n == upper_bound
            ));
            assert!(upper.exits.iter().any(|link| {
                bool_exitcase(link) == Some(true)
                    && link
                        .borrow()
                        .target
                        .as_ref()
                        .is_some_and(|target| BlockKey::of(target) == BlockKey::of(&returnblock))
            }));
            assert!(upper.exits.iter().any(|link| {
                bool_exitcase(link) == Some(false)
                    && link
                        .borrow()
                        .target
                        .as_ref()
                        .is_some_and(|target| BlockKey::of(target) == BlockKey::of(&exceptblock))
            }));
        }
    }

    #[test]
    fn lowlevel_helper_function_builds_int_py_div_ovf_wrapper_graph() {
        // rint.py:422-427: the overflow check is intentionally
        // non-short-circuiting `(x == INT_MIN) & (y == -1)`.
        let (_ann, rtyper) = make_live_rtyper();
        let llfn = rtyper
            .lowlevel_helper_function(
                "ll_int_py_div_ovf",
                vec![LowLevelType::Signed, LowLevelType::Signed],
                LowLevelType::Signed,
            )
            .unwrap();
        let pygraph = llfn.graph.expect("helper graph");
        let graph = pygraph.graph.borrow();
        let startblock = graph.startblock.clone();
        let exceptblock = graph.exceptblock.clone();
        drop(graph);

        let start = startblock.borrow();
        let opnames: Vec<_> = start
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["int_eq", "int_eq", "int_and"]);
        assert!(start.exits.iter().any(|link| {
            bool_exitcase(link) == Some(true)
                && link
                    .borrow()
                    .target
                    .as_ref()
                    .is_some_and(|target| BlockKey::of(target) == BlockKey::of(&exceptblock))
        }));
        let callblock = start
            .exits
            .iter()
            .find(|link| bool_exitcase(link) == Some(false))
            .and_then(|link| link.borrow().target.clone())
            .expect("normal path callblock");
        drop(start);

        let call = callblock.borrow();
        assert_eq!(call.operations.len(), 1);
        assert_eq!(call.operations[0].opname, "direct_call");
    }

    fn bool_exitcase(link: &LinkRef) -> Option<bool> {
        match &link.borrow().exitcase {
            Some(Hlvalue::Constant(Constant {
                value: ConstValue::Bool(value),
                ..
            })) => Some(*value),
            _ => None,
        }
    }

    #[test]
    fn rtyper_annotation_returns_none_without_binding() {
        // rtyper.py:166-168 `annotation(var)` returns the annotator's
        // bound value or None for unset variables — matches upstream.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let v = Hlvalue::Variable(Variable::new());
        assert!(rtyper.annotation(&v).is_none());
    }

    #[test]
    fn rtyper_binding_passes_through_annotator_value() {
        // rtyper.py:170-172 returns the annotator binding on the
        // positive path.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let mut var = Variable::new();
        var.annotation
            .replace(Some(Rc::new(SomeValue::Integer(SomeInteger::default()))));
        let v = Hlvalue::Variable(var);
        let binding = rtyper.binding(&v);
        assert!(matches!(binding, SomeValue::Integer(_)));
    }

    #[test]
    #[should_panic(expected = "KeyError: no binding")]
    fn rtyper_binding_panics_on_missing_variable() {
        // rtyper.py:170-172 — upstream raises KeyError when the
        // annotator has no binding. Rust mirrors that through the
        // direct `annotator.binding()` path.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let v = Hlvalue::Variable(Variable::new());
        let _ = rtyper.binding(&v);
    }

    #[test]
    fn getrepr_caches_impossible_as_shared_singleton() {
        // rtyper.py:149-164: first call materialises, second returns
        // cached entry. For SomeImpossibleValue both arms target the
        // same `impossible_repr` singleton, so pointer-equality via
        // Arc::ptr_eq must hold.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r1 = rtyper.getrepr(&SomeValue::Impossible).unwrap();
        let r2 = rtyper.getrepr(&SomeValue::Impossible).unwrap();
        assert!(Arc::ptr_eq(&r1, &r2));
        // Cache now carries exactly one entry keyed by Impossible.
        assert_eq!(rtyper.reprs.borrow().len(), 1);
    }

    #[test]
    fn getrepr_surfaces_missing_rtype_operation_for_unported_variants() {
        // rtyper.py:157 — `s_obj.rtyper_makerepr(self)` raises when
        // the variant has no concrete Repr yet. Pyre surfaces
        // `MissingRTypeOperation` so cascading callers can anchor on
        // the upstream module name in the error message.
        //
        // `SomeWeakRef` is the still-unported witness variant
        // (rweakref.py port not landed). Earlier this test exercised
        // `SomeString`; the Item 3 epic (Slice 3) ported `string_repr`
        // / `unicode_repr`, so rotated onto another unported variant
        // (robject.py / rproperty.py / rweakref.py — pick whichever
        // remains unported next).
        use crate::annotator::model::SomeWeakRef;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let err = rtyper
            .getrepr(&SomeValue::WeakRef(SomeWeakRef::new(None)))
            .unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("rweakref.py"));
    }

    #[test]
    fn bindingrepr_passes_through_getrepr() {
        // rtyper.py:174-175.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let mut var = Variable::new();
        var.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
        let v = Hlvalue::Variable(var);
        let repr = rtyper.bindingrepr(&v).unwrap();
        // impossible_repr lowleveltype is Void.
        assert_eq!(repr.lowleveltype(), &LowLevelType::Void);
    }

    #[test]
    fn setconcretetype_writes_lowleveltype_from_bindingrepr() {
        // rtyper.py:258-260.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let mut var = Variable::new();
        var.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
        assert_eq!(var.concretetype(), None);
        rtyper.setconcretetype(&var).unwrap();
        // impossible_repr lowleveltype is Void.
        assert_eq!(var.concretetype(), Some(LowLevelType::Void));
    }

    #[test]
    fn setup_block_entry_normal_path_writes_each_inputarg_concretetype() {
        // rtyper.py:272-278 normal path.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let mut v = Variable::new();
        v.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
        let block = Block::shared(vec![Hlvalue::Variable(v)]);
        let reprs = rtyper.setup_block_entry(&block).unwrap();
        assert_eq!(reprs.len(), 1);
        assert_eq!(reprs[0].lowleveltype(), &LowLevelType::Void);
        let b = block.borrow();
        match &b.inputargs[0] {
            Hlvalue::Variable(v) => assert_eq!(v.concretetype(), Some(LowLevelType::Void)),
            _ => panic!("expected Variable"),
        }
    }

    #[test]
    fn make_new_lloplist_wires_rtyper_and_originalblock() {
        // rtyper.py:280-281.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let block = Block::shared(vec![]);
        let llops = rtyper.make_new_lloplist(block.clone());
        assert!(
            Rc::ptr_eq(llops.rtyper.as_ref().expect("rtyper required"), &rtyper),
            "LowLevelOpList.rtyper must share identity with make_new_lloplist caller"
        );
        assert!(llops.originalblock.is_some());
        assert!(llops.ops.is_empty());
    }

    #[test]
    fn highlevelops_empty_block_yields_nothing() {
        // rtyper.py:423-424: `if block.operations:` gates the whole body.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let block = Block::shared(vec![]);
        let llops = Rc::new(RefCell::new(rtyper.make_new_lloplist(block.clone())));
        let hops = rtyper.highlevelops(&block, llops);
        assert!(hops.is_empty());
    }

    #[test]
    fn highlevelops_non_raising_block_yields_one_hop_per_op_with_empty_exclinks() {
        // rtyper.py:425-426 + 430-432 when canraise is false.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let v = Variable::new();
        let op1 = SpaceOperation::new("op1".to_string(), vec![], Hlvalue::Variable(v.clone()));
        let op2 = SpaceOperation::new("op2".to_string(), vec![], Hlvalue::Variable(v.clone()));
        let block = Block::shared(vec![]);
        block.borrow_mut().operations.push(op1);
        block.borrow_mut().operations.push(op2);
        let llops = Rc::new(RefCell::new(rtyper.make_new_lloplist(block.clone())));
        let hops = rtyper.highlevelops(&block, llops);
        assert_eq!(hops.len(), 2);
        assert_eq!(hops[0].spaceop.opname, "op1");
        assert_eq!(hops[1].spaceop.opname, "op2");
        assert!(hops[0].exceptionlinks.is_empty());
        assert!(hops[1].exceptionlinks.is_empty());
    }

    #[test]
    fn setup_block_entry_exception_block_returns_typed_error_pending_exceptiondata() {
        // rtyper.py:263-270 special-case. Blocked on ExceptionData port.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let mut v1 = Variable::new();
        v1.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
        let mut v2 = Variable::new();
        v2.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
        let block = Block::shared(vec![Hlvalue::Variable(v1), Hlvalue::Variable(v2)]);
        block.borrow_mut().mark_final();
        let err = rtyper.setup_block_entry(&block).unwrap_err();
        assert!(err.to_string().contains("ExceptionData"));
    }

    #[test]
    fn getrepr_adds_pendingsetup_once_per_key() {
        // rtyper.py:105-111 + 162: `add_pendingsetup` dedupes by
        // Repr instance; the second getrepr hit must not re-enqueue
        // the cached Arc.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let _ = rtyper.getrepr(&SomeValue::Impossible).unwrap();
        let after_first = rtyper.reprs_must_call_setup.borrow().len();
        let _ = rtyper.getrepr(&SomeValue::Impossible).unwrap();
        // Second call hits cache (rtyper.py:154-155) and skips
        // add_pendingsetup entirely, so pending-setup depth is
        // unchanged.
        assert_eq!(rtyper.reprs_must_call_setup.borrow().len(), after_first);
    }

    #[test]
    fn call_all_setups_drains_pending_reprs() {
        // rtyper.py:243-256.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let _ = rtyper.getrepr(&SomeValue::Impossible).unwrap();
        assert_eq!(rtyper.reprs_must_call_setup.borrow().len(), 1);
        rtyper.call_all_setups().unwrap();
        assert_eq!(rtyper.reprs_must_call_setup.borrow().len(), 0);
    }

    #[test]
    fn highlevelop_setup_fills_args_and_result_reprs() {
        // rtyper.py:625-633.
        // Build a spaceop with one Impossible-typed arg + Impossible
        // result; bind annotations so `rtyper.binding` succeeds.
        let ann_rc = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann_rc));
        let mut arg_var = Variable::new();
        arg_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Impossible)));
        let mut result_var = Variable::new();
        result_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Impossible)));
        let spaceop = SpaceOperation::new(
            "simple_call".to_string(),
            vec![Hlvalue::Variable(arg_var)],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, spaceop, Vec::new(), llops);
        hop.setup().unwrap();
        assert_eq!(hop.args_v.borrow().len(), 1);
        assert_eq!(hop.args_s.borrow().len(), 1);
        assert_eq!(hop.args_r.borrow().len(), 1);
        assert!(matches!(
            hop.s_result.borrow().as_ref().unwrap(),
            SomeValue::Impossible
        ));
        // Both args_r and r_result resolve to impossible_repr →
        // lowleveltype Void.
        let r_result = hop.r_result.borrow().as_ref().cloned().unwrap();
        assert_eq!(r_result.lowleveltype(), &LowLevelType::Void);
    }

    #[test]
    fn highlevelop_setup_surfaces_missing_rtype_operation_for_unported_arg() {
        // rtyper.py:625-633 → rtyper.py:157 `s_obj.rtyper_makerepr`
        // surfaces MissingRTypeOperation when the argument's
        // SomeValue variant has no concrete Repr yet. Rather than
        // silently fail, the setup path returns the structured
        // TyperError so callers know which upstream module to land.
        //
        // `SomeWeakRef` is the still-unported witness variant
        // (rweakref.py). Earlier this test used `SomeString`; that
        // landed in Slice 3 of the Item 3 epic, so rotated to
        // `SomeWeakRef`.
        use crate::annotator::model::SomeWeakRef;
        let ann_rc = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann_rc));
        let mut arg_var = Variable::new();
        arg_var
            .annotation
            .replace(Some(Rc::new(SomeValue::WeakRef(SomeWeakRef::new(None)))));
        let mut result_var = Variable::new();
        result_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Impossible)));
        let spaceop = SpaceOperation::new(
            "simple_call".to_string(),
            vec![Hlvalue::Variable(arg_var)],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, spaceop, Vec::new(), llops);
        let err = hop.setup().unwrap_err();
        assert!(err.is_missing_rtype_operation());
    }

    #[test]
    fn translate_no_return_value_on_impossible_s_result_writes_void_concretetype() {
        // rtyper.py:487-488 — when `hop.s_result == s_ImpossibleValue`,
        // upstream silently writes `op.result.concretetype = Void` and
        // returns. Pyre propagates the write through `Rc<RefCell>` so every
        // Variable clone of the same identity observes Void.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let result_var = Variable::new();
        result_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Impossible)));
        // Keep a second handle to the same Variable identity so we can
        // observe the mutation after the spaceop ownership transfers.
        let observer = result_var.clone();
        let spaceop = SpaceOperation::new(
            "simple_call".to_string(),
            vec![],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        *hop.s_result.borrow_mut() = Some(SomeValue::Impossible);
        assert_eq!(observer.concretetype(), None);
        rtyper.translate_no_return_value(&hop).unwrap();
        assert_eq!(observer.concretetype(), Some(LowLevelType::Void));
    }

    #[test]
    fn translate_no_return_value_on_constant_result_builds_void_typed_mirror() {
        // rtyper.py:488 `op.result.concretetype = Void` writes
        // unconditionally. When op.result is a Constant, Rust's
        // immutable Constant cannot be rewritten in place, so the
        // translate path builds a typed mirror Hlvalue. The method
        // still returns Ok (the write side-effect on Variable
        // op.results continues to work via Rc<RefCell>).
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let spaceop = SpaceOperation::new(
            "simple_call".to_string(),
            vec![],
            Hlvalue::Constant(Constant::new(ConstValue::Int(0))),
        );
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        *hop.s_result.borrow_mut() = Some(SomeValue::Impossible);
        rtyper.translate_no_return_value(&hop).unwrap();
    }

    #[test]
    fn translate_no_return_value_on_non_impossible_s_result_errors() {
        // rtyper.py:485-487 — any annotator state other than
        // s_ImpossibleValue trips the "annotator doesn't agree" TyperError.
        use crate::annotator::model::SomeInteger;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut result_var = Variable::new();
        result_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Integer(SomeInteger::new(
                true, false,
            )))));
        let spaceop = SpaceOperation::new(
            "simple_call".to_string(),
            vec![],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        *hop.s_result.borrow_mut() = Some(SomeValue::Integer(SomeInteger::new(true, false)));
        let err = rtyper.translate_no_return_value(&hop).unwrap_err();
        assert!(
            err.to_string().contains("annotator doesn't agree"),
            "error message did not mention annotator disagreement: {err}",
        );
        assert!(
            err.to_string().contains("simple_call"),
            "error message did not mention opname: {err}",
        );
    }

    #[test]
    fn gottypererror_records_graph_name_block_and_position_in_typererror_where_info() {
        // rtyper.py:490-493: `graph = annotator.annotated.get(block); exc.where = (graph, block, position)`.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let startblock = Block::shared(vec![]);
        let graph = Rc::new(RefCell::new(FunctionGraph::new(
            "my_func",
            startblock.clone(),
        )));
        // Mirror `annotator.annotated[block] = graph` (annrpython.py:
        // schedulependingblock).
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&startblock), Some(graph));
        let op = SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(Variable::named("a")),
                Hlvalue::Variable(Variable::named("b")),
            ],
            Hlvalue::Variable(Variable::named("r")),
        );
        let exc = TyperError::message("bad cast");
        let annotated = rtyper.gottypererror(exc, &startblock, &op);
        let rendered = annotated.to_string();
        assert!(
            rendered.contains("\n.. my_func"),
            "expected graph name in where.stage; got {rendered}",
        );
        assert!(
            rendered.contains("opname: \"int_add\""),
            "expected op Debug in where.op; got {rendered}",
        );
    }

    #[test]
    fn gottypererror_when_block_not_in_annotated_maps_to_missing_graph_sentinel() {
        // rtyper.py:492 `dict.get(block)` returns None when the block
        // has not been scheduled yet; upstream stores None in the
        // where trio. Pyre renders this as "<no graph>".
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let block = Block::shared(vec![]);
        let exc = TyperError::message("bad cast");
        let annotated = rtyper.gottypererror(exc, &block, &"link-placeholder");
        let rendered = annotated.to_string();
        assert!(
            rendered.contains("\n.. <no graph>"),
            "expected missing-graph sentinel; got {rendered}",
        );
    }

    fn bool_annotated_variable() -> Variable {
        let v = Variable::new();
        v.annotation
            .replace(Some(Rc::new(SomeValue::Bool(SomeBool::new()))));
        v
    }

    #[test]
    fn convert_link_exitcase_none_sets_llexitcase_none() {
        // rtyper.py:354+361 — None exitcase hits the `else` branch,
        // which zeroes out `link.llexitcase`.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let block = Block::shared(vec![]);
        let target = Block::shared(vec![]);
        let link = Link::new(vec![], Some(target), None).into_ref();
        // Pre-load llexitcase with a junk value to observe the reset.
        link.borrow_mut().llexitcase =
            Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(true))));
        rtyper._convert_link(&block, &link).unwrap();
        assert!(link.borrow().llexitcase.is_none());
    }

    #[test]
    fn convert_link_default_string_exitcase_sets_llexitcase_none() {
        // rtyper.py:354 — the `exitcase != 'default'` guard short-
        // circuits, so `link.llexitcase` is zeroed even though the
        // exitcase is non-None.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let block = Block::shared(vec![]);
        let target = Block::shared(vec![]);
        let exitcase = Some(Hlvalue::Constant(Constant::new(ConstValue::byte_str(
            "default",
        ))));
        let link = Link::new(vec![], Some(target), exitcase).into_ref();
        link.borrow_mut().llexitcase =
            Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(true))));
        rtyper._convert_link(&block, &link).unwrap();
        assert!(link.borrow().llexitcase.is_none());
    }

    #[test]
    fn convert_link_bool_exitswitch_and_bool_exitcase_converts_llexitcase() {
        // rtyper.py:355-360 — Variable exitswitch + concrete exitcase
        // goes through `bindingrepr(exitswitch).convert_const(exitcase)`
        // and lands the result on `link.llexitcase`.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let switch_var = bool_annotated_variable();
        let block = Block::shared(vec![]);
        block.borrow_mut().exitswitch = Some(Hlvalue::Variable(switch_var));
        let target = Block::shared(vec![]);
        let exitcase = Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(true))));
        let link = Link::new(vec![], Some(target), exitcase).into_ref();
        rtyper._convert_link(&block, &link).unwrap();
        let llexitcase = link.borrow().llexitcase.clone();
        match llexitcase {
            Some(Hlvalue::Constant(c)) => {
                assert!(matches!(c.value, ConstValue::Bool(true)));
                assert_eq!(c.concretetype, Some(LowLevelType::Bool));
            }
            other => panic!("expected Bool-typed Constant llexitcase, got {other:?}"),
        }
    }

    #[test]
    fn convert_link_canraise_branch_defers_without_rootclass_repr() {
        // rtyper.py:358-359 — non-Variable exitswitch on a canraise
        // block reads `rclass.get_type_repr(self)`, which pulls
        // `rtyper.rootclass_repr`. Without a prior
        // `initialize_exceptiondata()` call the port surfaces a
        // structured TyperError pointing at that missing init step.
        use crate::flowspace::model::c_last_exception;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let block = Block::shared(vec![]);
        block.borrow_mut().exitswitch = Some(Hlvalue::Constant(c_last_exception()));
        assert!(block.borrow().canraise());
        let target = Block::shared(vec![]);
        let exitcase = Some(Hlvalue::Constant(Constant::new(ConstValue::byte_str(
            "ValueError",
        ))));
        let link = Link::new(vec![], Some(target), exitcase).into_ref();
        let err = rtyper._convert_link(&block, &link).unwrap_err();
        assert!(
            err.to_string().contains("initialize_exceptiondata"),
            "expected initialize_exceptiondata defer; got {err}",
        );
    }

    #[test]
    fn convert_link_last_exception_set_defers_to_exceptiondata() {
        // rtyper.py:364-369 — any `link.last_exception` reads
        // `exceptiondata.lltype_of_exception_type` /
        // `r_exception_type`. Until the ExceptionData init surface
        // lands, the read surfaces a structured init-missing error.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let block = Block::shared(vec![]);
        let target = Block::shared(vec![]);
        let link = Link::new(vec![], Some(target), None).into_ref();
        link.borrow_mut().last_exception = Some(Hlvalue::Variable(Variable::named("etype")));
        let err = rtyper._convert_link(&block, &link).unwrap_err();
        assert!(
            err.to_string().contains("ExceptionData not initialised"),
            "expected ExceptionData init defer; got {err}",
        );
    }

    #[test]
    fn insert_link_conversions_single_exit_matching_reprs_generates_no_ops() {
        // rtyper.py:398-399 — when `r_a1 == r_a2` for every link arg,
        // no conversion ops are generated. The link ends up unchanged
        // and the source block stays empty of operations.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let src_var = bool_annotated_variable();
        let target_var = bool_annotated_variable();
        let target = Block::shared(vec![Hlvalue::Variable(target_var)]);
        let block = Block::shared(vec![]);
        let link = Link::new(vec![Hlvalue::Variable(src_var)], Some(target), None).into_ref();
        block.closeblock(vec![link.clone()]);

        rtyper.insert_link_conversions(&block, 0).unwrap();
        assert!(
            block.borrow().operations.is_empty(),
            "no conversion ops expected; got {:?}",
            block.borrow().operations,
        );
        assert_eq!(link.borrow().args.len(), 1);
    }

    #[test]
    fn insert_link_conversions_constant_link_arg_rewritten_to_typed_constant() {
        // rtyper.py:389-391 — `isinstance(a1, Constant)` short-circuits
        // through `inputconst(r_a2, a1.value)`, overwriting
        // `link.args[i]` with a typed Constant.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let target_var = bool_annotated_variable();
        let target = Block::shared(vec![Hlvalue::Variable(target_var)]);
        let block = Block::shared(vec![]);
        let raw_const = Hlvalue::Constant(Constant::new(ConstValue::Bool(true)));
        let link = Link::new(vec![raw_const], Some(target), None).into_ref();
        block.closeblock(vec![link.clone()]);

        rtyper.insert_link_conversions(&block, 0).unwrap();
        let lb = link.borrow();
        match lb.args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => {
                assert!(matches!(c.value, ConstValue::Bool(true)));
                assert_eq!(c.concretetype, Some(LowLevelType::Bool));
            }
            other => panic!("expected typed Constant; got {other:?}"),
        }
        assert!(block.borrow().operations.is_empty());
    }

    #[test]
    fn specialize_block_on_returnblock_fixes_graph_and_leaves_operations_empty() {
        // rtyper.py:284-301 — when the block has no operations (return
        // or except block), upstream returns after fixing the graph's
        // return var. Pyre matches: fixed_graphs gains the graph and
        // block.operations stays empty.
        use crate::flowspace::model::GraphKey;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let startblock = Block::shared(vec![]);
        let graph = Rc::new(RefCell::new(FunctionGraph::new("test_fn", startblock)));
        // Annotate graph.returnblock.inputargs[0] so setup_block_entry
        // can derive a repr via bindingrepr (normal path).
        let returnblock = graph.borrow().returnblock.clone();
        if let Hlvalue::Variable(v) = returnblock.borrow().inputargs[0].clone() {
            v.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
        }
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&returnblock), Some(graph.clone()));

        rtyper.specialize_block(&returnblock).unwrap();

        assert!(returnblock.borrow().operations.is_empty());
        let fixed = ann.fixed_graphs.borrow();
        assert!(
            fixed.contains_key(&GraphKey::of(&graph)),
            "specialize_block must register the graph in fixed_graphs",
        );
        // returnblock.inputargs[0] now carries concretetype=Void
        // (Impossible repr's lowleveltype).
        if let Hlvalue::Variable(v) = returnblock.borrow().inputargs[0].clone() {
            assert_eq!(v.concretetype(), Some(LowLevelType::Void));
        }
    }

    #[test]
    fn constant_result_values_agree_matches_rtyper_py_456_assert() {
        // rtyper.py:456-458 — upstream asserts `resultvar.value ==
        // hop.s_result.const` OR both are NaN. Covers the constant-
        // result mismatch path inside `translate_hl_to_ll` without
        // needing to spin up a full rtype dispatch.
        // Equality case.
        assert!(constant_result_values_agree(
            &ConstValue::Int(42),
            &ConstValue::Int(42),
        ));
        // Mismatch case — assert trigger.
        assert!(!constant_result_values_agree(
            &ConstValue::Int(42),
            &ConstValue::Int(7),
        ));
        // NaN-NaN case: upstream `math.isnan(a) and math.isnan(b)`
        // bypasses the equality check because `NaN == NaN` is false.
        let nan_bits = f64::NAN.to_bits();
        let other_nan_bits = f64::from_bits(nan_bits | 0x1).to_bits();
        assert!(constant_result_values_agree(
            &ConstValue::Float(nan_bits),
            &ConstValue::Float(other_nan_bits),
        ));
        // NaN vs finite: NOT agreeing.
        assert!(!constant_result_values_agree(
            &ConstValue::Float(nan_bits),
            &ConstValue::Float(1.0_f64.to_bits()),
        ));
    }

    #[test]
    fn specialize_block_empty_non_final_block_runs_insert_link_conversions() {
        // rtyper.py:300 — upstream's `block.operations == ()` check
        // fires only on final (return/except) blocks. Empty regular
        // blocks still flow into `insert_link_conversions`, which in
        // turn runs `setup_block_entry` on the link's target. The
        // observable side-effect: the target's inputargs gain
        // concretetypes even though the source block has no ops.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let startblock = Block::shared(vec![]);
        let graph = Rc::new(RefCell::new(FunctionGraph::new(
            "empty_fn",
            startblock.clone(),
        )));
        // specialize_block's first-visit branch calls setconcretetype
        // on graph.getreturnvar(), which needs an annotation.
        let returnblock = graph.borrow().returnblock.clone();
        if let Hlvalue::Variable(v) = returnblock.borrow().inputargs[0].clone() {
            v.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
        }

        // Regular non-final block with no operations but one exit to a
        // target carrying an annotated Variable inputarg.
        let source_var = bool_annotated_variable();
        let target_var = bool_annotated_variable();
        let target = Block::shared(vec![Hlvalue::Variable(target_var.clone())]);
        let block = Block::shared(vec![]);
        let link = Link::new(vec![Hlvalue::Variable(source_var)], Some(target), None).into_ref();
        block.closeblock(vec![link]);

        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&block), Some(graph.clone()));

        assert!(!block.borrow().is_final_block());
        assert!(block.borrow().operations.is_empty());
        // target.inputargs[0].concretetype starts unset.
        assert_eq!(target_var.concretetype(), None);

        rtyper.specialize_block(&block).unwrap();

        // insert_link_conversions → setup_block_entry stamped Bool
        // onto target_var.
        assert_eq!(target_var.concretetype(), Some(LowLevelType::Bool));
    }

    #[test]
    fn specialize_block_setup_block_entry_error_is_annotated_through_gottypererror() {
        // rtyper.py:292-296 — setup_block_entry failure is routed
        // through gottypererror with the "block-entry" position tag.
        // Pyre surfaces both the underlying ExceptionData defer and
        // the where-annotation (graph name from annotator.annotated).
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let startblock = Block::shared(vec![]);
        let graph = Rc::new(RefCell::new(FunctionGraph::new("my_graph", startblock)));
        // Annotate graph.returnblock.inputargs[0] — specialize_block's
        // first-time-for-graph branch calls setconcretetype on it.
        let returnblock = graph.borrow().returnblock.clone();
        if let Hlvalue::Variable(v) = returnblock.borrow().inputargs[0].clone() {
            v.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
        }
        // graph.exceptblock is already final+2-inputargs; register it
        // so annotator.annotated lookup resolves.
        let exceptblock = graph.borrow().exceptblock.clone();
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&exceptblock), Some(graph.clone()));

        let err = rtyper.specialize_block(&exceptblock).unwrap_err();
        let rendered = err.to_string();
        assert!(
            rendered.contains("ExceptionData"),
            "expected ExceptionData defer in error: {rendered}",
        );
        assert!(
            rendered.contains("\n.. my_graph"),
            "expected graph name via gottypererror where-stage: {rendered}",
        );
        // Position tag is the literal "block-entry".
        assert!(
            rendered.contains("block-entry"),
            "expected 'block-entry' position in where-op: {rendered}",
        );
    }

    #[test]
    fn specialize_top_driver_runs_simplify_finish_and_two_more_blocks_passes() {
        // rtyper.py:177-189 — smoke test that the end-to-end driver
        // runs cleanly on an empty annotator (simplify no-op, finish
        // populates classdef.repr for standard exceptions, both
        // specialize_more_blocks calls exit via the empty-pending
        // break).
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        rtyper.specialize(false).expect("specialize");
        let classdefs =
            crate::annotator::exception::standard_exception_classdefs(&ann.bookkeeper).unwrap();
        assert!(
            classdefs
                .iter()
                .all(|classdef| classdef.borrow().repr.is_some()),
            "specialize must run finish_exceptiondata, populating classdef.repr"
        );
    }

    #[test]
    fn specialize_without_initialize_surfaces_typererror() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let err = rtyper
            .specialize(false)
            .expect_err("specialize before initialize must fail");
        assert!(err.to_string().contains("ExceptionData not initialised"));
    }

    #[test]
    fn specialize_dont_simplify_again_skips_annotator_simplify() {
        // The dont_simplify_again=True path must still succeed even
        // without the annotator.simplify() call (upstream's
        // --no-simplify / caller-owns-simplify use case).
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        rtyper
            .specialize(true)
            .expect("specialize(dont_simplify_again=true)");
    }

    #[test]
    fn specialize_more_blocks_with_empty_annotator_returns_immediately() {
        // rtyper.py:210-213 — break when no pending blocks remain.
        // An empty annotator yields no pending, so the loop exits
        // without touching anything.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.specialize_more_blocks().unwrap();
        assert!(rtyper.already_seen.borrow().is_empty());
    }

    #[test]
    fn specialize_more_blocks_marks_annotated_returnblock_as_seen() {
        // rtyper.py:222-225 — the driver calls specialize_block on
        // each pending block and marks it already_seen. One pending
        // returnblock exercises the happy path end to end.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let startblock = Block::shared(vec![]);
        let graph = Rc::new(RefCell::new(FunctionGraph::new("g", startblock)));
        let returnblock = graph.borrow().returnblock.clone();
        if let Hlvalue::Variable(v) = returnblock.borrow().inputargs[0].clone() {
            v.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
        }
        let rb_key = BlockKey::of(&returnblock);
        ann.annotated
            .borrow_mut()
            .insert(rb_key.clone(), Some(graph));
        // Populate all_blocks so the BlockKey→BlockRef lookup succeeds.
        ann.all_blocks
            .borrow_mut()
            .insert(rb_key.clone(), returnblock.clone());

        rtyper.specialize_more_blocks().unwrap();

        assert!(
            rtyper.already_seen.borrow().contains_key(&rb_key),
            "specialize_more_blocks must mark visited block as seen",
        );
    }

    #[test]
    fn specialize_more_blocks_propagates_setup_block_entry_error() {
        // rtyper.py:222-224 — specialize_block errors bubble straight
        // through; pyre preserves the gottypererror-annotated payload.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let startblock = Block::shared(vec![]);
        let graph = Rc::new(RefCell::new(FunctionGraph::new("g_with_exc", startblock)));
        // returnblock must be annotated so specialize_block's first-
        // time setconcretetype path doesn't trip on None annotation.
        let returnblock = graph.borrow().returnblock.clone();
        if let Hlvalue::Variable(v) = returnblock.borrow().inputargs[0].clone() {
            v.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
        }
        // exceptblock (final + 2 inputargs) triggers the ExceptionData
        // defer inside setup_block_entry.
        let exceptblock = graph.borrow().exceptblock.clone();
        let eb_key = BlockKey::of(&exceptblock);
        ann.annotated
            .borrow_mut()
            .insert(eb_key.clone(), Some(graph.clone()));
        ann.all_blocks.borrow_mut().insert(eb_key, exceptblock);

        let err = rtyper.specialize_more_blocks().unwrap_err();
        let rendered = err.to_string();
        assert!(
            rendered.contains("ExceptionData"),
            "expected ExceptionData defer to propagate: {rendered}",
        );
        assert!(
            rendered.contains("g_with_exc"),
            "expected graph name in where-stage: {rendered}",
        );
    }

    #[test]
    fn insert_link_conversions_exception_target_block_error_propagates_without_gottypererror_wrap()
    {
        // rtyper.py:382-383 vs 400-403 — `_convert_link` and
        // `setup_block_entry` errors propagate uncaught; only the
        // `newops.convertvar` call is wrapped through `gottypererror`.
        // Exception-shaped target is still the trigger (ExceptionData
        // defer bubbles up from `setup_block_entry`), but the raised
        // TyperError must carry NO where-annotation.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        // Exception-shaped target: 2 inputargs + is_final (the
        // setup_block_entry special case).
        let v1 = Variable::new();
        v1.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
        let v2 = Variable::new();
        v2.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
        let target = Block::shared(vec![Hlvalue::Variable(v1), Hlvalue::Variable(v2)]);
        target.borrow_mut().mark_final();

        let block = Block::shared(vec![]);
        let link_args = vec![
            Hlvalue::Variable(Variable::named("etype_src")),
            Hlvalue::Variable(Variable::named("evalue_src")),
        ];
        let link = Link::new(link_args, Some(target), None).into_ref();
        block.closeblock(vec![link]);

        let err = rtyper.insert_link_conversions(&block, 0).unwrap_err();
        let rendered = err.to_string();
        assert!(
            rendered.contains("ExceptionData"),
            "expected ExceptionData defer in error: {rendered}",
        );
        assert!(
            !rendered.contains("\n.. <no graph>"),
            "setup_block_entry error must NOT be wrapped by gottypererror: {rendered}",
        );
    }
}
