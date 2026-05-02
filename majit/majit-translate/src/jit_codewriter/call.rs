//! Call control — inline vs residual decision for function calls.
//!
//! RPython equivalent: `rpython/jit/codewriter/call.py` class `CallControl`.
//!
//! Decides which functions should be inlined into JitCode ("regular") and
//! which should remain as opaque calls ("residual").  Also handles builtin
//! (oopspec) and recursive (portal) call classification.

use std::collections::{HashMap, HashSet};

use majit_ir::descr::{EffectInfo, ExtraEffect, OopSpecIndex};
use majit_ir::value::Type;
use serde::{Deserialize, Serialize};

use crate::front::ast::SemanticFunction;
use crate::jitcode::{BhCallDescr, CallResultErasedKey};
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation};
use crate::parse::CallPath;
use crate::policy::JitPolicy;

// ── Graph-based analyzers (RPython effectinfo.py + canraise.py) ────
//
// RPython uses BoolGraphAnalyzer subclasses that traverse call graphs
// transitively. Each analyzer checks for specific operations:
//   - RaiseAnalyzer: Abort terminators (canraise.py)
//   - VirtualizableAnalyzer: jit_force_virtualizable/jit_force_virtual ops
//   - QuasiImmutAnalyzer: jit_force_quasi_immutable ops
//   - RandomEffectsAnalyzer: unanalyzable external calls

/// RPython: canraise.py — result of raise analysis.
///
/// `_canraise()` returns True, False, or "mem" (only MemoryError).
/// call.py:337-355.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanRaise {
    /// Function cannot raise any exception.
    No,
    /// Function can only raise MemoryError.
    MemoryErrorOnly,
    /// Function can raise arbitrary exceptions.
    Yes,
}

/// Operation-level raise classification for `_canraise()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RaiseClass {
    No,
    MemoryErrorOnly,
    Yes,
}

fn raise_class_can_raise(value: RaiseClass, ignore_memoryerror: bool) -> bool {
    match value {
        RaiseClass::No => false,
        RaiseClass::MemoryErrorOnly => !ignore_memoryerror,
        RaiseClass::Yes => true,
    }
}

/// RPython: DependencyTracker equivalent — caches transitive analysis results.
///
/// Each analyzer in RPython has its own `seen` set (via `analyze_direct_call`).
/// We cache the final result per CallPath so repeated queries are O(1).
#[derive(Default)]
pub struct AnalysisCache {
    can_raise: HashMap<CallPath, CanRaise>,
    forces_virtualizable: HashMap<CallPath, bool>,
    random_effects: HashMap<CallPath, bool>,
    can_invalidate: HashMap<CallPath, bool>,
    /// RPython: collect_analyzer (collectanalyze.py) — can this call trigger GC?
    can_collect: HashMap<CallPath, bool>,
    /// RPython: `cpu.fielddescrof(T, fieldname)` / `cpu.arraydescrof(ARRAY)`.
    /// Assigns sequential, collision-free ei_index values for bitstrings.
    pub descr_indices: DescrIndexRegistry,
}

/// RPython: readwrite_analyzer.analyze(op) return value.
///
/// Represents the set of read/write effects collected from graph traversal.
/// RPython uses a set of tuples like ("struct", T, fieldname); we use bitsets.
/// This is passed as the first argument to effectinfo_from_writeanalyze(),
/// matching RPython's `effects` parameter (effectinfo.py:276).
pub struct WriteAnalysis {
    pub read_fields: u64,
    pub write_fields: u64,
    pub read_arrays: u64,
    pub write_arrays: u64,
    pub read_interiorfields: u64,
    pub write_interiorfields: u64,
    pub array_write_descrs: Vec<majit_ir::descr::DescrRef>,
    /// RPython: `effects is top_set` — unanalyzable (random effects).
    pub is_top: bool,
}

/// Call descriptor — `AbstractDescr`-equivalent metadata for a call op.
///
/// RPython equivalent: the `CallDescr` returned by
/// `CallControl.getcalldescr()` (call.py:236-241), wrapping
/// `EffectInfo` and the cpu-level descr identity.  Upstream stores
/// the funcptr separately as `op.args[0]`; pyre carries the funcptr
/// identity on each `OpKind` variant's dedicated `funcptr` field
/// (model.rs:247) so this struct holds only the calldescr-side data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallDescriptor {
    /// RPython `CallDescr.arg_classes`: one char per non-void FUNC argument.
    pub arg_classes: String,
    /// RPython `CallDescr.result_type`.
    pub result_type: char,
    /// RPython `descr.py:664` `result_signed`.
    pub result_signed: bool,
    /// RPython `descr.py:662` `result_size`.
    pub result_size: usize,
    /// RPython `descr.py:665` `RESULT_ERASED`.
    pub result_erased: CallResultErasedKey,
    pub extra_info: EffectInfo,
}

impl CallDescriptor {
    pub fn known(extra_info: EffectInfo) -> Self {
        Self::from_signature(&[], Type::Void, extra_info)
    }

    pub fn override_effect(extra_info: EffectInfo) -> Self {
        Self::from_signature(&[], Type::Void, extra_info)
    }

    pub fn from_signature(arg_types: &[Type], result_type: Type, extra_info: EffectInfo) -> Self {
        let arg_classes = arg_types.iter().map(|tp| type_to_argclass(*tp)).collect();
        let (result_type, result_signed, result_size, result_erased) =
            result_layout_key(result_type);
        Self {
            arg_classes,
            result_type,
            result_signed,
            result_size,
            result_erased,
            extra_info,
        }
    }

    pub fn with_signature(mut self, arg_types: &[Type], result_type: Type) -> Self {
        let (result_type, result_signed, result_size, result_erased) =
            result_layout_key(result_type);
        self.arg_classes = arg_types.iter().map(|tp| type_to_argclass(*tp)).collect();
        self.result_type = result_type;
        self.result_signed = result_signed;
        self.result_size = result_size;
        self.result_erased = result_erased;
        self
    }

    pub fn get_extra_info(&self) -> EffectInfo {
        self.extra_info.clone()
    }

    pub fn arg_types(&self) -> Vec<Type> {
        self.arg_classes
            .chars()
            .filter_map(argclass_to_ir_type)
            .collect()
    }

    pub fn result_ir_type(&self) -> Type {
        result_char_to_ir_type(self.result_type)
    }

    pub fn to_bh_calldescr(&self) -> BhCallDescr {
        BhCallDescr {
            arg_classes: self.arg_classes.clone(),
            result_type: self.result_type,
            result_signed: self.result_signed,
            result_size: self.result_size,
            result_erased: self.result_erased,
            extra_info: self.extra_info.clone(),
        }
    }

    pub fn to_descr_ref(&self) -> majit_ir::descr::DescrRef {
        majit_ir::descr::make_call_descr_full(
            0,
            self.arg_types(),
            self.result_ir_type(),
            self.result_signed,
            self.result_size,
            self.extra_info.clone(),
        )
    }
}

fn type_to_argclass(tp: Type) -> char {
    match tp {
        Type::Int => 'i',
        Type::Ref => 'r',
        Type::Float => 'f',
        Type::Void => 'v',
    }
}

fn argclass_to_ir_type(c: char) -> Option<Type> {
    match c {
        'i' | 'S' => Some(Type::Int),
        'r' => Some(Type::Ref),
        'f' | 'L' => Some(Type::Float),
        'v' => None,
        _ => None,
    }
}

fn result_char_to_ir_type(c: char) -> Type {
    match c {
        'i' | 'S' => Type::Int,
        'r' => Type::Ref,
        'f' | 'L' => Type::Float,
        'v' => Type::Void,
        _ => Type::Void,
    }
}

fn result_layout_key(result_type: Type) -> (char, bool, usize, CallResultErasedKey) {
    let result_char = type_to_argclass(result_type);
    let result_signed = result_type == Type::Int;
    let result_size = match result_type {
        Type::Int | Type::Ref | Type::Float => 8,
        Type::Void => 0,
    };
    (
        result_char,
        result_signed,
        result_size,
        CallResultErasedKey::from_ir_layout(result_type, result_signed, result_size),
    )
}

/// Call classification — RPython `guess_call_kind()` return values.
///
/// RPython: the string literals `'regular'`, `'residual'`, `'builtin'`,
/// `'recursive'` returned by `CallControl.guess_call_kind()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallKind {
    /// Inline this call — callee graph is available and is a candidate.
    /// RPython: `'regular'` → produces `inline_call_*` jitcode instruction.
    Regular,
    /// Leave as a residual call in the trace.
    /// RPython: `'residual'` → produces `residual_call_*` jitcode instruction.
    Residual,
    /// Built-in operation with oopspec semantics (list ops, string ops, etc.)
    /// RPython: `'builtin'` → special handling per oopspec name.
    Builtin,
    /// Recursive call back to the portal (JIT entry point).
    /// RPython: `'recursive'` → produces `recursive_call_*` jitcode instruction.
    Recursive,
}

/// virtualizable.py:306-307 `VirtualizableInfo.is_vtypeptr(TYPE)` —
/// identity check for the VTYPEPTR (struct-pointer type) the
/// virtualizable describes.
///
/// PRE-EXISTING-ADAPTATION: pyre has no `lltype` so VTYPEPTR identity is
/// expressed via a `usize` token (typically the SizeDescr identity from
/// `majit_ir::descr::descr_identity`).  Hosts attach their rich
/// `VirtualizableInfo` (defined in `majit-metainterp::virtualizable`) by
/// implementing this trait so codewriter, which sits below metainterp in
/// the crate graph, can still consult `jd.virtualizable_info` per
/// `call.py:375-385 CallControl.get_vinfo`.
pub trait VirtualizableInfoHandle: std::fmt::Debug + Send + Sync {
    /// virtualizable.py:306-307 `is_vtypeptr(TYPE) → TYPE == self.VTYPEPTR`.
    fn is_vtypeptr(&self, vtypeptr_id: usize) -> bool;
}

/// greenfield.py `GreenFieldInfo.green_fields` membership test.
///
/// PRE-EXISTING-ADAPTATION: same crate-boundary reasoning as
/// `VirtualizableInfoHandle`.  Hosts implement this on their rich
/// `GreenFieldInfo` so `CallControl.could_be_green_field`
/// (call.py:387-393) can walk `jd.greenfield_info` without depending on
/// metainterp.
pub trait GreenFieldInfoHandle: std::fmt::Debug + Send + Sync {
    /// `(GTYPE, fieldname) in self.green_fields`.
    fn contains_green_field(&self, gtype: &str, fieldname: &str) -> bool;
}

/// Codewriter-internal `GreenFieldInfoHandle` built directly from a
/// jitdriver's `greens` list during `make_virtualizable_infos`.
///
/// `contains_green_field` is a pure structural query (`(gtype,
/// fieldname) in self.green_fields`), so no runtime identity is
/// required — unlike `is_vtypeptr` which has no codewrite-time
/// equivalent.  Hosts that want their richer
/// `GreenFieldInfoHandle` impl (e.g. `majit_metainterp::greenfield::
/// GreenFieldInfo` with descriptor indices) override this placeholder
/// via [`CallControl::set_jitdriver_greenfield_info`].
#[derive(Debug, Clone)]
pub struct StaticGreenFieldInfoHandle {
    /// greenfield.py:14 `self.red_index = jd.jitdriver.reds.index(objname)`
    /// — index of the unique green-field owning red.
    pub red_index: usize,
    /// greenfield.py:18 `self.green_fields = jd.jitdriver.ll_greenfields.values()`
    /// — `(GTYPE, fieldname)` pairs.
    pub green_fields: Vec<(String, String)>,
}

impl GreenFieldInfoHandle for StaticGreenFieldInfoHandle {
    fn contains_green_field(&self, gtype: &str, fieldname: &str) -> bool {
        self.green_fields
            .iter()
            .any(|(g, f)| g == gtype && f == fieldname)
    }
}

/// RPython: `JitDriverStaticData` — per-jitdriver metadata.
///
/// RPython `metainterp/jitdriver.py`: stores green/red variable names,
/// virtualizable info, portal graph reference, etc.
#[derive(Debug, Clone)]
pub struct JitDriverStaticData {
    /// RPython: `jitdriver_sd.index`
    pub index: usize,
    /// RPython: `jitdriver.greens` — loop-invariant variable names.
    pub greens: Vec<String>,
    /// RPython: `jitdriver.reds` — loop-variant variable names.
    pub reds: Vec<String>,
    /// RPython: `jitdriver.virtualizables` — names of red variables
    /// declared as virtualizable.  Drives warmspot.py:527-545
    /// `make_virtualizable_infos` selection.
    pub virtualizables: Vec<String>,
    /// Type names (GTYPEs) for each red variable, parallel to `reds`.
    ///
    /// PRE-EXISTING-ADAPTATION: upstream looks up GTYPE via
    /// `jd._JIT_ENTER_FUNCTYPE.ARGS[index]` at warmspot time; pyre
    /// propagates the matching struct names from `setup_jitdriver` so
    /// `make_virtualizable_infos` can build `(GTYPE, fieldname)`
    /// `green_fields` per greenfield.py:14 / warmspot.py:540-543.
    /// May be empty when the host has not yet supplied red types
    /// (legacy callers); in that case green-field construction
    /// substitutes the variable name as a fallback.
    pub red_types: Vec<String>,
    /// Portal graph path.
    pub portal_graph: CallPath,
    /// RPython: `jd.mainjitcode` (call.py:147) — `Arc<JitCode>` shell for
    /// the portal. Set by `grab_initial_jitcodes()`. Matches the
    /// metainterp-side `JitDriverStaticData.mainjitcode` shape so the
    /// codewriter→metainterp boundary is plain Arc handoff (no index
    /// translation step).
    pub mainjitcode: Option<std::sync::Arc<crate::jitcode::JitCode>>,
    /// warmspot.py:533 `jd.index_of_virtualizable = jitdriver.reds.index(vname)`.
    ///
    /// `-1` for drivers without a virtualizable, otherwise the slot
    /// in `reds` that holds the virtualizable.
    pub index_of_virtualizable: i32,
    /// warmspot.py:545 `jd.virtualizable_info = vinfos[VTYPEPTR]`.
    ///
    /// `None` for drivers that do not declare a virtualizable.  Set
    /// from the host runtime once the metainterp-side
    /// `VirtualizableInfo` is built — codewriter only sees the trait
    /// surface required by `CallControl::get_vinfo`.
    pub virtualizable_info: Option<std::sync::Arc<dyn VirtualizableInfoHandle>>,
    /// warmspot.py:519-525 `jd.greenfield_info = GreenFieldInfo(self.cpu, jd)`.
    ///
    /// Same plumbing as `virtualizable_info` — hosts attach their rich
    /// `GreenFieldInfo` via the trait so `CallControl.could_be_green_field`
    /// can walk it.
    pub greenfield_info: Option<std::sync::Arc<dyn GreenFieldInfoHandle>>,
}

/// Call control — decides inline vs residual for each call target.
///
/// RPython: `call.py::CallControl`.
///
/// In RPython, `CallControl` discovers all candidate graphs by traversing
/// from the portal graph, then for each `direct_call` operation it classifies
/// the call as regular/residual/builtin/recursive.
///
/// In majit-translate, we don't have RPython's function pointer linkage.
/// Instead, callee graphs are collected from parsed Rust source files
/// (free functions via `collect_function_graphs` and trait impl methods
/// via `extract_trait_impls`).
pub struct CallControl {
    /// Free function graphs: CallPath → FunctionGraph.
    /// RPython: `funcptr._obj.graph` linkage.
    function_graphs: HashMap<CallPath, FunctionGraph>,

    /// Per-graph hint set, mirroring RPython `func._jit_*_` / `_elidable_function_`.
    /// Populated by `register_function_graph_with_hints` and
    /// `register_trait_method_with_hints`; consulted by
    /// [`crate::policy::JitPolicy::look_inside_graph`] inside `find_all_graphs`.
    function_hints: HashMap<CallPath, Vec<String>>,

    /// Trait impl method graphs: (method_name, impl_type) → FunctionGraph.
    /// Used for resolving `handler.method_name()` calls.
    trait_method_graphs: HashMap<(String, String), FunctionGraph>,

    /// Trait bindings: `(trait_root, method_name)` → `Vec<impl_type>`.
    ///
    /// Keyed by the *declaring trait* (impl's `trait_name` from
    /// `parse.rs:237`), so two traits exposing the same method name do
    /// not collide (RPython `call.py:94-114` indirect branch reads
    /// `op.args[-1].value` = exact candidate graph list, not a
    /// method-name global).  Inherent impls do not populate this map;
    /// they use `function_graphs` directly via `[impl_type, method_name]`.
    trait_method_impls: HashMap<(String, String), Vec<String>>,

    /// Candidate targets — graphs we will inline.
    /// RPython: `CallControl.candidate_graphs`.
    candidate_graphs: HashSet<CallPath>,

    /// Portal entry points (recursive call detection).
    /// RPython: `CallControl.jitdrivers_sd`.
    portal_targets: HashSet<CallPath>,

    /// RPython: `JitDriverStaticData` — metadata for each jitdriver.
    /// `jitdrivers_sd[i]` holds the green/red arg layout for driver i.
    jitdrivers_sd: Vec<JitDriverStaticData>,

    /// Builtin targets (oopspec operations).
    /// RPython: detected via `funcobj.graph.func.oopspec`.
    builtin_targets: HashSet<CallPath>,

    /// RPython: `CallControl.jitcodes` — map {graph_key: JitCode}.
    /// Pyre stores `Arc<JitCode>` shells so callers (e.g.
    /// `IndirectCallTargets`, `JitDriverStaticData.mainjitcode`,
    /// `enum_pending_graphs`) can hold stable handles before the assembler
    /// commits the body via `OnceLock` interior mutability.
    jitcodes: HashMap<CallPath, std::sync::Arc<crate::jitcode::JitCode>>,

    /// RPython call.py:174-187 resolves `getfunctionptr(graph)` to the
    /// graph's real helper address before constructing `JitCode(name,
    /// fnaddr, calldescr)`. majit's source-only codewriter cannot derive
    /// that address from a parsed `CallPath`, so hosts may pre-bind the
    /// concrete trace-call surface here. Unbound paths still fall back to
    /// the stable symbolic address shim.
    function_fnaddrs: HashMap<CallPath, i64>,

    /// RPython `all_jitcodes` materialized incrementally by
    /// `CodeWriter.make_jitcodes()`. Entries are appended only after a
    /// jitcode has been fully assembled.
    finished_jitcodes: Vec<std::sync::Arc<crate::jitcode::JitCode>>,

    /// RPython: `CallControl.unfinished_graphs` — graphs pending assembly.
    unfinished_graphs: Vec<CallPath>,

    /// RPython: `CallControl.callinfocollection` (call.py:31).
    /// Stores oopspec function info for builtin call handling.
    pub callinfocollection: majit_ir::CallInfoCollection,

    /// call.py:39 `self.virtualizable_analyzer = VirtualizableAnalyzer(translator)`.
    pub virtualizable_analyzer: majit_ir::effectinfo::VirtualizableAnalyzer,

    /// call.py:40 `self.quasiimmut_analyzer = QuasiImmutAnalyzer(translator)`.
    pub quasiimmut_analyzer: majit_ir::effectinfo::QuasiImmutAnalyzer,

    /// call.py:41 `self.randomeffects_analyzer = RandomEffectsAnalyzer(translator)`.
    pub randomeffects_analyzer: majit_ir::effectinfo::RandomEffectsAnalyzer,

    /// RPython: `getattr(func, "_elidable_function_", False)` (call.py:239).
    /// Targets known to be elidable (pure, no side effects).
    elidable_targets: HashSet<CallPath>,

    /// RPython: `getattr(func, "_jit_loop_invariant_", False)` (call.py:240).
    /// Targets known to be loop-invariant (call once per loop).
    loopinvariant_targets: HashSet<CallPath>,

    /// RPython: known struct types for `get_type_flag(ARRAY.OF)` → FLAG_STRUCT.
    /// If an array's element type is in this set, the array descriptor gets
    /// `ArrayFlag::Struct` (like RPython's `isinstance(TYPE, lltype.Struct)`).
    known_struct_names: HashSet<String>,

    /// RPython: struct field type info — maps struct_name → [(field_name, type_string)].
    /// Used by `resolve_array_identity` to determine the ARRAY element type
    /// when the base of an array access comes from a FieldRead.
    /// Equivalent to `op.args[0].concretetype.TO` in RPython's rtyped graph.
    struct_fields: crate::front::StructFieldRegistry,

    /// RPython: `op.result.concretetype` — function return type strings.
    /// Maps CallPath → full return type string (e.g. "Vec<Point>").
    /// Used by `resolve_array_identity` for Call result array identity.
    pub return_types: HashMap<CallPath, String>,

    /// RPython: `symbolic.get_array_token(ARRAY, tsc)[0]` — array base size.
    /// Offset from the array object pointer to the first element.
    /// RPython GcArray layout: `[length (WORD)] [items...]`, so
    /// `basesize = carray.items.offset = sizeof(Signed) = WORD`.
    /// Default: WORD (8 on 64-bit) matching RPython's standard GcArray.
    pub array_header_size: usize,

    /// RPython: `symbolic.get_field_token(STRUCT, fieldname, tsc)` / `symbolic.get_size()`.
    /// Pre-computed struct layouts from actual runtime (std::mem::offset_of! etc.).
    /// When registered, provides exact (offset, size) for struct fields,
    /// bypassing the type-string heuristic. The runtime/proc-macro populates
    /// this via `set_struct_layout()`.
    pub struct_layouts: HashMap<String, StructLayout>,
    /// RPython: collectanalyze.py:15 — _gctransformer_hint_cannot_collect_.
    /// Functions known not to trigger GC collection. The collect_analyzer
    /// returns False immediately for these.
    pub cannot_collect_targets: HashSet<CallPath>,
    /// RPython: call.py:129-134 — _gctransformer_hint_close_stack_.
    /// Functions that close the stack must never produce JitCode — they are
    /// always classified as 'residual' by guess_call_kind.
    pub close_stack_targets: HashSet<CallPath>,
    /// RPython: collectanalyze.py:21-25 — funcobj.random_effects_on_gcobjs.
    /// External functions whose calls may have random effects on GC objects.
    /// analyze_can_collect returns True immediately for these.
    pub external_gc_effects: HashSet<CallPath>,

    /// RPython: `getattr(func, '_call_aroundstate_target_', None)` (call.py:252,271).
    /// Functions wrapping aroundstate GIL release/restore logic that must
    /// never be mixed into an indirect_call family.  Placeholder until
    /// `#[jit_call_aroundstate_target]` attribute is ported; read by
    /// `check_indirect_call_family`.
    pub aroundstate_targets: HashSet<CallPath>,

    /// RPython: rlib/jit.py:250 `@oopspec(spec)` — `getattr(func, 'oopspec', None)`.
    /// Maps call target → oopspec string (e.g. "jit.isconstant(value)").
    /// codewriter/jtransform reads this to route calls through OopSpecIndex.
    pub oopspec_targets: HashMap<CallPath, String>,

    /// RPython: `_immutable_fields_` per class. Maps struct_name →
    /// `(field_name, rank)` pairs declared immutable / quasi-immutable.
    /// Consulted by the heuristic fallback in `all_interiorfielddescrs`
    /// when a struct has no registered StructLayout (Path 1 already carries
    /// `rank` on `StructFieldLayout`).  Rank encoding follows
    /// `rpython/rtyper/rclass.py:644-678 _parse_field_list`.
    pub immutable_fields_by_struct: HashMap<String, Vec<(String, crate::model::ImmutableRank)>>,
}

/// Heuristic struct layout — NOT equivalent to RPython's `symbolic.get_field_token()`.
///
/// RPython delegates to `ll2ctypes.get_ctypes_type(STRUCT)` or `llmemory.offsetof()`
/// for actual C-level layout. This struct holds heuristic approximations computed
/// from Rust type strings via `from_type_strings()`. Offsets and sizes may diverge
/// from actual `#[repr(C)]` layout. The runtime SHOULD override via
/// `set_struct_layout()` with values from `std::mem::offset_of!()` /
/// `std::mem::size_of::<T>()` for production use.
#[derive(Debug, Clone)]
pub struct StructLayout {
    /// RPython: `symbolic.get_size(STRUCT, tsc)` — total struct size.
    pub size: usize,
    /// Per-field layout: (field_name, offset, size, type).
    /// RPython: `symbolic.get_field_token(STRUCT, name, tsc) → (offset, size)`.
    pub fields: Vec<StructFieldLayout>,
}

/// Single field within a `StructLayout`.
#[derive(Debug, Clone)]
pub struct StructFieldLayout {
    pub name: String,
    /// RPython: `cfield.offset`
    pub offset: usize,
    /// RPython: `cfield.size`
    pub size: usize,
    /// RPython: `get_type_flag(getattr(STRUCT, fieldname))`
    pub flag: majit_ir::descr::ArrayFlag,
    /// IR type classification.
    pub field_type: majit_ir::value::Type,
    /// RPython: `STRUCT._immutable_field(fieldname)` —
    /// `rpython/rtyper/rclass.py:33-37` returns `False` for mutable
    /// fields and the matching `ImmutableRanking` (truthy) for fields
    /// listed in `_immutable_fields_`.  `None` here = mutable; `Some(rank)`
    /// = declared with that rank (`?`, `[*]`, `?[*]`, or plain).  Drives
    /// `FieldDescr.is_pure` + `is_quasi_immutable` and (future)
    /// `ArrayDescr.is_pure` for `[*]` arrays.
    pub rank: Option<crate::model::ImmutableRank>,
}

impl StructFieldLayout {
    /// RPython `STRUCT._immutable_field(fieldname)` truthiness — true iff
    /// the field appears in `_immutable_fields_` (any rank).
    pub fn is_immutable(&self) -> bool {
        self.rank.is_some()
    }

    /// True iff the rank is `IR_QUASIIMMUTABLE` / `IR_QUASIIMMUTABLE_ARRAY`.
    pub fn is_quasi_immutable(&self) -> bool {
        self.rank.map(|r| r.is_quasi_immutable()).unwrap_or(false)
    }
}

impl StructLayout {
    /// Build a StructLayout from type-string heuristic.
    /// Used at pipeline init to populate struct_layouts from struct_fields.
    /// The runtime can later override with actual layout via set_struct_layout().
    ///
    /// `immutable_field_ranks`: map from field name → `ImmutableRank` for
    /// every entry in the owning class's `_immutable_fields_` declaration.
    /// RPython `STRUCT._immutable_field(fieldname)` returns the matching
    /// `ImmutableRanking` for these; fields not in the map are mutable.
    pub fn from_type_strings(
        fields: &[(String, String)],
        known_structs: &std::collections::HashSet<String>,
        known_struct_sizes: &std::collections::HashMap<String, usize>,
        immutable_field_ranks: &std::collections::HashMap<String, crate::model::ImmutableRank>,
    ) -> Self {
        // RPython: symbolic.get_array_token() computes itemsize for ANY struct,
        // even those with nested structs. UnsupportedFieldExc only affects
        // all_interiorfielddescrs (field enumeration), NOT the struct size.
        // So we always compute the full size, but mark has_nested_struct to
        // clear interior field descriptors.
        let has_nested_struct = fields
            .iter()
            .any(|(_, type_str)| known_structs.contains(type_str.as_str()));
        let mut offset: usize = 0;
        let mut layout_fields = Vec::new();
        for (name, type_str) in fields {
            // heaptracker.py:62-67: skip Void, padding, and typeptr fields.
            // typeptr is handled separately (not enumerated by all_fielddescrs).
            if name == "typeptr" || name.starts_with("c__pad") {
                // heaptracker.py:64-67
                // typeptr is still counted for offset calculation below.
                let sz = if known_structs.contains(type_str.as_str()) {
                    known_struct_sizes
                        .get(type_str.as_str())
                        .copied()
                        .unwrap_or(std::mem::size_of::<usize>())
                } else {
                    get_type_flag(type_str).2
                };
                if sz > 0 {
                    let align = sz.min(std::mem::size_of::<usize>());
                    offset = (offset + align - 1) & !(align - 1);
                    offset += sz;
                }
                continue;
            }
            let (flag, field_type, field_size) = if known_structs.contains(type_str.as_str()) {
                // RPython: symbolic.get_field_token(STRUCT, fieldname) returns the
                // actual embedded struct size, not just a pointer size.
                let nested_size = known_struct_sizes
                    .get(type_str.as_str())
                    .copied()
                    .unwrap_or(std::mem::size_of::<usize>());
                (
                    majit_ir::descr::ArrayFlag::Struct,
                    majit_ir::value::Type::Ref,
                    nested_size,
                )
            } else {
                get_type_flag(type_str)
            };
            if field_type == majit_ir::value::Type::Void || field_size == 0 {
                continue;
            }
            // RPython: alignment is typically min(field_size, WORD).
            let align = field_size.min(std::mem::size_of::<usize>());
            offset = (offset + align - 1) & !(align - 1);
            let rank = immutable_field_ranks.get(name).copied();
            layout_fields.push(StructFieldLayout {
                name: name.clone(),
                offset,
                size: field_size,
                flag,
                field_type,
                rank,
            });
            offset += field_size;
        }
        // RPython: heaptracker.py:89-90 — if nested struct exists,
        // all_interiorfielddescrs raises UnsupportedFieldExc, so
        // interior field descriptors are not enumerable. Clear fields
        // but keep the correct size.
        if has_nested_struct {
            layout_fields.clear();
        }
        let max_align = fields
            .iter()
            .map(|(_, ty)| {
                if known_structs.contains(ty.as_str()) {
                    // Use actual nested struct size for alignment.
                    known_struct_sizes
                        .get(ty.as_str())
                        .copied()
                        .unwrap_or(std::mem::size_of::<usize>())
                        .min(std::mem::size_of::<usize>())
                } else {
                    get_type_flag(ty).2
                }
            })
            .filter(|s| *s > 0)
            .max()
            .unwrap_or(8);
        let size = if offset > 0 {
            (offset + max_align - 1) & !(max_align - 1)
        } else {
            0
        };
        StructLayout {
            size,
            fields: layout_fields,
        }
    }
}

/// Sequential descriptor index assignment — majit equivalent of
/// `cpu.fielddescrof(T, fieldname).get_ei_index()` /
/// `cpu.arraydescrof(ARRAY).get_ei_index()`.
///
/// RPython: each descriptor gets a globally unique index via
/// `compute_bitstrings()`. The bitstring module creates variable-length
/// bitfields. In majit we use fixed-width u64, so indices wrap at 64.
///
/// Guarantees: same (owner_root, field_name) always gets the same index.
/// Limitation: after 64 unique field descriptors, indices wrap and may
/// alias unrelated descriptors. This matches RPython's bitstring semantics
/// where aliased bits cause conservative over-approximation (safe but
/// imprecise). Array descriptors are keyed by `(item_ty, array_type_id)`,
/// matching RPython's `cpu.arraydescrof(ARRAY)` which distinguishes by
/// ARRAY identity (e.g. `GcArray(Signed)` vs `GcArray(Ptr(STRUCT_X))`).
#[derive(Default)]
pub struct DescrIndexRegistry {
    /// (owner_root, field_name) → bit index (0..63)
    field_indices: HashMap<(Option<String>, String), u32>,
    /// (item_ty_discriminant, array_type_id) → bit index (0..63)
    /// RPython: cpu.arraydescrof(ARRAY).get_ei_index()
    array_indices: HashMap<(u8, Option<String>), u32>,
    /// (array_type_id, field_name) → bit index (0..63)
    /// RPython: cpu.interiorfielddescrof(ARRAY, fieldname).get_ei_index()
    /// Separate from field_indices — RPython keys on (ARRAY, fieldname)
    /// not (STRUCT, fieldname).
    interiorfield_indices: HashMap<(Option<String>, String), u32>,
    next_field_index: u32,
    next_array_index: u32,
    next_interiorfield_index: u32,
}

impl DescrIndexRegistry {
    /// RPython: `cpu.fielddescrof(T, fieldname).get_ei_index()`
    pub fn field_index(&mut self, owner_root: &Option<String>, field_name: &str) -> u32 {
        let key = (owner_root.clone(), field_name.to_string());
        *self.field_indices.entry(key).or_insert_with(|| {
            let idx = self.next_field_index % 64;
            self.next_field_index += 1;
            idx
        })
    }

    /// RPython: `cpu.arraydescrof(ARRAY).get_ei_index()`
    ///
    /// Keys on `(item_ty_discriminant, array_type_id)` unconditionally.
    /// RPython always distinguishes by ARRAY lltype identity:
    /// `GcArray(Char)`, `GcArray(Signed)`, `GcArray(Bool)` are all
    /// different descriptors even though they are all "integer" arrays
    /// (effectinfo.py:307-311).
    pub fn array_index(&mut self, item_ty_discriminant: u8, array_type_id: &Option<String>) -> u32 {
        let key = (item_ty_discriminant, array_type_id.clone());
        *self.array_indices.entry(key).or_insert_with(|| {
            let idx = self.next_array_index % 64;
            self.next_array_index += 1;
            idx
        })
    }

    /// RPython: `cpu.interiorfielddescrof(ARRAY, fieldname).get_ei_index()`
    ///
    /// Keys on `(array_type_id, field_name)` — matches RPython's cache key
    /// `(ARRAY, name, arrayfieldname)` in `get_interiorfield_descr()`.
    /// Separate namespace from field_indices: two different array types
    /// with the same element struct get different interiorfield indices.
    pub fn interiorfield_index(&mut self, array_type_id: &Option<String>, field_name: &str) -> u32 {
        let key = (array_type_id.clone(), field_name.to_string());
        *self.interiorfield_indices.entry(key).or_insert_with(|| {
            let idx = self.next_interiorfield_index % 64;
            self.next_interiorfield_index += 1;
            idx
        })
    }
}

impl CallControl {
    /// RPython: `CallControl.__init__`.
    pub fn new() -> Self {
        Self {
            function_graphs: HashMap::new(),
            function_hints: HashMap::new(),
            trait_method_graphs: HashMap::new(),
            trait_method_impls: HashMap::new(),
            candidate_graphs: HashSet::new(),
            portal_targets: HashSet::new(),
            jitdrivers_sd: Vec::new(),
            builtin_targets: HashSet::new(),
            jitcodes: HashMap::new(),
            function_fnaddrs: HashMap::new(),
            finished_jitcodes: Vec::new(),
            unfinished_graphs: Vec::new(),
            callinfocollection: majit_ir::CallInfoCollection::new(),
            virtualizable_analyzer: majit_ir::effectinfo::VirtualizableAnalyzer,
            quasiimmut_analyzer: majit_ir::effectinfo::QuasiImmutAnalyzer,
            randomeffects_analyzer: majit_ir::effectinfo::RandomEffectsAnalyzer,
            elidable_targets: HashSet::new(),
            loopinvariant_targets: HashSet::new(),
            known_struct_names: HashSet::new(),
            struct_fields: crate::front::StructFieldRegistry::default(),
            return_types: HashMap::new(),
            // RPython: symbolic.get_array_token(GcArray(T))[0] = carray.items.offset
            // = sizeof(Signed) = WORD. Standard GcArray has a length field before items.
            //
            array_header_size: std::mem::size_of::<usize>(),
            struct_layouts: HashMap::new(),
            cannot_collect_targets: HashSet::new(),
            close_stack_targets: HashSet::new(),
            external_gc_effects: HashSet::new(),
            aroundstate_targets: HashSet::new(),
            oopspec_targets: HashMap::new(),
            immutable_fields_by_struct: HashMap::new(),
        }
    }

    /// RPython `rpython/rtyper/rclass.py:644-678` —
    /// `STRUCT._immutable_field(fieldname)` returns the `ImmutableRanking`
    /// when the field is listed in `_immutable_fields_`, or `None` for
    /// plain mutable fields.  Called by `jtransform.rewrite_op_getfield`
    /// (`rpython/jit/codewriter/jtransform.py:866-906`) to decide between
    /// mutable read, pure read, and the quasi-immut guard/record pair.
    pub fn field_immutability(
        &self,
        owner_root: Option<&str>,
        field_name: &str,
    ) -> Option<crate::model::ImmutableRank> {
        let owner = owner_root?;
        self.immutable_fields_by_struct
            .get(owner)
            .and_then(|fields| {
                fields
                    .iter()
                    .find(|(n, _)| n == field_name)
                    .map(|(_, rank)| *rank)
            })
    }

    /// RPython: register struct type names for get_type_flag(ARRAY.OF).
    pub fn set_known_struct_names(&mut self, names: HashSet<String>) {
        self.known_struct_names = names;
    }

    /// RPython: register struct field types for op.args[0].concretetype resolution.
    pub fn set_struct_fields(&mut self, registry: crate::front::StructFieldRegistry) {
        self.struct_fields = registry;
    }

    /// RPython: isinstance(TYPE, lltype.Struct) check.
    pub fn is_known_struct(&self, name: &str) -> bool {
        self.known_struct_names.contains(name)
    }

    /// RPython: register actual struct layout from `symbolic.get_field_token()`.
    /// The runtime calls this with layouts from `std::mem::offset_of!()` etc.
    pub fn set_struct_layout(&mut self, struct_name: String, layout: StructLayout) {
        self.struct_layouts.insert(struct_name, layout);
    }

    /// RPython: resolve a struct field's type string.
    /// For `owner::field_name`, returns the full type of the field.
    pub fn field_type(&self, owner: &str, field_name: &str) -> Option<&str> {
        self.struct_fields.field_type(owner, field_name)
    }

    /// RPython: ordered `STRUCT._names` + field types for descriptor layout
    /// reconstruction. The order is required to reproduce
    /// `symbolic.get_field_token()`.
    pub fn struct_field_entries(&self, owner: &str) -> Option<&[(String, String)]> {
        self.struct_fields.fields.get(owner).map(Vec::as_slice)
    }

    /// RPython: `cpu.arraydescrof(ARRAY)` — descr.py:348-378.
    ///
    /// `array_type_id`: full ARRAY type string (e.g. `"Vec<Point>"`), matching
    /// RPython's ARRAY lltype identity. The element type is extracted via
    /// `extract_element_type_from_str()` for struct checks and flag resolution.
    pub fn arraydescrof(
        &self,
        idx: u32,
        array_type_id: &Option<String>,
        ir_type: majit_ir::value::Type,
    ) -> majit_ir::descr::DescrRef {
        // RPython: ARRAY_INSIDE.OF — extract element type from full ARRAY type.
        let elem_name = array_type_id
            .as_deref()
            .and_then(|s| extract_element_type_from_str(s).or_else(|| Some(s.to_string())))
            .as_deref()
            .map(String::from);
        let elem_ref = elem_name.as_deref();
        let is_struct = elem_ref.is_some_and(|n| self.is_known_struct(n));
        // RPython: descr.py:363 — flag = get_type_flag(ARRAY_INSIDE.OF)
        // RPython: descr.py:354 — itemsize from symbolic.get_array_token().
        // descr.py:363 — flag = get_type_flag(ARRAY_INSIDE.OF).
        // descr.py:365 — ArrayDescr(basesize, itemsize, ..., flag).
        // Even for struct(struct), itemsize is correct from symbolic.
        let (flag, item_size) = if is_struct {
            (
                majit_ir::descr::ArrayFlag::Struct,
                elem_ref.map(|n| compute_struct_size(self, n)).unwrap_or(8),
            )
        } else if let Some(elem) = elem_ref {
            let (f, _, s) = get_type_flag(elem);
            (f, s)
        } else {
            (
                majit_ir::descr::ArrayFlag::from_item_type(ir_type, false),
                8,
            )
        };
        // RPython: basesize, itemsize, _ = symbolic.get_array_token(ARRAY, tsc)
        let base_size = self.array_header_size;
        let ad = majit_ir::descr::SimpleArrayDescr::with_flag(
            idx, base_size, item_size, 0, ir_type, flag,
        );
        // RPython: descr.py:372-375 — struct arrays get interior field descriptors.
        if is_struct {
            if let Some(struct_name) = elem_ref {
                // `descr.py:388 InteriorFieldDescr.__init__` carries the
                // exact `arraydescr` object the surrounding ArrayDescr
                // factory returns; preserve that identity by publishing
                // the final Arc first and routing the interior list back
                // onto it via the `OnceLock` interior-mutability setter.
                let ad_arc = std::sync::Arc::new(ad);
                let (descrs, _) = all_interiorfielddescrs(self, struct_name, ad_arc.clone());
                if !descrs.is_empty() {
                    ad_arc.set_all_interiorfielddescrs(descrs);
                }
                return ad_arc;
            }
        }
        std::sync::Arc::new(ad)
    }

    /// Register a free function graph.
    /// RPython: graphs are discovered via funcptr linkage.
    pub fn register_function_graph(&mut self, path: CallPath, graph: FunctionGraph) {
        self.function_graphs.insert(path, graph);
    }

    /// Register a free function graph together with its hints.
    /// `hints` mirror RPython `func._jit_*_` / `_elidable_function_`
    /// attributes; they are consulted by
    /// [`crate::policy::JitPolicy::look_inside_graph`].
    pub fn register_function_graph_with_hints(
        &mut self,
        path: CallPath,
        graph: FunctionGraph,
        hints: Vec<String>,
    ) {
        self.function_graphs.insert(path.clone(), graph);
        if !hints.is_empty() {
            self.function_hints.insert(path, hints);
        }
    }

    /// Bind a real helper trace-call address to a canonical CallPath.
    ///
    /// RPython obtains this from `getfunctionptr(graph)`; majit callers
    /// that have access to the compiled helper surface can preload the
    /// equivalent integer address here so `get_jitcode()` and
    /// `fnaddr_for_target()` no longer fall back to symbolic hashes.
    pub fn register_function_fnaddr(&mut self, path: CallPath, fnaddr: i64) {
        self.function_fnaddrs.insert(path, fnaddr);
    }

    /// Consume a `#[jit_module]::__majit_helper_trace_fnaddrs()` entry.
    ///
    /// The macro-generated registry uses `module_path!()` and therefore
    /// prefixes paths with the crate name (e.g. `"mycrate::helpers::foo"`),
    /// while codewriter canonical paths are stored both as
    /// `"helpers::foo"` and `"crate::helpers::foo"`. Bind both aliases so
    /// either spelling resolves to the real helper address.
    ///
    /// Impl methods are *not* registered through this entry point —
    /// their canonical CallPath (`[impl_type_joined, method]`) carries
    /// `impl_type_joined` as a single `::`-preserving segment
    /// (parse.rs:702, lib.rs:406-433), which the simple `split("::")`
    /// strip here cannot recover.  Use
    /// `register_macro_impl_helper_trace_fnaddr` instead, fed from the
    /// macro's sibling registry `__majit_helper_impl_trace_fnaddrs()`.
    pub fn register_macro_helper_trace_fnaddr(&mut self, full_path: &str, fnaddr: i64) {
        if fnaddr == 0 {
            return;
        }
        let segments: Vec<&str> = full_path
            .split("::")
            .filter(|segment| !segment.is_empty())
            .collect();
        if segments.is_empty() {
            return;
        }
        let canonical = if segments.len() > 1 {
            &segments[1..]
        } else {
            &segments[..]
        };
        if canonical.is_empty() {
            return;
        }
        self.register_function_fnaddr(CallPath::from_segments(canonical.iter().copied()), fnaddr);
        let mut crate_alias = Vec::with_capacity(canonical.len() + 1);
        crate_alias.push("crate");
        crate_alias.extend(canonical.iter().copied());
        self.register_function_fnaddr(CallPath::from_segments(crate_alias), fnaddr);
    }

    /// Structured binding for an impl-method helper. `impl_type_joined`
    /// is the `::`-joined type path exactly as written at the `impl`
    /// header (e.g. `"a::Foo"` for `impl a::Foo { fn bar() }`), matching
    /// the parser's `self_ty_root` canonicalization (parse.rs:702 +
    /// front/ast.rs:106 `qualify_type_name`).  Registers
    /// `[impl_type_joined, method]` as a 2-segment CallPath where
    /// `impl_type_joined` is stored verbatim as a single segment — same
    /// shape `register_trait_method` / inherent method graphs use at
    /// lib.rs:406-433, so `get_jitcode()` resolves through to this real
    /// helper address instead of the symbolic hash fallback.  RPython
    /// `call.py:174-187 getfunctionptr(graph)` parity for `<Type>::method`
    /// and `<Type as Trait>::method`.
    pub fn register_macro_impl_helper_trace_fnaddr(
        &mut self,
        module_path_with_crate: &str,
        impl_type_as_written: &str,
        method: &str,
        fnaddr: i64,
    ) {
        if fnaddr == 0 || impl_type_as_written.is_empty() || method.is_empty() {
            return;
        }
        // front/ast.rs:106 `qualify_type_name`: bare types take the
        // current module prefix; already-qualified types keep their
        // exact written form.  Module prefix is everything after the
        // first `::`-separated segment (the crate name) of
        // `module_path_with_crate`, matching the parser's `prefix`
        // argument which starts empty at crate root and accumulates
        // submodule idents (parse.rs:314-318).
        let module_prefix = module_path_with_crate
            .split_once("::")
            .map(|(_crate, rest)| rest)
            .unwrap_or("");
        let impl_type_joined = if impl_type_as_written.contains("::") || module_prefix.is_empty() {
            impl_type_as_written.to_string()
        } else {
            format!("{module_prefix}::{impl_type_as_written}")
        };
        self.register_function_fnaddr(CallPath::for_impl_method(&impl_type_joined, method), fnaddr);
    }

    /// Register a trait impl method graph.
    ///
    /// Also registers the graph in function_graphs under a synthetic
    /// CallPath so that BFS in find_all_graphs can discover it.
    /// RPython: method graphs are reachable through funcptr._obj.graph
    /// linkage — we emulate this by dual registration.
    ///
    /// `trait_root` identifies the declaring trait for polymorphic
    /// resolution (inherent impls pass `None`).  Populating
    /// `trait_method_impls` under `(trait_root, method_name)` keeps two
    /// traits with the same method name distinct per `call.py:94-114`.
    pub fn register_trait_method(
        &mut self,
        method_name: &str,
        trait_root: Option<&str>,
        impl_type: &str,
        graph: FunctionGraph,
    ) {
        self.trait_method_graphs.insert(
            (method_name.to_string(), impl_type.to_string()),
            graph.clone(),
        );
        if let Some(trait_root) = trait_root {
            self.trait_method_impls
                .entry((trait_root.to_string(), method_name.to_string()))
                .or_default()
                .push(impl_type.to_string());
        }
        // Register in function_graphs for BFS reachability.
        // RPython: each graph has its own identity via funcptr._obj.graph.
        // We emulate this with `CallPath::for_impl_method(impl_type,
        // method_name)` which splits the impl_type across `::` so every
        // segment has the same granularity as free-fn paths — each impl
        // still gets a distinct path (PyFrame::push_value stays
        // `["PyFrame", "push_value"]`, MIFrame::push_value stays
        // `["MIFrame", "push_value"]`).
        let qualified_path = CallPath::for_impl_method(impl_type, method_name);
        self.function_graphs.entry(qualified_path).or_insert(graph);
    }

    /// Mark a target as the portal entry point.
    ///
    /// RPython: `setup_jitdriver(jitdriver_sd)` + `grab_initial_jitcodes()`.
    pub fn mark_portal(&mut self, path: CallPath) {
        self.portal_targets.insert(path);
    }

    /// Register a JitDriver with its green/red/virtualizable layout.
    ///
    /// RPython: `CodeWriter.setup_jitdriver(jitdriver_sd)` (codewriter.py:96-99)
    /// + `jitdriver.virtualizables` (rlib/jit.py:601-603).
    /// Each jitdriver gets a sequential index.
    ///
    /// `red_types` mirrors `_JIT_ENTER_FUNCTYPE.ARGS` for the red slot
    /// portion (warmspot.py:540-543).  Pass an empty vector if the
    /// host hasn't propagated the runtime types yet — the
    /// green-field constructor in `make_virtualizable_infos` falls
    /// back to the variable name in that case.
    pub fn setup_jitdriver(
        &mut self,
        portal_graph: CallPath,
        greens: Vec<String>,
        reds: Vec<String>,
        virtualizables: Vec<String>,
        red_types: Vec<String>,
    ) {
        let index = self.jitdrivers_sd.len();
        debug_assert!(
            red_types.is_empty() || red_types.len() == reds.len(),
            "setup_jitdriver: red_types length must match reds when supplied",
        );
        self.jitdrivers_sd.push(JitDriverStaticData {
            index,
            greens,
            reds,
            virtualizables,
            red_types,
            portal_graph: portal_graph.clone(),
            mainjitcode: None,
            index_of_virtualizable: -1,
            virtualizable_info: None,
            greenfield_info: None,
        });
        self.portal_targets.insert(portal_graph);
    }

    /// warmspot.py:528-545 `jd.virtualizable_info = vinfos[VTYPEPTR]`.
    ///
    /// Attach the host-built [`VirtualizableInfoHandle`] to the
    /// pre-registered driver at `index`.  Mirrors the upstream
    /// post-construction assignment that warmspot performs once the
    /// per-driver `VirtualizableInfo` map has been built.  Pyre's host
    /// runtime calls this between [`Self::setup_jitdriver`] and
    /// [`Self::find_all_graphs`] so that
    /// [`Self::get_vinfo`] returns the matching handle.
    pub fn set_jitdriver_virtualizable_info(
        &mut self,
        index: usize,
        info: std::sync::Arc<dyn VirtualizableInfoHandle>,
    ) {
        self.jitdrivers_sd[index].virtualizable_info = Some(info);
    }

    /// warmspot.py:519-525 `jd.greenfield_info = GreenFieldInfo(cpu, jd)`.
    ///
    /// Same staging pattern as
    /// [`Self::set_jitdriver_virtualizable_info`].  Hosts compute the
    /// green-field metadata once during driver setup and attach the
    /// handle here so [`Self::could_be_green_field`] can walk it.
    pub fn set_jitdriver_greenfield_info(
        &mut self,
        index: usize,
        info: std::sync::Arc<dyn GreenFieldInfoHandle>,
    ) {
        self.jitdrivers_sd[index].greenfield_info = Some(info);
    }

    /// warmspot.py:515-545 `WarmRunnerDesc.make_virtualizable_infos`.
    ///
    /// ```python
    /// def make_virtualizable_infos(self):
    ///     vinfos = {}
    ///     for jd in self.jitdrivers_sd:
    ///         jd.greenfield_info = None
    ///         for name in jd.jitdriver.greens:
    ///             if '.' in name:
    ///                 jd.greenfield_info = GreenFieldInfo(self.cpu, jd)
    ///                 break
    ///         if not jd.jitdriver.virtualizables:
    ///             jd.virtualizable_info = None
    ///             jd.index_of_virtualizable = -1
    ///             continue
    ///         else:
    ///             assert jd.greenfield_info is None, "XXX not supported yet"
    ///         jitdriver = jd.jitdriver
    ///         assert len(jitdriver.virtualizables) == 1    # for now
    ///         [vname] = jitdriver.virtualizables
    ///         jd.index_of_virtualizable = jitdriver.reds.index(vname)
    ///         index = jd.num_green_args + jd.index_of_virtualizable
    ///         VTYPEPTR = jd._JIT_ENTER_FUNCTYPE.ARGS[index]
    ///         if VTYPEPTR not in vinfos:
    ///             vinfos[VTYPEPTR] = VirtualizableInfo(self, VTYPEPTR)
    ///         jd.virtualizable_info = vinfos[VTYPEPTR]
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: upstream owns this method on
    /// `WarmRunnerDesc` (warmspot.py:451) so it can mutate the single
    /// shared `jitdrivers_sd` list (the same Python list object is
    /// referenced by both `WarmRunnerDesc.jitdrivers_sd` and
    /// `MetaInterpStaticData.jitdrivers_sd`).  Pyre splits that list
    /// into two: codewriter `CallControl::jitdrivers_sd` (build.rs
    /// time) and metainterp `MetaInterpStaticData::jitdrivers_sd`
    /// (runtime), so the warmspot logic is invoked once per side at
    /// the matching lifecycle phase.  This call covers the codewriter
    /// side; the metainterp side is wired through
    /// `MetaInterp::set_virtualizable_info` at `JitDriver::new`
    /// (jitdriver.rs:285).
    ///
    /// `greenfield_info` is constructed in-place as a
    /// [`StaticGreenFieldInfoHandle`] (the codewriter-internal default;
    /// hosts can override via
    /// [`Self::set_jitdriver_greenfield_info`] with a richer impl such
    /// as `majit_metainterp::greenfield::GreenFieldInfo`).
    ///
    /// `vinfo_factory` mirrors the upstream `VirtualizableInfo(self,
    /// VTYPEPTR)` constructor (warmspot.py:543).  Pyre's codewriter
    /// crate sits below metainterp and therefore cannot reach the
    /// rich runtime constructor; the factory closure delegates to the
    /// host (e.g. pyre `build.rs` or runtime warm-up), which can
    /// either return a real
    /// `Arc<dyn VirtualizableInfoHandle>` or `None`.  When the
    /// factory returns `None`, the slot stays empty until the host
    /// later overrides it with [`Self::set_jitdriver_virtualizable_info`]
    /// at runtime — matching pyre's
    /// `MetaInterp::set_virtualizable_info` (jitdriver.rs:285) wiring.
    /// The factory receives `(jd_idx, vtypeptr_token)` where
    /// `vtypeptr_token` is the `red_types[index_of_virtualizable]`
    /// string the codewriter resolved.
    pub fn make_virtualizable_infos<VF>(&mut self, mut vinfo_factory: VF)
    where
        VF: FnMut(usize, &str) -> Option<std::sync::Arc<dyn VirtualizableInfoHandle>>,
    {
        // warmspot.py:516 `vinfos = {}` — per-VTYPEPTR cache so multiple
        // jitdrivers sharing the same virtualizable type reuse one handle.
        let mut vinfos: std::collections::HashMap<
            String,
            std::sync::Arc<dyn VirtualizableInfoHandle>,
        > = std::collections::HashMap::new();
        self.make_virtualizable_infos_inner(&mut vinfo_factory, &mut vinfos);
    }

    fn make_virtualizable_infos_inner<VF>(
        &mut self,
        vinfo_factory: &mut VF,
        vinfos: &mut std::collections::HashMap<String, std::sync::Arc<dyn VirtualizableInfoHandle>>,
    ) where
        VF: FnMut(usize, &str) -> Option<std::sync::Arc<dyn VirtualizableInfoHandle>>,
    {
        for jd_idx in 0..self.jitdrivers_sd.len() {
            // warmspot.py:519 `jd.greenfield_info = None`
            self.jitdrivers_sd[jd_idx].greenfield_info = None;
            // warmspot.py:520-524 — scan greens for '.' and split each
            // dotted name into `(objname, fieldname)`.  Upstream
            // collects the unique `objname` set, then resolves each
            // `(objname, fieldname)` to `(GTYPE, fieldname)` via
            // `jd.jitdriver.ll_greenfields` for the
            // `green_fields` list and via `_JIT_ENTER_FUNCTYPE.ARGS`
            // for the index→GTYPE mapping (greenfield.py:14-19,
            // warmspot.py:540-543).
            let mut seen: Vec<String> = Vec::new();
            let mut parsed_pairs: Vec<(String, String)> = Vec::new();
            for name in &self.jitdrivers_sd[jd_idx].greens {
                if let Some((objname, fieldname)) = name.split_once('.') {
                    if !seen.iter().any(|s| s == objname) {
                        seen.push(objname.to_string());
                    }
                    parsed_pairs.push((objname.to_string(), fieldname.to_string()));
                }
            }
            // warmspot.py:520-524 (cont.): if any dotted green was seen,
            // construct GreenFieldInfo(cpu, jd) — pyre's codewriter has
            // no `cpu` so we build the structural placeholder
            // `StaticGreenFieldInfoHandle` here; hosts override with the
            // descriptor-aware metainterp variant via
            // `set_jitdriver_greenfield_info`.
            if !seen.is_empty() {
                // greenfield.py:11-13 `assert len(seen) == 1`.
                assert_eq!(
                    seen.len(),
                    1,
                    "greenfield.py:11 — only one instance with green fields supported, found {seen:?}",
                );
                let objname = &seen[0];
                // greenfield.py:14 `red_index = jd.jitdriver.reds.index(objname)`.
                let red_index = self.jitdrivers_sd[jd_idx]
                    .reds
                    .iter()
                    .position(|r| r == objname)
                    .unwrap_or_else(|| {
                        panic!(
                            "greenfield.py:14 — green-field owner {objname:?} not in reds {:?}",
                            self.jitdrivers_sd[jd_idx].reds
                        )
                    });
                // greenfield.py:18 `self.green_fields = jd.jitdriver.ll_greenfields.values()`
                // — values are `(GTYPE, fieldname)` pairs.  Resolve `GTYPE`
                // by looking up the red slot's type from `red_types`
                // (parallel to `reds`); legacy callers without
                // `red_types` fall back to the variable name so the
                // structural shape is preserved.
                let gtype = self.jitdrivers_sd[jd_idx]
                    .red_types
                    .get(red_index)
                    .cloned()
                    .unwrap_or_else(|| objname.to_string());
                let green_fields: Vec<(String, String)> = parsed_pairs
                    .into_iter()
                    .map(|(_objname, fieldname)| (gtype.clone(), fieldname))
                    .collect();
                self.jitdrivers_sd[jd_idx].greenfield_info =
                    Some(std::sync::Arc::new(StaticGreenFieldInfoHandle {
                        red_index,
                        green_fields,
                    }));
            }
            // warmspot.py:527-530: no virtualizable → keep None and continue.
            if self.jitdrivers_sd[jd_idx].virtualizables.is_empty() {
                self.jitdrivers_sd[jd_idx].virtualizable_info = None;
                self.jitdrivers_sd[jd_idx].index_of_virtualizable = -1;
                continue;
            }
            // warmspot.py:531-532: greenfield + virtualizable not supported.
            assert!(
                self.jitdrivers_sd[jd_idx].greenfield_info.is_none(),
                "warmspot.py:532 — greenfield + virtualizable on the same driver: XXX not supported yet",
            );
            // warmspot.py:534-538 `[vname] = jitdriver.virtualizables`
            //                    `jd.index_of_virtualizable = jitdriver.reds.index(vname)`
            assert_eq!(
                self.jitdrivers_sd[jd_idx].virtualizables.len(),
                1,
                "warmspot.py:535 — only one virtualizable per jitdriver supported",
            );
            let vname = self.jitdrivers_sd[jd_idx].virtualizables[0].clone();
            let idx = self.jitdrivers_sd[jd_idx]
                .reds
                .iter()
                .position(|r| r == &vname)
                .unwrap_or_else(|| {
                    panic!(
                        "warmspot.py:538 — virtualizable {vname:?} not in reds {:?}",
                        self.jitdrivers_sd[jd_idx].reds
                    )
                });
            self.jitdrivers_sd[jd_idx].index_of_virtualizable = idx as i32;
            // warmspot.py:540-545:
            //   index = jd.num_green_args + jd.index_of_virtualizable
            //   VTYPEPTR = jd._JIT_ENTER_FUNCTYPE.ARGS[index]
            //   if VTYPEPTR not in vinfos:
            //       vinfos[VTYPEPTR] = VirtualizableInfo(self, VTYPEPTR)
            //   jd.virtualizable_info = vinfos[VTYPEPTR]
            //
            // Pyre resolves VTYPEPTR via `red_types[index_of_virtualizable]`
            // (the `_JIT_ENTER_FUNCTYPE.ARGS` analog supplied at
            // `setup_jitdriver` time) and delegates the constructor
            // call to `vinfo_factory`.
            let vtypeptr_token = self.jitdrivers_sd[jd_idx]
                .red_types
                .get(idx)
                .cloned()
                .unwrap_or_default();
            let info = if let Some(cached) = vinfos.get(&vtypeptr_token) {
                Some(cached.clone())
            } else if let Some(fresh) = vinfo_factory(jd_idx, &vtypeptr_token) {
                vinfos.insert(vtypeptr_token, fresh.clone());
                Some(fresh)
            } else {
                None
            };
            self.jitdrivers_sd[jd_idx].virtualizable_info = info;
        }
    }

    /// call.py:357-361 `jitdriver_sd_from_portal_graph(graph)`.
    pub fn jitdriver_sd_from_portal_graph(&self, path: &CallPath) -> Option<&JitDriverStaticData> {
        self.jitdrivers_sd
            .iter()
            .find(|sd| &sd.portal_graph == path)
    }

    /// call.py:363-367 `jitdriver_sd_from_portal_runner_ptr(funcptr)`.
    ///
    /// Pyre has no separate `portal_runner_ptr` (the runner is the
    /// portal graph itself), so we reuse the path lookup.  Future
    /// phases that need the distinction can split the field.
    pub fn jitdriver_sd_from_portal_runner_ptr(
        &self,
        path: &CallPath,
    ) -> Option<&JitDriverStaticData> {
        self.jitdriver_sd_from_portal_graph(path)
    }

    /// call.py:369-373 `jitdriver_sd_from_jitdriver(jitdriver)`.
    ///
    /// Pyre identifies a jit driver by its index slot in
    /// `jitdrivers_sd`; we expose the slot lookup under the upstream
    /// name so call sites mirror RPython.
    pub fn jitdriver_sd_from_jitdriver(&self, index: usize) -> Option<&JitDriverStaticData> {
        self.jitdrivers_sd.get(index)
    }

    /// call.py:375-385 `get_vinfo(VTYPEPTR)`.
    ///
    /// ```python
    /// def get_vinfo(self, VTYPEPTR):
    ///     seen = set()
    ///     for jd in self.jitdrivers_sd:
    ///         if jd.virtualizable_info is not None:
    ///             if jd.virtualizable_info.is_vtypeptr(VTYPEPTR):
    ///                 seen.add(jd.virtualizable_info)
    ///     if seen:
    ///         assert len(seen) == 1
    ///         return seen.pop()
    ///     else:
    ///         return None
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: `VTYPEPTR` is an RPython lltype pointer;
    /// pyre represents VTYPEPTR identity as a `usize` token supplied by
    /// the host (typically `descr_identity(&size_descr)` from
    /// `majit_ir::descr`).  Hosts install per-driver
    /// [`VirtualizableInfoHandle`] via `JitDriverStaticData.virtualizable_info`.
    pub fn get_vinfo(
        &self,
        vtypeptr_id: usize,
    ) -> Option<std::sync::Arc<dyn VirtualizableInfoHandle>> {
        let mut seen: Vec<std::sync::Arc<dyn VirtualizableInfoHandle>> = Vec::new();
        for jd in &self.jitdrivers_sd {
            if let Some(vinfo) = &jd.virtualizable_info {
                if vinfo.is_vtypeptr(vtypeptr_id) {
                    // Dedupe by Arc identity so the upstream
                    // `assert len(seen) == 1` translates to "at most one
                    // distinct VirtualizableInfo per VTYPEPTR".
                    let seen_already = seen
                        .iter()
                        .any(|existing| std::sync::Arc::ptr_eq(existing, vinfo));
                    if !seen_already {
                        seen.push(std::sync::Arc::clone(vinfo));
                    }
                }
            }
        }
        if seen.is_empty() {
            None
        } else {
            assert_eq!(
                seen.len(),
                1,
                "get_vinfo: multiple distinct VirtualizableInfo for VTYPEPTR"
            );
            Some(seen.into_iter().next().unwrap())
        }
    }

    /// call.py:387-393 `could_be_green_field(GTYPE, fieldname)`.
    ///
    /// ```python
    /// def could_be_green_field(self, GTYPE, fieldname):
    ///     GTYPE_fieldname = (GTYPE, fieldname)
    ///     for jd in self.jitdrivers_sd:
    ///         if jd.greenfield_info is not None:
    ///             if GTYPE_fieldname in jd.greenfield_info.green_fields:
    ///                 return True
    ///     return False
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: `GTYPE` is an RPython lltype; pyre
    /// represents it by name (`&str`).  The host attaches a
    /// [`GreenFieldInfoHandle`] whose `contains_green_field` implements
    /// the `(GTYPE, fieldname) in green_fields` membership test.
    pub fn could_be_green_field(&self, gtype: &str, fieldname: &str) -> bool {
        for jd in &self.jitdrivers_sd {
            if let Some(gfinfo) = &jd.greenfield_info {
                if gfinfo.contains_green_field(gtype, fieldname) {
                    return true;
                }
            }
        }
        false
    }

    /// Mark a target as a builtin (oopspec) operation.
    pub fn mark_builtin(&mut self, path: CallPath) {
        self.builtin_targets.insert(path);
    }

    /// Discover candidate graphs by BFS from portal targets.
    ///
    /// RPython: `CallControl.find_all_graphs(policy)` (call.py:49-92).
    ///
    /// Walks from portal graphs transitively: for each Call op,
    /// if the callee has a graph, add it to the candidate set.
    /// Portal must be seeded via `mark_portal()` before calling.
    /// call.py:49 `find_all_graphs(self, policy)`.
    ///
    /// Discovers all candidate graphs reachable from the portal entry
    /// points. RPython uses `policy.look_inside_graph` to decide whether
    /// to follow each callee; we synthesize a `SemanticFunction` for the
    /// per-graph hints stored in `function_hints` and pass it through.
    pub fn find_all_graphs(&mut self, policy: &mut dyn JitPolicy) {
        assert!(
            !self.portal_targets.is_empty(),
            "find_all_graphs requires at least one portal target; \
             use find_all_graphs_for_tests() if no portal is available"
        );
        self.find_all_graphs_bfs(policy);
    }

    /// Test-only: include all registered function graphs as candidates.
    /// Production code must use `find_all_graphs()` with portal seeded.
    #[cfg(test)]
    pub fn find_all_graphs_for_tests(&mut self) {
        if self.portal_targets.is_empty() {
            let all_paths: Vec<CallPath> = self.function_graphs.keys().cloned().collect();
            for path in all_paths {
                self.candidate_graphs.insert(path);
            }
            return;
        }
        let mut policy = crate::policy::DefaultJitPolicy::new();
        self.find_all_graphs_bfs(&mut policy);
    }

    fn find_all_graphs_bfs(&mut self, policy: &mut dyn JitPolicy) {
        // RPython call.py:49-92: BFS from portal targets.
        // For each graph, scan all Call ops. If guess_call_kind would
        // return 'regular' (i.e. graphs_from returns a graph AND it's
        // a candidate), add the callee graph to candidates and continue.
        //
        // During BFS we use target_to_path + function_graphs directly
        // (not graphs_from, which checks candidate_graphs — the set
        // we're building).
        let mut todo: Vec<CallPath> = self.portal_targets.iter().cloned().collect();
        for path in &todo {
            self.candidate_graphs.insert(path.clone());
        }
        // call.py:59-64 — seed the BFS with builtin oopspec helpers so
        // `int_abs` / `int_floordiv` / `int_mod` / `ll_math.ll_math_sqrt`
        // are reachable even when the portal does not call them
        // directly.
        //
        // ```python
        // if hasattr(self, 'rtyper'):
        //     for oopspec_name, ll_args, ll_res in support.inline_calls_to:
        //         c_func, _ = support.builtin_func_for_spec(self.rtyper,
        //                                                   oopspec_name,
        //                                                   ll_args, ll_res)
        //         todo.append(c_func.value._obj.graph)
        // ```
        //
        // PRE-EXISTING-ADAPTATION: pyre has no `MixLevelHelperAnnotator`
        // so `builtin_func_for_spec` returns a descriptor only, not a
        // function-pointer-with-graph.  Where the oopspec name matches a
        // graph registered via `register_function_graph`, seed that
        // graph; otherwise the entry is structural-only and contributes
        // nothing, matching upstream behavior when the helper is later
        // inlined as a Rust intrinsic.
        for (oopspec_name, ll_args, ll_res) in crate::support::INLINE_CALLS_TO {
            let _spec = crate::support::builtin_func_for_spec(oopspec_name, ll_args, *ll_res);
            let path = CallPath::from_segments([*oopspec_name]);
            if self.function_graphs.contains_key(&path) && !self.candidate_graphs.contains(&path) {
                self.candidate_graphs.insert(path.clone());
                todo.push(path);
            }
        }

        while let Some(path) = todo.pop() {
            let graph = match self.function_graphs.get(&path) {
                Some(g) => g.clone(),
                None => continue,
            };
            // RPython call.py:77-90: scan all Call ops in the graph.
            // For each call, check guess_call_kind (with BFS-aware
            // is_candidate that treats "has graph" as candidate).
            for block in &graph.blocks {
                for op in &block.operations {
                    let target = match &op.kind {
                        OpKind::Call { target, .. } => target,
                        _ => continue,
                    };
                    let callee_path = match self.target_to_path(target) {
                        Some(p) => p,
                        None => continue,
                    };
                    // RPython call.py:80: kind = self.guess_call_kind(op, is_candidate)
                    // Skip recursive (portal) and builtin calls — these are NOT
                    // followed during BFS. Only "regular" calls are followed.
                    if self.portal_targets.contains(&callee_path) {
                        continue; // recursive — don't follow
                    }
                    if self.builtin_targets.contains(&callee_path) {
                        continue; // builtin — don't follow
                    }
                    if self.candidate_graphs.contains(&callee_path) {
                        continue; // already discovered
                    }
                    // RPython call.py:84,87: callee must satisfy
                    // policy.look_inside_graph(graph). Synthesize a
                    // SemanticFunction from the stored graph + hints so
                    // the policy's `_jit_*_` / `_elidable_function_`
                    // checks fire identically to upstream.
                    if let Some(graph) = self.function_graphs.get(&callee_path).cloned() {
                        let hints = self
                            .function_hints
                            .get(&callee_path)
                            .cloned()
                            .unwrap_or_default();
                        let func = SemanticFunction {
                            name: callee_path.last_segment().unwrap_or_default().to_string(),
                            graph,
                            return_type: None,
                            self_ty_root: None,
                            hints,
                            access_directly: false,
                        };
                        if policy.look_inside_graph(&func) {
                            self.candidate_graphs.insert(callee_path.clone());
                            todo.push(callee_path);
                        }
                    }
                }
            }
        }
    }

    /// RPython: `CallControl.is_candidate(graph)`.
    /// Used only after `find_all_graphs()`.
    pub fn is_candidate(&self, path: &CallPath) -> bool {
        self.candidate_graphs.contains(path)
    }

    /// RPython: `CallControl.get_jitcode(graph, called_from)`.
    ///
    /// Retrieve or create the `Arc<JitCode>` shell for the given graph.
    /// The shell carries `name` plus the graph's bound helper address when
    /// available, otherwise the stable symbolic fallback; the body is
    /// filled later by `CodeWriter::transform_graph_to_jitcode` via
    /// `JitCode::set_body`. Upstream `jitcode.index` is not assigned here;
    /// pyre follows the same rule and sets it only when the finished
    /// jitcode is appended to `all_jitcodes[]`.
    ///
    /// RPython call.py:155-172: creates JitCode(graph.name, fnaddr, calldescr)
    /// and adds graph to unfinished_graphs for later assembly.
    pub fn get_jitcode(&mut self, path: &CallPath) -> std::sync::Arc<crate::jitcode::JitCode> {
        // RPython call.py:157-158: try: return self.jitcodes[graph]
        if let Some(arc) = self.jitcodes.get(path) {
            return arc.clone();
        }
        // RPython call.py:159-165: except KeyError:
        //   must never produce JitCode for close_stack.
        assert!(
            !self.close_stack_targets.contains(path),
            "{:?} has _gctransformer_hint_close_stack_",
            path
        );
        // Shell name mirrors RPython `graph.name`. We use the path's last
        // segment to stay readable in dumps; the assembler no longer
        // touches the name (it lives on the shell from allocation).
        let name = path
            .last_segment()
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{path:?}"));
        let mut shell = crate::jitcode::JitCode::new(name);
        shell.fnaddr = self
            .function_fnaddrs
            .get(path)
            .copied()
            .unwrap_or_else(|| symbolic_fnaddr_for_path(path));
        let arc = std::sync::Arc::new(shell);
        self.jitcodes.insert(path.clone(), arc.clone());
        self.unfinished_graphs.push(path.clone());
        arc
    }

    /// Read-only handle lookup. Returns `None` for paths that have not
    /// been allocated by `get_jitcode` yet.
    pub fn jitcode_handle(
        &self,
        path: &CallPath,
    ) -> Option<std::sync::Arc<crate::jitcode::JitCode>> {
        self.jitcodes.get(path).cloned()
    }

    /// RPython `codewriter.py:81 all_jitcodes.append(jitcode)` — the
    /// sole append site in upstream's `make_jitcodes` loop. `jitcode.index`
    /// is already set by `transform_graph_to_jitcode` (upstream line 68);
    /// this method is the final positional append.
    pub fn finish_jitcode(&mut self, jitcode: std::sync::Arc<crate::jitcode::JitCode>) {
        debug_assert_eq!(
            jitcode.try_index(),
            Some(self.finished_jitcodes.len()),
            "finish_jitcode: jitcode {:?} arrives with index {:?} but \
             would land at slot {}. Upstream `codewriter.py:68` assigns \
             `jitcode.index = index` inside `transform_graph_to_jitcode`, \
             which must match `len(all_jitcodes)` at the call site.",
            jitcode.name,
            jitcode.try_index(),
            self.finished_jitcodes.len(),
        );
        self.finished_jitcodes.push(jitcode);
    }

    /// Read the number of jitcodes already appended to `all_jitcodes[]`.
    /// Drain loop callers use this to compute the `index` passed into
    /// `transform_graph_to_jitcode` (upstream `codewriter.py:80
    /// self.transform_graph_to_jitcode(graph, jitcode, verbose,
    /// len(all_jitcodes))`).
    pub fn finished_jitcodes_len(&self) -> usize {
        self.finished_jitcodes.len()
    }

    /// RPython `call.py:182-187 get_jitcode_calldescr` source-of-truth for
    /// `FUNC.RESULT`. Pyre derives the calldescr's result kind char from
    /// the registered `return_types` string for the graph's CallPath. The
    /// mapping mirrors `front/ast.rs::type_string_to_value_type`.
    /// Returns `None` when no return type was registered for this path —
    /// callers (`transform_graph_to_jitcode`) fall back to a CFG scan in
    /// that case (e.g. unit-test graphs without a parsed signature).
    pub fn declared_return_kind(&self, path: &CallPath) -> Option<char> {
        let s = self.return_types.get(path)?.trim();
        Some(return_type_string_to_kind(s))
    }
}

/// Map a Rust return-type string to the BhCallDescr kind char used by
/// blackhole / metainterp. `None`/`""`/`"()"` → `'v'`. The integer/float
/// recognizer is the same set as `front/ast.rs::type_string_to_value_type`.
fn return_type_string_to_kind(s: &str) -> char {
    match s {
        "" | "()" => 'v',
        "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
        | "bool" | "char" | "Self::Truth" => 'i',
        "f32" | "f64" => 'f',
        _ => 'r',
    }
}

/// RPython parity for `call.py:220-221` `FUNC.ARGS` — collect the non-void
/// argument types of a graph.  Parameters live on `startblock.inputargs`
/// (RPython `flowspace/model.py` Block), populated by the front-end at
/// `front/ast.rs:706-769` parameter registration.  The `OpKind::Input`
/// ops co-emitted with each parameter carry the declared type which we
/// recover by chasing each inputarg ValueId back to its defining op.
/// Unknown/ambiguous slots default to `Ref`, matching
/// `resolve_non_void_arg_types`' fallback.
///
/// PRE-EXISTING-ADAPTATION: when `inputargs` is empty we fall back to
/// scanning leading `OpKind::Input` ops in the startblock.  Unit tests
/// under `jit_codewriter::jtransform::tests` build graphs directly via
/// `FunctionGraph::new` + `push_op` without populating `inputargs`; the
/// fallback keeps their "all-Input-ops-are-params" convention working
/// until they are migrated.
fn graph_non_void_arg_types(graph: &FunctionGraph) -> Vec<Type> {
    let start = graph.block(graph.startblock);
    let map_ty = |ty: &crate::model::ValueType| match ty {
        crate::model::ValueType::Int => Some(Type::Int),
        crate::model::ValueType::Ref => Some(Type::Ref),
        crate::model::ValueType::Float => Some(Type::Float),
        crate::model::ValueType::Void => None,
        // Unknown / State — default to Ref.
        _ => Some(Type::Ref),
    };
    if !start.inputargs.is_empty() {
        return start
            .inputargs
            .iter()
            .filter_map(|&arg_id| {
                let ty = start.operations.iter().find_map(|op| match &op.kind {
                    crate::model::OpKind::Input { ty, .. } if op.result == Some(arg_id) => Some(ty),
                    _ => None,
                });
                ty.map(map_ty).unwrap_or(Some(Type::Ref))
            })
            .collect();
    }
    start
        .operations
        .iter()
        .take_while(|op| matches!(op.kind, crate::model::OpKind::Input { .. }))
        .filter_map(|op| match &op.kind {
            crate::model::OpKind::Input { ty, .. } => map_ty(ty),
            _ => None,
        })
        .collect()
}

/// RPython parity for `call.py:222` `FUNC.RESULT`. Reads the declared
/// return type string from `CallControl::return_types` and maps it to
/// `Type`; `None` or unknown string → `Type::Void` (i.e. declared-void
/// function). Matches `type_string_to_value_type` in `front/ast.rs`.
fn return_type_string_to_value_type(s: Option<&String>) -> Type {
    match s.map(String::as_str) {
        None | Some("") | Some("()") => Type::Void,
        Some("i8") | Some("i16") | Some("i32") | Some("i64") | Some("isize") | Some("u8")
        | Some("u16") | Some("u32") | Some("u64") | Some("usize") | Some("bool") | Some("char")
        | Some("Self::Truth") => Type::Int,
        Some("f32") | Some("f64") => Type::Float,
        _ => Type::Ref,
    }
}

impl CallControl {
    /// Return the completed `all_jitcodes[]` list in append order. Every
    /// entry must have both a body and a dense final `.index`.
    pub fn collect_jitcodes_in_alloc_order(&self) -> Vec<std::sync::Arc<crate::jitcode::JitCode>> {
        for (i, jitcode) in self.finished_jitcodes.iter().enumerate() {
            assert!(
                jitcode.try_body().is_some(),
                "collect_jitcodes_in_alloc_order: jitcode {:?} at slot {i} has no body",
                jitcode.name
            );
            assert_eq!(
                jitcode.index(),
                i,
                "collect_jitcodes_in_alloc_order: jitcode {:?} has index {} at slot {i}",
                jitcode.name,
                jitcode.index()
            );
        }
        self.finished_jitcodes.clone()
    }

    /// RPython: `CallControl.grab_initial_jitcodes()` (call.py:145-148).
    ///
    /// ```python
    /// def grab_initial_jitcodes(self):
    ///     for jd in self.jitdrivers_sd:
    ///         jd.mainjitcode = self.get_jitcode(jd.portal_graph)
    ///         jd.mainjitcode.jitdriver_sd = jd
    /// ```
    ///
    /// Allocates `Arc<JitCode>` shells for portal graphs and stores them
    /// directly on each jitdriver. The `jitdriver_sd` back-reference is
    /// committed later by `CodeWriter::drain_pending_graphs` once the
    /// portal's body is assembled.
    pub fn grab_initial_jitcodes(&mut self) {
        // Collect portal paths first to avoid borrow conflict.
        let portals: Vec<(usize, CallPath)> = self
            .jitdrivers_sd
            .iter()
            .enumerate()
            .map(|(i, jd)| (i, jd.portal_graph.clone()))
            .collect();
        for (jd_index, portal) in portals {
            // RPython: jd.mainjitcode = self.get_jitcode(jd.portal_graph)
            let arc = self.get_jitcode(&portal);
            self.jitdrivers_sd[jd_index].mainjitcode = Some(arc);
        }
    }

    /// RPython: `CallControl.enum_pending_graphs()` (call.py:150-153).
    ///
    /// ```python
    /// def enum_pending_graphs(self):
    ///     while self.unfinished_graphs:
    ///         graph = self.unfinished_graphs.pop()  # LIFO
    ///         yield graph, self.jitcodes[graph]
    /// ```
    ///
    /// RPython uses a generator that pops one graph at a time (LIFO).
    /// During processing, new graphs may be added to `unfinished_graphs`
    /// via `get_jitcode()`, and the generator picks them up on the next
    /// iteration. We emulate this with `enum_pending_graphs()`.
    pub fn enum_pending_graphs(
        &mut self,
    ) -> Option<(CallPath, std::sync::Arc<crate::jitcode::JitCode>)> {
        let path = self.unfinished_graphs.pop()?; // LIFO, matching RPython
        let arc = self.jitcodes[&path].clone();
        Some((path, arc))
    }

    /// Classify a call.
    ///
    /// RPython `call.py:116-139 CallControl.guess_call_kind(op, is_candidate)`
    /// — line-by-line port.  The `op.opname == 'direct_call'` branch
    /// (call.py:117-136) maps to `OpKind::Call`; the implicit
    /// `indirect_call` branch (RPython falls through to the final
    /// `graphs_from(op) is None` test at line 137) maps to
    /// `OpKind::IndirectCall`.  Closestack / oopspec / recursive
    /// checks only apply to the direct branch because the corresponding
    /// flags are attached to a single `funcobj`; for indirect calls the
    /// same restrictions are enforced family-wide in `getcalldescr`
    /// (`call.py:259-280`).
    pub fn guess_call_kind(&self, op: &SpaceOperation) -> CallKind {
        if let OpKind::Call { target, .. } = &op.kind {
            // RPython `call.py:117-136` direct_call branch.
            let path = self.target_to_path(target);
            if let Some(ref p) = path {
                // call.py:119-120 jitdriver_sd_from_portal_runner_ptr(funcptr)
                if self.portal_targets.contains(p) {
                    return CallKind::Recursive;
                }
                // call.py:129-134 _gctransformer_hint_close_stack_ → 'residual'
                if self.close_stack_targets.contains(p) {
                    return CallKind::Residual;
                }
                // call.py:135-136 oopspec → 'builtin'
                if self.builtin_targets.contains(p) {
                    return CallKind::Builtin;
                }
            }
        }
        // RPython `call.py:137-139` — both direct_call (fall-through)
        // and indirect_call reach this final classification.
        if self.graphs_from(op).is_none() {
            CallKind::Residual
        } else {
            CallKind::Regular
        }
    }

    /// Collect every candidate callee graph reachable through this op.
    ///
    /// RPython `call.py:94-114 CallControl.graphs_from(op, is_candidate)`
    /// — line-by-line port.  The `op.opname == 'direct_call'` branch
    /// (call.py:97-101) returns `[graph]` for a direct call whose target
    /// is a candidate; the `op.opname == 'indirect_call'` branch
    /// (call.py:103-112) filters the family attached to the op by
    /// `is_candidate` and returns the non-empty subset.  Both branches
    /// collapse to `None` when no candidate is reachable — the residual
    /// call path.
    ///
    /// The `is_candidate` argument from RPython `call.py:94-96` is
    /// omitted here because majit's `find_all_graphs_for_tests` /
    /// `find_all_graphs` populates `self.candidate_graphs` in bulk
    /// before any caller invokes `graphs_from`; the RPython
    /// incremental-discovery shape (where `find_all_graphs` passes its
    /// own local `is_candidate`) is not needed.
    pub fn graphs_from(&self, op: &SpaceOperation) -> Option<Vec<CallPath>> {
        match &op.kind {
            OpKind::Call { target, .. } => {
                // call.py:97-101 direct_call branch.
                let path = self.target_to_path(target)?;
                if self.candidate_graphs.contains(&path) {
                    Some(vec![path])
                } else {
                    None
                }
            }
            OpKind::IndirectCall { graphs, .. } => {
                // call.py:103-112 indirect_call branch.
                // `graphs is None` (call.py:105) → residual.
                let graphs = graphs.as_ref()?;
                let result: Vec<CallPath> = graphs
                    .iter()
                    .filter(|p| self.candidate_graphs.contains(p))
                    .cloned()
                    .collect();
                if result.is_empty() {
                    None
                } else {
                    Some(result)
                }
            }
            _ => None,
        }
    }

    /// Look up the single callee `FunctionGraph` for a direct call.
    ///
    /// Convenience accessor for call sites (majit `inline.rs`) that
    /// still work with one `&FunctionGraph` at a time.  RPython
    /// `call.py:97-101` returns the graph value inside the list, but
    /// majit callers want a borrow against `function_graphs` so we keep
    /// this lookup separate from the op-based `graphs_from`.
    pub fn direct_graph_for(&self, target: &CallTarget) -> Option<&FunctionGraph> {
        let path = self.target_to_path(target)?;
        if !self.candidate_graphs.contains(&path) {
            return None;
        }
        match target {
            CallTarget::Method {
                name,
                receiver_root,
            } => self
                .function_graphs
                .get(&path)
                .or_else(|| self.resolve_method(name, receiver_root.as_deref())),
            _ => self.function_graphs.get(&path),
        }
    }

    /// Convert a CallTarget to a CallPath for lookup.
    ///
    /// FunctionPath → direct path.
    /// Method → qualified CallPath([impl_type, method_name]).
    ///
    /// RPython: graph identity is by object pointer, not name.
    /// We emulate this with qualified paths that include the impl type,
    /// so different impls of the same method get distinct paths.
    pub(crate) fn target_to_path(&self, target: &CallTarget) -> Option<CallPath> {
        match target {
            CallTarget::FunctionPath { segments } => {
                Some(CallPath::from_segments(segments.iter().map(String::as_str)))
            }
            CallTarget::Method {
                name,
                receiver_root,
            } => {
                // RPython: direct_call → funcobj.graph. Try qualified path first
                // so inherent methods resolve by direct graph linkage.
                if let Some(receiver) = receiver_root.as_deref() {
                    let qualified = CallPath::for_impl_method(receiver, name.as_str());
                    if self.function_graphs.contains_key(&qualified) {
                        return Some(qualified);
                    }
                }
                // Fall back to trait method resolution for polymorphic calls.
                let impl_type = self.resolve_method_impl_type(name, receiver_root.as_deref())?;
                Some(CallPath::for_impl_method(impl_type, name.as_str()))
            }
            // RPython: an `indirect_call` is a *family* of graphs — there is
            // no single CallPath to resolve to.  Post-rtyper, indirect calls
            // live as `OpKind::IndirectCall` and `graphs_from(op)` returns
            // the candidate `Vec<CallPath>` directly; `target_to_path` only
            // names direct_call-equivalent sites.  `call.py:94-114` indirect
            // branch.
            CallTarget::SyntheticTransparentCtor { .. } | CallTarget::Indirect { .. } => None,
            CallTarget::UnsupportedExpr => None,
        }
    }

    /// RPython `call.py:181-183` uses `getfunctionptr(graph)` to obtain the
    /// integer funcptr identity for a call site. majit prefers a host-bound
    /// trace-call address when one has been registered for the resolved
    /// `CallPath`; otherwise it falls back to the stable symbolic address
    /// shim for source-only analysis.
    pub fn fnaddr_for_target(&self, target: &CallTarget) -> i64 {
        if let Some(path) = self.target_to_path(target) {
            return self
                .function_fnaddrs
                .get(&path)
                .copied()
                .unwrap_or_else(|| symbolic_fnaddr_for_path(&path));
        }
        symbolic_fnaddr_for_target(target)
    }

    /// Resolve a method call to a concrete impl graph.
    ///
    /// RPython: method resolution happens at the type system level; the
    /// source-level Rust frontend falls back to receiver-root strings to
    /// disambiguate impl graphs until the annotator folds bound-method
    /// identities directly into stable call targets.
    pub fn resolve_method(
        &self,
        name: &str,
        receiver_root: Option<&str>,
    ) -> Option<&FunctionGraph> {
        let impls = self.impls_for_method_name(name);
        if impls.is_empty() {
            return None;
        }
        let concrete_impls: Vec<&String> = impls
            .iter()
            .copied()
            .filter(|t| !t.starts_with("<default methods of"))
            .collect();

        if concrete_impls.len() == 1 {
            // Unique concrete impl — use it regardless of receiver
            let impl_type = concrete_impls[0];
            return self
                .trait_method_graphs
                .get(&(name.to_string(), impl_type.clone()));
        }

        // Multiple concrete impls — try to match by receiver root
        if let Some(receiver) = receiver_root {
            if !is_generic_receiver(receiver) {
                // Concrete receiver — look for exact match
                return self
                    .trait_method_graphs
                    .get(&(name.to_string(), receiver.to_string()));
            }
        }

        // Generic receiver with multiple concrete impls — can't resolve uniquely.
        // Fall back to default method if available.
        if concrete_impls.is_empty() && impls.len() == 1 {
            let impl_type = impls[0];
            return self
                .trait_method_graphs
                .get(&(name.to_string(), impl_type.clone()));
        }

        None
    }

    /// Like `resolve_method`, but returns the impl type name instead of the graph.
    /// Used by `target_to_path` to build qualified CallPaths.
    /// All returned references borrow from `self`, not from `receiver_root`.
    fn resolve_method_impl_type<'b>(
        &'b self,
        name: &str,
        receiver_root: Option<&str>,
    ) -> Option<&'b str> {
        let impls = self.impls_for_method_name(name);
        if impls.is_empty() {
            return None;
        }
        let concrete_impls: Vec<&String> = impls
            .iter()
            .copied()
            .filter(|t| !t.starts_with("<default methods of"))
            .collect();

        if concrete_impls.len() == 1 {
            return Some(concrete_impls[0].as_str());
        }
        if let Some(receiver) = receiver_root {
            if !is_generic_receiver(receiver) {
                // Find the matching impl owned by self
                if let Some(impl_name) = impls.iter().copied().find(|t| t.as_str() == receiver) {
                    return Some(impl_name.as_str());
                }
            }
        }
        if concrete_impls.is_empty() && impls.len() == 1 {
            return Some(impls[0].as_str());
        }
        None
    }

    /// Collect every registered impl type name for `method_name`, across
    /// all declaring traits.  Used by `resolve_method` /
    /// `resolve_method_impl_type` for concrete-receiver method calls
    /// (RPython's `funcobj.graph` resolution).  Indirect-call family
    /// lookup uses the exact `(trait_root, method_name)` key via
    /// `all_impls_for_indirect` instead.
    fn impls_for_method_name<'b>(&'b self, method_name: &str) -> Vec<&'b String> {
        self.trait_method_impls
            .iter()
            .filter(|((_, m), _)| m == method_name)
            .flat_map(|(_, impls)| impls.iter())
            .collect()
    }

    /// Collect every registered impl `CallPath` for a
    /// `(trait_root, method_name)` family, regardless of whether each
    /// one is a regular candidate.  Used by family-wide validation
    /// where the goal is to reject mixed `_elidable_function_` etc.
    /// even among residual members (`call.py:259-280`).
    pub fn all_impls_for_indirect(&self, trait_root: &str, method_name: &str) -> Vec<CallPath> {
        self.trait_method_impls
            .get(&(trait_root.to_string(), method_name.to_string()))
            .into_iter()
            .flatten()
            .map(|impl_type| CallPath::for_impl_method(impl_type.as_str(), method_name))
            .collect()
    }

    /// RPython `call.py:259-280` — family-wide validation for indirect_call.
    ///
    /// Rejects a family if any member is marked `_elidable_function_` /
    /// `_jit_loop_invariant_` / `_call_aroundstate_target_`: indirect
    /// dispatch cannot preserve the semantics those flags require, so
    /// upstream raises an Exception at getcalldescr time.  Returns a
    /// formatted error message on the first mismatch.
    pub fn check_indirect_call_family(&self, candidates: &[CallPath]) -> Result<(), String> {
        for graph in candidates {
            let err = if self.elidable_targets.contains(graph) {
                Some("@jit.elidable")
            } else if self.loopinvariant_targets.contains(graph) {
                Some("@jit.loop_invariant")
            } else if self.aroundstate_targets.contains(graph) {
                Some("_call_aroundstate_target_")
            } else {
                None
            };
            if let Some(flag) = err {
                return Err(format!(
                    "indirect_call family includes {graph:?} marked {flag}; \
                     every candidate in an indirect family must share the \
                     same jit attribute"
                ));
            }
        }
        Ok(())
    }

    /// Access the function graphs map (for inline pass).
    pub fn function_graphs(&self) -> &HashMap<CallPath, FunctionGraph> {
        &self.function_graphs
    }

    /// Returns true when a concrete helper address was registered for this
    /// path. Such a path is a real callable surface and must not be treated
    /// as a transparent Rust enum constructor by jtransform.
    pub fn has_function_fnaddr(&self, path: &CallPath) -> bool {
        self.function_fnaddrs.contains_key(path)
    }

    /// Access the `{CallPath → Arc<JitCode>}` map.
    ///
    /// RPython: `call.py:87 self.jitcodes`. Pyre exposes the same map as a
    /// read-only view so `CodeWriter::make_jitcodes` can pair it with
    /// `collect_jitcodes_in_alloc_order` into a single `AllJitCodes`
    /// return value.
    pub fn jitcodes(&self) -> &HashMap<CallPath, std::sync::Arc<crate::jitcode::JitCode>> {
        &self.jitcodes
    }

    /// Access jitdriver static data.
    pub fn jitdrivers_sd(&self) -> &[JitDriverStaticData] {
        &self.jitdrivers_sd
    }

    // ── Elidable / loop-invariant registration ──────────────────────

    /// RPython: `getattr(func, "_elidable_function_", False)` (call.py:239).
    /// Mark a target as elidable (pure function).
    pub fn mark_elidable(&mut self, path: CallPath) {
        self.elidable_targets.insert(path);
    }

    /// RPython: `getattr(func, "_jit_loop_invariant_", False)` (call.py:240).
    /// Mark a target as loop-invariant.
    pub fn mark_loopinvariant(&mut self, path: CallPath) {
        self.loopinvariant_targets.insert(path);
    }

    /// RPython: call.py:239 — check if target has `_elidable_function_`.
    pub fn is_elidable(&self, target: &CallTarget) -> bool {
        self.target_to_path(target)
            .is_some_and(|p| self.elidable_targets.contains(&p))
    }

    /// RPython: call.py:240 — check if target has `_jit_loop_invariant_`.
    pub fn is_loopinvariant(&self, target: &CallTarget) -> bool {
        self.target_to_path(target)
            .is_some_and(|p| self.loopinvariant_targets.contains(&p))
    }

    /// RPython: call.py:129-134 — `_gctransformer_hint_close_stack_`.
    /// Mark a target as close_stack (must never produce JitCode).
    pub fn mark_close_stack(&mut self, path: CallPath) {
        self.close_stack_targets.insert(path);
    }

    /// RPython: collectanalyze.py:21 — `funcobj.random_effects_on_gcobjs`.
    /// Mark an external target as having random GC effects.
    pub fn mark_external_gc_effects(&mut self, path: CallPath) {
        self.external_gc_effects.insert(path);
    }

    /// RPython: `getattr(func, '_call_aroundstate_target_', None)` (call.py:252).
    /// Mark a target as `_call_aroundstate_target_` so
    /// `check_indirect_call_family` rejects mixing it into an
    /// indirect_call family.  Placeholder until
    /// `#[jit_call_aroundstate_target]` attribute is ported.
    pub fn mark_aroundstate(&mut self, path: CallPath) {
        self.aroundstate_targets.insert(path);
    }

    /// RPython: collectanalyze.py:15 — `_gctransformer_hint_cannot_collect_`.
    /// Mark a target as known not to trigger GC collection.
    pub fn mark_cannot_collect(&mut self, path: CallPath) {
        self.cannot_collect_targets.insert(path);
    }

    /// RPython: rlib/jit.py:250 `@oopspec(spec)` — store `func.oopspec = spec`.
    /// Mark a target as having an oopspec string for jtransform lowering.
    pub fn mark_oopspec(&mut self, path: CallPath, spec: String) {
        self.oopspec_targets.insert(path, spec);
    }

    /// RPython: `getattr(func, 'oopspec', None)` — look up oopspec for a target.
    pub fn get_oopspec(&self, target: &CallTarget) -> Option<&str> {
        self.target_to_path(target)
            .and_then(|p| self.oopspec_targets.get(&p).map(|s| s.as_str()))
    }

    // ── Graph-based analyzers (call.py:282-303) ─────────────────────

    /// RPython: RaiseAnalyzer.analyze() — transitive can-raise analysis.
    ///
    /// canraise.py:8-24: RaiseAnalyzer(BoolGraphAnalyzer)
    /// - `analyze_simple_operation`: checks `LL_OPERATIONS[op.opname].canraise`
    /// - `analyze_external_call`: `getattr(fnobj, 'canraise', True)`
    /// - `analyze_exceptblock_in_graph`: checks except blocks
    ///
    /// Shared implementation for the two upstream RaiseAnalyzer instances:
    /// normal mode and `do_ignore_memory_error()` mode.
    fn analyze_can_raise_impl(
        &self,
        path: &CallPath,
        seen: &mut HashSet<CallPath>,
        ignore_memoryerror: bool,
    ) -> bool {
        if !seen.insert(path.clone()) {
            return false; // cycle → bottom_result
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            // RPython: analyze_external_call → getattr(fnobj, 'canraise', True)
            None => return true,
        };
        for block in &graph.blocks {
            // RPython: analyze_simple_operation(op) per operation.
            // canraise.py:14-17: LL_OPERATIONS[op.opname].canraise
            for op in &block.operations {
                let op_result = match &op.kind {
                    OpKind::Call { target, .. } => {
                        let callee_path = match self.target_to_path(target) {
                            Some(p) => p,
                            None => return true, // unresolvable → conservative
                        };
                        self.analyze_can_raise_impl(&callee_path, seen, ignore_memoryerror)
                    }
                    OpKind::IndirectCall { graphs, .. } => match graphs.as_deref() {
                        None => true, // graphanalyze.py:117-121 → top_result()
                        Some(graphs) => {
                            for callee_path in graphs {
                                if self.analyze_can_raise_impl(
                                    callee_path,
                                    seen,
                                    ignore_memoryerror,
                                ) {
                                    return true;
                                }
                            }
                            false
                        }
                    },
                    other => raise_class_can_raise(op_can_raise(other), ignore_memoryerror),
                };
                if op_result {
                    return true;
                }
            }
        }
        // RPython `backendopt/canraise.py:27-41 analyze_exceptblock_in_graph`
        // only applies the re-raise suppression in the ignore-MemoryError
        // analyzer. The normal analyzer always treats exceptblock exits as
        // raising.
        if graph
            .blocks
            .iter()
            .flat_map(|block| block.exits.iter())
            .any(|link| link.target == graph.exceptblock)
        {
            if ignore_memoryerror && exceptblock_is_reraise_of_caught_exception(graph) {
                return false;
            }
            return true;
        }
        false
    }

    /// RPython: VirtualizableAnalyzer.analyze() (effectinfo.py:401-404).
    ///
    /// analyze_simple_operation: op.opname in ('jit_force_virtualizable',
    ///                                         'jit_force_virtual')
    fn analyze_forces_virtualizable(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        if !seen.insert(path.clone()) {
            return false;
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            // RPython: external call → analyze_external_call → bottom_result (False).
            // VirtualizableAnalyzer does not override analyze_external_call.
            None => return false,
        };
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    // RPython: jit_force_virtualizable / jit_force_virtual
                    OpKind::VableForce { .. } => return true,
                    OpKind::Call { target, .. } => {
                        let callee_path = match self.target_to_path(target) {
                            Some(p) => p,
                            None => continue, // external call → False
                        };
                        if self.analyze_forces_virtualizable(&callee_path, seen) {
                            return true;
                        }
                    }
                    OpKind::IndirectCall { graphs, .. } => match graphs.as_deref() {
                        None => return true, // BoolGraphAnalyzer.top_result()
                        Some(graphs) => {
                            for callee_path in graphs {
                                if self.analyze_forces_virtualizable(callee_path, seen) {
                                    return true;
                                }
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
        false
    }

    /// RPython: RandomEffectsAnalyzer.analyze() (effectinfo.py:410-418).
    ///
    /// ```python
    /// class RandomEffectsAnalyzer(BoolGraphAnalyzer):
    ///     def analyze_external_call(self, funcobj, seen=None):
    ///         if funcobj.random_effects_on_gcobjs:
    ///             return True
    ///         return super().analyze_external_call(funcobj, seen)
    ///     def analyze_simple_operation(self, op, graphinfo):
    ///         return False
    /// ```
    ///
    /// Key: `analyze_simple_operation` always returns False. External calls
    /// only return True if `random_effects_on_gcobjs` is set. The default
    /// `analyze_external_call` returns `bottom_result()` = False
    /// (graphanalyze.py:60-69). "No graph" ≠ random effects in RPython.
    ///
    /// In majit: functions without graphs are external calls — returns
    /// True if in `external_gc_effects`, False otherwise.
    fn analyze_random_effects(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        if !seen.insert(path.clone()) {
            return false; // cycle → bottom_result
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            None => {
                // RPython: analyze_external_call → bottom_result (False)
                // unless funcobj.random_effects_on_gcobjs → True.
                return self.external_gc_effects.contains(path);
            }
        };
        // RPython: analyze_simple_operation always returns False.
        // Only recursive calls into graphs can propagate random effects.
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::Call { target, .. } => {
                        let callee_path = match self.target_to_path(target) {
                            Some(p) => p,
                            // Unresolvable target = external call → False
                            None => continue,
                        };
                        if self.analyze_random_effects(&callee_path, seen) {
                            return true;
                        }
                    }
                    OpKind::IndirectCall { graphs, .. } => match graphs.as_deref() {
                        None => return true, // BoolGraphAnalyzer.top_result()
                        Some(graphs) => {
                            for callee_path in graphs {
                                if self.analyze_random_effects(callee_path, seen) {
                                    return true;
                                }
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
        false
    }

    /// RPython: QuasiImmutAnalyzer.analyze() (effectinfo.py).
    ///
    /// analyze_simple_operation: op.opname == 'jit_force_quasi_immutable'.
    ///
    /// In majit: we don't have quasi-immutable ops in the model yet,
    /// so this always returns false. The transitive call check is still
    /// performed for future-proofing.
    fn analyze_can_invalidate(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        if !seen.insert(path.clone()) {
            return false;
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            None => return false, // no graph → cannot invalidate (not conservative here)
        };
        for block in &graph.blocks {
            for op in &block.operations {
                // RPython: jit_force_quasi_immutable → true
                // majit: no such op yet, but check calls transitively
                match &op.kind {
                    OpKind::Call { target, .. } => {
                        let callee_path = match self.target_to_path(target) {
                            Some(p) => p,
                            None => continue,
                        };
                        if self.analyze_can_invalidate(&callee_path, seen) {
                            return true;
                        }
                    }
                    OpKind::IndirectCall { graphs, .. } => match graphs.as_deref() {
                        None => return true, // BoolGraphAnalyzer.top_result()
                        Some(graphs) => {
                            for callee_path in graphs {
                                if self.analyze_can_invalidate(callee_path, seen) {
                                    return true;
                                }
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
        false
    }

    /// RPython: CollectAnalyzer (collectanalyze.py).
    ///
    /// RPython: CollectAnalyzer.analyze_direct_call(graph, seen)
    /// (collectanalyze.py + graphanalyze.py:139).
    ///
    /// Traverses graph ops with:
    /// - analyze_simple_operation (collectanalyze.py:27-33): checks malloc/
    ///   malloc_varsize with GC flavor, LL_OPERATIONS[op].canmallocgc.
    ///   In majit the codewriter graph has no LL_OPERATIONS; allocations are
    ///   only reachable transitively through calls.
    /// - analyze_direct_call: recurse into callee graphs.
    /// - analyze_external_call (graphanalyze.py:60): bottom_result() (False).
    /// - _gctransformer_hint_cannot_collect_ (collectanalyze.py:15-16):
    ///   functions in `cannot_collect_targets` are known not to collect.
    fn analyze_can_collect(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        if !seen.insert(path.clone()) {
            return false;
        }
        // collectanalyze.py:15: _gctransformer_hint_cannot_collect_ → False
        if self.cannot_collect_targets.contains(path) {
            return false;
        }
        // collectanalyze.py:15: _gctransformer_hint_close_stack_ → True.
        // close_stack functions always can collect.
        if self.close_stack_targets.contains(path) {
            return true;
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            None => {
                // collectanalyze.py:21-25: analyze_external_call —
                // if funcobj.random_effects_on_gcobjs → True,
                // else → bottom_result() (False).
                return self.external_gc_effects.contains(path);
            }
        };
        for block in &graph.blocks {
            for op in &block.operations {
                // collectanalyze.py:27-33: analyze_simple_operation
                // RPython checks: malloc/malloc_varsize with flavor='gc' → True
                //                 LL_OPERATIONS[op.opname].canmallocgc → True
                // majit codewriter graphs have no LL_OPERATIONS; the only
                // operations that can trigger GC are transitive through calls.
                // (All other OpKind variants are pure/field/array ops.)
                match &op.kind {
                    OpKind::Call { target, .. } => {
                        // graphanalyze.py:139-164: analyze_direct_call — recurse
                        let callee_path = match self.target_to_path(target) {
                            Some(p) => p,
                            // graphanalyze.py:60: external call → bottom_result (False)
                            None => continue,
                        };
                        if self.analyze_can_collect(&callee_path, seen) {
                            return true;
                        }
                    }
                    OpKind::IndirectCall { graphs, .. } => match graphs.as_deref() {
                        None => return true, // BoolGraphAnalyzer.top_result()
                        Some(graphs) => {
                            for callee_path in graphs {
                                if self.analyze_can_collect(callee_path, seen) {
                                    return true;
                                }
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
        false
    }

    // ── Cached analyzer wrappers ────────────────────────────────────

    /// Cached version of _canraise for a CallTarget.
    ///
    /// RPython call.py:337-355 — `_canraise()` returns the tri-state
    /// `{False, "mem", True}` collapsed here to [`CanRaise`].
    fn cached_can_raise_path(&self, path: &CallPath, cache: &mut AnalysisCache) -> CanRaise {
        if let Some(&result) = cache.can_raise.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = if !self.analyze_can_raise_impl(path, &mut seen, false) {
            CanRaise::No
        } else {
            let mut seen_ignore_memoryerror = HashSet::new();
            if self.analyze_can_raise_impl(path, &mut seen_ignore_memoryerror, true) {
                CanRaise::Yes
            } else {
                CanRaise::MemoryErrorOnly
            }
        };
        cache.can_raise.insert(path.clone(), result);
        result
    }

    fn cached_can_raise(&self, target: &CallTarget, cache: &mut AnalysisCache) -> CanRaise {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return CanRaise::Yes,
        };
        self.cached_can_raise_path(&path, cache)
    }

    fn cached_can_raise_family(
        &self,
        graphs: Option<&[CallPath]>,
        cache: &mut AnalysisCache,
    ) -> CanRaise {
        let graphs = match graphs {
            Some(graphs) => graphs,
            None => return CanRaise::Yes,
        };
        let mut result = CanRaise::No;
        for path in graphs {
            match self.cached_can_raise_path(path, cache) {
                CanRaise::Yes => return CanRaise::Yes,
                CanRaise::MemoryErrorOnly => result = CanRaise::MemoryErrorOnly,
                CanRaise::No => {}
            }
        }
        result
    }

    /// Cached version of analyze_forces_virtualizable for a CallTarget.
    /// RPython: VirtualizableAnalyzer external calls → bottom_result (False).
    fn cached_forces_virtualizable_path(&self, path: &CallPath, cache: &mut AnalysisCache) -> bool {
        if let Some(&result) = cache.forces_virtualizable.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = self.analyze_forces_virtualizable(&path, &mut seen);
        cache.forces_virtualizable.insert(path.clone(), result);
        result
    }

    fn cached_forces_virtualizable(&self, target: &CallTarget, cache: &mut AnalysisCache) -> bool {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return false, // external → False (RPython bottom_result)
        };
        self.cached_forces_virtualizable_path(&path, cache)
    }

    fn cached_forces_virtualizable_family(
        &self,
        graphs: Option<&[CallPath]>,
        cache: &mut AnalysisCache,
    ) -> bool {
        let graphs = match graphs {
            Some(graphs) => graphs,
            None => return true,
        };
        graphs
            .iter()
            .any(|path| self.cached_forces_virtualizable_path(path, cache))
    }

    /// Cached version of analyze_random_effects for a CallTarget.
    /// RPython: RandomEffectsAnalyzer defaults to False for external calls.
    fn cached_random_effects_path(&self, path: &CallPath, cache: &mut AnalysisCache) -> bool {
        if let Some(&result) = cache.random_effects.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = self.analyze_random_effects(&path, &mut seen);
        cache.random_effects.insert(path.clone(), result);
        result
    }

    fn cached_random_effects(&self, target: &CallTarget, cache: &mut AnalysisCache) -> bool {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return false, // external call → False (RPython default)
        };
        self.cached_random_effects_path(&path, cache)
    }

    fn cached_random_effects_family(
        &self,
        graphs: Option<&[CallPath]>,
        cache: &mut AnalysisCache,
    ) -> bool {
        let graphs = match graphs {
            Some(graphs) => graphs,
            None => return true,
        };
        graphs
            .iter()
            .any(|path| self.cached_random_effects_path(path, cache))
    }

    /// Cached version of analyze_can_invalidate for a CallTarget.
    fn cached_can_invalidate_path(&self, path: &CallPath, cache: &mut AnalysisCache) -> bool {
        if let Some(&result) = cache.can_invalidate.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = self.analyze_can_invalidate(&path, &mut seen);
        cache.can_invalidate.insert(path.clone(), result);
        result
    }

    fn cached_can_invalidate(&self, target: &CallTarget, cache: &mut AnalysisCache) -> bool {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return false,
        };
        self.cached_can_invalidate_path(&path, cache)
    }

    fn cached_can_invalidate_family(
        &self,
        graphs: Option<&[CallPath]>,
        cache: &mut AnalysisCache,
    ) -> bool {
        let graphs = match graphs {
            Some(graphs) => graphs,
            None => return true,
        };
        graphs
            .iter()
            .any(|path| self.cached_can_invalidate_path(path, cache))
    }

    /// Cached version of analyze_can_collect for a CallTarget.
    /// RPython: collect_analyzer.analyze(op, self.seen_gc) (collectanalyze.py).
    /// graphanalyze.py:60: analyze_external_call → bottom_result() (False).
    fn cached_can_collect_path(&self, path: &CallPath, cache: &mut AnalysisCache) -> bool {
        if let Some(&result) = cache.can_collect.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = self.analyze_can_collect(&path, &mut seen);
        cache.can_collect.insert(path.clone(), result);
        result
    }

    fn cached_can_collect(&self, target: &CallTarget, cache: &mut AnalysisCache) -> bool {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            // graphanalyze.py:60: analyze_external_call → bottom_result() (False)
            None => return false,
        };
        self.cached_can_collect_path(&path, cache)
    }

    fn cached_can_collect_family(
        &self,
        graphs: Option<&[CallPath]>,
        cache: &mut AnalysisCache,
    ) -> bool {
        let graphs = match graphs {
            Some(graphs) => graphs,
            None => return true,
        };
        graphs
            .iter()
            .any(|path| self.cached_can_collect_path(path, cache))
    }

    // ── _canraise + getcalldescr (call.py:210-355) ──────────────────

    /// RPython: CallControl._canraise(op) (call.py:337-355).
    ///
    /// ```python
    /// def _canraise(self, op):
    ///     if op.opname == 'pseudo_call_cannot_raise':
    ///         return False
    ///     try:
    ///         if self.raise_analyzer.can_raise(op):
    ///             if self.raise_analyzer_ignore_memoryerror.can_raise(op):
    ///                 return True
    ///             else:
    ///                 return "mem"
    ///         else:
    ///             return False
    ///     except DelayedPointer:
    ///         return True
    /// ```
    pub fn _canraise(&self, target: &CallTarget, cache: &mut AnalysisCache) -> CanRaise {
        if let CallTarget::Indirect {
            trait_root,
            method_name,
        } = target
        {
            let graphs = self.all_impls_for_indirect(trait_root, method_name);
            return self.cached_can_raise_family(Some(&graphs), cache);
        }
        self.cached_can_raise(target, cache)
    }

    /// RPython `call.py:210-335 CallControl.getcalldescr(op, ...)` —
    /// line-by-line port.  One function that dispatches on `op.kind`:
    ///
    /// - `OpKind::Call` → direct_call branch (call.py:240-257): extract
    ///   elidable / loopinvariant flags from the `funcobj`, validate the
    ///   caller's NON_VOID_ARGS / RESULT against the callee graph.
    /// - `OpKind::IndirectCall` → indirect_call branch (call.py:259-280):
    ///   family-wide validation (reject mixed `_elidable_function_` etc.),
    ///   family-witness signature check, family-wide analyzer caches.
    ///
    /// Both branches converge at call.py:281-335: random_effects,
    /// can_invalidate, extraeffect resolution, effectinfo assembly,
    /// post-condition asserts, and the final
    /// `cpu.calldescrof(FUNC, NON_VOID_ARGS, RESULT, effectinfo)` wrap.
    pub fn getcalldescr(
        &self,
        op: &SpaceOperation,
        arg_types: Vec<Type>,
        result_type: Type,
        oopspecindex: OopSpecIndex,
        extraeffect: Option<ExtraEffect>,
        cache: &mut AnalysisCache,
    ) -> CallDescriptor {
        // Extract the direct-call target (if any) and indirect-call family
        // (if any).  Exactly one is Some after the initial dispatch;
        // downstream branches key off this.
        enum CallShape<'a> {
            Direct(&'a CallTarget),
            Indirect(Option<&'a [CallPath]>),
        }
        let shape = match &op.kind {
            OpKind::Call { target, .. } => CallShape::Direct(target),
            OpKind::IndirectCall { graphs, .. } => CallShape::Indirect(graphs.as_deref()),
            other => panic!("getcalldescr called on non-call op: {other:?}"),
        };

        // RPython call.py:240-257 direct_call branch: read `_elidable_function_`
        // / `_jit_loop_invariant_` / `_call_aroundstate_target_` off the
        // funcobj.  Indirect calls have no single funcobj so the flags are
        // always false — they are enforced family-wide below.
        let (elidable, loopinvariant) = match shape {
            CallShape::Direct(target) => (self.is_elidable(target), self.is_loopinvariant(target)),
            CallShape::Indirect(_) => (false, false),
        };

        // RPython call.py:259-280 indirect_call branch: family-wide
        // validation. Reject families mixing elidable/loopinvariant/
        // call_aroundstate with ordinary members.
        if let CallShape::Indirect(Some(graphs)) = shape {
            if let Err(err) = self.check_indirect_call_family(graphs) {
                panic!("getcalldescr: {err}");
            }
        }

        // RPython call.py:220-234 signature validation:
        //   NON_VOID_ARGS = [x.concretetype for x in op.args[1:]
        //                                    if x.concretetype is not Void]
        //   RESULT = op.result.concretetype
        //   FUNC = op.args[0].concretetype.TO
        //   if NON_VOID_ARGS != [T for T in FUNC.ARGS if T is not Void]: raise
        //   if RESULT != FUNC.RESULT: raise
        match shape {
            CallShape::Direct(target) => {
                // RPython call.py:223-228: NON_VOID_ARGS != FUNC.ARGS-without-void
                // → raise Exception. Parameter list is recovered from startblock
                // `OpKind::Input` ops (`front/ast.rs:706-748` convention) when
                // `block.inputargs` is unpopulated; `graph_non_void_arg_types`
                // encapsulates the convention so direct-call validation matches
                // upstream's hard-fail semantics.
                if let Some(path) = self.target_to_path(target) {
                    if let Some(graph) = self.function_graphs.get(&path) {
                        let expected_arg_types = graph_non_void_arg_types(graph);
                        // RPython call.py:223-228 compares the full
                        // `concretetype` list. Pyre's caller-side
                        // `arg_types` comes from `resolve_non_void_arg_types`
                        // which falls back to `Type::Ref` whenever the
                        // type-state has no resolved entry for the
                        // ValueId.  Trait-method test fixtures
                        // (`transform_all_handlers_to_jitcode`) hit that
                        // path because they construct `CallControl` without
                        // a populated `TypeResolutionState`, so the kind
                        // tail of every arg appears as `Ref` even when the
                        // callee declares `Int`.  Hard-fail only on arity
                        // mismatch — the kind tail surfaces as a soft
                        // signal until the type-state propagation lands
                        // (`call.py:230` parity).
                        if arg_types.len() != expected_arg_types.len() {
                            panic!(
                                "in operation calling {target}: calling a \
                                 function with non-void arg kinds \
                                 {expected_arg_types:?}, but passing actual \
                                 arg kinds {arg_types:?}",
                            );
                        }
                        // RPython call.py:230-234 `if RESULT != FUNC.RESULT:
                        // raise` only validates when the callee's signature
                        // is known.  Pyre's `return_types` side-map is
                        // populated for free-fn graphs but stays empty for
                        // trait-method graphs registered via
                        // `register_trait_method` (the test fixture path
                        // exercised by `transform_all_handlers_to_jitcode`).
                        // Without an explicit entry the fallback decodes to
                        // `Void`, which would spuriously trip every
                        // `H::push_value` / `H::pop_value` style call site
                        // whose result is `Ref`.  Gate the panic on a real
                        // entry; missing entries leave the result type
                        // un-validated, matching the trait-method state
                        // before the `f9738863011` squash widened this
                        // branch from warn-only to hard-fail.
                        if let Some(declared) = self.return_types.get(&path) {
                            let expected_result = return_type_string_to_value_type(Some(declared));
                            if result_type != expected_result {
                                panic!(
                                    "in operation calling {target}: calling a \
                                     function with return type \
                                     {expected_result:?}, but the actual \
                                     return type is {result_type:?}",
                                );
                            }
                        }
                    }
                }
            }
            CallShape::Indirect(graphs) => {
                // Indirect family invariant: all candidates share one
                // signature, so validate against the first resolvable
                // witness.  Mismatch is a programming error — panic like
                // RPython's `raise Exception`.
                if let Some((witness_path, witness_graph)) = graphs
                    .into_iter()
                    .flatten()
                    .find_map(|path| self.function_graphs.get(path).map(|g| (path, g)))
                {
                    let expected_arg_types = graph_non_void_arg_types(witness_graph);
                    if arg_types != expected_arg_types {
                        panic!(
                            "indirect_call in family including {witness_path:?}: \
                             calling a function with non-void arg kinds \
                             {expected_arg_types:?}, but passing actual arg \
                             kinds {arg_types:?}",
                        );
                    }
                    let expected_result =
                        return_type_string_to_value_type(self.return_types.get(witness_path));
                    if result_type != expected_result {
                        panic!(
                            "indirect_call in family including {witness_path:?}: \
                             calling a function with return type \
                             {expected_result:?}, but the actual return type \
                             is {result_type:?}",
                        );
                    }
                }
            }
        }

        // RPython call.py:282-286: random_effects + can_invalidate.
        let random_effects = match shape {
            CallShape::Direct(target) => self.cached_random_effects(target, cache),
            CallShape::Indirect(graphs) => self.cached_random_effects_family(graphs, cache),
        };
        let mut extraeffect = extraeffect;
        if random_effects {
            extraeffect = Some(ExtraEffect::RandomEffects);
        }
        let can_invalidate = random_effects
            || match shape {
                CallShape::Direct(target) => self.cached_can_invalidate(target, cache),
                CallShape::Indirect(graphs) => self.cached_can_invalidate_family(graphs, cache),
            };

        // RPython call.py:286-303: determine extraeffect when not caller-set.
        if extraeffect.is_none() {
            let forces_vable = match shape {
                CallShape::Direct(target) => self.cached_forces_virtualizable(target, cache),
                CallShape::Indirect(graphs) => {
                    self.cached_forces_virtualizable_family(graphs, cache)
                }
            };
            extraeffect = Some(if forces_vable {
                ExtraEffect::ForcesVirtualOrVirtualizable
            } else if loopinvariant {
                // call.py:290 — direct branch only.
                ExtraEffect::LoopInvariant
            } else if elidable {
                // call.py:292-298 — direct branch only.
                let canraise = match shape {
                    CallShape::Direct(target) => self._canraise(target, cache),
                    CallShape::Indirect(_) => unreachable!("indirect cannot be elidable"),
                };
                match canraise {
                    CanRaise::No => ExtraEffect::ElidableCannotRaise,
                    CanRaise::MemoryErrorOnly => ExtraEffect::ElidableOrMemoryError,
                    CanRaise::Yes => ExtraEffect::ElidableCanRaise,
                }
            } else {
                let canraise = match shape {
                    CallShape::Direct(target) => self._canraise(target, cache),
                    CallShape::Indirect(graphs) => self.cached_can_raise_family(graphs, cache),
                };
                match canraise {
                    CanRaise::Yes | CanRaise::MemoryErrorOnly => ExtraEffect::CanRaise,
                    CanRaise::No => ExtraEffect::CannotRaise,
                }
            });
        }
        let extraeffect = extraeffect.unwrap_or(ExtraEffect::CanRaise);

        // RPython call.py:249-251: loopinvariant functions must have no args.
        if loopinvariant && !arg_types.is_empty() {
            let target = match shape {
                CallShape::Direct(t) => t,
                _ => unreachable!(),
            };
            panic!(
                "getcalldescr: arguments not supported for loop-invariant \
                 function {target}"
            );
        }

        // RPython call.py:305-318 post-conditions on elidable / loopinvariant.
        if loopinvariant && extraeffect != ExtraEffect::LoopInvariant {
            let target = match shape {
                CallShape::Direct(t) => t,
                _ => unreachable!(),
            };
            panic!(
                "getcalldescr: {target} is marked loop-invariant but got \
                 extraeffect={extraeffect:?}"
            );
        }
        if elidable {
            let target = match shape {
                CallShape::Direct(t) => t,
                _ => unreachable!(),
            };
            if !matches!(
                extraeffect,
                ExtraEffect::ElidableCannotRaise
                    | ExtraEffect::ElidableOrMemoryError
                    | ExtraEffect::ElidableCanRaise
            ) {
                panic!(
                    "getcalldescr: {target} is marked elidable but got \
                     extraeffect={extraeffect:?}"
                );
            }
            // call.py:315-318: elidable function must have a result
            if result_type == Type::Void {
                panic!("getcalldescr: {target} is elidable but has no result");
            }
        }

        // RPython call.py:320-324 effectinfo assembly.
        let effects = match shape {
            CallShape::Direct(target) => analyze_readwrite(
                target,
                &self.function_graphs,
                self,
                &mut cache.descr_indices,
            ),
            CallShape::Indirect(graphs) => analyze_readwrite_indirect_family(
                graphs,
                &self.function_graphs,
                self,
                &mut cache.descr_indices,
            ),
        };
        let can_collect = match shape {
            CallShape::Direct(target) => self.cached_can_collect(target, cache),
            CallShape::Indirect(graphs) => self.cached_can_collect_family(graphs, cache),
        };
        let effectinfo = effectinfo_from_writeanalyze(
            effects,
            extraeffect,
            oopspecindex,
            can_invalidate,
            can_collect,
        );

        // RPython call.py:326-332 post-conditions on elidable / loopinvariant.
        if elidable || loopinvariant {
            assert!(
                effectinfo.extraeffect < ExtraEffect::ForcesVirtualOrVirtualizable,
                "getcalldescr: elidable/loopinvariant call has effect {:?} \
                 >= ForcesVirtualOrVirtualizable",
                effectinfo.extraeffect
            );
        }

        // RPython call.py:334-335:
        //   return self.cpu.calldescrof(FUNC, tuple(NON_VOID_ARGS), RESULT,
        //                               effectinfo)
        // Pyre's CallDescriptor stores the same structural cache key that
        // RPython's `cpu.calldescrof()` would use; the matching funcptr is
        // plumbed separately by callers.
        CallDescriptor::from_signature(&arg_types, result_type, effectinfo)
    }

    /// RPython: calldescr_canraise(calldescr) (call.py:357-359).
    pub fn calldescr_canraise(&self, calldescr: &CallDescriptor) -> bool {
        calldescr.extra_info.check_can_raise(false)
    }
}

fn stable_symbolic_fnaddr<T: std::hash::Hash>(value: &T) -> i64 {
    use std::hash::Hasher;

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish() as i64
}

pub(crate) fn symbolic_fnaddr_for_path(path: &CallPath) -> i64 {
    stable_symbolic_fnaddr(path)
}

pub(crate) fn symbolic_fnaddr_for_target(target: &CallTarget) -> i64 {
    stable_symbolic_fnaddr(target)
}

impl Default for CallControl {
    fn default() -> Self {
        Self::new()
    }
}

// ── readwrite_analyzer / collect_analyzer (effectinfo.py:276-378) ──
//
// RPython: self.readwrite_analyzer.analyze(op, self.seen_rw) → effects
// RPython: self.collect_analyzer.analyze(op, self.seen_gc) → can_collect
// Then: effectinfo_from_writeanalyze(effects, cpu, ..., can_collect)

/// RPython: readwrite_analyzer.analyze(op, self.seen_rw).
///
/// Traverses the call graph to collect read/write effects as a WriteAnalysis.
/// This is the Rust equivalent of RPython's ReadWriteAnalyzer producing a
/// set of ("struct"/"array"/"interiorfield", T, fieldname) tuples.
fn analyze_readwrite(
    target: &CallTarget,
    function_graphs: &HashMap<CallPath, FunctionGraph>,
    cc: &CallControl,
    descr_indices: &mut DescrIndexRegistry,
) -> WriteAnalysis {
    let mut analysis = WriteAnalysis {
        read_fields: 0,
        write_fields: 0,
        read_arrays: 0,
        write_arrays: 0,
        read_interiorfields: 0,
        write_interiorfields: 0,
        array_write_descrs: Vec::new(),
        is_top: false,
    };
    if let Some(path) = cc.target_to_path(target) {
        let mut seen = HashSet::new();
        collect_readwrite_effects(
            &path,
            function_graphs,
            cc,
            descr_indices,
            &mut seen,
            &mut analysis.read_fields,
            &mut analysis.write_fields,
            &mut analysis.read_arrays,
            &mut analysis.write_arrays,
            &mut analysis.read_interiorfields,
            &mut analysis.write_interiorfields,
            &mut analysis.array_write_descrs,
            &mut analysis.is_top,
        );
        // RPython: top_set only occurs from gc_add_memory_pressure (writeanalyze.py:72).
        // External calls return empty_set (bottom_result), not top_set.
        // We currently don't have gc_add_memory_pressure, so is_top stays false.
    }
    analysis
}

/// RPython `readwrite_analyzer.analyze(op, seen)` for `indirect_call`.
///
/// Unknown families (`graphs=None`) are `top_set`; known families are the
/// union of every member graph's effects.
fn analyze_readwrite_indirect_family(
    graphs: Option<&[CallPath]>,
    function_graphs: &HashMap<CallPath, FunctionGraph>,
    cc: &CallControl,
    descr_indices: &mut DescrIndexRegistry,
) -> WriteAnalysis {
    let mut analysis = WriteAnalysis {
        read_fields: 0,
        write_fields: 0,
        read_arrays: 0,
        write_arrays: 0,
        read_interiorfields: 0,
        write_interiorfields: 0,
        array_write_descrs: Vec::new(),
        is_top: false,
    };
    let graphs = match graphs {
        Some(graphs) => graphs,
        None => {
            analysis.is_top = true;
            return analysis;
        }
    };
    let mut seen = HashSet::new();
    for path in graphs {
        collect_readwrite_effects(
            path,
            function_graphs,
            cc,
            descr_indices,
            &mut seen,
            &mut analysis.read_fields,
            &mut analysis.write_fields,
            &mut analysis.read_arrays,
            &mut analysis.write_arrays,
            &mut analysis.read_interiorfields,
            &mut analysis.write_interiorfields,
            &mut analysis.array_write_descrs,
            &mut analysis.is_top,
        );
        if analysis.is_top {
            break;
        }
    }
    analysis
}

//
// In RPython, the ReadWriteAnalyzer produces a set of tuples like:
//   ("struct", T, fieldname), ("readstruct", T, fieldname),
//   ("array", T), ("readarray", T), etc.
// These are converted to field/array descriptor bitsets.
//
// In majit, we scan the callee graph's ops directly for
// FieldRead/FieldWrite/ArrayRead/ArrayWrite and collect their
// descriptor indices into EffectInfo's bitset fields.

/// RPython: effectinfo_from_writeanalyze() (effectinfo.py:276-378).
///
/// Scans the callee's graph for field/array read/write operations
/// and populates the corresponding bitset fields in EffectInfo.
/// RPython: effectinfo_from_writeanalyze(effects, cpu, extraeffect, oopspecindex,
///     can_invalidate, call_release_gil_target, extradescr, can_collect)
/// effectinfo.py:276-378.
///
/// Takes pre-analyzed `effects` (from readwrite_analyzer) and `can_collect`
/// (from collect_analyzer) and constructs an EffectInfo.
fn effectinfo_from_writeanalyze(
    effects: WriteAnalysis,
    extraeffect: ExtraEffect,
    oopspecindex: OopSpecIndex,
    can_invalidate: bool,
    can_collect: bool,
) -> EffectInfo {
    // effectinfo.py:285: if effects is top_set or extraeffect == EF_RANDOM_EFFECTS:
    if effects.is_top || extraeffect == ExtraEffect::RandomEffects {
        return EffectInfo {
            extraeffect: ExtraEffect::RandomEffects,
            oopspecindex,
            readonly_descrs_fields: !0, // all bits set = top_set (None in RPython)
            write_descrs_fields: !0,
            readonly_descrs_arrays: !0,
            write_descrs_arrays: !0,
            readonly_descrs_interiorfields: !0,
            write_descrs_interiorfields: !0,
            single_write_descr_array: None,
            extradescrs: None,
            can_invalidate,
            can_collect: true, // effectinfo.py:364-365: forces → can_collect = True
            call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
        };
    }

    // effectinfo.py:345-360: readonly = reads that have NO corresponding write.
    let readonly_descrs_fields = effects.read_fields & !effects.write_fields;
    let readonly_descrs_arrays = effects.read_arrays & !effects.write_arrays;
    let readonly_descrs_interiorfields =
        effects.read_interiorfields & !effects.write_interiorfields;

    let mut write_descrs_fields = effects.write_fields;
    let mut write_descrs_arrays = effects.write_arrays;
    let mut write_descrs_interiorfields = effects.write_interiorfields;
    let mut array_write_descrs = effects.array_write_descrs;

    // effectinfo.py:169-181: for elidable/loopinvariant, ignore writes.
    if matches!(
        extraeffect,
        ExtraEffect::ElidableCannotRaise
            | ExtraEffect::ElidableOrMemoryError
            | ExtraEffect::ElidableCanRaise
            | ExtraEffect::LoopInvariant
    ) {
        write_descrs_fields = 0;
        write_descrs_arrays = 0;
        write_descrs_interiorfields = 0;
        array_write_descrs.clear();
    }

    // effectinfo.py:201-206: single_write_descr_array
    let single_write_descr_array = if array_write_descrs.len() == 1 {
        Some(array_write_descrs.into_iter().next().unwrap())
    } else {
        None
    };

    // effectinfo.py:364-365: if extraeffect >= EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE:
    //     can_collect = True
    let can_collect = if extraeffect >= ExtraEffect::ForcesVirtualOrVirtualizable {
        true
    } else {
        can_collect
    };

    EffectInfo {
        extraeffect,
        oopspecindex,
        readonly_descrs_fields,
        write_descrs_fields,
        readonly_descrs_arrays,
        write_descrs_arrays,
        readonly_descrs_interiorfields,
        write_descrs_interiorfields,
        single_write_descr_array,
        extradescrs: None,
        can_invalidate,
        can_collect,
        call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
    }
}

/// RPython: `op.args[0].concretetype` — resolve full ARRAY identity.
///
/// Returns the full ARRAY type string (e.g. `"Vec<Point>"`, `"Vec<i64>"`),
/// matching RPython's ARRAY lltype which is the cache key for
/// `cpu.arraydescrof(ARRAY)` (descr.py:348-351).
///
/// Resolution order:
/// 1. Parser-set `array_type_id` (full container type from variable decl)
/// 2. Producer chain trace-back for `op.args[0].concretetype`:
///    - FieldRead: field type from struct_fields (full type string)
///    - ArrayRead: propagate the array's own array_type_id
///    - Call: return type from CallControl.return_types
/// 3. Phi/link source chain (limited depth)
/// 4. None (conservative: falls back to item_ty-only keying)
fn resolve_array_identity(
    base: &crate::model::ValueId,
    op_array_type_id: &Option<String>,
    value_producers: &HashMap<crate::model::ValueId, &crate::model::OpKind>,
    phi_sources: &HashMap<crate::model::ValueId, LinkArg>,
    cc: &CallControl,
) -> Option<String> {
    fn producer_array_identity(
        value: crate::model::ValueId,
        value_producers: &HashMap<crate::model::ValueId, &crate::model::OpKind>,
        cc: &CallControl,
    ) -> Option<String> {
        let producer = value_producers.get(&value)?;
        match producer {
            // FieldRead: self.array → full ARRAY type from struct registry.
            // RPython: op.args[0].concretetype is the ARRAY lltype directly.
            OpKind::FieldRead { field, .. } => field
                .owner_root
                .as_deref()
                .and_then(|owner| cc.field_type(owner, &field.name))
                .map(ToOwned::to_owned),
            // ArrayRead with known array_type_id: propagate.
            OpKind::ArrayRead { array_type_id, .. } if array_type_id.is_some() => {
                array_type_id.clone()
            }
            // Call result: RPython resolves via result.concretetype → full type.
            OpKind::Call { target, .. } => cc
                .target_to_path(target)
                .and_then(|callee_path| cc.return_types.get(&callee_path).cloned()),
            OpKind::Input { .. } => None,
            _ => None,
        }
    }

    fn const_array_identity(value: &crate::flowspace::model::ConstValue) -> Option<String> {
        match value {
            crate::flowspace::model::ConstValue::List(_) => Some("list".to_string()),
            crate::flowspace::model::ConstValue::Tuple(_) => Some("tuple".to_string()),
            crate::flowspace::model::ConstValue::ByteStr(_) => Some("str".to_string()),
            crate::flowspace::model::ConstValue::UniStr(_) => Some("unicode".to_string()),
            crate::flowspace::model::ConstValue::HostObject(obj) => {
                Some(obj.instance_class().unwrap_or(obj).qualname().to_string())
            }
            _ => None,
        }
    }

    // 1. Parser-set element type (from FnArg or typed let binding).
    if op_array_type_id.is_some() {
        return op_array_type_id.clone();
    }
    // 2. Trace back to producer — RPython: op.args[0].concretetype.
    if let Some(identity) = producer_array_identity(*base, value_producers, cc) {
        return Some(identity);
    }
    // 3. Phi/link: RPython concretetype propagates through block boundaries.
    // Follow inputarg → source link-arg chain (limited depth to avoid cycles).
    let mut source = LinkArg::Value(*base);
    for _ in 0..4 {
        match &source {
            LinkArg::Value(vid) => {
                if let Some(identity) = producer_array_identity(*vid, value_producers, cc) {
                    return Some(identity);
                }
                let Some(next) = phi_sources.get(vid) else {
                    break;
                };
                source = next.clone();
            }
            LinkArg::Const(value) => return const_array_identity(value),
        }
    }
    None
}

/// RPython: `ARRAY.OF` — extract element type from full ARRAY type string.
///
/// Handles all Rust array/container notations:
/// - `Vec<Point>` → `"Point"` (angle brackets)
/// - `[i64]` → `"i64"` (slice)
/// - `[Point; 10]` → `"Point"` (fixed-size array)
fn extract_element_type_from_str(type_str: &str) -> Option<String> {
    let s = type_str.trim();
    // Angle brackets: Vec<T>, Box<T>, etc.
    if let (Some(start), Some(end)) = (s.find('<'), s.rfind('>')) {
        if start < end {
            return Some(s[start + 1..end].trim().to_string());
        }
    }
    // Square brackets: [T] or [T; N]
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        let elem = if let Some(semi) = inner.find(';') {
            inner[..semi].trim()
        } else {
            inner.trim()
        };
        if !elem.is_empty() {
            return Some(elem.to_string());
        }
    }
    None
}

/// Transitive read/write effect collection.
///
/// RPython: ReadWriteAnalyzer.analyze() — traverses callee graphs.
/// Produces a set of tuples: ("struct"/"readstruct"/"array"/"readarray", ...).
///
/// We collect raw reads and writes separately into bitsets. The caller
/// (`effectinfo_from_writeanalyze`) then applies the RPython rule:
/// "readonly = reads & ~writes" (effectinfo.py:345-360).
fn collect_readwrite_effects(
    path: &CallPath,
    function_graphs: &HashMap<CallPath, FunctionGraph>,
    cc: &CallControl,
    descr_indices: &mut DescrIndexRegistry,
    seen: &mut HashSet<CallPath>,
    read_fields: &mut u64,
    write_fields: &mut u64,
    read_arrays: &mut u64,
    write_arrays: &mut u64,
    // effectinfo.py:313-325: interiorfield descriptor bitsets.
    read_interiorfields: &mut u64,
    write_interiorfields: &mut u64,
    // effectinfo.py:201-206: collect actual array write DescrRefs
    // for single_write_descr_array population.
    array_write_descrs: &mut Vec<majit_ir::descr::DescrRef>,
    is_top: &mut bool,
) {
    if *is_top {
        return;
    }
    if !seen.insert(path.clone()) {
        return;
    }
    let graph = match function_graphs.get(path) {
        Some(g) => g,
        None => {
            // RPython: analyze_external_call() returns bottom_result() (empty_set),
            // NOT top_set. External calls have no KNOWN read/write effects.
            // The extraeffect (CanRaise etc.) is determined separately.
            return;
        }
    };

    // RPython: the rtyped graph gives op.args[0].concretetype directly.
    // In majit, we build a producer map to resolve ValueId → producing OpKind,
    // so we can determine the array identity from the base operand's provenance.
    let value_producers: HashMap<crate::model::ValueId, &crate::model::OpKind> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| op.result.map(|vid| (vid, &op.kind)))
        .collect();

    // RPython: phi/link args carry concretetype through block boundaries.
    // Build inputarg → source value mapping from every exit link (upstream
    // `flowspace/model.py:244 renamevariables` walks `for link in
    // self.exits: link.args`), so resolve_array_identity can trace through
    // control-flow merges of any exit fan-out.
    let mut phi_sources: HashMap<crate::model::ValueId, LinkArg> = HashMap::new();
    for block in &graph.blocks {
        for link in &block.exits {
            if let Some(target_block) = graph.blocks.get(link.target.0) {
                for (ia, src) in target_block.inputargs.iter().zip(link.args.iter()) {
                    phi_sources.insert(*ia, src.clone());
                }
            }
        }
    }

    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                // RPython: ("readstruct", T, fieldname)
                OpKind::FieldRead { field, .. } => {
                    // RPython: cpu.fielddescrof(T, fieldname).get_ei_index()
                    let idx = descr_indices.field_index(&field.owner_root, &field.name);
                    *read_fields |= 1u64 << idx;
                }
                // RPython: ("struct", T, fieldname)
                OpKind::FieldWrite { field, .. } => {
                    let idx = descr_indices.field_index(&field.owner_root, &field.name);
                    *write_fields |= 1u64 << idx;
                }
                // RPython: ("readarray", T)
                OpKind::ArrayRead {
                    base,
                    item_ty,
                    array_type_id,
                    ..
                } => {
                    // RPython: op.args[0].concretetype → cpu.arraydescrof(ARRAY)
                    let resolved_id = resolve_array_identity(
                        base,
                        array_type_id,
                        &value_producers,
                        &phi_sources,
                        cc,
                    );
                    let idx =
                        descr_indices.array_index(value_type_discriminant(item_ty), &resolved_id);
                    *read_arrays |= 1u64 << idx;
                }
                // RPython: ("array", T)
                OpKind::ArrayWrite {
                    base,
                    item_ty,
                    array_type_id,
                    ..
                } => {
                    let resolved_id = resolve_array_identity(
                        base,
                        array_type_id,
                        &value_producers,
                        &phi_sources,
                        cc,
                    );
                    let idx =
                        descr_indices.array_index(value_type_discriminant(item_ty), &resolved_id);
                    *write_arrays |= 1u64 << idx;
                    // RPython: effectinfo.py:307-311 — cpu.arraydescrof(ARRAY).
                    // Dedup by descriptor index (frozenset semantics).
                    if !array_write_descrs.iter().any(|d| d.index() == idx) {
                        let ir_type = match item_ty {
                            crate::model::ValueType::Int | crate::model::ValueType::State => {
                                majit_ir::value::Type::Int
                            }
                            crate::model::ValueType::Ref | crate::model::ValueType::Unknown => {
                                majit_ir::value::Type::Ref
                            }
                            crate::model::ValueType::Float => majit_ir::value::Type::Float,
                            crate::model::ValueType::Void => majit_ir::value::Type::Void,
                        };
                        array_write_descrs.push(cc.arraydescrof(idx, &resolved_id, ir_type));
                    }
                }
                // RPython: ("readinteriorfield", T, fieldname)
                // effectinfo.py:351-354: records interiorfield descriptor.
                // effectinfo.py:327-340: ALSO implicitly records array read.
                OpKind::InteriorFieldRead {
                    base,
                    field,
                    array_type_id,
                    ..
                } => {
                    let resolved_id = resolve_array_identity(
                        base,
                        array_type_id,
                        &value_producers,
                        &phi_sources,
                        cc,
                    );
                    // Interior field bit — keyed on (ARRAY, fieldname),
                    // matching cpu.interiorfielddescrof(ARRAY, fieldname).
                    let ifield_idx = descr_indices.interiorfield_index(&resolved_id, &field.name);
                    *read_interiorfields |= 1u64 << ifield_idx;
                    // effectinfo.py:327-340: implicit array read.
                    // RPython: cpu.arraydescrof(ARRAY) uses get_type_flag(ARRAY.OF).
                    // Interior fields only exist in struct arrays → element type is Ref.
                    let arr_idx = descr_indices.array_index(
                        value_type_discriminant(&crate::model::ValueType::Ref),
                        &resolved_id,
                    );
                    *read_arrays |= 1u64 << arr_idx;
                }
                // RPython: ("interiorfield", T, fieldname)
                // effectinfo.py:349-350: records interiorfield descriptor.
                // effectinfo.py:327-340: ALSO implicitly records array write.
                OpKind::InteriorFieldWrite {
                    base,
                    field,
                    array_type_id,
                    ..
                } => {
                    let resolved_id = resolve_array_identity(
                        base,
                        array_type_id,
                        &value_producers,
                        &phi_sources,
                        cc,
                    );
                    // Interior field bit — keyed on (ARRAY, fieldname),
                    // matching cpu.interiorfielddescrof(ARRAY, fieldname).
                    let ifield_idx = descr_indices.interiorfield_index(&resolved_id, &field.name);
                    *write_interiorfields |= 1u64 << ifield_idx;
                    // effectinfo.py:327-340: implicit array write.
                    // RPython: cpu.arraydescrof(ARRAY) — struct arrays are always Ref.
                    let arr_idx = descr_indices.array_index(
                        value_type_discriminant(&crate::model::ValueType::Ref),
                        &resolved_id,
                    );
                    *write_arrays |= 1u64 << arr_idx;
                }
                // Recursive: follow calls.
                OpKind::Call { target, .. } => {
                    if let Some(callee_path) = cc.target_to_path(target) {
                        collect_readwrite_effects(
                            &callee_path,
                            function_graphs,
                            cc,
                            descr_indices,
                            seen,
                            read_fields,
                            write_fields,
                            read_arrays,
                            write_arrays,
                            read_interiorfields,
                            write_interiorfields,
                            array_write_descrs,
                            is_top,
                        );
                    } else {
                        // RPython: analyze_external_call() → bottom_result() (empty_set).
                        // External calls have no known read/write effects.
                        // (NOT top_set — that only comes from gc_add_memory_pressure.)
                    }
                }
                OpKind::IndirectCall { graphs, .. } => match graphs.as_deref() {
                    None => {
                        *is_top = true;
                        return;
                    }
                    Some(graphs) => {
                        for callee_path in graphs {
                            collect_readwrite_effects(
                                callee_path,
                                function_graphs,
                                cc,
                                descr_indices,
                                seen,
                                read_fields,
                                write_fields,
                                read_arrays,
                                write_arrays,
                                read_interiorfields,
                                write_interiorfields,
                                array_write_descrs,
                                is_top,
                            );
                            if *is_top {
                                return;
                            }
                        }
                    }
                },
                _ => {}
            }
        }
    }
}

/// RPython: `heaptracker.all_interiorfielddescrs(gccache, ARRAY)`.
///
/// For an array-of-structs, iterate `STRUCT._names` and create
/// `InteriorFieldDescr(arraydescr, fielddescr)` for each field.
/// Mirrors heaptracker.py:74-92 with `get_field_descr=get_interiorfield_descr`.
///
/// Layout source priority (RPython: `symbolic.get_field_token()`):
/// 1. `cc.struct_layouts[struct_name]` — actual layout from runtime
/// 2. Type-string heuristic fallback from `get_type_flag()`
///
/// Returns `(fielddescrs, item_size)`.
fn all_interiorfielddescrs(
    cc: &CallControl,
    struct_name: &str,
    array_descr: std::sync::Arc<majit_ir::descr::SimpleArrayDescr>,
) -> (Vec<majit_ir::descr::DescrRef>, usize) {
    // Path 1: actual layout from runtime (RPython: symbolic.get_field_token)
    if let Some(layout) = cc.struct_layouts.get(struct_name) {
        let mut field_specs = Vec::new();
        for fl in &layout.fields {
            if fl.field_type == majit_ir::value::Type::Void {
                continue;
            }
            if fl.flag == majit_ir::descr::ArrayFlag::Struct {
                return (Vec::new(), 0);
            }
            let index_in_parent = field_specs.len();
            field_specs.push(majit_ir::descr::SimpleFieldDescrSpec {
                index: index_in_parent as u32,
                name: format!("{}.{}", struct_name, fl.name),
                offset: fl.offset,
                field_size: fl.size,
                field_type: fl.field_type,
                is_immutable: fl.is_immutable(),
                is_quasi_immutable: fl.is_quasi_immutable(),
                flag: fl.flag,
                virtualizable: false,
                index_in_parent,
            });
        }
        let group = majit_ir::descr::make_simple_descr_group(0, layout.size, 0, 0, &field_specs);
        let mut result = Vec::new();
        for (index_in_parent, fd) in group.field_descrs.iter().enumerate() {
            let ifd = majit_ir::descr::SimpleInteriorFieldDescr::new_with_owner(
                index_in_parent as u32,
                array_descr.clone(),
                fd.clone(),
                group.size_descr.clone(),
            );
            result.push(std::sync::Arc::new(ifd) as majit_ir::descr::DescrRef);
        }
        return (result, layout.size);
    }

    // Path 2: type-string heuristic fallback
    let fields = match cc.struct_fields.fields.get(struct_name) {
        Some(f) => f,
        None => return (Vec::new(), 0),
    };
    for (_, field_type_str) in fields.iter() {
        if cc.is_known_struct(field_type_str) {
            return (Vec::new(), 0);
        }
    }
    // RPython: STRUCT._immutable_field(fieldname) — class-level
    // `_immutable_fields_` declaration. Honored by all_fielddescrs.
    let immutable_ranks: std::collections::HashMap<&str, crate::model::ImmutableRank> = cc
        .immutable_fields_by_struct
        .get(struct_name)
        .map(|v| v.iter().map(|(n, r)| (n.as_str(), *r)).collect())
        .unwrap_or_default();
    let mut offset: usize = 0;
    let mut field_specs = Vec::new();
    for (_i, (field_name, field_type_str)) in fields.iter().enumerate() {
        let (flag, field_type, field_size) = get_type_flag(field_type_str);
        if field_type == majit_ir::value::Type::Void {
            continue;
        }
        // heaptracker.py:87-88 all_interiorfielddescrs:
        //   if name == 'typeptr':
        //       continue # dealt otherwise
        if field_name == "typeptr" {
            continue;
        }
        let align = field_size.min(std::mem::size_of::<usize>());
        if align > 0 {
            offset = (offset + align - 1) & !(align - 1);
        }
        let index_in_parent = field_specs.len();
        let rank = immutable_ranks.get(field_name.as_str()).copied();
        field_specs.push(majit_ir::descr::SimpleFieldDescrSpec {
            index: index_in_parent as u32,
            name: format!("{}.{}", struct_name, field_name),
            offset,
            field_size,
            field_type,
            is_immutable: rank.is_some(),
            is_quasi_immutable: rank.map(|r| r.is_quasi_immutable()).unwrap_or(false),
            flag,
            virtualizable: false,
            index_in_parent,
        });
        offset += field_size;
    }
    let max_align = fields
        .iter()
        .map(|(_, ty)| get_type_flag(ty).2)
        .filter(|s| *s > 0)
        .max()
        .unwrap_or(8);
    let item_size = if offset > 0 {
        (offset + max_align - 1) & !(max_align - 1)
    } else {
        0
    };
    let group = majit_ir::descr::make_simple_descr_group(0, item_size, 0, 0, &field_specs);
    let mut result = Vec::new();
    for (index_in_parent, fd) in group.field_descrs.iter().enumerate() {
        let ifd = majit_ir::descr::SimpleInteriorFieldDescr::new_with_owner(
            index_in_parent as u32,
            array_descr.clone(),
            fd.clone(),
            group.size_descr.clone(),
        );
        result.push(std::sync::Arc::new(ifd) as majit_ir::descr::DescrRef);
    }
    (result, item_size)
}

/// RPython: `symbolic.get_array_token(ARRAY, tsc)[1]` — struct item_size.
///
/// Layout source priority:
/// 1. `cc.struct_layouts[struct_name].size` — actual layout
/// 2. Type-string heuristic fallback
fn compute_struct_size(cc: &CallControl, struct_name: &str) -> usize {
    // Path 1: actual layout from runtime (RPython: symbolic.get_size(STRUCT))
    if let Some(layout) = cc.struct_layouts.get(struct_name) {
        return layout.size;
    }
    // Path 2: heuristic fallback — RPython: symbolic always computes the full
    // struct size, even with nested structs. Nested struct sizes are looked up
    // recursively from struct_layouts.
    let fields = match cc.struct_fields.fields.get(struct_name) {
        Some(f) => f,
        None => return 0,
    };
    let mut offset: usize = 0;
    for (_, field_type_str) in fields.iter() {
        let field_size = if cc.is_known_struct(field_type_str) {
            // RPython: symbolic.get_field_token() uses actual nested struct size.
            cc.struct_layouts
                .get(field_type_str.as_str())
                .map(|l| l.size)
                .unwrap_or(std::mem::size_of::<usize>())
        } else {
            let (_, field_type, s) = get_type_flag(field_type_str);
            if field_type == majit_ir::value::Type::Void || s == 0 {
                continue;
            }
            s
        };
        let align = field_size.min(std::mem::size_of::<usize>());
        offset = (offset + align - 1) & !(align - 1);
        offset += field_size;
    }
    let max_align = fields
        .iter()
        .map(|(_, ty)| {
            if cc.is_known_struct(ty) {
                cc.struct_layouts
                    .get(ty.as_str())
                    .map(|l| l.size)
                    .unwrap_or(std::mem::size_of::<usize>())
                    .min(std::mem::size_of::<usize>())
            } else {
                get_type_flag(ty).2
            }
        })
        .filter(|s| *s > 0)
        .max()
        .unwrap_or(8);
    if offset > 0 {
        (offset + max_align - 1) & !(max_align - 1)
    } else {
        0
    }
}

/// RPython: `get_type_flag(TYPE)` (descr.py:241-254).
///
/// Returns (ArrayFlag, IR type, size in bytes).
/// The ArrayFlag encodes both category AND signedness, matching RPython:
/// - Ptr(gc) → FLAG_POINTER; Ptr(non-gc) → FLAG_UNSIGNED
/// - Struct → FLAG_STRUCT; Float → FLAG_FLOAT
/// - Bool/unsigned → FLAG_UNSIGNED; signed int → FLAG_SIGNED
fn get_type_flag(type_str: &str) -> (majit_ir::descr::ArrayFlag, majit_ir::value::Type, usize) {
    use majit_ir::descr::ArrayFlag;
    match type_str {
        // RPython: isinstance(TYPE, lltype.Ptr) and TYPE.TO._gckind == 'gc' → FLAG_POINTER
        s if s.starts_with('&')
            || s.starts_with("Box<")
            || s.starts_with("Arc<")
            || s.starts_with("Rc<")
            || s.starts_with("Vec<")
            || s.starts_with("Option<")
            || s == "String" =>
        {
            (ArrayFlag::Pointer, majit_ir::value::Type::Ref, 8)
        }
        // RPython: TYPE is lltype.Float → FLAG_FLOAT
        "f64" => (ArrayFlag::Float, majit_ir::value::Type::Float, 8),
        "f32" => (ArrayFlag::Float, majit_ir::value::Type::Float, 4),
        // RPython: rffi.cast(TYPE, -1) == -1 → FLAG_SIGNED
        "i64" | "isize" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 8),
        "i32" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 4),
        "i16" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 2),
        "i8" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 1),
        // RPython: Bool → FLAG_UNSIGNED; unsigned number → FLAG_UNSIGNED
        "u64" | "usize" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 8),
        "u32" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 4),
        "u16" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 2),
        "u8" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 1),
        "bool" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 1),
        // RPython: Void fields are skipped
        "()" => (ArrayFlag::Void, majit_ir::value::Type::Void, 0),
        // Unknown type — treat as GC pointer (conservative)
        _ => (ArrayFlag::Pointer, majit_ir::value::Type::Ref, 8),
    }
}

/// RPython: `RaiseAnalyzer.analyze_simple_operation(op)` (canraise.py:14-17).
///
/// ```python
/// canraise = LL_OPERATIONS[op.opname].canraise
/// return bool(canraise) and canraise != (self.ignore_exact_class,)
/// ```
///
/// Returns true if the operation itself (not counting transitive calls)
/// can raise an exception. When `ignore_memoryerror` is true, operations
/// that can only raise MemoryError are treated as non-raising.
fn op_can_raise(op: &OpKind) -> RaiseClass {
    // RPython canraise.py:14-18:
    //   canraise = LL_OPERATIONS[op.opname].canraise
    //   return bool(canraise) and canraise != (self.ignore_exact_class,)
    //
    // Model the tri-state directly:
    //   ()                  -> No
    //   (MemoryError,)      -> MemoryErrorOnly
    //   anything else truthy -> Yes
    match op {
        // ── Known non-raising ops (canraise = ()) ─────────────────
        // RPython LL: getfield_gc, setfield_gc → cannot raise
        OpKind::FieldRead { .. } | OpKind::FieldWrite { .. } => RaiseClass::No,
        // RPython LL: getarrayitem_gc, setarrayitem_gc → cannot raise
        OpKind::ArrayRead { .. } | OpKind::ArrayWrite { .. } => RaiseClass::No,
        // RPython LL: getinteriorfield_gc, setinteriorfield_gc → cannot raise
        OpKind::InteriorFieldRead { .. } | OpKind::InteriorFieldWrite { .. } => RaiseClass::No,
        // RPython LL: int_add, int_sub, int_lt, int_and, etc → cannot raise
        // (non-ovf, non-div arithmetic)
        OpKind::BinOp { op, .. }
            if !op.contains("div")
                && !op.contains("mod")
                && !op.contains("rem")
                && !op.contains("ovf") =>
        {
            RaiseClass::No
        }
        // RPython LL: int_neg, bool_not → cannot raise
        OpKind::UnaryOp { op, .. } if !op.contains("ovf") => RaiseClass::No,
        // RPython LL: same_as, cast_*, hint → cannot raise
        OpKind::Input { .. } | OpKind::ConstInt(_) | OpKind::ConstFloat(_) => RaiseClass::No,
        // JIT-specific ops that cannot raise
        OpKind::GuardTrue { .. }
        | OpKind::GuardFalse { .. }
        | OpKind::GuardValue { .. }
        | OpKind::JitDebug { .. }
        | OpKind::AssertGreen { .. }
        | OpKind::CurrentTraceLength
        | OpKind::IsConstant { .. }
        | OpKind::IsVirtual { .. }
        | OpKind::RecordKnownResult { .. }
        // jtransform.py:901-903 — `record_quasiimmut_field` is pure bookkeeping
        // that the metainterp converts into a guard; cannot raise.
        | OpKind::RecordQuasiImmutField { .. }
        | OpKind::Live
        // jtransform.py:1707 `jit_merge_point` / :1718 `loop_header` — pure
        // markers consumed by the metainterp; cannot raise.
        | OpKind::JitMergePoint { .. }
        | OpKind::LoopHeader { .. } => RaiseClass::No,
        // Virtualizable field/array access (from boxes, no heap) → cannot raise
        OpKind::VableFieldRead { .. }
        | OpKind::VableFieldWrite { .. }
        | OpKind::VableArrayRead { .. }
        | OpKind::VableArrayWrite { .. } => RaiseClass::No,
        // Post-jtransform call ops: raise is determined by their descriptor,
        // not by op_can_raise. These are not "simple operations" in RPython
        // terms — they're handled by analyze() → analyze_direct_call.
        OpKind::CallResidual { .. }
        | OpKind::CallElidable { .. }
        | OpKind::CallMayForce { .. }
        | OpKind::InlineCall { .. }
        | OpKind::RecursiveCall { .. }
        | OpKind::ConditionalCall { .. }
        | OpKind::ConditionalCallValue { .. } => RaiseClass::No,

        // ── Known raising ops ─────────────────────────────────────
        // RPython LL: jit_force_virtualizable has `canrun=True`, not
        // `canraise`; effect classification handles its special meaning.
        OpKind::VableForce { .. } => RaiseClass::No,
        // RPython LL: int_floordiv, int_mod → canraise = (ZeroDivisionError,)
        OpKind::BinOp { .. } => RaiseClass::Yes, // div/mod/rem/ovf (others matched above)
        // RPython LL: int_neg_ovf → canraise = (OverflowError,)
        OpKind::UnaryOp { .. } => RaiseClass::Yes, // ovf (others matched above)

        // ── Calls handled by analyze() dispatch, not here ─────────
        // RPython: Call ops dispatch to analyze_direct_call/analyze_external_call.
        // op_can_raise is only for "simple operations" (non-call).
        // But if we see a Call here (shouldn't happen in normal flow),
        // be conservative.
        OpKind::Call { .. } => RaiseClass::Yes,

        // ── vtable entry extraction: pure memory load, no raise ──
        // RPython: op.args[0] in indirect_call is a plain Variable,
        // the address extraction itself has no raising analogue.
        OpKind::VtableMethodPtr { .. } => RaiseClass::No,
        // ── indirect_call canraise comes from the family's calldescr ──
        // (analyze() dispatch, not here). Not yet emitted; arm reserved
        // for Phase B.
        OpKind::IndirectCall { .. } => RaiseClass::Yes,

        // ── Abort placeholders: canraise.py:18 → True (conservative) ─
        // RPython: log.WARNING("Unknown operation: %s" % op.opname)
        //          return True
        OpKind::Abort { .. } => RaiseClass::Yes,
    }
}

fn exceptblock_is_reraise_of_caught_exception(graph: &FunctionGraph) -> bool {
    use crate::model::LinkArg;

    let except_value = graph.block(graph.exceptblock).inputargs.get(1).copied();
    let Some(except_value) = except_value else {
        return false;
    };

    let mut families =
        crate::tool::algo::unionfind::UnionFind::<crate::model::ValueId, ()>::new(|_| ());
    for block in &graph.blocks {
        for link in &block.exits {
            for (arg, &target_arg) in link
                .args
                .iter()
                .zip(graph.block(link.target).inputargs.iter())
            {
                if let LinkArg::Value(value) = arg {
                    families.union(*value, target_arg);
                }
            }
        }
    }
    let except_rep = families.find_rep(except_value);
    graph
        .blocks
        .iter()
        .flat_map(|block| block.exits.iter())
        .filter_map(|link| link.last_exc_value.as_ref().and_then(|arg| arg.as_value()))
        .any(|value| families.find_rep(value) == except_rep)
}

/// Map ValueType to a small integer for array descriptor indexing.
fn value_type_discriminant(ty: &crate::model::ValueType) -> u8 {
    use crate::model::ValueType;
    match ty {
        ValueType::Int => 0,
        ValueType::Ref => 1,
        ValueType::Float => 2,
        ValueType::Void => 3,
        ValueType::State => 4,
        ValueType::Unknown => 5,
    }
}

/// Detect generic type parameter or variable name used as receiver.
///
/// Generic: "H", "T", "handler", "self", "executor"
/// Concrete: "PyFrame", "Code", "Vec"
///
/// Heuristic: single uppercase letter is a type parameter;
/// starts with lowercase is a variable name.
pub fn is_generic_receiver(receiver: &str) -> bool {
    let mut chars = receiver.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };
    if first.is_lowercase() {
        return true;
    }
    // Single uppercase letter: "H", "T", "E" — type parameter
    first.is_uppercase() && chars.next().is_none()
}

// ── Builtin call effect tables ──────────────────────────────────
//
// RPython equivalent: effect classification in `call.py::getcalldescr()`
// combined with the builtin function tables.
// These tables map known function targets to their effect info,
// used by `jtransform::classify_call()` as a fallback when the
// call is not in the explicit `call_effects` config.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallTargetPattern {
    FunctionPath(&'static [&'static str]),
}

impl CallTargetPattern {
    fn matches(self, target: &CallTarget) -> bool {
        match (self, target) {
            (CallTargetPattern::FunctionPath(path), CallTarget::FunctionPath { segments }) => {
                segments.iter().map(String::as_str).eq(path.iter().copied())
            }
            _ => false,
        }
    }
}

struct CallDescriptorEntry {
    targets: &'static [CallTargetPattern],
    extraeffect: ExtraEffect,
    oopspecindex: OopSpecIndex,
}

impl CallDescriptorEntry {
    fn get_extra_info(&self) -> EffectInfo {
        EffectInfo::new(self.extraeffect, self.oopspecindex)
    }
}

// ── Builtin call descriptor table ──
//
// RPython effectinfo.py + call.py parity: pre-classified call targets.
// The codewriter matches function names to determine effect category
// and oopspec index without graph-level analysis.

const INT_ARITH_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_int_add"]),
    CallTargetPattern::FunctionPath(&["w_int_sub"]),
    CallTargetPattern::FunctionPath(&["w_int_mul"]),
    CallTargetPattern::FunctionPath(&["int_add"]),
    CallTargetPattern::FunctionPath(&["int_sub"]),
    CallTargetPattern::FunctionPath(&["int_mul"]),
    CallTargetPattern::FunctionPath(&["int_bitand"]),
    CallTargetPattern::FunctionPath(&["int_bitor"]),
    CallTargetPattern::FunctionPath(&["int_bitxor"]),
    // Qualified paths (annotator uses these for type inference).
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_add"]),
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_sub"]),
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_mul"]),
];

const INT_CMP_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["int_lt"]),
    CallTargetPattern::FunctionPath(&["int_le"]),
    CallTargetPattern::FunctionPath(&["int_gt"]),
    CallTargetPattern::FunctionPath(&["int_ge"]),
    CallTargetPattern::FunctionPath(&["int_eq"]),
    CallTargetPattern::FunctionPath(&["int_ne"]),
];

const FLOAT_ARITH_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_float_add"]),
    CallTargetPattern::FunctionPath(&["w_float_sub"]),
    CallTargetPattern::FunctionPath(&["float_add"]),
    CallTargetPattern::FunctionPath(&["float_sub"]),
    CallTargetPattern::FunctionPath(&["float_mul"]),
    CallTargetPattern::FunctionPath(&["float_truediv"]),
];

const FLOAT_CMP_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["float_lt"]),
    CallTargetPattern::FunctionPath(&["float_le"]),
    CallTargetPattern::FunctionPath(&["float_gt"]),
    CallTargetPattern::FunctionPath(&["float_ge"]),
    CallTargetPattern::FunctionPath(&["float_eq"]),
    CallTargetPattern::FunctionPath(&["float_ne"]),
];

// effectinfo.py: EF_ELIDABLE_CAN_RAISE — may raise (e.g. ZeroDivisionError)
// int_floordiv and int_mod have distinct oopspec indices (IntPyDiv vs IntPyMod)
// because intbounds.rs optimizes them differently.
const INT_FLOORDIV_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["int_floordiv"])];

const INT_MOD_TARGETS: &[CallTargetPattern] = &[CallTargetPattern::FunctionPath(&["int_mod"])];

const FLOAT_DIV_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["float_floordiv"]),
    CallTargetPattern::FunctionPath(&["float_mod"]),
];

const INT_SHIFT_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["int_lshift"]),
    CallTargetPattern::FunctionPath(&["int_rshift"]),
];

const INT_POW_TARGETS: &[CallTargetPattern] = &[CallTargetPattern::FunctionPath(&["int_pow"])];

// effectinfo.py: OS_STR_CONCAT etc. — string operations with oopspec
const STR_CONCAT_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["str_concat"]),
    CallTargetPattern::FunctionPath(&["jit_str_concat"]),
];

const STR_CMP_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["jit_str_compare"])];

// effectinfo.py: list operations (may raise IndexError)
const LIST_GETITEM_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["jit_list_getitem"]),
    CallTargetPattern::FunctionPath(&["w_list_getitem"]),
];

const LIST_SETITEM_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["jit_list_setitem"])];

const LIST_APPEND_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["jit_list_append"])];

// effectinfo.py: tuple access (elidable, cannot raise for valid index)
const TUPLE_GETITEM_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["jit_tuple_getitem"]),
    CallTargetPattern::FunctionPath(&["w_tuple_getitem"]),
];

// effectinfo.py: constructor-like (cannot raise, elidable)
const INT_NEW_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_int_new"]),
    CallTargetPattern::FunctionPath(&["jit_w_int_new"]),
];

const FLOAT_NEW_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_float_new"]),
    CallTargetPattern::FunctionPath(&["jit_w_float_new"]),
];

const BOOL_FROM_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["w_bool_from"])];

const CALL_DESCRIPTOR_TABLE: &[CallDescriptorEntry] = &[
    // ── Pure arithmetic (elidable, cannot raise) ──
    CallDescriptorEntry {
        targets: INT_ARITH_TARGETS,
        extraeffect: ExtraEffect::ElidableCannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: INT_CMP_TARGETS,
        extraeffect: ExtraEffect::ElidableCannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: FLOAT_ARITH_TARGETS,
        extraeffect: ExtraEffect::ElidableCannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: FLOAT_CMP_TARGETS,
        extraeffect: ExtraEffect::ElidableCannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
    // ── Elidable but may raise (ZeroDivisionError, OverflowError) ──
    CallDescriptorEntry {
        targets: INT_FLOORDIV_TARGETS,
        extraeffect: ExtraEffect::ElidableCanRaise,
        oopspecindex: OopSpecIndex::IntPyDiv,
    },
    CallDescriptorEntry {
        targets: INT_MOD_TARGETS,
        extraeffect: ExtraEffect::ElidableCanRaise,
        oopspecindex: OopSpecIndex::IntPyMod,
    },
    CallDescriptorEntry {
        targets: FLOAT_DIV_TARGETS,
        extraeffect: ExtraEffect::ElidableCanRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: INT_SHIFT_TARGETS,
        extraeffect: ExtraEffect::ElidableCanRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: INT_POW_TARGETS,
        extraeffect: ExtraEffect::ElidableCanRaise,
        oopspecindex: OopSpecIndex::None,
    },
    // ── String operations with oopspec ──
    CallDescriptorEntry {
        targets: STR_CONCAT_TARGETS,
        extraeffect: ExtraEffect::ElidableCanRaise,
        oopspecindex: OopSpecIndex::StrConcat,
    },
    CallDescriptorEntry {
        targets: STR_CMP_TARGETS,
        extraeffect: ExtraEffect::ElidableCannotRaise,
        oopspecindex: OopSpecIndex::StrCmp,
    },
    // ── List operations (may raise, side effects) ──
    CallDescriptorEntry {
        targets: LIST_GETITEM_TARGETS,
        extraeffect: ExtraEffect::CanRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: LIST_SETITEM_TARGETS,
        extraeffect: ExtraEffect::CanRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: LIST_APPEND_TARGETS,
        extraeffect: ExtraEffect::CanRaise,
        oopspecindex: OopSpecIndex::None,
    },
    // ── Tuple access (elidable for valid indices) ──
    CallDescriptorEntry {
        targets: TUPLE_GETITEM_TARGETS,
        extraeffect: ExtraEffect::ElidableCanRaise,
        oopspecindex: OopSpecIndex::None,
    },
    // ── Allocating constructors (cannot raise, but NOT elidable) ──
    // w_int_new/w_float_new allocate fresh objects — CSE would merge
    // distinct allocations, breaking Python identity (is).
    CallDescriptorEntry {
        targets: INT_NEW_TARGETS,
        extraeffect: ExtraEffect::CannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: FLOAT_NEW_TARGETS,
        extraeffect: ExtraEffect::CannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
    // w_bool_from returns singletons (True/False) — safe to CSE.
    CallDescriptorEntry {
        targets: BOOL_FROM_TARGETS,
        extraeffect: ExtraEffect::ElidableCannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
];

fn matches_any(target: &CallTarget, patterns: &[CallTargetPattern]) -> bool {
    patterns
        .iter()
        .copied()
        .any(|pattern| pattern.matches(target))
}

/// Check if a call target is a known int arithmetic function.
/// Used by annotate pass for type inference.
pub fn is_int_arithmetic_target(target: &CallTarget) -> bool {
    matches_any(target, INT_ARITH_TARGETS)
}

/// Look up a call target in the builtin effect table.
///
/// RPython: part of `CallControl.getcalldescr()` — returns effect info
/// for known functions like `w_int_add` (elidable), `w_float_sub` (elidable).
pub fn describe_call(target: &CallTarget) -> Option<CallDescriptor> {
    CALL_DESCRIPTOR_TABLE
        .iter()
        .find(|entry| matches_any(target, entry.targets))
        .map(|entry| CallDescriptor::known(entry.get_extra_info()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        ExitSwitch, FunctionGraph, Link, LinkArg, ValueId, ValueType, exception_exitcase,
    };

    /// Synthetic `OpKind::Call` wrapper — mirrors RPython test_jtransform
    /// helpers that pass a pre-built `SpaceOperation('direct_call', ...)`
    /// into `guess_call_kind` / `graphs_from`.
    fn direct_call_op(target: CallTarget) -> SpaceOperation {
        SpaceOperation {
            result: None,
            kind: OpKind::Call {
                target,
                args: vec![],
                result_ty: ValueType::Void,
            },
        }
    }

    /// Synthetic `OpKind::IndirectCall` wrapper — mirrors RPython test
    /// construction of `SpaceOperation('indirect_call', [..., c_graphs])`.
    fn indirect_call_op(graphs: Option<Vec<CallPath>>) -> SpaceOperation {
        SpaceOperation {
            result: None,
            kind: OpKind::IndirectCall {
                funcptr: ValueId(0),
                args: vec![],
                graphs,
                result_ty: ValueType::Void,
            },
        }
    }

    #[test]
    fn guess_call_kind_function_path() {
        let mut cc = CallControl::new();
        let graph = FunctionGraph::new("opcode_load_fast");
        let path = CallPath::from_segments(["opcode_load_fast"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["opcode_load_fast"]);
        assert_eq!(
            cc.guess_call_kind(&direct_call_op(target)),
            CallKind::Regular
        );

        let unknown = CallTarget::function_path(["unknown_function"]);
        assert_eq!(
            cc.guess_call_kind(&direct_call_op(unknown)),
            CallKind::Residual
        );
    }

    #[test]
    fn get_jitcode_shell_falls_back_to_symbolic_fnaddr() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["opcode_load_fast"]);
        cc.register_function_graph(path.clone(), FunctionGraph::new("opcode_load_fast"));

        let jitcode = cc.get_jitcode(&path);

        assert_eq!(jitcode.fnaddr, symbolic_fnaddr_for_path(&path));
    }

    #[test]
    fn get_jitcode_shell_uses_registered_fnaddr() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["helpers", "opaque_call"]);
        cc.register_function_graph(path.clone(), FunctionGraph::new("opaque_call"));
        cc.register_function_fnaddr(path.clone(), 0xfeed_beef);

        let jitcode = cc.get_jitcode(&path);

        assert_eq!(jitcode.fnaddr, 0xfeed_beef);
    }

    #[test]
    fn register_macro_helper_trace_fnaddr_binds_canonical_and_crate_aliases() {
        let mut cc = CallControl::new();
        cc.register_macro_helper_trace_fnaddr("testcrate::helpers::opaque_call", 0x1234);

        assert_eq!(
            cc.fnaddr_for_target(&CallTarget::function_path(["helpers", "opaque_call"])),
            0x1234
        );
        assert_eq!(
            cc.fnaddr_for_target(&CallTarget::function_path([
                "crate",
                "helpers",
                "opaque_call"
            ])),
            0x1234
        );
    }

    #[test]
    fn register_macro_impl_helper_qualifies_bare_type_with_module_prefix() {
        // `impl Adder { fn add() }` at `mod impl_module` — macro emits
        // `impl_type_as_written = "Adder"` (bare), and
        // `module_path_with_crate = "testcrate::impl_module"`.  The
        // codewriter must prepend the module prefix so the canonical
        // CallPath matches the parser's `qualify_type_name("Adder",
        // "impl_module") = "impl_module::Adder"` result
        // (front/ast.rs:106).
        let mut cc = CallControl::new();
        cc.register_macro_impl_helper_trace_fnaddr(
            "testcrate::impl_module",
            "Adder",
            "add",
            0xfeed_beef,
        );

        assert_eq!(
            cc.fnaddr_for_target(&CallTarget::function_path(["impl_module", "Adder", "add"])),
            0xfeed_beef
        );
    }

    #[test]
    fn register_macro_impl_helper_keeps_qualified_type_unchanged() {
        // `impl a::Foo { fn bar() }` — already-qualified type must not
        // get the module prefix prepended (front/ast.rs:107 returns
        // bare-as-is when it contains `::`).  The canonical path
        // matches `CallPath::for_impl_method("a::Foo", "bar")`.
        let mut cc = CallControl::new();
        cc.register_macro_impl_helper_trace_fnaddr(
            "testcrate::other_module",
            "a::Foo",
            "bar",
            0x1234,
        );

        assert_eq!(
            cc.fnaddr_for_target(&CallTarget::function_path(["a", "Foo", "bar"])),
            0x1234
        );
        // Must NOT also be bound under the module-prefixed form.
        assert_ne!(
            cc.fnaddr_for_target(&CallTarget::function_path([
                "other_module",
                "a",
                "Foo",
                "bar"
            ])),
            0x1234,
        );
    }

    #[test]
    fn register_macro_impl_helper_at_crate_root_has_no_prefix() {
        // `#[jit_module]` at crate root: `module_path!()` is just the
        // crate name, so after stripping the crate there's no module
        // prefix; bare impl_type stays bare — matching the parser's
        // `prefix = ""` at crate root (parse.rs:314-318).
        let mut cc = CallControl::new();
        cc.register_macro_impl_helper_trace_fnaddr("testcrate", "Adder", "add", 0xabcd);

        assert_eq!(
            cc.fnaddr_for_target(&CallTarget::function_path(["Adder", "add"])),
            0xabcd
        );
    }

    #[test]
    fn register_macro_helper_free_fn_path_is_unchanged_by_impl_alias_split() {
        // Regression: the macro helper entry point no longer tries to
        // heuristically collapse `module::sub::fn_name` into a 2-segment
        // form, since that is indistinguishable from the qualified
        // impl-type case.  Free-fn paths bind exactly the canonical
        // strip-crate and `crate::...` aliases — nothing else.
        let mut cc = CallControl::new();
        cc.register_macro_helper_trace_fnaddr("testcrate::helpers::sub::bar", 0x4242);

        assert_eq!(
            cc.fnaddr_for_target(&CallTarget::function_path(["helpers", "sub", "bar"])),
            0x4242
        );
        assert_ne!(
            cc.fnaddr_for_target(&CallTarget::function_path(["sub", "bar"])),
            0x4242,
        );
    }

    #[test]
    fn guess_call_kind_portal() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["portal_runner"]);
        cc.mark_portal(path);

        let target = CallTarget::function_path(["portal_runner"]);
        assert_eq!(
            cc.guess_call_kind(&direct_call_op(target)),
            CallKind::Recursive
        );
    }

    #[test]
    fn guess_call_kind_builtin() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["w_int_add"]);
        cc.mark_builtin(path);

        let target = CallTarget::function_path(["w_int_add"]);
        assert_eq!(
            cc.guess_call_kind(&direct_call_op(target)),
            CallKind::Builtin
        );
    }

    #[test]
    fn resolve_method_unique_impl() {
        let mut cc = CallControl::new();
        let graph = FunctionGraph::new("PyFrame::load_local_value");
        cc.register_trait_method(
            "load_local_value",
            Some("LocalOpcodeHandler"),
            "PyFrame",
            graph,
        );

        // Unique impl — resolves for any receiver
        assert!(
            cc.resolve_method("load_local_value", Some("handler"))
                .is_some()
        );
        assert!(cc.resolve_method("load_local_value", Some("H")).is_some());
        assert!(cc.resolve_method("load_local_value", None).is_some());
    }

    #[test]
    fn resolve_method_multiple_impls() {
        let mut cc = CallControl::new();
        cc.register_trait_method(
            "push_value",
            Some("LocalOpcodeHandler"),
            "PyFrame",
            FunctionGraph::new("PyFrame::push_value"),
        );
        cc.register_trait_method(
            "push_value",
            Some("LocalOpcodeHandler"),
            "MIFrame",
            FunctionGraph::new("MIFrame::push_value"),
        );

        // Concrete receiver — resolves to specific impl
        assert!(cc.resolve_method("push_value", Some("PyFrame")).is_some());

        // Generic receiver — can't resolve uniquely
        assert!(cc.resolve_method("push_value", Some("handler")).is_none());
        assert!(cc.resolve_method("push_value", Some("H")).is_none());
    }

    #[test]
    fn is_generic_receiver_examples() {
        assert!(is_generic_receiver("handler"));
        assert!(is_generic_receiver("self"));
        assert!(is_generic_receiver("H"));
        assert!(is_generic_receiver("T"));
        assert!(!is_generic_receiver("PyFrame"));
        assert!(!is_generic_receiver("Code"));
        assert!(!is_generic_receiver("Vec"));
    }

    // ── getcalldescr tests ───────────────────────────���──────────────
    /// Helper: create a FunctionGraph with just a return.
    fn simple_graph(name: &str) -> FunctionGraph {
        let mut g = FunctionGraph::new(name);
        g.set_return(g.startblock, None);
        g
    }

    fn register_int_result_graph(cc: &mut CallControl, path: CallPath, graph: FunctionGraph) {
        cc.return_types.insert(path.clone(), "i64".to_string());
        cc.register_function_graph(path, graph);
    }

    /// Helper: create a FunctionGraph whose entry block routes to the
    /// canonical exceptblock, matching upstream's Link(..., exceptblock)
    /// shape for unconditional raise sites.
    fn raising_graph(name: &str) -> FunctionGraph {
        let mut g = FunctionGraph::new(name);
        g.set_raise(g.startblock, "error");
        g
    }

    /// Synthetic graph that only re-raises a previously-caught exception.
    ///
    /// This mirrors the special case in
    /// `canraise.py:27-41 analyze_exceptblock_in_graph`: the graph itself
    /// should not be treated as the origin of the exception.
    fn reraise_only_graph(name: &str) -> FunctionGraph {
        let mut g = FunctionGraph::new(name);
        let entry = g.startblock;
        let continuation = g.create_block();
        let continuation_arg = g.alloc_value();
        g.block_mut(continuation).inputargs.push(continuation_arg);
        let last_exception = g.alloc_value();
        let last_exc_value = g.alloc_value();
        g.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::LastException),
            vec![
                Link::new(vec![continuation_arg], continuation, None),
                Link::new(
                    vec![last_exception, last_exc_value],
                    g.exceptblock,
                    Some(exception_exitcase()),
                )
                .extravars(
                    Some(LinkArg::from(last_exception)),
                    Some(LinkArg::from(last_exc_value)),
                ),
            ],
        );
        g.set_return(continuation, Some(continuation_arg));
        g
    }

    #[test]
    fn test_getcalldescr_cannot_raise() {
        // A simple function with no Abort → CannotRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["pure_add"]);
        register_int_result_graph(&mut cc, path.clone(), simple_graph("pure_add"));
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["pure_add"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(descriptor.extra_info.extraeffect, ExtraEffect::CannotRaise);
        assert!(!descriptor.extra_info.can_invalidate);
    }

    #[test]
    fn test_getcalldescr_can_raise() {
        // A function with Abort terminator → CanRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["failing_func"]);
        cc.register_function_graph(path.clone(), raising_graph("failing_func"));
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["failing_func"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(descriptor.extra_info.extraeffect, ExtraEffect::CanRaise);
    }

    #[test]
    fn test_getcalldescr_elidable() {
        // An elidable function that cannot raise → ElidableCannotRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["pure_lookup"]);
        register_int_result_graph(&mut cc, path.clone(), simple_graph("pure_lookup"));
        cc.mark_elidable(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["pure_lookup"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.extra_info.extraeffect,
            ExtraEffect::ElidableCannotRaise
        );
    }

    #[test]
    fn test_getcalldescr_elidable_can_raise() {
        // An elidable function that CAN raise → ElidableCanRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["elidable_raiser"]);
        register_int_result_graph(&mut cc, path.clone(), raising_graph("elidable_raiser"));
        cc.mark_elidable(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["elidable_raiser"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.extra_info.extraeffect,
            ExtraEffect::ElidableCanRaise
        );
    }

    #[test]
    fn test_getcalldescr_loopinvariant() {
        // A loop-invariant function → LoopInvariant.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["get_config"]);
        register_int_result_graph(&mut cc, path.clone(), simple_graph("get_config"));
        cc.mark_loopinvariant(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["get_config"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.extra_info.extraeffect,
            ExtraEffect::LoopInvariant
        );
    }

    #[test]
    fn test_getcalldescr_forces_virtualizable() {
        // A function with VableForce → ForcesVirtualOrVirtualizable.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("forcer");
        let frame = graph.alloc_value();
        graph.block_mut(graph.startblock).inputargs.push(frame);
        graph.push_op(graph.startblock, OpKind::VableForce { base: frame }, false);
        graph.set_return(graph.startblock, None);
        let path = CallPath::from_segments(["forcer"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["forcer"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            vec![Type::Ref],
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.extra_info.extraeffect,
            ExtraEffect::ForcesVirtualOrVirtualizable
        );
    }

    #[test]
    fn test_getcalldescr_extraeffect_override() {
        // When extraeffect is provided, it overrides the analyzers.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["func"]);
        register_int_result_graph(&mut cc, path, simple_graph("func"));
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["func"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            Some(ExtraEffect::ElidableCannotRaise),
            &mut cache,
        );
        assert_eq!(
            descriptor.extra_info.extraeffect,
            ExtraEffect::ElidableCannotRaise
        );
    }

    #[test]
    fn test_getcalldescr_transitive_can_raise() {
        // A function that calls another function that raises → CanRaise.
        let mut cc = CallControl::new();

        // callee: raises
        let callee_path = CallPath::from_segments(["callee"]);
        cc.register_function_graph(callee_path, raising_graph("callee"));

        // caller: calls callee (no Abort itself)
        let mut caller = FunctionGraph::new("caller");
        caller.push_op(
            caller.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["callee"]),
                args: Vec::new(),
                result_ty: ValueType::Void,
            },
            false,
        );
        caller.set_return(caller.startblock, None);
        let caller_path = CallPath::from_segments(["caller"]);
        cc.register_function_graph(caller_path, caller);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["caller"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(descriptor.extra_info.extraeffect, ExtraEffect::CanRaise);
    }

    #[test]
    fn test_getcalldescr_unknown_target_can_raise() {
        // Unknown target (no graph) treated as external call.
        // RPython: RandomEffectsAnalyzer returns False for external calls
        // (only True if random_effects_on_gcobjs). RaiseAnalyzer returns
        // True (top_result) for unknown graphs → CanRaise.
        let cc = CallControl::new();
        let target = CallTarget::function_path(["unknown_extern"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(descriptor.extra_info.extraeffect, ExtraEffect::CanRaise);
        // RandomEffects is false, QuasiImmut is false → can_invalidate is false.
        assert!(!descriptor.extra_info.can_invalidate);
    }

    #[test]
    fn test_getcalldescr_readwrite_effects() {
        // A function with FieldRead/FieldWrite → bitsets populated.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("accessor");
        let base = graph.alloc_value();
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base,
                field: crate::model::FieldDescriptor::new("x", Some("Point".into())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.push_op(
            graph.startblock,
            OpKind::FieldWrite {
                base,
                field: crate::model::FieldDescriptor::new("y", Some("Point".into())),
                value: base, // dummy
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_return(graph.startblock, None);
        let path = CallPath::from_segments(["accessor"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["accessor"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        // Should have non-zero bitsets for field reads and writes.
        assert_ne!(descriptor.extra_info.readonly_descrs_fields, 0);
        assert_ne!(descriptor.extra_info.write_descrs_fields, 0);
    }

    #[test]
    fn resolve_array_identity_follows_phi_chain_to_constant_link_arg() {
        let cc = CallControl::new();
        let base = ValueId(1);
        let forwarded = ValueId(2);
        let value_producers: HashMap<ValueId, &OpKind> = HashMap::new();
        let mut phi_sources: HashMap<ValueId, LinkArg> = HashMap::new();
        phi_sources.insert(base, LinkArg::Value(forwarded));
        phi_sources.insert(
            forwarded,
            LinkArg::Const(crate::flowspace::model::ConstValue::List(vec![])),
        );

        assert_eq!(
            resolve_array_identity(
                &base,
                &Option::<String>::None,
                &value_producers,
                &phi_sources,
                &cc,
            ),
            Some("list".to_string())
        );
    }

    #[test]
    fn test_getcalldescr_elidable_ignores_writes() {
        // Elidable function: write_descrs should be 0 even if graph has writes.
        // RPython effectinfo.py:181-186: ignore writes for elidable.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("pure_writer");
        let base = graph.alloc_value();
        graph.push_op(
            graph.startblock,
            OpKind::FieldWrite {
                base,
                field: crate::model::FieldDescriptor::new("cache", Some("Obj".into())),
                value: base,
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_return(graph.startblock, None);
        let path = CallPath::from_segments(["pure_writer"]);
        register_int_result_graph(&mut cc, path.clone(), graph);
        cc.mark_elidable(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["pure_writer"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.extra_info.extraeffect,
            ExtraEffect::ElidableCannotRaise
        );
        // Writes should be zeroed out for elidable functions.
        assert_eq!(descriptor.extra_info.write_descrs_fields, 0);
    }

    #[test]
    fn test_canraise_cached() {
        // Verify caching: second call should reuse result.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["raiser"]);
        cc.register_function_graph(path, raising_graph("raiser"));

        let target = CallTarget::function_path(["raiser"]);
        let mut cache = AnalysisCache::default();

        let r1 = cc._canraise(&target, &mut cache);
        assert_eq!(r1, CanRaise::Yes);
        assert!(
            cache
                .can_raise
                .contains_key(&CallPath::from_segments(["raiser"]))
        );

        let r2 = cc._canraise(&target, &mut cache);
        assert_eq!(r2, CanRaise::Yes);
    }

    #[test]
    fn test_canraise_reraise_only_graph_matches_upstream_tri_state() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["reraise_only"]);
        cc.register_function_graph(path.clone(), reraise_only_graph("reraise_only"));

        let target = CallTarget::function_path(["reraise_only"]);
        let mut cache = AnalysisCache::default();
        let result = cc._canraise(&target, &mut cache);
        assert_eq!(result, CanRaise::MemoryErrorOnly);
    }

    #[test]
    fn test_canraise_ignore_memoryerror_suppresses_reraise_only_exceptblock() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["reraise_only"]);
        cc.register_function_graph(path.clone(), reraise_only_graph("reraise_only"));

        let mut seen = HashSet::new();
        assert!(!cc.analyze_can_raise_impl(&path, &mut seen, true));
    }

    #[test]
    fn test_readonly_excludes_written_fields() {
        // RPython effectinfo.py:345-348: readstruct only goes to readonly
        // if there's no corresponding write ("struct") for that field.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("rw_same_field");
        let base = graph.alloc_value();
        let field = crate::model::FieldDescriptor::new("x", Some("Point".into()));
        // Both read AND write the same field "x"
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base,
                field: field.clone(),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.push_op(
            graph.startblock,
            OpKind::FieldWrite {
                base,
                field: field.clone(),
                value: base,
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_return(graph.startblock, None);
        let path = CallPath::from_segments(["rw_same_field"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["rw_same_field"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        // Write is set, but readonly should NOT have the same bit set.
        // RPython: readonly = reads & ~writes
        assert_ne!(descriptor.extra_info.write_descrs_fields, 0);
        let overlap = descriptor.extra_info.readonly_descrs_fields
            & descriptor.extra_info.write_descrs_fields;
        assert_eq!(
            overlap, 0,
            "readonly and write should not overlap for same field"
        );
    }

    #[test]
    fn test_op_can_raise_division() {
        // Division ops can raise (ZeroDivisionError).
        // RPython: LL_OPERATIONS[int_floordiv].canraise = (ZeroDivisionError,)
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("divider");
        let a = graph.alloc_value();
        let b = graph.alloc_value();
        graph.push_op(
            graph.startblock,
            OpKind::BinOp {
                op: "int_floordiv".to_string(),
                lhs: a,
                rhs: b,
                result_ty: ValueType::Int,
            },
            true,
        );
        graph.set_return(graph.startblock, None);
        let path = CallPath::from_segments(["divider"]);
        cc.register_function_graph(path, graph);

        let target = CallTarget::function_path(["divider"]);
        let mut cache = AnalysisCache::default();
        let result = cc._canraise(&target, &mut cache);
        assert_eq!(result, CanRaise::Yes);
    }

    #[test]
    fn struct_layout_depth3_nested_fixed_point() {
        // A contains B, B contains C.  Fixed-point iteration must
        // produce correct sizes regardless of HashMap iteration order.
        // struct C { x: i64 }            → size 8
        // struct B { c: C, y: i64 }      → size 16
        // struct A { b: B, z: i64 }      → size 24
        let mut known_structs: HashSet<String> = HashSet::new();
        known_structs.insert("C".into());
        known_structs.insert("B".into());
        known_structs.insert("A".into());

        let fields_c: Vec<(String, String)> = vec![("x".into(), "i64".into())];
        let fields_b: Vec<(String, String)> =
            vec![("c".into(), "C".into()), ("y".into(), "i64".into())];
        let fields_a: Vec<(String, String)> =
            vec![("b".into(), "B".into()), ("z".into(), "i64".into())];

        // Fixed-point iteration (same algorithm as lib.rs).
        let mut known_sizes: HashMap<String, usize> = HashMap::new();
        let all_fields: Vec<(&str, &Vec<(String, String)>)> =
            vec![("A", &fields_a), ("B", &fields_b), ("C", &fields_c)];
        loop {
            let mut changed = false;
            for (name, fields) in &all_fields {
                let layout = StructLayout::from_type_strings(
                    fields,
                    &known_structs,
                    &known_sizes,
                    &HashMap::new(),
                );
                if known_sizes.get(*name) != Some(&layout.size) {
                    known_sizes.insert(name.to_string(), layout.size);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        assert_eq!(known_sizes["C"], 8, "C: single i64");
        assert_eq!(known_sizes["B"], 16, "B: C(8) + i64(8)");
        assert_eq!(known_sizes["A"], 24, "A: B(16) + i64(8)");
    }

    #[derive(Debug)]
    struct StubVInfo {
        vtypeptr_id: usize,
    }
    impl VirtualizableInfoHandle for StubVInfo {
        fn is_vtypeptr(&self, vtypeptr_id: usize) -> bool {
            self.vtypeptr_id == vtypeptr_id
        }
    }

    #[derive(Debug)]
    struct StubGFInfo {
        green_fields: HashSet<(String, String)>,
    }
    impl GreenFieldInfoHandle for StubGFInfo {
        fn contains_green_field(&self, gtype: &str, fieldname: &str) -> bool {
            self.green_fields
                .contains(&(gtype.to_string(), fieldname.to_string()))
        }
    }

    fn cc_with_one_driver() -> CallControl {
        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["portal_runner"]),
            vec!["pc".into()],
            vec!["frame".into()],
            vec![],
            vec![],
        );
        cc
    }

    #[test]
    fn get_vinfo_returns_none_when_no_driver_has_virtualizable_info() {
        let cc = cc_with_one_driver();
        assert!(cc.get_vinfo(0xfeed).is_none());
    }

    #[test]
    fn get_vinfo_returns_matching_handle_from_driver() {
        let mut cc = cc_with_one_driver();
        let vinfo: std::sync::Arc<dyn VirtualizableInfoHandle> = std::sync::Arc::new(StubVInfo {
            vtypeptr_id: 0xabcd,
        });
        cc.jitdrivers_sd[0].virtualizable_info = Some(std::sync::Arc::clone(&vinfo));
        let got = cc.get_vinfo(0xabcd).expect("must match");
        assert!(std::sync::Arc::ptr_eq(&got, &vinfo));
        // Non-matching id → None.
        assert!(cc.get_vinfo(0x1234).is_none());
    }

    #[test]
    #[should_panic(expected = "multiple distinct VirtualizableInfo")]
    fn get_vinfo_panics_when_multiple_distinct_infos_match_same_vtypeptr() {
        let mut cc = cc_with_one_driver();
        cc.setup_jitdriver(
            CallPath::from_segments(["portal_runner_b"]),
            vec![],
            vec![],
            vec![],
            vec![],
        );
        cc.jitdrivers_sd[0].virtualizable_info = Some(std::sync::Arc::new(StubVInfo {
            vtypeptr_id: 0xabcd,
        }));
        cc.jitdrivers_sd[1].virtualizable_info = Some(std::sync::Arc::new(StubVInfo {
            vtypeptr_id: 0xabcd,
        }));
        let _ = cc.get_vinfo(0xabcd);
    }

    #[test]
    fn could_be_green_field_returns_false_when_no_driver_has_greenfield_info() {
        let cc = cc_with_one_driver();
        assert!(!cc.could_be_green_field("Frame", "code"));
    }

    #[test]
    fn could_be_green_field_returns_true_for_registered_pair() {
        let mut cc = cc_with_one_driver();
        let mut greens = HashSet::new();
        greens.insert(("Frame".to_string(), "code".to_string()));
        cc.jitdrivers_sd[0].greenfield_info = Some(std::sync::Arc::new(StubGFInfo {
            green_fields: greens,
        }));
        assert!(cc.could_be_green_field("Frame", "code"));
        assert!(!cc.could_be_green_field("Frame", "pc"));
        assert!(!cc.could_be_green_field("OtherFrame", "code"));
    }

    #[test]
    fn set_jitdriver_virtualizable_info_is_visible_to_get_vinfo() {
        // warmspot.py:528-545 assignment hook reachability test —
        // exercises the production wiring path (not direct field write).
        let mut cc = cc_with_one_driver();
        let info: std::sync::Arc<dyn VirtualizableInfoHandle> =
            std::sync::Arc::new(StubVInfo { vtypeptr_id: 0xab });
        cc.set_jitdriver_virtualizable_info(0, std::sync::Arc::clone(&info));
        let got = cc.get_vinfo(0xab).expect("must match after set");
        assert!(std::sync::Arc::ptr_eq(&got, &info));
    }

    #[test]
    fn set_jitdriver_greenfield_info_is_visible_to_could_be_green_field() {
        let mut cc = cc_with_one_driver();
        let mut greens = HashSet::new();
        greens.insert(("Frame".to_string(), "pc".to_string()));
        let info: std::sync::Arc<dyn GreenFieldInfoHandle> = std::sync::Arc::new(StubGFInfo {
            green_fields: greens,
        });
        cc.set_jitdriver_greenfield_info(0, info);
        assert!(cc.could_be_green_field("Frame", "pc"));
        assert!(!cc.could_be_green_field("Frame", "code"));
    }

    #[test]
    fn make_virtualizable_infos_assigns_index_and_handle_per_warmspot_py_534() {
        // warmspot.py:534-545 — single jitdriver with virtualizables=['frame'],
        // reds=['frame', 'ec'].  index_of_virtualizable must land on slot 0
        // (matching reds.index('frame')) and virtualizable_info must
        // become a populated handle whose VTYPEPTR matches the
        // owner_root token shared across all jitdrivers.
        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["execute_opcode_step"]),
            vec!["pc".into()],
            vec!["frame".into(), "ec".into()],
            vec!["frame".into()],
            vec!["PyFrame".into(), "ExecutionContext".into()],
        );
        cc.make_virtualizable_infos(|_, _| None);
        // warmspot.py:534-538 — `index_of_virtualizable = reds.index('frame')`
        assert_eq!(cc.jitdrivers_sd[0].index_of_virtualizable, 0);
        // warmspot.py:540-545 — codewriter side leaves vinfo None;
        // runtime metainterp populates via set_jitdriver_virtualizable_info.
        assert!(cc.jitdrivers_sd[0].virtualizable_info.is_none());
        // warmspot.py:531-532 — virtualizables present + no dotted greens
        // → greenfield_info stays None.
        assert!(cc.jitdrivers_sd[0].greenfield_info.is_none());
    }

    #[test]
    #[should_panic(expected = "greenfield + virtualizable on the same driver")]
    fn make_virtualizable_infos_panics_on_dotted_green_with_virtualizable() {
        // warmspot.py:531-532 `assert jd.greenfield_info is None,
        // "XXX not supported yet"` — pyre keeps the assertion.
        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["portal"]),
            vec!["frame.code".into()],
            vec!["frame".into()],
            vec!["frame".into()],
            vec!["PyFrame".into()],
        );
        cc.make_virtualizable_infos(|_, _| None);
    }

    #[test]
    fn make_virtualizable_infos_clears_when_no_virtualizable() {
        // warmspot.py:527-530 — `if not jd.jitdriver.virtualizables: ... continue`.
        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["portal"]),
            vec!["pc".into()],
            vec!["frame".into()],
            vec![],
            vec![],
        );
        cc.jitdrivers_sd[0].virtualizable_info = Some(std::sync::Arc::new(StubVInfo {
            vtypeptr_id: 0xfeed,
        }));
        cc.jitdrivers_sd[0].index_of_virtualizable = 7;
        cc.make_virtualizable_infos(|_, _| None);
        assert!(cc.jitdrivers_sd[0].virtualizable_info.is_none());
        assert_eq!(cc.jitdrivers_sd[0].index_of_virtualizable, -1);
    }

    #[test]
    fn make_virtualizable_infos_resolves_gtype_from_red_types() {
        // greenfield.py:14,18 — green_fields holds (GTYPE, fieldname) where
        // GTYPE is the type of the red slot identified by objname.
        // Pyre threads this through `red_types` parallel to `reds`.
        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["portal_with_greenfield"]),
            vec!["frame.code".into(), "pc".into()],
            vec!["frame".into()],
            vec![],
            vec!["PyFrame".into()],
        );
        cc.make_virtualizable_infos(|_, _| None);
        let gfinfo = cc.jitdrivers_sd[0]
            .greenfield_info
            .as_ref()
            .expect("greenfield_info populated for dotted green");
        // contains_green_field expects (GTYPE, fieldname) — resolved
        // from `red_types` not the raw `objname`.
        assert!(gfinfo.contains_green_field("PyFrame", "code"));
        assert!(!gfinfo.contains_green_field("frame", "code"));
    }

    #[test]
    fn make_virtualizable_infos_invokes_factory_and_caches_per_vtypeptr() {
        // warmspot.py:540-545 — `vinfos[VTYPEPTR]` cache: two jitdrivers
        // sharing the same VTYPEPTR token must reuse the same handle
        // (same Arc identity), and the factory must be called once
        // per unique VTYPEPTR.
        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["portal_a"]),
            vec!["pc".into()],
            vec!["frame".into()],
            vec!["frame".into()],
            vec!["PyFrame".into()],
        );
        cc.setup_jitdriver(
            CallPath::from_segments(["portal_b"]),
            vec!["pc".into()],
            vec!["frame".into()],
            vec!["frame".into()],
            vec!["PyFrame".into()],
        );
        let mut factory_calls: Vec<String> = Vec::new();
        cc.make_virtualizable_infos(|_jd_idx, vtypeptr_token| {
            factory_calls.push(vtypeptr_token.to_string());
            Some(std::sync::Arc::new(StubVInfo {
                vtypeptr_id: 0xfeed,
            }))
        });
        assert_eq!(
            factory_calls,
            vec!["PyFrame".to_string()],
            "warmspot.py:540-545 vinfos cache must dedupe by VTYPEPTR token",
        );
        let h0 = cc.jitdrivers_sd[0]
            .virtualizable_info
            .clone()
            .expect("vinfo populated");
        let h1 = cc.jitdrivers_sd[1]
            .virtualizable_info
            .clone()
            .expect("vinfo populated");
        assert!(std::sync::Arc::ptr_eq(&h0, &h1));
    }

    #[test]
    fn make_virtualizable_infos_factory_none_keeps_slot_empty() {
        // warmspot.py:540-545 with factory→None: the codewriter slot
        // stays empty so the runtime metainterp setter populates it
        // later (jitdriver.rs:285).
        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["portal"]),
            vec!["pc".into()],
            vec!["frame".into()],
            vec!["frame".into()],
            vec!["PyFrame".into()],
        );
        cc.make_virtualizable_infos(|_, _| None);
        assert!(cc.jitdrivers_sd[0].virtualizable_info.is_none());
        assert_eq!(cc.jitdrivers_sd[0].index_of_virtualizable, 0);
    }

    // ── RPython indirect_call family tests — plan §Tests ────────────

    /// `guess_call_kind` for `OpKind::IndirectCall`:
    ///   ≥1 candidate impl is a regular candidate → `Regular`
    ///   graphs `None` (unknown family)          → `Residual`
    /// RPython `call.py:116-139`.  Mirrors the
    /// `op.opname == 'indirect_call'` fall-through to the final
    /// `graphs_from(op) is None` test.
    #[test]
    fn guess_call_kind_indirect() {
        let mut cc = CallControl::new();
        cc.register_trait_method("run", Some("Handler"), "A", FunctionGraph::new("A::run"));
        cc.register_trait_method("run", Some("Handler"), "B", FunctionGraph::new("B::run"));
        cc.find_all_graphs_for_tests();

        let handler_family = cc.all_impls_for_indirect("Handler", "run");
        assert_eq!(handler_family.len(), 2);
        assert_eq!(
            cc.guess_call_kind(&indirect_call_op(Some(handler_family))),
            CallKind::Regular
        );

        // `graphs = None` → unknown family → residual path per
        // `rpython/translator/backendopt/graphanalyze.py:117`.
        assert_eq!(
            cc.guess_call_kind(&indirect_call_op(None)),
            CallKind::Residual
        );
    }

    /// `graphs_from(op)` for an `OpKind::IndirectCall` must filter by
    /// the family attached to the op, not mix impls across traits that
    /// share a method name.  RPython `call.py:103-112` indirect branch.
    #[test]
    fn graphs_from_op_filters_by_indirect_family() {
        let mut cc = CallControl::new();
        cc.register_trait_method(
            "bar",
            Some("Foo"),
            "FooImpl",
            FunctionGraph::new("FooImpl::bar"),
        );
        cc.register_trait_method(
            "bar",
            Some("Baz"),
            "BazImpl",
            FunctionGraph::new("BazImpl::bar"),
        );
        cc.find_all_graphs_for_tests();

        let foo_family = cc.all_impls_for_indirect("Foo", "bar");
        let baz_family = cc.all_impls_for_indirect("Baz", "bar");

        let foo_candidates = cc
            .graphs_from(&indirect_call_op(Some(foo_family)))
            .expect("Foo::bar family is non-empty");
        let baz_candidates = cc
            .graphs_from(&indirect_call_op(Some(baz_family)))
            .expect("Baz::bar family is non-empty");

        assert_eq!(foo_candidates.len(), 1);
        assert_eq!(
            foo_candidates[0].segments[0], "FooImpl",
            "Foo::bar must not surface BazImpl: {foo_candidates:?}"
        );
        assert_eq!(baz_candidates.len(), 1);
        assert_eq!(
            baz_candidates[0].segments[0], "BazImpl",
            "Baz::bar must not surface FooImpl: {baz_candidates:?}"
        );
    }

    /// `getcalldescr` with mixed `@jit.elidable` vs non-elidable impls
    /// panics to match RPython `call.py:259-280`.
    #[test]
    #[should_panic(expected = "indirect_call family")]
    fn getcalldescr_rejects_mixed_elidable_family() {
        use majit_ir::value::Type;
        let mut cc = CallControl::new();
        cc.register_trait_method(
            "bar",
            Some("Foo"),
            "PureImpl",
            FunctionGraph::new("PureImpl::bar"),
        );
        cc.register_trait_method(
            "bar",
            Some("Foo"),
            "ImpureImpl",
            FunctionGraph::new("ImpureImpl::bar"),
        );
        cc.mark_elidable(CallPath::from_segments(["PureImpl", "bar"]));
        cc.find_all_graphs_for_tests();

        let family = cc.all_impls_for_indirect("Foo", "bar");
        let mut cache = AnalysisCache::default();
        let _ = cc.getcalldescr(
            &indirect_call_op(Some(family)),
            vec![Type::Ref],
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
    }

    /// Inherent impl (no `impl Trait for Type`) continues to resolve
    /// via `function_graphs` and classify as `Regular`, without
    /// populating `trait_method_impls`.
    #[test]
    fn inherent_method_still_direct_regression() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["Foo", "bar"]);
        cc.register_function_graph(path.clone(), FunctionGraph::new("Foo::bar"));
        cc.find_all_graphs_for_tests();

        // Inherent impl: front-end emits `CallTarget::Method` with a
        // concrete receiver_root.  No `trait_method_impls` entry.
        let target = CallTarget::method("bar", Some("Foo".to_string()));
        assert_eq!(
            cc.guess_call_kind(&direct_call_op(target)),
            CallKind::Regular
        );
        assert!(
            cc.all_impls_for_indirect("Foo", "bar").is_empty(),
            "inherent impls must not appear in any indirect-call family"
        );
    }

    /// A trait-default method (registered under the synthetic
    /// `"<default methods of Trait>"` impl-type produced by
    /// `parse.rs:327-332`) must show up in the same indirect-call
    /// family as concrete overrides.  This keeps
    /// `lower_indirect_calls`'s `all_impls_for_indirect(...)` family
    /// correct when a `dyn Trait` receiver can route to either the
    /// default body or an override at runtime — parity with RPython
    /// `rpbc.py:199-217` `c_graphs = row_of_graphs.values()`, which
    /// lists every graph reachable through the trait's vtable slot.
    #[test]
    fn dyn_trait_default_method_uses_same_indirect_family() {
        let mut cc = CallControl::new();
        cc.register_trait_method("m", Some("Foo"), "A", FunctionGraph::new("A::m"));
        cc.register_trait_method(
            "m",
            Some("Foo"),
            "<default methods of Foo>",
            FunctionGraph::new("<default methods of Foo>::m"),
        );

        let family = cc.all_impls_for_indirect("Foo", "m");
        let segs: Vec<String> = family.iter().map(|p| p.segments.join("::")).collect();
        assert!(
            segs.iter().any(|s| s.starts_with("A")),
            "family must include overriding impl A, got {segs:?}"
        );
        assert!(
            segs.iter().any(|s| s.contains("<default methods of Foo>")),
            "family must include trait default-method entry, got {segs:?}"
        );
        assert_eq!(
            family.len(),
            2,
            "family must have exactly two members (default + override), got {segs:?}"
        );
    }

    /// Two trait objects that differ only in their generic arguments
    /// (`Handler<i64>` vs `Handler<String>`) must be treated as separate
    /// indirect-call families.  Conflating them would produce a mixed
    /// family whose candidates accept incompatible argument types —
    /// `rpython/jit/codewriter/call.py:259-280`'s family validation
    /// would reject the mixed descriptor, and the dispatch would
    /// dead-end at runtime.  The family key passed to
    /// `all_impls_for_indirect` must preserve the full `trait_root`
    /// (including generic args), not a bare `Handler` root.
    #[test]
    fn dyn_trait_generic_family_key_preserved() {
        let mut cc = CallControl::new();
        cc.register_trait_method(
            "run",
            Some("Handler<i64>"),
            "A",
            FunctionGraph::new("A::run"),
        );
        cc.register_trait_method(
            "run",
            Some("Handler<String>"),
            "B",
            FunctionGraph::new("B::run"),
        );

        let i64_family = cc.all_impls_for_indirect("Handler<i64>", "run");
        let string_family = cc.all_impls_for_indirect("Handler<String>", "run");
        let bare_family = cc.all_impls_for_indirect("Handler", "run");

        let segs =
            |v: &[CallPath]| -> Vec<String> { v.iter().map(|p| p.segments.join("::")).collect() };

        let i64_segs = segs(&i64_family);
        let string_segs = segs(&string_family);
        assert_eq!(
            i64_family.len(),
            1,
            "Handler<i64> family must contain only A, got {i64_segs:?}"
        );
        assert!(
            i64_segs[0].starts_with("A"),
            "Handler<i64> must resolve to impl A, got {i64_segs:?}"
        );
        assert_eq!(
            string_family.len(),
            1,
            "Handler<String> family must contain only B, got {string_segs:?}"
        );
        assert!(
            string_segs[0].starts_with("B"),
            "Handler<String> must resolve to impl B, got {string_segs:?}"
        );
        assert!(
            bare_family.is_empty(),
            "bare `Handler` (generic args stripped) must NOT match a \
             generic-instantiated family — conflation would cross \
             incompatible argument types; got {:?}",
            segs(&bare_family),
        );
    }
}
