//! Line-by-line port of `rpython/jit/codewriter/flatten.py` lines 1-60.
//!
//! Covers the data structures the rest of the codewriter pipeline
//! (`liveness.py`, `assembler.py`) consumes: `SSARepr`, `Label`, `TLabel`,
//! `Register`, `ListOfKind`, `IndirectCallTargets`, and the `KINDS`
//! constant. The `GraphFlattener` driver at `flatten.py:60-350` has no
//! 1:1 analog in pyre because pyre's input is a CPython `CodeObject`
//! rather than an RPython `FunctionGraph`; the equivalent walker lives
//! in `codewriter.rs` and produces an `SSARepr` whose `insns` contents
//! follow the shapes declared here.
//!
//! The `Insn` enum and `Operand` enum do not exist as separate classes
//! in RPython — `ssarepr.insns` is a list of heterogeneous Python tuples.
//! Rust needs a typed representation, so the tuple shape is captured by
//! `Insn` (with cases for the well-known string markers `Label`, `-live-`,
//! `---` and generic `Op` instructions) plus `Operand` for everything
//! that appears inside a tuple.

use std::rc::Rc;

use majit_ir::Descr;
use majit_translate::jit_codewriter::flatten::reorder_renaming_list;
use majit_translate::jitcode::BhDescr;

use super::flow::{
    BlockRef, Constant, ConstantValue, FlowValue, LinkRef, SpaceOperation, SpaceOperationArg,
    Variable,
};

/// `rpython/jit/codewriter/flatten.py:59` `KINDS = ['int', 'ref', 'float']`.
///
/// RPython stores register kinds as strings; the `Kind` enum is the Rust
/// analog. The `as_str` method yields the exact RPython string so callers
/// that stringify ("int"/"ref"/"float") continue to behave identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Int,
    Ref,
    Float,
}

/// Type-level marker for a fresh `Variable` produced by
/// `SSARepr::fresh_var`. RPython's `flowspace/model.py:Variable()` ctor
/// returns object identity; pyre's codewriter walker emits against
/// pre-regalloc indices, so a u16 stands in for the identity. Wrapping
/// it in `VariableId` keeps the index/color distinction visible in
/// types — Phase 1 minimal slice (plan staged-sauteeing-koala). Until
/// later phases can fold the index into the post-regalloc color
/// derivation, callers extract the raw `u16` via `.0` at the consumer
/// boundary (`Register::new(Kind, u16)`, `Operand::reg(Kind, u16)`,
/// `JitCallArg::int/reference(u16)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VariableId(pub u16);

impl From<VariableId> for u16 {
    fn from(v: VariableId) -> u16 {
        v.0
    }
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Int, Kind::Ref, Kind::Float];

    /// Yields the RPython string for this kind ("int", "ref", "float").
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Int => "int",
            Kind::Ref => "ref",
            Kind::Float => "float",
        }
    }

    /// First character of the kind name ("i", "r", "f") — matches
    /// `self.kind[0]` in RPython `Register.__repr__` and the `argcodes`
    /// alphabet used by `assembler.py`.
    pub fn first_char(self) -> char {
        match self {
            Kind::Int => 'i',
            Kind::Ref => 'r',
            Kind::Float => 'f',
        }
    }

    /// Dense slot index `0..3` for indexing `[T; 3]` arrays keyed by
    /// `Kind`.  Pyre uses `[T; 3]` rather than `HashMap<Kind, T>` per
    /// [[feedback-no-hashmap-ever]] — the RPython `regallocs` dict has
    /// statically-known keys (`KINDS = ['int', 'ref', 'float']`) so the
    /// Rust analog is position-indexed not hash-keyed.
    pub fn index(self) -> usize {
        self as usize
    }
}

/// `flatten.py:6-10` `class SSARepr(object)`.
///
/// Python:
/// ```py
/// class SSARepr(object):
///     def __init__(self, name):
///         self.name = name
///         self.insns = []
///         self._insns_pos = None     # after being assembled
/// ```
#[derive(Debug, Clone)]
pub struct SSARepr {
    pub name: String,
    pub insns: Vec<Insn>,
    /// `flatten.py:10` `self._insns_pos = None # after being assembled`.
    /// `assembler.py:41` populates this with the byte position of each
    /// instruction after `assemble()`.
    pub insns_pos: Option<Vec<usize>>,
    /// Phase 2.2a (plan staged-sauteeing-koala, Tasks #158/#159/#122
    /// epic): per-kind fresh-Variable counter. RPython has no analog
    /// because RPython's `Variable()` constructor produces objects with
    /// implicit identity and `regalloc.py` numbers them densely after
    /// the FunctionGraph is final. Pyre's codewriter walks a CodeObject
    /// directly (no FunctionGraph + jtransform layer), so each fresh
    /// scratch-temp Variable needs an explicit u16 index at emit time.
    /// `fresh_var(kind, base)` returns and bumps the counter, ensuring
    /// scratches occupy indices distinct from Python locals/stack and
    /// from any hardcoded scratch slots still living in
    /// `RegisterLayout`. Once Phase 2.2 fully migrates the dispatcher,
    /// the counter becomes the sole source of scratch indices and
    /// `RegisterLayout`'s scratch fields can be retired.
    next_var_idx: [u16; 3],
}

impl SSARepr {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            insns: Vec::new(),
            insns_pos: None,
            next_var_idx: [0; 3],
        }
    }

    /// Allocate a fresh `(kind, index)` Variable for this SSARepr.
    ///
    /// `base` is the minimum index — the counter clamps up to `base` on
    /// first call, so callers can reserve a low index range for
    /// inputargs (`0..nlocals`) and stack slots while still getting a
    /// unique scratch index above. Each subsequent `fresh_var` for the
    /// same kind returns a strictly larger index. The returned index is
    /// safe to use directly in `Register::new(kind, idx)` /
    /// `Operand::reg(kind, idx)` without further bookkeeping —
    /// `regalloc::allocate_registers` will pick it up via the standard
    /// SSARepr scan and color it.
    pub fn fresh_var(&mut self, kind: Kind, base: u16) -> VariableId {
        let slot = match kind {
            Kind::Int => 0,
            Kind::Ref => 1,
            Kind::Float => 2,
        };
        let counter = &mut self.next_var_idx[slot];
        if *counter < base {
            *counter = base;
        }
        let idx = *counter;
        *counter += 1;
        VariableId(idx)
    }
}

/// `flatten.py:12-18` `class Label(object)`.
///
/// Python:
/// ```py
/// class Label(object):
///     def __init__(self, name):
///         self.name = name
///     def __repr__(self):
///         return "Label(%r)" % (self.name, )
///     def __eq__(self, other):
///         return isinstance(other, Label) and other.name == self.name
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Label {
    pub name: String,
}

impl Label {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// `flatten.py:20-26` `class TLabel(object)`.
///
/// Python:
/// ```py
/// class TLabel(object):
///     def __init__(self, name):
///         self.name = name
///     def __repr__(self):
///         return "TLabel(%r)" % (self.name, )
///     def __eq__(self, other):
///         return isinstance(other, TLabel) and other.name == self.name
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TLabel {
    pub name: String,
}

impl TLabel {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// pyre-only naming convention for the per-Python-PC anchor label that
/// the walker emits at every PC entry.  Upstream RPython's `flatten.py`
/// emits exactly one `Label(block)` per SpamBlock; the JIT resumes only
/// at `jit_merge_point` boundaries (loop headers + guard fail bounce
/// points) so per-PC anchors are unnecessary.  Pyre's runtime instead
/// resumes at arbitrary Python PCs (blackhole bridge-resume, inline
/// call tracing), which requires a per-PC byte-offset table populated
/// from anchor positions in the assembled JitCode.
///
/// All five emit / branch sites that materialise the `pc{N}` naming
/// route through these helpers so the convention is single-sourced.
/// A future Phase 4 step can rename the marker shape (e.g.,
/// `Insn::PcAnchor { py_pc }` distinct from `Insn::Label`) by changing
/// only these two functions plus the runtime consumers
/// (`pc_anchor_positions` / `live_marker_indices_by_pc`).
pub fn pc_label_name(py_pc: usize) -> String {
    format!("pc{py_pc}")
}

/// Companion to [`pc_label_name`] producing the matching `TLabel`
/// branch target.  Walker `goto Label("pc{N}")` callsites build the
/// target via this helper.
pub fn pc_tlabel(py_pc: usize) -> TLabel {
    TLabel::new(pc_label_name(py_pc))
}

/// Recover the Python PC index from a `Insn::PcAnchor`, or `None`
/// for any other Insn shape.  Consumed by the runtime's
/// `pc_anchor_positions` / `live_marker_indices_by_pc` scans.
///
/// `Insn::Label("pc{N}")` was the transitional shape before the
/// `Insn::PcAnchor` variant landed; every walker emit site and every
/// in-tree test fixture now produces `Insn::PcAnchor` directly, so the
/// recognition path is retired and `Insn::Label` is reserved purely
/// for upstream-orthodox block / link / catch-landing labels.
pub fn label_pc_index(insn: &Insn) -> Option<usize> {
    if let Insn::PcAnchor { py_pc } = insn {
        Some(*py_pc)
    } else {
        None
    }
}

/// `flatten.py:28-33` `class Register(object)`.
///
/// Python:
/// ```py
/// class Register(object):
///     def __init__(self, kind, index):
///         self.kind = kind          # 'int', 'ref' or 'float'
///         self.index = index
///     def __repr__(self):
///         return "%%%s%d" % (self.kind[0], self.index)
/// ```
///
/// RPython uses Python ints for `index`; pyre uses `u16` because pyre's
/// register allocator can exceed 255 in unusual cases (see
/// `liveness_regs_to_u8_sorted` in `codewriter.rs`). The RPython-orthodox
/// assertion `index < 256` is enforced at `assembler.emit_reg` time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Register {
    pub kind: Kind,
    pub index: u16,
}

impl Register {
    pub fn new(kind: Kind, index: u16) -> Self {
        Self { kind, index }
    }
}

/// `flatten.py:35-51` `class ListOfKind(object)`.
///
/// Python:
/// ```py
/// class ListOfKind(object):
///     # a list of Regs/Consts, all of the same 'kind'.
///     # We cannot use a plain list, because we wouldn't know what 'kind' of
///     # Regs/Consts would be expected in case the list is empty.
///     def __init__(self, kind, content):
///         assert kind in KINDS
///         self.kind = kind
///         self.content = tuple(content)
/// ```
#[derive(Debug, Clone)]
pub struct ListOfKind {
    pub kind: Kind,
    pub content: Vec<Operand>,
}

impl ListOfKind {
    pub fn new(kind: Kind, content: Vec<Operand>) -> Self {
        Self { kind, content }
    }

    /// `flatten.py:47` `def __nonzero__(self): return bool(self.content)`.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// `flatten.py:45-46` `def __iter__(self): return iter(self.content)`.
    pub fn iter(&self) -> std::slice::Iter<'_, Operand> {
        self.content.iter()
    }
}

/// `flatten.py:53-57` `class IndirectCallTargets(object)`.
///
/// Python:
/// ```py
/// class IndirectCallTargets(object):
///     def __init__(self, lst):
///         self.lst = lst       # list of JitCodes
/// ```
///
/// The list carries `JitCode` references (RPython `list of JitCodes`, in
/// which each entry is a Python-object reference shared with other call
/// sites and with the assembler's `indirectcalltargets` set).  pyre still
/// stores the runtime adapter `majit_metainterp::jitcode::JitCode` here;
/// `Arc` preserves the shared-reference semantics of the Python object
/// reference, but the canonical codewriter `majit_translate::jitcode::JitCode`
/// has not reached this runtime path yet.
#[derive(Debug, Clone, Default)]
pub struct IndirectCallTargets {
    pub lst: Vec<std::sync::Arc<majit_metainterp::jitcode::JitCode>>,
}

/// `rpython/jit/codewriter/jitcode.py:131-143` `class SwitchDictDescr`
/// as populated by `flatten.py:282-298`.
///
/// Python:
/// ```py
/// from rpython.jit.codewriter.jitcode import SwitchDictDescr
/// switchdict = SwitchDictDescr()
/// switchdict._labels = []
/// ...
/// for switch in switches:
///     key = lltype.cast_primitive(lltype.Signed, switch.llexitcase)
///     switchdict._labels.append((key, TLabel(switch)))
/// ```
///
/// The SSARepr-side descr carries `_labels` because the liveness pass
/// (`liveness.py:76-78`) iterates this list to follow switch-target
/// edges. RPython's runtime `SwitchDictDescr.dict` is set later by
/// `Assembler.fix_labels` (`assembler.py:258-263`, via `attach`); pyre
/// lowers the SSARepr-side `SwitchDictDescr` into the runtime
/// `BhDescr::SwitchDict` at assemble time.
#[derive(Debug, Clone, Default)]
pub struct SwitchDictDescr {
    /// `flatten.py:284,298` `switchdict._labels.append((key, TLabel(...)))`.
    pub labels: Vec<(i64, TLabel)>,
}

impl SwitchDictDescr {
    pub fn new() -> Self {
        Self { labels: Vec::new() }
    }
}

/// Descr operand shape visible inside an `SSARepr`.
///
/// RPython's `assembler.py:197-206` handles both regular descrs (runtime
/// `AbstractDescr`) and the not-yet-attached `SwitchDictDescr` carried as
/// `isinstance(x, SwitchDictDescr)` (checked at `liveness.py:76`). The
/// `DescrOperand` enum preserves that distinction so liveness sees
/// `_labels` and the assembler sees a finalised runtime descr.
#[derive(Debug, Clone)]
pub enum DescrOperand {
    /// Runtime descr already materialised as `BhDescr`.
    Bh(BhDescr),
    /// SSARepr-side `SwitchDictDescr` before `attach()`; liveness reads
    /// its `labels` field to follow control-flow edges.
    SwitchDict(SwitchDictDescr),
    /// `rpython/jit/codewriter/jtransform.py:414-435 rewrite_call` appends
    /// an `AbstractDescr` (the `calldescr`) at the end of every
    /// `residual_call_*` / `inline_call_*` arg list. The descr carries
    /// `EffectInfo` that downstream (`rpython/jit/metainterp/optimizeopt/
    /// rewrite.py`) consults to pick between `call_may_force_*`,
    /// `call_release_gil_*`, `call_loopinvariant_*`, `call_pure_*`, and
    /// `call_assembler_*`, plus `arg_types` that `bh_call_*` reads to
    /// reconstruct the C-function parameter order.
    ///
    /// pyre does not (yet) thread `EffectInfo` through the codewriter
    /// layer, so this variant stands in for the calldescr and carries the
    /// flavor plus per-arg kind sequence directly. The assembler dispatch
    /// consumes both: flavor picks the same builder method the optimizeopt
    /// layer would have selected; `arg_kinds` lets `dispatch_op` re-
    /// interleave the kind-separated `ListOfKind` sublists into pyre's
    /// flat `&[JitCallArg]` call order (pyre helpers have varied param
    /// sequences — `ref,int`, `int,ref,ref`, `ref,ref,int` etc.). SSARepr
    /// shape still matches upstream 1:1: one descr operand per residual
    /// call, final argument position.
    CallDescrStub(CallDescrStub),
    /// `rpython/jit/metainterp/virtualizable.py:73` `VirtualizableInfo
    /// .array_field_descrs[i]` — the `FieldDescr` for the frame field
    /// holding a virtualizable array's pointer.  RPython
    /// `jtransform.py:1882-1885 do_fixed_list_getitem` and `:1898-1906
    /// do_fixed_list_setitem` emit it as the second-to-last operand of
    /// `getarrayitem_vable_X` / `setarrayitem_vable_X` and as one of two
    /// trailing descrs on `arraylen_vable`.  pyre stores the per-array
    /// index (today always 0 — pyre's `PyFrame` has a single virtualizable
    /// array, `locals_cells_stack_w`); assembler dispatch turns it into
    /// a canonical `BhDescr::VableArray` descriptor.
    VableArrayField(u16),
    /// `rpython/jit/metainterp/virtualizable.py:58` `VirtualizableInfo
    /// .array_descrs[i]` — the `ArrayDescr` for the GcArray that the
    /// `array_field_descr` field points at.  Always paired with a
    /// `VableArrayField(i)` operand at `i+1` in upstream's argv;
    /// `assembler.py:80-138 emit_const` uses both to encode the per-op
    /// bytecode.  pyre carries it as a distinct SSARepr descriptor and the
    /// assembler emits a second `d` operand for the array descriptor.
    VableArray(u16),
    /// `rpython/jit/metainterp/virtualizable.py:71` `VirtualizableInfo
    /// .static_field_descrs[i]` — the `FieldDescr` for the i-th scalar
    /// (non-array) field of the virtualizable struct. RPython
    /// `jtransform.py:846` (getfield) emits it as the trailing descr
    /// operand of `getfield_vable_<kind>` after `v_inst`;
    /// `jtransform.py:927` (setfield) emits it after `v_inst, v_value`
    /// on `setfield_vable_<kind>`. pyre stores the per-field index and the
    /// assembler turns it into a canonical `BhDescr::VableField` descriptor.
    VableStaticField(u16),
}

/// Pyre-local stand-in for `rpython/jit/codewriter/effectinfo.py
/// AbstractDescr` on a residual call. Upstream's calldescr carries both
/// the flavor (EffectInfo) and the ordered arg-types used by `bh_call_*`
/// to rebuild the C-function parameter list from `args_i` / `args_r` /
/// `args_f` pools (`rpython/jit/backend/llsupport/llmodel.py:816-839
/// bh_call_*` + `calldescr.call_stub_*`). pyre needs both pieces at
/// dispatch time.
///
/// Slice 1 of the EffectInfo wire-up epic: `effect_info` carries the
/// upstream-shape `EffectInfo` derived from the producer's
/// [`CallFlavor`] at emit time via [`effect_info_for_call_flavor`].
/// Slice 3 flipped `dispatch_residual_call` to derive its branch from
/// [`dispatch_kind_for_effect_info`] applied to `effect_info`. Slice 3b
/// dropped the redundant `flavor` field: codewriter sites still take a
/// `CallFlavor` parameter as construction-site shorthand, but the stub
/// stores only the canonical `EffectInfo` form. `arg_kinds` stays for
/// the per-arg `JitCallArg` reassembly until upstream `descr.arg_types()`
/// (`majit-ir/src/descr.rs::SimpleCallDescr.arg_types`) becomes the
/// canonical source — that flip waits until pyre's residual_call SSARepr
/// trailing slot stores `Arc<SimpleCallDescr>` in place of this stub.
#[derive(Debug, Clone)]
pub struct CallDescrStub {
    /// Upstream-shape `EffectInfo` — the canonical dispatch source read
    /// by `dispatch_residual_call` via [`dispatch_kind_for_effect_info`]
    /// (`pyre/pyre-jit/src/jit/assembler.rs:1437`).
    pub effect_info: majit_ir::EffectInfo,
    /// Per-arg kind sequence in C-function parameter order. Exact length
    /// equals the sum of the int/ref/float `ListOfKind` sublists for the
    /// same residual_call Insn.
    pub arg_kinds: Vec<Kind>,
    /// `descr.py:665` carries `result_type` on both the cache key and
    /// the constructed `CallDescr`. Pyre mirrors that redundancy so
    /// `dispatch_residual_call` (`assembler.rs:1370`) can cross-check
    /// the descr-side answer against the opname-tail-derived `ResKind`
    /// it would have computed independently — RPython's invariant is
    /// that the two MUST agree per `descr.create_call_stub` /
    /// `descr.result_type` round-trip in `descr.py:670-674`.
    pub result_kind: Option<Kind>,
}

/// Make [`CallDescrStub`] addressable through `majit_ir::DescrRef` so the
/// graph layer can carry it as a `flow::SpaceOperationArg::Descr` arg
/// when graph-side `residual_call_*` recorders land (Task #42).  Other
/// `Descr` trait methods take their default values — the stub is not
/// indexed in the `JitCode.descrs` table, has no fail/size/array
/// downcast, and only needs to be downcast-recognisable from
/// `flatten_descr_by_ptr` via `as_any`.
impl Descr for CallDescrStub {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn repr(&self) -> String {
        format!(
            "CallDescrStub(ei={:?}, kinds={:?}, result={:?})",
            self.effect_info.extraeffect, self.arg_kinds, self.result_kind,
        )
    }
}

/// Intern a [`CallDescrStub`] by `(effect_info, arg_kinds, result_kind)`
/// and return the shared `Arc` upcast to `majit_ir::DescrRef` so it can
/// sit inside `flow::SpaceOperationArg::Descr(DescrByPtr(_))`.  The
/// first call for a given key allocates; subsequent calls return the
/// cached `Arc`.
///
/// `result_kind` is `Some(Kind)` for typed-result residual_call shapes
/// and `None` for the void-result `residual_call_*_v` form.
///
/// The cache lives on the active [`super::codewriter::CodeWriter`]
/// instance (per-instance like RPython's
/// `gc_ll_descr.gc_cache._cache_call`, `backend/llsupport/descr.py:14`),
/// reached via the thread-local `CodeWriter::instance()` singleton.
/// Each entry shares its `Arc` across the inline SSARepr emitter
/// (`codewriter::emit_residual_call_shape`) and the graph-side
/// `record_residual_call_graph_op` so a single allocation backs both
/// layers per call signature.  RPython parity table:
///
/// | RPython slot       | pyre slot                                |
/// |--------------------|------------------------------------------|
/// | `arg_classes`      | `Vec<Kind>`                              |
/// | `result_type`      | `Option<Kind>` (None = void)             |
/// | `result_signed`    | implicit `true` for `Kind::Int`          |
/// | `RESULT_ERASED`    | implicit `true` for `Kind::Ref` (gcref)  |
/// | `extrainfo`        | `EffectInfo`                             |
///
/// `EffectInfo`'s manual `Hash` (`majit-ir/src/effectinfo.rs`) skips the
/// same fields its `PartialEq` skips (`single_write_descr_array`,
/// `extradescrs`) so the cache key stays stable for stub-flavor inputs.
pub fn intern_call_descr_stub(
    effect_info: majit_ir::EffectInfo,
    arg_kinds: Vec<Kind>,
    result_kind: Option<Kind>,
) -> majit_ir::DescrRef {
    super::codewriter::CodeWriter::instance().intern_call_descr_stub(
        effect_info,
        arg_kinds,
        result_kind,
    )
}

/// Map a [`CallFlavor`] to the upstream-shape `EffectInfo` carrying the
/// equivalent `extraeffect`. The mapping mirrors
/// `rpython/jit/metainterp/optimizeopt/rewrite.py optimize_CALL_*`'s
/// branch selection: each pyre `CallFlavor` corresponds to the
/// `EffectInfo.extraeffect` value upstream's optimizer would have read
/// off the calldescr to pick the same `call_*_*` rewrite.
///
/// Slice 1 of the EffectInfo wire-up epic. `effect_info_for_call_flavor`
/// is the foundation for future slices that flip dispatch consumers
/// (assembler / blackhole / trace recorder) from reading `flavor`
/// directly to reading `effect_info.extraeffect`.
///
/// **Stub limitations.**
/// RPython `call.py:296-326 getcalldescr` constructs the EffectInfo
/// from four static analyzers run over the callee graph:
///
/// | EI field                       | RPython source                        |
/// |--------------------------------|---------------------------------------|
/// | `oopspecindex`                 | `jtransform.py:_handle_oopspec_call`  |
/// | `readonly_descrs_*` (bitsets)  | `readwrite_analyzer.analyze(op, ...)` |
/// | `write_descrs_*` (bitsets)     | `readwrite_analyzer.analyze(op, ...)` |
/// | `can_invalidate`               | `quasiimmut_analyzer.analyze(op)` OR  |
/// |                                | `randomeffects_analyzer.analyze(op)`  |
/// | `extraeffect` (elidable 3-way) | `_canraise(op)` (call.py:294-299)     |
/// | `call_release_gil_target`      | `_call_aroundstate_target_` decorator |
/// | `extradescrs`                  | `_jit_oopspec_extra_` decorator       |
/// | `can_collect`                  | `collect_analyzer.analyze(op)`        |
///
/// All six analyzers + the public `getcalldescr` are ported in
/// `majit-translate/src/jit_codewriter/call.rs`:
///
/// | Analyzer                | Pyre site                              |
/// |-------------------------|----------------------------------------|
/// | `RaiseAnalyzer`         | `analyze_can_raise_impl` (call.rs:2271)|
/// | `VirtualizableAnalyzer` | `analyze_forces_virtualizable` (:2341) |
/// | `RandomEffectsAnalyzer` | `analyze_random_effects` (:2401)       |
/// | `QuasiImmutAnalyzer`    | `analyze_can_invalidate` (:2452)       |
/// | `CollectAnalyzer`       | `analyze_can_collect` (:2505)          |
/// | `ReadWriteAnalyzer`     | `analyze_readwrite` (:3123)            |
/// | `_canraise` 3-way       | `_canraise` (:2773)                    |
/// | `getcalldescr`          | `getcalldescr` (:2799)                 |
///
/// What is NOT plumbed is the *consumer side*: this producer (the
/// pyre-jit `CallFlavor` enum) does not query
/// `CallControl::getcalldescr` for the residual call's callee.  Every
/// variant below therefore leaves the analyzer-derived fields at
/// `EffectInfo::default()` (`oopspecindex = None`, `*_descrs_*` = 0,
/// `can_invalidate = false`, `extradescrs = None`, `can_collect =
/// true` per the Default impl) — the stub answer for callees the
/// producer cannot identify by calldescr.
///
/// Implications for the optimizer (audited in
/// `majit-metainterp/src/optimizeopt/`):
///   - `oopspecindex == None` → every `match ei.oopspecindex { ... }`
///     site (vstring.rs:759, intbounds.rs:2825, rewrite.rs:2774/2915,
///     virtualize.rs:1397/1450/1493/1512/1531, earlyforce.rs:31,
///     heap.rs:1416-1429) takes the default arm and skips the
///     OS_*-specialized rewrite.  Functionally safe (default arm is
///     conservative; missing the rewrite costs trace quality, not
///     correctness) but means pyre never benefits from `OS_STR_CONCAT`,
///     `OS_DICT_LOOKUP`, `OS_RAW_MALLOC_VARSIZE_CHAR`,
///     `OS_JIT_FORCE_VIRTUALIZABLE` etc. specialization.
///   - `write_descrs_arrays == 0` → `rewrite.rs:1993` heap
///     invalidation reads "this call writes no arrays".  Currently
///     load-bearing only when the trace records both an array
///     write-in-callee and a subsequent reader of the same array;
///     pyre's active callees (`box_int`, `load_const`, etc.) don't
///     hit this pattern, but a future LoopInvariant or Pure callee
///     that mutates arrays could trigger an incorrect cached read.
///   - `can_invalidate == false` → quasiimmut invalidation guards
///     are conservatively elided (not currently load-bearing because
///     pyre's quasi-immutable layer is itself unported).
///   - `extradescrs == None` → `heap.rs:712 rordereddict` descriptor
///     specialization unreachable (also unported on the consumer
///     side; matches by missing).
///
/// Convergence (Task #64): build a callee-identity-keyed registry of
/// codewriter `getcalldescr` results so this producer can resolve a
/// residual call's `EffectInfo` from `CallControl` instead of returning
/// a `CallFlavor`-bucketed stub.  Until that plumbing lands, this
/// function is the producer-side fallback that future EI-aware
/// optimizations must be careful not to rely on.  When a concrete
/// callee needs a specific EI field set (e.g. an oopspec-specialized
/// helper), construct the `EffectInfo` directly at the call site
/// rather than extending this `CallFlavor` mnemonic.
/// Producer-side macro-time map from a [`CallFlavor`] to the
/// matching [`majit_metainterp::EffectInfoSlot`] entry that
/// `JitCallTarget` carries.
///
/// `call.py:282-303 getcalldescr` selects `extraeffect` at codewriter
/// time from the analyzer chain (raise / loopinvariant / elidable);
/// pyre's macro-time `CallFlavor` already encodes the same per-helper
/// classification, so this function picks the matching slot const so
/// the runtime [`JitCallTarget`] descriptor carries the same
/// `extraeffect` the producer used when registering the helper.
///
/// **Panics for `MayForce` and `ReleaseGil`** — mirroring
/// `jtransform.py:1677` (`assert not
/// calldescr.get_extra_info().check_forces_virtual_or_virtualizable()`)
/// and `pyjitpl.py:2128-2132 do_conditional_call`'s identical assertion.
/// Those flavors carry runtime-resolved EI fields.  Code paths that
/// register helpers for residual_call dispatch (which never reads
/// `JitCallTarget.effect_info_slot`) must classify them through a
/// flavor-aware split — see `register_helper_fn_pointers::bind` for
/// the canonical pattern — instead of routing through this function.
pub fn slot_for_call_flavor(flavor: CallFlavor) -> majit_metainterp::EffectInfoSlot {
    use majit_metainterp::EffectInfoSlot;
    match flavor {
        // `call.py:301 getcalldescr` — `EF_CAN_RAISE`.
        CallFlavor::Plain => EffectInfoSlot::CanRaise,
        // `call.py:303 getcalldescr` — `EF_CANNOT_RAISE` (`else` branch).
        // RPython has a single `EF_CANNOT_RAISE` constant; the "no heap
        // touched" property of `PlainCannotRaiseNoHeap` is captured in
        // the EI's raw/bitstring shape (`effectinfo.py:281-283 empty
        // frozenset`), not as a distinct slot kind. Collapse both
        // flavors to one slot here, keeping the EI shape
        // differentiation in `effect_info_for_call_flavor` below.
        CallFlavor::PlainCannotRaise | CallFlavor::PlainCannotRaiseNoHeap => {
            EffectInfoSlot::CannotRaise
        }
        // `call.py:291 getcalldescr` — `EF_LOOPINVARIANT`.
        CallFlavor::LoopInvariant => EffectInfoSlot::LoopInvariant,
        // `call.py:292-299 getcalldescr` 3-way elidable pick.
        CallFlavor::PureCannotRaise => EffectInfoSlot::ElidableCannotRaise,
        CallFlavor::PureOrMemerror => EffectInfoSlot::ElidableOrMemerror,
        CallFlavor::PureCanRaise => EffectInfoSlot::ElidableCanRaise,
        // `jtransform.py:1677 _rewrite_op_cond_call`'s assert and
        // `pyjitpl.py:2128-2132 do_conditional_call`'s `assert not
        // check_forces_virtual_or_virtualizable()` are violated by
        // these flavors. Crashing here instead of silently lowering
        // to `CanRaise` keeps the assertion semantics intact.
        CallFlavor::MayForce | CallFlavor::ReleaseGil => panic!(
            "slot_for_call_flavor: {flavor:?} cannot be encoded as an \
             EffectInfoSlot; cond_call / record_known_result reject \
             this flavor per jtransform.py:1677. The caller must \
             dispatch on CallFlavor pattern and register \
             slot-irrelevant residual_call helpers via \
             SSAReprEmitter::add_fn_ptr instead."
        ),
    }
}

/// Returns `true` for HLOp opnames that pyre's walker emits for shadow
/// consistency with upstream `flowcontext.py` / `flowobject.py` but
/// whose RPython-orthodox rtyper rewrites (typically `rclass.rtype_type`
/// / `rclass.rtype_getattr`) pyre's pipeline does not run.  The
/// canonical `flatten_graph` driver elides these HLOps under
/// `lowering_ctx` so the resulting `SSARepr` doesn't emit a literal
/// Insn opname that the runtime cannot dispatch.
///
/// `type` — emitted by `codewriter.rs::explicit_raise_exception_pair`
/// mirroring `flowcontext.py:635 op.type(w_value)`.  The HLOp result
/// Variable is consumed via `link.last_exception`; the link's
/// `generate_last_exc` emits a `last_exception` Insn that produces the
/// type via TLS, so eliding the `type` HLOp itself is safe (the
/// Variable's color stays allocated and gets written by
/// `last_exception` at the catch landing).  Upstream `rclass.py:828
/// rtype_type` rewrites this to `getfield_gc_r(v, '__class__')` —
/// pyre's runtime exception model bakes type into per-subclass
/// `W_TypeObject` (see [[project-exception-per-kind-pytype]]) so the
/// `getfield_gc_r` shape is not required.
///
/// `getattr` — emitted by `codewriter.rs::emit_frontend_getattr`
/// mirroring `flowcontext.py:862-867 op.getattr(w_obj, w_attributename)`.
/// Pyre's walker pairs every `emit_frontend_getattr` callsite with an
/// inline `emit_abort_permanent!` (see codewriter.rs:6867 LoadAttr
/// arm) because pyre's runtime cannot compile attribute lookups; the
/// trace bails out to the interpreter before any consumer reads the
/// HLOp's result Variable.  After the `emit_abort_permanent!` graph
/// dual-write (codewriter.rs:3756), canonical's per-block iteration
/// already emits the `abort_permanent` Insn that terminates the
/// compiled trace — emitting a literal `getattr` Insn would be both
/// unreachable at runtime AND undispatchable by the assembler.
/// Upstream `rclass.py:838 rtype_getattr` would rewrite this to
/// `getfield_gc_X(v, descr)` after rtyping; pyre's lack of rtyping
/// keeps the HLOp unmodified, so eliding it under lowering_ctx is the
/// production-safe path.
fn is_pyre_canonical_elidable_hlop(opname: &str) -> bool {
    matches!(opname, "type" | "getattr")
}

pub fn effect_info_for_call_flavor(flavor: CallFlavor) -> majit_ir::EffectInfo {
    use majit_ir::{EffectInfo, ExtraEffect};
    match flavor {
        // `EF_CAN_RAISE` — `call.py:300-301 elif self._canraise(op):`
        // row of `getcalldescr`, fed through
        // `effectinfo_from_writeanalyze` with the
        // `graphanalyze.py:60 analyze_external_call` default
        // (`bottom_result()` = empty set). `effectinfo.py:285` only
        // force-promotes to `EF_RANDOM_EFFECTS` when
        // `effects is top_set`; the no-analyzer-output case takes the
        // `else` branch at `:293-299` and lands at
        // `extraeffect=CanRaise + Some([])` raw sets.
        CallFlavor::Plain => majit_metainterp::default_effect_info(),
        // `EF_CANNOT_RAISE` — `call.py:303 else:` row of `getcalldescr`
        // (non-elidable + `_canraise(op) == False`). Same
        // analyzer-empty external-call shape as `Plain`, just with
        // `extraeffect=CannotRaise` so `check_can_raise()`
        // (`effectinfo.py:236 extraeffect > EF_CANNOT_RAISE`) returns
        // false and the walker omits the trailing `GUARD_NO_EXCEPTION`.
        CallFlavor::PlainCannotRaise => majit_metainterp::cannot_raise_effect_info(),
        // `EF_CANNOT_RAISE` + analyzer-confirmed "no heap touched".
        // `call.py:320-324 effectinfo_from_writeanalyze` produces this
        // shape when the read/write analyzers return empty frozensets
        // and `_canraise(op)==False`. The concrete-empty raw sets +
        // empty bitstrings + `extraeffect=CannotRaise` together give
        // `check_can_raise()=false` (no GUARD_NO_EXCEPTION),
        // `check_forces_virtual_or_virtualizable()=false` (no
        // GUARD_NOT_FORCED), `has_random_effects()=false` (no
        // clean_caches), and `force_from_effectinfo` finds no descr
        // bits set so no per-cached-descr flush either.
        CallFlavor::PlainCannotRaiseNoHeap => majit_metainterp::CANNOT_RAISE_NO_HEAP_EFFECT_INFO,
        // `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` —
        // `call.py:288-289 if self.virtualizable_analyzer.analyze(op):`
        // row of `getcalldescr`, fed through
        // `effectinfo_from_writeanalyze` with the
        // `graphanalyze.py:60` analyzer default (empty set). Distinct
        // from `EF_RANDOM_EFFECTS` (`call.py:282-283
        // randomeffects_analyzer` branch): both pass
        // `check_forces_virtual_or_virtualizable()` via `>=` ordering at
        // `effectinfo.py:249-250`, but only RandomEffects trips
        // `has_random_effects()` (`effectinfo.py:252`) → routes
        // OptHeap through `clean_caches`. Collapsing MayForce onto
        // `MOST_GENERAL` would over-invalidate heap state PyPy keeps
        // live for virtualizable-forcing callees with an empty heap
        // effects analysis.
        CallFlavor::MayForce => majit_metainterp::forces_virtual_or_virtualizable_effect_info(),
        // EF_LOOPINVARIANT — `effectinfo.py:18`.
        // `optimize_CALL_LOOPINVARIANT_*` branch.
        CallFlavor::LoopInvariant => EffectInfo {
            extraeffect: ExtraEffect::LoopInvariant,
            ..EffectInfo::default()
        },
        // `call.py:292-299 getcalldescr`'s 3-way elidable pick:
        //
        //     elif elidable:
        //         cr = self._canraise(op)
        //         if cr == "mem":      extraeffect = EF_ELIDABLE_OR_MEMORYERROR
        //         elif cr:             extraeffect = EF_ELIDABLE_CAN_RAISE
        //         else:                extraeffect = EF_ELIDABLE_CANNOT_RAISE
        //
        // Each `CallFlavor::Pure*` variant maps to the corresponding
        // `EF_ELIDABLE_*` const so the producer's per-callee `_canraise`
        // outcome reaches `do_residual_call`'s `check_can_raise(False)`
        // gate verbatim: `ElidableCannotRaise` (0) → false (no
        // `GUARD_NO_EXCEPTION`), `ElidableOrMemoryError` (3) /
        // `ElidableCanRaise` (4) → true (guard recorded). Producers
        // pick the right variant based on callee analysis; a producer
        // that does not yet have the analyzer wired falls back to
        // `PureCanRaise` (`Task #64` callee-identity-keyed registry).
        CallFlavor::PureCannotRaise => majit_metainterp::ELIDABLE_CANNOT_RAISE_EFFECT_INFO,
        CallFlavor::PureOrMemerror => majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        CallFlavor::PureCanRaise => majit_metainterp::ELIDABLE_EFFECT_INFO,
        // Release-gil cannot be encoded by this generic flavor mapper:
        // PyPy's `call.py:252-258` stores the real `(target_fn_addr,
        // save_err)` in the EffectInfo at descr creation time.  Pyre's
        // via-target lowering may use a temporary sentinel, but that
        // must be requested explicitly through
        // `unresolved_release_gil_effect_info_for_via_target()` so the
        // sentinel cannot escape from general `CallFlavor` conversion.
        CallFlavor::ReleaseGil => panic!(
            "effect_info_for_call_flavor: ReleaseGil requires the resolved \
             (target_fn_addr, save_err) pair from `call.py:252-258 \
             _call_aroundstate_target_`; use \
             `unresolved_release_gil_effect_info_for_via_target` only for \
             residual_call via-target lowering that immediately flows \
             through `resolve_call_release_gil_target`."
        ),
    }
}

/// Release-gil EffectInfo seed for residual-call via-target lowering only.
///
/// PyPy stores a real `(target_fn_addr, save_err)` in
/// `EffectInfo.call_release_gil_target` when the calldescr is created
/// (`call.py:252-258`).  Pyre's `CallDescrStub` path does not know the
/// concrete target until `JitCodeBuilder` resolves `descrs[fn_ptr_idx]`,
/// so this helper returns a non-zero `(1, 0)` sentinel solely for
/// `resolve_call_release_gil_target` to replace before materializing the
/// final calldescr.  Do not use it for cached/interned descriptors that
/// can bypass that resolver.
pub fn unresolved_release_gil_effect_info_for_via_target() -> majit_ir::EffectInfo {
    use majit_ir::EffectInfo;
    // PyPy `effectinfo.py:149-155`: every six raw `_*_descrs_*` set
    // MUST be None when extraeffect=RandomEffects. The previous shape
    // explicitly set the bitstrings to None but inherited the raw sets
    // from `..EffectInfo::default()` (= `Some(Vec::new())`), violating
    // the invariant. Cloning `MOST_GENERAL` and overlaying the
    // `(target_fn_addr, save_err)` sentinel keeps RandomEffects+raw=None+
    // bitstring=None consistent.
    EffectInfo {
        call_release_gil_target: (1, 0),
        ..EffectInfo::MOST_GENERAL.clone()
    }
}

/// Inverse of [`effect_info_for_call_flavor`]: derive the dispatch branch
/// `dispatch_residual_call` should pick from a calldescr's `EffectInfo`.
/// Mirrors `rpython/jit/metainterp/pyjitpl.py:1995-2126 do_residual_call`'s
/// branch precedence — `forces_virtual_or_virtualizable` (with the
/// `is_call_release_gil()` sub-case) wins first, then `EF_LOOPINVARIANT`,
/// then `check_is_elidable()`, else the plain `CALL_*` branch.
///
/// Precedence note: `is_call_release_gil()` is checked **before**
/// `check_forces_virtual_or_virtualizable()` because release-gil EIs
/// carry `EF_RANDOM_EFFECTS` (mirroring `call.py:282-289 getcalldescr`'s
/// `random_effects` upgrade for release-gil callees), which makes
/// `check_forces_virtual_or_virtualizable()` (`>= 6`) also return
/// true on those EI values.  The early `is_call_release_gil()` check
/// keeps the dispatch branch aligned with `pyjitpl.py:2063`, where the
/// release-gil sub-case is selected inside the outer forces branch.
pub fn dispatch_kind_for_effect_info(ei: &majit_ir::EffectInfo) -> CallFlavor {
    use majit_ir::ExtraEffect;
    if ei.is_call_release_gil() {
        return CallFlavor::ReleaseGil;
    }
    if ei.check_forces_virtual_or_virtualizable() {
        return CallFlavor::MayForce;
    }
    match ei.extraeffect {
        ExtraEffect::LoopInvariant => CallFlavor::LoopInvariant,
        // `call.py:292-299`'s 3-way pick survives the round-trip:
        // each `EF_ELIDABLE_*` lands on its matching `Pure*` variant.
        ExtraEffect::ElidableCannotRaise => CallFlavor::PureCannotRaise,
        ExtraEffect::ElidableOrMemoryError => CallFlavor::PureOrMemerror,
        ExtraEffect::ElidableCanRaise => CallFlavor::PureCanRaise,
        // `call.py:303 getcalldescr`'s non-elidable cannot-raise branch.
        ExtraEffect::CannotRaise => CallFlavor::PlainCannotRaise,
        _ => CallFlavor::Plain,
    }
}

/// `rpython/jit/metainterp/optimizeopt/rewrite.py` `Rewrite.optimize_CALL_XXX`
/// branches on `op.getdescr().effectinfo.extraeffect` to select between
/// `call_may_force`, `call_release_gil`, `call_loopinvariant`, and
/// `call_pure`. In pyre the codewriter knows statically which branch
/// applies for each per-PC helper, so the enum names the branch the
/// codewriter wants and [`effect_info_for_call_flavor`] expands it to
/// the `EffectInfo` that drives dispatch.
///
/// `CALL_ASSEMBLER` is intentionally not represented here — upstream
/// `rop.CALL_ASSEMBLER_*` is a separate operation chosen via
/// `OpHelpers.call_assembler_for_descr` (`resoperation.py:1251-1260`),
/// not derived from `EffectInfo`. pyre's portal-call lowering follows
/// the same split (`majit-ir/src/resoperation.rs:1120-1123
/// CallAssembler{I,R,F,N}`); reintroducing an `Assembler` flavor here
/// would push the wrong path back into the residual_call shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallFlavor {
    /// Plain residual call, conservative `EF_CAN_RAISE` default
    /// (`rpython/jit/codewriter/effectinfo.py:22`). `call.py:301
    /// getcalldescr` picks this when the analyzer reports the callee
    /// can raise but is not elidable / loop-invariant / forces.
    Plain,
    /// `EF_CANNOT_RAISE` (`effectinfo.py:19`). `call.py:303 getcalldescr`
    /// picks this on the non-elidable `else` branch when
    /// `_canraise(op) == False` — `pyjitpl.py:2111 do_residual_call`
    /// then drops the trailing `GUARD_NO_EXCEPTION`. pyre's analyzer
    /// port (Task #64) is the upstream replacement; producers select
    /// this flavor today only when the callee is statically known not
    /// to raise.
    ///
    /// Maps to `cannot_raise_effect_info()` =
    /// `EffectInfo::const_new(CannotRaise, None)` — `extraeffect=CannotRaise`,
    /// every six `_*_descrs_*` raw set + `*_descrs_*` bitstring =
    /// `Some(Vec::new())`, `can_collect=true`. This is the PyPy
    /// `effectinfo.py:293-299` else-branch shape (analyzer-empty
    /// `effects` is `bottom_result()` per `graphanalyze.py:60`, not
    /// `top_set`), distinct from `MOST_GENERAL`. Producers that can
    /// additionally prove "no heap touched + no GC" should use
    /// `PlainCannotRaiseNoHeap` (`can_collect=false`) instead.
    PlainCannotRaise,
    /// `EF_CANNOT_RAISE` + analyzer-confirmed "no heap touched". Maps
    /// to `CANNOT_RAISE_NO_HEAP_EFFECT_INFO` (`call_descr.rs:317-329`):
    /// `extraeffect=CannotRaise`, every six raw set `Some(empty Vec)`,
    /// every six bitstring `Some(empty Vec)`, `can_collect=false`.
    /// PyPy `effectinfo.py:281-283` produces the same shape when the
    /// analyzer reports an empty frozenset and `_canraise(op)==False`.
    /// Use for TLS-only / register-only helpers that the producer can
    /// statically prove touch no GC heap (e.g.,
    /// `get_current_exception_fn` / `set_current_exception_fn`).
    PlainCannotRaiseNoHeap,
    /// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`. The builder emits
    /// `call_may_force_*` so the metainterp forces virtualizable state
    /// before the call. Maps to `JitCodeBuilder::call_may_force_*_typed`.
    MayForce,
    /// `EF_LOOPINVARIANT`. One-shot call memoised across the trace loop.
    /// Maps to `JitCodeBuilder::call_loopinvariant_*_typed`.
    LoopInvariant,
    /// `EF_RELEASES_GIL`. Maps to `JitCodeBuilder::call_release_gil_*_typed`.
    ReleaseGil,
    /// `EF_ELIDABLE_CANNOT_RAISE` (`effectinfo.py:17`). `call.py:299
    /// getcalldescr` picks this branch when `_canraise(op) == False`.
    /// `pyjitpl.py:2126 do_residual_call` records `CALL_PURE_*` with
    /// no trailing `GUARD_NO_EXCEPTION` because
    /// `check_can_raise(False)` is false for `extraeffect == 0`.
    PureCannotRaise,
    /// `EF_ELIDABLE_OR_MEMORYERROR` (`effectinfo.py:20`). `call.py:295
    /// getcalldescr` picks this when `_canraise(op) == "mem"` — the
    /// elidable callee's only failure mode is `MemoryError`. Same
    /// dispatch as `PureCanRaise` but distinguished for optimizer
    /// metadata.
    PureOrMemerror,
    /// `EF_ELIDABLE_CAN_RAISE` (`effectinfo.py:21`). `call.py:297
    /// getcalldescr` picks this when `_canraise(op) == True`.
    PureCanRaise,
}

impl CallFlavor {
    /// Convenience predicate for the three elidable variants —
    /// `effectinfo.check_is_elidable()` parity (`effectinfo.py:225`).
    pub fn is_pure(self) -> bool {
        matches!(
            self,
            CallFlavor::PureCannotRaise | CallFlavor::PureOrMemerror | CallFlavor::PureCanRaise
        )
    }
}

/// `rpython/jit/codewriter/jtransform.py:423` `reskind =
/// getkind(op.result.concretetype)[0]`. The four result-kind suffixes
/// used by `residual_call_{kinds}_{reskind}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResKind {
    Int,
    Ref,
    Float,
    Void,
}

impl ResKind {
    /// The single-character suffix used in `residual_call_{kinds}_{reskind}`.
    /// `rpython/jit/codewriter/jtransform.py:434`
    /// `'%s_%s_%s' % (namebase, kinds, reskind)`.
    pub fn as_char(self) -> char {
        match self {
            ResKind::Int => 'i',
            ResKind::Ref => 'r',
            ResKind::Float => 'f',
            ResKind::Void => 'v',
        }
    }

    /// The non-void reskinds map to a `Kind` for the result `Register`.
    pub fn to_kind(self) -> Option<Kind> {
        match self {
            ResKind::Int => Some(Kind::Int),
            ResKind::Ref => Some(Kind::Ref),
            ResKind::Float => Some(Kind::Float),
            ResKind::Void => None,
        }
    }
}

// --------------------------------------------------------------------------
// Instruction shape.
//
// RPython's `ssarepr.insns` is a list of Python tuples with five possible
// shapes:
//   1. `(Label(name),)`                                              — block header
//   2. `('-live-', arg1, arg2, ...)`                                 — liveness marker
//   3. `('---',)`                                                    — unreachable
//   4. `(opname, arg1, arg2, ..., '->', result_register)`            — op with result
//   5. `(opname, arg1, arg2, ...)`                                   — op without result
//
// Rust has no untyped tuples, so `Insn` is an enum that captures these
// five shapes explicitly. The `Op` variant retains a separate `result`
// field to match RPython's `'->'` marker rather than requiring the caller
// to scan the args list.
// --------------------------------------------------------------------------

/// `flatten.py` / `assembler.py` tuple-argument types.
///
/// Exhaustive variant set for anything that can appear inside an Insn's
/// argument list. Constants follow `assembler.py:157-175` — int, ref
/// (GCREF pointer as `i64`), and float (bit-pattern stored as `i64`).
#[derive(Debug, Clone)]
pub enum Operand {
    Register(Register),
    /// `Constant(Signed)` in RPython.
    ConstInt(i64),
    /// `Constant(GCREF)` in RPython — a boxed reference stored as a
    /// pointer-sized int. `PY_NULL` is represented as `0`.
    ConstRef(i64),
    /// `Constant(Float)` in RPython — stored as bit-pattern for
    /// parity with `assembler.emit_const`'s float path.
    ConstFloat(i64),
    /// Forward reference to a `Label`.
    TLabel(TLabel),
    /// A same-kind list of Registers/Constants.
    ListOfKind(ListOfKind),
    /// `assembler.py:197-206` `elif isinstance(x, AbstractDescr)` /
    /// `liveness.py:76` `elif isinstance(x, SwitchDictDescr)` — either a
    /// runtime-resolved descr (`BhDescr`) or an SSARepr-side
    /// `SwitchDictDescr` with a live `_labels` table.
    ///
    /// Wrapped in `Rc` so cloning the `Operand` preserves Python object
    /// identity: `assembler.py:197-199` keys `self._descr_dict` on
    /// `id(x)`, and two SSARepr sites that share the SAME descr object
    /// MUST dedup to the same `descrs` index. With `Rc`, callers that
    /// build an SSARepr `clone()` the `Rc` (pointer-preserving) while
    /// still being able to construct distinct descrs with
    /// `Rc::new(...)` when identity should differ. A plain
    /// `DescrOperand` value field would lose this distinction on every
    /// `Clone`.
    Descr(Rc<DescrOperand>),
    /// `IndirectCallTargets` — list of jitcodes for `indirect_call`.
    IndirectCallTargets(IndirectCallTargets),
}

impl Operand {
    pub fn reg(kind: Kind, index: u16) -> Self {
        Operand::Register(Register::new(kind, index))
    }

    /// Wrap a `DescrOperand` into a fresh `Rc` and build an `Operand`.
    /// Callers that want two `Operand::Descr`s to dedup to the same
    /// `descrs` index must `Rc::clone` the `Rc` returned by this call
    /// rather than invoking `descr()` twice with equal values.
    pub fn descr(value: DescrOperand) -> Self {
        Operand::Descr(Rc::new(value))
    }

    /// Build an `Operand::Descr` from an existing `Rc<DescrOperand>`.
    /// Preserves identity for dedup (`assembler.py:197-199`).
    pub fn descr_rc(value: Rc<DescrOperand>) -> Self {
        Operand::Descr(value)
    }

    /// `Operand::Descr(Rc::new(DescrOperand::VableArrayField(idx)))` —
    /// `rpython/jit/metainterp/virtualizable.py:73 array_field_descrs[i]`.
    /// Used by `getarrayitem_vable_X` / `setarrayitem_vable_X` /
    /// `arraylen_vable` SSARepr ops to carry the array-field index for
    /// `assembler.py:80-138 emit_const` lowering.
    pub fn descr_vable_array_field(idx: u16) -> Self {
        Operand::Descr(Rc::new(DescrOperand::VableArrayField(idx)))
    }

    /// `Operand::Descr(Rc::new(DescrOperand::VableArray(idx)))` —
    /// `rpython/jit/metainterp/virtualizable.py:58 array_descrs[i]`.
    /// Paired with `descr_vable_array_field(i)` at the trailing operand
    /// position of every vable arrayitem op.
    pub fn descr_vable_array(idx: u16) -> Self {
        Operand::Descr(Rc::new(DescrOperand::VableArray(idx)))
    }

    /// `Operand::Descr(Rc::new(DescrOperand::VableStaticField(idx)))` —
    /// `rpython/jit/metainterp/virtualizable.py:71 static_field_descrs[i]`.
    /// Trailing descr operand of `getfield_vable_<kind>` (after `v_inst`,
    /// `jtransform.py:846`) and `setfield_vable_<kind>` (after `v_inst,
    /// v_value`, `jtransform.py:927`).
    pub fn descr_vable_static_field(idx: u16) -> Self {
        Operand::Descr(Rc::new(DescrOperand::VableStaticField(idx)))
    }
}

/// RPython `-live-` marker opname (`liveness.py:5-12`). Stored as the
/// first tuple element in RPython's ssarepr; pyre stores it as the
/// `opname` field of `Insn::Op`, matching the tuple-shape exactly.
pub const OPNAME_LIVE: &str = "-live-";

/// Instruction tuple (`ssarepr.insns[i]`).
///
/// The three RPython tuple shapes enumerated above: `Label`, `---`
/// (`Unreachable`), and regular op (which also carries the `-live-`
/// marker via `opname == OPNAME_LIVE`).  `-live-` shares the `Op`
/// variant with regular operations, matching RPython's tuple
/// representation where `insn[0] == '-live-'` is the discriminator.
#[derive(Debug, Clone)]
pub enum Insn {
    /// `(Label(name),)` — block-entry marker.
    Label(Label),
    /// pyre-only per-PC anchor marker emitted by the walker at every
    /// Python PC entry.  Semantically equivalent to a block-entry
    /// `Insn::Label` with the synthesized name `pc_label_name(py_pc)`,
    /// but kept as a distinct variant so the runtime can structurally
    /// distinguish "block entry per `flatten.py:116`" from "per-PC
    /// anchor for any-PC resume (pyre's NEW-DEVIATION; upstream PyPy
    /// only resumes at `jit_merge_point` boundaries)".  The assembler
    /// records the byte position under `pc_label_name(py_pc)` in
    /// `label_positions` so existing `TLabel("pc{N}")` branch targets
    /// continue to resolve.
    PcAnchor { py_pc: usize },
    /// `('---',)` — unreachable marker; clears the liveness pass's alive
    /// set (`liveness.py:70`).
    Unreachable,
    /// `(opname, args..., ['->' result])` — regular operation, including
    /// `-live-` liveness markers (`opname == OPNAME_LIVE`). `result` is
    /// `Some(register)` iff the RPython tuple contains a trailing
    /// `'->' result` pair; the assembler emits the `>` argcode in that
    /// case (`assembler.py:210-219`). `-live-` always has
    /// `result == None`.
    Op {
        opname: String,
        args: Vec<Operand>,
        result: Option<Register>,
    },
}

impl Insn {
    /// `Insn::Op` with no result register.
    pub fn op(opname: impl Into<String>, args: Vec<Operand>) -> Self {
        Insn::Op {
            opname: opname.into(),
            args,
            result: None,
        }
    }

    /// `Insn::Op` with a trailing `'->' result` pair.
    pub fn op_with_result(opname: impl Into<String>, args: Vec<Operand>, result: Register) -> Self {
        Insn::Op {
            opname: opname.into(),
            args,
            result: Some(result),
        }
    }

    /// `('-live-', args...)` marker, RPython `liveness.py` parity.
    pub fn live(args: Vec<Operand>) -> Self {
        Insn::Op {
            opname: OPNAME_LIVE.to_string(),
            args,
            result: None,
        }
    }

    /// pyre-only per-PC anchor for `py_pc`.  Walker emits one per
    /// Python PC entry; the runtime resolves `pc{N}` jumps via
    /// `pc_anchor_positions` / `live_marker_indices_by_pc` scans
    /// against this variant.
    pub fn pc_anchor(py_pc: usize) -> Self {
        Insn::PcAnchor { py_pc }
    }

    /// `true` iff this instruction is a `-live-` marker.
    pub fn is_live(&self) -> bool {
        matches!(self, Insn::Op { opname, .. } if opname == OPNAME_LIVE)
    }

    /// `Some(&args)` if this instruction is a `-live-` marker, else `None`.
    pub fn live_args(&self) -> Option<&[Operand]> {
        match self {
            Insn::Op { opname, args, .. } if opname == OPNAME_LIVE => Some(args),
            _ => None,
        }
    }

    /// `Some(&mut args)` if this instruction is a `-live-` marker, else `None`.
    pub fn live_args_mut(&mut self) -> Option<&mut Vec<Operand>> {
        match self {
            Insn::Op { opname, args, .. } if opname == OPNAME_LIVE => Some(args),
            _ => None,
        }
    }
}

/// Minimal production slice of `rpython/jit/codewriter/flatten.py:
/// 60-350` `GraphFlattener`.
///
/// Upstream owns the whole `FunctionGraph -> SSARepr` lowering. pyre is
/// still in the transitional dual-write phase, so this helper currently
/// serializes individual graph-level `SpaceOperation`s into `Insn`s and is
/// used only for the first production op migrated off direct SSA emission.
/// Expand this helper as more ops move from `codewriter.rs` into the
/// flow-graph + flatten pipeline.
pub struct GraphFlattener<'a, F, C = fn(&Constant) -> Operand> {
    ssarepr: &'a mut SSARepr,
    get_register: F,
    lower_constant: C,
    /// `rpython/jit/codewriter/flatten.py:103 self.seen_blocks = {}` —
    /// the recursive `make_bytecode_block` DFS tracks which blocks have
    /// been emitted to short-circuit back-edges into `goto TLabel(block)`.
    /// Per [[feedback-no-hashmap-ever]] pyre uses `Vec<BlockRef>` with
    /// linear scan: graph block counts stay in the dozens for production
    /// workloads, so O(N) `.contains()` is acceptable and matches the
    /// upstream dict's "identity membership" semantics without a hash.
    seen_blocks: Vec<BlockRef>,
    /// `flatten.py` does not name blocks directly — the upstream
    /// `Label(block)` shape uses the block's identity hash via Python's
    /// default `__hash__`.  Pyre needs a stringly Label/TLabel name for
    /// the runtime's PC-dispatch lookup, so this side-table assigns
    /// sequential `block{N}` names on first sight.  Same Vec-scan rule
    /// as `seen_blocks`.
    block_names: Vec<(BlockRef, String)>,
    /// Per-link counterpart to `block_names` for `Label(link)` /
    /// `TLabel(link)` shapes emitted at canraise / switch sites.
    link_names: Vec<(LinkRef, String)>,
    /// Pyre-only side table mapping graph blocks to their Python PC.
    /// When non-empty, `make_bytecode_block` emits an `Insn::PcAnchor
    /// { py_pc }` immediately after the block-entry `Label` for blocks
    /// present in this table, matching the walker's per-PC anchor
    /// emission in `codewriter.rs::emit_mark_label_pc!`.  This makes
    /// the canonical driver's output runtime-compatible with pyre's
    /// `pc_anchor_positions` / `live_marker_indices_by_pc` lookups
    /// without requiring runtime refactor.  Upstream RPython has no
    /// per-PC anchor concept (RPython's runtime dispatch keys off the
    /// recorded SpamBlock identity, not Python PC), so this is a pyre
    /// adaptation layer.  Pre-seeded by the bridge entry
    /// `flatten_graph_with_walker_slots`; empty for plain
    /// `flatten_graph` test fixtures.
    block_py_pcs: Vec<(BlockRef, usize)>,
    /// Pyre-only force-alive Register operands threaded into every
    /// `-live-` placeholder emitted by `make_bytecode_block` (matches
    /// walker's `emit_live_placeholder!` macro at
    /// `codewriter.rs::4329-4339`).  Walker keeps the portal red args
    /// (`pypy/module/pypyjit/interp_jit.py:67 reds = ['frame', 'ec']`)
    /// alive across every PC by emitting their Register operands into
    /// each per-PC `-live-` — `compute_liveness` then propagates them
    /// backward into the alive set per `liveness.py:11-12`.  Without
    /// the operands the bridge's `setup_bridge_sym` sees the portal
    /// `ec` register missing from the guard point's live R-bank set,
    /// crashing `bridge_registers_r[portal_ec_reg]` lookup.  Pre-
    /// seeded by `flatten_graph_with_walker_slots` from codewriter's
    /// `portal_frame_reg` / `portal_ec_reg`.
    live_force_alive_ops: Vec<Operand>,
    next_label_id: usize,
    include_all_exc_links: bool,
    /// `rpython/jit/codewriter/flatten.py:79 self.cpu = cpu`.
    ///
    /// Upstream `flatten_graph(graph, regallocs, _include_all_exc_links,
    /// cpu)` threads the LLGraphCPU through so `make_exception_link`
    /// can read `self.cpu.rtyper.exceptiondata.
    /// get_standard_ll_exc_instance_by_class(OverflowError)` on the
    /// `handling_ovf=True` arm (`flatten.py:166-170`).  Pyre stores it
    /// as a borrow; production callers thread `CodeWriter::cpu()`
    /// (`codewriter.rs:2661`).  Test fixtures that do not exercise
    /// the overflow path leave it `None`, matching upstream's
    /// `cpu=None` default at `flatten.py:64`.
    cpu: Option<&'a super::cpu::Cpu>,
    /// When `Some`, `flatten_space_operation` routes pre-rtype HLOp
    /// opnames from the four retired families (BINARY_OP / COMPARE_OP
    /// / BOOL / SETITEM) through
    /// `try_flatten_retired_family_hlop_to_insn`, producing the
    /// post-rtype `residual_call_*` Insn the assembler expects.  When
    /// `None`, the legacy opname-passthrough emits `Insn::op("add",
    /// ...)` etc. directly — used by tests against structural-only
    /// graphs.
    ///
    /// Production callers populate it via `cpu.lowering_ctx`
    /// (`codewriter.rs::transform_graph_to_jitcode`).
    lowering_ctx: Option<LoweringContext>,
}

impl<'a, F> GraphFlattener<'a, F>
where
    F: FnMut(Variable) -> Register,
{
    pub fn new(ssarepr: &'a mut SSARepr, get_register: F) -> Self {
        Self {
            ssarepr,
            get_register,
            lower_constant: flatten_constant_operand,
            seen_blocks: Vec::new(),
            block_names: Vec::new(),
            link_names: Vec::new(),
            block_py_pcs: Vec::new(),
            live_force_alive_ops: Vec::new(),
            next_label_id: 0,
            include_all_exc_links: false,
            cpu: None,
            lowering_ctx: None,
        }
    }
}

impl<'a, F, C> GraphFlattener<'a, F, C>
where
    F: FnMut(Variable) -> Register,
    C: FnMut(&Constant) -> Operand,
{
    pub fn new_with_constant_lowering(
        ssarepr: &'a mut SSARepr,
        get_register: F,
        lower_constant: C,
    ) -> Self {
        Self {
            ssarepr,
            get_register,
            lower_constant,
            seen_blocks: Vec::new(),
            block_names: Vec::new(),
            link_names: Vec::new(),
            block_py_pcs: Vec::new(),
            live_force_alive_ops: Vec::new(),
            next_label_id: 0,
            include_all_exc_links: false,
            cpu: None,
            lowering_ctx: None,
        }
    }

    /// GraphFlattener constructor that enables retired-family HLOp
    /// lowering.  Routes `add` / `lt` / ... / `bool` / `setitem`
    /// SpaceOperations through `try_flatten_retired_family_hlop_to_insn`
    /// (which lowers them to the matching `residual_call_*` Insn
    /// shape) before the passthrough opname-emit fallback.  Non-HLOp
    /// opnames (structural ops like `loop_header` /
    /// `jit_merge_point`, post-rtype `residual_call_*` ops recorded
    /// by factor-refactored families' graph dual-writes) keep their
    /// existing passthrough handling because the dispatcher's `try_*`
    /// returns `None` for them.
    pub fn new_with_full_lowering(
        ssarepr: &'a mut SSARepr,
        get_register: F,
        lower_constant: C,
        lowering_ctx: LoweringContext,
    ) -> Self {
        Self {
            ssarepr,
            get_register,
            lower_constant,
            seen_blocks: Vec::new(),
            block_names: Vec::new(),
            link_names: Vec::new(),
            block_py_pcs: Vec::new(),
            live_force_alive_ops: Vec::new(),
            next_label_id: 0,
            include_all_exc_links: false,
            cpu: None,
            lowering_ctx: Some(lowering_ctx),
        }
    }

    /// `flatten.py:63 def flatten_graph(graph, regallocs,
    /// _include_all_exc_links=False, cpu=None)` cpu kwarg parity.
    ///
    /// Production callers thread `CodeWriter::cpu()` so
    /// `make_exception_link`'s `handling_ovf=True` arm can fetch the
    /// `OverflowError` exception instance (`flatten.py:166-170`).
    /// Returns `self` to support builder-style chaining at construction.
    pub fn with_cpu(mut self, cpu: &'a super::cpu::Cpu) -> Self {
        self.cpu = Some(cpu);
        self
    }

    pub fn serialize_op(&mut self, op: &SpaceOperation) {
        // `flatten.py:373-380 serialize_op`: serialise an op into the
        // SSARepr via `emitline`.  Pyre's `flatten_space_operation`
        // splits out the per-op lowering (HLOp dispatch + arg /
        // result handling); the `emitline` call below matches
        // upstream's final push.
        //
        // Pyre-specific elide arm (canonical-driver-only): certain
        // HLOps emitted by pyre's walker for shadow consistency with
        // upstream `flowcontext.py` have no runtime Insn counterpart
        // and would be unsafe to emit literally.  Elide them only
        // under `lowering_ctx` (canonical production path); the
        // non-lowering path keeps upstream passthrough behavior
        // (preserves `graph_flattener_emits_generic_result_op` test
        // semantics).  See [[project-flatten-graph-canonical-driver-2026-05-17]].
        if self.lowering_ctx.is_some() && is_pyre_canonical_elidable_hlop(&op.opname) {
            return;
        }
        let insn = self.flatten_space_operation(op);
        self.emitline(insn);
    }

    fn emitline(&mut self, insn: Insn) {
        self.ssarepr.insns.push(insn);
    }

    /// `flatten.py:352-353 popline`: pop the most recently emitted
    /// insn off `ssarepr.insns`.  Used by the `_ovf` rewrite at
    /// `flatten.py:194 line = self.popline()` to retract the arithmetic
    /// op just emitted by `serialize_op` and replace it with the
    /// `*_jump_if_ovf` twin.
    fn popline(&mut self) -> Option<Insn> {
        self.ssarepr.insns.pop()
    }

    fn label_name_for_block(&mut self, block: &BlockRef) -> String {
        if let Some((_, name)) = self.block_names.iter().find(|(b, _)| b == block) {
            return name.clone();
        }
        let name = format!("block{}", self.next_label_id);
        self.next_label_id += 1;
        self.block_names.push((block.clone(), name.clone()));
        name
    }

    fn label_name_for_link(&mut self, link: &LinkRef) -> String {
        if let Some((_, name)) = self.link_names.iter().find(|(l, _)| l == link) {
            return name.clone();
        }
        let name = format!("link{}", self.next_label_id);
        self.next_label_id += 1;
        self.link_names.push((link.clone(), name.clone()));
        name
    }

    fn tlabel_for_block(&mut self, block: &BlockRef) -> Operand {
        Operand::TLabel(TLabel::new(self.label_name_for_block(block)))
    }

    fn tlabel_value_for_link(&mut self, link: &LinkRef) -> TLabel {
        TLabel::new(self.label_name_for_link(link))
    }

    fn tlabel_for_link(&mut self, link: &LinkRef) -> Operand {
        Operand::TLabel(self.tlabel_value_for_link(link))
    }

    fn label_for_block(&mut self, block: &BlockRef) -> Insn {
        Insn::Label(Label::new(self.label_name_for_block(block)))
    }

    fn label_for_link(&mut self, link: &LinkRef) -> Insn {
        Insn::Label(Label::new(self.label_name_for_link(link)))
    }

    fn flow_kind(value: &FlowValue) -> Option<Kind> {
        match value {
            FlowValue::Variable(variable) => variable.kind,
            FlowValue::Constant(constant) => constant.kind,
        }
    }

    fn rename_operand(&mut self, value: &FlowValue) -> RenameOperand {
        match self.getcolor(value) {
            Operand::Register(register) => RenameOperand::Register(register),
            Operand::ConstInt(value) => RenameOperand::ConstInt(value),
            Operand::ConstRef(value) => RenameOperand::ConstRef(value),
            Operand::ConstFloat(value) => RenameOperand::ConstFloat(value),
            other => panic!("insert_renamings expects register/constant, got {other:?}"),
        }
    }

    fn make_return(&mut self, args: &[FlowValue]) {
        match args {
            [value] => match Self::flow_kind(value) {
                None => self.emitline(Insn::op("void_return", Vec::new())),
                Some(kind) => {
                    let opname = format!("{}_return", kind.as_str());
                    let operand = self.getcolor(value);
                    self.emitline(Insn::op(opname, vec![operand]));
                }
            },
            [_, exc_value] => {
                if exc_value.as_variable().is_some() {
                    self.emitline(Insn::live(Vec::new()));
                }
                let operand = self.getcolor(exc_value);
                self.emitline(Insn::op("raise", vec![operand]));
            }
            _ => panic!("make_return expects 1 or 2 args, got {}", args.len()),
        }
        self.emitline(Insn::Unreachable);
    }

    fn make_link(&mut self, link: &LinkRef, handling_ovf: bool) {
        // `rpython/jit/codewriter/flatten.py:148-155 make_link`:
        //
        //     if (link.target.exits == ()
        //         and link.last_exception not in link.args
        //         and link.last_exc_value not in link.args):
        //         self.make_return(link.args)     # optimization only
        //         return
        //     self.insert_renamings(link)
        //     self.make_bytecode_block(link.target, handling_ovf)
        let (target, args, can_return_directly) = {
            let link_borrow = link.borrow();
            let target = link_borrow
                .target
                .clone()
                .expect("link target required for make_link");
            // `flatten.py:148-155 make_link` has no `target.dead`
            // check.  RPython `flowspace/flowcontext.py:455 mergeblock`
            // marks the superseded block dead and reroutes incoming
            // edges to the newblock via `recloseblock`, but the old
            // block itself stays linked from any predecessor whose
            // outgoing edge already named it as target — it serves as
            // a forwarding stub (`model.py:240-253 recloseblock` only
            // replaces exits; predecessors retain their original
            // target reference).  `iterblocks` (`model.py:55-77`)
            // follows links without filtering on `dead`, so flatten
            // legitimately recurses through a dead target whose
            // single exit forwards to the newblock.  Re-asserting
            // here would reject this legal upstream shape; the
            // empty-`operations` invariant set by mergeblock is
            // what carries the "no codegen" semantics.
            let target_is_final = target.borrow().exits.is_empty();
            let uses_last_exception = link_borrow.args.iter().any(|arg| {
                arg.as_ref()
                    .and_then(FlowValue::as_variable)
                    .is_some_and(|value| Some(value) == link_borrow.last_exception)
            });
            let uses_last_exc_value = link_borrow.args.iter().any(|arg| {
                arg.as_ref()
                    .and_then(FlowValue::as_variable)
                    .is_some_and(|value| Some(value) == link_borrow.last_exc_value)
            });
            let collected_args = link_borrow
                .args
                .iter()
                .flatten()
                .cloned()
                .collect::<Vec<_>>();
            (
                target,
                collected_args,
                target_is_final && !uses_last_exception && !uses_last_exc_value,
            )
        };
        if can_return_directly {
            self.make_return(&args);
            return;
        }
        self.insert_renamings(link);
        self.make_bytecode_block(target, handling_ovf);
    }

    fn make_exception_link(&mut self, link: &LinkRef, handling_ovf: bool) {
        // RPython `flatten.py:139-180 make_exception_link` requires
        // `link.last_exception` and `link.last_exc_value` to be seeded
        // by upstream `guessexception` (`flowcontext.py:130-143`).  In
        // pyre that seeding lives in `codewriter.rs::
        // attach_catch_exception_edge`'s `Link::extravars(Some, Some)`
        // call.  The W-4 self-loop fix retired the bypass path that
        // previously left exception edges with the pair unset (the
        // `[w-fallback w2/1648]` site that surfaced 4 hits in
        // raise_catch_loop pre-W-4); those hits dropped to 0 across
        // all 8 benches once the supersede chain stopped reaching
        // un-seeded join-points.  Inline the assertion below as
        // fail-loud so any future walker regression that produces an
        // exception edge without the pair surfaces immediately
        // instead of silently degrading to the regular `make_link`
        // path.
        let should_reraise = {
            let link_borrow = link.borrow();
            let Some(last_exception) = link_borrow.last_exception else {
                panic!(
                    "make_exception_link: link.last_exception is None \
                     (W-2 invariant: attach_catch_exception_edge must seed \
                     extravars per flowcontext.py:130-143)"
                );
            };
            let Some(last_exc_value) = link_borrow.last_exc_value else {
                panic!(
                    "make_exception_link: link.last_exc_value is None \
                     (W-2 invariant: attach_catch_exception_edge must seed \
                     extravars per flowcontext.py:130-143)"
                );
            };
            let target = link_borrow
                .target
                .clone()
                .expect("link target required for make_exception_link");
            target.borrow().operations.is_empty()
                && target.borrow().exits.is_empty()
                && link_borrow.args.len() == 2
                && link_borrow.args[0] == Some(last_exception.into())
                && link_borrow.args[1] == Some(last_exc_value.into())
        };
        if should_reraise {
            if handling_ovf {
                // `flatten.py:165-170` direct-OverflowError raise:
                //   exc_data = self.cpu.rtyper.exceptiondata
                //   ll_ovf = exc_data.get_standard_ll_exc_instance_by_class(
                //       OverflowError)
                //   c = Constant(ll_ovf, concretetype=lltype.typeOf(ll_ovf))
                //   self.emitline("raise", c)
                let cpu = self.cpu.expect(
                    "make_exception_link: handling_ovf=true requires a Cpu; \
                     production callers thread `CodeWriter::cpu()` so \
                     `cpu.rtyper.exceptiondata.\
                     get_standard_ll_exc_instance_by_class(OverflowError)` \
                     resolves per flatten.py:166-170",
                );
                let exc_data = &cpu.rtyper.exceptiondata;
                let ll_ovf = exc_data
                    .get_standard_ll_exc_instance_by_class("OverflowError")
                    .expect(
                        "ExceptionData::get_standard_ll_exc_instance_by_class\
                         (OverflowError) must succeed for the standard \
                         exception (flatten.py:167)",
                    );
                let operand = (self.lower_constant)(&ll_ovf);
                self.emitline(Insn::op("raise", vec![operand]));
            } else {
                self.emitline(Insn::op("reraise", Vec::new()));
            }
            self.emitline(Insn::Unreachable);
            return;
        }
        self.make_link(link, handling_ovf);
    }

    /// `rpython/jit/codewriter/flatten.py:189-204` `_ovf` rewrite of
    /// the canraise tail.
    ///
    /// Upstream:
    /// ```py
    /// if '_ovf' in opname:
    ///     line = self.popline()
    ///     self.emitline(opname[:7] + '_jump_if_ovf',
    ///                   TLabel(block.exits[1]), *line[1:])
    ///     assert len(block.exits) in (2, 3)
    ///     self.make_link(block.exits[0], False)
    ///     self.emitline(Label(block.exits[1]))
    ///     self.make_exception_link(block.exits[1], True)
    ///     if len(block.exits) == 3:
    ///         assert block.exits[2].exitcase is Exception
    ///         self.make_exception_link(block.exits[2], False)
    ///     return
    /// ```
    ///
    /// Pyre's `Insn::Op` carries `opname` separately from `args` /
    /// `result`, so `line[1:]` translates to "args + the trailing
    /// `->` result hint kept as `Insn::Op::result`".  The popped
    /// `_ovf` op's `opname[:7]` always yields the upstream "int_xxx"
    /// prefix (`int_add`, `int_sub`, `int_mul`, `int_neg`, …).
    fn flatten_ovf_canraise(&mut self, exits: &[LinkRef], raising_opname: &str) {
        // `flatten.py:194 line = self.popline()`.  The most recent
        // serialized op is the `_ovf` arithmetic itself.
        let line = self
            .popline()
            .expect("flatten_ovf_canraise: ssarepr.insns must contain the just-emitted _ovf op");
        let (popped_opname, popped_args, popped_result) = match line {
            Insn::Op {
                opname,
                args,
                result,
            } => (opname, args, result),
            other => panic!(
                "flatten_ovf_canraise: popline expected Insn::Op('{raising_opname}', ...), got {other:?}"
            ),
        };
        assert_eq!(
            popped_opname, raising_opname,
            "flatten_ovf_canraise: popped opname {popped_opname:?} disagrees with \
             block.raising_op() opname {raising_opname:?} — serialize_op \
             order is corrupted",
        );
        // `flatten.py:195-196` `opname[:7] + '_jump_if_ovf'`.
        assert!(
            popped_opname.len() >= 7,
            "flatten_ovf_canraise: opname {popped_opname:?} is shorter than the \
             upstream 7-char `int_xxx` prefix expected at flatten.py:195",
        );
        let jump_opname = format!("{}_jump_if_ovf", &popped_opname[..7]);
        // `flatten.py:196` `*line[1:]` — prepend the overflow target
        // TLabel, then keep all original args.  The result hint
        // (`Insn::Op::result`) stays on the new op since upstream's
        // `line[1:]` includes the trailing `'->', result` pair.
        let mut new_args = Vec::with_capacity(popped_args.len() + 1);
        new_args.push(self.tlabel_for_link(&exits[1]));
        new_args.extend(popped_args);
        let jump_insn = match popped_result {
            Some(result) => Insn::op_with_result(jump_opname, new_args, result),
            None => Insn::op(jump_opname, new_args),
        };
        self.emitline(jump_insn);
        // `flatten.py:197 assert len(block.exits) in (2, 3)`.
        assert!(
            matches!(exits.len(), 2 | 3),
            "flatten_ovf_canraise: _ovf canraise block must have 2 or 3 exits per \
             flatten.py:197 (got {})",
            exits.len(),
        );
        // `flatten.py:198 self.make_link(block.exits[0], False)`.
        self.make_link(&exits[0], false);
        // `flatten.py:199 self.emitline(Label(block.exits[1]))`.
        let exit1_label = self.label_for_link(&exits[1]);
        self.emitline(exit1_label);
        // `flatten.py:200 self.make_exception_link(block.exits[1], True)`.
        self.make_exception_link(&exits[1], true);
        if exits.len() == 3 {
            // `flatten.py:202 assert block.exits[2].exitcase is Exception`.
            // pyre represents the `Exception` catch-all by an exception
            // link with the extravars (`last_exception`, `last_exc_value`)
            // pair seeded by `attach_catch_exception_edge`, no typed
            // `llexitcase` (untyped catch-all), and arity-2 args mirroring
            // the reraise shape — together those distinguish the catch-all
            // from a normal-flow link (which has all three None) and from
            // a typed catch (which has `llexitcase = Some(case)`).
            let exit2 = exits[2].borrow();
            assert!(
                exit2.llexitcase.is_none(),
                "flatten_ovf_canraise: _ovf 3-exit canraise expects exits[2] to be \
                 the `Exception` catch-all (llexitcase=None) per flatten.py:202",
            );
            assert!(
                exit2.last_exception.is_some() && exit2.last_exc_value.is_some(),
                "flatten_ovf_canraise: _ovf 3-exit canraise expects exits[2] to be \
                 an exception link with extravars seeded per flowcontext.py:130-143 \
                 (matches `exitcase is Exception` invariant from flatten.py:202)",
            );
            drop(exit2);
            // `flatten.py:203 self.make_exception_link(block.exits[2], False)`.
            self.make_exception_link(&exits[2], false);
        }
    }

    fn insert_exits(&mut self, block: &BlockRef, handling_ovf: bool) {
        let exits = block.borrow().exits.clone();
        if exits.len() == 1 {
            // `flatten.py:181 assert link.exitcase in (None, False, True)`
            // — single-exit links carry either the default fall-through
            // marker (None) or a Bool case from a hand-hacked generator
            // graph (False/True).  Upstream's comment says "the cases
            // False or True should not really occur, but can show up in
            // the manually hacked graphs for generators…", so accept
            // them but fail loud on anything else.
            let link = &exits[0];
            let exitcase = link.borrow().exitcase.clone();
            match &exitcase {
                None => {}
                Some(super::flow::FlowValue::Constant(c)) => match &c.value {
                    super::flow::ConstantValue::Bool(_) => {}
                    other => panic!(
                        "flatten.py:181 invariant: single-exit link.exitcase \
                         must be None / False / True, got Constant({other:?})"
                    ),
                },
                Some(other) => panic!(
                    "flatten.py:181 invariant: single-exit link.exitcase \
                     must be None / False / True, got {other:?}"
                ),
            }
            self.make_link(link, handling_ovf);
            return;
        }
        if block.borrow().canraise() {
            // RPython `flatten.py:211` invariant: canraise blocks have
            // `exits[0]` = normal-flow link (exitcase=None,
            // last_exception=None, llexitcase=None); remaining exits
            // are catch cases.  The W-4 self-loop fix retired the
            // path where pyre's walker observed `[catch, normal]`
            // ordering at canraise blocks (the `[w-fallback w3/1709]`
            // 53 hits in raise_catch_loop pre-W-4); those hits dropped
            // to 0 across all 8 benches once the supersede-induced
            // catch-edge re-entries stopped landing before the normal
            // mergeblock.  Inline the assertion below as fail-loud so
            // any future walker regression that violates the
            // exits[0]=normal invariant surfaces immediately.
            // `flatten.py:211 assert block.exits[0].exitcase is None`.
            // Upstream's `flowcontext.py` guarantees the normal-flow
            // link is always exits[0] for canraise blocks; pyre's
            // `flatten.py:211` `assert exits[0].exitcase is None`.
            assert!(
                exits[0].borrow().exitcase.is_none(),
                "flatten.py:211 invariant: canraise block's exits[0] must \
                 be the normal-flow link (exitcase=None)"
            );
            let normal_link = exits[0].clone();
            // `flatten.py:189-204` `_ovf` rewrite.  When the last op
            // of a canraise block is an overflow-checked arithmetic
            // op (`int_add_ovf`, `int_sub_ovf`, `int_mul_ovf`,
            // `int_neg_ovf`, ...), pop the just-serialized op and
            // emit its `_jump_if_ovf` twin: the new op carries the
            // overflow-target TLabel as its first operand, followed
            // by the original op's args/result.  Then walk the normal
            // exit, emit `Label(exits[1])`, walk the OverflowError
            // edge with `handling_ovf=True`, and optionally walk the
            // `Exception` catch-all (`exits[2]`).
            //
            // The W-3 catch-link-order assertion below does NOT apply
            // to `_ovf` blocks because their `exits[1]` is the direct
            // OverflowError-reraise edge (not a typed catch), so check
            // for `_ovf` first and early-return before that assertion.
            let raising_opname = block
                .borrow()
                .raising_op()
                .map(|op| op.opname.clone())
                .unwrap_or_default();
            if raising_opname.contains("_ovf") {
                self.flatten_ovf_canraise(&exits, &raising_opname);
                return;
            }
            // RPython `flatten.py:223-238` invariant: typed catches
            // (`llexitcase = Some(case)`) precede the catch-all
            // (`llexitcase = None`); `flowcontext.py` enforces this
            // by graph construction (Exception catch-all always
            // emitted last).  Assert the order rather than re-sorting
            // — pyre's previous `catch_links.sort_by_key(llexitcase
            // .is_none())` was a normalizer for raise_catch_loop's
            // walker producing typed-then-all-then-typed shapes.  The
            // W-4 self-loop fix retired the supersede-induced catch-edge
            // re-entries that fed that shape; the order is now stable
            // out of the walker.
            // `flatten.py:223-238` walks the catch links in graph
            // order, breaking on the catch-all (`link.exitcase is
            // Exception`).  Upstream's `flowcontext.py` produces catch
            // links in typed-then-catch-all order at graph
            // construction time, so the iteration order is graph
            // order without sorting.
            let catch_links: Vec<LinkRef> = exits.iter().skip(1).cloned().collect();
            // `flatten.py:206-217` trailing `-live-` scan: walk
            // `block.operations` from the end skipping `-live-`
            // markers.  If the final op is NOT `-live-` (upstream's
            // `index == -1` after the loop), the canraise block does
            // not actually raise — emit the normal exit directly.
            // Without `_include_all_exc_links` this is the early
            // return path that upstream takes for canraise blocks
            // that survived metainterp policy but lack a real
            // raising-op-with-trailing-`-live-` pattern.
            let last_op_is_live = block
                .borrow()
                .operations
                .last()
                .map_or(false, |op| op.opname == OPNAME_LIVE);
            if !self.include_all_exc_links && !last_op_is_live {
                self.make_link(&normal_link, false);
                return;
            }
            let catch_label = self.tlabel_for_link(&normal_link);
            self.emitline(Insn::op("catch_exception", vec![catch_label]));
            self.make_link(&normal_link, false);
            let normal_label = self.label_for_link(&normal_link);
            self.emitline(normal_label);
            let mut captured_all = false;
            for link in &catch_links {
                let payload_shape = {
                    let link_borrow = link.borrow();
                    (
                        link_borrow.last_exception.is_some(),
                        link_borrow.last_exc_value.is_some(),
                    )
                };
                match payload_shape {
                    (false, false) => {
                        // Structural adaptation for pyre's bytecode-level
                        // graph builder.  RPython `flowcontext.py:130-156
                        // guessexception` closes a canraise block with
                        // `exits[0]` as the sole normal link and all
                        // following links seeded via `Link.extravars`.
                        // Pyre's PC-sequential walker can leave an
                        // additional normal/explicit-raise edge after
                        // the first slot.  Such a link is not an
                        // exception match arm; flatten it with ordinary
                        // `make_link` so final exceptblock targets still
                        // lower through `make_return`, and keep
                        // `make_exception_link` reserved for seeded
                        // exception links as upstream expects.
                        self.make_link(link, false);
                        continue;
                    }
                    (true, true) => {}
                    (last_exception, last_exc_value) => panic!(
                        "canraise catch link payload partially seeded: \
                         last_exception={last_exception}, last_exc_value={last_exc_value}",
                    ),
                }
                let llexitcase = link.borrow().llexitcase.clone();
                if let Some(case) = llexitcase {
                    let case_operand = self.getcolor(&case);
                    let mismatch_label = self.tlabel_for_link(link);
                    self.emitline(Insn::op(
                        "goto_if_exception_mismatch",
                        vec![case_operand, mismatch_label],
                    ));
                    self.make_exception_link(link, false);
                    let link_label = self.label_for_link(link);
                    self.emitline(link_label);
                } else {
                    self.make_exception_link(link, false);
                    captured_all = true;
                    break;
                }
            }
            if !captured_all {
                self.emitline(Insn::op("reraise", Vec::new()));
                self.emitline(Insn::Unreachable);
            }
            return;
        }
        let exitswitch = block.borrow().exitswitch.clone();
        if exits.len() == 2 && is_bool_or_tuple_exitswitch(&exits, &exitswitch) {
            let Some(exitswitch) = block.borrow().exitswitch.clone() else {
                panic!("flatten_graph: 2-exit block missing exitswitch");
            };
            let mut linkfalse = exits[0].clone();
            let mut linktrue = exits[1].clone();
            if linkfalse.borrow().llexitcase == Some(Constant::bool(true).into()) {
                std::mem::swap(&mut linkfalse, &mut linktrue);
            }
            let (opname, mut opargs) = match exitswitch {
                super::flow::ExitSwitch::Value(value) => {
                    ("goto_if_not".to_owned(), vec![self.getcolor(&value)])
                }
                super::flow::ExitSwitch::Tuple(values) => self.flatten_tuple_exitswitch(values),
            };
            opargs.push(self.tlabel_for_link(&linkfalse));
            self.emitline(Insn::live(Vec::new()));
            self.emitline(Insn::op(opname, opargs));
            self.make_link(&linktrue, handling_ovf);
            let false_label = self.label_for_link(&linkfalse);
            self.emitline(false_label);
            self.make_link(&linkfalse, handling_ovf);
            return;
        }
        self.insert_switch_exits(&exits, exitswitch, handling_ovf);
    }

    fn flatten_tuple_exitswitch(
        &mut self,
        values: Vec<super::flow::ExitSwitchElement>,
    ) -> (String, Vec<Operand>) {
        let mut iter = values.into_iter();
        let opname = match iter.next() {
            Some(super::flow::ExitSwitchElement::Marker(name)) => {
                format!("goto_if_not_{name}")
            }
            other => panic!("flatten_graph: tuple exitswitch missing opname marker: {other:?}"),
        };
        let mut values: Vec<_> = iter.collect();
        if matches!(
            values.last(),
            Some(super::flow::ExitSwitchElement::Marker(marker)) if marker == "-live-before"
        ) {
            values.pop();
        }
        let args = values
            .into_iter()
            .map(|value| match value {
                super::flow::ExitSwitchElement::Value(value) => self.getcolor(&value),
                super::flow::ExitSwitchElement::Marker(marker) => {
                    panic!("flatten_graph: unexpected tuple exitswitch marker {marker:?}")
                }
            })
            .collect();
        (opname, args)
    }

    fn insert_switch_exits(
        &mut self,
        exits: &[LinkRef],
        exitswitch: Option<super::flow::ExitSwitch>,
        handling_ovf: bool,
    ) {
        // Diagnostic: surface walker-emitted switch shape on entry so
        // the panic at `switch_llexitcase_key` includes the link
        // inventory (exitcase + llexitcase per link) rather than
        // forcing manual graph inspection.
        if std::env::var_os("PYRE_PHASE3_SWITCH_DIAG").is_some() {
            let exitcase_summary: Vec<String> = exits
                .iter()
                .map(|link| {
                    let lb = link.borrow();
                    format!(
                        "(exitcase={:?}, llexitcase={:?})",
                        lb.exitcase, lb.llexitcase
                    )
                })
                .collect();
            // Describe the source block's last few operations so we
            // can identify which walker emit site produced this shape.
            let block_summary = exits
                .first()
                .and_then(|link| link.borrow().prevblock.clone())
                .and_then(|weak| weak.upgrade())
                .map(|block_rc| {
                    let b = block_rc.borrow();
                    let last_ops: Vec<String> = b
                        .operations
                        .iter()
                        .rev()
                        .take(3)
                        .rev()
                        .map(|op| format!("{}@{}", op.opname, op.offset))
                        .collect();
                    format!("ops={} last_ops={:?}", b.operations.len(), last_ops,)
                })
                .unwrap_or_else(|| "<no-block>".to_string());
            eprintln!(
                "[phase3-switch-diag] insert_switch_exits: exits={} \
                 exitswitch={:?} {} shape={:?}",
                exits.len(),
                exitswitch,
                block_summary,
                exitcase_summary,
            );
        }
        let Some(super::flow::ExitSwitch::Value(exitswitch)) = exitswitch else {
            // RPython `flatten.py:282-309 insert_switch_exits` is only
            // called via `insert_exits` when the block already has a
            // Variable exitswitch (`flatten.py:107-115` dispatch by
            // `exits.len()` + `exitswitch.concretetype`).  A None
            // exitswitch on a multi-exit block is a malformed graph
            // shape upstream would never produce; fail loud so the
            // walker non-orthodoxy that produced it is visible rather
            // than silently materialising bytes.
            let exitcase_summary: Vec<String> = exits
                .iter()
                .map(|link| {
                    let lb = link.borrow();
                    format!("(llexitcase={:?})", lb.llexitcase)
                })
                .collect();
            panic!(
                "flatten.py:282 insert_switch_exits invariant: \
                 multi-exit block must carry a Variable exitswitch, got \
                 None on {} exits = {:?}",
                exits.len(),
                exitcase_summary,
            );
        };
        // `flatten.py:275-276` `kind = getkind(block.exitswitch.concretetype)
        // assert kind == 'int'    # XXX` — upstream enforces that a switch
        // dispatches on a Signed/Int Variable.  Pyre's `Variable.kind` is
        // already typed; assert structurally here so a walker non-orthodoxy
        // that produced a non-Int exitswitch surfaces at flatten time
        // rather than at runtime (or via a SwitchDictDescr key shape
        // mismatch downstream).
        let switch_kind = match &exitswitch {
            FlowValue::Variable(variable) => variable.kind,
            FlowValue::Constant(constant) => constant.kind,
        };
        assert!(
            matches!(switch_kind, Some(Kind::Int)),
            "flatten.py:275-276 invariant: switch exitswitch must be \
             Int-kinded (got {switch_kind:?})"
        );
        let mut switches: Vec<LinkRef> = exits
            .iter()
            .filter(|link| !is_default_exitcase(&link.borrow().exitcase))
            .cloned()
            .collect();
        switches.sort_by_key(|link| switch_llexitcase_key(&link.borrow().llexitcase));

        let mut switchdict = SwitchDictDescr::new();
        for switch in &switches {
            let key = switch_llexitcase_key(&switch.borrow().llexitcase);
            switchdict
                .labels
                .push((key, self.tlabel_value_for_link(switch)));
        }

        let switch_value = self.getcolor(&exitswitch);
        self.emitline(Insn::live(Vec::new()));
        self.emitline(Insn::op(
            "switch",
            vec![
                switch_value,
                Operand::descr(DescrOperand::SwitchDict(switchdict)),
            ],
        ));
        if let Some(default_link) = exits
            .last()
            .filter(|link| is_default_exitcase(&link.borrow().exitcase))
        {
            self.make_link(default_link, handling_ovf);
        } else {
            self.emitline(Insn::op("unreachable", Vec::new()));
            self.emitline(Insn::Unreachable);
        }
        for switch in switches {
            let link_label = self.label_for_link(&switch);
            self.emitline(link_label);
            self.emitline(Insn::live(Vec::new()));
            self.make_link(&switch, handling_ovf);
        }
    }

    fn insert_renamings(&mut self, link: &LinkRef) {
        let (target_inputargs, last_exception, last_exc_value, link_args) = {
            let link_borrow = link.borrow();
            let target = link_borrow
                .target
                .clone()
                .expect("link target required for insert_renamings");
            (
                target.borrow().inputargs.clone(),
                link_borrow.last_exception,
                link_borrow.last_exc_value,
                link_borrow.args.clone(),
            )
        };

        let mut pairs: Vec<(RenameOperand, Register)> = Vec::new();
        for (index, arg) in link_args.iter().enumerate() {
            let Some(src_value) = arg.as_ref() else {
                continue;
            };
            let Some(dst_variable) = target_inputargs[index].as_variable() else {
                continue;
            };
            let src_variable = src_value.as_variable();
            if src_variable == last_exception || src_variable == last_exc_value {
                continue;
            }
            let src = self.rename_operand(src_value);
            let dst = (self.get_register)(dst_variable);
            if src == RenameOperand::Register(dst) {
                continue;
            }
            pairs.push((src, dst));
        }
        pairs.sort_by_key(|(_, dst)| dst.index);

        // `[T; 3]` indexed by `Kind::index()` per [[feedback-no-hashmap-ever]].
        // Mirrors `rpython/jit/codewriter/flatten.py:306-334 insert_renamings`
        // which keys by kind string in a Python dict.
        let mut renamings: [(Vec<RenameOperand>, Vec<RenameOperand>); 3] = [
            (Vec::new(), Vec::new()),
            (Vec::new(), Vec::new()),
            (Vec::new(), Vec::new()),
        ];
        for (src, dst) in pairs {
            let (frm, to) = &mut renamings[dst.kind.index()];
            frm.push(src);
            to.push(RenameOperand::Register(dst));
        }
        for &kind in &Kind::ALL {
            let (frm, to) = &renamings[kind.index()];
            if frm.is_empty() {
                continue;
            }
            for (src, dst) in reorder_renaming_list(frm, to) {
                match (src, dst) {
                    (Some(src), Some(RenameOperand::Register(dst))) => {
                        self.emitline(Insn::op_with_result(
                            format!("{}_copy", kind.as_str()),
                            vec![src.into_operand()],
                            dst,
                        ));
                    }
                    (Some(RenameOperand::Register(src)), None) => {
                        self.emitline(Insn::op(
                            format!("{}_push", kind.as_str()),
                            vec![Operand::Register(src)],
                        ));
                    }
                    (None, Some(RenameOperand::Register(dst))) => {
                        self.emitline(Insn::op_with_result(
                            format!("{}_pop", kind.as_str()),
                            Vec::new(),
                            dst,
                        ));
                    }
                    other => panic!("unexpected renaming step {other:?}"),
                }
            }
        }
        let link_borrow = link.borrow();
        self.generate_last_exc(&link_borrow, &target_inputargs);
    }

    fn generate_last_exc(&mut self, link: &super::flow::Link, inputargs: &[FlowValue]) {
        if link.last_exception.is_none() && link.last_exc_value.is_none() {
            return;
        }
        for (arg, inputarg) in link.args.iter().zip(inputargs) {
            if arg.as_ref().and_then(FlowValue::as_variable) == link.last_exception {
                let dst = inputarg
                    .as_variable()
                    .expect("last_exception target must be a Variable");
                let dst_reg = (self.get_register)(dst);
                self.emitline(Insn::op_with_result("last_exception", Vec::new(), dst_reg));
            }
        }
        for (arg, inputarg) in link.args.iter().zip(inputargs) {
            if arg.as_ref().and_then(FlowValue::as_variable) == link.last_exc_value {
                let dst = inputarg
                    .as_variable()
                    .expect("last_exc_value target must be a Variable");
                let dst_reg = (self.get_register)(dst);
                self.emitline(Insn::op_with_result("last_exc_value", Vec::new(), dst_reg));
            }
        }
    }

    /// `rpython/jit/codewriter/flatten.py:88-100 enforce_input_args` —
    /// rotate startblock inputarg colors so each kind's inputargs occupy
    /// the canonical `0..n-1` slots, swapping any current allocation
    /// into the inputarg's target slot.  Upstream stores `regallocs` as
    /// `self.regallocs`; pyre threads it through the call because the
    /// `get_register` closure already encapsulates regallocs access for
    /// the rest of the flatten pass.  The two-callsite shape is a
    /// pre-Phase-4 adaptation; once the walker stops computing
    /// regallocs externally, both reduce to one canonical call.
    pub fn enforce_input_args(
        &mut self,
        graph: &super::flow::FunctionGraph,
        regallocs: &mut [super::regalloc::GraphAllocationResult; 3],
    ) {
        super::regalloc::enforce_input_args_simulation(graph, regallocs);
    }

    /// `rpython/jit/codewriter/flatten.py:102-104 generate_ssa_form` —
    /// reset `seen_blocks` and recurse from `graph.startblock`.  Upstream
    /// stores `graph` as `self.graph`; pyre threads it through because
    /// `make_bytecode_block` operates on the block argument and does not
    /// need a graph backreference for any other step.
    pub fn generate_ssa_form(&mut self, graph: &super::flow::FunctionGraph) {
        self.seen_blocks.clear();
        self.make_bytecode_block(graph.startblock.clone(), false);
    }

    fn make_bytecode_block(&mut self, block: BlockRef, handling_ovf: bool) {
        if block.borrow().exits.is_empty() {
            // `rpython/jit/codewriter/flatten.py:107-109`: empty-exits
            // blocks are `returnblock` (1 arg) or `exceptblock` (2 args)
            // and `make_return` handles both shapes.  Any other arg
            // count is a walker non-orthodoxy that fails fail-loud
            // inside `make_return` (upstream raises `Exception("?")` at
            // `flatten.py:145`).  Pyre keeps the upstream behavior:
            // delegate to `make_return` directly.
            let args = block.borrow().inputargs.clone();
            self.make_return(&args);
            return;
        }
        if self.seen_blocks.contains(&block) {
            let target = self.tlabel_for_block(&block);
            self.emitline(Insn::op("goto", vec![target]));
            self.emitline(Insn::Unreachable);
            return;
        }
        self.seen_blocks.push(block.clone());
        let block_label = self.label_for_block(&block);
        self.emitline(block_label);
        // Pyre adaptation: emit `Insn::PcAnchor { py_pc }` +
        // placeholder `-live-` marker pairs INTERLEAVED with the
        // block's ops based on `op.offset` (= py_pc the walker
        // recorded the op at).  Walker emits PcAnchor + `-live-`
        // immediately before dispatching each PC's bytecode
        // (`codewriter.rs::emit_mark_label_pc!` followed by
        // `emit_live_placeholder!()`), so the `-live-` placeholder
        // sits right before that PC's ops.  `compute_liveness`'s
        // backward walk captures the alive register set AT that
        // marker position; bulk-emitting all anchors at block entry
        // (the previous behaviour) collapses every PC's marker to
        // "block entry" and produces a single alive set shared
        // across all PCs — wrong liveness for any PC after the
        // first.  Per-op interleaving here reconstructs the walker's
        // per-PC marker positions so `filter_liveness_in_place`'s
        // post-pass populates each PC's `-live-` with the correct
        // alive set.  Owned PCs without any matching op (sparse
        // walker emissions for synthetic-only PCs) emit a trailing
        // PA + L pair after the op loop so `pc_anchor_positions`'s
        // assert that every PC has an anchor still holds.  Synthetic
        // graph ops (`offset == -1`, e.g. `abort_permanent`,
        // `emit_vsd!` int_copy / setfield_vable_i bookkeeping)
        // attach to whichever PC was most recently anchored, matching
        // walker's emit-where-encountered behaviour.  Upstream
        // RPython has no per-PC anchor concept; this is a pyre-only
        // extension required by per-PC runtime dispatch.
        let owned_pcs_sorted: Vec<usize> = {
            let mut v: Vec<usize> = self
                .block_py_pcs
                .iter()
                .filter_map(|(b, pc)| if b == &block { Some(*pc) } else { None })
                .collect();
            v.sort();
            v.dedup();
            v
        };
        let force_alive = self.live_force_alive_ops.clone();
        let operations = block.borrow().operations.clone();
        let exits_len = block.borrow().exits.len();
        let exitswitch_is_last_exception = block.borrow().canraise();
        let mut anchored_pcs: Vec<usize> = Vec::new();
        let mut current_pc: Option<usize> = None;
        for op in &operations {
            // Determine whether this op carries an owned-PC offset.
            // `op.offset >= 0` filters out synthetic ops (offset=-1)
            // which attach to the most recently anchored PC.  Ops
            // whose offset isn't in `owned_pcs_sorted` (rare — a
            // graph op recorded with a non-owned PC) also attach to
            // the current PC.
            let target_pc: Option<usize> = if op.offset >= 0 {
                let off = op.offset as usize;
                if owned_pcs_sorted.contains(&off) {
                    Some(off)
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(pc) = target_pc {
                if Some(pc) != current_pc {
                    // Emit anchors for all owned PCs at-or-before
                    // `pc` that haven't been emitted yet.  Catches
                    // "empty" PCs sitting between the previously
                    // anchored PC and `pc` (PCs whose walker emit
                    // contributed no real graph op) so every owned
                    // PC ends up anchored in ascending order.
                    for &p in &owned_pcs_sorted {
                        if p <= pc && !anchored_pcs.contains(&p) {
                            self.emitline(Insn::pc_anchor(p));
                            self.emitline(Insn::op(OPNAME_LIVE.to_string(), force_alive.clone()));
                            anchored_pcs.push(p);
                        }
                    }
                    current_pc = Some(pc);
                }
            } else if current_pc.is_none() {
                // Synthetic op precedes any real-PC op in this
                // block.  Emit the first owned PC's anchor so the
                // synthetic op attaches to it (matches walker's
                // behaviour of always opening a block with PA + L).
                if let Some(&first_pc) = owned_pcs_sorted.first() {
                    self.emitline(Insn::pc_anchor(first_pc));
                    self.emitline(Insn::op(OPNAME_LIVE.to_string(), force_alive.clone()));
                    anchored_pcs.push(first_pc);
                    current_pc = Some(first_pc);
                }
            }
            // `flatten.py:120-125` `_ovf` validity check: an overflow-
            // checked op must live in a canraise block with 2 or 3
            // exits; otherwise the rtyper-side guarantee that an
            // `OverflowError` is caught fails to hold, and the
            // `_jump_if_ovf` rewrite in `flatten_ovf_canraise`
            // (`flatten.py:189-204`) would have no overflow target.
            if op.opname.contains("_ovf") {
                assert!(
                    matches!(exits_len, 2 | 3) && exitswitch_is_last_exception,
                    "detected a block containing ovfcheck() but no \
                     OverflowError is caught, this is not legal in \
                     jitted blocks (op={op_name}, exits={exits_len}, \
                     canraise={exitswitch_is_last_exception}) — \
                     flatten.py:122-125",
                    op_name = op.opname,
                );
            }
            self.serialize_op(op);
        }
        // Emit anchors for any owned PCs that didn't appear in any
        // op's `offset` (e.g. PCs whose walker emit produced only
        // scaffold — PA + `-live-` — and no graph SpaceOp).
        // `pc_anchor_positions` asserts every PC has an anchor; this
        // tail emission satisfies that invariant.  These trailing
        // anchors sit between the block's last op and `insert_exits`'s
        // terminator emission — runtime entry into them would no-op
        // through the terminator's goto / return because they carry
        // no preceding ops in this block.
        for &p in &owned_pcs_sorted {
            if !anchored_pcs.contains(&p) {
                self.emitline(Insn::pc_anchor(p));
                self.emitline(Insn::op(OPNAME_LIVE.to_string(), force_alive.clone()));
            }
        }
        self.insert_exits(&block, handling_ovf);
    }

    fn flatten_space_operation(&mut self, op: &SpaceOperation) -> Insn {
        // If the GraphFlattener was constructed with a `LoweringContext`,
        // retired-family HLOp opnames (`add` / `lt` / ... / `bool` /
        // `setitem`) lower to the matching `residual_call_*` Insn shape
        // via the dispatcher.  Non-HLOp opnames return `None` from the
        // dispatcher and fall through to the legacy opname-passthrough
        // below.
        if let Some(ref ctx) = self.lowering_ctx {
            // Borrow-split: the dispatcher needs `&mut self.get_register`
            // and `&mut self.lower_constant` simultaneously, which Rust
            // accepts only when the two field accesses don't alias.
            let Self {
                get_register,
                lower_constant,
                ..
            } = self;
            if let Some(insn) =
                try_flatten_retired_family_hlop_to_insn(op, ctx, get_register, lower_constant)
            {
                return insn;
            }
        }
        let args = self.flatten_list(&op.args);
        match op.result {
            None => Insn::op(op.opname.clone(), args),
            Some(FlowValue::Variable(variable)) => {
                if variable.kind.is_none() {
                    return Insn::op(op.opname.clone(), args);
                }
                let result = (self.get_register)(variable);
                Insn::op_with_result(op.opname.clone(), args, result)
            }
            Some(FlowValue::Constant(ref constant)) => {
                panic!(
                    "GraphFlattener: op {} has Constant result {:?}; \
                     flow graph results must be Variables",
                    op.opname, constant
                )
            }
        }
    }

    fn flatten_arg(&mut self, arg: &SpaceOperationArg) -> Operand {
        match arg {
            SpaceOperationArg::Value(value) => self.getcolor(value),
            SpaceOperationArg::ListOfKind(list) => Operand::ListOfKind(ListOfKind::new(
                list.kind,
                list.content
                    .iter()
                    .map(|value| self.getcolor(value))
                    .collect(),
            )),
            // `flatten.py:365-367` passes AbstractDescr through
            // unchanged.  Pyre routes the `DescrByPtr` to the matching
            // `DescrOperand` variant via singleton `Arc::ptr_eq` —
            // see `flatten_descr_by_ptr`.
            SpaceOperationArg::Descr(descr_by_ptr) => flatten_descr_by_ptr(descr_by_ptr),
            // `flatten.py:365-367` also passes IndirectCallTargets
            // through unchanged.  `Operand::IndirectCallTargets` takes a
            // value, so clone the inner (the `Vec<Arc<JitCode>>` clone
            // is cheap — it bumps Arc refcounts).
            SpaceOperationArg::IndirectCallTargets(targets) => {
                Operand::IndirectCallTargets((*targets.0).clone())
            }
        }
    }

    /// `flatten.py:355-371 flatten_list(arglist)` — iterate `arglist`
    /// and dispatch each arg via `flatten_arg`.  Upstream inlines the
    /// per-variant dispatch inside `flatten_list`; pyre splits the
    /// per-arg dispatch into `flatten_arg` for reuse (e.g. by
    /// `flatten_op_to_insn` via `flatten_arg_with_lowering`).  The
    /// named wrapper matches upstream's `serialize_op` call shape
    /// (`args = self.flatten_list(op.args)`).
    fn flatten_list(&mut self, arglist: &[SpaceOperationArg]) -> Vec<Operand> {
        arglist.iter().map(|arg| self.flatten_arg(arg)).collect()
    }

    /// `flatten.py:382-391 GraphFlattener.getcolor(v)`: Variable →
    /// Register (via the regallocs coloring), Constant → pass-through
    /// lowered Operand.  Upstream returns `v` unchanged for the
    /// Constant case; pyre's typed `Operand` enum requires explicit
    /// lowering via `lower_constant` (a closure that resolves opaque
    /// pycode / jitdriver pointers — see `GraphFlattener.lower_constant`
    /// docstring).
    fn getcolor(&mut self, value: &FlowValue) -> Operand {
        match value {
            FlowValue::Variable(variable) => Operand::Register((self.get_register)(*variable)),
            FlowValue::Constant(constant) => (self.lower_constant)(constant),
        }
    }
}

fn is_bool_or_tuple_exitswitch(
    exits: &[LinkRef],
    exitswitch: &Option<super::flow::ExitSwitch>,
) -> bool {
    // `flatten.py:240-242` upstream check: `block.exitswitch` is a
    // tuple (jtransform-fused `goto_if_not_<opname>` form) OR
    // `block.exitswitch.concretetype == lltype.Bool` (post-rtype bool
    // exitswitch).  Pyre collapses upstream's `Bool` and `Signed`
    // lltypes into a single `Kind::Int`, so the second clause can't
    // be expressed as a direct kind comparison — instead, require
    // BOTH exit links to carry a Bool `llexitcase` (a True/False pair
    // populated by `set_last_bool_exitcase` for POP_JUMP_IF_FALSE /
    // POP_JUMP_IF_TRUE walker branches).  Requiring both (stricter
    // than `any`) prevents malformed 2-exit graphs from silently
    // falling into the bool-branch path with only one Bool exitcase.
    matches!(exitswitch, Some(super::flow::ExitSwitch::Tuple(_)))
        || exits
            .iter()
            .all(|link| is_bool_exitcase(&link.borrow().llexitcase))
}

fn is_bool_exitcase(exitcase: &Option<FlowValue>) -> bool {
    matches!(
        exitcase,
        Some(FlowValue::Constant(Constant {
            value: ConstantValue::Bool(_),
            ..
        }))
    )
}

fn is_default_exitcase(exitcase: &Option<FlowValue>) -> bool {
    // Upstream uses the string sentinel `"default"` for the catch-all
    // switch link (`flatten.py:280 if link.exitcase != 'default':`).
    // RPython parity: ONLY `Str("default")` counts as the catch-all;
    // `None` is a malformed shape upstream would never produce and
    // must surface from `switch_llexitcase_key` as a fail-loud panic
    // (caller `insert_switch_exits` filters defaults before the
    // Signed-key extraction).  Any `None` reaching this point would
    // be a walker non-orthodoxy that needs fixing at the graph-
    // construction site rather than papered over here — either set
    // `exitcase = Str("default")` for a true switch catch-all, or
    // make both bool-branch links carry a `Bool` `llexitcase` so the
    // bool-branch path (`flatten.rs:1761 is_bool_or_tuple_exitswitch`)
    // fires instead of the switch path.
    matches!(
        exitcase,
        Some(FlowValue::Constant(Constant {
            value: ConstantValue::Str(value),
            ..
        })) if value == "default"
    )
}

/// `rpython/jit/codewriter/flatten.py:296 lltype.cast_primitive(
/// lltype.Signed, switch.llexitcase)`.
///
/// RPython's switch path runs only after `flatten.py:275 kind ==
/// 'int'` asserts — Bool exitswitches go through the bool-branch
/// path at `flatten.py:240-267`, not this switch path.  So the
/// switch llexitcase keys are Signed only; `cast_primitive(Signed,
/// switch.llexitcase)` is effectively an identity cast.  Pyre's
/// previous Bool-widening clause was added when Bool exitcase pairs
/// fell through to switch handling, but that path no longer fires
/// (the stricter `is_bool_or_tuple_exitswitch` requires BOTH exits
/// to carry Bool llexitcase, so a partial-Bool 2-exit shape goes
/// to the switch path which now fails loud on the Bool).
fn switch_llexitcase_key(llexitcase: &Option<FlowValue>) -> i64 {
    match llexitcase {
        Some(FlowValue::Constant(Constant {
            value: ConstantValue::Signed(value),
            ..
        })) => *value,
        // `flatten.py:283` — `kind == 'int'` includes lltype `Bool`
        // (Bool is an Int subtype upstream); `cast_primitive(Signed,
        // link.llexitcase)` coerces False → 0, True → 1.  Pyre collapses
        // upstream's Bool/Signed lltypes into `Kind::Int` and represents
        // bool exitcases as `Constant(Bool(true|false))`, so accept and
        // coerce here matching the upstream semantics.
        Some(FlowValue::Constant(Constant {
            value: ConstantValue::Bool(value),
            ..
        })) => i64::from(*value),
        other => panic!(
            "flatten_graph: switch link requires Signed/Bool llexitcase per \
             flatten.py:275-296 (kind == 'int' assert + cast_primitive); \
             got {other:?}"
        ),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RenameOperand {
    Register(Register),
    ConstInt(i64),
    ConstRef(i64),
    ConstFloat(i64),
}

impl RenameOperand {
    fn into_operand(self) -> Operand {
        match self {
            Self::Register(register) => Operand::Register(register),
            Self::ConstInt(value) => Operand::ConstInt(value),
            Self::ConstRef(value) => Operand::ConstRef(value),
            Self::ConstFloat(value) => Operand::ConstFloat(value),
        }
    }
}

fn flatten_constant_operand(constant: &super::flow::Constant) -> Operand {
    match (&constant.value, constant.kind) {
        (ConstantValue::None, Some(Kind::Ref)) => Operand::ConstRef(0),
        (ConstantValue::Bool(value), Some(Kind::Int)) => Operand::ConstInt(i64::from(*value)),
        (ConstantValue::Signed(value), Some(Kind::Int)) => Operand::ConstInt(*value),
        // RPython rtyper post-pass: a `Constant(ll_ptr, concretetype=
        // lltype.Ptr(...))` carries an integer pointer in its `value`
        // field with a `Ptr` concretetype.  Pyre's canonical
        // `flatten_graph` consumes graphs that already carry this
        // post-rtype shape (the production walker emits Signed(Ref)
        // directly for resolved exception classes via `ExceptionData::
        // get_standard_ll_exc_instance_by_class`, and other Opaque(Ref)
        // sites resolve through the closure-based
        // `flatten_graph_with_lowering` path used today).
        (ConstantValue::Signed(value), Some(Kind::Ref)) => Operand::ConstRef(*value),
        // Pre-rtype Python string constant.  Upstream RPython's
        // `rtyper:specialize` substep interns the string into the static
        // data section and rewrites the Constant to a
        // `(Signed(addr), Kind::Ref)` post-rtype shape.  Pyre lacks the
        // rtyper today, so the lowered Insn carries a placeholder
        // `ConstRef(0)`.  Production callers route LOAD_ATTR /
        // STORE_ATTR through `emit_abort_permanent!` (codewriter.rs
        // ~6735, ~6756), so the placeholder is never executed.  When
        // pyre's string-intern infrastructure lands and LOAD_ATTR
        // production-flips, swap this arm for `Operand::ConstRef
        // (intern_pool.resolve(s) as i64)`.
        (ConstantValue::Str(_), Some(Kind::Ref)) => Operand::ConstRef(0),
        (ConstantValue::Opaque(_), Some(Kind::Ref)) => {
            panic!(
                "GraphFlattener: opaque ref constants must be resolved \
                 before the canonical flatten_graph driver — production \
                 callers run flatten_graph_with_lowering with a per-call \
                 lower_constant closure (matches rpython/rtyper/rtyper.py\
                 :specialize substep)"
            )
        }
        other => panic!("GraphFlattener: unsupported constant operand {other:?}"),
    }
}

/// Test-fixture lowering: same as [`flatten_constant_operand`] but
/// returns a placeholder for `Opaque(Ref)` instead of panicking.
///
/// Test fixtures construct synthetic `SpaceOperation` shapes whose
/// `Opaque(Ref)` constants don't have a real PyObject pointer; production
/// callers thread their per-call `lower_constant` closure
/// (`codewriter.rs::transform_graph_to_jitcode`) to recover the real
/// `w_code` pointer.  Tests pass this placeholder so they don't need a
/// production-grade closure.  Tests that compare two SSARepr streams
/// only compare opname + register kinds, so the `ConstRef(0)` value
/// doesn't matter for those assertions.
pub(super) fn flatten_constant_operand_for_test(constant: &super::flow::Constant) -> Operand {
    match (&constant.value, constant.kind) {
        (ConstantValue::Opaque(_), Some(Kind::Ref)) => Operand::ConstRef(0),
        _ => flatten_constant_operand(constant),
    }
}

/// Test-fixture entry mirroring `rpython/jit/codewriter/flatten.py:63
/// flatten_graph(graph, regallocs)` — wraps `GraphFlattener::
/// generate_ssa_form` for callers that already have closure-shaped
/// `get_register` / `lower_constant` rather than a `regallocs`
/// HashMap.  Production callers should use the canonical
/// [`flatten_graph`] entry which carries the full `flatten.py:63-70`
/// shape (regallocs swap + cpu + lowering_ctx).
///
/// The `enforce_input_args` step (`flatten.py:68`) is skipped because
/// callers supply `get_register` directly: the closure projects
/// whatever color scheme the test fixture computed, so no
/// inputarg-color rotation is needed.
pub fn flatten_graph_with_closures<F, C>(
    graph: &super::flow::FunctionGraph,
    ssarepr: &mut SSARepr,
    get_register: F,
    lower_constant: C,
) where
    F: FnMut(Variable) -> Register,
    C: FnMut(&Constant) -> Operand,
{
    let mut flattener =
        GraphFlattener::new_with_constant_lowering(ssarepr, get_register, lower_constant);
    // `flatten.py:69 flattener.generate_ssa_form()`.
    flattener.generate_ssa_form(graph);
}

/// Test-fixture variant of [`flatten_graph`] that takes
/// `get_register` and `lower_constant` as explicit closures and a
/// pre-built `LoweringContext`, rather than reading them off
/// `regallocs` and `cpu`.  Used by the retired-family HLOp lowering
/// tests where constructing a real `Cpu` and `regallocs` HashMap is
/// disproportionate to the test's scope.
///
/// Production callers should use [`flatten_graph`] (the canonical
/// entry matching `flatten.py:63-70`); this helper exists only to
/// keep test fixtures concise.
pub fn flatten_graph_with_lowering<'a, F, C>(
    graph: &super::flow::FunctionGraph,
    ssarepr: &'a mut SSARepr,
    lowering_ctx: LoweringContext,
    cpu: Option<&'a super::cpu::Cpu>,
    get_register: F,
    lower_constant: C,
) where
    F: FnMut(Variable) -> Register,
    C: FnMut(&Constant) -> Operand,
{
    let mut flattener =
        GraphFlattener::new_with_full_lowering(ssarepr, get_register, lower_constant, lowering_ctx);
    if let Some(cpu) = cpu {
        flattener = flattener.with_cpu(cpu);
    }
    // `flatten.py:69 flattener.generate_ssa_form()`.
    flattener.generate_ssa_form(graph);
}

/// `rpython/jit/codewriter/flatten.py:63-70 flatten_graph(graph,
/// regallocs, _include_all_exc_links=False, cpu=None)`.
///
/// The canonical entry point matching upstream signature exactly.
/// Constructs the `SSARepr` internally, derives `get_register` from
/// `regallocs[kind].getcolor(v)` (`flatten.py:382-391`), and threads
/// `cpu` through `make_exception_link` for `handling_ovf=True` reraise
/// targets (`flatten.py:166-170`).
///
/// **`LoweringContext` derivation.**
/// Upstream's `flatten_graph` does not take a `LoweringContext` because
/// the rtyper rewrites the graph to post-rtype shape BEFORE flatten_graph
/// runs.  Pyre's graph carries pre-rtype HLOps (BINARY_OP / COMPARE_OP /
/// BOOL / SETITEM families) directly; the canonical entry derives the
/// dispatcher's `LoweringContext` from `cpu.lowering_ctx` (a pyre
/// extension on `Cpu`) so the upstream `flatten_graph(graph, regallocs,
/// _include_all_exc_links, cpu)` signature stays intact.
///
/// **Upstream-orthodox entry** matching
/// `rpython/jit/codewriter/flatten.py:63-70 flatten_graph(graph,
/// regallocs, _include_all_exc_links=False, cpu=None)` signature with
/// no additional parameters.  Derives the dispatcher's
/// `LoweringContext` from `cpu.lowering_ctx` (a pyre-specific
/// extension on `Cpu` documented at `cpu.rs::Cpu::lowering_ctx`).
/// When `cpu` is `None` or `cpu.lowering_ctx` is unset, the HLOp
/// dispatcher is disabled and pre-rtype HLOp opnames passthrough to
/// the SSARepr — useful for tests against structural-only graphs.
pub fn flatten_graph<'a>(
    graph: &super::flow::FunctionGraph,
    regallocs: &'a mut [super::regalloc::GraphAllocationResult; 3],
    include_all_exc_links: bool,
    cpu: Option<&'a super::cpu::Cpu>,
) -> SSARepr {
    flatten_graph_with_walker_slots(
        graph,
        regallocs,
        &[],
        &[],
        &[],
        &[],
        &[],
        include_all_exc_links,
        cpu,
    )
}

/// Phase 4 bridge: `walker_slot_for_variable[var.id]` overrides the
/// graph regalloc color so production can splice canonical Insns into
/// the walker's SSARepr without Register-index mismatch.  Synthetic
/// graph-only Variables (e.g. `last_exception` from
/// `attach_catch_exception_edge`) have no entry and fall back to
/// `regallocs[kind].getcolor(v)` per `flatten.py:382-391`.
///
/// Task #50 T6 epic slice 1 — `block_name_overrides` pre-seeds the
/// flattener's `block_names` map so canonical's `Label(block)` /
/// `tlabel_for_block(target)` emit walker-compatible `pc{N}` names
/// (where N = block's FrameState `next_offset` = entry Python PC)
/// instead of the default sequential `block{N}` scheme.  Walker emits
/// `Label("pcN")` per Python PC (its per-PC dispatch convention);
/// without the override canonical and walker disagree on every branch
/// target label name even though they reference the same target
/// block.  Pre-seeded overrides are first-wins per
/// `label_name_for_block`'s lookup logic — entries not in the
/// override list fall back to the `block{N}` default (used for blocks
/// the walker created but never assigned a `pcN` to, e.g. the implicit
/// graph startblock when its py_pc has no per-PC anchor).
pub fn flatten_graph_with_walker_slots<'a>(
    graph: &super::flow::FunctionGraph,
    regallocs: &'a mut [super::regalloc::GraphAllocationResult; 3],
    walker_slot_for_variable: &'a [Option<u16>],
    block_name_overrides: &[(super::flow::BlockRef, String)],
    link_name_overrides: &[(super::flow::LinkRef, String)],
    block_py_pc_overrides: &[(super::flow::BlockRef, usize)],
    live_force_alive_ops: &[Operand],
    include_all_exc_links: bool,
    cpu: Option<&'a super::cpu::Cpu>,
) -> SSARepr {
    // `flatten.py:68 flattener.enforce_input_args()`.  Upstream stores
    // `regallocs` on `self.regallocs` and the method mutates it
    // in place; pyre's `get_register` closure (constructed below)
    // captures `&regallocs`, so the swap runs BEFORE the closure exists.
    super::regalloc::enforce_input_args_simulation(graph, regallocs);
    // `flatten.py:67 flattener = GraphFlattener(graph, regallocs,
    // _include_all_exc_links, cpu)`.
    let lowering_ctx = cpu.and_then(|c| c.lowering_ctx.read().ok().and_then(|guard| *guard));
    let mut ssarepr = SSARepr::new(graph.name.clone());
    // `flatten.py:382-391 getcolor(v)`, prepended with the Phase 4
    // bridge lookup documented on `flatten_graph_with_walker_slots`.
    let get_register = |variable: Variable| -> Register {
        let kind = variable.kind.unwrap_or(Kind::Ref);
        let walker_slot = walker_slot_for_variable
            .get(variable.id.0 as usize)
            .copied()
            .flatten();
        if let Some(slot) = walker_slot {
            return Register::new(kind, slot);
        }
        let color = regallocs[kind.index()]
            .coloring
            .get(&variable.id)
            .copied()
            .unwrap_or(u16::MAX);
        Register::new(kind, color)
    };
    // `flatten_constant_operand` panics on `Opaque(Ref)` constants
    // — upstream's `flatten` lowering relies on the rtyper preserving
    // `Constant(ll_ovf, concretetype=...)` through to the assembler,
    // and pyre's canonical entry takes the same fail-loud stance:
    // graphs that carry opaque Ref constants (production pycode /
    // jitdriver pointers, the OverflowError instance from
    // `make_exception_link`) MUST be flattened via
    // `flatten_graph_with_lowering` with a production `lower_constant`
    // closure that resolves the opaque to the runtime PyObject
    // pointer.  Calling the canonical entry on such a graph surfaces
    // the divergence instead of silently materialising `ConstRef(0)`.
    let lower_constant = flatten_constant_operand;
    let mut flattener = if let Some(ctx) = lowering_ctx {
        GraphFlattener::new_with_full_lowering(&mut ssarepr, get_register, lower_constant, ctx)
    } else {
        GraphFlattener::new_with_constant_lowering(&mut ssarepr, get_register, lower_constant)
    };
    if let Some(cpu) = cpu {
        flattener = flattener.with_cpu(cpu);
    }
    // `flatten.py:75 GraphFlattener.__init__ ._include_all_exc_links =
    // _include_all_exc_links`.
    flattener.include_all_exc_links = include_all_exc_links;
    // Task #50 T6 epic slice 1 — pre-seed block_names with walker's
    // `pc{N}` naming so `Label(block)` / `tlabel_for_block(target)`
    // resolve to walker-compatible names.  Overrides land first in the
    // Vec; `label_name_for_block` does a linear scan and returns the
    // first matching entry, so subsequent default `block{N}` insertions
    // for blocks not in the override list keep their fallback names.
    for (block, name) in block_name_overrides {
        flattener.block_names.push((block.clone(), name.clone()));
    }
    // Task #50 T6 epic slice 2 — pre-seed link_names with names that
    // ALIAS the link to its target block's `pc{N}`.  Walker emits no
    // separate link landing — its `goto_if_not Reg TLabel("pcN")`
    // jumps directly to the target block's per-PC anchor.  Canonical
    // emits `goto_if_not Reg TLabel(link_name)` + `Label(link_name)` +
    // (optional renamings) + recursive `make_bytecode_block(target)`
    // which emits `Label(target_block_name)`.  With the alias both
    // canonical Label entries carry the same name string; the runtime's
    // `pc_anchor_positions` first-wins resolves `pcN` to the link
    // landing byte position, which then either no-ops (empty
    // renamings) or runs the renamings before falling through to the
    // target block's body.  Insn-level byte_equivalent improves
    // because `TLabel` operand strings now match between walker and
    // canonical.  Functionality-preserving — extra Label("pcN") emits
    // are filtered by `count_real_ops` (Label is scaffold).
    for (link, name) in link_name_overrides {
        flattener.link_names.push((link.clone(), name.clone()));
    }
    // Task #50 T6 epic slice 4b — pre-seed block_py_pcs so
    // `make_bytecode_block` emits `Insn::PcAnchor { py_pc }` after
    // each block's Label, matching walker's per-PC anchor.  Makes
    // canonical's SSARepr stream runtime-compatible with pyre's
    // `pc_anchor_positions` lookup — necessary precondition for the
    // Phase 4 production splice.  Upstream RPython has no per-PC
    // anchor; this is a pyre adaptation layer.
    for (block, py_pc) in block_py_pc_overrides {
        flattener.block_py_pcs.push((block.clone(), *py_pc));
    }
    // Phase 4 slice 2 — pre-seed live force-alive ops so canonical's
    // per-PC `-live-` placeholders carry the portal red args matching
    // walker's `emit_live_placeholder!` shape.  Without these, the
    // bridge's `setup_bridge_sym` crashes at portal_ec_reg lookup
    // (compute_liveness misses the unforced register).
    flattener.live_force_alive_ops = live_force_alive_ops.to_vec();
    // `flatten.py:69 flattener.generate_ssa_form()`.
    flattener.generate_ssa_form(graph);
    ssarepr
}

/// Lower one `SpaceOperation` to a single `Insn::Op`.  Used by
/// [`flatten_op_to_insn_with_lowering`] as the passthrough fallback
/// when no HLOp lowering matches.  Constant operands lower via the
/// caller-supplied `lower_constant` closure (production threads its
/// real `lower_constant`; tests pass `flatten_constant_operand_for_test`).
fn flatten_op_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    let args: Vec<Operand> = op
        .args
        .iter()
        .map(|arg| flatten_arg_with_lowering(arg, get_register, lower_constant))
        .collect();
    let insn = match &op.result {
        None => Insn::op(op.opname.clone(), args),
        Some(FlowValue::Variable(variable)) => {
            let reg = get_register(*variable);
            Insn::op_with_result(op.opname.clone(), args, reg)
        }
        Some(FlowValue::Constant(constant)) => panic!(
            "GraphFlattener: op {} has Constant result {:?}; \
             flow graph results must be Variables",
            op.opname, constant
        ),
    };
    Some(insn)
}

/// Generalized `SpaceOperationArg → Operand` lowering with a
/// caller-supplied `lower_constant` closure that decides how
/// `FlowValue::Constant` values map to `Operand`s.
///
/// Production callers (`GraphFlattener::flatten_arg` via
/// `self.lower_constant`) thread the per-call closure that resolves
/// `Opaque(Ref)` to the real `w_code` PyObject pointer.  Tests pass
/// [`flatten_constant_operand_for_test`] (a `ConstRef(0)` placeholder).
/// Variable / list / descr / indirect-call-targets handling is the
/// same in both cases; only constant operand lowering is pluggable.
fn flatten_arg_with_lowering<F, LC>(
    arg: &super::flow::SpaceOperationArg,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Operand
where
    F: FnMut(Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    match arg {
        super::flow::SpaceOperationArg::Value(FlowValue::Variable(v)) => {
            Operand::Register(get_register(*v))
        }
        super::flow::SpaceOperationArg::Value(FlowValue::Constant(c)) => lower_constant(c),
        super::flow::SpaceOperationArg::ListOfKind(list) => Operand::ListOfKind(ListOfKind::new(
            list.kind,
            list.content
                .iter()
                .map(|value| match value {
                    FlowValue::Variable(v) => Operand::Register(get_register(*v)),
                    FlowValue::Constant(c) => lower_constant(c),
                })
                .collect(),
        )),
        // `flatten.py:365-367` passes `IndirectCallTargets` through the
        // generic flatten path unchanged.  Mirror that here so the
        // probe sees the same operand shape inline emits would
        // produce; `Operand::IndirectCallTargets` clones the inner
        // `Vec<Arc<JitCode>>` (cheap Arc bumps).
        super::flow::SpaceOperationArg::IndirectCallTargets(targets) => {
            Operand::IndirectCallTargets((*targets.0).clone())
        }
        // `flatten.py:365-367` also passes `AbstractDescr` through
        // unchanged.  The probe shares
        // `flatten_descr_by_ptr` with the production
        // `GraphFlattener::flatten_arg`; both match the `DescrByPtr`
        // singleton by `Arc::ptr_eq` and lower to the same
        // `DescrOperand` variant so the diagnostic shape compare at
        // `codewriter.rs:6013` sees identical operand sequences when
        // graph and SSA agree.
        super::flow::SpaceOperationArg::Descr(descr_by_ptr) => flatten_descr_by_ptr(descr_by_ptr),
    }
}

/// Lower a `flow::DescrByPtr` to the matching SSARepr-side
/// `DescrOperand`.  Two recognition paths:
///
/// 1. `Arc::ptr_eq` against the vable singleton accessors in
///    `majit_ir::descr` — array_field / array / static_field, emitted
///    by `record_graph_op` for vable get/setfield + get/setarrayitem
///    ops (`jtransform.py:846-927`, `:1880-1906`).
/// 2. `as_any` downcast to pyre's local [`CallDescrStub`], for graph-
///    side `residual_call_*` recorders (Task #42) that thread the
///    interned stub via [`intern_call_descr_stub`].  The downcast
///    clones the stub value into the SSA-side `DescrOperand` so the
///    consumer (`assembler::dispatch_residual_call`) sees the same
///    shape it would from a direct inline emit.
///
/// Other concrete descr flavors (`Bh`, `SwitchDict`) are constructed
/// directly at the SSARepr-emit site rather than going through
/// `SpaceOperationArg::Descr`, so this fn rejects them.  Adding a
/// new graph-side descr producer must extend the recognition arms.
fn flatten_descr_by_ptr(descr: &super::flow::DescrByPtr) -> Operand {
    let descr_ref = &descr.0;
    if std::sync::Arc::ptr_eq(descr_ref, &majit_ir::descr::vable_array_field_descr(0)) {
        return Operand::descr_vable_array_field(0);
    }
    if std::sync::Arc::ptr_eq(descr_ref, &majit_ir::descr::vable_array_descr(0)) {
        return Operand::descr_vable_array(0);
    }
    // VableStaticField: pyre's PyFrame _virtualizable_ has 6 static
    // fields (interp_jit.py:25-31, idx 0..=5).  Probe each idx in
    // turn and Arc::ptr_eq against the per-idx singleton.  Mirrors
    // the `array_field_descrs[i]` enumeration above.
    for idx in 0u16..6 {
        if std::sync::Arc::ptr_eq(descr_ref, &majit_ir::descr::vable_static_field_descr(idx)) {
            return Operand::descr_vable_static_field(idx);
        }
    }
    // CallDescrStub recognition: any Arc<dyn Descr> whose `as_any`
    // downcasts to pyre's `CallDescrStub`.  The graph-side
    // `residual_call_*` recorder threads the interned stub from
    // `intern_call_descr_stub`; the lowered SSA `Operand` must
    // structurally match what `emit_residual_call_shape` emits inline
    // (clone the stub value into a fresh `Rc<DescrOperand>`).
    if let Some(any) = descr_ref.as_any() {
        if let Some(stub) = any.downcast_ref::<CallDescrStub>() {
            return Operand::descr(DescrOperand::CallDescrStub(stub.clone()));
        }
    }
    panic!(
        "flatten_descr_by_ptr: unmapped DescrByPtr {} — only vable \
         array_field / array / static_field singletons + CallDescrStub \
         are recognised today",
        descr_ref.repr()
    )
}

// ---------------------------------------------------------------------------
// Task #48 — flatten-time pre-rtype HLOp lowering.
// ---------------------------------------------------------------------------

/// Per-CodeWriter context that the pre-rtype HLOp lowering arms read
/// to reconstruct the inline-equivalent `residual_call_*` Insn.
/// Mirrors how RPython's `flatten_graph(graph, regallocs)`
/// (`flatten.py:60`) threads per-graph data through the
/// `GraphFlattener`; pyre's pass is per-CodeWriter, but the threading
/// shape is identical.
///
/// Slice 1 of the epic introduced `binary_op_fn_idx` (BINARY_OP
/// family); slice 4 adds `compare_op_fn_idx` (COMPARE_OP family).
/// Subsequent slices add `truth_fn_idx`, `setitem_fn_idx`, etc.,
/// one per HLOp family that the lowering pass brings online.
#[derive(Debug, Clone, Copy)]
pub struct LoweringContext {
    /// `binary_op_fn` descrs-pool index — see codewriter.rs:3081
    /// (`descrs.intern_int_method_index("binary_op_fn", ...)`) for
    /// the production source.
    pub binary_op_fn_idx: u16,
    /// `compare_fn` descrs-pool index — see codewriter.rs:3076
    /// for the production source.  COMPARE_OP family
    /// (`lt`/`le`/`eq`/`ne`/`gt`/`ge`) shares the same `(ref, ref,
    /// int) → ref` signature as BINARY_OP, so the lowered Insn
    /// shape is identical apart from the leading fn-idx ConstInt.
    pub compare_op_fn_idx: u16,
    /// `truth_fn` descrs-pool index — see codewriter.rs:3091 for
    /// the production source.  BOOL family (single HLOp opname
    /// `bool`) lowers to `residual_call_r_i` (one Ref input, Int
    /// result) — different shape from the `_ir_r` family because
    /// `truth_fn` has signature `(ref) → int` and no scalar Int
    /// `op_val` argument.  Flavor = `MayForce` (truth_fn delegates
    /// to `opcode_ops::truth_value(obj)` which invokes Python
    /// `__bool__` / `__len__` per PyPy `descroperation.py:265` and
    /// may run user code that observes virtualizables — matches the
    /// `MayForce` bind site at codewriter.rs:2208 and the SSA
    /// helper at flatten.rs:`build_residual_call_r_i_insn_from_
    /// operands`).
    pub truth_fn_idx: u16,
    /// `store_subscr_fn` descrs-pool index — see
    /// codewriter.rs:3101 for the production source.  SETITEM
    /// family (single HLOp opname `setitem`) lowers to
    /// `residual_call_r_v` (three Ref inputs, void result) —
    /// different from the `_ir_r` and `_r_i` shapes because
    /// `store_subscr_fn` has signature `(ref, ref, ref) → void`,
    /// so the residual_call Insn has no result Register and no
    /// `ListI` (no scalar Int args).
    pub store_subscr_fn_idx: u16,
    /// `build_list_fn` descrs-pool index — see codewriter.rs:2401
    /// (`bind(assembler, cpu.build_list_fn as *const (),
    /// CallFlavor::Plain)`) for the production source.  BUILD_LIST
    /// (single HLOp opname `newlist`) lowers to `residual_call_ir_r`
    /// via [`build_build_list_fn_residual_call_ir_r_insn`] which
    /// pads unused item slots with `ConstInt(0)` matching the
    /// inline emit at codewriter.rs:6390-6398.  Walker only emits
    /// `newlist` HLOp on graph for argc ≤ 3; argc > 3 emits
    /// `abort_permanent` instead, so the canonical lowering arm
    /// returns `None` (passthrough) on argc > 3.
    pub build_list_fn_idx: u16,
    /// `call_fn_N` descrs-pool indices for nargs ∈ 0..=8 — see
    /// codewriter.rs:3206-3245 for the production source.  CALL
    /// (single HLOp opname `simple_call`) lowers to
    /// `residual_call_r_r(call_fn_N_idx, [callable, arg0, ...],
    /// Descr) → reg` via [`build_call_fn_residual_call_r_r_insn`].
    /// Indexed by nargs (`call_fn_idx_by_nargs[nargs]`) per
    /// [[feedback-no-hashmap-ever]] — `[u16; 9]` keeps the
    /// statically-known 0..=8 arity range position-indexed.
    /// `simple_call` HLOps with nargs > 8 are walker non-orthodox
    /// (the walker emits `abort_permanent` instead and skips the
    /// HLOp record), so the lowering arm returns `None`
    /// (passthrough) on nargs > 8.
    pub call_fn_idx_by_nargs: [u16; 9],
}

/// Map a BINARY_OP HLOp opname (`add`/.../`xor`/`getitem` plus the
/// `inplace_*` siblings) to the `op_val` integer that the inline emit
/// at codewriter.rs:5348 passes as the third `residual_call_ir_r`
/// argument.  The mapping mirrors
/// `pyre_interpreter::runtime_ops::binary_op_tag` — both decode
/// `BinaryOperator` to the same compact tag the blackhole interpreter
/// reads back via `binary_op_from_tag`.  Returns `None` for opnames
/// outside the BINARY_OP family so the caller can fall through to
/// other lowering arms.
fn binary_op_tag_for_opname(opname: &str) -> Option<i64> {
    Some(match opname {
        "add" | "inplace_add" => 0,
        "sub" | "inplace_sub" => 1,
        "mul" | "inplace_mul" => 2,
        "floordiv" | "inplace_floordiv" => 3,
        "mod" | "inplace_mod" => 4,
        "truediv" | "inplace_truediv" => 5,
        "getitem" => 6,
        "pow" | "inplace_pow" => 7,
        "lshift" | "inplace_lshift" => 8,
        "rshift" | "inplace_rshift" => 9,
        "and_" | "inplace_and" => 10,
        "or_" | "inplace_or" => 11,
        "xor" | "inplace_xor" => 12,
        _ => return None,
    })
}

/// Lower a BINARY_OP-family pre-rtype HLOp `add(lhs, rhs) → result`
/// to the equivalent post-rtype
/// `residual_call_ir_r(ConstInt(fn_idx), ListR([lhs, rhs]),
/// ConstInt(op_val), Descr) → reg` Insn.  The shape mirrors what
/// `emit_residual_call_shape` produces inline at the BinaryOp
/// callsite (codewriter.rs:5335-5352) and what
/// `record_residual_call_graph_op` records on the graph side
/// (codewriter.rs:5366-5377).  Both shapes coexist on portal graphs
/// today and are byte-equivalent when flattened — this helper
/// produces the same Insn directly from the HLOp, without going
/// through the dual-write.
///
/// Returns `None` when the SpaceOperation is not a BINARY_OP family
/// HLOp (caller falls through to the default opname-passthrough Insn
/// arm in `flatten_op_to_insn`).
///
/// BINARY_OP family lowering.  Production walker emits the lowered
/// `residual_call_ir_r` Insn directly via
/// [`build_binary_op_residual_call_ir_r_insn`] at the callsite; the
/// graph carries only the pre-rtype `add(...)` HLOp.  This helper
/// produces the same Insn from the HLOp for the post-walker
/// `flatten_op_to_insn_with_lowering` dispatcher.
pub fn lower_binary_op_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    let op_val = binary_op_tag_for_opname(&op.opname)?;
    if op.args.len() != 2 {
        return None;
    }
    let result_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let lhs_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let rhs_operand = flatten_arg_with_lowering(&op.args[1], get_register, lower_constant);
    Some(build_residual_call_ir_r_insn_from_operands(
        ctx.binary_op_fn_idx,
        op_val,
        lhs_operand,
        rhs_operand,
        CallFlavor::MayForce,
        result_reg,
    ))
}

/// Construct the BINARY_OP-family `residual_call_ir_r` Insn from
/// raw register indices.  Production codewriter callsite (Slice
/// micro-slice 3) bypasses the SpaceOperation→Insn round-trip and
/// emits this Insn directly into the SSARepr, replacing the prior
/// `emit_residual_call(binary_op_fn_idx, ...)` + matching graph
/// dual-write at codewriter.rs:5335-5378.
///
/// Mirrors [`lower_binary_op_hlop_to_insn`]'s output shape: the
/// post-walker dispatcher and the walker-time direct push both produce
/// the same `residual_call_ir_r` Insn bytes.
///
/// `op_val` is the `binary_op_tag` integer derived from the
/// `BinaryOperator` (e.g., `add → 0`, `sub → 1`); production
/// callsite obtains it directly from
/// `pyre_interpreter::runtime_ops::binary_op_tag(op_kind)`.
pub fn build_binary_op_residual_call_ir_r_insn(
    binary_op_fn_idx: u16,
    op_val: i64,
    lhs_reg: u16,
    rhs_reg: u16,
    dst_reg: u16,
) -> Insn {
    build_residual_call_ir_r_insn_from_operands(
        binary_op_fn_idx,
        op_val,
        Operand::Register(Register::new(Kind::Ref, lhs_reg)),
        Operand::Register(Register::new(Kind::Ref, rhs_reg)),
        CallFlavor::MayForce,
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Shared shape constructor used by both the probe-side
/// (`lower_*_hlop_to_insn`) and the production-side
/// (`build_*_residual_call_ir_r_insn`) lowering helpers for the
/// `(Ref, Ref, Int) → Ref` HLOp / helper families that lower to a
/// uniform `residual_call_ir_r` Insn shape.  Today: BINARY_OP
/// (`binary_op_fn`, MayForce), COMPARE_OP (`compare_fn`, MayForce),
/// and two-Ref residual helper families.  BINARY_OP and COMPARE_OP share
/// `arg_kinds = [Ref,
/// Ref, Int]`, ResKind Ref → kinds `"ir"` + reskind `'r'` → opname
/// `"residual_call_ir_r"`.  They differ in the leading `fn_idx`
/// literal, the per-family `op_val` (or callee-arg integer), and
/// the `CallFlavor` carried on the EffectInfo descr.
///
/// Inline arg order produced by `emit_residual_call_shape`
/// (codewriter.rs:2745-2802) buckets each `CallArgInput` by `Kind`
/// into per-kind lists then concatenates `[ConstInt(fn), ListI?,
/// ListR?, ListF?, Descr]`.  For these families the call_args are
/// `[Reg(Ref, lhs), Reg(Ref, rhs), ConstInt(op_val)]`, so:
///   * `args_i = [ConstInt(op_val)]`
///   * `args_r = [Reg(lhs), Reg(rhs)]`
///   * `args_f = []`
/// → final SSARepr Insn `[ConstInt(fn_idx),
///                         ListI([ConstInt(op_val)]),
///                         ListR([Reg(lhs), Reg(rhs)]), Descr]`.
fn build_residual_call_ir_r_insn_from_operands(
    fn_idx: u16,
    op_val: i64,
    lhs_operand: Operand,
    rhs_operand: Operand,
    flavor: CallFlavor,
    dst_reg: Register,
) -> Insn {
    let effect_info = effect_info_for_call_flavor(flavor);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
    }));
    Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![Operand::ConstInt(op_val)])),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![lhs_operand, rhs_operand])),
            descr_operand,
        ],
        dst_reg,
    )
}

/// Map a COMPARE_OP HLOp opname (`lt`/`le`/`eq`/`ne`/`gt`/`ge`)
/// to the `op_val` integer that the inline emit at
/// codewriter.rs:5406 passes as the third `residual_call_ir_r`
/// argument.  The mapping mirrors
/// `pyre_interpreter::runtime_ops::compare_op_tag` (codewriter
/// uses the same source of truth).  Returns `None` for opnames
/// outside the COMPARE_OP family so the caller can fall through.
fn compare_op_tag_for_opname(opname: &str) -> Option<i64> {
    Some(match opname {
        "lt" => 0,
        "le" => 1,
        "gt" => 2,
        "ge" => 3,
        "eq" => 4,
        "ne" => 5,
        _ => return None,
    })
}

/// Lower a COMPARE_OP-family pre-rtype HLOp `lt(lhs, rhs) → result`
/// (and the 5 sibling opnames) to the equivalent post-rtype
/// `residual_call_ir_r(ConstInt(fn_idx), ListI([ConstInt(op_val)]),
/// ListR([lhs, rhs]), Descr) → reg` Insn.  `compare_fn` shares the
/// same `(ref, ref, int) → ref` C signature as `binary_op_fn`, so
/// the lowered Insn shape is structurally identical apart from the
/// leading `fn_idx` literal.  Returns `None` for non-family opnames
/// so the caller can fall through.
///
/// COMPARE_OP family lowering — same pattern as
/// [`lower_binary_op_hlop_to_insn`].  Production walker emits via
/// [`build_compare_op_residual_call_ir_r_insn`]; this helper
/// reconstructs the same Insn from the HLOp for the post-walker
/// dispatcher.
pub fn lower_compare_op_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    let op_val = compare_op_tag_for_opname(&op.opname)?;
    if op.args.len() != 2 {
        return None;
    }
    let result_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    let lhs_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let rhs_operand = flatten_arg_with_lowering(&op.args[1], get_register, lower_constant);
    Some(build_residual_call_ir_r_insn_from_operands(
        ctx.compare_op_fn_idx,
        op_val,
        lhs_operand,
        rhs_operand,
        CallFlavor::MayForce,
        result_reg,
    ))
}

/// Construct the COMPARE_OP-family `residual_call_ir_r` Insn from
/// raw register indices.  Production codewriter callsite (Slice 4
/// retirement) bypasses the SpaceOperation→Insn round-trip and
/// emits this Insn directly into the SSARepr, replacing the prior
/// `emit_residual_call(compare_fn_idx, ...)` + matching graph
/// dual-write at codewriter.rs:5393-5428.
///
/// `op_val` is the `compare_op_tag` integer derived from the
/// `ComparisonOperator` (`lt → 0`, `le → 1`, `gt → 2`, `ge → 3`,
/// `eq → 4`, `ne → 5`); production callsite obtains it directly
/// from `pyre_interpreter::runtime_ops::compare_op_tag(op_kind)`.
pub fn build_compare_op_residual_call_ir_r_insn(
    compare_op_fn_idx: u16,
    op_val: i64,
    lhs_reg: u16,
    rhs_reg: u16,
    dst_reg: u16,
) -> Insn {
    build_residual_call_ir_r_insn_from_operands(
        compare_op_fn_idx,
        op_val,
        Operand::Register(Register::new(Kind::Ref, lhs_reg)),
        Operand::Register(Register::new(Kind::Ref, rhs_reg)),
        CallFlavor::MayForce,
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Construct the LOAD_GLOBAL-family `residual_call_ir_r` Insn from
/// raw register indices.  Production codewriter callsite (Slice
/// #48.8 factor refactor) replaces the prior `emit_residual_call(
/// load_global_fn_idx, ...)` SSARepr emit at codewriter.rs:5598-5615
/// with a single direct push of this helper's output.  The matching
/// graph dual-write at codewriter.rs:5622-5635 stays in place — this
/// slice is incremental factor refactor, not retirement.
///
/// `load_global_fn` has signature `(ns: Ref, code: Ref, frame: Ref, namei: Int)
/// → Ref` with `CallFlavor::Plain` (per codewriter.rs:2176-2185 —
/// `bh_load_global_fn` can `NameError` but cannot force virtuals; matches
/// `EF_CAN_RAISE`).  The explicit frame Ref is the Rust residual-helper
/// adaptation for PyPy's `_load_global(self, ...)` receiver.
///
/// The matching graph dual-write at codewriter.rs (LoadGlobal arm)
/// records `CallFlavor::Plain` so the SSA helper, the inline
/// SSARepr emit, and the graph residual_call agree end-to-end.
pub fn build_load_global_fn_residual_call_ir_r_insn(
    load_global_fn_idx: u16,
    namei: i64,
    ns_reg: u16,
    code_reg: u16,
    frame_reg: u16,
    dst_reg: u16,
) -> Insn {
    let effect_info = effect_info_for_call_flavor(CallFlavor::Plain);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
    }));
    Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(load_global_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![Operand::ConstInt(namei)])),
            Operand::ListOfKind(ListOfKind::new(
                Kind::Ref,
                vec![
                    Operand::Register(Register::new(Kind::Ref, ns_reg)),
                    Operand::Register(Register::new(Kind::Ref, code_reg)),
                    Operand::Register(Register::new(Kind::Ref, frame_reg)),
                ],
            )),
            descr_operand,
        ],
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Construct the CALL-family `residual_call_r_r` Insn from raw
/// register indices.  Production codewriter callsite replaces the prior `emit_residual_call(
/// call_fn_N_idx, ...)` SSARepr emit at codewriter.rs:5747-5754
/// (the `nargs <= 8` branch of the Instruction::Call arm) with a
/// single direct push of this helper's output.  The matching graph
/// dual-write at codewriter.rs:5760-5777 stays in place — this slice
/// is incremental factor refactor, not retirement.
///
/// `call_fn_N` has signature `(callable: Ref, arg0: Ref, ..., arg_
/// {N-1}: Ref) → Ref` with `CallFlavor::MayForce` for every
/// arity-specific variant `call_fn_0` / `call_fn` (= nargs=1) /
/// `call_fn_2` / ... / `call_fn_8` (per codewriter.rs:2175 and
/// 2238-2245).  All-Ref call_args produce a different SSARepr
/// shape from the `_ir_r` family: `args_i = []`, `args_r =
/// [Reg(callable), Reg(arg0), ..., Reg(arg_{N-1})]`, `args_f = []`
/// → opname `residual_call_r_r` (kinds `"r"` + reskind `'r'`)
/// with NO leading `ListI` (`emit_residual_call_shape` at
/// codewriter.rs:2745-2802 omits the per-kind list when `args_K`
/// is empty).
///
/// `arg_regs.len()` is the call's `nargs`; the resulting
/// `arg_kinds = vec![Kind::Ref; nargs + 1]` (callable + nargs
/// args).  Caller must ensure `nargs <= 8` — the codewriter falls
/// through to `emit_abort_permanent!` for `nargs > 8` and never
/// invokes this helper (no matching `call_fn_N` exists).
pub fn build_call_fn_residual_call_r_r_insn(
    call_fn_idx: u16,
    callable_reg: u16,
    arg_regs: &[u16],
    dst_reg: u16,
) -> Insn {
    let mut ref_operands: Vec<Operand> = Vec::with_capacity(1 + arg_regs.len());
    ref_operands.push(Operand::Register(Register::new(Kind::Ref, callable_reg)));
    for &reg in arg_regs {
        ref_operands.push(Operand::Register(Register::new(Kind::Ref, reg)));
    }
    build_residual_call_r_r_insn_from_operands(
        call_fn_idx,
        ref_operands,
        CallFlavor::MayForce,
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Construct the get_current_exception-family `residual_call_r_r`
/// Insn.  Production codewriter callsite replaces the prior `emit_residual_call(
/// get_current_exception_fn_idx, ...)` SSARepr emit at
/// codewriter.rs:6116-6123 (PushExcInfo).  The matching graph
/// dual-write at codewriter.rs:6141-6152 stays in place.
///
/// `get_current_exception_fn` has signature `() → Ref` with
/// `CallFlavor::PlainCannotRaiseNoHeap` (per codewriter.rs:2246-2252 —
/// TLS read of `CURRENT_EXCEPTION`; `EF_CANNOT_RAISE`, no heap access,
/// no GC).  Zero-arg (`ref_operands` empty) produces a
/// `residual_call_r_r(ConstInt(fn_idx), ListR([]), Descr) → Reg(Ref,
/// dst)` Insn — same opname as CALL family but with empty `ListR` and
/// PlainCannotRaiseNoHeap flavor.
pub fn build_get_current_exception_fn_residual_call_r_r_insn(
    get_current_exception_fn_idx: u16,
    dst_reg: u16,
) -> Insn {
    build_residual_call_r_r_insn_from_operands(
        get_current_exception_fn_idx,
        Vec::new(),
        CallFlavor::PlainCannotRaiseNoHeap,
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Shared shape constructor for `(Ref, Ref, ..., Ref) → Ref` HLOp
/// / helper families that lower to a uniform `residual_call_r_r`
/// Insn shape.  Today: CALL (`call_fn_N` for nargs ∈ 0..=8,
/// MayForce, ≥1 Refs), normalize_raise_varargs (MayForce, fixed 2
/// Refs), get_current_exception (PlainCannotRaiseNoHeap, 0 Refs).
/// `arg_kinds = vec![Kind::Ref; ref_operands.len()]`, ResKind = Ref
/// → kinds `"r"` + reskind `'r'` → opname `"residual_call_r_r"`.
/// No leading `ListI` (empty `args_i`).  Variable-arity + flavor.
pub fn build_residual_call_r_r_insn_from_operands(
    fn_idx: u16,
    ref_operands: Vec<Operand>,
    flavor: CallFlavor,
    dst_reg: Register,
) -> Insn {
    let arg_kinds = vec![Kind::Ref; ref_operands.len()];
    let effect_info = effect_info_for_call_flavor(flavor);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds,
        result_kind: Some(Kind::Ref),
    }));
    Insn::op_with_result(
        "residual_call_r_r",
        vec![
            Operand::ConstInt(fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, ref_operands)),
            descr_operand,
        ],
        dst_reg,
    )
}

/// Construct the RaiseVarargs-family `residual_call_r_r` Insn from
/// raw register indices.  Production codewriter callsite (Slice
/// #48.14 factor refactor) replaces the prior `emit_residual_call(
/// normalize_raise_varargs_fn_idx, ...)` SSARepr emit at
/// codewriter.rs:6068-6082 with a single direct push of this
/// helper's output.  No graph dual-write exists for
/// `normalize_raise_varargs_fn` (the graph carries an `emit_raise!`
/// edge instead).
///
/// `normalize_raise_varargs_fn` has signature `(exc: Ref, cause:
/// Ref) → Ref` with `CallFlavor::MayForce` (per codewriter.rs:2227-
/// 2236 — `bh_normalize_raise_varargs_fn` instantiates user
/// `__init__` and may observe virtualizables; matches
/// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`).
///
/// The `cause` operand is polymorphic at the callsite: when the
/// RAISE_VARARGS opcode arrives with `argc=2` the cause is a
/// popped stack register (`Operand::Register(Ref, cause_reg)`);
/// when `argc=1` the cause is `PY_NULL` constant
/// (`Operand::ConstRef(pyre_object::PY_NULL)`).  Both encode under
/// `Kind::Ref` so the bucket lands in `args_r` regardless and the
/// produced shape is `residual_call_r_r` — same opname as the CALL
/// family (`build_call_fn_residual_call_r_r_insn`), distinguished
/// only by being fixed-arity 2 vs CALL's variable arity.
///
/// The shared shape constructor
/// `build_residual_call_r_r_insn_from_operands` accepts arbitrary
/// `ref_operands.len()` plus a `flavor` parameter (extended from MayForce-only to support the
/// PlainCannotRaiseNoHeap exception-family helpers); this caller
/// passes `MayForce` matching the production source.
pub fn build_normalize_raise_varargs_fn_residual_call_r_r_insn(
    normalize_raise_varargs_fn_idx: u16,
    exc_reg: u16,
    cause: Operand,
    dst_reg: u16,
) -> Insn {
    build_residual_call_r_r_insn_from_operands(
        normalize_raise_varargs_fn_idx,
        vec![Operand::Register(Register::new(Kind::Ref, exc_reg)), cause],
        CallFlavor::MayForce,
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Construct the box_int-family `residual_call_ir_r` Insn from raw
/// register indices.  Production codewriter callsites replace three prior `emit_residual_call(
/// box_int_fn_idx, ...)` SSARepr emits with single direct pushes of
/// this helper's output:
///   * LoadSmallInt at codewriter.rs:4867-4874 (val = literal small
///     int from the consts table).
///   * UnaryNegative `box_int(0)` at codewriter.rs:5832-5839 (val =
///     0, materialises the zero operand for the trailing
///     `binary_op_fn(zero, operand, sub_tag)` emit).
///   * Exception-frame lasti boxing at codewriter.rs:6633-6640 (val
///     = `lasti_py_pc`, captures the frame's last-instruction offset
///     into the exception slot).
/// All 3 sites' graph dual-writes stay in place — incremental
/// factor refactor only.
///
/// `box_int_fn` has signature `(val: Int) → Ref` with
/// `CallFlavor::Plain` (per codewriter.rs:2200 — `bh_box_int_fn`
/// allocates a fresh `PyLong` wrapper without user dispatch and
/// cannot force virtuals; matches `EF_CAN_RAISE` for allocation
/// MemoryError).  RPython `jtransform.py:424-426 rewrite_call`
/// picks `kinds = 'ir'` whenever `lst_i` is non-empty (or `force_ir`
/// is set), so a single Int / no-Ref call lowers to
/// `residual_call_ir_r` with an EMPTY `ListR` between `ListI` and
/// the `Descr` slot — NOT `residual_call_i_r`.
pub fn build_box_int_fn_residual_call_ir_r_insn(
    box_int_fn_idx: u16,
    val: i64,
    dst_reg: u16,
) -> Insn {
    build_residual_call_ir_r_insn_from_int_only_operands(
        box_int_fn_idx,
        Operand::ConstInt(val),
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Shared shape constructor for `(Int) → Ref` HLOp / helper
/// families that lower to a uniform `residual_call_ir_r` Insn shape
/// with `CallFlavor::Plain` and an empty `ListR`.  Today: box_int
/// (`box_int_fn`).  `arg_kinds = [Int]`, ResKind = Ref → kinds
/// `"ir"` + reskind `'r'` → opname `"residual_call_ir_r"`.  Empty
/// `ListR` is required by RPython `jtransform.py:428-431`: whenever
/// `'r'` appears in `kinds` the `lst_r` sublist is appended even
/// when empty, so the trailing `Descr` slot stays in its canonical
/// position.  No `ListF` (empty `args_f`).
///
/// Inline arg order from `emit_residual_call_shape` for call-args
/// `[ConstInt(val)]`: `args_i = [ConstInt(val)]`, `args_r = []`,
/// `args_f = []` → final SSARepr Insn `[ConstInt(fn_idx),
///                                       ListI([ConstInt(val)]),
///                                       ListR([]),
///                                       Descr]`.
fn build_residual_call_ir_r_insn_from_int_only_operands(
    fn_idx: u16,
    int_operand: Operand,
    dst_reg: Register,
) -> Insn {
    let effect_info = effect_info_for_call_flavor(CallFlavor::Plain);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Int],
        result_kind: Some(Kind::Ref),
    }));
    Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![int_operand])),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![])),
            descr_operand,
        ],
        dst_reg,
    )
}

/// Construct the BuildList-family `residual_call_ir_r` Insn from
/// raw register indices.  Production codewriter callsite (Slice
/// #48.13 factor refactor) replaces the prior `emit_residual_call(
/// build_list_fn_idx, ...)` SSARepr emit at codewriter.rs:6002-6009
/// with a single direct push of this helper's output.  No graph
/// dual-write exists for `build_list_fn` (the graph carries
/// `newlist(items)` HLOp via `emit_frontend_newlist`); this is a
/// clean factor refactor with no asymmetry.
///
/// `build_list_fn` has C ABI `extern "C" fn(i64, i64, i64, i64)` —
/// always 4 i64 parameters dispatched internally by the leading
/// `argc` (`bh_build_list_fn` per `cpu.rs`).  The trailing 3 slots
/// hold real boxed-Ref pointers when the corresponding item is
/// present, or `0` (dummy bit pattern, routed through the int
/// constants pool per `make_three_lists` jtransform.py:437-445)
/// when absent.  Per codewriter.rs:5945-5954, `argc > 3` falls
/// through to `emit_abort_permanent` and never invokes this
/// helper, so `argc ∈ {0, 1, 2, 3}`.
///
/// Inline arg order from `emit_residual_call_shape` for call-args
/// `[ConstInt(argc), maybe_Reg_or_ConstInt(item0),
/// maybe_Reg_or_ConstInt(item1), maybe_Reg_or_ConstInt(item2)]`:
///   * `args_i` = leading argc + each absent item's `0` dummy.
///   * `args_r` = each present item's Reg.
///   * `arg_kinds` preserves call-order (NOT bucket-order): always
///     `[Int, item0_kind, item1_kind, item2_kind]` where each item kind is
///     `Ref` when present and `Int` when dummy.
///
/// Both `ListI` and `ListR` are always pushed because `args_i` is
/// always non-empty (leading argc) → `kinds = "ir"` → `residual_
/// call_ir_r`.  `ListR` is empty for `argc=0` but still emitted
/// (kind-selection logic at codewriter.rs:2771-2777 includes `'r'`
/// in `kinds` whenever a kind appears in `arg_kinds`, but
/// `emit_residual_call_shape` actually drives `kinds.contains('r')`
/// off `kinds` → the empty `ListR` is pushed to keep the trailing
/// `ListF?, Descr` slots in their canonical positions).  The
/// helper mirrors that exactly.
///
/// `arg_regs.len()` must equal `argc`; caller is the production
/// `Instruction::BuildList` arm which gathers the popped
/// `arg_regs: Vec<u16>` directly off the stack.  Hardcoded
/// `CallFlavor::Plain` matching the production source at
/// codewriter.rs:2226 (`build_list_fn` is allocation-only).
pub fn build_build_list_fn_residual_call_ir_r_insn(
    build_list_fn_idx: u16,
    argc: usize,
    arg_regs: &[u16],
    dst_reg: u16,
) -> Insn {
    assert_eq!(arg_regs.len(), argc, "arg_regs length must match argc");
    let item_operands: Vec<Operand> = arg_regs
        .iter()
        .map(|&reg| Operand::Register(Register::new(Kind::Ref, reg)))
        .collect();
    build_build_list_fn_residual_call_ir_r_insn_from_operands(
        build_list_fn_idx,
        argc,
        item_operands,
        dst_reg,
    )
}

/// Operand-flexible variant of `build_build_list_fn_residual_call_ir_r_insn`.
/// Each item slot can be a `Register` (resolved Variable) OR a `Const*`
/// (lowered Constant via `flatten_arg`'s Constant arm).  Used by the
/// canonical driver's `lower_newlist_hlop_to_insn` to handle graph
/// `newlist` HLOps whose items are Constants — upstream RPython's
/// rtype pass would have pre-loaded these into Variables, but pyre's
/// graph carries the un-rewritten Constants per
/// [[project-flatten-graph-canonical-driver-2026-05-17]].
pub fn build_build_list_fn_residual_call_ir_r_insn_from_operands(
    build_list_fn_idx: u16,
    argc: usize,
    item_operands: Vec<Operand>,
    dst_reg: u16,
) -> Insn {
    assert!(
        argc <= 3,
        "BuildList helper only supports argc ∈ {{0, 1, 2, 3}}"
    );
    assert_eq!(
        item_operands.len(),
        argc,
        "item_operands length must match argc"
    );
    let mut arg_kinds: Vec<Kind> = Vec::with_capacity(4);
    let mut args_i: Vec<Operand> = Vec::with_capacity(4);
    let mut args_r: Vec<Operand> = Vec::with_capacity(3);
    // Leading argc slot — always Int.
    arg_kinds.push(Kind::Int);
    args_i.push(Operand::ConstInt(argc as i64));
    // Trailing 3 slots — Ref if present, Int dummy `0` if absent.
    let mut item_iter = item_operands.into_iter();
    for i in 0..3 {
        if i < argc {
            arg_kinds.push(Kind::Ref);
            args_r.push(item_iter.next().unwrap());
        } else {
            arg_kinds.push(Kind::Int);
            args_i.push(Operand::ConstInt(0));
        }
    }
    let effect_info = effect_info_for_call_flavor(CallFlavor::Plain);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds,
        result_kind: Some(Kind::Ref),
    }));
    Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(build_list_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, args_i)),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, args_r)),
            descr_operand,
        ],
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Construct the BUILD_SLICE helper call.  Mirrors
/// `pypy/interpreter/pyopcode.py:1463-1472`: argc is 2 or 3, start/stop are
/// refs, and step is either a ref (argc=3) or an ignored int dummy (argc=2).
pub fn build_build_slice_fn_residual_call_ir_r_insn(
    build_slice_fn_idx: u16,
    argc: usize,
    start_reg: u16,
    stop_reg: u16,
    step_reg: Option<u16>,
    dst_reg: u16,
) -> Insn {
    assert!(
        matches!((argc, step_reg), (2, None) | (3, Some(_))),
        "BUILD_SLICE expects argc=2 without step or argc=3 with step"
    );
    let mut arg_kinds = vec![Kind::Int, Kind::Ref, Kind::Ref];
    let mut args_i = vec![Operand::ConstInt(argc as i64)];
    let mut args_r = vec![
        Operand::Register(Register::new(Kind::Ref, start_reg)),
        Operand::Register(Register::new(Kind::Ref, stop_reg)),
    ];
    if let Some(step_reg) = step_reg {
        arg_kinds.push(Kind::Ref);
        args_r.push(Operand::Register(Register::new(Kind::Ref, step_reg)));
    } else {
        arg_kinds.push(Kind::Int);
        args_i.push(Operand::ConstInt(0));
    }
    let effect_info = effect_info_for_call_flavor(CallFlavor::Plain);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds,
        result_kind: Some(Kind::Ref),
    }));
    Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(build_slice_fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, args_i)),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, args_r)),
            descr_operand,
        ],
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Lower a BOOL-family pre-rtype HLOp `bool(operand) → result` to
/// the equivalent post-rtype `residual_call_r_i(ConstInt(fn_idx),
/// ListR([operand]), Descr) → reg` Insn.  `truth_fn` has signature
/// `(ref) → int` (no Int `op_val` argument), so the lowered shape
/// has no leading `ListI` — the inline `emit_residual_call_shape`
/// at codewriter.rs:5453-5463 with `args_i = []`, `args_r =
/// [cond_reg]` produces `kinds = "r"` + `reskind = 'i'` →
/// `residual_call_r_i`.
///
/// `bool` is a single HLOp opname (no inplace siblings, unlike
/// BINARY_OP / COMPARE_OP).  Returns `None` for non-`bool`
/// opnames.
///
/// BOOL family lowering — same per-family pattern as the BINARY_OP
/// and COMPARE_OP retirements but a different residual_call shape.
/// Production walker emits the lowered Insn via
/// [`build_truth_fn_residual_call_r_i_insn`] at PopJumpIfFalse /
/// PopJumpIfTrue; this helper reconstructs the same Insn from the
/// `bool` HLOp for the post-walker dispatcher.
pub fn lower_bool_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "bool" {
        return None;
    }
    if op.args.len() != 1 {
        return None;
    }
    // `truth_fn` returns Int (the boolean 0/1 result), so the
    // dispatcher must emit `dst_reg` with `Kind::Int` regardless of
    // the HLOp result Variable's recorded kind.  `emit_frontend_bool`
    // currently records the result as `Kind::Ref` (matching the
    // upstream `op.bool(w_value)` HLOp result type at the flowspace
    // level), but the post-rtype `residual_call_r_i` Insn carries an
    // Int destination register — the byte-equivalence check vs the
    // inline emit (`scratch_truth = fresh_var(Kind::Int, ...)`)
    // requires forcing Int here.  Matches upstream
    // `jtransform.py`-style rtype rewrite that retypes the bool
    // result to `Bool`/`Int` before flatten_graph runs.
    let result_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => {
            let r = get_register(*var);
            Register::new(Kind::Int, r.index)
        }
        _ => return None,
    };
    let arg_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    Some(build_residual_call_r_i_insn_from_operands(
        ctx.truth_fn_idx,
        arg_operand,
        result_reg,
    ))
}

/// Construct the BOOL-family `residual_call_r_i` Insn from a raw
/// register index.  Production codewriter callsite (Slice 5
/// retirement) bypasses the SpaceOperation→Insn round-trip and
/// emits this Insn directly into the SSARepr, replacing the prior
/// `emit_residual_call(truth_fn_idx, ...)` + graph dual-write at
/// codewriter.rs:5453-5480 (PopJumpIfFalse) and :5518-5544
/// (PopJumpIfTrue).
pub fn build_truth_fn_residual_call_r_i_insn(
    truth_fn_idx: u16,
    cond_reg: u16,
    dst_reg: u16,
) -> Insn {
    build_residual_call_r_i_insn_from_operands(
        truth_fn_idx,
        Operand::Register(Register::new(Kind::Ref, cond_reg)),
        Register::new(Kind::Int, dst_reg),
    )
}

/// Shared shape constructor for `(Ref) → Int` HLOp families that
/// lower to a uniform `residual_call_r_i` Insn shape.  Today: BOOL
/// (`truth_fn`).  `arg_kinds = [Ref]`, flavor = `MayForce` —
/// `bh_truth_fn` delegates to `opcode_ops::truth_value(obj)` which
/// invokes Python `__bool__` / `__len__` and may run arbitrary user
/// code that observes (and therefore forces) virtualizables, matching
/// the `MayForce` binding at codewriter.rs:2208 and PyPy
/// `descroperation.py:265`.  ResKind = Int → kinds `"r"` + reskind
/// `'i'` → opname `"residual_call_r_i"`.
///
/// Inline arg order from `emit_residual_call_shape` with empty
/// `args_i` and `args_f`: `[ConstInt(fn), ListR([cond]), Descr]`
/// (no leading `ListI` because `args_i` is empty so the
/// `kinds.contains('i')` push branch doesn't fire).
fn build_residual_call_r_i_insn_from_operands(
    fn_idx: u16,
    arg_operand: Operand,
    dst_reg: Register,
) -> Insn {
    let effect_info = effect_info_for_call_flavor(CallFlavor::MayForce);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref],
        result_kind: Some(Kind::Int),
    }));
    Insn::op_with_result(
        "residual_call_r_i",
        vec![
            Operand::ConstInt(fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![arg_operand])),
            descr_operand,
        ],
        dst_reg,
    )
}

/// Lower a SETITEM-family pre-rtype HLOp `setitem(obj, key, value)`
/// (no result — `emit_frontend_setitem` records the SpaceOperation
/// with `result = None` per upstream rtyper-equivalent void
/// rewrite) to the equivalent post-rtype
/// `residual_call_r_v(ConstInt(fn_idx), ListR([obj, key, value]),
/// Descr)` Insn.  `store_subscr_fn` has signature
/// `(ref, ref, ref) → void` so the residual_call Insn carries no
/// trailing result Register.
///
/// Single HLOp opname `setitem`.  Returns `None` for non-`setitem`
/// opnames or non-void-result HLOps.
///
/// Task #48 micro-slice 6 (SETITEM retirement): same per-family
/// pattern as micro-slices 3-5.  Differences vs the prior shapes:
///   * void Insn (no result Register).
///   * 3-element ListR (vs 2 for BINARY_OP/COMPARE_OP, 1 for BOOL).
///   * MayForce flavor (matches the prior dual-write at
///     codewriter.rs:5274).
pub fn lower_setitem_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "setitem" {
        return None;
    }
    if op.args.len() != 3 {
        return None;
    }
    if op.result.is_some() {
        return None;
    }
    let obj_operand = flatten_arg_with_lowering(&op.args[0], get_register, lower_constant);
    let key_operand = flatten_arg_with_lowering(&op.args[1], get_register, lower_constant);
    let value_operand = flatten_arg_with_lowering(&op.args[2], get_register, lower_constant);
    Some(build_residual_call_r_v_insn_from_operands(
        ctx.store_subscr_fn_idx,
        vec![obj_operand, key_operand, value_operand],
        CallFlavor::MayForce,
    ))
}

/// Single-op lowering pass that dispatches the four retired pre-rtype
/// HLOp families (BINARY_OP / COMPARE_OP / BOOL / SETITEM) through
/// the matching `lower_*_hlop_to_insn` helper, falling through to the
/// passthrough [`flatten_op_to_insn`] for any other opname.
///
/// A future post-walker
/// `flatten_graph(graph, ssarepr, ctx)` driver calls this dispatcher
/// once per `block.operations` entry to translate graph ops back into
/// the SSARepr Insn stream the assembler consumes.  Today the
/// production path emits the inline `residual_call_*` Insns directly
/// via the `build_*_residual_call_*_insn` helpers at every walker
/// callsite while the graph carries pre-rtype HLOps for the retired
/// families and post-rtype `residual_call_*` ops for the
/// factor-refactored families.  After
/// retirement the walker would only record graph ops; this dispatcher
/// is the per-op core of the post-walker driver.
///
/// Dispatch order is incidental — the four retired-family opname sets
/// are disjoint (`binary_op_tag_for_opname` / `compare_op_tag_for_opname`
/// tables do not overlap, and `bool` / `setitem` are single-opname
/// families distinct from those tables), so any HLOp that matches one
/// arm is rejected by the other three regardless of ordering.
///
/// Returns `None` only when the underlying [`flatten_op_to_insn`]
/// passthrough returns `None` (currently never).
pub fn flatten_op_to_insn_with_lowering<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if let Some(insn) =
        try_flatten_retired_family_hlop_to_insn(op, ctx, get_register, lower_constant)
    {
        return Some(insn);
    }
    flatten_op_to_insn(op, get_register, lower_constant)
}

/// Retired-family-only variant of [`flatten_op_to_insn_with_lowering`]:
/// dispatches to the four `lower_*_hlop_to_insn` helpers and returns
/// `None` when no arm matches, instead of falling through to the
/// passthrough [`flatten_op_to_insn`].
///
/// `GraphFlattener::flatten_space_operation` uses this variant to
/// avoid double-handling of non-HLOp ops — the caller's own
/// passthrough emits them, so the dispatcher must only claim the
/// four retired families and leave everything else to the caller.
pub fn try_flatten_retired_family_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if let Some(insn) = lower_binary_op_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_compare_op_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_bool_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_setitem_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_newlist_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    if let Some(insn) = lower_simple_call_hlop_to_insn(op, ctx, get_register, lower_constant) {
        return Some(insn);
    }
    None
}

/// Lower a CALL-family pre-rtype HLOp `simple_call(callable, arg0,
/// arg1, ..., argN-1)` → `result: Ref` to the equivalent post-rtype
/// `residual_call_r_r(ConstInt(call_fn_N_idx), ListR([callable,
/// arg0, ...]), Descr) → reg` Insn.  Mirrors the inline emit at
/// codewriter.rs:6171-6179 (`build_call_fn_residual_call_r_r_insn`).
///
/// Arity dispatch: nargs = op.args.len() - 1 selects
/// `ctx.call_fn_idx_by_nargs[nargs]`.  Walker contract: CALL with
/// nargs > 8 takes the `abort_permanent` branch (codewriter.rs:6118-
/// 6133) and does NOT record `simple_call` on the graph, so a
/// graph-side `simple_call` with nargs > 8 indicates walker
/// non-orthodoxy — return `None` (passthrough).
///
/// Returns `None` for non-`simple_call` opnames so the caller can
/// fall through to other lowering arms.
pub fn lower_simple_call_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "simple_call" {
        return None;
    }
    if op.args.is_empty() {
        return None;
    }
    let nargs = op.args.len() - 1;
    if nargs > 8 {
        return None;
    }
    // First arg is the callable, rest are call arguments.  All Ref.
    // Constant args are accepted via `lower_constant` (matches
    // `flatten.py:340-345 flatten_arg`'s Constant arm); upstream
    // RPython's rtype pass would have rewritten Constant args to
    // pre-loaded Variables, but pyre's graph carries the un-rewritten
    // Constants per [[project-flatten-graph-canonical-driver-2026-05-17]].
    let mut operands: Vec<Operand> = Vec::with_capacity(op.args.len());
    for arg in &op.args {
        let operand = match arg {
            super::flow::SpaceOperationArg::Value(super::flow::FlowValue::Variable(var)) => {
                Operand::Register(get_register(*var))
            }
            super::flow::SpaceOperationArg::Value(super::flow::FlowValue::Constant(c)) => {
                lower_constant(c)
            }
            _ => return None,
        };
        operands.push(operand);
    }
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
        _ => return None,
    };
    Some(build_residual_call_r_r_insn_from_operands(
        ctx.call_fn_idx_by_nargs[nargs],
        operands,
        CallFlavor::MayForce,
        dst_reg,
    ))
}

/// Lower a BUILD_LIST-family pre-rtype HLOp `newlist(items)` →
/// `result: Ref` to the equivalent post-rtype `residual_call_ir_r(
/// ConstInt(build_list_fn_idx), ListI([argc, dummies]),
/// ListR([item_regs]), Descr) → reg` Insn.  Mirrors the inline
/// emit at codewriter.rs:6390-6398
/// (`push_walker_emit(build_build_list_fn_residual_call_ir_r_insn)`)
/// which pads unused item slots with `ConstInt(0)`.
///
/// Walker contract: `emit_frontend_newlist` only fires for argc ≤ 3
/// (codewriter.rs:6332-6346 — argc > 3 takes the `abort_permanent`
/// branch which does NOT record a `newlist` HLOp on the graph), so a
/// graph-side `newlist` with argc > 3 indicates a walker non-orthodoxy;
/// return `None` (passthrough) rather than asserting, matching the
/// other lowering arms' "no match → passthrough" pattern.
///
/// Returns `None` for non-`newlist` opnames so the caller can fall
/// through to other lowering arms.
pub fn lower_newlist_hlop_to_insn<F, LC>(
    op: &super::flow::SpaceOperation,
    ctx: &LoweringContext,
    get_register: &mut F,
    lower_constant: &mut LC,
) -> Option<Insn>
where
    F: FnMut(super::flow::Variable) -> Register,
    LC: FnMut(&Constant) -> Operand,
{
    if op.opname != "newlist" {
        return None;
    }
    let argc = op.args.len();
    if argc > 3 {
        return None;
    }
    // Walker emits each item Variable into a fresh Ref scratch via
    // `emit_ref_copy!(arg_regs[i], item_reg)` then passes those scratch
    // regs to the helper.  The canonical entry doesn't see those
    // inline `ref_copy`s — it reads the item Variables directly off
    // the SpaceOperation and resolves them through `get_register`.
    // Constant items lower via `lower_constant` per
    // `flatten.py:340-345 flatten_arg`'s Constant arm.
    let item_operands: Vec<Operand> = op
        .args
        .iter()
        .map(|arg| match arg {
            super::flow::SpaceOperationArg::Value(super::flow::FlowValue::Variable(var)) => {
                Some(Operand::Register(get_register(*var)))
            }
            super::flow::SpaceOperationArg::Value(super::flow::FlowValue::Constant(c)) => {
                Some(lower_constant(c))
            }
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let dst_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var).index,
        _ => return None,
    };
    Some(build_build_list_fn_residual_call_ir_r_insn_from_operands(
        ctx.build_list_fn_idx,
        argc,
        item_operands,
        dst_reg,
    ))
}

/// Construct the SETITEM-family `residual_call_r_v` Insn from raw
/// register indices.  Production codewriter callsite (Slice 6
/// retirement) bypasses the SpaceOperation→Insn round-trip and
/// emits this Insn directly into the SSARepr, replacing the prior
/// `emit_residual_call(store_subscr_fn_idx, ...)` + matching graph
/// dual-write at codewriter.rs:5244-5282.
pub fn build_store_subscr_fn_residual_call_r_v_insn(
    store_subscr_fn_idx: u16,
    obj_reg: u16,
    key_reg: u16,
    value_reg: u16,
) -> Insn {
    build_residual_call_r_v_insn_from_operands(
        store_subscr_fn_idx,
        vec![
            Operand::Register(Register::new(Kind::Ref, obj_reg)),
            Operand::Register(Register::new(Kind::Ref, key_reg)),
            Operand::Register(Register::new(Kind::Ref, value_reg)),
        ],
        CallFlavor::MayForce,
    )
}

/// Construct the set_current_exception-family `residual_call_r_v`
/// Insn from a raw register index.  Production codewriter callsites
/// replace the prior
/// `emit_residual_call(set_current_exception_fn_idx, ...)` SSARepr
/// emits at codewriter.rs:6134-6144 (PushExcInfo) and
/// codewriter.rs:6269-6279 (PopExcept).  Both sites' graph
/// dual-writes stay in place.
///
/// `set_current_exception_fn` has signature `(exc: Ref) → Void` with
/// `CallFlavor::PlainCannotRaiseNoHeap` (per codewriter.rs:2253-2258 —
/// TLS write to `CURRENT_EXCEPTION`; `EF_CANNOT_RAISE`, no heap access,
/// no GC).  Same opname `residual_call_r_v` as SETITEM but fixed-arity
/// 1 vs SETITEM's 3, plus PlainCannotRaiseNoHeap flavor vs SETITEM's
/// MayForce.
pub fn build_set_current_exception_fn_residual_call_r_v_insn(
    set_current_exception_fn_idx: u16,
    exc_reg: u16,
) -> Insn {
    build_residual_call_r_v_insn_from_operands(
        set_current_exception_fn_idx,
        vec![Operand::Register(Register::new(Kind::Ref, exc_reg))],
        CallFlavor::PlainCannotRaiseNoHeap,
    )
}

/// Shared shape constructor for `(Ref, ..., Ref) → Void` HLOp /
/// helper families that lower to a uniform `residual_call_r_v` Insn
/// shape.  Today: SETITEM (`store_subscr_fn`, MayForce, 3-Ref) and
/// set_current_exception (`set_current_exception_fn`,
/// PlainCannotRaiseNoHeap, 1-Ref).  `arg_kinds = vec![Kind::Ref;
/// ref_operands.len()]`, ResKind = Void → kinds `"r"` + reskind
/// `'v'` → opname `"residual_call_r_v"`.  No leading `ListI`
/// (empty `args_i`); no trailing result Register (Void).
/// Variable-arity: caller supplies `ref_operands` of any length.
fn build_residual_call_r_v_insn_from_operands(
    fn_idx: u16,
    ref_operands: Vec<Operand>,
    flavor: CallFlavor,
) -> Insn {
    let arg_kinds = vec![Kind::Ref; ref_operands.len()];
    let effect_info = effect_info_for_call_flavor(flavor);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds,
        result_kind: None,
    }));
    Insn::op(
        "residual_call_r_v",
        vec![
            Operand::ConstInt(fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, ref_operands)),
            descr_operand,
        ],
    )
}

/// Construct the LoadConst-family `residual_call_ir_r` Insn from raw
/// register indices.  Production codewriter callsite replaces the prior `emit_residual_call(
/// load_const_fn_idx, ...)` SSARepr emit at codewriter.rs:4933-4946
/// with a single direct push of this helper's output.  The matching
/// graph dual-write at codewriter.rs:4954-4965 stays in place — this
/// slice is incremental factor refactor, not retirement.
///
/// `load_const_fn` has signature `(pycode: Ref, idx: Int) → Ref` with
/// `CallFlavor::Plain` (per codewriter.rs:2207-2215 — `load_const_fn`
/// re-materializes int/float/str/bool constants per call but never
/// runs user `__bool__`/`__init__`, so it cannot force virtuals).
/// Distinct from BINARY_OP/COMPARE_OP's `_ir_r` arity (`(Ref, Ref,
/// Int) → Ref`, MayForce) — same opname `residual_call_ir_r` (kinds
/// `ir` + reskind `r`) but different `arg_kinds` and flavor.
///
/// Inline arg order from `emit_residual_call_shape` for call-args
/// `[Reg(Ref, pycode), ConstInt(idx)]`: `args_i = [ConstInt(idx)]`,
/// `args_r = [Reg(pycode)]`, `args_f = []` → final SSARepr Insn
/// `[ConstInt(fn_idx), ListI([ConstInt(idx)]), ListR([Reg(pycode)]),
/// Descr] → Reg(Ref, dst)`.
///
/// LoadConst has no frontend HLOp (the graph dual-write at
/// codewriter.rs:4954-4965 IS the canonical post-rtype graph
/// representation), so this slice adds no probe-side
/// `lower_load_const_hlop_to_insn` — only the production-side
/// builder.  Future `flatten_graph(graph, regallocs)` migration can
/// reuse this helper without further refactor.
pub fn build_load_const_fn_residual_call_ir_r_insn(
    load_const_fn_idx: u16,
    idx: i64,
    pycode_reg: u16,
    dst_reg: u16,
) -> Insn {
    build_residual_call_ir_r_single_ref_plain_insn_from_operands(
        load_const_fn_idx,
        idx,
        Operand::Register(Register::new(Kind::Ref, pycode_reg)),
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Shared shape constructor for the `(Ref, Int) → Ref` HLOp families
/// that lower to a uniform `residual_call_ir_r` Insn shape with
/// `CallFlavor::Plain`.  Today: LoadConst (`load_const_fn`).
/// `arg_kinds = [Ref, Int]`, flavor = `Plain`.  Distinct from
/// `build_residual_call_ir_r_insn_from_operands` which serves the
/// `(Ref, Ref, Int) → Ref` MayForce arity used by BINARY_OP /
/// COMPARE_OP.  Both produce opname `residual_call_ir_r` (kinds `ir`
/// + reskind `r`) but the bucketed argument layout and Descr
/// `arg_kinds` differ.
///
/// Inline arg order from `emit_residual_call_shape`:
///   * `args_i = [ConstInt(idx)]`
///   * `args_r = [Reg(arg_operand)]`
///   * `args_f = []`
/// → final SSARepr Insn `[ConstInt(fn_idx), ListI([ConstInt(idx)]),
///                         ListR([Reg(arg_operand)]), Descr]`.
fn build_residual_call_ir_r_single_ref_plain_insn_from_operands(
    fn_idx: u16,
    idx: i64,
    arg_operand: Operand,
    dst_reg: Register,
) -> Insn {
    let effect_info = effect_info_for_call_flavor(CallFlavor::Plain);
    let descr_operand = Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
        effect_info,
        arg_kinds: vec![Kind::Ref, Kind::Int],
        result_kind: Some(Kind::Ref),
    }));
    Insn::op_with_result(
        "residual_call_ir_r",
        vec![
            Operand::ConstInt(fn_idx as i64),
            Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![Operand::ConstInt(idx)])),
            Operand::ListOfKind(ListOfKind::new(Kind::Ref, vec![arg_operand])),
            descr_operand,
        ],
        dst_reg,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::flow::{FlowListOfKind, VariableId};

    #[test]
    fn pc_anchor_round_trip_matches_py_pc() {
        for py_pc in [0usize, 1, 42, 99, 123_456] {
            assert_eq!(pc_label_name(py_pc), format!("pc{py_pc}"));
            let insn = Insn::pc_anchor(py_pc);
            assert_eq!(
                label_pc_index(&insn),
                Some(py_pc),
                "round-trip py_pc={py_pc}"
            );
        }
    }

    #[test]
    fn label_pc_index_rejects_non_anchor_insns() {
        // Block / link / catch-landing labels must be rejected — only
        // `Insn::PcAnchor` returns Some(py_pc).
        for name in ["block0", "block42", "link1", "catch_landing_3"] {
            assert_eq!(
                label_pc_index(&Insn::Label(Label::new(name))),
                None,
                "label_pc_index must reject Label({name:?})"
            );
        }
        assert_eq!(label_pc_index(&Insn::Unreachable), None);
        assert_eq!(label_pc_index(&Insn::op("plain", vec![])), None);
    }

    #[test]
    fn call_flavor_round_trip_through_effect_info() {
        // call.py:282-303 maps each ExtraEffect to one CallFlavor; the
        // round-trip property is `dispatch_kind_for_effect_info(
        // effect_info_for_call_flavor(f)) == f` for every flavor whose
        // EI carries a structurally-distinct `extraeffect`.
        //
        // ReleaseGil is excluded because it needs the real
        // `(target_fn_addr, save_err)` seed.
        //
        // `Plain` → `EF_CAN_RAISE` (`call.py:300-301`, fed through
        // `effectinfo_from_writeanalyze` with the
        // `graphanalyze.py:60 analyze_external_call` default
        // `bottom_result()` = empty set; the `effectinfo.py:285`
        // top_set force-promotion only fires when `effects is top_set`,
        // not when the producer simply omits an analyzer). dispatch
        // reverses via the `_ => Plain` arm.
        //
        // `PlainCannotRaise` → `EF_CANNOT_RAISE` (`call.py:303`),
        // dispatch reverses via the `ExtraEffect::CannotRaise =>
        // PlainCannotRaise` arm.
        //
        // `MayForce` → ForcesVirtualOrVirtualizable; `LoopInvariant` →
        // LoopInvariant; `Pure*` → ElidableCannotRaise /
        // ElidableOrMemoryError / ElidableCanRaise. PlainCannotRaiseNoHeap
        // is excluded — it shares ExtraEffect::CannotRaise with
        // PlainCannotRaise; the heap distinction lives in the bitstrings,
        // not the extraeffect discriminant, so the round-trip correctly
        // collapses to PlainCannotRaise.
        for flavor in [
            CallFlavor::Plain,
            CallFlavor::PlainCannotRaise,
            CallFlavor::MayForce,
            CallFlavor::LoopInvariant,
            CallFlavor::PureCannotRaise,
            CallFlavor::PureOrMemerror,
            CallFlavor::PureCanRaise,
        ] {
            let ei = effect_info_for_call_flavor(flavor);
            assert_eq!(
                dispatch_kind_for_effect_info(&ei),
                flavor,
                "round-trip mismatch for {flavor:?}"
            );
        }

        // `PlainCannotRaiseNoHeap` resolves to `CANNOT_RAISE_NO_HEAP_EFFECT_INFO`
        // (`extraeffect=CannotRaise`); `dispatch_kind_for_effect_info`
        // discriminates only on `extraeffect` so the reverse mapping
        // collapses to `PlainCannotRaise`. Distinct flavors at the
        // codewriter producer side, identical post-trace EI dispatch.
        let ei = effect_info_for_call_flavor(CallFlavor::PlainCannotRaiseNoHeap);
        assert_eq!(
            dispatch_kind_for_effect_info(&ei),
            CallFlavor::PlainCannotRaise,
            "PlainCannotRaiseNoHeap shares extraeffect=CannotRaise with \
             PlainCannotRaise; dispatch reverse-maps both to PlainCannotRaise"
        );
    }

    #[test]
    fn unresolved_release_gil_effect_info_routes_to_release_gil_dispatch() {
        let ei = unresolved_release_gil_effect_info_for_via_target();
        assert_eq!(dispatch_kind_for_effect_info(&ei), CallFlavor::ReleaseGil);
        assert_eq!(ei.call_release_gil_target, (1, 0));
    }

    #[test]
    fn analyzer_absent_plain_and_may_force_carry_distinct_extra_effects() {
        // `call.py:288-303 getcalldescr` keeps `EF_CAN_RAISE`
        // (plain raising callees) and `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
        // (virtualizable forcing callees) as distinct extraeffect
        // values.  Collapsing them both onto `EF_RANDOM_EFFECTS = 7`
        // (`MOST_GENERAL`) would over-claim random-effects semantics on
        // plain calls, routing them through
        // `check_forces_virtual_or_virtualizable()` (`pyjitpl.py:2007`,
        // `effectinfo.py:250`) and tripping `has_random_effects()`
        // cache invalidation that `EF_CAN_RAISE` / `EF_FORCES` leave
        // intact.
        let plain_ei = effect_info_for_call_flavor(CallFlavor::Plain);
        let may_force_ei = effect_info_for_call_flavor(CallFlavor::MayForce);
        assert_ne!(plain_ei, may_force_ei);
        assert_eq!(plain_ei.extraeffect, majit_ir::ExtraEffect::CanRaise);
        assert_eq!(
            may_force_ei.extraeffect,
            majit_ir::ExtraEffect::ForcesVirtualOrVirtualizable
        );
        assert_eq!(dispatch_kind_for_effect_info(&plain_ei), CallFlavor::Plain);
        assert_eq!(
            dispatch_kind_for_effect_info(&may_force_ei),
            CallFlavor::MayForce
        );
    }

    #[test]
    fn intern_call_descr_stub_dedupes_by_effect_and_arg_kinds_and_result_kind() {
        // Two calls with the same (EffectInfo, Vec<Kind>, Option<Kind>)
        // must return the same `Arc` (shared identity for graph-side
        // recorders).
        let ei = effect_info_for_call_flavor(CallFlavor::Plain);
        let kinds = vec![Kind::Int, Kind::Ref];
        let a = intern_call_descr_stub(ei.clone(), kinds.clone(), Some(Kind::Ref));
        let b = intern_call_descr_stub(ei.clone(), kinds.clone(), Some(Kind::Ref));
        assert!(
            std::sync::Arc::ptr_eq(&a, &b),
            "intern_call_descr_stub must dedupe by (effect_info, arg_kinds, result_kind)"
        );

        // Different arg_kinds → distinct Arc.
        let c = intern_call_descr_stub(ei.clone(), vec![Kind::Ref, Kind::Int], Some(Kind::Ref));
        assert!(
            !std::sync::Arc::ptr_eq(&a, &c),
            "different arg_kinds must produce distinct Arcs"
        );

        // Different result_kind → distinct Arc.  Mirrors RPython
        // `gccache._cache_call` keying by `result_type`: two stubs
        // sharing `(arg_classes, extrainfo)` but differing in result
        // type are distinct cache entries upstream.
        let d = intern_call_descr_stub(ei.clone(), kinds.clone(), Some(Kind::Int));
        assert!(
            !std::sync::Arc::ptr_eq(&a, &d),
            "different result_kind (Ref vs Int) must produce distinct Arcs"
        );

        // void result_kind also distinct.
        let e = intern_call_descr_stub(ei, kinds, None);
        assert!(
            !std::sync::Arc::ptr_eq(&a, &e),
            "void result (None) must be distinct from typed result (Some)"
        );
    }

    #[test]
    fn flatten_descr_by_ptr_recognises_call_descr_stub() {
        // Round-trip property: an interned `CallDescrStub` lowered via
        // `flatten_descr_by_ptr` must produce a structurally-equivalent
        // SSA-side `Operand::Descr(DescrOperand::CallDescrStub(_))`.
        let ei = effect_info_for_call_flavor(CallFlavor::PureCanRaise);
        let kinds = vec![Kind::Int, Kind::Ref, Kind::Float];
        let arc = intern_call_descr_stub(ei.clone(), kinds.clone(), Some(Kind::Float));
        let by_ptr = super::super::flow::DescrByPtr(arc);
        match flatten_descr_by_ptr(&by_ptr) {
            Operand::Descr(rc) => match &*rc {
                DescrOperand::CallDescrStub(stub) => {
                    assert_eq!(stub.effect_info, ei);
                    assert_eq!(stub.arg_kinds, kinds);
                }
                other => panic!("expected DescrOperand::CallDescrStub, got {other:?}"),
            },
            other => panic!("expected Operand::Descr, got {other:?}"),
        }
    }

    #[test]
    fn plain_cannot_raise_no_heap_skips_check_can_raise() {
        // effectinfo.py:236 `check_can_raise(self, ignore_memoryerror=False)`:
        //   `return self.extraeffect > self.EF_CANNOT_RAISE`
        // EF_CANNOT_RAISE == 2, so the analyzer-confirmed-clean cannot-raise
        // shape (`PlainCannotRaiseNoHeap` → `CANNOT_RAISE_NO_HEAP_EFFECT_INFO`)
        // reads False — the canonical walker uses this to drop
        // `GUARD_NO_EXCEPTION` (`pyjitpl.py:2111-2115 do_residual_call`).
        let ei = effect_info_for_call_flavor(CallFlavor::PlainCannotRaiseNoHeap);
        assert!(!ei.check_can_raise(false));

        // `PlainCannotRaise` carries `extraeffect=CannotRaise` (== 2),
        // so `check_can_raise(false)` reads False, matching
        // `PlainCannotRaiseNoHeap`. The two flavors differ only in the
        // analyzer-confirmed empty raw-set/`can_collect=false` shape;
        // both drop `GUARD_NO_EXCEPTION` per `effectinfo.py:236`.
        let ei = effect_info_for_call_flavor(CallFlavor::PlainCannotRaise);
        assert!(
            !ei.check_can_raise(false),
            "PlainCannotRaise carries EF_CANNOT_RAISE; \
             check_can_raise(false) must be false per `effectinfo.py:236`"
        );

        // `Plain` carries `extraeffect=CanRaise` (== 5) per `call.py:300-301`.
        // `5 > 2` so `check_can_raise(false)` reads True — the walker
        // records `GUARD_NO_EXCEPTION` for plain raising callees.
        let ei = effect_info_for_call_flavor(CallFlavor::Plain);
        assert!(
            ei.check_can_raise(false),
            "Plain carries EF_CAN_RAISE; check_can_raise(false) must \
             be true per `effectinfo.py:236`"
        );
    }

    #[test]
    fn register_repr_matches_rpython() {
        // RPython `flatten.py:33` `return "%%%s%d" % (self.kind[0], self.index)`.
        assert_eq!(Register::new(Kind::Int, 0).kind.first_char(), 'i');
        assert_eq!(Register::new(Kind::Ref, 3).kind.first_char(), 'r');
        assert_eq!(Register::new(Kind::Float, 7).kind.first_char(), 'f');
    }

    #[test]
    fn kind_as_str_roundtrip() {
        // RPython `flatten.py:59` `KINDS = ['int', 'ref', 'float']`.
        assert_eq!(Kind::Int.as_str(), "int");
        assert_eq!(Kind::Ref.as_str(), "ref");
        assert_eq!(Kind::Float.as_str(), "float");
    }

    #[test]
    fn label_equality_follows_name() {
        // RPython `flatten.py:17-18` eq compares `.name`.
        assert_eq!(Label::new("foo"), Label::new("foo"));
        assert_ne!(Label::new("foo"), Label::new("bar"));
    }

    #[test]
    fn tlabel_equality_follows_name() {
        assert_eq!(TLabel::new("foo"), TLabel::new("foo"));
        assert_ne!(TLabel::new("foo"), TLabel::new("bar"));
    }

    #[test]
    fn descr_vable_array_field_helper_wraps_index() {
        // `rpython/jit/metainterp/virtualizable.py:73 array_field_descrs[i]`
        // is carried at SSARepr level via `DescrOperand::VableArrayField(i)`.
        match Operand::descr_vable_array_field(0) {
            Operand::Descr(rc) => match &*rc {
                DescrOperand::VableArrayField(idx) => assert_eq!(*idx, 0),
                other => panic!("expected VableArrayField(0), got {other:?}"),
            },
            other => panic!("expected Operand::Descr, got {other:?}"),
        }
    }

    #[test]
    fn descr_vable_array_helper_wraps_index() {
        // `rpython/jit/metainterp/virtualizable.py:58 array_descrs[i]` is
        // carried at SSARepr level via `DescrOperand::VableArray(i)`,
        // paired with `VableArrayField(i)` at the trailing operand
        // position of every vable arrayitem op.
        match Operand::descr_vable_array(0) {
            Operand::Descr(rc) => match &*rc {
                DescrOperand::VableArray(idx) => assert_eq!(*idx, 0),
                other => panic!("expected VableArray(0), got {other:?}"),
            },
            other => panic!("expected Operand::Descr, got {other:?}"),
        }
    }

    #[test]
    fn descr_vable_static_field_helper_wraps_index() {
        // `rpython/jit/metainterp/virtualizable.py:71 static_field_descrs[i]`
        // is carried at SSARepr level via `DescrOperand::VableStaticField(i)`,
        // emitted as the trailing descr operand of `getfield_vable_<kind>`
        // (after `v_inst`) and `setfield_vable_<kind>` (after `v_inst,
        // v_value`) — `jtransform.py:846, :927`.
        for idx in [0u16, 2, 5] {
            match Operand::descr_vable_static_field(idx) {
                Operand::Descr(rc) => match &*rc {
                    DescrOperand::VableStaticField(stored) => assert_eq!(*stored, idx),
                    other => panic!("expected VableStaticField({idx}), got {other:?}"),
                },
                other => panic!("expected Operand::Descr, got {other:?}"),
            }
        }
    }

    #[test]
    fn graph_flattener_emits_loop_header_from_graph_op() {
        let op = SpaceOperation::new("loop_header", vec![Constant::signed(0).into()], None, 17);
        let mut ssarepr = SSARepr::new("test");
        let mut flattener = GraphFlattener::new(&mut ssarepr, |_| {
            Register::new(Kind::Ref, VariableId(0).0 as u16)
        });

        flattener.serialize_op(&op);

        match &ssarepr.insns[..] {
            [
                Insn::Op {
                    opname,
                    args,
                    result,
                },
            ] => {
                assert_eq!(opname, "loop_header");
                assert!(result.is_none());
                assert!(matches!(args.as_slice(), [Operand::ConstInt(0)]));
            }
            other => panic!("unexpected insns: {other:?}"),
        }
    }

    #[test]
    fn graph_flattener_preserves_jit_merge_point_graph_shape() {
        let frame = Variable::new(VariableId(10), Kind::Ref);
        let ec = Variable::new(VariableId(11), Kind::Ref);
        let op = SpaceOperation::new(
            "jit_merge_point",
            vec![
                Constant::signed(0).into(),
                FlowListOfKind::new(
                    Kind::Int,
                    vec![Constant::signed(17).into(), Constant::signed(0).into()],
                )
                .into(),
                FlowListOfKind::new(
                    Kind::Ref,
                    vec![Constant::opaque("pycode", Some(Kind::Ref)).into()],
                )
                .into(),
                FlowListOfKind::new(Kind::Float, vec![]).into(),
                FlowListOfKind::new(Kind::Int, vec![]).into(),
                FlowListOfKind::new(Kind::Ref, vec![frame.into(), ec.into()]).into(),
                FlowListOfKind::new(Kind::Float, vec![]).into(),
            ],
            None,
            3,
        );
        let mut ssarepr = SSARepr::new("test");
        let mut flattener = GraphFlattener::new_with_constant_lowering(
            &mut ssarepr,
            |v| Register::new(v.kind.expect("typed variable"), v.id.0 as u16),
            |constant| match (&constant.value, constant.kind) {
                (ConstantValue::Signed(value), Some(Kind::Int)) => Operand::ConstInt(*value),
                (ConstantValue::Opaque(_), Some(Kind::Ref)) => Operand::ConstRef(99),
                other => panic!("unexpected test constant {other:?}"),
            },
        );

        flattener.serialize_op(&op);

        match &ssarepr.insns[..] {
            [
                Insn::Op {
                    opname,
                    args,
                    result,
                },
            ] => {
                assert_eq!(opname, "jit_merge_point");
                assert!(result.is_none());
                assert_eq!(args.len(), 7);
                assert!(matches!(args[0], Operand::ConstInt(0)));
                assert!(matches!(
                    &args[1],
                    Operand::ListOfKind(ListOfKind { kind: Kind::Int, content })
                        if matches!(content.as_slice(), [Operand::ConstInt(17), Operand::ConstInt(0)])
                ));
                assert!(matches!(
                    &args[2],
                    Operand::ListOfKind(ListOfKind { kind: Kind::Ref, content })
                        if matches!(content.as_slice(), [Operand::ConstRef(99)])
                ));
                assert!(matches!(
                    &args[5],
                    Operand::ListOfKind(ListOfKind { kind: Kind::Ref, content })
                        if matches!(
                            content.as_slice(),
                            [
                                Operand::Register(Register { kind: Kind::Ref, index: 10 }),
                                Operand::Register(Register { kind: Kind::Ref, index: 11 })
                            ]
                        )
                ));
            }
            other => panic!("unexpected insns: {other:?}"),
        }
    }

    #[test]
    fn flatten_graph_walks_all_blocks_and_emits_each_op() {
        // Synthetic graph with two blocks; each contains a loop_header
        // op with a distinct offset tag.  flatten_graph must walk every
        // block and emit one Insn per SpaceOperation.
        use crate::jit::flow::{Block, FunctionGraph};
        let start_arg = Variable::new(VariableId(0), Kind::Ref);
        let next_arg = Variable::new(VariableId(1), Kind::Ref);
        let start = Block::shared(vec![start_arg.into()]);
        let mut graph = FunctionGraph::new("flat_walk", start.clone(), None);
        let next = graph.new_block(vec![next_arg.into()]);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("loop_header", vec![Constant::signed(0).into()], None, 0),
        );
        super::super::flow::push_op(
            &next,
            SpaceOperation::new("loop_header", vec![Constant::signed(0).into()], None, 1),
        );
        start.closeblock(vec![
            super::super::flow::Link::new(vec![start_arg.into()], Some(next.clone()), None)
                .into_ref(),
        ]);
        next.closeblock(vec![
            super::super::flow::Link::new(
                vec![next_arg.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let mut ssarepr = SSARepr::new("flat_walk");
        flatten_graph_with_closures(
            &graph,
            &mut ssarepr,
            |v| Register::new(v.kind.unwrap_or(Kind::Ref), v.id.0 as u16),
            flatten_constant_operand,
        );

        // Two loop_header Insns emitted — one per block.
        let header_count = ssarepr
            .insns
            .iter()
            .filter(|i| matches!(i, Insn::Op { opname, .. } if opname == "loop_header"))
            .count();
        assert_eq!(
            header_count, 2,
            "flatten_graph should emit one Insn per SpaceOperation across all blocks; got {:?}",
            ssarepr.insns
        );
    }

    #[test]
    fn flatten_graph_inserts_renamings_for_fallthrough_links() {
        use crate::jit::flow::{Block, FunctionGraph, Link};
        let src = Variable::new(VariableId(0), Kind::Ref);
        let dst = Variable::new(VariableId(1), Kind::Ref);
        let start = Block::shared(vec![src.into()]);
        let mut graph = FunctionGraph::new("renaming", start.clone(), Some(dst));
        let middle = graph.new_block(vec![dst.into()]);
        start.closeblock(vec![
            Link::new(vec![src.into()], Some(middle.clone()), None).into_ref(),
        ]);
        middle.closeblock(vec![
            Link::new(vec![dst.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let mut ssarepr = SSARepr::new("renaming");
        flatten_graph_with_closures(
            &graph,
            &mut ssarepr,
            |v| Register::new(v.kind.expect("typed variable"), v.id.0 as u16),
            flatten_constant_operand,
        );

        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op {
                    opname,
                    args,
                    result: Some(Register {
                        kind: Kind::Ref,
                        index: 1
                    }),
                } if opname == "ref_copy"
                    && matches!(
                        args.as_slice(),
                        [Operand::Register(Register {
                            kind: Kind::Ref,
                            index: 0
                        })]
                    )
            )
        }));
    }

    #[test]
    fn flatten_graph_emits_exception_dispatch_and_last_exc_loads() {
        use crate::jit::flow::{Block, ExitSwitch, FunctionGraph, Link, c_last_exception};
        let normal = Variable::new(VariableId(0), Kind::Ref);
        let exc_type = Variable::new(VariableId(1), Kind::Int);
        let exc_value = Variable::new(VariableId(2), Kind::Ref);
        let catch_type = Variable::new(VariableId(3), Kind::Int);
        let catch_value = Variable::new(VariableId(4), Kind::Ref);
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("exc_dispatch", start.clone(), Some(normal));
        let typed_handler = graph.new_block(vec![exc_type.into(), exc_value.into()]);
        let catchall_handler = graph.new_block(vec![catch_type.into(), catch_value.into()]);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("call_can_raise", Vec::new(), None, 0),
        );
        // `flatten.py:206-210` recognises a canraise block as "actually
        // raising" only when there is a trailing `-live-` marker after
        // the raising op (the rtyper emits one per call/raisecheck;
        // pyre's frontend uses the same convention).  Append it so
        // `insert_exits` takes the catch_exception emission path.
        super::super::flow::push_op(
            &start,
            SpaceOperation::new(crate::jit::flatten::OPNAME_LIVE, Vec::new(), None, 0),
        );

        typed_handler.closeblock(vec![
            Link::new(
                vec![exc_value.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        catchall_handler.closeblock(vec![
            Link::new(
                vec![catch_value.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(c_last_exception().into()));
        let normal_link = Link::new(
            vec![Constant::none().into()],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        let mut typed_link = Link::new(
            vec![exc_type.into(), exc_value.into()],
            Some(typed_handler.clone()),
            None,
        )
        .with_llexitcase(Constant::signed(7).into());
        typed_link.extravars(Some(exc_type), Some(exc_value));
        let mut catchall_link = Link::new(
            vec![catch_type.into(), catch_value.into()],
            Some(catchall_handler.clone()),
            None,
        );
        catchall_link.extravars(Some(catch_type), Some(catch_value));
        start.closeblock(vec![
            normal_link,
            typed_link.into_ref(),
            catchall_link.into_ref(),
        ]);

        let mut ssarepr = SSARepr::new("exc_dispatch");
        flatten_graph_with_closures(
            &graph,
            &mut ssarepr,
            |v| Register::new(v.kind.expect("typed variable"), v.id.0 as u16),
            flatten_constant_operand,
        );

        assert!(
            ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "catch_exception"))
        );
        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "goto_if_exception_mismatch"
                        && matches!(args.as_slice(), [Operand::ConstInt(7), Operand::TLabel(_)])
            )
        }));
        assert!(
            ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "last_exception"))
        );
        assert!(
            ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "last_exc_value"))
        );
    }

    #[test]
    fn flatten_graph_emits_boolean_branch_exits() {
        use crate::jit::flow::{Block, Constant, ExitSwitch, FunctionGraph, Link};

        let cond = Variable::new(VariableId(0), Kind::Int);
        let retval = Variable::new(VariableId(1), Kind::Int);
        let start = Block::shared(vec![cond.into()]);
        let graph = FunctionGraph::new("bool_branch", start.clone(), Some(retval));

        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(cond.into()));
        let false_link = Link::new(
            vec![Constant::signed(0).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::bool(false).into()),
        )
        .with_llexitcase(Constant::bool(false).into())
        .into_ref();
        let true_link = Link::new(
            vec![Constant::signed(1).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::bool(true).into()),
        )
        .with_llexitcase(Constant::bool(true).into())
        .into_ref();
        start.closeblock(vec![false_link, true_link]);

        let mut ssarepr = SSARepr::new("bool_branch");
        flatten_graph_with_closures(
            &graph,
            &mut ssarepr,
            |v| Register::new(v.kind.expect("typed variable"), v.id.0 as u16),
            flatten_constant_operand,
        );

        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "goto_if_not"
                        && matches!(
                            args.as_slice(),
                            [Operand::Register(Register { kind: Kind::Int, index: 0 }), Operand::TLabel(_)]
                        )
            )
        }));
        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(insn, Insn::Op { opname, args, .. }
                if opname == "int_return"
                    && matches!(args.as_slice(), [Operand::ConstInt(0)] | [Operand::ConstInt(1)]))
        }));
    }

    #[test]
    fn flatten_graph_emits_integer_switch_exits() {
        use crate::jit::flow::{Block, Constant, ExitSwitch, FunctionGraph, Link};

        let selector = Variable::new(VariableId(0), Kind::Int);
        let retval = Variable::new(VariableId(1), Kind::Int);
        let start = Block::shared(vec![selector.into()]);
        let graph = FunctionGraph::new("int_switch", start.clone(), Some(retval));

        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(selector.into()));
        let case_three = Link::new(
            vec![Constant::signed(30).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::signed(3).into()),
        )
        .with_llexitcase(Constant::signed(3).into())
        .into_ref();
        let case_one = Link::new(
            vec![Constant::signed(10).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::signed(1).into()),
        )
        .with_llexitcase(Constant::signed(1).into())
        .into_ref();
        let default = Link::new(
            vec![Constant::signed(99).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::string("default").into()),
        )
        .into_ref();
        start.closeblock(vec![case_three, case_one, default]);

        let mut ssarepr = SSARepr::new("int_switch");
        flatten_graph_with_closures(
            &graph,
            &mut ssarepr,
            |v| Register::new(v.kind.expect("typed variable"), v.id.0 as u16),
            flatten_constant_operand,
        );

        let switch = ssarepr
            .insns
            .iter()
            .find_map(|insn| match insn {
                Insn::Op { opname, args, .. } if opname == "switch" => Some(args),
                _ => None,
            })
            .expect("integer exits should lower to switch");
        assert!(matches!(
            switch.as_slice(),
            [
                Operand::Register(Register {
                    kind: Kind::Int,
                    index: 0
                }),
                Operand::Descr(_),
            ]
        ));
        let Operand::Descr(descr) = &switch[1] else {
            panic!("switch second operand must be SwitchDictDescr");
        };
        let DescrOperand::SwitchDict(switchdict) = descr.as_ref() else {
            panic!("switch second operand must be SwitchDictDescr");
        };
        let keys: Vec<_> = switchdict.labels.iter().map(|(key, _)| *key).collect();
        assert_eq!(keys, vec![1, 3]);
        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "int_return" && matches!(args.as_slice(), [Operand::ConstInt(99)])
            )
        }));
    }

    #[test]
    fn flatten_graph_emits_unreachable_op_and_marker_for_switch_without_default() {
        use crate::jit::flow::{Block, Constant, ExitSwitch, FunctionGraph, Link};

        let selector = Variable::new(VariableId(0), Kind::Int);
        let retval = Variable::new(VariableId(1), Kind::Int);
        let start = Block::shared(vec![selector.into()]);
        let graph = FunctionGraph::new("int_switch_no_default", start.clone(), Some(retval));

        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(selector.into()));
        let case_one = Link::new(
            vec![Constant::signed(10).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::signed(1).into()),
        )
        .with_llexitcase(Constant::signed(1).into())
        .into_ref();
        let case_three = Link::new(
            vec![Constant::signed(30).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::signed(3).into()),
        )
        .with_llexitcase(Constant::signed(3).into())
        .into_ref();
        start.closeblock(vec![case_three, case_one]);

        let mut ssarepr = SSARepr::new("int_switch_no_default");
        flatten_graph_with_closures(
            &graph,
            &mut ssarepr,
            |v| Register::new(v.kind.expect("typed variable"), v.id.0 as u16),
            flatten_constant_operand,
        );

        let switch_idx = ssarepr
            .insns
            .iter()
            .position(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "switch"))
            .expect("integer exits should lower to switch");
        assert!(matches!(
            ssarepr.insns.get(switch_idx + 1),
            Some(Insn::Op { opname, args, result }) if opname == "unreachable" && args.is_empty() && result.is_none()
        ));
        assert!(matches!(
            ssarepr.insns.get(switch_idx + 2),
            Some(Insn::Unreachable)
        ));
    }

    #[test]
    fn flatten_graph_routes_str_default_link_to_default_branch() {
        // `flatten.py:280 if link.exitcase != 'default':` — the
        // catch-all switch link is identified by the string sentinel
        // `"default"`.  Pyre's strict `is_default_exitcase`
        // (flatten.rs::is_default_exitcase) recognises ONLY this shape
        // (not `None`).  Verify a 3-exit switch with two Signed cases
        // plus one `Str("default")` catch-all lowers to switch + the
        // default link, not to switch + unreachable.
        use crate::jit::flow::{Block, Constant, ExitSwitch, FunctionGraph, Link};

        let selector = Variable::new(VariableId(0), Kind::Int);
        let retval = Variable::new(VariableId(1), Kind::Int);
        let start = Block::shared(vec![selector.into()]);
        let graph = FunctionGraph::new("int_switch_with_default", start.clone(), Some(retval));

        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(selector.into()));
        let case_one = Link::new(
            vec![Constant::signed(10).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::signed(1).into()),
        )
        .with_llexitcase(Constant::signed(1).into())
        .into_ref();
        let case_three = Link::new(
            vec![Constant::signed(30).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::signed(3).into()),
        )
        .with_llexitcase(Constant::signed(3).into())
        .into_ref();
        let default = Link::new(
            vec![Constant::signed(99).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::string("default").into()),
        )
        .into_ref();
        // `insert_switch_exits` reads the default off `exits.last()` so
        // place the catch-all at the tail of the exits vec.
        start.closeblock(vec![case_three, case_one, default]);

        let mut ssarepr = SSARepr::new("int_switch_with_default");
        flatten_graph_with_closures(
            &graph,
            &mut ssarepr,
            |v| Register::new(v.kind.expect("typed variable"), v.id.0 as u16),
            flatten_constant_operand,
        );

        let switch_idx = ssarepr
            .insns
            .iter()
            .position(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "switch"))
            .expect("integer exits should lower to switch");
        // No unreachable: default link supplies the catch-all branch.
        assert!(
            !matches!(
                ssarepr.insns.get(switch_idx + 1),
                Some(Insn::Op { opname, .. }) if opname == "unreachable"
            ),
            "Str(\"default\") link must replace the unreachable catch-all"
        );
    }

    #[test]
    #[should_panic(expected = "flatten_graph: switch link requires Signed/Bool llexitcase")]
    fn flatten_graph_panics_on_none_exitcase_switch_link() {
        // RPython `flatten.py:280` recognises ONLY `Str("default")` as
        // the switch catch-all; `None` exitcase on a switch link is a
        // walker non-orthodoxy.  After the strict
        // `is_default_exitcase` change, a `None` link reaches
        // `switch_llexitcase_key`, which fails loud on a non-
        // Signed/Bool llexitcase.  The panic surfaces the malformed
        // shape so the walker site producing `None` can be fixed
        // (either to set `Str("default")` for true catch-alls, or to
        // ensure both bool-branch links carry `Bool` `llexitcase` so
        // the bool-branch path fires instead of the switch path).
        use crate::jit::flow::{Block, Constant, ExitSwitch, FunctionGraph, Link};

        let selector = Variable::new(VariableId(0), Kind::Int);
        let retval = Variable::new(VariableId(1), Kind::Int);
        let start = Block::shared(vec![selector.into()]);
        let graph = FunctionGraph::new("int_switch_none_exitcase", start.clone(), Some(retval));

        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(selector.into()));
        let case_one = Link::new(
            vec![Constant::signed(10).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::signed(1).into()),
        )
        .with_llexitcase(Constant::signed(1).into())
        .into_ref();
        // None exitcase + None llexitcase — the malformed shape.
        let none_link = Link::new(
            vec![Constant::signed(99).into()],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![case_one, none_link]);

        let mut ssarepr = SSARepr::new("int_switch_none_exitcase");
        flatten_graph_with_closures(
            &graph,
            &mut ssarepr,
            |v| Register::new(v.kind.expect("typed variable"), v.id.0 as u16),
            flatten_constant_operand,
        );
    }

    #[test]
    fn flatten_graph_emits_tuple_goto_if_not_exitswitch() {
        use crate::jit::flow::{
            Block, Constant, ExitSwitch, ExitSwitchElement, FunctionGraph, Link,
        };

        let ptr = Variable::new(VariableId(0), Kind::Ref);
        let retval = Variable::new(VariableId(1), Kind::Int);
        let start = Block::shared(vec![ptr.into()]);
        let graph = FunctionGraph::new("tuple_branch", start.clone(), Some(retval));

        start.borrow_mut().exitswitch = Some(ExitSwitch::Tuple(vec![
            ExitSwitchElement::Marker("ptr_nonzero".to_owned()),
            ExitSwitchElement::Value(ptr.into()),
            ExitSwitchElement::Marker("-live-before".to_owned()),
        ]));
        let false_link = Link::new(
            vec![Constant::signed(0).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::bool(false).into()),
        )
        .with_llexitcase(Constant::bool(false).into())
        .into_ref();
        let true_link = Link::new(
            vec![Constant::signed(1).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::bool(true).into()),
        )
        .with_llexitcase(Constant::bool(true).into())
        .into_ref();
        start.closeblock(vec![false_link, true_link]);

        let mut ssarepr = SSARepr::new("tuple_branch");
        flatten_graph_with_closures(
            &graph,
            &mut ssarepr,
            |v| Register::new(v.kind.expect("typed variable"), v.id.0 as u16),
            flatten_constant_operand,
        );

        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "goto_if_not_ptr_nonzero"
                        && matches!(
                            args.as_slice(),
                            [Operand::Register(Register { kind: Kind::Ref, index: 0 }), Operand::TLabel(_)]
                        )
            )
        }));
    }

    #[test]
    fn graph_flattener_emits_generic_result_op() {
        let src = Variable::new(VariableId(0), Kind::Ref);
        let dst = Variable::new(VariableId(1), Kind::Ref);
        let op = SpaceOperation::new("type", vec![src.into()], Some(dst.into()), 23);
        let mut ssarepr = SSARepr::new("generic");
        let mut flattener = GraphFlattener::new(&mut ssarepr, |variable| {
            Register::new(
                variable.kind.expect("test variable kind"),
                variable.id.0 as u16,
            )
        });

        flattener.serialize_op(&op);

        match &ssarepr.insns[..] {
            [
                Insn::Op {
                    opname,
                    args,
                    result: Some(result),
                },
            ] => {
                assert_eq!(opname, "type");
                assert!(matches!(
                    args.as_slice(),
                    [Operand::Register(Register {
                        kind: Kind::Ref,
                        index: 0
                    })]
                ));
                assert_eq!(*result, Register::new(Kind::Ref, 1));
            }
            other => panic!("unexpected insns: {other:?}"),
        }
    }

    #[test]
    fn flatten_graph_with_lowering_lowers_retired_family_hlops() {
        // a graph carrying one HLOp from each of the four
        // retired families must lower to the matching `residual_call_*`
        // Insn shape under `flatten_graph_with_lowering`.  Builds a
        // minimal start block with `add(lhs, rhs)` + `lt(lhs, rhs)` +
        // `bool(lhs)` + `setitem(lhs, rhs, val)` ops, runs the new
        // driver, then filters the SSARepr for residual_call_* Insns.
        use crate::jit::flow::{Block, FunctionGraph};
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let val = Variable::new(VariableId(2), Kind::Ref);
        let add_res = Variable::new(VariableId(3), Kind::Ref);
        let lt_res = Variable::new(VariableId(4), Kind::Ref);
        let bool_res = Variable::new(VariableId(5), Kind::Int);
        let start = Block::shared(vec![lhs.into(), rhs.into(), val.into()]);
        let graph = FunctionGraph::new("retired_families", start.clone(), None);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("add", vec![lhs.into(), rhs.into()], Some(add_res.into()), 0),
        );
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("lt", vec![lhs.into(), rhs.into()], Some(lt_res.into()), 1),
        );
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("bool", vec![lhs.into()], Some(bool_res.into()), 2),
        );
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("setitem", vec![lhs.into(), rhs.into(), val.into()], None, 3),
        );
        start.closeblock(vec![
            super::super::flow::Link::new(
                vec![add_res.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let ctx = LoweringContext {
            binary_op_fn_idx: 11,
            compare_op_fn_idx: 13,
            truth_fn_idx: 17,
            store_subscr_fn_idx: 19,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };

        let mut ssarepr = SSARepr::new("retired_families");
        flatten_graph_with_lowering(
            &graph,
            &mut ssarepr,
            ctx,
            None,
            |v| Register::new(v.kind.unwrap_or(Kind::Ref), v.id.0 as u16),
            flatten_constant_operand,
        );

        // BINARY_OP `add` → residual_call_ir_r with fn_idx=11.
        let binary = ssarepr.insns.iter().find(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "residual_call_ir_r"
                        && matches!(args.first(), Some(Operand::ConstInt(11)))
            )
        });
        assert!(
            binary.is_some(),
            "expected BINARY_OP residual_call: {:?}",
            ssarepr.insns
        );

        // COMPARE_OP `lt` → residual_call_ir_r with fn_idx=13.
        let compare = ssarepr.insns.iter().find(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "residual_call_ir_r"
                        && matches!(args.first(), Some(Operand::ConstInt(13)))
            )
        });
        assert!(
            compare.is_some(),
            "expected COMPARE_OP residual_call: {:?}",
            ssarepr.insns
        );

        // BOOL `bool` → residual_call_r_i with fn_idx=17.
        let bool_call = ssarepr.insns.iter().find(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "residual_call_r_i"
                        && matches!(args.first(), Some(Operand::ConstInt(17)))
            )
        });
        assert!(
            bool_call.is_some(),
            "expected BOOL residual_call: {:?}",
            ssarepr.insns
        );

        // SETITEM `setitem` → residual_call_r_v with fn_idx=19.
        let setitem = ssarepr.insns.iter().find(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "residual_call_r_v"
                        && matches!(args.first(), Some(Operand::ConstInt(19)))
            )
        });
        assert!(
            setitem.is_some(),
            "expected SETITEM residual_call: {:?}",
            ssarepr.insns
        );

        // Passthrough opnames (the four `add`/`lt`/`bool`/`setitem`)
        // must NOT appear as raw Insn opnames — the dispatcher
        // intercepted them.
        for raw in ["add", "lt", "bool", "setitem"] {
            let leaked = ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == raw));
            assert!(
                !leaked,
                "{raw:?} HLOp leaked through passthrough: {:?}",
                ssarepr.insns
            );
        }
    }

    #[test]
    fn flatten_graph_with_lowering_byte_equivalent_across_blocks() {
        // Pin that the per-family `(opname, fn_idx)` lowering survives the
        // GraphFlattener's `make_link` / `insert_exits` block boundary
        // emission (Labels, terminators, link renamings) without dropping
        // or reordering the retired-family residual_calls.  A 2-block
        // graph with one BINARY_OP `add` in the start block and one
        // COMPARE_OP `lt` in the second block must lower to a single
        // `residual_call_ir_r` per block in start-then-next order.
        use crate::jit::flow::{Block, FunctionGraph, Link};
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let add_res = Variable::new(VariableId(2), Kind::Ref);
        let lt_res = Variable::new(VariableId(3), Kind::Ref);
        let start = Block::shared(vec![lhs.into(), rhs.into()]);
        let mut graph = FunctionGraph::new("multi_block_lowering", start.clone(), None);
        let next = graph.new_block(vec![lhs.into(), rhs.into()]);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("add", vec![lhs.into(), rhs.into()], Some(add_res.into()), 0),
        );
        super::super::flow::push_op(
            &next,
            SpaceOperation::new("lt", vec![lhs.into(), rhs.into()], Some(lt_res.into()), 1),
        );
        start.closeblock(vec![
            Link::new(vec![lhs.into(), rhs.into()], Some(next.clone()), None).into_ref(),
        ]);
        next.closeblock(vec![
            Link::new(vec![lt_res.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let ctx = LoweringContext {
            binary_op_fn_idx: 11,
            compare_op_fn_idx: 13,
            truth_fn_idx: 17,
            store_subscr_fn_idx: 19,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };

        let mut ssarepr = SSARepr::new("multi_block_lowering");
        flatten_graph_with_lowering(
            &graph,
            &mut ssarepr,
            ctx,
            None,
            |v| Register::new(v.kind.unwrap_or(Kind::Ref), v.id.0 as u16),
            flatten_constant_operand,
        );

        // Filter the SSARepr by `(opname, fn_idx)` mirroring the
        // probe's per-family report.  Both families share the
        // `residual_call_ir_r` opname, so the leading ConstInt
        // arg distinguishes them.
        let binary: Vec<&Insn> = ssarepr
            .insns
            .iter()
            .filter(|insn| {
                matches!(
                    insn,
                    Insn::Op { opname, args, .. }
                        if opname == "residual_call_ir_r"
                            && matches!(args.first(), Some(Operand::ConstInt(11)))
                )
            })
            .collect();
        let compare: Vec<&Insn> = ssarepr
            .insns
            .iter()
            .filter(|insn| {
                matches!(
                    insn,
                    Insn::Op { opname, args, .. }
                        if opname == "residual_call_ir_r"
                            && matches!(args.first(), Some(Operand::ConstInt(13)))
                )
            })
            .collect();
        assert_eq!(
            binary.len(),
            1,
            "expected exactly one BINARY_OP residual_call across the 2-block graph: {:?}",
            ssarepr.insns
        );
        assert_eq!(
            compare.len(),
            1,
            "expected exactly one COMPARE_OP residual_call across the 2-block graph: {:?}",
            ssarepr.insns
        );

        // BINARY_OP (start block) must precede COMPARE_OP (next block)
        // in the emitted Insn stream.  GraphFlattener walks blocks in
        // DFS order from startblock, so start emits first.
        let binary_pos = ssarepr
            .insns
            .iter()
            .position(|insn| {
                matches!(
                    insn,
                    Insn::Op { opname, args, .. }
                        if opname == "residual_call_ir_r"
                            && matches!(args.first(), Some(Operand::ConstInt(11)))
                )
            })
            .expect("BINARY_OP residual_call must exist");
        let compare_pos = ssarepr
            .insns
            .iter()
            .position(|insn| {
                matches!(
                    insn,
                    Insn::Op { opname, args, .. }
                        if opname == "residual_call_ir_r"
                            && matches!(args.first(), Some(Operand::ConstInt(13)))
                )
            })
            .expect("COMPARE_OP residual_call must exist");
        assert!(
            binary_pos < compare_pos,
            "BINARY_OP must precede COMPARE_OP across block boundaries: pos {} vs {}",
            binary_pos,
            compare_pos
        );
    }

    #[test]
    #[should_panic(expected = "flatten.py:282 insert_switch_exits invariant")]
    fn flatten_graph_with_lowering_2_exit_no_exitswitch_panics() {
        // `flatten.py:282-309 insert_switch_exits` is only entered for
        // blocks that already carry a Variable exitswitch.  A 2-exit
        // block with `exitswitch = None` is a malformed graph shape
        // upstream would never produce; fail loud so the upstream
        // contract is preserved (codex review parity revert).
        use crate::jit::flow::{Block, FunctionGraph, Link};
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let add_res = Variable::new(VariableId(2), Kind::Ref);
        let lt_res = Variable::new(VariableId(3), Kind::Ref);
        let start = Block::shared(vec![lhs.into(), rhs.into()]);
        let mut graph = FunctionGraph::new("pyre_walker_2exit", start.clone(), None);
        let left = graph.new_block(vec![lhs.into(), rhs.into()]);
        let right = graph.new_block(vec![lhs.into(), rhs.into()]);
        super::super::flow::push_op(
            &left,
            SpaceOperation::new("add", vec![lhs.into(), rhs.into()], Some(add_res.into()), 0),
        );
        super::super::flow::push_op(
            &right,
            SpaceOperation::new("lt", vec![lhs.into(), rhs.into()], Some(lt_res.into()), 1),
        );
        start.closeblock(vec![
            Link::new(vec![lhs.into(), rhs.into()], Some(left.clone()), None).into_ref(),
            Link::new(vec![lhs.into(), rhs.into()], Some(right.clone()), None).into_ref(),
        ]);
        left.closeblock(vec![
            Link::new(vec![add_res.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);
        right.closeblock(vec![
            Link::new(vec![lt_res.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let ctx = LoweringContext {
            binary_op_fn_idx: 11,
            compare_op_fn_idx: 13,
            truth_fn_idx: 17,
            store_subscr_fn_idx: 19,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };

        let mut ssarepr = SSARepr::new("pyre_walker_2exit");
        flatten_graph_with_lowering(
            &graph,
            &mut ssarepr,
            ctx,
            None,
            |v| Register::new(v.kind.unwrap_or(Kind::Ref), v.id.0 as u16),
            flatten_constant_operand,
        );
    }

    #[test]
    fn flatten_graph_without_lowering_ctx_preserves_passthrough() {
        // when `flatten_graph` (no ctx) sees a retired-
        // family HLOp like `add`, the legacy passthrough must still
        // emit `Insn::op("add", ...)` — no silent rewrite via the
        // dispatcher.  This guards the "default GraphFlattener
        // produces opname-passthrough" invariant the existing 7
        // `flatten_graph_*` tests rely on.
        use crate::jit::flow::{Block, FunctionGraph};
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let start = Block::shared(vec![lhs.into(), rhs.into()]);
        let graph = FunctionGraph::new("passthrough", start.clone(), None);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("add", vec![lhs.into(), rhs.into()], Some(result.into()), 0),
        );
        start.closeblock(vec![
            super::super::flow::Link::new(
                vec![result.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let mut ssarepr = SSARepr::new("passthrough");
        flatten_graph_with_closures(
            &graph,
            &mut ssarepr,
            |v| Register::new(v.kind.unwrap_or(Kind::Ref), v.id.0 as u16),
            flatten_constant_operand,
        );

        let has_add_passthrough = ssarepr
            .insns
            .iter()
            .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "add"));
        assert!(
            has_add_passthrough,
            "legacy flatten_graph must preserve `add` opname passthrough: {:?}",
            ssarepr.insns
        );
        let has_residual_call = ssarepr.insns.iter().any(
            |insn| matches!(insn, Insn::Op { opname, .. } if opname.starts_with("residual_call_")),
        );
        assert!(
            !has_residual_call,
            "legacy flatten_graph must NOT lower `add` to residual_call: {:?}",
            ssarepr.insns
        );
    }

    #[test]
    fn canonical_flatten_graph_resolves_ovf_via_pre_resolved_exception_pointers() {
        // Combined Phase-4 pipeline: a graph with an `_ovf` rewrite
        // (canraise int_add_ovf with two exits → normal/overflow)
        // lowered through canonical `flatten_graph(graph, regallocs,
        // false, Some(&cpu))`.  The `handling_ovf=true` arm of
        // `make_exception_link` reaches `cpu.rtyper.exceptiondata.
        // get_standard_ll_exc_instance_by_class("OverflowError")` per
        // flatten.py:166-170 — when the exceptiondata has been
        // pre-resolved via `resolve_standard_exception_pointers`, the
        // returned Constant carries `Signed(pointer)` with `Kind::Ref`
        // (rtyped form) and `flatten_constant_operand` lowers it to
        // `Operand::ConstRef(pointer)` directly, without a per-flatten
        // `lower_constant` closure.
        use crate::jit::cpu::Cpu;
        use crate::jit::flow::{Block, ExitSwitch, FunctionGraph, Link, c_last_exception};
        use crate::jit::regalloc::perform_graph_register_allocation_all_kinds;

        let lhs = Variable::new(VariableId(0), Kind::Int);
        let rhs = Variable::new(VariableId(1), Kind::Int);
        let res = Variable::new(VariableId(2), Kind::Int);
        let except_etype = Variable::new(VariableId(3), Kind::Int);
        let except_evalue = Variable::new(VariableId(4), Kind::Ref);

        let start = Block::shared(vec![lhs.into(), rhs.into()]);
        let graph = FunctionGraph::new("canonical_ovf_with_resolved", start.clone(), Some(res));

        super::super::flow::push_op(
            &start,
            SpaceOperation::new(
                "int_add_ovf",
                vec![lhs.into(), rhs.into()],
                Some(res.into()),
                42,
            ),
        );
        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(c_last_exception().into()));

        let normal_link =
            Link::new(vec![res.into()], Some(graph.returnblock.clone()), None).into_ref();
        let mut ovf_link = Link::new(
            vec![except_etype.into(), except_evalue.into()],
            Some(graph.exceptblock.clone()),
            None,
        );
        ovf_link.extravars(Some(except_etype), Some(except_evalue));
        start.closeblock(vec![normal_link, ovf_link.into_ref()]);

        let mut cpu = Cpu::new();
        cpu.rtyper
            .exceptiondata
            .resolve_standard_exception_pointers(|name| match name {
                "OverflowError" => Some(0xface_beef),
                _ => None,
            });

        let mut regallocs = perform_graph_register_allocation_all_kinds(&graph);
        let ssarepr = super::flatten_graph(&graph, &mut regallocs, false, Some(&cpu));

        // handling_ovf=true emits `raise <ConstRef(pointer)>` per
        // flatten.py:165-170; the pointer is the pre-resolved value.
        let raise_args = ssarepr
            .insns
            .iter()
            .find_map(|insn| match insn {
                Insn::Op { opname, args, .. } if opname == "raise" => Some(args),
                _ => None,
            })
            .expect("canonical flatten_graph handling_ovf=true must emit `raise <const>`");
        assert_eq!(raise_args.len(), 1);
        assert!(
            matches!(raise_args[0], Operand::ConstRef(0xface_beef)),
            "raise must carry the pre-resolved OverflowError pointer 0xface_beef \
             (rtyped via resolve_standard_exception_pointers), got {:?}",
            raise_args[0]
        );
    }

    #[test]
    fn canonical_flatten_graph_lowers_retired_family_via_cpu_lowering_ctx() {
        // Production-shape pipeline: graph carries a BINARY_OP `add`
        // HLOp; canonical `flatten_graph(graph, regallocs, false,
        // Some(&cpu))` derives `LoweringContext` from
        // `cpu.lowering_ctx` and lowers the HLOp via the dispatcher
        // (`flatten_op_to_insn_with_lowering`) into the matching
        // `residual_call_ir_r` Insn shape.  This pins the post-Phase-3
        // pipeline equivalence between the walker's inline
        // `build_binary_op_residual_call_ir_r_insn` push and the
        // canonical entry's dispatcher-driven emission.
        use crate::jit::cpu::Cpu;
        use crate::jit::flow::{Block, FunctionGraph, Link};
        use crate::jit::regalloc::perform_graph_register_allocation_all_kinds;

        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let add_res = Variable::new(VariableId(2), Kind::Ref);
        let start = Block::shared(vec![lhs.into(), rhs.into()]);
        let graph = FunctionGraph::new("canonical_retired_family", start.clone(), None);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("add", vec![lhs.into(), rhs.into()], Some(add_res.into()), 0),
        );
        start.closeblock(vec![
            Link::new(vec![add_res.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let cpu = Cpu::new();
        *cpu.lowering_ctx.write().unwrap() = Some(LoweringContext {
            binary_op_fn_idx: 11,
            compare_op_fn_idx: 13,
            truth_fn_idx: 17,
            store_subscr_fn_idx: 19,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        });

        let mut regallocs = perform_graph_register_allocation_all_kinds(&graph);
        let ssarepr = super::flatten_graph(&graph, &mut regallocs, false, Some(&cpu));

        let has_binary_op_lowered = ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "residual_call_ir_r"
                        && matches!(args.first(), Some(Operand::ConstInt(11)))
            )
        });
        let has_raw_add = ssarepr
            .insns
            .iter()
            .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "add"));
        assert!(
            has_binary_op_lowered,
            "canonical flatten_graph with cpu.lowering_ctx must lower BINARY_OP \
             `add` HLOp to residual_call_ir_r(fn_idx=11, ...): {:?}",
            ssarepr.insns
        );
        assert!(
            !has_raw_add,
            "canonical flatten_graph must NOT passthrough `add` opname when \
             LoweringContext is present: {:?}",
            ssarepr.insns
        );
    }

    #[test]
    fn canonical_flatten_graph_lowers_rtyped_signed_ref_link_arg() {
        // End-to-end pipeline: graph carrying an already-rtyped
        // `Signed(0xfeed)/Ref` link arg (the post-rtype shape the
        // canonical `flatten_graph(graph, regallocs, false, cpu=None)`
        // entry expects) is lowered by `flatten_constant_operand` to
        // `Operand::ConstRef(0xfeed)`, and `insert_renamings` emits
        // a `ref_copy(ConstRef(0xfeed) -> Reg)` matching the
        // returnblock inputarg's color.  Test fixtures construct
        // post-rtype shapes directly; production callers route
        // pre-rtype `Opaque(Ref)` through the closure-based
        // `flatten_graph_with_lowering` path (a per-call
        // `lower_constant` closure handles the resolution).
        use crate::jit::flow::{Block, FunctionGraph, Link};
        use crate::jit::regalloc::perform_graph_register_allocation_all_kinds;

        let start = Block::shared(Vec::new());
        let graph = FunctionGraph::new("rtyped_canonical", start.clone(), None);
        let signed_ovf = super::super::flow::Constant::new(
            super::super::flow::ConstantValue::Signed(0xfeed),
            Some(Kind::Ref),
        );
        start.closeblock(vec![
            Link::new(
                vec![signed_ovf.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let mut regallocs = perform_graph_register_allocation_all_kinds(&graph);
        let ssarepr = super::flatten_graph(&graph, &mut regallocs, false, None);

        let has_const_ref_feed = ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op { args, .. }
                    if args.iter().any(|a| matches!(a, Operand::ConstRef(0xfeed)))
            )
        });
        assert!(
            has_const_ref_feed,
            "canonical flatten_graph must lower rtyped Signed(0xfeed)/Ref to \
             Operand::ConstRef(0xfeed): {:?}",
            ssarepr.insns
        );
    }

    /// Helper: build a `get_register` closure that maps each
    /// `Variable` to a `Register` whose index equals the
    /// `VariableId`, kind taken from the variable.  Used by the
    /// Task #48 lowering tests.
    fn identity_register_mapper() -> impl FnMut(Variable) -> Register {
        |variable: Variable| {
            Register::new(
                variable.kind.expect("test variable kind"),
                variable.id.0 as u16,
            )
        }
    }

    /// Test helper companion to `identity_register_mapper`: returns a
    /// closure suitable for passing as `lower_constant` to the
    /// `lower_*_hlop_to_insn` / dispatcher helpers.  Wraps
    /// `flatten_constant_operand_for_test`
    /// (`Opaque(Ref) → ConstRef(0)` placeholder) — the production-side
    /// `flatten_constant_operand` would panic on the placeholder
    /// fixtures these tests use.
    fn test_constant_lowering() -> impl FnMut(&Constant) -> Operand {
        flatten_constant_operand_for_test
    }

    #[test]
    fn binary_op_tag_for_opname_covers_runtime_ops_table() {
        // Task #48 micro-slice 1: tag mapping must agree with
        // `pyre_interpreter::runtime_ops::binary_op_tag` for every
        // BinaryOperator the codewriter encodes.  Keep the two tables
        // in lockstep — a divergence would record the wrong op_val on
        // the lowered Insn and silently miscompute at runtime.
        for (opname, expected) in [
            ("add", 0),
            ("inplace_add", 0),
            ("sub", 1),
            ("inplace_sub", 1),
            ("mul", 2),
            ("inplace_mul", 2),
            ("floordiv", 3),
            ("inplace_floordiv", 3),
            ("mod", 4),
            ("inplace_mod", 4),
            ("truediv", 5),
            ("inplace_truediv", 5),
            ("getitem", 6),
            ("pow", 7),
            ("inplace_pow", 7),
            ("lshift", 8),
            ("inplace_lshift", 8),
            ("rshift", 9),
            ("inplace_rshift", 9),
            ("and_", 10),
            ("inplace_and", 10),
            ("or_", 11),
            ("inplace_or", 11),
            ("xor", 12),
            ("inplace_xor", 12),
        ] {
            assert_eq!(
                binary_op_tag_for_opname(opname),
                Some(expected),
                "tag mismatch for opname {opname:?}",
            );
        }
        // Out-of-family opnames must return None so the lowering
        // arm falls through.
        for opname in [
            "lt",
            "bool",
            "neg",
            "simple_call",
            "setitem",
            "newlist",
            "newslice",
        ] {
            assert_eq!(
                binary_op_tag_for_opname(opname),
                None,
                "unexpected tag for non-BINARY_OP opname {opname:?}",
            );
        }
    }

    #[test]
    fn lower_binary_op_hlop_to_insn_emits_residual_call_ir_r() {
        // Task #48 micro-slice 1: lowering an `add(lhs, rhs) → result`
        // HLOp must produce the same Insn shape that
        // `emit_residual_call_shape` produces inline at the BINARY_OP
        // callsite (codewriter.rs:5335-5352): `residual_call_ir_r`
        // with args `[ConstInt(fn_idx), ListI([ConstInt(op_val)]),
        // ListR([lhs, rhs]), Descr(CallDescrStub)] → reg`.
        // (`emit_residual_call_shape` codewriter.rs:2745-2802 buckets
        // each call-arg by `Kind` then concatenates lists in
        // `i,r,f` order, so the `ConstInt(op_val)` rides inside `ListI`
        // — not as a trailing standalone `ConstInt`.)
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let op = SpaceOperation::new("add", vec![lhs.into(), rhs.into()], Some(result.into()), 42);
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            compare_op_fn_idx: 0,
            truth_fn_idx: 0,
            store_subscr_fn_idx: 0,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();

        let insn = lower_binary_op_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
            .expect("BINARY_OP HLOp must lower");

        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 2));
                assert_eq!(args.len(), 4);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 7),
                    other => panic!("expected ConstInt(7), got {other:?}"),
                }
                // args[1] = ListI([ConstInt(op_val)]).
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert_eq!(list.content.len(), 1);
                        match &list.content[0] {
                            Operand::ConstInt(v) => assert_eq!(*v, 0, "add → tag 0"),
                            other => panic!("expected ConstInt(0) in ListI, got {other:?}"),
                        }
                    }
                    other => panic!("expected ListOfKind(Int, 1), got {other:?}"),
                }
                // args[2] = ListR([Reg(lhs), Reg(rhs)]).
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 2);
                        assert!(matches!(
                            &list.content[0],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 0
                            })
                        ));
                        assert!(matches!(
                            &list.content[1],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 1
                            })
                        ));
                    }
                    other => panic!("expected ListOfKind(Ref, 2), got {other:?}"),
                }
                match &args[3] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(stub.arg_kinds, vec![Kind::Ref, Kind::Ref, Kind::Int]);
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn lower_binary_op_hlop_returns_none_for_non_family_opname() {
        // Out-of-family HLOp (`bool`, `lt`, ...): caller must fall
        // through to other lowering arms.  The helper returns None
        // without inspecting result/args so the caller can recover
        // cheaply.
        let v = Variable::new(VariableId(0), Kind::Ref);
        let r = Variable::new(VariableId(1), Kind::Int);
        let op = SpaceOperation::new("bool", vec![v.into()], Some(r.into()), 0);
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            compare_op_fn_idx: 0,
            truth_fn_idx: 0,
            store_subscr_fn_idx: 0,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        assert!(
            lower_binary_op_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
                .is_none()
        );
    }

    #[test]
    fn lower_binary_op_hlop_matches_dual_write_residual_call_shape() {
        // Byte-equivalence cross-check: the Insn produced from the
        // BINARY_OP HLOp via `lower_binary_op_hlop_to_insn` must match
        // the Insn produced by feeding the equivalent dual-write
        // `residual_call_ir_r` SpaceOperation through
        // `flatten_op_to_insn`.  This is the foundational invariant
        // for retiring the dual-write + inline emit in micro-slice 3
        // of the epic.
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let ctx = LoweringContext {
            binary_op_fn_idx: 11,
            compare_op_fn_idx: 0,
            truth_fn_idx: 0,
            store_subscr_fn_idx: 0,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };

        let hlop = SpaceOperation::new("sub", vec![lhs.into(), rhs.into()], Some(result.into()), 0);
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let lowered =
            lower_binary_op_hlop_to_insn(&hlop, &ctx, &mut get_register, &mut lower_constant)
                .expect("BINARY_OP HLOp must lower");

        // Build the equivalent dual-write residual_call SpaceOperation
        // by hand — same shape as `record_residual_call_graph_op`
        // produces at codewriter.rs:5366-5377 for `sub`:
        //   `[ConstInt(fn_idx), ListI([ConstInt(op_val)]),
        //     ListR([lhs, rhs]), Descr]`.
        // (`record_residual_call_graph_op` codewriter.rs:1378-1404
        // pushes `args_i` before `args_r` per the upstream
        // `i,r,f` order; the `op_val` is bucketed into args_i because
        // its `Kind` is `Int` per arg_kinds.)
        let descr = intern_call_descr_stub(
            effect_info_for_call_flavor(CallFlavor::MayForce),
            vec![Kind::Ref, Kind::Ref, Kind::Int],
            Some(Kind::Ref),
        );
        let dual_op = SpaceOperation::new(
            "residual_call_ir_r",
            vec![
                Constant::signed(11).into(),
                FlowListOfKind::new(Kind::Int, vec![Constant::signed(1).into()]).into(),
                FlowListOfKind::new(Kind::Ref, vec![lhs.into(), rhs.into()]).into(),
                descr.into(),
            ],
            Some(result.into()),
            0,
        );
        let dual = flatten_op_to_insn(
            &dual_op,
            &mut get_register,
            &mut flatten_constant_operand_for_test,
        )
        .expect("residual_call SpaceOperation must lower");

        // Compare via Debug formatting — Insn does not derive Eq, but
        // the Debug output is structurally faithful for the variants
        // we touch (Op, Operand::*, Register).
        assert_eq!(format!("{lowered:?}"), format!("{dual:?}"));
    }

    #[test]
    fn compare_op_tag_for_opname_covers_runtime_ops_table() {
        // Task #48 micro-slice 4: tag mapping must agree with
        // `pyre_interpreter::runtime_ops::compare_op_tag` for every
        // ComparisonOperator the codewriter encodes.  Six opnames
        // (no `inplace_*` siblings — comparisons are pure).
        for (opname, expected) in [
            ("lt", 0),
            ("le", 1),
            ("gt", 2),
            ("ge", 3),
            ("eq", 4),
            ("ne", 5),
        ] {
            assert_eq!(
                compare_op_tag_for_opname(opname),
                Some(expected),
                "tag mismatch for opname {opname:?}",
            );
        }
        // Out-of-family opnames must return None.
        for opname in ["add", "bool", "neg", "simple_call", "setitem"] {
            assert_eq!(
                compare_op_tag_for_opname(opname),
                None,
                "unexpected tag for non-COMPARE_OP opname {opname:?}",
            );
        }
    }

    #[test]
    fn lower_compare_op_hlop_to_insn_emits_residual_call_ir_r() {
        // Task #48 micro-slice 4: lowering an `lt(lhs, rhs) → result`
        // HLOp must produce the same Insn shape that
        // `emit_residual_call_shape` produces inline at the
        // CompareOp callsite (codewriter.rs:5393-5410):
        // `residual_call_ir_r` with args `[ConstInt(fn_idx),
        // ListI([ConstInt(op_val)]), ListR([lhs, rhs]), Descr]`.
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let op = SpaceOperation::new("lt", vec![lhs.into(), rhs.into()], Some(result.into()), 7);
        let ctx = LoweringContext {
            binary_op_fn_idx: 0,
            compare_op_fn_idx: 13,
            truth_fn_idx: 0,
            store_subscr_fn_idx: 0,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();

        let insn = lower_compare_op_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
            .expect("COMPARE_OP HLOp must lower");

        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 2));
                assert_eq!(args.len(), 4);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 13),
                    other => panic!("expected ConstInt(13), got {other:?}"),
                }
                // args[1] = ListI([ConstInt(op_val)]) — op_val = 0 for `lt`.
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert_eq!(list.content.len(), 1);
                        match &list.content[0] {
                            Operand::ConstInt(v) => assert_eq!(*v, 0, "lt → tag 0"),
                            other => panic!("expected ConstInt(0) in ListI, got {other:?}"),
                        }
                    }
                    other => panic!("expected ListOfKind(Int, 1), got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 2);
                    }
                    other => panic!("expected ListOfKind(Ref, 2), got {other:?}"),
                }
                match &args[3] {
                    Operand::Descr(_) => {}
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn lower_compare_op_hlop_returns_none_for_non_family_opname() {
        let v = Variable::new(VariableId(0), Kind::Ref);
        let r = Variable::new(VariableId(1), Kind::Ref);
        let op = SpaceOperation::new("add", vec![v.into(), v.into()], Some(r.into()), 0);
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            compare_op_fn_idx: 13,
            truth_fn_idx: 0,
            store_subscr_fn_idx: 0,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        assert!(
            lower_compare_op_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
                .is_none()
        );
    }

    #[test]
    fn build_compare_op_residual_call_ir_r_insn_matches_lower_helper() {
        // Byte-equivalence cross-check: the production helper
        // `build_compare_op_residual_call_ir_r_insn` (reg-index API)
        // produces the same Insn as
        // `lower_compare_op_hlop_to_insn` (SpaceOperation API) when
        // fed the corresponding pre-rtype HLOp.
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let ctx = LoweringContext {
            binary_op_fn_idx: 0,
            compare_op_fn_idx: 17,
            truth_fn_idx: 0,
            store_subscr_fn_idx: 0,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let hlop = SpaceOperation::new("eq", vec![lhs.into(), rhs.into()], Some(result.into()), 0);
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let lowered =
            lower_compare_op_hlop_to_insn(&hlop, &ctx, &mut get_register, &mut lower_constant)
                .expect("COMPARE_OP HLOp must lower");
        let prod = build_compare_op_residual_call_ir_r_insn(17, 4, 0, 1, 2);
        assert_eq!(format!("{lowered:?}"), format!("{prod:?}"));
    }

    #[test]
    fn lower_bool_hlop_to_insn_emits_residual_call_r_i() {
        // Task #48 micro-slice 5: lowering a `bool(operand) →
        // result` HLOp must produce the same Insn shape that
        // `emit_residual_call_shape` produces inline at the
        // PopJumpIfFalse / PopJumpIfTrue callsites
        // (codewriter.rs:5453-5463 / :5509-5519): `residual_call_r_i`
        // with args `[ConstInt(fn_idx), ListR([cond]), Descr]` and a
        // Register(Int) result.  No `ListI` — `truth_fn` has no
        // scalar Int arg, so `args_i` is empty in
        // `emit_residual_call_shape` and the `ListI` push branch
        // doesn't fire.
        let cond = Variable::new(VariableId(0), Kind::Ref);
        let result = Variable::new(VariableId(1), Kind::Int);
        let op = SpaceOperation::new("bool", vec![cond.into()], Some(result.into()), 5);
        let ctx = LoweringContext {
            binary_op_fn_idx: 0,
            compare_op_fn_idx: 0,
            truth_fn_idx: 23,
            store_subscr_fn_idx: 0,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();

        let insn = lower_bool_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
            .expect("BOOL HLOp must lower");

        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_r_i");
                assert_eq!(reg, Register::new(Kind::Int, 1));
                assert_eq!(args.len(), 3);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 23),
                    other => panic!("expected ConstInt(23), got {other:?}"),
                }
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 1);
                        assert!(matches!(
                            &list.content[0],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 0
                            })
                        ));
                    }
                    other => panic!("expected ListOfKind(Ref, 1), got {other:?}"),
                }
                match &args[2] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(stub.arg_kinds, vec![Kind::Ref]);
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn lower_bool_hlop_returns_none_for_non_bool_opname() {
        let cond = Variable::new(VariableId(0), Kind::Ref);
        let result = Variable::new(VariableId(1), Kind::Int);
        let op = SpaceOperation::new("neg", vec![cond.into()], Some(result.into()), 0);
        let ctx = LoweringContext {
            binary_op_fn_idx: 0,
            compare_op_fn_idx: 0,
            truth_fn_idx: 23,
            store_subscr_fn_idx: 0,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        assert!(
            lower_bool_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant).is_none()
        );
    }

    #[test]
    fn build_truth_fn_residual_call_r_i_insn_matches_lower_helper() {
        // Byte-equivalence cross-check: the production helper
        // (reg-index API) produces the same Insn as the probe-side
        // helper (SpaceOperation API) when fed the corresponding
        // `bool` HLOp.
        let cond = Variable::new(VariableId(0), Kind::Ref);
        let result = Variable::new(VariableId(1), Kind::Int);
        let ctx = LoweringContext {
            binary_op_fn_idx: 0,
            compare_op_fn_idx: 0,
            truth_fn_idx: 31,
            store_subscr_fn_idx: 0,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let hlop = SpaceOperation::new("bool", vec![cond.into()], Some(result.into()), 0);
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let lowered = lower_bool_hlop_to_insn(&hlop, &ctx, &mut get_register, &mut lower_constant)
            .expect("BOOL HLOp must lower");
        let prod = build_truth_fn_residual_call_r_i_insn(31, 0, 1);
        assert_eq!(format!("{lowered:?}"), format!("{prod:?}"));
    }

    #[test]
    fn lower_setitem_hlop_to_insn_emits_residual_call_r_v() {
        // Task #48 micro-slice 6: lowering a `setitem(obj, key,
        // value)` HLOp (no result — `emit_frontend_setitem` records
        // the SpaceOperation with `result = None` per
        // codewriter.rs:1518-1524) must produce the same Insn shape
        // that `emit_residual_call_shape` produces inline at the
        // StoreSubscr callsite (codewriter.rs:5244-5263):
        // `residual_call_r_v` with args `[ConstInt(fn_idx),
        // ListR([obj, key, value]), Descr]` and **no** result
        // Register.
        let obj = Variable::new(VariableId(0), Kind::Ref);
        let key = Variable::new(VariableId(1), Kind::Ref);
        let value = Variable::new(VariableId(2), Kind::Ref);
        let op = SpaceOperation::new(
            "setitem",
            vec![obj.into(), key.into(), value.into()],
            None,
            11,
        );
        let ctx = LoweringContext {
            binary_op_fn_idx: 0,
            compare_op_fn_idx: 0,
            truth_fn_idx: 0,
            store_subscr_fn_idx: 41,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();

        let insn = lower_setitem_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant)
            .expect("SETITEM HLOp must lower");

        match insn {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "residual_call_r_v");
                assert!(result.is_none(), "void Insn must have no result");
                assert_eq!(args.len(), 3);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 41),
                    other => panic!("expected ConstInt(41), got {other:?}"),
                }
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 3);
                    }
                    other => panic!("expected ListOfKind(Ref, 3), got {other:?}"),
                }
                match &args[2] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(stub.arg_kinds, vec![Kind::Ref, Kind::Ref, Kind::Ref]);
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn lower_setitem_hlop_returns_none_for_non_family() {
        let v = Variable::new(VariableId(0), Kind::Ref);
        // Wrong opname.
        let op = SpaceOperation::new("getitem", vec![v.into(), v.into(), v.into()], None, 0);
        let ctx = LoweringContext {
            binary_op_fn_idx: 0,
            compare_op_fn_idx: 0,
            truth_fn_idx: 0,
            store_subscr_fn_idx: 41,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        assert!(
            lower_setitem_hlop_to_insn(&op, &ctx, &mut get_register, &mut lower_constant).is_none()
        );
        // Right opname but with a result — must still return None
        // since `emit_frontend_setitem` always emits void.
        let r = Variable::new(VariableId(3), Kind::Ref);
        let op_with_result = SpaceOperation::new(
            "setitem",
            vec![v.into(), v.into(), v.into()],
            Some(r.into()),
            0,
        );
        assert!(
            lower_setitem_hlop_to_insn(
                &op_with_result,
                &ctx,
                &mut get_register,
                &mut lower_constant
            )
            .is_none()
        );
    }

    #[test]
    fn build_store_subscr_fn_residual_call_r_v_insn_matches_lower_helper() {
        let obj = Variable::new(VariableId(0), Kind::Ref);
        let key = Variable::new(VariableId(1), Kind::Ref);
        let value = Variable::new(VariableId(2), Kind::Ref);
        let ctx = LoweringContext {
            binary_op_fn_idx: 0,
            compare_op_fn_idx: 0,
            truth_fn_idx: 0,
            store_subscr_fn_idx: 53,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let hlop = SpaceOperation::new(
            "setitem",
            vec![obj.into(), key.into(), value.into()],
            None,
            0,
        );
        let mut get_register = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let lowered =
            lower_setitem_hlop_to_insn(&hlop, &ctx, &mut get_register, &mut lower_constant)
                .expect("SETITEM HLOp must lower");
        let prod = build_store_subscr_fn_residual_call_r_v_insn(53, 0, 1, 2);
        assert_eq!(format!("{lowered:?}"), format!("{prod:?}"));
    }

    #[test]
    fn flatten_op_to_insn_with_lowering_dispatches_binary_op_hlop() {
        // the unified dispatcher must route a `add`
        // BINARY_OP HLOp through `lower_binary_op_hlop_to_insn` and
        // produce the same Insn shape the per-family helper produces
        // on its own.
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let op = SpaceOperation::new("add", vec![lhs.into(), rhs.into()], Some(result.into()), 0);
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            compare_op_fn_idx: 19,
            truth_fn_idx: 31,
            store_subscr_fn_idx: 53,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let mut get_register_a = identity_register_mapper();
        let mut get_register_b = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let dispatched =
            flatten_op_to_insn_with_lowering(&op, &ctx, &mut get_register_a, &mut lower_constant)
                .expect("BINARY_OP HLOp must lower via dispatcher");
        let direct =
            lower_binary_op_hlop_to_insn(&op, &ctx, &mut get_register_b, &mut lower_constant)
                .expect("BINARY_OP HLOp must lower directly");
        assert_eq!(format!("{dispatched:?}"), format!("{direct:?}"));
    }

    #[test]
    fn flatten_op_to_insn_with_lowering_dispatches_compare_op_hlop() {
        // dispatcher routes `lt` through
        // `lower_compare_op_hlop_to_insn` even when binary_op /
        // truth / store_subscr fn indices are also set on the ctx.
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let op = SpaceOperation::new("lt", vec![lhs.into(), rhs.into()], Some(result.into()), 0);
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            compare_op_fn_idx: 19,
            truth_fn_idx: 31,
            store_subscr_fn_idx: 53,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let mut get_register_a = identity_register_mapper();
        let mut get_register_b = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let dispatched =
            flatten_op_to_insn_with_lowering(&op, &ctx, &mut get_register_a, &mut lower_constant)
                .expect("COMPARE_OP HLOp must lower via dispatcher");
        let direct =
            lower_compare_op_hlop_to_insn(&op, &ctx, &mut get_register_b, &mut lower_constant)
                .expect("COMPARE_OP HLOp must lower directly");
        assert_eq!(format!("{dispatched:?}"), format!("{direct:?}"));
    }

    #[test]
    fn flatten_op_to_insn_with_lowering_dispatches_bool_hlop() {
        // dispatcher routes `bool(v) → r` through
        // `lower_bool_hlop_to_insn`.  Different residual_call shape
        // (`_r_i` vs `_ir_r`) and different result Kind (Int vs Ref)
        // from BINARY_OP/COMPARE_OP, so this is non-trivial coverage.
        let v = Variable::new(VariableId(0), Kind::Ref);
        let r = Variable::new(VariableId(1), Kind::Int);
        let op = SpaceOperation::new("bool", vec![v.into()], Some(r.into()), 0);
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            compare_op_fn_idx: 19,
            truth_fn_idx: 31,
            store_subscr_fn_idx: 53,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let mut get_register_a = identity_register_mapper();
        let mut get_register_b = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let dispatched =
            flatten_op_to_insn_with_lowering(&op, &ctx, &mut get_register_a, &mut lower_constant)
                .expect("BOOL HLOp must lower via dispatcher");
        let direct = lower_bool_hlop_to_insn(&op, &ctx, &mut get_register_b, &mut lower_constant)
            .expect("BOOL HLOp must lower directly");
        assert_eq!(format!("{dispatched:?}"), format!("{direct:?}"));
    }

    #[test]
    fn flatten_op_to_insn_with_lowering_dispatches_setitem_hlop() {
        // dispatcher routes void-result `setitem(obj,
        // key, value)` through `lower_setitem_hlop_to_insn`.  The
        // void-result arm exercises the dispatcher's no-result path,
        // distinct from the value-producing arms above.
        let obj = Variable::new(VariableId(0), Kind::Ref);
        let key = Variable::new(VariableId(1), Kind::Ref);
        let value = Variable::new(VariableId(2), Kind::Ref);
        let op = SpaceOperation::new(
            "setitem",
            vec![obj.into(), key.into(), value.into()],
            None,
            0,
        );
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            compare_op_fn_idx: 19,
            truth_fn_idx: 31,
            store_subscr_fn_idx: 53,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let mut get_register_a = identity_register_mapper();
        let mut get_register_b = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let dispatched =
            flatten_op_to_insn_with_lowering(&op, &ctx, &mut get_register_a, &mut lower_constant)
                .expect("SETITEM HLOp must lower via dispatcher");
        let direct =
            lower_setitem_hlop_to_insn(&op, &ctx, &mut get_register_b, &mut lower_constant)
                .expect("SETITEM HLOp must lower directly");
        assert_eq!(format!("{dispatched:?}"), format!("{direct:?}"));
    }

    #[test]
    fn flatten_op_to_insn_with_lowering_falls_through_for_residual_call_op() {
        // for opnames outside the four retired families
        // (e.g. an already-lowered `residual_call_ir_r` SpaceOperation
        // as recorded by the factor-refactored families' graph
        // dual-write at codewriter.rs::record_residual_call_graph_op),
        // the dispatcher must fall through to the passthrough
        // `flatten_op_to_insn` and produce the same Insn the
        // passthrough produces directly.
        let lhs = Variable::new(VariableId(0), Kind::Ref);
        let rhs = Variable::new(VariableId(1), Kind::Ref);
        let result = Variable::new(VariableId(2), Kind::Ref);
        let descr = intern_call_descr_stub(
            effect_info_for_call_flavor(CallFlavor::MayForce),
            vec![Kind::Ref, Kind::Ref, Kind::Int],
            Some(Kind::Ref),
        );
        let op = SpaceOperation::new(
            "residual_call_ir_r",
            vec![
                Constant::signed(7).into(),
                FlowListOfKind::new(Kind::Int, vec![Constant::signed(0).into()]).into(),
                FlowListOfKind::new(Kind::Ref, vec![lhs.into(), rhs.into()]).into(),
                descr.into(),
            ],
            Some(result.into()),
            0,
        );
        let ctx = LoweringContext {
            binary_op_fn_idx: 7,
            compare_op_fn_idx: 19,
            truth_fn_idx: 31,
            store_subscr_fn_idx: 53,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        let mut get_register_a = identity_register_mapper();
        let mut get_register_b = identity_register_mapper();
        let mut lower_constant = test_constant_lowering();
        let dispatched =
            flatten_op_to_insn_with_lowering(&op, &ctx, &mut get_register_a, &mut lower_constant)
                .expect("residual_call SpaceOperation must lower via dispatcher");
        let direct = flatten_op_to_insn(
            &op,
            &mut get_register_b,
            &mut flatten_constant_operand_for_test,
        )
        .expect("residual_call SpaceOperation must lower via passthrough");
        assert_eq!(format!("{dispatched:?}"), format!("{direct:?}"));
    }

    #[test]
    fn build_load_const_fn_residual_call_ir_r_insn_emits_residual_call_ir_r() {
        // Task #48 micro-slice 7: LoadConst factor refactor.  The
        // helper must produce the same `residual_call_ir_r` Insn shape
        // that `emit_residual_call_shape` produced inline at
        // codewriter.rs:4933-4946 before the refactor: `[ConstInt(
        // fn_idx), ListI([ConstInt(idx)]), ListR([Reg(pycode)]),
        // Descr(CallDescrStub{Plain, [Ref, Int]})] → Reg(Ref, dst)`.
        // Distinct from BINARY_OP/COMPARE_OP `_ir_r` shape: 1-element
        // ListR (vs 2), Plain flavor (vs MayForce), `arg_kinds = [Ref,
        // Int]` (vs `[Ref, Ref, Int]`).
        let insn = build_load_const_fn_residual_call_ir_r_insn(
            /* load_const_fn_idx */ 9, /* idx */ 17, /* pycode_reg */ 4,
            /* dst_reg */ 5,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 5));
                assert_eq!(args.len(), 4);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 9),
                    other => panic!("expected ConstInt(9), got {other:?}"),
                }
                // args[1] = ListI([ConstInt(idx)]).
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert_eq!(list.content.len(), 1);
                        match &list.content[0] {
                            Operand::ConstInt(v) => assert_eq!(*v, 17),
                            other => panic!("expected ConstInt(17) in ListI, got {other:?}"),
                        }
                    }
                    other => panic!("expected ListOfKind(Int, 1), got {other:?}"),
                }
                // args[2] = ListR([Reg(Ref, pycode_reg)]).
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 1);
                        assert!(matches!(
                            &list.content[0],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 4
                            })
                        ));
                    }
                    other => panic!("expected ListOfKind(Ref, 1), got {other:?}"),
                }
                match &args[3] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(stub.arg_kinds, vec![Kind::Ref, Kind::Int]);
                            assert_eq!(
                                stub.effect_info,
                                effect_info_for_call_flavor(CallFlavor::Plain),
                            );
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_load_const_fn_residual_call_ir_r_insn_matches_flatten_of_residual_call_op() {
        // Byte-equivalence cross-check: the Insn produced by the
        // production helper must match the Insn produced by feeding
        // the equivalent `residual_call_ir_r` SpaceOperation through
        // `flatten_op_to_insn`.  This guarantees the factor refactor
        // at codewriter.rs:4933-4946 produces the same SSARepr bytes
        // `emit_residual_call_shape` would have produced before the
        // refactor — no behavior change, only a more direct emit
        // path.
        //
        // Shape construction mirrors what the (now-removed) inline
        // `emit_residual_call_shape` would have produced for call-args
        // `[Reg(Ref, pycode), ConstInt(idx)]`: bucketed to `args_i =
        // [ConstInt(idx)]`, `args_r = [Reg(pycode)]` per the upstream
        // `i,r,f` order.
        let pycode_var = Variable::new(VariableId(4), Kind::Ref);
        let dst_var = Variable::new(VariableId(5), Kind::Ref);
        let descr = intern_call_descr_stub(
            effect_info_for_call_flavor(CallFlavor::Plain),
            vec![Kind::Ref, Kind::Int],
            Some(Kind::Ref),
        );
        let dual_op = SpaceOperation::new(
            "residual_call_ir_r",
            vec![
                Constant::signed(9).into(),
                FlowListOfKind::new(Kind::Int, vec![Constant::signed(17).into()]).into(),
                FlowListOfKind::new(Kind::Ref, vec![pycode_var.into()]).into(),
                descr.into(),
            ],
            Some(dst_var.into()),
            0,
        );
        let mut get_register = identity_register_mapper();
        let dual = flatten_op_to_insn(
            &dual_op,
            &mut get_register,
            &mut flatten_constant_operand_for_test,
        )
        .expect("residual_call SpaceOperation must lower");
        let prod = build_load_const_fn_residual_call_ir_r_insn(9, 17, 4, 5);
        assert_eq!(format!("{prod:?}"), format!("{dual:?}"));
    }

    #[test]
    fn build_load_global_fn_residual_call_ir_r_insn_emits_residual_call_ir_r() {
        // Task #48 micro-slice 8: LoadGlobal factor refactor.  The
        // helper must produce the same `residual_call_ir_r` Insn shape
        // that `emit_residual_call_shape` produced inline at
        // codewriter.rs:5598-5615 before the refactor: `[ConstInt(
        // fn_idx), ListI([ConstInt(namei)]), ListR([Reg(ns), Reg(
        // code), Reg(frame)]), Descr(CallDescrStub{Plain,
        // [Ref, Ref, Ref, Int]})] → Reg(Ref, dst)`.  The explicit
        // frame Ref is the helper-level stand-in for PyPy's
        // `_load_global(self, ...)` receiver.
        let insn = build_load_global_fn_residual_call_ir_r_insn(
            /* load_global_fn_idx */ 12, /* namei */ 5, /* ns_reg */ 3,
            /* code_reg */ 4, /* frame_reg */ 6, /* dst_reg */ 7,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 7));
                assert_eq!(args.len(), 4);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 12),
                    other => panic!("expected ConstInt(12), got {other:?}"),
                }
                // args[1] = ListI([ConstInt(namei)]).
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert_eq!(list.content.len(), 1);
                        match &list.content[0] {
                            Operand::ConstInt(v) => assert_eq!(*v, 5),
                            other => panic!("expected ConstInt(5) in ListI, got {other:?}"),
                        }
                    }
                    other => panic!("expected ListOfKind(Int, 1), got {other:?}"),
                }
                // args[2] = ListR([Reg(Ref, ns_reg), Reg(Ref, code_reg), Reg(Ref, frame_reg)]).
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert_eq!(list.content.len(), 3);
                        assert!(matches!(
                            &list.content[0],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 3
                            })
                        ));
                        assert!(matches!(
                            &list.content[1],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 4
                            })
                        ));
                        assert!(matches!(
                            &list.content[2],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 6
                            })
                        ));
                    }
                    other => panic!("expected ListOfKind(Ref, 3), got {other:?}"),
                }
                match &args[3] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(
                                stub.arg_kinds,
                                vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Int]
                            );
                            assert_eq!(
                                stub.effect_info,
                                effect_info_for_call_flavor(CallFlavor::Plain),
                            );
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_load_global_fn_residual_call_ir_r_insn_matches_flatten_of_residual_call_op() {
        // Byte-equivalence cross-check: the Insn produced by the
        // production helper must match the Insn produced by feeding
        // the equivalent `residual_call_ir_r` SpaceOperation through
        // `flatten_op_to_insn`.  This guarantees the factor refactor
        // at codewriter.rs:5598-5615 produces the same SSARepr bytes
        // `emit_residual_call_shape` would have produced before the
        // refactor — no behavior change, only a more direct emit
        // path.
        let ns_var = Variable::new(VariableId(3), Kind::Ref);
        let code_var = Variable::new(VariableId(4), Kind::Ref);
        let frame_var = Variable::new(VariableId(6), Kind::Ref);
        let dst_var = Variable::new(VariableId(7), Kind::Ref);
        let descr = intern_call_descr_stub(
            effect_info_for_call_flavor(CallFlavor::Plain),
            vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Int],
            Some(Kind::Ref),
        );
        let dual_op = SpaceOperation::new(
            "residual_call_ir_r",
            vec![
                Constant::signed(12).into(),
                FlowListOfKind::new(Kind::Int, vec![Constant::signed(5).into()]).into(),
                FlowListOfKind::new(
                    Kind::Ref,
                    vec![ns_var.into(), code_var.into(), frame_var.into()],
                )
                .into(),
                descr.into(),
            ],
            Some(dst_var.into()),
            0,
        );
        let mut get_register = identity_register_mapper();
        let dual = flatten_op_to_insn(
            &dual_op,
            &mut get_register,
            &mut flatten_constant_operand_for_test,
        )
        .expect("residual_call SpaceOperation must lower");
        let prod = build_load_global_fn_residual_call_ir_r_insn(12, 5, 3, 4, 6, 7);
        assert_eq!(format!("{prod:?}"), format!("{dual:?}"));
    }

    #[test]
    fn build_load_global_helper_carries_distinct_effect_info_from_binary_op_helper() {
        // BINARY_OP records `MayForce` → ForcesVirtualOrVirtualizable;
        // LoadGlobal records `Plain` → CanRaise. PyPy `call.py:288-303
        // getcalldescr` keeps these as distinct `extraeffect` values:
        // the `virtualizable_analyzer` branch (`:288-289`) routes to
        // `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`, the `_canraise`
        // raising-callee branch (`:300-301`) routes to `EF_CAN_RAISE`.
        // `effectinfo.py:285` force-promotion to `EF_RANDOM_EFFECTS`
        // only fires when the analyzer literally returns `top_set`; the
        // analyzer-absent external-call default is `bottom_result()` =
        // empty set per `graphanalyze.py:60`, so the supplied
        // `extraeffect` survives the empty-effects else-branch
        // (`:293-299`). Both EIs therefore carry distinct shapes and
        // round-trip through `dispatch_kind_for_effect_info`.
        let bin = build_binary_op_residual_call_ir_r_insn(7, 0, 1, 2, 3);
        let glob = build_load_global_fn_residual_call_ir_r_insn(7, 0, 1, 2, 4, 3);
        let bin_descr = match &bin {
            Insn::Op { args, .. } => match &args[3] {
                Operand::Descr(rc) => match &**rc {
                    DescrOperand::CallDescrStub(stub) => stub.effect_info.clone(),
                    _ => panic!("BINARY_OP descr is not CallDescrStub"),
                },
                _ => panic!("BINARY_OP args[3] is not Descr"),
            },
            _ => panic!("BINARY_OP Insn is not Op"),
        };
        let glob_descr = match &glob {
            Insn::Op { args, .. } => match &args[3] {
                Operand::Descr(rc) => match &**rc {
                    DescrOperand::CallDescrStub(stub) => stub.effect_info.clone(),
                    _ => panic!("LoadGlobal descr is not CallDescrStub"),
                },
                _ => panic!("LoadGlobal args[3] is not Descr"),
            },
            _ => panic!("LoadGlobal Insn is not Op"),
        };
        assert_eq!(
            bin_descr.extraeffect,
            majit_ir::ExtraEffect::ForcesVirtualOrVirtualizable,
            "MayForce flavor produces EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE \
             per `call.py:288-289 virtualizable_analyzer` branch"
        );
        assert_eq!(
            glob_descr.extraeffect,
            majit_ir::ExtraEffect::CanRaise,
            "Plain flavor produces EF_CAN_RAISE per `call.py:300-301 \
             _canraise` branch"
        );
        assert_ne!(
            bin_descr, glob_descr,
            "MayForce and Plain carry distinct EffectInfo shapes; \
             collapsing them onto MOST_GENERAL would over-claim \
             random-effects semantics and trip `has_random_effects` \
             cache invalidation (`effectinfo.py:252`)"
        );
    }

    #[test]
    fn build_call_fn_residual_call_r_r_insn_emits_residual_call_r_r_for_nargs_2() {
        // Task #48 micro-slice 9: CALL family factor refactor.  The
        // helper must produce the same `residual_call_r_r` Insn shape
        // that `emit_residual_call_shape` produced inline at
        // codewriter.rs:5747-5754 before the refactor.  For nargs=2:
        // `[ConstInt(fn_idx), ListR([Reg(callable), Reg(arg0),
        // Reg(arg1)]), Descr(CallDescrStub{MayForce, [Ref, Ref, Ref]
        // })] → Reg(Ref, dst)`.  No leading `ListI` — `args_i` is
        // empty for all-Ref call_args.
        let insn = build_call_fn_residual_call_r_r_insn(
            /* call_fn_idx */ 21,
            /* callable_reg */ 5,
            /* arg_regs */ &[6, 7],
            /* dst_reg */ 8,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert_eq!(reg, Register::new(Kind::Ref, 8));
                // 3 args: ConstInt(fn_idx), ListR(refs), Descr.
                // No `ListI` because `args_i` is empty.
                assert_eq!(args.len(), 3);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 21),
                    other => panic!("expected ConstInt(21), got {other:?}"),
                }
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        // nargs+1 = 3: callable + 2 args.
                        assert_eq!(list.content.len(), 3);
                        assert!(matches!(
                            &list.content[0],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 5
                            })
                        ));
                        assert!(matches!(
                            &list.content[1],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 6
                            })
                        ));
                        assert!(matches!(
                            &list.content[2],
                            Operand::Register(Register {
                                kind: Kind::Ref,
                                index: 7
                            })
                        ));
                    }
                    other => panic!("expected ListOfKind(Ref, 3), got {other:?}"),
                }
                match &args[2] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(stub.arg_kinds, vec![Kind::Ref, Kind::Ref, Kind::Ref]);
                            assert_eq!(
                                stub.effect_info,
                                effect_info_for_call_flavor(CallFlavor::MayForce),
                            );
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_call_fn_residual_call_r_r_insn_handles_nargs_0_and_8_boundaries() {
        // Boundary cases: nargs=0 (just callable, ListR len=1,
        // arg_kinds=[Ref]) and nargs=8 (callable + 8 args, ListR
        // len=9, arg_kinds=[Ref;9]).  nargs > 8 falls through to
        // emit_abort_permanent at the codewriter level and never
        // invokes this helper, so 8 is the maximum we test.
        let nargs_0 = build_call_fn_residual_call_r_r_insn(10, 1, &[], 2);
        if let Insn::Op { args, .. } = &nargs_0 {
            if let Operand::ListOfKind(list) = &args[1] {
                assert_eq!(list.content.len(), 1, "nargs=0 → ListR len=1 (callable)");
            } else {
                panic!("expected ListOfKind at args[1]");
            }
            if let Operand::Descr(rc) = &args[2] {
                if let DescrOperand::CallDescrStub(stub) = &**rc {
                    assert_eq!(stub.arg_kinds, vec![Kind::Ref]);
                } else {
                    panic!("expected CallDescrStub");
                }
            } else {
                panic!("expected Descr at args[2]");
            }
        } else {
            panic!("expected Insn::Op");
        }

        let nargs_8 = build_call_fn_residual_call_r_r_insn(11, 1, &[2, 3, 4, 5, 6, 7, 8, 9], 10);
        if let Insn::Op { args, .. } = &nargs_8 {
            if let Operand::ListOfKind(list) = &args[1] {
                assert_eq!(
                    list.content.len(),
                    9,
                    "nargs=8 → ListR len=9 (callable + 8)"
                );
            } else {
                panic!("expected ListOfKind at args[1]");
            }
            if let Operand::Descr(rc) = &args[2] {
                if let DescrOperand::CallDescrStub(stub) = &**rc {
                    assert_eq!(stub.arg_kinds, vec![Kind::Ref; 9]);
                } else {
                    panic!("expected CallDescrStub");
                }
            } else {
                panic!("expected Descr at args[2]");
            }
        } else {
            panic!("expected Insn::Op");
        }
    }

    #[test]
    fn build_call_fn_residual_call_r_r_insn_matches_flatten_of_residual_call_op() {
        // Byte-equivalence cross-check: the Insn produced by the
        // production helper must match the Insn produced by feeding
        // the equivalent `residual_call_r_r` SpaceOperation through
        // `flatten_op_to_insn`.  Tested at nargs=3 (callable + 3
        // args) — the inline `emit_residual_call_shape` produces an
        // Insn with ListR=[callable, arg0, arg1, arg2], no ListI.
        let callable = Variable::new(VariableId(5), Kind::Ref);
        let arg0 = Variable::new(VariableId(6), Kind::Ref);
        let arg1 = Variable::new(VariableId(7), Kind::Ref);
        let arg2 = Variable::new(VariableId(8), Kind::Ref);
        let dst = Variable::new(VariableId(9), Kind::Ref);
        let descr = intern_call_descr_stub(
            effect_info_for_call_flavor(CallFlavor::MayForce),
            vec![Kind::Ref, Kind::Ref, Kind::Ref, Kind::Ref],
            Some(Kind::Ref),
        );
        let dual_op = SpaceOperation::new(
            "residual_call_r_r",
            vec![
                Constant::signed(33).into(),
                FlowListOfKind::new(
                    Kind::Ref,
                    vec![callable.into(), arg0.into(), arg1.into(), arg2.into()],
                )
                .into(),
                descr.into(),
            ],
            Some(dst.into()),
            0,
        );
        let mut get_register = identity_register_mapper();
        let dual = flatten_op_to_insn(
            &dual_op,
            &mut get_register,
            &mut flatten_constant_operand_for_test,
        )
        .expect("residual_call SpaceOperation must lower");
        let prod = build_call_fn_residual_call_r_r_insn(33, 5, &[6, 7, 8], 9);
        assert_eq!(format!("{prod:?}"), format!("{dual:?}"));
    }

    #[test]
    fn build_box_int_fn_residual_call_ir_r_insn_emits_residual_call_ir_r() {
        // Task #48 micro-slice 10: box_int_fn factor refactor.  The
        // helper must produce the same `residual_call_ir_r` Insn
        // shape that `emit_residual_call_shape` produced inline at
        // all 3 box_int_fn callsites (LoadSmallInt / UnaryNegative /
        // lasti): `[ConstInt(fn_idx), ListI([ConstInt(val)]),
        // ListR([]), Descr(CallDescrStub{Plain, [Int]})] →
        // Reg(Ref, dst)`.  Empty `ListR` is required by RPython
        // jtransform.py:425 (`elif lst_i or force_ir: kinds = 'ir'`)
        // and jtransform.py:430 (`if 'r' in kinds: sublists.append(
        // lst_r)`).
        let insn = build_box_int_fn_residual_call_ir_r_insn(
            /* box_int_fn_idx */ 4, /* val */ 42, /* dst_reg */ 7,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 7));
                // 4 args: ConstInt(fn_idx), ListI(consts), ListR(empty),
                // Descr.
                assert_eq!(args.len(), 4);
                match &args[0] {
                    Operand::ConstInt(v) => assert_eq!(*v, 4),
                    other => panic!("expected ConstInt(4), got {other:?}"),
                }
                match &args[1] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Int);
                        assert_eq!(list.content.len(), 1);
                        match &list.content[0] {
                            Operand::ConstInt(v) => assert_eq!(*v, 42),
                            other => panic!("expected ConstInt(42), got {other:?}"),
                        }
                    }
                    other => panic!("expected ListOfKind(Int, 1), got {other:?}"),
                }
                match &args[2] {
                    Operand::ListOfKind(list) => {
                        assert_eq!(list.kind, Kind::Ref);
                        assert!(list.content.is_empty(), "expected empty ListR");
                    }
                    other => panic!("expected ListOfKind(Ref, 0), got {other:?}"),
                }
                match &args[3] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(stub.arg_kinds, vec![Kind::Int]);
                            assert_eq!(
                                stub.effect_info,
                                effect_info_for_call_flavor(CallFlavor::Plain),
                            );
                        }
                        other => panic!("expected CallDescrStub, got {other:?}"),
                    },
                    other => panic!("expected Operand::Descr, got {other:?}"),
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_box_int_fn_residual_call_ir_r_insn_handles_zero_and_negative_vals() {
        // The 3 production callsites pass different `val`s (literal
        // small int, 0 from UnaryNegative, lasti_py_pc which can be
        // negative-cast for sentinel values).  Verify the helper
        // emits structurally identical Insns aside from the
        // ConstInt(val) carried in ListI.
        let zero = build_box_int_fn_residual_call_ir_r_insn(4, 0, 1);
        let neg = build_box_int_fn_residual_call_ir_r_insn(4, -1, 1);
        let pos = build_box_int_fn_residual_call_ir_r_insn(4, 9999, 1);
        for (insn, expected_val) in [(zero, 0i64), (neg, -1), (pos, 9999)] {
            if let Insn::Op { args, .. } = &insn {
                if let Operand::ListOfKind(list) = &args[1] {
                    if let Operand::ConstInt(v) = &list.content[0] {
                        assert_eq!(*v, expected_val);
                    } else {
                        panic!("expected ConstInt at ListI[0]");
                    }
                } else {
                    panic!("expected ListOfKind at args[1]");
                }
            } else {
                panic!("expected Insn::Op");
            }
        }
    }

    #[test]
    fn build_box_int_fn_residual_call_ir_r_insn_matches_flatten_of_residual_call_op() {
        // Byte-equivalence cross-check: the Insn produced by the
        // production helper must match the Insn produced by feeding
        // the equivalent `residual_call_ir_r` SpaceOperation through
        // `flatten_op_to_insn`.  This guarantees the factor refactor
        // produces the same SSARepr bytes `emit_residual_call_shape`
        // would have produced before the refactor.
        let dst = Variable::new(VariableId(7), Kind::Ref);
        let descr = intern_call_descr_stub(
            effect_info_for_call_flavor(CallFlavor::Plain),
            vec![Kind::Int],
            Some(Kind::Ref),
        );
        let dual_op = SpaceOperation::new(
            "residual_call_ir_r",
            vec![
                Constant::signed(4).into(),
                FlowListOfKind::new(Kind::Int, vec![Constant::signed(42).into()]).into(),
                FlowListOfKind::new(Kind::Ref, vec![]).into(),
                descr.into(),
            ],
            Some(dst.into()),
            0,
        );
        let mut get_register = identity_register_mapper();
        let dual = flatten_op_to_insn(
            &dual_op,
            &mut get_register,
            &mut flatten_constant_operand_for_test,
        )
        .expect("residual_call SpaceOperation must lower");
        let prod = build_box_int_fn_residual_call_ir_r_insn(4, 42, 7);
        assert_eq!(format!("{prod:?}"), format!("{dual:?}"));
    }

    #[test]
    fn build_build_list_fn_residual_call_ir_r_insn_emits_residual_call_ir_r_for_argc_2() {
        // Task #48 micro-slice 13: BuildList factor refactor.  argc=2:
        // two item slots are real Refs, and the third trailing ABI slot
        // is an Int dummy.  Expected shape:
        // `[ConstInt(fn_idx), ListI([ConstInt(2), ConstInt(0)]),
        // ListR([Reg(item0), Reg(item1)]),
        // Descr(CallDescrStub{Plain, [Int, Ref, Ref, Int]})] → Reg(Ref, dst)`.
        let insn = build_build_list_fn_residual_call_ir_r_insn(
            /* build_list_fn_idx */ 18,
            /* argc */ 2,
            /* arg_regs */ &[3, 4],
            /* dst_reg */ 5,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 5));
                assert_eq!(args.len(), 4);
                if let Operand::ConstInt(v) = &args[0] {
                    assert_eq!(*v, 18);
                } else {
                    panic!("expected ConstInt(18) at args[0]");
                }
                if let Operand::ListOfKind(list) = &args[1] {
                    assert_eq!(list.kind, Kind::Int);
                    assert_eq!(list.content.len(), 2, "argc=2 → ListI=[2, 0]");
                    if let Operand::ConstInt(v) = &list.content[0] {
                        assert_eq!(*v, 2);
                    } else {
                        panic!("expected ConstInt(2) in ListI");
                    }
                    if let Operand::ConstInt(v) = &list.content[1] {
                        assert_eq!(*v, 0);
                    } else {
                        panic!("expected trailing ConstInt(0) in ListI");
                    }
                } else {
                    panic!("expected ListOfKind(Int) at args[1]");
                }
                if let Operand::ListOfKind(list) = &args[2] {
                    assert_eq!(list.kind, Kind::Ref);
                    assert_eq!(list.content.len(), 2);
                } else {
                    panic!("expected ListOfKind(Ref) at args[2]");
                }
                if let Operand::Descr(rc) = &args[3] {
                    if let DescrOperand::CallDescrStub(stub) = &**rc {
                        assert_eq!(
                            stub.arg_kinds,
                            vec![Kind::Int, Kind::Ref, Kind::Ref, Kind::Int],
                        );
                        assert_eq!(
                            stub.effect_info,
                            effect_info_for_call_flavor(CallFlavor::Plain),
                        );
                    } else {
                        panic!("expected CallDescrStub");
                    }
                } else {
                    panic!("expected Descr at args[3]");
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_build_list_fn_residual_call_ir_r_insn_pads_argc_0_and_1() {
        // argc=0: no real items.  arg_kinds=[Int, Int, Int, Int], args_i=
        // [argc=0, dummy=0, dummy=0, dummy=0], args_r=[].  ListR is still
        // pushed (empty) because kinds="ir" includes 'r'.
        let argc_0 = build_build_list_fn_residual_call_ir_r_insn(18, 0, &[], 5);
        if let Insn::Op { args, .. } = &argc_0 {
            if let Operand::ListOfKind(list) = &args[1] {
                assert_eq!(list.content.len(), 4, "argc=0 → ListI=[0, 0, 0, 0]");
                for op in &list.content {
                    if let Operand::ConstInt(v) = op {
                        assert_eq!(*v, 0);
                    } else {
                        panic!("expected ConstInt in ListI");
                    }
                }
            } else {
                panic!("expected ListOfKind(Int) at args[1]");
            }
            if let Operand::ListOfKind(list) = &args[2] {
                assert_eq!(list.content.len(), 0, "argc=0 → ListR=[]");
            } else {
                panic!("expected ListOfKind(Ref) at args[2]");
            }
            if let Operand::Descr(rc) = &args[3] {
                if let DescrOperand::CallDescrStub(stub) = &**rc {
                    assert_eq!(
                        stub.arg_kinds,
                        vec![Kind::Int, Kind::Int, Kind::Int, Kind::Int],
                    );
                }
            }
        } else {
            panic!("expected Insn::Op");
        }

        // argc=1: 1 real item, 2 padding slots.  arg_kinds=[Int, Ref, Int, Int],
        // args_i=[argc=1, dummy=0, dummy=0], args_r=[reg].
        let argc_1 = build_build_list_fn_residual_call_ir_r_insn(18, 1, &[7], 5);
        if let Insn::Op { args, .. } = &argc_1 {
            if let Operand::ListOfKind(list) = &args[1] {
                assert_eq!(list.content.len(), 3, "argc=1 → ListI=[1, 0, 0]");
                let expected = [1, 0, 0];
                for (op, expected) in list.content.iter().zip(expected) {
                    if let Operand::ConstInt(v) = op {
                        assert_eq!(*v, expected);
                    } else {
                        panic!("expected ConstInt in ListI");
                    }
                }
            } else {
                panic!("expected ListOfKind(Int) at args[1]");
            }
            if let Operand::ListOfKind(list) = &args[2] {
                assert_eq!(list.content.len(), 1, "argc=1 → ListR=[reg]");
            } else {
                panic!("expected ListOfKind(Ref) at args[2]");
            }
            if let Operand::Descr(rc) = &args[3] {
                if let DescrOperand::CallDescrStub(stub) = &**rc {
                    assert_eq!(
                        stub.arg_kinds,
                        vec![Kind::Int, Kind::Ref, Kind::Int, Kind::Int],
                    );
                }
            }
        } else {
            panic!("expected Insn::Op");
        }

        // argc=3: all trailing ABI slots are Refs.  arg_kinds=
        // [Int, Ref, Ref, Ref], args_i=[argc=3], args_r=[reg, reg, reg].
        let argc_3 = build_build_list_fn_residual_call_ir_r_insn(18, 3, &[7, 8, 9], 5);
        if let Insn::Op { args, .. } = &argc_3 {
            if let Operand::ListOfKind(list) = &args[1] {
                assert_eq!(list.content.len(), 1, "argc=3 → ListI=[3]");
                if let Operand::ConstInt(v) = &list.content[0] {
                    assert_eq!(*v, 3);
                } else {
                    panic!("expected ConstInt(3) in ListI");
                }
            } else {
                panic!("expected ListOfKind(Int) at args[1]");
            }
            if let Operand::ListOfKind(list) = &args[2] {
                assert_eq!(list.content.len(), 3, "argc=3 → ListR=[reg, reg, reg]");
            } else {
                panic!("expected ListOfKind(Ref) at args[2]");
            }
            if let Operand::Descr(rc) = &args[3] {
                if let DescrOperand::CallDescrStub(stub) = &**rc {
                    assert_eq!(
                        stub.arg_kinds,
                        vec![Kind::Int, Kind::Ref, Kind::Ref, Kind::Ref],
                    );
                }
            }
        } else {
            panic!("expected Insn::Op");
        }
    }

    #[test]
    fn build_build_list_fn_residual_call_ir_r_insn_matches_flatten_of_residual_call_op() {
        // Byte-equivalence at argc=2 — feed an equivalent
        // residual_call_ir_r SpaceOperation through
        // `flatten_op_to_insn` and compare.
        let item0 = Variable::new(VariableId(3), Kind::Ref);
        let item1 = Variable::new(VariableId(4), Kind::Ref);
        let dst = Variable::new(VariableId(5), Kind::Ref);
        let descr = intern_call_descr_stub(
            effect_info_for_call_flavor(CallFlavor::Plain),
            vec![Kind::Int, Kind::Ref, Kind::Ref, Kind::Int],
            Some(Kind::Ref),
        );
        let dual_op = SpaceOperation::new(
            "residual_call_ir_r",
            vec![
                Constant::signed(18).into(),
                FlowListOfKind::new(
                    Kind::Int,
                    vec![Constant::signed(2).into(), Constant::signed(0).into()],
                )
                .into(),
                FlowListOfKind::new(Kind::Ref, vec![item0.into(), item1.into()]).into(),
                descr.into(),
            ],
            Some(dst.into()),
            0,
        );
        let mut get_register = identity_register_mapper();
        let dual = flatten_op_to_insn(
            &dual_op,
            &mut get_register,
            &mut flatten_constant_operand_for_test,
        )
        .expect("residual_call SpaceOperation must lower");
        let prod = build_build_list_fn_residual_call_ir_r_insn(18, 2, &[3, 4], 5);
        assert_eq!(format!("{prod:?}"), format!("{dual:?}"));
    }

    #[test]
    #[should_panic(expected = "BuildList helper only supports argc")]
    fn build_build_list_fn_residual_call_ir_r_insn_rejects_unsupported_argc() {
        let _ = build_build_list_fn_residual_call_ir_r_insn(18, 4, &[1, 2, 3, 4], 5);
    }

    #[test]
    #[should_panic(expected = "arg_regs length must match argc")]
    fn build_build_list_fn_residual_call_ir_r_insn_rejects_arg_reg_mismatch() {
        let _ = build_build_list_fn_residual_call_ir_r_insn(18, 2, &[1], 5);
    }

    #[test]
    #[should_panic(expected = "BUILD_SLICE expects argc=2 without step or argc=3 with step")]
    fn build_build_slice_fn_residual_call_ir_r_insn_rejects_two_arg_with_step() {
        let _ = build_build_slice_fn_residual_call_ir_r_insn(19, 2, 1, 2, Some(3), 4);
    }

    #[test]
    #[should_panic(expected = "BUILD_SLICE expects argc=2 without step or argc=3 with step")]
    fn build_build_slice_fn_residual_call_ir_r_insn_rejects_three_arg_without_step() {
        let _ = build_build_slice_fn_residual_call_ir_r_insn(19, 3, 1, 2, None, 4);
    }

    #[test]
    fn build_normalize_raise_varargs_fn_residual_call_r_r_insn_with_reg_cause() {
        // Task #48 micro-slice 14: `(exc:Ref, cause:Ref) → Ref`
        // MayForce.  Argc=2 callsite uses `Operand::Register(Ref,
        // cause_reg)` for the cause arg.
        let cause = Operand::Register(Register::new(Kind::Ref, 4));
        let insn = build_normalize_raise_varargs_fn_residual_call_r_r_insn(
            /* fn_idx */ 25, /* exc_reg */ 3, cause, /* dst_reg */ 3,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert_eq!(reg, Register::new(Kind::Ref, 3));
                assert_eq!(args.len(), 3);
                if let Operand::ConstInt(v) = &args[0] {
                    assert_eq!(*v, 25);
                } else {
                    panic!("expected ConstInt(25)");
                }
                if let Operand::ListOfKind(list) = &args[1] {
                    assert_eq!(list.kind, Kind::Ref);
                    assert_eq!(list.content.len(), 2);
                    assert!(matches!(
                        &list.content[0],
                        Operand::Register(Register {
                            kind: Kind::Ref,
                            index: 3
                        })
                    ));
                    assert!(matches!(
                        &list.content[1],
                        Operand::Register(Register {
                            kind: Kind::Ref,
                            index: 4
                        })
                    ));
                } else {
                    panic!("expected ListOfKind(Ref) at args[1]");
                }
                if let Operand::Descr(rc) = &args[2] {
                    if let DescrOperand::CallDescrStub(stub) = &**rc {
                        assert_eq!(stub.arg_kinds, vec![Kind::Ref, Kind::Ref]);
                        assert_eq!(
                            stub.effect_info,
                            effect_info_for_call_flavor(CallFlavor::MayForce),
                        );
                    } else {
                        panic!("expected CallDescrStub");
                    }
                } else {
                    panic!("expected Descr at args[2]");
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_normalize_raise_varargs_fn_residual_call_r_r_insn_with_const_ref_cause() {
        // argc=1 callsite passes `Operand::ConstRef(PY_NULL)` for the
        // cause arg.  ConstRef has Kind::Ref so the Insn shape stays
        // `residual_call_r_r` with arg_kinds=[Ref, Ref].
        let cause = Operand::ConstRef(0); // PY_NULL stand-in
        let insn = build_normalize_raise_varargs_fn_residual_call_r_r_insn(25, 3, cause, 3);
        if let Insn::Op { args, .. } = &insn {
            if let Operand::ListOfKind(list) = &args[1] {
                assert_eq!(list.content.len(), 2);
                assert!(matches!(
                    &list.content[0],
                    Operand::Register(Register {
                        kind: Kind::Ref,
                        index: 3
                    })
                ));
                assert!(matches!(&list.content[1], Operand::ConstRef(0)));
            } else {
                panic!("expected ListOfKind(Ref) at args[1]");
            }
        } else {
            panic!("expected Insn::Op");
        }
    }

    #[test]
    fn build_get_current_exception_fn_residual_call_r_r_insn_emits_zero_arg_call() {
        // Task #48 micro-slice 15: 0-arg `() → Ref` PlainCannotRaiseNoHeap.
        // Insn shape: `[ConstInt(fn_idx), ListR([]), Descr]
        // → Reg(Ref, dst)`.  arg_kinds is empty; flavor is PlainCannotRaiseNoHeap.
        let insn = build_get_current_exception_fn_residual_call_r_r_insn(
            /* fn_idx */ 30, /* dst_reg */ 5,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_r_r");
                assert_eq!(reg, Register::new(Kind::Ref, 5));
                assert_eq!(args.len(), 3);
                if let Operand::ConstInt(v) = &args[0] {
                    assert_eq!(*v, 30);
                } else {
                    panic!("expected ConstInt(30)");
                }
                if let Operand::ListOfKind(list) = &args[1] {
                    assert_eq!(list.kind, Kind::Ref);
                    assert_eq!(list.content.len(), 0, "0-arg → empty ListR");
                } else {
                    panic!("expected ListOfKind(Ref) at args[1]");
                }
                if let Operand::Descr(rc) = &args[2] {
                    if let DescrOperand::CallDescrStub(stub) = &**rc {
                        assert_eq!(stub.arg_kinds, Vec::<Kind>::new());
                        assert_eq!(
                            stub.effect_info,
                            effect_info_for_call_flavor(CallFlavor::PlainCannotRaiseNoHeap),
                        );
                    } else {
                        panic!("expected CallDescrStub");
                    }
                } else {
                    panic!("expected Descr at args[2]");
                }
            }
            other => panic!("expected Insn::Op, got {other:?}"),
        }
    }

    #[test]
    fn build_set_current_exception_fn_residual_call_r_v_insn_emits_void_call() {
        // 1-arg `(exc:Ref) → Void` PlainCannotRaiseNoHeap.  Insn shape:
        // `[ConstInt(fn_idx), ListR([Reg(exc)]), Descr]` (no result).
        let insn = build_set_current_exception_fn_residual_call_r_v_insn(
            /* fn_idx */ 31, /* exc_reg */ 7,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: None,
            } => {
                assert_eq!(opname, "residual_call_r_v");
                assert_eq!(args.len(), 3);
                if let Operand::ConstInt(v) = &args[0] {
                    assert_eq!(*v, 31);
                } else {
                    panic!("expected ConstInt(31)");
                }
                if let Operand::ListOfKind(list) = &args[1] {
                    assert_eq!(list.kind, Kind::Ref);
                    assert_eq!(list.content.len(), 1);
                    assert!(matches!(
                        &list.content[0],
                        Operand::Register(Register {
                            kind: Kind::Ref,
                            index: 7
                        })
                    ));
                } else {
                    panic!("expected ListOfKind(Ref) at args[1]");
                }
                if let Operand::Descr(rc) = &args[2] {
                    if let DescrOperand::CallDescrStub(stub) = &**rc {
                        assert_eq!(stub.arg_kinds, vec![Kind::Ref]);
                        assert_eq!(
                            stub.effect_info,
                            effect_info_for_call_flavor(CallFlavor::PlainCannotRaiseNoHeap),
                        );
                    } else {
                        panic!("expected CallDescrStub");
                    }
                } else {
                    panic!("expected Descr at args[2]");
                }
            }
            other => panic!("expected Insn::Op (Void), got {other:?}"),
        }
    }

    // ----------------------------------------------------------------
    // Phase 1 — `_ovf` popline rewrite + handling_ovf=true reraise.
    //
    // Tests for `rpython/jit/codewriter/flatten.py:120-204`:
    //   * `make_bytecode_block` `_ovf` validity check (lines 120-125)
    //   * `insert_exits` `_ovf` tail rewrite (lines 191-204) routed
    //     through `flatten_ovf_canraise`.
    //   * `make_exception_link` `handling_ovf=True` arm (lines 165-170)
    //     emitting `raise <OverflowError const>` via the cpu/rtyper/
    //     exceptiondata shim.
    // ----------------------------------------------------------------

    /// Build a canraise startblock containing a single overflow-checked
    /// arithmetic op, then close it with `(normal, ovf [, catch_all])`
    /// exits.  Returns `(graph, ssarepr)` after `flatten_graph` runs;
    /// the test asserts on the recorded `ssarepr.insns` shape.
    fn flatten_ovf_canraise_graph(
        name: &str,
        opname: &str,
        with_catch_all: bool,
        cpu: &super::super::cpu::Cpu,
    ) -> SSARepr {
        use crate::jit::flow::{Block, ExitSwitch, FunctionGraph, Link, c_last_exception};

        let lhs = Variable::new(VariableId(0), Kind::Int);
        let rhs = Variable::new(VariableId(1), Kind::Int);
        let res = Variable::new(VariableId(2), Kind::Int);
        let except_etype = Variable::new(VariableId(3), Kind::Int);
        let except_evalue = Variable::new(VariableId(4), Kind::Ref);

        let start = Block::shared(vec![lhs.into(), rhs.into()]);
        // `make_return` for the normal path expects the returnblock
        // input to be a single Variable; pass `res` as the return slot
        // (will be renamed via the normal link).
        let graph = FunctionGraph::new(name, start.clone(), Some(res));

        super::super::flow::push_op(
            &start,
            SpaceOperation::new(opname, vec![lhs.into(), rhs.into()], Some(res.into()), 42),
        );
        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(c_last_exception().into()));

        let normal_link =
            Link::new(vec![res.into()], Some(graph.returnblock.clone()), None).into_ref();

        let mut ovf_link = Link::new(
            vec![except_etype.into(), except_evalue.into()],
            Some(graph.exceptblock.clone()),
            None,
        );
        ovf_link.extravars(Some(except_etype), Some(except_evalue));

        let mut exits = vec![normal_link, ovf_link.into_ref()];
        if with_catch_all {
            let mut catch_all = Link::new(
                vec![except_etype.into(), except_evalue.into()],
                Some(graph.exceptblock.clone()),
                None,
            );
            catch_all.extravars(Some(except_etype), Some(except_evalue));
            exits.push(catch_all.into_ref());
        }
        start.closeblock(exits);

        let mut ssarepr = SSARepr::new(name);
        let ctx = LoweringContext {
            binary_op_fn_idx: 0,
            compare_op_fn_idx: 0,
            truth_fn_idx: 0,
            store_subscr_fn_idx: 0,
            build_list_fn_idx: 0,
            call_fn_idx_by_nargs: [0; 9],
        };
        flatten_graph_with_lowering(
            &graph,
            &mut ssarepr,
            ctx,
            Some(cpu),
            |v| Register::new(v.kind.unwrap_or(Kind::Ref), v.id.0 as u16),
            |c| match (&c.value, c.kind) {
                (ConstantValue::Opaque(opaque), Some(Kind::Ref)) => {
                    // Test-side lowering: encode the OverflowError
                    // marker as `ConstRef(0xFFFF_FFFF)` so the assertion
                    // can recognize it without resolving the runtime
                    // pointer.
                    let _ = opaque;
                    Operand::ConstRef(0xFFFF_FFFF)
                }
                _ => flatten_constant_operand(c),
            },
        );
        ssarepr
    }

    #[test]
    fn flatten_ovf_canraise_rewrites_to_jump_if_ovf_two_exits() {
        let mut cpu = super::super::cpu::Cpu::new();
        // ExceptionData fail-loud invariant: any caller of
        // get_standard_ll_exc_instance_by_class must pre-resolve.
        // Test pointer matches the `0xFFFF_FFFF` sentinel
        // flatten_ovf_canraise_graph's lower_constant closure recognises.
        cpu.rtyper
            .exceptiondata
            .resolve_standard_exception_pointers(|name| match name {
                "OverflowError" => Some(0xFFFF_FFFF),
                _ => None,
            });
        let ssarepr = flatten_ovf_canraise_graph("int_add_ovf_2exit", "int_add_ovf", false, &cpu);
        // Expected shape after flatten_graph:
        //   Label(startblock)
        //   int_add_jump_if_ovf TLabel(ovf_link) %i0 %i1 -> %i2
        //   <normal path: link renamings + ref_return>
        //   Label(ovf_link)
        //   raise ConstRef(OverflowError)
        //   ---
        let jump = ssarepr
            .insns
            .iter()
            .find(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "int_add_jump_if_ovf"))
            .expect("int_add_ovf must rewrite to int_add_jump_if_ovf per flatten.py:195");
        match jump {
            Insn::Op { args, result, .. } => {
                // First arg is the TLabel for the overflow target.
                assert!(matches!(args[0], Operand::TLabel(_)));
                // Remaining args are the original lhs/rhs registers.
                assert!(matches!(
                    args[1],
                    Operand::Register(Register {
                        kind: Kind::Int,
                        index: 0
                    })
                ));
                assert!(matches!(
                    args[2],
                    Operand::Register(Register {
                        kind: Kind::Int,
                        index: 1
                    })
                ));
                // Result is the original op's result Variable.
                assert!(matches!(
                    result,
                    Some(Register {
                        kind: Kind::Int,
                        index: 2
                    })
                ));
            }
            _ => unreachable!(),
        }
        // The handling_ovf=true arm of make_exception_link emits
        // `raise ConstRef(OverflowError)` (see flatten.py:165-170).
        let raise = ssarepr.insns.iter().find_map(|insn| match insn {
            Insn::Op { opname, args, .. } if opname == "raise" => Some(args),
            _ => None,
        });
        let raise_args =
            raise.expect("flatten.py:166-170 handling_ovf=true must emit `raise <const>`");
        assert_eq!(
            raise_args.len(),
            1,
            "raise carries exactly one operand: the OverflowError instance"
        );
        assert!(matches!(raise_args[0], Operand::ConstRef(0xFFFF_FFFF)));
        // The driver no longer emits `reraise` for the ovf direct-raise edge.
        assert!(
            !ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "reraise")),
            "handling_ovf=true must NOT emit reraise (upstream replaces it with `raise <const>`)"
        );
    }

    #[test]
    fn flatten_ovf_canraise_three_exits_emits_catch_all() {
        let mut cpu = super::super::cpu::Cpu::new();
        // ExceptionData fail-loud invariant: any caller of
        // get_standard_ll_exc_instance_by_class must pre-resolve.
        // Test pointer matches the `0xFFFF_FFFF` sentinel
        // flatten_ovf_canraise_graph's lower_constant closure recognises.
        cpu.rtyper
            .exceptiondata
            .resolve_standard_exception_pointers(|name| match name {
                "OverflowError" => Some(0xFFFF_FFFF),
                _ => None,
            });
        let ssarepr = flatten_ovf_canraise_graph("int_mul_ovf_3exit", "int_mul_ovf", true, &cpu);
        // Three-exit shape per flatten.py:201-203:
        //   - exits[1]: handling_ovf=true → raise <const>
        //   - exits[2]: handling_ovf=false → reraise (catch-all)
        // Both should appear in the SSARepr.
        let has_raise_const = ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "raise" && matches!(args.first(), Some(Operand::ConstRef(_)))
            )
        });
        let has_reraise = ssarepr
            .insns
            .iter()
            .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "reraise"));
        assert!(has_raise_const, "ovf direct edge must emit raise <const>");
        assert!(has_reraise, "Exception catch-all must emit reraise");
    }

    #[test]
    fn flatten_ovf_canraise_uses_seven_char_prefix() {
        // flatten.py:195 `opname[:7] + '_jump_if_ovf'` — verify the
        // prefix transform for each of the standard upstream `_ovf` ops.
        let mut cpu = super::super::cpu::Cpu::new();
        // ExceptionData fail-loud invariant: any caller of
        // get_standard_ll_exc_instance_by_class must pre-resolve.
        // Test pointer matches the `0xFFFF_FFFF` sentinel
        // flatten_ovf_canraise_graph's lower_constant closure recognises.
        cpu.rtyper
            .exceptiondata
            .resolve_standard_exception_pointers(|name| match name {
                "OverflowError" => Some(0xFFFF_FFFF),
                _ => None,
            });
        for (input, expected) in [
            ("int_add_ovf", "int_add_jump_if_ovf"),
            ("int_sub_ovf", "int_sub_jump_if_ovf"),
            ("int_mul_ovf", "int_mul_jump_if_ovf"),
        ] {
            let ssarepr = flatten_ovf_canraise_graph(input, input, false, &cpu);
            assert!(
                ssarepr
                    .insns
                    .iter()
                    .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == expected)),
                "{input} must rewrite to {expected} per flatten.py:195"
            );
        }
    }

    #[test]
    fn flatten_graph_canonical_four_arg_entry_works_without_lowering_ctx() {
        // Phase 4 — `flatten.py:63-70` orthodox 4-arg entry.  No
        // `lowering_ctx` parameter; `cpu=None` so dispatcher is
        // disabled and pre-rtype HLOp opnames passthrough.
        use crate::jit::flow::{Block, FunctionGraph};
        let retval = Variable::new(VariableId(99), Kind::Ref);
        let start = Block::shared(Vec::new());
        let graph = FunctionGraph::new("orthodox4arg", start.clone(), Some(retval));
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("loop_header", vec![Constant::signed(11).into()], None, 0),
        );
        start.closeblock(vec![
            super::super::flow::Link::new(
                vec![retval.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let mut regallocs =
            super::super::regalloc::perform_graph_register_allocation_all_kinds(&graph);
        let ssarepr = flatten_graph(&graph, &mut regallocs, false, None);
        assert_eq!(ssarepr.name, "orthodox4arg");
        assert!(
            ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "loop_header"))
        );
    }

    #[test]
    fn flatten_graph_with_regallocs_canonical_entry_returns_ssarepr() {
        // Phase 4 — `flatten.py:63-70` orthodox entry.
        // Build a trivial portal-like graph with a single
        // `loop_header` op (passthrough family — no LoweringContext
        // arm needs to fire) and verify the canonical entry returns a
        // populated `SSARepr` named after the graph.
        use crate::jit::flow::{Block, FunctionGraph};
        let retval = Variable::new(VariableId(0), Kind::Ref);
        let start = Block::shared(Vec::new());
        let graph = FunctionGraph::new("orthodox", start.clone(), Some(retval));
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("loop_header", vec![Constant::signed(7).into()], None, 0),
        );
        start.closeblock(vec![
            super::super::flow::Link::new(
                vec![retval.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let mut regallocs =
            super::super::regalloc::perform_graph_register_allocation_all_kinds(&graph);
        // Use the canonical `flatten_graph(graph, regallocs,
        // include_all_exc_links, cpu)` entry — the loop_header op is a
        // passthrough family and needs no LoweringContext arm to fire.
        let ssarepr = flatten_graph(&graph, &mut regallocs, false, None);
        assert_eq!(ssarepr.name, "orthodox");
        assert!(
            ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "loop_header")),
            "canonical entry must walk graph.startblock and emit the `loop_header` op"
        );
    }

    #[test]
    #[should_panic(
        expected = "detected a block containing ovfcheck() but no OverflowError is caught"
    )]
    fn flatten_ovf_op_in_non_canraise_block_panics() {
        // flatten.py:120-125 — `_ovf` op outside a 2-or-3-exit canraise
        // block is illegal.  Build such a block and verify the
        // `make_bytecode_block` validity check fires.
        use crate::jit::flow::{Block, FunctionGraph, Link};
        let lhs = Variable::new(VariableId(0), Kind::Int);
        let rhs = Variable::new(VariableId(1), Kind::Int);
        let res = Variable::new(VariableId(2), Kind::Int);
        let start = Block::shared(vec![lhs.into(), rhs.into()]);
        let graph = FunctionGraph::new("ovf_no_catch", start.clone(), Some(res));
        super::super::flow::push_op(
            &start,
            SpaceOperation::new(
                "int_add_ovf",
                vec![lhs.into(), rhs.into()],
                Some(res.into()),
                0,
            ),
        );
        // Close with a single normal link → returnblock; no canraise
        // exitswitch → triggers the validity panic.
        start.closeblock(vec![
            Link::new(vec![res.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);
        let mut ssarepr = SSARepr::new("ovf_no_catch");
        flatten_graph_with_closures(
            &graph,
            &mut ssarepr,
            |v| Register::new(v.kind.unwrap_or(Kind::Ref), v.id.0 as u16),
            flatten_constant_operand,
        );
    }
}
