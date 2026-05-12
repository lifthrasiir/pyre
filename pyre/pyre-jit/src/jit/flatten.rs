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

use std::{collections::HashMap, rc::Rc};

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
/// **Stub limitations (PRE-EXISTING-ADAPTATION on every variant).**
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

/// Classify whether an `Insn::Op` is a flatten_graph-emitted artifact
/// that has no walker-time `record_graph_op` counterpart by design.
/// Used by the `[phase4-graph-shape]` probe in `codewriter.rs` to
/// separate "true walker→graph gap" (closing requires
/// `record_graph_op` coverage) from "expected SSA-only emit"
/// (closes naturally once `flatten_graph(graph, regallocs)` becomes
/// the canonical SSARepr source — Task #227 walker restructure).
///
/// Currently recognises:
///   * Terminator ops emitted by `make_return` / `make_exception_link`
///     (`flatten.py:236-303`):
///     `int_return`, `ref_return`, `float_return`, `void_return`,
///     `raise`, `reraise`.
///   * Block-exit dispatch ops emitted by `insert_exits` /
///     `insert_switch_exits` (`flatten.py:107-200` 1-exit / 2-exit /
///     switch dispatch): `goto`, `goto_if_not`, `goto_if_not_<marker>`
///     (overflow / int_lt / ...), `goto_if_exception_mismatch`,
///     `switch`, `unreachable`, `catch_exception`.
///   * Link-rename `*_push`/`*_pop` and link-driven
///     `last_exception`/`last_exc_value` — `flatten.py:306-334
///     insert_renamings` + `generate_last_exc`. `*_push` carries one
///     register source (`int_push %iN`), `*_pop` carries one register
///     result (`int_pop -> %iN`); both arise from
///     `reorder_renaming_list` swap-cycle resolution.  `*_copy` is
///     not classified here: pyre's walker also emits register-source
///     copies for stack / local / call-argument movement, so
///     treating all register-source copies as SSA-only would hide
///     real walker→graph gaps until copy provenance is represented
///     explicitly.
///
///   * `OPNAME_LIVE` (`-live-`) — `liveness.py:5-12`. **Walker-shape
///     adaptation, not orthodox.** Upstream `-live-` placement comes
///     from two distinct producers:
///       1. **Graph-side, per raising/virtualizable/inline-call op**:
///          `jtransform.py:469-471 handle_residual_call` (post-call,
///          if `may_call_jitcodes` or `calldescr_canraise`),
///          `jtransform.py:481 handle_regular_call` (post inline_call),
///          `jtransform.py:845` (pre `getfield_vable_<kind>`).  These
///          end up in the SSARepr via `flatten.py:126 serialize_op`.
///       2. **SSA-only, per branch / raise / switch boundary**:
///          `flatten.py:142` (raise non-Variable),
///          `flatten.py:259` (before `goto_if_not`),
///          `flatten.py:285` (before `switch`),
///          `flatten.py:303` (per switch case label).  These have no
///          graph counterpart by design.
///     Pyre's renderer-side `flatten_graph` (`FlattenGraph::insert_exits`
///     / `make_return`) already mirrors group-2 line-for-line at
///     `flatten.rs:1000, 1139, 1208, 1228`.  Pyre's *codewriter walker*
///     (`emit_live_placeholder!` in
///     `codewriter.rs::transform_graph_to_jitcode`) is a different
///     producer: it runs 1:1 against the Python bytecode and pushes
///     `Insn::live(Vec::new())` at every PC entry to seed the
///     post-regalloc `all_liveness` table (`assembler.py:146-158`)
///     under pyre's bytecode-1:1 walker model.  That per-PC emission
///     intentionally has **no `record_graph_op` companion** — recording
///     it graph-side would create a `-live-` cluster the upstream
///     graph never holds and would mask real graph→inline gaps in the
///     `[phase4-graph-shape]` probe.  Classifying walker-emitted
///     per-PC `-live-` as an SSA-only artifact here is the matching
///     probe-side carveout.
///
///     **Convergence path back to RPython orthodox emission**:
///     when pyre's walker is restructured to drop per-PC `-live-` in
///     favour of group-2 emission only (mirror `flatten.py:142, 259,
///     285, 303` exactly the way `flatten_graph` already does) and
///     gain a graph-side group-1 emission (mirror jtransform's
///     residual_call / inline_call / vable getfield decomposition),
///     this clause must be removed so the probe surfaces real
///     graph→inline `-live-` gaps.  Until then it is a *known*
///     adaptation rather than a silent one — the docstring above
///     names the orthodox positions that pyre is not yet emitting.
pub fn is_ssa_only_artifact(insn: &Insn) -> bool {
    let Insn::Op { opname, .. } = insn else {
        return false;
    };
    // Walker-shape adaptation: the codewriter walker emits a `-live-`
    // placeholder at every PC entry (no `record_graph_op` companion),
    // unlike RPython's per-raising-op + per-branch emission. See the
    // docstring above for the orthodox emission sites and convergence
    // path.
    if opname == OPNAME_LIVE {
        return true;
    }
    // Terminators: `flatten.py:236-303 make_return` /
    // `make_exception_link`.
    if matches!(
        opname.as_str(),
        "int_return" | "ref_return" | "float_return" | "void_return" | "raise" | "reraise"
    ) {
        return true;
    }
    // Block-exit dispatch: `insert_exits` 1-/2-exit / `insert_switch_exits`
    // (`flatten.py:107-200`).
    if matches!(
        opname.as_str(),
        "goto"
            | "goto_if_not"
            | "goto_if_exception_mismatch"
            | "switch"
            | "unreachable"
            | "catch_exception"
    ) {
        return true;
    }
    // `goto_if_not_<marker>` from `flatten_tuple_exitswitch`
    // (`flatten.py:118-144`): overflow / int_lt / etc.
    if opname.starts_with("goto_if_not_") {
        return true;
    }
    // Link-driven loads: `generate_last_exc` (`flatten.py:336-352`).
    if matches!(opname.as_str(), "last_exception" | "last_exc_value") {
        return true;
    }
    // `*_copy` is deliberately counted as walker-emitted unless the
    // instruction carries explicit link-rename provenance. Pyre's walker
    // emits register-source copies for stack/local/call argument shuffles.
    // Link-rename `*_push %iN` / `*_pop -> %iN`
    // (`flatten.py:325-330 reorder_renaming_list`): swap-cycle
    // resolution. `*_push` carries a register source; `*_pop` carries
    // a register result. Both are flatten-time emissions with no
    // graph counterpart by design.
    matches!(
        opname.as_str(),
        "int_push" | "ref_push" | "float_push" | "int_pop" | "ref_pop" | "float_pop"
    )
}

/// Symmetric counterpart to [`is_ssa_only_artifact`]: classify whether
/// a graph-side `Insn::Op` is a pre-rtype `flowcontext.py` HLOperation
/// shape that has no inline-walker counterpart by design.
///
/// Pyre's codewriter pipeline runs two parallel emit streams at
/// different abstraction levels:
///
/// 1. **Graph (pre-rtype HLOperation, flowspace level)** —
///    `emit_frontend_*` helpers (`codewriter.rs:1390-1620`) record
///    `flowcontext.py:135-139 self.recorder.append(spaceop)` shapes
///    like `add(reg_r, reg_r) -> reg_r`, `lt(reg_r, reg_r) -> reg_r`,
///    `bool(reg_r) -> reg_i`, `setitem(reg_r, reg_r, reg_r)`,
///    `simple_call(reg_r, reg_r) -> reg_r`, etc.  Variables produced
///    here flow through `current_state.stack`, drive
///    `block.exitswitch`, and feed graph regalloc liveness — they are
///    load-bearing for graph-side analysis even though no runtime IR
///    consumer reads the `opname`.
///
/// 2. **Inline (post-rtype + helper-call, assembler level)** —
///    `emit_residual_call_shape` (`codewriter.rs:2513`) writes
///    `residual_call_{kinds}_{reskind}(const_int(fn_idx), list_X...,
///    descr)` directly into the SSARepr.  Pyre routes binary ops,
///    comparisons, `bool`, container access, etc., through helper
///    functions (`bh_binary_op_fn`, `bh_compare_op_fn`,
///    `bh_truth_fn`, `bh_setitem_fn`, ...).  This is RPython
///    post-rtype + helper-lowering — the proper assembler-level
///    shape pyre emits because pyre's values are uniformly
///    `PyObject*` (no rtyper, no `int_add`/`int_lt`/`direct_call`
///    typed split).
///
/// The two streams represent the SAME logical operations at different
/// abstraction levels.  Graph's `add(reg_r, reg_r)` corresponds to
/// inline's `residual_call_ir_r(box_int_fn) + residual_call_ir_r(
/// binary_add_fn) + ...` decomposition — multiple post-rtype helper
/// calls per graph HLOp.  **Op-count multiset_match across streams
/// cannot be 1:1 by design**.
///
/// `is_graph_only_artifact` lets the `[phase4-graph-shape]` probe
/// (`codewriter.rs:6420-6510`) classify the pre-rtype HLOps as
/// expected-graph-only, mirroring what `is_ssa_only_artifact` already
/// does for SSA-only artifacts (terminators, branch dispatch,
/// link-rename push/pop, walker per-PC `-live-`).  Without this
/// filter, the probe's "graph-only(top)" line surfaces every
/// `add/lt/bool/setitem/...` as if it were a graph→inline gap when in
/// fact the inline-side `residual_call_*` IS the corresponding
/// post-lowering emit.
///
/// Convergence path:
///   * Task #46 lands graph-side `residual_call_*` dual-write at each
///     `emit_residual_call` callsite (using Task #41's
///     `intern_call_descr_stub` + `flatten_descr_by_ptr` plumbing).
///     That closes the inline-only `residual_call_*` half of the gap
///     symmetrically.
///   * The pre-rtype HLOp filter here closes the graph-only half by
///     measurement honesty — the HLOp emit is intentional, not a bug.
///
/// Recognised opname families:
///   * Binary arithmetic (`emit_frontend_binary` →
///     `binary_opname`): `add`, `sub`, `mul`, `floordiv`, `mod`,
///     `truediv`, `pow`, `lshift`, `rshift`, `and_`, `or_`, `xor`
///     plus their `inplace_*` siblings, plus `getitem` (BINARY_SUBSCR).
///   * Comparison (`emit_frontend_compare` → `compare_opname`):
///     `lt`, `le`, `eq`, `ne`, `gt`, `ge`.
///   * Container subscript store (`emit_frontend_setitem`): `setitem`.
///   * Boolean coercion / unary (`emit_frontend_bool`,
///     `emit_frontend_neg`): `bool`, `neg`.
///   * Function call (`emit_frontend_simple_call`): `simple_call`.
///   * Sequence construction (`emit_frontend_newlist`): `newlist`.
///
/// **Not recognised** (intentionally surface as graph-only deviations):
/// `setattr` and `getattr` were previously included but removed
/// 2026-05-06 — inspection of `Instruction::StoreAttr` (codewriter.rs
/// :6116-6124) and `Instruction::LoadAttr` (codewriter.rs:6138-6146)
/// shows the inline side fires `emit_abort_permanent!` instead of a
/// `residual_call_*`, so there is no inline lowering to "match" the
/// graph HLOp.  Filtering them masked the lowering gap; surfacing
/// them as graph-only deviations in `[phase4-graph-shape]` honestly
/// reflects pyre's missing residual_call helpers for these families.
pub fn is_graph_only_artifact(insn: &Insn) -> bool {
    let Insn::Op { opname, .. } = insn else {
        return false;
    };
    if matches!(
        opname.as_str(),
        // Binary arithmetic + subscript (BinaryOperator → opname,
        // codewriter.rs:1541-1571 binary_opname).
        "add"
            | "sub"
            | "mul"
            | "floordiv"
            | "mod"
            | "truediv"
            | "pow"
            | "lshift"
            | "rshift"
            | "and_"
            | "or_"
            | "xor"
            | "getitem"
            | "inplace_add"
            | "inplace_sub"
            | "inplace_mul"
            | "inplace_floordiv"
            | "inplace_mod"
            | "inplace_truediv"
            | "inplace_pow"
            | "inplace_lshift"
            | "inplace_rshift"
            | "inplace_and"
            | "inplace_or"
            | "inplace_xor"
    ) {
        return true;
    }
    if matches!(
        opname.as_str(),
        // Comparison (ComparisonOperator → opname,
        // codewriter.rs:1592-1602 compare_opname).
        "lt" | "le" | "eq" | "ne" | "gt" | "ge"
    ) {
        return true;
    }
    matches!(
        opname.as_str(),
        // Container subscript store / unary / call / list construction
        // (emit_frontend_{setitem,bool,neg,simple_call,newlist}).
        // `setattr` and `getattr` are intentionally NOT here — see
        // function-level docstring for the abort_permanent rationale.
        "setitem" | "bool" | "neg" | "simple_call" | "newlist"
    )
}

/// Instruction tuple (`ssarepr.insns[i]`).
///
/// The four RPython tuple shapes enumerated above (`Label`, `-live-`,
/// `---`, regular op), plus one pyre-specific `PcAnchor` variant — see
/// its docstring for rationale. `-live-` shares the `Op` variant with
/// regular operations, matching RPython's tuple representation where
/// `insn[0] == '-live-'` is the discriminator.
#[derive(Debug, Clone)]
pub enum Insn {
    /// `(Label(name),)` — block-entry marker.
    Label(Label),
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
    /// PRE-EXISTING-ADAPTATION: pyre-only marker recording the SSARepr
    /// position where a Python bytecode (py_pc) starts. RPython has no
    /// equivalent because its jitcode is graph-derived (not Python-
    /// bytecode-1:1) and Python PCs do not appear in jitcode space.
    ///
    /// pyre's dispatch loop emits one `PcAnchor` at every Python PC so
    /// the trace-time dispatch can map `next_instr` to the JitCode byte
    /// offset post-assemble. The `compute_liveness` and `regalloc`
    /// passes ignore anchors entirely (no liveness, no interference, no
    /// rename); `assembler.assemble` records each anchor's byte offset
    /// without emitting any bytecode. This replaces the older
    /// dispatch-time `pc_map[py_pc] = current_pos()` snapshot, which
    /// became stale whenever `compute_liveness::remove_repeated_live`
    /// merged consecutive `-live-` markers and shifted insn indices.
    ///
    /// Closest RPython analog: `Label(block)` markers used to anchor
    /// merge-point block entries (`flatten.py`); pyre's anchor is the
    /// same idea applied per Python bytecode rather than per graph block.
    PcAnchor(usize),
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
    seen_blocks: HashMap<BlockRef, bool>,
    block_names: HashMap<BlockRef, String>,
    link_names: HashMap<LinkRef, String>,
    next_label_id: usize,
    include_all_exc_links: bool,
    /// Slice #48.18 (Option C pipeline-flip prep): when `Some`,
    /// `flatten_space_operation` routes pre-rtype HLOp opnames from
    /// the four retired families (BINARY_OP / COMPARE_OP / BOOL /
    /// SETITEM) through `try_flatten_retired_family_hlop_to_insn`,
    /// producing the post-rtype `residual_call_*` Insn the assembler
    /// expects.  When `None`, the legacy opname-passthrough behaves
    /// as before (emitting `Insn::op("add", ...)` etc. — useful for
    /// tests and structural-only graphs).
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
            seen_blocks: HashMap::new(),
            block_names: HashMap::new(),
            link_names: HashMap::new(),
            next_label_id: 0,
            include_all_exc_links: false,
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
            seen_blocks: HashMap::new(),
            block_names: HashMap::new(),
            link_names: HashMap::new(),
            next_label_id: 0,
            include_all_exc_links: false,
            lowering_ctx: None,
        }
    }

    /// Slice #48.18 (Option C pipeline-flip prep): GraphFlattener
    /// constructor that enables retired-family HLOp lowering.  Routes
    /// `add` / `lt` / ... / `bool` / `setitem` SpaceOperations through
    /// `try_flatten_retired_family_hlop_to_insn` (which lowers them to
    /// the matching `residual_call_*` Insn shape) before the
    /// passthrough opname-emit fallback.  Non-HLOp opnames (structural
    /// ops like `loop_header` / `jit_merge_point`, post-rtype
    /// `residual_call_*` ops recorded by factor-refactored families'
    /// graph dual-writes) keep their existing passthrough handling
    /// because the dispatcher's `try_*` returns `None` for them.
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
            seen_blocks: HashMap::new(),
            block_names: HashMap::new(),
            link_names: HashMap::new(),
            next_label_id: 0,
            include_all_exc_links: false,
            lowering_ctx: Some(lowering_ctx),
        }
    }

    pub fn emit_space_operation(&mut self, op: &SpaceOperation) {
        let insn = self.flatten_space_operation(op);
        self.ssarepr.insns.push(insn);
    }

    fn emitline(&mut self, insn: Insn) {
        self.ssarepr.insns.push(insn);
    }

    fn label_name_for_block(&mut self, block: &BlockRef) -> String {
        if let Some(name) = self.block_names.get(block) {
            return name.clone();
        }
        let name = format!("block{}", self.next_label_id);
        self.next_label_id += 1;
        self.block_names.insert(block.clone(), name.clone());
        name
    }

    fn label_name_for_link(&mut self, link: &LinkRef) -> String {
        if let Some(name) = self.link_names.get(link) {
            return name.clone();
        }
        let name = format!("link{}", self.next_label_id);
        self.next_label_id += 1;
        self.link_names.insert(link.clone(), name.clone());
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
        match self.flatten_value(value) {
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
                    let operand = self.flatten_value(value);
                    self.emitline(Insn::op(opname, vec![operand]));
                }
            },
            [_, exc_value] => {
                if exc_value.as_variable().is_some() {
                    self.emitline(Insn::live(Vec::new()));
                }
                let operand = self.flatten_value(exc_value);
                self.emitline(Insn::op("raise", vec![operand]));
            }
            _ => panic!("make_return expects 1 or 2 args, got {}", args.len()),
        }
        self.emitline(Insn::Unreachable);
    }

    fn make_link(&mut self, link: &LinkRef, handling_ovf: bool) {
        let (target, args, last_exception, last_exc_value, can_return_directly) = {
            let link_borrow = link.borrow();
            let target = link_borrow
                .target
                .clone()
                .expect("link target required for make_link");
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
            // RPython `flatten.py:130-160`: a final-target Link (target
            // has empty exits, i.e. returnblock or exceptblock) carries
            // exactly 1 or 2 args.  Any other arg count for a final
            // target is a walker NEW-DEVIATION (orphan join-point with
            // FrameState-merge inputargs — see
            // `w1_root_cause_analysis_2026_05_07.md`).  Assert the
            // invariant fail-loud; the `[phase4-flatten-graph]` probe
            // wraps in `catch_unwind` so probe runs report `panic=true`
            // rather than crashing production (production paths use
            // the inline ssarepr emit, not flatten_graph).
            if target_is_final {
                assert!(
                    matches!(collected_args.len(), 1 | 2),
                    "make_link: final-target Link with args.len={} \
                     (W-1 invariant: returnblock/exceptblock only carry 1 or 2 \
                     args per flatten.py:130-160)",
                    collected_args.len(),
                );
            }
            (
                target,
                collected_args,
                link_borrow.last_exception,
                link_borrow.last_exc_value,
                target_is_final && !uses_last_exception && !uses_last_exc_value,
            )
        };
        if can_return_directly {
            self.make_return(&args);
            return;
        }
        let _ = (last_exception, last_exc_value, handling_ovf);
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
            assert!(
                !handling_ovf,
                "overflow exception edges are not modeled in pyre flatten_graph yet"
            );
            self.emitline(Insn::op("reraise", Vec::new()));
            self.emitline(Insn::Unreachable);
            return;
        }
        self.make_link(link, handling_ovf);
    }

    fn insert_exits(&mut self, block: &BlockRef, handling_ovf: bool) {
        let exits = block.borrow().exits.clone();
        if exits.len() == 1 {
            self.make_link(&exits[0], handling_ovf);
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
            let normal_link = exits[0].clone();
            {
                let lb = normal_link.borrow();
                assert!(
                    lb.last_exception.is_none() && lb.llexitcase.is_none(),
                    "W-3 invariant: canraise block exits[0] must be the \
                     normal-flow link (last_exception=None, \
                     llexitcase=None) per flatten.py:211"
                );
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
            let catch_links: Vec<LinkRef> = exits.iter().skip(1).cloned().collect();
            {
                let mut catch_all_seen = false;
                for link in &catch_links {
                    let is_catch_all = link.borrow().llexitcase.is_none();
                    assert!(
                        !(catch_all_seen && !is_catch_all),
                        "W-3 invariant: canraise catch links must be in \
                         typed-then-catch-all order per flatten.py:223-238 \
                         (flowcontext.py emits Exception catch-all last)"
                    );
                    catch_all_seen = catch_all_seen || is_catch_all;
                }
            }
            if !self.include_all_exc_links && block.borrow().raising_op().is_none() {
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
                let llexitcase = link.borrow().llexitcase.clone();
                if let Some(case) = llexitcase {
                    let case_operand = self.flatten_value(&case);
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
                    ("goto_if_not".to_owned(), vec![self.flatten_value(&value)])
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
                super::flow::ExitSwitchElement::Value(value) => self.flatten_value(&value),
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
        let Some(super::flow::ExitSwitch::Value(exitswitch)) = exitswitch else {
            let block_for_panic = exits[0]
                .borrow()
                .prevblock
                .as_ref()
                .and_then(|w| w.upgrade())
                .expect("link prevblock");
            let block_borrow = block_for_panic.borrow();
            let exitcase_summary: Vec<String> = exits
                .iter()
                .map(|link| {
                    let lb = link.borrow();
                    format!("(llexitcase={:?})", lb.llexitcase)
                })
                .collect();
            panic!(
                "flatten_graph: unsupported exits shape for block with {} exits; exitswitch={:?}, src(ops.len={}, canraise={}), exits={:?}",
                exits.len(),
                exitswitch,
                block_borrow.operations.len(),
                block_borrow.canraise(),
                exitcase_summary,
            );
        };
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

        let switch_value = self.flatten_value(&exitswitch);
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

        let mut renamings: HashMap<Kind, (Vec<RenameOperand>, Vec<RenameOperand>)> = HashMap::new();
        for (src, dst) in pairs {
            let (frm, to) = renamings.entry(dst.kind).or_default();
            frm.push(src);
            to.push(RenameOperand::Register(dst));
        }
        for &kind in &Kind::ALL {
            if let Some((frm, to)) = renamings.get(&kind) {
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

    fn make_bytecode_block(&mut self, block: BlockRef, handling_ovf: bool) {
        if block.borrow().exits.is_empty() {
            let args = block.borrow().inputargs.clone();
            // RPython `flatten.py:130-160`: empty-exits blocks are
            // exclusively `returnblock` (1 arg) or `exceptblock`
            // (2 args), and both are handled by `make_return`.  Any
            // other empty-exits block shape is a walker NEW-DEVIATION
            // (orphan join-point left by the PC-sequential walker
            // creating a fresh `joinpoints[pc]` entry with FrameState-
            // merge inputargs and no incoming fall-through edge —
            // see `w1_root_cause_analysis_2026_05_07.md`).
            // Fail-loud here so any future flatten_graph driver
            // promotion against such graphs surfaces immediately
            // instead of silently emitting `Insn::Unreachable`; the
            // `[phase4-flatten-graph]` probe wraps this in
            // `catch_unwind` (codewriter.rs:7625), so probe runs
            // report `panic=true` rather than crashing production.
            assert!(
                matches!(args.len(), 1 | 2),
                "make_bytecode_block: empty-exits block with inputargs.len={} \
                 (W-1 invariant: only returnblock/exceptblock have empty exits, \
                 with 1 or 2 args per flatten.py:130-160)",
                args.len(),
            );
            self.make_return(&args);
            return;
        }
        if self.seen_blocks.contains_key(&block) {
            let target = self.tlabel_for_block(&block);
            self.emitline(Insn::op("goto", vec![target]));
            self.emitline(Insn::Unreachable);
            return;
        }
        self.seen_blocks.insert(block.clone(), true);
        let block_label = self.label_for_block(&block);
        self.emitline(block_label);
        let operations = block.borrow().operations.clone();
        for op in &operations {
            self.emit_space_operation(op);
        }
        self.insert_exits(&block, handling_ovf);
    }

    fn flatten_space_operation(&mut self, op: &SpaceOperation) -> Insn {
        // Slice #48.18: if the GraphFlattener was constructed with a
        // `LoweringContext`, retired-family HLOp opnames (`add` / `lt`
        // / ... / `bool` / `setitem`) lower to the matching
        // `residual_call_*` Insn shape via the dispatcher.  Non-HLOp
        // opnames return `None` from the dispatcher and fall through
        // to the legacy opname-passthrough below.
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
        let args = op.args.iter().map(|arg| self.flatten_arg(arg)).collect();
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
            SpaceOperationArg::Value(value) => self.flatten_value(value),
            SpaceOperationArg::ListOfKind(list) => Operand::ListOfKind(ListOfKind::new(
                list.kind,
                list.content
                    .iter()
                    .map(|value| self.flatten_value(value))
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

    fn flatten_value(&mut self, value: &FlowValue) -> Operand {
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
    matches!(
        exitcase,
        Some(FlowValue::Constant(Constant {
            value: ConstantValue::Str(value),
            ..
        })) if value == "default"
    )
}

fn switch_llexitcase_key(llexitcase: &Option<FlowValue>) -> i64 {
    match llexitcase {
        Some(FlowValue::Constant(Constant {
            value: ConstantValue::Signed(value),
            ..
        })) => *value,
        other => panic!("flatten_graph: switch link requires signed llexitcase, got {other:?}"),
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
        (ConstantValue::Opaque(_), Some(Kind::Ref)) => {
            panic!("GraphFlattener: opaque ref constants need runtime lowering support")
        }
        other => panic!("GraphFlattener: unsupported constant operand {other:?}"),
    }
}

/// Probe-side lowering: same as [`flatten_constant_operand`] but
/// returns a placeholder for `Opaque(Ref)` instead of panicking.
///
/// `flatten_arg_for_probe` runs from a debug-only shape comparison
/// pass that walks every graph op; portal `jit_merge_point` carries an
/// `Opaque(Ref)` pycode constant
/// (`pyre/pyre-jit/src/jit/codewriter.rs:4351-4356`), and production
/// emission threads a per-call `lower_constant` closure
/// (`codewriter.rs:4381-4396`) to recover the real `w_code` pointer.
/// The probe has no such closure available, so route `Opaque(Ref)` to
/// `Operand::ConstRef(0)`. `shape_descriptor`
/// (`codewriter.rs:1179-1215`) tags `Operand::ConstRef(_)` as
/// `"const_ref"` regardless of value, so the placeholder still
/// produces the same shape signature production emits when its
/// closure lowers the same op to `Operand::ConstRef(real_ptr)`.
pub(super) fn flatten_constant_operand_for_probe(constant: &super::flow::Constant) -> Operand {
    match (&constant.value, constant.kind) {
        (ConstantValue::Opaque(_), Some(Kind::Ref)) => Operand::ConstRef(0),
        _ => flatten_constant_operand(constant),
    }
}

/// Block-level walker matching `rpython/jit/codewriter/flatten.py:60
/// flatten_graph(graph, regallocs)`.  Walks every block in `graph`,
/// emits a `Label` for each block and an `Insn` for each
/// `SpaceOperation`, producing the SSARepr that the assembler consumes.
///
/// `regallocs` keyed by `Kind` provides the per-kind
/// `GraphAllocationResult` that the caller computed via
/// `regalloc::perform_graph_register_allocation_all_kinds(graph)`
/// (upstream `codewriter.py:44-46`'s `regallocs` dict).  `get_register`
/// projects `Variable` to `Register` using the appropriate per-kind
/// coloring; `lower_constant` handles non-raw constants (pycode opaque
/// refs, jitdriver descrs, etc.) that default `flatten_constant_operand`
/// cannot express on its own.
///
/// **Phase 1 scaffold for Task #224**: currently covers the ops whose
/// graph shape exists in production (`loop_header`, `jit_merge_point`
/// with the 7-arg upstream shape); ops that pyre's walker still emits
/// directly into the SSARepr (the bulk of opcodes in
/// `codewriter.rs::transform_graph_to_jitcode`) do not yet have a
/// corresponding graph-side `SpaceOperation`, so `flatten_graph`
/// cannot reproduce the walker's full output.  Wiring Phase 1c
/// replaces the direct SSARepr emission with `record_graph_op` at
/// every walker emit point and then switches production to call this
/// function in place of the walker's interleaved
/// `ssarepr.insns.push(...)` calls.
///
/// Matches upstream structure:
/// - `flatten_graph`: driver entry point (this function)
/// - `generate_ssa_form`: block iteration + per-op dispatch
///   (delegated here to `GraphFlattener::emit_space_operation`)
/// - `make_bytecode_block`/`make_link`/`insert_exits`: block boundary
///   handling — not yet implemented; `Label` insertion happens at
///   block entry only, `insert_exits` equivalent is not yet wired.
pub fn flatten_graph<F, C>(
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
    flattener.make_bytecode_block(graph.startblock.clone(), false);
}

/// Slice #48.18 (Option C pipeline-flip prep): variant of
/// [`flatten_graph`] that threads a `LoweringContext` so the
/// retired-family pre-rtype HLOps (`add` / `lt` / `bool` / `setitem`)
/// lower to their post-rtype `residual_call_*` Insn shapes during the
/// per-block walk.  Non-HLOp opnames retain the existing passthrough
/// handling (`Insn::op(opname, args)`), so this driver is byte-equivalent
/// to [`flatten_graph`] on graphs that contain no retired-family HLOps.
///
/// The dispatcher routes constants through the caller-supplied
/// `lower_constant` closure (matching the `flatten_arg` production
/// path), so `Opaque(Ref)` constants on `jit_merge_point` etc. lower
/// the same way they would through [`flatten_graph`].
///
/// **Still a Phase 1 SCAFFOLD**: production callers don't yet flip to
/// this driver because most ops the walker emits inline (LoadConst,
/// LoadGlobal, CALL, BuildList, ...) lack graph SpaceOperations.
/// Subsequent slices (Slice #48.19+) introduce frontend HLOp recording
/// at those walker arms; this driver becomes the production SSARepr
/// producer once every walker emit point has a graph counterpart.
pub fn flatten_graph_with_lowering<F, C>(
    graph: &super::flow::FunctionGraph,
    ssarepr: &mut SSARepr,
    lowering_ctx: LoweringContext,
    get_register: F,
    lower_constant: C,
) where
    F: FnMut(Variable) -> Register,
    C: FnMut(&Constant) -> Operand,
{
    let mut flattener =
        GraphFlattener::new_with_full_lowering(ssarepr, get_register, lower_constant, lowering_ctx);
    flattener.make_bytecode_block(graph.startblock.clone(), false);
}

/// Phase 4 Session 18 (Task #227 prerequisite) — single-family parallel
/// flatten probe.  Walks `graph.iterblocks()` (DFS from startblock per
/// `flowspace/model.py:66-77 FunctionGraph.iterblocks`) and emits one
/// `Insn::Op` per `block.operations` entry whose `opname` matches
/// `family_opname`, using `get_register` to project graph Variables onto
/// register slots.  Constants in arg position lower through
/// `flatten_constant_operand`.
///
/// **NOT a `flatten_graph` replacement.**  `rpython/jit/codewriter/
/// flatten.py:63 flatten_graph` runs `enforce_input_args()` followed by
/// `generate_ssa_form()` (`flatten.py:88` + `:102`); the latter walks
/// `make_bytecode_block`/`make_link`/`insert_exits` recursively and
/// emits the canonical `Label` per block, `make_return` /
/// `make_exception_link` per terminator, and `insert_renamings` per
/// link.  This helper deliberately skips ALL of that:
///   - no `enforce_input_args` simulation (start inputarg colors are
///     left at their raw chordal-coloring assignment, which can differ
///     from the canonical `0, 1, 2, …` per kind),
///   - no `Label` emission (the SSARepr would interleave block-entry
///     labels with the family ops),
///   - no `insert_exits` / `make_link` / `insert_renamings` (link-arg
///     rename `*_copy` / `*_push` / `*_pop` ops are absent),
///   - no return / exception terminator emission (`make_return`,
///     `make_exception_link`, `reraise`, `raise`),
///   - no `last_exception` / `last_exc_value` book-keeping.
///
/// The helper exists for the `[phase4-flatten-family]` probe at
/// `codewriter.rs::transform_graph_to_jitcode` whose goal is the
/// narrower question "does the graph carry the SAME `family_opname`
/// op sequence the inline walker emits, in the same order and with
/// the same operand shape, IGNORING register colors?".  Probe-positive
/// answer is necessary but not sufficient for retirement: full
/// retirement still requires regalloc unification (graph regalloc and
/// SSA `RegisterLayout::compute` produce different colorings) and
/// proper canonical `flatten_graph` plumbing including all the items
/// above.  Tracked as Task #227 walker restructure.
pub fn flatten_family_ops<F>(
    graph: &super::flow::FunctionGraph,
    family_opname: &str,
    mut get_register: F,
) -> Vec<Insn>
where
    F: FnMut(Variable) -> Register,
{
    let mut out = Vec::new();
    for block in graph.iterblocks() {
        let block = block.borrow();
        for op in &block.operations {
            if op.opname != family_opname {
                continue;
            }
            if let Some(insn) = flatten_op_to_insn(op, &mut get_register) {
                out.push(insn);
            }
        }
    }
    out
}

/// Walk every block in `graph.iterblocks()` DFS order and produce the
/// `Vec<Insn>` that a future `flatten_graph(graph, regallocs)` driver
/// would emit BEFORE `Label`/terminator/`insert_renamings` emission.
/// All-families generalisation of [`flatten_family_ops`]; same caveats
/// (`flatten_family_ops` docstring above) apply — this is NOT a full
/// `flatten_graph` replacement, only the `block.operations` body walk.
///
/// Used by the `[phase4-graph-shape]` probe to surface every remaining
/// orphan-inline emit family across the whole graph at once, not just
/// the per-family `[phase4-flatten-family]` cover. Probe-positive
/// answer ("graph multiset == inline multiset for ALL Op opnames") is
/// the structural precondition for retiring the entire walker → SSA
/// inline emit path in favour of post-walker `flatten_graph` (Task
/// #227 walker restructure endgame).
pub fn flatten_all_graph_ops<F>(
    graph: &super::flow::FunctionGraph,
    mut get_register: F,
) -> Vec<Insn>
where
    F: FnMut(Variable) -> Register,
{
    let mut out = Vec::new();
    for block in graph.iterblocks() {
        let block = block.borrow();
        for op in &block.operations {
            if let Some(insn) = flatten_op_to_insn(op, &mut get_register) {
                out.push(insn);
            }
        }
    }
    out
}

/// Lower one `SpaceOperation` to a single `Insn::Op` using probe-side
/// argument flattening. Shared between [`flatten_family_ops`] and
/// [`flatten_all_graph_ops`] so both paths agree on result-handling
/// and argument lowering. Returns `None` only when the op is not
/// representable as a single `Insn::Op` (currently never).
fn flatten_op_to_insn<F>(op: &super::flow::SpaceOperation, get_register: &mut F) -> Option<Insn>
where
    F: FnMut(Variable) -> Register,
{
    let args: Vec<Operand> = op
        .args
        .iter()
        .map(|arg| flatten_arg_for_probe(arg, get_register))
        .collect();
    let insn = match &op.result {
        None => Insn::op(op.opname.clone(), args),
        Some(FlowValue::Variable(variable)) => {
            let reg = get_register(*variable);
            Insn::op_with_result(op.opname.clone(), args, reg)
        }
        Some(FlowValue::Constant(constant)) => {
            // Same invariant as `flatten_space_operation` (line ~1271):
            // graph op results must be Variables. Panic with the same
            // message so a probe-side hit surfaces the broken graph
            // exactly the way the production path would, instead of
            // silently emitting a no-result Op that would falsely
            // shape-match an inline-walker entry with a `dst` register.
            panic!(
                "GraphFlattener probe: op {} has Constant result {:?}; \
                 flow graph results must be Variables",
                op.opname, constant
            );
        }
    };
    Some(insn)
}

fn flatten_arg_for_probe<F>(arg: &super::flow::SpaceOperationArg, get_register: &mut F) -> Operand
where
    F: FnMut(Variable) -> Register,
{
    flatten_arg_with_lowering(arg, get_register, &mut flatten_constant_operand_for_probe)
}

/// Generalized `SpaceOperationArg → Operand` lowering with a
/// caller-supplied `lower_constant` closure that decides how
/// `FlowValue::Constant` values map to `Operand`s.
///
/// Slice #48.18 (Option C pipeline-flip prep): the four
/// `lower_*_hlop_to_insn` helpers used to call `flatten_arg_for_probe`
/// directly, hardcoding probe-side constant lowering
/// (`flatten_constant_operand_for_probe` — `Opaque(Ref) → ConstRef(0)`
/// placeholder).  Production-side callers
/// (`GraphFlattener::flatten_arg` via `self.lower_constant`) need a
/// real pycode pointer for `Opaque(Ref)`, so this generalized version
/// accepts the lowering closure as a parameter.  Variable / list /
/// descr / indirect-call-targets handling is identical to
/// `flatten_arg_for_probe`; only constant operand lowering is
/// pluggable.
///
/// The probe-side wrapper [`flatten_arg_for_probe`] threads
/// `flatten_constant_operand_for_probe` as `lower_constant`; the
/// production side threads `&mut self.lower_constant`.
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
/// Task #48 micro-slice 1: BINARY_OP family lowering.  Not yet wired
/// into `flatten_op_to_insn`; lives standalone with unit tests until
/// micro-slice 2 introduces the `[phase4-flatten-lowering]`
/// sequence_match probe that compares this helper's output against
/// the existing dual-write residual_call across production fixtures.
/// Once verified byte-equivalent, micro-slice 3 retires the inline
/// `emit_residual_call(binary_op_fn_idx, ...)` callsite plus the
/// matching graph dual-write — the graph then carries only the
/// `add(...)` HLOp and flatten lowers it here into the SSARepr.
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
/// Mirrors [`lower_binary_op_hlop_to_insn`]'s output shape so the
/// `[phase4-flatten-lowering]` probe's `sequence_match=true`
/// invariant guarantees byte-equivalence with what
/// `emit_residual_call_shape` produced before retirement.
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
/// and LOAD_GLOBAL (`load_global_fn`, Plain — `(ns, code, namei)`
/// per codewriter.rs:5598-5615).  All share `arg_kinds = [Ref,
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
/// Task #48 micro-slice 4 (COMPARE_OP retirement): same pattern as
/// [`lower_binary_op_hlop_to_insn`].  The
/// `[phase4-flatten-lowering]` probe gains a parallel COMPARE_OP
/// arm; once `sequence_match=true` is verified the inline
/// `emit_residual_call(compare_fn_idx, ...)` callsite plus its
/// graph dual-write are retired in favour of the
/// `build_compare_op_residual_call_ir_r_insn` helper.
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
/// `load_global_fn` has signature `(ns: Ref, code: Ref, namei: Int)
/// → Ref` with `CallFlavor::Plain` (per codewriter.rs:2176-2185 —
/// `bh_load_global_fn` is a flat namespace lookup that can `NameError`
/// but cannot force virtuals; matches `EF_CAN_RAISE`).  Same
/// `(Ref, Ref, Int) → Ref` arity as BINARY_OP/COMPARE_OP — only the
/// CallFlavor on the EffectInfo descr differs.
///
/// The matching graph dual-write at codewriter.rs (LoadGlobal arm)
/// records `CallFlavor::Plain` so the SSA helper, the inline
/// SSARepr emit, and the graph residual_call agree end-to-end.
pub fn build_load_global_fn_residual_call_ir_r_insn(
    load_global_fn_idx: u16,
    namei: i64,
    ns_reg: u16,
    code_reg: u16,
    dst_reg: u16,
) -> Insn {
    build_residual_call_ir_r_insn_from_operands(
        load_global_fn_idx,
        namei,
        Operand::Register(Register::new(Kind::Ref, ns_reg)),
        Operand::Register(Register::new(Kind::Ref, code_reg)),
        CallFlavor::Plain,
        Register::new(Kind::Ref, dst_reg),
    )
}

/// Construct the CALL-family `residual_call_r_r` Insn from raw
/// register indices.  Production codewriter callsite (Slice #48.9
/// factor refactor) replaces the prior `emit_residual_call(
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
/// Insn.  Production codewriter callsite (Slice #48.15 factor
/// refactor) replaces the prior `emit_residual_call(
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
fn build_residual_call_r_r_insn_from_operands(
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
/// `ref_operands.len()` plus a `flavor` parameter (extended in
/// Slice #48.15 from MayForce-only to support the
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
/// register indices.  Production codewriter callsites (Slice #48.10
/// factor refactor) replace three prior `emit_residual_call(
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
/// `build_list_fn` has C ABI `extern "C" fn(i64, i64, i64)` —
/// always 3 i64 parameters dispatched internally by the leading
/// `argc` (`bh_build_list_fn` per `cpu.rs`).  The trailing 2 slots
/// hold real boxed-Ref pointers when the corresponding item is
/// present, or `0` (dummy bit pattern, routed through the int
/// constants pool per `make_three_lists` jtransform.py:437-445)
/// when absent.  Per codewriter.rs:5945-5954, `argc > 2` falls
/// through to `emit_abort_permanent` and never invokes this
/// helper, so `argc ∈ {0, 1, 2}`.
///
/// Inline arg order from `emit_residual_call_shape` for call-args
/// `[ConstInt(argc), maybe_Reg_or_ConstInt(item0),
/// maybe_Reg_or_ConstInt(item1)]`:
///   * `args_i` = leading argc + each absent item's `0` dummy.
///   * `args_r` = each present item's Reg.
///   * `arg_kinds` preserves call-order (NOT bucket-order): always
///     `[Int, item0_kind, item1_kind]` where each item kind is
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
    debug_assert!(
        argc <= 2,
        "BuildList helper only supports argc ∈ {{0, 1, 2}}"
    );
    debug_assert_eq!(arg_regs.len(), argc, "arg_regs length must match argc");
    let mut arg_kinds: Vec<Kind> = Vec::with_capacity(3);
    let mut args_i: Vec<Operand> = Vec::with_capacity(3);
    let mut args_r: Vec<Operand> = Vec::with_capacity(2);
    // Leading argc slot — always Int.
    arg_kinds.push(Kind::Int);
    args_i.push(Operand::ConstInt(argc as i64));
    // Trailing 2 slots — Ref if present, Int dummy `0` if absent.
    for i in 0..2 {
        if i < argc {
            arg_kinds.push(Kind::Ref);
            args_r.push(Operand::Register(Register::new(Kind::Ref, arg_regs[i])));
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
/// Task #48 micro-slice 5 (BOOL retirement): same per-family
/// pattern as micro-slices 3-4 but a different residual_call
/// shape.  After the `[phase4-flatten-lowering]` probe verifies
/// `sequence_match=true` on production fixtures, the inline
/// `emit_residual_call(truth_fn_idx, ...)` callsites at
/// PopJumpIfFalse / PopJumpIfTrue and their graph dual-writes
/// retire in favour of `build_truth_fn_residual_call_r_i_insn`.
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
    let result_reg = match &op.result {
        Some(super::flow::FlowValue::Variable(var)) => get_register(*var),
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
/// Slice #48.17 (Option C pipeline-flip prep): a future post-walker
/// `flatten_graph(graph, ssarepr, ctx)` driver calls this dispatcher
/// once per `block.operations` entry to translate graph ops back into
/// the SSARepr Insn stream the assembler consumes.  Today the
/// production path emits the inline `residual_call_*` Insns directly
/// via the `build_*_residual_call_*_insn` helpers at every walker
/// callsite while the graph carries pre-rtype HLOps for the retired
/// families and post-rtype `residual_call_*` ops for the
/// factor-refactored families (Slice #48.7-#48.16).  After
/// retirement the walker would only record graph ops; this dispatcher
/// is the per-op core of the post-walker driver.
///
/// The `[phase4-flatten-driver]` probe at codewriter.rs verifies
/// that walking `graph.iterblocks()` once through this dispatcher
/// produces a residual_call_* Insn sequence (per retired family,
/// filtered by `(opname, fn_idx)`) byte-equivalent in shape to the
/// inline-emitted SSARepr.  Probe-positive answer is the precondition
/// for switching production from inline emit to a post-walker
/// `flatten_graph` driver that uses this dispatcher.
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
    flatten_op_to_insn(op, get_register)
}

/// Retired-family-only variant of [`flatten_op_to_insn_with_lowering`]:
/// dispatches to the four `lower_*_hlop_to_insn` helpers and returns
/// `None` when no arm matches, instead of falling through to the
/// passthrough [`flatten_op_to_insn`].
///
/// Slice #48.18: `GraphFlattener::flatten_space_operation` uses this
/// variant to avoid double-handling of non-HLOp ops.  The dispatcher's
/// passthrough fallback uses `flatten_arg_for_probe` (probe-side
/// constant lowering, `Opaque(Ref) → ConstRef(0)` placeholder) which
/// would silently corrupt production lowering of `jit_merge_point`'s
/// pycode `Opaque(Ref)` constant; routing non-HLOp ops back into the
/// existing `flatten_space_operation` passthrough preserves the
/// production `self.lower_constant` behavior for those ops while still
/// using the dispatcher for the four retired families.
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
    None
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
/// (Slice #48.15 factor refactor) replace the prior
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
/// register indices.  Production codewriter callsite (Slice #48.7
/// factor refactor) replaces the prior `emit_residual_call(
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

        flattener.emit_space_operation(&op);

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

        flattener.emit_space_operation(&op);

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
        flatten_graph(
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
        flatten_graph(
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
        flatten_graph(
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
        let mut graph = FunctionGraph::new("bool_branch", start.clone(), Some(retval));

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
        flatten_graph(
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
        flatten_graph(
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
        flatten_graph(
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
        flatten_graph(
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

        flattener.emit_space_operation(&op);

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
        // Slice #48.18: a graph carrying one HLOp from each of the four
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
        };

        let mut ssarepr = SSARepr::new("retired_families");
        flatten_graph_with_lowering(
            &graph,
            &mut ssarepr,
            ctx,
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
        // Slice #48.19: the `[phase4-flatten-graph]` probe runs the
        // FULL `flatten_graph_with_lowering` driver end-to-end against
        // a fresh SSARepr.  Production graphs span multiple blocks, so
        // the per-family `(opname, fn_idx)` filter must survive the
        // GraphFlattener's `make_link` / `insert_exits` block boundary
        // emission (Labels, terminators, link renamings) without
        // dropping or reordering the retired-family residual_calls.
        // This test pins that invariant: a 2-block graph with one
        // BINARY_OP `add` in the start block and one COMPARE_OP `lt`
        // in the second block lowers to a single
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
        };

        let mut ssarepr = SSARepr::new("multi_block_lowering");
        flatten_graph_with_lowering(
            &graph,
            &mut ssarepr,
            ctx,
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
    #[should_panic(expected = "flatten_graph: unsupported exits shape for block with 2 exits")]
    fn flatten_graph_with_lowering_2_exit_no_exitswitch_now_fails_loud() {
        // The 2-exit-no-exitswitch shape was the W-4 self-loop fallback
        // (`[w-fallback w4/1797]`), retired in commit `537e26bccba` along
        // with the recognizer that walked both subtrees.  The follow-on
        // arm in `insert_switch_exits` now panics fail-loud if a future
        // walker regression reproduces the shape.  This test pins that
        // contract: constructing the shape and feeding it to
        // `flatten_graph_with_lowering` MUST panic with the documented
        // message.
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
        };

        let mut ssarepr = SSARepr::new("pyre_walker_2exit");
        flatten_graph_with_lowering(
            &graph,
            &mut ssarepr,
            ctx,
            |v| Register::new(v.kind.unwrap_or(Kind::Ref), v.id.0 as u16),
            flatten_constant_operand,
        );
    }

    #[test]
    fn flatten_graph_without_lowering_ctx_preserves_passthrough() {
        // Slice #48.18: when `flatten_graph` (no ctx) sees a retired-
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
        flatten_graph(
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
    /// `flatten_constant_operand_for_probe` (the probe-side default;
    /// `Opaque(Ref) → ConstRef(0)` placeholder) — the production-side
    /// `flatten_constant_operand` would panic on the placeholder
    /// fixtures these tests use.
    fn probe_constant_lowering() -> impl FnMut(&Constant) -> Operand {
        flatten_constant_operand_for_probe
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
        for opname in ["lt", "bool", "neg", "simple_call", "setitem", "newlist"] {
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
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();

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
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();
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
        };

        let hlop = SpaceOperation::new("sub", vec![lhs.into(), rhs.into()], Some(result.into()), 0);
        let mut get_register = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();
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
        let dual = flatten_op_to_insn(&dual_op, &mut get_register)
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
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();

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
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();
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
        };
        let hlop = SpaceOperation::new("eq", vec![lhs.into(), rhs.into()], Some(result.into()), 0);
        let mut get_register = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();
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
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();

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
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();
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
        };
        let hlop = SpaceOperation::new("bool", vec![cond.into()], Some(result.into()), 0);
        let mut get_register = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();
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
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();

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
        };
        let mut get_register = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();
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
        };
        let hlop = SpaceOperation::new(
            "setitem",
            vec![obj.into(), key.into(), value.into()],
            None,
            0,
        );
        let mut get_register = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();
        let lowered =
            lower_setitem_hlop_to_insn(&hlop, &ctx, &mut get_register, &mut lower_constant)
                .expect("SETITEM HLOp must lower");
        let prod = build_store_subscr_fn_residual_call_r_v_insn(53, 0, 1, 2);
        assert_eq!(format!("{lowered:?}"), format!("{prod:?}"));
    }

    #[test]
    fn flatten_op_to_insn_with_lowering_dispatches_binary_op_hlop() {
        // Slice #48.17: the unified dispatcher must route a `add`
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
        };
        let mut get_register_a = identity_register_mapper();
        let mut get_register_b = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();
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
        // Slice #48.17: dispatcher routes `lt` through
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
        };
        let mut get_register_a = identity_register_mapper();
        let mut get_register_b = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();
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
        // Slice #48.17: dispatcher routes `bool(v) → r` through
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
        };
        let mut get_register_a = identity_register_mapper();
        let mut get_register_b = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();
        let dispatched =
            flatten_op_to_insn_with_lowering(&op, &ctx, &mut get_register_a, &mut lower_constant)
                .expect("BOOL HLOp must lower via dispatcher");
        let direct = lower_bool_hlop_to_insn(&op, &ctx, &mut get_register_b, &mut lower_constant)
            .expect("BOOL HLOp must lower directly");
        assert_eq!(format!("{dispatched:?}"), format!("{direct:?}"));
    }

    #[test]
    fn flatten_op_to_insn_with_lowering_dispatches_setitem_hlop() {
        // Slice #48.17: dispatcher routes void-result `setitem(obj,
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
        };
        let mut get_register_a = identity_register_mapper();
        let mut get_register_b = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();
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
        // Slice #48.17: for opnames outside the four retired families
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
        };
        let mut get_register_a = identity_register_mapper();
        let mut get_register_b = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();
        let dispatched =
            flatten_op_to_insn_with_lowering(&op, &ctx, &mut get_register_a, &mut lower_constant)
                .expect("residual_call SpaceOperation must lower via dispatcher");
        let direct = flatten_op_to_insn(&op, &mut get_register_b)
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
        let mut lower_constant = probe_constant_lowering();
        let dual = flatten_op_to_insn(&dual_op, &mut get_register)
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
        // code)]), Descr(CallDescrStub{Plain, [Ref, Ref, Int]})] →
        // Reg(Ref, dst)`.  Same `(Ref, Ref, Int) → Ref` arity as
        // BINARY_OP/COMPARE_OP — distinguished only by the `Plain`
        // CallFlavor on the EffectInfo descr (per
        // codewriter.rs:2176-2185 — `bh_load_global_fn` cannot force
        // virtuals, matches `EF_CAN_RAISE`).
        let insn = build_load_global_fn_residual_call_ir_r_insn(
            /* load_global_fn_idx */ 12, /* namei */ 5, /* ns_reg */ 3,
            /* code_reg */ 4, /* dst_reg */ 3,
        );
        match insn {
            Insn::Op {
                opname,
                args,
                result: Some(reg),
            } => {
                assert_eq!(opname, "residual_call_ir_r");
                assert_eq!(reg, Register::new(Kind::Ref, 3));
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
                // args[2] = ListR([Reg(Ref, ns_reg), Reg(Ref, code_reg)]).
                match &args[2] {
                    Operand::ListOfKind(list) => {
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
                    }
                    other => panic!("expected ListOfKind(Ref, 2), got {other:?}"),
                }
                match &args[3] {
                    Operand::Descr(rc) => match &**rc {
                        DescrOperand::CallDescrStub(stub) => {
                            assert_eq!(stub.arg_kinds, vec![Kind::Ref, Kind::Ref, Kind::Int]);
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
        let dst_var = Variable::new(VariableId(3), Kind::Ref);
        let descr = intern_call_descr_stub(
            effect_info_for_call_flavor(CallFlavor::Plain),
            vec![Kind::Ref, Kind::Ref, Kind::Int],
            Some(Kind::Ref),
        );
        let dual_op = SpaceOperation::new(
            "residual_call_ir_r",
            vec![
                Constant::signed(12).into(),
                FlowListOfKind::new(Kind::Int, vec![Constant::signed(5).into()]).into(),
                FlowListOfKind::new(Kind::Ref, vec![ns_var.into(), code_var.into()]).into(),
                descr.into(),
            ],
            Some(dst_var.into()),
            0,
        );
        let mut get_register = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();
        let dual = flatten_op_to_insn(&dual_op, &mut get_register)
            .expect("residual_call SpaceOperation must lower");
        let prod = build_load_global_fn_residual_call_ir_r_insn(12, 5, 3, 4, 3);
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
        let glob = build_load_global_fn_residual_call_ir_r_insn(7, 0, 1, 2, 3);
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
        let mut lower_constant = probe_constant_lowering();
        let dual = flatten_op_to_insn(&dual_op, &mut get_register)
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
        let mut lower_constant = probe_constant_lowering();
        let dual = flatten_op_to_insn(&dual_op, &mut get_register)
            .expect("residual_call SpaceOperation must lower");
        let prod = build_box_int_fn_residual_call_ir_r_insn(4, 42, 7);
        assert_eq!(format!("{prod:?}"), format!("{dual:?}"));
    }

    #[test]
    fn build_build_list_fn_residual_call_ir_r_insn_emits_residual_call_ir_r_for_argc_2() {
        // Task #48 micro-slice 13: BuildList factor refactor.  argc=2:
        // both item slots are real Refs, no padding.  Expected shape:
        // `[ConstInt(fn_idx), ListI([ConstInt(2)]), ListR([Reg(item0),
        // Reg(item1)]), Descr(CallDescrStub{Plain, [Int, Ref, Ref]})]
        // → Reg(Ref, dst)`.
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
                    assert_eq!(list.content.len(), 1, "argc=2 → ListI=[2] only");
                    if let Operand::ConstInt(v) = &list.content[0] {
                        assert_eq!(*v, 2);
                    } else {
                        panic!("expected ConstInt(2) in ListI");
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
                        assert_eq!(stub.arg_kinds, vec![Kind::Int, Kind::Ref, Kind::Ref]);
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
        // argc=0: no real items.  arg_kinds=[Int, Int, Int], args_i=
        // [argc=0, dummy=0, dummy=0], args_r=[].  ListR is still
        // pushed (empty) because kinds="ir" includes 'r'.
        let argc_0 = build_build_list_fn_residual_call_ir_r_insn(18, 0, &[], 5);
        if let Insn::Op { args, .. } = &argc_0 {
            if let Operand::ListOfKind(list) = &args[1] {
                assert_eq!(list.content.len(), 3, "argc=0 → ListI=[0, 0, 0]");
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
                    assert_eq!(stub.arg_kinds, vec![Kind::Int, Kind::Int, Kind::Int]);
                }
            }
        } else {
            panic!("expected Insn::Op");
        }

        // argc=1: 1 real item, 1 padding.  arg_kinds=[Int, Ref, Int],
        // args_i=[argc=1, dummy=0], args_r=[reg].
        let argc_1 = build_build_list_fn_residual_call_ir_r_insn(18, 1, &[7], 5);
        if let Insn::Op { args, .. } = &argc_1 {
            if let Operand::ListOfKind(list) = &args[1] {
                assert_eq!(list.content.len(), 2, "argc=1 → ListI=[1, 0]");
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
                    assert_eq!(stub.arg_kinds, vec![Kind::Int, Kind::Ref, Kind::Int]);
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
            vec![Kind::Int, Kind::Ref, Kind::Ref],
            Some(Kind::Ref),
        );
        let dual_op = SpaceOperation::new(
            "residual_call_ir_r",
            vec![
                Constant::signed(18).into(),
                FlowListOfKind::new(Kind::Int, vec![Constant::signed(2).into()]).into(),
                FlowListOfKind::new(Kind::Ref, vec![item0.into(), item1.into()]).into(),
                descr.into(),
            ],
            Some(dst.into()),
            0,
        );
        let mut get_register = identity_register_mapper();
        let mut lower_constant = probe_constant_lowering();
        let dual = flatten_op_to_insn(&dual_op, &mut get_register)
            .expect("residual_call SpaceOperation must lower");
        let prod = build_build_list_fn_residual_call_ir_r_insn(18, 2, &[3, 4], 5);
        assert_eq!(format!("{prod:?}"), format!("{dual:?}"));
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
}
