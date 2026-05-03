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
}

/// Map a [`CallFlavor`] to the upstream-shape `EffectInfo` carrying the
/// equivalent `extraeffect` (and `call_release_gil_target` sentinel for
/// `ReleaseGil`). The mapping mirrors
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
/// None of those analyzers are ported to pyre yet — the entire
/// `majit-translate/src/{annotator,rtyper,translator}` infrastructure
/// (and the `readwrite_analyzer` / `quasiimmut_analyzer` /
/// `randomeffects_analyzer` / `collect_analyzer` callees in
/// `rpython/jit/codewriter/`) is on the unported roadmap.  In
/// consequence, every variant below leaves the analyzer-derived
/// fields at `EffectInfo::default()` (`oopspecindex = None`,
/// `*_descrs_*` = 0, `can_invalidate = false`, `extradescrs = None`,
/// `can_collect = true` per the Default impl).
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
/// Convergence: porting `majit-translate/src/jit_codewriter/call.rs`'s
/// analyzer trio (a multi-session epic — depends on annotator/rtyper/
/// translator) replaces the stub with real per-callee EI values.
/// Until then, this function is the producer-side stub that future
/// EI-aware optimizations must be careful not to rely on.  When a
/// concrete callee needs a specific EI field set (e.g. an
/// oopspec-specialized helper), construct the `EffectInfo` directly
/// at the call site rather than extending this `CallFlavor` mnemonic.
pub fn effect_info_for_call_flavor(flavor: CallFlavor) -> majit_ir::EffectInfo {
    use majit_ir::{EffectInfo, ExtraEffect};
    match flavor {
        // EF_CAN_RAISE — default normal call (`effectinfo.py:22`).
        // Mirror `majit-metainterp/src/call_descr.rs::DEFAULT_EFFECT_INFO`
        // (the conservative "all read/write descrs touched" fallback used
        // when the producer has no analyzer output).  `EffectInfo::default()`
        // alone leaves the descr bitsets at zero, which makes
        // `force_from_effectinfo` (`heap.py:540-560`) skip the cached
        // lazy_set / field flush — a regression vs main's plain void
        // residual call invalidation.  `call.py:296-326 getcalldescr` is
        // the upstream constructor; the analyzer port is a separate epic.
        CallFlavor::Plain => EffectInfo {
            extraeffect: ExtraEffect::CanRaise,
            readonly_descrs_fields: u64::MAX,
            write_descrs_fields: u64::MAX,
            readonly_descrs_arrays: u64::MAX,
            write_descrs_arrays: u64::MAX,
            readonly_descrs_interiorfields: u64::MAX,
            write_descrs_interiorfields: u64::MAX,
            ..EffectInfo::default()
        },
        // EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE — `effectinfo.py:23`.
        // `optimize_CALL_MAY_FORCE_*` branch.
        CallFlavor::MayForce => EffectInfo {
            extraeffect: ExtraEffect::ForcesVirtualOrVirtualizable,
            ..EffectInfo::default()
        },
        // EF_LOOPINVARIANT — `effectinfo.py:18`.
        // `optimize_CALL_LOOPINVARIANT_*` branch.
        CallFlavor::LoopInvariant => EffectInfo {
            extraeffect: ExtraEffect::LoopInvariant,
            ..EffectInfo::default()
        },
        // RPython `call.py:292-299 getcalldescr` chooses one of three
        // elidable extraeffects based on the callee's static raise
        // analysis:
        //
        //     elif elidable:
        //         cr = self._canraise(op)
        //         if cr == "mem":
        //             extraeffect = EF_ELIDABLE_OR_MEMORYERROR
        //         elif cr:
        //             extraeffect = EF_ELIDABLE_CAN_RAISE
        //         else:
        //             extraeffect = EF_ELIDABLE_CANNOT_RAISE
        //
        // The three differ in `check_can_raise(False)`:
        // `ElidableCannotRaise (0) < CannotRaise (2)` → false (no
        // GUARD_NO_EXCEPTION),
        // `ElidableOrMemoryError (3) > CannotRaise (2)` → true,
        // `ElidableCanRaise (4) > CannotRaise (2)` → true.  Picking
        // the wrong one drops a guard the trace needs.
        //
        // Pyre has no static-raise analyzer (`randomeffects_analyzer`
        // / `quasiimmut_analyzer` / `_canraise` infrastructure is
        // unported — see `majit-translate` roadmap).  An earlier
        // attempt to default to `ElidableCanRaise` (the conservative
        // variant that satisfies `check_can_raise(False)`) was
        // *itself* a NEW-DEVIATION because pyre's flatten path
        // collapses `CallFlavor::Pure` to `BC_CALL_PURE_*` bytecode
        // (`assembler.rs:1491` `(CallFlavor::Pure, _) => call_pure_*`),
        // and the runtime dispatcher (`majit-metainterp/src/pyjitpl/
        // dispatch.rs:1895 + 2004 + 2115`) consumes
        // `BC_CALL_PURE_*` via `record_result_of_call_pure` without
        // reading the descr's `EffectInfo` — so the
        // `EF_ELIDABLE_CAN_RAISE`'s `GUARD_NO_EXCEPTION` requirement
        // is silently dropped at runtime.  Producing
        // `ElidableCanRaise` here therefore produces a *worse*
        // parity surface than panicking: it looks correct on paper
        // but the guard is never actually emitted, regressing main's
        // fail-fast behaviour.
        //
        // Until the `BC_CALL_PURE_*` runtime sites read the descr's
        // EI and emit `GuardNoException` per
        // `pyjitpl.py:2111 exc = effectinfo.check_can_raise()`, no
        // `CallFlavor::Pure` producer can be honoured correctly.
        // No production `emit_residual_call(_, CallFlavor::Pure, ...)`
        // site exists today (verified: every codewriter
        // `emit_residual_call` call passes `Plain` or `MayForce`),
        // so panicking here forces any future producer to either:
        //   (1) wire `BC_CALL_PURE_*` runtime sites to emit
        //       `GuardNoException` from the descr's EI, then add an
        //       `ElidableCanRaise` arm here; or
        //   (2) construct an `EffectInfo` directly at the call site
        //       with the correct elidable variant and a runtime path
        //       that honours its guard requirement.
        //
        // The reverse mapper `dispatch_kind_for_effect_info` and the
        // dispatcher `dispatch_residual_call` (`assembler.rs:1491-
        // 1505`) keep their `Pure` arms intact so they remain ready
        // to consume a properly-formed elidable EI from such a
        // future producer.
        CallFlavor::Pure => panic!(
            "effect_info_for_call_flavor: CallFlavor::Pure has no production \
             producer in pyre, and the EF_ELIDABLE_* extraeffect cannot be \
             chosen correctly without static raise analysis (call.py:292-299). \
             Defaulting to ElidableCanRaise would silently drop \
             GUARD_NO_EXCEPTION at the BC_CALL_PURE_* runtime sites \
             (dispatch.rs:1895/2004/2115) which do not read the descr's EI. \
             Construct EffectInfo directly with the correct \
             ElidableCannotRaise / ElidableOrMemoryError / ElidableCanRaise \
             variant AND wire BC_CALL_PURE_* dispatch to honour \
             check_can_raise() before routing through this mnemonic."
        ),
        // EF_RANDOM_EFFECTS — `effectinfo.py:24`. Upstream's
        // `call.py:282-289 getcalldescr` upgrades release-gil callees
        // to `EF_RANDOM_EFFECTS` whenever the analyzer flags random
        // effects on the target (and the orthodox case for a
        // `_call_aroundstate_target_`-decorated callee always does, by
        // virtue of the host-call boundary). With
        // `extraeffect = EF_RANDOM_EFFECTS = 7`,
        // `check_forces_virtual_or_virtualizable()` (>= 6) and
        // `has_random_effects()` (>= 7) both return true — matching
        // `pyjitpl.py:2007-2068`'s outer forces branch (which release-gil
        // shares with `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`) where
        // `:2063 effectinfo.is_call_release_gil()` selects the
        // `direct_call_release_gil` sub-case.
        //
        // PRE-EXISTING-ADAPTATION: pyre's producer-side `CallFlavor`
        // does not carry the real `(target_fn_addr, save_err)` pair
        // that `effectinfo.py:114, 197 call_release_gil_target`
        // demands; pyre uses a non-zero sentinel `(1, 0)` solely to
        // make `is_call_release_gil()`
        // (`majit-ir/src/effectinfo.rs:292-295`) return true.  Real
        // target plumbing is deferred to whatever lands the
        // release-gil callee path; no production caller today.
        //
        // The round-trip property
        // `dispatch_kind_for_effect_info(effect_info_for_call_flavor(ReleaseGil)) == ReleaseGil`
        // holds because `dispatch_kind_for_effect_info` checks
        // `is_call_release_gil()` first (`flatten.rs:487`), short-
        // circuiting the `check_forces_virtual_or_virtualizable()`
        // arm which would otherwise return `MayForce`.
        CallFlavor::ReleaseGil => EffectInfo {
            extraeffect: ExtraEffect::RandomEffects,
            // call.py:286 `can_invalidate = random_effects or
            // self.quasiimmut_analyzer.analyze(op)` — `random_effects`
            // implies `can_invalidate`.  Sibling producer
            // `majit-metainterp/src/jitcode/assembler.rs:1821`
            // already sets this to `true`; leaving it `false` here
            // would let the heapcache `clear_caches_varargs` skip the
            // arraycopy / arraymove fast-paths for traces whose
            // release-gil descr came from this producer.
            can_invalidate: true,
            call_release_gil_target: (1, 0),
            ..EffectInfo::default()
        },
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
/// `check_forces_virtual_or_virtualizable()` because
/// [`effect_info_for_call_flavor`] tags `CallFlavor::ReleaseGil`
/// with `EF_RANDOM_EFFECTS` (mirroring `call.py:282-289 getcalldescr`'s
/// `random_effects` upgrade for release-gil callees), which makes
/// `check_forces_virtual_or_virtualizable()` (`>= 6`) also return
/// true on those EI values.  The early `is_call_release_gil()` check
/// keeps the round-trip property
/// `dispatch_kind_for_effect_info(effect_info_for_call_flavor(f)) == f`
/// AND mirrors `pyjitpl.py:2063`'s structure where the release-gil
/// sub-case is selected inside the outer forces branch.
pub fn dispatch_kind_for_effect_info(ei: &majit_ir::EffectInfo) -> CallFlavor {
    use majit_ir::ExtraEffect;
    if ei.is_call_release_gil() {
        return CallFlavor::ReleaseGil;
    }
    if ei.check_forces_virtual_or_virtualizable() {
        return CallFlavor::MayForce;
    }
    if ei.extraeffect == ExtraEffect::LoopInvariant {
        return CallFlavor::LoopInvariant;
    }
    if ei.check_is_elidable() {
        return CallFlavor::Pure;
    }
    CallFlavor::Plain
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
    /// Plain residual call with no extra effect
    /// (`rpython/jit/codewriter/effectinfo.py EF_CANNOT_RAISE` /
    /// `EF_CAN_RAISE`).
    Plain,
    /// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`. The builder emits
    /// `call_may_force_*` so the metainterp forces virtualizable state
    /// before the call. Maps to `JitCodeBuilder::call_may_force_*_typed`.
    MayForce,
    /// `EF_LOOPINVARIANT`. One-shot call memoised across the trace loop.
    /// Maps to `JitCodeBuilder::call_loopinvariant_*_typed`.
    LoopInvariant,
    /// `EF_RELEASES_GIL`. Maps to `JitCodeBuilder::call_release_gil_*_typed`.
    ReleaseGil,
    /// `EF_ELIDABLE_*`. Maps to `JitCodeBuilder::call_pure_*_typed`.
    Pure,
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
            (
                target,
                link_borrow
                    .args
                    .iter()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>(),
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
        let should_reraise = {
            let link_borrow = link.borrow();
            let Some(last_exception) = link_borrow.last_exception else {
                panic!("make_exception_link requires last_exception");
            };
            let Some(last_exc_value) = link_borrow.last_exc_value else {
                panic!("make_exception_link requires last_exc_value");
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
            if !self.include_all_exc_links && block.borrow().raising_op().is_none() {
                self.make_link(&exits[0], false);
                return;
            }
            let catch_label = self.tlabel_for_link(&exits[0]);
            self.emitline(Insn::op("catch_exception", vec![catch_label]));
            self.make_link(&exits[0], false);
            let normal_label = self.label_for_link(&exits[0]);
            self.emitline(normal_label);
            let mut captured_all = false;
            for link in exits.iter().skip(1) {
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
            panic!(
                "flatten_graph: unsupported exits shape for block with {} exits",
                exits.len()
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
fn flatten_constant_operand_for_probe(constant: &super::flow::Constant) -> Operand {
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
    match arg {
        super::flow::SpaceOperationArg::Value(FlowValue::Variable(v)) => {
            Operand::Register(get_register(*v))
        }
        super::flow::SpaceOperationArg::Value(FlowValue::Constant(c)) => {
            flatten_constant_operand_for_probe(c)
        }
        super::flow::SpaceOperationArg::ListOfKind(list) => Operand::ListOfKind(ListOfKind::new(
            list.kind,
            list.content
                .iter()
                .map(|value| match value {
                    FlowValue::Variable(v) => Operand::Register(get_register(*v)),
                    FlowValue::Constant(c) => flatten_constant_operand_for_probe(c),
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
/// `DescrOperand` by `Arc::ptr_eq` against the singleton accessors in
/// `majit_ir::descr`.  Today recognises the vable array_field /
/// array / static_field singletons emitted by `record_graph_op` for
/// vable get/setfield + get/setarrayitem ops
/// (`jtransform.py:846-927`, `:1880-1906`).  Other concrete descr
/// flavors (`Bh`, `SwitchDict`, `CallDescrStub`) are constructed
/// directly at the SSARepr-emit site rather than going through
/// `SpaceOperationArg::Descr`, so this fn rejects them.  Adding a
/// new graph-side descr producer must extend the singleton list.
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
    panic!(
        "flatten_descr_by_ptr: unmapped DescrByPtr {} — only vable \
         array_field / array / static_field singletons are \
         recognised today",
        descr_ref.repr()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::flow::{FlowListOfKind, VariableId};

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
}
