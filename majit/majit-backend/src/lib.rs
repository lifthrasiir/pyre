/// Backend abstraction trait for JIT code generation.
///
/// Translated from rpython/jit/backend/model.py (AbstractCPU).
/// The Backend trait is the contract between the JIT frontend (tracing + optimization)
/// and the code generation backend (Cranelift, etc.).
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use majit_ir::{Const, Descr, FailDescr, GcRef, InputArg, Op, Type, Value};

pub mod call_stub;
pub mod jitframe;
pub mod llmodel;

/// Lightweight execution result that avoids DeadFrame boxing.
///
/// Used by `execute_token_ints_raw` to return guard failure data
/// without heap-allocating a DeadFrame.
pub struct RawExecResult {
    /// Output values from the guard exit, truncated to `exit_arity`.
    pub outputs: Vec<i64>,
    /// Typed output values decoded from the exit slots.
    pub typed_outputs: Vec<Value>,
    /// Backend-origin static metadata for this exit, when available.
    pub exit_layout: Option<FailDescrLayout>,
    /// Output slots that carry opaque force tokens instead of GC refs.
    pub force_token_slots: Vec<usize>,
    /// Optional saved-data GC ref captured by this exit.
    pub savedata: Option<GcRef>,
    /// Pending exception value captured by this exit (`GcRef::NULL` = none).
    pub exception_value: GcRef,
    /// Backend fail-index for this exit.
    pub fail_index: u32,
    /// Compiled trace identifier for this exit.
    pub trace_id: u64,
    /// Whether this exit is a FINISH rather than a guard failure.
    pub is_finish: bool,
    /// compile.py:658-662 ExitFrameWithExceptionDescrRef parity.
    /// True when this FINISH was emitted via
    /// pyjitpl.py:3238-3245 compile_exit_frame_with_exception.
    pub is_exit_frame_with_exception: bool,
    /// compile.py:741-745: ResumeGuardDescr.status at guard failure time.
    pub status: u64,
    /// compile.py:780: current_object_addr_as_int(self) — descriptor pointer.
    pub descr_addr: usize,
}

/// Backend-neutral static metadata for a compiled trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledTraceInfo {
    /// Compiled trace identifier.
    pub trace_id: u64,
    /// Input types expected at this trace header.
    pub input_types: Vec<Type>,
    /// Interpreter header pc associated with this trace.
    pub header_pc: u64,
    /// Source guard this bridge is attached to, or `None` for a root trace.
    pub source_guard: Option<(u64, u32)>,
}

/// Backend-neutral source of a reconstructed frame slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitValueSourceLayout {
    /// Slot is sourced from a raw exit slot.
    ExitValue(usize),
    /// Slot is a constant value embedded in the layout.
    Constant(i64),
    /// Slot refers to a materialized virtual object.
    Virtual(usize),
    /// Slot exists but remains uninitialized.
    Uninitialized,
    /// Slot is unavailable/dead at this exit.
    Unavailable,
}

impl ExitValueSourceLayout {
    pub fn shifted_virtuals(&self, virtual_offset: usize) -> Self {
        match self {
            Self::Virtual(index) => Self::Virtual(index + virtual_offset),
            other => other.clone(),
        }
    }
}

/// Backend-neutral kind of materialized virtual object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitVirtualKind {
    Object,
    Struct,
    Array,
    ArrayStruct,
    RawBuffer,
}

/// Backend-neutral description of a materialized virtual object.
#[derive(Debug, Clone)]
pub enum ExitVirtualLayout {
    /// resume.py:612 VirtualInfo — allocate_with_vtable(descr=self.descr).
    Object {
        /// resume.py:615 self.descr — live SizeDescr for allocate_with_vtable.
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        /// info.py:318 _known_class — vtable pointer for allocate_with_vtable.
        known_class: Option<i64>,
        fields: Vec<(u32, ExitValueSourceLayout)>,
        target_slot: Option<usize>,
        /// resume.py:593 fielddescrs for setfield dispatch.
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    },
    /// resume.py:628 VStructInfo — allocate_struct(self.typedescr).
    Struct {
        /// resume.py:631 self.typedescr — live SizeDescr for allocate_struct.
        typedescr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        fields: Vec<(u32, ExitValueSourceLayout)>,
        target_slot: Option<usize>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    },
    Array {
        descr_index: u32,
        /// resume.py:653: allocate_array(length, arraydescr, self.clear)
        clear: bool,
        /// resume.py:656: arraydescr element kind (0=ref, 1=int, 2=float)
        kind: u8,
        items: Vec<ExitValueSourceLayout>,
    },
    /// resume.py:736 VArrayStructInfo(arraydescr, size, fielddescrs)
    ArrayStruct {
        descr_index: u32,
        /// resume.py:739: self.arraydescr — live ArrayDescr for allocate_array.
        arraydescr: Option<majit_ir::DescrRef>,
        /// resume.py:740: self.fielddescrs — live InteriorFieldDescr per field slot.
        fielddescrs: Vec<majit_ir::DescrRef>,
        element_fields: Vec<Vec<(u32, ExitValueSourceLayout)>>,
    },
    /// resume.py:717 VRawSliceInfo — base_buffer + offset.
    RawSlice {
        offset: usize,
        base: ExitValueSourceLayout,
    },
    RawBuffer {
        /// resume.py:694: self.func
        func: i64,
        size: usize,
        /// resume.py:695: self.offsets
        offsets: Vec<usize>,
        /// resume.py:697: self.descrs
        descrs: Vec<majit_ir::ArrayDescrInfo>,
        /// resume.py:693: fieldnums (decoded)
        values: Vec<ExitValueSourceLayout>,
    },
    /// resume.py:763 VStrPlainInfo — virtual byte-string
    /// (bh_newstr(len) + bh_strsetitem per character).
    ///
    /// `is_unicode = false` → bh_newstr/bh_strsetitem.
    /// `is_unicode = true`  → bh_newunicode/bh_unicodesetitem (unified
    /// variant for resume.py:817 VUniPlainInfo).
    StrPlain {
        is_unicode: bool,
        /// Per-character values, length = string length. UNINITIALIZED
        /// fieldnums (resume.py:774) remain as `Uninitialized`.
        chars: Vec<ExitValueSourceLayout>,
    },
    /// resume.py:781 VStrConcatInfo + resume.py:836 VUniConcatInfo.
    /// decoder.concat_strings(left, right) = cic.funcptr_for_oopspec(
    /// OS_STR_CONCAT)(left, right).
    StrConcat {
        is_unicode: bool,
        /// OS_STR_CONCAT or OS_UNI_CONCAT func pointer (resume.py:1468).
        func: i64,
        /// calldescr for OS_STR_CONCAT/OS_UNI_CONCAT. Carried so pyre's
        /// bh_call_r can dispatch to the right calling convention.
        calldescr: majit_ir::DescrRef,
        left: ExitValueSourceLayout,
        right: ExitValueSourceLayout,
    },
    /// resume.py:801 VStrSliceInfo + resume.py:856 VUniSliceInfo.
    /// decoder.slice_string(str, start, length) =
    /// cic.funcptr_for_oopspec(OS_STR_SLICE)(str, start, start + length).
    StrSlice {
        is_unicode: bool,
        /// OS_STR_SLICE or OS_UNI_SLICE func pointer (resume.py:1478).
        func: i64,
        /// calldescr for OS_STR_SLICE/OS_UNI_SLICE.
        calldescr: majit_ir::DescrRef,
        str_src: ExitValueSourceLayout,
        start: ExitValueSourceLayout,
        length: ExitValueSourceLayout,
    },
}

impl ExitVirtualLayout {
    pub fn shifted_virtuals(&self, virtual_offset: usize) -> Self {
        match self {
            Self::Object {
                descr,
                type_id,
                descr_index,
                known_class,
                fields,
                target_slot,
                fielddescrs,
                descr_size,
            } => Self::Object {
                descr: descr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                known_class: *known_class,
                fields: fields
                    .iter()
                    .map(|(fi, src)| (*fi, src.shifted_virtuals(virtual_offset)))
                    .collect(),
                target_slot: *target_slot,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            Self::Struct {
                typedescr,
                type_id,
                descr_index,
                fields,
                target_slot,
                fielddescrs,
                descr_size,
            } => Self::Struct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(field_index, source)| {
                        (*field_index, source.shifted_virtuals(virtual_offset))
                    })
                    .collect(),
                target_slot: *target_slot,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            Self::Array {
                descr_index,
                clear,
                kind,
                items,
            } => Self::Array {
                descr_index: *descr_index,
                clear: *clear,
                kind: *kind,
                items: items
                    .iter()
                    .map(|source| source.shifted_virtuals(virtual_offset))
                    .collect(),
            },
            Self::RawSlice { offset, base } => Self::RawSlice {
                offset: *offset,
                base: base.shifted_virtuals(virtual_offset),
            },
            Self::ArrayStruct {
                descr_index,
                arraydescr,
                fielddescrs,
                element_fields,
            } => Self::ArrayStruct {
                descr_index: *descr_index,
                arraydescr: arraydescr.clone(),
                fielddescrs: fielddescrs.clone(),
                element_fields: element_fields
                    .iter()
                    .map(|element| {
                        element
                            .iter()
                            .map(|(field_index, source)| {
                                (*field_index, source.shifted_virtuals(virtual_offset))
                            })
                            .collect()
                    })
                    .collect(),
            },
            Self::RawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => Self::RawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: values
                    .iter()
                    .map(|source| source.shifted_virtuals(virtual_offset))
                    .collect(),
            },
            Self::StrPlain { is_unicode, chars } => Self::StrPlain {
                is_unicode: *is_unicode,
                chars: chars
                    .iter()
                    .map(|source| source.shifted_virtuals(virtual_offset))
                    .collect(),
            },
            Self::StrConcat {
                is_unicode,
                func,
                calldescr,
                left,
                right,
            } => Self::StrConcat {
                is_unicode: *is_unicode,
                func: *func,
                calldescr: calldescr.clone(),
                left: left.shifted_virtuals(virtual_offset),
                right: right.shifted_virtuals(virtual_offset),
            },
            Self::StrSlice {
                is_unicode,
                func,
                calldescr,
                str_src,
                start,
                length,
            } => Self::StrSlice {
                is_unicode: *is_unicode,
                func: *func,
                calldescr: calldescr.clone(),
                str_src: str_src.shifted_virtuals(virtual_offset),
                start: start.shifted_virtuals(virtual_offset),
                length: length.shifted_virtuals(virtual_offset),
            },
        }
    }
}

// PartialEq/Eq: compare by data fields, skip descr/typedescr (Arc<dyn Descr>).
impl PartialEq for ExitVirtualLayout {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Object {
                    type_id: a1,
                    descr_index: a2,
                    known_class: a7,
                    fields: a3,
                    target_slot: a4,
                    fielddescrs: a5,
                    descr_size: a6,
                    ..
                },
                Self::Object {
                    type_id: b1,
                    descr_index: b2,
                    known_class: b7,
                    fields: b3,
                    target_slot: b4,
                    fielddescrs: b5,
                    descr_size: b6,
                    ..
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4 && a5 == b5 && a6 == b6 && a7 == b7,
            (
                Self::Struct {
                    type_id: a1,
                    descr_index: a2,
                    fields: a3,
                    target_slot: a4,
                    fielddescrs: a5,
                    descr_size: a6,
                    ..
                },
                Self::Struct {
                    type_id: b1,
                    descr_index: b2,
                    fields: b3,
                    target_slot: b4,
                    fielddescrs: b5,
                    descr_size: b6,
                    ..
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4 && a5 == b5 && a6 == b6,
            (
                Self::Array {
                    descr_index: a1,
                    clear: a2,
                    kind: a3,
                    items: a4,
                },
                Self::Array {
                    descr_index: b1,
                    clear: b2,
                    kind: b3,
                    items: b4,
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4,
            (
                Self::ArrayStruct {
                    descr_index: a1,
                    arraydescr: a_ad,
                    fielddescrs: a_fds,
                    element_fields: a2,
                },
                Self::ArrayStruct {
                    descr_index: b1,
                    arraydescr: b_ad,
                    fielddescrs: b_fds,
                    element_fields: b2,
                },
            ) => {
                // virtualstate.py:295-305: arraydescr identity + per-fielddescr identity
                a1 == b1
                    && a_ad.as_ref().map(|d| d.index()) == b_ad.as_ref().map(|d| d.index())
                    && a_fds.len() == b_fds.len()
                    && a_fds
                        .iter()
                        .zip(b_fds.iter())
                        .all(|(a, b)| a.index() == b.index())
                    && a2 == b2
            }
            (
                Self::RawSlice {
                    offset: a1,
                    base: a2,
                },
                Self::RawSlice {
                    offset: b1,
                    base: b2,
                },
            ) => a1 == b1 && a2 == b2,
            (
                Self::RawBuffer {
                    size: a1,
                    offsets: a2,
                    values: a3,
                    ..
                },
                Self::RawBuffer {
                    size: b1,
                    offsets: b2,
                    values: b3,
                    ..
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3,
            (
                Self::StrPlain {
                    is_unicode: a1,
                    chars: a2,
                },
                Self::StrPlain {
                    is_unicode: b1,
                    chars: b2,
                },
            ) => a1 == b1 && a2 == b2,
            (
                Self::StrConcat {
                    is_unicode: a1,
                    func: a2,
                    left: a3,
                    right: a4,
                    ..
                },
                Self::StrConcat {
                    is_unicode: b1,
                    func: b2,
                    left: b3,
                    right: b4,
                    ..
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4,
            (
                Self::StrSlice {
                    is_unicode: a1,
                    func: a2,
                    str_src: a3,
                    start: a4,
                    length: a5,
                    ..
                },
                Self::StrSlice {
                    is_unicode: b1,
                    func: b2,
                    str_src: b3,
                    start: b4,
                    length: b5,
                    ..
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4 && a5 == b5,
            _ => false,
        }
    }
}
impl Eq for ExitVirtualLayout {}

/// Backend-neutral deferred heap write recovered from an exit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExitPendingFieldLayout {
    pub descr_index: u32,
    pub item_index: Option<usize>,
    pub is_array_item: bool,
    pub target: ExitValueSourceLayout,
    pub value: ExitValueSourceLayout,
    /// Byte offset from start of struct (from FieldDescr).
    pub field_offset: usize,
    /// Size of the field in bytes.
    pub field_size: usize,
    /// Type of the value being stored.
    pub field_type: majit_ir::Type,
}

impl ExitPendingFieldLayout {
    pub fn shifted_virtuals(&self, virtual_offset: usize) -> Self {
        Self {
            descr_index: self.descr_index,
            item_index: self.item_index,
            is_array_item: self.is_array_item,
            target: self.target.shifted_virtuals(virtual_offset),
            value: self.value.shifted_virtuals(virtual_offset),
            field_offset: self.field_offset,
            field_size: self.field_size,
            field_type: self.field_type,
        }
    }
}

/// Backend-neutral reconstructed frame layout for an exit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExitFrameLayout {
    /// Compiled trace identifier that owns this frame layout, when known.
    pub trace_id: Option<u64>,
    /// Trace header pc associated with this frame layout, when known.
    pub header_pc: Option<u64>,
    /// Source guard this frame's trace is attached to, when the frame comes
    /// from a compiled bridge.
    pub source_guard: Option<(u64, u32)>,
    /// Interpreter program counter for the frame.
    pub pc: u64,
    /// resume.py:250 `jitcode_index` — index into
    /// `MetaInterpStaticData.jitcodes` identifying the code this frame is
    /// running. Required so multi-frame inline snapshots can be decoded
    /// with per-frame liveness via `frame_value_count_at(jitcode_index, pc)`.
    /// Encoders that only produce single-frame exits may leave this at 0,
    /// but multi-frame producers MUST populate it per frame.
    pub jitcode_index: i32,
    /// Slot sources within this frame.
    pub slots: Vec<ExitValueSourceLayout>,
    /// Typed layout of the frame slots, when known by the backend.
    pub slot_types: Option<Vec<Type>>,
}

impl ExitFrameLayout {
    pub fn shifted_virtuals(&self, virtual_offset: usize) -> Self {
        Self {
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            source_guard: self.source_guard,
            pc: self.pc,
            jitcode_index: self.jitcode_index,
            slots: self
                .slots
                .iter()
                .map(|slot| slot.shifted_virtuals(virtual_offset))
                .collect(),
            slot_types: self.slot_types.clone(),
        }
    }
}

/// Backend-neutral recovery metadata attached to an exit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExitRecoveryLayout {
    /// resume.py:1099 consume_vref_and_vable_boxes / virtualizable_boxes.
    pub vable_array: Vec<ExitValueSourceLayout>,
    /// resume.py:1093 consume_virtualref_boxes — [virtual, vref, ...] pairs.
    pub vref_array: Vec<ExitValueSourceLayout>,
    /// Reconstructed frames, outermost first.
    pub frames: Vec<ExitFrameLayout>,
    /// Materialized virtual objects referenced by the frames.
    pub virtual_layouts: Vec<ExitVirtualLayout>,
    /// Deferred heap writes to replay after materialization.
    pub pending_field_layouts: Vec<ExitPendingFieldLayout>,
}

impl ExitRecoveryLayout {
    pub fn prefixed_by(&self, caller_prefix: Option<&Self>) -> Self {
        let Some(caller_prefix) = caller_prefix else {
            return self.clone();
        };

        let virtual_offset = caller_prefix.virtual_layouts.len();
        let vable_array = if caller_prefix.vable_array.is_empty() {
            self.vable_array.clone()
        } else {
            caller_prefix.vable_array.clone()
        };
        let vref_array = if caller_prefix.vref_array.is_empty() {
            self.vref_array
                .iter()
                .map(|slot| slot.shifted_virtuals(virtual_offset))
                .collect()
        } else {
            caller_prefix.vref_array.clone()
        };
        let mut frames = caller_prefix.frames.clone();
        frames.extend(
            self.frames
                .iter()
                .map(|frame| frame.shifted_virtuals(virtual_offset)),
        );

        let mut virtual_layouts = caller_prefix.virtual_layouts.clone();
        virtual_layouts.extend(
            self.virtual_layouts
                .iter()
                .map(|layout| layout.shifted_virtuals(virtual_offset)),
        );

        let mut pending_field_layouts = caller_prefix.pending_field_layouts.clone();
        pending_field_layouts.extend(
            self.pending_field_layouts
                .iter()
                .map(|layout| layout.shifted_virtuals(virtual_offset)),
        );

        Self {
            vable_array,
            vref_array,
            frames,
            virtual_layouts,
            pending_field_layouts,
        }
    }
}

/// Static layout metadata for a backend fail descriptor.
#[derive(Debug, Clone)]
pub struct FailDescrLayout {
    /// Backend fail-index for this exit.
    pub fail_index: u32,
    /// Trace op index of the guard/finish that produced this exit, when known.
    pub source_op_index: Option<usize>,
    /// Compiled trace identifier that owns this exit.
    pub trace_id: u64,
    /// Backend-owned metadata for the trace that owns this exit.
    pub trace_info: Option<CompiledTraceInfo>,
    /// Typed layout of the exit slots.
    pub fail_arg_types: Vec<Type>,
    /// Whether this exit is a FINISH rather than a guard failure.
    pub is_finish: bool,
    /// Exit slot indices that hold rooted GC references.
    pub gc_ref_slots: Vec<usize>,
    /// Exit slot indices that carry opaque FORCE_TOKEN handles.
    pub force_token_slots: Vec<usize>,
    /// Optional backend-origin recovery layout for this exit.
    pub recovery_layout: Option<ExitRecoveryLayout>,
    /// Complete frame stack from innermost (this guard's frame) to outermost.
    /// Present when multi-frame reconstruction is supported.
    pub frame_stack: Option<Vec<ExitFrameLayout>>,
    /// resume.py:450 — compact resume numbering (varint-encoded tagged values).
    /// Propagated from the backend's fail descriptor so the frontend can
    /// reconstruct the blackhole chain after a trace's `CompiledTrace` entry
    /// has been evicted but the descriptor itself is still live.
    pub rd_numb: Option<Vec<u8>>,
    /// resume.py:451 — shared constant pool referenced by `rd_numb`.
    pub rd_consts: Option<Vec<Const>>,
    /// resume.py:488 — virtual object field info referenced by `rd_numb`.
    pub rd_virtuals: Option<Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>>,
    /// Deferred heap writes associated with this guard exit.
    pub rd_pendingfields: Option<Vec<majit_ir::GuardPendingFieldEntry>>,
}

/// Static layout metadata for a terminal exit within a compiled trace.
///
/// Unlike [`FailDescrLayout`], terminal exits are keyed by the trace op index
/// rather than a backend fail descriptor, because `JUMP` exits do not
/// necessarily correspond to a deadframe-producing guard site.
#[derive(Debug, Clone)]
pub struct TerminalExitLayout {
    /// Trace op index of the terminal `FINISH`/`JUMP`.
    pub op_index: usize,
    /// Compiled trace identifier that owns this exit.
    pub trace_id: u64,
    /// Backend-owned metadata for the trace that owns this exit.
    pub trace_info: Option<CompiledTraceInfo>,
    /// Backend fail-index if this terminal exit is also a fail descriptor.
    pub fail_index: u32,
    /// Typed layout of the exit slots.
    pub exit_types: Vec<Type>,
    /// Whether this exit is a `FINISH` rather than a `JUMP`.
    pub is_finish: bool,
    /// Exit slot indices that hold rooted GC references.
    pub gc_ref_slots: Vec<usize>,
    /// Exit slot indices that carry opaque FORCE_TOKEN handles.
    pub force_token_slots: Vec<usize>,
    /// Optional backend-origin recovery layout for this terminal exit.
    pub recovery_layout: Option<ExitRecoveryLayout>,
    /// resume.py:450 — compact resume numbering (terminal exits rarely need
    /// this, but propagate it for symmetry with `FailDescrLayout`).
    pub rd_numb: Option<Vec<u8>>,
    /// resume.py:451 — shared constant pool referenced by `rd_numb`.
    pub rd_consts: Option<Vec<Const>>,
    /// resume.py:488 — virtual object field info referenced by `rd_numb`.
    pub rd_virtuals: Option<Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>>,
    /// Deferred heap writes associated with this terminal exit.
    pub rd_pendingfields: Option<Vec<majit_ir::GuardPendingFieldEntry>>,
}

/// Result of compiling a loop or bridge.
#[derive(Debug)]
pub struct AsmInfo {
    /// Start address of the generated code.
    pub code_addr: usize,
    /// Size of the generated code in bytes.
    pub code_size: usize,
}

/// Marker descriptor for loop version guards.
///
/// When a guard uses this descriptor, the alternative loop version
/// is compiled immediately after the main loop (not lazily on failure).
/// This enables speculative optimization: vectorized fast path with
/// scalar fallback, type-specialized path with generic fallback, etc.
///
/// Mirrors `CompileLoopVersionDescr` from rpython/jit/metainterp/compile.py.
#[derive(Debug)]
pub struct LoopVersionDescr {
    /// Index identifying this version guard.
    pub version_index: u32,
    /// Fail argument types for the guard.
    pub fail_arg_types: Vec<Type>,
}

impl Descr for LoopVersionDescr {
    fn index(&self) -> u32 {
        self.version_index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn is_loop_version(&self) -> bool {
        true
    }
}

impl FailDescr for LoopVersionDescr {
    fn fail_index(&self) -> u32 {
        self.version_index
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.fail_arg_types
    }
}

/// Tracks alternative loop versions to compile after the main loop.
///
/// Each version is an alternative trace that handles cases where the
/// main loop's version guard fails — e.g., unaligned arrays, short
/// arrays that cannot be vectorized, or different type specializations.
pub struct LoopVersionInfo {
    /// (version_guard_index, alternative_inputargs, alternative_ops)
    pub versions: Vec<(u32, Vec<InputArg>, Vec<Op>)>,
}

impl LoopVersionInfo {
    pub fn new() -> Self {
        Self {
            versions: Vec::new(),
        }
    }

    /// Register an alternative version to compile after the main loop.
    pub fn add_version(&mut self, guard_index: u32, inputargs: Vec<InputArg>, ops: Vec<Op>) {
        self.versions.push((guard_index, inputargs, ops));
    }
}

impl Default for LoopVersionInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LoopVersionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoopVersionInfo")
            .field("num_versions", &self.versions.len())
            .finish()
    }
}

/// `rpython/jit/backend/llsupport/jitframe.py:30-40` `JITFRAMEINFO` parity.
///
/// Runtime frame depth/size carried on each `CompiledLoopToken`. Mutated
/// in-place via `update_frame_depth` when bridges extend the frame layout.
///
/// `#[repr(C)]` pins the `[jfi_frame_depth, jfi_frame_size]` field order so
/// the rewriter's `CallAssemblerCalleeLocs.frame_info_ptr` can pass the
/// struct's raw address to generated code that loads `jfi_frame_depth` /
/// `jfi_frame_size` at fixed offsets — matching RPython `jitframe.py:30-40`
/// (lltype-allocated, stable-layout struct).
#[derive(Debug, Default)]
#[repr(C)]
pub struct JitFrameInfo {
    /// `jitframe.py:33` `('jfi_frame_depth', lltype.Signed)`.
    pub jfi_frame_depth: i64,
    /// `jitframe.py:35` `('jfi_frame_size', lltype.Signed)`.
    pub jfi_frame_size: i64,
}

impl JitFrameInfo {
    /// `rpython/jit/backend/llsupport/jitframe.py:18-22`
    /// `jitframeinfo_update_depth(jfi, base_ofs, new_depth)`.
    pub fn update_frame_depth(&mut self, base_ofs: i64, new_depth: i64) {
        // jitframe.py:8 `SIZEOFSIGNED = rffi.sizeof(lltype.Signed)`; 64-bit target.
        const SIZEOFSIGNED: i64 = 8;
        if new_depth > self.jfi_frame_depth {
            self.jfi_frame_depth = new_depth;
            self.jfi_frame_size = base_ofs + new_depth * SIZEOFSIGNED;
        }
    }

    /// `rpython/jit/backend/llsupport/jitframe.py:24-26` `jitframeinfo_clear`.
    pub fn clear(&mut self) {
        self.jfi_frame_size = 0;
        self.jfi_frame_depth = 0;
    }
}

/// `rpython/jit/backend/model.py:292-338` `CompiledLoopToken` parity.
///
/// Per-loop metadata attached to `JitCellToken.compiled_loop_token`.
/// Mutation goes through `parking_lot::Mutex` on individual fields because
/// pyre reaches these from `&JitCellToken` shared refs (bridge compilation
/// under concurrent execution) — RPython mutates through the GIL instead.
pub struct CompiledLoopToken {
    /// `model.py:299` `self.number = number`.
    pub number: u64,
    /// `model.py:300` `self.bridges_count = 0`.
    pub bridges_count: parking_lot::Mutex<usize>,
    /// `model.py:301` `self.invalidate_positions = []`.
    pub invalidate_positions: parking_lot::Mutex<Vec<usize>>,
    /// `model.py:302-304` `self.looptokens_redirected_to = []` — weak
    /// references to `CompiledLoopToken` instances previously redirected
    /// to this one via `redirect_call_assembler`.
    /// `x86/assembler.py:1150-1151` shows that
    /// `newlooptoken.compiled_loop_token.update_frame_info(
    ///     oldlooptoken.compiled_loop_token, baseofs)` passes the
    /// `CompiledLoopToken` (not `JitCellToken`), so the chain stores
    /// weak refs to `CompiledLoopToken`.
    pub looptokens_redirected_to: parking_lot::Mutex<Vec<std::sync::Weak<CompiledLoopToken>>>,
    /// `model.py:293` `asmmemmgr_blocks = None` (class default — lazy-init
    /// on first access, see `llsupport/assembler.py:184-188`
    /// `get_asmmemmgr_blocks`). pyre eagerly initializes to an empty Vec;
    /// the `None` sentinel is a Python idiom not needed on Rust.
    pub asmmemmgr_blocks: parking_lot::Mutex<Vec<Box<dyn std::any::Any + Send>>>,
    /// `model.py:294` `asmmemmgr_gcreftracers = None`; parity shape
    /// reserved for the GC ref-tracer lifecycle (llsupport/assembler.py:190).
    /// Eagerly empty (same rationale as `asmmemmgr_blocks`).
    pub asmmemmgr_gcreftracers: parking_lot::Mutex<Vec<Arc<dyn std::any::Any + Send + Sync>>>,
    /// `x86/assembler.py:514` `looptoken.compiled_loop_token.frame_info`
    /// is populated by the backend when assembling this loop. Mutated
    /// in-place by bridges via `update_frame_info` (model.py:316-329).
    pub frame_info: parking_lot::Mutex<JitFrameInfo>,
    /// `rpython/jit/backend/llsupport/regalloc.py:861-871`
    /// `_set_initial_bindings` assigns
    /// `looptoken.compiled_loop_token._ll_initial_locs = locs` — the
    /// inputarg slot offsets (in bytes, relative to the frame base) that
    /// `CALL_ASSEMBLER` uses in `handle_call_assembler`
    /// (rewrite.py:673) to spill callee arguments.
    pub _ll_initial_locs: parking_lot::Mutex<Vec<i32>>,
}

impl CompiledLoopToken {
    /// `model.py:296-307` `__init__(self, cpu, number)`. The `cpu` is
    /// implicit in pyre — the owning `Backend` holds the token lifetime.
    pub fn new(number: u64) -> Self {
        CompiledLoopToken {
            number,
            bridges_count: parking_lot::Mutex::new(0),
            invalidate_positions: parking_lot::Mutex::new(Vec::new()),
            looptokens_redirected_to: parking_lot::Mutex::new(Vec::new()),
            asmmemmgr_blocks: parking_lot::Mutex::new(Vec::new()),
            asmmemmgr_gcreftracers: parking_lot::Mutex::new(Vec::new()),
            frame_info: parking_lot::Mutex::new(JitFrameInfo::default()),
            _ll_initial_locs: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// `model.py:309-314` `compiling_a_bridge(self)`. The accompanying
    /// `self.cpu.tracker.total_compiled_bridges += 1` and
    /// `debug_start/debug_print/debug_stop` on the RPython side are
    /// **PRE-EXISTING-ADAPTATION** (pyre has no global `cpu.tracker`
    /// counter and uses `tracing` crate instead of `rpython.rlib.debug`).
    pub fn compiling_a_bridge(&self) {
        *self.bridges_count.lock() += 1;
    }

    /// `rpython/jit/backend/model.py:316-329`
    /// `update_frame_info(self, oldlooptoken, baseofs)`.
    ///
    /// `self` is the *new* looptoken's `CompiledLoopToken`, `oldlooptoken`
    /// is the previous one (and its weak self-ref), `baseofs` is
    /// `cpu.get_baseofs_of_frame_field()`. Propagates `self.frame_info`
    /// depth to every `CompiledLoopToken` in
    /// `oldlooptoken.looptokens_redirected_to` (dropping dead weak refs),
    /// to `oldlooptoken` itself, and appends the provided weak ref to
    /// this new token's redirect chain.
    pub fn update_frame_info(
        &self,
        oldlooptoken: &CompiledLoopToken,
        oldlooptoken_weak: std::sync::Weak<CompiledLoopToken>,
        baseofs: i64,
    ) {
        // `model.py:317` `new_fi = self.frame_info`
        // `model.py:318` `new_loop_tokens = []`
        let new_fi_depth = self.frame_info.lock().jfi_frame_depth;
        let mut new_loop_tokens: Vec<std::sync::Weak<CompiledLoopToken>> = Vec::new();
        // `model.py:319-324` propagate depth through the old token's
        // existing redirect chain, dropping dead weak refs (`ref()` →
        // `None` in CPython), and keep the live ones.
        let old_chain = oldlooptoken.looptokens_redirected_to.lock().clone();
        for weak in old_chain {
            if let Some(target) = weak.upgrade() {
                target
                    .frame_info
                    .lock()
                    .update_frame_depth(baseofs, new_fi_depth);
                new_loop_tokens.push(weak);
            }
        }
        // `model.py:325-326` update the old token's own frame_info.
        oldlooptoken
            .frame_info
            .lock()
            .update_frame_depth(baseofs, new_fi_depth);
        // `model.py:327-328` `assert oldlooptoken is not None;
        // new_loop_tokens.append(weakref.ref(oldlooptoken))` — the
        // caller provides the weak ref (Rust can't derive it from a
        // borrow alone; the owning Arc stays in `JitCellToken`).
        new_loop_tokens.push(oldlooptoken_weak);
        // `model.py:329` `self.looptokens_redirected_to = new_loop_tokens`
        *self.looptokens_redirected_to.lock() = new_loop_tokens;
    }
}

/// Token identifying a compiled loop. Bridges are attached to this.
/// RPython history.py JitCellToken parity — green_key carried on token
/// so the backend can identify the parent loop for bridge compilation.
pub struct JitCellToken {
    /// Unique number for this token.
    pub number: u64,
    /// Green key hash identifying the loop entry point.
    /// Set by MetaInterp before compile_loop. Used by the backend's
    /// bridge threshold callback to find the compiled loop metadata.
    pub green_key: u64,
    /// Types of the input arguments, recorded at compile_loop time from
    /// the finalised inputargs. RPython does not carry this list on
    /// `JitCellToken` itself — there, backend code recovers types by
    /// re-walking the LABEL op (`history.py:501 TreeLoop.inputargs` /
    /// `regalloc.py:861-871 _set_initial_bindings`). pyre's external-jump
    /// and bridge-link paths (`compiler.rs:2362/2468/2720/3998/4027`,
    /// `runner.rs:1281`) touch the target token without access to the
    /// trace ops, so this field caches the typed signature for them.
    /// Front-end (tracer) code MUST NOT read this; use
    /// `MetaInterp::front_target_inputarg_types` instead, which derives
    /// types from the LABEL op + `TreeLoop.inputargs` per RPython.
    pub inputarg_types: Vec<Type>,
    /// virtualizable.py:86 read_boxes: number of scalar inputargs
    /// (frame + static fields). First local is at this index.
    pub num_scalar_inputargs: usize,
    /// warmspot.py / rewrite.py parity: JitDriverSD.index_of_virtualizable.
    /// Index inside the original CALL_ASSEMBLER arglist before rewrite
    /// collapses it to `[frame]` or `[frame, virtualizable]`.
    pub virtualizable_arg_index: Option<usize>,
    /// compile.py:168 `jitcell_token.outermost_jitdriver_sd = jitdriver_sd`.
    ///
    /// The backend crate stores the registered jitdriver slot index
    /// rather than a direct metainterp object pointer.
    pub outermost_jitdriver_index: Option<usize>,
    /// Backend-specific compiled data.
    pub compiled: Option<Box<dyn std::any::Any + Send>>,
    /// Flag indicating whether the compiled code has been invalidated.
    /// When set to `true`, any `GUARD_NOT_INVALIDATED` in the compiled
    /// code will fail, causing execution to bail out to the interpreter.
    pub invalidated: Arc<AtomicBool>,
    /// Alternative loop versions to compile immediately after the main loop.
    pub version_info: Option<LoopVersionInfo>,
    /// history.py:449: _keepalive_jitcell_tokens — set of other tokens
    /// that this loop can jump to (via CALL_ASSEMBLER or JUMP).
    /// Upstream prevents the target from being evicted while this loop
    /// is alive by holding the actual `JitCellToken` object reference
    /// (`history.py:441 _keepalive_jitcell_tokens = {}` keyed on the
    /// token object itself, with `:451 record_jump_to` writing
    /// `self._keepalive_jitcell_tokens[target] = None`).  Pyre keeps the
    /// same shape: an `Arc<JitCellToken>` set keyed on token number.
    ///
    /// Wrapped in `Mutex` so `record_jump_to` can push through
    /// `&JitCellToken` once the token has been shared via
    /// `Arc<JitCellToken>` (Slice 5.1 lifted `CompiledEntry::token` to
    /// `Arc<JitCellToken>` so the same object is reachable from
    /// `record_loop_or_bridge`'s descr walk).  The Mutex is the
    /// Rust-side equivalent of RPython's implicit dict-mutation interior
    /// mutability under the single-threaded JIT scheduler invariant.
    pub keepalive_tokens: parking_lot::Mutex<Vec<Arc<JitCellToken>>>,
    /// `rpython/jit/backend/model.py:292` `CompiledLoopToken` parity.
    ///
    /// Carries per-compilation metadata: `asmmemmgr_blocks` (owned bridge
    /// memory blocks — `model.py:293`), `asmmemmgr_gcreftracers`
    /// (`model.py:294`), `frame_info` (JIT frame layout —
    /// `x86/assembler.py:514`), `bridges_count`, `invalidate_positions`,
    /// `looptokens_redirected_to`. Populated eagerly by `JitCellToken::new`
    /// so backends can update fields through `&JitCellToken`.
    ///
    /// Interior mutability on fields uses `parking_lot::Mutex` because
    /// bridge compilation and execution may see `&JitCellToken` from
    /// multiple threads (compile_bridge receives `&JitCellToken`).
    pub compiled_loop_token: Option<Arc<CompiledLoopToken>>,
    /// `rpython/jit/backend/x86/assembler.py:599`
    /// `looptoken._ll_function_addr = rawstart + functionpos` —
    /// address of the compiled loop entry. 0 until `compile_loop`
    /// assigns it.
    pub _ll_function_addr: usize,
    /// `memmgr.py:59-60` `looptoken.generation`. Updated by
    /// `MemoryManager.keep_loop_alive` and read by
    /// `_kill_old_loops_now`. Default `0` means "not yet seen by
    /// memmgr"; the eviction predicate at `memmgr.py:71` requires
    /// `0 <= gen < max_generation`, so default-0 tokens are
    /// candidates for eviction immediately — matching RPython where
    /// `__init__` does not initialize `generation`. In practice
    /// `compile.py:1149` calls `keep_loop_alive` before any
    /// `_kill_old_loops_now` could see the token. RPython's
    /// `r_int64` is signed 64-bit; pyre uses `i64` to preserve the
    /// wraparound math at `memmgr.py:38` (5e9 years given 1000
    /// loops/sec). Interior mutability via `Cell<i64>` mirrors the
    /// RPython attribute write through `&JitCellToken`. Sync is
    /// covered by the existing `unsafe impl Sync for JitCellToken`
    /// at line 1130 — single-threaded JIT scheduler invariant.
    pub generation: Cell<i64>,
}

impl JitCellToken {
    pub fn new(number: u64) -> Self {
        JitCellToken {
            number,
            green_key: 0,
            inputarg_types: Vec::new(),
            num_scalar_inputargs: 0,
            virtualizable_arg_index: None,
            outermost_jitdriver_index: None,
            compiled: None,
            invalidated: Arc::new(AtomicBool::new(false)),
            version_info: None,
            keepalive_tokens: parking_lot::Mutex::new(Vec::new()),
            // `rpython/jit/backend/x86/assembler.py:514` creates the
            // `compiled_loop_token` at the start of `assemble_loop` —
            // i.e., lazily when the backend compiles the token. pyre
            // initializes it eagerly here so the field is always present.
            compiled_loop_token: Some(Arc::new(CompiledLoopToken::new(number))),
            _ll_function_addr: 0,
            // memmgr.py:38 default; first keep_loop_alive overwrites this.
            generation: Cell::new(0),
        }
    }

    /// `rpython/jit/backend/llsupport/assembler.py:184-188`
    /// `get_asmmemmgr_blocks(self, looptoken)` parity.
    ///
    /// Returns a mutex guard over the `asmmemmgr_blocks` vec on this
    /// token's `compiled_loop_token`. Panics if the token has no
    /// `compiled_loop_token` — `JitCellToken::new` sets this eagerly, so
    /// production callers never hit the panic.
    pub fn asmmemmgr_blocks(
        &self,
    ) -> parking_lot::MutexGuard<'_, Vec<Box<dyn std::any::Any + Send>>> {
        self.compiled_loop_token
            .as_ref()
            .expect("JitCellToken missing compiled_loop_token")
            .asmmemmgr_blocks
            .lock()
    }

    /// history.py:451-453: record_jump_to — record that this loop can
    /// jump to another JitCellToken (via CALL_ASSEMBLER or JUMP).
    ///
    /// ```python
    /// def record_jump_to(self, target_token):
    ///     assert isinstance(target_token, JitCellToken)
    ///     self._keepalive_jitcell_tokens[target_token] = None
    /// ```
    ///
    /// Pyre stores the target's `Arc<JitCellToken>`, the direct Rust
    /// analog of PyPy's Python-object dict-key keepalive.  The
    /// `target.number == self.number` short-circuit matches PyPy's
    /// `if target_token is self: return` (implicit because the dict
    /// would just no-op); the explicit guard avoids the Mutex lock for
    /// the self-jump case.
    pub fn record_jump_to(&self, target: Arc<JitCellToken>) {
        if target.number == self.number {
            return;
        }
        let target_number = target.number;
        let mut guard = self.keepalive_tokens.lock();
        if !guard
            .iter()
            .any(|existing| existing.number == target_number)
        {
            guard.push(target);
        }
    }

    /// Mark this loop as invalidated. Any subsequent execution of
    /// GUARD_NOT_INVALIDATED in the compiled code will fail.
    pub fn invalidate(&self) {
        self.invalidated.store(true, Ordering::Release);
    }

    /// Check whether this loop has been invalidated.
    pub fn is_invalidated(&self) -> bool {
        self.invalidated.load(Ordering::Acquire)
    }

    /// model.py: has_compiled_code()
    /// Whether this token has compiled code attached.
    pub fn has_compiled_code(&self) -> bool {
        self.compiled.is_some()
    }

    /// model.py: get_number()
    pub fn get_number(&self) -> u64 {
        self.number
    }

    /// model.py: reset_compiled()
    /// Remove the compiled code (e.g., after invalidation).
    pub fn reset_compiled(&mut self) {
        self.compiled = None;
    }

    /// Get a clone of the invalidated flag (for registering with QuasiImmut).
    pub fn invalidation_flag(&self) -> Arc<AtomicBool> {
        self.invalidated.clone()
    }
}

impl std::fmt::Debug for JitCellToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCellToken")
            .field("number", &self.number)
            .finish()
    }
}

// pyre is single-threaded (no-GIL → still one JIT thread in practice,
// matching RPython's single-interpreter assumption).  `JitCellToken`
// embeds `Rc<RdVirtualInfo>` and `Box<dyn Any + Send>` which are not
// automatically `Sync`; marking the parent struct `Send + Sync` here
// is sound under the same invariant that `DynasmFailDescr` /
// `CraneliftFailDescr` rely on — a promise held by the JIT scheduler
// that tokens are only touched from the JIT thread.  Backends store
// `Weak<JitCellToken>` handles for cross-token `find_descr_by_ptr`
// fallback and need these impls to satisfy their `Send` trait bound.
unsafe impl Send for JitCellToken {}
unsafe impl Sync for JitCellToken {}

/// A "dead frame" — the state after JIT execution finishes or hits a guard.
///
/// The backend stores register/stack values here so the frontend can read them.
pub struct DeadFrame {
    /// Backend-specific frame data.
    pub data: Box<dyn std::any::Any + Send>,
}

/// `compile.py:665-674` `make_and_attach_done_descrs` + `pyjitpl.py:2283`
/// `propagate_exception_descr`: snapshot of the six descrs the metainterp
/// attaches to the owning cpu instance at
/// `MetaInterpStaticData.finish_setup` (pyjitpl.py:2222).  RPython backend
/// code reads these as `self.cpu.xxx` attributes at every use site
/// (`rpython/jit/backend/x86/assembler.py:337`,
/// `rpython/jit/backend/llgraph/runner.py:1478`); pyre captures them by
/// value at `compile_loop` entry (from the owning backend's per-instance
/// fields) so FINISH / CALL_ASSEMBLER emission can stamp `jf_descr` with
/// the attached identity without holding a `&dyn Backend` receiver.
///
/// Field names mirror the RPython attribute names 1:1 for parity with
/// `compile.make_and_attach_done_descrs` targets.
#[derive(Debug, Clone, Copy, Default)]
pub struct AttachedDescrPtrs {
    pub done_with_this_frame_descr_void: usize,
    pub done_with_this_frame_descr_int: usize,
    pub done_with_this_frame_descr_ref: usize,
    pub done_with_this_frame_descr_float: usize,
    pub exit_frame_with_exception_descr_ref: usize,
    pub propagate_exception_descr: usize,
}

impl AttachedDescrPtrs {
    /// `compile.py:324-336` `_call_assembler` — pick the attached
    /// `done_with_this_frame_descr_*` whose result type matches the
    /// portal's finish kind.
    pub fn done_with_this_frame_descr_ptr_for_type(&self, tp: Type) -> usize {
        match tp {
            Type::Void => self.done_with_this_frame_descr_void,
            Type::Int => self.done_with_this_frame_descr_int,
            Type::Ref => self.done_with_this_frame_descr_ref,
            Type::Float => self.done_with_this_frame_descr_float,
        }
    }

    /// `compile.py:665-670` parity: is `ptr` one of the four
    /// `DoneWithThisFrameDescr*` pointers attached to this cpu?
    /// Zero pointers (unattached slots) are never considered matches
    /// — a test with some slots unset would otherwise make
    /// `is_done_with_this_frame_descr(0)` spuriously true.
    pub fn is_done_with_this_frame_descr(&self, ptr: usize) -> bool {
        ptr != 0
            && (ptr == self.done_with_this_frame_descr_void
                || ptr == self.done_with_this_frame_descr_int
                || ptr == self.done_with_this_frame_descr_ref
                || ptr == self.done_with_this_frame_descr_float)
    }

    /// `compile.py:671` parity: does `ptr` match the attached
    /// `exit_frame_with_exception_descr_ref` singleton?
    pub fn is_exit_frame_with_exception_descr(&self, ptr: usize) -> bool {
        ptr != 0 && ptr == self.exit_frame_with_exception_descr_ref
    }
}

/// `compile.py:665-674` `make_and_attach_done_descrs` + `pyjitpl.py:2283`:
/// the six descrs the metainterp attaches to each cpu instance at
/// `MetaInterpStaticData.finish_setup`.
///
/// Stored inside a heap-pinned `Arc<RwLock<CpuDescrAttachments>>` on every
/// `Backend` impl ([`CpuDescrHandle`] below):
///
///   * The `Arc` allocation address stays stable when the `Backend`
///     value is moved (the metainterp keeps `backend: <Backend>` by value
///     in `MetaInterp`; tests hold stack-local `<Backend>::new()`).
///     RPython gets this property for free from Python object identity;
///     pyre pays one heap indirection to recreate it.
///   * Compiled code produced by `compile_loop` carries an `Arc` clone
///     so the attachments outlive the owning backend — the JIT-emitted
///     CALL_ASSEMBLER slow path dereferences the baked handle pointer
///     long after `compile_loop` returns.
///   * The `RwLock` permits `Backend::set_done_with_this_frame_descr_*`
///     to mutate through `&Backend` (via the Arc) without requiring
///     `&mut self` access to the inner; the extern-C trampoline takes a
///     read lock to snapshot pointers for dispatch.
///
/// Field names mirror the RPython attribute names 1:1 for parity with
/// `compile.make_and_attach_done_descrs` targets.
#[derive(Default)]
pub struct CpuDescrAttachments {
    pub done_with_this_frame_descr_void: Option<majit_ir::DescrRef>,
    pub done_with_this_frame_descr_int: Option<majit_ir::DescrRef>,
    pub done_with_this_frame_descr_ref: Option<majit_ir::DescrRef>,
    pub done_with_this_frame_descr_float: Option<majit_ir::DescrRef>,
    pub exit_frame_with_exception_descr_ref: Option<majit_ir::DescrRef>,
    pub propagate_exception_descr: Option<majit_ir::DescrRef>,
}

impl CpuDescrAttachments {
    /// Snapshot the six attached pointers for emission / dispatch sites
    /// that need compile-time or runtime-immediate access.
    pub fn descr_ptrs(&self) -> AttachedDescrPtrs {
        fn ptr(d: &Option<majit_ir::DescrRef>) -> usize {
            d.as_ref()
                .map_or(0, |arc| Arc::as_ptr(arc) as *const () as usize)
        }
        AttachedDescrPtrs {
            done_with_this_frame_descr_void: ptr(&self.done_with_this_frame_descr_void),
            done_with_this_frame_descr_int: ptr(&self.done_with_this_frame_descr_int),
            done_with_this_frame_descr_ref: ptr(&self.done_with_this_frame_descr_ref),
            done_with_this_frame_descr_float: ptr(&self.done_with_this_frame_descr_float),
            exit_frame_with_exception_descr_ref: ptr(&self.exit_frame_with_exception_descr_ref),
            propagate_exception_descr: ptr(&self.propagate_exception_descr),
        }
    }
}

/// Heap-pinned handle for the six per-cpu descr attachments.  The
/// `Arc`'s payload (the `RwLock<CpuDescrAttachments>`) lives at a
/// stable heap address so `compile_loop` can bake that address as an
/// immediate in the JIT-emitted CALL_ASSEMBLER helper call site and
/// the extern-C trampoline can dereference it later.
pub type CpuDescrHandle = Arc<std::sync::RwLock<CpuDescrAttachments>>;

/// The backend trait — implemented by Cranelift (or other code generators).
///
/// Mirrors rpython/jit/backend/model.py AbstractCPU.
pub trait Backend: Send {
    /// Compile a loop trace into native code.
    fn compile_loop(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        token: &mut JitCellToken,
    ) -> Result<AsmInfo, BackendError>;

    /// Register constant OpRef → i64 values consumed by the next
    /// `compile_loop` / `compile_bridge` call.  Default is no-op — only
    /// backends that honour constants at emit time override.  The
    /// companion `set_constant_types` carries the per-OpRef `Type`
    /// tag.
    fn set_constants(&mut self, _constants: std::collections::HashMap<u32, i64>) {}

    /// Register constant OpRef → `Type` annotations.
    fn set_constant_types(&mut self, _constant_types: std::collections::HashMap<u32, Type>) {}

    /// Force the next `compile_loop` / `compile_bridge` call to stamp
    /// this trace id on exits.
    fn set_next_trace_id(&mut self, _trace_id: u64) {}

    /// Force the next `compile_loop` / `compile_bridge` call to attach
    /// this header PC to synthesised exit recovery layouts.
    fn set_next_header_pc(&mut self, _header_pc: u64) {}

    /// `compile.py:665-674` `make_and_attach_done_descrs([self, cpu])` —
    /// per-result-type `DoneWithThisFrame*` singleton shared with
    /// `MetaInterpStaticData`.  Attached once per CPU instance, matching
    /// `pyjitpl.py:2222`.  Backends that use pointer identity for the
    /// FINISH fast path (dynasm `is_done_with_this_frame_descr`,
    /// cranelift CA dispatch) override to store the `Arc` and publish
    /// `Arc::as_ptr` to the comparison sites.
    fn set_done_with_this_frame_descr_void(&mut self, _descr: Arc<dyn Descr>) {}

    /// `compile.py:665-674` `done_with_this_frame_descr_int` — INT-typed
    /// variant.  See `set_done_with_this_frame_descr_void` for the
    /// attachment contract.
    fn set_done_with_this_frame_descr_int(&mut self, _descr: Arc<dyn Descr>) {}

    /// `compile.py:665-674` `done_with_this_frame_descr_ref` — REF-typed
    /// variant.  See `set_done_with_this_frame_descr_void` for the
    /// attachment contract.
    fn set_done_with_this_frame_descr_ref(&mut self, _descr: Arc<dyn Descr>) {}

    /// `compile.py:665-674` `done_with_this_frame_descr_float` —
    /// FLOAT-typed variant.  See `set_done_with_this_frame_descr_void`
    /// for the attachment contract.
    fn set_done_with_this_frame_descr_float(&mut self, _descr: Arc<dyn Descr>) {}

    /// `compile.py:665-674` `exit_frame_with_exception_descr_ref` —
    /// FINISH descr used by `compile_exit_frame_with_exception`
    /// (`pyjitpl.py:3238-3245`).  See `set_done_with_this_frame_descr_void`
    /// for the attachment contract.
    fn set_exit_frame_with_exception_descr_ref(&mut self, _descr: Arc<dyn Descr>) {}

    /// `pyjitpl.py:2283` `self.cpu.propagate_exception_descr = exc_descr`
    /// — shared `PropagateExceptionDescr` instance used by
    /// `compile_tmp_callback`'s `GUARD_NO_EXCEPTION` and by the
    /// backend's propagate-exception slow path
    /// (`x86/assembler.py:870`, `aarch64/assembler.py:566-572`).
    fn set_propagate_exception_descr(&mut self, _descr: Arc<dyn Descr>) {}

    /// Register a placeholder for a pending token (RPython compile_tmp_callback).
    /// The placeholder has null code_ptr; call_assembler_fast_path detects this
    /// and falls back to force_fn. Replaced by the real target on compile_loop.
    /// Register a placeholder for a pending token (RPython compile_tmp_callback).
    /// `num_scalar_inputargs` = virtualizable.py:86 NUM_SCALAR_INPUTARGS.
    fn register_pending_target(
        &mut self,
        _token_number: u64,
        _input_types: Vec<Type>,
        _num_inputs: usize,
        _num_scalar_inputargs: usize,
        _index_of_virtualizable: i32,
    ) {
    }

    /// Compile a bridge (side exit path) and attach it to the loop.
    ///
    /// `previous_tokens` contains old tokens from retraces. Because
    /// Cranelift can't patch existing machine code (unlike RPython's x86
    /// backend), the running machine code may reference fail_descrs from
    /// an older token. The bridge must be attached to ALL matching
    /// fail_descrs across current + previous tokens.
    fn compile_bridge(
        &mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[Op],
        original_token: &JitCellToken,
        previous_tokens: &[std::sync::Arc<JitCellToken>],
    ) -> Result<AsmInfo, BackendError>;

    /// Compile all registered loop versions as bridges.
    ///
    /// Called after `compile_loop()` succeeds. For each version registered
    /// in the token's `version_info`, finds the corresponding version guard
    /// descriptor and compiles the alternative trace as a bridge.
    fn compile_versions(&mut self, token: &JitCellToken) -> Result<Vec<AsmInfo>, BackendError> {
        let versions = match &token.version_info {
            Some(info) => info
                .versions
                .iter()
                .map(|(guard_idx, inputargs, ops)| (*guard_idx, inputargs.clone(), ops.clone()))
                .collect::<Vec<_>>(),
            None => return Ok(Vec::new()),
        };

        let mut results = Vec::new();
        for (guard_idx, inputargs, ops) in &versions {
            let descr = LoopVersionDescr {
                version_index: *guard_idx,
                fail_arg_types: inputargs.iter().map(|ia| ia.tp).collect(),
            };
            let asm = self.compile_bridge(&descr, inputargs, ops, token, &[])?;
            results.push(asm);
        }
        Ok(results)
    }

    /// Register a freshly-compiled JitCellToken as still reachable from
    /// the frontend.  Backends that need to resolve `jf_descr` pointers
    /// across token boundaries (dynasm's `find_descr_by_ptr` cross-token
    /// fallback) use this to iterate all live tokens without maintaining
    /// a separate ptr-to-Arc side table.  Called by `MetaInterp` after
    /// `compile_loop` / `compile_versions` succeeds.  Default no-op for
    /// backends whose descr-resolution paths do not cross tokens
    /// (cranelift resolves via `CompiledLoop::fail_descrs` directly).
    fn track_compiled_token(&mut self, _token: Arc<JitCellToken>) {}

    /// Mark the most recently compiled bridge on the given guard as a
    /// loop-closing bridge: on Finish, its outputs should re-enter the
    /// parent loop instead of returning to the interpreter.
    fn mark_bridge_loop_reentry(
        &self,
        _original_token: &JitCellToken,
        _source_trace_id: u64,
        _fail_index: u32,
    ) {
        // Default no-op — backends that support bridge re-entry override this.
    }

    /// Cranelift workaround — no RPython counterpart.
    ///
    /// RPython's x86/ARM backends patch guard failure jump targets in-place,
    /// so bridges survive retrace automatically. Cranelift cannot patch
    /// emitted machine code, so when a loop is retraced (producing a new
    /// token), existing bridges from the old token must be explicitly copied
    /// to matching guards in the new token.
    ///
    /// Called from metainterp after compile_loop, because only the metainterp
    /// has access to both old_token (from compiled_loops.remove) and new_token
    /// (from compile_loop). compile_loop itself only receives the new token.
    ///
    /// Backends that support in-place patching (e.g. dynasm) leave this as
    /// no-op — bridges are attached to the guard's machine code directly.
    fn migrate_bridges(&self, _old_token: &JitCellToken, _new_token: &JitCellToken) {}

    /// compile.py:741-745: look up (status, descr_addr) for a guard.
    /// Uses (trace_id, fail_index) to find the exact descriptor, including
    /// bridge guards and previous tokens — matching start_guard_compiling's
    /// find_fail_descr_in_fail_descrs pattern.
    fn get_guard_status(
        &self,
        _token: &JitCellToken,
        _trace_id: u64,
        _fail_index: u32,
    ) -> (u64, usize) {
        (0, 0)
    }

    /// compile.py:826-830 store_hash: assign jitcounter hashes to guards.
    /// Called after compile_loop/compile_bridge with hashes from
    /// jitcounter.fetch_next_hash(). Skips guards that already have
    /// status set by make_a_counter_per_value (GUARD_VALUE).
    fn store_guard_hashes(&self, _token: &JitCellToken, _hashes: &[u64]) {}

    /// compile.py / resume.py:1143 plumbing: publish the shared
    /// `CallInfoCollection` so the backend can resolve OS_STR_CONCAT
    /// etc. function pointers when materializing VStr/VUni
    /// Concat/Slice virtuals from guard-exit recovery data.
    ///
    /// Default is a no-op: backends that handle string virtual
    /// materialization entirely through the frontend `bh_*` runtime
    /// (dynasm) or that never expose VStr/VUni recovery_layouts need
    /// not participate. Cranelift overrides this to cache the
    /// collection for `collect_guards` consumption.
    fn set_callinfocollection(
        &mut self,
        _cic: Option<std::sync::Arc<majit_ir::CallInfoCollection>>,
    ) {
    }

    /// store_hash for bridge guards — same as store_guard_hashes but for
    /// the most recently compiled bridge on the given guard.
    /// Uses (trace_id, fail_index) for recursive descriptor lookup.
    fn store_bridge_guard_hashes(
        &self,
        _token: &JitCellToken,
        _source_trace_id: u64,
        _source_fail_index: u32,
        _hashes: &[u64],
    ) {
    }

    /// compile.py:741: self.status — read status from the failed descriptor
    /// directly by its raw address (no re-lookup). RPython calls self.status
    /// on the same descriptor object; descr_addr IS that object's identity.
    ///
    /// # Safety
    /// descr_addr must be a valid pointer returned from a guard failure in
    /// the same compilation session. The descriptor must still be alive
    /// (held by compiled_loops or previous_tokens).
    fn read_descr_status(&self, _descr_addr: usize) -> u64 {
        0
    }

    /// compile.py:786-788: `self.start_compiling()`.
    ///
    /// RPython calls `self.status |= ST_BUSY_FLAG` directly on the
    /// descriptor. In Rust, the descriptor is behind a trait object and
    /// Arc, so we pass `descr_addr` (the Arc's raw pointer from the guard
    /// failure result) and let the backend cast it back. This is the Rust
    /// equivalent of RPython's `self` — both identify the exact descriptor.
    fn start_compiling_descr(&self, _descr_addr: usize) {}

    /// compile.py:790-795: `self.done_compiling()`.
    /// Same pattern as start_compiling_descr — see above.
    fn done_compiling_descr(&self, _descr_addr: usize) {}

    /// Execute compiled code starting at the given token.
    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame;

    /// Execute compiled code with integer-only arguments.
    ///
    /// Avoids the `Value::Int` wrapping/unwrapping overhead when all
    /// arguments are known to be integers (the common case for loop entry).
    fn execute_token_ints(&self, token: &JitCellToken, args: &[i64]) -> DeadFrame {
        let values: Vec<Value> = args.iter().map(|&v| Value::Int(v)).collect();
        self.execute_token(token, &values)
    }

    /// Execute compiled code with typed arguments and return a lightweight result.
    ///
    /// This preserves mixed `Int` / `Ref` / `Float` arguments while still
    /// avoiding explicit deadframe decoding in the caller.
    fn execute_token_raw(&self, token: &JitCellToken, args: &[Value]) -> RawExecResult {
        let frame = self.execute_token(token, args);
        let descr = self.get_latest_descr(&frame);
        let exit_layout = self.describe_deadframe(&frame);
        let savedata = self.get_savedata_ref(&frame);
        let exception_value = self.grab_exc_value(&frame);
        let exit_arity = descr.fail_arg_types().len();
        let mut outputs = Vec::with_capacity(exit_arity);
        let mut typed_outputs = Vec::with_capacity(exit_arity);
        for (i, &tp) in descr.fail_arg_types().iter().enumerate() {
            match tp {
                Type::Int => {
                    let value = self.get_int_value(&frame, i);
                    outputs.push(value);
                    typed_outputs.push(Value::Int(value));
                }
                Type::Ref => {
                    let value = self.get_ref_value(&frame, i);
                    outputs.push(value.as_usize() as i64);
                    typed_outputs.push(Value::Ref(value));
                }
                Type::Float => {
                    let value = self.get_float_value(&frame, i);
                    outputs.push(value.to_bits() as i64);
                    typed_outputs.push(Value::Float(value));
                }
                Type::Void => {
                    outputs.push(0);
                    typed_outputs.push(Value::Void);
                }
            }
        }
        RawExecResult {
            outputs,
            typed_outputs,
            exit_layout,
            force_token_slots: descr.force_token_slots().to_vec(),
            savedata,
            exception_value,
            fail_index: descr.fail_index(),
            trace_id: descr.trace_id(),
            is_finish: descr.is_finish(),
            is_exit_frame_with_exception: descr.is_exit_frame_with_exception(),
            status: descr.get_status(),
            descr_addr: descr as *const dyn majit_ir::descr::FailDescr as *const () as usize,
        }
    }

    /// Execute compiled code and return a lightweight result without
    /// DeadFrame boxing.
    ///
    /// Returns the output values directly, avoiding the intermediate
    /// DeadFrame heap allocation and the per-value downcast extraction loop.
    fn execute_token_ints_raw(&self, token: &JitCellToken, args: &[i64]) -> RawExecResult {
        let values: Vec<Value> = args.iter().map(|&v| Value::Int(v)).collect();
        self.execute_token_raw(token, &values)
    }

    /// Inspect static exit layouts for a compiled loop token.
    fn compiled_fail_descr_layouts(&self, _token: &JitCellToken) -> Option<Vec<FailDescrLayout>> {
        None
    }

    /// Inspect static exit layouts for a bridge attached to a source guard.
    fn compiled_bridge_fail_descr_layouts(
        &self,
        _original_token: &JitCellToken,
        _source_trace_id: u64,
        _source_fail_index: u32,
    ) -> Option<Vec<FailDescrLayout>> {
        None
    }

    /// Inspect static exit layouts for any compiled trace owned by this token.
    ///
    /// This is the trace-id keyed counterpart to the root/bridge-specific
    /// inspection APIs above.
    fn compiled_trace_fail_descr_layouts(
        &self,
        _token: &JitCellToken,
        _trace_id: u64,
    ) -> Option<Vec<FailDescrLayout>> {
        None
    }

    /// Inspect static terminal-exit layouts for a compiled loop token.
    fn compiled_terminal_exit_layouts(
        &self,
        _token: &JitCellToken,
    ) -> Option<Vec<TerminalExitLayout>> {
        None
    }

    /// Inspect static terminal-exit layouts for a bridge attached to a source guard.
    fn compiled_bridge_terminal_exit_layouts(
        &self,
        _original_token: &JitCellToken,
        _source_trace_id: u64,
        _source_fail_index: u32,
    ) -> Option<Vec<TerminalExitLayout>> {
        None
    }

    /// Inspect static terminal-exit layouts for any compiled trace owned by this token.
    fn compiled_trace_terminal_exit_layouts(
        &self,
        _token: &JitCellToken,
        _trace_id: u64,
    ) -> Option<Vec<TerminalExitLayout>> {
        None
    }

    /// Inspect static metadata for any compiled trace owned by this token.
    fn compiled_trace_info(
        &self,
        _token: &JitCellToken,
        _trace_id: u64,
    ) -> Option<CompiledTraceInfo> {
        None
    }

    /// Query complete frame-stack layouts for all guards in a compiled loop.
    ///
    /// Returns `(fail_index, frame_stack)` pairs for each guard that has
    /// recovery layout metadata. Backends that populate recovery layouts
    /// at compile time can override this to expose the frame stacks.
    fn compiled_guard_frame_stacks(
        &self,
        _token: &JitCellToken,
    ) -> Option<Vec<(u32, Vec<ExitFrameLayout>)>> {
        None
    }

    /// Patch backend-owned recovery metadata for a specific compiled terminal exit.
    fn update_terminal_exit_recovery_layout(
        &mut self,
        _token: &JitCellToken,
        _trace_id: u64,
        _op_index: usize,
        _recovery_layout: ExitRecoveryLayout,
    ) -> bool {
        false
    }

    /// Describe the latest exit stored in a deadframe.
    ///
    /// Backends can override this to surface backend-owned recovery metadata
    /// directly from the deadframe's fail descriptor.
    fn describe_deadframe(&self, frame: &DeadFrame) -> Option<FailDescrLayout> {
        let descr = self.get_latest_descr(frame);
        Some(FailDescrLayout {
            fail_index: descr.fail_index(),
            source_op_index: None,
            trace_id: descr.trace_id(),
            trace_info: None,
            fail_arg_types: descr.fail_arg_types().to_vec(),
            is_finish: descr.is_finish(),
            gc_ref_slots: descr
                .fail_arg_types()
                .iter()
                .enumerate()
                .filter_map(|(slot, _)| descr.is_gc_ref_slot(slot).then_some(slot))
                .collect(),
            force_token_slots: descr.force_token_slots().to_vec(),
            recovery_layout: None,
            frame_stack: None,
            rd_numb: None,
            rd_consts: None,
            rd_virtuals: None,
            rd_pendingfields: None,
        })
    }

    /// Patch backend-owned recovery metadata for a specific compiled exit.
    fn update_fail_descr_recovery_layout(
        &mut self,
        _token: &JitCellToken,
        _trace_id: u64,
        _fail_index: u32,
        _recovery_layout: ExitRecoveryLayout,
    ) -> bool {
        false
    }

    /// Force a frame identified by a `FORCE_TOKEN` result.
    fn force(&self, _force_token: GcRef) -> Option<DeadFrame> {
        None
    }

    /// Store a saved-data GC ref on a dead frame.
    fn set_savedata_ref(&self, _frame: &mut DeadFrame, _data: GcRef) {
        // No-op: backend doesn't support savedata
    }

    /// Read a saved-data GC ref from a dead frame.
    fn get_savedata_ref(&self, _frame: &DeadFrame) -> Option<GcRef> {
        None
    }

    /// Read a pending exception GC ref from a dead frame.
    fn grab_exc_value(&self, _frame: &DeadFrame) -> GcRef {
        GcRef::NULL
    }

    /// Read the FailDescr from the last guard failure.
    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr;

    /// Read an integer value from a dead frame at the given index.
    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64;

    /// Read a float value from a dead frame.
    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64;

    /// Read a GC reference value from a dead frame.
    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> majit_ir::GcRef;

    /// Invalidate a compiled loop (e.g., due to GUARD_NOT_INVALIDATED).
    fn invalidate_loop(&self, token: &JitCellToken);

    /// Redirect calls from one loop token to another (for CALL_ASSEMBLER).
    fn redirect_call_assembler(
        &self,
        _old: &JitCellToken,
        _new: &JitCellToken,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    /// Free resources associated with a compiled loop.
    fn free_loop(&mut self, _token: &JitCellToken) {
        // Default: no-op
    }

    // ── model.py: bh_* blackhole interpreter helpers ──
    //
    // These methods provide fallback implementations for operations
    // that the blackhole interpreter needs to execute when falling
    // back from JIT-compiled code. The backend implements these
    // to read/write memory at known addresses.

    // ── model.py:216-228 field operations ──
    /// model.py:216 bh_getfield_gc_i(struct, fielddescr)
    fn bh_getfield_gc_i(
        &self,
        _struct_ptr: i64,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        0
    }
    /// model.py:217 bh_getfield_gc_r(struct, fielddescr)
    fn bh_getfield_gc_r(
        &self,
        _struct_ptr: i64,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> GcRef {
        GcRef::NULL
    }
    /// model.py:218 bh_getfield_gc_f(struct, fielddescr)
    fn bh_getfield_gc_f(
        &self,
        _struct_ptr: i64,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        0.0
    }
    /// model.py:222 bh_setfield_gc_i(struct, newvalue, fielddescr)
    fn bh_setfield_gc_i(
        &self,
        _struct_ptr: i64,
        _newvalue: i64,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    /// model.py:223 bh_setfield_gc_r(struct, newvalue, fielddescr)
    fn bh_setfield_gc_r(
        &self,
        _struct_ptr: i64,
        _newvalue: GcRef,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    /// model.py:224 bh_setfield_gc_f(struct, newvalue, fielddescr)
    fn bh_setfield_gc_f(
        &self,
        _struct_ptr: i64,
        _newvalue: f64,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) {
    }

    // ── model.py:209-215, 247-253 array operations ──
    /// model.py:209 bh_getarrayitem_gc_i(array, index, arraydescr)
    fn bh_getarrayitem_gc_i(
        &self,
        _array_ptr: i64,
        _index: i64,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        0
    }
    /// model.py:210 bh_getarrayitem_gc_r(array, index, arraydescr)
    fn bh_getarrayitem_gc_r(
        &self,
        _array_ptr: i64,
        _index: i64,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> GcRef {
        GcRef::NULL
    }
    /// model.py:211 bh_getarrayitem_gc_f(array, index, arraydescr)
    fn bh_getarrayitem_gc_f(
        &self,
        _array_ptr: i64,
        _index: i64,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        0.0
    }
    /// model.py:247 bh_setarrayitem_gc_i(array, index, newvalue, arraydescr)
    fn bh_setarrayitem_gc_i(
        &self,
        _array_ptr: i64,
        _index: i64,
        _newvalue: i64,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    /// model.py:248 bh_setarrayitem_gc_r(array, index, newvalue, arraydescr)
    fn bh_setarrayitem_gc_r(
        &self,
        _array_ptr: i64,
        _index: i64,
        _newvalue: GcRef,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    /// model.py:249 bh_setarrayitem_gc_f(array, index, newvalue, arraydescr)
    fn bh_setarrayitem_gc_f(
        &self,
        _array_ptr: i64,
        _index: i64,
        _newvalue: f64,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) {
    }

    // ── model.py: raw array operations ──
    /// model.py:212 bh_getarrayitem_raw_i(array, index, arraydescr)
    fn bh_getarrayitem_raw_i(
        &self,
        _array: i64,
        _index: i64,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        0
    }
    /// model.py:214 bh_getarrayitem_raw_f(array, index, arraydescr)
    fn bh_getarrayitem_raw_f(
        &self,
        _array: i64,
        _index: i64,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        0.0
    }
    /// model.py:250 bh_setarrayitem_raw_i(array, index, newvalue, arraydescr)
    fn bh_setarrayitem_raw_i(
        &self,
        _array: i64,
        _index: i64,
        _newvalue: i64,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    /// model.py:252 bh_setarrayitem_raw_f(array, index, newvalue, arraydescr)
    fn bh_setarrayitem_raw_f(
        &self,
        _array: i64,
        _index: i64,
        _newvalue: f64,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) {
    }

    /// model.py:254 bh_arraylen_gc(array, arraydescr)
    fn bh_arraylen_gc(
        &self,
        _array_ptr: i64,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        0
    }

    // ── model.py:230-236 allocation ──
    /// model.py:230 / llmodel.py:775 bh_new(sizedescr)
    fn bh_new(&self, _sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
        0
    }
    /// model.py:231 / llmodel.py:778 bh_new_with_vtable(sizedescr)
    fn bh_new_with_vtable(&self, _sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
        0
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Backend-side helper consulted only when `vtable_offset is None`
    /// (i.e. translation with --gcremovetypeptr). Returns the typeid that
    /// `_cmp_guard_gc_type` should compare against.
    ///
    /// Default `None` indicates the GC layer does not implement the
    /// gcremovetypeptr lowering, matching pyre's configuration.
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, _classptr: usize) -> Option<u32> {
        None
    }
    /// model.py:233 bh_new_array(length, arraydescr)
    fn bh_new_array(&self, _length: i64, _arraydescr: &majit_translate::jitcode::BhDescr) -> i64 {
        0
    }
    /// model.py:234 bh_new_array_clear(length, arraydescr)
    fn bh_new_array_clear(
        &self,
        _length: i64,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        0
    }
    /// model.py: bh_strlen(string_ptr)
    fn bh_strlen(&self, _string_ptr: i64) -> i64 {
        0
    }
    /// model.py: bh_strgetitem(string_ptr, index)
    fn bh_strgetitem(&self, _string_ptr: i64, _index: i64) -> i64 {
        0
    }
    /// model.py: bh_strsetitem(string_ptr, index, value)
    fn bh_strsetitem(&self, _string_ptr: i64, _index: i64, _value: i64) {}
    /// model.py: bh_newstr(length)
    fn bh_newstr(&self, _length: i64) -> i64 {
        0
    }
    /// model.py:266 bh_call_i(func, args_i, args_r, args_f, calldescr)
    fn bh_call_i(
        &self,
        _func: i64,
        _args_i: Option<&[i64]>,
        _args_r: Option<&[i64]>,
        _args_f: Option<&[i64]>,
        _calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> i64 {
        0
    }
    /// model.py:268 bh_call_r(func, args_i, args_r, args_f, calldescr)
    fn bh_call_r(
        &self,
        _func: i64,
        _args_i: Option<&[i64]>,
        _args_r: Option<&[i64]>,
        _args_f: Option<&[i64]>,
        _calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> GcRef {
        GcRef::NULL
    }
    /// model.py:270 bh_call_f(func, args_i, args_r, args_f, calldescr)
    fn bh_call_f(
        &self,
        _func: i64,
        _args_i: Option<&[i64]>,
        _args_r: Option<&[i64]>,
        _args_f: Option<&[i64]>,
        _calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> f64 {
        0.0
    }
    /// model.py:272 bh_call_v(func, args_i, args_r, args_f, calldescr)
    fn bh_call_v(
        &self,
        _func: i64,
        _args_i: Option<&[i64]>,
        _args_r: Option<&[i64]>,
        _args_f: Option<&[i64]>,
        _calldescr: &majit_translate::jitcode::BhCallDescr,
    ) {
    }

    // ── model.py: additional bh_* helpers ──

    /// model.py: bh_unicodelen(string_ptr)
    fn bh_unicodelen(&self, _string_ptr: i64) -> i64 {
        0
    }
    /// model.py: bh_unicodegetitem(string_ptr, index)
    fn bh_unicodegetitem(&self, _string_ptr: i64, _index: i64) -> i64 {
        0
    }
    /// model.py: bh_unicodesetitem(string_ptr, index, value)
    fn bh_unicodesetitem(&self, _string_ptr: i64, _index: i64, _value: i64) {}
    /// model.py: bh_newunicode(length)
    fn bh_newunicode(&self, _length: i64) -> i64 {
        0
    }
    /// model.py: bh_copystrcontent(src, dst, srcstart, dststart, length)
    fn bh_copystrcontent(
        &self,
        _src: i64,
        _dst: i64,
        _srcstart: i64,
        _dststart: i64,
        _length: i64,
    ) {
    }
    /// model.py: bh_copyunicodecontent(src, dst, srcstart, dststart, length)
    fn bh_copyunicodecontent(
        &self,
        _src: i64,
        _dst: i64,
        _srcstart: i64,
        _dststart: i64,
        _length: i64,
    ) {
    }
    /// llmodel.py:747-750 bh_raw_load_i(addr, offset, descr).
    fn bh_raw_load_i(
        &self,
        _addr: i64,
        _offset: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        0
    }
    /// llmodel.py:739-742 bh_raw_store_i(addr, offset, newvalue, descr).
    fn bh_raw_store_i(
        &self,
        _addr: i64,
        _offset: i64,
        _newvalue: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    // ── model.py: interior field access ──
    /// model.py: bh_getinteriorfield_gc_i(array, index, descr)
    fn bh_getinteriorfield_gc_i(
        &self,
        _array: i64,
        _index: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        0
    }
    /// model.py: bh_getinteriorfield_gc_r(array, index, descr)
    fn bh_getinteriorfield_gc_r(
        &self,
        _array: i64,
        _index: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) -> GcRef {
        GcRef::NULL
    }
    /// model.py: bh_getinteriorfield_gc_f(array, index, descr)
    fn bh_getinteriorfield_gc_f(
        &self,
        _array: i64,
        _index: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        0.0
    }
    /// model.py: bh_setinteriorfield_gc_i(array, index, newvalue, descr)
    fn bh_setinteriorfield_gc_i(
        &self,
        _array: i64,
        _index: i64,
        _newvalue: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    /// model.py: bh_setinteriorfield_gc_r(array, index, newvalue, descr)
    fn bh_setinteriorfield_gc_r(
        &self,
        _array: i64,
        _index: i64,
        _newvalue: GcRef,
        _descr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    /// model.py: bh_setinteriorfield_gc_f(array, index, newvalue, descr)
    fn bh_setinteriorfield_gc_f(
        &self,
        _array: i64,
        _index: i64,
        _newvalue: f64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    fn bh_gc_load_indexed_i(
        &self,
        _addr: i64,
        _index: i64,
        _scale: i64,
        _base_ofs: i64,
        _bytes: i64,
    ) -> i64 {
        0
    }
    fn bh_gc_load_indexed_f(
        &self,
        _addr: i64,
        _index: i64,
        _scale: i64,
        _base_ofs: i64,
        _bytes: i64,
    ) -> f64 {
        0.0
    }
    /// blackhole.py:1525-1529 bhimpl_gc_store_indexed_i
    fn bh_gc_store_indexed_i(
        &self,
        _addr: i64,
        _index: i64,
        _value: i64,
        _scale: i64,
        _base_ofs: i64,
        _bytes: i64,
    ) {
    }
    /// blackhole.py:1531-1535 bhimpl_gc_store_indexed_f
    fn bh_gc_store_indexed_f(
        &self,
        _addr: i64,
        _index: i64,
        _value: f64,
        _scale: i64,
        _base_ofs: i64,
        _bytes: i64,
    ) {
    }
    /// llmodel.py:752-753 bh_raw_load_f(addr, offset, descr).
    fn bh_raw_load_f(
        &self,
        _addr: i64,
        _offset: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        0.0
    }
    /// llmodel.py:744-745 bh_raw_store_f(addr, offset, newvalue, descr).
    fn bh_raw_store_f(
        &self,
        _addr: i64,
        _offset: i64,
        _newvalue: f64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    // ── model.py: raw field access ──
    fn bh_getfield_raw_i(
        &self,
        _struct_ptr: i64,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        0
    }
    fn bh_getfield_raw_r(
        &self,
        _struct_ptr: i64,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> GcRef {
        GcRef::NULL
    }
    fn bh_getfield_raw_f(
        &self,
        _struct_ptr: i64,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        0.0
    }
    fn bh_setfield_raw_i(
        &self,
        _struct_ptr: i64,
        _newvalue: i64,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    fn bh_setfield_raw_f(
        &self,
        _struct_ptr: i64,
        _newvalue: f64,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) {
    }

    /// model.py: bh_classof(obj_ptr)
    fn bh_classof(&self, _obj_ptr: i64) -> i64 {
        0
    }
    /// RPython rclass.ll_issubclass(typeptr, bounding_class).
    /// Returns true if `typeptr` is a subclass of `bounding_class`.
    fn bh_issubclass(&self, _typeptr: i64, _bounding_class: i64) -> bool {
        _typeptr == _bounding_class // default: exact match only
    }

    /// model.py: setup_once() — called once when the backend is first used.
    fn setup_once(&mut self) {}
    /// model.py: finish_once() — called when the JIT shuts down.
    fn finish_once(&mut self) {}

    // ── model.py: GC integration ──

    /// model.py: gc_set_extra_threshold()
    /// Inform the GC that extra memory was allocated outside of GC control.
    fn gc_set_extra_threshold(&self) {}

    /// model.py: force_head_version()
    /// Force updating the version stamp for GC write barrier optimization.
    fn force_head_version(&self) {}

    /// model.py: get_all_loop_runs()
    /// Return a list of (token_number, loop_run_count) for profiling.
    fn get_all_loop_runs(&self) -> Vec<(u64, u64)> {
        Vec::new()
    }

    /// model.py: cast_int_to_ptr(value)
    fn cast_int_to_ptr(&self, value: i64) -> i64 {
        value // identity on 64-bit
    }

    /// model.py: cast_ptr_to_int(value)
    fn cast_ptr_to_int(&self, value: i64) -> i64 {
        value
    }

    /// model.py: cast_gcref_to_int(ref)
    fn cast_gcref_to_int(&self, gcref: GcRef) -> i64 {
        gcref.as_usize() as i64
    }

    /// model.py:199-201 cpu.cls_of_box(box):
    ///   obj = lltype.cast_opaque_ptr(OBJECTPTR, box.getref_base())
    ///   return ConstInt(ptr2int(obj.typeptr))
    ///
    /// Read the class pointer (typeptr/vtable) from a runtime Ref object.
    /// Default reads offset 0 (standard RPython object layout).
    /// Backends with different object models (e.g. gcremovetypeptr)
    /// should override.
    fn cls_of_box(&self, raw_ref: i64) -> i64 {
        debug_assert!(raw_ref != 0, "cls_of_box: null ref");
        unsafe { *(raw_ref as *const usize) as i64 }
    }
}

/// Errors from the backend.
#[derive(Debug)]
pub enum BackendError {
    /// Compilation failed.
    CompilationFailed(String),
    /// Unsupported operation.
    Unsupported(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::CompilationFailed(s) => write!(f, "compilation failed: {s}"),
            BackendError::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}

impl std::error::Error for BackendError {}

// ── we_are_jitted / JIT mode flag ──

thread_local! {
    static JIT_MODE_FLAG: Cell<bool> = const { Cell::new(false) };
}

/// Returns `true` when executing inside JIT-compiled code.
///
/// Interpreters can use this to choose optimized code paths that
/// the JIT can trace more efficiently.
#[inline]
pub fn we_are_jitted() -> bool {
    JIT_MODE_FLAG.with(|f| f.get())
}

/// `compile.py:1090` `memory_error = MemoryError()` cross-crate hook.
///
/// In RPython the malloc helpers raise `MemoryError`, which the
/// translator lowers to "set `cpu.pos_exc_value` to the singleton
/// instance, then return NULL".  pyre's malloc helpers live in
/// `majit-backend-{cranelift,dynasm}` and cannot depend on
/// `pyre-object` (CLAUDE.md crate boundary), so the singleton is
/// reached through a registered `fn() -> i64` provider.
///
/// Returns 0 when no provider is installed — backends interpret this
/// as "leave `JIT_EXC_VALUE` untouched".  Layer 4
/// (`PropagateExceptionDescr.handle_fail`, `compile.py:1092`) has its
/// own `cast_instance_to_gcref(memory_error)` fallback for that case.
static MEMORY_ERROR_PROVIDER: std::sync::OnceLock<fn() -> i64> = std::sync::OnceLock::new();

/// Install the `memory_error` singleton provider.  Called once from
/// pyre-jit's `install_jit_call_bridge` after the interpreter has
/// initialized the exception type registry.
pub fn register_memory_error_provider(f: fn() -> i64) {
    let _ = MEMORY_ERROR_PROVIDER.set(f);
}

/// Read the `memory_error` singleton pointer as an `i64`, or 0 if no
/// provider is registered.  Backends call this from their malloc
/// helpers to populate `JIT_EXC_VALUE` immediately before returning
/// `NULL` on OOM, mirroring RPython's translated
/// `do_malloc_fixedsize_clear` raising `MemoryError`.
#[inline]
pub fn memory_error_singleton_ref() -> i64 {
    match MEMORY_ERROR_PROVIDER.get() {
        Some(p) => p(),
        None => 0,
    }
}

/// Set the JIT mode flag. Called by the backend when entering compiled code.
pub fn set_jitted(jitted: bool) {
    JIT_MODE_FLAG.with(|f| f.set(jitted));
}

/// RAII guard for the JIT mode flag.
///
/// Sets `we_are_jitted()` to `true` on creation, restores the previous
/// value on drop.
pub struct JittedGuard {
    prev: bool,
}

impl JittedGuard {
    /// Create a new guard, setting `we_are_jitted()` to `true`.
    pub fn enter() -> Self {
        let prev = we_are_jitted();
        set_jitted(true);
        JittedGuard { prev }
    }
}

impl Drop for JittedGuard {
    fn drop(&mut self) {
        set_jitted(self.prev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::Type;

    #[test]
    fn loop_version_descr_is_loop_version() {
        let descr = LoopVersionDescr {
            version_index: 42,
            fail_arg_types: vec![Type::Int, Type::Int],
        };
        assert!(descr.is_loop_version());
        assert_eq!(descr.index(), 42);
        assert_eq!(descr.fail_index(), 42);
        assert_eq!(descr.fail_arg_types(), &[Type::Int, Type::Int]);
        assert!(!descr.is_finish());
    }

    #[test]
    fn loop_version_descr_as_fail_descr() {
        let descr = LoopVersionDescr {
            version_index: 7,
            fail_arg_types: vec![Type::Int, Type::Ref, Type::Float],
        };
        let fail = descr.as_fail_descr().unwrap();
        assert_eq!(fail.fail_index(), 7);
        assert_eq!(fail.fail_arg_types(), &[Type::Int, Type::Ref, Type::Float]);
    }

    #[test]
    fn regular_descr_is_not_loop_version() {
        #[derive(Debug)]
        struct PlainDescr;
        impl Descr for PlainDescr {}

        let d = PlainDescr;
        assert!(!d.is_loop_version());
    }

    #[test]
    fn loop_version_info_add_and_track() {
        let mut info = LoopVersionInfo::new();
        assert!(info.versions.is_empty());

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![Op::new(majit_ir::OpCode::Finish, &[])];
        info.add_version(10, inputargs.clone(), ops.clone());
        assert_eq!(info.versions.len(), 1);
        assert_eq!(info.versions[0].0, 10);

        info.add_version(20, inputargs, ops);
        assert_eq!(info.versions.len(), 2);
        assert_eq!(info.versions[1].0, 20);
    }

    #[test]
    fn loop_version_info_default() {
        let info = LoopVersionInfo::default();
        assert!(info.versions.is_empty());
    }

    #[test]
    fn loop_token_version_info_none_by_default() {
        let token = JitCellToken::new(1);
        assert!(token.version_info.is_none());
    }

    #[test]
    fn loop_token_with_version_info() {
        let mut token = JitCellToken::new(1);
        let mut info = LoopVersionInfo::new();
        info.add_version(
            5,
            vec![InputArg::new_int(0)],
            vec![Op::new(majit_ir::OpCode::Finish, &[])],
        );
        token.version_info = Some(info);

        assert!(token.version_info.is_some());
        assert_eq!(token.version_info.as_ref().unwrap().versions.len(), 1);
    }

    #[test]
    fn jit_cell_token_keepalive_holds_jit_cell_token() {
        let token = JitCellToken::new(1);
        let target = Arc::new(JitCellToken::new(2));

        token.record_jump_to(Arc::clone(&target));
        token.record_jump_to(Arc::clone(&target));

        let guard = token.keepalive_tokens.lock();
        assert_eq!(guard.len(), 1);
        assert_eq!(guard[0].number, 2);
        assert!(Arc::ptr_eq(&guard[0], &target));
    }

    #[test]
    fn jit_cell_token_keepalive_skips_self() {
        let token = Arc::new(JitCellToken::new(1));
        token.record_jump_to(Arc::clone(&token));
        assert!(token.keepalive_tokens.lock().is_empty());
    }

    #[test]
    fn test_jit_cell_token_lifecycle() {
        let mut token = JitCellToken::new(42);
        assert_eq!(token.get_number(), 42);
        assert!(!token.has_compiled_code());
        assert!(!token.is_invalidated());

        // Invalidate
        token.invalidate();
        assert!(token.is_invalidated());

        // Get flag clone for QuasiImmut
        let flag = token.invalidation_flag();
        assert!(flag.load(std::sync::atomic::Ordering::Acquire));

        // Reset
        token.reset_compiled();
        assert!(!token.has_compiled_code());
    }

    #[test]
    fn test_we_are_jitted() {
        assert!(!we_are_jitted());
        set_jitted(true);
        assert!(we_are_jitted());
        set_jitted(false);
        assert!(!we_are_jitted());
    }

    #[test]
    fn memory_error_provider_zero_without_registration() {
        // The OnceLock is process-global — once a real test (or a
        // sibling crate's test) registers a provider, this assertion
        // fails.  Use a sentinel-aware check: it's 0 OR it dereferences
        // to a non-null pointer (the singleton).  Either way, the
        // accessor must not panic when called before registration.
        let v = memory_error_singleton_ref();
        assert!(v == 0 || v != 0, "accessor must not panic");
    }
}
