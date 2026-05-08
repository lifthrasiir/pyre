//! Narrow semantic graph scaffold for the future graph-based translator.
//!
//! This is intentionally much smaller than a full Rust compiler IR.  It exists
//! to model only the semantics needed by majit's translation/codewriter layer.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

use crate::flowspace::model::ConstValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ValueId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueType {
    Int,
    Ref,
    Float,
    Void,
    State,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnknownKind {
    /// `syn::Stmt::Macro` — any macro invocation at statement position
    /// (`panic!(...)`, `debug_assert!(...)`, `println!(...)`, etc).
    /// Always skipped in `lower_stmt` (no flow-graph analogue — RPython
    /// source has no macros).
    MacroStmt,
    /// `syn::Expr::Lit` with a kind pyre cannot yet model.  The `variant`
    /// tag names the specific syn literal kind (`Str`, `Float`,
    /// `ByteStr`, `Verbatim`) so downstream diagnostics and
    /// `MAJIT_UNKNOWN_DUMP` logs show the exact failure category
    /// without re-walking the syn AST.
    UnsupportedLiteral { variant: UnsupportedLiteralKind },
    /// `syn::Expr::*` variants pyre cannot yet lower.  Tag names the
    /// specific expression kind so diagnostics and the `Unknown`
    /// markers left in SSA graphs identify the remaining port gap.
    UnsupportedExpr { variant: UnsupportedExprKind },
}

/// Reason an `syn::Lit::*` could not be lowered.  Parity with RPython
/// `flowspace/flowcontext.py` — unsupported literals raise
/// `FlowingError` with a kind-specific message; pyre's analogue
/// records the kind on the Unknown marker op so the abort path is
/// traceable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnsupportedLiteralKind {
    Str,
    Float,
    ByteStr,
    Verbatim,
    Other,
}

/// Reason an `syn::Expr::*` could not be lowered.  Same role as
/// `UnsupportedLiteralKind` — tags the specific `FlowingError`-
/// equivalent abort cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnsupportedExprKind {
    Array,
    Async,
    Await,
    Const,
    ForLoop,
    Group,
    Infer,
    Let,
    Macro,
    Range,
    RawAddr,
    Repeat,
    Struct,
    Tuple,
    TryBlock,
    Verbatim,
    Yield,
    OtherExpr,
}

/// RPython `rpython/rtyper/rclass.py:57-60` — `IR_IMMUTABLE` / `IR_IMMUTABLE_ARRAY`
/// / `IR_QUASIIMMUTABLE` / `IR_QUASIIMMUTABLE_ARRAY`.  Parsed from
/// `_immutable_fields_` string literals (`rclass.py:644-678 _parse_field_list`):
///
/// * `"field"`       → `Immutable`
/// * `"field?"`      → `QuasiImmutable`
/// * `"field[*]"`    → `ImmutableArray`
/// * `"field?[*]"`   → `QuasiImmutableArray`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImmutableRank {
    Immutable,
    QuasiImmutable,
    ImmutableArray,
    QuasiImmutableArray,
}

impl ImmutableRank {
    /// Parse a single RPython-style `_immutable_fields_` entry.  Returns the
    /// bare field name and its rank.  Suffix precedence matches
    /// `rclass.py:649-661`: `?[*]` → `[*]` → `?` → plain.
    pub fn parse(entry: &str) -> (String, Self) {
        if let Some(stripped) = entry.strip_suffix("?[*]") {
            (stripped.to_string(), Self::QuasiImmutableArray)
        } else if let Some(stripped) = entry.strip_suffix("[*]") {
            (stripped.to_string(), Self::ImmutableArray)
        } else if let Some(stripped) = entry.strip_suffix('?') {
            (stripped.to_string(), Self::QuasiImmutable)
        } else {
            (entry.to_string(), Self::Immutable)
        }
    }

    /// RPython `ImmutableRanking.pure` flag — `rclass.py:33-37`.  True for
    /// `IR_IMMUTABLE` / `IR_IMMUTABLE_ARRAY`; false for the quasi variants
    /// (they pin via guard, not via pure flag).
    pub fn is_immutable(self) -> bool {
        matches!(self, Self::Immutable | Self::ImmutableArray)
    }

    /// True for `IR_QUASIIMMUTABLE` / `IR_QUASIIMMUTABLE_ARRAY` —
    /// `jtransform.py:895 immut in (IR_QUASIIMMUTABLE, IR_QUASIIMMUTABLE_ARRAY)`.
    pub fn is_quasi_immutable(self) -> bool {
        matches!(self, Self::QuasiImmutable | Self::QuasiImmutableArray)
    }

    /// True for `IR_IMMUTABLE_ARRAY` / `IR_QUASIIMMUTABLE_ARRAY` —
    /// `rclass.py:670 rank in (IR_QUASIIMMUTABLE_ARRAY, IR_IMMUTABLE_ARRAY)`.
    pub fn is_array(self) -> bool {
        matches!(self, Self::ImmutableArray | Self::QuasiImmutableArray)
    }
}

impl Default for ImmutableRank {
    fn default() -> Self {
        Self::Immutable
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CallTarget {
    Method {
        name: String,
        receiver_root: Option<String>,
    },
    FunctionPath {
        segments: Vec<String>,
    },
    /// Rust frontend adaptation for constructors that RPython's rtyper erases
    /// before jtransform. This variant must only be produced after frontend
    /// resolution proves the call is not a user-defined function.
    SyntheticTransparentCtor {
        name: String,
    },
    /// RPython: `indirect_call` opname. Receiver's static type is a
    /// `dyn Trait` (Rust fat pointer); at JIT time the actual callee
    /// is resolved via vtable.  `trait_root` + `method_name` together
    /// key the candidate family in `CallControl.trait_method_impls`.
    Indirect {
        trait_root: String,
        method_name: String,
    },
    UnsupportedExpr,
}

impl CallTarget {
    pub fn method(name: impl Into<String>, receiver_root: Option<String>) -> Self {
        Self::Method {
            name: name.into(),
            receiver_root,
        }
    }

    pub fn function_path<I, S>(segments: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::FunctionPath {
            segments: segments.into_iter().map(Into::into).collect(),
        }
    }

    pub fn indirect(trait_root: impl Into<String>, method_name: impl Into<String>) -> Self {
        Self::Indirect {
            trait_root: trait_root.into(),
            method_name: method_name.into(),
        }
    }

    pub fn synthetic_transparent_ctor(name: impl Into<String>) -> Self {
        Self::SyntheticTransparentCtor { name: name.into() }
    }

    pub fn receiver_root(&self) -> Option<&str> {
        match self {
            CallTarget::Method { receiver_root, .. } => receiver_root.as_deref(),
            _ => None,
        }
    }

    pub fn path_segments(&self) -> Option<Vec<&str>> {
        match self {
            CallTarget::Method { name, .. } => Some(vec![name.as_str()]),
            CallTarget::FunctionPath { segments } => {
                Some(segments.iter().map(String::as_str).collect())
            }
            CallTarget::SyntheticTransparentCtor { name } => Some(vec![name.as_str()]),
            CallTarget::Indirect { method_name, .. } => Some(vec![method_name.as_str()]),
            CallTarget::UnsupportedExpr => None,
        }
    }
}

impl fmt::Display for CallTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallTarget::Method {
                name,
                receiver_root: Some(receiver_root),
            } => write!(f, "{receiver_root}.{name}"),
            CallTarget::Method {
                name,
                receiver_root: None,
            } => f.write_str(name),
            CallTarget::FunctionPath { segments } => f.write_str(&segments.join("::")),
            CallTarget::SyntheticTransparentCtor { name } => {
                write!(f, "<synthetic-transparent-ctor {name}>")
            }
            CallTarget::Indirect {
                trait_root,
                method_name,
            } => write!(f, "<dyn {trait_root}>::{method_name}"),
            CallTarget::UnsupportedExpr => f.write_str("<unsupported-call-expr>"),
        }
    }
}

/// RPython call ops always carry `op.args[0]` as the funcptr operand.
/// Pyre keeps the same semantic slot but needs two Rust-level shapes:
/// a symbolic direct-call target or a runtime `ValueId` for indirect calls.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CallFuncPtr {
    Target(CallTarget),
    Value(ValueId),
}

/// RPython `flatten.py:53-57`:
///
/// ```python
/// class IndirectCallTargets(object):
///     def __init__(self, lst):
///         self.lst = lst       # list of JitCodes
/// ```
///
/// Sidecar attached to `OpKind::CallResidual` when the residual call is the
/// tail of a regular-indirect lowering (`jtransform.py:547`).  The assembler
/// merges the JitCode handles into `Assembler.indirectcalltargets` so the
/// metainterp can later look up jitcodes by function-pointer address during
/// runtime dispatch (`pyjitpl.py:2325-2343`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndirectCallTargets {
    /// `Arc<JitCode>` shells (identity-keyed via `JitCodeHandle`) for
    /// every candidate impl in the `(trait, method)` family.  Matches
    /// RPython `flatten.py:55` `lst # list of JitCodes` shape.
    pub lst: Vec<crate::jitcode::JitCodeHandle>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FieldDescriptor {
    pub name: String,
    pub owner_root: Option<String>,
}

impl FieldDescriptor {
    pub fn new(name: impl Into<String>, owner_root: Option<String>) -> Self {
        Self {
            name: name.into(),
            owner_root,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpKind {
    Input {
        name: String,
        ty: ValueType,
    },
    ConstInt(i64),
    /// RPython `flowmodel.py:Constant(rfloat)` — a float constant whose
    /// `concretetype` is `lltype.Float`.  Stored as the f64 bit pattern
    /// (`history.py:265 ConstFloat.getfloatstorage`) so PartialEq/Hash
    /// stay derivable.  The assembler materialises this through the
    /// existing `constants_f` pool with a `float_copy` op, mirroring
    /// the `ConstInt` → `int_copy` lowering.
    ConstFloat(u64),
    FieldRead {
        base: ValueId,
        field: FieldDescriptor,
        ty: ValueType,
        /// RPython `jtransform.py:867-903` may rewrite immutable /
        /// quasi-immutable reads to `getfield_*_pure`.  Carries the
        /// chosen opcode flavour through flatten/assembly so the
        /// runtime sees the `_pure` bytecode variant instead of having
        /// to rediscover purity from the descriptor later.
        #[serde(default)]
        pure: bool,
    },
    FieldWrite {
        base: ValueId,
        field: FieldDescriptor,
        value: ValueId,
        ty: ValueType,
    },
    ArrayRead {
        base: ValueId,
        index: ValueId,
        item_ty: ValueType,
        /// RPython: ARRAY identity for `cpu.arraydescrof(ARRAY)`.
        /// Distinguishes arrays with the same item_ty but different
        /// container types (e.g. `Vec<Point>` vs `Vec<Line>`).
        array_type_id: Option<String>,
    },
    ArrayWrite {
        base: ValueId,
        index: ValueId,
        value: ValueId,
        item_ty: ValueType,
        /// RPython: ARRAY identity for `cpu.arraydescrof(ARRAY)`.
        array_type_id: Option<String>,
    },
    /// RPython: getinteriorfield_gc_i/r/f — read a field of an array-of-structs element.
    /// effectinfo.py:313-325: generates "readinteriorfield" effect.
    /// effectinfo.py:327-340: also implicitly generates "readarray" effect.
    InteriorFieldRead {
        base: ValueId,
        index: ValueId,
        field: FieldDescriptor,
        item_ty: ValueType,
        array_type_id: Option<String>,
    },
    /// RPython: setinteriorfield_gc — write a field of an array-of-structs element.
    /// effectinfo.py:349-350: generates "interiorfield" effect.
    /// effectinfo.py:327-340: also implicitly generates "array" effect.
    InteriorFieldWrite {
        base: ValueId,
        index: ValueId,
        field: FieldDescriptor,
        value: ValueId,
        item_ty: ValueType,
        array_type_id: Option<String>,
    },
    Call {
        target: CallTarget,
        args: Vec<ValueId>,
        result_ty: ValueType,
    },
    GuardTrue {
        cond: ValueId,
    },
    GuardFalse {
        cond: ValueId,
    },

    // ── JIT-specific ops (generated by jtransform pass) ──────────
    /// Guard that a value equals a compile-time constant.
    /// RPython: `int_guard_value`, `ref_guard_value`, `float_guard_value`.
    /// Emitted by `promote_greens()` before `recursive_call`.
    GuardValue {
        value: ValueId,
        /// 'i', 'r', or 'f' — the kind of the guarded value.
        kind_char: char,
    },
    /// Project a callee function pointer out of a `dyn Trait` receiver's
    /// vtable for the named method slot.  Result is integer-typed so it
    /// can be fed to `int_guard_value` (RPython `jtransform.py:546`).
    ///
    /// PRE-EXISTING-ADAPTATION of `rclass.py:371-377 getclsfield()`. RPython
    /// emits a `cast_pointer + getfield(vtable_struct, method_name)` chain
    /// because `ClassRepr` models the vtable as an explicit `Struct`. Rust
    /// `dyn Trait` vtable layout is compiler-internal (unstable ABI), so
    /// pyre cannot model the vtable as an IR struct — this single op stands
    /// in for the chain and must be emitted by the rtyper-equivalent layer
    /// (`translator/rtyper/rclass.rs`), never by `jtransform`.
    VtableMethodPtr {
        receiver: ValueId,
        trait_root: String,
        method_name: String,
    },
    /// Indirect call — `op.args[0]` is the funcptr ValueId already produced
    /// by the rtyper layer (e.g. from `VtableMethodPtr` for `dyn Trait`
    /// dispatch). `args` are the full call arguments, including the
    /// receiver. `graphs` mirrors the trailing `c_graphs` constant from
    /// `rpbc.py:216`: `Some(full_family)` when known, `None` otherwise.
    ///
    /// RPython: `rpython/rtyper/rpbc.py:216-217`
    /// ```python
    /// vlist.append(hop.inputconst(Void, row_of_graphs.values()))
    /// v = hop.genop('indirect_call', vlist, resulttype=rresult)
    /// ```
    /// Lowered downstream by `jtransform.py:410-412 rewrite_op_indirect_call`.
    IndirectCall {
        funcptr: ValueId,
        args: Vec<ValueId>,
        graphs: Option<Vec<crate::parse::CallPath>>,
        result_ty: ValueType,
    },
    /// Virtualizable field read → reads from boxes, no heap op.
    /// RPython: `getfield_vable_i/r/f`
    VableFieldRead {
        base: ValueId,
        field_index: usize,
        ty: ValueType,
    },
    /// Virtualizable field write → writes to boxes, no heap op.
    /// RPython: `setfield_vable_i/r/f`
    VableFieldWrite {
        base: ValueId,
        field_index: usize,
        value: ValueId,
        ty: ValueType,
    },
    /// Virtualizable array read → reads from boxes.
    /// RPython: `getarrayitem_vable_i/r/f`
    VableArrayRead {
        base: ValueId,
        array_index: usize,
        elem_index: ValueId,
        item_ty: ValueType,
        /// RPython: arraydescr.itemsize from VirtualizableInfo.array_descrs.
        array_itemsize: usize,
        /// RPython: arraydescr.is_item_signed() from VirtualizableInfo.array_descrs.
        array_is_signed: bool,
    },
    /// Virtualizable array write → writes to boxes.
    /// RPython: `setarrayitem_vable_i/r/f`
    VableArrayWrite {
        base: ValueId,
        array_index: usize,
        elem_index: ValueId,
        value: ValueId,
        item_ty: ValueType,
        /// RPython: arraydescr.itemsize from VirtualizableInfo.array_descrs.
        array_itemsize: usize,
        /// RPython: arraydescr.is_item_signed() from VirtualizableInfo.array_descrs.
        array_is_signed: bool,
    },
    /// Binary arithmetic/comparison operation.
    /// RPython: `int_add`, `int_lt`, etc.
    BinOp {
        op: String,
        lhs: ValueId,
        rhs: ValueId,
        result_ty: ValueType,
    },
    /// Unary operation.
    /// RPython: `int_neg`, `bool_not`, etc.
    UnaryOp {
        op: String,
        operand: ValueId,
        result_ty: ValueType,
    },

    /// Force virtualizable: flush boxes to heap.
    /// RPython: `hint_force_virtualizable(vable)`
    VableForce {
        base: ValueId,
    },

    // ── Call effect classification (generated by jtransform) ────
    //
    // RPython jtransform.py:414-435 `rewrite_call()`: args are split by kind
    // into three ListOfKind lists. The opname encodes the kind signature:
    //   residual_call_ir_i  = int+ref args, int result
    //   call_elidable_r_v   = ref args, void result
    //
    /// Elidable (pure) call — no side effects, result depends only on args.
    /// RPython: `call_elidable_{kinds}_{reskind}(funcptr, calldescr, [i], [r], [f])`
    /// `funcptr` mirrors `op.args[0]` in RPython. Direct calls keep a
    /// symbolic target; indirect calls carry the runtime `ValueId`
    /// produced by rtype.
    CallElidable {
        funcptr: CallFuncPtr,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<ValueId>,
        args_r: Vec<ValueId>,
        args_f: Vec<ValueId>,
        result_kind: char,
    },
    /// Residual call — has side effects, must be preserved.
    /// RPython: `residual_call_{kinds}_{reskind}(funcptr, calldescr, [i], [r], [f])`.
    /// See `CallElidable` for `funcptr` semantics.
    /// `indirect_targets` mirrors the `IndirectCallTargets(lst)` sidecar
    /// that RPython appends to the extraargs for an indirect_call family
    /// (`jtransform.py:547`).  `None` for direct-call lowering.
    CallResidual {
        funcptr: CallFuncPtr,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<ValueId>,
        args_r: Vec<ValueId>,
        args_f: Vec<ValueId>,
        result_kind: char,
        indirect_targets: Option<IndirectCallTargets>,
    },
    /// May-force call — can trigger GC or force virtualizables.
    /// RPython: `call_may_force_{kinds}_{reskind}(funcptr, calldescr, [i], [r], [f])`.
    /// See `CallElidable` for `funcptr` semantics.
    CallMayForce {
        funcptr: CallFuncPtr,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<ValueId>,
        args_r: Vec<ValueId>,
        args_f: Vec<ValueId>,
        result_kind: char,
    },

    // ── Call kind classification (generated by jtransform via CallControl) ──
    //
    // RPython jtransform.py:414-435 `rewrite_call()`: args are split by kind
    // into three lists (int, ref, float). The opname encodes the kind signature:
    //   inline_call_ir_i  = int+ref args, int result
    //   residual_call_r_v = ref args, void result
    //
    // `result_kind`: 'i', 'r', 'f', or 'v' (RPython `getkind(result.concretetype)`)
    /// Inline call — callee is a regular candidate graph.
    /// RPython: `inline_call_{kinds}_{reskind}(jitcode, [i_args], [r_args], [f_args])`
    /// RPython jtransform.py:473-482.
    InlineCall {
        /// RPython: `callcontrol.get_jitcode(targetgraph)` returns the
        /// callee JitCode object itself, not its final `index`.
        /// pyre carries the same identity-bearing handle until the
        /// assembler snapshots the final descriptor table.
        jitcode: crate::jitcode::JitCodeHandle,
        /// Integer arguments (RPython: ListOfKind('int', ...))
        args_i: Vec<ValueId>,
        /// Reference arguments (RPython: ListOfKind('ref', ...))
        args_r: Vec<ValueId>,
        /// Float arguments (RPython: ListOfKind('float', ...))
        args_f: Vec<ValueId>,
        /// Result kind: 'i', 'r', 'f', or 'v'
        result_kind: char,
    },
    /// Recursive call — back to the portal entry point.
    /// RPython: `recursive_call_{reskind}(jd_index, [green_i], [green_r], [green_f], [red_i], [red_r], [red_f])`
    /// RPython jtransform.py:522-534.
    RecursiveCall {
        /// RPython: `jitdriver_sd.index`
        jd_index: usize,
        /// Green args (loop-invariant) split by kind
        greens_i: Vec<ValueId>,
        greens_r: Vec<ValueId>,
        greens_f: Vec<ValueId>,
        /// Red args (loop-variant) split by kind
        reds_i: Vec<ValueId>,
        reds_r: Vec<ValueId>,
        reds_f: Vec<ValueId>,
        /// Result kind
        result_kind: char,
    },

    // ── JIT builtin ops (jtransform.py:1731-1743) ────────────
    //
    // These correspond to RPython's `_handle_jit_call()` in jtransform.py.
    // The codewriter converts calls to `jit.*` oopspec functions into
    // dedicated opcodes instead of residual calls.
    /// jtransform.py:1731 — `jit_debug(string, arg1, arg2, arg3, arg4)`.
    /// Emits debug info into the trace (like debug_merge_point).
    JitDebug {
        args: Vec<ValueId>,
    },
    /// jtransform.py:1733 — `{kind}_assert_green(value)`.
    /// Asserts the value is compile-time constant during tracing.
    AssertGreen {
        value: ValueId,
        kind_char: char,
    },
    /// jtransform.py:1736 — `current_trace_length()`.
    /// Returns the current length of the trace being compiled.
    CurrentTraceLength,
    /// jtransform.py:1738 — `{kind}_isconstant(value)`.
    /// Returns whether the value is currently known to be constant.
    IsConstant {
        value: ValueId,
        kind_char: char,
    },
    /// jtransform.py:1741 — `{kind}_isvirtual(value)`.
    /// Returns whether the value is currently virtualized.
    IsVirtual {
        value: ValueId,
        kind_char: char,
    },

    // ── Conditional call ops (jtransform.py:1665-1688) ──────
    //
    // RPython: `jit_conditional_call` / `jit_conditional_call_value` llops
    // are rewritten to `conditional_call_{kinds}_{reskind}`.
    /// jtransform.py:1685 — `conditional_call_{ir}_{v}`.
    /// If condition is true, call the function. Always produces void.
    /// RPython: `COND_CALL(condition, funcptr, calldescr, args...)`
    ConditionalCall {
        condition: ValueId,
        funcptr: CallTarget,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<ValueId>,
        args_r: Vec<ValueId>,
        args_f: Vec<ValueId>,
    },
    /// jtransform.py:1687 — `conditional_call_value_{ir}_{reskind}`.
    /// If value is falsy (0/NULL/None), call the function and return its result.
    /// RPython: `COND_CALL_VALUE(value, funcptr, calldescr, args...)`
    ConditionalCallValue {
        value: ValueId,
        funcptr: CallTarget,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<ValueId>,
        args_r: Vec<ValueId>,
        args_f: Vec<ValueId>,
        result_kind: char,
    },

    /// jtransform.py:292-313 — `record_known_result_{i|r}_ir_v`.
    /// Produced by `rewrite_op_jit_record_known_result`; pairs an elidable call
    /// with its known result for constant folding by OptPure.
    /// RPython layout: `record_known_result_{reskind}(result, funcptr, calldescr, [i], [r])`
    RecordKnownResult {
        /// The known result value (arg 0 of the jit_record_known_result llop).
        result_value: ValueId,
        funcptr: CallTarget,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<ValueId>,
        args_r: Vec<ValueId>,
        args_f: Vec<ValueId>,
        /// 'i' or 'r' — kind of the known result (no float support).
        result_kind: char,
    },

    /// RPython `record_quasiimmut_field(v_inst, fielddescr, mutatefielddescr)`
    /// — `jtransform.py:901-903`.  Emitted by `rewrite_op_getfield` when the
    /// field is quasi-immutable; paired with a subsequent pure-read.  The
    /// metainterp/blackhole counterpart (`blackhole.py:1537-1539
    /// bhimpl_record_quasiimmut_field`) is a no-op during blackhole execution
    /// but the descriptors drive guard/invalidation accounting in the
    /// optimizer (`quasiimmut.py`).
    ///
    /// PRE-EXISTING-ADAPTATION: RPython derives `mutate_field` via
    /// `quasiimmut.get_mutate_field_name(name)` which expects the lltype
    /// `inst_` prefix (`quasiimmut.py:11-15`).  Rust structs have no such
    /// prefix, so we use the literal `mutate_<fieldname>` convention.
    RecordQuasiImmutField {
        base: ValueId,
        field: FieldDescriptor,
        mutate_field: FieldDescriptor,
    },

    /// Liveness marker — RPython `-live-` operation.
    /// Inserted by jtransform after calls that may need guard resumption.
    /// Expanded by compute_liveness() to include all values alive at
    /// this point. RPython: jtransform.py:469,481,533.
    Live,

    /// JitDriver merge-point marker — RPython `jit_merge_point` opname.
    /// Emitted by `handle_jit_marker__jit_merge_point` (jtransform.py:1690-1712)
    /// with the portal jitdriver's index and green/red arguments split by
    /// kind (`make_three_lists`). In upstream the args are a flat
    /// `SpaceOperation.args` vec `[index_const, greens_i, greens_r,
    /// greens_f, reds_i, reds_r, reds_f]`; pyre stores them as structured
    /// fields to avoid re-splitting on every consumer.
    JitMergePoint {
        jitdriver_index: usize,
        greens_i: Vec<ValueId>,
        greens_r: Vec<ValueId>,
        greens_f: Vec<ValueId>,
        reds_i: Vec<ValueId>,
        reds_r: Vec<ValueId>,
        reds_f: Vec<ValueId>,
    },

    /// JitDriver loop-header marker — RPython `loop_header` opname.
    /// Emitted by `handle_jit_marker__loop_header` (jtransform.py:1714-1718)
    /// with the jitdriver's index as its single Constant arg.
    /// `can_enter_jit` markers alias to this (jtransform.py:1723).
    LoopHeader {
        jitdriver_index: usize,
    },

    /// pyre-only marker emitted by the front-end (`front/ast.rs`
    /// `continue_with_unknown*` / `stop_unsupported`) when a syntactic
    /// form cannot be lowered to a canonical opname.  Reaching the op at
    /// runtime means tracing or blackhole resume crossed an
    /// untranslatable graph slice; downstream handlers advance past it
    /// (see `blackhole.rs::handler_abort_marker_pyre`).  Distinct from
    /// RPython's `SwitchToBlackhole` exception path — RPython aborts
    /// before lowering so no equivalent opname exists.  Kept under
    /// `kind: UnknownKind` because the same diagnostic enum is reused
    /// by `FlowingError::Unsupported` (`front/ast.rs:53`).
    Abort {
        kind: UnknownKind,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpaceOperation {
    pub result: Option<ValueId>,
    pub kind: OpKind,
}

/// RPython `Block.exitswitch`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExitSwitch {
    Value(ValueId),
    LastException,
}

/// RPython `Link.exitcase`.
///
/// Upstream stores the concrete switch value itself here: `False` /
/// `True` for boolean branches, the Python `Exception` class object for
/// catch-all exception links, or a specific exception class object for
/// typed handlers (`flowspace/model.py:114-120`,
/// `flowspace/flowcontext.py:127-143`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitCase {
    Bool(bool),
    Const(ConstValue),
}

/// RPython `flowspace/model.py:109-168` `Link`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    pub args: Vec<LinkArg>,
    pub target: BlockId,
    pub exitcase: Option<ExitCase>,
    /// RPython `Link.prevblock` — the block this Link exits from.
    pub prevblock: Option<BlockId>,
    /// RPython `Link.llexitcase` — the low-level value matched by
    /// `goto_if_exception_mismatch` (`flatten.py:228-231`).  For
    /// typed exception links this is the rtyper-produced class
    /// identity constant; pyre carries it as a full `ConstValue` so
    /// non-Int llexitcase shapes (`lltype.Ptr`, host class objects)
    /// round-trip to the backend intact.
    pub llexitcase: Option<ConstValue>,
    pub last_exception: Option<LinkArg>,
    pub last_exc_value: Option<LinkArg>,
}

impl Link {
    pub fn new(args: Vec<ValueId>, target: BlockId, exitcase: Option<ExitCase>) -> Self {
        Self::new_mixed(
            args.into_iter().map(LinkArg::from).collect(),
            target,
            exitcase,
        )
    }

    pub fn new_mixed(args: Vec<LinkArg>, target: BlockId, exitcase: Option<ExitCase>) -> Self {
        Self {
            args,
            target,
            exitcase,
            prevblock: None,
            llexitcase: None,
            last_exception: None,
            last_exc_value: None,
        }
    }

    pub fn with_prevblock(mut self, prevblock: BlockId) -> Self {
        self.prevblock = Some(prevblock);
        self
    }

    pub fn with_llexitcase(mut self, llexitcase: ConstValue) -> Self {
        self.llexitcase = Some(llexitcase);
        self
    }

    /// Structural counterpart of RPython rtyper's
    /// `convert_link()` (`rtyper.py:1338`): copy the flow-level
    /// `exitcase` into the low-level `llexitcase` slot for primitive
    /// branch values.  The `"default"` switch sentinel is not a
    /// low-level case and stays `None`.
    pub fn with_llexitcase_from_exitcase(mut self) -> Self {
        self.llexitcase = match &self.exitcase {
            Some(ExitCase::Bool(value)) => Some(ConstValue::Bool(*value)),
            Some(ExitCase::Const(value)) if value.string_eq("default") => None,
            Some(ExitCase::Const(value)) => Some(value.clone()),
            None => None,
        };
        self
    }

    pub fn extravars(
        mut self,
        last_exception: Option<LinkArg>,
        last_exc_value: Option<LinkArg>,
    ) -> Self {
        self.last_exception = last_exception;
        self.last_exc_value = last_exc_value;
        self
    }

    /// RPython `flatten.py:224` `if link.exitcase is Exception`.
    pub fn catches_all_exceptions(&self) -> bool {
        self.exitcase == Some(exception_exitcase())
    }
}

pub fn exception_exitcase() -> ExitCase {
    ExitCase::Const(ConstValue::builtin("Exception"))
}

/// RPython `Link.args` items are Variables or Constants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkArg {
    Value(ValueId),
    Const(ConstValue),
}

impl LinkArg {
    pub fn as_value(&self) -> Option<ValueId> {
        match self {
            Self::Value(value) => Some(*value),
            Self::Const(_) => None,
        }
    }
}

impl From<ValueId> for LinkArg {
    fn from(value: ValueId) -> Self {
        Self::Value(value)
    }
}

impl From<ConstValue> for LinkArg {
    fn from(value: ConstValue) -> Self {
        Self::Const(value)
    }
}

/// A basic block in the control flow graph.
///
/// RPython equivalent: `flowspace/model.py:171-180 Block` — slots
/// `inputargs operations exitswitch exits`.  Upstream has no separate
/// terminator surface: fall-through is `exitswitch=None` with a single
/// `Link`, bool branches are `exitswitch=Variable` with two Links
/// carrying `Bool(false)`/`Bool(true)` exitcases, can-raise is
/// `exitswitch=c_last_exception`, and final blocks have `exits=()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub id: BlockId,
    /// Phi-node inputs: values provided by incoming Links.
    /// RPython: `Block.inputargs` — each predecessor Link carries
    /// values that map 1:1 to these inputargs.
    pub inputargs: Vec<ValueId>,
    pub operations: Vec<SpaceOperation>,
    /// RPython `Block.exitswitch`.
    pub exitswitch: Option<ExitSwitch>,
    /// RPython `Block.exits`.
    pub exits: Vec<Link>,
    /// Framestate snapshot captured at the moment this block was closed
    /// (its `set_goto` / `set_branch` / `set_return`).  Read by the
    /// frontend's lazy cross-block local installer to recover, after the
    /// fact, what `(name → ValueId)` mapping was visible to a
    /// predecessor when its outgoing link fired.  Build-time only —
    /// downstream rtyper/jtransform passes ignore this field; it is
    /// neither serialised nor required to remain populated past the
    /// front end.  RPython parity: `flowspace/flowcontext.py:38
    /// SpamBlock.__init__ self.framestate = framestate` attaches a
    /// framestate to the block representing the locals state visible at
    /// that block boundary; pyre stores the analogous "exit-time"
    /// snapshot here so cross-block reads in successors can derive
    /// `Link.args` retroactively.
    pub framestate: Option<FrameState>,
}

/// One slot of a `FrameState` snapshot — a single locally-bound
/// name, the `ValueId` it pointed at when the snapshot was taken, and
/// the `ValueType` that classifies the kind register bank.
///
/// RPython parity: corresponds to one slot of `frame.locals_w` at the
/// moment `flowspace/flowcontext.py` closes a block.  RPython keys
/// `locals_w` by CPython-assigned slot index; pyre carries the slot
/// name explicitly so the same `FrameStateEntry` can be located across
/// merge points without consulting a separate slot table — slot order
/// is enforced by the position of the entry in the containing
/// `FrameState.entries` Vec (first-bind positional, see `FrameState`
/// docstring).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameStateEntry {
    pub name: String,
    pub value_id: ValueId,
    pub value_type: ValueType,
}

/// Locals snapshot captured at a block boundary — see `Block.framestate`
/// for usage semantics.  Entries are stored **densely** at graph-wide
/// first-bind slot positions: slot `i` corresponds to the name at
/// `GraphBuildContext::local_first_bind_order[i]`, and `None` indicates
/// the slot is unbound at this snapshot point (RPython's "undefined
/// local" sentinel).  Names are appended to the graph-wide order on
/// first bind anywhere in the function and never moved or removed —
/// `LocalBindingSnapshot::restore` does NOT roll the order back, so the
/// slot index of a name is truly invariant across the entire lowering.
/// This mirrors RPython `co_varnames` slot order — every predecessor's
/// exit snapshot walks the same slot positions, so two predecessors of
/// the same merge point line up positionally even when one of them
/// rolled back its bindings via `LocalBindingSnapshot::restore`.
///
/// RPython parity: `flowspace/framestate.py:18 FrameState` — a tuple of
/// `(locals_w, stack, last_exception, blocklist, next_offset)`.  Pyre's
/// frontend runs over Rust source rather than Python bytecode, so
/// `stack` / `last_exception` / `blocklist` / `next_offset` have no
/// analogue here and only the `locals_w` slice is meaningful.  This
/// is a partial projection of upstream's FrameState — the bytecode
/// flowspace's full counterpart lives at
/// `flowspace::framestate::FrameState`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameState {
    /// Slot `i` ↔ graph-wide first-bind name at index `i`; `None` if
    /// unbound at this snapshot point.  RPython parity:
    /// `framestate.py:19 self.locals_w` — list of `Variable | Constant
    /// | None` indexed by `co_varnames` slot.
    pub entries: Vec<Option<FrameStateEntry>>,
}

impl FrameState {
    /// Iterate `(name, ValueId, &ValueType)` over **bound** slots in
    /// storage order.  Unbound slots (None-killed) are skipped.
    pub fn iter(&self) -> impl Iterator<Item = (&str, ValueId, &ValueType)> + '_ {
        self.entries
            .iter()
            .filter_map(|e| e.as_ref())
            .map(|e| (e.name.as_str(), e.value_id, &e.value_type))
    }

    /// `Link.args` payload for this snapshot — `ValueId`s of bound
    /// slots in storage order.  Unbound (None) slots are skipped.
    #[allow(dead_code)]
    pub fn link_args(&self) -> Vec<ValueId> {
        self.entries
            .iter()
            .filter_map(|e| e.as_ref().map(|e| e.value_id))
            .collect()
    }

    /// Compute a state at least as general as both `self` and `other`
    /// via positional zip over slots — direct port of RPython
    /// `framestate.py:14 _union` (`return [union(v1, v2) for v1, v2
    /// in zip(seq1, seq2)]`) and `framestate.py:73-90 FrameState.union`.
    /// When one side has fewer slots than the other (a name appended
    /// after the shorter side's snapshot was taken), the missing
    /// slots are treated as `None` so the positional zip extends to
    /// `max(len(self), len(other))`.
    ///
    /// Per-slot semantics (RPython `framestate.py:105-128 union`):
    ///   - `(None, _) | (_, None)` → None-kill (`framestate.py:
    ///     110-111`); merged slot is `None`.
    ///   - `(Some(s), Some(o))` with `s.value_id == o.value_id` →
    ///     carry through that vid (`framestate.py:108 if w1 == w2:
    ///     return w1`).
    ///   - `(Some(s), Some(o))` with disagreeing vids →
    ///     `graph.alloc_value()` for a fresh `ValueId` at the slot
    ///     (`framestate.py:113-114 return Variable()` analogue).
    ///
    /// `ValueType` disagreement is partitioned into two regimes:
    ///   - `Unknown` is a wildcard: a slot bound by inference on one
    ///     arm and by annotation on the other is structurally one
    ///     slot, not a conflict, so the merged kind adopts the
    ///     concrete side.  Mirrors PyPy where flowspace `Variable`
    ///     carries no kind at all — annotation- and inference-driven
    ///     types unify trivially because the common ancestor is the
    ///     untyped `Variable` itself.
    ///   - **Concrete vs concrete** with disagreeing kinds (e.g.
    ///     `Int` vs `Ref`) returns `Err(UnionError::TypeMismatch)`.
    ///     RPython parity: `flowspace/framestate.py:88 FrameState.union`
    ///     wraps `_union` in `try/except UnionError: return None`, and
    ///     the caller `flowcontext.py:430-436 mergeblock` reads the
    ///     `None` as "this candidate did not unify, generalize the
    ///     existing block / retry / make_next_block".  The signal is a
    ///     **whole-state failure**, not a per-slot drop — silently
    ///     dropping one slot would let post-merge reads of the dropped
    ///     name surface as undefined-local instead of blocking the
    ///     merge.  Pyre's single-candidate static AST lowering has no
    ///     retry surface, so production callers `.expect(...)` the
    ///     `Result` after asserting rustc's type checker has already
    ///     rejected sources that bind the same local to two different
    ///     concrete kinds across arms; the `Result` exists so the
    ///     parity contract stays honest at the model layer and tests
    ///     that bypass the Rust frontend can exercise the failure
    ///     path explicitly via `expect_err`.
    ///
    /// Returns `FrameState` directly with phi vids materialised:
    /// agreement slots carry the predecessor's vid, disagreement
    /// slots get a freshly-allocated vid (the upstream `Variable()`
    /// analogue).  Callers detect "this slot is a fresh phi" by
    /// comparing the merged `value_id` against the predecessor's vid
    /// for the same slot — when they differ, the union allocated a
    /// fresh vid.  The install is then driven via
    /// `lazy_install_local_at_current_block(.., Some(merged_vid))`
    /// so the Input op carries the same vid the merged state already
    /// refers to.
    pub fn union(
        &self,
        other: &FrameState,
        graph: &mut FunctionGraph,
    ) -> Result<FrameState, UnionError> {
        // Two-pass implementation: Pass 1 validates every slot (no
        // graph mutation), Pass 2 allocates fresh vids for
        // disagreement slots.  This keeps `graph.next_value()` from
        // advancing when a later slot trips `UnionError::TypeMismatch`
        // — atomic on failure, parity-equivalent to upstream where
        // `framestate.py:73-89 try/except UnionError: return None`
        // discards a partially-built result.
        let len = std::cmp::max(self.entries.len(), other.entries.len());
        // Pass 1: per-slot decisions without mutating `graph`.
        // `Some(Ok(entry))` → carry-through (vid known).
        // `Some(Err((name, value_type)))` → fresh vid required (Pass 2 allocates).
        // `None` → None-kill.
        #[allow(clippy::type_complexity)]
        let mut tentative: Vec<Option<Result<FrameStateEntry, (String, ValueType)>>> =
            Vec::with_capacity(len);
        for i in 0..len {
            let s = self.entries.get(i).and_then(|e| e.as_ref());
            let o = other.entries.get(i).and_then(|e| e.as_ref());
            match (s, o) {
                // `framestate.py:110-111`: one side None → None-kill.
                (None, _) | (_, None) => tentative.push(None),
                (Some(s_entry), Some(o_entry)) => {
                    debug_assert_eq!(
                        s_entry.name, o_entry.name,
                        "graph-wide first-bind invariant broken at slot {i}"
                    );
                    let value_type = match (&s_entry.value_type, &o_entry.value_type) {
                        (a, b) if a == b => a.clone(),
                        (ValueType::Unknown, b) => b.clone(),
                        (a, ValueType::Unknown) => a.clone(),
                        _ => {
                            return Err(UnionError::TypeMismatch {
                                name: s_entry.name.clone(),
                                self_type: s_entry.value_type.clone(),
                                other_type: o_entry.value_type.clone(),
                            });
                        }
                    };
                    if s_entry.value_id == o_entry.value_id {
                        // `framestate.py:108-109 if w1 == w2: return w1`
                        tentative.push(Some(Ok(FrameStateEntry {
                            name: s_entry.name.clone(),
                            value_id: s_entry.value_id,
                            value_type,
                        })));
                    } else {
                        // `framestate.py:113-114 return Variable()` —
                        // defer ValueId allocation until Pass 2.
                        tentative.push(Some(Err((s_entry.name.clone(), value_type))));
                    }
                }
            }
        }
        // Pass 2: commit allocations now that Pass 1 succeeded.
        let merged: Vec<Option<FrameStateEntry>> = tentative
            .into_iter()
            .map(|t| match t {
                None => None,
                Some(Ok(entry)) => Some(entry),
                Some(Err((name, value_type))) => Some(FrameStateEntry {
                    name,
                    value_id: graph.alloc_value(),
                    value_type,
                }),
            })
            .collect();
        Ok(FrameState { entries: merged })
    }

    /// Output arguments to thread `self` (a predecessor's exit state)
    /// into the merge block whose entry framestate is `target`.  RPython
    /// parity: `framestate.py:92-99 FrameState.getoutputargs` walks
    /// `targetstate.mergeable` positionally, picking `self.mergeable[i]`
    /// at every Variable position.  `target.entries` is dense at the
    /// graph-wide first-bind slot positions, with `None` at slots that
    /// were None-killed by the union; surviving (`Some`) slot `i` lines
    /// up with `self.entries[i]` because both predecessors share the
    /// same graph-wide first-bind order.
    ///
    /// Panics if `target` contains a `Some` slot at an index where
    /// `self.entries[i]` is `None` or out of range — that would mean
    /// `target` was not produced by `self.union(_)`, which violates the
    /// merge invariant.
    pub fn getoutputargs(&self, target: &FrameState) -> Vec<ValueId> {
        target
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                slot.as_ref().map(|t| {
                    let s = self
                        .entries
                        .get(i)
                        .and_then(|e| e.as_ref())
                        .expect("target slot must be bound in self — union invariant");
                    debug_assert_eq!(
                        s.name, t.name,
                        "graph-wide first-bind invariant broken at slot {i}"
                    );
                    s.value_id
                })
            })
            .collect()
    }
}

/// Remove dead operations and dead inputargs from `graph` per
/// backward dataflow over operation operands + exitswitches +
/// `Link.args`-as-dependencies.  Line-by-line port of
/// `simplify.transform_dead_op_vars_in_blocks(blocks, graphs,
/// translator=None)` (`rpython/translator/simplify.py:422-524`).
///
/// `blocks` is the BFS-reachable closure of every entry block (mirrors
/// `flowspace/model.py:66 iterblocks()`).
///
/// PRE-EXISTING-ADAPTATION: `start_blocks` is `{graph.startblock} ∪
/// {blocks with no incoming link}` rather than the strict single-graph
/// `{graphs[0].startblock}` of `simplify.py:428`.  The `generated::*`
/// pipeline emits closures / specialisations as secondary entries
/// (function parameters at a non-zero block id with no predecessors)
/// that are calling-convention contracts but are not registered with
/// the simplification pass as separate graphs.  Pinning them as
/// additional starts matches the *intent* of `simplify.py:431-433`'s
/// multi-graph branch (which pins one start per graph in the input
/// set) without requiring a `translator.annotator` to enumerate the
/// secondary entries.  Convergence: surface the secondary entries via
/// an explicit `start_blocks: HashSet<BlockId>` parameter or migrate
/// `generated::*` to register secondary entries with the simplifier
/// directly.
///
///   1. Walk every reachable block.  For each operation, evaluate
///      `canremove(op, block)` per `simplify.py:435-436`:
///      `op.opname in CanRemove and op is not block.raising_op`,
///      where `block.raising_op` is `block.operations[-1]` whenever
///      `block.canraise` (`flowspace/model.py:218-221`).  Pyre's
///      `Block::canraise()` mirrors the upstream
///      `block.exitswitch is c_last_exception` check; the raising
///      op is the last entry in `block.operations`.
///        - If `is_pure_op(kind)` AND the op is not the raising_op
///          AND `op.result.is_some()`: route operands via
///          `dependencies[op.result] += operands`
///          (`simplify.py:444-445 dependencies[op.result].
///          update(op.args)` for `canremove`-classified ops).
///          Operands become live only if `op.result` becomes live.
///        - Otherwise: operands join `read_vars` immediately
///          (`simplify.py:442-443 read_vars.update(op.args)`
///          for non-`canremove` ops).  Raising-op operands therefore
///          stay live unconditionally — the raise side-effect is
///          observable.
///      The `exitswitch` (if a Variable) joins `read_vars`.  For
///      each block with no exits (`returnblock` / `exceptblock` /
///      otherwise terminal), the inputargs are also added — return
///      and except blocks implicitly use their inputs
///      (`simplify.py:459-462`).
///   2. For every Link, record `dependencies[targetarg].add(linkarg)`
///      mapping (`simplify.py:457-458`).  This is the key
///      difference from a naïve forward pass — link args are NOT
///      direct reads, they are dependencies whose liveness flows
///      backward from a live target inputarg.  A self-referential
///      phi (back-edge `Link.args[i] == phi_i`) is therefore kept
///      alive only if `phi_i` itself is reached by some external
///      reader; without one, the dependency cycle dies.
///   3. The `startblock`'s inputargs are pinned to `read_vars` as a
///      calling-convention contract (`simplify.py:463-467`,
///      single-graph case `simplify.py:428-429 start_blocks =
///      {graphs[0].startblock}`).  `returnblock` / `exceptblock`
///      contracts are already covered by the empty-exits rule
///      above.
///   4. Backward-flow `read_vars` over `dependencies` to a fixpoint
///      (`simplify.py:471-479 flow_read_var_backward`).
///   5. Drop every pure op whose result is not in `read_vars`
///      (`simplify.py:484-488` removable-op drop), gated by the
///      same `canremove(op, block)` predicate from Step 1 — the
///      raising op of a `canraise` block is preserved here even if
///      its result is dead.  Inputarg-shaped `OpKind::Input` ops
///      are protected by Step 1+3+dependency-routing pinning their
///      result vid in `read_vars`; the dead ones are swept here
///      alongside Step 7's inputarg trim (the matching
///      `block.inputargs[i]` removal becomes a no-op when the
///      Input op is already gone).  Naked `OpKind::Input` ops
///      (legacy frontend fallback for cross-block reads not yet
///      routed through the lazy installer) are likewise removed
///      when their result vid is dead — `simplify.py` itself has
///      no standalone Input shape, but the line-by-line port
///      sweeps them under the same canremove rule.
///
///      PRE-EXISTING-ADAPTATION (translator-gated arms not ported).
///      Upstream's Step 5 has two further `elif` arms after the
///      canremove drop:
///        - `simplify.py:489-499 elif op.opname == 'simple_call'`
///          removes `simple_call(builtin, ...)` whose first arg is
///          a `Constant` whose value is in `CanRemoveBuiltins`
///          (`simplify.py:418-420 = {hasattr: True}`).  The
///          `FunctionGraph` IR pyre's prune_dead_phis operates on
///          carries call shapes as `OpKind::Call` /
///          `OpKind::IndirectCall` etc., not as `simple_call`-named
///          ops, and pyre's frontend never emits a Call to `hasattr`
///          — the arm is structurally inapplicable.
///        - `simplify.py:500-506 elif op.opname == 'direct_call'`
///          removes the call when `has_no_side_effects(translator,
///          graph) and op is not block.raising_op`.  Gated on
///          `translator is not None`; pyre's caller passes
///          `translator=None`, so the arm is unreachable.  The
///          `op is not block.raising_op` clause is moot for the
///          same reason.
///      Convergence path: surface a translator-equivalent (callee
///      effect classifier) and route `OpKind::Call`-with-elidable-EI
///      through this Step's drop path.
///   6. For each reachable block, walk its exits and drop
///      `Link.args[i]` for every `i` where `target.inputargs[i]`
///      is not in `read_vars` — inputarg-side trim happens in a
///      second pass to preserve the `len(link.args) ==
///      len(link.target.inputargs)` invariant (`simplify.py:
///      512-516`, the explicit assertion at `:513`).
///   7. Walk each reachable non-terminal block once more and drop
///      `block.inputargs[i]` (and its matching `OpKind::Input` op
///      if still present in `block.operations`) for every dead
///      vid (`simplify.py:520-524`).
pub fn prune_dead_phis(graph: &mut FunctionGraph) {
    use crate::inline::{is_pure_op, op_value_refs};
    use std::collections::HashMap;
    let start = graph.startblock;
    let return_block = graph.returnblock;
    let except_block = graph.exceptblock;
    let block_index: HashMap<BlockId, usize> = graph
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    // `start_blocks` mirrors `simplify.py:431-433`'s multi-graph branch:
    // `graph.startblock` plus every block with no incoming link
    // (calling-convention entries the `generated::*` pipeline emits
    // for closures / specialisations).  Each is a BFS root, so `blocks`
    // (the union of `iterblocks()` over each entry) covers every block
    // reachable from any entry.
    let with_predecessor: HashSet<BlockId> = graph
        .blocks
        .iter()
        .flat_map(|b| b.exits.iter().map(|e| e.target))
        .collect();
    let start_blocks: HashSet<BlockId> = std::iter::once(start)
        .chain(
            graph
                .blocks
                .iter()
                .map(|b| b.id)
                .filter(|id| !with_predecessor.contains(id)),
        )
        .collect();

    // BFS reachability mirrors `flowspace/model.py:66 iterblocks()`,
    // unioned across `start_blocks`:
    //
    //     def iterblocks(self):
    //         block = self.startblock
    //         yield block
    //         seen = {block: True}
    //         stack = list(block.exits[::-1])
    //         while stack:
    //             block = stack.pop().target
    //             if block not in seen:
    //                 yield block
    //                 seen[block] = True
    //                 stack += block.exits[::-1]
    let reachable: HashSet<BlockId> = {
        let mut seen: HashSet<BlockId> = HashSet::new();
        let mut stack: Vec<BlockId> = Vec::new();
        for &sb in &start_blocks {
            if seen.insert(sb) {
                if let Some(&i) = block_index.get(&sb) {
                    stack.extend(graph.blocks[i].exits.iter().rev().map(|e| e.target));
                }
            }
        }
        while let Some(target) = stack.pop() {
            if seen.insert(target) {
                if let Some(&i) = block_index.get(&target) {
                    stack.extend(graph.blocks[i].exits.iter().rev().map(|e| e.target));
                }
            }
        }
        seen
    };

    // Step 1 + 2: gather initial `read_vars` (op operands of non-pure
    // ops + exitswitches + terminal-block inputargs) and
    // `dependencies` (target inputarg ← link arg, plus pure-op
    // operands routed via `op.result`).
    //
    // `simplify.py:435-436 canremove`:
    //
    //     def canremove(op, block):
    //         return op.opname in CanRemove and op is not block.raising_op
    //
    // `block.raising_op` (`flowspace/model.py:218-221`) is
    // `block.operations[-1]` whenever `block.canraise` (i.e.
    // `block.exitswitch is c_last_exception`).  Pure-classified
    // ops parked there as overflow / zero-divide checks
    // (`int_add_ovf`, `int_floordiv_zer`, ...) MUST NOT be DCE'd
    // even if their result vid is unread — the raise side-effect
    // is observable.
    let mut read_vars: HashSet<ValueId> = HashSet::new();
    let mut dependencies: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    for block in &graph.blocks {
        if !reachable.contains(&block.id) {
            continue;
        }
        let raising_op_idx = if block.canraise() && !block.operations.is_empty() {
            Some(block.operations.len() - 1)
        } else {
            None
        };
        for (i, op) in block.operations.iter().enumerate() {
            let operands = op_value_refs(&op.kind);
            // `simplify.py:441-445`:
            //   if not canremove(op, block):    read_vars.update(args)
            //   else:                           dependencies[result] += args
            let removable = is_pure_op(&op.kind) && Some(i) != raising_op_idx;
            if let Some(result) = op.result
                && removable
            {
                dependencies.entry(result).or_default().extend(operands);
            } else {
                for vid in operands {
                    read_vars.insert(vid);
                }
            }
        }
        if let Some(ExitSwitch::Value(vid)) = &block.exitswitch {
            read_vars.insert(*vid);
        }
        // `simplify.py:459-462`: terminal blocks (no exits)
        // implicitly use every inputarg.
        if block.exits.is_empty() {
            for &iarg in &block.inputargs {
                read_vars.insert(iarg);
            }
        }
        for link in &block.exits {
            // `simplify.py:512-513` len-equality is asserted at the
            // exits-walk; the dependency-routing pass exercises the
            // same invariant by zipping `link.args` with
            // `link.target.inputargs`.
            let &target_idx = block_index
                .get(&link.target)
                .expect("simplify.py:512 — link.target must be a graph block");
            let target_block = &graph.blocks[target_idx];
            assert_eq!(
                link.args.len(),
                target_block.inputargs.len(),
                "simplify.py:513 — len(link.args) == len(link.target.inputargs)",
            );
            for (arg, &target_iarg) in link.args.iter().zip(target_block.inputargs.iter()) {
                if let LinkArg::Value(arg_vid) = arg {
                    dependencies.entry(target_iarg).or_default().push(*arg_vid);
                }
            }
        }
    }
    // Step 3: pin every `start_blocks` inputarg (multi-graph branch
    // `simplify.py:431-433`, with `graph.startblock` and orphan-entry
    // blocks treated as additional starts per the function-level doc
    // comment).
    for &sb in &start_blocks {
        if let Some(&i) = block_index.get(&sb) {
            for &iarg in &graph.blocks[i].inputargs {
                read_vars.insert(iarg);
            }
        }
    }

    // Step 4: backward flow.
    // `simplify.py:471-479 flow_read_var_backward`.
    let mut pending: Vec<ValueId> = read_vars.iter().copied().collect();
    while let Some(var) = pending.pop() {
        if let Some(deps) = dependencies.get(&var) {
            for &dep in deps {
                if read_vars.insert(dep) {
                    pending.push(dep);
                }
            }
        }
    }

    // Step 5: drop every pure op whose result is dead.
    // `simplify.py:484-488 if op.result not in read_vars: if
    // canremove(op, block): del block.operations[i]`.
    // Inputarg-shaped Input ops keep their result vid in `read_vars`
    // via Step 1+3+dependency-routing, so the live ones survive
    // unconditionally; dead ones get removed alongside their
    // matching inputarg trim in Step 7.  Naked Input ops (the
    // legacy fallback at `front/ast.rs` for cross-block reads not
    // yet routed through `lazy_install_local_at_current_block`) do
    // NOT have inputarg pinning, so the same `result not in
    // read_vars` predicate retires them — `simplify.py` itself has
    // no standalone Input op shape, but the line-by-line port
    // sweeps them under the same canremove rule.
    //
    // The reverse `for i in range(len-1, -1, -1)` walk
    // (`simplify.py:484`) is preserved: we resolve `raising_op_idx`
    // before the loop and check `Some(i) != raising_op_idx` so a
    // pure op that happens to be the block's raising_op
    // (`canremove`'s `op is not block.raising_op` clause,
    // `simplify.py:436`) survives even when its result is unread.
    for block in &mut graph.blocks {
        if !reachable.contains(&block.id) {
            continue;
        }
        let raising_op_idx = if block.canraise() && !block.operations.is_empty() {
            Some(block.operations.len() - 1)
        } else {
            None
        };
        for i in (0..block.operations.len()).rev() {
            let op = &block.operations[i];
            let dead = match op.result {
                Some(r) => !read_vars.contains(&r),
                None => false,
            };
            if dead && is_pure_op(&op.kind) && Some(i) != raising_op_idx {
                block.operations.remove(i);
            }
        }
    }

    // Step 6: trim Link.args at indices whose target inputarg is dead.
    // `simplify.py:512-516`.  Walk in reverse so removals don't shift
    // surviving indices.
    for block_idx in 0..graph.blocks.len() {
        if !reachable.contains(&graph.blocks[block_idx].id) {
            continue;
        }
        let exits_len = graph.blocks[block_idx].exits.len();
        for exit_idx in 0..exits_len {
            let target = graph.blocks[block_idx].exits[exit_idx].target;
            let target_iargs = {
                let &i = block_index
                    .get(&target)
                    .expect("simplify.py:512 — link.target must be a graph block");
                graph.blocks[i].inputargs.clone()
            };
            assert_eq!(
                graph.blocks[block_idx].exits[exit_idx].args.len(),
                target_iargs.len(),
                "simplify.py:513 — len(link.args) == len(link.target.inputargs)",
            );
            for i in (0..target_iargs.len()).rev() {
                let target_vid = target_iargs[i];
                if !read_vars.contains(&target_vid) {
                    graph.blocks[block_idx].exits[exit_idx].args.remove(i);
                }
            }
        }
    }

    // Step 7: drop dead inputargs + their matching `OpKind::Input` ops.
    // `simplify.py:520-524`.  Walk every reachable block; startblock is
    // already pinned in Step 3, return / except blocks in Step 1.
    for block_idx in 0..graph.blocks.len() {
        let block_id = graph.blocks[block_idx].id;
        if !reachable.contains(&block_id) || block_id == return_block || block_id == except_block {
            continue;
        }
        let inputargs = graph.blocks[block_idx].inputargs.clone();
        for i in (0..inputargs.len()).rev() {
            let vid = inputargs[i];
            if read_vars.contains(&vid) {
                continue;
            }
            graph.blocks[block_idx].inputargs.remove(i);
            if let Some(op_idx) = graph.blocks[block_idx]
                .operations
                .iter()
                .position(|op| matches!(op.kind, OpKind::Input { .. }) && op.result == Some(vid))
            {
                graph.blocks[block_idx].operations.remove(op_idx);
            }
        }
    }
}

/// Reasons `FrameState::union` may refuse to
/// merge — pyre's analogue of RPython `framestate.py:101 UnionError`.
/// PyPy `framestate.py:88 FrameState.union` wraps `_union` in
/// `try/except UnionError: return None`; `flowcontext.py:430-436
/// mergeblock` reads the `None` as a whole-state "this candidate did
/// not unify, generalize the existing block / retry / make_next_block"
/// signal.  Pyre's single-candidate static AST lowering has no retry
/// surface, so production callers `.expect(...)` the `Err` arm after
/// asserting rustc's type checker has already rejected sources that
/// bind the same local to two different concrete kinds across arms;
/// the `Result` exists so the parity contract stays honest at the
/// model layer and tests that bypass the Rust frontend can exercise
/// the failure path explicitly via `expect_err`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnionError {
    /// Two predecessor states disagree on a slot's `ValueType` (e.g.
    /// one side bound the slot to an `Int`, the other to a `Ref`).
    /// The merge cannot be lowered onto a single int/ref/float
    /// register bank, so the join is untypable.  RPython surfaces
    /// this through `framestate.py:117 raise UnionError` for
    /// `SpecTag` constants; pyre extends the case to mismatched
    /// kinds because pyre's `ValueType` is what pins register bank
    /// classification.
    TypeMismatch {
        name: String,
        self_type: ValueType,
        other_type: ValueType,
    },
}

impl Block {
    pub fn canraise(&self) -> bool {
        matches!(self.exitswitch, Some(ExitSwitch::LastException))
    }

    /// RPython `flowspace/model.py:247 closeblock` / `:250 recloseblock`
    /// mark a block's exits tuple as populated.  Pyre mirrors the
    /// "has this block been closed?" predicate by checking that either
    /// `exits` has at least one `Link` or `exitswitch` is set.
    /// During graph construction, an unclosed block has
    /// `exits=[]`, `exitswitch=None` — the upstream equivalent of
    /// `type(block.exits) is list` pre-`closeblock`.
    pub fn is_closed(&self) -> bool {
        !self.exits.is_empty() || self.exitswitch.is_some()
    }

    /// Complement of `is_closed` — true if the front-end has not yet
    /// stamped a terminator / exits onto the block.  Used to gate
    /// fall-through code that adds the block's own exit.
    pub fn is_open(&self) -> bool {
        !self.is_closed()
    }
}

/// RPython `flowspace/model.py:238-244 renamevariables` applies a
/// value renaming to `inputargs` / `operations` / `exitswitch` /
/// `link.args`.  Pyre threads the renamer out-of-band here so callers
/// can reshape both the exitswitch variable and every exit link in one
/// call.
pub fn remap_control_flow_metadata<FValue, FBlock>(
    exitswitch: &Option<ExitSwitch>,
    exits: &[Link],
    remap_value: FValue,
    remap_block: FBlock,
) -> (Option<ExitSwitch>, Vec<Link>)
where
    FValue: Fn(ValueId) -> ValueId,
    FBlock: Fn(BlockId) -> BlockId,
{
    let exitswitch = exitswitch.as_ref().map(|switch| match switch {
        ExitSwitch::Value(value) => ExitSwitch::Value(remap_value(*value)),
        ExitSwitch::LastException => ExitSwitch::LastException,
    });
    let exits = exits
        .iter()
        .map(|link| Link {
            args: link
                .args
                .iter()
                .map(|arg| match arg {
                    LinkArg::Value(value) => LinkArg::Value(remap_value(*value)),
                    LinkArg::Const(value) => LinkArg::Const(value.clone()),
                })
                .collect(),
            target: remap_block(link.target),
            exitcase: link.exitcase.clone(),
            prevblock: link.prevblock.map(&remap_block),
            llexitcase: link.llexitcase.clone(),
            last_exception: link.last_exception.as_ref().map(|arg| match arg {
                LinkArg::Value(value) => LinkArg::Value(remap_value(*value)),
                LinkArg::Const(value) => LinkArg::Const(value.clone()),
            }),
            last_exc_value: link.last_exc_value.as_ref().map(|arg| match arg {
                LinkArg::Value(value) => LinkArg::Value(remap_value(*value)),
                LinkArg::Const(value) => LinkArg::Const(value.clone()),
            }),
        })
        .collect();
    (exitswitch, exits)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionGraph {
    pub name: String,
    pub startblock: BlockId,
    /// RPython `flowspace/model.py:17-18 FunctionGraph.returnblock` —
    /// `Block([return_var])` with `operations=()` and `exits=()`.
    /// Blocks returning a value route to it via
    /// `Link([value], returnblock)` held in `Block.exits`.
    pub returnblock: BlockId,
    /// RPython `FunctionGraph.exceptblock` — `Block([etype, evalue])`.
    pub exceptblock: BlockId,
    pub blocks: Vec<Block>,
    pub notes: Vec<String>,
    next_value: usize,
    /// Variable names for debugging (RPython Variable._name).
    pub value_names: std::collections::HashMap<ValueId, String>,
}

impl FunctionGraph {
    pub fn new(name: impl Into<String>) -> Self {
        let entry = BlockId(0);
        let returnblock = BlockId(1);
        let exceptblock = BlockId(2);
        let return_value = ValueId(0);
        let last_exception = ValueId(1);
        let last_exc_value = ValueId(2);
        Self {
            name: name.into(),
            startblock: entry,
            returnblock,
            exceptblock,
            // RPython `flowspace/model.py:14-25 FunctionGraph.__init__`:
            //   startblock created empty; returnblock = Block([return_var]);
            //   exceptblock = Block([Variable('etype'), Variable('evalue')]).
            // Final blocks have `operations=()` and `exits=()`; fall-through
            // for the startblock is likewise `exits=[]` until the front-end
            // closes it.
            blocks: vec![
                Block {
                    id: entry,
                    inputargs: Vec::new(),
                    operations: Vec::new(),
                    exitswitch: None,
                    exits: Vec::new(),
                    framestate: None,
                },
                Block {
                    id: returnblock,
                    inputargs: vec![return_value],
                    operations: Vec::new(),
                    exitswitch: None,
                    exits: Vec::new(),
                    framestate: None,
                },
                Block {
                    id: exceptblock,
                    inputargs: vec![last_exception, last_exc_value],
                    operations: Vec::new(),
                    exitswitch: None,
                    exits: Vec::new(),
                    framestate: None,
                },
            ],
            notes: Vec::new(),
            next_value: 3,
            value_names: std::collections::HashMap::new(),
        }
    }

    /// Return the canonical exception block and its `(etype, evalue)`
    /// inputargs.
    ///
    /// RPython parity: `flowspace/model.py:21-25` `exceptblock` has
    /// two inputargs, `(etype, evalue)`, and exists eagerly on every
    /// graph.
    pub fn exceptblock_args(&self) -> (BlockId, ValueId, ValueId) {
        let args = &self.block(self.exceptblock).inputargs;
        (self.exceptblock, args[0], args[1])
    }

    /// Return the canonical return block and its single inputarg.
    ///
    /// RPython parity: `FunctionGraph.getreturnvar()` reads
    /// `graph.returnblock.inputargs[0]`.
    pub fn returnblock_arg(&self) -> (BlockId, ValueId) {
        let args = &self.block(self.returnblock).inputargs;
        (self.returnblock, args[0])
    }

    pub fn create_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len());
        self.blocks.push(Block {
            id,
            inputargs: Vec::new(),
            operations: Vec::new(),
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
        });
        id
    }

    /// Create a block with explicit inputargs (Phi nodes).
    pub fn create_block_with_args(&mut self, num_args: usize) -> (BlockId, Vec<ValueId>) {
        let id = BlockId(self.blocks.len());
        let args: Vec<ValueId> = (0..num_args).map(|_| self.alloc_value()).collect();
        self.blocks.push(Block {
            id,
            inputargs: args.clone(),
            operations: Vec::new(),
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
        });
        (id, args)
    }

    pub fn alloc_value(&mut self) -> ValueId {
        let id = ValueId(self.next_value);
        self.next_value += 1;
        id
    }

    /// Read-only view of the ValueId allocator cursor.  Used by passes
    /// that need to mint fresh ValueIds outside the graph (e.g.
    /// `Transformer::allocate_synthetic_value` in `jtransform.rs`).
    pub fn next_value(&self) -> usize {
        self.next_value
    }

    /// Re-seat the ValueId allocator cursor.  Must be called after a
    /// pass that synthesized values outside the graph so subsequent
    /// `alloc_value()` calls do not collide.
    pub fn set_next_value(&mut self, next: usize) {
        debug_assert!(
            next >= self.next_value,
            "set_next_value must not walk the cursor backward: {} -> {}",
            self.next_value,
            next,
        );
        self.next_value = next;
    }

    pub fn push_op(&mut self, block: BlockId, kind: OpKind, has_result: bool) -> Option<ValueId> {
        let result = has_result.then(|| self.alloc_value());
        self.blocks[block.0]
            .operations
            .push(SpaceOperation { result, kind });
        result
    }

    /// Push an op with a caller-supplied `result` ValueId.  Used when
    /// the vid was allocated upstream (e.g. `FrameState::union`
    /// pre-allocates phi vids) and the op now needs to be emitted with
    /// the same vid.
    pub fn push_op_with_result(&mut self, block: BlockId, kind: OpKind, result: ValueId) {
        self.blocks[block.0].operations.push(SpaceOperation {
            result: Some(result),
            kind,
        });
    }

    /// RPython `flowspace/model.py:250 recloseblock(*exits)` — stamp
    /// `link.prevblock` on each exit and install them as the block's
    /// exits, overwriting any previous contents.  The `exitswitch`
    /// field is passed alongside so callers who set a bool branch or
    /// can-raise shape keep both halves of the canonical CFG surface
    /// updated in a single call (pyre-side ergonomics; upstream writes
    /// `block.exitswitch = ...` before `closeblock`).
    pub fn set_control_flow_metadata(
        &mut self,
        block: BlockId,
        exitswitch: Option<ExitSwitch>,
        exits: Vec<Link>,
    ) {
        let block_ref = &mut self.blocks[block.0];
        block_ref.exitswitch = exitswitch;
        block_ref.exits = exits
            .into_iter()
            .map(|link| link.with_prevblock(block))
            .collect();
    }

    /// RPython `flowspace/model.py:250 recloseblock(*exits)` — stamp
    /// `link.prevblock` on each exit and install them as the block's
    /// exits, overwriting any previous contents.  Like upstream, this
    /// does not touch `exitswitch`; callers that want to change the
    /// branch/raise discriminator must set it separately before
    /// `closeblock`/`recloseblock`.
    pub fn recloseblock(&mut self, block: BlockId, exits: Vec<Link>) {
        self.blocks[block.0].exits = exits
            .into_iter()
            .map(|link| link.with_prevblock(block))
            .collect();
    }

    /// RPython `flowspace/model.py:246 closeblock(*exits)` —
    /// `assert self.exits == [], "block already closed"` before
    /// delegating to `recloseblock`.  Keep the invariant as a regular
    /// assert, not `debug_assert!`, so release builds match upstream's
    /// fail-fast behavior.
    pub fn closeblock(&mut self, block: BlockId, exits: Vec<Link>) {
        assert!(
            self.blocks[block.0].exits.is_empty(),
            "block {:?} already closed",
            block
        );
        self.recloseblock(block, exits);
    }

    /// Shorthand for the single-exit fall-through shape — one Link to
    /// `target` carrying `args`, `exitswitch = None`.  Upstream
    /// equivalent: `block.closeblock(Link(args, target))`
    /// (`flowspace/model.py:304`).
    pub fn set_goto(&mut self, block: BlockId, target: BlockId, args: Vec<ValueId>) {
        self.set_control_flow_metadata(block, None, vec![Link::new(args, target, None)]);
    }

    /// Shorthand for the boolean-branch shape — two Links with
    /// `Bool(false)` / `Bool(true)` exitcases, `exitswitch =
    /// ExitSwitch::Value(cond)`.  Upstream equivalent:
    /// `block.exitswitch = cond;
    ///  block.closeblock(Link(false_args, if_false, False),
    ///                   Link(true_args,  if_true,  True))`
    /// (`flowspace/model.py:175-180` + `:304`).
    pub fn set_branch(
        &mut self,
        block: BlockId,
        cond: ValueId,
        if_true: BlockId,
        true_args: Vec<ValueId>,
        if_false: BlockId,
        false_args: Vec<ValueId>,
    ) {
        self.set_control_flow_metadata(
            block,
            Some(ExitSwitch::Value(cond)),
            vec![
                Link::new(false_args, if_false, Some(ExitCase::Bool(false)))
                    .with_llexitcase_from_exitcase(),
                Link::new(true_args, if_true, Some(ExitCase::Bool(true)))
                    .with_llexitcase_from_exitcase(),
            ],
        );
    }

    /// Route a return through the graph's canonical `returnblock`.
    ///
    /// RPython `flowcontext.py` return handling produces a fresh
    /// prevblock-side Variable (Void Variable for `return None`), then
    /// builds a Link carrying that value into the returnblock's
    /// inputargs.  pyre's codewriter adaptation mirrors that shape: a
    /// `None` `value` allocates a fresh prevblock-side ValueId whose
    /// kind defaults to Void (no regalloc color, no emitted move), so
    /// `Link.args` is always a prevblock value per upstream's
    /// `flowspace/model.py:114` invariant.
    pub fn set_return(&mut self, block: BlockId, value: Option<ValueId>) {
        let (returnblock, _) = self.returnblock_arg();
        let value = value.unwrap_or_else(|| self.alloc_value());
        self.set_goto(block, returnblock, vec![value]);
    }

    /// Route `block` to the graph's canonical `exceptblock` — the
    /// upstream-shaped exit for an unrecoverable exception.
    ///
    /// RPython `flowspace/model.py:21-25` declares
    /// `exceptblock = Block([Variable('etype'), Variable('evalue')])` as
    /// the single raise destination per graph.  Predecessor blocks route
    /// to it via `Link(args=[etype, evalue], target=exceptblock)` held in
    /// `Block.exits` — there is no upstream terminator variant that
    /// flags "this block raises".
    ///
    /// Emits the same CFG shape as upstream: a Link to
    /// `graph.exceptblock` with two fresh prevblock-side ValueIds
    /// standing in for the `(etype, evalue)` pair.  The `_reason`
    /// string is retained for optional GraphTransformNote annotations
    /// (see `jtransform.rs::rewrite_graph`'s abort note); pass `""`
    /// when not applicable.
    ///
    /// Used only where no concrete exception payload is available at
    /// the raise site — the evaluated arguments path should call
    /// `set_raise_values` instead.
    pub fn set_raise(&mut self, block: BlockId, _reason: &str) {
        let (exceptblock, _, _) = self.exceptblock_args();
        let etype = self.alloc_value();
        let evalue = self.alloc_value();
        self.set_goto(block, exceptblock, vec![etype, evalue]);
    }

    /// Terminate `block` with a Link to `exceptblock` carrying the
    /// caller-provided `(etype, evalue)` pair.
    ///
    /// RPython `flowspace/flowcontext.py:1253 Raise.nomoreblocks`:
    /// ```python
    /// raise FSException(self.w_type, self.w_value)
    /// ```
    /// The two `W_*` values flow from the preceding `RAISE_VARARGS`
    /// at `flowspace/flowcontext.py:638-656`, where `popvalue()`s
    /// provide the exception type / value already on the stack.
    /// This API lets pyre's macro lowering thread the evaluated
    /// ValueIds into the exceptblock Link so the exception payload
    /// is preserved, not discarded in favor of fresh `alloc_value()`
    /// placeholders as `set_raise` does.
    pub fn set_raise_values(&mut self, block: BlockId, etype: ValueId, evalue: ValueId) {
        let (exceptblock, _, _) = self.exceptblock_args();
        self.set_goto(block, exceptblock, vec![etype, evalue]);
    }

    pub fn block(&self, block: BlockId) -> &Block {
        &self.blocks[block.0]
    }

    /// Name a value (RPython Variable._name).
    pub fn name_value(&mut self, id: ValueId, name: impl Into<String>) {
        self.value_names.insert(id, name.into());
    }

    /// Get the name of a value, if any.
    pub fn value_name(&self, id: ValueId) -> Option<&str> {
        self.value_names.get(&id).map(|s| s.as_str())
    }

    pub fn block_mut(&mut self, block: BlockId) -> &mut Block {
        &mut self.blocks[block.0]
    }

    // ── RPython FunctionGraph iteration methods ──────────────────

    /// Iterate all blocks. RPython: `graph.iterblocks()`.
    pub fn iter_blocks(&self) -> impl Iterator<Item = &Block> {
        self.blocks.iter()
    }

    /// Iterate all (block, op) pairs. RPython: `graph.iterblockops()`.
    pub fn iter_block_ops(&self) -> impl Iterator<Item = (&Block, &SpaceOperation)> {
        self.blocks
            .iter()
            .flat_map(|b| b.operations.iter().map(move |op| (b, op)))
    }

    /// Get successor block IDs for a block.
    ///
    /// RPython `flowspace/model.py:66-76 FunctionGraph.iterblocks`:
    /// successor set is derived from `Block.exits` only.  Final blocks
    /// (`exits == ()`) — returnblock / exceptblock — have no successors.
    pub fn successors(&self, block: BlockId) -> Vec<BlockId> {
        self.block(block)
            .exits
            .iter()
            .map(|link| link.target)
            .collect()
    }

    /// Get predecessor block IDs for a block.
    pub fn predecessors(&self, target: BlockId) -> Vec<BlockId> {
        self.blocks
            .iter()
            .filter(|b| self.successors(b.id).contains(&target))
            .map(|b| b.id)
            .collect()
    }

    /// Count total operations across all blocks.
    pub fn num_ops(&self) -> usize {
        self.blocks.iter().map(|b| b.operations.len()).sum()
    }

    /// Check if a block is a loop header (has a back-edge predecessor).
    pub fn is_loop_header(&self, block: BlockId) -> bool {
        self.predecessors(block)
            .iter()
            .any(|&pred| pred.0 >= block.0)
    }

    /// Pretty-print the graph (RPython `graph.show()`).
    pub fn dump(&self) -> String {
        let mut out = format!(
            "=== {} ({} blocks, {} ops) ===\n",
            self.name,
            self.blocks.len(),
            self.num_ops()
        );
        for block in &self.blocks {
            let args: Vec<String> = block.inputargs.iter().map(|v| self.fmt_value(*v)).collect();
            if args.is_empty() {
                out.push_str(&format!("  Block {}:\n", block.id.0));
            } else {
                out.push_str(&format!("  Block {}({}):\n", block.id.0, args.join(", ")));
            }
            for op in &block.operations {
                let result = op
                    .result
                    .map(|v| format!("{} = ", self.fmt_value(v)))
                    .unwrap_or_default();
                out.push_str(&format!("    {}{:?}\n", result, op.kind));
            }
            // Upstream `flowspace/model.py:199 __repr__` prints the block
            // shape as "block@N with K exits[(exitswitch)]".  Mirror the
            // same summary from pyre's canonical exitswitch/exits pair.
            match &block.exitswitch {
                Some(switch) => out.push_str(&format!(
                    "    → {} exits ({:?})\n",
                    block.exits.len(),
                    switch
                )),
                None => out.push_str(&format!("    → {} exits\n", block.exits.len())),
            }
        }
        out
    }

    fn fmt_value(&self, id: ValueId) -> String {
        if let Some(name) = self.value_name(id) {
            format!("v{}:{}", id.0, name)
        } else {
            format!("v{}", id.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_allocates_values_and_blocks() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let cond = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let next = graph.create_block();
        graph.set_branch(entry, cond, next, vec![], next, vec![]);
        assert_eq!(graph.blocks.len(), 4);
        assert_eq!(graph.block(entry).operations.len(), 1);
        assert_eq!(graph.block(graph.returnblock).inputargs.len(), 1);
        assert_eq!(graph.block(graph.exceptblock).inputargs.len(), 2);
    }

    #[test]
    fn set_control_flow_metadata_stamps_prevblock() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let next = graph.create_block();
        graph.set_control_flow_metadata(entry, None, vec![Link::new(vec![], next, None)]);
        assert_eq!(graph.block(entry).exits[0].prevblock, Some(entry));
    }

    #[test]
    fn set_return_routes_non_void_returns_via_returnblock() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let value = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(value));
        // Upstream `flowspace/model.py:171-180` identifies the routed
        // return by Block.exits carrying a single Link(value, returnblock)
        // with exitswitch=None.
        let entry_block = graph.block(entry);
        assert!(entry_block.exitswitch.is_none());
        assert_eq!(entry_block.exits.len(), 1);
        assert_eq!(entry_block.exits[0].prevblock, Some(entry));
        assert_eq!(entry_block.exits[0].target, graph.returnblock);
        assert_eq!(entry_block.exits[0].args, vec![LinkArg::from(value)]);
    }

    #[test]
    fn recloseblock_preserves_existing_exitswitch() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let cond = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "cond".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let target = graph.create_block();
        graph.block_mut(entry).exitswitch = Some(ExitSwitch::Value(cond));

        graph.recloseblock(entry, vec![Link::new(vec![], target, None)]);

        assert_eq!(graph.block(entry).exitswitch, Some(ExitSwitch::Value(cond)));
        assert_eq!(graph.block(entry).exits[0].prevblock, Some(entry));
    }

    #[test]
    #[should_panic(expected = "already closed")]
    fn closeblock_panics_when_called_twice() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let first = graph.create_block();
        let second = graph.create_block();

        graph.closeblock(entry, vec![Link::new(vec![], first, None)]);
        graph.closeblock(entry, vec![Link::new(vec![], second, None)]);
    }

    #[test]
    fn framestate_union_kills_one_sided_local() {
        // RPython parity: `framestate.py:110-111` "if w1 is None or w2
        // is None: return None" — a slot present in one predecessor
        // but missing in the other is killed (dropped) from the merged
        // state.  Pyre realises this via dense positional entries:
        // the graph-wide first-bind order assigns slots [x=0,
        // self_only=1, other_only=2]; `self` has slot 1 bound but slot
        // 2 unbound, `other` has slot 2 bound but slot 1 unbound, so
        // both one-sided slots collapse to None-kill at union.
        let self_state = FrameState {
            entries: vec![
                Some(FrameStateEntry {
                    name: "x".into(),
                    value_id: ValueId(0),
                    value_type: ValueType::Int,
                }),
                Some(FrameStateEntry {
                    name: "self_only".into(),
                    value_id: ValueId(1),
                    value_type: ValueType::Int,
                }),
                None,
            ],
        };
        let other_state = FrameState {
            entries: vec![
                Some(FrameStateEntry {
                    name: "x".into(),
                    value_id: ValueId(2),
                    value_type: ValueType::Int,
                }),
                None,
                Some(FrameStateEntry {
                    name: "other_only".into(),
                    value_id: ValueId(3),
                    value_type: ValueType::Int,
                }),
            ],
        };
        let mut graph = FunctionGraph::new("test");
        // Reserve vid space past the test fixtures' hardcoded vids so
        // a fresh allocation for the disagreement on slot 0 picks an
        // id distinguishable from both predecessors.
        graph.set_next_value(100);
        let merged = self_state
            .union(&other_state, &mut graph)
            .expect("type-compatible slots must union cleanly");
        assert_eq!(
            merged.entries.len(),
            3,
            "merged state preserves positional length = max(len)"
        );
        let surviving = merged.entries[0]
            .as_ref()
            .expect("slot 0 (x) is bound on both sides and must survive");
        assert_eq!(surviving.name, "x");
        // Slot 0's predecessors disagreed on `value_id` (0 vs 2), so
        // `union` allocated a fresh vid via `graph.alloc_value()`.
        assert_ne!(surviving.value_id, ValueId(0));
        assert_ne!(surviving.value_id, ValueId(2));
        assert_eq!(surviving.value_id, ValueId(100));
        assert_eq!(surviving.value_type, ValueType::Int);
        assert!(
            merged.entries[1].is_none() && merged.entries[2].is_none(),
            "one-sided slots must be None-killed positionally"
        );
        assert!(
            merged.entries.iter().all(|e| e
                .as_ref()
                .map(|s| s.name != "self_only" && s.name != "other_only")
                .unwrap_or(true)),
            "one-sided slot names must not appear among surviving entries"
        );
    }

    /// Convenience helper: install an `OpKind::Input` op AND register its
    /// vid as a phi inputarg on the same block.  Mirrors what the
    /// Slice 2 eager-phi installer at union sites does.
    fn install_phi(graph: &mut FunctionGraph, block: BlockId, name: &str) -> ValueId {
        let vid = graph
            .push_op(
                block,
                OpKind::Input {
                    name: name.to_string(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.block_mut(block).inputargs.push(vid);
        vid
    }

    #[test]
    fn prune_dead_phis_drops_orphan_phi_and_link_arg() {
        // entry → merge (phi 'x' never read) → returnblock(void)
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let const_v = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();
        let merge = graph.create_block();
        graph.set_goto(entry, merge, vec![const_v]);
        install_phi(&mut graph, merge, "x");
        graph.set_return(merge, None);

        prune_dead_phis(&mut graph);

        assert!(
            graph.block(merge).inputargs.is_empty(),
            "orphan phi inputarg must be removed"
        );
        let entry_exit = &graph.block(entry).exits[0];
        assert!(
            entry_exit.args.is_empty(),
            "predecessor link arg matching the pruned phi must be removed"
        );
        let has_phi_op = graph
            .block(merge)
            .operations
            .iter()
            .any(|op| matches!(&op.kind, OpKind::Input { name, .. } if name == "x"));
        assert!(!has_phi_op, "orphan phi `OpKind::Input` must be dropped");
    }

    #[test]
    fn prune_dead_phis_keeps_phi_with_reader() {
        // entry → merge(phi 'x' read by a BinOp whose result is the
        // function return value) → returnblock(reads return value).
        // The BinOp's result is genuinely live — it flows through the
        // returnblock's terminal-inputarg pin — so backward dataflow
        // marks it `read_vars`, then propagates back through the
        // pure-op dependencies entry to phi_x, keeping phi_x alive.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let const_v = graph.push_op(entry, OpKind::ConstInt(7), true).unwrap();
        let merge = graph.create_block();
        graph.set_goto(entry, merge, vec![const_v]);
        let phi_x = install_phi(&mut graph, merge, "x");
        // BinOp whose result IS read (by `set_return`).  Backward
        // dataflow needs a live consumer of the BinOp result for the
        // pure-op-args→dependencies routing to keep phi_x alive.
        let doubled = graph
            .push_op(
                merge,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: phi_x,
                    rhs: phi_x,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(merge, Some(doubled));

        prune_dead_phis(&mut graph);

        assert_eq!(
            graph.block(merge).inputargs,
            vec![phi_x],
            "phi with a live downstream reader must stay"
        );
        let entry_exit = &graph.block(entry).exits[0];
        assert_eq!(
            entry_exit.args.len(),
            1,
            "predecessor link arg matching a kept phi must stay"
        );
    }

    #[test]
    fn prune_dead_phis_collapses_dead_phi_feeding_dead_pure_op() {
        // A phi whose only "reader" is a pure op whose result is itself
        // dead.  Both must collapse together — `simplify.py:441-445`'s
        // `dependencies[op.result] += op.args` for `canremove` ops
        // means the phi vid never enters `read_vars` (because the
        // pure op's result never enters either), and Step 5's blanket
        // pure-op DCE removes the orphan op so the phi can be safely
        // trimmed at Step 7.
        //
        // entry → merge(phi 'x', dead-result BinOp reading phi_x) →
        // returnblock(void).
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let const_v = graph.push_op(entry, OpKind::ConstInt(7), true).unwrap();
        let merge = graph.create_block();
        graph.set_goto(entry, merge, vec![const_v]);
        let phi_x = install_phi(&mut graph, merge, "x");
        let doubled = graph
            .push_op(
                merge,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: phi_x,
                    rhs: phi_x,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        // Void return — `doubled` has no live consumer.
        graph.set_return(merge, None);

        prune_dead_phis(&mut graph);

        assert!(
            graph.block(merge).inputargs.is_empty(),
            "phi feeding only a dead pure op must collapse"
        );
        let has_binop = graph
            .block(merge)
            .operations
            .iter()
            .any(|op| op.result == Some(doubled) && matches!(op.kind, OpKind::BinOp { .. }));
        assert!(!has_binop, "dead pure op must be removed alongside the phi");
        let entry_exit = &graph.block(entry).exits[0];
        assert!(
            entry_exit.args.is_empty(),
            "predecessor link arg matching the pruned phi must be removed"
        );
    }

    #[test]
    fn prune_dead_phis_collapses_chained_dead_phis() {
        // entry → merge1(phi 'x') → merge2(phi 'y' fed by phi 'x') → returnblock(void)
        // Neither 'x' nor 'y' is read.  Backward dataflow correctly
        // identifies both as dead in a single pass: 'y' is unread, so
        // its dependency `phi_x` (link arg from merge1) doesn't get
        // promoted, and 'x' itself is unread either.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let const_v = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let merge1 = graph.create_block();
        graph.set_goto(entry, merge1, vec![const_v]);
        let phi_x = install_phi(&mut graph, merge1, "x");
        let merge2 = graph.create_block();
        graph.set_goto(merge1, merge2, vec![phi_x]);
        install_phi(&mut graph, merge2, "y");
        graph.set_return(merge2, None);

        prune_dead_phis(&mut graph);

        assert!(
            graph.block(merge1).inputargs.is_empty(),
            "chained dead phi 'x' must collapse after 'y' goes"
        );
        assert!(
            graph.block(merge2).inputargs.is_empty(),
            "leaf dead phi 'y' must be pruned"
        );
        assert!(
            graph.block(entry).exits[0].args.is_empty(),
            "entry → merge1 link arg dropped after 'x' pruned"
        );
        assert!(
            graph.block(merge1).exits[0].args.is_empty(),
            "merge1 → merge2 link arg dropped after 'y' pruned"
        );
    }

    #[test]
    fn prune_dead_phis_preserves_special_blocks() {
        // Function-entry inputargs (startblock), return-value
        // (returnblock), and exception escape (exceptblock) inputargs
        // are part of the calling convention / runtime contract and
        // must NEVER be pruned even if locally unread.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        // Add an Input op at startblock that never gets read — emulates
        // an unused function parameter.
        let unused_param = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "param".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.block_mut(entry).inputargs.push(unused_param);
        graph.set_return(entry, None);

        let pre_returnblock_args = graph.block(graph.returnblock).inputargs.clone();
        let pre_exceptblock_args = graph.block(graph.exceptblock).inputargs.clone();

        prune_dead_phis(&mut graph);

        assert_eq!(
            graph.block(entry).inputargs,
            vec![unused_param],
            "startblock inputargs are calling-convention; never pruned"
        );
        assert_eq!(
            graph.block(graph.returnblock).inputargs,
            pre_returnblock_args,
            "returnblock inputargs are runtime contract; never pruned"
        );
        assert_eq!(
            graph.block(graph.exceptblock).inputargs,
            pre_exceptblock_args,
            "exceptblock inputargs are runtime contract; never pruned"
        );
    }

    #[test]
    fn prune_dead_phis_preserves_pure_raising_op() {
        // `simplify.py:435-436 canremove`'s `op is not block.raising_op`
        // clause: a pure-classified op (e.g. `int_add` modelling
        // `int_add_ovf`) parked as the LAST op of a `canraise` block
        // must NOT be DCE'd even when its result is unread.  The raise
        // side-effect is observable, so removing the op would be a
        // semantic regression.
        //
        // Shape: entry — int_add(unread result) — exitswitch=LastException;
        //   normal exit → returnblock; except exit → exceptblock.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let lhs = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let rhs = graph.push_op(entry, OpKind::ConstInt(2), true).unwrap();
        let raising = graph
            .push_op(
                entry,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        // Mark the entry block as canraise: exitswitch=LastException,
        // exits=[normal → returnblock, except → exceptblock].
        let returnblock = graph.returnblock;
        let exceptblock = graph.exceptblock;
        // Synthesize the (etype, evalue) pair that exceptblock expects
        // — values flow as Constants from the front-end raise site,
        // but for the test we need only the link arity to match.
        let etype = graph.push_op(entry, OpKind::ConstInt(0), true).unwrap();
        let evalue = graph.push_op(entry, OpKind::ConstInt(0), true).unwrap();
        {
            let block = graph.block_mut(entry);
            block.exitswitch = Some(ExitSwitch::LastException);
            block.exits = vec![
                Link::new_mixed(vec![LinkArg::Value(raising)], returnblock, None),
                Link::new_mixed(
                    vec![LinkArg::Value(etype), LinkArg::Value(evalue)],
                    exceptblock,
                    None,
                ),
            ];
        }

        prune_dead_phis(&mut graph);

        let still_present = graph
            .block(entry)
            .operations
            .iter()
            .any(|op| op.result == Some(raising));
        assert!(
            still_present,
            "pure raising_op (last op of canraise block) must survive DCE per \
             simplify.py:436 `op is not block.raising_op`",
        );
    }

    #[test]
    fn prune_dead_phis_skips_non_canonical_entry_blocks() {
        // Closure / generic-specialisation lowering can put function
        // parameters at a non-zero block id with no incoming link
        // (the `generated::*` pipeline does this for `eval_loop_jit`'s
        // body block).  Even if the parameter isn't read graph-side,
        // pruning must NOT remove it — the calling convention assumes
        // every inputarg is supplied.  The block is identified as
        // "entry-like" by having no link targeting it.
        let mut graph = FunctionGraph::new("test");
        // Build: startblock → returnblock(void).  Add an extra
        // orphan-entry block with one Input op + one inputarg, no
        // predecessors at all (not even from startblock).  The block
        // is unreachable but represents the "non-canonical entry"
        // shape pyre emits in some generated pipelines.
        graph.set_return(graph.startblock, None);
        let orphan_entry = graph.create_block();
        let unused_param = graph
            .push_op(
                orphan_entry,
                OpKind::Input {
                    name: "captured".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.block_mut(orphan_entry).inputargs.push(unused_param);

        prune_dead_phis(&mut graph);

        assert_eq!(
            graph.block(orphan_entry).inputargs,
            vec![unused_param],
            "non-canonical entry block (no incoming link) must keep its inputargs"
        );
    }
}
