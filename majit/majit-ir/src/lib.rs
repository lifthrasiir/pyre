pub mod bitstring;
pub mod descr;
pub mod effectinfo;
pub mod op_type_index;
pub mod resoperation;
pub mod resumecode;
pub mod resumedata;
pub mod value;

// Re-export key types at crate root for convenience.
pub use descr::{
    AccumInfo, ArrayDescr, ArrayFlag, CallDescr, DebugMergePointDescr, DebugMergePointInfo, Descr,
    DescrRef, FailDescr, FieldDescr, GcCache, InteriorFieldDescr, JitCodeDescr, LLType,
    LoopTargetDescr, LoopTokenDescr, SimpleCallDescr, SimpleFailDescr, SimpleFieldDescr, SizeDescr,
    SwitchDescr, TargetArgLoc, UnpackAtExitInfo, VableExpansion, descr_identity, make_array_descr,
    make_array_descr_signed, make_call_descr, make_field_descr, make_loop_target_descr,
    make_malloc_array_calldescr, make_malloc_array_nonstandard_calldescr,
    make_malloc_big_fixedsize_calldescr, make_malloc_str_calldescr, make_malloc_unicode_calldescr,
    make_memcpy_calldescr, make_raw_malloc_calldescr, make_size_descr_full,
    make_size_descr_with_vtable, make_tid_field_descr, make_vtable_field_descr, memcpy_fn_addr,
    unpack_fielddescr,
};
pub use effectinfo::{
    CallInfoCollection, EffectInfo, ExtraEffect, OopSpecIndex, QuasiImmutAnalyzer,
    RandomEffectsAnalyzer, UnsupportedFieldExc, VirtualizableAnalyzer, consider_array,
    consider_struct, frozenset_or_none,
};
pub use op_type_index::OpTypeIndex;
pub use resoperation::{
    AbstractValue, ArrayDescrInfo, BoxEnv, FieldDescrInfo, GuardPendingFieldEntry, OPCODE_COUNT,
    Op, OpCode, OpRef, RdVirtualInfo, VectorizationInfo, VirtualFieldsInfo, format_trace,
};
pub use value::{
    Const, FAILARGS_LIMIT, GcRef, GreenAsI64, GreenKey, GreenType, InputArg, JitDriverVar, StrEqFn,
    StrHashFn, Type, Value, VarKind, green_type_to_ir, make_str_slot, set_str_resolver,
    set_unicode_resolver,
};
