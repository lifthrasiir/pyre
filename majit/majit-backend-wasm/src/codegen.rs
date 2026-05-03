/// IR → wasm bytecode compilation.
///
/// Generates a wasm module from majit IR ops using `wasm-encoder`.
/// Generated function signature: `(param $frame_ptr i32) (result i32)`
///
/// Frame layout in shared linear memory:
///   offset 0:       fail_index (i64)
///   offset 8:       slot[0] (i64)
///   offset 16:      slot[1] (i64)
///   ...
///   CALL_AREA_OFS:  func_ptr (i64)   — used by jit_call trampoline
///   CALL_AREA_OFS+8: num_args (i64)
///   CALL_AREA_OFS+16: arg[0] (i64)
///   CALL_AREA_OFS+24: arg[1] (i64)
///   ...
///   CALL_RESULT_OFS: result (i64)    — written by host after call
use std::collections::HashMap;

use majit_backend::BackendError;
use majit_gc::header::{GcHeader, TYPE_ID_MASK};
use majit_ir::{InputArg, Op, OpCode, OpRef, Type};
use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    ImportSection, InstructionSink, MemArg, MemoryType, Module, TypeSection, ValType,
};

/// Frame slot byte offset: slot[i] is at frame_ptr + 8 + i * 8.
const FRAME_SLOT_BASE: u64 = 8;
const SLOT_SIZE: u64 = 8;

/// Call area layout (fixed offsets from frame_ptr).
const CALL_RESULT_OFS: u64 = 2000;
const CALL_FUNC_OFS: u64 = 2008;
const CALL_NARGS_OFS: u64 = 2016;
const CALL_ARGS_OFS: u64 = 2024;

/// Minimum frame allocation size in bytes to accommodate the call area.
pub const MIN_FRAME_BYTES: usize = 2024 + 16 * 8; // 16 max call args

fn mem64(offset: u64) -> MemArg {
    MemArg {
        offset,
        align: 3,
        memory_index: 0,
    }
}

/// llsupport/gc.py:563 GcLLDescr_framework
///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
/// Looks up the materialized table populated by the runner from the
/// active gc_ll_descr. RPython resolves the same value via
/// `cpu.gc_ll_descr.get_typeid_from_classptr_if_gcremovetypeptr`.
fn lookup_typeid_from_classptr(table: &HashMap<i64, u32>, classptr: usize) -> Option<u32> {
    table.get(&(classptr as i64)).copied()
}

/// Information about a guard exit collected during pre-scan.
pub struct GuardExit {
    pub fail_index: u32,
    pub fail_arg_refs: Vec<OpRef>,
    pub fail_arg_types: Vec<Type>,
    pub is_finish: bool,
}

/// Pre-fetched GC-type-guard metadata for the wasm codegen.
///
/// RPython's `genop_guard_guard_*` methods call into
/// `self.cpu.gc_ll_descr` at codegen time to obtain the TYPE_INFO
/// table base, the `infobits` offset / byte mask, the subclassrange
/// field offset, and the `(subclassrange_min, subclassrange_max)`
/// bounds for the constant expected-class pointer. The wasm backend
/// has no direct handle on a `GcAllocator` at this layer, so the
/// caller (`WasmBackend::compile_loop`) pre-fetches each of those
/// values and bundles them here.
///
/// Parity references:
///  * `llsupport/gc.py:162` / `gc.py:318` — `supports_guard_gc_type`
///  * `llsupport/gc.py:592` — `get_translated_info_for_typeinfo`
///  * `llsupport/gc.py:619` — `get_translated_info_for_guard_is_object`
///  * `x86/assembler.py:1951` — `cpu.subclassrange_min_offset`
///  * `x86/assembler.py:1971-1974` — constant-time
///    `(vtable_ptr.subclassrange_min, vtable_ptr.subclassrange_max)`
///
/// The default sets `supports_guard_gc_type = false`, matching
/// `AbstractCPU.supports_guard_gc_type` in `backend/model.py:21`; the
/// codegen arms assert this flag before reading any other field.
#[derive(Default)]
pub struct GuardGcTypeInfo {
    pub supports_guard_gc_type: bool,
    /// `get_translated_info_for_typeinfo()` = (base, shift, sizeof_ti).
    pub base_type_info: usize,
    pub shift_by: u8,
    pub sizeof_ti: usize,
    /// `get_translated_info_for_guard_is_object()`
    ///     = (infobits_offset, T_IS_RPYTHON_INSTANCE_BYTE).
    pub infobits_offset: usize,
    pub is_object_flag: u8,
    /// `cpu.subclassrange_min_offset` (x86/assembler.py:1951).
    pub subclassrange_min_offset: usize,
    /// `(vtable_ptr.subclassrange_min, vtable_ptr.subclassrange_max)`
    /// looked up by constant classptr. Empty when
    /// `supports_guard_gc_type == false`.
    pub subclass_ranges: HashMap<i64, (i64, i64)>,
}

/// Check if any op in the trace is a CALL variant.
fn has_call_ops(ops: &[Op]) -> bool {
    ops.iter().any(|op| op.opcode.is_call())
}

fn collect_guards_and_vars(inputargs: &[InputArg], ops: &[Op]) -> (Vec<GuardExit>, u32) {
    let mut guards = Vec::new();
    let mut max_var: u32 = 0;

    for ia in inputargs {
        if ia.index + 1 > max_var {
            max_var = ia.index + 1;
        }
    }

    let mut fail_index = 0u32;
    for op in ops {
        if op.pos != OpRef::NONE && !op.pos.is_constant() && op.pos.raw() + 1 > max_var {
            max_var = op.pos.raw() + 1;
        }
        if op.opcode == OpCode::Label {
            for &a in &op.args {
                if a != OpRef::NONE && !a.is_constant() && a.raw() + 1 > max_var {
                    max_var = a.raw() + 1;
                }
            }
        }

        if op.opcode.is_guard() || op.opcode == OpCode::Finish {
            let fail_args = op
                .fail_args
                .as_ref()
                .map(|fa| fa.as_slice())
                .unwrap_or(&op.args);
            let fail_arg_types = op
                .fail_arg_types
                .as_ref()
                .cloned()
                .unwrap_or_else(|| fail_args.iter().map(|_| Type::Int).collect());

            guards.push(GuardExit {
                fail_index,
                fail_arg_refs: fail_args.to_vec(),
                fail_arg_types,
                is_finish: op.opcode == OpCode::Finish,
            });
            fail_index += 1;
        }
    }

    (guards, max_var)
}

/// Build a wasm module from majit IR.
pub fn build_wasm_module(
    inputargs: &[InputArg],
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    vtable_offset: Option<usize>,
    classptr_to_typeid: &HashMap<i64, u32>,
    guard_gc_type_info: &GuardGcTypeInfo,
) -> Result<(Vec<u8>, Vec<GuardExit>), BackendError> {
    let (guards, num_vars) = collect_guards_and_vars(inputargs, ops);
    let needs_call = has_call_ops(ops);

    let mut module = Module::new();

    // Type section
    let mut types = TypeSection::new();
    // Type 0: trace function (param i32) -> (result i32)
    types.ty().function(vec![ValType::I32], vec![ValType::I32]);
    if needs_call {
        // Type 1: jit_call trampoline (param i32) -> ()
        types.ty().function(vec![ValType::I32], vec![]);
    }
    module.section(&types);

    // Import section
    let mut imports = ImportSection::new();
    imports.import(
        "env",
        "memory",
        MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        },
    );
    if needs_call {
        // Import jit_call trampoline as function index 0
        imports.import("env", "jit_call", EntityType::Function(1));
    }
    module.section(&imports);

    // Function section
    let mut functions = FunctionSection::new();
    functions.function(0); // type 0
    module.section(&functions);

    // Export section: trace function index depends on whether we imported jit_call
    let trace_func_idx = if needs_call { 1 } else { 0 };
    let mut exports = ExportSection::new();
    exports.export("trace", ExportKind::Func, trace_func_idx);
    module.section(&exports);

    // Code section
    let mut codes = CodeSection::new();
    let jit_call_idx = if needs_call { Some(0u32) } else { None };
    let func = build_function(
        inputargs,
        ops,
        constants,
        num_vars,
        jit_call_idx,
        vtable_offset,
        classptr_to_typeid,
        guard_gc_type_info,
    )?;
    codes.function(&func);
    module.section(&codes);

    Ok((module.finish(), guards))
}

fn build_function(
    inputargs: &[InputArg],
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    num_vars: u32,
    jit_call_idx: Option<u32>,
    vtable_offset: Option<usize>,
    classptr_to_typeid: &HashMap<i64, u32>,
    guard_gc_type_info: &GuardGcTypeInfo,
) -> Result<Function, BackendError> {
    let mut func = Function::new(vec![(num_vars, ValType::I64)]);
    let mut sink = func.instructions();

    // Load inputs from frame into locals
    for ia in inputargs {
        let local_idx = 1 + ia.index;
        let offset = FRAME_SLOT_BASE + ia.index as u64 * SLOT_SIZE;
        sink.local_get(0)
            .i64_load(mem64(offset))
            .local_set(local_idx);
    }

    let has_loop = ops.iter().any(|op| op.opcode == OpCode::Label);
    if has_loop {
        sink.block(BlockType::Empty);
        sink.loop_(BlockType::Empty);
    }

    let mut guard_idx = 0u32;

    for op in ops {
        match op.opcode {
            OpCode::Label => {}

            OpCode::Jump => {
                let label_args = find_label_args(ops);
                for (i, &jump_arg) in op.args.iter().enumerate() {
                    if i < label_args.len() {
                        let target_local = 1 + label_args[i].raw();
                        emit_resolve(&mut sink, constants, jump_arg);
                        sink.local_set(target_local);
                    }
                }
                sink.br(0);
            }

            OpCode::Finish => {
                emit_guard_exit(&mut sink, constants, guard_idx, op);
                if has_loop {
                    sink.br(1);
                }
                guard_idx += 1;
            }

            // ── Guards ──
            OpCode::GuardTrue => {
                emit_guard_true(&mut sink, constants, guard_idx, op, has_loop);
                guard_idx += 1;
            }
            OpCode::GuardFalse => {
                emit_guard_false(&mut sink, constants, guard_idx, op, has_loop);
                guard_idx += 1;
            }
            OpCode::GuardValue => {
                emit_resolve(&mut sink, constants, op.arg(0));
                emit_resolve(&mut sink, constants, op.arg(1));
                sink.i64_ne();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, has_loop);
                guard_idx += 1;
            }
            OpCode::GuardNonnull => {
                emit_resolve(&mut sink, constants, op.arg(0));
                sink.i64_eqz();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, has_loop);
                guard_idx += 1;
            }
            OpCode::GuardIsnull => {
                emit_resolve(&mut sink, constants, op.arg(0));
                sink.i64_const(0);
                sink.i64_ne();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, has_loop);
                guard_idx += 1;
            }
            OpCode::GuardClass | OpCode::GuardNonnullClass => {
                // x86/assembler.py:1880-1891 _cmp_guard_class:
                //   offset = self.cpu.vtable_offset
                //   if offset is not None: CMP(mem(loc_ptr, offset), classptr)
                //   else:
                //       assert isinstance(loc_classptr, ImmedLoc)
                //       expected_typeid = gc_ll_descr.
                //           get_typeid_from_classptr_if_gcremovetypeptr(...)
                //       _cmp_guard_gc_type(loc_ptr, ImmedLoc(expected_typeid))
                if let Some(off_usize) = vtable_offset {
                    let off = off_usize as u64;
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.i64_load(mem64(off));
                    emit_resolve(&mut sink, constants, op.arg(1));
                    sink.i64_ne();
                } else {
                    // gcremovetypeptr fallback (assembler.py:1893-1901):
                    //   on x86_64 the typeid is a 32-bit value at offset 0.
                    let classptr = constants.get(&op.arg(1).raw()).copied().expect(
                        "_cmp_guard_class: gcremovetypeptr requires \
                         loc_classptr to be an immediate (assert \
                         isinstance(loc_classptr, ImmedLoc) in \
                         x86/assembler.py:1887)",
                    );
                    let expected_typeid =
                        lookup_typeid_from_classptr(classptr_to_typeid, classptr as usize).expect(
                            "GuardClass: vtable_offset is None but the wasm \
                                 backend has no gc_ll_descr.\
                                 get_typeid_from_classptr_if_gcremovetypeptr",
                        );
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.i32_wrap_i64();
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i32_const(expected_typeid as i32);
                    sink.i32_ne();
                }
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, has_loop);
                guard_idx += 1;
            }
            OpCode::GuardNoOverflow => {
                // RPython: 0 args — overflow flag implicit from preceding ovf op.
                // Wasm MVP doesn't detect overflow, so always passes.
                guard_idx += 1;
            }
            OpCode::GuardOverflow => {
                // Always fails (no overflow detected in wasm MVP).
                emit_guard_exit(&mut sink, constants, guard_idx, op);
                if has_loop {
                    sink.br(1);
                }
                guard_idx += 1;
            }
            // Guards that always pass in wasm MVP
            OpCode::GuardNotInvalidated
            | OpCode::GuardNotForced
            | OpCode::GuardNotForced2
            | OpCode::GuardNoException
            | OpCode::GuardException => {
                // No-op: these guards are about runtime state that
                // the wasm backend doesn't track yet.
                guard_idx += 1;
            }

            // ── Integer arithmetic ──
            OpCode::IntAdd => emit_binop(&mut sink, constants, op, BinOp::I64Add),
            OpCode::IntSub => emit_binop(&mut sink, constants, op, BinOp::I64Sub),
            OpCode::IntMul => emit_binop(&mut sink, constants, op, BinOp::I64Mul),
            OpCode::IntFloorDiv => emit_binop(&mut sink, constants, op, BinOp::I64DivS),
            OpCode::IntMod => emit_binop(&mut sink, constants, op, BinOp::I64RemS),
            OpCode::IntAnd => emit_binop(&mut sink, constants, op, BinOp::I64And),
            OpCode::IntOr => emit_binop(&mut sink, constants, op, BinOp::I64Or),
            OpCode::IntXor => emit_binop(&mut sink, constants, op, BinOp::I64Xor),
            OpCode::IntLshift => emit_binop(&mut sink, constants, op, BinOp::I64Shl),
            OpCode::IntRshift => emit_binop(&mut sink, constants, op, BinOp::I64ShrS),
            OpCode::UintRshift => emit_binop(&mut sink, constants, op, BinOp::I64ShrU),

            // Overflow variants: compute result + overflow flag
            OpCode::IntAddOvf => emit_ovf_binop(&mut sink, constants, op, BinOp::I64Add),
            OpCode::IntSubOvf => emit_ovf_binop(&mut sink, constants, op, BinOp::I64Sub),
            OpCode::IntMulOvf => emit_ovf_binop(&mut sink, constants, op, BinOp::I64Mul),

            // ── Integer comparisons (signed) ──
            OpCode::IntLt => emit_cmp(&mut sink, constants, op, CmpOp::I64LtS),
            OpCode::IntLe => emit_cmp(&mut sink, constants, op, CmpOp::I64LeS),
            OpCode::IntEq => emit_cmp(&mut sink, constants, op, CmpOp::I64Eq),
            OpCode::IntNe => emit_cmp(&mut sink, constants, op, CmpOp::I64Ne),
            OpCode::IntGt => emit_cmp(&mut sink, constants, op, CmpOp::I64GtS),
            OpCode::IntGe => emit_cmp(&mut sink, constants, op, CmpOp::I64GeS),

            // ── Integer comparisons (unsigned) ──
            OpCode::UintLt => emit_cmp(&mut sink, constants, op, CmpOp::I64LtU),
            OpCode::UintLe => emit_cmp(&mut sink, constants, op, CmpOp::I64LeU),
            OpCode::UintGt => emit_cmp(&mut sink, constants, op, CmpOp::I64GtU),
            OpCode::UintGe => emit_cmp(&mut sink, constants, op, CmpOp::I64GeU),

            // ── Pointer comparisons ──
            OpCode::PtrEq | OpCode::InstancePtrEq => {
                emit_cmp(&mut sink, constants, op, CmpOp::I64Eq);
            }
            OpCode::PtrNe | OpCode::InstancePtrNe => {
                emit_cmp(&mut sink, constants, op, CmpOp::I64Ne);
            }

            // ── Unary ops ──
            OpCode::IntNeg => emit_unary_vi(
                &mut sink,
                constants,
                op,
                |s| {
                    s.i64_const(0);
                },
                |s| {
                    s.i64_sub();
                },
            ),
            OpCode::IntInvert => emit_unary_vi(
                &mut sink,
                constants,
                op,
                |s| {
                    s.i64_const(-1);
                },
                |s| {
                    s.i64_xor();
                },
            ),
            OpCode::IntIsTrue => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.i64_const(0);
                    sink.i64_ne();
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::IntIsZero => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.i64_eqz();
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }

            // ── Extended integer ops ──
            OpCode::IntSignext => {
                // int_signext(val, num_bytes): sign-extend from num_bytes width
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    // num_bytes is arg(1), typically a constant
                    let num_bytes = if op.arg(1).is_constant() {
                        constants.get(&op.arg(1).raw()).copied().unwrap_or(8)
                    } else {
                        8 // default to no-op
                    };
                    let shift = 64 - num_bytes * 8;
                    if shift > 0 && shift < 64 {
                        sink.i64_const(shift);
                        sink.i64_shl();
                        sink.i64_const(shift);
                        sink.i64_shr_s();
                    }
                    sink.local_set(1 + vi);
                }
            }
            OpCode::IntForceGeZero => {
                // max(val, 0)
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    // if val < 0, use 0; else use val
                    // Wasm: local.tee + i64.const 0 + local.get + i64.lt_s + select
                    let tmp_local = 1 + vi; // reuse result local as temp
                    sink.local_tee(tmp_local);
                    sink.i64_const(0);
                    sink.local_get(tmp_local);
                    sink.i64_const(0);
                    sink.i64_lt_s();
                    sink.select();
                    sink.local_set(1 + vi);
                }
            }

            // ── Float comparisons ──
            OpCode::FloatLt => emit_float_cmp(&mut sink, constants, op, FloatCmp::Lt),
            OpCode::FloatLe => emit_float_cmp(&mut sink, constants, op, FloatCmp::Le),
            OpCode::FloatEq => emit_float_cmp(&mut sink, constants, op, FloatCmp::Eq),
            OpCode::FloatNe => emit_float_cmp(&mut sink, constants, op, FloatCmp::Ne),
            OpCode::FloatGt => emit_float_cmp(&mut sink, constants, op, FloatCmp::Gt),
            OpCode::FloatGe => emit_float_cmp(&mut sink, constants, op, FloatCmp::Ge),

            // ── Float floor/mod ──
            OpCode::FloatFloorDiv => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.f64_reinterpret_i64();
                    emit_resolve(&mut sink, constants, op.arg(1));
                    sink.f64_reinterpret_i64();
                    sink.f64_div();
                    sink.f64_floor();
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }

            // ── Float/Int conversions ──
            OpCode::CastFloatToInt => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.f64_reinterpret_i64();
                    sink.i64_trunc_sat_f64_s();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::CastIntToFloat => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.f64_convert_i64_s();
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::ConvertFloatBytesToLonglong | OpCode::ConvertLonglongBytesToFloat => {
                // These are bitcast (no-op on the i64 representation)
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.local_set(1 + vi);
                }
            }

            // ── Pointer/Int conversions ──
            OpCode::CastPtrToInt | OpCode::CastIntToPtr | OpCode::CastOpaquePtr => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.local_set(1 + vi);
                }
            }

            // ── SameAs (forwarding) ──
            OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.local_set(1 + vi);
                }
            }

            // ── Field access (direct memory operations) ──
            OpCode::GetfieldGcI | OpCode::GetfieldGcPureI | OpCode::GetfieldRawI => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0)); // struct ptr (i64)
                    sink.i32_wrap_i64(); // convert to i32 address
                    let field_offset = field_offset_from_descr(op);
                    sink.i64_load(mem64(field_offset));
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GetfieldGcR | OpCode::GetfieldGcPureR | OpCode::GetfieldRawR => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.i32_wrap_i64();
                    let field_offset = field_offset_from_descr(op);
                    // Load as i32 (pointer on wasm32) and extend to i64
                    sink.i32_load(MemArg {
                        offset: field_offset,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::SetfieldGc | OpCode::SetfieldRaw => {
                emit_resolve(&mut sink, constants, op.arg(0)); // struct ptr
                sink.i32_wrap_i64();
                let field_offset = field_offset_from_descr(op);
                emit_resolve(&mut sink, constants, op.arg(1)); // value
                sink.i64_store(mem64(field_offset));
            }

            // ── Float field access ──
            OpCode::GetfieldGcF | OpCode::GetfieldGcPureF | OpCode::GetfieldRawF => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.i32_wrap_i64();
                    let field_offset = field_offset_from_descr(op);
                    sink.f64_load(MemArg {
                        offset: field_offset,
                        align: 3,
                        memory_index: 0,
                    });
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }

            // ── Array access ──
            OpCode::ArraylenGc => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0)); // array ptr
                    sink.i32_wrap_i64();
                    let len_offset = array_len_offset_from_descr(op);
                    sink.i64_load(mem64(len_offset));
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcPureI | OpCode::GetarrayitemRawI => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    // addr = base + base_size + index * item_size
                    emit_array_addr(&mut sink, constants, op);
                    sink.i64_load(mem64(0));
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcPureR | OpCode::GetarrayitemRawR => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_array_addr(&mut sink, constants, op);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
                emit_array_addr(&mut sink, constants, op);
                emit_resolve(&mut sink, constants, op.arg(2)); // value
                sink.i64_store(mem64(0));
            }

            // ── Interior field access ──
            OpCode::GetinteriorfieldGcI | OpCode::GetinteriorfieldGcR => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    // getinteriorfield(array, index, offset)
                    emit_resolve(&mut sink, constants, op.arg(0)); // array ptr
                    sink.i32_wrap_i64();
                    let field_offset = field_offset_from_descr(op);
                    // Simplified: use field_offset directly (RPython computes base+index*itemsize+offset)
                    sink.i64_load(mem64(field_offset));
                    sink.local_set(1 + vi);
                }
            }
            OpCode::SetinteriorfieldGc => {
                emit_resolve(&mut sink, constants, op.arg(0));
                sink.i32_wrap_i64();
                let field_offset = field_offset_from_descr(op);
                emit_resolve(&mut sink, constants, op.arg(2)); // value
                sink.i64_store(mem64(field_offset));
            }

            // ── String/Unicode ops (direct memory access) ──
            OpCode::Strlen | OpCode::Unicodelen => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.i32_wrap_i64();
                    // Length at offset 8 (after ob_type pointer on wasm32)
                    sink.i64_load(mem64(8));
                    sink.local_set(1 + vi);
                }
            }
            OpCode::Strgetitem | OpCode::Unicodegetitem => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    // str[index]: base + header_size + index
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.i32_wrap_i64();
                    emit_resolve(&mut sink, constants, op.arg(1)); // index
                    sink.i32_wrap_i64();
                    sink.i32_add();
                    // String data starts after header (assume 16 bytes: ob_type + length)
                    sink.i32_load8_u(MemArg {
                        offset: 16,
                        align: 0,
                        memory_index: 0,
                    });
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }

            // ── GC memory ops ──
            OpCode::GcLoadI => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.i32_wrap_i64();
                    let offset = field_offset_from_descr(op);
                    sink.i64_load(mem64(offset));
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GcLoadR => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.i32_wrap_i64();
                    let offset = field_offset_from_descr(op);
                    sink.i32_load(MemArg {
                        offset,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GcStore => {
                emit_resolve(&mut sink, constants, op.arg(0));
                sink.i32_wrap_i64();
                let offset = field_offset_from_descr(op);
                emit_resolve(&mut sink, constants, op.arg(1));
                sink.i64_store(mem64(offset));
            }

            // ── Raw memory access ──
            OpCode::RawLoadI => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0)); // ptr
                    sink.i32_wrap_i64();
                    emit_resolve(&mut sink, constants, op.arg(1)); // offset
                    sink.i32_wrap_i64();
                    sink.i32_add();
                    sink.i64_load(mem64(0));
                    sink.local_set(1 + vi);
                }
            }
            OpCode::RawStore => {
                emit_resolve(&mut sink, constants, op.arg(0));
                sink.i32_wrap_i64();
                emit_resolve(&mut sink, constants, op.arg(1));
                sink.i32_wrap_i64();
                sink.i32_add();
                emit_resolve(&mut sink, constants, op.arg(2));
                sink.i64_store(mem64(0));
            }

            // ── Exception handling ──
            OpCode::SaveException | OpCode::SaveExcClass | OpCode::RestoreException => {
                // No-op in wasm MVP — exception state is managed by the host.
            }

            // ── Conditional calls ──
            OpCode::CondCallN | OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                // GC write barriers and conditional void calls — no-op in wasm.
            }

            // x86/assembler.py:1919-1922 genop_guard_guard_gc_type:
            // GUARD_GC_TYPE: args[0] = object ref, args[1] = expected
            // type_id. The majit runtime stores the typeid in the GC
            // header word placed immediately before the object payload
            // (`majit_gc::header::GcHeader::tid_and_flags`, lower 32
            // bits). The cranelift backend lowers the same op this way
            // (compiler.rs GuardGcType branch). This is NOT the RPython
            // gcremovetypeptr layout — pyre's GC keeps the typeid in the
            // header, not at `obj[0]`.
            OpCode::GuardGcType => {
                let _ = classptr_to_typeid; // typeid is already an immediate
                if op.args.len() >= 2 {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    // header address = obj - GcHeader::SIZE
                    sink.i64_const(GcHeader::SIZE as i64);
                    sink.i64_sub();
                    sink.i32_wrap_i64();
                    // Load 8-byte header word (tid_and_flags)
                    sink.i64_load(mem64(0));
                    // Mask lower TYPE_ID_BITS to extract the type id
                    sink.i64_const(TYPE_ID_MASK as i64);
                    sink.i64_and();
                    // Compare against expected_typeid (arg1 — already an
                    // i64 in the constant pool or a frame slot).
                    emit_resolve(&mut sink, constants, op.arg(1));
                    sink.i64_ne();
                    emit_guard_if_exit(&mut sink, constants, guard_idx, op, has_loop);
                }
                guard_idx += 1;
            }
            // x86/assembler.py:1924-1943 genop_guard_guard_is_object.
            //     assert self.cpu.supports_guard_gc_type
            //     [loc_object, loc_typeid] = locs
            //     if IS_X86_32:
            //         self.mc.MOVZX16(loc_typeid, mem(loc_object, 0))
            //     else:
            //         self.mc.MOV32(loc_typeid, mem(loc_object, 0))
            //     base_type_info, shift_by, sizeof_ti = (
            //         self.cpu.gc_ll_descr
            //             .get_translated_info_for_typeinfo())
            //     infobits_offset, IS_OBJECT_FLAG = (
            //         self.cpu.gc_ll_descr
            //             .get_translated_info_for_guard_is_object())
            //     loc_infobits = addr_add(imm(base_type_info),
            //                             loc_typeid,
            //                             scale=shift_by,
            //                             offset=infobits_offset)
            //     self.mc.TEST8(loc_infobits, imm(IS_OBJECT_FLAG))
            //     self.guard_success_cc = rx86.Conditions['NZ']
            //     self.implement_guard(guard_token)
            OpCode::GuardIsObject => {
                // assembler.py:1925 assert self.cpu.supports_guard_gc_type
                assert!(
                    guard_gc_type_info.supports_guard_gc_type,
                    "x86/assembler.py:1925: assert self.cpu.\
                     supports_guard_gc_type (GcAllocator has not \
                     installed a TYPE_INFO layout)"
                );
                // assembler.py:1931-1932 MOV32 loc_typeid, mem(loc_object, 0).
                // majit's GC header sits at obj - GcHeader::SIZE; the
                // typeid occupies the lower TYPE_ID_BITS of that word.
                emit_resolve(&mut sink, constants, op.arg(0));
                sink.i64_const(GcHeader::SIZE as i64);
                sink.i64_sub();
                sink.i32_wrap_i64();
                sink.i64_load(mem64(0));
                sink.i64_const(TYPE_ID_MASK as i64);
                sink.i64_and();
                // Stack: [..., loc_typeid]

                // assembler.py:1938-1939 addr_add(imm(base_type_info),
                //     loc_typeid, scale=shift_by, offset=infobits_offset)
                if guard_gc_type_info.shift_by > 0 {
                    sink.i64_const(guard_gc_type_info.shift_by as i64);
                    sink.i64_shl();
                }
                sink.i64_const(guard_gc_type_info.base_type_info as i64);
                sink.i64_add();
                sink.i64_const(guard_gc_type_info.infobits_offset as i64);
                sink.i64_add();
                sink.i32_wrap_i64();
                // Stack: [..., loc_infobits(i32 addr)]

                // assembler.py:1940 TEST8 [loc_infobits], IS_OBJECT_FLAG
                sink.i32_load8_u(MemArg {
                    offset: 0,
                    align: 0,
                    memory_index: 0,
                });
                sink.i32_const(guard_gc_type_info.is_object_flag as i32);
                sink.i32_and();
                // assembler.py:1942 guard_success_cc = Conditions['NZ']:
                // guard passes when byte & flag != 0; fail when == 0.
                sink.i32_eqz();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, has_loop);
                guard_idx += 1;
            }
            // x86/assembler.py:1945-1980 genop_guard_guard_subclass.
            //     assert self.cpu.supports_guard_gc_type
            //     [loc_object, loc_check_against_class, loc_tmp] = locs
            //     offset = self.cpu.vtable_offset
            //     offset2 = self.cpu.subclassrange_min_offset
            //     if offset is not None:
            //         self.mc.MOV_rm(loc_tmp, (loc_object, offset))
            //         self.mc.MOV_rm(loc_tmp, (loc_tmp, offset2))
            //     else:
            //         self.mc.MOV32(loc_tmp, mem(loc_object, 0))
            //         base_type_info, shift_by, sizeof_ti = (
            //             gc_ll_descr.get_translated_info_for_typeinfo())
            //         self.mc.MOV(loc_tmp, addr_add(
            //             imm(base_type_info), loc_tmp,
            //             scale=shift_by,
            //             offset=sizeof_ti + offset2))
            //     vtable_ptr = loc_check_against_class.getint()
            //     vtable_ptr = rffi.cast(rclass.CLASSTYPE, vtable_ptr)
            //     check_min = vtable_ptr.subclassrange_min
            //     check_max = vtable_ptr.subclassrange_max
            //     self.mc.SUB_ri(loc_tmp, check_min)
            //     self.mc.CMP_ri(loc_tmp, check_max - check_min)
            //     self.guard_success_cc = Conditions['B']
            //     self.implement_guard(guard_token)
            OpCode::GuardSubclass => {
                // assembler.py:1946 assert self.cpu.supports_guard_gc_type
                assert!(
                    guard_gc_type_info.supports_guard_gc_type,
                    "x86/assembler.py:1946: assert self.cpu.\
                     supports_guard_gc_type (GcAllocator has not \
                     installed a TYPE_INFO / rclass.CLASSTYPE layout)"
                );

                // assembler.py:1971 vtable_ptr = loc_check_against_class
                //   .getint(): the bounds are resolved at codegen time,
                //   so arg1 must be an immediate class pointer.
                let loc_check_against_class =
                    constants.get(&op.arg(1).raw()).copied().unwrap_or_else(|| {
                        panic!(
                            "x86/assembler.py:1971 vtable_ptr = \
                             loc_check_against_class.getint(): \
                             GUARD_SUBCLASS requires arg1 to be an \
                             immediate class pointer"
                        )
                    });
                // assembler.py:1973-1974: vtable_ptr.subclassrange_{min,max}
                let (check_min, check_max) = guard_gc_type_info
                    .subclass_ranges
                    .get(&loc_check_against_class)
                    .copied()
                    .unwrap_or_else(|| {
                        panic!(
                            "x86/assembler.py:1973-1974 vtable_ptr.\
                             subclassrange_min/max: GcAllocator has no \
                             rclass.CLASSTYPE entry for classptr {:#x}",
                            loc_check_against_class
                        )
                    });

                // assembler.py:1950-1951 offset / offset2.
                let offset2 = guard_gc_type_info.subclassrange_min_offset;
                if let Some(vtable_off) = vtable_offset {
                    // assembler.py:1953-1956
                    //     MOV_rm(loc_tmp, (loc_object, offset))
                    //     MOV_rm(loc_tmp, (loc_tmp, offset2))
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.i32_wrap_i64();
                    sink.i64_load(mem64(vtable_off as u64));
                    sink.i32_wrap_i64();
                    sink.i64_load(mem64(offset2 as u64));
                } else {
                    // assembler.py:1957-1969 gcremovetypeptr path.
                    //     MOV32 loc_tmp, mem(loc_object, 0)
                    //     base_type_info, shift_by, sizeof_ti = ...
                    //     MOV loc_tmp, [base_type_info
                    //         + (loc_tmp << shift_by)
                    //         + sizeof_ti + offset2]
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.i64_const(GcHeader::SIZE as i64);
                    sink.i64_sub();
                    sink.i32_wrap_i64();
                    sink.i64_load(mem64(0));
                    sink.i64_const(TYPE_ID_MASK as i64);
                    sink.i64_and();
                    if guard_gc_type_info.shift_by > 0 {
                        sink.i64_const(guard_gc_type_info.shift_by as i64);
                        sink.i64_shl();
                    }
                    sink.i64_const(guard_gc_type_info.base_type_info as i64);
                    sink.i64_add();
                    sink.i64_const((guard_gc_type_info.sizeof_ti + offset2) as i64);
                    sink.i64_add();
                    sink.i32_wrap_i64();
                    sink.i64_load(mem64(0));
                }
                // Stack: [..., loc_tmp (i64)]

                // assembler.py:1976-1978 unsigned comparison:
                //     (loc_tmp - check_min) <u (check_max - check_min)
                sink.i64_const(check_min);
                sink.i64_sub();
                sink.i64_const(check_max - check_min);
                // assembler.py:1979 guard_success_cc = Conditions['B']:
                // guard passes when sub <u limit; fail when sub >=u limit.
                sink.i64_ge_u();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, has_loop);
                guard_idx += 1;
            }
            OpCode::GuardFutureCondition | OpCode::GuardAlwaysFails => {
                // GuardAlwaysFails always exits.
                emit_guard_exit(&mut sink, constants, guard_idx, op);
                if has_loop {
                    sink.br(1);
                }
                guard_idx += 1;
            }

            // ── Quasi-immutable / record / assert ──
            OpCode::QuasiimmutField
            | OpCode::RecordExactClass
            | OpCode::RecordExactValueI
            | OpCode::RecordExactValueR
            | OpCode::AssertNotNone => {
                // Metadata-only ops, no codegen needed.
            }

            // ── Allocation via trampoline ──
            OpCode::Newstr | OpCode::Newunicode => {
                // These may appear in traces that materialize strings.
                // Use CALL trampoline if available, otherwise skip.
                if let Some(jit_call) = jit_call_idx {
                    let vi = op.pos.raw();
                    sink.local_get(0);
                    emit_resolve(&mut sink, constants, op.arg(0)); // length
                    sink.i64_store(mem64(CALL_ARGS_OFS));
                    sink.local_get(0);
                    sink.i64_const(0); // func_ptr = 0 signals "newstr" to host
                    sink.i64_store(mem64(CALL_FUNC_OFS));
                    sink.local_get(0);
                    sink.i64_const(1);
                    sink.i64_store(mem64(CALL_NARGS_OFS));
                    sink.local_get(0);
                    sink.call(jit_call);
                    if !OpRef::from_raw(vi).is_constant() {
                        sink.local_get(0);
                        sink.i64_load(mem64(CALL_RESULT_OFS));
                        sink.local_set(1 + vi);
                    }
                }
            }

            // ── String content copy ──
            OpCode::Copystrcontent | OpCode::Copyunicodecontent => {
                // Bulk memory copy — use CALL trampoline or skip
            }

            // ── Misc ops ──
            OpCode::NurseryPtrIncrement => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    emit_resolve(&mut sink, constants, op.arg(1));
                    sink.i64_add();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::CheckMemoryError => {
                // After allocation: check if result is null
                // No-op in wasm (allocations don't fail the same way)
            }
            OpCode::ZeroArray => {
                // Zero-initialize array region — skip for MVP
            }
            OpCode::LoadFromGcTable => {
                // Load from GC reference table — treat as field load
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.local_set(1 + vi);
                }
            }

            // ── CALL operations (via trampoline) ──
            OpCode::CallI
            | OpCode::CallR
            | OpCode::CallN
            | OpCode::CallF
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureN
            | OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceN
            | OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerN
            | OpCode::CallReleaseGilI
            | OpCode::CallReleaseGilN
            | OpCode::CondCallValueI
            | OpCode::CondCallValueR
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantN
            | OpCode::CallLoopinvariantF
            | OpCode::CallPureF
            | OpCode::CallMayForceF
            | OpCode::CallAssemblerF
            | OpCode::CallReleaseGilF => {
                let vi = op.pos.raw();
                let jit_call = jit_call_idx.expect("CALL op present but jit_call not imported");

                // args[0] = func_ptr, args[1..] = call arguments
                let func_ptr_ref = op.arg(0);
                let call_args = &op.args[1..];

                // Store func_ptr to call area
                sink.local_get(0);
                emit_resolve(&mut sink, constants, func_ptr_ref);
                sink.i64_store(mem64(CALL_FUNC_OFS));

                // Store num_args
                sink.local_get(0);
                sink.i64_const(call_args.len() as i64);
                sink.i64_store(mem64(CALL_NARGS_OFS));

                // Store each arg
                for (i, &arg) in call_args.iter().enumerate() {
                    sink.local_get(0);
                    emit_resolve(&mut sink, constants, arg);
                    sink.i64_store(mem64(CALL_ARGS_OFS + i as u64 * SLOT_SIZE));
                }

                // Call trampoline
                sink.local_get(0);
                sink.call(jit_call);

                // Read result (for non-void calls)
                let is_void = matches!(
                    op.opcode,
                    OpCode::CallN
                        | OpCode::CallPureN
                        | OpCode::CallMayForceN
                        | OpCode::CallAssemblerN
                        | OpCode::CallReleaseGilN
                        | OpCode::CallLoopinvariantN
                );
                if !OpRef::from_raw(vi).is_constant() && !is_void {
                    sink.local_get(0);
                    sink.i64_load(mem64(CALL_RESULT_OFS));
                    sink.local_set(1 + vi);
                }
            }

            // ── Allocation (via trampoline — treated as CALL) ──
            OpCode::New | OpCode::NewWithVtable | OpCode::NewArray | OpCode::NewArrayClear => {
                // These are handled by the optimizer and shouldn't normally
                // appear in optimized traces (they become virtuals).
                // If they do appear, skip — the host will handle on guard failure.
            }

            // ── Misc ──
            OpCode::ForceToken => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    sink.i64_const(0); // sentinel force token
                    sink.local_set(1 + vi);
                }
            }

            // Float operations
            OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    // Values stored as i64 (bitcast from f64)
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.f64_reinterpret_i64();
                    emit_resolve(&mut sink, constants, op.arg(1));
                    sink.f64_reinterpret_i64();
                    match op.opcode {
                        OpCode::FloatAdd => {
                            sink.f64_add();
                        }
                        OpCode::FloatSub => {
                            sink.f64_sub();
                        }
                        OpCode::FloatMul => {
                            sink.f64_mul();
                        }
                        OpCode::FloatTrueDiv => {
                            sink.f64_div();
                        }
                        _ => unreachable!(),
                    }
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::FloatNeg => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.f64_reinterpret_i64();
                    sink.f64_neg();
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::FloatAbs => {
                let vi = op.pos.raw();
                if !OpRef::from_raw(vi).is_constant() {
                    emit_resolve(&mut sink, constants, op.arg(0));
                    sink.f64_reinterpret_i64();
                    sink.f64_abs();
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }

            // Debug / metadata / no-op
            OpCode::DebugMergePoint
            | OpCode::IncrementDebugCounter
            | OpCode::EnterPortalFrame
            | OpCode::LeavePortalFrame
            | OpCode::VirtualRefFinish
            | OpCode::ForceSpill
            | OpCode::Keepalive => {}

            _ => {
                // Unsupported opcode — skip silently.
            }
        }
    }

    if has_loop {
        sink.end(); // end loop
        sink.end(); // end block
    }

    sink.local_get(0);
    sink.end(); // end function

    Ok(func)
}

// ── Helpers ──

fn find_label_args(ops: &[Op]) -> Vec<OpRef> {
    for op in ops {
        if op.opcode == OpCode::Label {
            return op.args.clone().to_vec();
        }
    }
    Vec::new()
}

fn emit_resolve(sink: &mut InstructionSink<'_>, constants: &HashMap<u32, i64>, opref: OpRef) {
    if opref.is_constant() {
        let val = constants.get(&opref.raw()).copied().unwrap_or(0);
        sink.i64_const(val);
    } else {
        sink.local_get(1 + opref.raw());
    }
}

/// Extract field offset from op's descr (FieldDescr).
fn field_offset_from_descr(op: &Op) -> u64 {
    if let Some(ref descr) = op.descr {
        if let Some(fd) = descr.as_field_descr() {
            return fd.offset() as u64;
        }
    }
    0
}

/// Extract array length offset from descr.
fn array_len_offset_from_descr(_op: &Op) -> u64 {
    // RPython arrays store length before the data.
    // On wasm32, the length is typically at a fixed offset.
    8 // default: length at offset 8 (after ob_type)
}

/// Compute array element address: base + base_size + index * item_size.
/// Leaves i32 address on the wasm stack.
fn emit_array_addr(sink: &mut InstructionSink<'_>, constants: &HashMap<u32, i64>, op: &Op) {
    let (base_size, item_size) = if let Some(ref descr) = op.descr {
        if let Some(ad) = descr.as_array_descr() {
            (ad.base_size() as u64, ad.item_size() as u64)
        } else {
            (16, 8) // default
        }
    } else {
        (16, 8)
    };
    emit_resolve(sink, constants, op.arg(0)); // array ptr
    sink.i32_wrap_i64();
    // base + base_size + index * item_size
    emit_resolve(sink, constants, op.arg(1)); // index
    sink.i32_wrap_i64();
    sink.i32_const(item_size as i32);
    sink.i32_mul();
    sink.i32_add();
    sink.i32_const(base_size as i32);
    sink.i32_add();
}

// ── Guard emission helpers ──

fn emit_guard_true(
    sink: &mut InstructionSink<'_>,
    constants: &HashMap<u32, i64>,
    guard_idx: u32,
    op: &Op,
    has_loop: bool,
) {
    emit_resolve(sink, constants, op.arg(0));
    sink.i64_eqz();
    emit_guard_if_exit(sink, constants, guard_idx, op, has_loop);
}

fn emit_guard_false(
    sink: &mut InstructionSink<'_>,
    constants: &HashMap<u32, i64>,
    guard_idx: u32,
    op: &Op,
    has_loop: bool,
) {
    emit_resolve(sink, constants, op.arg(0));
    sink.i64_const(0);
    sink.i64_ne();
    emit_guard_if_exit(sink, constants, guard_idx, op, has_loop);
}

/// Common guard exit: condition is on stack (i32), emit if + exit.
fn emit_guard_if_exit(
    sink: &mut InstructionSink<'_>,
    constants: &HashMap<u32, i64>,
    guard_idx: u32,
    op: &Op,
    has_loop: bool,
) {
    sink.if_(BlockType::Empty);
    emit_guard_exit(sink, constants, guard_idx, op);
    if has_loop {
        sink.br(2); // if=0, loop=1, block=2
    }
    sink.end();
}

fn emit_guard_exit(
    sink: &mut InstructionSink<'_>,
    constants: &HashMap<u32, i64>,
    guard_idx: u32,
    op: &Op,
) {
    let fail_args = op
        .fail_args
        .as_ref()
        .map(|fa| fa.as_slice())
        .unwrap_or(&op.args);

    for (i, &arg_ref) in fail_args.iter().enumerate() {
        let offset = FRAME_SLOT_BASE + i as u64 * SLOT_SIZE;
        sink.local_get(0);
        emit_resolve(sink, constants, arg_ref);
        sink.i64_store(mem64(offset));
    }

    sink.local_get(0);
    sink.i64_const(guard_idx as i64);
    sink.i64_store(mem64(0));
}

// ── Binary ops ──

enum BinOp {
    I64Add,
    I64Sub,
    I64Mul,
    I64DivS,
    I64RemS,
    I64And,
    I64Or,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
}

fn apply_binop(sink: &mut InstructionSink<'_>, op: BinOp) {
    match op {
        BinOp::I64Add => {
            sink.i64_add();
        }
        BinOp::I64Sub => {
            sink.i64_sub();
        }
        BinOp::I64Mul => {
            sink.i64_mul();
        }
        BinOp::I64DivS => {
            sink.i64_div_s();
        }
        BinOp::I64RemS => {
            sink.i64_rem_s();
        }
        BinOp::I64And => {
            sink.i64_and();
        }
        BinOp::I64Or => {
            sink.i64_or();
        }
        BinOp::I64Xor => {
            sink.i64_xor();
        }
        BinOp::I64Shl => {
            sink.i64_shl();
        }
        BinOp::I64ShrS => {
            sink.i64_shr_s();
        }
        BinOp::I64ShrU => {
            sink.i64_shr_u();
        }
    }
}

fn emit_binop(
    sink: &mut InstructionSink<'_>,
    constants: &HashMap<u32, i64>,
    op: &Op,
    binop: BinOp,
) {
    let vi = op.pos.raw();
    if OpRef::from_raw(vi).is_constant() {
        return;
    }
    emit_resolve(sink, constants, op.arg(0));
    emit_resolve(sink, constants, op.arg(1));
    apply_binop(sink, binop);
    sink.local_set(1 + vi);
}

/// Overflow binary op: stores result in pos, overflow flag convention.
/// The overflow flag is not stored separately — GuardNoOverflow/GuardOverflow
/// is handled by checking after the fact (simplified for wasm MVP).
fn emit_ovf_binop(
    sink: &mut InstructionSink<'_>,
    constants: &HashMap<u32, i64>,
    op: &Op,
    binop: BinOp,
) {
    // For wasm MVP, just compute the result without overflow detection.
    // GuardNoOverflow/GuardOverflow are treated as always-pass.
    emit_binop(sink, constants, op, binop);
}

// ── Comparison ops ──

enum CmpOp {
    I64LtS,
    I64LeS,
    I64Eq,
    I64Ne,
    I64GtS,
    I64GeS,
    I64LtU,
    I64LeU,
    I64GtU,
    I64GeU,
}

fn apply_cmp(sink: &mut InstructionSink<'_>, op: CmpOp) {
    match op {
        CmpOp::I64LtS => {
            sink.i64_lt_s();
        }
        CmpOp::I64LeS => {
            sink.i64_le_s();
        }
        CmpOp::I64Eq => {
            sink.i64_eq();
        }
        CmpOp::I64Ne => {
            sink.i64_ne();
        }
        CmpOp::I64GtS => {
            sink.i64_gt_s();
        }
        CmpOp::I64GeS => {
            sink.i64_ge_s();
        }
        CmpOp::I64LtU => {
            sink.i64_lt_u();
        }
        CmpOp::I64LeU => {
            sink.i64_le_u();
        }
        CmpOp::I64GtU => {
            sink.i64_gt_u();
        }
        CmpOp::I64GeU => {
            sink.i64_ge_u();
        }
    }
}

// ── Float comparison helper ──

enum FloatCmp {
    Lt,
    Le,
    Eq,
    Ne,
    Gt,
    Ge,
}

fn emit_float_cmp(
    sink: &mut InstructionSink<'_>,
    constants: &HashMap<u32, i64>,
    op: &Op,
    cmp: FloatCmp,
) {
    let vi = op.pos.raw();
    if OpRef::from_raw(vi).is_constant() {
        return;
    }
    emit_resolve(sink, constants, op.arg(0));
    sink.f64_reinterpret_i64();
    emit_resolve(sink, constants, op.arg(1));
    sink.f64_reinterpret_i64();
    match cmp {
        FloatCmp::Lt => {
            sink.f64_lt();
        }
        FloatCmp::Le => {
            sink.f64_le();
        }
        FloatCmp::Eq => {
            sink.f64_eq();
        }
        FloatCmp::Ne => {
            sink.f64_ne();
        }
        FloatCmp::Gt => {
            sink.f64_gt();
        }
        FloatCmp::Ge => {
            sink.f64_ge();
        }
    }
    sink.i64_extend_i32_u();
    sink.local_set(1 + vi);
}

fn emit_cmp(sink: &mut InstructionSink<'_>, constants: &HashMap<u32, i64>, op: &Op, cmpop: CmpOp) {
    let vi = op.pos.raw();
    if OpRef::from_raw(vi).is_constant() {
        return;
    }
    emit_resolve(sink, constants, op.arg(0));
    emit_resolve(sink, constants, op.arg(1));
    apply_cmp(sink, cmpop);
    sink.i64_extend_i32_u();
    sink.local_set(1 + vi);
}

// ── Unary op helper ──

fn emit_unary_vi(
    sink: &mut InstructionSink<'_>,
    constants: &HashMap<u32, i64>,
    op: &Op,
    prefix: impl FnOnce(&mut InstructionSink<'_>),
    suffix: impl FnOnce(&mut InstructionSink<'_>),
) {
    let vi = op.pos.raw();
    if !OpRef::from_raw(vi).is_constant() {
        prefix(sink);
        emit_resolve(sink, constants, op.arg(0));
        suffix(sink);
        sink.local_set(1 + vi);
    }
}
