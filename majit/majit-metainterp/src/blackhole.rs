//! Blackhole interpreter: evaluates IR operations with concrete values.
//!
//! When a guard fails in compiled code, the blackhole interpreter can replay
//! the remaining operations from the guard point to the end of the trace,
//! using concrete values from the DeadFrame.
//!
//! This is the RPython equivalent of `rpython/jit/metainterp/blackhole.py`.

use std::collections::HashMap;

use crate::jitexc::JitException;
use crate::resume::{
    MaterializedVirtual, ResolvedPendingFieldWrite, ResumeData, ResumeLayoutSummary,
};
use majit_ir::{GcRef, Op, OpCode};

/// blackhole.py:1068 parity: typed payload decoded from merge-point
/// bytecode operands. Corresponds to the 6 lists in
/// ContinueRunningNormally(gi, gr, gf, ri, rr, rf).
///
/// Corresponds to ContinueRunningNormally(gi, gr, gf, ri, rr, rf)
/// in jitexc.py:53 — the typed portal args, NOT live locals.
#[derive(Debug, Clone, Default)]
pub struct MergePointArgs {
    pub green_int: Vec<i64>,
    pub green_ref: Vec<i64>,
    pub green_float: Vec<i64>,
    pub red_int: Vec<i64>,
    pub red_ref: Vec<i64>,
    pub red_float: Vec<i64>,
}

use crate::executor::{OpResult, TraceValues, ValueStore, execute_one};

/// Trait for IR-based blackhole memory access.
///
/// **Deprecated**: Part of the IR-based blackhole path.
/// The jitcode-based `BlackholeInterpreter` delegates memory access
/// through concrete function pointers in `JitCode.fn_ptrs` instead.
pub trait BlackholeMemory {
    /// Load a GC-managed field from `base + offset`.
    fn gc_load_i(&self, base: i64, offset: i64) -> i64 {
        let _ = (base, offset);
        0
    }
    /// Load a GC ref field.
    fn gc_load_r(&self, base: i64, offset: i64) -> i64 {
        let _ = (base, offset);
        0
    }
    /// Load a float field (returned as bits).
    fn gc_load_f(&self, base: i64, offset: i64) -> i64 {
        let _ = (base, offset);
        0
    }
    /// Store an int value into a GC object field.
    fn gc_store(&self, base: i64, offset: i64, value: i64) {
        let _ = (base, offset, value);
    }
    /// Load from array at `base + index * scale + offset`.
    fn gc_load_indexed_i(&self, base: i64, index: i64, scale: i64, offset: i64) -> i64 {
        let _ = (base, index, scale, offset);
        0
    }
    fn gc_load_indexed_r(&self, base: i64, index: i64, scale: i64, offset: i64) -> i64 {
        let _ = (base, index, scale, offset);
        0
    }
    fn gc_load_indexed_f(&self, base: i64, index: i64, scale: i64, offset: i64) -> i64 {
        let _ = (base, index, scale, offset);
        0
    }
    /// Store into array at `base + index * scale + offset`.
    fn gc_store_indexed(&self, base: i64, index: i64, scale: i64, offset: i64, value: i64) {
        let _ = (base, index, scale, offset, value);
    }
    /// Get array length.
    fn arraylen(&self, base: i64) -> i64 {
        let _ = base;
        0
    }
    /// Get string length.
    fn strlen(&self, base: i64) -> i64 {
        let _ = base;
        0
    }
    /// Call a function pointer with integer args, returning an integer.
    fn call_i(&self, func: i64, args: &[i64]) -> i64 {
        let _ = (func, args);
        0
    }
    /// Call a function pointer returning a ref (as i64 pointer bits).
    fn call_r(&self, func: i64, args: &[i64]) -> i64 {
        let _ = (func, args);
        0
    }
    /// Call a function pointer returning a float (as i64 bit-cast).
    fn call_f(&self, func: i64, args: &[i64]) -> i64 {
        let _ = (func, args);
        0
    }
    /// Call a function pointer returning void.
    fn call_n(&self, func: i64, args: &[i64]) {
        let _ = (func, args);
    }
}

/// Default no-op memory implementation (returns 0 for all loads).
pub struct DefaultBlackholeMemory;
impl BlackholeMemory for DefaultBlackholeMemory {}

/// Exception state tracked during blackhole execution.
///
/// Mirrors RPython's exception tracking in the meta-interpreter.
/// Guards like GUARD_EXCEPTION and GUARD_NO_EXCEPTION check this state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExceptionState {
    /// The exception class pointer (0 = no exception pending).
    pub exc_class: i64,
    /// The exception value pointer.
    pub exc_value: i64,
    /// executor.py: metainterp.ovf_flag — set by Int*Ovf operations,
    /// checked by GuardNoOverflow / GuardOverflow.
    pub ovf_flag: bool,
}

impl ExceptionState {
    /// Whether an exception is currently pending.
    pub fn is_pending(&self) -> bool {
        self.exc_class != 0
    }

    /// Set a pending exception.
    pub fn set(&mut self, exc_class: i64, exc_value: i64) {
        self.exc_class = exc_class;
        self.exc_value = exc_value;
    }

    /// Clear the pending exception and return (class, value).
    pub fn clear(&mut self) -> (i64, i64) {
        let cls = self.exc_class;
        let val = self.exc_value;
        self.exc_class = 0;
        self.exc_value = 0;
        (cls, val)
    }
}

/// Result of IR-based blackhole execution.
///
/// **Deprecated**: Part of the IR-based blackhole path.
pub enum BlackholeResult {
    /// Reached a Finish operation with output values and their types.
    /// compile.py:636 DoneWithThisFrameDescr*.handle_fail:
    /// the descr type determines how to interpret the value.
    Finish {
        op_index: usize,
        values: Vec<i64>,
        /// Types from the FINISH op's descr (e.g. [Type::Int] or [Type::Ref]).
        value_types: Vec<majit_ir::Type>,
    },
    /// Reached a Jump — loop back to header with these values.
    Jump { op_index: usize, values: Vec<i64> },
    /// A guard failed during blackhole execution.
    GuardFailed {
        guard_index: usize,
        fail_values: Vec<i64>,
    },
    /// A guard failed and resume-data virtuals were materialized for recovery.
    GuardFailedWithVirtuals {
        guard_index: usize,
        fail_values: Vec<i64>,
        materialized_virtuals: Vec<MaterializedVirtual>,
        pending_field_writes: Vec<ResolvedPendingFieldWrite>,
    },
    /// Abort: encountered an unhandled operation.
    Abort(String),
}

/// Evaluate IR operations sequentially with concrete i64 values.
///
/// **Deprecated**: RPython's blackhole.py executes jitcode bytecodes, not IR ops.
/// Use `BlackholeInterpreter` + `resume_in_blackhole()` instead.
/// This function will be removed once pyre generates jitcode and
/// `pyjitpl.rs` guard failure recovery switches to jitcode-based blackhole.
#[deprecated(note = "use BlackholeInterpreter for RPython-parity jitcode execution")]
pub fn blackhole_execute(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
) -> BlackholeResult {
    blackhole_execute_with_exception(
        ops,
        constants,
        initial_values,
        start_index,
        ExceptionState::default(),
    )
}

/// Evaluate with a concrete memory backend for load/store operations.
///
/// **Deprecated**: see `blackhole_execute`.
#[deprecated(note = "use BlackholeInterpreter for RPython-parity jitcode execution")]
pub fn blackhole_execute_with_memory(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    memory: &dyn BlackholeMemory,
) -> BlackholeResult {
    blackhole_execute_full(
        ops,
        constants,
        initial_values,
        start_index,
        ExceptionState::default(),
        memory,
    )
    .0
}

/// blackhole_execute_with_state + BlackholeMemory support.
/// RPython _run_forever parity: Jump loops back to Label.
pub fn blackhole_execute_full(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    initial_exception: ExceptionState,
    memory: &dyn BlackholeMemory,
) -> (BlackholeResult, ExceptionState) {
    let mut merged = initial_values.clone();
    for (&k, &v) in constants {
        merged.entry(k).or_insert(v);
    }
    let mut tv = TraceValues::from_hashmap(&merged);
    drop(merged);
    let mut exc_state = initial_exception;

    // RPython _run_forever parity: find the Label op (loop header) so
    // that Jump can loop back. The trace is [Label, ..., Jump(→Label)].
    let label_index = ops
        .iter()
        .position(|op| op.opcode == OpCode::Label)
        .unwrap_or(0);
    let label_inputarg_positions: Vec<u32> = ops
        .get(label_index)
        .map(|op| op.args.iter().map(|a| a.raw()).collect())
        .unwrap_or_default();

    let mut op_idx = start_index;

    while op_idx < ops.len() {
        let op = &ops[op_idx];
        let result = execute_one_with_memory(op, &tv, &mut exc_state, memory);

        match result {
            OpResult::Value(v) => {
                if !op.pos.is_none() {
                    tv.set(op.pos.raw(), v);
                }
            }
            OpResult::Void => {}
            OpResult::Finish(args) => {
                let vals: Vec<i64> = args.iter().map(|&r| tv.resolve(r)).collect();
                let vtypes = op
                    .descr
                    .as_ref()
                    .and_then(|d| d.as_fail_descr())
                    .map(|fd| fd.fail_arg_types().to_vec())
                    .unwrap_or_default();
                return (
                    BlackholeResult::Finish {
                        op_index: op_idx,
                        values: vals,
                        value_types: vtypes,
                    },
                    exc_state,
                );
            }
            OpResult::Jump(args) => {
                // RPython _run_forever parity: Jump loops back to Label.
                let vals: Vec<i64> = args.iter().map(|&r| tv.resolve(r)).collect();
                for (pos, val) in label_inputarg_positions.iter().zip(vals.iter()) {
                    tv.set(*pos, *val);
                }
                op_idx = label_index + 1;
                continue;
            }
            OpResult::GuardFailed => {
                let fail_values = if let Some(ref fail_args) = op.fail_args {
                    fail_args.iter().map(|&r| tv.resolve(r)).collect()
                } else {
                    vec![]
                };
                return (
                    BlackholeResult::GuardFailed {
                        guard_index: op_idx,
                        fail_values,
                    },
                    exc_state,
                );
            }
            OpResult::Unsupported(msg) => {
                return (BlackholeResult::Abort(msg), exc_state);
            }
        }
        op_idx += 1;
    }

    (
        BlackholeResult::Abort("reached end of ops without finish/jump".to_string()),
        exc_state,
    )
}

/// Dispatch an op using real memory access when a BlackholeMemory backend is available.
fn execute_one_with_memory(
    op: &Op,
    values: &(impl ValueStore + ?Sized),
    exc: &mut ExceptionState,
    memory: &dyn BlackholeMemory,
) -> OpResult {
    match op.opcode {
        // Memory access ops with real backend
        OpCode::GcLoadI => {
            let base = values.resolve(op.args[0]);
            let offset = values.resolve(op.args[1]);
            OpResult::Value(memory.gc_load_i(base, offset))
        }
        OpCode::GcLoadR => {
            let base = values.resolve(op.args[0]);
            let offset = values.resolve(op.args[1]);
            OpResult::Value(memory.gc_load_r(base, offset))
        }
        OpCode::GcLoadF => {
            let base = values.resolve(op.args[0]);
            let offset = values.resolve(op.args[1]);
            OpResult::Value(memory.gc_load_f(base, offset))
        }
        OpCode::GcStore => {
            let base = values.resolve(op.args[0]);
            let offset = values.resolve(op.args[1]);
            let value = values.resolve(op.args[2]);
            memory.gc_store(base, offset, value);
            OpResult::Void
        }
        OpCode::GcLoadIndexedI => {
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let scale = values.resolve(op.args[2]);
            let offset = values.resolve(op.args[3]);
            OpResult::Value(memory.gc_load_indexed_i(base, index, scale, offset))
        }
        OpCode::GcLoadIndexedR => {
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let scale = values.resolve(op.args[2]);
            let offset = values.resolve(op.args[3]);
            OpResult::Value(memory.gc_load_indexed_r(base, index, scale, offset))
        }
        OpCode::GcLoadIndexedF => {
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let scale = values.resolve(op.args[2]);
            let offset = values.resolve(op.args[3]);
            OpResult::Value(memory.gc_load_indexed_f(base, index, scale, offset))
        }
        OpCode::GcStoreIndexed => {
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let value = values.resolve(op.args[2]);
            let scale = values.resolve(op.args[3]);
            let offset = values.resolve(op.args[4]);
            memory.gc_store_indexed(base, index, scale, offset, value);
            OpResult::Void
        }
        OpCode::ArraylenGc => {
            let base = values.resolve(op.args[0]);
            OpResult::Value(memory.arraylen(base))
        }
        OpCode::Strlen | OpCode::Unicodelen => {
            let base = values.resolve(op.args[0]);
            OpResult::Value(memory.strlen(base))
        }
        // ── Field access via descr offset ──
        OpCode::GetfieldGcI | OpCode::GetfieldRawI | OpCode::GetfieldGcPureI => {
            let base = values.resolve(op.args[0]);
            let offset = op
                .descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .map_or(0, |f| f.offset() as i64);
            OpResult::Value(memory.gc_load_i(base, offset))
        }
        OpCode::GetfieldGcR | OpCode::GetfieldRawR | OpCode::GetfieldGcPureR => {
            let base = values.resolve(op.args[0]);
            let offset = op
                .descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .map_or(0, |f| f.offset() as i64);
            OpResult::Value(memory.gc_load_r(base, offset))
        }
        OpCode::GetfieldGcF | OpCode::GetfieldRawF | OpCode::GetfieldGcPureF => {
            let base = values.resolve(op.args[0]);
            let offset = op
                .descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .map_or(0, |f| f.offset() as i64);
            OpResult::Value(memory.gc_load_f(base, offset))
        }
        OpCode::SetfieldGc | OpCode::SetfieldRaw => {
            let base = values.resolve(op.args[0]);
            let value = values.resolve(op.args[1]);
            let offset = op
                .descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .map_or(0, |f| f.offset() as i64);
            memory.gc_store(base, offset, value);
            OpResult::Void
        }
        // ── Array access via descr ──
        OpCode::GetarrayitemGcI | OpCode::GetarrayitemRawI | OpCode::GetarrayitemGcPureI => {
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let (item_size, base_ofs) = op
                .descr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .map_or((1, 0), |a| (a.item_size() as i64, a.base_size() as i64));
            OpResult::Value(memory.gc_load_indexed_i(base, index, item_size, base_ofs))
        }
        OpCode::GetarrayitemGcR | OpCode::GetarrayitemRawR | OpCode::GetarrayitemGcPureR => {
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let (item_size, base_ofs) = op
                .descr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .map_or((1, 0), |a| (a.item_size() as i64, a.base_size() as i64));
            OpResult::Value(memory.gc_load_indexed_r(base, index, item_size, base_ofs))
        }
        OpCode::GetarrayitemGcF | OpCode::GetarrayitemRawF | OpCode::GetarrayitemGcPureF => {
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let (item_size, base_ofs) = op
                .descr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .map_or((1, 0), |a| (a.item_size() as i64, a.base_size() as i64));
            OpResult::Value(memory.gc_load_indexed_f(base, index, item_size, base_ofs))
        }
        OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let value = values.resolve(op.args[2]);
            let (item_size, base_ofs) = op
                .descr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .map_or((1, 0), |a| (a.item_size() as i64, a.base_size() as i64));
            memory.gc_store_indexed(base, index, item_size, base_ofs, value);
            OpResult::Void
        }
        // Call with real dispatch (all call variants)
        OpCode::CallI
        | OpCode::CallPureI
        | OpCode::CallMayForceI
        | OpCode::CallReleaseGilI
        | OpCode::CallLoopinvariantI => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_i(func, &args))
        }
        OpCode::CallR | OpCode::CallPureR | OpCode::CallMayForceR | OpCode::CallLoopinvariantR => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_r(func, &args))
        }
        OpCode::CallF
        | OpCode::CallPureF
        | OpCode::CallMayForceF
        | OpCode::CallReleaseGilF
        | OpCode::CallLoopinvariantF => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_f(func, &args))
        }
        OpCode::CallN
        | OpCode::CallPureN
        | OpCode::CallMayForceN
        | OpCode::CallReleaseGilN
        | OpCode::CallLoopinvariantN => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            memory.call_n(func, &args);
            OpResult::Void
        }
        // CallAssembler — delegate to call dispatch
        OpCode::CallAssemblerI => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_i(func, &args))
        }
        OpCode::CallAssemblerR => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_r(func, &args))
        }
        OpCode::CallAssemblerF => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_f(func, &args))
        }
        OpCode::CallAssemblerN => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            memory.call_n(func, &args);
            OpResult::Void
        }
        // CondCallValue — delegate to call dispatch
        OpCode::CondCallValueI => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_i(func, &args))
        }
        OpCode::CondCallValueR => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_r(func, &args))
        }
        OpCode::CondCallN => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            memory.call_n(func, &args);
            OpResult::Void
        }
        // Fall through to the default execute_one for everything else
        _ => execute_one(op, values, exc),
    }
}

/// **Deprecated**: IR-based blackhole. Will be replaced by jitcode-based
/// `resume_in_blackhole()` once pyre generates jitcode.
/// blackhole.py:1095 bhimpl_recursive_call parity:
/// Callback to execute a CallAssembler op during IR blackhole.
/// Receives the callee frame pointer (args[0]) and returns the result.
/// In RPython this is `portal_runner` via `cpu.bh_call_i`.
pub type CallAssemblerFn = dyn Fn(i64) -> i64;

pub(crate) fn blackhole_execute_with_state(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    initial_exception: ExceptionState,
) -> (BlackholeResult, ExceptionState) {
    blackhole_execute_with_state_ca(
        ops,
        constants,
        initial_values,
        start_index,
        initial_exception,
        None,
    )
}

pub(crate) fn blackhole_execute_with_state_ca(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    initial_exception: ExceptionState,
    call_assembler_fn: Option<&CallAssemblerFn>,
) -> (BlackholeResult, ExceptionState) {
    let mut merged = initial_values.clone();
    for (&k, &v) in constants {
        merged.entry(k).or_insert(v);
    }
    let mut tv = TraceValues::from_hashmap(&merged);
    drop(merged);
    let mut exc_state = initial_exception;

    // RPython _run_forever parity: find the Label op (loop header) so
    // that Jump can loop back. The trace is [Label, ..., Jump(→Label)].
    let label_index = ops
        .iter()
        .position(|op| op.opcode == OpCode::Label)
        .unwrap_or(0);
    // Label's args define the loop's inputargs (their OpRef positions).
    let label_inputarg_positions: Vec<u32> = ops
        .get(label_index)
        .map(|op| op.args.iter().map(|a| a.raw()).collect())
        .unwrap_or_default();

    let mut op_idx = start_index;

    // blackhole.py:1752 _run_forever: no iteration cap.
    while op_idx < ops.len() {
        let op = &ops[op_idx];
        let result = execute_one(op, &tv, &mut exc_state);

        match result {
            OpResult::Value(v) => {
                if !op.pos.is_none() {
                    tv.set(op.pos.raw(), v);
                }
            }
            OpResult::Void => {}
            // blackhole.py:1095 bhimpl_recursive_call parity:
            // CallAssembler ops invoke portal_runner via call_assembler_fn.
            OpResult::Unsupported(_)
                if call_assembler_fn.is_some() && op.opcode.is_call_assembler() =>
            {
                let ca_fn = call_assembler_fn.unwrap();
                let frame_ptr = tv.resolve(op.args[0]);
                let result = ca_fn(frame_ptr);
                if !op.pos.is_none() {
                    tv.set(op.pos.raw(), result);
                }
            }
            OpResult::Finish(args) => {
                let vals: Vec<i64> = args.iter().map(|&r| tv.resolve(r)).collect();
                let vtypes = op
                    .descr
                    .as_ref()
                    .and_then(|d| d.as_fail_descr())
                    .map(|fd| fd.fail_arg_types().to_vec())
                    .unwrap_or_default();
                return (
                    BlackholeResult::Finish {
                        op_index: op_idx,
                        values: vals,
                        value_types: vtypes,
                    },
                    exc_state,
                );
            }
            OpResult::Jump(args) => {
                // RPython _run_forever parity: Jump loops back to Label.
                // Map jump args → label inputargs and restart from label+1.
                let vals: Vec<i64> = args.iter().map(|&r| tv.resolve(r)).collect();
                for (pos, val) in label_inputarg_positions.iter().zip(vals.iter()) {
                    tv.set(*pos, *val);
                }
                op_idx = label_index + 1;
                continue;
            }
            OpResult::GuardFailed => {
                let fail_values = if let Some(ref fail_args) = op.fail_args {
                    fail_args.iter().map(|&r| tv.resolve(r)).collect()
                } else {
                    vec![]
                };
                return (
                    BlackholeResult::GuardFailed {
                        guard_index: op_idx,
                        fail_values,
                    },
                    exc_state,
                );
            }
            OpResult::Unsupported(msg) => {
                return (BlackholeResult::Abort(msg), exc_state);
            }
        }
        op_idx += 1;
    }

    (
        BlackholeResult::Abort("reached end of ops without finish/jump".to_string()),
        exc_state,
    )
}

/// Evaluate IR operations sequentially with concrete i64 values and an
/// already-pending exception state.
///
/// **Deprecated**: see `blackhole_execute`.
#[deprecated(note = "use BlackholeInterpreter for RPython-parity jitcode execution")]
pub fn blackhole_execute_with_exception(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    initial_exception: ExceptionState,
) -> BlackholeResult {
    blackhole_execute_with_state(
        ops,
        constants,
        initial_values,
        start_index,
        initial_exception,
    )
    .0
}

/// Blackhole execution with virtual object materialization.
#[deprecated(note = "use BlackholeInterpreter for RPython-parity jitcode execution")]
pub fn blackhole_with_virtuals(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    resume_data: Option<&ResumeData>,
) -> BlackholeResult {
    blackhole_with_recovery_layout(
        ops,
        constants,
        initial_values,
        start_index,
        resume_data,
        None,
    )
}

/// Blackhole execution with semantic-free resume-layout materialization.
#[deprecated(note = "use BlackholeInterpreter for RPython-parity jitcode execution")]
pub fn blackhole_with_resume_layout(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    resume_layout: Option<&ResumeLayoutSummary>,
) -> BlackholeResult {
    blackhole_with_recovery_layout(
        ops,
        constants,
        initial_values,
        start_index,
        None,
        resume_layout,
    )
}

fn blackhole_with_recovery_layout(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    resume_data: Option<&ResumeData>,
    resume_layout: Option<&ResumeLayoutSummary>,
) -> BlackholeResult {
    let result = blackhole_execute(ops, constants, initial_values, start_index);

    // If guard failed and we have resume data with virtuals, materialize them
    if let BlackholeResult::GuardFailed {
        guard_index,
        ref fail_values,
    } = result
    {
        if let Some(rd) = resume_data {
            if !rd.virtuals.is_empty() || !rd.pending_fields.is_empty() {
                let materialized = rd.materialize_virtuals(fail_values);
                let pending_field_writes =
                    ResumeData::resolve_pending_field_writes(&rd.pending_fields, fail_values);
                return BlackholeResult::GuardFailedWithVirtuals {
                    guard_index,
                    fail_values: fail_values.clone(),
                    materialized_virtuals: materialized,
                    pending_field_writes,
                };
            }
        } else if let Some(layout) = resume_layout {
            if layout.num_virtuals > 0 || layout.pending_field_count > 0 {
                return BlackholeResult::GuardFailedWithVirtuals {
                    guard_index,
                    fail_values: fail_values.clone(),
                    materialized_virtuals: layout.materialize_virtuals(fail_values),
                    pending_field_writes: layout.resolve_pending_field_writes(fail_values),
                };
            }
        }
    }

    result
}

// ============================================================================
// RPython blackhole.py parity: BlackholeInterpreter
//
// Jitcode-based blackhole execution. When a guard fails in compiled code,
// resume_in_blackhole reconstructs execution frames from resume data and
// runs jitcode bytecodes with concrete values, following ALL code paths
// (unlike trace IR which only has the traced path).
// ============================================================================

use crate::jitcode::{self, JitArgKind, JitCode, JitCodeRuntimeExt};
use crate::pyjitpl::{MIFrame, MIFrameStack};
use crate::pyjitpl::{
    call_int_function, call_void_function, eval_binop_f, eval_binop_i, eval_unary_f, eval_unary_i,
};

// ── BlackholeInterpBuilder: setup_insns infrastructure ──────────────
//
// RPython `blackhole.py:52-103` `class BlackholeInterpBuilder` combines
// pool management AND dispatch setup. pyre's existing
// `BlackholeInterpBuilder` (below, at the pool management section) is the
// pool manager. The `setup_insns` infrastructure (opcode table + dispatch
// table) is being added incrementally as Phase D of the RPython parity
// plan. Handler function pointers and `dispatch_loop` will be wired in
// as `bhimpl_*` methods are ported one by one.

/// Handler function signature for the codewriter-orthodox dispatch table.
///
/// RPython `blackhole.py:107` `handler(self, code, position) -> position`.
/// Each handler decodes operands from `code[position..]` based on its
/// argcodes, calls the corresponding `bhimpl_*` method on `bh`, writes
/// results, and returns the updated position.
pub type BhOpcodeHandler =
    fn(bh: &mut BlackholeInterpreter, code: &[u8], position: usize) -> Result<usize, DispatchError>;

/// Default handler installed by `setup_insns` for opcodes whose
/// `bhimpl_*` has not been ported to `wire_bhimpl_handlers` yet.
///
/// RPython `blackhole.py:76-80` `setup_insns` immediately resolves every
/// entry via `_get_method` and would raise `AttributeError` right there
/// if a `bhimpl_*` is missing. pyre defers the crash to dispatch time so
/// the runtime can construct a partial builder and `unwired_opnames()`
/// can survey gaps without terminating the process.
///
/// Named (not a closure) so `unwired_opnames()` can compare fn pointers
/// reliably — closures get a fresh anonymous type per call site.
fn unwired_handler_placeholder(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    panic!("missing bhimpl for opcode (use wire_handler to register)")
}

/// Return type of a blackhole frame.
///
/// RPython: `BlackholeInterpreter._return_type`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BhReturnType {
    Int,
    Ref,
    Float,
    Void,
}

/// Re-export BhDescr from codewriter::jitcode — shared descriptor type
/// between codewriter assembler and blackhole interpreter.
/// RPython `history.py:AbstractDescr` parity.
pub use majit_translate::jitcode::{BhCallDescr, BhDescr};

/// Per-jitdriver static data visible to the blackhole interpreter.
///
/// Mirrors the subset of `metainterp_sd.jitdrivers_sd[jdindex]` that
/// `bhimpl_recursive_call_*` and the jitcode::BC_RECURSIVE_CALL dispatch consult
/// (blackhole.py:1080-1099):
/// - `result_type` selects which `bhimpl_recursive_call_{v,i,r,f}`
///   method handles the call (blackhole.py:1080-1093).
/// - `portal_runner_ptr` / `mainjitcode_calldescr` feed
///   `get_portal_runner` (blackhole.py:1095-1099).
#[derive(Clone, Debug, Default)]
pub struct BhJitDriverSd {
    /// warmspot.py:449 `jd.result_type` projected to the blackhole
    /// dispatch char ('v','i','r','f') via `BhReturnType`.
    pub result_type: BhReturnType,
    /// warmspot.py:1010-1013 `jd.portal_runner_ptr` — full portal arg
    /// ABI: `fn(all_i, all_r, all_f) -> i64`.
    pub portal_runner_ptr: Option<fn(&[i64], &[i64], &[i64]) -> i64>,
    /// `jitdriver_sd.mainjitcode.calldescr` — CallDescr of the portal
    /// function returned by `get_portal_runner` for `bh_call_*`.
    pub mainjitcode_calldescr: BhCallDescr,
}

impl Default for BhReturnType {
    fn default() -> Self {
        // Entry-creation defaults to Void so empty drivers stay inert
        // until populated.
        BhReturnType::Void
    }
}

/// Signal from dispatch_one to run().
///
/// RPython: `LeaveFrame` exception + `except Exception` in blackhole.py run()
#[derive(Debug)]
pub enum DispatchError {
    /// Normal return from frame (RPython: LeaveFrame).
    LeaveFrame,
    /// Exception raised — must call handle_exception_in_frame.
    /// Carries the exception value (GcRef pointer as i64).
    RaiseException(i64),
    /// blackhole.py:1068-1069: raise ContinueRunningNormally(*args)
    /// Bottommost blackhole reached the merge point — restart portal.
    ContinueRunningNormally(MergePointArgs),
}

/// Jitcode-based blackhole interpreter.
///
/// Executes jitcode bytecodes with concrete values. Each instance
/// represents one execution frame. Frame chain is linked via
/// `nextblackholeinterp`.
///
/// RPython: `BlackholeInterpreter` class in blackhole.py:282-306.
///
/// RPython `__init__` receives `builder` and stores:
///   self.cpu = builder.cpu
///   self.dispatch_loop = builder.dispatch_loop
///   self.descrs = builder.descrs
///   self.op_catch_exception = builder.op_catch_exception
///   self.op_rvmprof_code = builder.op_rvmprof_code
pub struct BlackholeInterpreter {
    /// RPython `blackhole.py:286` `self.cpu = builder.cpu`.
    /// Reference to the backend trait for `bh_*` concrete execution.
    /// Raw pointer because the interpreter is pool-managed and
    /// the Backend outlives all interpreter instances.
    /// RPython `blackhole.py:286/56` `self.cpu = builder.cpu`.
    /// Backend trait for `bh_*` concrete execution. None until set.
    pub cpu: Option<&'static dyn majit_backend::Backend>,
    /// RPython `blackhole.py:288` `self.descrs = builder.descrs`.
    /// Descriptor table from the assembler. In RPython, `descrs` is a list
    /// of `AbstractDescr` objects carrying field offsets, array item sizes,
    /// etc. In pyre, we store raw offsets (usize) as a simplification —
    /// descriptor-index argcode ('d', 2 bytes) indexes into this table.
    /// RPython `blackhole.py:288` `self.descrs = builder.descrs`.
    /// Descriptor table — heterogeneous like RPython AbstractDescr list.
    pub descrs: Vec<BhDescr>,
    /// RPython `blackhole.py:289` `self.op_catch_exception = builder.op_catch_exception`.
    pub op_catch_exception: u8,
    /// RPython `blackhole.py:290` `self.op_rvmprof_code = builder.op_rvmprof_code`.
    pub op_rvmprof_code: u8,
    /// RPython `blackhole.py:289` `self.op_live = builder.op_live`.
    pub op_live: u8,
    /// Integer register bank.
    /// Indices 0..num_regs_i are working registers.
    /// Indices num_regs_i..num_regs_i+constants_i.len() hold constants.
    pub registers_i: Vec<i64>,
    /// Reference register bank.
    pub registers_r: Vec<i64>,
    /// Float register bank.
    pub registers_f: Vec<i64>,
    /// Temporary register for int return value.
    pub tmpreg_i: i64,
    /// Temporary register for ref return value.
    pub tmpreg_r: i64,
    /// Temporary register for float return value.
    pub tmpreg_f: i64,
    /// Current jitcode being executed.
    pub jitcode: std::sync::Arc<JitCode>,
    /// Current bytecode position (program counter).
    pub position: usize,
    /// Caller frame in the blackhole frame chain.
    pub nextblackholeinterp: Option<Box<BlackholeInterpreter>>,
    /// Return type of this frame.
    pub return_type: BhReturnType,
    /// True when run() hit jitcode::BC_ABORT (unsupported bytecode). Callers
    /// must not treat abort as DoneWithThisFrame — side effects from
    /// partial execution have corrupted state.
    pub aborted: bool,
    /// RPython blackhole.py handle_exception_in_frame parity:
    /// True when a residual call raised an exception (returned NULL ref).
    /// Unlike `aborted`, this indicates a Python-level exception that
    /// should propagate up the blackhole chain, not a JIT infrastructure error.
    pub got_exception: bool,
    /// Position of the last dispatched opcode (before position advances past operands).
    /// Used by handle_exception_in_frame for handler lookup — the faulting instruction
    /// PC, not the next instruction PC. Public so caller-chain propagation in
    /// call_jit.rs can set it to the suspended caller's position.
    pub last_opcode_position: usize,
    /// blackhole.py:391 exception_last_value: the caught exception object.
    /// Set when handle_exception_in_frame finds a handler.
    /// Read by CheckExcMatch and other exception opcodes in the handler.
    pub exception_last_value: i64,
    /// blackhole.py bhimpl_getfield_vable_*: pointer to the virtualizable
    /// object (e.g. PyFrame). Used by jitcode::BC_GETFIELD_VABLE_* bytecodes.
    /// Set during blackhole setup from the guard failure's virtualizable ptr.
    pub virtualizable_ptr: i64,
    /// Pointer to the VirtualizableInfo describing field offsets.
    /// Used by vable bytecodes to compute memory offsets.
    pub virtualizable_info: *const crate::virtualizable::VirtualizableInfo,
    /// blackhole.py:1095-1099 / warmspot.py:1010-1013 — per-jitdriver
    /// static data indexed by `jdindex`.  Each entry carries the
    /// portal_runner_ptr, mainjitcode.calldescr and result_type that
    /// `get_portal_runner` and the jitcode::BC_RECURSIVE_CALL dispatch consult
    /// (blackhole.py:1080-1093 selects `bhimpl_recursive_call_{v,i,r,f}`
    /// from `sd.jitdrivers_sd[jdindex].result_type`).
    ///
    /// Replaces the prior single-driver flat fields
    /// (`portal_runner_ptr`/`mainjitcode_calldescr`) so multi-driver
    /// dispatch matches upstream line-by-line.
    pub jitdrivers_sd: Vec<BhJitDriverSd>,
    /// Pyre: absolute start index of the operand stack in
    /// PyFrame.locals_cells_stack_w. RPython does not need this because
    /// its JitCode already operates in register space.
    pub virtualizable_stack_base: usize,
    /// RPython `blackhole.py:287` `self.dispatch_loop = builder.dispatch_loop`.
    /// Per-instance reference to the builder's dispatch table.  Cloned
    /// from the builder via `Arc::clone` in `acquire_interp` so wired
    /// handler lookup is one indirect call (`self.dispatch_table[opcode]`)
    /// without going back to the builder.
    ///
    /// `Default::default()` and `for_inline_callee` start with an empty
    /// table; `dispatch_step_with_fallback` falls back to `dispatch_one`
    /// when the table is empty or the opcode is out of range.
    pub(crate) dispatch_table: std::sync::Arc<Vec<BhOpcodeHandler>>,
}

// blackhole.py: last exception value from a residual call.
// Set by pyre call helpers (bh_call_fn_impl etc.) on error.
// Read by dispatch_one to populate exception_last_value on handler dispatch.
thread_local! {
    pub static BH_LAST_EXC_VALUE: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
}

// pyre residual-call helpers need the parent virtualizable pointer to
// reconstruct a PyFrame for call_user_function_plain. This TLS is not
// used by canonical vable field/array bytecodes anymore.
thread_local! {
    pub static BH_VABLE_PTR: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
}

// rvmprof integration lives in the `rvmprof::cintf` module — the
// structural analog of RPython's `rpython.rlib.rvmprof.cintf`. Blackhole
// calls through `rvmprof::cintf::jit_rvmprof_code` directly, matching
// `blackhole.py:416, 438, 1600` where the C intf function is invoked
// without any hook-registry indirection visible to dispatch code.
use crate::rvmprof::cintf::jit_rvmprof_code;

impl Default for BlackholeInterpreter {
    /// Sentinel-value interpreter used by
    /// `BlackholeInterpBuilder::acquire_interp`'s `unwrap_or_default()`
    /// and by `for_inline_callee` as the receiver for
    /// `clone_context_from`.  The 6 builder-shared fields stay at
    /// `u8::MAX` / empty until either `acquire_interp` populates them
    /// (RPython `blackhole.py:284-289` parity) or
    /// `clone_context_from(parent)` copies them from a parent
    /// interpreter.
    fn default() -> Self {
        Self {
            cpu: None,
            descrs: Vec::new(),
            // RPython blackhole.py:289 — copied from builder in `acquire_interp`.
            // Sentinel `u8::MAX` matches RPython's `insns.get('…', -1)` fallback.
            op_catch_exception: u8::MAX,
            op_rvmprof_code: u8::MAX,
            op_live: u8::MAX,
            registers_i: Vec::new(),
            registers_r: Vec::new(),
            registers_f: Vec::new(),
            tmpreg_i: 0,
            tmpreg_r: 0,
            tmpreg_f: 0,
            jitcode: std::sync::Arc::new(JitCode::default()),
            position: 0,
            nextblackholeinterp: None,
            return_type: BhReturnType::Void,
            aborted: false,
            got_exception: false,
            last_opcode_position: 0,
            exception_last_value: 0,
            virtualizable_ptr: 0,
            virtualizable_info: std::ptr::null(),
            jitdrivers_sd: Vec::new(),
            virtualizable_stack_base: 0,
            dispatch_table: std::sync::Arc::new(Vec::new()),
        }
    }
}

impl BlackholeInterpreter {
    fn reset_position_state(&mut self, position: usize) {
        self.position = position;
        self.aborted = false;
        self.got_exception = false;
        self.last_opcode_position = position;
        self.exception_last_value = 0;
    }

    fn init_register_file_from_i64s(
        regs: &mut Vec<i64>,
        num_regs_and_consts: usize,
        target_index: usize,
        constants: impl IntoIterator<Item = i64>,
    ) {
        if num_regs_and_consts > 0 {
            regs.clear();
            regs.resize(num_regs_and_consts, 0);
            for (i, c) in constants.into_iter().enumerate() {
                regs[target_index + i] = c;
            }
        } else {
            regs.clear();
        }
    }

    fn init_register_files_from_runtime_jitcode(&mut self, jitcode: &JitCode) {
        Self::init_register_file_from_i64s(
            &mut self.registers_i,
            jitcode.num_regs_and_consts_i(),
            jitcode.num_regs_i() as usize,
            jitcode.constants_i.iter().copied(),
        );
        Self::init_register_file_from_i64s(
            &mut self.registers_r,
            jitcode.num_regs_and_consts_r(),
            jitcode.num_regs_r() as usize,
            jitcode.constants_r.iter().copied(),
        );
        Self::init_register_file_from_i64s(
            &mut self.registers_f,
            jitcode.num_regs_and_consts_f(),
            jitcode.num_regs_f() as usize,
            jitcode.constants_f.iter().copied(),
        );
    }

    /// RPython `blackhole.py:313-337` register-file setup against the
    /// canonical codewriter `JitCode`.
    ///
    /// This intentionally stops short of installing `self.jitcode` for
    /// dispatch: pyre's execution path still expects the runtime adapter
    /// `crate::jitcode::JitCode` for `exec.*` pools. The extracted helper
    /// keeps the upstream-common part of `setposition` usable without a
    /// build→runtime conversion layer.
    #[cfg(test)]
    pub(crate) fn prepare_registers_for_canonical_jitcode(
        &mut self,
        jitcode: &majit_translate::jitcode::JitCode,
        position: usize,
    ) {
        let body = jitcode.body();
        Self::init_register_file_from_i64s(
            &mut self.registers_i,
            jitcode.num_regs_and_consts_i(),
            jitcode.num_regs_i(),
            body.constants_i.iter().copied(),
        );
        Self::init_register_file_from_i64s(
            &mut self.registers_r,
            jitcode.num_regs_and_consts_r(),
            jitcode.num_regs_r(),
            body.constants_r.iter().copied(),
        );
        Self::init_register_file_from_i64s(
            &mut self.registers_f,
            jitcode.num_regs_and_consts_f(),
            jitcode.num_regs_f(),
            body.constants_f.iter().copied(),
        );
        self.reset_position_state(position);
    }

    /// Spawn a fresh interpreter that shares `parent`'s builder-context
    /// fields (cpu, descrs, op_*, dispatch_table).  Used by
    /// `handler_inline_call_pyre_nested` for the callee frame of an
    /// inline-call: a recursive `BlackholeInterpreter` rather than
    /// `BlackholeInterpBuilder::acquire_interp` to avoid re-entering the
    /// thread-local builder pool already lent to the caller.
    ///
    /// PRE-EXISTING-ADAPTATION: RPython `blackhole.py:1279-1320`
    /// `bhimpl_inline_call_*` does not allocate a callee interpreter at
    /// all — it calls `cpu.bh_call_*(jitcode.fnaddr, ..., jitcode.calldescr)`
    /// directly because RPython's inline_call jitcode carries a native
    /// entry point.  pyre's sub-jitcodes are byte-interpreted, so a
    /// nested interpreter is the closest equivalent.  Within that
    /// adaptation, `for_inline_callee` mirrors the builder→interp 6-field
    /// copy that `acquire_interp` performs.
    pub fn for_inline_callee(parent: &Self) -> Self {
        let mut callee = Self::default();
        callee.clone_context_from(parent);
        callee
    }

    /// Copy the builder-shared context fields from `parent` onto `self`.
    /// Mirrors `BlackholeInterpBuilder::acquire_interp` (blackhole.rs:3902)
    /// — the same 6 fields that builder→interp normally propagates.
    /// Used by `handler_inline_call_pyre_nested` to give the callee
    /// `BlackholeInterpreter` the same `dispatch_table` (so a nested
    /// `BC_INLINE_CALL` byte routes through the same handler) plus the
    /// CPU/descrs/op_* slots the wired handlers would consult.  pyre-only:
    /// RPython `bhimpl_inline_call_*` (blackhole.py:1279-1320) does not
    /// allocate a callee interpreter; it calls `cpu.bh_call_*(jitcode.fnaddr,
    /// ...)` directly because RPython's inline_call jitcode carries a
    /// native entry point.  pyre's sub-jitcodes are byte-interpreted, so
    /// the callee path needs an interpreter that shares the parent's
    /// dispatch table to keep recursive inline_call working.
    pub fn clone_context_from(&mut self, parent: &Self) {
        // Six builder-shared fields per `BlackholeInterpBuilder::acquire_interp`
        // (`blackhole.rs:3825-3842`).
        self.cpu = parent.cpu;
        self.descrs = parent.descrs.clone();
        self.op_catch_exception = parent.op_catch_exception;
        self.op_rvmprof_code = parent.op_rvmprof_code;
        self.op_live = parent.op_live;
        self.dispatch_table = std::sync::Arc::clone(&parent.dispatch_table);
        // Virtualizable / jitdriver state: RPython `bhimpl_inline_call_*`
        // (`blackhole.py:1278-1320`) reaches the callee via
        // `cpu.bh_call_*(jitcode.fnaddr, ...)` so the callee never sees a
        // BlackholeInterpreter context at all — vable / recursive_call
        // bytecodes are unreachable through that path upstream.  pyre's
        // nested interpreter executes the sub-jitcode byte-by-byte, so a
        // sub-jitcode that contains a vable / recursive_call opcode would
        // otherwise read the callee's zero-defaults.  Mirror the parent's
        // state defensively — the closest equivalent to RPython's "callee
        // never has its own context" shape.
        self.virtualizable_ptr = parent.virtualizable_ptr;
        self.virtualizable_info = parent.virtualizable_info;
        self.jitdrivers_sd = parent.jitdrivers_sd.clone();
        self.virtualizable_stack_base = parent.virtualizable_stack_base;
    }

    /// blackhole.py:312 setposition
    ///
    /// Initialize register arrays for a jitcode and set the position.
    /// Allocates registers sized to hold both working regs and constants,
    /// then copies constants into the upper portion of each register array.
    pub fn setposition(&mut self, jitcode: std::sync::Arc<JitCode>, position: usize) {
        self.init_register_files_from_runtime_jitcode(&jitcode);
        // RPython: descrs are shared on the builder (setup_descrs).
        self.jitcode = jitcode;
        self.reset_position_state(position);
    }

    /// interp_jit.py:64 parity: fill dedicated portal red-arg registers
    /// with virtualizable_ptr (frame) and ec (execution context).
    ///
    /// The register indices are produced by pyre's portal codewriter and
    /// passed as side metadata. RPython `JitCode` has no portal-register
    /// fields, so they must not be stored on `jitcode.py`'s Rust port.
    ///
    /// Must be called AFTER virtualizable_ptr is assigned — setposition()
    /// runs before the caller sets virtualizable_ptr, so portal registers
    /// are filled as a separate step.
    pub fn fill_portal_registers(&mut self, frame_reg: u16, ec_reg: u16, ec: i64) {
        if frame_reg != u16::MAX && (frame_reg as usize) < self.registers_r.len() {
            self.registers_r[frame_reg as usize] = self.virtualizable_ptr;
        }
        if ec_reg != u16::MAX && (ec_reg as usize) < self.registers_r.len() {
            self.registers_r[ec_reg as usize] = ec;
        }
    }

    /// blackhole.py:1095-1099 `get_portal_runner(jdindex)`.
    ///
    /// ```python
    /// def get_portal_runner(self, jdindex):
    ///     jitdriver_sd = self.builder.metainterp_sd.jitdrivers_sd[jdindex]
    ///     fnptr = adr2int(jitdriver_sd.portal_runner_adr)
    ///     calldescr = jitdriver_sd.mainjitcode.calldescr
    ///     return fnptr, calldescr
    /// ```
    pub fn get_portal_runner(&self, jdindex: usize) -> (i64, BhCallDescr) {
        let jd = &self.jitdrivers_sd[jdindex];
        let fnptr = jd.portal_runner_ptr.map(|f| f as usize as i64).unwrap_or(0);
        (fnptr, jd.mainjitcode_calldescr.clone())
    }

    /// Resolve field descriptor offsets in this interpreter's descrs table.
    /// Delegates to the same logic as BlackholeInterpBuilder::resolve_field_offsets.
    pub fn resolve_field_offsets(&mut self, resolver: impl Fn(&str, &str) -> usize) {
        for descr in &mut self.descrs {
            if let BhDescr::Field {
                offset,
                name,
                owner,
                parent,
                ..
            } = descr
            {
                if *offset == 0 && !name.is_empty() {
                    *offset = resolver(owner, name);
                    if let Some(parent) = parent {
                        let full_name = if owner.is_empty() || name.contains('.') {
                            name.clone()
                        } else {
                            format!("{owner}.{name}")
                        };
                        if let Some(field) = parent
                            .all_fielddescrs
                            .iter_mut()
                            .find(|field| field.name == full_name)
                        {
                            field.offset = *offset;
                        }
                    }
                }
            }
        }
    }

    /// Resolve JitCode fnaddr values in this interpreter's descrs table.
    pub fn resolve_jitcode_fnaddrs(&mut self, resolver: impl Fn(usize) -> i64) {
        for descr in &mut self.descrs {
            if let BhDescr::JitCode {
                jitcode_index,
                fnaddr,
                ..
            } = descr
            {
                if *fnaddr == 0 {
                    *fnaddr = resolver(*jitcode_index);
                }
            }
        }
    }

    /// blackhole.py:1109-1116 bhimpl_recursive_call_r:
    ///   fnptr, calldescr = self.get_portal_runner(jdindex)
    ///   return self.cpu.bh_call_r(fnptr, greens_i+reds_i, greens_r+reds_r,
    ///                             greens_f+reds_f, calldescr)
    pub fn bhimpl_recursive_call_r(
        &self,
        jdindex: usize,
        greens_i: Vec<i64>,
        greens_r: Vec<i64>,
        greens_f: Vec<i64>,
        reds_i: Vec<i64>,
        reds_r: Vec<i64>,
        reds_f: Vec<i64>,
    ) -> majit_ir::GcRef {
        let (fnptr, calldescr) = self.get_portal_runner(jdindex);
        // blackhole.py:1113-1116: greens + reds merged per kind.
        let mut all_i = greens_i;
        all_i.extend(&reds_i);
        let mut all_r = greens_r;
        all_r.extend(&reds_r);
        let mut all_f = greens_f;
        all_f.extend(&reds_f);
        self.cpu()
            .bh_call_r(fnptr, Some(&all_i), Some(&all_r), Some(&all_f), &calldescr)
    }

    /// Set an integer register value.
    ///
    /// RPython: `BlackholeInterpreter.setarg_i(index, value)`
    pub fn setarg_i(&mut self, index: usize, value: i64) {
        self.registers_i[index] = value;
    }

    /// Set a reference register value.
    pub fn setarg_r(&mut self, index: usize, value: i64) {
        self.registers_r[index] = value;
    }

    /// Set a float register value.
    pub fn setarg_f(&mut self, index: usize, value: i64) {
        self.registers_f[index] = value;
    }

    /// Get the int return value from a completed frame.
    ///
    /// RPython: `BlackholeInterpreter.get_tmpreg_i()`
    pub fn get_tmpreg_i(&self) -> i64 {
        self.tmpreg_i
    }

    pub fn get_tmpreg_r(&self) -> i64 {
        self.tmpreg_r
    }

    pub fn get_tmpreg_f(&self) -> i64 {
        self.tmpreg_f
    }

    /// Copy register state from a tracing MIFrame into this blackhole frame.
    ///
    /// RPython: `BlackholeInterpreter._copy_data_from_miframe(miframe)`
    pub fn copy_data_from_miframe(&mut self, miframe: &MIFrame) {
        self.setposition(miframe.jitcode.clone(), miframe.pc);
        for i in 0..self.jitcode.num_regs_i() as usize {
            if let Some(val) = miframe.int_values.get(i).copied().flatten() {
                self.setarg_i(i, val);
            }
        }
        for i in 0..self.jitcode.num_regs_r() as usize {
            if let Some(val) = miframe.ref_values.get(i).copied().flatten() {
                self.setarg_r(i, val);
            }
        }
        for i in 0..self.jitcode.num_regs_f() as usize {
            if let Some(val) = miframe.float_values.get(i).copied().flatten() {
                self.setarg_f(i, val);
            }
        }
    }

    /// blackhole.py:385 cleanup_registers
    ///
    /// Clear reference registers to avoid keeping objects alive.
    /// Does not clear constants (they are prebuilt).
    pub fn cleanup_registers(&mut self) {
        for i in 0..self.jitcode.num_regs_r() as usize {
            if i < self.registers_r.len() {
                self.registers_r[i] = 0;
            }
        }
        self.exception_last_value = 0;
    }

    /// blackhole.py:393-394 `get_current_position_info`.
    ///
    /// RPython returns an offset into `metainterp_sd.liveness_info`
    /// (via `jitcode.get_live_vars_info(self.position, self.builder.op_live)`).
    /// Pyre uses a temporary per-interpreter sidecar until its codewriter
    /// emits `-live-` opcodes directly into JitCode.code.
    pub fn get_current_position_info(&self) -> usize {
        self.jitcode.get_live_vars_info(self.position, self.op_live)
    }

    /// blackhole.py:1653 _setup_return_value_i
    ///
    /// Connect the return of values from the called frame to the
    /// 'xxx_call_yyy' instructions from the caller frame.
    /// blackhole.py:1653 _setup_return_value_i
    pub fn setup_return_value_i(&mut self, result: i64) {
        // blackhole.py:1655-1656
        let reg_idx = self.jitcode.code[self.position - 1] as usize;
        self.registers_i[reg_idx] = result;
    }

    /// blackhole.py:1657 _setup_return_value_r
    pub fn setup_return_value_r(&mut self, result: i64) {
        // blackhole.py:1658-1659
        let reg_idx = self.jitcode.code[self.position - 1] as usize;
        self.registers_r[reg_idx] = result;
    }

    /// blackhole.py:1660 _setup_return_value_f
    pub fn setup_return_value_f(&mut self, result: i64) {
        // blackhole.py:1661-1662
        let reg_idx = self.jitcode.code[self.position - 1] as usize;
        self.registers_f[reg_idx] = result;
    }

    /// blackhole.py:1664 _done_with_this_frame
    ///
    /// Rare case: the blackhole interps all returned normally
    /// (in general we get a ContinueRunningNormally exception).
    fn done_with_this_frame(&self) -> JitException {
        match self.return_type {
            BhReturnType::Void => JitException::DoneWithThisFrameVoid,
            BhReturnType::Int => JitException::DoneWithThisFrameInt(self.get_tmpreg_i()),
            BhReturnType::Ref => {
                JitException::DoneWithThisFrameRef(GcRef(self.get_tmpreg_r() as usize))
            }
            BhReturnType::Float => {
                JitException::DoneWithThisFrameFloat(f64::from_bits(self.get_tmpreg_f() as u64))
            }
        }
    }

    /// blackhole.py:1679 _exit_frame_with_exception
    fn exit_frame_with_exception(&self, exc: i64) -> JitException {
        JitException::ExitFrameWithExceptionRef(GcRef(exc as usize))
    }

    /// blackhole.py:1647 _prepare_resume_from_failure
    ///
    /// Extract exception from the CPU deadframe on guard failure.
    /// Returns the exception value (0 if none).
    pub fn prepare_resume_from_failure(deadframe_exc: i64) -> i64 {
        // RPython: lltype.cast_opaque_ptr(rclass.OBJECTPTR,
        //          self.cpu.grab_exc_value(deadframe))
        deadframe_exc
    }

    /// blackhole.py:1612 _resume_mainloop
    ///
    /// Execute one frame and handle its completion.
    /// Returns Ok(exc) where exc is the exception to propagate to caller (0 = none),
    /// or Err(JitException) for JIT-level control flow exits.
    pub fn resume_mainloop(&mut self, current_exc: i64) -> Result<i64, JitException> {
        // blackhole.py:1614-1618
        // If there is a current exception, raise it now
        // (it may be caught by a catch_operation in this frame)
        if current_exc != 0 {
            if !self.handle_exception_in_frame(current_exc) {
                // No handler: propagate
                if self.nextblackholeinterp.is_none() {
                    return Err(self.exit_frame_with_exception(current_exc));
                }
                return Ok(current_exc);
            }
        }

        // blackhole.py:1621 — run the bytecode.
        // blackhole.py:1612 `_resume_mainloop` does not catch
        // ContinueRunningNormally; it propagates out to `_run_forever`
        // and then to `handle_jitexception` (warmspot.py:961).
        if let Some(args) = self.run() {
            return Err(JitException::ContinueRunningNormally {
                green_int: args.green_int,
                green_ref: args.green_ref,
                green_float: args.green_float,
                red_int: args.red_int,
                red_ref: args.red_ref,
                red_float: args.red_float,
            });
        }

        // Check for exception during execution
        if self.got_exception {
            let exc = self.exception_last_value;
            if self.nextblackholeinterp.is_none() {
                // blackhole.py:1629
                return Err(self.exit_frame_with_exception(exc));
            }
            return Ok(exc);
        }

        if self.aborted {
            // Abort is treated as an infrastructure error, not a normal exit.
            // The caller should not treat this as DoneWithThisFrame.
            if self.nextblackholeinterp.is_none() {
                return Err(JitException::DoneWithThisFrameVoid);
            }
            return Ok(0);
        }

        // blackhole.py:1633 — pass the frame's return value to the caller
        if self.nextblackholeinterp.is_none() {
            // blackhole.py:1635 — bottommost frame
            return Err(self.done_with_this_frame());
        }

        // Copy return values to locals before borrowing caller mutably
        let ret_type = self.return_type;
        let tmp_i = self.tmpreg_i;
        let tmp_r = self.tmpreg_r;
        let tmp_f = self.tmpreg_f;

        let caller = self.nextblackholeinterp.as_mut().unwrap();
        match ret_type {
            BhReturnType::Int => caller.setup_return_value_i(tmp_i),
            BhReturnType::Ref => caller.setup_return_value_r(tmp_r),
            BhReturnType::Float => caller.setup_return_value_f(tmp_f),
            BhReturnType::Void => {}
        }

        // blackhole.py:1645 — return no exception
        Ok(0)
    }

    // -- Bytecode reading helpers (matching MIFrame.next_u8/next_u16) --

    fn next_u8(&mut self) -> u8 {
        jitcode::read_u8(&self.jitcode.code, &mut self.position)
    }

    fn next_u16(&mut self) -> u16 {
        jitcode::read_u16(&self.jitcode.code, &mut self.position)
    }

    fn peek_u16_at(&self, pos: usize) -> Option<u16> {
        let code = &self.jitcode.code;
        if pos + 1 >= code.len() {
            return None;
        }
        Some((code[pos] as u16) | ((code[pos + 1] as u16) << 8))
    }

    fn runtime_bh_descr(&self, descr_idx: usize) -> Option<&BhDescr> {
        if let Some(entry) = self.jitcode.exec.descrs.get(descr_idx) {
            return entry.as_bh_descr();
        }
        self.descrs.get(descr_idx)
    }

    /// Read the static-field index from a canonical `VableField` descr
    /// pool entry at `pos`. Panics if the bytes do not point at a
    /// `BhDescr::VableField` — Stage 3c-3 collapses the dual-mode
    /// auto-detect, so callers must already be on the canonical layout
    /// (`assembler.py:165-167` + `:197-207`).
    fn vable_field_index_at(&self, pos: usize) -> usize {
        let idx = self
            .peek_u16_at(pos)
            .expect("vable_field_index_at: descr operand out of bounds");
        match self.runtime_bh_descr(idx as usize) {
            Some(BhDescr::VableField { index }) => *index,
            other => {
                panic!("vable_field_index_at: expected VableField at descr {idx}, got {other:?}")
            }
        }
    }

    /// Read the array-field index from a canonical
    /// (`VableArray`, `Array`) descr pool pair at `field_pos` /
    /// `array_pos`. Panics if either entry is missing or the wrong
    /// variant — see [`Self::vable_field_index_at`].
    fn vable_array_index_pair_at(&self, field_pos: usize, array_pos: usize) -> usize {
        let field_idx = self
            .peek_u16_at(field_pos)
            .expect("vable_array_index_pair_at: field descr out of bounds");
        let array_idx = self
            .peek_u16_at(array_pos)
            .expect("vable_array_index_pair_at: array descr out of bounds");
        let field_descr = self.runtime_bh_descr(field_idx as usize);
        let array_descr = self.runtime_bh_descr(array_idx as usize);
        match (field_descr, array_descr) {
            (Some(BhDescr::VableArray { index }), Some(BhDescr::Array { .. })) => *index,
            other => panic!(
                "vable_array_index_pair_at: expected (VableArray, Array) at descrs ({field_idx}, {array_idx}), got {other:?}"
            ),
        }
    }

    fn bh_binop_i(&mut self, opcode: OpCode) {
        let dst = self.next_u16() as usize;
        let lhs_idx = self.next_u16() as usize;
        let rhs_idx = self.next_u16() as usize;
        let lhs = self.registers_i[lhs_idx];
        let rhs = self.registers_i[rhs_idx];
        self.registers_i[dst] = eval_binop_i(opcode, lhs, rhs);
    }

    /// Per-opname ref binop helper returning int. Mirrors
    /// `bhimpl_{ptr_eq,ptr_ne,instance_ptr_eq,instance_ptr_ne}`.
    fn bh_binop_r_to_i(&mut self, opcode: OpCode) {
        let dst = self.next_u16() as usize;
        let lhs_idx = self.next_u16() as usize;
        let rhs_idx = self.next_u16() as usize;
        let lhs = self.registers_r[lhs_idx];
        let rhs = self.registers_r[rhs_idx];
        self.registers_i[dst] = match opcode {
            OpCode::PtrEq | OpCode::InstancePtrEq => (lhs == rhs) as i64,
            OpCode::PtrNe | OpCode::InstancePtrNe => (lhs != rhs) as i64,
            other => panic!("bh_binop_r_to_i: unsupported opcode {other:?}"),
        };
    }

    /// Per-opname float binop helper. See `bh_binop_i` for the pattern.
    /// RPython equivalents: `bhimpl_float_{add,sub,mul,truediv}`
    /// (`blackhole.py:663-687`).
    fn bh_binop_f(&mut self, opcode: OpCode) {
        let dst = self.next_u16() as usize;
        let lhs_idx = self.next_u16() as usize;
        let rhs_idx = self.next_u16() as usize;
        let lhs = self.registers_f[lhs_idx];
        let rhs = self.registers_f[rhs_idx];
        self.registers_f[dst] = eval_binop_f(opcode, lhs, rhs);
    }

    /// Unary ptr nullity helpers — `bhimpl_ptr_iszero` / `bhimpl_ptr_nonzero`.
    fn bh_ptr_nullity(&mut self, nonzero: bool) {
        let dst = self.next_u16() as usize;
        let src_idx = self.next_u16() as usize;
        let value = self.registers_r[src_idx];
        self.registers_i[dst] = if nonzero {
            (value != 0) as i64
        } else {
            (value == 0) as i64
        };
    }

    /// blackhole.py:1732-1748 _get_list_of_values parity.
    ///
    /// Decodes a bytecode-encoded register list: [length:u8][indices:u8...].
    /// Returns a Vec of register values looked up from the appropriate
    /// register file (registers_i for 'I', registers_r for 'R').
    fn _get_list_of_values_i(&mut self) -> Vec<i64> {
        let length = self.next_u8() as usize;
        let mut values = Vec::with_capacity(length);
        for _ in 0..length {
            let index = self.next_u8() as usize;
            values.push(self.registers_i.get(index).copied().unwrap_or(0));
        }
        values
    }

    /// blackhole.py:1733-1748 _get_list_of_values(self, code, position, 'R')
    fn _get_list_of_values_r(&mut self) -> Vec<i64> {
        let length = self.next_u8() as usize;
        let mut values = Vec::with_capacity(length);
        for _ in 0..length {
            let index = self.next_u8() as usize;
            values.push(self.registers_r.get(index).copied().unwrap_or(0));
        }
        values
    }

    fn _get_list_of_values_f(&mut self) -> Vec<i64> {
        let length = self.next_u8() as usize;
        let mut values = Vec::with_capacity(length);
        for _ in 0..length {
            let index = self.next_u8() as usize;
            values.push(self.registers_f.get(index).copied().unwrap_or(0));
        }
        values
    }

    /// blackhole.py:1066-1093 `bhimpl_jit_merge_point`. Decodes
    /// `@arguments("self", "i", "I", "R", "F", "I", "R", "F")` from
    /// `self.position` (advancing it past the jdindex byte and the six
    /// typed register lists), then either raises `ContinueRunningNormally`
    /// (bottommost level) or recursive_call's into the portal jitcode and
    /// raises `LeaveFrame`.
    ///
    /// `opcode` selects how the jdindex byte is decoded
    /// (blackhole.py:113-123): `BC_JIT_MERGE_POINT` reads it as a
    /// `registers_i` pool slot index (`'i'` argcode); `BC_JIT_MERGE_POINT_C`
    /// reads it as a raw signed byte (`'c'` argcode, assembler.py:312
    /// `USE_C_FORM`).
    pub(crate) fn bhimpl_jit_merge_point(&mut self, opcode: u8) -> Result<(), DispatchError> {
        let nbody_debug = std::env::var_os("PYRE_NBODY_DEBUG").is_some();
        let jdindex_byte = self.next_u8();
        let jdindex = if opcode == jitcode::BC_JIT_MERGE_POINT_C {
            (jdindex_byte as i8) as usize
        } else {
            self.registers_i[jdindex_byte as usize] as usize
        };
        let gi = self._get_list_of_values_i();
        let gr = self._get_list_of_values_r();
        let gf = self._get_list_of_values_f();
        let ri = self._get_list_of_values_i();
        let rr = self._get_list_of_values_r();
        let rf = self._get_list_of_values_f();
        if nbody_debug {
            eprintln!(
                "[nbody-debug][bh-jmp] pos={} jdindex={} gi={:?} gr={:?} ri={:?} rr={:#x?}",
                self.last_opcode_position, jdindex, gi, gr, ri, rr,
            );
        }

        if self.nextblackholeinterp.is_none() {
            // blackhole.py:1068-1069: bottommost level.
            //   raise ContinueRunningNormally(*args)
            return Err(DispatchError::ContinueRunningNormally(MergePointArgs {
                green_int: gi,
                green_ref: gr,
                green_float: gf,
                red_int: ri,
                red_ref: rr,
                red_float: rf,
            }));
        }
        // blackhole.py:1074-1093: recursive portal level.
        //   sd = self.builder.metainterp_sd
        //   result_type = sd.jitdrivers_sd[jdindex].result_type
        //   if result_type == 'v': bhimpl_recursive_call_v + void_return
        //   elif result_type == 'i': bhimpl_recursive_call_i + int_return
        //   elif result_type == 'r': bhimpl_recursive_call_r + ref_return
        //   elif result_type == 'f': bhimpl_recursive_call_f + float_return
        //   assert False
        let result_type = self.jitdrivers_sd[jdindex].result_type;
        match result_type {
            BhReturnType::Void => {
                self.bhimpl_recursive_call_v(jdindex, gi, gr, gf, ri, rr, rf);
                self.return_type = BhReturnType::Void;
            }
            BhReturnType::Int => {
                let x = self.bhimpl_recursive_call_i(jdindex, gi, gr, gf, ri, rr, rf);
                self.tmpreg_i = x;
                self.return_type = BhReturnType::Int;
            }
            BhReturnType::Ref => {
                let x = self.bhimpl_recursive_call_r(jdindex, gi, gr, gf, ri, rr, rf);
                self.tmpreg_r = x.0 as i64;
                self.return_type = BhReturnType::Ref;
            }
            BhReturnType::Float => {
                let x = self.bhimpl_recursive_call_f(jdindex, gi, gr, gf, ri, rr, rf);
                self.tmpreg_f = x.to_bits() as i64;
                self.return_type = BhReturnType::Float;
            }
        }
        Err(DispatchError::LeaveFrame)
    }

    /// pyre-only `BC_ABORT_PERMANENT` body. Routes a TLS-stashed exception
    /// through `RaiseException` (so `handle_exception_in_frame` can pick
    /// the except handler); falls back to `aborted = true` + `LeaveFrame`
    /// when no exception is set. RPython has no direct counterpart — the
    /// canonical `abort_permanent/` handler is emitted by pyre's codegen
    /// for fail-paths that should always terminate the blackhole frame.
    pub(crate) fn bhimpl_abort_permanent(&mut self) -> Result<(), DispatchError> {
        let exc = BH_LAST_EXC_VALUE.with(|c| c.get());
        if exc != 0 {
            BH_LAST_EXC_VALUE.with(|c| c.set(0));
            return Err(DispatchError::RaiseException(exc));
        }
        self.aborted = true;
        Err(DispatchError::LeaveFrame)
    }

    fn finished(&self) -> bool {
        self.position >= self.jitcode.code.len()
    }

    /// blackhole.py:1600-1603 bhimpl_rvmprof_code.
    pub fn bhimpl_rvmprof_code(&self, leaving: i64, unique_id: i64) {
        jit_rvmprof_code(leaving, unique_id);
    }

    /// blackhole.py:396 handle_exception_in_frame: check if the current
    /// position has an immediately-following `catch_exception/L`.
    pub fn handle_exception_in_frame(&mut self, exc_value: i64) -> bool {
        let code = &self.jitcode.code;
        let mut position = self.position;
        if position < code.len() {
            let mut opcode = code[position];
            if opcode == self.op_live {
                position += majit_translate::liveness::OFFSET_SIZE + 1;
                if position >= code.len() {
                    return false;
                }
                opcode = code[position];
            }
            if opcode == self.op_catch_exception {
                self.exception_last_value = exc_value;
                if position + 2 >= code.len() {
                    return false;
                }
                let target = (code[position + 1] as usize) | ((code[position + 2] as usize) << 8);
                self.position = target;
                // blackhole.py:407 parity: once the handler is dispatched the
                // residual-call TLS slot is stale. Clear it so a subsequent
                // opcode that reads `BH_LAST_EXC_VALUE` without issuing a new
                // call can't pick up this already-caught exception.
                BH_LAST_EXC_VALUE.with(|c| c.set(0));
                return true;
            }
            if opcode == self.op_rvmprof_code {
                // blackhole.py:412-420: on exception immediately before a
                // `rvmprof_code/ii`, run the leaving hook and continue
                // propagating.
                let leaving = self.registers_i[code[position + 1] as usize];
                let unique_id = self.registers_i[code[position + 2] as usize];
                assert_eq!(leaving, 1);
                self.bhimpl_rvmprof_code(leaving, unique_id);
            }
        }
        false
    }

    /// blackhole.py:424-439 handle_rvmprof_enter.
    pub fn handle_rvmprof_enter(&mut self) {
        let code = &self.jitcode.code;
        let mut position = self.position;
        let mut opcode = code[position];
        if opcode == self.op_live {
            position += majit_translate::liveness::OFFSET_SIZE + 1;
            opcode = code[position];
        }
        if opcode == self.op_rvmprof_code {
            let leaving = self.registers_i[code[position + 1] as usize];
            let unique_id = self.registers_i[code[position + 2] as usize];
            if leaving == 1 {
                self.bhimpl_rvmprof_code(0, unique_id);
            }
        }
    }

    // -- Call argument reading --

    fn read_call_arg(&self, kind: JitArgKind, reg: u16) -> i64 {
        match kind {
            JitArgKind::Int => self.registers_i[reg as usize],
            JitArgKind::Ref => self.registers_r[reg as usize],
            JitArgKind::Float => self.registers_f[reg as usize],
        }
    }

    /// Execute the dispatch loop on the current jitcode.
    ///
    /// RPython: `BlackholeInterpreter.run()` catches `LeaveFrame` and breaks,
    /// catches exceptions and calls `handle_exception_in_frame`.
    /// Returns `Some(args)` for `ContinueRunningNormally` (RPython: raise
    /// jitexc.ContinueRunningNormally propagates through run→_run_forever).
    pub fn run(&mut self) -> Option<MergePointArgs> {
        let saved_bh_vable = BH_VABLE_PTR.with(|c| c.get());
        BH_VABLE_PTR.with(|c| c.set(self.virtualizable_ptr));
        let bh_depth = unsafe {
            majit_gc::shadow_stack::push_bh_regs(&mut self.registers_r, &mut self.tmpreg_r)
        };
        let result = self.run_inner();
        majit_gc::shadow_stack::pop_bh_regs_to(bh_depth);
        BH_VABLE_PTR.with(|c| c.set(saved_bh_vable));
        result
    }

    fn run_inner(&mut self) -> Option<MergePointArgs> {
        let trace = crate::majit_log_enabled();
        loop {
            if self.finished() {
                if trace {
                    eprintln!(
                        "[bh-trace] finished at pos={} reg0={}",
                        self.position,
                        self.registers_i.get(0).copied().unwrap_or(-1)
                    );
                }
                return None;
            }
            let pos_before = self.position;
            self.last_opcode_position = pos_before;
            let opcode = self.next_u8();
            if trace {
                eprintln!(
                    "[bh-trace] pos={} op={} reg0={} reg1={}",
                    pos_before,
                    opcode,
                    self.registers_i.get(0).copied().unwrap_or(-1),
                    self.registers_i.get(1).copied().unwrap_or(-1),
                );
            }
            match self.dispatch_step_with_fallback(opcode) {
                Ok(()) => {}
                Err(DispatchError::LeaveFrame) => {
                    if trace {
                        eprintln!(
                            "[bh-trace] leave-frame at pos={} ret_type={:?}",
                            pos_before, self.return_type,
                        );
                    }
                    return None;
                }
                Err(DispatchError::ContinueRunningNormally(args)) => {
                    // blackhole.py:1068: raise ContinueRunningNormally(*args)
                    // Propagates out of run() like RPython's JitException.
                    if trace {
                        eprintln!("[bh-trace] ContinueRunningNormally at pos={}", pos_before);
                    }
                    return Some(args);
                }
                Err(DispatchError::RaiseException(exc)) => {
                    // blackhole.py:359-361: except Exception → handle_exception_in_frame
                    if trace {
                        eprintln!("[bh-trace] exception at pos={} exc=0x{:x}", pos_before, exc);
                    }
                    if self.handle_exception_in_frame(exc) {
                        // Handler found, continue execution at handler target
                        continue;
                    }
                    // No handler: propagate exception via got_exception flag
                    self.got_exception = true;
                    self.exception_last_value = exc;
                    return None;
                }
            }
        }
    }

    /// Dispatch a single bytecode instruction through the per-instance
    /// `dispatch_table`, falling back to the legacy `dispatch_one` match
    /// when the table entry is the unwired placeholder or the table is
    /// shorter than `opcode + 1`.
    ///
    /// RPython parity: `blackhole.py:83-100` `dispatch_loop` calls
    /// `self.dispatch_table[opcode_byte](self, code, position)`
    /// unconditionally — every opcode is wired before the loop runs.
    /// Pyre's pyre-legacy assembler (`majit-metainterp/src/jitcode/
    /// assembler.rs::JitCodeBuilder`) emits `BC_*` bytes that have no
    /// canonical opname-key registration yet, so wired-handler coverage
    /// is incomplete.  The placeholder fallback bridges that gap by
    /// routing unwired opcodes through `dispatch_one` until the
    /// per-family migration of Stage 2 closes them.
    ///
    /// `position` invariant: caller is `run_inner`, which has already
    /// advanced `self.position` past the opcode byte via `next_u8`.
    /// Wired handlers expect `position` to point at the first operand
    /// byte (handler signature `(bh, code, position) -> Result<usize>`)
    /// and return the post-operand position; we then store it back into
    /// `self.position`.  Fallback handlers (`dispatch_one`) mutate
    /// `self.position` directly, so no sync is needed for that branch.
    fn dispatch_step_with_fallback(&mut self, opcode: u8) -> Result<(), DispatchError> {
        let placeholder_addr = unwired_handler_placeholder as *const () as usize;
        let table_handler = self
            .dispatch_table
            .get(opcode as usize)
            .copied()
            .filter(|h| (*h as *const () as usize) != placeholder_addr);
        let Some(handler) = table_handler else {
            // Stage 2 Slice 2.1 probe — env-gated logging of fallback
            // arrivals so Slice 2.2/2.3 can decide whether the legacy
            // `dispatch_one` path is reachable in production.  Removed in
            // Slice 2.3 once the fallback turns into a panic.
            if std::env::var_os("MAJIT_DISPATCH_ONE_PROBE").is_some() {
                eprintln!(
                    "[bh-probe] dispatch_one fallback byte={opcode:#x} \
                     pos={} table_len={} jitcode={:?}",
                    self.last_opcode_position,
                    self.dispatch_table.len(),
                    self.jitcode.name,
                );
            }
            return self.dispatch_one(opcode);
        };
        // Clone the Arc to detach the `code` borrow from `self`, so the
        // handler can take `&mut self` without aliasing the code slice.
        // `Arc::clone` is a single atomic increment.
        let jitcode_arc = std::sync::Arc::clone(&self.jitcode);
        let code: &[u8] = &jitcode_arc.code;
        let new_pos = handler(self, code, self.position)?;
        self.position = new_pos;
        Ok(())
    }

    /// Dispatch a single bytecode instruction with concrete execution.
    ///
    /// RPython: bytecode dispatch in `dispatch_loop()`, each `bhimpl_*` method
    fn dispatch_one(&mut self, opcode: u8) -> Result<(), DispatchError> {
        match opcode {
            jitcode::BC_LIVE => {
                // blackhole.py:1605 bhimpl_live(pc): return pc + OFFSET_SIZE.
                self.position += majit_translate::liveness::OFFSET_SIZE;
            }
            jitcode::BC_CATCH_EXCEPTION => {
                // blackhole.py:969 bhimpl_catch_exception(target): no-op in
                // normal execution; skip the 2-byte label operand.
                self.position += 2;
            }
            jitcode::BC_LAST_EXCEPTION => {
                let dst = self.next_u16() as usize;
                let exc_obj = self.exception_last_value;
                let typeptr = if let Some(cpu) = self.cpu {
                    cpu.bh_classof(exc_obj)
                } else {
                    exc_obj
                };
                self.registers_i[dst] = typeptr;
            }
            jitcode::BC_LAST_EXC_VALUE => {
                let dst = self.next_u16() as usize;
                self.registers_r[dst] = self.exception_last_value;
            }
            jitcode::BC_RVMPROF_CODE => {
                let leaving_idx = self.next_u8() as usize;
                let unique_id_idx = self.next_u8() as usize;
                let leaving = self.registers_i[leaving_idx];
                let unique_id = self.registers_i[unique_id_idx];
                self.bhimpl_rvmprof_code(leaving, unique_id);
            }
            // RPython `blackhole.py:638-646` `bhimpl_{int,ref,float}_copy`.
            // Operand order follows the `i>i` / `r>r` / `f>f` argcodes
            // (`assembler.py:165-174`): source register is encoded
            // first, destination last.
            jitcode::BC_MOVE_I => {
                let src = self.next_u16() as usize;
                let dst = self.next_u16() as usize;
                self.registers_i[dst] = self.registers_i[src];
            }
            jitcode::BC_MOVE_R => {
                let src = self.next_u16() as usize;
                let dst = self.next_u16() as usize;
                self.registers_r[dst] = self.registers_r[src];
            }
            jitcode::BC_MOVE_F => {
                let src = self.next_u16() as usize;
                let dst = self.next_u16() as usize;
                self.registers_f[dst] = self.registers_f[src];
            }
            jitcode::BC_INT_ADD => self.bh_binop_i(OpCode::IntAdd),
            jitcode::BC_INT_SUB => self.bh_binop_i(OpCode::IntSub),
            jitcode::BC_INT_MUL => self.bh_binop_i(OpCode::IntMul),
            // Upstream `blackhole.py` has no `bhimpl_int_floordiv` /
            // `bhimpl_int_mod`: `jtransform.py:575-577` rewrites both
            // primitives to `direct_call(ll_int_py_*)` before jitcode
            // emission, so the bytecode never appears in real RPython
            // jitcodes.  Pyre's β' redirect at
            // `majit-translate/src/codegen.rs::generated_binary_int_value`
            // covers the runtime trace path; Slice 4 + Slice 5 + the
            // sibling pyjitpl arm trap producers earlier in the chain.
            // Reaching this dispatch arm in the BH interpreter means
            // the upstream-absent `BC_INT_FLOORDIV` / `BC_INT_MOD` byte
            // somehow got into a jitcode array — fail loud immediately.
            jitcode::BC_INT_FLOORDIV => panic!(
                "BC_INT_FLOORDIV reached blackhole::run; upstream `blackhole.py` has no \
                 `bhimpl_int_floordiv` (rewritten to `direct_call(ll_int_py_div)` at \
                 `jtransform.py:576`).  See Task #97 for the deletion sequence."
            ),
            jitcode::BC_INT_MOD => panic!(
                "BC_INT_MOD reached blackhole::run; upstream `blackhole.py` has no \
                 `bhimpl_int_mod` (rewritten to `direct_call(ll_int_py_mod)` at \
                 `jtransform.py:577`).  See Task #97 for the deletion sequence."
            ),
            jitcode::BC_INT_AND => self.bh_binop_i(OpCode::IntAnd),
            jitcode::BC_INT_OR => self.bh_binop_i(OpCode::IntOr),
            jitcode::BC_INT_XOR => self.bh_binop_i(OpCode::IntXor),
            jitcode::BC_INT_LSHIFT => self.bh_binop_i(OpCode::IntLshift),
            jitcode::BC_INT_RSHIFT => self.bh_binop_i(OpCode::IntRshift),
            jitcode::BC_INT_EQ => self.bh_binop_i(OpCode::IntEq),
            jitcode::BC_INT_NE => self.bh_binop_i(OpCode::IntNe),
            jitcode::BC_INT_LT => self.bh_binop_i(OpCode::IntLt),
            jitcode::BC_INT_LE => self.bh_binop_i(OpCode::IntLe),
            jitcode::BC_INT_GT => self.bh_binop_i(OpCode::IntGt),
            jitcode::BC_INT_GE => self.bh_binop_i(OpCode::IntGe),
            jitcode::BC_UINT_RSHIFT => self.bh_binop_i(OpCode::UintRshift),
            jitcode::BC_UINT_MUL_HIGH => self.bh_binop_i(OpCode::UintMulHigh),
            jitcode::BC_UINT_LT => self.bh_binop_i(OpCode::UintLt),
            jitcode::BC_UINT_LE => self.bh_binop_i(OpCode::UintLe),
            jitcode::BC_UINT_GT => self.bh_binop_i(OpCode::UintGt),
            jitcode::BC_UINT_GE => self.bh_binop_i(OpCode::UintGe),
            jitcode::BC_INT_NEG => {
                let dst = self.next_u16() as usize;
                let src_idx = self.next_u16() as usize;
                let value = self.registers_i[src_idx];
                self.registers_i[dst] = eval_unary_i(OpCode::IntNeg, value);
            }
            jitcode::BC_INT_INVERT => {
                let dst = self.next_u16() as usize;
                let src_idx = self.next_u16() as usize;
                let value = self.registers_i[src_idx];
                self.registers_i[dst] = eval_unary_i(OpCode::IntInvert, value);
            }
            // Ref-valued comparison / nullity primitives — RPython
            // `blackhole.py` `bhimpl_{ptr_eq,ptr_ne,instance_ptr_eq,
            // instance_ptr_ne,ptr_iszero,ptr_nonzero}`.
            jitcode::BC_PTR_EQ => self.bh_binop_r_to_i(OpCode::PtrEq),
            jitcode::BC_PTR_NE => self.bh_binop_r_to_i(OpCode::PtrNe),
            jitcode::BC_INSTANCE_PTR_EQ => self.bh_binop_r_to_i(OpCode::InstancePtrEq),
            jitcode::BC_INSTANCE_PTR_NE => self.bh_binop_r_to_i(OpCode::InstancePtrNe),
            jitcode::BC_PTR_ISZERO => self.bh_ptr_nullity(false),
            jitcode::BC_PTR_NONZERO => self.bh_ptr_nullity(true),
            jitcode::BC_FLOAT_ADD => self.bh_binop_f(OpCode::FloatAdd),
            jitcode::BC_FLOAT_SUB => self.bh_binop_f(OpCode::FloatSub),
            jitcode::BC_FLOAT_MUL => self.bh_binop_f(OpCode::FloatMul),
            jitcode::BC_FLOAT_TRUEDIV => self.bh_binop_f(OpCode::FloatTrueDiv),
            jitcode::BC_FLOAT_NEG => {
                let dst = self.next_u16() as usize;
                let src_idx = self.next_u16() as usize;
                let value = self.registers_f[src_idx];
                self.registers_f[dst] = eval_unary_f(OpCode::FloatNeg, value);
            }
            jitcode::BC_FLOAT_ABS => {
                let dst = self.next_u16() as usize;
                let src_idx = self.next_u16() as usize;
                let value = self.registers_f[src_idx];
                self.registers_f[dst] = eval_unary_f(OpCode::FloatAbs, value);
            }
            jitcode::BC_GOTO_IF_NOT_INT_IS_TRUE => {
                let cond_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let cond = self.registers_i[cond_idx];
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[bh-trace] GOTO_IF_NOT_INT_IS_TRUE cond_idx={} cond={} target={} taken={}",
                        cond_idx,
                        cond,
                        target,
                        cond == 0
                    );
                }
                if cond == 0 {
                    self.position = target;
                }
            }
            // blackhole.py:872-911 bhimpl_goto_if_not_int_*:
            //   if a <cmp> b: return pc else return target
            // i.e. take the target when the comparison is FALSE.
            jitcode::BC_GOTO_IF_NOT_INT_LT => {
                let a_idx = self.next_u16() as usize;
                let b_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = self.registers_i[a_idx];
                let b = self.registers_i[b_idx];
                if !(a < b) {
                    self.position = target;
                }
            }
            jitcode::BC_GOTO_IF_NOT_INT_LE => {
                let a_idx = self.next_u16() as usize;
                let b_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = self.registers_i[a_idx];
                let b = self.registers_i[b_idx];
                if !(a <= b) {
                    self.position = target;
                }
            }
            jitcode::BC_GOTO_IF_NOT_INT_EQ => {
                let a_idx = self.next_u16() as usize;
                let b_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = self.registers_i[a_idx];
                let b = self.registers_i[b_idx];
                if a != b {
                    self.position = target;
                }
            }
            jitcode::BC_GOTO_IF_NOT_INT_NE => {
                let a_idx = self.next_u16() as usize;
                let b_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = self.registers_i[a_idx];
                let b = self.registers_i[b_idx];
                if a == b {
                    self.position = target;
                }
            }
            jitcode::BC_GOTO_IF_NOT_INT_GT => {
                let a_idx = self.next_u16() as usize;
                let b_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = self.registers_i[a_idx];
                let b = self.registers_i[b_idx];
                if !(a > b) {
                    self.position = target;
                }
            }
            jitcode::BC_GOTO_IF_NOT_INT_GE => {
                let a_idx = self.next_u16() as usize;
                let b_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = self.registers_i[a_idx];
                let b = self.registers_i[b_idx];
                if !(a >= b) {
                    self.position = target;
                }
            }
            // blackhole.py:916-920 bhimpl_goto_if_not_int_is_zero:
            //   if not a: return pc else return target
            // i.e. take the target when `a != 0` (truthy).
            jitcode::BC_GOTO_IF_NOT_INT_IS_ZERO => {
                let a_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = self.registers_i[a_idx];
                if a != 0 {
                    self.position = target;
                }
            }
            // blackhole.py:751-798 bhimpl_goto_if_not_float_*:
            //   a = longlong.getrealfloat(a); b = longlong.getrealfloat(b)
            //   if a <cmp> b: return pc else return target
            jitcode::BC_GOTO_IF_NOT_FLOAT_LT => {
                let a_idx = self.next_u16() as usize;
                let b_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = f64::from_bits(self.registers_f[a_idx] as u64);
                let b = f64::from_bits(self.registers_f[b_idx] as u64);
                if !(a < b) {
                    self.position = target;
                }
            }
            jitcode::BC_GOTO_IF_NOT_FLOAT_LE => {
                let a_idx = self.next_u16() as usize;
                let b_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = f64::from_bits(self.registers_f[a_idx] as u64);
                let b = f64::from_bits(self.registers_f[b_idx] as u64);
                if !(a <= b) {
                    self.position = target;
                }
            }
            jitcode::BC_GOTO_IF_NOT_FLOAT_EQ => {
                let a_idx = self.next_u16() as usize;
                let b_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = f64::from_bits(self.registers_f[a_idx] as u64);
                let b = f64::from_bits(self.registers_f[b_idx] as u64);
                if a != b {
                    self.position = target;
                }
            }
            jitcode::BC_GOTO_IF_NOT_FLOAT_NE => {
                let a_idx = self.next_u16() as usize;
                let b_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = f64::from_bits(self.registers_f[a_idx] as u64);
                let b = f64::from_bits(self.registers_f[b_idx] as u64);
                if a == b {
                    self.position = target;
                }
            }
            jitcode::BC_GOTO_IF_NOT_FLOAT_GT => {
                let a_idx = self.next_u16() as usize;
                let b_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = f64::from_bits(self.registers_f[a_idx] as u64);
                let b = f64::from_bits(self.registers_f[b_idx] as u64);
                if !(a > b) {
                    self.position = target;
                }
            }
            jitcode::BC_GOTO_IF_NOT_FLOAT_GE => {
                let a_idx = self.next_u16() as usize;
                let b_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = f64::from_bits(self.registers_f[a_idx] as u64);
                let b = f64::from_bits(self.registers_f[b_idx] as u64);
                if !(a >= b) {
                    self.position = target;
                }
            }
            // blackhole.py:922-934 bhimpl_goto_if_not_ptr_{eq,ne}.
            jitcode::BC_GOTO_IF_NOT_PTR_EQ => {
                let a_idx = self.next_u16() as usize;
                let b_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = self.registers_r[a_idx];
                let b = self.registers_r[b_idx];
                if a != b {
                    self.position = target;
                }
            }
            jitcode::BC_GOTO_IF_NOT_PTR_NE => {
                let a_idx = self.next_u16() as usize;
                let b_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = self.registers_r[a_idx];
                let b = self.registers_r[b_idx];
                if a == b {
                    self.position = target;
                }
            }
            jitcode::BC_GOTO_IF_NOT_PTR_ISZERO => {
                let a_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = self.registers_r[a_idx];
                if a != 0 {
                    self.position = target;
                }
            }
            jitcode::BC_GOTO_IF_NOT_PTR_NONZERO => {
                let a_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let a = self.registers_r[a_idx];
                if a == 0 {
                    self.position = target;
                }
            }
            jitcode::BC_JUMP => {
                let target = self.next_u16() as usize;
                self.position = target;
            }
            jitcode::BC_JIT_MERGE_POINT | jitcode::BC_JIT_MERGE_POINT_C => {
                self.bhimpl_jit_merge_point(opcode)?;
            }
            jitcode::BC_LOOP_HEADER => {
                // blackhole.py:1062-1064 bhimpl_loop_header(jdindex): no-op.
                // Argument byte is a `registers_i` index pointing at the
                // constants pool slot holding the jitdriver index
                // (upstream `@arguments("i")`).
                let jdindex_reg = self.next_u8() as usize;
                let _jdindex = self.registers_i[jdindex_reg];
            }
            jitcode::BC_REF_RETURN => {
                // RPython bhimpl_ref_return: return ref value to caller.
                let src = self.next_u16() as usize;
                self.tmpreg_r = self.registers_r[src];
                self.return_type = BhReturnType::Ref;
                return Err(DispatchError::LeaveFrame);
            }
            jitcode::BC_INT_RETURN => {
                // RPython bhimpl_int_return (blackhole.py:841): stash the
                // int result in tmpreg_i, tag the frame as int-returning,
                // and unwind via LeaveFrame.
                let src = self.next_u16() as usize;
                self.tmpreg_i = self.registers_i[src];
                self.return_type = BhReturnType::Int;
                return Err(DispatchError::LeaveFrame);
            }
            jitcode::BC_FLOAT_RETURN => {
                // RPython bhimpl_float_return (blackhole.py:853).
                let src = self.next_u16() as usize;
                self.tmpreg_f = self.registers_f[src];
                self.return_type = BhReturnType::Float;
                return Err(DispatchError::LeaveFrame);
            }
            jitcode::BC_VOID_RETURN => {
                // RPython bhimpl_void_return (blackhole.py:859).
                self.return_type = BhReturnType::Void;
                return Err(DispatchError::LeaveFrame);
            }
            jitcode::BC_ABORT => {
                self.aborted = true;
                return Err(DispatchError::LeaveFrame);
            }
            jitcode::BC_ABORT_PERMANENT => {
                self.bhimpl_abort_permanent()?;
            }
            jitcode::BC_UNREACHABLE => {
                // blackhole.py:962 `bhimpl_unreachable()` raises
                // `AssertionError("unreachable")`. A jitcode reaching this
                // opcode is a codegen bug — there is no recovery path.
                panic!("bhimpl_unreachable reached");
            }
            // blackhole.py:1000 bhimpl_raise(excvalue)
            jitcode::BC_RAISE => {
                let src = self.next_u16() as usize;
                let exc = self.registers_r[src];
                if exc != 0 {
                    return Err(DispatchError::RaiseException(exc));
                }
                self.aborted = true;
                return Err(DispatchError::LeaveFrame);
            }
            // blackhole.py:1006 bhimpl_reraise()
            jitcode::BC_RERAISE => {
                let exc = self.exception_last_value;
                if exc != 0 {
                    return Err(DispatchError::RaiseException(exc));
                }
                self.aborted = true;
                return Err(DispatchError::LeaveFrame);
            }
            // BC_INLINE_CALL is wired to `handler_inline_call_pyre_nested`
            // (`blackhole.rs::handler_inline_call_pyre_nested`).  Slice 3.2.1
            // gave the callee `clone_context_from(parent)` so a nested
            // `BC_INLINE_CALL` byte fires the same handler instead of
            // falling through here, retiring the legacy arm.
            // -- Int-typed calls --
            // Parity #14 Slice C.5 retired BC_CALL_PURE_INT (Pure now flows
            // through canonical BC_RESIDUAL_CALL_*_I); only
            // BC_CALL_ASSEMBLER_INT reaches this arm.
            jitcode::BC_CALL_ASSEMBLER_INT => {
                let fn_ptr_idx = self.next_u16() as usize;
                let dst = self.next_u16() as usize;
                let num_args = self.next_u16() as usize;
                let args = self.read_call_args(num_args);
                let target = self.jitcode.call_target(fn_ptr_idx);
                BH_LAST_EXC_VALUE.with(|c| c.set(0));
                let result = call_int_function(target.concrete_ptr, &args);
                // Check if call raised an exception (TLS set by helper).
                let exc_val = BH_LAST_EXC_VALUE.with(|c| c.get());
                if exc_val != 0 {
                    if self.handle_exception_in_frame(exc_val) {
                        return Ok(());
                    }
                    self.exception_last_value = exc_val;
                    self.got_exception = true;
                    return Err(DispatchError::LeaveFrame);
                }
                self.registers_i[dst] = result;
            }
            // -- Ref-typed calls --
            // Parity #14 Slice C.5: see Int arm above.  Only
            // BC_CALL_ASSEMBLER_REF reaches this arm.
            jitcode::BC_CALL_ASSEMBLER_REF => {
                let fn_ptr_idx = self.next_u16() as usize;
                let dst = self.next_u16() as usize;
                let num_args = self.next_u16() as usize;
                let args = self.read_call_args(num_args);
                let target = self.jitcode.call_target(fn_ptr_idx);
                // Clear stale exception before call to prevent false positives.
                BH_LAST_EXC_VALUE.with(|c| c.set(0));
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[bh] call_ref fn={} dst={} nargs={} args={:?} pos={}",
                        fn_ptr_idx,
                        dst,
                        num_args,
                        &args[..args.len().min(4)],
                        self.last_opcode_position,
                    );
                }
                let result = call_int_function(target.concrete_ptr, &args);
                // Check if call raised an exception (TLS set by helper).
                let exc_val = BH_LAST_EXC_VALUE.with(|c| c.get());
                if exc_val != 0 {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[bh] call_ref EXCEPTION fn={} pos={} nargs={} exc={:#x}",
                            fn_ptr_idx, self.last_opcode_position, num_args, exc_val,
                        );
                    }
                    // Actual exception: try handler dispatch.
                    if self.handle_exception_in_frame(exc_val) {
                        return Ok(());
                    }
                    self.exception_last_value = exc_val;
                    self.got_exception = true;
                    return Err(DispatchError::LeaveFrame);
                }
                // result == 0 without exception is a legitimate null ref
                // (e.g., None object has non-zero address in pyre, so this
                // only happens for actual PY_NULL returns which are rare
                // but legal in generic majit jitcode::BC_CALL_REF semantics).
                if crate::majit_log_enabled() {
                    // Heuristic: assume W_IntObject layout (intval at +0x10).
                    let intval_at_10 = if result != 0 {
                        unsafe { Some(*((result + 0x10) as *const i64)) }
                    } else {
                        None
                    };
                    eprintln!(
                        "[bh] call_ref result fn={} pos={} ref=0x{:x} maybe_intval={:?}",
                        fn_ptr_idx, self.last_opcode_position, result, intval_at_10
                    );
                }
                self.registers_r[dst] = result;
            }
            // -- Float-typed calls --
            // PRE-EXISTING-ADAPTATION: blackhole reads `target.concrete_ptr`,
            // which `#[jit_module]` (majit-macros/src/lib.rs:267) emits as
            // `extern "C" fn(...) -> i64` — the helper's f64 result is
            // already pre-packed via `f64::to_bits()` inside the wrapper.
            // We therefore call through `call_int_function` and store the
            // i64 directly into `registers_f` (i64-carrier convention
            // identical to RPython's `longlong.ZEROF` packing).  The f64-ABI
            // wrapper lives at `target.trace_ptr` and is consumed only by
            // the tracing path; using `call_float_function` here would
            // transmute the i64-returning concrete wrapper through an
            // `extern "C" fn(...) -> f64` signature and break the ABI.
            // Parity #14 Slice C.5: see Int arm above.  Only
            // BC_CALL_ASSEMBLER_FLOAT reaches this arm.
            jitcode::BC_CALL_ASSEMBLER_FLOAT => {
                let fn_ptr_idx = self.next_u16() as usize;
                let dst = self.next_u16() as usize;
                let num_args = self.next_u16() as usize;
                let args = self.read_call_args(num_args);
                let target = self.jitcode.call_target(fn_ptr_idx);
                BH_LAST_EXC_VALUE.with(|c| c.set(0));
                let result = call_int_function(target.concrete_ptr, &args);
                let exc_val = BH_LAST_EXC_VALUE.with(|c| c.get());
                if exc_val != 0 {
                    if self.handle_exception_in_frame(exc_val) {
                        return Ok(());
                    }
                    self.exception_last_value = exc_val;
                    self.got_exception = true;
                    return Err(DispatchError::LeaveFrame);
                }
                self.registers_f[dst] = result;
            }
            // -- Void-typed calls --
            // RPython blackhole.py:1218-1222 routes void CALL_* through
            // `cpu.bh_call_v` (no return register at all).  Pyre's
            // `call_void_function` does the same.  Using the int variant
            // here would force an `extern "C" fn(...) -> i64` ABI on a
            // helper that may genuinely return nothing.
            // BC_CALL_ASSEMBLER_VOID retains the legacy
            // `(fn_ptr_idx:u16, num_args:u16, [(kind:u8, reg:u16)]...)`
            // layout — Slice 4 of pyre-call-family-canonical-migration.md
            // owns its migration.
            jitcode::BC_CALL_ASSEMBLER_VOID => {
                let fn_ptr_idx = self.next_u16() as usize;
                let num_args = self.next_u16() as usize;
                let args = self.read_call_args(num_args);
                let target = self.jitcode.call_target(fn_ptr_idx);
                BH_LAST_EXC_VALUE.with(|c| c.set(0));
                call_void_function(target.concrete_ptr, &args);
                let exc_val = BH_LAST_EXC_VALUE.with(|c| c.get());
                if exc_val != 0 {
                    if self.handle_exception_in_frame(exc_val) {
                        return Ok(());
                    }
                    self.exception_last_value = exc_val;
                    self.got_exception = true;
                    return Err(DispatchError::LeaveFrame);
                }
            }
            // Slices 1-2 of `pyre-call-family-canonical-migration.md`:
            // canonical *_v opcodes (=159..=170) live outside the
            // translator-allocated insns range (max byte 74), so
            // dispatch_step_with_fallback always lands here.
            //
            // Decode the canonical byte layout (`blackhole.rs:6459-6481
            // read_list_*` + `blackhole.rs:5853 read_descr` parity), then
            // dispatch through `majit_backend::call_stub::bh_call_i_dispatch`
            // for ABI-correct (separate int/float register-file) call —
            // the same arity table cranelift's `bh_call_i` and dynasm's
            // Slice 0d `bh_call_v` ride on. We cannot delegate to the
            // wired handler bodies because production `BlackholeInterpreter`
            // has `cpu = None` (`builder.cpu` is never set in pyre's
            // resume path — `pyjitpl.py:2247-2253` parity is satisfied
            // via direct dispatch rather than cpu.bh_call_v).
            //
            // Wrap with the `BH_LAST_EXC_VALUE` TLS handshake so any
            // exception pyre-jit's call helpers stash into the TLS
            // (`pyre/pyre-jit/src/call_jit.rs:514, 2976, 3129`)
            // propagates as a frame-level exception.
            //
            // The blackhole side does not distinguish residual_call /
            // may_force / release_gil / loopinvariant by opcode. RPython
            // carries that policy on `calldescr.extra_info`; the bytecode
            // remains the canonical `residual_call_*_v` family.
            jitcode::BC_RESIDUAL_CALL_R_V
            | jitcode::BC_RESIDUAL_CALL_IR_V
            | jitcode::BC_RESIDUAL_CALL_IRF_V => {
                let has_int = matches!(
                    opcode,
                    jitcode::BC_RESIDUAL_CALL_IR_V | jitcode::BC_RESIDUAL_CALL_IRF_V
                );
                let has_float = opcode == jitcode::BC_RESIDUAL_CALL_IRF_V;
                let funcptr_reg = self.next_u8() as usize;
                let func = self.registers_i[funcptr_reg];
                let mut args_i: Vec<i64> = Vec::new();
                let mut args_r: Vec<i64> = Vec::new();
                let mut args_f: Vec<i64> = Vec::new();
                if has_int {
                    let count = self.next_u8() as usize;
                    for _ in 0..count {
                        let reg = self.next_u8() as usize;
                        args_i.push(self.registers_i[reg]);
                    }
                }
                let count_r = self.next_u8() as usize;
                for _ in 0..count_r {
                    let reg = self.next_u8() as usize;
                    args_r.push(self.registers_r[reg]);
                }
                if has_float {
                    let count = self.next_u8() as usize;
                    for _ in 0..count {
                        let reg = self.next_u8() as usize;
                        args_f.push(self.registers_f[reg]);
                    }
                }
                let descr_idx_lo = self.next_u8() as usize;
                let descr_idx_hi = self.next_u8() as usize;
                let descr_idx = descr_idx_lo | (descr_idx_hi << 8);
                // blackhole.py:150-157 + read_descr (`blackhole.rs:5931-5940`)
                // parity: `descrs[idx]` first looks at the per-jitcode
                // runtime pool (`JitCodeBuilder` output), then falls
                // back to the builder-shared `bh.descrs` for
                // build-time / global jitcodes.
                let calldescr = match self.jitcode.exec.descrs.get(descr_idx) {
                    Some(entry) => entry
                        .as_bh_descr()
                        .expect("BC_RESIDUAL_CALL_*_V descr is not BhDescr")
                        .as_calldescr()
                        .clone(),
                    None => self
                        .descrs
                        .get(descr_idx)
                        .unwrap_or_else(|| {
                            panic!(
                                "BC_RESIDUAL_CALL_*_V descr index {descr_idx} \
                                 out of range in jitcode.exec.descrs ({}) \
                                 and bh.descrs ({})",
                                self.jitcode.exec.descrs.len(),
                                self.descrs.len(),
                            )
                        })
                        .as_calldescr()
                        .clone(),
                };
                BH_LAST_EXC_VALUE.with(|c| c.set(0));
                if func != 0 {
                    let (int_args, float_args) = majit_backend::call_stub::collect_call_args(
                        &calldescr.arg_classes,
                        Some(&args_i),
                        Some(&args_r),
                        Some(&args_f),
                    );
                    // llmodel.py:834 bh_call_v: void-returning callees
                    // require the void-typed dispatcher; reading rax/x0
                    // through `bh_call_i_dispatch` for a genuinely void
                    // helper is undefined behavior.
                    unsafe {
                        majit_backend::call_stub::bh_call_v_dispatch(
                            func as usize,
                            &int_args,
                            &float_args,
                        );
                    }
                }
                let exc_val = BH_LAST_EXC_VALUE.with(|c| c.get());
                if exc_val != 0 {
                    if self.handle_exception_in_frame(exc_val) {
                        return Ok(());
                    }
                    self.exception_last_value = exc_val;
                    self.got_exception = true;
                    return Err(DispatchError::LeaveFrame);
                }
            }
            // ── Slice 4 Slice 1c.1: canonical typed (i/r/f) blackhole arms ──
            //
            // Mirror of the void arm above for the int / ref / float
            // result kinds.  Identical bytecode decode + BH_LAST_EXC_VALUE
            // handshake; the only differences are:
            //   * dispatcher: `bh_call_v_dispatch` →
            //     `bh_call_i_dispatch` (int / ref) or
            //     `bh_call_f_dispatch` (float)
            //   * trailing `dst` register byte after the descr index
            //     (`emit_canonical_call_typed` /
            //     `emit_canonical_call_typed_irf_f` at
            //     `jitcode/assembler.rs:1932 / 2100` always emit it)
            //   * register write-back: `registers_i[dst]` /
            //     `registers_r[dst]` / `registers_f[dst]`
            //   * float arm reads all three (count, regs) pairs
            //     unconditionally — only `IRF_F` exists per
            //     `resoperation.py:1238-1248`
            //
            // The blackhole side does not distinguish residual_call /
            // may_force / release_gil / loopinvariant by opcode (matches
            // the void arm comment at line 2566); the policy lives on
            // `calldescr.extra_info`.
            jitcode::BC_RESIDUAL_CALL_R_I
            | jitcode::BC_RESIDUAL_CALL_IR_I
            | jitcode::BC_RESIDUAL_CALL_IRF_I => {
                let has_int = matches!(
                    opcode,
                    jitcode::BC_RESIDUAL_CALL_IR_I | jitcode::BC_RESIDUAL_CALL_IRF_I
                );
                let has_float = opcode == jitcode::BC_RESIDUAL_CALL_IRF_I;
                let funcptr_reg = self.next_u8() as usize;
                let func = self.registers_i[funcptr_reg];
                let mut args_i: Vec<i64> = Vec::new();
                let mut args_r: Vec<i64> = Vec::new();
                let mut args_f: Vec<i64> = Vec::new();
                if has_int {
                    let count = self.next_u8() as usize;
                    for _ in 0..count {
                        let reg = self.next_u8() as usize;
                        args_i.push(self.registers_i[reg]);
                    }
                }
                let count_r = self.next_u8() as usize;
                for _ in 0..count_r {
                    let reg = self.next_u8() as usize;
                    args_r.push(self.registers_r[reg]);
                }
                if has_float {
                    let count = self.next_u8() as usize;
                    for _ in 0..count {
                        let reg = self.next_u8() as usize;
                        args_f.push(self.registers_f[reg]);
                    }
                }
                let descr_idx_lo = self.next_u8() as usize;
                let descr_idx_hi = self.next_u8() as usize;
                let descr_idx = descr_idx_lo | (descr_idx_hi << 8);
                let dst = self.next_u8() as usize;
                let calldescr = match self.jitcode.exec.descrs.get(descr_idx) {
                    Some(entry) => entry
                        .as_bh_descr()
                        .expect("BC_RESIDUAL_CALL_*_I descr is not BhDescr")
                        .as_calldescr()
                        .clone(),
                    None => self
                        .descrs
                        .get(descr_idx)
                        .unwrap_or_else(|| {
                            panic!(
                                "BC_RESIDUAL_CALL_*_I descr index {descr_idx} \
                                 out of range in jitcode.exec.descrs ({}) \
                                 and bh.descrs ({})",
                                self.jitcode.exec.descrs.len(),
                                self.descrs.len(),
                            )
                        })
                        .as_calldescr()
                        .clone(),
                };
                BH_LAST_EXC_VALUE.with(|c| c.set(0));
                let result = if func != 0 {
                    let (int_args, float_args) = majit_backend::call_stub::collect_call_args(
                        &calldescr.arg_classes,
                        Some(&args_i),
                        Some(&args_r),
                        Some(&args_f),
                    );
                    unsafe {
                        majit_backend::call_stub::bh_call_i_dispatch(
                            func as usize,
                            &int_args,
                            &float_args,
                        )
                    }
                } else {
                    0
                };
                // blackhole.py:351-361 run() — exceptions raised inside
                // bhimpl propagate via the host runtime's try/except in
                // run(), so the bhimpl returns early and never reaches
                // the destination-register store.  Pyre transports the
                // exception via BH_LAST_EXC_VALUE TLS and does the same:
                // only write the result when no exception is pending.
                let exc_val = BH_LAST_EXC_VALUE.with(|c| c.get());
                if exc_val != 0 {
                    if self.handle_exception_in_frame(exc_val) {
                        return Ok(());
                    }
                    self.exception_last_value = exc_val;
                    self.got_exception = true;
                    return Err(DispatchError::LeaveFrame);
                }
                self.registers_i[dst] = result;
            }
            jitcode::BC_RESIDUAL_CALL_R_R
            | jitcode::BC_RESIDUAL_CALL_IR_R
            | jitcode::BC_RESIDUAL_CALL_IRF_R => {
                let has_int = matches!(
                    opcode,
                    jitcode::BC_RESIDUAL_CALL_IR_R | jitcode::BC_RESIDUAL_CALL_IRF_R
                );
                let has_float = opcode == jitcode::BC_RESIDUAL_CALL_IRF_R;
                let funcptr_reg = self.next_u8() as usize;
                let func = self.registers_i[funcptr_reg];
                let mut args_i: Vec<i64> = Vec::new();
                let mut args_r: Vec<i64> = Vec::new();
                let mut args_f: Vec<i64> = Vec::new();
                if has_int {
                    let count = self.next_u8() as usize;
                    for _ in 0..count {
                        let reg = self.next_u8() as usize;
                        args_i.push(self.registers_i[reg]);
                    }
                }
                let count_r = self.next_u8() as usize;
                for _ in 0..count_r {
                    let reg = self.next_u8() as usize;
                    args_r.push(self.registers_r[reg]);
                }
                if has_float {
                    let count = self.next_u8() as usize;
                    for _ in 0..count {
                        let reg = self.next_u8() as usize;
                        args_f.push(self.registers_f[reg]);
                    }
                }
                let descr_idx_lo = self.next_u8() as usize;
                let descr_idx_hi = self.next_u8() as usize;
                let descr_idx = descr_idx_lo | (descr_idx_hi << 8);
                let dst = self.next_u8() as usize;
                let calldescr = match self.jitcode.exec.descrs.get(descr_idx) {
                    Some(entry) => entry
                        .as_bh_descr()
                        .expect("BC_RESIDUAL_CALL_*_R descr is not BhDescr")
                        .as_calldescr()
                        .clone(),
                    None => self
                        .descrs
                        .get(descr_idx)
                        .unwrap_or_else(|| {
                            panic!(
                                "BC_RESIDUAL_CALL_*_R descr index {descr_idx} \
                                 out of range in jitcode.exec.descrs ({}) \
                                 and bh.descrs ({})",
                                self.jitcode.exec.descrs.len(),
                                self.descrs.len(),
                            )
                        })
                        .as_calldescr()
                        .clone(),
                };
                BH_LAST_EXC_VALUE.with(|c| c.set(0));
                // Ref-result returns as i64 (PyObjectRef pointer value).
                let result = if func != 0 {
                    let (int_args, float_args) = majit_backend::call_stub::collect_call_args(
                        &calldescr.arg_classes,
                        Some(&args_i),
                        Some(&args_r),
                        Some(&args_f),
                    );
                    unsafe {
                        majit_backend::call_stub::bh_call_i_dispatch(
                            func as usize,
                            &int_args,
                            &float_args,
                        )
                    }
                } else {
                    0
                };
                // blackhole.py:351-361 run() — see int sibling above for
                // the full citation.  Exception path skips the register
                // write so a raising bhimpl does not clobber `dst`.
                let exc_val = BH_LAST_EXC_VALUE.with(|c| c.get());
                if exc_val != 0 {
                    if self.handle_exception_in_frame(exc_val) {
                        return Ok(());
                    }
                    self.exception_last_value = exc_val;
                    self.got_exception = true;
                    return Err(DispatchError::LeaveFrame);
                }
                self.registers_r[dst] = result;
            }
            // Float result: only `IRF_F` exists per
            // `resoperation.py:1238-1248`.  Reads all three
            // (count, regs) pairs unconditionally —
            // `emit_canonical_call_typed_irf_f`
            // (`jitcode/assembler.rs:2100`) always emits them.
            jitcode::BC_RESIDUAL_CALL_IRF_F => {
                let funcptr_reg = self.next_u8() as usize;
                let func = self.registers_i[funcptr_reg];
                let mut args_i: Vec<i64> = Vec::new();
                let mut args_r: Vec<i64> = Vec::new();
                let mut args_f: Vec<i64> = Vec::new();
                let count_i = self.next_u8() as usize;
                for _ in 0..count_i {
                    let reg = self.next_u8() as usize;
                    args_i.push(self.registers_i[reg]);
                }
                let count_r = self.next_u8() as usize;
                for _ in 0..count_r {
                    let reg = self.next_u8() as usize;
                    args_r.push(self.registers_r[reg]);
                }
                let count_f = self.next_u8() as usize;
                for _ in 0..count_f {
                    let reg = self.next_u8() as usize;
                    args_f.push(self.registers_f[reg]);
                }
                let descr_idx_lo = self.next_u8() as usize;
                let descr_idx_hi = self.next_u8() as usize;
                let descr_idx = descr_idx_lo | (descr_idx_hi << 8);
                let dst = self.next_u8() as usize;
                let calldescr = match self.jitcode.exec.descrs.get(descr_idx) {
                    Some(entry) => entry
                        .as_bh_descr()
                        .expect("BC_RESIDUAL_CALL_IRF_F descr is not BhDescr")
                        .as_calldescr()
                        .clone(),
                    None => self
                        .descrs
                        .get(descr_idx)
                        .unwrap_or_else(|| {
                            panic!(
                                "BC_RESIDUAL_CALL_IRF_F descr index {descr_idx} \
                                 out of range in jitcode.exec.descrs ({}) \
                                 and bh.descrs ({})",
                                self.jitcode.exec.descrs.len(),
                                self.descrs.len(),
                            )
                        })
                        .as_calldescr()
                        .clone(),
                };
                BH_LAST_EXC_VALUE.with(|c| c.set(0));
                let result_f = if func != 0 {
                    let (int_args, float_args) = majit_backend::call_stub::collect_call_args(
                        &calldescr.arg_classes,
                        Some(&args_i),
                        Some(&args_r),
                        Some(&args_f),
                    );
                    unsafe {
                        majit_backend::call_stub::bh_call_f_dispatch(
                            func as usize,
                            &int_args,
                            &float_args,
                        )
                    }
                } else {
                    0.0f64
                };
                // blackhole.py:351-361 run() — see int sibling above for
                // the full citation.  Exception path skips the register
                // write so a raising bhimpl does not clobber `dst`.
                let exc_val = BH_LAST_EXC_VALUE.with(|c| c.get());
                if exc_val != 0 {
                    if self.handle_exception_in_frame(exc_val) {
                        return Ok(());
                    }
                    self.exception_last_value = exc_val;
                    self.got_exception = true;
                    return Err(DispatchError::LeaveFrame);
                }
                self.registers_f[dst] = result_f.to_bits() as i64;
            }
            // -- State field access --
            // Argcodes: `d` = u16 descr (`assembler.py:197-207`),
            // `i` = u8 register index (`assembler.py:165-167`).
            jitcode::BC_LOAD_STATE_FIELD => {
                // /di : u16 descr + u8 dst = 3 bytes
                let _field_idx = self.next_u16();
                let _dst = self.next_u8();
                // No-op in blackhole: state fields are only meaningful
                // during tracing with a JitCodeSym.
            }
            jitcode::BC_STORE_STATE_FIELD => {
                let _field_idx = self.next_u16();
                let _src = self.next_u8();
            }
            jitcode::BC_LOAD_STATE_ARRAY | jitcode::BC_LOAD_STATE_VARRAY => {
                // /dii : u16 descr + u8 index + u8 dst = 4 bytes
                let _array_idx = self.next_u16();
                let _elem_idx = self.next_u8();
                let _dst = self.next_u8();
            }
            jitcode::BC_STORE_STATE_ARRAY | jitcode::BC_STORE_STATE_VARRAY => {
                let _array_idx = self.next_u16();
                let _elem_idx = self.next_u8();
                let _src = self.next_u8();
            }
            // -- Virtualizable field/array access --
            // blackhole.py:1446-1458 bhimpl_getfield_vable_i/r/f:
            //   fielddescr.get_vinfo().clear_vable_token(struct)
            //   return cpu.bh_getfield_gc_i/r/f(struct, fielddescr)
            jitcode::BC_GETFIELD_VABLE_I => {
                let field_idx = self.vable_field_index_at(self.position + 1);
                let base = self.next_u8() as usize;
                self.position += 2;
                let dst = self.next_u8() as usize;
                let ptr = self.registers_r[base] as *mut u8;
                let vinfo = unsafe { &*self.virtualizable_info };
                // virtualizable.py:218-222 clear_vable_token
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let offset = vinfo.static_fields[field_idx].offset;
                let value = unsafe { *(ptr.add(offset) as *const i64) };
                self.registers_i[dst] = value;
            }
            jitcode::BC_GETFIELD_VABLE_R => {
                let field_idx = self.vable_field_index_at(self.position + 1);
                let base = self.next_u8() as usize;
                self.position += 2;
                let dst = self.next_u8() as usize;
                let ptr = self.registers_r[base] as *mut u8;
                let vinfo = unsafe { &*self.virtualizable_info };
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let offset = vinfo.static_fields[field_idx].offset;
                let value = unsafe { *(ptr.add(offset) as *const i64) };
                self.registers_r[dst] = value;
            }
            jitcode::BC_GETFIELD_VABLE_F => {
                let field_idx = self.vable_field_index_at(self.position + 1);
                let base = self.next_u8() as usize;
                self.position += 2;
                let dst = self.next_u8() as usize;
                let ptr = self.registers_r[base] as *mut u8;
                let vinfo = unsafe { &*self.virtualizable_info };
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let offset = vinfo.static_fields[field_idx].offset;
                let value = unsafe { *(ptr.add(offset) as *const i64) };
                self.registers_f[dst] = value;
            }
            // blackhole.py:1485-1495 bhimpl_setfield_vable_i/r/f
            jitcode::BC_SETFIELD_VABLE_I => {
                let field_idx = self.vable_field_index_at(self.position + 2);
                let base = self.next_u8() as usize;
                let src = self.next_u8() as usize;
                self.position += 2;
                let ptr = self.registers_r[base] as *mut u8;
                let vinfo = unsafe { &*self.virtualizable_info };
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let offset = vinfo.static_fields[field_idx].offset;
                let value = self.registers_i[src];
                unsafe { *(ptr.add(offset) as *mut i64) = value };
            }
            jitcode::BC_SETFIELD_VABLE_R => {
                let field_idx = self.vable_field_index_at(self.position + 2);
                let base = self.next_u8() as usize;
                let src = self.next_u8() as usize;
                self.position += 2;
                let ptr = self.registers_r[base] as *mut u8;
                let vinfo = unsafe { &*self.virtualizable_info };
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let offset = vinfo.static_fields[field_idx].offset;
                let value = self.registers_r[src];
                unsafe { *(ptr.add(offset) as *mut i64) = value };
            }
            jitcode::BC_SETFIELD_VABLE_F => {
                let field_idx = self.vable_field_index_at(self.position + 2);
                let base = self.next_u8() as usize;
                let src = self.next_u8() as usize;
                self.position += 2;
                let ptr = self.registers_r[base] as *mut u8;
                let vinfo = unsafe { &*self.virtualizable_info };
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let offset = vinfo.static_fields[field_idx].offset;
                let value = self.registers_f[src];
                unsafe { *(ptr.add(offset) as *mut i64) = value };
            }
            // blackhole.py:1374-1387 bhimpl_getarrayitem_vable_i/r/f:
            //   fielddescr.get_vinfo().clear_vable_token(vable)
            //   array = cpu.bh_getfield_gc_r(vable, fielddescr)
            //   return cpu.bh_getarrayitem_gc_i/r/f(array, index, arraydescr)
            jitcode::BC_GETARRAYITEM_VABLE_I
            | jitcode::BC_GETARRAYITEM_VABLE_R
            | jitcode::BC_GETARRAYITEM_VABLE_F => {
                let nbody_debug = std::env::var_os("PYRE_NBODY_DEBUG").is_some();
                let array_idx =
                    self.vable_array_index_pair_at(self.position + 2, self.position + 4);
                let base = self.next_u8() as usize;
                let index_reg = self.next_u8() as usize;
                self.position += 4;
                let dst = self.next_u8() as usize;
                let ptr = self.registers_r[base] as *mut u8;
                let vinfo = unsafe { &*self.virtualizable_info };
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let ainfo = &vinfo.array_fields[array_idx];
                let index = self.registers_i[index_reg] as usize;
                let value = unsafe {
                    crate::virtualizable::vable_read_array_item(ptr as *const u8, ainfo, index)
                };
                if nbody_debug && array_idx == 0 && matches!(index, 5 | 6 | 8 | 9) {
                    eprintln!(
                        "[nbody-debug][bh-vable-get-inline] position={} opcode={} index={} value={:#x}",
                        self.position, opcode, index, value as usize
                    );
                }
                match opcode {
                    jitcode::BC_GETARRAYITEM_VABLE_I => self.registers_i[dst] = value,
                    jitcode::BC_GETARRAYITEM_VABLE_R => self.registers_r[dst] = value,
                    _ => self.registers_f[dst] = value,
                }
            }
            // blackhole.py:1390-1403 bhimpl_setarrayitem_vable_i/r/f
            jitcode::BC_SETARRAYITEM_VABLE_I
            | jitcode::BC_SETARRAYITEM_VABLE_R
            | jitcode::BC_SETARRAYITEM_VABLE_F => {
                let nbody_debug = std::env::var_os("PYRE_NBODY_DEBUG").is_some();
                let array_idx =
                    self.vable_array_index_pair_at(self.position + 3, self.position + 5);
                let base = self.next_u8() as usize;
                let index_reg = self.next_u8() as usize;
                let src = self.next_u8() as usize;
                self.position += 4;
                let ptr = self.registers_r[base] as *mut u8;
                let vinfo = unsafe { &*self.virtualizable_info };
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let ainfo = &vinfo.array_fields[array_idx];
                let index = self.registers_i[index_reg] as usize;
                let value = match opcode {
                    jitcode::BC_SETARRAYITEM_VABLE_I => self.registers_i[src],
                    jitcode::BC_SETARRAYITEM_VABLE_R => self.registers_r[src],
                    _ => self.registers_f[src],
                };
                if nbody_debug && array_idx == 0 && matches!(index, 5 | 6 | 8 | 9) {
                    eprintln!(
                        "[nbody-debug][bh-vable-set-inline] position={} opcode={} index={} value={:#x}",
                        self.position, opcode, index, value as usize
                    );
                }
                unsafe {
                    crate::virtualizable::vable_write_array_item(ptr, ainfo, index, value);
                }
            }
            // blackhole.py:1406-1409 bhimpl_arraylen_vable
            jitcode::BC_ARRAYLEN_VABLE => {
                let array_idx =
                    self.vable_array_index_pair_at(self.position + 1, self.position + 3);
                let base = self.next_u8() as usize;
                self.position += 4;
                let dst = self.next_u8() as usize;
                let ptr = self.registers_r[base] as *mut u8;
                let vinfo = unsafe { &*self.virtualizable_info };
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let ainfo = &vinfo.array_fields[array_idx];
                let len =
                    unsafe { crate::virtualizable::bhimpl_arraylen_vable(ptr as *const u8, ainfo) };
                self.registers_i[dst] = len as i64;
            }
            jitcode::BC_HINT_FORCE_VIRTUALIZABLE => {
                let _vable_reg = self.next_u8();
                // No-op in blackhole
            }
            other => {
                panic!("blackhole: unknown jitcode bytecode {other}");
            }
        }
        Ok(())
    }

    /// Read call arguments from bytecode (kind:u8, reg:u16 per arg).
    fn read_call_args(&mut self, num_args: usize) -> Vec<i64> {
        let mut args = Vec::with_capacity(num_args);
        let mut metas: Vec<(u8, u16)> = Vec::with_capacity(num_args);
        for _ in 0..num_args {
            let kind_byte = self.next_u8();
            let kind = JitArgKind::decode(kind_byte);
            let reg = self.next_u16();
            metas.push((kind_byte, reg));
            args.push(self.read_call_arg(kind, reg));
        }
        if crate::majit_log_enabled() {
            // Heuristic: for Ref args, peek intval at +0x10 (W_IntObject layout).
            let mut intvals: Vec<Option<i64>> = Vec::with_capacity(args.len());
            for ((kind_byte, _reg), val) in metas.iter().zip(args.iter()) {
                if *kind_byte == 1 && *val != 0 {
                    intvals.push(Some(unsafe { *((*val as usize + 0x10) as *const i64) }));
                } else {
                    intvals.push(None);
                }
            }
            eprintln!(
                "[bh] read_call_args pos={} num={} kinds_regs={:?} vals={:?} maybe_intvals={:?}",
                self.last_opcode_position, num_args, metas, args, intvals
            );
        }
        args
    }
}

// ════════════════════════════════════════════════════════════════════════
// bhimpl_*_call_* family (blackhole.py:1095-1320)
// ════════════════════════════════════════════════════════════════════════
//
// These methods mirror RPython's blackhole call dispatch table. Each
// variant unpacks one of the three calling-convention shapes
// (`r` / `ir` / `irf`) into a `Backend::bh_call_{i,r,f,v}` invocation
// and returns the typed result.  The `recursive_call` family adds a
// jitdriver-index lookup via `get_portal_runner`.
//
// **Wired into the codewriter-orthodox dispatch table.**  The
// `handler_residual_call_*` / `handler_inline_call_*` /
// `handler_recursive_call_*` functions registered by
// `wire_bhimpl_handlers` (`BlackholeInterpBuilder::wire_handler` →
// `dispatch_table`) decode the bytecode (func/jitcode + ListOfKind
// args + descr) and then call the corresponding `bhimpl_*_call_*`
// method on the BlackholeInterpreter, which in turn routes through
// `cpu().bh_call_{i,r,f,v}` exactly like RPython.  The legacy inline
// `jitcode::BC_RESIDUAL_CALL_*` / `jitcode::BC_CALL_MAY_FORCE_*` branches in
// `JitCodeMachine::dispatch_one` are pyre's IR-side fast path for
// JIT-compiled traces and operate on a different bytecode encoding
// (raw fn pointers, no calldescr); the codewriter-orthodox path runs
// through these handlers + bhimpl methods.

impl BlackholeInterpreter {
    fn cpu(&self) -> &'static dyn majit_backend::Backend {
        self.cpu.expect("blackhole cpu reference unset")
    }

    // ── bhimpl_residual_call_* (blackhole.py:1224-1255) ──

    /// blackhole.py:1225-1226
    pub fn bhimpl_residual_call_r_i(
        &self,
        func: i64,
        args_r: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> i64 {
        self.cpu()
            .bh_call_i(func, None, Some(args_r), None, calldescr)
    }

    /// blackhole.py:1227-1229
    pub fn bhimpl_residual_call_r_r(
        &self,
        func: i64,
        args_r: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        self.cpu()
            .bh_call_r(func, None, Some(args_r), None, calldescr)
    }

    /// blackhole.py:1230-1232
    pub fn bhimpl_residual_call_r_v(
        &self,
        func: i64,
        args_r: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) {
        self.cpu()
            .bh_call_v(func, None, Some(args_r), None, calldescr);
    }

    /// blackhole.py:1234-1236
    pub fn bhimpl_residual_call_ir_i(
        &self,
        func: i64,
        args_i: &[i64],
        args_r: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> i64 {
        self.cpu()
            .bh_call_i(func, Some(args_i), Some(args_r), None, calldescr)
    }

    /// blackhole.py:1237-1239
    pub fn bhimpl_residual_call_ir_r(
        &self,
        func: i64,
        args_i: &[i64],
        args_r: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        self.cpu()
            .bh_call_r(func, Some(args_i), Some(args_r), None, calldescr)
    }

    /// blackhole.py:1240-1242
    pub fn bhimpl_residual_call_ir_v(
        &self,
        func: i64,
        args_i: &[i64],
        args_r: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) {
        self.cpu()
            .bh_call_v(func, Some(args_i), Some(args_r), None, calldescr);
    }

    /// blackhole.py:1244-1246
    pub fn bhimpl_residual_call_irf_i(
        &self,
        func: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> i64 {
        self.cpu()
            .bh_call_i(func, Some(args_i), Some(args_r), Some(args_f), calldescr)
    }

    /// blackhole.py:1247-1249
    pub fn bhimpl_residual_call_irf_r(
        &self,
        func: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        self.cpu()
            .bh_call_r(func, Some(args_i), Some(args_r), Some(args_f), calldescr)
    }

    /// blackhole.py:1250-1252
    pub fn bhimpl_residual_call_irf_f(
        &self,
        func: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> f64 {
        self.cpu()
            .bh_call_f(func, Some(args_i), Some(args_r), Some(args_f), calldescr)
    }

    /// blackhole.py:1253-1255
    pub fn bhimpl_residual_call_irf_v(
        &self,
        func: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) {
        self.cpu()
            .bh_call_v(func, Some(args_i), Some(args_r), Some(args_f), calldescr);
    }

    // ── bhimpl_inline_call_* (blackhole.py:1278-1319) ──
    //
    // RPython unpacks `jitcode.fnaddr` and `jitcode.calldescr` from the
    // jitcode parameter; pyre passes them directly so the pyre tracer
    // can reuse the helpers without a JitCode object.

    /// blackhole.py:1279-1281
    pub fn bhimpl_inline_call_r_i(
        &self,
        fnaddr: i64,
        args_r: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> i64 {
        self.bhimpl_residual_call_r_i(fnaddr, args_r, calldescr)
    }
    /// blackhole.py:1282-1285
    pub fn bhimpl_inline_call_r_r(
        &self,
        fnaddr: i64,
        args_r: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        self.bhimpl_residual_call_r_r(fnaddr, args_r, calldescr)
    }
    /// blackhole.py:1286-1289
    pub fn bhimpl_inline_call_r_v(
        &self,
        fnaddr: i64,
        args_r: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) {
        self.bhimpl_residual_call_r_v(fnaddr, args_r, calldescr);
    }
    /// blackhole.py:1291-1294
    pub fn bhimpl_inline_call_ir_i(
        &self,
        fnaddr: i64,
        args_i: &[i64],
        args_r: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> i64 {
        self.bhimpl_residual_call_ir_i(fnaddr, args_i, args_r, calldescr)
    }
    /// blackhole.py:1295-1298
    pub fn bhimpl_inline_call_ir_r(
        &self,
        fnaddr: i64,
        args_i: &[i64],
        args_r: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        self.bhimpl_residual_call_ir_r(fnaddr, args_i, args_r, calldescr)
    }
    /// blackhole.py:1299-1302
    pub fn bhimpl_inline_call_ir_v(
        &self,
        fnaddr: i64,
        args_i: &[i64],
        args_r: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) {
        self.bhimpl_residual_call_ir_v(fnaddr, args_i, args_r, calldescr);
    }
    /// blackhole.py:1304-1307
    pub fn bhimpl_inline_call_irf_i(
        &self,
        fnaddr: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> i64 {
        self.bhimpl_residual_call_irf_i(fnaddr, args_i, args_r, args_f, calldescr)
    }
    /// blackhole.py:1308-1311
    pub fn bhimpl_inline_call_irf_r(
        &self,
        fnaddr: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        self.bhimpl_residual_call_irf_r(fnaddr, args_i, args_r, args_f, calldescr)
    }
    /// blackhole.py:1312-1315
    pub fn bhimpl_inline_call_irf_f(
        &self,
        fnaddr: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> f64 {
        self.bhimpl_residual_call_irf_f(fnaddr, args_i, args_r, args_f, calldescr)
    }
    /// blackhole.py:1316-1319
    pub fn bhimpl_inline_call_irf_v(
        &self,
        fnaddr: i64,
        args_i: &[i64],
        args_r: &[i64],
        args_f: &[i64],
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) {
        self.bhimpl_residual_call_irf_v(fnaddr, args_i, args_r, args_f, calldescr);
    }

    // ── bhimpl_recursive_call_{i,f,v} (blackhole.py:1102-1132) ──
    //
    // The `_r` variant already lives further up in this file (line ~1165),
    // pre-existing pyre code that uses `jdindex` + `get_portal_runner`.
    // The remaining three follow the same pattern for parity.

    /// blackhole.py:1102-1108
    pub fn bhimpl_recursive_call_i(
        &self,
        jdindex: usize,
        greens_i: Vec<i64>,
        greens_r: Vec<i64>,
        greens_f: Vec<i64>,
        reds_i: Vec<i64>,
        reds_r: Vec<i64>,
        reds_f: Vec<i64>,
    ) -> i64 {
        let (fnptr, calldescr) = self.get_portal_runner(jdindex);
        let mut all_i = greens_i;
        all_i.extend(&reds_i);
        let mut all_r = greens_r;
        all_r.extend(&reds_r);
        let mut all_f = greens_f;
        all_f.extend(&reds_f);
        self.cpu()
            .bh_call_i(fnptr, Some(&all_i), Some(&all_r), Some(&all_f), &calldescr)
    }

    /// blackhole.py:1117-1124
    pub fn bhimpl_recursive_call_f(
        &self,
        jdindex: usize,
        greens_i: Vec<i64>,
        greens_r: Vec<i64>,
        greens_f: Vec<i64>,
        reds_i: Vec<i64>,
        reds_r: Vec<i64>,
        reds_f: Vec<i64>,
    ) -> f64 {
        let (fnptr, calldescr) = self.get_portal_runner(jdindex);
        let mut all_i = greens_i;
        all_i.extend(&reds_i);
        let mut all_r = greens_r;
        all_r.extend(&reds_r);
        let mut all_f = greens_f;
        all_f.extend(&reds_f);
        self.cpu()
            .bh_call_f(fnptr, Some(&all_i), Some(&all_r), Some(&all_f), &calldescr)
    }

    /// blackhole.py:1125-1132
    pub fn bhimpl_recursive_call_v(
        &self,
        jdindex: usize,
        greens_i: Vec<i64>,
        greens_r: Vec<i64>,
        greens_f: Vec<i64>,
        reds_i: Vec<i64>,
        reds_r: Vec<i64>,
        reds_f: Vec<i64>,
    ) {
        let (fnptr, calldescr) = self.get_portal_runner(jdindex);
        let mut all_i = greens_i;
        all_i.extend(&reds_i);
        let mut all_r = greens_r;
        all_r.extend(&reds_r);
        let mut all_f = greens_f;
        all_f.extend(&reds_f);
        self.cpu()
            .bh_call_v(fnptr, Some(&all_i), Some(&all_r), Some(&all_f), &calldescr);
    }
}

/// Pool manager + dispatch builder for blackhole interpreters.
///
/// RPython `blackhole.py:52-103` `class BlackholeInterpBuilder`.
///
/// Combines two responsibilities:
/// 1. Interpreter pool management (acquire/release/release_chain).
/// 2. Codewriter-orthodox dispatch setup (`setup_insns` → dispatch table
///    + `dispatch_loop`). Phase D incrementally wires this up as
///    `bhimpl_*` methods are ported from RPython.
pub struct BlackholeInterpBuilder {
    pool: Vec<BlackholeInterpreter>,
    /// RPython `blackhole.py:56` `self.cpu = codewriter.cpu`.
    /// Stored as raw pointer; the Backend outlives the builder.
    /// RPython `blackhole.py:286/56` `self.cpu = builder.cpu`.
    /// Backend trait for `bh_*` concrete execution. None until set.
    pub cpu: Option<&'static dyn majit_backend::Backend>,
    /// RPython `blackhole.py:68` `self._insns`: opcode byte → "opname/argcodes".
    /// Populated by `setup_insns`; empty until called.
    pub _insns: Vec<String>,
    /// RPython `blackhole.py:72` `self.op_live = insns.get('live/', -1)`.
    pub op_live: u8,
    /// RPython `blackhole.py:73` `self.op_catch_exception`.
    pub op_catch_exception: u8,
    /// RPython `blackhole.py:74` `self.op_rvmprof_code`.
    pub op_rvmprof_code: u8,
    /// RPython `blackhole.py:103` `self.descrs`.
    /// Populated by `setup_descrs()` from the assembler's descriptor table.
    pub descrs: Vec<BhDescr>,
    /// Dispatch table: opcode byte → handler fn pointer.
    /// RPython builds `dispatch_loop` closure via `unrolling_iterable`;
    /// Rust uses indirect call through this table.
    ///
    /// `Arc<Vec<...>>` so each `acquire_interp` cheaply shares the same
    /// table snapshot with the returned interpreter (RPython
    /// `blackhole.py:287` `self.dispatch_loop = builder.dispatch_loop`
    /// instance binding).  Wiring mutators (`setup_insns`,
    /// `wire_handler`) take `&mut self` and use `Arc::make_mut`; once
    /// wiring is done before the first `acquire_interp`, the inner Vec
    /// is uniquely owned and `make_mut` is cheap.
    pub(crate) dispatch_table: std::sync::Arc<Vec<BhOpcodeHandler>>,
}

impl Default for BlackholeInterpBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BlackholeInterpBuilder {
    pub fn new() -> Self {
        Self {
            pool: Vec::new(),
            cpu: None,
            _insns: Vec::new(),
            op_live: u8::MAX,
            op_catch_exception: u8::MAX,
            op_rvmprof_code: u8::MAX,
            descrs: Vec::new(),
            dispatch_table: std::sync::Arc::new(Vec::new()),
        }
    }

    /// PRE-EXISTING-ADAPTATION narrowed for parity:
    /// some Rust call sites construct a fresh builder away from the
    /// codewriter, but they already hold the cached opcode ids that
    /// RPython `setup_insns(insns)` would have stored on the builder.
    ///
    /// Copy those three cached control opcodes directly instead of
    /// consulting a synthetic global `wellknown_bh_insns()` table.
    /// Callers with a real assembler `insns` dict should still use
    /// `setup_insns`, which also fills `_insns` and `dispatch_table`
    /// exactly like `blackhole.py:66-100`.
    pub fn setup_cached_control_opcodes(
        &mut self,
        op_live: i32,
        op_catch_exception: i32,
        op_rvmprof_code: i32,
    ) {
        let to_u8 = |opcode: i32| -> u8 {
            if opcode < 0 {
                u8::MAX
            } else {
                u8::try_from(opcode).expect("cached blackhole opcode does not fit in u8")
            }
        };
        self.op_live = to_u8(op_live);
        self.op_catch_exception = to_u8(op_catch_exception);
        self.op_rvmprof_code = to_u8(op_rvmprof_code);
    }

    /// RPython `blackhole.py:66-100` `setup_insns(insns)`.
    ///
    /// ```python
    /// def setup_insns(self, insns):
    ///     assert len(insns) <= 256, "too many instructions!"
    ///     self._insns = [None] * len(insns)
    ///     for key, value in insns.items():
    ///         assert self._insns[value] is None
    ///         self._insns[value] = key
    ///     self.op_live = insns.get('live/', -1)
    ///     self.op_catch_exception = insns.get('catch_exception/L', -1)
    ///     self.op_rvmprof_code = insns.get('rvmprof_code/ii', -1)
    /// ```
    ///
    /// Builds the reverse opcode table and dispatch function table from the
    /// assembler's `insns` dict. For now, all dispatch table entries are
    /// placeholder handlers — real `bhimpl_*` methods are wired in
    /// incrementally as Phase D progresses.
    pub fn setup_insns(&mut self, insns: &std::collections::HashMap<String, u8>) {
        assert!(insns.len() <= 256, "too many instructions!");
        // RPython blackhole.py:68-71: build reverse table.
        //
        // PRE-EXISTING-ADAPTATION: RPython sizes `_insns` by `len(insns)`
        // because every opname is dynamically numbered `0..len-1`
        // (`Assembler.insns.setdefault(key, len(self.insns))`), so the
        // length and the maximum byte coincide.  Pyre's canonical-routing
        // (`majit-translate::insns::insn_byte_opt`) pins canonical keys
        // to fixed `BC_*` bytes (sparse, up to 168) and pushes
        // translator-only keys past `CANONICAL_BYTE_CEILING`, so the
        // byte space is sparse and `len(insns) < max_byte + 1`.  Size
        // the reverse table by `max_byte + 1` instead so a byte read at
        // dispatch time does not index past the end.  Empty slots in
        // the gaps stay as `String::new()` and surface as the
        // unwired-handler placeholder if dispatched against — same
        // behaviour RPython relies on for unregistered opcodes.
        // Empty `insns` → empty reverse table (RPython parity:
        // `[None] * len(insns)` with `len == 0`).  Non-empty → size to
        // `max_byte + 1` so a byte read at dispatch time does not
        // index past the end.
        let table_len = match insns.values().copied().max() {
            Some(max_byte) => (max_byte as usize) + 1,
            None => 0,
        };
        self._insns = vec![String::new(); table_len];
        for (key, &value) in insns {
            assert!(
                self._insns[value as usize].is_empty(),
                "duplicate opcode {value} for key {key:?}"
            );
            self._insns[value as usize] = key.clone();
        }
        // RPython blackhole.py:72-74: resolve well-known opcodes
        self.op_live = insns.get("live/").copied().unwrap_or(u8::MAX);
        self.op_catch_exception = insns.get("catch_exception/L").copied().unwrap_or(u8::MAX);
        self.op_rvmprof_code = insns.get("rvmprof_code/ii").copied().unwrap_or(u8::MAX);
        // RPython blackhole.py:76-80: build handler table.
        //
        // RPython immediately calls _get_method(name, argcodes) for every
        // insns key and panics if the corresponding bhimpl_* is missing.
        // We match that behavior: the default handler panics with the
        // opname so missing bhimpl_* methods surface at dispatch time
        // instead of being silently swallowed.
        self.dispatch_table = std::sync::Arc::new(vec![
            unwired_handler_placeholder
                as BhOpcodeHandler;
            self._insns.len()
        ]);
    }

    /// List of opnames whose dispatch table entry is still the
    /// `setup_insns` placeholder — i.e. no `bhimpl_*` handler has been
    /// wired for them yet.
    ///
    /// RPython has no analogue: `setup_insns` (blackhole.py:66) resolves
    /// every entry via `_get_method` and would raise `AttributeError`
    /// immediately for a missing `bhimpl_*`. pyre splits that into two
    /// phases — `setup_insns` fills placeholders, `wire_bhimpl_handlers`
    /// resolves them — so the final fail-fast check lives at the caller
    /// (see `build_default_bh_builder` in `pyre-jit-trace`). This
    /// accessor is the diagnostic handle used by that assertion.
    pub fn unwired_opnames(&self) -> Vec<&str> {
        let placeholder = unwired_handler_placeholder as BhOpcodeHandler;
        self._insns
            .iter()
            .enumerate()
            .filter_map(|(i, key)| {
                if key.is_empty() {
                    return None;
                }
                if self.dispatch_table[i] as usize == placeholder as usize {
                    Some(key.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// RPython `blackhole.py:102-103` `setup_descrs(descrs)`.
    pub fn setup_descrs(&mut self, descrs: Vec<BhDescr>) {
        self.descrs = descrs;
    }

    /// Resolve JitCode fnaddr values from a mapping function.
    /// RPython: fnaddr is already set on JitCode objects when they're stored in descrs.
    /// pyre: fnaddr is 0 at assembly time, resolved here after compilation.
    /// `resolver(jitcode_index) -> fnaddr`.
    pub fn resolve_jitcode_fnaddrs(&mut self, resolver: impl Fn(usize) -> i64) {
        for descr in &mut self.descrs {
            if let BhDescr::JitCode {
                jitcode_index,
                fnaddr,
                ..
            } = descr
            {
                if *fnaddr == 0 {
                    *fnaddr = resolver(*jitcode_index);
                }
            }
        }
    }

    /// Resolve field descriptor offsets from a mapping function.
    /// RPython: FieldDescr carries actual byte offset from rtyper.
    /// pyre: offset is 0 at assembly time, resolved here from runtime layout.
    /// `resolver(owner, field_name) -> byte_offset`.
    pub fn resolve_field_offsets(&mut self, resolver: impl Fn(&str, &str) -> usize) {
        for descr in &mut self.descrs {
            if let BhDescr::Field {
                offset,
                name,
                owner,
                parent,
                ..
            } = descr
            {
                if *offset == 0 && !name.is_empty() {
                    *offset = resolver(owner, name);
                    if let Some(parent) = parent {
                        let full_name = if owner.is_empty() || name.contains('.') {
                            name.clone()
                        } else {
                            format!("{owner}.{name}")
                        };
                        if let Some(field) = parent
                            .all_fielddescrs
                            .iter_mut()
                            .find(|field| field.name == full_name)
                        {
                            field.offset = *offset;
                        }
                    }
                }
            }
        }
    }

    /// RPython `blackhole.py:83-100` `dispatch_loop(self, code, position)`.
    ///
    /// Runs the codewriter-orthodox bytecode dispatch loop. Each iteration
    /// reads one opcode byte, looks up the handler in `dispatch_table`,
    /// and calls it to advance position.
    pub fn dispatch_loop(
        &self,
        bh: &mut BlackholeInterpreter,
        code: &[u8],
        mut position: usize,
    ) -> Result<(), DispatchError> {
        loop {
            // RPython `blackhole.py:86-91`:
            //
            // ```python
            // if (not we_are_translated()
            //     and self.jitcode._startpoints is not None):
            //     assert position in self.jitcode._startpoints, (
            //         "the current position %d is in the middle of "
            //         "an instruction!" % position)
            // ```
            //
            // pyre is "non-translated" today; `_startpoints is None`
            // (jitcode.py:24 default) is the assembler's opt-out
            // signal. Assembled jitcodes always carry `Some(set)`
            // (assembler.rs `make_jitcode`); helper jitcodes built
            // without the assembler keep `None` and skip the check.
            if let Some(startpoints) = bh.jitcode.startpoints.as_ref() {
                debug_assert!(
                    startpoints.contains(&position),
                    "dispatch_loop: position {position} is in the middle of an instruction \
                     (jitcode {:?})",
                    bh.jitcode.name,
                );
            }
            let opcode = code[position] as usize;
            position += 1;
            if opcode >= self.dispatch_table.len() {
                panic!("bad opcode {opcode} at position {}", position - 1);
            }
            position = self.dispatch_table[opcode](bh, code, position)?;
        }
    }

    /// `dispatch_loop` with a per-opcode probe hook (Phase D-2 shadow
    /// execution scaffolding).
    ///
    /// PRE-EXISTING-ADAPTATION: RPython's `BlackholeInterpBuilder.dispatch_loop`
    /// has no probe parameter because upstream runs a single dispatch
    /// path (jitcode). Pyre is mid-migration: trait-based
    /// `MIFrame::execute_opcode_step` (trace_opcode.rs) and this
    /// codewriter-orthodox jitcode path coexist while Phase D
    /// validates the latter against the former. The probe lets a
    /// shadow-execute caller capture the jitcode op sequence so it
    /// can compare against the trace-side IR emitted by
    /// `execute_opcode_step`.
    ///
    /// Convergence path: removed in Phase E (eval-loop automation
    /// plan) when trait dispatch is deleted and `dispatch_loop`
    /// becomes the single production path. Until then this method
    /// stays alongside `dispatch_loop` without disturbing it — the
    /// non-probe loop is verbatim RPython parity, the probe variant
    /// is the opt-in transitional surface.
    ///
    /// `probe(bh_view, pc, opcode_byte, opname_key)` fires once per
    /// dispatched opcode, BEFORE the handler runs.
    ///
    /// * `bh_view` — shared reborrow of the interpreter. The probe
    ///   can read `registers_i/r/f`, `tmpreg_*`, `descrs`, etc. to
    ///   capture the input data flow consumed by the upcoming
    ///   handler. The reborrow expires at the end of the probe call,
    ///   so the handler runs against the same `&mut BlackholeInterpreter`
    ///   the loop holds.
    /// * `pc` — position of the opcode byte (not the position after
    ///   operand decode). `code[pc] == opcode_byte`.
    /// * `opname_key` — the `_insns[opcode]` entry (e.g.
    ///   `"int_add/ii>i"`).
    pub fn dispatch_loop_with_probe(
        &self,
        bh: &mut BlackholeInterpreter,
        code: &[u8],
        mut position: usize,
        mut probe: impl FnMut(&BlackholeInterpreter, usize, u8, &str),
    ) -> Result<(), DispatchError> {
        loop {
            let pc = position;
            let opcode = code[position];
            let opcode_idx = opcode as usize;
            if opcode_idx >= self.dispatch_table.len() {
                panic!("bad opcode {opcode_idx} at position {pc}");
            }
            let opname = self._insns[opcode_idx].as_str();
            probe(&*bh, pc, opcode, opname);
            position += 1;
            position = self.dispatch_table[opcode_idx](bh, code, position)?;
        }
    }

    /// Wire a handler for a specific opname/argcodes key into the dispatch table.
    ///
    /// RPython `blackhole.py:76-80`: iterates `_insns` and calls
    /// `_get_method(name, argcodes)` for each. In Rust we wire specific
    /// opnames one by one during Phase D migration. Returns false if
    /// the opname is not present in the insns table.
    /// Wire a handler for a specific opname/argcodes key.
    ///
    /// Returns true if the key was found in the insns table.
    /// Callers in wire_bhimpl_handlers use `try_wire_handler` for optional
    /// keys (aliases that may not exist in all assembler configurations).
    pub(crate) fn wire_handler(&mut self, opname_key: &str, handler: BhOpcodeHandler) -> bool {
        for (i, key) in self._insns.iter().enumerate() {
            if key == opname_key {
                std::sync::Arc::make_mut(&mut self.dispatch_table)[i] = handler;
                return true;
            }
        }
        false
    }

    /// Acquire an interpreter from the pool or create a new one.
    ///
    /// RPython `blackhole.py:245-251`:
    /// ```python
    /// def acquire_interp(self):
    ///     res = self.blackholeinterps
    ///     if res is not None:
    ///         self.blackholeinterps = res.back
    ///         return res
    ///     else:
    ///         return BlackholeInterpreter(self)
    /// ```
    /// Note: RPython's `BlackholeInterpreter(self)` passes `builder` to
    /// `__init__`, which stores `self.cpu = builder.cpu`. We propagate
    /// the `cpu` field from the builder to each acquired interpreter.
    pub fn acquire_interp(&mut self) -> BlackholeInterpreter {
        let mut bh = self.pool.pop().unwrap_or_default();
        // RPython blackhole.py:284-289:
        //   self.cpu = builder.cpu
        //   self.dispatch_loop = builder.dispatch_loop
        //   self.descrs = builder.descrs
        //   self.op_catch_exception = builder.op_catch_exception
        //   self.op_rvmprof_code = builder.op_rvmprof_code
        bh.cpu = self.cpu;
        // RPython blackhole.py:288: self.descrs = builder.descrs
        bh.descrs = self.descrs.clone();
        bh.op_catch_exception = self.op_catch_exception;
        bh.op_rvmprof_code = self.op_rvmprof_code;
        //   self.op_live = builder.op_live
        bh.op_live = self.op_live;
        // RPython blackhole.py:287: self.dispatch_loop = builder.dispatch_loop
        bh.dispatch_table = std::sync::Arc::clone(&self.dispatch_table);
        bh
    }

    /// blackhole.py:253 release_interp
    pub fn release_interp(&mut self, mut interp: BlackholeInterpreter) {
        // blackhole.py:254
        interp.cleanup_registers();
        // Pool management (RPython uses linked-list via .back; Rust uses Vec)
        interp.nextblackholeinterp = None;
        interp.aborted = false;
        interp.got_exception = false;
        interp.virtualizable_stack_base = 0;
        self.pool.push(interp);
    }

    /// Release an entire blackhole chain (including all nextblackholeinterps).
    pub fn release_chain(&mut self, chain: Option<BlackholeInterpreter>) {
        let mut current = chain;
        while let Some(mut bh) = current {
            let next = bh.nextblackholeinterp.take().map(|b| *b);
            self.release_interp(bh);
            current = next;
        }
    }
}

/// warmspot.py:961 handle_jitexception parity.
///
/// Dispatches on JitException type and returns (return_type, value).
/// For ContinueRunningNormally, calls portal_runner to re-enter the
/// portal function. The portal_runner may itself raise a JitException,
/// which is returned as Err for the caller to re-dispatch (while loop).
fn handle_jitexception_dispatch(
    exc: JitException,
    portal_runner: Option<&dyn Fn(&JitException) -> Result<(BhReturnType, i64), JitException>>,
) -> Result<(BhReturnType, i64), JitException> {
    match exc {
        // warmspot.py:986-987
        JitException::DoneWithThisFrameVoid => Ok((BhReturnType::Void, 0)),
        // warmspot.py:988-990
        JitException::DoneWithThisFrameInt(result) => Ok((BhReturnType::Int, result)),
        // warmspot.py:991-993
        JitException::DoneWithThisFrameRef(result) => Ok((BhReturnType::Ref, result.0 as i64)),
        // warmspot.py:994-996
        JitException::DoneWithThisFrameFloat(result) => {
            Ok((BhReturnType::Float, result.to_bits() as i64))
        }
        // warmspot.py:998-1005
        JitException::ExitFrameWithExceptionRef(_) => Err(exc),
        // warmspot.py:970-983
        JitException::ContinueRunningNormally { .. } => {
            // warmspot.py:976-978: result = portal_ptr(*args)
            // May raise JitException → Err propagated for re-dispatch.
            let runner = portal_runner.expect("ContinueRunningNormally requires portal_runner");
            runner(&exc)
        }
    }
}

/// blackhole.py:1684 _handle_jitexception_in_portal +
/// warmspot.py:1039 handle_jitexception_from_blackhole
///
/// Handle a JitException at a recursive portal level.
/// warmspot.py:1040: result = handle_jitexception(e)
/// warmspot.py:1041-1050: bhcaller._setup_return_value_{i,r,f}(result)
///
/// Returns Ok(()) on success (return value set in bhcaller),
/// or Err(exc_value) if the exception should be propagated as a
/// regular exception (ExitFrameWithExceptionRef).
fn handle_jitexception_in_portal(
    bhcaller: &mut BlackholeInterpreter,
    exc: JitException,
    portal_runner: Option<&dyn Fn(&JitException) -> Result<(BhReturnType, i64), JitException>>,
) -> Result<(), i64> {
    // warmspot.py:961 handle_jitexception: while True loop.
    // ContinueRunningNormally → portal_runner → may raise JitException → loop.
    let mut current_exc = exc;
    loop {
        match handle_jitexception_dispatch(current_exc, portal_runner) {
            Ok((ret_type, result)) => {
                // warmspot.py:1041-1050
                match ret_type {
                    BhReturnType::Void => {}
                    BhReturnType::Int => bhcaller.setup_return_value_i(result),
                    BhReturnType::Ref => bhcaller.setup_return_value_r(result),
                    BhReturnType::Float => bhcaller.setup_return_value_f(result),
                }
                return Ok(());
            }
            Err(JitException::ExitFrameWithExceptionRef(exc_ref)) => {
                // warmspot.py:998-1005: raise as regular exception
                return Err(exc_ref.0 as i64);
            }
            Err(next_exc) => {
                // warmspot.py:967-968, 979-980: JitException from portal_runner
                // or EnterJitAssembler → loop back in handle_jitexception
                current_exc = next_exc;
                continue;
            }
        }
    }
}

/// blackhole.py:1762 _handle_jitexception
///
/// Route a JitException through the blackhole frame chain.
/// Walks up the chain until a portal frame is found. If the portal
/// is the bottommost frame, the exception propagates out. Otherwise
/// it's handled at the recursive portal level.
fn handle_jitexception(
    builder: &mut BlackholeInterpBuilder,
    mut bh: BlackholeInterpreter,
    exc: JitException,
    portal_runner: Option<&dyn Fn(&JitException) -> Result<(BhReturnType, i64), JitException>>,
) -> Result<(BlackholeInterpreter, i64), JitException> {
    // blackhole.py:1764: while blackholeinterp.jitcode.jitdriver_sd is None
    while bh.jitcode.jitdriver_sd().is_none() {
        let next = bh.nextblackholeinterp.take();
        builder.release_interp(bh);
        match next.map(|b| *b) {
            Some(caller) => bh = caller,
            None => return Err(exc), // no portal found
        }
    }

    // blackhole.py:1767-1769
    if bh.nextblackholeinterp.is_none() {
        // Bottommost entry: exception goes through
        builder.release_interp(bh);
        return Err(exc);
    }

    // blackhole.py:1770-1780: recursive portal level.
    // _handle_jitexception_in_portal(exc) calls jd.handle_jitexc_from_bh,
    // which is warmspot.py:1039 handle_jitexception_from_blackhole:
    //   result = handle_jitexception(e)
    //   bhcaller._setup_return_value_{i,r,f}(result)
    //
    // handle_jitexception (warmspot.py:961) extracts the result from
    // DoneWithThisFrame{Int,Ref,Float,Void} and returns it.
    //
    // In Rust we can do this directly since JitException carries the result.
    let caller = bh.nextblackholeinterp.as_mut().unwrap();
    let current_exc = match handle_jitexception_in_portal(caller, exc, portal_runner) {
        Ok(()) => 0,
        Err(regular_exc) => regular_exc,
    };
    // blackhole.py:1780: return blackholeinterp, lle
    Ok((bh, current_exc))
}

/// blackhole.py:1752 _run_forever
///
/// Execute a blackhole frame chain to completion.
/// Loops through frames: runs each one via `resume_mainloop`, releases it,
/// then moves to the caller frame. Terminates when the bottommost frame
/// raises a JitException (DoneWithThisFrame* or ExitFrameWithException*).
///
/// Returns the JitException that terminated execution.
pub fn run_forever(
    builder: &mut BlackholeInterpBuilder,
    bh: BlackholeInterpreter,
    current_exc: i64,
) -> JitException {
    run_forever_with_portal(builder, bh, current_exc, None)
}

/// blackhole.py:1752 _run_forever with optional portal runner callback.
///
/// `portal_runner` is warmspot.py:961 handle_jitexception parity:
/// when ContinueRunningNormally is raised at a recursive portal level,
/// this callback re-enters the portal function with the exception's
/// green/red args and returns the result.
pub fn run_forever_with_portal(
    builder: &mut BlackholeInterpBuilder,
    mut bh: BlackholeInterpreter,
    mut current_exc: i64,
    portal_runner: Option<&dyn Fn(&JitException) -> Result<(BhReturnType, i64), JitException>>,
) -> JitException {
    loop {
        // blackhole.py:1754-1755
        match bh.resume_mainloop(current_exc) {
            Ok(exc) => {
                current_exc = exc;
            }
            Err(jit_exc) => {
                // blackhole.py:1756-1758
                match handle_jitexception(builder, bh, jit_exc, portal_runner) {
                    Ok((new_bh, exc)) => {
                        // Handled at recursive portal level — continue
                        bh = new_bh;
                        current_exc = exc;
                        continue;
                    }
                    Err(propagated_exc) => {
                        // Bottommost or unhandled — propagate out
                        return propagated_exc;
                    }
                }
            }
        }

        // blackhole.py:1759
        let next = bh.nextblackholeinterp.take();
        builder.release_interp(bh);
        // blackhole.py:1760
        // RPython: blackholeinterp = blackholeinterp.nextblackholeinterp
        // In RPython this can be None, but _resume_mainloop on the
        // bottommost frame always raises a JitException (via
        // _done_with_this_frame or _exit_frame_with_exception), so
        // this code is unreachable in normal operation.
        bh = *next.expect("_run_forever: nextblackholeinterp is None (unreachable)");
    }
}

/// blackhole.py:1798 convert_and_run_from_pyjitpl
///
/// Get a chain of blackhole interpreters and fill them by copying
/// 'metainterp.framestack'.
pub fn convert_and_run_from_pyjitpl(
    builder: &mut BlackholeInterpBuilder,
    framestack: &MIFrameStack,
    last_exc_value: i64,
    raising_exception: bool,
) -> JitException {
    // blackhole.py:1803-1810
    let mut next_bh: Option<Box<BlackholeInterpreter>> = None;

    for frame in &framestack.frames {
        let mut cur_bh = builder.acquire_interp();
        cur_bh.copy_data_from_miframe(frame);
        cur_bh.nextblackholeinterp = next_bh;
        next_bh = Some(Box::new(cur_bh));
    }

    let Some(first_bh_box) = next_bh else {
        return JitException::DoneWithThisFrameVoid;
    };
    let mut first_bh = *first_bh_box;

    // blackhole.py:1812-1818
    let current_exc = if raising_exception {
        last_exc_value
    } else {
        first_bh.exception_last_value = last_exc_value;
        0
    };

    run_forever(builder, first_bh, current_exc)
}

/// blackhole.py:1782 resume_in_blackhole
///
/// Resume execution in the blackhole interpreter after a compiled
/// code guard failure. Builds a frame chain from resume data, extracts
/// exception from deadframe, and runs the chain to completion.
///
/// `resolve_jitcode` is `metainterp_sd.jitcodes[jitcode_pos]` in RPython.
pub fn resume_in_blackhole(
    builder: &mut BlackholeInterpBuilder,
    resolve_jitcode: &dyn Fn(i32, i32) -> Option<crate::resume::ResolvedJitCode>,
    rd_numb: &[u8],
    rd_consts: &[majit_ir::Const],
    all_liveness: &[u8],
    deadframe: &[i64],
    deadframe_exc: i64,
) -> JitException {
    // blackhole.py:1786-1792
    let null_alloc = crate::resume::NullAllocator;
    let bh = crate::resume::blackhole_from_resumedata(
        builder,
        resolve_jitcode,
        rd_numb,
        rd_consts,
        all_liveness,
        deadframe,
        None, // deadframe_types
        None, // rd_virtuals
        None, // rd_guard_pendingfields
        None, // vrefinfo
        None, // vinfo
        None, // ginfo
        &null_alloc,
    );

    let Some((bh, _virtualizable_ptr)) = bh else {
        return JitException::DoneWithThisFrameVoid;
    };

    // blackhole.py:1794
    let current_exc = BlackholeInterpreter::prepare_resume_from_failure(deadframe_exc);

    // blackhole.py:1795
    run_forever(builder, bh, current_exc)
}

#[cfg(test)]
mod tests {
    //! Upstream parity anchor: `rpython/jit/metainterp/test/test_blackhole.py`
    //! for interpreter pooling, exception-state handling, and blackhole control
    //! flow.
    //!
    //! The opcode-by-opcode `execute_one` coverage below is mostly original
    //! Rust unit coverage. PyPy does not keep a dedicated `test_executor.py` in
    //! this tree, so these tests exist to pin down the local Rust dispatcher
    //! and arithmetic edge cases directly.

    use super::*;
    use majit_ir::OpRef;

    fn mk_op(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let mut op = Op::new(opcode, args);
        op.pos = OpRef::int_op(pos);
        op
    }

    #[test]
    fn test_blackhole_simple_arithmetic() {
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::int_op(0), OpRef::int_op(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::IntAdd, &[OpRef::int_op(0), OpRef::int_op(1)], 2),
            mk_op(OpCode::Finish, &[OpRef::int_op(2)], OpRef::NONE.raw()),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 10i64);
        initial.insert(1, 20i64);

        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::Finish { values: vals, .. } => assert_eq!(vals, vec![30]),
            other => panic!(
                "expected Finish, got {:?}",
                match other {
                    BlackholeResult::Abort(s) => s,
                    _ => "other".to_string(),
                }
            ),
        }
    }

    #[test]
    fn test_blackhole_guard_passes() {
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::int_op(0)], OpRef::NONE.raw()),
            mk_op(OpCode::GuardTrue, &[OpRef::int_op(0)], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::int_op(0)], OpRef::NONE.raw()),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 1i64);

        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::Finish { values: vals, .. } => assert_eq!(vals, vec![1]),
            _ => panic!("expected Finish"),
        }
    }

    #[test]
    fn test_blackhole_exception_guard_no_exception_passes() {
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::int_op(0)], OpRef::NONE.raw()),
            mk_op(OpCode::GuardNoException, &[], OpRef::NONE.raw()),
            mk_op(OpCode::Finish, &[OpRef::int_op(0)], OpRef::NONE.raw()),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 42i64);

        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::Finish { values: vals, .. } => assert_eq!(vals, vec![42]),
            _ => panic!("expected Finish when no exception is pending"),
        }
    }

    #[test]
    fn test_blackhole_initial_exception_fails_guard_no_exception_without_restore() {
        let mut guard_op = mk_op(OpCode::GuardNoException, &[], OpRef::NONE.raw());
        guard_op.fail_args = Some(smallvec::smallvec![OpRef::int_op(0)]);

        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::int_op(0)], OpRef::NONE.raw()),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef::int_op(0)], OpRef::NONE.raw()),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 42i64);

        match blackhole_execute_with_exception(
            &ops,
            &HashMap::new(),
            &initial,
            0,
            ExceptionState {
                exc_class: 100,
                exc_value: 200,
                ovf_flag: false,
            },
        ) {
            BlackholeResult::GuardFailed { .. } => {}
            _ => panic!("expected GuardFailed when initial exception is pending"),
        }
    }

    #[test]
    fn test_blackhole_initial_exception_satisfies_guard_exception_without_restore() {
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::int_op(0)], OpRef::NONE.raw()),
            mk_op(OpCode::GuardException, &[OpRef::int_op(0)], 1),
            mk_op(OpCode::Finish, &[OpRef::int_op(1)], OpRef::NONE.raw()),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 100i64);

        match blackhole_execute_with_exception(
            &ops,
            &HashMap::new(),
            &initial,
            0,
            ExceptionState {
                exc_class: 100,
                exc_value: 200,
                ovf_flag: false,
            },
        ) {
            BlackholeResult::Finish { values: vals, .. } => assert_eq!(vals, vec![200]),
            _ => panic!("expected Finish when initial exception class matches"),
        }
    }

    #[test]
    fn test_blackhole_exception_save_exc_class() {
        // RestoreException sets exception state, then SaveExcClass reads it.
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::int_op(0), OpRef::int_op(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::RestoreException,
                &[OpRef::int_op(0), OpRef::int_op(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::SaveExcClass, &[], 2),
            mk_op(OpCode::SaveException, &[], 3),
            mk_op(
                OpCode::Finish,
                &[OpRef::int_op(2), OpRef::int_op(3)],
                OpRef::NONE.raw(),
            ),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 100i64); // exc_class
        initial.insert(1, 200i64); // exc_value

        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::Finish { values: vals, .. } => assert_eq!(vals, vec![100, 200]),
            _ => panic!("expected Finish"),
        }
    }

    #[test]
    fn test_blackhole_guard_no_exception_fails_with_exception() {
        // RestoreException then GuardNoException should fail.
        let mut guard_op = mk_op(OpCode::GuardNoException, &[], OpRef::NONE.raw());
        guard_op.fail_args = Some(smallvec::smallvec![OpRef::int_op(0)]);

        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::int_op(0), OpRef::int_op(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::RestoreException,
                &[OpRef::int_op(0), OpRef::int_op(1)],
                OpRef::NONE.raw(),
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef::int_op(0)], OpRef::NONE.raw()),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 100i64);
        initial.insert(1, 200i64);

        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::GuardFailed { .. } => {} // expected
            _ => panic!("expected GuardFailed when exception is pending"),
        }
    }

    #[test]
    fn test_blackhole_guard_exception_matches() {
        // Set exception, then GuardException with matching class should pass.
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::int_op(0), OpRef::int_op(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::RestoreException,
                &[OpRef::int_op(0), OpRef::int_op(1)],
                OpRef::NONE.raw(),
            ),
            // GuardException expects class in arg(0)
            mk_op(OpCode::GuardException, &[OpRef::int_op(0)], 2),
            mk_op(OpCode::Finish, &[OpRef::int_op(2)], OpRef::NONE.raw()),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 100i64); // exc_class
        initial.insert(1, 200i64); // exc_value

        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::Finish { values: vals, .. } => {
                // GuardException should return the exc_value
                assert_eq!(vals, vec![200]);
            }
            _ => panic!("expected Finish when exception class matches"),
        }
    }

    #[test]
    fn test_blackhole_guard_fails() {
        let mut guard_op = mk_op(OpCode::GuardTrue, &[OpRef::int_op(0)], OpRef::NONE.raw());
        guard_op.fail_args = Some(smallvec::smallvec![OpRef::int_op(0)]);

        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::int_op(0)], OpRef::NONE.raw()),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef::int_op(0)], OpRef::NONE.raw()),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 0i64);

        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::GuardFailed { fail_values, .. } => {
                assert_eq!(fail_values, vec![0]);
            }
            _ => panic!("expected GuardFailed"),
        }
    }

    // ── Executor opcode tests (Rust-original dispatcher coverage) ──
    //
    // Systematic correctness tests for each opcode category handled by the
    // local Rust blackhole dispatcher.

    /// Helper: build and execute a single binop, returning the i64 result.
    fn exec_binop(opcode: OpCode, a: i64, b: i64) -> i64 {
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::int_op(0), OpRef::int_op(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(opcode, &[OpRef::int_op(0), OpRef::int_op(1)], 2),
            mk_op(OpCode::Finish, &[OpRef::int_op(2)], OpRef::NONE.raw()),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, a);
        initial.insert(1, b);
        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::Finish { values, .. } => values[0],
            other => panic!(
                "expected Finish for {:?}, got {:?}",
                opcode,
                match other {
                    BlackholeResult::Abort(s) => s,
                    _ => "other".to_string(),
                }
            ),
        }
    }

    /// Helper: build and execute a single unary op, returning the i64 result.
    fn exec_unop(opcode: OpCode, a: i64) -> i64 {
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::int_op(0)], OpRef::NONE.raw()),
            mk_op(opcode, &[OpRef::int_op(0)], 1),
            mk_op(OpCode::Finish, &[OpRef::int_op(1)], OpRef::NONE.raw()),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, a);
        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::Finish { values, .. } => values[0],
            other => panic!(
                "expected Finish for {:?}, got {:?}",
                opcode,
                match other {
                    BlackholeResult::Abort(s) => s,
                    _ => "other".to_string(),
                }
            ),
        }
    }

    // ── Integer arithmetic & comparison (consolidated) ──

    #[test]
    fn test_executor_int_arithmetic() {
        // ADD
        assert_eq!(exec_binop(OpCode::IntAdd, 3, 4), 7);
        assert_eq!(exec_binop(OpCode::IntAdd, -1, 1), 0);
        assert_eq!(exec_binop(OpCode::IntAdd, 0, 0), 0);
        // SUB
        assert_eq!(exec_binop(OpCode::IntSub, 10, 3), 7);
        assert_eq!(exec_binop(OpCode::IntSub, 0, 5), -5);
        // MUL
        assert_eq!(exec_binop(OpCode::IntMul, 6, 7), 42);
        assert_eq!(exec_binop(OpCode::IntMul, -3, 4), -12);
        assert_eq!(exec_binop(OpCode::IntMul, 0, 999), 0);
        // FLOORDIV
        assert_eq!(exec_binop(OpCode::IntFloorDiv, 17, 5), 3);
        assert_eq!(exec_binop(OpCode::IntFloorDiv, -17, 5), -3);
        assert_eq!(exec_binop(OpCode::IntFloorDiv, 100, 1), 100);
        // MOD
        assert_eq!(exec_binop(OpCode::IntMod, 17, 5), 2);
        assert_eq!(exec_binop(OpCode::IntMod, 10, 3), 1);
        assert_eq!(exec_binop(OpCode::IntMod, 6, 3), 0);
    }

    #[test]
    fn test_executor_int_comparisons() {
        // LT
        assert_eq!(exec_binop(OpCode::IntLt, 3, 4), 1);
        assert_eq!(exec_binop(OpCode::IntLt, 4, 4), 0);
        assert_eq!(exec_binop(OpCode::IntLt, 5, 4), 0);
        // GE
        assert_eq!(exec_binop(OpCode::IntGe, 4, 4), 1);
        assert_eq!(exec_binop(OpCode::IntGe, 5, 4), 1);
        assert_eq!(exec_binop(OpCode::IntGe, 3, 4), 0);
        // EQ / NE
        assert_eq!(exec_binop(OpCode::IntEq, 5, 5), 1);
        assert_eq!(exec_binop(OpCode::IntEq, 5, 6), 0);
        assert_eq!(exec_binop(OpCode::IntNe, 5, 5), 0);
        assert_eq!(exec_binop(OpCode::IntNe, 5, 6), 1);
        // LE / GT
        assert_eq!(exec_binop(OpCode::IntLe, 3, 4), 1);
        assert_eq!(exec_binop(OpCode::IntLe, 4, 4), 1);
        assert_eq!(exec_binop(OpCode::IntLe, 5, 4), 0);
        assert_eq!(exec_binop(OpCode::IntGt, 5, 4), 1);
        assert_eq!(exec_binop(OpCode::IntGt, 4, 4), 0);
    }

    #[test]
    fn test_executor_int_unary() {
        assert_eq!(exec_unop(OpCode::IntNeg, 42), -42);
        assert_eq!(exec_unop(OpCode::IntNeg, -1), 1);
        assert_eq!(exec_unop(OpCode::IntInvert, 0), -1);
        assert_eq!(exec_unop(OpCode::IntInvert, -1), 0);
    }

    #[test]
    fn test_executor_float_arithmetic() {
        let fb = |v: f64| f64::to_bits(v) as i64;
        let fr = |r: i64| f64::from_bits(r as u64);
        assert_eq!(fr(exec_binop(OpCode::FloatAdd, fb(1.5), fb(2.5))), 4.0);
        assert_eq!(fr(exec_binop(OpCode::FloatMul, fb(3.0), fb(4.0))), 12.0);
        assert_eq!(fr(exec_binop(OpCode::FloatSub, fb(10.0), fb(3.5))), 6.5);
        assert_eq!(fr(exec_binop(OpCode::FloatTrueDiv, fb(10.0), fb(4.0))), 2.5);
    }

    #[test]
    fn test_executor_float_unary() {
        let fb = |v: f64| f64::to_bits(v) as i64;
        let fr = |r: i64| f64::from_bits(r as u64);
        assert_eq!(fr(exec_unop(OpCode::FloatNeg, fb(3.14))), -3.14);
        assert_eq!(fr(exec_unop(OpCode::FloatAbs, fb(-2.5))), 2.5);
        assert_eq!(fr(exec_unop(OpCode::FloatAbs, fb(2.5))), 2.5);
    }

    #[test]
    fn test_executor_int_float_casts() {
        let fr = |r: i64| f64::from_bits(r as u64);
        assert_eq!(fr(exec_unop(OpCode::CastIntToFloat, 42)), 42.0);
        assert_eq!(
            fr(exec_unop(OpCode::CastIntToFloat, i64::MAX)),
            i64::MAX as f64
        );
        assert_eq!(
            exec_unop(OpCode::CastFloatToInt, f64::to_bits(3.7) as i64),
            3
        );
        assert_eq!(
            exec_unop(OpCode::CastFloatToInt, f64::to_bits(-2.9) as i64),
            -2
        );
        assert_eq!(exec_unop(OpCode::CastFloatToInt, f64_bits(2.9)), 2);
        assert_eq!(exec_unop(OpCode::CastFloatToInt, f64_bits(-2.9)), -2);
        assert_eq!(exec_unop(OpCode::CastFloatToInt, f64_bits(0.999)), 0);
    }

    // ── Overflow arithmetic ──

    #[test]
    fn test_executor_overflow_arithmetic_uses_blackhole_wrapping_semantics() {
        assert_eq!(exec_binop(OpCode::IntAddOvf, 10, 20), 30);
        assert_eq!(exec_binop(OpCode::IntSubOvf, 10, 3), 7);
        assert_eq!(exec_binop(OpCode::IntMulOvf, 6, 7), 42);

        // In blackhole mode, overflow ops use wrapping arithmetic.
        // GuardNoOverflow/GuardOverflow are separate ops.
        let result = exec_binop(OpCode::IntAddOvf, i64::MAX, 1);
        assert_eq!(result, i64::MIN);
        assert_eq!(exec_binop(OpCode::IntSubOvf, i64::MIN, 1), i64::MAX);
        assert_eq!(exec_binop(OpCode::IntMulOvf, i64::MAX, 2), -2);
    }

    // ── Bitwise ops ──

    #[test]
    fn test_executor_bitwise_and_shift_ops() {
        assert_eq!(exec_binop(OpCode::IntAnd, 0xFF, 0x0F), 0x0F);
        assert_eq!(exec_binop(OpCode::IntAnd, 0xAB, 0x00), 0x00);
        assert_eq!(exec_binop(OpCode::IntAnd, -1, -1), -1);
        assert_eq!(exec_binop(OpCode::IntAnd, -1, 0), 0);
        assert_eq!(exec_binop(OpCode::IntOr, 0xF0, 0x0F), 0xFF);
        assert_eq!(exec_binop(OpCode::IntOr, 0, 0), 0);
        assert_eq!(exec_binop(OpCode::IntOr, -1, 0), -1);
        assert_eq!(exec_binop(OpCode::IntOr, i64::MAX, i64::MIN), -1);
        assert_eq!(exec_binop(OpCode::IntXor, 0xFF, 0x0F), 0xF0);
        assert_eq!(exec_binop(OpCode::IntXor, 42, 42), 0);
        assert_eq!(exec_binop(OpCode::IntXor, i64::MAX, i64::MAX), 0);
        assert_eq!(exec_binop(OpCode::IntXor, i64::MIN, i64::MIN), 0);
        assert_eq!(exec_unop(OpCode::IntInvert, i64::MAX), i64::MIN);
        assert_eq!(exec_unop(OpCode::IntInvert, i64::MIN), i64::MAX);
        assert_eq!(exec_binop(OpCode::IntLshift, 1, 4), 16);
        assert_eq!(exec_binop(OpCode::IntLshift, 0xFF, 8), 0xFF00);
        assert_eq!(exec_binop(OpCode::IntRshift, 16, 4), 1);
        assert_eq!(exec_binop(OpCode::IntRshift, -1, 1), -1); // arithmetic shift
        // Logical (unsigned) right shift.
        let result = exec_binop(OpCode::UintRshift, -1, 1);
        assert_eq!(result, i64::MAX);
    }

    // ── Boolean predicates ──

    #[test]
    fn test_executor_int_predicates_and_force_ge_zero() {
        assert_eq!(exec_unop(OpCode::IntIsZero, 0), 1);
        assert_eq!(exec_unop(OpCode::IntIsZero, 1), 0);
        assert_eq!(exec_unop(OpCode::IntIsZero, -1), 0);
        assert_eq!(exec_unop(OpCode::IntIsZero, i64::MAX), 0);
        assert_eq!(exec_binop(OpCode::IntAdd, i64::MAX, 0), i64::MAX); // sanity
        assert_eq!(exec_unop(OpCode::IntIsTrue, 0), 0);
        assert_eq!(exec_unop(OpCode::IntIsTrue, 1), 1);
        assert_eq!(exec_unop(OpCode::IntIsTrue, -42), 1);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, 5), 5);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, 0), 0);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, -10), 0);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, i64::MIN), 0);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, i64::MAX), i64::MAX);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, 1), 1);
    }

    // ── Float comparisons ──

    #[test]
    fn test_executor_float_comparisons() {
        let f2 = f64::to_bits(2.0) as i64;
        let f3 = f64::to_bits(3.0) as i64;
        assert_eq!(exec_binop(OpCode::FloatLt, f2, f3), 1);
        assert_eq!(exec_binop(OpCode::FloatLt, f3, f2), 0);
        assert_eq!(exec_binop(OpCode::FloatLe, f2, f2), 1);
        assert_eq!(exec_binop(OpCode::FloatGt, f3, f2), 1);
        assert_eq!(exec_binop(OpCode::FloatGe, f2, f2), 1);
        assert_eq!(exec_binop(OpCode::FloatEq, f3, f3), 1);
        assert_eq!(exec_binop(OpCode::FloatNe, f2, f3), 1);
    }

    // ── FloatFloorDiv / FloatMod ──

    #[test]
    fn test_executor_float_floordiv_and_mod() {
        let a = f64::to_bits(7.0) as i64;
        let b = f64::to_bits(2.0) as i64;
        let result = exec_binop(OpCode::FloatFloorDiv, a, b);
        assert_eq!(f64::from_bits(result as u64), 3.0);
        let a = f64::to_bits(7.0) as i64;
        let b = f64::to_bits(3.0) as i64;
        let result = exec_binop(OpCode::FloatMod, a, b);
        assert_eq!(f64::from_bits(result as u64), 1.0);
        // -7.0 // 2.0 = -4.0
        let result = exec_binop(OpCode::FloatFloorDiv, f64_bits(-7.0), f64_bits(2.0));
        assert_eq!(bits_f64(result), -4.0);
        let result = exec_binop(OpCode::FloatMod, f64_bits(-7.0), f64_bits(3.0));
        let r = bits_f64(result);
        assert!(r.is_finite());
    }

    // ── Pointer comparisons ──

    #[test]
    fn test_executor_ptr_eq_ne() {
        assert_eq!(exec_binop(OpCode::PtrEq, 100, 100), 1);
        assert_eq!(exec_binop(OpCode::PtrEq, 100, 200), 0);
        assert_eq!(exec_binop(OpCode::PtrNe, 100, 200), 1);
        assert_eq!(exec_binop(OpCode::PtrNe, 100, 100), 0);
    }

    // ── SameAs ──

    #[test]
    fn test_executor_same_as() {
        assert_eq!(exec_unop(OpCode::SameAsI, 42), 42);
        assert_eq!(exec_unop(OpCode::SameAsR, 0xDEAD), 0xDEAD);
    }

    #[test]
    fn test_executor_same_as_zero_arg_constant_placeholder() {
        let op = mk_op(OpCode::SameAsI, &[], 8);
        let mut values = HashMap::new();
        values.insert(8, 123);
        let mut exc = ExceptionState::default();
        match execute_one(&op, &values, &mut exc) {
            OpResult::Value(v) => assert_eq!(v, 123),
            _ => panic!("zero-arg SameAsI should resolve through its constant slot"),
        }
    }

    // ── UintMulHigh ──

    #[test]
    fn test_executor_uint_mul_high() {
        // Upper 64 bits of unsigned 128-bit multiply.
        // 0x8000_0000_0000_0000 * 2 = 0x1_0000_0000_0000_0000 → high = 1
        let result = exec_binop(OpCode::UintMulHigh, i64::MIN, 2);
        assert_eq!(result, 1);

        // Small values: high 64 bits should be 0.
        let result = exec_binop(OpCode::UintMulHigh, 100, 200);
        assert_eq!(result, 0);

        // -1i64 = 0xFFFF_FFFF_FFFF_FFFF as u64, * 2 → high = 1
        let result = exec_binop(OpCode::UintMulHigh, -1, 2);
        assert_eq!(result, 1);
        // u64::MAX * u64::MAX = (2^64-1)^2 = 2^128 - 2^65 + 1
        // High 64 bits = 0xFFFFFFFFFFFFFFFE
        let result = exec_binop(OpCode::UintMulHigh, -1, -1);
        assert_eq!(result, -2); // 0xFFFFFFFFFFFFFFFE as i64 = -2
        // Any value * 1 has high 64 bits = 0
        assert_eq!(exec_binop(OpCode::UintMulHigh, i64::MAX, 1), 0);
        assert_eq!(exec_binop(OpCode::UintMulHigh, -1, 1), 0);
    }

    // ══════════════════════════════════════════════════════════════════
    // Executor edge-case parity tests
    // Ported from rpython/jit/metainterp/test/test_executor.py
    // ══════════════════════════════════════════════════════════════════

    // ── Division by zero ──

    #[test]
    fn test_executor_int_divmod_by_zero_returns_blackhole_default() {
        // Division by zero returns 0 (blackhole convention: no panic).
        assert_eq!(exec_binop(OpCode::IntFloorDiv, 42, 0), 0);
        assert_eq!(exec_binop(OpCode::IntFloorDiv, -1, 0), 0);
        assert_eq!(exec_binop(OpCode::IntFloorDiv, 0, 0), 0);
        assert_eq!(exec_binop(OpCode::IntMod, 42, 0), 0);
        assert_eq!(exec_binop(OpCode::IntMod, -1, 0), 0);
        assert_eq!(exec_binop(OpCode::IntMod, 0, 0), 0);
    }

    // ── Integer overflow boundaries ──

    #[test]
    fn test_executor_integer_overflow_boundaries() {
        // Wrapping: i64::MAX + 1 = i64::MIN
        assert_eq!(exec_binop(OpCode::IntAdd, i64::MAX, 1), i64::MIN);
        // i64::MIN - 1 wraps to i64::MAX
        assert_eq!(exec_binop(OpCode::IntSub, i64::MIN, 1), i64::MAX);
        assert_eq!(exec_binop(OpCode::IntMul, i64::MAX, 2), -2);
        assert_eq!(exec_binop(OpCode::IntMul, i64::MIN, -1), i64::MIN); // wrapping
        // -i64::MIN wraps to i64::MIN (two's complement)
        assert_eq!(exec_unop(OpCode::IntNeg, i64::MIN), i64::MIN);
        // i64::MIN / -1 would overflow; wrapping_div wraps to i64::MIN
        assert_eq!(exec_binop(OpCode::IntFloorDiv, i64::MIN, -1), i64::MIN);
        // i64::MIN % -1 would overflow; wrapping_rem wraps to 0
        assert_eq!(exec_binop(OpCode::IntMod, i64::MIN, -1), 0);
    }

    // ── Float special values ──

    fn f64_bits(v: f64) -> i64 {
        f64::to_bits(v) as i64
    }

    fn bits_f64(v: i64) -> f64 {
        f64::from_bits(v as u64)
    }

    #[test]
    fn test_executor_float_special_values() {
        let result = exec_binop(OpCode::FloatAdd, f64_bits(f64::NAN), f64_bits(1.0));
        assert!(bits_f64(result).is_nan());
        // Inf * 0 = NaN
        let result = exec_binop(OpCode::FloatMul, f64_bits(f64::INFINITY), f64_bits(0.0));
        assert!(bits_f64(result).is_nan());
        // 1.0 / 0.0 = Inf
        let result = exec_binop(OpCode::FloatTrueDiv, f64_bits(1.0), f64_bits(0.0));
        assert_eq!(bits_f64(result), f64::INFINITY);
        // -1.0 / 0.0 = -Inf
        let result = exec_binop(OpCode::FloatTrueDiv, f64_bits(-1.0), f64_bits(0.0));
        assert_eq!(bits_f64(result), f64::NEG_INFINITY);
        // 0.0 / 0.0 = NaN
        let result = exec_binop(OpCode::FloatTrueDiv, f64_bits(0.0), f64_bits(0.0));
        assert!(bits_f64(result).is_nan());
        // Inf - Inf = NaN
        let result = exec_binop(
            OpCode::FloatSub,
            f64_bits(f64::INFINITY),
            f64_bits(f64::INFINITY),
        );
        assert!(bits_f64(result).is_nan());
        // -NaN is still NaN
        let result = exec_unop(OpCode::FloatNeg, f64_bits(f64::NAN));
        assert!(bits_f64(result).is_nan());
        let result = exec_unop(OpCode::FloatAbs, f64_bits(f64::NEG_INFINITY));
        assert_eq!(bits_f64(result), f64::INFINITY);
        // All comparisons with NaN return false (0).
        let nan = f64_bits(f64::NAN);
        let one = f64_bits(1.0);
        assert_eq!(exec_binop(OpCode::FloatLt, nan, one), 0);
        assert_eq!(exec_binop(OpCode::FloatLe, nan, one), 0);
        assert_eq!(exec_binop(OpCode::FloatGt, nan, one), 0);
        assert_eq!(exec_binop(OpCode::FloatGe, nan, one), 0);
        assert_eq!(exec_binop(OpCode::FloatEq, nan, nan), 0);
        assert_eq!(exec_binop(OpCode::FloatNe, nan, nan), 1);
    }

    // ── Unsigned comparisons with negative values ──

    #[test]
    fn test_executor_uint_comparisons() {
        // -1i64 as u64 is u64::MAX, which is larger than 1
        assert_eq!(exec_binop(OpCode::UintLt, -1, 1), 0);
        assert_eq!(exec_binop(OpCode::UintGt, -1, 1), 1);
        // -1 as u64 = 0xFFFFFFFFFFFFFFFF (max unsigned)
        assert_eq!(exec_binop(OpCode::UintLt, -1, 0), 0);
        assert_eq!(exec_binop(OpCode::UintGe, -1, 0), 1);
        assert_eq!(exec_binop(OpCode::UintLe, 0, -1), 1);
        assert_eq!(exec_binop(OpCode::UintGt, 0, -1), 0);

        // Equal values
        assert_eq!(exec_binop(OpCode::UintLt, 5, 5), 0);
        assert_eq!(exec_binop(OpCode::UintLe, 5, 5), 1);
        assert_eq!(exec_binop(OpCode::UintGe, 5, 5), 1);
        assert_eq!(exec_binop(OpCode::UintGt, 5, 5), 0);

        // i64::MIN as u64 = 0x8000000000000000 (large positive unsigned)
        assert_eq!(exec_binop(OpCode::UintLt, i64::MIN, 1), 0);
        assert_eq!(exec_binop(OpCode::UintGt, i64::MIN, 1), 1);
    }

    // ── Shift edge cases ──

    #[test]
    fn test_executor_shift_edge_cases() {
        // 1 << 63 = i64::MIN (sign bit)
        assert_eq!(exec_binop(OpCode::IntLshift, 1, 63), i64::MIN);
        // Arithmetic shift: -1 >> 63 = -1 (sign bit fills)
        assert_eq!(exec_binop(OpCode::IntRshift, -1, 63), -1);
        // Logical shift: -1u >> 63 = 1 (top bit only)
        assert_eq!(exec_binop(OpCode::UintRshift, -1, 63), 1);
        assert_eq!(exec_binop(OpCode::IntLshift, 42, 0), 42);
        assert_eq!(exec_binop(OpCode::IntRshift, 42, 0), 42);
        assert_eq!(exec_binop(OpCode::UintRshift, 42, 0), 42);
        // -5 << 2 = -20
        assert_eq!(exec_binop(OpCode::IntLshift, -5, 2), -20);
    }

    // ── IntSignext ──

    #[test]
    fn test_executor_int_signext() {
        // 0xFF sign-extended from 1 byte = -1
        assert_eq!(exec_binop(OpCode::IntSignext, 0xFF, 1), -1);
        // 0x7F sign-extended from 1 byte = 127
        assert_eq!(exec_binop(OpCode::IntSignext, 0x7F, 1), 127);
        // 0x80 sign-extended from 1 byte = -128
        assert_eq!(exec_binop(OpCode::IntSignext, 0x80, 1), -128);
        // 0xFFFF sign-extended from 2 bytes = -1
        assert_eq!(exec_binop(OpCode::IntSignext, 0xFFFF, 2), -1);
        // 0x7FFF = 32767
        assert_eq!(exec_binop(OpCode::IntSignext, 0x7FFF, 2), 32767);
        // 0x8000 = -32768
        assert_eq!(exec_binop(OpCode::IntSignext, 0x8000, 2), -32768);
        // 0xFFFFFFFF sign-extended from 4 bytes = -1
        assert_eq!(exec_binop(OpCode::IntSignext, 0xFFFFFFFF_i64, 4), -1);
        // 0x7FFFFFFF = 2147483647
        assert_eq!(exec_binop(OpCode::IntSignext, 0x7FFFFFFF, 4), 2147483647);
        // 0x80000000 = -2147483648
        assert_eq!(
            exec_binop(OpCode::IntSignext, 0x80000000_i64, 4),
            -2147483648
        );
    }

    // ── ConvertFloatBytesToLonglong / ConvertLonglongBytesToFloat roundtrip ──

    #[test]
    fn test_executor_convert_float_bytes_roundtrip() {
        // ConvertFloatBytesToLonglong is identity (f64 bits as i64)
        // ConvertLonglongBytesToFloat is identity (i64 bits as f64)
        let val = f64::to_bits(3.14) as i64;
        let ll = exec_unop(OpCode::ConvertFloatBytesToLonglong, val);
        assert_eq!(ll, val);
        let back = exec_unop(OpCode::ConvertLonglongBytesToFloat, ll);
        assert_eq!(back, val);
    }

    #[test]
    fn test_executor_convert_float_bytes_special_values() {
        for v in [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let bits = f64::to_bits(v) as i64;
            let ll = exec_unop(OpCode::ConvertFloatBytesToLonglong, bits);
            assert_eq!(ll, bits);
            let back = exec_unop(OpCode::ConvertLonglongBytesToFloat, ll);
            assert_eq!(back, bits);
        }
    }

    /// Verify every OpCode variant has an explicit handler in execute_one
    /// (i.e., none falls through to the Unsupported catch-all).
    #[test]
    fn test_all_opcodes_have_blackhole_handler() {
        let dummy_args = &[
            OpRef::int_op(10_000),
            OpRef::int_op(10_001),
            OpRef::int_op(10_002),
        ];
        let mut constants = HashMap::new();
        constants.insert(10_000, 1i64);
        constants.insert(10_001, 2i64);
        constants.insert(10_002, 3i64);

        let mut values: HashMap<u32, i64> = constants.clone();
        let mut exc = ExceptionState::default();

        for opcode in OpCode::all() {
            // Build a minimal op with enough args for any opcode
            let arity = opcode.arity().unwrap_or(3) as usize;
            let args = &dummy_args[..arity.min(3)];
            let mut op = Op::new(opcode, args);
            op.pos = OpRef::int_op(opcode.as_u16() as u32 + 20_000);

            let result = execute_one(&op, &values, &mut exc);
            // Store the result so subsequent ops can reference it
            if let OpResult::Value(v) = &result {
                values.insert(op.pos.raw(), *v);
            }

            match result {
                OpResult::Unsupported(msg) => {
                    // CallAssembler ops intentionally return Unsupported —
                    // they require force_fn fallback, not blackhole execution.
                    let is_call_assembler = matches!(
                        opcode,
                        OpCode::CallAssemblerI
                            | OpCode::CallAssemblerR
                            | OpCode::CallAssemblerF
                            | OpCode::CallAssemblerN
                    );
                    if !is_call_assembler {
                        panic!("OpCode {:?} returned Unsupported: {}", opcode, msg);
                    }
                }
                // Any other result (Value, Void, Finish, Jump, GuardFailed) is acceptable
                _ => {}
            }
        }
    }

    // ================================================================
    // Tests for jitcode-based BlackholeInterpreter.
    // Upstream parity anchor: `rpython/jit/metainterp/test/test_blackhole.py`
    // plus the dispatch-loop setup in `rpython/jit/metainterp/blackhole.py`.
    // ================================================================

    mod bh_interp_tests {
        use super::super::*;
        use crate::jitcode::JitCodeBuilder;

        #[test]
        fn test_bh_interp_load_const_and_binop() {
            // Build jitcode: r0 = const(10), r1 = const(20), r2 = r0 + r1
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 10);
            b.load_const_i_value(1, 20);
            b.record_binop_i(2, OpCode::IntAdd, 0, 1);
            let jitcode = b.finish();

            let mut builder = BlackholeInterpBuilder::new();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[2], 30);
        }

        #[test]
        fn test_bh_interp_goto_if_not_int_is_true_taken() {
            // Build jitcode: r0 = 0; if r0==0 goto end; r1 = 42; end: r2 = 99
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 0);
            let lbl = b.new_label();
            b.goto_if_not_int_is_true(0, lbl);
            b.load_const_i_value(1, 42); // should be skipped
            b.mark_label(lbl);
            b.load_const_i_value(2, 99);
            let jitcode = b.finish();

            let mut builder = BlackholeInterpBuilder::new();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 0); // skipped, still 0
            assert_eq!(bh.registers_i[2], 99);
        }

        #[test]
        fn test_bh_interp_goto_if_not_int_is_true_not_taken() {
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 1); // nonzero
            let lbl = b.new_label();
            b.goto_if_not_int_is_true(0, lbl);
            b.load_const_i_value(1, 42); // NOT skipped
            b.mark_label(lbl);
            b.load_const_i_value(2, 99);
            let jitcode = b.finish();

            let mut builder = BlackholeInterpBuilder::new();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 42);
            assert_eq!(bh.registers_i[2], 99);
        }

        #[test]
        fn test_bh_interp_jump() {
            let mut b = JitCodeBuilder::default();
            let lbl = b.new_label();
            b.jump(lbl);
            b.load_const_i_value(0, 42); // skipped
            b.mark_label(lbl);
            b.load_const_i_value(1, 99);
            let jitcode = b.finish();

            let mut builder = BlackholeInterpBuilder::new();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[0], 0); // skipped
            assert_eq!(bh.registers_i[1], 99);
        }

        #[test]
        fn test_bh_interp_move() {
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 42);
            b.move_i(1, 0);
            let jitcode = b.finish();

            let mut builder = BlackholeInterpBuilder::new();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 42);
        }

        #[test]
        fn test_bh_interp_unary_neg() {
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 42);
            b.record_unary_i(1, OpCode::IntNeg, 0);
            let jitcode = b.finish();

            let mut builder = BlackholeInterpBuilder::new();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], -42);
        }

        #[test]
        fn test_bh_interp_ptr_nonzero() {
            let mut b = JitCodeBuilder::default();
            b.load_const_r_value(0, 0x1234);
            b.ptr_nonzero(1, 0);
            let jitcode = b.finish();

            let mut builder = BlackholeInterpBuilder::new();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 1);
        }

        #[test]
        fn test_bh_interp_goto_if_not_ptr_nonzero_taken() {
            let mut b = JitCodeBuilder::default();
            b.load_const_r_value(0, 0);
            let lbl = b.new_label();
            b.goto_if_not_ptr_nonzero(0, lbl);
            b.load_const_i_value(1, 42); // should be skipped
            b.mark_label(lbl);
            b.load_const_i_value(2, 99);
            let jitcode = b.finish();

            let mut builder = BlackholeInterpBuilder::new();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 0);
            assert_eq!(bh.registers_i[2], 99);
        }

        #[test]
        fn test_bh_interp_setarg() {
            let mut b = JitCodeBuilder::default();
            // Just record a binop to read r0 + r1
            b.record_binop_i(2, OpCode::IntMul, 0, 1);
            let jitcode = b.finish();

            let mut builder = BlackholeInterpBuilder::new();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            bh.setarg_i(0, 7);
            bh.setarg_i(1, 6);
            let _ = bh.run();

            assert_eq!(bh.registers_i[2], 42);
        }

        #[test]
        fn test_bh_interp_builder_pool() {
            let mut builder = BlackholeInterpBuilder::new();

            let bh1 = builder.acquire_interp();
            assert!(bh1.registers_i.is_empty());

            builder.release_interp(bh1);
            let bh2 = builder.acquire_interp();
            // Reused from pool
            assert!(bh2.registers_i.is_empty());
        }

        #[test]
        fn test_builder_new_leaves_control_opcodes_unset_until_explicit_setup() {
            let mut builder = BlackholeInterpBuilder::new();
            assert_eq!(builder.op_live, u8::MAX);
            assert_eq!(builder.op_catch_exception, u8::MAX);
            assert_eq!(builder.op_rvmprof_code, u8::MAX);

            builder.setup_cached_control_opcodes(88, 89, 91);
            assert_eq!(builder.op_live, 88);
            assert_eq!(builder.op_catch_exception, 89);
            assert_eq!(builder.op_rvmprof_code, 91);
        }

        #[test]
        fn test_bh_interp_inline_call() {
            // Build sub-jitcode: r0 = arg, result = r0 + r0, return r1.
            // pyjitpl.py:2247-2253 requires every callee to terminate
            // in a typed return opcode so the caller's BC_INLINE_CALL
            // can recover the callee's return register via
            // `trailing_return_info()`.
            let mut sub = JitCodeBuilder::default();
            sub.record_binop_i(1, OpCode::IntAdd, 0, 0);
            sub.int_return(1);
            let sub_jitcode = sub.finish();

            // Build main jitcode: r0 = 21, inline_call(sub, arg=r0) → r1
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 21);
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_ir_i(sub_idx, &[(0, 0)], &[], Some(1));
            let jitcode = b.finish();

            // Slice 3.1d: route through `handler_inline_call_pyre_nested`
            // (the production builder shape) instead of the legacy
            // `dispatch_one::BC_INLINE_CALL` fallback so this test
            // exercises the same path as the production blackhole resume.
            let mut builder = super::build_inline_call_only_bh_builder();
            assert!(
                builder.unwired_opnames().is_empty(),
                "build_inline_call_only_bh_builder left opnames unwired: {:?}",
                builder.unwired_opnames(),
            );
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 42);
        }

        /// Slice 3.1e Tier 2.1: ref-typed return propagation through
        /// `handler_inline_call_pyre_nested`.  Sub-jitcode is the ref
        /// identity (passes its ref arg through and `ref_return`s it);
        /// caller passes a non-trivial constant ref and verifies the
        /// caller-side ref dst slot received the same word.
        #[test]
        fn test_bh_interp_inline_call_ref_return() {
            let mut sub = JitCodeBuilder::default();
            sub.ref_return(0);
            let sub_jitcode = sub.finish();

            let mut b = JitCodeBuilder::default();
            b.load_const_r_value(0, 0xDEAD_BEEF);
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_ir_r(sub_idx, &[], &[(0, 0)], Some(1));
            let jitcode = b.finish();

            let mut builder = super::build_inline_call_only_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_r[1], 0xDEAD_BEEF);
        }

        /// Slice 3.1e Tier 2.1: float-typed return propagation.
        /// Sub-jitcode is the float identity; caller passes a constant
        /// float bit-pattern and verifies the caller-side float dst.
        #[test]
        fn test_bh_interp_inline_call_float_return() {
            let mut sub = JitCodeBuilder::default();
            sub.float_return(0);
            let sub_jitcode = sub.finish();

            let bits = f64::to_bits(3.14_f64) as i64;
            let mut b = JitCodeBuilder::default();
            b.load_const_f_value(0, bits);
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_irf_f(sub_idx, &[], &[], &[(0, 0)], Some(1));
            let jitcode = b.finish();

            let mut builder = super::build_inline_call_only_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert_eq!(bh.registers_f[1], bits);
        }

        /// Slice 3.1e Tier 2.1: void-return path skips the
        /// `trailing_return_info()` block in the handler (returns None
        /// for `BC_VOID_RETURN`-terminated jitcodes per
        /// `JitCodeRuntimeExt::trailing_return_info`).  No caller dst
        /// slot is consumed; the caller proceeds without aborted /
        /// got_exception flags being set.  Sub-jitcode keeps register
        /// files empty by skipping all `touch_reg` paths — `void_return`
        /// alone does not touch any register, so a no-arg call is the
        /// minimal void scenario.
        #[test]
        fn test_bh_interp_inline_call_void_return() {
            let mut sub = JitCodeBuilder::default();
            sub.void_return();
            let sub_jitcode = sub.finish();

            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 99);
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_ir_v(sub_idx, &[], &[], None);
            let jitcode = b.finish();

            let mut builder = super::build_inline_call_only_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert!(!bh.aborted, "void inline_call should not abort");
            assert!(
                !bh.got_exception,
                "void inline_call should not set got_exception"
            );
        }

        /// Slice 3.1f Tier 2.2: exception raised inside the callee is
        /// caught by the caller's `catch_exception/L` immediately
        /// following the inline_call.  The handler should sync
        /// `bh.position` to operand-end before invoking
        /// `handle_exception_in_frame`, then return `Ok(bh.position)`
        /// (= catch-handler PC) so subsequent dispatch resumes there.
        #[test]
        fn test_bh_interp_inline_call_raises_caught_by_caller() {
            const EXC_VAL: i64 = 0xCAFE_F00D;
            const FALLTHROUGH_SENTINEL: i64 = 999;
            const CAUGHT_SENTINEL: i64 = 42;

            let mut sub = JitCodeBuilder::default();
            sub.load_const_r_value(0, EXC_VAL);
            sub.emit_raise(0);
            let sub_jitcode = sub.finish();

            let mut b = JitCodeBuilder::default();
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_ir_v(sub_idx, &[], &[], None);
            let handler_lbl = b.new_label();
            b.catch_exception(handler_lbl);
            // Fallthrough path — must not run if catch dispatch is wired.
            b.load_const_i_value(2, FALLTHROUGH_SENTINEL);
            b.int_return(2);
            // Catch handler — only reached if `handle_exception_in_frame`
            // honoured the immediately-following `catch_exception/L`.
            b.mark_label(handler_lbl);
            b.load_const_i_value(2, CAUGHT_SENTINEL);
            b.int_return(2);
            let jitcode = b.finish();

            let mut builder = super::build_inline_call_only_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert!(
                !bh.got_exception,
                "caller-side catch_exception should clear got_exception"
            );
            assert_eq!(
                bh.tmpreg_i, CAUGHT_SENTINEL,
                "caught path must run; fallthrough sentinel is {FALLTHROUGH_SENTINEL}"
            );
            assert_eq!(bh.return_type, BhReturnType::Int);
        }

        /// Slice 3.1f Tier 2.2: exception raised inside the callee
        /// without a caller-side `catch_exception/L` propagates as
        /// `LeaveFrame` with `bh.got_exception` and
        /// `bh.exception_last_value` set.  The handler's
        /// `handle_exception_in_frame` returns false because no
        /// matching catch byte follows the inline_call operands.
        #[test]
        fn test_bh_interp_inline_call_raises_uncaught_propagates() {
            const EXC_VAL: i64 = 0x1234_5678;

            let mut sub = JitCodeBuilder::default();
            sub.load_const_r_value(0, EXC_VAL);
            sub.emit_raise(0);
            let sub_jitcode = sub.finish();

            let mut b = JitCodeBuilder::default();
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_ir_v(sub_idx, &[], &[], None);
            // No `catch_exception` — the next opcode is unreachable but
            // still needs to be a valid encoding so `JitCodeBuilder::finish`
            // produces a well-formed jitcode tail.
            b.load_const_i_value(0, 0);
            b.int_return(0);
            let jitcode = b.finish();

            let mut builder = super::build_inline_call_only_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert!(
                bh.got_exception,
                "uncaught exception should set got_exception"
            );
            assert_eq!(
                bh.exception_last_value, EXC_VAL,
                "exception_last_value must propagate the callee's raised ref"
            );
        }

        /// Slice 3.1g Tier 2.3: callee `abort_permanent/` propagates
        /// `aborted = true` to the caller via the handler's
        /// `if callee.aborted { bh.aborted = true; LeaveFrame }` arm.
        /// `bhimpl_abort_permanent` (blackhole.rs:1714) sets aborted
        /// + returns LeaveFrame when no `BH_LAST_EXC_VALUE` is pending,
        /// which is the case for a clean callee spawned via
        /// `BlackholeInterpreter::for_inline_callee(parent)` (TLS reset
        /// between tests via thread isolation).
        #[test]
        fn test_bh_interp_inline_call_abort_permanent_in_sub_propagates() {
            let mut sub = JitCodeBuilder::default();
            sub.abort_permanent();
            let sub_jitcode = sub.finish();

            let mut b = JitCodeBuilder::default();
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_ir_v(sub_idx, &[], &[], None);
            // Trailing return — must not run on the abort path.
            b.load_const_i_value(0, 0);
            b.int_return(0);
            let jitcode = b.finish();

            let mut builder = super::build_inline_call_only_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert!(
                bh.aborted,
                "callee `abort_permanent/` must propagate aborted=true to caller"
            );
            assert!(
                !bh.got_exception,
                "abort_permanent without pending TLS exception must not set got_exception"
            );
        }

        /// Slice 3.1g Tier 2.3: callee `abort/` propagates aborted=true.
        /// Slice 3.2.1: callee shares the parent's dispatch_table, so
        /// byte `BC_ABORT` either fires the wired handler
        /// (`handler_abort_marker_pyre`, blackhole.rs:5837) or falls
        /// back to `dispatch_one`'s legacy arm
        /// (blackhole.rs:2284-2287); both set `aborted = true` +
        /// LeaveFrame after Slice 3.2.0 unified the two paths.  The
        /// `if callee.aborted { bh.aborted = true; LeaveFrame }` arm
        /// inside `handler_inline_call_pyre_nested` then propagates to
        /// the caller.
        #[test]
        fn test_bh_interp_inline_call_abort_in_sub_propagates() {
            let mut sub = JitCodeBuilder::default();
            sub.abort();
            let sub_jitcode = sub.finish();

            let mut b = JitCodeBuilder::default();
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_ir_v(sub_idx, &[], &[], None);
            b.load_const_i_value(0, 0);
            b.int_return(0);
            let jitcode = b.finish();

            let mut builder = super::build_inline_call_only_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(jitcode), 0);
            let _ = bh.run();

            assert!(
                bh.aborted,
                "callee `abort/` must propagate aborted=true to caller"
            );
        }

        /// Slice 3.2.2: 2-level nested inline_call.  `outer_sub` itself
        /// emits `BC_INLINE_CALL` to invoke `inner_sub`.  Acceptance
        /// criterion for Slice 3.2.1's callee runtime context clone:
        /// the inner-most callee inherits the parent's dispatch_table
        /// via `clone_context_from`, so byte 17 routes to
        /// `handler_inline_call_pyre_nested` recursively instead of
        /// only working through `dispatch_one`'s legacy arm.  The
        /// caller-side ground truth is that the int value threaded
        /// through three frames lands in the outermost caller's
        /// destination register.
        #[test]
        fn test_bh_interp_inline_call_two_level_nested_int_propagates() {
            // inner_sub: receives r0, returns it.
            let mut inner = JitCodeBuilder::default();
            inner.int_return(0);
            let inner_jitcode = inner.finish();

            // outer_sub: receives r0, calls inner_sub(r0) → r1, returns r1.
            let mut outer = JitCodeBuilder::default();
            let inner_idx_in_outer = outer.add_sub_jitcode(inner_jitcode);
            outer.inline_call_ir_i(inner_idx_in_outer, &[(0, 0)], &[], Some(1));
            outer.int_return(1);
            let outer_jitcode = outer.finish();

            // main: r0 = 17, inline_call(outer_sub, arg=(0, 0)) → r1, return r1.
            let mut main = JitCodeBuilder::default();
            main.load_const_i_value(0, 17);
            let outer_idx_in_main = main.add_sub_jitcode(outer_jitcode);
            main.inline_call_ir_i(outer_idx_in_main, &[(0, 0)], &[], Some(1));
            main.int_return(1);
            let main_jitcode = main.finish();

            let mut builder = super::build_inline_call_only_bh_builder();
            let mut bh = builder.acquire_interp();
            bh.setposition(std::sync::Arc::new(main_jitcode), 0);
            let _ = bh.run();

            assert_eq!(
                bh.registers_i[1], 17,
                "2-level nested inline_call must thread int value through both frames"
            );
        }

        /// `clone_context_from` mirrors
        /// `BlackholeInterpBuilder::acquire_interp`'s 6 builder-shared
        /// fields plus parent's virtualizable / jitdriver state (Fix 7
        /// extension).  Direct unit check that the callee receives the
        /// parent's `dispatch_table` Arc, the cached control opcode
        /// bytes, and the virtualizable / jitdriver_sd / stack-base
        /// state a sub-jitcode might consult if it contains a vable /
        /// recursive_call opcode.
        #[test]
        fn test_clone_context_from_mirrors_acquire_interp_fields() {
            let mut builder = super::build_inline_call_only_bh_builder();
            let mut parent = builder.acquire_interp();
            // Make the parent's vable / jitdriver state non-default so
            // the assertion below distinguishes "copied" from
            // "callee-default".
            parent.virtualizable_ptr = 0xDEAD_BEEF;
            parent.virtualizable_stack_base = 7;

            let mut callee = BlackholeInterpreter::default();
            callee.clone_context_from(&parent);

            assert!(
                std::sync::Arc::ptr_eq(&callee.dispatch_table, &parent.dispatch_table),
                "callee must share the parent's dispatch_table Arc"
            );
            assert_eq!(callee.op_catch_exception, parent.op_catch_exception);
            assert_eq!(callee.op_rvmprof_code, parent.op_rvmprof_code);
            assert_eq!(callee.op_live, parent.op_live);
            assert_eq!(callee.descrs.len(), parent.descrs.len());
            assert_eq!(callee.virtualizable_ptr, parent.virtualizable_ptr);
            assert_eq!(
                callee.virtualizable_stack_base,
                parent.virtualizable_stack_base
            );
            assert_eq!(callee.jitdrivers_sd.len(), parent.jitdrivers_sd.len());
        }

        /// Slice 3.2.0: `handler_abort_marker_pyre` mirrors
        /// `dispatch_one::BC_ABORT` arm semantics
        /// (blackhole.rs:2284-2287).  Reaching `OpKind::Abort` at
        /// runtime sets `aborted = true` and exits the frame —
        /// continuing dispatch past it would misread the next bytes
        /// as opcodes.
        #[test]
        fn test_handler_abort_marker_pyre_sets_aborted_and_leaves_frame() {
            let mut builder = BlackholeInterpBuilder::new();
            let mut bh = builder.acquire_interp();
            assert!(!bh.aborted);
            let result = super::handler_abort_marker_pyre(&mut bh, &[], 0);
            match result {
                Err(super::DispatchError::LeaveFrame) => {}
                other => {
                    panic!("handler_abort_marker_pyre must return Err(LeaveFrame), got {other:?}")
                }
            }
            assert!(bh.aborted);
        }

        /// Integration test: build bytecode manually with known opcode
        /// assignments and run it through the orthodox dispatch_loop.
        ///
        /// This validates the Phase D setup_insns → dispatch_loop → bhimpl
        /// pipeline end-to-end without depending on SSARepr assembly.
        ///
        /// RPython equivalent: the setup_insns + dispatch_loop closure + bhimpl
        /// flow described in blackhole.py:52-103 and 452-460.
        #[test]
        fn test_orthodox_dispatch_loop_int_add() {
            // Build a minimal insns dict (as if the assembler had produced it).
            // Opcode 0 = "live/" (liveness marker, skip 2 bytes)
            // Opcode 1 = "int_add/ii>i" (3 register bytes: a, b, dst)
            // Opcode 2 = "int_return/i" (1 register byte)
            let mut insns = HashMap::new();
            insns.insert("live/".to_string(), 0u8);
            insns.insert("int_add/ii>i".to_string(), 1u8);
            insns.insert("int_return/i".to_string(), 2u8);

            // Setup builder
            let mut builder = BlackholeInterpBuilder::new();
            builder.setup_insns(&insns);
            super::wire_bhimpl_handlers(&mut builder);

            // Hand-assemble bytecode: live + int_add(r0, r1) → r2 + int_return(r2)
            let code: Vec<u8> = vec![
                0, 0, 0, // opcode 0 = live/, 2 bytes liveness offset (skipped)
                1, 0, 1, 2, // opcode 1 = int_add, a=r0, b=r1, dst=r2
                2, 2, // opcode 2 = int_return, src=r2
            ];

            // Acquire BlackholeInterpreter from the same builder so the
            // 6 builder-shared fields (op_live etc.) flow through, then
            // size the int register file for r0..=r2.
            let mut bh = builder.acquire_interp();
            bh.registers_i = vec![0i64; 3];
            bh.registers_i[0] = 10; // r0 = 10
            bh.registers_i[1] = 32; // r1 = 32

            // Run dispatch_loop
            let result = builder.dispatch_loop(&mut bh, &code, 0);

            // Should leave frame with LeaveFrame
            assert!(matches!(result, Err(DispatchError::LeaveFrame)));
            // tmpreg_i should hold 10 + 32 = 42
            assert_eq!(bh.tmpreg_i, 42, "int_add(10, 32) should produce 42");
            assert_eq!(bh.return_type, BhReturnType::Int);
        }

        #[test]
        fn wire_bhimpl_handlers_wires_tagged_int_base_access_aliases() {
            // Canonical RPython opnames (`_i`/`_r`/`_f`) emitted by pyre's
            // build-time assembler — including the pyre tagged-int base
            // variant (`/id>X`, `/iXd`, `/iid>X`, `/ird>X`) that carries
            // the base pointer in an int register. Emit side:
            // majit-translate assembler.rs FieldRead/FieldWrite/ArrayRead
            // /VableFieldRead/VableFieldWrite derive the opname kind
            // suffix from the value/result register kind.
            let mut insns = HashMap::new();
            insns.insert("getfield_gc_i/id>i".to_string(), 0u8);
            insns.insert("getfield_gc_r/id>r".to_string(), 1u8);
            insns.insert("setfield_gc_i/iid".to_string(), 2u8);
            insns.insert("setfield_gc_r/ird".to_string(), 3u8);
            insns.insert("getarrayitem_gc_i/iid>i".to_string(), 4u8);
            insns.insert("getarrayitem_gc_i/ird>i".to_string(), 5u8);
            insns.insert("getfield_vable_i/id>i".to_string(), 6u8);
            insns.insert("setfield_vable_i/iid".to_string(), 7u8);
            insns.insert("setfield_vable_r/ird".to_string(), 8u8);

            let mut builder = BlackholeInterpBuilder::new();
            builder.setup_insns(&insns);
            super::wire_bhimpl_handlers(&mut builder);

            let placeholder = super::unwired_handler_placeholder as *const () as usize;
            for slot in 0usize..=8 {
                assert_ne!(
                    builder.dispatch_table[slot] as *const () as usize, placeholder,
                    "slot {slot} must be wired",
                );
            }
        }

        #[test]
        fn wire_bhimpl_handlers_leaves_dead_v_access_aliases_unwired() {
            // Pyre-invented `_v` sentinel forms that no longer leave the
            // assembler (see majit-translate/src/jit_codewriter/
            // assembler.rs:2106-2130,2226-2250 negative asserts). Kept as
            // guard tests so any regression that reintroduces a `_v` key
            // surfaces at setup_insns time rather than at first dispatch.
            let mut insns = HashMap::new();
            insns.insert("setfield_gc_v/rid".to_string(), 0u8);
            insns.insert("setfield_gc_v/iid".to_string(), 1u8);
            insns.insert("setfield_gc_v/ird".to_string(), 2u8);
            insns.insert("setarrayitem_gc_v/riid".to_string(), 3u8);
            insns.insert("setarrayitem_gc_v/iiid".to_string(), 4u8);
            insns.insert("getfield_gc_v/id>i".to_string(), 5u8);
            insns.insert("getarrayitem_gc_v/rid>i".to_string(), 6u8);
            insns.insert("getarrayitem_gc_v/iid>i".to_string(), 7u8);
            insns.insert("getarrayitem_gc_v/ird>i".to_string(), 8u8);
            insns.insert("setfield_vable_v/id".to_string(), 9u8);
            insns.insert("setfield_vable_v/rd".to_string(), 10u8);
            insns.insert("setfield_vable_v/ird".to_string(), 11u8);
            insns.insert("getfield_vable_v/d>i".to_string(), 12u8);
            insns.insert("getfield_vable_v/id>i".to_string(), 13u8);
            insns.insert("input_v/>i".to_string(), 14u8);

            let mut builder = BlackholeInterpBuilder::new();
            builder.setup_insns(&insns);
            super::wire_bhimpl_handlers(&mut builder);

            let placeholder = super::unwired_handler_placeholder as *const () as usize;
            for slot in 0usize..=14 {
                assert_eq!(
                    builder.dispatch_table[slot] as *const () as usize, placeholder,
                    "slot {slot} must stay unwired — pyre `_v` keys are no \
                     longer produced, so re-wiring would normalise a fake \
                     surface back into the dispatch table",
                );
            }
        }

        #[test]
        fn wire_bhimpl_handlers_leaves_mixed_ref_int_int_ops_unwired() {
            // RPython blackhole integer arithmetic is `@arguments("i", "i",
            // returns="i")` (blackhole.py:458+). Ref/int-mixed `int_*`
            // opnames are kind-flow bugs, not alternate blackhole surfaces.
            let mut insns = HashMap::new();
            let fake_opnames = [
                "int_add/ri>i",
                "int_add/ir>i",
                "int_add/rr>i",
                "int_lshift/ri>i",
                "int_rshift/ir>i",
                "int_xor/rr>i",
                "int_same_as/r>i",
                "int_neg/r>i",
                "int_invert/r>i",
            ];
            for (opcode, opname) in fake_opnames.iter().enumerate() {
                insns.insert((*opname).to_string(), opcode as u8);
            }

            let mut builder = BlackholeInterpBuilder::new();
            builder.setup_insns(&insns);
            super::wire_bhimpl_handlers(&mut builder);

            let placeholder = super::unwired_handler_placeholder as *const () as usize;
            for slot in 0usize..fake_opnames.len() {
                assert_eq!(
                    builder.dispatch_table[slot] as *const () as usize, placeholder,
                    "slot {slot} ({}) must stay unwired; mixed ref/int int \
                     opnames should be fixed at the codewriter/rtyper source",
                    fake_opnames[slot],
                );
            }
        }
    }
}

// ── bhimpl_* methods (RPython blackhole.py:452+) ────────────────────
//
// RPython defines each bhimpl_* as a static method decorated with
// @arguments("i", "i", returns="i") etc. The handler closure generated
// by _get_method decodes args from the bytecode stream and calls the
// bhimpl method.
//
// In Rust we define each bhimpl as a standalone fn and generate
// BhOpcodeHandler wrappers. The handler decodes operands, calls the
// bhimpl fn, stores the result, and returns the updated position.

// ── handler generators for common patterns ──────────────────────────

/// Decode pattern `@arguments("i", "i", returns="i")` — argcodes `"ii>i"`.
///
/// Read 2 int-register indices, call bhimpl fn, write result, advance by 3.
/// Each concrete handler is its own fn (not a closure) so it can be stored
/// as a bare BhOpcodeHandler fn pointer.
macro_rules! bhhandler_ii_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            let b = bh.registers_i[code[position + 1] as usize];
            bh.registers_i[code[position + 2] as usize] = $bhimpl(a, b);
            Ok(position + 3)
        }
    };
}

/// Decode pattern `@arguments("i", returns="i")` — argcodes `"i>i"`.
macro_rules! bhhandler_i_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            bh.registers_i[code[position + 1] as usize] = $bhimpl(a);
            Ok(position + 2)
        }
    };
}

// ── bhimpl methods (line-by-line from RPython blackhole.py) ─────────

/// blackhole.py:454-456 `bhimpl_int_same_as`.
fn bhimpl_int_same_as(a: i64) -> i64 {
    a
}

/// blackhole.py:458-460 `bhimpl_int_add(a, b): return intmask(a + b)`.
fn bhimpl_int_add(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}

/// blackhole.py:462-464 `bhimpl_int_sub(a, b): return intmask(a - b)`.
fn bhimpl_int_sub(a: i64, b: i64) -> i64 {
    a.wrapping_sub(b)
}

/// blackhole.py:466-468 `bhimpl_int_mul(a, b): return intmask(a * b)`.
fn bhimpl_int_mul(a: i64, b: i64) -> i64 {
    a.wrapping_mul(b)
}

/// RPython `rint.py:399-408 ll_int_py_div` (oopspec `int.py_div`).
/// The OS_INT_PY_DIV residual call lands here at runtime. The JIT
/// trace contains two explicit guards upstream of this call,
/// produced by the inlined `_ovf_zer` wrapper (`rint.py:429
/// ll_int_py_div_ovf_zer`):
///   * `int_eq(rhs, 0) -> guard_false` (zero divisor),
///   * `int_and(int_eq(lhs, INT_MIN), int_eq(rhs, -1)) ->
///     guard_false` (overflow corner — `INT_MIN // -1` would
///     overflow to `INT_MIN` in two's-complement; PyPy
///     `intobject.py:316/491/804` routes this case through
///     `ovf2long` to long arithmetic).
/// Other negative operand combinations are valid: PyPy
/// `intobject.py:316 _floordiv` only handles `ZeroDivisionError`,
/// and the no-branch floor correction
/// (`(a ^ b) < 0 && d * b != a -> d - 1`) yields Python-floor
/// semantics for every legal sign combination of `(a, b)`.
///
/// `wrapping_div(0)` panics (Rust integer division by zero is always
/// a panic, in both debug and release builds), and
/// `INT_MIN.wrapping_div(-1)` returns `INT_MIN` (wrong but
/// well-defined).  Both corners are unreachable from the JIT trace
/// path: the `_ovf_zer` wrapper's `int_eq(rhs, 0) -> guard_false` and
/// `(lhs == INT_MIN) & (rhs == -1) -> guard_false` runtime guards
/// (emitted at `codegen.rs::generated_binary_int_value`) bail out the
/// trace before this helper is invoked, matching RPython's
/// `rint.py:429 ll_int_py_div_ovf_zer` shape.  Direct (non-traced)
/// callers must respect the same precondition.
pub fn bhimpl_int_floordiv(a: i64, b: i64) -> i64 {
    let d = a.wrapping_div(b);
    if (a ^ b) < 0 && d.wrapping_mul(b) != a {
        d.wrapping_sub(1)
    } else {
        d
    }
}

/// RPython `rint.py:496-500 ll_int_py_mod` (oopspec `int.py_mod`).
/// See [`bhimpl_int_floordiv`] for the JIT-side runtime guard
/// rationale (`int_eq(rhs, 0)` + `(lhs == INT_MIN) & (rhs == -1)`).
/// Uses `wrapping_rem` for the C-style remainder step, then applies
/// RPython `ll_int_py_mod`'s sign correction only when the remainder
/// and divisor have opposite signs.  As with `bhimpl_int_floordiv`,
/// `wrapping_rem(0)` and `INT_MIN % -1` are unreachable from the
/// trace path; non-traced callers must respect the same precondition.
pub fn bhimpl_int_mod(a: i64, b: i64) -> i64 {
    let r = a.wrapping_rem(b);
    if r != 0 && (r ^ b) < 0 {
        r.wrapping_add(b)
    } else {
        r
    }
}

/// blackhole.py:499-501 `bhimpl_int_and(a, b): return a & b`.
fn bhimpl_int_and(a: i64, b: i64) -> i64 {
    a & b
}

/// blackhole.py:503-505 `bhimpl_int_or(a, b): return a | b`.
fn bhimpl_int_or(a: i64, b: i64) -> i64 {
    a | b
}

/// blackhole.py:507-509 `bhimpl_int_xor(a, b): return a ^ b`.
fn bhimpl_int_xor(a: i64, b: i64) -> i64 {
    a ^ b
}

/// blackhole.py:511-513 `bhimpl_int_rshift(a, b): return a >> b`.
fn bhimpl_int_rshift(a: i64, b: i64) -> i64 {
    a >> (b & 63)
}

/// blackhole.py:516-518 `bhimpl_int_lshift(a, b): return intmask(a << b)`.
fn bhimpl_int_lshift(a: i64, b: i64) -> i64 {
    a.wrapping_shl((b & 63) as u32)
}

/// blackhole.py:521-524 `bhimpl_uint_rshift(a, b): return intmask(r_uint(a) >> r_uint(b))`.
fn bhimpl_uint_rshift(a: i64, b: i64) -> i64 {
    ((a as u64) >> ((b as u64) & 63)) as i64
}

/// blackhole.py:527-529 `bhimpl_int_neg(a): return intmask(-a)`.
fn bhimpl_int_neg(a: i64) -> i64 {
    a.wrapping_neg()
}

/// blackhole.py:531-533 `bhimpl_int_invert(a): return ~a`.
fn bhimpl_int_invert(a: i64) -> i64 {
    !a
}

/// blackhole.py:535 `bhimpl_int_lt(a, b): return int(a < b)`.
fn bhimpl_int_lt(a: i64, b: i64) -> i64 {
    (a < b) as i64
}

/// blackhole.py:539 `bhimpl_int_le(a, b): return int(a <= b)`.
fn bhimpl_int_le(a: i64, b: i64) -> i64 {
    (a <= b) as i64
}

/// blackhole.py:543 `bhimpl_int_eq(a, b): return int(a == b)`.
fn bhimpl_int_eq(a: i64, b: i64) -> i64 {
    (a == b) as i64
}

/// blackhole.py:547 `bhimpl_int_ne(a, b): return int(a != b)`.
fn bhimpl_int_ne(a: i64, b: i64) -> i64 {
    (a != b) as i64
}

/// blackhole.py:551 `bhimpl_int_gt(a, b): return int(a > b)`.
fn bhimpl_int_gt(a: i64, b: i64) -> i64 {
    (a > b) as i64
}

/// blackhole.py:555 `bhimpl_int_ge(a, b): return int(a >= b)`.
fn bhimpl_int_ge(a: i64, b: i64) -> i64 {
    (a >= b) as i64
}

/// blackhole.py:559 `bhimpl_int_is_true(a): return int(bool(a))`.
fn bhimpl_int_is_true(a: i64) -> i64 {
    (a != 0) as i64
}

/// blackhole.py:563 `bhimpl_int_is_zero(a): return int(not a)`.
fn bhimpl_int_is_zero(a: i64) -> i64 {
    (a == 0) as i64
}

/// blackhole.py:567 `bhimpl_int_force_ge_zero(a): if a < 0: return 0; return a`.
fn bhimpl_int_force_ge_zero(a: i64) -> i64 {
    if a < 0 { 0 } else { a }
}

// Generate handler fns from bhimpl methods via macros.
// @arguments("i", returns="i") → argcodes "i>i" → 1 src reg + 1 dst reg = 2 bytes
bhhandler_i_i!(handler_int_same_as, bhimpl_int_same_as);
bhhandler_i_i!(handler_int_neg, bhimpl_int_neg);
bhhandler_i_i!(handler_int_invert, bhimpl_int_invert);
bhhandler_i_i!(handler_int_is_true, bhimpl_int_is_true);
bhhandler_i_i!(handler_int_is_zero, bhimpl_int_is_zero);
bhhandler_i_i!(handler_int_force_ge_zero, bhimpl_int_force_ge_zero);

// @arguments("i", "i", returns="i") → argcodes "ii>i" → 2 src regs + 1 dst reg = 3 bytes
bhhandler_ii_i!(handler_int_add, bhimpl_int_add);
bhhandler_ii_i!(handler_int_sub, bhimpl_int_sub);

// pyre-only: `a += b` in Rust front-end (`front/ast.rs:1604` maps
// `syn::BinOp::AddAssign` to the BinOp name `"add_assign"`). Under SSA
// the compound assignment becomes two reads + one write, so the
// bytecode shape is identical to `int_add/ii>i` — same primitive. Has
// no RPython analog; canonical RPython lowers `+=` to plain BINARY_ADD
// before reaching jtransform.
bhhandler_ii_i!(handler_int_add_assign_pyre, bhimpl_int_add);
bhhandler_ii_i!(handler_int_sub_assign_pyre, bhimpl_int_sub);

// pyre-only: `*x` in Rust front-end (`front/ast.rs:1577` maps
// `syn::UnOp::Deref` to the UnaryOp name `"deref"`). After rtyper
// lowering the &i64 input has already been resolved to a plain i64
// value, so the operation degenerates to a copy at dispatch time. Same
// primitive as `int_same_as`.
bhhandler_i_i!(handler_int_deref_pyre, bhimpl_int_same_as);

// pyre-only: `OpKind::Abort` placeholder emitted by the front-end when a
// syntax node has no dedicated OpKind variant (assembler.rs
// `OpKind::Abort { .. } => "abort".into()`). Carries zero operand
// bytes (empty argcodes). Reaching here at runtime means a graph that
// made it through rtyper still has an untranslatable op — abort the
// frame rather than continue dispatching past it: subsequent bytes
// would be misread as opcodes. Mirror of dispatch_one::BC_ABORT arm
// (blackhole.rs:2284): `aborted = true` + `LeaveFrame`. RPython has no
// direct analog: its codewriter raises before lowering, so a jitcode
// never carries an unrecognized op.
fn handler_abort_marker_pyre(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    bh.aborted = true;
    Err(DispatchError::LeaveFrame)
}

/// Handler for pyre-only `abort/>i` — a no-op result marker emitted by
/// `Assembler::encode_op`'s default branch for `OpKind::Abort { .. }`.
/// The bytecode layout has no args and a single destination register byte.
fn handler_abort_result_marker_i(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 1)
}

// pyre-only: `state_field` family. RPython has no counterpart — these are
// emitted by `#[jit_interp]`'s `jitcode_lower` macro to express loads/stores
// against a Rust-port `state_fields` register file used during tracing.
// Blackhole has nothing to do for them (the dispatch_one arm at
// blackhole.rs:2505-2524 is also a no-op): the state-field semantics live
// only in the trace path with a `JitCodeSym`. Handlers exist solely so the
// dispatch_table entry is wired (not placeholder), keeping RPython
// `blackhole.py:287` `self.dispatch_loop = builder.dispatch_loop`'s
// table-driven invariant.

/// No-op handler for `load_state_field/di` / `store_state_field/di` —
/// canonical encoding per `assembler.py`: 1× u16 descr + 1× u8 register
/// = 3 bytes total.
fn handler_state_field_noop_di(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 3)
}

/// No-op handler for `load_state_array/dii` / `store_state_array/dii` /
/// `load_state_varray/dii` / `store_state_varray/dii` — canonical
/// encoding: 1× u16 descr + 2× u8 register = 4 bytes total.
fn handler_state_array_noop_dii(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 4)
}

/// Handler for `jit_merge_point/iIRFIRF` — `BC_JIT_MERGE_POINT`.
/// Forwards to `bhimpl_jit_merge_point` which uses `self.position`-mutating
/// helpers; `dispatch_step_with_fallback` already passes `self.position`
/// as the `position` argument so the local copy stays in sync.
fn handler_jit_merge_point_i(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    bh.bhimpl_jit_merge_point(jitcode::BC_JIT_MERGE_POINT)?;
    Ok(bh.position)
}

/// Handler for `jit_merge_point/cIRFIRF` — `BC_JIT_MERGE_POINT_C`
/// (assembler.py:312 `USE_C_FORM`: jdindex inlined as a signed byte).
fn handler_jit_merge_point_c(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    bh.bhimpl_jit_merge_point(jitcode::BC_JIT_MERGE_POINT_C)?;
    Ok(bh.position)
}

/// Handler for `abort_permanent/` — pyre-only fail-path opcode that
/// forwards a TLS-stashed exception or aborts the frame. Carries no
/// operand bytes.
fn handler_abort_permanent(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    bh.bhimpl_abort_permanent()?;
    Ok(bh.position)
}

/// Handler for pyre-only `abort/>r` — counterpart of
/// `abort/>i` with a Ref-classified result register.  Emerges when
/// pyre's rtyper routes an untranslatable op's result through the
/// Abort→GcRef fallback (`rtyper.rs::infer_concrete_from_op`).
fn handler_abort_result_marker_r(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 1)
}
bhhandler_ii_i!(handler_int_mul, bhimpl_int_mul);
bhhandler_ii_i!(handler_int_floordiv, bhimpl_int_floordiv);
bhhandler_ii_i!(handler_int_mod, bhimpl_int_mod);
bhhandler_ii_i!(handler_int_and, bhimpl_int_and);
bhhandler_ii_i!(handler_int_or, bhimpl_int_or);
bhhandler_ii_i!(handler_int_xor, bhimpl_int_xor);
bhhandler_ii_i!(handler_int_rshift, bhimpl_int_rshift);
bhhandler_ii_i!(handler_int_lshift, bhimpl_int_lshift);
bhhandler_ii_i!(handler_uint_rshift, bhimpl_uint_rshift);
bhhandler_ii_i!(handler_int_lt, bhimpl_int_lt);
bhhandler_ii_i!(handler_int_le, bhimpl_int_le);
bhhandler_ii_i!(handler_int_eq, bhimpl_int_eq);
bhhandler_ii_i!(handler_int_ne, bhimpl_int_ne);
bhhandler_ii_i!(handler_int_gt, bhimpl_int_gt);
bhhandler_ii_i!(handler_int_ge, bhimpl_int_ge);

// ── control flow + copy handlers ─────────────────────────────────────

// blackhole.py:638-640 `bhimpl_int_copy(a): return a` — @arguments("i", returns="i").
// Decoded as `i>i` (same as int_same_as). Already have handler_int_same_as.
// Wire as alias.
bhhandler_i_i!(handler_int_copy, bhimpl_int_same_as);

/// Handler for `live/` — liveness marker. Argcodes: empty, but the assembler
/// emits a 2-byte offset after the opcode. Skip those 2 bytes.
/// RPython blackhole.py:146-158 (inside _get_method for `-live-` ops).
fn handler_live(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // Skip the 2-byte liveness offset (RPython: OFFSET_SIZE = 2).
    Ok(position + 2)
}

/// Handler for `goto/L` — unconditional jump. Argcodes: `L` (2-byte label).
/// RPython blackhole.py:950-952: `def bhimpl_goto(target): return target`.
/// Returns="L" means the handler returns the label value as new position.
fn handler_goto(
    _bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let target = (code[position] as usize) | ((code[position + 1] as usize) << 8);
    Ok(target)
}

/// Handler for `goto_if_not/iL` — conditional jump.
/// RPython blackhole.py:864-869:
/// ```python
/// @arguments("i", "L", "pc", returns="L")
/// def bhimpl_goto_if_not(a, target, pc):
///     if a: return pc
///     else: return target
/// ```
/// "pc" means the current position AFTER decoding all operands.
fn handler_goto_if_not(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[position] as usize];
    let target = (code[position + 1] as usize) | ((code[position + 2] as usize) << 8);
    let pc = position + 3; // position after all operands
    if a != 0 {
        Ok(pc) // condition true: fall through
    } else {
        Ok(target) // condition false: jump to target
    }
}

/// Handler for `int_return/i` — RPython blackhole.py:841-845.
/// @arguments("self", "i"): read one int register, store in tmpreg_i,
/// raise LeaveFrame.
fn handler_int_return(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[position] as usize];
    bh.tmpreg_i = a;
    bh.return_type = BhReturnType::Int;
    // RPython blackhole.py:169 `_get_method` stores the decoded position
    // back into `self.position` before invoking the bhimpl_*; this is
    // visible after a LeaveFrame since the frame teardown reads the
    // post-operand position.
    bh.position = position + 1;
    Err(DispatchError::LeaveFrame)
}

/// Handler for `ref_return/r` — RPython blackhole.py:847-851.
fn handler_ref_return(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_r[code[position] as usize];
    bh.tmpreg_r = a;
    bh.return_type = BhReturnType::Ref;
    bh.position = position + 1; // blackhole.py:169 parity (see int_return).
    Err(DispatchError::LeaveFrame)
}

/// Handler for `void_return/` — RPython blackhole.py:859-862.
fn handler_void_return(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    bh.return_type = BhReturnType::Void;
    Err(DispatchError::LeaveFrame)
}

// ── float bhimpl methods (RPython blackhole.py:676-808) ─────────────

// RPython stores floats as longlong (i64 bits). pyre stores f64 in
// registers_f directly. The bhimpl methods work on f64 values.

fn bhimpl_float_neg(a: f64) -> f64 {
    -a
}
fn bhimpl_float_abs(a: f64) -> f64 {
    a.abs()
}
fn bhimpl_float_add(a: f64, b: f64) -> f64 {
    a + b
}
fn bhimpl_float_sub(a: f64, b: f64) -> f64 {
    a - b
}
fn bhimpl_float_mul(a: f64, b: f64) -> f64 {
    a * b
}
fn bhimpl_float_truediv(a: f64, b: f64) -> f64 {
    a / b
}

/// Decode pattern `@arguments("f", "f", returns="f")` — argcodes `"ff>f"`.
macro_rules! bhhandler_ff_f {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
            let b = f64::from_bits(bh.registers_f[code[position + 1] as usize] as u64);
            bh.registers_f[code[position + 2] as usize] = $bhimpl(a, b).to_bits() as i64;
            Ok(position + 3)
        }
    };
}

/// Decode pattern `@arguments("f", returns="f")` — argcodes `"f>f"`.
macro_rules! bhhandler_f_f {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
            bh.registers_f[code[position + 1] as usize] = $bhimpl(a).to_bits() as i64;
            Ok(position + 2)
        }
    };
}

/// Decode pattern `@arguments("f", "f", returns="i")` — argcodes `"ff>i"`.
macro_rules! bhhandler_ff_i {
    ($name:ident, $cmp:expr) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
            let b = f64::from_bits(bh.registers_f[code[position + 1] as usize] as u64);
            bh.registers_i[code[position + 2] as usize] = $cmp(a, b) as i64;
            Ok(position + 3)
        }
    };
}

bhhandler_ff_f!(handler_float_add, bhimpl_float_add);
bhhandler_ff_f!(handler_float_sub, bhimpl_float_sub);
bhhandler_ff_f!(handler_float_mul, bhimpl_float_mul);
bhhandler_ff_f!(handler_float_truediv, bhimpl_float_truediv);
bhhandler_f_f!(handler_float_neg, bhimpl_float_neg);
bhhandler_f_f!(handler_float_abs, bhimpl_float_abs);
bhhandler_ff_i!(handler_float_lt, |a: f64, b: f64| a < b);
bhhandler_ff_i!(handler_float_le, |a: f64, b: f64| a <= b);
bhhandler_ff_i!(handler_float_eq, |a: f64, b: f64| a == b);
bhhandler_ff_i!(handler_float_ne, |a: f64, b: f64| a != b);
bhhandler_ff_i!(handler_float_gt, |a: f64, b: f64| a > b);
bhhandler_ff_i!(handler_float_ge, |a: f64, b: f64| a >= b);

// ── unsigned comparison bhimpl (RPython blackhole.py:571-582) ────────

fn bhimpl_uint_lt(a: i64, b: i64) -> i64 {
    ((a as u64) < (b as u64)) as i64
}
fn bhimpl_uint_le(a: i64, b: i64) -> i64 {
    ((a as u64) <= (b as u64)) as i64
}
fn bhimpl_uint_gt(a: i64, b: i64) -> i64 {
    ((a as u64) > (b as u64)) as i64
}
fn bhimpl_uint_ge(a: i64, b: i64) -> i64 {
    ((a as u64) >= (b as u64)) as i64
}

bhhandler_ii_i!(handler_uint_lt, bhimpl_uint_lt);
bhhandler_ii_i!(handler_uint_le, bhimpl_uint_le);
bhhandler_ii_i!(handler_uint_gt, bhimpl_uint_gt);
bhhandler_ii_i!(handler_uint_ge, bhimpl_uint_ge);

// ── goto_if_not_int_* conditionals (RPython blackhole.py:871-920) ───

/// Decode pattern `@arguments("i", "i", "L", "pc", returns="L")`.
/// Read 2 int registers + 2-byte label; compare; return target or pc.
macro_rules! bhhandler_goto_if_not_ii {
    ($name:ident, $cmp:expr) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            let b = bh.registers_i[code[position + 1] as usize];
            let target = (code[position + 2] as usize) | ((code[position + 3] as usize) << 8);
            let pc = position + 4;
            if $cmp(a, b) { Ok(pc) } else { Ok(target) }
        }
    };
}

bhhandler_goto_if_not_ii!(handler_goto_if_not_int_lt, |a: i64, b: i64| a < b);
bhhandler_goto_if_not_ii!(handler_goto_if_not_int_le, |a: i64, b: i64| a <= b);
bhhandler_goto_if_not_ii!(handler_goto_if_not_int_eq, |a: i64, b: i64| a == b);
bhhandler_goto_if_not_ii!(handler_goto_if_not_int_ne, |a: i64, b: i64| a != b);
bhhandler_goto_if_not_ii!(handler_goto_if_not_int_gt, |a: i64, b: i64| a > b);
bhhandler_goto_if_not_ii!(handler_goto_if_not_int_ge, |a: i64, b: i64| a >= b);

// ── ref operations (RPython blackhole.py:584-610) ───────────────────

fn bhimpl_ptr_eq(a: i64, b: i64) -> i64 {
    (a == b) as i64
}
fn bhimpl_ptr_ne(a: i64, b: i64) -> i64 {
    (a != b) as i64
}
fn bhimpl_ptr_iszero(a: i64) -> i64 {
    (a == 0) as i64
}
fn bhimpl_ptr_nonzero(a: i64) -> i64 {
    (a != 0) as i64
}

/// `@arguments("r", "r", returns="i")` — rr>i: read 2 ref regs, result int.
macro_rules! bhhandler_rr_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_r[code[position] as usize];
            let b = bh.registers_r[code[position + 1] as usize];
            bh.registers_i[code[position + 2] as usize] = $bhimpl(a, b);
            Ok(position + 3)
        }
    };
}

/// `@arguments("r", returns="i")` — r>i: read 1 ref reg, result int.
macro_rules! bhhandler_r_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_r[code[position] as usize];
            bh.registers_i[code[position + 1] as usize] = $bhimpl(a);
            Ok(position + 2)
        }
    };
}

bhhandler_rr_i!(handler_ptr_eq, bhimpl_ptr_eq);
bhhandler_rr_i!(handler_ptr_ne, bhimpl_ptr_ne);
bhhandler_rr_i!(handler_instance_ptr_eq, bhimpl_ptr_eq);
bhhandler_rr_i!(handler_instance_ptr_ne, bhimpl_ptr_ne);
bhhandler_r_i!(handler_ptr_iszero, bhimpl_ptr_iszero);
bhhandler_r_i!(handler_ptr_nonzero, bhimpl_ptr_nonzero);

// ref/float copy (blackhole.py:641-645)
fn handler_ref_copy(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_r[code[position + 1] as usize] = bh.registers_r[code[position] as usize];
    Ok(position + 2)
}
fn handler_float_copy(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_f[code[position + 1] as usize] = bh.registers_f[code[position] as usize];
    Ok(position + 2)
}

// float_return (blackhole.py:853-857)
fn handler_float_return(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.tmpreg_f = bh.registers_f[code[position] as usize];
    bh.return_type = BhReturnType::Float;
    bh.position = position + 1; // blackhole.py:169 parity (see int_return).
    Err(DispatchError::LeaveFrame)
}

// ── guard_value — no-op in blackhole (blackhole.py:648-656) ─────────
fn handler_int_guard_value(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 1)
}
fn handler_ref_guard_value(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 1)
}
fn handler_float_guard_value(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 1)
}

// ── push/pop (blackhole.py:661-679) ─────────────────────────────────
fn handler_int_push(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.tmpreg_i = bh.registers_i[code[position] as usize];
    Ok(position + 1)
}
fn handler_ref_push(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.tmpreg_r = bh.registers_r[code[position] as usize];
    Ok(position + 1)
}
fn handler_float_push(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.tmpreg_f = bh.registers_f[code[position] as usize];
    Ok(position + 1)
}
fn handler_int_pop(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[position] as usize] = bh.tmpreg_i;
    Ok(position + 1)
}
fn handler_ref_pop(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_r[code[position] as usize] = bh.tmpreg_r;
    Ok(position + 1)
}
fn handler_float_pop(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_f[code[position] as usize] = bh.tmpreg_f;
    Ok(position + 1)
}

// ── record_exact_class/value — no-op (blackhole.py:616-636) ─────────
fn handler_record_exact_class(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 2)
}
fn handler_record_exact_value_r(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 2)
}
fn handler_record_exact_value_i(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 2)
}

// ── cast operations (blackhole.py:800-831) ──────────────────────────
fn handler_cast_float_to_int(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
    bh.registers_i[code[position + 1] as usize] = a as i64;
    Ok(position + 2)
}
fn handler_cast_int_to_float(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[position] as usize];
    bh.registers_f[code[position + 1] as usize] = (a as f64).to_bits() as i64;
    Ok(position + 2)
}

// ── int_signext (blackhole.py:566-569) ──────────────────────────────
fn handler_int_signext(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[position] as usize];
    let numbytes = bh.registers_i[code[position + 1] as usize];
    let result = match numbytes {
        1 => (a as i8) as i64,
        2 => (a as i16) as i64,
        4 => (a as i32) as i64,
        _ => a,
    };
    bh.registers_i[code[position + 2] as usize] = result;
    Ok(position + 3)
}

// ── overflow ops (blackhole.py:478-497) ─────────────────────────────

fn handler_int_add_jump_if_ovf(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let target = (code[position] as usize) | ((code[position + 1] as usize) << 8);
    let a = bh.registers_i[code[position + 2] as usize];
    let b = bh.registers_i[code[position + 3] as usize];
    let pc = position + 5;
    match a.checked_add(b) {
        Some(result) => {
            bh.registers_i[code[position + 4] as usize] = result;
            Ok(pc)
        }
        None => Ok(target),
    }
}
fn handler_int_sub_jump_if_ovf(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let target = (code[position] as usize) | ((code[position + 1] as usize) << 8);
    let a = bh.registers_i[code[position + 2] as usize];
    let b = bh.registers_i[code[position + 3] as usize];
    let pc = position + 5;
    match a.checked_sub(b) {
        Some(result) => {
            bh.registers_i[code[position + 4] as usize] = result;
            Ok(pc)
        }
        None => Ok(target),
    }
}
fn handler_int_mul_jump_if_ovf(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let target = (code[position] as usize) | ((code[position + 1] as usize) << 8);
    let a = bh.registers_i[code[position + 2] as usize];
    let b = bh.registers_i[code[position + 3] as usize];
    let pc = position + 5;
    match a.checked_mul(b) {
        Some(result) => {
            bh.registers_i[code[position + 4] as usize] = result;
            Ok(pc)
        }
        None => Ok(target),
    }
}

// ── misc simple ops ─────────────────────────────────────────────────

fn handler_assert_not_none(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 1)
}

fn handler_virtual_ref(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_r[code[position + 1] as usize] = bh.registers_r[code[position] as usize];
    Ok(position + 2)
}
fn handler_virtual_ref_finish(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 1)
}
fn handler_loop_header(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // blackhole.py:1062-1064 bhimpl_loop_header(jdindex): no-op.
    // Advance past the 1-byte jdindex operand.
    Ok(position + 1)
}
fn handler_ref_isconstant(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[position + 1] as usize] = 0;
    Ok(position + 2)
}
fn handler_ref_isvirtual(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[position + 1] as usize] = 0;
    Ok(position + 2)
}
fn handler_goto_if_not_int_is_zero(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[position] as usize];
    let target = (code[position + 1] as usize) | ((code[position + 2] as usize) << 8);
    let pc = position + 3;
    if a == 0 { Ok(pc) } else { Ok(target) }
}
fn handler_goto_if_not_ptr_iszero(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_r[code[position] as usize];
    let target = (code[position + 1] as usize) | ((code[position + 2] as usize) << 8);
    let pc = position + 3;
    if a == 0 { Ok(pc) } else { Ok(target) }
}
fn handler_goto_if_not_ptr_nonzero(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_r[code[position] as usize];
    let target = (code[position + 1] as usize) | ((code[position + 2] as usize) << 8);
    let pc = position + 3;
    if a != 0 { Ok(pc) } else { Ok(target) }
}
fn handler_unreachable(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    panic!("bhimpl_unreachable reached");
}

// ── cpu-dependent field/array operations ─────────────────────────────
//
// RPython blackhole.py:1432-1481: bhimpl_getfield_gc_*/setfield_gc_*
// These call `cpu.bh_getfield_gc_i(struct_ptr, descr)` etc.
// The 'd' argcode is a 2-byte descriptor index into `bh.descrs`.
// In pyre, descrs[index] resolves to a field offset (usize).

/// RPython `blackhole.py:150-157`: read a 2-byte descriptor index from
/// bytecode and return `(descr_object, new_position)`.
///
/// In RPython: `value = self.descrs[index]`. In pyre: returns the
/// `BhDescr` enum variant.
#[inline]
fn read_descr<'a>(bh: &'a BlackholeInterpreter, code: &[u8], pos: usize) -> (&'a BhDescr, usize) {
    let descr_idx = (code[pos] as usize) | ((code[pos + 1] as usize) << 8);
    if let Some(entry) = bh.jitcode.exec.descrs.get(descr_idx) {
        let descr = entry.as_bh_descr().unwrap_or_else(|| {
            panic!("d-arg descrs[{descr_idx}] is not a BhDescr entry: {entry:?}")
        });
        return (descr, pos + 2);
    }
    let descr = &bh.descrs[descr_idx]; // RPython: no fallback, index must be valid
    (descr, pos + 2)
}

/// Read a VableField descriptor and resolve to a synthesized BhDescr::Field
/// with the resolved byte offset via VirtualizableInfo.
/// RPython: fielddescr carries byte offset directly; pyre VableField.index
/// needs vinfo.static_fields[index].offset resolution.
#[inline]
fn read_descr_vable_field(bh: &BlackholeInterpreter, code: &[u8], pos: usize) -> (BhDescr, usize) {
    let (descr, pos) = read_descr(bh, code, pos);
    let field_index = descr.as_vable_field_index();
    // Resolve field_index → byte offset via VirtualizableInfo.
    let offset = if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        vinfo
            .static_fields
            .get(field_index)
            .map(|f| f.offset)
            .unwrap_or(field_index)
    } else {
        field_index
    };
    (
        BhDescr::Field {
            offset,
            field_size: 8,
            field_type: majit_ir::value::Type::Ref,
            field_flag: majit_ir::descr::ArrayFlag::Pointer,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: 0,
            parent: None,
            name: String::new(),
            owner: String::new(),
        },
        pos,
    )
}

/// Read a VableArray descriptor and resolve to a synthesized BhDescr::Field
/// with the resolved field_offset via VirtualizableInfo.
/// RPython: fielddescr carries byte offset for the array pointer field.
/// pyre: VableArray.index → vinfo.array_fields[index].field_offset.
#[inline]
fn read_descr_vable_array(bh: &BlackholeInterpreter, code: &[u8], pos: usize) -> (BhDescr, usize) {
    let (descr, pos) = read_descr(bh, code, pos);
    let array_index = descr.as_vable_array_index();
    let offset = if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        vinfo
            .array_fields
            .get(array_index)
            .map(|a| a.field_offset)
            .unwrap_or(array_index)
    } else {
        array_index
    };
    (
        BhDescr::Field {
            offset,
            field_size: 8,
            field_type: majit_ir::value::Type::Ref,
            field_flag: majit_ir::descr::ArrayFlag::Pointer,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: 0,
            parent: None,
            name: String::new(),
            owner: String::new(),
        },
        pos,
    )
}

// bhimpl_getfield_gc_i: @arguments("cpu", "r", "d", returns="i")
fn handler_getfield_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    let result = cpu.bh_getfield_gc_i(struct_ptr, descr);
    bh.registers_i[code[pos] as usize] = result;
    Ok(pos + 1)
}
fn handler_getfield_gc_i_intbase(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    let result = cpu.bh_getfield_gc_i(struct_ptr, descr);
    bh.registers_i[code[pos] as usize] = result;
    Ok(pos + 1)
}
fn handler_getfield_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    let result = cpu.bh_getfield_gc_r(struct_ptr, descr);
    bh.registers_r[code[pos] as usize] = result.0 as i64;
    Ok(pos + 1)
}
fn handler_getfield_gc_r_intbase(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    let result = cpu.bh_getfield_gc_r(struct_ptr, descr);
    bh.registers_r[code[pos] as usize] = result.0 as i64;
    Ok(pos + 1)
}
fn handler_getfield_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    let result = cpu.bh_getfield_gc_f(struct_ptr, descr);
    bh.registers_f[code[pos] as usize] = result.to_bits() as i64;
    Ok(pos + 1)
}
fn handler_getfield_gc_f_intbase(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    let result = cpu.bh_getfield_gc_f(struct_ptr, descr);
    bh.registers_f[code[pos] as usize] = result.to_bits() as i64;
    Ok(pos + 1)
}

// bhimpl_setfield_gc_i: @arguments("cpu", "r", "i", "d")
fn handler_setfield_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let value = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_i(struct_ptr, value, descr);
    Ok(pos)
}
fn handler_setfield_gc_i_intbase(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize];
    let value = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_i(struct_ptr, value, descr);
    Ok(pos)
}
fn handler_setfield_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let value = bh.registers_r[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_r(struct_ptr, majit_ir::GcRef(value as usize), descr);
    Ok(pos)
}
fn handler_setfield_gc_r_intbase(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize];
    let value = bh.registers_r[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_r(struct_ptr, majit_ir::GcRef(value as usize), descr);
    Ok(pos)
}
fn handler_setfield_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let value = f64::from_bits(bh.registers_f[code[position + 1] as usize] as u64);
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_f(struct_ptr, value, descr);
    Ok(pos)
}

// bhimpl_arraylen_gc: @arguments("cpu", "r", "d", returns="i")
fn handler_arraylen_gc(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array_ptr = bh.registers_r[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    let result = cpu.bh_arraylen_gc(array_ptr, descr);
    bh.registers_i[code[pos] as usize] = result;
    Ok(pos + 1)
}

// ── getarrayitem_gc (blackhole.py:1329-1341) ────────────────────────
// @arguments("cpu", "r", "i", "d", returns="X")

fn handler_getarrayitem_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[pos] as usize] = cpu.bh_getarrayitem_gc_i(array, index, descr);
    Ok(pos + 1)
}
fn handler_getarrayitem_gc_i_intbase(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_i[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[pos] as usize] = cpu.bh_getarrayitem_gc_i(array, index, descr);
    Ok(pos + 1)
}
fn handler_getarrayitem_gc_i_intbase_refindex(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_i[code[position] as usize];
    let index = bh.registers_r[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[pos] as usize] = cpu.bh_getarrayitem_gc_i(array, index, descr);
    Ok(pos + 1)
}
fn handler_getarrayitem_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[pos] as usize] = cpu.bh_getarrayitem_gc_r(array, index, descr).0 as i64;
    Ok(pos + 1)
}

// pyre-only `getarrayitem_gc_r/rrd>r`: index lands in a Ref register
// (tagged-int-in-ref deviation — same root as `/rr>i` arithmetic
// aliases). `registers_r`/`registers_i` both `Vec<i64>` so the raw
// bits read back as the intended integer index. Disappears once
// rtyper classifies integer array indices as `Signed`.
fn handler_getarrayitem_gc_r_refindex(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_r[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[pos] as usize] = cpu.bh_getarrayitem_gc_r(array, index, descr).0 as i64;
    Ok(pos + 1)
}

// ── setarrayitem_gc (blackhole.py:1350-1358) ────────────────────────
// @arguments("cpu", "r", "i", "X", "d")

fn handler_setarrayitem_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = bh.registers_i[code[position + 2] as usize];
    let (descr, pos) = read_descr(bh, code, position + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setarrayitem_gc_i(array, index, value, descr);
    Ok(pos)
}
fn handler_setarrayitem_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = bh.registers_r[code[position + 2] as usize];
    let (descr, pos) = read_descr(bh, code, position + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setarrayitem_gc_r(array, index, majit_ir::GcRef(value as usize), descr);
    Ok(pos)
}

// pyre-only `setarrayitem_gc_r/rrrd`: index lands in a Ref register
// (same rtyper coverage gap as `getarrayitem_gc_r/rrd>r`). Identical
// body to `handler_setarrayitem_gc_r` except index is read from
// `registers_r` instead of `registers_i`.
fn handler_setarrayitem_gc_r_refindex(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_r[code[position + 1] as usize];
    let value = bh.registers_r[code[position + 2] as usize];
    let (descr, pos) = read_descr(bh, code, position + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setarrayitem_gc_r(array, index, majit_ir::GcRef(value as usize), descr);
    Ok(pos)
}

// ── getfield_raw (blackhole.py:1464-1472) ───────────────────────────
fn handler_getfield_raw_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize]; // raw ptr is int
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[pos] as usize] = cpu.bh_getfield_raw_i(struct_ptr, descr);
    Ok(pos + 1)
}
fn handler_getfield_raw_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[pos] as usize] = cpu.bh_getfield_raw_f(struct_ptr, descr).to_bits() as i64;
    Ok(pos + 1)
}

// ── setfield_raw (blackhole.py:1497-1502) ───────────────────────────
fn handler_setfield_raw_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize];
    let value = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_raw_i(struct_ptr, value, descr);
    Ok(pos)
}
fn handler_setfield_raw_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize];
    let value = f64::from_bits(bh.registers_f[code[position + 1] as usize] as u64);
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_raw_f(struct_ptr, value, descr);
    Ok(pos)
}

// ── new / new_with_vtable / new_array (blackhole.py:1301-1327) ──────
// These need SizeDescr which pyre doesn't fully have yet.
// Stub handlers that read the descriptor and call cpu.bh_new etc.

fn handler_new(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "d", returns="r")
    let (descr, pos) = read_descr(bh, code, position);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[pos] as usize] = cpu.bh_new(descr);
    Ok(pos + 1)
}
fn handler_new_with_vtable(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let (descr, pos) = read_descr(bh, code, position);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[pos] as usize] = cpu.bh_new_with_vtable(descr);
    Ok(pos + 1)
}
fn handler_new_array(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "i", "d", returns="r")
    let length = bh.registers_i[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[pos] as usize] = cpu.bh_new_array(length, descr);
    Ok(pos + 1)
}
fn handler_new_array_clear(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let length = bh.registers_i[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[pos] as usize] = cpu.bh_new_array_clear(length, descr);
    Ok(pos + 1)
}

// ── string operations (blackhole.py:1200-1283) ──────────────────────
fn handler_strlen(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[position + 1] as usize] = cpu.bh_strlen(s);
    Ok(position + 2)
}
fn handler_strgetitem(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[position + 2] as usize] = cpu.bh_strgetitem(s, index);
    Ok(position + 3)
}
fn handler_strsetitem(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = bh.registers_i[code[position + 2] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_strsetitem(s, index, value);
    Ok(position + 3)
}
fn handler_newstr(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let length = bh.registers_i[code[position] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[position + 1] as usize] = cpu.bh_newstr(length);
    Ok(position + 2)
}
fn handler_unicodelen(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[position + 1] as usize] = cpu.bh_unicodelen(s);
    Ok(position + 2)
}
fn handler_unicodegetitem(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[position + 2] as usize] = cpu.bh_unicodegetitem(s, index);
    Ok(position + 3)
}
fn handler_unicodesetitem(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = bh.registers_i[code[position + 2] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_unicodesetitem(s, index, value);
    Ok(position + 3)
}
fn handler_newunicode(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let length = bh.registers_i[code[position] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[position + 1] as usize] = cpu.bh_newunicode(length);
    Ok(position + 2)
}

// ── exception handling (blackhole.py:969-1009) ──────────────────────
fn handler_catch_exception(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // @arguments("L") — no-op, skip 2-byte label
    Ok(position + 2)
}

// ── misc no-ops (blackhole.py:1017-1049) ────────────────────────────
fn handler_jit_debug(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // @arguments("r", "i", "i", "i", "i") = 1 ref + 4 int = 5 regs
    Ok(position + 5)
}
fn handler_jit_enter_portal_frame(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // @arguments("i") = 1 int
    Ok(position + 1)
}
fn handler_jit_leave_portal_frame(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position)
}

// ── interiorfield_gc (blackhole.py:1411-1429) ───────────────────────
// @arguments("cpu", "r", "i", "d", returns="X")
fn handler_getinteriorfield_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[pos] as usize] = cpu.bh_getinteriorfield_gc_i(array, index, descr);
    Ok(pos + 1)
}
fn handler_setinteriorfield_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = bh.registers_i[code[position + 2] as usize];
    let (descr, pos) = read_descr(bh, code, position + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setinteriorfield_gc_i(array, index, value, descr);
    Ok(pos)
}

// ── call operations (blackhole.py:1224-1276) ────────────────────────

#[inline]
fn read_list_i(bh: &BlackholeInterpreter, code: &[u8], pos: usize) -> (Vec<i64>, usize) {
    let count = code[pos] as usize;
    let values: Vec<i64> = (0..count)
        .map(|i| bh.registers_i[code[pos + 1 + i] as usize])
        .collect();
    (values, pos + 1 + count)
}
#[inline]
fn read_list_r(bh: &BlackholeInterpreter, code: &[u8], pos: usize) -> (Vec<i64>, usize) {
    let count = code[pos] as usize;
    let values: Vec<i64> = (0..count)
        .map(|i| bh.registers_r[code[pos + 1 + i] as usize])
        .collect();
    (values, pos + 1 + count)
}
#[inline]
fn read_list_f(bh: &BlackholeInterpreter, code: &[u8], pos: usize) -> (Vec<i64>, usize) {
    let count = code[pos] as usize;
    let values: Vec<i64> = (0..count)
        .map(|i| bh.registers_f[code[pos + 1 + i] as usize])
        .collect();
    (values, pos + 1 + count)
}

/// blackhole.py:351-360 `BlackholeInterpreter.run` exception path: after a
/// residual call, route a non-zero `BH_LAST_EXC_VALUE` either into the next
/// `catch_exception` handler in the current frame (returning `Ok(target)`),
/// or out of the frame as `LeaveFrame` so the outer `run()` propagates.
///
/// Mirrors the legacy direct-dispatch arms (`blackhole.rs:2400-2470`,
/// `2823-2925`) which already perform the same handshake; the wired
/// `dispatch_table` handlers must do the same so the table path does not
/// silently swallow exceptions raised by `cpu.bh_call_*`.
#[inline]
fn check_residual_call_exception_after(
    bh: &mut BlackholeInterpreter,
) -> Result<Option<usize>, DispatchError> {
    let exc_val = BH_LAST_EXC_VALUE.with(|c| c.get());
    if exc_val == 0 {
        return Ok(None);
    }
    if bh.handle_exception_in_frame(exc_val) {
        return Ok(Some(bh.position));
    }
    bh.exception_last_value = exc_val;
    bh.got_exception = true;
    Err(DispatchError::LeaveFrame)
}

// residual_call_irf_*
fn handler_residual_call_irf_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    let dst = code[p] as usize;
    // blackhole.py:1244-1246 → bhimpl_residual_call_irf_i.
    BH_LAST_EXC_VALUE.with(|c| c.set(0));
    let result = bh.bhimpl_residual_call_irf_i(func, &ai, &ar, &af, &calldescr);
    if let Some(handler_pc) = check_residual_call_exception_after(bh)? {
        return Ok(handler_pc);
    }
    bh.registers_i[dst] = result;
    Ok(p + 1)
}
fn handler_residual_call_irf_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    let dst = code[p] as usize;
    // blackhole.py:1247-1249 → bhimpl_residual_call_irf_r.
    BH_LAST_EXC_VALUE.with(|c| c.set(0));
    let result = bh.bhimpl_residual_call_irf_r(func, &ai, &ar, &af, &calldescr);
    if let Some(handler_pc) = check_residual_call_exception_after(bh)? {
        return Ok(handler_pc);
    }
    bh.registers_r[dst] = result.0 as i64;
    Ok(p + 1)
}
fn handler_residual_call_irf_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    let dst = code[p] as usize;
    // blackhole.py:1250-1252 → bhimpl_residual_call_irf_f.
    BH_LAST_EXC_VALUE.with(|c| c.set(0));
    let result = bh.bhimpl_residual_call_irf_f(func, &ai, &ar, &af, &calldescr);
    if let Some(handler_pc) = check_residual_call_exception_after(bh)? {
        return Ok(handler_pc);
    }
    bh.registers_f[dst] = result.to_bits() as i64;
    Ok(p + 1)
}
fn handler_residual_call_irf_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    // blackhole.py:1253-1255 routes through bhimpl_residual_call_irf_v
    // which forwards to cpu.bh_call_v.
    BH_LAST_EXC_VALUE.with(|c| c.set(0));
    bh.bhimpl_residual_call_irf_v(func, &ai, &ar, &af, &calldescr);
    if let Some(handler_pc) = check_residual_call_exception_after(bh)? {
        return Ok(handler_pc);
    }
    Ok(p)
}
// residual_call_ir_*
fn handler_residual_call_ir_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    let dst = code[p] as usize;
    // blackhole.py:1234-1236 → bhimpl_residual_call_ir_i.
    BH_LAST_EXC_VALUE.with(|c| c.set(0));
    let result = bh.bhimpl_residual_call_ir_i(func, &ai, &ar, &calldescr);
    if let Some(handler_pc) = check_residual_call_exception_after(bh)? {
        return Ok(handler_pc);
    }
    bh.registers_i[dst] = result;
    Ok(p + 1)
}
fn handler_residual_call_ir_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    let dst = code[p] as usize;
    // blackhole.py:1237-1239 → bhimpl_residual_call_ir_r.
    BH_LAST_EXC_VALUE.with(|c| c.set(0));
    let result = bh.bhimpl_residual_call_ir_r(func, &ai, &ar, &calldescr);
    if let Some(handler_pc) = check_residual_call_exception_after(bh)? {
        return Ok(handler_pc);
    }
    bh.registers_r[dst] = result.0 as i64;
    Ok(p + 1)
}
fn handler_residual_call_ir_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    // blackhole.py:1240-1242 → bhimpl_residual_call_ir_v.
    BH_LAST_EXC_VALUE.with(|c| c.set(0));
    bh.bhimpl_residual_call_ir_v(func, &ai, &ar, &calldescr);
    if let Some(handler_pc) = check_residual_call_exception_after(bh)? {
        return Ok(handler_pc);
    }
    Ok(p)
}
// residual_call_r_*
fn handler_residual_call_r_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ar, p) = read_list_r(bh, code, position + 1);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    let dst = code[p] as usize;
    // blackhole.py:1225-1226 → bhimpl_residual_call_r_i.
    BH_LAST_EXC_VALUE.with(|c| c.set(0));
    let result = bh.bhimpl_residual_call_r_i(func, &ar, &calldescr);
    if let Some(handler_pc) = check_residual_call_exception_after(bh)? {
        return Ok(handler_pc);
    }
    bh.registers_i[dst] = result;
    Ok(p + 1)
}
fn handler_residual_call_r_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ar, p) = read_list_r(bh, code, position + 1);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    let dst = code[p] as usize;
    // blackhole.py:1227-1229 → bhimpl_residual_call_r_r.
    BH_LAST_EXC_VALUE.with(|c| c.set(0));
    let result = bh.bhimpl_residual_call_r_r(func, &ar, &calldescr);
    if let Some(handler_pc) = check_residual_call_exception_after(bh)? {
        return Ok(handler_pc);
    }
    bh.registers_r[dst] = result.0 as i64;
    Ok(p + 1)
}
fn handler_residual_call_r_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ar, p) = read_list_r(bh, code, position + 1);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    // blackhole.py:1230-1232 → bhimpl_residual_call_r_v.
    BH_LAST_EXC_VALUE.with(|c| c.set(0));
    bh.bhimpl_residual_call_r_v(func, &ar, &calldescr);
    if let Some(handler_pc) = check_residual_call_exception_after(bh)? {
        return Ok(handler_pc);
    }
    Ok(p)
}

// ── pyre-u16 register-width adapters (Sub-slice C.1) ────────────────
//
// PRE-EXISTING-ADAPTATION (pyre register-width axis):
// pyre's `JitCodeBuilder::push_u16` (`assembler.rs:3300`) writes 2
// bytes per register operand because pyre lifts RPython's 256-register
// cap.  RPython's argcode contract (`blackhole.py:107`) reads 1 byte
// per register (`code[position] as usize`).  These `*_pyre_u16`
// handlers decode pyre's 2-byte register layout for the inline-call-
// only production builder's `setup_insns` entries.  Each variant is
// otherwise byte-for-byte identical to its canonical sister handler.
// Convergence path: revisit only if pyre's register allocator caps
// under 256 (Task #45 Sub-slice C-A, currently rejected as
// register-rich pyre jitcodes panic at 256+).

fn handler_int_copy_pyre_u16(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let src = u16::from_le_bytes([code[position], code[position + 1]]) as usize;
    let dst = u16::from_le_bytes([code[position + 2], code[position + 3]]) as usize;
    bh.registers_i[dst] = bh.registers_i[src];
    Ok(position + 4)
}

fn handler_ref_copy_pyre_u16(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let src = u16::from_le_bytes([code[position], code[position + 1]]) as usize;
    let dst = u16::from_le_bytes([code[position + 2], code[position + 3]]) as usize;
    bh.registers_r[dst] = bh.registers_r[src];
    Ok(position + 4)
}

fn handler_ref_return_pyre_u16(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let src = u16::from_le_bytes([code[position], code[position + 1]]) as usize;
    bh.tmpreg_r = bh.registers_r[src];
    bh.return_type = BhReturnType::Ref;
    bh.position = position + 2;
    Err(DispatchError::LeaveFrame)
}

fn handler_raise_pyre_u16(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let src = u16::from_le_bytes([code[p], code[p + 1]]) as usize;
    let exc = bh.registers_r[src];
    bh.position = p + 2;
    // `blackhole.py:1000` `bhimpl_raise(excvalue)` — `assert e` requires
    // a non-null exception object.  The legacy `dispatch_one`
    // `BC_RAISE` arm (`blackhole.rs:2362-2366`) gates `exc != 0` and
    // routes a null `exc` through `aborted = true` + `LeaveFrame`;
    // mirror that here so the pyre-u16 production builder path keeps
    // the same fallback semantics.
    if exc != 0 {
        Err(DispatchError::RaiseException(exc))
    } else {
        bh.aborted = true;
        Err(DispatchError::LeaveFrame)
    }
}

fn handler_last_exc_value_pyre_u16(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let dst = u16::from_le_bytes([code[p], code[p + 1]]) as usize;
    bh.registers_r[dst] = bh.exception_last_value;
    Ok(p + 2)
}

fn handler_goto_if_not_int_is_true_pyre_u16(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let reg = u16::from_le_bytes([code[position], code[position + 1]]) as usize;
    let a = bh.registers_i[reg];
    let target = u16::from_le_bytes([code[position + 2], code[position + 3]]) as usize;
    let pc = position + 4;
    if a != 0 { Ok(pc) } else { Ok(target) }
}

/// Process-global Backend instance for blackhole's `bh_getfield_gc_*` /
/// `bh_setfield_gc_*` / `bh_getarrayitem_gc_*` / `bh_arraylen_gc` reads.
///
/// PRE-EXISTING-ADAPTATION: RPython `blackhole.py:55-56,286` reads
/// `self.cpu = builder.cpu`, where `builder.cpu` is the metainterp-shared
/// AbstractCPU subclass (`LLOpHelpers` in tests, real native cpu in
/// production).  The cpu's `bh_getfield_gc_*` etc. methods are stateless
/// thin wrappers around `lltype.cast_*` / pointer arithmetic in RPython.
///
/// pyre's `MetaInterp::backend` is per-instance (not `'static`) and
/// owns trace-compilation state (descr registries, etc.) that would be
/// inappropriate to share with the blackhole `bh.cpu` field which only
/// needs the stateless GC memory-access methods.  We leak ONE
/// `BackendImpl` instance for the process — its `bh_getfield_gc_*` and
/// related methods do direct unsafe pointer arithmetic (`runner.rs:2244-2271`
/// for dynasm), which is identical to the `dispatch_one` direct-vinfo
/// access path that the canonical-routing migration replaces.
///
/// Convergence path (Task #45 Stage 2 Slice 2.3 Sub-slice C.5): once
/// `dispatch_step_with_fallback`'s placeholder bridge retires, the
/// metainterp-owned backend should be plumbed through to blackhole's
/// `cpu` field — RPython parity ties them together.  Until then, this
/// leaked instance suffices because vable handlers only invoke
/// stateless reads.
#[cfg(any(feature = "dynasm", feature = "cranelift"))]
pub fn pyre_production_cpu() -> &'static dyn majit_backend::Backend {
    // Per-thread leak: `BackendImpl` is not `Sync`, so we keep one
    // instance per thread.  Mirrors `BH_BUILDER3` (`call_jit.rs:702`)
    // which is itself a `thread_local!` — production blackhole resume
    // already runs on the same thread that owns the trace's metainterp.
    thread_local! {
        static CPU: std::cell::Cell<Option<&'static dyn majit_backend::Backend>> =
            const { std::cell::Cell::new(None) };
    }
    CPU.with(|cell| {
        if let Some(cpu) = cell.get() {
            return cpu;
        }
        let backend: Box<dyn majit_backend::Backend> = Box::new(crate::pyjitpl::BackendImpl::new());
        let leaked: &'static dyn majit_backend::Backend = Box::leak(backend);
        cell.set(Some(leaked));
        leaked
    })
}

/// Build a `BlackholeInterpBuilder` for pyre's blackhole resume path,
/// wiring only the canonical bytes whose `JitCodeBuilder` emit-side
/// payload matches the bhimpl handler byte-for-byte.
///
/// PRE-EXISTING-ADAPTATION: pyre's `JitCodeBuilder` emits other
/// runtime opcodes (vable, residual_call, etc.) in a 2-byte-register
/// payload that does not match the canonical RPython argcode contract
/// that `wire_bhimpl_handlers` would route those bytes through.  This
/// builder's `setup_insns` registers only the byte-identical canonical
/// keys plus the misc-family pyre-u16 variants; every other byte falls
/// through `dispatch_step_with_fallback` to `dispatch_one`'s legacy
/// arms.  Currently registered:
///   - `inline_call_pyre_nested/P` at `BC_INLINE_CALL` (Slice 3.1)
///   - Sub-slice B byte-identical canonical: `live/`, `loop_header/i`,
///     `goto/L`, `catch_exception/L`, `jit_merge_point/cIRFIRF`
///   - Sub-slice C.1 misc-family pyre-u16: `int_copy_pyre_u16/i>i`,
///     `ref_copy_pyre_u16/r>r`, `ref_return_pyre_u16/r`,
///     `raise_pyre_u16/r`, `last_exc_value_pyre_u16/>r`,
///     `goto_if_not_int_is_true_pyre_u16/iL`
/// The pyre-jit production thread-locals (`BH_BUILDER3`,
/// `BH_BUILDER_RD`) and inline-call unit fixtures share this builder
/// shape.
///
/// See `pyre-jit-trace/src/jitcode_dispatch.rs:5988-5996` for the
/// `pipeline.insns` ↔ `wellknown_bh_insns` table-unification epic
/// that this minimal install side-steps.
pub fn build_inline_call_only_bh_builder() -> BlackholeInterpBuilder {
    use std::collections::HashMap;
    let mut builder = BlackholeInterpBuilder::new();
    // Sub-slice C.2.0 (`subslice_c2_attempt_failure_cpu_prereq_2026_05_07.md`):
    // wire the blackhole cpu BEFORE C.2.1 vable canonical routing.
    // RPython `blackhole.py:286 self.cpu = builder.cpu` parity — the
    // production builder must carry a non-None cpu so canonical handlers
    // like `handler_getfield_vable_r` don't trip
    // `bh.cpu.expect("cpu not set")` once their setup_insns entries land
    // in C.2.1.  The leaked instance services only stateless GC reads;
    // residual_call (`bh_call_*`, C.3-C.4) prereq audit pending.
    #[cfg(any(feature = "dynasm", feature = "cranelift"))]
    {
        builder.cpu = Some(pyre_production_cpu());
    }
    let mut insns = HashMap::new();
    insns.insert(
        "inline_call_pyre_nested/P".to_string(),
        majit_translate::insns::BC_INLINE_CALL,
    );
    // Sub-slice B (`pyre-bh-setup-insns-byte-identical-subset.md`):
    // five canonical keys whose `JitCodeBuilder` emit-side payload
    // matches the wired `bhimpl_*` handler byte-for-byte.  All other
    // canonical bytes still fall through `dispatch_step_with_fallback`
    // to `dispatch_one` because pyre's u16 register operands diverge
    // from RPython's u8 register operands in `code[position]` reads.
    //   * `live/` — operand-less; both paths skip 2-byte liveness offset.
    //   * `loop_header/i` — `i` is a 1-byte const-pool slot, RPython-aligned.
    //   * `goto/L` — operand-less for registers; 2-byte label.
    //   * `catch_exception/L` — same shape as `goto/L`.
    //   * `jit_merge_point/cIRFIRF` — both paths delegate to
    //     `bhimpl_jit_merge_point`, which decodes the post-opcode
    //     payload via `self.position`-mutating helpers; the IRFIRF
    //     register-list bytes are u8 in BOTH the emit-side and the
    //     decoder.
    insns.insert("live/".to_string(), majit_translate::insns::BC_LIVE);
    insns.insert(
        "loop_header/i".to_string(),
        majit_translate::insns::BC_LOOP_HEADER,
    );
    insns.insert("goto/L".to_string(), majit_translate::insns::BC_JUMP);
    insns.insert(
        "catch_exception/L".to_string(),
        majit_translate::insns::BC_CATCH_EXCEPTION,
    );
    insns.insert(
        "jit_merge_point/cIRFIRF".to_string(),
        majit_translate::insns::BC_JIT_MERGE_POINT_C,
    );
    // Sub-slice C.1 (`subslice_c_register_width_axis_plan_2026_05_07.md`):
    // misc family — six BC_* whose `JitCodeBuilder` emit-side payload
    // uses pyre's u16 register-width layout that diverges from the
    // canonical 1-byte argcode contract.  Each `*_pyre_u16` handler
    // decodes 2-byte register operands; labels remain u16 in both
    // pyre and RPython, so only register operands needed widening.
    insns.insert(
        "int_copy_pyre_u16/i>i".to_string(),
        majit_translate::insns::BC_MOVE_I,
    );
    insns.insert(
        "ref_copy_pyre_u16/r>r".to_string(),
        majit_translate::insns::BC_MOVE_R,
    );
    insns.insert(
        "ref_return_pyre_u16/r".to_string(),
        majit_translate::insns::BC_REF_RETURN,
    );
    insns.insert(
        "raise_pyre_u16/r".to_string(),
        majit_translate::insns::BC_RAISE,
    );
    insns.insert(
        "last_exc_value_pyre_u16/>r".to_string(),
        majit_translate::insns::BC_LAST_EXC_VALUE,
    );
    insns.insert(
        "goto_if_not_int_is_true_pyre_u16/iL".to_string(),
        majit_translate::insns::BC_GOTO_IF_NOT_INT_IS_TRUE,
    );
    builder.setup_insns(&insns);
    // `setup_insns` already derives `op_live` and `op_catch_exception`
    // from the registered canonical subset above.  `rvmprof_code/ii` is
    // not registered in this minimal builder yet, but blackhole exception
    // and live handling expects these cached values to stay in the same
    // fixed-byte space as pyre's runtime opcodes.  Reapply the canonical
    // `BC_*` constants explicitly so this builder stays synchronized as
    // the registered subset changes.
    builder.setup_cached_control_opcodes(
        majit_translate::insns::BC_LIVE as i32,
        majit_translate::insns::BC_CATCH_EXCEPTION as i32,
        majit_translate::insns::BC_RVMPROF_CODE as i32,
    );
    wire_bhimpl_handlers(&mut builder);
    builder
}

/// Wire all currently-ported bhimpl methods into a `BlackholeInterpBuilder`'s
/// dispatch table. Called once after `setup_insns`.
///
/// RPython builds all handlers in `setup_insns` via `_get_method`; pyre
/// wires them incrementally as methods are ported from RPython.
pub fn wire_bhimpl_handlers(builder: &mut BlackholeInterpBuilder) {
    // @arguments("i", returns="i") pattern
    builder.wire_handler("int_same_as/i>i", handler_int_same_as);
    builder.wire_handler("int_neg/i>i", handler_int_neg);
    builder.wire_handler("int_invert/i>i", handler_int_invert);
    builder.wire_handler("int_is_true/i>i", handler_int_is_true);
    builder.wire_handler("int_is_zero/i>i", handler_int_is_zero);
    builder.wire_handler("int_force_ge_zero/i>i", handler_int_force_ge_zero);

    // @arguments("i", "i", returns="i") pattern
    builder.wire_handler("int_add/ii>i", handler_int_add);
    builder.wire_handler("int_sub/ii>i", handler_int_sub);
    // pyre-only primitives — see handler comments for rationale.
    builder.wire_handler("int_add_assign/ii>i", handler_int_add_assign_pyre);
    builder.wire_handler("int_sub_assign/ii>i", handler_int_sub_assign_pyre);
    builder.wire_handler("int_deref/i>i", handler_int_deref_pyre);
    builder.wire_handler("abort/", handler_abort_marker_pyre);

    // pyre-only state_field family (no RPython counterpart). Blackhole
    // treats these as no-ops; handlers exist only to advance position past
    // the operand bytes so dispatch_table[opcode] is wired (not placeholder)
    // and `dispatch_step_with_fallback` no longer routes through dispatch_one
    // for them.
    builder.wire_handler("load_state_field/di", handler_state_field_noop_di);
    builder.wire_handler("store_state_field/di", handler_state_field_noop_di);
    builder.wire_handler("load_state_array/dii", handler_state_array_noop_dii);
    builder.wire_handler("store_state_array/dii", handler_state_array_noop_dii);
    builder.wire_handler("load_state_varray/dii", handler_state_array_noop_dii);
    builder.wire_handler("store_state_varray/dii", handler_state_array_noop_dii);

    // jit_merge_point + abort_permanent — bodies live on
    // `BlackholeInterpreter::bhimpl_jit_merge_point` / `bhimpl_abort_permanent`
    // and are shared with the legacy dispatch_one arms so test fixtures that
    // bypass `acquire_interp` continue to work.
    builder.wire_handler("jit_merge_point/iIRFIRF", handler_jit_merge_point_i);
    builder.wire_handler("jit_merge_point/cIRFIRF", handler_jit_merge_point_c);
    builder.wire_handler("abort_permanent/", handler_abort_permanent);

    builder.wire_handler("int_mul/ii>i", handler_int_mul);
    builder.wire_handler("int_div/ii>i", handler_int_floordiv);
    builder.wire_handler("int_floordiv/ii>i", handler_int_floordiv);
    builder.wire_handler("int_mod/ii>i", handler_int_mod);
    builder.wire_handler("int_and/ii>i", handler_int_and);
    builder.wire_handler("int_or/ii>i", handler_int_or);
    builder.wire_handler("int_xor/ii>i", handler_int_xor);
    builder.wire_handler("int_rshift/ii>i", handler_int_rshift);
    builder.wire_handler("int_lshift/ii>i", handler_int_lshift);
    builder.wire_handler("uint_rshift/ii>i", handler_uint_rshift);
    builder.wire_handler("int_lt/ii>i", handler_int_lt);
    builder.wire_handler("int_le/ii>i", handler_int_le);
    builder.wire_handler("int_eq/ii>i", handler_int_eq);
    builder.wire_handler("int_ne/ii>i", handler_int_ne);
    builder.wire_handler("int_gt/ii>i", handler_int_gt);
    builder.wire_handler("int_ge/ii>i", handler_int_ge);

    // Copy operations
    builder.wire_handler("int_copy/i>i", handler_int_copy);
    // pyre-only abort placeholder emitted by `Assembler::encode_op`'s
    // default branch for `OpKind::Abort { .. }`.
    builder.wire_handler("abort/>i", handler_abort_result_marker_i);
    builder.wire_handler("abort/>r", handler_abort_result_marker_r);

    // Control flow
    builder.wire_handler("live/", handler_live);
    builder.wire_handler("goto/L", handler_goto);
    builder.wire_handler("goto_if_not/iL", handler_goto_if_not);

    // Unsigned comparisons (blackhole.py:571-582)
    builder.wire_handler("uint_lt/ii>i", handler_uint_lt);
    builder.wire_handler("uint_le/ii>i", handler_uint_le);
    builder.wire_handler("uint_gt/ii>i", handler_uint_gt);
    builder.wire_handler("uint_ge/ii>i", handler_uint_ge);

    // Float arithmetic (blackhole.py:676-718)
    builder.wire_handler("float_neg/f>f", handler_float_neg);
    builder.wire_handler("float_abs/f>f", handler_float_abs);
    builder.wire_handler("float_add/ff>f", handler_float_add);
    builder.wire_handler("float_sub/ff>f", handler_float_sub);
    builder.wire_handler("float_mul/ff>f", handler_float_mul);
    builder.wire_handler("float_truediv/ff>f", handler_float_truediv);

    // Float comparisons → int result (blackhole.py:720-749)
    builder.wire_handler("float_lt/ff>i", handler_float_lt);
    builder.wire_handler("float_le/ff>i", handler_float_le);
    builder.wire_handler("float_eq/ff>i", handler_float_eq);
    builder.wire_handler("float_ne/ff>i", handler_float_ne);
    builder.wire_handler("float_gt/ff>i", handler_float_gt);
    builder.wire_handler("float_ge/ff>i", handler_float_ge);

    // Ref operations (blackhole.py:584-610)
    builder.wire_handler("ptr_eq/rr>i", handler_ptr_eq);
    builder.wire_handler("ptr_ne/rr>i", handler_ptr_ne);
    builder.wire_handler("instance_ptr_eq/rr>i", handler_instance_ptr_eq);
    builder.wire_handler("instance_ptr_ne/rr>i", handler_instance_ptr_ne);
    builder.wire_handler("ptr_iszero/r>i", handler_ptr_iszero);
    builder.wire_handler("ptr_nonzero/r>i", handler_ptr_nonzero);

    // Copy operations (ref + float)
    builder.wire_handler("ref_copy/r>r", handler_ref_copy);
    builder.wire_handler("float_copy/f>f", handler_float_copy);

    // Conditional jumps (blackhole.py:871-920)
    builder.wire_handler("goto_if_not_int_lt/iiL", handler_goto_if_not_int_lt);
    builder.wire_handler("goto_if_not_int_le/iiL", handler_goto_if_not_int_le);
    builder.wire_handler("goto_if_not_int_eq/iiL", handler_goto_if_not_int_eq);
    builder.wire_handler("goto_if_not_int_ne/iiL", handler_goto_if_not_int_ne);
    builder.wire_handler("goto_if_not_int_gt/iiL", handler_goto_if_not_int_gt);
    builder.wire_handler("goto_if_not_int_ge/iiL", handler_goto_if_not_int_ge);
    // goto_if_not_int_is_true = goto_if_not (alias, blackhole.py:913)
    builder.wire_handler("goto_if_not_int_is_true/iL", handler_goto_if_not);

    // Guard values — no-op in blackhole (blackhole.py:648-656)
    builder.wire_handler("int_guard_value/i", handler_int_guard_value);
    builder.wire_handler("ref_guard_value/r", handler_ref_guard_value);
    builder.wire_handler("float_guard_value/f", handler_float_guard_value);

    // Push/pop tmpreg (blackhole.py:661-679)
    builder.wire_handler("int_push/i", handler_int_push);
    builder.wire_handler("ref_push/r", handler_ref_push);
    builder.wire_handler("float_push/f", handler_float_push);
    builder.wire_handler("int_pop/>i", handler_int_pop);
    builder.wire_handler("ref_pop/>r", handler_ref_pop);
    builder.wire_handler("float_pop/>f", handler_float_pop);

    // Record — no-op (blackhole.py:616-636)
    builder.wire_handler("record_exact_class/ri", handler_record_exact_class);
    builder.wire_handler("record_exact_value_r/rr", handler_record_exact_value_r);
    builder.wire_handler("record_exact_value_i/ii", handler_record_exact_value_i);

    // Cast operations (blackhole.py:800-831)
    builder.wire_handler("cast_float_to_int/f>i", handler_cast_float_to_int);
    builder.wire_handler("cast_int_to_float/i>f", handler_cast_int_to_float);

    // int_signext (blackhole.py:566-569)
    builder.wire_handler("int_signext/ii>i", handler_int_signext);

    // Overflow ops (blackhole.py:478-497)
    builder.wire_handler("int_add_jump_if_ovf/Lii>i", handler_int_add_jump_if_ovf);
    builder.wire_handler("int_sub_jump_if_ovf/Lii>i", handler_int_sub_jump_if_ovf);
    builder.wire_handler("int_mul_jump_if_ovf/Lii>i", handler_int_mul_jump_if_ovf);

    // Misc simple ops
    builder.wire_handler("assert_not_none/r", handler_assert_not_none);
    builder.wire_handler("virtual_ref/r>r", handler_virtual_ref);
    builder.wire_handler("virtual_ref_finish/r", handler_virtual_ref_finish);
    builder.wire_handler("loop_header/i", handler_loop_header);
    builder.wire_handler("ref_isconstant/r>i", handler_ref_isconstant);
    builder.wire_handler("ref_isvirtual/r>i", handler_ref_isvirtual);
    builder.wire_handler(
        "goto_if_not_int_is_zero/iL",
        handler_goto_if_not_int_is_zero,
    );
    builder.wire_handler("goto_if_not_ptr_iszero/rL", handler_goto_if_not_ptr_iszero);
    builder.wire_handler(
        "goto_if_not_ptr_nonzero/rL",
        handler_goto_if_not_ptr_nonzero,
    );
    builder.wire_handler("unreachable/", handler_unreachable);

    // Field operations (blackhole.py:1432-1481).
    //
    // Canonical `/rd>X` and `/rXd` are the RPython-exact keys. The
    // `/id>X` / `/iXd` variants carry a pyre tagged-int base in an int
    // register (same backend primitive `bh_{get,set}field_gc_*`, only
    // the base's register class differs). The emit side at
    // `majit-translate/src/jit_codewriter/assembler.rs` OpKind::FieldRead/
    // FieldWrite derives the opname kind suffix from the VALUE / RESULT
    // register kind and the argcodes from each register's class, so the
    // tagged-int variant is indistinguishable from the canonical one
    // except at the first argcode character.
    builder.wire_handler("getfield_gc_i/rd>i", handler_getfield_gc_i);
    builder.wire_handler("getfield_gc_r/rd>r", handler_getfield_gc_r);
    builder.wire_handler("getfield_gc_f/rd>f", handler_getfield_gc_f);
    builder.wire_handler("getfield_gc_i/id>i", handler_getfield_gc_i_intbase);
    builder.wire_handler("getfield_gc_r/id>r", handler_getfield_gc_r_intbase);
    builder.wire_handler("getfield_gc_f/id>f", handler_getfield_gc_f_intbase);
    builder.wire_handler("getfield_gc_i_pure/rd>i", handler_getfield_gc_i);
    builder.wire_handler("getfield_gc_r_pure/rd>r", handler_getfield_gc_r);
    builder.wire_handler("getfield_gc_f_pure/rd>f", handler_getfield_gc_f);
    builder.wire_handler("setfield_gc_i/rid", handler_setfield_gc_i);
    builder.wire_handler("setfield_gc_r/rrd", handler_setfield_gc_r);
    builder.wire_handler("setfield_gc_f/rfd", handler_setfield_gc_f);
    builder.wire_handler("setfield_gc_i/iid", handler_setfield_gc_i_intbase);
    builder.wire_handler("setfield_gc_r/ird", handler_setfield_gc_r_intbase);
    builder.wire_handler("arraylen_gc/rd>i", handler_arraylen_gc);

    // Array item operations (blackhole.py:1329-1365). Same tagged-int
    // base rationale as above for the `/iid>i` and `/ird>i` variants.
    builder.wire_handler("getarrayitem_gc_i/rid>i", handler_getarrayitem_gc_i);
    builder.wire_handler("getarrayitem_gc_r/rid>r", handler_getarrayitem_gc_r);
    // pyre-only `/rrd>r`: index register is Ref-classified (rtyper gap).
    builder.wire_handler(
        "getarrayitem_gc_r/rrd>r",
        handler_getarrayitem_gc_r_refindex,
    );
    builder.wire_handler("getarrayitem_gc_i/iid>i", handler_getarrayitem_gc_i_intbase);
    builder.wire_handler(
        "getarrayitem_gc_i/ird>i",
        handler_getarrayitem_gc_i_intbase_refindex,
    );
    builder.wire_handler("getarrayitem_gc_i_pure/rid>i", handler_getarrayitem_gc_i);
    builder.wire_handler("getarrayitem_gc_r_pure/rid>r", handler_getarrayitem_gc_r);
    builder.wire_handler("setarrayitem_gc_i/riid", handler_setarrayitem_gc_i);
    builder.wire_handler("setarrayitem_gc_r/rird", handler_setarrayitem_gc_r);
    // pyre-only `/rrrd`: index register is Ref-classified (rtyper gap).
    builder.wire_handler("setarrayitem_gc_r/rrrd", handler_setarrayitem_gc_r_refindex);

    // Raw field operations (blackhole.py:1464-1502)
    builder.wire_handler("getfield_raw_i/id>i", handler_getfield_raw_i);
    builder.wire_handler("getfield_raw_f/id>f", handler_getfield_raw_f);
    builder.wire_handler("setfield_raw_i/iid", handler_setfield_raw_i);
    builder.wire_handler("setfield_raw_f/ifd", handler_setfield_raw_f);

    // Greenfield aliases
    builder.wire_handler("getfield_gc_i_greenfield/rd>i", handler_getfield_gc_i);
    builder.wire_handler("getfield_gc_r_greenfield/rd>r", handler_getfield_gc_r);
    builder.wire_handler("getfield_gc_f_greenfield/rd>f", handler_getfield_gc_f);

    // New operations (blackhole.py:1301-1327)
    builder.wire_handler("new/d>r", handler_new);
    builder.wire_handler("new_with_vtable/d>r", handler_new_with_vtable);
    builder.wire_handler("new_array/id>r", handler_new_array);
    builder.wire_handler("new_array_clear/id>r", handler_new_array_clear);

    // String operations (blackhole.py:1200-1283)
    builder.wire_handler("strlen/r>i", handler_strlen);
    builder.wire_handler("strgetitem/ri>i", handler_strgetitem);
    builder.wire_handler("strsetitem/rii", handler_strsetitem);
    builder.wire_handler("newstr/i>r", handler_newstr);
    builder.wire_handler("unicodelen/r>i", handler_unicodelen);
    builder.wire_handler("unicodegetitem/ri>i", handler_unicodegetitem);
    builder.wire_handler("unicodesetitem/rii", handler_unicodesetitem);
    builder.wire_handler("newunicode/i>r", handler_newunicode);

    // Exception handling (blackhole.py:969-975)
    builder.wire_handler("catch_exception/L", handler_catch_exception);

    // Interior field operations (blackhole.py:1411-1429)
    // _r/_f deferred until Backend gains those variants.
    builder.wire_handler("getinteriorfield_gc_i/rid>i", handler_getinteriorfield_gc_i);
    builder.wire_handler("setinteriorfield_gc_i/riid", handler_setinteriorfield_gc_i);

    // Residual call operations (blackhole.py:1224-1255)
    builder.wire_handler("residual_call_irf_i/iIRFd>i", handler_residual_call_irf_i);
    builder.wire_handler("residual_call_irf_r/iIRFd>r", handler_residual_call_irf_r);
    builder.wire_handler("residual_call_irf_f/iIRFd>f", handler_residual_call_irf_f);
    builder.wire_handler("residual_call_irf_v/iIRFd", handler_residual_call_irf_v);
    builder.wire_handler("residual_call_ir_i/iIRd>i", handler_residual_call_ir_i);
    builder.wire_handler("residual_call_ir_r/iIRd>r", handler_residual_call_ir_r);
    builder.wire_handler("residual_call_ir_v/iIRd", handler_residual_call_ir_v);
    builder.wire_handler("residual_call_r_i/iRd>i", handler_residual_call_r_i);
    builder.wire_handler("residual_call_r_r/iRd>r", handler_residual_call_r_r);
    builder.wire_handler("residual_call_r_v/iRd", handler_residual_call_r_v);
    // Misc no-ops (blackhole.py:1017-1049)
    builder.wire_handler("jit_debug/riiii", handler_jit_debug);
    builder.wire_handler("jit_enter_portal_frame/i", handler_jit_enter_portal_frame);
    builder.wire_handler("jit_leave_portal_frame/", handler_jit_leave_portal_frame);

    // Float conditional jumps (blackhole.py:751-798)
    builder.wire_handler("goto_if_not_float_lt/ffL", handler_goto_if_not_float_lt);
    builder.wire_handler("goto_if_not_float_le/ffL", handler_goto_if_not_float_le);
    builder.wire_handler("goto_if_not_float_eq/ffL", handler_goto_if_not_float_eq);
    builder.wire_handler("goto_if_not_float_ne/ffL", handler_goto_if_not_float_ne);
    builder.wire_handler("goto_if_not_float_gt/ffL", handler_goto_if_not_float_gt);
    builder.wire_handler("goto_if_not_float_ge/ffL", handler_goto_if_not_float_ge);
    builder.wire_handler("goto_if_not_ptr_eq/rrL", handler_goto_if_not_ptr_eq);
    builder.wire_handler("goto_if_not_ptr_ne/rrL", handler_goto_if_not_ptr_ne);

    // Assert/isconstant (no-ops in blackhole)
    builder.wire_handler("int_assert_green/i", handler_int_assert_green);
    builder.wire_handler("ref_assert_green/r", handler_ref_assert_green);
    builder.wire_handler("float_assert_green/f", handler_float_assert_green);
    builder.wire_handler("int_isconstant/i>i", handler_int_isconstant);
    builder.wire_handler("float_isconstant/f>i", handler_float_isconstant);

    // Misc integer ops
    builder.wire_handler("uint_mul_high/ii>i", handler_uint_mul_high);
    builder.wire_handler("int_between/iii>i", handler_int_between);

    // String hashing (stubs)
    builder.wire_handler("strhash/r>i", handler_strhash);
    builder.wire_handler("unicodehash/r>i", handler_unicodehash);

    // Float <-> longlong / singlefloat conversions
    builder.wire_handler(
        "convert_float_bytes_to_longlong/f>i",
        handler_convert_float_bytes_to_longlong,
    );
    builder.wire_handler(
        "convert_longlong_bytes_to_float/i>f",
        handler_convert_longlong_bytes_to_float,
    );
    builder.wire_handler(
        "cast_float_to_singlefloat/f>i",
        handler_cast_float_to_singlefloat,
    );
    builder.wire_handler(
        "cast_singlefloat_to_float/i>f",
        handler_cast_singlefloat_to_float,
    );

    // Misc
    // RPython `rpython/jit/metainterp/blackhole.py:1546-1548`:
    //   @arguments("r")
    //   def bhimpl_hint_force_virtualizable(r): pass
    // Canonical key is `hint_force_virtualizable/r`, not the previous
    // pyre-invented `/rd` shape that inserted a phantom descr slot.
    builder.wire_handler(
        "hint_force_virtualizable/r",
        handler_hint_force_virtualizable,
    );
    // RPython `rpython/jit/metainterp/blackhole.py:1558-1560`:
    //   @arguments("cpu", "r", returns="i")
    //   def bhimpl_guard_class(cpu, struct): return cpu.bh_classof(struct)
    // Canonical key is `guard_class/r>i`; the previous `/ri` shape was
    // a pyre-invented bigram that omitted the `>i` return marker.
    builder.wire_handler("guard_class/r>i", handler_guard_class);
    // RPython `rpython/jit/metainterp/blackhole.py:1537-1539`:
    //   @arguments("r", "d", "d")
    //   def bhimpl_record_quasiimmut_field(struct, fielddescr, mutatefielddescr):
    builder.wire_handler(
        "record_quasiimmut_field/rdd",
        handler_record_quasiimmut_field,
    );
    // PRE-EXISTING-ADAPTATION op for Rust fat-pointer dispatch — see
    // `majit/majit-translate/src/model.rs OpKind::VtableMethodPtr`.
    // No runtime consumer ships yet; the handler panics on dispatch so a
    // future regression that triggers this path in pyre fails loudly
    // instead of silently miscompiling.
    builder.wire_handler(
        "vtable_method_ptr/rd>i",
        handler_vtable_method_ptr_unimplemented,
    );
    builder.wire_handler(
        "jit_force_quasi_immutable/rd",
        handler_jit_force_quasi_immutable,
    );
    builder.wire_handler(
        "record_known_result_i_ir_v/iiIRd",
        handler_record_known_result_i_ir_v,
    );
    builder.wire_handler(
        "record_known_result_r_ir_v/riIRd",
        handler_record_known_result_r_ir_v,
    );
    builder.wire_handler("str_guard_value/rid>r", handler_str_guard_value);
    builder.wire_handler("rvmprof_code/ii", handler_rvmprof_code);
    builder.wire_handler("copystrcontent/rriii", handler_copystrcontent);
    builder.wire_handler("copyunicodecontent/rriii", handler_copyunicodecontent);
    builder.wire_handler("current_trace_length/>i", handler_current_trace_length);

    // Exception ops (blackhole.py:976-1009)
    builder.wire_handler("raise/r", handler_raise);
    builder.wire_handler("reraise/", handler_reraise);
    builder.wire_handler("last_exception/>i", handler_last_exception);
    builder.wire_handler("last_exc_value/>r", handler_last_exc_value);
    builder.wire_handler(
        "goto_if_exception_mismatch/iL",
        handler_goto_if_exception_mismatch,
    );
    builder.wire_handler("debug_fatalerror/r", handler_debug_fatalerror);

    // Cast ptr<->int (blackhole.py:603-610)
    builder.wire_handler("cast_ptr_to_int/r>i", handler_cast_ptr_to_int);
    builder.wire_handler("cast_int_to_ptr/i>r", handler_cast_int_to_ptr);

    // Vable field operations — canonical `/rd>X` / `/rXd` RPython shape
    // (blackhole.py:1446-1495). pyre tagged-int base adds `/id>X` /
    // `/iXd` variants handled by the `*_intbase` helpers. See the
    // Field operations comment above for the Void/State/Unknown
    // rationale — emit now derives kind suffix from the value/result
    // register's kind so the `_v` sentinel form is no longer produced.
    builder.wire_handler("getfield_vable_i/rd>i", handler_getfield_vable_i);
    builder.wire_handler("getfield_vable_r/rd>r", handler_getfield_vable_r);
    builder.wire_handler("getfield_vable_f/rd>f", handler_getfield_vable_f);
    builder.wire_handler("getfield_vable_i/id>i", handler_getfield_vable_i_intbase);
    builder.wire_handler("setfield_vable_i/rid", handler_setfield_vable_i);
    builder.wire_handler("setfield_vable_r/rrd", handler_setfield_vable_r);
    builder.wire_handler("setfield_vable_f/rfd", handler_setfield_vable_f);
    builder.wire_handler("setfield_vable_i/iid", handler_setfield_vable_i_intbase);
    builder.wire_handler("setfield_vable_r/ird", handler_setfield_vable_r_intbase);
    builder.wire_handler("getarrayitem_vable_i/ridd>i", handler_getarrayitem_vable_i);
    builder.wire_handler("getarrayitem_vable_r/ridd>r", handler_getarrayitem_vable_r);
    builder.wire_handler("setarrayitem_vable_i/riidd", handler_setarrayitem_vable_i);
    builder.wire_handler("setarrayitem_vable_r/rirdd", handler_setarrayitem_vable_r);
    builder.wire_handler("arraylen_vable/rdd>i", handler_arraylen_vable);
    builder.wire_handler("getarrayitem_raw_i/iid>i", handler_getarrayitem_raw_i);
    builder.wire_handler("setarrayitem_raw_i/iiid", handler_setarrayitem_raw_i);
    builder.wire_handler("conditional_call_ir_v/iiIRd", handler_conditional_call_ir_v);
    builder.wire_handler(
        "conditional_call_value_ir_i/iiIRd>i",
        handler_conditional_call_value_ir_i,
    );
    builder.wire_handler(
        "conditional_call_value_ir_r/riIRd>r",
        handler_conditional_call_value_ir_r,
    );
    builder.wire_handler("getlistitem_gc_i/ridd>i", handler_getlistitem_gc_i);
    builder.wire_handler("getlistitem_gc_r/ridd>r", handler_getlistitem_gc_r);
    builder.wire_handler("setlistitem_gc_i/riidd", handler_setlistitem_gc_i);
    builder.wire_handler("setlistitem_gc_r/rirdd", handler_setlistitem_gc_r);
    builder.wire_handler("switch/id", handler_switch);
    builder.wire_handler("getlistitem_gc_f/ridd>f", handler_getlistitem_gc_f);
    builder.wire_handler("setlistitem_gc_f/rifdd", handler_setlistitem_gc_f);
    builder.wire_handler("check_neg_index/rid>i", handler_check_neg_index);
    builder.wire_handler(
        "check_resizable_neg_index/rid>i",
        handler_check_resizable_neg_index,
    );

    // Float/raw array ops
    builder.wire_handler("getarrayitem_gc_f/rid>f", handler_getarrayitem_gc_f);
    builder.wire_handler("getarrayitem_gc_f_pure/rid>f", handler_getarrayitem_gc_f);
    builder.wire_handler("setarrayitem_gc_f/rifd", handler_setarrayitem_gc_f);
    builder.wire_handler("getarrayitem_raw_f/iid>f", handler_getarrayitem_raw_f);
    builder.wire_handler("setarrayitem_raw_f/iifd", handler_setarrayitem_raw_f);
    builder.wire_handler("getfield_raw_r/id>r", handler_getfield_raw_r);
    builder.wire_handler("getinteriorfield_gc_f/rid>f", handler_getinteriorfield_gc_f);
    builder.wire_handler("getinteriorfield_gc_r/rid>r", handler_getinteriorfield_gc_r);
    builder.wire_handler("setinteriorfield_gc_f/rifd", handler_setinteriorfield_gc_f);
    builder.wire_handler("setinteriorfield_gc_r/rird", handler_setinteriorfield_gc_r);
    // RPython `rpython/jit/metainterp/blackhole.py:1518-1534` canonical
    // signatures (`@arguments("cpu", "r", "i", "i", "i", "i", returns="X")`
    // and `@arguments("cpu", "r", "i", "X", "i", "i", "i", "d")`).
    // The previous wire keys carried one extra `i` on loads and one
    // missing `i` on stores relative to the handler body, so they
    // disagreed with both RPython and the handler's own byte math.
    builder.wire_handler("gc_load_indexed_i/riiii>i", handler_gc_load_indexed_i);
    builder.wire_handler("gc_load_indexed_f/riiii>f", handler_gc_load_indexed_f);
    builder.wire_handler("gc_store_indexed_i/riiiiid", handler_gc_store_indexed_i);
    builder.wire_handler("gc_store_indexed_f/rifiiid", handler_gc_store_indexed_f);
    builder.wire_handler("raw_store_i/iiid", handler_raw_store_i);
    builder.wire_handler("raw_store_f/iifd", handler_raw_store_f);
    builder.wire_handler("raw_load_i/iid>i", handler_raw_load_i);
    builder.wire_handler("raw_load_f/iid>f", handler_raw_load_f);
    builder.wire_handler("newlist/idddd>r", handler_newlist);
    builder.wire_handler("newlist_clear/idddd>r", handler_newlist_clear);
    builder.wire_handler("newlist_hint/idddd>r", handler_newlist_hint);
    builder.wire_handler("getarrayitem_vable_f/ridd>f", handler_getarrayitem_vable_f);
    builder.wire_handler("setarrayitem_vable_f/rifdd", handler_setarrayitem_vable_f);

    // Inline call (stub — needs frame-chain)
    builder.wire_handler("inline_call_irf_i/dIRF>i", handler_inline_call_irf_i);
    builder.wire_handler("inline_call_irf_r/dIRF>r", handler_inline_call_irf_r);
    builder.wire_handler("inline_call_irf_f/dIRF>f", handler_inline_call_irf_f);
    builder.wire_handler("inline_call_irf_v/dIRF", handler_inline_call_irf_v);
    builder.wire_handler("inline_call_ir_i/dIR>i", handler_inline_call_ir_i);
    builder.wire_handler("inline_call_ir_r/dIR>r", handler_inline_call_ir_r);
    builder.wire_handler("inline_call_ir_v/dIR", handler_inline_call_ir_v);
    builder.wire_handler("inline_call_r_i/dR>i", handler_inline_call_r_i);
    builder.wire_handler("inline_call_r_r/dR>r", handler_inline_call_r_r);
    builder.wire_handler("inline_call_r_v/dR", handler_inline_call_r_v);

    // PRE-EXISTING-ADAPTATION: pyre nested-bytecode `inline_call`.  See
    // the comment on `handler_inline_call_pyre_nested` for rationale.
    // The canonical keys above remain wired but stay inert in production
    // because pyre never registers them in `wellknown_bh_insns()` (no
    // matching entry in `setup_insns`-built `_insns`).  Byte 17 is
    // exposed via the pyre-only `inline_call_pyre_nested/P` key in
    // `pyre_extension_insns()`.
    builder.wire_handler("inline_call_pyre_nested/P", handler_inline_call_pyre_nested);

    // Recursive call (stub — needs portal runner)
    // RPython `rpython/jit/metainterp/blackhole.py:1101-1132`:
    //   @arguments("self", "i", "I", "R", "F", "I", "R", "F", returns="X")
    // canonical keys `recursive_call_{i,r,f,v}/iIRFIRF{>X,}`. `recursive_call`
    // is not in `assembler.py:312 USE_C_FORM`, so the `c` short-const
    // variant is not valid here.
    builder.wire_handler("recursive_call_i/iIRFIRF>i", handler_recursive_call_i);
    builder.wire_handler("recursive_call_r/iIRFIRF>r", handler_recursive_call_r);
    builder.wire_handler("recursive_call_f/iIRFIRF>f", handler_recursive_call_f);
    builder.wire_handler("recursive_call_v/iIRFIRF", handler_recursive_call_v);

    // Returns
    builder.wire_handler("int_return/i", handler_int_return);
    builder.wire_handler("ref_return/r", handler_ref_return);
    builder.wire_handler("float_return/f", handler_float_return);
    builder.wire_handler("void_return/", handler_void_return);

    // PRE-EXISTING-ADAPTATION: pyre-u16 register-width adapters (Sub-slice C.1).
    // See the block-header comment on `handler_int_copy_pyre_u16` for the
    // RPython argcode contract divergence rationale.  These wirings are
    // inert for builders whose `setup_insns` does not register the
    // matching opname (`build_default_bh_builder` etc.); they only
    // become live in `build_inline_call_only_bh_builder`'s
    // pyre-namespaced insns map.
    builder.wire_handler("int_copy_pyre_u16/i>i", handler_int_copy_pyre_u16);
    builder.wire_handler("ref_copy_pyre_u16/r>r", handler_ref_copy_pyre_u16);
    builder.wire_handler("ref_return_pyre_u16/r", handler_ref_return_pyre_u16);
    builder.wire_handler("raise_pyre_u16/r", handler_raise_pyre_u16);
    builder.wire_handler(
        "last_exc_value_pyre_u16/>r",
        handler_last_exc_value_pyre_u16,
    );
    builder.wire_handler(
        "goto_if_not_int_is_true_pyre_u16/iL",
        handler_goto_if_not_int_is_true_pyre_u16,
    );
}

// ── goto_if_not_float (blackhole.py:751-798) ────────────────────────
macro_rules! bhhandler_goto_if_not_ff {
    ($name:ident, $cmp:expr) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
            let b = f64::from_bits(bh.registers_f[code[position + 1] as usize] as u64);
            let target = (code[position + 2] as usize) | ((code[position + 3] as usize) << 8);
            let pc = position + 4;
            if $cmp(a, b) { Ok(pc) } else { Ok(target) }
        }
    };
}
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_lt, |a: f64, b: f64| a < b);
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_le, |a: f64, b: f64| a <= b);
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_eq, |a: f64, b: f64| a == b);
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_ne, |a: f64, b: f64| a != b);
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_gt, |a: f64, b: f64| a > b);
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_ge, |a: f64, b: f64| a >= b);

// goto_if_not_ptr_eq/ne (reuse ii macro with ref registers)
fn handler_goto_if_not_ptr_eq(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_r[code[position] as usize];
    let b = bh.registers_r[code[position + 1] as usize];
    let target = (code[position + 2] as usize) | ((code[position + 3] as usize) << 8);
    if a == b { Ok(position + 4) } else { Ok(target) }
}
fn handler_goto_if_not_ptr_ne(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_r[code[position] as usize];
    let b = bh.registers_r[code[position + 1] as usize];
    let target = (code[position + 2] as usize) | ((code[position + 3] as usize) << 8);
    if a != b { Ok(position + 4) } else { Ok(target) }
}

// assert_green / isconstant — no-ops
fn handler_int_assert_green(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Ok(p + 1)
}
fn handler_ref_assert_green(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Ok(p + 1)
}
fn handler_float_assert_green(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Ok(p + 1)
}
fn handler_int_isconstant(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[p + 1] as usize] = 0;
    Ok(p + 2)
}
fn handler_float_isconstant(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[p + 1] as usize] = 0;
    Ok(p + 2)
}

// misc
fn handler_uint_mul_high(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[p] as usize] as u64;
    let b = bh.registers_i[code[p + 1] as usize] as u64;
    bh.registers_i[code[p + 2] as usize] = ((a as u128 * b as u128) >> 64) as i64;
    Ok(p + 3)
}
fn handler_int_between(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[p] as usize];
    let b = bh.registers_i[code[p + 1] as usize];
    let c = bh.registers_i[code[p + 2] as usize];
    bh.registers_i[code[p + 3] as usize] = (a <= b && b < c) as i64;
    Ok(p + 4)
}
fn handler_strhash(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[p + 1] as usize] = 0;
    Ok(p + 2)
}
fn handler_unicodehash(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[p + 1] as usize] = 0;
    Ok(p + 2)
}
fn handler_convert_float_bytes_to_longlong(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[p + 1] as usize] = bh.registers_f[code[p] as usize];
    Ok(p + 2)
}
fn handler_convert_longlong_bytes_to_float(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_f[code[p + 1] as usize] = bh.registers_i[code[p] as usize];
    Ok(p + 2)
}
fn handler_cast_float_to_singlefloat(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let f = f64::from_bits(bh.registers_f[code[p] as usize] as u64);
    bh.registers_i[code[p + 1] as usize] = (f as f32).to_bits() as i64;
    Ok(p + 2)
}
fn handler_cast_singlefloat_to_float(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let f = f32::from_bits(bh.registers_i[code[p] as usize] as u32) as f64;
    bh.registers_f[code[p + 1] as usize] = f.to_bits() as i64;
    Ok(p + 2)
}
fn handler_hint_force_virtualizable(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // RPython `blackhole.py:1547` `bhimpl_hint_force_virtualizable(r): pass`
    // — body is a pure no-op. With canonical argcode `r` (one 1-byte
    // register index in RPython assembler.py's `chr(...)` packing) the
    // handler advances past that single operand byte.
    Ok(p + 1)
}
fn handler_guard_class(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Ok(p + 2)
}
/// `vtable_method_ptr` reaches the blackhole only when a `dyn Trait`
/// indirect call survives unfrozen into a metainterp resume.  pyre's hot
/// path does not currently emit this pattern; intentionally panic so any
/// future regression is loud rather than silent.  The codewriter still
/// emits the op + descriptor (PRE-EXISTING-ADAPTATION of
/// `rpython/rtyper/rclass.py:371-377 getclsfield()`) so the IR survives
/// serialization for the next integration step.
fn handler_vtable_method_ptr_unimplemented(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _p: usize,
) -> Result<usize, DispatchError> {
    unimplemented!("vtable_method_ptr blackhole consumer (backend epic)");
}

fn handler_record_quasiimmut_field(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // RPython `bhimpl_record_quasiimmut_field` is a no-op during blackhole
    // execution (`blackhole.py:1539 pass`); the metainterp consumes the two
    // descriptors during tracing instead.  Skip past `r` (1 byte) +
    // `d` (2 bytes) + `d` (2 bytes) to reach the next opcode.
    let p = p + 1; // r
    let (_, p) = read_descr(bh, code, p);
    let (_, p) = read_descr(bh, code, p);
    Ok(p)
}
fn handler_jit_force_quasi_immutable(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Ok(p + 3)
}
fn handler_record_known_result_i_ir_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let p = position + 2;
    let (_, p) = read_list_i(bh, code, p);
    let (_, p) = read_list_r(bh, code, p);
    let (_, p) = read_descr(bh, code, p);
    Ok(p)
}
fn handler_record_known_result_r_ir_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let p = position + 2;
    let (_, p) = read_list_i(bh, code, p);
    let (_, p) = read_list_r(bh, code, p);
    let (_, p) = read_descr(bh, code, p);
    Ok(p)
}
fn handler_str_guard_value(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let r = bh.registers_r[code[position] as usize];
    let (_, p) = read_descr(bh, code, position + 2);
    bh.registers_r[code[p] as usize] = r;
    Ok(p + 1)
}
fn handler_rvmprof_code(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let leaving = bh.registers_i[code[p] as usize];
    let unique_id = bh.registers_i[code[p + 1] as usize];
    bh.bhimpl_rvmprof_code(leaving, unique_id);
    Ok(p + 2)
}
/// RPython `blackhole.py:1575-1578` `bhimpl_copystrcontent`.
fn handler_copystrcontent(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let src = bh.registers_r[code[p] as usize];
    let dst = bh.registers_r[code[p + 1] as usize];
    let srcstart = bh.registers_i[code[p + 2] as usize];
    let dststart = bh.registers_i[code[p + 3] as usize];
    let length = bh.registers_i[code[p + 4] as usize];
    if let Some(cpu) = bh.cpu {
        cpu.bh_copystrcontent(src, dst, srcstart, dststart, length);
    }
    Ok(p + 5)
}
/// RPython `blackhole.py:1580-1583` `bhimpl_copyunicodecontent`.
fn handler_copyunicodecontent(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let src = bh.registers_r[code[p] as usize];
    let dst = bh.registers_r[code[p + 1] as usize];
    let srcstart = bh.registers_i[code[p + 2] as usize];
    let dststart = bh.registers_i[code[p + 3] as usize];
    let length = bh.registers_i[code[p + 4] as usize];
    if let Some(cpu) = bh.cpu {
        cpu.bh_copyunicodecontent(src, dst, srcstart, dststart, length);
    }
    Ok(p + 5)
}
fn handler_raise(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let exc = bh.registers_r[code[p] as usize];
    // RPython blackhole.py:169 `_get_method` stores the decoded position
    // back to `self.position` before invoking the bhimpl_*. Required here
    // because `run_inner`'s RaiseException arm calls
    // `handle_exception_in_frame`, which reads `self.position` to find
    // the immediately-following `catch_exception/L` (blackhole.py:396).
    // Without this update the search would start one byte short of the
    // post-operand position.
    bh.position = p + 1;
    Err(DispatchError::RaiseException(exc))
}
fn handler_reraise(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _p: usize,
) -> Result<usize, DispatchError> {
    Err(DispatchError::RaiseException(bh.exception_last_value))
}
/// RPython `blackhole.py:987-991`:
/// ```python
/// @arguments("self", returns="i")
/// def bhimpl_last_exception(self):
///     real_instance = self.exception_last_value
///     assert real_instance
///     return ptr2int(real_instance.typeptr)
/// ```
/// Returns the CLASS POINTER (typeptr) of the caught exception, not the
/// exception object itself. Uses `cpu.bh_classof(obj)` to get the typeptr.
fn handler_last_exception(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let exc_obj = bh.exception_last_value;
    // RPython: ptr2int(real_instance.typeptr) — get class pointer
    let typeptr = if let Some(cpu) = bh.cpu {
        cpu.bh_classof(exc_obj)
    } else {
        exc_obj // fallback: use object pointer as-is if no cpu
    };
    bh.registers_i[code[p] as usize] = typeptr;
    Ok(p + 1)
}
/// RPython `blackhole.py:993-997`:
/// ```python
/// @arguments("self", returns="r")
/// def bhimpl_last_exc_value(self):
///     return cast_opaque_ptr(GCREF, self.exception_last_value)
/// ```
fn handler_last_exc_value(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_r[code[p] as usize] = bh.exception_last_value;
    Ok(p + 1)
}
/// RPython `blackhole.py:976-985`:
/// ```python
/// @arguments("self", "i", "L", "pc", returns="L")
/// def bhimpl_goto_if_exception_mismatch(self, vtable, target, pc):
///     bounding_class = cast_adr_to_ptr(int2adr(vtable), CLASSTYPE)
///     real_instance = self.exception_last_value
///     if rclass.ll_issubclass(real_instance.typeptr, bounding_class):
///         return pc  # match → fall through
///     else:
///         return target  # mismatch → jump
/// ```
/// Uses `cpu.bh_classof` to get the exception's typeptr and compares
/// against the bounding class vtable. For now uses pointer equality
/// (correct for exact match; subclass check needs rclass infrastructure).
fn handler_goto_if_exception_mismatch(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let bounding_vtable = bh.registers_i[code[p] as usize];
    let target = (code[p + 1] as usize) | ((code[p + 2] as usize) << 8);
    let pc = p + 3;
    let exc_obj = bh.exception_last_value;
    let exc_typeptr = if let Some(cpu) = bh.cpu {
        cpu.bh_classof(exc_obj)
    } else {
        exc_obj
    };
    // RPython: rclass.ll_issubclass(real_instance.typeptr, bounding_class).
    // Uses Backend::bh_issubclass for the subclass check.
    let is_match = if let Some(cpu) = bh.cpu {
        cpu.bh_issubclass(exc_typeptr, bounding_vtable)
    } else {
        exc_typeptr == bounding_vtable
    };
    if is_match {
        Ok(pc) // match → fall through
    } else {
        Ok(target) // mismatch → jump to target
    }
}
fn handler_debug_fatalerror(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _p: usize,
) -> Result<usize, DispatchError> {
    panic!("bhimpl_debug_fatalerror");
}
fn handler_cast_ptr_to_int(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[p + 1] as usize] = bh.registers_r[code[p] as usize];
    Ok(p + 2)
}
fn handler_cast_int_to_ptr(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_r[code[p + 1] as usize] = bh.registers_i[code[p] as usize];
    Ok(p + 2)
}
fn handler_current_trace_length(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Ok(p + 1)
}

// ── vable field operations (blackhole.py:1446-1495) ─────────────────
// RPython: fielddescr.get_vinfo().clear_vable_token(struct)
//          return cpu.bh_getfield_gc_*(struct, fielddescr)
// pyre: read_descr_vable_field resolves VableField.index → byte offset via VirtualizableInfo.

fn handler_getfield_vable_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, struct_ptr as *mut u8) };
    }
    let (descr, p) = read_descr_vable_field(bh, code, p + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[p] as usize] = cpu.bh_getfield_gc_i(struct_ptr, &descr);
    Ok(p + 1)
}
fn handler_getfield_vable_i_intbase(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[p] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, struct_ptr as *mut u8) };
    }
    let (descr, p) = read_descr_vable_field(bh, code, p + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[p] as usize] = cpu.bh_getfield_gc_i(struct_ptr, &descr);
    Ok(p + 1)
}
fn handler_getfield_vable_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, struct_ptr as *mut u8) };
    }
    let (descr, p) = read_descr_vable_field(bh, code, p + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[p] as usize] = cpu.bh_getfield_gc_r(struct_ptr, &descr).0 as i64;
    Ok(p + 1)
}
fn handler_getfield_vable_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, struct_ptr as *mut u8) };
    }
    let (descr, p) = read_descr_vable_field(bh, code, p + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[p] as usize] = cpu.bh_getfield_gc_f(struct_ptr, &descr).to_bits() as i64;
    Ok(p + 1)
}

fn handler_setfield_vable_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    let value = bh.registers_i[code[p + 1] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, struct_ptr as *mut u8) };
    }
    let (descr, p) = read_descr_vable_field(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_i(struct_ptr, value, &descr);
    Ok(p)
}
fn handler_setfield_vable_i_intbase(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[p] as usize];
    let value = bh.registers_i[code[p + 1] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, struct_ptr as *mut u8) };
    }
    let (descr, p) = read_descr_vable_field(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_i(struct_ptr, value, &descr);
    Ok(p)
}
fn handler_setfield_vable_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    let value = bh.registers_r[code[p + 1] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, struct_ptr as *mut u8) };
    }
    let (descr, p) = read_descr_vable_field(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_r(struct_ptr, majit_ir::GcRef(value as usize), &descr);
    Ok(p)
}
fn handler_setfield_vable_r_intbase(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[p] as usize];
    let value = bh.registers_r[code[p + 1] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, struct_ptr as *mut u8) };
    }
    let (descr, p) = read_descr_vable_field(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_r(struct_ptr, majit_ir::GcRef(value as usize), &descr);
    Ok(p)
}
fn handler_setfield_vable_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 1] as usize] as u64);
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, struct_ptr as *mut u8) };
    }
    let (descr, p) = read_descr_vable_field(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_f(struct_ptr, value, &descr);
    Ok(p)
}

// ── vable array operations (blackhole.py:1374-1409) ─────────────────
// @arguments("cpu", "r", "i", "d", "d", returns="X")
// Two descriptors: fielddescr (VableArray) + arraydescr (Array).
// RPython: fielddescr.get_vinfo().clear_vable_token(vable)
//          array = cpu.bh_getfield_gc_r(vable, fielddescr)
//          return cpu.bh_getarrayitem_gc_*(array, index, arraydescr)
fn handler_getarrayitem_vable_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, vable as *mut u8) };
    }
    let (field_descr, p) = read_descr_vable_array(bh, code, p + 2);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let array = cpu.bh_getfield_gc_r(vable, &field_descr).0 as i64;
    bh.registers_i[code[p] as usize] = cpu.bh_getarrayitem_gc_i(array, index, array_descr);
    Ok(p + 1)
}
fn handler_getarrayitem_vable_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let nbody_debug = std::env::var_os("PYRE_NBODY_DEBUG").is_some();
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, vable as *mut u8) };
    }
    let (field_descr, p) = read_descr_vable_array(bh, code, p + 2);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let array = cpu.bh_getfield_gc_r(vable, &field_descr).0 as i64;
    let value = cpu.bh_getarrayitem_gc_r(array, index, array_descr).0 as i64;
    if nbody_debug && matches!(index, 5 | 6 | 8 | 9) {
        eprintln!(
            "[nbody-debug][bh-vable-get-r] position={} last_opcode_position={} index={} value={:#x}",
            bh.position, bh.last_opcode_position, index, value as usize
        );
    }
    bh.registers_r[code[p] as usize] = value;
    Ok(p + 1)
}
fn handler_setarrayitem_vable_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_i[code[p + 2] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, vable as *mut u8) };
    }
    let (field_descr, p) = read_descr_vable_array(bh, code, p + 3);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let array = cpu.bh_getfield_gc_r(vable, &field_descr).0 as i64;
    cpu.bh_setarrayitem_gc_i(array, index, value, array_descr);
    Ok(p)
}
fn handler_setarrayitem_vable_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let nbody_debug = std::env::var_os("PYRE_NBODY_DEBUG").is_some();
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_r[code[p + 2] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, vable as *mut u8) };
    }
    let (field_descr, p) = read_descr_vable_array(bh, code, p + 3);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let array = cpu.bh_getfield_gc_r(vable, &field_descr).0 as i64;
    if nbody_debug && matches!(index, 5 | 6 | 8 | 9) {
        eprintln!(
            "[nbody-debug][bh-vable-set-r] position={} last_opcode_position={} index={} value={:#x}",
            bh.position, bh.last_opcode_position, index, value as usize
        );
    }
    cpu.bh_setarrayitem_gc_r(array, index, majit_ir::GcRef(value as usize), array_descr);
    Ok(p)
}
fn handler_arraylen_vable(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, vable as *mut u8) };
    }
    let (field_descr, p) = read_descr_vable_array(bh, code, p + 1);
    let (array_len_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let array = cpu.bh_getfield_gc_r(vable, &field_descr).0 as i64;
    bh.registers_i[code[p] as usize] = cpu.bh_arraylen_gc(array, array_len_descr);
    Ok(p + 1)
}

// ── getarrayitem_raw / setarrayitem_raw (blackhole.py:1343-1365) ────
/// RPython `blackhole.py:1343-1345` `bhimpl_getarrayitem_raw_i`:
/// `return cpu.bh_getarrayitem_raw_i(array, index, arraydescr)`.
/// Raw memory access — NOT GC-managed. Uses unsafe direct read.
fn handler_getarrayitem_raw_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_i[code[p] as usize] as usize; // raw ptr
    let index = bh.registers_i[code[p + 1] as usize] as usize;
    let (descr, p) = read_descr(bh, code, p + 2);
    let item_size = descr.as_offset();
    let offset = index * item_size.max(1);
    let value = unsafe { *((array + offset) as *const i64) };
    bh.registers_i[code[p] as usize] = value;
    Ok(p + 1)
}
/// RPython `blackhole.py:1360-1362` `bhimpl_setarrayitem_raw_i`.
fn handler_setarrayitem_raw_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_i[code[p] as usize] as usize;
    let index = bh.registers_i[code[p + 1] as usize] as usize;
    let value = bh.registers_i[code[p + 2] as usize];
    let (descr, p) = read_descr(bh, code, p + 3);
    let item_size = descr.as_offset();
    let offset = index * item_size.max(1);
    unsafe { *((array + offset) as *mut i64) = value };
    Ok(p)
}

// ── conditional call (blackhole.py:1257-1276) ───────────────────────
fn handler_conditional_call_ir_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let condition = bh.registers_i[code[p] as usize];
    let func = bh.registers_i[code[p + 1] as usize];
    let (ai, p) = read_list_i(bh, code, p + 2);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    if condition != 0 {
        bh.cpu
            .expect("cpu")
            .bh_call_v(func, Some(&ai), Some(&ar), None, &calldescr);
    }
    Ok(p)
}
fn handler_conditional_call_value_ir_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let mut value = bh.registers_i[code[p] as usize];
    let func = bh.registers_i[code[p + 1] as usize];
    let (ai, p) = read_list_i(bh, code, p + 2);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    if value == 0 {
        value = bh
            .cpu
            .expect("cpu")
            .bh_call_i(func, Some(&ai), Some(&ar), None, &calldescr);
    }
    bh.registers_i[code[p] as usize] = value;
    Ok(p + 1)
}
fn handler_conditional_call_value_ir_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let mut value = bh.registers_r[code[p] as usize];
    let func = bh.registers_i[code[p + 1] as usize];
    let (ai, p) = read_list_i(bh, code, p + 2);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    if value == 0 {
        value = bh
            .cpu
            .expect("cpu")
            .bh_call_r(func, Some(&ai), Some(&ar), None, &calldescr)
            .0 as i64;
    }
    bh.registers_r[code[p] as usize] = value;
    Ok(p + 1)
}

// ── list ops (blackhole.py:1195-1219) — compound: getfield_gc_r + getarrayitem ──
fn handler_getlistitem_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (items_descr, p) = read_descr(bh, code, p + 2);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    bh.registers_i[code[p] as usize] = cpu.bh_getarrayitem_gc_i(items, index, array_descr);
    Ok(p + 1)
}
fn handler_getlistitem_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (items_descr, p) = read_descr(bh, code, p + 2);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    bh.registers_r[code[p] as usize] = cpu.bh_getarrayitem_gc_r(items, index, array_descr).0 as i64;
    Ok(p + 1)
}
fn handler_setlistitem_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_i[code[p + 2] as usize];
    let (items_descr, p) = read_descr(bh, code, p + 3);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    cpu.bh_setarrayitem_gc_i(items, index, value, array_descr);
    Ok(p)
}
fn handler_setlistitem_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_r[code[p + 2] as usize];
    let (items_descr, p) = read_descr(bh, code, p + 3);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    cpu.bh_setarrayitem_gc_r(items, index, majit_ir::GcRef(value as usize), array_descr);
    Ok(p)
}

// ── switch (blackhole.py:954-960) ───────────────────────────────────
/// RPython `blackhole.py:954-960`:
/// ```python
/// @arguments("i", "d", "pc", returns="L")
/// def bhimpl_switch(switchvalue, switchdict, pc):
///     assert isinstance(switchdict, SwitchDictDescr)
///     try:
///         return switchdict.dict[switchvalue]
///     except KeyError:
///         return pc
/// ```
fn handler_switch(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let switchvalue = bh.registers_i[code[p] as usize];
    let (descr, pos) = read_descr(bh, code, p + 1);
    // RPython blackhole.py:954-960:
    //   try: return switchdict.dict[switchvalue]
    //   except KeyError: return pc
    if let Some(target) = descr.switch_lookup(switchvalue) {
        Ok(target)
    } else {
        Ok(pos) // fallthrough (KeyError path)
    }
}

// ── check_neg_index / check_resizable_neg_index (blackhole.py:1148-1158) ─
fn handler_check_neg_index(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let mut index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    if index < 0 {
        let cpu = bh.cpu.expect("cpu not set");
        index += cpu.bh_arraylen_gc(array, descr);
    }
    bh.registers_i[code[p] as usize] = index;
    Ok(p + 1)
}
fn handler_check_resizable_neg_index(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let mut index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    if index < 0 {
        let cpu = bh.cpu.expect("cpu not set");
        index += cpu.bh_getfield_gc_i(lst, descr);
    }
    bh.registers_i[code[p] as usize] = index;
    Ok(p + 1)
}

// ── getarrayitem_gc_f / setarrayitem_gc_f ───────────────────────────
// blackhole.py:1336-1337 bhimpl_getarrayitem_gc_f
fn handler_getarrayitem_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[p] as usize] =
        cpu.bh_getarrayitem_gc_f(array, index, descr).to_bits() as i64;
    Ok(p + 1)
}
// blackhole.py:1357-1358 bhimpl_setarrayitem_gc_f
fn handler_setarrayitem_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let (descr, p) = read_descr(bh, code, p + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setarrayitem_gc_f(array, index, value, descr);
    Ok(p)
}
// blackhole.py:1347-1348 bhimpl_getarrayitem_raw_f
fn handler_getarrayitem_raw_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_i[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[p] as usize] =
        cpu.bh_getarrayitem_raw_f(array, index, descr).to_bits() as i64;
    Ok(p + 1)
}
// blackhole.py:1363-1364 bhimpl_setarrayitem_raw_f
fn handler_setarrayitem_raw_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_i[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let (descr, p) = read_descr(bh, code, p + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setarrayitem_raw_f(array, index, value, descr);
    Ok(p)
}
// getfield_raw_r (pure only, blackhole.py:1467-1469)
fn handler_getfield_raw_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[p] as usize];
    let (descr, p) = read_descr(bh, code, p + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[p] as usize] = cpu.bh_getfield_raw_r(struct_ptr, descr).0 as i64;
    Ok(p + 1)
}
// getinteriorfield_gc_f / setinteriorfield_gc_f / setinteriorfield_gc_r
/// RPython `blackhole.py:1417-1419`:
/// `return cpu.bh_getinteriorfield_gc_f(array, index, descr)`
fn handler_getinteriorfield_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[p] as usize] =
        cpu.bh_getinteriorfield_gc_f(array, index, descr).to_bits() as i64;
    Ok(p + 1)
}
fn handler_getinteriorfield_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[p] as usize] = cpu.bh_getinteriorfield_gc_r(array, index, descr).0 as i64;
    Ok(p + 1)
}
fn handler_setinteriorfield_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let (descr, p) = read_descr(bh, code, p + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setinteriorfield_gc_f(array, index, value, descr);
    Ok(p)
}
fn handler_setinteriorfield_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_r[code[p + 2] as usize];
    let (descr, p) = read_descr(bh, code, p + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setinteriorfield_gc_r(array, index, majit_ir::GcRef(value as usize), descr);
    Ok(p)
}
// gc_load_indexed_i/f, gc_store_indexed_i/f (blackhole.py:1518-1540)
fn handler_gc_load_indexed_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let addr = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let scale = bh.registers_i[code[p + 2] as usize];
    let base_ofs = bh.registers_i[code[p + 3] as usize];
    let bytes = bh.registers_i[code[p + 4] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[p + 5] as usize] =
        cpu.bh_gc_load_indexed_i(addr, index, scale, base_ofs, bytes);
    Ok(p + 6)
}
fn handler_gc_load_indexed_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let addr = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let scale = bh.registers_i[code[p + 2] as usize];
    let base_ofs = bh.registers_i[code[p + 3] as usize];
    let bytes = bh.registers_i[code[p + 4] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[p + 5] as usize] = cpu
        .bh_gc_load_indexed_f(addr, index, scale, base_ofs, bytes)
        .to_bits() as i64;
    Ok(p + 6)
}
// blackhole.py:1525-1529 bhimpl_gc_store_indexed_i
fn handler_gc_store_indexed_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "r", "i", "i", "i", "i", "i", "d")
    let addr = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_i[code[p + 2] as usize];
    let scale = bh.registers_i[code[p + 3] as usize];
    let base_ofs = bh.registers_i[code[p + 4] as usize];
    let bytes = bh.registers_i[code[p + 5] as usize];
    let (_, p) = read_descr(bh, code, p + 6);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_gc_store_indexed_i(addr, index, value, scale, base_ofs, bytes);
    Ok(p)
}
// blackhole.py:1531-1535 bhimpl_gc_store_indexed_f
fn handler_gc_store_indexed_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "r", "i", "f", "i", "i", "i", "d")
    let addr = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let scale = bh.registers_i[code[p + 3] as usize];
    let base_ofs = bh.registers_i[code[p + 4] as usize];
    let bytes = bh.registers_i[code[p + 5] as usize];
    let (_, p) = read_descr(bh, code, p + 6);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_gc_store_indexed_f(addr, index, value, scale, base_ofs, bytes);
    Ok(p)
}
// blackhole.py:1504-1509 bhimpl_raw_store_i/f
fn handler_raw_store_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "i", "i", "i", "d")
    let addr = bh.registers_i[code[p] as usize];
    let offset = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_i[code[p + 2] as usize];
    let (descr, p) = read_descr(bh, code, p + 3);
    // blackhole.py:1505-1506: cpu.bh_raw_store_i(addr, offset, newvalue, arraydescr)
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_raw_store_i(addr, offset, value, descr);
    Ok(p)
}
fn handler_raw_store_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "i", "i", "f", "d")
    let addr = bh.registers_i[code[p] as usize];
    let offset = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let (descr, p) = read_descr(bh, code, p + 3);
    // blackhole.py:1510-1511: cpu.bh_raw_store_f(addr, offset, newvalue, arraydescr)
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_raw_store_f(addr, offset, value, descr);
    Ok(p)
}
fn handler_raw_load_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let addr = bh.registers_i[code[p] as usize];
    let offset = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    // blackhole.py:1500-1501: cpu.bh_raw_load_i(addr, offset, arraydescr)
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[p] as usize] = cpu.bh_raw_load_i(addr, offset, descr);
    Ok(p + 1)
}
fn handler_raw_load_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let addr = bh.registers_i[code[p] as usize];
    let offset = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    // blackhole.py:1503-1504: cpu.bh_raw_load_f(addr, offset, arraydescr)
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[p] as usize] = cpu.bh_raw_load_f(addr, offset, descr).to_bits() as i64;
    Ok(p + 1)
}
// newlist / newlist_clear / newlist_hint (blackhole.py:1160-1193)
// RPython: compound allocation: bh_new(structdescr) + setfield + bh_new_array + setfield.
// 4 descriptors: structdescr, lengthdescr, itemsdescr, arraydescr.
fn handler_newlist(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let length = bh.registers_i[code[p] as usize];
    let (structdescr, p) = read_descr(bh, code, p + 1);
    let (lengthdescr, p) = read_descr(bh, code, p);
    let (itemsdescr, p) = read_descr(bh, code, p);
    let (arraydescr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    // blackhole.py:1163: result = cpu.bh_new(structdescr)
    let result = cpu.bh_new(structdescr);
    // blackhole.py:1164: cpu.bh_setfield_gc_i(result, length, lengthdescr)
    cpu.bh_setfield_gc_i(result, length, lengthdescr);
    // blackhole.py:1165-1169: bh_new_array_clear when is_array_of_structs or is_array_of_pointers
    let items = if arraydescr.is_array_of_structs() || arraydescr.is_array_of_pointers() {
        cpu.bh_new_array_clear(length, arraydescr)
    } else {
        cpu.bh_new_array(length, arraydescr)
    };
    // blackhole.py:1170: cpu.bh_setfield_gc_r(result, items, itemsdescr)
    cpu.bh_setfield_gc_r(result, majit_ir::GcRef(items as usize), itemsdescr);
    bh.registers_r[code[p] as usize] = result;
    Ok(p + 1)
}
// blackhole.py:1173-1180: newlist_clear always uses bh_new_array_clear.
fn handler_newlist_clear(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let length = bh.registers_i[code[p] as usize];
    let (structdescr, p) = read_descr(bh, code, p + 1);
    let (lengthdescr, p) = read_descr(bh, code, p);
    let (itemsdescr, p) = read_descr(bh, code, p);
    let (arraydescr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let result = cpu.bh_new(structdescr);
    cpu.bh_setfield_gc_i(result, length, lengthdescr);
    // blackhole.py:1178: items = cpu.bh_new_array_clear(length, arraydescr)
    let items = cpu.bh_new_array_clear(length, arraydescr);
    cpu.bh_setfield_gc_r(result, majit_ir::GcRef(items as usize), itemsdescr);
    bh.registers_r[code[p] as usize] = result;
    Ok(p + 1)
}
// blackhole.py:1182-1193: newlist_hint — length=0, allocate with hint capacity.
fn handler_newlist_hint(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lengthhint = bh.registers_i[code[p] as usize];
    let (structdescr, p) = read_descr(bh, code, p + 1);
    let (lengthdescr, p) = read_descr(bh, code, p);
    let (itemsdescr, p) = read_descr(bh, code, p);
    let (arraydescr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let result = cpu.bh_new(structdescr);
    // blackhole.py:1186: cpu.bh_setfield_gc_i(result, 0, lengthdescr)
    cpu.bh_setfield_gc_i(result, 0, lengthdescr);
    // blackhole.py:1187-1191: bh_new_array_clear when is_array_of_structs or is_array_of_pointers
    let items = if arraydescr.is_array_of_structs() || arraydescr.is_array_of_pointers() {
        cpu.bh_new_array_clear(lengthhint, arraydescr)
    } else {
        cpu.bh_new_array(lengthhint, arraydescr)
    };
    cpu.bh_setfield_gc_r(result, majit_ir::GcRef(items as usize), itemsdescr);
    bh.registers_r[code[p] as usize] = result;
    Ok(p + 1)
}
// blackhole.py:1384-1387 bhimpl_getarrayitem_vable_f
fn handler_getarrayitem_vable_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, vable as *mut u8) };
    }
    let (field_descr, p) = read_descr_vable_array(bh, code, p + 2);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let array = cpu.bh_getfield_gc_r(vable, &field_descr).0 as i64;
    bh.registers_f[code[p] as usize] = cpu
        .bh_getarrayitem_gc_f(array, index, array_descr)
        .to_bits() as i64;
    Ok(p + 1)
}
// blackhole.py:1400-1403 bhimpl_setarrayitem_vable_f
fn handler_setarrayitem_vable_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, vable as *mut u8) };
    }
    let (field_descr, p) = read_descr_vable_array(bh, code, p + 3);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let array = cpu.bh_getfield_gc_r(vable, &field_descr).0 as i64;
    cpu.bh_setarrayitem_gc_f(array, index, value, array_descr);
    Ok(p)
}
// blackhole.py:1204-1206 bhimpl_getlistitem_gc_f
fn handler_getlistitem_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (items_descr, p) = read_descr(bh, code, p + 2);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    bh.registers_f[code[p] as usize] = cpu
        .bh_getarrayitem_gc_f(items, index, array_descr)
        .to_bits() as i64;
    Ok(p + 1)
}
// blackhole.py:1217-1219 bhimpl_setlistitem_gc_f
fn handler_setlistitem_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let (items_descr, p) = read_descr(bh, code, p + 3);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    cpu.bh_setarrayitem_gc_f(items, index, value, array_descr);
    Ok(p)
}
// inline_call — RPython blackhole.py:1278-1319
// RPython: cpu.bh_call_*(adr2int(jitcode.fnaddr), args_i, args_r, args_f, jitcode.calldescr)
// The 'j' argcode reads a JitCode descriptor carrying fnaddr + calldescr.
// pyre: fnaddr is stored in BhDescr::JitCode; calldescr not yet modeled.
// TODO: Full implementation should use jitcode_index for frame-chain push/pop.
fn read_inline_call_jitcode(
    bh: &BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> (usize, i64, BhCallDescr, usize) {
    let (jc_descr, p) = read_descr(bh, code, p);
    match jc_descr {
        BhDescr::JitCode {
            jitcode_index,
            fnaddr,
            calldescr,
        } => (*jitcode_index, *fnaddr, calldescr.clone(), p),
        _ => panic!("expected JitCode descriptor"),
    }
}
fn handler_inline_call_irf_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    // blackhole.py:1304-1307 → bhimpl_inline_call_irf_i.
    bh.registers_i[code[p] as usize] =
        bh.bhimpl_inline_call_irf_i(fnaddr, &ai, &ar, &af, &calldescr);
    Ok(p + 1)
}
fn handler_inline_call_irf_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    // blackhole.py:1308-1311 → bhimpl_inline_call_irf_r.
    bh.registers_r[code[p] as usize] = bh
        .bhimpl_inline_call_irf_r(fnaddr, &ai, &ar, &af, &calldescr)
        .0 as i64;
    Ok(p + 1)
}
fn handler_inline_call_irf_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    // blackhole.py:1312-1315 → bhimpl_inline_call_irf_f.
    bh.registers_f[code[p] as usize] = bh
        .bhimpl_inline_call_irf_f(fnaddr, &ai, &ar, &af, &calldescr)
        .to_bits() as i64;
    Ok(p + 1)
}
fn handler_inline_call_irf_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    // blackhole.py:1316-1319 → bhimpl_inline_call_irf_v.
    bh.bhimpl_inline_call_irf_v(fnaddr, &ai, &ar, &af, &calldescr);
    Ok(p)
}
fn handler_inline_call_ir_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    // blackhole.py:1291-1294 → bhimpl_inline_call_ir_i.
    bh.registers_i[code[p] as usize] = bh.bhimpl_inline_call_ir_i(fnaddr, &ai, &ar, &calldescr);
    Ok(p + 1)
}
fn handler_inline_call_ir_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    // blackhole.py:1295-1298 → bhimpl_inline_call_ir_r.
    bh.registers_r[code[p] as usize] =
        bh.bhimpl_inline_call_ir_r(fnaddr, &ai, &ar, &calldescr).0 as i64;
    Ok(p + 1)
}
fn handler_inline_call_ir_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    // blackhole.py:1299-1302 → bhimpl_inline_call_ir_v.
    bh.bhimpl_inline_call_ir_v(fnaddr, &ai, &ar, &calldescr);
    Ok(p)
}
fn handler_inline_call_r_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    // blackhole.py:1279-1281 → bhimpl_inline_call_r_i.
    bh.registers_i[code[p] as usize] = bh.bhimpl_inline_call_r_i(fnaddr, &ar, &calldescr);
    Ok(p + 1)
}
fn handler_inline_call_r_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    // blackhole.py:1282-1285 → bhimpl_inline_call_r_r.
    bh.registers_r[code[p] as usize] = bh.bhimpl_inline_call_r_r(fnaddr, &ar, &calldescr).0 as i64;
    Ok(p + 1)
}
fn handler_inline_call_r_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    // blackhole.py:1286-1289 → bhimpl_inline_call_r_v.
    bh.bhimpl_inline_call_r_v(fnaddr, &ar, &calldescr);
    Ok(p)
}

/// PRE-EXISTING-ADAPTATION: pyre nested-bytecode `inline_call`.
///
/// Pyre does not compile inlined helpers into separate native functions —
/// trace IR merges them into a single compiled trace.  When a guard
/// fails and the blackhole interpreter resumes the original jitcode,
/// the inline call must be re-interpreted as nested bytecode because
/// no per-helper `fnaddr` exists.  RPython's canonical `inline_call_*`
/// handlers (`handler_inline_call_irf_i` etc.) instead expect a real
/// C-ABI fnaddr stored on `BhDescr::JitCode`.
///
/// This handler is the line-by-line port of the legacy
/// `dispatch_one::BC_INLINE_CALL` arm under the
/// `(bh, code, position) -> Result<usize, _>` signature.  Operand
/// payload (pyre-only):
///   `sub_idx: u16`, `num_args: u16`,
///   `num_args × (kind: u8, caller_src: u16, callee_dst: u16)`,
///   `return_i: u16`, `return_r: u16`, `return_f: u16`
/// `u16::MAX` in any return slot encodes "no caller destination".
///
/// Registered via the pyre-only opname `inline_call_pyre_nested/P`
/// in `pyre_extension_insns()`; canonical `inline_call_*` keys remain
/// quarantined out of `wellknown_bh_insns()`.
fn handler_inline_call_pyre_nested(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let mut p = p;
    let sub_idx = jitcode::read_u16(code, &mut p) as usize;
    let num_args = jitcode::read_u16(code, &mut p) as usize;
    let mut arg_triples = Vec::with_capacity(num_args);
    for _ in 0..num_args {
        let kind = JitArgKind::decode(jitcode::read_u8(code, &mut p));
        let caller_src = jitcode::read_u16(code, &mut p) as usize;
        let callee_dst = jitcode::read_u16(code, &mut p) as usize;
        arg_triples.push((kind, caller_src, callee_dst));
    }
    let return_i = decode_return_slot_at(code, &mut p);
    let return_r = decode_return_slot_at(code, &mut p);
    let return_f = decode_return_slot_at(code, &mut p);

    // blackhole.py:150-157 `j` argcode resolves via `descrs[idx]`
    // asserted to be a JitCode entry; pyre's helper-side
    // `RuntimeBhDescr::JitCode(Arc<JitCode>)` is the analogous slot.
    let sub_jitcode = bh
        .jitcode
        .exec
        .descrs
        .get(sub_idx)
        .and_then(crate::jitcode::RuntimeBhDescr::as_jitcode)
        .unwrap_or_else(|| {
            panic!(
                "BC_INLINE_CALL: descrs[{sub_idx}] is not a JitCode entry \
                 (runtime pool has {} items)",
                bh.jitcode.exec.descrs.len()
            )
        })
        .clone();

    let mut callee = BlackholeInterpreter::for_inline_callee(bh);
    callee.setposition(sub_jitcode, 0);

    for (kind, caller_src, callee_dst) in arg_triples {
        match kind {
            JitArgKind::Int => {
                callee.registers_i[callee_dst] = bh.registers_i[caller_src];
            }
            JitArgKind::Ref => {
                callee.registers_r[callee_dst] = bh.registers_r[caller_src];
            }
            JitArgKind::Float => {
                callee.registers_f[callee_dst] = bh.registers_f[caller_src];
            }
        }
    }

    // RPython `bhimpl_inline_call_*` calls `cpu.bh_call_*(jitcode.fnaddr,
    // ...)` directly, so any `JitException` (`ContinueRunningNormally`,
    // `DoneWithThisFrame*`, `ExitFrameWithExceptionRef`) raised by the
    // callee bubbles up to the portal runner via the C-level call
    // (`blackhole.py:1623`).  pyre's nested interpreter executes the
    // sub-jitcode in-process, so we have to re-raise
    // `ContinueRunningNormally` ourselves; otherwise a callee that hits a
    // recursive merge point silently completes "normally" and the parent
    // frame never sees the portal-restart signal.
    //
    // RPython's generic `dispatch_loop` wrapper (`blackhole.py:171`)
    // sets `self.position = position` AFTER decoding operands/result
    // and BEFORE re-raising, so the parent frame's post-exception
    // inspectors see the post-op cursor.  Mirror that invariant here:
    // sync `bh.position` to the operand-decoded post-op `p` before
    // propagating the JitException.
    if let Some(args) = callee.run() {
        bh.position = p;
        return Err(DispatchError::ContinueRunningNormally(args));
    }

    // RPython `blackhole.py:171` invariant: after the handler has
    // decoded operands and consumed the result payload, every
    // exceptional exit path must update `self.position` to the
    // post-op cursor before re-raising.  Mirror that here for the
    // aborted / got_exception paths — `dispatch_step_with_fallback`
    // (`blackhole.rs:1993`) only writes `self.position = new_pos`
    // when the handler returns `Ok`, so any `Err(LeaveFrame)` that
    // skips the assignment leaves the parent's cursor pre-op.
    if callee.aborted {
        bh.position = p;
        bh.aborted = true;
        return Err(DispatchError::LeaveFrame);
    }
    if callee.got_exception {
        bh.position = p;
        let exc_val = callee.exception_last_value;
        // `handle_exception_in_frame` peeks at `bh.position` to look
        // for an immediately-following `catch_exception/L`; on a
        // successful handler dispatch it moves `bh.position` to the
        // catch target — propagate that.
        if exc_val != 0 && bh.handle_exception_in_frame(exc_val) {
            return Ok(bh.position);
        }
        bh.exception_last_value = exc_val;
        bh.got_exception = true;
        return Err(DispatchError::LeaveFrame);
    }

    if let Some((return_kind, callee_src)) = callee.jitcode.trailing_return_info() {
        match return_kind {
            JitArgKind::Int => {
                let caller_dst = return_i.expect("inline int return missing caller destination");
                bh.registers_i[caller_dst] = callee.registers_i[callee_src as usize];
            }
            JitArgKind::Ref => {
                let caller_dst = return_r.expect("inline ref return missing caller destination");
                bh.registers_r[caller_dst] = callee.registers_r[callee_src as usize];
            }
            JitArgKind::Float => {
                let caller_dst = return_f.expect("inline float return missing caller destination");
                bh.registers_f[caller_dst] = callee.registers_f[callee_src as usize];
            }
        }
    }

    Ok(p)
}

/// Mirror of `BlackholeInterpreter::decode_return_slot` for handler
/// callsites that thread a local cursor instead of mutating
/// `self.position`.  `u16::MAX` encodes the "no caller destination"
/// sentinel.
fn decode_return_slot_at(code: &[u8], cursor: &mut usize) -> Option<usize> {
    let dst = jitcode::read_u16(code, cursor) as usize;
    if dst == u16::MAX as usize {
        None
    } else {
        Some(dst)
    }
}

// recursive_call — stub (needs portal runner)
/// RPython `blackhole.py:1095-1099` `get_portal_runner(jdindex)`:
/// Returns (fnptr, calldescr) from jitdrivers_sd[jdindex].
/// pyre: uses portal_runner_ptr directly (single jitdriver).
///
/// RPython `blackhole.py:1101-1132`:
/// ```python
/// def bhimpl_recursive_call_i(self, jdindex, greens_i, greens_r, greens_f,
///                                            reds_i, reds_r, reds_f):
///     fnptr, calldescr = self.get_portal_runner(jdindex)
///     return self.cpu.bh_call_i(fnptr, greens_i+reds_i, greens_r+reds_r,
///                               greens_f+reds_f, calldescr)
/// ```
/// Read recursive_call args and merge greens+reds per kind.
/// Returns (jdindex, all_i, all_r, all_f, next_position).
fn read_recursive_call_args(
    bh: &BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> (
    usize,
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    usize,
) {
    // RPython `blackhole.py:1101-1132` `bhimpl_recursive_call_*` declares
    // jd_index as `@arguments("self", "i", ...)` — `i` argcode means read
    // through the int-register file. The build-time assembler emits a
    // register-index (into the constant pool tail of `registers_i`), not
    // the jd_index literal byte, so dispatch must deref through the
    // register file to recover the jd_index value.
    let jdindex = bh.registers_i[code[p] as usize] as usize;
    let p = p + 1;
    let (greens_i, p) = read_list_i(bh, code, p);
    let (greens_r, p) = read_list_r(bh, code, p);
    let (greens_f, p) = read_list_f(bh, code, p);
    let (reds_i, p) = read_list_i(bh, code, p);
    let (reds_r, p) = read_list_r(bh, code, p);
    let (reds_f, p) = read_list_f(bh, code, p);
    // The greens+reds concatenation per kind happens inside
    // bhimpl_recursive_call_* (mirroring blackhole.py:1105-1108) — the
    // handlers hand over the raw greens / reds tuples so the bhimpl
    // method owns the merge.
    (
        jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f, p,
    )
}
// blackhole.py:1101-1108 bhimpl_recursive_call_i
fn handler_recursive_call_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f, p) =
        read_recursive_call_args(bh, code, p);
    bh.registers_i[code[p] as usize] = bh.bhimpl_recursive_call_i(
        jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f,
    );
    Ok(p + 1)
}
// blackhole.py:1109-1116 bhimpl_recursive_call_r
fn handler_recursive_call_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f, p) =
        read_recursive_call_args(bh, code, p);
    bh.registers_r[code[p] as usize] = bh
        .bhimpl_recursive_call_r(
            jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f,
        )
        .0 as i64;
    Ok(p + 1)
}
// blackhole.py:1117-1124 bhimpl_recursive_call_f
fn handler_recursive_call_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f, p) =
        read_recursive_call_args(bh, code, p);
    bh.registers_f[code[p] as usize] = bh
        .bhimpl_recursive_call_f(
            jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f,
        )
        .to_bits() as i64;
    Ok(p + 1)
}
// blackhole.py:1125-1132 bhimpl_recursive_call_v
fn handler_recursive_call_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f, p) =
        read_recursive_call_args(bh, code, p);
    bh.bhimpl_recursive_call_v(
        jdindex, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f,
    );
    Ok(p)
}
