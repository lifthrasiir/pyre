//! Operation execution for the blackhole interpreter.
//!
//! Mirrors RPython's `executor.py`: executes individual JIT IR operations
//! by dispatching on the opcode and computing the result.

use std::collections::HashMap;

use majit_ir::{GcRef, Op, OpCode, OpRef};

use crate::blackhole::ExceptionState;

/// llgraph/runner.py:1245-1251 `execute_guard_class` parity helper:
/// mirrors `gc_ll_descr.get_actual_typeid(gcptr)` (gc.py:624-629)
/// through the thread-local GC seam installed by the active backend.
///
/// For managed objects the active callback reads the GC header word
/// and returns its typeid half-word; for pyre's foreign PyObject
/// layout it consults the `vtable_to_type_id` table populated via
/// `register_vtable_for_type`. Returns `None` when no backend is
/// installed (unit tests dispatching with placeholder integers), which
/// the caller translates into `executor.py:351-358` skip semantics.
fn read_typeid(obj_ptr: i64) -> Option<u32> {
    if obj_ptr <= 0 {
        return None;
    }
    let addr = obj_ptr as usize;
    if addr < 4096 {
        return None;
    }
    majit_gc::get_actual_typeid(GcRef(addr))
}

/// Fast value store for trace execution.
///
/// Op results (non-constant OpRef) → `results` Vec, direct indexed.
/// Constants (constant-namespace OpRef) → `constants` Vec, indexed by const_index.
///
/// Replaces `HashMap<u32, i64>` on the hot path with O(1) Vec indexing.
pub(crate) struct TraceValues {
    /// Op results, indexed by OpRef.0 (operation namespace).
    pub results: Vec<i64>,
    /// Constants, indexed by OpRef.const_index() (constant namespace).
    pub constants: Vec<i64>,
}

impl TraceValues {
    pub fn new(num_ops: usize, num_constants: usize) -> Self {
        Self {
            results: vec![0; num_ops],
            constants: vec![0; num_constants],
        }
    }

    pub fn from_hashmap(map: &HashMap<u32, i64>) -> Self {
        // Index-keyed pool namespace probe (Slice P3 category E):
        // raw u32 keys carry the constant-namespace bit directly, so use
        // the bit-helpers instead of round-tripping through
        // `OpRef::from_raw(k)` which lands on the to-be-retired
        // `OpRef::Untyped` variant.
        let max_op = map
            .keys()
            .filter(|&&k| !OpRef::raw_is_constant(k))
            .max()
            .copied()
            .unwrap_or(0) as usize;
        let max_const = map
            .keys()
            .filter(|&&k| OpRef::raw_is_constant(k))
            .max()
            .map(|&k| OpRef::raw_const_index(k) as usize)
            .unwrap_or(0);
        let mut tv = Self::new(max_op + 1, max_const + 1);
        for (&k, &v) in map {
            tv.set(k, v);
        }
        tv
    }

    #[inline(always)]
    pub fn get(&self, idx: u32) -> i64 {
        if OpRef::raw_is_constant(idx) {
            let ci = OpRef::raw_const_index(idx) as usize;
            if ci < self.constants.len() {
                self.constants[ci]
            } else {
                0
            }
        } else {
            let i = idx as usize;
            if i < self.results.len() {
                self.results[i]
            } else {
                0
            }
        }
    }

    #[inline(always)]
    pub fn set(&mut self, idx: u32, value: i64) {
        if OpRef::raw_is_constant(idx) {
            let ci = OpRef::raw_const_index(idx) as usize;
            if ci >= self.constants.len() {
                self.constants.resize(ci + 1, 0);
            }
            self.constants[ci] = value;
        } else {
            let i = idx as usize;
            if i >= self.results.len() {
                self.results.resize(i + 1, 0);
            }
            self.results[i] = value;
        }
    }

    #[inline(always)]
    pub fn resolve(&self, opref: OpRef) -> i64 {
        self.get(opref.raw())
    }
}

/// Trait for resolving OpRef → i64 values in trace execution.
/// Allows both HashMap (legacy) and TraceValues (fast) backends.
pub(crate) trait ValueStore {
    fn resolve(&self, opref: OpRef) -> i64;
}

impl ValueStore for HashMap<u32, i64> {
    #[inline(always)]
    fn resolve(&self, opref: OpRef) -> i64 {
        self.get(&opref.raw()).copied().unwrap_or(0)
    }
}

impl ValueStore for TraceValues {
    #[inline(always)]
    fn resolve(&self, opref: OpRef) -> i64 {
        self.get(opref.raw())
    }
}

pub(crate) enum OpResult {
    Value(i64),
    Void,
    Finish(Vec<OpRef>),
    Jump(Vec<OpRef>),
    GuardFailed,
    Unsupported(String),
}

pub(crate) fn execute_one(
    op: &Op,
    values: &(impl ValueStore + ?Sized),
    exc: &mut ExceptionState,
) -> OpResult {
    match op.opcode {
        // ── Control flow ──
        OpCode::Label => OpResult::Void,
        OpCode::Finish => OpResult::Finish(op.args.to_vec()),
        OpCode::Jump => OpResult::Jump(op.args.to_vec()),

        // ── Integer arithmetic ──
        OpCode::IntAdd => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_add(b))
        }
        OpCode::IntSub => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_sub(b))
        }
        OpCode::IntMul => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_mul(b))
        }
        OpCode::IntFloorDiv => {
            let (a, b) = binop(values, op);
            if b == 0 {
                OpResult::Value(0)
            } else {
                OpResult::Value(a.wrapping_div(b))
            }
        }
        OpCode::IntMod => {
            let (a, b) = binop(values, op);
            if b == 0 {
                OpResult::Value(0)
            } else {
                OpResult::Value(a.wrapping_rem(b))
            }
        }
        OpCode::IntAnd => {
            let (a, b) = binop(values, op);
            OpResult::Value(a & b)
        }
        OpCode::IntOr => {
            let (a, b) = binop(values, op);
            OpResult::Value(a | b)
        }
        OpCode::IntXor => {
            let (a, b) = binop(values, op);
            OpResult::Value(a ^ b)
        }
        OpCode::IntLshift => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_shl(b as u32))
        }
        OpCode::IntRshift => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_shr(b as u32))
        }
        OpCode::UintRshift => {
            let (a, b) = binop(values, op);
            OpResult::Value((a as u64).wrapping_shr(b as u32) as i64)
        }
        OpCode::IntNeg => {
            let a = unop(values, op);
            OpResult::Value(a.wrapping_neg())
        }
        OpCode::IntInvert => {
            let a = unop(values, op);
            OpResult::Value(!a)
        }
        OpCode::IntSignext => {
            let (a, b) = binop(values, op);
            // Sign extend from b bytes to i64
            let bits = b * 8;
            let shift = 64 - bits;
            OpResult::Value((a << shift) >> shift)
        }

        // ── Integer comparisons ──
        OpCode::IntLt => {
            let (a, b) = binop(values, op);
            OpResult::Value((a < b) as i64)
        }
        OpCode::IntLe => {
            let (a, b) = binop(values, op);
            OpResult::Value((a <= b) as i64)
        }
        OpCode::IntGe => {
            let (a, b) = binop(values, op);
            OpResult::Value((a >= b) as i64)
        }
        OpCode::IntGt => {
            let (a, b) = binop(values, op);
            OpResult::Value((a > b) as i64)
        }
        OpCode::IntEq => {
            let (a, b) = binop(values, op);
            OpResult::Value((a == b) as i64)
        }
        OpCode::IntNe => {
            let (a, b) = binop(values, op);
            OpResult::Value((a != b) as i64)
        }
        OpCode::UintLt => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u64) < (b as u64)) as i64)
        }
        OpCode::UintLe => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u64) <= (b as u64)) as i64)
        }
        OpCode::UintGe => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u64) >= (b as u64)) as i64)
        }
        OpCode::UintGt => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u64) > (b as u64)) as i64)
        }
        OpCode::IntIsZero => {
            let a = unop(values, op);
            OpResult::Value((a == 0) as i64)
        }
        OpCode::IntIsTrue => {
            let a = unop(values, op);
            OpResult::Value((a != 0) as i64)
        }
        OpCode::IntForceGeZero => {
            let a = unop(values, op);
            OpResult::Value(a.max(0))
        }
        OpCode::IntBetween => {
            // int_between(a, b, c) => a <= b < c
            let a = values.resolve(op.args[0]);
            let b = values.resolve(op.args[1]);
            let c = values.resolve(op.args[2]);
            OpResult::Value((a <= b && b < c) as i64)
        }

        // ── Float operations ──
        OpCode::FloatAdd => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a + b) as i64)
        }
        OpCode::FloatSub => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a - b) as i64)
        }
        OpCode::FloatMul => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a * b) as i64)
        }
        OpCode::FloatTrueDiv => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a / b) as i64)
        }
        OpCode::FloatNeg => {
            let a = float_unop(values, op);
            OpResult::Value(f64::to_bits(-a) as i64)
        }
        OpCode::FloatAbs => {
            let a = float_unop(values, op);
            OpResult::Value(f64::to_bits(a.abs()) as i64)
        }
        OpCode::CastFloatToInt => {
            let a = float_unop(values, op);
            OpResult::Value(a as i64)
        }
        OpCode::CastIntToFloat => {
            let a = unop(values, op);
            OpResult::Value(f64::to_bits(a as f64) as i64)
        }

        // ── Guards ──
        OpCode::GuardTrue | OpCode::VecGuardTrue => {
            let a = unop(values, op);
            if a != 0 {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardFalse | OpCode::VecGuardFalse => {
            let a = unop(values, op);
            if a == 0 {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardValue => {
            let (a, b) = binop(values, op);
            if a == b {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardNonnull => {
            let a = unop(values, op);
            if a != 0 {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardIsnull => {
            let a = unop(values, op);
            if a == 0 {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardClass | OpCode::GuardNonnullClass => {
            // llgraph/runner.py:1247-1255 execute_guard_class /
            //   execute_guard_nonnull_class: value.typeptr == klass
            let (a, b) = binop(values, op);
            if a == b {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardSubclass => {
            // llgraph/runner.py:1271-1281 execute_guard_subclass parity:
            //
            //     value = lltype.cast_opaque_ptr(rclass.OBJECTPTR, arg)
            //     expected_class = ...
            //     if (expected_class.subclassrange_min
            //             <= value.typeptr.subclassrange_min
            //             <= expected_class.subclassrange_max):
            //         pass
            //     else:
            //         self.fail_guard(descr)
            //
            // RPython lowers `a <= b <= c` via `op_int_between` /
            // x86/assembler.py:1975-1979 as the unsigned half-open
            // `(b - a) < (c - a)`; the majit backends emit the same
            // half-open form (`majit-backend-cranelift/src/compiler.rs`
            // :6439-6448, `majit-backend-wasm/src/codegen.rs:978`).
            // The executor matches the assembler / rtyper contract
            // rather than the inclusive bounds in the llgraph fallback,
            // because the executor is the trace re-execution path the
            // bridge / blackhole loop relies on — it must agree with
            // the JIT-compiled side bit-for-bit.
            //
            // Object side: `read_typeid` reads the GC header (managed)
            // or the registered vtable (pyre PyObject seam), then
            // `typeid_subclass_range` resolves the preorder lower
            // bound. Expected side: `subclass_range` translates the
            // constant vtable pointer embedded in the guard arg.
            //
            // RPython has no fallback for "typeid lookup failed" —
            // `cast_opaque_ptr` either succeeds or crashes. majit
            // can't crash on synthetic test inputs, so an unresolved
            // typeid translates to `GuardFailed`: if the executor
            // can't prove the guard succeeds, the safe answer is
            // "deopt to interpretation", never "silently allow".
            let (obj_ptr, expected_classptr) = binop(values, op);
            let value_min = read_typeid(obj_ptr)
                .and_then(|tid| majit_gc::typeid_subclass_range(tid).map(|(min, _)| min));
            let expected = majit_gc::subclass_range(expected_classptr as usize);
            match (value_min, expected) {
                (Some(vm), Some((emin, emax))) => {
                    if emin <= vm && vm < emax {
                        OpResult::Void
                    } else {
                        OpResult::GuardFailed
                    }
                }
                _ => OpResult::GuardFailed,
            }
        }
        OpCode::GuardNoOverflow => {
            if exc.ovf_flag {
                OpResult::GuardFailed
            } else {
                OpResult::Void
            }
        }
        OpCode::GuardOverflow => {
            if !exc.ovf_flag {
                OpResult::GuardFailed
            } else {
                OpResult::Void
            }
        }
        OpCode::GuardNotForced | OpCode::GuardNotForced2 => {
            // In blackhole, check if a call set an exception (simulated force).
            if exc.is_pending() {
                OpResult::GuardFailed
            } else {
                OpResult::Void
            }
        }
        OpCode::GuardNotInvalidated | OpCode::GuardFutureCondition => OpResult::Void,
        OpCode::GuardAlwaysFails => OpResult::GuardFailed,
        OpCode::GuardNoException => {
            if exc.is_pending() {
                OpResult::GuardFailed
            } else {
                OpResult::Void
            }
        }
        OpCode::GuardException => {
            // Guard expects an exception of a specific class.
            // arg(0) is the expected exception class.
            if exc.is_pending() {
                let expected_class = values.resolve(op.args[0]);
                if exc.exc_class == expected_class {
                    // Match — return the exception value and clear exception state.
                    let (_, val) = exc.clear();
                    return OpResult::Value(val);
                }
            }
            OpResult::GuardFailed
        }
        OpCode::GuardGcType => {
            // llgraph/runner.py:1257-1261 execute_guard_gc_type literal:
            //
            //     assert isinstance(typeid, TypeIDSymbolic)
            //     TYPE = arg._obj.container._TYPE
            //     if TYPE != typeid.STRUCT_OR_ARRAY:
            //         self.fail_guard(descr)
            //
            // majit reads `arg`'s typeid from the GC header and
            // compares it against the immediate `typeid` constant
            // carried in arg[1] (rewrite.py emits a `ConstInt(type_id)`
            // there). RPython has no fallback for "typeid unreadable":
            // `arg._obj.container._TYPE` either succeeds or crashes.
            // For majit, an unreadable typeid translates to
            // `GuardFailed` so the bridge / blackhole loop deopts to
            // interpretation rather than silently allowing the guard
            // to pass.
            let (obj_ptr, expected_tid) = binop(values, op);
            match read_typeid(obj_ptr) {
                Some(actual_tid) => {
                    if actual_tid as i64 == expected_tid {
                        OpResult::Void
                    } else {
                        OpResult::GuardFailed
                    }
                }
                None => OpResult::GuardFailed,
            }
        }
        OpCode::GuardIsObject => {
            // llgraph/runner.py:1263-1269 execute_guard_is_object literal:
            //
            //     TYPE = arg._obj.container._TYPE
            //     while TYPE is not rclass.OBJECT:
            //         if not isinstance(TYPE, lltype.GcStruct):
            //             self.fail_guard(descr)
            //             return
            //         _, TYPE = TYPE._first_struct()
            //
            // The walk inspects each `_first_struct` ancestor for
            // GcStruct-ness; the loop returns silently when it
            // reaches `rclass.OBJECT`, otherwise fails the guard.
            // majit's TYPE_INFO table pre-computes the same answer
            // (`T_IS_RPYTHON_INSTANCE`, gctypelayout.py:642), so the
            // equivalent is a single seam lookup. Same fallback rule
            // as the other typeid guards: an unreadable typeid means
            // `GuardFailed`, never `Void`. Earlier majit revisions
            // skipped to `Void` here, which silently degraded the
            // guard into a nonnull check — that's the regression the
            // user flagged.
            let obj_ptr = unop(values, op);
            match read_typeid(obj_ptr).and_then(majit_gc::typeid_is_object) {
                Some(true) => OpResult::Void,
                Some(false) => OpResult::GuardFailed,
                None => OpResult::GuardFailed,
            }
        }

        // ── SameAs / Copy ──
        OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF => {
            let a = same_as_value(values, op);
            OpResult::Value(a)
        }

        // ── No-op markers ──
        OpCode::Keepalive
        | OpCode::ForceSpill
        | OpCode::VirtualRefFinish
        | OpCode::RecordExactClass
        | OpCode::RecordExactValueR
        | OpCode::RecordExactValueI
        | OpCode::RecordKnownResult
        | OpCode::QuasiimmutField
        | OpCode::AssertNotNone
        | OpCode::IncrementDebugCounter => OpResult::Void,

        // ── ForceToken ──
        OpCode::ForceToken => {
            // Return a dummy token in blackhole mode
            OpResult::Value(0)
        }

        // ── Exception operations ──
        OpCode::SaveException => {
            // Return the pending exception value.
            OpResult::Value(exc.exc_value)
        }
        OpCode::SaveExcClass => {
            // Return the pending exception class.
            OpResult::Value(exc.exc_class)
        }
        OpCode::RestoreException => {
            // Restore exception state from (class, value) args.
            let cls = values.resolve(op.args[0]);
            let val = values.resolve(op.args[1]);
            exc.set(cls, val);
            OpResult::Void
        }
        OpCode::CheckMemoryError => {
            // If the allocation returned null, set a MemoryError exception.
            let ptr = values.resolve(op.args[0]);
            if ptr == 0 {
                // Set a generic memory error (class=1 by convention).
                exc.set(1, 0);
            }
            OpResult::Void
        }

        // ── Overflow arithmetic ──
        // executor.py: do_int_add_ovf/sub_ovf/mul_ovf set ovf_flag on overflow, return 0
        OpCode::IntAddOvf => {
            let (a, b) = binop(values, op);
            exc.ovf_flag = false;
            match a.checked_add(b) {
                Some(z) => OpResult::Value(z),
                None => {
                    exc.ovf_flag = true;
                    OpResult::Value(a.wrapping_add(b))
                }
            }
        }
        OpCode::IntSubOvf => {
            let (a, b) = binop(values, op);
            exc.ovf_flag = false;
            match a.checked_sub(b) {
                Some(z) => OpResult::Value(z),
                None => {
                    exc.ovf_flag = true;
                    OpResult::Value(a.wrapping_sub(b))
                }
            }
        }
        OpCode::IntMulOvf => {
            let (a, b) = binop(values, op);
            exc.ovf_flag = false;
            match a.checked_mul(b) {
                Some(z) => OpResult::Value(z),
                None => {
                    exc.ovf_flag = true;
                    OpResult::Value(a.wrapping_mul(b))
                }
            }
        }

        // ── Float comparisons ──
        OpCode::FloatLt => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a < b) as i64)
        }
        OpCode::FloatLe => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a <= b) as i64)
        }
        OpCode::FloatGt => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a > b) as i64)
        }
        OpCode::FloatGe => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a >= b) as i64)
        }
        OpCode::FloatEq => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a == b) as i64)
        }
        OpCode::FloatNe => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a != b) as i64)
        }

        // ── Additional float operations ──
        OpCode::FloatFloorDiv => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits((a / b).floor()) as i64)
        }
        OpCode::FloatMod => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a % b) as i64)
        }

        // ── VirtualRef (pass through in blackhole) ──
        OpCode::VirtualRefI | OpCode::VirtualRefR => {
            let a = unop(values, op);
            OpResult::Value(a)
        }

        // ── Call operations (pass through with concrete values) ──
        // In blackhole mode, calls should re-execute with concrete args.
        // For now, we handle CALL_PURE variants (can evaluate if all args known).
        // Call operations — return placeholder 0 in no-memory path.
        // The execute_one_with_memory path handles actual dispatch.
        OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF => OpResult::Value(0),
        OpCode::CallPureN => OpResult::Void,
        OpCode::CallI
        | OpCode::CallR
        | OpCode::CallF
        | OpCode::CallMayForceI
        | OpCode::CallMayForceR
        | OpCode::CallMayForceF
        | OpCode::CallReleaseGilI
        | OpCode::CallReleaseGilF => OpResult::Value(0),
        OpCode::CallN | OpCode::CallMayForceN | OpCode::CallReleaseGilN => OpResult::Void,

        // ── Memory access (raw) ──
        // In a full blackhole, these would dereference actual pointers.
        // For now, return 0 as placeholder.
        OpCode::GetfieldGcI
        | OpCode::GetfieldGcR
        | OpCode::GetfieldGcF
        | OpCode::GetfieldRawI
        | OpCode::GetfieldRawR
        | OpCode::GetfieldRawF
        | OpCode::GetfieldGcPureI
        | OpCode::GetfieldGcPureR
        | OpCode::GetfieldGcPureF => OpResult::Value(0),
        OpCode::SetfieldGc | OpCode::SetfieldRaw => {
            let resolved_args: Vec<i64> = op.args.iter().map(|&r| values.resolve(r)).collect();
            if let (Some(&obj_ptr), Some(&value)) = (resolved_args.first(), resolved_args.get(1)) {
                if let Some(fd) = op.descr.as_ref().and_then(|d| d.as_field_descr()) {
                    let offset = fd.offset();
                    if obj_ptr != 0 {
                        unsafe {
                            let dest = (obj_ptr as *mut u8).add(offset) as *mut i64;
                            *dest = value;
                        }
                    }
                }
            }
            OpResult::Void
        }

        // ── Array access ──
        OpCode::GetarrayitemGcI
        | OpCode::GetarrayitemGcR
        | OpCode::GetarrayitemGcF
        | OpCode::GetarrayitemRawI
        | OpCode::GetarrayitemRawR
        | OpCode::GetarrayitemRawF
        | OpCode::GetarrayitemGcPureI
        | OpCode::GetarrayitemGcPureR
        | OpCode::GetarrayitemGcPureF => OpResult::Value(0),
        OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => OpResult::Void,

        // ── Array/string length ──
        OpCode::ArraylenGc => OpResult::Value(0),
        OpCode::Strlen | OpCode::Unicodelen => OpResult::Value(0),

        // ── Allocation ──
        // Allocate a real object so SetfieldGc can write fields.
        // IR blackhole may encounter New+SetfieldGc when the trace
        // contains unoptimized allocation (e.g. result_type=Ref finish).
        OpCode::New | OpCode::NewWithVtable => {
            let size = op
                .descr
                .as_ref()
                .and_then(|d| d.as_size_descr())
                .map_or(16, |sd| sd.size());
            let layout = std::alloc::Layout::from_size_align(size, 8)
                .unwrap_or(std::alloc::Layout::new::<[u8; 16]>());
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            OpResult::Value(ptr as i64)
        }
        OpCode::NewArray | OpCode::NewArrayClear => OpResult::Value(0),
        OpCode::Newstr | OpCode::Newunicode => OpResult::Value(0),

        // ── String/char access ──
        OpCode::Strgetitem | OpCode::Unicodegetitem => OpResult::Value(0),
        OpCode::Strsetitem | OpCode::Unicodesetitem => OpResult::Void,
        OpCode::Strhash | OpCode::Unicodehash => OpResult::Value(0),

        // ── Interior field access ──
        OpCode::GetinteriorfieldGcI | OpCode::GetinteriorfieldGcR | OpCode::GetinteriorfieldGcF => {
            OpResult::Value(0)
        }
        OpCode::SetinteriorfieldGc | OpCode::SetinteriorfieldRaw => OpResult::Void,

        // ── Raw memory ──
        OpCode::RawStore => OpResult::Void,
        OpCode::RawLoadI | OpCode::RawLoadF => OpResult::Value(0),

        // ── GC write barriers (no-op in blackhole) ──
        OpCode::CondCallGcWb | OpCode::CondCallGcWbArray | OpCode::ZeroArray => OpResult::Void,

        // ── Nursery allocation (no-op in blackhole) ──
        OpCode::CallMallocNursery
        | OpCode::CallMallocNurseryVarsize
        | OpCode::CallMallocNurseryVarsizeFrame
        | OpCode::NurseryPtrIncrement => OpResult::Value(0),

        // ── Pointer comparisons/casts ──
        OpCode::PtrEq | OpCode::InstancePtrEq => {
            let (a, b) = binop(values, op);
            OpResult::Value((a == b) as i64)
        }
        OpCode::PtrNe | OpCode::InstancePtrNe => {
            let (a, b) = binop(values, op);
            OpResult::Value((a != b) as i64)
        }
        OpCode::CastPtrToInt => {
            let a = unop(values, op);
            OpResult::Value(a)
        }
        OpCode::CastIntToPtr | OpCode::CastOpaquePtr => {
            let a = unop(values, op);
            OpResult::Value(a)
        }

        // ── CALL_ASSEMBLER: cannot be executed in the blackhole ──
        // Must fall back to force_fn which creates a proper callee frame.
        OpCode::CallAssemblerI | OpCode::CallAssemblerR | OpCode::CallAssemblerF => {
            OpResult::Unsupported("CallAssembler requires force_fn fallback".to_string())
        }
        OpCode::CallAssemblerN => {
            OpResult::Unsupported("CallAssemblerN requires force_fn fallback".to_string())
        }

        // ── Cond call (conditional function call) ──
        OpCode::CondCallValueI | OpCode::CondCallValueR => OpResult::Value(0),
        OpCode::CondCallN => OpResult::Void,

        // ── Thread-local ref ──
        OpCode::ThreadlocalrefGet => OpResult::Value(0),

        // ── Loopinvariant calls ──
        OpCode::CallLoopinvariantI | OpCode::CallLoopinvariantR | OpCode::CallLoopinvariantF => {
            OpResult::Value(0)
        }
        OpCode::CallLoopinvariantN => OpResult::Void,

        // ── GC loads ──
        OpCode::GcLoadI | OpCode::GcLoadR | OpCode::GcLoadF => OpResult::Value(0),
        OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF => {
            OpResult::Value(0)
        }
        OpCode::GcStore | OpCode::GcStoreIndexed => OpResult::Void,

        // ── Vec loads/stores ──
        OpCode::VecLoadI | OpCode::VecLoadF => OpResult::Value(0),
        OpCode::VecStore => OpResult::Void,

        // ── Vector arithmetic (scalar emulation in blackhole) ──
        OpCode::VecIntAdd => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_add(b))
        }
        OpCode::VecIntSub => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_sub(b))
        }
        OpCode::VecIntMul => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_mul(b))
        }
        OpCode::VecIntAnd => {
            let (a, b) = binop(values, op);
            OpResult::Value(a & b)
        }
        OpCode::VecIntOr => {
            let (a, b) = binop(values, op);
            OpResult::Value(a | b)
        }
        OpCode::VecIntXor => {
            let (a, b) = binop(values, op);
            OpResult::Value(a ^ b)
        }
        OpCode::VecFloatAdd => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a + b) as i64)
        }
        OpCode::VecFloatSub => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a - b) as i64)
        }
        OpCode::VecFloatMul => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a * b) as i64)
        }
        OpCode::VecFloatTrueDiv => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a / b) as i64)
        }
        OpCode::VecFloatNeg => {
            let a = float_unop(values, op);
            OpResult::Value(f64::to_bits(-a) as i64)
        }
        OpCode::VecFloatAbs => {
            let a = float_unop(values, op);
            OpResult::Value(f64::to_bits(a.abs()) as i64)
        }

        // ── Vector comparisons (scalar emulation) ──
        OpCode::VecFloatEq => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a == b) as i64)
        }
        OpCode::VecFloatNe => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a != b) as i64)
        }
        OpCode::VecFloatXor => {
            let (a, b) = binop(values, op);
            OpResult::Value(a ^ b)
        }
        OpCode::VecIntIsTrue => {
            let a = unop(values, op);
            OpResult::Value((a != 0) as i64)
        }
        OpCode::VecIntNe => {
            let (a, b) = binop(values, op);
            OpResult::Value((a != b) as i64)
        }
        OpCode::VecIntEq => {
            let (a, b) = binop(values, op);
            OpResult::Value((a == b) as i64)
        }
        OpCode::VecIntSignext => {
            let (a, b) = binop(values, op);
            let bits = b * 8;
            let shift = 64 - bits;
            OpResult::Value((a << shift) >> shift)
        }

        // ── Vector casts (scalar emulation) ──
        OpCode::VecCastFloatToInt => {
            let a = float_unop(values, op);
            OpResult::Value(a as i64)
        }
        OpCode::VecCastIntToFloat => {
            let a = unop(values, op);
            OpResult::Value(f64::to_bits(a as f64) as i64)
        }
        OpCode::VecCastFloatToSinglefloat => {
            let a = float_unop(values, op);
            let f32_val = a as f32;
            OpResult::Value(f32_val.to_bits() as i64)
        }
        OpCode::VecCastSinglefloatToFloat => {
            let a = unop(values, op);
            let f32_val = f32::from_bits(a as u32);
            OpResult::Value(f64::to_bits(f32_val as f64) as i64)
        }

        // ── Vector pack/unpack/expand (scalar emulation) ──
        OpCode::VecI => OpResult::Value(0),
        OpCode::VecF => OpResult::Value(f64::to_bits(0.0) as i64),
        OpCode::VecUnpackI | OpCode::VecUnpackF => {
            // unpack(vec, lane, count) -> return vec (first scalar)
            let a = unop(values, op);
            OpResult::Value(a)
        }
        OpCode::VecPackI | OpCode::VecPackF => {
            // pack(vec, scalar, lane, count) -> return scalar
            let scalar = values.resolve(op.args[1]);
            OpResult::Value(scalar)
        }
        OpCode::VecExpandI | OpCode::VecExpandF => {
            // expand(scalar) -> return scalar
            let a = unop(values, op);
            OpResult::Value(a)
        }

        // ── String/unicode copy ──
        OpCode::Copystrcontent | OpCode::Copyunicodecontent => OpResult::Void,

        // ── Misc conversions ──
        OpCode::UintMulHigh => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u64 as u128 * b as u64 as u128) >> 64) as i64)
        }
        OpCode::CastFloatToSinglefloat => {
            let a = float_unop(values, op);
            let f32_val = a as f32;
            OpResult::Value(f32_val.to_bits() as i64)
        }
        OpCode::CastSinglefloatToFloat => {
            let a = unop(values, op);
            let f32_val = f32::from_bits(a as u32);
            OpResult::Value(f64::to_bits(f32_val as f64) as i64)
        }
        OpCode::ConvertFloatBytesToLonglong => {
            let a = unop(values, op);
            OpResult::Value(a) // f64 bits already stored as i64
        }
        OpCode::ConvertLonglongBytesToFloat => {
            let a = unop(values, op);
            OpResult::Value(a) // i64 bits reinterpreted as f64
        }

        // ── Debug / portal frame markers ──
        OpCode::DebugMergePoint
        | OpCode::EnterPortalFrame
        | OpCode::LeavePortalFrame
        | OpCode::JitDebug => OpResult::Void,

        // ── LoadFromGcTable / LoadEffectiveAddress ──
        OpCode::LoadFromGcTable | OpCode::LoadEffectiveAddress => OpResult::Value(0),

        // All OpCode variants are explicitly handled above.
        // This arm is unreachable but kept for forward-compatibility
        // when new opcodes are added to the IR.
        #[allow(unreachable_patterns)]
        other => OpResult::Unsupported(format!(
            "blackhole: opcode {:?} has no interpreter handler",
            other
        )),
    }
}

pub(crate) fn binop(values: &(impl ValueStore + ?Sized), op: &Op) -> (i64, i64) {
    let a = values.resolve(op.args[0]);
    let b = values.resolve(op.args[1]);
    (a, b)
}

pub(crate) fn unop(values: &(impl ValueStore + ?Sized), op: &Op) -> i64 {
    values.resolve(op.args[0])
}

pub(crate) fn same_as_value(values: &(impl ValueStore + ?Sized), op: &Op) -> i64 {
    if op.num_args() > 0 {
        unop(values, op)
    } else if !op.pos.is_none() {
        values.resolve(op.pos)
    } else {
        0
    }
}

pub(crate) fn float_binop(values: &(impl ValueStore + ?Sized), op: &Op) -> (f64, f64) {
    let a = f64::from_bits(values.resolve(op.args[0]) as u64);
    let b = f64::from_bits(values.resolve(op.args[1]) as u64);
    (a, b)
}

pub(crate) fn float_unop(values: &(impl ValueStore + ?Sized), op: &Op) -> f64 {
    f64::from_bits(values.resolve(op.args[0]) as u64)
}

/// rpython/jit/metainterp/executor.py:524-528 `execute_varargs(cpu, metainterp, opnum, argboxes, descr)`.
///
/// ```python
/// def execute_varargs(cpu, metainterp, opnum, argboxes, descr):
///     # only for opnums with a variable arity (calls, typically)
///     check_descr(descr)
///     func = get_execute_function(opnum, -1, True)
///     return func(cpu, metainterp, argboxes, descr)
/// ```
///
/// For CALL_* opcodes, `func` ultimately calls `cpu.bh_call_*(funcaddr,
/// args)`.  Pyre routes every arm through `dispatch::call_int_function`
/// / `dispatch::call_void_function` using the concrete values carried
/// alongside each typed argbox.  The Float arm shares the i64-return
/// ABI: helper concrete pointers built by `#[jit_module]` pre-pack the
/// f64 result via `f64::to_bits` (majit-macros/src/lib.rs:194), and
/// callers recover the f64 with `f64::from_bits(resvalue as u64)`
/// (pyjitpl/mod.rs:8901-8902) — bit-identical to the BC_CALL_FLOAT
/// family in blackhole.rs:2349-2371 which uses the same convention.
/// `argboxes[0]` is the funcbox (carrying the function pointer in its
/// `i64` slot) and the remaining slots are the typed call arguments.
///
/// `metainterp` mirrors RPython's `self` parameter: helper-side
/// exceptions published on the `BH_LAST_EXC_VALUE` thread-local seam
/// (the convention `bh_call_fn_impl` and friends use) are transcribed
/// onto `metainterp.last_exc_value` before returning, the same way
/// RPython's cpu auto-publishes onto `cpu.last_exc_value`.  Pyre has
/// no separate `cpu` object; the metainterp owns the field directly.
///
/// Returns the concrete result value as an `i64` — for void-returning
/// calls returns `0` (caller ignores it); for float-returning calls
/// the i64 carries the f64 bits via `f64::to_bits` and the caller
/// must unpack with `f64::from_bits(resvalue as u64)`.  When the
/// helper raises (BH_LAST_EXC_VALUE seam fires), the post-call hook
/// transcribes onto `metainterp.last_exc_value` and overrides the
/// result with `0` — matching `executor.py:52-78`'s neutral-zero
/// behavior across INT, REF (NULL), FLOAT (longlong.ZEROF), and VOID.
pub fn execute_varargs<M: Clone>(
    metainterp: &mut crate::pyjitpl::MetaInterp<M>,
    opnum: OpCode,
    argboxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
    descr: &dyn majit_ir::descr::CallDescr,
) -> i64 {
    debug_assert!(opnum.is_call(), "execute_varargs requires a call opcode");
    // RPython's `cpu` parameter is the seam through which helper-side
    // exceptions reach `metainterp.last_exc_value` (`cpu.bh_call_*`
    // writes onto `cpu.last_exception` and the metainterp's
    // post-execute hook copies it).  Pyre has no separate cpu; we use
    // the `BH_LAST_EXC_VALUE` thread-local that production helpers
    // (`bh_call_fn_impl` etc.) publish on, mirroring the same convention
    // every blackhole.rs CALL_* arm uses (blackhole.rs:2270-2392).
    // Clear before dispatch so a stale value from a prior call cannot
    // bleed into this one.
    crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    // Inner closure carries every existing return path so the outer
    // wrapper can run BH_LAST_EXC_VALUE transcription regardless of
    // which arm fires.
    let result = (|| -> i64 {
        // COND_CALL / COND_CALL_VALUE_* layout (pyjitpl.py:2128-2151):
        //   argboxes[0] = condbox / valuebox
        //   argboxes[1] = funcbox
        //   argboxes[2..] = call args
        // RPython's executor handles cond-call dispatch via a per-opcode
        // execute function (`do_cond_call_*`); pyre inlines the same
        // semantics here.
        //
        // Two distinct shapes (blackhole.py:1257-1276):
        //   bhimpl_conditional_call_ir_v(condition, func, ...):
        //       if condition: cpu.bh_call_v(func, ...)        # void
        //   bhimpl_conditional_call_value_ir_{i,r}(value, func, ...):
        //       if value == 0: value = cpu.bh_call_*(func, ...)
        //       return value
        if matches!(opnum, OpCode::CondCallN) {
            debug_assert!(
                argboxes.len() >= 2,
                "COND_CALL_N requires [condbox, funcbox, *args]",
            );
            let cond = argboxes[0].2;
            if cond == 0 {
                // condition false → skip the call.
                return 0;
            }
            let func_ptr = argboxes[1].2 as *const ();
            let concrete_args: Vec<i64> = argboxes[2..].iter().map(|(_, _, c)| *c).collect();
            crate::pyjitpl::call_void_function(func_ptr, &concrete_args);
            return 0;
        }
        if matches!(opnum, OpCode::CondCallValueI | OpCode::CondCallValueR) {
            debug_assert!(
                argboxes.len() >= 2,
                "COND_CALL_VALUE_* requires [valuebox, funcbox, *args]",
            );
            let value = argboxes[0].2;
            if value != 0 {
                // blackhole.py:1267 / 1274: nonzero `value` short-circuits
                // and returns the existing value without calling.
                return value;
            }
            // value == 0 → call and return the call's result.
            let func_ptr = argboxes[1].2 as *const ();
            let concrete_args: Vec<i64> = argboxes[2..].iter().map(|(_, _, c)| *c).collect();
            return crate::pyjitpl::call_int_function(func_ptr, &concrete_args);
        }
        debug_assert!(
            !argboxes.is_empty(),
            "execute_varargs: argboxes must include funcbox at slot 0",
        );
        let func_ptr = argboxes[0].2 as *const ();
        let concrete_args: Vec<i64> = argboxes[1..].iter().map(|(_, _, c)| *c).collect();
        match descr.result_type() {
            // RPython dispatches Int and Ref through the same backend
            // primitive cpu.bh_call_i (returns i64); pyre's
            // call_int_function does the same — Ref is bit-identical to Int
            // at the ABI level.
            majit_ir::Type::Int | majit_ir::Type::Ref => {
                crate::pyjitpl::call_int_function(func_ptr, &concrete_args)
            }
            majit_ir::Type::Void => {
                crate::pyjitpl::call_void_function(func_ptr, &concrete_args);
                0
            }
            majit_ir::Type::Float => {
                // pyjitpl.py:2119 — CALL_F dispatches through cpu.bh_call_f.
                // Caller-contract: every path that reaches this arm carries
                // `funcbox.2` as a hand-written or `#[jit_module]`-generated
                // function pointer with i64-return ABI:
                //   * `do_recursive_call` (mod.rs:11148-11152) sets funcbox.2
                //     to `targetjitdriver_sd.portal_runner_adr`.  Pyre's
                //     portal entry is `bh_portal_runner(all_i, all_r, all_f)
                //     -> i64` (pyre-jit/src/call_jit.rs:467); it never
                //     declares an f64 return.
                //   * `#[jit_module]` (majit-macros/src/lib.rs:267) emits a
                //     Float helper's `concrete_ptr` as `extern "C" fn(...)
                //     -> i64` with the f64 result pre-packed via
                //     `f64::to_bits`; the f64-ABI `trace_ptr` is consumed
                //     only by pyre-jit-trace's `TraceCtx::call_may_force_*`
                //     family, which has its own seam and never reaches
                //     this arm.
                // Therefore route through `call_int_function` and let the
                // caller recover the f64 via `f64::from_bits` when needed
                // — bit-identical to blackhole.rs:2349-2371 BC_CALL_FLOAT
                // family which makes the same ABI choice for the same
                // reason (`registers_f` is an i64-carrier mirroring
                // RPython's `longlong.ZEROF` packing).
                crate::pyjitpl::call_int_function(func_ptr, &concrete_args)
            }
        }
    })();
    // Mirror RPython's executor.py:52-78 post-call exception flow:
    // each `cpu.bh_call_*` arm wraps the call in `try: ... except
    // Exception as e: metainterp.execute_raised(e); result = ZERO`.
    // Pyre observes the same condition through the BH_LAST_EXC_VALUE
    // thread-local seam.  Two pieces have to fire together:
    //   1. `metainterp.execute_raised(bh_exc, constant=False)` — sets
    //      `last_exc_value` AND clears `class_of_last_exc_is_const`,
    //      so a stale `True` from a prior GUARD_EXCEPTION cannot make
    //      `handle_possible_exception` treat the new exception's
    //      class as constant (pyjitpl.py:2745-2755).
    //   2. Override the returned value with the type's neutral zero —
    //      `make_result_of_lastop` (pyjitpl/mod.rs:8893) snapshots the
    //      concrete result *before* `handle_possible_exception` runs,
    //      so leaving the helper's return value in place can pin a
    //      garbage value into the resume snapshot.  `i64 == 0` covers
    //      INT=0, REF=NULL, and FLOAT=longlong.ZEROF (0.0_f64.to_bits()
    //      as i64 == 0); VOID callers ignore the slot.
    let bh_exc = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| {
        let v = c.get();
        c.set(0);
        v
    });
    if bh_exc != 0 {
        metainterp.execute_raised(bh_exc, false);
        return 0;
    }
    result
}
