//! JIT driver access from pyre-jit-trace.
//!
//! make_green_key is a pure function. driver_pair() is accessed through
//! callbacks since the JIT_DRIVER TLS lives in pyre-jit/eval.rs.

use crate::callbacks;
use crate::state::PyreJitState;
use majit_ir::{GreenKey, Type};

/// pypy/module/pypyjit/interp_jit.py:67-70:
/// `greens = ['next_instr', 'is_being_profiled', 'pycode']`.
#[inline(always)]
pub fn pypyjit_greenkey(code_ptr: *const (), pc: usize, is_being_profiled: bool) -> GreenKey {
    GreenKey::with_types(
        vec![
            pc as i64,
            if is_being_profiled { 1 } else { 0 },
            code_ptr as i64,
        ],
        vec![Type::Int, Type::Int, Type::Ref],
    )
}

/// Hash bucket for PyPyJitDriver's full typed green tuple.
#[inline(always)]
pub fn make_green_key(code_ptr: *const (), pc: usize, is_being_profiled: bool) -> u64 {
    pypyjit_greenkey(code_ptr, pc, is_being_profiled).hash_u64()
}

#[inline(always)]
pub fn make_green_key_for_frame(frame: &pyre_interpreter::pyframe::PyFrame, pc: usize) -> u64 {
    make_green_key(frame.pycode, pc, frame.get_is_being_profiled())
}

/// Type alias for the JIT driver pair. Must match pyre-jit/eval.rs JitDriverPair.
pub type JitDriverPair = (
    majit_metainterp::JitDriver<PyreJitState>,
    std::sync::Arc<majit_metainterp::virtualizable::VirtualizableInfo>,
);

/// Get the JIT driver pair through callbacks.
#[inline]
pub fn driver_pair() -> &'static mut JitDriverPair {
    let ptr = (callbacks::get().driver_pair)();
    unsafe { &mut *(ptr as *mut JitDriverPair) }
}
