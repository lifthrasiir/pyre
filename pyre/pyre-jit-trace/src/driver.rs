//! JIT driver access from pyre-jit-trace.
//!
//! make_green_key is a pure function. driver_pair() is accessed through
//! callbacks since the JIT_DRIVER TLS lives in pyre-jit/eval.rs.

use crate::callbacks;
use crate::state::PyreJitState;

/// RPython green_key = (pycode, next_instr).
/// Each (code, pc) pair has independent warmup counter and compiled loop.
///
/// PRE-EXISTING-ADAPTATION: pyre's green-key is fixed `(W_CodeObject*, pc)`
/// and reduced to a single `u64` identityhash + Signed cast over the
/// two-tuple. RPython's `hash_whatever(TYPE, x)` (`warmstate.py:115`)
/// hashes individual primitives from a structural tuple of green boxes
/// keyed by tuple identity in the JitCell dict. Pyre's two-component
/// hash is a faithful Rust simplification — pyre's portal greens are
/// always exactly `(W_CodeObject, next_instr)`.
#[inline(always)]
pub fn make_green_key(code_ptr: *const (), pc: usize) -> u64 {
    (code_ptr as u64).wrapping_mul(1000003) ^ (pc as u64)
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
