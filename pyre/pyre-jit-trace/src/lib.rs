//! pyre-jit-trace: Trace-time JIT for pyre.
//!
//! This crate contains MIFrame (the meta-interpreter frame) and all
//! trace-time logic. It is compiled as a separate compilation unit
//! from pyre-jit's eval_loop_jit to prevent MIFrame's monomorphization
//! of `execute_opcode_step<E>` from bloating the eval loop's codegen.

pub mod assembler;
pub mod callbacks;
pub mod canonical_bridge;
pub mod descr;
pub mod driver;
pub mod frame_layout;
pub mod helpers;
pub mod jitcode_dispatch;
pub mod jitcode_runtime;
pub mod liveness;
pub mod metainterp;
pub mod pyjitcode;
pub mod shadow_walker;
pub mod state;
pub mod super_inst_expand;
mod trace_opcode;
pub use pyjitcode::{PyJitCode, PyJitCodeMetadata};
pub mod trace;
pub mod virtualizable_gen;
pub mod virtualizable_spec;

/// Auto-generated trace functions from majit-translate.
#[allow(dead_code, unsafe_op_in_unsafe_fn, unused_imports, unused_variables)]
pub mod generated {
    use pyre_interpreter::bytecode::{BinaryOperator, ComparisonOperator};
    include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));
}

// Re-export top-level auto-generated functions for crate-level access.
// Keep generated-code lint allowances scoped to this include wrapper.
#[allow(dead_code, unsafe_op_in_unsafe_fn, unused_variables)]
mod generated_root {
    use pyre_interpreter::bytecode::{BinaryOperator, ComparisonOperator};
    include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));
}
pub use generated_root::*;

// Auto-generated `OpcodeHandler` trait impls. Lives in a separate file
// because jit_trace_gen.rs is `include!`d twice (once inside `pub mod
// generated`, once at crate root) and trait impls cannot be duplicated.
use pyre_interpreter::bytecode::{BinaryOperator, ComparisonOperator};
include!(concat!(env!("OUT_DIR"), "/jit_trace_trait_impls.rs"));
