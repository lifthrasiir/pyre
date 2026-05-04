//! Line-by-line port of `rpython/jit/codewriter/`.
//!
//! Contains the low-level codewriter stage that converts an rtyped graph
//! into `JitCode` consumed by `majit-metainterp`. Sibling modules in this
//! directory mirror `rpython/jit/codewriter/*.py` one-to-one.

pub mod annotation_state;
pub mod assembler;
pub mod call;
pub mod codewriter;
pub mod flatten;
pub mod format;
pub mod insns;
pub mod jitcode;
pub mod jtransform;
pub mod liveness;
pub mod policy;
pub mod regalloc;
pub mod support;
pub mod type_state;
