//! `translator/backendopt/` — mirror of upstream
//! `rpython/translator/backendopt/`. Houses the post-annotator graph
//! optimisations (SSA conversion, escape analysis, etc.).
//!
//! Each file name matches upstream (e.g. `ssa.py` → `ssa.rs`).

pub mod all;
pub mod canraise;
pub mod collectanalyze;
pub mod constfold;
pub mod finalizer;
pub mod gilanalysis;
pub mod graphanalyze;
pub mod inline;
pub mod merge_if_blocks;
pub mod removeassert;
pub mod removenoops;
pub mod ssa;
pub mod stat;
pub mod storesink;
pub mod support;
