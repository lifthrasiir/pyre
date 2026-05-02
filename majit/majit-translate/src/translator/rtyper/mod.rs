//! `translator/rtyper/` — RPython-orthodox `rpython/rtyper/` counterparts
//! that are not part of the LEGACY `translator_legacy/rtyper/` infra.
//!
//! Files in this module mirror upstream `rpython/rtyper/` 1:1 by name and
//! by structure (`rclass.py` → `rclass.rs`, `rpbc.py` → `rpbc.rs`).
//! Until the standalone `majit-rtyper` crate (roadmap Phase 6) lands,
//! the underlying `TypeResolutionState` machinery still lives under
//! `translator_legacy/rtyper/rtyper.rs`; these modules import it from
//! there as a temporary wiring bridge.

pub mod annlowlevel;
pub mod callparse;
pub mod cutover;
pub mod error;
pub mod extregistry;
pub mod flowspace_adapter;
pub mod llannotation;
pub mod llinterp;
pub mod lltypesystem;
pub mod normalizecalls;
pub mod pairtype;
pub mod rbool;
pub mod rbuiltin;
pub mod rclass;
pub mod rfloat;
pub mod rint;
pub mod rmodel;
pub mod rnone;
pub mod rpbc;
pub mod rstr;
pub mod rtuple;
pub mod rtyper;
