//! Rust translator scaffolding around the orthodox codewriter modules.
//!
//! **LEGACY.** Ad-hoc majit-local equivalent of
//! `rpython/{annotator,rtyper}` that predates the line-by-line port
//! roadmap (see `.claude/plans/majestic-forging-meteor.md`). Every
//! item in this subtree will be replaced by its proper RPython
//! counterpart in `majit-annotator` (Phase 5 exit) and `majit-rtyper`
//! (Phase 6 exit), and the whole subtree is deleted at roadmap
//! commit P8.11.
//!
//! Until then, keep this tree consumer-accessible — it is what the
//! majit-translate Rust-source analysis path (via `crate::front::`)
//! uses for jitcode emission, and replacing it is gated on the new
//! pipeline being end-to-end working under `PYRE_RTYPER=1`.

pub mod annotator;
pub mod pipeline;
pub mod rtyper;
