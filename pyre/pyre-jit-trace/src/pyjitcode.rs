//! `PyJitCode`: pyre's per-CodeObject JitCode wrapper.
//!
//! RPython's `JitCode` (jitcode.py:9) is a single class that owns
//! both the bytecode (`code` / `constants_*` / `c_num_regs_*`) and
//! the per-graph metadata (`name`, `fnaddr`, `calldescr`,
//! `jitdriver_sd`). pyre still has a split runtime representation:
//!
//!   * `majit_metainterp::jitcode::JitCode` is the current runtime
//!     adapter bytecode container (`code`, `constants_*`, `num_regs_*`,
//!     plus pyre-only `exec.*` pools). It is not the canonical
//!     codewriter `majit_translate::jitcode::JitCode`.
//!   * `PyJitCode` (this struct) wraps that JitCode together with
//!     pyre-only translation metadata — `pc_map` (Python PC → byte
//!     offset), `merge_point_pc`, the runtime `w_code` wrapper, and
//!     register layout — that RPython
//!     does not need because RPython's bytecode PCs are already
//!     JitCode PCs.
//!
//! The struct lives in `pyre-jit-trace` (the lower crate) so that
//! both the codewriter (`pyre-jit::jit::codewriter`) and the
//! trace/blackhole runtime (this crate) can hold the same
//! `Arc<PyJitCode>` instances. RPython's `MetaInterpStaticData.jitcodes`
//! list and `CallControl.jitcodes` dict reference identical
//! `JitCode` Python objects via Python's reference semantics; pyre
//! mirrors the shared-identity part with `Arc<PyJitCode>`. The wrapped
//! runtime `JitCode` allocation is also kept stable when the codewriter
//! fills a shell, because `inline_call_*` descriptors hold the callee
//! `JitCode` object itself in the RPython model.
//!
//! ## Discriminator: 3-state mode mapping
//!
//! A `PyJitCode` is one of three modes, encoded across two flags:
//!
//! | mode             | `jitcode.code` | `metadata.pc_map` | predicate                |
//! |------------------|----------------|--------------------|--------------------------|
//! | Skeleton         | empty          | empty              | [`PyJitCode::is_skeleton`]       |
//! | PortalBridge     | non-empty      | empty              | [`PyJitCode::is_portal_bridge`]  |
//! | PerCodeObject    | non-empty      | non-empty          | [`PyJitCode::is_populated`]      |
//!
//! `code` and `pc_map` are independent because the portal-bridged
//! install ([`crate::canonical_bridge::install_portal_for`]) reuses
//! the canonical portal `JitCode.code` byte stream but skips the
//! per-Python-PC mapping (the portal dispatches via its own arms on
//! `pycode.instructions[pc]`). Drained CodeWriter installs do both:
//! fill real instructions into `code` and stamp `pc_map` to
//! `code.instructions.len()`. Skeletons have neither because they are
//! placeholder slots inserted by `CallControl::get_jitcode` before the
//! assembler drain runs.
//!
//! The fourth combination (`code` empty, `pc_map` non-empty) is not
//! produced by any production path; the predicates classify it as
//! neither Skeleton nor PortalBridge nor PerCodeObject. Test fixtures
//! that fabricate this combination (e.g. by calling [`PyJitCode::skeleton`]
//! and then pushing into `metadata.pc_map`) flow as PerCodeObject for
//! [`PyJitCode::is_populated`] purposes (the historical predicate
//! looks at `pc_map` only).
//!
//! Convergence path: RPython's single `JitCode` class has neither flag
//! to consult — `assembler.assemble` populates `code` in place and
//! per-PC mapping is implicit in the bytecode stream. pyre will lose
//! the dual-mode discrimination once the codewriter routes Python
//! bytecode through the canonical RPython codewriter pipeline (Phase
//! G.4.4+). Until then, the mode mapping above is the source of truth
//! for every reader that branches on install style.

use majit_metainterp::jitcode::JitCode as RuntimeJitCode;
use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};

/// Pyre-only metadata attached to a Python CodeObject's compiled JitCode.
///
/// RPython does not need these fields because its bytecode PCs are already
/// JitCode PCs. Pyre translates CPython bytecode to JitCode lazily, so the
/// translation maps live here instead of polluting either upstream's
/// canonical `JitCode` or pyre's eventual single-store replacement.
pub struct PyJitCodeMetadata {
    /// py_pc → jitcode byte offset. Named for RPython's `frame.pc →
    /// jitcode position` flow; the runtime side reads this to map
    /// the Python frame's `next_instr` to the JitCode entry point
    /// for blackhole resume / inline call tracing.
    pub pc_map: Vec<usize>,
    /// Value-stack depth at each Python PC, in slots above stack_base.
    pub depth_at_py_pc: Vec<u16>,
    /// Post-regalloc Ref-bank color of the portal jitdriver's first red
    /// argument (`frame`).  RPython parity: `pypy/module/pypyjit/
    /// interp_jit.py:67 reds = ['frame', 'ec']` declares the portal
    /// calling convention, and `JitDriverStaticData.red_args_indices`
    /// (`rpython/jit/metainterp/warmspot.py`) records the inputarg
    /// position of each red arg.  This field is the pyre equivalent —
    /// the snapshot serializer at
    /// `pyre-jit-trace::trace_opcode::get_list_of_active_boxes` uses it
    /// to map the live_r color back to the symbolic `sym.frame` OpRef.
    /// `u16::MAX` for portal-bridge installs that don't run the
    /// per-CodeObject regalloc (the snapshot helper sentinel-skips).
    pub portal_frame_reg: u16,
    /// Post-regalloc Ref-bank color of the portal jitdriver's second red
    /// argument (`ec`, `pypy/module/pypyjit/interp_jit.py:67`).
    /// Snapshot serializer maps this color to `sym.execution_context`.
    pub portal_ec_reg: u16,
    /// Absolute start index of the operand stack in PyFrame.locals_cells_stack_w.
    pub stack_base: usize,
    /// Phase 2 commit 2.1 (Tasks #158/#159/#122 epic, plan
    /// `~/.claude/plans/staged-sauteeing-koala.md`): post-regalloc
    /// color of each Python-semantic stack slot.
    /// `stack_slot_color_map[d]` = `apply_rename(Kind::Ref, nlocals + d)`
    /// for `d in 0..code.max_stackdepth` (= CPython `co_stacksize`).
    /// The `+ nlocals` here is the register-space stack base used by
    /// the codewriter (`RegisterLayout::stack_base`, which
    /// `RegisterLayout::compute` sets to `nlocals as u16`), NOT the
    /// `stack_base` field above (which is the PyFrame absolute
    /// `varnames.len() + ncells`). Populated in `finalize_jitcode`
    /// after `apply_rename` runs; portal-bridge installs
    /// (`canonical_bridge::install_portal_for`) populate it as
    /// identity over the same range.
    ///
    /// Length invariant: `stack_slot_color_map.len() == code.max_stackdepth`,
    /// so the bridge fallback at `state.rs::setup_bridge_sym`
    /// (`stack_base + stack_slot_color_map.len()`) reconstructs the full
    /// runtime PyFrame allocation
    /// (`pyframe.rs:1576` `nlocals + ncells + max_stackdepth`). Earlier
    /// per-CodeObject installs sized this to
    /// `max_stack_depth_observed = max(depth_at_pc)` which under-sized
    /// the map when JIT-traced PCs did not reach the static peak; the
    /// `co_stacksize` invariant restores parity with the runtime.
    ///
    /// After Phase 2.1c (commit `3fd64d5b0f3`, regalloc.rs:448-466 +
    /// :527-535) the stack-slot input-arg pinning is gone, so this
    /// map is no longer the identity `[stack_base, stack_base+1, …]`
    /// — entries are whatever color `apply_rename` produced. Decoders
    /// (`state.rs`, `trace_opcode.rs`, `codewriter.rs`) must read
    /// through the map; they cannot assume the old `nlocals + d`
    /// invariant.
    ///
    /// Tail caveat: `regalloc::allocate_registers` only pins
    /// `[0..nlocals)` plus the portal red args; pre-indices
    /// `nlocals + d` for `d >= max(depth_at_py_pc)` never appear in
    /// any SSA op, so `apply_rename` falls through with identity and
    /// `stack_slot_color_map[d] == nlocals + d` by accident, not by
    /// post-regalloc decision. Today every consumer reads only the
    /// `[0..depth_at_py_pc[pc])` prefix for value recovery and uses
    /// `len()` solely for frame-allocation length matching, so the
    /// tail's identity-by-fallthrough is harmless. If a future
    /// consumer needs full-range colors as real post-regalloc values,
    /// extend `external` in `regalloc.rs:680-697` to cover
    /// `(nlocals..nlocals + max_stackdepth)` so `enforce_input_args`
    /// pins the tail too (parity with `flatten.py:88-100`).
    pub stack_slot_color_map: Vec<u16>,
    /// SSA-authoritative live_r epic slice 3a:
    /// post-regalloc color of each Python-semantic local slot.
    /// `pyre_color_for_semantic_local[i]` = `apply_rename(Kind::Ref, i)`
    /// for `i in 0..code.varnames.len()`. Populated in `finalize_jitcode`
    /// after `apply_rename` runs, parallel to `stack_slot_color_map`;
    /// portal-bridge installs (`canonical_bridge::install_portal_for`)
    /// populate it as identity over the same range.
    ///
    /// Length invariant: `pyre_color_for_semantic_local.len() == nlocals`,
    /// matching the locals prefix of the runtime PyFrame allocation.
    ///
    /// Today `enforce_input_args` (`flatten.py:88-100` parity)
    /// pins each local-i inputarg color to identity (`color = i`),
    /// so this map is `[0, 1, ..., nlocals-1]` for every populated
    /// jitcode. Slice 3b-2 (`get_list_of_active_boxes`) derives the
    /// semantic index from the color via this map for locals and
    /// `stack_slot_color_map` for stack slots.
    pub pyre_color_for_semantic_local: Vec<u16>,
}

/// Compiled JitCode plus pyre-only metadata.
pub struct PyJitCodePayload {
    pub jitcode: std::sync::Arc<RuntimeJitCode>,
    pub metadata: PyJitCodeMetadata,
    /// pyre's graph identity for the cached jitcode slot.
    ///
    /// RPython indexes `CallControl.jitcodes` and `unfinished_graphs`
    /// directly by graph object. pyre still keys the public cache by
    /// `w_code` when available, but the cached object carries the raw
    /// CodeObject pointer so the queue can stay a bare graph list.
    pub code_ptr: *const pyre_interpreter::CodeObject,
    /// pyre-only wrapper identity for trace-side jitcode lookup.
    pub w_code: *const (),
    /// True if the jitcode contains BC_ABORT opcodes (unsupported bytecodes).
    /// Precomputed at compile time to avoid repeated bytecode scanning.
    pub has_abort: bool,
    /// Python PC of the jit_merge_point opcode (trace entry header).
    pub merge_point_pc: Option<usize>,
}

/// Shared `PyJitCode` identity whose payload is filled in place.
///
/// Held by `Arc` so the same instance can sit in both
/// `MetaInterpStaticData.jitcodes` (the runtime list) and
/// `CallControl.jitcodes` (the codewriter dict) without duplicating
/// the bytecode buffer or metadata vectors. RPython's
/// `JitCode` references are shared the same way through Python's
/// refcount semantics.
///
/// RPython mutates the `JitCode` shell inserted by `call.py:168-170`
/// when `assembler.assemble(..., jitcode, ...)` runs in
/// `codewriter.py:67`. Pyre's assembler still returns a fresh payload,
/// so the outer `PyJitCode` uses interior mutability to preserve the
/// same object identity while filling the payload during the writer
/// drain. The inner runtime `JitCode` allocation is filled in place as
/// well, so any caller-side `RuntimeBhDescr::JitCode(Arc<JitCode>)`
/// created by a future orthodox `inline_call_*` rewrite keeps pointing
/// at the populated callee object after the drain.
///
/// Production mutation is confined to the single-threaded codewriter
/// publication path before runtime readers observe the populated object.
pub struct PyJitCode {
    payload: UnsafeCell<PyJitCodePayload>,
}

// SAFETY: `PyJitCode` payload replacement is restricted to the codewriter
// publication path, which runs under pyre's single-threaded JIT setup before
// the populated object is handed to runtime readers. Runtime-visible index
// stamping uses atomics on the inner `RuntimeJitCode`.
unsafe impl Sync for PyJitCode {}

impl Deref for PyJitCode {
    type Target = PyJitCodePayload;

    fn deref(&self) -> &Self::Target {
        // SAFETY: shared readers only observe immutable payload references.
        unsafe { &*self.payload.get() }
    }
}

impl DerefMut for PyJitCode {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.payload.get_mut()
    }
}

impl PyJitCode {
    pub fn new(payload: PyJitCodePayload) -> Self {
        Self {
            payload: UnsafeCell::new(payload),
        }
    }

    pub fn from_parts(
        jitcode: std::sync::Arc<RuntimeJitCode>,
        metadata: PyJitCodeMetadata,
        code_ptr: *const pyre_interpreter::CodeObject,
        w_code: *const (),
        has_abort: bool,
        merge_point_pc: Option<usize>,
    ) -> Self {
        Self::new(PyJitCodePayload {
            jitcode,
            metadata,
            code_ptr,
            w_code,
            has_abort,
            merge_point_pc,
        })
    }

    /// Fill the cached payload without changing the outer `PyJitCode`
    /// allocation or the inner runtime `JitCode` allocation, even if
    /// setup-time call descriptors have already cloned the inner
    /// `Arc<JitCode>` shell. This is pyre's Rust-side stand-in for RPython
    /// `assembler.assemble(..., jitcode, ...)` mutating the existing
    /// `JitCode` object from `CallControl.jitcodes[graph]`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee no runtime thread is currently reading
    /// the payload through any cloned `Arc<PyJitCode>` or cloned inner
    /// `Arc<JitCode>`. RPython relies
    /// on the implicit single-threaded semantics of the translation /
    /// codewriter setup phase — the JitCode shell is filled in place
    /// before any runtime reader observes it. Pyre cannot encode that
    /// invariant in the Rust type system without a heavyweight lock,
    /// so callers must check the precondition manually:
    ///
    /// * Only invoke this from the JIT setup / codewriter publication
    ///   path, before runtime tracing or blackhole resume can dispatch
    ///   on the same code.
    /// * In particular, do NOT call this to roll a populated payload
    ///   back to a skeleton — that breaks the "runtime reader never
    ///   observes a reset shell" invariant. Skeleton resets must
    ///   replace the outer `Arc` instead (see
    ///   `CallControl::reset_jitcode_skeleton`).
    pub unsafe fn replace_with(&self, next: PyJitCode) {
        let PyJitCodePayload {
            jitcode: next_jitcode,
            metadata,
            code_ptr,
            w_code,
            has_abort,
            merge_point_pc,
        } = next.payload.into_inner();
        let next_jitcode = std::sync::Arc::try_unwrap(next_jitcode)
            .expect("freshly assembled PyJitCode must uniquely own its runtime JitCode");
        unsafe {
            let current = &mut *self.payload.get();
            // RPython's call descriptors keep the callee JitCode object itself.
            // During setup, an inline_call descr may therefore already point at
            // this shell before assembler.assemble() fills it. Rust's Arc cannot
            // express "shared for setup identity, exclusively mutated before
            // runtime publication", so we write through the stable allocation
            // under the setup-phase precondition documented above.
            let current_jitcode = std::sync::Arc::as_ptr(&current.jitcode) as *mut RuntimeJitCode;
            *current_jitcode = next_jitcode;
            current.metadata = metadata;
            current.code_ptr = code_ptr;
            current.w_code = w_code;
            current.has_abort = has_abort;
            current.merge_point_pc = merge_point_pc;
        }
    }

    /// Check if this jitcode has BC_ABORT opcodes.
    pub fn has_abort_opcode(&self) -> bool {
        self.has_abort
    }

    /// "Has `assembler.assemble` been run on this jitcode yet?" A
    /// freshly-constructed RPython `JitCode(name, fnaddr, calldescr,
    /// ...)` (jitcode.py:14, call.py:168) leaves `self.code` unset
    /// until `setup` (jitcode.py:22) is invoked by
    /// `assembler.assemble(ssarepr, jitcode, num_regs)`
    /// (codewriter.py:67); pyre's split wrapper uses `pc_map.is_empty()`
    /// as the same "still a shell" test.
    ///
    /// PerCodeObject mode in the discriminator table on the module
    /// doc.
    pub fn is_populated(&self) -> bool {
        !self.metadata.pc_map.is_empty()
    }

    /// Skeleton slot inserted by [`Self::skeleton`] — neither `code`
    /// nor `pc_map` populated yet. See the discriminator table on
    /// the module doc.
    ///
    /// Strictly equivalent to `!is_populated() && !is_portal_bridge()`
    /// (DeMorgan-expanded: `pc_map.is_empty() && (code.is_empty() ||
    /// !pc_map.is_empty())` reduces to the conjunction below).
    /// Callers prefer this name over the negated-pair form because it
    /// names the third mode in the discriminator table directly.
    pub fn is_skeleton(&self) -> bool {
        self.jitcode.code.is_empty() && self.metadata.pc_map.is_empty()
    }

    /// Is this `PyJitCode` a portal-bridged install (G.3a
    /// `canonical_bridge::install_portal_for`)?
    ///
    /// Discriminator:
    ///   * `jitcode.code` non-empty (rules out `PyJitCode::skeleton`,
    ///     which clones `Arc::new(RuntimeJitCode::default())` whose
    ///     `code` is empty).
    ///   * `metadata.pc_map` empty (rules out drained CodeWriter
    ///     installs, whose setup-time drain populates `pc_map` to
    ///     `code.instructions.len()`).
    ///
    /// Used by readers that have to branch on portal-mode semantics —
    /// portal entry has no per-Python-PC `pc_map` because the portal
    /// jitcode dispatches on `pycode.instructions[pc]` at runtime via
    /// its own dispatch arms.  See
    /// `canonical_bridge::install_portal_for` for the full reader
    /// audit (G.3a).
    ///
    /// G.3b landed this discriminator for reader-audit probes. The
    /// orthodox redirect path now avoids binding portal-bridge payloads as
    /// `jd.mainjitcode`; production readers still branch on this predicate
    /// only for explicit bridge-probe installs.
    pub fn is_portal_bridge(&self) -> bool {
        !self.jitcode.code.is_empty() && self.metadata.pc_map.is_empty()
    }

    /// Empty `PyJitCode` slot inserted by `CallControl::get_jitcode`
    /// (call.py:168 `jitcode = JitCode(graph.name, fnaddr, calldescr, ...)`).
    ///
    /// In RPython the `JitCode` constructor returns a fresh object whose
    /// `code` / `descrs` / `liveness` arrays are all empty until
    /// `assembler.assemble(...)` populates them later in
    /// `make_jitcodes`'s drain loop (codewriter.py:80).  The skeleton
    /// gives the dict an entry with a stable identity so re-entrant
    /// `get_jitcode` calls (or pyre's `merge_point_pc` refinement
    /// shortcut) can find an existing key without recompiling.
    ///
    /// Until the drain replaces the slot, the only field with meaningful
    /// content is `merge_point_pc` (the refinement hint passed in by
    /// `get_jitcode`).
    pub fn skeleton(
        code_ptr: *const pyre_interpreter::CodeObject,
        w_code: *const (),
        merge_point_pc: Option<usize>,
    ) -> Self {
        Self::from_parts(
            std::sync::Arc::new(RuntimeJitCode::default()),
            PyJitCodeMetadata {
                pc_map: Vec::new(),
                depth_at_py_pc: Vec::new(),
                portal_frame_reg: 0,
                portal_ec_reg: 0,
                stack_base: 0,
                stack_slot_color_map: Vec::new(),
                pyre_color_for_semantic_local: Vec::new(),
            },
            code_ptr,
            w_code,
            false,
            merge_point_pc,
        )
    }
}
