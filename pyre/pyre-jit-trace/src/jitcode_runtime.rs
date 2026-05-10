//! Runtime access to the build-time `pipeline.jitcodes` table.
//!
//! RPython: `MetaInterpStaticData.jitcodes` (warmspot.py:281-282) — the list
//! of `JitCode` objects produced by `CodeWriter.make_jitcodes()`
//! (codewriter.py:89). In RPython this list is passed by reference from
//! `CallControl.jitcodes` directly into `MetaInterpStaticData`; the two
//! stores reference the same Python objects.
//!
//! majit's build-time side lives in `majit_translate::jitcode::JitCode`
//! (serde-serializable, emitted by `build.rs` into
//! `$OUT_DIR/opcode_jitcodes.bin`). This module deserializes that blob once
//! on first access and hands out `Arc<JitCode>` shells. The `PipelineOpcodeArm
//! .entry_jitcode_index` field (already present in the canonical pipeline
//! result) indexes into this table.
//!
//! No side-table serialization: the only persisted collection is
//! `pipeline.jitcodes`, in allocation order, matching RPython's single-store
//! model (`feedback_single_jitcodes_store`).
//!
//! Phase D-1 Step 2 of the eval-loop automation plan.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use majit_ir::DescrRef;
use majit_translate::jitcode::{BhDescr, JitCode};
use majit_translate::opcode_dispatch::PipelineOpcodeArm;
use majit_translate::{CallPath, OpcodeDispatchSelector};
use pyre_interpreter::bytecode::Instruction;

/// Deserialized `pipeline.jitcodes` — RPython `all_jitcodes[]` from
/// codewriter.py:89.  Dense: `ALL_JITCODES[i].index == i` (RPython
/// codewriter.py:80 invariant, preserved by
/// `collect_jitcodes_in_alloc_order`). Use `get_jitcode_by_index` or
/// direct indexing for lookup by `entry_jitcode_index`.
static ALL_JITCODES: LazyLock<Vec<Arc<JitCode>>> = LazyLock::new(|| {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/opcode_jitcodes.bin"));
    let vec: Vec<Arc<JitCode>> = bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize opcode_jitcodes.bin \
             ({} bytes): {e}",
            BYTES.len(),
        )
    });
    // RPython codewriter.py:80: `all_jitcodes[jitcode.index] is jitcode`.
    // Check at load time so any regression in
    // `collect_jitcodes_in_alloc_order` is caught immediately.
    for (i, jc) in vec.iter().enumerate() {
        assert_eq!(
            jc.index(),
            i,
            "pyre-jit-trace: jitcode[{i}].index = {} (expected {i}); \
             RPython invariant `all_jitcodes[i].index == i` broken",
            jc.index(),
        );
    }
    vec
});

/// Deserialized `pipeline.opcode_dispatch` — the arm table. Each entry
/// carries `arm_id`, `selector`, and `entry_jitcode_index` (logical index
/// into `ALL_JITCODES` via `.index`).
static ALL_OPCODE_ARMS: LazyLock<Vec<PipelineOpcodeArm>> = LazyLock::new(|| {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/opcode_dispatch.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize opcode_dispatch.bin \
             ({} bytes): {e}",
            BYTES.len(),
        )
    })
});

/// RPython: `metainterp_sd.jitcodes` — full all_jitcodes table.
pub fn all_jitcodes() -> &'static [Arc<JitCode>] {
    &ALL_JITCODES
}

/// RPython: `metainterp_sd.jitcodes[index]` where `index == jitcode.index`.
///
/// Build-time `PipelineOpcodeArm.entry_jitcode_index` is this logical
/// index. The dense invariant (`ALL_JITCODES[i].index == i`) is asserted
/// at load time, so direct vec indexing is correct.
pub fn get_jitcode_by_index(index: usize) -> Option<Arc<JitCode>> {
    ALL_JITCODES.get(index).cloned()
}

/// Cached index of the build-time portal jitcode within `ALL_JITCODES`.
///
/// RPython `warmspot.py:281-282` + `call.py:147-148`:
/// `jd.mainjitcode = self.get_jitcode(jd.portal_graph)` followed by
/// `jd.mainjitcode.jitdriver_sd = jd`. The single jitcode whose
/// `jitdriver_sd` is set is the portal — every other entry in
/// `metainterp_sd.jitcodes` is either an inlined callee or an indirect
/// call target. Pyre's bincode preserves that flag through
/// `oncelock_usize_serde`, so the scan below identifies the same
/// jitcode the codewriter side stored in
/// `JitDriverStaticData.mainjitcode`.
///
/// Production identity: `pyre-jit-trace/build.rs:33-54` (Task #140)
/// includes `pyre/pyre-jit/src/eval.rs` in the source-only walk so the
/// scan below picks up the `eval_loop_jit` jitcode whose
/// `jitdriver_sd` is populated by `assign_portal_jitdriver_indices`
/// (`pyre-jit/src/jit/codewriter.rs:6544` — call.py:148 deferred port).
static PORTAL_JITCODE_INDEX: LazyLock<Option<usize>> = LazyLock::new(|| {
    let mut hits = ALL_JITCODES
        .iter()
        .enumerate()
        .filter(|(_, jc)| jc.jitdriver_sd().is_some())
        .map(|(i, _)| i);
    let first = hits.next();
    // RPython `call.py:147` `jd.mainjitcode = self.get_jitcode(jd.portal_graph)`
    // assigns once per JitDriverStaticData; pyre runs a single jitdriver
    // (PyJitDriver) so at most one `jitdriver_sd` flag should be set in
    // the build-time pipeline. A second hit signals a structural
    // regression in `setup_jitdriver` and must surface immediately.
    if hits.next().is_some() {
        panic!(
            "pyre-jit-trace: build-time pipeline has more than one portal \
             jitcode (jitdriver_sd populated). RPython `call.py:147` allows \
             exactly one per `JitDriverStaticData`."
        );
    }
    first
});

/// RPython: `metainterp_sd.jitcodes[jitdriver_sd.mainjitcode.index]`
/// (warmspot.py:281-282 + call.py:147-148) — the single portal jitcode
/// that `find_all_graphs(portal, policy)` seeds the jitcode closure
/// from. Returns `None` only when the build-time pipeline has no
/// jitdriver registered (e.g. compact test inputs).
///
/// Phase G consumers route trace-side user-function calls
/// (`callee_frame_helper`, `jit_create_callee_frame_*`,
/// `jit_force_callee_frame`) through this accessor instead of emitting
/// runtime per-CodeObject jitcodes. The orthodox model treats every
/// user CodeObject as the portal's `pycode` input argument and reuses
/// the single portal JitCode for every call — see RPython
/// `pypy/module/pypyjit/interp_jit.py portal_runner` and
/// `rpython/jit/codewriter/jtransform.py:473` `inline_call_*` emit.
///
/// G.2 introduces this accessor as the surface that G.3 will plug
/// callee dispatch into; G.2 itself does not redirect any caller.
pub fn portal_jitcode() -> Option<Arc<JitCode>> {
    let idx = (*PORTAL_JITCODE_INDEX)?;
    ALL_JITCODES.get(idx).cloned()
}

/// RPython: opcode dispatch arm table (analogue of PyPy's per-opcode
/// Python methods). One `PipelineOpcodeArm` per Rust `match` arm.
pub fn all_opcode_arms() -> &'static [PipelineOpcodeArm] {
    &ALL_OPCODE_ARMS
}

/// Returns the arm with the given `arm_id`.
pub fn get_arm(arm_id: usize) -> Option<&'static PipelineOpcodeArm> {
    ALL_OPCODE_ARMS.iter().find(|a| a.arm_id == arm_id)
}

/// Convenience: resolve `arm_id` → entry jitcode. Returns `None` if arm
/// doesn't exist or the arm has no body graph.
pub fn jitcode_for_arm(arm_id: usize) -> Option<Arc<JitCode>> {
    let arm = get_arm(arm_id)?;
    let idx = arm.entry_jitcode_index?;
    get_jitcode_by_index(idx)
}

/// Variant-name → arm_id index built from `ALL_OPCODE_ARMS` selectors.
/// Multi-pattern `Instruction::A | Instruction::B` arms expand each
/// variant to the same arm_id, matching the RPython model where a
/// single Python method is registered under each dispatched opcode.
///
/// Keyed by the bare variant name (e.g. `"PopTop"`, `"LoadFast"`) — the
/// last segment of `OpcodeDispatchSelector::Path.segments`. Derived Debug
/// on `Instruction` yields the same variant prefix, so
/// `arm_id_for_instruction` uses this to resolve an `Instruction` value
/// at runtime.
static ARM_ID_BY_VARIANT: LazyLock<HashMap<String, usize>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    for arm in all_opcode_arms() {
        collect_variant_names(&arm.selector, arm.arm_id, &mut map);
    }
    map
});

fn collect_variant_names(
    sel: &OpcodeDispatchSelector,
    arm_id: usize,
    out: &mut HashMap<String, usize>,
) {
    match sel {
        OpcodeDispatchSelector::Path(cp) => {
            if let Some(name) = variant_name_from_path(cp) {
                out.insert(name.to_string(), arm_id);
            }
        }
        OpcodeDispatchSelector::Or(cases) => {
            for c in cases {
                collect_variant_names(c, arm_id, out);
            }
        }
        OpcodeDispatchSelector::Wildcard | OpcodeDispatchSelector::Unsupported => {}
    }
}

fn variant_name_from_path(cp: &CallPath) -> Option<&str> {
    cp.segments.last().map(String::as_str)
}

/// Extracts the variant identifier from `format!("{instr:?}")`. The
/// derived `Debug` for an `Instruction` variant starts with the variant
/// name, optionally followed by ` { .. }` for struct variants or ` (..)`
/// for tuple variants.
fn extract_variant_name(instr_debug: &str) -> &str {
    instr_debug
        .split(|c: char| c.is_whitespace() || c == '(' || c == '{')
        .next()
        .unwrap_or(instr_debug)
}

/// Runtime variant name for `Instruction` — matches the last segment of
/// the build-time `OpcodeDispatchSelector::Path` emitted by the parser
/// (e.g. `Instruction::PopTop` → `"PopTop"`).
pub fn instruction_variant_name(instruction: &Instruction) -> String {
    extract_variant_name(&format!("{instruction:?}")).to_string()
}

/// Resolve an `Instruction` to its `PipelineOpcodeArm.arm_id`.
///
/// Returns `None` for variants not covered by any dispatch arm — either
/// because the parser emitted `Wildcard`/`Unsupported` for that arm, or
/// the variant has no match arm in `execute_opcode_step`.
pub fn arm_id_for_instruction(instruction: &Instruction) -> Option<usize> {
    ARM_ID_BY_VARIANT
        .get(&instruction_variant_name(instruction))
        .copied()
}

/// Resolve an `Instruction` directly to its entry jitcode. This is the
/// MIFrame-side entry for Phase D-2 shadow dispatch.
pub fn jitcode_for_instruction(instruction: &Instruction) -> Option<Arc<JitCode>> {
    jitcode_for_arm(arm_id_for_instruction(instruction)?)
}

/// Deserialized `pipeline.insns` — the opname → u8 table
/// `Assembler.write_insn` grew during assembly. `JitCode.code[i]` bytes
/// can be mapped back to opnames through the inverted view exposed by
/// `opname_for_byte`. Matches RPython `setup_insns(insns)` consumption
/// at pyjitpl.py:2227-2243.
static INSNS_OPNAME_TO_BYTE: LazyLock<HashMap<String, u8>> = LazyLock::new(|| {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/opcode_insns.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize opcode_insns.bin \
             ({} bytes): {e}",
            BYTES.len(),
        )
    })
});

/// Inverted view: `u8` opcode byte → opname string. Built lazily on
/// first access from `INSNS_OPNAME_TO_BYTE`. Duplicate u8 values should
/// never occur (assembler.py assigns sequential ids), and we assert that
/// at load time.
static INSNS_BYTE_TO_OPNAME: LazyLock<HashMap<u8, String>> = LazyLock::new(|| {
    let mut map = HashMap::with_capacity(INSNS_OPNAME_TO_BYTE.len());
    for (name, &byte) in INSNS_OPNAME_TO_BYTE.iter() {
        assert!(
            map.insert(byte, name.clone()).is_none(),
            "duplicate opcode byte {byte} in pipeline.insns table"
        );
    }
    map
});

/// RPython `setup_insns(insns)` — full opname → opcode-byte table.
pub fn insns_opname_to_byte() -> &'static HashMap<String, u8> {
    &INSNS_OPNAME_TO_BYTE
}

/// Inverse lookup: `JitCode.code[i]` byte → opname. None for unknown
/// bytes (e.g. operand bytes, not opcode bytes).
pub fn opname_for_byte(byte: u8) -> Option<&'static str> {
    INSNS_BYTE_TO_OPNAME.get(&byte).map(String::as_str)
}

/// Inverse of `insns_opname_to_byte()` — full `u8 -> opname/argcodes` table.
pub fn insns_byte_to_opname() -> &'static HashMap<u8, String> {
    &INSNS_BYTE_TO_OPNAME
}

/// Deserialized `pipeline.descrs` — RPython `Assembler.descrs`
/// (assembler.py:23). Handed to `BlackholeInterpBuilder.setup_descrs`
/// at builder construction (blackhole.py:59 `self.setup_descrs(asm.descrs)`,
/// :102-103 `def setup_descrs(self, descrs): self.descrs = descrs`).
///
/// Each 'd'/'j' argcode in a `JitCode.code` byte stream is a 2-byte
/// little-endian index into this pool. The resolved `BhDescr` is what
/// every `bhimpl_*` handler reads for field offsets, call descriptors,
/// sub-JitCodes, and switch dicts.
static ALL_DESCRS: LazyLock<Vec<BhDescr>> = LazyLock::new(|| {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/opcode_descrs.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize opcode_descrs.bin \
             ({} bytes): {e}",
            BYTES.len(),
        )
    })
});

/// RPython: `metainterp_sd.all_descrs` — full shared descr pool.
pub fn all_descrs() -> &'static [BhDescr] {
    &ALL_DESCRS
}

/// Pool of `DescrRef`s indexed alongside [`all_descrs`] so the
/// trace-side jitcode walker
/// ([`crate::jitcode_dispatch::dispatch_via_miframe`]) can resolve each
/// `d`/`j` argcode operand to a real `Arc<dyn Descr>`. Every entry runs
/// through [`crate::descr::make_descr_from_bh`] — the per-variant
/// adapter that maps each `BhDescr` shape to its RPython-orthodox
/// counterpart on the metainterp side:
///
/// - `Field` → `make_field_descr` / immutable variants with full
///   offset, size, type, signedness, and purity flags.
/// - `Array` → `make_array_descr` with full base size, item size,
///   type id, item type, signedness, and struct-array classification.
/// - `Size` → `make_size_descr_with_type_and_vtable`.
/// - `Call` → `make_call_descr_with_effect` — `arg_classes` /
///   `result_type` are reshaped into typed inputs/output AND
///   `extra_info` is threaded through so RPython
///   `call.py:320 effectinfo_from_writeanalyze` parity is preserved
///   (descr cache key + can_raise / oopspec / read-write descrs all
///   match the codewriter's stamp).
/// - `JitCode` → `make_jitcode_descr` (the existing adapter the
///   walker's `inline_call_*` recursion exercises).
///
/// `Switch` / `VableField` / `VableArray` / `VtableMethod` also get
/// concrete trace-side descriptor adapters so the pool shape stays
/// equivalent to `Assembler.descrs`.
///
/// Identity (`Arc::ptr_eq`) currently matches across the walker and
/// trait paths only for the `JitCode` slot — content-derived adapters
/// for `Field` / `Array` / `Size` / `Call` produce fresh `Arc`s
/// per-resolution, so trait-side record sites that need the same
/// `Arc` instance still build their own at the call site (the
/// allow-list in [`crate::shadow_walker::opname_in_shadow_allow_list`]
/// avoids those opnames until the by-index identity factories land).
static ALL_DESCR_REFS: LazyLock<Vec<DescrRef>> = LazyLock::new(|| {
    all_descrs()
        .iter()
        .map(crate::descr::make_descr_from_bh)
        .collect()
});

/// `&'static [DescrRef]` view over [`ALL_DESCR_REFS`] for the walker's
/// `WalkContext::descr_refs` parameter.
pub fn all_descr_refs() -> &'static [DescrRef] {
    &ALL_DESCR_REFS
}

/// Build a `BlackholeInterpBuilder` pre-configured for this binary's
/// jitcodes, paired with the list of `insns` opnames that
/// `wire_bhimpl_handlers` did not assign a handler to.
///
/// RPython parity: `BlackholeInterpBuilder.__init__` (blackhole.py:55-61)
/// runs `setup_insns(asm.insns)` + `setup_descrs(asm.descrs)` and
/// `setup_insns` (blackhole.py:66) resolves each opname via
/// `_get_method` eagerly, raising `AttributeError` if any `bhimpl_*`
/// is missing. The Rust port mirrors the `setup_insns` + `setup_descrs`
/// + `wire_bhimpl_handlers` sequence, but surfaces the unwired list as
/// a return value instead of panicking.
///
/// PRE-EXISTING-ADAPTATION (diagnostic). RPython has no equivalent
/// of this Result-returning shape because upstream's `setup_insns`
/// has total opname coverage at startup. The strict variant
/// [`build_default_bh_builder`] is the production path; this helper is
/// retained for tests that need to inspect the coverage set directly.
pub fn build_default_bh_builder_with_unwired_report() -> (
    majit_metainterp::blackhole::BlackholeInterpBuilder,
    Vec<String>,
) {
    let mut builder = majit_metainterp::blackhole::BlackholeInterpBuilder::new();
    // blackhole.py:58-59 order: setup_insns, then setup_descrs.
    builder.setup_insns(insns_opname_to_byte());
    builder.setup_descrs(all_descrs().to_vec());
    majit_metainterp::blackhole::wire_bhimpl_handlers(&mut builder);
    let unwired: Vec<String> = builder
        .unwired_opnames()
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    (builder, unwired)
}

/// Production-side strict blackhole builder for pyre's guard-failure
/// resume path.  It delegates to `build_inline_call_only_bh_builder`,
/// which installs the audited pyre dispatch surface: byte-identical
/// canonical keys, pyre-u16 register-width adapters, residual_call/vable
/// families, state-field adapters, and the pyre nested inline-call
/// handler.
///
/// Why not `build_default_bh_builder()`: pyre runtime bytecode still
/// contains helper-side layouts that are not canonical RPython argcodes.
/// This builder therefore registers only shapes with an explicit handler
/// contract.  Any emitted byte outside that setup surface now reaches
/// `dispatch_step`'s unwired-opcode panic; there is no legacy fallback.
/// `cond_call_*` / `record_known_result_*` bytes are now wired through
/// `_pyre/P` adapter handlers (`insns.rs:674-678`,
/// payload decoder at `pyre_p_payload_len` below).
pub fn build_pyre_production_bh_builder() -> majit_metainterp::blackhole::BlackholeInterpBuilder {
    majit_metainterp::blackhole::build_inline_call_only_bh_builder()
}

/// Strict-coverage variant of [`build_default_bh_builder_with_unwired_report`]
/// — panics when any `insns` opname lacks a `bhimpl_*` handler.
///
/// RPython parity: matches the `AttributeError` raised by upstream's
/// `setup_insns` (blackhole.py:66) when `_get_method(name, argcodes)`
/// fails. Use this in any code path that must run real production
/// jitcodes; tests that inspect the raw coverage list can use
/// [`build_default_bh_builder_with_unwired_report`].
pub fn build_default_bh_builder() -> majit_metainterp::blackhole::BlackholeInterpBuilder {
    let (builder, unwired) = build_default_bh_builder_with_unwired_report();
    if !unwired.is_empty() {
        panic!(
            "build_default_bh_builder: {} insns opnames have no bhimpl_* \
             handler (RPython blackhole.py:66 raises AttributeError here): \
             {:?}",
            unwired.len(),
            unwired,
        );
    }
    builder
}

/// Decoded one jitcode instruction. Mirrors the static slice that RPython
/// `BlackholeInterpBuilder._get_method` would walk over, without any
/// execution of `bhimpl_*`. Lifetime is tied to the `insns` table, so the
/// opname stays valid while the runtime is alive (`'static`).
///
/// RPython parity: `blackhole.py:105-232` `_get_method.handler` consumes
/// operand bytes per `argcodes` char; this struct captures the same byte
/// layout without executing.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DecodedOp {
    /// The `opname/argcodes` key from the `insns` table.
    pub key: &'static str,
    /// `opname` part of `key` (before the `/`).
    pub opname: &'static str,
    /// `argcodes` part of `key` (after the `/`). Empty for `live/`.
    pub argcodes: &'static str,
    /// Position of the opcode byte in the jitcode.
    pub pc: usize,
    /// Position of the first byte *after* this instruction. `None` if the
    /// instruction reads a variable-length operand (`I`/`R`/`F`) that
    /// overflowed the code slice.
    pub next_pc: usize,
}

/// Decode the byte length of a `/P` adapter payload starting at
/// `cursor`.  Each `_pyre/P` opname has its own pyre helper-side flat
/// payload shape; the `P` pseudo-argcode is just an opt-out from the
/// canonical RPython argcode alphabet.  Producers and shapes:
///
/// | opname                              | producer (`assembler.rs`)             | payload bytes |
/// |-------------------------------------|---------------------------------------|---------------|
/// | `inline_call_pyre_nested`           | `:nested_inline_call_*_typed_args`    | `4 + num_args*5 + 6`  (`sub_idx u16 + num_args u16 + num_args × (kind u8, caller_src u16, callee_dst u16) + return_i u16 + return_r u16 + return_f u16`) |
/// | `call_assembler_int_pyre`           | `:3429 call_assembler_int_like`       | `6 + num_args*3`  (`target_idx u16 + dst u16 + num_args u16 + num_args × (kind u8, reg u16)`) |
/// | `call_assembler_ref_pyre`           | `:3451 call_assembler_ref_like`       | same as int |
/// | `call_assembler_float_pyre`         | `:3489 call_assembler_float_like`     | same as int |
/// | `call_assembler_void_pyre`          | `:3370 call_assembler_void_like`      | `4 + num_args*3`  (omits `dst u16`) |
/// | `cond_call_void_pyre`               | `:2642 call_cond_like`                | `5 + arg_count*3`  (`first_reg u16 + fn_ptr_idx u16 + arg_count u8 + arg_count × (kind u8) + arg_count × (reg u16)`) |
/// | `record_known_result_int_pyre`      | `:2618 call_cond_like`                | same as cond_call |
/// | `record_known_result_ref_pyre`      | `:2630 call_cond_like`                | same as cond_call |
/// | `cond_call_value_int_pyre`          | `:2660 call_cond_value_like`          | `7 + arg_count*3`  (cond_call layout + trailing `dst u16`) |
/// | `cond_call_value_ref_pyre`          | `:2660 call_cond_value_like`          | same as int variant |
fn pyre_p_payload_len(opname: &str, code: &[u8], cursor: usize) -> Option<usize> {
    match opname {
        "inline_call_pyre_nested" => {
            // sub_idx u16 + num_args u16 + num_args × (kind u8, caller_src
            // u16, callee_dst u16) + return_i u16 + return_r u16 + return_f u16
            let num_args =
                u16::from_le_bytes([*code.get(cursor + 2)?, *code.get(cursor + 3)?]) as usize;
            Some(4 + num_args * 5 + 6)
        }
        "call_assembler_int_pyre" | "call_assembler_ref_pyre" | "call_assembler_float_pyre" => {
            // target_idx u16 + dst u16 + num_args u16 + num_args × (kind u8, reg u16)
            let num_args =
                u16::from_le_bytes([*code.get(cursor + 4)?, *code.get(cursor + 5)?]) as usize;
            Some(6 + num_args * 3)
        }
        "call_assembler_void_pyre" => {
            // target_idx u16 + num_args u16 + num_args × (kind u8, reg u16)
            let num_args =
                u16::from_le_bytes([*code.get(cursor + 2)?, *code.get(cursor + 3)?]) as usize;
            Some(4 + num_args * 3)
        }
        "cond_call_void_pyre" | "record_known_result_int_pyre" | "record_known_result_ref_pyre" => {
            // first_reg u16 + fn_ptr_idx u16 + arg_count u8 + arg_count × (kind u8) + arg_count × (reg u16)
            let arg_count = *code.get(cursor + 4)? as usize;
            Some(5 + arg_count * 3)
        }
        "cond_call_value_int_pyre" | "cond_call_value_ref_pyre" => {
            // cond_call shape + trailing dst u16
            let arg_count = *code.get(cursor + 4)? as usize;
            Some(7 + arg_count * 3)
        }
        _ => None,
    }
}

/// Statically walk one jitcode instruction starting at `pc`. Returns
/// `None` when `pc` is beyond `code.len()` or when the opcode byte at
/// `pc` is unknown to the `insns` table.
///
/// The `argcodes` char-byte mapping follows `blackhole.py:112-157`:
///
/// ```text
///   'i'|'c'|'r'|'f'  -> 1 byte (register index or signed const byte)
///   'L'              -> 2 bytes (unsigned label)
///   'd'|'j'          -> 2 bytes (descr index)
///   'I'|'R'|'F'      -> 1 + N bytes (N = first byte = list length)
///   '>' + ('i'|'r'|'f') -> 1 byte (result destination register)
/// ```
///
/// `live/` is special-cased to advance by `liveness::OFFSET_SIZE` per
/// `blackhole.py:1603-1605` (`bhimpl_live(pc): return pc + OFFSET_SIZE`).
pub fn decode_op_at(code: &[u8], pc: usize) -> Option<DecodedOp> {
    let opcode_byte = *code.get(pc)?;
    let key: &'static str = INSNS_BYTE_TO_OPNAME.get(&opcode_byte)?.as_str();
    let (opname, argcodes) = split_key(key);

    let mut cursor = pc + 1;
    if opname == "live" {
        // blackhole.py:1603-1605 bhimpl_live(pc): position += OFFSET_SIZE.
        // The `live/` key has empty argcodes so the generic walker would
        // advance 0 bytes, but dispatch skips 2 bytes of liveness offset.
        cursor += majit_translate::liveness::OFFSET_SIZE;
        if cursor > code.len() {
            return None;
        }
        return Some(DecodedOp {
            key,
            opname,
            argcodes,
            pc,
            next_pc: cursor,
        });
    }

    let mut chars = argcodes.chars();
    while let Some(c) = chars.next() {
        match c {
            'i' | 'c' | 'r' | 'f' => cursor += 1,
            'L' | 'd' | 'j' => cursor += 2,
            'I' | 'R' | 'F' => {
                // blackhole.py:139-143: varlist opens with a 1-byte length,
                // followed by that many 1-byte register indices.
                let list_len = *code.get(cursor)? as usize;
                cursor += 1 + list_len;
            }
            '>' => {
                // blackhole.py:185-209: result destination is 1 byte,
                // following `>i`, `>r`, or `>f`.
                let rt = chars.next()?;
                if !matches!(rt, 'i' | 'r' | 'f') {
                    return None;
                }
                cursor += 1;
            }
            // PRE-EXISTING-ADAPTATION pseudo-argcode for pyre-only
            // helper-side opnames.  Each `*_pyre/P` opname carries its
            // own flat payload shape; see `pyre_p_payload_len` for the
            // table.
            'P' => {
                cursor += pyre_p_payload_len(opname, code, cursor)?;
            }
            _ => return None,
        }
    }

    if cursor > code.len() {
        return None;
    }
    Some(DecodedOp {
        key,
        opname,
        argcodes,
        pc,
        next_pc: cursor,
    })
}

/// Iterator over every instruction in a jitcode `code` slice. Yields
/// `DecodedOp` in linear order (fallthrough layout — branch targets in
/// `L`-typed operands are not followed). Stops on the first decode
/// failure, which surfaces either end-of-code or an insns-table miss.
pub fn decoded_ops(code: &[u8]) -> impl Iterator<Item = DecodedOp> + '_ {
    let mut pc = 0;
    std::iter::from_fn(move || {
        let op = decode_op_at(code, pc)?;
        pc = op.next_pc;
        Some(op)
    })
}

fn split_key(key: &str) -> (&str, &str) {
    match key.split_once('/') {
        Some((name, codes)) => (name, codes),
        None => (key, ""),
    }
}

/// Where a resolved operand came from and the value read at that slot.
///
/// RPython `blackhole.py:112-157` argcodes consume a register index byte
/// or a small-constant byte and produce the value the `bhimpl_*` method
/// receives. This enum captures both the source byte(s) and the resolved
/// value, so diagnostics and shadow-execution paths can surface either
/// without re-walking the code.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedOperand {
    /// `i`: `self.registers_i[code[pc]]` (blackhole.py:120).
    IntReg { reg: u8, value: i64 },
    /// `c`: signed byte treated as a small constant (blackhole.py:121-123,
    /// `signedord`).
    ConstByte { byte: i8 },
    /// `r`: `self.registers_r[code[pc]]` (blackhole.py:124-126).
    RefReg { reg: u8, value: i64 },
    /// `f`: `self.registers_f[code[pc]]` (blackhole.py:127-129).
    FloatReg { reg: u8, value: i64 },
    /// `L`: 2-byte little-endian unsigned label (blackhole.py:133-138).
    Label { target: u16 },
    /// `d`|`j`: 2-byte little-endian descr index into
    /// `metainterp_sd.descrs` (blackhole.py:150-157). `j` carries the
    /// additional RPython assertion `isinstance(value, JitCode)`; the
    /// resolver here records the index only.
    DescrIdx { index: u16, is_jitcode: bool },
    /// `I`: `[registers_i[idx] for idx in list]` (blackhole.py:139-143 via
    /// `_get_list_of_values`).
    IntList(Vec<(u8, i64)>),
    /// `R`: ref-list variant of the above.
    RefList(Vec<(u8, i64)>),
    /// `F`: float-list variant of the above.
    FloatList(Vec<(u8, i64)>),
}

/// Where a `bhimpl_*` result would be written back.
///
/// RPython `blackhole.py:185-223` handles `>i`, `>r`, `>f` result slots
/// (and the `iL` split for `goto_if_*` which the resolver treats as
/// `Int` here — the shadow layer can interpret the `L`-branch later).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedResult {
    Int { reg: u8 },
    Ref { reg: u8 },
    Float { reg: u8 },
}

/// One `DecodedOp` enriched with resolved operand values and the result
/// destination (if any). Construction is pure — walking the byte stream
/// with a read-only register file.
///
/// RPython parity: `_get_method.handler` up to the `unboundmethod(*args)`
/// call. The bhimpl dispatch itself is intentionally left out; this
/// struct is the data the shadow-record layer (Phase D-2) or a diff-only
/// analyzer can inspect without executing any side effect.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedOp {
    pub decoded: DecodedOp,
    pub operands: Vec<ResolvedOperand>,
    pub result: Option<ResolvedResult>,
}

/// Read-only view over the three register files a `bhimpl_*` would
/// consume. Empty slots read as 0 (RPython has no corresponding case —
/// in-range reads are a translation-time invariant). The resolver here
/// returns `None` for out-of-range indices so the caller can treat an
/// oversized jitcode as a decode failure rather than silently misread.
#[derive(Debug, Clone, Copy)]
pub struct RegisterFileView<'a> {
    pub registers_i: &'a [i64],
    pub registers_r: &'a [i64],
    pub registers_f: &'a [i64],
}

impl<'a> RegisterFileView<'a> {
    pub fn empty() -> Self {
        Self {
            registers_i: &[],
            registers_r: &[],
            registers_f: &[],
        }
    }
}

/// Port of `_get_method.handler` operand-resolution phase (everything up
/// to the `unboundmethod(*args)` call at blackhole.py:170).
///
/// Decodes the op at `pc`, then walks `argcodes` once more — this time
/// reading each operand byte *and* resolving it via the register files.
/// Returns `None` on any of:
///   - `pc` beyond `code.len()` or unknown opcode byte (same as
///     `decode_op_at`).
///   - register index out of range for the active file.
///   - unrecognized argcode char.
///
/// The function is intentionally stateless; the caller supplies a
/// `RegisterFileView` borrowing from whatever concrete register storage
/// the shadow layer uses.
pub fn resolve_op_at(code: &[u8], pc: usize, regs: RegisterFileView<'_>) -> Option<ResolvedOp> {
    let decoded = decode_op_at(code, pc)?;
    if decoded.opname == "live" {
        // bhimpl_live consumes OFFSET_SIZE operand bytes but produces no
        // resolved operands — decode_op_at already advanced past them.
        return Some(ResolvedOp {
            decoded,
            operands: Vec::new(),
            result: None,
        });
    }

    let mut cursor = pc + 1;
    let mut operands: Vec<ResolvedOperand> = Vec::new();
    let mut result: Option<ResolvedResult> = None;
    let mut chars = decoded.argcodes.chars();
    while let Some(c) = chars.next() {
        match c {
            'i' => {
                let reg = *code.get(cursor)?;
                cursor += 1;
                let value = *regs.registers_i.get(reg as usize)?;
                operands.push(ResolvedOperand::IntReg { reg, value });
            }
            'c' => {
                // blackhole.py:121-123 `signedord`: signed byte constant.
                let byte = *code.get(cursor)? as i8;
                cursor += 1;
                operands.push(ResolvedOperand::ConstByte { byte });
            }
            'r' => {
                let reg = *code.get(cursor)?;
                cursor += 1;
                let value = *regs.registers_r.get(reg as usize)?;
                operands.push(ResolvedOperand::RefReg { reg, value });
            }
            'f' => {
                let reg = *code.get(cursor)?;
                cursor += 1;
                let value = *regs.registers_f.get(reg as usize)?;
                operands.push(ResolvedOperand::FloatReg { reg, value });
            }
            'L' => {
                let lo = *code.get(cursor)? as u16;
                let hi = *code.get(cursor + 1)? as u16;
                cursor += 2;
                operands.push(ResolvedOperand::Label {
                    target: lo | (hi << 8),
                });
            }
            'd' | 'j' => {
                let lo = *code.get(cursor)? as u16;
                let hi = *code.get(cursor + 1)? as u16;
                cursor += 2;
                operands.push(ResolvedOperand::DescrIdx {
                    index: lo | (hi << 8),
                    is_jitcode: c == 'j',
                });
            }
            'I' | 'R' | 'F' => {
                let list_len = *code.get(cursor)? as usize;
                cursor += 1;
                let mut entries = Vec::with_capacity(list_len);
                let file: &[i64] = match c {
                    'I' => regs.registers_i,
                    'R' => regs.registers_r,
                    'F' => regs.registers_f,
                    _ => unreachable!(),
                };
                for _ in 0..list_len {
                    let reg = *code.get(cursor)?;
                    cursor += 1;
                    let value = *file.get(reg as usize)?;
                    entries.push((reg, value));
                }
                operands.push(match c {
                    'I' => ResolvedOperand::IntList(entries),
                    'R' => ResolvedOperand::RefList(entries),
                    'F' => ResolvedOperand::FloatList(entries),
                    _ => unreachable!(),
                });
            }
            '>' => {
                let rt = chars.next()?;
                let reg = *code.get(cursor)?;
                cursor += 1;
                result = Some(match rt {
                    'i' => ResolvedResult::Int { reg },
                    'r' => ResolvedResult::Ref { reg },
                    'f' => ResolvedResult::Float { reg },
                    _ => return None,
                });
            }
            // PRE-EXISTING-ADAPTATION: pyre `*_pyre/P` payloads are
            // opaque to the canonical operand-resolver — no matching
            // `ResolvedOperand` variant exists.  Advance the cursor by
            // the per-opname computed length so the trailing
            // `cursor == decoded.next_pc` invariant holds; consumers
            // dispatch on `decoded.opname` for payload-aware handling.
            'P' => {
                cursor += pyre_p_payload_len(decoded.opname, code, cursor)?;
            }
            _ => return None,
        }
    }

    // Sanity: the walker here must land on the same next_pc decode_op_at
    // computed. If not, our argcodes handling disagrees with decode_op_at
    // and something silently miscounted operand bytes.
    debug_assert_eq!(
        cursor, decoded.next_pc,
        "resolve_op_at cursor {cursor} != decode_op_at next_pc {} for key {}",
        decoded.next_pc, decoded.key,
    );
    Some(ResolvedOp {
        decoded,
        operands,
        result,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_jitcodes_without_error() {
        let jitcodes = all_jitcodes();
        assert!(!jitcodes.is_empty(), "expected at least one jitcode");
    }

    #[test]
    fn deserializes_arms_without_error() {
        let arms = all_opcode_arms();
        assert!(!arms.is_empty(), "expected at least one opcode arm");
    }

    #[test]
    fn portal_jitcode_resolves_to_unique_jitdriver_entry() {
        // Phase G G.2 — verify the portal accessor returns the single
        // build-time JitCode whose `jitdriver_sd` is set (RPython
        // call.py:147 `jd.mainjitcode = self.get_jitcode(jd.portal_graph)`).
        // Production identity is currently `execute_opcode_step` because
        // `pyre-jit-trace/build.rs` does not yet include
        // `pyre/pyre-jit/src/eval.rs` in its source manifest. The
        // assertion is on uniqueness + jitdriver flag, not name, so the
        // test stays green when the manifest later widens to make
        // `eval_loop_jit` the portal.
        let portal = portal_jitcode().expect("build-time pipeline must register a portal jitcode");
        assert!(
            portal.jitdriver_sd().is_some(),
            "portal jitcode must carry a populated `jitdriver_sd` (call.py:148)"
        );
        let jitdriver_count = all_jitcodes()
            .iter()
            .filter(|jc| jc.jitdriver_sd().is_some())
            .count();
        assert_eq!(
            jitdriver_count, 1,
            "RPython call.py:147 invariant: exactly one portal jitcode per JitDriverStaticData"
        );
    }

    #[test]
    fn pop_top_lookup() {
        // Phase D-1 MVP target: Instruction::PopTop is arm_id=14 at build
        // time (was 13 before the StoreGlobal arm split out from
        // StoreName).  Confirm arm → jitcode resolution works end-to-end
        // and the jitcode carries bytecode bytes (not an empty shell).
        let jc = jitcode_for_arm(14).expect("PopTop arm should resolve to a jitcode");
        assert!(
            !jc.code.is_empty(),
            "PopTop jitcode should have non-empty bytecode"
        );
        assert_eq!(
            jc.name, "Instruction::PopTop#14",
            "jitcode name should match the arm selector"
        );
    }

    #[test]
    fn extract_variant_name_handles_unit_and_struct_variants() {
        assert_eq!(extract_variant_name("PopTop"), "PopTop");
        assert_eq!(extract_variant_name("LoadFast { var_num: 3 }"), "LoadFast");
        assert_eq!(extract_variant_name("Resume { arg: 0 }"), "Resume");
    }

    #[test]
    fn instruction_variant_name_round_trips_via_debug() {
        let instr = Instruction::PopTop;
        assert_eq!(instruction_variant_name(&instr), "PopTop");
    }

    #[test]
    fn arm_id_for_pop_top_matches_arm_14() {
        // PopTop is a single-variant arm; `Instruction::PopTop` must
        // resolve to the same arm_id as the direct `jitcode_for_arm(14)`
        // lookup above.
        let arm_id =
            arm_id_for_instruction(&Instruction::PopTop).expect("PopTop must resolve to an arm_id");
        assert_eq!(arm_id, 14);
    }

    #[test]
    fn jitcode_for_instruction_matches_arm_lookup() {
        let jc = jitcode_for_instruction(&Instruction::PopTop)
            .expect("PopTop must resolve to a jitcode");
        assert_eq!(jc.name, "Instruction::PopTop#14");
        assert!(!jc.code.is_empty());
    }

    #[test]
    fn insns_table_is_populated() {
        let table = insns_opname_to_byte();
        assert!(
            !table.is_empty(),
            "pipeline.insns should contain at least the core ops"
        );
    }

    #[test]
    fn opname_round_trips_through_byte() {
        // RPython assembler.py keys are `opname/argcode` (the argcode is
        // appended during `write_insn`). `live/` — the canonical BC_LIVE
        // marker emitted by liveness insertion — is the stablest key to
        // assert on, since every flattened graph touches it at least
        // once. If this test fails the assembler stopped emitting BC_LIVE,
        // which is a bigger parity break.
        let byte = *insns_opname_to_byte()
            .get("live/")
            .expect("`live/` opcode must be in the insns table");
        assert_eq!(opname_for_byte(byte), Some("live/"));
    }

    #[test]
    fn first_byte_of_pop_top_jitcode_decodes() {
        // End-to-end: pop top's jitcode bytes must start with an opcode
        // byte that `opname_for_byte` can decode.
        let jc = jitcode_for_instruction(&Instruction::PopTop)
            .expect("PopTop must resolve to a jitcode");
        let first = *jc
            .code
            .first()
            .expect("PopTop jitcode should have at least one opcode byte");
        assert!(
            opname_for_byte(first).is_some(),
            "first byte {first} of PopTop jitcode is unknown to the insns table",
        );
    }

    #[test]
    fn decode_live_skips_offset_size() {
        // `live/` is special-cased: empty argcodes but +OFFSET_SIZE (2)
        // bytes per blackhole.py:1603-1605.
        let live_byte = *insns_opname_to_byte()
            .get("live/")
            .expect("`live/` must be in insns table");
        let code = [live_byte, 0x00, 0x00];
        let op = decode_op_at(&code, 0).expect("live/ must decode");
        assert_eq!(op.opname, "live");
        assert_eq!(op.argcodes, "");
        assert_eq!(op.pc, 0);
        assert_eq!(
            op.next_pc,
            1 + majit_translate::liveness::OFFSET_SIZE,
            "live/ must advance by OFFSET_SIZE past the opcode byte",
        );
    }

    #[test]
    fn decode_int_add_reads_ii_operands_and_one_result_byte() {
        // `int_add/ii>i` — 1+1 operand bytes + 1 result byte = 3 bytes
        // after the opcode.
        let op_byte = *insns_opname_to_byte()
            .get("int_add/ii>i")
            .expect("int_add/ii>i must be in insns table");
        let code = [op_byte, 0x01, 0x02, 0x03];
        let op = decode_op_at(&code, 0).expect("int_add must decode");
        assert_eq!(op.opname, "int_add");
        assert_eq!(op.argcodes, "ii>i");
        assert_eq!(op.next_pc, 4);
    }

    #[test]
    fn decode_pop_top_jitcode_walks_to_end() {
        // PopTop's jitcode is ~41 bytes. Walking with `decoded_ops`
        // must reach exactly code.len() if every byte decodes cleanly.
        let jc = jitcode_for_instruction(&Instruction::PopTop)
            .expect("PopTop must resolve to a jitcode");
        let mut last_next = 0;
        let mut step_count = 0;
        for op in decoded_ops(&jc.code) {
            last_next = op.next_pc;
            step_count += 1;
        }
        assert!(step_count > 0, "should decode at least one op");
        assert_eq!(
            last_next,
            jc.code.len(),
            "decoded stream must end exactly at code.len() for PopTop \
             (stopped at {last_next} after {step_count} ops, code.len()={})",
            jc.code.len(),
        );
    }

    #[test]
    fn pop_top_jitcode_op_sequence_matches_expected_shape() {
        // Phase D-2 advance: lock in the assembler-emitted op sequence
        // for `Instruction::PopTop`'s arm jitcode. The shape mirrors
        // what RPython's `assemble.assemble(ssarepr, jitcode)`
        // (assembler.py:60-86) would emit for a jtransformed
        // `direct_call(pop_value) ; (return-or-raise)` graph: an
        // `inline_call_*` for the sub-jitcode, a `live/` jit_merge_point
        // marker, the standard `catch_exception/goto/reraise` exception
        // shoulder, then the trampoline tail.
        //
        // RPython parity: same byte stream `BlackholeInterpBuilder.dispatch_loop`
        // walks (blackhole.py:65-100). When the codewriter pipeline
        // changes shape this test fails — that's the contract: it's
        // the smallest end-to-end witness that
        // `pipeline.opcode_dispatch[PopTop].flattened` produces a
        // dispatch-able byte stream.
        let jc = jitcode_for_instruction(&Instruction::PopTop)
            .expect("PopTop must resolve to a jitcode");
        assert_eq!(
            jc.code.len(),
            25,
            "PopTop jitcode size shifted — refresh the expected sequence below",
        );
        let sequence: Vec<(String, String)> = decoded_ops(&jc.code)
            .map(|op| (op.opname.to_string(), op.argcodes.to_string()))
            .collect();
        // Note: the trailing `int_copy + residual_call_r_r` pair that
        // previously wrapped `Ok(...)` is gone — jtransform now elides
        // single-arg `Ok`/`Err`/`Some` constructor calls via
        // `RewriteResult::Identity` so the trace recorder no longer
        // emits a synthetic `CallR` for the return-type wrapper.
        let expected: Vec<(String, String)> = [
            ("inline_call_r_r", "dR>r"),
            ("live", ""),
            ("catch_exception", "L"),
            ("goto", "L"),
            ("reraise", ""),
            ("ref_return", "r"),
            ("live", ""),
            ("raise", "r"),
            ("ref_return", "r"),
        ]
        .iter()
        .map(|(o, a)| (o.to_string(), a.to_string()))
        .collect();
        assert_eq!(
            sequence, expected,
            "PopTop op sequence drifted — investigate before updating",
        );
    }

    #[test]
    fn decode_varlist_reads_length_byte_plus_items() {
        // Synthetic: inline_call_ir_r/dIR>r — d(2) + I(1+N) + R(1+M) + r(1).
        let op_byte = *insns_opname_to_byte()
            .get("inline_call_ir_r/dIR>r")
            .expect("inline_call_ir_r/dIR>r must be in insns table");
        // opcode, d=0x0001, I-list len=2 + 2 regs, R-list len=1 + 1 reg, dst
        // = 1 + 2 + (1+2) + (1+1) + 1 = 9 bytes
        let code = [
            op_byte, 0x01, 0x00, // d
            0x02, 0x00, 0x01, // I: len=2, [0,1]
            0x01, 0x00, // R: len=1, [0]
            0x03, // >r: dst=3
        ];
        let op = decode_op_at(&code, 0).expect("inline_call_ir_r must decode");
        assert_eq!(op.opname, "inline_call_ir_r");
        assert_eq!(op.next_pc, 9);
    }

    #[test]
    fn decode_unknown_opcode_returns_none() {
        // Byte 0xFF should not be a valid opcode — 21 entries go 0..=20.
        let code = [0xFF];
        assert!(
            decode_op_at(&code, 0).is_none(),
            "unknown opcode byte must yield None",
        );
    }

    #[test]
    fn resolve_int_add_reads_both_register_values() {
        // `int_add/ii>i`: canonical — both operands read from int-regs,
        // result written to int-reg. RPython
        // `blackhole.py:@arguments("i", "i", returns="i")`.
        let op_byte = *insns_opname_to_byte()
            .get("int_add/ii>i")
            .expect("int_add/ii>i must be in insns table");
        // code: [opcode, i_idx=2, i_idx=1, dst=0]
        let code = [op_byte, 0x02, 0x01, 0x00];
        let regs = RegisterFileView {
            registers_i: &[0, 7, 42, 0],
            registers_r: &[],
            registers_f: &[],
        };
        let op = resolve_op_at(&code, 0, regs).expect("int_add must resolve");
        assert_eq!(op.decoded.opname, "int_add");
        assert_eq!(op.operands.len(), 2);
        assert_eq!(
            op.operands[0],
            ResolvedOperand::IntReg { reg: 2, value: 42 }
        );
        assert_eq!(op.operands[1], ResolvedOperand::IntReg { reg: 1, value: 7 });
        assert_eq!(op.result, Some(ResolvedResult::Int { reg: 0 }));
    }

    #[test]
    fn resolve_live_yields_no_operands() {
        // `live/`: empty argcodes, OFFSET_SIZE skipped; no operands, no
        // result.
        let live_byte = *insns_opname_to_byte()
            .get("live/")
            .expect("live/ must be in insns table");
        let code = [live_byte, 0x00, 0x00];
        let op = resolve_op_at(&code, 0, RegisterFileView::empty()).expect("live/ must resolve");
        assert!(op.operands.is_empty());
        assert!(op.result.is_none());
        assert_eq!(op.decoded.opname, "live");
    }

    #[test]
    fn resolve_out_of_range_int_reg_returns_none() {
        // int_add/ii>i: opcode reads registers_i[5], but registers_i is
        // only 2 wide. Must surface as decode failure, not a silent 0.
        let op_byte = *insns_opname_to_byte()
            .get("int_add/ii>i")
            .expect("int_add/ii>i must be in insns table");
        let code = [op_byte, 0x00, 0x05, 0x00];
        let regs = RegisterFileView {
            registers_i: &[10, 20],
            registers_r: &[],
            registers_f: &[],
        };
        assert!(resolve_op_at(&code, 0, regs).is_none());
    }

    #[test]
    fn resolve_varlist_reads_each_member() {
        // inline_call_ir_r/dIR>r: d(2) + I(len+items) + R(len+items) + >r(1).
        let op_byte = *insns_opname_to_byte()
            .get("inline_call_ir_r/dIR>r")
            .expect("inline_call_ir_r/dIR>r must be in insns table");
        // descr=0x0102, I=[reg1, reg2], R=[reg0], dst=4
        let code = [
            op_byte, 0x02, 0x01, //
            0x02, 0x01, 0x02, //
            0x01, 0x00, //
            0x04,
        ];
        let regs = RegisterFileView {
            registers_i: &[0, 111, 222],
            registers_r: &[333],
            registers_f: &[],
        };
        let op = resolve_op_at(&code, 0, regs).expect("inline_call_ir_r must resolve");
        assert_eq!(op.operands.len(), 3);
        assert_eq!(
            op.operands[0],
            ResolvedOperand::DescrIdx {
                index: 0x0102,
                is_jitcode: false,
            },
        );
        assert_eq!(
            op.operands[1],
            ResolvedOperand::IntList(vec![(1, 111), (2, 222)]),
        );
        assert_eq!(op.operands[2], ResolvedOperand::RefList(vec![(0, 333)]));
        assert_eq!(op.result, Some(ResolvedResult::Ref { reg: 4 }));
    }

    #[test]
    fn arm_id_covers_or_grouped_variants() {
        // arm_id=3 is the `Instruction::LoadFast | Instruction::LoadFastBorrow`
        // group (see build-time opcode_dispatch). Both variants must land on
        // the same arm_id — matching the RPython model where `Or` selectors
        // register a single Python method under each opcode.
        use pyre_interpreter::bytecode::Arg;
        let id_a = arm_id_for_instruction(&Instruction::LoadFast {
            var_num: Arg::marker(),
        })
        .expect("LoadFast must resolve");
        let id_b = arm_id_for_instruction(&Instruction::LoadFastBorrow {
            var_num: Arg::marker(),
        })
        .expect("LoadFastBorrow must resolve");
        assert_eq!(id_a, id_b, "Or-grouped variants must share an arm_id");
    }

    // Task #85 closure check: the strict builder covered every opname
    // in the generated insns table at the time the snapshot was set.
    // Earlier revisions kept this ignored while kind-flow bugs emitted
    // pyre-only mixed signatures such as `int_ge/ir>i`; the
    // codewriter/rtyper/jtransform parity fixes made those opnames
    // disappear at the source.  This variant now uses the report
    // builder (not the strict-panic builder) so the symmetry assertion
    // tolerates the known unwired set documented in
    // `default_bh_builder_unwired_set_matches_task_85_snapshot`.
    #[test]
    fn build_default_bh_builder_matches_insns_table() {
        // Slice 3a: the runtime-side `BlackholeInterpBuilder` is reachable
        // from pyre-jit-trace. After `setup_insns + wire_bhimpl_handlers`
        // it must carry the same byte<->opname mapping as the build-time
        // insns bincode, and it must resolve the three well-known
        // opcodes (`live/`, `catch_exception/L`, `rvmprof_code/ii`) when
        // they appear in the table.
        let (builder, _unwired) = build_default_bh_builder_with_unwired_report();
        let expected_live = insns_opname_to_byte().get("live/").copied();
        assert_eq!(Some(builder.op_live), expected_live);
        // Reverse mapping parity: every opname in the build-time table
        // must appear at the same byte index in builder._insns.
        for (key, &byte) in insns_opname_to_byte() {
            assert_eq!(
                &builder._insns[byte as usize], key,
                "builder._insns[{byte}] disagrees with build-time key {key:?}",
            );
        }
    }

    #[test]
    fn default_bh_builder_unwired_set_matches_task_85_snapshot() {
        // Task #85 lock-in: every generated opname must either be wired
        // by `wire_bhimpl_handlers`, or be on this known-gap list that
        // documents a pending upstream rewrite.  A new entry outside
        // that list means codewriter/regalloc emitted a kind shape that
        // no RPython blackhole handler has — fix the upstream emission,
        // do NOT add a `*_r>i`/`*_ir>i` bhhandler alias (pyre-only
        // mixed signatures masked over real type-inference gaps and
        // were rolled back in `jtransform: drop Ref-operand
        // auto-coercion arms`).
        //
        // Currently expected unwired entries (TODO: close each one by
        // implementing the named upstream rewrite, not by wiring an
        // alias):
        //
        // - `int_mod/ii>i` — RPython `jtransform.py:576-577
        //   rewrite_op_int_mod` is wired to `_do_builtin_call`, which
        //   replaces the SpaceOperation with `direct_call(ll_int_py_mod,
        //   ...)` BEFORE the assembler encoder runs, so upstream's
        //   `pipeline.insns` never carries the bare `int_mod` opname.
        //   pyre's `jit_codewriter/assembler.rs:2778-2789` emits the
        //   opname directly from `syn::BinOp::Rem` on `i64` operands
        //   without an intervening rewrite, so the key leaks into
        //   `pipeline.insns`.  Closing this requires porting
        //   `_do_builtin_call` (or equivalent) into pyre's IR-rewrite
        //   phase so the SSA op converts to a `Call` targeting
        //   `ll_int_py_mod` before encode.  `int_floordiv/ii>i` has
        //   the same parity gap in principle but does not currently
        //   appear because no analyzed source emits it (Rust's `i64 /
        //   i64` truncates toward zero, distinct from Python's
        //   floor-division semantics, so the front-end may translate
        //   it differently — leave it off this list until it surfaces).
        let (_builder, mut unwired) = build_default_bh_builder_with_unwired_report();
        unwired.sort();
        let expected: Vec<String> = vec!["int_mod/ii>i".to_string()];
        assert_eq!(
            unwired, expected,
            "Task #85 unwired-opname snapshot drifted. If a new entry \
             appeared, find the new kind-flow bug upstream instead of \
             adding a bhhandler alias.  Existing entries document a \
             pending upstream rewrite — see the `expected` literal.",
        );
    }

    #[test]
    fn default_bh_builder_handler_coverage_report() {
        // Diagnostic: surface the opnames in the real insns table that
        // `wire_bhimpl_handlers` did NOT override. These fall back to
        // the `setup_insns` placeholder and would panic on dispatch.
        //
        // `BlackholeInterpBuilder::unwired_opnames()` is the accessor
        // that returns the gap. Each unwired opname is a concrete
        // bhimpl port that Phase D-2 must land before flipping the
        // production dispatch path to insns-table-driven.
        //
        // The test does NOT fail on unwired opnames — it just reports
        // them. Gating turns on later once the table path is the sole
        // production dispatch route.  Use the report builder rather
        // than the strict-panic variant so the diagnostic survives
        // documented unwired entries (see
        // `default_bh_builder_unwired_set_matches_task_85_snapshot`).
        let (builder, _) = build_default_bh_builder_with_unwired_report();
        let total = insns_opname_to_byte().len();
        let mut unwired: Vec<&str> = builder.unwired_opnames();
        unwired.sort_unstable();
        let wired = total - unwired.len();
        eprintln!(
            "[jitcode_runtime] coverage: {wired}/{total} opnames wired; \
             {} unwired: {:?}",
            unwired.len(),
            unwired,
        );
        // Sanity: `live/` must be present in the insns table and wired.
        // Any binary that lacks it is structurally broken.
        assert!(
            insns_opname_to_byte().contains_key("live/"),
            "live/ missing from real insns table — broken build",
        );
        assert!(
            !unwired.iter().any(|k| *k == "live/"),
            "live/ must be wired by wire_bhimpl_handlers",
        );
    }

    #[test]
    fn pop_top_jitcode_is_complete_in_canonical_store() {
        // RPython parity target: the deserialized `ALL_JITCODES` entries
        // are themselves the canonical objects produced by
        // `CodeWriter.make_jitcodes()`. Avoid the transitional
        // build→runtime `From` adapter here and assert directly on the
        // canonical object that build.rs persisted.
        let arm_id =
            arm_id_for_instruction(&Instruction::PopTop).expect("PopTop must resolve to an arm_id");
        let arm = get_arm(arm_id).expect("PopTop arm must exist");
        let bt_jc = jitcode_for_instruction(&Instruction::PopTop)
            .expect("PopTop must resolve to a jitcode");
        assert!(!bt_jc.code.is_empty());
        assert_eq!(bt_jc.name, "Instruction::PopTop#14");
        assert_eq!(arm.entry_jitcode_index, Some(bt_jc.index()));
        assert_eq!(
            bt_jc.num_regs_and_consts_i(),
            bt_jc.num_regs_i() + bt_jc.constants_i.len()
        );
        assert_eq!(
            bt_jc.num_regs_and_consts_r(),
            bt_jc.num_regs_r() + bt_jc.constants_r.len()
        );
        assert_eq!(
            bt_jc.num_regs_and_consts_f(),
            bt_jc.num_regs_f() + bt_jc.constants_f.len()
        );
    }

    #[test]
    fn dispatch_loop_executes_int_add_via_real_insns_table() {
        // Phase D-2.0: confirm the build-time `pipeline.insns` byte
        // assignments resolve to the real `wire_bhimpl_handlers`
        // dispatch entries — a hand-assembled bytecode using those
        // bytes runs end-to-end through
        // `BlackholeInterpBuilder::dispatch_loop` and lands the
        // expected `bhimpl_int_add` result.
        //
        // RPython parity: same shape as `setup_insns + dispatch_loop +
        // bhimpl_int_add` (blackhole.py:66-100 + 452-460), but driven
        // by the artifact this binary actually loads — not a synthetic
        // 3-entry insns dict like the analogous test inside
        // majit-metainterp. Closes the build-artifact → runtime →
        // BlackholeInterpBuilder round trip end-to-end.
        //
        // Use the reporting helper here because this test inspects a tiny
        // synthetic bytecode slice and does not need the strict coverage
        // assertion exercised by `build_default_bh_builder()`.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let int_add_byte = *table
            .get("int_add/ii>i")
            .expect("`int_add/ii>i` must be in insns");
        let int_return_byte = *table
            .get("int_return/i")
            .expect("`int_return/i` must be in insns");

        // live + int_add(r0, r1) → r2 + int_return(r2). The two zero
        // bytes after `live/` are the OFFSET_SIZE liveness offset that
        // `bhimpl_live` skips (blackhole.py:1603-1605).
        let code: Vec<u8> = vec![
            live_byte,
            0x00,
            0x00, //
            int_add_byte,
            0x00,
            0x01,
            0x02, //
            int_return_byte,
            0x02,
        ];

        let mut bh = builder.acquire_interp();
        bh.registers_i = vec![0i64; 3];
        bh.registers_i[0] = 10;
        bh.registers_i[1] = 32;

        let result = builder.dispatch_loop(&mut bh, &code, 0);
        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop must terminate with LeaveFrame after \
             int_return; got {result:?}",
        );
        assert_eq!(bh.tmpreg_i, 42, "int_add(10, 32) must produce 42");
    }

    #[test]
    fn dispatch_loop_chains_int_add_then_int_sub_via_real_insns_table() {
        // Phase D-2.1: chain two binops + a label-free linear control
        // flow through `dispatch_loop`. Validates that the second
        // binop reads the register the first wrote (multi-step value
        // flow through the register file) and that two distinct wired
        // bhimpl handlers (`bhimpl_int_add`, `bhimpl_int_sub`) advance
        // `position` correctly back to back.
        //
        // RPython parity: blackhole.py:452-460 `bhimpl_int_add` +
        // :462-464 `bhimpl_int_sub` chained with the same register
        // file, identical to RPython's per-op `_get_method.handler`
        // dispatch.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let int_add_byte = *table
            .get("int_add/ii>i")
            .expect("`int_add/ii>i` must be in insns");
        let int_sub_byte = *table
            .get("int_sub/ii>i")
            .expect("`int_sub/ii>i` must be in insns");
        let int_return_byte = *table
            .get("int_return/i")
            .expect("`int_return/i` must be in insns");

        // live + int_add(r0=10, r1=32) → r2 (=42)
        //      + int_sub(r2, r0)        → r3 (=32)
        //      + int_return(r3)
        let code: Vec<u8> = vec![
            live_byte,
            0x00,
            0x00, //
            int_add_byte,
            0x00,
            0x01,
            0x02, //
            int_sub_byte,
            0x02,
            0x00,
            0x03, //
            int_return_byte,
            0x03,
        ];

        let mut bh = builder.acquire_interp();
        bh.registers_i = vec![0i64; 4];
        bh.registers_i[0] = 10;
        bh.registers_i[1] = 32;

        let result = builder.dispatch_loop(&mut bh, &code, 0);
        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop must terminate with LeaveFrame after \
             int_return; got {result:?}",
        );
        assert_eq!(bh.registers_i[2], 42, "first int_add must store 42 into r2",);
        assert_eq!(
            bh.registers_i[3], 32,
            "second int_sub must store (42-10)=32 into r3",
        );
        assert_eq!(bh.tmpreg_i, 32, "int_return must place r3 into tmpreg_i",);
    }

    #[test]
    fn dispatch_loop_executes_count_up_loop_via_real_insns_table() {
        // Phase D-2.2: closed loop via `int_lt/ii>i` + `goto_if_not/iL`
        // + `goto/L` — exercises the dispatch_loop's absolute-target
        // label semantics on both backward and forward jumps.
        //
        // pyre's build-time assembler does not currently emit the
        // fused `goto_if_not_int_*` family; the orthodox unfused
        // shape (`int_lt` produces a 0/1 register, `goto_if_not`
        // dispatches on it) is what `pipeline.insns` actually
        // contains, so the loop is wired against the unfused pair.
        //
        // Loop body (count r0 from 0 up to r1=5, step r2=1):
        //
        //   PC=0:  live/
        //   PC=3:  LOOP: int_lt r0, r1 → r3
        //   PC=7:        goto_if_not r3, END   (forward jump)
        //   PC=11:       int_add r0, r2 → r0
        //   PC=15:       goto LOOP=3            (backward jump)
        //   PC=18: END:  int_return r0
        //
        // RPython parity: blackhole.py:864-869 `bhimpl_goto_if_not`
        // — target is an absolute byte offset into the jitcode
        // `code` array.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let int_lt_byte = *table
            .get("int_lt/ii>i")
            .expect("`int_lt/ii>i` must be in insns");
        let goto_if_not_byte = *table
            .get("goto_if_not/iL")
            .expect("`goto_if_not/iL` must be in insns");
        let int_add_byte = *table
            .get("int_add/ii>i")
            .expect("`int_add/ii>i` must be in insns");
        let goto_byte = *table.get("goto/L").expect("`goto/L` must be in insns");
        let int_return_byte = *table
            .get("int_return/i")
            .expect("`int_return/i` must be in insns");

        // PC offsets (must match the layout above): LOOP_HEAD = 3, END = 18.
        const LOOP_HEAD: u8 = 3;
        const END: u8 = 18;
        let code: Vec<u8> = vec![
            // PC=0:  live/  (1 + OFFSET_SIZE = 3 bytes)
            live_byte,
            0x00,
            0x00, //
            // PC=3:  int_lt r0, r1 → r3  (1 + 3 = 4 bytes)
            int_lt_byte,
            0x00,
            0x01,
            0x03, //
            // PC=7:  goto_if_not r3, END  (1 + 1 + 2 = 4 bytes)
            goto_if_not_byte,
            0x03,
            END,
            0x00, //
            // PC=11: int_add r0, r2 → r0  (1 + 3 = 4 bytes)
            int_add_byte,
            0x00,
            0x02,
            0x00, //
            // PC=15: goto LOOP_HEAD  (1 + 2 = 3 bytes)
            goto_byte,
            LOOP_HEAD,
            0x00, //
            // PC=18: int_return r0  (1 + 1 = 2 bytes)
            int_return_byte,
            0x00,
        ];
        assert_eq!(code.len(), 20, "loop bytecode must be exactly 20 bytes");

        let mut bh = builder.acquire_interp();
        bh.registers_i = vec![0i64; 4];
        bh.registers_i[0] = 0; // counter
        bh.registers_i[1] = 5; // limit
        bh.registers_i[2] = 1; // step
        // r3 is the int_lt result slot — left zero, written each loop iter.

        let result = builder.dispatch_loop(&mut bh, &code, 0);
        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop must terminate with LeaveFrame after the \
             loop's int_return; got {result:?}",
        );
        assert_eq!(
            bh.registers_i[0], 5,
            "counter r0 must reach the limit (5) before the loop \
             exits via int_lt(5,5)=0 → goto_if_not jumps to END",
        );
        assert_eq!(bh.tmpreg_i, 5, "int_return must place r0 into tmpreg_i");
    }

    #[test]
    fn dispatch_loop_executes_float_add_via_real_insns_table() {
        // Phase D-2.3: float register file + ff>f decode/encode +
        // void_return termination.
        //
        //   PC=0:  live/
        //   PC=3:  float_add r0, r1 → r2
        //   PC=7:  void_return
        //
        // RPython parity: blackhole.py:574-577 `bhimpl_float_add` +
        // :859-862 `bhimpl_void_return`. Pyre's `registers_f` stores
        // `f64::to_bits() as i64`; the `bhhandler_ff_f!` macro decodes
        // via `f64::from_bits` on read and `to_bits()` on write, so
        // the test setup mirrors that encoding.
        //
        // `float_return/f` is wired in `wire_bhimpl_handlers` but is
        // NOT in the build-time `pipeline.insns` (no jitcode emitted
        // by the production assembler currently returns a float — all
        // float ops feed back into either box_float or another float
        // op). The test validates the float register file + binop
        // decode end-to-end and inspects `registers_f[2]` directly.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let float_add_byte = *table
            .get("float_add/ff>f")
            .expect("`float_add/ff>f` must be in insns");
        let void_return_byte = *table
            .get("void_return/")
            .expect("`void_return/` must be in insns");

        let code: Vec<u8> = vec![
            live_byte,
            0x00,
            0x00, //
            float_add_byte,
            0x00,
            0x01,
            0x02, //
            void_return_byte,
        ];

        let mut bh = builder.acquire_interp();
        bh.registers_f = vec![0i64; 3];
        bh.registers_f[0] = (1.5_f64).to_bits() as i64;
        bh.registers_f[1] = (2.5_f64).to_bits() as i64;

        let result = builder.dispatch_loop(&mut bh, &code, 0);
        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop must terminate with LeaveFrame after \
             void_return; got {result:?}",
        );
        assert_eq!(
            f64::from_bits(bh.registers_f[2] as u64),
            4.0,
            "float_add(1.5, 2.5) must store 4.0 (bits) into r2",
        );
        assert_eq!(
            bh.return_type,
            majit_metainterp::blackhole::BhReturnType::Void,
            "void_return must set return_type = Void",
        );
    }

    #[test]
    fn dispatch_loop_executes_cross_type_cast_then_float_add() {
        // Phase D-2.4: cross-type register file bridge —
        // `cast_int_to_float/i>f` reads from `registers_i` and writes
        // to `registers_f` in the same dispatch pass.
        //
        //   PC=0:  live/
        //   PC=3:  cast_int_to_float r0 → f0
        //   PC=6:  cast_int_to_float r1 → f1
        //   PC=9:  float_add f0, f1 → f2
        //   PC=13: void_return
        //
        // RPython parity: `bhimpl_cast_int_to_float`
        // (blackhole.py:565-568) — `(a as f64).to_bits()`. The
        // dispatch loop must drive the int→float bridge twice and
        // then a float binop on the resulting registers without
        // crossing wires.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let cast_byte = *table
            .get("cast_int_to_float/i>f")
            .expect("`cast_int_to_float/i>f` must be in insns");
        let float_add_byte = *table
            .get("float_add/ff>f")
            .expect("`float_add/ff>f` must be in insns");
        let void_return_byte = *table
            .get("void_return/")
            .expect("`void_return/` must be in insns");

        let code: Vec<u8> = vec![
            live_byte,
            0x00,
            0x00, //
            cast_byte,
            0x00,
            0x00, // r0 → f0
            cast_byte,
            0x01,
            0x01, // r1 → f1
            float_add_byte,
            0x00,
            0x01,
            0x02, //
            void_return_byte,
        ];
        assert_eq!(code.len(), 14, "code must be 14 bytes");

        let mut bh = builder.acquire_interp();
        bh.registers_i = vec![10, 20];
        bh.registers_f = vec![0i64; 3];

        let result = builder.dispatch_loop(&mut bh, &code, 0);
        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop must terminate with LeaveFrame after \
             void_return; got {result:?}",
        );
        assert_eq!(
            f64::from_bits(bh.registers_f[0] as u64),
            10.0,
            "cast_int_to_float r0=10 must store 10.0 into f0",
        );
        assert_eq!(
            f64::from_bits(bh.registers_f[1] as u64),
            20.0,
            "cast_int_to_float r1=20 must store 20.0 into f1",
        );
        assert_eq!(
            f64::from_bits(bh.registers_f[2] as u64),
            30.0,
            "float_add(10.0, 20.0) must store 30.0 into f2",
        );
    }

    #[test]
    fn dispatch_loop_loads_constant_via_setposition_lifecycle() {
        // Phase D-2.5: full RPython-shape lifecycle —
        // `acquire_interp` + `setposition` + `dispatch_loop`. Earlier
        // dispatch_loop tests bypassed `setposition` by hand-setting
        // `bh.registers_i = vec![...]`; here we construct a real
        // runtime `JitCode` with `c_num_regs_i = 1` and
        // `constants_i = [42]` so `setposition` allocates a register
        // file of `num_regs_and_consts_i() = 2` slots and copies the
        // constant into slot 1 (RPython
        // `blackhole.py:312 setposition` parity).
        //
        //   slot 0 = scratch dst
        //   slot 1 = constant 42 (preloaded by setposition)
        //
        //   PC=0:  live/
        //   PC=3:  int_copy r1 → r0    (r0 := constant)
        //   PC=6:  int_return r0
        //
        // RPython parity: `bhimpl_int_copy` (blackhole.py:455-457)
        // reads from `registers_i[code[pc]]` and writes
        // `registers_i[code[pc+1]]`, which validates that the
        // constants area is reachable through the same register-index
        // protocol the bhimpl handlers consume.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let int_copy_byte = *table
            .get("int_copy/i>i")
            .expect("`int_copy/i>i` must be in insns");
        let int_return_byte = *table
            .get("int_return/i")
            .expect("`int_return/i` must be in insns");

        let code: Vec<u8> = vec![
            live_byte,
            0x00,
            0x00, //
            int_copy_byte,
            0x01,
            0x00, //   r1 (= constant) → r0
            int_return_byte,
            0x00,
        ];

        let jc = {
            let inner = majit_metainterp::jitcode::JitCode::new("test_setposition");
            inner.set_body(majit_translate::jitcode::JitCodeBody {
                c_num_regs_i: 1,
                constants_i: vec![42],
                code: code.clone(),
                ..Default::default()
            });
            std::sync::Arc::new(inner)
        };

        let mut bh = builder.acquire_interp();
        bh.setposition(jc.clone(), 0);

        // setposition must have allocated num_regs_i + constants_i.len() slots
        // and copied the constant into the upper half.
        assert_eq!(
            bh.registers_i.len(),
            2,
            "setposition must size registers_i to num_regs_and_consts_i = 2",
        );
        assert_eq!(
            bh.registers_i[1], 42,
            "setposition must copy constants_i[0]=42 into slot num_regs_i = 1",
        );

        let result = builder.dispatch_loop(&mut bh, &jc.code, 0);
        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop must terminate with LeaveFrame after \
             int_return; got {result:?}",
        );
        assert_eq!(
            bh.registers_i[0], 42,
            "int_copy r1→r0 must move the preloaded constant 42 into r0",
        );
        assert_eq!(bh.tmpreg_i, 42, "int_return must surface 42 via tmpreg_i");
    }

    #[test]
    fn dispatch_loop_executes_ref_return_via_real_insns_table() {
        // Phase D-2.6: ref register file + ref_return r-typed
        // termination — fills the third register-file dimension that
        // the earlier dispatch_loop tests did not touch.
        //
        // The build-time insns table has only `ref_return/r` from the
        // `ref_*` family (no `ref_copy/r>r`, no `ref_push/r`, no
        // `ref_pop/>r`) — pyre's production assembler does not emit
        // those today. So the smallest ref-typed round-trip uses
        // `ref_return/r` as the sole ref-side opcode and validates
        // that `registers_r[k]` is reachable through the standard
        // `r`-argcode protocol that `bhhandler_*` macros consume.
        //
        //   PC=0: live/
        //   PC=3: ref_return r0
        //
        // RPython parity: blackhole.py:847-851 `bhimpl_ref_return`.
        // `registers_r` and `tmpreg_r` store ref pointers as raw `i64`
        // bits; the test uses an arbitrary nonzero pattern to verify
        // the read is byte-for-byte without dereferencing.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let ref_return_byte = *table
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns");

        let code: Vec<u8> = vec![
            live_byte,
            0x00,
            0x00, //
            ref_return_byte,
            0x00,
        ];

        let mut bh = builder.acquire_interp();
        bh.registers_r = vec![0i64; 1];
        // Arbitrary nonzero pattern. Treated as a raw ref pointer; the
        // test does not dereference it.
        let probe: i64 = 0x1234_5678_9abc_def0_u64 as i64;
        bh.registers_r[0] = probe;

        let result = builder.dispatch_loop(&mut bh, &code, 0);
        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop must terminate with LeaveFrame after \
             ref_return; got {result:?}",
        );
        assert_eq!(
            bh.tmpreg_r, probe,
            "ref_return must place r0's bits into tmpreg_r byte-for-byte",
        );
        assert_eq!(
            bh.return_type,
            majit_metainterp::blackhole::BhReturnType::Ref,
            "ref_return must set return_type = Ref",
        );
    }

    #[test]
    fn dispatch_loop_with_probe_captures_opcode_sequence_and_preserves_result() {
        // Phase D-2.7: probe-hook variant of dispatch_loop — first
        // shadow-execution scaffold. Each dispatched opcode invokes
        // the probe BEFORE the handler runs, so a shadow caller
        // (Phase D plan: MIFrame side) can capture the jitcode op
        // sequence and compare it against the trace IR emitted by the
        // trait-based `execute_opcode_step`. The probe must NOT
        // change the dispatch result — running the same int_add
        // bytecode through `dispatch_loop_with_probe` must produce
        // the same `tmpreg_i==42` + LeaveFrame as the bare
        // `dispatch_loop` (D-2.0 baseline).
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let int_add_byte = *table
            .get("int_add/ii>i")
            .expect("`int_add/ii>i` must be in insns");
        let int_return_byte = *table
            .get("int_return/i")
            .expect("`int_return/i` must be in insns");

        // Same shape as D-2.0 baseline:
        //   PC=0: live/                3 bytes
        //   PC=3: int_add r0, r1 → r2  4 bytes
        //   PC=7: int_return r2        2 bytes
        let code: Vec<u8> = vec![
            live_byte,
            0x00,
            0x00, //
            int_add_byte,
            0x00,
            0x01,
            0x02, //
            int_return_byte,
            0x02,
        ];

        let mut bh = builder.acquire_interp();
        bh.registers_i = vec![0i64; 3];
        bh.registers_i[0] = 10;
        bh.registers_i[1] = 32;

        let mut captured: Vec<(usize, u8, String)> = Vec::new();
        let result =
            builder.dispatch_loop_with_probe(&mut bh, &code, 0, |_bh, pc, opcode, opname| {
                captured.push((pc, opcode, opname.to_string()));
            });

        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop_with_probe must terminate with LeaveFrame after \
             int_return; got {result:?}",
        );
        assert_eq!(
            bh.tmpreg_i, 42,
            "probe must not perturb dispatch — int_add(10,32)→42 still \
             surfaces via tmpreg_i",
        );

        // Probe fired exactly once per opcode at the opcode-byte
        // position (not after operand decode).
        assert_eq!(
            captured.len(),
            3,
            "probe must fire exactly once per opcode (live + int_add + \
             int_return); got {captured:?}",
        );
        assert_eq!(captured[0], (0, live_byte, "live/".to_string()));
        assert_eq!(captured[1], (3, int_add_byte, "int_add/ii>i".to_string()));
        assert_eq!(
            captured[2],
            (7, int_return_byte, "int_return/i".to_string())
        );
    }

    #[test]
    fn dispatch_loop_probe_observes_register_state_at_each_op() {
        // Phase D-2.8: probe receives `&BlackholeInterpreter` at each
        // firing — the second piece of shadow-execution scaffolding.
        // The closure can read register values BEFORE the upcoming
        // handler runs, capturing the input data flow needed to
        // compare against the trace IR emitted by the trait-based
        // `execute_opcode_step`.
        //
        // Bytecode: live + int_add(r0,r1)→r2 + int_sub(r2,r0)→r3
        //         + int_return(r3)  (same shape as D-2.1).
        //
        // Probe captures `registers_i[0..4]` at every firing. The
        // sequence must be:
        //
        //   probe[0] (live/):       [10, 32, 0, 0]   (initial state)
        //   probe[1] (int_add):     [10, 32, 0, 0]   (operands visible, dst still 0)
        //   probe[2] (int_sub):     [10, 32, 42, 0]  (int_add's effect now visible)
        //   probe[3] (int_return):  [10, 32, 42, 32] (int_sub's effect now visible)
        //
        // The non-trivial validation: at probe[2], `registers_i[2] == 42`
        // — proving the probe sees the live data flow exactly as the
        // chained handlers produce it. Shadow-execute logic can use
        // this to read input register values per op without re-running
        // the handler chain.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let int_add_byte = *table
            .get("int_add/ii>i")
            .expect("`int_add/ii>i` must be in insns");
        let int_sub_byte = *table
            .get("int_sub/ii>i")
            .expect("`int_sub/ii>i` must be in insns");
        let int_return_byte = *table
            .get("int_return/i")
            .expect("`int_return/i` must be in insns");

        let code: Vec<u8> = vec![
            live_byte,
            0x00,
            0x00, //
            int_add_byte,
            0x00,
            0x01,
            0x02, //
            int_sub_byte,
            0x02,
            0x00,
            0x03, //
            int_return_byte,
            0x03,
        ];

        let mut bh = builder.acquire_interp();
        bh.registers_i = vec![0i64; 4];
        bh.registers_i[0] = 10;
        bh.registers_i[1] = 32;

        let mut snapshots: Vec<(String, [i64; 4])> = Vec::new();
        let result =
            builder.dispatch_loop_with_probe(&mut bh, &code, 0, |bh_view, _pc, _opcode, opname| {
                snapshots.push((
                    opname.to_string(),
                    [
                        bh_view.registers_i[0],
                        bh_view.registers_i[1],
                        bh_view.registers_i[2],
                        bh_view.registers_i[3],
                    ],
                ));
            });

        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop_with_probe must terminate with LeaveFrame; \
             got {result:?}",
        );
        assert_eq!(bh.tmpreg_i, 32, "int_return must place r3=32 into tmpreg_i");
        assert_eq!(
            snapshots.len(),
            4,
            "probe must fire 4 times (live + int_add + int_sub + int_return)",
        );
        // probe[0] live/: pre-everything snapshot.
        assert_eq!(snapshots[0].0, "live/");
        assert_eq!(snapshots[0].1, [10, 32, 0, 0]);
        // probe[1] int_add: about to compute r2 = r0 + r1; r2 still 0.
        assert_eq!(snapshots[1].0, "int_add/ii>i");
        assert_eq!(snapshots[1].1, [10, 32, 0, 0]);
        // probe[2] int_sub: int_add's effect (r2=42) now visible; r3 still 0.
        assert_eq!(snapshots[2].0, "int_sub/ii>i");
        assert_eq!(snapshots[2].1, [10, 32, 42, 0]);
        // probe[3] int_return: int_sub's effect (r3=32) now visible.
        assert_eq!(snapshots[3].0, "int_return/i");
        assert_eq!(snapshots[3].1, [10, 32, 42, 32]);
    }
}
