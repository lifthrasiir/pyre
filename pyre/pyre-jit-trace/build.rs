#[path = "src/call_spec.rs"]
mod call_spec;
#[path = "src/virtualizable_spec.rs"]
mod virtualizable_spec;

use walkdir::WalkDir;

/// Build script for pyre-jit: runs majit-translate on the active pyre
/// interpreter to auto-generate tracing code. This is the Rust
/// equivalent of RPython's translation pipeline.
///
/// Analyzes all source files from:
/// - pyre-object (Python object types: W_IntObject, W_FloatObject, etc.)
/// - pyre-interpreter (object space, bytecode dispatch, eval loop)
fn main() {
    // Run on a worker thread with a large stack: on Windows the main
    // thread's default stack is 1 MiB, which `syn`'s recursive parsing
    // of the ~90 collected source files overflows
    // (STATUS_STACK_OVERFLOW 0xc00000fd).
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(real_main)
        .expect("spawn build-script worker")
        .join()
        .expect("build-script worker panicked");
}

fn real_main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let pyre_base = format!("{manifest_dir}/..");
    let repo_root = format!("{manifest_dir}/../..");

    // Collect ALL source files from the active interpreter crates.
    let source_dirs = [
        format!("{pyre_base}/pyre-object/src"),
        format!("{pyre_base}/pyre-interpreter/src"),
    ];

    let mut sources = Vec::new();
    let mut source_paths = Vec::new();

    for dir in &source_dirs {
        collect_rs_files(dir, &mut sources, &mut source_paths);
    }

    // Phase G follow-up — include the portal canonical source so
    // `majit-translate` finds `eval_loop_jit` in `call_control
    // .function_graphs()` and the default-portal logic at
    // `majit-translate/src/lib.rs:621-644` flips from
    // `execute_opcode_step` (single-opcode dispatch helper) to
    // `eval_loop_jit` (the real portal with the `match instruction`
    // dispatch loop).  Pre-fix `portal_jitcode()` returned the
    // canonical for `execute_opcode_step` which a portal-bridge
    // install (`canonical_bridge::install_portal_for`) cloned, leaving
    // BH `setposition(pyjitcode.jitcode, jitcode_pc)` unable to
    // resume into a user PC because `execute_opcode_step` lacks the
    // dispatch loop the BH needs to walk forward.
    //
    // Single-file inclusion (not the whole `pyre-jit/src` tree)
    // because `eval_loop_jit` is the only function in pyre-jit that
    // belongs in the portal closure; the rest of pyre-jit
    // (codewriter, assembler, regalloc, etc.) is JIT infrastructure
    // that must NOT be analyzed as user code, and the orchestration
    // would inflate analysis time and risk pulling unrelated
    // helpers into `find_all_graphs(portal)` BFS.
    let eval_path = format!("{pyre_base}/pyre-jit/src/eval.rs");
    collect_single_file(&eval_path, &mut sources, &mut source_paths);

    // `materialize_virtual_from_rd` (`pyre-jit/src/eval.rs`) destructures
    // `majit_ir::RdVirtualInfo` enum variants whose named fields carry
    // primitive concretetypes (`kind: u8`, `descr_index: u32`, ...).
    // Without the enum declaration in the analyser source set the variant
    // arms surface as untyped, and the rtyper's exitswitch emission for
    // `match descr.kind { 0 => ..., 1 => ..., 2 => ..., }` falls back to
    // 'r' (Ref), which `flatten.rs:385` rejects as
    // `switch exitswitch must be int`.  Single-file inclusion mirrors the
    // `pyre-jit/src/eval.rs` carve-out above: only the resoperation enum
    // declarations belong in the analyser closure, not the rest of
    // majit-ir which would inflate analysis time and pull JIT
    // infrastructure into the user-code BFS.
    let resoperation_path = format!("{repo_root}/majit/majit-ir/src/resoperation.rs");
    collect_single_file(&resoperation_path, &mut sources, &mut source_paths);

    eprintln!(
        "[pyre-jit-trace build.rs] reading {} source files from {} dirs (+ pyre-jit/src/eval.rs): {:?}",
        sources.len(),
        source_dirs.len(),
        source_paths,
    );

    // Run analysis on ALL files with PyFrame virtualizable metadata.
    //
    // This keeps the generic analyzer closer to the proc-macro/runtime path:
    // graph rewrite can recognize `next_instr`, `valuestackdepth`, and
    // `locals_cells_stack_w[*]` as virtualizable accesses before legacy
    // TracePattern classification runs.
    let source_refs: Vec<&str> = sources.iter().map(|s| s.as_str()).collect();
    let analyze_config = majit_translate::AnalyzeConfig {
        pipeline: majit_translate::PipelineConfig {
            transform: majit_translate::GraphTransformConfig {
                vable_fields: virtualizable_spec::PYFRAME_VABLE_FIELDS
                    .iter()
                    .map(|(name, idx)| {
                        majit_translate::VirtualizableFieldDescriptor::new(
                            *name,
                            Some(virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT.to_string()),
                            *idx,
                        )
                    })
                    .collect(),
                vable_arrays: virtualizable_spec::PYFRAME_VABLE_ARRAYS
                    .iter()
                    .map(|(name, idx)| {
                        // virtualizable.py:58 — VirtualizableInfo.array_descrs[i] =
                        // cpu.arraydescrof(getattr(VTYPE, name).TO). Python frame
                        // locals are PyObjectRef pointers: itemsize=8, is_signed=false.
                        majit_translate::VirtualizableFieldDescriptor::new_with_arraydescr(
                            *name,
                            Some(virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT.to_string()),
                            *idx,
                            8,     // itemsize: PyObjectRef is a pointer
                            false, // is_signed: pointers are unsigned
                        )
                    })
                    .collect(),
                call_effects: build_call_effect_overrides(),
                ..Default::default()
            },
            portal: None,
        },
    };
    // warmspot.py:516 `vinfos[VTYPEPTR] = VirtualizableInfo(self, VTYPEPTR)` —
    // pyre's runtime `VirtualizableInfo` constructor lives in the
    // `majit-metainterp` crate (`__build_virtualizable_info`) and runs
    // at `JitDriver::new` (jitdriver.rs:285) where the field offsets
    // resolved by `mem::offset_of!` are available.  build.rs cannot
    // import that crate (no metainterp build-dep, and the offsets are
    // a runtime fact), so the codewriter-side factory returns `None`
    // here; the codewriter slot stays empty until the runtime metainterp
    // setter overrides it.  PRE-EXISTING-ADAPTATION documented at
    // `CallControl::make_virtualizable_infos`.
    let vinfo_factory: &majit_translate::VirtualizableInfoFactory<'_> = &|_jd_idx, _vtype| None;
    let fnaddr_bindings = pyre_interpreter::jit_trace_fnaddrs();
    let pipeline = majit_translate::analyze_multiple_pipeline_with_vinfo_and_fnaddr_bindings(
        &source_refs,
        &analyze_config,
        None,
        vinfo_factory,
        &fnaddr_bindings,
    );

    // Generate tracing code from the canonical graph-first analysis result.
    let code = majit_translate::generate_trace_code_from_pipeline(&pipeline);

    let out_dir = std::env::var("OUT_DIR").unwrap();
    std::fs::write(format!("{out_dir}/jit_trace_gen.rs"), &code).unwrap();

    // Trait impls live in a separate file because lib.rs `include!`s
    // jit_trace_gen.rs twice (once at crate root and once inside
    // `pub mod generated`). Trait impls would conflict (E0119) under double
    // inclusion, so they live in their own file included only once.
    //
    // Assembled from THREE pieces (Phase B of the eval-loop automation plan):
    //   1. `opcode_handler_impls_pre.template.rs` — header + variant
    //      `SharedOpcodeHandler` impl (transcription).
    //   2. `majit_translate::handler_spec::emit_simple_trait_impls()` —
    //      the 5 simple traits (Constant/Stack/Truth/Iter/Local), emitted
    //      from the spec table in majit-translate/src/handler_spec.rs.
    //   3. `opcode_handler_impls_post.template.rs` — remaining variant
    //      `ControlFlow/Branch/Namespace/Arithmetic` impls (transcription).
    //
    // `tests/trait_impls_snapshot.rs` guards against drift by comparing the
    // assembled output against a checked-in snapshot.
    let pre_path = format!("{manifest_dir}/src/opcode_handler_impls_pre.template.rs");
    let post_path = format!("{manifest_dir}/src/opcode_handler_impls_post.template.rs");
    let pre = std::fs::read_to_string(&pre_path).unwrap_or_else(|e| {
        panic!("[pyre-jit-trace build.rs] cannot read {pre_path}: {e}");
    });
    let post = std::fs::read_to_string(&post_path).unwrap_or_else(|e| {
        panic!("[pyre-jit-trace build.rs] cannot read {post_path}: {e}");
    });
    let simple = majit_translate::handler_spec::emit_simple_trait_impls();
    // pre ends with `}\n\n` (Shared close + blank), simple ends with `}\n`,
    // post starts with `\n` (blank). Concat = `...}\n\nimpl Constant...}\n\nimpl ControlFlow...`
    // which matches the original single-template structure byte-for-byte.
    let trait_impls_code = format!("{pre}{simple}{post}");
    std::fs::write(
        format!("{out_dir}/jit_trace_trait_impls.rs"),
        &trait_impls_code,
    )
    .unwrap();
    println!("cargo::rerun-if-changed={pre_path}");
    println!("cargo::rerun-if-changed={post_path}");

    // JSON metadata for debugging
    let json = serde_json::to_string_pretty(&pipeline).unwrap();
    std::fs::write(format!("{out_dir}/jit_metadata.json"), &json).unwrap();

    // Phase D-1 Step 1 (eval-loop automation plan): persist
    // `pipeline.jitcodes` (RPython `all_jitcodes` from codewriter.py:89) as a
    // single bincode artifact. Runtime deserializes this once into the shared
    // MetaInterpStaticData jitcodes store — same single-store model as
    // RPython `warmspot.py:281-282` `self.metainterp_sd.jitcodes =
    // codewriter.make_jitcodes()`. No side-table serialization: arm→jitcode
    // linking goes through the existing `PipelineOpcodeArm.entry_jitcode_index`
    // field which is already in `jit_metadata.json`.
    let jitcodes_bin = bincode::serialize(&pipeline.jitcodes).unwrap();
    std::fs::write(format!("{out_dir}/opcode_jitcodes.bin"), &jitcodes_bin).unwrap();

    // Phase D-1 Step 4: persist `pipeline.opcode_dispatch` (the arm table)
    // alongside the jitcodes so the runtime can map opcode → arm_id →
    // entry_jitcode_index → JitCode. This is the same `PipelineOpcodeArm`
    // shape already present in `jit_metadata.json`; bincode just avoids the
    // cost of JSON parse at startup.
    let dispatch_bin = bincode::serialize(&pipeline.opcode_dispatch).unwrap();
    std::fs::write(format!("{out_dir}/opcode_dispatch.bin"), &dispatch_bin).unwrap();

    // Phase D-2 prerequisite: persist `pipeline.insns` — the opname → u8
    // table `Assembler.write_insn` grew during assembly. JitCode bytecode
    // bytes are opaque without it (the mapping is assembler-local, not
    // part of the serialized JitCode), so any runtime consumer that
    // wants to decode `JitCode.code` (shadow dispatch, IR diffing)
    // needs this side by side with opcode_jitcodes.bin. RPython
    // equivalent: the table handed to `BlackholeInterpBuilder::setup_insns`
    // at metainterp startup (pyjitpl.py:2227-2243).
    //
    // Serialize through a `BTreeMap` view so the byte output is stable
    // across processes.  `pipeline.insns` is a `HashMap<String, u8>`
    // and `bincode` walks it in iteration order; Rust's `HashMap` uses
    // SipHash with a per-process random seed, which would otherwise
    // make `opcode_insns.bin` byte-differ on each build.  RPython's
    // equivalent (`assembler.py:220` `self.insns = {}`) is a Python
    // dict with stable insertion-order iteration, so run-to-run
    // stability is the expected behaviour.  Serde's map format is
    // HashMap / BTreeMap compatible, so the runtime deserializer that
    // receives a `HashMap<String, u8>` back is unaffected.
    let insns_sorted: std::collections::BTreeMap<&String, &u8> = pipeline.insns.iter().collect();
    let insns_bin = bincode::serialize(&insns_sorted).unwrap();
    std::fs::write(format!("{out_dir}/opcode_insns.bin"), &insns_bin).unwrap();

    // RPython `blackhole.py:59 self.setup_descrs(asm.descrs)` + `:102-103
    // def setup_descrs(self, descrs): self.descrs = descrs`. Persists the
    // build-time assembler's shared descr pool so that 'd'/'j' argcodes
    // in `JitCode.code` resolve at runtime via
    // `BlackholeInterpBuilder::setup_descrs(...)` — the single-store
    // model (same list consumed by every `BlackholeInterpreter` produced
    // by `acquire_interp`).
    let descrs_bin = bincode::serialize(&pipeline.descrs).unwrap();
    std::fs::write(format!("{out_dir}/opcode_descrs.bin"), &descrs_bin).unwrap();

    // Report
    let arms_with_jitcode = pipeline
        .opcode_dispatch
        .iter()
        .filter(|arm| arm.entry_jitcode_index.is_some())
        .count();
    eprintln!(
        "[pyre-jit-trace build.rs] canonical analysis: {} opcode arms ({} flattened, {} indexed), {} functions, {} blocks, {} flat ops, {} all_jitcodes ({} bytes bincode), generated {} bytes",
        pipeline.opcode_dispatch.len(),
        pipeline
            .opcode_dispatch
            .iter()
            .filter(|arm| arm.flattened.is_some())
            .count(),
        arms_with_jitcode,
        pipeline.functions.len(),
        pipeline.total_blocks,
        pipeline.total_ops,
        pipeline.jitcodes.len(),
        jitcodes_bin.len(),
        code.len(),
    );

    // Rerun if any source file changes
    for path in &source_paths {
        println!("cargo::rerun-if-changed={path}");
    }
    emit_rerun_if_changed_recursive(&format!("{repo_root}/majit/majit-translate/src"));
    println!(
        "cargo::rerun-if-changed={}",
        format!("{repo_root}/majit/majit-translate/build.rs")
    );
    println!("cargo::rerun-if-changed=src/virtualizable_spec.rs");
    println!("cargo::rerun-if-changed=src/call_spec.rs");
}

fn build_call_effect_overrides() -> Vec<majit_translate::CallEffectOverride> {
    call_spec::PYFRAME_CALL_EFFECTS
        .iter()
        .map(|spec| {
            let target = match spec.target {
                call_spec::CallTargetSpec::Method {
                    name,
                    receiver_root,
                } => majit_translate::CallTarget::method(name, Some(receiver_root.to_string())),
                call_spec::CallTargetSpec::FunctionPath(segments) => {
                    majit_translate::CallTarget::function_path(segments.iter().copied())
                }
            };
            let effect = match spec.effect {
                call_spec::CallEffectKind::Elidable => majit_translate::CallEffectKind::Elidable,
                call_spec::CallEffectKind::Residual => majit_translate::CallEffectKind::Residual,
            };
            majit_translate::CallEffectOverride::new(target, effect)
        })
        .collect()
}

/// Collect a single `.rs` file by absolute path, mirroring
/// `collect_rs_files`'s read-into-vecs convention.  Used by Phase G
/// follow-up to thread `pyre-jit/src/eval.rs` (the portal canonical)
/// into the analysis without including the rest of pyre-jit's JIT
/// infrastructure (codewriter, assembler, regalloc, ...).
fn collect_single_file(path: &str, sources: &mut Vec<String>, paths: &mut Vec<String>) {
    match std::fs::read_to_string(path) {
        Ok(content) => {
            paths.push(path.to_string());
            sources.push(content);
        }
        Err(e) => {
            eprintln!("[pyre-jit-trace build.rs] warning: cannot read {path}: {e}");
        }
    }
}

/// Collect all `.rs` files from a directory tree.
fn collect_rs_files(dir: &str, sources: &mut Vec<String>, paths: &mut Vec<String>) {
    for entry in WalkDir::new(dir) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() || entry.path().extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        let path = entry.path();
        let path_str = path.to_string_lossy().to_string();
        match std::fs::read_to_string(path) {
            Ok(content) => {
                paths.push(path_str);
                sources.push(content);
            }
            Err(e) => {
                eprintln!("[pyre-jit-trace build.rs] warning: cannot read {path_str}: {e}");
            }
        }
    }
}

fn emit_rerun_if_changed_recursive(dir: &str) {
    for entry in WalkDir::new(dir) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() || entry.path().extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        println!("cargo::rerun-if-changed={}", entry.path().display());
    }
}
