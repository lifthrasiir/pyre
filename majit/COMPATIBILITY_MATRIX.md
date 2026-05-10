# majit Compatibility Matrix

Last updated: March 20, 2026

Tracks equivalence between majit (Rust, 81k LOC) and the in-tree RPython JIT sources.

## Legend

- `implemented`: intended to work and covered by focused tests
- `partial`: some behavior exists, but parity is incomplete

## Subsystem Status

| Subsystem | RPython Reference | Status | Notes |
|---|---|---|---|
| IR opcode surface | `resoperation.py` | `implemented` | 238 opcode variants, 1:1 match. 1442+ tests. |
| Optimizer pipeline | `optimizeopt/` | `implemented` | 8-pass default + unroll/short preamble/bridgeopt/earlyforce/virtualstate/vector. |
| Warm state / jitcell | `warmstate.py` | `implemented` | JitCell state machine, set_param(), LoopAging, QuasiImmut, confirm_enter_jit, get_location. |
| Trace recorder | `pyjitpl.py` | `implemented` | `#[jit_interp]` → JitCode + MIFrame. Multi-branch → IntEq + guard chains. |
| Resume / blackhole | `resume.py`, `blackhole.py` | `implemented` | EncodedResumeData roundtrip, 5 virtual kinds, BlackholeMemory trait, run_with_blackhole_fallback(). |
| Compilation | `compile.py` | `implemented` | compile.rs: guard metadata, exit layouts, unboxing. BridgeFailDescrProxy. |
| Cranelift backend | `backend/model.py` | `implemented` | All 238 opcodes lowered. CALL_ASSEMBLER, CALL_MAY_FORCE, native SIMD. |
| GC rewriter | `rewrite.py` | `implemented` | Nursery/write-barrier rewriting, pending zero flush. |
| GC runtime | `incminimark.py` | `implemented` | Nursery + oldgen + incremental marking + card marking. 93 tests. |
| Driver macros | `rlib/jit.py` | `implemented` | `#[jit_interp]`, `#[jit_driver]`, `#[jit_module]`, JitHookInterface. |
| Static analyzer | `codewriter/*` | `partial` | Graph-based pipeline with canonical CallDescriptor table + per-opcode dispatch via synthetic graphs (Phase C v2). `JitCode` now lives in its own `jitcode.rs` module (RPython filename parity) with `get_live_vars_info`/`follow_jump`/`num_regs_*`/`enumerate_vars`/`SwitchDictDescr`/`MissingLiveness` ported. Remaining: full descriptor/effectinfo object parity, runtime `BlackholeInterpreter::setup_insns` infrastructure. |

## File-Level Parity (optimizeopt/)

| majit | RPython | LOC | Parity | Gap |
|---|---|---|---|---|
| resoperation.rs | resoperation.py | 3,360 | 99% | — |
| intutils.rs | intutils.py | 2,669 | 95% | make_guards, getnullness |
| intbounds.rs | intbounds.py | 2,204 | 95% | — |
| pure.rs | pure.py | 1,502 | 95% | — |
| simplify.rs | simplify.py | 242 | 95% | — |
| unroll.rs | unroll.py | 3,461 | 95% | ~~inline_short_preamble~~ 추가됨 |
| optimizer.rs | optimizer.py | 1,863 | 95% | ~~guard sharing~~ ~~_maybe_replace_guard_value~~ 추가됨 |
| virtualstate.rs | virtualstate.py | 1,683 | 90% | — |
| shortpreamble.rs | shortpreamble.py | 1,980 | 90% | ~~CompoundOp~~ 추가됨 |
| info.rs | info.py | 1,214 | 85% | — |
| descr.rs | effectinfo.py | 1,500 | 85% | bitstring optimization |
| rewrite.rs | rewrite.py | 3,546 | 95% | ~~INT_PY_DIV~~ ~~oois_ooisnot~~ ~~replace_old_guard_with_guard_value~~ ~~CALL_N arraycopy~~ 추가됨. try_boolinvers 분리, serialize/deserialize 이름 parity. ~~optimize_float_abs~~ 메서드 추출 |
| virtualize.rs | virtualize.py | 3,986 | 80% | ~~COND_CALL, JIT_FORCE_VIRTUAL~~ 추가됨 |
| heap.rs | heap.py | 3,592 | 90% | ~~pendingfields, DICT_LOOKUP, serialization, variable-index, aliasing~~ 추가됨. ~~_optimize_CALL_DICT_LOOKUP~~ scaffolding (cached_dict_reads, corresponding_array_descrs, GUARD_NO_EXCEPTION REMOVED 연동, miss-시 emit_residual_call 분기, flush() dict cache clear, args_dict same_box 키, last_emitted_operation overwrite-on-emit, corresponding_array_descrs first-encounter registration). clean_caches 정제 남음. **codewriter extradescrs gap**: jtransform.py:2186 `_handle_dict_lookup_call` 에서 `cpu.fielddescrof(STRUCT, 'entries')` / `cpu.arraydescrof(STRUCT.entries.TO)` 를 만들 때 `op.args[1].concretetype.TO` (dict struct lltype) 가 필요한데 pyre `TypeResolutionState` 는 4-kind `ConcreteType` 만 보존 → extradescrs 는 `None` 으로 들어가고 production 에서 cache 가 형성되지 않음. `_optimize_call_dict_lookup` 은 None 시 false 반환하고 emit_residual_call → force_from_effectinfo 로 정상 fall-through. lltype 전파 후에야 캐시 활성화 가능 |
| vstring.rs | vstring.py | 1,298 | 80% | ~~STR_CONCAT, copy_str_content~~ 추가됨. ~~메서드 이름 parity~~ (strgetitem, getstrlen, int_add/int_sub, force_box, handle_str_equal_level1, opt_call_stroruni_* 분리). handle_str_equal_level2, generate_modified_call 남음 |
| bridgeopt.rs | bridgeopt.py | 829 | 95% | serialize/deserialize/tag_box/decode_box 모두 구현됨 |
| guard.rs | guard.py | 931 | 85% | ~~implies~~ ~~transitive_imply~~ ~~eliminate_array_bound_checks~~ 추가됨. IndexVar 연동 |
| vector.rs | vector.py | 917 | 55% | loop unrolling |
| dependency.rs | dependency.py | 233 | — | vector.rs에서 분리 |
| schedule.rs | schedule.py | 382 | — | vector.rs에서 분리 |
| version.rs | version.py | 85 | — | scaffold |
| optimize.rs | optimize.py | 30 | — | InvalidLoop, SpeculativeError |
| intdiv.rs | intdiv.py | 371 | — | — |
| earlyforce.rs | earlyforce.py | 157 | — | — |
| walkvirtual.rs | walkvirtual.py | 277 | — | — |

## File-Level Parity (metainterp/ tracing)

| majit | RPython | LOC | Parity | Gap |
|---|---|---|---|---|
| heapcache.rs | heapcache.py | 685 | 95% | — |
| history.rs | history.py | 764 | 95% | — |
| warmstate.rs | warmstate.py | 2,286 | 85% | ~~confirm_enter_jit~~ 추가됨 |
| opencoder.rs | opencoder.py | 1,139 | 65% | constant pooling |
| counter.rs | counter.py | 397 | 60% | JitCell chain |
| logger.rs | logger.py | 552 | — | different approach |
| memmgr.rs | memmgr.py | 90 | — | LoopAging |
| recorder.rs | (tracing infra) | 1,386 | — | majit-specific |

## File-Level Parity (metainterp/ meta)

| majit | RPython | LOC | Parity | Gap |
|---|---|---|---|---|
| blackhole.rs | blackhole.py | 2,841 | 80% | control flow ops (현 전략에서 불필요) |
| virtualizable.rs | virtualizable.py | 1,960 | 80% | — |
| quasiimmut.rs | quasiimmut.py | 297 | 80% | — |
| virtualize.rs + rewrite.rs | compile.py | 75% | — |
| virtualref.rs | virtualref.py | 259 | 65% | graph transformation |
| resume.rs | resume.py | 3,735 | 85% | ~~VStrPlain/6종~~ 추가됨. 남은: VirtualCache |
| pyjitpl.rs | pyjitpl.py | 7,606 | Different | Rust=compiled execution |
| compile.rs | compile.py | 1,163 | — | guard metadata, exit layouts, unboxing |
| executor.rs | executor.py | 731 | — | execute_one (blackhole에서 분리) |
| jitdriver.rs | jitdriver.py + warmspot.py | 2,527 | 70% | — |
| jitcode.rs | jitcode.py | (module) | Different | Rust=interpreter+builder. NOTE: `majit-translate::jitcode::JitCode` (codewriter side, RPython orthodox encoding) is now separate from `majit-metainterp::jitcode::JitCode` (runtime side, BC_* opcodes). RPython has a single shared type; Phase D will unify these. |
| jitexc.rs | jitexc.py | 58 | — | JIT exception enums |
| greenfield.rs | greenfield.py | 38 | — | scaffold |
| trace_ctx.rs | (majit-specific) | 2,714 | — | — |
| jit_state.rs | (majit-specific) | 2,003 | — | — |
| fail_descr.rs | (compile.py subset) | 220 | — | — |

## File-Level Parity (IR / codegen / GC)

| majit | RPython | LOC | Parity |
|---|---|---|---|
| majit-ir/resoperation.rs | resoperation.py | 3,360 | 99% |
| majit-ir/descr.rs | effectinfo.py | 1,500 | 85% |
| majit-ir/value.rs | history.py (Const, Type) | 312 | — |
| majit-codegen/lib.rs | backend/model.py | 1,167 | Different, bh_* 22개 |
| majit-gc/*.rs | gc/* | 6,192 | implemented |
| majit-runtime/lib.rs | rlib/jit.py | 416 | — |

## Backend Policy

1. Unsupported opcodes must return `BackendError::Unsupported`.
2. Void opcodes must not silently degrade to no-ops unless documented.
3. Guard/FINISH exits must preserve Int, Ref, Float values exactly.

## Detailed Gap List (per file)

### optimizeopt/

**intbounds.rs** (95%)
- Missing: diagnostic logging (log_inputargs/log_op/log_result), print_rewrite_rule_statistics
- Note: 이들은 디버깅 전용이며 최적화 정확성에 영향 없음

**rewrite.rs** (95%)
- Implemented: `_optimize_oois_ooisnot()`, `_optimize_nullness()`, postprocess inline
- Implemented: `replace_old_guard_with_guard_value()` (기존 가드를 GUARD_VALUE로 교체)
- Implemented: `optimize_CALL_N()` arraycopy/arraymove dispatch
- Implemented: `serialize_optrewrite()` / `deserialize_optrewrite()` (bridgeopt loop-invariant 직렬화)
- Implemented: `try_boolinvers()` 별도 메서드 추출 (rewrite.py:56-66 parity)
- Implemented: `find_rewritable_bool()` 3단계: inverse + reflex + inverse·reflex (rewrite.py:85-91)
- Implemented: `optimize_float_abs()` 별도 메서드 추출 (rewrite.py:155-161)
- Note: GUARD_SUBCLASS/IS_OBJECT/NONNULL/CLASS, COND_CALL, INT_PY_DIV는 propagate_forward match arm으로 구현됨

**heap.rs** (90%)
- Implemented: variable-index array caching (`ArrayCacheSubMap::cached_varindex_triples`)
- Implemented: aliasing analysis (`cannot_alias_via_content`, `_cannot_alias_via_classes_or_lengths` inline)
- Implemented: `export_cached_fields()` / `import_cached_fields()` (bridge 지식 직렬화)
- Implemented: `force_lazy_sets_for_guard()` (가드 전용 선택적 lazy set forcing → pendingfields → rd_pendingfields)
- Implemented (heap.rs side): `_optimize_CALL_DICT_LOOKUP()` — FLAG_LOOKUP/FLAG_STORE 캐싱, cached_dict_reads, corresponding_array_descrs first-encounter registration (heap.py:507-511 KeyError arm), GUARD_NO_EXCEPTION REMOVED 연동, miss-시 emit_residual_call 분기 (heap.py:472-475 self.emit(op) → force_from_effectinfo parity), flush()/clean_caches dict cache clear, inner cache key는 `DictArgKey` 로 args_dict same_box semantics (util.py:100 / history.py:204 ConstInt value-equality), `last_emitted_operation` 은 emit-time overwrite (optimizer.py:84)
- **Blocked (codewriter side, jtransform.py:2186 parity gap)**: `_handle_dict_lookup_call` 은 `op.args[1].concretetype.TO` (dict struct lltype) 으로 `cpu.fielddescrof(STRUCT, 'entries')` + `cpu.arraydescrof(STRUCT.entries.TO)` 를 만들고 extradescrs 로 전달. pyre `TypeResolutionState` 는 4-kind `ConcreteType` 만 carry 하므로 dict struct 식별이 jit_codewriter 단계에서 불가 → 현재는 `None` 을 전달하고, production trace 에서 dict.lookup cache 는 형성되지 않음 (`_optimize_call_dict_lookup` 이 false 반환 → generic residual call 경로). main `64aa6544c96` 시점과 동일 동작. lltype propagation (call args 의 owner_root 보존) 이 선결 과제
- Missing: `clean_caches()` 정제 (순수 필드만 보존하는 선택적 무효화 — 부분 구현)
- Note: pendingfields, ~~DICT_LOOKUP~~, postponed ops, GUARD_NO_EXCEPTION, arraycopy 무효화 모두 구현됨

**virtualize.rs** (80%)
- Missing: `optimize_INT_ADD()` raw slice optimization
- Missing: `optimize_GETARRAYITEM_RAW_I/F`, `optimize_SETARRAYITEM_RAW`
- Missing: `optimize_FINISH()` + `_last_guard_not_forced_2` tracking
- Note: COND_CALL, CALL_MAY_FORCE, JIT_FORCE_VIRTUAL, RAW_MALLOC, GUARD_NO_EXCEPTION은 구현됨

**pure.rs** (95%)
- Missing: `optimize_call_pure_old()` (이전 결과 재사용 로직)
- Missing: `_can_reuse_oldop()` (기존 op 재사용 가능성 판단)
- Note: postponed_op, constant_fold, COND_CALL_VALUE, lookup1/2, RECORD_KNOWN_RESULT은 구현됨

**guard.rs** (85%)
- Implemented: `Guard.implies()` with IndexVar.compare() (guard.py:51-71)
- Implemented: `Guard.transitive_imply()` / `transitive_cmpop()` (guard.py:73-104)
- Implemented: `GuardEliminator.eliminate_array_bound_checks()` (guard.py:279-303)
- Implemented: `Guard.inhert_attributes()`, `Guard.set_to_none()`, `Guard.of()` with index_vars
- Note: Rust guard.rs는 중복 제거 + truthy_values subsumption + descriptor fusion + IndexVar 기반 함의 추론 있음

**vstring.rs** (70%)
- Missing: `handle_str_equal_level1()` / `handle_str_equal_level2()` (다단계 문자열 비교 최적화)
- Missing: `setup_slice()`, `shrink()` (VStringPlainInfo 메서드)
- Missing: `generate_modified_call()` (변형 호출 생성)
- Missing: unicode 전용 ops (NEWUNICODE, UNICODELEN, UNICODEGETITEM, COPYUNICODECONTENT)
- Note: STR_CONCAT, STR_SLICE, STR_EQUAL, STR_CMP, copy_str_content, initialize_forced_string은 구현됨

**optimizer.rs** (90%)
- Missing: `emit_guard_operation()` (가드 공유/resume data 복사)
- Missing: `_copy_resume_data_from()` (가드 resume data 상속)
- Missing: `store_final_boxes_in_guard()` (ResumeDataVirtualAdder 통합)
- Missing: `_maybe_replace_guard_value()` (가드 값 교체 최적화)
- Missing: `constant_fold()` / `protect_speculative_operation()` (투기적 연산 보호)

**shortpreamble.rs** (90%)
- Missing: `ShortBoxes.create_short_boxes()` 전체 탐색 (potential ops → short boxes 변환)
- Missing: `ShortBoxes._pick_op_index()` (op 인덱스 선택 휴리스틱)
- Note: PreambleOp, HeapOp, PureOp, LoopInvariantOp, CompoundOp, ExtendedShortPreambleBuilder은 구현됨

**virtualstate.rs** (90%)
- Note: GenerateGuardState, lenbound, make_inputargs, force_boxes, VirtualStateConstructor, compute_renum 모두 구현됨

**info.rs** (85%)
- Missing: per-type `force_box()` (InstancePtrInfo, ArrayPtrInfo 등 타입별 forcing)
- Missing: `getstrlen()` / `getstrhash()` 실질 값 계산
- Missing: `copy_fields_to_const()` (상수 객체 필드 복제)
- Note: visitor_walk_recursive, force_at_the_end_of_preamble, make_guards은 구현됨

**vector.rs** (55%)
- Missing: `unroll_loop_iterations()` (벡터화 전 루프 언롤링)
- Missing: index guard analysis (가드 → 루프 헤더 이동)
- Note: DependencyGraph, PackSet, GenericCostModel, extend/combine_packset, AccumulationPack은 구현됨

### metainterp/ tracing

**warmstate.rs** (85%)
- Missing: `make_entry_point()`, `make_unwrap_greenkey()`, `make_jitcell_subclass()` (NOT_RPYTHON, 동적 코드 생성)
- Missing: `make_jitdriver_callbacks()` (can_inline_callable 등 콜백 생성)
- Missing: `get_location_str()` (디버깅 위치 정보)
- Note: 이들은 RPython 번역기 전용(NOT_RPYTHON)이며 Rust에서는 컴파일 타임에 해결됨

**heapcache.rs** (95%)
- Missing: `call_loopinvariant_known_result()` / `call_loopinvariant_now_known()` (loop invariant 결과 캐시)
- Missing: version/generation 기반 reset (Python의 HF_* 플래그 시스템)
- Note: array_cache, quasi_immut, arraylen, escape propagation은 구현됨

**opencoder.rs** (65%)
- Missing: `Trace._cached_const_int()` / `_cached_const_ptr()` (상수 값 풀링)
- Missing: `Trace._encode()` 전체 box 인코딩 시스템
- Missing: `Trace._list_of_boxes_virtualizable()` (virtualizable box 리스트)
- Missing: `Trace.get_live_ranges()` / `get_dead_ranges()` (live/dead 범위 분석)
- Note: Snapshot/TopSnapshot/SnapshotStorage/TraceIterator/CutTrace/tag/untag은 구현됨

**counter.rs** (60%)
- Missing: JitCell chain 관리 (`lookup_chain`, `cleanup_chain`, `install_new_cell`)
- Missing: `change_current_fraction()` (특정 카운터 값 직접 설정)
- Note: tick, reset, compute_threshold, fetch_next_hash, decay_all_counters은 구현됨

### metainterp/ meta

**resume.rs** (85%)
- Missing: `VirtualCache` (반복 물질화 캐시)
- Missing: `VectorInfo` / `UnpackAtExitInfo` / `AccumInfo` (벡터 최적화 전용 resume info)
- Note: ResumeDataLoopMemo, ResumeDataVirtualAdder, ResumeDataReader, EncodedResumeData, VStr/VUni 6종 모두 구현됨

**blackhole.rs + executor.rs** (80%)
- Missing: `handle_exception_in_frame()` (프레임 내 예외 처리)
- Missing: `get_current_position_info()` (위치 추적)
- Missing: `copy_constants()` (상수 복사)
- Note: execute_one에 모든 bhimpl_* 연산 구현됨. 아키텍처 차이: Python=바이트코드 디스패치, Rust=직접 VM

**virtualref.rs** (65%)
- Missing: `replace_force_virtual_with_call()` (그래프 변환 — translator 전용)
- Missing: `get_force_virtual_fnptr()` / `get_is_virtual_fnptr()` (함수 포인터 생성)
- Note: force_virtual, is_virtual_ref, virtual_ref_during_tracing, tracing callbacks은 구현됨

**compile.rs** (—)
- RPython compile.py의 ResumeGuardDescr 계층 (15+ subclass)은 Rust에서 fail_descr.rs + codegen trait으로 대체
- `send_loop_to_backend()` / `send_bridge_to_backend()`은 pyjitpl.rs MetaInterp 메서드로 구현
- `compile_loop()` / `compile_retrace()`는 pyjitpl.rs의 close_and_compile/finish_and_compile로 구현
- Note: 구조적 차이. RPython 함수들이 다른 형태로 존재함

## Summary

**실질적으로 구현이 필요한 갭 (우선순위):**
1. **heap.rs**: variable-index caching, aliasing analysis, serialize/deserialize
2. **guard.rs**: Guard.implies() 수치 함의 추론, transitive loop bounds
3. **optimizer.rs**: emit_guard_operation, constant_fold, speculative ops
4. **vstring.rs**: multi-level STR_EQUAL, unicode ops
5. **opencoder.rs**: constant pooling, live/dead range analysis
6. **vector.rs**: loop unrolling

**아키텍처 차이로 불필요한 갭:**
- warmstate.py의 NOT_RPYTHON 메서드들 (동적 코드 생성 → 컴파일 타임)
- compile.py의 ResumeGuardDescr 계층 (trait 기반으로 대체)
- blackhole.py의 바이트코드 디스패치 (직접 VM으로 대체)
- resume.py의 class 계층 (enum 기반으로 대체)

## Known Bugs

**Compiled loop infinite loop on nested-loop functions (nbody advance)**
- Symptom: func-entry JIT compiles advance() with nested while loops, compiled code runs forever
- Trace has `OpRef(u32::MAX)` (NONE) in guard fail_args → suspect unresolved forwarding
- Raw pointer constants in Jump args → GC stale risk
- Status: Compilation succeeds, compiled code execution hangs (infinite loop, not crash)
- Repro: `while k<8: advance(bodies, 0.01)` with 5-body nbody

**Compiled loop segfault on 5-body nbody with float ops**
- Symptom: compiled loop runs once then segfaults on second iteration
- Suspect: GC nursery pointer invalidation across loop back-edge
- Status: Under investigation
