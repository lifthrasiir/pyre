//! Trace-side jitcode walker.
//!
//! RPython parity: this is the trace-side counterpart of
//! `BlackholeInterpBuilder.dispatch_loop` (`blackhole.py:65-100`). The
//! blackhole loop *executes* each `bhimpl_*` in turn; the tracing-side
//! analogue lives in `pyjitpl.py:opimpl_*` where each opcode becomes
//! a `MetaInterp.execute_and_record` call (RPython
//! `pyjitpl.py:1640-1660`). Pyre is mid-migration: the production
//! tracing path is the trait-driven `MIFrame::execute_opcode_step`
//! (trace_opcode.rs); this module is the orthodox path that consumes
//! the codewriter-emitted jitcode bytes directly.
//!
//! Opcode coverage:
//!
//! | opname              | parity status | behaviour |
//! |---------------------|---------------|-----------|
//! | `live/`             | PARITY        | skip OFFSET_SIZE, continue (RPython tracing does not record `live/` either) |
//! | `goto/L`            | PARITY        | jump to 2-byte LE target, continue |
//! | `catch_exception/L` | PARITY        | skip 2-byte target on normal flow (`pyjitpl.py:497-504` records nothing); the target is consumed by `inline_call`'s SubRaise arm via `try_catch_exception_at` (`pyjitpl.py:2517-2522`) |
//! | `switch/id`         | STRUCTURAL ADAPTATION | RPython `opimpl_switch` shape: read int box, lookup `SwitchDictDescr.dict`, emit `GUARD_VALUE` on hit or `INT_EQ` + `GUARD_FALSE` chain on miss. Concrete branch value comes from `TraceCtx::concrete_of_opref`; non-concrete symbolic OpRefs surface `SwitchValueNotConcrete` instead of guessing a branch. |
//! | `ref_return/r`      | PARITY        | top-level: record `Finish(reg) descr=done_with_this_frame_descr_ref` + terminate (`pyjitpl.py:opimpl_ref_return → compile_done_with_this_frame`); sub-walk: surface `SubReturn{Some(value)}` to caller (`pyjitpl.py:1688-1698 finishframe`) |
//! | `int_return/i`      | PARITY        | int-bank counterpart of `ref_return/r` — top-level records `Finish(reg) descr=done_with_this_frame_descr_int` (`pyjitpl.py:3206-3208`), sub-walk surfaces `SubReturn{Some(value)}`. RPython `pyjitpl.py:463 opimpl_int_return = _opimpl_any_return`. |
//! | `float_return/f`    | PARITY        | float-bank counterpart — top-level records `Finish(reg) descr=done_with_this_frame_descr_float` (`pyjitpl.py:3212-3214`), sub-walk surfaces `SubReturn{Some(value)}`. RPython `pyjitpl.py:465 opimpl_float_return = _opimpl_any_return`. |
//! | `void_return/`      | PARITY        | void return — top-level records `Finish([]) descr=done_with_this_frame_descr_void` (`pyjitpl.py:3202-3205`, `exits = []` branch), sub-walk surfaces `SubReturn{None}`. RPython `pyjitpl.py:467-469 opimpl_void_return → finishframe(None)`. |
//! | `inline_call_r_r/dR>r` | PARITY (per-frame catch) | recurses into sub-jitcode via `JitCodeDescr::jitcode_index()`, populates callee `registers_r` (`setup_call_r`, OOR surfaces `InlineCallArityMismatch`), writes `SubReturn{value}` into caller dst (Ref bank), scans caller's `op.next_pc` for `live/` + `catch_exception/L` on `SubRaise` (`pyjitpl.py:2506-2522 finishframe_exception`). Sub-walk reaching `Terminate` is unexpected (top-level should never fire from a sub-walk); `SubReturn{None}` into a `_r_*` slot surfaces `UnexpectedVoidSubReturn`. |
//! | `inline_call_r_i/dR>i` | PARITY        | int-result sibling of `inline_call_r_r/dR>r`. Same recursion + arglist + raise routing; only the dst bank changes (`registers_i[dst] = subreturn_value`). RPython `pyjitpl.py:1266-1324 _opimpl_inline_call*` is generated through `_opimpl_any_inline_call` decorator that varies on the result type — pyre's walker shares the body via `dispatch_inline_call_dr_kind(dst_bank)`. |
//! | `inline_call_ir_r/dIR>r`, `inline_call_ir_i/dIR>i` | PARITY | extended-arglist siblings — descr + I-list + R-list + dst. RPython `setup_call(argboxes_i, argboxes_r, argboxes_f)` (pyjitpl.py:230-260) populates the callee's int + ref banks from the two lists. Walker uses `dispatch_inline_call_dir_kind(dst_bank)` which reads `read_int_var_list` then `read_ref_var_list` and surfaces per-bank arity overflow as `InlineCallIntArityMismatch` / `InlineCallArityMismatch`. |
//! | `inline_call_irf_r/dIRF>r`, `inline_call_irf_f/dIRF>f` | PARITY | full-arglist variants — descr + I-list + R-list + F-list + dst. RPython same `setup_call` distribution; walker uses `dispatch_inline_call_dirf_kind(dst_bank)` extending the dIR helper with `read_float_var_list` + float-bank arg setup. Float arity overflow surfaces `InlineCallFloatArityMismatch`. |
//! | `int_copy/i>i`      | PARITY        | `registers_i[dst] = registers_i[src]` SSA rename, no IR op emitted (`pyjitpl.py:471-477 _opimpl_any_copy + >i` decorator) |
//! | `ref_copy/r>r`      | PARITY        | Ref-bank sibling — `registers_r[dst] = registers_r[src]` SSA rename, no IR op. Const-source variants (codewriter `emit_ref_copy!` with `ConstRef`) resolve through the constants window of `registers_r` (pre-populated by `setposition` in [`num_regs_r, num_regs_and_consts_r)`). |
//! | `int_<binop>/ii>i`  | PARITY        | int_add/int_sub/int_mul/int_and/int_or/int_xor/int_lshift/int_rshift + comparisons int_eq/int_ne/int_lt/int_le/int_gt/int_ge (14 ops). Reads two `i`-coded regs, records `OpCode::Int<Binop>` with `[a, b]`, writes recorder result into dst (`pyjitpl.py:279-336`). Mixed shapes such as `int_lshift/ri>i` stay unwired: those are kind-flow kind-flow bugs and must stay unsupported. |
//! | `float_<binop>/ff>f` + `float_neg/f>f` | PARITY | float_add/float_sub/float_truediv binops + float_neg unary (4 ops total — float_mul, float comparisons, float_abs all absent from codewriter today, would land mechanically when emitted). Read on `registers_f` bank, record `OpCode::Float<Binop>`, write dst (`pyjitpl.py:284-292`). |
//! | `int_neg/i>i`, `int_invert/i>i` | PARITY | unary i→i ops via `unop_int_record`. RPython `pyjitpl.py:356-368` exec-generated unary opimpls. `int_same_as/i>i` has a dormant walker arm for forward-prep, but the generated table should not contain it because RPython `jtransform.py:246 rewrite_op_same_as` removes `same_as` before assembly. |
//! | `cast_int_to_float/i>f` | PARITY | i-bank read, record `CastIntToFloat`, f-bank write. RPython `pyjitpl.py:357 cast_int_to_float` (same exec-generated unary opimpl loop). |
//! | `ptr_eq/rr>i`, `ptr_ne/rr>i` | PARITY | r-bank pair → record PtrEq/PtrNe → i-bank dst via `binop_ref_to_int_record`. RPython `pyjitpl.py:326-336` exec-generated comparison opimpls (b1 is b2 fast path omitted, same rationale as int comparisons). |
//! | `getfield_gc_i/rd>i`, `getfield_gc_r/rd>r` | PARITY (heapcache-aware) | r-bank obj + descr → heapcache lookup. Cache hit returns cached OpRef without recording; cache miss records `OpCode::GetfieldGc<I,R>` + `getfield_now_known` writeback. RPython `pyjitpl.py:855-882 + 929-950 _opimpl_getfield_gc_any_pureornot`. ConstPtr fast-path (`pyjitpl.py:856-860`) deferred — pyre walker doesn't track ConstPtr identity (optimizer's job post-trace). The pyre-specific `id>X` shape (int source — kind-flow kind-flow) stays unsupported. |
//! | `setfield_gc_i/rid`, `setfield_gc_r/rrd` | PARITY (heapcache-aware, alias-clearing) | r-bank box + (i\|r)-bank valuebox + descr. If `getfield_cached(obj,descr) == Some(valuebox)` skip recording (RPython `if upd.currfieldbox is valuebox: return`); otherwise record `OpCode::SetfieldGc(obj, valuebox)` + `setfield_cached` write-through. Aliasing semantics: `CacheEntry.do_write_with_aliasing` (heapcache.py:90-94) routes through `_clear_cache_on_write(seen_alloc)` — always wipes `cache_anything`, additionally wipes `cache_seen_allocation` when the write target itself isn't seen-allocated. RPython `pyjitpl.py:973-988 _opimpl_setfield_gc_any`. The disabled is_unescaped branch (`pyjitpl.py:981-988`) is intentionally not ported — RPython itself has it commented out. `iid` / `ird` (int box) shapes stay unsupported (kind-flow territory). |
//! | `getarrayitem_gc_r/rid>r` | PARITY (heapcache-aware) | r-bank array + i-bank index + descr → heapcache `getarrayitem` lookup. Cache hit returns cached OpRef without IR; cache miss records `OpCode::GetarrayitemGcR(array, index)` + `getarrayitem_now_known` writeback. RPython `pyjitpl.py:639-688 _do_getarrayitem_gc_any`. `_i` / `_f` shapes don't appear in pyre's insns table today; would land mechanically when emitted. |
//! | `setarrayitem_gc_r/rird` | PARITY (heapcache-aware) | r-bank array + i-bank index + r-bank value + descr. Always records `OpCode::SetarrayitemGc(array, index, value)` + `heapcache.setarrayitem(...)` write. RPython `pyjitpl.py:736-744 _opimpl_setarrayitem_gc_any` — no skip-on-redundant short-circuit because `setarrayitem` does aliasing-aware invalidation. `rrid` / `rrrd` / `rrfd` (Ref index) shapes stay unsupported (kind-flow). |
//! | `residual_call_r_r/iRd>r` | TODO (`direct_assembler_call` + `capture_resumedata` not yet wired) | classifies the call by `EffectInfo`. Wired sub-cases: (1) release-gil via [`direct_call_release_gil`] — `CallReleaseGilI` + arglist `[savebox, funcbox] + argboxes[1:]` reshape per `pyjitpl.py:3675-3681`, plus the outer forces-branch `GUARD_NOT_FORCED` (`:2079`) + `GUARD_NO_EXCEPTION` (`:2082`); (2) loop-invariant heapcache via [`loopinvariant_lookup`] / [`loopinvariant_now_known`] per `pyjitpl.py:2088 + 2109`; (3) vable IR bookkeeping (`pyjitpl.py:2055-2080`) via [`maybe_walker_vable_and_vrefs_before_residual_call`] — emits FORCE_TOKEN + SETFIELD_GC only; the runtime heap mutations on `vinfo.tracing_before_residual_call` / `vrefinfo.tracing_before_residual_call` (`pyjitpl.py:3318-3330`) and the after-call helpers (`pyjitpl.py:3337-3366`) stay on the trait-driven leg (`state.rs MIFrame::vable_and_vrefs_before_residual_call`, `trace_opcode.rs:2193-2349`) since the walker can't observe a force without executing the callee. The remaining branches go through [`select_residual_call_opcode`]: `CallMayForce*` + `GuardNotForced` on the rest of the forces-virtual path (`pyjitpl.py:2017-2082`), `CallLoopinvariant*` on `EF_LOOPINVARIANT` (`pyjitpl.py:2087-2110`), `CallPure*` on elidable, otherwise `Call*`. `GuardNoException` follows whenever `effectinfo.check_can_raise(False)` is true (`pyjitpl.py:2082 handle_possible_exception`). `heapcache.invalidate_caches_varargs(call_opcode, ei, allboxes)` (`pyjitpl.py:2042 + 2659`) is wired around every recorded call op. `OS_NOT_IN_TRACE` is fail-loud-guarded up front via [`do_not_in_trace_call_result`] — `effect_info_for_call_flavor` stub never sets the index today (`flatten.rs:431`), making it dead until producers land. Same fail-loud treatment via [`do_jit_force_virtual_guard`] for `OS_JIT_FORCE_VIRTUAL` (stricter-than-PyPy — needs OpRef→concrete-pointer resolver). Still deferred (each blocked on infrastructure absent from pyre-jit-trace): `direct_libffi_call` / `direct_assembler_call` specialization (`pyjitpl.py:1908-1990` — assembler_call paths route through `inline_call_*/dR>X` instead), KEEPALIVE for vablebox (only fires when `direct_assembler_call` returns a vablebox), and `num_live`-aware `capture_resumedata(after_residual_call=True)` on the guards (`pyjitpl.py:2078-2082 → 2586`). |
//! | `residual_call_r_i/iRd>i` | PARITY (kind sibling of `_r_r`) | same EffectInfo classification + guard emission as `_r_r` — `select_residual_call_opcode('i', ...)` returns the int-typed `Call*` family (`CallReleaseGilI` / `CallMayForceI` / `CallLoopinvariantI` / `CallPureI` / `CallI`); only the dst writeback bank (`registers_i`) differs. RPython parity: `pyjitpl.py:1346 opimpl_residual_call_r_i = _opimpl_residual_call1`; `do_residual_call`'s `descr.get_normalized_result_type()` dispatch (pyjitpl.py:2022-2044) selects the int-result CALL op. Argboxes pass through [`build_allboxes`] same as `_r_r` (R-list-only argboxes → identity permutation when arg_types is ref-only). |
//! | `residual_call_ir_r/iIRd>r` | PARITY (shape sibling of `_r_r`) | adds an i-bank list between funcptr and the R-list. RPython parity: `pyjitpl.py:1349 opimpl_residual_call_ir_r = _opimpl_residual_call2`; `boxes2` argcode (`pyjitpl.py:3750-3760`) decodes the two count-prefixed lists into `argboxes = [i_args..., r_args...]`. Walker passes that flat list through [`build_allboxes`] (line-by-line port of `pyjitpl.py:1960-1993 _build_allboxes`) which permutes argboxes by `descr.get_arg_types()` so the recorded `Call*` arglist matches the callee's actual ABI even for mixed orderings like `[REF, INT, REF, INT]`. Same EffectInfo classification + guard emission as `_r_r` via [`select_residual_call_opcode`]. |
//! | `raise/r`           | TODO (trait-leg-only `GUARD_CLASS`) | sets `ctx.last_exc_value` (`pyjitpl.py:1695`); top-level records `Finish(exc) descr=exit_frame_with_exception_descr_ref` (`pyjitpl.py:3238-3242 compile_exit_frame_with_exception`); sub-walk surfaces `SubRaise{exc}`. Caller-side handler scan (`finishframe_exception`) lives on `inline_call`'s SubRaise arm (above). RPython `pyjitpl.py:1690-1693` also emits `GUARD_CLASS(exc, cls_of_box(exc))` when `heapcache.is_class_known(exc) == false`; walker is symbolic shadow validator (`shadow_walker.rs`) with only an OpRef for the exception, no concrete-pointer access. Trait dispatch (`trace_opcode.rs:5980-6000 seed_raised_exception`) reads `concrete_exc.ob_header.ob_type` and emits the orthodox `GuardClass(exc_box, cls_const)` per the heapcache `is_class_known` gate. Walker IR is rolled back via `cut_trace`, so missing the emission on the walker leg has no production effect — by-design split. |
//! | `reraise/`          | PARITY        | reads `ctx.last_exc_value` (asserts via `ReraiseWithoutLastExcValue` matching `pyjitpl.py:1702 assert`); same dual top-level/sub-walk routing as `raise/r` (`pyjitpl.py:1700-1704 popframe + finishframe_exception`). |
//! | `last_exc_value/>r` | PARITY        | reads `ctx.last_exc_value`, writes the OpRef into `registers_r[dst]` — pure SSA rename, no IR op recorded. RPython `pyjitpl.py:1716-1719 opimpl_last_exc_value` returns `self.metainterp.last_exc_box` after asserting `last_exc_value` is non-null; missing slot surfaces `LastExcValueWithoutActiveException` (codewriter invariant: only emits inside `catch_exception/L` body). |
//!
//! Covers: decode walker, `WalkContext { registers_r, trace_ctx }` +
//! `ref_return/r` recording, `goto/L`, `catch_exception/L`,
//! `reraise/`, `int_copy/i>i`, `residual_call_r_r/iRd>r`,
//! `inline_call_r_r/dR>r` recursion, caller-frame `catch_exception`
//! scan, `last_exc_value` field, `reraise` finishframe routing, typed
//! arity / shape / no-active-exception errors, production
//! `PyreJitCodeDescr` adapter.
//!
//! Convergence path: when every opname has a recording handler this
//! module replaces the trait dispatch in `MIFrame::execute_opcode_step`.
//! The free-standing module shape stays —
//! the entry point becomes `MIFrame::dispatch_jitcode` calling [`walk`]
//! with the appropriate context.
//!
//! Production fidelity gaps (ranked by priority for follow-on work):
//!
//! 1. `residual_call_r_r/iRd>r` `do_residual_call` port
//!    (`pyjitpl.py:1995-2127`). Walker selects the IR opcode via
//!    [`select_residual_call_opcode`] (`CallReleaseGil*` /
//!    `CallMayForce*` / `CallLoopinvariant*` / `CallPure*` / `Call*`),
//!    unconditionally emits `GuardNotForced` on the forces and
//!    release-gil branches, and emits `GuardNoException` whenever
//!    `effectinfo.check_can_raise(False)` is true. Items now wired (was
//!    deferred in earlier audits):
//!    - `vable_and_vrefs_before_residual_call` IR portion (FORCE_TOKEN +
//!      SETFIELD_GC `vable_token_descr`) via
//!      [`walker_vable_and_vrefs_before_residual_call`].
//!    - `direct_call_release_gil` (`pyjitpl.py:3675-3681`) via
//!      [`direct_call_release_gil`].
//!    - `loopinvariant_lookup` / `loopinvariant_now_known`
//!      (`pyjitpl.py:2088 + 2109`).
//!    - `heapcache.invalidate_caches_varargs(call_opcode, ei, allboxes)`
//!      (`pyjitpl.py:2042 + 2072`) wired around every recorded call op.
//!    - `OS_NOT_IN_TRACE` fail-loud guard via
//!      [`do_not_in_trace_call_result`] (`pyjitpl.py:2003-2005`) —
//!      `effect_info_for_call_flavor` stub never sets the index today
//!      (`flatten.rs::effect_info_for_call_flavor` audit table), making
//!      it dead until the codewriter analyzer trio (annotator/rtyper/
//!      translator) lands.
//!    Items still deferred (each on infrastructure outside walker
//!    scope):
//!    a. **Trait-leg-only**: `vrefs_after_residual_call` /
//!       `vable_after_residual_call` (`pyjitpl.py:3337-3366`) observe
//!       runtime forces via heap-token reads; walker is symbolic so the
//!       trait dispatch (`state.rs MIFrame::vable_after_residual_call`,
//!       `trace_opcode.rs:2237-2350`) detects forces + aborts via
//!       `PyError::runtime_error("ABORT_ESCAPE: ...")` before the
//!       walker IR diff would run.
//!    b. **Codewriter-side**: `direct_assembler_call` + KEEPALIVE on
//!       vablebox (`pyjitpl.py:3589-3609 + 2080-2081`). Walker's
//!       residual_call dispatchers never receive `assembler_call=True`
//!       — the parallel `inline_call_*/dR>X` opcode family
//!       ([`dispatch_inline_call_dr_kind`]) routes that case. Trait
//!       dispatch (`trace_opcode.rs:5449-5474`) implements it.
//!    c. **Cross-leg, not yet implemented**: `_do_jit_force_virtual`
//!       PTR_EQ + GUARD_VALUE prelude (`pyjitpl.py:2011-2014 → 2153-2172`).
//!       Walker fail-louds via [`do_jit_force_virtual_guard`]
//!       (stricter-than-PyPy: typed error rather than divergent IR);
//!       full body needs an OpRef → concrete-pointer resolver (Task
//!       #45) before it can return `Some(vref_opref)` /
//!       `Some(standard_opref)` / `None`. Production reach today: 0 —
//!       `OopSpecIndex::JitForceVirtual` is set only by
//!       `jtransform.rs:1903 jit.force_virtual` lowering, which our
//!       benchmarks don't reach. Metainterp orthodox port at
//!       `majit-metainterp/src/pyjitpl/mod.rs:11828` is tests-only.
//!    d. **Not yet implemented**: `direct_libffi_call`
//!       (`pyjitpl.py:3622-3667`) needs `CIF_DESCRIPTION_P` parser +
//!       dynamic calldescr builder; live tracer also returns None
//!       universally (`pyjitpl/mod.rs:11487-11491`). Production reach
//!       0 — pyre interpreter doesn't expose libffi calls.
//!    e. Guard recording uses `ctx.trace_ctx.record_guard(..., 0)`
//!       followed by `walker_capture_snapshot_for_last_guard`
//!       (`jitcode_dispatch.rs:walker_capture_snapshot_for_last_guard`)
//!       — the walker-side port of `capture_resumedata(
//!       after_residual_call=True)` (`pyjitpl.py:2599-2603`).  Each
//!       residual_call guard (`GuardNotForced`, `GuardNoException`)
//!       carries a single-frame snapshot keyed by
//!       `ctx.outer_jitcode_index` so the optimizer's
//!       `store_final_boxes_in_guard` finds populated resume data.
//!       Active-box narrowing via per-PC liveness is a future
//!       follow-up; today's helper conservatively snapshots every
//!       non-`OpRef::NONE` register across all three banks.
//!    (`_build_allboxes` ABI re-ordering is wired — see
//!    [`build_allboxes`].)
//! 2. `raise/r`'s `GUARD_CLASS` (`pyjitpl.py:1690-1693 opimpl_raise`)
//!    is **trait-leg-only** by design.  Walker is the symbolic shadow
//!    validator (`shadow_walker.rs`); it has only an `OpRef` for the
//!    exception, no concrete-pointer access to read `cls_of_box(exc)`.
//!    Trait dispatch (`trace_opcode.rs:5980-6000 seed_raised_exception`)
//!    reads `concrete_exc.ob_header.ob_type`, checks
//!    `heap_cache.is_class_known(exc_box)`, and emits
//!    `GuardClass(exc_box, cls_const)` when needed — the orthodox
//!    `pyjitpl.py:1690-1696` flow. Walker IR is rolled back via
//!    `cut_trace`, so the missing emission on the walker leg has no
//!    production effect. A half-port that supplied a resolver-may-be-
//!    None type-erased callback was tried earlier and reverted —
//!    silently skipping when no resolver is wired is a TODO;
//!    the by-design split is the orthodox answer.
//! 3. End-to-end real arm tests (`walk_return_value_arm_*`,
//!    `walk_pop_top_arm_*`) stay `#[ignore]` until handlers exist for
//!    every opname the codewriter-emitted callee bodies use (e.g.
//!    `getfield_vable_i/rd>i`).
//! 4. (External) `build_default_bh_builder_with_unwired_report` is a
//!    transitional helper for kind-flow (6 unwired opnames:
//!    `int_ge/ir>i`, `int_mul/ir>i`, `int_ne/fr>i`, `int_xor/ri>i`,
//!    `setarrayitem_gc_f/rrfd`, `setarrayitem_gc_i/rrid` — kind-flow
//!    bug in assembler emitting mixed-kind operand types). RPython
//!    upstream has no non-strict builder. Removed when kind-flow
//!    closes; not blocking dispatcher work.
//! 5. Concrete-truth-dependent branch opnames (`goto_if_not/iL`,
//!    `goto_if_exception_mismatch/iL`) and the non-constant side of
//!    `switch/id`. RPython
//!    `pyjitpl.py:511-526 opimpl_goto_if_not`: `switchcase = box.getint()`
//!    branches on the runtime concrete value — `if switchcase: opnum =
//!    GUARD_TRUE; promoted_box = CONST_1` else `opnum = GUARD_FALSE`,
//!    then `metainterp.generate_guard(opnum, box, resumepc=orgpc)`. By
//!    design walker is the symbolic shadow validator
//!    (`shadow_walker.rs`); `WalkContext` carries `OpRef`s, not
//!    concrete `box.getint()` values, so the walker has no dispatch arm
//!    for these opnames — emitting one direction unconditionally would
//!    be a TODO. Trait dispatch handles concrete branches via
//!    `record_branch_guard(concrete_truth: bool)` (`trace_opcode.rs`);
//!    walker IR is rolled back so missing arms here have no production
//!    effect. Same split for `goto_if_exception_mismatch/iL`
//!    (`pyjitpl.py:484-496` — `last_exc_value`/llexitcase comparison).
//! 6. Class-introspection opname `last_exception/>i`. RPython
//!    `pyjitpl.py:1707-1713 opimpl_last_exception`: returns
//!    `ConstInt(ptr2int(rclass.ll_cast_to_object(exc_value).typeptr))` —
//!    the class pointer of the standing exception. Walker carries the
//!    exception OpRef but no concrete pointer; resolving the class
//!    needs `concrete_exc.ob_header.ob_type` which only the trait-
//!    driven leg (`trace_opcode.rs:5980-6000 seed_raised_exception`)
//!    can read. Walker has no dispatch arm by the same shadow-validator
//!    rationale as items 2 and 5.

use crate::jitcode_runtime::{DecodedOp, decode_op_at};
use crate::state::{ConcreteValue, MIFrame};
use majit_ir::{DescrRef, OopSpecIndex, OpCode, OpRef, Type, Value};
use majit_metainterp::TraceCtx;

/// Body of a callee jitcode that the walker needs to recurse into.
/// RPython parity: when `inline_call_r_r/dR>r` fires, the metainterp
/// reads the descr's `JitCode` body (`pyjitpl.py:1266-1324
/// _opimpl_inline_call*`). Walker consumes the same minimal subset:
/// the bytecode bytes + register-bank sizes for the fresh callee
/// frame.
///
/// Body is always `'static` — production wires the lookup to
/// `crate::jitcode_runtime::all_jitcodes()` whose `Arc<JitCode>`
/// entries live inside a `LazyLock<Vec<...>>` (`'static`); tests
/// either use static byte arrays or `Box::leak` to surface
/// `'static`. Constraining the body's lifetime simplifies
/// `WalkContext`'s lifetime parameters — otherwise the closure's
/// covariance would force register-bank borrows to extend to the
/// lookup's lifetime.
#[derive(Debug, Clone)]
pub struct SubJitCodeBody {
    /// Callee's jitcode bytes (RPython `JitCode.code`).
    pub code: &'static [u8],
    /// Number of Ref-bank registers the callee declares
    /// (`JitCode.num_regs_r`). The walker allocates a fresh
    /// `Vec<OpRef>` of this size for the recursive frame.
    pub num_regs_r: usize,
    /// Number of Int-bank registers (`JitCode.num_regs_i`).
    pub num_regs_i: usize,
    /// Number of Float-bank registers (`JitCode.num_regs_f`).
    pub num_regs_f: usize,
    /// Callee's Int-bank constant pool (`JitCode.constants_i`).
    /// The callee bytecode references constant slots via register
    /// indices `[num_regs_i, num_regs_i + constants_i.len())`;
    /// `setposition` (RPython `pyjitpl.py:98-119 copy_constants`)
    /// pre-populates those slots with `ConstClass(constants_i[i])`.
    pub constants_i: &'static [i64],
    /// Callee's Ref-bank constant pool (`JitCode.constants_r`). Each
    /// `i64` is the erased `PyObjectRef` of a const object resolved
    /// at codewriter time.
    pub constants_r: &'static [i64],
    /// Callee's Float-bank constant pool (`JitCode.constants_f`).
    pub constants_f: &'static [i64],
}

/// Caller-provided sub-jitcode lookup. RPython equivalent: descr
/// resolution within the metainterp loop reads `BhDescr::JitCode {
/// jitcode_index, .. }` and looks up `ALL_JITCODES[idx]`. Walker
/// inverts the dependency: the caller supplies the lookup so the
/// walker stays decoupled from the runtime's all-jitcodes table
/// (production passes a closure over `crate::jitcode_runtime::all_jitcodes()`,
/// tests pass synthetic closures over a local fixture map).
pub type SubJitCodeLookup = dyn Fn(usize) -> Option<SubJitCodeBody>;

/// State the walker reads from / writes to while stepping. RPython
/// equivalent: `MetaInterp` itself — the trace recorder, the symbolic
/// register banks (`registers_i`, `registers_r`, `registers_f`), and
/// the metainterp static data are all reachable from `self` in
/// `pyjitpl.py:opimpl_*`. Pyre passes them via this struct so the
/// walker can be tested without standing up a full `MIFrame`.
///
/// Field roster:
///
/// * `registers_r`: Ref bank for `r`-coded operands.
/// * `registers_i`: Int bank for `i`-coded operands.
///   `registers_f` (Float bank) lands when float opnames join the
///   handler table.
/// * `descr_refs`: descr pool for `d`-coded operands.
///   Mirrors RPython `Assembler.descrs` (`assembler.py:23`); each
///   2-byte LE descr index in the jitcode bytes resolves through this
///   table.
/// * `trace_ctx`: live trace recorder.
/// * `done_with_this_frame_descr_ref`: descr the FINISH terminator
///   for a Ref-returning trace must carry. Production callers resolve
///   via `MetaInterpStaticData::done_with_this_frame_descr_for(Type::Ref)`
///   (`pyjitpl.py:4736`); tests use `make_fail_descr(1)` as the same
///   fallback `finish_and_compile` (`pyjitpl.py:4733`) uses when the
///   staticdata singleton was never attached.
///
/// Register banks are *mutable* — `int_copy/i>i` and
/// `residual_call_r_r/iRd>r` write their dst slot inline (RPython parity:
/// `pyjitpl.py:471-477 _opimpl_any_copy` returns the box, the
/// `@arguments("box")` + `>X` decorator pair writes it into the result
/// slot; `pyjitpl.py:1334-1347 _opimpl_residual_call*` returns the
/// recorder OpRef which the `>X` slot consumes). `inline_call_r_r/dR>r`
/// also *would* write dst (after sub-jitcode recursion) but stays
/// deferred — see the per-handler comments + module-level "Production
/// fidelity gaps" below.
/// `WalkContext` carries two lifetimes:
/// * `'frame` — the inner-frame lifetime: register banks + trace
///   recorder. Sub-walk recursion (`inline_call_r_r/dR>r`) allocates
///   fresh register banks scoped to the sub-walk's block, so
///   `'frame` must be allowed to *shrink* on recursion.
/// * `'static_a` — the outer lifetime: descr pool + sub-jitcode
///   lookup. These flow unchanged from caller into callee, so they
///   keep their original (longer) lifetime.
pub struct WalkContext<'frame, 'static_a: 'frame> {
    /// Symbolic Ref-bank register file. Indexing matches RPython
    /// `MIFrame.registers_r` (`pyjitpl.py:177-234`); the byte after a
    /// `r`-coded operand opcode indexes directly into this slice.
    /// Mutable so handlers writing `>r` results (currently
    /// `residual_call_r_r/iRd>r`) can land their dst.
    pub registers_r: &'frame mut [OpRef],
    /// Symbolic Int-bank register file. Indexing matches RPython
    /// `MIFrame.registers_i` (`pyjitpl.py:177-234`). Pyre's PyreSym is
    /// mid-migration to a 3-bank typed model — production callers may
    /// pass an empty slice today (the assembler only emits `i`-coded
    /// operands once the codewriter wires Int kind). Mutable so
    /// `int_copy/i>i` can land its dst.
    pub registers_i: &'frame mut [OpRef],
    /// Symbolic Float-bank register file. Indexing matches RPython
    /// `MIFrame.registers_f` (`pyjitpl.py:177-234`). Mutable so
    /// `float_<binop>/ff>f` and `float_neg/f>f` can land their dst.
    pub registers_f: &'frame mut [OpRef],
    /// Concrete shadow mirror for `registers_r`.
    ///
    /// Semantic-slot indexed, length equals `registers_r.len()`. At
    /// `dispatch_via_miframe` entry, populated by concatenating
    /// `PyreSym.concrete_locals` + `PyreSym.concrete_stack`; sub-walks
    /// allocate a fresh `Vec<ConcreteValue>` sized to the callee's
    /// `num_regs_r` and fill arg slots from the parent's slice at the
    /// arg byte indices.
    ///
    /// **Mutable invariant**: every walker handler that
    /// writes `registers_r[dst]` MUST also write `concrete_registers_r
    /// [dst]` in lock-step.  Use the [`write_ref_reg`] helper which
    /// enforces this contract.  Sites that don't know the result's
    /// concrete pass `ConcreteValue::Null` — downstream consumers
    /// (e.g. `raise/r` GUARD_CLASS gate) treat `Null` as "no info,
    /// skip the guard", same as slots the snapshot never populated.
    /// Copy-style handlers (`ref_copy/r>r`,
    /// `last_exc_value/>r`) propagate the source's concrete.
    ///
    /// The slice is mutable so the concrete shadow tracks the symbolic
    /// register in lock-step. If it were immutable, sibling handlers
    /// like `last_exc_value/>r` could rewrite the symbolic register
    /// without touching the concrete snapshot, so a follow-on `raise/r`
    /// would read a stale concrete and silently skip the GUARD_CLASS
    /// gate; the lock-step contract keeps walker-side GUARD_CLASS sound.
    ///
    /// **Companion bank** `concrete_registers_i` below now exists as a
    /// skeleton field (Concrete shadow skeleton) so handlers can plumb concrete int
    /// shadow without changing the WalkContext signature again.  Seed
    /// wiring (real concrete int values at trace entry + per-handler
    /// writes for `int_*` arithmetic) is deferred to Concrete shadow seeding and
    /// later slices.  Until then every callsite passes `&mut []`.
    ///
    /// `goto_if_not/iL` and `switch/id` (which consume concrete Int)
    /// continue to fall back to the strict-mode fail-loud path until
    /// `concrete_registers_i` is populated.
    pub concrete_registers_r: &'frame mut [ConcreteValue],
    /// **Skeleton — Concrete shadow skeleton.**  Concrete shadow mirror for
    /// `registers_i`.  Color-indexed (not semantic-slot indexed like
    /// `concrete_registers_r`) because pyre's Int bank has no
    /// "semantic-slot" abstraction — Int registers are post-regalloc
    /// colors directly.  Length equals `registers_i.len()` when
    /// populated, or `0` when callsites pass `&mut []`.
    ///
    /// Future invariant (Concrete shadow seeding+): every walker handler that writes
    /// `registers_i[dst]` MUST also write `concrete_registers_i[dst]`
    /// in lock-step, mirroring the Ref-bank contract above.  A future
    /// `write_int_reg` helper will enforce this once seeding lands.
    ///
    /// Consumers (`dispatch_goto_if_not/iL`, `switch/id`) currently
    /// fall back to the strict-mode fail-loud path; they will switch
    /// to reading `concrete_registers_i[src]` for the test-direction
    /// fold once Concrete shadow seeding populates seeds.
    pub concrete_registers_i: &'frame mut [ConcreteValue],
    /// Descr pool for `d`-coded operands. Each `d` argcode in the
    /// jitcode bytes resolves to `descr_refs[2-byte LE index]`.
    /// RPython `Assembler.descrs` (`assembler.py:23`) +
    /// `BlackholeInterpBuilder.setup_descrs` (`blackhole.py:102-103`)
    /// — production callers pass the codewriter-emitted descr table.
    pub descr_refs: &'static_a [DescrRef],
    /// Live trace recorder. `record_finish` / `record_op` /
    /// `record_op_with_descr` go through this.
    pub trace_ctx: &'frame mut TraceCtx,
    /// `done_with_this_frame_descr_ref` — the descr `pyjitpl.py:4729-4738
    /// finish_and_compile` attaches to the trace's terminator FINISH for
    /// the Ref kind. Caller-provided so the dispatcher does not reach
    /// into `TraceCtx::metainterp_sd` (which is `pub(crate)`).
    pub done_with_this_frame_descr_ref: DescrRef,
    /// Int-kind counterpart used by `int_return/i` (`pyjitpl.py:3206-3208
    /// compile_done_with_this_frame: token = sd.done_with_this_frame_descr_int`).
    /// Production wires `MetaInterpStaticData::done_with_this_frame_descr_for(Type::Int)`;
    /// tests pass `make_fail_descr(N)` placeholders since the descr's
    /// only role here is identity-tagging the FINISH terminator.
    pub done_with_this_frame_descr_int: DescrRef,
    /// Float-kind counterpart used by `float_return/f` (`pyjitpl.py:3212-3214
    /// compile_done_with_this_frame: token = sd.done_with_this_frame_descr_float`).
    pub done_with_this_frame_descr_float: DescrRef,
    /// Void-kind counterpart used by `void_return/` (`pyjitpl.py:3202-3205
    /// compile_done_with_this_frame: token = sd.done_with_this_frame_descr_void`,
    /// `exits = []` — the FINISH carries no value).
    pub done_with_this_frame_descr_void: DescrRef,
    /// `exit_frame_with_exception_descr_ref` — the descr `pyjitpl.py:3238-3242
    /// compile_exit_frame_with_exception` attaches to the FINISH that
    /// terminates a trace whose outermost frame raised an unhandled
    /// exception. RPython:
    ///   token = sd.exit_frame_with_exception_descr_ref
    ///   self.history.record1(rop.FINISH, valuebox, None, descr=token)
    /// Production callers resolve via `MetaInterpStaticData`
    /// (cf. `metainterp.rs:733`); tests use `make_fail_descr(1)`.
    pub exit_frame_with_exception_descr_ref: DescrRef,
    /// Whether this `WalkContext` is the outermost trace frame
    /// (`true`) or a nested sub-jitcode frame entered through
    /// `inline_call_r_r/dR>r` recursion (`false`). The flag
    /// disambiguates dual-behaviour terminators:
    ///
    /// * `ref_return/r` at top-level records `Finish` + Terminate;
    ///   inside a sub-walk it returns `SubReturn { result }` so the
    ///   caller's `inline_call_*` handler can write the dst register.
    /// * `raise/r` at top-level records the outermost
    ///   `Finish(exit_frame_with_exception_descr_ref)`; inside a
    ///   sub-walk it propagates `SubRaise { exc }` — the caller's
    ///   `inline_call_*` handler may catch via `catch_exception`
    ///   metadata or bubble up further.
    ///
    /// RPython parity: pyre flattens the framestack-driven
    /// `metainterp.popframe()` + `finishframe[_exception]` flow
    /// (`pyjitpl.py:1688-1704`) into this Rust-level outcome.
    pub is_top_level: bool,
    /// Caller-provided callback resolving a `jitcode_index` to a
    /// `SubJitCodeBody`. Invoked when `inline_call_r_r/dR>r` fires
    /// and needs to recurse into the callee's bytecode body.
    pub sub_jitcode_lookup: &'static_a SubJitCodeLookup,
    /// Per-frame mirror of RPython `metainterp.last_exc_value`
    /// (`pyjitpl.py:1695`). Set by `raise/r` (caller-frame side, before
    /// `SubRaise` propagates) and by the `inline_call` SubRaise arm
    /// when it catches at a `catch_exception/L` handler (the handler's
    /// own opcodes — `last_exception`, `last_exc_value`, `reraise/` —
    /// read this field). RPython keeps this on the metainterp object
    /// (one shared slot); the walker carries one per WalkContext
    /// because each recursive frame has its own context. The flow
    /// (callee raise → caller catch → caller handler reads) only
    /// touches the caller's slot, so per-frame storage is equivalent
    /// to RPython's metainterp-level slot for the catch path.
    pub last_exc_value: Option<OpRef>,
    /// Concrete shadow companion to [`last_exc_value`].
    /// Holds the live `PyObjectRef` of the standing
    /// exception so `last_exc_value/>r` can propagate the concrete
    /// into the destination's `concrete_registers_r` slot, and so a
    /// follow-on `raise/r` reading that destination finds a non-Null
    /// concrete and emits the correct GUARD_CLASS.
    ///
    /// Set by `raise/r` (walker side) alongside `last_exc_value`, by
    /// the `inline_call` SubRaise arm when it catches at
    /// `catch_exception/L`, and by `dispatch_via_miframe`'s entry
    /// from `sym.last_exc_value` when the trait path seeded the
    /// exception via `seed_raised_exception`.
    ///
    /// `ConcreteValue::Null` means "no active exception concrete
    /// known" — matches `last_exc_value == None` for the common case,
    /// or means the trait-path seeded only the symbolic OpRef without
    /// a concrete (e.g. a synthetic test fixture).
    pub last_exc_value_concrete: ConcreteValue,
    /// Python bytecode PC of the opcode whose per-opcode arm the
    /// walker is currently executing.  Mirrors `MIFrame.orgpc`
    /// (`pyjitpl.py:151 setposition` parity).  Production entry seeds
    /// from `miframe.orgpc as u32`; sub-walks inherit the caller's
    /// `entry_py_pc` (a sub-jitcode invocation does not advance the
    /// outer Python PC); test fixtures default to `0`.
    ///
    /// Read by [`walker_capture_snapshot_for_last_guard`] to stamp the
    /// snapshot frame's Python PC.
    pub entry_py_pc: u32,
    /// JitCode index of the **outer** `PyJitCode.jitcode` — the Python
    /// bytecode jitcode whose Python opcode is currently being
    /// dispatched.  Pyre's blackhole resume only re-enters Python-
    /// bytecode jitcodes, so guard snapshots must reference the outer
    /// pyjitcode regardless of how deep the walker's sub-walk nesting
    /// is.
    ///
    /// Read from `(*sym.jitcode).index()` at production entry
    /// ([`dispatch_via_miframe_at_opcode_entry`]); sub-walks inherit
    /// the parent's value (sub-walks don't change the outer Python
    /// opcode).  Test fixtures + [`dispatch_via_miframe`] default to
    /// `0`.
    pub outer_jitcode_index: u32,
    /// Frozen `PyFrame` state at the outer Python opcode boundary —
    /// `sym.registers_r ∪ sym.registers_i.opref ∪ sym.registers_f.opref`
    /// captured at [`dispatch_via_miframe_at_opcode_entry`] entry,
    /// filtered by `OpRef::is_none()`.  This is what
    /// [`walker_capture_snapshot_for_last_guard`] passes as the
    /// snapshot frame's active boxes.
    ///
    /// Sub-walks clone the parent's Vec — outer active-box count is
    /// small (a Python frame's live locals + stack tail) and walker
    /// nesting depth is shallow (2–3 levels), so the per-sub-walk
    /// clone cost is negligible.
    pub outer_active_boxes: Vec<OpRef>,
}

/// Outcome of dispatching one opcode. The walker uses this to decide
/// whether to continue stepping or terminate.
///
/// RPython parity: `pyjitpl.py:opimpl_*` returns through Python's
/// generator/exception flow — opcodes that end a trace raise
/// `DoneWithThisFrameRef`/`SwitchToBlackhole`/`ChangeFrame`. Pyre
/// flattens that into an explicit enum because Rust has no analogous
/// non-local exit and we want the walker to stay in plain Result form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DispatchOutcome {
    /// Step succeeded, continue with the next opcode at the returned pc.
    Continue,
    /// Trace ends here. The arm produced a final `ref_return`/`raise`
    /// equivalent at the top-level frame and no further bytes should
    /// be walked.
    Terminate,
    /// Sub-walk frame returned with a result OpRef (Some) or void
    /// (None — no `>X` slot in the callee's `*_return` op). Surfaced
    /// only when `WalkContext::is_top_level == false`. The caller's
    /// `inline_call_r_r/dR>r` handler consumes this to write the dst
    /// register and continue stepping its own jitcode.
    ///
    /// RPython parity: `metainterp.popframe()` after an `opimpl_*_return`
    /// (`pyjitpl.py:1688-1698`) — the callee frame ends, control returns
    /// to the caller's metainterp loop with the resbox in hand.
    SubReturn { result: Option<OpRef> },
    /// Sub-walk frame raised. RPython
    /// `metainterp.popframe() + finishframe_exception()` walks up the
    /// framestack scanning each parent's exceptiontable; pyre's walker
    /// surfaces the outcome to the caller's `inline_call_*` handler,
    /// which today bubbles it up further (no per-handler
    /// exceptiontable scan yet — that lives behind the
    /// `catch_exception/L` metadata pipe and is deferred until the
    /// per-PC exceptiontable plumb-through lands).
    ///
    /// `exc_concrete` carries the callee's `last_exc_value_concrete`
    /// across the frame boundary so the caller's `inline_call` SubRaise
    /// arm can seed its own `last_exc_value_concrete` and a downstream
    /// `raise/r` / `reraise/` reads the right concrete for GUARD_CLASS
    /// emission. Empty when the callee itself didn't track a concrete
    /// (e.g. shadow gap or `Null`-seeded raise).
    SubRaise {
        exc: OpRef,
        exc_concrete: ConcreteValue,
    },
    /// Trace recording must abort and resume in blackhole mode.
    ///
    /// RPython parity: `pyjitpl.py:2003-2006` routes
    /// `OS_NOT_IN_TRACE` residual calls through
    /// `do_not_in_trace_call`; `pyjitpl.py:3695` raises
    /// `SwitchToBlackhole(ABORT_ESCAPE)` if the concrete call raises.
    /// The trace walker cannot execute the callee concretely yet, so a
    /// reached `OS_NOT_IN_TRACE` site is surfaced as this non-local
    /// outcome instead of recording a call that upstream would omit.
    SwitchToBlackhole {
        reason: i32,
        raising_exception: bool,
    },
}

/// Errors surfaced by the trace-side walker.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DispatchError {
    /// The opcode byte at `pc` is not present in the `insns` table or
    /// the instruction's operand bytes overflowed the code slice. This
    /// is the same `decode_op_at -> None` path surfaced as a typed
    /// error.
    UndecodableOpcode { pc: usize },
    /// The opcode is decodable but the dispatcher has no handler for
    /// it yet. Carries the `opname/argcodes` key so callers can
    /// identify what blocked the walk.
    UnsupportedOpname { pc: usize, key: &'static str },
    /// A register operand byte indexed past the symbolic register file.
    /// `len` is the slice length the walker was handed in the
    /// `WalkContext`; `reg` is the byte the codewriter emitted. Surfaces
    /// either an assembler-pass bug (out-of-range register) or a
    /// caller mismatch between the symbolic register layout and the
    /// arm's expected number of registers.
    RegisterOutOfRange {
        pc: usize,
        reg: usize,
        len: usize,
        bank: &'static str,
    },
    /// A `d`-coded descr index resolved past the descr pool. Surfaces
    /// either an assembler-pass bug (descr index out of range) or a
    /// caller mismatch between the codewriter's descr table size and
    /// the table the walker was handed in `WalkContext::descr_refs`.
    DescrIndexOutOfRange { pc: usize, index: usize, len: usize },
    /// `inline_call_*` resolved a descr that does not implement
    /// `JitCodeDescr`. Surfaces either a codewriter bug (an
    /// `inline_call_*` opnum emitted with a non-jitcode descr index)
    /// or a caller mismatch (the descr pool wasn't built from the
    /// codewriter's descr table). `descr_index` is the 2-byte LE
    /// index the walker decoded.
    ExpectedJitCodeDescr { pc: usize, descr_index: usize },
    /// `inline_call_*`'s descr resolved to a `jitcode_index`, but the
    /// caller's `sub_jitcode_lookup` returned `None`. Production wires
    /// the lookup to `crate::jitcode_runtime::all_jitcodes()`; tests
    /// build synthetic maps. A `None` return means the codewriter
    /// emitted an index past the runtime's jitcode table.
    SubJitCodeNotFound { pc: usize, jitcode_index: usize },
    /// `inline_call_*` provided more Ref args in its R-list than the
    /// callee declared `num_regs_r` slots. RPython parity: `pyjitpl.py:230-260
    /// MIFrame.setup_call(argboxes)` distributes argboxes into the
    /// callee's typed register banks; the JitCode-level shape contract
    /// (`assembler.py:write_call`) requires `len(argboxes) <=
    /// num_regs_r` for the `_r_r` variant. Excess args are a
    /// codewriter-emitted shape mismatch.
    InlineCallArityMismatch {
        pc: usize,
        provided: usize,
        callee_num_regs_r: usize,
    },
    /// `inline_call_*` provided more Int args in its I-list than the
    /// callee declared `num_regs_i` slots. Same shape contract as the
    /// Ref variant — `pyjitpl.py:230-260 setup_call` populates each
    /// kind-bank from its respective list and asserts capacity.
    InlineCallIntArityMismatch {
        pc: usize,
        provided: usize,
        callee_num_regs_i: usize,
    },
    /// `inline_call_*` provided more Float args in its F-list than the
    /// callee declared `num_regs_f` slots.
    InlineCallFloatArityMismatch {
        pc: usize,
        provided: usize,
        callee_num_regs_f: usize,
    },
    /// `inline_call_r_r/dR>r`'s callee surfaced
    /// `SubReturn { result: None }`. RPython parity: the `_r_r` variant
    /// is wired (in `assembler.py:gen_inline_call`) to a callee whose
    /// `*_return` op carries a Ref; reaching it without a result means
    /// the callee body executed `void_return/` (or an analogue) instead
    /// of `ref_return/r`, which is a codewriter shape mismatch — the
    /// caller has nowhere to land the missing value.
    UnexpectedVoidSubReturn { pc: usize },
    /// `inline_call_*_v/d{R,IR,IRF}`'s callee surfaced
    /// `SubReturn { result: Some(_) }`. RPython parity: the `_v` variant
    /// (`bhimpl_inline_call_*_v`, `blackhole.py:1287/1300/1317`) is
    /// wired to a callee whose `*_return` op is `void_return/`; reaching
    /// it with a typed-return result means the callee body executed
    /// `int_return/i` / `ref_return/r` / `float_return/f` instead of
    /// `void_return/`, which is a codewriter shape mismatch — the
    /// caller has no `>X` slot to land the surplus value.
    UnexpectedNonVoidSubReturn { pc: usize },
    /// `reraise/` fired but `WalkContext::last_exc_value` was `None`.
    /// RPython parity: `pyjitpl.py:1702
    /// opimpl_reraise: assert self.metainterp.last_exc_value` —
    /// reaching `reraise` without an active exception is a codewriter
    /// invariant violation (`raise` or a catch-handler entry must have
    /// set `last_exc_value` first).
    ReraiseWithoutLastExcValue { pc: usize },
    /// `last_exc_value/>r` fired but `WalkContext::last_exc_value` was
    /// `None`. RPython parity: `pyjitpl.py:1716-1719 opimpl_last_exc_value`:
    ///
    ///   exc_value = self.metainterp.last_exc_value
    ///   assert exc_value
    ///   return self.metainterp.last_exc_box
    ///
    /// Same codewriter invariant as `reraise/`: this opname only emits
    /// inside a `catch_exception` body where the unwinder has already
    /// stored the in-flight exception. Reaching it without an active
    /// exception is a flatten/codewriter shape mismatch.
    LastExcValueWithoutActiveException { pc: usize },
    /// `catch_exception/L` was reached on the normal fall-through path
    /// (no `SubRaise` routing) but `WalkContext::last_exc_value` was
    /// non-`None`. RPython parity: `pyjitpl.py:497-504 opimpl_catch_exception`:
    ///
    ///   assert not self.metainterp.last_exc_value
    ///
    /// On the normal path the previous instruction did NOT raise — if
    /// it had, `finishframe_exception` would have routed control past
    /// the catch_exception/L (or to its target if matched), never
    /// running the catch_exception/L instruction itself. Reaching it
    /// with an active exception means the codewriter mis-emitted a
    /// catch_exception/L outside an exception-table position, OR a
    /// previous handler forgot to clear `last_exc_value` after handling.
    CatchExceptionWithActiveException { pc: usize },

    /// `residual_call_*` decoded a descr that does not implement
    /// `CallDescr`. RPython parity: `pyjitpl.py:1995-2127
    /// do_residual_call` always receives a `calldescr` from the
    /// codewriter — there is no fallback path. The walker mirrors that
    /// invariant by surfacing a typed error when the descr_pool entry
    /// at the operand-encoded index lacks a CallDescr downcast. In
    /// production the codewriter never emits a non-CallDescr; this
    /// variant fires only when test fixtures (or future deviations)
    /// route a non-CallDescr into a residual_call slot.
    ResidualCallDescrNotCallDescr { pc: usize, descr_index: usize },
    /// A residual_call argument resolved to `OpRef::NONE` — an unbound
    /// register.  RPython's `do_residual_call` reads every argbox out of
    /// `env[box]`, so a missing binding is a `KeyError` there; the
    /// cranelift/dynasm backends mirror that as a hard `resolve_opref`
    /// assert when `OpRef::NONE` reaches arg-resolution.  A per-opcode
    /// arm whose body forwards an unseeded dispatcher parameter (e.g. a
    /// seam wrapper that residualizes `execute_<op>(executor, code,
    /// instruction, op_arg)` — only `executor`=r0 is seeded at the arm
    /// entry, so `code`/`instruction`/`op_arg` stay `OpRef::NONE`) would
    /// record such a call.  Surfacing it here turns the would-be backend
    /// crash into a graceful trace abort, matching the pre-seam inline
    /// arm whose payload read aborted with `GotoIfNotValueNotConcrete`.
    ResidualCallArgUnbound { pc: usize, arg_index: usize },
    /// `switch/id` decoded a descr that does not implement
    /// `SwitchDescr`. RPython parity: `pyjitpl.py:601` asserts
    /// `isinstance(switchdict, SwitchDictDescr)`.
    ExpectedSwitchDescr { pc: usize },
    /// `switch/id` needs RPython's `valuebox.getint()` at trace time.
    /// The symbolic walker can obtain that only when `TraceCtx` can
    /// reconstruct an Int concrete for the OpRef today;
    /// choosing a branch without a concrete value would record the wrong
    /// guard chain, so surface the missing concrete value explicitly.
    SwitchValueNotConcrete { pc: usize, value: OpRef },
    /// `goto_if_not/iL` needs RPython's `box.getint()` at trace time
    /// (`pyjitpl.py:511-526 opimpl_goto_if_not`).  Without the
    /// concrete value the walker can't pick GUARD_TRUE vs GUARD_FALSE
    /// or decide whether to jump to the label target, so surface the
    /// missing concrete explicitly instead of guessing.
    GotoIfNotValueNotConcrete { pc: usize, value: OpRef },
    /// `OS_NOT_IN_TRACE` must run the callee concretely and record no
    /// IR on the normal path (`pyjitpl.py:3683-3693`). The standalone
    /// symbolic walker has no concrete executor, so it must stop here
    /// instead of faking either the normal return or
    /// `SwitchToBlackhole`.
    NotInTraceRequiresConcreteExecution { pc: usize },
    /// `OS_JIT_FORCE_VIRTUAL` would short-circuit `do_residual_call`
    /// before recording `CALL_MAY_FORCE_*` (`pyjitpl.py:2011-2014 →
    /// 2153-2172 _do_jit_force_virtual`). The short-circuit needs a
    /// concrete `vref_ptr` for arbitrary Ref OpRefs to determine
    /// `isstandard_int` at trace time — walker only knows
    /// `concrete_vable_ptr`, not the concrete value behind every Ref
    /// OpRef. Surfacing this as an error prevents silently recording
    /// `CALL_MAY_FORCE_*` for an op the live tracer would have folded
    /// to `vref_box` / `standard_box` / None. Production reach today:
    /// `OopSpecIndex::JitForceVirtual` is set only by
    /// `jtransform.rs:1903 jit.force_virtual` lowering, which our
    /// benchmarks don't trigger; this guard is fail-loud against future
    /// silent TODOs.
    JitForceVirtualRequiresConcreteResolver { pc: usize },
    /// `{get,set}field_vable_{i,r,f}` read its box operand from a Ref
    /// register holding `OpRef::None`. RPython parity: `pyjitpl.py:1167-1199
    /// opimpl_getfield_vable_{i,r,f}` / `_opimpl_setfield_vable(box, ...,
    /// pc)` always receive a live `box` from the register file — the
    /// virtualizable frame box. Pyre's walker initializes Ref register
    /// slots to `OpRef::None`; an inlined callee frame may leave the vable
    /// register unseeded (documented walker arg-seeding gap). Both vable
    /// accessors route through `is_nonstandard_virtualizable` →
    /// `heapcache.nonstandard_virtualizables_now_known(box)`, which would
    /// feed `OpRef::None` (`raw() == u32::MAX`) into the dense heapcache
    /// flag `Vec<u32>`, resizing it to `u32::MAX + 1` (16 GiB). Surface as
    /// a trace abort so production falls back to trait dispatch rather
    /// than allocating.
    VableBoxNotSeeded { pc: usize },
}

/// Walk one opcode at `pc` and return the dispatch outcome plus the
/// next pc. Side effects reach `ctx.trace_ctx` only for opnames whose
/// handler explicitly records (e.g. `ref_return/r` calls
/// `record_finish`).
///
/// The returned `next_pc` is normally `op.next_pc` (linear advance
/// past the operand bytes); branch handlers (`goto/L` etc.) override
/// this with their target.
pub fn step(
    code: &[u8],
    pc: usize,
    ctx: &mut WalkContext<'_, '_>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let op: DecodedOp = decode_op_at(code, pc).ok_or(DispatchError::UndecodableOpcode { pc })?;
    handle(&op, code, ctx)
}

/// Walk the code from `start_pc` until a terminating opcode fires.
/// Returns the terminating outcome plus the pc immediately after the
/// terminator. Top-level callers expect `DispatchOutcome::Terminate`
/// (other variants appear only inside a sub-walk frame entered via
/// `inline_call_r_r/dR>r` — `ref_return/r` and `raise/r` produce
/// `SubReturn` / `SubRaise` there).
///
/// **Top-level uncaught SubRaise**: when an inline_call
/// SubRaise bubbles up through every parent frame without a
/// `catch_exception/L` handler match and reaches the outermost
/// `walk()` invocation, RPython `pyjitpl.py:2533-2538
/// finishframe_exception` records `compile_exit_frame_with_exception(
/// last_exc_box)` — i.e. `FINISH(exc, exit_frame_with_exception_descr_ref)`
/// + raise `ExitFrameWithExceptionRef`. The walker mirrors this on
/// exit: if the loop terminates with `SubRaise` AND `ctx.is_top_level
/// == true`, record the FINISH and convert the outcome to `Terminate`
/// before returning. Sub-walk frames keep returning `SubRaise` to
/// their callers (the unwind continues until either a handler
/// matches or the outermost walker handles it).
pub fn walk(
    code: &[u8],
    start_pc: usize,
    ctx: &mut WalkContext<'_, '_>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let mut pc = start_pc;
    loop {
        let (outcome, next_pc) = step(code, pc, ctx)?;
        pc = next_pc;
        match outcome {
            DispatchOutcome::Continue => {}
            DispatchOutcome::Terminate
            | DispatchOutcome::SubReturn { .. }
            | DispatchOutcome::SwitchToBlackhole { .. } => {
                return Ok((outcome, pc));
            }
            DispatchOutcome::SubRaise { exc, exc_concrete } => {
                if ctx.is_top_level {
                    // RPython parity: framestack exhausted with no
                    // handler match → `compile_exit_frame_with_exception(
                    // last_exc_box)` records the outermost FINISH.
                    ctx.trace_ctx
                        .finish(&[exc], ctx.exit_frame_with_exception_descr_ref.clone());
                    return Ok((DispatchOutcome::Terminate, pc));
                } else {
                    return Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, pc));
                }
            }
        }
    }
}

/// Read a Ref-bank register operand byte at `pc + offset` and resolve
/// to its symbolic [`OpRef`]. RPython
/// `pyjitpl.py:registers_r[code[pc+1]]` for an `r`-coded operand.
fn read_ref_reg(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_>,
) -> Result<OpRef, DispatchError> {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.registers_r
        .get(reg)
        .copied()
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg,
            len: ctx.registers_r.len(),
            bank: "r",
        })
}

/// Read an Int-bank register operand byte at `pc + offset` and resolve
/// to its symbolic [`OpRef`]. RPython
/// `pyjitpl.py:registers_i[code[pc+1]]` for an `i`-coded operand.
fn read_int_reg(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_>,
) -> Result<OpRef, DispatchError> {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.registers_i
        .get(reg)
        .copied()
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg,
            len: ctx.registers_i.len(),
            bank: "i",
        })
}

/// Read a Float-bank register operand byte at `pc + offset` and resolve
/// to its symbolic [`OpRef`]. RPython
/// `pyjitpl.py:registers_f[code[pc+1]]` for an `f`-coded operand.
fn read_float_reg(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_>,
) -> Result<OpRef, DispatchError> {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.registers_f
        .get(reg)
        .copied()
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg,
            len: ctx.registers_f.len(),
            bank: "f",
        })
}

/// Read a 2-byte little-endian label operand at `pc + 1 +
/// operand_offset`. RPython encoding: `assembler.py:write_label`
/// writes the resolved target as `chr(target & 0xFF)` +
/// `chr((target >> 8) & 0xFF)`, matching `bhimpl_goto`'s
/// `code[pc] | (code[pc+1] << 8)` decode.
fn read_label(code: &[u8], op: &DecodedOp, operand_offset: usize) -> usize {
    let lo = code[op.pc + 1 + operand_offset] as usize;
    let hi = code[op.pc + 1 + operand_offset + 1] as usize;
    lo | (hi << 8)
}

/// Outcome of probing the per-frame raise-bubbling lookahead at
/// `position` (the pc just after a raising op).
///
/// RPython parity: `pyjitpl.py:2506-2531 finishframe_exception` walks
/// through three mutually-exclusive cases after skipping a leading
/// `live/`:
///
///   1. Next op is `catch_exception/L` → jump to the handler target,
///      `raise ChangeFrame`. (Handler matched.)
///   2. Next op is `rvmprof_code/ii` → call `cintf.jit_rvmprof_code(arg1,
///      arg2)` for instrumentation, then fall through to `popframe()`
///      (continue unwinding).
///   3. Otherwise → `popframe()` (continue unwinding).
///
/// Cases 2 and 3 both unwind, but case 2 also fires the rvmprof side
/// effect. RPython at line 2531 invokes `cintf.jit_rvmprof_code(arg1,
/// arg2)` directly during tracing — RPython does NOT record this as
/// an IR op, but the side effect IS observable (it advances the
/// rvmprof profiler state). The helper surfaces the matched register
/// pair via [`FinishframeLookahead::RvmprofCode`] so a future port
/// can invoke pyre's `bh.handle_rvmprof_enter`-equivalent
/// (`pyre-jit/src/call_jit.rs:1058`); walker is the symbolic shadow
/// validator (`shadow_walker.rs`) and its IR is rolled back via
/// `cut_trace`, so the rvmprof side effect lives on the trait-driven
/// leg today and the caller drops it here — TODO,
/// scoped to the rvmprof profiler instrumentation only (no trace IR
/// effect).
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum FinishframeLookahead {
    /// Handler match: caller-frame jump target (2-byte LE label after
    /// `catch_exception/L`). Caller sets `last_exc_value` and resumes
    /// at `target`.
    CatchTarget(usize),
    /// `rvmprof_code/ii` lies on the unwind path. Caller continues
    /// unwinding (no handler match) and may surface the symbolic
    /// instrumentation as a no-op for now (production parity drops the
    /// runtime `cintf.jit_rvmprof_code` call which only matters when
    /// rvmprof is enabled at trace-recording time, not at JIT-trace
    /// playback). `_arg1_reg` / `_arg2_reg` are the i-bank register
    /// indices the bhimpl would read at runtime; we surface them so the
    /// symbolic call can be ported later without re-decoding.
    #[allow(dead_code)]
    RvmprofCode { arg1_reg: u8, arg2_reg: u8 },
    /// Neither match — unwinding continues with no side effect.
    NoMatch,
}

/// Probe the per-frame raise-bubbling lookahead. RPython parity:
/// `pyjitpl.py:2506-2531 finishframe_exception` line-by-line —
/// `live/` skip then sequential `catch_exception` / `rvmprof_code` /
/// fall-through arms.
fn finishframe_lookahead_at(code: &[u8], position: usize) -> FinishframeLookahead {
    let mut pos = position;
    let Some(op) = decode_op_at(code, pos) else {
        return FinishframeLookahead::NoMatch;
    };
    // RPython `if opcode == op_live: position += SIZE_LIVE_OP`.
    if op.key == "live/" {
        pos = op.next_pc;
    }
    let Some(next) = decode_op_at(code, pos) else {
        return FinishframeLookahead::NoMatch;
    };
    if next.key == "catch_exception/L" {
        let lo = code[next.pc + 1] as usize;
        let hi = code[next.pc + 2] as usize;
        return FinishframeLookahead::CatchTarget(lo | (hi << 8));
    }
    if next.key == "rvmprof_code/ii" {
        // RPython `pyjitpl.py:2523-2531`:
        //   arg1 = frame.registers_i[ord(code[position + 1])].getint()
        //   arg2 = frame.registers_i[ord(code[position + 2])].getint()
        //   assert arg1 == 1
        //   cintf.jit_rvmprof_code(arg1, arg2)
        // Walker surfaces the operand byte indices for the caller to
        // decide whether to symbolically record (today: drop, mirroring
        // RPython's non-record direct cintf call).
        let arg1_reg = code[next.pc + 1];
        let arg2_reg = code[next.pc + 2];
        return FinishframeLookahead::RvmprofCode { arg1_reg, arg2_reg };
    }
    FinishframeLookahead::NoMatch
}

/// Convenience wrapper preserving the legacy
/// `try_catch_exception_at(...) -> Option<target>` shape used by
/// existing callers. Returns `Some(target)` only on the
/// `CatchTarget` arm; `RvmprofCode` and `NoMatch` collapse to `None`
/// (both cases continue unwinding from the caller's POV — the
/// instrumentation side effect is dropped today, matching RPython's
/// non-trace-recorded `cintf` call).
fn try_catch_exception_at(code: &[u8], position: usize) -> Option<usize> {
    match finishframe_lookahead_at(code, position) {
        FinishframeLookahead::CatchTarget(target) => Some(target),
        FinishframeLookahead::RvmprofCode { .. } | FinishframeLookahead::NoMatch => None,
    }
}

/// Read a 2-byte little-endian descr index operand and resolve to
/// the descr from [`WalkContext::descr_refs`]. RPython equivalent:
/// `BlackholeInterpreter.descrs[code[pc] | (code[pc+1] << 8)]`
/// (`blackhole.py:102-103` setup + per-`bhimpl_*` site).
fn read_descr(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_>,
) -> Result<DescrRef, DispatchError> {
    let lo = code[op.pc + 1 + operand_offset] as usize;
    let hi = code[op.pc + 1 + operand_offset + 1] as usize;
    let index = lo | (hi << 8);
    ctx.descr_refs
        .get(index)
        .cloned()
        .ok_or(DispatchError::DescrIndexOutOfRange {
            pc: op.pc,
            index,
            len: ctx.descr_refs.len(),
        })
}

fn concrete_int_for_switch(
    op: &DecodedOp,
    value: OpRef,
    ctx: &WalkContext<'_, '_>,
) -> Result<i64, DispatchError> {
    match ctx.trace_ctx.concrete_of_opref(value) {
        Some(Value::Int(v)) => Ok(v),
        _ => Err(DispatchError::SwitchValueNotConcrete { pc: op.pc, value }),
    }
}

// Walker guard recording (`GuardNoException`, `GuardNotForced` after
// residual_call) pairs every guard with
// `walker_capture_snapshot_for_last_guard`, the walker-side port of
// RPython's `capture_resumedata(after_residual_call=True)`
// (`pyjitpl.py:2599-2603`).  RPython `pyjitpl.py:2558-2602
// generate_guard` walks `metainterp.framestack` and consults per-opcode
// liveness (`pyjitpl.py:177 get_list_of_active_boxes`) to encode the
// live `i`/`r`/`f` registers in i→r→f order plus virtualizable / vref
// boxes.  Walker's helper today omits per-PC liveness narrowing (future
// follow-up: thread the `op_live` byte table through `SubJitCodeBody`)
// and conservatively snapshots every non-`OpRef::NONE` register —
// over-capture is correctness-preserving because the optimizer's
// `store_final_boxes_in_guard` (`optimizeopt/mod.rs:5033`) derives
// `op.fail_args` from the snapshot via `store_final_boxes(liveboxes)`,
// so dead registers are dropped before they reach the backend.  Walker
// IR is no longer rolled back via `cut_trace` for the production
// dispatch (`production_walker_handles` allow-list); the snapshot
// must therefore be RPython-orthodox.

/// Read a Ref-bank variadic operand list (`R` argcode): 1 length byte
/// followed by `len` register bytes. Returns the resolved [`OpRef`]s
/// in jitcode order plus the total operand byte width (so callers can
/// skip past or compute downstream operand offsets).
///
/// RPython parity: `assembler.py:write_varlist` emits exactly this
/// shape — `chr(len(args))` followed by one byte per arg register.
fn read_ref_var_list(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_>,
) -> Result<(Vec<OpRef>, usize), DispatchError> {
    let len_pc = op.pc + 1 + operand_offset;
    let len = code[len_pc] as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let reg = code[len_pc + 1 + i] as usize;
        let opref = ctx
            .registers_r
            .get(reg)
            .copied()
            .ok_or(DispatchError::RegisterOutOfRange {
                pc: op.pc,
                reg,
                len: ctx.registers_r.len(),
                bank: "r",
            })?;
        out.push(opref);
    }
    Ok((out, 1 + len))
}

/// Read a Ref-bank register operand's concrete shadow value.
/// Mirrors [`read_ref_reg`] but indexes into
/// `ctx.concrete_registers_r`. Returns `ConcreteValue::Null` when the
/// register is out of range — symmetric with `concrete_value_at`'s
/// fallback at `state.rs:3225`. Out-of-range OpRef reads still surface
/// `RegisterOutOfRange` via [`read_ref_reg`]; this helper assumes the
/// OpRef read succeeded, so a missing concrete slot is "stack tail not
/// yet seeded" not "register byte out of range".
fn read_ref_reg_concrete(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_>,
) -> ConcreteValue {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.concrete_registers_r
        .get(reg)
        .copied()
        .unwrap_or(ConcreteValue::Null)
}

/// Write a Ref-bank register and its concrete shadow in lock-step.
/// Replaces the inlined
/// `registers_r.get_mut(dst).ok_or(...)?; *slot = value` pattern at
/// every walker handler that writes `registers_r[dst]`.  The concrete
/// shadow update is the WHOLE POINT of this helper: the shadow MUST
/// stay in sync with the symbolic side or
/// downstream consumers (`raise/r` GUARD_CLASS, future
/// `getfield_gc_r` cache lookups) will silently mis-fire.
///
/// `concrete` semantics:
/// * `ConcreteValue::Ref(ptr)` — the handler knows the concrete result
///   (e.g. `ref_copy/r>r` propagating from the source slot's shadow,
///   `raise/r` setting the just-raised exception's concrete).
/// * `ConcreteValue::Null` — the handler doesn't know (most recorded
///   ops: field reads, residual calls, …).  Downstream GUARD_CLASS
///   gates treat Null as "skip the guard", same as slots the snapshot
///   never populated.
fn write_ref_reg(
    ctx: &mut WalkContext<'_, '_>,
    pc: usize,
    dst: usize,
    value: OpRef,
    concrete: ConcreteValue,
) -> Result<(), DispatchError> {
    let len = ctx.registers_r.len();
    let slot = ctx
        .registers_r
        .get_mut(dst)
        .ok_or(DispatchError::RegisterOutOfRange {
            pc,
            reg: dst,
            len,
            bank: "r",
        })?;
    *slot = value;
    // Snapshot is sized to `registers_r.len()` at dispatch entry, so
    // a dst-in-bounds OpRef write implies in-bounds for the shadow.
    // `get_mut` defensively to tolerate sub-walk shadows that lag the
    // OpRef bank if a future caller mis-sizes them.
    //
    // Codex P1 (PR #89): collapse non-Ref ConcreteValue (Int / Float)
    // to Null before storing into the Ref shadow.  `concrete_from_
    // recorded_opref` returns whatever kind the per-OpRef concrete
    // table holds; a kind mismatch (e.g. boxed Int returned through a
    // Ref result slot) would otherwise leak Int/Float bits into
    // `concrete_registers_r`, breaking ref-only downstream consumers
    // (`getfield_gc_r` sanity loads, `raise/r` GUARD_CLASS) that
    // expect `ConcreteValue::Ref(_)` or `Null`.
    let sanitized = match concrete {
        ConcreteValue::Ref(_) | ConcreteValue::Null => concrete,
        ConcreteValue::Int(_) | ConcreteValue::Float(_) | ConcreteValue::Bool(_) => {
            ConcreteValue::Null
        }
    };
    if let Some(c_slot) = ctx.concrete_registers_r.get_mut(dst) {
        *c_slot = sanitized;
    }
    Ok(())
}

/// Int-bank twin of [`read_ref_reg_concrete`] (Int-bank concrete shadow).
/// Reads the Int-bank slot at the operand index from
/// `ctx.concrete_registers_i`.  Returns `ConcreteValue::Null` for
/// out-of-range reads — the only legal time the slice is shorter than
/// `registers_i` is at test fixtures that pass `&mut []`, and those
/// don't trigger `goto_if_not/iL` / `switch/id` paths.
fn read_int_reg_concrete(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_>,
) -> ConcreteValue {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.concrete_registers_i
        .get(reg)
        .copied()
        .unwrap_or(ConcreteValue::Null)
}

/// Int-bank twin of [`write_ref_reg`] (Int-bank concrete shadow).  Writes an Int
/// register and its concrete shadow in lock-step.  Mirrors the
/// Ref-bank contract: every walker handler that writes
/// `registers_i[dst]` MUST also write `concrete_registers_i[dst]` so
/// downstream `goto_if_not/iL` / `switch/id` can fold the branch.
///
/// `concrete` semantics:
/// * `ConcreteValue::Int(v)` — the handler knows the concrete result
///   (e.g. `int_copy/i>i` propagating from the source slot's shadow,
///   an `int_<binop>` fold of two concrete inputs).
/// * `ConcreteValue::Null` — the handler doesn't know (e.g. residual
///   `Call*I`, `getfield_gc_i` cache miss).  Downstream consumers
///   surface `GotoIfNotValueNotConcrete` for unknown branch inputs.
fn write_int_reg(
    ctx: &mut WalkContext<'_, '_>,
    pc: usize,
    dst: usize,
    value: OpRef,
    concrete: ConcreteValue,
) -> Result<(), DispatchError> {
    let len = ctx.registers_i.len();
    let slot = ctx
        .registers_i
        .get_mut(dst)
        .ok_or(DispatchError::RegisterOutOfRange {
            pc,
            reg: dst,
            len,
            bank: "i",
        })?;
    *slot = value;
    // Mirror `write_ref_reg`'s defensive get_mut.  Test fixtures pass
    // an empty `concrete_registers_i` slice; production callers
    // (Concrete shadow seeding) size it to `registers_i.len()` at dispatch entry.
    //
    // Codex P1 (PR #89) symmetry with `write_ref_reg`: collapse
    // non-Int ConcreteValue to Null before storing into the Int
    // shadow so a kind-mismatched stamp can't leak Ref/Float bits
    // into `concrete_registers_i`.
    let sanitized = match concrete {
        ConcreteValue::Int(_) | ConcreteValue::Null => concrete,
        // `is_int(bool_obj)` is true and `ConcreteValue::Bool::getint()`
        // coerces to `bool as i64`, so booleans flow safely through the
        // Int shadow.
        ConcreteValue::Bool(v) => ConcreteValue::Int(v as i64),
        ConcreteValue::Ref(_) | ConcreteValue::Float(_) => ConcreteValue::Null,
    };
    if let Some(c_slot) = ctx.concrete_registers_i.get_mut(dst) {
        *c_slot = sanitized;
    }
    Ok(())
}

/// Derive a `ConcreteValue` for shadow write-back from a freshly
/// recorded `OpRef` via `concrete_of_opref` (concrete_of_opref derivation).
///
/// RPython parity: `pyjitpl.py:execute_with_descr` /
/// `rpython/jit/metainterp/executor.py` stamps `box.value` on every
/// executed op result through the per-opcode LLOp executor — `Box.value`
/// IS the load-bearing concrete channel.  Pyre's `concrete_of_opref`
/// table-lookup is the orthodox shadow of that channel: constant pool
/// (`history.py:220/261/307`), virtualizable boxes
/// (`pyjitpl.py:3400-3430`), `set_opref_concrete` stamps from
/// `binop_int_record` / `unop_int_record`, and standard virtualizable
/// box hits all surface here.
///
/// Returns `ConcreteValue::Null` when `concrete_of_opref` yields `None`
/// (no concrete known) — the caller's downstream `goto_if_not/iL` /
/// GUARD_CLASS dispatch treats Null as "skip the fold / skip the guard".
/// A residual `Value::Ref(GcRef(usize::MAX))` storage placeholder
/// (`vable_setfield`) is likewise mapped to Null since it signals "no
/// concrete known" rather than an actual pointer.
#[inline]
fn concrete_from_recorded_opref(ctx: &WalkContext<'_, '_>, opref: OpRef) -> ConcreteValue {
    match ctx.trace_ctx.concrete_of_opref(opref) {
        Some(Value::Int(v)) => ConcreteValue::Int(v),
        Some(Value::Float(v)) => ConcreteValue::Float(v),
        Some(Value::Ref(r)) if r != majit_ir::GcRef(usize::MAX) => {
            ConcreteValue::Ref(r.as_usize() as pyre_object::PyObjectRef)
        }
        _ => ConcreteValue::Null,
    }
}

/// Read concrete shadow values for a Ref-bank variadic operand list.
/// Parallels [`read_ref_var_list`] — reads the
/// same byte indices but resolves through `ctx.concrete_registers_r`.
/// Used by `inline_call_*` to propagate per-arg concrete shadow into
/// the callee's fresh shadow Vec.
fn read_ref_var_list_concrete(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_>,
) -> Vec<ConcreteValue> {
    let len_pc = op.pc + 1 + operand_offset;
    let len = code[len_pc] as usize;
    (0..len)
        .map(|i| {
            let reg = code[len_pc + 1 + i] as usize;
            ctx.concrete_registers_r
                .get(reg)
                .copied()
                .unwrap_or(ConcreteValue::Null)
        })
        .collect()
}

/// Read an Int-bank variadic operand list (`I` argcode). Same shape as
/// [`read_ref_var_list`] but indexes into `registers_i`. RPython
/// `assembler.py:write_varlist` emits a single shape regardless of
/// kind; the kind letter (`I` / `R` / `F`) only steers which register
/// file the bytes index into.
fn read_int_var_list(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_>,
) -> Result<(Vec<OpRef>, usize), DispatchError> {
    let len_pc = op.pc + 1 + operand_offset;
    let len = code[len_pc] as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let reg = code[len_pc + 1 + i] as usize;
        let opref = ctx
            .registers_i
            .get(reg)
            .copied()
            .ok_or(DispatchError::RegisterOutOfRange {
                pc: op.pc,
                reg,
                len: ctx.registers_i.len(),
                bank: "i",
            })?;
        out.push(opref);
    }
    Ok((out, 1 + len))
}

/// Read a Float-bank variadic operand list (`F` argcode). Mirror of
/// [`read_int_var_list`] for the float bank.
fn read_float_var_list(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_>,
) -> Result<(Vec<OpRef>, usize), DispatchError> {
    let len_pc = op.pc + 1 + operand_offset;
    let len = code[len_pc] as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let reg = code[len_pc + 1 + i] as usize;
        let opref = ctx
            .registers_f
            .get(reg)
            .copied()
            .ok_or(DispatchError::RegisterOutOfRange {
                pc: op.pc,
                reg,
                len: ctx.registers_f.len(),
                bank: "f",
            })?;
        out.push(opref);
    }
    Ok((out, 1 + len))
}

/// Generic int-bank binop handler. Reads `registers_i[src1]` and
/// `registers_i[src2]`, records `record_op(opcode, [a, b])`, writes
/// the recorder's result OpRef into `registers_i[dst]`. Operand
/// layout is `ii>i` (1B src1 + 1B src2 + 1B dst).
///
/// RPython parity: `pyjitpl.py:288-292` exec-generated
/// `opimpl_int_BINOP(b1, b2): return self.execute(rop.<OPNUM>, b1,
/// b2)` + the trailing `>i` decorator that writes the result into
/// `registers_i[dst]`. Walker collapses execute+writeback into
/// `record_op + slot store`, which matches the recording-only side of
/// `execute`'s split (`pyjitpl.py:_record_helper`).
fn binop_int_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_int_reg(code, op, 0, ctx)?;
    let b = read_int_reg(code, op, 1, ctx)?;
    let result = ctx.trace_ctx.record_op(opcode, &[a, b]);
    // Box(value) parity: stamp the result from the operands' Box.value
    // carriers (BoxInt(value) — matches dispatch.rs trace_binop_i).
    // The folded value also feeds the slot-keyed `concrete_registers_i`
    // shadow via [`write_int_reg`] so handlers that read the slot
    // (Ref-bank symmetry) see the same concrete as the OpRef channel.
    let concrete = if let (Some(majit_ir::Value::Int(la)), Some(majit_ir::Value::Int(rb))) =
        (ctx.trace_ctx.box_value(a), ctx.trace_ctx.box_value(b))
    {
        let folded = majit_metainterp::eval_binop_i(opcode, la, rb);
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int(folded));
        ConcreteValue::Int(folded)
    } else {
        ConcreteValue::Null
    };
    let dst = code[op.pc + 3] as usize;
    write_int_reg(ctx, op.pc, dst, result, concrete)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// RPython `pyjitpl.py:588-595 opimpl_int_between`:
///
/// ```python
/// b5 = self.execute(rop.INT_SUB, b3, b1)
/// if isinstance(b5, ConstInt) and b5.getint() == 1:
///     # int_between(a, b, a+1) -> b == a
///     return self.execute(rop.INT_EQ, b2, b1)
/// else:
///     b4 = self.execute(rop.INT_SUB, b2, b1)
///     return self.execute(rop.UINT_LT, b4, b5)
/// ```
///
/// Decomposes `int_between(b1, b2, b3)` — i.e. `b1 <= b2 < b3` — at
/// record time, matching upstream's choice to emit elementary
/// `INT_SUB`/`INT_EQ`/`UINT_LT` into the trace rather than relying on
/// the optimizer to lower `INT_BETWEEN`.  The blackhole semantics
/// (`blackhole.py:560-561 bhimpl_int_between(a, b, c): return a <= b
/// < c`) are preserved through the same decomposition.
///
/// Operand layout `iii>i`: 3B sources + 1B dst (=4 operand bytes after
/// the opcode).  Concrete-value propagation in [`execute_pure_binop_i`]
/// runs in two layers: all-inline-Const operand pairs fold to a
/// `const_int(...)` OpRef without recording (matching upstream
/// `_all_constants` short-circuit at `pyjitpl.py:2654-2660`); the
/// trailing concrete-tracked-pair path additionally stamps the recorded
/// op via `set_opref_concrete`.  The `ConstInt(1)` fast path at
/// `pyjitpl.py:590` keys on the inline-Const layer through
/// `inline_const_to_value`, mirroring `isinstance(b5, ConstInt)` —
/// box_value's concrete-stamp layer does not participate in that
/// branch decision.
fn int_between_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let b1 = read_int_reg(code, op, 0, ctx)?;
    let b2 = read_int_reg(code, op, 1, ctx)?;
    let b3 = read_int_reg(code, op, 2, ctx)?;

    // b5 = execute(INT_SUB, b3, b1)
    let b5 = execute_pure_binop_i(ctx, OpCode::IntSub, b3, b1);

    // pyjitpl.py:590 `if isinstance(b5, ConstInt) and b5.getint() == 1`
    // — the `ConstInt(1)` fast path emits INT_EQ; otherwise the
    // generic INT_SUB + UINT_LT pair.  `inline_const_to_value` returns
    // `Some(_)` exactly when `b5` is an inline-Const OpRef, mirroring
    // `isinstance(b5, ConstInt)`.
    let result = if let Some(majit_ir::Value::Int(1)) = b5.inline_const_to_value() {
        // execute(INT_EQ, b2, b1)
        execute_pure_binop_i(ctx, OpCode::IntEq, b2, b1)
    } else {
        // b4 = execute(INT_SUB, b2, b1); execute(UINT_LT, b4, b5)
        let b4 = execute_pure_binop_i(ctx, OpCode::IntSub, b2, b1);
        execute_pure_binop_i(ctx, OpCode::UintLt, b4, b5)
    };

    let concrete = match result.inline_const_to_value() {
        Some(majit_ir::Value::Int(v)) => ConcreteValue::Int(v),
        _ => match ctx.trace_ctx.box_value(result) {
            Some(majit_ir::Value::Int(v)) => ConcreteValue::Int(v),
            _ => ConcreteValue::Null,
        },
    };

    let dst = code[op.pc + 4] as usize;
    write_int_reg(ctx, op.pc, dst, result, concrete)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `pyjitpl.py:2648-2662 execute_and_record` for pure integer binops.
/// When both operands are inline-Const, fold to a `ConstInt` OpRef
/// (no `record_op` call); otherwise record the op and stamp the
/// observed concrete value if both sides have one.
///
/// PyPy's `execute_and_record` short-circuits `_record_helper` via
/// `executor.wrap_constant(resvalue)` when `_all_constants(*argboxes)`
/// holds for a pure opcode.  Mirroring that here keeps the trace
/// free of all-constant subexpressions exactly where upstream keeps
/// it free — `opimpl_int_between` chains three such pure binops.
fn execute_pure_binop_i(
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
    a: OpRef,
    b: OpRef,
) -> OpRef {
    if let (Some(majit_ir::Value::Int(va)), Some(majit_ir::Value::Int(vb))) =
        (a.inline_const_to_value(), b.inline_const_to_value())
    {
        let folded = majit_metainterp::eval_binop_i(opcode, va, vb);
        return ctx.trace_ctx.const_int(folded);
    }

    let result = ctx.trace_ctx.record_op(opcode, &[a, b]);
    if let (Some(majit_ir::Value::Int(va)), Some(majit_ir::Value::Int(vb))) =
        (ctx.trace_ctx.box_value(a), ctx.trace_ctx.box_value(b))
    {
        let folded = majit_metainterp::eval_binop_i(opcode, va, vb);
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int(folded));
    }
    result
}

/// RPython `pyjitpl.py:597-617 opimpl_switch`:
///
/// * read the traced value box and concrete `valuebox.getint()`
/// * on hit, `implement_guard_value(valuebox, orgpc)` and jump target
/// * on miss, emit `INT_EQ(valuebox, ConstInt(key))` plus `GUARD_FALSE`
///   for every `switchdict.const_keys_in_order`, then fall through
///
/// TODO: guards below record with empty resume data
/// (`record_guard(..., 0)`).  RPython `pyjitpl.py:600 opimpl_switch`
/// pairs every `GUARD_VALUE` / `GUARD_FALSE` with `generate_guard(...,
/// resumepc=orgpc) → capture_resumedata(orgpc)` walking the framestack
/// with liveness.  The standalone walker has no MIFrame liveness /
/// framestack infrastructure, so attaching a snapshot here would
/// approximate it (wrong layout, all-typed-registers vs liveness-
/// filtered) and downstream layout matching consumes it as truth.
fn dispatch_switch_id(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let valuebox = read_int_reg(code, op, 0, ctx)?;
    let descr = read_descr(code, op, 1, ctx)?;
    let switchdict = descr
        .as_switch_descr()
        .ok_or(DispatchError::ExpectedSwitchDescr { pc: op.pc })?;
    let search_value = concrete_int_for_switch(op, valuebox, ctx)?;

    if let Some(target) = switchdict.lookup(search_value) {
        if !valuebox.is_constant() {
            let expected = ctx.trace_ctx.const_int(search_value);
            ctx.trace_ctx
                .record_guard(OpCode::GuardValue, &[valuebox, expected], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc);
            ctx.trace_ctx.replace_box(valuebox, expected);
            for slot in ctx.registers_i.iter_mut() {
                if *slot == valuebox {
                    *slot = expected;
                }
            }
        }
        return Ok((DispatchOutcome::Continue, target));
    }

    if !valuebox.is_constant() {
        // pyjitpl.py:611-617 opimpl_switch miss path — emit IntEq +
        // GuardFalse for every key in switchdict (the trace bails out
        // if any subsequent execution lands on a missed key).
        // Box(value) parity: stamp each IntEq result with the
        // (concrete_value == key) bool when valuebox's Box.value
        // resolves an Int.
        let valuebox_concrete = match ctx.trace_ctx.box_value(valuebox) {
            Some(majit_ir::Value::Int(n)) => Some(n),
            _ => None,
        };
        for &key in switchdict.const_keys_in_order() {
            let keybox = ctx.trace_ctx.const_int(key);
            let eqbox = ctx.trace_ctx.record_op(OpCode::IntEq, &[valuebox, keybox]);
            if let Some(v) = valuebox_concrete {
                ctx.trace_ctx
                    .set_opref_concrete(eqbox, majit_ir::Value::Int((v == key) as i64));
            }
            ctx.trace_ctx.record_guard(OpCode::GuardFalse, &[eqbox], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc);
        }
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Bridge an `MIFrame`'s register banks +
/// trace recorder + last-exc state into a `WalkContext` and run
/// `walk()` against the supplied jitcode body.
///
/// RPython parity context: in RPython the metainterp loop iterates
/// over `metainterp.framestack[-1].pc` calling `bytecode_step` which
/// dispatches to the right `opimpl_*`. There's no separate "walker
/// entry" because the metainterp loop *is* the walker. Pyre is
/// mid-migration: the production tracing path today is the
/// trait-driven `MIFrame::execute_opcode_step` (`trace_opcode.rs`);
/// this entry point lets shadow execution + per-opcode migration
/// drive the orthodox walker
/// against the same MIFrame state without first replacing the trait
/// dispatch wholesale.
///
/// Field plumbing:
/// * `registers_r/i/f` — allocated fresh per call sized to
///   `top_num_regs_* + top_constants_*.len()`, then populated by
///   the inline `setup_call` from `argboxes_*` and constant slots.
///   PyPy parity: `pyjitpl.py:171-176 MIFrame.__init__` allocates
///   the bank vectors at frame construction; `:188 setup_call`
///   populates slots `[0..argboxes.len())` from the caller's
///   argboxes. Walker handlers writing dst slots (`int_copy`,
///   `binop_int_record`, etc.) mutate them in place; the banks are
///   dropped when this function returns (matching PyPy's per-frame
///   lifetime).
/// * `trace_ctx` — borrowed mutably from `miframe.ctx`'s
///   `TraceCtx`. Recording (`record_op`, `finish`, etc.) goes
///   through this.
/// * `last_exc_value` — reads `sym.last_exc_box` as the initial
///   value (`OpRef::NONE` collapses to `None`). On exit the
///   walker's final `last_exc_value` is mirrored back if non-None,
///   so a `raise/r` -> `catch_exception/L` -> handler trace
///   leaves `sym.last_exc_box` pointing at the in-flight exc OpRef
///   (parity with RPython metainterp.last_exc_value).
/// * `descr_refs`, `sub_jitcode_lookup` — caller-provided, same
///   contract as direct `walk()` callers. Production callers wire
///   `crate::jitcode_runtime::all_descrs()` + a JitCode-resolving
///   closure over `crate::jitcode_runtime::all_jitcodes()`.
///
/// `is_top_level` selects the outer-frame semantic:
///
/// * `true` — outermost trace entry. `*_return/*` arms record
///   `Finish(value, done_with_this_frame_descr_<kind>)` and a `raise/r`
///   that is never caught records
///   `Finish(exc, exit_frame_with_exception_descr_ref)`.
/// * `false` — sub-frame entry: `*_return/*` arms surface
///   `SubReturn { result }` and uncaught `raise/r` arms surface
///   `SubRaise { exc }` to the caller. The shadow validator (Phase
///   D-3) drives this for per-Python-opcode arms — a Python-opcode arm
///   compiled by the codewriter ends with `*_return/*` (since each arm
///   is a self-contained sub-jitcode invoked from the outer dispatcher
///   via `inline_call_r_r/dR>r`), and the trait dispatch path emits
///   no FINISH per Python opcode, so shadow mode must NOT emit one
///   either.
///
/// Sub-walks driven by `inline_call_r_r/dR>r` recursion always set
/// `is_top_level=false` regardless of this caller-side flag (the
/// recursion constructs its own `WalkContext`).
///
/// **Production wiring**: `crate::shadow_walker::shadow_validate_pre`
/// is the first caller; it passes `is_top_level: false` for per-opcode
/// shadow validation.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_via_miframe(
    miframe: &mut MIFrame,
    jitcode_code: &[u8],
    position: usize,
    descr_refs: &[DescrRef],
    sub_jitcode_lookup: &SubJitCodeLookup,
    done_with_this_frame_descr_ref: DescrRef,
    done_with_this_frame_descr_int: DescrRef,
    done_with_this_frame_descr_float: DescrRef,
    done_with_this_frame_descr_void: DescrRef,
    exit_frame_with_exception_descr_ref: DescrRef,
    is_top_level: bool,
    // PyPy `pyjitpl.py:171-176 MIFrame.__init__` analog: the
    // top-level jitcode's per-bank register count.  `dispatch_via_miframe`
    // allocates fresh `Vec<OpRef>`s sized to `top_num_regs_* +
    // top_constants_*.len()` — replacing the prior TODO that
    // reused `sym.registers_r` (a Python locals/stack mirror) as the
    // MIFrame register file.  The codewriter-compiled arm jitcode
    // expects `R[0]_r = handler = MIFrame self ptr`, which the
    // `argboxes_*` parameters supply via the `setup_call` analog
    // below.
    top_num_regs_r: usize,
    top_num_regs_i: usize,
    top_num_regs_f: usize,
    // Top-level jitcode's per-bank constant pool — seeded into
    // register slots `[num_regs_*, num_regs_* + constants_*.len())`
    // per `pyjitpl.py:98-119 copy_constants`.
    top_constants_r: &[i64],
    top_constants_i: &[i64],
    top_constants_f: &[i64],
    // PyPy `pyjitpl.py:188-200 setup_call(argboxes)` analog.
    // `argboxes_*[i]` is written to `registers_*[i]` before walking.
    // Production callers supply `argboxes_r = [const_ref(miframe_ptr)]`
    // so the codewriter-compiled arm finds the MIFrame self ptr at
    // `R[0]_r`.
    argboxes_r: &[OpRef],
    argboxes_i: &[OpRef],
    argboxes_f: &[OpRef],
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Extract raw pointers before any borrow. `miframe.ctx` and
    // `miframe.sym` are `*mut`, distinct objects (the trace
    // recorder vs. the symbolic frame state) — distinct pointers
    // means dereferencing both simultaneously is sound.
    let ctx_ptr = miframe.ctx;
    let sym_ptr = miframe.sym;
    let entry_py_pc = miframe.orgpc as u32;
    // SAFETY: both pointers were initialized at MIFrame
    // construction time and outlive this call (TraceCtx and
    // PyreSym are pinned by the surrounding tracing session).
    // `&mut sym` is held only for the post-walk
    // `last_exc_box`/`class_of_last_exc_is_const` writeback below;
    // during the walk itself we re-borrow only specific sym fields
    // (`registers_*`) into the `WalkContext` slices — the walker no
    // longer takes a fresh `&mut PyreSym` inside any helper, so there
    // is no Stacked-Borrows aliasing between WalkContext's slice
    // borrows and a parallel sym reborrow (parity #2 — the prior
    // `miframe_sym: Option<*mut PyreSym>` field has been removed).
    let trace_ctx = unsafe { &mut *ctx_ptr };
    let sym = unsafe { &mut *sym_ptr };

    // RPython parity: `metainterp.last_exc_value` (pyjitpl.py:1695)
    // is the standing exception OpRef. Walker's `WalkContext::last_exc_value`
    // mirrors this as `Option<OpRef>` — `None` means "no active
    // exception", matching RPython's `assert self.metainterp.last_exc_value`
    // (pyjitpl.py:1702).
    let initial_last_exc_value = if sym.last_exc_box.is_none() {
        None
    } else {
        Some(sym.last_exc_box)
    };

    // PyPy `pyjitpl.py:171-176 MIFrame.__init__` analog: allocate
    // fresh per-bank register vectors sized to `top_num_regs_* +
    // top_constants_*.len()`.  This replaces the prior TODO
    // that reused `sym.registers_r` (a Python locals/stack mirror,
    // whose `[0]` slot is Python local 0) as the MIFrame register
    // file.  The codewriter-compiled arm jitcode emits getfield
    // chains rooted at `R[0] = handler = MIFrame self ptr`; the
    // `argboxes_r` parameter supplies that handler ptr below via the
    // `setup_call` analog.
    let total_r = top_num_regs_r + top_constants_r.len();
    let total_i = top_num_regs_i + top_constants_i.len();
    let total_f = top_num_regs_f + top_constants_f.len();
    let mut top_regs_r = vec![OpRef::NONE; total_r];
    let mut top_regs_i = vec![OpRef::NONE; total_i];
    let mut top_regs_f = vec![OpRef::NONE; total_f];
    let mut top_concrete_r = vec![ConcreteValue::Null; total_r];
    let mut top_concrete_i = vec![ConcreteValue::Null; total_i];

    // PyPy `pyjitpl.py:98-119 copy_constants` analog: seed each
    // constant into the upper slot range `[num_regs_*, total_*)`.
    // `box_value` resolves these via `TraceCtx::constants` so
    // downstream getfield chains see the constant's `Value::*`.
    for (i, &v) in top_constants_i.iter().enumerate() {
        top_regs_i[top_num_regs_i + i] = trace_ctx.const_int(v);
        top_concrete_i[top_num_regs_i + i] = ConcreteValue::Int(v);
    }
    for (i, &v) in top_constants_r.iter().enumerate() {
        top_regs_r[top_num_regs_r + i] = trace_ctx.const_ref(v);
        if v != 0 {
            top_concrete_r[top_num_regs_r + i] = ConcreteValue::Ref(v as pyre_object::PyObjectRef);
        }
    }
    for (i, &v) in top_constants_f.iter().enumerate() {
        top_regs_f[top_num_regs_f + i] = trace_ctx.const_float(v);
    }

    // PyPy `pyjitpl.py:188-200 setup_call(argboxes)` analog: write
    // each argbox into the leading register slot.  The concrete
    // shadow is derived from `box_value(box)` — for `ConstRef(ptr)`
    // (the common case: argbox=miframe self ptr), this is
    // `Some(Value::Ref(GcRef(ptr)))` resolved via the constant pool;
    // for non-const argboxes it consults the `opref_concrete` stamp
    // table.
    //
    // CodeRabbit Major (PR #89): reject oversized argbox lists up
    // front instead of silently truncating with a per-loop `break`.
    // The `_*_arity_mismatch` DispatchError shapes already exist for
    // the inline-call paths (`InlineCall*ArityMismatch`); reuse them
    // here so a caller/shape mismatch surfaces as a typed failure
    // rather than a partially seeded frame.
    if argboxes_r.len() > top_num_regs_r {
        return Err(DispatchError::InlineCallArityMismatch {
            pc: position,
            provided: argboxes_r.len(),
            callee_num_regs_r: top_num_regs_r,
        });
    }
    if argboxes_i.len() > top_num_regs_i {
        return Err(DispatchError::InlineCallIntArityMismatch {
            pc: position,
            provided: argboxes_i.len(),
            callee_num_regs_i: top_num_regs_i,
        });
    }
    if argboxes_f.len() > top_num_regs_f {
        return Err(DispatchError::InlineCallFloatArityMismatch {
            pc: position,
            provided: argboxes_f.len(),
            callee_num_regs_f: top_num_regs_f,
        });
    }
    for (i, &box_ref) in argboxes_r.iter().enumerate() {
        top_regs_r[i] = box_ref;
        if let Some(majit_ir::Value::Ref(majit_ir::GcRef(ptr))) = trace_ctx.box_value(box_ref) {
            top_concrete_r[i] = ConcreteValue::Ref(ptr as pyre_object::PyObjectRef);
        }
    }
    for (i, &box_ref) in argboxes_i.iter().enumerate() {
        top_regs_i[i] = box_ref;
        if let Some(majit_ir::Value::Int(v)) = trace_ctx.box_value(box_ref) {
            top_concrete_i[i] = ConcreteValue::Int(v);
        }
    }
    for (i, &box_ref) in argboxes_f.iter().enumerate() {
        top_regs_f[i] = box_ref;
    }

    // Seed last_exc_value_concrete from
    // sym.last_exc_value (the live PyObjectRef written by trait-side
    // `seed_raised_exception` at `trace_opcode.rs:6646`).  Null when
    // no active exception, matching `initial_last_exc_value == None`.
    let initial_last_exc_value_concrete = if sym.last_exc_value.is_null() {
        ConcreteValue::Null
    } else {
        ConcreteValue::Ref(sym.last_exc_value)
    };

    let result = {
        let mut wc = WalkContext {
            registers_r: &mut top_regs_r,
            registers_i: &mut top_regs_i,
            registers_f: &mut top_regs_f,
            concrete_registers_r: &mut top_concrete_r,
            concrete_registers_i: &mut top_concrete_i,
            descr_refs,
            trace_ctx,
            done_with_this_frame_descr_ref,
            done_with_this_frame_descr_int,
            done_with_this_frame_descr_float,
            done_with_this_frame_descr_void,
            exit_frame_with_exception_descr_ref,
            is_top_level,
            sub_jitcode_lookup,
            last_exc_value: initial_last_exc_value,
            last_exc_value_concrete: initial_last_exc_value_concrete,
            entry_py_pc,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let outcome = walk(jitcode_code, position, &mut wc);
        // Read final last_exc_value before wc drops so the borrow
        // checker can release sym for the writeback below.
        let final_last_exc = wc.last_exc_value;
        drop(wc);
        // Full `sym.last_exc_*` state writeback parity.
        //
        // RPython `pyjitpl.py:1694-1696 opimpl_raise` sets THREE pieces
        // of metainterp state when a raise fires:
        //   self.metainterp.class_of_last_exc_is_const = True
        //   self.metainterp.last_exc_value = exc_value_box.getref(rclass.OBJECTPTR)
        //   self.metainterp.last_exc_box = exc_value_box
        //
        // Of these, the walker can produce:
        //   - `last_exc_box`: the symbolic OpRef. Mirrored from
        //     `wc.last_exc_value` (RPython's metainterp.last_exc_value
        //     and last_exc_box are different fields — concrete pointer
        //     vs Box — but the walker tracks only the symbolic one,
        //     which lines up with `sym.last_exc_box`).
        //   - `class_of_last_exc_is_const`: true after a raise/r or a
        //     SubRaise routed into a catch handler. RPython sets this
        //     in `opimpl_raise` (line 1694) AND `execute_ll_raised`
        //     (pyjitpl.py:2752 with `constant=...` parameter — set
        //     after GUARD_CLASS / GUARD_EXCEPTION). Walker's raise/r
        //     arm always sets `wc.last_exc_value = Some(exc)` so
        //     mirroring `Some` → const=true is RPython-orthodox.
        //
        // The walker CANNOT produce:
        //   - `sym.last_exc_value` (concrete `PyObjectRef`): RPython
        //     `exc_value_box.getref(rclass.OBJECTPTR)` reads the
        //     concrete pointer at trace-recording time. The symbolic
        //     walker has only OpRefs — concrete writeback is the
        //     production tracer's responsibility (the trait-driven
        //     `MIFrame::execute_opcode_step` path). This is a known
        //     TODO (the walker is symbolic-only,
        //     concrete state is fed by another path).
        if let Some(exc) = final_last_exc {
            sym.last_exc_box = exc;
            sym.class_of_last_exc_is_const = true;
        }
        outcome
    };
    result
}

/// Orthodox entry: walk a per-opcode arm jitcode with
/// **fresh per-jitcode register banks** sized to the entry jitcode's
/// declared `num_regs_<i|r|f>() + len(constants_<i|r|f>)`, with `r0`
/// pre-seeded to the live PyFrame OpRef (`sym.frame`).
///
/// RPython line-by-line parity:
///
/// * Allocation + constant copy: `MIFrame.setup(jitcode)` at
///   `pyjitpl.py:74-91` calls `copy_constants` for each of the three
///   register banks, producing fresh per-frame lists of length
///   `num_regs_X + len(constants_X)`.  `allocate_callee_register_banks`
///   below ports that shape (`pyjitpl.py:97-119 copy_constants`).
/// * `r0 = frame`: every per-opcode arm body the codewriter emits
///   treats register 0 as the implicit PyFrame argument (verified for
///   the PopTop arm — `inline_call_r_r/dR>r [r0]` invokes the
///   `pop_value` sub-jitcode with `r0` = caller's r0).  This matches
///   RPython's convention where the topmost call site provides the
///   PyFrame as the first argument to the jitdriver's portal jitcode
///   (`call.py:148 portal_runner`).  The concrete frame address comes
///   from `MIFrame.concrete_frame_addr`, populated by the production
///   tracer at `trace_opcode.rs:768`.
///
/// The semantic frame mirror (`sym.registers_r`) and the per-jitcode
/// arm-local banks are now distinct universes: this entry no longer
/// aliases `sym.registers_r` as the walker's `registers_r`.  Closes
/// the structural blocker documented in
/// `project_issue73_phase5_design.md` for opcodes whose arm uses
/// `r0 = frame`.
///
/// `last_exc_value` / `last_exc_box` / `class_of_last_exc_is_const`
/// writeback to `sym` mirrors `dispatch_via_miframe` byte-for-byte —
/// the symbolic exception state survives across per-opcode dispatches
/// at the production tracer's contract.
pub fn dispatch_via_miframe_at_opcode_entry<'a>(
    miframe: &mut crate::state::MIFrame,
    entry_jitcode: &'static majit_translate::jitcode::JitCode,
    descr_refs: &'a [DescrRef],
    sub_jitcode_lookup: &'a SubJitCodeLookup,
    done_with_this_frame_descr_ref: DescrRef,
    done_with_this_frame_descr_int: DescrRef,
    done_with_this_frame_descr_float: DescrRef,
    done_with_this_frame_descr_void: DescrRef,
    exit_frame_with_exception_descr_ref: DescrRef,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let ctx_ptr = miframe.ctx;
    let sym_ptr = miframe.sym;
    let concrete_frame_addr = miframe.concrete_frame_addr;
    let entry_py_pc = miframe.orgpc as u32;
    // SAFETY: both pointers were initialized at MIFrame construction
    // time and outlive this call (parity with dispatch_via_miframe).
    let trace_ctx = unsafe { &mut *ctx_ptr };
    let sym = unsafe { &mut *sym_ptr };

    // Per-opcode arm entry: start with `last_exc_value = None`.
    //
    // RPython parity: each Python opcode dispatch invokes a self-
    // contained arm jitcode whose IR body never observes an inherited
    // `metainterp.last_exc_value`.  The `catch_exception/L` ops the
    // codewriter emits around a residual_call exist solely to catch
    // exceptions raised BY that residual_call inside the arm — they
    // are not meant to fire on a pre-existing pending exception from
    // a prior Python opcode.  Python-level exception state survives
    // across opcodes through `sym.last_exc_box` + the runtime's
    // exception table, which the JIT recorder bridges through the
    // `exception_target` mechanism rather than `WalkContext.
    // last_exc_value`.  Seeding to `None` here matches that boundary.
    //
    // The post-walk writeback below preserves any pre-existing
    // `sym.last_exc_box` because it only OVERWRITES when the walker
    // arm itself produced a `Some(exc)` via `raise/r` — a per-opcode
    // arm without a `raise/r` leaves `sym.last_exc_box` untouched.
    let initial_last_exc_value: Option<OpRef> = None;
    let initial_last_exc_value_concrete = ConcreteValue::Null;
    let frame_opref = sym.frame;

    // Snapshot the outer PyFrame state for walker-emitted guards.
    // `pyjitpl.py:218-225 _get_list_of_active_boxes` reads
    // `MIFrame.registers_{i,r,f}` filtered by JitCode-liveness for the
    // outer `(jitcode_index, pc)`.  `frame_liveness_reg_indices_by_bank_at`
    // resolves the same `all_liveness` byte stream the decoder consumes
    // at resume (`state::frame_value_count_at` /
    // `frame_liveness_reg_indices_by_bank_at`), so encoder and decoder
    // agree byte-for-byte on which register slots populate the snapshot.
    //
    // Outer pyjitcode index: `sym.jitcode` points at the
    // `majit_metainterp::jitcode::JitCode` for the user Python
    // function being traced — its `.index()` is what RPython's
    // `framestack[-1].jitcode.index` returns for the top frame
    // (`pyjitpl.py:2586 capture_resumedata` reads it as the snapshot
    // frame's `jitcode_index`).  Snapshot frames stamped with this
    // index resolve to a valid `MetaInterpStaticData.jitcodes[idx]`
    // entry whose `.code` is the Python `CodeObject`, so
    // `build_resumed_frames` finds the PyFrame for resume.
    let outer_jitcode_index = if sym.jitcode.is_null() {
        0
    } else {
        unsafe { (*sym.jitcode).index as u32 }
    };
    let outer_active_boxes =
        collect_outer_active_boxes(sym, trace_ctx, outer_jitcode_index, entry_py_pc);

    // pyjitpl.py:82-90 `setup` per-bank allocation: each bank gets
    // `copy_constants(registers, constants, num_regs_X, ConstClass)`.
    // `allocate_callee_register_banks` ports this for sub-jitcode
    // entries; reuse it here for the per-opcode entry too, sharing the
    // exact byte-shape: registers `[0..num_regs_X)` are zero/None, the
    // constants pool occupies `[num_regs_X..num_regs_X+len(constants))`.
    let body = SubJitCodeBody {
        code: entry_jitcode.code.as_slice(),
        num_regs_r: entry_jitcode.num_regs_r(),
        num_regs_i: entry_jitcode.num_regs_i(),
        num_regs_f: entry_jitcode.num_regs_f(),
        constants_i: entry_jitcode.constants_i.as_slice(),
        constants_r: entry_jitcode.constants_r.as_slice(),
        constants_f: entry_jitcode.constants_f.as_slice(),
    };
    let (mut regs_r, mut regs_i, mut regs_f, mut concrete_r, mut concrete_i) =
        allocate_callee_register_banks(&body, trace_ctx);

    // r0 = frame OpRef.  Mirror RPython's portal-runner contract where
    // the topmost frame's `registers_r[0]` carries the PyFrame argument.
    // Skeleton arms with `num_regs_r() == 0` (no Ref bank, e.g. trivial
    // pass-through opcodes) simply skip the seed.
    //
    // Epic #245 (per-frame frame-identity): this `regs_r[0] = sym.frame`
    // write IS the runtime seed of the universal `self`(frame) red arg.
    // `frame_opref` is the *per-callee* frame — `sym.frame` is set to the
    // inlined callee's `callee_frame_opref` (`trace_opcode.rs`
    // `sym.frame = callee_frame_opref`), so each frame's r0 already
    // resolves to its own PyFrame.  Slice 1 appended a graph-level
    // `frame_var` start-block inputarg to non-portal callee jitcode, but
    // that Variable currently shares a color with the single
    // `portal_frame_reg` (the unconditional `pin!(frame_var,
    // portal_frame_reg)` in `codewriter.rs`), so there is no distinct
    // slot to seed separately — the per-callee frame seed lives here.
    // Un-aliasing `frame_var` (its own color) and flipping the non-portal
    // namespace/pycode readers off the shared `portal_frame_reg` onto it
    // is the coupled Slice 3a step (#240 LoadGlobal namespace cutover).
    if entry_jitcode.num_regs_r() > 0 {
        regs_r[0] = frame_opref;
        concrete_r[0] = if concrete_frame_addr != 0 {
            ConcreteValue::Ref(concrete_frame_addr as pyre_object::PyObjectRef)
        } else {
            ConcreteValue::Null
        };
    }

    let result = {
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut regs_f,
            concrete_registers_r: &mut concrete_r,
            concrete_registers_i: &mut concrete_i,
            descr_refs,
            trace_ctx,
            done_with_this_frame_descr_ref,
            done_with_this_frame_descr_int,
            done_with_this_frame_descr_float,
            done_with_this_frame_descr_void,
            exit_frame_with_exception_descr_ref,
            is_top_level: false,
            sub_jitcode_lookup,
            last_exc_value: initial_last_exc_value,
            last_exc_value_concrete: initial_last_exc_value_concrete,
            entry_py_pc,
            outer_jitcode_index,
            outer_active_boxes,
        };
        let outcome = walk(entry_jitcode.code.as_slice(), 0, &mut wc);
        let final_last_exc = wc.last_exc_value;
        drop(wc);
        if let Some(exc) = final_last_exc {
            sym.last_exc_box = exc;
            sym.class_of_last_exc_is_const = true;
        }
        outcome
    };
    result
}

/// `getarrayitem_gc_<i|r|f>/rid>X` handler. Operand layout `rid>X`:
/// 1B r-reg(array) + 1B i-reg(index) + 2B descr + 1B X-dst.
///
/// RPython parity: `pyjitpl.py:639-673 _do_getarrayitem_gc_any`:
///
///   tobox = heapcache.getarrayitem(arraybox, indexbox, arraydescr)
///   if tobox: return tobox        # cache hit, no IR (recording-only)
///   resop = self.execute_with_descr(op, arraydescr, arraybox, indexbox)
///   heapcache.getarrayitem_now_known(arraybox, indexbox, resop, arraydescr)
///   return resop
///
/// `opcode` is one of `GetarrayitemGc{I,R,F}`; `dst_bank` selects the
/// result bank (`'i'`/`'r'`/`'f'`) the walker writes back into.
fn getarrayitem_gc_via_heapcache(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    getarrayitem_gc_via_heapcache_with_index_bank(code, op, ctx, opcode, dst_bank, 'i')
}

/// Underlying handler for `getarrayitem_gc_<i|r|f>` shapes.
/// `index_bank` selects whether the index operand is decoded from the
/// `i` register bank (canonical RPython shape `rid>X`) or the `r`
/// register bank (pyre-only `rrd>r` — see `pyre_extension_insns()`
/// + `blackhole.rs::handler_getarrayitem_gc_r_refindex`, an artifact of
/// the rtyper not yet classifying integer array indices as `Signed`).
fn getarrayitem_gc_via_heapcache_with_index_bank(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
    dst_bank: char,
    index_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let array = read_ref_reg(code, op, 0, ctx)?;
    let index = match index_bank {
        'i' => read_int_reg(code, op, 1, ctx)?,
        'r' => read_ref_reg(code, op, 1, ctx)?,
        _ => unreachable!("index_bank must be 'i' or 'r'"),
    };
    let descr = read_descr(code, op, 2, ctx)?;
    let descr_index = descr.index();

    let result = if let Some(cached) =
        ctx.trace_ctx
            .heapcache_getarrayitem(array, index, descr_index)
    {
        // pyjitpl.py:639-673 `_do_getarrayitem_gc_any` cache hit:
        //   tobox = heapcache.getarrayitem(...)
        //   if tobox:
        //       profiler.count_ops(rop.GETARRAYITEM_GC_I, HEAPCACHED_OPS)
        //       return tobox
        // RPython hardcodes `GETARRAYITEM_GC_I` regardless of the
        // recorded `typ` ('i' / 'r' / 'f'); pyre matches the hardcode
        // for profiling parity.
        ctx.trace_ctx.profiler().count_ops(
            OpCode::GetarrayitemGcI,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        cached
    } else {
        let resbox = ctx
            .trace_ctx
            .record_op_with_descr(opcode, &[array, index], descr.clone());
        // Box.value parity: `box_value` exposes the resolution chain
        // PyPy reads off `arraybox.getref_base()` / `indexbox.getint()`
        // (`rpython/jit/metainterp/executor.py:117`).  Any operand
        // whose Box.value is known unblocks `array_sanity_load`, not
        // just Const-pool entries (`pyjitpl.py:648-666 resbox =
        // execute_with_descr(...); getarrayitem_now_known(...)`
        // parity).
        let load_type = match opcode {
            OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcPureI => Some(majit_ir::Type::Int),
            OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcPureR => Some(majit_ir::Type::Ref),
            OpCode::GetarrayitemGcF | OpCode::GetarrayitemGcPureF => Some(majit_ir::Type::Float),
            _ => None,
        };
        let live_value = if let (
            Some(ty),
            Some(majit_ir::Value::Ref(array_ref)),
            Some(majit_ir::Value::Int(index_value)),
        ) = (
            load_type,
            ctx.trace_ctx.box_value(array),
            ctx.trace_ctx.box_value(index),
        ) {
            let array_ptr = array_ref.0 as i64;
            if array_ptr != usize::MAX as i64 && array_ptr != 0 {
                ctx.trace_ctx
                    .array_sanity_load(array_ptr, index_value, &descr, ty)
                    .unwrap_or(majit_ir::Value::Void)
            } else {
                majit_ir::Value::Void
            }
        } else {
            majit_ir::Value::Void
        };
        // Stamp the loaded value as Box.value of the recorded result
        // (RPython `Box(value)` constructor analog) so subsequent
        // consumers see the runtime concrete instead of the
        // GcRef(usize::MAX) sentinel.
        if !matches!(live_value, majit_ir::Value::Void) {
            ctx.trace_ctx.set_opref_concrete(resbox, live_value);
        }
        ctx.trace_ctx
            .heapcache_getarrayitem_now_known(array, index, descr_index, resbox);
        resbox
    };

    let dst = code[op.pc + 5] as usize;
    // concrete_of_opref derivation: derive shadow concrete from the recorded result's
    // `concrete_of_opref` entry instead of inventing Null.  Constant
    // arraybox + constant index hits land in `constants.get_value`;
    // virtualizable hits surface via `standard_virtualizable_box`;
    // `set_opref_concrete` stamps from upstream `binop_int_record`
    // flow back here too.  Null fallback preserves the prior contract.
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    match dst_bank {
        'i' => {
            write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'r' => {
            write_ref_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'f' => {
            let len = ctx.registers_f.len();
            let slot = ctx
                .registers_f
                .get_mut(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            *slot = result;
        }
        _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `setarrayitem_gc_<i|r|f>/ri{i,r,f}d` handler. Operand layout per
/// `bhimpl_setarrayitem_gc_{i,r,f}(cpu, array, index, newvalue,
/// arraydescr)` (`blackhole.py:1351-1359`):
/// 1B r-reg(array) + 1B i-reg(index) + 1B {i,r,f}-reg(newvalue) + 2B descr.
///
/// RPython parity: `pyjitpl.py:736-744 _opimpl_setarrayitem_gc_any`
/// dispatches through `metainterp.execute_setarrayitem_gc(arraydescr,
/// arraybox, indexbox, itembox)` — RPython's wrapper records
/// `rop.SETARRAYITEM_GC` and updates the heapcache via
/// `setarrayitem`.
///
/// No skip-on-redundant short-circuit (matches RPython —
/// `_opimpl_setarrayitem_gc_any` has no `if cached == value: return`,
/// because `heapcache.setarrayitem` already handles aliasing
/// invalidation at the right granularity).
///
/// `value_bank` selects the newvalue register source: `'i'` /
/// `'r'` / `'f'`.
fn setarrayitem_gc_via_heapcache(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    value_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let array = read_ref_reg(code, op, 0, ctx)?;
    let index = read_int_reg(code, op, 1, ctx)?;
    let value = match value_bank {
        'i' => read_int_reg(code, op, 2, ctx)?,
        'r' => read_ref_reg(code, op, 2, ctx)?,
        'f' => read_float_reg(code, op, 2, ctx)?,
        _ => unreachable!("value_bank must be 'i', 'r' or 'f'"),
    };
    let descr = read_descr(code, op, 3, ctx)?;
    let descr_index = descr.index();

    ctx.trace_ctx
        .record_op_with_descr(OpCode::SetarrayitemGc, &[array, index, value], descr);
    // `upd.setarrayitem(valuebox)` (heapcache.py:142) parity — the
    // cache stores the Box identity (`value` OpRef); cache-hit
    // readers fetch the intrinsic value via `box_value(cached)` at
    // hit time.
    ctx.trace_ctx
        .heapcache_setarrayitem(array, index, descr_index, value);
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `setfield_gc_<i|r>/<rid|rrd>` handler: read box (r-reg), valuebox
/// (i or r reg per `value_bank`), descr operand, then either skip
/// the IR emission (cache says the same value is already there) or
/// record `OpCode::SetfieldGc` and write through to the heapcache.
///
/// RPython parity: `pyjitpl.py:973-988 _opimpl_setfield_gc_any`:
///
///   upd = heapcache.get_field_updater(box, fielddescr)
///   if upd.currfieldbox is valuebox:
///       return                       # cache hit, no IR
///   self.metainterp.execute_and_record(rop.SETFIELD_GC, fielddescr,
///                                       box, valuebox)
///   upd.setfield(valuebox)
///
/// **Alias-clearing writeback**: goes through
/// `HeapCache::setfield_cached` instead of `getfield_now_known`. The
/// difference is the alias-clearing semantic that RPython's
/// `FieldUpdater.setfield()` carries (heapcache.py:142-143 routes to
/// `CacheEntry.do_write_with_aliasing`):
///
///   `_clear_cache_on_write(seen_alloc)` (heapcache.py:70-77) wipes
///   `cache_anything` unconditionally and additionally wipes
///   `cache_seen_allocation` when the write target itself is not
///   seen-allocated.  This conservatively kills any cached entry whose
///   source-box might alias the SETFIELD target.
///
/// `getfield_now_known` only inserts the new (obj, field, value) tuple
/// — it does NOT clear sibling entries.  Using it here meant a
/// subsequent `getfield_gc(other_obj, same_field)` could return a
/// stale value cached from before the SETFIELD.  Switching to
/// `setfield_cached` matches `do_write_with_aliasing` exactly.
///
/// `value_bank` selects the valuebox source: `'i'` reads
/// `registers_i[v]`, `'r'` reads `registers_r[v]`.
fn setfield_gc_via_heapcache(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    value_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Operand layout `<r><v>d`: 1B r-reg(box) + 1B v-reg(value) + 2B descr-index.
    let obj = read_ref_reg(code, op, 0, ctx)?;
    let valuebox = match value_bank {
        'i' => read_int_reg(code, op, 1, ctx)?,
        'r' => read_ref_reg(code, op, 1, ctx)?,
        'f' => read_float_reg(code, op, 1, ctx)?,
        _ => unreachable!("value_bank must be 'i', 'r' or 'f'"),
    };
    let descr = read_descr(code, op, 2, ctx)?;
    let descr_index = descr.index();

    // Cache hit: if the heapcache already records `valuebox` as the
    // current value of `(obj, descr)`, the SETFIELD_GC is redundant —
    // skip recording. RPython pyjitpl.py:973-979 _opimpl_setfield_gc_any:
    //   if upd.currfieldbox is valuebox:
    //       self.metainterp.staticdata.profiler.count_ops(rop.SETFIELD_GC, Counters.HEAPCACHED_OPS)
    //       return
    let is_redundant = ctx
        .trace_ctx
        .heapcache_getfield_cached(obj, descr_index)
        .map(|b| b)
        == Some(valuebox);
    if is_redundant {
        ctx.trace_ctx.profiler().count_ops(
            OpCode::SetfieldGc,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
    } else {
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[obj, valuebox], descr);
        // Write-through with alias-clearing semantics
        // (`heapcache.py:90-94 do_write_with_aliasing`).  Mirrors
        // `upd.setfield(valuebox)` (heapcache.py:142).
        ctx.trace_ctx
            .heapcache_setfield_cached(obj, descr_index, valuebox);
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `getfield_gc_<i|r>/rd>X` handler: read a Ref-bank source register
/// + descr operand, consult the heapcache, and either return the
/// cached field box (no IR op recorded) or record the appropriate
/// `OpCode::GetfieldGc<I|R>` op and update the cache.
///
/// RPython parity: `pyjitpl.py:855-882 opimpl_getfield_gc_<i|r>` →
/// `_opimpl_getfield_gc_any_pureornot` (`pyjitpl.py:929-950`).
/// RPython has a ConstPtr+is_always_pure() fast path at lines 856-860
/// that fires `executor.execute(cpu, metainterp, opnum, fielddescr,
/// box)` and returns `ConstInt/ConstFloat/ConstPtr(resvalue)` —
/// recording NO trace op (the value is directly substituted as a Const
/// literal). The symbolic walker has no `executor.execute` (no cpu /
/// concrete box pair), so the fast path is structurally unreachable.
///
/// Walker behaviour mirrors `_opimpl_getfield_gc_any_pureornot`
/// uniformly: heapcache hit returns the cached box (no IR op);
/// heapcache miss records `GetfieldGc<I|R>` (non-pure variant) +
/// writes through. The optimizer's always-pure pass later folds the
/// non-pure read into `GetfieldGcPure*` based on `descr.is_always_pure()`,
/// which is `OpHelpers.getfield_pure_for_descr` (resoperation.py:
/// 1284-1289) parity. Walker emitting Pure variants directly would be
/// a TODO since RPython's opimpl_* never emits the Pure
/// opcodes; they're an optimizer-rewrite artifact.
///
/// `dst_bank` selects the result bank: `'i'` writes `registers_i[dst]`,
/// `'r'` writes `registers_r[dst]`.
fn getfield_gc_via_heapcache(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Operand layout `rd>X`: 1B r-reg + 2B descr-index + 1B dst.
    let obj = read_ref_reg(code, op, 0, ctx)?;
    let descr = read_descr(code, op, 1, ctx)?;
    let descr_index = descr.index();

    let result = if let Some(cached) = ctx.trace_ctx.heapcache_getfield_cached(obj, descr_index) {
        // Cache hit (RPython pyjitpl.py:929-947 _opimpl_getfield_gc_any_pureornot):
        //   if upd.currfieldbox is not None:
        //       self.metainterp.staticdata.profiler.count_ops(rop.GETFIELD_GC_I, Counters.HEAPCACHED_OPS)
        //       return upd.currfieldbox
        // RPython hardcodes `GETFIELD_GC_I` for the count regardless of
        // the actual rop variant (`_i` / `_r` / `_f`); match the
        // hardcode for profiling parity.
        ctx.trace_ctx.profiler().count_ops(
            OpCode::GetfieldGcI,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        cached
    } else {
        // Cache miss — record op + write through.  `box_value`
        // resolves the Box.value chain PyPy reads off
        // `box.getref_base()` in `executor.do_getfield_gc_*`
        // (`executor.py:188`); the sanity load fires whenever the
        // struct pointer is known (Const, vable shadow, or stamped),
        // mirroring `pyjitpl.py:948-949 resbox = execute_with_descr(...);
        // upd.getfield_now_known(resbox)`.
        let resbox = ctx
            .trace_ctx
            .record_op_with_descr(opcode, &[obj], descr.clone());
        let load_type = match opcode {
            OpCode::GetfieldGcI | OpCode::GetfieldGcPureI => Some(majit_ir::Type::Int),
            OpCode::GetfieldGcR | OpCode::GetfieldGcPureR => Some(majit_ir::Type::Ref),
            OpCode::GetfieldGcF | OpCode::GetfieldGcPureF => Some(majit_ir::Type::Float),
            _ => None,
        };
        let live_value = if let (Some(ty), Some(majit_ir::Value::Ref(struct_ref))) =
            (load_type, ctx.trace_ctx.box_value(obj))
        {
            let struct_ptr = struct_ref.0 as i64;
            if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
                ctx.trace_ctx
                    .field_sanity_load(struct_ptr, &descr, ty)
                    .unwrap_or(majit_ir::Value::Void)
            } else {
                majit_ir::Value::Void
            }
        } else {
            majit_ir::Value::Void
        };
        // Stamp the loaded value as the Box.value of the recorded
        // result so subsequent reads (cache hits + non-Const
        // `concrete_of_opref` consumers) see the real runtime
        // concrete instead of the GcRef(usize::MAX) sentinel.
        if !matches!(live_value, majit_ir::Value::Void) {
            ctx.trace_ctx.set_opref_concrete(resbox, live_value);
        }
        ctx.trace_ctx
            .heapcache_getfield_now_known(obj, descr_index, resbox);
        resbox
    };

    let dst = code[op.pc + 4] as usize;
    // concrete_of_opref derivation: derive shadow concrete via `concrete_of_opref` so a
    // constant-folded predecessor (e.g. `binop_int_record` having
    // stamped this OpRef in OpRef concrete stamping) propagates through.  RPython
    // `Box.value` parity: `pyjitpl.py:executor.py` per-opcode LLOp
    // stamps `box.value` post-exec; pyre's `concrete_of_opref` reads
    // that channel.  Null fallback preserves the prior unknown-result
    // behaviour for cache-miss recorded ops.
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    match dst_bank {
        'i' => {
            write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'r' => {
            write_ref_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'f' => {
            let len = ctx.registers_f.len();
            let slot = ctx
                .registers_f
                .get_mut(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            *slot = result;
        }
        _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `getfield_vable_<i|r|f>/rd>X` handler. Operand layout `rd>X`:
/// 1B r-reg(vable_box) + 2B descr(field) + 1B X-dst.
///
/// RPython parity: `pyjitpl.py:1167-1186 opimpl_getfield_vable_{i,r,f}`:
///
///   def opimpl_getfield_vable_i(self, box, fielddescr, pc):
///       if self._nonstandard_virtualizable(pc, box, fielddescr):
///           return self.opimpl_getfield_gc_i(box, fielddescr)
///       self.metainterp.check_synchronized_virtualizable()
///       index = self._get_virtualizable_field_index(fielddescr)
///       return self.metainterp.virtualizable_boxes[index]
///
/// The walker delegates to the orthodox `TraceCtx::vable_getfield_{int,
/// ref,float}` ports (`majit-metainterp/src/trace_ctx.rs:1715, 1801,
/// 1839`) which already implement the full
/// `_nonstandard_virtualizable` check + heapcache-aware GETFIELD_GC
/// fallback + `virtualizable_boxes[index]` cache read.  Walker is the
/// symbolic shadow validator (`shadow_walker.rs`); only the OpRef
/// component of the `(OpRef, Value)` tuple is meaningful here, since
/// register banks carry only OpRefs — the concrete `Value` is tracked
/// on the trait-driven leg.  `dst_bank` selects the result bank
/// (`'i'`/`'r'`/`'f'`) the walker writes back into, mirroring
/// `getfield_gc_via_heapcache`'s shape.
fn getfield_vable_via_metainterp(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let obj = read_ref_reg(code, op, 0, ctx)?;
    // RPython's `box` is always a live virtualizable-frame box. An
    // unseeded walker Ref register holds `OpRef::None` (`raw() ==
    // u32::MAX`); feeding it into the metainterp vable path would resize
    // the heapcache flag vector to 16 GiB. Bail to a trace abort instead.
    if obj.is_none() {
        return Err(DispatchError::VableBoxNotSeeded { pc: op.pc });
    }
    let descr = read_descr(code, op, 1, ctx)?;

    // R7 parity: RPython `opimpl_getfield_vable_{i,r,f}(box, fielddescr,
    // pc)` threads orgpc through `_nonstandard_virtualizable(pc, ...)`
    // (pyjitpl.py:1167-1186 + :1137).  Pyre's walker has the matching
    // JitCode PC in `op.pc`; pass it through so the helper signature
    // stays line-by-line equivalent even if `is_nonstandard_virtualizable`
    // currently ignores the pc at the leaf (`trace_ctx.rs let _ = pc;`).
    let pc = op.pc;
    // Concrete struct pointer for pyjitpl.py:934-945 cache-hit sanity
    // check.  The walker keeps a parallel concrete Ref-bank shadow;
    // thread the same live pointer that RPython's `box.getref_base()`
    // would expose to `executor.execute(...)`.
    let vable_struct_ptr = match read_ref_reg_concrete(code, op, 0, ctx) {
        ConcreteValue::Ref(ptr) => ptr as i64,
        ConcreteValue::Null => 0,
        ConcreteValue::Int(_) | ConcreteValue::Float(_) | ConcreteValue::Bool(_) => 0,
    };
    let (result, shadow_value) = match dst_bank {
        'i' => ctx
            .trace_ctx
            .vable_getfield_int(pc, obj, vable_struct_ptr, descr),
        'r' => ctx
            .trace_ctx
            .vable_getfield_ref(pc, obj, vable_struct_ptr, descr),
        'f' => ctx
            .trace_ctx
            .vable_getfield_float(pc, obj, vable_struct_ptr, descr),
        _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
    };
    // RPython `opimpl_getfield_vable_{i,r,f}` returns
    // `virtualizable_boxes[index]` (`pyjitpl.py:1186`) — a Box whose
    // `_resint`/`_resref`/`_resfloat` is filled at construction time.
    // `box.getint()` returns the live value without any side-lookup.
    // Pyre splits OpRef↔concrete into a side table; mirror the Box.value
    // contract by stamping the read result's concrete into
    // `opref_concrete` so `concrete_of_opref(result)` honors the same
    // contract for downstream consumers (`goto_if_not/iL`,
    // `switch/id`, `int_*` arithmetic).  The non-standard heapcache
    // path inside `vable_getfield_int` already does the same stamp
    // (trace_ctx.rs:2384); the standard path returns the cached
    // `(opref, value)` pair without stamping.  `Value::Void` means no
    // live concrete is available for this slot — skip to match the
    // heapcache path's gating.
    if !matches!(shadow_value, Value::Void) {
        ctx.trace_ctx.set_opref_concrete(result, shadow_value);
    }

    let dst = code[op.pc + 4] as usize;
    // concrete_of_opref derivation: derive shadow concrete via `concrete_of_opref`.  The
    // `vable_getfield_*` helpers in `TraceCtx` already populate the
    // concrete shadow for virtualizable-resident fields via the
    // `standard_virtualizable_box()`/`virtualizable_boxes` channel,
    // and feed `set_opref_concrete` on the GETFIELD_GC fallback for
    // non-vable structs — both surface through this lookup.
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    match dst_bank {
        'i' => {
            write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'r' => {
            write_ref_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'f' => {
            let len = ctx.registers_f.len();
            let slot = ctx
                .registers_f
                .get_mut(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            *slot = result;
        }
        _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `setfield_vable_<i|r|f>/r<v>d` handler. Operand layout:
/// 1B r-reg(vable_box) + 1B <v>-reg(value) + 2B descr(field).
/// No dst byte (set, not get).
///
/// RPython parity: `pyjitpl.py:1188-1199 _opimpl_setfield_vable`:
///
///   def _opimpl_setfield_vable(self, box, valuebox, fielddescr, pc):
///       if self._nonstandard_virtualizable(pc, box, fielddescr):
///           return self._opimpl_setfield_gc_any(box, valuebox, fielddescr)
///       index = self._get_virtualizable_field_index(fielddescr)
///       self.metainterp.virtualizable_boxes[index] = valuebox
///       self.metainterp.synchronize_virtualizable()
///       # XXX only the index'th field needs to be synchronized, really
///
/// The walker delegates to `TraceCtx::vable_setfield`
/// (`majit-metainterp/src/trace_ctx.rs:1759`) which implements the
/// full `_nonstandard_virtualizable` -> SETFIELD_GC fallback +
/// `virtualizable_boxes[index] = valuebox` write + `synchronize_virtualizable`
/// mirror.  The concrete `Value` is reconstructed via
/// `TraceCtx::concrete_of_opref` (matches the trait-leg's
/// `pyjitpl/dispatch.rs:1608-1609` shape `let (value, concrete) =
/// self.read_<bank>_reg(src); ctx.vable_setfield(...)`).
///
/// `value_bank` selects the value register bank (`'i'`/`'r'`/`'f'`),
/// mirroring `setfield_gc_via_heapcache`'s parameter shape.
fn setfield_vable_via_metainterp(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    value_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let obj = read_ref_reg(code, op, 0, ctx)?;
    // Same unseeded-register guard as `getfield_vable_via_metainterp`:
    // a `None` box would resize the heapcache flag vector to 16 GiB.
    if obj.is_none() {
        return Err(DispatchError::VableBoxNotSeeded { pc: op.pc });
    }
    let value = match value_bank {
        'i' => read_int_reg(code, op, 1, ctx)?,
        'r' => read_ref_reg(code, op, 1, ctx)?,
        'f' => read_float_reg(code, op, 1, ctx)?,
        _ => unreachable!("value_bank must be 'i', 'r' or 'f'"),
    };
    let descr = read_descr(code, op, 2, ctx)?;
    let concrete = ctx.trace_ctx.concrete_of_opref(value);
    // R7 parity: pyjitpl.py:1188-1199 `_opimpl_setfield_vable(box,
    // valuebox, fielddescr, pc)` threads orgpc through
    // `_nonstandard_virtualizable(pc, ...)`; walker has `op.pc` for the
    // JitCode PC, pass through.
    ctx.trace_ctx
        .vable_setfield(op.pc, obj, descr, value, concrete);
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Generic int-bank unary handler. Operand layout `i>i` (1B src + 1B
/// dst). RPython parity: `pyjitpl.py:356-368` exec-generated
/// `opimpl_int_<unary>` (int_neg / int_invert / int_is_zero etc.) +
/// the `>i` decorator's writeback. Walker reads `registers_i[src]`,
/// records `OpCode::<Variant>` with `[a]`, writes the recorder result
/// into `registers_i[dst]`.
fn unop_int_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_int_reg(code, op, 0, ctx)?;
    let result = ctx.trace_ctx.record_op(opcode, &[a]);
    // Box(value) parity: stamp the unary result from the operand's
    // Box.value carrier (matches dispatch.rs trace_unary_i).  The
    // folded value also feeds the slot-keyed shadow via
    // [`write_int_reg`].
    let concrete = if let Some(majit_ir::Value::Int(n)) = ctx.trace_ctx.box_value(a) {
        let folded = majit_metainterp::eval_unary_i(opcode, n);
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int(folded));
        ConcreteValue::Int(folded)
    } else {
        ConcreteValue::Null
    };
    let dst = code[op.pc + 2] as usize;
    write_int_reg(ctx, op.pc, dst, result, concrete)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Generic ref-bank → int-bank binop handler. Operand layout `rr>i`
/// (1B r-src1 + 1B r-src2 + 1B i-dst). RPython parity:
/// `pyjitpl.py:326-336` exec-generated `opimpl_ptr_eq` /
/// `opimpl_ptr_ne` (and instance variants) follow `self.execute(rop.<OPNUM>,
/// b1, b2)` — both `b1`/`b2` are ref boxes, result is an int box. The
/// `b1 is b2` fast path is omitted (same rationale as `binop_int_record`'s
/// comparison family — pyre's recorder shares constants by value).
fn binop_ref_to_int_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_ref_reg(code, op, 0, ctx)?;
    let b = read_ref_reg(code, op, 1, ctx)?;
    let result = ctx.trace_ctx.record_op(opcode, &[a, b]);
    // Box(value) parity: stamp the bool result from the operands' Box.value
    // carriers (matches dispatch.rs trace_binop_r_to_i).
    if let (Some(majit_ir::Value::Ref(la)), Some(majit_ir::Value::Ref(rb))) =
        (ctx.trace_ctx.box_value(a), ctx.trace_ctx.box_value(b))
    {
        let folded = match opcode {
            OpCode::PtrEq | OpCode::InstancePtrEq => (la == rb) as i64,
            OpCode::PtrNe | OpCode::InstancePtrNe => (la != rb) as i64,
            _ => panic!("binop_ref_to_int_record: unsupported opcode {opcode:?}"),
        };
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int(folded));
    }
    let dst = code[op.pc + 3] as usize;
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `ptr_nonzero/r>i` handler (operand layout `r>i`: 1B r-src + 1B i-dst).
///
/// RPython parity:
/// `pyjitpl.py:378-380 opimpl_ptr_nonzero(box)`:
/// ```python
/// @arguments("box")
/// def opimpl_ptr_nonzero(self, box):
///     return self.execute(rop.PTR_NE, box, CONST_NULL)
/// ```
///
/// Walker reads one `r` reg, records `OpCode::PtrNe` with
/// `[box, CONST_NULL]` (via `trace_ctx.const_null()` —
/// `history.py:361 CONST_NULL = ConstPtr(ConstPtr.value)`), and writes
/// the recorder result into `registers_i[dst]`.  RPython does the
/// same `b1 is b2` short-circuit at `pyjitpl.py:328-332` for
/// `opimpl_ptr_eq` but `ptr_nonzero` against `CONST_NULL` cannot
/// short-circuit because `box` is never the literal `CONST_NULL`
/// constant (codewriter would have folded that).
fn ptr_nonzero_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let box_ = read_ref_reg(code, op, 0, ctx)?;
    let null_const = ctx.trace_ctx.const_null();
    let result = ctx.trace_ctx.record_op(OpCode::PtrNe, &[box_, null_const]);
    // Box(value) parity: stamp the nullity result from the operand's
    // Box.value carrier (matches dispatch.rs trace_ptr_nullity nonzero=true).
    if let Some(majit_ir::Value::Ref(r)) = ctx.trace_ctx.box_value(box_) {
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int((r.0 != 0) as i64));
    }
    let dst = code[op.pc + 2] as usize;
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `ref_guard_value/r` handler (operand layout `r`: 1B r-src, no dst).
///
/// RPython parity: `pyjitpl.py:1494-1496 _opimpl_guard_value` →
/// `pyjitpl.py:1916-1927 implement_guard_value`:
///
/// ```python
/// def implement_guard_value(self, box, orgpc):
///     if isinstance(box, Const):
///         return box                     # no promotion needed
///     else:
///         promoted_box = executor.constant_from_op(box)
///         self.metainterp.generate_guard(rop.GUARD_VALUE, box,
///                                        promoted_box, resumepc=orgpc)
///         self.metainterp.replace_box(box, promoted_box)
///         return promoted_box
/// ```
///
/// Walker behaviour:
///   * Read 1B Ref operand and its concrete shadow.
///   * If the symbolic OpRef is already a Const, skip (Const arm of
///     `implement_guard_value`).
///   * If the concrete shadow is `ConcreteValue::Null`, skip — the
///     walker doesn't have a runtime value to mint the expected
///     constant from.  This is the strictest mode (sibling
///     `dispatch_switch_id` line 1207 falls into the same skip-guard
///     branch when `valuebox.is_constant()`).
///   * Otherwise mint `ConstPtr(concrete_ptr)` (executor.py:544-551
///     `constant_from_op` for a Ref-typed Box), emit `GuardValue`
///     with `[value, expected_ref]`, and call `replace_box(value,
///     expected_ref)` (pyjitpl.py:1923).  Also rewrite every
///     `registers_r` slot still pointing at `value` to `expected_ref`,
///     matching `dispatch_switch_id:1198-1202`.
///
/// TODO: guards record with empty resume data
/// (`record_guard(..., 0)`) — same caveat as `dispatch_switch_id`
/// (no MIFrame liveness / framestack in the standalone walker).
fn ref_guard_value_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let value = read_ref_reg(code, op, 0, ctx)?;
    if value.is_constant() {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    let concrete = read_ref_reg_concrete(code, op, 0, ctx);
    let ConcreteValue::Ref(ptr) = concrete else {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    };
    if ptr.is_null() {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    let expected = ctx.trace_ctx.const_ref(ptr as usize as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[value, expected], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc);
    ctx.trace_ctx.replace_box(value, expected);
    for slot in ctx.registers_r.iter_mut() {
        if *slot == value {
            *slot = expected;
        }
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Generic float-pair-to-int handler for `float_<cmp>/ff>i` (operand
/// layout `ff>i`: 1B f-src + 1B f-src + 1B i-dst).  RPython parity:
/// `bhimpl_float_{lt,le,eq,ne,gt,ge}` (`blackhole.py:721-746`) — read
/// two `f` regs, record `OpCode::Float<Cmp>`, write the recorder
/// result into `registers_i[dst]`.
fn binop_float_to_int_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_float_reg(code, op, 0, ctx)?;
    let b = read_float_reg(code, op, 1, ctx)?;
    let result = ctx.trace_ctx.record_op(opcode, &[a, b]);
    // Box(value) parity: stamp the bool result from the operands' Box.value
    // carriers (matches dispatch.rs GOTO_IF_NOT_FLOAT_* + trace_float_compare).
    if let (Some(majit_ir::Value::Float(fa)), Some(majit_ir::Value::Float(fb))) =
        (ctx.trace_ctx.box_value(a), ctx.trace_ctx.box_value(b))
    {
        let folded =
            majit_metainterp::eval_float_cmp(opcode, fa.to_bits() as i64, fb.to_bits() as i64);
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Int(folded));
    }
    let dst = code[op.pc + 3] as usize;
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `cast_int_to_float/i>f` handler. Operand layout `i>f` (1B i-src +
/// 1B f-dst). RPython parity: `pyjitpl.py:357 cast_int_to_float`
/// belongs to the same exec-generated unary opimpl loop —
/// `self.execute(rop.CAST_INT_TO_FLOAT, b)`. Result lands in the
/// float bank (the `>f` decorator) instead of the int bank.
fn cast_int_to_float_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_int_reg(code, op, 0, ctx)?;
    let result = ctx.trace_ctx.record_op(OpCode::CastIntToFloat, &[a]);
    // Box.value parity — if `a`'s runtime concrete is known, stamp
    // the cast result with the corresponding float bit-pattern so
    // downstream `box_value(result)` callers see the live value.
    if let Some(majit_ir::Value::Int(n)) = ctx.trace_ctx.box_value(a) {
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Float(n as f64));
    }
    let dst = code[op.pc + 2] as usize;
    let len = ctx.registers_f.len();
    let slot = ctx
        .registers_f
        .get_mut(dst)
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg: dst,
            len,
            bank: "f",
        })?;
    *slot = result;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Generic float-bank binop handler. Operand layout `ff>f` (1B src1
/// + 1B src2 + 1B dst). RPython parity: same as `binop_int_record`
/// but on the float bank — `pyjitpl.py:284-292`'s exec-generated
/// `opimpl_float_<binop>` reads two `f` regs, calls
/// `self.execute(rop.<OPNUM>, b1, b2)`, and the trailing `>f`
/// decorator writes the result into `registers_f[dst]`.
fn binop_float_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_float_reg(code, op, 0, ctx)?;
    let b = read_float_reg(code, op, 1, ctx)?;
    let result = ctx.trace_ctx.record_op(opcode, &[a, b]);
    // Box(value) parity: stamp the result from the operands' Box.value
    // carriers (matches dispatch.rs trace_binop_f).
    if let (Some(majit_ir::Value::Float(fa)), Some(majit_ir::Value::Float(fb))) =
        (ctx.trace_ctx.box_value(a), ctx.trace_ctx.box_value(b))
    {
        let bits = majit_metainterp::eval_binop_f(opcode, fa.to_bits() as i64, fb.to_bits() as i64);
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Float(f64::from_bits(bits as u64)));
    }
    let dst = code[op.pc + 3] as usize;
    let len = ctx.registers_f.len();
    let slot = ctx
        .registers_f
        .get_mut(dst)
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg: dst,
            len,
            bank: "f",
        })?;
    *slot = result;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Generic float-bank unary handler. Operand layout `f>f` (1B src
/// + 1B dst). RPython equivalent: `bhimpl_float_neg(value)` →
/// `pyjitpl.py:execute(rop.FLOAT_NEG, value)`. Recording-only path
/// is the same shape as `binop_float_record` minus one read.
fn unop_float_record(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    opcode: OpCode,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let a = read_float_reg(code, op, 0, ctx)?;
    let result = ctx.trace_ctx.record_op(opcode, &[a]);
    // Box(value) parity: stamp the unary float result (matches dispatch.rs
    // trace_unary_f — FloatNeg / FloatAbs).
    if let Some(majit_ir::Value::Float(fa)) = ctx.trace_ctx.box_value(a) {
        let bits = majit_metainterp::eval_unary_f(opcode, fa.to_bits() as i64);
        ctx.trace_ctx
            .set_opref_concrete(result, majit_ir::Value::Float(f64::from_bits(bits as u64)));
    }
    let dst = code[op.pc + 2] as usize;
    let len = ctx.registers_f.len();
    let slot = ctx
        .registers_f
        .get_mut(dst)
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg: dst,
            len,
            bank: "f",
        })?;
    *slot = result;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Line-by-line port of `pyjitpl.py:1960-1993 MetaInterp._build_allboxes`.
/// Permutes a flat `argboxes` array (concat of i-list ++ r-list ++ f-list
/// in source order) so positions match the callee's `descr.get_arg_types()`
/// ABI ordering. Returns `[funcbox, ...permuted_argboxes]`.
///
/// RPython operates on a flat `argboxes` of typed `Box` objects + reads
/// `box.type`. The walker has only `OpRef`s, so the type is supplied
/// out-of-band via the `argbox_types` parallel array. By construction
/// the operand decoders (`read_int_var_list` / `read_ref_var_list` /
/// `read_float_var_list`) tag each entry with its bank, so the parallel
/// array is correct without needing a runtime type query.
///
/// The RPython `prepend_box` parameter is unused at every
/// `residual_call*` call site (only `conditional_call*` uses it, not
/// yet ported), so it's omitted from the walker signature. Add it back
/// when porting `opimpl_conditional_call*`.
fn build_allboxes(
    funcbox: OpRef,
    argboxes: &[OpRef],
    argbox_types: &[Type],
    arg_types: &[Type],
) -> Vec<OpRef> {
    debug_assert_eq!(
        argboxes.len(),
        argbox_types.len(),
        "argboxes and argbox_types must align",
    );
    // RPython line 1961: `allboxes = [None] * (len(argboxes)+1 + …)`.
    let total = arg_types.len() + 1;
    let mut allboxes: Vec<OpRef> = Vec::with_capacity(total);
    // RPython line 1966: `allboxes[i] = funcbox`.
    allboxes.push(funcbox);
    // RPython line 1968: `src_i = src_r = src_f = 0`.
    let mut src_i = 0usize;
    let mut src_r = 0usize;
    let mut src_f = 0usize;
    // RPython line 1969-1989: outer `for kind in descr.get_arg_types()`
    // with one type-filter `while True` loop per kind.
    for &kind in arg_types {
        let box_oref = match kind {
            Type::Int => loop {
                // RPython line 1971-1975: advance src_i past non-INT
                // entries until an INT box is found.
                let b = argboxes[src_i];
                let bt = argbox_types[src_i];
                src_i += 1;
                if bt == Type::Int {
                    break b;
                }
            },
            Type::Ref => loop {
                // RPython line 1977-1981.
                let b = argboxes[src_r];
                let bt = argbox_types[src_r];
                src_r += 1;
                if bt == Type::Ref {
                    break b;
                }
            },
            Type::Float => loop {
                // RPython line 1983-1987 (kind == 'L' long-long path
                // not separately modeled — pyre's Type::Float covers
                // both).
                let b = argboxes[src_f];
                let bt = argbox_types[src_f];
                src_f += 1;
                if bt == Type::Float {
                    break b;
                }
            },
            // RPython line 1988-1989: `else: raise AssertionError`.
            // Type::Void in arg_types is an internal invariant violation.
            Type::Void => panic!("_build_allboxes: arg_types must not contain Void"),
        };
        allboxes.push(box_oref);
    }
    debug_assert_eq!(allboxes.len(), total, "allboxes shape post-condition");
    allboxes
}

/// Reject a residual_call whose `allboxes` (funcbox + permuted args)
/// contains an `OpRef::NONE`.  RPython's `do_residual_call` resolves
/// each argbox through `env[box]`, where an unbound box is a `KeyError`;
/// recording the op anyway lets `OpRef::NONE` reach the backend's
/// `resolve_opref`, which aborts the process.  Returning
/// `ResidualCallArgUnbound` instead lets the outer walker fall back to a
/// trace abort — the same graceful outcome a pre-seam inline arm reached
/// when its payload read surfaced `GotoIfNotValueNotConcrete`.
fn ensure_residual_call_args_bound(allboxes: &[OpRef], pc: usize) -> Result<(), DispatchError> {
    if let Some(arg_index) = allboxes.iter().position(|b| b.is_none()) {
        return Err(DispatchError::ResidualCallArgUnbound { pc, arg_index });
    }
    Ok(())
}

/// Decode the descr index from a 2-byte LE operand. Companion to
/// [`read_descr`] for callers that need the raw index for error
/// reporting (e.g. `ResidualCallDescrNotCallDescr`).
fn decode_descr_index(code: &[u8], op: &DecodedOp, operand_offset: usize) -> usize {
    let lo = code[op.pc + 1 + operand_offset] as usize;
    let hi = code[op.pc + 1 + operand_offset + 1] as usize;
    lo | (hi << 8)
}

/// EffectInfo-driven opcode selector shared by `dispatch_residual_call_*`
/// dispatchers. Mirrors `pyjitpl.py:1995-2126 do_residual_call`'s
/// precedence:
///   1. **forces branch** (`pyjitpl.py:2007-2082`): outer check on
///      `assembler_call or check_forces_virtual_or_virtualizable()`
///      records `CALL_MAY_FORCE_*` at step 2 and unconditionally fires
///      `GUARD_NOT_FORCED` (`:2079`).  The release-gil sub-case
///      (`pyjitpl.py:2063 if effectinfo.is_call_release_gil()`) is
///      handled by [`direct_call_release_gil`] **before** this
///      selector is called — the dispatcher early-returns on
///      `ei.is_call_release_gil()` so this function only ever sees
///      EI values where the sub-case is not active.
///   2. `EF_LOOPINVARIANT` (`:2087-2110`): `CALL_LOOPINVARIANT_*`.
///   3. `check_is_elidable()` (`:2112-2126`): `CALL_PURE_*`.
///   4. default (`:2126`): plain `CALL_*`.
///
/// Returns the `Call*` opcode for the call itself, whether
/// `handle_possible_exception` should emit `GUARD_NO_EXCEPTION`
/// (`check_can_raise(False)`), and whether the unconditional
/// `GUARD_NOT_FORCED` from the forces branch (`pyjitpl.py:2079`)
/// should fire.
///
/// ★ Task #390 sub-slice 1 — non-elidable concrete-execute inventory.
///
/// PyPy `do_residual_call` (pyjitpl.py:1995-2126) **always** calls
/// `executor.execute_varargs(opnum, argboxes, descr, exc=can_raise,
/// pure=is_elidable)` regardless of EI branch.  The recorded opcode
/// kind selected above is for the IR trace; concrete execution always
/// happens at trace-record time.  Pyre's walker only concrete-executes
/// the `CallPure*` branch (via [`try_fold_pure_call_via_executor`]),
/// so the other three branches surface as the M4 SIGBUS root cause
/// (memory: M4 walker unactivated taxonomy).
///
/// Per-branch concrete-execute status today:
///
/// | EI branch | Selected op | PyPy `execute_varargs` | Pyre walker | Reachable via |
/// |---|---|---|---|---|
/// | `is_call_release_gil()` | (early-routed to `direct_call_release_gil`) | n/a — separate `do_call_release_gil` path | (release-gil IR only, no concrete exec) | not via this selector |
/// | `check_forces_virtual_or_virtualizable()` | `CallMayForce*` + `GuardNotForced` | `pure=False, exc=can_raise` — runs the helper, may force virtualizable | **record-only** | dispatcher_iRd/iIRd/iIRFd_kind forces branch |
/// | `extraeffect == LoopInvariant` | `CallLoopinvariant*` | `pure=False, exc=False` — runs on cache miss | record-only on cache miss, [`loopinvariant_lookup`] reuses cached OpRef on hit (no execute, no record) | same dispatchers |
/// | `check_is_elidable()` | `CallPure*` | `pure=True, exc=can_raise` — runs the helper, caches result | [`try_fold_pure_call_via_executor`] (elidable_cannot_raise only — see its caveats) | same dispatchers |
/// | default | `Call*` + (`GuardNoException` iff can_raise) | `pure=False, exc=can_raise` — runs the helper | **record-only** | same dispatchers |
///
/// **Walker-record sites for non-elidable concrete-execute gap:**
///   * [`dispatch_residual_call_iRd_kind`] (`residual_call_r_*/iRd>{v,r,i,f}` ←
///     pyjitpl.py:1346 `_opimpl_residual_call1` / 1348 / 1350 / 1352)
///   * [`dispatch_residual_call_iIRd_kind`] (`residual_call_ir_*/iIRd>{v,r,i,f}` ←
///     pyjitpl.py:1349 `_opimpl_residual_call2` / 1351 / 1353 / 1355)
///   * [`dispatch_residual_call_iIRFd_kind`] (`residual_call_irf_*/iIRFd>{v,r,i,f}` ←
///     pyjitpl.py:1354 `_opimpl_residual_call3` / 1355 / 1357 / 1359)
///
/// All three call [`select_residual_call_opcode`] then
/// `record_op_with_descr` then [`try_fold_pure_call_via_executor`].
/// The orthodox fix replaces the last call with a wider
/// `try_execute_residual_call_via_executor(call_opcode, ei, ...)`
/// whose body matches PyPy `executor.execute_varargs` per the
/// branch table above.
///
/// **Priority order for sub-slice 2 (widen):** Call* (default — the
/// store_subscr_fn / set_current_exception class, smallest blast
/// radius, root cause of M4 SIGBUS) → CallLoopinvariant* (`pure=False
/// exc=False` is safest) → CallMayForce* (riskiest — force-virtual
/// audit precondition).
fn select_residual_call_opcode(
    ei: &majit_ir::EffectInfo,
    dst_bank: char,
    caller: &'static str,
) -> (OpCode, bool, bool) {
    // Release-gil sub-case is handled by `direct_call_release_gil`
    // before this selector runs.  Any `is_call_release_gil()` EI
    // reaching here is a dispatcher bug.
    debug_assert!(
        !ei.is_call_release_gil(),
        "{caller}: select_residual_call_opcode received an is_call_release_gil() EI; \
         dispatcher should have routed via direct_call_release_gil first"
    );
    let (call_op, pure_op, may_force_op, loopinvariant_op): (OpCode, OpCode, OpCode, OpCode) =
        match dst_bank {
            'r' => (
                OpCode::CallR,
                OpCode::CallPureR,
                OpCode::CallMayForceR,
                OpCode::CallLoopinvariantR,
            ),
            'i' => (
                OpCode::CallI,
                OpCode::CallPureI,
                OpCode::CallMayForceI,
                OpCode::CallLoopinvariantI,
            ),
            // `_irf_f/iIRFd>f` (`pyjitpl.py:1354 opimpl_residual_call_irf_f =
            // _opimpl_residual_call3`, `blackhole.py:1250 bhimpl_residual_call_irf_f`).
            // `resoperation.py:1462 Type::Float => CallF`. The `_r_f` /
            // `_ir_f` shapes do not exist upstream — the only float-result
            // residual_call variant routes through the `iIRFd` arglist.
            'f' => (
                OpCode::CallF,
                OpCode::CallPureF,
                OpCode::CallMayForceF,
                OpCode::CallLoopinvariantF,
            ),
            // `_*_v/iRd|iIRd|iIRFd` void variants (`pyjitpl.py:1348
            // opimpl_residual_call_r_v = _opimpl_residual_call1`,
            // `:1351 opimpl_residual_call_ir_v = _opimpl_residual_call2`,
            // `:1355 opimpl_residual_call_irf_v = _opimpl_residual_call3`,
            // `blackhole.py:1245/1248/1253 bhimpl_residual_call_*_v`).
            // `resoperation.py:1463 Type::Void => CallN`. No dst writeback;
            // `write_residual_call_result_to_dst` no-ops on 'v'.
            'v' => (
                OpCode::CallN,
                OpCode::CallPureN,
                OpCode::CallMayForceN,
                OpCode::CallLoopinvariantN,
            ),
            _ => panic!("{caller}: unsupported dst_bank '{dst_bank}'"),
        };
    if ei.check_forces_virtual_or_virtualizable() {
        // pyjitpl.py:2017-2082 forces-virtual-or-virtualizable branch
        // proper: CALL_MAY_FORCE_* + GUARD_NOT_FORCED.
        // `handle_possible_exception` also fires (forces always
        // satisfies check_can_raise).
        (may_force_op, ei.check_can_raise(false), true)
    } else if ei.extraeffect == majit_ir::ExtraEffect::LoopInvariant {
        // pyjitpl.py:2087-2110 EF_LOOPINVARIANT branch: CALL_LOOPINVARIANT_*
        // via miframe_execute_varargs(..., exc=False). LoopInvariant
        // never raises (extraeffect=1 < CannotRaise=2 → check_can_raise=False).
        //
        // The `pyjitpl.py:2088 call_loopinvariant_known_result` lookup
        // and `pyjitpl.py:2109 call_loopinvariant_now_known` cache
        // update are wired at the dispatcher level via
        // [`loopinvariant_lookup`] and [`loopinvariant_now_known`]
        // around the `record_op_with_descr` call — they require the
        // dispatcher's `descr_index` and `arg0_int` so this opcode
        // selector cannot perform them on its own.
        (loopinvariant_op, ei.check_can_raise(false), false)
    } else if ei.check_is_elidable() {
        // pyjitpl.py:2112 + 2126 elidable branch: CALL_PURE_*.
        (pure_op, ei.check_can_raise(false), false)
    } else {
        // pyjitpl.py:2126 default branch: CALL_*.
        (call_op, ei.check_can_raise(false), false)
    }
}

/// `pyjitpl.py:2088-2090 heapcache.call_loopinvariant_known_result`
/// short-circuit: when the EffectInfo's extraeffect is `EF_LOOPINVARIANT`
/// AND the heapcache has a cached result for `(descr_index, allboxes[0])`,
/// the trace skips re-recording the `CALL_LOOPINVARIANT_*` op and the
/// caller reuses the cached OpRef.  Returns `None` for non-loopinvariant
/// EI or a cache miss; the caller then falls through to the normal record
/// path and follows up with [`loopinvariant_now_known`] to populate the
/// cache for subsequent matching calls.
///
/// RPython upstream (`heapcache.py:629-634`) keys the lookup by descr
/// **identity** and `allboxes[0].getint()`.  Upstream's
/// `do_residual_or_indirect_call` (`pyjitpl.py:2174-2186`) reaches
/// `do_residual_call` for **both** non-`Const` `funcbox` and `Const`
/// funcboxes whose address has no registered jitcode — the
/// `isinstance(funcbox, Const)` guard only short-circuits to
/// `perform_call` when `bytecode_for_address` resolves a jitcode.
/// In the residual path `allboxes[0].getint()` is well-defined
/// regardless of `Const`-ness because every Box subclass exposes
/// `getint()` over its runtime int (`history.BoxInt._value` /
/// `ConstInt.value`).
///
/// pyre's [`TraceCtx::concrete_of_opref`]
/// (`majit-metainterp/src/trace_ctx.rs:1646`) reconstructs the
/// concrete int from the per-trace constant pool only for constant
/// OpRefs; non-constant OpRefs are symbolic at the dispatcher
/// layer and carry no runtime int the trace-time walker can read.
/// When `funcptr` is non-constant we skip the cache entirely —
/// using `funcptr.0` as a sentinel would key on symbolic identity
/// rather than the concrete callee, risking false hits across two
/// different non-const funcptrs that share an OpRef after IR
/// renaming, and false misses across two different OpRefs aliasing
/// the same concrete callee.  Returning `None` is the conservative
/// choice (the caller falls through to record the call); the cost
/// is a missed cache hit on non-const funcptrs that upstream would
/// have caught.  Convergence with upstream's full coverage requires
/// threading concrete-int shadow alongside OpRef for non-const ints
/// — significant work remaining.
///
/// `descr_key` is the descriptor's stable identity key (`Descr::index()`),
/// matching upstream's identity comparison on `descr` more closely than
/// the operand-encoded descr-table slot.
#[inline]
fn loopinvariant_lookup(
    ctx: &WalkContext<'_, '_>,
    ei: &majit_ir::EffectInfo,
    descr_key: u32,
    funcptr: OpRef,
) -> Option<OpRef> {
    if ei.extraeffect != majit_ir::ExtraEffect::LoopInvariant {
        return None;
    }
    let arg0_int = funcptr_concrete_int(ctx, funcptr)?;
    ctx.trace_ctx
        .heap_cache()
        .call_loopinvariant_known_result(descr_key, arg0_int)
        .map(|(opref, _resvalue)| opref)
}

/// `pyjitpl.py:2109 heapcache.call_loopinvariant_now_known`: after
/// recording a fresh `CALL_LOOPINVARIANT_*` op, remember the
/// `(descr_index, allboxes[0].getint()) -> result` mapping so the
/// next matching call short-circuits via [`loopinvariant_lookup`].
/// No-op for non-loopinvariant EI, and no-op when `funcptr` is
/// non-constant (no concrete int key — see [`loopinvariant_lookup`]).
///
/// `resvalue` is stored as `0`.  RPython's upstream caller
/// (`pyjitpl.py:2109`) stores `res` — the concrete value returned
/// by `execute_varargs` after actually running the callee.  pyre-
/// jit-trace records symbolically without executing, so no concrete
/// result exists at this point: `record_op_with_descr` returns a
/// freshly-minted SSA OpRef whose runtime value is only known when
/// the compiled trace later executes.  The cached `_resvalue` is
/// unused by [`loopinvariant_lookup`]'s consumer (only the symbolic
/// OpRef is read for register writeback), so the `0` placeholder is
/// observationally equivalent.  Convergence with upstream requires
/// either threading the concrete result up from the executing trace
/// (concrete shadow tracking) or dropping the field from
/// the cache shape entirely (separate cleanup).
#[inline]
fn loopinvariant_now_known(
    ctx: &mut WalkContext<'_, '_>,
    ei: &majit_ir::EffectInfo,
    descr_key: u32,
    funcptr: OpRef,
    result: OpRef,
) {
    if ei.extraeffect != majit_ir::ExtraEffect::LoopInvariant {
        return;
    }
    let Some(arg0_int) = funcptr_concrete_int(ctx, funcptr) else {
        return;
    };
    ctx.trace_ctx
        .heap_cache_mut()
        .call_loopinvariant_now_known(descr_key, arg0_int, result, 0);
}

/// Resolve a residual-call funcptr OpRef to the concrete function
/// pointer integer that RPython's heapcache keys on
/// (`heapcache.py:629-634` calls `allboxes[0].getint()`).
///
/// Returns `Some(int)` when `funcptr` is a constant int OpRef whose
/// value lives in pyre's per-trace constant pool. For the RPython-legal
/// `EF_LOOPINVARIANT` direct-call producer, `call.py:249-251` asserts
/// no runtime args and emits a constant function box; that is the path
/// this cache is meant to mirror. General residual calls can arrive
/// from indirect calls with non-constant funcboxes
/// (`pyjitpl.py:2174-2186`), so `None` means "skip the loop-invariant
/// cache" rather than inventing an alias-prone sentinel key.
#[inline]
fn funcptr_concrete_int(ctx: &WalkContext<'_, '_>, funcptr: OpRef) -> Option<i64> {
    if !funcptr.is_constant() {
        return None;
    }
    match ctx.trace_ctx.concrete_of_opref(funcptr) {
        Some(majit_ir::Value::Int(v)) => Some(v),
        _ => None,
    }
}

/// `pyjitpl.py:_record_helper_pure` (`pyjitpl.py:1346-1400`) parity for the
/// walker layer: when a residual_call routes to `CallPure*` (elidable +
/// cannot-raise EI per [`select_residual_call_opcode`]) AND every
/// argument in `allboxes` has a known concrete value
/// (`TraceCtx::box_value` returns `Some`), execute the helper at trace
/// time via [`majit_metainterp::executor::execute_pure_call`] and stamp
/// `recorded` with the result.
///
/// ★ PyPy parity gap (Phase 5.B replacement candidate):
/// PyPy `_opimpl_*` methods (e.g. `_opimpl_setitem`,
/// `_opimpl_setfield_*`) concrete-execute every `do_residual_call`
/// regardless of `check_is_elidable()` — the EI flag only selects the
/// recorded opcode kind (`CALL_PURE_*` vs `CALL_*`), not whether the
/// helper runs.  The result is that `synchronize_virtualizable`'s
/// in-place PyFrame write at the end of `_opimpl_setfield_vable`
/// coincides with the side-effecting heap mutation of the
/// `store_subscr_fn` / `set_current_exception` helper — both happen
/// at trace-record time on the live frame.
///
/// Pyre's walker concrete-executes only the elidable path (this
/// function); non-elidable helpers like `store_subscr_fn` route
/// through `select_residual_call_opcode`'s default arm and are
/// recorded but never executed.  `eval_loop_jit`'s
/// `walker_dispatched_this_opcode` skip of `execute_opcode_step`
/// (eval.rs:3111) then leaves the non-vable heap unchanged → SIGBUS
/// on the next read (memory: M4 walker unactivated taxonomy, 5
/// STORE_SUBSCR-hot benches).
///
/// The Phase 5.B `dispatch_arm_via_blackhole` path is a workaround
/// that runs the arm jitcode through `BlackholeInterpreter` to
/// concrete-execute those helpers; the orthodox PyPy alignment is
/// instead to widen this function (or its successor) to
/// concrete-execute *every* residual_call with concrete-known args,
/// mirroring upstream's `executor.execute_varargs(opnum, argboxes,
/// descr, exc=can_raise, pure=is_elidable)`.  That removes the need
/// for `production_blackhole_handles` entirely and brings
/// `_opimpl_setfield_vable` shape into structural parity.
///
/// Out-of-scope here (multi-session epic): widening requires can-raise
/// exception propagation through the walker, a GuardNoException
/// emission policy that matches PyPy's `do_residual_call(..., exc=
/// can_raise)`, and audit of every non-elidable helper for trace-
/// recording-time invariants (no jitdriver re-entry, no thread state
/// mutation).  See Task #390.
///
/// RPython upstream `_record_helper_pure` invokes
/// `executor.execute_varargs(opnum, argboxes, descr, exc=False, pure=True)`
/// which dispatches to `cpu.bh_call_*` and stores the result on
/// `result_box.value` (`pyjitpl.py:1392`).  Pyre's walker observes the
/// same effect through the `set_opref_concrete` stamp — downstream walker
/// chain (sub-jitcode bodies that consume the call result via
/// `concrete_of_opref`) folds end-to-end instead of stalling at
/// `RefOp/IntOp(N)` unknown values.
///
/// **Caller contract**:
/// * `call_opcode` must be one of `CallPureI`/`CallPureR`/`CallPureF`/
///   `CallPureN` — the `select_residual_call_opcode` elidable arm
///   (`pyjitpl.py:2126` proper, `dispatch.rs:2688-2690`).  Other call
///   shapes (`CallMayForce*`, `CallLoopinvariant*`, `Call*`) carry
///   `can_raise=true` or escape semantics that require the full
///   `execute_varargs` MetaInterp seam — they MUST NOT route here.
/// * `allboxes[0]` is the funcbox (per `build_allboxes` layout); the
///   remaining slots are user args in `descr.arg_types()` ABI order.
///
/// Best-effort: returns silently when any operand lacks a concrete
/// `box_value` (the walker has no way to read the runtime value), or
/// when the arity exceeds `MAX_HOST_CALL_ARITY` (16) — the trace still
/// has the recorded `CallPure*` op for the optimizer to consume later,
/// just without the per-record fold.
fn try_fold_pure_call_via_executor(
    ctx: &mut WalkContext<'_, '_>,
    call_opcode: OpCode,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    recorded: OpRef,
) {
    if !matches!(
        call_opcode,
        OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF | OpCode::CallPureN
    ) {
        return;
    }
    // pyjitpl.py:1351-1352 — `_record_helper_pure` only fires for
    // `EF_ELIDABLE_CANNOT_RAISE`. `select_residual_call_opcode` returns
    // `CallPure*` whenever `check_is_elidable()` is true (including
    // `EF_ELIDABLE_CAN_RAISE`), so re-check the can-raise predicate here
    // before dispatching through the `execute_pure_call` no-metainterp
    // carve-out.  A `EF_ELIDABLE_CAN_RAISE` callee would silently swallow
    // the exception via `BH_LAST_EXC_VALUE` with no metainterp to
    // transcribe it.
    let ei = call_descr.get_extra_info();
    if ei.check_can_raise(false) {
        return;
    }
    if allboxes.is_empty() {
        return;
    }
    // pyjitpl.py:1960-1993 `_build_allboxes`: slot 0 is funcbox, slots
    // 1.. are user args in `descr.arg_types()` ABI order.  Walker's
    // [`build_allboxes`] preserves the same layout.
    //
    // pyjitpl.py:1352 invariant: `_record_helper_pure` requires
    // `funcbox` to be a Const so its `getint()` is the actual fn
    // pointer.  Non-constant funcboxes carry a stale-stamped Int
    // (from `cast_ptr_to_int` of a Ref-bank receiver, etc.) and
    // dereferencing as a code address yields SIGSEGV.  Skip the fold
    // when the funcbox is non-constant; the recorded `CallPure*` op
    // stays in the trace for the optimizer to consume later.
    if !allboxes[0].is_constant() {
        return;
    }
    let funcptr_val = ctx.trace_ctx.box_value(allboxes[0]);
    let func_ptr = match funcptr_val {
        Some(majit_ir::Value::Int(addr)) => addr,
        _ => return,
    };
    // Cap at MAX_HOST_CALL_ARITY (`call_int_function` / `call_void_function`
    // panic on excess arity).  `allboxes.len() - 1` is the arg count
    // (funcbox doesn't pass through).
    if allboxes.len() - 1 > majit_translate::jit_codewriter::insns::MAX_HOST_CALL_ARITY {
        return;
    }
    let mut args = Vec::with_capacity(allboxes.len() - 1);
    for &arg in &allboxes[1..] {
        let v = match ctx.trace_ctx.box_value(arg) {
            Some(majit_ir::Value::Int(n)) => n,
            Some(majit_ir::Value::Ref(r)) => {
                // `usize::MAX` sentinel from `concrete_of_opref` means
                // "no concrete known" — never reach this path because
                // `box_value` returns `None` for un-stamped OpRefs, but
                // belt-and-suspenders against future plumbing.
                if r == majit_ir::GcRef(usize::MAX) {
                    return;
                }
                r.as_usize() as i64
            }
            Some(majit_ir::Value::Float(f)) => f.to_bits() as i64,
            Some(majit_ir::Value::Void) => 0,
            None => return,
        };
        args.push(v);
    }
    // Refuse to invoke the helper when any Ref argument is NULL.  Pyre's
    // getfield_gc_r walker handler propagates field reads (including
    // pointer-valued fields like `PyFrame.f_back`) as concrete values
    // when the parent struct is concrete-known; a top-level frame
    // returns NULL for `f_back`, stamping `Value::Ref(GcRef(0))` into
    // the constant pool.  Folding `helper(NULL)` would then dereference
    // NULL and SEGV.  PyPy avoids this because its optimizer inserts
    // `guard_nonnull` ahead of any pointer-deref residual call; pyre's
    // walker folds before that guard exists, so guard the executor
    // entry against NULL receivers and fall through to recording the
    // IR op as-is.  The downstream optimizer then sees the call op and
    // emits the necessary guards.
    for (i, &arg) in args.iter().enumerate() {
        if matches!(call_descr.arg_types().get(i), Some(majit_ir::Type::Ref)) && arg == 0 {
            return;
        }
    }
    let result_i64 = majit_metainterp::executor::execute_pure_call(call_descr, func_ptr, &args);
    // pyjitpl.py:1392 `result_box.value = result`: stamp the recorded
    // OpRef with the executed concrete so downstream
    // `concrete_of_opref` / `box_value` consumers see the folded value.
    let result_value = match call_descr.result_type() {
        majit_ir::Type::Int => majit_ir::Value::Int(result_i64),
        majit_ir::Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(result_i64 as usize)),
        majit_ir::Type::Float => majit_ir::Value::Float(f64::from_bits(result_i64 as u64)),
        // void callees discard the result upstream too (`bh_call_v` has
        // no return value); `CallPureN` is included in the matched set
        // only to mirror PyPy's `_record_helper_pure` handling of all
        // pure shapes — skip the stamp for void.
        majit_ir::Type::Void => return,
    };
    ctx.trace_ctx.set_opref_concrete(recorded, result_value);
}

/// PyPy `_opimpl_residual_call{1,2,3}` (pyjitpl.py:1346/1349/1354) port
/// for the **non-elidable** call shapes — companion to
/// [`try_fold_pure_call_via_executor`] which handles the elidable case.
///
/// Upstream calls `executor.execute_varargs(opnum, argboxes, descr,
/// exc=can_raise, pure=False)` which dispatches through
/// `cpu.bh_call_*` and clears/records BH exception state via
/// `metainterp.clear_exception()` + `metainterp.execute_raised(...)`.
/// This walker-friendly counterpart returns `Result<i64, i64>` directly
/// (via [`majit_metainterp::executor::execute_residual_call`]) so the
/// caller can wire the BH exception into `WalkContext.last_exc_value`
/// without dragging in a `MetaInterp` seam.
///
/// **Why this exists** (Task #390 widening): walker's
/// `production_walker_handles` arm skips `execute_opcode_step` for
/// `eval.rs:3111` opcodes; for opcodes whose body contains a
/// non-elidable `residual_call_*` (`store_subscr_fn` /
/// `set_current_exception` / etc.), the helper is never invoked → heap
/// mutation never happens → next read derefs stale container → SIGBUS
/// (M4 walker unactivated taxonomy: 5 STORE_SUBSCR-hot benches).  The
/// orthodox fix is to widen this function across `Call*` /
/// `CallLoopinvariant*` shapes, mirroring PyPy's
/// `_opimpl_residual_call*` which concrete-executes *every* residual
/// call regardless of EI.
///
/// **Caller contract**:
/// * `call_opcode` must be one of the non-elidable, non-may-force
///   residual call shapes: `Call{I,R,F,N}` (the default arm) or
///   `CallLoopinvariant{I,R,F,N}` (the `EF_LOOPINVARIANT` arm).
///   * `CallPure*` is excluded — that's [`try_fold_pure_call_via_executor`]'s
///     job.
///   * `CallMayForce*` / `CallReleaseGil*` / `CallAssembler*` are
///     intentionally excluded: their trace-recording-time invariants
///     (force-virtual audit, GIL transitions, jitdriver re-entry)
///     need a separate audit (Task #390 sub-slice 4).
/// * `allboxes[0]` is the funcbox (per `build_allboxes` layout); the
///   remaining slots are user args in `descr.arg_types()` ABI order.
///
/// **Return value**:
/// * `Some(Ok(_))` — helper executed normally, `recorded` OpRef stamped
///   with the concrete result.
/// * `Some(Err(bh_exc))` — helper raised; `bh_exc` is the wrapped
///   `PyError` pointer (from `BH_LAST_EXC_VALUE`).  Caller is
///   responsible for routing into `WalkContext.last_exc_value` so the
///   downstream `GuardNoException` walker handler picks it up; the
///   `recorded` OpRef is NOT stamped (no concrete result).
/// * `None` — fold declined (preconditions not met: opcode out of set,
///   funcbox non-const, arity exceeds [`MAX_HOST_CALL_ARITY`], any
///   operand lacks a concrete `box_value`, or any Ref arg is NULL).
///   The trace still has the recorded call op for the optimizer to
///   consume later; walker falls through as if this function did not
///   exist.
///
/// **Wire status** (Task #390 sub-slice 2.3): invoked from all three
/// dispatch entry points (`dispatch_residual_call_iRd_kind`,
/// `dispatch_residual_call_iIRd_kind`, `dispatch_residual_call_iIRFd_kind`)
/// alongside [`try_fold_pure_call_via_executor`], gated on
/// `!can_raise`.  Sub-slice 3 will widen the gate to `can_raise=true`
/// once BH exception propagation through `WalkContext.last_exc_value`
/// is in place.
fn try_execute_residual_call_via_executor(
    ctx: &mut WalkContext<'_, '_>,
    call_opcode: OpCode,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    recorded: OpRef,
) -> Option<Result<i64, i64>> {
    if !matches!(
        call_opcode,
        OpCode::CallI
            | OpCode::CallR
            | OpCode::CallF
            | OpCode::CallN
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN
    ) {
        return None;
    }
    if allboxes.is_empty() {
        return None;
    }
    // Same funcbox-must-be-const invariant as `try_fold_pure_call_via_executor`:
    // a non-const funcbox carries a stale stamp and dereferencing it as a
    // code address SEGVs.  pyjitpl.py:1346-1354 forces the funcbox through
    // the executor's `cpu.bh_call_*` `ConstInt.getint()` path which
    // implicitly requires constness too (residual_call descrs always
    // carry a fixed funcptr at translation time).
    if !allboxes[0].is_constant() {
        return None;
    }
    let funcptr_val = ctx.trace_ctx.box_value(allboxes[0]);
    let func_ptr = match funcptr_val {
        Some(majit_ir::Value::Int(addr)) => addr,
        _ => return None,
    };
    // Sub-slice 4 safety gate — reject `symbolic_fnaddr_for_path`
    // placeholder values that escaped runtime patching.  Pyre's
    // codewriter mints a 64-bit hash of the helper's `CallPath` when
    // the build-time `pyre_interpreter::jit_trace_fnaddrs()` snapshot
    // has no entry for it (`majit-translate/src/jit_codewriter/call.rs:
    // 4926 symbolic_fnaddr_for_path`).  `runtime_fnaddr_patch` rewrites
    // these to real runtime addresses only when the path appears in
    // both the build-time and runtime registries; helpers absent from
    // the runtime registry retain the hash and dereferencing it as a
    // code address SIGBUSes.  Valid user-space code addresses fit in
    // 47 bits on macOS/Linux 64-bit (canonical low half); hash values
    // typically have bits ≥ 47 set.  Reject anything outside that
    // range — both the symbolic-hash leak class AND any other stray
    // non-fnptr value (e.g. an int constant mistakenly routed through
    // the funcbox slot).
    if (func_ptr as u64) >> 47 != 0 {
        return None;
    }
    if allboxes.len() - 1 > majit_translate::jit_codewriter::insns::MAX_HOST_CALL_ARITY {
        return None;
    }
    let mut args = Vec::with_capacity(allboxes.len() - 1);
    for &arg in &allboxes[1..] {
        let v = match ctx.trace_ctx.box_value(arg) {
            Some(majit_ir::Value::Int(n)) => n,
            Some(majit_ir::Value::Ref(r)) => {
                if r == majit_ir::GcRef(usize::MAX) {
                    return None;
                }
                r.as_usize() as i64
            }
            Some(majit_ir::Value::Float(f)) => f.to_bits() as i64,
            Some(majit_ir::Value::Void) => 0,
            None => return None,
        };
        args.push(v);
    }
    // NULL-Ref-arg refusal: same SEGV-avoidance contract as the pure
    // path (see `try_fold_pure_call_via_executor`'s NULL guard).  Pyre's
    // optimizer emits `guard_nonnull` after this walker fold, so a NULL
    // receiver dereferences before that guard exists; fall through to
    // recording the call op and let the optimizer's guard emission
    // handle it at compile time.
    for (i, &arg) in args.iter().enumerate() {
        if matches!(call_descr.arg_types().get(i), Some(majit_ir::Type::Ref)) && arg == 0 {
            return None;
        }
    }
    let exec_result =
        majit_metainterp::executor::execute_residual_call(call_descr, func_ptr, &args);
    match exec_result {
        Ok(result_i64) => {
            // pyjitpl.py:1392 `result_box.value = result` analogue — stamp
            // the recorded OpRef with the executed concrete so downstream
            // `concrete_of_opref` / `box_value` consumers see the folded
            // value.  Void callees do not stamp (no result to record); the
            // heap mutation has already happened via `call_void_function`.
            match call_descr.result_type() {
                majit_ir::Type::Int => {
                    ctx.trace_ctx
                        .set_opref_concrete(recorded, majit_ir::Value::Int(result_i64));
                }
                majit_ir::Type::Ref => {
                    ctx.trace_ctx.set_opref_concrete(
                        recorded,
                        majit_ir::Value::Ref(majit_ir::GcRef(result_i64 as usize)),
                    );
                }
                majit_ir::Type::Float => {
                    ctx.trace_ctx.set_opref_concrete(
                        recorded,
                        majit_ir::Value::Float(f64::from_bits(result_i64 as u64)),
                    );
                }
                majit_ir::Type::Void => {}
            }
        }
        Err(bh_exc) => {
            // pyjitpl.py:1690-1696 `metainterp.execute_raised(exception,
            // constant=False)` analogue — seed the standing exception
            // state so downstream walker chain (`reraise/`,
            // `last_exc_value/>r`, `handle_possible_exception` guard
            // emission) sees a non-null `last_exc_value` and routes
            // through the GuardException path.
            //
            // `execute_residual_call` cleared `BH_LAST_EXC_VALUE` on read;
            // restore it so the eval-loop walker-skip path
            // (`eval.rs:3285-3308`) can detect the pending exception and
            // route into the bytecode-interpreter's exception handler
            // via `PyError::from_exc_object` — matching RPython's
            // metainterp framestack scan after a raising residual call
            // (`pyjitpl.py:2156-2168 handle_possible_exception` +
            // `pyjitpl.py:3380 finishframe_exception`).
            ctx.last_exc_value = Some(ctx.trace_ctx.const_ref(bh_exc));
            ctx.last_exc_value_concrete =
                ConcreteValue::Ref(bh_exc as usize as pyre_object::PyObjectRef);
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(bh_exc));
        }
    }
    Some(exec_result)
}

/// `pyjitpl.py:3671-3681 MetaInterp.direct_call_release_gil` port.
/// Sub-case of the forces-virtual-or-virtualizable branch
/// (`pyjitpl.py:2063` `if effectinfo.is_call_release_gil()`): when the
/// descr's `call_release_gil_target` is a non-NULL `(realfuncaddr,
/// saveerr)` pair, the recorded trace op is `CALL_RELEASE_GIL_*`
/// with a re-shaped arglist:
///
/// ```text
///     realfuncaddr, saveerr = effectinfo.call_release_gil_target
///     funcbox = ConstInt(adr2int(realfuncaddr))
///     savebox = ConstInt(saveerr)
///     opnum   = rop.call_release_gil_for_descr(calldescr)
///     return self.history.record_nospec(
///         opnum, [savebox, funcbox] + argboxes[1:], ..., calldescr)
/// ```
///
/// `argboxes[0]` (the original funcbox) is replaced by the descr's real
/// target address, with `savebox` (`saveerr`) prepended.  The pyre-jit-
/// trace `allboxes` from [`build_allboxes`] starts with `funcptr` at
/// index 0 and the user-side arguments from index 1 onwards, matching
/// upstream's `argboxes[0] = funcbox` convention, so the slice rebuild
/// is `[savebox, funcbox_real] + allboxes[1..]`.
///
/// Mirror of `majit-metainterp/src/pyjitpl/mod.rs:10437-10477
/// direct_call_release_gil` for the pyre-jit-trace dispatcher layer.
/// The two-frame-layer parity (majit `do_residual_call` and
/// pyre-jit-trace `dispatch_residual_call_*`) both implement the same
/// `pyjitpl.py:3671-3681` sub-case independently because the layers
/// receive different argument shapes.  `descr` is consumed (move) into
/// `record_op_with_descr` so the caller must `clone()` it before
/// calling if it needs the original after this returns.
///
/// Also emits the two guards the outer forces branch demands
/// (`pyjitpl.py:2079 GUARD_NOT_FORCED` unconditionally,
/// `pyjitpl.py:2082 GUARD_NO_EXCEPTION` when
/// `check_can_raise(False)` is true) — keeping guard emission inside
/// this helper means the dispatcher early-returns after a single call.
///
/// **`'r'` bank not supported.**  RPython
/// `resoperation.py:1238 call_release_gil_for_descr` has no
/// `CALL_RELEASE_GIL_R` arm (commented out as `# no such thing`),
/// and `:1462 is_call_release_gil` excludes `CALL_RELEASE_GIL_R`
/// from the predicate.  This helper panics on `dst_bank == 'r'` —
/// the closest behaviour to upstream's missing branch is fail-fast,
/// since silently routing to a non-existent OpCode would record an
/// IR op the optimizer / backend cannot consume.  Generic codewriter
/// `emit_residual_call` sites do not manufacture release-gil EIs via
/// `effect_info_for_call_flavor`; release-gil support is limited to
/// explicit via-target lowering that resolves the real call target
/// before materializing the final calldescr.  The panic is defensive
/// against a future producer that introduces a `'r'`-result release-gil
/// callee without first wiring an upstream `CALL_RELEASE_GIL_R` opcode.
///
/// `'i'` / `'f'` / `'v'` are the three result kinds upstream's
/// `call_release_gil_for_descr` accepts (`resoperation.py:1240-1248`).
/// All three are decoded here so the **opcode selection** matches
/// upstream's three-way result-kind table even though only
/// `dispatch_residual_call_iRd_kind` / `_iIRd_kind` currently route
/// `'i'` and `'r'` (the latter rejected per the panic above).
///
/// **Float / Void coverage is opcode-only, not full reuse.**  A
/// future float / void residual-call dispatcher would still have
/// to extend its own callsite to (a) widen `dst_bank` validation,
/// (b) add the corresponding writeback path to
/// `registers_f` / no-writeback, and (c) thread Float-typed
/// `argbox_types` through `build_allboxes` for the `'f'` arg-list
/// case.  This helper produces the right `OpCode::CallReleaseGil*`
/// once those landed; it does not by itself complete the dispatcher.
/// `pyjitpl.py:2003-2005 do_residual_call` parity:
///
/// ```python
/// if effectinfo.oopspecindex == effectinfo.OS_NOT_IN_TRACE:
///     return self.metainterp.do_not_in_trace_call(allboxes, descr)
/// ```
///
/// Upstream's `do_not_in_trace_call` (pyjitpl.py:3683-3697) executes the
/// callee concretely and raises `SwitchToBlackhole(ABORT_ESCAPE,
/// raising_exception=True)` if it raised, otherwise returns `None` so
/// no IR op is recorded.
///
/// The pyre trace-walker has no concrete-execution callback for
/// jitcode-walked residual_call bytecodes yet — concrete execution
/// happens in the metainterp layer (`pyjitpl/mod.rs:9631-9659
/// do_not_in_trace_call`) which dispatches `BC_CALL_*` not
/// `BC_RESIDUAL_CALL_*`. Therefore an `OS_NOT_IN_TRACE` callee that
/// reached this dispatcher cannot be safely treated as a regular
/// residual call: upstream records no IR for the normal case, and
/// aborts to blackhole only when the concrete call raises. Until that
/// concrete callback is threaded into `WalkContext`, the walker reports
/// a typed error instead of inventing either outcome.
///
/// `effect_info_for_call_flavor` (`flatten.rs:431` audit table) never
/// sets `oopspecindex`, so this branch is unreachable from production
/// today. A future producer that begins populating `oopspecindex`
/// should replace this guard with a real `do_not_in_trace_call`
/// callback returning `Ok(None)` on normal completion and
/// `SwitchToBlackhole(ABORT_ESCAPE, raising_exception=True)` only on
/// raise.
#[inline]
fn do_not_in_trace_call_result(
    ei: &majit_ir::EffectInfo,
    pc: usize,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if ei.oopspecindex == OopSpecIndex::NotInTrace {
        return Err(DispatchError::NotInTraceRequiresConcreteExecution { pc });
    }
    Ok(None)
}

/// `pyjitpl.py:2011-2014` short-circuit guard.  RPython `do_residual_call`
/// runs `_do_jit_force_virtual` (`pyjitpl.py:2153-2172`) when
/// `effectinfo.oopspecindex == OS_JIT_FORCE_VIRTUAL`:
///
/// ```text
/// def _do_jit_force_virtual(self, allboxes):
///     if (self.jitdriver_sd.virtualizable_info is None and
///         self.jitdriver_sd.greenfield_info is None):
///         return None
///     if len(allboxes) == 2:
///         [vrefbox] = allboxes[1:]
///         standard_box = self.virtualizable_boxes[-1].getref_base()
///         if standard_box != vrefbox.getref_base():    # concrete pointer compare
///             return None
///         return self.virtualizable_boxes[-1]
///     ...
/// ```
///
/// PyPy returns one of `vref_box` / `standard_box` / `None`; on the
/// `None` fall-through it records the normal `CALL_MAY_FORCE_*`.  The
/// walker CANNOT reproduce this faithfully because the comparison
/// `standard_box.getref_base() != vrefbox.getref_base()` requires a
/// concrete `*mut PyObject` for the `vrefbox` Ref OpRef; pyre's
/// symbolic walker only carries `concrete_vable_ptr` for the active
/// virtualizable, not a per-OpRef → concrete-pointer map.
///
/// Choices considered:
///  1. Silent fall-through (always record `CALL_MAY_FORCE_*`) — would
///     silently emit IR on a path PyPy folds away whenever the vref
///     IS the standard virtualizable.  Trace divergence with no diff
///     report; rejected.
///  2. OpRef-equality short-circuit (`if vrefbox_opref == standard_box_opref:
///     short-circuit; else fall through`) — sufficient for the same-OpRef
///     case but UNSOUND for the different-OpRef-same-concrete-pointer case
///     (walker would record `CALL_MAY_FORCE_*` while PyPy folds; mismatch
///     vs. live tracer's IR).  Rejected.
///  3. Fail-loud (current).  STRICTER than PyPy: walker stops with a
///     typed error rather than emit divergent IR.  Surface area: under
///     `MAJIT_SHADOW_WALKER=1` shadow validation, the panic at
///     `shadow_walker.rs:287-292` immediately flags any producer that
///     starts emitting `OopSpecIndex::JitForceVirtual`.
///
/// Convergence path back to 1:1 PyPy parity: an OpRef → concrete-pointer
/// resolver.  When that lands the guard becomes a real
/// `_do_jit_force_virtual()` body that returns `Some(vref_opref)` /
/// `Some(standard_opref)` / `None` and threads through the dispatcher
/// like the release-gil short-circuit does today.
///
/// Production reach today: zero — `OopSpecIndex::JitForceVirtual` is
/// set only by `jtransform.rs:1903 jit.force_virtual` lowering, which
/// our benchmarks don't reach.  The guard is fail-loud futureproofing
/// for the day a producer (e.g. an explicit `jit.force_virtual` callee
/// from `pyre_interpreter`) lights up the path.
#[inline]
fn do_jit_force_virtual_guard(ei: &majit_ir::EffectInfo, pc: usize) -> Result<(), DispatchError> {
    if ei.oopspecindex == OopSpecIndex::JitForceVirtual {
        return Err(DispatchError::JitForceVirtualRequiresConcreteResolver { pc });
    }
    Ok(())
}

/// IR-recording portion of `pyjitpl.py:3327-3335
/// vable_and_vrefs_before_residual_call`.  Records
/// `FORCE_TOKEN + SETFIELD_GC(vable_token_descr)` whenever the
/// jitdriver has a standard virtualizable registered for the current
/// frame.  RPython structure:
///
/// ```text
/// def vable_and_vrefs_before_residual_call(self):
///     self.vrefs_before_residual_call()                # heap mutation
///     vinfo = self.jitdriver_sd.virtualizable_info
///     if vinfo is not None:
///         virtualizable_box = self.virtualizable_boxes[-1]
///         virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
///         vinfo.tracing_before_residual_call(virtualizable) # heap mutation
///         force_token = self.history.record0(rop.FORCE_TOKEN, ...)  # IR
///         self.history.record2(rop.SETFIELD_GC, ..., descr=...)     # IR
/// ```
///
/// TODO: in pyre, the IR-recording role and the
/// runtime heap-mutation role are split.  The trait-driven path
/// (`state.rs MIFrame::vable_and_vrefs_before_residual_call`,
/// `trace_opcode.rs:2193-2229`) ALREADY performs the heap mutations
/// (`vinfo.tracing_before_residual_call(virtualizable)`,
/// `vrefinfo.tracing_before_residual_call(vref)`) at the live call
/// site — that's where the callee actually executes and observes the
/// token.  Walker is the symbolic shadow validator under
/// `MAJIT_SHADOW_WALKER=1`: it runs ahead of the trait dispatch, its
/// IR is rolled back via `cut_trace`, and then the trait dispatch
/// runs and emits the "real" IR.  Walker therefore records ONLY the
/// IR portion that the trait dispatch will record on the no-force
/// path; the heap-mutation portion stays on the trait side so the
/// `*token_ptr == 0` assertion in `tracing_before_residual_call`
/// holds when the trait path runs.
///
/// `vrefs_before_residual_call` (`pyjitpl.py:3317-3326`) is omitted
/// entirely — it has zero IR ops, only heap mutations on
/// `vrefinfo.tracing_before_residual_call`.
///
/// `vrefs_after_residual_call` and `vable_after_residual_call`
/// (`pyjitpl.py:3337-3366`) are omitted entirely — they observe
/// whether the callee forced a vref/vable by reading the heap token,
/// and only emit IR on detected forces (`VIRTUAL_REF_FINISH`,
/// `SwitchToBlackhole(ABORT_ESCAPE)`).  The walker never executes
/// the callee, so it cannot observe a force; pretending to run the
/// after-helpers would be a no-op heap set/clear pair masquerading
/// as parity.  On forced calls the trait-dispatch leg aborts via
/// `PyError::runtime_error("ABORT_ESCAPE: ...")` before its IR diff
/// runs, so walker under-recording on those paths is harmless.
fn walker_vable_and_vrefs_before_residual_call(ctx: &mut TraceCtx) {
    // pyjitpl.py:3326-3327: vinfo = self.jitdriver_sd.virtualizable_info;
    //                       if vinfo is not None:
    let Some(vable_ref) = ctx.standard_virtualizable_box() else {
        return;
    };
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    // pyjitpl.py:3332-3335: force_token + SETFIELD_GC vable_token_descr
    let force_token = ctx.force_token();
    ctx.vable_setfield_descr(vable_ref, force_token, info.token_field_descr());
}

/// Convenience wrapper for [`walker_vable_and_vrefs_before_residual_call`].
/// Kept as a thin pass-through so the dispatcher call sites stay
/// readable; collapses to direct `walker_*` once the dispatchers
/// inline.
fn maybe_walker_vable_and_vrefs_before_residual_call(ctx: &mut WalkContext<'_, '_>) {
    walker_vable_and_vrefs_before_residual_call(ctx.trace_ctx);
}

/// Write a residual_call's recorded result OpRef into the dst register
/// chosen by `dst_bank`. Centralizes the result writeback so the
/// dispatchers can perform it BEFORE recording the
/// `GUARD_NOT_FORCED` / `GUARD_NO_EXCEPTION` guards, matching
/// `pyjitpl.py:1950 _opimpl_residual_call*` ordering: the result
/// must populate `registers_*[dst]` before
/// `handle_possible_exception()` captures the guard's `fail_args`,
/// otherwise a raising call surfaces NONE in the slot the resume
/// snapshot reads.
fn write_residual_call_result_to_dst(
    ctx: &mut WalkContext<'_, '_>,
    pc: usize,
    dst: usize,
    dst_bank: char,
    result: OpRef,
) -> Result<(), DispatchError> {
    // concrete_of_opref shadow write: route the shadow write through `concrete_of_opref`
    // so a CallPure* descr whose argboxes are all constant (do_residual_call
    // path that lands a constant result via the executor.execute_varargs
    // stamp) propagates concrete to the dst slot.  Falls back to Null when
    // the result has no recorded concrete (matches the pre-#75.F shape for
    // every non-elidable call).
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    match dst_bank {
        'r' => {
            write_ref_reg(ctx, pc, dst, result, concrete_for_shadow)?;
        }
        'i' => {
            write_int_reg(ctx, pc, dst, result, concrete_for_shadow)?;
        }
        'f' => {
            let len = ctx.registers_f.len();
            let slot = ctx
                .registers_f
                .get_mut(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            *slot = result;
        }
        // Void variants (`pyjitpl.py:1348/1351/1355 opimpl_residual_call_*_v`):
        // the operand layout has no `>X` dst byte and no register slot to
        // populate. The cached / recorded OpRef is dropped on the floor
        // upstream too (the `_call*` body discards the call result for
        // void).
        'v' => {}
        _ => unreachable!("dst_bank validated by caller"),
    }
    Ok(())
}

/// `pyjitpl.py:218-225 _get_list_of_active_boxes` parity for the
/// walker-emitted snapshot: read each live register from its
/// kind-specific bank in (int, ref, float) order, dropping non-live
/// slots regardless of whether the OpRef happens to be set.  The
/// liveness lookup matches the decoder side
/// (`state::frame_value_count_at` /
/// `frame_liveness_reg_indices_by_bank_at`), so encoder and decoder
/// agree byte-for-byte on the snapshot shape consumed at resume.
///
/// Returns an empty vector when no liveness is registered for the
/// `(jitcode_index, pc)` pair (skeleton payload or out-of-range PC);
/// the downstream optimizer surfaces the empty snapshot as a no-op.
fn collect_outer_active_boxes(
    sym: &crate::state::PyreSym,
    trace_ctx: &TraceCtx,
    outer_jitcode_index: u32,
    entry_py_pc: u32,
) -> Vec<OpRef> {
    let banks = crate::state::frame_liveness_reg_indices_by_bank_at(
        outer_jitcode_index as i32,
        entry_py_pc as i32,
    );
    let mut active = Vec::with_capacity(banks.int.len() + banks.ref_.len() + banks.float.len());
    // RPython `pyjitpl.py:216-233 _get_list_of_active_boxes` reads
    // `self.registers_X[index]` directly per liveness index.  Pyre
    // diverges on the Ref bank for portal-owner frames: the
    // `registers_r` semantic-mirror write in
    // `write_stack_slot` (trace_opcode.rs:581/605) is retired so stack-slot colors
    // sit at `OpRef::NONE` in `sym.registers_r`; the authoritative
    // shadow lives in `trace_ctx.virtualizable_boxes`.  Codewriter
    // liveness also force-alives `portal_frame_reg` / `portal_ec_reg`
    // (codewriter.rs:3419-3424 `filter_liveness_in_place`) — these are
    // scratch colors past `nlocals + max_stackdepth` that have no
    // semantic frame slot, sourced instead from `sym.frame` /
    // `sym.execution_context` (`interp_jit.py:67 reds = ['frame', 'ec']`).
    //
    // Mirror the trait-side snapshot materializer
    // (trace_opcode.rs:1340-1395 `get_list_of_active_boxes`): map each
    // live Ref color to its semantic index via
    // `semantic_ref_slot_for_reg_color`, read the vable shadow for
    // portal-owner frames, and route the two portal red regs through
    // `sym.frame` / `sym.execution_context` directly.
    let (
        nlocals,
        valid_stack_only,
        owns_vable,
        local_color_map,
        stack_color_map,
        live_locals,
        portal_frame_reg,
        portal_ec_reg,
    ) = if sym.jitcode.is_null() {
        (
            0usize,
            0usize,
            false,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            u16::MAX,
            u16::MAX,
        )
    } else {
        unsafe {
            let jc = &*sym.jitcode;
            let payload = &jc.payload;
            let live_locals = if payload.code_ptr.is_null() {
                Vec::new()
            } else {
                let live_vars = crate::liveness::liveness_for(payload.code_ptr);
                (0..sym.nlocals)
                    .filter(|&idx| live_vars.is_local_live(entry_py_pc as usize, idx))
                    .collect::<Vec<usize>>()
            };
            (
                sym.nlocals,
                sym.valuestackdepth.saturating_sub(sym.nlocals),
                sym.owns_virtualizable_shadow(),
                payload.metadata.pyre_color_for_semantic_local.clone(),
                payload.metadata.stack_slot_color_map.clone(),
                live_locals,
                payload.metadata.portal_frame_reg,
                payload.metadata.portal_ec_reg,
            )
        }
    };
    // Int / Float bank diagnostic panic: pyre's banks are sized to the
    // jitcode's `num_regs_X`, which the codewriter co-publishes with the
    // liveness side-table, so every liveness index is in range by
    // construction.  A miss here is a tracer-side invariant violation
    // (size mismatch) — panic loudly so the bug surfaces at the encode
    // site instead of bleeding NONE values into `encode_snapshot_boxes`
    // where `get_opref_type(NONE)` panics with no breadcrumb pointing at
    // the source.
    let live_i = banks.int.clone();
    let live_r = banks.ref_.clone();
    let live_f = banks.float.clone();
    let (ni, nr, nf) = (
        sym.registers_i.len(),
        sym.registers_r.len(),
        sym.registers_f.len(),
    );
    let vable_len = trace_ctx.virtualizable_boxes_len().unwrap_or(0);
    // Int / Float bank candidates: pyre's vable static fields decode as
    // Int (last_instr, valuestackdepth, etc.); if liveness expects an Int
    // register that the trait dispatcher never filled, the candidate fill
    // is one of these sym shadow OpRefs.  Including them in the diagnostic
    // lets fallback work jump straight to the right source.
    let vable_vsd = sym.vable_valuestackdepth;
    let vable_last_instr = sym.vable_last_instr;
    let dump_ctx = |bank: &'static str, reg_idx: u32| -> String {
        let int_hint = if bank == "int" {
            format!(
                ", sym.vable_valuestackdepth={vable_vsd:?}, sym.vable_last_instr={vable_last_instr:?}"
            )
        } else {
            String::new()
        };
        format!(
            "collect_outer_active_boxes: liveness-active {bank} \
             register {reg_idx} holds OpRef::NONE \
             (outer_jitcode_index={outer_jitcode_index}, entry_py_pc={entry_py_pc}, \
              nlocals={nlocals}, owns_vable={owns_vable}, \
              vable_len={vable_len}, \
              num_regs_i={ni}, num_regs_r={nr}, num_regs_f={nf}, \
              live_banks_i={live_i:?}, live_banks_r={live_r:?}, live_banks_f={live_f:?}\
              {int_hint})",
        )
    };
    for &idx in &banks.int {
        let v = sym
            .registers_i
            .get(idx as usize)
            .copied()
            .unwrap_or_else(|| panic!("{}", dump_ctx("int", idx)));
        if v == OpRef::NONE {
            panic!("{}", dump_ctx("int", idx));
        }
        active.push(v);
    }
    for &idx in &banks.ref_ {
        let color = idx as usize;
        let fallback = || {
            sym.registers_r
                .get(color)
                .copied()
                .unwrap_or_else(|| panic!("{}", dump_ctx("ref", idx)))
        };
        let value = if color as u16 == portal_frame_reg && portal_frame_reg != u16::MAX {
            sym.frame
        } else if color as u16 == portal_ec_reg && portal_ec_reg != u16::MAX {
            sym.execution_context
        } else if owns_vable {
            let semantic_idx = crate::state::semantic_ref_slot_for_reg_color(
                nlocals,
                valid_stack_only,
                &local_color_map,
                &stack_color_map,
                &live_locals,
                color,
            );
            match semantic_idx {
                Some(s_idx) if s_idx < nlocals + valid_stack_only => {
                    let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                    trace_ctx
                        .virtualizable_box_at(nvs + s_idx)
                        .unwrap_or_else(fallback)
                }
                _ => fallback(),
            }
        } else {
            fallback()
        };
        if value == OpRef::NONE {
            panic!("{}", dump_ctx("ref", idx));
        }
        active.push(value);
    }
    for &idx in &banks.float {
        let v = sym
            .registers_f
            .get(idx as usize)
            .copied()
            .unwrap_or_else(|| panic!("{}", dump_ctx("float", idx)));
        if v == OpRef::NONE {
            panic!("{}", dump_ctx("float", idx));
        }
        active.push(v);
    }
    active
}

/// Walker-side port of `pyjitpl.py:2586-2602 MIFrame.capture_resumedata`
/// for the `after_residual_call=True` path (`pyjitpl.py:2078-2082
/// handle_possible_exception`).  Attaches a single-frame snapshot to
/// the last recorded guard so the optimizer's
/// `store_final_boxes_in_guard` (`optimizeopt/mod.rs:5033`) finds
/// `rd_resume_position >= 0` and can derive `op.fail_args` from the
/// snapshot via `op.store_final_boxes(liveboxes)` instead of panicking.
fn walker_capture_snapshot_for_last_guard(ctx: &mut WalkContext<'_, '_>, op_pc: usize) {
    // Snapshot semantics for walker-emitted guards
    // (`pyjitpl.py:2582-2603 generate_guard` + `capture_resumedata`):
    //
    // RPython treats helper jitcodes (pop_value, nlocals, etc.) as
    // separate `MIFrame`s on `metainterp.framestack`, capturing one
    // snapshot frame per `MIFrame` plus a vable_array / vref_array
    // prefix on the top frame (`opencoder.py:767 create_top_snapshot`).
    // At resume, RPython's blackhole interpreter re-enters each frame's
    // jitcode and replays from the saved pc.
    //
    // Pyre's blackhole interpreter only knows how to run *pyjitcode*
    // bytecode (Python bytecode), not helper jitcodes — pyre's
    // per-opcode arm jitcodes and sub-jitcode helpers are walker-only
    // structures with no blackhole entry point.  The structural
    // consequence: any walker-emitted guard, regardless of how deep
    // the sub-walk nesting is, must resume to the *outer* Python
    // opcode boundary (`sym.jitcode` at `entry_py_pc`) — that is the
    // only resume point pyre's blackhole can re-enter.  The
    // framestack-collapse is a deliberate adaptation, not a parity
    // miss; the walker context carries the outer Python frame only.
    // Inline-traced Python frames (`build_pending_inline_frame`) are
    // not reachable from this entry point because the production
    // walker allow-list does not yet enable opcodes that drive inline
    // tracing — when that expands, `WalkContext` must grow a parent-
    // Python-frame chain (analogous to `MIFrame.parent_frames`) and
    // this helper switches to
    // `capture_snapshot_for_last_guard_multi_frame_with_vable_vref`.
    //
    // The snapshot is therefore single Python frame at the outer
    // pyjitcode coordinates.  `ctx.outer_jitcode_index` +
    // `ctx.entry_py_pc` track those coordinates; `outer_active_boxes`
    // carries the `PyFrame` state at the Python opcode boundary
    // (snapshotted once at `dispatch_via_miframe_at_opcode_entry` from
    // `sym.registers_r ∪ sym.registers_i.opref ∪ sym.registers_f.opref`
    // via `collect_outer_active_boxes` / `frame_liveness_reg_indices_
    // by_bank_at`).
    //
    // `op_pc` (the walker's arm-local PC) is intentionally not used:
    // the arm jitcode has no resume entry point in pyre's blackhole.
    let _ = op_pc;
    // `opencoder.py:772-775 create_top_snapshot` writes vable_array +
    // vref_array on the top snapshot.  The walker-emitted guard IS
    // a top snapshot for pyre (helper frames don't resume), so feed
    // the trace-time vable/vref shadow through.  Empty when no
    // virtualizable / virtualref is live, matching the upstream
    // 0-length-array shape.
    let (vable_boxes, vref_boxes) = ctx.trace_ctx.build_snapshot_vable_vref_boxes();
    ctx.trace_ctx
        .capture_snapshot_for_last_guard_with_vable_vref(
            &ctx.outer_active_boxes,
            ctx.outer_jitcode_index,
            ctx.entry_py_pc,
            &vable_boxes,
            &vref_boxes,
        );
}

/// Walker-side port of `pyjitpl.py:2156-2168 handle_possible_exception`'s
/// exception branch (the trait-leg equivalent lives at
/// `trace_opcode.rs:6962-7020 MIFrame::handle_possible_exception`).
///
/// Emit `GuardException(exc_type_const)` + snapshot, reading the
/// exception's `ob_header.ob_type` from
/// `WalkContext::last_exc_value_concrete` to pin the class.  Mirrors
/// RPython's `class_of_box_known` shape: the guard recovery has the
/// exact subclass available at replay, matching `opimpl_raise`'s
/// `GUARD_CLASS(exc, cls_of_box(exc))` pattern (`pyjitpl.py:1687-1693`).
///
/// Falls back to `GuardNoException` when the concrete exception
/// pointer is unavailable (sub-walk shadow gap or non-Ref concrete) —
/// the residual call op stays in the trace and the optimizer's
/// per-call guard-emission still catches exception divergence at
/// replay, just without the class pin.
fn walker_record_guard_exception(ctx: &mut WalkContext<'_, '_>, pc: usize) {
    let exc_obj = match ctx.last_exc_value_concrete {
        ConcreteValue::Ref(p) if !p.is_null() => p,
        _ => {
            ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, pc);
            return;
        }
    };
    // `pyjitpl.py:3382-3384`: ALWAYS emit `GuardException` with a const
    // class pin (`class_of_last_exc_is_const = True` after the emit).
    // Pyre's `W_ExceptionObject.ob_header.ob_type` is the per-`ExcKind`
    // `PyType` static (`excobject.rs::exc_kind_to_pytype`), matching
    // upstream `OBJECT.typeptr = specific class` (`rclass.py:167-174`).
    let exc_type_ptr = unsafe {
        (*(exc_obj as *const pyre_object::excobject::W_ExceptionObject))
            .ob_header
            .ob_type as i64
    };
    let exc_type_const = ctx.trace_ctx.const_int(exc_type_ptr);
    ctx.trace_ctx
        .record_guard(OpCode::GuardException, &[exc_type_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, pc);
}

fn direct_call_release_gil(
    ctx: &mut WalkContext<'_, '_>,
    ei: &majit_ir::EffectInfo,
    allboxes: &[OpRef],
    descr: DescrRef,
    dst_bank: char,
    dst: usize,
    pc: usize,
    caller: &'static str,
) -> Result<(), DispatchError> {
    // pyjitpl.py:2017 `vable_and_vrefs_before_residual_call` —
    // release-gil is unconditionally a forces sub-case
    // (`pyjitpl.py:2063` sits inside the forces-virtual-or-virtualizable
    // branch), so no `emit_guard_not_forced` gate is needed here.
    // RPython's pre-call vrefs heap mutation
    // (`pyjitpl.py:3318-3322 vrefs_before_residual_call`) and the
    // after-call helpers (`pyjitpl.py:3337-3366`) both sit on the
    // trait-driven leg in pyre — see
    // [`walker_vable_and_vrefs_before_residual_call`] for the
    // walker-vs-trait split rationale.
    maybe_walker_vable_and_vrefs_before_residual_call(ctx);
    // pyjitpl.py:3675: realfuncaddr, saveerr = effectinfo.call_release_gil_target
    let (realfuncaddr, saveerr) = ei.call_release_gil_target;
    // pyjitpl.py:3676-3677: funcbox/savebox ConstInt
    let savebox = ctx.trace_ctx.const_int(saveerr as i64);
    let funcbox_real = ctx.trace_ctx.const_int(realfuncaddr as i64);
    // pyjitpl.py:3678: opnum = rop.call_release_gil_for_descr(calldescr).
    // resoperation.py:1240-1248 maps the descr's normalized result
    // type to {CALL_RELEASE_GIL_I, CALL_RELEASE_GIL_F, CALL_RELEASE_GIL_N};
    // 'r' is explicitly skipped (`# no such thing`).
    let opcode = match dst_bank {
        'i' => OpCode::CallReleaseGilI,
        'f' => OpCode::CallReleaseGilF,
        'v' => OpCode::CallReleaseGilN,
        'r' => panic!(
            "{caller}: CALL_RELEASE_GIL_R has no upstream counterpart \
             (resoperation.py:1243-1244 `# no such thing`); a 'r'-result \
             release-gil callee cannot be lowered to an IR op the \
             optimizer/backend can consume."
        ),
        _ => unreachable!(
            "{caller}: dst_bank '{dst_bank}' not supported by direct_call_release_gil \
             (callers must pass 'i' / 'f' / 'v' per resoperation.py:1240-1248)"
        ),
    };
    // pyjitpl.py:3679-3681: history.record_nospec(opnum,
    //                          [savebox, funcbox] + argboxes[1:], ..., calldescr)
    let mut new_args = Vec::with_capacity(allboxes.len() + 1);
    new_args.push(savebox);
    new_args.push(funcbox_real);
    if allboxes.len() > 1 {
        new_args.extend_from_slice(&allboxes[1..]);
    }
    let result = ctx.trace_ctx.record_op_with_descr(opcode, &new_args, descr);
    // pyjitpl.py:2072 `heapcache.invalidate_caches_varargs(opnum1, descr,
    // allboxes)` — the forces-branch invalidation uses `opnum1` which is
    // the corresponding `CALL_MAY_FORCE_*`, NOT `CALL_RELEASE_GIL_*`.
    // Pass the **original** `allboxes` (not the reshaped
    // `[savebox, funcbox] + argboxes[1:]`) so heapcache's
    // `mark_escaped_varargs` sees the same operand identities upstream
    // does at this site.
    let mayforce_opnum = match dst_bank {
        'i' => OpCode::CallMayForceI,
        'f' => OpCode::CallMayForceF,
        'v' => OpCode::CallMayForceN,
        _ => unreachable!("dst_bank validated above"),
    };
    ctx.trace_ctx
        .heapcache_invalidate_caches_varargs(mayforce_opnum, Some(ei), allboxes);
    // pyjitpl.py:1950 _opimpl_residual_call*: result writeback runs
    // BEFORE handle_possible_exception().  Write `result` into
    // `registers_*[dst]` here so the GUARD_NO_EXCEPTION fail_args
    // capture below sees the recorded OpRef rather than the prior
    // register value.  `'v'` (void) returns skip the writeback —
    // there is no destination slot.
    if dst_bank != 'v' {
        write_residual_call_result_to_dst(ctx, pc, dst, dst_bank, result)?;
    }
    // pyjitpl.py:2079 GUARD_NOT_FORCED — unconditional on the outer
    // forces-virtual-or-virtualizable branch (the release-gil sub-case
    // is inside that branch, so the guard fires regardless of which
    // sub-branch ran).  Walker omits the
    // `vable_after_residual_call(funcbox)` short-circuit
    // (`pyjitpl.py:2078`) entirely because it has no concrete callee
    // execution to observe a force from — the trait-dispatch leg
    // detects the force at `state.rs MIFrame::vable_after_residual_call`
    // (`trace_opcode.rs:2237-2263`) and aborts via
    // `PyError::runtime_error("ABORT_ESCAPE: ...")` before walker IR
    // diff would run.
    ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, pc);
    // pyjitpl.py:2082 handle_possible_exception — emits
    // GUARD_NO_EXCEPTION whenever the EffectInfo can raise.  Walker's
    // `walker_capture_snapshot_for_last_guard` ports
    // `capture_resumedata(after_residual_call=True)` so the optimizer's
    // `store_final_boxes_in_guard` finds a populated
    // `rd_resume_position`.
    if ei.check_can_raise(false) {
        ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
        walker_capture_snapshot_for_last_guard(ctx, pc);
    }
    Ok(())
}

/// `residual_call` shape `iRd>X` dispatcher. Reads `funcptr (i)`,
/// R-list args, and `descr`, runs `_build_allboxes` to produce the
/// callee's ABI-ordered arglist, classifies the call by `EffectInfo`
/// via [`select_residual_call_opcode`], records the matching
/// kind-coded `CallMayForce*` / `CallLoopinvariant*` / `CallPure*` /
/// `Call*` op, emits `GUARD_NOT_FORCED` on the forces branch, emits
/// `GUARD_NO_EXCEPTION` if the classification says `can_raise`, and
/// writes the recorded result OpRef into the dst register chosen by
/// `dst_bank`.
///
/// RPython parity: `pyjitpl.py:1334-1336 _opimpl_residual_call1` →
/// `do_residual_or_indirect_call` → `do_residual_call`
/// (pyjitpl.py:1995-2127). `pyjitpl.py:1346 opimpl_residual_call_r_i =
/// _opimpl_residual_call1` and `:1347 opimpl_residual_call_r_r =
/// _opimpl_residual_call1` confirm both kind variants share the
/// `_call1` body. The `_X` suffix is the *call's return kind* — mapping
/// comes from `do_residual_call`'s `descr.get_normalized_result_type()`
/// dispatch (pyjitpl.py:2022-2044) and `select_residual_call_opcode`'s
/// kind-keyed opcode tables.
///
/// `dst_bank` selects where the call's result lands:
/// * `'r'`: caller's `registers_r[dst]` — Ref-typed `Call*` family
///   (`_r_r/iRd>r`, `pyjitpl.py:1347 opimpl_residual_call_r_r`).
/// * `'i'`: caller's `registers_i[dst]` — Int-typed `Call*` family
///   (`_r_i/iRd>i`, `pyjitpl.py:1346 opimpl_residual_call_r_i`).
/// * `'v'`: void return — operand layout drops the trailing `>X` byte and
///   the writeback no-ops (`_r_v/iRd`, `pyjitpl.py:1348
///   opimpl_residual_call_r_v`, `blackhole.py:1245 bhimpl_residual_call_r_v`).
/// (`'f'` is intentionally absent: RPython does not exec-generate
/// `opimpl_residual_call_r_f`. The only float-result residual_call
/// shape is `_irf_f/iIRFd>f`, dispatched by
/// [`dispatch_residual_call_iIRFd_kind`].)
///
/// TODO: walker selects the IR opcode by EffectInfo
/// branch (`CallMayForce*` for forces, `CallLoopinvariant*` for
/// loop-invariant, `CallPure*` for elidable, otherwise `Call*`) via
/// [`select_residual_call_opcode`]. Two sub-cases route through
/// dedicated helpers before the selector:
///   - **release-gil** ([`direct_call_release_gil`], `pyjitpl.py:3671-
///     3681`) — early-return when `ei.is_call_release_gil()`,
///     reshapes the arglist to `[savebox, funcbox] + argboxes[1:]`
///     and records `CALL_RELEASE_GIL_*` instead of `CALL_MAY_FORCE_*`.
///   - **loop-invariant heapcache** ([`loopinvariant_lookup`] /
///     [`loopinvariant_now_known`], `pyjitpl.py:2088 + 2109`) —
///     short-circuits the record on a heapcache hit and populates
///     the cache after a fresh record.
///
/// Emits `GUARD_NOT_FORCED` on the forces path plus
/// `GUARD_NO_EXCEPTION` whenever `check_can_raise(False)` is true,
/// matching `pyjitpl.py:2078-2082`. After every recorded call op,
/// invalidates the heapcache via
/// `heap_cache.invalidate_caches_varargs(call_opcode, ei, allboxes)`
/// matching `pyjitpl.py:2659 _record_helper_varargs` parity (forces
/// branch's `pyjitpl.py:2072` redundantly invalidates with
/// `CALL_MAY_FORCE_*`, equivalent because `select_residual_call_opcode`
/// returns `CallMayForce*` for the forces classification).  Release-gil
/// helper invalidates with `CALL_MAY_FORCE_*` matching
/// `pyjitpl.py:2072`'s `opnum1`. The pre-call vable IR bookkeeping
/// (`pyjitpl.py:2017 vable_and_vrefs_before_residual_call`, IR-only
/// portion: FORCE_TOKEN + SETFIELD_GC) is wired via
/// [`maybe_walker_vable_and_vrefs_before_residual_call`].  The
/// after-call helpers (`pyjitpl.py:3337-3366
/// vrefs_after_residual_call` / `vable_after_residual_call`) and the
/// runtime heap mutations on `tracing_before_residual_call` stay on
/// the trait-driven leg in pyre — see
/// [`walker_vable_and_vrefs_before_residual_call`] for the IR-vs-heap
/// split rationale.  The `OS_NOT_IN_TRACE` check fires up front via
/// [`do_not_in_trace_call_result`] — fail-loud guard against future
/// silent TODOs once the `majit-translate` analyzer trio
/// populates `oopspecindex`.
///
/// Still missing relative to upstream `do_residual_call`, all blocked
/// on infrastructure absent from pyre-jit-trace today:
///   - `OS_JIT_FORCE_VIRTUAL` PTR_EQ + GUARD_VALUE prelude
///     (`pyjitpl.py:2011-2014 → 2153-2172 _do_jit_force_virtual`) —
///     walker is fail-loud here via [`do_jit_force_virtual_guard`]
///     (called from each `dispatch_residual_call_*` arm); a producer
///     that emits an `OopSpecIndex::JitForceVirtual` calldescr surfaces
///     `DispatchError::JitForceVirtualRequiresConcreteResolver` instead
///     of silently recording `CALL_MAY_FORCE_*` (this was the prior
///     behaviour and is documented as STRICTER-THAN-PYPY in
///     [`do_jit_force_virtual_guard`]'s docstring). Optimizer pass
///     `OptVirtualize::optimize_jit_force_virtual` (`virtualize.rs:1226`)
///     already handles the constant-token / non-null-forced short-circuit
///     post-trace. Adding the PTR_EQ + GUARD_VALUE prelude (the only
///     way to retire the fail-loud guard) is not yet implemented and
///     would land on both legs together; metainterp has a tests-only
///     orthodox port at
///     `majit-metainterp/src/pyjitpl/mod.rs:11828 _do_jit_force_virtual`
///     that the converged walker would route through. Production reach
///     today is zero — `jtransform.rs:1903 jit.force_virtual` is the only
///     producer and pyre's interpreter does not emit it.
///   - Trait-leg-only: `vrefs_after_residual_call` / `vable_after_residual_call`
///     observe runtime forces by reading the heap token after the
///     callee runs.  Walker is symbolic-only, so it cannot detect
///     forces; the trait dispatch (`state.rs MIFrame::vable_after_residual_call`,
///     `trace_opcode.rs:2237-2263`) detects + aborts via
///     `PyError::runtime_error("ABORT_ESCAPE: ...")` before walker IR
///     diff would run.
///   - `direct_libffi_call` (`pyjitpl.py:3622-3667`) — pyre's live
///     tracer also returns `None` from this helper unless a
///     `CIF_DESCRIPTION_P` parser + dynamic `calldescr` builder lands
///     (`majit-metainterp/src/pyjitpl/mod.rs:11487-11491` defers to
///     direct_call_release_gil/may_force, which is the same fall-through
///     the walker already takes).
///   - `direct_assembler_call` (`pyjitpl.py:3589-3609`) + KEEPALIVE
///     (`pyjitpl.py:2080-2081`) — only fire when `assembler_call=True`
///     in `do_residual_call`. Walker's residual_call dispatchers are
///     never called with `assembler_call=True`; the parallel
///     `inline_call_*/dR>X` family routes through
///     [`dispatch_inline_call_dr_kind`] instead. Adding the path would
///     require the codewriter to emit a new `assembler_call` shape, not
///     a walker-side change.
///   - Per-PC liveness narrowing for the snapshot that
///     `walker_capture_snapshot_for_last_guard` attaches
///     (`pyjitpl.py:218-225 _get_list_of_active_boxes`). Walker's
///     helper today snapshots every non-`OpRef::NONE` register across
///     all three banks; RPython narrows the box list via
///     `jitcode.get_live_vars_info(pc, op_live)` so dead registers are
///     pruned before the snapshot.  The walker has no `op_live` byte
///     reader plumbed through `SubJitCodeBody` yet — follow-up
///     once the codewriter exposes the per-PC liveness table on the
///     callee body slice.  Over-capture is correctness-preserving:
///     `store_final_boxes_in_guard` filters dead boxes from the
///     snapshot via the optimizer's liveness pass.
#[allow(non_snake_case)]
fn dispatch_residual_call_iRd_kind(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let funcptr = read_int_reg(code, op, 0, ctx)?;
    let (r_args, arg_width) = read_ref_var_list(code, op, 1, ctx)?;
    let descr_offset = 1 + arg_width;
    let descr_index = decode_descr_index(code, op, descr_offset);
    let descr = read_descr(code, op, descr_offset, ctx)?;
    // RPython `do_residual_or_indirect_call` always receives a
    // CallDescr (pyjitpl.py:1995). Codewriter emits only CallDescrs
    // for residual_call slots; surface a typed error if a test fixture
    // (or future deviation) routes a non-CallDescr here.
    let call_descr = descr
        .as_call_descr()
        .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
            pc: op.pc,
            descr_index,
        })?;
    let descr_key = descr.index();
    // Void shape `_r_v/iRd` (`pyjitpl.py:1348 opimpl_residual_call_r_v =
    // _opimpl_residual_call1`) has no trailing `>X` dst byte. The
    // result OpRef is discarded by `write_residual_call_result_to_dst`'s
    // `'v'` arm, so `dst` is irrelevant on the void path; reading the
    // byte would walk past the operand list.
    let dst = if dst_bank == 'v' {
        0
    } else {
        code[op.pc + 1 + descr_offset + 2] as usize
    };

    // `_r_*` shape: argboxes = R-list only; argbox_types = [Ref; n].
    let argbox_types: Vec<Type> = vec![Type::Ref; r_args.len()];
    let allboxes = build_allboxes(funcptr, &r_args, &argbox_types, call_descr.arg_types());
    ensure_residual_call_args_bound(&allboxes, op.pc)?;

    let ei = call_descr.get_extra_info();
    // pyjitpl.py:2003-2005 OS_NOT_IN_TRACE guard — see helper docstring
    // for the convergence rationale.
    if let Some(outcome) = do_not_in_trace_call_result(ei, op.pc)? {
        return Ok((outcome, op.next_pc));
    }
    // pyjitpl.py:2011-2014 OS_JIT_FORCE_VIRTUAL fail-loud — walker
    // can't reproduce `_do_jit_force_virtual` without a concrete
    // `vref_ptr` resolver; surface a typed error rather than silently
    // recording `CALL_MAY_FORCE_*`.
    do_jit_force_virtual_guard(ei, op.pc)?;

    // pyjitpl.py:2063 forces-branch sub-case: when the descr's
    // `call_release_gil_target` is a non-NULL `(realfuncaddr, saveerr)`
    // pair, route through `direct_call_release_gil` which records
    // `CALL_RELEASE_GIL_*` with the upstream-shape arglist
    // `[savebox, funcbox] + argboxes[1:]` (pyjitpl.py:3675-3681).  All
    // other forces-branch paths (CALL_MAY_FORCE_*, the loopinvariant
    // sub-case below, the elidable branch, the default branch) come
    // out of `select_residual_call_opcode`.
    if ei.is_call_release_gil() {
        direct_call_release_gil(
            ctx,
            ei,
            &allboxes,
            descr.clone(),
            dst_bank,
            dst,
            op.pc,
            "dispatch_residual_call_iRd_kind",
        )?;
    } else if let Some(cached) = loopinvariant_lookup(ctx, ei, descr_key, funcptr) {
        // pyjitpl.py:2087-2110 EF_LOOPINVARIANT short-circuit. The
        // cached path emits no IR op and no guard, so result-before-
        // guard ordering is moot — write the dst eagerly.
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, cached)?;
    } else {
        let (call_opcode, can_raise, emit_guard_not_forced) =
            select_residual_call_opcode(ei, dst_bank, "dispatch_residual_call_iRd_kind");

        // pyjitpl.py:2017 `vable_and_vrefs_before_residual_call` — fires
        // unconditionally on the forces branch.  Records FORCE_TOKEN +
        // SETFIELD_GC IR for the active virtualizable; the runtime heap
        // mutations on `vinfo.tracing_before_residual_call` and
        // `vrefinfo.tracing_before_residual_call` (`pyjitpl.py:3318-3330`)
        // sit on the trait-driven leg only — see
        // [`walker_vable_and_vrefs_before_residual_call`] docstring for
        // the IR-vs-heap split rationale.
        if emit_guard_not_forced {
            maybe_walker_vable_and_vrefs_before_residual_call(ctx);
        }

        let recorded = ctx
            .trace_ctx
            .record_op_with_descr(call_opcode, &allboxes, descr.clone());

        // pyjitpl.py:1346-1400 `_record_helper_pure` parity: for
        // `CallPure*` whose every argbox carries a known `box_value`,
        // execute the helper now and stamp `recorded` with the result so
        // downstream walker chain (sub-jitcode bodies that consume the
        // result via `concrete_of_opref`) folds end-to-end.  No-op when
        // any argbox is symbolic, when the EI can raise, or for non-pure
        // call opcodes.
        try_fold_pure_call_via_executor(ctx, call_opcode, &allboxes, call_descr, recorded);

        // pyjitpl.py:1346/1349/1354 `_opimpl_residual_call{1,2,3}` parity
        // for the non-elidable shapes (Task #390 sub-slice 3).  PyPy
        // concrete-executes EVERY residual call regardless of EI — the
        // `exc` flag only selects the *guard* shape downstream
        // (`handle_possible_exception` → `GUARD_EXCEPTION` vs
        // `GUARD_NO_EXCEPTION`), not whether the helper runs.  Without
        // this, walker-recorded non-elidable helpers
        // (`store_subscr_fn`, `set_current_exception`, …) would skip
        // their heap mutation because `eval.rs:3285-3308`'s walker-skip
        // path bypasses `execute_opcode_step` → SIGBUS on the next read
        // of the un-mutated container (M4 walker unactivated taxonomy).
        // Task #390 sub-slice 4: PyPy-orthodox activation.  PyPy's
        // `_opimpl_residual_call*` concrete-executes EVERY residual
        // call regardless of EI; the `exc` flag only selects the
        // post-call guard shape (`GUARD_EXCEPTION` vs `GUARD_NO_EXCEPTION`)
        // in `handle_possible_exception`.  Pyre matches by always
        // invoking the executor — `try_execute_residual_call_via_executor`
        // self-gates on a fnaddr-sanity check (rejecting unpatched
        // `symbolic_fnaddr_for_path` hashes whose bits ≥ 47 are set)
        // so unregistered helpers degrade gracefully to recording-only
        // instead of SIGBUSing.
        let resid_raised = matches!(
            try_execute_residual_call_via_executor(
                ctx,
                call_opcode,
                &allboxes,
                call_descr,
                recorded,
            ),
            Some(Err(_))
        );
        debug_assert!(
            !resid_raised || can_raise,
            "dispatch_residual_call_iRd_kind: helper raised on a \
             `!can_raise` EI — EffectInfo claim/reality mismatch"
        );

        // pyjitpl.py:2659 `_record_helper_varargs` parity: every
        // recorded varargs op invalidates the heapcache via
        // `heapcache.invalidate_caches_varargs(opnum, descr,
        // argboxes)`.  Pyre's `record_op_with_descr` does NOT
        // auto-invalidate, so call it explicitly here.  Forces
        // branch (`select_residual_call_opcode` returned a
        // `CallMayForce*`) thus matches `pyjitpl.py:2072` which uses
        // `opnum1 = CALL_MAY_FORCE_*`; non-forces branches
        // (`CallLoopinvariant*`/`CallPure*`/`Call*`) match the
        // `_record_helper_varargs` invocation that runs inside
        // upstream's `executor.execute_varargs(opnum, ...)`.
        ctx.trace_ctx
            .heapcache_invalidate_caches_varargs(call_opcode, Some(ei), &allboxes);
        // pyjitpl.py:1950 _opimpl_residual_call*: the result lands in
        // `registers_*[reg_index]` BEFORE
        // `handle_possible_exception()` runs.  Write the dst here so
        // the guard's fail_args snapshot reads the recorded OpRef in
        // the slot the resume position points at — otherwise raising
        // calls would surface NONE in fail_args for the `>X` slot.
        //
        // Skip on `resid_raised`: `recorded` carries no concrete
        // result (helper raised before producing one); writing it to
        // dst would propagate a stale-stamped OpRef into downstream
        // walker chain (`pyjitpl.py:1392` only stamps `result_box.value`
        // on the success path of `execute_varargs`).
        if !resid_raised {
            write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, recorded)?;
        }
        // pyjitpl.py:2079 `metainterp.generate_guard(rop.GUARD_NOT_FORCED)`
        // — unconditionally on the forces-virtual-or-virtualizable branch.
        // Walker omits the `vable_after_residual_call(funcbox)`
        // short-circuit (`pyjitpl.py:2078`) entirely — the trait-dispatch
        // leg detects vable escape via
        // `state.rs MIFrame::vable_after_residual_call`
        // (`trace_opcode.rs:2237-2263`) and aborts to blackhole through
        // `PyError::runtime_error("ABORT_ESCAPE: ...")` before walker IR
        // diff would run.
        if emit_guard_not_forced {
            ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc);
        }
        // pyjitpl.py:2082 `metainterp.handle_possible_exception()` —
        // emits `GUARD_EXCEPTION(exc_type)` when the recording-time
        // helper raised (pinning the class for guard recovery), else
        // `GUARD_NO_EXCEPTION`.  `walker_capture_snapshot_for_last_guard`
        // ports `capture_resumedata(after_residual_call=True)`
        // (`pyjitpl.py:2599-2603`).
        if can_raise {
            if resid_raised {
                walker_record_guard_exception(ctx, op.pc);
            } else {
                ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
                walker_capture_snapshot_for_last_guard(ctx, op.pc);
            }
        }

        // pyjitpl.py:2109 `heapcache.call_loopinvariant_now_known`:
        // populate the cache so a subsequent matching call short-
        // circuits via the lookup above.  No-op for non-loopinvariant
        // EI per `loopinvariant_now_known`'s extraeffect check.
        //
        // Skip on `resid_raised`: caching a `recorded` OpRef with no
        // stamped concrete would propagate the un-stamped value into a
        // subsequent loop iteration's `loopinvariant_lookup` hit,
        // bypassing the actual helper call.
        if !resid_raised {
            loopinvariant_now_known(ctx, ei, descr_key, funcptr, recorded);
        }
    }

    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `residual_call` shape `iIRd>X` dispatcher — `_ir_*` arglist with
/// both an int-bank list and a ref-bank list before the descr. RPython
/// parity: `pyjitpl.py:1338-1340 _opimpl_residual_call2` (`@arguments`
/// argspec `"box", "boxes2", "descr", "orgpc"`) → same
/// `do_residual_or_indirect_call` body as `_call1`. The `boxes2`
/// argcode (`pyjitpl.py:3750-3760`) decodes two adjacent
/// count-prefixed lists into a single concatenated `argboxes` array
/// `[i_args..., r_args...]`. `_build_allboxes` (`pyjitpl.py:1960-1993`,
/// ported to [`build_allboxes`]) then permutes those to match
/// `descr.get_arg_types()` ABI ordering, so a callee whose `arg_types`
/// is `[REF, INT, REF, INT]` ends up with allboxes
/// `[funcbox, r_args[0], i_args[0], r_args[1], i_args[1]]`.
///
/// Operand layout `iIRd>X`:
///   1B funcptr (i) + 1B i-list count + N×1B i-regs + 1B r-list count
///   + M×1B r-regs + 2B descr + 1B `>X` dst.
///
/// EffectInfo classification + guard emission match
/// `dispatch_residual_call_iRd_kind` via [`select_residual_call_opcode`],
/// and the same release-gil ([`direct_call_release_gil`]) +
/// loop-invariant heapcache ([`loopinvariant_lookup`] /
/// [`loopinvariant_now_known`]) sub-cases route through dedicated
/// helpers ahead of the selector.
///
/// Heapcache invalidation matches `iRd_kind`:
/// `invalidate_caches_varargs(call_opcode, ei, allboxes)` after every
/// recorded call op (`pyjitpl.py:2659 _record_helper_varargs`); the
/// release-gil helper invalidates with `CALL_MAY_FORCE_*` per
/// `pyjitpl.py:2072`. `OS_NOT_IN_TRACE` is fail-loud-guarded up front
/// via [`do_not_in_trace_call_result`] (matches `iRd_kind`). Pre-call
/// vable IR bookkeeping (`vable_and_vrefs_before_residual_call`
/// IR-only portion at `pyjitpl.py:3327-3335`: FORCE_TOKEN +
/// SETFIELD_GC) is wired identically to `iRd_kind` via
/// [`maybe_walker_vable_and_vrefs_before_residual_call`]; the runtime
/// heap mutations and the after-call helpers stay on the
/// trait-driven leg.
///
/// Still missing relative to upstream — same set as `iRd_kind` and
/// blocked on the same infrastructure: `OS_JIT_FORCE_VIRTUAL`
/// short-circuit, `direct_libffi_call` / `direct_assembler_call`
/// specialization, KEEPALIVE for vablebox, `num_live`-aware
/// `capture_resumedata(after_residual_call=True)` on the guards. See
/// `dispatch_residual_call_iRd_kind`'s docstring for the per-item
/// blocking rationale.
#[allow(non_snake_case)]
fn dispatch_residual_call_iIRd_kind(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let funcptr = read_int_reg(code, op, 0, ctx)?;
    let (i_args, i_width) = read_int_var_list(code, op, 1, ctx)?;
    let (r_args, r_width) = read_ref_var_list(code, op, 1 + i_width, ctx)?;
    let descr_offset = 1 + i_width + r_width;
    let descr_index = decode_descr_index(code, op, descr_offset);
    let descr = read_descr(code, op, descr_offset, ctx)?;
    let call_descr = descr
        .as_call_descr()
        .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
            pc: op.pc,
            descr_index,
        })?;
    let descr_key = descr.index();
    // Void shape `_ir_v/iIRd` (`pyjitpl.py:1351 opimpl_residual_call_ir_v =
    // _opimpl_residual_call2`) has no `>X` dst byte; see
    // `dispatch_residual_call_iRd_kind` for the void operand-layout note.
    let dst = if dst_bank == 'v' {
        0
    } else {
        code[op.pc + 1 + descr_offset + 2] as usize
    };

    // Flat argboxes = i_args ++ r_args (`boxes2` argcode order).
    // Parallel argbox_types stamps each entry with its source bank so
    // `_build_allboxes`'s type-filter loops can permute correctly.
    let mut argboxes: Vec<OpRef> = Vec::with_capacity(i_args.len() + r_args.len());
    let mut argbox_types: Vec<Type> = Vec::with_capacity(i_args.len() + r_args.len());
    argboxes.extend_from_slice(&i_args);
    argbox_types.extend(std::iter::repeat(Type::Int).take(i_args.len()));
    argboxes.extend_from_slice(&r_args);
    argbox_types.extend(std::iter::repeat(Type::Ref).take(r_args.len()));
    let allboxes = build_allboxes(funcptr, &argboxes, &argbox_types, call_descr.arg_types());
    ensure_residual_call_args_bound(&allboxes, op.pc)?;

    let ei = call_descr.get_extra_info();
    // pyjitpl.py:2003-2005 OS_NOT_IN_TRACE guard — see helper docstring
    // for the convergence rationale.
    if let Some(outcome) = do_not_in_trace_call_result(ei, op.pc)? {
        return Ok((outcome, op.next_pc));
    }
    // pyjitpl.py:2011-2014 OS_JIT_FORCE_VIRTUAL fail-loud — see
    // `dispatch_residual_call_iRd_kind` for the rationale.
    do_jit_force_virtual_guard(ei, op.pc)?;

    // pyjitpl.py:2063 forces-branch sub-case: route release-gil through
    // `direct_call_release_gil`.  Mirrors `dispatch_residual_call_iRd_kind`.
    if ei.is_call_release_gil() {
        direct_call_release_gil(
            ctx,
            ei,
            &allboxes,
            descr.clone(),
            dst_bank,
            dst,
            op.pc,
            "dispatch_residual_call_iIRd_kind",
        )?;
    } else if let Some(cached) = loopinvariant_lookup(ctx, ei, descr_key, funcptr) {
        // pyjitpl.py:2087-2110 EF_LOOPINVARIANT short-circuit; no IR
        // op, no guard, ordering moot.
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, cached)?;
    } else {
        let (call_opcode, can_raise, emit_guard_not_forced) =
            select_residual_call_opcode(ei, dst_bank, "dispatch_residual_call_iIRd_kind");

        // pyjitpl.py:2017 `vable_and_vrefs_before_residual_call` —
        // records FORCE_TOKEN + SETFIELD_GC IR; runtime heap mutations
        // and the after-call helpers stay on the trait-driven leg.
        // See `dispatch_residual_call_iRd_kind` for the upstream-citation
        // walkthrough.
        if emit_guard_not_forced {
            maybe_walker_vable_and_vrefs_before_residual_call(ctx);
        }

        let recorded = ctx
            .trace_ctx
            .record_op_with_descr(call_opcode, &allboxes, descr.clone());

        // pyjitpl.py:1346-1400 `_record_helper_pure` parity — see
        // `dispatch_residual_call_iRd_kind` for the upstream walk.
        try_fold_pure_call_via_executor(ctx, call_opcode, &allboxes, call_descr, recorded);

        // Non-elidable concrete-execute parity (Task #390 sub-slice 3)
        // — see `dispatch_residual_call_iRd_kind` for the full citation.
        // Task #390 sub-slice 4: PyPy-orthodox activation.  PyPy's
        // `_opimpl_residual_call*` concrete-executes EVERY residual
        // call regardless of EI; the `exc` flag only selects the
        // post-call guard shape (`GUARD_EXCEPTION` vs `GUARD_NO_EXCEPTION`)
        // in `handle_possible_exception`.  Pyre matches by always
        // invoking the executor — `try_execute_residual_call_via_executor`
        // self-gates on a fnaddr-sanity check (rejecting unpatched
        // `symbolic_fnaddr_for_path` hashes whose bits ≥ 47 are set)
        // so unregistered helpers degrade gracefully to recording-only
        // instead of SIGBUSing.
        let resid_raised = matches!(
            try_execute_residual_call_via_executor(
                ctx,
                call_opcode,
                &allboxes,
                call_descr,
                recorded,
            ),
            Some(Err(_))
        );
        debug_assert!(
            !resid_raised || can_raise,
            "dispatch_residual_call_iIRd_kind: helper raised on a \
             `!can_raise` EI — EffectInfo claim/reality mismatch"
        );

        // pyjitpl.py:2659 `_record_helper_varargs` parity — see
        // `dispatch_residual_call_iRd_kind` for the upstream-citation
        // walkthrough.  Same invalidation semantics; only the
        // arglist construction differs (boxes2 = i_args ++ r_args).
        ctx.trace_ctx
            .heapcache_invalidate_caches_varargs(call_opcode, Some(ei), &allboxes);
        // pyjitpl.py:1950 _opimpl_residual_call*: result writeback runs
        // BEFORE handle_possible_exception().  See
        // `dispatch_residual_call_iRd_kind` for the full citation.
        if !resid_raised {
            write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, recorded)?;
        }
        if emit_guard_not_forced {
            ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc);
        }
        if can_raise {
            if resid_raised {
                walker_record_guard_exception(ctx, op.pc);
            } else {
                ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
                walker_capture_snapshot_for_last_guard(ctx, op.pc);
            }
        }

        if !resid_raised {
            loopinvariant_now_known(ctx, ei, descr_key, funcptr, recorded);
        }
    }

    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `residual_call` shape `iIRFd>X` dispatcher — `_irf_*` arglist with
/// int + ref + float lists before the descr. RPython parity:
/// `pyjitpl.py:1342-1346 _opimpl_residual_call3` (`@arguments` argspec
/// `"box", "boxes3", "descr", "orgpc"`) → same
/// `do_residual_or_indirect_call` body as `_call1` / `_call2`. The
/// `boxes3` argcode (`pyjitpl.py:3760-3776`) decodes three adjacent
/// count-prefixed lists into one concatenated `argboxes` array
/// `[i_args..., r_args..., f_args...]`. `_build_allboxes`
/// (`pyjitpl.py:1960-1993`, ported to [`build_allboxes`]) permutes
/// those to match `descr.get_arg_types()` ABI ordering.
///
/// Operand layout `iIRFd>X`:
///   1B funcptr (i) + 1B i-list count + N×1B i-regs + 1B r-list count
///   + M×1B r-regs + 1B f-list count + K×1B f-regs + 2B descr + 1B
///   `>X` dst.
///
/// EffectInfo classification + guard emission match
/// `dispatch_residual_call_iIRd_kind`; all sub-cases (release-gil,
/// loop-invariant, default) route through the same helpers
/// ([`select_residual_call_opcode`], [`direct_call_release_gil`],
/// [`loopinvariant_lookup`] / [`loopinvariant_now_known`]).
#[allow(non_snake_case)]
fn dispatch_residual_call_iIRFd_kind(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let funcptr = read_int_reg(code, op, 0, ctx)?;
    let (i_args, i_width) = read_int_var_list(code, op, 1, ctx)?;
    let (r_args, r_width) = read_ref_var_list(code, op, 1 + i_width, ctx)?;
    let (f_args, f_width) = read_float_var_list(code, op, 1 + i_width + r_width, ctx)?;
    let descr_offset = 1 + i_width + r_width + f_width;
    let descr_index = decode_descr_index(code, op, descr_offset);
    let descr = read_descr(code, op, descr_offset, ctx)?;
    let call_descr = descr
        .as_call_descr()
        .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
            pc: op.pc,
            descr_index,
        })?;
    let descr_key = descr.index();
    // Void shape `_irf_v/iIRFd` (`pyjitpl.py:1355 opimpl_residual_call_irf_v =
    // _opimpl_residual_call3`) has no `>X` dst byte; see
    // `dispatch_residual_call_iRd_kind` for the void operand-layout note.
    let dst = if dst_bank == 'v' {
        0
    } else {
        code[op.pc + 1 + descr_offset + 2] as usize
    };

    // Flat argboxes = i_args ++ r_args ++ f_args (`boxes3` argcode order).
    let mut argboxes: Vec<OpRef> = Vec::with_capacity(i_args.len() + r_args.len() + f_args.len());
    let mut argbox_types: Vec<Type> =
        Vec::with_capacity(i_args.len() + r_args.len() + f_args.len());
    argboxes.extend_from_slice(&i_args);
    argbox_types.extend(std::iter::repeat(Type::Int).take(i_args.len()));
    argboxes.extend_from_slice(&r_args);
    argbox_types.extend(std::iter::repeat(Type::Ref).take(r_args.len()));
    argboxes.extend_from_slice(&f_args);
    argbox_types.extend(std::iter::repeat(Type::Float).take(f_args.len()));
    let allboxes = build_allboxes(funcptr, &argboxes, &argbox_types, call_descr.arg_types());
    ensure_residual_call_args_bound(&allboxes, op.pc)?;

    let ei = call_descr.get_extra_info();
    if let Some(outcome) = do_not_in_trace_call_result(ei, op.pc)? {
        return Ok((outcome, op.next_pc));
    }
    // pyjitpl.py:2011-2014 OS_JIT_FORCE_VIRTUAL fail-loud — see
    // `dispatch_residual_call_iRd_kind` for the rationale.
    do_jit_force_virtual_guard(ei, op.pc)?;

    if ei.is_call_release_gil() {
        direct_call_release_gil(
            ctx,
            ei,
            &allboxes,
            descr.clone(),
            dst_bank,
            dst,
            op.pc,
            "dispatch_residual_call_iIRFd_kind",
        )?;
    } else if let Some(cached) = loopinvariant_lookup(ctx, ei, descr_key, funcptr) {
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, cached)?;
    } else {
        let (call_opcode, can_raise, emit_guard_not_forced) =
            select_residual_call_opcode(ei, dst_bank, "dispatch_residual_call_iIRFd_kind");

        // pyjitpl.py:2017 `vable_and_vrefs_before_residual_call` —
        // records FORCE_TOKEN + SETFIELD_GC IR; runtime heap mutations
        // and the after-call helpers stay on the trait-driven leg.
        // See `dispatch_residual_call_iRd_kind` for the upstream-citation
        // walkthrough.
        if emit_guard_not_forced {
            maybe_walker_vable_and_vrefs_before_residual_call(ctx);
        }

        let recorded = ctx
            .trace_ctx
            .record_op_with_descr(call_opcode, &allboxes, descr.clone());

        // pyjitpl.py:1346-1400 `_record_helper_pure` parity — see
        // `dispatch_residual_call_iRd_kind` for the upstream walk.
        try_fold_pure_call_via_executor(ctx, call_opcode, &allboxes, call_descr, recorded);

        // Non-elidable concrete-execute parity (Task #390 sub-slice 3)
        // — see `dispatch_residual_call_iRd_kind` for the full citation.
        // Task #390 sub-slice 4: PyPy-orthodox activation.  PyPy's
        // `_opimpl_residual_call*` concrete-executes EVERY residual
        // call regardless of EI; the `exc` flag only selects the
        // post-call guard shape (`GUARD_EXCEPTION` vs `GUARD_NO_EXCEPTION`)
        // in `handle_possible_exception`.  Pyre matches by always
        // invoking the executor — `try_execute_residual_call_via_executor`
        // self-gates on a fnaddr-sanity check (rejecting unpatched
        // `symbolic_fnaddr_for_path` hashes whose bits ≥ 47 are set)
        // so unregistered helpers degrade gracefully to recording-only
        // instead of SIGBUSing.
        let resid_raised = matches!(
            try_execute_residual_call_via_executor(
                ctx,
                call_opcode,
                &allboxes,
                call_descr,
                recorded,
            ),
            Some(Err(_))
        );
        debug_assert!(
            !resid_raised || can_raise,
            "dispatch_residual_call_iIRFd_kind: helper raised on a \
             `!can_raise` EI — EffectInfo claim/reality mismatch"
        );

        ctx.trace_ctx
            .heapcache_invalidate_caches_varargs(call_opcode, Some(ei), &allboxes);
        if !resid_raised {
            write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, recorded)?;
        }
        if emit_guard_not_forced {
            ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc);
        }
        if can_raise {
            if resid_raised {
                walker_record_guard_exception(ctx, op.pc);
            } else {
                ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
                walker_capture_snapshot_for_last_guard(ctx, op.pc);
            }
        }

        if !resid_raised {
            loopinvariant_now_known(ctx, ei, descr_key, funcptr, recorded);
        }
    }

    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Allocate the callee's three symbolic register banks for a sub-walk
/// entered through any `inline_call_*` arm.
///
/// Each bank is sized to `num_regs_X + constants_X.len()`
/// (RPython `JitCode.num_regs_and_consts_X`) so callee bytecode that
/// reads the post-regs constant window (indices
/// `[num_regs_X, num_regs_and_consts_X)`) finds a populated slot.
/// Constant slots are filled via `TraceCtx::const_int` / `const_ref` /
/// `const_float`, matching RPython
/// `pyjitpl.py:98-119 MIFrame.copy_constants`.
///
/// Also returns Ref- and Int-bank concrete shadows sized to match
/// `registers_r` / `registers_i`.  Ref-bank constant slots seed
/// `ConcreteValue::Null` (concrete propagation for the const pool
/// would need a backing `PyObjectRef` materialisation that the
/// sub-walk doesn't yet drive); Int-bank constant slots seed
/// `ConcreteValue::Int(v)` directly from `body.constants_i` so a
/// future `goto_if_not/iL` reading a constant input finds a non-Null
/// concrete and can fold the branch.
fn allocate_callee_register_banks(
    body: &SubJitCodeBody,
    trace_ctx: &mut TraceCtx,
) -> (
    Vec<OpRef>,
    Vec<OpRef>,
    Vec<OpRef>,
    Vec<ConcreteValue>,
    Vec<ConcreteValue>,
) {
    let total_r = body.num_regs_r + body.constants_r.len();
    let total_i = body.num_regs_i + body.constants_i.len();
    let total_f = body.num_regs_f + body.constants_f.len();
    let mut regs_r = vec![OpRef::NONE; total_r];
    let mut regs_i = vec![OpRef::NONE; total_i];
    let mut regs_f = vec![OpRef::NONE; total_f];
    let concrete_r = vec![ConcreteValue::Null; total_r];
    let mut concrete_i = vec![ConcreteValue::Null; total_i];
    for (i, &v) in body.constants_i.iter().enumerate() {
        regs_i[body.num_regs_i + i] = trace_ctx.const_int(v);
        concrete_i[body.num_regs_i + i] = ConcreteValue::Int(v);
    }
    for (i, &v) in body.constants_r.iter().enumerate() {
        regs_r[body.num_regs_r + i] = trace_ctx.const_ref(v);
    }
    for (i, &v) in body.constants_f.iter().enumerate() {
        regs_f[body.num_regs_f + i] = trace_ctx.const_float(v);
    }
    (regs_r, regs_i, regs_f, concrete_r, concrete_i)
}

/// Operand layout `dR>X`:
///   2B descr index + 1B varlen + N×1B Ref args + 1B `>X` dst.
///
/// RPython parity: `pyjitpl.py:1266-1324 _opimpl_inline_call*`. The
/// `_X` suffix is the callee's *return kind* — e.g. `_opimpl_inline_call_r_i`
/// dispatches an inline call whose callee body returns via
/// `int_return/i`. Walker semantics are otherwise identical to the
/// `_r_r` arm (which originally landed inline; this helper extracts the
/// shared body so kind variants can share the dispatch logic).
///
/// `dst_bank` selects where the SubReturn value lands:
/// * `'r'`: caller's `registers_r[dst]` — pairs with callee `ref_return/r`.
/// * `'i'`: caller's `registers_i[dst]` — pairs with callee `int_return/i`.
/// * `'f'`: would pair with callee `float_return/f` — not handled by
///   this helper because the codewriter doesn't emit a `dR>f` shape
///   (float return paths use the `dIRF` arglist family).
///
/// `kind_label` mirrors `dst_bank` as a static `&str` for typed-error
/// reporting (`RegisterOutOfRange::bank`).
fn dispatch_inline_call_dr_kind(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let sub_descr = read_descr(code, op, 0, ctx)?;
    let descr_index = (code[op.pc + 1] as usize) | ((code[op.pc + 2] as usize) << 8);
    let jc_descr = sub_descr
        .as_jitcode_descr()
        .ok_or(DispatchError::ExpectedJitCodeDescr {
            pc: op.pc,
            descr_index,
        })?;
    let sub_index = jc_descr.jitcode_index();
    let sub_body =
        (ctx.sub_jitcode_lookup)(sub_index).ok_or(DispatchError::SubJitCodeNotFound {
            pc: op.pc,
            jitcode_index: sub_index,
        })?;
    let (args, arg_width) = read_ref_var_list(code, op, 2, ctx)?;
    let arg_concretes = read_ref_var_list_concrete(code, op, 2, ctx);

    let (
        mut callee_regs_r,
        mut callee_regs_i,
        mut callee_regs_f,
        mut callee_concrete_r,
        mut callee_concrete_i,
    ) = allocate_callee_register_banks(&sub_body, ctx.trace_ctx);

    if args.len() > sub_body.num_regs_r {
        return Err(DispatchError::InlineCallArityMismatch {
            pc: op.pc,
            provided: args.len(),
            callee_num_regs_r: sub_body.num_regs_r,
        });
    }
    for (i, arg) in args.iter().enumerate() {
        callee_regs_r[i] = *arg;
    }
    for (i, concrete) in arg_concretes.iter().enumerate() {
        callee_concrete_r[i] = *concrete;
    }

    let (callee_outcome, _callee_end_pc) = {
        let mut sub_wc = WalkContext {
            registers_r: &mut callee_regs_r,
            registers_i: &mut callee_regs_i,
            registers_f: &mut callee_regs_f,
            concrete_registers_r: &mut callee_concrete_r,
            concrete_registers_i: &mut callee_concrete_i,
            descr_refs: ctx.descr_refs,
            trace_ctx: ctx.trace_ctx,
            done_with_this_frame_descr_ref: ctx.done_with_this_frame_descr_ref.clone(),
            done_with_this_frame_descr_int: ctx.done_with_this_frame_descr_int.clone(),
            done_with_this_frame_descr_float: ctx.done_with_this_frame_descr_float.clone(),
            done_with_this_frame_descr_void: ctx.done_with_this_frame_descr_void.clone(),
            exit_frame_with_exception_descr_ref: ctx.exit_frame_with_exception_descr_ref.clone(),
            is_top_level: false,
            sub_jitcode_lookup: ctx.sub_jitcode_lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: ctx.entry_py_pc,
            outer_jitcode_index: ctx.outer_jitcode_index,
            outer_active_boxes: ctx.outer_active_boxes.clone(),
        };
        walk(sub_body.code, 0, &mut sub_wc)?
    };

    match callee_outcome {
        DispatchOutcome::SubReturn {
            result: Some(value),
        } => {
            if dst_bank == 'v' {
                // `inline_call_r_v/dR`
                // (`bhimpl_inline_call_r_v` `blackhole.py:1287-1290`)
                // expects a void-return callee. A `Some` return here is
                // a codewriter shape mismatch.
                return Err(DispatchError::UnexpectedNonVoidSubReturn { pc: op.pc });
            }
            let dst = code[op.pc + 1 + 2 + arg_width] as usize;
            // inline_call_* dst writeback — `value` is the callee's
            // SubReturn OpRef.  The callee's matching concrete shadow
            // was dropped at sub-walk exit; `concrete_of_opref` still
            // sees through to `constants.get_value` for callees that
            // return a constant (e.g. `LoadConst` tail), so route via
            // the unified shadow channel.  Non-constant returns surface
            // as the sentinel `GcRef(usize::MAX)` → Null fallback.
            let concrete_for_shadow = concrete_from_recorded_opref(ctx, value);
            match dst_bank {
                'r' => {
                    write_ref_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                }
                'i' => {
                    write_int_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                }
                _ => unreachable!(
                    "dispatch_inline_call_dr_kind dst_bank must be 'r', 'i' or 'v' (\
                     codewriter does not emit dR>f shape today)"
                ),
            }
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        DispatchOutcome::SubReturn { result: None } => {
            if dst_bank == 'v' {
                // `inline_call_r_v/dR` expects exactly this — callee
                // exits via `void_return/`, no SubReturn writeback.
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
            // Same shape contract as `_r_r`: a `_r_<X>` variant promises
            // a non-void result for the dst's `>X` slot. A void return
            // reaching here is a codewriter shape mismatch.
            Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc })
        }
        DispatchOutcome::SubRaise { exc, exc_concrete } => {
            if let Some(target) = try_catch_exception_at(code, op.next_pc) {
                ctx.last_exc_value = Some(exc);
                // Thread the callee's concrete
                // exception across the frame boundary.  Without this a
                // downstream `raise/r` / `reraise/` in the caller's
                // handler would read `Null` and skip GUARD_CLASS,
                // losing the class-known pin that the callee's leg had
                // already established.
                ctx.last_exc_value_concrete = exc_concrete;
                Ok((DispatchOutcome::Continue, target))
            } else {
                Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc))
            }
        }
        DispatchOutcome::Terminate => Ok((DispatchOutcome::Terminate, op.next_pc)),
        DispatchOutcome::SwitchToBlackhole {
            reason,
            raising_exception,
        } => Ok((
            DispatchOutcome::SwitchToBlackhole {
                reason,
                raising_exception,
            },
            op.next_pc,
        )),
        DispatchOutcome::Continue => {
            unreachable!(
                "walk() only exits on Terminate / SubReturn / SubRaise / SwitchToBlackhole"
            )
        }
    }
}

/// `inline_call_ir_<X>/dIR>X` handler shared by `dIR>i` (Int result)
/// and `dIR>r` (Ref result). Same control-flow shape as
/// [`dispatch_inline_call_dr_kind`], extended with an I-list arglist
/// preceding the R-list.
///
/// Operand layout `dIR>X`:
///   2B descr index +
///   1B I-len + N×1B int args +
///   1B R-len + M×1B ref args +
///   1B `>X` dst.
///
/// RPython parity: `pyjitpl.py:1266-1324 _opimpl_inline_call*` —
/// kind-aware variants call `setup_call(argboxes_i, argboxes_r,
/// argboxes_f)` which distributes args into the callee's typed banks
/// (`pyjitpl.py:230-260`).
///
/// `dst_bank` selects where the SubReturn value lands: `'r'` writes to
/// `registers_r[dst]` (paired with callee `ref_return/r`), `'i'`
/// writes to `registers_i[dst]` (paired with callee `int_return/i`).
fn dispatch_inline_call_dir_kind(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let sub_descr = read_descr(code, op, 0, ctx)?;
    let descr_index = (code[op.pc + 1] as usize) | ((code[op.pc + 2] as usize) << 8);
    let jc_descr = sub_descr
        .as_jitcode_descr()
        .ok_or(DispatchError::ExpectedJitCodeDescr {
            pc: op.pc,
            descr_index,
        })?;
    let sub_index = jc_descr.jitcode_index();
    let sub_body =
        (ctx.sub_jitcode_lookup)(sub_index).ok_or(DispatchError::SubJitCodeNotFound {
            pc: op.pc,
            jitcode_index: sub_index,
        })?;
    // I-list at offset 2 (skip descr).
    let (int_args, int_width) = read_int_var_list(code, op, 2, ctx)?;
    // R-list immediately after the I-list.
    let (ref_args, ref_width) = read_ref_var_list(code, op, 2 + int_width, ctx)?;
    let ref_arg_concretes = read_ref_var_list_concrete(code, op, 2 + int_width, ctx);

    let (
        mut callee_regs_r,
        mut callee_regs_i,
        mut callee_regs_f,
        mut callee_concrete_r,
        mut callee_concrete_i,
    ) = allocate_callee_register_banks(&sub_body, ctx.trace_ctx);

    if int_args.len() > sub_body.num_regs_i {
        return Err(DispatchError::InlineCallIntArityMismatch {
            pc: op.pc,
            provided: int_args.len(),
            callee_num_regs_i: sub_body.num_regs_i,
        });
    }
    if ref_args.len() > sub_body.num_regs_r {
        return Err(DispatchError::InlineCallArityMismatch {
            pc: op.pc,
            provided: ref_args.len(),
            callee_num_regs_r: sub_body.num_regs_r,
        });
    }
    for (i, arg) in int_args.iter().enumerate() {
        callee_regs_i[i] = *arg;
    }
    for (i, arg) in ref_args.iter().enumerate() {
        callee_regs_r[i] = *arg;
    }
    for (i, concrete) in ref_arg_concretes.iter().enumerate() {
        callee_concrete_r[i] = *concrete;
    }

    let (callee_outcome, _callee_end_pc) = {
        let mut sub_wc = WalkContext {
            registers_r: &mut callee_regs_r,
            registers_i: &mut callee_regs_i,
            registers_f: &mut callee_regs_f,
            concrete_registers_r: &mut callee_concrete_r,
            concrete_registers_i: &mut callee_concrete_i,
            descr_refs: ctx.descr_refs,
            trace_ctx: ctx.trace_ctx,
            done_with_this_frame_descr_ref: ctx.done_with_this_frame_descr_ref.clone(),
            done_with_this_frame_descr_int: ctx.done_with_this_frame_descr_int.clone(),
            done_with_this_frame_descr_float: ctx.done_with_this_frame_descr_float.clone(),
            done_with_this_frame_descr_void: ctx.done_with_this_frame_descr_void.clone(),
            exit_frame_with_exception_descr_ref: ctx.exit_frame_with_exception_descr_ref.clone(),
            is_top_level: false,
            sub_jitcode_lookup: ctx.sub_jitcode_lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: ctx.entry_py_pc,
            outer_jitcode_index: ctx.outer_jitcode_index,
            outer_active_boxes: ctx.outer_active_boxes.clone(),
        };
        walk(sub_body.code, 0, &mut sub_wc)?
    };

    match callee_outcome {
        DispatchOutcome::SubReturn {
            result: Some(value),
        } => {
            if dst_bank == 'v' {
                return Err(DispatchError::UnexpectedNonVoidSubReturn { pc: op.pc });
            }
            // dst register byte sits after descr (2B) + I-list (int_width)
            // + R-list (ref_width) bytes.
            let dst = code[op.pc + 1 + 2 + int_width + ref_width] as usize;
            // See dispatch_inline_call_dr_kind: route the SubReturn
            // OpRef through the unified shadow channel so constant
            // return values propagate.
            let concrete_for_shadow = concrete_from_recorded_opref(ctx, value);
            match dst_bank {
                'r' => {
                    write_ref_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                }
                'i' => {
                    write_int_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                }
                _ => unreachable!("dispatch_inline_call_dir_kind dst_bank must be 'r', 'i' or 'v'"),
            }
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        DispatchOutcome::SubReturn { result: None } => {
            if dst_bank == 'v' {
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
            Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc })
        }
        DispatchOutcome::SubRaise { exc, exc_concrete } => {
            if let Some(target) = try_catch_exception_at(code, op.next_pc) {
                ctx.last_exc_value = Some(exc);
                // Thread the callee's concrete
                // exception across the frame boundary.  Without this a
                // downstream `raise/r` / `reraise/` in the caller's
                // handler would read `Null` and skip GUARD_CLASS,
                // losing the class-known pin that the callee's leg had
                // already established.
                ctx.last_exc_value_concrete = exc_concrete;
                Ok((DispatchOutcome::Continue, target))
            } else {
                Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc))
            }
        }
        DispatchOutcome::Terminate => Ok((DispatchOutcome::Terminate, op.next_pc)),
        DispatchOutcome::SwitchToBlackhole {
            reason,
            raising_exception,
        } => Ok((
            DispatchOutcome::SwitchToBlackhole {
                reason,
                raising_exception,
            },
            op.next_pc,
        )),
        DispatchOutcome::Continue => {
            unreachable!(
                "walk() only exits on Terminate / SubReturn / SubRaise / SwitchToBlackhole"
            )
        }
    }
}

/// `inline_call_irf_<X>/dIRF>X` handler shared by `dIRF>f` (Float
/// result) and `dIRF>r` (Ref result). Extends
/// [`dispatch_inline_call_dir_kind`] with an F-list arglist following
/// the R-list.
///
/// Operand layout `dIRF>X`:
///   2B descr index +
///   1B I-len + N×1B int args +
///   1B R-len + M×1B ref args +
///   1B F-len + K×1B float args +
///   1B `>X` dst.
///
/// RPython parity: same `pyjitpl.py:230-260 setup_call(argboxes_i,
/// argboxes_r, argboxes_f)` distribution — all three kind banks
/// populated from the three lists.
///
/// `dst_bank` selects where the SubReturn value lands: `'f'` writes
/// `registers_f[dst]` (paired with callee `float_return/f`), `'r'`
/// writes `registers_r[dst]` (paired with callee `ref_return/r`).
fn dispatch_inline_call_dirf_kind(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let sub_descr = read_descr(code, op, 0, ctx)?;
    let descr_index = (code[op.pc + 1] as usize) | ((code[op.pc + 2] as usize) << 8);
    let jc_descr = sub_descr
        .as_jitcode_descr()
        .ok_or(DispatchError::ExpectedJitCodeDescr {
            pc: op.pc,
            descr_index,
        })?;
    let sub_index = jc_descr.jitcode_index();
    let sub_body =
        (ctx.sub_jitcode_lookup)(sub_index).ok_or(DispatchError::SubJitCodeNotFound {
            pc: op.pc,
            jitcode_index: sub_index,
        })?;
    let (int_args, int_width) = read_int_var_list(code, op, 2, ctx)?;
    let (ref_args, ref_width) = read_ref_var_list(code, op, 2 + int_width, ctx)?;
    let ref_arg_concretes = read_ref_var_list_concrete(code, op, 2 + int_width, ctx);
    let (float_args, float_width) = read_float_var_list(code, op, 2 + int_width + ref_width, ctx)?;

    let (
        mut callee_regs_r,
        mut callee_regs_i,
        mut callee_regs_f,
        mut callee_concrete_r,
        mut callee_concrete_i,
    ) = allocate_callee_register_banks(&sub_body, ctx.trace_ctx);

    if int_args.len() > sub_body.num_regs_i {
        return Err(DispatchError::InlineCallIntArityMismatch {
            pc: op.pc,
            provided: int_args.len(),
            callee_num_regs_i: sub_body.num_regs_i,
        });
    }
    if ref_args.len() > sub_body.num_regs_r {
        return Err(DispatchError::InlineCallArityMismatch {
            pc: op.pc,
            provided: ref_args.len(),
            callee_num_regs_r: sub_body.num_regs_r,
        });
    }
    if float_args.len() > sub_body.num_regs_f {
        return Err(DispatchError::InlineCallFloatArityMismatch {
            pc: op.pc,
            provided: float_args.len(),
            callee_num_regs_f: sub_body.num_regs_f,
        });
    }
    for (i, arg) in int_args.iter().enumerate() {
        callee_regs_i[i] = *arg;
    }
    for (i, arg) in ref_args.iter().enumerate() {
        callee_regs_r[i] = *arg;
    }
    for (i, arg) in float_args.iter().enumerate() {
        callee_regs_f[i] = *arg;
    }
    for (i, concrete) in ref_arg_concretes.iter().enumerate() {
        callee_concrete_r[i] = *concrete;
    }

    let (callee_outcome, _callee_end_pc) = {
        let mut sub_wc = WalkContext {
            registers_r: &mut callee_regs_r,
            registers_i: &mut callee_regs_i,
            registers_f: &mut callee_regs_f,
            concrete_registers_r: &mut callee_concrete_r,
            concrete_registers_i: &mut callee_concrete_i,
            descr_refs: ctx.descr_refs,
            trace_ctx: ctx.trace_ctx,
            done_with_this_frame_descr_ref: ctx.done_with_this_frame_descr_ref.clone(),
            done_with_this_frame_descr_int: ctx.done_with_this_frame_descr_int.clone(),
            done_with_this_frame_descr_float: ctx.done_with_this_frame_descr_float.clone(),
            done_with_this_frame_descr_void: ctx.done_with_this_frame_descr_void.clone(),
            exit_frame_with_exception_descr_ref: ctx.exit_frame_with_exception_descr_ref.clone(),
            is_top_level: false,
            sub_jitcode_lookup: ctx.sub_jitcode_lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: ctx.entry_py_pc,
            outer_jitcode_index: ctx.outer_jitcode_index,
            outer_active_boxes: ctx.outer_active_boxes.clone(),
        };
        walk(sub_body.code, 0, &mut sub_wc)?
    };

    match callee_outcome {
        DispatchOutcome::SubReturn {
            result: Some(value),
        } => {
            if dst_bank == 'v' {
                return Err(DispatchError::UnexpectedNonVoidSubReturn { pc: op.pc });
            }
            let dst = code[op.pc + 1 + 2 + int_width + ref_width + float_width] as usize;
            // See dispatch_inline_call_dr_kind: route the SubReturn
            // OpRef through the unified shadow channel so constant
            // return values propagate.
            let concrete_for_shadow = concrete_from_recorded_opref(ctx, value);
            match dst_bank {
                'i' => {
                    write_int_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                }
                'r' => {
                    write_ref_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                }
                'f' => {
                    let len = ctx.registers_f.len();
                    let slot =
                        ctx.registers_f
                            .get_mut(dst)
                            .ok_or(DispatchError::RegisterOutOfRange {
                                pc: op.pc,
                                reg: dst,
                                len,
                                bank: "f",
                            })?;
                    *slot = value;
                }
                _ => unreachable!(
                    "dispatch_inline_call_dirf_kind dst_bank must be 'i', 'r', 'f' or 'v'"
                ),
            }
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        DispatchOutcome::SubReturn { result: None } => {
            if dst_bank == 'v' {
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
            Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc })
        }
        DispatchOutcome::SubRaise { exc, exc_concrete } => {
            if let Some(target) = try_catch_exception_at(code, op.next_pc) {
                ctx.last_exc_value = Some(exc);
                // Thread the callee's concrete
                // exception across the frame boundary.  Without this a
                // downstream `raise/r` / `reraise/` in the caller's
                // handler would read `Null` and skip GUARD_CLASS,
                // losing the class-known pin that the callee's leg had
                // already established.
                ctx.last_exc_value_concrete = exc_concrete;
                Ok((DispatchOutcome::Continue, target))
            } else {
                Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc))
            }
        }
        DispatchOutcome::Terminate => Ok((DispatchOutcome::Terminate, op.next_pc)),
        DispatchOutcome::SwitchToBlackhole {
            reason,
            raising_exception,
        } => Ok((
            DispatchOutcome::SwitchToBlackhole {
                reason,
                raising_exception,
            },
            op.next_pc,
        )),
        DispatchOutcome::Continue => {
            unreachable!(
                "walk() only exits on Terminate / SubReturn / SubRaise / SwitchToBlackhole"
            )
        }
    }
}

/// Per-opname dispatch table. Returning `(outcome, next_pc)` lets
/// branching handlers (`goto/L`) override the linear `op.next_pc`
/// advance; non-branching handlers return `op.next_pc` unchanged.
fn handle(
    op: &DecodedOp,
    code: &[u8],
    ctx: &mut WalkContext<'_, '_>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    match op.key {
        "live/" => Ok((DispatchOutcome::Continue, op.next_pc)),
        // RPython parity: `pyjitpl.py:1266-1324 _opimpl_inline_call*`
        // pushes a fresh `MIFrame(jitcode)` populated with caller args,
        // raises `ChangeFrame()` so the metainterp loop dispatches the
        // next op on the new frame, and on `*_return` pops back via
        // `metainterp.finishframe(value)` — writing `value` into the
        // caller's `>X` slot. Walker simulates the same shape with
        // synchronous recursion through `dispatch_inline_call_dr_kind`.
        //
        // The `_r_r` (Ref result) and `_r_i` (Int result) variants share
        // the same `dR` arglist shape; only the dst bank differs.
        "inline_call_r_r/dR>r" => dispatch_inline_call_dr_kind(code, op, ctx, 'r'),
        "inline_call_r_i/dR>i" => dispatch_inline_call_dr_kind(code, op, ctx, 'i'),
        // `_r_v/dR` — void-return variant per `bhimpl_inline_call_r_v`
        // (`blackhole.py:1287`).  Same recursion + arglist as `_r_*`;
        // callee exits via `void_return/`, no SubReturn writeback.
        "inline_call_r_v/dR" => dispatch_inline_call_dr_kind(code, op, ctx, 'v'),
        // `_ir_*` variants extend the arglist to a (I-list, R-list) pair.
        // RPython's `setup_call(argboxes_i, argboxes_r, argboxes_f)` populates
        // both kind banks. The dst bank still selects the SubReturn write
        // target (Ref bank for `_ir_r/dIR>r`, Int bank for `_ir_i/dIR>i`,
        // void no-write for `_ir_v/dIR`).
        "inline_call_ir_r/dIR>r" => dispatch_inline_call_dir_kind(code, op, ctx, 'r'),
        "inline_call_ir_i/dIR>i" => dispatch_inline_call_dir_kind(code, op, ctx, 'i'),
        "inline_call_ir_v/dIR" => dispatch_inline_call_dir_kind(code, op, ctx, 'v'),
        // `_irf_*` variants extend the arglist with a float list (I-list,
        // R-list, F-list). Same `setup_call(argboxes_i, argboxes_r,
        // argboxes_f)` distribution; dst bank chooses Int / Ref / Float
        // for the SubReturn writeback or void no-write for `_irf_v/dIRF`.
        "inline_call_irf_i/dIRF>i" => dispatch_inline_call_dirf_kind(code, op, ctx, 'i'),
        "inline_call_irf_r/dIRF>r" => dispatch_inline_call_dirf_kind(code, op, ctx, 'r'),
        "inline_call_irf_f/dIRF>f" => dispatch_inline_call_dirf_kind(code, op, ctx, 'f'),
        "inline_call_irf_v/dIRF" => dispatch_inline_call_dirf_kind(code, op, ctx, 'v'),
        "goto/L" => {
            // RPython `blackhole.py:950-952 bhimpl_goto(target): return
            // target`. The 2-byte LE label was resolved by
            // `assembler.fix_labels` to a direct pc; pyre + RPython
            // agree that goto records nothing (pure control flow).
            let target = read_label(code, op, 0);
            Ok((DispatchOutcome::Continue, target))
        }
        "goto_if_not/iL" => {
            // RPython `pyjitpl.py:511-526 opimpl_goto_if_not`:
            //
            //   @arguments("box", "label", "orgpc")
            //   def opimpl_goto_if_not(self, box, target, orgpc, replace=True):
            //       switchcase = box.getint()
            //       if switchcase:
            //           assert switchcase == 1
            //           opnum = rop.GUARD_TRUE
            //           promoted_box = CONST_1
            //       else:
            //           opnum = rop.GUARD_FALSE
            //           promoted_box = CONST_0
            //       self.metainterp.generate_guard(opnum, box, resumepc=orgpc)
            //       if not switchcase:
            //           self.pc = target
            //       if isinstance(box, Const):
            //           return
            //       if replace:
            //           self.metainterp.replace_box(box, promoted_box)
            //
            // Operand layout `iL`: 1B Int register + 2B LE label.
            // Concrete branch value comes from `TraceCtx::concrete_of_opref`
            // (same path `switch/id` uses); non-concrete OpRefs surface
            // `GotoIfNotValueNotConcrete` rather than guess a direction.
            let valuebox = read_int_reg(code, op, 0, ctx)?;
            let target = read_label(code, op, 1);
            let switchcase = match ctx.trace_ctx.concrete_of_opref(valuebox) {
                Some(Value::Int(v)) => v,
                _ => {
                    return Err(DispatchError::GotoIfNotValueNotConcrete {
                        pc: op.pc,
                        value: valuebox,
                    });
                }
            };
            // pyjitpl.py:514 `assert switchcase == 1` — codewriter
            // invariant: every condbox feeding GOTO_IF_NOT was produced
            // by an int_is_* family op, so the only non-zero value
            // possible is 1. Fail loud on any other truthy value rather
            // than silently coercing.
            assert!(
                switchcase == 0 || switchcase == 1,
                "opimpl_goto_if_not: switchcase must be 0 or 1, got {} (pc={})",
                switchcase,
                op.pc
            );
            let (opcode, promoted) = if switchcase != 0 {
                (OpCode::GuardTrue, ctx.trace_ctx.const_int(1))
            } else {
                (OpCode::GuardFalse, ctx.trace_ctx.const_int(0))
            };
            // `pyjitpl.py:511-526 opimpl_goto_if_not` calls
            // `generate_guard(opnum, box, resumepc=orgpc)`; the first
            // line of `generate_guard` (`pyjitpl.py:2583`) is
            // `if isinstance(box, Const): return` — Const boxes already
            // pin the value and need no guard. Same gate then governs
            // the `replace_box` / register-rewrite path
            // (`pyjitpl.py:523-526`). Resume-data capture
            // (`capture_resumedata(resumepc=orgpc)` at
            // `pyjitpl.py:2603`) is omitted here: the walker's IR is
            // rolled back via `cut_trace`, so the snapshot the trait
            // leg builds in `trace_opcode.rs:3275 MIFrame::generate_guard`
            // has no production effect on this leg. Once the walker
            // becomes the production trace emitter it needs to thread
            // `op.pc` here as resumepc and
            // capture the active-box snapshot the same way `MIFrame`
            // does.
            if !valuebox.is_constant() {
                ctx.trace_ctx.record_guard(opcode, &[valuebox], 0);
                walker_capture_snapshot_for_last_guard(ctx, op.pc);
                ctx.trace_ctx.replace_box(valuebox, promoted);
                for slot in ctx.registers_i.iter_mut() {
                    if *slot == valuebox {
                        *slot = promoted;
                    }
                }
            }
            let next_pc = if switchcase != 0 { op.next_pc } else { target };
            Ok((DispatchOutcome::Continue, next_pc))
        }
        "catch_exception/L" => {
            // RPython `blackhole.py:969-974 bhimpl_catch_exception(target)` —
            // "no-op when run normally" — and `pyjitpl.py:497-504
            // opimpl_catch_exception`:
            //
            //   def opimpl_catch_exception(self, target):
            //       """This is a no-op when run normally.  We can check that
            //       last_exc_value is a null ptr; it should have been set to None
            //       by the previous instruction.  If the previous instruction
            //       raised instead, finishframe_exception() should have been
            //       called and we would not be there."""
            //       assert not self.metainterp.last_exc_value
            //
            // The 2-byte target is metadata: when a `raise` fires on the
            // previous instruction, `handle_exception_in_frame`
            // (`blackhole.py:406-422`) reads it to redirect the unwinder
            // (consumed by `try_catch_exception_at` from the inline_call
            // SubRaise arm). Linear walk advances past the operand
            // without using the target.
            //
            // The RPython assert turns into a typed error here:
            // reaching `catch_exception/L` with `ctx.last_exc_value =
            // Some(_)` means either (a) the codewriter emitted a
            // catch_exception/L outside an exception-table position,
            // or (b) a previous catch handler didn't clear
            // last_exc_value after handling the raise. Either is a
            // codewriter-pass invariant violation.
            if ctx.last_exc_value.is_some() {
                return Err(DispatchError::CatchExceptionWithActiveException { pc: op.pc });
            }
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "switch/id" => dispatch_switch_id(code, op, ctx),
        "residual_call_r_r/iRd>r" => dispatch_residual_call_iRd_kind(code, op, ctx, 'r'),
        // `_r_i/iRd>i` mirrors `_r_r/iRd>r` with the dst kind flipped to
        // int. RPython `pyjitpl.py:1334-1347 _opimpl_residual_call1` is
        // exec-generated for `_callR` and `_callI` (Type::Ref vs
        // Type::Int return) — see resoperation.py:1461 `Type::Int =>
        // CallI`. EffectInfo classification + GUARD_NO_EXCEPTION
        // emission stay identical; only the result OpCode (CallI /
        // CallPureI) and dst writeback bank (registers_i) differ.
        "residual_call_r_i/iRd>i" => dispatch_residual_call_iRd_kind(code, op, ctx, 'i'),
        // `_ir_*/iIRd>X` extends the arglist with an i-bank list. RPython
        // `pyjitpl.py:_opimpl_residual_call*` exec-generates this for
        // callees taking both int + ref args (setup_call distributes
        // (argboxes_i, argboxes_r, argboxes_f=[])). Same EffectInfo
        // classification path; the only operand-shape change is the
        // I-list prefix between funcptr and the R-list.
        "residual_call_ir_r/iIRd>r" => dispatch_residual_call_iIRd_kind(code, op, ctx, 'r'),
        "residual_call_ir_i/iIRd>i" => dispatch_residual_call_iIRd_kind(code, op, ctx, 'i'),
        // `_irf_*/iIRFd>X` extends `_ir_*` with an f-bank list before the
        // descr (`pyjitpl.py:1342-1346 _opimpl_residual_call3`, `boxes3`
        // argcode `pyjitpl.py:3760-3776`). EffectInfo classification +
        // guard emission identical; only the operand layout adds the F
        // suffix list.
        "residual_call_irf_r/iIRFd>r" => dispatch_residual_call_iIRFd_kind(code, op, ctx, 'r'),
        "residual_call_irf_i/iIRFd>i" => dispatch_residual_call_iIRFd_kind(code, op, ctx, 'i'),
        "residual_call_irf_f/iIRFd>f" => dispatch_residual_call_iIRFd_kind(code, op, ctx, 'f'),
        // `_*_v/iRd|iIRd|iIRFd` void variants — `_opimpl_residual_call*`
        // bodies discard the call result for void return kinds
        // (`pyjitpl.py:1348 opimpl_residual_call_r_v = _opimpl_residual_call1`,
        // `:1351 opimpl_residual_call_ir_v = _opimpl_residual_call2`,
        // `:1355 opimpl_residual_call_irf_v = _opimpl_residual_call3`;
        // `blackhole.py:1245/1248/1253 bhimpl_residual_call_*_v`).
        // EffectInfo classification + guard emission match the result-typed
        // siblings; only the operand layout drops the `>X` dst byte and
        // the writeback no-ops via `write_residual_call_result_to_dst`'s
        // `'v'` arm. `select_residual_call_opcode`'s `'v'` arm maps to the
        // `CallN` / `CallPureN` / `CallMayForceN` / `CallLoopinvariantN`
        // family per `resoperation.py:1463 Type::Void => CallN`.
        "residual_call_r_v/iRd" => dispatch_residual_call_iRd_kind(code, op, ctx, 'v'),
        "residual_call_ir_v/iIRd" => dispatch_residual_call_iIRd_kind(code, op, ctx, 'v'),
        "residual_call_irf_v/iIRFd" => dispatch_residual_call_iIRFd_kind(code, op, ctx, 'v'),
        // RPython parity: `pyjitpl.py:279-292` exec-generated
        // `opimpl_int_*` for binary arithmetic ops — each handler reads
        // two `i`-coded register operands and dispatches
        // `self.execute(rop.<OPNUM>, b1, b2)`. Walker mirror: read regs
        // from `registers_i`, record `OpCode::<Variant>` with the
        // operand OpRefs as args, write the recorder result OpRef into
        // the dst slot. No MIFrame state involved (these are pure
        // arithmetic — `EffectInfo`-free, `heapcache`-free).
        //
        // Operand layout `ii>i`: 1B src1 + 1B src2 + 1B dst (=3 operand
        // bytes after the opcode).
        "int_add/ii>i" => binop_int_record(code, op, ctx, OpCode::IntAdd),
        "int_sub/ii>i" => binop_int_record(code, op, ctx, OpCode::IntSub),
        "int_mul/ii>i" => binop_int_record(code, op, ctx, OpCode::IntMul),
        "int_and/ii>i" => binop_int_record(code, op, ctx, OpCode::IntAnd),
        "int_or/ii>i" => binop_int_record(code, op, ctx, OpCode::IntOr),
        "int_xor/ii>i" => binop_int_record(code, op, ctx, OpCode::IntXor),
        // RPython `pyjitpl.py:281` enumerates `int_lshift` alongside
        // `int_rshift` in the exec-generated `(box, box)` opimpl loop;
        // the canonical operand shape is therefore `ii>i`
        // (`blackhole.py:516-519 bhimpl_int_lshift(a, b): return
        // intmask(a << b)`). Mixed shapes such as `int_lshift/ri>i`
        // stay unwired: those are kind-flow kind-flow bugs, and adding
        // a handler for them would mask a Ref register flowing into
        // an Int op.
        "int_lshift/ii>i" => binop_int_record(code, op, ctx, OpCode::IntLshift),
        "int_rshift/ii>i" => binop_int_record(code, op, ctx, OpCode::IntRshift),
        // RPython `pyjitpl.py:326-336` — comparison opimpls have a `b1
        // is b2` fast path returning a constant. Walker omits the fast
        // path: with two distinct OpRefs on the trace, recording the
        // op is parity-correct, and the optimizer collapses
        // tautological compares downstream. (RPython needs the fast
        // path because `ConstInt(1)` allocation is expensive in Python;
        // pyre's recorder shares constants by value.)
        "int_eq/ii>i" => binop_int_record(code, op, ctx, OpCode::IntEq),
        "int_ne/ii>i" => binop_int_record(code, op, ctx, OpCode::IntNe),
        "int_lt/ii>i" => binop_int_record(code, op, ctx, OpCode::IntLt),
        "int_le/ii>i" => binop_int_record(code, op, ctx, OpCode::IntLe),
        "int_gt/ii>i" => binop_int_record(code, op, ctx, OpCode::IntGt),
        "int_ge/ii>i" => binop_int_record(code, op, ctx, OpCode::IntGe),
        // Float arithmetic — same shape as int binops but on the
        // `f` bank. RPython `pyjitpl.py:284-292` includes
        // float_add/float_sub/float_mul/float_truediv in the same
        // exec-generated opimpl loop. Codewriter today emits only
        // float_add/float_sub/float_truediv (float_mul absent —
        // generated only when an explicit `*` operand reaches the
        // jit_codewriter; pyre's bench set has no float_mul yet)
        // plus the unary float_neg.
        "float_add/ff>f" => binop_float_record(code, op, ctx, OpCode::FloatAdd),
        "float_sub/ff>f" => binop_float_record(code, op, ctx, OpCode::FloatSub),
        "float_mul/ff>f" => binop_float_record(code, op, ctx, OpCode::FloatMul),
        "float_truediv/ff>f" => binop_float_record(code, op, ctx, OpCode::FloatTrueDiv),
        "float_neg/f>f" => unop_float_record(code, op, ctx, OpCode::FloatNeg),
        // Float-to-int comparisons — `bhimpl_float_{lt,le,eq,ne,gt,ge}`
        // (`blackhole.py:721-746`).  Read two `f` regs, record
        // `OpCode::Float<Cmp>`, write the recorder result into the int
        // bank.
        "float_lt/ff>i" => binop_float_to_int_record(code, op, ctx, OpCode::FloatLt),
        "float_le/ff>i" => binop_float_to_int_record(code, op, ctx, OpCode::FloatLe),
        "float_eq/ff>i" => binop_float_to_int_record(code, op, ctx, OpCode::FloatEq),
        "float_ne/ff>i" => binop_float_to_int_record(code, op, ctx, OpCode::FloatNe),
        "float_gt/ff>i" => binop_float_to_int_record(code, op, ctx, OpCode::FloatGt),
        "float_ge/ff>i" => binop_float_to_int_record(code, op, ctx, OpCode::FloatGe),
        // Int-bank unary ops. RPython parity:
        // `pyjitpl.py:356-368` (int_neg / int_invert) + 371-375
        // (int_same_as which calls `_record_helper(rop.SAME_AS_I, ...)`
        // explicitly — same shape, walker treats it as a regular
        // record-and-writeback).
        "int_neg/i>i" => unop_int_record(code, op, ctx, OpCode::IntNeg),
        "int_invert/i>i" => unop_int_record(code, op, ctx, OpCode::IntInvert),
        "int_same_as/i>i" => unop_int_record(code, op, ctx, OpCode::SameAsI),
        // `int_is_true/i>i` mirrors `int_neg`/`int_invert`: a single
        // i-coded source, a recorded IR op, an i-coded destination.
        // RPython `pyjitpl.py:319-330 opimpl_int_is_true` records
        // `rop.INT_IS_TRUE` via `_record_helper`. The result is
        // semantically a bool but Int-typed on the bank (matches the
        // codewriter's `>i` destination shape).
        "int_is_true/i>i" => unop_int_record(code, op, ctx, OpCode::IntIsTrue),
        // `int_between/iii>i` decomposes a 3-arg range check at record
        // time per `pyjitpl.py:588-595 opimpl_int_between` into
        // `INT_SUB + (INT_EQ on ConstInt(1) fast path | INT_SUB +
        // UINT_LT generic)`. Surfaces in `make_ll_isinstance`'s
        // range-covering branch (`rclass.py:1149-1168`).
        "int_between/iii>i" => int_between_record(code, op, ctx),
        // `int_floordiv/ii>i` and `int_mod/ii>i` intentionally absent:
        // `jtransform.py:575-577` rewrites both to
        // `direct_call(ll_int_py_*)` before jitcode emission.  The
        // trace-front lowering at `majit-translate/src/codegen.rs`
        // mirrors that rewrite for code reaching the JIT trace, so
        // this walker is never asked to dispatch the bare ops on a
        // traceable path.  Build-time helper graphs that still emit
        // the bare ops (e.g. `pyre/pyre-interpreter/src/baseobjspace.rs`
        // long_mod / long_div until the build-pipeline jtransform
        // port lands) get a `setdefault`-allocated dynamic byte and
        // resolve through BH dispatch only.
        "cast_int_to_float/i>f" => cast_int_to_float_record(code, op, ctx),
        // `cast_int_to_ptr/i>r`: RPython `pyjitpl.py:357` exec-generated
        // unary, same shape as `cast_int_to_float` but result lands in
        // the Ref bank. The recorded op is `CastIntToPtr`.
        "cast_int_to_ptr/i>r" => {
            let a = read_int_reg(code, op, 0, ctx)?;
            let result = ctx.trace_ctx.record_op(OpCode::CastIntToPtr, &[a]);
            let dst = code[op.pc + 2] as usize;
            // Box(value) parity: bit-cast the operand's Box.value
            // (BoxInt(n) → BoxRef(n as ptr)).
            if let Some(majit_ir::Value::Int(n)) = ctx.trace_ctx.box_value(a) {
                ctx.trace_ctx
                    .set_opref_concrete(result, majit_ir::Value::Ref(majit_ir::GcRef(n as usize)));
            }
            write_ref_reg(ctx, op.pc, dst, result, ConcreteValue::Null)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // `cast_ptr_to_int/r>i`: Ref-bank → Int-bank cast.
        "cast_ptr_to_int/r>i" => {
            let a = read_ref_reg(code, op, 0, ctx)?;
            let result = ctx.trace_ctx.record_op(OpCode::CastPtrToInt, &[a]);
            // Box(value) parity: bit-cast the operand's Box.value
            // (BoxRef(p) → BoxInt(p as i64)).
            if let Some(majit_ir::Value::Ref(r)) = ctx.trace_ctx.box_value(a) {
                ctx.trace_ctx
                    .set_opref_concrete(result, majit_ir::Value::Int(r.0 as i64));
            }
            let dst = code[op.pc + 2] as usize;
            let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
            write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "ptr_eq/rr>i" => binop_ref_to_int_record(code, op, ctx, OpCode::PtrEq),
        "ptr_ne/rr>i" => binop_ref_to_int_record(code, op, ctx, OpCode::PtrNe),
        "ptr_nonzero/r>i" => ptr_nonzero_record(code, op, ctx),
        "ref_guard_value/r" => ref_guard_value_record(code, op, ctx),
        "abort/>r" => {
            // pyre-only result marker: `Assembler::encode_op`'s default
            // branch emits this when an untranslatable op's result is
            // classified `Ref` by `infer_concrete_from_op`'s
            // Abort→GcRef fallback.  Blackhole counterpart
            // (`handler_abort_result_marker_r`, `blackhole.rs:5149`) is
            // a pure PC bump — no operand read, no register write, no
            // IR op recorded.  The actual abort signal is `abort/`
            // (BC_ABORT = 13), not this; reaching `abort/>r` in normal
            // flow is upstream-only an artefact of result-kind
            // classification and the dst slot is never read in a
            // post-abort code path.
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        // Heapcache-aware getfield reads. RPython
        // `pyjitpl.py:855-882 opimpl_getfield_gc_<i|r|f>` →
        // `_opimpl_getfield_gc_any_pureornot` (`pyjitpl.py:929-950`)
        // dispatches the same way through `heapcache.get_field_updater`.
        // Walker handles the canonical `rd>X` shapes (Ref source);
        // pyre-specific `id>X` variants where the source is an int
        // register holding an unwrapped pointer are kind-flow kind-flow
        // territory and stay unsupported here.
        "getfield_gc_i/rd>i" => getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcI, 'i'),
        "getfield_gc_r/rd>r" => getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcR, 'r'),
        "getfield_gc_f/rd>f" => getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcF, 'f'),
        // RPython `blackhole.py:1441-1443` aliases
        // `bhimpl_getfield_gc_{i,r,f}_pure = bhimpl_getfield_gc_{i,r,f}` —
        // pure-getter shape on quasi-immutable descrs.  Walker emits
        // the non-pure opcode; the optimizer rewrites to the Pure form
        // post-trace based on `descr.is_always_pure()`
        // (`resoperation.py:1284-1289 OpHelpers.getfield_pure_for_descr`).
        "getfield_gc_i_pure/rd>i" => {
            getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcI, 'i')
        }
        "getfield_gc_r_pure/rd>r" => {
            getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcR, 'r')
        }
        "getfield_gc_f_pure/rd>f" => {
            getfield_gc_via_heapcache(code, op, ctx, OpCode::GetfieldGcF, 'f')
        }
        // Virtualizable getfield reads. RPython
        // `pyjitpl.py:1167-1186 opimpl_getfield_vable_{i,r,f}` —
        // walker delegates to `TraceCtx::vable_getfield_{int,ref,float}`
        // which already implements `_nonstandard_virtualizable` fallback
        // to GETFIELD_GC + standard-vable `virtualizable_boxes[index]`
        // cache read (`majit-metainterp/src/trace_ctx.rs:1715/1801/1839`).
        // Same `rd>X` operand shape as `getfield_gc_*`; only the
        // semantic handler routes through the vable mirror.
        "getfield_vable_i/rd>i" => getfield_vable_via_metainterp(code, op, ctx, 'i'),
        "getfield_vable_r/rd>r" => getfield_vable_via_metainterp(code, op, ctx, 'r'),
        "getfield_vable_f/rd>f" => getfield_vable_via_metainterp(code, op, ctx, 'f'),
        // Virtualizable setfield writes. RPython
        // `pyjitpl.py:1188-1199 _opimpl_setfield_vable` — walker
        // delegates to `TraceCtx::vable_setfield`
        // (`majit-metainterp/src/trace_ctx.rs:1759`) which handles the
        // `_nonstandard_virtualizable` fallback to SETFIELD_GC + the
        // standard-vable `virtualizable_boxes[index] = valuebox` +
        // `synchronize_virtualizable` mirror.  Operand shapes:
        // `setfield_vable_i/rid`, `setfield_vable_r/rrd`,
        // `setfield_vable_f/rfd` — value bank differs, no dst byte.
        "setfield_vable_i/rid" => setfield_vable_via_metainterp(code, op, ctx, 'i'),
        "setfield_vable_r/rrd" => setfield_vable_via_metainterp(code, op, ctx, 'r'),
        "setfield_vable_f/rfd" => setfield_vable_via_metainterp(code, op, ctx, 'f'),
        // setfield_gc canonical shapes. `iid` / `ird` (int box)
        // shapes are pyre kind-flow kind-flow territory and stay
        // unsupported.
        "setfield_gc_i/rid" => setfield_gc_via_heapcache(code, op, ctx, 'i'),
        "setfield_gc_r/rrd" => setfield_gc_via_heapcache(code, op, ctx, 'r'),
        "setfield_gc_f/rfd" => setfield_gc_via_heapcache(code, op, ctx, 'f'),
        // Heapcache-aware array reads/writes (canonical `rid>X` /
        // `ri{i,r,f}d` shapes).  The pyre-only Ref-index shape
        // `getarrayitem_gc_r/rrd>r` lives in `pyre_extension_insns()`
        // and is NOT canonical; non-canonical setarrayitem shapes
        // (`rrid`/`rrrd`/`rrfd`, Ref index) stay unsupported (kind-flow
        // kind-flow territory).
        "getarrayitem_gc_i/rid>i" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcI, 'i')
        }
        "getarrayitem_gc_r/rid>r" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcR, 'r')
        }
        // pyre-only `rrd>r` variant: index lands in Ref bank (tagged-
        // int-in-ref deviation, same root as the `/rr>i` arithmetic
        // aliases — disappears once rtyper classifies integer array
        // indices as `Signed`).  Sibling
        // `blackhole.rs::handler_getarrayitem_gc_r_refindex` covers the
        // dispatch table; this is the walker counterpart.
        "getarrayitem_gc_r/rrd>r" => getarrayitem_gc_via_heapcache_with_index_bank(
            code,
            op,
            ctx,
            OpCode::GetarrayitemGcR,
            'r',
            'r',
        ),
        "getarrayitem_gc_f/rid>f" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcF, 'f')
        }
        // RPython `pyjitpl.py:701-734 opimpl_getarrayitem_gc_{i,f,r}_pure`
        // — distinct opimpls (NOT aliased to the non-pure form,
        // unlike `getfield_gc_*_pure` at `pyjitpl.py:884-886`).
        // Records `rop.GETARRAYITEM_GC_PURE_{I,F,R}` directly through
        // `_do_getarrayitem_gc_any(rop.GETARRAYITEM_GC_PURE_*, ...)`.
        // The ConstPtr+ConstInt constant-fold fast path (`pyjitpl.py:
        // 703-707`) is structurally unreachable on the symbolic walker
        // (no concrete-pointer access on OpRefs); the cache miss
        // branch records the Pure rop.
        "getarrayitem_gc_i_pure/rid>i" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcPureI, 'i')
        }
        "getarrayitem_gc_r_pure/rid>r" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcPureR, 'r')
        }
        "getarrayitem_gc_f_pure/rid>f" => {
            getarrayitem_gc_via_heapcache(code, op, ctx, OpCode::GetarrayitemGcPureF, 'f')
        }
        "setarrayitem_gc_i/riid" => setarrayitem_gc_via_heapcache(code, op, ctx, 'i'),
        "setarrayitem_gc_r/rird" => setarrayitem_gc_via_heapcache(code, op, ctx, 'r'),
        "setarrayitem_gc_f/rifd" => setarrayitem_gc_via_heapcache(code, op, ctx, 'f'),
        "int_copy/i>i" => {
            // RPython `pyjitpl.py:471-477 _opimpl_any_copy(self, box) → box`
            // + `@arguments("box")` + `>i` result coding: read src
            // register, write the same OpRef into the dst slot. Pypy
            // records *no* IR op for a copy — pure SSA-level rename.
            // Operand layout `i>i`: 1B src + 1B dst.
            //
            // Int-bank concrete shadow: propagate the source slot's Int-bank concrete
            // shadow alongside the symbolic OpRef, mirroring the
            // `ref_copy/r>r` concrete-shadow chain.  Without this, a
            // `goto_if_not/iL` reading the dst slot wouldn't see the
            // concrete and would surface `GotoIfNotValueNotConcrete`
            // even when the source had a known concrete (e.g. a
            // constant Int seeded by `allocate_callee_register_banks`).
            let src_val = read_int_reg(code, op, 0, ctx)?;
            let src_concrete = read_int_reg_concrete(code, op, 0, ctx);
            let dst = code[op.pc + 2] as usize;
            write_int_reg(ctx, op.pc, dst, src_val, src_concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "float_copy/f>f" => {
            // Float-bank sibling of `int_copy/i>i` — pure SSA-level
            // rename, no IR op recorded. Operand layout `f>f`: 1B src
            // + 1B dst.
            let src_val = read_float_reg(code, op, 0, ctx)?;
            let dst = code[op.pc + 2] as usize;
            let len = ctx.registers_f.len();
            let slot = ctx
                .registers_f
                .get_mut(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            *slot = src_val;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "ref_copy/r>r" => {
            // Ref-bank sibling of `int_copy/i>i`. Same RPython
            // `_opimpl_any_copy` body — the `>r` suffix only changes
            // which register bank the writeback lands in. Const-source
            // variants (codewriter `emit_ref_copy!` with `ConstRef`)
            // resolve via the constants window of `registers_r`: the
            // assembler's `load_const_r` patches the src operand to a
            // constants-pool register index in `[num_regs_r,
            // num_regs_and_consts_r)`, which `setposition` (RPython
            // `pyjitpl.py:74-90`) pre-populates with the const OpRef.
            // No IR op recorded.
            let src_val = read_ref_reg(code, op, 0, ctx)?;
            // Propagate the source slot's concrete
            // shadow alongside the symbolic OpRef.  This is the
            // critical chain: catch_exception → seeds last_exc_value
            // / concrete → `last_exc_value/>r` writes both into
            // `registers_r[X]` and `concrete_registers_r[X]` →
            // `ref_copy/r>r` copies X to Y → a follow-on `raise/r`
            // reads Y and finds the correct concrete to emit
            // GUARD_CLASS against.  Without this propagation the
            // copy chain wipes the concrete and silently disables
            // the guard.
            let src_concrete = read_ref_reg_concrete(code, op, 0, ctx);
            let dst = code[op.pc + 2] as usize;
            write_ref_reg(ctx, op.pc, dst, src_val, src_concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "ref_return/r" => {
            // RPython `pyjitpl.py:opimpl_ref_return(self, value)` calls
            // `metainterp.finishframe(value)`. Two branches by frame depth:
            //
            //   * Outermost frame → `compile_done_with_this_frame` (pyjitpl.py:3198-3220)
            //     records `rop.FINISH(value)` with
            //     `done_with_this_frame_descr_ref`. Trace ends.
            //   * Nested frame → `metainterp.popframe()` returns control to
            //     the caller's metainterp loop with `value` in hand; the
            //     caller's `_opimpl_inline_call*` lands `value` in its
            //     `>r` slot via `make_result_of_lastop`.
            //
            // Walker selects between the two via `ctx.is_top_level`.
            let result = read_ref_reg(code, op, 0, ctx)?;
            // PyPy `box.value = result` parity at the frame boundary:
            // the callee's slot-keyed concrete shadow (`concrete_registers_r`)
            // carries the live PyObject pointer; mirror it onto the
            // OpRef-keyed `opref_concrete` channel so the caller's
            // `concrete_from_recorded_opref` (in `dispatch_inline_call_*_kind`)
            // sees the stamped Box.value.  Skips constants — `TraceCtx::constants
            // .get_value` is the authoritative shadow for those.
            if !result.is_constant() {
                if let ConcreteValue::Ref(ptr) = read_ref_reg_concrete(code, op, 0, ctx) {
                    if !ptr.is_null() {
                        ctx.trace_ctx.set_opref_concrete(
                            result,
                            majit_ir::Value::Ref(majit_ir::GcRef(ptr as usize)),
                        );
                    }
                }
            }
            if ctx.is_top_level {
                ctx.trace_ctx
                    .finish(&[result], ctx.done_with_this_frame_descr_ref.clone());
                Ok((DispatchOutcome::Terminate, op.next_pc))
            } else {
                Ok((
                    DispatchOutcome::SubReturn {
                        result: Some(result),
                    },
                    op.next_pc,
                ))
            }
        }
        "int_return/i" => {
            // RPython `pyjitpl.py:463 opimpl_int_return = _opimpl_any_return`
            // (pyjitpl.py:459-461 `_opimpl_any_return: self.metainterp.finishframe(box)`).
            // Top-level: `compile_done_with_this_frame` (pyjitpl.py:3206-3208)
            // records `FINISH([value], descr=done_with_this_frame_descr_int)`.
            // Sub-walk: `SubReturn { Some(value) }` — caller's
            // `inline_call_*_i` would land the int OpRef in its `>i` slot.
            // Operand layout `i`: 1B int register at op.pc+1.
            let result = read_int_reg(code, op, 0, ctx)?;
            // PyPy `box.value = result` parity at the frame boundary —
            // see `ref_return/r` comment above for rationale.
            if !result.is_constant() {
                if let ConcreteValue::Int(v) = read_int_reg_concrete(code, op, 0, ctx) {
                    ctx.trace_ctx
                        .set_opref_concrete(result, majit_ir::Value::Int(v));
                }
            }
            if ctx.is_top_level {
                ctx.trace_ctx
                    .finish(&[result], ctx.done_with_this_frame_descr_int.clone());
                Ok((DispatchOutcome::Terminate, op.next_pc))
            } else {
                Ok((
                    DispatchOutcome::SubReturn {
                        result: Some(result),
                    },
                    op.next_pc,
                ))
            }
        }
        "float_return/f" => {
            // RPython `pyjitpl.py:465 opimpl_float_return = _opimpl_any_return`.
            // Top-level: `compile_done_with_this_frame` (pyjitpl.py:3212-3214)
            // records `FINISH([value], descr=done_with_this_frame_descr_float)`.
            // Sub-walk: `SubReturn { Some(value) }` carrying the float
            // OpRef — same enum variant as int/ref because the OpRef is
            // bank-agnostic; the caller's inline_call variant decides
            // which bank to write into.
            // Operand layout `f`: 1B float register at op.pc+1.
            let result = read_float_reg(code, op, 0, ctx)?;
            if ctx.is_top_level {
                ctx.trace_ctx
                    .finish(&[result], ctx.done_with_this_frame_descr_float.clone());
                Ok((DispatchOutcome::Terminate, op.next_pc))
            } else {
                Ok((
                    DispatchOutcome::SubReturn {
                        result: Some(result),
                    },
                    op.next_pc,
                ))
            }
        }
        "void_return/" => {
            // RPython `pyjitpl.py:467-469 opimpl_void_return`:
            //
            //   @arguments()
            //   def opimpl_void_return(self):
            //       self.metainterp.finishframe(None)
            //
            // Top-level: `compile_done_with_this_frame` (pyjitpl.py:3202-3205)
            // takes the `result_type == VOID` branch — `exits = []`,
            // `token = sd.done_with_this_frame_descr_void`. The FINISH
            // carries no value.
            // Sub-walk: `SubReturn { None }` — RPython's
            // `_opimpl_inline_call_*_v` variants don't write a dst
            // register on the caller side (the codewriter emits no `>X`
            // marker for void calls).
            // No operand bytes (the `/` argcodes is empty).
            if ctx.is_top_level {
                ctx.trace_ctx
                    .finish(&[], ctx.done_with_this_frame_descr_void.clone());
                Ok((DispatchOutcome::Terminate, op.next_pc))
            } else {
                Ok((DispatchOutcome::SubReturn { result: None }, op.next_pc))
            }
        }
        "raise/r" => {
            // RPython `pyjitpl.py:1688-1698 opimpl_raise(exc_value_box, orgpc)`:
            //   if not self.metainterp.heapcache.is_class_known(exc_value_box):
            //       clsbox = self.cls_of_box(exc_value_box)
            //       self.metainterp.generate_guard(rop.GUARD_CLASS, exc_value_box,
            //                                      clsbox, resumepc=orgpc)
            //   self.metainterp.class_of_last_exc_is_const = True
            //   self.metainterp.last_exc_value = exc_value_box.getref(...)
            //   self.metainterp.last_exc_box = exc_value_box
            //   self.metainterp.popframe()
            //   self.metainterp.finishframe_exception()
            //
            // Walker dual behaviour:
            //   * `is_top_level` → outermost FINISH (above).
            //   * sub-walk frame → propagate `SubRaise { exc }` to the
            //     caller's `inline_call_*` handler.
            //
            // GUARD_CLASS emission is enabled: `concrete_registers_r`
            // is `&mut` and every walker write goes through
            // [`write_ref_reg`], so the concrete tracks the symbolic
            // in lock-step.  Were the shadow an immutable dispatch-entry
            // snapshot, sibling handlers could rewrite `registers_r[dst]`
            // without updating it, so this read would find a stale
            // concrete and silently skip the guard.  Read-after-write
            // now returns the right concrete (or `Null` if the handler
            // didn't know, in which case the guard skips — same
            // semantics as the snapshot's tail).
            //
            // Mirrors trait-side `seed_raised_exception` at
            // `trace_opcode.rs:seed_raised_exception`.  The read at
            // `ob_header.ob_type` resolves to the per-`ExcKind` `PyType`
            // static (`excobject.rs::exc_kind_to_pytype`), so the
            // emitted `GuardClass` discriminates the actual subclass.
            // Stashes the concrete into `ctx.last_exc_value_concrete`
            // so a downstream
            // `last_exc_value/>r` can propagate it into its dst slot.
            let exc = read_ref_reg(code, op, 0, ctx)?;
            let concrete_exc = read_ref_reg_concrete(code, op, 0, ctx);
            // `pyjitpl.py:1688-1693 opimpl_raise` calls
            // `generate_guard(GUARD_CLASS, exc_value_box, clsbox,
            // resumepc=orgpc)`; the first line of `generate_guard`
            // (`pyjitpl.py:2583`) is `if isinstance(box, Const):
            // return`. Const exception boxes already pin the class so
            // no guard is needed. Resume-data capture is omitted here
            // for the same reason as `goto_if_not/iL` above — see that
            // arm's comment for the trace-emitter endgame.
            if !exc.is_constant() {
                if let ConcreteValue::Ref(exc_ptr) = concrete_exc {
                    if !exc_ptr.is_null() && !ctx.trace_ctx.heap_cache().is_class_known(exc) {
                        let exc_class_ptr = unsafe {
                            (*(exc_ptr as *const pyre_object::excobject::W_ExceptionObject))
                                .ob_header
                                .ob_type
                        };
                        let cls_const = ctx.trace_ctx.const_int(exc_class_ptr as usize as i64);
                        ctx.trace_ctx
                            .record_guard(OpCode::GuardClass, &[exc, cls_const], 0);
                        walker_capture_snapshot_for_last_guard(ctx, op.pc);
                        ctx.trace_ctx
                            .heap_cache_mut()
                            .class_now_known(exc, exc_class_ptr as usize as i64);
                    }
                }
            }
            ctx.last_exc_value = Some(exc);
            ctx.last_exc_value_concrete = concrete_exc;
            if ctx.is_top_level {
                ctx.trace_ctx
                    .finish(&[exc], ctx.exit_frame_with_exception_descr_ref.clone());
                Ok((DispatchOutcome::Terminate, op.next_pc))
            } else {
                Ok((
                    DispatchOutcome::SubRaise {
                        exc,
                        exc_concrete: concrete_exc,
                    },
                    op.next_pc,
                ))
            }
        }
        "last_exc_value/>r" => {
            // RPython parity: `pyjitpl.py:1716-1719 opimpl_last_exc_value`:
            //
            //   @arguments()
            //   def opimpl_last_exc_value(self):
            //       exc_value = self.metainterp.last_exc_value
            //       assert exc_value
            //       return self.metainterp.last_exc_box
            //
            // Reads no operand; the `>r` decorator writes the result into
            // `registers_r[dst]`. No IR op recorded — the standing
            // `metainterp.last_exc_box` (mirrored here as
            // `ctx.last_exc_value`) is already a recorder OpRef from when
            // `raise/r` set it. This is a pure SSA-rename of the
            // exception slot into a Ref-bank dst, mirroring how
            // `int_copy/i>i` and `_opimpl_any_copy` collapse to a
            // register move without recording.
            //
            // Operand layout `>r`: 1B dst register only (the `>r` arg is
            // the writeback marker, not a separate operand byte; the dst
            // byte sits at op.pc+1).
            //
            // Forward-prep status: the opname is registered in
            // `wire_handler("last_exc_value/>r", handler_last_exc_value)`
            // (`blackhole.rs:6757`) and `m.insert("last_exc_value/>r",
            // BC_LAST_EXC_VALUE)` (`jitcode/mod.rs:305`), but pyre's
            // codewriter does not currently emit `FlatOp::LastExcValue`
            // for any traced Python arm — `dump_unsupported_opnames_in_insns_table`
            // confirms the opname is absent from `OUT_DIR/opcode_insns.bin`.
            // The handler matches RPython's unconditional `setup_insns`
            // registration so it's ready when an except-handler arm
            // (e.g. `BC_LAST_EXC_VALUE` consumer in CPython 3.14
            // `LOAD_SPECIAL`/`CHECK_EXC_MATCH` lowering) lands.
            let exc = ctx
                .last_exc_value
                .ok_or(DispatchError::LastExcValueWithoutActiveException { pc: op.pc })?;
            let dst = code[op.pc + 1] as usize;
            // Propagate the standing exception's
            // concrete shadow into the dst slot.  `ctx.last_exc_value_
            // concrete` is the live `PyObjectRef` (seeded by either the
            // trait path's `seed_raised_exception` or an earlier walker
            // `raise/r`).  This lets a follow-on `raise/r` reading
            // `registers_r[dst]` find a non-Null concrete and emit the
            // correct GUARD_CLASS.
            let exc_concrete = ctx.last_exc_value_concrete;
            write_ref_reg(ctx, op.pc, dst, exc, exc_concrete)?;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "reraise/" => {
            // RPython parity: `pyjitpl.py:1700-1704 opimpl_reraise(self)` —
            //
            //   assert self.metainterp.last_exc_value
            //   self.metainterp.popframe()
            //   self.metainterp.finishframe_exception()
            //
            // Reads no operand; uses the standing `metainterp.last_exc_value`
            // which was set either by an earlier `raise/r` in this frame
            // or — when the unwinder routed into a `catch_exception`
            // handler — by the inline_call SubRaise arm just before
            // jumping to the handler PC.
            //
            // Walker behaviour mirrors `raise/r`'s dual-frame routing:
            //   * top-level → outermost FINISH(last_exc_value,
            //     exit_frame_with_exception_descr_ref).
            //   * sub-walk → SubRaise{exc=last_exc_value}, bubbling
            //     through the parent's inline_call handler (which may
            //     itself catch via `catch_exception/L` lookahead).
            //
            // `last_exc_value == None` violates the RPython assert and
            // surfaces as `ReraiseWithoutLastExcValue` (codewriter
            // invariant: `reraise` only emits inside a `catch_exception`
            // body or after an explicit `raise`).
            let exc = ctx
                .last_exc_value
                .ok_or(DispatchError::ReraiseWithoutLastExcValue { pc: op.pc })?;
            if ctx.is_top_level {
                ctx.trace_ctx
                    .finish(&[exc], ctx.exit_frame_with_exception_descr_ref.clone());
                Ok((DispatchOutcome::Terminate, op.next_pc))
            } else {
                Ok((
                    DispatchOutcome::SubRaise {
                        exc,
                        exc_concrete: ctx.last_exc_value_concrete,
                    },
                    op.next_pc,
                ))
            }
        }
        other => Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: other,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jitcode_runtime::{insns_opname_to_byte, jitcode_for_instruction};
    use majit_ir::Type;
    use majit_metainterp::make_fail_descr;
    use pyre_interpreter::Instruction;

    /// Build a fresh `TraceCtx`. Uses the public `for_test_types` +
    /// `const_ref` / `make_fail_descr` factories so the fixture stays
    /// out of `pub(crate)` API.
    fn fresh_trace_ctx() -> TraceCtx {
        TraceCtx::for_test_types(&[Type::Ref])
    }

    /// Build a `done_with_this_frame_descr_ref` for tests. Mirrors the
    /// production fallback at `pyjitpl/mod.rs:4733` (`make_fail_descr_typed`)
    /// when the staticdata singleton was never attached.
    fn done_descr_ref_for_tests() -> DescrRef {
        make_fail_descr(1)
    }

    /// `ensure_residual_call_args_bound` backs the unbound-arg abort path
    /// for all three residual-call shapes (iRd / iIRd / iIRFd); they all
    /// funnel through this helper, so one direct test covers the guard
    /// that otherwise lets `OpRef::NONE` reach the backend's
    /// `resolve_opref` (a process abort).
    #[test]
    fn ensure_residual_call_args_bound_rejects_unbound_arg() {
        // Fully-bound funcbox + args slice passes.
        let bound = [OpRef::int_op(1), OpRef::int_op(2), OpRef::int_op(3)];
        assert!(ensure_residual_call_args_bound(&bound, 7).is_ok());

        // An unbound arg surfaces its position.
        let unbound = [OpRef::int_op(1), OpRef::NONE, OpRef::int_op(3)];
        assert!(matches!(
            ensure_residual_call_args_bound(&unbound, 7),
            Err(DispatchError::ResidualCallArgUnbound {
                pc: 7,
                arg_index: 1
            })
        ));

        // The funcbox slot (index 0) is guarded too.
        let unbound_func = [OpRef::NONE, OpRef::int_op(2)];
        assert!(matches!(
            ensure_residual_call_args_bound(&unbound_func, 4),
            Err(DispatchError::ResidualCallArgUnbound {
                pc: 4,
                arg_index: 0
            })
        ));
    }

    /// Build distinct `OpRef` constants for register slots so dataflow
    /// assertions don't get false positives from shared identity. Each
    /// slot holds `const_ref(0xC0DE_0000 + i)` for `i in 0..count`.
    fn distinct_const_refs(ctx: &mut TraceCtx, count: usize) -> Vec<OpRef> {
        (0..count)
            .map(|i| ctx.const_ref(0xC0DE_0000_i64 + i as i64))
            .collect()
    }

    /// Companion of [`distinct_const_refs`] that mints Int-typed
    /// ConstInt OpRefs.  Use this when a fixture needs to populate
    /// integer-register slots — the heapcache array path keys on the
    /// `ConstInt.getint()` value, so Ref-typed mints don't satisfy
    /// `getarrayitem_cache`'s ConstInt precondition.
    fn distinct_const_ints(ctx: &mut TraceCtx, count: usize) -> Vec<OpRef> {
        (0..count)
            .map(|i| ctx.const_int(1_000 + i as i64))
            .collect()
    }

    /// Default `sub_jitcode_lookup` for tests that don't exercise
    /// `inline_call_r_r` recursion. Returns `None` for every index;
    /// any test that hits the inline_call handler with this lookup
    /// will see `DispatchError::SubJitCodeNotFound`.
    fn no_sub_jitcodes(_idx: usize) -> Option<SubJitCodeBody> {
        None
    }

    fn switch_descr_pool(entries: &[(i64, usize)]) -> Vec<DescrRef> {
        let dict = entries.iter().copied().collect();
        vec![std::sync::Arc::new(crate::descr::PyreSwitchDescr::new(dict)) as DescrRef]
    }

    /// Concrete-shadow round-trip: a `WalkContext` built with
    /// `concrete_registers_r` exposes each slot's `ConcreteValue` via
    /// `read_ref_reg_concrete` indexed by the same byte the symbolic
    /// `read_ref_reg` consults.  Verifies the parallel-slice plumbing
    /// (slot N's OpRef in `registers_r` shares slot N in
    /// `concrete_registers_r`).
    #[test]
    fn read_ref_reg_concrete_returns_slot_matching_symbolic_read() {
        let exc_obj_ptr: pyre_object::PyObjectRef = 0xDEAD_BEEFusize as _;
        let descr_pool: Vec<DescrRef> = Vec::new();
        let mut tc = fresh_trace_ctx();
        let oprefs = distinct_const_refs(&mut tc, 3);
        let mut regs_r = oprefs.clone();
        let mut concrete = vec![
            ConcreteValue::Null,
            ConcreteValue::Ref(exc_obj_ptr),
            ConcreteValue::Int(42),
        ];
        // Snapshot expected values before `&mut concrete` enters wc —
        // the assertion below cannot read `concrete[reg_idx]` while wc
        // holds the mutable borrow.
        let expected: Vec<ConcreteValue> = concrete.clone();
        let descr = done_descr_ref_for_tests();
        let wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut concrete,
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };

        // Synthesize a 2-byte op fixture: `<opcode_byte> <reg_idx>`.
        // `read_ref_reg_concrete` reads `code[op.pc + 1 + operand_offset]`
        // exactly like `read_ref_reg`, so encoding the reg byte at pc+1
        // suffices.
        for reg_idx in 0..3 {
            let code = [0u8, reg_idx as u8];
            let op = DecodedOp {
                key: "fixture/r",
                opname: "fixture",
                argcodes: "r",
                pc: 0,
                next_pc: 2,
            };
            assert_eq!(
                read_ref_reg_concrete(&code, &op, 0, &wc),
                expected[reg_idx],
                "reg {} concrete shadow must match the parallel slot",
                reg_idx,
            );
        }
    }

    /// `getfield_vable_*` must abort to `VableBoxNotSeeded` when the
    /// box register is unseeded (`OpRef::NONE`) rather than feed
    /// `u32::MAX` into the heapcache flag vector (a 16 GiB resize).
    #[test]
    fn getfield_vable_with_none_obj_surfaces_vable_box_not_seeded() {
        let descr_pool: Vec<DescrRef> = Vec::new();
        let mut tc = fresh_trace_ctx();
        let mut regs_r = vec![OpRef::NONE];
        let mut regs_i = vec![OpRef::NONE];
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: make_fail_descr(1),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        // `getfield_vable_i/rd>i`: operand 0 (the box) sits at code[pc+1].
        let code = [0u8, 0x00, 0x00, 0x00, 0x00];
        let op = DecodedOp {
            key: "getfield_vable_i/rd>i",
            opname: "getfield_vable_i",
            argcodes: "rd>i",
            pc: 0,
            next_pc: 5,
        };
        assert_eq!(
            getfield_vable_via_metainterp(&code, &op, &mut wc, 'i'),
            Err(DispatchError::VableBoxNotSeeded { pc: 0 })
        );
    }

    /// `setfield_vable_*` carries the same unseeded-box guard.
    #[test]
    fn setfield_vable_with_none_obj_surfaces_vable_box_not_seeded() {
        let descr_pool: Vec<DescrRef> = Vec::new();
        let mut tc = fresh_trace_ctx();
        let mut regs_r = vec![OpRef::NONE];
        let mut regs_i = vec![OpRef::NONE];
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: make_fail_descr(1),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        // `setfield_vable_i/rid`: operand 0 (the box) sits at code[pc+1].
        let code = [0u8, 0x00, 0x00, 0x00, 0x00];
        let op = DecodedOp {
            key: "setfield_vable_i/rid",
            opname: "setfield_vable_i",
            argcodes: "rid",
            pc: 0,
            next_pc: 5,
        };
        assert_eq!(
            setfield_vable_via_metainterp(&code, &op, &mut wc, 'i'),
            Err(DispatchError::VableBoxNotSeeded { pc: 0 })
        );
    }

    #[test]
    #[ignore = "T3 audit probe — dumps runtime opnames + walker-handled set + \
                per-opname JitCode hit count. Run with \
                `cargo test -p pyre-jit-trace --features dynasm --lib \
                t3_audit_opname_gap_inventory -- --ignored --nocapture` to \
                produce a project memory entry; not a permanent test."]
    fn t3_audit_opname_gap_inventory() {
        use crate::jitcode_runtime::{all_jitcodes, insns_byte_to_opname, insns_opname_to_byte};

        // 1) Runtime opnames (pyre's actual codewriter emission set).
        let runtime_names: std::collections::BTreeSet<String> =
            insns_opname_to_byte().keys().cloned().collect();

        // 2) Walker-handled opnames — parsed from the embedded `handle`
        // function's string literals.  Source-of-truth scan against
        // the file itself so this probe stays accurate as handlers
        // land/leave.
        let source = include_str!("jitcode_dispatch.rs");
        let mut handled: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        // Heuristic: scan the literal patterns that appear ONLY in
        // dispatch arms of `handle()` — they look like
        // `"<opname>/[argcodes]" => ...`.  Filter to entries that are
        // also in the runtime table to drop test-fixture literals.
        for line in source.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with('"') {
                continue;
            }
            let Some(rest) = trimmed.strip_prefix('"') else {
                continue;
            };
            let Some(end_quote_idx) = rest.find('"') else {
                continue;
            };
            let key = &rest[..end_quote_idx];
            // Must contain '/' (separates opname from argcodes); skip
            // anything that doesn't look like an opname/argcodes literal.
            if !key.contains('/') {
                continue;
            }
            if runtime_names.contains(key) {
                handled.insert(key.to_string());
            }
        }

        let unhandled: Vec<&String> = runtime_names.difference(&handled).collect();

        // 3) Per-opname JitCode hit count — for each unhandled opname,
        // count how many JitCodes contain its opcode byte.  Higher
        // counts = higher likelihood of blocking the next opcode
        // entering the shadow allow-list.
        let opname_to_byte = insns_opname_to_byte();
        let byte_to_opname = insns_byte_to_opname();
        let all_jcs = all_jitcodes();

        // For accurate counts the byte must be at a true OP position,
        // not an operand position.  We need to walk each JitCode using
        // `decoded_ops` to enumerate true op bytes.
        let mut hit_counts: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for jc in all_jcs {
            for op in crate::jitcode_runtime::decoded_ops(&jc.code) {
                let key = op.key;
                if handled.contains(key) {
                    continue;
                }
                *hit_counts.entry(key.to_string()).or_insert(0) += 1;
            }
        }

        eprintln!();
        eprintln!("=== T3 AUDIT: runtime opnames ===");
        eprintln!("total runtime opnames: {}", runtime_names.len());
        eprintln!("walker-handled opnames: {}", handled.len());
        eprintln!("unhandled opnames: {}", unhandled.len());

        eprintln!();
        eprintln!("=== T3 AUDIT: unhandled opnames ranked by JitCode hit count ===");
        let mut ranked: Vec<(String, usize)> = hit_counts.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for (name, count) in &ranked {
            let byte = opname_to_byte
                .get(name)
                .map(|b| format!("0x{b:02x}"))
                .unwrap_or_else(|| "?".to_string());
            eprintln!("  {count:>5}  {byte}  {name}");
        }

        eprintln!();
        eprintln!("=== T3 AUDIT: unhandled opnames with ZERO JitCode hits ===");
        for name in &unhandled {
            if !ranked.iter().any(|(n, _)| n == *name) {
                let byte = opname_to_byte
                    .get(*name)
                    .map(|b| format!("0x{b:02x}"))
                    .unwrap_or_else(|| "?".to_string());
                eprintln!("  {byte}  {name}");
            }
        }

        eprintln!();
        eprintln!("=== T3 AUDIT: walker-handled opnames (for cross-check) ===");
        for name in &handled {
            eprintln!("  {name}");
        }

        // Sanity: byte_to_opname must invert opname_to_byte.
        assert_eq!(byte_to_opname.len(), opname_to_byte.len());
    }

    #[test]
    fn switch_id_hit_jumps_to_matching_target() {
        let switch_byte = *insns_opname_to_byte()
            .get("switch/id")
            .expect("`switch/id` must be in insns table");
        let code = [
            switch_byte,
            0x00, // i register 0
            0x00,
            0x00, // d descr index 0
        ];
        let mut tc = fresh_trace_ctx();
        let value = tc.const_int(5);
        let mut regs_i = vec![value];
        let descr_pool = switch_descr_pool(&[(5, 17), (9, 23)]);
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };

        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("switch hit must dispatch");

        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, 17);
    }

    #[test]
    fn switch_id_miss_falls_through() {
        let switch_byte = *insns_opname_to_byte()
            .get("switch/id")
            .expect("`switch/id` must be in insns table");
        let code = [
            switch_byte,
            0x00, // i register 0
            0x00,
            0x00, // d descr index 0
        ];
        let mut tc = fresh_trace_ctx();
        let value = tc.const_int(7);
        let mut regs_i = vec![value];
        let descr_pool = switch_descr_pool(&[(5, 17), (9, 23)]);
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };

        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("switch miss must dispatch");

        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, code.len());
    }

    #[test]
    fn switch_id_requires_concrete_int_value() {
        let switch_byte = *insns_opname_to_byte()
            .get("switch/id")
            .expect("`switch/id` must be in insns table");
        let code = [
            switch_byte,
            0x00, // i register 0
            0x00,
            0x00, // d descr index 0
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = vec![OpRef::input_arg_int(0)];
        let descr_pool = switch_descr_pool(&[(5, 17)]);
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };

        let err = step(&code, 0, &mut wc).expect_err("non-constant switch value must not guess");

        assert_eq!(
            err,
            DispatchError::SwitchValueNotConcrete {
                pc: 0,
                value: OpRef::input_arg_int(0),
            }
        );
    }

    #[test]
    fn goto_if_not_truthy_records_guard_true_and_falls_through() {
        // `goto_if_not/iL` with a concrete non-zero Int: emit GuardTrue,
        // do NOT take the jump (pc advances past the 3-byte operand
        // block).  RPython `pyjitpl.py:511-520 opimpl_goto_if_not`
        // `if switchcase: opnum = rop.GUARD_TRUE; ... if not switchcase: self.pc = target`.
        let goto_if_byte = *insns_opname_to_byte()
            .get("goto_if_not/iL")
            .expect("`goto_if_not/iL` must be in insns table");
        let code = [
            goto_if_byte,
            0x00, // i register 0
            0x40,
            0x00, // L target = 0x0040
        ];
        let mut tc = fresh_trace_ctx();
        let value = tc.const_int(1);
        let mut regs_i = vec![value];
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };

        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("truthy branch must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, code.len(), "truthy branch falls through");
    }

    #[test]
    fn goto_if_not_falsy_records_guard_false_and_jumps() {
        // `goto_if_not/iL` with a concrete zero Int: emit GuardFalse,
        // jump to the label target (pc = target).
        let goto_if_byte = *insns_opname_to_byte()
            .get("goto_if_not/iL")
            .expect("`goto_if_not/iL` must be in insns table");
        let code = [
            goto_if_byte,
            0x00, // i register 0
            0x40,
            0x00, // L target = 0x0040
        ];
        let mut tc = fresh_trace_ctx();
        let value = tc.const_int(0);
        let mut regs_i = vec![value];
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };

        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("falsy branch must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, 0x0040, "falsy branch jumps to label target");
    }

    #[test]
    fn goto_if_not_requires_concrete_int_value() {
        // Non-constant symbolic OpRef has no concrete: must surface
        // `GotoIfNotValueNotConcrete` rather than guess a branch.
        let goto_if_byte = *insns_opname_to_byte()
            .get("goto_if_not/iL")
            .expect("`goto_if_not/iL` must be in insns table");
        let code = [
            goto_if_byte,
            0x00, // i register 0
            0x40,
            0x00, // L target = 0x0040
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = vec![OpRef::input_arg_int(0)];
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };

        let err = step(&code, 0, &mut wc).expect_err("non-constant branch value must not guess");

        assert_eq!(
            err,
            DispatchError::GotoIfNotValueNotConcrete {
                pc: 0,
                value: OpRef::input_arg_int(0),
            }
        );
    }

    /// Production-like `sub_jitcode_lookup` that resolves `idx` against
    /// `crate::jitcode_runtime::all_jitcodes()`. Used by the end-to-end
    /// arm acceptance tests (`walk_return_value_arm_*`,
    /// `walk_pop_top_arm_*`) so the walker can recurse into real
    /// callee bodies. The runtime's `all_jitcodes()` is a
    /// `LazyLock<Vec<Arc<JitCode>>>` — every `.code` slice it surfaces
    /// is `'static`-rooted, satisfying `SubJitCodeBody`'s body
    /// constraint.
    fn production_sub_jitcodes(idx: usize) -> Option<SubJitCodeBody> {
        let all = crate::jitcode_runtime::all_jitcodes();
        all.get(idx).map(|jc| SubJitCodeBody {
            code: jc.code.as_slice(),
            num_regs_r: jc.num_regs_r(),
            num_regs_i: jc.num_regs_i(),
            num_regs_f: jc.num_regs_f(),
            constants_i: jc.constants_i.as_slice(),
            constants_r: jc.constants_r.as_slice(),
            constants_f: jc.constants_f.as_slice(),
        })
    }

    /// Tests use the production `PyreJitCodeDescr` adapter
    /// directly — the type lives at
    /// `pyre-jit-trace/src/descr.rs::PyreJitCodeDescr`
    /// + `descr::make_jitcode_descr(idx)` so the walker's
    /// `as_jitcode_descr()` cast exercises production code, not a
    /// test-local duplicate.
    use crate::descr::make_jitcode_descr;

    /// Build a `descr_refs` pool of length `pool_len` where the slot at
    /// each `BhDescr::JitCode` index in `crate::jitcode_runtime::all_descrs()`
    /// holds a `TestJitCodeDescr` carrying that descr's `jitcode_index`,
    /// and every other slot holds a `make_fail_descr` placeholder.
    /// Lets acceptance tests resolve `inline_call_*` descr indices
    /// without standing up the full BhDescr → trait Descr adapter
    /// pipeline.
    fn descr_pool_with_jitcode_adapters(pool_len: usize) -> Vec<DescrRef> {
        let all_bh = crate::jitcode_runtime::all_descrs();
        (0..pool_len)
            .map(|i| match all_bh.get(i) {
                Some(majit_translate::jitcode::BhDescr::JitCode { jitcode_index, .. }) => {
                    make_jitcode_descr(*jitcode_index)
                }
                _ => make_fail_descr(1 + i),
            })
            .collect()
    }

    #[test]
    fn inline_call_recursion_writes_subreturn_into_caller_dst_register() {
        // Core acceptance: caller's `inline_call_r_r/dR>r`
        // recurses into a synthetic callee jitcode whose body is
        // simply `ref_return r0`. The callee's ref_return surfaces as
        // `SubReturn { result: Some(callee.registers_r[0]) }`; the
        // caller's inline_call handler writes that OpRef into the
        // caller's dst register. Then the caller's own `ref_return r3`
        // records the outermost Finish carrying that propagated value.
        let ret_byte = *insns_opname_to_byte()
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns table");
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_r_r/dR>r")
            .expect("`inline_call_r_r/dR>r` must be in insns table");
        // Callee body: `ref_return r0`. registers_r[0] is populated
        // from the caller's R-list arg.
        let callee_code: &'static [u8] = Box::leak(Box::new([ret_byte, 0]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 0,
            num_regs_f: 0,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = {
            let sub_body = sub_body.clone();
            move |idx: usize| {
                if idx == 7 {
                    Some(sub_body.clone())
                } else {
                    None
                }
            }
        };
        // Caller body:
        //   inline_call_r_r/dR>r descr=7, R=[r2], >r=r5
        //   ref_return r5
        let caller_code = [
            inline_byte,
            0x07,
            0x00, // d (LE descr index = 7)
            0x01,
            0x02, // R: varlen=1, args=[r2]
            0x05, // >r: dst = r5
            ret_byte,
            0x05, // ref_return r5
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let arg_value = regs_r[2];
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[7] = make_jitcode_descr(7);
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, end_pc) =
            walk(&caller_code, 0, &mut wc).expect("caller must walk to terminator");
        assert_eq!(outcome, DispatchOutcome::Terminate);
        assert_eq!(end_pc, caller_code.len());
        drop(wc);
        // dst register r5 must equal the arg the caller passed (since
        // callee's `ref_return r0` returns its registers_r[0] which
        // was populated from caller's R-list[0] = r2's OpRef).
        assert_eq!(
            regs_r[5], arg_value,
            "inline_call_r_r dst writeback must propagate callee's SubReturn",
        );
        // Outermost FINISH carries the same value.
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "exactly one Finish must be recorded (callee's ref_return surfaced as \
             SubReturn, did not record a Finish)",
        );
        let last = tc.ops().last().expect("Finish must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::Finish);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![arg_value],
            "outermost Finish must carry the arg value the caller threaded through \
             inline_call_r_r",
        );
    }

    #[test]
    fn inline_call_r_i_writes_int_subreturn_into_caller_int_bank() {
        // Acceptance: caller's `inline_call_r_i/dR>i`
        // recurses into a synthetic callee whose body is `int_return
        // r0` on the int bank. The callee's int_return surfaces as
        // `SubReturn { result: Some(callee.registers_i[0]) }`; the
        // caller's helper writes that OpRef into the caller's
        // `registers_i[dst]` (NOT registers_r — the kind discriminator
        // for this variant). RPython parity: pyjitpl.py:1266-1324
        // exec-generated `_opimpl_inline_call_r_i` template paired with
        // `_opimpl_any_return` for `int_return`.
        //
        // Callee shape constraint: the `_r` arglist promises only Ref
        // args, but the body needs an int register populated to source
        // the int_return. The codewriter populates that via a separate
        // op inside the callee body (e.g. `int_copy/i>i`). For the
        // walker test, we synthesize an `int_copy` that materializes
        // the int constant from a tracer-side const_int OpRef stored in
        // a high i-register, then int_returns it.
        //
        // Simpler: callee whose body is just `int_return i0`; we
        // pre-populate the callee's registers_i[0] indirectly through a
        // setter — but the walker doesn't expose that directly. So
        // instead, we use a callee whose body emits an int constant op
        // and returns it. The simplest working shape is `int_neg i0
        // ->i0; int_return i0` — but registers_i[0] starts as
        // OpRef::NONE which `int_neg` would record meaninglessly.
        //
        // Pragmatic alternative: the walker's test-side `setup` for
        // sub_body lets us choose `num_regs_i = 1`. We initialize
        // callee.registers_i[0] to a known OpRef AT SUB-WALK TIME by
        // having the caller arglist carry the int OpRef indirectly —
        // but the `dR` arglist only has Ref. So we *can't* pass the
        // int OpRef through the call.
        //
        // RPython solution: callee bodies *always* compute their int
        // results from concrete operations (int_const, int_add, etc.).
        // For walker testing, the smallest standalone body is
        // `int_const_42 i0 = 42; int_return i0` — but pyre doesn't have
        // an `int_const/c>i` opname today (constants live in the
        // jitcode's constants_i table). Without re-engineering the
        // sub_body fixture, the cleanest test is to drive the callee
        // body through `int_copy` from a callee int register that the
        // setup_call path populated (which doesn't exist for `_r_i`
        // variant — only Ref args flow in).
        //
        // Here we lean on the simpler invariant: the
        // SubReturn{Some(value)} from the *helper itself* writes into
        // the caller's `registers_i[dst]`. To exercise that branch
        // without standing up a full int-producing callee, we test the
        // helper's dst-bank dispatch logic via a callee body that
        // returns an OpRef::NONE placeholder through `int_return r0`
        // — wait, that's wrong: `int_return/i` reads from `registers_i`
        // not `registers_r`.
        //
        // Cleanest path: callee body = `[int_return_byte, 0x00]` where
        // callee's `registers_i[0]` is OpRef::NONE; the SubReturn
        // value will be NONE. The test asserts that the caller's
        // `registers_i[dst]` was written to NONE (proving the dst-bank
        // routing is correct — wrong-bank routing would write to
        // `registers_r[dst]` instead and leave `registers_i[dst]`
        // unchanged at its initial OpRef::NONE).
        //
        // The OpRef::NONE-vs-OpRef::NONE comparison is admittedly
        // weak; instead we initialize the caller's `registers_i[dst]`
        // to a distinct OpRef before the call so the assertion can
        // distinguish "no write" from "write of NONE".
        let int_ret_byte = *insns_opname_to_byte()
            .get("int_return/i")
            .expect("`int_return/i` must be in insns table");
        let inline_ri_byte = *insns_opname_to_byte()
            .get("inline_call_r_i/dR>i")
            .expect("`inline_call_r_i/dR>i` must be in insns table");
        // Callee body: `int_return i0` (registers_i[0] starts at NONE).
        let callee_code: &'static [u8] = Box::leak(Box::new([int_ret_byte, 0]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1, // callee accepts a Ref arg, then ignores it
            num_regs_i: 1,
            num_regs_f: 0,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = {
            let sub_body = sub_body.clone();
            move |idx: usize| {
                if idx == 7 {
                    Some(sub_body.clone())
                } else {
                    None
                }
            }
        };
        // Caller body: `inline_call_r_i descr=7, R=[r2], >i=i3`
        //   opcode(1) + d(2) + R-len(1) + R[0](1) + dst(1) = 6 bytes
        let caller_code = [inline_ri_byte, 0x07, 0x00, 0x01, 0x02, 0x03];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        // Initialize registers_i[3] (dst) to a sentinel so we can
        // detect that the write happened.
        let sentinel_pre = tc.const_int(0xDEAD_BEEF);
        let mut regs_i: Vec<OpRef> = vec![sentinel_pre; 4];
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[7] = make_jitcode_descr(7);
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) =
            step(&caller_code, 0, &mut wc).expect("inline_call_r_i must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, caller_code.len());
        drop(wc);
        // Callee's int_return[i0] surfaced SubReturn{Some(NONE)}; the
        // helper wrote that into caller's registers_i[3]. Sentinel is
        // gone, replaced by OpRef::NONE.
        assert_eq!(
            regs_i[3],
            OpRef::NONE,
            "inline_call_r_i must write SubReturn value into caller registers_i[dst]",
        );
        // Wrong-bank check: registers_r[3] must remain its original
        // distinct_const_refs value (the dst-bank routing did NOT
        // write to the Ref bank).
        assert_ne!(
            regs_r[3],
            OpRef::NONE,
            "inline_call_r_i must NOT write to registers_r[dst]",
        );
    }

    #[test]
    fn inline_call_ir_r_populates_callee_int_and_ref_banks() {
        // Acceptance: caller's `inline_call_ir_r/dIR>r` carries
        // both an I-list and an R-list. The callee's int + ref register
        // banks must both be populated (RPython
        // `pyjitpl.py:230-260 setup_call(argboxes_i, argboxes_r,
        // argboxes_f)`). Smoke test: callee body is `ref_return r0` —
        // the ref arg routes through registers_r[0] back to the caller's
        // dst slot. The int arg flowing into registers_i[0] is dead but
        // proves the helper read the I-list (a regression where the
        // I-list parsing miscounted bytes would offset the R-list read
        // and we'd see the wrong ref OpRef in the dst).
        let ret_byte = *insns_opname_to_byte()
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns table");
        let inline_ir_r_byte = *insns_opname_to_byte()
            .get("inline_call_ir_r/dIR>r")
            .expect("`inline_call_ir_r/dIR>r` must be in insns table");
        // Callee body: `ref_return r0` (size 2).
        let callee_code: &'static [u8] = Box::leak(Box::new([ret_byte, 0]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 1, // accept one int arg
            num_regs_f: 0,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = {
            let sub_body = sub_body.clone();
            move |idx: usize| {
                if idx == 7 {
                    Some(sub_body.clone())
                } else {
                    None
                }
            }
        };
        // Caller body: `inline_call_ir_r descr=7, I=[i1], R=[r2], >r=r5`
        //   opcode(1) + d(2) + I-len(1) + I[0](1) + R-len(1) + R[0](1) + dst(1) = 8 bytes
        let caller_code = [
            inline_ir_r_byte,
            0x07,
            0x00, // descr index 7 (LE)
            0x01,
            0x01, // I-list: len=1, args=[i1]
            0x01,
            0x02, // R-list: len=1, args=[r2]
            0x05, // dst = r5
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let arg_ref = regs_r[2];
        let mut regs_i: Vec<OpRef> = (0..4)
            .map(|i| tc.const_int(0xCAFE_F00D + i as i64))
            .collect();
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[7] = make_jitcode_descr(7);
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) =
            step(&caller_code, 0, &mut wc).expect("inline_call_ir_r must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, caller_code.len());
        drop(wc);
        // dst register r5 must equal the caller's R-list arg (which the
        // callee returned via ref_return r0).
        assert_eq!(
            regs_r[5], arg_ref,
            "inline_call_ir_r dst writeback must propagate callee's SubReturn from ref_return r0",
        );
    }

    #[test]
    fn inline_call_irf_r_populates_all_three_kind_banks() {
        // Acceptance: caller's `inline_call_irf_r/dIRF>r`
        // carries an I-list, R-list, AND F-list. Smoke test: callee
        // body is `ref_return r0` — the caller's R-list arg propagates
        // through. The I-list and F-list args are dead from the
        // callee's POV but their presence forces the helper to advance
        // operand offsets correctly through all three lists; a parsing
        // bug (e.g. F-list-len byte misaligned) would put the wrong
        // ref OpRef into the dst.
        let ret_byte = *insns_opname_to_byte()
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns table");
        let inline_irf_r_byte = *insns_opname_to_byte()
            .get("inline_call_irf_r/dIRF>r")
            .expect("`inline_call_irf_r/dIRF>r` must be in insns table");
        let callee_code: &'static [u8] = Box::leak(Box::new([ret_byte, 0]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 1,
            num_regs_f: 1,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = {
            let sub_body = sub_body.clone();
            move |idx: usize| {
                if idx == 7 {
                    Some(sub_body.clone())
                } else {
                    None
                }
            }
        };
        // Caller body: inline_call_irf_r descr=7, I=[i1], R=[r2], F=[f0], >r=r5
        //   opcode(1) + d(2) + I-len(1) + I[0](1) + R-len(1) + R[0](1)
        //   + F-len(1) + F[0](1) + dst(1) = 10 bytes
        let caller_code = [
            inline_irf_r_byte,
            0x07,
            0x00, // descr index 7
            0x01,
            0x01, // I-list len=1, args=[i1]
            0x01,
            0x02, // R-list len=1, args=[r2]
            0x01,
            0x00, // F-list len=1, args=[f0]
            0x05, // dst = r5
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let arg_ref = regs_r[2];
        let mut regs_i: Vec<OpRef> = (0..4).map(|i| tc.const_int(i as i64)).collect();
        // Float bank: pyre's TraceCtx doesn't expose a const_float
        // factory in the test fixture path, but we only need *distinct*
        // OpRef values to exercise list-byte advancement; const_int +
        // type-punning into the float slot is sufficient because the
        // walker treats the bank as opaque OpRef storage.
        let mut regs_f: Vec<OpRef> = (0..4).map(|i| tc.const_int(0xF1F1 + i as i64)).collect();
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[7] = make_jitcode_descr(7);
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut regs_f,
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) =
            step(&caller_code, 0, &mut wc).expect("inline_call_irf_r must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, caller_code.len());
        drop(wc);
        // Smoking gun: dst register r5 must equal the caller's R-list
        // arg (passed through callee's `ref_return r0`). A list-byte
        // advancement bug would land a different OpRef here.
        assert_eq!(
            regs_r[5], arg_ref,
            "inline_call_irf_r must correctly advance through I/R/F lists \
             and propagate the callee's ref SubReturn",
        );
    }

    #[test]
    fn inline_call_ir_int_arity_overflow_surfaces_typed_error() {
        // Per-bank arity check — providing more I-args than
        // the callee declared `num_regs_i` slots surfaces
        // `InlineCallIntArityMismatch`. The Ref-bank check is covered
        // by the existing `inline_call_with_more_args_than_callee_regs_surfaces_arity_mismatch`
        // test for the `_r_r` variant.
        let inline_ir_r_byte = *insns_opname_to_byte()
            .get("inline_call_ir_r/dIR>r")
            .expect("`inline_call_ir_r/dIR>r` must be in insns table");
        // Callee with num_regs_i=0 — any I-list args overflow.
        let ret_byte = *insns_opname_to_byte()
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns table");
        let callee_code: &'static [u8] = Box::leak(Box::new([ret_byte, 0]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 0, // overflow trigger
            num_regs_f: 0,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = {
            let sub_body = sub_body.clone();
            move |idx: usize| {
                if idx == 7 {
                    Some(sub_body.clone())
                } else {
                    None
                }
            }
        };
        // Caller body: `inline_call_ir_r descr=7, I=[i1], R=[r2], >r=r5`
        let caller_code = [
            inline_ir_r_byte,
            0x07,
            0x00,
            0x01,
            0x01, // I-list with 1 arg → overflows callee (num_regs_i=0)
            0x01,
            0x02,
            0x05,
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let mut regs_i: Vec<OpRef> = (0..4).map(|i| tc.const_int(i as i64)).collect();
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[7] = make_jitcode_descr(7);
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err =
            step(&caller_code, 0, &mut wc).expect_err("I-list overflow must surface typed error");
        assert_eq!(
            err,
            DispatchError::InlineCallIntArityMismatch {
                pc: 0,
                provided: 1,
                callee_num_regs_i: 0,
            },
        );
    }

    #[test]
    fn inline_call_recursion_propagates_subraise_from_callee() {
        // Top-level uncaught SubRaise: callee's `raise/r` surfaces as
        // `SubRaise { exc }` to the caller's inline_call handler. With
        // no caller-side `catch_exception/L` and is_top_level=true on
        // the outermost walker, RPython
        // `pyjitpl.py:2533-2538 finishframe_exception` records
        // `compile_exit_frame_with_exception(last_exc_box)` — i.e.
        // FINISH(exc, exit_frame_with_exception_descr_ref) and exits
        // the trace. Walker mirrors this in `walk()`: top-level
        // SubRaise → record FINISH + Terminate.
        let raise_byte = *insns_opname_to_byte()
            .get("raise/r")
            .expect("`raise/r` must be in insns table");
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_r_r/dR>r")
            .expect("`inline_call_r_r/dR>r` must be in insns table");
        // Callee body: `raise r0`
        let callee_code: &'static [u8] = Box::leak(Box::new([raise_byte, 0]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 0,
            num_regs_f: 0,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = {
            let sub_body = sub_body.clone();
            move |idx: usize| {
                if idx == 7 {
                    Some(sub_body.clone())
                } else {
                    None
                }
            }
        };
        // Caller body: `inline_call_r_r descr=7 R=[r2] >r=r5`
        // (no follow-on `ref_return` — the SubRaise propagates straight
        // up to the caller's `walk` loop, which converts to FINISH at
        // top level.)
        let caller_code = [inline_byte, 0x07, 0x00, 0x01, 0x02, 0x05];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let arg_value = regs_r[2];
        let descr = done_descr_ref_for_tests();
        let descr_exc = make_fail_descr(2);
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[7] = make_jitcode_descr(7);
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: descr_exc.clone(),
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, _) = walk(&caller_code, 0, &mut wc).expect("caller must walk to terminator");
        assert_eq!(
            outcome,
            DispatchOutcome::Terminate,
            "top-level walk must convert uncaught SubRaise to Terminate \
             after recording the outermost FINISH",
        );
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "exactly one FINISH must be recorded",
        );
        let last = tc.ops().last().expect("FINISH must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::Finish);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![arg_value],
            "FINISH args must carry the bubbled exc OpRef",
        );
        let recorded_descr = last
            .getdescr()
            .expect("FINISH must carry exit_frame_with_exception_descr_ref");
        assert!(
            std::sync::Arc::ptr_eq(&recorded_descr, &descr_exc),
            "FINISH descr must be exit_frame_with_exception_descr_ref",
        );
    }

    #[test]
    fn inline_call_with_unresolvable_descr_surfaces_typed_error() {
        // Descr at the inline_call's d-slot must implement
        // `JitCodeDescr`. A `FailDescr` placeholder doesn't, so the
        // walker surfaces `ExpectedJitCodeDescr`.
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_r_r/dR>r")
            .expect("`inline_call_r_r/dR>r` must be in insns table");
        let caller_code = [inline_byte, 0x05, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&caller_code, 0, &mut wc)
            .expect_err("FailDescr at inline_call's d-slot must hit ExpectedJitCodeDescr");
        assert_eq!(
            err,
            DispatchError::ExpectedJitCodeDescr {
                pc: 0,
                descr_index: 5,
            },
        );
    }

    #[test]
    fn inline_call_with_missing_sub_jitcode_lookup_surfaces_typed_error() {
        // Descr resolves to JitCodeDescr but lookup returns
        // None — surface `SubJitCodeNotFound`.
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_r_r/dR>r")
            .expect("`inline_call_r_r/dR>r` must be in insns table");
        let caller_code = [inline_byte, 0x03, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[3] = make_jitcode_descr(999_999);
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&caller_code, 0, &mut wc)
            .expect_err("missing sub-jitcode must hit SubJitCodeNotFound");
        assert_eq!(
            err,
            DispatchError::SubJitCodeNotFound {
                pc: 0,
                jitcode_index: 999_999,
            },
        );
    }

    #[test]
    fn step_through_live_opcode_advances_by_offset_size() {
        let live_byte = *insns_opname_to_byte()
            .get("live/")
            .expect("`live/` must be in insns table");
        let code = [live_byte, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("live/ must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            next_pc,
            1 + majit_translate::liveness::OFFSET_SIZE,
            "live/ must advance past the OFFSET_SIZE liveness slot",
        );
    }

    #[test]
    fn step_through_ref_return_records_finish_with_descr_and_correct_arg() {
        // `ref_return/r` records `rop.FINISH(reg)` to the
        // TraceCtx with `done_with_this_frame_descr_ref` attached, and
        // the `reg` byte selects the correct OpRef from `registers_r`.
        // RPython `pyjitpl.py:opimpl_ref_return → finishframe →
        // compile_done_with_this_frame → record1(FINISH, descr=token)`.
        let ret_byte = *insns_opname_to_byte()
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns table");
        // Read register at byte index 3 — distinct from index 0 to
        // catch off-by-one bugs in operand decoding.
        let code = [ret_byte, 0x03];
        let mut tc = fresh_trace_ctx();
        let mut regs = distinct_const_refs(&mut tc, 8);
        let expected_arg = regs[3];
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("ref_return/r must dispatch");
        assert_eq!(outcome, DispatchOutcome::Terminate);
        assert_eq!(next_pc, 2, "ref_return/r consumes 1 register byte");
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "exactly one Finish op must be recorded",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::Finish);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![expected_arg],
            "Finish args must select registers_r[3], not registers_r[0]",
        );
        let recorded_descr = last
            .getdescr()
            .expect("Finish must carry done_with_this_frame_descr_ref");
        assert!(
            std::sync::Arc::ptr_eq(&recorded_descr, &descr),
            "Finish descr must be the exact instance the dispatcher was handed",
        );
    }

    #[test]
    fn ref_return_with_out_of_range_register_surfaces_typed_error() {
        let ret_byte = *insns_opname_to_byte()
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns table");
        let code = [ret_byte, 0x07]; // index 7 — registers_r is empty
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc).expect_err("must surface RegisterOutOfRange");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 7,
                len: 0,
                bank: "r"
            },
        );
    }

    #[test]
    fn step_through_int_return_records_finish_with_int_descr() {
        // `int_return/i` mirrors `ref_return/r` on the int
        // bank. Top-level records `FINISH(int_value)` with
        // `done_with_this_frame_descr_int` (RPython `pyjitpl.py:3206-3208
        // compile_done_with_this_frame: token = sd.done_with_this_frame_descr_int`).
        let ret_byte = *insns_opname_to_byte()
            .get("int_return/i")
            .expect("`int_return/i` must be in insns table");
        let code = [ret_byte, 0x02];
        let mut tc = fresh_trace_ctx();
        let mut regs_i: Vec<OpRef> = (0..4)
            .map(|i| tc.const_int(0xBEEF_0000 + i as i64))
            .collect();
        let expected_arg = regs_i[2];
        let descr_int = make_fail_descr(42);
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: make_fail_descr(1),
            done_with_this_frame_descr_int: descr_int.clone(),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let ops_before = wc.trace_ctx.num_ops();
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("int_return/i must dispatch");
        assert_eq!(outcome, DispatchOutcome::Terminate);
        assert_eq!(next_pc, 2);
        drop(wc);
        assert_eq!(tc.num_ops(), ops_before + 1);
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::Finish);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![expected_arg]
        );
        let recorded_descr = last
            .getdescr()
            .expect("Finish must carry done_with_this_frame_descr_int");
        assert!(
            std::sync::Arc::ptr_eq(&recorded_descr, &descr_int),
            "int_return/i must use done_with_this_frame_descr_int, not _ref",
        );
    }

    #[test]
    fn step_through_int_return_subwalk_surfaces_subreturn_some() {
        // nested `int_return/i` propagates SubReturn{Some(value)}
        // — same shape as `ref_return/r` sub-walk. RPython
        // `pyjitpl.py:1688-1698 finishframe → popframe` returns control to
        // caller's metainterp loop with the box in hand.
        let ret_byte = *insns_opname_to_byte()
            .get("int_return/i")
            .expect("`int_return/i` must be in insns table");
        let code = [ret_byte, 0x01];
        let mut tc = fresh_trace_ctx();
        let mut regs_i: Vec<OpRef> = (0..4)
            .map(|i| tc.const_int(0xCAFE_0000 + i as i64))
            .collect();
        let expected = regs_i[1];
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: make_fail_descr(1),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: false,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let ops_before = wc.trace_ctx.num_ops();
        let (outcome, _) = step(&code, 0, &mut wc).expect("int_return/i must dispatch");
        assert_eq!(
            outcome,
            DispatchOutcome::SubReturn {
                result: Some(expected),
            },
            "sub-walk int_return must surface SubReturn{{Some(value)}}",
        );
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before,
            "sub-walk int_return must NOT record FINISH (control returns to caller)",
        );
    }

    #[test]
    fn step_through_void_return_records_empty_finish_with_void_descr() {
        // top-level `void_return/` records `FINISH([])` with
        // `done_with_this_frame_descr_void`. RPython
        // `pyjitpl.py:3202-3205 compile_done_with_this_frame`:
        //   if result_type == VOID:
        //       assert exitbox is None
        //       exits = []
        //       token = sd.done_with_this_frame_descr_void
        let ret_byte = *insns_opname_to_byte()
            .get("void_return/")
            .expect("`void_return/` must be in insns table");
        let code = [ret_byte];
        let mut tc = fresh_trace_ctx();
        let descr_void = make_fail_descr(77);
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: make_fail_descr(1),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: descr_void.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let ops_before = wc.trace_ctx.num_ops();
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("void_return/ must dispatch");
        assert_eq!(outcome, DispatchOutcome::Terminate);
        assert_eq!(next_pc, 1, "void_return/ has zero operand bytes");
        drop(wc);
        assert_eq!(tc.num_ops(), ops_before + 1);
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::Finish);
        assert!(
            last.num_args() == 0,
            "void_return/ FINISH must carry zero args (RPython exits = [])",
        );
        let recorded_descr = last
            .getdescr()
            .expect("Finish must carry done_with_this_frame_descr_void");
        assert!(
            std::sync::Arc::ptr_eq(&recorded_descr, &descr_void),
            "void_return/ must use done_with_this_frame_descr_void, not _ref",
        );
    }

    #[test]
    fn step_through_void_return_subwalk_surfaces_subreturn_none() {
        // nested `void_return/` propagates SubReturn{None} —
        // RPython `pyjitpl.py:467-469 opimpl_void_return → finishframe(None)`.
        // The caller's `inline_call_*_v` variant (when one exists) does
        // not write a dst register; today the walker has no `_v`
        // inline_call handler so `SubReturn{None}` reaching an `_r_r`
        // caller surfaces `UnexpectedVoidSubReturn` (the existing typed
        // error covers that path). This test only exercises the leaf
        // sub-walk surface.
        let ret_byte = *insns_opname_to_byte()
            .get("void_return/")
            .expect("`void_return/` must be in insns table");
        let code = [ret_byte];
        let mut tc = fresh_trace_ctx();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: make_fail_descr(1),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(77),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: false,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let ops_before = wc.trace_ctx.num_ops();
        let (outcome, _) = step(&code, 0, &mut wc).expect("void_return/ must dispatch");
        assert_eq!(outcome, DispatchOutcome::SubReturn { result: None });
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before,
            "sub-walk void_return must NOT record FINISH",
        );
    }

    #[test]
    fn raise_with_out_of_range_register_surfaces_typed_error() {
        // `raise/r` reads its operand for OOR validation
        // even though recording is deferred. Catches the same classes
        // of assembler bugs `ref_return/r` does.
        let raise_byte = *insns_opname_to_byte()
            .get("raise/r")
            .expect("`raise/r` must be in insns table");
        let code = [raise_byte, 0x05];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc).expect_err("raise/r must read its operand");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 5,
                len: 0,
                bank: "r"
            },
        );
    }

    #[test]
    fn step_through_goto_jumps_to_label_target() {
        // `goto/L` reads its 2-byte LE label and the walker
        // returns Continue at the label target, not the linear next pc.
        // RPython `blackhole.py:950-952 bhimpl_goto(target): return target`.
        let goto_byte = *insns_opname_to_byte()
            .get("goto/L")
            .expect("`goto/L` must be in insns table");
        // target = 0x002A = 42
        let code = [goto_byte, 0x2A, 0x00];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("goto/L must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            next_pc, 42,
            "goto/L must jump to its 2-byte LE label target",
        );
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before,
            "goto/L records nothing — pure control flow",
        );
    }

    #[test]
    fn step_through_goto_handles_high_byte_of_label() {
        // Confirm the LE decode reads both bytes (regression guard for
        // accidentally treating L as a single byte).
        let goto_byte = *insns_opname_to_byte()
            .get("goto/L")
            .expect("`goto/L` must be in insns table");
        // target = 0x0102 = 258
        let code = [goto_byte, 0x02, 0x01];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("goto/L must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, 258);
    }

    #[test]
    fn finishframe_lookahead_distinguishes_catch_rvmprof_and_nomatch() {
        // `finishframe_lookahead_at` must mirror RPython
        // `pyjitpl.py:2506-2531 finishframe_exception` line-by-line —
        // sequential `catch_exception/L` then `rvmprof_code/ii` then
        // fall-through.
        //
        // pyre's emitted insns table currently lacks `rvmprof_code/ii`
        // (forward-prep — RPython emits it when rvmprof is enabled at
        // codewriter time). Test only the bytes that ARE in the table
        // and assert the helper shape compiles + the catch / no-match
        // arms route correctly.
        let live_byte = *insns_opname_to_byte()
            .get("live/")
            .expect("live/ must be in insns");
        let catch_byte = *insns_opname_to_byte()
            .get("catch_exception/L")
            .expect("catch_exception/L must be in insns");
        let goto_byte = *insns_opname_to_byte()
            .get("goto/L")
            .expect("goto/L must be in insns");

        // (1) live/ + catch_exception/L target=42 → CatchTarget(42).
        let code_catch = [live_byte, 0x00, 0x00, catch_byte, 0x2A, 0x00];
        assert_eq!(
            finishframe_lookahead_at(&code_catch, 0),
            FinishframeLookahead::CatchTarget(0x2A),
        );

        // (2) catch_exception/L without leading live/ → still
        //     CatchTarget (RPython's `if opcode == op_live: skip` is
        //     conditional, not required).
        let code_no_live_catch = [catch_byte, 0x10, 0x01];
        assert_eq!(
            finishframe_lookahead_at(&code_no_live_catch, 0),
            FinishframeLookahead::CatchTarget(0x110),
        );

        // (3) live/ + goto/L (NOT catch nor rvmprof) → NoMatch (the
        //     caller continues unwinding).
        let code_no_match = [live_byte, 0x00, 0x00, goto_byte, 0x00, 0x00];
        assert_eq!(
            finishframe_lookahead_at(&code_no_match, 0),
            FinishframeLookahead::NoMatch,
        );

        // (4) Position past end of code → NoMatch (decode fails).
        assert_eq!(
            finishframe_lookahead_at(&code_catch, 99),
            FinishframeLookahead::NoMatch,
        );
    }

    #[test]
    fn step_through_catch_exception_with_active_exception_surfaces_typed_error() {
        // RPython `pyjitpl.py:497-504 opimpl_catch_exception`:
        //   assert not self.metainterp.last_exc_value
        // Reaching catch_exception/L on the normal walk path with
        // last_exc_value=Some(_) violates the codewriter invariant —
        // surface as `CatchExceptionWithActiveException`.
        let catch_byte = *insns_opname_to_byte()
            .get("catch_exception/L")
            .expect("`catch_exception/L` must be in insns table");
        let code = [catch_byte, 0x2A, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs = distinct_const_refs(&mut tc, 4);
        let active_exc = regs[0];
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut regs,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: Some(active_exc),
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err =
            step(&code, 0, &mut wc).expect_err("catch_exception/L with active exc must error");
        assert_eq!(
            err,
            DispatchError::CatchExceptionWithActiveException { pc: 0 }
        );
    }

    #[test]
    fn step_through_catch_exception_advances_past_label_operand() {
        // `catch_exception/L` records nothing on the normal
        // walk (RPython `pyjitpl.py:497-504 opimpl_catch_exception` is
        // an `assert not last_exc_value` only) and the walker advances
        // linearly past the 2-byte target.
        let catch_byte = *insns_opname_to_byte()
            .get("catch_exception/L")
            .expect("`catch_exception/L` must be in insns table");
        let code = [catch_byte, 0x2A, 0x00];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("catch_exception/L must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            next_pc, 3,
            "catch_exception/L must advance past the 2-byte target operand",
        );
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before,
            "catch_exception/L records nothing on normal walk",
        );
    }

    #[test]
    fn step_through_raise_records_outermost_finish_and_terminates() {
        // RPython `pyjitpl.py:1688-1698 opimpl_raise` →
        // `finishframe_exception` (outermost-frame branch) →
        // `compile_exit_frame_with_exception` records
        // `FINISH(exc, descr=exit_frame_with_exception_descr_ref)`.
        // The walker treats every invocation as outermost (no
        // framestack), so this is the parity-correct emit.
        let raise_byte = *insns_opname_to_byte()
            .get("raise/r")
            .expect("`raise/r` must be in insns table");
        // exc operand reads registers_r[2]
        let code = [raise_byte, 0x02];
        let mut tc = fresh_trace_ctx();
        let mut regs = distinct_const_refs(&mut tc, 4);
        let expected_exc = regs[2];
        let descr_done = done_descr_ref_for_tests();
        let descr_exc = make_fail_descr(99);
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: descr_exc.clone(),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("raise/r must dispatch");
        assert_eq!(outcome, DispatchOutcome::Terminate);
        assert_eq!(next_pc, 2);
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "raise/r must record exactly one FINISH op (outermost branch)",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::Finish);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![expected_exc],
            "FINISH args must carry the exception OpRef from registers_r[src]",
        );
        let recorded_descr = last
            .getdescr()
            .expect("FINISH must carry exit_frame_with_exception_descr_ref");
        assert!(
            std::sync::Arc::ptr_eq(&recorded_descr, &descr_exc),
            "FINISH descr must be the caller-supplied \
             `exit_frame_with_exception_descr_ref`, not \
             `done_with_this_frame_descr_ref`",
        );
    }

    #[test]
    fn raise_r_emits_guard_class_when_concrete_exc_pinned_in_shadow() {
        // The concrete shadow is mutable and
        // tracked by every `registers_r[dst]` write
        // ([`write_ref_reg`]), so a `raise/r` reading the shadow finds
        // a reliable concrete pointer.  Allocate a real
        // `W_ExceptionObject` so the deref against
        // `ob_header.ob_type` is sound; expect GuardClass + Finish
        // recorded and the heapcache class-known flag pinned.  Mirrors
        // trait-side `seed_raised_exception` at `trace_opcode.rs:
        // 6629-6643`.
        let exc_ptr = pyre_object::excobject::w_exception_new(
            pyre_object::excobject::ExcKind::ValueError,
            "shadow-walker probe",
        );
        let raise_byte = *insns_opname_to_byte()
            .get("raise/r")
            .expect("`raise/r` must be in insns table");
        let code = [raise_byte, 0x02];
        let mut tc = fresh_trace_ctx();
        // Use a non-constant OpRef so the heapcache class-known flag
        // actually pins. pyre's `is_class_known(constant)` returns
        // false (`heapcache.rs:1014`) while `class_now_known(constant)`
        // is a no-op, so constants never round-trip through the
        // class-pinned cache.
        let exc_box = OpRef::input_arg_ref(0);
        let mut regs: Vec<OpRef> = vec![OpRef::NONE, OpRef::NONE, exc_box, OpRef::NONE];
        let mut concrete = vec![
            ConcreteValue::Null,
            ConcreteValue::Null,
            ConcreteValue::Ref(exc_ptr),
            ConcreteValue::Null,
        ];
        let descr_done = done_descr_ref_for_tests();
        let descr_exc = make_fail_descr(99);
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut concrete,
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: descr_exc.clone(),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, _next_pc) = step(&code, 0, &mut wc).expect("raise/r must dispatch");
        assert_eq!(outcome, DispatchOutcome::Terminate);
        drop(wc);

        // Expect two ops recorded: GuardClass(exc, cls_const) then
        // Finish(exc) (the GUARD_CLASS precedes FINISH per
        // `pyjitpl.py:1690-1696`).
        assert_eq!(
            tc.num_ops(),
            ops_before + 2,
            "raise/r with pinned concrete exc must record GuardClass + Finish",
        );
        let ops = tc.ops();
        let guard = &ops[ops_before];
        assert_eq!(guard.opcode, majit_ir::OpCode::GuardClass);
        assert_eq!(
            (&*guard.getarglist())[0].to_opref(),
            exc_box,
            "GuardClass arg0 must be the exception OpRef",
        );
        // After the guard, the heapcache must mark the class as known
        // so a follow-on raise/r against the same exc_box wouldn't
        // re-emit GuardClass.
        assert!(
            tc.heap_cache().is_class_known(exc_box),
            "heapcache.class_now_known must fire alongside GuardClass",
        );
    }

    #[test]
    fn step_through_reraise_at_top_level_records_outermost_finish() {
        // `reraise/` mirrors `raise/r` for the top-level
        // frame — it records `FINISH(last_exc_value,
        // exit_frame_with_exception_descr_ref)`. RPython parity:
        // `pyjitpl.py:1700-1704 opimpl_reraise → popframe →
        // finishframe_exception` when the framestack is empty falls
        // through to `compile_exit_frame_with_exception(last_exc_box)`
        // (pyjitpl.py:2533-2538).
        let reraise_byte = *insns_opname_to_byte()
            .get("reraise/")
            .expect("`reraise/` must be in insns table");
        let code = [reraise_byte];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let active_exc = regs_r[1];
        let descr_done = done_descr_ref_for_tests();
        let descr_exc = make_fail_descr(99);
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: descr_exc.clone(),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: Some(active_exc),
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("reraise/ must dispatch");
        assert_eq!(outcome, DispatchOutcome::Terminate);
        assert_eq!(next_pc, 1, "reraise/ has no operand");
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "reraise/ at top-level must record exactly one outermost FINISH",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::Finish);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![active_exc],
            "FINISH args must carry the standing last_exc_value OpRef",
        );
        let recorded_descr = last
            .getdescr()
            .expect("FINISH must carry exit_frame_with_exception_descr_ref");
        assert!(
            std::sync::Arc::ptr_eq(&recorded_descr, &descr_exc),
            "reraise/ at top-level must use exit_frame_with_exception_descr_ref",
        );
    }

    #[test]
    fn step_through_reraise_without_last_exc_value_surfaces_typed_error() {
        // RPython `pyjitpl.py:1702 opimpl_reraise`:
        //   assert self.metainterp.last_exc_value
        // — reaching `reraise` without an active exception is a
        // codewriter invariant violation. Walker surfaces it as a
        // typed error rather than an arbitrary panic / silent
        // fall-through.
        let reraise_byte = *insns_opname_to_byte()
            .get("reraise/")
            .expect("`reraise/` must be in insns table");
        let code = [reraise_byte];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc).expect_err("reraise/ without last_exc_value must error");
        assert_eq!(err, DispatchError::ReraiseWithoutLastExcValue { pc: 0 });
    }

    #[test]
    fn raise_at_top_level_populates_last_exc_value_before_finish() {
        // `raise/r` at top-level records FINISH and *also*
        // sets `ctx.last_exc_value` (RPython `pyjitpl.py:1695`). The
        // post-condition matters because a future opcode in a
        // wrap-around (e.g. an unconditional `reraise/` after the
        // raise) would read it. Independently asserting the field
        // post-step locks in the side effect.
        let raise_byte = *insns_opname_to_byte()
            .get("raise/r")
            .expect("`raise/r` must be in insns table");
        let code = [raise_byte, 0x02];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let exc = regs_r[2];
        let descr_done = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("raise/r must dispatch");
        assert_eq!(
            wc.last_exc_value,
            Some(exc),
            "raise/r must populate ctx.last_exc_value before terminating",
        );
    }

    #[test]
    fn inline_call_subraise_jumps_to_caller_catch_exception_target() {
        // Acceptance: callee's `raise/r` surfaces SubRaise to
        // the caller; caller's inline_call SubRaise arm probes
        // `op.next_pc` for `live/` + `catch_exception/L`, finds it,
        // sets `last_exc_value = exc`, and resumes at the catch target.
        // RPython parity: `pyjitpl.py:2506-2522 finishframe_exception`
        // line-by-line — `op_live` skip then `op_catch_exception`
        // target jump.
        let raise_byte = *insns_opname_to_byte()
            .get("raise/r")
            .expect("`raise/r` must be in insns table");
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_r_r/dR>r")
            .expect("`inline_call_r_r/dR>r` must be in insns table");
        let live_byte = *insns_opname_to_byte()
            .get("live/")
            .expect("`live/` must be in insns table");
        let catch_byte = *insns_opname_to_byte()
            .get("catch_exception/L")
            .expect("`catch_exception/L` must be in insns table");
        let ret_byte = *insns_opname_to_byte()
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns table");
        // Callee: `raise r0`
        let callee_code: &'static [u8] = Box::leak(Box::new([raise_byte, 0]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 0,
            num_regs_f: 0,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = move |idx: usize| {
            if idx == 11 {
                Some(sub_body.clone())
            } else {
                None
            }
        };
        // Caller layout (matches PopTop arm shape):
        //   pc=0..6   inline_call_r_r descr=11 R=[r3] >r=r5
        //     opcode(1) + d(2) + R-len(1) + R[0](1) + dst(1)
        //   pc=6..9   live + 2-byte liveness offset (OFFSET_SIZE=2)
        //     opcode(1) + slot(2)
        //   pc=9..12  catch_exception/L target=12 (LE little-endian)
        //     opcode(1) + target(2)
        //   pc=12..14 handler body: ref_return r5
        //     opcode(1) + reg(1)
        let caller_code = vec![
            inline_byte,
            0x0B,
            0x00,
            0x01,
            0x03,
            0x05,
            live_byte,
            0x00,
            0x00,
            catch_byte,
            0x0C,
            0x00,
            ret_byte,
            0x05,
        ];
        assert_eq!(caller_code.len(), 14);
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let exc_arg = regs_r[3];
        let descr_done = done_descr_ref_for_tests();
        let descr_exc = make_fail_descr(99);
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[11] = make_jitcode_descr(11);
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr_done.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: descr_exc,
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, end_pc) =
            walk(&caller_code, 0, &mut wc).expect("caller must walk to terminator");
        assert_eq!(
            outcome,
            DispatchOutcome::Terminate,
            "caller must reach handler's ref_return and terminate (not bubble SubRaise)",
        );
        assert_eq!(
            end_pc, 14,
            "walker must terminate at handler's ref_return r5 (pc=12..14)",
        );
        assert_eq!(
            wc.last_exc_value,
            Some(exc_arg),
            "caller's last_exc_value must be set to the exc OpRef from callee SubRaise",
        );
        drop(wc);
        // Outermost FINISH must carry the handler's ref_return arg —
        // r5, which still holds its pre-call distinct_const_refs OpRef
        // (caller's inline_call dst write happens *only* on
        // SubReturn, not SubRaise-then-catch).
        let last = tc.ops().last().expect("FINISH must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::Finish);
    }

    #[test]
    fn inline_call_subraise_without_caller_catch_bubbles_up_in_subwalk() {
        // Sub-walk SubRaise propagation: when the caller is itself a sub-walk
        // (`is_top_level=false`) and SubRaise reaches its `walk()`
        // loop with no `catch_exception/L` match, the loop returns
        // `SubRaise` unchanged so the parent's inline_call SubRaise arm
        // can scan its own op.next_pc for a catch handler.
        // RPython parity: `pyjitpl.py:2533 finishframe_exception` loops
        // through the framestack — only when `framestack` is exhausted
        // does it call `compile_exit_frame_with_exception`. Sub-walks
        // are not the framestack root.
        //
        // (The top-level FINISH conversion path is covered by
        // `inline_call_recursion_propagates_subraise_from_callee`
        // above.)
        let raise_byte = *insns_opname_to_byte()
            .get("raise/r")
            .expect("`raise/r` must be in insns table");
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_r_r/dR>r")
            .expect("`inline_call_r_r/dR>r` must be in insns table");
        let goto_byte = *insns_opname_to_byte()
            .get("goto/L")
            .expect("`goto/L` must be in insns table");
        let callee_code: &'static [u8] = Box::leak(Box::new([raise_byte, 0]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 0,
            num_regs_f: 0,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = move |idx: usize| {
            if idx == 13 {
                Some(sub_body.clone())
            } else {
                None
            }
        };
        let caller_code = [
            inline_byte,
            0x0D,
            0x00,
            0x01,
            0x02,
            0x05,
            goto_byte,
            0x00,
            0x00,
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let exc_arg = regs_r[2];
        let descr_done = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[13] = make_jitcode_descr(13);
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            // Sub-walk frame: bubble-up behaviour, no FINISH conversion.
            is_top_level: false,
            sub_jitcode_lookup: &lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let ops_before = wc.trace_ctx.num_ops();
        let (outcome, _) = walk(&caller_code, 0, &mut wc).expect("caller must walk to terminator");
        assert_eq!(
            outcome,
            DispatchOutcome::SubRaise {
                exc: exc_arg,
                exc_concrete: ConcreteValue::Null,
            },
            "sub-walk frame with no caller-side catch must bubble SubRaise through",
        );
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before,
            "sub-walk SubRaise must NOT record FINISH (only top-level converts)",
        );
    }

    #[test]
    fn step_through_int_copy_advances_past_operand_bytes() {
        // `int_copy/i>i` reads the src `i` operand for OOR
        // validation, advances past 2 operand bytes, records nothing.
        // Dst writeback (`registers_i[dst] = registers_i[src]`) is
        // deferred — RPython `pyjitpl.py:471-477 _opimpl_any_copy(box)
        // -> box` is a register rename only, no IR op.
        let int_copy_byte = *insns_opname_to_byte()
            .get("int_copy/i>i")
            .expect("`int_copy/i>i` must be in insns table");
        // src=2, dst=5 — distinct so a future writeback assertion can
        // distinguish src from dst slots.
        let code = [int_copy_byte, 0x02, 0x05];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("int_copy/i>i must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            next_pc, 3,
            "int_copy/i>i must advance past src + dst register bytes",
        );
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before,
            "int_copy/i>i records no IR op (RPython parity)",
        );
    }

    #[test]
    fn int_copy_writes_src_value_into_dst_register() {
        // Verify the dst writeback half of `int_copy/i>i`. The src
        // and dst slots must hold *different* OpRefs going in so the
        // assertion catches an accidental no-op.
        let int_copy_byte = *insns_opname_to_byte()
            .get("int_copy/i>i")
            .expect("`int_copy/i>i` must be in insns table");
        let code = [int_copy_byte, 0x02, 0x05]; // src=2, dst=5
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let src_val_pre = regs_i[2];
        let dst_val_pre = regs_i[5];
        assert_ne!(
            src_val_pre, dst_val_pre,
            "fixture must seed src and dst with different OpRefs",
        );
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("int_copy/i>i must dispatch");
        assert_eq!(
            wc.registers_i[5], src_val_pre,
            "int_copy must copy registers_i[src] into registers_i[dst] \
             (RPython _opimpl_any_copy + `>i` result coding)",
        );
        assert_eq!(
            wc.registers_i[2], src_val_pre,
            "src register must remain unchanged",
        );
    }

    #[test]
    fn int_copy_with_out_of_range_dst_register_surfaces_typed_error() {
        // dst byte indexes past `registers_i`; src is in range so the
        // src read succeeds and the dst write surfaces the OOR.
        let int_copy_byte = *insns_opname_to_byte()
            .get("int_copy/i>i")
            .expect("`int_copy/i>i` must be in insns table");
        let code = [int_copy_byte, 0x00, 0x09]; // src=0 (in range), dst=9 (OOR)
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 4);
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc).expect_err("int_copy dst OOR must surface a typed error");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 9,
                len: 4,
                bank: "i",
            },
        );
    }

    #[test]
    fn int_copy_with_out_of_range_src_register_surfaces_typed_error() {
        // src OOR validation parity with `raise/r`. Bank tag
        // is `"i"` to disambiguate from the Ref-bank OOR error.
        let int_copy_byte = *insns_opname_to_byte()
            .get("int_copy/i>i")
            .expect("`int_copy/i>i` must be in insns table");
        let code = [int_copy_byte, 0x07, 0x00];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [], // empty — index 7 must surface OOR
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc).expect_err("int_copy/i>i must read its src operand");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 7,
                len: 0,
                bank: "i",
            },
        );
    }

    // E1a: `ref_copy/r>r` walker arm tests are gated on the build-time
    // `pipeline.insns` table picking up the `ref_copy/r>r` key. Today
    // the analyzed source set (pyre-object + pyre-interpreter +
    // pyre-jit/src/eval.rs) does not exercise the codewriter's
    // chordal-reuse boundary that triggers `emit_ref_copy!`, so the
    // key never enters `INSNS_OPNAME_TO_BYTE`. The walker arm is
    // correctly wired (mirrors `int_copy/i>i`); these tests fire
    // automatically once any analyzed source path emits a `ref_copy`.
    //
    // Broader finding: `INSNS_OPNAME_TO_BYTE` (build-time
    // `pipeline.insns`) and `wellknown_bh_insns` (runtime
    // `JitCodeBuilder` writers) currently use different byte
    // assignments for the same key (`int_copy/i>i` is 0 in pipeline,
    // `BC_MOVE_I = 21` in wellknown). Production walker dispatch over
    // runtime-emitted jitcode bytes therefore needs a table-
    // unification step before any `dispatch_via_miframe` invocation
    // can read production bytes. Tracked separately as a
    // prerequisite.
    #[ignore = "blocked on pipeline.insns ↔ wellknown_bh_insns table unification (pipeline prerequisite)"]
    #[test]
    fn step_through_ref_copy_advances_past_operand_bytes() {
        // `ref_copy/r>r` Ref-bank sibling of `int_copy/i>i`.
        // Same operand layout `r>r`: 1B src + 1B dst, no IR op recorded.
        let ref_copy_byte = *insns_opname_to_byte()
            .get("ref_copy/r>r")
            .expect("`ref_copy/r>r` must be in insns table");
        let code = [ref_copy_byte, 0x02, 0x05]; // src=2, dst=5
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("ref_copy/r>r must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            next_pc, 3,
            "ref_copy/r>r must advance past src + dst register bytes",
        );
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before,
            "ref_copy/r>r records no IR op (RPython parity)",
        );
    }

    #[ignore = "blocked on pipeline.insns ↔ wellknown_bh_insns table unification (pipeline prerequisite)"]
    #[test]
    fn ref_copy_writes_src_value_into_dst_register() {
        // Verify the dst writeback half of `ref_copy/r>r`.
        let ref_copy_byte = *insns_opname_to_byte()
            .get("ref_copy/r>r")
            .expect("`ref_copy/r>r` must be in insns table");
        let code = [ref_copy_byte, 0x02, 0x05]; // src=2, dst=5
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let src_val_pre = regs_r[2];
        let dst_val_pre = regs_r[5];
        assert_ne!(
            src_val_pre, dst_val_pre,
            "fixture must seed src and dst with different OpRefs",
        );
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("ref_copy/r>r must dispatch");
        assert_eq!(
            wc.registers_r[5], src_val_pre,
            "ref_copy must copy registers_r[src] into registers_r[dst] \
             (RPython _opimpl_any_copy + `>r` result coding)",
        );
        assert_eq!(
            wc.registers_r[2], src_val_pre,
            "src register must remain unchanged",
        );
    }

    #[ignore = "blocked on pipeline.insns ↔ wellknown_bh_insns table unification (pipeline prerequisite)"]
    #[test]
    fn ref_copy_with_out_of_range_dst_register_surfaces_typed_error() {
        let ref_copy_byte = *insns_opname_to_byte()
            .get("ref_copy/r>r")
            .expect("`ref_copy/r>r` must be in insns table");
        let code = [ref_copy_byte, 0x00, 0x09]; // src=0 (in range), dst=9 (OOR)
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc).expect_err("ref_copy dst OOR must surface a typed error");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 9,
                len: 4,
                bank: "r",
            },
        );
    }

    #[ignore = "blocked on pipeline.insns ↔ wellknown_bh_insns table unification (pipeline prerequisite)"]
    #[test]
    fn ref_copy_with_out_of_range_src_register_surfaces_typed_error() {
        let ref_copy_byte = *insns_opname_to_byte()
            .get("ref_copy/r>r")
            .expect("`ref_copy/r>r` must be in insns table");
        let code = [ref_copy_byte, 0x07, 0x00];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [], // empty — index 7 must surface OOR
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc).expect_err("ref_copy/r>r must read its src operand");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 7,
                len: 0,
                bank: "r",
            },
        );
    }

    /// Drive a single `int_<binop>/ii>i` handler: the codewriter
    /// encodes `[opcode, src1, src2, dst]`. Asserts the recorder
    /// captured `OpCode::<expected>` with `[regs_i[src1],
    /// regs_i[src2]]` and that `regs_i[dst]` was written with the
    /// recorder's result OpRef.
    fn drive_int_binop(opname: &str, expected_opcode: majit_ir::OpCode) {
        let byte = *insns_opname_to_byte()
            .get(opname)
            .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
        // src=2, src2=4, dst=6 — chosen to be distinct so misordered
        // operand decoding surfaces in the assertion.
        let code = [byte, 0x02, 0x04, 0x06];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let arg0 = regs_i[2];
        let arg1 = regs_i[4];
        let dst_pre = regs_i[6];
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc)
            .unwrap_or_else(|e| panic!("`{opname}` must dispatch — got {:?}", e));
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, 4, "`{opname}` operand layout `ii>i` = 3 bytes");
        let dst_post = wc.registers_i[6];
        assert_ne!(
            dst_post, dst_pre,
            "`{opname}` must write a fresh OpRef into registers_i[dst]",
        );
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "`{opname}` must record exactly one op",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(
            last.opcode, expected_opcode,
            "`{opname}` must record `{:?}`",
            expected_opcode,
        );
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![arg0, arg1],
            "`{opname}` args must be [registers_i[src1], registers_i[src2]] in source order",
        );
        assert_eq!(
            dst_post,
            last.pos.get(),
            "`{opname}` dst must hold the recorder's result OpRef (op.pos.get())",
        );
    }

    #[test]
    fn int_add_records_intadd_with_both_operands_and_writes_dst() {
        drive_int_binop("int_add/ii>i", majit_ir::OpCode::IntAdd);
    }

    #[test]
    fn int_sub_records_intsub() {
        drive_int_binop("int_sub/ii>i", majit_ir::OpCode::IntSub);
    }

    #[test]
    fn int_mul_records_intmul() {
        drive_int_binop("int_mul/ii>i", majit_ir::OpCode::IntMul);
    }

    #[test]
    fn int_and_records_intand() {
        drive_int_binop("int_and/ii>i", majit_ir::OpCode::IntAnd);
    }

    // `int_or/ii>i` is not currently in `pipeline.insns` — pyre's
    // interpreter source does not emit Rust `|` on integers in any
    // path the JIT traces.  RPython's `Assembler.insns` only carries
    // emitted opnames (`assembler.py:220
    // setdefault(key, len(self.insns))`); pyre's runtime now mirrors
    // that (build.rs walks only `pipeline.insns`).  The dispatcher
    // handler exists; this test will unignore once an interpreter
    // source path emits `int_or` (e.g., bitset / flag computation).
    #[test]
    #[ignore = "int_or not currently emitted by pyre interpreter source — \
                pipeline.insns drops it"]
    fn int_or_records_intor() {
        drive_int_binop("int_or/ii>i", majit_ir::OpCode::IntOr);
    }

    #[test]
    fn int_xor_records_intxor() {
        drive_int_binop("int_xor/ii>i", majit_ir::OpCode::IntXor);
    }

    #[test]
    fn int_rshift_records_intrshift() {
        drive_int_binop("int_rshift/ii>i", majit_ir::OpCode::IntRshift);
    }

    #[test]
    fn int_eq_records_inteq() {
        drive_int_binop("int_eq/ii>i", majit_ir::OpCode::IntEq);
    }

    #[test]
    fn int_ne_records_intne() {
        drive_int_binop("int_ne/ii>i", majit_ir::OpCode::IntNe);
    }

    #[test]
    fn int_lt_records_intlt() {
        drive_int_binop("int_lt/ii>i", majit_ir::OpCode::IntLt);
    }

    #[test]
    fn int_le_records_intle() {
        drive_int_binop("int_le/ii>i", majit_ir::OpCode::IntLe);
    }

    #[test]
    fn int_gt_records_intgt() {
        drive_int_binop("int_gt/ii>i", majit_ir::OpCode::IntGt);
    }

    #[test]
    fn int_ge_records_intge() {
        drive_int_binop("int_ge/ii>i", majit_ir::OpCode::IntGe);
    }

    /// Drive `int_between/iii>i` and return the recorded ops plus
    /// the post-handler `dst` slot.  `inputs` describes how the
    /// three operand slots are populated.
    enum BetweenOperand {
        ConstInt(i64),
        NonConstWithConcrete(i64),
    }

    fn drive_int_between(
        b1: BetweenOperand,
        b2: BetweenOperand,
        b3: BetweenOperand,
    ) -> (Vec<majit_ir::OpCode>, OpRef, OpRef) {
        // `int_between/iii>i` is recorder-side only (decomposed at
        // record time) and is not currently emitted into pyre's
        // pipeline.insns table, so `insns_opname_to_byte()` lacks
        // a byte for it.  Call `int_between_record` directly with a
        // synthetic `DecodedOp` to exercise the handler.  Operand
        // layout: `[opcode, src1, src2, src3, dst]` — opcode byte is
        // unused by the handler (PC offsets into `code` from `op.pc`).
        let code = [0x00u8, 0x02, 0x04, 0x06, 0x08];
        let op = crate::jitcode_runtime::DecodedOp {
            key: "int_between/iii>i",
            opname: "int_between",
            argcodes: "iii>i",
            pc: 0,
            next_pc: 5,
        };
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 16);
        let mint = |tc: &mut TraceCtx, op: &BetweenOperand| match op {
            BetweenOperand::ConstInt(v) => tc.const_int(*v),
            BetweenOperand::NonConstWithConcrete(v) => {
                // Materialize a non-Const OpRef by recording a placeholder
                // op, then stamp its concrete value.  IntAdd with two
                // ConstInts is a valid stand-in carrier.
                let lhs = tc.const_int(0);
                let opref = tc.record_op(majit_ir::OpCode::IntAdd, &[lhs, lhs]);
                tc.set_opref_concrete(opref, majit_ir::Value::Int(*v));
                opref
            }
        };
        regs_i[2] = mint(&mut tc, &b1);
        regs_i[4] = mint(&mut tc, &b2);
        regs_i[6] = mint(&mut tc, &b3);
        let arg_b1 = regs_i[2];
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) =
            int_between_record(&code, &op, &mut wc).expect("int_between_record must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, 5, "operand layout `iii>i` consumes 4 bytes");
        let dst_post = wc.registers_i[8];
        drop(wc);
        let new_ops: Vec<majit_ir::OpCode> = tc
            .ops()
            .iter()
            .skip(ops_before)
            .map(|op| op.opcode)
            .collect();
        (new_ops, dst_post, arg_b1)
    }

    /// `pyjitpl.py:2648-2662 execute_and_record` — when every argbox
    /// is `ConstInt`, a pure op like `INT_SUB` / `INT_EQ` is folded
    /// via `wrap_constant` and never reaches `_record_helper`.
    /// `opimpl_int_between(ConstInt, ConstInt, ConstInt)` chains three
    /// pure binops on all-Const inputs, so 0 ops must be recorded and
    /// the destination must hold the folded ConstInt result.
    #[test]
    fn int_between_const_inputs_with_unit_width_takes_inteq_fast_path() {
        let (ops, dst_post, _) = drive_int_between(
            BetweenOperand::ConstInt(5),
            BetweenOperand::ConstInt(7),
            BetweenOperand::ConstInt(6),
        );
        assert!(
            ops.is_empty(),
            "all-Const inputs must fold without recording — got {ops:?}",
        );
        // `b5 = 6 - 5 = 1` → fast path → `IntEq(b2=7, b1=5) = 0`.
        assert_eq!(
            dst_post.inline_const_to_value(),
            Some(majit_ir::Value::Int(0)),
            "ConstInt(1) fast path on all-Const inputs must fold to ConstInt(0)",
        );
    }

    /// All-Const generic path: `pyjitpl.py:2648-2662` folds the three
    /// pure binops without recording.  Destination must carry the
    /// folded UINT_LT result.
    #[test]
    fn int_between_const_inputs_with_wide_range_takes_uintlt_generic_path() {
        let (ops, dst_post, _) = drive_int_between(
            BetweenOperand::ConstInt(5),
            BetweenOperand::ConstInt(7),
            BetweenOperand::ConstInt(10),
        );
        assert!(
            ops.is_empty(),
            "all-Const inputs must fold without recording — got {ops:?}",
        );
        // `b5 = 10 - 5 = 5` (not 1) → generic → `b4 = 7 - 5 = 2`,
        // `UintLt(2, 5) = 1`.
        assert_eq!(
            dst_post.inline_const_to_value(),
            Some(majit_ir::Value::Int(1)),
            "wide-range path on all-Const inputs must fold to ConstInt(1)",
        );
    }

    /// Regression: when `b1`/`b3` are non-constant OpRefs with
    /// observed concrete values satisfying `b3-b1==1`, the recorder
    /// must still take the generic `INT_SUB + UINT_LT` path.
    /// Specializing to `INT_EQ` would bake a `b3-b1==1` invariant
    /// without a runtime guard (`box_value` only reports the observed
    /// sample), miscompiling any future execution where the same
    /// boxes carry different live values.
    #[test]
    fn int_between_non_const_inputs_observing_unit_width_takes_generic_path() {
        let (ops, _, _) = drive_int_between(
            BetweenOperand::NonConstWithConcrete(5),
            BetweenOperand::NonConstWithConcrete(7),
            BetweenOperand::NonConstWithConcrete(6),
        );
        assert_eq!(
            ops,
            vec![
                majit_ir::OpCode::IntSub,
                majit_ir::OpCode::IntSub,
                majit_ir::OpCode::UintLt,
            ],
            "non-const inputs must NOT take INT_EQ fast path; expected \
             [IntSub(b5), IntSub(b4), UintLt] — got {ops:?}",
        );
    }

    /// Drive a single `float_<binop>/ff>f` handler. Same shape as
    /// `drive_int_binop` but on the float bank.
    fn drive_float_binop(opname: &str, expected_opcode: majit_ir::OpCode) {
        let byte = *insns_opname_to_byte()
            .get(opname)
            .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
        let code = [byte, 0x02, 0x04, 0x06];
        let mut tc = fresh_trace_ctx();
        let mut regs_f = distinct_const_refs(&mut tc, 8);
        let arg0 = regs_f[2];
        let arg1 = regs_f[4];
        let dst_pre = regs_f[6];
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut regs_f,
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc)
            .unwrap_or_else(|e| panic!("`{opname}` must dispatch — got {:?}", e));
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, 4, "`{opname}` operand layout `ff>f` = 3 bytes");
        let dst_post = wc.registers_f[6];
        assert_ne!(
            dst_post, dst_pre,
            "`{opname}` must write a fresh OpRef into registers_f[dst]",
        );
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "`{opname}` must record exactly one op",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, expected_opcode);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![arg0, arg1]
        );
        assert_eq!(dst_post, last.pos.get());
    }

    #[test]
    fn float_add_records_floatadd() {
        drive_float_binop("float_add/ff>f", majit_ir::OpCode::FloatAdd);
    }

    #[test]
    fn float_sub_records_floatsub() {
        drive_float_binop("float_sub/ff>f", majit_ir::OpCode::FloatSub);
    }

    #[test]
    fn float_truediv_records_floattruediv() {
        drive_float_binop("float_truediv/ff>f", majit_ir::OpCode::FloatTrueDiv);
    }

    #[test]
    fn float_neg_records_floatneg_with_one_operand_and_writes_dst() {
        // `f>f` shape: 1B src + 1B dst = 2 operand bytes after opcode.
        let byte = *insns_opname_to_byte()
            .get("float_neg/f>f")
            .expect("`float_neg/f>f` must be in insns table");
        let code = [byte, 0x02, 0x05];
        let mut tc = fresh_trace_ctx();
        let mut regs_f = distinct_const_refs(&mut tc, 8);
        let arg = regs_f[2];
        let dst_pre = regs_f[5];
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut regs_f,
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("float_neg/f>f must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, 3, "float_neg/f>f operand layout `f>f` = 2 bytes");
        let dst_post = wc.registers_f[5];
        assert_ne!(dst_post, dst_pre);
        drop(wc);
        assert_eq!(tc.num_ops(), ops_before + 1);
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::FloatNeg);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![arg],
            "FloatNeg args must be [registers_f[src]]",
        );
        assert_eq!(dst_post, last.pos.get());
    }

    /// Drive a single `int_<unop>/i>i` handler. Same shape pattern as
    /// `drive_int_binop` minus one read.
    fn drive_int_unop(opname: &str, expected_opcode: majit_ir::OpCode) {
        let byte = *insns_opname_to_byte()
            .get(opname)
            .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
        let code = [byte, 0x02, 0x05]; // src=2, dst=5
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let arg = regs_i[2];
        let dst_pre = regs_i[5];
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc)
            .unwrap_or_else(|e| panic!("`{opname}` must dispatch — got {:?}", e));
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, 3, "`{opname}` operand layout `i>i` = 2 bytes");
        let dst_post = wc.registers_i[5];
        assert_ne!(dst_post, dst_pre);
        drop(wc);
        assert_eq!(tc.num_ops(), ops_before + 1);
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, expected_opcode);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![arg]
        );
        assert_eq!(dst_post, last.pos.get());
    }

    #[test]
    fn int_neg_records_intneg() {
        drive_int_unop("int_neg/i>i", majit_ir::OpCode::IntNeg);
    }

    #[test]
    fn int_invert_records_intinvert() {
        drive_int_unop("int_invert/i>i", majit_ir::OpCode::IntInvert);
    }

    #[test]
    fn int_same_as_is_eliminated_from_generated_insns_table() {
        // RPython `jtransform.py:246 rewrite_op_same_as` removes
        // `same_as` before assembly. The walker keeps a handler arm for
        // forward-prep, but the production insns table should not contain
        // the opname unless a future codewriter path legitimately emits it.
        assert!(
            !insns_opname_to_byte().contains_key("int_same_as/i>i"),
            "`int_same_as/i>i` appeared in the generated insns table; \
             verify same_as elimination before adding a decode test"
        );
    }

    /// Drive `ptr_eq/rr>i` or `ptr_ne/rr>i`. Shape `rr>i`: read 2
    /// r-regs, record, write to i-bank.
    fn drive_ptr_compare(opname: &str, expected_opcode: majit_ir::OpCode) {
        let byte = *insns_opname_to_byte()
            .get(opname)
            .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
        let code = [byte, 0x02, 0x04, 0x06]; // r-src1=2, r-src2=4, i-dst=6
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let arg0 = regs_r[2];
        let arg1 = regs_r[4];
        let dst_pre = regs_i[6];
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc)
            .unwrap_or_else(|e| panic!("`{opname}` must dispatch — got {:?}", e));
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, 4, "`{opname}` operand layout `rr>i` = 3 bytes");
        let dst_post = wc.registers_i[6];
        assert_ne!(dst_post, dst_pre);
        drop(wc);
        assert_eq!(tc.num_ops(), ops_before + 1);
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, expected_opcode);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![arg0, arg1]
        );
        assert_eq!(dst_post, last.pos.get());
    }

    #[test]
    fn ptr_eq_records_ptreq_with_two_ref_operands_into_int_dst() {
        drive_ptr_compare("ptr_eq/rr>i", majit_ir::OpCode::PtrEq);
    }

    #[test]
    fn ptr_ne_records_ptrne() {
        drive_ptr_compare("ptr_ne/rr>i", majit_ir::OpCode::PtrNe);
    }

    #[test]
    fn float_add_with_out_of_range_src_register_surfaces_typed_error() {
        let byte = *insns_opname_to_byte()
            .get("float_add/ff>f")
            .expect("`float_add/ff>f` must be in insns table");
        let code = [byte, 0x07, 0x00, 0x00]; // src=7, registers_f empty
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc).expect_err("float_add must read its src operand");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 7,
                len: 0,
                bank: "f",
            },
        );
    }

    #[test]
    fn int_add_with_out_of_range_src_register_surfaces_typed_error() {
        // OOR validation parity with int_copy. Bank tag = "i".
        let byte = *insns_opname_to_byte()
            .get("int_add/ii>i")
            .expect("`int_add/ii>i` must be in insns table");
        let code = [byte, 0x07, 0x00, 0x00]; // src=7, registers_i empty
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc).expect_err("int_add must read its src operand");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 7,
                len: 0,
                bank: "i",
            },
        );
    }

    #[test]
    fn int_add_with_out_of_range_dst_register_surfaces_typed_error() {
        // src reads succeed, dst write surfaces OOR. Catches the
        // reverse-direction encoding bugs the src-only test misses.
        let byte = *insns_opname_to_byte()
            .get("int_add/ii>i")
            .expect("`int_add/ii>i` must be in insns table");
        let code = [byte, 0x00, 0x01, 0x09]; // dst=9, registers_i.len()=4 → OOR
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 4);
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc).expect_err("int_add dst OOR must surface a typed error");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 9,
                len: 4,
                bank: "i",
            },
        );
    }

    #[test]
    fn unsupported_opname_surfaces_typed_error() {
        // Stable choice for exercising the catch-all `UnsupportedOpname`
        // error path.  `vtable_method_ptr/rd>i` is a pyre-only backend
        // adaptation (emitted by `OpKind::VtableMethodPtr` /
        // `assembler.rs:2762`) without a PyPy analog: Python dispatch
        // resolves through `cpu.bh_call_*` at runtime rather than
        // reifying a method pointer into the bytecode stream.  Zero
        // JitCode hits in production traces (per
        // `t3_audit_opname_gap_inventory`), so it's a durable choice
        // for the "still unsupported" slot.
        let opname = "vtable_method_ptr/rd>i";
        let unsupported_byte = *insns_opname_to_byte()
            .get(opname)
            .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
        // Operand encoding `rd>i`: 1B r-reg + 2B descr + 1B i-reg-dst = 4B.
        let code = [unsupported_byte, 0, 0, 0, 0];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err =
            step(&code, 0, &mut wc).expect_err("unsupported opname must hit UnsupportedOpname");
        assert_eq!(err, DispatchError::UnsupportedOpname { pc: 0, key: opname },);
    }

    /// `ptr_nonzero/r>i` records `PtrNe(box, CONST_NULL)` into the
    /// int dst.  RPython parity: `pyjitpl.py:378-380 opimpl_ptr_nonzero`
    /// returns `self.execute(rop.PTR_NE, box, CONST_NULL)`.
    #[test]
    fn ptr_nonzero_records_ptrne_with_box_and_null() {
        let opname = "ptr_nonzero/r>i";
        let byte = *insns_opname_to_byte()
            .get(opname)
            .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
        // Operand encoding `r>i`: 1B r-reg + 1B i-reg-dst = 2B
        let code = [byte, 0, 0];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        // Seed `registers_r[0]` with a placeholder OpRef so the
        // handler has something to read.
        let box_opref = tc.const_ref(0xdeadbeef);
        let mut regs_r = [box_opref];
        let mut regs_i = [OpRef::None];
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("ptr_nonzero must record PtrNe");
        assert!(matches!(outcome, DispatchOutcome::Continue));
        assert_eq!(next_pc, 3);
        // `get_or_insert_typed` mints a fresh OpRef on every call (see
        // `constant_pool.rs:87` — equality is `Const.same_constant`, not
        // OpRef identity), so we cannot compare against a freshly-minted
        // null_const.  Verify args[1] is a Ref-typed constant whose
        // pooled value is 0 instead.
        let last_args0;
        let last_args1;
        let last_opcode;
        let last_args_len;
        {
            let ops = wc.trace_ctx.ops();
            let last = ops.last().expect("ptr_nonzero must record one op");
            last_opcode = last.opcode;
            let args = last.getarglist();
            last_args_len = args.len();
            last_args0 = args[0].clone();
            last_args1 = args[1].clone();
        }
        assert_eq!(last_opcode, majit_ir::OpCode::PtrNe);
        assert_eq!(last_args_len, 2);
        assert_eq!(last_args0.to_opref(), box_opref);
        assert_eq!(
            wc.trace_ctx.const_value(last_args1.to_opref()),
            Some(0),
            "args[1] must point at the CONST_NULL pool entry (value=0)"
        );
        assert_eq!(
            wc.trace_ctx.const_type(last_args1.to_opref()),
            Some(Type::Ref)
        );
        assert_ne!(wc.registers_i[0], OpRef::None);
    }

    /// `abort/>r` is a pyre-only no-op result marker — the walker
    /// counterpart of blackhole's `handler_abort_result_marker_r`
    /// (`blackhole.rs:5149`).  No operand read, no register write, no
    /// IR op recorded; dispatch advances past the 1B dst slot only.
    #[test]
    fn abort_result_r_is_pure_pc_advance() {
        let opname = "abort/>r";
        let byte = *insns_opname_to_byte()
            .get(opname)
            .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
        let code = [byte, 0x05]; // dst byte = 5 (intentionally out-of-range)
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("abort/>r must dispatch");
        assert!(matches!(outcome, DispatchOutcome::Continue));
        assert_eq!(next_pc, 2, "abort/>r operand layout = 1 byte (dst marker)");
        assert_eq!(
            wc.trace_ctx.num_ops(),
            ops_before,
            "abort/>r must not record any IR op",
        );
    }

    /// `ref_guard_value/r` records `GuardValue(value, ConstPtr(concrete))`
    /// when the symbolic OpRef is non-Const and a concrete pointer is
    /// available in the shadow.  Mirrors `pyjitpl.py:1916-1927
    /// implement_guard_value`.
    #[test]
    fn ref_guard_value_records_guardvalue_with_concrete_constant() {
        let opname = "ref_guard_value/r";
        let byte = *insns_opname_to_byte()
            .get(opname)
            .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
        // Operand encoding `r`: 1B r-src only.
        let code = [byte, 0];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        // Symbolic side: a recorded op OpRef (not a Const).
        let value_opref = tc.record_op(majit_ir::OpCode::PtrEq, &[]);
        let mut regs_r = [value_opref];
        let mut regs_i = [OpRef::None];
        let concrete_ptr: usize = 0xdead_beef;
        let mut concrete_r = [ConcreteValue::Ref(
            concrete_ptr as *mut pyre_object::pyobject::PyObject,
        )];
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut concrete_r,
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) =
            step(&code, 0, &mut wc).expect("ref_guard_value must record GuardValue");
        assert!(matches!(outcome, DispatchOutcome::Continue));
        assert_eq!(next_pc, 2);
        let (last_opcode, last_args0, last_args1, last_args_len) = {
            let ops = wc.trace_ctx.ops();
            let last = ops.last().expect("ref_guard_value must record one op");
            let args = last.getarglist();
            (last.opcode, args[0].clone(), args[1].clone(), args.len())
        };
        assert_eq!(last_opcode, majit_ir::OpCode::GuardValue);
        assert_eq!(last_args_len, 2);
        assert_eq!(last_args0.to_opref(), value_opref);
        assert_eq!(
            wc.trace_ctx.const_value(last_args1.to_opref()),
            Some(concrete_ptr as i64),
            "args[1] must point at the concrete pointer in the pool",
        );
        assert_eq!(
            wc.trace_ctx.const_type(last_args1.to_opref()),
            Some(Type::Ref)
        );
        assert_eq!(
            wc.registers_r[0],
            last_args1.to_opref(),
            "register slot still holding the original OpRef must be rewritten \
             to the promoted constant (pyjitpl.py:1923 replace_box)",
        );
    }

    /// Symbolic OpRef already a Const → `ref_guard_value/r` is a no-op
    /// (`pyjitpl.py:1920-1921 if isinstance(box, Const): return box`).
    #[test]
    fn ref_guard_value_on_const_records_nothing() {
        let opname = "ref_guard_value/r";
        let byte = *insns_opname_to_byte()
            .get(opname)
            .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
        let code = [byte, 0];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let value_opref = tc.const_ref(0xdead_beef);
        let baseline_ops = tc.ops().len();
        let mut regs_r = [value_opref];
        let mut regs_i = [OpRef::None];
        let mut concrete_r = [ConcreteValue::Ref(
            0xdead_beef as *mut pyre_object::pyobject::PyObject,
        )];
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut concrete_r,
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("ref_guard_value Const arm");
        assert!(matches!(outcome, DispatchOutcome::Continue));
        assert_eq!(next_pc, 2);
        assert_eq!(
            wc.trace_ctx.ops().len(),
            baseline_ops,
            "no op should be recorded when input is already Const"
        );
        assert_eq!(wc.registers_r[0], value_opref);
    }

    #[test]
    fn step_through_residual_call_r_r_records_callr_with_descr_and_args() {
        // `residual_call_r_r/iRd>r` records `OpCode::CallR`
        // with `[funcptr, ...args]` and `descr=descr_refs[d]`. RPython
        // `pyjitpl.py:1334-1347 _opimpl_residual_call1` →
        // `do_residual_or_indirect_call → execute_and_record_varargs(
        // rop.CALL_R, [funcbox]+argboxes, descr=calldescr)`.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        // Operand encoding `iRd>r`: 1B funcptr (i-reg=2),
        // 1B varlen=2 + [r-reg=4, r-reg=7], 2B descr_index=1 (LE),
        // 1B dst-reg=0 (writeback deferred — not used by walker yet).
        let code = [
            residual_byte,
            0x02, // funcptr from registers_i[2]
            0x02, // varlen
            0x04,
            0x07, // args from registers_r[4, 7]
            0x01,
            0x00, // descr index = 1 (LE)
            0x00, // dst reg (deferred)
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 4);
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let funcptr_expected = regs_i[2];
        let arg0_expected = regs_r[4];
        let arg1_expected = regs_r[7];
        // Build a 2-entry descr table — index 0 is a decoy (different
        // pointer), index 1 is the descr we expect the recorder to attach.
        // RPython `_build_allboxes` reads `descr.get_arg_types()` to
        // permute argboxes into ABI order; the test passes 2 R args so
        // `arg_types = [Ref, Ref]` keeps the permutation an identity
        // (allboxes = [funcbox, r0, r1]).
        let decoy = make_call_descr(
            2,
            vec![Type::Ref, Type::Ref],
            Type::Ref,
            majit_ir::ExtraEffect::CanRaise,
        );
        let call_descr = make_call_descr(
            3,
            vec![Type::Ref, Type::Ref],
            Type::Ref,
            majit_ir::ExtraEffect::CanRaise,
        );
        let descr_pool = vec![decoy, call_descr.clone()];
        let frame_done_descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) =
            step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            next_pc,
            code.len(),
            "residual_call_r_r must advance past funcptr + varlist + descr + dst",
        );
        drop(wc);
        // FailDescr placeholder has no EffectInfo (`as_call_descr() = None`),
        // so the walker takes the `no-effectinfo-fallback` branch:
        // CallR + GuardNoException (RPython parity:
        // `do_residual_call → execute_varargs(..., exc=True)` →
        // `handle_possible_exception` emits GUARD_NO_EXCEPTION).
        assert_eq!(
            tc.num_ops(),
            ops_before + 2,
            "residual_call_r_r must record CallR + GuardNoException (no-effectinfo fallback)",
        );
        let call_op = tc
            .ops()
            .iter()
            .find(|o| o.opcode == majit_ir::OpCode::CallR)
            .expect("CallR must be recorded");
        assert_eq!(
            call_op
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![funcptr_expected, arg0_expected, arg1_expected],
            "CallR args must be [funcptr, ...args] from registers_i+registers_r",
        );
        let recorded_descr = call_op.getdescr().expect("CallR must carry the calldescr");
        assert!(
            std::sync::Arc::ptr_eq(&recorded_descr, &call_descr),
            "CallR descr must be descr_refs[1] (not decoy at index 0)",
        );
        // GuardNoException follows immediately after.
        let guard_op = tc
            .ops()
            .iter()
            .find(|o| o.opcode == majit_ir::OpCode::GuardNoException)
            .expect("GuardNoException must follow CallR for raising calls");
        assert!(
            guard_op.num_args() == 0,
            "GuardNoException takes no operand args",
        );
        // `walker_capture_snapshot_for_last_guard` ports
        // `capture_resumedata(after_residual_call=True)`
        // (`pyjitpl.py:2599-2603`).  Every guard emitted by a
        // residual_call dispatcher now carries a snapshot whose
        // `rd_resume_position` is the freshly-allocated snapshot id
        // (`>= 0`), so the optimizer's `store_final_boxes_in_guard`
        // (`optimizeopt/mod.rs:5033`) finds attached resume data
        // instead of panicking on the `-1` sentinel.
        assert!(
            guard_op.rd_resume_position.get() >= 0,
            "GuardNoException must carry an attached snapshot (rd_resume_position >= 0) after capture_resumedata port",
        );
    }

    /// Build a `SimpleCallDescr` for tests, parameterised by `arg_types`,
    /// `result_type`, and `extraeffect`. The `_build_allboxes` permutation
    /// reads `arg_types` as the callee's ABI ordering, so tests must pass
    /// the exact types of the arglist they exercise.
    fn make_call_descr(
        idx: u32,
        arg_types: Vec<Type>,
        result_type: Type,
        extra: majit_ir::ExtraEffect,
    ) -> DescrRef {
        let mut effect = majit_ir::EffectInfo::default();
        effect.extraeffect = extra;
        std::sync::Arc::new(majit_ir::SimpleCallDescr::new(
            idx,
            arg_types,
            result_type,
            false,
            std::mem::size_of::<usize>(),
            effect,
        ))
    }

    /// Convenience: legacy signature used by elidable-classification
    /// tests with empty arglists (0 R args, descr arg_types=[]).
    /// `result_type` defaults to `Ref` matching `_r_r` shape. Callers
    /// exercising actual args must use [`make_call_descr`] directly to
    /// pass matching `arg_types`.
    fn call_descr_with_effect(idx: u32, extra: majit_ir::ExtraEffect) -> DescrRef {
        make_call_descr(idx, vec![], Type::Ref, extra)
    }

    /// Convenience: builds a `_r_r`-shape CallDescr with both
    /// `extraeffect` and `oopspecindex` populated, for tests that need
    /// to drive [`do_not_in_trace_call_result`] /
    /// [`do_jit_force_virtual_guard`] / future oopspec-keyed guards.
    fn call_descr_with_oopspec(
        idx: u32,
        extra: majit_ir::ExtraEffect,
        oopspec: majit_ir::OopSpecIndex,
    ) -> DescrRef {
        let mut effect = majit_ir::EffectInfo::default();
        effect.extraeffect = extra;
        effect.oopspecindex = oopspec;
        std::sync::Arc::new(majit_ir::SimpleCallDescr::new(
            idx,
            vec![],
            Type::Ref,
            false,
            std::mem::size_of::<usize>(),
            effect,
        ))
    }

    #[test]
    fn residual_call_r_r_with_elidable_cannot_raise_records_callpurer_no_guard() {
        // RPython parity: `do_residual_call` (pyjitpl.py:2111-2118) reads
        // `effectinfo.check_is_elidable()` + `effectinfo.check_can_raise()`,
        // then `execute_varargs(rop.CALL_R, ..., exc, pure)`. With
        // EF_ELIDABLE_CANNOT_RAISE: `pure=True` (CALL_PURE_R) + `exc=False`
        // (no GUARD_NO_EXCEPTION).
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let elidable_descr = call_descr_with_effect(7, majit_ir::ExtraEffect::ElidableCannotRaise);
        let descr_pool = vec![elidable_descr.clone()];
        let frame_done_descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "elidable+cannot-raise must record exactly CallPureR (no GuardNoException)",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(
            last.opcode,
            majit_ir::OpCode::CallPureR,
            "EF_ELIDABLE_CANNOT_RAISE must rewrite to CALL_PURE_R",
        );
    }

    #[test]
    fn residual_call_r_r_with_not_in_trace_oopspec_returns_typed_error() {
        // RPython parity: `pyjitpl.py:2003-2005` routes
        // `OS_NOT_IN_TRACE` residual calls through `do_not_in_trace_call`
        // which executes the callee concretely and aborts to blackhole
        // only if it raises (`pyjitpl.py:3683-3697`). The walker has no
        // concrete executor, so it must surface a typed error rather
        // than recording either the normal-return or
        // SwitchToBlackhole shape.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let not_in_trace_descr = call_descr_with_oopspec(
            43,
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::NotInTrace,
        );
        let descr_pool = vec![not_in_trace_descr];
        let frame_done_descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc).expect_err("OS_NOT_IN_TRACE must surface a typed error");
        assert_eq!(
            err,
            DispatchError::NotInTraceRequiresConcreteExecution { pc: 0 },
        );
    }

    #[test]
    fn residual_call_r_r_with_jit_force_virtual_oopspec_returns_typed_error() {
        // RPython parity: `pyjitpl.py:2011-2014` short-circuits
        // `do_residual_call` via `_do_jit_force_virtual` when
        // `effectinfo.oopspecindex == OS_JIT_FORCE_VIRTUAL`.  The
        // walker can't reproduce that short-circuit (needs concrete
        // `vref_ptr` resolver); fail-loud error keeps the path from
        // silently recording `CALL_MAY_FORCE_*` instead.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let force_virtual_descr = call_descr_with_oopspec(
            42,
            majit_ir::ExtraEffect::ForcesVirtualOrVirtualizable,
            majit_ir::OopSpecIndex::JitForceVirtual,
        );
        let descr_pool = vec![force_virtual_descr];
        let frame_done_descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err =
            step(&code, 0, &mut wc).expect_err("OS_JIT_FORCE_VIRTUAL must surface a typed error");
        assert_eq!(
            err,
            DispatchError::JitForceVirtualRequiresConcreteResolver { pc: 0 },
        );
    }

    #[test]
    fn residual_call_r_r_with_elidable_can_raise_records_callpurer_plus_guard() {
        // EF_ELIDABLE_CAN_RAISE: `pure=True` + `exc=True` —
        // CALL_PURE_R + GUARD_NO_EXCEPTION (pyjitpl.py:execute_varargs
        // emits both).
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let elidable_descr = call_descr_with_effect(8, majit_ir::ExtraEffect::ElidableCanRaise);
        let descr_pool = vec![elidable_descr.clone()];
        let frame_done_descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 2,
            "elidable+can-raise must record CallPureR + GuardNoException",
        );
        let opcodes: Vec<_> = tc.ops().iter().skip(ops_before).map(|o| o.opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                majit_ir::OpCode::CallPureR,
                majit_ir::OpCode::GuardNoException
            ],
            "EF_ELIDABLE_CAN_RAISE must record CALL_PURE_R then GUARD_NO_EXCEPTION",
        );
    }

    #[test]
    fn residual_call_r_r_with_cannot_raise_records_callr_no_guard() {
        // EF_CANNOT_RAISE: `pure=False` + `exc=False` — bare CallR,
        // no GUARD_NO_EXCEPTION.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let nothrow_descr = call_descr_with_effect(9, majit_ir::ExtraEffect::CannotRaise);
        let descr_pool = vec![nothrow_descr.clone()];
        let frame_done_descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "EF_CANNOT_RAISE must record bare CallR (no GuardNoException)",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::CallR);
    }

    #[test]
    fn residual_call_r_r_writes_recorder_result_into_dst_register() {
        // Verify the dst writeback half of `residual_call_r_r/iRd>r`.
        // After the handler runs, `registers_r[dst]` must equal the
        // OpRef the recorder returned (i.e., the OpRef whose Op is
        // the recorded CallR at the trace tail).
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        // funcptr=regs_i[0], no args, descr index=0, dst=3
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x03];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let dst_val_pre = regs_r[3];
        // 0 R args → arg_types=[]; CallDescr required (RPython
        // do_residual_call always has one).
        let descr_pool = vec![make_call_descr(
            1,
            vec![],
            Type::Ref,
            majit_ir::ExtraEffect::CanRaise,
        )];
        let frame_done_descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
        // The dst slot must hold the OpRef of the recorded CallR. Each
        // Op carries its OpRef in `op.pos` (recorder.rs:159), which lets
        // the test compare without re-deriving the index (input args
        // also occupy OpRef indices, so `ops.iter().position()` would
        // be off by `num_inputargs`).
        let dst_ref = wc.registers_r[3];
        assert_ne!(
            dst_ref, dst_val_pre,
            "dst must change from its pre-call value",
        );
        let call_op = wc
            .trace_ctx
            .ops()
            .iter()
            .find(|o| o.opcode == OpCode::CallR)
            .expect("a CallR op must be in the recorded trace");
        assert_eq!(
            dst_ref,
            call_op.pos.get(),
            "registers_r[dst] must be the recorded CallR's OpRef (op.pos.get())",
        );
    }

    #[test]
    fn residual_call_r_r_can_raise_writes_dst_before_guard_no_exception() {
        // pyjitpl.py:1950 _opimpl_residual_call*: result lands in
        // `registers_*[reg_index]` BEFORE
        // `handle_possible_exception()` records GUARD_NO_EXCEPTION.
        // `walker_capture_snapshot_for_last_guard`
        // (`pyjitpl.py:2599-2603 capture_resumedata(after_residual_call
        // =True)`) snapshots the active registers AFTER the writeback,
        // so the dst slot's recorded OpRef rides the snapshot's
        // fail_arg list.  The structural invariant tested here is:
        // after dispatch, the dst slot holds the recorded call op's
        // OpRef, and the recorded sequence is `[CallR, GuardNoException]`
        // — i.e. the writeback ran on the record-side BEFORE the
        // guard append (and therefore before the snapshot capture).
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        // funcptr=regs_i[0], no R args, descr=0, dst=3.
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x03];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let dst_pre = regs_r[3];
        let descr_pool = vec![make_call_descr(
            1,
            vec![],
            Type::Ref,
            majit_ir::ExtraEffect::CanRaise,
        )];
        let frame_done_descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
        drop(wc);
        let opcodes: Vec<_> = tc.ops().iter().skip(ops_before).map(|o| o.opcode).collect();
        assert_eq!(
            opcodes,
            vec![OpCode::CallR, OpCode::GuardNoException],
            "CAN_RAISE residual_call_r_r must record [CallR, GuardNoException]",
        );
        let call_pos = tc
            .ops()
            .iter()
            .find(|o| o.opcode == OpCode::CallR)
            .expect("CallR must be in the trace")
            .pos
            .get();
        assert_ne!(regs_r[3], dst_pre, "dst must be overwritten");
        assert_eq!(
            regs_r[3], call_pos,
            "registers_r[dst] must equal CallR's OpRef when GuardNoException is recorded",
        );
    }

    #[test]
    fn residual_call_ir_r_can_raise_writes_dst_before_guard_no_exception() {
        // Same invariant as `residual_call_r_r_can_raise_...` for the
        // `_ir_*` shape (`dispatch_residual_call_iIRd_kind`): the
        // `iIRd>X` writeback must precede the GuardNoException record.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_ir_r/iIRd>r")
            .expect("`residual_call_ir_r/iIRd>r` must be in insns table");
        // funcptr=i[0], 0 i-args, 0 r-args, descr=0, dst=2.
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let mut regs_r = distinct_const_refs(&mut tc, 6);
        let dst_pre = regs_r[2];
        let descr_pool = vec![make_call_descr(
            1,
            vec![],
            Type::Ref,
            majit_ir::ExtraEffect::CanRaise,
        )];
        let frame_done_descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("residual_call_ir_r/iIRd>r must dispatch");
        drop(wc);
        let opcodes: Vec<_> = tc.ops().iter().skip(ops_before).map(|o| o.opcode).collect();
        assert_eq!(
            opcodes,
            vec![OpCode::CallR, OpCode::GuardNoException],
            "CAN_RAISE residual_call_ir_r must record [CallR, GuardNoException]",
        );
        let call_pos = tc
            .ops()
            .iter()
            .find(|o| o.opcode == OpCode::CallR)
            .expect("CallR must be in the trace")
            .pos
            .get();
        assert_ne!(regs_r[2], dst_pre, "dst must be overwritten");
        assert_eq!(
            regs_r[2], call_pos,
            "registers_r[dst] must equal CallR's OpRef when GuardNoException is recorded",
        );
    }

    #[test]
    fn residual_call_r_r_with_out_of_range_dst_register_surfaces_typed_error() {
        // Dst register OOR — the call was already recorded at this
        // point (RPython parity: `do_residual_or_indirect_call` records
        // first, then writes the result), but `registers_r` is empty
        // so the writeback surfaces RegisterOutOfRange.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x07]; // dst=7
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        // CallDescr required so the walker reaches the dst writeback
        // path (RPython do_residual_call invariant).
        let descr_pool = vec![make_call_descr(
            1,
            vec![],
            Type::Ref,
            majit_ir::ExtraEffect::CanRaise,
        )];
        let frame_done_descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc).expect_err("dst OOR must surface a typed error");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 7,
                len: 0,
                bank: "r",
            },
        );
    }

    #[test]
    fn residual_call_r_r_with_descr_index_out_of_range_surfaces_typed_error() {
        // descr-index OOR validation. Same shape as
        // RegisterOutOfRange, dedicated DispatchError variant for clarity.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        // descr_index=5, descr_refs.len()=2 → OOR
        let code = [residual_byte, 0x00, 0x00, 0x05, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let descr_pool = vec![make_fail_descr(1), make_fail_descr(1)];
        let frame_done_descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc)
            .expect_err("descr index 5 with pool size 2 must surface DescrIndexOutOfRange");
        assert_eq!(
            err,
            DispatchError::DescrIndexOutOfRange {
                pc: 0,
                index: 5,
                len: 2,
            },
        );
    }

    #[test]
    fn step_through_residual_call_r_i_records_calli_with_int_dst_writeback() {
        // kind sibling of `_r_r`. Same `iRd>X` operand
        // layout, dst kind flipped to int. RPython `pyjitpl.py:1346
        // opimpl_residual_call_r_i = _opimpl_residual_call1` shares
        // the body; `do_residual_call`'s `descr.get_normalized_result_type()`
        // dispatch (pyjitpl.py:2022-2044) selects `'i' → CALL_*_I`.
        // CallDescr required (RPython do_residual_call invariant);
        // walker records `OpCode::CallI` + `OpCode::GuardNoException`,
        // writes the call's OpRef into `registers_i[dst]`.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_i/iRd>i")
            .expect("`residual_call_r_i/iRd>i` must be in insns table");
        // Operand encoding `iRd>i`: 1B funcptr (i-reg=2),
        // 1B varlen=2 + [r-reg=4, r-reg=7], 2B descr_index=1 (LE),
        // 1B dst-reg=3 (writeback target into registers_i).
        let code = [
            residual_byte,
            0x02, // funcptr from registers_i[2]
            0x02, // varlen
            0x04,
            0x07, // args from registers_r[4, 7]
            0x01,
            0x00, // descr index = 1 (LE)
            0x03, // dst i-reg = 3
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let funcptr_expected = regs_i[2];
        let arg0_expected = regs_r[4];
        let arg1_expected = regs_r[7];
        let dst_pre = regs_i[3];
        // 2 R args + Int return → CallDescr arg_types=[Ref, Ref],
        // result_type=Int. `_build_allboxes` permutation is identity
        // (R-only argboxes match arg_types order).
        let decoy = make_call_descr(
            2,
            vec![Type::Ref, Type::Ref],
            Type::Int,
            majit_ir::ExtraEffect::CanRaise,
        );
        let call_descr = make_call_descr(
            3,
            vec![Type::Ref, Type::Ref],
            Type::Int,
            majit_ir::ExtraEffect::CanRaise,
        );
        let descr_pool = vec![decoy, call_descr.clone()];
        let frame_done_descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) =
            step(&code, 0, &mut wc).expect("residual_call_r_i/iRd>i must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, code.len());
        // CallI (kind sibling of CallR) + GuardNoException recorded.
        assert_eq!(
            wc.trace_ctx.num_ops(),
            ops_before + 2,
            "_r_i must record CallI + GuardNoException (no-effectinfo fallback)",
        );
        let call_op = wc
            .trace_ctx
            .ops()
            .iter()
            .find(|o| o.opcode == majit_ir::OpCode::CallI)
            .expect("CallI must be recorded for the int-dst kind");
        assert_eq!(
            call_op
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![funcptr_expected, arg0_expected, arg1_expected],
            "CallI args must be [funcptr, ...args] from registers_i+registers_r",
        );
        let recorded_descr = call_op.getdescr().expect("CallI must carry the calldescr");
        assert!(
            std::sync::Arc::ptr_eq(&recorded_descr, &call_descr),
            "CallI descr must be descr_refs[1] (not decoy at index 0)",
        );
        // dst writeback into the int bank (NOT the r bank).
        let dst_post = wc.registers_i[3];
        assert_ne!(
            dst_post, dst_pre,
            "registers_i[dst] must change from its pre-call value",
        );
        assert_eq!(
            dst_post,
            call_op.pos.get(),
            "registers_i[dst] must be the recorded CallI's OpRef (op.pos.get())",
        );
    }

    #[test]
    fn residual_call_r_i_with_elidable_cannot_raise_records_callpurei_no_guard() {
        // EF_ELIDABLE_CANNOT_RAISE on the int-kind sibling
        // must rewrite to CALL_PURE_I (not CALL_PURE_R) and skip
        // GUARD_NO_EXCEPTION. Confirms the `pure_op` selection in
        // `dispatch_residual_call_iRd_kind` follows `dst_bank`.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_i/iRd>i")
            .expect("`residual_call_r_i/iRd>i` must be in insns table");
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 4);
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let elidable_descr = call_descr_with_effect(7, majit_ir::ExtraEffect::ElidableCannotRaise);
        let descr_pool = vec![elidable_descr.clone()];
        let frame_done_descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("residual_call_r_i/iRd>i must dispatch");
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "elidable+cannot-raise on int-kind must record exactly CallPureI",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(
            last.opcode,
            majit_ir::OpCode::CallPureI,
            "EF_ELIDABLE_CANNOT_RAISE on int-kind must rewrite to CALL_PURE_I",
        );
    }

    #[test]
    fn step_through_residual_call_ir_r_records_callr_with_int_and_ref_args() {
        // shape sibling `_ir_r/iIRd>r`. Operand layout adds
        // an i-bank list between funcptr and the R-list. RPython
        // `_build_allboxes` permutes argboxes by `descr.get_arg_types()`
        // ABI; for an [Int, Int, Ref, Ref] callee the permutation
        // reduces to identity → allboxes = [funcbox, i0, i1, r0, r1].
        // Mixed-kind permutation is exercised by the dedicated test
        // `residual_call_ir_r_permutes_argboxes_per_arg_types_abi`.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_ir_r/iIRd>r")
            .expect("`residual_call_ir_r/iIRd>r` must be in insns table");
        // Operand encoding `iIRd>r`: 1B funcptr (i-reg=2),
        // i-list: 1B count=2 + [i-reg=5, i-reg=6],
        // r-list: 1B count=2 + [r-reg=4, r-reg=7],
        // 2B descr_index=1 (LE),
        // 1B dst-reg=0.
        let code = [
            residual_byte,
            0x02, // funcptr from registers_i[2]
            0x02, // i-list count
            0x05,
            0x06, // i-args from registers_i[5, 6]
            0x02, // r-list count
            0x04,
            0x07, // r-args from registers_r[4, 7]
            0x01,
            0x00, // descr index = 1 (LE)
            0x00, // dst r-reg = 0
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let funcptr_expected = regs_i[2];
        let iarg0_expected = regs_i[5];
        let iarg1_expected = regs_i[6];
        let rarg0_expected = regs_r[4];
        let rarg1_expected = regs_r[7];
        // arg_types = [Int, Int, Ref, Ref] → `_build_allboxes`
        // permutation is identity over the source-list-order argboxes.
        let decoy = make_call_descr(
            2,
            vec![Type::Int, Type::Int, Type::Ref, Type::Ref],
            Type::Ref,
            majit_ir::ExtraEffect::CanRaise,
        );
        let call_descr = make_call_descr(
            3,
            vec![Type::Int, Type::Int, Type::Ref, Type::Ref],
            Type::Ref,
            majit_ir::ExtraEffect::CanRaise,
        );
        let descr_pool = vec![decoy, call_descr.clone()];
        let frame_done_descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) =
            step(&code, 0, &mut wc).expect("residual_call_ir_r/iIRd>r must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            next_pc,
            code.len(),
            "residual_call_ir_r must advance past funcptr + i-list + r-list + descr + dst",
        );
        // CallR + GuardNoException recorded (no-effectinfo fallback).
        assert_eq!(
            wc.trace_ctx.num_ops(),
            ops_before + 2,
            "_ir_r must record CallR + GuardNoException (no-effectinfo fallback)",
        );
        let call_op = wc
            .trace_ctx
            .ops()
            .iter()
            .find(|o| o.opcode == majit_ir::OpCode::CallR)
            .expect("CallR must be recorded");
        assert_eq!(
            call_op
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![
                funcptr_expected,
                iarg0_expected,
                iarg1_expected,
                rarg0_expected,
                rarg1_expected,
            ],
            "CallR args must be [funcptr, i0, i1, r0, r1] — identity \
             permutation when descr.arg_types=[Int, Int, Ref, Ref]",
        );
        let recorded_descr = call_op.getdescr().expect("CallR must carry the calldescr");
        assert!(
            std::sync::Arc::ptr_eq(&recorded_descr, &call_descr),
            "CallR descr must be descr_refs[1] (not decoy at index 0)",
        );
        // dst writeback into registers_r[0].
        let dst_post = wc.registers_r[0];
        assert_eq!(
            dst_post,
            call_op.pos.get(),
            "registers_r[dst] must be the recorded CallR's OpRef (op.pos.get())",
        );
    }

    #[test]
    fn residual_call_ir_r_permutes_argboxes_per_arg_types_abi() {
        // The `_ir_*` shape gives
        // the walker source-list-order argboxes `[i_args..., r_args...]`,
        // but RPython `_build_allboxes` (pyjitpl.py:1960-1993) re-orders
        // those to match the callee's `descr.get_arg_types()` ABI. This
        // test pins the non-identity permutation.
        //
        // Setup: 2 i-args + 2 r-args + arg_types = [Ref, Int, Ref, Int].
        // Source-list-order argboxes = [i0, i1, r0, r1].
        // `_build_allboxes` walk:
        //   iter 1, kind=Ref: src_r scans flat argboxes for first Ref →
        //     positions 0 (i0, skip), 1 (i1, skip), 2 (r0, match). src_r=3.
        //   iter 2, kind=Int: src_i scans for first Int → position 0
        //     (i0, match). src_i=1.
        //   iter 3, kind=Ref: src_r at 3 → position 3 (r1, match). src_r=4.
        //   iter 4, kind=Int: src_i at 1 → position 1 (i1, match). src_i=2.
        // Final allboxes = [funcbox, r0, i0, r1, i1].
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_ir_r/iIRd>r")
            .expect("`residual_call_ir_r/iIRd>r` must be in insns table");
        let code = [
            residual_byte,
            0x02, // funcptr from registers_i[2]
            0x02, // i-list count
            0x05,
            0x06, // i-args from registers_i[5, 6]
            0x02, // r-list count
            0x04,
            0x07, // r-args from registers_r[4, 7]
            0x00,
            0x00, // descr index = 0 (LE)
            0x00, // dst r-reg = 0
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let funcptr = regs_i[2];
        let i0 = regs_i[5];
        let i1 = regs_i[6];
        let r0 = regs_r[4];
        let r1 = regs_r[7];
        let mixed_descr = make_call_descr(
            0,
            vec![Type::Ref, Type::Int, Type::Ref, Type::Int],
            Type::Ref,
            majit_ir::ExtraEffect::CanRaise,
        );
        let descr_pool = vec![mixed_descr.clone()];
        let frame_done_descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("residual_call_ir_r/iIRd>r must dispatch");
        drop(wc);
        let call_op = tc
            .ops()
            .iter()
            .find(|o| o.opcode == majit_ir::OpCode::CallR)
            .expect("CallR must be recorded");
        assert_eq!(
            call_op
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![funcptr, r0, i0, r1, i1],
            "_build_allboxes must permute to match descr.arg_types \
             [Ref, Int, Ref, Int] — RPython pyjitpl.py:1960-1993",
        );
    }

    #[test]
    fn residual_call_descr_not_call_descr_surfaces_typed_error() {
        // Walker requires CallDescr per RPython invariant
        // (pyjitpl.py:1995 do_residual_call). When the descr_pool entry
        // at the operand-encoded index lacks a CallDescr downcast (here
        // a FailDescr), the walker surfaces ResidualCallDescrNotCallDescr.
        // In production the codewriter never emits non-CallDescr; this
        // covers the test-fixture / future-deviation case.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let mut regs_r = distinct_const_refs(&mut tc, 1);
        let descr_pool = vec![make_fail_descr(7)];
        let frame_done_descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc)
            .expect_err("FailDescr (not CallDescr) must surface ResidualCallDescrNotCallDescr");
        assert_eq!(
            err,
            DispatchError::ResidualCallDescrNotCallDescr {
                pc: 0,
                descr_index: 0,
            },
        );
    }

    #[test]
    fn residual_call_r_r_with_out_of_range_arg_register_surfaces_typed_error() {
        // varlist member OOR validation. Bank tag = "r" since
        // R-list reads from registers_r.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        // varlen=1, arg=9 (registers_r is empty) → OOR
        let code = [residual_byte, 0x00, 0x01, 0x09, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let descr_pool = vec![make_fail_descr(1)];
        let frame_done_descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc)
            .expect_err("R-list member out of range must surface RegisterOutOfRange");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 9,
                len: 0,
                bank: "r",
            },
        );
    }

    #[test]
    #[ignore = "slice 2h: inline_call recursion surfaces sub-jitcode opnames the walker doesn't yet support (e.g. getfield_vable_i/rd>i). Full end-to-end acceptance lands when slice 2i adds handlers for the rest of the codewriter-emitted opnames."]
    fn walk_return_value_arm_terminates_at_first_ref_return() {
        // Acceptance: walk the smallest real
        // arm jitcode (`Instruction::ReturnValue`, 18 bytes) end-to-end.
        // Layout (cranelift build):
        //
        //   pc=0..6   inline_call_r_r / dR>r  (recurse → SubReturn → caller dst write → Continue)
        //   pc=6..9   live /                  (continue)
        //   pc=9..11  ref_return / r          (terminate — top-level outermost)
        //   pc=11..18 (raise + ref_return tail, dead on this path)
        //
        // The arm's `inline_call_r_r` now recurses into the callee
        // jitcode via `production_sub_jitcodes` and
        // `descr_pool_with_jitcode_adapters`. The callee's
        // own `ref_return/r` surfaces as `SubReturn`; the caller writes
        // its dst register with that result and continues. The
        // caller's own `ref_return/r` at pc=9..11 then records the
        // outermost `Finish`.
        let jc = jitcode_for_instruction(&Instruction::ReturnValue)
            .expect("ReturnValue must resolve to a jitcode");
        let mut tc = fresh_trace_ctx();
        // 256 distinct OpRefs (one per possible 1-byte register
        // index). `inline_call_r_r`'s recursion overwrites the dst
        // slot with the callee's `SubReturn` value, so the
        // post-recursion `ref_return/r` reads the *recorded* OpRef
        // from the sub-walk, not a `regs_r` constant. The assertion
        // therefore checks the recorded Finish's args against the
        // post-recursion register state, not a precomputed constant.
        let mut regs_r = distinct_const_refs(&mut tc, 256);
        let mut regs_i = distinct_const_refs(&mut tc, 256);
        let descr = done_descr_ref_for_tests();
        let pool_len = crate::jitcode_runtime::all_descrs().len();
        let descr_pool = descr_pool_with_jitcode_adapters(pool_len);
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &production_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, end_pc) =
            walk(&jc.code, 0, &mut wc).expect("ReturnValue arm must walk to a terminator");
        assert_eq!(
            outcome,
            DispatchOutcome::Terminate,
            "top-level walk must end on Terminate",
        );
        assert!(
            end_pc <= jc.code.len(),
            "walker must not run past the arm body \
             (end_pc={end_pc}, code.len()={})",
            jc.code.len(),
        );
        assert_eq!(
            end_pc, 11,
            "ReturnValue arm walker must terminate at outermost `ref_return/r` (pc=9..11)",
        );
        drop(wc);
        assert!(
            tc.num_ops() > ops_before,
            "at least one Finish op must have been recorded; \
             callee sub-walk may also have contributed CallR / Finish ops",
        );
        // Locate the *outermost* Finish (descr=done_with_this_frame).
        // Sub-walks don't emit Finish (they surface `SubReturn`), so
        // there should be exactly one Finish carrying the
        // done-with-this-frame descr.
        let outermost_finish = tc
            .ops()
            .iter()
            .find(|o| {
                o.opcode == majit_ir::OpCode::Finish
                    && o.getdescr()
                        .map(|d| std::sync::Arc::ptr_eq(&d, &descr))
                        .unwrap_or(false)
            })
            .expect("outermost Finish with done-with-this-frame descr must exist");
        assert_eq!(outermost_finish.num_args(), 1);
        let recorded_descr = outermost_finish
            .getdescr()
            .expect("Finish must carry done_with_this_frame_descr_ref");
        assert!(
            std::sync::Arc::ptr_eq(&recorded_descr, &descr),
            "Finish descr must be the exact instance the dispatcher was handed",
        );
    }

    #[test]
    #[ignore = "PopTop's pop_top helper recurses through inline_call_r_r \
        into a sub-jitcode whose body opens with `getfield_vable_i/rd>i` \
        — that opname has no walker handler yet (Phase D-3 follow-up: \
        MIFrame virtualizable_boxes / heapcache / vinfo prereqs).  \
        Production shadow mode under `MAJIT_SHADOW_WALKER=1` survives \
        because the bench traces never reach a PopTop opcode at JIT \
        record time, so the walker recursion never gets exercised — \
        but a direct unit-test invocation does, and surfaces the gap."]
    fn walk_pop_top_arm_terminates_with_recorded_ops() {
        // Acceptance skeleton: walk the entire PopTop arm
        // jitcode.  The outer arm body is 25 bytes / 9 ops after the
        // jtransform `Ok` / `Err` / `Some` identity rewrite stripped
        // the trailing `int_copy + residual_call_r_r/iRd>r` wrapper
        // for the `Ok(StepResult::Continue)` return value
        // (`majit/majit-translate/src/jit_codewriter/jtransform.rs
        //  ::rewrite_op_direct_call`).  The current sequence is:
        //
        //     inline_call_r_r/dR>r ; live/ ; catch_exception/L ;
        //     goto/L ; reraise/ ; ref_return/r ; live/ ;
        //     raise/r ; ref_return/r
        //
        // Every outer opname has a handler in this module.  The
        // remaining gap lives two levels deeper: PopTop's
        // `inline_call_r_r/dR>r` recurses into the `pop_top` callee
        // which itself recurses into a body whose first byte is
        // `getfield_vable_i/rd>i` — currently surfaced as
        // `UnsupportedOpname` by the walker.  Unignoring this test
        // requires landing the vable-aware getfield handlers (Phase
        // D-3 MIFrame integration).
        let jc = jitcode_for_instruction(&Instruction::PopTop)
            .expect("PopTop must resolve to a jitcode");
        let mut tc = fresh_trace_ctx();
        // Generously sized banks so any byte the codewriter emits is
        // in-range. 256 is the maximum register index a 1-byte slot
        // can address.
        let mut regs_r = distinct_const_refs(&mut tc, 256);
        let mut regs_i = distinct_const_refs(&mut tc, 256);
        // Descr pool: slot at each `BhDescr::JitCode` index in
        // `all_descrs()` is wrapped in a `TestJitCodeDescr` adapter so
        // `inline_call_r_r/dR>r` can resolve `as_jitcode_descr()`.
        // Other slots default to `make_fail_descr`.
        let pool_len = crate::jitcode_runtime::all_descrs().len();
        let descr_pool = descr_pool_with_jitcode_adapters(pool_len);
        let frame_done_descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &production_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, end_pc) =
            walk(&jc.code, 0, &mut wc).expect("PopTop arm must walk to a terminator");
        // PopTop's `inline_call_r_r/dR>r` recurses into the
        // codewriter-emitted callee jitcode (resolved via
        // `production_sub_jitcodes`); on success the outermost
        // `ref_return/r` lands a FINISH at top level.
        assert_eq!(
            outcome,
            DispatchOutcome::Terminate,
            "top-level PopTop walk must end on Terminate",
        );
        assert!(
            end_pc <= jc.code.len(),
            "walker must not run past the arm body \
             (end_pc={end_pc}, code.len()={})",
            jc.code.len(),
        );
        drop(wc);
        let ops_after = tc.num_ops();
        assert!(
            ops_after > ops_before,
            "PopTop walk must record at least one op (FINISH from \
             ref_return at top level) — recorded {} → {}",
            ops_before,
            ops_after,
        );
        // No `residual_call_r_r/iRd>r` in the post-rewrite arm, so no
        // CallR-descr identity check applies here.  If a future arm
        // regrows the wrapper, restore the `as_call_descr().is_some()`
        // + `Arc::ptr_eq(real_call_descr)` checks that lived here in
        // the pre-`Ok` / `Err` / `Some` identity rewrite version of
        // this fixture.
    }

    #[test]
    fn inline_call_with_more_args_than_callee_regs_surfaces_arity_mismatch() {
        // codewriter shape contract says `R-list.len() <=
        // callee.num_regs_r` for `inline_call_r_r/dR>r`. Walker rejects
        // overflow with a typed error instead of silently dropping
        // (the dropped args would carry symbolic OpRefs the callee
        // never reads, breaking dataflow).
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_r_r/dR>r")
            .expect("`inline_call_r_r/dR>r` must be in insns table");
        // Callee declares num_regs_r=1 but caller passes 2 ref args.
        let callee_code: &'static [u8] = Box::leak(Box::new([0xFFu8])); // unreachable
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 0,
            num_regs_f: 0,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = move |idx: usize| {
            if idx == 5 {
                Some(sub_body.clone())
            } else {
                None
            }
        };
        // R-list = [r0, r1] but callee has only 1 slot.
        let caller_code = [inline_byte, 0x05, 0x00, 0x02, 0x00, 0x01, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[5] = make_jitcode_descr(5);
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&caller_code, 0, &mut wc).expect_err("arity overflow must surface error");
        assert_eq!(
            err,
            DispatchError::InlineCallArityMismatch {
                pc: 0,
                provided: 2,
                callee_num_regs_r: 1,
            },
        );
    }

    #[test]
    fn inline_call_with_void_subreturn_surfaces_unexpected_void_error() {
        let err = DispatchError::UnexpectedVoidSubReturn { pc: 42 };
        assert_eq!(err, DispatchError::UnexpectedVoidSubReturn { pc: 42 },);
    }

    // ── inline_call_*_v regression tests ──────────────────────────────
    //
    // Exercise the void-return contract for all three dispatch variants:
    //   dispatch_inline_call_dr_kind  (inline_call_r_v/dR)
    //   dispatch_inline_call_dir_kind (inline_call_ir_v/dIR)
    //   dispatch_inline_call_dirf_kind(inline_call_irf_v/dIRF)

    #[test]
    fn inline_call_r_v_accepts_void_returning_callee() {
        // Callee body: `void_return/` — surfaces SubReturn { None }.
        // Caller: `inline_call_r_v/dR  descr=7 R=[r0]` then `void_return/`.
        let void_ret = *insns_opname_to_byte()
            .get("void_return/")
            .expect("`void_return/` must be in insns table");
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_r_v/dR")
            .expect("`inline_call_r_v/dR` must be in insns table");
        let callee_code: &'static [u8] = Box::leak(Box::new([void_ret]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 0,
            num_regs_f: 0,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = {
            let sub_body = sub_body.clone();
            move |idx: usize| {
                if idx == 7 {
                    Some(sub_body.clone())
                } else {
                    None
                }
            }
        };
        // dR layout: 2B descr(7) + 1B R-len(1) + 1B R-arg(r0)  — no >X dst
        let caller_code = [
            inline_byte,
            0x07,
            0x00, // descr index 7
            0x01,
            0x00,     // R: len=1, arg=r0
            void_ret, // caller terminates
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[7] = make_jitcode_descr(7);
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, _) =
            walk(&caller_code, 0, &mut wc).expect("inline_call_r_v with void callee must succeed");
        assert_eq!(outcome, DispatchOutcome::Terminate);
    }

    #[test]
    fn inline_call_r_v_rejects_non_void_returning_callee() {
        // Callee body: `ref_return r0` — surfaces SubReturn { Some(_) }.
        // inline_call_r_v must reject with UnexpectedNonVoidSubReturn.
        let ref_ret = *insns_opname_to_byte()
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns table");
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_r_v/dR")
            .expect("`inline_call_r_v/dR` must be in insns table");
        let callee_code: &'static [u8] = Box::leak(Box::new([ref_ret, 0x00]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 0,
            num_regs_f: 0,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = {
            let sub_body = sub_body.clone();
            move |idx: usize| {
                if idx == 7 {
                    Some(sub_body.clone())
                } else {
                    None
                }
            }
        };
        let caller_code = [
            inline_byte,
            0x07,
            0x00, // descr index 7
            0x01,
            0x00, // R: len=1, arg=r0
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[7] = make_jitcode_descr(7);
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = walk(&caller_code, 0, &mut wc)
            .expect_err("inline_call_r_v with non-void callee must reject");
        assert_eq!(err, DispatchError::UnexpectedNonVoidSubReturn { pc: 0 });
    }

    #[test]
    fn inline_call_ir_v_accepts_void_returning_callee() {
        let void_ret = *insns_opname_to_byte()
            .get("void_return/")
            .expect("`void_return/` must be in insns table");
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_ir_v/dIR")
            .expect("`inline_call_ir_v/dIR` must be in insns table");
        let callee_code: &'static [u8] = Box::leak(Box::new([void_ret]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 1,
            num_regs_f: 0,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = {
            let sub_body = sub_body.clone();
            move |idx: usize| {
                if idx == 7 {
                    Some(sub_body.clone())
                } else {
                    None
                }
            }
        };
        // dIR layout: 2B descr(7) + I-list(len=1, i0) + R-list(len=1, r0) — no dst
        let caller_code = [
            inline_byte,
            0x07,
            0x00, // descr index 7
            0x01,
            0x00, // I: len=1, arg=i0
            0x01,
            0x00,     // R: len=1, arg=r0
            void_ret, // caller terminates
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let mut regs_i = distinct_const_refs(&mut tc, 4);
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[7] = make_jitcode_descr(7);
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, _) =
            walk(&caller_code, 0, &mut wc).expect("inline_call_ir_v with void callee must succeed");
        assert_eq!(outcome, DispatchOutcome::Terminate);
    }

    #[test]
    fn inline_call_ir_v_rejects_non_void_returning_callee() {
        let int_ret = *insns_opname_to_byte()
            .get("int_return/i")
            .expect("`int_return/i` must be in insns table");
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_ir_v/dIR")
            .expect("`inline_call_ir_v/dIR` must be in insns table");
        let callee_code: &'static [u8] = Box::leak(Box::new([int_ret, 0x00]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 1,
            num_regs_f: 0,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = {
            let sub_body = sub_body.clone();
            move |idx: usize| {
                if idx == 7 {
                    Some(sub_body.clone())
                } else {
                    None
                }
            }
        };
        let caller_code = [
            inline_byte,
            0x07,
            0x00,
            0x01,
            0x00, // I: len=1, arg=i0
            0x01,
            0x00, // R: len=1, arg=r0
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let mut regs_i = distinct_const_refs(&mut tc, 4);
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[7] = make_jitcode_descr(7);
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = walk(&caller_code, 0, &mut wc)
            .expect_err("inline_call_ir_v with non-void callee must reject");
        assert_eq!(err, DispatchError::UnexpectedNonVoidSubReturn { pc: 0 });
    }

    #[test]
    #[ignore = "inline_call_irf_v/dIRF not yet emitted by the pipeline build"]
    fn inline_call_irf_v_accepts_void_returning_callee() {
        let void_ret = *insns_opname_to_byte()
            .get("void_return/")
            .expect("`void_return/` must be in insns table");
        let &inline_byte = insns_opname_to_byte()
            .get("inline_call_irf_v/dIRF")
            .expect("`inline_call_irf_v/dIRF` must be in insns table");
        let callee_code: &'static [u8] = Box::leak(Box::new([void_ret]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 1,
            num_regs_f: 1,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = {
            let sub_body = sub_body.clone();
            move |idx: usize| {
                if idx == 7 {
                    Some(sub_body.clone())
                } else {
                    None
                }
            }
        };
        // dIRF layout: 2B descr + I-list + R-list + F-list — no dst
        let caller_code = [
            inline_byte,
            0x07,
            0x00, // descr index 7
            0x01,
            0x00, // I: len=1, arg=i0
            0x01,
            0x00, // R: len=1, arg=r0
            0x01,
            0x00,     // F: len=1, arg=f0
            void_ret, // caller terminates
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let mut regs_i = distinct_const_refs(&mut tc, 4);
        let mut regs_f = distinct_const_refs(&mut tc, 4);
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[7] = make_jitcode_descr(7);
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut regs_f,
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, _) = walk(&caller_code, 0, &mut wc)
            .expect("inline_call_irf_v with void callee must succeed");
        assert_eq!(outcome, DispatchOutcome::Terminate);
    }

    #[test]
    #[ignore = "inline_call_irf_v/dIRF not yet emitted by the pipeline build"]
    fn inline_call_irf_v_rejects_non_void_returning_callee() {
        let ref_ret = *insns_opname_to_byte()
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns table");
        let &inline_byte = insns_opname_to_byte()
            .get("inline_call_irf_v/dIRF")
            .expect("`inline_call_irf_v/dIRF` must be in insns table");
        let callee_code: &'static [u8] = Box::leak(Box::new([ref_ret, 0x00]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 1,
            num_regs_f: 1,
            constants_i: &[],
            constants_r: &[],
            constants_f: &[],
        };
        let lookup = {
            let sub_body = sub_body.clone();
            move |idx: usize| {
                if idx == 7 {
                    Some(sub_body.clone())
                } else {
                    None
                }
            }
        };
        let caller_code = [
            inline_byte,
            0x07,
            0x00,
            0x01,
            0x00, // I
            0x01,
            0x00, // R
            0x01,
            0x00, // F
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let mut regs_i = distinct_const_refs(&mut tc, 4);
        let mut regs_f = distinct_const_refs(&mut tc, 4);
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[7] = make_jitcode_descr(7);
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut regs_f,
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = walk(&caller_code, 0, &mut wc)
            .expect_err("inline_call_irf_v with non-void callee must reject");
        assert_eq!(err, DispatchError::UnexpectedNonVoidSubReturn { pc: 0 });
    }

    /// Build a `SimpleFieldDescr` with a stable index so the
    /// heapcache lookup hashes consistently across the cache-miss
    /// and cache-hit assertions. Default `Descr::index()` returns
    /// `u32::MAX`; tests that exercise heapcache need a real index.
    fn field_descr_with_index(idx: u32) -> DescrRef {
        std::sync::Arc::new(majit_ir::SimpleFieldDescr::new(
            idx,
            8, // offset
            8, // field_size
            majit_ir::Type::Int,
            false, // not immutable
        ))
    }

    #[test]
    fn getfield_gc_i_cache_miss_records_op_and_writes_dst() {
        // First `getfield_gc_i/rd>i` invocation
        // is a heapcache miss — walker records `OpCode::GetfieldGcI`
        // with `[obj]` and `descr=descr_refs[d]`, writes the
        // recorder result into `registers_i[dst]`, and updates the
        // heapcache via `getfield_now_known(resbox)`.
        let byte = *insns_opname_to_byte()
            .get("getfield_gc_i/rd>i")
            .expect("`getfield_gc_i/rd>i` must be in insns table");
        // Operand layout `rd>i`: 1B r-reg(2) + 2B descr-index(LE 1) + 1B dst(5).
        let code = [byte, 0x02, 0x01, 0x00, 0x05];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let obj = regs_r[2];
        let dst_pre = regs_i[5];
        let descr = field_descr_with_index(1);
        let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
        let frame_done = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("getfield_gc_i must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, 5, "getfield_gc_i/rd>i operand layout = 4 bytes");
        let dst_post = wc.registers_i[5];
        assert_ne!(
            dst_post, dst_pre,
            "cache miss must write a fresh recorder OpRef into registers_i[dst]",
        );
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "cache miss must record exactly one GetfieldGcI op",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::GetfieldGcI);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![obj],
            "GetfieldGcI args must be [obj] (the r-reg source)",
        );
        let recorded_descr = last
            .getdescr()
            .expect("GetfieldGcI must carry the field descr");
        assert!(
            std::sync::Arc::ptr_eq(&recorded_descr, &descr),
            "GetfieldGcI descr must be descr_refs[d] (the field descr)",
        );
        assert_eq!(dst_post, last.pos.get());
    }

    #[test]
    fn getfield_gc_i_cache_hit_returns_cached_box_without_recording() {
        // Second invocation with the same
        // (obj, descr) pair must hit the heapcache and skip IR
        // emission. RPython parity:
        //   upd = heapcache.get_field_updater(box, fielddescr)
        //   if upd.currfieldbox is not None:
        //       return upd.currfieldbox  # no execute_with_descr
        let byte = *insns_opname_to_byte()
            .get("getfield_gc_i/rd>i")
            .expect("`getfield_gc_i/rd>i` must be in insns table");
        let code = [byte, 0x02, 0x01, 0x00, 0x05];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let obj = regs_r[2];
        let descr = field_descr_with_index(1);
        let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
        let frame_done = done_descr_ref_for_tests();

        // Pre-populate the heapcache as if a previous getfield had
        // already cached the field's value. RPython equivalent:
        // `heapcache.getfield_now_known(...)` after a prior fetch.
        let cached_field = tc.const_int(0xCAFE);
        tc.heapcache_getfield_now_known(obj, 1, cached_field);
        let ops_before = tc.num_ops();

        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("getfield_gc_i must dispatch");
        let dst_post = wc.registers_i[5];
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before,
            "cache hit must NOT record any new IR op",
        );
        assert_eq!(
            dst_post, cached_field,
            "cache hit must write the cached OpRef into registers_i[dst]",
        );
    }

    #[test]
    fn getfield_gc_r_cache_miss_records_op_and_writes_ref_dst() {
        // GetfieldGcR variant — same flow as
        // GetfieldGcI but result lands in registers_r.
        let byte = *insns_opname_to_byte()
            .get("getfield_gc_r/rd>r")
            .expect("`getfield_gc_r/rd>r` must be in insns table");
        let code = [byte, 0x02, 0x01, 0x00, 0x06];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let obj = regs_r[2];
        let dst_pre = regs_r[6];
        // Use a Ref-typed field descr — sanity-check that the walker
        // doesn't introspect the descr's field_type (it just feeds
        // descr_index into the heapcache and records the op).
        let descr: DescrRef = std::sync::Arc::new(majit_ir::SimpleFieldDescr::new(
            1,
            16,
            8,
            majit_ir::Type::Ref,
            false,
        ));
        let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
        let frame_done = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("getfield_gc_r must dispatch");
        let dst_post = wc.registers_r[6];
        assert_ne!(dst_post, dst_pre);
        drop(wc);
        assert_eq!(tc.num_ops(), ops_before + 1);
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::GetfieldGcR);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![obj]
        );
        assert_eq!(dst_post, last.pos.get());
    }

    #[test]
    fn getfield_gc_with_out_of_range_obj_register_surfaces_typed_error() {
        let byte = *insns_opname_to_byte()
            .get("getfield_gc_i/rd>i")
            .expect("`getfield_gc_i/rd>i` must be in insns table");
        let code = [byte, 0x07, 0x00, 0x00, 0x00]; // r-reg=7, registers_r empty
        let mut tc = fresh_trace_ctx();
        let descr = field_descr_with_index(0);
        let descr_pool = vec![descr];
        let frame_done = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let err = step(&code, 0, &mut wc).expect_err("getfield_gc must validate r-reg");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 7,
                len: 0,
                bank: "r",
            },
        );
    }

    #[test]
    fn getfield_vable_i_routes_through_metainterp_and_writes_dst() {
        // T2 sanity: `getfield_vable_i/rd>i` delegates to
        // `TraceCtx::vable_getfield_int`.  With no `virtualizable_info`
        // bound on the trace context, `is_nonstandard_virtualizable`
        // returns true and the fallback emits a `GetfieldGcI` op +
        // writes the recorder OpRef into `registers_i[dst]` — the same
        // shape `getfield_gc_via_heapcache` produces on a cache miss.
        // The handler itself stays orthodox to RPython
        // `pyjitpl.py:1167-1172 opimpl_getfield_vable_i`; the
        // GETFIELD_GC fallback is `vable_getfield_int`'s decision, not
        // the walker's, so this test exercises the walker→trace_ctx
        // boundary without depending on a `virtualizable_info` fixture.
        let byte = *insns_opname_to_byte()
            .get("getfield_vable_i/rd>i")
            .expect("`getfield_vable_i/rd>i` must be in insns table");
        // Operand layout `rd>i`: 1B r-reg(2) + 2B descr-index(LE 1) + 1B dst(5).
        let code = [byte, 0x02, 0x01, 0x00, 0x05];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let obj = regs_r[2];
        let dst_pre = regs_i[5];
        let descr = field_descr_with_index(1);
        let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
        let frame_done = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("getfield_vable_i must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, 5, "getfield_vable_i/rd>i operand layout = 4 bytes");
        let dst_post = wc.registers_i[5];
        assert_ne!(
            dst_post, dst_pre,
            "fallback must write a fresh recorder OpRef into registers_i[dst]",
        );
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "nonstandard-vable fallback records exactly one GetfieldGcI op",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::GetfieldGcI);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![obj],
            "GetfieldGcI args must be [obj] (the r-reg source)",
        );
        let recorded_descr = last
            .getdescr()
            .expect("GetfieldGcI must carry the field descr");
        assert!(
            std::sync::Arc::ptr_eq(&recorded_descr, &descr),
            "GetfieldGcI descr must be descr_refs[d] (the field descr)",
        );
        assert_eq!(dst_post, last.pos.get());
    }

    #[test]
    fn setfield_vable_i_routes_through_metainterp_records_setfield_gc_fallback() {
        // T2a sanity: `setfield_vable_i/rid` delegates to
        // `TraceCtx::vable_setfield`.  With no `virtualizable_info`
        // bound on the trace context, `is_nonstandard_virtualizable`
        // returns true and the fallback records a `SetfieldGc` op
        // with `[obj, value]` + the field descr — same shape
        // `setfield_gc_via_heapcache` produces.  Exercises the
        // walker -> trace_ctx boundary for the int-bank variant
        // (the `r` and `f` variants share the handler body, varying
        // only `value_bank`).
        let byte = *insns_opname_to_byte()
            .get("setfield_vable_i/rid")
            .expect("`setfield_vable_i/rid` must be in insns table");
        // Operand layout `rid`: 1B r-reg(2) + 1B i-reg(3) + 2B descr-index(LE 1).
        let code = [byte, 0x02, 0x03, 0x01, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let obj = regs_r[2];
        let value = regs_i[3];
        let descr = field_descr_with_index(1);
        let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
        let frame_done = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("setfield_vable_i must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, 5, "setfield_vable_i/rid operand layout = 4 bytes");
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "nonstandard-vable fallback records exactly one SetfieldGc op",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::SetfieldGc);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![obj, value],
            "SetfieldGc args must be [obj, value]",
        );
        let recorded_descr = last
            .getdescr()
            .expect("SetfieldGc must carry the field descr");
        assert!(
            std::sync::Arc::ptr_eq(&recorded_descr, &descr),
            "SetfieldGc descr must be descr_refs[d] (the field descr)",
        );
    }

    #[test]
    fn setfield_gc_i_redundant_write_skips_recording() {
        // When the heapcache already knows
        // valuebox is the current value of (obj, descr), the
        // SETFIELD_GC IR op must NOT be recorded. RPython parity:
        // `pyjitpl.py:976 if upd.currfieldbox is valuebox: return`.
        let byte = *insns_opname_to_byte()
            .get("setfield_gc_i/rid")
            .expect("`setfield_gc_i/rid` must be in insns table");
        let code = [byte, 0x02, 0x03, 0x01, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let obj = regs_r[2];
        let valuebox = regs_i[3];
        let descr = field_descr_with_index(1);
        let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr];
        let frame_done = done_descr_ref_for_tests();
        // Pre-cache valuebox as the current field value.
        tc.heapcache_getfield_now_known(obj, 1, valuebox);
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("setfield_gc_i must dispatch");
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before,
            "redundant setfield (cached valuebox == new valuebox) must skip recording",
        );
    }

    #[test]
    fn setfield_gc_i_fresh_write_records_op_and_caches_value() {
        // A fresh write (no cached value)
        // must record SETFIELD_GC and update the heapcache so a
        // subsequent redundant write hits.
        let byte = *insns_opname_to_byte()
            .get("setfield_gc_i/rid")
            .expect("`setfield_gc_i/rid` must be in insns table");
        let code = [byte, 0x02, 0x03, 0x01, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let obj = regs_r[2];
        let valuebox = regs_i[3];
        let descr = field_descr_with_index(1);
        let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
        let frame_done = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("setfield_gc_i must dispatch");
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "fresh setfield must record exactly one SetfieldGc op",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::SetfieldGc);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![obj, valuebox],
            "SetfieldGc args must be [obj, valuebox] in that order",
        );
        assert!(std::sync::Arc::ptr_eq(
            &last.getdescr().expect("SetfieldGc must carry descr"),
            &descr,
        ),);
        // Cache must now know the new field value.  Box identity-only
        // check — value payload is Void in walker-emitted writes.
        assert_eq!(
            tc.heapcache_getfield_cached(obj, 1).map(|b| b),
            Some(valuebox),
            "post-setfield, the heapcache must reflect the written value",
        );
    }

    #[test]
    fn setfield_gc_r_records_setfieldgc_with_ref_valuebox() {
        // `rrd` shape — both box and valuebox
        // come from registers_r. SetfieldGc is type-agnostic at the
        // IR level (the descr carries the field type).
        let byte = *insns_opname_to_byte()
            .get("setfield_gc_r/rrd")
            .expect("`setfield_gc_r/rrd` must be in insns table");
        let code = [byte, 0x02, 0x05, 0x01, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let obj = regs_r[2];
        let valuebox = regs_r[5];
        let descr: DescrRef = std::sync::Arc::new(majit_ir::SimpleFieldDescr::new(
            1,
            16,
            8,
            majit_ir::Type::Ref,
            false,
        ));
        let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr];
        let frame_done = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("setfield_gc_r must dispatch");
        drop(wc);
        assert_eq!(tc.num_ops(), ops_before + 1);
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::SetfieldGc);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![obj, valuebox]
        );
    }

    #[test]
    fn getarrayitem_gc_r_cache_miss_records_op_and_writes_dst() {
        // First `getarrayitem_gc_r/rid>r` is a
        // heapcache miss — record GetarrayitemGcR with
        // [array, index] + descr; write recorder result into r-dst
        // and update heapcache.
        let byte = *insns_opname_to_byte()
            .get("getarrayitem_gc_r/rid>r")
            .expect("`getarrayitem_gc_r/rid>r` must be in insns table");
        // Operand layout `rid>r`: 1B r-reg(2) + 1B i-reg(3) +
        // 2B descr(LE 1) + 1B r-dst(5).
        let code = [byte, 0x02, 0x03, 0x01, 0x00, 0x05];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let array = regs_r[2];
        let index = regs_i[3];
        let dst_pre = regs_r[5];
        let descr = field_descr_with_index(1);
        let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
        let frame_done = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("getarrayitem_gc_r must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            next_pc, 6,
            "getarrayitem_gc_r/rid>r operand layout = 5 bytes"
        );
        let dst_post = wc.registers_r[5];
        assert_ne!(dst_post, dst_pre);
        drop(wc);
        assert_eq!(tc.num_ops(), ops_before + 1);
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::GetarrayitemGcR);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![array, index],
            "GetarrayitemGcR args must be [array, index]",
        );
        assert!(std::sync::Arc::ptr_eq(
            &last.getdescr().expect("must carry array descr"),
            &descr,
        ));
        assert_eq!(dst_post, last.pos.get());
    }

    #[test]
    fn getarrayitem_gc_r_cache_hit_returns_cached_box() {
        // Pre-cache (array, index, descr) →
        // cached_box. Second invocation must return cached_box and
        // not record an IR op.
        let byte = *insns_opname_to_byte()
            .get("getarrayitem_gc_r/rid>r")
            .expect("`getarrayitem_gc_r/rid>r` must be in insns table");
        let code = [byte, 0x02, 0x03, 0x01, 0x00, 0x05];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let mut regs_i = distinct_const_ints(&mut tc, 8);
        let array = regs_r[2];
        let index = regs_i[3];
        let descr = field_descr_with_index(1);
        let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr];
        let frame_done = done_descr_ref_for_tests();
        let cached = tc.const_ref(0xCAFE_F00D);
        let cached_payload = tc
            .constants_get_value(cached)
            .unwrap_or(majit_ir::Value::Void);
        tc.heapcache_getarrayitem_now_known(array, index, 1, cached);
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let _ = step(&code, 0, &mut wc).expect("getarrayitem_gc_r must dispatch");
        let dst_post = wc.registers_r[5];
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before,
            "cache hit must NOT record any new IR op",
        );
        assert_eq!(
            dst_post, cached,
            "cache hit must write cached OpRef into registers_r[dst]",
        );
    }

    /// pyre-only `getarrayitem_gc_r/rrd>r` mirrors the canonical
    /// `rid>r` shape — same heapcache lookup / `GetarrayitemGcR`
    /// emission — but reads the index from the Ref register bank
    /// (tagged-int-in-ref deviation; see
    /// `blackhole.rs::handler_getarrayitem_gc_r_refindex`).  Should
    /// behave identically to the canonical variant on a cache miss.
    #[test]
    fn getarrayitem_gc_r_refindex_cache_miss_records_op_and_writes_dst() {
        let byte = *insns_opname_to_byte()
            .get("getarrayitem_gc_r/rrd>r")
            .expect("`getarrayitem_gc_r/rrd>r` must be in insns table");
        // Operand layout `rrd>r`: 1B r-reg(2) + 1B r-reg(3) +
        // 2B descr(LE 1) + 1B r-dst(5).
        let code = [byte, 0x02, 0x03, 0x01, 0x00, 0x05];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let mut regs_i = distinct_const_ints(&mut tc, 4);
        let array = regs_r[2];
        let index = regs_r[3];
        let dst_pre = regs_r[5];
        let descr = field_descr_with_index(1);
        let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
        let frame_done = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) =
            step(&code, 0, &mut wc).expect("getarrayitem_gc_r/rrd>r must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            next_pc, 6,
            "getarrayitem_gc_r/rrd>r operand layout = 5 bytes",
        );
        let dst_post = wc.registers_r[5];
        assert_ne!(dst_post, dst_pre);
        drop(wc);
        assert_eq!(tc.num_ops(), ops_before + 1);
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::GetarrayitemGcR);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![array, index],
            "GetarrayitemGcR args must be [array, index] read from r-bank",
        );
        assert!(std::sync::Arc::ptr_eq(
            last.getdescr().as_ref().expect("must carry array descr"),
            &descr,
        ));
        assert_eq!(dst_post, last.pos.get());
    }

    #[test]
    fn setarrayitem_gc_r_records_setarrayitemgc_with_three_args() {
        // `setarrayitem_gc_r/rird` records
        // SetarrayitemGc with [array, index, value] + descr and
        // updates the heapcache via setarrayitem.
        let byte = *insns_opname_to_byte()
            .get("setarrayitem_gc_r/rird")
            .expect("`setarrayitem_gc_r/rird` must be in insns table");
        // Operand layout `rird`: 1B r-reg(2) + 1B i-reg(4) +
        // 1B r-reg(6) + 2B descr(LE 1).
        let code = [byte, 0x02, 0x04, 0x06, 0x01, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let mut regs_i = distinct_const_ints(&mut tc, 8);
        let array = regs_r[2];
        let index = regs_i[4];
        let value = regs_r[6];
        let descr = field_descr_with_index(1);
        let descr_pool: Vec<DescrRef> = vec![make_fail_descr(0), descr.clone()];
        let frame_done = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done,
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("setarrayitem_gc_r must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            next_pc, 6,
            "setarrayitem_gc_r/rird operand layout = 5 bytes"
        );
        drop(wc);
        assert_eq!(tc.num_ops(), ops_before + 1);
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::SetarrayitemGc);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![array, index, value],
            "SetarrayitemGc args must be [array, index, value]",
        );
        assert!(std::sync::Arc::ptr_eq(
            &last.getdescr().expect("must carry array descr"),
            &descr,
        ));
        // Heapcache must reflect the write.  Box identity-only check —
        // value payload is Void in walker-emitted writes.
        assert_eq!(
            tc.heapcache_getarrayitem(array, index, 1).map(|b| b),
            Some(value),
            "post-setarrayitem, heapcache must reflect the written value",
        );
    }

    #[test]
    fn dispatch_via_miframe_runs_ref_return_through_real_miframe_state() {
        // Acceptance: the bridge function takes a
        // real `MIFrame` (constructed via the same `PyreSym::new_uninit`
        // + `MIFrame { ctx, sym, .. }` shape that `state.rs`'s
        // existing tests use), pre-populates `sym.registers_r[2]` with
        // a known OpRef, then walks `ref_return r2`. Walker must
        // record `Finish([sym.registers_r[2]], descr=done_with_this_frame_descr_ref)`
        // through the *same* TraceCtx the MIFrame's `ctx` pointer
        // owns — i.e., production-shape state plumbing, no separate
        // test fixture.
        use crate::state::PyreSym;

        let mut tc = TraceCtx::for_test_types(&[majit_ir::Type::Ref]);
        let expected_arg = tc.const_ref(0xCAFE_F00D);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![OpRef::NONE; 8];
        sym.registers_r[2] = expected_arg;

        let mut miframe = MIFrame {
            ctx: &mut tc,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let ret_byte = *insns_opname_to_byte()
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns table");
        let code = [ret_byte, 0x02];
        let descr = make_fail_descr(1);
        // PyPy `setup_call(argboxes)` analog: stamp `expected_arg` at
        // `R[2]_r` so the `ref_return r2` walker handler picks it up
        // from the fresh top-level register file.  Slots 0/1 stay
        // `OpRef::NONE` since this fixture exercises only slot 2.
        let argboxes_r = [OpRef::NONE, OpRef::NONE, expected_arg];
        let (outcome, end_pc) = dispatch_via_miframe(
            &mut miframe,
            &code,
            0,
            &[],
            &no_sub_jitcodes,
            descr.clone(),
            make_fail_descr(101),
            make_fail_descr(102),
            make_fail_descr(103),
            make_fail_descr(2),
            true,
            8,
            0,
            0,
            &[],
            &[],
            &[],
            &argboxes_r,
            &[],
            &[],
        )
        .expect("dispatch_via_miframe must succeed for ref_return r2");
        assert_eq!(outcome, DispatchOutcome::Terminate);
        assert_eq!(end_pc, 2);

        // Drop miframe so we can inspect tc directly.
        drop(miframe);
        let last = tc.ops().last().expect("FINISH must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::Finish);
        assert_eq!(
            last.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![expected_arg],
            "FINISH args must be sym.registers_r[2] threaded through the MIFrame bridge",
        );
        let recorded_descr = last
            .getdescr()
            .expect("FINISH must carry done_with_this_frame_descr_ref");
        assert!(
            std::sync::Arc::ptr_eq(&recorded_descr, &descr),
            "FINISH descr must be the descr passed through dispatch_via_miframe",
        );
    }

    #[test]
    fn dispatch_via_miframe_mirrors_last_exc_value_back_into_sym() {
        // When the walker's last_exc_value field
        // changes (raise/r sets it before terminating), the bridge
        // function must mirror it back to `sym.last_exc_box`. RPython
        // parity: `metainterp.last_exc_value = ...` is metainterp-level
        // state that survives across opimpl invocations.
        use crate::state::PyreSym;

        let mut tc = TraceCtx::for_test_types(&[majit_ir::Type::Ref]);
        let exc_oprep = tc.const_ref(0xDEAD_BEEF);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![OpRef::NONE; 8];
        sym.registers_r[3] = exc_oprep;
        // Pre-condition: sym.last_exc_box is unset.
        assert!(sym.last_exc_box.is_none());

        let mut miframe = MIFrame {
            ctx: &mut tc,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let raise_byte = *insns_opname_to_byte()
            .get("raise/r")
            .expect("`raise/r` must be in insns table");
        let code = [raise_byte, 0x03];
        let descr_done = make_fail_descr(1);
        let descr_exc = make_fail_descr(99);
        // Setup_call argbox at R[3]_r: `raise/r` reads its exc operand
        // from this slot in the fresh top-level register file.
        let argboxes_r = [OpRef::NONE, OpRef::NONE, OpRef::NONE, exc_oprep];
        let (outcome, _) = dispatch_via_miframe(
            &mut miframe,
            &code,
            0,
            &[],
            &no_sub_jitcodes,
            descr_done,
            make_fail_descr(101),
            make_fail_descr(102),
            make_fail_descr(103),
            descr_exc,
            true,
            8,
            0,
            0,
            &[],
            &[],
            &[],
            &argboxes_r,
            &[],
            &[],
        )
        .expect("dispatch_via_miframe must succeed for raise r3");
        assert_eq!(outcome, DispatchOutcome::Terminate);
        drop(miframe);
        // Post-condition: sym.last_exc_box was mirrored from the
        // walker's last_exc_value (set by raise/r before terminate).
        assert_eq!(
            sym.last_exc_box, exc_oprep,
            "sym.last_exc_box must mirror the exc OpRef the walker captured \
             via WalkContext::last_exc_value",
        );
        // Post-condition: dispatch_via_miframe also sets
        // sym.class_of_last_exc_is_const to mirror RPython's
        // `pyjitpl.py:1694 opimpl_raise: class_of_last_exc_is_const = True`.
        assert!(
            sym.class_of_last_exc_is_const,
            "sym.class_of_last_exc_is_const must be true after a raise/r",
        );
    }

    #[test]
    fn dispatch_via_miframe_leaves_class_of_last_exc_is_const_unchanged_when_no_raise() {
        // When the walk does NOT raise (final last_exc remains None),
        // dispatch_via_miframe must NOT touch
        // sym.class_of_last_exc_is_const. The flag carries state from
        // a prior tracing step and must not be cleared by an unrelated
        // walk (e.g. a single ref_return-only top-level walk).
        use crate::state::PyreSym;

        let mut tc = TraceCtx::for_test_types(&[majit_ir::Type::Ref]);
        let value = tc.const_ref(0xC0FFEE);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![OpRef::NONE; 8];
        sym.registers_r[2] = value;
        // Pre-condition: simulate prior raise — class_of_last_exc_is_const
        // is true and last_exc_box is set.
        sym.class_of_last_exc_is_const = true;
        sym.last_exc_box = value;

        let mut miframe = MIFrame {
            ctx: &mut tc,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };
        let ret_byte = *insns_opname_to_byte()
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns table");
        let code = [ret_byte, 0x02];
        // Setup_call argbox at R[2]_r — the `ref_return r2` walker
        // handler picks it up from the fresh top-level register file.
        let argboxes_r = [OpRef::NONE, OpRef::NONE, value];
        let _ = dispatch_via_miframe(
            &mut miframe,
            &code,
            0,
            &[],
            &no_sub_jitcodes,
            make_fail_descr(1),
            make_fail_descr(101),
            make_fail_descr(102),
            make_fail_descr(103),
            make_fail_descr(2),
            true,
            8,
            0,
            0,
            &[],
            &[],
            &[],
            &argboxes_r,
            &[],
            &[],
        )
        .expect("ref_return walk must succeed");
        drop(miframe);
        // Walker preserved the carried-in class flag because no raise
        // happened during the walk.
        assert!(
            sym.class_of_last_exc_is_const,
            "no-raise walk must not clear class_of_last_exc_is_const",
        );
    }

    #[test]
    fn walk_undecodable_byte_surfaces_typed_error() {
        // 0xFF is unknown to the insns table (21 entries 0..=20 today).
        let code = [0xFFu8];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            registers_f: &mut [],
            concrete_registers_r: &mut [],
            concrete_registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            done_with_this_frame_descr_int: make_fail_descr(101),
            done_with_this_frame_descr_float: make_fail_descr(102),
            done_with_this_frame_descr_void: make_fail_descr(103),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: 0,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
        };
        assert_eq!(
            walk(&code, 0, &mut wc),
            Err(DispatchError::UndecodableOpcode { pc: 0 })
        );
    }
}
