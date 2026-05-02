//! `rpython/jit/metainterp/memmgr.py` parity.
//!
//! `MemoryManager` is the **sole long-living strong reference** to
//! compiled `JitCellToken` objects (`memmgr.py:9-12`):
//!
//! > All the long-lived references to LoopToken are weakrefs (see
//! > JitCell in warmstate.py), apart from the 'alive_loops' set in
//! > MemoryManager, which is the only (long-living) place that keeps
//! > them alive. If a loop was not called for long enough, then it is
//! > removed from 'alive_loops'. It will soon be freed by the GC.
//! > `LoopToken.__del__` calls the method `cpu.free_loop_and_bridges()`.
//!
//! In pyre this is achieved by `alive_loops` storing
//! `Arc<JitCellToken>`. When an entry is removed, the Arc drops; when
//! the last strong reference goes, `JitCellToken::drop` (TBD —
//! `model.py:289 cpu.free_loop_and_bridges` parity) frees the backend
//! resources.
//!
//! Pyre currently has additional strong references in `BaseJitCell.
//! loop_token` and embedded `JitCellToken` values inside
//! `pyjitpl::compiled_loops`. Achieving "alive_loops is the SOLE
//! strong owner" requires the slicing chain in
//! `.claude/plans/memmgr-line-by-line-parity.md` (Slices 3.5–3.6),
//! whose tail is blocked on Task #195 (Arc lift of CompiledEntry.token).
//! Today `alive_loops` is a *third* strong owner; eviction tracking
//! works for the warmstate side, but actual token frees only happen
//! once Slices 3.5–3.6 prune the other strong owners.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use majit_backend::JitCellToken;

/// `memmgr.py:23` `class MemoryManager`. Pyre also pins the
/// retrace/unroll parameters here, mirroring RPython's lazy attribute
/// writes via `warmstate.py:299-320 set_param_*`. RPython treats them
/// as Python `int` attributes; pyre declares them as typed fields and
/// initializes them to the `rlib/jit.py:588 PARAMETERS` defaults.
pub struct MemoryManager {
    /// `memmgr.py:38` `self.current_generation = r_int64(1)`.
    pub current_generation: i64,
    /// `memmgr.py:39` `self.max_age = max_age` — set by
    /// `set_max_age` (`memmgr.py:42-50`).  `<= 0` disables eviction.
    pub max_age: i64,
    /// `memmgr.py:39` `self.next_check = r_int64(-1)`.  Generation
    /// at which `_kill_old_loops_now` next fires; `-1` means
    /// "eviction disabled" (`memmgr.py:43-44`).
    pub next_check: i64,
    /// `memmgr.py:26` `self.check_frequency = -1`.  Number of
    /// generations between successive `_kill_old_loops_now` sweeps.
    /// `-1` is "uninitialized"; `set_max_age` derives a real value
    /// (`int(sqrt(max_age))` by default per `memmgr.py:47-48`).
    pub check_frequency: i64,

    /// `memmgr.py:40` `self.alive_loops = {}` — a dict keyed on the
    /// looptoken object itself.  In Rust the dict key uses the Arc's
    /// pointer-id (stable while the Arc lives) and the value IS the
    /// strong Arc reference: removing the entry drops the Arc, and
    /// dropping the last strong reference fires `JitCellToken::drop`
    /// (mirrors `LoopToken.__del__` calling `cpu.free_loop_and_bridges`
    /// at `memmgr.py:13-14`).
    ///
    /// **Pointer-key soundness:** `*const JitCellToken` is used **only
    /// as HashMap key**.  The Arc value held alongside guarantees the
    /// pointee is alive for the lifetime of the entry, so pointer
    /// identity is stable until removal.
    pub alive_loops: HashMap<*const JitCellToken, Arc<JitCellToken>>,

    /// `warmstate.py:299-302` `set_param_retrace_limit` writes here.
    /// `unroll.py:215` reader.
    pub retrace_limit: u32,
    /// `warmstate.py:307-310` `set_param_max_retrace_guards`.
    /// `unroll.py:265` reader.
    pub max_retrace_guards: u32,
    /// `warmstate.py:312-315` `set_param_max_unroll_loops`.
    /// `pyjitpl.py:2946` reader.
    pub max_unroll_loops: u32,
    /// `warmstate.py:317-320` `set_param_max_unroll_recursion`.
    /// `pyjitpl.py:1404` reader.
    pub max_unroll_recursion: u32,
}

impl MemoryManager {
    /// `memmgr.py:25-40` `MemoryManager.__init__`. Note RPython splits
    /// init from `set_max_age`; pyre takes `max_age` upfront for
    /// ergonomics — `set_max_age` later overwrites it just like the
    /// upstream call sequence at `warmspot.py:118` /
    /// `set_user_param('loop_longevity=...')`.
    pub fn new(max_age: i64) -> Self {
        let mut mgr = MemoryManager {
            // memmgr.py:38 current_generation = r_int64(1)
            current_generation: 1,
            // memmgr.py:39 next_check = r_int64(-1)
            next_check: -1,
            // memmgr.py:26 check_frequency = -1
            check_frequency: -1,
            max_age: 0,
            alive_loops: HashMap::new(),
            // rlib/jit.py:588 PARAMETERS defaults.
            retrace_limit: 0,
            max_retrace_guards: 15,
            max_unroll_loops: 0,
            max_unroll_recursion: 7,
        };
        // memmgr.py:42-50 set_max_age — derives next_check / check_frequency.
        mgr.set_max_age(max_age, 0);
        mgr
    }

    /// `memmgr.py:42-50` `set_max_age(max_age, check_frequency=0)`.
    ///
    /// ```python
    /// def set_max_age(self, max_age, check_frequency=0):
    ///     if max_age <= 0:
    ///         self.next_check = r_int64(-1)
    ///     else:
    ///         self.max_age = max_age
    ///         if check_frequency <= 0:
    ///             check_frequency = int(math.sqrt(max_age))
    ///         self.check_frequency = check_frequency
    ///         self.next_check = self.current_generation + 1
    /// ```
    pub fn set_max_age(&mut self, max_age: i64, check_frequency: i64) {
        if max_age <= 0 {
            self.next_check = -1;
        } else {
            self.max_age = max_age;
            let cf = if check_frequency <= 0 {
                (max_age as f64).sqrt() as i64
            } else {
                check_frequency
            };
            self.check_frequency = cf;
            self.next_check = self.current_generation + 1;
        }
    }

    /// Read accessor.  Used by `set_user_param('loop_longevity', ...)`
    /// readback in `warmstate.rs`.
    pub fn max_age(&self) -> i64 {
        self.max_age
    }

    /// Pyre-only readback for `get_param("loop_longevity")`.
    ///
    /// `set_max_age(<= 0)` matches `memmgr.py:43 self.next_check = -1`
    /// without touching `max_age`, so the raw `max_age` field stays at
    /// the previous positive value. RPython has no `get_param` for
    /// `loop_longevity`, so the field staleness is invisible upstream.
    /// Pyre adds the readback, so report 0 when eviction is disabled
    /// (`next_check == -1`) — matching the user-facing semantics of
    /// `set_user_param('loop_longevity=0')`.
    pub fn loop_longevity_param(&self) -> i64 {
        if self.next_check == -1 {
            0
        } else {
            self.max_age
        }
    }

    /// `memmgr.py:58-61` `keep_loop_alive(looptoken)`.
    ///
    /// ```python
    /// def keep_loop_alive(self, looptoken):
    ///     if looptoken.generation != self.current_generation:
    ///         looptoken.generation = self.current_generation
    ///         self.alive_loops[looptoken] = None
    /// ```
    pub fn keep_loop_alive(&mut self, looptoken: &Arc<JitCellToken>) {
        if looptoken.generation.get() != self.current_generation {
            looptoken.generation.set(self.current_generation);
            let key: *const JitCellToken = Arc::as_ptr(looptoken);
            self.alive_loops
                .entry(key)
                .or_insert_with(|| Arc::clone(looptoken));
        }
    }

    /// `memmgr.py:52-56` `next_generation`.
    ///
    /// ```python
    /// def next_generation(self):
    ///     self.current_generation += 1
    ///     if self.current_generation == self.next_check:
    ///         self._kill_old_loops_now()
    ///         self.next_check = self.current_generation + self.check_frequency
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: returns `Vec<Arc<JitCellToken>>` of the
    /// evicted token objects.  Upstream returns `None` because
    /// `LoopToken.__del__` (`memmgr.py:13`) dispatches
    /// `cpu.free_loop_and_bridges` automatically when the only strong
    /// owner (`alive_loops`) drops the token.  Pyre still has a second
    /// strong owner — `MetaInterp::compiled_loops` keyed by green_key
    /// — so the caller (`pyjitpl::try_to_free_some_loops`) must drop
    /// the matching entry to actually free backend resources.
    ///
    /// The return value is a `Vec<Arc<JitCellToken>>` rather than
    /// `Vec<u64>` (green_keys) so the caller can match by **token-object
    /// identity** (`Arc::ptr_eq`) — mirroring `memmgr.py:73`'s
    /// `del self.alive_loops[looptoken]`, which keys on the looptoken
    /// itself.  Returning green_keys would let an evicted stale
    /// looptoken kick out the *current* compiled token at the same
    /// green_key (the recompile case where `compiled_loops[gk].token`
    /// has already been replaced).  This adaptation is removed once
    /// Slice X-G converts `compiled_loops` to `Weak`
    /// (memmgr-line-by-line-parity.md Slice 3.6).
    pub fn next_generation(&mut self) -> Vec<Arc<JitCellToken>> {
        self.current_generation += 1;
        if self.current_generation == self.next_check {
            let evicted = self._kill_old_loops_now();
            self.next_check = self.current_generation + self.check_frequency;
            evicted
        } else {
            Vec::new()
        }
    }

    /// `memmgr.py:63-83` `_kill_old_loops_now`.  RPython:
    /// ```python
    /// debug_start("jit-mem-collect")
    /// oldtotal = len(self.alive_loops)
    /// debug_print("Current generation:", self.current_generation)
    /// debug_print("Loop tokens before:", oldtotal)
    /// max_generation = self.current_generation - (self.max_age - 1)
    /// for looptoken in self.alive_loops.keys():
    ///     if (0 <= looptoken.generation < max_generation
    ///         or looptoken.invalidated):
    ///         del self.alive_loops[looptoken]
    /// newtotal = len(self.alive_loops)
    /// debug_print("Loop tokens freed: ", oldtotal - newtotal)
    /// debug_print("Loop tokens left:  ", newtotal)
    /// debug_stop("jit-mem-collect")
    /// ```
    /// Pyre uses `HashMap::retain` to fuse the iterate + delete steps.
    /// PRE-EXISTING-ADAPTATION: pyre uses a single `MAJIT_LOG` switch
    /// instead of `rlib/debug.py debug_start/stop` per-channel
    /// registry.  Channel name embedded in the message prefix
    /// (`[jit-mem-collect]`) maintains grep-equivalence with upstream
    /// traces.  Future: port `debug_start/stop` channel registry.
    fn _kill_old_loops_now(&mut self) -> Vec<Arc<JitCellToken>> {
        let log = crate::majit_log_enabled();
        let oldtotal = if log { self.alive_loops.len() } else { 0 };
        if log {
            eprintln!(
                "[jit-mem-collect] Current generation: {}",
                self.current_generation
            );
            eprintln!("[jit-mem-collect] Loop tokens before: {}", oldtotal);
        }
        let max_generation = self.current_generation - (self.max_age - 1);
        // memmgr.py:70-73 `for looptoken in self.alive_loops.keys(): if
        // (0 <= looptoken.generation < max_generation or
        // looptoken.invalidated): del self.alive_loops[looptoken]`.
        // Pyre returns the looptoken Arcs themselves so the caller can
        // match by Arc identity (cf. RPython's `del` keying on the
        // looptoken object) when dropping `compiled_loops` entries.
        let mut evicted_tokens = Vec::new();
        self.alive_loops.retain(|_key, token| {
            let token_gen = token.generation.get();
            let invalidated = token.invalidated.load(Ordering::Relaxed);
            let evict = (0 <= token_gen && token_gen < max_generation) || invalidated;
            if evict {
                evicted_tokens.push(Arc::clone(token));
            }
            !evict
        });
        if log {
            let newtotal = self.alive_loops.len();
            eprintln!(
                "[jit-mem-collect] Loop tokens freed: {}",
                oldtotal - newtotal
            );
            eprintln!("[jit-mem-collect] Loop tokens left: {}", newtotal);
        }
        evicted_tokens
    }

    /// `memmgr.py:85-89` `release_all_loops`.
    ///
    /// ```python
    /// debug_start("jit-mem-releaseall")
    /// debug_print("Loop tokens cleared:", len(self.alive_loops))
    /// self.alive_loops.clear()
    /// debug_stop("jit-mem-releaseall")
    /// ```
    pub fn release_all_loops(&mut self) {
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit-mem-releaseall] Loop tokens cleared: {}",
                self.alive_loops.len()
            );
        }
        self.alive_loops.clear();
    }

    /// Number of loops currently tracked.  Test/debug accessor; no
    /// upstream counterpart since RPython's `alive_loops` is a Python
    /// dict whose `len()` is read directly.
    pub fn alive_count(&self) -> usize {
        self.alive_loops.len()
    }

    /// `memmgr.py:38` `current_generation` read accessor.  Test/debug
    /// only; production code reads the field directly.
    pub fn current_generation(&self) -> i64 {
        self.current_generation
    }

    /// Test/debug accessor — `looptoken in self.alive_loops` upstream.
    pub fn contains(&self, looptoken: &Arc<JitCellToken>) -> bool {
        self.alive_loops.contains_key(&Arc::as_ptr(looptoken))
    }
}
