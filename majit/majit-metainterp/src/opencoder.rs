//! Literal port of `rpython/jit/metainterp/opencoder.py` — the byte-
//! stream trace recorder + iterator + snapshot chain reader used by
//! the meta-interpreter.
use std::collections::HashMap;

use majit_ir::{InputArg, OPCODE_COUNT, Op, OpCode, OpRef, Type, Value};

use crate::r#box::BoxRef;
use crate::constant_pool::ConstantPool;
use crate::history::TreeLoop;

fn u16_to_opcode(v: u16) -> OpCode {
    assert!(
        (v as usize) < OPCODE_COUNT,
        "invalid opcode discriminant: {v}"
    );
    // SAFETY: OpCode is #[repr(u16)] and we checked the discriminant is in range.
    unsafe { std::mem::transmute(v) }
}

// ── Signed varint encoding (opencoder.py:59-89) ──
//
// Literal port of rpython/jit/metainterp/opencoder.py encode_varint_signed
// / decode_varint_signed. Either 2 bytes or 4 bytes, chosen by range:
//
//   2 bytes: value in [-2^14, 2^14-1]  (bit 7 of byte 0 == 0)
//     byte 0: low 7 bits of value; bit 7 = 0
//     byte 1: bits 7..14 of value; bit 7 of byte 1 acts as sign bit
//
//   4 bytes: value in [MIN_VALUE, MAX_VALUE] = [-2^30, 2^30-1]
//     byte 0: low 7 bits of value; bit 7 = 1 (flag)
//     byte 1: bits 7..14 of value
//     byte 2: bits 15..22 of value
//     byte 3: bits 23..30 of value; bit 7 of byte 3 is sign bit
//
// Decoding sign-extends from bit (shift) using the top bit of the last
// written byte.
//
// This replaces the pre-Phase-B zigzag LEB128 encoder so the pyre wire
// format binary-matches RPython.

/// opencoder.py:59-73 encode_varint_signed.
pub fn encode_varint_signed(buf: &mut Vec<u8>, value: i64) {
    debug_assert!(
        MIN_VALUE <= value && value <= MAX_VALUE,
        "encode_varint_signed out of range: {value}"
    );
    let mut v = value;
    let flag: u8 = if !(-(1 << 14)..(1 << 14)).contains(&v) {
        0x80
    } else {
        0
    };
    buf.push(((v & 0b0111_1111) as u8) | flag);
    v >>= 7;
    buf.push((v & 0xff) as u8);
    if flag != 0 {
        v >>= 8;
        buf.push((v & 0xff) as u8);
        v >>= 8;
        buf.push((v & 0xff) as u8);
    }
}

/// opencoder.py:75-89 decode_varint_signed. Returns (value, bytes_consumed).
pub fn decode_varint_signed(buf: &[u8]) -> (i64, usize) {
    let byte0 = buf[0];
    let byte1 = buf[1];
    let mut res: i64 = ((byte0 & 0b0111_1111) as i64) | ((byte1 as i64) << 7);
    let mut shift: u32 = 15;
    let mut index: usize = 2;
    let mut lastbyte = byte1;
    if byte0 & 0b1000_0000 != 0 {
        let byte2 = buf[index];
        let byte3 = buf[index + 1];
        res |= ((byte2 as i64) << 15) | ((byte3 as i64) << 23);
        shift = 31;
        index += 2;
        lastbyte = byte3;
    }
    // sign-extend: top bit of the last written byte is the sign bit.
    if lastbyte & 0b1000_0000 != 0 {
        res |= -1i64 << shift;
    }
    (res, index)
}

// ── Snapshot management (opencoder.py) ──

/// Tag constants for snapshot encoding.
/// opencoder.py: TAGINT, TAGCONSTPTR, TAGCONSTOTHER, TAGBOX
pub const TAGINT: u8 = 0; // small integer constant
pub const TAGCONSTPTR: u8 = 1; // GC pointer constant
pub const TAGCONSTOTHER: u8 = 2; // big int or float constant
pub const TAGBOX: u8 = 3; // reference to a traced value (OpRef)

/// Number of tag bits.
pub const TAG_SHIFT: u32 = 2;
/// Mask for tag extraction.
pub const TAG_MASK: u8 = (1 << TAG_SHIFT) - 1;
/// RPython compatibility constants.
pub const TAGMASK: u8 = TAG_MASK;
pub const TAGSHIFT: u32 = TAG_SHIFT;

/// Initial trace buffer size.
pub const INIT_SIZE: usize = 4096;

/// RPython compatibility numeric limits for signed varints in this model.
pub const MIN_VALUE: i64 = -(1 << 30);
pub const MAX_VALUE: i64 = (1 << 30) - 1;

/// RPython compatibility bounds for small int tagging.
pub const SMALL_INT_START: i64 = -(1 << 28);
pub const SMALL_INT_STOP: i64 = 1 << 28;

/// RPython compatibility sentinel values for snapshot linked lists.
pub const SNAPSHOT_PREV_NEEDS_PATCHING: i32 = -3;
pub const SNAPSHOT_PREV_NONE: i32 = -2;
pub const SNAPSHOT_PREV_COMES_NEXT: i32 = -1;

/// opencoder.py:91-100 skip_varint_signed.
///
/// Each varint is 2 bytes if bit 7 of byte 0 is clear, 4 bytes if set.
/// `skip` is the number of varints to skip over; returns the new cursor.
#[inline]
pub fn skip_varint_signed(buf: &[u8], mut index: usize, skip: usize) -> usize {
    assert!(skip > 0);
    let mut remaining = skip;
    loop {
        let byte = buf[index];
        if byte & 0b1000_0000 != 0 {
            index += 2;
        }
        index += 2;
        remaining -= 1;
        if remaining == 0 {
            return index;
        }
    }
}

#[inline]
pub fn varint_only_decode(buf: &[u8], index: usize, skip: usize) -> i64 {
    let start = if skip > 0 {
        skip_varint_signed(buf, index, skip)
    } else {
        index
    };
    decode_varint_signed(&buf[start..]).0
}

#[inline]
pub fn combine_uint(index1: u32, index2: u32) -> u32 {
    (index1 << 16) | index2
}

#[inline]
pub fn unpack_uint(packed: u32) -> (u32, u32) {
    ((packed >> 16) & 0xffff, packed & 0xffff)
}

/// Encode a tagged value for snapshot storage.
/// opencoder.py: tag(kind, value)
pub fn tag(kind: u8, value: u32) -> u32 {
    (value << TAG_SHIFT) | kind as u32
}

/// Decode a tagged value from snapshot storage.
/// opencoder.py: untag(tagged) -> (kind, value)
pub fn untag(tagged: u32) -> (u8, u32) {
    let kind = (tagged & TAG_MASK as u32) as u8;
    let value = tagged >> TAG_SHIFT;
    (kind, value)
}

// ── Trace Iterator ──
// opencoder.py:249-406 TraceIterator(BaseTrace).
//
// Literal port of RPython's TraceIterator. Each call to `next()` produces
// a *fresh* operation whose `pos` is a freshly allocated OpRef (from the
// majit-specific `_fresh` counter) and whose args have been translated
// through `_cache` from raw trace positions to the per-iteration fresh
// OpRefs.
//
// In RPython every visited op is a freshly allocated `cls()` ResOperation
// Python object whose identity is `is`. Two iterators over the same trace
// therefore produce DIFFERENT box objects at the same `_index` slot — the
// `_index` field walks `_cache` in raw trace position coordinates; the
// freshness comes from `cls()`. majit has no separate identity from
// position, so we split the RPython `_index` into two fields:
//
//   * `_index`  — raw trace position coordinate, matching RPython's
//                 `self._cache[self._index] = res; self._index += 1`. Used
//                 by `replace_last_cached`/`kill_cache_at` which key on
//                 `_cache` slots.
//   * `_fresh`  — majit-specific fresh OpRef allocation counter, seeded
//                 from the `start_fresh` constructor argument. Phase 1
//                 starts at `num_inputs`; Phase 2 starts at Phase 1's
//                 `_fresh` high water so the two iterations produce
//                 disjoint OpRef ranges.
//
// RPython keeps these two roles fused because Python object identity
// makes `_index`'s dual use safe.

/// opencoder.py:249 class TraceIterator(BaseTrace).
pub struct TraceIterator<'a> {
    /// opencoder.py:252 self.trace
    pub trace: &'a [majit_ir::Op],
    /// opencoder.py:255 self._cache: per-iterator map from raw trace
    /// position to fresh box (OpRef) materialized for this iteration.
    /// In RPython this is `[None] * trace._index`; here `Vec<Option<OpRef>>`.
    pub _cache: Vec<Option<OpRef>>,
    /// opencoder.py:259-262 self.inputargs: fresh inputarg boxes for this
    /// iteration. Each is an `OpRef` allocated from the iterator's
    /// `_fresh` counter at construction time.
    pub inputargs: Vec<OpRef>,
    /// opencoder.py:268 self.start
    pub start: usize,
    /// opencoder.py:269 self.pos
    pub pos: usize,
    /// opencoder.py:270 self._count
    pub _count: u32,
    /// opencoder.py:271 self._index: raw trace position coordinate of the
    /// next `_cache` slot to write. RPython uses this as both the cache
    /// slot and the implicit fresh-ResOp position; majit uses it ONLY for
    /// the cache slot (see `_fresh` for the OpRef allocation counter).
    /// Advanced on every non-void op processed by `next()`.
    pub _index: u32,
    /// opencoder.py:272 self.start_index — raw trace position coordinate
    /// of the iteration start.
    pub start_index: u32,
    /// opencoder.py:273 self.end
    pub end: usize,
    /// majit-specific fresh OpRef allocation counter. Seeded from the
    /// `start_fresh` constructor argument. Advances by one per non-void
    /// op in `next()` and per inputarg in `new()`. Two iterations over
    /// the same trace produce disjoint OpRef ranges by passing different
    /// `start_fresh` values.
    pub _fresh: u32,
    /// Per-iter `BoxRef` pool mirroring `trace.get_iter()`'s per-call
    /// `inputarg_from_tp(...)` and `cls()` allocations. Each `TraceIterator`
    /// instance owns its own fresh `Rc<Box>` set so Phase 1 / Phase 2
    /// sub-Optimizers do not share `Box._forwarded` cells.
    ///
    /// Layout: dense `Vec<BoxRef>` indexed by raw OpRef value.
    /// `[0..p1_full_prefix.len())` are `Rc::clone`s of Phase 1's
    /// `final_ctx.box_pool` (inputarg + emit BoxRefs) when
    /// `p1_full_prefix` is `Some` — Phase 2 chain walks observe Phase 1's
    /// accumulated `_forwarded` info naturally (`unroll.py` Phase 2 reads
    /// Phase 1 `box._forwarded` through shared Python identity parity).
    /// `import_state(targetargs, ...)` writes `Forwarded::Box(target)` on
    /// each Phase 2 inputarg whose `target.raw() < num_inputs` — chain
    /// walks land directly on Phase 1's inputarg BoxRef cell rather than a
    /// Type::Void placeholder.
    /// Remaining `[..start_fresh)` filled with placeholder
    /// `BoxRef::new_resop(Type::Void)` for any unsnapshotted gap.
    /// Inputargs at `[start_fresh..start_fresh + num_inputargs)`.
    /// Op results pushed in `next()` for both void and non-void ops.
    pub box_pool: crate::r#box::BoxPool,
}

impl<'a> TraceIterator<'a> {
    /// opencoder.py:250-273 TraceIterator.__init__.
    ///
    /// `force_inputargs` corresponds to the same RPython parameter
    /// (cut-trace path); when None, the iterator pre-seeds the cache with
    /// the trace's own inputargs at positions `[0, num_inputargs)`.
    /// `start_fresh` is majit-specific: it seeds the fresh OpRef counter
    /// so two iterations over the same trace can produce disjoint OpRef
    /// ranges. RPython does not need this because each iteration's
    /// `cls()` allocation produces distinct Python objects.
    pub fn new(
        trace: &'a [majit_ir::Op],
        start: usize,
        end: usize,
        force_inputargs: Option<&[OpRef]>,
        inputarg_types: &[majit_ir::Type],
        start_fresh: u32,
        p1_full_prefix: Option<Vec<BoxRef>>,
    ) -> Self {
        // self._cache = [None] * trace._index
        // The iterator's cache must be large enough to hold any raw trace
        // position we may encounter. RPython sizes it from `trace._index`
        // (the encoder's monotonic op-result counter); here the equivalent
        // is the maximum raw position seen in `trace[start..end]` plus one.
        let num_inputargs = inputarg_types.len();
        let max_pos = trace[start..end]
            .iter()
            .flat_map(|op| {
                std::iter::once(op.pos)
                    .chain(op.args.iter().copied())
                    .chain(op.fail_args.iter().flat_map(|fa| fa.iter().copied()))
            })
            .filter(|opref| !opref.is_none() && !opref.is_constant())
            .map(|opref| opref.raw())
            .max()
            .unwrap_or(0);
        let cache_size = ((max_pos as usize) + 1).max(num_inputargs);
        let mut _cache: Vec<Option<OpRef>> = vec![None; cache_size];
        let mut _fresh = start_fresh;
        // Per-iter BoxRef pool. When `p1_full_prefix` is `Some` (Phase 2
        // path), `[0..p1_full_prefix.len())` is `Rc::clone`s of Phase 1's
        // `final_ctx.box_pool` — inputarg BoxRefs at `[0..num_inputs)` plus
        // emit BoxRefs at `[num_inputs..p1_iter_fresh_hw)`. Chain walks via
        // `Forwarded::Box(rc)` observe Phase 1's accumulated `_forwarded`
        // info naturally (`unroll.py` Phase 2 reads Phase 1 box._forwarded
        // through shared Python identity); `import_state` setting
        // `Forwarded::Box(target)` on Phase 2 inputargs lands the chain on
        // the same Phase 1 inputarg cell rather than a Type::Void
        // placeholder. Remaining `[p1_full_prefix.len()..start_fresh)` is
        // filled with placeholder `BoxRef::new_resop(Type::Void)` only for
        // any unsnapshotted gap (or all of `[..start_fresh)` when prefix is
        // `None`). Phase 2 owns the Rc::cloned BoxRefs as its single-writer
        // post-Phase-1-completion (audit memo Risk 1).
        let p1_prefix = p1_full_prefix.unwrap_or_default();
        let p1_end = p1_prefix.len().min(start_fresh as usize);
        let mut box_pool: crate::r#box::BoxPool = p1_prefix
            .into_iter()
            .take(p1_end)
            .chain((p1_end as u32..start_fresh).map(|i| BoxRef::new_resop(Type::Void, i)))
            .collect::<Vec<_>>()
            .into();
        let inputargs: Vec<OpRef>;
        if let Some(force) = force_inputargs {
            // opencoder.py:259-262 self.inputargs =
            //     [rop.inputarg_from_tp(arg.type) for arg in force_inputargs]
            //   for i, arg in enumerate(force_inputargs):
            //       self._cache[arg.get_position()] = self.inputargs[i]
            inputargs = force
                .iter()
                .map(|arg| {
                    // opencoder.py:259-262: rop.inputarg_from_tp(arg.type)
                    // — every force_inputargs entry has a known type tag.
                    let tp = arg.ty().unwrap_or_else(|| {
                        panic!(
                            "TraceIterator::new(force_inputargs): untyped OpRef ({:?}) — \
                             upstream `inputarg_from_tp(arg.type)` requires every entry to \
                             carry a type tag",
                            arg
                        )
                    });
                    let r = OpRef::input_arg_typed(_fresh, tp);
                    // Slice 77b.A: companion BoxRef per inputarg, fresh per iter.
                    box_pool.push(BoxRef::new_inputarg(tp, Some(_fresh)));
                    _fresh += 1;
                    r
                })
                .collect();
            for (i, &arg) in force.iter().enumerate() {
                let p = arg.raw() as usize;
                if p >= _cache.len() {
                    _cache.resize(p + 1, None);
                }
                _cache[p] = Some(inputargs[i]);
            }
        } else {
            // opencoder.py:264-267 self.inputargs =
            //     [rop.inputarg_from_tp(arg.type) for arg in self.trace.inputargs]
            //   for i, arg in enumerate(self.inputargs):
            //       self._cache[self.trace.inputargs[i].get_position()] = arg
            inputargs = inputarg_types
                .iter()
                .map(|&tp| {
                    let r = OpRef::input_arg_typed(_fresh, tp);
                    // Slice 77b.A: companion BoxRef per inputarg, fresh per iter.
                    box_pool.push(BoxRef::new_inputarg(tp, Some(_fresh)));
                    _fresh += 1;
                    r
                })
                .collect();
            for i in 0..num_inputargs {
                _cache[i] = Some(inputargs[i]);
            }
        }
        TraceIterator {
            trace,
            _cache,
            inputargs,
            start,
            pos: start,
            _count: start as u32,
            _index: start as u32,
            start_index: start as u32,
            end,
            _fresh,
            box_pool,
        }
    }

    /// opencoder.py:286-289 _get(self, i).
    fn _get(&self, i: usize) -> OpRef {
        match self._cache.get(i).copied().flatten() {
            Some(res) => res,
            None => panic!(
                "TraceIterator._get cache miss at {i} (cache_len={}, pos={}, start={}, end={}, index={}, start_index={}, fresh={})",
                self._cache.len(),
                self.pos,
                self.start,
                self.end,
                self._index,
                self.start_index,
                self._fresh,
            ),
        }
    }

    /// opencoder.py:291-292 done().
    pub fn done(&self) -> bool {
        self.pos >= self.end
    }

    /// opencoder.py:321-335 _untag(tagged).
    ///
    /// In RPython this dispatches on the tag (TAGBOX/TAGINT/TAGCONSTPTR/
    /// TAGCONSTOTHER). majit's OpRef carries the tag implicitly: constants
    /// have `OpRef >= CONST_BASE`, NONE is `u32::MAX`. Both pass through
    /// unchanged; only TAGBOX-equivalent OpRefs go through `_get`.
    fn _untag(&self, opref: OpRef) -> OpRef {
        if opref.is_none() || opref.is_constant() {
            opref
        } else {
            self._get(opref.raw() as usize)
        }
    }

    /// opencoder.py:278-280 kill_cache_at(pos).
    pub fn kill_cache_at(&mut self, pos: usize) {
        if pos != 0 {
            self._cache[pos] = None;
        }
    }

    /// opencoder.py:282-284 replace_last_cached(oldbox, box).
    /// RPython writes `_cache[_index - 1]` where `_index` walks `_cache`
    /// densely in ops-count coordinates. In majit the key is the raw
    /// trace position (which is monotonic for non-void ops in a
    /// well-formed trace), so `_index - 1` lands on the same cache slot
    /// the previous `next()` call wrote.
    pub fn replace_last_cached(&mut self, oldbox: OpRef, new_box: OpRef) {
        let last_idx = (self._index - 1) as usize;
        debug_assert_eq!(self._cache[last_idx], Some(oldbox));
        self._cache[last_idx] = Some(new_box);
    }

    /// opencoder.py:362-406 next() — produce the next operation as a fresh
    /// box (`cls()` in RPython) with translated args.
    pub fn next(&mut self) -> Option<majit_ir::Op> {
        if self.done() {
            return None;
        }
        let src = &self.trace[self.pos];
        self.pos += 1;
        let mut res = src.clone();
        // for i in range(argnum):
        //     res.setarg(i, self._untag(self._next()))
        for arg in res.args.iter_mut() {
            *arg = self._untag(*arg);
        }
        if let Some(ref mut fa) = res.fail_args {
            for arg in fa.iter_mut() {
                *arg = self._untag(*arg);
            }
        }
        // RPython opencoder.py:399-401:
        //     res = ResOperation(opnum, args, descr)   # fresh cls() object
        //     if res.type != 'v':
        //         self._cache[self._index] = res
        //         self._index += 1
        //
        // RPython allocates a fresh `cls()` ResOperation Python object on
        // EVERY visited op (void or non-void); only non-void results land
        // in `_cache`, but the freshness is unconditional. majit's
        // analogue of "fresh cls()" is the `_fresh` OpRef counter, so we
        // must advance `_fresh` for void ops too — otherwise a non-void
        // op processed after a void op gets a fresh OpRef that collides
        // with the void op's raw `pos`. With this in place,
        // `start_fresh = 0` over a recorder-emitted trace produces a
        // bit-identical sequence of OpRefs (every op `i` gets
        // `OpRef::from_raw(num_inputs + i)`, matching `recorder.record_op`'s
        // monotonic `op_count`).
        let is_void_result = src.pos.is_none() || src.opcode.result_type() == majit_ir::Type::Void;
        if !is_void_result {
            let orig = src.pos.raw() as usize;
            if orig >= self._cache.len() {
                self._cache.resize(orig + 1, None);
            }
            // opencoder.py:399-401 always allocates a fresh ResOperation
            // instance and then overwrites `_cache[self._index]` with it.
            // Reusing an existing cache entry collapses distinct box
            // identities when a raw op slot collides with an inputarg or
            // earlier phase-local box. The ResOp class (`opclasses[opnum]`
            // / cls()) is intrinsically typed via the IntOp/FloatOp/RefOp
            // mixin (resoperation.py:564-638); pyre uses
            // `opcode.result_type()`.
            let fresh = OpRef::op_typed(self._fresh, src.opcode.result_type());
            // Slice 77b.A: companion BoxRef for the fresh op result.
            self.box_pool
                .push(BoxRef::new_resop(src.opcode.result_type(), self._fresh));
            self._fresh += 1;
            self._cache[orig] = Some(fresh);
            res.pos = fresh;
            // RPython `_index` parity: advance past the cache slot we
            // just wrote. In RPython this happens via `_index += 1`
            // because `_index` == cache slot index; in majit the cache
            // key is the raw trace position, so we assign `orig + 1`
            // directly to keep `_cache[_index - 1]` (`replace_last_cached`)
            // pointing at the slot we just wrote.
            self._index = orig as u32 + 1;
        } else if !src.pos.is_none() {
            // Void op carrying a raw trace position: still allocate a
            // fresh OpRef so the `_fresh` counter stays in lockstep with
            // the raw trace position counter. The op is not cached
            // (RPython doesn't cache void results either), so later args
            // never reference it. `AbstractResOp.type = 'v'`
            // (resoperation.py:260) → VoidOp variant.
            let f = OpRef::void_op(self._fresh);
            // Slice 77b.A: companion Void-typed BoxRef placeholder so
            // `box_pool[idx]` indexing stays dense.
            self.box_pool
                .push(BoxRef::new_resop(Type::Void, self._fresh));
            self._fresh += 1;
            res.pos = f;
        } else {
            res.pos = src.pos;
        }
        // self._count += 1
        self._count += 1;
        Some(res)
    }
}

// ── Byte-stream TraceIterator (M4) ─────────────────────────────────────
//
// `ByteTraceIter` is the structurally-correct port of RPython's
// `TraceIterator` (opencoder.py:249-406).  It walks
// `TraceRecordBuffer._ops` byte-by-byte, producing a fresh `Op` per
// iteration.  The existing `TraceIterator<'a>` above still exists for
// legacy consumers that hold a pre-materialized `&[Op]` slice (via
// `TreeLoop.ops`); both coexist during the M2–M7 migration and
// `ByteTraceIter` takes over once every consumer is migrated off the
// structured-IR slice.
//
// Current scope (M4 step 1): fixed-arity 0/1/2/3 opcodes with TAGBOX
// args only.  Constants (TAGINT/TAGCONSTPTR/TAGCONSTOTHER), variable-
// arity ops (arity == None), and descr-bearing ops panic with a
// dedicated message — those decode branches land in M4 step 2 alongside
// ConstantPool + descr-table wiring.

/// opencoder.py:249-406 `TraceIterator`.  Byte-stream walker over
/// `TraceRecordBuffer._ops`.
/// opencoder.py:408-427 `class CutTrace(BaseTrace)` — thin view over
/// a `TraceRecordBuffer` that starts iteration at `start` (a byte
/// offset captured from `cut_point()`) and substitutes a fresh
/// inputarg list for the op stream between `start` and
/// `trace._pos`.
///
/// `CutTrace` borrows its parent trace shared-read; the bridge
/// compilation path produces one per guard failure that triggers a
/// sub-trace retrace.
pub struct CutTrace<'a> {
    pub trace: *mut TraceRecordBuffer,
    pub start: usize,
    pub count: u32,
    pub index: u32,
    /// Templates: each typed `OpRef` names an op position in the
    /// parent trace and carries that op's result type; `get_iter`
    /// seeds `_cache[opref.raw()]` with the fresh OpRef so downstream
    /// TAGBOX(p) decodes resolve to the new iterator-local inputarg.
    /// Mirrors RPython `cut_trace_from(cut, inputargs)` where
    /// `inputargs` is a list of `Box{Int,Float,Ref}` — each Box
    /// carries `Box.type` (`history.py:182 AbstractValue.type`) which
    /// pyre encodes as the typed `OpRef` variant tag.
    pub inputargs: Vec<OpRef>,
    _marker: std::marker::PhantomData<&'a TraceRecordBuffer>,
}

impl<'a> CutTrace<'a> {
    /// opencoder.py:416-418 `CutTrace.cut_at(cut)` — delegate to the
    /// parent trace after asserting the cut lies beyond this cut-trace's
    /// starting count.
    pub fn cut_at(&self, cut: crate::recorder::TracePosition) {
        assert!(cut._count > self.count, "cut must lie after CutTrace start");
        unsafe { (&mut *self.trace).cut_at(cut) };
    }

    /// opencoder.py:420-427 `CutTrace.get_iter` — build a
    /// `ByteTraceIter` that walks the parent trace from `start` to
    /// `trace._pos`, with `_cache` / `inputargs` seeded from the cut's
    /// inputarg templates.
    pub fn get_iter(&self) -> ByteTraceIter<'a> {
        let trace = unsafe { &*self.trace };
        ByteTraceIter::new_for_cut(
            trace,
            self.start,
            trace._pos,
            self.count,
            self.index,
            &self.inputargs,
            None,
        )
    }

    /// Pool-aware variant for TAGINT / TAGCONST* decode (matches
    /// `get_byte_iter_with_pool` on the parent trace).
    pub fn get_iter_with_pool(
        &self,
        const_pool: &'a mut crate::constant_pool::ConstantPool,
    ) -> ByteTraceIter<'a> {
        let trace = unsafe { &*self.trace };
        ByteTraceIter::new_for_cut(
            trace,
            self.start,
            trace._pos,
            self.count,
            self.index,
            &self.inputargs,
            Some(const_pool),
        )
    }
}

pub struct ByteTraceIter<'a> {
    /// Enclosing trace buffer — source of `_ops`, `_refs`, `_bigints`,
    /// `_floats`, `_descrs`, `metainterp_sd.all_descrs`.
    pub trace: &'a TraceRecordBuffer,
    /// opencoder.py:255 `self._cache` — raw trace position → fresh
    /// per-iteration OpRef.
    pub _cache: Vec<Option<OpRef>>,
    /// opencoder.py:259-263 `self.inputargs` — fresh iterator-local
    /// OpRefs bound to the trace's inputargs.
    pub inputargs: Vec<OpRef>,
    pub start: usize,
    pub pos: usize,
    pub end: usize,
    pub _count: u32,
    pub _index: u32,
    pub start_index: u32,
    /// Majit-specific fresh OpRef counter (same role as
    /// `TraceIterator::_fresh` on the legacy walker).
    pub _fresh: u32,
    /// Optional ConstantPool for materializing TAGINT / TAGCONSTPTR /
    /// TAGCONSTOTHER constants into caller-owned OpRefs.  When `None`,
    /// any constant-tagged arg panics (M4 step 1 back-compat).  When
    /// `Some`, constants are de-duplicated via
    /// `ConstantPool::get_or_insert_typed` — the returned OpRef is
    /// owned by the caller's pool so downstream consumers can resolve
    /// values through the same pool.
    pub const_pool: Option<&'a mut crate::constant_pool::ConstantPool>,
}

impl<'a> ByteTraceIter<'a> {
    /// opencoder.py:250-273 `TraceIterator.__init__`.
    ///
    /// `start` / `end` are byte offsets into `trace._ops`.  Pass
    /// `trace._start as usize` / `trace._pos` for a full walk.
    /// `start_fresh` seeds the fresh-OpRef counter (pyre-only — enables
    /// disjoint OpRef namespaces across successive iterations).
    pub fn new(trace: &'a TraceRecordBuffer, start: usize, end: usize, start_fresh: u32) -> Self {
        Self::new_with_pool(trace, start, end, start_fresh, None)
    }

    /// M4 step 2: constructor that attaches a ConstantPool for
    /// constant-arg decode.  When `const_pool` is `Some`, TAGINT /
    /// TAGCONSTPTR / TAGCONSTOTHER args are resolved through the pool
    /// (`get_or_insert_typed`); when `None`, they panic (the M4 step 1
    /// behaviour that keeps TAGBOX-only tests honest).
    pub fn new_with_pool(
        trace: &'a TraceRecordBuffer,
        start: usize,
        end: usize,
        start_fresh: u32,
        const_pool: Option<&'a mut crate::constant_pool::ConstantPool>,
    ) -> Self {
        let cache_size = (trace._index as usize).max(trace.max_num_inputargs as usize);
        let mut _cache: Vec<Option<OpRef>> = vec![None; cache_size];
        let mut _fresh = start_fresh;
        // opencoder.py:264-265 `[rop.inputarg_from_tp(arg.type) for arg in
        // self.trace.inputargs]` — type comes from each `InputArg.tp`.
        let inputargs: Vec<OpRef> = trace
            .inputargs
            .iter()
            .map(|ia| {
                let r = OpRef::input_arg_typed(_fresh, ia.tp);
                _fresh += 1;
                r
            })
            .collect();
        for (i, ia) in trace.inputargs.iter().enumerate() {
            let p = ia.index as usize;
            if p >= _cache.len() {
                _cache.resize(p + 1, None);
            }
            _cache[p] = Some(inputargs[i]);
        }
        ByteTraceIter {
            trace,
            _cache,
            inputargs,
            start,
            pos: start,
            end,
            _count: start as u32,
            _index: start as u32,
            start_index: start as u32,
            _fresh,
            const_pool,
        }
    }

    /// opencoder.py:250-273 + opencoder.py:421-427 `CutTrace.get_iter`
    /// — specialised `TraceIterator` constructor for `cut_trace_from`
    /// that seeds `_cache` from explicit typed-`OpRef` inputarg
    /// templates.  Each template names an op position in the uncut
    /// trace and carries that op's result type via its variant tag,
    /// matching RPython's `Box.type` (`history.py:182`).  The seeded
    /// fresh OpRefs reuse the template's type via
    /// `OpRef::input_arg_typed`, so downstream consumers preserve the
    /// `Box.type` invariant the cut sub-trace operates under.
    pub fn new_for_cut(
        trace: &'a TraceRecordBuffer,
        start: usize,
        end: usize,
        count: u32,
        index: u32,
        inputarg_templates: &[OpRef],
        const_pool: Option<&'a mut crate::constant_pool::ConstantPool>,
    ) -> Self {
        let cache_size = (trace._index as usize).max(trace.max_num_inputargs as usize);
        let mut _cache: Vec<Option<OpRef>> = vec![None; cache_size];
        let mut inputargs: Vec<OpRef> = Vec::with_capacity(inputarg_templates.len());
        let mut _fresh = 0u32;
        for &template in inputarg_templates {
            // Constant templates would imply the cut sub-trace treats a
            // literal as an inputarg — not a shape RPython's
            // `cut_trace_from` emits — so reject up front rather than
            // silently substitute.
            if template.is_constant() {
                panic!(
                    "ByteTraceIter::new_for_cut: constant inputarg template {:?} has \
                     no original trace position to map into `_cache`",
                    template
                );
            }
            let tp = template.ty().unwrap_or_else(|| {
                panic!(
                    "ByteTraceIter::new_for_cut: untyped inputarg template {:?} has no \
                     `Box.type` to carry into the fresh inputarg variant",
                    template
                )
            });
            let fresh_ref = OpRef::input_arg_typed(_fresh, tp);
            _fresh += 1;
            inputargs.push(fresh_ref);
            let slot = template.raw() as usize;
            if slot >= _cache.len() {
                _cache.resize(slot + 1, None);
            }
            _cache[slot] = Some(fresh_ref);
        }
        ByteTraceIter {
            trace,
            _cache,
            inputargs,
            start,
            pos: start,
            end,
            _count: count,
            _index: index,
            start_index: index,
            _fresh,
            const_pool,
        }
    }

    /// opencoder.py:291-292 `done`.
    pub fn done(&self) -> bool {
        self.pos >= self.end
    }

    /// opencoder.py:294-298 `_nextbyte`.
    fn _nextbyte(&mut self) -> u8 {
        let b = self.trace._ops[self.pos];
        self.pos += 1;
        b
    }

    /// opencoder.py:300-318 `_next` — signed varint decode from
    /// `trace._ops` at `self.pos`.  Shares the wire format with
    /// `TraceRecordBuffer::append_int`.
    fn _next(&mut self) -> i64 {
        let (v, consumed) = decode_varint_signed(&self.trace._ops[self.pos..]);
        self.pos += consumed;
        v
    }

    /// opencoder.py:286-289 `_get`.
    fn _get(&self, i: usize) -> OpRef {
        self._cache[i].expect("ByteTraceIter._get: cache miss")
    }

    /// opencoder.py:321-335 `_untag` — full dispatch.
    ///
    /// TAGBOX → `_cache` lookup.  TAGINT → small int pool entry.
    /// TAGCONSTPTR → `trace._refs[v]` → pool entry.  TAGCONSTOTHER →
    /// either `trace._floats[v >> 1]` (bit 0 == 1) or
    /// `trace._bigints[v >> 1]` (bit 0 == 0), routed through the pool.
    ///
    /// When `const_pool` is `None`, non-TAGBOX tags panic (back-compat
    /// with M4 step 1 TAGBOX-only usage).
    ///
    /// `pub` because `SnapshotIterator::get` / `unpack_array`
    /// (opencoder.py:222-231) dispatch through `main_iter._untag`.
    pub fn _untag(&mut self, tagged: i64) -> OpRef {
        // RPython opencoder.py:321-322 uses arithmetic shift on a
        // Python int; in Rust we preserve sign by going through i64
        // rather than u32 for the value.
        let tag = (tagged & TAG_MASK as i64) as u8;
        let v = tagged >> TAG_SHIFT;
        match tag {
            TAGBOX => {
                debug_assert!(v >= 0, "TAGBOX value must be non-negative, got {}", v);
                self._get(v as usize)
            }
            TAGINT => {
                // opencoder.py:326-327 ConstInt(v) — signed small int.
                match self.const_pool.as_deref_mut() {
                    Some(pool) => pool.get_or_insert_typed(v, Type::Int),
                    None => panic!(
                        "ByteTraceIter: TAGINT arg with value {} but no ConstantPool \
                         attached — construct via `new_with_pool(..., Some(&mut pool))`",
                        v
                    ),
                }
            }
            TAGCONSTPTR => {
                // opencoder.py:328-329 ConstPtr(self.trace._refs[v]).
                let addr = self.trace._refs[v as usize];
                match self.const_pool.as_deref_mut() {
                    Some(pool) => pool.get_or_insert_typed(addr as i64, Type::Ref),
                    None => panic!("ByteTraceIter: TAGCONSTPTR arg but no ConstantPool attached"),
                }
            }
            TAGCONSTOTHER => {
                // opencoder.py:330-334 bigint vs float split on bit 0.
                let pool_idx = (v >> 1) as usize;
                if v & 1 != 0 {
                    let bits = self.trace._floats[pool_idx];
                    match self.const_pool.as_deref_mut() {
                        Some(pool) => pool.get_or_insert_typed(bits as i64, Type::Float),
                        None => panic!(
                            "ByteTraceIter: TAGCONSTOTHER (float) but no ConstantPool attached"
                        ),
                    }
                } else {
                    let val = self.trace._bigints[pool_idx];
                    match self.const_pool.as_deref_mut() {
                        Some(pool) => pool.get_or_insert_typed(val, Type::Int),
                        None => panic!(
                            "ByteTraceIter: TAGCONSTOTHER (bigint) but no ConstantPool attached"
                        ),
                    }
                }
            }
            other => unreachable!("ByteTraceIter: unknown tag {}", other),
        }
    }
}

impl<'a> Iterator for ByteTraceIter<'a> {
    type Item = majit_ir::Op;
    fn next(&mut self) -> Option<majit_ir::Op> {
        if self.done() {
            return None;
        }
        // opencoder.py:391 `opnum = self._nextbyte()`.
        let opnum = self._nextbyte();
        let opcode = OpCode::from_u16(opnum as u16)
            .unwrap_or_else(|| panic!("ByteTraceIter: unknown opnum {}", opnum));
        // opencoder.py:392-394 `argnum = oparity[opnum]; if argnum == -1: argnum = self._next()`.
        let arity = match opcode.arity() {
            Some(n) => n as usize,
            // opencoder.py:394 variable-arity path: `argnum = self._next()`.
            None => self._next() as usize,
        };
        // opencoder.py:395-408 — read `argnum` tagged args and untag.
        let mut args: smallvec::SmallVec<[OpRef; 3]> = smallvec::SmallVec::new();
        for _ in 0..arity {
            let tagged = self._next();
            args.push(self._untag(tagged));
        }
        // opencoder.py:409-424 `opwithdescr` branch.  Guards thread the
        // snapshot index into `rd_resume_position` but emit `descr=None`
        // at the Op level; every other descr-bearing op resolves its
        // descr via `metainterp_sd.all_descrs` (global slot, index <
        // `all_descr_len + 1`) or `trace._descrs` (local slot, offset
        // by `all_descr_len + 1`).  `descr_index == 0` is the
        // placeholder used by `record_op` for no-descr writes.
        let (descr, rd_resume_position) = if opcode.has_descr() {
            let descr_index = self._next();
            let resume_pos = if opcode.is_guard() {
                descr_index as i32
            } else {
                -1
            };
            let resolved = if descr_index == 0 || opcode.is_guard() {
                None
            } else {
                let all_descrs = self.trace.metainterp_sd.all_descrs.lock().unwrap();
                let all_descr_len = all_descrs.len() as i64;
                if descr_index < all_descr_len + 1 {
                    Some(all_descrs[(descr_index - 1) as usize].clone())
                } else {
                    Some(
                        self.trace._descrs[(descr_index - all_descr_len - 1) as usize]
                            .clone()
                            .expect("ByteTraceIter: trace._descrs slot was None"),
                    )
                }
            };
            (resolved, resume_pos)
        } else {
            (None, -1)
        };
        // RPython opencoder.py:425-431: `res = ResOperation(opnum, args)`
        // is a fresh Python object; the cache only receives non-void
        // results.  majit allocates an OpRef for every op (void or not)
        // so later non-void ops get contiguous positions.
        let mut op = match descr {
            Some(d) => majit_ir::Op::with_descr(opcode, &args, d),
            None => majit_ir::Op::new(opcode, &args),
        };
        op.rd_resume_position = rd_resume_position;
        // opencoder.py:373-374 `cls = opclasses[opnum]; res = cls()` —
        // the ResOp class is intrinsically typed via the IntOp/FloatOp/
        // RefOp mixin (resoperation.py:564-638) or the AbstractResOp
        // default `'v'` (resoperation.py:260). pyre routes through
        // `op_typed` which lands on the matching variant.
        let fresh_pos = OpRef::op_typed(self._fresh, opcode.result_type());
        self._fresh += 1;
        op.pos = fresh_pos;
        // opencoder.py:429-431 — cache non-void result at `_index`, bump.
        if opcode.result_type() != Type::Void {
            let slot = self._index as usize;
            if slot >= self._cache.len() {
                self._cache.resize(slot + 1, None);
            }
            self._cache[slot] = Some(fresh_pos);
            self._index += 1;
        }
        // opencoder.py:432 `self._count += 1`.
        self._count += 1;
        Some(op)
    }
}

/// opencoder.py:848-850 `Trace.get_iter()` — byte-stream form.
impl TraceRecordBuffer {
    pub fn get_byte_iter(&self) -> ByteTraceIter<'_> {
        ByteTraceIter::new(
            self,
            self._start as usize,
            self._pos,
            self.max_num_inputargs,
        )
    }
}

/// opencoder.py:438-463 `class BoxArrayIter` — iterator over the
/// encoded tagged box values stored in `_snapshot_array_data`.
///
/// Layout at offset `array_idx`:
/// `[length:varint][box0:varint][box1:varint]...`. `array_idx == 0`
/// always decodes to the empty iterator (opencoder.py:465
/// `BoxArrayIter.BOXARRAYITER0`; also opencoder.py:728-733
/// `new_array(0) -> 0`).
pub struct BoxArrayIter<'a> {
    data: &'a [u8],
    pos: usize,
    remaining: i64,
    /// opencoder.py:441 `self.total_length = self.length` — the
    /// decoded array length, captured at construction. `remaining`
    /// counts down as `next()` consumes items, while `total_length`
    /// keeps the original count so `SnapshotIterator.size`
    /// (opencoder.py:210) can read it back without re-decoding.
    pub total_length: i64,
}

impl<'a> BoxArrayIter<'a> {
    /// opencoder.py:449-453 BoxArrayIter.make(index, data).
    pub fn new(data: &'a [u8], array_idx: i64) -> Self {
        if array_idx == 0 {
            return Self {
                data,
                pos: 0,
                remaining: 0,
                total_length: 0,
            };
        }
        let pos = array_idx as usize;
        let (length, consumed) = decode_varint_signed(&data[pos..]);
        Self {
            data,
            pos: pos + consumed,
            remaining: length,
            total_length: length,
        }
    }
}

impl<'a> Iterator for BoxArrayIter<'a> {
    type Item = i64;
    fn next(&mut self) -> Option<i64> {
        if self.remaining <= 0 {
            return None;
        }
        let (tagged, consumed) = decode_varint_signed(&self.data[self.pos..]);
        self.pos += consumed;
        self.remaining -= 1;
        Some(tagged)
    }
}

/// opencoder.py:141-199 `class TopDownSnapshotIterator` — walks the
/// snapshot chain encoded into `_snapshot_data` in outermost-to-
/// innermost order.
///
/// Yields byte offsets of each parent snapshot record (in the
/// `[jitcode][pc][array][prev]` layout written by `_encode_snapshot`).
/// The top snapshot's leading `vable_array_index` /
/// `vref_array_index` are extracted in `new()` (opencoder.py:150-151)
/// and exposed as struct fields.
pub struct TopDownSnapshotIterator<'a> {
    snapshot_data: &'a [u8],
    /// opencoder.py:150 `self.vable_array_index`.
    pub vable_array_index: i64,
    /// opencoder.py:151 `self.vref_array_index`.
    pub vref_array_index: i64,
    /// opencoder.py:152 `self.snapshot_index` — position of the current
    /// parent snapshot's `jitcode` varint.
    current: usize,
    /// opencoder.py:196 `if res == SNAPSHOT_PREV_NONE: raise StopIteration`.
    /// Pyre signals termination via a sentinel field because `usize` can't
    /// carry the -2 value RPython stores in `snapshot_index`.
    done: bool,
}

impl<'a> TopDownSnapshotIterator<'a> {
    /// opencoder.py:146-152 TopDownSnapshotIterator.__init__.
    pub fn new(snapshot_data: &'a [u8], snapshot_index: usize) -> Self {
        let mut p = snapshot_index;
        let (vable, c) = decode_varint_signed(&snapshot_data[p..]);
        p += c;
        let (vref, c) = decode_varint_signed(&snapshot_data[p..]);
        p += c;
        Self {
            snapshot_data,
            vable_array_index: vable,
            vref_array_index: vref,
            current: p,
            done: false,
        }
    }

    /// opencoder.py:160-162 iter_array(snapshot_index).
    ///
    /// `snapshot_index` is the byte offset of a parent snapshot's
    /// `jitcode` varint. Returns a byte iterator over the tagged
    /// boxes in that frame's array. `snapshot_array_data` is supplied
    /// as a separate argument to keep the returned iterator free of
    /// a back-borrow on `self`.
    pub fn iter_array<'b>(
        &self,
        snapshot_index: usize,
        snapshot_array_data: &'b [u8],
    ) -> BoxArrayIter<'b> {
        let array_idx = varint_only_decode(self.snapshot_data, snapshot_index, 2);
        BoxArrayIter::new(snapshot_array_data, array_idx)
    }

    /// opencoder.py:164-167 length(snapshot_index).
    ///
    /// Returns the length of the box array attached to the parent
    /// snapshot at `snapshot_index`. Reads the array offset from the
    /// third varint of the snapshot record, then decodes the length
    /// prefix at that offset in `_snapshot_array_data`.
    pub fn length(&self, snapshot_index: usize, snapshot_array_data: &[u8]) -> i64 {
        let array_idx = varint_only_decode(self.snapshot_data, snapshot_index, 2);
        varint_only_decode(snapshot_array_data, array_idx as usize, 0)
    }

    /// opencoder.py:177-181 unpack_jitcode_pc(snapshot_index).
    ///
    /// Decodes the leading `(jitcode_index, pc)` pair of a parent
    /// snapshot record. `snapshot_index` is the byte offset of the
    /// `jitcode` varint (the same coordinate yielded by the iterator
    /// and passed through `iter_array`).
    pub fn unpack_jitcode_pc(&self, snapshot_index: usize) -> (i64, i64) {
        let mut p = snapshot_index;
        let (jitcode_index, c) = decode_varint_signed(&self.snapshot_data[p..]);
        p += c;
        let (pc, _) = decode_varint_signed(&self.snapshot_data[p..]);
        (jitcode_index, pc)
    }

    /// opencoder.py:183-185 is_empty_snapshot(snapshot_index).
    ///
    /// Checks whether the parent snapshot at `snapshot_index` was
    /// written by `create_empty_top_snapshot` (jitcode_index == -1 and
    /// array == empty). Uses the array-index == -1 sentinel to mirror
    /// RPython's `varint_only_decode(..., skip=2) == -1` check.
    pub fn is_empty_snapshot(&self, snapshot_index: usize) -> bool {
        varint_only_decode(self.snapshot_data, snapshot_index, 2) == -1
    }
}

impl<'a> Iterator for TopDownSnapshotIterator<'a> {
    type Item = usize;
    /// opencoder.py:194-199 TopDownSnapshotIterator.next.
    fn next(&mut self) -> Option<usize> {
        if self.done {
            return None;
        }
        let res = self.current;
        // opencoder.py:170-175 prev(snapshot_index):
        //   self._index = skip_varint_signed(snapshot_data, snapshot_index, skip=3)
        //   prev = self.decode_snapshot_int()
        //   if prev == SNAPSHOT_PREV_COMES_NEXT: prev = self._index
        let after_array = skip_varint_signed(self.snapshot_data, res, 3);
        let (prev, consumed) = decode_varint_signed(&self.snapshot_data[after_array..]);
        debug_assert!(
            prev as i32 != SNAPSHOT_PREV_NEEDS_PATCHING,
            "snapshot chain contains unpatched NEEDS_PATCHING sentinel"
        );
        let next_cursor = match prev as i32 {
            SNAPSHOT_PREV_NONE => {
                self.done = true;
                res
            }
            SNAPSHOT_PREV_COMES_NEXT => after_array + consumed,
            _ => prev as usize,
        };
        self.current = next_cursor;
        Some(res)
    }
}

/// opencoder.py:202-232 `class SnapshotIterator` — eager reversed
/// walker over a snapshot chain.
///
/// RPython's `SnapshotIterator` holds the enclosing `TraceIterator` as
/// `main_iter` so `get(tagged)` can dispatch through `_untag` and the
/// per-iteration fresh-box cache. Pyre's `TraceIterator` still walks
/// a `&[Op]` slice (pre-byte-stream), so this port exposes the
/// read-side surface without coupling to `TraceIterator`: callers that
/// need TAGBOX → fresh-box mapping use a separate `_untag` helper on
/// their `TraceIterator` instance. The non-TAGBOX surface
/// (iter_array, iter_vable_array, iter_vref_array, unpack_jitcode_pc,
/// size, framestack) ports line-by-line.
///
/// `framestack` matches RPython's bottom-up order: outermost frame
/// first, innermost frame last. Empty when the snapshot was written
/// via `create_empty_top_snapshot` (jitcode_index == -1).
pub struct SnapshotIterator<'a> {
    /// opencoder.py:207 self.topdown_snapshot_iter.
    pub topdown_snapshot_iter: TopDownSnapshotIterator<'a>,
    /// opencoder.py:208 `self.vable_array = it.iter_vable_array()`.
    /// Stored as the `_snapshot_array_data` offset from which a fresh
    /// `BoxArrayIter` can be constructed via `iter_vable_array()`.
    pub vable_array_index: i64,
    /// opencoder.py:209 `self.vref_array = it.iter_vref_array()`.
    pub vref_array_index: i64,
    /// opencoder.py:210 `self.size = vable.total_length +
    /// vref.total_length + 3 + sum(length(f) + 2 for f in framestack)`.
    /// Used by resume-data consumers to pre-size output arrays.
    pub size: i64,
    /// opencoder.py:211,214-217 self.framestack — snapshot byte
    /// offsets in bottom-up order (outermost frame first, innermost
    /// frame last), built by reversing the top-down iterator.
    pub framestack: Vec<usize>,
    /// Back-reference to `_snapshot_array_data` so callers can
    /// construct fresh `BoxArrayIter` values without rethreading the
    /// buffer. Matches RPython's implicit `main_iter.trace._snapshot_array_data`
    /// access path.
    snapshot_array_data: &'a [u8],
}

impl<'a> SnapshotIterator<'a> {
    /// opencoder.py:203-217 SnapshotIterator.__init__(main_iter,
    /// snapshot_index).
    pub fn new(
        snapshot_data: &'a [u8],
        snapshot_array_data: &'a [u8],
        snapshot_index: usize,
    ) -> Self {
        let mut it = TopDownSnapshotIterator::new(snapshot_data, snapshot_index);
        let vable_idx = it.vable_array_index;
        let vref_idx = it.vref_array_index;
        let vable_len = BoxArrayIter::new(snapshot_array_data, vable_idx).total_length;
        let vref_len = BoxArrayIter::new(snapshot_array_data, vref_idx).total_length;
        // opencoder.py:210 `size = vable.total_length + vref.total_length + 3`.
        let mut size = vable_len + vref_len + 3;
        let mut framestack = Vec::new();
        // opencoder.py:212-213 early return for empty top snapshot.
        if !it.is_empty_snapshot(snapshot_index) {
            // opencoder.py:214-216 `for snapshot_index in it: ...`
            while let Some(snap_idx) = it.next() {
                framestack.push(snap_idx);
                size += it.length(snap_idx, snapshot_array_data) + 2;
            }
            // opencoder.py:217 `self.framestack.reverse()`.
            framestack.reverse();
        }
        SnapshotIterator {
            topdown_snapshot_iter: it,
            vable_array_index: vable_idx,
            vref_array_index: vref_idx,
            size,
            framestack,
            snapshot_array_data,
        }
    }

    /// opencoder.py:208 `it.iter_vable_array()` accessor — returns a
    /// fresh `BoxArrayIter` rather than caching one iterator (Rust
    /// iterators are single-use).
    pub fn iter_vable_array(&self) -> BoxArrayIter<'a> {
        BoxArrayIter::new(self.snapshot_array_data, self.vable_array_index)
    }

    /// opencoder.py:209 `it.iter_vref_array()` accessor.
    pub fn iter_vref_array(&self) -> BoxArrayIter<'a> {
        BoxArrayIter::new(self.snapshot_array_data, self.vref_array_index)
    }

    /// opencoder.py:219-220 iter_array(snapshot_index).
    pub fn iter_array(&self, snapshot_index: usize) -> BoxArrayIter<'a> {
        self.topdown_snapshot_iter
            .iter_array(snapshot_index, self.snapshot_array_data)
    }

    /// opencoder.py:225-226 unpack_jitcode_pc(snapshot_index).
    pub fn unpack_jitcode_pc(&self, snapshot_index: usize) -> (i64, i64) {
        self.topdown_snapshot_iter.unpack_jitcode_pc(snapshot_index)
    }

    /// opencoder.py:222-223 `get(index)` — resolve a tagged value to
    /// the enclosing iterator's fresh box (OpRef in pyre).
    ///
    /// RPython stores `self.main_iter` on the SnapshotIterator at
    /// construction (opencoder.py:204); pyre's `SnapshotIterator`
    /// does not own a `main_iter` because the legacy
    /// `TraceIterator<'a>` (pre-byte-stream) cannot be constructed
    /// without the `&[Op]` slice it walks, and the byte-stream
    /// `ByteTraceIter` carries its own `ConstantPool` lifetime.
    /// Callers pass the iterator explicitly instead so this helper
    /// stays structurally equivalent to `main_iter._untag(index)`.
    pub fn get(&self, tagged: i64, main_iter: &mut ByteTraceIter<'_>) -> OpRef {
        main_iter._untag(tagged)
    }

    /// opencoder.py:228-231 `unpack_array(arr)` — `[self.get(i) for
    /// i in arr]`.  The RPython comment marks it NOT_RPYTHON (tests
    /// only); pyre exposes the same surface so test fixtures and
    /// debug tooling can read a snapshot's frame array without
    /// re-implementing the decode loop.
    pub fn unpack_array(
        &self,
        arr: BoxArrayIter<'_>,
        main_iter: &mut ByteTraceIter<'_>,
    ) -> Vec<OpRef> {
        arr.map(|tagged| main_iter._untag(tagged)).collect()
    }
}

/// opencoder.py:239-247 update_liveranges(snapshot_index, trace, index,
/// liveranges).
///
/// Walks the snapshot chain starting at the top snapshot at
/// `_snapshot_data[snapshot_index..]` and, for every TAGBOX tagged
/// value in the vable array, the vref array, or any frame's active-
/// box array, writes `liveranges[v] = index`.
///
/// Split into a free function because `get_live_ranges` borrows
/// `self._ops` while simultaneously invoking this; the free-function
/// signature takes the two byte buffers by shared reference so it can
/// coexist with the op-stream walk.
pub fn update_liveranges(
    snapshot_data: &[u8],
    snapshot_array_data: &[u8],
    snapshot_index: usize,
    index: usize,
    liveranges: &mut [usize],
) {
    let mut it = TopDownSnapshotIterator::new(snapshot_data, snapshot_index);
    let vable_idx = it.vable_array_index;
    let vref_idx = it.vref_array_index;
    for tagged in BoxArrayIter::new(snapshot_array_data, vable_idx) {
        update_live_from_tagged(tagged, index, liveranges);
    }
    for tagged in BoxArrayIter::new(snapshot_array_data, vref_idx) {
        update_live_from_tagged(tagged, index, liveranges);
    }
    while let Some(snap_idx) = it.next() {
        for tagged in it.iter_array(snap_idx, snapshot_array_data) {
            update_live_from_tagged(tagged, index, liveranges);
        }
    }
}

/// opencoder.py:234-237 `_update_liverange(item, index, liveranges)`.
#[inline]
fn update_live_from_tagged(tagged: i64, index: usize, liveranges: &mut [usize]) {
    let (tag, v) = untag(tagged as u32);
    if tag == TAGBOX {
        let idx = v as usize;
        if idx < liveranges.len() {
            liveranges[idx] = index;
        }
    }
}

// ── Trace Recording API (opencoder.py: Trace class) ──

/// opencoder.py:603-640 `_encode(box)` — caller-provided view of an arg
/// to `record_op*`. RPython discriminates via `isinstance(box,
/// Const|AbstractResOp)`; pyre needs an explicit tag because Rust
/// lacks inheritance dispatch.
///
/// Variants mirror RPython class names 1:1:
/// * `ConstInt` / `ConstFloat` / `ConstPtr` — `Const` subclasses
///   (`history.py:220/261/307`), payloads match `getint() /
///   getfloatstorage() / getref_base()`.
/// * `ResOp` — RPython `AbstractResOp.get_position()` value. Also
///   covers `AbstractInputArg` because both types expose
///   `get_position()` returning the same `_index` slot (pyre's
///   `OpRef::from_raw(u32)` unifies them).
///
/// Tagged values can also be pre-computed via the free `_encode_*`
/// helpers (`_encode_smallint`, `_encode_bigint`, `_encode_float`,
/// `_encode_ptr`, `_encode_box_position`); callers that already have
/// an `i64` tag (e.g. snapshot writers) use `append_int` directly
/// without going through `Box`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Box {
    /// `history.py:220` `class ConstInt(Const)`. `_encode` auto-
    /// selects the SMALL_INT tag (`SMALL_INT_START..SMALL_INT_STOP`)
    /// vs the big-int pool based on the value.
    ConstInt(i64),
    /// `history.py:261` `class ConstFloat(Const)` —
    /// `getfloatstorage()` raw bits; no dedup.
    ConstFloat(u64),
    /// `history.py:307` `class ConstPtr(Const)` — `getref_base()`
    /// address; `_cached_const_ptr` dedups via `_refs_dict`.
    ConstPtr(u64),
    /// `resoperation.py:233` `AbstractResOpOrInputArg.get_position()`
    /// — the `_index` slot into the trace's op/inputarg namespace.
    ResOp(u32),
}

impl Box {
    /// Convenience: `Box::ResOp(opref.raw())`.
    #[inline]
    pub fn of_op(op: majit_ir::OpRef) -> Self {
        Box::ResOp(op.raw())
    }
}

/// opencoder.py: Trace — compact trace recording buffer.
///
/// Literal port of `rpython/jit/metainterp/opencoder.py::Trace`. The state
/// shape matches RPython 1:1 so the wire format, snapshot chain, and
/// liverange computations can all track RPython behavior. Field names are
/// preserved with a leading underscore where RPython uses one, to make
/// line-by-line porting unambiguous.
///
/// NOTE: methods `record_op*`, `capture_resumedata`, `create_top_snapshot`,
/// `_encode_descr`, `tag_overflow_imminent`, and `tracing_done` are
/// rebuilt across Phase B2–B11 in small sub-phase commits. B1 lands only
/// the state fields and the lowest-level `append_byte` / `append_int` /
/// `_double_ops` / `length` / `cut_point` / `cut_at` helpers that RPython
/// puts next to the struct definition (opencoder.py:468-578).
pub type Trace = TraceRecordBuffer;
pub struct TraceRecordBuffer {
    /// opencoder.py:473 self._ops — pre-allocated byte buffer that records
    /// the operation stream. `_pos` walks forward over it; `_double_ops`
    /// doubles the buffer when `_pos + 4 > len`. The length-of-trace is
    /// `_pos`, NOT `_ops.len()` (opencoder.py:564 length()).
    pub _ops: Vec<u8>,
    /// opencoder.py:475 self._pos — next write position into `_ops`.
    pub _pos: usize,
    /// opencoder.py:497 self._count — total count of ops recorded, seeded
    /// to `max_num_inputargs`. Advances for every op including voids.
    pub _count: u32,
    /// opencoder.py:498 self._index — position of resulting resops, seeded
    /// to `max_num_inputargs`. Advances only for non-void ops (the ones
    /// that produce a box-valued result).
    pub _index: u32,
    /// opencoder.py:499 self._start — frozen baseline equal to
    /// `max_num_inputargs`; iterator construction uses this as the start
    /// cursor (see `get_iter`).
    pub _start: u32,
    /// opencoder.py:496 self.max_num_inputargs — cap on the number of
    /// input-arg boxes. Inputarg positions must stay under this value.
    pub max_num_inputargs: u32,
    /// opencoder.py:491,504 self.inputargs — the actual inputarg boxes
    /// after `set_inputargs` is called. Empty until `set_inputargs` runs.
    pub inputargs: Vec<InputArg>,
    /// opencoder.py:481 self._descrs — `_descrs[0]` is always None; new
    /// descrs append. Indexed by `descr_index - all_descrs_len - 1` when
    /// the descr has no global `get_descr_index` (opencoder.py:702-707).
    pub _descrs: Vec<Option<majit_ir::DescrRef>>,
    /// opencoder.py:482 self._refs — `_refs[0]` is always nullptr (index
    /// 0 reserved for the 0-length snapshot array). Non-null GC refs
    /// append; Rust adaptation roots them on the shadow stack via
    /// `rooted_ref_indices` so a moving GC can update the pointers
    /// in-place (gcreftracer.py parity).
    pub _refs: Vec<u64>,
    /// Rust adaptation: `(_refs index, shadow stack index)` for each
    /// non-null entry pushed through `_encode_ptr`. RPython's `_refs` is
    /// a Python list tracked by the host GC; `Vec<u64>` is not, so we
    /// register each pointer with `majit_gc::shadow_stack` and copy
    /// updated values back via `refresh_from_gc`. Dropped via
    /// `release_roots` on `Drop` so the shadow stack returns to its
    /// creation-time depth.
    rooted_ref_indices: Vec<(usize, usize)>,
    /// Shadow-stack depth captured at `Trace::new` — `release_roots`
    /// pops back to this level.
    shadow_stack_base: usize,
    /// opencoder.py:483 self._refs_dict — caches addr → index into
    /// `_refs`. Cleared by `tracing_done`.
    pub _refs_dict: HashMap<u64, u32>,
    /// opencoder.py:484 self._bigints — constant pool for big ints
    /// (> SMALL_INT_STOP). Indexed via `(idx << 1)` in TAGCONSTOTHER
    /// (bit 0 = 0 means bigint).
    pub _bigints: Vec<i64>,
    /// opencoder.py:485 self._bigints_dict — caches value → index.
    /// Cleared by `tracing_done`.
    pub _bigints_dict: HashMap<i64, u32>,
    /// opencoder.py:486 self._floats — constant pool for floats. Indexed
    /// via `(idx << 1) | 1` in TAGCONSTOTHER (bit 0 = 1 means float).
    pub _floats: Vec<u64>,
    /// opencoder.py:487 self._snapshot_data — byte chain encoding the
    /// per-guard snapshot records (jitcode_index, pc, array_index,
    /// prev) in RPython `encode_varint_signed` format.
    pub _snapshot_data: Vec<u8>,
    /// opencoder.py:488 self._snapshot_array_data — byte chain encoding
    /// the box arrays referenced by snapshots. Index 0 is reserved for
    /// the empty (length 0) array — `append_snapshot_array_data_int(0)`
    /// is called in the constructor (opencoder.py:489).
    pub _snapshot_array_data: Vec<u8>,
    /// opencoder.py:478 self._total_snapshots — monotonic snapshot
    /// counter; bumped in `create_snapshot` / `create_top_snapshot`.
    pub _total_snapshots: u32,
    /// opencoder.py:501 self.tag_overflow — set when `append_int` or
    /// `append_snapshot_*_int` receives a value outside [MIN_VALUE,
    /// MAX_VALUE]. Consulted by `tracing_done` to decide whether to
    /// raise `SwitchToBlackhole(ABORT_TOO_LONG)`.
    pub tag_overflow: bool,
    /// opencoder.py:476-477,479-480 instrumentation counters. `nodict`
    /// tracks `ConstPtrJitCode` entries that hit the opencoder-index
    /// fast path (dict-bypass).
    pub _consts_bigint: u32,
    pub _consts_float: u32,
    pub _consts_ptr: u32,
    pub _consts_ptr_nodict: u32,

    /// Dead-range memoization cache. opencoder.py:469 and :873-883 —
    /// `_deadranges = (-1, None)` means "not computed". Stored as
    /// `Option<(u32, Vec<usize>)>` where the u32 is the `_count` snapshot
    /// at cache time.
    pub _deadranges: Option<(u32, Vec<usize>)>,

    /// opencoder.py:472 `self.metainterp_sd = metainterp_sd`.
    ///
    /// Stores the full static data object so `_encode_descr` and
    /// `TraceIterator` can read `metainterp_sd.all_descrs` (length +
    /// contents) for descriptor index arithmetic. Held as
    /// `Arc<MetaInterpStaticData>` so the Trace can outlive its
    /// creating MetaInterp frame without reshuffling the reference.
    pub metainterp_sd: std::sync::Arc<crate::MetaInterpStaticData>,
}

impl TraceRecordBuffer {
    /// opencoder.py:471-501 `Trace.__init__(self, max_num_inputargs,
    /// metainterp_sd)`.
    pub fn new(
        max_num_inputargs: u32,
        metainterp_sd: std::sync::Arc<crate::MetaInterpStaticData>,
    ) -> Self {
        // opencoder.py:475 `self._pos = 0` initial, then :500 `self._pos
        // = max_num_inputargs` as the final assignment. The first
        // `max_num_inputargs` bytes of `_ops` are reserved as placeholder
        // territory that `TraceIterator.pos = start` walks past —
        // iteration and all write positions operate in a unified
        // [max_num_inputargs, _pos) range.
        let mut t = TraceRecordBuffer {
            _ops: vec![0u8; INIT_SIZE],
            _pos: max_num_inputargs as usize,
            _count: max_num_inputargs,
            _index: max_num_inputargs,
            _start: max_num_inputargs,
            max_num_inputargs,
            inputargs: Vec::new(),
            // opencoder.py:481 — `_descrs = [None]` so index 0 is reserved.
            _descrs: vec![None],
            // opencoder.py:482 — `_refs = [lltype.nullptr(GCREF.TO)]` so
            // index 0 is the null reference / empty-array sentinel.
            _refs: vec![0u64],
            rooted_ref_indices: Vec::new(),
            shadow_stack_base: majit_gc::shadow_stack::depth(),
            _refs_dict: HashMap::new(),
            _bigints: Vec::new(),
            _bigints_dict: HashMap::new(),
            _floats: Vec::new(),
            _snapshot_data: Vec::new(),
            _snapshot_array_data: Vec::new(),
            _total_snapshots: 0,
            tag_overflow: false,
            _consts_bigint: 0,
            _consts_float: 0,
            _consts_ptr: 0,
            _consts_ptr_nodict: 0,
            _deadranges: None,
            metainterp_sd,
        };
        // opencoder.py:489 — `append_snapshot_array_data_int(0)` so all
        // zero-length arrays share index 0.
        t.append_snapshot_array_data_int(0);
        t
    }

    /// opencoder.py:707 `len(self.metainterp_sd.all_descrs)` — read
    /// the global descriptor table length from the attached
    /// metainterp_sd.
    fn all_descrs_len(&self) -> u32 {
        self.metainterp_sd.all_descrs.lock().unwrap().len() as u32
    }

    /// opencoder.py:503-508 set_inputargs(inputargs).
    ///
    /// Records the actual inputarg boxes. Positions must be < max_num_inputargs
    /// (checked in debug mode only, mirroring RPython's `we_are_translated`
    /// guard on the check).
    pub fn set_inputargs(&mut self, inputargs: Vec<InputArg>) {
        debug_assert!(
            inputargs
                .iter()
                .all(|ia| (ia.index as u32) < self.max_num_inputargs),
            "inputarg position exceeds max_num_inputargs"
        );
        self.inputargs = inputargs;
    }

    /// Append a single inputarg, returning its positional OpRef.
    ///
    /// Adapter matching pyre `recorder::Trace::record_input_arg`. Required
    /// because pyre's `setup_tracing` / `start_retrace` build the inputarg
    /// list incrementally from the live-value enumeration, whereas RPython
    /// always hands `set_inputargs` a fully-constructed list built by
    /// `MetaInterp.create_empty_loop`.
    ///
    /// Panics if `max_num_inputargs` would be exceeded — the byte-stream
    /// `_start`/`_count`/`_index` fields are seeded against this cap at
    /// `Trace::new`, so callers must size the cap correctly.
    pub fn record_input_arg(&mut self, tp: Type) -> OpRef {
        let index = self.inputargs.len() as u32;
        assert!(
            index < self.max_num_inputargs,
            "record_input_arg: inputarg count ({}) exceeds max_num_inputargs ({}) — \
             Trace::new's cap was too small",
            index + 1,
            self.max_num_inputargs
        );
        self.inputargs.push(InputArg::from_type(tp, index));
        OpRef::input_arg_typed(index, tp)
    }

    /// Number of inputargs registered so far.
    pub fn num_inputargs(&self) -> usize {
        self.inputargs.len()
    }

    /// Inputarg slice accessor — RPython parity with `Trace.inputargs`.
    pub fn inputargs(&self) -> &[InputArg] {
        &self.inputargs
    }

    /// Inputarg types in loop-header order.
    pub fn inputarg_types(&self) -> Vec<Type> {
        self.inputargs.iter().map(|ia| ia.tp).collect()
    }

    /// history.py:725 `length`: number of non-inputarg ops recorded so far.
    /// = `_count - max_num_inputargs`. Compared against
    /// `warmstate.trace_limit` by `MetaInterp.blackhole_if_trace_too_long`
    /// (pyjitpl.py:2791).
    pub fn num_ops(&self) -> usize {
        (self._count - self.max_num_inputargs) as usize
    }

    // ── M2 Step 2c · read API via ByteTraceIter ────────────────────────
    // RPython has no direct equivalent of these methods: opencoder.py
    // exposes only `get_iter()` and callers iterate. Pyre adds thin
    // walk-to-collect helpers so consumers migrating off
    // `recorder::Trace.ops()` / `get_op_by_pos` / `last_op` can keep the
    // same call shapes; each helper is a single ByteTraceIter walk.
    //
    // The optional `pool` routes constant args (TAGINT / TAGCONSTPTR /
    // TAGCONSTOTHER) through `ConstantPool::get_or_insert_typed`; pass
    // `None` when the buffer is known to contain only TAGBOX args.

    /// Materialize every recorded op in order. O(n) byte walk.
    pub fn ops(&self, pool: Option<&mut ConstantPool>) -> Vec<Op> {
        let mut iter = ByteTraceIter::new_with_pool(
            self,
            self._start as usize,
            self._pos,
            self.max_num_inputargs,
            pool,
        );
        let mut ops = Vec::new();
        while let Some(op) = iter.next() {
            ops.push(op);
        }
        ops
    }

    /// Return the first materialized op whose `.pos == pos`. O(n) byte
    /// walk; `None` if no op claims that position.
    pub fn get_op_by_pos(&self, pos: OpRef, pool: Option<&mut ConstantPool>) -> Option<Op> {
        let mut iter = ByteTraceIter::new_with_pool(
            self,
            self._start as usize,
            self._pos,
            self.max_num_inputargs,
            pool,
        );
        while let Some(op) = iter.next() {
            if op.pos == pos {
                return Some(op);
            }
        }
        None
    }

    /// Return the last recorded op, or `None` if the trace is empty.
    /// O(n) byte walk — opencoder's byte layout has no back-pointer.
    pub fn last_op(&self, pool: Option<&mut ConstantPool>) -> Option<Op> {
        let mut iter = ByteTraceIter::new_with_pool(
            self,
            self._start as usize,
            self._pos,
            self.max_num_inputargs,
            pool,
        );
        let mut last = None;
        while let Some(op) = iter.next() {
            last = Some(op);
        }
        last
    }

    /// opencoder.py:510-511 _double_ops — double the byte buffer when
    /// `_pos` gets close to the end.
    fn _double_ops(&mut self) {
        let new_len = self._ops.len().max(1) * 2;
        self._ops.resize(new_len, 0);
    }

    /// opencoder.py:513-518 append_byte(c) — write a single byte and
    /// advance `_pos`. Doubles the buffer if needed.
    pub fn append_byte(&mut self, c: u8) {
        if self._pos >= self._ops.len() {
            self._double_ops();
        }
        self._ops[self._pos] = c;
        self._pos += 1;
    }

    /// opencoder.py:520-541 append_int(i). Writes a signed varint into
    /// `_ops` at `_pos` using the same 2-or-4 byte layout as
    /// `encode_varint_signed`, but inlined so the buffer-doubling check
    /// happens once (RPython checks `_pos + 4 > len(_ops)` then writes in
    /// place). Out-of-range values trip `tag_overflow` and encode 0.
    pub fn append_int(&mut self, i: i64) {
        let mut v = i;
        if !(MIN_VALUE..=MAX_VALUE).contains(&v) {
            self.tag_overflow = true;
            v = 0;
        }
        if self._pos + 4 > self._ops.len() {
            self._double_ops();
        }
        let flag: u8 = if !(-(1 << 14)..(1 << 14)).contains(&v) {
            0x80
        } else {
            0
        };
        self._ops[self._pos] = ((v & 0b0111_1111) as u8) | flag;
        self._pos += 1;
        v >>= 7;
        self._ops[self._pos] = (v & 0xff) as u8;
        self._pos += 1;
        if flag != 0 {
            v >>= 8;
            self._ops[self._pos] = (v & 0xff) as u8;
            self._pos += 1;
            v >>= 8;
            self._ops[self._pos] = (v & 0xff) as u8;
            self._pos += 1;
        }
    }

    /// opencoder.py:543-544 tag_overflow_imminent — returns true once
    /// the trace byte stream has grown past 80% of `MAX_VALUE`.
    ///
    /// Checked by the tracer so it can force-close the trace (via the
    /// segmented-trace path) BEFORE any `append_int` actually trips
    /// `tag_overflow`. `tracing_done()` converts `tag_overflow == true`
    /// into `AbortReason::TooLong`, which lands the caller in the
    /// blackhole — calling this `imminent` check gives a graceful exit
    /// ahead of that hard abort.
    pub fn tag_overflow_imminent(&self) -> bool {
        (self._pos as i64) > (MAX_VALUE as f64 * 0.8) as i64
    }

    /// opencoder.py:564-565 length — the encoded trace length is `_pos`,
    /// NOT `_ops.len()` (the buffer is pre-allocated with padding).
    pub fn length(&self) -> usize {
        self._pos
    }

    /// opencoder.py:567-568 `cut_point` — 5-tuple `(_pos, _count, _index,
    /// len(_snapshot_data), len(_snapshot_array_data))` returned as a
    /// `TracePosition` struct so `cut_at` can restore all five fields
    /// atomically (RPython uses a bare tuple; the Rust struct carries the
    /// same shape plus field names).
    pub fn cut_point(&self) -> crate::recorder::TracePosition {
        crate::recorder::TracePosition {
            _pos: self._pos,
            _count: self._count,
            _index: self._index,
            snapshot_data_len: self._snapshot_data.len(),
            snapshot_array_data_len: self._snapshot_array_data.len(),
        }
    }

    /// opencoder.py:570-575 `cut_at(end)` — restore the first three
    /// fields of the cut_point tuple. The snapshot lengths are
    /// intentionally NOT truncated in RPython (they grow monotonically
    /// for a single trace; bridge compilation handles snapshot reuse
    /// separately).
    pub fn cut_at(&mut self, end: crate::recorder::TracePosition) {
        self._pos = end._pos;
        self._count = end._count;
        // opencoder.py:574 `assert index >= 0` — guarded here by the
        // `u32` type of `_index`.
        self._index = end._index;
        // `end.snapshot_data_len` / `end.snapshot_array_data_len` are
        // ignored by `cut_at` in RPython — snapshot chains grow
        // monotonically for a single trace. Parity preserved.
    }

    /// opencoder.py:577-578 `cut_trace_from((start, count, index, _, _),
    /// inputargs)` — produces a `CutTrace` view over this trace's byte
    /// stream that iterates from the cut_point onward, treating
    /// `inputargs` as the new iterator-local inputargs.
    ///
    /// Used by bridge compilation: when a guard fails at some cut_point,
    /// the tracer resumes a new trace from that point, with the
    /// guard's fail_args (which reference ops recorded before the cut)
    /// acting as fresh inputargs for the resumed trace.
    pub fn cut_trace_from(
        &self,
        cut: crate::recorder::TracePosition,
        inputargs: Vec<OpRef>,
    ) -> CutTrace<'_> {
        CutTrace {
            trace: self as *const _ as *mut _,
            start: cut._pos,
            count: cut._count,
            index: cut._index,
            inputargs,
            _marker: std::marker::PhantomData,
        }
    }

    /// opencoder.py:546-562 tracing_done() — finalize the trace.
    ///
    /// Returns `Err(AbortReason::TooLong)` if `tag_overflow` was set
    /// while recording — any `append_int` that saw a value outside
    /// `[MIN_VALUE, MAX_VALUE]` trips it (opencoder.py:521-522). This
    /// mirrors RPython's `raise SwitchToBlackhole(Counters.ABORT_TOO_LONG)`
    /// path (opencoder.py:548-549) and lets the caller route into the
    /// blackhole.
    ///
    /// On success, clears the `_bigints_dict` / `_refs_dict` dedup
    /// tables (opencoder.py:550-551) — once tracing is done, no further
    /// constants will be interned and the encode hot-path dictionaries
    /// can be reclaimed. The stored `_bigints` / `_refs` Vecs are kept
    /// so `TraceIterator` can still resolve encoded constants back to
    /// values. RPython also emits debug-log lines through
    /// `debug_start`/`debug_print`/`debug_stop`; pyre skips that —
    /// `MAJIT_LOG` covers equivalent telemetry from other hooks.
    pub fn tracing_done(&mut self) -> Result<(), crate::pyjitpl::AbortReason> {
        if self.tag_overflow {
            return Err(crate::pyjitpl::AbortReason::TooLong);
        }
        self._bigints_dict.clear();
        self._refs_dict.clear();
        Ok(())
    }

    // ── Box encoding (opencoder.py:603-640 _encode + _cached_const_*) ──

    /// opencoder.py:603-608 _encode for ConstInt that fits in SMALL_INT
    /// range — return `tag(TAGINT, value)`.
    ///
    /// Values outside [SMALL_INT_START, SMALL_INT_STOP) must route
    /// through `_encode_bigint` because the signed varint can't pack
    /// them in the TAGINT namespace.
    pub fn _encode_smallint(value: i64) -> i64 {
        debug_assert!(
            (SMALL_INT_START..SMALL_INT_STOP).contains(&value),
            "value {value} out of SMALL_INT range — use _encode_bigint"
        );
        // RPython `tag(TAGINT, value)` evaluates with Python's
        // arbitrary-precision signed ints — `(-1) << 2 | 0 == -4`.
        // The `u32`-based `tag()` helper works for snapshot array
        // positions where values are always non-negative, but
        // SMALL_INT values are signed and `value as u32` zero-extends
        // negatives into a 32-bit positive that exceeds MAX_VALUE on
        // `append_int`. Use i64 arithmetic so the sign survives the
        // encode/decode round trip — `_untag`'s `tagged >> TAG_SHIFT`
        // is already an i64 arithmetic shift.
        (value << TAG_SHIFT) | (TAGINT as i64)
    }

    /// opencoder.py:609-622 _encode for ConstInt that does NOT fit in
    /// SMALL_INT range — interns into `_bigints` (dedup via
    /// `_bigints_dict`) and returns `tag(TAGCONSTOTHER, (idx << 1))`.
    /// Bit 0 of `idx << 1` is 0 → bigint; bit 0 == 1 → float.
    pub fn _encode_bigint(&mut self, value: i64) -> i64 {
        self._consts_bigint += 1;
        let v = if let Some(&idx) = self._bigints_dict.get(&value) {
            idx as u64
        } else {
            let idx = self._bigints.len() as u64;
            self._bigints.push(value);
            self._bigints_dict.insert(value, idx as u32);
            idx
        };
        tag(TAGCONSTOTHER, (v << 1) as u32) as i64
    }

    /// opencoder.py:623-628 _encode for ConstFloat — no caching; appends
    /// to `_floats` and returns `tag(TAGCONSTOTHER, (idx << 1) | 1)`.
    /// Bit 0 == 1 flags the float path on decode.
    pub fn _encode_float(&mut self, raw: u64) -> i64 {
        self._consts_float += 1;
        let idx = self._floats.len() as u32;
        self._floats.push(raw);
        tag(TAGCONSTOTHER, (idx << 1) | 1) as i64
    }

    /// opencoder.py:583-601 _cached_const_ptr + :629-632 _encode for
    /// ConstPtr — dedup via `_refs_dict` (by address), push to `_refs`,
    /// return `tag(TAGCONSTPTR, idx)`. Index 0 is reserved for nullptr
    /// (seeded by the constructor).
    ///
    /// Rust adaptation: non-null `addr` is registered on the shadow
    /// stack so a moving GC can update the entry in-place. RPython's
    /// `_refs` list is natively GC-tracked; `Vec<u64>` is not.
    pub fn _encode_ptr(&mut self, addr: u64) -> i64 {
        self._consts_ptr += 1;
        if addr == 0 {
            return tag(TAGCONSTPTR, 0) as i64;
        }
        let v = if let Some(&idx) = self._refs_dict.get(&addr) {
            idx
        } else {
            let idx = self._refs.len() as u32;
            self._refs.push(addr);
            let ss_idx = majit_gc::shadow_stack::push(majit_ir::GcRef(addr as usize));
            self.rooted_ref_indices.push((idx as usize, ss_idx));
            self._refs_dict.insert(addr, idx);
            idx
        };
        tag(TAGCONSTPTR, v) as i64
    }

    /// opencoder.py:633-638 _encode for AbstractResOp — boxes that live
    /// in the trace's own op stream. Position is the box's `_index`
    /// slot from record time.
    pub fn _encode_box_position(position: u32) -> i64 {
        tag(TAGBOX, position) as i64
    }

    /// opencoder.py:603-640 `_encode(box)` — dispatch a `Box` to the
    /// appropriate tagging helper and return the encoded `i64` tag.
    ///
    /// Mirrors RPython's `isinstance(box, Const|AbstractResOp)` chain.
    /// `ConstInt` further splits on the SMALL_INT range
    /// (opencoder.py:605-608 vs 609-622).
    pub fn _encode(&mut self, b: Box) -> i64 {
        match b {
            // opencoder.py:605-608 ConstInt within SMALL_INT range.
            Box::ConstInt(v) if (SMALL_INT_START..SMALL_INT_STOP).contains(&v) => {
                Self::_encode_smallint(v)
            }
            // opencoder.py:609-622 ConstInt outside SMALL_INT range.
            Box::ConstInt(v) => self._encode_bigint(v),
            // opencoder.py:623-628 ConstFloat.
            Box::ConstFloat(raw) => self._encode_float(raw),
            // opencoder.py:629-632 ConstPtr.
            Box::ConstPtr(addr) => self._encode_ptr(addr),
            // opencoder.py:633-638 AbstractResOp.get_position().
            Box::ResOp(p) => Self::_encode_box_position(p),
        }
    }

    // ── Snapshot writers (opencoder.py:712-817) ──

    /// opencoder.py:728-733 new_array(lgt).
    ///
    /// Length-0 arrays always return index 0 (the pre-seeded empty
    /// entry from the constructor). Non-zero lengths write the length
    /// as the first varint at the tail of `_snapshot_array_data` and
    /// return the offset before the write; callers then write the
    /// array contents via `_add_box_to_storage`.
    pub fn new_array(&mut self, lgt: usize) -> i64 {
        if lgt == 0 {
            return 0;
        }
        let res = self._snapshot_array_data.len() as i64;
        self.append_snapshot_array_data_int(lgt as i64);
        res
    }

    /// opencoder.py:735-736 _add_box_to_storage(tagged_box).
    ///
    /// Appends a pre-encoded tagged box value to `_snapshot_array_data`.
    /// The caller computes the tag via the `_encode_*` helpers above.
    pub fn _add_box_to_storage(&mut self, tagged: i64) {
        self.append_snapshot_array_data_int(tagged);
    }

    /// opencoder.py:735-736 `_add_box_to_storage(box)` — RPython takes
    /// the Box object directly and runs `self._encode(box)` inline.
    /// `_add_box_to_storage` (above) is the pyre-adapter tagged-int
    /// path used by callers that already have a tagged value; this
    /// method matches the RPython call shape for callers that hold a
    /// `Box` (e.g. `MIFrame::get_list_of_active_boxes`).
    pub fn _add_box_to_storage_box(&mut self, b: Box) {
        let tagged = self._encode(b);
        self.append_snapshot_array_data_int(tagged);
    }

    /// opencoder.py:712-716 _list_of_boxes(boxes). Writes an array
    /// header then each box, returns the array index (0 for empty).
    pub fn _list_of_boxes(&mut self, boxes: &[i64]) -> i64 {
        let res = self.new_array(boxes.len());
        for &b in boxes {
            self._add_box_to_storage(b);
        }
        res
    }

    /// opencoder.py:718-726 _list_of_boxes_virtualizable(boxes).
    ///
    /// The virtualizable lives at the end of the locals_cells_stack in
    /// the tracer's view, but the snapshot encoding wants it at the
    /// FRONT so the resume reader can pull it out before the local
    /// frame slots. This reorders `[a, b, c, vable]` to `[vable, a, b, c]`.
    pub fn _list_of_boxes_virtualizable(&mut self, boxes: &[i64]) -> i64 {
        if boxes.is_empty() {
            return self.new_array(0);
        }
        let res = self.new_array(boxes.len());
        // boxes[-1] first (the virtualizable), then boxes[:-1].
        self._add_box_to_storage(*boxes.last().unwrap());
        for &b in &boxes[..boxes.len() - 1] {
            self._add_box_to_storage(b);
        }
        res
    }

    /// opencoder.py:750-765 _encode_snapshot(index, pc, array, is_last).
    ///
    /// Writes jitcode_index + pc + array_index + prev-sentinel into
    /// `_snapshot_data`. `prev` is either `SNAPSHOT_PREV_NONE` for the
    /// last snapshot in a chain (end of framestack) or
    /// `SNAPSHOT_PREV_NEEDS_PATCHING` for internal snapshots — those
    /// get patched later via `snapshot_add_prev`.
    ///
    /// Returns the snapshot_index (offset into `_snapshot_data` before
    /// the write).
    pub fn _encode_snapshot(&mut self, index: i64, pc: i64, array: i64, is_last: bool) -> i64 {
        let res = self._snapshot_data.len() as i64;
        self.append_snapshot_data_int(index);
        self.append_snapshot_data_int(pc);
        self.append_snapshot_data_int(array);
        let prev = if is_last {
            SNAPSHOT_PREV_NONE as i64
        } else {
            SNAPSHOT_PREV_NEEDS_PATCHING as i64
        };
        self.append_snapshot_data_int(prev);
        res
    }

    /// opencoder.py:812-817 snapshot_add_prev(prev).
    ///
    /// Overwrites the trailing prev-sentinel slot in `_snapshot_data`
    /// (written as `SNAPSHOT_PREV_NEEDS_PATCHING` by the previous
    /// `_encode_snapshot`) with the actual prev index. Called from
    /// `create_snapshot` (SNAPSHOT_PREV_COMES_NEXT) and
    /// `_ensure_parent_resumedata` (the real parent snapshot index).
    ///
    /// INVARIANT: the last 2 bytes of `_snapshot_data` must be the
    /// signed-varint encoding of `SNAPSHOT_PREV_NEEDS_PATCHING` (-3).
    /// `-3` fits in 2 bytes (RPython literally asserts the bytes
    /// `'}' '\xff'` at opencoder.py:813-814). We assert the same.
    pub fn snapshot_add_prev(&mut self, prev: i32) {
        debug_assert!(self._snapshot_data.len() >= 2, "snapshot_data too short");
        let n = self._snapshot_data.len();
        debug_assert_eq!(
            self._snapshot_data[n - 2],
            b'}',
            "prev slot byte 0 not \\x7d (NEEDS_PATCHING marker)"
        );
        debug_assert_eq!(
            self._snapshot_data[n - 1],
            0xffu8,
            "prev slot byte 1 not \\xff (NEEDS_PATCHING marker)"
        );
        self._snapshot_data.pop();
        self._snapshot_data.pop();
        self.append_snapshot_data_int(prev as i64);
    }

    // ── Snapshot chain entry points (opencoder.py:767-843) ──

    /// opencoder.py:767-785 create_top_snapshot(frame, vable_boxes,
    /// vref_boxes, after_residual_call, is_last).
    ///
    /// The caller has already computed the active-boxes array via
    /// `new_array` + `_add_box_to_storage` calls (that's the `array`
    /// parameter here — RPython does this inline through the frame's
    /// `get_list_of_active_boxes` callback). vable_boxes / vref_boxes
    /// are pre-encoded tagged values; this writes their arrays into
    /// `_snapshot_array_data`, then the top snapshot record into
    /// `_snapshot_data`, and finally patches the preceding guard's
    /// descr slot to the snapshot index.
    pub fn create_top_snapshot(
        &mut self,
        jitcode_index: i64,
        pc: i64,
        array: i64,
        vable_boxes: &[i64],
        vref_boxes: &[i64],
        is_last: bool,
    ) -> i64 {
        self._total_snapshots += 1;
        let s = self._snapshot_data.len() as i64;
        let vable_array = self._list_of_boxes_virtualizable(vable_boxes);
        let vref_array = self._list_of_boxes(vref_boxes);
        self.append_snapshot_data_int(vable_array);
        self.append_snapshot_data_int(vref_array);
        self._encode_snapshot(jitcode_index, pc, array, is_last);
        // Patch the guard's trailing descr slot to the snapshot index.
        self.patch_last_guard_descr_slot(s);
        s
    }

    /// opencoder.py:787-804 create_empty_top_snapshot(vable_boxes, vref_boxes).
    ///
    /// No frame data — used when tracing starts at a guard before any
    /// frame has been entered. Writes a snapshot with jitcode_index=-1,
    /// pc=0, an empty box array, and SNAPSHOT_PREV_NONE.
    pub fn create_empty_top_snapshot(&mut self, vable_boxes: &[i64], vref_boxes: &[i64]) -> i64 {
        self._total_snapshots += 1;
        let s = self._snapshot_data.len() as i64;
        let empty_array = self._list_of_boxes(&[]);
        let vable_array = self._list_of_boxes_virtualizable(vable_boxes);
        let vref_array = self._list_of_boxes(vref_boxes);
        self.append_snapshot_data_int(vable_array);
        self.append_snapshot_data_int(vref_array);
        self._encode_snapshot(-1, 0, empty_array, true);
        self.patch_last_guard_descr_slot(s);
        s
    }

    /// opencoder.py:806-810 create_snapshot(frame, is_last).
    ///
    /// Called while walking the framestack toward the outermost frame
    /// to capture parent resumedata. Writes a `SNAPSHOT_PREV_COMES_NEXT`
    /// sentinel on the PREVIOUS snapshot (via `snapshot_add_prev`),
    /// then encodes this snapshot with `SNAPSHOT_PREV_NEEDS_PATCHING`
    /// (to be resolved on the NEXT iteration).
    pub fn create_snapshot(
        &mut self,
        jitcode_index: i64,
        pc: i64,
        array: i64,
        is_last: bool,
    ) -> i64 {
        self._total_snapshots += 1;
        self.snapshot_add_prev(SNAPSHOT_PREV_COMES_NEXT);
        self._encode_snapshot(jitcode_index, pc, array, is_last)
    }

    /// opencoder.py:337-338 `TraceIterator.get_snapshot_iter(index)`
    /// — pyre hosts this on the Trace itself since the pre-byte-stream
    /// `TraceIterator` does not yet hold a `&TraceRecordBuffer`.
    ///
    /// Returns an eager `SnapshotIterator` that has already reversed
    /// the snapshot chain rooted at `snapshot_index` and captured the
    /// vable / vref array offsets. `snapshot_index` is the byte offset
    /// returned by `capture_resumedata` / `create_top_snapshot`.
    pub fn get_snapshot_iter(&self, snapshot_index: usize) -> SnapshotIterator<'_> {
        SnapshotIterator::new(
            &self._snapshot_data,
            &self._snapshot_array_data,
            snapshot_index,
        )
    }

    /// opencoder.py:767-785 `create_top_snapshot(frame, vable_boxes,
    /// vref_boxes, after_residual_call, is_last)` — frame-based
    /// wrapper that calls `frame.get_list_of_active_boxes` to produce
    /// the innermost-frame box array, then emits the top snapshot
    /// record.
    ///
    /// Matches the RPython call shape 1:1: the caller provides the
    /// `frame` and virtualizable / virtualref box arrays; the trace
    /// takes care of writing the active-box array, vable / vref
    /// arrays, and the snapshot record header. `op_live` and
    /// `all_liveness` are threaded through from the caller's
    /// `MetaInterpStaticData` (RPython reads them off
    /// `self.metainterp_sd` inline; pyre passes explicitly to avoid
    /// an overlapping borrow on `metainterp_sd` during the write).
    ///
    /// `in_a_call` is forced to `false` at the topmost frame — the
    /// RPython invariant at opencoder.py:769.
    pub fn create_top_snapshot_from_frame(
        &mut self,
        frame: &mut crate::pyjitpl::MIFrame,
        vable_boxes: &[Box],
        vref_boxes: &[Box],
        op_live: u8,
        all_liveness: &[u8],
        after_residual_call: bool,
        is_last: bool,
    ) -> i64 {
        self._total_snapshots += 1;
        // opencoder.py:769 frame.get_list_of_active_boxes(False, ...).
        // The topmost frame always uses `in_a_call=false`, so the
        // ConstantPool is unused — pass None.
        let array = frame.get_list_of_active_boxes(
            /* in_a_call */ false,
            self,
            /* pool */ None,
            op_live,
            all_liveness,
            after_residual_call,
        );
        // opencoder.py:771-780 — write vable / vref arrays, then the
        // snapshot record. `s` captures the snapshot_data offset
        // before any writes so `patch_last_guard_descr_slot` can
        // patch the preceding guard's 0-placeholder slot.
        let s = self._snapshot_data.len() as i64;
        let vable_array = self._list_of_boxes_virtualizable_from_boxes(vable_boxes);
        let vref_array = self._list_of_boxes_from_boxes(vref_boxes);
        self.append_snapshot_data_int(vable_array);
        self.append_snapshot_data_int(vref_array);
        let jitcode_index = frame.jitcode.try_index().map(|i| i as i64).unwrap_or(-1);
        let pc = frame.pc as i64;
        self._encode_snapshot(jitcode_index, pc, array, is_last);
        self.patch_last_guard_descr_slot(s);
        s
    }

    /// opencoder.py:806-810 `create_snapshot(frame, is_last)` —
    /// frame-based wrapper for parent frames in a chain. Writes
    /// `SNAPSHOT_PREV_COMES_NEXT` on the previous snapshot (via
    /// `snapshot_add_prev`), then emits this snapshot with
    /// `SNAPSHOT_PREV_NEEDS_PATCHING`.
    ///
    /// `in_a_call=true` is forced on parent frames (pyjitpl.py:808):
    /// the paired `get_list_of_active_boxes` clears the in-flight
    /// CALL's result register to a zero constant via the supplied
    /// ConstantPool, mirroring RPython's `self.registers_i[index] =
    /// history.CONST_FALSE` mutation line-by-line.  When `pool` is
    /// `None` (test fixtures that do not wire a ConstantPool) the
    /// fallback substitutes `Box::Const*(0)` directly into the
    /// snapshot bytes — identical encoded output but the register
    /// slot keeps its pre-call contents.
    pub fn create_snapshot_from_frame(
        &mut self,
        frame: &mut crate::pyjitpl::MIFrame,
        pool: Option<&mut ConstantPool>,
        op_live: u8,
        all_liveness: &[u8],
        is_last: bool,
    ) -> i64 {
        self._total_snapshots += 1;
        let array = frame.get_list_of_active_boxes(
            /* in_a_call */ true,
            self,
            pool,
            op_live,
            all_liveness,
            /* after_residual_call */ false,
        );
        self.snapshot_add_prev(SNAPSHOT_PREV_COMES_NEXT);
        let jitcode_index = frame.jitcode.try_index().map(|i| i as i64).unwrap_or(-1);
        let pc = frame.pc as i64;
        self._encode_snapshot(jitcode_index, pc, array, is_last)
    }

    /// opencoder.py:819-832 `capture_resumedata(framestack,
    /// virtualizable_boxes, virtualref_boxes, after_residual_call)`.
    ///
    /// Top-level resumedata capture: emits the innermost frame's top
    /// snapshot, then walks parent frames via
    /// `_ensure_parent_resumedata`. Returns the snapshot_index of
    /// the top snapshot (matches RPython `return result`).
    pub fn capture_resumedata(
        &mut self,
        framestack: &mut [crate::pyjitpl::MIFrame],
        virtualizable_boxes: &[Box],
        virtualref_boxes: &[Box],
        pool: Option<&mut ConstantPool>,
        op_live: u8,
        all_liveness: &[u8],
        after_residual_call: bool,
    ) -> i64 {
        // opencoder.py:820 `n = len(framestack) - 1`.
        let framestack_len = framestack.len();
        if framestack_len >= 1 {
            let n = framestack_len - 1;
            // opencoder.py:822 `top = framestack[n]`. Split the
            // framestack so the topmost frame can be borrowed
            // mutably while `_ensure_parent_resumedata` later walks
            // parents.
            let result = {
                let top = &mut framestack[n];
                self.create_top_snapshot_from_frame(
                    top,
                    virtualizable_boxes,
                    virtualref_boxes,
                    op_live,
                    all_liveness,
                    after_residual_call,
                    /* is_last */ n == 0,
                )
            };
            // opencoder.py:828 self._ensure_parent_resumedata(framestack, n).
            // The parent walk uses `in_a_call=true`, so the pool is
            // threaded through to the per-frame register-clear path.
            self._ensure_parent_resumedata(framestack, n, pool, op_live, all_liveness);
            result
        } else {
            // opencoder.py:829-831 — empty framestack → empty top
            // snapshot.
            let _ = pool;
            self.create_empty_top_snapshot_from_boxes(virtualizable_boxes, virtualref_boxes)
        }
    }

    /// opencoder.py:834-843 `_ensure_parent_resumedata(framestack, n)`.
    ///
    /// Walks parent frames backwards from `n-1` down to `0`, emitting
    /// a `create_snapshot` for each. Shortcuts via
    /// `frame.parent_snapshot` memo: if a parent frame has already
    /// produced its snapshot (e.g. re-entrancy, or a prior
    /// `capture_resumedata` in the same trace), patch the previous
    /// snapshot's prev-pointer directly to `frame.parent_snapshot`
    /// and return. Otherwise emit a fresh snapshot and memo it on
    /// the frame.
    pub fn _ensure_parent_resumedata(
        &mut self,
        framestack: &mut [crate::pyjitpl::MIFrame],
        n: usize,
        mut pool: Option<&mut ConstantPool>,
        op_live: u8,
        all_liveness: &[u8],
    ) {
        let mut n = n;
        while n > 0 {
            // opencoder.py:836-840 memo short-circuit via target frame.
            let target_parent = framestack[n].parent_snapshot;
            if target_parent >= 0 {
                self.snapshot_add_prev(target_parent as i32);
                return;
            }
            // opencoder.py:841 s = self.create_snapshot(back, is_last).
            let is_last = n == 1;
            let s = {
                let back = &mut framestack[n - 1];
                // Re-borrow `pool` per iteration so the &mut threads
                // through the loop without being consumed.
                self.create_snapshot_from_frame(
                    back,
                    pool.as_deref_mut(),
                    op_live,
                    all_liveness,
                    is_last,
                )
            };
            // opencoder.py:842 `target.parent_snapshot = s`.
            framestack[n].parent_snapshot = s;
            n -= 1;
        }
    }

    /// opencoder.py:712-716 `_list_of_boxes(boxes)` — Box-taking
    /// sibling of the tagged-int `_list_of_boxes`. Used by
    /// `capture_resumedata` and friends which hold `Box` values.
    fn _list_of_boxes_from_boxes(&mut self, boxes: &[Box]) -> i64 {
        let res = self.new_array(boxes.len());
        for &b in boxes {
            self._add_box_to_storage_box(b);
        }
        res
    }

    /// opencoder.py:718-726 `_list_of_boxes_virtualizable(boxes)` —
    /// Box-taking sibling; reorders `[a, b, c, vable]` to
    /// `[vable, a, b, c]` at encode time.
    fn _list_of_boxes_virtualizable_from_boxes(&mut self, boxes: &[Box]) -> i64 {
        if boxes.is_empty() {
            return self.new_array(0);
        }
        let res = self.new_array(boxes.len());
        self._add_box_to_storage_box(*boxes.last().unwrap());
        for &b in &boxes[..boxes.len() - 1] {
            self._add_box_to_storage_box(b);
        }
        res
    }

    /// opencoder.py:787-804 `create_empty_top_snapshot(vable_boxes,
    /// vref_boxes)` — Box-taking sibling for `capture_resumedata`'s
    /// empty-framestack branch.
    fn create_empty_top_snapshot_from_boxes(
        &mut self,
        vable_boxes: &[Box],
        vref_boxes: &[Box],
    ) -> i64 {
        self._total_snapshots += 1;
        let s = self._snapshot_data.len() as i64;
        let empty_array = self._list_of_boxes_from_boxes(&[]);
        let vable_array = self._list_of_boxes_virtualizable_from_boxes(vable_boxes);
        let vref_array = self._list_of_boxes_from_boxes(vref_boxes);
        self.append_snapshot_data_int(vable_array);
        self.append_snapshot_data_int(vref_array);
        self._encode_snapshot(-1, 0, empty_array, true);
        self.patch_last_guard_descr_slot(s);
        s
    }

    /// opencoder.py:781-784 `_pos -= 2; self.append_int(s)` — patch the
    /// guard's trailing descr slot to the snapshot index.
    ///
    /// INVARIANT: guards record their trailing descr via `append_int(0)`
    /// (see `_op_end` + opencoder.py:653-657). `0` always encodes as
    /// exactly 2 bytes in the RPython signed-varint layout (both bytes
    /// are `\x00`), so `capture_resumedata` / `create_top_snapshot` can
    /// safely rewind `_pos -= 2` and overwrite the slot with the actual
    /// snapshot_index via a fresh `append_int(s)`. The new value may
    /// occupy 2 *or* 4 bytes; `_ops` is pre-grown by `_double_ops` so
    /// both sizes fit.
    ///
    /// RPython inlines this sequence in `create_top_snapshot` and
    /// `create_snapshot`; pyre factors it out so Phase B7's snapshot
    /// code is shorter and the assertion that the slot was indeed a
    /// guard 0-placeholder is always executed.
    pub fn patch_last_guard_descr_slot(&mut self, snapshot_index: i64) {
        debug_assert!(
            self._pos >= 2,
            "patch_last_guard_descr_slot called with _pos < 2"
        );
        debug_assert_eq!(
            self._ops[self._pos - 2],
            0u8,
            "guard descr placeholder byte 0 was not \\x00"
        );
        debug_assert_eq!(
            self._ops[self._pos - 1],
            0u8,
            "guard descr placeholder byte 1 was not \\x00"
        );
        self._pos -= 2;
        self.append_int(snapshot_index);
    }

    /// opencoder.py:702-707 `_encode_descr(descr)`.
    ///
    /// Two-tier descr numbering mirroring RPython:
    ///   * Global descrs — `descr.get_descr_index() >= 0`. Encoded as
    ///     `global_index + 1`. Index range `[1, all_descrs_len + 1)`.
    ///   * Local descrs — `get_descr_index() == -1`. Appended to
    ///     `self._descrs` (which starts with a `None` sentinel at [0]
    ///     so index 0 means "no descr") and encoded as
    ///     `all_descrs_len + len(_descrs) - 1 + 1`. The TraceIterator
    ///     decode path looks up such descrs via
    ///     `_descrs[descr_index - all_descr_len - 1]`.
    ///
    /// Reads `len(self.metainterp_sd.all_descrs)` at encode time —
    /// this field is populated by `Trace::new` (opencoder.py:471
    /// `Trace.__init__(max_num_inputargs, metainterp_sd)`).
    pub fn _encode_descr(&mut self, descr: &majit_ir::DescrRef) -> i64 {
        let descr_index = descr.get_descr_index();
        if descr_index >= 0 {
            return (descr_index as i64) + 1;
        }
        self._descrs.push(Some(descr.clone()));
        // _descrs[0] is the sentinel None; new descrs append from
        // index 1. RPython returns `len(_descrs) - 1 + all_descrs_len
        // + 1`. `len - 1` because index 0 is reserved; we want the
        // local-only offset.
        let local_index = (self._descrs.len() as i64) - 1;
        local_index + (self.all_descrs_len() as i64) + 1
    }

    /// opencoder.py:642-650 _op_start(opnum, num_argboxes).
    ///
    /// Writes the opnum byte. For fixed-arity ops the decoder reads the
    /// arity from `OpCode::arity()`; for variadic ops (arity == None)
    /// the writer records the actual `num_argboxes` count inline as a
    /// signed varint. Returns the old `_pos` so `_op_end` can patch the
    /// trailing descr int if the opnum turns out to be a guard that gets
    /// its snapshot index filled in later.
    pub fn _op_start(&mut self, opcode: OpCode, num_argboxes: usize) -> usize {
        let old_pos = self._pos;
        let opnum = opcode.as_u16();
        debug_assert!(opnum <= 0xFF, "opnum {opnum} exceeds 1 byte");
        self.append_byte(opnum as u8);
        match opcode.arity() {
            None => {
                self.append_int(num_argboxes as i64);
            }
            Some(expected) => {
                debug_assert_eq!(
                    num_argboxes, expected as usize,
                    "fixed-arity mismatch for {:?}: expected {expected}, got {num_argboxes}",
                    opcode
                );
            }
        }
        old_pos
    }

    /// opencoder.py:652-662 _op_end(opnum, descr, old_pos).
    ///
    /// Writes the trailing descr int if the opcode carries one
    /// (`has_descr()`). `descr_index` is the already-encoded descr slot
    /// from `_encode_descr` (Phase B4); guards pass 0 to get a
    /// placeholder that `capture_resumedata` patches to the snapshot
    /// index (Phase B5/B7). Bumps `_count` unconditionally and `_index`
    /// only for non-void ops (opencoder.py:661).
    pub fn _op_end(&mut self, opcode: OpCode, descr_index: i64, _old_pos: usize) {
        if opcode.has_descr() {
            self.append_int(descr_index);
        }
        self._count += 1;
        if opcode.result_type() != Type::Void {
            self._index += 1;
        }
    }

    /// opencoder.py:652-662 `_op_end` (descr-object overload).
    ///
    /// Drops the RPython `descr=None` branch (caller decides: guards and
    /// no-descr ops use `None`, which encodes the 0-placeholder; all
    /// other descrs are routed through `_encode_descr` first). Keeps the
    /// `_count` / `_index` bump parity with `_op_end`.
    fn _op_end_descr(
        &mut self,
        opcode: OpCode,
        descr: Option<&majit_ir::DescrRef>,
        old_pos: usize,
    ) {
        let descr_index = match descr {
            Some(d) => self._encode_descr(d),
            None => 0,
        };
        self._op_end(opcode, descr_index, old_pos);
    }

    /// opencoder.py:664-670 record_op(opnum, argboxes, descr=None).
    ///
    /// Returns the pre-bump `_index` — the box position that THIS op's
    /// result will occupy if it is non-void (opencoder.py:665 `pos =
    /// self._index`). Each `Box` is passed through `_encode` to
    /// produce the wire-format tagged value. `descr=None` emits the
    /// 0-placeholder (guards and no-descr ops); `Some(&descr)` routes
    /// through `_encode_descr`.
    pub fn record_op(
        &mut self,
        opcode: OpCode,
        argboxes: &[Box],
        descr: Option<&majit_ir::DescrRef>,
    ) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, argboxes.len());
        for &b in argboxes {
            let tagged = self._encode(b);
            self.append_int(tagged);
        }
        self._op_end_descr(opcode, descr, old_pos);
        pos
    }

    /// opencoder.py:672-676 record_op0(opnum, descr=None).
    pub fn record_op0(&mut self, opcode: OpCode, descr: Option<&majit_ir::DescrRef>) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, 0);
        self._op_end_descr(opcode, descr, old_pos);
        pos
    }

    /// opencoder.py:678-683 record_op1(opnum, argbox1, descr=None).
    pub fn record_op1(
        &mut self,
        opcode: OpCode,
        arg0: Box,
        descr: Option<&majit_ir::DescrRef>,
    ) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, 1);
        let t0 = self._encode(arg0);
        self.append_int(t0);
        self._op_end_descr(opcode, descr, old_pos);
        pos
    }

    /// opencoder.py:685-691 record_op2(opnum, argbox1, argbox2, descr=None).
    pub fn record_op2(
        &mut self,
        opcode: OpCode,
        arg0: Box,
        arg1: Box,
        descr: Option<&majit_ir::DescrRef>,
    ) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, 2);
        let t0 = self._encode(arg0);
        self.append_int(t0);
        let t1 = self._encode(arg1);
        self.append_int(t1);
        self._op_end_descr(opcode, descr, old_pos);
        pos
    }

    /// opencoder.py:693-700 record_op3(opnum, argbox1, argbox2, argbox3, descr=None).
    pub fn record_op3(
        &mut self,
        opcode: OpCode,
        arg0: Box,
        arg1: Box,
        arg2: Box,
        descr: Option<&majit_ir::DescrRef>,
    ) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, 3);
        let t0 = self._encode(arg0);
        self.append_int(t0);
        let t1 = self._encode(arg1);
        self.append_int(t1);
        let t2 = self._encode(arg2);
        self.append_int(t2);
        self._op_end_descr(opcode, descr, old_pos);
        pos
    }

    /// M2 Step 2b: encode a pyre `OpRef` into the wire `i64` tag.
    ///
    /// Pyre OpRef carries a `CONST_BIT` that marks constant-namespace
    /// entries whose concrete values live in an external `ConstantPool`.
    /// RPython opencoder expects `Const{Int,Float,Ptr}` / `AbstractResOp`
    /// boxes inline, so this helper resolves the constant (via `pool`)
    /// and then routes through the existing `_encode(Box)` path so the
    /// resulting bytes match RPython byte-for-byte.
    ///
    /// Panics when a constant OpRef has no pool entry — that is a
    /// genuine invariant break (every `is_constant()` OpRef must have
    /// been minted via `ConstantPool::get_or_insert_typed` or friends).
    pub fn _encode_opref(&mut self, opref: OpRef, pool: &ConstantPool) -> i64 {
        let b = if opref.is_constant() {
            let value = pool
                .get_value(opref)
                .unwrap_or_else(|| panic!("_encode_opref: constant {:?} not in pool", opref));
            match value {
                Value::Int(v) => Box::ConstInt(v),
                Value::Float(f) => Box::ConstFloat(f.to_bits()),
                Value::Ref(r) => Box::ConstPtr(r.as_usize() as u64),
                Value::Void => {
                    panic!("_encode_opref: constant {:?} has Void type", opref)
                }
            }
        } else {
            Box::of_op(opref)
        };
        self._encode(b)
    }

    /// M2 Step 2b: OpRef-taking adapter over `record_op(&[Box], descr)`.
    ///
    /// Mirrors opencoder.py:664-670 `record_op(opnum, argboxes, descr=None)`
    /// but accepts pyre's constant-tagged OpRefs. Constants are resolved
    /// through `pool`; non-constant OpRefs route through the TAGBOX path
    /// via `Box::of_op`. Wire bytes are identical to the equivalent
    /// `record_op(&[Box])` call.
    pub fn record_op_oprefs(
        &mut self,
        opcode: OpCode,
        argrefs: &[OpRef],
        descr: Option<&majit_ir::DescrRef>,
        pool: &ConstantPool,
    ) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, argrefs.len());
        for &r in argrefs {
            let tagged = self._encode_opref(r, pool);
            self.append_int(tagged);
        }
        self._op_end_descr(opcode, descr, old_pos);
        pos
    }

    // ── M2 Step 2d · close_loop / finish helpers ──────────────────────
    //
    // Thin wrappers over `record_op_oprefs` that mirror the
    // `recorder::Trace` surface (`close_loop`, `close_loop_with_descr`,
    // `finish`).

    /// Close the loop: append a JUMP op with no descr.  Mirrors
    /// `recorder::Trace::close_loop`.
    pub fn close_loop_oprefs(&mut self, jump_args: &[OpRef], pool: &ConstantPool) -> u32 {
        self.record_op_oprefs(OpCode::Jump, jump_args, None, pool)
    }

    /// Close the loop with an explicit JUMP descriptor (tentative JUMP
    /// target token recorded before compile_trace, pyjitpl.py:3188).
    pub fn close_loop_oprefs_with_descr(
        &mut self,
        jump_args: &[OpRef],
        descr: Option<&majit_ir::DescrRef>,
        pool: &ConstantPool,
    ) -> u32 {
        self.record_op_oprefs(OpCode::Jump, jump_args, descr, pool)
    }

    /// Finish the trace: append a FINISH op with its FailDescr.
    /// pyjitpl.py:1637 `history.record1(rop.FINISH, ..., descr=token)`.
    pub fn finish_oprefs(
        &mut self,
        finish_args: &[OpRef],
        descr: &majit_ir::DescrRef,
        pool: &ConstantPool,
    ) -> u32 {
        self.record_op_oprefs(OpCode::Finish, finish_args, Some(descr), pool)
    }

    /// opencoder.py:738-742 append_snapshot_array_data_int(i).
    /// Writes a signed varint into `_snapshot_array_data`; values outside
    /// [MIN_VALUE, MAX_VALUE] trip `tag_overflow` (and encode 0 to keep
    /// the stream parseable).
    ///
    /// Phase B1 keeps this on the zigzag-based `encode_varint_signed`
    /// currently in this file so the constructor's reserve-index-0 call
    /// succeeds. Phase B2 replaces the encoder with RPython's 2/4-byte
    /// format, which will make this method fully parity-compliant.
    pub fn append_snapshot_array_data_int(&mut self, i: i64) {
        if !(MIN_VALUE..=MAX_VALUE).contains(&i) {
            self.tag_overflow = true;
            encode_varint_signed(&mut self._snapshot_array_data, 0);
        } else {
            encode_varint_signed(&mut self._snapshot_array_data, i);
        }
    }

    /// opencoder.py:744-748 append_snapshot_data_int(i).
    pub fn append_snapshot_data_int(&mut self, i: i64) {
        if !(MIN_VALUE..=MAX_VALUE).contains(&i) {
            self.tag_overflow = true;
            encode_varint_signed(&mut self._snapshot_data, 0);
        } else {
            encode_varint_signed(&mut self._snapshot_data, i);
        }
    }

    /// opencoder.py:852-858 Trace.get_live_ranges().
    ///
    /// Walks the encoded byte stream `_ops` + `_snapshot_data` +
    /// `_snapshot_array_data` and returns `liveranges[v] =
    /// last_use_index` for every raw trace position `v` (opencoder.py:
    /// 855 `index = t._count`; walk via `next_element_update_live_range`
    /// at opencoder.py:340-360). Guard fail-arg equivalents come from
    /// the attached snapshot chain through `update_liveranges`
    /// (opencoder.py:239-247).
    pub fn get_live_ranges(&self) -> Vec<usize> {
        let mut liveranges = vec![0usize; self._index as usize];
        // opencoder.py:855 `index = t._count` with `t._count = start` at
        // iterator construction time (opencoder.py:270). `start` is the
        // Trace's `_start` field (= `max_num_inputargs`).
        let mut index = self._start as usize;
        // opencoder.py:269 `self.pos = start` — byte cursor starts at
        // `_start`, the same value as the op index counter, because
        // the first `max_num_inputargs` bytes of `_ops` are reserved
        // as the inputarg placeholder region.
        let mut p: usize = self._start as usize;
        let end = self._pos;
        while p < end {
            let opnum = self._ops[p];
            p += 1;
            let opcode = u16_to_opcode(opnum as u16);
            // opencoder.py:644-649 — variadic ops store `num_argboxes`
            // inline as a signed varint; fixed-arity ops do not.
            let argnum = match opcode.arity() {
                Some(n) => n as usize,
                None => {
                    let (n, consumed) = decode_varint_signed(&self._ops[p..]);
                    p += consumed;
                    n as usize
                }
            };
            // opencoder.py:346-350 — TAGBOX args set liveranges[v] = index.
            for _ in 0..argnum {
                let (tagged, consumed) = decode_varint_signed(&self._ops[p..]);
                p += consumed;
                let (tag, v) = untag(tagged as u32);
                if tag == TAGBOX {
                    let idx = v as usize;
                    if idx < liveranges.len() {
                        liveranges[idx] = index;
                    }
                }
            }
            let result_type = opcode.result_type();
            // opencoder.py:351-352 — non-void self-defines its slot.
            if result_type != Type::Void && index < liveranges.len() {
                liveranges[index] = index;
            }
            // opencoder.py:353-357 — guards drive the snapshot walk.
            if opcode.has_descr() {
                let (descr_index, consumed) = decode_varint_signed(&self._ops[p..]);
                p += consumed;
                if opcode.is_guard() {
                    update_liveranges(
                        &self._snapshot_data,
                        &self._snapshot_array_data,
                        descr_index as usize,
                        index,
                        &mut liveranges,
                    );
                }
            }
            // opencoder.py:358-360 — `_index` advances only for non-void.
            if result_type != Type::Void {
                index += 1;
            }
        }
        liveranges
    }

    /// opencoder.py:860-884 Trace.get_dead_ranges().
    ///
    /// For each index `x`, `deadranges[x]` names a value that is
    /// *certainly* dead before step `x`. Collisions (multiple values
    /// dying at the same step) are resolved by linear probing forward
    /// through `deadranges` — the nested `insert(ranges, pos, v)`
    /// helper at opencoder.py:865-871.
    ///
    /// Result is memoized in `_deadranges = (self._count, deadranges)`
    /// and returned unchanged across calls until `_count` advances
    /// (opencoder.py:873-875, 883).
    pub fn get_dead_ranges(&mut self) -> Vec<usize> {
        // opencoder.py:873-875 cache hit path.
        if let Some((cached_count, cached)) = &self._deadranges {
            if *cached_count == self._count {
                return cached.clone();
            }
        }
        // opencoder.py:876 `liveranges = self.get_live_ranges()`.
        let liveranges = self.get_live_ranges();
        // opencoder.py:877-878 `deadranges = [0] * (self._index + 2)`.
        let mut deadranges = vec![0usize; (self._index as usize) + 2];
        debug_assert_eq!(deadranges.len(), liveranges.len() + 2);
        // opencoder.py:879-882 — skip inputargs (i < _start), insert
        // live end-points with linear probing.
        for i in (self._start as usize)..liveranges.len() {
            let elem = liveranges[i];
            if elem != 0 {
                // opencoder.py:865-871 nested insert(ranges, pos, v).
                let mut pos = elem + 1;
                while pos < deadranges.len() && deadranges[pos] != 0 {
                    pos += 1;
                    if pos == deadranges.len() {
                        break;
                    }
                }
                if pos < deadranges.len() {
                    deadranges[pos] = i;
                }
            }
        }
        // opencoder.py:883 — memoize.
        self._deadranges = Some((self._count, deadranges.clone()));
        deadranges
    }

    /// Rust adaptation: mirror pointer moves performed by the GC back
    /// into `_refs`. The shadow stack holds the authoritative post-move
    /// pointer for each entry pushed by `_encode_ptr`; copy those
    /// values into `_refs` so TAGCONSTPTR decodes land on the current
    /// address. Called at the same boundaries as
    /// `ConstantPool::refresh_from_gc`.
    pub fn refresh_from_gc(&mut self) {
        for &(ref_idx, ss_idx) in &self.rooted_ref_indices {
            self._refs[ref_idx] = majit_gc::shadow_stack::get(ss_idx).0 as u64;
        }
    }

    /// Pop the shadow stack back to the depth captured at construction
    /// and forget the rooted-index table. Idempotent.
    fn release_roots(&mut self) {
        if !self.rooted_ref_indices.is_empty() {
            majit_gc::shadow_stack::pop_to(self.shadow_stack_base);
            self.rooted_ref_indices.clear();
        }
    }
}

impl Drop for TraceRecordBuffer {
    fn drop(&mut self) {
        self.release_roots();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Test fixture equivalent of RPython's `metainterp_sd` fixture used
    /// by `test/test_opencoder.py`. Produces a fresh empty static data
    /// object so `Trace.__init__` (opencoder.py:471) has something to
    /// hold in `self.metainterp_sd`.
    fn empty_sd() -> Arc<crate::MetaInterpStaticData> {
        Arc::new(crate::MetaInterpStaticData::new())
    }

    #[test]
    fn test_signed_varint_roundtrip() {
        // opencoder.py enforces MIN_VALUE=-2^30, MAX_VALUE=2^30-1. Test
        // the 2-byte path (values in [-2^14, 2^14-1]) and the 4-byte path
        // (outside it). `i64::MAX` / `i64::MIN` would trip the
        // debug_assert — they are out of the encoder's documented range.
        let values = [
            0i64,
            1,
            -1,
            127,
            -128,
            1000,
            -1000,
            // 2-byte boundaries (stay in ±2^14 range)
            (1 << 14) - 1,
            -(1 << 14),
            // 4-byte range
            (1 << 14),
            -(1 << 14) - 1,
            (1 << 20),
            -(1 << 20),
            MIN_VALUE,
            MAX_VALUE,
        ];
        for &val in &values {
            let mut buf = Vec::new();
            encode_varint_signed(&mut buf, val);
            let (decoded, _consumed) = decode_varint_signed(&buf);
            assert_eq!(decoded, val, "signed varint roundtrip failed for {val}");
        }
    }

    /// Phase B2: encoder must match RPython's 2-byte / 4-byte layout
    /// bit-exactly for snapshot and op stream compatibility. Verify
    /// small values stay 2 bytes and big values expand to 4 bytes.
    #[test]
    fn test_signed_varint_wire_format_b2() {
        // 2-byte values: flag bit (bit 7 of byte 0) MUST be clear.
        for &val in &[0i64, 1, -1, (1 << 14) - 1, -(1 << 14)] {
            let mut buf = Vec::new();
            encode_varint_signed(&mut buf, val);
            assert_eq!(buf.len(), 2, "value {val} should be 2 bytes");
            assert_eq!(buf[0] & 0x80, 0, "flag bit set for 2-byte value {val}");
        }
        // 4-byte values: flag bit MUST be set.
        for &val in &[1i64 << 14, -(1i64 << 14) - 1, MAX_VALUE, MIN_VALUE] {
            let mut buf = Vec::new();
            encode_varint_signed(&mut buf, val);
            assert_eq!(buf.len(), 4, "value {val} should be 4 bytes");
            assert_eq!(buf[0] & 0x80, 0x80, "flag bit clear for 4-byte value {val}");
        }
    }

    #[test]
    fn test_tag_untag_roundtrip() {
        for kind in [TAGINT, TAGCONSTPTR, TAGCONSTOTHER, TAGBOX] {
            for value in [0u32, 1, 42, 255, 1000] {
                let tagged = tag(kind, value);
                let (k, v) = untag(tagged);
                assert_eq!(k, kind, "tag kind mismatch");
                assert_eq!(v, value, "tag value mismatch");
            }
        }
    }

    fn iarg(pos: u32) -> OpRef {
        OpRef::input_arg_int(pos)
    }

    fn farg(pos: u32) -> OpRef {
        OpRef::input_arg_float(pos)
    }

    fn rarg(pos: u32) -> OpRef {
        OpRef::input_arg_ref(pos)
    }

    fn iop(pos: u32) -> OpRef {
        OpRef::int_op(pos)
    }

    fn fop(pos: u32) -> OpRef {
        OpRef::float_op(pos)
    }

    fn rop(pos: u32) -> OpRef {
        OpRef::ref_op(pos)
    }

    fn vop(pos: u32) -> OpRef {
        OpRef::void_op(pos)
    }

    /// Helper: build an Op with a specific trace position and the same
    /// typed result class RPython's `opclasses[opnum]` would instantiate.
    fn op_at(pos: u32, opcode: majit_ir::OpCode, args: &[OpRef]) -> majit_ir::Op {
        let mut op = majit_ir::Op::new(opcode, args);
        op.pos = OpRef::op_typed(pos, opcode.result_type());
        op
    }

    #[test]
    fn test_trace_iterator_next_basic() {
        // opencoder.py:362-406 TraceIterator.next() parity:
        //
        // Phase-1 iteration over a 2-inputarg trace producing two IntAdd
        // results. Inputargs occupy positions [0, 2); the first non-void
        // result is allocated as BoxInt at position 2 (= start_fresh = num_inputargs).
        let ops = vec![
            op_at(2, majit_ir::OpCode::IntAdd, &[iarg(0), iarg(1)]),
            op_at(3, majit_ir::OpCode::IntAdd, &[iop(2), iarg(0)]),
            majit_ir::Op::new(majit_ir::OpCode::Finish, &[iop(3)]),
        ];
        let inputarg_types = vec![majit_ir::Type::Int; 2];

        // Phase 1 / legacy layout: start_fresh = 0 → inputargs allocated
        // as InputArgInt(0..2), op results follow as BoxInt(2..).
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, &inputarg_types, 0, None);
        assert_eq!(iter.inputargs, vec![iarg(0), iarg(1)]);
        assert_eq!(iter._cache[0], Some(iarg(0)));
        assert_eq!(iter._cache[1], Some(iarg(1)));

        // First IntAdd: result at fresh BoxInt(2), args translated via cache.
        let r1 = iter.next().unwrap();
        assert_eq!(r1.pos, iop(2));
        assert_eq!(r1.args[0], iarg(0));
        assert_eq!(r1.args[1], iarg(1));

        // Second IntAdd: result at fresh BoxInt(3); first arg references the
        // previous result (raw pos 2 → cached BoxInt(2)).
        let r2 = iter.next().unwrap();
        assert_eq!(r2.pos, iop(3));
        assert_eq!(r2.args[0], iop(2));
        assert_eq!(r2.args[1], iarg(0));

        // Finish is void: pos unchanged, arg cached → BoxInt(3).
        let finish = iter.next().unwrap();
        assert_eq!(finish.args[0], iop(3));
        assert!(iter.done());
    }

    #[test]
    fn test_trace_iterator_phase1_phase2_disjoint() {
        // opencoder.py:362-406 TraceIterator.next() identity parity:
        //
        // RPython distinguishes Phase 1 boxes from Phase 2 boxes by Python
        // `is`-identity even when the two iterators visit the same source
        // trace position. majit lacks separate identity from position, so
        // each phase passes a different `start_fresh` and the iterator
        // produces disjoint OpRefs by construction.
        let ops = vec![
            op_at(2, majit_ir::OpCode::IntAdd, &[iarg(0), iarg(1)]),
            op_at(3, majit_ir::OpCode::IntAdd, &[iop(2), iarg(0)]),
            majit_ir::Op::new(majit_ir::OpCode::Finish, &[iop(3)]),
        ];
        let inputarg_types = vec![majit_ir::Type::Int; 2];

        // Phase 1: start_fresh = 0 reproduces the legacy positional layout
        // (inputargs at [0, num_inputargs), op results at [num_inputargs, …)).
        let mut p1 = TraceIterator::new(&ops, 0, ops.len(), None, &inputarg_types, 0, None);
        let mut p1_ops = Vec::new();
        while let Some(op) = p1.next() {
            p1_ops.push(op);
        }
        let p1_high_water = p1._fresh;
        assert_eq!(p1_high_water, 4);
        // RPython `_index` walks the `_cache` slots monotonically; after
        // the 2 non-void ops it should point just past the last written
        // raw trace position (3 + 1 = 4).
        assert_eq!(p1._index, 4);

        // Phase 2: start_fresh = phase1 high water → disjoint namespace.
        let mut p2 = TraceIterator::new(
            &ops,
            0,
            ops.len(),
            None,
            &inputarg_types,
            p1_high_water,
            None,
        );
        let mut p2_ops = Vec::new();
        while let Some(op) = p2.next() {
            p2_ops.push(op);
        }
        assert_eq!(p2.inputargs, vec![iarg(4), iarg(5)]);
        assert_eq!(p2_ops[0].pos, iop(6));
        assert_eq!(p2_ops[0].args[0], iarg(4));
        assert_eq!(p2_ops[0].args[1], iarg(5));
        assert_eq!(p2_ops[1].pos, iop(7));
        assert_eq!(p2_ops[1].args[0], iop(6));
        assert_eq!(p2_ops[1].args[1], iarg(4));
        assert_eq!(p2_ops[2].args[0], iop(7));

        // No Phase 1 OpRef equals any Phase 2 OpRef.
        let p1_set: std::collections::HashSet<u32> = p1_ops
            .iter()
            .map(|op| op.pos.raw())
            .filter(|&p| p != u32::MAX)
            .collect();
        let p2_set: std::collections::HashSet<u32> = p2_ops
            .iter()
            .map(|op| op.pos.raw())
            .filter(|&p| p != u32::MAX)
            .collect();
        assert!(p1_set.is_disjoint(&p2_set));
    }

    #[test]
    fn test_trace_iterator_constants_passthrough() {
        // opencoder.py:321-335 _untag(): TAGINT/TAGCONSTPTR/TAGCONSTOTHER
        // pass through unchanged. In majit, constant OpRefs (is_constant
        // high-bit marker) must NOT be remapped through `_cache`.
        let const_ref = OpRef::const_int(5);
        let ops = vec![op_at(1, majit_ir::OpCode::IntAdd, &[iarg(0), const_ref])];
        let inputarg_types = vec![majit_ir::Type::Int; 1];
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, &inputarg_types, 0, None);
        let r = iter.next().unwrap();
        assert_eq!(r.pos, iop(1));
        assert_eq!(r.args[0], iarg(0));
        assert_eq!(r.args[1], const_ref);
    }

    #[test]
    fn test_trace_iterator_replace_last_cached_shifted_phase() {
        // opencoder.py:282-284 replace_last_cached parity. RPython walks
        // `_cache` densely by ops-count (`_index`), so `_cache[_index - 1]`
        // always targets the slot just written by `next()`. majit's
        // `_index` is separated from the fresh OpRef counter — in a
        // shifted phase (Phase 2 with `start_fresh` >> 0) the cache key
        // is still the raw trace position, so `_index` must stay in raw
        // trace position coordinates.
        let ops = vec![
            op_at(1, majit_ir::OpCode::IntAdd, &[iarg(0), iarg(0)]),
            op_at(2, majit_ir::OpCode::IntAdd, &[iop(1), iarg(0)]),
            majit_ir::Op::new(majit_ir::OpCode::Finish, &[iop(2)]),
        ];
        // Shifted phase: start_fresh = 100 so any confusion between
        // _index (raw trace position) and _fresh (fresh OpRef counter)
        // surfaces as an out-of-bounds panic or stale cache slot.
        let inputarg_types = vec![majit_ir::Type::Int; 1];
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, &inputarg_types, 100, None);
        let r1 = iter.next().unwrap();
        // After writing raw pos 1, _index should be 2 (next slot).
        assert_eq!(iter._index, 2);
        assert_eq!(r1.pos, iop(101));
        // replace_last_cached(oldbox=BoxInt(101), new_box=BoxInt(999))
        // must target _cache[_index - 1] = _cache[1], where the last
        // write placed BoxInt(101).
        iter.replace_last_cached(iop(101), iop(999));
        assert_eq!(iter._cache[1], Some(iop(999)));

        // The next op references raw pos 1 as its first arg, so the
        // replaced value should flow through _untag → _get.
        let r2 = iter.next().unwrap();
        assert_eq!(r2.args[0], iop(999));
        assert_eq!(r2.args[1], iarg(100));
        // After writing raw pos 2, _index advances to 3.
        assert_eq!(iter._index, 3);
    }

    #[test]
    fn test_trace_iterator_fail_args_remapped() {
        // opencoder.py:362-406 next() routes guard fail_args through the
        // same _untag/_get path as regular args.
        let mut guard = op_at(2, majit_ir::OpCode::GuardTrue, &[iop(1)]);
        guard.fail_args = Some(vec![iarg(0), iop(1)].into());
        let ops = vec![
            op_at(1, majit_ir::OpCode::IntEq, &[iarg(0), iarg(0)]),
            guard,
        ];
        let inputarg_types = vec![majit_ir::Type::Int; 1];
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, &inputarg_types, 10, None);
        let r1 = iter.next().unwrap();
        assert_eq!(r1.pos, iop(11));
        assert_eq!(r1.args[0], iarg(10));
        assert_eq!(r1.args[1], iarg(10));
        let r2 = iter.next().unwrap();
        assert_eq!(r2.args[0], iop(11));
        let fa = r2.fail_args.as_ref().unwrap();
        assert_eq!(fa[0], iarg(10));
        assert_eq!(fa[1], iop(11));
    }

    #[test]
    fn test_trace_iterator_overwrites_inputarg_cache_slot_with_fresh_result() {
        // RPython next() allocates a fresh result box even when the raw trace
        // position collides with an already-seeded inputarg cache slot.
        let ops = vec![
            op_at(3, majit_ir::OpCode::GetfieldRawI, &[rarg(0)]),
            op_at(1, majit_ir::OpCode::GetarrayitemGcR, &[iop(3), rarg(1)]),
            majit_ir::Op::new(majit_ir::OpCode::Finish, &[rop(1)]),
        ];
        let inputarg_types = vec![
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
            majit_ir::Type::Int,
            majit_ir::Type::Int,
        ];
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, &inputarg_types, 100, None);

        assert_eq!(
            iter.inputargs,
            vec![rarg(100), rarg(101), iarg(102), iarg(103)]
        );
        assert_eq!(iter._cache[1], Some(rarg(101)));

        let op0 = iter.next().unwrap();
        assert_eq!(op0.pos, iop(104));
        assert_eq!(op0.args.as_slice(), &[rarg(100)]);

        let op1 = iter.next().unwrap();
        assert_eq!(op1.args.as_slice(), &[iop(104), rarg(101)]);
        assert_eq!(op1.pos, rop(105));
        assert_eq!(iter._cache[1], Some(rop(105)));

        let finish = iter.next().unwrap();
        assert_eq!(finish.args.as_slice(), &[rop(105)]);
    }

    // Phase B1 intentionally drops `test_trace_record_buffer` and
    // `test_trace_record_snapshot`: they exercised the pre-Phase-B
    // legacy record_op / capture_resumedata / snapshots API that
    // diverged from rpython/jit/metainterp/opencoder.py and has been
    // removed. Phase B3 (wire format), B5 (guard 0-placeholder), B7
    // (capture_resumedata + create_top_snapshot), and Phase C
    // (RPython test_opencoder.py ports) replace them with parity
    // tests that exercise the RPython-shaped API.

    #[test]
    fn test_trace_record_buffer_initial_state_b1() {
        // Phase B1 smoke test: `Trace::new(max_num_inputargs)` seeds the
        // counters per opencoder.py:497-500 (all four — including
        // `_pos` — equal `max_num_inputargs`) and reserves
        // `snapshot_array_data` index 0 for the empty-length array
        // (opencoder.py:489 + :728-733).
        let buf = TraceRecordBuffer::new(3, empty_sd());
        assert_eq!(buf.max_num_inputargs, 3);
        assert_eq!(buf._count, 3);
        assert_eq!(buf._index, 3);
        assert_eq!(buf._start, 3);
        assert_eq!(buf._pos, 3);
        // opencoder.py:564 `length() = self._pos` — byte length of the
        // encoded stream INCLUDES the reserved inputarg placeholder
        // prefix.
        assert_eq!(buf.length(), 3);
        assert!(buf.inputargs.is_empty());
        assert!(!buf.tag_overflow);
        // _descrs starts with one None entry so index 0 means "no descr".
        assert_eq!(buf._descrs.len(), 1);
        assert!(buf._descrs[0].is_none());
        // _refs starts with nullptr so index 0 means "null".
        assert_eq!(buf._refs, vec![0u64]);
        // _snapshot_array_data has one signed-varint-encoded 0 recorded.
        assert!(!buf._snapshot_array_data.is_empty());
    }

    /// Phase B8: `get_live_ranges` walks `_ops` + `_snapshot_data`
    /// + `_snapshot_array_data` and produces RPython-parity
    /// `liveranges[v] = last_use_index` output, with guards driving
    /// the snapshot-chain walk via `update_liveranges`.
    #[test]
    fn test_get_live_ranges_linear_b8() {
        // Build: inputarg i0 at position 0. Op1 = IntAdd(i0, i0) at
        // position 1 (result index 1). Op2 = IntSub(Op1.result, i0)
        // at position 2. Linear chain — no guards, no snapshots.
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let p1 = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(0), None);
        let p2 = buf.record_op2(OpCode::IntSub, Box::ResOp(p1), Box::ResOp(0), None);
        assert_eq!(p1, 1);
        assert_eq!(p2, 2);

        let lr = buf.get_live_ranges();
        // lr has length _index == 3 (0, 1, 2).
        assert_eq!(lr.len(), 3);
        // i0 last used at index 2 (IntSub). Op1.result last used at
        // index 2 (IntSub arg). Op2 self-defines at index 2.
        assert_eq!(lr[0], 2, "inputarg last-used at the IntSub step");
        assert_eq!(lr[1], 2, "IntAdd result last-used by IntSub");
        assert_eq!(lr[2], 2, "IntSub self-defines its own slot");
    }

    /// Phase B8: when a guard fires, `update_liveranges` walks the
    /// snapshot chain and marks the snapshot's tagged TAGBOX items
    /// at the guard's step index.
    #[test]
    fn test_get_live_ranges_guard_snapshot_b8() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        // Op0 = GuardTrue(p0) — records a 0-placeholder that the
        // snapshot attach patches.
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        // Build a small active-box array containing TAGBOX(1)
        // (inputarg 1). create_top_snapshot then patches the guard's
        // descr slot to the snapshot byte offset.
        let array = buf.new_array(1);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_box_position(1));
        let _snap = buf.create_top_snapshot(0, 0, array, &[], &[], true);

        let lr = buf.get_live_ranges();
        // _index hasn't advanced (GuardTrue is void), so the liveranges
        // array still has length == _start (= 2 inputargs).
        assert_eq!(lr.len(), 2);
        // The guard's step index is _start (2). p0 is referenced as the
        // guard's direct arg — last-used at step 2. p1 is referenced via
        // the snapshot's active-box array — also last-used at step 2.
        assert_eq!(lr[0], 2, "GuardTrue arg → liveranges[0]");
        assert_eq!(lr[1], 2, "snapshot chain → liveranges[1]");
    }

    /// Phase B8 + RPython opencoder.py:860-884 `get_dead_ranges()` —
    /// cache-hit path: first call populates `_deadranges = (_count,
    /// vec)`, second call with the same `_count` must return the
    /// cached vec unchanged.
    #[test]
    fn test_get_dead_ranges_memoization_b8() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        // A couple of non-void ops to populate liveranges.
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(0), None);
        let _ = buf.record_op2(OpCode::IntSub, Box::ResOp(1), Box::ResOp(0), None);
        let first = buf.get_dead_ranges();
        let cached_count_after_first = buf._deadranges.as_ref().map(|(c, _)| *c);
        assert_eq!(cached_count_after_first, Some(buf._count));

        // opencoder.py:874 — `self._deadranges[0] == self._count` returns
        // the cached list without rewalking the trace.
        let second = buf.get_dead_ranges();
        assert_eq!(first, second);

        // Recording one more op advances `_count`, invalidating the
        // cache (opencoder.py:873-875 gate).
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(2), Box::ResOp(0), None);
        let third = buf.get_dead_ranges();
        // The third call sees a new `_count` and recomputes — the
        // resulting vec is one slot longer (`_index + 2` grew by one).
        assert_eq!(third.len(), second.len() + 1);
    }

    /// Step 2a: `record_input_arg` returns a positional OpRef and appends
    /// an `InputArg` whose `tp` field matches the passed type. `num_ops`
    /// stays at 0 because recording inputargs advances `_count` purely
    /// through `_start = max_num_inputargs`, not through ops.
    #[test]
    fn test_record_input_arg_2a() {
        let mut buf = TraceRecordBuffer::new(3, empty_sd());
        let a = buf.record_input_arg(Type::Int);
        let b = buf.record_input_arg(Type::Float);
        let c = buf.record_input_arg(Type::Ref);
        assert_eq!(a, iarg(0));
        assert_eq!(b, farg(1));
        assert_eq!(c, rarg(2));
        assert_eq!(buf.num_inputargs(), 3);
        assert_eq!(
            buf.inputarg_types(),
            vec![Type::Int, Type::Float, Type::Ref]
        );
        assert_eq!(buf.inputargs().len(), 3);
        // `_count` was seeded to max_num_inputargs; record_input_arg must
        // NOT advance it (see history.py:725 length parity — the formula
        // stays `_count - max_num_inputargs`).
        assert_eq!(buf.num_ops(), 0);
    }

    /// Step 2a: `record_input_arg` panics when asked to exceed the
    /// `max_num_inputargs` cap supplied to `Trace::new`.
    #[test]
    #[should_panic(expected = "exceeds max_num_inputargs")]
    fn test_record_input_arg_overflow_panics_2a() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        buf.record_input_arg(Type::Int);
        // Second call must panic — cap is 1.
        buf.record_input_arg(Type::Int);
    }

    // ── M4 · ByteTraceIter round-trip tests ─────────────────────────

    /// M4 step 1: empty trace iterates to nothing, inputargs get fresh
    /// OpRefs seeded from `max_num_inputargs`.
    #[test]
    fn test_byte_trace_iter_empty_m4() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        buf.record_input_arg(Type::Int);
        buf.record_input_arg(Type::Int);
        let mut it = buf.get_byte_iter();
        assert!(it.done());
        assert!(it.next().is_none());
        assert_eq!(it.inputargs.len(), 2);
        // start_fresh = max_num_inputargs = 2; inputargs get InputArgInt(2), InputArgInt(3).
        assert_eq!(it.inputargs[0], iarg(2));
        assert_eq!(it.inputargs[1], iarg(3));
    }

    /// M4 step 1: TAGBOX-only round-trip for 2-arg fixed-arity op.
    /// After `record_op2(IntAdd, box0, box1)` the byte stream must
    /// decode back to an Op whose args are the fresh-OpRef form of the
    /// two inputargs.
    #[test]
    fn test_byte_trace_iter_tagbox_round_trip_m4() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        let _ = buf.record_input_arg(Type::Int);
        // `record_op2` encodes `Box::ResOp(0)` / `Box::ResOp(1)` as
        // TAGBOX tagged values.  The fresh iterator cache maps raw
        // position 0 / 1 to the iterator-local OpRefs `inputargs[0..=1]`.
        let pos = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(1), None);
        assert_eq!(pos, 2); // pre-bump `_index`, seeded to max_num_inputargs

        let mut it = buf.get_byte_iter();
        let fresh_i0 = it.inputargs[0];
        let fresh_i1 = it.inputargs[1];
        let op = it.next().expect("one op");
        assert_eq!(op.opcode, OpCode::IntAdd);
        assert_eq!(op.args.len(), 2);
        assert_eq!(op.args[0], fresh_i0);
        assert_eq!(op.args[1], fresh_i1);
        assert!(it.done());
        assert!(it.next().is_none());
    }

    /// M4 step 1: chained ops — the second op references the first
    /// op's result.  `_cache` / `_index` bookkeeping must keep the
    /// per-iteration OpRefs coherent across multiple `next()` calls.
    #[test]
    fn test_byte_trace_iter_chained_m4() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        buf.record_input_arg(Type::Int);
        buf.record_input_arg(Type::Int);
        // r2 = Int(0) + Int(1);  r3 = r2 * Int(0)
        let r2 = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(1), None);
        let _r3 = buf.record_op2(OpCode::IntMul, Box::ResOp(r2), Box::ResOp(0), None);

        let mut it = buf.get_byte_iter();
        let fresh_i0 = it.inputargs[0];
        let fresh_i1 = it.inputargs[1];
        let add = it.next().expect("IntAdd");
        assert_eq!(add.opcode, OpCode::IntAdd);
        assert_eq!(add.args[0], fresh_i0);
        assert_eq!(add.args[1], fresh_i1);
        let mul = it.next().expect("IntMul");
        assert_eq!(mul.opcode, OpCode::IntMul);
        // The first arg of IntMul referenced the first op's result via
        // TAGBOX.  The iterator's `_cache` must map raw position 2 →
        // `add.pos` (the fresh OpRef emitted one `next()` ago).
        assert_eq!(mul.args[0], add.pos);
        assert_eq!(mul.args[1], fresh_i0);
        assert!(it.done());
    }

    /// M4 step 1: constant args panic when the iterator was built
    /// without a ConstantPool (the default `get_byte_iter()` /
    /// `new()` path).  M4 step 2 introduced `new_with_pool` for the
    /// with-pool path; this test keeps the back-compat panic surface
    /// honest.
    #[test]
    #[should_panic(expected = "no ConstantPool attached")]
    fn test_byte_trace_iter_const_arg_panics_without_pool_m4() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        buf.record_input_arg(Type::Int);
        // `Box::ConstInt(0)` routes through `_encode_smallint` → TAGINT.
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ConstInt(0), None);
        let mut it = buf.get_byte_iter();
        let _ = it.next(); // should panic inside `_untag` on the TAGINT arg
    }

    /// M4 step 2: TAGINT arg with a ConstantPool attached resolves into
    /// a pool-allocated OpRef whose value matches the recorded
    /// `Box::ConstInt(v)`.
    #[test]
    fn test_byte_trace_iter_tagint_with_pool_m4() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ConstInt(42), None);

        let mut pool = crate::constant_pool::ConstantPool::new();
        let mut it = ByteTraceIter::new_with_pool(
            &buf,
            buf._start as usize,
            buf._pos,
            buf.max_num_inputargs,
            Some(&mut pool),
        );
        let op = it.next().expect("one op");
        assert_eq!(op.opcode, OpCode::IntAdd);
        let const_arg = op.args[1];
        assert!(const_arg.is_constant());
        // The pool entry for the TAGINT arg must round-trip to 42 (Int).
        drop(it);
        assert_eq!(pool.get_value(const_arg), Some(majit_ir::Value::Int(42)));
    }

    /// M4 step 2: same-value constants dedupe through the pool — two
    /// TAGINT args with value 7 must resolve to the same OpRef.
    #[test]
    fn test_byte_trace_iter_tagint_dedup_m4() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ConstInt(7), None);
        // Second op's first arg is also `ConstInt(7)` — should dedupe.
        let _ = buf.record_op2(OpCode::IntAdd, Box::ConstInt(7), Box::ResOp(0), None);

        let mut pool = crate::constant_pool::ConstantPool::new();
        let mut it = ByteTraceIter::new_with_pool(
            &buf,
            buf._start as usize,
            buf._pos,
            buf.max_num_inputargs,
            Some(&mut pool),
        );
        let first = it.next().unwrap();
        let second = it.next().unwrap();
        assert_eq!(first.args[1], second.args[0]);
    }

    /// M4 step 3: guard opcode decode — the decoded Op's `descr` is
    /// `None` (guards resolve snapshot placeholders separately) but its
    /// `rd_resume_position` carries the encoded descr index.
    #[test]
    fn test_byte_trace_iter_guard_rd_resume_position_m4() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        // `record_op1` encodes a guard with descr_index=0 placeholder.
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        // Patch the placeholder to a concrete snapshot index.
        buf.patch_last_guard_descr_slot(7);

        let mut it = buf.get_byte_iter();
        let op = it.next().expect("one op");
        assert_eq!(op.opcode, OpCode::GuardTrue);
        assert!(op.descr.is_none()); // guards do NOT carry a resolved descr
        assert_eq!(op.rd_resume_position, 7); // opencoder.py:423 parity
    }

    /// M4 step 3: non-guard descr-bearing opcode routes through the
    /// local `_descrs` pool (for descrs without a global
    /// `get_descr_index`).  Decoded `descr` must resolve back to the
    /// same DescrRef that `record_op_with_descr` interned.
    #[test]
    fn test_byte_trace_iter_local_descr_m4() {
        use std::sync::Arc;

        #[derive(Debug)]
        struct LocalDescr {
            tag: u32,
        }
        impl majit_ir::Descr for LocalDescr {
            fn index(&self) -> u32 {
                self.tag
            }
        }

        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        let descr: majit_ir::DescrRef = Arc::new(LocalDescr { tag: 99 });
        // `record_op1` with a `Some(&descr)` exercises the
        // `_encode_descr` → local `_descrs` append path.
        let _ = buf.record_op1(OpCode::GetfieldGcI, Box::ResOp(0), Some(&descr));

        let mut it = buf.get_byte_iter();
        let op = it.next().expect("one op");
        assert_eq!(op.opcode, OpCode::GetfieldGcI);
        let resolved = op.descr.expect("descr must resolve");
        assert_eq!(resolved.index(), 99);
        assert_eq!(op.rd_resume_position, -1); // non-guard sentinel
    }

    /// M4 step 3: non-guard descr-bearing opcode with `descr_index == 0`
    /// (no-descr placeholder) decodes back to `descr: None` without
    /// panicking in the local pool lookup.  Uses `record_op1(_, _, None)`
    /// on a has_descr opcode.
    #[test]
    fn test_byte_trace_iter_descr_placeholder_zero_m4() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        let _ = buf.record_op1(OpCode::GetfieldGcI, Box::ResOp(0), None);

        let mut it = buf.get_byte_iter();
        let op = it.next().expect("one op");
        assert_eq!(op.opcode, OpCode::GetfieldGcI);
        assert!(op.descr.is_none());
        assert_eq!(op.rd_resume_position, -1);
    }

    // ── SnapshotIterator::get / unpack_array parity tests ──

    /// opencoder.py:222-223 `SnapshotIterator.get(index)` — resolve
    /// a tagged value via `main_iter._untag`.  Build a real empty
    /// top-snapshot so `SnapshotIterator::new` has a valid record
    /// header to parse, then verify TAGBOX / TAGINT resolution.
    #[test]
    fn test_snapshot_iterator_get_via_main_iter() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        // `create_empty_top_snapshot` invokes `patch_last_guard_descr_slot`,
        // which requires `_pos >= 2`.  Record a guard op first so the
        // byte stream has a patch target.
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        // Write an empty top snapshot so snapshot_data has the valid
        // header SnapshotIterator::new expects.
        let snap_idx = buf.create_empty_top_snapshot(&[], &[]);

        let mut pool = crate::constant_pool::ConstantPool::new();
        let mut main_iter = ByteTraceIter::new_with_pool(
            &buf,
            buf._start as usize,
            buf._pos,
            buf.max_num_inputargs,
            Some(&mut pool),
        );

        let snap_it = SnapshotIterator::new(
            &buf._snapshot_data,
            &buf._snapshot_array_data,
            snap_idx as usize,
        );
        // TAGBOX(0) → inputarg[0]'s fresh OpRef (pre-seeded in
        // `_cache` at ByteTraceIter::new_with_pool).
        let tagbox = TraceRecordBuffer::_encode_box_position(0);
        let resolved_box = snap_it.get(tagbox, &mut main_iter);
        let fresh_i0 = main_iter.inputargs[0];
        assert_eq!(resolved_box, fresh_i0);
        // TAGINT(42) → pool-allocated constant OpRef.
        let tagint = TraceRecordBuffer::_encode_smallint(42);
        let resolved_int = snap_it.get(tagint, &mut main_iter);
        assert!(resolved_int.is_constant());
    }

    /// opencoder.py:228-231 `unpack_array(arr)` — `[self.get(i) for i
    /// in arr]`.  Build a 2-element tagged box array, iterate it
    /// through `unpack_array`, and verify each tagged slot resolves
    /// to the expected fresh-iterator OpRef.
    #[test]
    fn test_snapshot_iterator_unpack_array() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        let _ = buf.record_input_arg(Type::Int);

        // Write a 2-element tagged box array into `_snapshot_array_data`.
        let array_idx = buf.new_array(2);
        buf._add_box_to_storage_box(Box::ResOp(0));
        buf._add_box_to_storage_box(Box::ResOp(1));

        // `create_empty_top_snapshot` patches the preceding guard; a
        // guard op needs to precede the snapshot write.
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        // Write an empty top snapshot so SnapshotIterator::new has a
        // valid record header (parses vable/vref indices from it).
        let snap_idx = buf.create_empty_top_snapshot(&[], &[]);

        let mut pool = crate::constant_pool::ConstantPool::new();
        let mut main_iter = ByteTraceIter::new_with_pool(
            &buf,
            buf._start as usize,
            buf._pos,
            buf.max_num_inputargs,
            Some(&mut pool),
        );
        let expected_0 = main_iter.inputargs[0];
        let expected_1 = main_iter.inputargs[1];

        let snap_it = SnapshotIterator::new(
            &buf._snapshot_data,
            &buf._snapshot_array_data,
            snap_idx as usize,
        );
        let arr = BoxArrayIter::new(&buf._snapshot_array_data, array_idx);
        let unpacked = snap_it.unpack_array(arr, &mut main_iter);
        assert_eq!(unpacked, vec![expected_0, expected_1]);
    }

    /// `_encode_smallint` must preserve the sign bit so TAGINT decode
    /// through `ByteTraceIter::_untag` round-trips negative values like
    /// Python's arithmetic shift in opencoder.py.
    #[test]
    fn test_encode_smallint_preserves_negative_sign() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_input_arg(Type::Int);
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ConstInt(-7), None);

        let mut pool = crate::constant_pool::ConstantPool::new();
        let mut it = ByteTraceIter::new_with_pool(
            &buf,
            buf._start as usize,
            buf._pos,
            buf.max_num_inputargs,
            Some(&mut pool),
        );
        let op = it.next().expect("one op");
        let const_arg = op.args[1];
        drop(it);
        assert_eq!(pool.get_value(const_arg), Some(majit_ir::Value::Int(-7)));
    }

    /// Phase B11: `tracing_done()` returns `Err(AbortReason::TooLong)`
    /// iff `tag_overflow` was tripped while recording, and clears the
    /// dedup dictionaries on success (opencoder.py:546-562).
    #[test]
    fn test_tracing_done_clears_dicts_b11() {
        let mut buf = TraceRecordBuffer::new(0, empty_sd());
        // Intern a couple of constants so the dedup dicts are non-empty.
        let _ = buf._encode_bigint(1 << 40);
        let _ = buf._encode_ptr(0xdead_beef);
        assert!(!buf._bigints_dict.is_empty());
        assert!(!buf._refs_dict.is_empty());
        // Preserve the interned values — they back the TraceIterator.
        let bigints_len = buf._bigints.len();
        let refs_len = buf._refs.len();

        assert_eq!(buf.tracing_done(), Ok(()));
        assert!(buf._bigints_dict.is_empty());
        assert!(buf._refs_dict.is_empty());
        assert_eq!(buf._bigints.len(), bigints_len);
        assert_eq!(buf._refs.len(), refs_len);
    }

    /// Phase B11: an `append_int` outside [MIN_VALUE, MAX_VALUE] trips
    /// `tag_overflow`, which `tracing_done()` reports as
    /// `AbortReason::TooLong` — mirroring
    /// `raise SwitchToBlackhole(Counters.ABORT_TOO_LONG)`.
    #[test]
    fn test_tracing_done_too_long_b11() {
        let mut buf = TraceRecordBuffer::new(0, empty_sd());
        buf.append_int(MAX_VALUE + 1);
        assert!(
            buf.tag_overflow,
            "append_int out of range must trip tag_overflow"
        );
        assert_eq!(
            buf.tracing_done(),
            Err(crate::pyjitpl::AbortReason::TooLong)
        );
    }

    /// Phase B11: `tag_overflow_imminent` flips once `_pos > 0.8 *
    /// MAX_VALUE`, letting the tracer force-close the trace before
    /// the hard overflow is recorded.
    #[test]
    fn test_tag_overflow_imminent_threshold_b11() {
        let mut buf = TraceRecordBuffer::new(0, empty_sd());
        assert!(!buf.tag_overflow_imminent());
        // Fake the cursor forward past 80% of MAX_VALUE.
        buf._pos = (MAX_VALUE as f64 * 0.8) as usize + 1;
        assert!(buf.tag_overflow_imminent());
    }

    /// Phase B9: cut_point/cut_at shape matches opencoder.py:567-575.
    /// `cut_point` returns the 5-tuple (_pos, _count, _index,
    /// len(_snapshot_data), len(_snapshot_array_data)). `cut_at`
    /// restores _pos, _count, _index; the snapshot-chain lengths in
    /// slots 3/4 are observed but NOT rewound (they grow monotonically
    /// in RPython; bridge compilation reuses earlier snapshots).
    #[test]
    fn test_cut_point_five_tuple_b9() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let before = buf.cut_point();
        assert_eq!(before._pos, buf._pos);
        assert_eq!(before._count, buf._count);
        assert_eq!(before._index, buf._index);
        assert_eq!(before.snapshot_data_len, buf._snapshot_data.len());
        assert_eq!(
            before.snapshot_array_data_len,
            buf._snapshot_array_data.len()
        );

        // Advance all five independently.
        buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        let array = buf.new_array(1);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_smallint(7));
        let _ = buf.create_top_snapshot(1, 10, array, &[], &[], true);
        let after = buf.cut_point();
        assert_ne!(after._pos, before._pos);
        assert_ne!(after._count, before._count); // _count bumped by guard.
        assert_eq!(after._index, before._index); // _index did NOT bump (guard is void).
        assert_ne!(after.snapshot_data_len, before.snapshot_data_len);
        assert_ne!(
            after.snapshot_array_data_len,
            before.snapshot_array_data_len
        );

        // cut_at restores only the first three fields.
        buf.cut_at(before);
        assert_eq!(buf._pos, before._pos);
        assert_eq!(buf._count, before._count);
        assert_eq!(buf._index, before._index);
    }

    /// Phase B7 smoke test: record a guard, then call
    /// `create_top_snapshot` with small box arrays. Verify:
    /// - The guard's trailing 0-placeholder is patched to the snapshot
    ///   index returned by `create_top_snapshot`.
    /// - `_snapshot_data` grew, `_snapshot_array_data` grew for
    ///   non-empty arrays, `_total_snapshots` bumped.
    /// - Length-0 arrays collapse to index 0 (the pre-seeded slot).
    #[test]
    fn test_create_top_snapshot_b7() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        // Record a guard so we have a 0-placeholder to patch.
        buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        let pos_after_guard = buf._pos;
        let total_before = buf._total_snapshots;
        let sd_len_before = buf._snapshot_data.len();

        // Build the active-boxes array via new_array + _add_box_to_storage,
        // mimicking how the frame's get_list_of_active_boxes drives it.
        let array = buf.new_array(2);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_smallint(1));
        buf._add_box_to_storage(TraceRecordBuffer::_encode_smallint(2));

        // No vable / vref boxes.
        let s = buf.create_top_snapshot(0, 0, array, &[], &[], true);
        assert_eq!(s as usize, sd_len_before);
        assert_eq!(buf._total_snapshots, total_before + 1);
        assert!(buf._snapshot_data.len() > sd_len_before);

        // The guard's trailing 2 zero bytes should have been replaced
        // with `s` (which fits in 2 bytes since it's 0 here).
        assert_eq!(buf._pos, pos_after_guard);
        let (decoded, _) = decode_varint_signed(&buf._ops[pos_after_guard - 2..]);
        assert_eq!(decoded, s, "guard descr slot not patched to snapshot index");
    }

    /// Phase B7 smoke test: create_snapshot patches the previous
    /// snapshot's NEEDS_PATCHING sentinel with SNAPSHOT_PREV_COMES_NEXT
    /// (-1) and then writes a new snapshot with its own
    /// NEEDS_PATCHING sentinel at the end.
    #[test]
    fn test_create_snapshot_prev_chain_b7() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);

        // First: a top snapshot with is_last=false so it leaves a
        // NEEDS_PATCHING sentinel for the next create_snapshot to chain
        // onto.
        let top_array = buf.new_array(1);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_smallint(7));
        let _top_idx = buf.create_top_snapshot(1, 10, top_array, &[], &[], false);
        // The snapshot_data now ends with NEEDS_PATCHING (bytes 0x7d 0xff).
        let n = buf._snapshot_data.len();
        assert_eq!(buf._snapshot_data[n - 2], 0x7d);
        assert_eq!(buf._snapshot_data[n - 1], 0xff);

        // Now append a parent snapshot: create_snapshot patches the
        // trailing sentinel to SNAPSHOT_PREV_COMES_NEXT and writes the
        // new snapshot record.
        let parent_array = buf.new_array(1);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_smallint(9));
        let _parent_idx = buf.create_snapshot(0, 5, parent_array, true);
        // The top snapshot's prev slot is now COMES_NEXT (-1 encodes
        // as 0x7f 0xff in the signed-varint 2-byte layout).
        // Decoding at the original offset should yield -1.
        let (decoded, _) = decode_varint_signed(&buf._snapshot_data[n - 2..n]);
        assert_eq!(decoded, SNAPSHOT_PREV_COMES_NEXT as i64);
    }

    /// Phase B6 parity: `_count` advances on EVERY recorded op, `_index`
    /// advances ONLY for ops whose `opclasses[opnum].type != 'v'`
    /// (opencoder.py:660-662). Interleave non-void (IntAdd) and void
    /// (GuardTrue) ops and verify the two counters progress according
    /// to the RPython rule.
    #[test]
    fn test_void_op_index_policy_b6() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        // Initial state: _start = _count = _index = max_num_inputargs = 2.
        assert_eq!(buf._count, 2);
        assert_eq!(buf._index, 2);

        // IntAdd (non-void Int result) — bumps both counters.
        let p1 = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(1), None);
        assert_eq!(p1, 2, "record_op returns pre-bump _index");
        assert_eq!(buf._count, 3);
        assert_eq!(buf._index, 3);

        // GuardTrue (void) — bumps _count only.
        let p2 = buf.record_op1(OpCode::GuardTrue, Box::ResOp(2), None);
        assert_eq!(
            p2, 3,
            "guard returns pre-bump _index (same as last non-void)"
        );
        assert_eq!(buf._count, 4);
        assert_eq!(buf._index, 3, "void must NOT bump _index");

        // IntSub (non-void) — bumps both again. Note _index picks up
        // from 3, not from _count=4, so the next non-void result
        // position is 3.
        let p3 = buf.record_op2(OpCode::IntSub, Box::ResOp(0), Box::ResOp(2), None);
        assert_eq!(p3, 3, "non-void after void uses the void's pre-bump _index");
        assert_eq!(buf._count, 5);
        assert_eq!(buf._index, 4);
    }

    /// Phase B5 smoke test: guard records trailing descr as 0 (two
    /// \\x00 bytes in the 2-byte signed-varint form).
    /// `patch_last_guard_descr_slot` rewinds 2 bytes and replaces them
    /// with the actual snapshot index via append_int — works for both
    /// 2-byte and 4-byte encodings of the patched value.
    #[test]
    fn test_guard_descr_placeholder_patch_b5() {
        // Phase B3's record_op1 writes opnum + arg + descr-int. For a
        // guard with has_descr + descr_index=0, the trailing 2 bytes
        // are the zero-placeholder.
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        let pos_before_patch = buf._pos;
        assert_eq!(
            buf._ops[pos_before_patch - 2],
            0u8,
            "guard descr placeholder missing"
        );
        assert_eq!(buf._ops[pos_before_patch - 1], 0u8);

        // Patch to a small snapshot index (fits in 2 bytes).
        buf.patch_last_guard_descr_slot(42);
        assert_eq!(
            buf._pos, pos_before_patch,
            "2-byte patch should keep _pos at the same offset"
        );
        // Verify we can decode it back.
        let tail_start = pos_before_patch - 2;
        let (decoded, consumed) = decode_varint_signed(&buf._ops[tail_start..]);
        assert_eq!(decoded, 42);
        assert_eq!(consumed, 2);

        // Patch to a large snapshot index (requires 4 bytes).
        let mut buf2 = TraceRecordBuffer::new(1, empty_sd());
        buf2.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        let pos_before = buf2._pos;
        buf2.patch_last_guard_descr_slot(MAX_VALUE);
        assert_eq!(
            buf2._pos,
            pos_before + 2,
            "4-byte patch should grow _pos by 2"
        );
    }

    /// Phase B4 + S smoke test: `_encode_descr` returns `global_index
    /// + 1` for descrs with a global index, else appends to `_descrs`
    /// and returns `all_descrs_len + len(_descrs) - 1 + 1` where
    /// `all_descrs_len = len(metainterp_sd.all_descrs)`
    /// (RPython opencoder.py:702-707).
    #[test]
    fn test_encode_descr_reads_metainterp_sd() {
        use std::sync::Arc;

        #[derive(Debug)]
        struct D {
            idx: i32,
        }
        impl majit_ir::Descr for D {
            fn index(&self) -> u32 {
                0
            }
            fn get_descr_index(&self) -> i32 {
                self.idx
            }
        }

        let mut sd = crate::MetaInterpStaticData::new();
        // Seed all_descrs with 7 dummies so the length drives the encoding.
        for _ in 0..7 {
            sd.all_descrs
                .get_mut()
                .unwrap()
                .push(Arc::new(D { idx: 0 }));
        }
        let mut buf = TraceRecordBuffer::new(0, Arc::new(sd));

        // Global descr returns `get_descr_index() + 1`.
        let d_global: majit_ir::DescrRef = Arc::new(D { idx: 3 });
        assert_eq!(buf._encode_descr(&d_global), 4);
        assert_eq!(buf._descrs.len(), 1, "global descr must not append");

        // Local descr encodes to all_descrs.len() + local_slot + 1 = 7 + 2.
        let d_local: majit_ir::DescrRef = Arc::new(D { idx: -1 });
        assert_eq!(buf._encode_descr(&d_local), 9);
        assert_eq!(buf._descrs.len(), 2);
    }

    /// Phase B3 smoke test: fixed-arity record_op* does NOT write a
    /// count varint (opencoder.py:642-650), only the opnum byte + args.
    /// Variadic record_op writes opnum + count + args. _count bumps
    /// once per call; _index bumps only for non-void results.
    #[test]
    fn test_record_op_wire_format_b3() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        let start_index = buf._index;
        let start_count = buf._count;
        let start_pos = buf._pos;
        // IntAdd is fixed-arity 2, result_type Int → non-void.
        // Args passed as RPython-encoded tagged values.
        let pos = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(1), None);
        assert_eq!(pos, start_index, "record_op returns pre-bump _index");
        assert_eq!(buf._count, start_count + 1);
        assert_eq!(buf._index, start_index + 1, "non-void should bump _index");
        // opnum byte at `start_pos` (= max_num_inputargs = 2, opencoder.py:500),
        // then 2 varints for args. IntAdd has no descr.
        assert_eq!(buf._ops[start_pos], OpCode::IntAdd.as_u16() as u8);
        // 2 args × 2-byte varint each = 4 bytes after opnum.
        assert_eq!(buf._pos, start_pos + 1 + 2 + 2, "opnum + 2 varint-2 args");

        // A void op must not bump _index but must bump _count.
        let void_start_index = buf._index;
        let void_start_count = buf._count;
        // GuardTrue is fixed-arity 1, result_type Void, has_descr.
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        assert_eq!(buf._index, void_start_index, "void should NOT bump _index");
        assert_eq!(buf._count, void_start_count + 1);
    }

    // ── M2 Step 2b · record_op_oprefs adapter tests ───────────────────

    /// Step 2b: `record_op_oprefs` with non-constant OpRef args must
    /// emit the same wire bytes as `record_op(&[Box::ResOp(...)])`.
    #[test]
    fn test_record_op_oprefs_resop_parity_2b() {
        let pool = crate::constant_pool::ConstantPool::new();
        let mut expected = TraceRecordBuffer::new(2, empty_sd());
        let mut actual = TraceRecordBuffer::new(2, empty_sd());
        let pos_e = expected.record_op(OpCode::IntAdd, &[Box::ResOp(0), Box::ResOp(1)], None);
        let pos_a = actual.record_op_oprefs(OpCode::IntAdd, &[iarg(0), iarg(1)], None, &pool);
        assert_eq!(pos_e, pos_a);
        assert_eq!(expected._ops[..expected._pos], actual._ops[..actual._pos]);
        assert_eq!(expected._count, actual._count);
        assert_eq!(expected._index, actual._index);
    }

    /// Step 2b: constant OpRef (Int) must resolve via ConstantPool and
    /// produce the same bytes as passing `Box::ConstInt(v)` directly.
    #[test]
    fn test_record_op_oprefs_const_int_2b() {
        let mut pool = crate::constant_pool::ConstantPool::new();
        let c = pool.get_or_insert_typed(42, Type::Int);
        assert!(c.is_constant());
        let mut expected = TraceRecordBuffer::new(1, empty_sd());
        let mut actual = TraceRecordBuffer::new(1, empty_sd());
        let pos_e = expected.record_op(OpCode::IntAdd, &[Box::ResOp(0), Box::ConstInt(42)], None);
        let pos_a = actual.record_op_oprefs(OpCode::IntAdd, &[iarg(0), c], None, &pool);
        assert_eq!(pos_e, pos_a);
        assert_eq!(expected._ops[..expected._pos], actual._ops[..actual._pos]);
    }

    /// Step 2b: constant OpRef (Float) must resolve to `Box::ConstFloat`.
    #[test]
    fn test_record_op_oprefs_const_float_2b() {
        let mut pool = crate::constant_pool::ConstantPool::new();
        let raw = (3.14_f64).to_bits() as i64;
        let c = pool.get_or_insert_typed(raw, Type::Float);
        assert!(c.is_constant());
        let mut expected = TraceRecordBuffer::new(1, empty_sd());
        let mut actual = TraceRecordBuffer::new(1, empty_sd());
        let pos_e = expected.record_op(
            OpCode::FloatAdd,
            &[Box::ResOp(0), Box::ConstFloat(raw as u64)],
            None,
        );
        let pos_a = actual.record_op_oprefs(OpCode::FloatAdd, &[farg(0), c], None, &pool);
        assert_eq!(pos_e, pos_a);
        assert_eq!(expected._ops[..expected._pos], actual._ops[..actual._pos]);
        // Both should have registered the float constant in the pool.
        assert_eq!(expected._floats, actual._floats);
    }

    /// Step 2b: constant OpRef (Ref) must resolve to `Box::ConstPtr`.
    #[test]
    fn test_record_op_oprefs_const_ref_2b() {
        let mut pool = crate::constant_pool::ConstantPool::new();
        let addr = 0xdead_beef_u64;
        let c = pool.get_or_insert_typed(addr as i64, Type::Ref);
        assert!(c.is_constant());
        let mut expected = TraceRecordBuffer::new(1, empty_sd());
        let mut actual = TraceRecordBuffer::new(1, empty_sd());
        let pos_e = expected.record_op(OpCode::PtrEq, &[Box::ResOp(0), Box::ConstPtr(addr)], None);
        let pos_a = actual.record_op_oprefs(OpCode::PtrEq, &[rarg(0), c], None, &pool);
        assert_eq!(pos_e, pos_a);
        assert_eq!(expected._ops[..expected._pos], actual._ops[..actual._pos]);
        assert_eq!(expected._refs, actual._refs);
    }

    /// Step 2b: an orphan constant OpRef (CONST_BIT set but no pool
    /// entry) is a genuine invariant break — the adapter must panic.
    #[test]
    #[should_panic(expected = "not in pool")]
    fn test_record_op_oprefs_orphan_constant_panics_2b() {
        let pool = crate::constant_pool::ConstantPool::new();
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let orphan = OpRef::const_int(7);
        buf.record_op_oprefs(OpCode::IntAdd, &[iarg(0), orphan], None, &pool);
    }

    // ── M2 Step 2c · ops / get_op_by_pos / last_op tests ──────────────

    /// Step 2c: `ops()` walks the byte stream and materializes every
    /// recorded op. Opcodes come back in record order.
    #[test]
    fn test_ops_materializer_2c() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        buf.record_input_arg(Type::Int);
        buf.record_input_arg(Type::Int);
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(1), None);
        let _ = buf.record_op2(OpCode::IntMul, Box::ResOp(2), Box::ResOp(0), None);
        let ops = buf.ops(None);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opcode, OpCode::IntAdd);
        assert_eq!(ops[1].opcode, OpCode::IntMul);
    }

    /// Step 2c: `get_op_by_pos` returns the Op whose `.pos` equals the
    /// requested OpRef. `ByteTraceIter` seeds `_fresh` at
    /// `max_num_inputargs` and then bumps it once per inputarg before
    /// any op, so with 2 inputargs the first op has `op.pos == IntOp(4)`
    /// (pyre-only disjoint-namespace behaviour documented on
    /// `ByteTraceIter::new`).
    #[test]
    fn test_get_op_by_pos_2c() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        buf.record_input_arg(Type::Int);
        buf.record_input_arg(Type::Int);
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(1), None);
        let _ = buf.record_op2(OpCode::IntMul, Box::ResOp(2), Box::ResOp(0), None);
        let first = buf.get_op_by_pos(iop(4), None).expect("first op present");
        assert_eq!(first.opcode, OpCode::IntAdd);
        assert_eq!(first.pos, iop(4));
        let second = buf.get_op_by_pos(iop(5), None).expect("second op present");
        assert_eq!(second.opcode, OpCode::IntMul);
        assert!(buf.get_op_by_pos(iop(99), None).is_none());
    }

    /// Step 2c: `last_op` returns the final recorded op.  On an empty
    /// buffer the result is `None`.
    #[test]
    fn test_last_op_2c() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        buf.record_input_arg(Type::Int);
        buf.record_input_arg(Type::Int);
        assert!(buf.last_op(None).is_none());
        let _ = buf.record_op2(OpCode::IntAdd, Box::ResOp(0), Box::ResOp(1), None);
        let last = buf.last_op(None).expect("one op recorded");
        assert_eq!(last.opcode, OpCode::IntAdd);
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        let last = buf.last_op(None).expect("two ops recorded");
        assert_eq!(last.opcode, OpCode::GuardTrue);
    }

    // ── M2 Step 2d · guard / close_loop / finish helpers tests ───────

    fn dummy_descr() -> majit_ir::DescrRef {
        use std::sync::Arc;
        #[derive(Debug)]
        struct D;
        impl majit_ir::Descr for D {
            fn index(&self) -> u32 {
                0
            }
            fn get_descr_index(&self) -> i32 {
                -1
            }
        }
        Arc::new(D)
    }

    /// Step 2d: `close_loop_oprefs` records a JUMP with no descr.
    #[test]
    fn test_close_loop_oprefs_2d() {
        let pool = crate::constant_pool::ConstantPool::new();
        let mut expected = TraceRecordBuffer::new(2, empty_sd());
        let mut actual = TraceRecordBuffer::new(2, empty_sd());
        expected.record_input_arg(Type::Int);
        expected.record_input_arg(Type::Int);
        actual.record_input_arg(Type::Int);
        actual.record_input_arg(Type::Int);
        let pos_e = expected.record_op(OpCode::Jump, &[Box::ResOp(0), Box::ResOp(1)], None);
        let pos_a =
            actual.close_loop_oprefs(&[OpRef::input_arg_int(0), OpRef::input_arg_int(1)], &pool);
        assert_eq!(pos_e, pos_a);
        assert_eq!(expected._ops[..expected._pos], actual._ops[..actual._pos]);
    }

    /// Step 2d: `close_loop_oprefs_with_descr` records a JUMP with the
    /// given tentative descr.  Matches
    /// `record_op(Jump, args, Some(&descr))`.
    #[test]
    fn test_close_loop_oprefs_with_descr_2d() {
        let pool = crate::constant_pool::ConstantPool::new();
        let descr = dummy_descr();
        let mut expected = TraceRecordBuffer::new(1, empty_sd());
        let mut actual = TraceRecordBuffer::new(1, empty_sd());
        expected.record_input_arg(Type::Int);
        actual.record_input_arg(Type::Int);
        let pos_e = expected.record_op(OpCode::Jump, &[Box::ResOp(0)], Some(&descr));
        let pos_a =
            actual.close_loop_oprefs_with_descr(&[OpRef::input_arg_int(0)], Some(&descr), &pool);
        assert_eq!(pos_e, pos_a);
        assert_eq!(expected._ops[..expected._pos], actual._ops[..actual._pos]);
    }

    /// Step 2d: `finish_oprefs` records a FINISH op with its terminal
    /// FailDescr.
    #[test]
    fn test_finish_oprefs_2d() {
        let pool = crate::constant_pool::ConstantPool::new();
        let descr = dummy_descr();
        let mut expected = TraceRecordBuffer::new(1, empty_sd());
        let mut actual = TraceRecordBuffer::new(1, empty_sd());
        expected.record_input_arg(Type::Int);
        actual.record_input_arg(Type::Int);
        let pos_e = expected.record_op(OpCode::Finish, &[Box::ResOp(0)], Some(&descr));
        let pos_a = actual.finish_oprefs(&[OpRef::input_arg_int(0)], &descr, &pool);
        assert_eq!(pos_e, pos_a);
        assert_eq!(expected._ops[..expected._pos], actual._ops[..actual._pos]);
    }

    /// Step 2c: a trace that contains constant args requires a pool to
    /// decode; `ops(Some(&mut pool))` must succeed when the same pool
    /// was used to record the constants.
    #[test]
    fn test_ops_materializer_with_constants_2c() {
        let mut pool = crate::constant_pool::ConstantPool::new();
        let c = pool.get_or_insert_typed(42, Type::Int);
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        buf.record_input_arg(Type::Int);
        buf.record_op_oprefs(OpCode::IntAdd, &[OpRef::input_arg_int(0), c], None, &pool);
        let ops = buf.ops(Some(&mut pool));
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::IntAdd);
    }

    /// Step 2b: `record_op_oprefs` with a descr must encode the descr
    /// exactly as `record_op(&[Box], Some(&descr))` would.
    #[test]
    fn test_record_op_oprefs_with_descr_2b() {
        use std::sync::Arc;
        #[derive(Debug)]
        struct D;
        impl majit_ir::Descr for D {
            fn index(&self) -> u32 {
                0
            }
            fn get_descr_index(&self) -> i32 {
                -1
            }
        }
        let descr: majit_ir::DescrRef = Arc::new(D);
        let pool = crate::constant_pool::ConstantPool::new();
        let mut expected = TraceRecordBuffer::new(1, empty_sd());
        let mut actual = TraceRecordBuffer::new(1, empty_sd());
        let pos_e = expected.record_op(OpCode::CallN, &[Box::ResOp(0)], Some(&descr));
        let pos_a = actual.record_op_oprefs(
            OpCode::CallN,
            &[OpRef::input_arg_int(0)],
            Some(&descr),
            &pool,
        );
        assert_eq!(pos_e, pos_a);
        assert_eq!(expected._ops[..expected._pos], actual._ops[..actual._pos]);
        assert_eq!(expected._descrs.len(), actual._descrs.len());
    }

    #[test]
    fn test_trace_cut_point_roundtrip_b1() {
        // opencoder.py:567-575 cut_point / cut_at tuple shape:
        // (_pos, _count, _index, len(_snapshot_data), len(_snapshot_array_data)).
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        let p0 = buf.cut_point();
        // append_byte advances _pos; append_snapshot_*_int extends the chains.
        buf.append_byte(0x42);
        buf.append_snapshot_data_int(123);
        buf.append_snapshot_array_data_int(42);
        let p1 = buf.cut_point();
        assert_ne!(p0, p1);
        buf.cut_at(p0);
        assert_eq!(buf._pos, p0._pos);
        assert_eq!(buf._count, p0._count);
        assert_eq!(buf._index, p0._index);
        // RPython's cut_at intentionally does NOT rewind the snapshot chains
        // (they grow monotonically). Only _pos/_count/_index come back.
    }

    /// Step 1 (SnapshotIterator): an empty top snapshot
    /// (`create_empty_top_snapshot`, jitcode_index == -1) has an empty
    /// framestack and `size == vable.total_length + vref.total_length + 3`.
    /// Mirrors opencoder.py:212-213 early-return branch.
    #[test]
    fn test_snapshot_iterator_empty_top() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        // Need a preceding guard for patch_last_guard_descr_slot.
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        let vable = [TraceRecordBuffer::_encode_box_position(0)];
        let vref: [i64; 0] = [];
        let snap = buf.create_empty_top_snapshot(&vable, &vref);

        let it = buf.get_snapshot_iter(snap as usize);
        assert!(it.framestack.is_empty(), "empty top → empty framestack");
        // vable has 1 entry, vref has 0, so size = 1 + 0 + 3 = 4.
        assert_eq!(it.size, 4);
        assert_eq!(it.iter_vable_array().total_length, 1);
        assert_eq!(it.iter_vref_array().total_length, 0);
    }

    /// Step 1 (SnapshotIterator): `capture_resumedata` with a single
    /// frame produces a non-empty framestack in bottom-up order with
    /// outermost == innermost (length 1).
    #[test]
    fn test_snapshot_iterator_single_frame() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        // Build the innermost-frame active-box array.
        let arr = buf.new_array(2);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_box_position(0));
        buf._add_box_to_storage(TraceRecordBuffer::_encode_box_position(1));
        let vable: [i64; 0] = [];
        let vref: [i64; 0] = [];
        // jitcode_index=7, pc=42.
        let snap = buf.create_top_snapshot(7, 42, arr, &vable, &vref, true);

        let it = buf.get_snapshot_iter(snap as usize);
        assert_eq!(it.framestack.len(), 1, "one frame in framestack");
        let (jit_idx, pc) = it.unpack_jitcode_pc(it.framestack[0]);
        assert_eq!(jit_idx, 7);
        assert_eq!(pc, 42);
        let items: Vec<i64> = it.iter_array(it.framestack[0]).collect();
        assert_eq!(items.len(), 2);
    }

    /// Step 1c: `capture_resumedata` on a framestack with a single
    /// frame drives `create_top_snapshot_from_frame` and returns a
    /// snapshot_index consistent with `get_snapshot_iter`.
    /// Parent chain is empty (n == 0 ⇒ is_last = true), so no
    /// `create_snapshot_from_frame` calls fire.
    #[test]
    fn test_capture_resumedata_single_frame_c() {
        use std::sync::Arc;

        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);

        // Build a minimal jitcode with 1 int reg and a LIVE op at pc 0.
        let mut builder = crate::jitcode::JitCodeBuilder::new();
        let mut jc = builder.finish();
        const LIVE_OP: u8 = 0x42;
        {
            let body = jc.body_mut();
            body.c_num_regs_i = 1;
            body.c_num_regs_r = 0;
            body.c_num_regs_f = 0;
            body.code = vec![LIVE_OP, 0x00, 0x00];
        }
        jc.set_index(11);
        let jc = Arc::new(jc);

        // all_liveness: len_i=1 len_r=0 len_f=0, bitmask = 0b1.
        let all_liveness: Vec<u8> = vec![1, 0, 0, 0b0000_0001];

        let mut frame = crate::pyjitpl::MIFrame::new(jc, 0);
        frame.pc = 0;
        frame.int_regs[0] = Some(majit_ir::OpRef::int_op(1));
        frame.int_values[0] = Some(123);

        let mut stack = vec![frame];
        let snap = buf.capture_resumedata(
            &mut stack,
            &[],
            &[],
            /* pool */ None,
            LIVE_OP,
            &all_liveness,
            /* after_residual_call */ true,
        );

        let it = buf.get_snapshot_iter(snap as usize);
        assert_eq!(it.framestack.len(), 1, "single-frame capture");
        let (j, p) = it.unpack_jitcode_pc(it.framestack[0]);
        assert_eq!(j, 11, "frame.jitcode.index encoded into snapshot");
        assert_eq!(p, 0, "frame.pc encoded into snapshot");
        let boxes: Vec<i64> = it.iter_array(it.framestack[0]).collect();
        assert_eq!(
            boxes,
            vec![TraceRecordBuffer::_encode_box_position(1)],
            "live int register emitted via get_list_of_active_boxes"
        );
    }

    /// Step 1c: `capture_resumedata` with an empty framestack takes
    /// the `create_empty_top_snapshot_from_boxes` branch. Snapshot's
    /// `jitcode_index == -1`, `framestack` empty.
    #[test]
    fn test_capture_resumedata_empty_framestack_c() {
        let mut buf = TraceRecordBuffer::new(1, empty_sd());
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);

        let mut stack: Vec<crate::pyjitpl::MIFrame> = Vec::new();
        let snap = buf.capture_resumedata(&mut stack, &[], &[], None, 0, &[], false);
        let it = buf.get_snapshot_iter(snap as usize);
        assert!(
            it.framestack.is_empty(),
            "empty framestack → empty snapshot"
        );
    }

    /// Step 1 (SnapshotIterator): multi-frame chain produces a
    /// framestack ordered outermost-first. Simulates RPython's
    /// capture_resumedata + `_ensure_parent_resumedata` pattern by
    /// emitting one `create_top_snapshot` followed by
    /// `create_snapshot` calls for parent frames.
    #[test]
    fn test_snapshot_iterator_multi_frame_order() {
        let mut buf = TraceRecordBuffer::new(2, empty_sd());
        let _ = buf.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        // innermost frame (n=2, jit=3 pc=30)
        let inner = buf.new_array(1);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_box_position(0));
        let vable: [i64; 0] = [];
        let vref: [i64; 0] = [];
        let snap = buf.create_top_snapshot(3, 30, inner, &vable, &vref, false);
        // middle frame (n=1, jit=2 pc=20)
        let middle = buf.new_array(1);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_box_position(1));
        let _ = buf.create_snapshot(2, 20, middle, false);
        // outermost frame (n=0, jit=1 pc=10)
        let outer = buf.new_array(0);
        let _ = buf.create_snapshot(1, 10, outer, true);

        let it = buf.get_snapshot_iter(snap as usize);
        assert_eq!(it.framestack.len(), 3, "three frames in chain");
        // opencoder.py:217 `framestack.reverse()` → bottom-up
        // (outermost first).
        let (j0, p0) = it.unpack_jitcode_pc(it.framestack[0]);
        let (j1, p1) = it.unpack_jitcode_pc(it.framestack[1]);
        let (j2, p2) = it.unpack_jitcode_pc(it.framestack[2]);
        assert_eq!((j0, p0), (1, 10), "framestack[0] == outermost");
        assert_eq!((j1, p1), (2, 20), "framestack[1] == middle");
        assert_eq!((j2, p2), (3, 30), "framestack[2] == innermost");
    }

    // ── Phase C: RPython test_opencoder.py parity ports ─────────────────
    // Line-by-line ports of the canonical tests in
    // `rpython/jit/metainterp/test/test_opencoder.py`. Each Rust test cites
    // the upstream line it mirrors so regressions stay traceable to the
    // Python fixture.

    /// test_opencoder.py:71 `TestOpencoder.test_simple_iterator` —
    /// canonical two-op `IntAdd` chain with a small-int constant argument.
    /// Ensures `record_op2` + `ByteTraceIter::new_with_pool` round-trip
    /// the op list with the expected arity, opcode, and arg identities.
    #[test]
    fn test_simple_iterator_c() {
        let mut t = TraceRecordBuffer::new(2, empty_sd());
        let i0 = t.record_input_arg(Type::Int);
        let i1 = t.record_input_arg(Type::Int);
        // add = t.record_op(rop.INT_ADD, [i0, i1])
        let add_pos = t.record_op2(
            OpCode::IntAdd,
            Box::ResOp(i0.raw()),
            Box::ResOp(i1.raw()),
            None,
        );
        // t.record_op(rop.INT_ADD, [add, ConstInt(1)])
        let _ = t.record_op2(OpCode::IntAdd, Box::ResOp(add_pos), Box::ConstInt(1), None);

        // `self.unpack(t)` equivalent — ConstInt(1) decodes via the pool.
        let mut pool = crate::constant_pool::ConstantPool::new();
        let mut it = ByteTraceIter::new_with_pool(
            &t,
            t._start as usize,
            t._pos,
            t.max_num_inputargs,
            Some(&mut pool),
        );
        let fresh_i0 = it.inputargs[0];
        let fresh_i1 = it.inputargs[1];
        let l0 = it.next().expect("first IntAdd");
        let l1 = it.next().expect("second IntAdd");
        assert!(it.done(), "only two ops recorded");
        // assert l[0].opnum == rop.INT_ADD
        assert_eq!(l0.opcode, OpCode::IntAdd);
        // assert l[1].opnum == rop.INT_ADD
        assert_eq!(l1.opcode, OpCode::IntAdd);
        // assert l[0].getarg(0) is i0; getarg(1) is i1
        assert_eq!(l0.args[0], fresh_i0);
        assert_eq!(l0.args[1], fresh_i1);
        // assert l[1].getarg(0) is l[0]
        assert_eq!(l1.args[0], l0.pos);
        // assert l[1].getarg(1).getint() == 1 — pool-resolved constant.
        drop(it);
        assert_eq!(pool.get_value(l1.args[1]), Some(majit_ir::Value::Int(1)));
    }

    /// test_opencoder.py:250 `test_constint_small` —
    /// SMALL_INT values encode to 2 or 4 bytes depending on the varint
    /// boundary, leave `_consts_bigint` untouched (no bigint pool use),
    /// and round-trip to the original value via `ByteTraceIter::_untag`.
    ///
    /// RPython uses Hypothesis; pyre iterates a hand-picked representative
    /// sample that spans the 2-byte / 4-byte split (upstream's `-2**12`
    /// boundary) plus SMALL_INT_START / SMALL_INT_STOP extremes.
    #[test]
    fn test_constint_small_c() {
        // Representative sample matching RPython's Hypothesis strategy
        // `integers(SMALL_INT_START, SMALL_INT_STOP - 1)`.
        let sample: &[i64] = &[
            0,
            1,
            -1,
            127,
            -128,
            (1 << 12) - 1,
            -(1 << 12),
            (1 << 12),
            -(1 << 12) - 1,
            SMALL_INT_START,
            SMALL_INT_STOP - 1,
        ];
        for &num in sample {
            let mut t = TraceRecordBuffer::new(0, empty_sd());
            // t.append_int(t._encode(ConstInt(num))) — SMALL_INT path.
            t.append_int(TraceRecordBuffer::_encode_smallint(num));
            // assert t._consts_bigint == 0
            assert_eq!(
                t._consts_bigint, 0,
                "SMALL_INT must not hit bigint pool for {num}"
            );
            // _pos counts only the appended varint (no inputargs reserve
            // because `max_num_inputargs == 0`).
            let expected_len = if (-(1i64 << 12)..(1i64 << 12)).contains(&num) {
                2
            } else {
                4
            };
            assert_eq!(
                t._pos, expected_len,
                "SMALL_INT {num} should encode as {expected_len} bytes, got {}",
                t._pos,
            );
            // Round-trip: `it._next()` reads the tagged i64, `_untag`
            // returns an OpRef that resolves to `Value::Int(num)` via the
            // constant pool.
            let mut pool = crate::constant_pool::ConstantPool::new();
            let mut it =
                ByteTraceIter::new_with_pool(&t, 0, t._pos, t.max_num_inputargs, Some(&mut pool));
            let tagged = it._next();
            let opref = it._untag(tagged);
            drop(it);
            assert_eq!(
                pool.get_value(opref),
                Some(majit_ir::Value::Int(num)),
                "SMALL_INT {num} round-trip via ByteTraceIter",
            );
        }
    }

    /// test_opencoder.py:200 `TestOpencoder.test_liveranges` —
    /// NEW_WITH_VTABLE + GUARD_TRUE + capture_resumedata with non-empty
    /// vable/vref arrays but an empty framestack. Drives the snapshot
    /// chain through `update_liveranges` and asserts every input +
    /// NEW_WITH_VTABLE slot is marked live at the guard's index (4).
    #[test]
    fn test_liveranges_c() {
        let mut t = TraceRecordBuffer::new(3, empty_sd());
        let i0 = t.record_input_arg(Type::Int);
        let i1 = t.record_input_arg(Type::Int);
        let i2 = t.record_input_arg(Type::Int);
        // p0 = t.record_op(rop.NEW_WITH_VTABLE, [], descr=SomeDescr())
        let some_descr: majit_ir::DescrRef = majit_ir::descr::make_size_descr(64);
        let p0 = t.record_op0(OpCode::NewWithVtable, Some(&some_descr));
        // t.record_op(rop.GUARD_TRUE, [i0])
        let _ = t.record_op1(OpCode::GuardTrue, Box::ResOp(i0.raw()), None);
        // t.capture_resumedata([], [i1, i2, p0], [p0, i1])
        let vable = [Box::ResOp(i1.raw()), Box::ResOp(i2.raw()), Box::ResOp(p0)];
        let vref = [Box::ResOp(p0), Box::ResOp(i1.raw())];
        let mut stack: Vec<crate::pyjitpl::MIFrame> = Vec::new();
        let _ = t.capture_resumedata(&mut stack, &vable, &vref, None, 0, &[], false);
        // assert t.get_live_ranges() == [4, 4, 4, 4]
        assert_eq!(t.get_live_ranges(), vec![4, 4, 4, 4]);
    }

    /// test_opencoder.py:163 `TestOpencoder.test_cut_trace_from` —
    /// record a prefix, take a cut_point, continue recording, then call
    /// `cut_trace_from(cut_point, new_inputargs)` to produce a CutTrace
    /// that iterates only the post-cut ops with `new_inputargs` acting
    /// as iterator-local inputargs. The first op's args should resolve
    /// to the fresh inputarg OpRefs (not the original trace positions).
    #[test]
    fn test_cut_trace_from_c() {
        let mut t = TraceRecordBuffer::new(3, empty_sd());
        let i0 = t.record_input_arg(Type::Int);
        let i1 = t.record_input_arg(Type::Int);
        let _i2 = t.record_input_arg(Type::Int);
        // add1 = t.record_op(rop.INT_ADD, [i0, i1])
        let add1 = t.record_op2(
            OpCode::IntAdd,
            Box::ResOp(i0.raw()),
            Box::ResOp(i1.raw()),
            None,
        );
        // cut_point = t.cut_point()
        let cut_point = t.cut_point();
        // add2 = t.record_op(rop.INT_ADD, [add1, i1])
        let add2 = t.record_op2(OpCode::IntAdd, Box::ResOp(add1), Box::ResOp(i1.raw()), None);
        // t.record_op(rop.GUARD_TRUE, [add2])
        let _ = t.record_op1(OpCode::GuardTrue, Box::ResOp(add2), None);
        // t.record_op(rop.INT_SUB, [add2, add1])
        let _ = t.record_op2(OpCode::IntSub, Box::ResOp(add2), Box::ResOp(add1), None);
        // t2 = t.cut_trace_from(cut_point, [add1, i1])
        // RPython `cut_trace_from(cut, inputargs)` takes BoxInt/Float/Ref objects
        // carrying `Box.type` (history.py:182); pyre's typed-OpRef variants encode
        // the same type tag — `IntOp(p)` for IntAdd's result, the `i1` InputArgInt
        // verbatim.
        let cut = t.cut_trace_from(cut_point, vec![OpRef::int_op(add1), i1]);
        // (i0, i1), l, iter = self.unpack(t2)
        let mut it = cut.get_iter();
        let fresh_add1 = it.inputargs[0];
        let fresh_i1 = it.inputargs[1];
        let mut ops = Vec::new();
        while let Some(op) = it.next() {
            ops.push(op);
        }
        // assert len(l) == 3
        assert_eq!(ops.len(), 3, "cut sub-trace contains 3 ops");
        // assert l[0].getarglist() == [add1_fresh, i1_fresh]
        // (RPython writes this as `[i0, i1]` because the test rebinds
        // the outer `i0, i1` identifiers to the cut's unpacked
        // inputargs; the pyre port keeps the names distinct.)
        assert_eq!(ops[0].opcode, OpCode::IntAdd);
        assert_eq!(ops[0].args.as_slice(), &[fresh_add1, fresh_i1]);
        // Second op is the guard; third is the INT_SUB tail.
        assert_eq!(ops[1].opcode, OpCode::GuardTrue);
        assert_eq!(ops[2].opcode, OpCode::IntSub);
    }

    /// test_opencoder.py:178 `TestOpencoder.test_virtualizable_virtualref` —
    /// assert the vable/vref arrays decode in the canonical order:
    /// `[p0, i1, i2]` (the virtualizable tail box moves to index 0) and
    /// `[p0, i1]` (vref verbatim).
    #[test]
    fn test_virtualizable_virtualref_c() {
        let mut t = TraceRecordBuffer::new(3, empty_sd());
        let _i0 = t.record_input_arg(Type::Int);
        let i1 = t.record_input_arg(Type::Int);
        let i2 = t.record_input_arg(Type::Int);
        let some_descr: majit_ir::DescrRef = majit_ir::descr::make_size_descr(64);
        let p0 = t.record_op0(OpCode::NewWithVtable, Some(&some_descr));
        let _ = t.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        let vable = [Box::ResOp(i1.raw()), Box::ResOp(i2.raw()), Box::ResOp(p0)];
        let vref = [Box::ResOp(p0), Box::ResOp(i1.raw())];
        let mut stack: Vec<crate::pyjitpl::MIFrame> = Vec::new();
        let snap = t.capture_resumedata(&mut stack, &vable, &vref, None, 0, &[], false);
        let it = t.get_snapshot_iter(snap as usize);
        // Empty framestack — equivalent to `assert not l[1].framestack`.
        assert!(it.framestack.is_empty());
        let vable_decoded: Vec<i64> = it.iter_vable_array().collect();
        let vref_decoded: Vec<i64> = it.iter_vref_array().collect();
        let p0_tag = TraceRecordBuffer::_encode_box_position(p0);
        let i1_tag = TraceRecordBuffer::_encode_box_position(i1.raw());
        let i2_tag = TraceRecordBuffer::_encode_box_position(i2.raw());
        // `virtualizables == [p0, i1, i2]` after the last-first rotation.
        assert_eq!(vable_decoded, vec![p0_tag, i1_tag, i2_tag]);
        // `vref_boxes == [p0, i1]` verbatim.
        assert_eq!(vref_decoded, vec![p0_tag, i1_tag]);
    }

    /// test_opencoder.py:189 `TestOpencoder.test_virtualizable_bug` —
    /// 128 virtualizable boxes (p0 + 127 × i1) exercise the array-length
    /// varint at the 1-byte / 2-byte threshold. The canonical assertion
    /// is that the snapshot decoder returns the exact same box list in
    /// order; i.e. `SnapshotIterator::unpack_array(vable_array)` must
    /// decode `[p0, i1, i1, ..., i1]` (128 entries) without truncation.
    #[test]
    fn test_virtualizable_bug_c() {
        let mut t = TraceRecordBuffer::new(3, empty_sd());
        let _i0 = t.record_input_arg(Type::Int);
        let i1 = t.record_input_arg(Type::Int);
        let _i2 = t.record_input_arg(Type::Int);
        let some_descr: majit_ir::DescrRef = majit_ir::descr::make_size_descr(64);
        let p0 = t.record_op0(OpCode::NewWithVtable, Some(&some_descr));
        let _ = t.record_op1(OpCode::GuardTrue, Box::ResOp(0), None);
        // capture_resumedata([], [i1] * 127 + [p0], [p0, i1])
        let mut vable: Vec<Box> = (0..127).map(|_| Box::ResOp(i1.raw())).collect();
        vable.push(Box::ResOp(p0));
        let vref = [Box::ResOp(p0), Box::ResOp(i1.raw())];
        let mut stack: Vec<crate::pyjitpl::MIFrame> = Vec::new();
        let snap = t.capture_resumedata(&mut stack, &vable, &vref, None, 0, &[], false);
        // Unpack the snapshot and decode the vable/vref arrays. The
        // 128-entry vable array stresses the varint length field
        // (threshold at 2**7 / 2**14 = 128). If the encoder framed the
        // length as a single byte, the decoder would truncate or spill.
        let it = t.get_snapshot_iter(snap as usize);
        let vable_decoded: Vec<i64> = it.iter_vable_array().collect();
        let vref_decoded: Vec<i64> = it.iter_vref_array().collect();
        assert_eq!(vable_decoded.len(), 128, "128 vable entries decoded");
        assert_eq!(vref_decoded.len(), 2, "2 vref entries decoded");
        // `_list_of_boxes_virtualizable_from_boxes` writes the last entry
        // first, then the remaining forward — matching RPython's
        // `_list_of_boxes(virt)` with an already-rotated list. The
        // decoded stream should therefore open with p0 and continue with
        // 127 × i1.
        let p0_tag = TraceRecordBuffer::_encode_box_position(p0);
        let i1_tag = TraceRecordBuffer::_encode_box_position(i1.raw());
        assert_eq!(vable_decoded[0], p0_tag);
        for (idx, entry) in vable_decoded.iter().enumerate().skip(1) {
            assert_eq!(*entry, i1_tag, "vable[{idx}] == i1");
        }
        assert_eq!(vref_decoded[0], p0_tag);
        assert_eq!(vref_decoded[1], i1_tag);
    }

    /// test_opencoder.py:263 `test_varint_hypothesis` —
    /// `encode_varint_signed` / `decode_varint_signed` round-trip over
    /// the full `[MIN_VALUE, MAX_VALUE]` range, including the case where
    /// the encoded bytes sit behind an arbitrary prefix (decoder's
    /// `start` offset). Pyre samples the hypothesis surface with a
    /// hand-picked set of boundary values.
    #[test]
    fn test_varint_hypothesis_c() {
        let values: &[i64] = &[
            0,
            1,
            -1,
            (1 << 14) - 1,
            -(1 << 14),
            1 << 14,
            -(1 << 14) - 1,
            (1 << 20),
            -(1 << 20),
            MIN_VALUE,
            MAX_VALUE,
        ];
        let prefixes: &[&[u8]] = &[&[], &[0xAA], &[0x00, 0xFF, 0x42]];
        for &i in values {
            let mut encoded = Vec::new();
            encode_varint_signed(&mut encoded, i);
            // Bare decode.
            let (res, consumed) = decode_varint_signed(&encoded);
            assert_eq!(res, i, "bare decode roundtrip for {i}");
            assert_eq!(consumed, encoded.len(), "consumed full encoding for {i}");
            // Decode after an arbitrary prefix — `decode_varint_signed`
            // operates on a slice starting at the varint, so pyre mirrors
            // RPython's `decode_varint_signed(prefix + b, len(prefix))`
            // by slicing off the prefix before the call.
            for &prefix in prefixes {
                let mut combined = prefix.to_vec();
                combined.extend_from_slice(&encoded);
                let (res, consumed) = decode_varint_signed(&combined[prefix.len()..]);
                assert_eq!(res, i, "prefixed decode roundtrip for {i}");
                assert_eq!(consumed, encoded.len(), "prefixed consumed for {i}");
            }
        }
    }
}
