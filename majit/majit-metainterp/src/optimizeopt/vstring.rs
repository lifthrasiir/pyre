/// String virtualization optimization pass.
///
/// Translated from rpython/jit/metainterp/optimizeopt/vstring.py.
///
/// Virtualizes string operations so that short, fully-tracked strings never
/// allocate at runtime:
///
/// - NEWSTR with constant length -> virtual string (characters tracked)
/// - STRSETITEM on virtual string -> stores character in tracked array
/// - STRGETITEM on virtual string -> returns tracked character
/// - STRLEN on virtual string -> constant length
/// - COPYSTRCONTENT -> updates tracked characters or emits forced copies
///
/// When a virtual string escapes (used in an unhandled context), it is
/// "forced": NEWSTR + STRSETITEM ops are emitted to materialize it.
use std::collections::HashMap;

use majit_ir::{EffectInfo, OopSpecIndex, Op, OpCode, OpRef, Value};

use crate::optimizeopt::info::{
    PtrInfo, StrPtrInfo, VStringConcatInfo, VStringPlainInfo, VStringSliceInfo, VStringVariant,
};
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

/// vstring.py:18 MAX_CONST_LEN
const MAX_CONST_LEN: usize = 100;

/// vstring.py mode_string / mode_unicode discriminators.
pub const mode_string: u8 = 0;
pub const mode_unicode: u8 = 1;

/// history.py:377-387 get_const_ptr_for_string(s)
///
/// Creates a constant GcRef from byte-string character values.
/// Returns None when the runtime hook is not installed.
pub fn get_const_ptr_for_string(chars: &[i64], ctx: &OptContext) -> Option<majit_ir::GcRef> {
    let alloc_fn = ctx.string_constant_alloc.as_ref()?;
    let gcref = alloc_fn(chars, false);
    if gcref.is_null() { None } else { Some(gcref) }
}

/// history.py:390-402 get_const_ptr_for_unicode(s)
///
/// Creates a constant GcRef from unicode character values.
/// Returns None when the runtime hook is not installed.
pub fn get_const_ptr_for_unicode(chars: &[i64], ctx: &OptContext) -> Option<majit_ir::GcRef> {
    let alloc_fn = ctx.string_constant_alloc.as_ref()?;
    let gcref = alloc_fn(chars, true);
    if gcref.is_null() { None } else { Some(gcref) }
}
/// vstring.py:371-381 _int_add(optstring, box1, box2)
///
/// Constant-folding INT_ADD: folds add-0 and const+const at the optimizer
/// level. Non-constant adds emit an INT_ADD operation.
pub fn _int_add(box1: OpRef, box2: OpRef, ctx: &mut OptContext) -> OpRef {
    if let Some(v1) = ctx.get_constant_int(box1) {
        if v1 == 0 {
            return box2;
        }
        if let Some(v2) = ctx.get_constant_int(box2) {
            return ctx.emit_constant_int(v1 + v2);
        }
    } else if ctx.get_constant_int(box2) == Some(0) {
        return box1;
    }
    let op = Op::new(OpCode::IntAdd, &[box1, box2]);
    ctx.emit_for_force(op)
}

/// vstring.py:337-369 copy_str_content(optstring, srcbox, targetbox,
///     srcoffsetbox, offsetbox, lengthbox, mode, need_next_offset=True)
///
/// Emits either inline STRGETITEM/STRSETITEM (for small constant lengths)
/// or a single COPYSTRCONTENT/COPYUNICODECONTENT operation.
pub fn copy_str_content(
    ctx: &mut OptContext,
    srcbox: OpRef,
    targetbox: OpRef,
    srcoffsetbox: OpRef,
    offsetbox: OpRef,
    lengthbox: OpRef,
    mode: u8,
    need_next_offset: bool,
) -> OpRef {
    let srcbox = ctx.get_box_replacement(srcbox);
    let (set_opcode, copy_opcode, get_opcode) = if mode != 0 {
        (
            OpCode::Unicodesetitem,
            OpCode::Copyunicodecontent,
            OpCode::Unicodegetitem,
        )
    } else {
        (
            OpCode::Strsetitem,
            OpCode::Copystrcontent,
            OpCode::Strgetitem,
        )
    };

    // vstring.py:341-347: determine inline threshold M using intbound
    let srcoffset_bound = ctx.getintbound(srcoffsetbox);
    let lgt_bound = ctx.getintbound(lengthbox);
    // vstring.py:343: isinstance(srcbox, ConstPtr)
    // getconst handles constant_types_for_numbering.
    let src_is_const = ctx
        .getconst(srcbox)
        .is_some_and(|(_, tp)| tp == majit_ir::Type::Ref);
    let m = if src_is_const && srcoffset_bound.is_constant() {
        5
    } else {
        2
    };

    // vstring.py:347: if lgt.is_constant() and lgt.get_constant_int() <= M
    if lgt_bound.is_constant() {
        let length = lgt_bound.get_constant_int();
        if length >= 0 && (length as usize) <= m {
            // vstring.py:350-357: inline STRGETITEM/STRSETITEM
            // RPython calls optstring.strgetitem(None, srcbox, srcoffsetbox, mode)
            // which tries PtrInfo lookup first (virtual chars), falling back to
            // emitting STRGETITEM.
            let mut src_offset = srcoffsetbox;
            let mut dst_offset = offsetbox;
            let one = ctx.emit_constant_int(1);
            for _i in 0..length {
                let charbox = {
                    // vstring.py:350-351: charbox = optstring.strgetitem(
                    //     None, srcbox, srcoffsetbox, mode)
                    // vstring.py:495: vindex = self.getintbound(index)
                    let vindex = ctx.getintbound(src_offset);
                    let resolved_idx = if vindex.is_constant() {
                        Some(vindex.get_constant_int())
                    } else {
                        None
                    };
                    // vstring.py:496-503: virtual Plain/Concat dispatch
                    let from_info = resolved_idx.and_then(|idx| {
                        ctx.getptrinfo(srcbox)
                            .and_then(|info| info.as_ref().strgetitem(idx, &*ctx))
                    });
                    if let Some(ch) = from_info {
                        ch
                    } else if let Some(idx) = resolved_idx {
                        // vstring.py:394: isinstance(strbox, ConstPtr) + ConstInt
                        let from_const = match ctx.getconst(srcbox) {
                            Some((raw, majit_ir::Type::Ref)) if raw != 0 => {
                                let r = majit_ir::GcRef(raw as usize);
                                ctx.string_content_resolver
                                    .as_deref()
                                    .and_then(|resolver| resolver(r, mode))
                                    .and_then(|chars| chars.get(idx as usize).copied())
                            }
                            _ => None,
                        };
                        if let Some(ch_val) = from_const {
                            ctx.emit_constant_int(ch_val)
                        } else {
                            let getitem_op = Op::new(get_opcode, &[srcbox, src_offset]);
                            ctx.emit_for_force(getitem_op)
                        }
                    } else {
                        let getitem_op = Op::new(get_opcode, &[srcbox, src_offset]);
                        ctx.emit_for_force(getitem_op)
                    }
                };
                src_offset = _int_add(src_offset, one, ctx);
                let setitem_op = Op::new(set_opcode, &[targetbox, dst_offset, charbox]);
                ctx.emit_for_force(setitem_op);
                dst_offset = _int_add(dst_offset, one, ctx);
            }
            return dst_offset;
        }
    }

    // vstring.py:359-368: bulk COPYSTRCONTENT
    let next_offset = if need_next_offset {
        _int_add(offsetbox, lengthbox, ctx)
    } else {
        offsetbox // caller doesn't need it
    };
    let copy_op = Op::new(
        copy_opcode,
        &[srcbox, targetbox, srcoffsetbox, offsetbox, lengthbox],
    );
    ctx.emit_for_force(copy_op);
    next_offset
}

/// vstring.py:132-140 / 185-205 / 230-233 / 309-317
/// string_copy_parts — recursive dispatch to copy string content
/// into an already-allocated target string at `offsetbox`.
/// Returns the updated offset after the copy.
///
/// This is the Rust equivalent of RPython's per-subclass
/// `string_copy_parts` / `initialize_forced_string` polymorphic dispatch.
pub fn string_copy_parts(
    opref: OpRef,
    targetbox: OpRef,
    offsetbox: OpRef,
    mode: u8,
    ctx: &mut OptContext,
) -> OpRef {
    let resolved = ctx.get_box_replacement(opref);

    // Extract variant data without keeping PtrInfo borrow alive.
    // RPython dispatches via subclass; we dispatch via enum variant.
    enum Action {
        /// vstring.py:194-205 VStringPlainInfo.initialize_forced_string
        Plain(Vec<Option<OpRef>>),
        /// vstring.py:230-233 VStringSliceInfo.string_copy_parts
        Slice {
            s: OpRef,
            start: OpRef,
            lgtop: OpRef,
        },
        /// vstring.py:309-317 VStringConcatInfo.string_copy_parts
        Concat { vleft: OpRef, vright: OpRef },
        /// vstring.py:132-140 StrPtrInfo.string_copy_parts (base class, non-virtual)
        NonVirtual,
    }

    let action = match ctx.getptrinfo(resolved) {
        Some(info) => match info.as_ref() {
            PtrInfo::Str(sinfo) if sinfo.is_virtual() => match &sinfo.variant {
                VStringVariant::Plain(p) => Action::Plain(p._chars.clone()),
                VStringVariant::Slice(s) => Action::Slice {
                    s: s.s,
                    start: s.start,
                    lgtop: s.lgtop,
                },
                VStringVariant::Concat(c) => Action::Concat {
                    vleft: c.vleft,
                    vright: c.vright,
                },
                VStringVariant::Ptr => Action::NonVirtual,
            },
            _ => Action::NonVirtual,
        },
        None => Action::NonVirtual,
    };

    let set_opcode = if mode != 0 {
        OpCode::Unicodesetitem
    } else {
        OpCode::Strsetitem
    };

    match action {
        Action::Plain(chars) => {
            // vstring.py:194-205 VStringPlainInfo.initialize_forced_string
            let mut offset = offsetbox;
            let one = ctx.emit_constant_int(1);
            for ch in &chars {
                if let Some(ch_ref) = ch {
                    let ch_resolved = ctx.get_box_replacement(*ch_ref);
                    let setitem_op = Op::new(set_opcode, &[targetbox, offset, ch_resolved]);
                    ctx.emit_for_force(setitem_op);
                }
                offset = _int_add(offset, one, ctx);
            }
            offset
        }
        Action::Slice { s, start, lgtop } => {
            // vstring.py:230-233 VStringSliceInfo.string_copy_parts
            copy_str_content(ctx, s, targetbox, start, offsetbox, lgtop, mode, true)
        }
        Action::Concat { vleft, vright } => {
            // vstring.py:309-317 VStringConcatInfo.string_copy_parts
            let offset = string_copy_parts(vleft, targetbox, offsetbox, mode, ctx);
            string_copy_parts(vright, targetbox, offset, mode, ctx)
        }
        Action::NonVirtual => {
            // vstring.py:132-140 StrPtrInfo.string_copy_parts (base class)
            // lengthbox = self.getstrlen(op, optstring, mode)
            // srcbox = self.force_box(op, optstring)  -- no-op for non-virtual
            let lengthbox = ctx.getstrlen_opref(resolved, mode);
            let srcbox = force_child_for_string(resolved, ctx);
            let zero = ctx.emit_constant_int(0);
            copy_str_content(
                ctx, srcbox, targetbox, zero, offsetbox, lengthbox, mode, true,
            )
        }
    }
}

/// Force a string-typed OpRef if it's virtual. Used by string_copy_parts
/// base class path (vstring.py:138: srcbox = self.force_box(op, optstring)).
fn force_child_for_string(opref: OpRef, ctx: &mut OptContext) -> OpRef {
    let resolved = ctx.get_box_replacement(opref);
    if ctx.get_ptr_info(resolved).is_some_and(|i| i.is_virtual()) {
        let mut info = ctx.take_ptr_info(resolved).unwrap();
        let forced = info.force_box(resolved, ctx);
        return ctx.get_box_replacement(forced);
    }
    resolved
}

/// vstring.py:413 `class OptString(Optimization)` — stateless string/unicode
/// optimization pass. All per-string state lives on `PtrInfo::Str`
/// (length, mode, virtual variant, lgtop). STRLEN/UNICODELEN caching is
/// wired through `OptPure` via `pure_from_args1`.
pub struct OptString;

impl OptString {
    pub fn new() -> Self {
        OptString
    }

    fn get_plain_info<'a>(
        &self,
        opref: OpRef,
        ctx: &'a OptContext,
    ) -> Option<&'a VStringPlainInfo> {
        let resolved = ctx.get_box_replacement(opref);
        match ctx.get_ptr_info(resolved) {
            Some(PtrInfo::Str(sinfo)) => match &sinfo.variant {
                VStringVariant::Plain(info) => Some(info),
                _ => None,
            },
            _ => None,
        }
    }

    fn get_plain_info_mut<'a>(
        &self,
        opref: OpRef,
        ctx: &'a mut OptContext,
    ) -> Option<&'a mut VStringPlainInfo> {
        let resolved = ctx.get_box_replacement(opref);
        match ctx.get_ptr_info_mut(resolved) {
            Some(PtrInfo::Str(sinfo)) => match &mut sinfo.variant {
                VStringVariant::Plain(info) => Some(info),
                _ => None,
            },
            _ => None,
        }
    }

    fn is_virtual_plain(&self, opref: OpRef, ctx: &OptContext) -> bool {
        self.get_plain_info(opref, ctx).is_some()
    }

    fn get_concat_info<'a>(
        &self,
        opref: OpRef,
        ctx: &'a OptContext,
    ) -> Option<&'a VStringConcatInfo> {
        let resolved = ctx.get_box_replacement(opref);
        match ctx.get_ptr_info(resolved) {
            Some(PtrInfo::Str(sinfo)) => match &sinfo.variant {
                VStringVariant::Concat(info) => Some(info),
                _ => None,
            },
            _ => None,
        }
    }

    fn is_virtual_concat(&self, opref: OpRef, ctx: &OptContext) -> bool {
        self.get_concat_info(opref, ctx).is_some()
    }

    fn get_slice_info<'a>(
        &self,
        opref: OpRef,
        ctx: &'a OptContext,
    ) -> Option<&'a VStringSliceInfo> {
        let resolved = ctx.get_box_replacement(opref);
        match ctx.get_ptr_info(resolved) {
            Some(PtrInfo::Str(sinfo)) => match &sinfo.variant {
                VStringVariant::Slice(info) => Some(info),
                _ => None,
            },
            _ => None,
        }
    }

    fn is_virtual_slice(&self, opref: OpRef, ctx: &OptContext) -> bool {
        self.get_slice_info(opref, ctx).is_some()
    }

    /// vstring.py: read the string mode (0 = byte string, 1 = unicode) from
    /// the installed `StrPtrInfo`. Returns 0 when no PtrInfo is set — callers
    /// inside the pass only hit this path for constant/forwarded refs where
    /// the mode is not observable and defaulting to string is harmless.
    fn get_mode(&self, opref: OpRef, ctx: &OptContext) -> u8 {
        let resolved = ctx.get_box_replacement(opref);
        match ctx.get_ptr_info(resolved) {
            Some(PtrInfo::Str(sinfo)) => sinfo.mode,
            _ => 0,
        }
    }

    /// vstring.py:76-103 StrPtrInfo.force_box — delegate to PtrInfo::force_box.
    fn force_box(&mut self, opref: OpRef, ctx: &mut OptContext) -> OpRef {
        let resolved = ctx.get_box_replacement(opref);
        if ctx.get_ptr_info(resolved).is_some_and(|i| i.is_virtual()) {
            let mut info = ctx.take_ptr_info(resolved).unwrap();
            let forced = info.force_box(resolved, ctx);
            return ctx.get_box_replacement(forced);
        }
        resolved
    }

    /// Emit a SameAsI op that produces a constant integer value.
    ///
    /// We need a way to reference constant values as OpRefs. We emit a
    /// SameAsI(dummy) and record the constant in the context.
    fn emit_constant_int(&self, value: i64, ctx: &mut OptContext) -> OpRef {
        // Emit a dummy SameAsI to get an OpRef, then record the constant.
        let op = Op::new(OpCode::SameAsI, &[OpRef::NONE]);
        let opref = ctx.emit(op);
        ctx.make_constant(opref, Value::Int(value));
        opref
    }

    /// vstring.py:110-119 StrPtrInfo.getstrlen — delegates to
    /// OptContext::getstrlen_opref which handles per-variant dispatch
    /// and lgtop caching (box identity reuse).
    fn getstrlen(&self, opref: OpRef, ctx: &mut OptContext) -> OpRef {
        let resolved = ctx.get_box_replacement(opref);
        let mode = self.get_mode(resolved, ctx);
        ctx.getstrlen_opref(resolved, mode)
    }

    /// vstring.py:112-114 — get the strlen OpRef if already known,
    /// without emitting a new op. Checks lgtop first (RPython parity),
    /// then structurally-known constant length on the virtual variant.
    fn getstrlen_if_known(&self, opref: OpRef, ctx: &mut OptContext) -> Option<OpRef> {
        let resolved = ctx.get_box_replacement(opref);
        // vstring.py:112: if self.lgtop is not None: return self.lgtop
        if let Some(info) = ctx.getptrinfo(resolved) {
            if let Some(lgtop) = info.as_ref().get_cached_lgtop() {
                return Some(lgtop);
            }
        }
        // vstring.py:174: self.lgtop = ConstInt(len(self._chars))
        // RPython creates a pure ConstInt — no op emission.
        let known_len = ctx.getptrinfo(resolved).and_then(|info| {
            let mode = self.get_mode(resolved, ctx);
            info.as_ref().get_known_str_length(ctx, mode)
        });
        if let Some(len) = known_len {
            let len_opref = ctx.make_constant_int(len);
            // Cache in lgtop for identity reuse
            ctx.set_str_lgtop(resolved, len_opref);
            return Some(len_opref);
        }
        None
    }

    /// vstring.py:486-517 OptString.strgetitem + vstring.py:393-403 _strgetitem
    ///
    /// Tries virtual dispatch (Plain/Slice/Concat), then ConstPtr constant
    /// resolution. Returns None only when the char can't be determined.
    fn strgetitem(&self, opref: OpRef, index: i64, ctx: &mut OptContext) -> Option<OpRef> {
        let resolved = ctx.get_box_replacement(opref);
        // Virtual dispatch: PtrInfo::Str → VStringInfo.strgetitem
        let from_virtual = ctx
            .getptrinfo(resolved)
            .and_then(|info| info.as_ref().strgetitem(index, &*ctx));
        if from_virtual.is_some() {
            return from_virtual;
        }
        // vstring.py:393-403 _strgetitem: isinstance(strbox, ConstPtr)
        let mode = self.get_mode(resolved, ctx);
        match ctx.getconst(resolved) {
            Some((raw, majit_ir::Type::Ref)) if raw != 0 => {
                let r = majit_ir::GcRef(raw as usize);
                let ch_val = ctx
                    .string_content_resolver
                    .as_deref()
                    .and_then(|resolver| resolver(r, mode))
                    .and_then(|chars| chars.get(index as usize).copied())?;
                Some(ctx.emit_constant_int(ch_val))
            }
            _ => None,
        }
    }

    /// Get the known length of a virtual string as a constant, if available.
    /// Delegates to `PtrInfo::Str::getstrlen` which walks Plain/Slice/Concat
    /// variants. Matches vstring.py:171/251/281 `getstrlen()` per-variant.
    fn get_known_length(&self, opref: OpRef, ctx: &OptContext) -> Option<i64> {
        let resolved = ctx.get_box_replacement(opref);
        let info = ctx.getptrinfo(resolved)?;
        let mode = self.get_mode(resolved, ctx);
        info.as_ref().get_known_str_length(ctx, mode)
    }

    /// vstring.py:440-453 `_optimize_NEWSTR(self, op, mode)`. Virtualize if
    /// length is a small constant; otherwise install a non-virtual StrPtrInfo
    /// and emit the op. `postprocess_NEWSTR` (vstring.py:455-459) registers
    /// `pure(STRLEN, op) = length_arg` for CSE via the pure cache.
    fn _optimize_newstr(&mut self, op: &Op, mode: u8, ctx: &mut OptContext) -> OptimizationResult {
        let len_ref = op.arg(0);
        if let Some(len) = ctx.get_constant_int(len_ref) {
            if len >= 0 && (len as usize) <= MAX_CONST_LEN {
                // vstring.py:450: self.make_vstring_plain(op, mode, length)
                ctx.set_ptr_info(
                    op.pos,
                    PtrInfo::Str(StrPtrInfo {
                        lenbound: None,
                        lgtop: None,
                        mode,
                        length: len as i32,
                        variant: VStringVariant::Plain(VStringPlainInfo {
                            _chars: vec![None; len as usize],
                        }),
                        last_guard_pos: -1,
                        cached_vinfo: std::cell::RefCell::new(None),
                    }),
                );
                return OptimizationResult::Remove;
            }
        }
        // vstring.py:452: self.make_nonnull_str(op, mode); return self.emit(op)
        ctx.make_nonnull_str(op.pos, mode);
        // vstring.py:455-459 postprocess_NEWSTR / postprocess_NEWUNICODE:
        //   self.pure_from_args1(mode.STRLEN, op, op.getarg(0))
        let strlen_opcode = if mode == 1 {
            OpCode::Unicodelen
        } else {
            OpCode::Strlen
        };
        ctx.register_pure_from_args1(strlen_opcode, op.pos, len_ref);
        OptimizationResult::PassOn
    }

    /// Handle STRSETITEM: if target is virtual Plain and index is constant, track.
    fn optimize_strsetitem(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let str_ref = ctx.get_box_replacement(op.arg(0));
        let idx_ref = op.arg(1);
        let char_ref = op.arg(2);
        let char_resolved = ctx.get_box_replacement(char_ref);

        if let Some(idx) = ctx.get_constant_int(idx_ref) {
            if let Some(info) = self.get_plain_info_mut(str_ref, ctx) {
                let i = idx as usize;
                if i < info._chars.len() {
                    info._chars[i] = Some(char_resolved);
                    return OptimizationResult::Remove;
                }
            }
        }
        // Not virtual or index not constant -> force and emit.
        self.force_if_virtual(str_ref, ctx);
        OptimizationResult::PassOn
    }

    /// Handle STRGETITEM: if source is virtual, resolve the character.
    fn optimize_strgetitem(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let str_ref = ctx.get_box_replacement(op.arg(0));
        let idx_ref = op.arg(1);

        if let Some(idx) = ctx.get_constant_int(idx_ref) {
            if let Some(ch_ref) = self.strgetitem(str_ref, idx, ctx) {
                let ch_resolved = ctx.get_box_replacement(ch_ref);
                ctx.replace_op(op.pos, ch_resolved);
                return OptimizationResult::Remove;
            }
        }
        // Not fully resolved -> force the string.
        self.force_if_virtual(str_ref, ctx);
        OptimizationResult::PassOn
    }

    /// vstring.py:525-533 _optimize_STRLEN
    fn optimize_strlen(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let mode = if op.opcode == OpCode::Unicodelen {
            1u8
        } else {
            0u8
        };
        // vstring.py:526-527
        let arg1 = ctx.get_box_replacement(op.arg(0));
        let has_info = ctx.getptrinfo(arg1).is_some();
        if has_info {
            // vstring.py:529: lgtop = opinfo.getstrlen(arg1, self, mode)
            let lgtop = ctx.getstrlen_opref(arg1, mode);
            // vstring.py:531: self.make_equal_to(op, lgtop)
            ctx.make_equal_to(op.pos, lgtop);
            return OptimizationResult::Remove;
        }
        // vstring.py:533: return self.emit(op)
        OptimizationResult::PassOn
    }

    fn get_constant_int_bound(&self, opref: OpRef, ctx: &OptContext) -> Option<i64> {
        ctx.get_int_bound(opref)
            .filter(|bound| bound.is_constant())
            .map(|bound| bound.get_constant_int())
            .or_else(|| ctx.get_constant_int(opref))
    }

    /// vstring.py:556-589 _optimize_COPYSTRCONTENT
    fn optimize_copystrcontent(
        &mut self,
        op: &Op,
        mode: u8,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // copystrcontent(src, dst, src_start, dst_start, length)
        let src_ref = ctx.get_box_replacement(op.arg(0));
        let dst_ref = ctx.get_box_replacement(op.arg(1));
        let src_start_ref = op.arg(2);
        let dst_start_ref = op.arg(3);
        let length_ref = op.arg(4);
        let src_info = ctx.getptrinfo(src_ref);
        let src_is_virtual_or_constant = src_info
            .as_ref()
            .is_some_and(|info| info.as_ref().is_virtual() || info.as_ref().is_constant());
        let dst_virtual = self.is_virtual_plain(dst_ref, ctx);
        let src_start = self.get_constant_int_bound(src_start_ref, ctx);
        let dst_start = self.get_constant_int_bound(dst_start_ref, ctx);
        let length = self.get_constant_int_bound(length_ref, ctx);

        if length == Some(0) {
            return OptimizationResult::Remove;
        }

        if let (Some(src_start), Some(dst_start), Some(length)) = (src_start, dst_start, length) {
            if src_is_virtual_or_constant
                && (length < 20 || (src_is_virtual_or_constant && dst_virtual))
            {
                let getitem_opcode = if mode == mode_unicode {
                    OpCode::Unicodegetitem
                } else {
                    OpCode::Strgetitem
                };
                let setitem_opcode = if mode == mode_unicode {
                    OpCode::Unicodesetitem
                } else {
                    OpCode::Strsetitem
                };
                let mut dst_chars = Vec::with_capacity(length as usize);
                for index in 0..length {
                    let char_ref =
                        if let Some(ch_ref) = self.strgetitem(src_ref, src_start + index, ctx) {
                            ctx.get_box_replacement(ch_ref)
                        } else {
                            // vstring.py:580-581 → _strgetitem → emit_extra
                            let index_ref = ctx.make_constant_int(src_start + index);
                            ctx.emit_extra(
                                ctx.current_pass_idx,
                                Op::new(getitem_opcode, &[src_ref, index_ref]),
                            )
                        };
                    if dst_virtual {
                        dst_chars.push(Some(char_ref));
                    } else {
                        // vstring.py:585-589: self.emit_extra(new_op)
                        let dst_index_ref = ctx.make_constant_int(dst_start + index);
                        ctx.emit_extra(
                            ctx.current_pass_idx,
                            Op::new(setitem_opcode, &[dst_ref, dst_index_ref, char_ref]),
                        );
                    }
                }
                if dst_virtual {
                    if let Some(info) = self.get_plain_info_mut(dst_ref, ctx) {
                        for (index, ch_ref) in dst_chars.into_iter().enumerate() {
                            let dst_index = (dst_start as usize) + index;
                            if dst_index < info._chars.len() {
                                info._chars[dst_index] = ch_ref;
                            }
                        }
                    }
                }
                return OptimizationResult::Remove;
            }
        }

        // vstring.py:590-593: fallback — emit via copy_str_content
        // which may still inline small constant-length copies.
        copy_str_content(
            ctx,
            op.arg(0),
            op.arg(1),
            op.arg(2),
            op.arg(3),
            op.arg(4),
            mode,
            false, // need_next_offset=False
        );
        OptimizationResult::Remove
    }

    /// Force a string if it is virtual.
    fn force_if_virtual(&mut self, opref: OpRef, ctx: &mut OptContext) {
        let resolved = ctx.get_box_replacement(opref);
        if self.is_virtual(resolved, ctx) {
            self.force_box(resolved, ctx);
        }
    }

    /// Check if an OpRef references a virtual string (after forwarding).
    #[allow(dead_code)]
    fn is_virtual(&self, opref: OpRef, ctx: &OptContext) -> bool {
        let resolved = ctx.get_box_replacement(opref);
        ctx.get_ptr_info(resolved)
            .is_some_and(|info| info.is_virtual())
    }

    /// vstring.py:383-391 _int_sub — constant-fold if both args are constant,
    /// otherwise emit INT_SUB.
    fn int_sub(&self, a: OpRef, b: OpRef, ctx: &mut OptContext) -> OpRef {
        if let Some(vb) = ctx.get_constant_int(b) {
            if vb == 0 {
                return a;
            }
            if let Some(va) = ctx.get_constant_int(a) {
                return self.emit_constant_int(va - vb, ctx);
            }
        }
        let op = Op::new(OpCode::IntSub, &[a, b]);
        ctx.emit(op)
    }

    /// vstring.py: postprocess — after STRLEN on a known-length string,
    /// record as pure (for CSE with OptPure).
    fn postprocess_strlen(&self, op: &Op, ctx: &mut OptContext) {
        // vstring.py: postprocess_STRLEN → make_nonnull_str
        let mode = if op.opcode == OpCode::Strlen {
            0u8
        } else {
            1u8
        };
        ctx.make_nonnull_str(op.arg(0), mode);
        let str_ref = ctx.get_box_replacement(op.arg(0));
        if let Some(len) = self.get_known_length(str_ref, ctx) {
            let _ = len;
        }
    }

    fn force_args_if_virtual(&mut self, op: &Op, ctx: &mut OptContext) {
        // Collect refs first to avoid borrow issues.
        let args: Vec<OpRef> = op
            .args
            .iter()
            .map(|a| ctx.get_box_replacement(*a))
            .collect();
        for arg in args {
            if self.is_virtual(arg, ctx) {
                self.force_box(arg, ctx);
            }
        }
    }

    /// vstring.py:594-621 optimize_CALL_I — dispatch oopspec calls to
    /// specialized handlers. Str* variants get `mode_string`, Uni* variants
    /// get `mode_unicode`, paralleling the RPython table walk via
    /// `_OS_offset_uni`.
    fn optimize_oopspec_call(
        &mut self,
        op: &Op,
        ei: &EffectInfo,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        match ei.oopspecindex {
            OopSpecIndex::StrConcat => self.opt_call_stroruni_str_concat(op, mode_string, ctx),
            OopSpecIndex::StrSlice => self.opt_call_stroruni_str_slice(op, mode_string, ctx),
            OopSpecIndex::StrEqual => self.opt_call_stroruni_str_equal(op, mode_string, ctx),
            OopSpecIndex::StrCmp => self.opt_call_stroruni_str_cmp(op, mode_string, ctx),
            OopSpecIndex::UniConcat => self.opt_call_stroruni_str_concat(op, mode_unicode, ctx),
            OopSpecIndex::UniSlice => self.opt_call_stroruni_str_slice(op, mode_unicode, ctx),
            OopSpecIndex::UniEqual => self.opt_call_stroruni_str_equal(op, mode_unicode, ctx),
            OopSpecIndex::UniCmp => self.opt_call_stroruni_str_cmp(op, mode_unicode, ctx),
            OopSpecIndex::ShrinkArray => self.opt_call_shrink_array(op, ctx),
            _ => {
                self.force_args_if_virtual(op, ctx);
                OptimizationResult::PassOn
            }
        }
    }

    /// vstring.py:653-661 opt_call_stroruni_STR_CONCAT
    fn opt_call_stroruni_str_concat(
        &mut self,
        op: &Op,
        mode: u8,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if op.num_args() >= 3 {
            let vleft = ctx.get_box_replacement(op.arg(1));
            let vright = ctx.get_box_replacement(op.arg(2));
            ctx.make_nonnull_str(vleft, mode);
            ctx.make_nonnull_str(vright, mode);
            ctx.set_ptr_info(
                op.pos,
                PtrInfo::Str(StrPtrInfo {
                    lenbound: None,
                    lgtop: None,
                    mode,
                    length: -1,
                    variant: VStringVariant::Concat(VStringConcatInfo {
                        vleft,
                        vright,
                        _is_virtual: true,
                    }),
                    last_guard_pos: -1,
                    cached_vinfo: std::cell::RefCell::new(None),
                }),
            );
            return OptimizationResult::Remove;
        }
        self.force_args_if_virtual(op, ctx);
        OptimizationResult::PassOn
    }

    /// vstring.py:662-690 opt_call_stroruni_STR_SLICE
    fn opt_call_stroruni_str_slice(
        &mut self,
        op: &Op,
        mode: u8,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if op.num_args() >= 4 {
            let mut s = ctx.get_box_replacement(op.arg(1));
            ctx.make_nonnull_str(s, mode);
            let mut start = ctx.get_box_replacement(op.arg(2));
            let stop = ctx.get_box_replacement(op.arg(3));
            let lgtop = self.int_sub(stop, start, ctx);
            if let Some(info) = self.get_slice_info(s, ctx) {
                let source = info.s;
                let source_start = info.start;
                s = source;
                start = _int_add(source_start, start, ctx);
            }
            // vstring.py:220-225: VStringSliceInfo.__init__ sets
            // self.lgtop = length on the inherited StrPtrInfo field.
            ctx.set_ptr_info(
                op.pos,
                PtrInfo::Str(StrPtrInfo {
                    lenbound: None,
                    lgtop: Some(lgtop),
                    mode,
                    length: -1,
                    variant: VStringVariant::Slice(VStringSliceInfo { s, start, lgtop }),
                    last_guard_pos: -1,
                    cached_vinfo: std::cell::RefCell::new(None),
                }),
            );
            return OptimizationResult::Remove;
        }
        self.force_args_if_virtual(op, ctx);
        OptimizationResult::PassOn
    }

    /// vstring.py:692-733 opt_call_stroruni_STR_EQUAL
    fn opt_call_stroruni_str_equal(
        &mut self,
        op: &Op,
        mode: u8,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if op.num_args() < 3 {
            self.force_args_if_virtual(op, ctx);
            return OptimizationResult::PassOn;
        }
        // vstring.py:693-696
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let arg2 = ctx.get_box_replacement(op.arg(2));
        let i1 = ctx.getptrinfo(arg1).is_some();
        let i2 = ctx.getptrinfo(arg2).is_some();
        // vstring.py:698-705: l1box = i1.getstrlen(arg1, self, mode)
        let l1box = if i1 {
            Some(ctx.getstrlen_opref(arg1, mode))
        } else {
            None
        };
        let l2box = if i2 {
            Some(ctx.getstrlen_opref(arg2, mode))
        } else {
            None
        };
        // vstring.py:706-712: isinstance(ConstInt) + different values
        if let (Some(l1), Some(l2)) = (l1box, l2box) {
            let l1c = ctx.get_constant_int(l1);
            let l2c = ctx.get_constant_int(l2);
            if let (Some(v1), Some(v2)) = (l1c, l2c) {
                if v1 != v2 {
                    ctx.make_constant(op.pos, Value::Int(0));
                    return OptimizationResult::Remove;
                }
            }
        }
        // vstring.py:714-718: handle_str_equal_level1 both directions
        if let Some(result) = self.handle_str_equal_level1(arg1, arg2, op, mode, ctx) {
            return result;
        }
        if let Some(result) = self.handle_str_equal_level1(arg2, arg1, op, mode, ctx) {
            return result;
        }
        // vstring.py:720-724: handle_str_equal_level2 both directions
        if let Some(result) = self.handle_str_equal_level2(arg1, arg2, op, mode, ctx) {
            return result;
        }
        if let Some(result) = self.handle_str_equal_level2(arg2, arg1, op, mode, ctx) {
            return result;
        }
        // vstring.py:727-732: nonnull fallback with same_box check
        let a_nonnull = i1 && self.is_known_nonnull(arg1, ctx);
        let b_nonnull = i2 && self.is_known_nonnull(arg2, ctx);
        if a_nonnull && b_nonnull {
            // vstring.py:728: l1box.same_box(l2box)
            // history.py:204: same identity OR both constants with equal value
            let same_len = match (l1box, l2box) {
                (Some(a), Some(b)) if a == b => true,
                (Some(a), Some(b)) => {
                    ctx.get_constant_int(a).is_some()
                        && ctx.get_constant_int(a) == ctx.get_constant_int(b)
                }
                _ => false,
            };
            let oopspec = if same_len {
                OopSpecIndex::StreqLengthok
            } else {
                OopSpecIndex::StreqNonnull
            };
            if let Some(result) = self.generate_modified_call(oopspec, &[arg1, arg2], op, ctx) {
                return result;
            }
        }
        self.force_args_if_virtual(op, ctx);
        OptimizationResult::PassOn
    }

    /// vstring.py:735-787 handle_str_equal_level1
    fn handle_str_equal_level1(
        &self,
        arg1: OpRef,
        arg2: OpRef,
        op: &Op,
        mode: u8,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        // vstring.py:740-741: l2box = i2.getstrlen(arg2, self, mode)
        let i2 = ctx.getptrinfo(arg2).is_some();
        let l2box = if i2 {
            Some(ctx.getstrlen_opref(arg2, mode))
        } else {
            None
        };
        let l2_const = l2box.and_then(|r| ctx.get_constant_int(r));
        // vstring.py:742-756: isinstance(l2box, ConstInt) checks
        if let Some(l2val) = l2_const {
            if l2val == 0 {
                // vstring.py:744-755: len-0 check
                if self.is_known_nonnull(arg1, ctx) {
                    // vstring.py:745: self.make_nonnull_str(arg1, mode)
                    ctx.make_nonnull_str(arg1, mode);
                    // vstring.py:747: lengthbox = i1.getstrlen(arg1, self, mode)
                    let lengthbox = ctx.getstrlen_opref(arg1, mode);
                    let zero = ctx.emit_constant_int(0);
                    let mut eq_op = Op::new(OpCode::IntEq, &[lengthbox, zero]);
                    eq_op.pos = op.pos;
                    return Some(OptimizationResult::Emit(eq_op));
                }
            }
            if l2val == 1 {
                // vstring.py:758-759: l1box = i1.getstrlen(arg1, self, mode)
                let i1 = ctx.getptrinfo(arg1).is_some();
                let l1box = if i1 {
                    Some(ctx.getstrlen_opref(arg1, mode))
                } else {
                    None
                };
                let l1_const = l1box.and_then(|r| ctx.get_constant_int(r));
                if l1_const == Some(1) {
                    // vstring.py:761-768: both length 1 → compare chars
                    let c1 = self.strgetitem(arg1, 0, ctx);
                    let c2 = self.strgetitem(arg2, 0, ctx);
                    if let (Some(ch1), Some(ch2)) = (c1, c2) {
                        let mut eq_op = Op::new(OpCode::IntEq, &[ch1, ch2]);
                        eq_op.pos = op.pos;
                        return Some(OptimizationResult::Emit(eq_op));
                    }
                }
                // vstring.py:769-774: arg1 is a virtual slice, arg2 is length 1
                let resolved1 = ctx.get_box_replacement(arg1);
                if let Some(info) = self.get_slice_info(resolved1, ctx) {
                    let source = info.s;
                    let start = info.start;
                    let length = info.lgtop;
                    if let Some(vchar) = self.strgetitem(arg2, 0, ctx) {
                        return self.generate_modified_call(
                            OopSpecIndex::StreqSliceChar,
                            &[source, start, length, vchar],
                            op,
                            ctx,
                        );
                    }
                }
            }
        }
        // vstring.py:776-787: arg2 is null
        if self.is_known_null(arg2, ctx) {
            if self.is_known_nonnull(arg1, ctx) {
                ctx.make_constant(op.pos, Value::Int(0));
                return Some(OptimizationResult::Remove);
            }
            if self.is_known_null(arg1, ctx) {
                ctx.make_constant(op.pos, Value::Int(1));
                return Some(OptimizationResult::Remove);
            }
            // vstring.py:784: PTR_EQ against CONST_NULL (ref-null, not int-zero)
            let null_const = ctx.emit_constant_ref(majit_ir::GcRef::NULL);
            let mut eq_op = Op::new(OpCode::PtrEq, &[arg1, null_const]);
            eq_op.pos = op.pos;
            return Some(OptimizationResult::Emit(eq_op));
        }
        None
    }

    /// vstring.py:789-814 handle_str_equal_level2
    fn handle_str_equal_level2(
        &self,
        arg1: OpRef,
        arg2: OpRef,
        op: &Op,
        mode: u8,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        // vstring.py:792-794: l2box = i2.getstrlen(arg1, self, mode)
        // RPython calls getstrlen on i2 (arg2's info) with arg1 as the op.
        let i2 = ctx.getptrinfo(arg2).is_some();
        let l2box = if i2 {
            Some(ctx.getstrlen_for(arg2, arg1, mode))
        } else {
            None
        };
        // vstring.py:795-805: l2info = self.getintbound(l2box)
        if let Some(l2ref) = l2box {
            let l2info = ctx.getintbound(l2ref);
            if l2info.is_constant() && l2info.get_constant_int() == 1 {
                // vstring.py:799: vchar = self.strgetitem(None, arg2, CONST_0, mode)
                if let Some(vchar) = self.strgetitem(arg2, 0, ctx) {
                    // vstring.py:800-804
                    let oopspec = if self.is_known_nonnull(arg1, ctx) {
                        OopSpecIndex::StreqNonnullChar
                    } else {
                        OopSpecIndex::StreqChecknullChar
                    };
                    return self.generate_modified_call(oopspec, &[arg1, vchar], op, ctx);
                }
            }
        }
        // vstring.py:807-813: if arg1 is a virtual slice
        let resolved1 = ctx.get_box_replacement(arg1);
        if let Some(info) = self.get_slice_info(resolved1, ctx) {
            let source = info.s;
            let start = info.start;
            let length = info.lgtop;
            let oopspec = if self.is_known_nonnull(arg2, ctx) {
                OopSpecIndex::StreqSliceNonnull
            } else {
                OopSpecIndex::StreqSliceChecknull
            };
            return self.generate_modified_call(oopspec, &[source, start, length, arg2], op, ctx);
        }
        None
    }

    /// vstring.py:776 `i2 and i2.is_null()` — uses getptrinfo which
    /// synthesizes ConstPtrInfo for constant refs.
    fn is_known_null(&self, opref: OpRef, ctx: &OptContext) -> bool {
        if let Some(info) = ctx.getptrinfo(opref) {
            return info.is_null();
        }
        false
    }

    /// vstring.py:777,800,808 `i1 and i1.is_nonnull()` — uses getptrinfo
    /// which synthesizes ConstPtrInfo for constant refs.
    fn is_known_nonnull(&self, opref: OpRef, ctx: &OptContext) -> bool {
        if let Some(info) = ctx.getptrinfo(opref) {
            return info.is_nonnull() || info.is_virtual();
        }
        false
    }

    /// vstring.py:853-860 generate_modified_call
    ///
    /// Look up the calldescr and func_ptr for the given oopspec in the
    /// CallInfoCollection, and emit a CALL_I with those args.
    fn generate_modified_call(
        &self,
        oopspec: OopSpecIndex,
        args: &[OpRef],
        result_op: &Op,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        // Clone Arc to avoid borrow conflict with ctx
        let cic = ctx.callinfocollection.clone()?;
        let &(ref calldescr, func_addr) = cic.callinfo_for_oopspec(oopspec)?;
        let func_const = ctx.alloc_op_position_typed(majit_ir::Type::Int);
        ctx.make_constant(func_const, Value::Int(func_addr as i64));
        let mut call_args = vec![func_const];
        call_args.extend_from_slice(args);
        let mut call_op = Op::with_descr(OpCode::CallI, &call_args, calldescr.clone());
        call_op.pos = result_op.pos;
        Some(OptimizationResult::Emit(call_op))
    }

    /// vstring.py:816-838 opt_call_stroruni_STR_CMP
    fn opt_call_stroruni_str_cmp(
        &mut self,
        op: &Op,
        mode: u8,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if op.num_args() < 3 {
            self.force_args_if_virtual(op, ctx);
            return OptimizationResult::PassOn;
        }
        let arg1 = ctx.get_box_replacement(op.arg(1));
        let arg2 = ctx.get_box_replacement(op.arg(2));
        // vstring.py:819-822: bail out if either info is missing
        let i1 = ctx.getptrinfo(arg1).is_some();
        let i2 = ctx.getptrinfo(arg2).is_some();
        if !i1 || !i2 {
            self.force_args_if_virtual(op, ctx);
            return OptimizationResult::PassOn;
        }
        // vstring.py:823-824: l1box = i1.getstrlen(arg1, self, mode)
        let l1box = ctx.getstrlen_opref(arg1, mode);
        let l2box = ctx.getstrlen_opref(arg2, mode);
        // vstring.py:825-828: isinstance(ConstInt) and both == 1
        let l1c = ctx.get_constant_int(l1box);
        let l2c = ctx.get_constant_int(l2box);
        if l1c == Some(1) && l2c == Some(1) {
            // vstring.py:830-836: extract chars and INT_SUB
            if let (Some(char1), Some(char2)) = (
                self.strgetitem(op.arg(1), 0, ctx),
                self.strgetitem(op.arg(2), 0, ctx),
            ) {
                let result = self.int_sub(char1, char2, ctx);
                ctx.replace_op(op.pos, result);
                return OptimizationResult::Remove;
            }
        }
        self.force_args_if_virtual(op, ctx);
        OptimizationResult::PassOn
    }

    /// vstring.py:155-158 VStringPlainInfo.shrink
    ///
    /// ```text
    /// def shrink(self, length):
    ///     assert length >= 0
    ///     self.length = length
    ///     del self._chars[length:]
    /// ```
    ///
    fn vstring_plain_shrink(sinfo: &mut StrPtrInfo, length: usize) {
        sinfo.length = length as i32;
        if let VStringVariant::Plain(info) = &mut sinfo.variant {
            info._chars.truncate(length);
        }
    }

    /// vstring.py:839-851 opt_call_SHRINK_ARRAY
    ///
    /// ```text
    /// def opt_call_SHRINK_ARRAY(self, op):
    ///     i1 = getptrinfo(op.getarg(1))
    ///     i2 = self.getintbound(op.getarg(2))
    ///     # If the index is constant, if the argument is virtual (we only
    ///     # support VStringPlainValue for now) we can optimize away the call.
    ///     if (i2 and i2.is_constant() and i1 and i1.is_virtual() and
    ///         isinstance(i1, VStringPlainInfo)):
    ///         length = i2.get_constant_int()
    ///         i1.shrink(length)
    ///         self.last_emitted_operation = REMOVED
    ///         self.make_equal_to(op, op.getarg(1))
    ///         return True
    ///     return False
    /// ```
    fn opt_call_shrink_array(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if op.num_args() >= 3 {
            let arg1 = ctx.get_box_replacement(op.arg(1));
            let length = ctx.get_constant_int(op.arg(2));
            // vstring.py:844-845: i2.is_constant() && i1.is_virtual() &&
            // isinstance(i1, VStringPlainInfo)
            if let Some(length) = length {
                if let Some(PtrInfo::Str(sinfo)) = ctx.get_ptr_info_mut(arg1) {
                    if matches!(sinfo.variant, VStringVariant::Plain(_)) {
                        // vstring.py:847: i1.shrink(length)
                        Self::vstring_plain_shrink(sinfo, length as usize);
                        // vstring.py:849: self.make_equal_to(op, op.getarg(1))
                        ctx.replace_op(op.pos, arg1);
                        return OptimizationResult::Remove;
                    }
                }
            }
        }
        self.force_args_if_virtual(op, ctx);
        OptimizationResult::PassOn
    }
}

impl Default for OptString {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptString {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        match op.opcode {
            // vstring.py:440-444 optimize_NEWSTR / optimize_NEWUNICODE:
            // both dispatch to _optimize_NEWSTR(op, mode).
            OpCode::Newstr => self._optimize_newstr(op, mode_string, ctx),
            OpCode::Newunicode => self._optimize_newstr(op, mode_unicode, ctx),
            OpCode::Strsetitem => self.optimize_strsetitem(op, ctx),
            OpCode::Strgetitem => self.optimize_strgetitem(op, ctx),
            OpCode::Strlen => self.optimize_strlen(op, ctx),
            OpCode::Copystrcontent => self.optimize_copystrcontent(op, mode_string, ctx),

            // vstring.py: Unicode operations — same logic as string ops
            // but with unicode-specific opcodes.
            OpCode::Unicodesetitem => self.optimize_strsetitem(op, ctx),
            OpCode::Unicodegetitem => self.optimize_strgetitem(op, ctx),
            OpCode::Unicodelen => self.optimize_strlen(op, ctx),
            OpCode::Copyunicodecontent => self.optimize_copystrcontent(op, mode_unicode, ctx),

            // vstring.py: STRHASH/UNICODEHASH — force virtual string and emit.
            OpCode::Strhash | OpCode::Unicodehash => {
                let src = ctx.get_box_replacement(op.arg(0));
                self.force_if_virtual(src, ctx);
                OptimizationResult::PassOn
            }

            // vstring.py: optimize_GUARD_NO_EXCEPTION — if the last
            // emitted operation was removed (e.g. a string oopspec call
            // was virtualized), skip the guard.
            OpCode::GuardNoException => {
                // Delegate to default — the pure.rs pass handles this
                // via last_emitted_was_removed tracking.
                OptimizationResult::PassOn
            }

            // vstring.py: oopspec call handlers for string operations.
            // STR_CONCAT, STR_SLICE, STR_EQUAL are dispatched by OopSpecIndex
            // on CALL_* ops. For now, check if the call is a string oopspec.
            // vstring.py:621-627: optimize_CALL_R/F/N + optimize_CALL_PURE_*
            // are all aliased to optimize_CALL_I.
            OpCode::CallI
            | OpCode::CallR
            | OpCode::CallF
            | OpCode::CallN
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureF
            | OpCode::CallPureN => {
                if let Some(ref descr) = op.descr {
                    if let Some(cd) = descr.as_call_descr() {
                        let ei = cd.get_extra_info();
                        if ei.has_oopspec() {
                            return self.optimize_oopspec_call(op, &ei, ctx);
                        }
                    }
                }
                self.force_args_if_virtual(op, ctx);
                OptimizationResult::PassOn
            }

            _ => {
                // For any other op, force virtual strings that appear as arguments.
                self.force_args_if_virtual(op, ctx);
                OptimizationResult::PassOn
            }
        }
    }

    fn name(&self) -> &'static str {
        "string"
    }
}

#[cfg(test)]
mod tests {
    //! Upstream parity anchor: `rpython/jit/metainterp/test/test_string.py`
    //! for string-builder and copy-content behavior, plus
    //! `rpython/jit/metainterp/optimizeopt/vstring.py`.
    //!
    //! Tests that focus on `IntBound`-only constants, `lgtop` caching identity,
    //! or partial-pass behavior are original Rust regressions for helper paths
    //! that upstream usually exercises only through larger optimizer tests.

    use super::*;
    use crate::optimizeopt::info::{
        PtrInfo, StrPtrInfo, VStringConcatInfo, VStringPlainInfo, VStringSliceInfo, VStringVariant,
    };
    use crate::optimizeopt::optimizer::Optimizer;

    /// Assign sequential positions to ops and pre-seed constants in OptContext.
    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef::from_raw(i as u32);
        }
    }

    /// Run the OptString pass on a list of ops, with given pre-seeded constants.
    fn run_with_constants(ops: &[Op], constants: &[(OpRef, i64)]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptString::new()));

        // Seed constants into the context. Since Optimizer::optimize
        // creates its own context, we use a custom approach: run the pass
        // manually. Seed reserve_pos above any trace op.pos so that
        // force_virtual's synthesized ops don't collide with the original
        // trace positions — matches the invariant
        // `optimize_with_constants_and_inputs` maintains
        // (start_next_pos = max(num_inputs, max_pos + 1)).
        let max_pos = ops
            .iter()
            .map(|op| op.pos)
            .filter(|op| !op.is_none() && !op.is_constant())
            .map(|op| op.raw())
            .max()
            .unwrap_or(0);
        let start_next_pos = (max_pos + 1).max(ops.len() as u32);
        let mut ctx = OptContext::with_num_inputs_and_start_pos(ops.len(), 0, 0, start_next_pos);
        for &(opref, val) in constants {
            ctx.make_constant(opref, Value::Int(val));
        }

        let mut pass = OptString::new();
        pass.setup();

        for op in ops {
            // Resolve forwarded arguments.
            let mut resolved_op = op.clone();
            for arg in &mut resolved_op.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved_op, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::Remove => {
                    // Op removed, nothing emitted.
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved_op);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        pass.flush(&mut ctx);
        // Drain extra_operations_after (from emit_extra during force_box)
        // into new_operations so the test can see all emitted ops.
        while let Some((_pass_idx, extra_op)) = ctx.extra_operations_after.pop_front() {
            ctx.new_operations.push(extra_op);
        }
        ctx.new_operations
    }

    fn set_vstring_plain(ctx: &mut OptContext, opref: OpRef, chars: Vec<Option<OpRef>>) {
        let length = chars.len() as i32;
        ctx.set_ptr_info(
            opref,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length,
                variant: VStringVariant::Plain(VStringPlainInfo { _chars: chars }),
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );
    }

    fn set_vstring_concat(ctx: &mut OptContext, opref: OpRef, vleft: OpRef, vright: OpRef) {
        ctx.set_ptr_info(
            opref,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: -1,
                variant: VStringVariant::Concat(VStringConcatInfo {
                    vleft,
                    vright,
                    _is_virtual: true,
                }),
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );
    }

    fn set_vstring_slice(ctx: &mut OptContext, opref: OpRef, s: OpRef, start: OpRef, lgtop: OpRef) {
        ctx.set_ptr_info(
            opref,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: Some(lgtop), // vstring.py:223: self.lgtop = length
                mode: 0,
                length: -1,
                variant: VStringVariant::Slice(VStringSliceInfo { s, start, lgtop }),
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );
    }

    // ── Test 1: STRGETITEM on virtual string returns tracked character ──

    #[test]
    fn test_strgetitem_virtual_plain() {
        // Setup:
        //   i100 = const 3   (length)
        //   i101 = const 0   (index 0)
        //   i102 = const 1   (index 1)
        //   i200 = <some char value for index 0>
        //   i201 = <some char value for index 1>
        //
        // Trace:
        //   p0 = newstr(i100)         -> virtual, removed
        //   _  = strsetitem(p0, i101, i200)  -> stored in virtual, removed
        //   _  = strsetitem(p0, i102, i201)  -> stored in virtual, removed
        //   i3 = strgetitem(p0, i101) -> should resolve to i200, removed

        let mut ops = vec![
            Op::new(OpCode::Newstr, &[OpRef::from_raw(100)]), // op 0: p0 = newstr(3)
            Op::new(
                OpCode::Strsetitem,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(101),
                    OpRef::from_raw(200),
                ],
            ), // op 1
            Op::new(
                OpCode::Strsetitem,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(102),
                    OpRef::from_raw(201),
                ],
            ), // op 2
            Op::new(
                OpCode::Strgetitem,
                &[OpRef::from_raw(0), OpRef::from_raw(101)],
            ), // op 3: get char at 0
        ];
        assign_positions(&mut ops);

        let constants = vec![
            (OpRef::from_raw(100), 3), // length = 3
            (OpRef::from_raw(101), 0), // index 0
            (OpRef::from_raw(102), 1), // index 1
        ];

        let result = run_with_constants(&ops, &constants);

        // All ops should be removed (string is fully virtual).
        assert!(
            result.is_empty(),
            "Expected all ops removed, got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    // ── Test 2: STRLEN on virtual string returns constant ──

    #[test]
    fn test_strlen_virtual() {
        // p0 = newstr(5)
        // i1 = strlen(p0) -> should be constant 5
        let mut ops = vec![
            Op::new(OpCode::Newstr, &[OpRef::from_raw(100)]), // op 0
            Op::new(OpCode::Strlen, &[OpRef::from_raw(0)]),   // op 1
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef::from_raw(100), 5)];

        let result = run_with_constants(&ops, &constants);

        // Both ops removed: newstr virtualized, strlen resolved to constant.
        assert!(
            result.is_empty(),
            "Expected empty result, got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    // ── Test 3: Virtual string that escapes -> forced ──

    #[test]
    fn test_force_virtual_on_escape() {
        // p0 = newstr(2)
        // strsetitem(p0, 0, c0)
        // strsetitem(p0, 1, c1)
        // call_n(p0)     -> forces the string
        let mut ops = vec![
            Op::new(OpCode::Newstr, &[OpRef::from_raw(100)]), // op 0
            Op::new(
                OpCode::Strsetitem,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(101),
                    OpRef::from_raw(200),
                ],
            ), // op 1
            Op::new(
                OpCode::Strsetitem,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(102),
                    OpRef::from_raw(201),
                ],
            ), // op 2
            Op::new(OpCode::CallN, &[OpRef::from_raw(0)]),    // op 3: forces
        ];
        assign_positions(&mut ops);

        let constants = vec![
            (OpRef::from_raw(100), 2),
            (OpRef::from_raw(101), 0),
            (OpRef::from_raw(102), 1),
        ];

        let result = run_with_constants(&ops, &constants);

        // After forcing, we expect:
        // - SameAsI (constant 2 for length)
        // - Newstr
        // - SameAsI (constant 0), Strsetitem (char at 0)
        // - SameAsI (constant 1), Strsetitem (char at 1)
        // - CallN (with forwarded ref to the new Newstr)
        //
        // The exact count depends on how many constant-int SameAsI ops are emitted.
        // Key check: there should be a Newstr, Strsetitem ops, and the call.

        let newstr_count = result.iter().filter(|o| o.opcode == OpCode::Newstr).count();
        let setitem_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::Strsetitem)
            .count();
        let call_count = result.iter().filter(|o| o.opcode == OpCode::CallN).count();

        assert_eq!(newstr_count, 1, "Should have 1 Newstr after forcing");
        assert_eq!(setitem_count, 2, "Should have 2 Strsetitem after forcing");
        assert_eq!(call_count, 1, "Should have 1 CallN");
    }

    // ── Test 4: Concat virtual string length ──

    #[test]
    fn test_concat_length() {
        // Build two virtual strings, create a concat in PtrInfo::Str, then
        // query length.

        let mut pass = OptString::new();
        let mut ctx = OptContext::new(10);

        // Constant length refs
        ctx.make_constant(OpRef::from_raw(100), Value::Int(3));
        ctx.make_constant(OpRef::from_raw(101), Value::Int(4));

        // Virtual plain strings
        let left_ref = OpRef::from_raw(10);
        let right_ref = OpRef::from_raw(11);
        set_vstring_plain(&mut ctx, left_ref, vec![None; 3]);
        set_vstring_plain(&mut ctx, right_ref, vec![None; 4]);

        // Virtual concat
        let concat_ref = OpRef::from_raw(12);
        set_vstring_concat(&mut ctx, concat_ref, left_ref, right_ref);

        // Check total length = 3 + 4 = 7
        let total_len = pass.get_known_length(concat_ref, &ctx);
        assert_eq!(total_len, Some(7));
    }

    // ── Test 5: Slice virtual string ──

    #[test]
    fn test_slice_get_char() {
        // Build a virtual plain string, create a slice, get a character.
        let pass = OptString::new();
        let mut ctx = OptContext::new(10);

        // source = "abc" (chars at indices 0, 1, 2)
        let src_ref = OpRef::from_raw(10);
        set_vstring_plain(
            &mut ctx,
            src_ref,
            vec![
                Some(OpRef::from_raw(200)),
                Some(OpRef::from_raw(201)),
                Some(OpRef::from_raw(202)),
            ],
        );

        // slice = source[1:3] (start=1, length=2)
        ctx.make_constant(OpRef::from_raw(300), Value::Int(1)); // start
        ctx.make_constant(OpRef::from_raw(301), Value::Int(2)); // length
        let slice_ref = OpRef::from_raw(11);
        set_vstring_slice(
            &mut ctx,
            slice_ref,
            src_ref,
            OpRef::from_raw(300),
            OpRef::from_raw(301),
        );

        // Get char at index 0 of the slice -> should be source[1] = OpRef::from_raw(201)
        let ch = pass.strgetitem(slice_ref, 0, &mut ctx);
        assert_eq!(ch, Some(OpRef::from_raw(201)));

        // Get char at index 1 of the slice -> should be source[2] = OpRef::from_raw(202)
        let ch = pass.strgetitem(slice_ref, 1, &mut ctx);
        assert_eq!(ch, Some(OpRef::from_raw(202)));
    }

    #[test]
    fn test_slice_get_char_with_intbound_constant_start() {
        use crate::optimizeopt::intutils::IntBound;

        let mut pass = OptString::new();
        let mut ctx = OptContext::new(10);

        let src_ref = OpRef::from_raw(10);
        set_vstring_plain(
            &mut ctx,
            src_ref,
            vec![
                Some(OpRef::from_raw(200)),
                Some(OpRef::from_raw(201)),
                Some(OpRef::from_raw(202)),
            ],
        );

        // start is not a literal ConstInt box; it is only known via IntBound.
        let start_ref = OpRef::from_raw(300);
        ctx.with_intbound_mut(start_ref, |b| {
            *b = IntBound::from_constant(1);
        });
        ctx.make_constant(OpRef::from_raw(301), Value::Int(2)); // length

        let slice_ref = OpRef::from_raw(11);
        set_vstring_slice(
            &mut ctx,
            slice_ref,
            src_ref,
            start_ref,
            OpRef::from_raw(301),
        );

        assert_eq!(
            pass.strgetitem(slice_ref, 0, &mut ctx),
            Some(OpRef::from_raw(201))
        );
        assert_eq!(
            pass.strgetitem(slice_ref, 1, &mut ctx),
            Some(OpRef::from_raw(202))
        );
    }

    // ── Test 6: Slice length via STRLEN ──

    #[test]
    fn test_slice_strlen() {
        let mut pass = OptString::new();
        let mut ctx = OptContext::new(10);

        let src_ref = OpRef::from_raw(10);
        set_vstring_plain(&mut ctx, src_ref, vec![None; 5]);

        ctx.make_constant(OpRef::from_raw(300), Value::Int(1)); // start
        ctx.make_constant(OpRef::from_raw(301), Value::Int(3)); // length

        let slice_ref = OpRef::from_raw(11);
        set_vstring_slice(
            &mut ctx,
            slice_ref,
            src_ref,
            OpRef::from_raw(300),
            OpRef::from_raw(301),
        );

        let len = pass.get_known_length(slice_ref, &ctx);
        assert_eq!(len, Some(3));
    }

    #[test]
    fn test_getstrlen_uses_unicodelen_for_unicode() {
        let pass = OptString::new();
        let mut ctx = OptContext::new(10);
        let unicode_ref = OpRef::from_raw(7);
        // vstring.py:452 make_nonnull_str(op, mode_unicode) installs a
        // non-virtual StrPtrInfo with `mode = 1` so that later getstrlen
        // selects UNICODELEN instead of STRLEN.
        ctx.make_nonnull_str(unicode_ref, 1);

        let len_ref = pass.getstrlen(unicode_ref, &mut ctx);
        // getstrlen delegates to ctx.getstrlen_opref which emits via
        // emit_extra (downstream pipeline), so check extra_operations_after.
        let (_pass_idx, last_op) = ctx
            .extra_operations_after
            .back()
            .expect("getstrlen must emit a len op");

        assert_eq!(len_ref, last_op.pos);
        assert_eq!(last_op.opcode, OpCode::Unicodelen);
        assert_eq!(last_op.args.as_slice(), &[unicode_ref]);
    }

    // ── Test 7: Non-constant length NEWSTR passes through ──

    #[test]
    fn test_newstr_non_constant_passes_through() {
        // newstr(i0) where i0 is not a known constant -> should emit.
        let mut ops = vec![Op::new(OpCode::Newstr, &[OpRef::from_raw(50)])];
        assign_positions(&mut ops);

        let result = run_with_constants(&ops, &[]);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::Newstr);
    }

    // ── Test 8: Too-large NEWSTR passes through ──

    #[test]
    fn test_newstr_too_large_passes_through() {
        let mut ops = vec![Op::new(OpCode::Newstr, &[OpRef::from_raw(50)])];
        assign_positions(&mut ops);

        let constants = vec![(OpRef::from_raw(50), (MAX_CONST_LEN + 1) as i64)];
        let result = run_with_constants(&ops, &constants);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::Newstr);
    }

    // ── Test 9: STRGETITEM on non-virtual string passes through ──

    #[test]
    fn test_strgetitem_non_virtual() {
        let mut ops = vec![Op::new(
            OpCode::Strgetitem,
            &[OpRef::from_raw(50), OpRef::from_raw(51)],
        )];
        assign_positions(&mut ops);

        let constants = vec![(OpRef::from_raw(51), 0)];
        let result = run_with_constants(&ops, &constants);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::Strgetitem);
    }

    // ── Test 10: Force empty virtual string ──

    #[test]
    fn test_force_empty_virtual() {
        // p0 = newstr(0) -> virtual (0 chars)
        // call_n(p0)      -> force: emits newstr(0) only, no strsetitem
        let mut ops = vec![
            Op::new(OpCode::Newstr, &[OpRef::from_raw(100)]),
            Op::new(OpCode::CallN, &[OpRef::from_raw(0)]),
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef::from_raw(100), 0)];
        let result = run_with_constants(&ops, &constants);

        let newstr_count = result.iter().filter(|o| o.opcode == OpCode::Newstr).count();
        let setitem_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::Strsetitem)
            .count();
        assert_eq!(newstr_count, 1);
        assert_eq!(setitem_count, 0);
    }

    // ── Test 11: COPYSTRCONTENT into virtual string ──

    #[test]
    fn test_copystrcontent_virtual_to_virtual() {
        // src = newstr(2), strsetitem(src, 0, c0), strsetitem(src, 1, c1)
        // dst = newstr(2)
        // copystrcontent(src, dst, 0, 0, 2)
        // strgetitem(dst, 0) -> c0
        let mut ops = vec![
            Op::new(OpCode::Newstr, &[OpRef::from_raw(100)]), // op 0: src
            Op::new(
                OpCode::Strsetitem,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(101),
                    OpRef::from_raw(200),
                ],
            ), // op 1
            Op::new(
                OpCode::Strsetitem,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(102),
                    OpRef::from_raw(201),
                ],
            ), // op 2
            Op::new(OpCode::Newstr, &[OpRef::from_raw(100)]), // op 3: dst
            Op::new(
                OpCode::Copystrcontent,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(3),
                    OpRef::from_raw(101),
                    OpRef::from_raw(101),
                    OpRef::from_raw(100),
                ],
            ), // op 4: copy src->dst
            Op::new(
                OpCode::Strgetitem,
                &[OpRef::from_raw(3), OpRef::from_raw(101)],
            ), // op 5: get dst[0]
        ];
        assign_positions(&mut ops);

        let constants = vec![
            (OpRef::from_raw(100), 2),
            (OpRef::from_raw(101), 0),
            (OpRef::from_raw(102), 1),
        ];

        let result = run_with_constants(&ops, &constants);

        // All ops should be removed since everything is virtual.
        assert!(
            result.is_empty(),
            "Expected all ops removed, got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_copyunicodecontent_inline_uses_unicodegetitem() {
        let mut ops = vec![
            Op::new(OpCode::Newunicode, &[OpRef::from_raw(100)]),
            Op::new(OpCode::Newunicode, &[OpRef::from_raw(100)]),
            Op::new(
                OpCode::Copyunicodecontent,
                &[
                    OpRef::from_raw(0),
                    OpRef::from_raw(1),
                    OpRef::from_raw(101),
                    OpRef::from_raw(101),
                    OpRef::from_raw(100),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef::from_raw(100), 2), (OpRef::from_raw(101), 0)];

        let result = run_with_constants(&ops, &constants);

        let unicode_getitem_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::Unicodegetitem)
            .count();
        let str_getitem_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::Strgetitem)
            .count();

        assert_eq!(unicode_getitem_count, 2);
        assert_eq!(str_getitem_count, 0);
    }

    // ── Test 12: Multiple STRLEN calls on same virtual ──

    #[test]
    fn test_strlen_multiple_calls() {
        // p0 = newstr(3)
        // i1 = strlen(p0) -> const 3
        // i2 = strlen(p0) -> const 3
        let mut ops = vec![
            Op::new(OpCode::Newstr, &[OpRef::from_raw(100)]),
            Op::new(OpCode::Strlen, &[OpRef::from_raw(0)]),
            Op::new(OpCode::Strlen, &[OpRef::from_raw(0)]),
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef::from_raw(100), 3)];
        let result = run_with_constants(&ops, &constants);

        // All should be removed.
        assert!(result.is_empty());
    }

    // ── Test 13: STRGETITEM with uninitialized char falls through ──

    #[test]
    fn test_strgetitem_uninitialized_char() {
        // p0 = newstr(3), no strsetitem for index 0
        // strgetitem(p0, 0) -> char not set, must force and emit
        let mut ops = vec![
            Op::new(OpCode::Newstr, &[OpRef::from_raw(100)]),
            Op::new(
                OpCode::Strgetitem,
                &[OpRef::from_raw(0), OpRef::from_raw(101)],
            ),
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef::from_raw(100), 3), (OpRef::from_raw(101), 0)];

        let result = run_with_constants(&ops, &constants);

        // The string is forced, so we should see at least a Newstr + Strgetitem.
        let has_newstr = result.iter().any(|o| o.opcode == OpCode::Newstr);
        let has_getitem = result.iter().any(|o| o.opcode == OpCode::Strgetitem);
        assert!(has_newstr, "Should have forced Newstr");
        assert!(has_getitem, "Should have Strgetitem in output");
    }

    // ── Test 14: Concat virtual get_known_length with nested concat ──

    #[test]
    fn test_nested_concat_length() {
        let mut pass = OptString::new();
        let mut ctx = OptContext::new(10);

        ctx.make_constant(OpRef::from_raw(100), Value::Int(2));
        ctx.make_constant(OpRef::from_raw(101), Value::Int(3));
        ctx.make_constant(OpRef::from_raw(102), Value::Int(4));

        let a = OpRef::from_raw(10);
        let b = OpRef::from_raw(11);
        let c = OpRef::from_raw(12);
        set_vstring_plain(&mut ctx, a, vec![None; 2]);
        set_vstring_plain(&mut ctx, b, vec![None; 3]);
        set_vstring_plain(&mut ctx, c, vec![None; 4]);

        // ab = concat(a, b)
        let ab = OpRef::from_raw(20);
        set_vstring_concat(&mut ctx, ab, a, b);

        // abc = concat(ab, c)
        let abc = OpRef::from_raw(21);
        set_vstring_concat(&mut ctx, abc, ab, c);

        assert_eq!(pass.get_known_length(abc, &ctx), Some(9));
    }

    // ── Test 15: Concat get char across boundary ──

    #[test]
    fn test_concat_get_char() {
        let mut pass = OptString::new();
        let mut ctx = OptContext::new(10);

        ctx.make_constant(OpRef::from_raw(100), Value::Int(2));
        ctx.make_constant(OpRef::from_raw(101), Value::Int(2));

        let left = OpRef::from_raw(10);
        let right = OpRef::from_raw(11);
        set_vstring_plain(
            &mut ctx,
            left,
            vec![Some(OpRef::from_raw(200)), Some(OpRef::from_raw(201))],
        );
        set_vstring_plain(
            &mut ctx,
            right,
            vec![Some(OpRef::from_raw(202)), Some(OpRef::from_raw(203))],
        );

        let concat = OpRef::from_raw(12);
        set_vstring_concat(&mut ctx, concat, left, right);

        // Index 0 -> left[0] = 200
        assert_eq!(
            pass.strgetitem(concat, 0, &mut ctx),
            Some(OpRef::from_raw(200))
        );
        // Index 1 -> left[1] = 201
        assert_eq!(
            pass.strgetitem(concat, 1, &mut ctx),
            Some(OpRef::from_raw(201))
        );
        // Index 2 -> right[0] = 202
        assert_eq!(
            pass.strgetitem(concat, 2, &mut ctx),
            Some(OpRef::from_raw(202))
        );
        // Index 3 -> right[1] = 203
        assert_eq!(
            pass.strgetitem(concat, 3, &mut ctx),
            Some(OpRef::from_raw(203))
        );
    }

    #[test]
    fn test_strlen_caching_non_virtual() {
        // Original Rust smoke test: `OptString` alone does not eliminate the
        // second non-virtual `STRLEN`, but this still guards the local
        // `known_lengths` cache wiring from panicking or regressing.
        // STRLEN on a non-virtual string should be cached for the second call.
        let mut ops = vec![
            Op::new(OpCode::Strlen, &[OpRef::from_raw(100)]),
            Op::new(OpCode::Strlen, &[OpRef::from_raw(100)]),
            Op::new(OpCode::Finish, &[]),
        ];
        assign_positions(&mut ops);
        let result = run_with_constants(&ops, &[]);
        // Second STRLEN should be eliminated by heap.rs STRLEN caching
        // (if running through full pipeline) or by vstring.rs known_lengths.
        // With just OptString pass, the first STRLEN passes through and
        // records in known_lengths, but the second one checks known_lengths
        // which maps OpRef::from_raw(100) → OpRef::from_raw(0) (result of first STRLEN).
        // Since OpRef::from_raw(0) is not a constant, it won't be removed by OptString alone.
        // This test just verifies no crash occurs.
        assert!(result.len() >= 1);
    }

    #[test]
    fn test_concat_oopspec_creates_virtual() {
        // Verify that STR_CONCAT creates a virtual Concat.
        let mut pass = OptString::new();
        pass.setup();

        let left = OpRef::from_raw(100);

        // Simulate: NEWSTR(2) for left
        let mut left_op = Op::new(OpCode::Newstr, &[OpRef::from_raw(200)]);
        left_op.pos = left;
        let mut ctx = OptContext::new(10);
        ctx.make_constant(OpRef::from_raw(200), Value::Int(2));

        // Process NEWSTR → creates virtual Plain
        let _ = pass.propagate_forward(&left_op, &mut ctx);
        assert!(pass.is_virtual(left, &ctx));
    }

    // ── Box/state parity tests ──

    /// vstring.py:174: VStringPlainInfo.getstrlen caches lgtop.
    /// Second call must return the SAME OpRef (identity reuse).
    #[test]
    fn test_lgtop_reuse_plain() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(4, 0, 0, 50);
        let p0 = OpRef::from_raw(0);
        set_vstring_plain(
            &mut ctx,
            p0,
            vec![
                Some(OpRef::from_raw(10)),
                Some(OpRef::from_raw(11)),
                Some(OpRef::from_raw(12)),
            ],
        );

        let first = ctx.getstrlen_opref(p0, 0);
        let second = ctx.getstrlen_opref(p0, 0);
        // vstring.py:174: self.lgtop = ConstInt(len(self._chars))
        // Both calls must return the SAME cached OpRef.
        assert_eq!(
            first, second,
            "lgtop must be reused: first={:?}, second={:?}",
            first, second
        );
        // The cached value must equal the Plain length (3).
        assert_eq!(ctx.get_constant_int(first), Some(3));
    }

    /// vstring.py:117: StrPtrInfo.getstrlen caches STRLEN result in lgtop.
    /// After emitting STRLEN, the second call must return the cached OpRef.
    #[test]
    fn test_lgtop_reuse_nonvirtual() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(4, 0, 0, 50);
        let p0 = OpRef::from_raw(0);
        // Non-virtual Str with unknown length
        ctx.set_ptr_info(
            p0,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: -1,
                variant: VStringVariant::Ptr,
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );

        let first = ctx.getstrlen_opref(p0, 0);
        let second = ctx.getstrlen_opref(p0, 0);
        // vstring.py:117: self.lgtop = lengthop — cached STRLEN result
        assert_eq!(
            first, second,
            "STRLEN result must be cached in lgtop: first={:?}, second={:?}",
            first, second
        );
    }

    /// vstring.py:728: l1box.same_box(l2box) succeeds when both strings
    /// have the same cached lgtop. getstrlen_if_known must return the
    /// cached OpRef, not a freshly-created constant.
    #[test]
    fn test_same_box_identity() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(4, 0, 0, 50);
        let pass = OptString::new();

        let p0 = OpRef::from_raw(0);
        let p1 = OpRef::from_raw(1);
        // Two virtual strings of the same length (3 chars).
        set_vstring_plain(
            &mut ctx,
            p0,
            vec![
                Some(OpRef::from_raw(10)),
                Some(OpRef::from_raw(11)),
                Some(OpRef::from_raw(12)),
            ],
        );
        set_vstring_plain(
            &mut ctx,
            p1,
            vec![
                Some(OpRef::from_raw(20)),
                Some(OpRef::from_raw(21)),
                Some(OpRef::from_raw(22)),
            ],
        );

        // First call caches lgtop on each string.
        let l1 = pass.getstrlen_if_known(p0, &mut ctx);
        let l2 = pass.getstrlen_if_known(p1, &mut ctx);
        assert!(l1.is_some() && l2.is_some());

        // Second call must return the same cached OpRef.
        let l1_again = pass.getstrlen_if_known(p0, &mut ctx);
        let l2_again = pass.getstrlen_if_known(p1, &mut ctx);
        assert_eq!(l1, l1_again, "lgtop identity: p0 must return same OpRef");
        assert_eq!(l2, l2_again, "lgtop identity: p1 must return same OpRef");

        // Both have value 3, and RPython's same_box checks constant equality.
        assert_eq!(ctx.get_constant_int(l1.unwrap()), Some(3));
        assert_eq!(ctx.get_constant_int(l2.unwrap()), Some(3));
    }

    /// vstring.py:341-347: copy_str_content uses getintbound().is_constant()
    /// for the inline threshold check. Verify intbound-based constant
    /// detection enables the same inlining as literal constant detection.
    #[test]
    fn test_copy_str_content_intbound_inline() {
        use crate::optimizeopt::intutils::IntBound;

        let mut ctx = OptContext::with_num_inputs_and_start_pos(10, 0, 0, 50);

        // srcbox (p0): non-null string, not virtual
        let p0 = OpRef::from_raw(0);
        ctx.set_ptr_info(
            p0,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: -1,
                variant: VStringVariant::Ptr,
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );
        // targetbox: non-virtual
        let p1 = OpRef::from_raw(1);

        // lengthbox (i2): int with constant intbound = 2
        // Use an OpRef with IntBound set (not a literal constant)
        let i2 = OpRef::from_raw(2);
        ctx.with_intbound_mut(i2, |b| {
            *b = IntBound::from_constant(2);
        });

        // offsetbox and srcoffsetbox: constant 0
        let off = ctx.emit_constant_int(0);

        // Call copy_str_content. With intbound-constant length = 2 <= M=2,
        // it should inline to STRGETITEM+STRSETITEM instead of COPYSTRCONTENT.
        let _result = copy_str_content(&mut ctx, p0, p1, off, off, i2, 0, true);

        // emit_for_force routes to extra_operations_after; drain it.
        while let Some((_pass_idx, extra_op)) = ctx.extra_operations_after.pop_front() {
            ctx.new_operations.push(extra_op);
        }

        // Check that STRGETITEM ops were emitted (inline path) instead of
        // a single COPYSTRCONTENT (bulk path).
        let getitem_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::Strgetitem)
            .count();
        let copy_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::Copystrcontent)
            .count();
        assert!(
            getitem_count > 0 && copy_count == 0,
            "intbound-constant length should trigger inline path: \
             getitem={}, copy={}",
            getitem_count,
            copy_count,
        );
    }

    /// vstring.py:110-119 getstrlen_opref parity:
    /// getstrlen_opref(opref, mode) looks up info from opref and emits
    /// STRLEN(opref) on cache miss. Cached lgtop is returned on second call.
    #[test]
    fn test_getstrlen_opref_on_nonvirtual() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(10, 0, 0, 50);
        let arg2 = OpRef::from_raw(1);

        ctx.set_ptr_info(
            arg2,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: -1,
                variant: VStringVariant::Ptr,
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );

        let strlen_ref = ctx.getstrlen_opref(arg2, 0);

        let (_pass_idx, strlen_op) = ctx
            .extra_operations_after
            .back()
            .expect("should have emitted STRLEN");
        assert_eq!(strlen_op.opcode, OpCode::Strlen);
        assert_eq!(strlen_op.args.as_slice(), &[arg2]);
        assert_eq!(strlen_ref, strlen_op.pos);

        // Subsequent call must return the cached lgtop.
        let strlen_ref2 = ctx.getstrlen_opref(arg2, 0);
        assert_eq!(strlen_ref, strlen_ref2);
    }

    /// vstring.py:794 handle_str_equal_level2 parity:
    /// i2.getstrlen(arg1, self, mode) — info from arg2, fallback STRLEN(arg1).
    /// getstrlen_for(arg2, arg1, mode) must use arg2's info but emit
    /// STRLEN(arg1) when lgtop is not cached.
    #[test]
    fn test_getstrlen_for_level2_uses_arg1_as_fallback() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(10, 0, 0, 50);
        let arg1 = OpRef::from_raw(0);
        let arg2 = OpRef::from_raw(1);

        // Attach non-virtual StrPtrInfo to arg2 with lgtop=None.
        ctx.set_ptr_info(
            arg2,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: -1,
                variant: VStringVariant::Ptr,
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );

        // getstrlen_for(arg2, arg1, 0): info from arg2, STRLEN fallback on arg1.
        let strlen_ref = ctx.getstrlen_for(arg2, arg1, 0);

        let (_pass_idx, strlen_op) = ctx
            .extra_operations_after
            .back()
            .expect("should have emitted STRLEN");
        assert_eq!(strlen_op.opcode, OpCode::Strlen);
        assert_eq!(
            strlen_op.args.as_slice(),
            &[arg1],
            "STRLEN must use arg1 (op), not arg2 (info source)"
        );
        assert_eq!(strlen_ref, strlen_op.pos);

        // lgtop is cached on arg2's info — second call returns cached value.
        let strlen_ref2 = ctx.getstrlen_for(arg2, arg1, 0);
        assert_eq!(strlen_ref, strlen_ref2);
    }

    #[test]
    fn test_force_then_strlen_reuse() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(10, 0, 0, 50);

        let p0 = OpRef::from_raw(0);
        // Virtual Plain string with 3 chars.
        set_vstring_plain(
            &mut ctx,
            p0,
            vec![
                Some(OpRef::from_raw(10)),
                Some(OpRef::from_raw(11)),
                Some(OpRef::from_raw(12)),
            ],
        );

        // getstrlen_opref should cache lgtop = ConstInt(3).
        let len1 = ctx.getstrlen_opref(p0, 0);
        assert_eq!(ctx.get_constant_int(len1), Some(3));

        // Query again — must return the same cached OpRef.
        let len2 = ctx.getstrlen_opref(p0, 0);
        assert_eq!(len1, len2, "force-then-strlen: lgtop must be reused");
    }
}
