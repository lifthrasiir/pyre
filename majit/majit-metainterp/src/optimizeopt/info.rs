use crate::optimizeopt::intutils::IntBound;
/// Abstract information attached to operations during optimization.
///
/// Translated from rpython/jit/metainterp/optimizeopt/info.py.
/// Each operation can have associated analysis info (e.g., known integer bounds,
/// pointer info, virtual object state).
use majit_ir::{DescrRef, GcRef, Op, OpCode, OpRef, Value};

fn lookup_field_descr(field_descrs: &[DescrRef], field_idx: u32) -> Option<DescrRef> {
    field_descrs.get(field_idx as usize).cloned()
}

/// resoperation.py: AbstractResOpOrInputArg._forwarded
///
/// RPython uses a single `_forwarded` field per Box that holds EITHER:
/// - None (no forwarding, no info)
/// - another Box (forwarding to that box)
/// - a PtrInfo instance (terminal info)
/// - a Const value (terminal — RPython: _forwarded = constbox)
///
/// `get_box_replacement` follows Box→Box links, stops at None/PtrInfo/Const.
/// `getptrinfo` reads PtrInfo from the terminal Box.
#[derive(Clone, Debug)]
pub enum Forwarded {
    /// No forwarding or info set.
    None,
    /// Forwarding to another OpRef (RPython: _forwarded = other_box).
    Op(OpRef),
    /// Terminal info (RPython: _forwarded = PtrInfo instance).
    Info(PtrInfo),
    /// Terminal constant (RPython: _forwarded = constbox).
    /// optimizer.py:432: box.set_forwarded(constbox).
    Const(majit_ir::Value),
    /// Terminal IntBound (RPython: _forwarded = IntBound instance).
    /// intutils.py:73: IntBound(AbstractInfo) — stored directly in
    /// _forwarded, retrieved by optimizer.py:99 getintbound().
    IntBound(crate::optimizeopt::intutils::IntBound),
}

impl Default for Forwarded {
    fn default() -> Self {
        Forwarded::None
    }
}

/// shortpreamble.py:11-49: PreambleOp
///
/// Wrapper stored in PtrInfo._fields during Phase 2 import.
/// When `_getfield` (heap.py:177-187) encounters this in a field slot,
/// it calls `force_op_from_preamble()` to lazily resolve the value
/// via the short preamble builder.
///
/// RPython stores PreambleOp directly in `_fields[]` (Python's dynamic
/// typing). Rust mirrors this with the `FieldEntry` enum stored in the
/// same `fields` / `items` vectors.
#[derive(Clone, Debug)]
pub struct PreambleOp {
    /// Phase 1 result box — RPython: PreambleOp.op (aka HeapOp.res)
    pub op: OpRef,
    /// Fresh Phase 2 OpRef with distinct identity from label_args.
    /// Returned by force_op_from_preamble as the resolved field value.
    /// (Rust equivalent of RPython's Phase 1 Box identity isolation.)
    pub resolved: OpRef,
    /// RPython: PreambleOp.invented_name
    pub invented_name: bool,
    /// RPython: PreambleOp.preamble_op — the actual replay operation
    /// for the short preamble. Always present (RPython parity).
    pub preamble_op: majit_ir::Op,
}

/// RPython _fields[] element — either a concrete value or a PreambleOp sentinel.
///
/// info.py:203 `setfield` stores either a normal Box or a PreambleOp into
/// `_fields[]`. heap.py:177 `_getfield` checks `isinstance(res, PreambleOp)`
/// to decide whether to force the value via the short preamble.
///
/// Rust equivalent: typed enum instead of Python's duck-typed list.
#[derive(Clone, Debug)]
pub enum FieldEntry {
    /// Normal cached field value (info.py:203 setfield).
    Value(OpRef),
    /// shortpreamble.py:11 PreambleOp — sentinel stored during Phase 2 import.
    Preamble(PreambleOp),
}

impl FieldEntry {
    /// Extract the concrete OpRef if this is a `Value` entry.
    /// Returns `None` for `Preamble` entries (those need special handling
    /// via `force_op_from_preamble`).
    pub fn as_opref(&self) -> Option<OpRef> {
        match self {
            FieldEntry::Value(opref) => Some(*opref),
            FieldEntry::Preamble(_) => None,
        }
    }

    /// Returns true if this is a `Preamble` entry.
    pub fn is_preamble(&self) -> bool {
        matches!(self, FieldEntry::Preamble(_))
    }

    /// Extract the `PreambleOp` if this is a `Preamble` entry.
    pub fn as_preamble(&self) -> Option<&PreambleOp> {
        match self {
            FieldEntry::Preamble(pop) => Some(pop),
            FieldEntry::Value(_) => None,
        }
    }

    /// View this slot the same way RPython reads `_fields[]` / `_items[]`
    /// in non-forcing paths such as `serialize_optheap`,
    /// `produce_short_preamble_ops`, and `_expand_infos_from_virtual`.
    ///
    /// Normal values return the stored OpRef. `PreambleOp` entries expose
    /// their original Phase 1 source box (`pop.op`), matching PyPy's
    /// `get_box_replacement(PreambleOp(...))` behavior.
    pub fn as_seen_opref(&self) -> OpRef {
        match self {
            FieldEntry::Value(opref) => *opref,
            FieldEntry::Preamble(pop) => pop.op,
        }
    }

    /// Consume and extract the `PreambleOp` if this is a `Preamble` entry.
    pub fn into_preamble(self) -> Option<PreambleOp> {
        match self {
            FieldEntry::Preamble(pop) => Some(pop),
            FieldEntry::Value(_) => None,
        }
    }
}

/// Information about an operation's result, attached during optimization.
///
/// info.py: AbstractInfo hierarchy — the base class for all optimization info.
#[derive(Clone, Debug)]
pub enum OpInfo {
    /// No information known.
    Unknown,
    /// Known constant value (integer or pointer).
    Constant(Value),
    /// Known integer bounds.
    IntBound(IntBound),
    /// Pointer info (non-null, known class, virtual, etc.).
    Ptr(PtrInfo),
    /// Known constant float value.
    /// info.py: FloatConstInfo — tracks float constants separately
    /// because they need special boxing on 32-bit platforms.
    FloatConst(f64),
}

impl OpInfo {
    pub fn is_constant(&self) -> bool {
        matches!(
            self,
            OpInfo::Constant(_) | OpInfo::FloatConst(_) | OpInfo::Ptr(PtrInfo::Constant(_))
        )
    }

    pub fn get_constant(&self) -> Option<&Value> {
        match self {
            OpInfo::Constant(v) => Some(v),
            _ => None,
        }
    }

    /// Get the constant float value if this is a FloatConst.
    pub fn get_constant_float(&self) -> Option<f64> {
        match self {
            OpInfo::FloatConst(f) => Some(*f),
            OpInfo::Constant(Value::Float(f)) => Some(*f),
            _ => None,
        }
    }

    pub fn get_int_bound(&self) -> Option<&IntBound> {
        match self {
            OpInfo::IntBound(b) => Some(b),
            _ => None,
        }
    }

    /// Whether this info is known non-null.
    /// info.py: is_nonnull()
    pub fn is_nonnull(&self) -> bool {
        match self {
            OpInfo::Ptr(ptr) => ptr.is_nonnull(),
            OpInfo::Constant(Value::Int(v)) => *v != 0,
            _ => false,
        }
    }

    /// Whether this info represents a virtual (allocation-removed) object.
    /// info.py: is_virtual()
    pub fn is_virtual(&self) -> bool {
        matches!(self, OpInfo::Ptr(ptr) if ptr.is_virtual())
    }

    /// Get the PtrInfo if present.
    pub fn get_ptr_info(&self) -> Option<&PtrInfo> {
        match self {
            OpInfo::Ptr(p) => Some(p),
            _ => None,
        }
    }
}

/// Information about a pointer value.
///
/// info.py: PtrInfo hierarchy:
///   NonNullPtrInfo → AbstractVirtualPtrInfo → {InstancePtrInfo, StructPtrInfo,
///   ArrayPtrInfo, ArrayStructInfo, RawBufferPtrInfo, RawStructPtrInfo, RawSlicePtrInfo}
///   ConstPtrInfo
#[derive(Clone, Debug)]
pub enum PtrInfo {
    /// Known to be non-null, nothing else.
    /// info.py: NonNullPtrInfo
    NonNull {
        /// info.py:91-92: NonNullPtrInfo.last_guard_pos = -1
        last_guard_pos: i32,
    },
    /// Known constant pointer.
    /// info.py: ConstPtrInfo (does NOT inherit NonNullPtrInfo)
    Constant(GcRef),
    /// Non-virtual GC object with cached field info.
    /// info.py: InstancePtrInfo (is_virtual = False).
    /// `make_constant_class` results — class set, no descr, no fields —
    /// are also stored here as `Instance(descr=None, known_class=Some(...))`,
    /// matching PyPy's `info.InstancePtrInfo(None, class_const)` factory
    /// at optimizer.py:147.
    Instance(InstancePtrInfo),
    /// Non-virtual GC struct with cached field info.
    /// info.py: StructPtrInfo (is_virtual = False)
    Struct(StructPtrInfo),
    /// Non-virtual GC array with cached item info and lenbound.
    /// info.py: ArrayPtrInfo (is_virtual = False)
    Array(ArrayPtrInfo),
    /// Virtual object (allocation removed by the optimizer).
    /// info.py: InstancePtrInfo
    Virtual(VirtualInfo),
    /// Virtual array.
    /// info.py: ArrayPtrInfo
    VirtualArray(VirtualArrayInfo),
    /// Virtual struct (no vtable).
    /// info.py: StructPtrInfo
    VirtualStruct(VirtualStructInfo),
    /// Virtual array of structs (interior field access).
    /// info.py: ArrayStructInfo
    VirtualArrayStruct(VirtualArrayStructInfo),
    /// Virtual raw buffer.
    /// info.py: RawBufferPtrInfo
    VirtualRawBuffer(VirtualRawBufferInfo),
    /// Virtual raw slice (offset alias into a parent raw buffer).
    /// info.py: RawSlicePtrInfo
    VirtualRawSlice(VirtualRawSliceInfo),
    /// Virtualizable object (interpreter frame).
    Virtualizable(VirtualizableFieldState),
    /// vstring.py:50: StrPtrInfo — string with known length bounds.
    /// Tracks lenbound (IntBound) and mode (string vs unicode).
    Str(StrPtrInfo),
}

/// vstring.py:50-140: StrPtrInfo
#[derive(Clone, Debug)]
pub struct StrPtrInfo {
    /// vstring.py: self.lenbound — IntBound for string length.
    pub lenbound: Option<IntBound>,
    /// vstring.py:53 self.lgtop — cached length OpRef (set by getstrlen).
    /// After force_box, this preserves the computed length so subsequent
    /// STRLEN queries reuse it instead of emitting a new STRLEN op.
    pub lgtop: Option<OpRef>,
    /// vstring.py: self.mode — 0 = mode_string, 1 = mode_unicode.
    pub mode: u8,
    /// vstring.py: self.length — known exact length (-1 if unknown).
    pub length: i32,
    /// vstring.py: subclass-specific state
    /// (`VStringPlainInfo` / `VStringSliceInfo` / `VStringConcatInfo`).
    pub variant: VStringVariant,
    /// info.py:91-92: last_guard_pos
    pub last_guard_pos: i32,
    /// info.py:124-128 `AbstractVirtualPtrInfo._cached_vinfo` — inherited
    /// through `StrPtrInfo(AbstractVirtualPtrInfo)` (vstring.py:50,55).
    /// Same semantics as the sibling Virtual/VirtualArray/... variants:
    /// `make_virtual_info` dedups across finish() calls by comparing
    /// fieldnums (resume.py:309-314).
    pub cached_vinfo: std::cell::RefCell<Option<std::rc::Rc<majit_ir::RdVirtualInfo>>>,
}

/// Runtime hook for `ConstPtrInfo.getstrlen1(mode)` (info.py:810-822).
/// Returns `Some(length)` when `gcref` points at a known string of the
/// requested mode, `None` otherwise. Cloned (Arc) into each
/// `EnsuredPtrInfo` instance so the helper can satisfy
/// `getlenbound(Some(mode))` for constant string args without re-borrowing
/// `OptContext`.
pub type StringLengthResolver = std::sync::Arc<dyn Fn(GcRef, u8) -> Option<i64> + Send + Sync>;

/// info.py:788-790 `ConstPtrInfo._unpack_str(mode)` — runtime hook for
/// extracting characters from a constant string GcRef. Returns the char
/// values as `Vec<i64>`. Set by the host runtime (pyre etc.).
pub type StringContentResolver =
    std::sync::Arc<dyn Fn(GcRef, u8) -> Option<Vec<i64>> + Send + Sync>;

/// history.py:377-387 `get_const_ptr_for_string(s)` — runtime hook for
/// creating a constant string GcRef from char values. The bool indicates
/// unicode (true) vs byte-string (false). Set by the host runtime.
pub type StringConstantAllocator = std::sync::Arc<dyn Fn(&[i64], bool) -> GcRef + Send + Sync>;

/// Result of `OptContext::ensure_ptr_info_arg0(op)` — direct line-by-line
/// equivalent of PyPy's `ensure_ptr_info_arg0` return value
/// (`optimizer.py:461-499`).
///
/// PyPy returns a Python `PtrInfo` object that the caller invokes methods on
/// (`structinfo.setfield(...)`, `arrayinfo.getlenbound(None).make_gt_const(...)`).
/// The Rust port can't expose `&mut PtrInfo` directly when the arg0 is a
/// constant — there's no `Forwarded::Info` slot to borrow from — so the enum
/// distinguishes the two cases:
///
/// - **`Constant { gcref, .. }`** — `arg0.is_constant()`
///   (`optimizer.py:464-466`). PyPy returns a freshly-constructed
///   `info.ConstPtrInfo(arg0)`. The Rust variant carries the resolved
///   `GcRef` so methods like `getlenbound` can synthesize the same answer
///   on demand. The optional `string_length_resolver` Arc allows
///   `getlenbound(Some(mode))` to return an exact constant length when the
///   runtime can read the underlying string object — matching PyPy's
///   `getstrlen1(mode)` path through `_unpack_str(mode)`.
///
/// - **`Forwarded(&mut PtrInfo)`** — `arg0.get_forwarded()` returns either an
///   existing `AbstractVirtualPtrInfo` subclass (early-return path) or a
///   freshly-installed Instance/Struct/Array/Str etc. (`optimizer.py:475-498`).
///   The mutable reference is backed by `OptContext::forwarded[idx]` so
///   `info.setfield()` / `info.setitem()` mutate the canonical PtrInfo
///   in-place — matching PyPy's `arg0.set_forwarded(opinfo)` followed by
///   `opinfo.setfield(...)`.
pub enum EnsuredPtrInfo<'a> {
    /// `info.ConstPtrInfo(arg0)` — synthesized from a constant Ref / raw-pointer
    /// Int OpRef. Read-only by construction.
    Constant {
        gcref: GcRef,
        /// Optional runtime hook for `getstrlen1(mode)` lookups.
        string_length_resolver: Option<StringLengthResolver>,
    },
    /// `arg0.get_forwarded()` — direct mutable handle into the
    /// `OptContext::forwarded` slot.
    Forwarded(&'a mut PtrInfo),
}

impl<'a> EnsuredPtrInfo<'a> {
    /// `info.py PtrInfo.getlenbound(mode)` — direct delegation to the underlying
    /// PtrInfo. For `Constant` the call routes through the optional
    /// `string_length_resolver` so an exact constant length can be returned
    /// when the runtime knows it (PyPy `ConstPtrInfo.getlenbound` →
    /// `getstrlen1(mode)` → `_unpack_str(mode)` at info.py:796-822).
    pub fn getlenbound(&mut self, mode: Option<u8>) -> Option<IntBound> {
        match self {
            EnsuredPtrInfo::Constant {
                gcref,
                string_length_resolver,
            } => {
                // info.py:796-802 ConstPtrInfo.getlenbound(mode):
                //
                //     def getlenbound(self, mode):
                //         length = self.getstrlen1(mode)
                //         if length < 0:
                //             return IntBound.nonnegative()
                //         return IntBound.from_constant(length)
                //
                // info.py:810-824 ConstPtrInfo.getstrlen1(mode):
                //
                //     def getstrlen1(self, mode):
                //         if mode is vstring.mode_string:    ...
                //         elif mode is vstring.mode_unicode: ...
                //         else:
                //             return -1
                //
                // PyPy returns `IntBound.nonnegative()` regardless of
                // mode whenever `getstrlen1` cannot supply an exact
                // length. The Rust port mirrors that:
                //   * mode == None        → getstrlen1 returns -1 →
                //                           nonnegative()
                //   * mode == Some(0|1)   → resolver returns Some(len) →
                //                           from_constant(len);
                //                           else nonnegative()
                let length = match mode {
                    Some(mode_value) => {
                        if gcref.is_null() {
                            -1
                        } else if let Some(resolver) = string_length_resolver.as_deref() {
                            resolver(*gcref, mode_value).unwrap_or(-1)
                        } else {
                            -1
                        }
                    }
                    // info.py:823-824 `else: return -1` for mode == None.
                    None => -1,
                };
                if length < 0 {
                    Some(IntBound::nonnegative())
                } else {
                    Some(IntBound::from_constant(length))
                }
            }
            EnsuredPtrInfo::Forwarded(info) => info.getlenbound(mode),
        }
    }

    /// Mutable access to the underlying `PtrInfo`. Returns `None` for the
    /// `Constant` variant — PyPy's `ConstPtrInfo.setfield/setitem` route
    /// through `optheap.const_infos`, not through the constant box's own
    /// info slot (info.py:738-752).
    pub fn as_mut(&mut self) -> Option<&mut PtrInfo> {
        match self {
            EnsuredPtrInfo::Constant { .. } => None,
            EnsuredPtrInfo::Forwarded(info) => Some(info),
        }
    }

    /// Whether the helper produced a synthesized `ConstPtrInfo` rather than a
    /// real forwarded entry. Mirrors `isinstance(opinfo, ConstPtrInfo)` at
    /// the call site.
    pub fn is_constant(&self) -> bool {
        matches!(self, EnsuredPtrInfo::Constant { .. })
    }
}

/// vstring.py:142-334 subclass state carried by `StrPtrInfo`.
#[derive(Clone, Debug)]
pub enum VStringVariant {
    /// Non-virtual base `StrPtrInfo`.
    Ptr,
    /// vstring.py:142 `VStringPlainInfo`.
    Plain(VStringPlainInfo),
    /// vstring.py:214 `VStringSliceInfo`.
    Slice(VStringSliceInfo),
    /// vstring.py:266 `VStringConcatInfo`.
    Concat(VStringConcatInfo),
}

/// vstring.py:142-212 `VStringPlainInfo`
#[derive(Clone, Debug)]
pub struct VStringPlainInfo {
    pub _chars: Vec<Option<OpRef>>,
}

/// vstring.py:214-264 `VStringSliceInfo`
#[derive(Clone, Debug)]
pub struct VStringSliceInfo {
    pub s: OpRef,
    pub start: OpRef,
    pub lgtop: OpRef,
}

/// vstring.py:266-334 `VStringConcatInfo`
#[derive(Clone, Debug)]
pub struct VStringConcatInfo {
    pub vleft: OpRef,
    pub vright: OpRef,
    pub _is_virtual: bool,
}

impl StrPtrInfo {
    /// vstring.py:168 / 227 / 278 `is_virtual()` on the string ptrinfo classes.
    pub fn is_virtual(&self) -> bool {
        match &self.variant {
            VStringVariant::Ptr => false,
            VStringVariant::Plain(_) | VStringVariant::Slice(_) => true,
            VStringVariant::Concat(info) => info._is_virtual,
        }
    }

    /// vstring.py:110/171/251/281 `getstrlen()` on the string ptrinfo classes.
    ///
    /// Returns the structurally-known constant length for virtual variants
    /// (Plain/Slice/Concat). For the non-virtual `Ptr` variant, returns
    /// `None` — RPython's base StrPtrInfo.getstrlen() (vstring.py:110-119)
    /// always emits STRLEN and attaches lenbound as metadata; it never
    /// extracts a constant from lenbound directly.
    pub fn getstrlen(&self, ctx: &crate::optimizeopt::OptContext, mode: u8) -> Option<i64> {
        // vstring.py:112: if self.lgtop is not None: return self.lgtop
        if let Some(lgtop) = self.lgtop {
            return ctx.get_constant_int(lgtop).or_else(|| {
                ctx.get_int_bound(lgtop)
                    .filter(|b| b.is_constant())
                    .map(|b| b.get_constant_int())
            });
        }
        match &self.variant {
            // vstring.py:110-119: base StrPtrInfo.getstrlen always emits
            // STRLEN and caches in lgtop; never returns a constant from
            // lenbound. The caller (getstrlen_opref) handles STRLEN emission.
            VStringVariant::Ptr => None,
            // vstring.py:171-175: VStringPlainInfo.getstrlen
            VStringVariant::Plain(info) => Some(info._chars.len() as i64),
            // vstring.py:251-253: VStringSliceInfo.getstrlen → self.lgtop
            VStringVariant::Slice(info) => ctx.get_constant_int_or_bound(info.lgtop),
            // vstring.py:281-295: VStringConcatInfo.getstrlen
            VStringVariant::Concat(info) => {
                let left = ctx.getptrinfo(info.vleft)?;
                let right = ctx.getptrinfo(info.vright)?;
                let len1 = left.as_ref().get_known_str_length(ctx, mode)?;
                let len2 = right.as_ref().get_known_str_length(ctx, mode)?;
                Some(len1 + len2)
            }
        }
    }

    /// vstring.py:161 / 172 / 298 `get_constant_string_spec()` on the string
    /// ptrinfo classes.
    ///
    /// The upstream method returns either a low-level string or unicode object.
    /// majit keeps the same recursive shape but represents the constant string
    /// as character/codepoint integers until a runtime string allocator is
    /// wired in.
    pub fn get_constant_string_spec(
        &self,
        ctx: &crate::optimizeopt::OptContext,
        mode: u8,
    ) -> Option<Vec<i64>> {
        let _ = mode;
        match &self.variant {
            VStringVariant::Ptr => None,
            VStringVariant::Plain(info) => {
                let mut chars = Vec::with_capacity(info._chars.len());
                for ch in &info._chars {
                    let ch = ctx.get_box_replacement((*ch)?);
                    chars.push(ctx.get_constant_int(ch)?);
                }
                Some(chars)
            }
            VStringVariant::Slice(info) => {
                // vstring.py:236-248: use getintbound().is_constant()
                let source = ctx.getptrinfo(info.s)?;
                let source_chars = source.as_ref().get_constant_string_spec(ctx, mode)?;
                let start = usize::try_from(ctx.get_constant_int_or_bound(info.start)?).ok()?;
                let length = usize::try_from(ctx.get_constant_int_or_bound(info.lgtop)?).ok()?;
                let stop = start.checked_add(length)?;
                if stop > source_chars.len() {
                    return None;
                }
                Some(source_chars[start..stop].to_vec())
            }
            VStringVariant::Concat(info) => {
                let left = ctx.getptrinfo(info.vleft)?;
                let right = ctx.getptrinfo(info.vright)?;
                let mut chars = left.as_ref().get_constant_string_spec(ctx, mode)?;
                chars.extend(right.as_ref().get_constant_string_spec(ctx, mode)?);
                Some(chars)
            }
        }
    }

    /// vstring.py:158 / 172 / 230 `strgetitem()` shape, collapsed into a single
    /// variant-dispatch method on the Rust side.
    pub fn strgetitem(&self, index: i64, ctx: &crate::optimizeopt::OptContext) -> Option<OpRef> {
        let index = usize::try_from(index).ok()?;
        match &self.variant {
            VStringVariant::Ptr => None,
            VStringVariant::Plain(info) => info._chars.get(index).copied().flatten(),
            VStringVariant::Slice(info) => {
                // vstring.py:491: index = _int_add(sinfo.start, index)
                // Accept intbound-constant starts, not just literal constants.
                let start = ctx.get_constant_int_or_bound(info.start)?;
                let source = ctx.getptrinfo(info.s)?;
                source.as_ref().strgetitem(index as i64 + start, ctx)
            }
            VStringVariant::Concat(info) => {
                let left = ctx.getptrinfo(info.vleft)?;
                let left_len =
                    usize::try_from(left.as_ref().get_known_str_length(ctx, self.mode)?).ok()?;
                if index < left_len {
                    left.as_ref().strgetitem(index as i64, ctx)
                } else {
                    let right = ctx.getptrinfo(info.vright)?;
                    right.as_ref().strgetitem((index - left_len) as i64, ctx)
                }
            }
        }
    }
}

impl PtrInfo {
    // ── Constructors (info.py: factory methods) ──

    /// Create a NonNull PtrInfo.
    pub fn nonnull() -> Self {
        PtrInfo::NonNull { last_guard_pos: -1 }
    }

    // ── info.py:100-118: last_guard_pos methods ──

    /// info.py:100-103: get_last_guard
    pub fn get_last_guard_pos(&self) -> Option<usize> {
        let pos = match self {
            PtrInfo::NonNull { last_guard_pos, .. } => *last_guard_pos,
            PtrInfo::Instance(i) => i.last_guard_pos,
            PtrInfo::Struct(s) => s.last_guard_pos,
            PtrInfo::Array(a) => a.last_guard_pos,
            PtrInfo::Virtual(v) => v.last_guard_pos,
            PtrInfo::VirtualArray(v) => v.last_guard_pos,
            PtrInfo::VirtualStruct(v) => v.last_guard_pos,
            PtrInfo::VirtualArrayStruct(v) => v.last_guard_pos,
            PtrInfo::VirtualRawBuffer(v) => v.last_guard_pos,
            PtrInfo::VirtualRawSlice(v) => v.last_guard_pos,
            PtrInfo::Virtualizable(v) => v.last_guard_pos,
            PtrInfo::Str(s) => s.last_guard_pos,
            PtrInfo::Constant(_) => return None, // ConstPtrInfo has no last_guard_pos
        };
        if pos < 0 { None } else { Some(pos as usize) }
    }

    /// Raw last_guard_pos value as i32 (-1 if none).
    pub fn last_guard_pos(&self) -> Option<i32> {
        let pos = match self {
            PtrInfo::NonNull { last_guard_pos, .. } => *last_guard_pos,
            PtrInfo::Instance(i) => i.last_guard_pos,
            PtrInfo::Struct(s) => s.last_guard_pos,
            PtrInfo::Array(a) => a.last_guard_pos,
            PtrInfo::Virtual(v) => v.last_guard_pos,
            PtrInfo::VirtualArray(v) => v.last_guard_pos,
            PtrInfo::VirtualStruct(v) => v.last_guard_pos,
            PtrInfo::VirtualArrayStruct(v) => v.last_guard_pos,
            PtrInfo::VirtualRawBuffer(v) => v.last_guard_pos,
            PtrInfo::VirtualRawSlice(v) => v.last_guard_pos,
            PtrInfo::Virtualizable(v) => v.last_guard_pos,
            PtrInfo::Str(s) => s.last_guard_pos,
            PtrInfo::Constant(_) => return None,
        };
        Some(pos)
    }

    /// info.py:111-118: mark_last_guard
    pub fn set_last_guard_pos(&mut self, pos: i32) {
        match self {
            PtrInfo::NonNull { last_guard_pos, .. } => *last_guard_pos = pos,
            PtrInfo::Instance(i) => i.last_guard_pos = pos,
            PtrInfo::Struct(s) => s.last_guard_pos = pos,
            PtrInfo::Array(a) => a.last_guard_pos = pos,
            PtrInfo::Virtual(v) => v.last_guard_pos = pos,
            PtrInfo::VirtualArray(v) => v.last_guard_pos = pos,
            PtrInfo::VirtualStruct(v) => v.last_guard_pos = pos,
            PtrInfo::VirtualArrayStruct(v) => v.last_guard_pos = pos,
            PtrInfo::VirtualRawBuffer(v) => v.last_guard_pos = pos,
            PtrInfo::VirtualRawSlice(v) => v.last_guard_pos = pos,
            PtrInfo::Virtualizable(v) => v.last_guard_pos = pos,
            PtrInfo::Str(s) => s.last_guard_pos = pos,
            PtrInfo::Constant(_) => {} // ConstPtrInfo: no-op
        }
    }

    /// info.py:108-109: reset_last_guard_pos
    pub fn reset_last_guard_pos(&mut self) {
        self.set_last_guard_pos(-1);
    }

    /// Create a Constant PtrInfo.
    pub fn constant(gcref: GcRef) -> Self {
        PtrInfo::Constant(gcref)
    }

    /// `optimizer.py:137-152 make_constant_class` parity:
    ///
    /// ```python
    /// def make_constant_class(self, op, class_const, ...):
    ///     ...
    ///     opinfo = info.InstancePtrInfo(None, class_const)
    ///     opinfo.last_guard_pos = last_guard_pos
    ///     op.set_forwarded(opinfo)
    /// ```
    ///
    /// PyPy stores known-class state on `InstancePtrInfo` itself (with
    /// `descr=None` and an empty `_fields`). The Rust port mirrors that
    /// directly so there is no separate "class only" enum variant — every
    /// `make_constant_class` result is an `Instance` that subsequent
    /// `setfield`/`setitem` calls extend with field caches just like
    /// PyPy's lazy `init_fields`.
    ///
    /// `is_nonnull` is accepted for source-compatibility with the prior
    /// constructor signature; PyPy `InstancePtrInfo` always inherits
    /// `NonNullPtrInfo.is_nonnull() == True`, so the parameter is unused
    /// at the storage level.
    pub fn known_class(class_ptr: GcRef, _is_nonnull: bool) -> Self {
        PtrInfo::Instance(InstancePtrInfo {
            descr: None,
            known_class: Some(class_ptr),
            fields: Vec::new(),
            field_descrs: Vec::new(),
            last_guard_pos: -1,
        })
    }

    /// Create a non-virtual InstancePtrInfo.
    pub fn instance(descr: Option<DescrRef>, known_class: Option<GcRef>) -> Self {
        PtrInfo::Instance(InstancePtrInfo {
            descr,
            known_class,
            fields: Vec::new(),
            field_descrs: Vec::new(),
            last_guard_pos: -1,
        })
    }

    /// Create a non-virtual StructPtrInfo.
    pub fn struct_ptr(descr: DescrRef) -> Self {
        PtrInfo::Struct(StructPtrInfo {
            descr,
            fields: Vec::new(),
            field_descrs: Vec::new(),
            last_guard_pos: -1,
        })
    }

    /// Create a non-virtual ArrayPtrInfo.
    pub fn array(descr: DescrRef, lenbound: IntBound) -> Self {
        PtrInfo::Array(ArrayPtrInfo {
            descr,
            lenbound,
            items: Vec::new(),
            last_guard_pos: -1,
        })
    }

    /// Create a Virtual PtrInfo (allocation removed).
    pub fn virtual_obj(descr: DescrRef, known_class: Option<GcRef>) -> Self {
        PtrInfo::Virtual(VirtualInfo {
            descr,
            known_class,
            ob_type_descr: None,
            fields: Vec::new(),
            field_descrs: Vec::new(),
            last_guard_pos: -1,
            cached_vinfo: std::cell::RefCell::new(None),
        })
    }

    /// Create a VirtualArray PtrInfo.
    pub fn virtual_array(descr: DescrRef, length: usize, clear: bool) -> Self {
        PtrInfo::VirtualArray(VirtualArrayInfo {
            descr,
            clear,
            items: vec![OpRef::NONE; length],
            last_guard_pos: -1,
            cached_vinfo: std::cell::RefCell::new(None),
        })
    }

    /// Create a VirtualStruct PtrInfo.
    pub fn virtual_struct(descr: DescrRef) -> Self {
        PtrInfo::VirtualStruct(VirtualStructInfo {
            descr,
            fields: Vec::new(),
            field_descrs: Vec::new(),
            last_guard_pos: -1,
            cached_vinfo: std::cell::RefCell::new(None),
        })
    }

    // ── Query methods ──

    /// Whether this pointer is known to be non-null.
    /// info.py: is_nonnull()
    pub fn is_nonnull(&self) -> bool {
        match self {
            PtrInfo::NonNull { .. } => true,
            PtrInfo::Constant(gcref) => !gcref.is_null(),
            PtrInfo::Instance(_)
            | PtrInfo::Struct(_)
            | PtrInfo::Array(_)
            | PtrInfo::Virtual(_)
            | PtrInfo::VirtualArray(_)
            | PtrInfo::VirtualStruct(_)
            | PtrInfo::VirtualArrayStruct(_)
            | PtrInfo::VirtualRawBuffer(_)
            | PtrInfo::VirtualRawSlice(_)
            | PtrInfo::Virtualizable(_)
            | PtrInfo::Str(_) => true,
        }
    }

    /// Whether this pointer is a virtual (allocation removed).
    /// info.py: is_virtual()
    ///
    /// Variant-specific gates match RPython:
    ///
    /// - `VirtualRawSlice`: `info.py:464-465`
    ///   `def is_virtual(self): return self.parent is not None`.
    ///   pyre encodes "no parent" as `OpRef::NONE` (set by
    ///   `force_box_impl` when the slice is materialized; see
    ///   `info.py:473-476 RawSlicePtrInfo._force_elements`).
    /// - The other `Virtual*` variants carry no per-instance sentinel
    ///   in pyre; their enum tag alone marks them virtual, which
    ///   matches the RPython subclasses whose `is_virtual()` defaults
    ///   stay True until the info itself is replaced (e.g.
    ///   `RawBufferPtrInfo.is_virtual()` at `info.py:417-418` flips via
    ///   `self.size != -1` — tracked as a separate future fix so the
    ///   enum-tag gate remains structurally accurate for those types).
    pub fn is_virtual(&self) -> bool {
        match self {
            PtrInfo::Virtual(_)
            | PtrInfo::VirtualArray(_)
            | PtrInfo::VirtualStruct(_)
            | PtrInfo::VirtualArrayStruct(_)
            | PtrInfo::VirtualRawBuffer(_) => true,
            PtrInfo::VirtualRawSlice(slice) => !slice.parent.is_none(),
            PtrInfo::Str(sinfo) => sinfo.is_virtual(),
            _ => false,
        }
    }

    /// Whether this is a constant pointer.
    /// info.py: isinstance(info, ConstPtrInfo)
    pub fn is_constant(&self) -> bool {
        matches!(self, PtrInfo::Constant(_))
    }

    /// info.py:763-772 `ConstPtrInfo.get_known_class(cpu)` +
    /// the other PtrInfo subclasses' `_known_class` accessors:
    ///
    /// ```text
    /// def get_known_class(self, cpu):
    ///     if not self._const.nonnull():
    ///         return None
    ///     if cpu.supports_guard_gc_type:
    ///         if not cpu.check_is_object(self._const.getref_base()):
    ///             return None
    ///     return cpu.cls_of_box(self._const)
    /// ```
    ///
    /// - `Instance`/`Virtual`: return the stored `known_class` field
    ///   (PyPy `InstancePtrInfo._known_class`). A class-only result of
    ///   `make_constant_class` is also stored as `Instance(descr=None,
    ///   known_class=Some(...))`.
    /// - `Constant`: null constants → `None`; otherwise, when the
    ///   backend supports `guard_gc_type` (`majit_gc::supports_guard_gc_type`),
    ///   gate `cls_of_box` on `majit_gc::check_is_object` so that
    ///   non-object constant pointers are rejected and the optimizer
    ///   does not read garbage at offset 0. When the backend does
    ///   not support `guard_gc_type`, RPython skips the
    ///   `check_is_object` call entirely and still returns
    ///   `cls_of_box(self._const)`; this port follows that.
    /// - Everything else: `None`.
    pub fn get_known_class(&self) -> Option<GcRef> {
        match self {
            PtrInfo::Instance(v) => v.known_class,
            PtrInfo::Virtual(v) => v.known_class,
            PtrInfo::Constant(gcref) => {
                // info.py:764: `if not self._const.nonnull(): return None`
                if gcref.is_null() {
                    return None;
                }
                // info.py:765-767: gate the `check_is_object` call on
                // `supports_guard_gc_type`. When the backend doesn't
                // support guard_gc_type, RPython simply skips the
                // `check_is_object` step and still calls `cls_of_box`.
                if majit_gc::supports_guard_gc_type() && !majit_gc::check_is_object(*gcref) {
                    return None;
                }
                // info.py:768 / llmodel.py:556-561 `cls_of_box`: read
                // the typeptr at offset 0 of the payload.
                let vtable = unsafe { *(gcref.0 as *const usize) };
                if vtable == 0 {
                    None
                } else {
                    Some(GcRef(vtable))
                }
            }
            _ => None,
        }
    }

    /// Get constant GcRef value if this is a constant pointer.
    pub fn get_constant_ref(&self) -> Option<&GcRef> {
        match self {
            PtrInfo::Constant(r) => Some(r),
            _ => None,
        }
    }

    /// info.py:83: make_guards(op, short, optimizer)
    /// info.py: make_guards(self, op, short, optimizer)
    ///
    /// Append guard operations to `short` that check this PtrInfo's
    /// properties hold for `op`. Used by use_box (shortpreamble.py:382).
    /// `alloc_const` allocates a constant-namespace OpRef and seeds the
    /// value — RPython equivalent: ConstInt(value) / ConstPtr(value).
    pub fn make_guards(
        &self,
        op: OpRef,
        short: &mut Vec<Op>,
        alloc_const: &mut impl FnMut(Value) -> OpRef,
    ) {
        match self {
            // info.py:83-84: PtrInfo base — no-op
            PtrInfo::NonNull { .. } => {
                // info.py:120-122: NonNullPtrInfo.make_guards
                short.push(Op::new(OpCode::GuardNonnull, &[op]));
            }
            PtrInfo::Instance(info) => {
                // info.py:336-353: InstancePtrInfo.make_guards
                if let Some(cls) = &info.known_class {
                    // remove_gctypeptr branch
                    let class_ref = alloc_const(Value::Ref(*cls));
                    short.push(Op::new(OpCode::GuardNonnullClass, &[op, class_ref]));
                } else if let Some(descr) = &info.descr {
                    // info.py:346-351: descr-only branch.
                    //   short.append(GUARD_NONNULL[op])
                    //   short.append(GUARD_SUBCLASS[op, ConstInt(descr.get_vtable())])
                    short.push(Op::new(OpCode::GuardNonnull, &[op]));
                    let vtable = descr
                        .as_size_descr()
                        .map(|sd| sd.vtable() as i64)
                        .unwrap_or(0);
                    let vtable_const = alloc_const(Value::Int(vtable));
                    short.push(Op::new(OpCode::GuardSubclass, &[op, vtable_const]));
                } else {
                    // info.py:353: fall back to AbstractStructPtrInfo →
                    // NonNullPtrInfo.make_guards (just GUARD_NONNULL).
                    short.push(Op::new(OpCode::GuardNonnull, &[op]));
                }
            }
            PtrInfo::Struct(info) => {
                // info.py:360-366: StructPtrInfo.make_guards.
                //   if self.descr is not None:
                //       c_typeid = ConstInt(self.descr.get_type_id())
                //       short.extend([GUARD_NONNULL[op],
                //                     GUARD_GC_TYPE[op, c_typeid]])
                let type_id = info
                    .descr
                    .as_size_descr()
                    .map(|sd| sd.type_id() as i64)
                    .unwrap_or(0);
                let type_id_const = alloc_const(Value::Int(type_id));
                short.push(Op::new(OpCode::GuardNonnull, &[op]));
                short.push(Op::new(OpCode::GuardGcType, &[op, type_id_const]));
            }
            PtrInfo::Constant(gcref) => {
                // info.py:715-716: ConstPtrInfo.make_guards
                let c = alloc_const(Value::Ref(*gcref));
                short.push(Op::new(OpCode::GuardValue, &[op, c]));
            }
            PtrInfo::Array(info) => {
                // info.py:632-639: ArrayPtrInfo.make_guards.
                //   AbstractVirtualPtrInfo.make_guards → NonNullPtrInfo.make_guards
                //   short.append(GUARD_GC_TYPE[op, ConstInt(descr.get_type_id())])
                //   if self.lenbound is not None:
                //       lenop = ARRAYLEN_GC[op] (descr=self.descr)
                //       short.append(lenop)
                //       self.lenbound.make_guards(lenop, short, optimizer)
                short.push(Op::new(OpCode::GuardNonnull, &[op]));
                let type_id = info
                    .descr
                    .as_array_descr()
                    .map(|ad| ad.type_id() as i64)
                    .unwrap_or(0);
                let type_id_const = alloc_const(Value::Int(type_id));
                short.push(Op::new(OpCode::GuardGcType, &[op, type_id_const]));
                // Always emit ARRAYLEN_GC + bound guards: pyre's
                // ArrayPtrInfo.lenbound is a plain `IntBound`, not an
                // `Option`, so the parity check is on `is_unbounded()`
                // rather than `is None`.
                if !info.lenbound.is_unbounded() {
                    let lenop = Op::with_descr(OpCode::ArraylenGc, &[op], info.descr.clone());
                    let lenop_pos = lenop.pos;
                    short.push(lenop);
                    info.lenbound.make_guards(lenop_pos, short, alloc_const);
                }
            }
            // info.py:379-384 `AbstractRawPtrInfo.make_guards`:
            //
            // ```python
            // def make_guards(self, op, short, optimizer):
            //     from rpython.jit.metainterp.optimizeopt.optimizer import CONST_0
            //     op = ResOperation(rop.INT_EQ, [op, CONST_0])
            //     short.append(op)
            //     op = ResOperation(rop.GUARD_FALSE, [op])
            //     short.append(op)
            // ```
            //
            // Emits "must not be 0" check (null-pointer equivalent for
            // Int-typed raw pointers) at the short-preamble entry.
            // Both `RawBufferPtrInfo` (info.py:386) and
            // `RawSlicePtrInfo` (info.py:459) inherit this override.
            PtrInfo::VirtualRawBuffer(_) | PtrInfo::VirtualRawSlice(_) => {
                let zero = alloc_const(Value::Int(0));
                let eq_op = Op::new(OpCode::IntEq, &[op, zero]);
                let eq_pos = eq_op.pos;
                short.push(eq_op);
                short.push(Op::new(OpCode::GuardFalse, &[eq_pos]));
            }
            PtrInfo::Str(sinfo) => {
                // vstring.py:116-126: StrPtrInfo.make_guards
                short.push(Op::new(OpCode::GuardNonnull, &[op]));
                if let Some(ref bound) = sinfo.lenbound {
                    if bound.lower >= 1 {
                        let lenop_code = if sinfo.mode == 0 {
                            OpCode::Strlen
                        } else {
                            OpCode::Unicodelen
                        };
                        let lenop = Op::new(lenop_code, &[op]);
                        let lenop_pos = lenop.pos;
                        short.push(lenop);
                        // intutils.py:1264-1289 IntBound.make_guards: emits the
                        // chained INT_GE/INT_LE/INT_AND → GUARD_TRUE/GUARD_VALUE
                        // pairs against `lenop_pos`.
                        bound.make_guards(lenop_pos, short, alloc_const);
                    }
                }
            }
            // Virtuals/Virtualizable: no guards needed in short preamble
            _ => {}
        }
    }

    /// vstring.py:112: return self.lgtop — cached length OpRef if available.
    pub fn get_cached_lgtop(&self) -> Option<OpRef> {
        match self {
            PtrInfo::Str(info) => info.lgtop,
            _ => None,
        }
    }

    /// info.py:74-75 / vstring.py:103-105 / 249-258 — common string-length
    /// query across `ConstPtrInfo` and `StrPtrInfo`.
    pub fn get_known_str_length(
        &self,
        ctx: &crate::optimizeopt::OptContext,
        mode: u8,
    ) -> Option<i64> {
        match self {
            PtrInfo::Str(info) => info.getstrlen(ctx, mode),
            // info.py:804-808 ConstPtrInfo.getstrlen — delegate to
            // the runtime resolver for constant string pointers.
            PtrInfo::Constant(gcref) if !gcref.is_null() => ctx
                .string_length_resolver
                .as_deref()
                .and_then(|resolver| resolver(*gcref, mode)),
            _ => None,
        }
    }

    /// info.py:793 ConstPtrInfo.get_constant_string_spec and
    /// vstring.py:178 / 236 / 298 — recursive constant string extraction.
    pub fn get_constant_string_spec(
        &self,
        ctx: &crate::optimizeopt::OptContext,
        mode: u8,
    ) -> Option<Vec<i64>> {
        match self {
            PtrInfo::Str(info) => info.get_constant_string_spec(ctx, mode),
            // info.py:793: ConstPtrInfo.get_constant_string_spec
            // delegates to _unpack_str(mode) → extracts chars from the
            // constant GcRef.
            PtrInfo::Constant(gcref) if !gcref.is_null() => ctx
                .string_content_resolver
                .as_deref()
                .and_then(|resolver| resolver(*gcref, mode)),
            PtrInfo::Constant(_) => None,
            _ => None,
        }
    }

    /// vstring.py:172 / 230 `strgetitem()` on string ptrinfo — virtual dispatch only.
    /// ConstPtr constant resolution is handled by `OptString::strgetitem`
    /// (vstring.py:393-403 `_strgetitem`), which needs `&mut OptContext`.
    pub fn strgetitem(&self, index: i64, ctx: &crate::optimizeopt::OptContext) -> Option<OpRef> {
        match self {
            PtrInfo::Str(info) => info.strgetitem(index, ctx),
            _ => None,
        }
    }

    /// info.py:826-838 ConstPtrInfo.getstrhash
    ///
    /// ```text
    /// def getstrhash(self, op, mode):
    ///     from rpython.jit.metainterp.optimizeopt import vstring
    ///     if mode is vstring.mode_string:
    ///         s = self._unpack_str(vstring.mode_string)
    ///         if s is None:
    ///             return None
    ///         return ConstInt(compute_hash(s))
    ///     else:
    ///         s = self._unpack_str(vstring.mode_unicode)
    ///         if s is None:
    ///             return None
    ///         return ConstInt(compute_hash(s))
    /// ```
    ///
    /// Like `getstrlen`, the actual hash needs a runtime hook because
    /// majit's `GcRef` is opaque. Returns `None` until pyre wires a
    /// `hash_resolver` into `OptContext`.
    pub fn getstrhash<F>(&self, mode: u8, mut resolver: F) -> Option<i64>
    where
        F: FnMut(majit_ir::GcRef, u8) -> Option<i64>,
    {
        match self {
            PtrInfo::Constant(gcref) if !gcref.is_null() => resolver(*gcref, mode),
            _ => None,
        }
    }

    /// Count the number of fields/items in this virtual object.
    /// info.py: _get_num_items() / num_fields
    pub fn num_fields(&self) -> usize {
        match self {
            PtrInfo::Instance(v) => v.fields.len(),
            PtrInfo::Struct(v) => v.fields.len(),
            PtrInfo::Array(v) => v.items.len(),
            PtrInfo::Virtual(v) => v.fields.len(),
            PtrInfo::VirtualArray(v) => v.items.len(),
            PtrInfo::VirtualStruct(v) => v.fields.len(),
            PtrInfo::VirtualArrayStruct(v) => v.element_fields.len(),
            PtrInfo::VirtualRawBuffer(v) => v.offsets.len(),
            _ => 0,
        }
    }

    /// Enumerate all OpRef values stored in this virtual's fields/items.
    /// info.py: visitor_walk_recursive — walks all fields of a virtual.
    pub fn visitor_walk_recursive(&self) -> Vec<OpRef> {
        match self {
            PtrInfo::Instance(v) => v.fields.iter().filter_map(|(_, e)| e.as_opref()).collect(),
            PtrInfo::Struct(v) => v.fields.iter().filter_map(|(_, e)| e.as_opref()).collect(),
            PtrInfo::Array(v) => v.items.iter().filter_map(|e| e.as_opref()).collect(),
            PtrInfo::Virtual(v) => v.fields.iter().map(|(_, r)| *r).collect(),
            PtrInfo::VirtualArray(v) => v.items.clone(),
            PtrInfo::VirtualStruct(v) => v.fields.iter().map(|(_, r)| *r).collect(),
            PtrInfo::VirtualArrayStruct(v) => v
                .element_fields
                .iter()
                .flat_map(|fields| fields.iter().map(|(_, r)| *r))
                .collect(),
            PtrInfo::VirtualRawBuffer(v) => v.values.clone(),
            // info.py:478-482 `RawSlicePtrInfo._visitor_walk_recursive`:
            //
            // ```python
            // def _visitor_walk_recursive(self, op, visitor):
            //     source_op = get_box_replacement(op.getarg(0))
            //     visitor.register_virtual_fields(op, [source_op])
            //     if self.parent.is_virtual():
            //         self.parent.visitor_walk_recursive(source_op, visitor)
            // ```
            //
            // RPython registers the parent OpRef as the sole "field" of the
            // slice; the subsequent recursive walk into `self.parent` is
            // driven by the visitor itself once it sees the parent OpRef.
            // pyre's walker returns the flat list of OpRefs a visitor
            // should enqueue, so surfacing the parent here is sufficient —
            // the visitor's outer loop re-enters `visitor_walk_recursive`
            // on the parent once it drains the queue.
            PtrInfo::VirtualRawSlice(v) => vec![v.parent],
            PtrInfo::Virtualizable(v) => {
                let mut refs: Vec<OpRef> = v.fields.iter().map(|(_, r)| *r).collect();
                for (_, items) in &v.arrays {
                    refs.extend(items.iter().copied());
                }
                refs
            }
            _ => Vec::new(),
        }
    }

    /// info.py: force_at_the_end_of_preamble(op, optforce, rec)
    ///
    /// RPython does not blindly materialize every virtual at the end of the
    /// preamble. Struct-like virtuals recurse into pointer children and update
    /// those field/item boxes in place, while leaving the top-level virtual in
    /// the exported virtual state.
    pub fn force_at_the_end_of_preamble<F>(&mut self, mut recurse: F)
    where
        F: FnMut(OpRef) -> OpRef,
    {
        match self {
            PtrInfo::Virtual(v) => {
                for (_, field) in &mut v.fields {
                    if !field.is_none() {
                        *field = recurse(*field);
                    }
                }
            }
            PtrInfo::VirtualStruct(v) => {
                for (_, field) in &mut v.fields {
                    if !field.is_none() {
                        *field = recurse(*field);
                    }
                }
            }
            PtrInfo::VirtualArray(v) => {
                for item in &mut v.items {
                    if !item.is_none() {
                        *item = recurse(*item);
                    }
                }
            }
            PtrInfo::VirtualArrayStruct(v) => {
                for fields in &mut v.element_fields {
                    for (_, field) in fields {
                        if !field.is_none() {
                            *field = recurse(*field);
                        }
                    }
                }
            }
            // info.py:374-384 `AbstractRawPtrInfo` inherits
            // `AbstractVirtualPtrInfo._force_at_the_end_of_preamble`
            // (info.py:159-160) unchanged — the base calls `force_box()`
            // to materialize instead of recursing into fields.  pyre's
            // dispatcher at `optimizer.rs::force_at_the_end_of_preamble_rec`
            // routes `VirtualRawBuffer` / `VirtualRawSlice` to `force_box`
            // directly, so neither variant reaches this recurse path.  No
            // explicit arm is needed; the `_` below falls through.
            _ => {}
        }
    }

    /// info.py `_cached_vinfo` accessor.
    ///
    /// Returns the per-instance `RefCell<Option<RdVirtualInfo>>` cache when
    /// `self` is one of the virtual variants that stores it; `None` for
    /// non-virtual variants. `make_virtual_info` (resume.py:307-315) uses
    /// this to dedup RdVirtualInfo allocations across multiple finish()
    /// calls that reference the same virtual.
    pub fn cached_vinfo(
        &self,
    ) -> Option<&std::cell::RefCell<Option<std::rc::Rc<majit_ir::RdVirtualInfo>>>> {
        match self {
            PtrInfo::Virtual(v) => Some(&v.cached_vinfo),
            PtrInfo::VirtualStruct(v) => Some(&v.cached_vinfo),
            PtrInfo::VirtualArray(v) => Some(&v.cached_vinfo),
            PtrInfo::VirtualArrayStruct(v) => Some(&v.cached_vinfo),
            PtrInfo::VirtualRawBuffer(v) => Some(&v.cached_vinfo),
            PtrInfo::VirtualRawSlice(v) => Some(&v.cached_vinfo),
            // info.py:124-128 + vstring.py:50,55 — StrPtrInfo inherits
            // _cached_vinfo from AbstractVirtualPtrInfo.
            PtrInfo::Str(v) => Some(&v.cached_vinfo),
            _ => None,
        }
    }

    /// info.py:331 / 369 / 376 / 445 / 485 / 598 / 701 +
    /// vstring.py:211 / 263 / 333 `visitor_dispatch_virtual_type`.
    ///
    /// Each virtual `PtrInfo` subclass implements `visitor_dispatch_virtual_type(visitor)`
    /// which calls the corresponding `visitor.visit_*()` method with the
    /// subclass's static metadata (descr, fielddescrs, array clear flag,
    /// raw buffer offsets, etc.). The visitor is free to produce a
    /// `VInfo` per call; the same visitor pattern is shared by
    /// `ResumeDataVirtualAdder` (resume.py:312) and `VirtualStateConstructor`
    /// (virtualstate.py:721).
    ///
    /// Returns `None` for non-virtual `PtrInfo` variants — RPython's
    /// `visitor_dispatch_virtual_type` is only defined on
    /// `AbstractVirtualPtrInfo` subclasses, so callers must check
    /// `is_virtual()` first.
    pub fn visitor_dispatch_virtual_type<V: crate::walkvirtual::VirtualVisitor>(
        &self,
        visitor: &mut V,
    ) -> Option<V::VInfo> {
        match self {
            // info.py:331-334 InstancePtrInfo.visitor_dispatch_virtual_type.
            // `fields` still stores sparse `(field_index, OpRef)` entries, but
            // the visitor now rebuilds the full descriptor-order slot list so
            // resume.py can pair `fielddescrs` and `fieldnums` 1:1 again.
            PtrInfo::Virtual(info) => {
                let indices: Vec<u32> = info.fields.iter().map(|(fi, _)| *fi).collect();
                Some(visitor.visit_virtual(&info.descr, &indices, &info.field_descrs))
            }
            // info.py:369-372 StructPtrInfo.visitor_dispatch_virtual_type
            PtrInfo::VirtualStruct(info) => {
                let indices: Vec<u32> = info.fields.iter().map(|(fi, _)| *fi).collect();
                Some(visitor.visit_vstruct(&info.descr, &indices, &info.field_descrs))
            }
            // info.py:598-599 ArrayPtrInfo.visitor_dispatch_virtual_type
            PtrInfo::VirtualArray(info) => Some(visitor.visit_varray(&info.descr, info.clear)),
            // info.py:701-704 ArrayStructInfo.visitor_dispatch_virtual_type.
            // The visitor consumes the canonical `fielddescrs` ordering; the
            // compatibility indices are the same descriptor-order slot numbers.
            PtrInfo::VirtualArrayStruct(info) => {
                let indices: Vec<u32> = (0..info.fielddescrs.len()).map(|i| i as u32).collect();
                Some(visitor.visit_varraystruct(
                    &info.descr,
                    info.element_fields.len(),
                    &indices,
                    &info.fielddescrs,
                ))
            }
            // info.py:445-450 RawBufferPtrInfo.visitor_dispatch_virtual_type
            PtrInfo::VirtualRawBuffer(info) => {
                Some(visitor.visit_vrawbuffer(info.func, info.size, &info.offsets, &info.descrs))
            }
            // info.py:485-486 RawSlicePtrInfo.visitor_dispatch_virtual_type
            PtrInfo::VirtualRawSlice(info) => Some(visitor.visit_vrawslice(info.offset)),
            // vstring.py:211-212 / 263-264 / 333-334 per-variant dispatch
            PtrInfo::Str(info) if info.is_virtual() => {
                let is_unicode = info.mode != 0;
                Some(match &info.variant {
                    VStringVariant::Plain(_) => visitor.visit_vstrplain(is_unicode),
                    VStringVariant::Concat(_) => visitor.visit_vstrconcat(is_unicode),
                    VStringVariant::Slice(_) => visitor.visit_vstrslice(is_unicode),
                    VStringVariant::Ptr => unreachable!("non-virtual Str reached virtual arm"),
                })
            }
            _ => None,
        }
    }

    /// info.py:137-160 / 222-226: force_box() emits the allocation and
    /// field writes via emit_extra(), recursively forcing child virtuals.
    ///
    /// Generated ops are routed via emit_extra() (RPython
    /// emit_extra parity) so downstream passes can observe them.
    pub fn force_box(&mut self, opref: OpRef, ctx: &mut crate::optimizeopt::OptContext) -> OpRef {
        self.force_box_impl(opref, ctx)
    }

    fn force_box_impl(&mut self, opref: OpRef, ctx: &mut crate::optimizeopt::OptContext) -> OpRef {
        use majit_ir::{Op, OpCode};

        fn force_child(value_ref: OpRef, ctx: &mut crate::optimizeopt::OptContext) -> OpRef {
            let value_ref = ctx.get_box_replacement(value_ref);
            if ctx.get_ptr_info(value_ref).is_some_and(|i| i.is_virtual()) {
                let mut info = ctx.take_ptr_info(value_ref).unwrap();
                let forced = info.force_box_impl(value_ref, ctx);
                return ctx.get_box_replacement(forced);
            }
            value_ref
        }

        // RPython info.py:148,226: optforce.emit_extra(op)
        // `optforce` determines where emitted ops enter the pass chain:
        //   optforce=Optimizer (in_final_emission) → emit directly
        //   optforce=OptEarlyForce → route from earlyforce.next (= heap)
        // When called from EarlyForce pass, current_pass_idx == earlyforce_idx
        // so emit_extra automatically routes from earlyforce.next.
        // When called from _emit_operation, in_final_emission=true → direct.
        let emit_op = |ctx: &mut crate::optimizeopt::OptContext, op: Op| -> OpRef {
            if ctx.in_final_emission {
                ctx.emit(op)
            } else {
                ctx.emit_extra(ctx.current_pass_idx, op)
            }
        };

        // RPython info.py:140-145: immutable virtual filled with constants
        // → constant fold to a compile-time constant pointer.
        if self.is_immutable_and_filled_with_constants(ctx) {
            if let Some(ref alloc_fn) = ctx.constant_fold_alloc {
                let (descr, fields, field_descrs) = match self {
                    PtrInfo::Virtual(v) => (&v.descr, &v.fields, &v.field_descrs),
                    PtrInfo::VirtualStruct(v) => (&v.descr, &v.fields, &v.field_descrs),
                    _ => unreachable!(),
                };
                let obj_size = descr.as_size_descr().map(|sd| sd.size()).unwrap_or(0);
                if obj_size > 0 {
                    let ptr = alloc_fn(obj_size);
                    if !ptr.is_null() {
                        // info.py:144: _force_elements_immutable
                        // Write constant field values directly to the allocated memory.
                        for &(field_idx, val_ref) in fields.iter() {
                            let resolved = ctx.get_box_replacement(val_ref);
                            if let Some(value) = ctx.get_constant(resolved) {
                                if let Some(fd) = lookup_field_descr(field_descrs, field_idx) {
                                    if let Some(field_d) = fd.as_field_descr() {
                                        let offset = field_d.offset();
                                        match value {
                                            Value::Int(v) => unsafe {
                                                let dest =
                                                    (ptr.0 as *mut u8).add(offset) as *mut i64;
                                                *dest = *v;
                                            },
                                            Value::Ref(r) => unsafe {
                                                let dest =
                                                    (ptr.0 as *mut u8).add(offset) as *mut usize;
                                                *dest = r.0;
                                            },
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                        // info.py:142: op.set_forwarded(constptr)
                        let const_ref = GcRef(ptr.0);
                        ctx.make_constant(opref, Value::Ref(const_ref));
                        ctx.set_ptr_info(opref, PtrInfo::Constant(const_ref));
                        return opref;
                    }
                }
            }
            // No allocator or size unknown: fall through to normal force.
        }

        match self {
            PtrInfo::VirtualStruct(vinfo) => {
                // RPython info.py:216-226 _force_elements clears each
                // `self._fields[i] = None` BEFORE `optforce.emit_extra(setfieldop)`.
                // After force, the non-virtual structinfo carries no field cache,
                // so heap.py do_setfield records the SETFIELD_GC as a lazy_set
                // instead of MUST_ALIAS-eliding it against the preserved value.
                let preserved = PtrInfo::Struct(StructPtrInfo {
                    descr: vinfo.descr.clone(),
                    fields: Vec::new(),
                    field_descrs: vinfo.field_descrs.clone(),
                    last_guard_pos: -1,
                });
                let mut new_op = Op::new(OpCode::New, &[]);
                // RPython info.py:146-151 force_box emits the ORIGINAL box op.
                // Preserve that identity here instead of inventing a fresh
                // OpRef, so later passes (earlyforce → heap → call) all talk
                // about the same concrete allocation.
                new_op.pos = opref;
                new_op.descr = Some(vinfo.descr.clone());
                let alloc_ref = emit_op(ctx, new_op);
                // Store preserved info on alloc_ref (canonical after force).
                ctx.set_ptr_info(alloc_ref, preserved);
                if crate::optimizeopt::majit_log_enabled() {
                    eprintln!(
                        "[jit][force-box] virtual-struct {:?} -> {:?} in_final_emission={} pass_idx={}",
                        opref, alloc_ref, ctx.in_final_emission, ctx.current_pass_idx
                    );
                }
                if opref != alloc_ref {
                    ctx.replace_op(opref, alloc_ref);
                }
                for (field_idx, value_ref) in std::mem::take(&mut vinfo.fields) {
                    let value_ref = force_child(value_ref, ctx);
                    let descr = lookup_field_descr(&vinfo.field_descrs, field_idx);
                    debug_assert!(
                        descr.is_some(),
                        "force_box: field_idx={} has value but no descriptor \
                         — field_descrs out of sync with fields",
                        field_idx,
                    );
                    let descr = descr.expect(
                        "force_box: field_idx must resolve through descr.get_all_fielddescrs()[i]",
                    );
                    let mut set_op = Op::new(OpCode::SetfieldGc, &[alloc_ref, value_ref]);
                    set_op.descr = Some(descr);
                    emit_op(ctx, set_op);
                }
                alloc_ref
            }
            PtrInfo::Virtual(vinfo) => {
                // info.py:216-226 — see VirtualStruct branch above. Build the
                // non-virtual replacement with no field cache so heap.py
                // do_setfield does not MUST_ALIAS-elide the materialization
                // SETFIELD_GC against the preserved value.
                let preserved = PtrInfo::Instance(InstancePtrInfo {
                    descr: Some(vinfo.descr.clone()),
                    known_class: vinfo.known_class,
                    fields: Vec::new(),
                    field_descrs: vinfo.field_descrs.clone(),
                    last_guard_pos: -1,
                });
                let mut new_op = Op::new(OpCode::NewWithVtable, &[]);
                // RPython info.py:146-151 force_box emits the ORIGINAL box op.
                // Preserve that identity here instead of inventing a fresh
                // OpRef, so later passes (earlyforce → heap → call) all talk
                // about the same concrete allocation.
                new_op.pos = opref;
                new_op.descr = Some(vinfo.descr.clone());
                let alloc_ref = emit_op(ctx, new_op);
                ctx.set_ptr_info(alloc_ref, preserved);
                if crate::optimizeopt::majit_log_enabled() {
                    eprintln!(
                        "[jit][force-box] virtual {:?} -> {:?} in_final_emission={} pass_idx={}",
                        opref, alloc_ref, ctx.in_final_emission, ctx.current_pass_idx
                    );
                }
                if opref != alloc_ref {
                    ctx.replace_op(opref, alloc_ref);
                }
                for (field_idx, value_ref) in std::mem::take(&mut vinfo.fields) {
                    let value_ref = force_child(value_ref, ctx);
                    let descr = lookup_field_descr(&vinfo.field_descrs, field_idx);
                    let descr = descr.expect(
                        "force_box: field_idx must resolve through descr.get_all_fielddescrs()[i]",
                    );
                    let mut set_op = Op::new(OpCode::SetfieldGc, &[alloc_ref, value_ref]);
                    set_op.descr = Some(descr);
                    emit_op(ctx, set_op);
                }
                alloc_ref
            }
            PtrInfo::VirtualArray(vinfo) => {
                // info.py:540-558 ArrayPtrInfo._force_elements
                let len = vinfo.items.len();
                ctx.set_ptr_info(opref, PtrInfo::nonnull());

                let len_ref = ctx.emit_constant_int(len as i64);
                let alloc_opcode = if vinfo.clear {
                    OpCode::NewArrayClear
                } else {
                    OpCode::NewArray
                };
                let mut alloc_op = Op::new(alloc_opcode, &[len_ref]);
                alloc_op.pos = opref;
                alloc_op.descr = Some(vinfo.descr.clone());
                let alloc_ref = emit_op(ctx, alloc_op);
                if opref != alloc_ref {
                    ctx.replace_op(opref, alloc_ref);
                }

                // info.py:542: const = optforce.optimizer.new_const_item(self.descr)
                // info.py:546-548: skip items equal to the default when _clear=True
                let items = std::mem::take(&mut vinfo.items);
                let clear = vinfo.clear;
                let descr = vinfo.descr.clone();
                for (i, item_ref) in items.into_iter().enumerate() {
                    if item_ref == OpRef::NONE {
                        continue;
                    }
                    // info.py:543: const = optforce.optimizer.new_const_item(self.descr)
                    // info.py:546-548: if self._clear and const.same_constant(item)
                    // new_const_item returns CONST_0/CONST_NULL/CONST_ZERO_FLOAT
                    // (all raw=0). getconst handles constant_types_for_numbering.
                    if clear {
                        let resolved = ctx.get_box_replacement(item_ref);
                        let is_default = matches!(ctx.getconst(resolved), Some((0, _)));
                        if is_default {
                            continue;
                        }
                    }
                    let subbox = force_child(item_ref, ctx);
                    let idx_ref = ctx.emit_constant_int(i as i64);
                    let mut set_op = Op::new(OpCode::SetarrayitemGc, &[alloc_ref, idx_ref, subbox]);
                    set_op.descr = Some(descr.clone());
                    emit_op(ctx, set_op);
                }
                // info.py:557: optforce.pure_from_args(ARRAYLEN_GC, [op], ConstInt(len))
                ctx.pure_from_args_arraylen(alloc_ref, len as i64);
                alloc_ref
            }
            PtrInfo::VirtualArrayStruct(vinfo) => {
                // info.py:670-684 ArrayStructInfo._force_elements
                // virtualize.py:31: assert clear — ArrayStruct is always
                // created with clear=True, so the original op is always
                // NEW_ARRAY_CLEAR.
                let num_elements = vinfo.element_fields.len();
                ctx.set_ptr_info(opref, PtrInfo::nonnull());

                let len_ref = ctx.emit_constant_int(num_elements as i64);
                let mut alloc_op = Op::new(OpCode::NewArrayClear, &[len_ref]);
                alloc_op.pos = opref;
                alloc_op.descr = Some(vinfo.descr.clone());
                let alloc_ref = emit_op(ctx, alloc_op);
                if opref != alloc_ref {
                    ctx.replace_op(opref, alloc_ref);
                }

                // info.py:672: fielddescrs = op.getdescr().get_all_fielddescrs()
                let fielddescrs: Vec<majit_ir::DescrRef> = vinfo
                    .descr
                    .as_array_descr()
                    .and_then(|ad| ad.get_all_interiorfielddescrs())
                    .map(|fds| fds.to_vec())
                    .unwrap_or_else(|| vinfo.fielddescrs.clone());
                let element_fields = std::mem::take(&mut vinfo.element_fields);
                // info.py:673-684:
                //   for index in range(self.length):
                //       for fielddescr in fielddescrs:
                //           fld = self._items[i]
                //           if fld is not None:
                //               subbox = optforce.optimizer.force_box(fld)
                //               setfieldop = ResOperation(SETINTERIORFIELD_GC,
                //                   [op, ConstInt(index), subbox], descr=fielddescr)
                //               optforce.emit_extra(setfieldop)
                //           i += 1
                for (elem_idx, fields) in element_fields.into_iter().enumerate() {
                    let idx_ref = ctx.emit_constant_int(elem_idx as i64);
                    for (field_idx, value_ref) in fields {
                        if value_ref.is_none() {
                            continue;
                        }
                        let subbox = force_child(value_ref, ctx);
                        let mut set_op =
                            Op::new(OpCode::SetinteriorfieldGc, &[alloc_ref, idx_ref, subbox]);
                        set_op.descr = fielddescrs.get(field_idx as usize).cloned();
                        emit_op(ctx, set_op);
                    }
                }
                alloc_ref
            }
            PtrInfo::VirtualRawBuffer(vinfo) => {
                // info.py:420-436: RawBufferPtrInfo._force_elements()
                // info.py:421: self.size = -1 (mark as no longer virtual)
                let offsets = std::mem::take(&mut vinfo.offsets);
                let descrs = std::mem::take(&mut vinfo.descrs);
                let values = std::mem::take(&mut vinfo.values);
                let func = vinfo.func;
                let size = vinfo.size;
                let calldescr = vinfo.calldescr.take();

                // info.py:148: emit CALL_I(func, ConstInt(size), descr=calldescr)
                let func_ref = ctx.emit_constant_int(func);
                let size_ref = ctx.emit_constant_int(size as i64);
                let mut call_op = Op::new(OpCode::CallI, &[func_ref, size_ref]);
                call_op.pos = opref;
                call_op.descr = calldescr;
                let alloc_ref = emit_op(ctx, call_op);

                ctx.set_ptr_info(alloc_ref, PtrInfo::nonnull());
                if opref != alloc_ref {
                    ctx.replace_op(opref, alloc_ref);
                }

                // info.py:425: CHECK_MEMORY_ERROR
                let check_op = Op::new(OpCode::CheckMemoryError, &[alloc_ref]);
                emit_op(ctx, check_op);

                // info.py:429-436: emit RAW_STORE for each buffered write
                for i in 0..offsets.len() {
                    let value_ref = force_child(values[i], ctx);
                    let offset_ref = ctx.emit_constant_int(offsets[i] as i64);
                    let mut store_op =
                        Op::new(OpCode::RawStore, &[alloc_ref, offset_ref, value_ref]);
                    store_op.descr = Some(descrs[i].clone());
                    emit_op(ctx, store_op);
                }

                alloc_ref
            }
            PtrInfo::VirtualRawSlice(slice) => {
                // `info.py:473-476` `RawSlicePtrInfo._force_elements`:
                //
                // ```python
                // def _force_elements(self, op, optforce, descr):
                //     if self.parent.is_virtual():
                //         self.parent._force_elements(op, optforce, descr)
                //     self.parent = None
                // ```
                //
                // RPython keeps the `RawSlicePtrInfo` attached to the op and
                // flips it to non-virtual by setting `self.parent = None`
                // (`is_virtual` at info.py:464-465 is `self.parent is not None`).
                // The info class stays RawSlicePtrInfo so subsequent
                // `getrawptrinfo` lookups still identify it as a raw slice.
                //
                // pyre's `VirtualRawSliceInfo` stores `parent: OpRef`; the
                // `OpRef::NONE` sentinel plays the role of `None`, and
                // `PtrInfo::is_virtual` gates on `slice.parent.is_none()`.
                // Overwriting with `PtrInfo::nonnull()` would lose the
                // raw-slice identity and mis-route any later
                // `get_virtual_fields` / raw-guard path.
                let parent_forced = force_child(slice.parent, ctx);
                let parent_forced = ctx.get_box_replacement(parent_forced);
                let offset_ref = ctx.emit_constant_int(slice.offset as i64);
                let mut add_op = Op::new(OpCode::IntAdd, &[parent_forced, offset_ref]);
                add_op.pos = opref;
                let new_ref = emit_op(ctx, add_op);
                // Preserve raw-slice identity; mark non-virtual via
                // `parent = OpRef::NONE` (RPython `self.parent = None`).
                ctx.set_ptr_info(
                    new_ref,
                    PtrInfo::VirtualRawSlice(VirtualRawSliceInfo {
                        offset: slice.offset,
                        parent: OpRef::NONE,
                        last_guard_pos: slice.last_guard_pos,
                        cached_vinfo: std::cell::RefCell::new(None),
                    }),
                );
                if opref != new_ref {
                    ctx.replace_op(opref, new_ref);
                }
                new_ref
            }
            PtrInfo::Str(sinfo) if sinfo.is_virtual() => {
                // vstring.py:76-103 StrPtrInfo.force_box
                let mode = sinfo.mode;
                let is_unicode = mode != 0;

                // vstring.py:79-90: if self.mode is mode_string / else
                let c_s = if mode == crate::optimizeopt::vstring::mode_string {
                    // vstring.py:80-84
                    sinfo
                        .get_constant_string_spec(&*ctx, mode)
                        .and_then(|chars| {
                            crate::optimizeopt::vstring::get_const_ptr_for_string(&chars, ctx)
                        })
                } else {
                    // vstring.py:86-90
                    sinfo
                        .get_constant_string_spec(&*ctx, mode)
                        .and_then(|chars| {
                            crate::optimizeopt::vstring::get_const_ptr_for_unicode(&chars, ctx)
                        })
                };
                if let Some(gcref) = c_s {
                    // vstring.py:83: get_box_replacement(op).set_forwarded(c_s)
                    ctx.make_constant(opref, Value::Ref(gcref));
                    return opref;
                }

                // vstring.py:91: self._is_virtual = False
                let sinfo_full = match std::mem::replace(self, PtrInfo::nonnull()) {
                    PtrInfo::Str(s) => s,
                    _ => unreachable!(),
                };
                let variant = sinfo_full.variant;

                // vstring.py:92: lengthbox = self.getstrlen(op, optstring, mode)
                let lengthbox = match &variant {
                    VStringVariant::Plain(info) => ctx.emit_constant_int(info._chars.len() as i64),
                    VStringVariant::Slice(info) => ctx.get_box_replacement(info.lgtop),
                    VStringVariant::Concat(info) => {
                        let left_len = ctx.getstrlen_opref(info.vleft, mode);
                        let right_len = ctx.getstrlen_opref(info.vright, mode);
                        crate::optimizeopt::vstring::_int_add(left_len, right_len, ctx)
                    }
                    VStringVariant::Ptr => unreachable!(),
                };

                // vstring.py:93-96: newop = ResOperation(mode.NEWSTR, [lengthbox])
                let new_opcode = if is_unicode {
                    OpCode::Newunicode
                } else {
                    OpCode::Newstr
                };
                let mut newstr_op = Op::new(new_opcode, &[lengthbox]);
                newstr_op.pos = opref;
                let newop = emit_op(ctx, newstr_op);

                // vstring.py:98: newop.set_forwarded(self)
                ctx.set_ptr_info(
                    newop,
                    PtrInfo::Str(StrPtrInfo {
                        lenbound: sinfo_full.lenbound,
                        lgtop: Some(lengthbox), // vstring.py:98 preserve computed length
                        mode: sinfo_full.mode,
                        length: sinfo_full.length,
                        variant: VStringVariant::Ptr, // non-virtual
                        last_guard_pos: sinfo_full.last_guard_pos,
                        cached_vinfo: std::cell::RefCell::new(None),
                    }),
                );

                // vstring.py:99-100: op.set_forwarded(newop)
                if opref != newop {
                    ctx.replace_op(opref, newop);
                }

                // vstring.py:101-102: initialize_forced_string(op, optstring, op, CONST_0, mode)
                let zero = ctx.emit_constant_int(0);
                let set_opcode = if is_unicode {
                    OpCode::Unicodesetitem
                } else {
                    OpCode::Strsetitem
                };

                match variant {
                    VStringVariant::Plain(info) => {
                        // vstring.py:194-205 VStringPlainInfo.initialize_forced_string
                        let mut offset = zero;
                        let one = ctx.emit_constant_int(1);
                        for ch in &info._chars {
                            if let Some(ch_ref) = ch {
                                let ch_resolved = ctx.get_box_replacement(*ch_ref);
                                let setitem_op = Op::new(set_opcode, &[newop, offset, ch_resolved]);
                                emit_op(ctx, setitem_op);
                            }
                            offset = crate::optimizeopt::vstring::_int_add(offset, one, ctx);
                        }
                    }
                    VStringVariant::Concat(info) => {
                        // vstring.py:309-317 VStringConcatInfo.string_copy_parts
                        let offset = crate::optimizeopt::vstring::string_copy_parts(
                            info.vleft, newop, zero, mode, ctx,
                        );
                        crate::optimizeopt::vstring::string_copy_parts(
                            info.vright,
                            newop,
                            offset,
                            mode,
                            ctx,
                        );
                    }
                    VStringVariant::Slice(info) => {
                        // vstring.py:230-233 VStringSliceInfo.string_copy_parts
                        crate::optimizeopt::vstring::copy_str_content(
                            ctx, info.s, newop, info.start, zero, info.lgtop, mode, true,
                        );
                    }
                    VStringVariant::Ptr => unreachable!(),
                }

                newop
            }
            _ => opref,
        }
    }

    /// info.py: make_guards(op, short_boxes, optimizer)
    /// Generate guard opcodes (without args) to verify this pointer info.
    /// Legacy helper for tests — use make_guards() for full guard emission.
    pub fn guard_opcodes(&self) -> Vec<majit_ir::OpCode> {
        match self {
            PtrInfo::NonNull { .. } => vec![majit_ir::OpCode::GuardNonnull],
            PtrInfo::Instance(info) if info.known_class.is_some() => {
                vec![majit_ir::OpCode::GuardNonnullClass]
            }
            PtrInfo::Instance(info) if info.descr.is_some() => vec![
                majit_ir::OpCode::GuardNonnull,
                majit_ir::OpCode::GuardIsObject,
                majit_ir::OpCode::GuardSubclass,
            ],
            PtrInfo::Struct(_) | PtrInfo::Array(_) => {
                vec![
                    majit_ir::OpCode::GuardNonnull,
                    majit_ir::OpCode::GuardGcType,
                ]
            }
            PtrInfo::Constant(_) => vec![majit_ir::OpCode::GuardValue],
            _ => Vec::new(),
        }
    }

    /// info.py: is_null() — whether this pointer is known to be null.
    pub fn is_null(&self) -> bool {
        match self {
            PtrInfo::Constant(gcref) => gcref.is_null(),
            _ => false,
        }
    }

    /// info.py:64-69 `PtrInfo.getnullness()` parity (line-by-line port).
    ///
    /// ```python
    /// def getnullness(self):
    ///     if self.is_null():
    ///         return INFO_NULL
    ///     elif self.is_nonnull():
    ///         return INFO_NONNULL
    ///     return INFO_UNKNOWN
    /// ```
    ///
    /// Returns one of `INFO_NULL` / `INFO_NONNULL` / `INFO_UNKNOWN`
    /// (info.py:13-15). majit's representation matches RPython's
    /// integer enum: NULL=0, NONNULL=1, UNKNOWN=2.
    pub fn getnullness(&self) -> i8 {
        if self.is_null() {
            crate::optimizeopt::INFO_NULL
        } else if self.is_nonnull() {
            crate::optimizeopt::INFO_NONNULL
        } else {
            crate::optimizeopt::INFO_UNKNOWN
        }
    }

    /// `info.py:44-45` `PtrInfo.is_about_object(): return False` (base
    /// default) / `info.py:327-328`
    /// `InstancePtrInfo.is_about_object(): return True` (override).
    ///
    /// RPython only overrides `is_about_object` on `InstancePtrInfo`;
    /// `StructPtrInfo`, `ArrayPtrInfo`, the raw variants, and the
    /// other abstract subclasses inherit the `False` default.  pyre's
    /// `Virtual` variant maps 1:1 to `InstancePtrInfo` in its virtual
    /// state (has `known_class` / vtable), so it mirrors True; all
    /// other variants must return False to keep `optimize_GUARD_IS_OBJECT`
    /// (rewrite.py:210-211) and `optimize_GUARD_SUBCLASS` (rewrite.py:241-243)
    /// from eliding guards on non-instance pointers.
    pub fn is_about_object(&self) -> bool {
        matches!(self, PtrInfo::Instance(_) | PtrInfo::Virtual(_))
    }

    /// `info.py:28-29` `PtrInfo.is_precise(): return False` (base default)
    /// / `info.py:134-135` `AbstractVirtualPtrInfo.is_precise(): return True`
    /// (overrides for every virtual-descriptor-carrying subclass).
    ///
    /// RPython's class hierarchy:
    ///   - `PtrInfo`, `NonNullPtrInfo`, `ConstPtrInfo` → inherit the False
    ///     default.
    ///   - `AbstractVirtualPtrInfo` and its subclasses (`InstancePtrInfo`,
    ///     `StructPtrInfo`, `ArrayPtrInfo`, `VArrayStructInfo`,
    ///     `RawBufferPtrInfo`, `RawSlicePtrInfo`, + virtual variants of
    ///     the above) → return True.
    ///
    /// pyre's enum flattens the class hierarchy, so the `True` list matches
    /// AbstractVirtualPtrInfo's subclasses directly.  `PtrInfo::Constant`
    /// maps to `ConstPtrInfo` which inherits from `PtrInfo` (not
    /// AbstractVirtualPtrInfo), so it gets the False default.  The `Str`
    /// variant maps to `StrPtrInfo(AbstractVirtualPtrInfo)` (vstring.py:50),
    /// so it inherits `is_precise=True`.
    pub fn is_precise(&self) -> bool {
        matches!(
            self,
            PtrInfo::Instance(_)
                | PtrInfo::Struct(_)
                | PtrInfo::Array(_)
                | PtrInfo::Virtual(_)
                | PtrInfo::VirtualArray(_)
                | PtrInfo::VirtualStruct(_)
                | PtrInfo::VirtualArrayStruct(_)
                | PtrInfo::VirtualRawBuffer(_)
                | PtrInfo::VirtualRawSlice(_)
                | PtrInfo::Str(_)
        )
    }

    /// info.py: same_info(other) — whether two PtrInfos describe the same value.
    /// info.py:71-72: same_info() → `self is other` (identity).
    /// ConstPtrInfo overrides to compare constant values (info.py:774-777).
    pub fn same_info(&self, other: &PtrInfo) -> bool {
        match (self, other) {
            (PtrInfo::Constant(a), PtrInfo::Constant(b)) => a == b,
            _ => std::ptr::eq(self, other),
        }
    }

    /// info.py: get_descr() — get the size/type descriptor for virtual objects.
    pub fn get_descr(&self) -> Option<&DescrRef> {
        match self {
            PtrInfo::Instance(v) => v.descr.as_ref(),
            PtrInfo::Struct(v) => Some(&v.descr),
            PtrInfo::Array(v) => Some(&v.descr),
            PtrInfo::Virtual(v) => Some(&v.descr),
            PtrInfo::VirtualArray(v) => Some(&v.descr),
            PtrInfo::VirtualStruct(v) => Some(&v.descr),
            PtrInfo::VirtualArrayStruct(v) => Some(&v.descr),
            _ => None,
        }
    }

    /// `getlenbound(mode)` — polymorphic dispatch matching the PyPy class
    /// hierarchy:
    ///
    /// - info.py:61-62 `PtrInfo.getlenbound(mode)` — base default returns None
    /// - info.py:515-521 `ArrayPtrInfo.getlenbound(mode)` — asserts mode is None,
    ///   lazy-creates `nonnegative` lenbound on first access
    /// - vstring.py:62-70 `StrPtrInfo.getlenbound(mode)` — lazy-creates from
    ///   `self.length` (constant) or `nonnegative`
    /// - info.py:796-802 `ConstPtrInfo.getlenbound(mode)` — handled by
    ///   `EnsuredPtrInfo::Constant::getlenbound`, which routes through the
    ///   runtime `string_length_resolver`. The base `PtrInfo` method
    ///   below intentionally returns `None` for `PtrInfo::Constant` so
    ///   callers that bypass `EnsuredPtrInfo` don't accidentally produce
    ///   a stale `nonnegative` answer without consulting the resolver.
    ///
    /// Returns an owned `IntBound` so callers (which typically need `&mut
    /// OptContext` next for `setintbound`) don't have to juggle borrows.
    pub fn getlenbound(&mut self, mode: Option<u8>) -> Option<IntBound> {
        match self {
            // info.py:515-521 ArrayPtrInfo.getlenbound: assert mode is None
            PtrInfo::Array(v) => {
                debug_assert!(
                    mode.is_none(),
                    "ArrayPtrInfo.getlenbound: mode must be None"
                );
                Some(v.lenbound.clone())
            }
            // info.py:ArrayPtrInfo (virtual branch) + info.py:641-647
            // `ArrayStructInfo(ArrayPtrInfo).__init__` which stores
            // `self.lenbound = IntBound.from_constant(size)`.  pyre's
            // `VirtualArrayInfo` keeps the array size implicit as
            // `items.len()` and `VirtualArrayStructInfo` keeps it as
            // `element_fields.len()`; synthesize the constant bound so
            // `ARRAYLEN_GC` / `INT_GE` / `INT_LT` postprocessing sees
            // the same information the ArrayPtrInfo branch does.
            PtrInfo::VirtualArray(v) => {
                debug_assert!(
                    mode.is_none(),
                    "VirtualArrayInfo.getlenbound: mode must be None"
                );
                Some(IntBound::from_constant(v.items.len() as i64))
            }
            PtrInfo::VirtualArrayStruct(v) => {
                debug_assert!(
                    mode.is_none(),
                    "VirtualArrayStructInfo.getlenbound: mode must be None"
                );
                Some(IntBound::from_constant(v.element_fields.len() as i64))
            }
            // vstring.py:62-70 StrPtrInfo.getlenbound: lazy lenbound
            PtrInfo::Str(sinfo) => {
                if sinfo.lenbound.is_none() {
                    sinfo.lenbound = Some(if sinfo.length == -1 {
                        IntBound::nonnegative()
                    } else {
                        IntBound::from_constant(sinfo.length as i64)
                    });
                }
                sinfo.lenbound.clone()
            }
            // info.py:61-62 base PtrInfo.getlenbound returns None.
            // The constant case is handled by EnsuredPtrInfo (which has
            // access to the runtime string_length_resolver).
            _ => None,
        }
    }

    pub fn init_fields(&mut self, descr: DescrRef, index: usize) {
        let Some(size_descr) = descr.as_size_descr() else {
            return;
        };
        let all_fielddescrs: Vec<DescrRef> = size_descr
            .all_fielddescrs()
            .iter()
            .map(|field_descr| field_descr.clone() as DescrRef)
            .collect();
        match self {
            PtrInfo::Instance(v) => {
                if v.field_descrs.is_empty() {
                    v.descr = Some(descr);
                    v.field_descrs = all_fielddescrs;
                } else if index >= v.field_descrs.len() {
                    v.descr = Some(descr);
                    v.field_descrs
                        .extend(all_fielddescrs.into_iter().skip(v.field_descrs.len()));
                }
            }
            PtrInfo::Struct(v) => {
                if v.field_descrs.is_empty() {
                    v.descr = descr;
                    v.field_descrs = all_fielddescrs;
                } else if index >= v.field_descrs.len() {
                    v.descr = descr;
                    v.field_descrs
                        .extend(all_fielddescrs.into_iter().skip(v.field_descrs.len()));
                }
            }
            PtrInfo::Virtual(v) => {
                if v.field_descrs.is_empty() {
                    v.descr = descr;
                    v.field_descrs = all_fielddescrs;
                } else if index >= v.field_descrs.len() {
                    v.descr = descr;
                    v.field_descrs
                        .extend(all_fielddescrs.into_iter().skip(v.field_descrs.len()));
                }
            }
            PtrInfo::VirtualStruct(v) => {
                if v.field_descrs.is_empty() {
                    v.descr = descr;
                    v.field_descrs = all_fielddescrs;
                } else if index >= v.field_descrs.len() {
                    v.descr = descr;
                    v.field_descrs
                        .extend(all_fielddescrs.into_iter().skip(v.field_descrs.len()));
                }
            }
            _ => {}
        }
    }

    /// info.py: setfield(field_descr, value) — set a field on a virtual object.
    /// info.py:176-200 setfield — update the field value in the virtual.
    /// RPython: _fields[fielddescr.get_index()] = op. In majit, fields
    /// is a (field_idx, OpRef) list; field_descrs is managed separately
    /// by OptVirtualize (optimize_setfield_gc).
    pub fn setfield(&mut self, field_idx: u32, value: OpRef) {
        match self {
            PtrInfo::Instance(v) => {
                for entry in &mut v.fields {
                    if entry.0 == field_idx {
                        entry.1 = FieldEntry::Value(value);
                        return;
                    }
                }
                v.fields.push((field_idx, FieldEntry::Value(value)));
            }
            PtrInfo::Struct(v) => {
                for entry in &mut v.fields {
                    if entry.0 == field_idx {
                        entry.1 = FieldEntry::Value(value);
                        return;
                    }
                }
                v.fields.push((field_idx, FieldEntry::Value(value)));
            }
            PtrInfo::Virtual(v) => {
                for entry in &mut v.fields {
                    if entry.0 == field_idx {
                        entry.1 = value;
                        return;
                    }
                }
                v.fields.push((field_idx, value));
            }
            PtrInfo::VirtualStruct(v) => {
                for entry in &mut v.fields {
                    if entry.0 == field_idx {
                        entry.1 = value;
                        return;
                    }
                }
                v.fields.push((field_idx, value));
            }
            _ => {}
        }
    }

    /// shortpreamble.py:73-79: HeapOp.produce_op stores PreambleOp in _fields.
    /// RPython: `opinfo.setfield(descr, struct, pop, optheap, cf)`
    /// where `pop` is a PreambleOp wrapper.
    pub fn set_preamble_field(&mut self, field_idx: u32, pop: PreambleOp) {
        // shortpreamble.py:74: assert not opinfo.is_virtual()
        assert!(!self.is_virtual(), "set_preamble_field on virtual");
        match self {
            PtrInfo::Instance(v) => {
                v.fields.retain(|(k, _)| *k != field_idx);
                v.fields.push((field_idx, FieldEntry::Preamble(pop)));
            }
            PtrInfo::Struct(v) => {
                v.fields.retain(|(k, _)| *k != field_idx);
                v.fields.push((field_idx, FieldEntry::Preamble(pop)));
            }
            _ => {
                // RPython: AbstractStructPtrInfo always supports _fields.
                // In majit, NonNull / Constant / Str / Virtualizable etc.
                // lack _fields. Upgrade to Instance — known_class
                // is None because make_constant_class would already have
                // installed an Instance with the class set, which would
                // have hit the first match arm above.
                *self = PtrInfo::Instance(InstancePtrInfo {
                    descr: None,
                    known_class: None,
                    fields: vec![(field_idx, FieldEntry::Preamble(pop))],
                    field_descrs: Vec::new(),
                    last_guard_pos: -1,
                });
            }
        }
    }

    /// shortpreamble.py:80-85 stores `PreambleOp` in array `_items[index]`.
    /// Rust keeps these separate from `items` for the same reason as fields:
    /// `PreambleOp` is not an `OpRef`.
    pub fn set_preamble_item(&mut self, index: usize, pop: PreambleOp) {
        // shortpreamble.py:74: assert not opinfo.is_virtual()
        assert!(!self.is_virtual(), "set_preamble_item on virtual");
        if let PtrInfo::Array(v) = self {
            if index >= v.items.len() {
                v.items.resize(index + 1, FieldEntry::Value(OpRef::NONE));
            }
            v.items[index] = FieldEntry::Preamble(pop);
        }
    }

    /// RPython: `isinstance(res, PreambleOp)` check in _getfield.
    /// Returns true if preamble_fields has an entry for this field_idx.
    pub fn has_preamble_field(&self, field_idx: u32) -> bool {
        match self {
            PtrInfo::Instance(v) => v
                .fields
                .iter()
                .any(|(k, e)| *k == field_idx && e.is_preamble()),
            PtrInfo::Struct(v) => v
                .fields
                .iter()
                .any(|(k, e)| *k == field_idx && e.is_preamble()),
            _ => false,
        }
    }

    /// RPython: `isinstance(res, PreambleOp)` check in ArrayCachedItem._getfield.
    /// Returns true if preamble_items has an entry for this index.
    pub fn has_preamble_item(&self, index: usize) -> bool {
        match self {
            PtrInfo::Array(v) => v.items.get(index).map_or(false, |e| e.is_preamble()),
            _ => false,
        }
    }

    /// heap.py:177-187: CachedField._getfield detects PreambleOp in _fields.
    /// Returns and removes the PreambleOp if present for this field.
    /// RPython: `isinstance(res, PreambleOp)` check in _getfield.
    pub fn take_preamble_field(&mut self, field_idx: u32) -> Option<PreambleOp> {
        match self {
            PtrInfo::Instance(v) => {
                if let Some(pos) = v
                    .fields
                    .iter()
                    .position(|(k, e)| *k == field_idx && e.is_preamble())
                {
                    v.fields.remove(pos).1.into_preamble()
                } else {
                    None
                }
            }
            PtrInfo::Struct(v) => {
                if let Some(pos) = v
                    .fields
                    .iter()
                    .position(|(k, e)| *k == field_idx && e.is_preamble())
                {
                    v.fields.remove(pos).1.into_preamble()
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// heap.py:238-250: ArrayCachedItem._getfield detects `PreambleOp` in
    /// `_items[index]`, forces it, and writes the resolved result back.
    pub fn take_preamble_item(&mut self, index: usize) -> Option<PreambleOp> {
        match self {
            PtrInfo::Array(v) => {
                if let Some(entry) = v.items.get_mut(index) {
                    if entry.is_preamble() {
                        let taken = std::mem::replace(entry, FieldEntry::Value(OpRef::NONE));
                        taken.into_preamble()
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// info.py:273-303: _is_immutable_and_filled_with_constants
    ///
    /// ```text
    /// if not self.descr.is_immutable():
    ///     return False
    /// for op in self._fields:
    ///     if op is None:
    ///         return False     # uninitialized field
    ///     ...
    /// ```
    ///
    /// Check if this virtual is immutable and all fields are constants.
    /// Used by force_box to determine if the virtual can be constant-folded.
    pub fn is_immutable_and_filled_with_constants(
        &self,
        ctx: &crate::optimizeopt::OptContext,
    ) -> bool {
        let (fields, descr) = match self {
            PtrInfo::Virtual(v) => (&v.fields, &v.descr),
            PtrInfo::VirtualStruct(v) => (&v.fields, &v.descr),
            _ => return false,
        };
        // info.py:281: `if not self.descr.is_immutable()`.
        let Some(size_descr) = descr.as_size_descr() else {
            return false;
        };
        if !size_descr.is_immutable() {
            return false;
        }
        // info.py:286-288: `for op in self._fields: if op is None: return False`.
        // RPython's _fields is pre-allocated to len(descr.get_all_fielddescrs())
        // with None for unset slots; pyre stores only set entries in `fields`,
        // so parity requires fields.len() to match all_fielddescrs().len()
        // before treating the virtual as filled.
        if fields.len() != size_descr.all_fielddescrs().len() {
            return false;
        }
        for &(_, val) in fields {
            let resolved = ctx.get_box_replacement(val);
            if !ctx.is_constant(resolved) {
                // Check if it's a virtual that is also immutable+constant
                if let Some(info) = ctx.get_ptr_info(resolved) {
                    if info.is_virtual() && info.is_immutable_and_filled_with_constants(ctx) {
                        continue;
                    }
                }
                return false;
            }
        }
        true
    }

    /// heap.py:194: opinfo._fields[descr.get_index()] = None
    /// Clear a cached field value. Used by CachedField.invalidate().
    /// RPython stores PreambleOp in _fields[] too, so clearing a field
    /// index removes both regular and preamble entries.
    pub fn clear_field(&mut self, field_idx: u32) {
        match self {
            PtrInfo::Instance(v) => {
                v.fields.retain(|(k, _)| *k != field_idx);
            }
            PtrInfo::Struct(v) => {
                v.fields.retain(|(k, _)| *k != field_idx);
            }
            PtrInfo::Virtual(v) => v.fields.retain(|(k, _)| *k != field_idx),
            PtrInfo::VirtualStruct(v) => v.fields.retain(|(k, _)| *k != field_idx),
            _ => {}
        }
    }

    /// info.py: all_items() — return all cached field entries as (idx, OpRef).
    /// heap.py:211,214: opinfo.all_items() used by _cannot_alias_via_content.
    /// For Instance/Struct, preserve inline `PreambleOp` entries by exposing
    /// their source box identity (`pop.op`), which is what PyPy sees when it
    /// iterates `_fields[]` / `_items[]`.
    /// info.py:200-201: all_items — returns _fields directly.
    /// RPython: includes PreambleOp entries alongside normal values.
    pub fn all_items(&self) -> Vec<(u32, FieldEntry)> {
        match self {
            PtrInfo::Instance(v) => v.fields.clone(),
            PtrInfo::Struct(v) => v.fields.clone(),
            PtrInfo::Virtual(v) => v
                .fields
                .iter()
                .map(|(k, v)| (*k, FieldEntry::Value(*v)))
                .collect(),
            PtrInfo::VirtualStruct(v) => v
                .fields
                .iter()
                .map(|(k, v)| (*k, FieldEntry::Value(*v)))
                .collect(),
            // info.py:530 ArrayPtrInfo.all_items() returns self._items
            PtrInfo::Array(v) => v
                .items
                .iter()
                .enumerate()
                .map(|(i, e)| (i as u32, e.clone()))
                .collect(),
            PtrInfo::VirtualArray(v) => v
                .items
                .iter()
                .enumerate()
                .map(|(i, val)| (i as u32, FieldEntry::Value(*val)))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// info.py:212-214 AbstractStructPtrInfo.getfield
    ///
    /// Returns `FieldEntry` — callers must handle both `Value` and `Preamble`
    /// variants. RPython returns the _fields[] element directly which may be
    /// a PreambleOp sentinel; this matches that design.
    pub fn getfield(&self, field_idx: u32) -> Option<FieldEntry> {
        match self {
            PtrInfo::Instance(v) => v
                .fields
                .iter()
                .find(|(k, _)| *k == field_idx)
                .map(|(_, e)| e.clone()),
            PtrInfo::Struct(v) => v
                .fields
                .iter()
                .find(|(k, _)| *k == field_idx)
                .map(|(_, e)| e.clone()),
            PtrInfo::Virtual(v) => v
                .fields
                .iter()
                .find(|(k, _)| *k == field_idx)
                .map(|(_, v)| FieldEntry::Value(*v)),
            PtrInfo::VirtualStruct(v) => v
                .fields
                .iter()
                .find(|(k, _)| *k == field_idx)
                .map(|(_, v)| FieldEntry::Value(*v)),
            _ => None,
        }
    }

    /// info.py: setitem(index, value) — set an item in a virtual array.
    pub fn setitem(&mut self, index: usize, value: OpRef) {
        match self {
            PtrInfo::Array(v) => {
                if index >= v.items.len() {
                    v.items.resize(index + 1, FieldEntry::Value(OpRef::NONE));
                }
                v.items[index] = FieldEntry::Value(value);
            }
            PtrInfo::VirtualArray(v) => {
                if index < v.items.len() {
                    v.items[index] = value;
                }
            }
            _ => {}
        }
    }

    /// info.py: getitem(index) — get an item from a virtual array.
    /// Returns `FieldEntry` — callers must handle `Preamble` variants.
    pub fn getitem(&self, index: usize) -> Option<FieldEntry> {
        match self {
            PtrInfo::Array(v) => v.items.get(index).cloned(),
            PtrInfo::VirtualArray(v) => v.items.get(index).map(|r| FieldEntry::Value(*r)),
            _ => None,
        }
    }

    /// heap.py:257-262: ArrayCachedItem.invalidate clears
    /// `opinfo._items[self.index] = None` for cached_infos. The Rust
    /// port mirrors that by writing `OpRef::NONE` into the slot —
    /// matching `clear_field` semantics for struct fields.
    pub fn clear_item(&mut self, index: usize) {
        match self {
            PtrInfo::Array(v) => {
                if index < v.items.len() {
                    v.items[index] = FieldEntry::Value(OpRef::NONE);
                }
            }
            PtrInfo::VirtualArray(v) => {
                if index < v.items.len() {
                    v.items[index] = OpRef::NONE;
                }
            }
            _ => {}
        }
    }

    /// info.py:651-656: _compute_index(index, fielddescr)
    /// Computes flat index into VirtualArrayStruct's element_fields.
    fn compute_interior_index(
        &self,
        element_index: usize,
        field_descr_index: u32,
    ) -> Option<(usize, usize)> {
        match self {
            PtrInfo::VirtualArrayStruct(v) => {
                if element_index >= v.element_fields.len() {
                    return None;
                }
                // Find the slot for field_descr_index within this element.
                let fields = &v.element_fields[element_index];
                for (slot, &(fdidx, _)) in fields.iter().enumerate() {
                    if fdidx == field_descr_index {
                        return Some((element_index, slot));
                    }
                }
                // Field not yet present — return element index for insertion.
                Some((element_index, fields.len()))
            }
            _ => None,
        }
    }

    /// info.py:663-668: getinteriorfield_virtual(index, fielddescr)
    pub fn getinteriorfield_virtual(
        &self,
        element_index: usize,
        field_descr_index: u32,
    ) -> Option<OpRef> {
        match self {
            PtrInfo::VirtualArrayStruct(v) => {
                if element_index >= v.element_fields.len() {
                    return None;
                }
                v.element_fields[element_index]
                    .iter()
                    .find(|&&(fdidx, _)| fdidx == field_descr_index)
                    .map(|&(_, opref)| opref)
            }
            _ => None,
        }
    }

    /// info.py:658-661: setinteriorfield_virtual(index, fielddescr, fld)
    pub fn setinteriorfield_virtual(
        &mut self,
        element_index: usize,
        field_descr_index: u32,
        value: OpRef,
    ) {
        match self {
            PtrInfo::VirtualArrayStruct(v) => {
                if element_index >= v.element_fields.len() {
                    v.element_fields.resize(element_index + 1, Vec::new());
                }
                let fields = &mut v.element_fields[element_index];
                // Update existing or insert new.
                if let Some(entry) = fields
                    .iter_mut()
                    .find(|(fdidx, _)| *fdidx == field_descr_index)
                {
                    entry.1 = value;
                } else {
                    fields.push((field_descr_index, value));
                }
            }
            _ => {}
        }
    }

    /// info.py: produce_short_preamble_ops(structbox, descr, index, optimizer, shortboxes)
    ///
    /// Add cached field values to the short preamble builder.
    /// For each non-null field in the virtual, register a descriptor-carrying
    /// GETFIELD read so the bridge can re-populate the optimizer's field cache.
    pub fn produce_short_preamble_ops(&self, structbox: OpRef) -> Vec<Op> {
        let mut result = Vec::new();
        // Fields are accessed per-variant below
        if let PtrInfo::Virtual(v) = self {
            for &(field_idx, value) in &v.fields {
                if !value.is_none() {
                    let descr = lookup_field_descr(&v.field_descrs, field_idx)
                        .expect("produce_short_preamble_ops: virtual field descr missing");
                    result.push(Op::with_descr(OpCode::GetfieldGcI, &[structbox], descr));
                }
            }
        }
        if let PtrInfo::VirtualStruct(v) = self {
            for &(field_idx, value) in &v.fields {
                if !value.is_none() {
                    let descr = lookup_field_descr(&v.field_descrs, field_idx)
                        .expect("produce_short_preamble_ops: virtual struct field descr missing");
                    result.push(Op::with_descr(OpCode::GetfieldGcI, &[structbox], descr));
                }
            }
        }
        result
    }
}

/// A virtual object whose allocation has been removed.
///
/// Fields are tracked as OpRefs to the operations that produce their values.
///
/// ## Invariant: `fields` NEVER contains typeptr (offset 0)
///
/// Matches RPython upstream: `heaptracker.py:66-67 all_fielddescrs()` skips
/// `typeptr`, so `info.py:180 AbstractStructPtrInfo.init_fields` sizes
/// `_fields` with typeptr excluded from the indexable range. The typeptr
/// (offset 0) is tracked separately via `known_class` and emitted by the
/// GC rewriter's `gen_initialize_vtable` path (rewrite.py:479-484), NOT
/// from the force-path field loop.
///
/// Enforced by:
/// - `virtualize.rs optimize_setfield_gc` Virtual arm: runtime check that
///   returns early on `offset == Some(0)` before calling `set_field`.
/// - `virtualize.rs force_virtual_instance`: `debug_assert_no_typeptr`
///   at the entry of the field-emit loop.
/// - `virtualstate.rs export_single_value`:
///   `debug_assert_no_typeptr` on the fields collection boundary.
#[derive(Clone, Debug)]
pub struct VirtualInfo {
    /// The size descriptor of this object.
    pub descr: DescrRef,
    /// Known class (if any).
    pub known_class: Option<GcRef>,
    /// ob_type field descriptor for force path. In RPython the vtable is
    /// set by allocate_with_vtable, not as a struct field. pyre stores
    /// ob_type at offset 0 explicitly. This descr lets force emit
    /// SetfieldGc(ob_type) without polluting `fields` (which feeds rd_virtuals).
    pub ob_type_descr: Option<DescrRef>,
    /// Field values: `(field_descr_index, value_opref)`.
    /// **Invariant**: never contains typeptr (offset 0) — see struct-level docs.
    pub fields: Vec<(u32, OpRef)>,
    /// Original field descriptors, in parent-local slot order.
    pub field_descrs: Vec<DescrRef>,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py `_cached_vinfo` — cached RdVirtualInfo for resume data
    /// dedup (resume.py:309-314). RefCell for interior mutability so
    /// the immutable-receiver `make_virtual_info` trait method can
    /// populate the cache on first miss.
    pub cached_vinfo: std::cell::RefCell<Option<std::rc::Rc<majit_ir::RdVirtualInfo>>>,
}

/// A virtual array.
#[derive(Clone, Debug)]
pub struct VirtualArrayInfo {
    /// The array descriptor.
    pub descr: DescrRef,
    /// Whether this was created by NewArrayClear (zero-initialized).
    pub clear: bool,
    /// Element values.
    pub items: Vec<OpRef>,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py `_cached_vinfo` — see VirtualInfo.cached_vinfo.
    pub cached_vinfo: std::cell::RefCell<Option<std::rc::Rc<majit_ir::RdVirtualInfo>>>,
}

/// A non-virtual object with cached field info.
///
/// Mirrors RPython's InstancePtrInfo in the non-virtual case.
#[derive(Clone, Debug)]
pub struct InstancePtrInfo {
    /// Best-known instance descriptor, if any.
    pub descr: Option<DescrRef>,
    /// Known class pointer, if guarded exactly.
    pub known_class: Option<GcRef>,
    /// info.py:175 _fields — cached field values.
    /// RPython stores both normal Boxes and PreambleOp sentinels in the
    /// same list. Rust mirrors this with `Vec<(u32, FieldEntry)>`.
    pub fields: Vec<(u32, FieldEntry)>,
    /// Original field descriptors, in parent-local slot order.
    pub field_descrs: Vec<DescrRef>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A non-virtual GC struct with cached field info.
///
/// Mirrors RPython's StructPtrInfo in the non-virtual case.
#[derive(Clone, Debug)]
pub struct StructPtrInfo {
    /// Exact struct descriptor.
    pub descr: DescrRef,
    /// info.py:175 _fields — cached field values (same as InstancePtrInfo).
    pub fields: Vec<(u32, FieldEntry)>,
    /// Original field descriptors, in parent-local slot order.
    pub field_descrs: Vec<DescrRef>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A non-virtual GC array with cached item info and lenbound.
///
/// Mirrors RPython's ArrayPtrInfo in the non-virtual case.
#[derive(Clone, Debug)]
pub struct ArrayPtrInfo {
    /// Exact array descriptor.
    pub descr: DescrRef,
    /// Known bounds on the array length.
    pub lenbound: IntBound,
    /// info.py:579 _items — cached item values for constant indices.
    /// RPython stores both normal Boxes and PreambleOp sentinels.
    pub items: Vec<FieldEntry>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A virtual struct (no vtable).
#[derive(Clone, Debug)]
pub struct VirtualStructInfo {
    /// The size descriptor.
    pub descr: DescrRef,
    /// Field values: (field_index, value, optional original field descriptor).
    pub fields: Vec<(u32, OpRef)>,
    /// Original field descriptors, in parent-local slot order, used for force.
    pub field_descrs: Vec<DescrRef>,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py `_cached_vinfo` — see VirtualInfo.cached_vinfo.
    pub cached_vinfo: std::cell::RefCell<Option<std::rc::Rc<majit_ir::RdVirtualInfo>>>,
}

/// A virtual array of structs (interior field access pattern).
///
/// Mirrors RPython's VArrayStructInfo where each array element
/// is a fixed-size struct with named fields. Used for RPython arrays
/// with complex item types (e.g., hash table entries with key+value fields).
#[derive(Clone, Debug)]
pub struct VirtualArrayStructInfo {
    /// The array descriptor (arraydescr).
    pub descr: DescrRef,
    /// Per-element fields: outer Vec = elements, inner Vec = (field_descr_index, value_opref).
    pub element_fields: Vec<Vec<(u32, OpRef)>>,
    /// resume.py VArrayStructInfo.fielddescrs — InteriorFieldDescr per field.
    /// Used by _number_virtuals to extract item_size/field_offset/field_size.
    pub fielddescrs: Vec<DescrRef>,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py `_cached_vinfo` — see VirtualInfo.cached_vinfo.
    pub cached_vinfo: std::cell::RefCell<Option<std::rc::Rc<majit_ir::RdVirtualInfo>>>,
}

/// info.py:RawSlicePtrInfo — alias view into a parent virtual raw buffer.
///
/// Created by `make_virtual_raw_slice` (virtualize.py:60-65) when an
/// `INT_ADD(rawbuf, const_offset)` is folded against a virtual raw buffer.
/// Reads / writes through a slice add `offset` to the requested byte
/// offset and forward to the parent buffer.
#[derive(Clone, Debug)]
pub struct VirtualRawSliceInfo {
    /// Slice offset relative to the parent buffer's base.
    pub offset: usize,
    /// OpRef of the parent VirtualRawBuffer (or another VirtualRawSlice
    /// — `optimize_int_add` flattens chained slices when the underlying
    /// info is `VirtualRawBufferInfo`/`VirtualRawSliceInfo`).
    pub parent: OpRef,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py `_cached_vinfo` — see VirtualInfo.cached_vinfo.
    pub cached_vinfo: std::cell::RefCell<Option<std::rc::Rc<majit_ir::RdVirtualInfo>>>,
}

/// A virtual raw memory buffer.
///
/// rawbuffer.py:13 RawBuffer — 4 parallel lists: offsets, lengths, descrs, values.
/// Sorted by offset. Invariant: offsets[i]+lengths[i] <= offsets[i+1].
#[derive(Clone, Debug)]
pub struct VirtualRawBufferInfo {
    /// resume.py:695: self.func — raw malloc function pointer.
    pub func: i64,
    /// Size of the buffer in bytes.
    pub size: usize,
    /// rawbuffer.py:14: self.offsets
    pub offsets: Vec<usize>,
    /// rawbuffer.py:15: self.lengths
    pub lengths: Vec<usize>,
    /// rawbuffer.py:16: self.descrs — per-entry ArrayDescr.
    pub descrs: Vec<DescrRef>,
    /// rawbuffer.py:17: self.values
    pub values: Vec<OpRef>,
    /// info.py:91-92
    pub last_guard_pos: i32,
    /// info.py:420: calldescr for CALL_I(func, size) raw malloc.
    /// Saved from the original CALL_I op during virtualization.
    pub calldescr: Option<DescrRef>,
    /// info.py `_cached_vinfo` — see VirtualInfo.cached_vinfo.
    pub cached_vinfo: std::cell::RefCell<Option<std::rc::Rc<majit_ir::RdVirtualInfo>>>,
}

/// Error returned when a raw buffer operation violates invariants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RawBufferError {
    /// A write overlaps with an existing write.
    OverlappingWrite {
        new_offset: usize,
        new_length: usize,
        existing_offset: usize,
        existing_length: usize,
    },
    /// A read from an offset that was never written.
    UninitializedRead { offset: usize, length: usize },
    /// A read whose length/offset doesn't match the write at that offset.
    IncompatibleRead {
        offset: usize,
        read_length: usize,
        write_length: usize,
    },
}

impl VirtualRawBufferInfo {
    /// rawbuffer.py:83: _descrs_are_compatible(d1, d2)
    /// Two arraydescrs are compatible if they have the same basesize,
    /// itemsize and sign.
    fn descrs_are_compatible(d1: &DescrRef, d2: &DescrRef) -> bool {
        let (Some(a1), Some(a2)) = (d1.as_array_descr(), d2.as_array_descr()) else {
            return false;
        };
        a1.base_size() == a2.base_size()
            && a1.item_size() == a2.item_size()
            && a1.is_item_signed() == a2.is_item_signed()
    }

    /// rawbuffer.py:89: write_value(offset, length, descr, value).
    /// Maintains sorted order by offset. Same-offset update only
    /// replaces value (rawbuffer.py:102), never descr.
    pub fn write_value(
        &mut self,
        offset: usize,
        length: usize,
        descr: DescrRef,
        value: OpRef,
    ) -> Result<(), RawBufferError> {
        let mut insert_pos = 0;
        for i in 0..self.offsets.len() {
            let wo = self.offsets[i];
            let wl = self.lengths[i];
            if wo == offset {
                // rawbuffer.py:94-95: length and descr must be compatible.
                if wl != length || !Self::descrs_are_compatible(&descr, &self.descrs[i]) {
                    return Err(RawBufferError::OverlappingWrite {
                        new_offset: offset,
                        new_length: length,
                        existing_offset: wo,
                        existing_length: wl,
                    });
                }
                // rawbuffer.py:102: only replace value, keep existing descr.
                self.values[i] = value;
                return Ok(());
            } else if wo > offset {
                break;
            }
            insert_pos = i + 1;
        }
        // rawbuffer.py:108: check overlap with next entry.
        if insert_pos < self.offsets.len() {
            let next_off = self.offsets[insert_pos];
            if offset + length > next_off {
                return Err(RawBufferError::OverlappingWrite {
                    new_offset: offset,
                    new_length: length,
                    existing_offset: next_off,
                    existing_length: self.lengths[insert_pos],
                });
            }
        }
        // rawbuffer.py:111: check overlap with previous entry.
        if insert_pos > 0 {
            let prev_off = self.offsets[insert_pos - 1];
            let prev_len = self.lengths[insert_pos - 1];
            if prev_off + prev_len > offset {
                return Err(RawBufferError::OverlappingWrite {
                    new_offset: offset,
                    new_length: length,
                    existing_offset: prev_off,
                    existing_length: prev_len,
                });
            }
        }
        // rawbuffer.py:115-118: insert new entry.
        self.offsets.insert(insert_pos, offset);
        self.lengths.insert(insert_pos, length);
        self.descrs.insert(insert_pos, descr);
        self.values.insert(insert_pos, value);
        Ok(())
    }

    /// rawbuffer.py:120: read_value(offset, length, descr).
    pub fn read_value(
        &self,
        offset: usize,
        length: usize,
        descr: &DescrRef,
    ) -> Result<OpRef, RawBufferError> {
        for i in 0..self.offsets.len() {
            if self.offsets[i] == offset {
                if self.lengths[i] != length || !Self::descrs_are_compatible(descr, &self.descrs[i])
                {
                    return Err(RawBufferError::IncompatibleRead {
                        offset,
                        read_length: length,
                        write_length: self.lengths[i],
                    });
                }
                return Ok(self.values[i]);
            }
        }
        Err(RawBufferError::UninitializedRead { offset, length })
    }

    /// Check if a read at `(offset, size)` is fully covered by previous writes.
    pub fn is_read_fully_covered(&self, offset: usize, size: usize) -> bool {
        (0..size).all(|i| {
            let byte = offset + i;
            self.offsets
                .iter()
                .zip(self.lengths.iter())
                .any(|(&wo, &wl)| byte >= wo && byte < wo + wl)
        })
    }

    /// Find the index of an existing write that is completely overwritten
    /// by a new write at `(offset, size)`.
    pub fn find_overwritten_write(&self, offset: usize, size: usize) -> Option<usize> {
        self.offsets
            .iter()
            .zip(self.lengths.iter())
            .position(|(&wo, &wl)| offset <= wo && offset + size >= wo + wl)
    }
}

/// Tracked field state for a virtualizable object (interpreter frame).
///
/// Mirrors RPython's virtualizable handling in the optimizer:
/// the frame already exists on the heap, but during JIT execution its
/// fields are kept in registers. The optimizer tracks the current value
/// of each field so that redundant setfield/getfield ops are eliminated.
///
/// When the virtualizable is "forced" (escapes to non-JIT code), field
/// values are written back to the heap via SETFIELD_RAW ops.
#[derive(Clone, Debug)]
pub struct VirtualizableFieldState {
    /// Tracked static field values: (field_descr_index, current_value_opref).
    /// Indices correspond to VirtualizableInfo::static_fields order.
    pub fields: Vec<(u32, OpRef)>,
    /// Original field descriptors: (field_descr_index, original_descr).
    /// Used to emit correct SetfieldRaw ops when forcing.
    pub field_descrs: Vec<(u32, DescrRef)>,
    /// Tracked array field values: (array_field_index, element_values).
    /// Indices correspond to VirtualizableInfo::array_fields order.
    pub arrays: Vec<(u32, Vec<OpRef>)>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::OptContext;
    use majit_ir::{Descr, OpCode, Value};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestDescr;
    impl Descr for TestDescr {}

    /// Create an int ArrayDescr for tests (base_size=0, item_size=8, Int).
    fn int_descr() -> DescrRef {
        majit_ir::descr::make_array_descr(0, 8, majit_ir::Type::Int)
    }

    /// Create an int ArrayDescr with specified item_size for tests.
    fn int_descr_sz(item_size: usize) -> DescrRef {
        majit_ir::descr::make_array_descr(0, item_size, majit_ir::Type::Int)
    }

    fn make_buf(size: usize) -> VirtualRawBufferInfo {
        VirtualRawBufferInfo {
            func: 0,
            size,
            offsets: Vec::new(),
            lengths: Vec::new(),
            descrs: Vec::new(),
            values: Vec::new(),
            last_guard_pos: -1,
            calldescr: None,
            cached_vinfo: std::cell::RefCell::new(None),
        }
    }

    #[test]
    fn rawbuffer_write_and_read() {
        let mut buf = make_buf(32);
        let d = int_descr();
        let d4 = int_descr_sz(4);
        buf.write_value(0, 8, d.clone(), OpRef::from_raw(10))
            .unwrap();
        buf.write_value(8, 4, d4.clone(), OpRef::from_raw(20))
            .unwrap();
        buf.write_value(16, 8, d.clone(), OpRef::from_raw(30))
            .unwrap();

        assert_eq!(buf.read_value(0, 8, &d).unwrap(), OpRef::from_raw(10));
        assert_eq!(buf.read_value(8, 4, &d4).unwrap(), OpRef::from_raw(20));
        assert_eq!(buf.read_value(16, 8, &d).unwrap(), OpRef::from_raw(30));
    }

    #[test]
    fn rawbuffer_update_same_offset() {
        let mut buf = make_buf(16);
        let d = int_descr();
        buf.write_value(0, 8, d.clone(), OpRef::from_raw(10))
            .unwrap();
        buf.write_value(0, 8, d.clone(), OpRef::from_raw(99))
            .unwrap();

        assert_eq!(buf.read_value(0, 8, &d).unwrap(), OpRef::from_raw(99));
        assert_eq!(buf.offsets.len(), 1);
    }

    #[test]
    fn rawbuffer_overlap_next() {
        let mut buf = make_buf(32);
        let d = int_descr();
        buf.write_value(8, 8, d.clone(), OpRef::from_raw(10))
            .unwrap();
        // Write at offset 4 with length 8 overlaps [8, 16)
        let err = buf
            .write_value(4, 8, d.clone(), OpRef::from_raw(20))
            .unwrap_err();
        assert!(matches!(err, RawBufferError::OverlappingWrite { .. }));
    }

    #[test]
    fn rawbuffer_overlap_prev() {
        let mut buf = make_buf(32);
        let d = int_descr();
        let d4 = int_descr_sz(4);
        buf.write_value(0, 8, d.clone(), OpRef::from_raw(10))
            .unwrap();
        // Write at offset 4 overlaps with [0, 8)
        let err = buf.write_value(4, 4, d4, OpRef::from_raw(20)).unwrap_err();
        assert!(matches!(err, RawBufferError::OverlappingWrite { .. }));
    }

    #[test]
    fn rawbuffer_incompatible_length_at_same_offset() {
        let mut buf = make_buf(16);
        let d = int_descr();
        let d4 = int_descr_sz(4);
        buf.write_value(0, 8, d, OpRef::from_raw(10)).unwrap();
        let err = buf.write_value(0, 4, d4, OpRef::from_raw(20)).unwrap_err();
        assert!(matches!(err, RawBufferError::OverlappingWrite { .. }));
    }

    #[test]
    fn rawbuffer_uninitialized_read() {
        let buf = make_buf(16);
        let d = int_descr();
        let err = buf.read_value(0, 8, &d).unwrap_err();
        assert_eq!(
            err,
            RawBufferError::UninitializedRead {
                offset: 0,
                length: 8
            }
        );
    }

    #[test]
    fn rawbuffer_incompatible_read_length() {
        let mut buf = make_buf(16);
        let d = int_descr();
        let d4 = int_descr_sz(4);
        buf.write_value(0, 8, d, OpRef::from_raw(10)).unwrap();
        let err = buf.read_value(0, 4, &d4).unwrap_err();
        assert_eq!(
            err,
            RawBufferError::IncompatibleRead {
                offset: 0,
                read_length: 4,
                write_length: 8,
            }
        );
    }

    #[test]
    fn rawbuffer_read_fully_covered() {
        let mut buf = make_buf(32);
        let d = int_descr();
        buf.write_value(0, 8, d.clone(), OpRef::from_raw(10))
            .unwrap();
        buf.write_value(8, 8, d.clone(), OpRef::from_raw(20))
            .unwrap();

        // [0, 16) is fully covered by [0,8) + [8,16)
        assert!(buf.is_read_fully_covered(0, 16));
        // [0, 8) covered
        assert!(buf.is_read_fully_covered(0, 8));
        // [4, 8) falls within [0, 8)
        assert!(buf.is_read_fully_covered(4, 4));
    }

    #[test]
    fn rawbuffer_read_partially_covered_fails() {
        let mut buf = make_buf(32);
        let d4 = int_descr_sz(4);
        buf.write_value(0, 4, d4.clone(), OpRef::from_raw(10))
            .unwrap();
        buf.write_value(8, 4, d4.clone(), OpRef::from_raw(20))
            .unwrap();

        // Bytes 4..8 are not covered by any write
        assert!(!buf.is_read_fully_covered(0, 8));
        // Byte 16 was never written
        assert!(!buf.is_read_fully_covered(16, 4));
    }

    #[test]
    fn rawbuffer_overwritten_write_detected() {
        let mut buf = make_buf(32);
        let d4 = int_descr_sz(4);
        buf.write_value(4, 4, d4.clone(), OpRef::from_raw(10))
            .unwrap();
        buf.write_value(12, 4, d4.clone(), OpRef::from_raw(20))
            .unwrap();

        // A write [4, 12) fully contains [4, 8)
        assert_eq!(buf.find_overwritten_write(4, 8), Some(0));
        // A write [0, 16) fully contains [4, 8)
        assert_eq!(buf.find_overwritten_write(0, 16), Some(0));
        // A write [12, 20) fully contains [12, 16)
        assert_eq!(buf.find_overwritten_write(12, 8), Some(1));
        // A write [0, 4) does not contain any existing entry
        assert_eq!(buf.find_overwritten_write(0, 4), None);
    }

    #[test]
    fn rawbuffer_sorted_insertion() {
        let mut buf = make_buf(32);
        let d4 = int_descr_sz(4);
        buf.write_value(16, 4, d4.clone(), OpRef::from_raw(30))
            .unwrap();
        buf.write_value(0, 4, d4.clone(), OpRef::from_raw(10))
            .unwrap();
        buf.write_value(8, 4, d4.clone(), OpRef::from_raw(20))
            .unwrap();

        // Entries should be sorted by offset
        assert_eq!(buf.offsets[0], 0);
        assert_eq!(buf.offsets[1], 8);
        assert_eq!(buf.offsets[2], 16);
    }

    /// test_rawbuffer.py:66 test_unpack_descrs
    /// Two different descr objects with same (basesize, itemsize, signed)
    /// must be compatible. Different signed-ness must be incompatible.
    #[test]
    fn test_unpack_descrs() {
        use majit_ir::descr::SimpleArrayDescr;

        // ArrayS_8_1 and ArrayS_8_2: same (base=0, item=8, signed=true)
        let array_s_8_1: DescrRef = std::sync::Arc::new(SimpleArrayDescr::with_flag(
            0,
            0,
            8,
            0,
            majit_ir::Type::Int,
            majit_ir::descr::ArrayFlag::Signed,
        ));
        let array_s_8_2: DescrRef = std::sync::Arc::new(SimpleArrayDescr::with_flag(
            1,
            0,
            8,
            0,
            majit_ir::Type::Int,
            majit_ir::descr::ArrayFlag::Signed,
        ));
        // ArrayU_8: same size but unsigned
        let array_u_8: DescrRef = std::sync::Arc::new(SimpleArrayDescr::with_flag(
            2,
            0,
            8,
            0,
            majit_ir::Type::Int,
            majit_ir::descr::ArrayFlag::Unsigned,
        ));

        assert!(!std::sync::Arc::ptr_eq(&array_s_8_1, &array_s_8_2));

        let mut buf = make_buf(16);

        // Write with ArrayS_8_1
        buf.write_value(0, 4, array_s_8_1.clone(), OpRef::from_raw(10))
            .unwrap();

        // Read with same descr
        assert_eq!(
            buf.read_value(0, 4, &array_s_8_1).unwrap(),
            OpRef::from_raw(10)
        );
        // Read with non-identical but compatible descr
        assert_eq!(
            buf.read_value(0, 4, &array_s_8_2).unwrap(),
            OpRef::from_raw(10)
        );

        // Overwrite with non-identical compatible descr
        buf.write_value(0, 4, array_s_8_2.clone(), OpRef::from_raw(20))
            .unwrap();
        assert_eq!(
            buf.read_value(0, 4, &array_s_8_1).unwrap(),
            OpRef::from_raw(20)
        );

        // Incompatible descr (unsigned) must fail
        assert!(buf.read_value(0, 4, &array_u_8).is_err());
        assert!(
            buf.write_value(0, 4, array_u_8, OpRef::from_raw(30))
                .is_err()
        );
    }

    #[test]
    fn test_ptr_info_factories() {
        let nonnull = PtrInfo::nonnull();
        assert!(nonnull.is_nonnull());
        assert!(!nonnull.is_virtual());

        let constant = PtrInfo::constant(GcRef(0x1000));
        assert!(constant.is_nonnull());
        assert!(constant.is_constant());

        let kc = PtrInfo::known_class(GcRef(0x2000), true);
        assert!(kc.is_nonnull());
        assert!(kc.get_known_class().is_some());
    }

    #[test]
    fn test_ptr_info_virtual_factories() {
        let descr: DescrRef = Arc::new(TestDescr);

        let virtual_obj = PtrInfo::virtual_obj(descr.clone(), Some(GcRef(0x3000)));
        assert!(virtual_obj.is_virtual());
        assert!(virtual_obj.is_nonnull());
        assert!(virtual_obj.get_descr().is_some());

        let virtual_arr = PtrInfo::virtual_array(descr.clone(), 5, false);
        assert!(virtual_arr.is_virtual());
        assert_eq!(virtual_arr.num_fields(), 5);

        let virtual_struct = PtrInfo::virtual_struct(descr);
        assert!(virtual_struct.is_virtual());
    }

    #[test]
    fn test_const_ptr_info_getlenbound_returns_none_at_base() {
        // The base `PtrInfo::getlenbound` returns None for `PtrInfo::Constant`
        // — the constant string-length lookup runs through
        // `EnsuredPtrInfo::Constant::getlenbound`, which threads in the
        // runtime `string_length_resolver`. Callers that bypass
        // EnsuredPtrInfo (and thus skip the resolver) must not get a
        // misleading nonnegative answer here.
        let mut info = PtrInfo::constant(GcRef(0x1000));

        assert_eq!(info.getlenbound(Some(0)), None);
        assert_eq!(info.getlenbound(Some(1)), None);
        assert_eq!(info.getlenbound(None), None);
    }

    #[test]
    fn test_str_ptr_info_virtual_variants() {
        let plain = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: 2,
            variant: VStringVariant::Plain(VStringPlainInfo {
                _chars: vec![None, None],
            }),
            last_guard_pos: -1,
            cached_vinfo: std::cell::RefCell::new(None),
        });
        assert!(plain.is_virtual());

        let slice = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: Some(OpRef::from_raw(3)), // vstring.py:223: self.lgtop = length
            mode: 0,
            length: -1,
            variant: VStringVariant::Slice(VStringSliceInfo {
                s: OpRef::from_raw(1),
                start: OpRef::from_raw(2),
                lgtop: OpRef::from_raw(3),
            }),
            last_guard_pos: -1,
            cached_vinfo: std::cell::RefCell::new(None),
        });
        assert!(slice.is_virtual());

        let concat = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: -1,
            variant: VStringVariant::Concat(VStringConcatInfo {
                vleft: OpRef::from_raw(4),
                vright: OpRef::from_raw(5),
                _is_virtual: true,
            }),
            last_guard_pos: -1,
            cached_vinfo: std::cell::RefCell::new(None),
        });
        assert!(concat.is_virtual());

        let ptr = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: -1,
            variant: VStringVariant::Ptr,
            last_guard_pos: -1,
            cached_vinfo: std::cell::RefCell::new(None),
        });
        assert!(!ptr.is_virtual());
    }

    #[test]
    fn test_str_ptr_info_constant_string_spec_and_strgetitem() {
        let mut ctx = OptContext::new(16);
        ctx.make_constant(OpRef::from_raw(10), Value::Int(97));
        ctx.make_constant(OpRef::from_raw(11), Value::Int(98));
        ctx.make_constant(OpRef::from_raw(12), Value::Int(99));

        let info = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: 3,
            variant: VStringVariant::Plain(VStringPlainInfo {
                _chars: vec![
                    Some(OpRef::from_raw(10)),
                    Some(OpRef::from_raw(11)),
                    Some(OpRef::from_raw(12)),
                ],
            }),
            last_guard_pos: -1,
            cached_vinfo: std::cell::RefCell::new(None),
        });

        assert_eq!(
            info.get_constant_string_spec(&ctx, 0),
            Some(vec![97, 98, 99])
        );
        assert_eq!(info.get_known_str_length(&ctx, 0), Some(3));
        assert_eq!(info.strgetitem(1, &ctx), Some(OpRef::from_raw(11)));
    }

    #[test]
    fn test_str_ptr_info_slice_and_concat_dispatch() {
        let mut ctx = OptContext::new(32);
        ctx.make_constant(OpRef::from_raw(10), Value::Int(97));
        ctx.make_constant(OpRef::from_raw(11), Value::Int(98));
        ctx.make_constant(OpRef::from_raw(12), Value::Int(99));
        ctx.make_constant(OpRef::from_raw(20), Value::Int(1));
        ctx.make_constant(OpRef::from_raw(21), Value::Int(2));

        let source = OpRef::from_raw(1);
        ctx.set_ptr_info(
            source,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: 3,
                variant: VStringVariant::Plain(VStringPlainInfo {
                    _chars: vec![
                        Some(OpRef::from_raw(10)),
                        Some(OpRef::from_raw(11)),
                        Some(OpRef::from_raw(12)),
                    ],
                }),
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );

        let slice = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: Some(OpRef::from_raw(21)), // vstring.py:223: self.lgtop = length
            mode: 0,
            length: -1,
            variant: VStringVariant::Slice(VStringSliceInfo {
                s: source,
                start: OpRef::from_raw(20),
                lgtop: OpRef::from_raw(21),
            }),
            last_guard_pos: -1,
            cached_vinfo: std::cell::RefCell::new(None),
        });
        assert_eq!(slice.get_known_str_length(&ctx, 0), Some(2));
        assert_eq!(slice.get_constant_string_spec(&ctx, 0), Some(vec![98, 99]));
        assert_eq!(slice.strgetitem(0, &ctx), Some(OpRef::from_raw(11)));

        let concat = PtrInfo::Str(StrPtrInfo {
            lenbound: None,
            lgtop: None,
            mode: 0,
            length: -1,
            variant: VStringVariant::Concat(VStringConcatInfo {
                vleft: source,
                vright: OpRef::from_raw(2),
                _is_virtual: true,
            }),
            last_guard_pos: -1,
            cached_vinfo: std::cell::RefCell::new(None),
        });
        ctx.set_ptr_info(
            OpRef::from_raw(2),
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: 2,
                variant: VStringVariant::Plain(VStringPlainInfo {
                    _chars: vec![Some(OpRef::from_raw(11)), Some(OpRef::from_raw(12))],
                }),
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );

        assert_eq!(concat.get_known_str_length(&ctx, 0), Some(5));
        assert_eq!(
            concat.get_constant_string_spec(&ctx, 0),
            Some(vec![97, 98, 99, 98, 99])
        );
        assert_eq!(concat.strgetitem(3, &ctx), Some(OpRef::from_raw(11)));
    }

    #[test]
    fn test_ptr_info_set_getfield() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::virtual_obj(descr, None);

        assert!(info.getfield(0).is_none());
        info.setfield(0, OpRef::from_raw(10));
        assert_eq!(
            info.getfield(0).and_then(|e| e.as_opref()),
            Some(OpRef::from_raw(10))
        );
        info.setfield(0, OpRef::from_raw(20)); // overwrite
        assert_eq!(
            info.getfield(0).and_then(|e| e.as_opref()),
            Some(OpRef::from_raw(20))
        );
        info.setfield(1, OpRef::from_raw(30));
        assert_eq!(
            info.getfield(1).and_then(|e| e.as_opref()),
            Some(OpRef::from_raw(30))
        );
    }

    #[test]
    fn test_ptr_info_set_getitem() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::virtual_array(descr, 3, false);

        assert_eq!(
            info.getitem(0).and_then(|e| e.as_opref()),
            Some(OpRef::NONE)
        ); // initialized to NONE
        info.setitem(0, OpRef::from_raw(10));
        assert_eq!(
            info.getitem(0).and_then(|e| e.as_opref()),
            Some(OpRef::from_raw(10))
        );
        info.setitem(2, OpRef::from_raw(30));
        assert_eq!(
            info.getitem(2).and_then(|e| e.as_opref()),
            Some(OpRef::from_raw(30))
        );
        assert!(info.getitem(5).is_none()); // out of bounds
    }

    #[test]
    fn test_preamble_item_keeps_regular_array_item_visible() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::array(descr, crate::optimizeopt::intutils::IntBound::nonnegative());
        info.setitem(1, OpRef::from_raw(77));
        assert_eq!(
            info.getitem(1).and_then(|e| e.as_opref()),
            Some(OpRef::from_raw(77))
        );

        let mut replay = Op::new(
            OpCode::GetarrayitemGcI,
            &[OpRef::from_raw(10), OpRef::from_const(0)],
        );
        replay.pos = OpRef::from_raw(88);
        let pop = PreambleOp {
            op: OpRef::from_raw(88),
            resolved: OpRef::from_raw(99),
            invented_name: false,
            preamble_op: replay,
        };
        info.set_preamble_item(1, pop.clone());

        assert!(info.has_preamble_item(1));
        // After set_preamble_item, getitem returns Preamble (not the old Value)
        assert!(info.getitem(1).map_or(false, |e| e.is_preamble()));
        let recovered = info
            .take_preamble_item(1)
            .expect("preamble item should be recoverable");
        assert_eq!(recovered.resolved, OpRef::from_raw(99));
        // After take_preamble_item, slot is Value(NONE)
        assert_eq!(
            info.getitem(1).and_then(|e| e.as_opref()),
            Some(OpRef::NONE)
        );
    }

    #[test]
    fn test_all_items_exposes_preamble_source_box() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::instance(Some(descr), None);
        let replay = Op::new(OpCode::GetfieldGcI, &[OpRef::from_raw(10)]);
        let pop = PreambleOp {
            op: OpRef::from_raw(88),
            resolved: OpRef::from_raw(99),
            invented_name: false,
            preamble_op: replay,
        };
        info.set_preamble_field(3, pop);

        // all_items includes Preamble entries (RPython parity: _fields returns raw)
        let items = info.all_items();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, 3);
        assert!(items[0].1.is_preamble());
    }

    #[test]
    fn test_ptr_info_guard_opcodes() {
        let nonnull = PtrInfo::nonnull();
        let guards = nonnull.guard_opcodes();
        assert!(guards.contains(&OpCode::GuardNonnull));

        let constant = PtrInfo::constant(GcRef(0x1000));
        let guards = constant.guard_opcodes();
        assert!(guards.contains(&OpCode::GuardValue));

        let kc = PtrInfo::known_class(GcRef(0x2000), true);
        let guards = kc.guard_opcodes();
        assert!(guards.contains(&OpCode::GuardNonnullClass));
    }

    #[test]
    fn test_ptr_info_visitor_walk() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::virtual_obj(descr, None);
        info.setfield(0, OpRef::from_raw(10));
        info.setfield(1, OpRef::from_raw(20));
        let refs = info.visitor_walk_recursive();
        assert_eq!(refs, vec![OpRef::from_raw(10), OpRef::from_raw(20)]);
    }

    #[test]
    fn test_opinfo_is_nonnull() {
        assert!(!OpInfo::Unknown.is_nonnull());
        assert!(OpInfo::Constant(Value::Int(42)).is_nonnull());
        assert!(!OpInfo::Constant(Value::Int(0)).is_nonnull());
        assert!(OpInfo::Ptr(PtrInfo::nonnull()).is_nonnull());
    }

    #[test]
    fn test_opinfo_float_const() {
        let info = OpInfo::FloatConst(3.14);
        assert!(info.is_constant());
        assert_eq!(info.get_constant_float(), Some(3.14));
    }
}
