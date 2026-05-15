/// Object tracing for GC reference discovery.
///
/// During collection, the GC needs to find all GC references within
/// a live object so it can update them (for copying collection) or
/// mark the targets (for mark-sweep).
///
/// Instead of a closure-based trace approach (which causes lifetime issues
/// with the borrow checker), we use an offset-based approach: each type
/// declares the offsets of its GC pointer fields relative to the object
/// payload start. The collector reads/writes GcRef values at these offsets
/// directly.
use majit_ir::GcRef;

/// One `gctypelayout.GCData.TYPE_INFO` entry of the materialized
/// type-info group (gc.py:592, x86/assembler.py:1924-1943).
///
/// Mirrors the shape of `TYPE_INFO` that `genop_guard_guard_is_object`
/// reads: `infobits` at offset 0 carries `T_IS_RPYTHON_INSTANCE`. The
/// rest of the struct is reserved so the size matches
/// `rffi.sizeof(GCData.TYPE_INFO) = 16` on 64-bit majit. (RPython's
/// 64-bit `TYPE_INFO` carries additional fields majit does not yet need
/// — `customdata`, `fixedsize`, `ofstoptrs` — totalling 32 bytes in the
/// C backend. majit only consults `infobits` from this struct and keeps
/// the other fields out of this layout; the remaining padding is still
/// present so the per-entry stride stays at a power of two.)
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct TypeInfoLayout {
    pub infobits: u64,
    /// Reserved for future TYPE_INFO fields (customdata / fixedsize /
    /// ofstoptrs in the RPython struct, gctypelayout.py:36-42). Kept
    /// as a raw word so `sizeof_ti` matches the layout the backend
    /// lowering expects.
    pub _reserved: u64,
}

/// One `rclass.CLASSTYPE` entry of the materialized type-info group.
///
/// RPython's `GcLLDescr_framework.add_vtable_after_typeinfo`
/// (gctypelayout.py:359-374) appends the vtable struct directly after
/// each `TYPE_INFO` entry. `genop_guard_guard_subclass`
/// (x86/assembler.py:1968-1969) then addresses `subclassrange_min` as
/// `base + (typeid << shift_by) + sizeof_ti + offset2` — i.e. "skip
/// past the TYPE_INFO to read the CLASSTYPE that follows it".
///
/// majit mirrors that layout: each registered type gets a `TypeEntry`
/// pair whose `TypeInfoLayout` is immediately followed by
/// `ClassTypeLayout`. `subclassrange_min` / `subclassrange_max` mirror
/// the equally-named `rclass.CLASSTYPE` fields that RPython populates
/// via preorder numbering (rtyper/normalizecalls.py:385-389).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct ClassTypeLayout {
    pub subclassrange_min: i64,
    pub subclassrange_max: i64,
}

/// Paired `(TYPE_INFO, CLASSTYPE)` entry in the type-info group.
///
/// Matches the two-struct pattern RPython's translator emits in the
/// type_info_group (see `add_vtable_after_typeinfo`
/// gctypelayout.py:359-374). The fields must stay in this order —
/// backends walk from `type_info` to `classtype` using
/// `offset += sizeof_ti`.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct TypeEntry {
    pub type_info: TypeInfoLayout,
    pub classtype: ClassTypeLayout,
}

impl TypeInfoLayout {
    /// `T_MEMBER_INDEX` (gctypelayout.py:191). Lowest 16 bits of
    /// `infobits` carry the group member index; majit doesn't use
    /// this field today but reserves the bits to keep the bit layout
    /// compatible with the RPython constants below.
    pub const T_MEMBER_INDEX: u64 = 0xffff;
    /// `T_IS_VARSIZE` (gctypelayout.py:192).
    pub const T_IS_VARSIZE: u64 = 0x010000;
    /// `T_HAS_GCPTR_IN_VARSIZE` (gctypelayout.py:193).
    pub const T_HAS_GCPTR_IN_VARSIZE: u64 = 0x020000;
    /// `T_IS_GCARRAY_OF_GCPTR` (gctypelayout.py:194).
    pub const T_IS_GCARRAY_OF_GCPTR: u64 = 0x040000;
    /// `T_IS_WEAKREF` (gctypelayout.py:195).
    pub const T_IS_WEAKREF: u64 = 0x080000;
    /// `T_IS_RPYTHON_INSTANCE` — the type is a subclass of OBJECT
    /// (gctypelayout.py:196). Stored in `infobits` and tested by
    /// `genop_guard_guard_is_object` via the byte mask returned by
    /// `gc_ll_descr.get_translated_info_for_guard_is_object`
    /// (gc.py:603-622).
    pub const T_IS_RPYTHON_INSTANCE: u64 = 0x100000;
    /// `T_HAS_CUSTOM_TRACE` (gctypelayout.py:197).
    pub const T_HAS_CUSTOM_TRACE: u64 = 0x200000;
    /// `T_HAS_OLDSTYLE_FINALIZER` (gctypelayout.py:198).
    pub const T_HAS_OLDSTYLE_FINALIZER: u64 = 0x400000;
    /// `T_HAS_GCPTR` (gctypelayout.py:199).
    pub const T_HAS_GCPTR: u64 = 0x1000000;
    /// `T_HAS_MEMORY_PRESSURE` (gctypelayout.py:200) — first field is
    /// memory pressure field.
    pub const T_HAS_MEMORY_PRESSURE: u64 = 0x2000000;
    /// `T_KEY_MASK` (gctypelayout.py:201) — bug detection mask.
    /// `_check_valid_type_info` asserts `infobits & T_KEY_MASK ==
    /// T_KEY_VALUE`.
    pub const T_KEY_MASK: u64 = 0xFC000000;
    /// `T_KEY_VALUE` (gctypelayout.py:202) — bug detection sentinel
    /// stored in every valid `TYPE_INFO.infobits`.
    pub const T_KEY_VALUE: u64 = 0x58000000;

    pub const INFOBITS_OFFSET: usize = 0;
    /// `rffi.sizeof(GCData.TYPE_INFO)` equivalent. Reported by
    /// `get_translated_info_for_typeinfo` as `sizeof_ti` so the
    /// backend `genop_guard_guard_subclass` formula
    /// `base + (typeid << shift_by) + sizeof_ti + offset2` lands on
    /// the `ClassTypeLayout` that follows this struct in memory.
    pub const SIZE_OF_TI: usize = std::mem::size_of::<TypeInfoLayout>();
}

impl ClassTypeLayout {
    pub const SUBCLASSRANGE_MIN_OFFSET: usize =
        std::mem::offset_of!(ClassTypeLayout, subclassrange_min);
    pub const SUBCLASSRANGE_MAX_OFFSET: usize =
        std::mem::offset_of!(ClassTypeLayout, subclassrange_max);
}

impl TypeEntry {
    /// Stride between consecutive materialized type entries. Used as
    /// the `<< shift_by` scale the backend applies to a small-integer
    /// typeid before adding it to the table base — see the
    /// `get_translated_info_for_typeinfo` docstring.
    pub const STRIDE: usize = std::mem::size_of::<TypeEntry>();
    /// `log2(STRIDE)`. The backend uses this as `shift_by`
    /// (x86/assembler.py:1934) so the formula
    /// `base + (typeid << shift_by) + offset` lands on the right
    /// entry. RPython's 64-bit port sets `shift_by = 0` because its
    /// typeid is already pre-scaled to a byte offset
    /// (`GROUP_MEMBER_OFFSET`, translator/c/src/llgroup.h:36); majit
    /// keeps typeid as a small integer index and encodes the scale
    /// here instead.
    pub const SHIFT_BY: u8 = {
        // Compile-time assertion that STRIDE is a power of two.
        assert!(Self::STRIDE.is_power_of_two());
        Self::STRIDE.trailing_zeros() as u8
    };

    fn from_type_info(info: &TypeInfo, index: u32) -> Self {
        TypeEntry {
            type_info: TypeInfoLayout {
                infobits: encode_type_shape(info, index),
                _reserved: 0,
            },
            classtype: ClassTypeLayout {
                subclassrange_min: info.subclassrange_min,
                subclassrange_max: info.subclassrange_max,
            },
        }
    }
}

/// `gctypelayout.encode_type_shape` parity (gctypelayout.py:237-296).
///
/// Builds the `infobits` word from the type's structural flags. The
/// resulting bit layout is:
///
/// ```text
///   bits  0..15 : T_MEMBER_INDEX (typeid index)
///   bit   16    : T_IS_VARSIZE
///   bit   17    : T_HAS_GCPTR_IN_VARSIZE
///   bit   18    : T_IS_GCARRAY_OF_GCPTR
///   bit   19    : T_IS_WEAKREF
///   bit   20    : T_IS_RPYTHON_INSTANCE
///   bit   21    : T_HAS_CUSTOM_TRACE
///   bit   22    : T_HAS_OLDSTYLE_FINALIZER
///   bit   24    : T_HAS_GCPTR
///   bit   25    : T_HAS_MEMORY_PRESSURE
///   bits 26..31 : T_KEY_VALUE  (T_KEY_MASK sanity check)
/// ```
///
/// majit's `TypeInfo` doesn't carry every concept RPython encodes
/// (no weakref or memory pressure surface), so the corresponding bits
/// stay zero — but the bits we DO have are mapped one-for-one against
/// the RPython source for line-by-line parity:
///
/// ```python
///   infobits = index
///   if len(offsets) > 0: infobits |= T_HAS_GCPTR
///   if has custom trace: infobits |= T_HAS_CUSTOM_TRACE / T_HAS_OLDSTYLE_FINALIZER
///   if varsize: infobits |= T_IS_VARSIZE
///   if varsize array of gc ptrs: infobits |= T_IS_GCARRAY_OF_GCPTR
///   if varsize has gc ptrs: infobits |= T_HAS_GCPTR_IN_VARSIZE | T_HAS_GCPTR
///   if weakref: infobits |= T_IS_WEAKREF
///   if subclass of OBJECT: infobits |= T_IS_RPYTHON_INSTANCE
///   info.infobits = infobits | T_KEY_VALUE
/// ```
pub fn encode_type_shape(info: &TypeInfo, index: u32) -> u64 {
    // gctypelayout.py:240 `infobits = index` — typeid in low 16 bits.
    let mut infobits: u64 = (index as u64) & TypeInfoLayout::T_MEMBER_INDEX;
    // gctypelayout.py:242-243: T_HAS_GCPTR if there are GC pointer
    // fields in the fixed part.
    if !info.gc_ptr_offsets.is_empty() {
        infobits |= TypeInfoLayout::T_HAS_GCPTR;
    }
    // gctypelayout.py:245-257 customdata / destructor / oldstyle
    // finalizer / memory pressure. majit only models the custom
    // tracer half of this; treat custom_trace.is_some() as the
    // T_HAS_CUSTOM_TRACE bit (info.py:178 `q_has_custom_trace`).
    if info.custom_trace.is_some() {
        infobits |= TypeInfoLayout::T_HAS_CUSTOM_TRACE;
        // jitframe.py registers JITFRAME via custom_trace and is also
        // T_HAS_GCPTR by virtue of holding GC refs the tracer
        // discovers; mark T_HAS_GCPTR so q_has_gcptr returns True for
        // it (gctypelayout.py:81-83).
        infobits |= TypeInfoLayout::T_HAS_GCPTR;
    }
    // gctypelayout.py:266-291 varsize encoding.
    if info.item_size > 0 {
        infobits |= TypeInfoLayout::T_IS_VARSIZE;
        if info.items_have_gc_ptrs {
            // gctypelayout.py:288-289: variable-size array carrying
            // GC pointers sets both flags.
            infobits |= TypeInfoLayout::T_HAS_GCPTR_IN_VARSIZE;
            infobits |= TypeInfoLayout::T_HAS_GCPTR;
            // gctypelayout.py:278-280: pure GcArray-of-GcPtr.
            if info.gc_ptr_offsets.is_empty() {
                infobits |= TypeInfoLayout::T_IS_GCARRAY_OF_GCPTR;
            }
        }
    }
    // gctypelayout.py:294-295 `is_subclass_of_object`.
    if info.is_object {
        infobits |= TypeInfoLayout::T_IS_RPYTHON_INSTANCE;
    }
    // gctypelayout.py:296 — T_KEY_VALUE sanity tag.
    infobits | TypeInfoLayout::T_KEY_VALUE
}

/// One element of a `TotalOrderSymbolic.orderwitness` list
/// (rtyper/normalizecalls.py:302-354). RPython mixes Python ints
/// (the `_unique_cdef_id` of each MRO entry) with the float
/// `MAX = 1E100` sentinel that's appended to the maxid witness, and
/// relies on tuple comparison treating any int as smaller than `MAX`.
///
/// majit replaces the implicit-Python type juggling with an explicit
/// enum: `Cdef(_) < Max` follows from the variant order under the
/// derived `Ord` impl, which line-by-line preserves RPython's
/// "real id sorts before MAX sentinel" invariant.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum WitnessElement {
    Cdef(u32),
    Max,
}

/// Registry mapping type IDs to their type descriptors and object sizes.
///
/// Mirrors RPython's `gctypelayout.TypeLayoutBuilder` and the surrounding
/// `type_info_group` machinery (gctypelayout.py:300-400). The translator
/// adds members during the "build types" phase and freezes the table
/// before any compiled code runs; majit reproduces the same lifecycle
/// with `register` (matches `gctypelayout.add_vtable_after_typeinfo`)
/// and `freeze_types` (matches the implicit translator-end freeze
/// captured by `can_add_new_types`).
pub struct TypeRegistry {
    /// `gctypelayout.id_of_type` parity — owns one `TypeInfo` per
    /// registered class. Indexed by typeid as a small integer (the
    /// majit-internal counterpart to `_unique_cdef_id`).
    entries: Vec<TypeInfo>,
    /// `gctypelayout.type_info_group` parity — the contiguous
    /// `(TYPE_INFO, CLASSTYPE)` member array the backend reads at
    /// codegen time via `get_translated_info_for_typeinfo`.
    ///
    /// Pre-allocated with `MAX_TYPES` capacity at construction so the
    /// backing storage never reallocates: every `register_type` call
    /// is in-place, and the base address is stable for the lifetime of
    /// the registry. RPython's translator places `type_info_group`
    /// at a fixed C address after translation; majit can't quite do
    /// that (types are registered at runtime) but the pre-allocated
    /// `Vec` capacity is the closest local equivalent.
    layout_table: Vec<TypeEntry>,
    /// `gctypelayout.can_add_new_types` parity. `true` until the
    /// frontend calls `freeze_types`; after that, `register_type`
    /// panics. Mirrors how RPython's translator stops accepting new
    /// types at the end of `make_type_info_group` /
    /// `encode_type_shapes_now` (gctypelayout.py:393-398).
    can_add_new_types: bool,
}

impl TypeRegistry {
    /// Maximum number of types the backing `layout_table` can hold
    /// without reallocating. Sized generously above the dozen-or-so
    /// types pyre currently registers; bumps require recompiling
    /// majit-gc.
    ///
    /// RPython's `type_info_group` is bounded by the half-word width
    /// of the inline `GROUP_MEMBER_OFFSET` (translator/c/src/llgroup.h)
    /// — 64KB on 32-bit and 4GB on 64-bit. majit's bound is set by
    /// the maximum number of distinct GC types we expect to register,
    /// not by an addressing limit.
    pub const MAX_TYPES: usize = 1024;
}

/// Custom trace function type.
///
/// RPython parity: `rgc.register_custom_trace_hook(TYPE, trace_fn)`.
/// When set, the GC calls this instead of the generic offset-based
/// tracing. Used for types with dynamic GC reference layouts
/// (e.g. JitFrame with gcmap bitmap).
///
/// `obj_addr` is the object payload start. The callback `f` must be
/// called for each GC reference slot address.
pub type CustomTraceFn = unsafe fn(obj_addr: usize, f: &mut dyn FnMut(*mut GcRef));

/// Information about a GC-managed type.
pub struct TypeInfo {
    /// Fixed size of the object (excluding header), or base size for varsize objects.
    pub size: usize,
    /// Whether this type contains GC pointers.
    pub has_gc_ptrs: bool,
    /// Byte offsets of GC pointer fields within the object payload (fixed part).
    pub gc_ptr_offsets: Vec<usize>,
    /// For variable-size objects: item size. 0 for fixed-size objects.
    pub item_size: usize,
    /// For variable-size objects: offset from object start to the length field.
    pub length_offset: usize,
    /// For variable-size objects: whether items contain GC pointers.
    /// If true, each item is treated as a GcRef.
    pub items_have_gc_ptrs: bool,
    /// RPython `rgc.register_custom_trace_hook` parity.
    /// When set, overrides offset-based tracing entirely.
    pub custom_trace: Option<CustomTraceFn>,
    /// gc.py:642 `T_IS_RPYTHON_INSTANCE` parity. True when this type has
    /// `rclass.OBJECT` layout — the first word of the payload is the
    /// `typeptr` (ob_type) and `cls_of_box(gcref)` is valid. False for
    /// raw GC structs / arrays whose first word is not a class pointer.
    /// Consumed by `check_is_object(gcref)` (llmodel.py:541-546) and by
    /// `genop_guard_guard_is_object` (x86/assembler.py:1924-1943) which
    /// reads it through the materialized `TYPE_INFO` table's `infobits`
    /// byte (gc.py:631-642).
    pub is_object: bool,
    /// Parent class typeid in the `rclass.OBJECT` hierarchy, or
    /// `None` for `rclass.OBJECT` itself / non-OBJECT types.
    /// Mirrors what `classdef.getmro()` traverses in
    /// `rtyper/normalizecalls.py:assign_inheritance_ids`. The parent
    /// chain feeds into `freeze_types`, which sorts the types
    /// lexicographically by reversed-MRO and assigns
    /// `subclassrange_{min,max}` from the resulting peer order
    /// (TotalOrderSymbolic, normalizecalls.py:302-354).
    pub parent: Option<u32>,
    /// `rclass.CLASSTYPE.subclassrange_min` for this type's class —
    /// i.e. the lowest preorder index of the class subtree rooted at
    /// this type. Populated by `freeze_types`. Used by
    /// `genop_guard_guard_subclass` (x86/assembler.py:1971-1978) and
    /// `execute_guard_subclass` (llgraph/runner.py:1271-1281).
    /// Encoded into the materialized `TYPE_INFO` table at
    /// `subclassrange_min_offset`.
    pub subclassrange_min: i64,
    /// `rclass.CLASSTYPE.subclassrange_max` for this type's class —
    /// the exclusive upper bound of the same subtree. Populated by
    /// `freeze_types`.
    pub subclassrange_max: i64,
}

impl TypeInfo {
    /// Create a type info for a fixed-size object with no GC pointers.
    pub fn simple(size: usize) -> Self {
        TypeInfo {
            size,
            has_gc_ptrs: false,
            gc_ptr_offsets: Vec::new(),
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            custom_trace: None,
            is_object: false,
            parent: None,
            subclassrange_min: 0,
            subclassrange_max: 0,
        }
    }

    /// Create a type info for a fixed-size `rclass.OBJECT`-layout
    /// instance (gc.py:642 `T_IS_RPYTHON_INSTANCE`). The first word of
    /// the payload is the `typeptr` (ob_type). `cls_of_box` and
    /// `guard_class` rely on this invariant, and
    /// `check_is_object(gcref)` returns true for such types.
    ///
    /// The class is treated as a leaf with no `rclass.OBJECT` parent
    /// in this registry. Use `object_subclass(size, parent_typeid)`
    /// to attach the type into an `rclass.OBJECT` inheritance chain
    /// before calling `freeze_types`.
    pub fn object(size: usize) -> Self {
        TypeInfo {
            size,
            has_gc_ptrs: false,
            gc_ptr_offsets: Vec::new(),
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            custom_trace: None,
            is_object: true,
            parent: None,
            subclassrange_min: 0,
            subclassrange_max: 0,
        }
    }

    /// Create a type info for an `rclass.OBJECT`-layout instance
    /// that inherits from a previously-registered class.
    ///
    /// Mirrors how RPython attaches `classdef.bases[]` to its
    /// inheritance graph; `freeze_types` later walks the parent chain
    /// to compute reversed-MRO witnesses
    /// (rtyper/normalizecalls.py:assign_inheritance_ids).
    pub fn object_subclass(size: usize, parent_typeid: u32) -> Self {
        TypeInfo {
            size,
            has_gc_ptrs: false,
            gc_ptr_offsets: Vec::new(),
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            custom_trace: None,
            is_object: true,
            parent: Some(parent_typeid),
            subclassrange_min: 0,
            subclassrange_max: 0,
        }
    }

    /// `object_subclass` variant for instances whose payload contains
    /// inline `GcRef`-typed fields. `offsets` lists their byte offsets
    /// within the struct (after `ob_type` at offset 0). The collector
    /// walks these during minor collection so the parent-chain
    /// inheritance information is preserved alongside the trace
    /// metadata.
    ///
    /// Use this when an `rclass.OBJECT`-shaped class has fields whose
    /// rtyped form is `Ptr(rclass.OBJECT)` — for example
    /// `pypy/interpreter/function.py:47-86 W_Function`'s `closure`,
    /// `defs_w`, `w_kw_defs`, `w_module`.
    pub fn object_subclass_with_gc_ptrs(
        size: usize,
        parent_typeid: u32,
        offsets: Vec<usize>,
    ) -> Self {
        let has_gc_ptrs = !offsets.is_empty();
        TypeInfo {
            size,
            has_gc_ptrs,
            gc_ptr_offsets: offsets,
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            custom_trace: None,
            is_object: true,
            parent: Some(parent_typeid),
            subclassrange_min: 0,
            subclassrange_max: 0,
        }
    }

    /// `object_subclass` variant for instances whose GC references are
    /// discovered by a custom trace hook rather than fixed inline offsets.
    pub fn object_subclass_with_custom_trace(
        size: usize,
        parent_typeid: u32,
        trace_fn: CustomTraceFn,
    ) -> Self {
        TypeInfo {
            size,
            has_gc_ptrs: true,
            gc_ptr_offsets: Vec::new(),
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            custom_trace: Some(trace_fn),
            is_object: true,
            parent: Some(parent_typeid),
            subclassrange_min: 0,
            subclassrange_max: 0,
        }
    }

    /// Create a type info for a fixed-size object with GC pointer fields.
    /// `offsets` lists byte offsets of GcRef fields within the payload.
    pub fn with_gc_ptrs(size: usize, offsets: Vec<usize>) -> Self {
        let has_gc_ptrs = !offsets.is_empty();
        TypeInfo {
            size,
            has_gc_ptrs,
            gc_ptr_offsets: offsets,
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            custom_trace: None,
            is_object: false,
            parent: None,
            subclassrange_min: 0,
            subclassrange_max: 0,
        }
    }

    /// Create a type info for a variable-size object.
    pub fn varsize(
        base_size: usize,
        item_size: usize,
        length_offset: usize,
        items_have_gc_ptrs: bool,
        gc_ptr_offsets: Vec<usize>,
    ) -> Self {
        TypeInfo {
            size: base_size,
            has_gc_ptrs: !gc_ptr_offsets.is_empty() || items_have_gc_ptrs,
            gc_ptr_offsets,
            item_size,
            length_offset,
            items_have_gc_ptrs,
            custom_trace: None,
            is_object: false,
            parent: None,
            subclassrange_min: 0,
            subclassrange_max: 0,
        }
    }

    /// Create a type info with a custom trace hook (fixed-size).
    ///
    /// RPython parity: `rgc.register_custom_trace_hook(TYPE, trace_fn)`.
    pub fn with_custom_trace(size: usize, trace_fn: CustomTraceFn) -> Self {
        TypeInfo {
            size,
            has_gc_ptrs: true,
            gc_ptr_offsets: Vec::new(),
            item_size: 0,
            length_offset: 0,
            items_have_gc_ptrs: false,
            custom_trace: Some(trace_fn),
            is_object: false,
            parent: None,
            subclassrange_min: 0,
            subclassrange_max: 0,
        }
    }

    /// Create a varsize type info with a custom trace hook.
    ///
    /// RPython parity: `rgc.register_custom_trace_hook` on a
    /// `GcStruct(..., Array(Signed))` — e.g. JITFRAME.
    pub fn varsize_with_custom_trace(
        base_size: usize,
        item_size: usize,
        length_offset: usize,
        trace_fn: CustomTraceFn,
    ) -> Self {
        TypeInfo {
            size: base_size,
            has_gc_ptrs: true,
            gc_ptr_offsets: Vec::new(),
            item_size,
            length_offset,
            items_have_gc_ptrs: false, // custom_trace handles ref tracing
            custom_trace: Some(trace_fn),
            is_object: false,
            parent: None,
            subclassrange_min: 0,
            subclassrange_max: 0,
        }
    }

    /// Compute the total size of an instance (excluding GC header).
    /// RPython lltypelayout.py:93-100 sizeof(TYPE, i):
    ///   fixedsize = get_fixed_size(TYPE)
    ///   varsize = get_variable_size(TYPE)
    ///   return fixedsize + i * varsize
    ///
    /// Both `get_fixed_size(lltype.Array)` and the `_size` field of a
    /// `get_layout(Struct-with-array)` already account for the length
    /// word, so no extra WORD is added here.
    pub fn total_instance_size(&self, length: usize) -> usize {
        self.size + self.item_size * length
    }

    /// Iterate all GC pointer slot addresses for a given object.
    ///
    /// # Safety
    /// `obj_addr` must point to valid memory of at least `self.size` bytes (plus
    /// variable part if applicable).
    pub unsafe fn for_each_gc_ptr(&self, obj_addr: usize, mut f: impl FnMut(*mut GcRef)) {
        // RPython custom_trace_hook parity: if a custom trace function
        // is registered, use it instead of generic offset-based tracing.
        if let Some(trace_fn) = self.custom_trace {
            unsafe { trace_fn(obj_addr, &mut f) };
            return;
        }

        // Fixed-part GC pointer fields.
        for &offset in &self.gc_ptr_offsets {
            f((obj_addr + offset) as *mut GcRef);
        }

        // Variable-part GC pointer items.
        if self.items_have_gc_ptrs && self.item_size > 0 {
            let length = unsafe { *((obj_addr + self.length_offset) as *const usize) };
            let items_start = obj_addr + self.size;
            for i in 0..length {
                f((items_start + i * self.item_size) as *mut GcRef);
            }
        }
    }
}

impl TypeRegistry {
    pub fn new() -> Self {
        TypeRegistry {
            entries: Vec::with_capacity(Self::MAX_TYPES),
            layout_table: Vec::with_capacity(Self::MAX_TYPES),
            can_add_new_types: true,
        }
    }

    /// Register a new type and return its type ID.
    ///
    /// `gctypelayout.add_vtable_after_typeinfo` parity (gctypelayout.
    /// py:359-374): pushes the `TypeInfo` and reserves a paired
    /// `TypeEntry` slot in the `type_info_group` array. Both the
    /// `entries` and `layout_table` vectors are pre-allocated to
    /// `MAX_TYPES`, so this never reallocates and the
    /// `type_info_table` base address is stable for the lifetime of
    /// the registry.
    ///
    /// `subclassrange_{min,max}` are intentionally left at 0 here;
    /// `freeze_types` walks the inheritance tree afterwards and
    /// assigns preorder bounds via `assign_inheritance_ids`
    /// (normalizecalls.py:373-389), then refreshes the materialized
    /// `TypeEntry` rows.
    pub fn register(&mut self, info: TypeInfo) -> u32 {
        assert!(
            self.can_add_new_types,
            "TypeRegistry::register called after freeze_types \
             (gctypelayout.can_add_new_types == False)"
        );
        let id = self.entries.len();
        assert!(
            id < Self::MAX_TYPES,
            "TypeRegistry exceeded MAX_TYPES = {} — bump trace.rs",
            Self::MAX_TYPES
        );
        let entry = TypeEntry::from_type_info(&info, id as u32);
        self.entries.push(info);
        self.layout_table.push(entry);
        id as u32
    }

    /// `gctypelayout.encode_type_shapes_now` parity
    /// (gctypelayout.py:393-398): freezes the registry so subsequent
    /// `register_type` calls panic and assigns each `is_object`
    /// type its preorder `subclassrange_{min,max}`.
    ///
    /// The bounds come from `rtyper/normalizecalls.py:373-389
    /// assign_inheritance_ids`: each class is sorted lexicographically
    /// by reversed MRO (witness), and its position in that sorted list
    /// becomes the preorder index. The exclusive upper bound is the
    /// position of the matching `[witness + MAX]` peer, which lies
    /// just past the class's last descendant. The resulting
    /// `subclassrange_{min,max}` satisfy
    /// `int_between(cls.min, subcls.min, cls.max)` from
    /// `rclass.py:1133-1137 ll_issubclass`.
    pub fn freeze_types(&mut self) {
        if !self.can_add_new_types {
            return;
        }
        self.can_add_new_types = false;
        self.assign_inheritance_ids();
        // Refresh layout_table rows for is_object types whose
        // subclassrange_{min,max} just changed.
        for (i, info) in self.entries.iter().enumerate() {
            self.layout_table[i] = TypeEntry::from_type_info(info, i as u32);
        }
    }

    /// `rtyper/normalizecalls.py:373-389 assign_inheritance_ids` /
    /// `TotalOrderSymbolic` parity (normalizecalls.py:302-354).
    ///
    /// For each `is_object` type, builds
    /// `witness = reversed(MRO of cdef ids)` and pairs it with a
    /// matching `witness + [MAX]` peer:
    ///
    /// ```python
    /// classdef.minid = TotalOrderSymbolic(witness, lst)
    /// classdef.maxid = TotalOrderSymbolic(witness + [MAX], lst)
    /// MAX = 1E100
    /// ```
    ///
    /// All peers are then sorted lexicographically; the position of
    /// each peer in the sorted list becomes its preorder value
    /// (`compute_fn` in normalizecalls.py:342-354). The result
    /// satisfies `int_between(cls.min, subcls.min, cls.max)` from
    /// `rclass.py:1133-1137 ll_issubclass`.
    ///
    /// RPython encodes the sentinel as the float `1E100` so that any
    /// real cdef-id (a Python int) sorts before it under tuple
    /// comparison. majit replicates this with the explicit
    /// `WitnessElement::Max` variant — `Cdef(_) < Max` falls out of
    /// the derived `Ord`, structurally matching the RPython peer
    /// list without relying on a magic numeric sentinel.
    fn assign_inheritance_ids(&mut self) {
        let n = self.entries.len();
        // Build reversed-MRO witness for each is_object type.
        // Witness for typeid T = [T_root, ..., T_grandparent, T_parent, T]
        // where T_root is the topmost rclass.OBJECT-layout ancestor.
        let mut witness: Vec<Option<Vec<WitnessElement>>> = vec![None; n];
        for id in 0..n {
            if !self.entries[id].is_object {
                continue;
            }
            let mut mro: Vec<WitnessElement> = Vec::new();
            let mut cur = Some(id as u32);
            while let Some(c) = cur {
                mro.push(WitnessElement::Cdef(c));
                cur = self.entries[c as usize].parent;
            }
            mro.reverse();
            witness[id] = Some(mro);
        }

        // Build the peer list: each is_object type contributes a
        // `(witness, owner)` Min peer and a `(witness + [Max], owner)`
        // Max peer. Sort lexicographically by witness; the peer's
        // position becomes its preorder value (TotalOrderSymbolic
        // .compute_fn, normalizecalls.py:342-354).
        #[derive(Clone)]
        struct Peer {
            witness: Vec<WitnessElement>,
            owner: u32,
            is_max: bool,
        }
        let mut peers: Vec<Peer> = Vec::new();
        for id in 0..n {
            let Some(w) = witness[id].as_ref() else {
                continue;
            };
            peers.push(Peer {
                witness: w.clone(),
                owner: id as u32,
                is_max: false,
            });
            let mut w_max = w.clone();
            w_max.push(WitnessElement::Max);
            peers.push(Peer {
                witness: w_max,
                owner: id as u32,
                is_max: true,
            });
        }
        peers.sort_by(|a, b| a.witness.cmp(&b.witness));

        // Assign preorder values. minid = index of the Min peer,
        // maxid = index of the Max peer. The exclusive bound matches
        // RPython's int_between (a <= b < c) semantics from
        // rclass.py:1133-1137.
        for (idx, peer) in peers.iter().enumerate() {
            let owner = peer.owner as usize;
            if peer.is_max {
                self.entries[owner].subclassrange_max = idx as i64;
            } else {
                self.entries[owner].subclassrange_min = idx as i64;
            }
        }
    }

    /// Whether the registry has been frozen
    /// (gctypelayout.can_add_new_types == False).
    pub fn is_frozen(&self) -> bool {
        !self.can_add_new_types
    }

    /// Look up type info by ID.
    pub fn get(&self, type_id: u32) -> &TypeInfo {
        &self.entries[type_id as usize]
    }

    /// Number of registered types.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the `type_info_group` array of paired
    /// `(TYPE_INFO, CLASSTYPE)` entries.
    ///
    /// Mirrors `llop.gc_get_type_info_group` (gc.py:585): RPython's
    /// translator emits a contiguous group at translation time whose
    /// members alternate between `TYPE_INFO` and `CLASSTYPE`
    /// (gctypelayout.py:359-374 `add_vtable_after_typeinfo`). majit's
    /// equivalent is the pre-allocated `layout_table` `Vec<TypeEntry>`
    /// maintained eagerly by `register_type` / `freeze_types`. Its
    /// `as_ptr()` is stable as long as `MAX_TYPES` is not exceeded,
    /// matching the post-translation immutability of
    /// `type_info_group` so that compiled guards can embed the base
    /// address.
    pub fn type_info_table(&self) -> &[TypeEntry] {
        &self.layout_table
    }
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_registry() {
        let mut reg = TypeRegistry::new();
        let id0 = reg.register(TypeInfo::simple(16));
        let id1 = reg.register(TypeInfo::simple(32));

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(reg.get(id0).size, 16);
        assert_eq!(reg.get(id1).size, 32);
        assert!(!reg.get(id0).has_gc_ptrs);
    }

    #[test]
    fn test_type_with_gc_ptrs() {
        let mut reg = TypeRegistry::new();
        let id = reg.register(TypeInfo::with_gc_ptrs(24, vec![0, 8, 16]));
        assert!(reg.get(id).has_gc_ptrs);
        assert_eq!(reg.get(id).gc_ptr_offsets.len(), 3);
    }

    #[test]
    fn test_varsize_type() {
        let info = TypeInfo::varsize(8, 8, 0, true, Vec::new());
        assert_eq!(info.total_instance_size(10), 88); // 8 + 8*10
        assert_eq!(info.total_instance_size(0), 8);
    }

    #[test]
    fn test_for_each_gc_ptr() {
        // Object with two GcRef fields at offsets 0 and 8.
        let info = TypeInfo::with_gc_ptrs(16, vec![0, 8]);

        let mut data = [0u8; 16];
        let obj_addr = data.as_mut_ptr() as usize;

        let mut visited = Vec::new();
        unsafe {
            info.for_each_gc_ptr(obj_addr, |ptr| {
                visited.push(ptr as usize);
            });
        }

        assert_eq!(visited.len(), 2);
        assert_eq!(visited[0], obj_addr);
        assert_eq!(visited[1], obj_addr + 8);
    }

    #[test]
    fn test_encode_type_shape_object_flag() {
        // gctypelayout.py:294-295: T_IS_RPYTHON_INSTANCE is set on
        // is_object types and stored in the byte that
        // _setup_guard_is_object pulls out — byte 2 of a
        // little-endian Signed for T_IS_RPYTHON_INSTANCE = 0x100000.
        let info = TypeInfo::object(16);
        let bits = encode_type_shape(&info, 5);
        // Lower 16 bits hold the index.
        assert_eq!(bits & TypeInfoLayout::T_MEMBER_INDEX, 5);
        // T_IS_RPYTHON_INSTANCE bit is set.
        assert!(bits & TypeInfoLayout::T_IS_RPYTHON_INSTANCE != 0);
        // T_KEY_VALUE sanity tag is set.
        assert_eq!(
            bits & TypeInfoLayout::T_KEY_MASK,
            TypeInfoLayout::T_KEY_VALUE
        );
        // Byte 2 of the little-endian word carries 0x10
        // (T_IS_RPYTHON_INSTANCE >> 16 = 0x10).
        let bytes = bits.to_le_bytes();
        assert_eq!(bytes[2] & 0x10, 0x10);
    }

    #[test]
    fn test_encode_type_shape_varsize_with_gcptrs() {
        // gctypelayout.py:266-291: varsize encoding sets T_IS_VARSIZE
        // and, when items carry GcRefs, both T_HAS_GCPTR_IN_VARSIZE
        // and T_HAS_GCPTR. A pure GcArray-of-GcPtr also sets
        // T_IS_GCARRAY_OF_GCPTR.
        let info = TypeInfo::varsize(8, 8, 0, true, Vec::new());
        let bits = encode_type_shape(&info, 0);
        assert!(bits & TypeInfoLayout::T_IS_VARSIZE != 0);
        assert!(bits & TypeInfoLayout::T_HAS_GCPTR_IN_VARSIZE != 0);
        assert!(bits & TypeInfoLayout::T_HAS_GCPTR != 0);
        assert!(bits & TypeInfoLayout::T_IS_GCARRAY_OF_GCPTR != 0);
    }

    #[test]
    fn test_assign_inheritance_ids_singleton_for_leaf() {
        // Pure leaf classes get singleton ranges so
        // ll_issubclass(self, self) holds while subclasses are
        // excluded. Mirrors the int_between(a, b, c) = a <= b < c
        // semantics.
        let mut reg = TypeRegistry::new();
        let a = reg.register(TypeInfo::object(16));
        let b = reg.register(TypeInfo::object(16));
        reg.freeze_types();
        let info_a = reg.get(a);
        let info_b = reg.get(b);
        assert!(info_a.subclassrange_min < info_a.subclassrange_max);
        assert!(info_b.subclassrange_min < info_b.subclassrange_max);
        // No overlap between distinct leaf classes.
        assert!(
            info_a.subclassrange_max <= info_b.subclassrange_min
                || info_b.subclassrange_max <= info_a.subclassrange_min
        );
    }

    #[test]
    fn test_assign_inheritance_ids_parent_contains_child() {
        // For a parent->child chain, the parent's preorder range
        // [min, max) must contain the child's range. This is what
        // makes ll_issubclass(child, parent) true and
        // ll_issubclass(parent, child) false.
        let mut reg = TypeRegistry::new();
        let parent = reg.register(TypeInfo::object(16));
        let child = reg.register(TypeInfo::object_subclass(16, parent));
        reg.freeze_types();
        let p = reg.get(parent);
        let c = reg.get(child);
        // child is inside parent's range.
        assert!(p.subclassrange_min <= c.subclassrange_min);
        assert!(c.subclassrange_min < p.subclassrange_max);
        // parent is NOT inside child's range (unless they're equal).
        assert!(
            p.subclassrange_min < c.subclassrange_min || p.subclassrange_max > c.subclassrange_max
        );
    }

    #[test]
    fn test_for_each_gc_ptr_varsize() {
        // Variable-size object: base_size=8 (length field at offset 0),
        // items are GcRef (item_size=8).
        let info = TypeInfo::varsize(8, 8, 0, true, Vec::new());

        // Layout: [length: usize][item0: GcRef][item1: GcRef][item2: GcRef]
        let mut data = [0u8; 32]; // 8 base + 3*8 items
        let obj_addr = data.as_mut_ptr() as usize;

        // Write length = 3.
        unsafe {
            *(obj_addr as *mut usize) = 3;
        }

        let mut visited = Vec::new();
        unsafe {
            info.for_each_gc_ptr(obj_addr, |ptr| {
                visited.push(ptr as usize);
            });
        }

        assert_eq!(visited.len(), 3);
        assert_eq!(visited[0], obj_addr + 8);
        assert_eq!(visited[1], obj_addr + 16);
        assert_eq!(visited[2], obj_addr + 24);
    }
}
