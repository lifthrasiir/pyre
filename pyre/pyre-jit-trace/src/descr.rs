//! Field descriptors for JIT IR operations.
//!
//! GetfieldGcI, GetfieldGcR, and SetfieldGc require a `DescrRef`
//! carrying field offset, size, and type information. This module
//! provides a concrete `PyreFieldDescr` implementing majit's
//! `FieldDescr` trait for pyre's `#[repr(C)]` object layout.

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Weak;

use majit_ir::{
    ArrayDescr, Descr, DescrRef, FieldDescr, JitCodeDescr, SizeDescr, SwitchDescr, Type,
};

// PRE-EXISTING-ADAPTATION: tag bits in the high nibble of the descr
// index discriminate Field/Array/Size descrs. RPython stores all descrs
// in `setup_descrs`'s flat `all_descrs` list (descr.py:25-47) and
// recovers the type via `isinstance` on the descr object. Pyre cannot
// downcast `Arc<dyn Descr>` to a specific concrete trait via type id,
// so the index itself encodes the discriminant.
//
// The Field tag is also load-bearing for `FieldIndexDescr` in
// `optimizeopt/virtualize.rs:1620-1654` — that synthetic descriptor
// reconstructs `offset`/`field_size`/`field_type`/`signed` from the
// packed bits. Replacing the tag with a flat counter is contingent on
// that synthetic descriptor being replaced with a real
// `Arc<dyn FieldDescr>` lookup.
const FIELD_DESCR_TAG: u32 = 0x1000_0000;
const ARRAY_DESCR_TAG: u32 = 0x2000_0000;
const SIZE_DESCR_TAG: u32 = 0x3000_0000;

fn type_bits(tp: Type) -> u32 {
    match tp {
        Type::Int => 0,
        Type::Ref => 1,
        Type::Float => 2,
        Type::Void => 3,
    }
}

fn stable_field_index(offset: usize, field_size: usize, field_type: Type, signed: bool) -> u32 {
    FIELD_DESCR_TAG
        | (((offset as u32) & 0x000f_ffff) << 4)
        | (((field_size as u32) & 0x7) << 1)
        | ((signed as u32) << 3)
        | type_bits(field_type)
}

fn stable_array_index(base_size: usize, item_size: usize, item_type: Type, signed: bool) -> u32 {
    ARRAY_DESCR_TAG
        | (((base_size as u32) & 0x0000_0fff) << 12)
        | (((item_size as u32) & 0x0000_00ff) << 4)
        | ((signed as u32) << 3)
        | type_bits(item_type)
}

/// Concrete field descriptor for pyre object fields.
/// RPython FieldDescr: describes a field in a GC/raw struct.
#[derive(Debug)]
pub struct PyreFieldDescr {
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
    /// RPython: is_immutable_field(). Immutable fields survive cache invalidation.
    immutable: bool,
    /// RPython: _is_quasi_immutable(). Fields that rarely change but CAN change.
    /// When read during tracing, emits QUASIIMMUT_FIELD + GUARD_NOT_INVALIDATED.
    /// If mutated at runtime, invalidates all compiled loops watching this field.
    quasi_immutable: bool,
    /// RPython descr.py:227 — field name for heaptracker.py:66 filtering.
    name: &'static str,
    index_in_parent: usize,
    parent_descr: Option<Weak<dyn Descr>>,
}

/// Concrete array descriptor for pointer-backed runtime arrays.
#[derive(Debug)]
pub struct PyreArrayDescr {
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    signed: bool,
    /// `descr.py:359-362 ArrayDescr.lendescr` parity. `None` for the
    /// `nolength=True` shape (ints/floats backing arrays where items
    /// start at offset 0); `Some` for length-prefixed `Ptr(GcArray(T))`
    /// shapes whose `gen_initialize_len` (`rewrite.py:565,572`) writes
    /// the runtime length so `ArraylenGc` / `bhimpl_arraylen_vable`
    /// can read it back.
    len_descr: Option<DescrRef>,
}

impl Descr for PyreFieldDescr {
    fn index(&self) -> u32 {
        stable_field_index(self.offset, self.field_size, self.field_type, self.signed)
    }

    fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
        Some(self)
    }

    /// PyPy FieldDescr.is_always_pure(): immutable fields survive cache invalidation.
    fn is_always_pure(&self) -> bool {
        self.immutable
    }

    fn is_quasi_immutable(&self) -> bool {
        self.quasi_immutable
    }
}

impl Descr for PyreArrayDescr {
    fn index(&self) -> u32 {
        stable_array_index(self.base_size, self.item_size, self.item_type, self.signed)
    }

    fn as_array_descr(&self) -> Option<&dyn ArrayDescr> {
        Some(self)
    }
}

impl FieldDescr for PyreFieldDescr {
    fn offset(&self) -> usize {
        self.offset
    }
    fn field_size(&self) -> usize {
        self.field_size
    }
    fn field_type(&self) -> Type {
        self.field_type
    }
    fn is_field_signed(&self) -> bool {
        self.signed
    }
    fn field_name(&self) -> &str {
        self.name
    }
    fn index_in_parent(&self) -> usize {
        self.index_in_parent
    }
    fn get_parent_descr(&self) -> Option<DescrRef> {
        self.parent_descr
            .as_ref()
            .and_then(|parent| parent.upgrade())
    }
}

impl ArrayDescr for PyreArrayDescr {
    fn base_size(&self) -> usize {
        self.base_size
    }

    fn item_size(&self) -> usize {
        self.item_size
    }

    fn type_id(&self) -> u32 {
        self.type_id
    }

    fn item_type(&self) -> Type {
        self.item_type
    }

    fn is_item_signed(&self) -> bool {
        self.signed
    }

    fn len_descr(&self) -> Option<&dyn FieldDescr> {
        self.len_descr.as_ref().and_then(|d| d.as_field_descr())
    }
}

/// Create a field descriptor for an object field.
pub fn make_field_descr(
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
) -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset,
        field_size,
        field_type,
        signed,
        immutable: false,
        quasi_immutable: false,
        name: "",
        index_in_parent: 0,
        parent_descr: None,
    })
}

/// Create a field descr with an explicit parent SizeDescr.
///
/// RPython parity: `fielddescr.get_parent_descr()` returns the owning
/// struct's SizeDescr, enabling `info.py:180 init_fields(parent_descr,
/// index)`. Without parent_descr, `descr_index()` falls back to
/// `stable_field_index` (a hash) instead of `index_in_parent` (a small
/// sequential index), causing OOM in `ensure_field_descr_slot`.
///
/// The `index_in_parent` is computed by scanning the parent SizeDescr's
/// `all_fielddescrs` for a matching offset.
pub fn make_field_descr_with_parent(
    parent: DescrRef,
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
) -> DescrRef {
    // Derive index_in_parent from the parent SizeDescr's field list.
    let index = parent
        .as_size_descr()
        .and_then(|sd| {
            sd.all_fielddescrs()
                .iter()
                .enumerate()
                .find(|(_, fd)| fd.as_field_descr().map_or(false, |f| f.offset() == offset))
                .map(|(i, _)| i)
        })
        .unwrap_or_else(|| {
            panic!("FieldDescr offset {offset} is not present in parent SizeDescr all_fielddescrs")
        });
    Arc::new(PyreFieldDescr {
        offset,
        field_size,
        field_type,
        signed,
        immutable: false,
        quasi_immutable: false,
        name: "",
        index_in_parent: index,
        parent_descr: Some(Arc::downgrade(&parent)),
    })
}

pub fn make_field_descr_full(
    _index: u32,
    offset: usize,
    field_size: usize,
    field_type: Type,
    immutable: bool,
) -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset,
        field_size,
        field_type,
        signed: false,
        immutable,
        quasi_immutable: false,
        name: "",
        index_in_parent: 0,
        parent_descr: None,
    })
}

/// Create a field descriptor for an immutable field (RPython is_immutable_field).
/// Cache entries for immutable fields survive call invalidation.
pub fn make_immutable_field_descr(
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
) -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset,
        field_size,
        field_type,
        signed,
        immutable: true,
        quasi_immutable: false,
        name: "",
        index_in_parent: 0,
        parent_descr: None,
    })
}

/// Create a field descriptor for a quasi-immutable field.
/// When read during tracing, emits QUASIIMMUT_FIELD + GUARD_NOT_INVALIDATED.
pub fn make_quasi_immutable_field_descr(
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
) -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset,
        field_size,
        field_type,
        signed,
        immutable: false,
        quasi_immutable: true,
        name: "",
        index_in_parent: 0,
        parent_descr: None,
    })
}

/// Concrete size descriptor for fixed-size object allocations.
#[derive(Debug)]
pub struct PyreSizeDescr {
    obj_size: usize,
    type_id: u32,
    /// descr.get_vtable() parity: ob_type pointer for NewWithVtable.
    /// optimize_new_with_vtable reads this to set VirtualInfo.known_class.
    vtable: usize,
    /// descr.py:72 `self.all_fielddescrs = all_fielddescrs`.
    all_fielddescrs: Vec<Arc<dyn FieldDescr>>,
    /// descr.py:71 `self.gc_fielddescrs = gc_fielddescrs` — precomputed
    /// subset of `all_fielddescrs` via `is_pointer_field()`
    /// (heaptracker.py:94-95 + :70 filter).
    gc_fielddescrs: Vec<Arc<dyn FieldDescr>>,
}

struct PyreObjectDescrGroup {
    size_descr: Arc<PyreSizeDescr>,
}

/// GC type id for the `rclass.OBJECT` root — pyre's static `INSTANCE_TYPE`
/// PyType (`name = "object"`). All `PyObject`-layout subclasses chain
/// their `parent` field to this id so `assign_inheritance_ids`
/// (normalizecalls.py:373-389) emits a `subclassrange_{min,max}` covering
/// every descendant. `GUARD_SUBCLASS(obj, &INSTANCE_TYPE)` then succeeds
/// for any `is_object` instance via `int_between(root.min, obj_typeid.min,
/// root.max)` (rclass.py:1133-1137 `ll_issubclass`).
pub const OBJECT_GC_TYPE_ID: u32 = 0;
// `W_INT_GC_TYPE_ID` / `W_FLOAT_GC_TYPE_ID` live in `pyre-object`
// alongside the `W_IntObject` / `W_FloatObject` structs they describe,
// so `pyre-object`'s host-side allocators can reach them without a
// back-channel. Re-exported here for existing call sites.
pub use pyre_object::floatobject::W_FLOAT_GC_TYPE_ID;
pub use pyre_object::intobject::W_INT_GC_TYPE_ID;
/// GC type id for JitFrame (jitframe.py:49 register_custom_trace_hook).
pub const JITFRAME_GC_TYPE_ID: u32 = 3;
/// GC type id for JitVirtualRef (virtualref.py — JIT_VIRTUAL_REF).
pub const VREF_GC_TYPE_ID: u32 = 4;
/// GC type id for W_BoolObject. `bool` inherits from `int` per
/// `objectobject.py W_BoolObject.typedef`, so this chains to
/// `W_INT_GC_TYPE_ID` as its parent via `TypeInfo::object_subclass`
/// (heaptracker.py:23-30 setup_cache_gcstruct2vtable — one typeid per
/// distinct STRUCT, not per root layout).
pub const W_BOOL_GC_TYPE_ID: u32 = 5;
/// GC type id for W_RangeIterator. Inherits from `object`
/// (rangeobject.rs:10 RANGE_ITER_TYPE).
pub const RANGE_ITER_GC_TYPE_ID: u32 = 6;
// `W_LIST_GC_TYPE_ID` / `W_TUPLE_GC_TYPE_ID` live in `pyre-object`
// alongside their structs (matching W_INT/W_FLOAT pattern); re-exported
// here for existing call sites.
pub use pyre_object::listobject::W_LIST_GC_TYPE_ID;
/// GC type id for the variable-length backing block of `PyObjectArray`
/// (the list/tuple items storage). Shape matches `rlist.py:84,116`
/// `GcArray(OBJECTPTR)` — a `T_IS_VARSIZE` block with an 8-byte
/// single-slot `capacity` header (= upstream's GcArray length header,
/// rlist.py:251 `len(l.items)`) followed by inline `PyObjectRef`
/// items. Registered via `TypeInfo::varsize(8, 8, 0,
/// items_have_gc_ptrs=true, [])` so the GC walks each item slot as a
/// Ref (`gctypelayout.py:266-291 T_IS_VARSIZE / T_IS_GCARRAY_OF_GCPTR`);
/// live list length is stored on the enclosing `W_ListObject` wrapper
/// (`PyObjectArray.len`) to match rlist.py:116 `("length", Signed)`.
///
// `PY_OBJECT_ARRAY_GC_TYPE_ID` lives in `pyre-object` alongside the
// `ItemsBlock` struct it describes (matching W_INT/W_FLOAT/W_LIST/
// W_TUPLE pattern). Re-exported here for existing call sites.
pub use pyre_object::object_array::PY_OBJECT_ARRAY_GC_TYPE_ID;
pub use pyre_object::tupleobject::W_TUPLE_GC_TYPE_ID;
// GC type ids for `W_SpecialisedTupleObject_{ii,ff,oo}` live in
// `pyre-object` alongside the structs they describe; re-exported here
// for existing call sites. See
// `pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_*_GC_TYPE_ID`.
pub use pyre_object::specialisedtupleobject::{
    SPECIALISED_TUPLE_FF_GC_TYPE_ID, SPECIALISED_TUPLE_II_GC_TYPE_ID,
    SPECIALISED_TUPLE_OO_GC_TYPE_ID,
};
// `BUILTIN_CODE_GC_TYPE_ID` lives in `pyre-interpreter::gateway`
// alongside the `BuiltinCode` struct it describes. `FUNCTION_GC_TYPE_ID`
// lives in `pyre-interpreter::function` for the same reason and covers
// `Function`, `BuiltinFunction`, and `FunctionWithFixedCode` (the
// latter two are Rust type aliases of `Function`). Re-exported here
// for the JIT registration site (`pyre-jit/src/eval.rs`).
pub use pyre_interpreter::function::FUNCTION_GC_TYPE_ID;
pub use pyre_interpreter::gateway::BUILTIN_CODE_GC_TYPE_ID;
// `W_CELL_GC_TYPE_ID` lives in `pyre-object::cellobject` alongside the
// `W_CellObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::cellobject::W_CELL_GC_TYPE_ID;
// `W_METHOD_GC_TYPE_ID` lives in `pyre-object::methodobject` alongside
// the `W_MethodObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::methodobject::W_METHOD_GC_TYPE_ID;
// `W_SLICE_GC_TYPE_ID` lives in `pyre-object::sliceobject` alongside
// the `W_SliceObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::sliceobject::W_SLICE_GC_TYPE_ID;
// `W_SUPER_GC_TYPE_ID` lives in `pyre-object::superobject` alongside
// the `W_SuperObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::superobject::W_SUPER_GC_TYPE_ID;
// `W_PROPERTY_GC_TYPE_ID` / `W_STATICMETHOD_GC_TYPE_ID` /
// `W_CLASSMETHOD_GC_TYPE_ID` live in `pyre-object::propertyobject`
// alongside their structs. Re-exported for the JIT registration site.
pub use pyre_object::propertyobject::{
    W_CLASSMETHOD_GC_TYPE_ID, W_PROPERTY_GC_TYPE_ID, W_STATICMETHOD_GC_TYPE_ID,
};
// `W_UNION_GC_TYPE_ID` lives in `pyre-object::unionobject` alongside
// the `W_UnionType` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::unionobject::W_UNION_GC_TYPE_ID;
// `W_SEQ_ITER_GC_TYPE_ID` lives in `pyre-object::rangeobject`
// alongside the `W_SeqIterator` struct it describes. Re-exported for
// the JIT registration site.
pub use pyre_object::rangeobject::W_SEQ_ITER_GC_TYPE_ID;
// `W_COUNT_GC_TYPE_ID` / `W_REPEAT_GC_TYPE_ID` live in
// `pyre-object::itertoolsmodule` alongside the `W_Count` /
// `W_Repeat` structs they describe. Re-exported for the JIT
// registration site.
pub use pyre_object::itertoolsmodule::{W_COUNT_GC_TYPE_ID, W_REPEAT_GC_TYPE_ID};
// `W_MEMBER_GC_TYPE_ID` lives in `pyre-object::memberobject`
// alongside the `W_MemberDescr` struct it describes. Re-exported for
// the JIT registration site.
pub use pyre_object::memberobject::W_MEMBER_GC_TYPE_ID;
// `W_BYTES_GC_TYPE_ID` lives in `pyre-object::bytesobject` alongside
// the `W_BytesObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::bytesobject::W_BYTES_GC_TYPE_ID;
// `W_BYTEARRAY_GC_TYPE_ID` lives in `pyre-object::bytearrayobject`
// alongside the `W_BytearrayObject` struct it describes. Re-exported
// for the JIT registration site.
pub use pyre_object::bytearrayobject::W_BYTEARRAY_GC_TYPE_ID;
// `W_DICT_GC_TYPE_ID` lives in `pyre-object::dictobject` alongside
// the `W_DictObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::dictobject::W_DICT_GC_TYPE_ID;
// `W_SET_GC_TYPE_ID` lives in `pyre-object::setobject` alongside the
// `W_SetObject` struct it describes (covers both `set` and
// `frozenset` PyTypes — same Rust struct). Re-exported for the JIT
// registration site.
pub use pyre_object::setobject::W_SET_GC_TYPE_ID;
// `W_EXCEPTION_GC_TYPE_ID` lives in `pyre-object::excobject`
// alongside the `W_ExceptionObject` struct it describes. Re-exported
// for the JIT registration site.
pub use pyre_object::excobject::W_EXCEPTION_GC_TYPE_ID;
// `W_GENERATOR_GC_TYPE_ID` lives in `pyre-object::generatorobject`
// alongside the `W_GeneratorObject` struct it describes. Re-exported
// for the JIT registration site.
pub use pyre_object::generatorobject::W_GENERATOR_GC_TYPE_ID;
// `W_TYPE_GC_TYPE_ID` lives in `pyre-object::typeobject` alongside
// the `W_TypeObject` struct it describes. Re-exported for the JIT
// registration site. (`TYPE_TYPE` is in `all_foreign_pytypes()` but
// the foreign-pytype loop's `sizeof(PyObject)` approximation would
// drastically under-count the W_TypeObject payload.)
pub use pyre_object::typeobject::W_TYPE_GC_TYPE_ID;
// `W_STR_GC_TYPE_ID` / `W_LONG_GC_TYPE_ID` / `W_MODULE_GC_TYPE_ID`
// live alongside their structs in
// `pyre-object::{strobject, longobject, moduleobject}`. Re-exported
// for the JIT registration site. `W_InstanceObject` shares
// `OBJECT_GC_TYPE_ID` with the `object` root (see comment on the
// struct) so it has no separate id.
pub use pyre_object::longobject::W_LONG_GC_TYPE_ID;
pub use pyre_object::moduleobject::W_MODULE_GC_TYPE_ID;
pub use pyre_object::strobject::W_STR_GC_TYPE_ID;
// `PYFRAME_GC_TYPE_ID` lives in `pyre-interpreter::pyframe` alongside
// the `PyFrame` struct it describes. Re-exported for the JIT
// registration site (`pyre-jit/src/eval.rs`). Phase 2.3 옵션 B
// foundation — registered ahead of any future
// `NewWithVtable(PyFrame)` in trace IR.
pub use pyre_interpreter::pyframe::PYFRAME_GC_TYPE_ID;

fn field_descr_from_group(group: &PyreObjectDescrGroup, index: usize) -> DescrRef {
    let field_descr = group
        .size_descr
        .all_fielddescrs
        .get(index)
        .expect("field descriptor index out of bounds")
        .clone();
    field_descr
}

fn build_object_descr_group(
    obj_size: usize,
    type_id: u32,
    vtable: usize,
    fields: &[(&'static str, usize, usize, Type, bool, bool, bool)],
) -> PyreObjectDescrGroup {
    let size_descr = Arc::new_cyclic(|weak_size: &Weak<PyreSizeDescr>| {
        let parent_descr: Weak<dyn Descr> = weak_size.clone();
        let all_fielddescrs: Vec<Arc<dyn FieldDescr>> = fields
            .iter()
            .enumerate()
            .map(
                |(
                    index_in_parent,
                    &(name, offset, field_size, field_type, signed, immutable, quasi_immutable),
                )| {
                    Arc::new(PyreFieldDescr {
                        offset,
                        field_size,
                        field_type,
                        signed,
                        immutable,
                        quasi_immutable,
                        name,
                        index_in_parent,
                        parent_descr: Some(parent_descr.clone()),
                    }) as Arc<dyn FieldDescr>
                },
            )
            .collect();
        // descr.py:123-126 precompute both lists; `gc_fielddescrs` is
        // `all_fielddescrs(only_gc=True)` per heaptracker.py:94-95.
        let gc_fielddescrs: Vec<Arc<dyn FieldDescr>> = all_fielddescrs
            .iter()
            .filter(|fd| fd.is_pointer_field())
            .cloned()
            .collect();
        PyreSizeDescr {
            obj_size,
            type_id,
            vtable,
            all_fielddescrs,
            gc_fielddescrs,
        }
    });
    PyreObjectDescrGroup { size_descr }
}

static W_INT_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group(
        std::mem::size_of::<W_IntObject>(),
        W_INT_GC_TYPE_ID,
        &INT_TYPE as *const _ as usize,
        &[(
            "W_IntObject.intval",
            INT_INTVAL_OFFSET,
            8,
            Type::Int,
            true,
            true,
            false,
        )],
    )
});

static W_FLOAT_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group(
        std::mem::size_of::<W_FloatObject>(),
        W_FLOAT_GC_TYPE_ID,
        &FLOAT_TYPE as *const _ as usize,
        &[(
            "W_FloatObject.floatval",
            FLOAT_FLOATVAL_OFFSET,
            8,
            Type::Float,
            false,
            true,
            false,
        )],
    )
});

static W_BOOL_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group(
        std::mem::size_of::<pyre_object::boolobject::W_BoolObject>(),
        W_BOOL_GC_TYPE_ID,
        &pyre_object::pyobject::BOOL_TYPE as *const _ as usize,
        &[(
            "W_BoolObject.boolval",
            BOOL_BOOLVAL_OFFSET,
            1,
            Type::Int,
            false,
            true,
            false,
        )],
    )
});

static RANGE_ITER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group(
        std::mem::size_of::<pyre_object::rangeobject::W_RangeIterator>(),
        RANGE_ITER_GC_TYPE_ID,
        &pyre_object::rangeobject::RANGE_ITER_TYPE as *const _ as usize,
        &[
            (
                "W_RangeIterator.current",
                RANGE_ITER_CURRENT_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "W_RangeIterator.stop",
                RANGE_ITER_STOP_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "W_RangeIterator.step",
                RANGE_ITER_STEP_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
        ],
    )
});

static W_LIST_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    // Upstream `rpython/rtyper/lltypesystem/rlist.py:116`
    //     GcStruct("list", ("length", Signed), ("items", Ptr(ITEMARRAY)))
    // The parity-field pair is `(length, items)`. `strategy` +
    // `int_items` / `float_items` are pyre-only PRE-EXISTING-
    // ADAPTATIONs for the PyPy interp-level strategy split.
    build_object_descr_group(
        std::mem::size_of::<W_ListObject>(),
        W_LIST_GC_TYPE_ID,
        &pyre_object::pyobject::LIST_TYPE as *const _ as usize,
        &[
            // rlist.py:116 `("length", Signed)`. Mutable: Object-strategy
            // push/pop/insert/remove/drain update it.
            (
                "W_ListObject.length",
                std::mem::offset_of!(W_ListObject, length),
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            // rlist.py:116 `("items", Ptr(GcArray(OBJECTPTR)))`. Points
            // at the `ItemsBlock` GcArray body. Mutable: re-pointed when
            // the Object-strategy storage is reallocated
            // (`list.object_grow` → `grow_list_items_block`) or when the
            // strategy switches.
            (
                "W_ListObject.items",
                std::mem::offset_of!(W_ListObject, items),
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                // `W_ListObject.strategy` is MUTABLE: `switch_to_object_strategy`
                // flips it from Integer/Float to Object when an
                // incompatible item is stored. A trace that folded
                // `strategy == Float` at trace-time into a constant would
                // then read from `float_items.ptr` (empty after the
                // switch) and dereference garbage — spectral_norm n=10
                // SIGSEGV root cause diagnosed in
                // memory/spectral_norm_small_n_crash_2026_04_17.md.
                //
                // Upstream PyPy handles this with a quasi-immutable flag
                // + invalidate_compiled_code hook on strategy change;
                // pyre has no such hook yet, so `strategy` stays
                // plain-mutable. NEW-DEVIATION — strategy split itself
                // is a pyre-only adaptation vs rlist.py.
                "W_ListObject.strategy",
                std::mem::offset_of!(W_ListObject, strategy),
                1,
                Type::Int,
                false,
                false,
                false,
            ),
            // Integer-strategy typed storage (pyre-only
            // PRE-EXISTING-ADAPTATION vs listobject.py's
            // IntegerListStrategy at the interp level — upstream keeps
            // the unwrap inline and doesn't add a separate backing
            // array).
            (
                "W_ListObject.int_items.ptr",
                std::mem::offset_of!(W_ListObject, int_items) + INT_ARRAY_PTR_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "W_ListObject.int_items.len",
                std::mem::offset_of!(W_ListObject, int_items) + INT_ARRAY_LEN_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "W_ListObject.int_items.heap_cap",
                std::mem::offset_of!(W_ListObject, int_items) + INT_ARRAY_HEAP_CAP_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            // Float-strategy typed storage.
            (
                "W_ListObject.float_items.ptr",
                std::mem::offset_of!(W_ListObject, float_items) + FLOAT_ARRAY_PTR_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "W_ListObject.float_items.len",
                std::mem::offset_of!(W_ListObject, float_items) + FLOAT_ARRAY_LEN_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "W_ListObject.float_items.heap_cap",
                std::mem::offset_of!(W_ListObject, float_items) + FLOAT_ARRAY_HEAP_CAP_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
        ],
    )
});

static W_TUPLE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    // `pypy/objspace/std/tupleobject.py:376-390` `W_TupleObject` stores
    // `wrappeditems: list` with `_immutable_fields_ =
    // ['wrappeditems[*]']`. After translation this becomes
    // `Ptr(GcArray(OBJECTPTR))`; `wrappeditems[*]` flows into both
    // the field descr (`immutable: true`) AND the GcArray contents
    // (read via `getfield_gc_pure_r`). Length comes from the GcArray
    // header via `arraylen_gc(items_block)` — no inline length cache.
    build_object_descr_group(
        std::mem::size_of::<W_TupleObject>(),
        W_TUPLE_GC_TYPE_ID,
        &pyre_object::pyobject::TUPLE_TYPE as *const _ as usize,
        &[
            // `Ptr(GcArray(OBJECTPTR))` — wrappeditems body. Immutable.
            (
                "W_TupleObject.wrappeditems",
                std::mem::offset_of!(W_TupleObject, wrappeditems),
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
        ],
    )
});

static SPECIALISED_TUPLE_II_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    // `specialisedtupleobject.py:34` `_immutable_fields_ = ['value0',
    // 'value1']` — both fields immutable. Inline-field shape, no array
    // indirection.
    use pyre_object::specialisedtupleobject::*;
    build_object_descr_group(
        std::mem::size_of::<W_SpecialisedTupleObject_ii>(),
        SPECIALISED_TUPLE_II_GC_TYPE_ID,
        &SPECIALISED_TUPLE_II_TYPE as *const _ as usize,
        &[
            (
                "W_SpecialisedTupleObject_ii.value0",
                SPECIALISED_TUPLE_II_VALUE0_OFFSET,
                8,
                Type::Int,
                true,
                true,
                false,
            ),
            (
                "W_SpecialisedTupleObject_ii.value1",
                SPECIALISED_TUPLE_II_VALUE1_OFFSET,
                8,
                Type::Int,
                true,
                true,
                false,
            ),
        ],
    )
});

static SPECIALISED_TUPLE_FF_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::specialisedtupleobject::*;
    build_object_descr_group(
        std::mem::size_of::<W_SpecialisedTupleObject_ff>(),
        SPECIALISED_TUPLE_FF_GC_TYPE_ID,
        &SPECIALISED_TUPLE_FF_TYPE as *const _ as usize,
        &[
            (
                "W_SpecialisedTupleObject_ff.value0",
                SPECIALISED_TUPLE_FF_VALUE0_OFFSET,
                8,
                Type::Float,
                false,
                true,
                false,
            ),
            (
                "W_SpecialisedTupleObject_ff.value1",
                SPECIALISED_TUPLE_FF_VALUE1_OFFSET,
                8,
                Type::Float,
                false,
                true,
                false,
            ),
        ],
    )
});

static SPECIALISED_TUPLE_OO_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::specialisedtupleobject::*;
    build_object_descr_group(
        std::mem::size_of::<W_SpecialisedTupleObject_oo>(),
        SPECIALISED_TUPLE_OO_GC_TYPE_ID,
        &SPECIALISED_TUPLE_OO_TYPE as *const _ as usize,
        &[
            (
                "W_SpecialisedTupleObject_oo.value0",
                SPECIALISED_TUPLE_OO_VALUE0_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "W_SpecialisedTupleObject_oo.value1",
                SPECIALISED_TUPLE_OO_VALUE1_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
        ],
    )
});

static DICT_STORAGE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group(
        std::mem::size_of::<pyre_interpreter::DictStorage>(),
        0,
        0,
        &[
            (
                "DictStorage.values.ptr",
                DICT_STORAGE_VALUES_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
            (
                "DictStorage.values.len",
                DICT_STORAGE_VALUES_LEN_OFFSET,
                8,
                Type::Int,
                false,
                false,
                false,
            ),
        ],
    )
});

static PYFRAME_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group(
        std::mem::size_of::<pyre_interpreter::pyframe::PyFrame>(),
        PYFRAME_GC_TYPE_ID,
        0,
        &[
            (
                "PyFrame.locals_cells_stack_w",
                crate::frame_layout::PYFRAME_LOCALS_CELLS_STACK_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.valuestackdepth",
                crate::frame_layout::PYFRAME_VALUESTACKDEPTH_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "PyFrame.last_instr",
                crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET,
                8,
                Type::Int,
                true,
                false,
                false,
            ),
            (
                "PyFrame.pycode",
                crate::frame_layout::PYFRAME_PYCODE_OFFSET,
                8,
                Type::Ref,
                true,
                false,
                false,
            ),
            (
                "PyFrame.w_globals",
                crate::frame_layout::PYFRAME_W_GLOBALS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.debugdata",
                crate::frame_layout::PYFRAME_DEBUGDATA_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.lastblock",
                crate::frame_layout::PYFRAME_LASTBLOCK_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            // Phase 2.3 옵션 B prerequisite: inline PyFrame 생성 시 새 frame 의
            // execution_context 슬롯에 caller 의 ec 를 SetfieldGc 로 쓰기 위해
            // 필요. RPython parity 는 interp_jit.py:67 reds=[frame, ec] 의 ec
            // 슬롯과 동등 — pyre 는 ec 를 PyFrame 헤더에 inline 저장.
            (
                "PyFrame.execution_context",
                crate::frame_layout::PYFRAME_EXECUTION_CONTEXT_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.f_generator_nowref",
                crate::frame_layout::PYFRAME_F_GENERATOR_NOWREF_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.w_yielding_from",
                crate::frame_layout::PYFRAME_W_YIELDING_FROM_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            (
                "PyFrame.f_backref",
                crate::frame_layout::PYFRAME_F_BACKREF_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
    )
});

impl Descr for PyreSizeDescr {
    fn index(&self) -> u32 {
        SIZE_DESCR_TAG | (self.obj_size as u32 & 0x0FFF_FFFF)
    }

    fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
        Some(self)
    }
}

impl SizeDescr for PyreSizeDescr {
    fn size(&self) -> usize {
        self.obj_size
    }

    fn type_id(&self) -> u32 {
        self.type_id
    }

    fn vtable(&self) -> usize {
        self.vtable
    }

    fn is_immutable(&self) -> bool {
        false
    }
    fn all_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
        &self.all_fielddescrs
    }
    fn gc_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
        &self.gc_fielddescrs
    }
    /// descr.py SizeDescr.is_object: every PyreSizeDescr that ships a
    /// vtable corresponds to a Python object (W_IntObject / W_ListObject /
    /// W_RangeIterator / …). `ensure_ptr_info_arg0` (optimizer.py:480)
    /// uses this to dispatch InstancePtrInfo vs StructPtrInfo.
    fn is_object(&self) -> bool {
        self.vtable != 0
    }
}

/// Empty-struct fallback for `BhDescr::Size` decode (`make_descr_from_bh`).
/// RPython `descr.py:188 init_size_descr` records an empty
/// `all_fielddescrs` list when the underlying STRUCT has no GC fields, so
/// the consumer-side decoder still needs a constructor that produces a
/// `PyreSizeDescr` with an empty field-list rather than refusing to build
/// one. Producers carrying a populated field-list go through
/// `simple_descr_group_from_bh_size` instead.
pub fn make_size_descr_with_type_and_vtable(
    obj_size: usize,
    type_id: u32,
    vtable: usize,
) -> DescrRef {
    Arc::new(PyreSizeDescr {
        obj_size,
        type_id,
        vtable,
        all_fielddescrs: Vec::new(),
        gc_fielddescrs: Vec::new(),
    })
}

/// Synthetic `len` field descriptor matching upstream
/// `descr.py:264 FieldDescr("len", ofs, WORD, FLAG_SIGNED)`. Lives at
/// offset 0 of the `Ptr(GcArray(T))` block (FixedObjectArray /
/// pyobject_gcarray layout): items start at `base_size`, so the word
/// before items is the length header. Returned from
/// `PyreArrayDescr::len_descr()` so `gen_initialize_len`
/// (`rewrite.py:565,572`) emits the runtime length store after
/// `CallMallocNurseryVarsize`.
fn array_lendescr_at_offset_zero() -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset: 0,
        field_size: std::mem::size_of::<usize>(),
        field_type: Type::Int,
        signed: true,
        immutable: false,
        quasi_immutable: false,
        name: "len",
        index_in_parent: 0,
        parent_descr: None,
    })
}

/// Construct the optional `len_descr` for an array layout. RPython
/// `descr.py:359 nolength` flag parity: arrays without a length-prefix
/// header (`base_size == 0`, used by raw int/float-backed varsize
/// stores) return `None`; length-prefixed layouts (`base_size >= WORD`,
/// e.g. `Ptr(GcArray(PyObjectRef))` whose first word IS the length
/// header) return a synthetic FieldDescr at offset 0.
fn maybe_array_lendescr(base_size: usize) -> Option<DescrRef> {
    (base_size >= std::mem::size_of::<usize>()).then(array_lendescr_at_offset_zero)
}

/// Create an array descriptor for a pointer-backed array field.
pub fn make_array_descr(
    base_size: usize,
    item_size: usize,
    item_type: Type,
    signed: bool,
) -> DescrRef {
    Arc::new(PyreArrayDescr {
        base_size,
        item_size,
        type_id: 0,
        item_type,
        signed,
        len_descr: maybe_array_lendescr(base_size),
    })
}

pub fn make_array_descr_with_type(
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    signed: bool,
) -> DescrRef {
    Arc::new(PyreArrayDescr {
        base_size,
        item_size,
        type_id,
        item_type,
        signed,
        len_descr: maybe_array_lendescr(base_size),
    })
}

// ── Range iterator field descriptors ─────────────────────────────────

use pyre_interpreter::{DICT_STORAGE_VALUES_LEN_OFFSET, DICT_STORAGE_VALUES_OFFSET};
use pyre_object::floatobject::{FLOAT_FLOATVAL_OFFSET, W_FloatObject};
use pyre_object::intobject::W_IntObject;
use pyre_object::pyobject::OB_TYPE_OFFSET;
use pyre_object::rangeobject::{
    RANGE_ITER_CURRENT_OFFSET, RANGE_ITER_STEP_OFFSET, RANGE_ITER_STOP_OFFSET,
};
use pyre_object::{
    BOOL_BOOLVAL_OFFSET, DICT_LEN_OFFSET, FLOAT_ARRAY_HEAP_CAP_OFFSET, FLOAT_ARRAY_LEN_OFFSET,
    FLOAT_ARRAY_PTR_OFFSET, INT_ARRAY_HEAP_CAP_OFFSET, INT_ARRAY_LEN_OFFSET, INT_ARRAY_PTR_OFFSET,
    INT_INTVAL_OFFSET, STR_LEN_OFFSET, W_ListObject, W_TupleObject,
};
use pyre_object::{FLOAT_TYPE, INT_TYPE};

/// Field descriptor for `PyObject.w_class` (Ref, mutable).
///
/// PyObject layout: [ob_type(8)] [w_class(8)]
/// The w_class field holds the Python class for all object types.
///
/// RPython parity: jit.promote(w_obj.__class__) reads typeptr via
/// getfield_gc_r then GUARD_VALUE. This is the pyre equivalent — a
/// field read on the common PyObject header.
///
/// Mutable because __class__ assignment can change it.
pub fn w_class_descr() -> DescrRef {
    make_field_descr(pyre_object::pyobject::W_CLASS_OFFSET, 8, Type::Ref, false)
}

/// Alias for backward compatibility — same as w_class_descr().
pub fn instance_w_type_descr() -> DescrRef {
    w_class_descr()
}

/// Field descriptor for `W_RangeIterator.current` (i64, signed).
pub fn range_iter_current_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_DESCR_GROUP, 0)
}

/// Field descriptor for `W_RangeIterator.stop` (i64, signed).
pub fn range_iter_stop_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_DESCR_GROUP, 1)
}

/// Field descriptor for `W_RangeIterator.step` (i64, signed).
pub fn range_iter_step_descr() -> DescrRef {
    field_descr_from_group(&RANGE_ITER_DESCR_GROUP, 2)
}

/// rlist.py:116 `l.length` — live length of a list under the Object
/// strategy. Under Integer/Float strategies this field is 0 and
/// consumers must dispatch on `list.strategy` first.
pub fn list_length_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 0)
}

/// rlist.py:116 `l.items: Ptr(GcArray(OBJECTPTR))` — pointer to the
/// `ItemsBlock` GcArray body. Callers that need items[i] must combine
/// with the `PY_OBJECT_ARRAY` array descr (item_size=8, Ref,
/// base_size=`ITEMS_BLOCK_ITEMS_OFFSET`); callers that need capacity
/// must issue `ArraylenGc` against the same array descr.
pub fn list_items_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 1)
}

pub fn list_strategy_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 2)
}

pub fn list_int_items_ptr_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 3)
}

pub fn list_int_items_len_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 4)
}

pub fn list_int_items_heap_cap_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 5)
}

pub fn list_float_items_ptr_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 6)
}

pub fn list_float_items_len_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 7)
}

pub fn list_float_items_heap_cap_descr() -> DescrRef {
    field_descr_from_group(&W_LIST_DESCR_GROUP, 8)
}

/// `Ptr(GcArray(OBJECTPTR))` — `wrappeditems` body per
/// `tupleobject.py:381` `_immutable_fields_ = ['wrappeditems[*]']`.
/// Immutable. Length comes from `arraylen_gc(items_block,
/// pyobject_gcarray_descr)` against the GcArray header — no
/// `tuple_length_descr` exists per upstream tupleobject.py:376-390
/// (`W_TupleObject` carries `wrappeditems` only).
pub fn tuple_wrappeditems_descr() -> DescrRef {
    field_descr_from_group(&W_TUPLE_DESCR_GROUP, 0)
}

/// `W_SpecialisedTupleObject_ii.value0` — inline `i64` per
/// `specialisedtupleobject.py:34-44`. Immutable.
pub fn specialised_tuple_ii_value0_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_II_DESCR_GROUP, 0)
}

/// `W_SpecialisedTupleObject_ii.value1` — inline `i64`. Immutable.
pub fn specialised_tuple_ii_value1_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_II_DESCR_GROUP, 1)
}

/// `W_SpecialisedTupleObject_ff.value0` — inline `f64`. Immutable.
pub fn specialised_tuple_ff_value0_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_FF_DESCR_GROUP, 0)
}

/// `W_SpecialisedTupleObject_ff.value1` — inline `f64`. Immutable.
pub fn specialised_tuple_ff_value1_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_FF_DESCR_GROUP, 1)
}

/// `W_SpecialisedTupleObject_oo.value0` — inline `PyObjectRef`. Immutable.
pub fn specialised_tuple_oo_value0_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_OO_DESCR_GROUP, 0)
}

/// `W_SpecialisedTupleObject_oo.value1` — inline `PyObjectRef`. Immutable.
pub fn specialised_tuple_oo_value1_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_OO_DESCR_GROUP, 1)
}

/// `ItemsBlock.capacity` — the GcArray length header at offset 0 of
/// an `ItemsBlock`, matching `rlist.py:84/251` `len(l.items)`
/// (allocated capacity, not live length). Immutable: once a block is
/// allocated the capacity is fixed; resize allocates a fresh block.
/// Callers combine `list_items_descr()` / `tuple_wrappeditems_descr()`
/// → `ItemsBlock*` with this descr to read the block's allocated size.
pub fn items_block_capacity_descr() -> DescrRef {
    make_immutable_field_descr(0, 8, Type::Int, false)
}

pub fn int_intval_descr() -> DescrRef {
    field_descr_from_group(&W_INT_DESCR_GROUP, 0)
}

pub fn bool_boolval_descr() -> DescrRef {
    field_descr_from_group(&W_BOOL_DESCR_GROUP, 0)
}

pub fn float_floatval_descr() -> DescrRef {
    field_descr_from_group(&W_FLOAT_DESCR_GROUP, 0)
}

pub fn str_len_descr() -> DescrRef {
    make_immutable_field_descr(STR_LEN_OFFSET, 8, Type::Int, false)
}

pub fn dict_len_descr() -> DescrRef {
    make_field_descr(DICT_LEN_OFFSET, 8, Type::Int, false)
}

pub fn dict_storage_values_ptr_descr() -> DescrRef {
    field_descr_from_group(&DICT_STORAGE_DESCR_GROUP, 0)
}

pub fn dict_storage_values_len_descr() -> DescrRef {
    field_descr_from_group(&DICT_STORAGE_DESCR_GROUP, 1)
}

// ── Object header & allocation descriptors ──────────────────────────

/// Field descriptor for ob_type (PyObject.ob_type pointer) — immutable.
/// heaptracker.py:66: `if name == 'typeptr': continue`
pub fn ob_type_descr() -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset: OB_TYPE_OFFSET,
        field_size: 8,
        field_type: Type::Int,
        signed: false,
        immutable: true,
        quasi_immutable: false,
        name: "typeptr",
        index_in_parent: 0,
        parent_descr: None,
    })
}

/// Size descriptor for W_IntObject allocation via NewWithVtable.
/// vtable = &INT_TYPE (ob_type for virtual materialization).
pub fn w_int_size_descr() -> DescrRef {
    W_INT_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for W_BoolObject allocation via NewWithVtable.
/// vtable = &BOOL_TYPE; type_id = 0 (bool reuses the OBJECT root id).
pub fn w_bool_size_descr() -> DescrRef {
    W_BOOL_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for W_RangeIterator allocation via NewWithVtable.
/// vtable = &RANGE_ITER_TYPE; type_id = 0.
pub fn w_range_iter_size_descr() -> DescrRef {
    RANGE_ITER_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for W_FloatObject allocation via NewWithVtable.
/// vtable = &FLOAT_TYPE (ob_type for virtual materialization).
pub fn w_float_size_descr() -> DescrRef {
    W_FLOAT_DESCR_GROUP.size_descr.clone()
}

/// Cached SizeDescr for the host PyFrame virtualizable.
///
/// RPython's `GcCache.get_size_descr()` returns a stable descriptor
/// object for a given struct. Pyre keeps the PyFrame descriptors in the
/// `PYFRAME_DESCR_GROUP` singleton, so callers that need the parent
/// SizeDescr for `VirtualizableInfo::finalize_arc` must reuse that
/// cached Arc instead of allocating a fresh ephemeral `SizeDescr`.
pub fn pyframe_size_descr() -> DescrRef {
    PYFRAME_DESCR_GROUP.size_descr.clone()
}

pub fn pyframe_locals_cells_stack_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 0)
}

pub fn pyframe_stack_depth_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 1)
}

pub fn pyframe_next_instr_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 2)
}

pub fn pyframe_code_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 3)
}

pub fn pyframe_dict_storage_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 4)
}

/// rewrite.py:665-695 handle_call_assembler scalar field read for the
/// `debugdata` slot of the virtualizable expansion (Phase D-1 prereq).
pub fn pyframe_debugdata_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 5)
}

/// rewrite.py:665-695 handle_call_assembler scalar field read for the
/// `lastblock` slot of the virtualizable expansion (Phase D-1 prereq).
pub fn pyframe_lastblock_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 6)
}

/// Phase 2.3 옵션 B prerequisite: PyFrame.execution_context FieldDescr.
/// inline PyFrame 생성 시 caller 의 ec 를 새 frame 으로 SetfieldGc 하기 위해.
/// 호출 사이트는 다음 세션의 `helpers.rs::emit_new_pyframe_inline*`.
pub fn pyframe_execution_context_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 7)
}

pub fn pyframe_f_generator_nowref_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 8)
}

pub fn pyframe_w_yielding_from_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 9)
}

pub fn pyframe_f_backref_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_descr_indices_are_stable_and_distinct() {
        let a = make_field_descr(8, 8, Type::Int, false);
        let b = make_field_descr(8, 8, Type::Int, false);
        let c = make_field_descr(16, 8, Type::Int, false);

        assert_eq!(a.index(), b.index());
        assert_ne!(a.index(), c.index());
    }

    #[test]
    fn test_array_descr_indices_are_stable_and_distinct() {
        let a = make_array_descr(0, 8, Type::Int, false);
        let b = make_array_descr(0, 8, Type::Int, false);
        let c = make_array_descr(0, 8, Type::Ref, false);

        assert_eq!(a.index(), b.index());
        assert_ne!(a.index(), c.index());
    }

    #[test]
    fn make_call_descr_from_bh_round_trips_most_general_effectinfo() {
        use majit_ir::EffectInfo;
        use majit_translate::jitcode::BhCallDescr;

        let bh = BhCallDescr::from_arg_classes("r".to_string(), 'r', EffectInfo::MOST_GENERAL);

        let descr = make_call_descr_from_bh(&bh);
        let call = descr
            .as_call_descr()
            .expect("make_call_descr_from_bh must produce a CallDescr-shaped descr");

        assert_eq!(call.arg_types(), &[Type::Ref]);
        assert_eq!(call.result_type(), Type::Ref);
        assert_eq!(call.result_size(), 8);
        assert!(!call.is_result_signed());
        assert_eq!(call.get_extra_info(), &EffectInfo::MOST_GENERAL);
        assert!(call.get_extra_info().check_can_raise(false));
    }

    #[test]
    fn make_call_descr_from_bh_round_trips_cannot_raise_effectinfo() {
        use majit_ir::{EffectInfo, ExtraEffect, OopSpecIndex};
        use majit_translate::jitcode::BhCallDescr;

        let extra_info = EffectInfo::const_new(ExtraEffect::CannotRaise, OopSpecIndex::None);
        let bh = BhCallDescr::from_arg_classes("ir".to_string(), 'v', extra_info.clone());

        let descr = make_call_descr_from_bh(&bh);
        let call = descr
            .as_call_descr()
            .expect("make_call_descr_from_bh must produce a CallDescr-shaped descr");

        assert_eq!(call.arg_types(), &[Type::Int, Type::Ref]);
        assert_eq!(call.result_type(), Type::Void);
        assert_eq!(call.result_size(), 0);
        assert_eq!(call.get_extra_info(), &extra_info);
        assert!(!call.get_extra_info().check_can_raise(false));
    }

    #[test]
    fn make_call_descr_from_bh_preserves_singlefloat_result_layout() {
        use majit_ir::EffectInfo;
        use majit_translate::jitcode::{BhCallDescr, CallResultErasedKey};

        let bh = BhCallDescr::from_arg_classes("S".to_string(), 'S', EffectInfo::MOST_GENERAL);

        assert_eq!(bh.arg_classes, "S");
        assert_eq!(bh.result_type, 'S');
        assert_eq!(bh.result_size, 4);
        assert!(!bh.result_signed);
        assert_eq!(bh.result_erased, CallResultErasedKey::SingleFloat);

        let descr = make_call_descr_from_bh(&bh);
        let call = descr
            .as_call_descr()
            .expect("make_call_descr_from_bh must produce a CallDescr-shaped descr");

        assert_eq!(call.arg_types(), &[Type::Int]);
        assert_eq!(call.result_type(), Type::Int);
        assert_eq!(call.result_class(), 'S');
        assert_eq!(call.result_size(), 4);
        assert!(!call.is_result_signed());
    }

    #[test]
    fn make_descr_from_bh_field_preserves_parent_name_index() {
        use majit_ir::descr::ArrayFlag;
        use majit_translate::jitcode::{BhDescr, BhFieldSpec, BhSizeSpec};

        let parent = BhSizeSpec {
            size: 24,
            type_id: 7,
            vtable: 0,
            all_fielddescrs: vec![
                BhFieldSpec {
                    index: 0,
                    name: "Cell.next".into(),
                    offset: 8,
                    field_size: 8,
                    field_type: Type::Ref,
                    field_flag: ArrayFlag::Pointer,
                    is_field_signed: false,
                    is_immutable: false,
                    is_quasi_immutable: false,
                    index_in_parent: 0,
                },
                BhFieldSpec {
                    index: 1,
                    name: "Cell.value".into(),
                    offset: 16,
                    field_size: 8,
                    field_type: Type::Int,
                    field_flag: ArrayFlag::Signed,
                    is_field_signed: true,
                    is_immutable: true,
                    is_quasi_immutable: false,
                    index_in_parent: 1,
                },
            ],
        };

        let descr = make_descr_from_bh(&BhDescr::Field {
            offset: 16,
            field_size: 8,
            field_type: Type::Int,
            field_flag: ArrayFlag::Signed,
            is_field_signed: true,
            is_immutable: true,
            is_quasi_immutable: false,
            index_in_parent: 1,
            parent: Some(parent),
            name: "value".into(),
            owner: "Cell".into(),
        });
        let field = descr.as_field_descr().expect("Field BhDescr -> FieldDescr");

        assert_eq!(field.field_name(), "Cell.value");
        assert_eq!(field.index_in_parent(), 1);
        assert_eq!(field.offset(), 16);
        assert!(field.is_immutable());
        let parent = field
            .get_parent_descr()
            .expect("FieldDescr.parent_descr must be preserved");
        let size = parent
            .as_size_descr()
            .expect("parent_descr must be a SizeDescr");
        assert_eq!(size.size(), 24);
        assert_eq!(size.type_id(), 7);
        assert_eq!(size.all_fielddescrs().len(), 2);
        assert_eq!(size.all_fielddescrs()[1].field_name(), "Cell.value");
    }

    #[test]
    fn make_descr_from_bh_struct_array_preserves_type_and_interior_fields() {
        use majit_ir::descr::ArrayFlag;
        use majit_translate::jitcode::{BhDescr, BhFieldSpec, BhInteriorFieldSpec, BhSizeSpec};

        let fields = vec![
            BhFieldSpec {
                index: 0,
                name: "Point.x".into(),
                offset: 0,
                field_size: 8,
                field_type: Type::Int,
                field_flag: ArrayFlag::Signed,
                is_field_signed: true,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 0,
            },
            BhFieldSpec {
                index: 1,
                name: "Point.y".into(),
                offset: 8,
                field_size: 8,
                field_type: Type::Float,
                field_flag: ArrayFlag::Float,
                is_field_signed: false,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 1,
            },
        ];
        let owner = BhSizeSpec {
            size: 16,
            type_id: 11,
            vtable: 0,
            all_fielddescrs: fields.clone(),
        };
        let interior_fields = vec![
            BhInteriorFieldSpec {
                index: 0,
                field: fields[0].clone(),
                owner: owner.clone(),
            },
            BhInteriorFieldSpec {
                index: 1,
                field: fields[1].clone(),
                owner,
            },
        ];

        let descr = make_descr_from_bh(&BhDescr::Array {
            base_size: 8,
            itemsize: 16,
            type_id: 42,
            item_type: Type::Ref,
            is_array_of_pointers: false,
            is_array_of_structs: true,
            is_item_signed: false,
            interior_fields,
        });
        let array = descr.as_array_descr().expect("Array BhDescr -> ArrayDescr");

        assert!(array.is_array_of_structs());
        assert_eq!(array.type_id(), 42);
        assert_eq!(array.item_type(), Type::Ref);
        let interior = array
            .get_all_interiorfielddescrs()
            .expect("struct array must preserve interior field descrs");
        assert_eq!(interior.len(), 2);
        let second = interior[1]
            .as_interior_field_descr()
            .expect("interior field descr shape");
        assert_eq!(second.field_descr().field_name(), "Point.y");
        let parent = second
            .field_descr()
            .get_parent_descr()
            .expect("interior field parent_descr must be preserved");
        assert_eq!(parent.as_size_descr().unwrap().size(), 16);
    }
}

/// resume.py:1124-1132: allocate_raw_buffer uses
/// callinfo_for_oopspec(OS_RAW_MALLOC_VARSIZE_CHAR) to get the calldescr.
pub fn make_raw_malloc_calldescr() -> DescrRef {
    majit_ir::make_raw_malloc_calldescr()
}

/// descr.py:273 ArrayDescr for array-of-structs (FLAG_STRUCT).
/// resume.py:749: allocate_array(self.size, self.arraydescr, clear=True).
pub fn make_struct_array_descr(descr_index: u32, base_size: usize, item_size: usize) -> DescrRef {
    make_struct_array_descr_full(descr_index, base_size, item_size, 0, Type::Void, &[])
}

fn simple_field_spec_from_bh(
    spec: &majit_translate::jitcode::BhFieldSpec,
) -> majit_ir::descr::SimpleFieldDescrSpec {
    majit_ir::descr::SimpleFieldDescrSpec {
        index: spec.index,
        name: spec.name.clone(),
        offset: spec.offset,
        field_size: spec.field_size,
        field_type: spec.field_type,
        is_immutable: spec.is_immutable,
        is_quasi_immutable: spec.is_quasi_immutable,
        flag: spec.field_flag,
        virtualizable: false,
        index_in_parent: spec.index_in_parent,
    }
}

fn simple_descr_group_from_bh_size(
    spec: &majit_translate::jitcode::BhSizeSpec,
) -> majit_ir::descr::SimpleDescrGroup {
    let field_specs: Vec<_> = spec
        .all_fielddescrs
        .iter()
        .map(simple_field_spec_from_bh)
        .collect();
    majit_ir::descr::make_simple_descr_group(
        u32::MAX,
        spec.size,
        spec.type_id,
        spec.vtable,
        &field_specs,
    )
}

#[derive(Debug)]
struct ParentBackedFieldDescr {
    field: Arc<majit_ir::descr::SimpleFieldDescr>,
    parent: Arc<majit_ir::descr::SimpleSizeDescr>,
}

impl Descr for ParentBackedFieldDescr {
    fn index(&self) -> u32 {
        self.field.index()
    }
    fn get_descr_index(&self) -> i32 {
        self.field.get_descr_index()
    }
    fn set_descr_index(&self, index: i32) {
        self.field.set_descr_index(index);
    }
    fn is_always_pure(&self) -> bool {
        self.field.is_always_pure()
    }
    fn is_quasi_immutable(&self) -> bool {
        self.field.is_quasi_immutable()
    }
    fn is_virtualizable(&self) -> bool {
        self.field.is_virtualizable()
    }
    fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
        Some(self)
    }
}

impl FieldDescr for ParentBackedFieldDescr {
    fn offset(&self) -> usize {
        self.field.offset()
    }
    fn field_size(&self) -> usize {
        self.field.field_size()
    }
    fn field_type(&self) -> Type {
        self.field.field_type()
    }
    fn is_pointer_field(&self) -> bool {
        self.field.is_pointer_field()
    }
    fn is_float_field(&self) -> bool {
        self.field.is_float_field()
    }
    fn is_field_signed(&self) -> bool {
        self.field.is_field_signed()
    }
    fn is_immutable(&self) -> bool {
        self.field.is_immutable()
    }
    fn field_name(&self) -> &str {
        self.field.field_name()
    }
    fn index_in_parent(&self) -> usize {
        self.field.index_in_parent()
    }
    fn get_parent_descr(&self) -> Option<DescrRef> {
        Some(self.parent.clone() as DescrRef)
    }
    fn get_vinfo(&self) -> Option<Arc<dyn majit_ir::descr::VinfoMarker>> {
        self.field.get_vinfo()
    }
}

fn field_descr_from_bh_field(
    field: &majit_translate::jitcode::BhFieldSpec,
    parent: Option<&majit_translate::jitcode::BhSizeSpec>,
) -> DescrRef {
    if let Some(parent) = parent {
        let group = simple_descr_group_from_bh_size(parent);
        if let Some((pos, _)) = parent.all_fielddescrs.iter().enumerate().find(|(_, spec)| {
            spec.index_in_parent == field.index_in_parent && spec.name == field.name
        }) {
            if let Some(descr) = group.field_descrs.get(pos) {
                return Arc::new(ParentBackedFieldDescr {
                    field: descr.clone(),
                    parent: group.size_descr.clone(),
                });
            }
        }
    }

    let descr = majit_ir::descr::SimpleFieldDescr::new_with_name(
        field.index,
        field.offset,
        field.field_size,
        field.field_type,
        field.is_immutable,
        field.field_flag,
        field.name.clone(),
    )
    .with_quasi_immutable(field.is_quasi_immutable);
    Arc::new(descr)
}

pub fn make_struct_array_descr_full(
    descr_index: u32,
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    interior_fields: &[majit_translate::jitcode::BhInteriorFieldSpec],
) -> DescrRef {
    use majit_ir::descr::{ArrayFlag, SimpleArrayDescr};
    // RPython `descr.py:388 InteriorFieldDescr.__init__(arraydescr,
    // fielddescr)` keeps the exact `arraydescr` object passed in.
    // Publish the final `Arc<SimpleArrayDescr>` first so every
    // `SimpleInteriorFieldDescr` clones the same Arc, then push the
    // interior list back onto that Arc through the `OnceLock`-backed
    // setter — `Arc::ptr_eq(interior.array_descr,
    // returned_array_descr)` now holds, matching the upstream
    // identity invariant.
    let array_descr = Arc::new(SimpleArrayDescr::with_flag(
        descr_index,
        base_size,
        item_size,
        type_id,
        item_type,
        ArrayFlag::Struct,
    ));
    if interior_fields.is_empty() {
        return array_descr;
    }

    let mut descrs = Vec::new();
    for interior in interior_fields {
        let owner_group = simple_descr_group_from_bh_size(&interior.owner);
        let field_pos = interior
            .owner
            .all_fielddescrs
            .iter()
            .position(|field| {
                field.index_in_parent == interior.field.index_in_parent
                    && field.name == interior.field.name
            })
            .unwrap_or(interior.field.index_in_parent);
        if let Some(field_descr) = owner_group.field_descrs.get(field_pos) {
            let ifd = majit_ir::descr::SimpleInteriorFieldDescr::new_with_owner(
                interior.index,
                array_descr.clone(),
                field_descr.clone(),
                owner_group.size_descr.clone(),
            );
            descrs.push(Arc::new(ifd) as DescrRef);
        }
    }

    array_descr.set_all_interiorfielddescrs(descrs);
    array_descr
}

/// Concrete `JitCodeDescr` adapter for `inline_call_*` opcodes.
///
/// RPython parity: `JitCode(AbstractDescr)` carries `fnaddr` +
/// `calldescr` + the callee's bytecode body and is emitted directly as
/// the descr operand of `inline_call_*`. The codewriter side surfaces
/// this as `BhDescr::JitCode { jitcode_index, fnaddr, calldescr }`
/// (`majit-translate/src/jit_codewriter/jitcode.rs:667`); the trace-side
/// walker (`jitcode_dispatch.rs::WalkContext`) consumes
/// `&[Arc<dyn Descr>]` and queries `as_jitcode_descr()` /
/// `jitcode_index()`.
///
/// `PyreJitCodeDescr` bridges those two layers: production callers
/// build a `Vec<DescrRef>` from the codewriter's `BhDescr` pool via
/// [`make_descr_from_bh`] (each `BhDescr::JitCode` wraps in this
/// struct so the walker's `as_jitcode_descr() -> Some(&self)` cast
/// succeeds; Field/Array/Size become `PyreFieldDescr` /
/// `PyreArrayDescr` / `PyreSizeDescr`; `Call` becomes a
/// `MetaCallDescr` carrying the codewriter's `EffectInfo`).
///
/// Tests in `jitcode_dispatch.rs` previously used a `TestJitCodeDescr`
/// duplicate of this shape — production code now goes through the same
/// type so the test fixture can be progressively replaced without
/// behaviour drift.
#[derive(Debug)]
pub struct PyreJitCodeDescr {
    jitcode_index: usize,
}

impl PyreJitCodeDescr {
    /// Build a `PyreJitCodeDescr` with the given runtime jitcode index.
    /// `jitcode_index` indexes into the runtime's all-jitcodes table
    /// (`pyre-jit-trace/src/jitcode_runtime.rs::ALL_JITCODES`); the
    /// walker's `sub_jitcode_lookup` resolves it to the callee's body.
    pub fn new(jitcode_index: usize) -> Self {
        Self { jitcode_index }
    }
}

impl Descr for PyreJitCodeDescr {
    fn as_jitcode_descr(&self) -> Option<&dyn JitCodeDescr> {
        Some(self)
    }
}

impl JitCodeDescr for PyreJitCodeDescr {
    fn jitcode_index(&self) -> usize {
        self.jitcode_index
    }
}

/// Build a `DescrRef` carrying a `PyreJitCodeDescr`. Production callers
/// use this when materializing the descr pool from a codewriter
/// `&[BhDescr]` (`BhDescr::JitCode { jitcode_index, .. }` → this
/// adapter).
pub fn make_jitcode_descr(jitcode_index: usize) -> DescrRef {
    Arc::new(PyreJitCodeDescr::new(jitcode_index))
}

/// Trace-side `SwitchDictDescr` adapter. The bytecode blackhole keeps
/// `BhDescr::Switch` directly; the MIFrame walker needs an `Arc<dyn Descr>`
/// slot for the same `Assembler.descrs` index.
#[derive(Debug)]
pub struct PyreSwitchDescr {
    dict: std::collections::HashMap<i64, usize>,
    const_keys_in_order: Vec<i64>,
}

impl PyreSwitchDescr {
    pub fn new(dict: std::collections::HashMap<i64, usize>) -> Self {
        let mut const_keys_in_order: Vec<i64> = dict.keys().copied().collect();
        const_keys_in_order.sort_unstable();
        Self {
            dict,
            const_keys_in_order,
        }
    }
}

impl Descr for PyreSwitchDescr {
    fn repr(&self) -> String {
        let entries = self
            .const_keys_in_order
            .iter()
            .map(|key| {
                let target = self
                    .dict
                    .get(key)
                    .expect("const_keys_in_order must mirror SwitchDictDescr.dict");
                format!("{key}: {target}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("<SwitchDictDescr {{{entries}}}>")
    }

    fn as_switch_descr(&self) -> Option<&dyn SwitchDescr> {
        Some(self)
    }
}

impl SwitchDescr for PyreSwitchDescr {
    fn lookup(&self, value: i64) -> Option<usize> {
        self.dict.get(&value).copied()
    }

    fn const_keys_in_order(&self) -> &[i64] {
        &self.const_keys_in_order
    }
}

#[cfg(test)]
mod switch_descr_tests {
    use super::*;

    #[test]
    fn pyre_switch_descr_repr_matches_rpython_switchdictdescr() {
        let descr = PyreSwitchDescr::new(std::collections::HashMap::from([(9, 23), (5, 17)]));

        assert_eq!(
            <PyreSwitchDescr as Descr>::repr(&descr),
            "<SwitchDictDescr {5: 17, 9: 23}>"
        );
        assert_eq!(descr.const_keys_in_order(), &[5, 9]);
    }
}

/// Trace-side adapter for pyre's Rust-vtable method descriptor.
#[derive(Debug)]
pub struct PyreVtableMethodDescr {
    trait_root: String,
    method_name: String,
}

impl PyreVtableMethodDescr {
    pub fn new(trait_root: String, method_name: String) -> Self {
        Self {
            trait_root,
            method_name,
        }
    }
}

impl Descr for PyreVtableMethodDescr {
    fn repr(&self) -> String {
        format!(
            "VtableMethodDescr({}::{})",
            self.trait_root, self.method_name
        )
    }
}

/// `assembler.py:23 Assembler.descrs` parity adapter — translate one
/// codewriter-side `BhDescr` slot (`majit-translate/src/jit_codewriter/jitcode.rs`)
/// into the matching trace-side `Arc<dyn Descr>` so trace ops emitted
/// by both the walker (`crate::jitcode_dispatch::dispatch_via_miframe`)
/// and the trait dispatch (`MIFrame::execute_opcode_step`) can carry
/// real-content descrs instead of `make_fail_descr` placeholders.
///
/// RPython parity: in upstream the metainterp + blackhole interpreter
/// share one `metainterp_sd.all_descrs` list — the same Python object
/// is the field/array/call descr regardless of which path is reading
/// it. pyre carries the codewriter-side typed list (`BhDescr`) and the
/// trait-side `Arc<dyn Descr>` view (`DescrRef`) as separate Rust
/// types because `Arc<dyn Descr>` cannot be downcast safely; this
/// adapter is the single point that bridges them.
///
/// Every branch builds the same descriptor kind carried by the
/// codewriter-side `BhDescr`:
/// * `Field` — `offset`, `field_size`, `field_type`, signedness, and
///   immutable/quasi-immutable flags are preserved.
/// * `Array` — `base_size`, `itemsize`, `type_id`, item type, signedness,
///   and array-of-structs classification are preserved.
/// * `Size` — `size`, `type_id`, and `vtable` are preserved.
/// * `Call` — `BhCallDescr.arg_classes` (e.g. `"iR"`) maps to
///   `Vec<Type>` per char (`i`->Int, `r`->Ref, `f`->Float; `R`/`I`/`F`
///   var-list markers split into the per-arg base type), and
///   `result_type` (one of `'i','r','f','v'`) maps to the `Type` of
///   the call result. `extra_info` is threaded into
///   `make_call_descr_with_effect`, preserving RPython `call.py:320`
///   effectinfo_from_writeanalyze parity for descr cache keys and
///   residual-call classification.
/// * `Switch` / `VableField` / `VableArray` / `VtableMethod` — trace-side
///   adapters preserve the descriptor slot instead of substituting a
///   fail-descr placeholder.
pub fn make_descr_from_bh(bh: &majit_translate::jitcode::BhDescr) -> DescrRef {
    use majit_translate::jitcode::BhDescr;
    match bh {
        BhDescr::Field {
            offset,
            field_size,
            field_type,
            field_flag,
            is_field_signed,
            is_immutable,
            is_quasi_immutable,
            index_in_parent,
            parent,
            name,
            owner,
            ..
        } => {
            let full_name = if owner.is_empty() || name.contains('.') {
                name.clone()
            } else {
                format!("{owner}.{name}")
            };
            // RPython `descr.py:214 FieldDescr.get_index()` returns
            // the value `heaptracker.get_fielddescr_index_in(STRUCT,
            // name)` recorded into `FieldDescr.index` at construction
            // time (`descr.py:200`).  Pyre's `BhDescr::Field` carries
            // that as `index_in_parent`; thread it through as
            // `BhFieldSpec.index` so the `parent` matching fallback
            // produces a `SimpleFieldDescr` whose `index()` matches the
            // upstream value rather than a `u32::MAX` sentinel.
            let field = majit_translate::jitcode::BhFieldSpec {
                index: *index_in_parent as u32,
                name: full_name,
                offset: *offset,
                field_size: *field_size,
                field_type: *field_type,
                field_flag: *field_flag,
                is_field_signed: *is_field_signed,
                is_immutable: *is_immutable,
                is_quasi_immutable: *is_quasi_immutable,
                index_in_parent: *index_in_parent,
            };
            field_descr_from_bh_field(&field, parent.as_ref())
        }
        BhDescr::Array {
            base_size,
            itemsize,
            type_id,
            item_type,
            is_array_of_structs,
            is_item_signed,
            interior_fields,
            ..
        } => {
            if *is_array_of_structs {
                make_struct_array_descr_full(
                    u32::MAX,
                    *base_size,
                    *itemsize,
                    *type_id,
                    *item_type,
                    interior_fields,
                )
            } else {
                make_array_descr_with_type(
                    *base_size,
                    *itemsize,
                    *type_id,
                    *item_type,
                    *is_item_signed,
                )
            }
        }
        BhDescr::Size {
            size,
            type_id,
            vtable,
            all_fielddescrs,
            ..
        } => {
            // RPython `descr.py:120 get_size_descr` → `:188 init_size_descr`
            // populates `SizeDescr.all_fielddescrs` (and the
            // `gc_fielddescrs` subset) from
            // `heaptracker.all_fielddescrs(STRUCT)` so consumers like
            // `info.py:180 init_fields` (`optimizeopt/info.rs:1989`)
            // see the full struct field list off the descr without a
            // round-trip through the codewriter.  When the producer
            // shipped a non-empty `all_fielddescrs`, build the parent
            // `SimpleSizeDescr` via the cyclic `make_simple_descr_group`
            // path so `Arc<SimpleFieldDescr>` parents back-reference
            // the same `SimpleSizeDescr` (`descr.py:200` parent slot).
            // The transient short-lived `BhDescr::Size` constructed in
            // `pyre-jit/src/eval.rs` (`bh_new` / `bh_new_with_vtable`
            // dispatch) carries an empty list and falls through to the
            // bare ctor, which is the existing test-helper shape.
            if all_fielddescrs.is_empty() {
                make_size_descr_with_type_and_vtable(*size, *type_id, *vtable)
            } else {
                let spec = majit_translate::jitcode::BhSizeSpec {
                    size: *size,
                    type_id: *type_id,
                    vtable: *vtable,
                    all_fielddescrs: all_fielddescrs.clone(),
                };
                simple_descr_group_from_bh_size(&spec).size_descr.clone()
            }
        }
        BhDescr::Call { calldescr } => make_call_descr_from_bh(calldescr),
        BhDescr::JitCode { jitcode_index, .. } => make_jitcode_descr(*jitcode_index),
        BhDescr::Switch { dict } => Arc::new(PyreSwitchDescr::new(dict.clone())),
        BhDescr::VableField { index } => majit_ir::descr::vable_static_field_descr(*index as u16),
        BhDescr::VableArray { index } => majit_ir::descr::vable_array_field_descr(*index as u16),
        BhDescr::VtableMethod {
            trait_root,
            method_name,
        } => Arc::new(PyreVtableMethodDescr::new(
            trait_root.clone(),
            method_name.clone(),
        )),
    }
}

/// `BhCallDescr` -> `CallDescr` adapter. RPython parity: codewriter
/// `Assembler.descrs` carries the same `CallDescr` instance the
/// metainterp pulls during op recording. pyre keeps the codewriter-side
/// call descr as serializable fields and rebuilds a `MetaCallDescr` on
/// demand here, preserving the per-call-site `EffectInfo`.
///
/// `arg_classes` is RPython `CallDescr.arg_classes`: one char per non-void
/// function argument. Uppercase `I/R/F` are assembler list markers and must not
/// appear here.
pub fn make_call_descr_from_bh(bh: &majit_translate::jitcode::BhCallDescr) -> DescrRef {
    let arg_types: Vec<Type> = bh
        .arg_classes
        .chars()
        .filter_map(|c| match c {
            'i' | 'S' => Some(Type::Int),
            'r' => Some(Type::Ref),
            'f' | 'L' => Some(Type::Float),
            _ => None,
        })
        .collect();
    let result_type = match bh.result_type {
        'i' | 'S' => Type::Int,
        'r' => Type::Ref,
        'f' | 'L' => Type::Float,
        _ => Type::Void,
    };
    // call.py:320 effectinfo_from_writeanalyze parity: the descr consumed
    // by pyjitpl/residual-call recording must expose the same EffectInfo
    // that the codewriter classified for this call site.
    majit_ir::descr::make_call_descr_full_with_result_class(
        u32::MAX,
        arg_types,
        result_type,
        bh.result_type,
        bh.result_signed,
        bh.result_size,
        bh.extra_info.clone(),
    )
}

/// descr.py:384 InteriorFieldDescr for SETINTERIORFIELD_GC.
/// assert arraydescr.flag == FLAG_STRUCT.
/// llmodel.py:648-665: bh_setinteriorfield_gc_{i,r,f} computes
/// offset = arraydescr.basesize + itemindex * itemsize + fielddescr.offset.
pub fn make_interior_field_descr(
    array_descr_index: u32,
    base_size: usize,
    item_size: usize,
    field_offset: usize,
    field_size: usize,
    field_type: u8, // 0=ref, 1=int, 2=float
    field_descr_index: u32,
) -> DescrRef {
    use majit_ir::descr::{
        ArrayFlag, SimpleArrayDescr, SimpleFieldDescr, SimpleInteriorFieldDescr,
    };
    let tp = match field_type {
        0 => Type::Ref,
        2 => Type::Float,
        _ => Type::Int,
    };
    // descr.py:387: assert arraydescr.flag == FLAG_STRUCT
    let array_descr = Arc::new(SimpleArrayDescr::with_flag(
        array_descr_index,
        base_size,
        item_size,
        0,
        Type::Void,
        ArrayFlag::Struct,
    ));
    let field_descr = Arc::new(SimpleFieldDescr::new(
        field_descr_index,
        field_offset,
        field_size,
        tp,
        true, // immutable (struct fields in array-of-struct)
    ));
    Arc::new(SimpleInteriorFieldDescr::new(
        field_descr_index,
        array_descr,
        field_descr,
    ))
}
