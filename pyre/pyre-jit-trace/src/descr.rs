//! Field descriptors for JIT IR operations.
//!
//! GetfieldGcI, GetfieldGcR, and SetfieldGc require a `DescrRef`
//! carrying field offset, size, and type information. This module
//! provides a concrete `PyreFieldDescr` implementing majit's
//! `FieldDescr` trait for pyre's `#[repr(C)]` object layout.

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::Weak;
use std::sync::atomic::{AtomicU32, Ordering};

use majit_ir::{
    ArrayDescr, Descr, DescrRef, FieldDescr, JitCodeDescr, SizeDescr, SwitchDescr, Type, VecMapExt,
};

// TODO: tag bits in the high nibble of the descr
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
    /// `effectinfo.py:465 compute_bitstrings` ei_index. `u32::MAX` until
    /// the codewriter publishes its `field_index` (`effectinfo.py:307-311`)
    /// onto this descr.
    ei_index: AtomicU32,
}

/// Structural key for `ARRAY_DESCR_REGISTRY`. Combination of all fields
/// that PyPy treats as part of `ArrayDescr` identity (`descr.py:273-279
/// + lendescr`). Two array descrs sharing this tuple share the same
/// `descr_id`.
///
/// `array_type_id` carries the codewriter lltype-identity proxy
/// (`majit-translate/src/jit_codewriter/call.rs::DescrIndexRegistry::array_index`
/// key) so the runtime registry's identity domain matches PyPy's
/// `gccache._cache_array[ARRAY_OR_STRUCT]` (`descr.py:348-360`) keyed
/// on the actual lltype object: two BhDescr::Array entries that
/// disagree only on the Rust type spelling
/// (e.g. `"Vec<Foo>"` vs `"Vec<Bar>"` with both at `type_id == 0`)
/// land on distinct registry slots, preventing the second
/// `set_ei_index` from clobbering the first.
///
/// `None` for legacy descrs minted by pyre-jit-trace internal
/// factories with no source-level array_type_id context; two `None`
/// entries still collide on the remaining structural tuple just as
/// the pre-bridge baseline did.
#[derive(Hash, Eq, PartialEq, Clone)]
struct ArrayDescrKey {
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type_bits: u32,
    signed: bool,
    len_offset: Option<usize>,
    array_type_id: Option<String>,
}

static NEXT_ARRAY_DESCR_ID: AtomicU32 = AtomicU32::new(0);

/// Maximum sequential ARRAY descr id. Bits 0-27 of the index are
/// available below `ARRAY_DESCR_TAG`; bit 28 is reserved by
/// `FIELD_DESCR_TAG`.
const ARRAY_DESCR_ID_MAX: u32 = 1 << 28;

static ARRAY_DESCR_REGISTRY: LazyLock<Mutex<majit_ir::VecAssoc<ArrayDescrKey, DescrRef>>> =
    LazyLock::new(|| Mutex::new(majit_ir::VecAssoc::new()));

fn alloc_array_descr_id() -> u32 {
    let id = NEXT_ARRAY_DESCR_ID.fetch_add(1, Ordering::Relaxed);
    assert!(
        id < ARRAY_DESCR_ID_MAX,
        "array descr registry exhausted (>2^28 instances) — index() bit 28 belongs to FIELD_DESCR_TAG"
    );
    id
}

/// `descr.py:241-254 get_type_flag(ARRAY.OF)` — element classification
/// for the runtime array mint.  Preserves the predicates the dropped
/// `PyreArrayDescr` reported: `is_item_signed` (Signed vs Unsigned for
/// ints), `is_array_of_pointers` (Pointer), `is_array_of_floats`
/// (Float).  Non-struct only — struct arrays are minted as
/// `SimpleArrayDescr(FLAG_STRUCT)` in `make_descr_from_bh`.
fn runtime_array_flag(item_type: Type, signed: bool) -> majit_ir::descr::ArrayFlag {
    use majit_ir::descr::ArrayFlag;
    match item_type {
        Type::Ref => ArrayFlag::Pointer,
        Type::Float => ArrayFlag::Float,
        Type::Int if signed => ArrayFlag::Signed,
        Type::Int => ArrayFlag::Unsigned,
        Type::Void => ArrayFlag::Void,
    }
}

fn get_or_create_array_descr(
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    signed: bool,
    len_offset: Option<usize>,
) -> DescrRef {
    get_or_create_array_descr_with_full_id(
        base_size, item_size, type_id, item_type, signed, len_offset, None,
    )
}

fn get_or_create_array_descr_with_full_id(
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    signed: bool,
    len_offset: Option<usize>,
    array_type_id: Option<String>,
) -> DescrRef {
    let key = ArrayDescrKey {
        base_size,
        item_size,
        type_id,
        item_type_bits: type_bits(item_type),
        signed,
        len_offset,
        array_type_id,
    };
    let mut cache = ARRAY_DESCR_REGISTRY
        .lock()
        .expect("ARRAY_DESCR_REGISTRY poisoned");
    if let Some(existing) = cache.get(&key) {
        return existing.clone();
    }
    // `descr.py:348-378 get_array_descr(gccache, ARRAY)`: when the
    // caller has an `array_type_id` (the codewriter lltype identity
    // proxy), `gc_cache._cache_array[LLType::Array(path_hash(atid))]`
    // is the authoritative cache slot — consult it FIRST so a prior
    // analyzer-side `gc_cache.get_array_descr` mint
    // (`SimpleArrayDescr`) is reused instead of layered under a fresh
    // runtime `SimpleArrayDescr`.  Matches PyPy `cpu.arraydescrof(ARRAY)`
    // per-ARRAY object identity — both analyzer and pyre runtime
    // consumers share one Arc per `LLType::Array(path_hash(atid))`.
    if let Some(ref atid) = key.array_type_id {
        let gc_key = majit_ir::descr::LLType::Array(majit_ir::descr::path_hash(atid));
        if let Some(existing) = majit_ir::descr::gc_cache()
            .lock()
            .unwrap()
            ._cache_array
            .get(&gc_key)
            .cloned()
        {
            // Memoise into the local structural cache so subsequent
            // `get_or_create_array_descr_with_full_id` calls with the
            // same structural key hit the local fast path without
            // re-consulting gc_cache.
            cache.insert(key.clone(), existing.clone());
            return existing;
        }
    }
    let descr_id = alloc_array_descr_id();
    // `array_type_id` Some → `LLType::Array(path_hash(atid))` cache
    // slot (analyzer ↔ runtime convergence path).
    // `array_type_id` None but `type_id != 0` → no codewriter
    // lltype-identity carrier but a stable GC-tid is available
    // (`make_array_descr_with_type` path).  Widening that tid to
    // u64 preserves per-tid identity in `BhDescr::Array.type_id`,
    // matching the behaviour producer sites in `eval.rs` /
    // `assembler.rs` / `jitcode.rs` relied on before the
    // `cache_key()` migration — without this fallback, every
    // `PY_OBJECT_ARRAY_GC_TYPE_ID`-class runtime descr collapsed
    // onto slot 0 at the `BhDescr` boundary.
    // `array_type_id` None and `type_id == 0` → no identity carrier
    // at all (legacy `make_array_descr` no-identity path); stay 0.
    let cache_key = match key.array_type_id.as_deref() {
        Some(atid) => majit_ir::descr::path_hash(atid),
        None if type_id != 0 => type_id as u64,
        None => 0,
    };
    // `descr.py:273-279` ArrayDescr — a single flag-bearing descriptor.
    // The runtime mint produces the same `SimpleArrayDescr` the
    // analyzer / `make_descr_from_bh` path mints, stamping the
    // content-addressed `ARRAY_DESCR_TAG | descr_id` index so `index()`
    // keeps the value the retired `PyreArrayDescr::index()` returned
    // (`ARRAY_DESCR_REGISTRY` still returns one Arc per structural key,
    // preserving `Arc::as_ptr` identity).
    let index = ARRAY_DESCR_TAG | (descr_id & 0x0FFF_FFFF);
    let mut array_descr = majit_ir::descr::SimpleArrayDescr::with_flag(
        index,
        base_size,
        item_size,
        type_id,
        item_type,
        runtime_array_flag(item_type, signed),
    );
    array_descr.lendescr = maybe_array_lendescr_at_offset(len_offset);
    array_descr.set_cache_key(cache_key);
    let arc: DescrRef = Arc::new(array_descr);
    cache.insert(key.clone(), arc.clone());
    // Publish the freshly-minted SimpleArrayDescr into
    // `gc_cache._cache_array` keyed on `LLType::Array(path_hash(atid))`
    // so later analyzer-side `gc_cache.get_array_descr` cache-hit
    // returns this exact Arc.  Without an `array_type_id` (legacy
    // `make_array_descr` callers), only the local
    // `ARRAY_DESCR_REGISTRY` carries the descr — gc_cache cannot
    // identify it.
    if let Some(ref atid) = key.array_type_id {
        majit_ir::descr_registry::register_keyed_array(
            majit_ir::descr::LLType::Array(majit_ir::descr::path_hash(atid)),
            arc.clone(),
        );
    }
    arc
}

impl Descr for PyreFieldDescr {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn index(&self) -> u32 {
        stable_field_index(self.offset, self.field_size, self.field_type, self.signed)
    }

    fn get_ei_index(&self) -> u32 {
        self.ei_index.load(Ordering::Relaxed)
    }

    fn set_ei_index(&self, ei_index: u32) {
        self.ei_index.store(ei_index, Ordering::Relaxed);
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
        ei_index: AtomicU32::new(u32::MAX),
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
pub fn make_field_descr_with_parent(parent: DescrRef, offset: usize) -> DescrRef {
    // resume.py:597-603 self.setfields → decoder.setfield(struct, fieldnum,
    // fielddescr) uses the parent's canonical FieldDescr. Return the *live*
    // field descr the parent SizeDescr already owns (full immutable /
    // quasi-immutable / name / ei_index) instead of minting a partial copy
    // that drops those properties; the parent SizeDescr is the same live
    // descr the original trace recorded against, so its `all_fielddescrs()`
    // are the canonical field descriptors keyed by index_in_parent.
    let sd = parent
        .as_size_descr()
        .expect("make_field_descr_with_parent: parent is not a SizeDescr");
    let fd = sd
        .all_fielddescrs()
        .iter()
        .find(|fd| fd.offset() == offset)
        .unwrap_or_else(|| {
            panic!("FieldDescr offset {offset} is not present in parent SizeDescr all_fielddescrs")
        });
    fd.clone() as DescrRef
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
        ei_index: AtomicU32::new(u32::MAX),
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
        ei_index: AtomicU32::new(u32::MAX),
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
        ei_index: AtomicU32::new(u32::MAX),
    })
}

/// Concrete size descriptor for fixed-size object allocations.
#[derive(Debug)]
pub struct PyreSizeDescr {
    obj_size: usize,
    type_id: u32,
    /// `_cache_size[LLType::Struct(cache_key)]` 슬롯 키 — `path_hash`로
    /// 만들어진 STRUCT 구조 identity (publish 슬롯과 동일).  `type_id`
    /// 는 `gc.alloc_gc_typed`용 dense u32 GC tid 이고, `cache_key` 는
    /// `descr.py:108-118 cache[STRUCT]`의 lltype-object identity 와 1:1
    /// 대응한다.  `SizeDescr.cache_key()` 가 이 값을 반환해
    /// `bh_size_spec_from_descr` 역방향 reader 가 publish 슬롯과 같은
    /// `LLType::Struct(cache_key)` 로 round-trip 한다.  init 0 은 단발
    /// fixture 용 fall-back (구조 identity 없는 케이스).
    cache_key: u64,
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
// Array GC type ids live in `pyre-object` alongside the backing storage
// structs/constants they describe (matching W_INT/W_FLOAT/W_LIST/W_TUPLE
// pattern). Re-exported here for existing call sites.
pub use pyre_object::object_array::{
    GC_FLOAT_ARRAY_GC_TYPE_ID, GC_INT_ARRAY_GC_TYPE_ID, PY_OBJECT_ARRAY_GC_TYPE_ID,
};
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
// `W_DICT_GC_TYPE_ID` lives in `pyre-object::dictmultiobject` alongside
// the `W_DictObject` struct it describes. Re-exported for the JIT
// registration site.
pub use pyre_object::dictmultiobject::W_DICT_GC_TYPE_ID;
// `W_MODULE_DICT_GC_TYPE_ID` lives in `pyre-object::dictmultiobject`
// alongside the `W_ModuleDictObject` struct it describes (the PyPy
// `dictmultiobject.py:328 W_ModuleDictObject` port).  Re-exported
// for the JIT registration site.
pub use pyre_object::dictmultiobject::W_MODULE_DICT_GC_TYPE_ID;
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
// `W_DICT_PROXY_GC_TYPE_ID` lives in `pyre-object::dictproxyobject`
// alongside the `W_DictProxyObject` struct it describes.  Re-exported
// for the JIT registration site so the typeid stays in the
// pyre-jit-trace exports table next to its sibling Module/PyFrame
// entries.
pub use pyre_object::dictproxyobject::W_DICT_PROXY_GC_TYPE_ID;
pub use pyre_object::strobject::W_STR_GC_TYPE_ID;
// `PYFRAME_GC_TYPE_ID` lives in `pyre-interpreter::pyframe` alongside
// the `PyFrame` struct it describes. Re-exported for the JIT
// registration site (`pyre-jit/src/eval.rs`).
// Registered ahead of any future
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

/// Build a SizeDescr group for a runtime PyObject layout and publish
/// it into `gc_cache._cache_size` under both the simple-name slot
/// AND the crate-stripped def-path slot.  PyPy `cache[STRUCT]`
/// collapses both into a single lltype-object identity; pyre's
/// analyzer currently hashes the simple name (use-site bare
/// identifier — collect_struct_names registers top-level structs by
/// `simple_name`) so that slot is the de-facto convergence point.
/// The def-path slot is published alongside as a forward-compatible
/// alias for the future analyzer use-import resolver (B-5 follow-up):
/// when that lands, analyzer's `owner_root` switches to qualified
/// form and the SAME `Arc<PyreSizeDescr>` is reachable via the
/// qualified hash.  `register_keyed_size` is first-write-wins per
/// `descr.py:25-47 setup_descrs` cache-iteration invariant — the
/// second publish's losing Arc does NOT enter `_cache_size_order`,
/// so `all_descrs` enumerates exactly one entry per logical
/// SizeDescr (PyPy's per-tuple identity).
///
/// `def_path` empty (or equal to `simple_name`) → single publish.
fn build_object_descr_group_with_def_path(
    obj_size: usize,
    type_id: u32,
    vtable: usize,
    fields: &[(&'static str, usize, usize, Type, bool, bool, bool)],
    simple_name: &str,
    def_path: &str,
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
                        ei_index: AtomicU32::new(u32::MAX),
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
        // `descr.py:108-118 get_size_descr` cache key — `path_hash`로
        // 만들어진 lltype-object identity.  Prefer the canonical
        // *def-path* qualifier (PyPy's `lltype.Struct` identity has
        // a single module-path keyed slot); fall back to simple-name
        // for legacy registrations without `def_path`.  Both
        // `path_hash(simple_name)` and `path_hash(def_path)` are still
        // dual-published as Arc aliases so untransformed bare-name
        // analyzer lookups (pending the use_imports lexical resolver,
        // [[orthodox-6item-2026-05-17]]) still hit.  Round-trip via
        // `bh_size_spec_from_descr` lands on the canonical def-path
        // slot.
        let cache_key = if !def_path.is_empty() {
            majit_ir::descr::path_hash(def_path)
        } else if !simple_name.is_empty() {
            majit_ir::descr::path_hash(simple_name)
        } else {
            0
        };
        PyreSizeDescr {
            obj_size,
            type_id,
            cache_key,
            vtable,
            all_fielddescrs,
            gc_fielddescrs,
        }
    });
    // Dual-publish: register under BOTH the simple-name slot AND
    // (when supplied) the crate-stripped def-path slot.
    //
    // PyPy `cache[STRUCT]` collapses both namespaces into a single
    // lltype-object identity; pyre's analyzer currently hashes the
    // use-site bare identifier (collect_struct_names registers
    // top-level structs at simple-name) so the simple-name slot is
    // the primary cache hit point.  The def-path slot is published
    // alongside as a forward-compatible alias for the future
    // analyzer use-import resolver (B-5 follow-up): when that lands,
    // analyzer's `owner_root` switches to qualified form and the
    // SAME `Arc<PyreSizeDescr>` is reachable via the qualified
    // hash.  `register_keyed_size` is first-write-wins per
    // `descr.py:25-47 setup_descrs` cache-iteration invariant — the
    // second registration's losing Arc does NOT enter
    // `_cache_size_order`, so `all_descrs` enumerates exactly one
    // entry per logical SizeDescr (PyPy's per-tuple identity).
    if !simple_name.is_empty() {
        let key = majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(simple_name));
        majit_ir::descr_registry::register_keyed_size(
            key,
            size_descr.clone() as majit_ir::DescrRef,
        );
    }
    if !def_path.is_empty() && def_path != simple_name {
        let key = majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(def_path));
        majit_ir::descr_registry::register_keyed_size(
            key,
            size_descr.clone() as majit_ir::DescrRef,
        );
    }
    PyreObjectDescrGroup { size_descr }
}

static W_INT_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
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
        "W_IntObject",
        "intobject::W_IntObject",
    )
});

static W_FLOAT_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
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
        "W_FloatObject",
        "floatobject::W_FloatObject",
    )
});

static W_BOOL_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
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
        "W_BoolObject",
        "boolobject::W_BoolObject",
    )
});

static RANGE_ITER_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
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
        "W_RangeIterator",
        "rangeobject::W_RangeIterator",
    )
});

/// `W_MethodObject` field layout — `w_function`, `w_self`, `w_class` per
/// `methodobject.rs:9-15`. All three are Ref slots; the JIT only consumes
/// `w_function` (for guarding which method) and `w_self` (for recovering
/// the receiver `OpRef` discarded by `LOAD_METHOD`). `w_class` is included
/// for layout completeness so the descrs match the struct order.
///
/// `w_function` and `w_self` are marked immutable per
/// `pypy/interpreter/function.py:567`
/// `_Method._immutable_fields_ = ['w_function', 'w_instance']`. `w_class`
/// is not listed there and stays mutable.
static W_METHOD_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::methodobject::{
        METHOD_W_CLASS_OFFSET, METHOD_W_FUNCTION_OFFSET, METHOD_W_SELF_OFFSET, W_METHOD_GC_TYPE_ID,
        W_METHOD_OBJECT_SIZE,
    };
    build_object_descr_group_with_def_path(
        W_METHOD_OBJECT_SIZE,
        W_METHOD_GC_TYPE_ID,
        &pyre_object::methodobject::METHOD_TYPE as *const _ as usize,
        &[
            (
                "W_MethodObject.w_function",
                METHOD_W_FUNCTION_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "W_MethodObject.w_self",
                METHOD_W_SELF_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "W_MethodObject.w_class",
                METHOD_W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "W_MethodObject",
        "methodobject::W_MethodObject",
    )
});

/// `pypy/objspace/std/typeobject.py:26-34 ObjectMutableCell`. The single
/// `w_value` field is read LIVE on the module-global cell fast path: a
/// frequently-rewritten global mutates the cell payload in place without
/// bumping `mstrategy.version` (`typeobject.py:55-71 write_cell`), so the
/// JIT cell read must observe the current value and stays MUTABLE (not
/// immutable / quasi-immutable). Cell identity is what the `version?`
/// quasi-immutable guard protects.
static W_OBJECT_MUTABLE_CELL_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::celldict::{
        OBJECT_MUTABLE_CELL_TYPE, W_OBJECT_MUTABLE_CELL_GC_PTR_OFFSETS,
        W_OBJECT_MUTABLE_CELL_GC_TYPE_ID, W_OBJECT_MUTABLE_CELL_OBJECT_SIZE,
    };
    build_object_descr_group_with_def_path(
        W_OBJECT_MUTABLE_CELL_OBJECT_SIZE,
        W_OBJECT_MUTABLE_CELL_GC_TYPE_ID,
        &OBJECT_MUTABLE_CELL_TYPE as *const _ as usize,
        &[(
            "W_ObjectMutableCell.w_value",
            W_OBJECT_MUTABLE_CELL_GC_PTR_OFFSETS[0],
            8,
            Type::Ref,
            false,
            false,
            false,
        )],
        "W_ObjectMutableCell",
        "celldict::W_ObjectMutableCell",
    )
});

static W_LIST_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    // Upstream `rpython/rtyper/lltypesystem/rlist.py:116`
    //     GcStruct("list", ("length", Signed), ("items", Ptr(ITEMARRAY)))
    // The parity-field pair is `(length, items)`. `strategy` +
    // `int_items` / `float_items` are pyre-only PRE-EXISTING-
    // ADAPTATIONs for the PyPy interp-level strategy split.
    build_object_descr_group_with_def_path(
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
                // plain-mutable. TODO — strategy split itself
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
            // TODO vs listobject.py's
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
        "W_ListObject",
        "listobject::W_ListObject",
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
    build_object_descr_group_with_def_path(
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
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "W_TupleObject",
        "tupleobject::W_TupleObject",
    )
});

static SPECIALISED_TUPLE_II_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    // `specialisedtupleobject.py:34` `_immutable_fields_ = ['value0',
    // 'value1']` — both fields immutable. Inline-field shape, no array
    // indirection.
    use pyre_object::specialisedtupleobject::*;
    build_object_descr_group_with_def_path(
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
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "W_SpecialisedTupleObject_ii",
        "specialisedtupleobject::W_SpecialisedTupleObject_ii",
    )
});

static SPECIALISED_TUPLE_FF_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::specialisedtupleobject::*;
    build_object_descr_group_with_def_path(
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
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "W_SpecialisedTupleObject_ff",
        "specialisedtupleobject::W_SpecialisedTupleObject_ff",
    )
});

static SPECIALISED_TUPLE_OO_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::specialisedtupleobject::*;
    build_object_descr_group_with_def_path(
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
            (
                "PyObject.w_class",
                pyre_object::pyobject::W_CLASS_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "W_SpecialisedTupleObject_oo",
        "specialisedtupleobject::W_SpecialisedTupleObject_oo",
    )
});

static DICT_STORAGE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
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
        "DictStorage",
        "executioncontext::DictStorage",
    )
});

// `pypy/objspace/std/sliceobject.py:13` `W_SliceObject._immutable_fields_ =
// ['w_start', 'w_stop', 'w_step']` — all three Ref fields are immutable
// once `__init__` runs.  The `space.newslice(w_start, w_end, w_step)` JIT
// shape allocates the W_SliceObject inline so the optimizer can virtualize
// the three SetfieldGc writes when the slice never escapes (per
// `optimizeopt/virtualize.py optimize_NEW_WITH_VTABLE`).
static W_SLICE_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    use pyre_object::sliceobject::*;
    build_object_descr_group_with_def_path(
        W_SLICE_OBJECT_SIZE,
        W_SLICE_GC_TYPE_ID,
        &pyre_object::sliceobject::SLICE_TYPE as *const _ as usize,
        &[
            (
                "W_SliceObject.w_start",
                SLICE_START_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "W_SliceObject.w_stop",
                SLICE_STOP_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
            (
                "W_SliceObject.w_step",
                SLICE_STEP_OFFSET,
                8,
                Type::Ref,
                false,
                true,
                false,
            ),
        ],
        "W_SliceObject",
        "sliceobject::W_SliceObject",
    )
});

static PYFRAME_DESCR_GROUP: LazyLock<PyreObjectDescrGroup> = LazyLock::new(|| {
    build_object_descr_group_with_def_path(
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
                crate::frame_layout::PYFRAME_W_GLOBALS_OBJ_OFFSET,
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
            // Inline PyFrame 생성 시 새 frame 의
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
            (
                "PyFrame.w_builtin",
                crate::frame_layout::PYFRAME_W_BUILTIN_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
            // R3.3b prep: canonical W_DictObject sibling slot
            // (`pyframe.py:49 self.w_globals` parity).  Used by the inline
            // new-PyFrame helper to populate the slot from the
            // function's `w_func_globals_obj` cache so trace-time
            // chases through `w_dict_get_dict_storage_proxy` observe
            // a non-null PyObjectRef.  R3.3 cutover will retire the
            // adjacent raw `PyFrame.w_globals` entry above and rename
            // this one to fully match PyPy's pyframe.py:49 shape.
            (
                "PyFrame.w_globals_obj",
                crate::frame_layout::PYFRAME_W_GLOBALS_OBJ_OFFSET,
                8,
                Type::Ref,
                false,
                false,
                false,
            ),
        ],
        "PyFrame",
        "pyframe::PyFrame",
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

    /// `descr.py:108-118 get_size_descr` cache identity 와 line-by-line
    /// 동등: `register_keyed_size` 가 publish 한 슬롯 키 (현재
    /// `path_hash_stripped_crate(module_path!(), bare_name)` —
    /// 즉 def-path 기반의 `path_hash("module::Bare")`).  매크로 publish
    /// 시 `__majit_type_id()` 에서 계산되어 인스턴스 슬롯에 저장된
    /// 값을 그대로 반환한다.  `bh_size_spec_from_descr` 역방향 reader 는
    /// 이 값을 `BhSizeSpec.type_id` 에 넣고 `simple_descr_group_from_bh_size`
    /// 는 `LLType::Struct(spec.type_id)` 로 publish 슬롯에 round-trip 한다.
    /// `type_id` (dense GC tid) 와 `cache_key` (structural identity) 는
    /// `descr.rs:1928-1934` 트레이트 doc 의 분리 contract 를 따른다.
    fn cache_key(&self) -> u64 {
        self.cache_key
    }

    fn vtable(&self) -> usize {
        self.vtable
    }

    /// The canonical `w_class` (Python class object) for instances of
    /// this type — `get_instantiate(vtable_type)`. Read live (not cached
    /// at construction) since the type objects are installed after the
    /// descrs are built. `None` before `init_typeobjects()` runs.
    fn w_class_obj(&self) -> Option<i64> {
        if self.vtable == 0 {
            return None;
        }
        let tp = self.vtable as *const pyre_object::pyobject::PyType;
        let w_class = unsafe { pyre_object::pyobject::get_instantiate(&*tp) };
        if w_class.is_null() {
            None
        } else {
            Some(w_class as i64)
        }
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
    // 빈 fielddescr fallback — `BhDescr::Size` 디코더가 구조 identity
    // 캐리어 없이 호출하는 자리.  `cache_key = 0` 은 round-trip 시
    // `simple_descr_group_from_bh_size` 의 no-identity branch
    // (`descr.rs:2382-2388`) 가 per-call distinct 처리하므로 안전.
    Arc::new(PyreSizeDescr {
        obj_size,
        type_id,
        cache_key: 0,
        vtable,
        all_fielddescrs: Vec::new(),
        gc_fielddescrs: Vec::new(),
    })
}

/// Synthetic `len` field descriptor matching upstream
/// `descr.py:264 FieldDescr("len", ofs, WORD, FLAG_SIGNED)`. Lives at
/// offset 0 of the `Ptr(GcArray(T))` block (FixedObjectArray /
/// pyobject_gcarray layout): items start at `base_size`, so the word
/// before items is the length header. Stored as
/// `SimpleArrayDescr.lendescr` so `gen_initialize_len`
/// (`rewrite.py:565,572`) emits the runtime length store after
/// `CallMallocNurseryVarsize`.
fn array_lendescr_at_offset(offset: usize) -> DescrRef {
    Arc::new(PyreFieldDescr {
        offset,
        field_size: std::mem::size_of::<usize>(),
        field_type: Type::Int,
        signed: true,
        immutable: false,
        quasi_immutable: false,
        name: "len",
        index_in_parent: 0,
        parent_descr: None,
        ei_index: AtomicU32::new(u32::MAX),
    })
}

/// Lift `Option<usize>` ↔ `Option<DescrRef>` so `make_array_descr*`
/// callers express nolength/length-prefixed shapes directly. PyPy
/// `descr.py:359-362` decides this from
/// `ARRAY_INSIDE._hints.get('nolength', False)`; the explicit
/// `Option<usize>` is the structural equivalent.
fn maybe_array_lendescr_at_offset(len_offset: Option<usize>) -> Option<DescrRef> {
    len_offset.map(array_lendescr_at_offset)
}

/// Create a fresh ARRAY descriptor without identity carrier.
///
/// `len_offset`: `None` for the `nolength=True` shape (descr.py:360);
/// `Some(off)` for length-prefixed layouts (descr.py:362).
///
/// PyPy `descr.py:348-378 get_array_descr(gccache, ARRAY)` keys
/// `_cache_array[ARRAY_OR_STRUCT]` on the ARRAY object identity, never
/// on its structural shape — two distinct lltype ARRAYs that share
/// `(base_size, item_size, item_type, signed, len_offset)` get
/// distinct `ArrayDescr` Arcs.  Pyre's no-identity-carrier callers
/// (this function: `array_type_id = None`, `type_id = 0`) cannot
/// participate in the keyed cache because they have no ARRAY-object
/// surrogate to hash; the orthodox behaviour is therefore "each call
/// is a distinct ARRAY" — mint fresh `SimpleArrayDescr` per call so
/// shape-coincident-but-logically-distinct ARRAYs receive distinct
/// `descr_id` slots.  Callers that need singleton semantics
/// (`int_array_descr`, `float_array_descr`, `pyobject_array_descr`,
/// …) route through [`make_array_descr_with_full_id`] with a stable
/// identity string instead — the keyed cache canonicalises by that
/// string.
pub fn make_array_descr(
    base_size: usize,
    item_size: usize,
    len_offset: Option<usize>,
    item_type: Type,
    signed: bool,
) -> DescrRef {
    let descr_id = alloc_array_descr_id();
    // No identity carrier — fresh mint per call (cache_key = 0 means "no
    // cache slot").  Single flag-bearing `SimpleArrayDescr` (descr.py:273-279);
    // index carries the content-addressed `ARRAY_DESCR_TAG | descr_id`.
    let index = ARRAY_DESCR_TAG | (descr_id & 0x0FFF_FFFF);
    let mut array_descr = majit_ir::descr::SimpleArrayDescr::with_flag(
        index,
        base_size,
        item_size,
        0,
        item_type,
        runtime_array_flag(item_type, signed),
    );
    array_descr.lendescr = maybe_array_lendescr_at_offset(len_offset);
    Arc::new(array_descr)
}

pub fn make_array_descr_with_type(
    base_size: usize,
    item_size: usize,
    type_id: u32,
    len_offset: Option<usize>,
    item_type: Type,
    signed: bool,
) -> DescrRef {
    get_or_create_array_descr(base_size, item_size, type_id, item_type, signed, len_offset)
}

/// Bridge-only factory that threads the codewriter's `array_type_id`
/// (`majit-translate::jit_codewriter::call::DescrIndexRegistry::array_index`
/// key) into `ArrayDescrKey` so two BhDescr::Array entries with
/// identical structural fields but different lltype spellings receive
/// distinct registry slots — matching upstream
/// `gccache._cache_array[ARRAY_OR_STRUCT]` (`descr.py:348-360`).
pub fn make_array_descr_with_full_id(
    base_size: usize,
    item_size: usize,
    type_id: u32,
    len_offset: Option<usize>,
    item_type: Type,
    signed: bool,
    array_type_id: Option<String>,
) -> DescrRef {
    get_or_create_array_descr_with_full_id(
        base_size,
        item_size,
        type_id,
        item_type,
        signed,
        len_offset,
        array_type_id,
    )
}

// ── Range iterator field descriptors ─────────────────────────────────

use pyre_interpreter::{DICT_STORAGE_VALUES_LEN_OFFSET, DICT_STORAGE_VALUES_OFFSET};
use pyre_object::floatobject::{FLOAT_FLOATVAL_OFFSET, W_FloatObject};
use pyre_object::intobject::W_IntObject;
use pyre_object::pyobject::OB_TYPE_OFFSET;
use pyre_object::rangeobject::{
    RANGE_ITER_CURRENT_OFFSET, RANGE_ITER_STEP_OFFSET, RANGE_ITER_STOP_OFFSET,
};
use pyre_object::strobject::STR_LEN_OFFSET;
use pyre_object::{
    BOOL_BOOLVAL_OFFSET, FLOAT_ARRAY_HEAP_CAP_OFFSET, FLOAT_ARRAY_LEN_OFFSET,
    FLOAT_ARRAY_PTR_OFFSET, INT_ARRAY_HEAP_CAP_OFFSET, INT_ARRAY_LEN_OFFSET, INT_ARRAY_PTR_OFFSET,
    INT_INTVAL_OFFSET, W_ListObject, W_TupleObject,
};
// Re-import the rest without duplication
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
    // Named "w_class" so `FieldDescr::is_w_class()` recognises the
    // header field; OptVirtualize must resolve it from the object's
    // class identity rather than indexing it against a virtual's value
    // fields (its `index_in_parent` of 0 would otherwise collide with
    // the first value field, e.g. `W_IntObject.intval`).
    Arc::new(PyreFieldDescr {
        offset: pyre_object::pyobject::W_CLASS_OFFSET,
        field_size: 8,
        field_type: Type::Ref,
        signed: false,
        immutable: false,
        quasi_immutable: false,
        name: "w_class",
        index_in_parent: 0,
        parent_descr: None,
        ei_index: AtomicU32::new(u32::MAX),
    })
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

/// `W_MethodObject.w_function` — the underlying function (W_FunctionObject
/// or W_BuiltinFunction) bound by `getattr(obj, name)`. Marked immutable
/// per `pypy/interpreter/function.py:567` `_Method._immutable_fields_`,
/// so reads survive cache invalidation across calls. Used by the
/// bound-method specialization in `call_callable_value`.
pub fn method_w_function_descr() -> DescrRef {
    field_descr_from_group(&W_METHOD_DESCR_GROUP, 0)
}

/// `W_MethodObject.w_self` — the receiver object. The bound-method
/// specialization extracts this via `GetfieldGcR` to recover the receiver
/// `OpRef` after `LOAD_METHOD` discarded it (load_method.rs:6334 pushes
/// `null_value` for `is_method` attrs). Immutable per
/// `_Method._immutable_fields_`.
pub fn method_w_self_descr() -> DescrRef {
    field_descr_from_group(&W_METHOD_DESCR_GROUP, 1)
}

/// `typeobject.py:26-34 ObjectMutableCell.w_value` — the boxed payload of
/// a module-global cell, read LIVE on the cell fast path. The value is
/// rewritten in place when a hot global is reassigned without bumping the
/// strategy version, so this descriptor is mutable (not immutable /
/// quasi-immutable); the `version?` guard protects cell identity, not the
/// payload.
pub fn object_mutable_cell_value_descr() -> DescrRef {
    field_descr_from_group(&W_OBJECT_MUTABLE_CELL_DESCR_GROUP, 0)
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

pub fn tuple_w_class_descr() -> DescrRef {
    field_descr_from_group(&W_TUPLE_DESCR_GROUP, 1)
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

pub fn specialised_tuple_ii_w_class_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_II_DESCR_GROUP, 2)
}

/// `W_SpecialisedTupleObject_ff.value0` — inline `f64`. Immutable.
pub fn specialised_tuple_ff_value0_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_FF_DESCR_GROUP, 0)
}

/// `W_SpecialisedTupleObject_ff.value1` — inline `f64`. Immutable.
pub fn specialised_tuple_ff_value1_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_FF_DESCR_GROUP, 1)
}

pub fn specialised_tuple_ff_w_class_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_FF_DESCR_GROUP, 2)
}

/// `W_SpecialisedTupleObject_oo.value0` — inline `PyObjectRef`. Immutable.
pub fn specialised_tuple_oo_value0_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_OO_DESCR_GROUP, 0)
}

/// `W_SpecialisedTupleObject_oo.value1` — inline `PyObjectRef`. Immutable.
pub fn specialised_tuple_oo_value1_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_OO_DESCR_GROUP, 1)
}

pub fn specialised_tuple_oo_w_class_descr() -> DescrRef {
    field_descr_from_group(&SPECIALISED_TUPLE_OO_DESCR_GROUP, 2)
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
    // Python len(str) returns codepoint count.
    // unicodeobject.py:165 W_UnicodeObject._len() → _length field.
    make_immutable_field_descr(STR_LEN_OFFSET, 8, Type::Int, false)
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
        ei_index: AtomicU32::new(u32::MAX),
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

/// Size descriptor for canonical `W_TupleObject`.
pub fn w_tuple_size_descr() -> DescrRef {
    W_TUPLE_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `W_SpecialisedTupleObject_ii`.
pub fn specialised_tuple_ii_size_descr() -> DescrRef {
    SPECIALISED_TUPLE_II_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `W_SpecialisedTupleObject_ff`.
pub fn specialised_tuple_ff_size_descr() -> DescrRef {
    SPECIALISED_TUPLE_FF_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for `W_SpecialisedTupleObject_oo`.
pub fn specialised_tuple_oo_size_descr() -> DescrRef {
    SPECIALISED_TUPLE_OO_DESCR_GROUP.size_descr.clone()
}

/// Size descriptor for W_SliceObject allocation via NewWithVtable.
/// vtable = &SLICE_TYPE (ob_type for virtual materialization).
/// Mirrors `pypy/objspace/std/objspace.py:385` `space.newslice` →
/// `W_SliceObject(w_start, w_end, w_step)` allocation shape.
pub fn w_slice_size_descr() -> DescrRef {
    W_SLICE_DESCR_GROUP.size_descr.clone()
}

/// `W_SliceObject.w_start` — `Ptr(W_Root)` per
/// `sliceobject.py:13` `_immutable_fields_ = ['w_start', ...]`. Immutable.
pub fn slice_w_start_descr() -> DescrRef {
    field_descr_from_group(&W_SLICE_DESCR_GROUP, 0)
}

/// `W_SliceObject.w_stop` — `Ptr(W_Root)`. Immutable.
pub fn slice_w_stop_descr() -> DescrRef {
    field_descr_from_group(&W_SLICE_DESCR_GROUP, 1)
}

/// `W_SliceObject.w_step` — `Ptr(W_Root)`. Immutable.
pub fn slice_w_step_descr() -> DescrRef {
    field_descr_from_group(&W_SLICE_DESCR_GROUP, 2)
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

/// R3.3b prep: canonical `PyFrame.w_globals_obj` slot
/// (PYFRAME_W_GLOBALS_OBJ_OFFSET).  Used by
/// `emit_new_pyframe_inline_self_recursive` to populate the
/// W_DictObject sibling so trace-time chases observe a non-null
/// PyObjectRef.  R3.3 cutover will fold `pyframe_dict_storage_descr`
/// into this entry after retiring the adjacent raw slot.
pub fn pyframe_w_globals_obj_descr() -> DescrRef {
    field_descr_from_group(&PYFRAME_DESCR_GROUP, 12)
}

/// R3.3-b: `W_ModuleDictObject.dict_storage_proxy` field — a raw
/// `*mut DictStorage` pointer set during module init and stable
/// for the object's lifetime.  Used by `frame_get_namespace` to
/// chase from `w_globals_obj` (a W_ModuleDictObject) to the
/// DictStorage that `load/store_namespace_value` reads.
pub fn module_dict_storage_proxy_descr() -> DescrRef {
    make_immutable_field_descr(
        pyre_object::dictmultiobject::W_MODULE_DICT_STORAGE_PROXY_OFFSET,
        8,
        Type::Ref,
        false,
    )
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

/// PyFrame.execution_context FieldDescr.
/// inline PyFrame 생성 시 caller 의 ec 를 새 frame 으로 SetfieldGc 하기 위해.
/// 호출 사이트는 `helpers.rs::emit_new_pyframe_inline*`.
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
    fn test_array_descr_indices_are_distinct_per_call() {
        // PyPy `descr.py:350-351 cache[ARRAY_OR_STRUCT]` keys on ARRAY
        // object identity; `make_array_descr` callers without an
        // identity carrier (`array_type_id = None`) each produce a
        // distinct ARRAY → distinct `descr_id`.  Singleton semantics
        // require routing through `make_array_descr_with_full_id` with
        // a stable identity string instead.
        let a = make_array_descr(0, 8, None, Type::Int, false);
        let b = make_array_descr(0, 8, None, Type::Int, false);
        let c = make_array_descr(0, 8, None, Type::Ref, false);

        assert_ne!(a.index(), b.index());
        assert_ne!(a.index(), c.index());
        assert_ne!(b.index(), c.index());
    }

    #[test]
    fn test_array_descr_with_full_id_singleton_per_identity() {
        // `descr.py:348-378 get_array_descr` cache hit on
        // `LLType::Array(path_hash(atid))` returns the existing Arc
        // — `make_array_descr_with_full_id` with the same identity
        // string is a singleton.
        let a = crate::descr::make_array_descr_with_full_id(
            0,
            8,
            0,
            None,
            Type::Int,
            false,
            Some("pyre::test_singleton_id".to_string()),
        );
        let b = crate::descr::make_array_descr_with_full_id(
            0,
            8,
            0,
            None,
            Type::Int,
            false,
            Some("pyre::test_singleton_id".to_string()),
        );
        assert!(
            std::sync::Arc::ptr_eq(&a, &b),
            "same identity carrier must collapse to the same Arc"
        );
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
        // descr.py:524-526 `get_result_type()` parity — the raw 'S' char
        // must survive the BhCallDescr -> CallDescr conversion, so
        // downstream consumers can distinguish singlefloat from a real
        // int result.  pyre's `result_class()` returns the raw char
        // (matches `descr.py:526 get_result_type()`); the normalized
        // form per descr.py:527-532 (collapsing 'S' → 'i') is not yet
        // exposed as a separate method but the underlying `result_type`
        // is already `Type::Int`, which is the normalized view.
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
            len_offset: Some(0),
            type_id: 42,
            item_type: Type::Ref,
            is_array_of_pointers: false,
            is_array_of_structs: true,
            is_item_signed: false,
            ei_index: u32::MAX,
            array_type_id: None,
            interior_fields,
        });
        let array = descr.as_array_descr().expect("Array BhDescr -> ArrayDescr");

        assert!(array.is_array_of_structs());
        // `type_id` is the dense sequential GC tid allocated by
        // `GcCache::init_array_descr` (analog of `gc.py:544-549
        // GcLLDescr_framework.init_array_descr` + `gctypelayout.py:301-357
        // TypeLayoutBuilder.get_type_id`).  Exact value depends on the
        // global allocator state — test-suite ordering is non-deterministic
        // so we only assert it is non-zero (tid 0 reserved per
        // `gctypelayout.py:328-331`).  The structural identity that
        // round-trips through `BhDescr::Array.type_id` (path_hash payload)
        // lives in `cache_key` (descr.rs:2120-2131), independent of the
        // GC tid.
        assert_ne!(array.type_id(), 0);
        assert_eq!(array.cache_key(), 42);
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

/// CallDescr for `pyre_object::longobject::jit_w_long_fits_int(obj) -> i64`.
/// `rbigint.fits_int()` is not annotated `@jit.elidable` upstream; it is only
/// used as a cannot-raise runtime guard before the elidable `toint()` call.
pub fn make_jit_w_long_fits_int_calldescr() -> DescrRef {
    majit_ir::make_call_descr(
        vec![Type::Ref],
        Type::Int,
        majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    )
}

/// CallDescr for `pyre_object::longobject::jit_w_long_toint(obj) -> i64`.
/// `W_LongObject.toint()` (longobject.py:138) → `rbigint.toint()`
/// (rbigint.py:465) — `EF_ELIDABLE_CANNOT_RAISE` because the caller
/// emits a fits_int GUARD_TRUE before invoking; OverflowError is
/// statically unreachable post-guard.
pub fn make_jit_w_long_toint_calldescr() -> DescrRef {
    majit_ir::make_call_descr(
        vec![Type::Ref],
        Type::Int,
        majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::ElidableCannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    )
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

/// `descr.py:108-118 get_size_descr` cache parity.
///
/// PyPy `gc_cache._cache_size[STRUCT]` keys on the **STRUCT object
/// identity**, not on its layout — two distinct RPython STRUCTs that
/// share `(size, vtable, fieldlist)` get distinct `SizeDescr` Arcs.
/// Pyre's analogue of "STRUCT identity" is `BhSizeSpec.type_id`
/// (`jit_struct.rs:__majit_type_id` → `path_hash(module_path::TypeName)`):
/// every struct type has a unique `type_id`, and two RPython STRUCTs
/// with coincidentally-identical layout end up with distinct
/// `type_id`s.  Keying the cache on `type_id` alone matches PyPy's
/// per-type identity, where structural-equality keying (the prior
/// `BhSizeSpec`-by-value variant) would have collapsed identity for
/// layout-coincident-but-logically-distinct structs.
///
/// `spec.type_id == 0` is the legacy fallback path
/// (`assembler.rs:2244 bh_size_spec_from_callcontrol` stamps zero
/// when the analyzer-time callcontrol has no host-type carrier).
/// Without a STRUCT-identity carrier we MUST NOT key the cache by
/// the zero sentinel — different STRUCTs with `type_id == 0` would
/// alias onto the first one inserted (`or_insert` "first wins"),
/// silently mixing their field tables.  PyPy's `_cache_size[STRUCT]`
/// never aliases distinct STRUCTs; absent a real identity carrier,
/// the closest orthodox behaviour is "each call is a distinct
/// STRUCT" — mint fresh per call.
static SIMPLE_DESCR_GROUP_CACHE: std::sync::OnceLock<
    std::sync::Mutex<majit_ir::VecAssoc<u64, majit_ir::descr::SimpleDescrGroup>>,
> = std::sync::OnceLock::new();

fn simple_descr_group_from_bh_size(
    spec: &majit_translate::jitcode::BhSizeSpec,
) -> majit_ir::descr::SimpleDescrGroup {
    let mint = || -> majit_ir::descr::SimpleDescrGroup {
        let field_specs: Vec<_> = spec
            .all_fielddescrs
            .iter()
            .map(simple_field_spec_from_bh)
            .collect();
        // `descr.py:108-118 get_size_descr` + `:218-239 get_field_descr`
        // keyed publish: `spec.type_id` is the u64 `path_hash` cache
        // key matching the runtime macro's `__majit_type_id`.  Route
        // through the keyed factory so analyzer-side `cc.fielddescrof`
        // lookups (via `gc_cache.get_field_descr(LLType::Struct(key),
        // name, ...)`) resolve to the same Arc this mint produces —
        // restoring PyPy `cpu.fielddescrof` per-`(STRUCT, name)`
        // identity.  The u32 truncation for the SimpleSizeDescr's gc
        // tid is a TODO (the tid is allocated by
        // gc_cache.init_size_descr in the canonical path; this factory
        // bypasses that, so the tid stays a path_hash-derived u32 with
        // birthday-paradox collision risk around 2^16 distinct STRUCTs).
        majit_ir::descr::make_simple_descr_group_keyed(
            u32::MAX,
            spec.size,
            spec.type_id as u32,
            spec.type_id,
            spec.vtable,
            &field_specs,
        )
    };

    if spec.type_id == 0 {
        // No STRUCT-identity carrier — mint fresh per call so distinct
        // type_id-less STRUCTs don't collapse onto the first-inserted
        // descr group.  Per-STRUCT caching kicks in only when callers
        // route through a real `type_id` source.
        return mint();
    }

    let cache =
        SIMPLE_DESCR_GROUP_CACHE.get_or_init(|| std::sync::Mutex::new(majit_ir::VecAssoc::new()));
    {
        let cache = cache.lock().unwrap();
        if let Some(group) = cache.get(&spec.type_id) {
            return group.clone();
        }
    }
    let group = mint();
    let mut cache = cache.lock().unwrap();
    cache.entry_or_insert_with(spec.type_id, || group).clone()
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
    /// `effectinfo.py:526` `descr.ei_index = …` parity — delegate to
    /// the inner `SimpleFieldDescr`'s atomic so `compute_bitstrings`'s
    /// `set_ei_index` write reaches the same storage that
    /// `heap.rs::field_effect_index` reads through any cloned wrapper.
    fn get_ei_index(&self) -> u32 {
        self.field.get_ei_index()
    }
    fn set_ei_index(&self, index: u32) {
        self.field.set_ei_index(index);
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
        // `descr.py:218-239 get_field_descr` cache-hit: when the parent
        // STRUCT is published in `_cache_size`, walk its
        // `all_fielddescrs()` directly so the returned Arc is the
        // runtime `PyreFieldDescr` (or analyzer-published
        // `SimpleFieldDescr`).  Both back-reference the same parent
        // SizeDescr via `parent_descr` (descr.py:200), so the
        // `ParentBackedFieldDescr` wrapper is unnecessary on this path
        // — analyzer raw-set Arcs and runtime allocator descrs share
        // one identity slot.
        if parent.type_id != 0 {
            let key = majit_ir::descr::LLType::Struct(parent.type_id);
            let parent_size = majit_ir::descr::gc_cache()
                .lock()
                .unwrap()
                ._cache_size
                .get(&key)
                .cloned();
            if let Some(parent_descr) = parent_size {
                if let Some(parent_sd) = parent_descr.as_size_descr() {
                    for fd in parent_sd.all_fielddescrs() {
                        if fd.index_in_parent() == field.index_in_parent
                            && (fd.field_name() == field.name
                                || fd.field_name().ends_with(&format!(".{}", field.name)))
                        {
                            return fd.clone() as DescrRef;
                        }
                    }
                }
            }
        }
        // Cache miss / non-keyed parent — fall back to the legacy
        // SimpleDescrGroup mint + `ParentBackedFieldDescr` wrapper so
        // the cyclic parent_descr Weak still binds correctly.
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
    let arc: DescrRef = Arc::new(descr);
    // descr.py:225-235 `get_field_descr` cache-miss path — register the
    // freshly-minted field descr so `compute_bitstrings` enumerates it.
    majit_ir::descr_registry::register_field(arc.clone());
    arc
}

/// Keyed sibling: accepts the u64 `cache_key` (= `path_hash(array_type_id)`)
/// so the freshly-minted `SimpleArrayDescr` lands in
/// `gc_cache._cache_array[LLType::Array(cache_key)]` in addition to
/// the snapshot order Vec.  Mirrors PyPy `cpu.arraydescrof(ARRAY)`
/// per-ARRAY cache identity (`descr.py:348-378`).  `cache_key == 0`
/// is the no-identity sentinel — registers via the non-keyed path.
pub fn make_struct_array_descr_full_keyed(
    descr_index: u32,
    base_size: usize,
    item_size: usize,
    len_offset: Option<usize>,
    type_id: u32,
    cache_key: u64,
    item_type: Type,
    interior_fields: &[majit_translate::jitcode::BhInteriorFieldSpec],
) -> DescrRef {
    use majit_ir::descr::{ArrayFlag, LLType, SimpleArrayDescr, gc_cache, try_downcast_arc};
    // `descr.py:348-378 get_array_descr(gccache, ARRAY)` cache-or-mint:
    // an `LLType::Array(cache_key)` cache hit returns the existing Arc
    // (the `SimpleArrayDescr` in the slot — from a prior analyzer call
    // or from a runtime mint via `make_array_descr_with_full_id`); only
    // a miss mints a fresh
    // descr.  Matches PyPy `cpu.arraydescrof(ARRAY)` per-ARRAY object
    // identity — both pyre runtime mint sites and analyzer share a
    // single Arc per cache key.  `cache_key == 0` is the no-identity
    // sentinel (legacy non-keyed callers) — mint locally, no cache
    // publication.
    let array_descr_dyn: DescrRef = if cache_key != 0 {
        let array_key = LLType::Array(cache_key);
        let cached = gc_cache().lock().unwrap().get_array_descr(
            array_key.clone(),
            base_size,
            item_size,
            ArrayFlag::Struct,
            item_type,
            len_offset.is_none(),
            len_offset.unwrap_or(0),
            false,
            '\x00',
        );
        // PyPy `gc.py:544-549 init_array_descr` stamps `descr.tid`
        // from `layoutbuilder.get_type_id(A)` — a dense sequential
        // GC type id.  Pyre does not yet port the layoutbuilder
        // analog, so the cache-hit branch only
        // updates the per-trace `descr_index` and leaves
        // `SimpleArrayDescr.type_id` at its mint default (0, set in
        // `get_array_descr` at descr.rs:515).  The
        // `BhDescr::Array.type_id` payload threaded through this
        // helper is the producer-side `path_hash(array_type_id)` and
        // already lands in `SimpleArrayDescr.cache_key` via the
        // `get_array_descr` cache-miss-mint stamp at descr.rs:526-528
        // — structural identity (`cache_key`) is decoupled from GC tid
        // (`type_id`) per the trait doc at descr.rs:2120-2131.  Runtime
        // registrations (`SimpleArrayDescr` from the runtime mint
        // factories) carry their real GC tid at mint and win the cache
        // slot.
        cached.set_index(descr_index);
        cached
    } else {
        // No cache identity — local mint.  Two `cache_key == 0`
        // entries are intentionally distinct STRUCTs sharing the
        // no-identity sentinel; per-`make_array_descr` legacy callers
        // rely on this.
        let mut raw_array_descr = SimpleArrayDescr::with_flag(
            descr_index,
            base_size,
            item_size,
            type_id,
            item_type,
            ArrayFlag::Struct,
        );
        raw_array_descr.lendescr = maybe_array_lendescr_at_offset(len_offset);
        let arc: DescrRef = Arc::new(raw_array_descr);
        majit_ir::descr_registry::register_array(arc.clone());
        arc
    };
    if interior_fields.is_empty() {
        return array_descr_dyn;
    }

    // Upcast the cached array descr to `Arc<dyn ArrayDescr>` for
    // `SimpleInteriorFieldDescr.array_descr` storage.  The cache slot
    // always holds a concrete `SimpleArrayDescr` (analyzer mint,
    // gc_cache internal mint, or runtime mint); downcast to the
    // concrete Arc type, then upcast.
    let array_descr_for_interior: Arc<dyn majit_ir::descr::ArrayDescr> =
        try_downcast_arc::<SimpleArrayDescr>(array_descr_dyn.clone())
            .expect("array descriptor cache must hold SimpleArrayDescr for struct arrays");

    let mut descrs: Vec<DescrRef> = Vec::new();
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
            // `descr.py:423-438 get_interiorfield_descr` cache-or-mint
            // is keyed on the outer ARRAY's lltype identity.  When the
            // outer array carries `cache_key != 0`, route through the
            // keyed `_cache_interiorfield[(LLType::Array(cache_key),
            // name, "")]` so both analyzer and runtime share one Arc
            // per `(ARRAY, name)` tuple.  With `cache_key == 0`
            // (no-identity outer array) PyPy has NO "merge several
            // ARRAYs' interiors into one slot" behavior — local mint
            // a fresh `SimpleInteriorFieldDescr` per call so distinct
            // no-identity arrays do not alias on their interior field
            // descrs.
            //
            // Bare interior field name (`spec.name`) is the cache key per
            // `descr.py:221 cache[STRUCT][fieldname]` shape.
            let bare_name = interior
                .field
                .name
                .rsplit_once('.')
                .map(|(_, n)| n.to_string())
                .unwrap_or_else(|| interior.field.name.clone());
            let field_dyn: Arc<dyn majit_ir::descr::FieldDescr> = field_descr.clone();
            let ifd: DescrRef = if cache_key != 0 {
                gc_cache().lock().unwrap().get_interiorfield_descr(
                    LLType::Array(cache_key),
                    bare_name,
                    String::new(),
                    array_descr_for_interior.clone(),
                    field_dyn,
                )
            } else {
                Arc::new(majit_ir::descr::SimpleInteriorFieldDescr::new(
                    u32::MAX,
                    array_descr_for_interior.clone(),
                    field_dyn,
                )) as DescrRef
            };
            // Per-trace `interior.index` stamp matches the analyzer's
            // `cc.interiorfielddescrof` codewriter idx convention.
            ifd.set_index(interior.index);
            descrs.push(ifd);
        }
    }

    // `descr.py:372-375 arraydescr.all_interiorfielddescrs = descrs`
    // set-once via OnceLock.  Cache-hit case: a prior populator already
    // set the list; our set is a no-op which is the desired semantic.
    array_descr_for_interior.set_all_interiorfielddescrs(descrs);
    array_descr_dyn
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
/// `SimpleArrayDescr` / `PyreSizeDescr`; `Call` becomes a
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
            len_offset,
            type_id,
            item_type,
            is_array_of_structs,
            is_item_signed,
            ei_index,
            array_type_id,
            interior_fields,
            ..
        } => {
            let descr = if *is_array_of_structs {
                // `descr.py:348-378 get_array_descr(gccache, ARRAY)`:
                // the u64 `*type_id` from `BhDescr::Array` is the cache
                // key (`path_hash` of the producer-side `array_type_id`,
                // see `BhSizeSpec.type_id` doc); thread it into the
                // keyed factory so `gc_cache._cache_array[LLType::Array(
                // cache_key)]` is populated and subsequent lookups
                // resolve to the same Arc.  The u32 truncation for the
                // SimpleArrayDescr gc tid is a TODO
                // (gc tid should come from `init_array_descr`
                // sequential allocation).
                make_struct_array_descr_full_keyed(
                    u32::MAX,
                    *base_size,
                    *itemsize,
                    *len_offset,
                    *type_id as u32,
                    *type_id,
                    *item_type,
                    interior_fields,
                )
            } else {
                // `descr.py:348-360 gccache._cache_array[ARRAY_OR_STRUCT]`
                // is keyed on lltype object identity; thread the
                // codewriter `array_type_id` across the BhDescr
                // boundary into the runtime `ArrayDescrKey` so two
                // BhDescr::Array entries that disagree only on the
                // Rust type spelling don't collapse to the same
                // registry slot (`set_ei_index` clobber).
                make_array_descr_with_full_id(
                    *base_size,
                    *itemsize,
                    // TODO: same u32 gc tid truncation.
                    *type_id as u32,
                    *len_offset,
                    *item_type,
                    *is_item_signed,
                    array_type_id.clone(),
                )
            };
            // `effectinfo.py:465 compute_bitstrings` republish: the
            // codewriter-side `array_index` carried across the BhDescr
            // boundary lands on the runtime descr so heap.rs
            // `force_from_effectinfo` (`heap.py:537-571`) reads the
            // same bitstring slot the producer wrote.
            if *ei_index != u32::MAX {
                descr.set_ei_index(*ei_index);
            }
            descr
        }
        BhDescr::Size {
            size,
            type_id,
            vtable,
            all_fielddescrs,
            ..
        } => {
            // `descr.py:108-118 get_size_descr` cache-hit semantics:
            // when the producer's `type_id` matches a runtime publish
            // (`build_object_descr_group_with_def_path` →
            // `register_keyed_size`), the published `Arc<PyreSizeDescr>`
            // is the canonical object identity for that STRUCT.  Return
            // it directly so analyzer side-tables, runtime allocations,
            // and BhDescr round-trip all share one `Arc<dyn Descr>` —
            // matching PyPy `cache[STRUCT]` per-tuple identity.  Falls
            // through to the legacy mint path on cache miss (transient
            // empty-field `bh_new` descrs from `pyre-jit/src/eval.rs`,
            // test fixtures, etc.).
            if *type_id != 0 {
                let key = majit_ir::descr::LLType::Struct(*type_id);
                let hit = majit_ir::descr::gc_cache()
                    .lock()
                    .unwrap()
                    ._cache_size
                    .get(&key)
                    .cloned();
                if let Some(descr) = hit {
                    return descr;
                }
            }
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
                // TODO: `make_size_descr_with_type_and_vtable`
                // takes the u32 gc tid; `*type_id` is the u64 cache key.
                // Truncate `as u32` until gc_cache routing.
                make_size_descr_with_type_and_vtable(*size, *type_id as u32, *vtable)
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
        BhDescr::InteriorField { array, field } => {
            // `descr.py:388 InteriorFieldDescr(arraydescr, fielddescr)`:
            // recompose the array + field sub-descrs.  The nested
            // BhDescrs are always `BhDescr::Array` / `BhDescr::Field`
            // (`BhDescr::from_interior_field_descr`); a short stream is an
            // encoder bug surfaced here rather than silently mis-typed.
            let (base_size, itemsize, len_offset, type_id, item_type, interior_fields) =
                match array.as_ref() {
                    BhDescr::Array {
                        base_size,
                        itemsize,
                        len_offset,
                        type_id,
                        item_type,
                        interior_fields,
                        ..
                    } => (
                        *base_size,
                        *itemsize,
                        *len_offset,
                        *type_id,
                        *item_type,
                        interior_fields,
                    ),
                    other => panic!(
                        "BhDescr::InteriorField array slot must be BhDescr::Array, got {other:?}"
                    ),
                };
            // `descr.py:430` `get_interiorfield_descr` builds the interior
            // descr from `get_array_descr(gc_ll_descr, ARRAY)` — the SAME
            // cached arraydescr the `BhDescr::Array` arm restores.  Route the
            // outer struct array through the identical keyed factory so the
            // rebuilt `InteriorFieldDescr.arraydescr` shares the
            // `gc_cache._cache_array[LLType::Array(cache_key)]` identity rather
            // than a fresh local mint, and re-attaches `all_interiorfielddescrs`
            // (`descr.py:372-375`).  `type_id` is the producer-side cache key
            // (`ArrayDescr.cache_key` = `path_hash(array_type_id)`); the u32
            // gc-tid slot is ignored on the cache path (`init_array_descr`
            // stamps the real tid), keeping ARRAY identity cache key separate
            // from GC layout tid.  On a cache MISS the factory requests
            // `ArrayFlag::Struct`, so the minted arraydescr is FLAG_STRUCT; on a
            // cache HIT the existing slot is reused verbatim, so the resolved
            // arraydescr is re-checked against `descr.py:389`'s
            // `assert arraydescr.flag == FLAG_STRUCT` below.
            let array_dyn = make_struct_array_descr_full_keyed(
                u32::MAX,
                base_size,
                itemsize,
                len_offset,
                type_id as u32,
                type_id,
                item_type,
                interior_fields,
            );
            // The cache slot always holds a concrete `SimpleArrayDescr`
            // (analyzer / gc_cache / runtime mint).
            let array_descr: Arc<dyn majit_ir::descr::ArrayDescr> =
                match majit_ir::descr::try_downcast_arc::<majit_ir::descr::SimpleArrayDescr>(
                    array_dyn,
                ) {
                    Ok(simple) => simple,
                    Err(other) => panic!(
                        "BhDescr::InteriorField array resolved to an unknown ArrayDescr type: {other:?}"
                    ),
                };
            // `descr.py:389 InteriorFieldDescr.__init__`:
            //   assert arraydescr.flag == FLAG_STRUCT
            // The factory requests `ArrayFlag::Struct` on a cache MISS, but a
            // cache HIT returns the existing `LLType::Array(cache_key)` slot
            // verbatim — which may be a non-struct array (a contaminated cache
            // key).  Reject those here rather than silently rebuilding the
            // InteriorFieldDescr around a non-struct array.
            assert!(
                array_descr.is_array_of_structs(),
                "BhDescr::InteriorField arraydescr must be FLAG_STRUCT (descr.py:389)"
            );
            let (offset, field_size, field_type, field_flag, index_in_parent, name) =
                match field.as_ref() {
                    BhDescr::Field {
                        offset,
                        field_size,
                        field_type,
                        field_flag,
                        index_in_parent,
                        name,
                        ..
                    } => (
                        *offset,
                        *field_size,
                        *field_type,
                        *field_flag,
                        *index_in_parent,
                        name.clone(),
                    ),
                    other => panic!(
                        "BhDescr::InteriorField field slot must be BhDescr::Field, got {other:?}"
                    ),
                };
            // Bare interior field name (`descr.py:221 cache[STRUCT][fieldname]`
            // shape) — the `get_interiorfield_descr` cache key the
            // `make_struct_array_descr_full_keyed` interior loop used.
            let bare_name = name
                .rsplit_once('.')
                .map(|(_, n)| n.to_string())
                .unwrap_or_else(|| name.clone());
            // `descr.py:393` `InteriorFieldDescr.get_index()` delegates to
            // `fielddescr.get_index()` (= `index_in_parent`), read back by
            // `info.py:573-594` getinteriorfield/setinteriorfield_virtual.
            let field_descr: Arc<dyn majit_ir::descr::FieldDescr> = Arc::new(
                majit_ir::descr::SimpleFieldDescr::new_with_name(
                    u32::MAX,
                    offset,
                    field_size,
                    field_type,
                    false,
                    field_flag,
                    name,
                )
                .with_index_in_parent(index_in_parent),
            );
            // `descr.py:423-438 get_interiorfield_descr` cache-or-mint keyed on
            // the outer ARRAY identity, so the analyzer's
            // `cc.interiorfielddescrof`, the array's `all_interiorfielddescrs`,
            // and this restore share one `Arc` per `(ARRAY, name)` tuple — the
            // `make_struct_array_descr_full_keyed` interior loop above already
            // populated `_cache_interiorfield` (with the per-trace
            // `Descr::index()`) for `cache_key != 0`, so this hits and returns
            // that same descr.  `cache_key == 0` is the legacy no-identity
            // sentinel — local mint a fresh `SimpleInteriorFieldDescr` (the
            // `u32::MAX` slot is the "no setup_descrs index assigned" default;
            // `get_index()` comes from `field_descr.index_in_parent` above).
            if type_id != 0 {
                majit_ir::descr::gc_cache()
                    .lock()
                    .unwrap()
                    .get_interiorfield_descr(
                        majit_ir::descr::LLType::Array(type_id),
                        bare_name,
                        String::new(),
                        array_descr,
                        field_descr,
                    )
            } else {
                Arc::new(majit_ir::descr::SimpleInteriorFieldDescr::new(
                    u32::MAX,
                    array_descr,
                    field_descr,
                ))
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
    //
    // descr.py:524-526 `get_result_type()` parity — preserve the raw
    // `bh.result_type` char ('i'/'r'/'f'/'v'/'S'/'L') so downstream
    // consumers (`bhimpl_call_*` dispatch, `is_result_signed`) can
    // recover the original singlefloat/longlong classification that the
    // normalized `Type` collapses.
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
    majit_ir::descr_registry::register_array(array_descr.clone() as DescrRef);
    let field_descr = Arc::new(SimpleFieldDescr::new(
        field_descr_index,
        field_offset,
        field_size,
        tp,
        true, // immutable (struct fields in array-of-struct)
    ));
    majit_ir::descr_registry::register_field(field_descr.clone() as DescrRef);
    let interior: DescrRef = Arc::new(SimpleInteriorFieldDescr::new(
        field_descr_index,
        array_descr,
        field_descr,
    ));
    majit_ir::descr_registry::register_interior_field(interior.clone());
    interior
}
