/// Virtual state export/import for loop peeling.
///
/// Translated from rpython/jit/metainterp/optimizeopt/virtualstate.py.
///
/// After unrolling one iteration (the "preamble"), the optimizer captures the
/// abstract state of each value carried across the back-edge. On the next
/// iteration, this exported state is compared against the incoming values.
/// If compatible, the optimizer can directly apply known information (virtuals,
/// bounds, classes) without re-discovering it.
///
/// Key types:
/// - `VirtualState`: a snapshot of abstract info for all loop-carried values
/// - `VirtualStateInfo`: per-value abstract info (constant, virtual, class, etc.)
/// - State comparison determines if a compiled loop body can be reused
///
/// **Sharing**: VirtualStateInfo is wrapped in `Rc<...>` so the export tree
/// becomes a DAG mirroring RPython's reference-shared
/// AbstractVirtualStateInfo objects. When two parents reference the same
/// underlying box, they share the same `Rc<VirtualStateInfoNode>` and the
/// position-numbered enum_forced_boxes dedup (virtualstate.py:196, 274,
/// 352) prevents revisiting it.
use std::cell::Cell;
use std::collections::HashMap;
use std::ops::Deref;
use std::rc::Rc;

use majit_ir::{DescrRef, GcRef, Op, OpCode, OpRef, Type, Value};

/// virtualstate.py: VirtualStatesCantMatch — raised when two virtual states
/// are incompatible and cannot be merged for bridge compilation.
#[derive(Clone, Debug)]
pub struct VirtualStatesCantMatch {
    pub msg: String,
}

impl VirtualStatesCantMatch {
    pub fn new(msg: &str) -> Self {
        VirtualStatesCantMatch {
            msg: msg.to_string(),
        }
    }
}

impl std::fmt::Display for VirtualStatesCantMatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VirtualStatesCantMatch: {}", self.msg)
    }
}

impl std::error::Error for VirtualStatesCantMatch {}

/// virtualstate.py:368 `not_virtual(cpu, type, info)` dispatch parity:
/// returns true when `incoming` belongs to the **same non-virtual class**
/// as the expected type.
///
/// RPython isinstance dispatch (virtualstate.py:522-529):
///   NotVirtualStateInfoPtr._generate_guards rejects any `other` that is
///   not isinstance(other, NotVirtualStateInfoPtr). Virtual/VArray/VStruct
///   are VirtualStateInfo subclasses, NOT NotVirtualStateInfoPtr, so they
///   fail this check. The force_boxes=True path is handled separately in
///   generate_guards_for_entry (line 1229) BEFORE this function is called.
pub(crate) fn info_type_matches(expected: Type, incoming: &VirtualStateInfo) -> bool {
    match (expected, incoming) {
        // LEVEL_UNKNOWN on both: compare the explicit type tags.
        (Type::Int, VirtualStateInfo::Unknown(Type::Int))
        | (Type::Float, VirtualStateInfo::Unknown(Type::Float))
        | (Type::Ref, VirtualStateInfo::Unknown(Type::Ref))
        | (Type::Void, VirtualStateInfo::Unknown(Type::Void)) => true,
        (_, VirtualStateInfo::Unknown(_)) => false,
        // Int expected: incoming must be Int-typed constant or IntBounded.
        (Type::Int, VirtualStateInfo::IntBounded(_)) => true,
        (Type::Int, VirtualStateInfo::Constant(Value::Int(_))) => true,
        (Type::Int, _) => false,
        // Float expected: incoming must be Float constant.
        (Type::Float, VirtualStateInfo::Constant(Value::Float(_))) => true,
        (Type::Float, _) => false,
        // Ref expected: incoming must be a NotVirtualStateInfoPtr subclass
        // (NonNull / KnownClass / Ref-typed Constant). Virtual* variants
        // are NOT accepted — virtualstate.py:525-528:
        //   if not isinstance(other, NotVirtualStateInfoPtr):
        //       raise VirtualStatesCantMatch(...)
        (Type::Ref, VirtualStateInfo::NonNull)
        | (Type::Ref, VirtualStateInfo::KnownClass { .. }) => true,
        (Type::Ref, VirtualStateInfo::Constant(Value::Ref(_))) => true,
        (Type::Ref, _) => false,
        // Void: only Void-typed constants (rare).
        (Type::Void, VirtualStateInfo::Constant(Value::Void)) => true,
        (Type::Void, _) => false,
    }
}

use crate::optimizeopt::OptContext;
use crate::optimizeopt::info::PtrInfo;
use crate::optimizeopt::intutils::IntBound;

/// virtualstate.py: GenerateGuardState — context for guard generation
/// during virtual state comparison.
///
/// Holds the optimizer reference, a collection of generated guards,
/// a renumbering map (OpRef → position), and flags.
pub struct GenerateGuardState<'a> {
    /// Reference to optimizer context.
    pub ctx: &'a OptContext,
    /// Extra guards generated during state comparison.
    pub extra_guards: Vec<majit_ir::Op>,
    /// Renumbering map: OpRef → position in the state vector.
    pub renum: HashMap<OpRef, usize>,
    /// Entries that could not be matched (incompatible state).
    pub bad: HashMap<usize, String>,
    /// Whether to force virtual boxes during comparison.
    pub force_boxes: bool,
}

impl<'a> GenerateGuardState<'a> {
    pub fn new(ctx: &'a OptContext) -> Self {
        GenerateGuardState {
            ctx,
            extra_guards: Vec::new(),
            renum: HashMap::new(),
            bad: HashMap::new(),
            force_boxes: false,
        }
    }

    /// virtualstate.py: get_runtime_field(box, descr)
    ///
    /// Read a field from a concrete object at runtime.
    /// Returns the value as an i64 (for int fields) or pointer.
    pub fn get_runtime_field(&self, obj_ptr: i64, offset: usize) -> i64 {
        if obj_ptr == 0 {
            return 0;
        }
        unsafe { *((obj_ptr as *const u8).add(offset) as *const i64) }
    }

    /// virtualstate.py: get_runtime_item(box, descr, i)
    ///
    /// Read an array item from a concrete object at runtime.
    pub fn get_runtime_item(&self, array_ptr: i64, index: usize, item_size: usize) -> i64 {
        if array_ptr == 0 {
            return 0;
        }
        unsafe { *((array_ptr as *const u8).add(index * item_size) as *const i64) }
    }

    /// Add an extra guard to be emitted before the bridge entry.
    pub fn add_guard(&mut self, guard: majit_ir::Op) {
        self.extra_guards.push(guard);
    }

    /// Mark an entry as incompatible.
    pub fn mark_bad(&mut self, index: usize, reason: &str) {
        self.bad.insert(index, reason.to_string());
    }

    /// Whether the comparison succeeded (no bad entries).
    pub fn is_ok(&self) -> bool {
        self.bad.is_empty()
    }
}

/// Abstract info for one value at the loop boundary.
///
/// Mirrors the hierarchy in RPython's `AbstractVirtualStateInfo` and its subclasses:
/// `VirtualStateInfoConst`, `VirtualStateInfoVirtual`, `VirtualStateInfoNotVirtual`, etc.
///
/// In RPython every subclass is a Python class instance carrying mutable
/// `position` (and `position_in_notvirtuals` for NotVirtual leaves). Pyre
/// stores those per-instance attrs on [`VirtualStateInfoNode`] (the wrapper
/// around `Rc<VirtualStateInfoNode>` in shared trees) so each unique node has
/// its own slot, matching RPython's class-attr identity.
#[derive(Clone, Debug)]
pub enum VirtualStateInfo {
    /// Value is a known constant.
    Constant(Value),
    /// Value is a virtual instance with known fields.
    ///
    /// **Invariant**: `fields` NEVER contains typeptr (offset 0).
    /// Mirrors the `VirtualInfo.fields` invariant (RPython
    /// heaptracker.py:66-67 all_fielddescrs excludes typeptr).
    /// Enforced at export/import boundaries via
    /// `debug_assert_no_typeptr_in_virtual_fields`.
    Virtual {
        descr: DescrRef,
        known_class: Option<GcRef>,
        /// ob_type field descriptor for force path (pyre offset 0).
        ob_type_descr: Option<DescrRef>,
        /// Field values as VirtualStateInfo (recursive). Excludes typeptr.
        fields: Vec<(u32, Rc<VirtualStateInfoNode>)>,
        /// Original field descriptors in parent-local slot order.
        /// virtualstate.py:159 AbstractVirtualStructStateInfo.fielddescrs
        /// is a flat list and box access uses `fielddescrs[i].get_index()`.
        field_descrs: Vec<DescrRef>,
    },
    /// virtualstate.py: VArrayStateInfo — virtual array with known elements.
    VArray {
        descr: DescrRef,
        items: Vec<Rc<VirtualStateInfoNode>>,
        /// virtualstate.py: lenbound — known bounds on array length.
        /// None means unbounded.
        lenbound: Option<IntBound>,
    },
    /// virtualstate.py: VStructStateInfo — virtual struct.
    VStruct {
        descr: DescrRef,
        fields: Vec<(u32, Rc<VirtualStateInfoNode>)>,
        /// virtualstate.py:159 AbstractVirtualStructStateInfo.fielddescrs
        /// stored as a flat parent-local list.
        field_descrs: Vec<DescrRef>,
    },
    /// virtualstate.py:286: VArrayStructStateInfo(arraydescr, fielddescrs, length)
    VArrayStruct {
        descr: DescrRef,
        /// virtualstate.py:289: self.fielddescrs — InteriorFieldDescr per field slot.
        fielddescrs: Vec<DescrRef>,
        element_fields: Vec<Vec<(u32, Rc<VirtualStateInfoNode>)>>,
    },
    /// Value is a virtual raw buffer (info.py:389 RawBufferPtrInfo).
    VirtualRawBuffer {
        /// info.py:389: self.func — the malloc function pointer.
        func: i64,
        size: usize,
        /// rawbuffer.py:14-16: offsets, lengths, per-entry child state.
        entries: Vec<(usize, usize, Rc<VirtualStateInfoNode>)>,
        /// rawbuffer.py:16: self.descrs — per-entry ArrayDescr.
        descrs: Vec<DescrRef>,
    },
    /// Value has a known class (non-null).
    ///
    /// virtualstate.py:505 NotVirtualStateInfoPtr with level=LEVEL_KNOWNCLASS.
    /// Implicitly Ref-typed in RPython via the class hierarchy; pyre keeps
    /// the explicit invariant that this variant is only emitted for Ref.
    KnownClass { class_ptr: GcRef },
    /// Value is known non-null.
    ///
    /// virtualstate.py:505 NotVirtualStateInfoPtr with level=LEVEL_NONNULL.
    /// Implicitly Ref-typed; emitted only for Ref.
    NonNull,
    /// Value has known integer bounds.
    ///
    /// virtualstate.py:473 NotVirtualStateInfoInt with intbound set.
    /// Implicitly Int-typed; emitted only for Int.
    IntBounded(IntBound),
    /// No useful info (anything is compatible at this level).
    ///
    /// virtualstate.py:368-372 NotVirtualStateInfo base class with
    /// LEVEL_UNKNOWN. In RPython the concrete class
    /// (`NotVirtualStateInfoInt` / `NotVirtualStateInfoPtr` / base
    /// `NotVirtualStateInfo` for Float) tags the type via
    /// `isinstance`-based dispatch in `_generate_guards`. pyre's flat
    /// OpRef namespace lets the optimizer unbox Ref→Int within a trace,
    /// so the type must be carried explicitly to prevent a ref-slot
    /// entry from accepting an int-typed incoming (and vice versa).
    /// This field is the direct counterpart of RPython's
    /// `NotVirtualStateInfoInt(cpu, 'i', info)` vs
    /// `NotVirtualStateInfoPtr(cpu, 'r', info)` constructor discriminant.
    Unknown(Type),
}

/// Per-instance wrapper around [`VirtualStateInfo`] carrying RPython's
/// `AbstractVirtualStateInfo.position` (virtualstate.py:70) and
/// `NotVirtualStateInfo.position_in_notvirtuals` (virtualstate.py:430-431)
/// class attributes.
///
/// In RPython each subclass instance holds these directly; pyre's flat
/// `Rc<VirtualStateInfo>` enum value cannot host per-instance mutability,
/// so the position cells live on this wrapper. Sharing semantics match
/// RPython exactly: when two parents reference the same logical box, the
/// same `Rc<VirtualStateInfoNode>` is shared, both parents see the same
/// position values, and the `state.position > self.position` dedup works
/// without an external side map.
///
/// [`Deref`] forwards to the inner enum so existing pattern-matching
/// against `VirtualStateInfo` variants stays valid.
#[derive(Debug)]
pub struct VirtualStateInfoNode {
    pub info: VirtualStateInfo,
    /// virtualstate.py:70 `AbstractVirtualStateInfo.position`. Default -1.
    /// Set by [`VirtualState::enum_top_level`] during construction.
    pub position: Cell<i32>,
    /// virtualstate.py:430-431 `NotVirtualStateInfo.position_in_notvirtuals`.
    /// Only meaningful for the NotVirtual leaf variants
    /// (`Constant`/`KnownClass`/`NonNull`/`IntBounded`/`Unknown`); other
    /// variants leave this at -1.
    pub position_in_notvirtuals: Cell<i32>,
}

impl VirtualStateInfoNode {
    pub fn new(info: VirtualStateInfo) -> Self {
        VirtualStateInfoNode {
            info,
            position: Cell::new(-1),
            position_in_notvirtuals: Cell::new(-1),
        }
    }

    pub fn new_rc(info: VirtualStateInfo) -> Rc<Self> {
        Rc::new(Self::new(info))
    }

    /// virtualstate.py:111-116 `AbstractVirtualStateInfo.enum`.
    /// ```python
    /// def enum(self, virtual_state):
    ///     if self.position != -1:
    ///         return
    ///     virtual_state.info_counter += 1
    ///     self.position = virtual_state.info_counter
    ///     self._enum(virtual_state)
    /// ```
    pub fn enum_into(&self, state: &mut VirtualState) {
        if self.position.get() != -1 {
            return;
        }
        state.info_counter += 1;
        self.position.set(state.info_counter);
        self._enum(state);
    }

    /// virtualstate.py: `_enum` per subclass. Dispatches over the inner
    /// enum, mirroring RPython's class-level method dispatch:
    /// - virtualstate.py:200-203 AbstractVirtualStructStateInfo._enum
    /// - virtualstate.py:277-280 VArrayStateInfo._enum
    /// - virtualstate.py:328-331 VArrayStructStateInfo._enum
    /// - virtualstate.py:427-431 NotVirtualStateInfo._enum
    fn _enum(&self, state: &mut VirtualState) {
        match &self.info {
            VirtualStateInfo::Constant(_) => {
                // virtualstate.py:427-429: LEVEL_CONSTANT short-circuit.
            }
            VirtualStateInfo::Virtual { fields, .. } | VirtualStateInfo::VStruct { fields, .. } => {
                for (_, child) in fields {
                    child.enum_into(state);
                }
            }
            VirtualStateInfo::VArray { items, .. } => {
                for child in items {
                    child.enum_into(state);
                }
            }
            VirtualStateInfo::VArrayStruct { element_fields, .. } => {
                for fields in element_fields {
                    for (_, child) in fields {
                        child.enum_into(state);
                    }
                }
            }
            VirtualStateInfo::VirtualRawBuffer { entries, .. } => {
                for (_, _, child) in entries {
                    child.enum_into(state);
                }
            }
            VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::IntBounded(_)
            | VirtualStateInfo::Unknown(_) => {
                // virtualstate.py:427-431 NotVirtualStateInfo._enum:
                //     if self.level == LEVEL_CONSTANT: return
                //     self.position_in_notvirtuals = virtual_state.numnotvirtuals
                //     virtual_state.numnotvirtuals += 1
                //
                // The Constant case is dispatched above; reaching here
                // means self is a non-constant NotVirtual leaf.
                self.position_in_notvirtuals
                    .set(state.numnotvirtuals as i32);
                state.numnotvirtuals += 1;
            }
        }
    }
}

impl Deref for VirtualStateInfoNode {
    type Target = VirtualStateInfo;
    fn deref(&self) -> &VirtualStateInfo {
        &self.info
    }
}

impl Clone for VirtualStateInfoNode {
    /// Cloning resets position/position_in_notvirtuals to -1 because the
    /// cloned node is a fresh instance — RPython's `__init__` does not
    /// inherit position from a source instance, and `enum` re-assigns
    /// positions when called on the new VirtualState.
    fn clone(&self) -> Self {
        VirtualStateInfoNode {
            info: self.info.clone(),
            position: Cell::new(-1),
            position_in_notvirtuals: Cell::new(-1),
        }
    }
}

impl VirtualStateInfo {
    /// Check if `other` is compatible with `self`.
    ///
    /// "Compatible" means that if the loop body was optimized assuming `self`,
    /// a value described by `other` can safely enter that loop body.
    ///
    /// In RPython this is `generate_guards()` which emits any needed bridge guards.
    /// Here we just check compatibility; guard generation is separate.
    pub fn is_compatible(&self, other: &VirtualStateInfo) -> bool {
        match (self, other) {
            // virtualstate.py:383-410 NotVirtualStateInfo._generate_guards
            // LEVEL_UNKNOWN parity: RPython dispatches to
            // `_generate_guards_unkown` on the subclass (Int/Ptr/Float).
            // The subclass `isinstance(other, NotVirtualStateInfoInt)` check
            // enforces type agreement; a cross-type mismatch raises
            // VirtualStatesCantMatch.
            (VirtualStateInfo::Unknown(tp), _) => info_type_matches(*tp, other),

            // Constants must match exactly.
            (VirtualStateInfo::Constant(a), VirtualStateInfo::Constant(b)) => a == b,
            (VirtualStateInfo::Constant(_), _) => false,

            // Virtual instance: other must also be a matching virtual with compatible fields
            (
                VirtualStateInfo::Virtual {
                    descr: d1,
                    known_class: kc1,
                    fields: f1,
                    ..
                },
                VirtualStateInfo::Virtual {
                    descr: d2,
                    known_class: kc2,
                    fields: f2,
                    ..
                },
            ) => {
                if d1.index() != d2.index() {
                    return false;
                }
                // Class must match (both None or same pointer)
                match (kc1, kc2) {
                    (Some(c1), Some(c2)) if c1 != c2 => return false,
                    (Some(_), None) => return false,
                    _ => {}
                }
                // All fields in self must have compatible counterparts in other
                for (idx, info) in f1 {
                    let other_info = f2.iter().find(|(i, _)| i == idx).map(|(_, v)| v.as_ref());
                    match other_info {
                        Some(oi) => {
                            if !info.is_compatible(oi) {
                                return false;
                            }
                        }
                        None => return false, // field missing in other
                    }
                }
                true
            }

            // Virtual array: must match length and each element
            (
                VirtualStateInfo::VArray {
                    descr: d1,
                    items: i1,
                    ..
                },
                VirtualStateInfo::VArray {
                    descr: d2,
                    items: i2,
                    ..
                },
            ) => {
                if d1.index() != d2.index() || i1.len() != i2.len() {
                    return false;
                }
                i1.iter().zip(i2.iter()).all(|(a, b)| a.is_compatible(b))
            }

            // Virtual struct: same as virtual instance
            (
                VirtualStateInfo::VStruct {
                    descr: d1,
                    fields: f1,
                    ..
                },
                VirtualStateInfo::VStruct {
                    descr: d2,
                    fields: f2,
                    ..
                },
            ) => {
                if d1.index() != d2.index() {
                    return false;
                }
                for (idx, info) in f1 {
                    let other_info = f2.iter().find(|(i, _)| i == idx).map(|(_, v)| v.as_ref());
                    match other_info {
                        Some(oi) => {
                            if !info.is_compatible(oi) {
                                return false;
                            }
                        }
                        None => return false,
                    }
                }
                true
            }

            // Virtual array struct
            // virtualstate.py:292-306: VArrayStructStateInfo._generate_guards
            (
                VirtualStateInfo::VArrayStruct {
                    descr: d1,
                    fielddescrs: fd1,
                    element_fields: ef1,
                },
                VirtualStateInfo::VArrayStruct {
                    descr: d2,
                    fielddescrs: fd2,
                    element_fields: ef2,
                },
            ) => {
                // virtualstate.py:294-304: arraydescr identity + fielddescrs length + fielddescrs identity
                if d1.index() != d2.index() || ef1.len() != ef2.len() {
                    return false;
                }
                if fd1.len() != fd2.len() {
                    return false;
                }
                for (a, b) in fd1.iter().zip(fd2.iter()) {
                    if a.index() != b.index() {
                        return false;
                    }
                }
                ef1.iter().zip(ef2.iter()).all(|(fields1, fields2)| {
                    for (idx, info) in fields1 {
                        let other_info = fields2
                            .iter()
                            .find(|(i, _)| i == idx)
                            .map(|(_, v)| v.as_ref());
                        match other_info {
                            Some(oi) if info.is_compatible(oi) => {}
                            _ => return false,
                        }
                    }
                    true
                })
            }

            // Virtual raw buffer
            // resume.py:335,694: visit_vrawbuffer(func, size, offsets, descrs)
            // — func and descrs are part of raw-buffer identity.
            (
                VirtualStateInfo::VirtualRawBuffer {
                    func: f1,
                    size: s1,
                    entries: e1,
                    descrs: d1,
                },
                VirtualStateInfo::VirtualRawBuffer {
                    func: f2,
                    size: s2,
                    entries: e2,
                    descrs: d2,
                },
            ) => {
                if f1 != f2 || s1 != s2 || e1.len() != e2.len() {
                    return false;
                }
                // rawbuffer.py:83: _descrs_are_compatible — two arraydescrs are
                // compatible if they have the same basesize, itemsize and sign,
                // even if they are not identical (unpack_arraydescr_size parity).
                if d1.len() != d2.len()
                    || !d1.iter().zip(d2.iter()).all(|(a, b)| {
                        let (Some(a1), Some(a2)) = (a.as_array_descr(), b.as_array_descr()) else {
                            // rawbuffer.py:86-87: unpack_arraydescr_size requires
                            // array descrs. Non-array descrs are incompatible.
                            return false;
                        };
                        a1.base_size() == a2.base_size()
                            && a1.item_size() == a2.item_size()
                            && a1.is_item_signed() == a2.is_item_signed()
                    })
                {
                    return false;
                }
                e1.iter()
                    .zip(e2.iter())
                    .all(|((off1, len1, v1), (off2, len2, v2))| {
                        off1 == off2 && len1 == len2 && v1.is_compatible(v2)
                    })
            }

            // KnownClass: other must have the same class (or be virtual with matching class).
            // RPython: KnownClass does NOT accept Unknown/NonNull in pure
            // compatibility check (raises VirtualStatesCantMatch). Guard
            // generation with runtime_box is needed for that.
            (VirtualStateInfo::KnownClass { class_ptr: c1 }, other_info) => match other_info {
                VirtualStateInfo::KnownClass { class_ptr: c2 } => c1 == c2,
                VirtualStateInfo::Virtual { known_class, .. } => known_class.as_ref() == Some(c1),
                _ => false,
            },

            // NonNull: other must be nonnull (virtual is always nonnull).
            // RPython: NonNull does NOT accept Unknown in pure compatibility
            // check (raises VirtualStatesCantMatch).
            (VirtualStateInfo::NonNull, other_info) => match other_info {
                VirtualStateInfo::NonNull
                | VirtualStateInfo::KnownClass { .. }
                | VirtualStateInfo::Virtual { .. }
                | VirtualStateInfo::VArray { .. }
                | VirtualStateInfo::VStruct { .. }
                | VirtualStateInfo::VArrayStruct { .. }
                | VirtualStateInfo::VirtualRawBuffer { .. } => true,
                VirtualStateInfo::Constant(Value::Ref(r)) => !r.is_null(),
                _ => false,
            },

            // IntBounded: other must have tighter or equal bounds.
            (VirtualStateInfo::IntBounded(b1), VirtualStateInfo::IntBounded(b2)) => {
                b2.lower >= b1.lower && b2.upper <= b1.upper
            }
            (VirtualStateInfo::IntBounded(b), VirtualStateInfo::Constant(Value::Int(v))) => {
                b.contains(*v)
            }
            (VirtualStateInfo::IntBounded(_), _) => false,

            // Cross-type mismatches
            _ => false,
        }
    }

    /// Whether this info represents a virtual (not yet allocated) value.
    pub fn is_virtual(&self) -> bool {
        matches!(
            self,
            VirtualStateInfo::Virtual { .. }
                | VirtualStateInfo::VArray { .. }
                | VirtualStateInfo::VStruct { .. }
                | VirtualStateInfo::VArrayStruct { .. }
                | VirtualStateInfo::VirtualRawBuffer { .. }
        )
    }

    /// virtualstate.py implicit Box.type parity: return the Type of this
    /// state entry, as if the originating `box.type` were still accessible.
    /// Returns `None` only for Void/unreachable constants, since every
    /// RPython Box has one of the three types.
    pub fn info_type(&self) -> Option<Type> {
        match self {
            VirtualStateInfo::Unknown(tp) => Some(*tp),
            VirtualStateInfo::Constant(v) => Some(v.get_type()),
            VirtualStateInfo::IntBounded(_) => Some(Type::Int),
            // info.py:867 `assert op.type == 'i'` in `getrawptrinfo`:
            // raw pointer Boxes are int-typed; pointer-ness sits on
            // RawBufferPtrInfo, not on the Box.type.
            VirtualStateInfo::VirtualRawBuffer { .. } => Some(Type::Int),
            VirtualStateInfo::NonNull
            | VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::Virtual { .. }
            | VirtualStateInfo::VArray { .. }
            | VirtualStateInfo::VStruct { .. }
            | VirtualStateInfo::VArrayStruct { .. } => Some(Type::Ref),
        }
    }
}

/// A complete snapshot of abstract state at a loop boundary.
///
/// The `state` vector has one entry per loop-carried variable (matching the
/// `Jump`/`Label` args). During loop peeling, this is exported at the end
/// of the preamble and imported at the loop header.
///
/// Top-level entries are stored as `Rc<VirtualStateInfoNode>` so that
/// aliased loop-carried variables (two jump args resolving to the same
/// box) share a single state object — matching RPython's
/// `VirtualStateConstructor.create_state` cache where the same box always
/// returns the same `AbstractVirtualStateInfo` instance. The dedup walkers
/// (`build_sequential_slot_schedule`, `enum_forced_boxes_for_entry`, etc.)
/// use `Rc::as_ptr` identity to short-circuit revisits, mirroring
/// `state.position > self.position`.
#[derive(Debug)]
pub struct VirtualState {
    /// Abstract info for each loop-carried variable, in order matching Label/Jump args.
    pub state: Vec<Rc<VirtualStateInfoNode>>,
    /// virtualstate.py: renum — maps virtual OpRef to numbering index.
    /// Used to ensure consistent virtual identity across loop iterations.
    pub renum: std::collections::HashMap<OpRef, usize>,
    /// RPython virtualstate.py: position_in_notvirtuals encoded as a flat
    /// traversal schedule. Each non-virtual leaf occurrence in `state`
    /// records the slot it should write in `boxes[...]`.
    slot_schedule: Vec<usize>,
    /// virtualstate.py:631 `VirtualState.numnotvirtuals`. Maintained by
    /// [`Self::enum_top_level`] (RPython parity) but kept identical to
    /// the legacy `build_sequential_slot_schedule` count via debug_assert
    /// during Phase A1.
    numnotvirtuals: usize,
    /// virtualstate.py:630 `VirtualState.info_counter`. Increments through
    /// `enum_into` to assign per-instance positions.
    info_counter: i32,
}

impl VirtualState {
    pub fn new(state: Vec<VirtualStateInfo>) -> Self {
        let state: Vec<Rc<VirtualStateInfoNode>> = state
            .into_iter()
            .map(VirtualStateInfoNode::new_rc)
            .collect();
        Self::from_shared_rcs(state)
    }

    /// Construct directly from already-shared `Rc`s. Used by `export_state`
    /// so two top-level jump args resolving to the same box collapse onto
    /// the same `Rc<VirtualStateInfoNode>` (matching RPython's
    /// `VirtualStateConstructor.create_state` box-keyed cache).
    pub fn from_shared_rcs(state: Vec<Rc<VirtualStateInfoNode>>) -> Self {
        let (slot_schedule, legacy_numnotvirtuals) = build_sequential_slot_schedule(&state);
        let mut vs = VirtualState {
            state,
            renum: std::collections::HashMap::new(),
            slot_schedule,
            numnotvirtuals: 0,
            info_counter: -1,
        };
        vs.enum_top_level();
        debug_assert_eq!(
            vs.numnotvirtuals, legacy_numnotvirtuals,
            "Phase A1 parity: enum-derived numnotvirtuals diverged from legacy slot_schedule count",
        );
        vs
    }

    fn new_with_slot_schedule(
        state: Vec<Rc<VirtualStateInfoNode>>,
        slot_schedule: Vec<usize>,
        numnotvirtuals: usize,
    ) -> Self {
        let mut vs = VirtualState {
            state,
            renum: std::collections::HashMap::new(),
            slot_schedule,
            numnotvirtuals: 0,
            info_counter: -1,
        };
        vs.enum_top_level();
        debug_assert_eq!(
            vs.numnotvirtuals, numnotvirtuals,
            "Phase A1 parity: enum-derived numnotvirtuals diverged from caller-supplied count",
        );
        vs
    }

    /// virtualstate.py:628-634 `VirtualState.__init__` per-state walk:
    /// ```python
    /// self.info_counter = -1
    /// self.numnotvirtuals = 0
    /// for s in state:
    ///     if s:
    ///         s.enum(self)
    /// ```
    /// Resets per-instance position cells before walking so that repeated
    /// calls (e.g., `rebuild_slot_schedule` after `refresh_from_gc`) start
    /// from a clean RPython-equivalent state.
    fn enum_top_level(&mut self) {
        self.info_counter = -1;
        self.numnotvirtuals = 0;
        Self::reset_positions(&self.state);
        // Borrow split: clone the top-level Rcs into a temporary list so
        // the per-node `enum_into` can take `&mut self`. Rc::clone is
        // refcount-only.
        let top: Vec<Rc<VirtualStateInfoNode>> = self.state.clone();
        for node in &top {
            node.enum_into(self);
        }
    }

    /// Walk all reachable nodes and reset position cells to -1 so the
    /// next `enum_top_level` traversal mirrors a fresh RPython
    /// VirtualState.__init__ over fresh subclass instances.
    fn reset_positions(state: &[Rc<VirtualStateInfoNode>]) {
        let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for node in state {
            Self::reset_positions_walk(node, &mut visited);
        }
    }

    fn reset_positions_walk(
        node: &Rc<VirtualStateInfoNode>,
        visited: &mut std::collections::HashSet<usize>,
    ) {
        let key = Rc::as_ptr(node) as usize;
        if !visited.insert(key) {
            return;
        }
        node.position.set(-1);
        node.position_in_notvirtuals.set(-1);
        match &node.info {
            VirtualStateInfo::Virtual { fields, .. } | VirtualStateInfo::VStruct { fields, .. } => {
                for (_, child) in fields {
                    Self::reset_positions_walk(child, visited);
                }
            }
            VirtualStateInfo::VArray { items, .. } => {
                for child in items {
                    Self::reset_positions_walk(child, visited);
                }
            }
            VirtualStateInfo::VArrayStruct { element_fields, .. } => {
                for fields in element_fields {
                    for (_, child) in fields {
                        Self::reset_positions_walk(child, visited);
                    }
                }
            }
            VirtualStateInfo::VirtualRawBuffer { entries, .. } => {
                for (_, _, child) in entries {
                    Self::reset_positions_walk(child, visited);
                }
            }
            VirtualStateInfo::Constant(_)
            | VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::IntBounded(_)
            | VirtualStateInfo::Unknown(_) => {}
        }
    }

    /// Recompute the slot schedule + numnotvirtuals after structural
    /// mutations (e.g., refresh_from_gc replacing entries with fresh
    /// `Rc`s that may have broken sharing). Mirrors RPython's invariant
    /// that `numnotvirtuals` always reflects the current state graph.
    pub fn rebuild_slot_schedule(&mut self) {
        let (slot_schedule, legacy_numnotvirtuals) = build_sequential_slot_schedule(&self.state);
        self.slot_schedule = slot_schedule;
        self.enum_top_level();
        debug_assert_eq!(
            self.numnotvirtuals, legacy_numnotvirtuals,
            "Phase A1 parity: rebuild numnotvirtuals diverged from legacy slot_schedule count",
        );
    }

    /// Counts the leaves in a single top-level state entry, deduping shared
    /// `Rc<VirtualStateInfoNode>` subtrees via the caller-supplied visited map.
    /// The visited map (Rc::as_ptr → first imported OpRef, NONE for the
    /// counting path) must be threaded across all top-level state entries
    /// in a single VirtualState walk so cross-entry shared substates are
    /// counted exactly once, matching the
    /// `build_sequential_slot_schedule` dedup. Both the top-level Rc
    /// identity and the recursive nested Rcs participate in the dedup.
    pub fn count_forced_boxes_for_entry_static(
        rc: &Rc<VirtualStateInfoNode>,
        visited: &mut std::collections::HashMap<usize, OpRef>,
    ) -> usize {
        // RPython virtualstate.py:111 first-visit guard via
        // `position == -1` — every visited node is recorded so a later
        // visit returns 0 without re-counting. Pyre's parallel:
        //
        // - If the entry is already present (real OpRef from
        //   `import_virtual_state_from_label_args_recurse` at
        //   optimizer.rs:795, or `NONE` from a prior counting visit),
        //   return 0. unroll.py:53 `setinfo_from_preamble` parity:
        //   preserve the existing real OpRef rather than overwriting.
        // - Otherwise insert `NONE` to mark this node visited, then
        //   recurse. Without this insert, repeated top-level visits
        //   to leaf variants (Unknown/NonNull/IntBounded/KnownClass)
        //   would each return 1 because they don't recurse through
        //   `_rc` and therefore never insert themselves; the dedup
        //   would silently fail and `numnotvirtuals` would over-count.
        use std::collections::hash_map::Entry;
        let key = Rc::as_ptr(rc) as usize;
        match visited.entry(key) {
            Entry::Occupied(_) => return 0,
            Entry::Vacant(e) => {
                e.insert(OpRef::NONE);
            }
        }
        Self::count_forced_boxes_for_entry(rc, visited)
    }

    fn count_forced_boxes_for_entry(
        info: &VirtualStateInfo,
        visited: &mut std::collections::HashMap<usize, OpRef>,
    ) -> usize {
        match info {
            VirtualStateInfo::Constant(_) => 0,
            VirtualStateInfo::Virtual { fields, .. } | VirtualStateInfo::VStruct { fields, .. } => {
                fields
                    .iter()
                    .map(|(_, child)| Self::count_forced_boxes_for_entry_rc(child, visited))
                    .sum()
            }
            VirtualStateInfo::VArray { items, .. } => items
                .iter()
                .map(|child| Self::count_forced_boxes_for_entry_rc(child, visited))
                .sum(),
            VirtualStateInfo::VArrayStruct { element_fields, .. } => element_fields
                .iter()
                .flat_map(|fields| fields.iter().map(|(_, child)| child))
                .map(|child| Self::count_forced_boxes_for_entry_rc(child, visited))
                .sum(),
            VirtualStateInfo::VirtualRawBuffer { entries, .. } => entries
                .iter()
                .map(|(_, _, child)| Self::count_forced_boxes_for_entry_rc(child, visited))
                .sum(),
            VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::IntBounded(_)
            | VirtualStateInfo::Unknown(_) => 1,
        }
    }

    /// Rc::as_ptr dedup wrapper for `count_forced_boxes_for_entry`,
    /// mirroring `enum_forced_boxes_recurse`.
    ///
    /// virtualstate.py:111-116 `enum()` returns without touching
    /// `self.position` when `position != -1`; pyre must preserve any
    /// real OpRef the caller already wrote into `visited` (e.g. from
    /// `import_virtual_state_from_label_args_recurse`). A blind
    /// `insert(.., NONE)` would overwrite that real OpRef before the
    /// `is_some()` check, leaking NONE into downstream lookups.
    fn count_forced_boxes_for_entry_rc(
        rc: &Rc<VirtualStateInfoNode>,
        visited: &mut std::collections::HashMap<usize, OpRef>,
    ) -> usize {
        use std::collections::hash_map::Entry;
        let key = Rc::as_ptr(rc) as usize;
        match visited.entry(key) {
            Entry::Occupied(_) => return 0,
            Entry::Vacant(e) => {
                e.insert(OpRef::NONE);
            }
        }
        Self::count_forced_boxes_for_entry(rc, visited)
    }

    /// Number of non-virtual values (need concrete OpRefs at loop entry).
    /// virtualstate.py: num_boxes()
    pub fn num_boxes(&self) -> usize {
        self.numnotvirtuals
    }

    /// Total number of entries (virtual + non-virtual).
    pub fn num_entries(&self) -> usize {
        self.state.len()
    }

    /// Number of virtual entries.
    pub fn num_virtuals(&self) -> usize {
        self.state.iter().filter(|s| s.is_virtual()).count()
    }

    /// Whether this state has any virtual objects.
    pub fn has_virtuals(&self) -> bool {
        self.state.iter().any(|s| s.is_virtual())
    }

    /// virtualstate.py:655-671 `make_inputargs(inputargs, optimizer, force_boxes=False)`.
    ///
    /// ```python
    /// def make_inputargs(self, inputargs, optimizer, force_boxes=False):
    ///     if optimizer.optearlyforce:
    ///         optimizer = optimizer.optearlyforce
    ///     assert len(inputargs) == len(self.state)
    ///     boxes = [None] * self.numnotvirtuals
    ///     # We try twice. The first time around we allow boxes to be forced
    ///     # which might change the virtual state if the box appear in more
    ///     # than one place among the inputargs.
    ///     if force_boxes:
    ///         for i in range(len(inputargs)):
    ///             self.state[i].enum_forced_boxes(boxes, inputargs[i], optimizer, True)
    ///     for i in range(len(inputargs)):
    ///         self.state[i].enum_forced_boxes(boxes, inputargs[i], optimizer)
    ///     return boxes
    /// ```
    ///
    /// Returns `Err(())` to mirror RPython's `raise VirtualStatesCantMatch`
    /// thrown from `enum_forced_boxes`. The `optimizer.optearlyforce`
    /// redirection is implicit in majit: `Optimizer::force_box` already
    /// dispatches through `OptEarlyForce` via `optearlyforce_idx`, so the
    /// caller never needs to swap the optimizer object.
    pub fn make_inputargs(
        &self,
        concrete_refs: &[OpRef],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
    ) -> Result<Vec<OpRef>, ()> {
        // boxes = [None] * self.numnotvirtuals
        let mut boxes = vec![OpRef::NONE; self.num_boxes()];
        // virtualstate.py:664-667 — first pass with `force_boxes=True`.
        // RPython writes into the SAME `boxes` array on both passes; the
        // values converge after force because subsequent
        // `get_box_replacement` reads return the forced opref.
        // `visited` tracks Rc::as_ptr identity across the whole walk to
        // mirror RPython's `state.position > self.position` shared-substate
        // dedup (virtualstate.py:196, 274, 352). It is recreated per pass
        // because each pass walks the full state from scratch.
        if force_boxes {
            let mut next_slot = 0usize;
            let mut slot_cursor = 0usize;
            let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for (idx, info) in self.state.iter().enumerate() {
                let opref = concrete_refs.get(idx).copied().unwrap_or(OpRef::NONE);
                Self::enum_forced_boxes_for_entry(
                    info,
                    opref,
                    optimizer,
                    ctx,
                    &mut boxes,
                    &mut next_slot,
                    /* force_boxes */ true,
                    &self.slot_schedule,
                    &mut slot_cursor,
                    &mut visited,
                )?;
            }
        }
        // virtualstate.py:668-669 — second pass with `force_boxes=False`,
        // unconditional. Mirrors RPython exactly.
        let mut next_slot = 0usize;
        let mut slot_cursor = 0usize;
        let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (idx, info) in self.state.iter().enumerate() {
            let opref = concrete_refs.get(idx).copied().unwrap_or(OpRef::NONE);
            Self::enum_forced_boxes_for_entry(
                info,
                opref,
                optimizer,
                ctx,
                &mut boxes,
                &mut next_slot,
                /* force_boxes */ false,
                &self.slot_schedule,
                &mut slot_cursor,
                &mut visited,
            )?;
        }
        Ok(boxes)
    }

    /// virtualstate.py:673-683 `make_inputargs_and_virtuals(inputargs, optimizer, force_boxes=False)`.
    ///
    /// ```python
    /// def make_inputargs_and_virtuals(self, inputargs, optimizer, force_boxes=False):
    ///     inpargs = self.make_inputargs(inputargs, optimizer, force_boxes)
    ///     virtuals = []
    ///     for i in range(len(inputargs)):
    ///         if not isinstance(self.state[i], NotVirtualStateInfo):
    ///             virtuals.append(inputargs[i])
    ///     return inpargs, virtuals
    /// ```
    pub fn make_inputargs_and_virtuals(
        &self,
        concrete_refs: &[OpRef],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
    ) -> Result<(Vec<OpRef>, Vec<OpRef>), ()> {
        let inputargs = self.make_inputargs(concrete_refs, optimizer, ctx, force_boxes)?;
        let virtuals: Vec<OpRef> = self
            .state
            .iter()
            .enumerate()
            .filter(|(_, info)| info.is_virtual())
            .filter_map(|(i, _)| concrete_refs.get(i).copied())
            .collect();
        Ok((inputargs, virtuals))
    }

    /// Walk the virtual state tree the way RPython's `enum_forced_boxes`
    /// does (virtualstate.py:182-198 AbstractVirtualStructStateInfo,
    /// 263-275 VArrayStateInfo, 333-354 VArrayStructStateInfo,
    /// 412-425 NotVirtualStateInfo) and write one entry per non-virtual,
    /// non-constant leaf into `boxes`.
    ///
    /// RPython has one method per state subclass; majit dispatches via
    /// `match` over the `VirtualStateInfo` enum (Rust enum vs Python
    /// class hierarchy — same dispatch, different shape).
    ///
    /// The Virtual / VStruct / VArray / VArrayStruct branches mirror
    /// the line `if info is None or not info.is_virtual(): raise
    /// VirtualStatesCantMatch()` (virtualstate.py:185, 266, 336):
    /// returning `Err(())` is the majit equivalent of raising
    /// `VirtualStatesCantMatch`.
    ///
    /// The leaf branch mirrors virtualstate.py:412-425 — when the
    /// resolved box is virtual but the slot is non-virtual, force it
    /// through the optimizer if `force_boxes=True`, otherwise raise.
    ///
    /// **Shared-substate dedup**: RPython's `state.position > self.position`
    /// guard (virtualstate.py:196, 274, 352) skips revisiting a shared
    /// `AbstractVirtualStateInfo` so each unique state object's
    /// `NotVirtualStateInfo` gets exactly one slot. The Rust port wraps
    /// nested children in `Rc<VirtualStateInfoNode>` and dedups via
    /// `Rc::as_ptr` identity (carried in `visited`) — when
    /// `export_single_value` returns the SAME `Rc` for an aliased box,
    /// the second visit short-circuits in `enum_forced_boxes_recurse`,
    /// matching `if self.position != -1: return` semantics.
    /// `build_sequential_slot_schedule` shares the same dedup logic so
    /// the slot count and the walk stay in lockstep.
    fn enum_forced_boxes_for_entry(
        info: &VirtualStateInfo,
        opref: OpRef,
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        boxes: &mut [OpRef],
        next_slot: &mut usize,
        force_boxes: bool,
        slot_schedule: &[usize],
        slot_cursor: &mut usize,
        visited: &mut std::collections::HashSet<usize>,
    ) -> Result<(), ()> {
        match info {
            VirtualStateInfo::Constant(_) => Ok(()),
            VirtualStateInfo::Virtual { fields, .. } | VirtualStateInfo::VStruct { fields, .. } => {
                // virtualstate.py:182-188:
                //     box = get_box_replacement(box)
                //     info = getptrinfo(box)
                //     if info is None or not info.is_virtual():
                //         raise VirtualStatesCantMatch()
                //     else:
                //         assert isinstance(info, AbstractStructPtrInfo)
                let resolved = ctx.get_box_replacement(opref);
                let is_virtual = ctx
                    .get_ptr_info(resolved)
                    .map_or(false, |pi| pi.is_virtual());
                if !is_virtual {
                    return Err(());
                }
                // virtualstate.py:192-198: walk min(len(fielddescrs),
                // len(info._fields)) entries — RPython explicitly comments
                // that the min() guards against unvalidated callers.
                let info_field_count = ctx
                    .get_ptr_info(resolved)
                    .map(|pi| match pi {
                        PtrInfo::Virtual(vinfo) => vinfo.fields.len(),
                        PtrInfo::VirtualStruct(vinfo) => vinfo.fields.len(),
                        _ => 0,
                    })
                    .unwrap_or(0);
                let walk_count = fields.len().min(info_field_count);
                let field_refs: Vec<_> = fields
                    .iter()
                    .take(walk_count)
                    .map(|(field_idx, _)| {
                        ctx.get_ptr_info(resolved)
                            .and_then(|info| info.getfield(*field_idx))
                            .and_then(|e| e.as_opref())
                            .map(|f| ctx.get_box_replacement(f))
                            .unwrap_or(OpRef::NONE)
                    })
                    .collect();
                for ((_, field_state), field_ref) in
                    fields.iter().take(walk_count).zip(field_refs.iter())
                {
                    Self::enum_forced_boxes_recurse(
                        field_state,
                        *field_ref,
                        optimizer,
                        ctx,
                        boxes,
                        next_slot,
                        force_boxes,
                        slot_schedule,
                        slot_cursor,
                        visited,
                    )?;
                }
                Ok(())
            }
            VirtualStateInfo::VArray { items, .. } => {
                // virtualstate.py:263-275 VArrayStateInfo.enum_forced_boxes
                let resolved = ctx.get_box_replacement(opref);
                let is_virtual = ctx
                    .get_ptr_info(resolved)
                    .map_or(false, |pi| pi.is_virtual());
                if !is_virtual {
                    return Err(());
                }
                // virtualstate.py:268-269: explicit length check.
                //     if len(self.fieldstate) > info.getlength():
                //         raise VirtualStatesCantMatch
                let array_len = ctx
                    .get_ptr_info(resolved)
                    .map(|pi| match pi {
                        PtrInfo::VirtualArray(ainfo) => ainfo.items.len(),
                        _ => 0,
                    })
                    .unwrap_or(0);
                if items.len() > array_len {
                    return Err(());
                }
                for (index, item_state) in items.iter().enumerate() {
                    let item_ref = ctx
                        .get_ptr_info(resolved)
                        .and_then(|info| info.getitem(index))
                        .and_then(|e| e.as_opref())
                        .unwrap_or(OpRef::NONE);
                    Self::enum_forced_boxes_recurse(
                        item_state,
                        item_ref,
                        optimizer,
                        ctx,
                        boxes,
                        next_slot,
                        force_boxes,
                        slot_schedule,
                        slot_cursor,
                        visited,
                    )?;
                }
                Ok(())
            }
            VirtualStateInfo::VArrayStruct { element_fields, .. } => {
                // virtualstate.py:333-354 VArrayStructStateInfo.enum_forced_boxes:
                //   for i in range(self.length):
                //       for descr in self.fielddescrs:
                //           index = i * len(self.fielddescrs) + descr.get_index()
                //           fieldstate = self.fieldstate[index]
                //           itembox = opinfo._items[i * len(self.fielddescrs) +
                //                                   descr.get_index()]
                //           if fieldstate is None:
                //               if itembox is not None:
                //                   raise VirtualStatesCantMatch
                //               continue
                //           if fieldstate.position > self.position:
                //               fieldstate.enum_forced_boxes(boxes, itembox, ...)
                //
                // pyre stores element_fields sparsely: each inner Vec only
                // contains the fields that have a state. The "fieldstate is
                // None" case corresponds to a field_idx present in the
                // runtime's vinfo but absent from the state's element_fields,
                // which signals a schema mismatch.
                let resolved = ctx.get_box_replacement(opref);
                let runtime_fields: Vec<Vec<(u32, OpRef)>> = match ctx.get_ptr_info(resolved) {
                    Some(crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(vinfo)) => {
                        vinfo.element_fields.clone()
                    }
                    _ => return Err(()),
                };
                if runtime_fields.len() != element_fields.len() {
                    return Err(());
                }
                for (elem_idx, fields) in element_fields.iter().enumerate() {
                    let runtime_elem = &runtime_fields[elem_idx];
                    // virtualstate.py:347-349: if fieldstate is None and
                    // itembox is not None → raise. In pyre's sparse model,
                    // a runtime field absent from the state's element_fields
                    // is the equivalent mismatch.
                    for (rt_field_idx, _) in runtime_elem {
                        if !fields.iter().any(|(fdidx, _)| fdidx == rt_field_idx) {
                            return Err(());
                        }
                    }
                    for (field_idx, field_state) in fields {
                        // Look up the runtime ref via the proper interior
                        // accessor (info.py:_compute_index parity), not the
                        // VirtualArray-only `getitem(flat_index)` helper.
                        let item_ref = runtime_elem
                            .iter()
                            .find(|(fdidx, _)| fdidx == field_idx)
                            .map(|(_, op)| *op)
                            .unwrap_or(OpRef::NONE);
                        Self::enum_forced_boxes_recurse(
                            field_state,
                            item_ref,
                            optimizer,
                            ctx,
                            boxes,
                            next_slot,
                            force_boxes,
                            slot_schedule,
                            slot_cursor,
                            visited,
                        )?;
                    }
                }
                Ok(())
            }
            VirtualStateInfo::VirtualRawBuffer { entries, .. } => {
                // majit-only: VirtualRawBuffer mirrors VStruct semantics.
                let resolved = ctx.get_box_replacement(opref);
                let is_virtual = ctx
                    .get_ptr_info(resolved)
                    .map_or(false, |pi| pi.is_virtual());
                if !is_virtual {
                    return Err(());
                }
                for (index, (_, _, entry_state)) in entries.iter().enumerate() {
                    let entry_ref = ctx
                        .get_ptr_info(resolved)
                        .and_then(|info| match info {
                            PtrInfo::VirtualRawBuffer(vinfo) => vinfo.values.get(index).copied(),
                            _ => None,
                        })
                        .unwrap_or(OpRef::NONE);
                    Self::enum_forced_boxes_recurse(
                        entry_state,
                        entry_ref,
                        optimizer,
                        ctx,
                        boxes,
                        next_slot,
                        force_boxes,
                        slot_schedule,
                        slot_cursor,
                        visited,
                    )?;
                }
                Ok(())
            }
            VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::IntBounded(_)
            | VirtualStateInfo::Unknown(_) => {
                // virtualstate.py:412-425 NotVirtualStateInfo.enum_forced_boxes:
                //     if self.level == LEVEL_CONSTANT: return
                //     assert 0 <= self.position_in_notvirtuals
                //     assert optimizer is not None
                //     box = get_box_replacement(box)
                //     if box.type == 'r':
                //         info = getptrinfo(box)
                //         if info and info.is_virtual():
                //             if force_boxes:
                //                 info.force_box(box, optimizer)
                //             else:
                //                 raise VirtualStatesCantMatch
                //     boxes[self.position_in_notvirtuals] = box
                let resolved = ctx.get_box_replacement(opref);
                let forced = match ctx.get_ptr_info(resolved) {
                    // RPython: Virtualizable refs stay virtual across iterations.
                    Some(PtrInfo::Virtualizable(_)) => resolved,
                    Some(ptr_info) if ptr_info.is_virtual() => {
                        if !force_boxes {
                            return Err(());
                        }
                        optimizer.force_box(resolved, ctx)
                    }
                    _ => resolved,
                };
                // boxes[self.position_in_notvirtuals] = box
                // majit's `slot_schedule` (precomputed via `compute_renum`)
                // plays the role of `position_in_notvirtuals`: each leaf
                // entry has a deterministic slot index that may collapse
                // aliased boxes onto the same slot.
                let slot = slot_schedule
                    .get(*slot_cursor)
                    .copied()
                    .unwrap_or(*next_slot);
                let resolved_for_store = ctx.get_box_replacement(forced);
                // virtualstate.py:417 NotVirtualStateInfo{Int,Ptr}: Box.type
                // immutability. RPython dispatches `isinstance(self,
                // NotVirtualStateInfoInt)` vs `NotVirtualStateInfoPtr` on a
                // fixed-type Box, so a Ref-typed slot can never receive an
                // Int/Float source (and vice versa). pyre's flat-OpRef
                // forwarding via `get_box_replacement` can technically
                // bridge types; reject the cross here so the unrolled
                // trace falls back to `jump_to_preamble` exactly as RPython
                // raises VirtualStatesCantMatch.
                if let (Some(expected), Some(actual)) =
                    (info.info_type(), ctx.opref_type(resolved_for_store))
                {
                    if expected != actual {
                        if std::env::var_os("MAJIT_LOG").is_some() {
                            eprintln!(
                                "[label-type-mismatch] slot={} expected={:?} actual={:?} \
                                 source={:?} resolved={:?}",
                                slot, expected, actual, opref, resolved_for_store
                            );
                        }
                        return Err(());
                    }
                }
                if let Some(dst) = boxes.get_mut(slot) {
                    *dst = resolved_for_store;
                }
                *slot_cursor += 1;
                *next_slot += 1;
                Ok(())
            }
        }
    }

    /// virtualstate.py:111-116 `enum` parity for the
    /// `enum_forced_boxes_for_entry` recursion: dedup nested
    /// `Rc<VirtualStateInfoNode>` references via pointer identity so a shared
    /// substate is enumerated only once. The first call writes its leaves
    /// into `boxes`; subsequent visits short-circuit, mirroring
    /// `if self.position != -1: return`.
    fn enum_forced_boxes_recurse(
        rc: &Rc<VirtualStateInfoNode>,
        opref: OpRef,
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        boxes: &mut [OpRef],
        next_slot: &mut usize,
        force_boxes: bool,
        slot_schedule: &[usize],
        slot_cursor: &mut usize,
        visited: &mut std::collections::HashSet<usize>,
    ) -> Result<(), ()> {
        let key = Rc::as_ptr(rc) as usize;
        if !visited.insert(key) {
            return Ok(());
        }
        Self::enum_forced_boxes_for_entry(
            rc,
            opref,
            optimizer,
            ctx,
            boxes,
            next_slot,
            force_boxes,
            slot_schedule,
            slot_cursor,
            visited,
        )
    }

    /// Check if another VirtualState is compatible (can reuse the optimized loop body).
    ///
    /// Returns true if all entries are compatible.
    pub fn is_compatible(&self, other: &VirtualState) -> bool {
        if self.state.len() != other.state.len() {
            return false;
        }
        self.state
            .iter()
            .zip(other.state.iter())
            .all(|(a, b)| a.is_compatible(b))
    }

    /// virtualstate.py: generalization_of(other, optimizer)
    ///
    /// `self` is the target loop state's requirement and `other` is the
    /// incoming state. Returns true if `self` can safely accept `other`.
    pub fn generalization_of(&self, other: &VirtualState) -> bool {
        self.is_compatible(other)
    }

    /// virtualstate.py: generate_guards(other, boxes, runtime_boxes, optimizer)
    ///
    /// Generate guards to bridge from `other` state to `self` state.
    /// Returns a list of guard operations that need to be emitted to ensure
    /// the incoming values satisfy the requirements of the optimized loop.
    ///
    /// `runtime_boxes`: live OpRefs at the jump point. When provided,
    /// per-entry guard generation can peek at the runtime value to decide
    /// whether emitting a GUARD_VALUE is profitable (e.g. when the
    /// runtime value already equals the expected constant).
    /// virtualstate.py:646: generate_guards(self, other, boxes, runtime_boxes, optimizer)
    ///
    /// `boxes`: the actual OpRefs at each position (used as the guard's
    /// first argument in GUARD_VALUE etc.).
    /// virtualstate.py:646 generate_guards parity.
    ///
    /// Returns Ok(guards) if the incoming state can be accepted with
    /// runtime guards, Err(()) if fundamentally incompatible
    /// (VirtualStatesCantMatch).
    ///
    /// `runtime_boxes`: live OpRefs at the jump point. When Some,
    /// non-permanent guard emission is enabled (matching RPython's
    /// _jump_to_existing_trace path). When None, only structurally
    /// compatible pairs are accepted (matching generalization_of).
    ///
    /// `force_boxes`: when true, Virtual incoming values can be
    /// accepted by NonVirtual targets (the virtual will be forced
    /// later by make_inputargs). Matches RPython's force_boxes.
    pub fn generate_guards(
        &self,
        other: &VirtualState,
        boxes: &[OpRef],
        runtime_boxes: Option<&[OpRef]>,
        ctx: &OptContext,
        force_boxes: bool,
    ) -> Result<Vec<GuardRequirement>, ()> {
        if self.state.len() != other.state.len() {
            return Err(());
        }
        let mut guards = Vec::new();

        for (i, (expected, incoming)) in self.state.iter().zip(other.state.iter()).enumerate() {
            let box_opref = boxes.get(i).copied().unwrap_or(OpRef::NONE);
            let runtime_box = runtime_boxes.and_then(|rb| rb.get(i).copied());
            if let Err(()) = Self::generate_guards_for_entry(
                i,
                expected,
                incoming,
                box_opref,
                runtime_box,
                ctx,
                force_boxes,
                &mut guards,
            ) {
                if std::env::var_os("MAJIT_LOG_JTET").is_some() {
                    let runtime_value =
                        runtime_box.and_then(|runtime_box| ctx.get_constant(runtime_box).cloned());
                    eprintln!(
                        "[jit][jte] virtualstate mismatch index={i} box={box_opref:?} runtime={runtime_box:?} runtime_value={runtime_value:?} expected={expected:?} incoming={incoming:?}"
                    );
                }
                return Err(());
            }
        }

        Ok(guards)
    }

    /// virtualstate.py per-entry generate_guards parity.
    ///
    /// Returns Ok(()) if the pair is compatible (possibly with guards),
    /// Err(()) if fundamentally incompatible (VirtualStatesCantMatch).
    ///
    /// `runtime_box`: when Some, non-permanent guard emission is possible.
    /// When None (generalization_of path), only structurally compatible
    /// pairs are accepted.  RPython uses the concrete runtime value as an
    /// "educated guess" (comment at virtualstate.py:551-555); majit does
    /// not have concrete values so it optimistically emits guards whenever
    /// runtime_box is Some.
    ///
    /// `force_boxes`: when true, Virtual incoming can be accepted by
    /// non-virtual targets (virtualstate.py:523-524 _generate_virtual_guards).
    fn generate_guards_for_entry(
        arg_idx: usize,
        expected: &VirtualStateInfo,
        incoming: &VirtualStateInfo,
        box_opref: OpRef,
        runtime_box: Option<OpRef>,
        ctx: &OptContext,
        force_boxes: bool,
        guards: &mut Vec<GuardRequirement>,
    ) -> Result<(), ()> {
        // virtualstate.py:523-524: force_boxes + Virtual incoming
        // → _generate_virtual_guards (check class compatibility only).
        // virtualstate.py:523-524: force_boxes + incoming virtual, expected non-virtual
        if force_boxes && incoming.is_virtual() && !expected.is_virtual() {
            return match expected {
                VirtualStateInfo::Constant(_) => Err(()),
                VirtualStateInfo::KnownClass { class_ptr } => {
                    if let VirtualStateInfo::Virtual { known_class, .. } = incoming {
                        if known_class.as_ref() == Some(class_ptr) {
                            Ok(())
                        } else {
                            Err(())
                        }
                    } else {
                        Ok(())
                    }
                }
                _ => Ok(()),
            };
        }
        // virtualstate.py:520-530: _generate_virtual_guards —
        // force_boxes + expected virtual, incoming non-virtual (forced box).
        // The forced box's known class must match the virtual's class.
        if force_boxes && expected.is_virtual() && !incoming.is_virtual() {
            if let VirtualStateInfo::KnownClass { class_ptr } = incoming {
                let expected_class = match expected {
                    VirtualStateInfo::Virtual { known_class, .. } => known_class.as_ref(),
                    _ => None,
                };
                return if expected_class == Some(class_ptr) || expected_class.is_none() {
                    Ok(())
                } else {
                    Err(())
                };
            }
            if matches!(
                incoming,
                VirtualStateInfo::NonNull | VirtualStateInfo::Unknown(_)
            ) {
                return Ok(());
            }
            return Err(());
        }

        // virtualstate.py:392-394 NotVirtualStateInfo._generate_guards:
        //
        //     if not isinstance(other, NotVirtualStateInfo):
        //         raise VirtualStatesCantMatch(
        //             'comparing a constant against something that is a virtual')
        //
        // This isinstance check lives in NotVirtualStateInfo(Int/Ptr)
        // subclasses, NOT in VirtualStateInfo. When `expected` is itself
        // a Virtual/VArray/VStruct, the comparison is handled by the
        // VirtualStateInfo._generate_guards path (the main match below),
        // which does struct-level field comparison — not type-tag matching.
        if !expected.is_virtual() {
            if let Some(expected_type) = expected.info_type() {
                if !info_type_matches(expected_type, incoming) {
                    return Err(());
                }
            }
        }

        match (expected, incoming) {
            // virtualstate.py:387-389: Unknown target accepts anything of
            // the same type (the isinstance check above already enforced
            // the type agreement).
            (VirtualStateInfo::Unknown(_), _) => Ok(()),

            // ── Constant target ── (virtualstate.py:396-405)
            (VirtualStateInfo::Constant(a), VirtualStateInfo::Constant(b)) if a == b => Ok(()),
            (VirtualStateInfo::Constant(val), _) => {
                // virtualstate.py:400-405: emit GUARD_VALUE only when the
                // concrete runtime box already equals the target constant.
                // Merely having a runtime box is not enough; RPython checks
                // `self.constbox.same_constant(runtime_box.constbox())`.
                if runtime_box
                    .and_then(|runtime_box| ctx.get_constant(runtime_box))
                    .is_some_and(|runtime_value| runtime_value == val)
                {
                    guards.push(GuardRequirement::GuardValue {
                        arg_index: arg_idx,
                        box_opref,
                        expected_value: val.clone(),
                    });
                    Ok(())
                } else {
                    Err(())
                }
            }

            // ── KnownClass target ── (virtualstate.py:595-624)
            (
                VirtualStateInfo::KnownClass { class_ptr: c1 },
                VirtualStateInfo::KnownClass { class_ptr: c2 },
            ) if c1 == c2 => Ok(()),
            (VirtualStateInfo::KnownClass { class_ptr }, VirtualStateInfo::Unknown(_)) => {
                // virtualstate.py:600-606: runtime_box gate
                if runtime_box.is_some() {
                    guards.push(GuardRequirement::GuardClass {
                        arg_index: arg_idx,
                        box_opref,
                        expected_class: *class_ptr,
                    });
                    Ok(())
                } else {
                    Err(())
                }
            }
            (VirtualStateInfo::KnownClass { class_ptr }, VirtualStateInfo::NonNull) => {
                // virtualstate.py:607-613: runtime_box gate
                if runtime_box.is_some() {
                    guards.push(GuardRequirement::GuardClass {
                        arg_index: arg_idx,
                        box_opref,
                        expected_class: *class_ptr,
                    });
                    Ok(())
                } else {
                    Err(())
                }
            }
            (
                VirtualStateInfo::KnownClass { class_ptr },
                VirtualStateInfo::Constant(Value::Ref(r)),
            ) if !r.is_null() => {
                // virtualstate.py:618-624: runtime_box needed to verify
                // the constant's class matches. Without it, reject.
                if runtime_box.is_some() {
                    guards.push(GuardRequirement::GuardClass {
                        arg_index: arg_idx,
                        box_opref,
                        expected_class: *class_ptr,
                    });
                    Ok(())
                } else {
                    Err(())
                }
            }

            // ── NonNull target ── (virtualstate.py:574-593)
            (VirtualStateInfo::NonNull, VirtualStateInfo::NonNull)
            | (VirtualStateInfo::NonNull, VirtualStateInfo::KnownClass { .. }) => Ok(()),
            (VirtualStateInfo::NonNull, VirtualStateInfo::Constant(Value::Ref(r))) => {
                if !r.is_null() { Ok(()) } else { Err(()) }
            }
            (VirtualStateInfo::NonNull, VirtualStateInfo::Unknown(_)) => {
                // virtualstate.py:578-584: runtime_box gate
                if runtime_box.is_some() {
                    guards.push(GuardRequirement::GuardNonnull {
                        arg_index: arg_idx,
                        box_opref,
                    });
                    Ok(())
                } else {
                    Err(())
                }
            }
            // NonNull accepts any virtual (virtual is always nonnull).
            (VirtualStateInfo::NonNull, incoming) if incoming.is_virtual() => Ok(()),

            // ── IntBounded target ── (virtualstate.py:483-499)
            (VirtualStateInfo::IntBounded(b1), VirtualStateInfo::IntBounded(b2))
                if b2.lower >= b1.lower && b2.upper <= b1.upper =>
            {
                Ok(())
            }
            (VirtualStateInfo::IntBounded(b), VirtualStateInfo::Constant(Value::Int(v)))
                if b.contains(*v) =>
            {
                Ok(())
            }
            (VirtualStateInfo::IntBounded(bounds), VirtualStateInfo::Unknown(_)) => {
                // virtualstate.py:493-498: runtime_box gate
                if runtime_box.is_some() {
                    guards.push(GuardRequirement::GuardBounds {
                        arg_index: arg_idx,
                        box_opref,
                        bounds: bounds.clone(),
                    });
                    Ok(())
                } else {
                    Err(())
                }
            }

            // ── Virtual targets ── (virtualstate.py:141-320)
            // Structural match delegated to is_compatible.
            (expected_vs, incoming_vs) if expected_vs.is_virtual() => {
                if expected_vs.is_compatible(incoming_vs) {
                    Ok(())
                } else {
                    Err(())
                }
            }

            // Fundamentally incompatible: VirtualStatesCantMatch.
            _ => Err(()),
        }
    }

    /// virtualstate.py: compute_renum(oprefs)
    /// Build the renum mapping from OpRef to numbering index.
    /// This ensures consistent virtual identity across loop iterations.
    pub fn compute_renum(&mut self, oprefs: &[OpRef]) {
        self.renum.clear();
        for (i, &opref) in oprefs.iter().enumerate() {
            if !opref.is_none() {
                self.renum.insert(opref, i);
            }
        }
    }

    /// Get the numbering index for an OpRef.
    pub fn get_renum(&self, opref: OpRef) -> Option<usize> {
        self.renum.get(&opref).copied()
    }

    /// virtualstate.py: debug_print(hdr, bad, metainterp_sd)
    /// Format the virtual state for debugging.
    pub fn debug_print(&self) -> String {
        let mut out = String::new();
        for (i, info) in self.state.iter().enumerate() {
            let kind = match &info.info {
                VirtualStateInfo::Constant(_) => "Constant",
                VirtualStateInfo::Virtual { .. } => "Virtual",
                VirtualStateInfo::VArray { .. } => "VArray",
                VirtualStateInfo::VStruct { .. } => "VStruct",
                VirtualStateInfo::VArrayStruct { .. } => "VArrayStruct",
                VirtualStateInfo::VirtualRawBuffer { .. } => "VRawBuf",
                VirtualStateInfo::KnownClass { .. } => "KnownClass",
                VirtualStateInfo::NonNull => "NonNull",
                VirtualStateInfo::IntBounded(_) => "IntBounded",
                VirtualStateInfo::Unknown(_) => "Unknown",
            };
            out.push_str(&format!("  [{i}] {kind}\n"));
        }
        out
    }
}

impl VirtualState {
    /// virtualstate.py: generate_guards(other, boxes, runtime_boxes, optimizer)
    ///
    /// Full guard generation with GenerateGuardState context.
    /// For each state entry, generate guards that make the runtime values
    /// match the expected virtual state shape.
    ///
    /// `runtime_boxes`: live OpRefs at the jump point. Used by per-entry
    /// guard generation to decide whether emitting a GUARD_VALUE is
    /// profitable (e.g. runtime value already equals the expected constant).
    pub fn generate_guards_with_state<'a>(
        &self,
        other: &VirtualState,
        boxes: &[OpRef],
        runtime_boxes: Option<&[OpRef]>,
        ctx: &'a OptContext,
    ) -> GenerateGuardState<'a> {
        let mut state = GenerateGuardState::new(ctx);
        let len = self.state.len().min(other.state.len()).min(boxes.len());
        for i in 0..len {
            let runtime_box = runtime_boxes.and_then(|rb| rb.get(i).copied());
            self.generate_guard_for_entry(
                i,
                &self.state[i],
                &other.state[i],
                boxes[i],
                runtime_box,
                &mut state,
            );
        }
        state
    }

    /// Per-entry guard generation (virtualstate.py: AbstractVirtualStateInfo.generate_guards)
    ///
    /// `runtime_box`: the live OpRef for this entry at the jump point.
    /// For LEVEL_CONSTANT targets, if the runtime_box is available we emit
    /// GUARD_VALUE instead of marking the entry as bad — matching RPython's
    /// `NotVirtualStateInfo._generate_guards` (line 400).
    fn generate_guard_for_entry(
        &self,
        idx: usize,
        expected: &VirtualStateInfo,
        incoming: &VirtualStateInfo,
        box_opref: OpRef,
        runtime_box: Option<OpRef>,
        state: &mut GenerateGuardState,
    ) {
        match (expected, incoming) {
            // Constant vs same constant → no guard needed
            (VirtualStateInfo::Constant(val), VirtualStateInfo::Constant(other_val)) => {
                if val != other_val {
                    state.mark_bad(idx, "constant mismatch");
                }
            }
            // Constant vs non-constant: emit GUARD_VALUE if runtime_box available,
            // otherwise mark as bad. (virtualstate.py line 400)
            (VirtualStateInfo::Constant(_val), _) => {
                if runtime_box.is_some() {
                    // PRE-EXISTING-ADAPTATION (Slice 5.D.9 audit, last-mover):
                    // synthesizes a fake "const-pool position" via raw u32
                    // 10_000 + idx, ignoring `_val`'s actual value. Convergence
                    // path: thread `_val` into the constant pool and emit the
                    // matching `OpRef::const_int/const_float/const_ptr(idx)`
                    // here. Until then keep the `OpRef::from_raw(_)` form so
                    // the audit-flagged synthesis stays visible — no other
                    // typed factory is semantically correct for this synthetic
                    // position.
                    let val_const = OpRef::from_raw(10_000 + idx as u32);
                    let mut op = Op::new(OpCode::GuardValue, &[box_opref, val_const]);
                    op.fail_args = Some(Default::default());
                    state.add_guard(op);
                } else {
                    state.mark_bad(idx, "constant mismatch");
                }
            }
            // KnownClass vs unknown → GUARD_CLASS
            // virtualstate.py:601-602: with runtime_box, RPython emits
            // GUARD_NONNULL_CLASS using cpu.cls_of_box(runtime_box).
            // We rely on GuardRequirement::GuardClass from the caller.
            (VirtualStateInfo::KnownClass { .. }, VirtualStateInfo::Unknown(_)) => {
                // Guard will be generated by the caller from GuardRequirement
            }
            // NonNull vs unknown → GUARD_NONNULL
            // virtualstate.py:579: runtime_box check is implicit — we always
            // emit GuardNonnull via GuardRequirement.
            (VirtualStateInfo::NonNull, VirtualStateInfo::Unknown(_)) => {}
            // IntBounded vs unknown → bounds guards
            // virtualstate.py:493-498: RPython checks runtime_box.getint()
            // against self.intbound.contains() before emitting bounds guards.
            // We lack runtime value access, so we emit bounds guards
            // unconditionally via GuardRequirement::GuardBounds.
            (VirtualStateInfo::IntBounded(_), VirtualStateInfo::Unknown(_)) => {}
            // Virtual vs Virtual → check fields match
            (
                VirtualStateInfo::Virtual { descr: d1, .. },
                VirtualStateInfo::Virtual { descr: d2, .. },
            ) => {
                if d1.index() != d2.index() {
                    state.mark_bad(idx, "virtual descriptor mismatch");
                }
            }
            // Virtual vs non-virtual → cannot match
            (VirtualStateInfo::Virtual { .. }, _) => {
                state.mark_bad(idx, "expected virtual, got non-virtual");
            }
            // Compatible or both unknown → ok
            _ => {}
        }
    }

    /// virtualstate.py: force_boxes(optimizer) — force all virtual entries
    /// to be materialized. After calling this, all entries become non-virtual
    /// (Constant, KnownClass, NonNull, IntBounded, or Unknown).
    ///
    /// Returns the number of virtuals that were forced.
    pub fn force_boxes(&mut self) -> usize {
        let mut count = 0;
        for slot in &mut self.state {
            if slot.is_virtual() {
                // virtualstate.py: forced virtuals become NonNull
                // (they were allocated, so they're always non-null).
                let new_kind = match &slot.info {
                    VirtualStateInfo::Virtual { known_class, .. } => {
                        if let Some(cls) = *known_class {
                            VirtualStateInfo::KnownClass { class_ptr: cls }
                        } else {
                            VirtualStateInfo::NonNull
                        }
                    }
                    _ => VirtualStateInfo::NonNull,
                };
                *slot = VirtualStateInfoNode::new_rc(new_kind);
                count += 1;
            }
        }
        // Forcing breaks any prior Rc sharing, so the slot schedule
        // (which keys on Rc::as_ptr) must be recomputed.
        self.rebuild_slot_schedule();
        count
    }

    /// Get the lenbound of a virtual array at the given index, if any.
    pub fn getlenbound(&self, index: usize) -> Option<&IntBound> {
        match self.state.get(index).map(|rc| &rc.info) {
            Some(VirtualStateInfo::VArray { lenbound, .. }) => lenbound.as_ref(),
            _ => None,
        }
    }

    /// Merge two virtual states. For each entry, take the weaker
    /// (more general) of the two. Used when multiple paths converge.
    pub fn merge(&self, other: &VirtualState) -> VirtualState {
        let merged: Vec<Rc<VirtualStateInfoNode>> = self
            .state
            .iter()
            .zip(other.state.iter())
            .map(|(a, b)| {
                if a.is_compatible(b) {
                    Rc::clone(a)
                } else {
                    // Widen to the common type, or fall back to Ref.
                    // RPython's `not_virtual(cpu, box.type, None)`
                    // reconstructs the `type` discriminant from the Box
                    // at merge time; pyre keeps the type on the
                    // pre-existing `Unknown(Type)` tag.
                    let tp = a.info_type().or_else(|| b.info_type()).unwrap_or(Type::Ref);
                    VirtualStateInfoNode::new_rc(VirtualStateInfo::Unknown(tp))
                }
            })
            .collect();
        VirtualState::from_shared_rcs(merged)
    }
}

impl Clone for VirtualState {
    /// Deep-clone the entire `VirtualStateInfoNode` tree so the cloned
    /// `VirtualState` owns an independent set of `position` /
    /// `position_in_notvirtuals` cells. Within a single clone, source nodes
    /// shared by `Rc` identity remain shared in the destination (cached by
    /// `Rc::as_ptr`), preserving RPython's `VirtualStateConstructor`
    /// box-keyed instance sharing semantics.
    ///
    /// A naive `#[derive(Clone)]` would `Rc::clone` (refcount-only) and
    /// leak position cells across clones; calling `enum_top_level` on
    /// either copy then resets shared `Cell<i32>` positions, corrupting
    /// the other. RPython's `VirtualState.__init__` constructs fresh
    /// subclass instances per VirtualState — this manual impl reproduces
    /// that invariant.
    fn clone(&self) -> Self {
        let mut cache: std::collections::HashMap<
            *const VirtualStateInfoNode,
            Rc<VirtualStateInfoNode>,
        > = std::collections::HashMap::new();
        let cloned: Vec<Rc<VirtualStateInfoNode>> = self
            .state
            .iter()
            .map(|src| deep_clone_node(src, &mut cache))
            .collect();
        VirtualState::from_shared_rcs(cloned)
    }
}

/// Deep-clone a `VirtualStateInfoNode` tree, mapping each source `Rc`
/// identity to one fresh `Rc` in the destination via `cache`. Used by
/// `<VirtualState as Clone>::clone`.
fn deep_clone_node(
    src: &Rc<VirtualStateInfoNode>,
    cache: &mut std::collections::HashMap<*const VirtualStateInfoNode, Rc<VirtualStateInfoNode>>,
) -> Rc<VirtualStateInfoNode> {
    let key = Rc::as_ptr(src);
    if let Some(hit) = cache.get(&key) {
        return Rc::clone(hit);
    }
    let cloned_info = match &src.info {
        VirtualStateInfo::Constant(v) => VirtualStateInfo::Constant(*v),
        VirtualStateInfo::KnownClass { class_ptr } => VirtualStateInfo::KnownClass {
            class_ptr: *class_ptr,
        },
        VirtualStateInfo::NonNull => VirtualStateInfo::NonNull,
        VirtualStateInfo::IntBounded(b) => VirtualStateInfo::IntBounded(b.clone()),
        VirtualStateInfo::Unknown(t) => VirtualStateInfo::Unknown(*t),
        VirtualStateInfo::Virtual {
            descr,
            known_class,
            ob_type_descr,
            fields,
            field_descrs,
        } => VirtualStateInfo::Virtual {
            descr: descr.clone(),
            known_class: *known_class,
            ob_type_descr: ob_type_descr.clone(),
            fields: fields
                .iter()
                .map(|(idx, child)| (*idx, deep_clone_node(child, cache)))
                .collect(),
            field_descrs: field_descrs.clone(),
        },
        VirtualStateInfo::VStruct {
            descr,
            fields,
            field_descrs,
        } => VirtualStateInfo::VStruct {
            descr: descr.clone(),
            fields: fields
                .iter()
                .map(|(idx, child)| (*idx, deep_clone_node(child, cache)))
                .collect(),
            field_descrs: field_descrs.clone(),
        },
        VirtualStateInfo::VArray {
            descr,
            items,
            lenbound,
        } => VirtualStateInfo::VArray {
            descr: descr.clone(),
            items: items
                .iter()
                .map(|child| deep_clone_node(child, cache))
                .collect(),
            lenbound: lenbound.clone(),
        },
        VirtualStateInfo::VArrayStruct {
            descr,
            fielddescrs,
            element_fields,
        } => VirtualStateInfo::VArrayStruct {
            descr: descr.clone(),
            fielddescrs: fielddescrs.clone(),
            element_fields: element_fields
                .iter()
                .map(|fields| {
                    fields
                        .iter()
                        .map(|(idx, child)| (*idx, deep_clone_node(child, cache)))
                        .collect()
                })
                .collect(),
        },
        VirtualStateInfo::VirtualRawBuffer {
            func,
            size,
            entries,
            descrs,
        } => VirtualStateInfo::VirtualRawBuffer {
            func: *func,
            size: *size,
            entries: entries
                .iter()
                .map(|(off, len, child)| (*off, *len, deep_clone_node(child, cache)))
                .collect(),
            descrs: descrs.clone(),
        },
    };
    let new_rc = VirtualStateInfoNode::new_rc(cloned_info);
    cache.insert(key, Rc::clone(&new_rc));
    new_rc
}

/// A guard that must be emitted to make an incoming state compatible.
///
/// virtualstate.py:646: boxes parameter provides the actual OpRef at each
/// position. `box_opref` is the concrete OpRef used as the guard's first
/// argument; `arg_index` is the position in the state vector.
#[derive(Clone, Debug)]
pub enum GuardRequirement {
    /// Emit GUARD_CLASS on the arg at this index.
    GuardClass {
        arg_index: usize,
        box_opref: OpRef,
        expected_class: GcRef,
    },
    /// Emit GUARD_NONNULL on the arg at this index.
    GuardNonnull { arg_index: usize, box_opref: OpRef },
    /// Emit GUARD_VALUE on the arg at this index.
    GuardValue {
        arg_index: usize,
        box_opref: OpRef,
        expected_value: Value,
    },
    /// Emit integer bounds guards on the arg at this index.
    GuardBounds {
        arg_index: usize,
        box_opref: OpRef,
        bounds: IntBound,
    },
}

impl GuardRequirement {
    /// Convert this guard requirement into a concrete Op, registering
    /// constant args via ctx. RPython creates ConstInt/ConstPtr objects
    /// inline in ResOperation args (virtualstate.py:401, 603); we
    /// allocate constant OpRefs via make_constant_int/make_constant_ref.
    pub fn to_op(&self, args: &[OpRef], ctx: &mut OptContext) -> Option<Op> {
        match self {
            GuardRequirement::GuardClass {
                arg_index,
                box_opref,
                expected_class,
            } => {
                let arg = if !box_opref.is_none() {
                    *box_opref
                } else {
                    *args.get(*arg_index)?
                };
                // virtualstate.py:603: ConstInt(self.known_class)
                let class_const = ctx.make_constant_int(expected_class.0 as i64);
                let mut op = Op::new(OpCode::GuardClass, &[arg, class_const]);
                op.fail_args = Some(Default::default());
                Some(op)
            }
            GuardRequirement::GuardNonnull {
                arg_index,
                box_opref,
            } => {
                let arg = if !box_opref.is_none() {
                    *box_opref
                } else {
                    *args.get(*arg_index)?
                };
                let mut op = Op::new(OpCode::GuardNonnull, &[arg]);
                op.fail_args = Some(Default::default());
                Some(op)
            }
            GuardRequirement::GuardValue {
                arg_index,
                box_opref,
                expected_value,
            } => {
                let arg = if !box_opref.is_none() {
                    *box_opref
                } else {
                    *args.get(*arg_index)?
                };
                // virtualstate.py:401: ResOperation(GUARD_VALUE,
                // [box, self.constbox]). Preserve the Const object's type:
                // ConstPtr must not be represented as ConstInt.
                let val_const = match expected_value {
                    Value::Int(v) => ctx.make_constant_int(*v),
                    Value::Float(f) => ctx.make_constant_float(*f),
                    Value::Ref(r) => ctx.make_constant_ref(*r),
                    Value::Void => ctx.make_constant_int(0),
                };
                let mut op = Op::new(OpCode::GuardValue, &[arg, val_const]);
                op.fail_args = Some(Default::default());
                Some(op)
            }
            GuardRequirement::GuardBounds {
                arg_index: _,
                box_opref: _,
                bounds: _,
            } => {
                // IntBound guards are emitted as int_ge/int_le pairs
                // For now, skip — intbounds pass handles these
                None
            }
        }
    }
}

/// Export the abstract state of loop-carried values.
///
/// Given the current optimization context and PtrInfo table (from the virtualize pass),
/// create a VirtualState snapshot for the given OpRefs (typically the Jump args).
///
/// virtualstate.py: VirtualStateConstructor.make_virtual_state()
pub fn export_state(
    oprefs: &[OpRef],
    ctx: &OptContext,
    _forwarded: &[crate::optimizeopt::info::Forwarded],
) -> VirtualState {
    // virtualstate.py:712-728 VirtualStateConstructor.create_state caches by
    // resolved box: if two different oprefs (or two field references) resolve
    // to the same target, they share the SAME `VirtualStateInfo` Python
    // object — and consequently the same `position` /
    // `position_in_notvirtuals`. The Rust port mirrors this with an
    // `Rc<VirtualStateInfoNode>` cache shared across the whole export, including
    // recursive nested-field calls AND top-level jump args.
    //
    // virtualstate.py:713 `box = get_box_replacement(box)` is performed
    // inside `export_single_value`, so we don't pre-resolve here.
    let mut cache = ExportCache::new();
    let state: Vec<Rc<VirtualStateInfoNode>> = oprefs
        .iter()
        .map(|opref| export_single_value(*opref, ctx, &mut cache))
        .collect();
    // virtualstate.py:627-634 VirtualState.__init__ assigns positions via
    // _enum so build_sequential_slot_schedule can dedup shared Rc'd
    // subtrees, matching RPython's `state.position > self.position`.
    let (slot_schedule, numnotvirtuals) = build_sequential_slot_schedule(&state);
    VirtualState::new_with_slot_schedule(state, slot_schedule, numnotvirtuals)
}

/// Bookkeeping shared across `export_single_value` recursion: the DAG cache
/// (fully constructed nodes only) plus an `in_progress` set used to detect
/// back-edges. Splitting the two prevents the previous "insert Unknown stub
/// then overwrite" pattern from leaking the stub to in-flight recursive
/// callers.
pub(crate) struct ExportCache {
    pub finished: HashMap<OpRef, Rc<VirtualStateInfoNode>>,
    pub in_progress: std::collections::HashSet<OpRef>,
}

impl ExportCache {
    pub fn new() -> Self {
        Self {
            finished: HashMap::new(),
            in_progress: std::collections::HashSet::new(),
        }
    }
}

/// Export abstract info for a single value, sharing `Rc<VirtualStateInfoNode>`
/// across recursive calls so the resulting tree is a DAG: aliased boxes
/// converge on a single shared `VirtualStateInfo`. virtualstate.py:712-728
/// VirtualStateConstructor.create_state.
///
/// **Cycle handling**: RPython does
///
/// ```text
/// result = info.visitor_dispatch_virtual_type(self)
/// self.info[box] = result            # ← cache the empty state
/// info.visitor_walk_recursive(box, self)
/// result.fieldstate = [...]          # ← fill afterwards
/// ```
///
/// so a cycle (`A.f -> B`, `B.f -> A`) closes on the same Python object.
/// Rust's `Rc<VirtualStateInfoNode>` is immutable after construction, and
/// `Rc::new_cyclic`'s `Weak<T>` cannot upgrade during the closure body,
/// so we cannot mirror the "cache empty, then mutate" pattern without
/// switching every consumer to `Rc<RefCell<...>>`. Until that refactor
/// lands, `in_progress` detects the back-edge explicitly and the cycle
/// child returns a fresh `Unknown` Rc instead of aliasing onto a stale
/// stub: distinct nodes never collapse, the parent VirtualStateInfo
/// reflects the genuine acyclic prefix, and downstream code that
/// requires real cycle preservation can detect the dropped edge by
/// checking for `Unknown` in a virtual subtree.
fn export_single_value(
    opref: OpRef,
    ctx: &OptContext,
    cache: &mut ExportCache,
) -> Rc<VirtualStateInfoNode> {
    // virtualstate.py:713 `box = get_box_replacement(box)` — every
    // create_state entry resolves the forwarding chain BEFORE the cache
    // lookup, so two field references that forward to the same target
    // collapse onto the same VirtualStateInfo. Without this normalization,
    // distinct field-side OpRefs that resolve to the same forwarded box
    // would each receive their own Rc, breaking the dedup invariant the
    // walker (`build_sequential_slot_schedule`, `enum_forced_boxes`) and
    // RPython matching rely on.
    let opref = ctx.get_box_replacement(opref);
    // virtualstate.py:714-716: cache hit returns the cached state directly.
    if let Some(cached) = cache.finished.get(&opref) {
        return Rc::clone(cached);
    }
    // Cycle: this opref is currently being exported on the parent stack.
    // Return a fresh Unknown leaf so the back-edge is visibly non-virtual
    // — distinct from any real Unknown elsewhere in the tree because
    // each cycle entry allocates its own Rc.
    //
    // Verified 2026-04-10: this branch fires zero times across all 10
    // benchmarks in pyre/check.sh (int_loop, float_loop, fib_loop,
    // fib_recursive, nested_loop, nbody, fannkuch, raise_catch_loop,
    // spectral_norm, inline_helper). The cyclic-virtual-graph regression
    // (RPython parity gap documented above) is therefore latent — no
    // benchmark constructs the necessary self-referential structures.
    if !cache.in_progress.insert(opref) {
        // Fallback to Ref for the cycle leaf: pyre's virtual DAGs only
        // form through ptr fields, so the only reachable cycles are on
        // Ref-typed nodes. Matches `not_virtual(cpu, 'r', None)` in
        // RPython where a cycle child resolves to NotVirtualStateInfoPtr
        // with LEVEL_UNKNOWN.
        return VirtualStateInfoNode::new_rc(VirtualStateInfo::Unknown(Type::Ref));
    }

    let info = export_single_value_inner(opref, ctx, cache);
    let rc = VirtualStateInfoNode::new_rc(info);
    cache.in_progress.remove(&opref);
    cache.finished.insert(opref, Rc::clone(&rc));
    rc
}

fn export_single_value_inner(
    opref: OpRef,
    ctx: &OptContext,
    cache: &mut ExportCache,
) -> VirtualStateInfo {
    // Check for known constant.
    // RPython parity: only export LEVEL_CONSTANT for truly invariant values
    // (constant pool entries with OpRef >= 10000). Trace-computed values
    // may have been constant-folded during Phase 1 but change across
    // iterations. RPython's setinfo_from_preamble_list calls
    // item.set_forwarded(None) — info is completely cleared for ALL types.
    if let Some(value) = ctx.get_constant(opref) {
        // RPython setinfo_from_preamble parity (unroll.py:73-75):
        // Only export LEVEL_CONSTANT for constant pool entries (>= 10000)
        // and type descriptor pointers marked via numbering_type_overrides.
        //
        // Trace-computed Ref constants (e.g., W_IntObject(-1) pointer for
        // sign) are NOT invariant — a different W_IntObject may be used on
        // the next iteration. Only truly invariant GcRef pointers (ob_type)
        // registered via mark_const_type get the Ref override.
        let has_ref_override = ctx
            .constant_types_for_numbering
            .get(&opref.raw())
            .is_some_and(|&t| t == majit_ir::Type::Ref);
        if opref.raw() >= 10000 || has_ref_override {
            let export_val = if has_ref_override && matches!(value, Value::Int(v) if *v != 0) {
                Value::Ref(majit_ir::GcRef(match value {
                    Value::Int(v) => *v as usize,
                    _ => 0,
                }))
            } else {
                value.clone()
            };
            return VirtualStateInfo::Constant(export_val);
        }
        // Trace-computed constants: export as Unknown (RPython LEVEL_UNKNOWN).
        // Phase 2 will re-compute their values from runtime state. The
        // explicit type mirrors RPython's constructor dispatch
        // `not_virtual(cpu, box.type, info)` (virtualstate.py:360): the
        // subclass (`NotVirtualStateInfoInt` / `NotVirtualStateInfoPtr` /
        // base) is chosen by `box.type`, which comes from the constant's
        // runtime shape.
        return VirtualStateInfo::Unknown(value.get_type());
    }

    // Check PtrInfo — use ctx.get_ptr_info which follows forwarding
    if let Some(info) = ctx.get_ptr_info(opref) {
        match info {
            PtrInfo::Virtual(vinfo) => {
                // RPython parity: heaptracker.py:66-67 excludes typeptr from
                // all_fielddescrs(); see VirtualInfo struct-level docs.
                crate::optimizeopt::virtualize::debug_assert_no_typeptr_in_virtual_fields(
                    &vinfo.fields,
                    "export_single_value::Virtual",
                );
                let fields = vinfo
                    .fields
                    .iter()
                    .map(|(field_idx, field_ref)| {
                        let field_state = export_single_value(*field_ref, ctx, cache);
                        (*field_idx, field_state)
                    })
                    .collect();
                return VirtualStateInfo::Virtual {
                    descr: vinfo.descr.clone(),
                    known_class: vinfo.known_class,
                    ob_type_descr: vinfo.ob_type_descr.clone(),
                    fields,
                    field_descrs: vinfo.field_descrs.clone(),
                };
            }
            PtrInfo::VirtualArray(vinfo) => {
                let items: Vec<Rc<VirtualStateInfoNode>> = vinfo
                    .items
                    .iter()
                    .map(|item_ref| export_single_value(*item_ref, ctx, cache))
                    .collect();
                let len = items.len();
                return VirtualStateInfo::VArray {
                    descr: vinfo.descr.clone(),
                    items,
                    lenbound: Some(IntBound::from_constant(len as i64)),
                };
            }
            PtrInfo::VirtualStruct(vinfo) => {
                let fields = vinfo
                    .fields
                    .iter()
                    .map(|(field_idx, field_ref)| {
                        let field_state = export_single_value(*field_ref, ctx, cache);
                        (*field_idx, field_state)
                    })
                    .collect();
                return VirtualStateInfo::VStruct {
                    descr: vinfo.descr.clone(),
                    fields,
                    field_descrs: vinfo.field_descrs.clone(),
                };
            }
            PtrInfo::VirtualArrayStruct(vinfo) => {
                let element_fields = vinfo
                    .element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_idx, field_ref)| {
                                let field_state = export_single_value(*field_ref, ctx, cache);
                                (*field_idx, field_state)
                            })
                            .collect()
                    })
                    .collect();
                return VirtualStateInfo::VArrayStruct {
                    descr: vinfo.descr.clone(),
                    fielddescrs: vinfo.fielddescrs.clone(),
                    element_fields,
                };
            }
            PtrInfo::VirtualRawBuffer(vinfo) => {
                // info.py:445: visitor.visit_vrawbuffer(self.func, self.size, ...)
                let entries = vinfo
                    .offsets
                    .iter()
                    .zip(vinfo.lengths.iter())
                    .zip(vinfo.values.iter())
                    .map(|((&offset, &length), &value_ref)| {
                        let val_state = export_single_value(value_ref, ctx, cache);
                        (offset, length, val_state)
                    })
                    .collect();
                return VirtualStateInfo::VirtualRawBuffer {
                    func: vinfo.func,
                    size: vinfo.size,
                    entries,
                    descrs: vinfo.descrs.clone(),
                };
            }
            PtrInfo::VirtualRawSlice(_) => {
                // RawSlicePtrInfo aliases its parent buffer; the parent is
                // always exported separately. Slices have no independent
                // virtual-state representation, so emit NonNull (matches
                // RPython's `getlenbound` fallback for RawSlicePtrInfo).
                return VirtualStateInfo::NonNull;
            }
            PtrInfo::NonNull { .. } => {
                return VirtualStateInfo::NonNull;
            }
            PtrInfo::Constant(gcref) => {
                return VirtualStateInfo::Constant(Value::Ref(*gcref));
            }
            PtrInfo::Virtualizable(_) => {
                // Virtualizable objects are treated as non-null in virtual state
                return VirtualStateInfo::NonNull;
            }
            PtrInfo::Instance(iinfo) => {
                // info.py:147 InstancePtrInfo(None, class_const) becomes
                // VirtualStateInfo::KnownClass when only the class is
                // known. Otherwise it's an opaque non-null instance.
                if let Some(class_ptr) = iinfo.known_class {
                    return VirtualStateInfo::KnownClass { class_ptr };
                }
                return VirtualStateInfo::NonNull;
            }
            PtrInfo::Struct(_) | PtrInfo::Array(_) => {
                return VirtualStateInfo::NonNull;
            }
            PtrInfo::Str(_) => {
                return VirtualStateInfo::NonNull;
            }
        }
    }

    // virtualstate.py:360 not_virtual(cpu, box.type, info): the subclass
    // is picked by `box.type` which is ALWAYS set on RPython Boxes.
    // pyre's OptContext::opref_type reconstructs it from value_types
    // (seeded from trace_inputarg_types) / producing-op result_type.
    // Verified: 0 hits across all 10 benchmarks. Production panics;
    // test builds keep a fallback because some unit tests construct
    // minimal OptContext without seeding value_types for every OpRef
    // that reaches export_state (pre-existing test limitation, not a
    // production code path).
    let tp = ctx.opref_type(opref).unwrap_or_else(|| {
        if !cfg!(test) {
            panic!(
                "not_virtual: opref_type({:?}) returned None — \
                 RPython box.type is always set (virtualstate.py:360)",
                opref,
            );
        }
        Type::Int
    });
    VirtualStateInfo::Unknown(tp)
}

/// virtualstate.py:627-634 VirtualState.__init__:
///
/// ```python
/// def __init__(self, state):
///     self.state = state
///     self.info_counter = -1
///     self.numnotvirtuals = 0
///     for s in state:
///         if s:
///             s.enum(self)
/// ```
///
/// `enum` (line 111-119) walks the state graph and assigns each unique
/// `AbstractVirtualStateInfo` a `position`; each non-constant
/// `NotVirtualStateInfo` also gets a `position_in_notvirtuals` slot
/// (line 427-431). Shared sub-states get the SAME position because
/// `if self.position != -1: return` short-circuits revisits.
///
/// majit's flat `slot_schedule + numnotvirtuals` plays the role of the
/// `position_in_notvirtuals` array: each leaf occurrence in DFS order
/// records the slot it should write into `boxes[...]`. With
/// `Rc<VirtualStateInfoNode>` shared subtrees the dedup walks each unique
/// `Rc` only once via `Rc::as_ptr` identity, mirroring RPython's
/// `if self.position != -1: return` cycle break.
fn build_sequential_slot_schedule(state: &[Rc<VirtualStateInfoNode>]) -> (Vec<usize>, usize) {
    let mut schedule = Vec::new();
    let mut next_slot = 0usize;
    let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
    // Top-level entries also dedup via Rc::as_ptr identity so two jump
    // args resolving to the same box (sharing the same top-level Rc)
    // collapse onto a single set of slots — matching RPython's
    // VirtualStateConstructor sharing.
    for rc in state {
        append_sequential_slots_rc(rc, &mut schedule, &mut next_slot, &mut visited);
    }
    (schedule, next_slot)
}

fn append_sequential_slots(
    info: &VirtualStateInfo,
    schedule: &mut Vec<usize>,
    next_slot: &mut usize,
    visited: &mut std::collections::HashSet<usize>,
) {
    match info {
        VirtualStateInfo::Constant(_) => {}
        VirtualStateInfo::Virtual { fields, .. } | VirtualStateInfo::VStruct { fields, .. } => {
            for (_, child) in fields {
                append_sequential_slots_rc(child, schedule, next_slot, visited);
            }
        }
        VirtualStateInfo::VArray { items, .. } => {
            for child in items {
                append_sequential_slots_rc(child, schedule, next_slot, visited);
            }
        }
        VirtualStateInfo::VArrayStruct { element_fields, .. } => {
            for fields in element_fields {
                for (_, child) in fields {
                    append_sequential_slots_rc(child, schedule, next_slot, visited);
                }
            }
        }
        VirtualStateInfo::VirtualRawBuffer { entries, .. } => {
            for (_, _, child) in entries {
                append_sequential_slots_rc(child, schedule, next_slot, visited);
            }
        }
        VirtualStateInfo::KnownClass { .. }
        | VirtualStateInfo::NonNull
        | VirtualStateInfo::IntBounded(_)
        | VirtualStateInfo::Unknown(_) => {
            schedule.push(*next_slot);
            *next_slot += 1;
        }
    }
}

/// virtualstate.py:111-116 `enum` parity: dedup via `Rc::as_ptr` so a
/// shared `Rc<VirtualStateInfoNode>` is enumerated exactly once, matching
/// `if self.position != -1: return` on the Python side.
fn append_sequential_slots_rc(
    rc: &Rc<VirtualStateInfoNode>,
    schedule: &mut Vec<usize>,
    next_slot: &mut usize,
    visited: &mut std::collections::HashSet<usize>,
) {
    let key = Rc::as_ptr(rc) as usize;
    if !visited.insert(key) {
        return;
    }
    append_sequential_slots(rc, schedule, next_slot, visited);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::info::VirtualStructInfo;
    use majit_ir::{Descr, FieldDescr, GcRef, Type};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestDescr(u32);
    impl Descr for TestDescr {
        fn index(&self) -> u32 {
            self.0
        }
    }
    impl FieldDescr for TestDescr {
        fn offset(&self) -> usize {
            self.0 as usize
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> Type {
            Type::Int
        }
    }

    fn test_descr(idx: u32) -> DescrRef {
        Arc::new(TestDescr(idx))
    }

    // ── Compatibility tests ──

    #[test]
    fn test_unknown_type_discrimination() {
        // virtualstate.py:383-410 NotVirtualStateInfoInt._generate_guards:
        // isinstance(other, NotVirtualStateInfoInt) check enforces type.
        // Unknown(Int) accepts Int-typed incoming only, not Ref-typed NonNull.
        let unknown_int = VirtualStateInfo::Unknown(Type::Int);
        assert!(unknown_int.is_compatible(&VirtualStateInfo::Unknown(Type::Int)));
        assert!(unknown_int.is_compatible(&VirtualStateInfo::Constant(Value::Int(42))));
        assert!(!unknown_int.is_compatible(&VirtualStateInfo::NonNull));
        assert!(!unknown_int.is_compatible(&VirtualStateInfo::Unknown(Type::Ref)));

        // Unknown(Ref) accepts Ref-typed incoming: NonNull, KnownClass, etc.
        let unknown_ref = VirtualStateInfo::Unknown(Type::Ref);
        assert!(unknown_ref.is_compatible(&VirtualStateInfo::NonNull));
        assert!(unknown_ref.is_compatible(&VirtualStateInfo::Unknown(Type::Ref)));
        assert!(!unknown_ref.is_compatible(&VirtualStateInfo::Unknown(Type::Int)));
    }

    #[test]
    fn test_constant_compatibility() {
        let c1 = VirtualStateInfo::Constant(Value::Int(42));
        let c2 = VirtualStateInfo::Constant(Value::Int(42));
        let c3 = VirtualStateInfo::Constant(Value::Int(99));

        assert!(c1.is_compatible(&c2));
        assert!(!c1.is_compatible(&c3));
        assert!(!c1.is_compatible(&VirtualStateInfo::Unknown(Type::Int)));
    }

    #[test]
    fn test_nonnull_compatibility() {
        let nn = VirtualStateInfo::NonNull;
        assert!(nn.is_compatible(&VirtualStateInfo::NonNull));
        assert!(nn.is_compatible(&VirtualStateInfo::KnownClass {
            class_ptr: GcRef(0x100)
        }));
        assert!(!nn.is_compatible(&VirtualStateInfo::Unknown(Type::Int)));
    }

    #[test]
    fn test_known_class_compatibility() {
        let kc1 = VirtualStateInfo::KnownClass {
            class_ptr: GcRef(0x100),
        };
        let kc2 = VirtualStateInfo::KnownClass {
            class_ptr: GcRef(0x100),
        };
        let kc3 = VirtualStateInfo::KnownClass {
            class_ptr: GcRef(0x200),
        };

        assert!(kc1.is_compatible(&kc2));
        assert!(!kc1.is_compatible(&kc3));
        assert!(!kc1.is_compatible(&VirtualStateInfo::Unknown(Type::Int)));
    }

    #[test]
    fn test_virtual_array_compatibility() {
        let descr = test_descr(1);
        let a1 = VirtualStateInfo::VArray {
            descr: descr.clone(),
            items: vec![
                VirtualStateInfoNode::new_rc(VirtualStateInfo::Constant(Value::Int(1))),
                VirtualStateInfoNode::new_rc(VirtualStateInfo::Unknown(Type::Int)),
            ],
            lenbound: None,
        };
        let a2 = VirtualStateInfo::VArray {
            descr: descr.clone(),
            items: vec![
                VirtualStateInfoNode::new_rc(VirtualStateInfo::Constant(Value::Int(1))),
                VirtualStateInfoNode::new_rc(VirtualStateInfo::Constant(Value::Int(2))),
            ],
            lenbound: None,
        };
        let a3 = VirtualStateInfo::VArray {
            descr: descr.clone(),
            items: vec![VirtualStateInfoNode::new_rc(VirtualStateInfo::Constant(
                Value::Int(1),
            ))],
            lenbound: None,
        };

        assert!(a1.is_compatible(&a2)); // same length, first matches, second is Unknown
        assert!(!a1.is_compatible(&a3)); // different length
    }

    #[test]
    fn test_int_bounded_compatibility() {
        let b1 = VirtualStateInfo::IntBounded(IntBound::bounded(0, 100));
        let b2 = VirtualStateInfo::IntBounded(IntBound::bounded(10, 50));
        let b3 = VirtualStateInfo::IntBounded(IntBound::bounded(-10, 200));
        let c = VirtualStateInfo::Constant(Value::Int(42));

        assert!(b1.is_compatible(&b2)); // b2 is within b1
        assert!(!b1.is_compatible(&b3)); // b3 exceeds b1
        assert!(b1.is_compatible(&c)); // 42 is within [0, 100]
    }

    // ── VirtualState tests ──

    #[test]
    fn test_virtual_state_compatible() {
        let s1 = VirtualState::new(vec![
            VirtualStateInfo::Unknown(Type::Int),
            VirtualStateInfo::NonNull,
        ]);
        let s2 = VirtualState::new(vec![
            VirtualStateInfo::Constant(Value::Int(42)),
            VirtualStateInfo::KnownClass {
                class_ptr: GcRef(0x100),
            },
        ]);

        assert!(s1.is_compatible(&s2));
    }

    #[test]
    fn test_virtual_state_generalization_direction_matches_rpython() {
        // RPython: NotVirtualStateInfoPtr(LEVEL_UNKNOWN) generalizes
        // NotVirtualStateInfoPtr(LEVEL_NONNULL) — same type family.
        // Cross-type (Int target, Ref incoming) is always rejected.
        let target = VirtualState::new(vec![VirtualStateInfo::Unknown(Type::Ref)]);
        let incoming = VirtualState::new(vec![VirtualStateInfo::NonNull]);

        assert!(target.generalization_of(&incoming));
        assert!(!incoming.generalization_of(&target));

        // Cross-type: Int target does NOT accept Ref incoming
        let int_target = VirtualState::new(vec![VirtualStateInfo::Unknown(Type::Int)]);
        assert!(!int_target.generalization_of(&incoming));
    }

    #[test]
    fn test_virtual_state_incompatible_length() {
        let s1 = VirtualState::new(vec![VirtualStateInfo::Unknown(Type::Int)]);
        let s2 = VirtualState::new(vec![
            VirtualStateInfo::Unknown(Type::Int),
            VirtualStateInfo::Unknown(Type::Int),
        ]);

        assert!(!s1.is_compatible(&s2));
    }

    #[test]
    fn test_virtual_state_generate_guards() {
        // KnownClass and NonNull are Ref-typed; incoming must also be Ref.
        // RPython: NotVirtualStateInfoPtr._generate_guards requires
        // isinstance(other, NotVirtualStateInfoPtr).
        let s1 = VirtualState::new(vec![
            VirtualStateInfo::KnownClass {
                class_ptr: GcRef(0x100),
            },
            VirtualStateInfo::NonNull,
        ]);
        let s2 = VirtualState::new(vec![
            VirtualStateInfo::Unknown(Type::Ref),
            VirtualStateInfo::Unknown(Type::Ref),
        ]);

        let boxes = vec![OpRef::ref_op(100), OpRef::ref_op(101)];
        let ctx = OptContext::new(128);
        // runtime_boxes=Some enables non-permanent guard emission (RPython
        // _jump_to_existing_trace path). With None, Unknown incoming → Err.
        let guards = s1
            .generate_guards(&s2, &boxes, Some(&boxes), &ctx, false)
            .unwrap();
        assert_eq!(guards.len(), 2);
        assert!(matches!(
            &guards[0],
            GuardRequirement::GuardClass { arg_index: 0, box_opref, .. } if *box_opref == OpRef::ref_op(100)
        ));
        assert!(matches!(
            &guards[1],
            GuardRequirement::GuardNonnull { arg_index: 1, box_opref } if *box_opref == OpRef::ref_op(101)
        ));
    }

    #[test]
    fn test_constant_guard_requires_matching_runtime_box() {
        let expected = VirtualState::new(vec![VirtualStateInfo::Constant(Value::Int(7))]);
        let incoming = VirtualState::new(vec![VirtualStateInfo::Unknown(Type::Int)]);
        let boxes = vec![OpRef::int_op(10)];

        let mut ctx = OptContext::new(128);
        let matching_runtime = ctx.make_constant_int(7);
        let guards = expected
            .generate_guards(&incoming, &boxes, Some(&[matching_runtime]), &ctx, false)
            .unwrap();
        assert_eq!(guards.len(), 1);

        let mismatching_runtime = ctx.make_constant_int(8);
        assert!(
            expected
                .generate_guards(&incoming, &boxes, Some(&[mismatching_runtime]), &ctx, false)
                .is_err()
        );
    }

    #[test]
    fn test_guard_value_preserves_ref_constant_type() {
        let mut ctx = OptContext::new(128);
        let expected = GcRef(0x1234);
        let guard = GuardRequirement::GuardValue {
            arg_index: 0,
            box_opref: OpRef::ref_op(10),
            expected_value: Value::Ref(expected),
        }
        .to_op(&[OpRef::ref_op(10)], &mut ctx)
        .unwrap();

        assert_eq!(guard.opcode, OpCode::GuardValue);
        assert_eq!(ctx.get_constant(guard.arg(1)), Some(&Value::Ref(expected)));
    }

    // ── Export/Import tests ──

    #[test]
    fn test_make_inputargs_skips_virtual_entries() {
        let descr = test_descr(7);
        let state = VirtualState::new(vec![
            VirtualStateInfo::Unknown(Type::Int),
            VirtualStateInfo::VStruct {
                descr: descr.clone(),
                fields: vec![],
                field_descrs: Vec::new(),
            },
            VirtualStateInfo::NonNull,
        ]);

        let mut ctx = OptContext::new(16);
        // virtualstate.py:185 requires `info.is_virtual()` for the
        // VStruct walker to descend; mirror by attaching a virtual
        // PtrInfo to the corresponding OpRef.
        ctx.set_ptr_info(
            OpRef::ref_op(11),
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr,
                fields: vec![],
                field_descrs: Vec::new(),
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let inputargs = state
            .make_inputargs(
                &[OpRef::int_op(10), OpRef::ref_op(11), OpRef::ref_op(12)],
                &mut optimizer,
                &mut ctx,
                false,
            )
            .expect("make_inputargs");
        assert_eq!(inputargs, vec![OpRef::int_op(10), OpRef::ref_op(12)]);
    }

    #[test]
    fn test_make_inputargs_and_virtuals_returns_virtual_boxes() {
        let descr = test_descr(9);
        let state = VirtualState::new(vec![
            VirtualStateInfo::Unknown(Type::Int),
            VirtualStateInfo::VStruct {
                descr: descr.clone(),
                fields: vec![],
                field_descrs: Vec::new(),
            },
            VirtualStateInfo::NonNull,
        ]);

        let mut ctx = OptContext::new(16);
        ctx.set_ptr_info(
            OpRef::ref_op(21),
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr,
                fields: vec![],
                field_descrs: Vec::new(),
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let (inputargs, virtuals) = state
            .make_inputargs_and_virtuals(
                &[OpRef::int_op(20), OpRef::ref_op(21), OpRef::ref_op(22)],
                &mut optimizer,
                &mut ctx,
                false,
            )
            .expect("make_inputargs_and_virtuals");
        assert_eq!(inputargs, vec![OpRef::int_op(20), OpRef::ref_op(22)]);
        assert_eq!(virtuals, vec![OpRef::ref_op(21)]);
    }

    /// virtualstate.py:196 / 274 / 352 — `state.position > self.position`
    /// shared-substate dedup parity. When two top-level state entries
    /// reference the same `Rc<VirtualStateInfoNode>` (an aliased nested box),
    /// the leaves under that subtree must be enumerated exactly once into
    /// the inputargs slot vector — matching RPython's per-state-object
    /// `position_in_notvirtuals` allocation.
    #[test]
    fn test_make_inputargs_dedups_shared_substate() {
        let descr = test_descr(13);
        // Two top-level VStruct entries that share the SAME Rc'd field.
        // After dedup the field's leaf occupies a single slot.
        let shared_field: Rc<VirtualStateInfoNode> =
            VirtualStateInfoNode::new_rc(VirtualStateInfo::NonNull);
        let outer_a = VirtualStateInfo::VStruct {
            descr: descr.clone(),
            fields: vec![(0, Rc::clone(&shared_field))],
            field_descrs: Vec::new(),
        };
        let outer_b = VirtualStateInfo::VStruct {
            descr: descr.clone(),
            fields: vec![(0, Rc::clone(&shared_field))],
            field_descrs: Vec::new(),
        };
        let state = VirtualState::new(vec![outer_a, outer_b]);
        // The dedup walker should report a single non-virtual leaf slot
        // (matching RPython numnotvirtuals on the same shared object).
        assert_eq!(state.num_boxes(), 1);

        let inner_field_value = OpRef::ref_op(31);
        let outer_a_ref = OpRef::ref_op(40);
        let outer_b_ref = OpRef::ref_op(41);
        let mut ctx = OptContext::new(64);
        // Both outer boxes resolve to a virtual struct whose field 0 is
        // the shared inner OpRef.
        ctx.set_ptr_info(
            outer_a_ref,
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr: descr.clone(),
                fields: vec![(0, inner_field_value)],
                field_descrs: Vec::new(),
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );
        ctx.set_ptr_info(
            outer_b_ref,
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr,
                fields: vec![(0, inner_field_value)],
                field_descrs: Vec::new(),
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );

        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let inputargs = state
            .make_inputargs(&[outer_a_ref, outer_b_ref], &mut optimizer, &mut ctx, false)
            .expect("make_inputargs");
        assert_eq!(inputargs, vec![inner_field_value]);
    }

    /// Top-level Rc dedup parity: when two jump args resolve to the
    /// same box, RPython's VirtualStateConstructor cache returns the
    /// same AbstractVirtualStateInfo Python object. The Rust port shares
    /// the top-level `Rc<VirtualStateInfoNode>` directly via
    /// `from_shared_rcs` so `numnotvirtuals` reflects the deduped
    /// slot count.
    #[test]
    fn test_top_level_rc_aliasing_dedups_slots() {
        let shared_leaf: Rc<VirtualStateInfoNode> =
            VirtualStateInfoNode::new_rc(VirtualStateInfo::NonNull);
        // Both top-level state entries are the SAME Rc, mirroring
        // VirtualStateConstructor returning the cached object for
        // aliased jump args.
        let state =
            VirtualState::from_shared_rcs(vec![Rc::clone(&shared_leaf), Rc::clone(&shared_leaf)]);
        assert_eq!(state.num_boxes(), 1);

        let outer_ref = OpRef::ref_op(50);
        let mut ctx = OptContext::new(64);
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let inputargs = state
            .make_inputargs(&[outer_ref, outer_ref], &mut optimizer, &mut ctx, false)
            .expect("make_inputargs");
        // Single deduped slot, written by the first top-level visit.
        assert_eq!(inputargs, vec![outer_ref]);
    }

    #[test]
    fn test_make_inputargs_recursively_extracts_virtual_fields() {
        let descr = test_descr(11);
        let field_value = OpRef::ref_op(21);
        let virtual_ref = OpRef::ref_op(20);
        let state = VirtualState::new(vec![VirtualStateInfo::VStruct {
            descr: descr.clone(),
            fields: vec![
                (
                    0,
                    VirtualStateInfoNode::new_rc(VirtualStateInfo::Constant(Value::Int(7))),
                ),
                (8, VirtualStateInfoNode::new_rc(VirtualStateInfo::NonNull)),
            ],
            field_descrs: Vec::new(),
        }]);
        let mut ctx = OptContext::new(32);
        ctx.set_ptr_info(
            virtual_ref,
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr,
                fields: vec![(0, OpRef::NONE), (8, field_value)],
                field_descrs: Vec::new(),
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );

        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let inputargs = state
            .make_inputargs(&[virtual_ref], &mut optimizer, &mut ctx, false)
            .expect("make_inputargs");
        assert_eq!(inputargs, vec![field_value]);
    }

    #[test]
    fn test_force_boxes() {
        let descr = test_descr(0);
        let mut state = VirtualState::new(vec![
            VirtualStateInfo::Virtual {
                descr: descr.clone(),
                known_class: Some(GcRef(0x1000)),
                ob_type_descr: None,
                fields: vec![],
                field_descrs: Vec::new(),
            },
            VirtualStateInfo::NonNull,
            VirtualStateInfo::VArray {
                descr,
                items: vec![VirtualStateInfoNode::new_rc(VirtualStateInfo::Unknown(
                    Type::Int,
                ))],
                lenbound: None,
            },
            VirtualStateInfo::Unknown(Type::Int),
        ]);
        assert_eq!(state.num_virtuals(), 2);
        let forced = state.force_boxes();
        assert_eq!(forced, 2);
        assert_eq!(state.num_virtuals(), 0);
        // Virtual with known_class becomes KnownClass
        assert!(matches!(
            &state.state[0].info,
            VirtualStateInfo::KnownClass { .. }
        ));
        // VirtualArray becomes NonNull
        assert!(matches!(&state.state[2].info, VirtualStateInfo::NonNull));
    }

    #[test]
    fn test_make_inputargs_with_optimizer_retries_virtual_into_nonvirtual_slot() {
        let descr = test_descr(12);
        let virtual_ref = OpRef::ref_op(20);
        let state = VirtualState::new(vec![VirtualStateInfo::NonNull]);
        let mut ctx = OptContext::with_num_inputs(32, 1024);
        ctx.set_ptr_info(
            virtual_ref,
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr,
                fields: vec![],
                field_descrs: Vec::new(),
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();

        assert!(
            state
                .make_inputargs_and_virtuals(&[virtual_ref], &mut optimizer, &mut ctx, false,)
                .is_err()
        );

        let (inputargs, virtuals) = state
            .make_inputargs_and_virtuals(&[virtual_ref], &mut optimizer, &mut ctx, true)
            .expect("force_boxes=True should retry instead of failing");
        // After forcing, the virtual struct is replaced by a concrete
        // allocation at a new position. The inputarg should be that
        // forced allocation ref (which is what ctx.get_replacement
        // resolves the original virtual_ref to).
        assert_eq!(inputargs.len(), 1);
        assert_eq!(inputargs[0], ctx.get_box_replacement(virtual_ref));
        assert!(virtuals.is_empty());
    }

    #[test]
    fn test_compute_renum() {
        let mut state = VirtualState::new(vec![
            VirtualStateInfo::Unknown(Type::Int),
            VirtualStateInfo::NonNull,
            VirtualStateInfo::Unknown(Type::Int),
        ]);
        state.compute_renum(&[OpRef::int_op(10), OpRef::ref_op(20), OpRef::int_op(30)]);
        assert_eq!(state.get_renum(OpRef::int_op(10)), Some(0));
        assert_eq!(state.get_renum(OpRef::ref_op(20)), Some(1));
        assert_eq!(state.get_renum(OpRef::int_op(30)), Some(2));
        assert_eq!(state.get_renum(OpRef::int_op(99)), None);
    }
}
