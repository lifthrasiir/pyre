//! Constant pool for trace recording with GC root tracking.
//!
//! RPython manages constants implicitly in Trace — ConstPtr boxes are
//! GC-managed objects, so GC can update them when objects move.
//!
//! majit stores Ref constants as raw i64 in a HashMap, invisible to GC.
//! To achieve RPython parity, Ref constants are rooted on the shadow
//! stack (gcreftracer.py:GCREFTRACER parity). GC's walk_roots updates
//! shadow stack entries in place; refresh_from_gc copies updated values
//! back to the HashMap before consumption.

use std::collections::HashMap;

use majit_gc::shadow_stack;
use majit_ir::{GcRef, OpRef, Type};

/// Constant pool for trace recording.
///
/// Manages the mapping from constant-namespace OpRef to i64 values.
/// Deduplicates identical values within the same type.
///
/// gcreftracer.py parity: Ref-typed constants are pushed onto the GC
/// shadow stack so that GC can trace and update them if objects move.
/// On consumption (into_inner / snapshot), the HashMap is refreshed
/// from the shadow stack to pick up any GC-updated pointers.
pub struct ConstantPool {
    /// Keyed by OpRef.0 (tagged constant value, i.e. index | CONST_BIT).
    constants: HashMap<u32, i64>,
    /// Type of each constant OpRef. Populated by `get_or_insert_typed`.
    constant_types: HashMap<u32, Type>,
    /// Resume-data-only type overrides. Not exposed to Cranelift backend.
    /// RPython parity: GcRef pointers recorded as const_int need Ref type
    /// for resume data encoding without triggering GC root tracking.
    numbering_type_overrides: HashMap<u32, Type>,
    /// Zero-based counter for allocating new constant indices.
    next_const_idx: u32,
    /// gcreftracer.py parity: (OpRef key, shadow stack index) for each
    /// rooted Ref constant. walk_roots updates shadow stack entries;
    /// refresh_from_gc copies values back to `constants`.
    rooted_refs: Vec<(u32, usize)>,
    /// Shadow stack depth at pool creation. release_roots pops to here.
    shadow_stack_base: usize,
}

impl ConstantPool {
    pub fn new() -> Self {
        ConstantPool {
            constants: HashMap::new(),
            constant_types: HashMap::new(),
            numbering_type_overrides: HashMap::new(),
            next_const_idx: 0,
            rooted_refs: Vec::new(),
            shadow_stack_base: shadow_stack::depth(),
        }
    }

    /// Get or create a constant OpRef for a given i64 value.
    /// Only matches Int-typed entries (not Ref/Float).
    /// Returns the same OpRef for the same value (deduplication).
    ///
    /// RPython parity: equivalent to constructing `ConstInt(value)` and
    /// relying on memo-deduping via `ResumeDataLoopMemo.large_ints`. The
    /// `constant_types` slot is always set to `Type::Int` — RPython Box
    /// subclasses pin the type at creation (history.py:361 ConstInt),
    /// so our constant pool must do the same. Resume-data-only Ref
    /// overrides for raw-pointer ConstInts go through `mark_type` →
    /// `numbering_type_overrides`, which is a separate channel that does
    /// not disturb the intrinsic `op.type == 'i'` invariant.
    pub fn get_or_insert(&mut self, value: i64) -> OpRef {
        for (&idx, &v) in &self.constants {
            if v == value {
                // RPython parity: ConstInt / ConstPtr / ConstFloat are
                // separate classes (history.py:220, 261, 307), so Int
                // and Ref constants with the same raw value MUST NOT
                // alias. Skip Ref/Float entries. Both `constant_types`
                // (pool-internal type) and `numbering_type_overrides`
                // (bridge-injected resume-data type) mark a slot as
                // non-Int.
                let tp = self
                    .constant_types
                    .get(&idx)
                    .or_else(|| self.numbering_type_overrides.get(&idx))
                    .copied();
                match tp {
                    Some(Type::Ref) | Some(Type::Float) => continue,
                    _ => return OpRef(idx),
                }
            }
        }
        let opref = OpRef::from_const(self.next_const_idx);
        self.next_const_idx += 1;
        self.constants.insert(opref.0, value);
        // Box.type immutability (history.py:361 ConstInt): a ConstInt Box
        // always reports `op.type == 'i'`. Record the type explicitly so
        // `constant_type(opref)` returns `Some(Int)` instead of `None`;
        // otherwise callers that fall back on a slot-layout declared type
        // (pyre-jit-trace/trace_opcode.rs `opref_to_snapshot_tagged_for_slot`)
        // retag this ConstInt as Ref when it lands in a Ref-declared array
        // slot, and the bridge optimizer then panics in `getintbound` once
        // the bridge reseeds `const_pool` with `Value::Ref(GcRef(small_int))`.
        self.constant_types.insert(opref.0, Type::Int);
        opref
    }

    /// Get or create a typed constant OpRef.
    /// gcreftracer.py parity: Ref constants are rooted on the shadow
    /// stack so GC can update them when objects move.
    pub fn get_or_insert_typed(&mut self, value: i64, tp: Type) -> OpRef {
        // Refresh Ref constants from shadow stack before dedup —
        // GC may have moved objects, changing their addresses.
        if tp == Type::Ref {
            self.refresh_from_gc();
        }
        // Type-aware dedup: only match entries with matching type.
        // Consult both `constant_types` (intrinsic pool type) and
        // `numbering_type_overrides` (bridge-injected resume-data type)
        // to prevent Ref-tagged slots from aliasing with Int constants —
        // same rule as `get_or_insert`.
        for (&idx, &v) in &self.constants {
            if v == value {
                let pool_tp = self.constant_types.get(&idx).copied();
                let override_tp = self.numbering_type_overrides.get(&idx).copied();
                let existing_tp = pool_tp.or(override_tp);
                match existing_tp {
                    Some(existing) if existing == tp => return OpRef(idx),
                    None if tp == Type::Int => return OpRef(idx),
                    _ => continue,
                }
            }
        }
        let opref = OpRef::from_const(self.next_const_idx);
        self.next_const_idx += 1;
        self.constants.insert(opref.0, value);
        self.constant_types.insert(opref.0, tp);
        // Root non-null Ref constants on shadow stack.
        if tp == Type::Ref && value != 0 {
            let ss_idx = shadow_stack::push(GcRef(value as usize));
            self.rooted_refs.push((opref.0, ss_idx));
        }
        opref
    }

    /// Root an Int-typed constant on the shadow stack for GC safety.
    /// The constant stays Int-typed (optimizer sees Value::Int), but
    /// the referenced object won't be freed by GC.
    pub fn root_int_as_ref(&mut self, opref: OpRef) {
        if let Some(&value) = self.constants.get(&opref.0) {
            if value != 0 {
                let ss_idx = shadow_stack::push(GcRef(value as usize));
                self.rooted_refs.push((opref.0, ss_idx));
            }
        }
    }

    /// Get the type of a constant, if recorded.
    pub fn constant_type(&self, opref: OpRef) -> Option<Type> {
        self.constant_types.get(&opref.0).copied()
    }

    /// pyjitpl.py:3572 executor.constant_from_op(a) parity:
    /// return the typed Value for a constant OpRef, or None if
    /// the OpRef is not a known constant.
    pub fn get_value(&self, opref: OpRef) -> Option<majit_ir::Value> {
        let &raw = self.constants.get(&opref.0)?;
        // history.py:220/261/307: ConstInt/Float/Ptr Box.type is pinned at
        // construction. `get_or_insert*` always inserts into both
        // `constants` and `constant_types`, so a constant present in
        // `constants` MUST also be present in `constant_types`.
        let tp = self
            .constant_types
            .get(&opref.0)
            .copied()
            .unwrap_or_else(|| {
                panic!(
                    "constant_types missing entry for constant OpRef({}) (raw={}) \
                 though `constants` has it: get_or_insert / get_or_insert_typed \
                 must populate both maps in lockstep",
                    opref.0, raw
                )
            });
        Some(match tp {
            Type::Int => majit_ir::Value::Int(raw),
            Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(raw as usize)),
            Type::Float => majit_ir::Value::Float(f64::from_bits(raw as u64)),
            Type::Void => majit_ir::Value::Void,
        })
    }

    /// Mark an existing constant with a specific type for resume data only.
    /// RPython parity: GcRef pointers stored via const_int() need Ref type
    /// for correct resume data encoding, but Cranelift must treat them as Int
    /// (no GC root tracking for static type descriptor pointers).
    pub fn mark_type(&mut self, opref: OpRef, tp: Type) {
        // Store in numbering_type_overrides only — NOT constant_types.
        // This type info is for resume data (consumer switchover) only,
        // NOT for Cranelift backend (which would trigger GC root tracking).
        self.numbering_type_overrides.insert(opref.0, tp);
    }

    /// Update HashMap from shadow stack — GC may have moved Ref objects.
    /// gcreftracer.py:gcrefs_trace parity.
    fn refresh_from_gc(&mut self) {
        for &(opref_key, ss_idx) in &self.rooted_refs {
            let current = shadow_stack::get(ss_idx);
            self.constants.insert(opref_key, current.0 as i64);
        }
    }

    /// Release shadow stack roots.
    /// gcreftracer.py parity: release GC roots for this pool's constants.
    /// XXX majit-only: in RPython, ConstantPool consumption is strictly
    /// LIFO so pop_to always succeeds. In majit, ExportedState may pop
    /// the shadow stack between this pool's creation and release. Until
    /// the LIFO ordering is enforced structurally, guard against this.
    fn release_roots(&mut self) {
        if !self.rooted_refs.is_empty() {
            let current = shadow_stack::depth();
            if current >= self.shadow_stack_base {
                shadow_stack::pop_to(self.shadow_stack_base);
            }
            self.rooted_refs.clear();
        }
    }

    /// Consume the pool and return the constants map.
    pub fn into_inner(mut self) -> HashMap<u32, i64> {
        self.refresh_from_gc();
        let constants = std::mem::take(&mut self.constants);
        self.release_roots();
        constants
    }

    /// Consume the pool, returning both value and type maps.
    pub fn into_inner_with_types(mut self) -> (HashMap<u32, i64>, HashMap<u32, Type>) {
        self.refresh_from_gc();
        let constants = std::mem::take(&mut self.constants);
        let types = std::mem::take(&mut self.constant_types);
        self.release_roots();
        (constants, types)
    }

    /// Get numbering type overrides (for resume data only, not Cranelift).
    pub fn numbering_type_overrides(&self) -> &HashMap<u32, Type> {
        &self.numbering_type_overrides
    }

    /// Get a mutable reference to the inner constants map.
    pub fn as_mut(&mut self) -> &mut HashMap<u32, i64> {
        &mut self.constants
    }

    /// Ensure `next_const_idx` is beyond the given const-namespace key.
    /// Used by bridge injection: constants with pre-assigned indices must
    /// not be overwritten by subsequent `get_or_insert` allocations.
    pub fn reserve_index_past(&mut self, opref_key: u32) {
        let raw_idx = opref_key & !(1 << 31);
        self.next_const_idx = self.next_const_idx.max(raw_idx + 1);
    }

    /// Get a shared reference to the inner constants map.
    pub fn as_ref(&self) -> &HashMap<u32, i64> {
        &self.constants
    }

    /// Clone the constants map without consuming the pool.
    /// Refreshes from GC first to pick up moved Ref pointers.
    pub fn snapshot(&mut self) -> HashMap<u32, i64> {
        self.refresh_from_gc();
        self.constants.clone()
    }

    /// Clone the constant type map without consuming the pool.
    pub fn constant_types_snapshot(&self) -> HashMap<u32, Type> {
        self.constant_types.clone()
    }
}

impl Default for ConstantPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ConstantPool {
    fn drop(&mut self) {
        self.release_roots();
    }
}
