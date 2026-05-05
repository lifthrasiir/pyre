/// Guard optimization pass.
///
/// Removes redundant guards, strengthens weak guards when a stronger guard
/// on the same value appears later, and fuses consecutive guards that can
/// share fail descriptors.
///
/// ## Redundant Guard Removal
///
/// If the same foldable guard condition (opcode + arguments) has already been
/// verified, the duplicate guard is removed:
///
/// ```text
/// guard_true(v1)          # first check
/// ...
/// guard_true(v1)          # redundant – removed
/// ```
///
/// ## Guard Strengthening
///
/// A weaker guard is subsumed by a stronger guard on the same value:
///
/// ```text
/// guard_true(v1)          # v1 is truthy  (weak)
/// guard_value(v1, 42)     # v1 == 42      (strong, implies truthy)
/// ```
///
/// When the stronger guard is encountered the weak guard is recognized as
/// already satisfied, so it would have been removed by redundant-guard
/// removal anyway.  Strengthening here means we record that `v1` is
/// *nonnull/truthy* after seeing `guard_nonnull(v1)` or `guard_true(v1)`,
/// so a later `guard_nonnull(v1)` is eliminated.
///
/// ## Consecutive Guard Fusion
///
/// Consecutive guards on different values are kept but, when both lack a
/// descriptor, the second inherits the first's descriptor so the backend
/// can share resume data.
use std::collections::{HashMap, HashSet};

use majit_ir::{DescrRef, Op, OpCode, OpRef};

use crate::optimizeopt::dependency::IndexVar;
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

/// Key that uniquely identifies a guard condition.
///
/// Two foldable guards are redundant when they have the same opcode and the same
/// arguments (after forwarding resolution).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GuardKey {
    opcode: OpCode,
    args: Vec<OpRef>,
}

impl GuardKey {
    fn from_op(op: &Op) -> Self {
        GuardKey {
            opcode: op.opcode,
            args: op.args.to_vec(),
        }
    }
}

/// guard.py:16-163: Guard — wraps a guard op with its comparison op for
/// implication analysis (vector optimizer).
///
/// Stores the full `Op` for both the guard and its comparison, matching
/// RPython's `_attrs_ = ('index', 'op', 'cmp_op', 'rhs', 'lhs')`.
/// This preserves descr/fail_args for inhert_attributes and emit_operations.
#[derive(Clone, Debug)]
pub struct Guard {
    /// guard.py:20 — position in the operations list.
    pub index: usize,
    /// guard.py:20 — the guard op (GuardTrue/GuardFalse), with descr + fail_args.
    pub op: Op,
    /// guard.py:20 — the comparison op (IntLt/IntLe/IntGt/IntGe).
    pub cmp_op: Op,
    /// guard.py:27-29 — left-hand side IndexVar.
    pub lhs: IndexVar,
    /// guard.py:31-33 — right-hand side IndexVar.
    pub rhs: IndexVar,
}

impl Guard {
    /// guard.py:22-34: Guard.__init__(index, op, cmp_op, index_vars)
    pub fn new(
        index: usize,
        guard_op: Op,
        cmp_op: Op,
        index_vars: &HashMap<OpRef, IndexVar>,
    ) -> Self {
        let lhs_arg = cmp_op.arg(0);
        let lhs = index_vars
            .get(&lhs_arg)
            .cloned()
            .unwrap_or_else(|| IndexVar::new(lhs_arg));
        let rhs_arg = if cmp_op.num_args() > 1 {
            cmp_op.arg(1)
        } else {
            OpRef::NONE
        };
        let rhs = index_vars
            .get(&rhs_arg)
            .cloned()
            .unwrap_or_else(|| IndexVar::new(rhs_arg));
        Guard {
            index,
            op: guard_op,
            cmp_op,
            lhs,
            rhs,
        }
    }

    /// guard.py:158-163: Guard.of(boolarg, operations, index, index_vars)
    pub fn of(
        index: usize,
        guard_op: &Op,
        cmp_op: &Op,
        index_vars: &HashMap<OpRef, IndexVar>,
    ) -> Option<Self> {
        if !guard_op.opcode.is_guard() {
            return None;
        }
        match cmp_op.opcode {
            OpCode::IntLt | OpCode::IntLe | OpCode::IntGt | OpCode::IntGe => {}
            _ => return None,
        }
        Some(Self::new(
            index,
            guard_op.clone(),
            cmp_op.clone(),
            index_vars,
        ))
    }

    /// guard.py:36-43: setindex / setoperation / setcmp
    pub fn setindex(&mut self, index: usize) {
        self.index = index;
    }

    pub fn setoperation(&mut self, op: Op) {
        self.op = op;
    }

    pub fn setcmp(&mut self, cmp: Op) {
        self.cmp_op = cmp;
    }

    /// guard.py:106-111: get_compare_opnum
    pub fn get_compare_opnum(&self) -> OpCode {
        if self.op.opcode == OpCode::GuardTrue {
            self.cmp_op.opcode
        } else {
            // guard_false inversion (cmp_op.boolinverse)
            match self.cmp_op.opcode {
                OpCode::IntLt => OpCode::IntGe,
                OpCode::IntLe => OpCode::IntGt,
                OpCode::IntGt => OpCode::IntLe,
                OpCode::IntGe => OpCode::IntLt,
                OpCode::IntEq => OpCode::IntNe,
                OpCode::IntNe => OpCode::IntEq,
                other => other,
            }
        }
    }

    /// guard.py:51-71: implies(other)
    pub fn implies(&self, other: &Guard, _opt: Option<&OptContext>) -> bool {
        if self.op.opcode != other.op.opcode {
            return false;
        }
        if self.getleftkey() != other.getleftkey() {
            return false;
        }
        let (lhs_valid, lc) = self.lhs.compare(&other.lhs);
        if !lhs_valid {
            return false;
        }
        let (rhs_valid, rc) = self.rhs.compare(&other.rhs);
        if !rhs_valid {
            return false;
        }
        let opnum = self.get_compare_opnum();
        match opnum {
            OpCode::IntLe | OpCode::IntLt => lc >= 0 && rc <= 0,
            OpCode::IntGe | OpCode::IntGt => lc <= 0 && rc >= 0,
            _ => false,
        }
    }

    /// guard.py:45-46: getleftkey / getrightkey
    pub fn getleftkey(&self) -> OpRef {
        self.lhs.getvariable()
    }

    pub fn getrightkey(&self) -> OpRef {
        self.rhs.getvariable()
    }

    /// guard.py:126-132: emit_varops(opt, var, old_arg)
    fn emit_varops(
        var: &IndexVar,
        old_arg: OpRef,
        new_ops: &mut Vec<Op>,
        renamer: &mut HashMap<OpRef, OpRef>,
        next_const_pos: &mut u32,
        const_values: &mut HashMap<OpRef, i64>,
    ) -> OpRef {
        if var.is_identity() {
            return var.var;
        }
        // RPython: ConstInt(value) creates inline constant boxes.
        // In majit we allocate constant OpRefs and record the value.
        let ncp = next_const_pos;
        let cv = const_values;
        let ops = var.get_operations(|value| {
            let cref = OpRef::const_int(*ncp);
            *ncp += 1;
            cv.insert(cref, value);
            cref
        });
        let mut last = var.var;
        for op in ops {
            last = op.pos;
            new_ops.push(op);
        }
        // guard.py:131: opt.renamer.start_renaming(old_arg, box)
        if !last.is_constant() {
            renamer.insert(old_arg, last);
        }
        last
    }

    /// guard.py:73-97: transitive_imply(other, opt, loop)
    ///
    /// Emit a transitive guard that eliminates a loop guard.
    /// `label_args` = `loop.label.getarglist_copy()`.
    /// Emits compare + guard into `new_ops`. Returns the guard op.
    pub fn transitive_imply(
        &self,
        other: &Guard,
        label_args: &[OpRef],
        new_ops: &mut Vec<Op>,
        renamer: &mut HashMap<OpRef, OpRef>,
        next_const_pos: &mut u32,
        const_values: &mut HashMap<OpRef, i64>,
    ) -> Option<Op> {
        if self.op.opcode != other.op.opcode {
            return None;
        }
        if self.getleftkey() != other.getleftkey() {
            return None;
        }
        if !self.rhs.is_identity() {
            return None;
        }
        // guard.py:83
        let opnum = Self::transitive_cmpop(self.cmp_op.opcode);
        // guard.py:84-85: emit_varops
        let box_rhs = Self::emit_varops(
            &self.rhs,
            self.cmp_op.arg(1),
            new_ops,
            renamer,
            next_const_pos,
            const_values,
        );
        let other_rhs = Self::emit_varops(
            &other.rhs,
            other.cmp_op.arg(1),
            new_ops,
            renamer,
            next_const_pos,
            const_values,
        );
        // guard.py:86-87: compare = ResOperation(opnum, [box_rhs, other_rhs])
        let compare = Op::new(opnum, &[box_rhs, other_rhs]);
        new_ops.push(compare.clone());
        // guard.py:89-91:
        //   descr = CompileLoopVersionDescr()
        //   descr.copy_all_attributes_from(self.op.getdescr())
        //   descr.rd_vector_info = None
        //
        // Always create a NEW CompileLoopVersionDescr (loop_version()=true),
        // then copy resume attributes from the source guard's descr.
        // compile.py:861-872 copy_all_attributes_from copies:
        //   rd_consts, rd_pendingfields, rd_virtuals, rd_numb
        // In pyre these live on the FailDescr (compile.py:855 `_attrs_`);
        // make_compile_loop_version_descr_from reference-shares each
        // Arc<[T]> slot from the donor onto the fresh descr.
        let fresh_descr = crate::compile::make_compile_loop_version_descr_from(&self.op);
        let mut guard_op = Op::new(self.op.opcode, &[compare.pos]);
        guard_op.descr = Some(fresh_descr);
        // guard.py:94: guard.setfailargs(loop.label.getarglist_copy())
        guard_op.fail_args = Some(label_args.into());
        // copy_all_attributes_from parity: compile.py:861-872 copies
        // rd_consts / rd_pendingfields / rd_virtuals / rd_numb.  In
        // pyre these live on the FailDescr (compile.py:855 `_attrs_`)
        // and `make_compile_loop_version_descr_from` (called above)
        // reference-shares the donor's payload onto the fresh descr,
        // so `guard_op.descr.fail_descr().rd_*()` returns the donor's
        // values automatically.
        guard_op.fail_arg_types = self.op.fail_arg_types.clone();
        guard_op.rd_resume_position = self.op.rd_resume_position;
        // guard.py:95: opt.emit_operation(guard)
        new_ops.push(guard_op.clone());
        Some(guard_op)
    }

    /// guard.py:99-104: transitive_cmpop(opnum)
    pub fn transitive_cmpop(opnum: OpCode) -> OpCode {
        match opnum {
            OpCode::IntLt => OpCode::IntLe,
            OpCode::IntGt => OpCode::IntGe,
            other => other,
        }
    }

    /// guard.py:113-124: inhert_attributes(other)
    ///
    /// Copy index and resume attributes from `other` onto self's existing
    /// descr, preserving `myop.descr` identity (fail_index / status /
    /// subtype tag).  Mirrors RPython's:
    ///   descr = myop.getdescr()
    ///   descr.copy_all_attributes_from(other.op.getdescr())
    ///   myop.setfailargs(otherop.getfailargs()[:])
    /// where `descr` is the strengthened guard's *own* ResumeGuardDescr.
    pub fn inhert_attributes(&mut self, other: &Guard) {
        // guard.py:118
        self.index = other.index;
        // guard.py:120-121: descr.copy_all_attributes_from(other.op.getdescr())
        // compile.py:861-872 ResumeGuardDescr.copy_all_attributes_from:
        //     other = other.get_resumestorage()
        //     assert isinstance(other, ResumeGuardDescr)
        //     self.rd_consts = other.rd_consts
        //     self.rd_pendingfields = other.rd_pendingfields
        //     self.rd_virtuals = other.rd_virtuals
        //     self.rd_numb = other.rd_numb
        // i.e. the strengthened guard keeps its own descr identity; only
        // the resume payload is copied from the donor (chasing through
        // ResumeGuardCopiedDescr.prev via get_resumestorage()).
        let my_descr = self
            .op
            .descr
            .as_ref()
            .expect("guard.py:120 myop.getdescr() must exist");
        let donor_descr = other
            .op
            .descr
            .as_ref()
            .expect("guard.py:121 otherop.getdescr() must exist");
        // compile.py:861-872 / 840-842: in-place copy preserving
        // myop.descr identity (`fail_index` / status / subtype tag).
        crate::compile::copy_all_attributes_from(my_descr, donor_descr);
        self.op.rd_resume_position = other.op.rd_resume_position;
        // guard.py:123: myop.setfailargs(otherop.getfailargs()[:])
        self.op.fail_args = other.op.fail_args.clone();
        self.op.fail_arg_types = other.op.fail_arg_types.clone();
    }

    /// guard.py:134-147: emit_operations(opt)
    ///
    /// Re-emit the guard: materialize lhs/rhs via emit_varops,
    /// create fresh cmp + guard, emit them.
    pub fn emit_operations(
        &mut self,
        new_ops: &mut Vec<Op>,
        renamer: &mut HashMap<OpRef, OpRef>,
        next_const_pos: &mut u32,
        const_values: &mut HashMap<OpRef, i64>,
    ) {
        // guard.py:136-137: lhs/rhs via emit_varops
        let lhs = Self::emit_varops(
            &self.lhs,
            self.cmp_op.arg(0),
            new_ops,
            renamer,
            next_const_pos,
            const_values,
        );
        let rhs = Self::emit_varops(
            &self.rhs,
            self.cmp_op.arg(1),
            new_ops,
            renamer,
            next_const_pos,
            const_values,
        );
        // guard.py:138-140: cmp_op = ResOperation(opnum, [lhs, rhs])
        let cmp_op = Op::new(self.cmp_op.opcode, &[lhs, rhs]);
        new_ops.push(cmp_op.clone());
        // guard.py:142-144: guard = ResOperation(opnum, [cmp_op], descr)
        let mut guard = Op::new(self.op.opcode, &[cmp_op.pos]);
        guard.descr = self.op.descr.clone();
        guard.fail_args = self.op.fail_args.clone();
        guard.fail_arg_types = self.op.fail_arg_types.clone();
        guard.rd_resume_position = self.op.rd_resume_position;
        // compile.py:855 _attrs_ on descr; Arc-clone above shares them.
        new_ops.push(guard.clone());
        // guard.py:145-147
        self.setindex(new_ops.len() - 1);
        self.setoperation(guard);
        self.setcmp(cmp_op);
    }

    /// guard.py:149-156: set_to_none(info, loop)
    pub fn set_to_none(&self, ops: &mut [Option<Op>]) {
        ops[self.index] = None;
        if self.index > 0 {
            // guard.py:154: if operations[self.index-1] is self.cmp_op
            if let Some(ref prev) = ops[self.index - 1] {
                if prev.pos == self.cmp_op.pos {
                    ops[self.index - 1] = None;
                }
            }
        }
    }
}

/// guard.py:165-303: GuardStrengthenOpt (vector optimizer guard pass).
///
/// RPython name: `GuardStrengthenOpt`. Collects guard information from
/// a complete loop, determines implication/strengthening, re-emits with
/// proper descr/fail_args, and optionally eliminates array bound checks.
pub struct GuardStrengthenOpt {
    /// guard.py:168
    pub index_vars: HashMap<OpRef, IndexVar>,
    /// guard.py:169
    _newoperations: Vec<Op>,
    /// guard.py:170
    pub strength_reduced: usize,
    /// guard.py:171
    pub strongest_guards: HashMap<OpRef, Vec<Guard>>,
    /// guard.py:172
    guards: HashMap<usize, Option<Guard>>,
    /// renamer.py: Renamer — maps old OpRef → new OpRef for renamed vars.
    renamer: HashMap<OpRef, OpRef>,
    /// Zero-based counter for constant-namespace OpRef allocation.
    next_const_pos: u32,
    /// Materialized constant values: OpRef → i64.
    /// RPython uses ConstInt boxes inline; majit stores const values here.
    pub const_values: HashMap<OpRef, i64>,
}

impl GuardStrengthenOpt {
    /// guard.py:167
    pub fn new(index_vars: HashMap<OpRef, IndexVar>) -> Self {
        GuardStrengthenOpt {
            index_vars,
            _newoperations: Vec::new(),
            strength_reduced: 0,
            strongest_guards: HashMap::new(),
            guards: HashMap::new(),
            renamer: HashMap::new(),
            next_const_pos: 0,
            const_values: HashMap::new(),
        }
    }

    /// guard.py:175-187: collect_guard_information(loop)
    pub fn collect_guard_information(&mut self, ops: &[Op]) {
        for (i, op) in ops.iter().enumerate() {
            if !op.opcode.is_guard() {
                continue;
            }
            if op.opcode != OpCode::GuardTrue && op.opcode != OpCode::GuardFalse {
                continue;
            }
            // guard.py:183: Guard.of(op.getarg(0), operations, i, self.index_vars)
            let bool_arg = op.arg(0);
            let cmp_op = ops.iter().rfind(|o| o.pos == bool_arg);
            if let Some(cmp) = cmp_op {
                if let Some(guard) = Guard::of(i, op, cmp, &self.index_vars) {
                    let lk = guard.getleftkey();
                    let rk = guard.getrightkey();
                    self.record_guard(lk, guard.clone());
                    self.record_guard(rk, guard);
                }
            }
        }
    }

    /// guard.py:189-219: record_guard(key, guard)
    fn record_guard(&mut self, key: OpRef, guard: Guard) {
        if key.is_none() {
            return;
        }
        let others = self.strongest_guards.entry(key).or_default();
        if !others.is_empty() {
            let mut replaced = false;
            for i in 0..others.len() {
                if guard.implies(&others[i], None) {
                    // guard.py:204-210: strengthened
                    let old = others[i].clone();
                    self.guards.insert(guard.index, None); // mark new as 'do not emit'
                    let mut new_guard = guard.clone();
                    new_guard.inhert_attributes(&old);
                    self.guards.insert(old.index, Some(new_guard.clone()));
                    others[i] = new_guard;
                    replaced = true;
                } else if others[i].implies(&guard, None) {
                    // guard.py:211-215: implied
                    self.guards.insert(guard.index, None);
                    replaced = true;
                }
            }
            if !replaced {
                others.push(guard);
            }
        } else {
            others.push(guard);
        }
    }

    /// guard.py:221-249: eliminate_guards(loop)
    pub fn eliminate_guards(&mut self, ops: &[Op]) -> Vec<Op> {
        // guard.py:222: self.renamer = Renamer()
        self.renamer = HashMap::new();
        self._newoperations = Vec::with_capacity(ops.len());

        // Take guards out of self to satisfy borrow checker.
        let mut guards = std::mem::take(&mut self.guards);
        let index_vars = self.index_vars.clone();

        for (i, op) in ops.iter().enumerate() {
            if op.opcode.is_guard() {
                if let Some(replacement) = guards.get_mut(&i) {
                    self.strength_reduced += 1;
                    match replacement {
                        None => {
                            // guard.py:233: implied → skip
                            continue;
                        }
                        Some(guard) => {
                            // guard.py:234: guard.emit_operations(self)
                            guard.emit_operations(
                                &mut self._newoperations,
                                &mut self.renamer,
                                &mut self.next_const_pos,
                                &mut self.const_values,
                            );
                            continue;
                        }
                    }
                } else {
                    // guard.py:237: self.emit_operation(op)
                    let mut renamed = op.clone();
                    self.rename_op(&mut renamed);
                    self._newoperations.push(renamed);
                    continue;
                }
            }
            // guard.py:239-245: non-void index_var → emit_operations + rename
            if op.opcode.result_type() != majit_ir::Type::Void {
                if let Some(index_var) = index_vars.get(&op.pos) {
                    if !index_var.is_identity() {
                        let ncp = &mut self.next_const_pos;
                        let cv = &mut self.const_values;
                        let result = index_var.emit_operations(&mut self._newoperations, |value| {
                            let cref = OpRef::const_int(*ncp);
                            *ncp += 1;
                            cv.insert(cref, value);
                            cref
                        });
                        self.renamer.insert(op.pos, result);
                        continue;
                    }
                }
            }
            // guard.py:246: self.emit_operation(op)
            let mut renamed = op.clone();
            self.rename_op(&mut renamed);
            self._newoperations.push(renamed);
        }
        self.guards = guards;
        self._newoperations.clone()
    }

    /// guard.py:251-269: propagate_all_forward(info, loop, user_code)
    ///
    /// `version_info`: optional LoopVersionInfo for loop-version tracking.
    /// `label_args`: label arglist for transitive guard fail_args.
    /// `user_code`: if true, run eliminate_array_bound_checks.
    ///
    /// Returns `(ops, const_values)`. `const_values` maps constant OpRefs
    /// (allocated by IndexVar materialization) to their i64 values.
    /// The caller must register these in the trace's constant pool.
    /// guard.py:251-269: propagate_all_forward(info, loop, user_code)
    pub fn propagate_all_forward(
        &mut self,
        ops: &[Op],
        info: &mut super::version::LoopVersionInfo,
        label_args: &[OpRef],
        user_code: bool,
    ) -> (Vec<Op>, HashMap<OpRef, i64>) {
        self.collect_guard_information(ops);
        let mut result = self.eliminate_guards(ops);

        // guard.py:257-266: track loop-version guards.
        assert!(
            info.versions.len() == 1,
            "guard.py:257 assert len(info.versions) == 1"
        );
        let version = info.versions[0].clone();
        for op in &result {
            if !op.opcode.is_guard() {
                continue;
            }
            if let Some(ref descr) = op.descr {
                if let Some(fd) = descr.as_fail_descr() {
                    if fd.loop_version() {
                        info.track(fd.fail_index(), version.clone());
                    }
                }
            }
        }

        // guard.py:268-269
        if user_code {
            let prefix = self.eliminate_array_bound_checks(&mut result, label_args, info);
            if !prefix.is_empty() {
                let mut combined = prefix;
                combined.append(&mut result);
                result = combined;
            }
        }

        (result, std::mem::take(&mut self.const_values))
    }

    /// renamer.py:20-22: rename(op) — apply renamer map to op args.
    fn rename_op(&self, op: &mut Op) {
        for arg in op.args.iter_mut() {
            if let Some(&replacement) = self.renamer.get(arg) {
                *arg = replacement;
            }
        }
    }

    /// guard.py:272-274: emit_operation(op)
    pub fn emit_operation(&mut self, mut op: Op) {
        self.rename_op(&mut op);
        self._newoperations.push(op);
    }

    /// guard.py:276-277: operation_position()
    pub fn operation_position(&self) -> usize {
        self._newoperations.len()
    }
}

/// Helper: track a guard in version_info without borrow conflicts.
fn info_track_guard(
    info: &mut super::version::LoopVersionInfo,
    fail_index: u32,
    version: super::version::LoopVersion,
) {
    info.track(fail_index, version);
}

impl GuardStrengthenOpt {
    /// guard.py:279-303: eliminate_array_bound_checks(info, loop)
    ///
    /// `version_info`: LoopVersionInfo for tracking.
    /// `label_args` = `loop.label.getarglist_copy()`.
    /// Mutates `ops` in place (nullifies removed guards then compacts).
    /// Returns prefix ops to prepend to the loop.
    pub fn eliminate_array_bound_checks(
        &mut self,
        ops: &mut Vec<Op>,
        label_args: &[OpRef],
        version_info: &mut super::version::LoopVersionInfo,
    ) -> Vec<Op> {
        // guard.py:280
        version_info.mark();
        // guard.py:281
        let mut version: Option<super::version::LoopVersion> = None;
        // guard.py:282
        self._newoperations = Vec::new();

        // guard.py:283-299
        let mut opt_ops: Vec<Option<Op>> = ops.drain(..).map(Some).collect();
        let guards_snapshot: HashMap<OpRef, Vec<Guard>> = self.strongest_guards.clone();
        for guards in guards_snapshot.values() {
            if guards.len() <= 1 {
                continue;
            }
            let one = &guards[0];
            for other in &guards[1..] {
                // guard.py:291
                let transitive_guard = one.transitive_imply(
                    other,
                    label_args,
                    &mut self._newoperations,
                    &mut self.renamer,
                    &mut self.next_const_pos,
                    &mut self.const_values,
                );
                if let Some(tg) = transitive_guard {
                    // guard.py:293-294: version = info.snapshot(loop)
                    if version.is_none() {
                        let flat_ops: Vec<Op> = opt_ops.iter().filter_map(|o| o.clone()).collect();
                        version = Some(version_info.snapshot(&flat_ops, label_args));
                    }
                    // guard.py:295: info.remove(other.op.getdescr())
                    // RPython: unconditional call. version.py:38-42 asserts
                    // descr is in leads_to — if not, it's a programming error.
                    if let Some(fd) = other.op.descr.as_ref().and_then(|d| d.as_fail_descr()) {
                        version_info.remove(fd.fail_index());
                    }
                    // guard.py:296: other.set_to_none(info, loop)
                    other.set_to_none(&mut opt_ops);
                    // guard.py:297-299: info.track(transitive_guard, descr, version)
                    if let Some(fd) = tg.descr.as_ref().and_then(|d| d.as_fail_descr()) {
                        info_track_guard(
                            version_info,
                            fd.fail_index(),
                            version.as_ref().unwrap().clone(),
                        );
                    }
                }
            }
        }
        // guard.py:300
        version_info.clear();
        // guard.py:303: loop.operations = [op for op in loop.operations if op]
        *ops = opt_ops.into_iter().flatten().collect();

        // guard.py:302: loop.prefix = self._newoperations + loop.prefix
        std::mem::take(&mut self._newoperations)
    }
}

impl Default for GuardStrengthenOpt {
    fn default() -> Self {
        Self::new(HashMap::new())
    }
}

pub struct OptGuard {
    /// Set of guard conditions already verified in the trace.
    seen: HashSet<GuardKey>,

    /// Set of values known to be truthy (guarded by `guard_true` or
    /// `guard_nonnull`).
    truthy_values: HashSet<OpRef>,

    /// guard.py: values with known class (from GuardClass/GuardNonnullClass).
    known_classes: std::collections::HashMap<OpRef, majit_ir::GcRef>,

    /// guard.py: values known to be specific constants (from GuardValue).
    known_constants: std::collections::HashMap<OpRef, i64>,

    /// Descriptor of the last emitted guard, used for consecutive-guard
    /// fusion.
    last_guard_descr: Option<DescrRef>,
}

impl OptGuard {
    pub fn new() -> Self {
        OptGuard {
            seen: HashSet::new(),
            truthy_values: HashSet::new(),
            known_classes: std::collections::HashMap::new(),
            known_constants: std::collections::HashMap::new(),
            last_guard_descr: None,
        }
    }

    /// Record the "truthy" implications of a guard that has been verified.
    ///
    /// `guard_true(v)`       → v is truthy
    /// `guard_nonnull(v)`    → v is nonnull (truthy)
    /// `guard_value(v, c)`   → v == c, which also means v is truthy (if c != 0)
    /// `guard_class(v, cls)` → v has class cls, which also means v is nonnull
    /// `guard_nonnull_class` → v is nonnull (and has class)
    fn record_implications(&mut self, op: &Op, ctx: &OptContext) {
        match op.opcode {
            OpCode::GuardTrue => {
                self.truthy_values.insert(op.arg(0));
            }
            OpCode::GuardNonnull => {
                self.truthy_values.insert(op.arg(0));
            }
            OpCode::GuardFalse => {
                // guard.py: guard_false(v) means v is known to be 0/false.
                // This is the opposite of guard_true: we don't add to truthy_values.
                // But we record it so that a later guard_false(v) can be eliminated.
            }
            OpCode::GuardValue => {
                let val_arg = op.arg(0);
                if let Some(c) = ctx.get_constant_int(op.arg(1)) {
                    // guard.py: record known constant value.
                    self.known_constants.insert(val_arg, c);
                    if c != 0 {
                        self.truthy_values.insert(val_arg);
                    }
                }
            }
            OpCode::GuardClass | OpCode::GuardNonnullClass => {
                // Having a class implies the object is nonnull.
                self.truthy_values.insert(op.arg(0));
                // guard.py: record the known class for subsumption checks.
                if op.num_args() >= 2 {
                    if let Some(class_val) = ctx.get_constant_int(op.arg(1)) {
                        self.known_classes
                            .insert(op.arg(0), majit_ir::GcRef(class_val as usize));
                    }
                }
            }
            _ => {}
        }
    }

    /// Check whether this guard is subsumed by previously recorded
    /// implications (guard strengthening).
    ///
    /// Returns `true` if the guard can be removed.
    fn is_subsumed(&self, op: &Op, ctx: &OptContext) -> bool {
        match op.opcode {
            OpCode::GuardTrue => {
                // guard.py: also subsumed if value is a known nonzero constant.
                if self.truthy_values.contains(&op.arg(0)) {
                    return true;
                }
                if let Some(&c) = self.known_constants.get(&op.arg(0)) {
                    return c != 0;
                }
                false
            }
            OpCode::GuardFalse => {
                // Subsumed if value is a known zero constant.
                if let Some(&c) = self.known_constants.get(&op.arg(0)) {
                    return c == 0;
                }
                false
            }
            OpCode::GuardNonnull => {
                // RPython guard.py does NOT handle GUARD_NONNULL.
                // It is handled exclusively by rewrite.py:optimize_GUARD_NONNULL.
                false
            }
            // rewrite.py: optimize_GUARD_CLASS — if the class is already
            // known for this value, and it matches, remove the guard.
            OpCode::GuardClass if op.num_args() >= 2 => {
                if let Some(&known_class) = self.known_classes.get(&op.arg(0)) {
                    if let Some(expected) = ctx.get_constant_int(op.arg(1)) {
                        return known_class.0 as i64 == expected;
                    }
                }
                // RPython: setinfo_from_preamble sets PtrInfo.KnownClass or
                // Instance with known_class. Check ctx for imported info.
                if let Some(expected) = ctx.get_constant_int(op.arg(1)) {
                    if let Some(class_ptr) = ctx.get_known_class(op.arg(0)) {
                        return class_ptr.0 as i64 == expected;
                    }
                }
                false
            }
            // guard.py: GUARD_VALUE subsumed if value is already known to be
            // that exact constant from a previous GuardValue.
            OpCode::GuardValue if op.num_args() >= 2 => {
                if let Some(&known) = self.known_constants.get(&op.arg(0)) {
                    if let Some(expected) = ctx.get_constant_int(op.arg(1)) {
                        return known == expected;
                    }
                }
                false
            }
            // guard.py: GUARD_NONNULL_CLASS subsumed if value is known nonnull
            // AND a previous GUARD_CLASS with same args was already seen.
            OpCode::GuardNonnullClass if op.num_args() >= 2 => {
                // RPython: subsumed if nonnull is known AND class matches.
                let nonnull_known = self.truthy_values.contains(&op.arg(0))
                    || ctx.get_ptr_info(op.arg(0)).is_some_and(|i| i.is_nonnull());
                if !nonnull_known {
                    return false;
                }
                // Check class via guard pass state
                if let Some(&known_class) = self.known_classes.get(&op.arg(0)) {
                    if let Some(expected) = ctx.get_constant_int(op.arg(1)) {
                        return known_class.0 as i64 == expected;
                    }
                }
                // Check class via imported PtrInfo
                if let Some(expected) = ctx.get_constant_int(op.arg(1)) {
                    if let Some(class_ptr) = ctx.get_known_class(op.arg(0)) {
                        return class_ptr.0 as i64 == expected;
                    }
                }
                // Fallback: seen as a previous GuardClass with same args
                let class_key = GuardKey {
                    opcode: OpCode::GuardClass,
                    args: op.args.to_vec(),
                };
                self.seen.contains(&class_key)
            }
            _ => false,
        }
    }

    /// Try to apply consecutive-guard descriptor fusion.
    ///
    /// If the current guard has no descriptor and the previous guard had
    /// one, reuse it.
    fn try_fuse_descr(&self, op: &mut Op) {
        if op.descr.is_none() {
            if let Some(ref descr) = self.last_guard_descr {
                op.descr = Some(descr.clone());
            }
        }
    }

    fn can_remove_as_duplicate(opcode: OpCode) -> bool {
        opcode.is_foldable_guard()
    }
}

impl Default for OptGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptGuard {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if !op.opcode.is_guard() {
            // Non-guard operations invalidate the last-guard descriptor
            // tracking (no longer consecutive).
            self.last_guard_descr = None;
            return OptimizationResult::PassOn;
        }

        // GuardNotForced must reach the Heap pass, which holds the
        // postponed CallMayForce op.  Return PassOn so Heap can emit
        // the call immediately before the guard.
        if matches!(op.opcode, OpCode::GuardNotForced | OpCode::GuardNotForced2) {
            return OptimizationResult::PassOn;
        }

        // --- Redundant guard removal (exact match) ---
        if Self::can_remove_as_duplicate(op.opcode) {
            let key = GuardKey::from_op(op);
            if self.seen.contains(&key) {
                return OptimizationResult::Remove;
            }

            self.seen.insert(key);
        }

        // --- Guard strengthening (subsumption) ---
        if self.is_subsumed(op, ctx) {
            return OptimizationResult::Remove;
        }

        // The guard is not redundant – record it and emit.
        self.record_implications(op, ctx);

        // RPython: GuardValue makes the value a known constant in the
        // optimizer context, enabling export to Phase 2.
        if op.opcode == OpCode::GuardValue {
            if let Some(c) = ctx.get_constant_int(op.arg(1)) {
                ctx.make_constant(op.arg(0), majit_ir::Value::Int(c));
            }
        }

        // --- Consecutive guard fusion ---
        let mut fused_op = op.clone();
        self.try_fuse_descr(&mut fused_op);

        // Track this guard's descriptor for the next consecutive guard.
        self.last_guard_descr = fused_op.descr.clone();

        OptimizationResult::Emit(fused_op)
    }

    fn setup(&mut self) {
        self.seen.clear();
        self.truthy_values.clear();
        self.known_classes.clear();
        self.known_constants.clear();
        self.last_guard_descr = None;
    }

    fn name(&self) -> &'static str {
        "guard"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::optimizer::Optimizer;

    /// Helper: assign sequential positions to ops starting from a high base
    /// to avoid colliding with argument OpRefs.  Also attaches a fresh
    /// `ResumeGuardDescr` to every guard op that lacks one, mirroring
    /// RPython's invariant (`optimizer.py:691 assert isinstance(last_descr,
    /// compile.ResumeGuardDescr)`) that every guard reaching the optimizer
    /// has a head-of-chain ResumeGuardDescr — this lets `_copy_resume_data_from`
    /// share via `make_resume_guard_copied_descr(prev)` without panicking on
    /// a missing donor.
    fn assign_positions(ops: &mut [Op], base: u32) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef::from_raw(base + i as u32);
            if op.opcode.is_guard() && op.descr.is_none() {
                op.descr = Some(crate::compile::make_resume_guard_descr_typed(Vec::new()));
            }
        }
    }

    fn run_guard_pass(ops: &[Op]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptGuard::new()));
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(ops);
        opt.snapshot_boxes = snapshots;
        opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 1024)
    }

    // ── Redundant Guard Removal ─────────────────────────────────────────

    #[test]
    fn test_duplicate_guard_true_removed() {
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]), // duplicate
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardTrue)
            .count();
        assert_eq!(guard_count, 1, "duplicate guard_true should be removed");
        assert_eq!(result.len(), 2); // guard_true + int_add
    }

    #[test]
    fn test_duplicate_guard_false_removed() {
        let mut ops = vec![
            Op::new(OpCode::GuardFalse, &[OpRef::from_raw(0)]),
            Op::new(OpCode::GuardFalse, &[OpRef::from_raw(0)]), // duplicate
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardFalse)
            .count();
        assert_eq!(guard_count, 1);
    }

    #[test]
    fn test_duplicate_guard_nonnull_removed() {
        let mut ops = vec![
            Op::new(OpCode::GuardNonnull, &[OpRef::from_raw(5)]),
            Op::new(OpCode::GuardNonnull, &[OpRef::from_raw(5)]), // duplicate
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GuardNonnull);
    }

    #[test]
    fn test_duplicate_guard_value_removed() {
        let mut ops = vec![
            Op::new(
                OpCode::GuardValue,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
            ),
            Op::new(
                OpCode::GuardValue,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
            ), // duplicate
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GuardValue);
    }

    #[test]
    fn test_duplicate_guard_class_removed() {
        let mut ops = vec![
            Op::new(
                OpCode::GuardClass,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
            ),
            Op::new(
                OpCode::GuardClass,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
            ), // duplicate
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 1);
    }

    // ── Non-duplicate guards are preserved ──────────────────────────────

    #[test]
    fn test_different_guard_args_kept() {
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(1)]), // different arg
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(
            result.len(),
            2,
            "guards on different values should both be kept"
        );
    }

    #[test]
    fn test_different_guard_opcodes_kept() {
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
            Op::new(OpCode::GuardFalse, &[OpRef::from_raw(0)]), // different opcode
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(
            result.len(),
            2,
            "guard_true and guard_false on same arg are not duplicates"
        );
    }

    // ── Guard ordering preservation ─────────────────────────────────────

    #[test]
    fn test_guard_order_preserved() {
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
            Op::new(OpCode::GuardFalse, &[OpRef::from_raw(1)]),
            Op::new(OpCode::GuardNonnull, &[OpRef::from_raw(2)]),
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GuardTrue);
        assert_eq!(result[1].opcode, OpCode::GuardFalse);
        assert_eq!(result[2].opcode, OpCode::GuardNonnull);
    }

    // ── Guard Strengthening ─────────────────────────────────────────────

    #[test]
    fn test_guard_nonnull_after_guard_true_removed() {
        // guard_true(v) implies v is truthy/nonnull,
        // so a later guard_nonnull(v) is redundant.
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
            Op::new(OpCode::GuardNonnull, &[OpRef::from_raw(0)]), // subsumed
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GuardTrue);
    }

    #[test]
    fn test_guard_true_after_guard_nonnull_removed() {
        // guard_nonnull(v) implies v is truthy,
        // so a later guard_true(v) is subsumed.
        let mut ops = vec![
            Op::new(OpCode::GuardNonnull, &[OpRef::from_raw(0)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]), // subsumed
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GuardNonnull);
    }

    #[test]
    fn test_guard_nonnull_after_guard_class_removed() {
        // guard_class(v, cls) implies v is nonnull,
        // so a later guard_nonnull(v) is subsumed.
        let mut ops = vec![
            Op::new(
                OpCode::GuardClass,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
            ),
            Op::new(OpCode::GuardNonnull, &[OpRef::from_raw(0)]), // subsumed
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GuardClass);
    }

    #[test]
    fn test_guard_true_after_guard_nonnull_class_removed() {
        // guard_nonnull_class(v, cls) implies v is nonnull/truthy.
        let mut ops = vec![
            Op::new(
                OpCode::GuardNonnullClass,
                &[OpRef::from_raw(0), OpRef::from_raw(1)],
            ),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]), // subsumed
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GuardNonnullClass);
    }

    #[test]
    fn test_guard_strengthening_different_values_not_subsumed() {
        // guard_true(v0) does NOT subsume guard_nonnull(v1).
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
            Op::new(OpCode::GuardNonnull, &[OpRef::from_raw(1)]), // different value
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 2);
    }

    // ── Consecutive Guard Fusion ────────────────────────────────────────

    #[test]
    fn test_consecutive_guards_without_descr_no_crash() {
        // Two consecutive guards without descriptors should not crash.
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(1)]),
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 2);
    }

    // ── Mixed guards and non-guards ─────────────────────────────────────

    #[test]
    fn test_non_guard_ops_pass_through() {
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]),
            Op::new(OpCode::IntSub, &[OpRef::from_raw(0), OpRef::from_raw(1)]),
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::IntSub);
    }

    #[test]
    fn test_guards_interleaved_with_ops() {
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]), // duplicate, removed
            Op::new(OpCode::IntSub, &[OpRef::from_raw(0), OpRef::from_raw(1)]),
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GuardTrue);
        assert_eq!(result[1].opcode, OpCode::IntAdd);
        assert_eq!(result[2].opcode, OpCode::IntSub);
    }

    #[test]
    fn test_three_duplicate_guards() {
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]), // duplicate
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]), // duplicate
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 1);
    }

    // ── Integration: OptGuard as standalone pass ─────────────────────
    // RPython: guard.py is NOT in the default pipeline (only used by vectorization).
    // These tests exercise OptGuard independently.

    #[test]
    fn test_guard_strengthen_pass_removes_duplicate() {
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]), // should be removed
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 100);

        let result = run_guard_pass(&ops);

        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardTrue)
            .count();
        assert_eq!(
            guard_count, 1,
            "duplicate guard should be removed by OptGuard"
        );
    }

    #[test]
    fn test_guard_no_exception_duplicates_preserved() {
        // guard_no_exception depends on ambient exception state, so distinct
        // checks cannot be merged just because they have no explicit args.
        let mut ops = vec![
            Op::new(OpCode::GuardNoException, &[]),
            Op::new(OpCode::GuardNoException, &[]),
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_guard_no_overflow_duplicates_preserved() {
        let mut ops = vec![
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::GuardNoOverflow, &[]),
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_overflow_guards_preserved_in_full_pipeline() {
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(1)]),
            Op::new(OpCode::IntSubOvf, &[OpRef::from_raw(0), OpRef::from_raw(2)]),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::IntMulOvf, &[OpRef::from_raw(2), OpRef::from_raw(1)]),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(
                OpCode::Jump,
                &[
                    OpRef::from_raw(101),
                    OpRef::from_raw(101),
                    OpRef::from_raw(103),
                ],
            ),
        ];
        assign_positions(&mut ops, 100);

        let mut opt = Optimizer::default_pipeline();
        // Overflow guards and IntMulOvf work on Int-typed args — override
        // the test default so renamed_inputarg_types sees Int, not Ref.
        opt.trace_inputarg_types = vec![majit_ir::Type::Int; 1024];
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );
        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNoOverflow)
            .count();

        // With intbounds postprocess_GUARD_TRUE, OpRef::from_raw(1) becomes known
        // constant 1 after GuardTrue(OpRef::from_raw(1)). IntMulOvf(x, 1) cannot
        // overflow → second GuardNoOverflow is removed. This matches
        // RPython intbounds.py:52-58 _postprocess_guard_true_false_value.
        assert_eq!(
            guard_count, 1,
            "first overflow guard survives; second removed (mul by constant 1 cannot overflow)"
        );
    }

    #[test]
    fn test_guard_nonnull_class_subsumed_by_nonnull_plus_class() {
        // GUARD_NONNULL(v) then GUARD_CLASS(v, cls) already seen
        // → later GUARD_NONNULL_CLASS(v, cls) is subsumed.
        let mut ops = vec![
            Op::new(OpCode::GuardNonnull, &[OpRef::from_raw(100)]),
            Op::new(
                OpCode::GuardClass,
                &[OpRef::from_raw(100), OpRef::from_raw(200)],
            ),
            Op::new(
                OpCode::GuardNonnullClass,
                &[OpRef::from_raw(100), OpRef::from_raw(200)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
        ];
        assign_positions(&mut ops, 0);
        let result = run_guard_pass(&ops);
        // GuardNonnullClass should be removed (subsumed)
        let nnc_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnullClass)
            .count();
        assert_eq!(nnc_count, 0, "GuardNonnullClass should be subsumed");
    }

    #[test]
    fn test_guard_value_to_guard_true() {
        // GUARD_VALUE(v, 1) → GUARD_TRUE(v) via rewrite pass.
        let mut ops = vec![
            {
                let mut op = Op::new(
                    OpCode::GuardValue,
                    &[OpRef::from_raw(100), OpRef::from_raw(200)],
                );
                op.pos = OpRef::from_raw(0);
                op
            },
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
        ];
        ops[1].pos = OpRef::from_raw(1);

        // Pre-seed constant 1 for OpRef::from_raw(200)
        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.add_pass(Box::new(crate::optimizeopt::rewrite::OptRewrite::new()));
        let mut constants = std::collections::HashMap::new();
        constants.insert(200, 1i64);
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1024);

        // GUARD_VALUE should be replaced with GUARD_TRUE
        assert!(
            result.iter().any(|o| o.opcode == OpCode::GuardTrue),
            "GUARD_VALUE(v, 1) should become GUARD_TRUE(v)"
        );
        assert!(
            !result.iter().any(|o| o.opcode == OpCode::GuardValue),
            "GUARD_VALUE should be gone"
        );
    }
}
