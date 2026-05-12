//! RPython `rpython/annotator/unaryop.py` — single-dispatch
//! (`dispatch == 1`) operation handlers on the `SomeValue` lattice.
//!
//! Commit split (following the plan in
//! `.claude/plans/0-warm-raccoon.md`):
//!
//! * **Commit 6a (this file initial drop)** — `UNARY_OPERATIONS`
//!   listing + `@op.type.register(SomeObject)` +
//!   `@op.contains.register(SomeNone|SomeInteger|SomeFloat|SomeBool)` +
//!   simple-returning `class __extend__(SomeObject)` defaults +
//!   primitive overrides on `SomeFloat` / `SomeInteger` / `SomeBool` +
//!   `class __extend__(SomeTuple)` unary methods.
//! * **Commit 6a2** — `@op.bool.register(SomeObject)` with knowntypedata
//!   branch refinement, `@op.isinstance.register` /
//!   `simple_call` / `call_args` / `issubtype` / `bool_behavior` /
//!   `getattr` / `setattr` / `hash` refinement with `find_method`.
//! * **Commit 6b** — list/dict `method_*` analyzers.
//! * **Commit 6c** — string/bytearray/char/unicode `method_*` analyzers.
//! * **Commit 6d** — Instance / Iterator / Builtin / PBC / None /
//!   Exception / WeakRef handlers.

use std::rc::Rc;

use super::super::flowspace::model::ConstValue;
use super::super::flowspace::model::Constant;
use super::super::flowspace::model::Hlvalue;
use super::super::flowspace::operation::{
    BuiltinException, CanOnlyThrow, HLOperation, OpKind, Specialization, Transformation,
    register_single,
};
use super::annrpython::RPythonAnnotator;
use super::model::{
    AnnotatorError, SomeBool, SomeBuiltinMethod, SomeFloat, SomeInteger, SomeIterator,
    SomeObjectTrait, SomeString, SomeTuple, SomeTypeOf, SomeUnicodeString, SomeValue, SomeValueTag,
    add_knowntypedata, s_bool, s_impossible_value, s_none,
};

/// RPython `UNARY_OPERATIONS` (unaryop.py:26-28).
///
/// ```python
/// UNARY_OPERATIONS = set([oper.opname for oper in op.__dict__.values()
///                         if oper.dispatch == 1])
/// UNARY_OPERATIONS.remove('contains')
/// ```
pub static UNARY_OPERATIONS: &[OpKind] = &[
    OpKind::Id,
    OpKind::Type,
    OpKind::IsSubtype,
    OpKind::IsInstance,
    OpKind::Repr,
    OpKind::Str,
    OpKind::Len,
    OpKind::Hash,
    OpKind::SetAttr,
    OpKind::DelAttr,
    OpKind::GetSlice,
    OpKind::SetSlice,
    OpKind::DelSlice,
    OpKind::Pos,
    OpKind::Neg,
    OpKind::NegOvf,
    OpKind::Bool,
    OpKind::Abs,
    OpKind::AbsOvf,
    OpKind::Hex,
    OpKind::Oct,
    OpKind::Bin,
    OpKind::Ord,
    OpKind::Invert,
    OpKind::Int,
    OpKind::Float,
    OpKind::Long,
    OpKind::Hint,
    OpKind::Iter,
    OpKind::Next,
    OpKind::GetAttr,
    OpKind::SimpleCall,
    OpKind::CallArgs,
    // `Contains` is removed from UNARY_OPERATIONS upstream line 28 —
    // it's a binary op on the `in` operand semantically. The `@op.contains.register`
    // hooks below still live in unaryop.py, so they land here.
];

/// Module-import-time population of `_REGISTRY_SINGLE` — mirrors the
/// side-effect of RPython importing unaryop.py (which runs all the
/// `@op.X.register(Some_cls)` decorators at import time).
pub fn init(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    init_type_register(reg);
    init_contains_register(reg);
    init_someobject_defaults(reg);
    init_somefloat_overrides(reg);
    init_someinteger_overrides(reg);
    init_somebool_overrides(reg);
    init_sometuple_overrides(reg);
    init_somelist_overrides(reg);
    init_somedict_overrides(reg);
    init_somestring_overrides(reg);
    init_someunicodestring_overrides(reg);
    init_somebytearray_overrides(reg);
    init_somechar_overrides(reg);
    init_someunicodecp_overrides(reg);
    init_someiterator_overrides(reg);
    init_somepbc_overrides(reg);
    init_someptr_overrides(reg);
    init_somenone_overrides(reg);
    init_someweakref_overrides(reg);
}

/// Module-import time side effect mirroring
/// `@op.<name>.register_transform(...)` decorators in unaryop.py. Runs
/// at `_TRANSFORM_SINGLE` thread_local init; one block per upstream
/// `@op.X.register_transform(Y)` decorator.
pub fn init_transform(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Transformation>,
    >,
) {
    init_instance_single_transform(reg);
    init_instance_attr_transform(reg);
    init_object_call_args_transform(reg);
}

fn register(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
    op: OpKind,
    tag: SomeValueTag,
    spec: Specialization,
) {
    reg.entry(op).or_default().insert(tag, spec);
}

fn register_transform(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Transformation>,
    >,
    op: OpKind,
    tag: SomeValueTag,
    tx: Transformation,
) {
    reg.entry(op).or_default().insert(tag, tx);
}

// =====================================================================
// unaryop.py:31-33 — @op.type.register(SomeObject)
// =====================================================================
//
//     @op.type.register(SomeObject)
//     def type_SomeObject(annotator, v_arg):
//         return SomeTypeOf([v_arg])

fn init_type_register(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    register(
        reg,
        OpKind::Type,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, hl| {
                // upstream unaryop.py:31-33 — `SomeTypeOf([v_arg])`.
                // Constants are ignored: upstream `type(const)` would
                // never enter this dispatcher (it lands on the
                // const-folding path of `Type.constfold`). When the
                // arg happens to be a constant we fall back to an
                // empty is_type_of list.
                let vars: Vec<Rc<super::super::flowspace::model::Variable>> = match &hl.args[0] {
                    super::super::flowspace::model::Hlvalue::Variable(v) => {
                        vec![Rc::new(v.clone())]
                    }
                    super::super::flowspace::model::Hlvalue::Constant(_) => Vec::new(),
                };
                SomeValue::TypeOf(SomeTypeOf::new(vars))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:91-111 — @op.contains.register(SomeObject|SomeNone|…)
// =====================================================================

fn init_contains_register(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:91-94 — SomeObject: returns s_Bool, can_only_throw=[].
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Bool(SomeBool::new())),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:96-104 — SomeNone: returns SomeBool(const=False).
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::None_,
        Specialization {
            apply: Box::new(|_ann, _hl| {
                let mut s = SomeBool::new();
                s.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
                SomeValue::Bool(s)
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:107-111 — SomeInteger/Float/Bool: raise AnnotatorError.
    let number_tags = &[
        SomeValueTag::Integer,
        SomeValueTag::Float,
        SomeValueTag::Bool,
    ];
    for tag in number_tags {
        register(
            reg,
            OpKind::Contains,
            *tag,
            Specialization {
                apply: Box::new(|_ann, _hl| panic!("AnnotatorError: number is not iterable")),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
}

// =====================================================================
// unaryop.py:156-254 — class __extend__(SomeObject) simple-return defaults
// =====================================================================
//
// `class __extend__(SomeObject)` upstream defines the defaults fetched by
// `SingleDispatchMixin.get_specialization` when `getattr(s_arg, opname)`
// hits the base class. Rust port registers one spec per opname on
// `SomeValueTag::Object`; the MRO fallback handles every subclass that
// doesn't override.

fn init_someobject_defaults(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:158-159 — len(self): return SomeInteger(nonneg=True)
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Integer(SomeInteger::new(true, false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:78-87 / 151-156 — bool(self): bool_behavior +
    // truth-branch knowntypedata via nonnoneify().
    register(
        reg,
        OpKind::Bool,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_obj = ann.annotation(&hl.args[0]).expect("bool: object unbound");
                let mut r = SomeBool::new();
                match &s_obj {
                    SomeValue::PBC(pbc) if !pbc.can_be_none => {
                        r.base.const_box = Some(Constant::new(ConstValue::Bool(true)));
                    }
                    SomeValue::None_(_) => {
                        r.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
                    }
                    _ => {
                        if s_obj.is_immutable_constant()
                            && let Some(c) = s_obj.const_()
                            && let Some(truthy) = c.truthy()
                        {
                            r.base.const_box = Some(Constant::new(ConstValue::Bool(truthy)));
                        } else {
                            let len_op = HLOperation::new(OpKind::Len, vec![hl.args[0].clone()]);
                            let s_len = len_op.consider(ann).expect("bool: len() dispatch failed");
                            if let Some(ConstValue::Int(n)) = s_len.const_() {
                                r.base.const_box = Some(Constant::new(ConstValue::Bool(*n > 0)));
                            }
                        }
                    }
                }
                let mut s_nonnone_obj = s_obj.clone();
                if s_nonnone_obj.can_be_none() {
                    s_nonnone_obj = s_nonnone_obj.nonnoneify();
                }
                let vars = match &hl.args[0] {
                    Hlvalue::Variable(v) => vec![Rc::new(v.clone())],
                    Hlvalue::Constant(_) => Vec::new(),
                };
                let mut knowntypedata = std::collections::HashMap::new();
                add_knowntypedata(&mut knowntypedata, true, &vars, s_nonnone_obj);
                r.set_knowntypedata(knowntypedata);
                SomeValue::Bool(r)
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:169-170 — hash: raise AnnotatorError("cannot use hash() in RPython").
    register(
        reg,
        OpKind::Hash,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| panic!("AnnotatorError: cannot use hash() in RPython")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:172-173 — str: SomeString().
    register(
        reg,
        OpKind::Str,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::String(SomeString::new(false, false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:178-179 — repr: SomeString().
    register(
        reg,
        OpKind::Repr,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::String(SomeString::new(false, false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:181-188 — hex/bin/oct: SomeString().
    for op in &[OpKind::Hex, OpKind::Bin, OpKind::Oct] {
        register(
            reg,
            *op,
            SomeValueTag::Object,
            Specialization {
                apply: Box::new(|_ann, _hl| SomeValue::String(SomeString::new(false, false))),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
    // unaryop.py:190-192 — id: raise AnnotatorError.
    register(
        reg,
        OpKind::Id,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| {
                panic!("AnnotatorError: cannot use id() in RPython; see objectmodel.compute_xxx()")
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:194-195 — int: SomeInteger().
    register(
        reg,
        OpKind::Int,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Integer(SomeInteger::default())),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:197-198 — float: SomeFloat().
    register(
        reg,
        OpKind::Float,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Float(SomeFloat::new())),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:215-229 — SomeObject.getattr(self, s_attr).
    register(
        reg,
        OpKind::GetAttr,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_self = ann.annotation(&hl.args[0]).expect("getattr: self unbound");
                let s_attr = ann.annotation(&hl.args[1]).expect("getattr: attr unbound");
                // unaryop.py:280-285 `_getattr_(s_obj, s_attr)` — `if
                // s_attr.is_constant() and isinstance(s_attr.const, str)`.
                // Upstream Python 2 `str` is bytes; pyre uses
                // [`ConstValue::as_pystr`] so a unicode literal does not
                // sneak through.
                let Some(attr) = s_attr
                    .const_()
                    .and_then(ConstValue::as_pystr)
                    .map(str::to_owned)
                else {
                    panic!(
                        "AnnotatorError: getattr({:?}, {:?}) has non-constant argument",
                        s_self, s_attr
                    );
                };
                if let Some(s_method) = s_self.find_method(&attr) {
                    return s_method;
                }
                // upstream unaryop.py:215-229 constant receiver path
                // shares the `getattr(obj, name)` call with
                // `flowspace::operation::GetAttr.constfold`; both
                // route through `const_runtime_getattr` and only
                // diverge at the wrap step (annotator wraps via
                // `immutablevalue`, flowspace via `const`).
                if s_self.is_immutable_constant()
                    && let Some(c) = s_self.const_()
                {
                    match crate::flowspace::model::const_runtime_getattr(c, &attr) {
                        Ok(Some(value)) => {
                            return ann.bookkeeper.immutablevalue(&value).unwrap_or_else(|err| {
                                panic!("getattr immutablevalue failed: {}", err)
                            });
                        }
                        Ok(None) => {
                            // upstream `WrapException` swallow — fall
                            // through to the "cannot find attr" panic.
                        }
                        Err(msg) => {
                            panic!("AnnotatorError in getattr constant path: {msg}")
                        }
                    }
                }
                panic!(
                    "AnnotatorError: Cannot find attribute {:?} on {:?}",
                    attr, s_self
                )
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:972-980 — SomePBC.getattr(self, s_attr).
    //
    //   def getattr(self, s_attr):
    //       assert s_attr.is_constant()
    //       if s_attr.const == '__name__':
    //           from rpython.annotator.classdesc import ClassDesc
    //           if self.getKind() is ClassDesc:
    //               return SomeString()
    //       bookkeeper = getbookkeeper()
    //       return bookkeeper.pbc_getattr(self, s_attr)
    register(
        reg,
        OpKind::GetAttr,
        SomeValueTag::PBC,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_self = ann
                    .annotation(&hl.args[0])
                    .expect("pbc getattr: self unbound");
                let s_attr = ann
                    .annotation(&hl.args[1])
                    .expect("pbc getattr: attr unbound");
                let SomeValue::PBC(pbc) = &s_self else {
                    unreachable!("PBC-tag dispatch must carry SomePBC, got {s_self:?}");
                };
                // unaryop.py:972-980 — `SomePBC.getattr(self, s_attr)`.
                // Upstream only asserts `s_attr.is_constant()` and
                // uses `s_attr.const` directly — no `isinstance(...,
                // str)` gate. Python 2 string equality crosses the
                // bytes/unicode boundary (`'__name__' == u'__name__'`
                // is True), so accept both `ByteStr` and `UniStr` via
                // [`ConstValue::as_text`] to keep parity with the
                // upstream constant-only check.
                let Some(attr) = s_attr
                    .const_()
                    .and_then(ConstValue::as_text)
                    .map(str::to_owned)
                else {
                    panic!(
                        "AnnotatorError: pbc getattr({:?}, {:?}) has non-constant argument",
                        s_self, s_attr
                    );
                };
                // __name__ special-case: ClassDesc PBC → SomeString.
                if attr == "__name__" && matches!(pbc.get_kind(), Ok(super::model::DescKind::Class))
                {
                    return SomeValue::String(SomeString::new(false, false));
                }
                match ann.bookkeeper.pbc_getattr(pbc, &s_attr) {
                    Ok(s) => s,
                    Err(super::model::AnnotatorException::Annotator(e)) => {
                        panic!("AnnotatorError in pbc_getattr: {e}")
                    }
                    Err(super::model::AnnotatorException::Harmless(_)) => {
                        // upstream lets `HarmlesslyBlocked` propagate out of
                        // flowin; the Rust port has no exception channel
                        // across Specialization::apply yet, so surface as
                        // Impossible and let the reflow loop retry once the
                        // enforced attr family widens.
                        super::model::s_impossible_value()
                    }
                }
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:200-204 — delattr: warning-only. Returns no value
    // (upstream implicit None). Port as a void specialization returning
    // Impossible.
    register(
        reg,
        OpKind::DelAttr,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Impossible),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:114-118 — `@op.simple_call.register(SomeObject)`.
    //
    //     @op.simple_call.register(SomeObject)
    //     def simple_call_SomeObject(annotator, func, *args):
    //         s_func = annotator.annotation(func)
    //         argspec = simple_args([annotator.annotation(arg) for arg in args])
    //         return s_func.call(argspec)
    //
    // Polymorphic dispatch entry — registered against `SomeObject` so
    // every `s_func` tag (`SomePBC`, `SomeBuiltinMethod`, `SomePtr`,
    // …) inherits this specialization via mro lookup.  Each subtype's
    // `.call(argspec)` then routes to the type-specific handler:
    // `SomePBC.call → Bookkeeper.pbc_call → FunctionDesc.pycall →
    // annotator.recursivecall → addpendingblock`
    // (`description.py:283-305`, `annrpython.py:315-336`).
    register(
        reg,
        OpKind::SimpleCall,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|ann, hl| {
                // Mirror RPython `unaryop.py:114-118`:
                //
                //     s_func = annotator.annotation(func)
                //     argspec = simple_args([annotator.annotation(arg)
                //                            for arg in args])
                //     return s_func.call(argspec)
                //
                // `annotator.annotation(arg)` returns `None` when the
                // variable is not yet bound (`annrpython.py:273-280`);
                // pyre's `arguments_w: Vec<Option<SomeValue>>` carries
                // the None through to consumers, matching
                // `bookkeeper.py:152` parity.  When `s_func` itself is
                // unbound, the call is deferred (return Impossible so
                // the fixpoint propagates Bottom upward; the next
                // round re-fires once the callable's annotation
                // populates).
                let Some(s_func) = ann.annotation(&hl.args[0]) else {
                    return super::model::s_impossible_value();
                };
                let args_s: Vec<Option<SomeValue>> =
                    hl.args[1..].iter().map(|a| ann.annotation(a)).collect();
                let argspec = super::argument::simple_args_opt(args_s);
                match s_func.call(&argspec) {
                    Ok(s) => s,
                    // Match upstream's substring so the dual-gate
                    // `is_known_unported` Skip classification matches
                    // RPython's `compute_at_fixpoint` failures
                    // (`bookkeeper.py:108-118` propagates uncaught).
                    Err(e) => panic!("compute_at_fixpoint failed: {e}"),
                }
            }),
            // upstream `simple_call.can_only_throw` is unset →
            // `CanOnlyThrow::Absent` matches `_can_only_throw_None`'s
            // path for ops without an explicit static throw set.
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:240-241 — hint(self, *args_s): return self.
    register(
        reg,
        OpKind::Hint,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|ann, hl| ann.annotation(&hl.args[0]).expect("hint: self unbound")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:243-250 — getslice/setslice/delslice: s_ImpossibleValue.
    for op in &[OpKind::GetSlice, OpKind::SetSlice, OpKind::DelSlice] {
        register(
            reg,
            *op,
            SomeValueTag::Object,
            Specialization {
                apply: Box::new(|_ann, _hl| s_impossible_value()),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
    // unaryop.py:252-254 — pos/neg/abs/ord/invert/long/iter/next: s_ImpossibleValue.
    //
    //     def pos(self):
    //         return s_ImpossibleValue
    //     neg = abs = ord = invert = long = iter = next = pos
    for op in &[
        OpKind::Pos,
        OpKind::Neg,
        OpKind::Abs,
        OpKind::Ord,
        OpKind::Invert,
        OpKind::Long,
        OpKind::Iter,
        OpKind::Next,
    ] {
        register(
            reg,
            *op,
            SomeValueTag::Object,
            Specialization {
                apply: Box::new(|_ann, _hl| s_impossible_value()),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
}

// =====================================================================
// unaryop.py:257-273 — class __extend__(SomeFloat)
// =====================================================================

fn init_somefloat_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:259-260 — pos(self): return self.
    register(
        reg,
        OpKind::Pos,
        SomeValueTag::Float,
        Specialization {
            apply: Box::new(|ann, hl| ann.annotation(&hl.args[0]).expect("float.pos: unbound")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:262-265 — neg(self): SomeFloat(). abs = neg.
    for op in &[OpKind::Neg, OpKind::Abs] {
        register(
            reg,
            *op,
            SomeValueTag::Float,
            Specialization {
                apply: Box::new(|_ann, _hl| SomeValue::Float(SomeFloat::new())),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
    // unaryop.py:267-270 — bool(self): SomeBool (const if self is constant).
    register(
        reg,
        OpKind::Bool,
        SomeValueTag::Float,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s = ann.annotation(&hl.args[0]).expect("float.bool: unbound");
                if let SomeValue::Float(f) = &s {
                    if f.is_immutable_constant() {
                        if let Some(c) = &f.base.const_box {
                            if let ConstValue::Float(bits) = c.value {
                                let mut r = SomeBool::new();
                                r.base.const_box = Some(Constant::new(ConstValue::Bool(
                                    f64::from_bits(bits) != 0.0,
                                )));
                                return SomeValue::Bool(r);
                            }
                        }
                    }
                }
                SomeValue::Bool(SomeBool::new())
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:272-273 — len(self): raise AnnotatorError.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Float,
        Specialization {
            apply: Box::new(|_ann, _hl| panic!("AnnotatorError: 'float' has no length")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:275-302 — class __extend__(SomeInteger)
// =====================================================================

fn init_someinteger_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:277-285 — invert/pos/int: SomeInteger(knowntype=self.knowntype).
    //                      can_only_throw = [].
    let same_kt_int = |ann: &RPythonAnnotator, hl: &HLOperation| -> SomeValue {
        let kt = match ann.annotation(&hl.args[0]) {
            Some(SomeValue::Integer(i)) => i.base.knowntype,
            _ => panic!("integer op: arg 0 not SomeInteger"),
        };
        SomeValue::Integer(SomeInteger::new_with_knowntype(false, kt))
    };
    for op in &[OpKind::Invert, OpKind::Pos, OpKind::Int] {
        register(
            reg,
            *op,
            SomeValueTag::Integer,
            Specialization {
                apply: Box::new(same_kt_int),
                can_only_throw: CanOnlyThrow::List(vec![]),
            },
        );
    }
    // unaryop.py:289-293 — neg: SomeInteger(knowntype=self.knowntype). neg_ovf overflows.
    register(
        reg,
        OpKind::Neg,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(same_kt_int),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::NegOvf,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(same_kt_int),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::OverflowError]),
        },
    );
    // unaryop.py:295-299 — abs: SomeInteger(nonneg=True, knowntype=self.knowntype).
    let abs_int = |ann: &RPythonAnnotator, hl: &HLOperation| -> SomeValue {
        let kt = match ann.annotation(&hl.args[0]) {
            Some(SomeValue::Integer(i)) => i.base.knowntype,
            _ => panic!("integer.abs: arg 0 not SomeInteger"),
        };
        // upstream: SomeInteger(nonneg=True, knowntype=self.knowntype).
        let mut i = SomeInteger::new_with_knowntype(true, kt);
        i.nonneg = true;
        SomeValue::Integer(i)
    };
    register(
        reg,
        OpKind::Abs,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(abs_int),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::AbsOvf,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(abs_int),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::OverflowError]),
        },
    );
    // unaryop.py:301-302 — len: raise.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(|_ann, _hl| panic!("AnnotatorError: 'int' has no length")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:305-330 — class __extend__(SomeBool)
// =====================================================================

fn init_somebool_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:306-307 — bool(self): return self.
    register(
        reg,
        OpKind::Bool,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(|ann, hl| ann.annotation(&hl.args[0]).expect("bool.bool: unbound")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:309-312 — invert: SomeInteger(). can_only_throw=[].
    register(
        reg,
        OpKind::Invert,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Integer(SomeInteger::default())),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:314-318 — neg: SomeInteger(). neg_ovf overflows.
    register(
        reg,
        OpKind::Neg,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Integer(SomeInteger::default())),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::NegOvf,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Integer(SomeInteger::default())),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::OverflowError]),
        },
    );
    // unaryop.py:320-330 — abs / abs_ovf / pos / int: SomeInteger(nonneg=True).
    let nonneg_int = |_ann: &RPythonAnnotator, _hl: &HLOperation| -> SomeValue {
        let mut i = SomeInteger::default();
        i.nonneg = true;
        SomeValue::Integer(i)
    };
    register(
        reg,
        OpKind::Abs,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(nonneg_int),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::AbsOvf,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(nonneg_int),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::OverflowError]),
        },
    );
    register(
        reg,
        OpKind::Pos,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(nonneg_int),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::Int,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(nonneg_int),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:332-348 — class __extend__(SomeTuple)
// =====================================================================

fn init_sometuple_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:334-335 — len(self): immutablevalue(len(self.items)).
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Tuple,
        Specialization {
            apply: Box::new(|ann, hl| {
                let t = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::Tuple(t)) => t,
                    _ => panic!("tuple.len: arg 0 not SomeTuple"),
                };
                let mut i = SomeInteger::new(true, false);
                i.base.const_box = Some(Constant::new(ConstValue::Int(t.items.len() as i64)));
                SomeValue::Integer(i)
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:337-339 — iter(self): SomeIterator(self). can_only_throw=[].
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::Tuple,
        Specialization {
            apply: Box::new(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("tuple.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:344-348 — getslice((self, s_start, s_stop)): require
    // constant indices, slice items, rebuild SomeTuple. Registered as
    // the primary SomeTuple.getslice handler (the SomeObject fallback
    // would have returned s_ImpossibleValue).
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::Tuple,
        Specialization {
            apply: Box::new(|ann, hl| {
                let t = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::Tuple(t)) => t,
                    _ => panic!("tuple.getslice: arg 0 not SomeTuple"),
                };
                let start = ann
                    .annotation(&hl.args[1])
                    .expect("tuple.getslice: start unbound");
                let stop = ann
                    .annotation(&hl.args[2])
                    .expect("tuple.getslice: stop unbound");
                let start_c = match &start {
                    SomeValue::Integer(i) if i.is_immutable_constant() => i
                        .base
                        .const_box
                        .as_ref()
                        .and_then(|c| match c.value {
                            ConstValue::Int(v) => Some(v),
                            _ => None,
                        })
                        .expect("tuple.getslice: const_box mismatch"),
                    _ => panic!("AnnotatorError: tuple slicing: needs constants (start not const)"),
                };
                let stop_c = match &stop {
                    SomeValue::Integer(i) if i.is_immutable_constant() => i
                        .base
                        .const_box
                        .as_ref()
                        .and_then(|c| match c.value {
                            ConstValue::Int(v) => Some(v),
                            _ => None,
                        })
                        .expect("tuple.getslice: const_box mismatch"),
                    _ => panic!("AnnotatorError: tuple slicing: needs constants (stop not const)"),
                };
                let len = t.items.len() as i64;
                let clamp = |idx: i64| -> usize {
                    if idx < 0 {
                        (len + idx).max(0) as usize
                    } else {
                        (idx as usize).min(t.items.len())
                    }
                };
                let lo = clamp(start_c);
                let hi = clamp(stop_c).max(lo);
                let items = t.items[lo..hi].to_vec();
                SomeValue::Tuple(SomeTuple::new(items))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:350-443 — @op.contains.register(SomeList) + class __extend__(SomeList)
// =====================================================================

fn init_somelist_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:350-354 — contains: generalize + s_Bool, can_only_throw=[].
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::List,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_list = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::List(s)) => s,
                    _ => panic!("contains(SomeList): arg 0 not SomeList"),
                };
                let s_element = ann
                    .annotation(&hl.args[1])
                    .expect("contains(SomeList): element unbound");
                s_list
                    .listdef
                    .generalize(&s_element)
                    .expect("contains.SomeList: generalize failed");
                SomeValue::Bool(SomeBool::new())
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:391-396 — len(self): position read_item → 0 if impossible,
    // else SomeObject.len.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::List,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_list = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::List(s)) => s,
                    _ => panic!("list.len: arg 0 not SomeList"),
                };
                let position = ann.bookkeeper.current_position_key();
                let s_item = s_list.listdef.read_item(position);
                if matches!(s_item, SomeValue::Impossible) {
                    // upstream: immutablevalue(0) → SomeInteger with const=0.
                    let mut i = SomeInteger::new(true, false);
                    i.base.const_box = Some(Constant::new(ConstValue::Int(0)));
                    SomeValue::Integer(i)
                } else {
                    // upstream: SomeObject.len(self) → SomeInteger(nonneg=True).
                    SomeValue::Integer(SomeInteger::new(true, false))
                }
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:398-400 — iter(self): SomeIterator(self). can_only_throw=[].
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::List,
        Specialization {
            apply: Box::new(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("list.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:420-423 — getslice((self, s_start, s_stop)):
    //     check_negative_slice(...); return listdef.offspring(bk).
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::List,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_list = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::List(s)) => s,
                    _ => panic!("list.getslice: arg 0 not SomeList"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("list.getslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("list.getslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                let res = s_list
                    .listdef
                    .offspring(&ann.bookkeeper, &[])
                    .expect("listdef.offspring failed");
                SomeValue::List(res)
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:425-431 — setslice((self, s_start, s_stop), s_iterable):
    //     check_negative + isinstance(SomeList) + mutate + agree + resize.
    register(
        reg,
        OpKind::SetSlice,
        SomeValueTag::List,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_list = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::List(s)) => s,
                    _ => panic!("list.setslice: arg 0 not SomeList"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("list.setslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("list.setslice: stop unbound");
                let s_iterable = ann
                    .annotation(&hl.args[3])
                    .expect("list.setslice: iterable unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                let s_other = match s_iterable {
                    SomeValue::List(o) => o,
                    _ => panic!("AnnotatorError: list[start:stop] = x: x must be a list"),
                };
                s_list.listdef.mutate().expect("listdef.mutate failed");
                s_list
                    .listdef
                    .agree(&ann.bookkeeper, &s_other.listdef)
                    .expect("listdef.agree failed");
                s_list.listdef.resize().expect("listdef.resize failed");
                SomeValue::Impossible
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:433-435 — delslice((self, s_start, s_stop)):
    //     check_negative + resize.
    register(
        reg,
        OpKind::DelSlice,
        SomeValueTag::List,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_list = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::List(s)) => s,
                    _ => panic!("list.delslice: arg 0 not SomeList"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("list.delslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("list.delslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                s_list.listdef.resize().expect("listdef.resize failed");
                SomeValue::Impossible
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

/// RPython `check_negative_slice(s_start, s_stop, error="slicing")`
/// (unaryop.py:437-443). Helper shared by list and string slice paths.
fn check_negative_slice(s_start: &SomeValue, s_stop: &SomeValue, error: &str) {
    if let SomeValue::Integer(i) = s_start {
        if !i.nonneg {
            panic!("AnnotatorError: {error}: not proven to have non-negative start");
        }
    }
    if let SomeValue::Integer(i) = s_stop {
        if !i.nonneg {
            // upstream: allow const == -1 as sentinel for "slice to end".
            let is_minus_one = i
                .base
                .const_box
                .as_ref()
                .map(|c| matches!(c.value, ConstValue::Int(-1)))
                .unwrap_or(false);
            if !is_minus_one {
                panic!("AnnotatorError: {error}: not proven to have non-negative stop");
            }
        }
    }
}

// =====================================================================
// unaryop.py:357-418 — SomeList.method_* free functions
// =====================================================================
//
// Upstream these are class methods on `class __extend__(SomeList)`. They
// are NOT directly `@op.X.register` targets; instead `SomeObject.getattr`
// looks up `method_<name>` via `find_method` and wraps them in a
// `SomeBuiltinMethod` for downstream `simple_call`. The Rust port keeps
// the analyzer bodies as free functions and the `find_method` /
// `SomeBuiltinMethod.call` bridge below dispatches to them.

#[allow(dead_code)]
pub fn list_method_append(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    s_value: &SomeValue,
) -> SomeValue {
    // unaryop.py:359-361
    s_self.listdef.resize().expect("resize");
    s_self.listdef.generalize(s_value).expect("generalize");
    SomeValue::Impossible
}

#[allow(dead_code)]
pub fn list_method_extend(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    s_iterable: &SomeValue,
) -> SomeValue {
    // unaryop.py:363-369
    s_self.listdef.resize().expect("resize");
    if let SomeValue::List(other) = s_iterable {
        s_self
            .listdef
            .agree(&ann.bookkeeper, &other.listdef)
            .expect("agree");
    } else {
        // upstream:
        //     s_iter = s_iterable.iter()
        //     self.method_append(s_iter.next())
        //
        // `SomeIterator(s_iterable, variant=())` mirrors every
        // `SomeX.iter()` handler (unaryop.py:337/389/660/806/…),
        // which all reduce to `SomeIterator(self)` on the iterable.
        // `someiterator_next` is the extracted body of upstream
        // `SomeIterator.next` (unaryop.py:818-828).
        let s_iter = SomeIterator::new(s_iterable.clone(), vec![]);
        let s_item = someiterator_next(ann, &s_iter);
        list_method_append(ann, s_self, &s_item);
    }
    SomeValue::Impossible
}

#[allow(dead_code)]
pub fn list_method_reverse(_ann: &RPythonAnnotator, s_self: &super::model::SomeList) -> SomeValue {
    s_self.listdef.mutate().expect("listdef.mutate");
    SomeValue::Impossible
}

#[allow(dead_code)]
pub fn list_method_insert(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    _s_index: &SomeValue,
    s_value: &SomeValue,
) -> SomeValue {
    // unaryop.py:374-375 — delegates to method_append.
    list_method_append(ann, s_self, s_value)
}

#[allow(dead_code)]
pub fn list_method_remove(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    s_value: &SomeValue,
) -> SomeValue {
    // unaryop.py:377-379 — resize + generalize.
    s_self.listdef.resize().expect("resize");
    s_self.listdef.generalize(s_value).expect("generalize");
    SomeValue::Impossible
}

#[allow(dead_code)]
pub fn list_method_pop(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    _s_index: Option<&SomeValue>,
) -> SomeValue {
    // unaryop.py:381-385 — resize + read_item(position). can_only_throw=[IndexError].
    s_self.listdef.resize().expect("resize");
    let position = ann.bookkeeper.current_position_key();
    s_self.listdef.read_item(position)
}

#[allow(dead_code)]
pub fn list_method_index(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    s_value: &SomeValue,
) -> SomeValue {
    // unaryop.py:387-389 — generalize + SomeInteger(nonneg=True).
    s_self.listdef.generalize(s_value).expect("generalize");
    SomeValue::Integer(SomeInteger::new(true, false))
}

// =====================================================================
// unaryop.py:446-460 — @op.contains.register(SomeDict) + dict_contains helper
// =====================================================================

/// RPython `dict_contains(s_dct, s_element, position)` (unaryop.py:446-452).
///
/// ```python
/// def dict_contains(s_dct, s_element, position):
///     s_dct.dictdef.generalize_key(s_element)
///     if s_dct._is_empty(position):
///         s_bool = SomeBool(); s_bool.const = False; return s_bool
///     return s_Bool
/// ```
fn dict_contains(
    s_dct: &super::model::SomeDict,
    s_element: &SomeValue,
    position: Option<super::bookkeeper::PositionKey>,
) -> SomeValue {
    s_dct
        .dictdef
        .generalize_key(s_element)
        .expect("dict_contains: generalize_key failed");
    if dict_is_empty(s_dct, position) {
        let mut s = SomeBool::new();
        s.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
        return SomeValue::Bool(s);
    }
    SomeValue::Bool(SomeBool::new())
}

/// RPython `SomeDict._is_empty(self, position)` (unaryop.py:464-468).
fn dict_is_empty(
    s_dct: &super::model::SomeDict,
    position: Option<super::bookkeeper::PositionKey>,
) -> bool {
    let s_key = s_dct.dictdef.read_key(position.clone());
    let s_value = s_dct.dictdef.read_value(position);
    matches!(s_key, SomeValue::Impossible) || matches!(s_value, SomeValue::Impossible)
}

fn init_somedict_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:454-460 — @op.contains.register(SomeDict):
    //     position = annotator.bookkeeper.position_key
    //     return dict_contains(...); can_only_throw = _dict_can_only_throw_nothing.
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::Dict,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_dct = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::Dict(d)) => d,
                    _ => panic!("contains(SomeDict): arg 0 not SomeDict"),
                };
                let s_elem = ann
                    .annotation(&hl.args[1])
                    .expect("contains(SomeDict): element unbound");
                let position = ann.bookkeeper.current_position_key();
                dict_contains(&s_dct, &s_elem, position)
            }),
            can_only_throw: CanOnlyThrow::Callable(Box::new(|args_s| {
                // Mirror binaryop::_dict_can_only_throw_nothing (binaryop.py:532-535).
                match args_s.first() {
                    Some(SomeValue::Dict(d)) => {
                        if d.dictdef.custom_eq_hash() {
                            None
                        } else {
                            Some(vec![])
                        }
                    }
                    _ => Some(vec![]),
                }
            })),
        },
    );
    // unaryop.py:470-474 — len(self): 0 if empty, else SomeObject.len.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Dict,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_dct = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::Dict(d)) => d,
                    _ => panic!("dict.len: arg 0 not SomeDict"),
                };
                let position = ann.bookkeeper.current_position_key();
                if dict_is_empty(&s_dct, position) {
                    let mut i = SomeInteger::new(true, false);
                    i.base.const_box = Some(Constant::new(ConstValue::Int(0)));
                    SomeValue::Integer(i)
                } else {
                    SomeValue::Integer(SomeInteger::new(true, false))
                }
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:476-478 — iter(self): SomeIterator(self). can_only_throw=[].
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::Dict,
        Specialization {
            apply: Box::new(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("dict.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
}

// =====================================================================
// unaryop.py:462-591 — SomeDict.method_* free functions
// =====================================================================

#[allow(dead_code)]
pub fn dict_method_get(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    s_dfl: &SomeValue,
) -> SomeValue {
    // unaryop.py:502-506
    let position = ann.bookkeeper.current_position_key();
    s_self
        .dictdef
        .generalize_key(s_key)
        .expect("generalize_key");
    s_self
        .dictdef
        .generalize_value(s_dfl)
        .expect("generalize_value");
    s_self.dictdef.read_value(position)
}

#[allow(dead_code)]
pub fn dict_method_copy(_ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py:510-511 — SomeDict(self.dictdef).
    SomeValue::Dict(super::model::SomeDict::new(s_self.dictdef.clone()))
}

#[allow(dead_code)]
pub fn dict_method_update(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_other: &SomeValue,
) -> SomeValue {
    // unaryop.py:513-516
    //     if s_None.contains(dct2): return SomeImpossibleValue()
    //     dct1.dictdef.union(dct2.dictdef)
    if let SomeValue::None_(_) = s_other {
        return s_impossible_value();
    }
    let other = match s_other {
        SomeValue::Dict(d) => d,
        _ => panic!("dict.update: other not SomeDict or SomeNone"),
    };
    s_self
        .dictdef
        .union_with(&other.dictdef)
        .expect("dictdef.union failed");
    SomeValue::Impossible
}

#[allow(dead_code)]
pub fn dict_method_prepare_dict_update(
    _ann: &RPythonAnnotator,
    _s_self: &super::model::SomeDict,
    _s_num: &SomeValue,
) -> SomeValue {
    // unaryop.py:518-519 — pass.
    SomeValue::Impossible
}

#[allow(dead_code)]
pub fn dict_method_keys(ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py:521-523 — bk.newlist(self.dictdef.read_key(bk.position_key)).
    let position = ann.bookkeeper.current_position_key();
    let s_key = s_self.dictdef.read_key(position);
    let s_list = ann
        .bookkeeper
        .newlist(&[s_key], None)
        .expect("bookkeeper.newlist failed");
    SomeValue::List(s_list)
}

#[allow(dead_code)]
pub fn dict_method_values(ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py:525-527
    let position = ann.bookkeeper.current_position_key();
    let s_value = s_self.dictdef.read_value(position);
    let s_list = ann
        .bookkeeper
        .newlist(&[s_value], None)
        .expect("bookkeeper.newlist failed");
    SomeValue::List(s_list)
}

#[allow(dead_code)]
pub fn dict_method_items(ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py:529-531 — bk.newlist(self.getanyitem(..., variant='items')).
    let position = ann.bookkeeper.current_position_key();
    let s_item = container_getanyitem(&SomeValue::Dict(s_self.clone()), Some("items"), position);
    let s_list = ann
        .bookkeeper
        .newlist(&[s_item], None)
        .expect("bookkeeper.newlist failed");
    SomeValue::List(s_list)
}

#[allow(dead_code)]
pub fn dict_method_iterkeys(_ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py:533-534 — SomeIterator(self, 'keys').
    SomeValue::Iterator(SomeIterator::new(
        SomeValue::Dict(s_self.clone()),
        vec!["keys".into()],
    ))
}

#[allow(dead_code)]
pub fn dict_method_itervalues(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
) -> SomeValue {
    SomeValue::Iterator(SomeIterator::new(
        SomeValue::Dict(s_self.clone()),
        vec!["values".into()],
    ))
}

#[allow(dead_code)]
pub fn dict_method_iteritems(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
) -> SomeValue {
    SomeValue::Iterator(SomeIterator::new(
        SomeValue::Dict(s_self.clone()),
        vec!["items".into()],
    ))
}

#[allow(dead_code)]
pub fn dict_method_iterkeys_with_hash(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
) -> SomeValue {
    SomeValue::Iterator(SomeIterator::new(
        SomeValue::Dict(s_self.clone()),
        vec!["keys_with_hash".into()],
    ))
}

#[allow(dead_code)]
pub fn dict_method_iteritems_with_hash(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
) -> SomeValue {
    SomeValue::Iterator(SomeIterator::new(
        SomeValue::Dict(s_self.clone()),
        vec!["items_with_hash".into()],
    ))
}

#[allow(dead_code)]
pub fn dict_method_clear(_ann: &RPythonAnnotator, _s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py:548-549 — pass.
    SomeValue::Impossible
}

#[allow(dead_code)]
pub fn dict_method_popitem(ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py:551-553 — return self.getanyitem(position, variant='items').
    let position = ann.bookkeeper.current_position_key();
    container_getanyitem(&SomeValue::Dict(s_self.clone()), Some("items"), position)
}

#[allow(dead_code)]
pub fn dict_method_pop(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    s_dfl: Option<&SomeValue>,
) -> SomeValue {
    // unaryop.py:555-560
    s_self
        .dictdef
        .generalize_key(s_key)
        .expect("generalize_key");
    if let Some(s) = s_dfl {
        s_self
            .dictdef
            .generalize_value(s)
            .expect("generalize_value");
    }
    let position = ann.bookkeeper.current_position_key();
    s_self.dictdef.read_value(position)
}

#[allow(dead_code)]
pub fn dict_method_contains_with_hash(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    _s_hash: &SomeValue,
) -> SomeValue {
    // unaryop.py:562-565.
    let position = ann.bookkeeper.current_position_key();
    dict_contains(s_self, s_key, position)
}

#[allow(dead_code)]
pub fn dict_method_setitem_with_hash(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    _s_hash: &SomeValue,
    s_value: &SomeValue,
) -> SomeValue {
    // unaryop.py:567-568.
    s_self
        .dictdef
        .generalize_key(s_key)
        .expect("generalize_key failed");
    s_self
        .dictdef
        .generalize_value(s_value)
        .expect("generalize_value failed");
    SomeValue::Impossible
}

#[allow(dead_code)]
pub fn dict_method_getitem_with_hash(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    _s_hash: &SomeValue,
) -> SomeValue {
    // unaryop.py:570-574 — copy of binaryop.getitem_SomeDict.
    s_self
        .dictdef
        .generalize_key(s_key)
        .expect("generalize_key failed");
    let position = ann.bookkeeper.current_position_key();
    s_self.dictdef.read_value(position)
}

#[allow(dead_code)]
pub fn dict_method_delitem_with_hash(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    _s_hash: &SomeValue,
) -> SomeValue {
    // unaryop.py:576-577.
    s_self
        .dictdef
        .generalize_key(s_key)
        .expect("generalize_key failed");
    SomeValue::Impossible
}

#[allow(dead_code)]
pub fn dict_method_delitem_if_value_is(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    s_value: &SomeValue,
) -> SomeValue {
    // unaryop.py:579-581.
    let _ = dict_method_setitem_with_hash(ann, s_self, s_key, &s_impossible_value(), s_value);
    dict_method_delitem_with_hash(ann, s_self, s_key, &s_impossible_value())
}

#[allow(dead_code)]
pub fn dict_method_move_to_end(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    s_last: &SomeValue,
) -> SomeValue {
    // unaryop.py:588-591 on SomeOrderedDict. Rust collapses SomeDict =
    // SomeOrderedDict (model.py:416), so the same analyzer lives on SomeDict.
    assert!(
        s_bool().contains(s_last),
        "AnnotatorError: move_to_end(last) expects SomeBool"
    );
    dict_method_delitem_with_hash(ann, s_self, s_key, &s_impossible_value())
}

// =====================================================================
// unaryop.py:593-690 — @op.contains.register(SomeString|SomeUnicodeString)
// + class __extend__(SomeString, SomeUnicodeString) shared overrides
// =====================================================================

fn init_somestring_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:593-605 — contains_String:
    //   if const '\0': knowntypedata-based refinement, else SomeObject fallback.
    //   can_only_throw=[].
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::String,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_string = ann
                    .annotation(&hl.args[0])
                    .expect("contains(String): string unbound");
                let s_char = ann
                    .annotation(&hl.args[1])
                    .expect("contains(String): char unbound");
                if const_str_of(&s_char).as_deref() == Some("\0") {
                    let mut r = SomeBool::new();
                    let mut knowntypedata = std::collections::HashMap::new();
                    let vars = match &hl.args[0] {
                        Hlvalue::Variable(v) => vec![Rc::new(v.clone())],
                        Hlvalue::Constant(_) => Vec::new(),
                    };
                    let s_nonnul = s_string
                        .nonnulify()
                        .expect("contains(String): nonnulify must succeed");
                    add_knowntypedata(&mut knowntypedata, false, &vars, s_nonnul);
                    r.set_knowntypedata(knowntypedata);
                    return SomeValue::Bool(r);
                }
                SomeValue::Bool(SomeBool::new())
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:660-662 — iter(self): SomeIterator(self). can_only_throw=[].
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::String,
        Specialization {
            apply: Box::new(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("string.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:684-687 — getslice: check_negative_slice + basestringclass(no_nul).
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::String,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_self = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::String(s)) => s,
                    _ => panic!("string.getslice: arg 0 not SomeString"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("string.getslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("string.getslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                SomeValue::String(SomeString::new(false, s_self.inner.no_nul))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:720-725 — len(self): immutablevalue(len(const)) if const, else SomeInteger(nonneg).
    register(
        reg,
        OpKind::Len,
        SomeValueTag::String,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::String(s)) => s,
                    _ => panic!("string.len: arg 0 not SomeString"),
                };
                if s.is_constant() {
                    if let Some(c) = &s.inner.base.const_box {
                        if let ConstValue::ByteStr(v) = &c.value {
                            let mut i = SomeInteger::new(true, false);
                            i.base.const_box = Some(Constant::new(ConstValue::Int(v.len() as i64)));
                            return SomeValue::Integer(i);
                        }
                    }
                }
                SomeValue::Integer(SomeInteger::new(true, false))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

fn init_someunicodestring_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:593-605 — contains on SomeUnicodeString (same path).
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::UnicodeString,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_string = ann
                    .annotation(&hl.args[0])
                    .expect("contains(UnicodeString): string unbound");
                let s_char = ann
                    .annotation(&hl.args[1])
                    .expect("contains(UnicodeString): char unbound");
                if const_str_of(&s_char).as_deref() == Some("\0") {
                    let mut r = SomeBool::new();
                    let mut knowntypedata = std::collections::HashMap::new();
                    let vars = match &hl.args[0] {
                        Hlvalue::Variable(v) => vec![Rc::new(v.clone())],
                        Hlvalue::Constant(_) => Vec::new(),
                    };
                    let s_nonnul = s_string
                        .nonnulify()
                        .expect("contains(UnicodeString): nonnulify must succeed");
                    add_knowntypedata(&mut knowntypedata, false, &vars, s_nonnul);
                    r.set_knowntypedata(knowntypedata);
                    return SomeValue::Bool(r);
                }
                SomeValue::Bool(SomeBool::new())
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:660-662 — iter on SomeUnicodeString.
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::UnicodeString,
        Specialization {
            apply: Box::new(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("unicode.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:684-687 — getslice on SomeUnicodeString (basestringclass=SomeUnicodeString).
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::UnicodeString,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_self = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::UnicodeString(s)) => s,
                    _ => panic!("unicode.getslice: arg 0 not SomeUnicodeString"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("unicode.getslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("unicode.getslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                SomeValue::UnicodeString(SomeUnicodeString::new(false, s_self.inner.no_nul))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

fn init_somebytearray_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:693-696 — SomeByteArray.getslice: check + SomeByteArray().
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::ByteArray,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("bytearray.getslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("bytearray.getslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                SomeValue::ByteArray(super::model::SomeByteArray::new(false))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:771-797 — class __extend__(SomeChar, SomeUnicodeCodePoint) / SomeChar
// =====================================================================

fn init_somechar_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:771-774 — len(self) = immutablevalue(1).
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Char,
        Specialization {
            apply: Box::new(|_ann, _hl| {
                let mut i = SomeInteger::new(true, false);
                i.base.const_box = Some(Constant::new(ConstValue::Int(1)));
                SomeValue::Integer(i)
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:776-779 — ord(self) = SomeInteger(nonneg=True).
    register(
        reg,
        OpKind::Ord,
        SomeValueTag::Char,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Integer(SomeInteger::new(true, false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

fn init_someunicodecp_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:771-774 — len(self) = immutablevalue(1) on SomeUnicodeCodePoint.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::UnicodeCodePoint,
        Specialization {
            apply: Box::new(|_ann, _hl| {
                let mut i = SomeInteger::new(true, false);
                i.base.const_box = Some(Constant::new(ConstValue::Int(1)));
                SomeValue::Integer(i)
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:799-804 — ord(self): SomeInteger(nonneg=True).
    register(
        reg,
        OpKind::Ord,
        SomeValueTag::UnicodeCodePoint,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Integer(SomeInteger::new(true, false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:608-690 / 720-748 / 776-797 — method_* free fns (String/Char)
// =====================================================================

/// RPython "basestringclass" picks SomeString vs SomeUnicodeString based
/// on the actual class of the SomeValue. SomeChar → SomeString,
/// SomeUnicodeCodePoint → SomeUnicodeString.
fn basestring_of(sv: &SomeValue, no_nul: bool) -> SomeValue {
    match sv {
        SomeValue::String(_) | SomeValue::Char(_) => {
            SomeValue::String(SomeString::new(false, no_nul))
        }
        SomeValue::UnicodeString(_) | SomeValue::UnicodeCodePoint(_) => {
            SomeValue::UnicodeString(SomeUnicodeString::new(false, no_nul))
        }
        other => panic!("basestring_of: unexpected {:?}", other),
    }
}

#[allow(dead_code)]
pub fn str_method_startswith(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    s_frag: &SomeValue,
) -> SomeValue {
    // unaryop.py:611-614 — const+const → immutablevalue(self.const.startswith(frag.const))
    //                       else s_Bool.
    if let (Some(a), Some(b)) = (const_str_of(s_self), const_str_of(s_frag)) {
        let mut r = SomeBool::new();
        r.base.const_box = Some(Constant::new(ConstValue::Bool(a.starts_with(&b))));
        return SomeValue::Bool(r);
    }
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
pub fn str_method_endswith(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    s_frag: &SomeValue,
) -> SomeValue {
    if let (Some(a), Some(b)) = (const_str_of(s_self), const_str_of(s_frag)) {
        let mut r = SomeBool::new();
        r.base.const_box = Some(Constant::new(ConstValue::Bool(a.ends_with(&b))));
        return SomeValue::Bool(r);
    }
    SomeValue::Bool(SomeBool::new())
}

fn const_str_of(sv: &SomeValue) -> Option<String> {
    let cb = match sv {
        SomeValue::String(s) => &s.inner.base.const_box,
        SomeValue::UnicodeString(s) => &s.inner.base.const_box,
        SomeValue::Char(s) => &s.inner.base.const_box,
        SomeValue::UnicodeCodePoint(s) => &s.inner.base.const_box,
        _ => return None,
    };
    cb.as_ref().and_then(|c| match &c.value {
        ConstValue::ByteStr(v) => String::from_utf8(v.clone()).ok(),
        ConstValue::UniStr(v) => Some(v.clone()),
        _ => None,
    })
}

fn const_int_of(sv: &SomeValue) -> Option<i64> {
    match sv.const_() {
        Some(ConstValue::Int(v)) => Some(*v),
        _ => None,
    }
}

fn stringish_no_nul(sv: &SomeValue) -> bool {
    match sv {
        SomeValue::String(s) => s.inner.no_nul,
        SomeValue::UnicodeString(s) => s.inner.no_nul,
        SomeValue::Char(s) => s.inner.no_nul,
        SomeValue::UnicodeCodePoint(s) => s.inner.no_nul,
        _ => false,
    }
}

fn s_const_int(value: i64) -> SomeValue {
    let mut s = SomeInteger::new(value >= 0, false);
    s.base.const_box = Some(Constant::new(ConstValue::Int(value)));
    SomeValue::Integer(s)
}

#[allow(dead_code)]
pub fn str_method_find(
    _ann: &RPythonAnnotator,
    _s_self: &SomeValue,
    _s_frag: &SomeValue,
    s_start: Option<&SomeValue>,
    s_end: Option<&SomeValue>,
) -> SomeValue {
    // unaryop.py:621-623 — check_negative_slice(start,end,"find") → SomeInteger().
    if let (Some(start), Some(end)) = (s_start, s_end) {
        check_negative_slice(start, end, "find");
    }
    SomeValue::Integer(SomeInteger::default())
}

#[allow(dead_code)]
pub fn str_method_rfind(
    _ann: &RPythonAnnotator,
    _s_self: &SomeValue,
    _s_frag: &SomeValue,
    s_start: Option<&SomeValue>,
    s_end: Option<&SomeValue>,
) -> SomeValue {
    if let (Some(start), Some(end)) = (s_start, s_end) {
        check_negative_slice(start, end, "rfind");
    }
    SomeValue::Integer(SomeInteger::default())
}

#[allow(dead_code)]
pub fn str_method_count(
    _ann: &RPythonAnnotator,
    _s_self: &SomeValue,
    _s_frag: &SomeValue,
    s_start: Option<&SomeValue>,
    s_end: Option<&SomeValue>,
) -> SomeValue {
    if let (Some(start), Some(end)) = (s_start, s_end) {
        check_negative_slice(start, end, "count");
    }
    SomeValue::Integer(SomeInteger::new(true, false))
}

fn str_method_strip_like(s_self: &SomeValue, s_chr: Option<&SomeValue>, opname: &str) -> SomeValue {
    // unaryop.py:633-644 — strip/lstrip/rstrip share the same shape.
    if s_chr.is_none() && matches!(s_self, SomeValue::UnicodeString(_)) {
        panic!("AnnotatorError: unicode.{opname}() with no arg is not RPython");
    }
    basestring_of(s_self, stringish_no_nul(s_self))
}

#[allow(dead_code)]
pub fn str_method_strip(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    s_chr: Option<&SomeValue>,
) -> SomeValue {
    str_method_strip_like(s_self, s_chr, "strip")
}

#[allow(dead_code)]
pub fn str_method_lstrip(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    s_chr: Option<&SomeValue>,
) -> SomeValue {
    str_method_strip_like(s_self, s_chr, "lstrip")
}

#[allow(dead_code)]
pub fn str_method_rstrip(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    s_chr: Option<&SomeValue>,
) -> SomeValue {
    str_method_strip_like(s_self, s_chr, "rstrip")
}

#[allow(dead_code)]
pub fn str_method_join(
    ann: &RPythonAnnotator,
    s_self: &SomeValue,
    s_list: &SomeValue,
) -> SomeValue {
    // unaryop.py:648-658.
    if s_none().contains(s_list) {
        return SomeValue::Impossible;
    }
    let SomeValue::List(s_list) = s_list else {
        panic!("AnnotatorError: string.join() argument must be a list");
    };
    let position = ann.bookkeeper.current_position_key();
    let s_item = s_list.listdef.read_item(position);
    if s_none().contains(&s_item) {
        // upstream unaryop.py:653-656:
        //     if isinstance(self, SomeUnicodeString):
        //         return immutablevalue(u"")
        //     return immutablevalue("")
        // The SomeString branch routes through bookkeeper.immutablevalue
        // so no_nul / const_box carry the same invariants as any other
        // empty-string constant.
        let s_empty = ann
            .bookkeeper
            .immutablevalue(&ConstValue::byte_str(""))
            .expect("immutablevalue(\"\") must succeed");
        return match s_self {
            SomeValue::UnicodeString(_) => {
                let SomeValue::String(s) = s_empty else {
                    unreachable!("immutablevalue(\"\") always returns SomeString");
                };
                let mut u = SomeUnicodeString::new(false, s.inner.no_nul);
                u.inner.base.const_box = s.inner.base.const_box;
                SomeValue::UnicodeString(u)
            }
            _ => s_empty,
        };
    }
    let no_nul = stringish_no_nul(s_self) && stringish_no_nul(&s_item);
    basestring_of(s_self, no_nul)
}

#[allow(dead_code)]
pub fn str_method_split(
    ann: &RPythonAnnotator,
    s_self: &SomeValue,
    s_patt: &SomeValue,
    s_max: Option<&SomeValue>,
) -> SomeValue {
    // unaryop.py:667-675.
    let max_value = s_max.and_then(const_int_of).unwrap_or(-1);
    let no_nul = if max_value == -1
        && const_str_of(s_patt)
            .map(|p| p.chars().count() == 1 && p.chars().next() == Some('\0'))
            .unwrap_or(false)
    {
        true
    } else {
        stringish_no_nul(s_self)
    };
    let s_item = basestring_of(s_self, no_nul);
    let s_list = ann
        .bookkeeper
        .newlist(&[s_item], None)
        .expect("bookkeeper.newlist failed");
    SomeValue::List(s_list)
}

#[allow(dead_code)]
pub fn str_method_rsplit(
    ann: &RPythonAnnotator,
    s_self: &SomeValue,
    _s_patt: &SomeValue,
    _s_max: Option<&SomeValue>,
) -> SomeValue {
    // unaryop.py:677-679.
    let s_item = basestring_of(s_self, stringish_no_nul(s_self));
    let s_list = ann
        .bookkeeper
        .newlist(&[s_item], None)
        .expect("bookkeeper.newlist failed");
    SomeValue::List(s_list)
}

#[allow(dead_code)]
pub fn str_method_upper(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    // unaryop.py:736-737 / 793-797 — SomeChar.upper returns self;
    // SomeString/SomeUnicodeString inherit basestringclass().
    match _s_self {
        SomeValue::Char(_) => _s_self.clone(),
        _ => basestring_of(_s_self, false),
    }
}

#[allow(dead_code)]
pub fn str_method_lower(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    match _s_self {
        SomeValue::Char(_) => _s_self.clone(),
        _ => basestring_of(_s_self, false),
    }
}

#[allow(dead_code)]
pub fn str_method_isdigit(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
pub fn str_method_isalpha(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
pub fn str_method_isalnum(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
pub fn str_method_replace(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    _s1: &SomeValue,
    s2: &SomeValue,
) -> SomeValue {
    // unaryop.py:681-682 — basestringclass(no_nul=self.no_nul and s2.no_nul).
    let self_no_nul = match s_self {
        SomeValue::String(s) => s.inner.no_nul,
        SomeValue::UnicodeString(s) => s.inner.no_nul,
        _ => false,
    };
    let s2_no_nul = match s2 {
        SomeValue::String(s) => s.inner.no_nul,
        SomeValue::UnicodeString(s) => s.inner.no_nul,
        SomeValue::Char(s) => s.inner.no_nul,
        SomeValue::UnicodeCodePoint(s) => s.inner.no_nul,
        _ => false,
    };
    basestring_of(s_self, self_no_nul && s2_no_nul)
}

#[allow(dead_code)]
pub fn str_method_splitlines(ann: &RPythonAnnotator, s_self: &SomeValue) -> SomeValue {
    // unaryop.py:744-748.
    let s_item = basestring_of(s_self, false);
    let s_list = ann
        .bookkeeper
        .newlist(&[s_item], None)
        .expect("bookkeeper.newlist failed");
    s_list.listdef.resize().expect("listdef.resize failed");
    SomeValue::List(s_list)
}

#[allow(dead_code)]
pub fn str_method_format(
    _ann: &RPythonAnnotator,
    _s_self: &SomeValue,
    _args: &[SomeValue],
) -> SomeValue {
    // unaryop.py:689-690 — always raises.
    panic!("AnnotatorError: Method format() is not RPython")
}

#[allow(dead_code)]
pub fn char_method_isspace(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
pub fn char_method_islower(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
pub fn char_method_isupper(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

/// RPython `SomeObject.find_method(self, name)` (unaryop.py:206-213).
pub fn find_method(s_self: &SomeValue, name: &str) -> Option<SomeBuiltinMethod> {
    let analyser_name = match s_self {
        SomeValue::List(_) => match name {
            "append" => "list_method_append",
            "extend" => "list_method_extend",
            "reverse" => "list_method_reverse",
            "insert" => "list_method_insert",
            "remove" => "list_method_remove",
            "pop" => "list_method_pop",
            "index" => "list_method_index",
            _ => return None,
        },
        SomeValue::Dict(_) => match name {
            "get" | "setdefault" => "dict_method_get",
            "copy" => "dict_method_copy",
            "update" => "dict_method_update",
            "__prepare_dict_update" => "dict_method_prepare_dict_update",
            "keys" => "dict_method_keys",
            "values" => "dict_method_values",
            "items" => "dict_method_items",
            "iterkeys" => "dict_method_iterkeys",
            "itervalues" => "dict_method_itervalues",
            "iteritems" => "dict_method_iteritems",
            "iterkeys_with_hash" => "dict_method_iterkeys_with_hash",
            "iteritems_with_hash" => "dict_method_iteritems_with_hash",
            "clear" => "dict_method_clear",
            "popitem" => "dict_method_popitem",
            "pop" => "dict_method_pop",
            "contains_with_hash" => "dict_method_contains_with_hash",
            "setitem_with_hash" => "dict_method_setitem_with_hash",
            "getitem_with_hash" => "dict_method_getitem_with_hash",
            "delitem_with_hash" => "dict_method_delitem_with_hash",
            "delitem_if_value_is" => "dict_method_delitem_if_value_is",
            "move_to_end" => "dict_method_move_to_end",
            _ => return None,
        },
        SomeValue::String(_) => match name {
            "startswith" => "str_method_startswith",
            "endswith" => "str_method_endswith",
            "find" => "str_method_find",
            "rfind" => "str_method_rfind",
            "count" => "str_method_count",
            "strip" => "str_method_strip",
            "lstrip" => "str_method_lstrip",
            "rstrip" => "str_method_rstrip",
            "join" => "str_method_join",
            "split" => "str_method_split",
            "rsplit" => "str_method_rsplit",
            "upper" => "str_method_upper",
            "lower" => "str_method_lower",
            "isdigit" => "str_method_isdigit",
            "isalpha" => "str_method_isalpha",
            "isalnum" => "str_method_isalnum",
            "replace" => "str_method_replace",
            "splitlines" => "str_method_splitlines",
            "format" => "str_method_format",
            _ => return None,
        },
        SomeValue::UnicodeString(_) => match name {
            "startswith" => "str_method_startswith",
            "endswith" => "str_method_endswith",
            "find" => "str_method_find",
            "rfind" => "str_method_rfind",
            "count" => "str_method_count",
            "strip" => "str_method_strip",
            "lstrip" => "str_method_lstrip",
            "rstrip" => "str_method_rstrip",
            "join" => "str_method_join",
            "split" => "str_method_split",
            "rsplit" => "str_method_rsplit",
            "upper" => "str_method_upper",
            "lower" => "str_method_lower",
            "isdigit" => "str_method_isdigit",
            "isalpha" => "str_method_isalpha",
            "isalnum" => "str_method_isalnum",
            "replace" => "str_method_replace",
            "format" => "str_method_format",
            _ => return None,
        },
        SomeValue::Char(_) => match name {
            "upper" => "str_method_upper",
            "lower" => "str_method_lower",
            "isdigit" => "str_method_isdigit",
            "isalpha" => "str_method_isalpha",
            "isalnum" => "str_method_isalnum",
            "isspace" => "char_method_isspace",
            "islower" => "char_method_islower",
            "isupper" => "char_method_isupper",
            _ => return None,
        },
        SomeValue::UnicodeCodePoint(_) => match name {
            "upper" => "str_method_upper",
            "lower" => "str_method_lower",
            "isdigit" => "str_method_isdigit",
            "isalpha" => "str_method_isalpha",
            "isalnum" => "str_method_isalnum",
            _ => return None,
        },
        _ => return None,
    };
    Some(SomeBuiltinMethod::new(analyser_name, s_self.clone(), name))
}

fn builtin_method_receiver_error(method: &SomeBuiltinMethod) -> AnnotatorError {
    AnnotatorError::new(format!(
        "SomeBuiltinMethod.call(): receiver/type mismatch for {}",
        method.methodname
    ))
}

fn builtin_method_arg_error(method: &SomeBuiltinMethod) -> AnnotatorError {
    AnnotatorError::new(format!(
        "SomeBuiltinMethod.call(): wrong argument shape for {}",
        method.methodname
    ))
}

/// Model upstream `SomeBuiltinMethod.call(self, args)` argument
/// binding (unaryop.py:961-967):
///
/// ```python
/// def call(self, args, implicit_init=False):
///     args_s, kwds = args.unpack()
///     # prefix keyword arguments with 's_'
///     kwds_s = {}
///     for key, s_value in kwds.items():
///         kwds_s['s_'+key] = s_value
///     return self.analyser(self.s_self, *args_s, **kwds_s)
/// ```
///
/// In RPython the final `self.analyser(self.s_self, *args_s, **kwds_s)`
/// relies on Python's native calling convention to pair positional
/// args + `s_`-prefixed kwargs against the `method_X(self, ...)`
/// signature, plus the method's default values for missing positions.
/// Rust has no reflection, so the port binds the arguments explicitly
/// through [`ArgumentsForTranslation::match_signature`] (the Rust
/// port of `argument.py:126-133`) with the analyser's formal
/// `argnames` and `defaults` threaded in from each dispatch arm.
/// The resulting `Vec<SomeValue>` matches positional order 1:1 with
/// the analyser's formals, letting the call sites destructure via
/// `let [..] = scope.as_slice()` without inspecting `kwds` again.
fn bind_builtin_method_args(
    args_s: Vec<Option<SomeValue>>,
    kwds: std::collections::HashMap<String, Option<SomeValue>>,
    argnames: &[&str],
    defaults: Option<Vec<SomeValue>>,
    analyser_name: &str,
) -> Result<Vec<SomeValue>, AnnotatorError> {
    let signature = crate::flowspace::argument::Signature::new(
        argnames.iter().map(|name| (*name).to_string()).collect(),
        None,
        None,
    );
    // upstream: `kwds_s['s_'+key] = s_value` — `s_value` may be None
    // (unbound caller arg).  The Option-aware threading propagates
    // through `match_signature`'s `scope_w` carrier, which itself is
    // `Vec<Option<SomeValue>>` so unbound slots reach the per-touch
    // unwrap below.  Pre-prefixed dict stays `Option<SomeValue>`-valued
    // to preserve None across the analyser boundary.
    let prefixed_kwds: std::collections::HashMap<String, Option<SomeValue>> = kwds
        .into_iter()
        .map(|(key, value)| (format!("s_{key}"), value))
        .collect();
    // upstream `unaryop.py:961-967` passes `args_s` straight through —
    // an unbound caller cell would AttributeError on first attribute
    // touch in the analyser body.  Thread `Vec<Option<SomeValue>>`
    // through `match_signature` (the binding layer is Option-aware),
    // then panic per-slot at the boundary so the failure surfaces with
    // the analyser name + slot index.
    let call_args =
        super::argument::ArgumentsForTranslation::new(args_s, Some(prefixed_kwds), None);
    let scope_w = call_args
        .match_signature(&signature, defaults.as_deref())
        .map_err(|err| {
            AnnotatorError::new(format!("SomeBuiltinMethod.call(): {}", err.getmsg()))
        })?;
    Ok(scope_w
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            c.unwrap_or_else(|| {
                panic!(
                    "{analyser_name}: positional arg {i} is unannotated \
                     (upstream `args_s[{i}].knowntype` AttributeError)",
                )
            })
        })
        .collect())
}

/// RPython `SomeBuiltinMethod.call(self, args, implicit_init=False)`
/// dispatch helper (unaryop.py:961-967).
pub fn call_builtin_method(
    ann: &RPythonAnnotator,
    method: &SomeBuiltinMethod,
    args: &super::argument::ArgumentsForTranslation,
) -> Result<SomeValue, AnnotatorError> {
    let (args_s, kwds) = args
        .unpack()
        .map_err(|err| AnnotatorError::new(err.getmsg()))?;
    // `unaryop.py:961-967 SomeBuiltinMethod.call` passes args_s
    // through `bookkeeper.consider_call_site` and into the analyser
    // body.  Optional-default methods (list.pop(s_index=None),
    // dict.pop(s_dfl=None), str.find(start=None, end=None)) accept
    // missing positional slots that bind to Python None via
    // signature defaults.  Thread `Vec<Option<SomeValue>>` through
    // bind_builtin_method_args so the per-slot panic only fires on
    // post-default-binding `None`, matching upstream's first
    // attribute-touch AttributeError surface.

    let result = match method.analyser_name.as_str() {
        "list_method_append" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_value"], None, &method.analyser_name)?;
            let [s_value] = scope.as_slice() else {
                unreachable!();
            };
            list_method_append(ann, s_self, s_value)
        }
        "list_method_extend" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_iterable"],
                None,
                &method.analyser_name,
            )?;
            let [s_iterable] = scope.as_slice() else {
                unreachable!();
            };
            list_method_extend(ann, s_self, s_iterable)
        }
        "list_method_reverse" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            list_method_reverse(ann, s_self)
        }
        "list_method_insert" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_index", "s_value"],
                None,
                &method.analyser_name,
            )?;
            let [s_index, s_value] = scope.as_slice() else {
                unreachable!();
            };
            list_method_insert(ann, s_self, s_index, s_value)
        }
        "list_method_remove" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_value"], None, &method.analyser_name)?;
            let [s_value] = scope.as_slice() else {
                unreachable!();
            };
            list_method_remove(ann, s_self, s_value)
        }
        "list_method_pop" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_index"],
                Some(vec![s_none()]),
                &method.analyser_name,
            )?;
            let [s_index] = scope.as_slice() else {
                unreachable!();
            };
            if matches!(s_index, SomeValue::None_(_)) {
                list_method_pop(ann, s_self, None)
            } else {
                list_method_pop(ann, s_self, Some(s_index))
            }
        }
        "list_method_index" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_value"], None, &method.analyser_name)?;
            let [s_value] = scope.as_slice() else {
                unreachable!();
            };
            list_method_index(ann, s_self, s_value)
        }
        "dict_method_get" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_dfl"],
                Some(vec![s_none()]),
                &method.analyser_name,
            )?;
            let [s_key, s_dfl] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_get(ann, s_self, s_key, s_dfl)
        }
        "dict_method_copy" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_copy(ann, s_self)
        }
        "dict_method_update" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_other"], None, &method.analyser_name)?;
            let [s_other] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_update(ann, s_self, s_other)
        }
        "dict_method_prepare_dict_update" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_num"], None, &method.analyser_name)?;
            let [s_num] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_prepare_dict_update(ann, s_self, s_num)
        }
        "dict_method_keys" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_keys(ann, s_self)
        }
        "dict_method_values" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_values(ann, s_self)
        }
        "dict_method_items" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_items(ann, s_self)
        }
        "dict_method_iterkeys" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_iterkeys(ann, s_self)
        }
        "dict_method_itervalues" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_itervalues(ann, s_self)
        }
        "dict_method_iteritems" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_iteritems(ann, s_self)
        }
        "dict_method_iterkeys_with_hash" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_iterkeys_with_hash(ann, s_self)
        }
        "dict_method_iteritems_with_hash" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_iteritems_with_hash(ann, s_self)
        }
        "dict_method_clear" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_clear(ann, s_self)
        }
        "dict_method_popitem" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_popitem(ann, s_self)
        }
        "dict_method_pop" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_dfl"],
                Some(vec![s_none()]),
                &method.analyser_name,
            )?;
            let [s_key, s_dfl] = scope.as_slice() else {
                unreachable!();
            };
            if matches!(s_dfl, SomeValue::None_(_)) {
                dict_method_pop(ann, s_self, s_key, None)
            } else {
                dict_method_pop(ann, s_self, s_key, Some(s_dfl))
            }
        }
        "dict_method_contains_with_hash" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_hash"],
                None,
                &method.analyser_name,
            )?;
            let [s_key, s_hash] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_contains_with_hash(ann, s_self, s_key, s_hash)
        }
        "dict_method_setitem_with_hash" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_hash", "s_value"],
                None,
                &method.analyser_name,
            )?;
            let [s_key, s_hash, s_value] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_setitem_with_hash(ann, s_self, s_key, s_hash, s_value)
        }
        "dict_method_getitem_with_hash" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_hash"],
                None,
                &method.analyser_name,
            )?;
            let [s_key, s_hash] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_getitem_with_hash(ann, s_self, s_key, s_hash)
        }
        "dict_method_delitem_with_hash" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_hash"],
                None,
                &method.analyser_name,
            )?;
            let [s_key, s_hash] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_delitem_with_hash(ann, s_self, s_key, s_hash)
        }
        "dict_method_delitem_if_value_is" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_value"],
                None,
                &method.analyser_name,
            )?;
            let [s_key, s_value] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_delitem_if_value_is(ann, s_self, s_key, s_value)
        }
        "dict_method_move_to_end" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_last"],
                None,
                &method.analyser_name,
            )?;
            let [s_key, s_last] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_move_to_end(ann, s_self, s_key, s_last)
        }
        "str_method_startswith" => {
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_frag"], None, &method.analyser_name)?;
            let [s_frag] = scope.as_slice() else {
                unreachable!();
            };
            str_method_startswith(ann, &method.s_self, s_frag)
        }
        "str_method_endswith" => {
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_frag"], None, &method.analyser_name)?;
            let [s_frag] = scope.as_slice() else {
                unreachable!();
            };
            str_method_endswith(ann, &method.s_self, s_frag)
        }
        "str_method_find" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_frag", "s_start", "s_end"],
                Some(vec![s_none(), s_none()]),
                &method.analyser_name,
            )?;
            let [s_frag, s_start, s_end] = scope.as_slice() else {
                unreachable!();
            };
            str_method_find(
                ann,
                &method.s_self,
                s_frag,
                if matches!(s_start, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_start)
                },
                if matches!(s_end, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_end)
                },
            )
        }
        "str_method_rfind" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_frag", "s_start", "s_end"],
                Some(vec![s_none(), s_none()]),
                &method.analyser_name,
            )?;
            let [s_frag, s_start, s_end] = scope.as_slice() else {
                unreachable!();
            };
            str_method_rfind(
                ann,
                &method.s_self,
                s_frag,
                if matches!(s_start, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_start)
                },
                if matches!(s_end, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_end)
                },
            )
        }
        "str_method_count" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_frag", "s_start", "s_end"],
                Some(vec![s_none(), s_none()]),
                &method.analyser_name,
            )?;
            let [s_frag, s_start, s_end] = scope.as_slice() else {
                unreachable!();
            };
            str_method_count(
                ann,
                &method.s_self,
                s_frag,
                if matches!(s_start, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_start)
                },
                if matches!(s_end, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_end)
                },
            )
        }
        "str_method_strip" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_chr"],
                Some(vec![s_none()]),
                &method.analyser_name,
            )?;
            let [s_chr] = scope.as_slice() else {
                unreachable!();
            };
            str_method_strip(
                ann,
                &method.s_self,
                if matches!(s_chr, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_chr)
                },
            )
        }
        "str_method_lstrip" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_chr"],
                Some(vec![s_none()]),
                &method.analyser_name,
            )?;
            let [s_chr] = scope.as_slice() else {
                unreachable!();
            };
            str_method_lstrip(
                ann,
                &method.s_self,
                if matches!(s_chr, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_chr)
                },
            )
        }
        "str_method_rstrip" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_chr"],
                Some(vec![s_none()]),
                &method.analyser_name,
            )?;
            let [s_chr] = scope.as_slice() else {
                unreachable!();
            };
            str_method_rstrip(
                ann,
                &method.s_self,
                if matches!(s_chr, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_chr)
                },
            )
        }
        "str_method_join" => {
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_list"], None, &method.analyser_name)?;
            let [s_list] = scope.as_slice() else {
                unreachable!();
            };
            str_method_join(ann, &method.s_self, s_list)
        }
        "str_method_split" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_patt", "s_max"],
                Some(vec![s_const_int(-1)]),
                &method.analyser_name,
            )?;
            let [s_patt, s_max] = scope.as_slice() else {
                unreachable!();
            };
            str_method_split(ann, &method.s_self, s_patt, Some(s_max))
        }
        "str_method_rsplit" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_patt", "s_max"],
                Some(vec![s_const_int(-1)]),
                &method.analyser_name,
            )?;
            let [s_patt, s_max] = scope.as_slice() else {
                unreachable!();
            };
            str_method_rsplit(ann, &method.s_self, s_patt, Some(s_max))
        }
        "str_method_upper" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            str_method_upper(ann, &method.s_self)
        }
        "str_method_lower" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            str_method_lower(ann, &method.s_self)
        }
        "str_method_isdigit" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            str_method_isdigit(ann, &method.s_self)
        }
        "str_method_isalpha" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            str_method_isalpha(ann, &method.s_self)
        }
        "str_method_isalnum" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            str_method_isalnum(ann, &method.s_self)
        }
        "str_method_replace" => {
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s1", "s2"], None, &method.analyser_name)?;
            let [s1, s2] = scope.as_slice() else {
                unreachable!();
            };
            str_method_replace(ann, &method.s_self, s1, s2)
        }
        "str_method_splitlines" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_keep_newlines"],
                Some(vec![s_none()]),
                &method.analyser_name,
            )?;
            let [_s_keep_newlines] = scope.as_slice() else {
                unreachable!();
            };
            str_method_splitlines(ann, &method.s_self)
        }
        "str_method_format" => {
            if !kwds.is_empty() {
                return Err(builtin_method_arg_error(method));
            }
            // upstream `str_method_format` always raises — the args are
            // unused, so the Option layer can be flattened with
            // Impossible at this single call site.
            let args_concrete: Vec<SomeValue> = args_s
                .iter()
                .map(|s| s.clone().unwrap_or_else(super::model::s_impossible_value))
                .collect();
            str_method_format(ann, &method.s_self, &args_concrete)
        }
        "char_method_isspace" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            char_method_isspace(ann, &method.s_self)
        }
        "char_method_islower" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            char_method_islower(ann, &method.s_self)
        }
        "char_method_isupper" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            char_method_isupper(ann, &method.s_self)
        }
        _ => {
            return Err(AnnotatorError::new(format!(
                "SomeBuiltinMethod.call(): unknown analyser {}",
                method.analyser_name
            )));
        }
    };
    Ok(result)
}

// =====================================================================
// unaryop.py:806-830 — class __extend__(SomeIterator)
// =====================================================================

fn init_someiterator_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:808-810 — iter(self): return self. can_only_throw=[].
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::Iterator,
        Specialization {
            apply: Box::new(|ann, hl| ann.annotation(&hl.args[0]).expect("iterator.iter: unbound")),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:818-830 — next(self):
    //   position = getbookkeeper().position_key
    //   if s_None.contains(self.s_container): return s_ImpossibleValue
    //   if self.variant and self.variant[0] == "enumerate":
    //       s_item = self.s_container.getanyitem(position)
    //       return SomeTuple((SomeInteger(nonneg=True), s_item))
    //   variant = self.variant
    //   if variant == ("reversed",): variant = ()
    //   return self.s_container.getanyitem(position, *variant)
    register(
        reg,
        OpKind::Next,
        SomeValueTag::Iterator,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s = ann.annotation(&hl.args[0]).expect("iterator.next: unbound");
                let it = match s {
                    SomeValue::Iterator(i) => i,
                    _ => panic!("iterator.next: arg 0 not SomeIterator"),
                };
                someiterator_next(ann, &it)
            }),
            can_only_throw: CanOnlyThrow::Callable(Box::new(|args_s| {
                // upstream `_can_only_throw` (unaryop.py:812-816):
                //   can_throw = [StopIteration]
                //   if isinstance(self.s_container, SomeDict):
                //       can_throw.append(RuntimeError)
                //   return can_throw
                let mut throws = vec![BuiltinException::StopIteration];
                if let Some(SomeValue::Iterator(i)) = args_s.first() {
                    if matches!(&*i.s_container, SomeValue::Dict(_)) {
                        throws.push(BuiltinException::RuntimeError);
                    }
                }
                Some(throws)
            })),
        },
    );
}

/// RPython `SomeIterator.next(self)` (unaryop.py:818-828). Extracted
/// into a free function so `list_method_extend` (unaryop.py:363-369)
/// can call `s_iterable.iter().next()` directly without going through
/// the HLOperation dispatch framework — upstream also calls it as a
/// plain method invocation on a SomeIterator value.
fn someiterator_next(ann: &RPythonAnnotator, it: &SomeIterator) -> SomeValue {
    let position = ann.bookkeeper.current_position_key();
    // upstream: `if s_None.contains(self.s_container): return s_ImpossibleValue`.
    if matches!(&*it.s_container, SomeValue::None_(_)) {
        return s_impossible_value();
    }
    // upstream: `if self.variant and self.variant[0] == "enumerate": ...`.
    if it.variant.first().map(String::as_str) == Some("enumerate") {
        let s_item = container_getanyitem(&it.s_container, None, position.clone());
        return SomeValue::Tuple(SomeTuple::new(vec![
            SomeValue::Integer(SomeInteger::new(true, false)),
            s_item,
        ]));
    }
    // upstream: `if variant == ("reversed",): variant = ()`.
    let variant: Option<&str> = if it.variant.len() == 1 && it.variant[0] == "reversed" {
        None
    } else {
        it.variant.first().map(String::as_str)
    };
    container_getanyitem(&it.s_container, variant, position)
}

/// RPython `container.getanyitem(position, *variant)` dispatch —
/// pulled out into a free function so [`SomeIterator.next`] can call
/// it once for the enumerate branch (with variant forced to `()`) and
/// once for the default branch (variant = `it.variant`).
///
/// Upstream definitions:
/// * SomeTuple (unaryop.py:341-342): `unionof(*self.items)`.
/// * SomeList (unaryop.py:402-403):  `self.listdef.read_item(position)`.
/// * SomeDict (unaryop.py:480-500):  variant-dispatched.
/// * SomeStringOrUnicode (unaryop.py:664-665): `self.basecharclass()`.
pub fn container_getanyitem(
    s_container: &SomeValue,
    variant: Option<&str>,
    position: Option<super::bookkeeper::PositionKey>,
) -> SomeValue {
    use super::super::flowspace::argument::Signature;
    let _ = Signature::new; // silence dead-imports on cold paths
    match s_container {
        SomeValue::Tuple(t) => {
            if t.items.is_empty() {
                s_impossible_value()
            } else {
                super::model::unionof(t.items.iter()).expect("getanyitem(tuple): unionof failed")
            }
        }
        SomeValue::List(l) => l.listdef.read_item(position),
        SomeValue::Dict(d) => {
            // unaryop.py:480-500 — per-variant dispatch.
            match variant.unwrap_or("keys") {
                "keys" => d.dictdef.read_key(position),
                "values" => d.dictdef.read_value(position),
                "items" => {
                    let s_key = d.dictdef.read_key(position.clone());
                    let s_value = d.dictdef.read_value(position);
                    if matches!(s_key, SomeValue::Impossible)
                        || matches!(s_value, SomeValue::Impossible)
                    {
                        s_impossible_value()
                    } else {
                        SomeValue::Tuple(SomeTuple::new(vec![s_key, s_value]))
                    }
                }
                "items_with_hash" => {
                    let s_key = d.dictdef.read_key(position.clone());
                    let s_value = d.dictdef.read_value(position);
                    if matches!(s_key, SomeValue::Impossible)
                        || matches!(s_value, SomeValue::Impossible)
                    {
                        s_impossible_value()
                    } else {
                        SomeValue::Tuple(SomeTuple::new(vec![
                            s_key,
                            s_value,
                            SomeValue::Integer(SomeInteger::default()),
                        ]))
                    }
                }
                "keys_with_hash" => {
                    let s_key = d.dictdef.read_key(position);
                    if matches!(s_key, SomeValue::Impossible) {
                        s_impossible_value()
                    } else {
                        SomeValue::Tuple(SomeTuple::new(vec![
                            s_key,
                            SomeValue::Integer(SomeInteger::default()),
                        ]))
                    }
                }
                other => panic!("getanyitem(dict): unknown variant {other:?}"),
            }
        }
        // unaryop.py:664-665 — `SomeStringOrUnicode.getanyitem` returns
        // `self.basecharclass()`. model.py:326-329 assigns
        //   SomeString.basecharclass = SomeChar
        //   SomeUnicodeString.basecharclass = SomeUnicodeCodePoint
        // and the class is called with no arguments — so the char's
        // `no_nul` defaults to False regardless of the string's.
        SomeValue::String(_) => SomeValue::Char(super::model::SomeChar::new(false)),
        SomeValue::UnicodeString(_) => {
            SomeValue::UnicodeCodePoint(super::model::SomeUnicodeCodePoint::new(false))
        }
        other => panic!("getanyitem: unsupported container {other:?}"),
    }
}

// =====================================================================
// unaryop.py:970-998 — class __extend__(SomePBC)
// =====================================================================

fn init_somepbc_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:982-983 — setattr raises.
    register(
        reg,
        OpKind::SetAttr,
        SomeValueTag::PBC,
        Specialization {
            apply: Box::new(|_ann, _hl| {
                panic!("AnnotatorError: Cannot modify attribute of a pre-built constant")
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:997-998 — len raises.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::PBC,
        Specialization {
            apply: Box::new(|_ann, _hl| panic!("AnnotatorError: Cannot call len on a pbc")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// lltype.py:1566-1570 — SomePtr.bool
// =====================================================================

fn init_someptr_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    fn with_ptr_like<R>(
        s: &SomeValue,
        f: impl FnOnce(&crate::annotator::model::SomePtr) -> R,
        g: impl FnOnce(&crate::annotator::model::SomeInteriorPtr) -> R,
    ) -> R {
        match s {
            SomeValue::Ptr(ptr) => f(ptr),
            SomeValue::InteriorPtr(ptr) => g(ptr),
            other => unreachable!("Ptr-tag dispatch must carry ptr-like value, got {other:?}"),
        }
    }
    register(
        reg,
        OpKind::GetAttr,
        SomeValueTag::Ptr,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s = ann
                    .annotation(&hl.args[0])
                    .expect("ptr.getattr: self unbound");
                let s_attr = ann
                    .annotation(&hl.args[1])
                    .expect("ptr.getattr: attr unbound");
                with_ptr_like(
                    &s,
                    |ptr| ptr.getattr(&s_attr).expect("ptr.getattr must succeed"),
                    |ptr| {
                        ptr.getattr(&s_attr)
                            .expect("interiorptr.getattr must succeed")
                    },
                )
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Ptr,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s = ann.annotation(&hl.args[0]).expect("ptr.len: unbound");
                with_ptr_like(
                    &s,
                    |ptr| ptr.len().expect("ptr.len must succeed"),
                    |ptr| ptr.len().expect("interiorptr.len must succeed"),
                )
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    register(
        reg,
        OpKind::Bool,
        SomeValueTag::Ptr,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s = ann.annotation(&hl.args[0]).expect("ptr.bool: unbound");
                with_ptr_like(&s, |ptr| ptr.bool(), |ptr| ptr.bool())
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    register(
        reg,
        OpKind::SetAttr,
        SomeValueTag::Ptr,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s = ann
                    .annotation(&hl.args[0])
                    .expect("ptr.setattr: self unbound");
                let s_attr = ann
                    .annotation(&hl.args[1])
                    .expect("ptr.setattr: attr unbound");
                let s_value = ann
                    .annotation(&hl.args[2])
                    .expect("ptr.setattr: value unbound");
                with_ptr_like(
                    &s,
                    |ptr| {
                        ptr.setattr(&s_attr, &s_value)
                            .expect("ptr.setattr must succeed")
                    },
                    |ptr| {
                        ptr.setattr(&s_attr, &s_value)
                            .expect("interiorptr.setattr must succeed")
                    },
                )
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:1000-1021 — class __extend__(SomeNone)
// =====================================================================

fn init_somenone_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:1004-1006 — getattr: s_ImpossibleValue, can_only_throw=[].
    register(
        reg,
        OpKind::GetAttr,
        SomeValueTag::None_,
        Specialization {
            apply: Box::new(|_ann, _hl| s_impossible_value()),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:1008-1009 — setattr: return None (void → SomeImpossible).
    register(
        reg,
        OpKind::SetAttr,
        SomeValueTag::None_,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Impossible),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:1011-1012 — call / simple_call / call_args: s_ImpossibleValue.
    for op in &[OpKind::SimpleCall, OpKind::CallArgs] {
        register(
            reg,
            *op,
            SomeValueTag::None_,
            Specialization {
                apply: Box::new(|_ann, _hl| s_impossible_value()),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
    // unaryop.py:1017-1021 — len(self): SomeImpossibleValue.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::None_,
        Specialization {
            apply: Box::new(|_ann, _hl| s_impossible_value()),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:1032-1037 — class __extend__(SomeWeakRef)
// =====================================================================

fn init_someweakref_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:1033-1037 — simple_call(self):
    //   if classdef is None: return s_None (dead weakref)
    //   else: SomeInstance(classdef, can_be_None=True)
    register(
        reg,
        OpKind::SimpleCall,
        SomeValueTag::WeakRef,
        Specialization {
            apply: Box::new(|ann, hl| {
                let wr = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::WeakRef(w)) => w,
                    _ => panic!("weakref.simple_call: arg 0 not SomeWeakRef"),
                };
                match &wr.classdef {
                    None => super::model::s_none(),
                    Some(cd) => SomeValue::Instance(super::model::SomeInstance::new(
                        Some(Rc::clone(cd)),
                        true,
                        std::collections::BTreeMap::new(),
                    )),
                }
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// Suppress unused warnings for items reserved for the follow-up
// commits in the same family (register_single, SomeUnicodeString,
// SomeBool const-propagation paths, Rc).
#[allow(dead_code)]
fn _keep_imports_live() {
    let _ = register_single;
    let _ = SomeUnicodeString::default;
    let _: Option<Rc<()>> = None;
}

// =====================================================================
// unaryop.py:120-139 — @op.call_args.register_transform(SomeObject)
// =====================================================================
//
// Upstream rewrites `op.call_args(v_func, v_shape, *data_v)` when the
// vararg tail is a `SomeTuple`, unpacking the tuple elements to turn
// the op back into `op.simple_call(...)` (or `op.call_args(...)` if
// keywords remain). The Rust port requires `CallShape` decoding from
// the `ConstValue::Tuple` payload on `v_shape` (mirrors
// `CallSpec.fromshape`).

fn init_object_call_args_transform(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Transformation>,
    >,
) {
    use super::super::flowspace::argument::CallSpec;

    // unaryop.py:120-139 — @op.call_args.register_transform(SomeObject).
    // ```python
    // def transform_varargs(annotator, v_func, v_shape, *data_v):
    //     callspec = CallSpec.fromshape(v_shape.value, list(data_v))
    //     v_vararg = callspec.w_stararg
    //     if callspec.w_stararg:
    //         s_vararg = annotator.annotation(callspec.w_stararg)
    //         if not isinstance(s_vararg, SomeTuple):
    //             raise AnnotatorError(...)
    //         n_items = len(s_vararg.items)
    //         ops = [op.getitem(v_vararg, const(i)) for i in range(n_items)]
    //         new_args = callspec.arguments_w + [hlop.result for hlop in ops]
    //         if callspec.keywords:
    //             newspec = CallSpec(new_args, callspec.keywords)
    //             shape, data_v = newspec.flatten()
    //             call_op = op.call_args(v_func, const(shape), *data_v)
    //         else:
    //             call_op = op.simple_call(v_func, *new_args)
    //         ops.append(call_op)
    //         return ops
    // ```
    register_transform(
        reg,
        OpKind::CallArgs,
        SomeValueTag::Object,
        Box::new(|ann, args| {
            // args = [v_func, v_shape, *data_v]
            if args.len() < 2 {
                return None;
            }
            let v_func = args[0].clone();
            let v_shape = &args[1];
            // upstream: `CallSpec.fromshape(v_shape.value, list(data_v))`.
            let shape_const = match v_shape {
                Hlvalue::Constant(c) => &c.value,
                _ => return None,
            };
            let shape = decode_call_shape(shape_const)?;
            let data_v: Vec<Hlvalue> = args[2..].to_vec();
            let callspec = CallSpec::fromshape(&shape, data_v);
            // upstream: `v_vararg = callspec.w_stararg; if callspec.w_stararg: ...`.
            let Some(v_vararg) = callspec.w_stararg.clone() else {
                return None;
            };
            // upstream: `s_vararg = annotator.annotation(v_vararg)`.
            let s_vararg = ann.annotation(&v_vararg)?;
            // upstream: `if not isinstance(s_vararg, SomeTuple): raise AnnotatorError`.
            let super::model::SomeValue::Tuple(ref tup) = s_vararg else {
                panic!("AnnotatorError: Calls like f(..., *arg) require 'arg' to be a tuple");
            };
            let n_items = tup.items.len();
            // upstream: `ops = [op.getitem(v_vararg, const(i)) for i in range(n_items)]`.
            let mut ops: Vec<HLOperation> = Vec::with_capacity(n_items + 1);
            let mut unpacked: Vec<Hlvalue> = Vec::with_capacity(n_items);
            for i in 0..n_items {
                let getitem = mk_hlop(
                    OpKind::GetItem,
                    vec![
                        v_vararg.clone(),
                        Hlvalue::Constant(Constant::new(ConstValue::Int(i as i64))),
                    ],
                );
                unpacked.push(Hlvalue::Variable(getitem.result.clone()));
                ops.push(getitem);
            }
            // upstream: `new_args = callspec.arguments_w + [hlop.result for ...]`.
            let mut new_args: Vec<Hlvalue> = callspec.arguments_w.clone();
            new_args.extend(unpacked.into_iter());
            // upstream: `if callspec.keywords: newspec = CallSpec(new_args, callspec.keywords);
            //            shape, data_v = newspec.flatten(); op.call_args(v_func, const(shape), *data_v)`
            //           else: `op.simple_call(v_func, *new_args)`.
            let call_op = if !callspec.keywords.is_empty() {
                let newspec = CallSpec::new(new_args, Some(callspec.keywords), None);
                let (new_shape, new_data_v) = newspec.flatten();
                let mut call_args = vec![
                    v_func,
                    Hlvalue::Constant(Constant::new(encode_call_shape(&new_shape))),
                ];
                call_args.extend(new_data_v.into_iter());
                mk_hlop(OpKind::CallArgs, call_args)
            } else {
                let mut simple_args = vec![v_func];
                simple_args.extend(new_args.into_iter());
                mk_hlop(OpKind::SimpleCall, simple_args)
            };
            ops.push(call_op);
            Some(ops)
        }),
    );
}

/// Decode the `CallShape` triple encoded in a `call_args` shape
/// constant (see `flowcontext::build_call_shape_constant` for the
/// encoder — ConstValue::Tuple([Int(cnt), Tuple([Str(key), …]), Bool(star)])).
pub(crate) fn decode_call_shape(
    cv: &ConstValue,
) -> Option<super::super::flowspace::argument::CallShape> {
    let items = match cv {
        ConstValue::Tuple(items) if items.len() == 3 => items,
        _ => return None,
    };
    let shape_cnt = match &items[0] {
        ConstValue::Int(n) => *n as usize,
        _ => return None,
    };
    let shape_keys: Vec<String> = match &items[1] {
        ConstValue::Tuple(keys) => keys
            .iter()
            .map(|k| match k {
                value => value.as_text().map(str::to_owned),
            })
            .collect::<Option<Vec<_>>>()?,
        _ => return None,
    };
    let shape_star = match &items[2] {
        ConstValue::Bool(b) => *b,
        _ => return None,
    };
    Some(super::super::flowspace::argument::CallShape {
        shape_cnt,
        shape_keys,
        shape_star,
    })
}

/// Encode a `CallShape` as the `ConstValue::Tuple` payload expected by
/// `call_args` — mirrors `flowcontext::build_call_shape_constant`.
fn encode_call_shape(shape: &super::super::flowspace::argument::CallShape) -> ConstValue {
    ConstValue::Tuple(vec![
        ConstValue::Int(shape.shape_cnt as i64),
        ConstValue::Tuple(shape.shape_keys.iter().map(ConstValue::byte_str).collect()),
        ConstValue::Bool(shape.shape_star),
    ])
}

// =====================================================================
// unaryop.py:867-892 — @op.{len,iter,next,getslice,setslice}.register_transform(SomeInstance)
// =====================================================================

/// Helper — build a new [`HLOperation`] with a fresh result variable,
/// matching upstream `HLOperation.__init__` (operation.py:73-75).
fn mk_hlop(kind: OpKind, args: Vec<Hlvalue>) -> HLOperation {
    HLOperation::new(kind, args)
}

/// RPython `_find_property_meth(s_obj, attr, meth)` (unaryop.py:895-906).
///
/// Walks `s_obj.classdef.getmro()` and collects the `property` descriptor's
/// `fget` / `fset` / `fdel` method reference for `attr`. Returns `None`
/// when the attribute appears in any classdict entry that isn't a
/// `property` (mirrors upstream's `return` without a result).
///
/// `meth` selects which property slot to read — "fget" / "fset" / "fdel" —
/// matching upstream's `getattr(obj.value, meth)`.
fn find_property_meth(
    s_obj: &super::model::SomeInstance,
    attr: &str,
    meth: &str,
) -> Option<Vec<Option<super::super::flowspace::model::HostObject>>> {
    use super::super::flowspace::model::ConstValue;
    let classdef = s_obj.classdef.as_ref()?;
    let mro = super::classdesc::ClassDef::getmro(classdef);
    let mut result: Vec<Option<super::super::flowspace::model::HostObject>> = Vec::new();
    for clsdef in &mro {
        let classdesc = clsdef.borrow().classdesc.clone();
        let entry = classdesc.borrow().classdict.get(attr).cloned();
        let Some(entry) = entry else { continue };
        match entry {
            super::classdesc::ClassDictEntry::Constant(c) => {
                // upstream unaryop.py:902-904:
                //   if (not isinstance(obj, Constant) or
                //           not isinstance(obj.value, property)):
                //       return
                //   result.append(getattr(obj.value, meth))
                let ConstValue::HostObject(host) = &c.value else {
                    return None;
                };
                if !host.is_property() {
                    return None;
                }
                let slot = match meth {
                    "fget" => host.property_fget().cloned(),
                    "fset" => host.property_fset().cloned(),
                    "fdel" => host.property_fdel().cloned(),
                    _ => return None,
                };
                result.push(slot);
            }
            // Non-Constant classdict entries (FunctionDesc etc.) never
            // stand in for a property; upstream's isinstance-check falls
            // through to `return`.
            super::classdesc::ClassDictEntry::Desc(_) => return None,
        }
    }
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

fn init_instance_single_transform(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Transformation>,
    >,
) {
    // unaryop.py:867-870 — len(v_arg) -> [getattr(v_arg, '__len__'), simple_call(getattr.result)]
    register_transform(
        reg,
        OpKind::Len,
        SomeValueTag::Instance,
        Box::new(|_ann, args| {
            let v_arg = args[0].clone();
            let get_len = mk_hlop(
                OpKind::GetAttr,
                vec![
                    v_arg,
                    Hlvalue::Constant(Constant::new(ConstValue::byte_str("__len__"))),
                ],
            );
            let getattr_result = Hlvalue::Variable(get_len.result.clone());
            let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result]);
            Some(vec![get_len, call])
        }),
    );
    // unaryop.py:872-875 — iter
    register_transform(
        reg,
        OpKind::Iter,
        SomeValueTag::Instance,
        Box::new(|_ann, args| {
            let v_arg = args[0].clone();
            let get_iter = mk_hlop(
                OpKind::GetAttr,
                vec![
                    v_arg,
                    Hlvalue::Constant(Constant::new(ConstValue::byte_str("__iter__"))),
                ],
            );
            let getattr_result = Hlvalue::Variable(get_iter.result.clone());
            let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result]);
            Some(vec![get_iter, call])
        }),
    );
    // unaryop.py:877-880 — next
    register_transform(
        reg,
        OpKind::Next,
        SomeValueTag::Instance,
        Box::new(|_ann, args| {
            let v_arg = args[0].clone();
            let get_next = mk_hlop(
                OpKind::GetAttr,
                vec![
                    v_arg,
                    Hlvalue::Constant(Constant::new(ConstValue::byte_str("next"))),
                ],
            );
            let getattr_result = Hlvalue::Variable(get_next.result.clone());
            let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result]);
            Some(vec![get_next, call])
        }),
    );
    // unaryop.py:882-885 — getslice
    register_transform(
        reg,
        OpKind::GetSlice,
        SomeValueTag::Instance,
        Box::new(|_ann, args| {
            let v_obj = args[0].clone();
            let v_start = args[1].clone();
            let v_stop = args[2].clone();
            let get_getslice = mk_hlop(
                OpKind::GetAttr,
                vec![
                    v_obj,
                    Hlvalue::Constant(Constant::new(ConstValue::byte_str("__getslice__"))),
                ],
            );
            let getattr_result = Hlvalue::Variable(get_getslice.result.clone());
            let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result, v_start, v_stop]);
            Some(vec![get_getslice, call])
        }),
    );
    // unaryop.py:888-892 — setslice
    register_transform(
        reg,
        OpKind::SetSlice,
        SomeValueTag::Instance,
        Box::new(|_ann, args| {
            let v_obj = args[0].clone();
            let v_start = args[1].clone();
            let v_stop = args[2].clone();
            let v_iterable = args[3].clone();
            let get_setslice = mk_hlop(
                OpKind::GetAttr,
                vec![
                    v_obj,
                    Hlvalue::Constant(Constant::new(ConstValue::byte_str("__setslice__"))),
                ],
            );
            let getattr_result = Hlvalue::Variable(get_setslice.result.clone());
            let call = mk_hlop(
                OpKind::SimpleCall,
                vec![getattr_result, v_start, v_stop, v_iterable],
            );
            Some(vec![get_setslice, call])
        }),
    );
    // binaryop.py:744-747 — contains. `Contains` is Dispatch::Single
    // upstream; the Rust dispatch table agrees (flowspace/operation.rs
    // OpKind::Contains => Dispatch::Single). Transform body is the same
    // getattr+simple_call pair; the second arg is `v_idx`.
    register_transform(
        reg,
        OpKind::Contains,
        SomeValueTag::Instance,
        Box::new(|_ann, args| {
            let v_ins = args[0].clone();
            let v_idx = args[1].clone();
            let get_contains = mk_hlop(
                OpKind::GetAttr,
                vec![
                    v_ins,
                    Hlvalue::Constant(Constant::new(ConstValue::byte_str("__contains__"))),
                ],
            );
            let getattr_result = Hlvalue::Variable(get_contains.result.clone());
            let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result, v_idx]);
            Some(vec![get_contains, call])
        }),
    );
}

// =====================================================================
// unaryop.py:909-936 — @op.{getattr,setattr}.register_transform(SomeInstance)
// =====================================================================

/// RPython `getattr_SomeInstance` / `setattr_SomeInstance`
/// (unaryop.py:909-936) — property descriptor dispatch.
///
/// Upstream rewrites `getattr(v_obj, 'prop')` into
/// `[getattr(v_obj, 'prop__getter__'), simple_call(getter.result)]`
/// when every classdef in the MRO has `prop` bound as a `property`
/// descriptor, and raises `AnnotatorError` when none of them do (the
/// "mixed property/non-property" case falls through with no rewrite).
/// `setattr` is the setter analogue — three-arg `simple_call(setter,
/// v_value)` instead.
fn init_instance_attr_transform(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Transformation>,
    >,
) {
    // unaryop.py:909-921 — getattr
    register_transform(
        reg,
        OpKind::GetAttr,
        SomeValueTag::Instance,
        Box::new(|ann, args| {
            let v_obj = args[0].clone();
            let v_attr = &args[1];
            // upstream: `s_attr = annotator.annotation(v_attr)`.
            let s_attr = ann.annotation(v_attr)?;
            // upstream: `if not s_attr.is_constant() or not isinstance(s_attr.const, str): return`.
            // `isinstance(x, str)` is Python 2 bytes-only — narrow with
            // [`ConstValue::as_pystr`] to keep parity with the upstream
            // short-circuit on unicode constants.
            let attr = match s_attr.const_() {
                Some(value) => value.as_pystr().map(str::to_owned)?,
                _ => return None,
            };
            // upstream: `getters = _find_property_meth(annotator.annotation(v_obj), attr, 'fget')`.
            let s_obj = ann.annotation(&v_obj)?;
            let super::model::SomeValue::Instance(ref s_inst) = s_obj else {
                return None;
            };
            let getters = find_property_meth(s_inst, &attr, "fget")?;
            // upstream: `if all(getters): … elif not any(getters): raise`.
            if getters.iter().all(|g| g.is_some()) {
                let getter_name = format!("{attr}__getter__");
                let get_getter = mk_hlop(
                    OpKind::GetAttr,
                    vec![
                        v_obj,
                        Hlvalue::Constant(Constant::new(ConstValue::byte_str(getter_name))),
                    ],
                );
                let getter_result = Hlvalue::Variable(get_getter.result.clone());
                let call = mk_hlop(OpKind::SimpleCall, vec![getter_result]);
                Some(vec![get_getter, call])
            } else if !getters.iter().any(|g| g.is_some()) {
                panic!("AnnotatorError: Attribute {attr:?} is unreadable");
            } else {
                None
            }
        }),
    );
    // unaryop.py:924-936 — setattr_SomeInstance.
    //
    //   if not s_attr.is_constant() or not isinstance(s_attr.const, str): return
    //
    // Mirror getter's `isinstance(value, str)` Python 2 bytes-only
    // gate via [`ConstValue::as_pystr`] so a unicode literal short-
    // circuits the same way it does on the getattr side, instead of
    // dispatching against `_find_property_meth(..., 'fset')`.
    register_transform(
        reg,
        OpKind::SetAttr,
        SomeValueTag::Instance,
        Box::new(|ann, args| {
            let v_obj = args[0].clone();
            let v_attr = &args[1];
            let v_value = args[2].clone();
            let s_attr = ann.annotation(v_attr)?;
            let attr = match s_attr.const_() {
                Some(value) => value.as_pystr().map(str::to_owned)?,
                _ => return None,
            };
            let s_obj = ann.annotation(&v_obj)?;
            let super::model::SomeValue::Instance(ref s_inst) = s_obj else {
                return None;
            };
            let setters = find_property_meth(s_inst, &attr, "fset")?;
            if setters.iter().all(|s| s.is_some()) {
                let setter_name = format!("{attr}__setter__");
                let get_setter = mk_hlop(
                    OpKind::GetAttr,
                    vec![
                        v_obj,
                        Hlvalue::Constant(Constant::new(ConstValue::byte_str(setter_name))),
                    ],
                );
                let setter_result = Hlvalue::Variable(get_setter.result.clone());
                let call = mk_hlop(OpKind::SimpleCall, vec![setter_result, v_value]);
                Some(vec![get_setter, call])
            } else if !setters.iter().any(|s| s.is_some()) {
                panic!("AnnotatorError: Attribute {attr:?} is unwritable");
            } else {
                None
            }
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::super::super::flowspace::model::{ConstValue, Constant, Hlvalue, Variable};
    use super::*;

    fn mk_ann() -> Rc<RPythonAnnotator> {
        RPythonAnnotator::new(None, None, None, false)
    }

    fn hl1(op: OpKind, sv: SomeValue) -> (HLOperation, Rc<RPythonAnnotator>) {
        let ann = mk_ann();
        let mut v = Variable::named("x");
        ann.setbinding(&mut v, sv);
        let hl = HLOperation::new(op, vec![Hlvalue::Variable(v)]);
        (hl, ann)
    }

    #[test]
    fn unary_operations_list_excludes_contains() {
        assert!(!UNARY_OPERATIONS.contains(&OpKind::Contains));
        assert!(UNARY_OPERATIONS.contains(&OpKind::Len));
        assert!(UNARY_OPERATIONS.contains(&OpKind::Bool));
    }

    #[test]
    fn consider_getattr_prefers_find_method_on_lists() {
        let ann = mk_ann();
        let mut v = Variable::named("lst");
        ann.setbinding(
            &mut v,
            SomeValue::List(super::super::model::SomeList::new(
                super::super::listdef::ListDef::new(
                    Some(ann.bookkeeper.clone()),
                    SomeValue::Integer(SomeInteger::default()),
                    false,
                    false,
                ),
            )),
        );
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![
                Hlvalue::Variable(v),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("append"))),
            ],
        );

        let result = hl.consider(&ann).expect("getattr must succeed");

        let SomeValue::BuiltinMethod(method) = result else {
            panic!("expected SomeBuiltinMethod");
        };
        assert_eq!(method.analyser_name, "list_method_append");
        assert_eq!(method.methodname, "append");
    }

    #[test]
    fn transform_len_someinstance_rewrites_to_getattr_plus_simple_call() {
        // unaryop.py:867-870 — `op.len(v_ins)` → [getattr(v_ins, '__len__'), simple_call(r)].
        use super::super::classdesc::ClassDef;
        use super::super::model::SomeInstance;

        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("pkg.X", None);
        let mut v = Variable::named("inst");
        ann.setbinding(
            &mut v,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let hl = HLOperation::new(OpKind::Len, vec![Hlvalue::Variable(v)]);
        let new_ops = hl
            .transform(&ann)
            .expect("Len(SomeInstance) must register_transform");
        assert_eq!(new_ops.len(), 2);
        assert!(matches!(new_ops[0].kind, OpKind::GetAttr));
        assert!(matches!(new_ops[1].kind, OpKind::SimpleCall));
    }

    #[test]
    fn transform_call_args_unpacks_tuple_stararg_to_simple_call() {
        // unaryop.py:120-139 — `f(*tup)` where tup has 3 items and no
        // keywords should rewrite to 3 getitem ops + one simple_call.
        use super::super::model::{SomeInteger, SomeTuple};
        let ann = mk_ann();
        let mut v_func = Variable::named("f");
        ann.setbinding(&mut v_func, SomeValue::object());
        let mut v_tup = Variable::named("tup");
        ann.setbinding(
            &mut v_tup,
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
            ])),
        );
        // `f(*tup)` encodes as: call_args(f, shape, tup) where shape =
        // (shape_cnt=0, shape_keys=(), shape_star=True).
        let shape_const = Hlvalue::Constant(Constant::new(ConstValue::Tuple(vec![
            ConstValue::Int(0),
            ConstValue::Tuple(vec![]),
            ConstValue::Bool(true),
        ])));
        let hl = HLOperation::new(
            OpKind::CallArgs,
            vec![
                Hlvalue::Variable(v_func),
                shape_const,
                Hlvalue::Variable(v_tup),
            ],
        );
        let new_ops = hl
            .transform(&ann)
            .expect("call_args(SomeObject, *SomeTuple) must transform");
        // 3 getitem + 1 simple_call = 4 ops
        assert_eq!(new_ops.len(), 4);
        assert!(matches!(new_ops[0].kind, OpKind::GetItem));
        assert!(matches!(new_ops[1].kind, OpKind::GetItem));
        assert!(matches!(new_ops[2].kind, OpKind::GetItem));
        assert!(matches!(new_ops[3].kind, OpKind::SimpleCall));
        // simple_call receives v_func + 3 unpacked values
        assert_eq!(new_ops[3].args.len(), 4);
    }

    #[test]
    fn transform_call_args_without_stararg_is_noop() {
        // upstream: if callspec.w_stararg is None, transform_varargs
        // falls through without returning ops.
        let ann = mk_ann();
        let mut v_func = Variable::named("f");
        ann.setbinding(&mut v_func, SomeValue::object());
        let shape_const = Hlvalue::Constant(Constant::new(ConstValue::Tuple(vec![
            ConstValue::Int(0),
            ConstValue::Tuple(vec![]),
            ConstValue::Bool(false),
        ])));
        let hl = HLOperation::new(
            OpKind::CallArgs,
            vec![Hlvalue::Variable(v_func), shape_const],
        );
        assert!(hl.transform(&ann).is_none());
    }

    #[test]
    fn transform_getattr_someinstance_without_property_is_noop() {
        // unaryop.py:909-921 — when the target attribute isn't a
        // property descriptor in any classdef, `_find_property_meth`
        // returns None and the transform leaves the op alone.
        use super::super::classdesc::ClassDef;
        use super::super::model::SomeInstance;
        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("pkg.X", None);
        let mut v_obj = Variable::named("obj");
        ann.setbinding(
            &mut v_obj,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let v_attr_const = Hlvalue::Constant(Constant::new(ConstValue::byte_str("prop")));
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![Hlvalue::Variable(v_obj), v_attr_const],
        );
        // Registered but returns None because the classdict carries no
        // `property` descriptor (Rust HostObject property variant not
        // yet modelled).
        assert!(hl.transform(&ann).is_none());
    }

    #[test]
    fn transform_setattr_someinstance_without_property_is_noop() {
        // unaryop.py:924-936 setter analogue.
        use super::super::classdesc::ClassDef;
        use super::super::model::{SomeInstance, SomeInteger};
        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("pkg.X", None);
        let mut v_obj = Variable::named("obj");
        ann.setbinding(
            &mut v_obj,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let mut v_value = Variable::named("val");
        ann.setbinding(&mut v_value, SomeValue::Integer(SomeInteger::default()));
        let v_attr_const = Hlvalue::Constant(Constant::new(ConstValue::byte_str("prop")));
        let hl = HLOperation::new(
            OpKind::SetAttr,
            vec![
                Hlvalue::Variable(v_obj),
                v_attr_const,
                Hlvalue::Variable(v_value),
            ],
        );
        assert!(hl.transform(&ann).is_none());
    }

    #[test]
    fn transform_getattr_someinstance_with_property_rewrites_via_getter_hidden_fn() {
        // unaryop.py:909-921 — when every classdef in the MRO binds
        // `prop` as a `property` descriptor, the transform rewrites
        // `getattr(v_obj, 'prop')` into `[getattr(v_obj, 'prop__getter__'),
        // simple_call(getter.result)]`.
        use super::super::super::flowspace::model::GraphFunc;
        use super::super::classdesc::{ClassDef, ClassDictEntry};
        use super::super::model::SomeInstance;
        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("pkg.Y", None);
        // Build a property(fget, fset) and bind it as the `prop` entry
        // in the classdesc's classdict — simulating what
        // `ClassDesc::add_source_attribute` produces when processing a
        // property descriptor.
        let fget_host =
            super::super::super::flowspace::model::HostObject::new_user_function(GraphFunc::new(
                "get_prop",
                Constant::new(ConstValue::Dict(Default::default())),
            ));
        let fset_host =
            super::super::super::flowspace::model::HostObject::new_user_function(GraphFunc::new(
                "set_prop",
                Constant::new(ConstValue::Dict(Default::default())),
            ));
        let prop_host = super::super::super::flowspace::model::HostObject::new_property(
            "pkg.Y.prop",
            Some(fget_host),
            Some(fset_host),
            None,
        );
        let classdesc = classdef.borrow().classdesc.clone();
        classdesc.borrow_mut().classdict.insert(
            "prop".to_string(),
            ClassDictEntry::constant(ConstValue::HostObject(prop_host)),
        );
        let mut v_obj = Variable::named("obj");
        ann.setbinding(
            &mut v_obj,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let v_attr_const = Hlvalue::Constant(Constant::new(ConstValue::byte_str("prop")));
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![Hlvalue::Variable(v_obj), v_attr_const],
        );
        let new_ops = hl
            .transform(&ann)
            .expect("property attr must rewrite to getter+simple_call");
        assert_eq!(new_ops.len(), 2);
        assert!(matches!(new_ops[0].kind, OpKind::GetAttr));
        // The rewritten getattr target is the hidden getter name.
        let target_attr = &new_ops[0].args[1];
        match target_attr {
            Hlvalue::Constant(c) => match &c.value {
                value if value.string_eq("prop__getter__") => {}
                other => panic!("expected Str, got {other:?}"),
            },
            other => panic!("expected Constant attr, got {other:?}"),
        }
        assert!(matches!(new_ops[1].kind, OpKind::SimpleCall));
    }

    #[test]
    fn transform_getitem_someinstance_rewrites_to_getattr_plus_simple_call() {
        // binaryop.py:727-730 — `op.getitem(v_ins, v_idx)` → getattr + simple_call.
        use super::super::classdesc::ClassDef;
        use super::super::model::{SomeInstance, SomeInteger};

        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("pkg.X", None);
        let mut v_ins = Variable::named("inst");
        ann.setbinding(
            &mut v_ins,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let mut v_idx = Variable::named("i");
        ann.setbinding(&mut v_idx, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::GetItem,
            vec![Hlvalue::Variable(v_ins), Hlvalue::Variable(v_idx)],
        );
        let new_ops = hl
            .transform(&ann)
            .expect("GetItem(SomeInstance, SomeObject) must register_transform");
        assert_eq!(new_ops.len(), 2);
        assert!(matches!(new_ops[0].kind, OpKind::GetAttr));
        assert!(matches!(new_ops[1].kind, OpKind::SimpleCall));
    }

    #[test]
    fn consider_someobject_len_returns_nonneg_integer() {
        // unaryop.py:158-159 — default len returns SomeInteger(nonneg=True).
        let (hl, ann) = hl1(OpKind::Len, SomeValue::object());
        let r = hl.consider(&ann).unwrap();
        match r {
            SomeValue::Integer(i) => assert!(i.nonneg, "nonneg not set"),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_somefloat_neg_returns_somefloat() {
        let (hl, ann) = hl1(OpKind::Neg, SomeValue::Float(SomeFloat::new()));
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Float(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somefloat_pos_returns_self() {
        let (hl, ann) = hl1(OpKind::Pos, SomeValue::Float(SomeFloat::new()));
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Float(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someobject_bool_on_none_is_constant_false() {
        let (hl, ann) = hl1(
            OpKind::Bool,
            SomeValue::None_(super::super::model::SomeNone::new()),
        );
        let r = hl.consider(&ann).unwrap();
        let SomeValue::Bool(b) = r else {
            panic!("expected SomeBool");
        };
        assert_eq!(b.base.const_box.unwrap().value, ConstValue::Bool(false));
    }

    #[test]
    fn consider_someobject_bool_adds_truth_branch_nonnone_refinement() {
        let ann = mk_ann();
        let mut v = Variable::named("maybe_s");
        ann.setbinding(&mut v, SomeValue::String(SomeString::new(true, false)));
        let hl = HLOperation::new(OpKind::Bool, vec![Hlvalue::Variable(v.clone())]);
        let r = hl.consider(&ann).unwrap();
        let SomeValue::Bool(b) = r else {
            panic!("expected SomeBool");
        };
        let ktd = b.knowntypedata.expect("knowntypedata must be populated");
        let narrowed = ktd
            .get(&true)
            .and_then(|inner| inner.get(&Rc::new(v)))
            .expect("truth branch must refine the object");
        match narrowed {
            SomeValue::String(s) => assert!(!s.inner.can_be_none),
            other => panic!("expected SomeString refinement, got {other:?}"),
        }
    }

    #[test]
    fn consider_someobject_bool_uses_constant_len_when_available() {
        let (hl, ann) = hl1(OpKind::Bool, SomeValue::Tuple(SomeTuple::new(vec![])));
        let r = hl.consider(&ann).unwrap();
        let SomeValue::Bool(b) = r else {
            panic!("expected SomeBool");
        };
        assert_eq!(b.base.const_box.unwrap().value, ConstValue::Bool(false));
    }

    #[test]
    fn consider_someinteger_abs_is_nonneg() {
        let (hl, ann) = hl1(OpKind::Abs, SomeValue::Integer(SomeInteger::default()));
        let r = hl.consider(&ann).unwrap();
        match r {
            SomeValue::Integer(i) => assert!(i.nonneg),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_somebool_bool_returns_self() {
        let (hl, ann) = hl1(OpKind::Bool, SomeValue::Bool(SomeBool::new()));
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someptr_bool_uses_ptr_override() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, LowLevelType, Ptr, PtrTarget, functionptr,
        };

        let ann = mk_ann();
        let ptr = functionptr(
            FuncType {
                args: vec![],
                result: LowLevelType::Void,
            },
            "f",
            None,
            None,
        );
        let mut s_ptr = super::super::model::SomePtr::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Void,
            }),
        });
        s_ptr.base.const_box = Some(Constant::new(ConstValue::LLPtr(Box::new(ptr))));
        let mut v = Variable::named("p");
        ann.setbinding(&mut v, SomeValue::Ptr(s_ptr));
        let hl = HLOperation::new(OpKind::Bool, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        let SomeValue::Bool(b) = r else {
            panic!("expected SomeBool");
        };
        assert_eq!(b.base.const_box.unwrap().value, ConstValue::Bool(true));
    }

    #[test]
    fn consider_someptr_len_uses_ptr_override() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            ArrayType, LowLevelType, Ptr, PtrTarget,
        };

        // upstream `SomePtr.len()` (lltype.py:1550-1555) propagates
        // `_fixedlength`'s `len()` probe, which raises `TypeError` on
        // non-array pointers. Use an Array ptr so the override returns
        // `SomeInteger(nonneg=True)` (the varsize-array path).
        let ann = mk_ann();
        let s_ptr = super::super::model::SomePtr::new(Ptr {
            TO: PtrTarget::Array(ArrayType::new(LowLevelType::Signed)),
        });
        let mut v = Variable::named("p");
        ann.setbinding(&mut v, SomeValue::Ptr(s_ptr));
        let hl = HLOperation::new(OpKind::Len, vec![Hlvalue::Variable(v)]);

        let r = hl.consider(&ann).unwrap();
        let SomeValue::Integer(i) = r else {
            panic!("expected SomeInteger");
        };
        assert!(i.nonneg);
    }

    #[test]
    fn consider_someinteriorptr_bool_uses_ptr_override() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            InteriorOffset, InteriorPtr, LowLevelType, StructType,
        };

        let ann = mk_ann();
        let mut v = Variable::named("p");
        ann.setbinding(
            &mut v,
            SomeValue::InteriorPtr(super::super::model::SomeInteriorPtr::new(InteriorPtr {
                PARENTTYPE: Box::new(LowLevelType::Struct(Box::new(StructType::new(
                    "S",
                    vec![("x".into(), LowLevelType::Signed)],
                )))),
                TO: Box::new(LowLevelType::Signed),
                offsets: vec![InteriorOffset::Field("x".into())],
            })),
        );
        let hl = HLOperation::new(OpKind::Bool, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    #[test]
    #[should_panic(expected = "non-constant field-name")]
    fn consider_someptr_getattr_uses_ptr_override() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, LowLevelType, Ptr, PtrTarget,
        };

        let ann = mk_ann();
        let s_ptr = super::super::model::SomePtr::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Void,
            }),
        });
        let mut v_ptr = Variable::named("p");
        ann.setbinding(&mut v_ptr, SomeValue::Ptr(s_ptr));
        let mut v_attr = Variable::named("name");
        ann.setbinding(
            &mut v_attr,
            SomeValue::String(super::super::model::SomeString::new(false, false)),
        );
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![Hlvalue::Variable(v_ptr), Hlvalue::Variable(v_attr)],
        );

        let _ = hl.consider(&ann);
    }

    #[test]
    #[should_panic(expected = "non-constant field-name")]
    fn consider_someptr_setattr_uses_ptr_override() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, LowLevelType, Ptr, PtrTarget,
        };

        let ann = mk_ann();
        let s_ptr = super::super::model::SomePtr::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Void,
            }),
        });
        let mut v_ptr = Variable::named("p");
        ann.setbinding(&mut v_ptr, SomeValue::Ptr(s_ptr));
        let mut v_attr = Variable::named("name");
        ann.setbinding(
            &mut v_attr,
            SomeValue::String(super::super::model::SomeString::new(false, false)),
        );
        let mut v_value = Variable::named("value");
        ann.setbinding(
            &mut v_value,
            SomeValue::Integer(SomeInteger::new(false, false)),
        );
        let hl = HLOperation::new(
            OpKind::SetAttr,
            vec![
                Hlvalue::Variable(v_ptr),
                Hlvalue::Variable(v_attr),
                Hlvalue::Variable(v_value),
            ],
        );

        let _ = hl.consider(&ann);
    }

    #[test]
    fn consider_somebool_invert_returns_someinteger() {
        let (hl, ann) = hl1(OpKind::Invert, SomeValue::Bool(SomeBool::new()));
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Integer(_)), "got {:?}", r);
    }

    #[test]
    fn consider_hash_reifies_annotator_error() {
        let (hl, ann) = hl1(OpKind::Hash, SomeValue::object());
        let err = hl
            .consider(&ann)
            .expect_err("hash() must return AnnotatorError");
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("cannot use hash() in RPython")
        );
    }

    #[test]
    fn consider_sometuple_len_is_constant() {
        // unaryop.py:334-335 — len returns immutablevalue(len(items)).
        let (hl, ann) = hl1(
            OpKind::Len,
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Float(SomeFloat::new()),
                SomeValue::Bool(SomeBool::new()),
            ])),
        );
        let r = hl.consider(&ann).unwrap();
        match r {
            SomeValue::Integer(i) => {
                let c = i.base.const_box.expect("const not propagated");
                assert_eq!(c.value, ConstValue::Int(3));
            }
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_sometuple_iter_returns_someiterator() {
        let (hl, ann) = hl1(OpKind::Iter, SomeValue::Tuple(SomeTuple::new(vec![])));
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someobject_str_returns_somestring() {
        let (hl, ann) = hl1(OpKind::Str, SomeValue::object());
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::String(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someobject_hex_returns_somestring() {
        let (hl, ann) = hl1(OpKind::Hex, SomeValue::object());
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::String(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someobject_pos_returns_impossible() {
        // unaryop.py:252-254 — default pos returns s_ImpossibleValue.
        let (hl, ann) = hl1(OpKind::Pos, SomeValue::object());
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Impossible), "got {:?}", r);
    }

    fn mk_list_of(item: SomeValue) -> super::super::model::SomeList {
        let ann = mk_ann();
        let listdef = super::super::listdef::ListDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            item,
            false,
            false,
        );
        super::super::model::SomeList::new(listdef)
    }

    #[test]
    fn consider_somelist_contains_returns_bool() {
        let ann = mk_ann();
        let mut v0 = Variable::named("lst");
        let mut v1 = Variable::named("x");
        ann.setbinding(
            &mut v0,
            SomeValue::List(mk_list_of(SomeValue::Integer(SomeInteger::default()))),
        );
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::Contains,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somelist_iter_returns_iterator() {
        let (hl, ann) = hl1(
            OpKind::Iter,
            SomeValue::List(mk_list_of(SomeValue::Integer(SomeInteger::default()))),
        );
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somedict_iter_returns_iterator() {
        let ann = mk_ann();
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let s_dict = super::super::model::SomeDict::new(dictdef);
        let mut v = Variable::named("d");
        ann.setbinding(&mut v, SomeValue::Dict(s_dict));
        let hl = HLOperation::new(OpKind::Iter, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somestring_len_const_returns_const_int() {
        // unaryop.py:720-725 — len(const "abc") → SomeInteger(const=3).
        let mut s = SomeString::new(false, false);
        s.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("abc")));
        let (hl, ann) = hl1(OpKind::Len, SomeValue::String(s));
        let r = hl.consider(&ann).unwrap();
        match r {
            SomeValue::Integer(i) => {
                let c = i.base.const_box.expect("const not propagated");
                assert_eq!(c.value, ConstValue::Int(3));
            }
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_somechar_len_is_const_1() {
        // unaryop.py:771-774 — SomeChar.len() → immutablevalue(1).
        let (hl, ann) = hl1(
            OpKind::Len,
            SomeValue::Char(super::super::model::SomeChar::new(false)),
        );
        let r = hl.consider(&ann).unwrap();
        match r {
            SomeValue::Integer(i) => {
                let c = i.base.const_box.expect("const not propagated");
                assert_eq!(c.value, ConstValue::Int(1));
            }
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_somechar_ord_returns_nonneg_integer() {
        let (hl, ann) = hl1(
            OpKind::Ord,
            SomeValue::Char(super::super::model::SomeChar::new(false)),
        );
        let r = hl.consider(&ann).unwrap();
        match r {
            SomeValue::Integer(i) => assert!(i.nonneg),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_someiterator_iter_returns_self() {
        // unaryop.py:808-810 — iter on iterator returns self.
        let ann = mk_ann();
        let it = SomeIterator::new(SomeValue::Tuple(SomeTuple::new(vec![])), vec![]);
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Iter, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someiterator_next_on_tuple_unions_items() {
        let ann = mk_ann();
        let it = SomeIterator::new(
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
            ])),
            vec![],
        );
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Integer(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someiterator_next_enumerate_returns_tuple_int_item() {
        // upstream unaryop.py:822-824 — enumerate variant yields
        // SomeTuple((SomeInteger(nonneg=True), getanyitem(container))).
        let ann = mk_ann();
        let it = SomeIterator::new(
            SomeValue::Tuple(SomeTuple::new(vec![SomeValue::Integer(
                SomeInteger::default(),
            )])),
            vec!["enumerate".to_string()],
        );
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        let SomeValue::Tuple(t) = r else {
            panic!("expected SomeTuple");
        };
        assert_eq!(t.items.len(), 2);
        assert!(matches!(t.items[0], SomeValue::Integer(_)));
        assert!(matches!(t.items[1], SomeValue::Integer(_)));
    }

    #[test]
    fn consider_someiterator_next_reversed_acts_as_default() {
        // upstream unaryop.py:826-827 — `("reversed",)` collapses to ().
        let ann = mk_ann();
        let it = SomeIterator::new(
            SomeValue::Tuple(SomeTuple::new(vec![SomeValue::Integer(
                SomeInteger::default(),
            )])),
            vec!["reversed".to_string()],
        );
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Integer(_)));
    }

    #[test]
    fn consider_someiterator_next_on_string_returns_char() {
        use super::super::model::SomeString;
        let ann = mk_ann();
        let it = SomeIterator::new(SomeValue::String(SomeString::new(false, true)), vec![]);
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Char(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someiterator_next_on_unicode_returns_codepoint() {
        let ann = mk_ann();
        let it = SomeIterator::new(
            SomeValue::UnicodeString(SomeUnicodeString::new(false, true)),
            vec![],
        );
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::UnicodeCodePoint(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somenone_getattr_returns_impossible() {
        let (hl, ann) = hl1(
            OpKind::GetAttr,
            SomeValue::None_(super::super::model::SomeNone::new()),
        );
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Impossible), "got {:?}", r);
    }

    #[test]
    fn consider_somepbc_class_name_returns_somestring() {
        let ann = mk_ann();
        let class_host =
            super::super::super::flowspace::model::HostObject::new_class("Abc", vec![]);
        let desc = ann.bookkeeper.getdesc(&class_host).expect("class desc");
        let mut v_cls = Variable::named("cls");
        ann.setbinding(
            &mut v_cls,
            SomeValue::PBC(super::super::model::SomePBC::new([desc], false)),
        );
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![
                Hlvalue::Variable(v_cls),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("__name__"))),
            ],
        );
        let r = hl.consider(&ann).expect("class __name__ getattr");
        assert!(matches!(r, SomeValue::String(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somepbc_single_desc_routes_through_s_read_attribute() {
        // upstream pbc_getattr single-desc fast path: `if len(descs) == 1:
        // return first.s_read_attribute(attr)`. Construct a ClassDesc PBC
        // whose class has a const attribute and read it through PBC.getattr.
        let ann = mk_ann();
        let class_host =
            super::super::super::flowspace::model::HostObject::new_class("Box", vec![]);
        class_host.class_set("N", ConstValue::Int(42));
        let desc = ann.bookkeeper.getdesc(&class_host).expect("class desc");
        let mut v_cls = Variable::named("cls");
        ann.setbinding(
            &mut v_cls,
            SomeValue::PBC(super::super::model::SomePBC::new([desc], false)),
        );
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![
                Hlvalue::Variable(v_cls),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("N"))),
            ],
        );
        let r = hl.consider(&ann).expect("class const attr getattr");
        assert!(
            matches!(r.const_(), Some(ConstValue::Int(42))),
            "got {:?}",
            r
        );
    }

    #[test]
    fn consider_somenone_len_returns_impossible() {
        let (hl, ann) = hl1(
            OpKind::Len,
            SomeValue::None_(super::super::model::SomeNone::new()),
        );
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Impossible), "got {:?}", r);
    }

    #[test]
    fn consider_somestring_iter_returns_iterator() {
        let (hl, ann) = hl1(
            OpKind::Iter,
            SomeValue::String(SomeString::new(false, false)),
        );
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_contains_string_nul_sets_false_branch_nonnul_refinement() {
        let ann = mk_ann();
        let mut v_string = Variable::named("s");
        ann.setbinding(
            &mut v_string,
            SomeValue::String(SomeString::new(false, false)),
        );
        let mut nul_char = super::super::model::SomeChar::new(false);
        nul_char.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("\0")));
        let mut v_char = Variable::named("c");
        ann.setbinding(&mut v_char, SomeValue::Char(nul_char));
        let hl = HLOperation::new(
            OpKind::Contains,
            vec![
                Hlvalue::Variable(v_string.clone()),
                Hlvalue::Variable(v_char),
            ],
        );
        let r = hl.consider(&ann).unwrap();
        let SomeValue::Bool(b) = r else {
            panic!("expected SomeBool");
        };
        let ktd = b.knowntypedata.expect("knowntypedata must be populated");
        let narrowed = ktd
            .get(&false)
            .and_then(|inner| inner.get(&Rc::new(v_string)))
            .expect("false branch must refine the string");
        match narrowed {
            SomeValue::String(s) => assert!(s.inner.no_nul),
            other => panic!("expected SomeString refinement, got {other:?}"),
        }
    }

    #[test]
    fn consider_contains_unicode_nul_sets_false_branch_nonnul_refinement() {
        let ann = mk_ann();
        let mut v_string = Variable::named("u");
        ann.setbinding(
            &mut v_string,
            SomeValue::UnicodeString(SomeUnicodeString::new(false, false)),
        );
        let mut nul_cp = super::super::model::SomeUnicodeCodePoint::new(false);
        nul_cp.inner.base.const_box = Some(Constant::new(ConstValue::uni_str("\0")));
        let mut v_char = Variable::named("c");
        ann.setbinding(&mut v_char, SomeValue::UnicodeCodePoint(nul_cp));
        let hl = HLOperation::new(
            OpKind::Contains,
            vec![
                Hlvalue::Variable(v_string.clone()),
                Hlvalue::Variable(v_char),
            ],
        );
        let r = hl.consider(&ann).unwrap();
        let SomeValue::Bool(b) = r else {
            panic!("expected SomeBool");
        };
        let ktd = b.knowntypedata.expect("knowntypedata must be populated");
        let narrowed = ktd
            .get(&false)
            .and_then(|inner| inner.get(&Rc::new(v_string)))
            .expect("false branch must refine the unicode string");
        match narrowed {
            SomeValue::UnicodeString(s) => assert!(s.inner.no_nul),
            other => panic!("expected SomeUnicodeString refinement, got {other:?}"),
        }
    }

    #[test]
    fn consider_somedict_contains_position_required() {
        let ann = mk_ann();
        let pk = super::super::bookkeeper::PositionKey::new(3, 0, 0);
        ann.bookkeeper.set_position_key(Some(pk));
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let s_dict = super::super::model::SomeDict::new(dictdef);
        let mut v0 = Variable::named("d");
        let mut v1 = Variable::named("k");
        ann.setbinding(&mut v0, SomeValue::Dict(s_dict));
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::Contains,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    #[test]
    fn list_method_extend_uses_iter_next_for_non_list_iterable() {
        let ann = mk_ann();
        let s_list = mk_list_of(SomeValue::Integer(SomeInteger::default()));
        let s_iterable = SomeValue::Tuple(SomeTuple::new(vec![SomeValue::Integer(
            SomeInteger::default(),
        )]));
        let result = list_method_extend(&ann, &s_list, &s_iterable);
        assert!(matches!(result, SomeValue::Impossible));
        assert!(matches!(
            s_list.listdef.read_item(None),
            SomeValue::Integer(_)
        ));
    }

    #[test]
    fn builtin_method_call_binds_keywords_like_python_call() {
        let ann = mk_ann();
        let method = SomeBuiltinMethod::new(
            "str_method_find",
            SomeValue::String(SomeString::new(false, false)),
            "find",
        );
        let mut kwds = std::collections::HashMap::new();
        kwds.insert("start".to_string(), Some(s_const_int(1)));
        kwds.insert("end".to_string(), Some(s_const_int(3)));
        let args = super::super::argument::ArgumentsForTranslation::new(
            vec![Some(SomeValue::String(SomeString::new(false, false)))],
            Some(kwds),
            None,
        );
        let result = call_builtin_method(&ann, &method, &args).expect("builtin call must bind");
        assert!(matches!(result, SomeValue::Integer(_)));
    }

    #[test]
    fn find_method_exposes_dict_items_and_string_split_surface() {
        let ann = mk_ann();
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let s_dict = SomeValue::Dict(super::super::model::SomeDict::new(dictdef));
        let s_string = SomeValue::String(SomeString::new(false, false));

        let SomeValue::BuiltinMethod(dict_items) = s_dict.find_method("items").unwrap() else {
            panic!("dict.items must be recognized");
        };
        assert_eq!(dict_items.analyser_name, "dict_method_items");

        let SomeValue::BuiltinMethod(str_split) = s_string.find_method("split").unwrap() else {
            panic!("str.split must be recognized");
        };
        assert_eq!(str_split.analyser_name, "str_method_split");
    }

    #[test]
    fn str_method_upper_preserves_unicode_family() {
        let ann = mk_ann();
        let unicode = SomeValue::UnicodeString(SomeUnicodeString::new(false, false));
        let codepoint =
            SomeValue::UnicodeCodePoint(super::super::model::SomeUnicodeCodePoint::new(false));

        assert!(matches!(
            str_method_upper(&ann, &unicode),
            SomeValue::UnicodeString(_)
        ));
        assert!(matches!(
            str_method_upper(&ann, &codepoint),
            SomeValue::UnicodeString(_)
        ));
    }

    #[test]
    fn str_method_lower_preserves_unicode_family() {
        let ann = mk_ann();
        let unicode = SomeValue::UnicodeString(SomeUnicodeString::new(false, false));
        let codepoint =
            SomeValue::UnicodeCodePoint(super::super::model::SomeUnicodeCodePoint::new(false));

        assert!(matches!(
            str_method_lower(&ann, &unicode),
            SomeValue::UnicodeString(_)
        ));
        assert!(matches!(
            str_method_lower(&ann, &codepoint),
            SomeValue::UnicodeString(_)
        ));
    }

    #[test]
    fn dict_method_items_returns_list_of_pairs() {
        let ann = mk_ann();
        let pk = super::super::bookkeeper::PositionKey::new(7, 0, 0);
        ann.bookkeeper.set_position_key(Some(pk));
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let s_dict = super::super::model::SomeDict::new(dictdef);
        let result = dict_method_items(&ann, &s_dict);
        let SomeValue::List(list) = result else {
            panic!("expected SomeList");
        };
        assert!(matches!(list.listdef.read_item(None), SomeValue::Tuple(_)));
    }

    #[test]
    fn find_method_exposes_dict_hash_and_move_to_end_surface() {
        let ann = mk_ann();
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let s_dict = SomeValue::Dict(super::super::model::SomeDict::new(dictdef));

        let SomeValue::BuiltinMethod(getitem) = s_dict.find_method("getitem_with_hash").unwrap()
        else {
            panic!("dict.getitem_with_hash must be recognized");
        };
        assert_eq!(getitem.analyser_name, "dict_method_getitem_with_hash");

        let SomeValue::BuiltinMethod(move_to_end) = s_dict.find_method("move_to_end").unwrap()
        else {
            panic!("dict.move_to_end must be recognized");
        };
        assert_eq!(move_to_end.analyser_name, "dict_method_move_to_end");
    }

    #[test]
    fn dict_method_getitem_with_hash_reads_value_type() {
        let ann = mk_ann();
        let pk = super::super::bookkeeper::PositionKey::new(9, 0, 0);
        ann.bookkeeper.set_position_key(Some(pk));
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let s_dict = super::super::model::SomeDict::new(dictdef);
        let result = dict_method_getitem_with_hash(
            &ann,
            &s_dict,
            &SomeValue::Integer(SomeInteger::default()),
            &SomeValue::Integer(SomeInteger::default()),
        );
        assert!(matches!(result, SomeValue::String(_)));
    }

    #[test]
    fn builtin_method_call_binds_hash_method_keywords() {
        let ann = mk_ann();
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let method = SomeBuiltinMethod::new(
            "dict_method_contains_with_hash",
            SomeValue::Dict(super::super::model::SomeDict::new(dictdef)),
            "contains_with_hash",
        );
        let mut kwds = std::collections::HashMap::new();
        kwds.insert("hash".to_string(), Some(s_const_int(11)));
        let args = super::super::argument::ArgumentsForTranslation::new(
            vec![Some(SomeValue::Integer(SomeInteger::default()))],
            Some(kwds),
            None,
        );
        let result = call_builtin_method(&ann, &method, &args).expect("builtin call must bind");
        assert!(matches!(result, SomeValue::Bool(_)));
    }
}
