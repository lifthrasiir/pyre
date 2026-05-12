//! `model::FunctionGraph` (pyre surface DSL) →
//! `flowspace::FunctionGraph` (RPython-orthodox) adapter.
//!
//! TODO(retire-this-adapter): this file has no upstream RPython
//! counterpart. RPython's pipeline has only one graph model — the
//! annotator builds its `FunctionGraph` (`rpython/flowspace/model.py`)
//! directly and the rtyper consumes it in place.  Pyre carries two
//! graph models in parallel: `crate::model::FunctionGraph` for the
//! surface DSL emitted by `parse → front → SemanticProgram`, and
//! `crate::flowspace::model::FunctionGraph` for the real
//! `translator/rtyper/` pipeline.  This adapter bridges the gap.
//! Retire when the surface DSL is replaced with `flowspace`-native
//! producers; until then the adapter remains the per-graph entry
//! into `RPythonAnnotator` / `RPythonTyper::specialize`.
//!
//! ## Why not `PyGraph`?
//!
//! `PyGraph` (`flowspace::pygraph::PyGraph`) wraps a `FunctionGraph`
//! with `GraphFunc` / `HostCode` / `Signature` / `defaults` —
//! Python-runtime function metadata that pyre's surface DSL does not
//! produce (`parse → front → SemanticProgram` operates on Rust source,
//! not CPython callables). `RPythonTyper::specialize`
//! (`rtyper.rs:1743`) does NOT consume `PyGraph` directly — it iterates
//! `RPythonAnnotator.annotated` / `all_blocks`, which Slice 2's
//! `specialize_legacy_graph` will populate with the
//! [`FlowspaceAdapterOutput`] this adapter returns. Skipping the PyGraph
//! wrapping avoids fabricating fake `GraphFunc` / `HostCode` instances.
//!
//! ## Layout
//!
//! The adapter performs three responsibilities, all line-by-line at
//! `function_graph_to_flowspace`:
//!
//! 1. **Annotation lift** — project pyre's
//!    `AnnotationState.types` (`ValueId → ValueType`) to `SomeValue`
//!    shells on freshly-allocated `flowspace::Variable`s. Variable
//!    identity is block-local per `flowspace/model.py:checkgraph`;
//!    the adapter keeps a `ValueId → Variable` representative map for
//!    post-specialize readback.
//! 2. **Per-OpKind translation** — `translate_op` maps each pyre
//!    `model::SpaceOperation` to a `flowspace::SpaceOperation` over
//!    `Hlvalue` operands.  Pre-rtyper variants (`Input`, `ConstInt`,
//!    `ConstFloat`, `BinOp`, `Call`, `FieldRead`, `ArrayRead`, ...)
//!    have explicit arms; post-rtyper jtransform variants are
//!    classified by [`post_rtyper_jtransform_variant_name`] and
//!    fail-loud with a stage-mismatch message.
//! 3. **Block topology** — wires `flowspace::Block` per legacy
//!    `Block`, translates `exits` / `exitcase` / `exitswitch`,
//!    designates `startblock` / `returnblock` / `exceptblock`, and
//!    assembles a `flowspace::FunctionGraph`.  `getreturnvar`
//!    (`rtyper.rs:1633-1638`) is non-degenerate because the
//!    returnblock's inputarg is materialised as the canonical
//!    flowspace return `Variable`.
//!
//! [`crate::translator::rtyper::cutover::specialize_legacy_graph_with_registry_seed`]
//! drives this adapter, runs `RPythonTyper::specialize`, projects
//! `LowLevelType → ConcreteType`, and returns a [`TypeResolutionState`]
//! keyed by the original pyre `ValueId`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::flowspace::model::{
    self as flowspace_model, Block as FlowspaceBlock, BlockRef, ConstValue, Constant,
    FunctionGraph as FlowspaceGraph, Hlvalue, HostObject, Link as FlowspaceLink,
    SpaceOperation as FlowspaceOp, Variable, c_last_exception,
};
use crate::jit_codewriter::annotation_state::AnnotationState;
use crate::model::{
    BlockId, ExitCase, ExitSwitch, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueId,
};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

/// Map from legacy `ValueId` to a representative `flowspace::Variable`
/// the adapter created for readback.
///
/// This is not the graph's identity model. RPython `checkgraph` requires
/// block inputargs and operation results to be defined in exactly one
/// block, so [`function_graph_to_flowspace`] uses block-local Variables
/// while translating the actual graph. The representative map lets
/// Slice 2 project `Variable.concretetype` back into pyre's legacy
/// `ValueId` keyed `TypeResolutionState`.
pub type ValueIdToVariable = HashMap<ValueId, Variable>;

pub use crate::jit_codewriter::annotation_state::valuetype_to_someshell;

/// Allocate a fresh `flowspace::Variable` and attach the projected
/// `SomeValue` shell to its `annotation` slot.
///
/// The legacy `ValueId` does NOT carry over to `Variable.id` —
/// `Variable::new` allocates a fresh process-wide identity
/// (`flowspace/model.rs:2042`). Identity correspondence is preserved
/// out-of-band by [`ValueIdToVariable`].
/// Return `true` when `value` is a synthetic placeholder allocated
/// by `model.rs::set_return(_, None)` for a void return — i.e. a
/// `ValueId` that appears in `Link.args` but is **not** defined as
/// any block's `inputargs` entry or any op's `result`.  Mirrors
/// `translator/rtyper/legacy_annotator.rs:183-193
/// is_synthetic_return_void_value` line-by-line so the adapter and
/// the legacy annotator agree on which `Link.args` ValueIds are
/// "void return placeholders" eligible for the Void pre-seed and
/// which are genuine `undefined operand` producer bugs that should
/// remain fail-loud.
fn is_synthetic_return_void_value(graph: &FunctionGraph, value: ValueId) -> bool {
    for block in &graph.blocks {
        if block.inputargs.contains(&value) {
            return false;
        }
        if block.operations.iter().any(|op| op.result == Some(value)) {
            return false;
        }
    }
    true
}

fn seed_variable(vid: ValueId, annotations: &AnnotationState) -> Variable {
    let var = Variable::new();
    // Prefer the precise per-`ValueId` `SomeValue` when a producer
    // attached one (`AnnotationState::set_some`), matching upstream
    // `_setbinding(v, s_value)` semantics
    // (`rpython/annotator/annrpython.py:333-340`).  Fall back to the
    // `ValueType` -> `SomeValue` projection at `valuetype_to_someshell`,
    // which lifts `Int`/`Float`/`Ref`/`State` to definite shells and
    // returns `None` for `Unknown` (annotation gap → leave the
    // `Variable.annotation` slot empty, surfacing the producer-side
    // gap as a fail-loud `KeyError: no binding for arg` at
    // `bindingrepr` instead of silently bridging to `GcRef` via a
    // fabricated `SomeInstance(None)` shell).
    if let Some(s) = annotations.some(vid) {
        *var.annotation.borrow_mut() = Some(s.clone());
    } else if let Some(shell) = valuetype_to_someshell(annotations.get(vid)) {
        *var.annotation.borrow_mut() = Some(Rc::new(shell));
    }
    var
}

/// Build the `ValueId → flowspace::Variable` map for every value
/// reachable from `legacy.blocks`.
///
/// Three reference-site classes seed the map:
///
/// 1. **Definitions** — `block.inputargs` (RPython-orthodox phi nodes)
///    and `op.result`. Every operand referenced via `op.args` /
///    `link.args` / `exitswitch` resolves to a definition site in the
///    same graph (legacy `FunctionGraph` is mostly SSA), so seeding
///    definitions covers most of the value set.
///
/// 2. **Link-side sentinels** — `link.args` / `link.last_exception` /
///    `link.last_exc_value`. RPython `flowspace/model.py:114` and
///    pyre's front (`front/ast.rs:5320-5331`) allow a `Link.args` slot
///    to carry a *fresh* prevblock-side `Variable` whose only "defining
///    site" is the link itself — the value flows into the target
///    block's inputarg via this synthetic Variable. The adapter must
///    seed a `Variable` for each such `ValueId` so Slice 1c's link
///    translation can resolve the operand without tripping the
///    "undefined operand" invariant in `lookup_operand`.
///
/// 3. **Exitswitch values** — `block.exitswitch = Some(ExitSwitch::Value(vid))`
///    sometimes references a `ValueId` defined in a successor block's
///    inputarg context (rarely but legitimately in legacy graphs).
///    Seeded for the same reason.
///
/// Each `ValueId` is seeded exactly once via `entry().or_insert_with`,
/// preserving operand identity across multiple readers — Slice 1b's op
/// translator looks up the same Variable instance for every reader of a
/// given `ValueId`, matching upstream Python's reference semantics where
/// `op.args[i]` and `op2.args[j]` may be the same `Variable` object.
///
/// **Restricted to the adapter / its tests.**  `function_graph_to_flowspace`
/// builds a *block-local* `ValueId -> Variable` map per block in the
/// topology assembly pass, mirroring `RPython` `checkgraph`'s
/// per-block Variable invariant
/// (`rpython/flowspace/model.py:585-590`: a Variable must be defined in
/// exactly one block).  Using this whole-graph helper as the source of
/// truth for the adapter's main path would violate that invariant by
/// reusing a single `Variable` across blocks.  The helper stays in the
/// crate solely to back the adapter's regression tests
/// (`build_value_to_variable_map_*`); production cutover code must use
/// the per-block maps owned by `function_graph_to_flowspace`.
#[cfg(test)]
pub(crate) fn build_value_to_variable_map(
    legacy: &FunctionGraph,
    annotations: &AnnotationState,
) -> ValueIdToVariable {
    let mut map: ValueIdToVariable = HashMap::new();
    for block in &legacy.blocks {
        // Class 1a — block-inputarg definitions.
        for &vid in &block.inputargs {
            map.entry(vid)
                .or_insert_with(|| seed_variable(vid, annotations));
        }

        // Per-block name → inputarg-Variable lookup for `OpKind::Input`
        // rebind-aliasing. Pyre's surface front (`front/ast.rs`) emits a
        // *leading* `Input{name, ty}` op for each named parameter whose
        // `op.result` matches a `block.inputargs` entry, and may emit
        // *additional* `Input{same name}` ops with fresh `op.result`
        // ValueIds for body-side rebinds. RPython's flowspace has no
        // such Input op machinery — the parameter Variable IS the
        // inputarg. Without aliasing the rebind result to the
        // canonical inputarg Variable, `setup_block_entry` writes
        // `concretetype` only on the inputarg's Rc<RefCell> and the
        // body's BinOp lookup hits a fresh Variable with `None`
        // concretetype, tripping `genop`'s "wrong level!" assertion.
        let mut name_to_inputarg_var: HashMap<&str, Variable> = HashMap::new();
        for op in &block.operations {
            if let OpKind::Input { name, .. } = &op.kind {
                if let Some(result) = op.result {
                    if block.inputargs.contains(&result) {
                        if let Some(var) = map.get(&result) {
                            name_to_inputarg_var
                                .entry(name.as_str())
                                .or_insert_with(|| var.clone());
                        }
                    }
                }
            }
        }

        // Class 1b — op-result definitions, with Input rebind aliasing.
        for op in &block.operations {
            let Some(result) = op.result else { continue };
            if map.contains_key(&result) {
                continue;
            }
            let var = if let OpKind::Input { name, .. } = &op.kind {
                name_to_inputarg_var
                    .get(name.as_str())
                    .cloned()
                    .unwrap_or_else(|| seed_variable(result, annotations))
            } else {
                seed_variable(result, annotations)
            };
            map.insert(result, var);
        }
        // Class 3 — exitswitch-referenced values.
        if let Some(crate::model::ExitSwitch::Value(vid)) = &block.exitswitch {
            map.entry(*vid)
                .or_insert_with(|| seed_variable(*vid, annotations));
        }
        // Class 2 — link-side sentinels.
        for link in &block.exits {
            for arg in &link.args {
                if let LinkArg::Value(vid) = arg {
                    map.entry(*vid)
                        .or_insert_with(|| seed_variable(*vid, annotations));
                }
            }
            if let Some(LinkArg::Value(vid)) = &link.last_exception {
                map.entry(*vid)
                    .or_insert_with(|| seed_variable(*vid, annotations));
            }
            if let Some(LinkArg::Value(vid)) = &link.last_exc_value {
                map.entry(*vid)
                    .or_insert_with(|| seed_variable(*vid, annotations));
            }
        }
    }
    map
}

/// `ValueId → Hlvalue` map combining the [`ValueIdToVariable`] map with
/// constant-inlining of `OpKind::ConstInt` / `ConstFloat` define-ops.
///
/// RPython's flowspace inlines constants natively as `Hlvalue::Constant`
/// in `op.args` (`flowspace/operation.py:152` `simple_call(target,
/// *args)` — `target` and each `arg` is either a `Variable` or
/// `Constant`). Pyre's legacy graph splits constants into define-ops
/// (`OpKind::ConstInt(n)` produces a fresh `ValueId` consumed
/// elsewhere). The adapter must recombine: every `ValueId` defined as a
/// const becomes a `Hlvalue::Constant`; every other defined `ValueId`
/// remains a `Hlvalue::Variable` from the Slice 1a map.
///
/// Constants are wrapped with their low-level concretetype attached,
/// matching RPython's `Constant.concretetype` shape. The legacy graph
/// used a separate `ValueId` for the define-op; after inlining, that
/// `ValueId` is tracked separately for readback.
pub fn build_value_to_hlvalue_map(
    legacy: &FunctionGraph,
    value_to_var: &ValueIdToVariable,
) -> HashMap<ValueId, Hlvalue> {
    let mut map: HashMap<ValueId, Hlvalue> = value_to_var
        .iter()
        .map(|(&vid, var)| (vid, Hlvalue::Variable(var.clone())))
        .collect();

    for block in &legacy.blocks {
        for op in &block.operations {
            let Some(result) = op.result else {
                continue;
            };
            match &op.kind {
                OpKind::ConstInt(n) => {
                    map.insert(
                        result,
                        Hlvalue::Constant(Constant::with_concretetype(
                            ConstValue::Int(*n),
                            LowLevelType::Signed,
                        )),
                    );
                }
                OpKind::ConstBool(b) => {
                    map.insert(
                        result,
                        Hlvalue::Constant(Constant::with_concretetype(
                            ConstValue::Bool(*b),
                            LowLevelType::Bool,
                        )),
                    );
                }
                OpKind::ConstFloat(bits) => {
                    map.insert(
                        result,
                        Hlvalue::Constant(Constant::with_concretetype(
                            ConstValue::Float(*bits),
                            LowLevelType::Float,
                        )),
                    );
                }
                _ => {}
            }
        }
    }
    map
}

/// Look up the `Hlvalue` for a `ValueId` operand. Surfaces a
/// fail-loud `TyperError` when the operand is undefined (every
/// referenced `ValueId` must have been seeded by Slice 1a's
/// [`build_value_to_variable_map`] or shadowed by
/// [`build_value_to_hlvalue_map`]'s const inlining).
///
/// The error message embeds the enclosing `SpaceOperation` (variant
/// name + result vid) and the role of the failing argument (e.g.
/// `"lhs"`, `"rhs"`, `"base"`, `"index"`, `"value"`, `"operand"`,
/// `"args[i]"`, `"result"`) so per-graph diagnosis can locate the
/// broken op without re-traversing the graph. The required substring
/// `"undefined operand ValueId"` is preserved verbatim so the dual
/// gate's `is_known_unported` predicate (`cutover.rs:441`) keeps
/// matching this category.
fn lookup_operand(
    value_map: &HashMap<ValueId, Hlvalue>,
    vid: ValueId,
    op: &SpaceOperation,
    arg_role: &str,
) -> Result<Hlvalue, TyperError> {
    value_map.get(&vid).cloned().ok_or_else(|| {
        TyperError::message(format!(
            "translate_op: undefined operand {vid:?} as {arg_role} of {opkind} \
             (result {result:?}) — adapter invariant broken (every referenced \
             ValueId must be defined as a block inputarg or op result)",
            opkind = opkind_variant_name(&op.kind),
            result = op.result,
        ))
    })
}

/// Resolve the `Hlvalue` result slot for a legacy op. When the op has
/// no result (`Option::None`), allocate a fresh anonymous Variable per
/// RPython convention (every `SpaceOperation.result` slot is non-None
/// upstream — model.py:432-438; void-result ops use a throwaway
/// `Variable()`).
fn resolve_result_hlvalue(
    op: &SpaceOperation,
    value_map: &HashMap<ValueId, Hlvalue>,
) -> Result<Hlvalue, TyperError> {
    match op.result {
        Some(vid) => lookup_operand(value_map, vid, op, "result"),
        None => Ok(Hlvalue::Variable(Variable::new())),
    }
}

/// Map a pyre-frontend unary op name (`front/ast.rs:3274-3281
/// unary_op_name`) onto the RPython flowspace operator name
/// (`rpython/flowspace/operation.py:465-474`).
///
/// `neg` and `bool` pass through (registered upstream as
/// `add_operator('neg', 1, ..)` at line 466 and `add_operator('bool',
/// 1, ..)` at line 467).  `cast_ptr_to_int` is the one cross-class
/// cast op pyre's frontend emits that has a real rtyper handler
/// (`rbuiltin.rs::rtype_cast_ptr_to_int`, registered as the
/// `"cast_ptr_to_int"` arm of `RPythonTyper::translate_operation`).
///
/// TODO(cast-dispatch-route): in upstream RPython, `same_as` and the
/// `cast_*` family are NOT high-level flowspace SpaceOperations.
/// `same_as` is generated by the rtyper itself (`rtyper.py:478-481`)
/// as a post-translation renaming op; the `cast_*` family is
/// generated through repr conversions or `lltype.cast_*` builtin
/// call paths (`rbuiltin.py:543-558`).  Pyre emits them directly at
/// the AST surface in `front/ast.rs::cast_op_name` (~line 5873), so
/// this adapter has to accept them at the rtyper input.  The handler
/// bodies in
/// `rbuiltin.rs::rtype_{same_as,cast_int_to_float,cast_float_to_int,
/// cast_int_to_ptr}` are line-by-line ports of upstream
/// (`rint.py:651-668`, `rfloat.py:31-53`, `rbuiltin.py:543-557`); the
/// remaining divergence is the dispatch route, not the handler logic.
/// Retire by lifting these out of `Expr::Cast` lowering and routing
/// through repr conversions instead (multi-session epic).
///
/// `not` and `deref` are the only fail-loud arms: pyre's frontend
/// eliminates both at the source (`front/ast.rs::Expr::Unary`
/// UnOp::Not desugar / Deref pass-through, both landed 2026-05-04 on
/// `annrpython`).  Reaching either arm means a synthetic graph
/// injected the op directly.  RPython distinguishes logical `not`
/// (UNARY_NOT, lowered as `bool(operand)` + branch —
/// `flowcontext.py:531-538`) from bitwise `invert` (UNARY_INVERT —
/// `flowcontext.py:190` → `op.invert`); without static type info,
/// the adapter cannot discriminate, so fail-loud is the only safe
/// choice.
fn normalize_unary_op_name(pyre_name: &str) -> Result<String, TyperError> {
    match pyre_name {
        "neg" => Ok("neg".to_string()),
        // RPython `bool` is registered as a unary op at
        // `operation.py:467 add_operator('bool', 1, ..)` and emitted
        // by `flowcontext.py:531-538 UNARY_NOT` /
        // `:766-777 JUMP_IF_*_OR_POP` as the discriminator before a
        // `guessbool` fork.  Pyre's frontend emits
        // `OpKind::UnaryOp { op: "bool", .. }` from the `&&` / `||`
        // short-circuit desugar at `front/ast.rs::Expr::Binary`,
        // mirroring `build_flow.rs:1191 lower_short_circuit`.
        // Pass through unchanged.
        "bool" => Ok("bool".to_string()),
        // `invert` — PyPy `add_operator('invert', 1, .., pure=True)` at
        // `operation.py:474`, emitted by `flowcontext.py:188-191
        // UNARY_INVERT` and dispatched through
        // `RPythonTyper::translate_op`'s `"invert"` arm
        // (`rtyper.rs:2025`) into `IntegerRepr::rtype_invert`
        // (`rint.py:107-110` → `rint.rs:284`). Pyre's
        // `front/ast.rs::Expr::Unary(UnOp::Not(_))` literal-int branch
        // emits `OpKind::UnaryOp { op: "invert", .. }` directly when
        // the operand is a `syn::Lit::Int` (the bitwise-complement
        // case Rust's `!42_i64` denotes).  Without this arm the
        // literal-int parity path Skip-classifies in the real rtyper.
        "invert" => Ok("invert".to_string()),
        // `cast_ptr_to_int` — `lloperation.py`-registered SpaceOperation;
        // dispatched through `RPythonTyper::translate_operation`'s
        // `cast_ptr_to_int` arm into `rbuiltin.rs::rtype_cast_ptr_to_int`
        // (line-by-line port of `rbuiltin.py:543-548`).
        "cast_ptr_to_int" => Ok("cast_ptr_to_int".to_string()),
        // Cast family — pyre's `front/ast.rs:5873 cast_op_name` emits
        // these for surface `as T` casts whose source/target categories
        // diverge.  Each has a dedicated `RPythonTyper::translate_
        // operation` arm into `rbuiltin.rs::rtype_*` (verbatim ports
        // of upstream `@typer_for(lltype.cast_*)` bodies):
        //
        //   - `same_as`           → `rtype_same_as`
        //                           (rtyper.py:478-481 internal renaming)
        //   - `cast_int_to_float` → `rtype_cast_int_to_float`
        //                           (rint.py:651-652)
        //   - `cast_float_to_int` → `rtype_cast_float_to_int`
        //                           (rint.py:667-668; rfloat.py:53)
        //   - `cast_int_to_ptr`   → `rtype_cast_int_to_ptr`
        //                           (rbuiltin.py:551-557)
        "same_as" => Ok("same_as".to_string()),
        "cast_int_to_float" => Ok("cast_int_to_float".to_string()),
        "cast_float_to_int" => Ok("cast_float_to_int".to_string()),
        "cast_int_to_ptr" => Ok("cast_int_to_ptr".to_string()),
        // Bool widening / truthiness — `rbool.py:49` cast_bool_to_int /
        // cast_bool_to_float, `rint.py:rtype_int__Bool` int_is_true,
        // `rfloat.py:rtype_Float__Bool` float_is_true.
        "cast_bool_to_int" => Ok("cast_bool_to_int".to_string()),
        "cast_bool_to_float" => Ok("cast_bool_to_float".to_string()),
        "int_is_true" => Ok("int_is_true".to_string()),
        "float_is_true" => Ok("float_is_true".to_string()),
        // Unsigned cross-signedness casts — `rint.py:rtype_uint_*`,
        // `rbool.py:77-83 uint_is_true`.  Pyre emits these from
        // `front/ast.rs::cast_op_name` for `u* as T` surface casts.
        "cast_uint_to_int" => Ok("cast_uint_to_int".to_string()),
        "cast_int_to_uint" => Ok("cast_int_to_uint".to_string()),
        "cast_uint_to_float" => Ok("cast_uint_to_float".to_string()),
        "cast_float_to_uint" => Ok("cast_float_to_uint".to_string()),
        "cast_bool_to_uint" => Ok("cast_bool_to_uint".to_string()),
        "uint_is_true" => Ok("uint_is_true".to_string()),
        other => Err(TyperError::missing_rtype_operation(format!(
            "normalize_unary_op_name: pyre UnaryOp `{other}` has no \
             flowspace counterpart (operation.py:465-474 registers \
             only `pos` / `neg` / `invert` / `bool` as unary ops; \
             `same_as` / `cast_int_to_float` / `cast_float_to_int` / \
             `cast_int_to_ptr` / `cast_bool_to_int` / `cast_bool_to_float` / \
             `int_is_true` / `float_is_true` / `cast_uint_to_int` / \
             `cast_int_to_uint` / `cast_uint_to_float` / `cast_float_to_uint` / \
             `cast_bool_to_uint` / `uint_is_true` route through \
             `rbuiltin.rs::rtype_*` via `RPythonTyper::translate_operation`).  \
             Frontend must distinguish bitwise `invert` from logical `not` \
             and remove `deref` before reaching the rtyper."
        ))),
    }
}

/// Map a pyre-frontend binary op name (`front/ast.rs:3227-3258
/// binary_op_name`) onto the RPython flowspace operator name
/// (`rpython/flowspace/operation.py:485-507 add_operator(...)`).
///
/// Rust-side identifiers (`bitand`, `bitor`, `bitxor`, `add_assign`,
/// ...) become the trailing-underscore / `inplace_*` forms RPython
/// registers and `RPythonTyper::translate_op_with_map`
/// (`rtyper.rs:2023-2078`) dispatches on.  Names already matching
/// RPython (`add`, `sub`, `mul`, `mod`, `lshift`, `rshift`, `lt`, ...)
/// pass through unchanged.
///
/// Pyre's short-circuit `and` / `or` (Rust `&&` / `||`) are NOT
/// flowspace operations — Python's `and`/`or` are control flow and
/// `operation.py:475-510` does not register them as binary operators.
///
/// Both frontends desugar `&&` / `||` into
/// `JUMP_IF_FALSE_OR_POP` / `JUMP_IF_TRUE_OR_POP`-shaped control flow
/// before the graph reaches this adapter:
///
/// - The RPython-parity frontend at
///   `flowspace/rust_source/build_flow.rs:1191 lower_short_circuit`.
/// - The legacy `front/ast.rs::Expr::Binary` arm now does the same
///   (commit landed 2026-05-04 on `annrpython`), mirroring the
///   build_flow.rs port: emit `bool(lhs)` + `set_branch` fork + 1-arg
///   join carrying `lhs_raw` (short-circuit) or `rhs_raw` (full eval).
///
/// The fail-loud arm below survives for synthetic graphs (test
/// fixtures, future ad-hoc producers) that inject `OpKind::BinOp {
/// op: "and"/"or", .. }` directly without going through either
/// frontend, plus any future RPython binop opnames that have not yet
/// been ported.
fn normalize_binop_name(pyre_name: &str) -> Result<String, TyperError> {
    let normalized = match pyre_name {
        "bitand" => "and_",
        "bitor" => "or_",
        "bitxor" => "xor",
        "add_assign" => "inplace_add",
        "sub_assign" => "inplace_sub",
        "mul_assign" => "inplace_mul",
        "div_assign" => "inplace_div",
        "mod_assign" => "inplace_mod",
        "bitand_assign" => "inplace_and",
        "bitor_assign" => "inplace_or",
        "bitxor_assign" => "inplace_xor",
        "lshift_assign" => "inplace_lshift",
        "rshift_assign" => "inplace_rshift",
        "and" | "or" => {
            return Err(TyperError::missing_rtype_operation(format!(
                "normalize_binop_name: pyre BinOp `{pyre_name}` has no \
                 flowspace counterpart (operation.py:475-510 does not \
                 register short-circuit `and`/`or` as binary operators; \
                 they are control flow). Frontend must desugar `&&`/`||` \
                 to short-circuit blocks before reaching the rtyper."
            )));
        }
        other => other,
    };
    Ok(normalized.to_string())
}

/// Translate a single legacy `model::SpaceOperation` into zero or more
/// `flowspace::SpaceOperation`s.
///
/// Returns `Ok(Vec::new())` when the op is **fully consumed by other
/// adapter infrastructure** — `OpKind::Input` (handled by Slice 1c
/// block topology, where the result `ValueId` becomes a
/// `block.inputargs` entry) and `OpKind::ConstInt` / `ConstFloat`
/// (handled by [`build_value_to_hlvalue_map`], which inlines the
/// constant at every consuming op's args site).
///
/// Returns `Err(TyperError)` for variants whose lowering is deferred to
/// a Slice 1b followup commit. The error message names the specific
/// variant so Slice 4's dual-gate failure cleanly identifies which
/// followup needs to land.
pub fn translate_op(
    op: &SpaceOperation,
    value_map: &HashMap<ValueId, Hlvalue>,
    // The call registry is consulted by the `OpKind::Call::FunctionPath`
    // arm to resolve a registered `(HostObject, FunctionDesc)` pair
    // and emit a flowspace `simple_call` (`operation.py:152`,
    // `rpbc.rs:1621 FunctionRepr::rtype_simple_call`).  Empty registry
    // callsites surface a distinct fail-loud message; producers
    // must pre-register every reachable FunctionPath.
    call_registry: &crate::translator::rtyper::pyre_call_registry::PyreCallRegistry,
) -> Result<Vec<FlowspaceOp>, TyperError> {
    match &op.kind {
        // ─── Skipped: fully consumed by other adapter infrastructure ───
        OpKind::Input { .. } => Ok(Vec::new()),
        OpKind::ConstInt(_) | OpKind::ConstBool(_) | OpKind::ConstFloat(_) => Ok(Vec::new()),
        // ─── Skipped: pyre JIT trace markers without a flowspace peer ───
        // `GuardTrue` / `GuardFalse` / `GuardValue` are JIT-side
        // assertions emitted by pyre's tracer — they constrain the
        // runtime value of an existing SSA operand and produce no new
        // SSA result.  `VableForce` is a virtualizable-flush hint
        // (`hint_force_virtualizable`, no operands, no result).
        // RPython's flowspace abstract interpreter does not have any
        // of these at the high-level rtyper input
        // (`rpython/flowspace/operation.py:475-510`); the equivalent
        // checks appear later in `pyjitpl` / `codewriter` after the
        // rtyper has already lowered to lltype.  For the type
        // resolution pass driven by `specialize_legacy_graph` they
        // are pure no-ops: skipping them preserves the SSA chain
        // (any operand they read is defined elsewhere; the absence
        // of a result means no consumer is left unsatisfied).
        OpKind::GuardTrue { .. }
        | OpKind::GuardFalse { .. }
        | OpKind::GuardValue { .. }
        | OpKind::VableForce { .. } => Ok(Vec::new()),

        // ─── Pre-rtyper opname normalization ───
        // `binary_op_name` (`front/ast.rs:3227-3258`) emits Rust-side
        // names (`bitand`, `bitor`, `bitxor`, `add_assign`, ...).
        // RPython flowspace registers operators via
        // `add_operator('and_', 2, ...)` etc.
        // (`rpython/flowspace/operation.py:485-507`): `and_`, `or_`,
        // `xor`, `inplace_add`, `inplace_sub`, ...  Translate the
        // pyre-side name to its RPython counterpart so the rtyper's
        // `translate_op` arm matching (`rtyper.rs:2023-2078`) finds
        // the proper `pair_rtype_*` dispatch.
        OpKind::BinOp {
            op: opname,
            lhs,
            rhs,
            ..
        } => {
            let l = lookup_operand(value_map, *lhs, op, "lhs")?;
            let r = lookup_operand(value_map, *rhs, op, "rhs")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                normalize_binop_name(opname)?,
                vec![l, r],
                result,
            )])
        }

        // ─── Pre-rtyper opname normalization for unary ops ───
        // `unary_op_name` (`front/ast.rs:3274-3281`) emits Rust-side
        // names (`neg`, `not`, `deref`).  RPython flowspace registers
        // unary operators via `add_operator('neg', 1, ..)` /
        // `add_operator('invert', 1, ..)` /
        // `add_operator('pos', 1, ..)` etc.
        // (`rpython/flowspace/operation.py:465-474`).  Translate the
        // pyre-side name to its RPython counterpart so the rtyper's
        // unary dispatch (`rtyper.rs:2023-2078 translate_op_*`) finds
        // the proper `unary_rtype_*` arm.
        OpKind::UnaryOp {
            op: opname,
            operand,
            ..
        } => {
            let v = lookup_operand(value_map, *operand, op, "operand")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                normalize_unary_op_name(opname)?,
                vec![v],
                result,
            )])
        }

        // ─── Slice 1b: FieldRead / FieldWrite ports ───
        // RPython `flowspace/operation.py:617 GetAttr.opname = 'getattr'`
        // and `setattr` (operation.py: same module). The high-level
        // attribute-access op carries the field name as a
        // `ConstValue::ByteStr` (Python 2 `str`), matching the rtyper's
        // `rtype_getattr` / `rtype_setattr` dispatch
        // (`rtyper.rs:2013-2014`). InstanceRepr later lowers the
        // `getattr`/`setattr` op into a `getfield_*` / `setfield_*`
        // bytecode keyed on the field's lltype kind.
        OpKind::FieldRead { base, field, .. } => {
            let base_hl = lookup_operand(value_map, *base, op, "base")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                "getattr",
                vec![
                    base_hl,
                    Hlvalue::Constant(Constant::new(ConstValue::byte_str(&field.name))),
                ],
                result,
            )])
        }
        OpKind::FieldWrite {
            base, field, value, ..
        } => {
            let base_hl = lookup_operand(value_map, *base, op, "base")?;
            let value_hl = lookup_operand(value_map, *value, op, "value")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                "setattr",
                vec![
                    base_hl,
                    Hlvalue::Constant(Constant::new(ConstValue::byte_str(&field.name))),
                    value_hl,
                ],
                result,
            )])
        }

        // ─── Slice 1b: ArrayRead / ArrayWrite ports ───
        // RPython `flowspace/operation.py: GetItem.opname = 'getitem'`
        // and `setitem`. The base[index] form maps directly to
        // `getitem(base, index)` / `setitem(base, index, value)`; the
        // rtyper's `rtype_getitem` / `rtype_setitem` later route through
        // ListRepr / TupleRepr / Fixed-array repr based on the receiver's
        // resolved type, lowering to `getarrayitem_gc_*` /
        // `setarrayitem_gc_*` bytecodes.
        OpKind::ArrayRead { base, index, .. } => {
            let base_hl = lookup_operand(value_map, *base, op, "base")?;
            let index_hl = lookup_operand(value_map, *index, op, "index")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                "getitem",
                vec![base_hl, index_hl],
                result,
            )])
        }
        OpKind::ArrayWrite {
            base, index, value, ..
        } => {
            let base_hl = lookup_operand(value_map, *base, op, "base")?;
            let index_hl = lookup_operand(value_map, *index, op, "index")?;
            let value_hl = lookup_operand(value_map, *value, op, "value")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                "setitem",
                vec![base_hl, index_hl, value_hl],
                result,
            )])
        }

        // ─── Slice 1b: InteriorFieldRead / InteriorFieldWrite ports ───
        // RPython `effectinfo.py:313-340` notes that `getinteriorfield_gc`
        // implicitly carries both a `readarray` and a `readinteriorfield`
        // effect — the array-of-structs pattern is fundamentally a
        // chained `getitem(base, index) -> elem` followed by
        // `getattr(elem, field_name)` (or `setattr` for writes). Pyre's
        // legacy IR collapses these into a single `InteriorField*` op
        // for direct lowering convenience, but the rtyper sees the
        // chained form, so unfold here into two flowspace ops with an
        // intermediate `Variable` carrying the array element.
        OpKind::InteriorFieldRead {
            base, index, field, ..
        } => {
            let base_hl = lookup_operand(value_map, *base, op, "base")?;
            let index_hl = lookup_operand(value_map, *index, op, "index")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            let elem_var = Hlvalue::Variable(Variable::new());
            Ok(vec![
                FlowspaceOp::new("getitem", vec![base_hl, index_hl], elem_var.clone()),
                FlowspaceOp::new(
                    "getattr",
                    vec![
                        elem_var,
                        Hlvalue::Constant(Constant::new(ConstValue::byte_str(&field.name))),
                    ],
                    result,
                ),
            ])
        }
        OpKind::InteriorFieldWrite {
            base,
            index,
            field,
            value,
            ..
        } => {
            let base_hl = lookup_operand(value_map, *base, op, "base")?;
            let index_hl = lookup_operand(value_map, *index, op, "index")?;
            let value_hl = lookup_operand(value_map, *value, op, "value")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            let elem_var = Hlvalue::Variable(Variable::new());
            Ok(vec![
                FlowspaceOp::new("getitem", vec![base_hl, index_hl], elem_var.clone()),
                FlowspaceOp::new(
                    "setattr",
                    vec![
                        elem_var,
                        Hlvalue::Constant(Constant::new(ConstValue::byte_str(&field.name))),
                        value_hl,
                    ],
                    result,
                ),
            ])
        }

        // ─── Slice 1b: Call port (CallTarget per variant) ───
        // RPython `flowspace/operation.py:663 SimpleCall.opname =
        // 'simple_call'`. The first arg is a Constant wrapping the
        // callable (or a Variable carrying a runtime function pointer).
        // Each `CallTarget` variant maps to a different shape:
        //
        //   FunctionPath { segments }     — direct call to `path::func`.
        //                                    Wrap the joined qualname in
        //                                    a `HostObject::new_opaque(...)`
        //                                    Constant; rtyper's
        //                                    `rtype_simple_call` dispatches
        //                                    on the callable's resolved
        //                                    repr (PBCRepr / etc.).
        //   SyntheticTransparentCtor      — Rust struct ctor `Class { .. }`.
        //                                    Same shape as FunctionPath:
        //                                    opaque host wrapping the type
        //                                    qualname; the rtyper-equivalent
        //                                    layer routes the ctor to its
        //                                    InstanceRepr.
        //   Method { name, .. }           — `obj.method(args)` — chains
        //                                    `getattr(args[0], name) → meth`
        //                                    into `simple_call(meth, args[1..])`,
        //                                    mirroring `flowspace/
        //                                    flowcontext.py:LOAD_ATTR +
        //                                    CALL_FUNCTION` shape.
        //   Indirect { trait_root, name } — `dyn Trait` dispatch. Pyre's
        //                                    `rclass.rs` rewrites this into
        //                                    a `VtableMethodPtr` followed
        //                                    by an `IndirectCall`; reaching
        //                                    the adapter means rclass.rs
        //                                    didn't run, so fail-loud.
        //   UnsupportedExpr               — frontend coverage gap; fail-loud
        //                                    surfaces the missing
        //                                    `front/ast.rs` arm.
        OpKind::Call { target, args, .. } => {
            use crate::model::CallTarget;
            let arg_hls: Result<Vec<_>, _> = args
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let role = format!("args[{i}]");
                    lookup_operand(value_map, *v, op, &role)
                })
                .collect();
            let arg_hls = arg_hls?;
            let result = resolve_result_hlvalue(op, value_map)?;
            match target {
                // Slice A.3c — `FunctionPath` resolves through
                // `PyreCallRegistry`, returning the registry entry's
                // `HostObject::UserFunction` instead of an opaque
                // wrapper. The rtyper's `pair_simple_call` then
                // short-circuits on `bookkeeper.descs` (pre-populated
                // by the registry) and routes through
                // `FunctionRepr::call(hop)` (`rpbc.py:199`).
                CallTarget::FunctionPath { segments } => {
                    let key =
                        crate::translator::rtyper::pyre_call_registry::FunctionPathKey::from_segments(
                            segments.iter().cloned(),
                        );
                    let entry = call_registry.lookup(&key).ok_or_else(|| {
                        TyperError::message(format!(
                            "translate_op: OpKind::Call::FunctionPath {{ segments: {:?} }} \
                             not registered in PyreCallRegistry — the production \
                             builder (a SemanticProgram walker, or a test fixture \
                             building the registry directly) must register the path \
                             with its parameter Signature before specialize_legacy_graph \
                             consults the rtyper. Result ValueId = {:?}",
                            segments, op.result,
                        ))
                    })?;
                    let callable = Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                        entry.host_object.clone(),
                    )));
                    let mut call_args = Vec::with_capacity(arg_hls.len() + 1);
                    call_args.push(callable);
                    call_args.extend(arg_hls);
                    Ok(vec![FlowspaceOp::new("simple_call", call_args, result)])
                }
                CallTarget::SyntheticTransparentCtor { name } => {
                    // RPython parity: tagged-union ctor `Foo(x)` annotates as
                    // `SomePBC([ClassDesc(Foo)])` then `pair_simple_call`
                    // constructs `SomeInstance(classdef)` (`bookkeeper.py:
                    // 315-316`).  Wrapping the ctor name as
                    // `HostObject::new_class(name, [])` routes through the
                    // existing `is_class()` arm in
                    // [`crate::annotator::bookkeeper::Bookkeeper::immutablevalue_hostobject`]
                    // (`bookkeeper.rs:1984`) → `getdesc` → `ClassDesc::new`
                    // (`classdesc.rs:708`) → `SomePBC([ClassDesc])`, instead
                    // of falling through to the "Don't know how to represent"
                    // error that `HostObject::new_opaque` produces.  The
                    // resulting `SomeInstance(classdef)` projects to
                    // `ConcreteType::GcRef`, matching legacy
                    // `resolve_types(Unknown) → GcRef`.  Post-jtransform
                    // [`crate::jit_codewriter::jtransform`] still unwraps
                    // the simple_call to its inner value (the transparent
                    // semantics survive at the codewriter layer).
                    let callable = Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                        HostObject::new_class(name.clone(), Vec::new()),
                    )));
                    let mut call_args = Vec::with_capacity(arg_hls.len() + 1);
                    call_args.push(callable);
                    call_args.extend(arg_hls);
                    Ok(vec![FlowspaceOp::new("simple_call", call_args, result)])
                }
                CallTarget::Method { name, .. } => {
                    let mut iter = arg_hls.into_iter();
                    let receiver = iter.next().ok_or_else(|| {
                        TyperError::message(
                            "Call::Method has empty args: receiver must be args[0]".to_string(),
                        )
                    })?;
                    let bound_method = Hlvalue::Variable(Variable::new());
                    let mut call_args = Vec::with_capacity(iter.size_hint().0 + 1);
                    call_args.push(bound_method.clone());
                    call_args.extend(iter);
                    Ok(vec![
                        FlowspaceOp::new(
                            "getattr",
                            vec![
                                receiver,
                                Hlvalue::Constant(Constant::new(ConstValue::byte_str(name))),
                            ],
                            bound_method,
                        ),
                        FlowspaceOp::new("simple_call", call_args, result),
                    ])
                }
                CallTarget::Indirect { .. } => Err(TyperError::message(format!(
                    "translate_op: Call with CallTarget::Indirect at result={:?} \
                     must be lowered to VtableMethodPtr + IndirectCall by \
                     rclass.rs before reaching the flowspace adapter; \
                     reaching here means the rclass rewrite didn't run",
                    op.result
                ))),
                CallTarget::UnsupportedExpr => Err(TyperError::message(format!(
                    "translate_op: Call with CallTarget::UnsupportedExpr at \
                     result={:?} — frontend coverage gap; the `front/ast.rs` \
                     arm that emitted this Call must classify the call shape \
                     before the rtyper sees it",
                    op.result
                ))),
            }
        }

        // ─── Pyre-internal: IndirectCall ───
        // RPython `rpython/rtyper/rpbc.py:216-217`:
        // ```python
        // vlist.append(hop.inputconst(Void, row_of_graphs.values()))
        // v = hop.genop('indirect_call', vlist, resulttype=rresult)
        // ```
        // The trailing `c_graphs` Constant must carry actual graph
        // identities — pyre's parity emits `ConstValue::Graphs(Vec<usize>)`
        // via `GraphKey::of(&g.graph).as_usize()` (see
        // `translator/rtyper/rpbc.rs:1481-1490`). The flowspace adapter
        // doesn't have access to the graph registry that resolves
        // `CallPath` segments to `Rc<RefCell<FunctionGraph>>` references,
        // so it cannot construct a faithful `ConstValue::Graphs`. A
        // synthetic `ConstValue::List(byte_str(qualname))` would silently
        // drop indirect-call analysis (`graphanalyze.rs:333` falls back
        // to `top_result()` for any non-Graphs ConstValue), so fail-loud
        // is the parity-correct behaviour: `IndirectCall` must be lowered
        // by `rpbc.rs` (the rtyper-equivalent layer that owns the graph
        // registry) before reaching the flowspace adapter.
        OpKind::IndirectCall { .. } => Err(TyperError::message(format!(
            "translate_op: IndirectCall at result={:?} must be lowered to \
             a flowspace `indirect_call` op with `ConstValue::Graphs(Vec<\
             usize>)` candidate-graph keys by `rpbc.rs:1481-1490` before \
             reaching the adapter; synthesising a `ConstValue::List` here \
             would break `graphanalyze.rs:333` indirect-call analysis",
            op.result
        ))),

        // ─── Pyre-internal: VtableMethodPtr ───
        // TODO(rclass-vtable-rework): pyre-only adaptation of
        // `rclass.py:371-377 getclsfield()`.  Emitted by
        // `translator/rtyper/rclass.rs` to project the function
        // pointer out of a `dyn Trait` receiver's vtable. It exists
        // only *inside* the rtyper pipeline (rclass produces it;
        // jtransform consumes it), so reaching the flowspace adapter
        // input means an rtyper-stage layer missed its own emit/
        // consume invariant.
        OpKind::VtableMethodPtr { .. } => Err(TyperError::message(format!(
            "translate_op: VtableMethodPtr at result={:?} is rtyper-internal \
             (TODO(rclass-vtable-rework) of rclass.py:371-377); rclass.rs \
             emits it and the jtransform layer consumes it before flowspace \
             adapter input — reaching here means the rclass→jtransform \
             pipeline broke",
            op.result
        ))),

        // ─── Stage-invariant fail-loud catch-all ───
        // No remaining variants reach here legitimately: every legitimate
        // pre-rtyper input shape has an explicit arm above, every
        // post-rtyper jtransform-emitted variant is enumerated in
        // `post_rtyper_jtransform_variant_name` and short-circuits with
        // a stage-mismatch message before this fall-through, and every
        // pyre-internal rtyper-cutover variant (`IndirectCall`,
        // `VtableMethodPtr`) has its own targeted fail-loud arm.  Hitting
        // this catch-all means a brand-new `OpKind` was added without
        // updating either the explicit translate arm OR the variant-name
        // table — fail-loud with a clear pointer at where the missing
        // arm should land.
        other => {
            let variant = opkind_variant_name(other);
            if let Some(stage_msg) = post_rtyper_jtransform_variant_name(other) {
                Err(TyperError::message(format!(
                    "translate_op: post-rtyper jtransform variant `{stage_msg}` \
                     reached the flowspace adapter at result={:?}.  RPython \
                     `rpython/jit/codewriter/jtransform.py` runs *after* the \
                     rtyper has lowered every high-level op, so this variant \
                     must NEVER appear at the rtyper input.  Source of the \
                     leak is upstream — check `rpbc.rs` / `rclass.rs` / the \
                     pre-rtyper graph builder for an emit site that should \
                     have produced a pre-rtyper shape (e.g. `FieldRead` / \
                     `ArrayRead` / `Call`) instead of `{variant}`.",
                    op.result,
                )))
            } else {
                Err(TyperError::message(format!(
                    "translate_op: OpKind variant `{variant}` has no \
                     translate arm and no stage-invariant classification.  \
                     A new pyre-internal variant was added to \
                     `model::OpKind` without updating \
                     `flowspace_adapter::translate_op` or \
                     `opkind_variant_name`.  Add an explicit translate arm \
                     above (lower to flowspace) or, if the variant is \
                     post-rtyper-only, list it in \
                     `post_rtyper_jtransform_variant_name` so the \
                     stage-mismatch message fires.  result={:?}",
                    op.result,
                )))
            }
        }
    }
}

/// Stable variant name for fail-loud messages. Matches the RPython
/// convention of identifying ops by their opname stem so Slice 4
/// dual-gate failures are immediately greppable.
fn opkind_variant_name(kind: &OpKind) -> &'static str {
    #[allow(unreachable_patterns)]
    match kind {
        OpKind::Input { .. } => "Input",
        OpKind::ConstInt(_) => "ConstInt",
        OpKind::ConstBool(_) => "ConstBool",
        OpKind::ConstFloat(_) => "ConstFloat",
        OpKind::FieldRead { .. } => "FieldRead",
        OpKind::FieldWrite { .. } => "FieldWrite",
        OpKind::ArrayRead { .. } => "ArrayRead",
        OpKind::ArrayWrite { .. } => "ArrayWrite",
        OpKind::InteriorFieldRead { .. } => "InteriorFieldRead",
        OpKind::InteriorFieldWrite { .. } => "InteriorFieldWrite",
        OpKind::Call { .. } => "Call",
        OpKind::GuardTrue { .. } => "GuardTrue",
        OpKind::GuardFalse { .. } => "GuardFalse",
        OpKind::GuardValue { .. } => "GuardValue",
        OpKind::VtableMethodPtr { .. } => "VtableMethodPtr",
        OpKind::IndirectCall { .. } => "IndirectCall",
        OpKind::BinOp { .. } => "BinOp",
        OpKind::UnaryOp { .. } => "UnaryOp",
        OpKind::VableForce { .. } => "VableForce",
        OpKind::VableFieldRead { .. } => "VableFieldRead",
        OpKind::VableFieldWrite { .. } => "VableFieldWrite",
        OpKind::VableArrayRead { .. } => "VableArrayRead",
        OpKind::VableArrayWrite { .. } => "VableArrayWrite",
        OpKind::CallElidable { .. } => "CallElidable",
        OpKind::CallResidual { .. } => "CallResidual",
        OpKind::CallMayForce { .. } => "CallMayForce",
        OpKind::InlineCall { .. } => "InlineCall",
        OpKind::RecursiveCall { .. } => "RecursiveCall",
        OpKind::JitDebug { .. } => "JitDebug",
        OpKind::AssertGreen { .. } => "AssertGreen",
        OpKind::CurrentTraceLength => "CurrentTraceLength",
        OpKind::IsConstant { .. } => "IsConstant",
        OpKind::IsVirtual { .. } => "IsVirtual",
        OpKind::ConditionalCall { .. } => "ConditionalCall",
        OpKind::ConditionalCallValue { .. } => "ConditionalCallValue",
        OpKind::RecordKnownResult { .. } => "RecordKnownResult",
        OpKind::RecordQuasiImmutField { .. } => "RecordQuasiImmutField",
        OpKind::Live => "Live",
        OpKind::JitMergePoint { .. } => "JitMergePoint",
        OpKind::LoopHeader { .. } => "LoopHeader",
        OpKind::Abort { .. } => "Abort",
        // Catch-all for variants pyre may add without bumping this
        // table — surfaces as `<unknown>` in the fail-loud message
        // rather than a misleading variant tag.  The catch-all message
        // (above) instructs the reader to extend this table.
        _ => "<unknown OpKind variant>",
    }
}

/// Identify whether `kind` is a post-rtyper jtransform-emitted variant
/// (i.e., emitted by `rpython/jit/codewriter/jtransform.py` AFTER the
/// rtyper has lowered every high-level op).  These variants must NEVER
/// reach the flowspace adapter — a leak indicates the upstream pre-
/// rtyper graph builder (or pyre's `rpbc.rs` / `rclass.rs`) emitted the
/// post-rtyper shape directly instead of routing through the rtyper-
/// equivalent lowering step.  Returns `Some(name)` for fail-loud messages.
fn post_rtyper_jtransform_variant_name(kind: &OpKind) -> Option<&'static str> {
    Some(match kind {
        OpKind::VableFieldRead { .. } => "VableFieldRead (jtransform.py:651-927)",
        OpKind::VableFieldWrite { .. } => "VableFieldWrite (jtransform.py:651-927)",
        OpKind::VableArrayRead { .. } => "VableArrayRead (jtransform.py:651-927)",
        OpKind::VableArrayWrite { .. } => "VableArrayWrite (jtransform.py:651-927)",
        OpKind::CallElidable { .. } => "CallElidable (jtransform.py:414-435 rewrite_call)",
        OpKind::CallResidual { .. } => "CallResidual (jtransform.py:414-435 rewrite_call)",
        OpKind::CallMayForce { .. } => "CallMayForce (jtransform.py:414-435 rewrite_call)",
        OpKind::InlineCall { .. } => "InlineCall (jtransform.py:473-482)",
        OpKind::RecursiveCall { .. } => "RecursiveCall (jtransform.py:522-534)",
        OpKind::JitDebug { .. } => "JitDebug (jtransform.py:1731-1743)",
        OpKind::AssertGreen { .. } => "AssertGreen (jtransform.py:1731-1743)",
        OpKind::CurrentTraceLength => "CurrentTraceLength (jtransform.py:1731-1743)",
        OpKind::IsConstant { .. } => "IsConstant (jtransform.py:1731-1743)",
        OpKind::IsVirtual { .. } => "IsVirtual (jtransform.py:1731-1743)",
        OpKind::ConditionalCall { .. } => "ConditionalCall (jtransform.py:1665-1688)",
        OpKind::ConditionalCallValue { .. } => "ConditionalCallValue (jtransform.py:1665-1688)",
        OpKind::RecordKnownResult { .. } => "RecordKnownResult (jtransform.py:292-313)",
        OpKind::RecordQuasiImmutField { .. } => "RecordQuasiImmutField (jtransform.py:901-903)",
        OpKind::Live => "Live (jtransform.py:469,481,533)",
        OpKind::JitMergePoint { .. } => "JitMergePoint (jtransform.py:1690-1718)",
        OpKind::LoopHeader { .. } => "LoopHeader (jtransform.py:1690-1718)",
        // `OpKind::Abort` is pyre-only — RPython raises `FlowingError`
        // (`flowspace/flowcontext.py:258,417`) and drops the function
        // before reaching the rtyper.  No real-path lowering exists;
        // the dual-gate Skip-classifies the graph and the legacy path
        // covers it.  The Slice 10C ratchet that synthesised a
        // `SomeInstance(None) -> Ptr -> GcRef` projection was a
        // NEW-DEVIATION (no upstream peer) and has been reverted —
        // closing this entry requires retiring every Abort emit-site
        // in the front-end (Expr::ForLoop placeholder, etc.) and
        // replacing them with proper RPython-orthodox lowerings.
        OpKind::Abort { .. } => "Abort (pyre-only abort marker)",
        _ => return None,
    })
}

/// Output of [`function_graph_to_flowspace`] — the assembled
/// `flowspace::FunctionGraph` plus enough side tables for Slice 2's
/// `specialize_legacy_graph` wrapper to drive `RPythonTyper::specialize`
/// against pyre's annotator surface and read back per-`ValueId`
/// concretetypes.
#[derive(Debug)]
pub struct FlowspaceAdapterOutput {
    /// Assembled `flowspace::FunctionGraph` carrying every legacy block
    /// translated to a `flowspace::Block` over `Hlvalue` operands.
    /// Wrapped in `Rc<RefCell<_>>` to match RPython's
    /// `FunctionDesc.cache` ownership shape — Slice 2 hands this to
    /// `RPythonAnnotator` directly.
    pub graph: Rc<RefCell<FlowspaceGraph>>,
    /// `ValueId → flowspace::Variable` (Slice 1a output) — Slice 2 reads
    /// `Variable.concretetype` per `ValueId` after `specialize` returns.
    pub value_to_var: ValueIdToVariable,
    /// Legacy constant define `ValueId` -> `Constant.concretetype`.
    /// Materialised at lift time from `OpKind::ConstInt` / `ConstFloat`
    /// via `Constant::with_concretetype` (`flowspace_adapter.rs:518-527`),
    /// matching RPython's `Constant.concretetype` ground truth.  Slice 2
    /// reads the per-`ValueId` `LowLevelType` directly so the projector
    /// does not have to reconstruct the kind from `AnnotationState`.
    pub constant_concretetypes: HashMap<ValueId, LowLevelType>,
    /// `BlockId → flowspace::BlockRef` mapping. Includes the canonical
    /// `returnblock` and `exceptblock` (mapped to the
    /// `FunctionGraph::with_return_var`-allocated final blocks) so any
    /// legacy Link targeting them resolves correctly.
    pub block_map: HashMap<BlockId, BlockRef>,
}

/// Translate a legacy `ExitCase` into the `Hlvalue` slot expected by
/// `flowspace::Link.exitcase`. RPython encodes the discriminating value
/// as a `Hlvalue::Constant` carrying the matched bool / Python value
/// (`flowspace/model.py:114-120`).
fn exitcase_to_hlvalue(exitcase: Option<&ExitCase>) -> Option<Hlvalue> {
    match exitcase {
        None => None,
        Some(ExitCase::Bool(b)) => Some(Hlvalue::Constant(constant_from_constvalue(
            ConstValue::Bool(*b),
        ))),
        Some(ExitCase::Const(cv)) => Some(Hlvalue::Constant(constant_from_constvalue(cv.clone()))),
    }
}

fn constant_from_constvalue(value: ConstValue) -> Constant {
    match value {
        ConstValue::Int(n) => Constant::with_concretetype(ConstValue::Int(n), LowLevelType::Signed),
        ConstValue::Bool(b) => Constant::with_concretetype(ConstValue::Bool(b), LowLevelType::Bool),
        ConstValue::Float(bits) => {
            Constant::with_concretetype(ConstValue::Float(bits), LowLevelType::Float)
        }
        other => Constant::new(other),
    }
}

fn legacy_const_define_hlvalue(op: &SpaceOperation) -> Option<(ValueId, Hlvalue)> {
    let result = op.result?;
    match &op.kind {
        OpKind::ConstInt(n) => Some((
            result,
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(*n),
                LowLevelType::Signed,
            )),
        )),
        OpKind::ConstBool(b) => Some((
            result,
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Bool(*b),
                LowLevelType::Bool,
            )),
        )),
        OpKind::ConstFloat(bits) => Some((
            result,
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Float(*bits),
                LowLevelType::Float,
            )),
        )),
        _ => None,
    }
}

/// Translate a single legacy `LinkArg` into a `Hlvalue`. `LinkArg::Value`
/// resolves through `value_map` (which carries Variable identities for
/// regular operands and inlined constants for `OpKind::ConstInt` /
/// `ConstFloat` defines per Slice 1b-core's
/// [`build_value_to_hlvalue_map`]). `LinkArg::Const` materialises a
/// fresh `Hlvalue::Constant`.
///
/// `source_block_id` / `target_block_id` / `arg_index` carry the
/// surrounding context for fail-loud diagnostics — when the lookup
/// misses, the message embeds the predecessor and successor block ids
/// plus the slot index in `Link.args`, so per-graph diagnosis can
/// locate the broken link without re-traversing the graph.  Mirrors
/// the role-bearing enrichment of `lookup_operand` (variant name +
/// arg role).  The required substring `"undefined operand ValueId"`
/// is preserved verbatim for `is_known_unported`
/// (`cutover.rs:441`).
fn link_arg_to_hlvalue(
    arg: &LinkArg,
    value_map: &HashMap<ValueId, Hlvalue>,
    source_block_id: BlockId,
    target_block_id: BlockId,
    arg_index: usize,
) -> Result<Hlvalue, TyperError> {
    match arg {
        LinkArg::Value(vid) => value_map.get(vid).cloned().ok_or_else(|| {
            TyperError::message(format!(
                "translate_op: undefined operand {vid:?} as Link.args[{arg_index}] entry \
                 (source block {source_block_id:?} -> target block {target_block_id:?}) — \
                 adapter invariant broken (every referenced ValueId must be \
                 defined as a block inputarg or op result)"
            ))
        }),
        LinkArg::Const(cv) => Ok(Hlvalue::Constant(constant_from_constvalue(cv.clone()))),
    }
}

/// Translate an exception-link extra variable.
///
/// RPython `flowspace/model.py:636-642` defines `link.last_exception`
/// and `link.last_exc_value` in the link scope before checking
/// `link.args`; those same Variables may then appear in `link.args`.
/// Pyre's legacy graph represents them as fresh `ValueId`s whose only
/// definition site is the link, so the adapter must materialise them in
/// a per-link map instead of requiring a block-local definition.
fn link_extravar_to_hlvalue(
    arg: &LinkArg,
    value_map: &mut HashMap<ValueId, Hlvalue>,
    value_to_var: &mut ValueIdToVariable,
    annotations: &AnnotationState,
) -> Result<Hlvalue, TyperError> {
    match arg {
        LinkArg::Value(vid) => {
            if let Some(existing) = value_map.get(vid).cloned() {
                return Ok(existing);
            }
            let var = seed_variable(*vid, annotations);
            value_to_var.entry(*vid).or_insert_with(|| var.clone());
            let hlvalue = Hlvalue::Variable(var);
            value_map.insert(*vid, hlvalue.clone());
            Ok(hlvalue)
        }
        LinkArg::Const(cv) => Ok(Hlvalue::Constant(constant_from_constvalue(cv.clone()))),
    }
}

/// Derive per-inputarg `SomeValue` cells for a subject's startblock,
/// preferring the explicit `seed_annotations` source (test-fixture
/// hand-built graphs without front-end Input ops) and falling through
/// to the `OpKind::Input { name, ty }` ops the front-end emits at
/// `front/ast.rs:2107-2125` (self) and `:2168-2184` (typed params).
///
/// Returns one `SomeValue` per `startblock.inputargs` entry, in
/// position order.
///
/// Resolution order per inputarg `vid`:
/// 1. `seed_annotations.get_some_value(vid)` (`Slice 12.2` test entry)
///    — minimal fixtures supply Variable-shape annotations explicitly.
/// 2. Matching `OpKind::Input { ty }` op result == `vid` at the
///    startblock — production graphs from `front/ast.rs`.
/// 3. `seed_annotations.get_value_type(vid)` projected through
///    `valuetype_to_someshell` — covers the
///    `function_graph_to_flowspace_with_seed_annotations` path that
///    `Slice 12.4` flattens onto the per-vid types map without
///    materialising the SomeValue.
///
/// Errors:
///
/// - All three sources miss for an inputarg — front-end producer
///   divergence (every typed param emits the Input op alongside the
///   inputargs registration in the front pass; a missing Input op
///   means the producer wired the inputarg without declaring its
///   type and no seed_annotations entry was supplied either).
/// - `valuetype_to_someshell(ty)` returns `None` for the resolved
///   `ValueType` (only `ValueType::Unknown`) — the inputarg's type
///   is an annotation gap; the helper surfaces it the same way
///   `seed_variable` does (`flowspace_adapter.rs:99-115`).
pub(crate) fn derive_subject_inputcells(
    legacy: &FunctionGraph,
    seed_annotations: Option<&crate::jit_codewriter::annotation_state::AnnotationState>,
) -> Result<Vec<crate::annotator::model::SomeValue>, TyperError> {
    let startblock = &legacy.blocks[legacy.startblock.0];
    let mut input_ty_by_result: HashMap<ValueId, &crate::model::ValueType> = HashMap::new();
    for op in &startblock.operations {
        if let (Some(result), OpKind::Input { ty, .. }) = (op.result, &op.kind) {
            input_ty_by_result.insert(result, ty);
        }
    }
    let mut cells = Vec::with_capacity(startblock.inputargs.len());
    for (idx, vid) in startblock.inputargs.iter().enumerate() {
        // 1. Explicit SomeValue seed (test fixtures and any caller
        //    that wants to bypass the ValueType projection).
        if let Some(seed) = seed_annotations {
            if let Some(rc) = seed.some_values.get(vid) {
                cells.push((**rc).clone());
                continue;
            }
        }
        // 2. Front-end Input op at the startblock.
        if let Some(ty) = input_ty_by_result.get(vid) {
            let shell = valuetype_to_someshell(ty).ok_or_else(|| {
                TyperError::message(format!(
                    "derive_subject_inputcells: startblock.inputargs[{idx}] \
                     ({vid:?}) has `ValueType::{ty:?}` (from Input op) whose \
                     `valuetype_to_someshell` projection is `None` (annotation gap — \
                     only `ValueType::Unknown` lacks a SomeValue shell)"
                ))
            })?;
            cells.push(shell);
            continue;
        }
        // 3. seed_annotations ValueType fallback.
        if let Some(seed) = seed_annotations {
            if let Some(vt) = seed.types.get(vid) {
                let shell = valuetype_to_someshell(vt).ok_or_else(|| {
                    TyperError::message(format!(
                        "derive_subject_inputcells: startblock.inputargs[{idx}] \
                         ({vid:?}) has `ValueType::{vt:?}` (from seed_annotations.types) \
                         whose `valuetype_to_someshell` projection is `None`"
                    ))
                })?;
                cells.push(shell);
                continue;
            }
        }
        return Err(TyperError::message(format!(
            "derive_subject_inputcells: startblock.inputargs[{idx}] \
             ({vid:?}) has no matching `OpKind::Input {{ ty }}` op at \
             the startblock and no `seed_annotations` entry — \
             front-end producer divergence (every typed parameter emits \
             the Input op alongside the inputargs registration; see \
             `front/ast.rs:2107-2184`)"
        )));
    }
    Ok(cells)
}

/// One-way conversion from the legacy `crate::model::FunctionGraph` +
/// `AnnotationState` pair into a `flowspace::FunctionGraph` whose
/// blocks carry `Hlvalue` operands and per-value `SomeValue`
/// annotations on its `Variable`s.
///
/// Two-pass topology assembly:
///
/// 1. **Pass 1** — allocate one `flowspace::BlockRef` per legacy
///    non-final block, allocating fresh `Variable`s for each block's
///    inputargs. Assemble the `flowspace::FunctionGraph` via
///    `FunctionGraph::with_return_var`, supplying the canonical
///    returnblock inputarg so the rtyper's `getreturnvar`
///    (`rtyper.rs:1633-1638`) finds a real return `Variable`.
/// 2. **Pass 2** — for each non-final block, translate `operations` via
///    [`translate_op`], translate `exits` (link args + targets +
///    exitcase) via [`link_arg_to_hlvalue`] / [`exitcase_to_hlvalue`],
///    and translate `exitswitch` via the `value_map`.
///
/// Slice 1c lands the topology assembly. Per-OpKind operation
/// translation depends on Slice 1b followups; Slice 1c uses
/// [`translate_op`] as-is, which means any legacy graph carrying an
/// unported OpKind variant surfaces a fail-loud `TyperError` from this
/// function. Trivial graphs (only `Input` / `ConstInt` / `ConstFloat`
/// op definitions) flow through cleanly.
pub fn function_graph_to_flowspace(
    legacy: &FunctionGraph,
    // Slice A.2 plumbing — see [`translate_op`].
    call_registry: &crate::translator::rtyper::pyre_call_registry::PyreCallRegistry,
) -> Result<FlowspaceAdapterOutput, TyperError> {
    function_graph_to_flowspace_with_seed_annotations(legacy, None, call_registry)
}

/// Slice 12.2 — explicit annotation seed entry for unit-test fixtures
/// that build minimal SSA graphs without `OpKind::Input { ty }` ops.
///
/// Production callers go through [`function_graph_to_flowspace`] and
/// rely on the internal `legacy_annotator::annotate(legacy)` call to
/// recover types from production-shape Input / FieldRead / Call ops
/// the front-end emits.  Tests that hand-roll an SSA graph without
/// those ops must seed the annotator-state explicitly so
/// `seed_variable` can attach `Variable.annotation` shells the rtyper
/// reads at `bindingrepr` time.
///
/// `seed_annotations: None` keeps the production path; `Some(state)`
/// lets the test override the internal computation with hand-built
/// `(ValueId, ValueType)` pairs.
pub fn function_graph_to_flowspace_with_seed_annotations(
    legacy: &FunctionGraph,
    seed_annotations: Option<&AnnotationState>,
    call_registry: &crate::translator::rtyper::pyre_call_registry::PyreCallRegistry,
) -> Result<FlowspaceAdapterOutput, TyperError> {
    // Phase 2 (addpendingblock conversion) — production path no longer
    // pre-seeds `Variable.annotation` from `legacy_annotator::annotate`.
    // Once the cutover entry queues the subject's startblock onto the
    // orthodox `addpendingblock` queue
    // (`cutover.rs:specialize_legacy_graph_with_registry_seed`),
    // `complete_pending_blocks` drives `flowin` which writes
    // `Variable.annotation` for every reachable inputarg and op result.
    // Carrying the legacy pre-seed alongside flowin caused
    // `setbinding: new value does not contain old` panics at
    // `annrpython.rs:459` whenever flowin's `follow_link` computed a
    // narrower annotation (e.g., constant-tracking `SomeInteger{const,
    // nonneg}`) than legacy_annotator's wider lift.
    //
    // Test fixtures keep the explicit `seed_annotations` injection so
    // hand-built minimal SSA graphs that bypass front-end Input ops can
    // still seed the `value_to_var` shells `seed_variable` reads.
    let empty_annotations;
    let annotations: &AnnotationState = match seed_annotations {
        Some(s) => s,
        None => {
            empty_annotations = AnnotationState::new();
            &empty_annotations
        }
    };
    let mut value_to_var: ValueIdToVariable = HashMap::new();
    let mut constant_hlvalues: HashMap<ValueId, Hlvalue> = HashMap::new();
    let mut constant_concretetypes: HashMap<ValueId, LowLevelType> = HashMap::new();

    for legacy_block in &legacy.blocks {
        for legacy_op in &legacy_block.operations {
            if let Some((vid, hlvalue)) = legacy_const_define_hlvalue(legacy_op) {
                if let Hlvalue::Constant(c) = &hlvalue {
                    if let Some(ct) = &c.concretetype {
                        constant_concretetypes.insert(vid, ct.clone());
                    }
                }
                constant_hlvalues.insert(vid, hlvalue);
            }
        }
    }

    // ─── Synthetic void-return placeholder seeding ───
    // `model.rs::set_return(block, None)` (`model.rs:1177`) emits a
    // Link to the canonical returnblock with a fresh
    // `alloc_value()`-allocated ValueId in `args[0]` — a pyre
    // divergence from RPython's `RETURN_VALUE`
    // (`flowcontext.py:687-689`) which carries `Constant(None)` from
    // a preceding `LOAD_CONST None` directly.  The legacy annotator
    // already pre-seeds these synthetic placeholders as
    // `ValueType::Void` (`translator/rtyper/legacy_annotator.rs:
    // 47-62` + `is_synthetic_return_void_value`); the adapter
    // mirrors the same pre-seed by materializing a Variable with
    // the Void shell that `valuetype_to_someshell(&ValueType::Void)`
    // returns (`SomeImpossible`).  `RPythonTyper::specialize` then
    // assigns `Void` LowLevelType, the LL→Concrete projector maps
    // back to `Void`, and the dual-gate matches legacy's `Void`
    // resolution — keeping the synthetic placeholder resolvable in
    // `link_arg_to_hlvalue` without dropping the fail-loud invariant
    // for genuinely-undefined Link.args producers.
    let mut synthetic_void_hlvalues: HashMap<ValueId, Hlvalue> = HashMap::new();
    for legacy_block in &legacy.blocks {
        for link in &legacy_block.exits {
            if link.target != legacy.returnblock {
                continue;
            }
            let Some(LinkArg::Value(vid)) = link.args.first() else {
                continue;
            };
            if !is_synthetic_return_void_value(legacy, *vid) {
                continue;
            }
            let var = Variable::new();
            if let Some(shell) = valuetype_to_someshell(&crate::model::ValueType::Void) {
                *var.annotation.borrow_mut() = Some(Rc::new(shell));
            }
            value_to_var.entry(*vid).or_insert_with(|| var.clone());
            synthetic_void_hlvalues
                .entry(*vid)
                .or_insert(Hlvalue::Variable(var));
        }
    }

    // ──────────────────────────────────────────────────────────────
    // Pass 1 — allocate fresh `flowspace::BlockRef` for every legacy
    // non-final block. The legacy `returnblock` and `exceptblock` are
    // skipped here; `FunctionGraph::with_return_var` allocates the
    // canonical flowspace finals, and the block_map is populated with
    // those after graph construction.
    // ──────────────────────────────────────────────────────────────

    let mut block_map: HashMap<BlockId, BlockRef> = HashMap::new();
    let mut block_inputarg_vars: HashMap<BlockId, HashMap<ValueId, Variable>> = HashMap::new();

    for legacy_block in &legacy.blocks {
        if legacy_block.id == legacy.returnblock || legacy_block.id == legacy.exceptblock {
            continue;
        }
        let mut local_inputs: HashMap<ValueId, Variable> = HashMap::new();
        let mut inputargs: Vec<Hlvalue> = Vec::with_capacity(legacy_block.inputargs.len());
        for &vid in &legacy_block.inputargs {
            let var = seed_variable(vid, annotations);
            value_to_var.entry(vid).or_insert_with(|| var.clone());
            local_inputs.insert(vid, var.clone());
            inputargs.push(Hlvalue::Variable(var));
        }
        block_inputarg_vars.insert(legacy_block.id, local_inputs);
        block_map.insert(legacy_block.id, FlowspaceBlock::shared(inputargs));
    }

    let startblock = block_map.get(&legacy.startblock).cloned().ok_or_else(|| {
        TyperError::message(format!(
            "function_graph_to_flowspace: legacy.startblock {:?} not in legacy.blocks",
            legacy.startblock
        ))
    })?;

    // Resolve the returnblock's inputarg as a fresh final-block
    // Variable. Even when the legacy graph reuses the source ValueId
    // here, RPython's checkgraph treats target inputargs as definitions
    // in the target block, not as the predecessor's Variable object.
    //
    // RPython `flowmodel.py:281 FunctionGraph.getreturnvar(self)`
    // returns `self.returnblock.inputargs[0]` unconditionally — there
    // is no fallback for an empty `inputargs` list, and a malformed
    // graph raises `IndexError` at this site rather than fabricating a
    // fresh Variable.  Pyre's `model::FunctionGraph::with_return_var`
    // (`model.rs:983-988`) builds the returnblock with `inputargs:
    // vec![return_value]` by invariant, so the lookup is guaranteed to
    // succeed; surface the violation as a `TyperError` instead of
    // silently producing a `Variable::new()` placeholder.
    let return_var = legacy
        .blocks
        .iter()
        .find(|b| b.id == legacy.returnblock)
        .and_then(|b| b.inputargs.first().copied())
        .map(|vid| {
            let var = seed_variable(vid, annotations);
            value_to_var.entry(vid).or_insert_with(|| var.clone());
            Hlvalue::Variable(var)
        })
        .ok_or_else(|| {
            TyperError::message(format!(
                "function_graph_to_flowspace: legacy graph {:?} has no \
                 returnblock {:?} with at least one inputarg — \
                 `model::FunctionGraph::with_return_var` (model.rs:983-988) \
                 builds the returnblock with `inputargs: vec![return_value]` \
                 by invariant; matches RPython `flowmodel.py:281 \
                 getreturnvar()` which indexes `returnblock.inputargs[0]` \
                 without a fallback",
                legacy.name, legacy.returnblock,
            ))
        })?;

    let graph = FlowspaceGraph::with_return_var(legacy.name.clone(), startblock, return_var);
    let returnblock_ref = graph.returnblock.clone();
    let exceptblock_ref = graph.exceptblock.clone();

    if let Some(legacy_exceptblock) = legacy.blocks.iter().find(|b| b.id == legacy.exceptblock) {
        if legacy_exceptblock.inputargs.len() == 2 {
            let mut except_inputargs = Vec::with_capacity(2);
            for &vid in &legacy_exceptblock.inputargs {
                let var = seed_variable(vid, annotations);
                value_to_var.entry(vid).or_insert_with(|| var.clone());
                except_inputargs.push(Hlvalue::Variable(var));
            }
            exceptblock_ref.borrow_mut().inputargs = except_inputargs;
        }
    }

    let graph_ref = Rc::new(RefCell::new(graph));

    // Map the canonical finals so any legacy Link targeting them
    // resolves to the flowspace finals constructed above.
    block_map.insert(legacy.returnblock, returnblock_ref);
    block_map.insert(legacy.exceptblock, exceptblock_ref);

    // ──────────────────────────────────────────────────────────────
    // Pass 2 — fill operations + exits + exitswitch for each non-final
    // legacy block. Final blocks (returnblock / exceptblock) are
    // already terminal in flowspace — `mark_final()` was set by
    // `FunctionGraph::with_return_var`.
    // ──────────────────────────────────────────────────────────────

    for legacy_block in &legacy.blocks {
        if legacy_block.id == legacy.returnblock || legacy_block.id == legacy.exceptblock {
            continue;
        }
        let block_ref = block_map[&legacy_block.id].clone();
        let mut value_map = constant_hlvalues.clone();
        // Mirror the legacy annotator's `is_synthetic_return_void_value`
        // pre-seed: any synthetic void-return placeholder ValueId is
        // available in every block's value_map so `link_arg_to_hlvalue`
        // resolves the Link.args[0] entry instead of failing
        // `undefined operand`.
        for (&vid, hlv) in &synthetic_void_hlvalues {
            value_map.entry(vid).or_insert_with(|| hlv.clone());
        }
        let mut name_to_value: HashMap<String, Hlvalue> = HashMap::new();

        if let Some(inputs) = block_inputarg_vars.get(&legacy_block.id) {
            for (&vid, var) in inputs {
                let hlvalue = Hlvalue::Variable(var.clone());
                value_map.insert(vid, hlvalue.clone());
                if let Some(name) = legacy.value_name(vid) {
                    name_to_value.entry(name.to_string()).or_insert(hlvalue);
                }
            }
        }
        for legacy_op in &legacy_block.operations {
            if let (Some(result), OpKind::Input { name, ty: _ }) =
                (legacy_op.result, &legacy_op.kind)
            {
                if legacy_block.inputargs.contains(&result) {
                    if let Some(existing) = value_map.get(&result).cloned() {
                        name_to_value.entry(name.clone()).or_insert(existing);
                    }
                }
            }
        }

        // Translate operations.
        let mut translated_ops: Vec<FlowspaceOp> = Vec::new();
        for legacy_op in &legacy_block.operations {
            if let Some((vid, hlvalue)) = legacy_const_define_hlvalue(legacy_op) {
                value_map.insert(vid, hlvalue.clone());
                if let Some(name) = legacy.value_name(vid) {
                    name_to_value.insert(name.to_string(), hlvalue);
                }
                translated_ops.extend(translate_op(legacy_op, &value_map, call_registry)?);
                continue;
            }

            if let (Some(result), OpKind::Input { name, ty: _ }) =
                (legacy_op.result, &legacy_op.kind)
            {
                if !value_map.contains_key(&result) {
                    if let Some(alias) = name_to_value.get(name).cloned() {
                        // Same-block name match: alias the body `Input`
                        // result to the prior `Hlvalue` for `name`.
                        // Mirrors `front/ast.rs:2127-2156`'s same-block
                        // LOAD_FAST dedup that the front already
                        // enforces (no SSA divergence at this site).
                        if let Hlvalue::Variable(var) = &alias {
                            value_to_var.entry(result).or_insert_with(|| var.clone());
                        }
                        value_map.insert(result, alias);
                    } else {
                        // Cross-block body `Input`: name not in the
                        // current block's `name_to_value`.  RPython
                        // flowspace has no analogue for this — every
                        // local cross-block reference is threaded via
                        // the predecessor `Link.args` and the target
                        // block's `inputargs` (`flowcontext.py:872-884
                        // LOAD_FAST` writes the local into
                        // `self.locals_w`; cross-block reads always go
                        // through the target block's pre-allocated
                        // `inputargs[]`, never via a fresh Variable).
                        //
                        // Fail-loud here surfaces the producer-side
                        // gap: either the target block's inputargs were
                        // not extended to carry `name`, or the
                        // predecessor link's args do not include the
                        // ValueId producer.  The dual-gate at
                        // `cutover.rs:439 is_known_unported` matches
                        // the substring `"adapter cross-block body
                        // Input"` and Skip-classifies the graph,
                        // routing it through `legacy_state` until the
                        // Cat 2.1 cross-block locals threading covers
                        // every shape.
                        return Err(TyperError::message(format!(
                            "translate_op: adapter cross-block body Input — \
                             name {name:?} (result {result:?}) was not threaded \
                             through Link.args / target inputargs by the \
                             predecessor block.  RPython has no body-`Input` \
                             op (flowcontext.py:872-884 LOAD_FAST writes locals \
                             into self.locals_w; cross-block reads go via the \
                             target block's pre-allocated inputargs).  Producer \
                             gap — either Cat 2.1 cross-block locals threading \
                             missed this shape, or the front-end's body-`Input` \
                             emission needs to be extended to predeclare the \
                             name in the predecessor link."
                        )));
                    }
                }
                translated_ops.extend(translate_op(legacy_op, &value_map, call_registry)?);
                continue;
            }

            if let Some(result) = legacy_op.result {
                if !value_map.contains_key(&result) {
                    let var = seed_variable(result, annotations);
                    value_to_var.entry(result).or_insert_with(|| var.clone());
                    value_map.insert(result, Hlvalue::Variable(var));
                }
            }
            translated_ops.extend(translate_op(legacy_op, &value_map, call_registry)?);
            if let Some(result) = legacy_op.result {
                if let Some(name) = legacy.value_name(result) {
                    if let Some(value) = value_map.get(&result).cloned() {
                        name_to_value.insert(name.to_string(), value);
                    }
                }
            }
        }

        // Translate exits.
        let mut translated_exits: Vec<flowspace_model::LinkRef> =
            Vec::with_capacity(legacy_block.exits.len());
        for legacy_link in &legacy_block.exits {
            let mut link_value_map = value_map.clone();
            let last_exception = legacy_link
                .last_exception
                .as_ref()
                .map(|arg| {
                    link_extravar_to_hlvalue(
                        arg,
                        &mut link_value_map,
                        &mut value_to_var,
                        annotations,
                    )
                })
                .transpose()?;
            let last_exc_value = legacy_link
                .last_exc_value
                .as_ref()
                .map(|arg| {
                    link_extravar_to_hlvalue(
                        arg,
                        &mut link_value_map,
                        &mut value_to_var,
                        annotations,
                    )
                })
                .transpose()?;
            let target = block_map.get(&legacy_link.target).cloned().ok_or_else(|| {
                TyperError::message(format!(
                    "function_graph_to_flowspace: legacy link target {:?} not found in \
                     legacy.blocks (block id={:?})",
                    legacy_link.target, legacy_block.id,
                ))
            })?;
            let args: Vec<Hlvalue> = legacy_link
                .args
                .iter()
                .enumerate()
                .map(|(idx, arg)| {
                    link_arg_to_hlvalue(
                        arg,
                        &link_value_map,
                        legacy_block.id,
                        legacy_link.target,
                        idx,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let exitcase = exitcase_to_hlvalue(legacy_link.exitcase.as_ref());
            let mut link = FlowspaceLink::new(args, Some(target), exitcase);
            // RPython `Link.__init__` (`flowspace/model.rs:Link::new`) leaves
            // `llexitcase` unset; `RPythonTyper._convert_link`
            // (`rpython/rtyper/rtyper.py:353-360`) populates it during
            // specialization via `r_case.convert_const(link.exitcase)`.
            // Pre-seeding `llexitcase` here would diverge from upstream
            // `pre-rtyper` graph shape; the rtyper overwrites it anyway.
            link.extravars(last_exception, last_exc_value);
            link.prevblock = Some(Rc::downgrade(&block_ref));
            translated_exits.push(link.into_ref());
        }

        // Translate exitswitch.
        let translated_exitswitch = match &legacy_block.exitswitch {
            None => None,
            Some(ExitSwitch::Value(vid)) => Some(value_map.get(vid).cloned().ok_or_else(|| {
                // Inline counterpart of `lookup_operand` for the
                // block.exitswitch path (no enclosing
                // SpaceOperation). Required substring
                // `"undefined operand ValueId"` is preserved
                // verbatim for `is_known_unported`
                // (`cutover.rs:441`).
                TyperError::message(format!(
                    "translate_op: undefined operand {vid:?} as block.exitswitch — \
                         adapter invariant broken (every referenced ValueId must be \
                         defined as a block inputarg or op result)"
                ))
            })?),
            Some(ExitSwitch::LastException) => Some(Hlvalue::Constant(c_last_exception())),
        };

        // Commit to the flowspace::Block. Borrow scope is tight to
        // avoid alias-clash with link.prevblock's Weak above.
        {
            let mut block = block_ref.borrow_mut();
            block.operations = translated_ops;
            block.exits = translated_exits;
            block.exitswitch = translated_exitswitch;
        }
    }

    flowspace_model::checkgraph(&graph_ref.borrow());

    Ok(FlowspaceAdapterOutput {
        graph: graph_ref,
        value_to_var,
        constant_concretetypes,
        block_map,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::bookkeeper::Bookkeeper;
    use crate::annotator::model::{KnownType, SomeObjectTrait, SomeValue};
    use crate::model::{
        Block, BlockId, FunctionGraph as LegacyGraph, OpKind, SpaceOperation, ValueType,
    };
    use crate::translator::rtyper::pyre_call_registry::PyreCallRegistry;

    /// Helper: empty `PyreCallRegistry` for tests that don't exercise
    /// the Slice A.2/A.3 Call resolution path.  The registry's
    /// bookkeeper is freshly minted because translate_op tests don't
    /// share state with an enclosing annotator.
    fn empty_call_registry() -> PyreCallRegistry {
        PyreCallRegistry::new(Rc::new(Bookkeeper::new()))
    }

    #[test]
    fn valuetype_int_lifts_to_someinteger_default() {
        let s = valuetype_to_someshell(&ValueType::Int).expect("Int must project");
        match s {
            SomeValue::Integer(i) => {
                assert_eq!(i.knowntype(), KnownType::Int);
                assert!(!i.nonneg, "default SomeInteger.nonneg must be false");
                assert!(!i.unsigned, "default SomeInteger.unsigned must be false");
            }
            other => panic!("ValueType::Int must lift to SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn valuetype_float_lifts_to_somefloat_default() {
        let s = valuetype_to_someshell(&ValueType::Float).expect("Float must project");
        match s {
            SomeValue::Float(f) => {
                assert_eq!(f.knowntype(), KnownType::Float);
                assert!(f.immutable(), "SomeFloat is immutable per model.py:164-183");
            }
            other => panic!("ValueType::Float must lift to SomeFloat, got {other:?}"),
        }
    }

    #[test]
    fn valuetype_ref_lifts_to_someinstance_classdef_none() {
        let s = valuetype_to_someshell(&ValueType::Ref).expect("Ref must project");
        match s {
            SomeValue::Instance(inst) => {
                assert!(
                    inst.classdef.is_none(),
                    "Ref placeholder must carry classdef=None (model.py:438 default)"
                );
                assert!(
                    !inst.can_be_none,
                    "SomeInstance.__init__ default `can_be_None=False` (model.py:438)"
                );
                assert!(
                    inst.flags.is_empty(),
                    "SomeInstance.__init__ default `flags={{}}` (model.py:438)"
                );
            }
            other => panic!(
                "ValueType::Ref must lift to SomeValue::Instance(classdef=None), got {other:?}"
            ),
        }
    }

    #[test]
    fn valuetype_void_lifts_to_impossible_lattice_bottom() {
        let s = valuetype_to_someshell(&ValueType::Void).expect("Void must project");
        assert!(
            matches!(s, SomeValue::Impossible),
            "ValueType::Void must lift to SomeValue::Impossible, got {s:?}"
        );
    }

    #[test]
    fn valuetype_state_lifts_to_someinstance_classdef_none() {
        // TODO(state-shell): pyre-only `State` (JIT state pointer)
        // — temporary fallback shell.
        let s = valuetype_to_someshell(&ValueType::State).expect("State must project");
        match s {
            SomeValue::Instance(inst) => {
                assert!(
                    inst.classdef.is_none(),
                    "State placeholder must carry classdef=None"
                );
            }
            other => panic!(
                "ValueType::State must lift to SomeValue::Instance(classdef=None), got {other:?}"
            ),
        }
    }

    #[test]
    fn valuetype_unknown_returns_none_for_failloud_at_bindingrepr() {
        // Cat 2.4 fix: `Unknown` is an annotation gap with no
        // annotation-stage shell.  Returning `None` leaves
        // `Variable.annotation` empty so `bindingrepr` panics with
        // `KeyError: no binding for arg`
        // (`annotator/annrpython.rs:418`), surfacing the producer-side
        // gap instead of silently bridging it to `GcRef` via a
        // fabricated `SomeInstance(None)` shell — that bridging
        // conflated the annotation-stage lattice node with the
        // **legacy** `resolve_types(Unknown) -> ConcreteType::Unknown
        // -> GcRef` resolver-stage backfill.
        assert!(
            valuetype_to_someshell(&ValueType::Unknown).is_none(),
            "ValueType::Unknown must return None — annotation gap, no \
             annotation-stage shell"
        );
    }

    #[test]
    fn seed_variable_attaches_lifted_annotation_observable_via_clone() {
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(7), ValueType::Int);
        let var = seed_variable(ValueId(7), &annotations);

        // Reference semantics: the annotation Rc-shares across clones
        // (flowspace/model.rs:2010-2018), so a clone observes the same
        // shell instance.
        let clone = var.clone();
        let clone_ann = clone.annotation.borrow();
        let shell = clone_ann
            .as_ref()
            .expect("seed_variable must populate annotation slot");
        assert!(
            matches!(shell.as_ref(), SomeValue::Integer(_)),
            "annotation must round-trip the lifted SomeInteger shell"
        );
    }

    #[test]
    fn seed_variable_unknown_value_id_leaves_annotation_empty_for_failloud() {
        // Cat 2.4 fix: missing entries in AnnotationState.types resolve
        // to ValueType::Unknown via AnnotationState::get
        // (annotation_state.rs:117). The adapter must NOT fabricate a
        // SomeInstance(classdef=None) shell for these — that would
        // silently bridge an annotation gap to GcRef via the
        // resolver-stage backfill at the wrong layer. Instead, leave
        // Variable.annotation empty so `bindingrepr` panics with
        // `KeyError: no binding for arg`
        // (annotator/annrpython.rs:418), surfacing the producer-side
        // gap as a fail-loud signal.
        let annotations = AnnotationState::new();
        let var = seed_variable(ValueId(42), &annotations);
        let ann = var.annotation.borrow();
        assert!(
            ann.is_none(),
            "Unknown ValueId must leave annotation empty (Cat 2.4 fail-loud), \
             got {:?}",
            ann.as_ref()
        );
    }

    fn legacy_graph_with_inputarg_and_result(input: ValueId, result: ValueId) -> LegacyGraph {
        let mut graph = LegacyGraph::new("test");
        let mut block = Block {
            id: BlockId(0),
            inputargs: vec![input],
            operations: vec![SpaceOperation {
                result: Some(result),
                kind: OpKind::ConstInt(0),
            }],
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];
        graph
    }

    #[test]
    fn build_value_to_variable_map_seeds_inputargs_and_op_results() {
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Ref);
        let graph = legacy_graph_with_inputarg_and_result(ValueId(1), ValueId(2));

        let map = build_value_to_variable_map(&graph, &annotations);

        assert_eq!(
            map.len(),
            2,
            "map must seed both the inputarg and the op result"
        );
        assert!(
            matches!(
                map[&ValueId(1)]
                    .annotation
                    .borrow()
                    .as_ref()
                    .map(|s| s.as_ref().clone()),
                Some(SomeValue::Integer(_))
            ),
            "inputarg ValueId(1) (Int) must be seeded with SomeInteger"
        );
        assert!(
            matches!(
                map[&ValueId(2)]
                    .annotation
                    .borrow()
                    .as_ref()
                    .map(|s| s.as_ref().clone()),
                Some(SomeValue::Instance(_))
            ),
            "op-result ValueId(2) (Ref) must be seeded with SomeInstance(classdef=None)"
        );
    }

    #[test]
    fn build_value_to_variable_map_dedupes_by_value_id() {
        // Two ops both reading the same inputarg (legacy graphs are SSA
        // — every ValueId has one definition, but multiple readers).
        // Slice 1a must produce one Variable identity per ValueId.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);
        annotations.set(ValueId(3), ValueType::Int);

        let mut graph = LegacyGraph::new("dedup_test");
        let mut block = Block {
            id: BlockId(0),
            inputargs: vec![ValueId(1)],
            operations: vec![
                SpaceOperation {
                    result: Some(ValueId(2)),
                    kind: OpKind::ConstInt(7),
                },
                SpaceOperation {
                    result: Some(ValueId(3)),
                    kind: OpKind::ConstInt(11),
                },
            ],
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];

        let map = build_value_to_variable_map(&graph, &annotations);

        assert_eq!(map.len(), 3, "three distinct ValueIds → three Variables");
        // The identity invariant: the inputarg's Variable is one fresh
        // identity, the two op results are two more fresh identities, and
        // they don't collide.
        assert_ne!(map[&ValueId(1)], map[&ValueId(2)]);
        assert_ne!(map[&ValueId(1)], map[&ValueId(3)]);
        assert_ne!(map[&ValueId(2)], map[&ValueId(3)]);
    }

    #[test]
    fn build_value_to_variable_map_aliases_input_rebind_to_inputarg() {
        // Pyre's surface front emits a leading `Input{name}` op whose
        // result IS a block.inputarg, plus follow-up `Input{same name}`
        // ops with FRESH result ValueIds for body-side rebinds. The
        // adapter must alias the rebind result to the canonical
        // inputarg Variable so `setup_block_entry`'s
        // `concretetype` write reaches both — otherwise the body's
        // BinOp lookup hits a fresh Variable with no concretetype and
        // trips genop's "wrong level!" assertion.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int); // rebind of same name

        let mut graph = LegacyGraph::new("rebind_alias");
        let mut block = Block {
            id: BlockId(0),
            inputargs: vec![ValueId(1)],
            operations: vec![
                // Leading definition: result IS the inputarg.
                SpaceOperation {
                    result: Some(ValueId(1)),
                    kind: OpKind::Input {
                        name: "x".to_string(),
                        ty: ValueType::Int,
                    },
                },
                // Rebind: result is fresh; same name → alias to ValueId(1)'s Variable.
                SpaceOperation {
                    result: Some(ValueId(2)),
                    kind: OpKind::Input {
                        name: "x".to_string(),
                        ty: ValueType::Int,
                    },
                },
            ],
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];

        let map = build_value_to_variable_map(&graph, &annotations);
        assert_eq!(
            map[&ValueId(1)],
            map[&ValueId(2)],
            "Input rebind result must alias to inputarg Variable identity"
        );
    }

    // ───── Slice 1b-core tests: dispatcher + skip arms + fail-loud ─────

    #[test]
    fn build_value_to_hlvalue_map_inlines_const_defines() {
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);
        annotations.set(ValueId(3), ValueType::Float);

        let mut graph = LegacyGraph::new("const_inline");
        let mut block = Block {
            id: BlockId(0),
            inputargs: vec![ValueId(1)],
            operations: vec![
                SpaceOperation {
                    result: Some(ValueId(2)),
                    kind: OpKind::ConstInt(42),
                },
                SpaceOperation {
                    result: Some(ValueId(3)),
                    kind: OpKind::ConstFloat(0xC000_0000_0000_0000), // f64::from_bits → -2.0
                },
            ],
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];

        let var_map = build_value_to_variable_map(&graph, &annotations);
        let hl_map = build_value_to_hlvalue_map(&graph, &var_map);

        // Inputarg keeps its Variable identity.
        assert!(matches!(hl_map[&ValueId(1)], Hlvalue::Variable(_)));

        // ConstInt define is inlined as Hlvalue::Constant(Int).
        match &hl_map[&ValueId(2)] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Int(n) => assert_eq!(*n, 42),
                other => panic!("ValueId(2) must be ConstValue::Int, got {other:?}"),
            },
            other => panic!("ValueId(2) must be inlined as Hlvalue::Constant, got {other:?}"),
        }

        // ConstFloat define is inlined as Hlvalue::Constant(Float).
        match &hl_map[&ValueId(3)] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Float(bits) => assert_eq!(*bits, 0xC000_0000_0000_0000),
                other => panic!("ValueId(3) must be ConstValue::Float, got {other:?}"),
            },
            other => panic!("ValueId(3) must be inlined as Hlvalue::Constant, got {other:?}"),
        }
    }

    #[test]
    fn translate_op_skips_input_define() {
        let value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        let op = SpaceOperation {
            result: Some(ValueId(1)),
            kind: OpKind::Input {
                name: "x".to_string(),
                ty: ValueType::Int,
            },
        };
        let result = translate_op(&op, &value_map, &empty_call_registry())
            .expect("Input must translate to skip");
        assert!(
            result.is_empty(),
            "Input define has no SpaceOperation analogue (handled by Slice 1c \
             via block.inputargs); translate_op must yield empty Vec"
        );
    }

    #[test]
    fn derive_subject_inputcells_projects_each_typed_input_op() {
        let mut graph = LegacyGraph::new("subject");
        let entry = graph.startblock;
        let x_vid = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "x".to_string(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let y_vid = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "y".to_string(),
                    ty: ValueType::Float,
                },
                true,
            )
            .unwrap();
        let z_vid = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "z".to_string(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.block_mut(entry).inputargs.push(x_vid);
        graph.block_mut(entry).inputargs.push(y_vid);
        graph.block_mut(entry).inputargs.push(z_vid);

        let cells = derive_subject_inputcells(&graph, None)
            .expect("typed Input ops must project to definite SomeValue cells");
        assert_eq!(cells.len(), 3);
        assert!(
            matches!(cells[0], SomeValue::Integer(_)),
            "x: Int -> SomeInteger, got {:?}",
            cells[0],
        );
        assert!(
            matches!(cells[1], SomeValue::Float(_)),
            "y: Float -> SomeFloat, got {:?}",
            cells[1],
        );
        assert!(
            matches!(cells[2], SomeValue::Instance(_)),
            "z: Ref -> SomeInstance(classdef=None), got {:?}",
            cells[2],
        );
    }

    #[test]
    fn derive_subject_inputcells_fails_loud_on_inputarg_without_input_op() {
        let mut graph = LegacyGraph::new("subject");
        let entry = graph.startblock;
        let orphan = graph.alloc_value();
        graph.block_mut(entry).inputargs.push(orphan);
        let err = derive_subject_inputcells(&graph, None)
            .expect_err("inputarg without matching Input op must surface as TyperError");
        let msg = format!("{err}");
        assert!(
            msg.contains("no matching `OpKind::Input"),
            "error must name the missing Input op invariant, got: {msg}"
        );
    }

    #[test]
    fn derive_subject_inputcells_fails_loud_on_unknown_valuetype() {
        let mut graph = LegacyGraph::new("subject");
        let entry = graph.startblock;
        let vid = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "u".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.block_mut(entry).inputargs.push(vid);
        let err = derive_subject_inputcells(&graph, None)
            .expect_err("ValueType::Unknown has no SomeValue projection");
        let msg = format!("{err}");
        assert!(
            msg.contains("Unknown") && msg.contains("None"),
            "error must mention Unknown + missing projection, got: {msg}"
        );
    }

    #[test]
    fn translate_op_skips_const_int_define() {
        let value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        let op = SpaceOperation {
            result: Some(ValueId(1)),
            kind: OpKind::ConstInt(7),
        };
        let result = translate_op(&op, &value_map, &empty_call_registry())
            .expect("ConstInt must translate to skip");
        assert!(
            result.is_empty(),
            "ConstInt define is inlined by build_value_to_hlvalue_map; \
             translate_op must yield empty Vec"
        );
    }

    #[test]
    fn translate_op_skips_const_float_define() {
        let value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        let op = SpaceOperation {
            result: Some(ValueId(1)),
            kind: OpKind::ConstFloat(0),
        };
        let result = translate_op(&op, &value_map, &empty_call_registry())
            .expect("ConstFloat must translate to skip");
        assert!(result.is_empty());
    }

    #[test]
    fn translate_op_binop_lowers_to_passthrough_spaceop() {
        // BinOp arm: `add` / `sub` / `lt` / ... pass through to a
        // flowspace SpaceOperation with the same opname; lhs/rhs args
        // get resolved via lookup_operand and the result Hlvalue via
        // resolve_result_hlvalue.
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        let lhs_var = Hlvalue::Variable(Variable::new());
        let rhs_var = Hlvalue::Variable(Variable::new());
        let result_var = Hlvalue::Variable(Variable::new());
        value_map.insert(ValueId(1), lhs_var.clone());
        value_map.insert(ValueId(2), rhs_var.clone());
        value_map.insert(ValueId(3), result_var.clone());

        let op = SpaceOperation {
            result: Some(ValueId(3)),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: ValueId(1),
                rhs: ValueId(2),
                result_ty: ValueType::Int,
            },
        };
        let translated =
            translate_op(&op, &value_map, &empty_call_registry()).expect("BinOp arm must lower");
        assert_eq!(translated.len(), 1, "BinOp lowers to exactly one SpaceOp");
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "add", "opname passes through unchanged");
        assert_eq!(lowered.args.len(), 2);
    }

    #[test]
    fn translate_op_binop_undefined_lhs_surfaces_invariant_break() {
        // BinOp arm threads operand lookups through lookup_operand, so a
        // missing lhs surfaces the "adapter invariant broken" message
        // rather than a silent panic.
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        value_map.insert(ValueId(2), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(3), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(ValueId(3)),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: ValueId(99), // not in value_map
                rhs: ValueId(2),
                result_ty: ValueType::Int,
            },
        };
        let err = translate_op(&op, &value_map, &empty_call_registry())
            .expect_err("undefined BinOp operand must surface invariant break");
        let msg = format!("{err}");
        assert!(msg.contains("undefined operand"));
    }

    #[test]
    fn translate_op_call_function_path_lowers_to_simple_call() {
        // Call::FunctionPath → `simple_call(callable_host, args...)` per
        // `flowspace/operation.py:663 SimpleCall.opname = 'simple_call'`.
        // The callable Constant wraps the `PyreCallRegistry` entry's
        // synthetic `HostObject::UserFunction` so the rtyper's
        // `bookkeeper.getdesc` short-circuits onto the registered
        // FunctionDesc.
        use crate::flowspace::argument::Signature;
        use crate::translator::rtyper::pyre_call_registry::FunctionPathKey;
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        value_map.insert(ValueId(1), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(2), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(ValueId(2)),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["a".into(), "b".into()],
                },
                args: vec![ValueId(1)],
                result_ty: ValueType::Int,
            },
        };
        let registry = empty_call_registry();
        registry.get_or_register(
            FunctionPathKey::from_segments(["a", "b"]),
            Signature::new(vec!["x".into()], None, None),
        );
        let translated =
            translate_op(&op, &value_map, &registry).expect("Call::FunctionPath must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "simple_call");
        assert_eq!(lowered.args.len(), 2, "callable + 1 arg");
        let Hlvalue::Constant(ref callable) = lowered.args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("FunctionPath callable must be ConstValue::HostObject");
        };
        // Synthetic GraphFunc takes the last path segment as its
        // `__name__`, mirroring upstream `func.__name__` (the leaf
        // identifier, not the dotted module path).
        assert_eq!(host.qualname(), "b");
    }

    #[test]
    fn translate_op_call_synthetic_transparent_ctor_lowers_to_simple_call() {
        // Call::SyntheticTransparentCtor mirrors Rust's `Class { fields }`
        // ctor — flowspace receives a `simple_call(class_const, fields)`
        // shape just like FunctionPath; rtyper's InstanceRepr handles it.
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        value_map.insert(ValueId(1), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(2), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(ValueId(2)),
            kind: OpKind::Call {
                target: crate::model::CallTarget::SyntheticTransparentCtor {
                    name: "Point".into(),
                },
                args: vec![ValueId(1)],
                result_ty: ValueType::Ref,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("Call::SyntheticTransparentCtor must lower");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "simple_call");
        let Hlvalue::Constant(ref callable) = translated[0].args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("ctor callable must be ConstValue::HostObject");
        };
        assert_eq!(host.qualname(), "Point");
    }

    #[test]
    fn translate_op_call_method_chains_getattr_simple_call() {
        // Call::Method `obj.method(args)` → 2-op chain `[getattr(obj,
        // "method") -> meth, simple_call(meth, args[1..])]`, mirroring
        // `flowspace/flowcontext.py: LOAD_ATTR + CALL_FUNCTION` shape.
        // args[0] is the receiver (matches Rust method-call lowering).
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        value_map.insert(ValueId(1), Hlvalue::Variable(Variable::new())); // receiver
        value_map.insert(ValueId(2), Hlvalue::Variable(Variable::new())); // arg
        value_map.insert(ValueId(3), Hlvalue::Variable(Variable::new())); // result
        let op = SpaceOperation {
            result: Some(ValueId(3)),
            kind: OpKind::Call {
                target: crate::model::CallTarget::Method {
                    name: "push".into(),
                    receiver_root: Some("Vec".into()),
                },
                args: vec![ValueId(1), ValueId(2)],
                result_ty: ValueType::Int,
            },
        };
        let translated =
            translate_op(&op, &value_map, &empty_call_registry()).expect("Call::Method must lower");
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].opname, "getattr");
        assert_eq!(translated[1].opname, "simple_call");
        // getattr's name arg is the method name as a byte string.
        let Hlvalue::Constant(ref name_const) = translated[0].args[1] else {
            panic!("getattr method-name arg must be a Constant");
        };
        assert!(matches!(
            name_const.value,
            ConstValue::ByteStr(ref bytes) if bytes == b"push"
        ));
        // Bound-method Variable identity threads from getattr.result into
        // simple_call.args[0].
        let Hlvalue::Variable(ref m1) = translated[0].result else {
            panic!("getattr result must be Variable");
        };
        let Hlvalue::Variable(ref m2) = translated[1].args[0] else {
            panic!("simple_call's first arg must be Variable (bound method)");
        };
        assert_eq!(
            m1.id(),
            m2.id(),
            "bound method Variable identity must thread"
        );
        // simple_call args = [bound_method, args[1..]]
        assert_eq!(translated[1].args.len(), 2);
    }

    #[test]
    fn translate_op_call_indirect_surfaces_rclass_invariant() {
        // Call::Indirect must be lowered to VtableMethodPtr +
        // IndirectCall by `rclass.rs` before reaching the flowspace
        // adapter. Reaching here means the rclass rewrite didn't run;
        // surface the structural invariant break.
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        value_map.insert(ValueId(1), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(2), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(ValueId(2)),
            kind: OpKind::Call {
                target: crate::model::CallTarget::Indirect {
                    trait_root: "MyTrait".into(),
                    method_name: "do_it".into(),
                },
                args: vec![ValueId(1)],
                result_ty: ValueType::Int,
            },
        };
        let err = translate_op(&op, &value_map, &empty_call_registry())
            .expect_err("Call::Indirect must surface rclass invariant break");
        let msg = format!("{err}");
        assert!(
            msg.contains("Indirect") && msg.contains("rclass"),
            "fail-loud must cite Indirect + rclass.rs, got: {msg}"
        );
    }

    #[test]
    fn translate_op_indirect_call_surfaces_rpbc_invariant() {
        // OpKind::IndirectCall must be lowered by `rpbc.rs:1481-1490`
        // (the rtyper-equivalent layer that owns the graph registry
        // and can resolve CallPath → ConstValue::Graphs(Vec<usize>))
        // before reaching the flowspace adapter. Synthesising
        // `ConstValue::List(byte_str)` here would break
        // `graphanalyze.rs:333` indirect-call analysis (any non-Graphs
        // ConstValue falls back to `top_result()`); fail-loud is the
        // parity-correct behaviour.
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        value_map.insert(ValueId(1), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(2), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(3), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(ValueId(3)),
            kind: OpKind::IndirectCall {
                funcptr: ValueId(1),
                args: vec![ValueId(2)],
                graphs: None,
                result_ty: ValueType::Int,
            },
        };
        let err = translate_op(&op, &value_map, &empty_call_registry())
            .expect_err("IndirectCall must surface rpbc.rs invariant break");
        let msg = format!("{err}");
        assert!(
            msg.contains("IndirectCall") && msg.contains("rpbc.rs"),
            "fail-loud must cite IndirectCall + rpbc.rs:1481, got: {msg}"
        );
    }

    #[test]
    fn translate_op_field_read_lowers_to_getattr() {
        // FieldRead → flowspace `getattr(base, ConstValue::ByteStr(name))`
        // mirroring `flowspace/operation.py:617 GetAttr.opname = 'getattr'`.
        // The rtyper later dispatches via `rtype_getattr` based on the
        // base operand's resolved repr (InstanceRepr / etc.).
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        let base_var = Hlvalue::Variable(Variable::new());
        let result_var = Hlvalue::Variable(Variable::new());
        value_map.insert(ValueId(1), base_var.clone());
        value_map.insert(ValueId(2), result_var.clone());

        let op = SpaceOperation {
            result: Some(ValueId(2)),
            kind: OpKind::FieldRead {
                base: ValueId(1),
                field: crate::model::FieldDescriptor::new("f", Some("Owner".into())),
                ty: ValueType::Int,
                pure: false,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("FieldRead arm must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "getattr");
        assert_eq!(lowered.args.len(), 2);
        let Hlvalue::Constant(ref name_const) = lowered.args[1] else {
            panic!("FieldRead lowering must pass field name as Hlvalue::Constant");
        };
        assert!(matches!(
            name_const.value,
            ConstValue::ByteStr(ref bytes) if bytes == b"f"
        ));
    }

    #[test]
    fn translate_op_field_write_lowers_to_setattr() {
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        value_map.insert(ValueId(1), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(2), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: ValueId(1),
                field: crate::model::FieldDescriptor::new("g", Some("Owner".into())),
                value: ValueId(2),
                ty: ValueType::Int,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("FieldWrite arm must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "setattr");
        assert_eq!(lowered.args.len(), 3);
        let Hlvalue::Constant(ref name_const) = lowered.args[1] else {
            panic!("FieldWrite lowering must pass field name as Hlvalue::Constant");
        };
        assert!(matches!(
            name_const.value,
            ConstValue::ByteStr(ref bytes) if bytes == b"g"
        ));
    }

    #[test]
    fn translate_op_array_read_lowers_to_getitem() {
        // ArrayRead → flowspace `getitem(base, index)` mirroring
        // `flowspace/operation.py: GetItem.opname = 'getitem'`. RTyper's
        // `rtype_getitem` dispatches via the receiver's resolved repr
        // (ListRepr / TupleRepr / FixedSizeArrayRepr) and lowers to
        // `getarrayitem_gc_*`.
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        value_map.insert(ValueId(1), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(2), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(3), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(ValueId(3)),
            kind: OpKind::ArrayRead {
                base: ValueId(1),
                index: ValueId(2),
                item_ty: ValueType::Int,
                array_type_id: None,
                nolength: false,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("ArrayRead arm must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "getitem");
        assert_eq!(lowered.args.len(), 2);
    }

    #[test]
    fn translate_op_array_write_lowers_to_setitem() {
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        value_map.insert(ValueId(1), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(2), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(3), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: None,
            kind: OpKind::ArrayWrite {
                base: ValueId(1),
                index: ValueId(2),
                value: ValueId(3),
                item_ty: ValueType::Int,
                array_type_id: None,
                nolength: false,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("ArrayWrite arm must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "setitem");
        assert_eq!(lowered.args.len(), 3);
    }

    #[test]
    fn translate_op_interior_field_read_unfolds_to_getitem_getattr_chain() {
        // InteriorFieldRead → `getitem(base, index)` chained into
        // `getattr(elem, field_name)`, mirroring `effectinfo.py:313-340`'s
        // implicit `readarray + readinteriorfield` effects. Two flowspace
        // ops surface from one legacy op; the rtyper sees the chain.
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        value_map.insert(ValueId(1), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(2), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(3), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(ValueId(3)),
            kind: OpKind::InteriorFieldRead {
                base: ValueId(1),
                index: ValueId(2),
                field: crate::model::FieldDescriptor::new("x", Some("Point".into())),
                item_ty: ValueType::Int,
                array_type_id: None,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("InteriorFieldRead arm must lower");
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].opname, "getitem");
        assert_eq!(translated[0].args.len(), 2);
        assert_eq!(translated[1].opname, "getattr");
        assert_eq!(translated[1].args.len(), 2);
        // The second op's first arg must be the element variable
        // produced by the first op.
        let Hlvalue::Variable(ref elem_v1) = translated[0].result else {
            panic!("getitem result must be a Variable");
        };
        let Hlvalue::Variable(ref elem_v2) = translated[1].args[0] else {
            panic!("getattr base arg must be the chained element Variable");
        };
        assert_eq!(
            elem_v1.id(),
            elem_v2.id(),
            "elem Variable identity must thread"
        );
    }

    #[test]
    fn translate_op_interior_field_write_unfolds_to_getitem_setattr_chain() {
        let mut value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        value_map.insert(ValueId(1), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(2), Hlvalue::Variable(Variable::new()));
        value_map.insert(ValueId(3), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: None,
            kind: OpKind::InteriorFieldWrite {
                base: ValueId(1),
                index: ValueId(2),
                field: crate::model::FieldDescriptor::new("y", Some("Point".into())),
                value: ValueId(3),
                item_ty: ValueType::Int,
                array_type_id: None,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("InteriorFieldWrite arm must lower");
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].opname, "getitem");
        assert_eq!(translated[1].opname, "setattr");
        assert_eq!(translated[1].args.len(), 3);
    }

    #[test]
    fn translate_op_undefined_operand_surfaces_invariant_break() {
        // Although Slice 1b-core's only implemented arm with operands is
        // gone (Call → followup), the lookup_operand helper is shared
        // with future arms. Validate it surfaces a clear "adapter
        // invariant broken" message and embeds the enriched diagnostic
        // context (op variant + arg role) added by the verbose-mode
        // groundwork pass.
        let value_map: HashMap<ValueId, Hlvalue> = HashMap::new();
        let op = SpaceOperation {
            result: Some(ValueId(100)),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: ValueId(99),
                rhs: ValueId(0),
                result_ty: ValueType::Int,
            },
        };
        let err = lookup_operand(&value_map, ValueId(99), &op, "lhs")
            .expect_err("undefined operand lookup must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("undefined operand") && msg.contains("invariant"),
            "fail-loud message must explain the invariant, got: {msg}"
        );
        assert!(
            msg.contains("as lhs of BinOp") && msg.contains("ValueId(100)"),
            "verbose diagnostic must include arg role + op variant + result vid, got: {msg}"
        );
    }

    // ───── Slice 1c tests: topology assembly ─────

    fn link_to_returnblock(args: Vec<LinkArg>, returnblock_id: BlockId) -> crate::model::Link {
        let mut link = crate::model::Link::new_mixed(args, returnblock_id, None);
        link.prevblock = None;
        link
    }

    fn legacy_minimal_identity_return_graph() -> LegacyGraph {
        // Smallest valid legacy graph: one inputarg, returns it
        // directly. Slice 1c must produce a flowspace::FunctionGraph
        // whose startblock has the single inputarg Variable,
        // exits→returnblock, and the returnblock's inputarg is the same
        // Variable identity (so RPythonTyper.getreturnvar resolves
        // correctly).
        //
        // RPython convention: returnblock canonically has one inputarg
        // (`flowspace/model.py:13-18`). True void returns use a
        // `SomeNone` / `Void`-typed argument; pyre's legacy graph
        // mirrors that by always emitting a single ValueId in the
        // returnblock's inputargs.
        let mut graph = LegacyGraph::new("identity_return");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1)],
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(ValueId(1))],
                graph.returnblock,
            )],
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(1)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
        };
        graph.blocks = vec![startblock, returnblock];
        graph
    }

    #[test]
    fn function_graph_to_flowspace_minimal_identity_return_assembles_graph() {
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        let legacy = legacy_minimal_identity_return_graph();

        let output = function_graph_to_flowspace(&legacy, &empty_call_registry())
            .expect("minimal graph must assemble");

        // value_to_var must contain the inputarg.
        assert!(
            output.value_to_var.contains_key(&ValueId(1)),
            "value_to_var must seed the legacy inputarg"
        );

        // block_map must contain startblock + returnblock + exceptblock.
        assert!(output.block_map.contains_key(&legacy.startblock));
        assert!(output.block_map.contains_key(&legacy.returnblock));
        assert!(output.block_map.contains_key(&legacy.exceptblock));

        // The flowspace graph's startblock is the same BlockRef as the
        // mapped legacy.startblock.
        let graph = output.graph.borrow();
        assert!(Rc::ptr_eq(
            &graph.startblock,
            &output.block_map[&legacy.startblock],
        ));

        // The flowspace graph's returnblock is the mapped legacy.returnblock.
        assert!(Rc::ptr_eq(
            &graph.returnblock,
            &output.block_map[&legacy.returnblock],
        ));

        // The startblock has exactly one exit, targeting the returnblock.
        let startblock = graph.startblock.borrow();
        assert_eq!(startblock.exits.len(), 1);
        let exit = startblock.exits[0].borrow();
        assert!(Rc::ptr_eq(
            exit.target.as_ref().expect("link must have target"),
            &graph.returnblock,
        ));
    }

    #[test]
    fn function_graph_to_flowspace_returnvar_identity_preserved() {
        // When the returnblock has an inputarg ValueId, the flowspace
        // graph's returnblock must use the SAME Variable identity (so
        // RPythonTyper.getreturnvar finds the right Variable —
        // rtyper.rs:1633-1638).
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);

        let mut graph = LegacyGraph::new("with_return_var");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1)],
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(ValueId(1))],
                graph.returnblock,
            )],
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(2)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
        };
        graph.blocks = vec![startblock, returnblock];

        let output = function_graph_to_flowspace(&graph, &empty_call_registry())
            .expect("graph must assemble");

        // The flowspace returnblock's single inputarg must be the
        // Variable we seeded for ValueId(2).
        let flowspace_graph = output.graph.borrow();
        let returnblock_ref = &flowspace_graph.returnblock;
        let returnblock = returnblock_ref.borrow();
        assert_eq!(
            returnblock.inputargs.len(),
            1,
            "returnblock must carry the single return-value inputarg"
        );
        match &returnblock.inputargs[0] {
            Hlvalue::Variable(v) => {
                let expected = &output.value_to_var[&ValueId(2)];
                assert_eq!(
                    v, expected,
                    "returnblock inputarg must preserve Variable identity from value_to_var"
                );
            }
            other => panic!("returnblock inputarg must be a Variable, got {other:?}"),
        }
    }

    #[test]
    fn function_graph_to_flowspace_const_define_op_inlined_in_link_args() {
        // ConstInt(7) defines ValueId(2). Slice 1b's
        // build_value_to_hlvalue_map inlines it into Link.args as
        // Hlvalue::Constant — Slice 1c's link translation must use that
        // mapping rather than wrapping the unused Variable.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);

        let mut graph = LegacyGraph::new("const_link_arg");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1)],
            operations: vec![SpaceOperation {
                result: Some(ValueId(2)),
                kind: OpKind::ConstInt(7),
            }],
            exitswitch: None,
            // Return ValueId(2), the ConstInt define.
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(ValueId(2))],
                graph.returnblock,
            )],
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(3)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
        };
        annotations.set(ValueId(3), ValueType::Int);
        graph.blocks = vec![startblock, returnblock];

        let output = function_graph_to_flowspace(&graph, &empty_call_registry())
            .expect("graph must assemble");

        let flowspace_graph = output.graph.borrow();
        let startblock = flowspace_graph.startblock.borrow();
        // ConstInt define is a skip arm — operations must be empty.
        assert!(
            startblock.operations.is_empty(),
            "ConstInt define has no flowspace::SpaceOperation analogue"
        );
        // The exit's link arg must be the inlined Constant.
        let exit = startblock.exits[0].borrow();
        assert_eq!(exit.args.len(), 1);
        match exit.args[0].as_ref().expect("link arg is Some") {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Int(n) => assert_eq!(*n, 7),
                other => panic!("link arg constant must be Int, got {other:?}"),
            },
            other => panic!("link arg must be Hlvalue::Constant, got {other:?}"),
        }
    }

    #[test]
    fn function_graph_to_flowspace_exception_link_materialises_extravars_before_args() {
        // RPython checkgraph defines exception-link extravars before
        // validating link.args. Pyre's legacy graph represents those
        // as fresh ValueIds whose only definition site is the link.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);
        annotations.set(ValueId(3), ValueType::Int);
        annotations.set(ValueId(4), ValueType::Int);
        annotations.set(ValueId(10), ValueType::Int);
        annotations.set(ValueId(11), ValueType::Ref);

        let mut graph = LegacyGraph::new("canraise_with_extravars");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1), ValueId(2)],
            operations: vec![SpaceOperation {
                result: Some(ValueId(3)),
                kind: OpKind::BinOp {
                    op: "add".to_string(),
                    lhs: ValueId(1),
                    rhs: ValueId(2),
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: Some(crate::model::ExitSwitch::LastException),
            exits: vec![
                link_to_returnblock(vec![LinkArg::Value(ValueId(3))], graph.returnblock),
                crate::model::Link::new_mixed(
                    vec![LinkArg::Value(ValueId(10)), LinkArg::Value(ValueId(11))],
                    graph.exceptblock,
                    Some(crate::model::exception_exitcase()),
                )
                .extravars(
                    Some(LinkArg::Value(ValueId(10))),
                    Some(LinkArg::Value(ValueId(11))),
                ),
            ],
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(4)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
        };
        let exceptblock = Block {
            id: graph.exceptblock,
            inputargs: vec![ValueId(10), ValueId(11)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
        };
        graph.blocks = vec![startblock, returnblock, exceptblock];

        let output = function_graph_to_flowspace(&graph, &empty_call_registry())
            .expect("exception graph assembles");
        let flowspace_graph = output.graph.borrow();
        let startblock = flowspace_graph.startblock.borrow();
        let exc_link = startblock.exits[1].borrow();
        assert!(exc_link.last_exception.is_some());
        assert!(exc_link.last_exc_value.is_some());
        assert_eq!(
            exc_link.args[0].as_ref(),
            exc_link.last_exception.as_ref(),
            "exception type arg must reuse link.last_exception Variable"
        );
        assert_eq!(
            exc_link.args[1].as_ref(),
            exc_link.last_exc_value.as_ref(),
            "exception value arg must reuse link.last_exc_value Variable"
        );
    }

    #[test]
    fn function_graph_to_flowspace_unported_opkind_surfaces_failloud() {
        // A graph carrying a still-fail-loud OpKind (Call::Indirect —
        // requires rclass.rs lowering to VtableMethodPtr + IndirectCall
        // before reaching the adapter) must surface that op's
        // translate_op error from inside Pass 2, not silently emit a
        // partial graph.
        let mut annotations = AnnotationState::new();
        annotations.set(ValueId(1), ValueType::Int);
        annotations.set(ValueId(2), ValueType::Int);

        let mut graph = LegacyGraph::new("unported_op");
        let startblock = Block {
            id: graph.startblock,
            inputargs: vec![ValueId(1)],
            operations: vec![SpaceOperation {
                result: Some(ValueId(2)),
                kind: OpKind::Call {
                    target: crate::model::CallTarget::Indirect {
                        trait_root: "MyTrait".into(),
                        method_name: "do_it".into(),
                    },
                    args: vec![ValueId(1)],
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(ValueId(2))],
                graph.returnblock,
            )],
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![ValueId(3)],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
        };
        annotations.set(ValueId(3), ValueType::Int);
        graph.blocks = vec![startblock, returnblock];

        let err = function_graph_to_flowspace(&graph, &empty_call_registry())
            .expect_err("unported OpKind must surface as TyperError");
        let msg = format!("{err}");
        assert!(
            msg.contains("Indirect") && msg.contains("rclass"),
            "unported-op error must propagate translate_op's fail-loud message, got: {msg}"
        );
    }

    #[test]
    fn opkind_variant_name_covers_core_variants() {
        // Sanity: variant_name maps each core OpKind to a stable string.
        // Any new OpKind variant added without a corresponding arm here
        // surfaces as "<unknown OpKind variant>" in fail-loud messages,
        // which prompts the developer to extend this table.
        assert_eq!(opkind_variant_name(&OpKind::ConstInt(0)), "ConstInt");
        assert_eq!(opkind_variant_name(&OpKind::ConstFloat(0)), "ConstFloat");
        assert_eq!(
            opkind_variant_name(&OpKind::Input {
                name: "x".into(),
                ty: ValueType::Int
            }),
            "Input"
        );
    }
}
