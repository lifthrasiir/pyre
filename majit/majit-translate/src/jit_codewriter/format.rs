//! Format the SSA representation as text for tests.
//!
//! Translated from `rpython/jit/codewriter/format.py`. Pyre's
//! [`SSARepr`](crate::flatten::SSARepr) holds typed [`FlatOp`] variants
//! rather than raw tuples, so the formatter produces the same textual
//! shape as upstream's `format_assembler` while reading the Rust types
//! directly.
//!
//! - Registers print as `%i<n>`, `%r<n>`, `%f<n>`.
//! - Constants print as `$<value>` (matching `format.py:23`).
//! - Labels print as `L<index>` (assigned in textual order, matching
//!   `format.py`'s `getlabelname`).
//! - `ListOfKind` argument groups print as `I[…]`, `R[…]`, `F[…]`
//!   (matching `format.py:27`).
//! - Call descriptors print via their `Debug` repr (matching
//!   `format.py:32-33` `repr(AbstractDescr)`).
//! - When `ssarepr.insns_pos` is set the formatter prefixes each line
//!   with `'%4d  '` (matching `format.py:57-60`).
//! - The trailing `('---',)` sentinel (`format.py:54-55`) is trimmed.
//!
//! The reverse direction (`unformat_assembler`) is intentionally not
//! ported — the parity tests we run end-to-end build SSARepr through
//! the codewriter pipeline rather than parsing assembler text.  When a
//! caller actually needs to round-trip text → SSARepr the missing
//! function should be ported here mirroring `format.py:104-167` line
//! by line.

use std::collections::HashMap;
use std::fmt::Write;

use crate::flatten::{FlatOp, Label, RegKind, SSARepr};
use crate::model::ValueId;

/// format.py:12-81 `format_assembler(ssarepr)`.
pub fn format_assembler(ssarepr: &SSARepr) -> String {
    // First pass: collect every label that appears as a target so the
    // numbering matches format.py's getlabelname (labels are numbered in
    // first-seen order).
    let mut seenlabels: HashMap<Label, usize> = HashMap::new();
    let mut next_label = 0usize;
    let name_label = |label: Label, seen: &mut HashMap<Label, usize>, next: &mut usize| {
        *seen.entry(label).or_insert_with(|| {
            *next += 1;
            *next
        })
    };
    for op in &ssarepr.insns {
        match op {
            FlatOp::Jump(label)
            | FlatOp::CatchException { target: label }
            | FlatOp::GotoIfExceptionMismatch { target: label, .. }
            | FlatOp::IntBinOpJumpIfOvf { target: label, .. }
            | FlatOp::GotoIfNot { target: label, .. } => {
                name_label(*label, &mut seenlabels, &mut next_label);
            }
            _ => {}
        }
    }

    // format.py:53-55:
    //   insns = ssarepr.insns
    //   if insns and insns[-1] == ('---',):
    //       insns = insns[:-1]
    let insns: &[FlatOp] = match ssarepr.insns.last() {
        Some(FlatOp::Unreachable) => &ssarepr.insns[..ssarepr.insns.len() - 1],
        _ => &ssarepr.insns[..],
    };

    let mut out = String::new();
    for (i, op) in insns.iter().enumerate() {
        // format.py:57-60: prefix = '%4d  ' % ssarepr._insns_pos[i] when set.
        let prefix = match &ssarepr.insns_pos {
            Some(positions) => positions
                .get(i)
                .map(|p| format!("{p:>4}  "))
                .unwrap_or_default(),
            None => String::new(),
        };
        match op {
            FlatOp::Label(label) => {
                if let Some(num) = seenlabels.get(label) {
                    let _ = writeln!(out, "{prefix}L{num}:");
                }
            }
            FlatOp::Op(space_op) => {
                let args = op_args_repr(space_op, &ssarepr.value_kinds);
                if args.is_empty() {
                    let _ = writeln!(out, "{prefix}{}", op_name(space_op));
                } else {
                    let _ = writeln!(out, "{prefix}{} {args}", op_name(space_op));
                }
            }
            FlatOp::Jump(label) => {
                let num = name_label(*label, &mut seenlabels, &mut next_label);
                let _ = writeln!(out, "{prefix}goto L{num}");
            }
            FlatOp::CatchException { target } => {
                let num = name_label(*target, &mut seenlabels, &mut next_label);
                let _ = writeln!(out, "{prefix}catch_exception L{num}");
            }
            FlatOp::GotoIfExceptionMismatch { llexitcase, target } => {
                let num = name_label(*target, &mut seenlabels, &mut next_label);
                let _ = writeln!(
                    out,
                    "{prefix}goto_if_exception_mismatch ${llexitcase:?}, L{num}"
                );
            }
            FlatOp::GotoIfNot { cond, target } => {
                let num = name_label(*target, &mut seenlabels, &mut next_label);
                let _ = writeln!(
                    out,
                    "{prefix}goto_if_not {}, L{num}",
                    register_repr(*cond, &ssarepr.value_kinds)
                );
            }
            FlatOp::IntBinOpJumpIfOvf {
                op,
                target,
                lhs,
                rhs,
                dst,
            } => {
                let opname = match op {
                    crate::flatten::IntOvfOp::Add => "int_add_jump_if_ovf",
                    crate::flatten::IntOvfOp::Sub => "int_sub_jump_if_ovf",
                    crate::flatten::IntOvfOp::Mul => "int_mul_jump_if_ovf",
                };
                let num = name_label(*target, &mut seenlabels, &mut next_label);
                let _ = writeln!(
                    out,
                    "{prefix}{opname} L{num}, {}, {} -> {}",
                    register_repr(*lhs, &ssarepr.value_kinds),
                    register_repr(*rhs, &ssarepr.value_kinds),
                    register_repr(*dst, &ssarepr.value_kinds)
                );
            }
            // `flatten.py:333-335` emits opnames prefixed by kind —
            // `int_copy`/`ref_copy`/`float_copy`,
            // `int_push`/`ref_push`/`float_push`,
            // `int_pop`/`ref_pop`/`float_pop` — so the formatter just
            // prints `asm[0]` verbatim. Mirror that here by deriving
            // the kind from the moved register's `value_kinds` entry.
            FlatOp::Move { dst, src } => {
                let kind = linkarg_kind_name(src, &ssarepr.value_kinds);
                let _ = writeln!(
                    out,
                    "{prefix}{kind}_copy {} -> {}",
                    linkarg_repr(src, &ssarepr.value_kinds),
                    register_repr(*dst, &ssarepr.value_kinds)
                );
            }
            FlatOp::Push(src) => {
                let kind = kind_name(*src, &ssarepr.value_kinds);
                let _ = writeln!(
                    out,
                    "{prefix}{kind}_push {}",
                    register_repr(*src, &ssarepr.value_kinds)
                );
            }
            FlatOp::Pop(dst) => {
                let kind = kind_name(*dst, &ssarepr.value_kinds);
                let _ = writeln!(
                    out,
                    "{prefix}{kind}_pop -> {}",
                    register_repr(*dst, &ssarepr.value_kinds)
                );
            }
            FlatOp::LastException { dst } => {
                let _ = writeln!(
                    out,
                    "{prefix}last_exception -> {}",
                    register_repr(*dst, &ssarepr.value_kinds)
                );
            }
            FlatOp::LastExcValue { dst } => {
                let _ = writeln!(
                    out,
                    "{prefix}last_exc_value -> {}",
                    register_repr(*dst, &ssarepr.value_kinds)
                );
            }
            FlatOp::Live { live_values } => {
                let mut names: Vec<String> = live_values
                    .iter()
                    .map(|v| register_repr(*v, &ssarepr.value_kinds))
                    .collect();
                // format.py:76: `if asm[0] == '-live-': lst.sort()`.
                names.sort();
                let _ = writeln!(out, "{prefix}-live- {}", names.join(", "));
            }
            FlatOp::Reraise => {
                let _ = writeln!(out, "{prefix}reraise");
            }
            FlatOp::IntReturn(v) => {
                let _ = writeln!(
                    out,
                    "{prefix}int_return {}",
                    linkarg_repr(v, &ssarepr.value_kinds)
                );
            }
            FlatOp::RefReturn(v) => {
                let _ = writeln!(
                    out,
                    "{prefix}ref_return {}",
                    linkarg_repr(v, &ssarepr.value_kinds)
                );
            }
            FlatOp::FloatReturn(v) => {
                let _ = writeln!(
                    out,
                    "{prefix}float_return {}",
                    linkarg_repr(v, &ssarepr.value_kinds)
                );
            }
            FlatOp::VoidReturn => {
                let _ = writeln!(out, "{prefix}void_return");
            }
            FlatOp::Raise(v) => {
                let _ = writeln!(
                    out,
                    "{prefix}raise {}",
                    linkarg_repr(v, &ssarepr.value_kinds)
                );
            }
            FlatOp::Unreachable => {
                let _ = writeln!(out, "{prefix}---");
            }
        }
    }
    out
}

/// format.py:83-102 `assert_format(ssarepr, expected)`.
///
/// Compares the formatted SSARepr with `expected` line by line.  When a
/// line differs we emit the same `Got:` / `Expected:` diff format as
/// upstream so failing tests are easy to read.
pub fn assert_format(ssarepr: &SSARepr, expected: &str) {
    let asm = format_assembler(ssarepr);
    let expected = if expected.is_empty() {
        String::new()
    } else {
        // Normalize multiline raw-string indentation the way
        // py.code.Source(expected).strip() does in upstream.
        normalize_expected(expected)
    };
    let asm_lines: Vec<&str> = asm.split('\n').collect();
    let exp_lines: Vec<&str> = expected.split('\n').collect();
    for (asm_line, exp_line) in asm_lines.iter().zip(exp_lines.iter()) {
        if asm_line != exp_line {
            let mut msg = String::new();
            msg.push_str("\n");
            let _ = writeln!(msg, "Got:      {asm_line}");
            let _ = writeln!(msg, "Expected: {exp_line}");
            let mut common = 0usize;
            for (a, e) in asm_line.chars().zip(exp_line.chars()) {
                if a == e {
                    common += 1;
                } else {
                    break;
                }
            }
            let _ = writeln!(msg, "          {}^^^^", " ".repeat(common));
            panic!("{msg}");
        }
    }
    assert_eq!(asm_lines.len(), exp_lines.len(), "line-count mismatch");
}

fn normalize_expected(expected: &str) -> String {
    // Strip the leading and trailing blank lines, then trim the common
    // indentation (similar to py.code.Source(...).strip()).
    let raw_lines: Vec<&str> = expected.split('\n').collect();
    let trimmed: Vec<&str> = raw_lines
        .iter()
        .copied()
        .skip_while(|l| l.trim().is_empty())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .skip_while(|l| l.trim().is_empty())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let indent = trimmed
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);
    let mut out = trimmed
        .iter()
        .map(|l| if l.len() >= indent { &l[indent..] } else { *l })
        .collect::<Vec<_>>()
        .join("\n");
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

fn register_repr(v: ValueId, kinds: &HashMap<ValueId, RegKind>) -> String {
    let kind = kinds.get(&v).copied().unwrap_or(RegKind::Ref);
    let prefix = match kind {
        RegKind::Int => 'i',
        RegKind::Ref => 'r',
        RegKind::Float => 'f',
    };
    format!("%{prefix}{}", v.0)
}

/// `flatten.py:326-335` — the kind prefix used when spelling
/// `int_copy`/`ref_copy`/`float_copy` (and `*_push`/`*_pop`).
fn kind_name(v: ValueId, kinds: &HashMap<ValueId, RegKind>) -> &'static str {
    match kinds.get(&v).copied().unwrap_or(RegKind::Ref) {
        RegKind::Int => "int",
        RegKind::Ref => "ref",
        RegKind::Float => "float",
    }
}

/// Upstream `getcolor(v)` returns either a `Register` (→ `%{i,r,f}<n>`)
/// or a `Constant` (→ `$<value>`); reproduce both shapes for the
/// `int_copy`/`ref_copy`/`float_copy` + `*_push` + `*_return` + `raise`
/// source operand printing.
fn linkarg_repr(arg: &crate::model::LinkArg, kinds: &HashMap<ValueId, RegKind>) -> String {
    match arg {
        crate::model::LinkArg::Value(v) => register_repr(*v, kinds),
        crate::model::LinkArg::Const(cv) => format!("${cv}"),
    }
}

/// Kind prefix for a [`LinkArg`] source — `value_kinds` for Variables,
/// `ConstValue` shape (via `flatten::constvalue_kind`) for Constants.
fn linkarg_kind_name(
    arg: &crate::model::LinkArg,
    kinds: &HashMap<ValueId, RegKind>,
) -> &'static str {
    match arg {
        crate::model::LinkArg::Value(v) => kind_name(*v, kinds),
        crate::model::LinkArg::Const(cv) => match crate::flatten::constvalue_kind(cv) {
            'i' => "int",
            'f' => "float",
            _ => "ref",
        },
    }
}

/// format.py:26-27 `ListOfKind` formatter.
///
/// Upstream emits `'%s[%s]' % (x.kind[0].upper(), ', '.join(map(repr, x)))`.
/// Pyre's call-family `OpKind` variants split args into typed
/// `args_i`/`args_r`/`args_f` Vecs, so the kind char is fixed per slot.
fn list_of_kind_repr(
    kind_char: char,
    args: &[ValueId],
    kinds: &HashMap<ValueId, RegKind>,
) -> String {
    let parts: Vec<String> = args.iter().map(|v| register_repr(*v, kinds)).collect();
    format!("{}[{}]", kind_char.to_ascii_uppercase(), parts.join(", "))
}

/// format.py:20-23 — render a `funcptr` slot.
///
/// Upstream emits `$<* struct <name>>` for `Constant(lltype.Ptr(Struct))`
/// and `$<value>` otherwise.  Pyre's codewrite-time funcptr surrogate is
/// either a symbolic [`crate::model::CallTarget`] or a runtime
/// [`crate::model::ValueId`].
fn call_target_repr(target: &crate::model::CallTarget) -> String {
    use crate::model::CallTarget;
    match target {
        CallTarget::Method {
            name,
            receiver_root,
        } => match receiver_root {
            Some(root) => format!("$<* function '{root}.{name}'>"),
            None => format!("$<* function '{name}'>"),
        },
        CallTarget::FunctionPath { segments } => {
            format!("$<* function '{}'>", segments.join("."))
        }
        CallTarget::SyntheticTransparentCtor { name } => {
            format!("$<* synthetic-transparent-ctor '{name}'>")
        }
        CallTarget::Indirect {
            trait_root,
            method_name,
        } => format!("$<* indirect 'dyn {trait_root}::{method_name}'>"),
        CallTarget::UnsupportedExpr => "$<unsupported call target>".to_string(),
    }
}

fn call_funcptr_repr(
    funcptr: &crate::model::CallFuncPtr,
    kinds: &std::collections::HashMap<crate::model::ValueId, crate::flatten::RegKind>,
) -> String {
    match funcptr {
        crate::model::CallFuncPtr::Target(target) => call_target_repr(target),
        crate::model::CallFuncPtr::Value(value) => register_repr(*value, kinds),
    }
}

fn op_name(op: &crate::model::SpaceOperation) -> String {
    use crate::model::OpKind;
    match &op.kind {
        OpKind::Call { .. } => "call".to_string(),
        OpKind::ConstInt(_) => "const_int".to_string(),
        OpKind::ConstFloat(_) => "const_float".to_string(),
        OpKind::CallElidable {
            result_kind,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            format!(
                "call_elidable_{}_{result_kind}",
                kind_signature(args_i, args_r, args_f)
            )
        }
        OpKind::CallResidual {
            result_kind,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            format!(
                "residual_call_{}_{result_kind}",
                kind_signature(args_i, args_r, args_f)
            )
        }
        OpKind::CallMayForce {
            result_kind,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            format!(
                "call_may_force_{}_{result_kind}",
                kind_signature(args_i, args_r, args_f)
            )
        }
        OpKind::InlineCall {
            result_kind,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            format!(
                "inline_call_{}_{result_kind}",
                kind_signature(args_i, args_r, args_f)
            )
        }
        OpKind::RecursiveCall { result_kind, .. } => {
            format!("recursive_call_{result_kind}")
        }
        // For the rest, fall back on a stable Debug-derived discriminant.
        other => format!("{:?}", other)
            .split('{')
            .next()
            .unwrap_or("?")
            .split('(')
            .next()
            .unwrap_or("?")
            .trim()
            .to_lowercase(),
    }
}

/// jtransform.py:414-435 — call-family opcode kind suffix.
///
/// Encodes the (int, ref, float) arg tuple as a single-character
/// signature ("i", "r", "f", "ir", "irf", …).  Empty bins drop out so
/// `(args_i=[a], args_r=[], args_f=[])` produces `"i"`.
fn kind_signature(args_i: &[ValueId], args_r: &[ValueId], args_f: &[ValueId]) -> String {
    let mut out = String::new();
    if !args_i.is_empty() {
        out.push('i');
    }
    if !args_r.is_empty() {
        out.push('r');
    }
    if !args_f.is_empty() {
        out.push('f');
    }
    out
}

fn op_args_repr(op: &crate::model::SpaceOperation, kinds: &HashMap<ValueId, RegKind>) -> String {
    use crate::model::OpKind;
    let mut out = String::new();
    match &op.kind {
        OpKind::Call { args, .. } => {
            let parts: Vec<String> = args.iter().map(|v| register_repr(*v, kinds)).collect();
            out.push_str(&parts.join(", "));
        }
        // format.py:23 `'$%r' % (x.value,)` — constants print as $<value>.
        OpKind::ConstInt(value) => {
            let _ = write!(out, "${value}");
        }
        OpKind::ConstFloat(bits) => {
            let _ = write!(out, "${}", f64::from_bits(*bits));
        }
        // jtransform.py:414-435 `rewrite_call`:
        //   sublists = [lst_i?, lst_r?, lst_f?, calldescr?]   # only kinds present
        //   args = initialargs + sublists
        // → for residual_call/call_may_force/call_elidable upstream emits
        //   `$<funcptr>, I[…]?, R[…]?, F[…]?, <descr>` where the I/R/F
        //   slots are gated on the opname kind signature.  Pyre carries
        //   the funcptr identity on the dedicated `funcptr` field per
        //   jtransform.py:457 `[op.args[0]] + extraargs`.
        OpKind::CallElidable {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            ..
        }
        | OpKind::CallResidual {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            ..
        }
        | OpKind::CallMayForce {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut parts = vec![call_funcptr_repr(funcptr, kinds)];
            // jtransform.py:430-433 — emit each ListOfKind only when the
            // matching kind char is in the signature.
            if !args_i.is_empty() {
                parts.push(list_of_kind_repr('i', args_i, kinds));
            }
            if !args_r.is_empty() {
                parts.push(list_of_kind_repr('r', args_r, kinds));
            }
            if !args_f.is_empty() {
                parts.push(list_of_kind_repr('f', args_f, kinds));
            }
            // jtransform.py:434 — descr is the last sublist when set.
            parts.push(format!("{:?}", descriptor.extra_info));
            out.push_str(&parts.join(", "));
        }
        // jtransform.py:473-482 `handle_regular_call`:
        //   args = [jitcode] + [I?, R?, F? sublists]   # only kinds present
        // → format.py:34-35 renders the JitCode object via JitCode.__repr__.
        //   Before the codewriter assigns the final dense index, fall back
        //   to the symbolic jitcode name for debugging.
        OpKind::InlineCall {
            jitcode,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let head = match jitcode.try_index() {
                Some(index) => format!("<JitCode #{index}>"),
                None => format!("<JitCode {:?}>", jitcode.name),
            };
            let mut parts = vec![head];
            if !args_i.is_empty() {
                parts.push(list_of_kind_repr('i', args_i, kinds));
            }
            if !args_r.is_empty() {
                parts.push(list_of_kind_repr('r', args_r, kinds));
            }
            if !args_f.is_empty() {
                parts.push(list_of_kind_repr('f', args_f, kinds));
            }
            out.push_str(&parts.join(", "));
        }
        // jtransform.py:522-534 `handle_recursive_call`:
        //   args = [Constant(jdindex, lltype.Signed)] + green sublists + red sublists
        // → format.py:23 renders `Constant(jdindex)` as `$<jdindex>`.
        OpKind::RecursiveCall {
            jd_index,
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            ..
        } => {
            let mut parts = vec![format!("${jd_index}")];
            parts.push(list_of_kind_repr('i', greens_i, kinds));
            parts.push(list_of_kind_repr('r', greens_r, kinds));
            parts.push(list_of_kind_repr('f', greens_f, kinds));
            parts.push(list_of_kind_repr('i', reds_i, kinds));
            parts.push(list_of_kind_repr('r', reds_r, kinds));
            parts.push(list_of_kind_repr('f', reds_f, kinds));
            out.push_str(&parts.join(", "));
        }
        _ => {
            // **Stub branch.**  Pyre's `OpKind` carries typed payloads
            // rather than positional argument tuples, so an upstream-
            // shaped formatter would need a per-variant projection.
            // Variants not covered here (FieldRead/FieldWrite, etc.)
            // print just the op name; extend this match when a parity
            // test demands it.
        }
    }
    if let Some(result) = op.result {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str("-> ");
        out.push_str(&register_repr(result, kinds));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flatten::{FlatOp, Label, SSARepr};

    fn empty_ssa() -> SSARepr {
        SSARepr {
            name: "test".into(),
            insns: Vec::new(),
            num_values: 0,
            num_blocks: 0,
            value_kinds: HashMap::new(),
            insns_pos: None,
        }
    }

    #[test]
    fn format_jump_emits_label() {
        let mut ssa = empty_ssa();
        let target = Label(0);
        ssa.insns.push(FlatOp::Jump(target));
        let text = format_assembler(&ssa);
        assert!(text.contains("goto L1"));
    }

    #[test]
    fn format_label_uses_first_seen_numbering() {
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::Jump(Label(7)));
        ssa.insns.push(FlatOp::Label(Label(7)));
        let text = format_assembler(&ssa);
        assert!(text.contains("goto L1"));
        assert!(text.contains("L1:"));
    }

    #[test]
    fn format_unreachable_marker() {
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::Unreachable);
        let text = format_assembler(&ssa);
        // format.py:54-55 trims a trailing ('---',) sentinel.
        assert_eq!(text, "");
    }

    #[test]
    fn format_unreachable_in_middle_is_kept() {
        // Trim only when `---` is the last instruction (format.py:54-55).
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::Unreachable);
        ssa.insns.push(FlatOp::Jump(Label(0)));
        let text = format_assembler(&ssa);
        assert!(text.contains("---"));
        assert!(text.contains("goto L1"));
    }

    #[test]
    fn assert_format_matches_simple_program() {
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::Jump(Label(0)));
        ssa.insns.push(FlatOp::Label(Label(0)));
        ssa.insns.push(FlatOp::Unreachable);
        assert_format(
            &ssa,
            "
            goto L1
            L1:
            ",
        );
    }

    #[test]
    fn format_constint_emits_dollar_value() {
        // format.py:23 `'$%r' % (x.value,)`.
        use crate::model::{OpKind, SpaceOperation};
        let mut ssa = empty_ssa();
        ssa.value_kinds.insert(ValueId(0), RegKind::Int);
        ssa.insns.push(FlatOp::Op(SpaceOperation {
            kind: OpKind::ConstInt(42),
            result: Some(ValueId(0)),
        }));
        let text = format_assembler(&ssa);
        assert!(text.contains("$42"), "expected `$42` in: {text}");
        assert!(text.contains("-> %i0"), "expected `-> %i0` in: {text}");
    }

    #[test]
    fn format_residual_call_emits_descr_and_listofkind() {
        // jtransform.py:414-435 + format.py:27,32-33.
        use crate::call::CallDescriptor;
        use crate::model::{CallFuncPtr, CallTarget, OpKind, SpaceOperation};
        use majit_ir::descr::EffectInfo;

        let mut ssa = empty_ssa();
        ssa.value_kinds.insert(ValueId(1), RegKind::Int);
        ssa.value_kinds.insert(ValueId(2), RegKind::Ref);
        ssa.value_kinds.insert(ValueId(3), RegKind::Int);

        let funcptr = CallTarget::function_path(["foo"]);
        let descriptor = CallDescriptor::known(EffectInfo::default());
        ssa.insns.push(FlatOp::Op(SpaceOperation {
            kind: OpKind::CallResidual {
                funcptr: CallFuncPtr::Target(funcptr),
                descriptor,
                args_i: vec![ValueId(1)],
                args_r: vec![ValueId(2)],
                args_f: vec![],
                result_kind: 'i',
                indirect_targets: None,
            },
            result: Some(ValueId(3)),
        }));
        let text = format_assembler(&ssa);
        assert!(
            text.contains("residual_call_ir_i "),
            "expected residual_call_ir_i in: {text}"
        );
        // jtransform.py:456-462 emits funcptr as args[0], calldescr via
        // SpaceOperation.descr.  Pyre carries the funcptr identity on
        // descriptor.target and renders it as `$<* function 'name'>`
        // mirroring format.py:21-23 Ptr-to-Struct repr.
        assert!(
            text.contains("$<* function 'foo'>"),
            "expected funcptr slot in: {text}"
        );
        assert!(text.contains("I[%i1]"), "expected I[%i1] in: {text}");
        assert!(text.contains("R[%r2]"), "expected R[%r2] in: {text}");
        // jtransform.py:430-433 — empty kind slots are dropped, matching
        // upstream where `kinds = "ir"` excludes the F sublist entirely.
        assert!(
            !text.contains("F["),
            "F[] must not appear when 'f' kind absent: {text}"
        );
        assert!(text.contains("-> %i3"));
    }

    #[test]
    fn format_inline_call_emits_jitcode_and_listofkind() {
        use crate::model::{OpKind, SpaceOperation};
        let mut ssa = empty_ssa();
        ssa.value_kinds.insert(ValueId(1), RegKind::Ref);
        let callee = std::sync::Arc::new(crate::jitcode::JitCode::new("callee"));
        callee.set_index(7);
        ssa.insns.push(FlatOp::Op(SpaceOperation {
            kind: OpKind::InlineCall {
                jitcode: crate::jitcode::JitCodeHandle::new(callee),
                args_i: vec![],
                args_r: vec![ValueId(1)],
                args_f: vec![],
                result_kind: 'v',
            },
            result: None,
        }));
        let text = format_assembler(&ssa);
        assert!(
            text.contains("inline_call_r_v "),
            "expected inline_call_r_v in: {text}"
        );
        // jtransform.py:478 stores the JitCode object as args[0]; format.py
        // renders it via JitCode.__repr__ which carries the index-keyed
        // identity.  Pyre prints it as `<JitCode #N>` so the parity test
        // sees the same shape.
        assert!(text.contains("<JitCode #7>"), "got: {text}");
        assert!(text.contains("R[%r1]"));
    }

    #[test]
    fn format_recursive_call_emits_jd_and_six_listofkinds() {
        use crate::model::{OpKind, SpaceOperation};
        let mut ssa = empty_ssa();
        ssa.value_kinds.insert(ValueId(1), RegKind::Int);
        ssa.value_kinds.insert(ValueId(2), RegKind::Ref);
        ssa.insns.push(FlatOp::Op(SpaceOperation {
            kind: OpKind::RecursiveCall {
                jd_index: 0,
                greens_i: vec![ValueId(1)],
                greens_r: vec![],
                greens_f: vec![],
                reds_i: vec![],
                reds_r: vec![ValueId(2)],
                reds_f: vec![],
                result_kind: 'v',
            },
            result: None,
        }));
        let text = format_assembler(&ssa);
        assert!(text.contains("recursive_call_v "), "got: {text}");
        // jtransform.py:530 stores `Constant(jdindex, lltype.Signed)` as
        // args[0]; format.py:23 renders it as `$<value>`.  Pyre mirrors
        // the shape exactly: `$0` for jd_index=0.
        assert!(text.contains(" $0,"), "got: {text}");
        // Six ListOfKind groups: greens (i,r,f) + reds (i,r,f).
        let groups: Vec<&str> = text.matches('[').collect();
        assert_eq!(groups.len(), 6, "expected 6 ListOfKind groups, got: {text}");
    }

    #[test]
    fn format_with_insns_pos_prepends_position_prefix() {
        // format.py:57-60 `prefix = '%4d  ' % ssarepr._insns_pos[i]`.
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::Jump(Label(0)));
        ssa.insns.push(FlatOp::Label(Label(0)));
        ssa.insns_pos = Some(vec![0, 12]);
        let text = format_assembler(&ssa);
        assert!(
            text.contains("   0  goto L1"),
            "expected '   0  goto L1' in: {text}"
        );
        assert!(
            text.contains("  12  L1:"),
            "expected '  12  L1:' in: {text}"
        );
    }
}
