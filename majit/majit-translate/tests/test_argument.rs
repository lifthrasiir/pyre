//! Line-by-line port of `rpython/flowspace/test/test_argument.py`.
//!
//! Upstream (74 LOC, 5 tests — 1 class-based + 4 module-level): pins
//! `Signature` helpers, equality, `find_argname`, tuply-unpacking, and
//! `CallSpec.flatten`. Ported verbatim wherever Rust allows.
//!
//! `Signature.__repr__` is not tested (RPython uses `%r`-style string
//! formatting; the Rust `Debug` impl differs).

use majit_translate::flowspace::argument::{CallSpec, Signature, SignatureItem};
use majit_translate::flowspace::model::{ConstValue, Constant, Hlvalue};
use std::collections::HashMap;

// ---- TestSignature ----

#[test]
fn test_helpers() {
    let sig = Signature::new(vec!["a".into(), "b".into(), "c".into()], None, None);
    assert_eq!(sig.num_argnames(), 3);
    assert!(!sig.has_vararg());
    assert!(!sig.has_kwarg());
    assert_eq!(sig.scope_length(), 3);
    assert_eq!(sig.getallvarnames(), vec!["a", "b", "c"]);

    let sig = Signature::new(
        vec!["a".into(), "b".into(), "c".into()],
        Some("c".into()),
        None,
    );
    assert_eq!(sig.num_argnames(), 3);
    assert!(sig.has_vararg());
    assert!(!sig.has_kwarg());
    assert_eq!(sig.scope_length(), 4);
    assert_eq!(sig.getallvarnames(), vec!["a", "b", "c", "c"]);

    let sig = Signature::new(
        vec!["a".into(), "b".into(), "c".into()],
        None,
        Some("c".into()),
    );
    assert_eq!(sig.num_argnames(), 3);
    assert!(!sig.has_vararg());
    assert!(sig.has_kwarg());
    assert_eq!(sig.scope_length(), 4);
    assert_eq!(sig.getallvarnames(), vec!["a", "b", "c", "c"]);

    let sig = Signature::new(
        vec!["a".into(), "b".into(), "c".into()],
        Some("d".into()),
        Some("c".into()),
    );
    assert_eq!(sig.num_argnames(), 3);
    assert!(sig.has_vararg());
    assert!(sig.has_kwarg());
    assert_eq!(sig.scope_length(), 5);
    assert_eq!(sig.getallvarnames(), vec!["a", "b", "c", "d", "c"]);
}

#[test]
fn test_eq() {
    let sig1 = Signature::new(
        vec!["a".into(), "b".into(), "c".into()],
        Some("d".into()),
        Some("c".into()),
    );
    let sig2 = Signature::new(
        vec!["a".into(), "b".into(), "c".into()],
        Some("d".into()),
        Some("c".into()),
    );
    assert_eq!(sig1, sig2);
}

#[test]
fn test_find_argname() {
    let sig = Signature::new(vec!["a".into(), "b".into(), "c".into()], None, None);
    assert_eq!(sig.find_argname("a"), 0);
    assert_eq!(sig.find_argname("b"), 1);
    assert_eq!(sig.find_argname("c"), 2);
    assert_eq!(sig.find_argname("d"), -1);
}

#[test]
fn test_tuply() {
    let sig = Signature::new(
        vec!["a".into(), "b".into(), "c".into()],
        Some("d".into()),
        Some("e".into()),
    );
    let (x, y, z) = sig.tuple_view();
    assert_eq!(x, ["a", "b", "c"]);
    assert_eq!(y, Some("d"));
    assert_eq!(z, Some("e"));
}

// ---- module-level `test_flatten_CallSpec` ----

fn hi(n: i64) -> Hlvalue {
    Hlvalue::Constant(Constant::new(ConstValue::Int(n)))
}

fn kw(pairs: &[(&str, i64)]) -> HashMap<String, Hlvalue> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), hi(*v)))
        .collect()
}

#[test]
fn test_flatten_callspec() {
    // `CallSpec([1, 2, 3]).flatten() == ((3, (), False), [1, 2, 3])`
    let args = CallSpec::new(vec![hi(1), hi(2), hi(3)], None, None);
    let (shape, data) = args.flatten();
    assert_eq!(shape.shape_cnt, 3);
    assert!(shape.shape_keys.is_empty());
    assert!(!shape.shape_star);
    assert_eq!(data, vec![hi(1), hi(2), hi(3)]);

    // `CallSpec([1]).flatten() == ((1, (), False), [1])`
    let args = CallSpec::new(vec![hi(1)], None, None);
    let (shape, data) = args.flatten();
    assert_eq!(shape.shape_cnt, 1);
    assert!(shape.shape_keys.is_empty());
    assert!(!shape.shape_star);
    assert_eq!(data, vec![hi(1)]);

    // `CallSpec([1, 2, 3, 4, 5]).flatten() == ((5, (), False), [1..5])`
    let args = CallSpec::new(vec![hi(1), hi(2), hi(3), hi(4), hi(5)], None, None);
    let (shape, data) = args.flatten();
    assert_eq!(shape.shape_cnt, 5);
    assert_eq!(data, vec![hi(1), hi(2), hi(3), hi(4), hi(5)]);

    // `CallSpec([1], {'c': 3, 'b': 2}).flatten() == ((1, ('b', 'c'), False), [1, 2, 3])`
    let args = CallSpec::new(vec![hi(1)], Some(kw(&[("c", 3), ("b", 2)])), None);
    let (shape, data) = args.flatten();
    assert_eq!(shape.shape_cnt, 1);
    assert_eq!(shape.shape_keys, vec!["b".to_string(), "c".into()]);
    assert_eq!(data, vec![hi(1), hi(2), hi(3)]);

    // `CallSpec([1], {'c': 5}).flatten() == ((1, ('c',), False), [1, 5])`
    let args = CallSpec::new(vec![hi(1)], Some(kw(&[("c", 5)])), None);
    let (shape, data) = args.flatten();
    assert_eq!(shape.shape_cnt, 1);
    assert_eq!(shape.shape_keys, vec!["c".to_string()]);
    assert_eq!(data, vec![hi(1), hi(5)]);

    // `CallSpec([1], {'c': 5, 'd': 7}).flatten() == ((1, ('c', 'd'), False), [1, 5, 7])`
    let args = CallSpec::new(vec![hi(1)], Some(kw(&[("c", 5), ("d", 7)])), None);
    let (shape, data) = args.flatten();
    assert_eq!(shape.shape_keys, vec!["c".to_string(), "d".into()]);
    assert_eq!(data, vec![hi(1), hi(5), hi(7)]);

    // `CallSpec([1, 2, 3, 4, 5], {'e': 5, 'd': 7}).flatten() ==
    //     ((5, ('d', 'e'), False), [1, 2, 3, 4, 5, 7, 5])`
    let args = CallSpec::new(
        vec![hi(1), hi(2), hi(3), hi(4), hi(5)],
        Some(kw(&[("e", 5), ("d", 7)])),
        None,
    );
    let (shape, data) = args.flatten();
    assert_eq!(shape.shape_cnt, 5);
    assert_eq!(shape.shape_keys, vec!["d".to_string(), "e".into()]);
    assert_eq!(data, vec![hi(1), hi(2), hi(3), hi(4), hi(5), hi(7), hi(5)]);
}

#[test]
fn signature_tuple_protocol_matches_upstream() {
    // upstream `rpython/flowspace/argument.py:49` — `Signature.__len__`
    // 은 항상 3, `Signature.__getitem__` 은 argnames / varargname /
    // kwargname 순.
    let sig = Signature::new(
        vec!["a".into(), "b".into()],
        Some("va".into()),
        Some("kw".into()),
    );
    assert_eq!(sig.len_tuple(), 3);
    match sig.getitem(0) {
        SignatureItem::Argnames(names) => {
            assert_eq!(names, &["a".to_string(), "b".into()]);
        }
        other => panic!("sig[0] unexpected: {other:?}"),
    }
    match sig.getitem(1) {
        SignatureItem::Varargname(Some("va")) => {}
        other => panic!("sig[1] unexpected: {other:?}"),
    }
    match sig.getitem(2) {
        SignatureItem::Kwargname(Some("kw")) => {}
        other => panic!("sig[2] unexpected: {other:?}"),
    }
}

#[test]
#[should_panic(expected = "Signature index out of range")]
fn signature_getitem_three_panics() {
    let sig = Signature::new(Vec::new(), None, None);
    let _ = sig.getitem(3);
}
