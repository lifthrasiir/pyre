//! Slice 92.4 — content equality + content hash for `GreenType::Str` greens
//! routes through pyre's `default_str_eq` / `default_str_hash` slot ABI
//! decoder (`warmstate.py:108-128 lltype.Ptr STR/UNICODE` parity).
//!
//! `equal_whatever` / `hash_whatever`'s frontend-agnostic fallback is
//! pointer equality / `value as u64` (safe for any frontend's slot
//! ABI); pyre registers its content-aware decoder via
//! `majit_ir::set_str_resolver` / `set_unicode_resolver` at startup.
//! This test simulates that registration so the equality / hash paths
//! exercise pyre's [`default_str_eq`] / [`default_str_hash`] /
//! [`default_unicode_hash`] decoders.
//!
//! The macro path emits the canonical pyre slot ABI for `name: str`
//! greens via `majit_ir::make_str_slot` — the i64 is `*const &'static
//! str` cast through `usize`, pointing to a leaked slot that holds
//! the fat `&str` pointer (data + len).  RPython's `rstr.STR*`
//! carries length internally, so pyre stores the fat pointer at a
//! stable address rather than the bare data pointer.  The registered
//! resolver dereferences the slot to recover the `&str` and compares
//! by content.

use majit_ir::GreenType;
use majit_ir::value::{
    default_str_eq, default_str_hash, default_unicode_hash, equal_whatever, hash_whatever,
    set_str_resolver, set_unicode_resolver,
};
use std::sync::Once;

/// Register pyre's slot-ABI-aware resolver exactly once per test
/// process so every test in this file routes Str / Unicode greens
/// through `default_str_eq` / `default_str_hash` /
/// `default_unicode_hash`.  `OnceLock`-backed setters are
/// init-once, so the first registration wins regardless of test
/// ordering.
fn ensure_pyre_resolvers() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        set_str_resolver(default_str_eq, default_str_hash);
        set_unicode_resolver(default_str_eq, default_unicode_hash);
    });
}

#[test]
fn str_greens_compare_by_content() {
    ensure_pyre_resolvers();
    let s1: &str = "hello";
    // Distinct backing &str slot (different stack allocation), same content.
    let owned = String::from("hello");
    let s2: &str = owned.as_str();
    let p1 = (&s1) as *const _ as usize as i64;
    let p2 = (&s2) as *const _ as usize as i64;
    assert_ne!(
        p1, p2,
        "Slice 92.4 fixture expects distinct &str storage \
         pointers — pointer-Ref equality would fail",
    );
    assert!(
        equal_whatever(GreenType::Str, p1, p2),
        "GreenType::Str must route through default_str_eq for content equality \
         (warmstate.py:108-112 ll_streq parity)",
    );
}

/// Stand-alone reimplementation of RPython
/// `rpython/rlib/objectmodel.py:596 _hash_string` (modified Fowler-
/// Noll-Vo over a byte stream) plus `rpython/rtyper/lltypesystem/
/// rstr.py:405 _ll_strhash` zero substitute (`if x == 0: x = 29872897`
/// — translation pre-zeroed memory uses 0 as the not-yet-computed
/// sentinel, so an actual hash of 0 is folded to a fixed constant).
/// The empty-string path returns `-1` because the FNV step short-
/// circuits before the zero-substitute branch.  Used to pin the
/// production hash to the RPython algorithm rather than any host-
/// language hash (Rust `DefaultHasher`, `SipHash13`, etc.).
fn rpython_ll_strhash_reference(s: &str) -> u64 {
    let bytes = s.as_bytes();
    let length = bytes.len();
    let raw: i64 = if length == 0 {
        // objectmodel.py:601 `if length == 0: return -1` — short-
        // circuits BEFORE the zero substitute, so empty hashes to -1
        // (cast through u64 widens to 0xFFFF_FFFF_FFFF_FFFF).
        -1
    } else {
        let mut x: i64 = (bytes[0] as i64) << 7;
        for &b in bytes {
            x = 1000003i64.wrapping_mul(x) ^ (b as i64);
        }
        x ^= length as i64;
        x
    };
    // rstr.py:411 — zero substitute folds 0 → 29872897.
    let zero_subbed = if raw == 0 { 29872897i64 } else { raw };
    zero_subbed as u64
}

#[test]
fn str_greens_hash_by_content() {
    ensure_pyre_resolvers();
    let s1: &str = "match-me";
    let owned = String::from("match-me");
    let s2: &str = owned.as_str();
    let p1 = (&s1) as *const _ as usize as i64;
    let p2 = (&s2) as *const _ as usize as i64;
    let h1 = hash_whatever(GreenType::Str, p1);
    let h2 = hash_whatever(GreenType::Str, p2);
    assert_eq!(
        h1, h2,
        "GreenType::Str must route through default_str_hash for content hashing \
         (warmstate.py:115-128 ll_strhash parity)",
    );
    // Pin the production hash to the RPython modified-FNV algorithm —
    // not Rust's `DefaultHasher` (`SipHash13`) or any other host hash.
    // A divergence here means `default_str_hash` deviated from
    // `objectmodel.py:596 _hash_string`.
    let expected = rpython_ll_strhash_reference("match-me");
    assert_eq!(
        h1, expected,
        "GreenType::Str hash must match the literal RPython modified-FNV \
         result (objectmodel.py:596 _hash_string + rstr.py:411 zero \
         substitute) — divergence indicates default_str_hash drifted \
         from the RPython-translated `_ll_strhash` output",
    );
}

/// Empty-string boundary: RPython `objectmodel.py:601 _hash_string`
/// short-circuits with `return -1` on length 0 (BEFORE the zero
/// substitute folds 0 → 29872897, so empty stays at -1).  pyre's
/// `default_str_hash` must surface the same sentinel rather than `0`.
#[test]
fn str_green_empty_hashes_to_minus_one_per_rpython() {
    ensure_pyre_resolvers();
    let s: &str = "";
    let p = (&s) as *const _ as usize as i64;
    let h = hash_whatever(GreenType::Str, p);
    let expected = rpython_ll_strhash_reference("");
    assert_eq!(
        h, expected,
        "empty STR green must hash to `-1` (objectmodel.py:601 \
         `if length == 0: return -1`)",
    );
}

#[test]
fn distinct_content_str_greens_are_unequal() {
    ensure_pyre_resolvers();
    let s1: &str = "foo";
    let s2: &str = "bar";
    let p1 = (&s1) as *const _ as usize as i64;
    let p2 = (&s2) as *const _ as usize as i64;
    assert!(
        !equal_whatever(GreenType::Str, p1, p2),
        "distinct content must compare unequal under GreenType::Str",
    );
}

/// `make_str_slot` slot ABI contract — every call materialises a
/// stable-address `*const &'static str` slot that decodes via
/// `default_str_eq` / `default_str_hash` like every other STR slot,
/// and compares content-equal to a stack-local slot carrying the
/// same content.  RPython's `rstr.STR*` is GC-allocated once per
/// JitCell (`warmstate.py:564 _green_args_spec` + `_cell_cache`);
/// pyre's helper leaks a fresh slot per call — the GreenKey HashMap
/// content-de-dupes via `default_str_eq`, so semantically every
/// merge-point hit collapses to a single cache entry.  A
/// process-global string intern was rejected as non-orthodox
/// (RPython does not maintain one); the structural fix
/// (per-JitCell owned-string field) is a multi-session refactor.
#[test]
fn make_str_slot_decodes_via_default_str_eq() {
    use majit_ir::make_str_slot;
    ensure_pyre_resolvers();
    let p_macro_emitted = make_str_slot("decoded-content");
    let owned = String::from("decoded-content");
    let s_local: &str = owned.as_str();
    let p_local = (&s_local) as *const _ as usize as i64;
    assert!(
        equal_whatever(GreenType::Str, p_macro_emitted, p_local),
        "macro-emitted STR slot must compare content-equal to a local stack slot \
         carrying the same content",
    );
    let h_macro_emitted = hash_whatever(GreenType::Str, p_macro_emitted);
    let h_local = hash_whatever(GreenType::Str, p_local);
    assert_eq!(
        h_macro_emitted, h_local,
        "macro-emitted and local STR slots with same content must hash identically",
    );
}
