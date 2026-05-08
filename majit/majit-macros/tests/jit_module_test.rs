use majit_macros::jit_module;

#[jit_module]
mod basic_module {
    use majit_macros::{dont_look_inside, elidable};

    #[elidable]
    pub fn helper_square(x: i64) -> i64 {
        x * x
    }

    #[dont_look_inside]
    pub fn helper_opaque(a: i64, b: i64) -> i64 {
        a + b
    }

    pub fn not_jit(x: i64) -> i64 {
        x + 1
    }
}

#[test]
fn test_discovered_helpers_names() {
    let helpers = basic_module::__MAJIT_DISCOVERED_HELPERS;
    assert_eq!(helpers.len(), 2);
    assert!(helpers.contains(&"helper_square"));
    assert!(helpers.contains(&"helper_opaque"));
    // not_jit should not be discovered
    assert!(!helpers.contains(&"not_jit"));
}

#[test]
fn test_discovered_helper_policies() {
    let policies = basic_module::__MAJIT_HELPER_POLICIES;
    assert_eq!(policies.len(), 2);
    assert!(policies.contains(&("helper_square", "elidable")));
    assert!(policies.contains(&("helper_opaque", "dont_look_inside")));
}

#[test]
fn test_helper_trace_fnaddrs_registry() {
    let trace_fnaddrs = basic_module::__majit_helper_trace_fnaddrs();
    assert_eq!(trace_fnaddrs.len(), 2);
    assert!(trace_fnaddrs.iter().any(|(path, addr)| {
        *path == concat!(module_path!(), "::basic_module::helper_square")
            && *addr == basic_module::__majit_call_target_helper_square as *const () as usize as i64
    }));
    assert!(trace_fnaddrs.iter().any(|(path, addr)| {
        *path == concat!(module_path!(), "::basic_module::helper_opaque")
            && *addr == basic_module::__majit_call_target_helper_opaque as *const () as usize as i64
    }));
}

#[test]
fn test_functions_still_callable() {
    assert_eq!(basic_module::helper_square(5), 25);
    assert_eq!(basic_module::helper_opaque(2, 3), 5);
    assert_eq!(basic_module::not_jit(10), 11);
}

#[jit_module]
mod empty_module {
    pub fn plain_fn() -> i64 {
        42
    }
}

#[test]
fn test_empty_discovery() {
    let helpers = empty_module::__MAJIT_DISCOVERED_HELPERS;
    assert!(helpers.is_empty());
    let policies = empty_module::__MAJIT_HELPER_POLICIES;
    assert!(policies.is_empty());
    let trace_fnaddrs = empty_module::__majit_helper_trace_fnaddrs();
    assert!(trace_fnaddrs.is_empty());
}

#[jit_module]
mod multi_attr_module {
    use majit_macros::{
        dont_look_inside, elidable, jit_loop_invariant, jit_may_force, jit_release_gil,
    };

    #[elidable]
    pub fn pure_fn(x: i64) -> i64 {
        x * 2
    }

    #[dont_look_inside]
    pub fn opaque_fn(x: i64) -> i64 {
        x + 1
    }

    #[jit_may_force]
    pub fn force_fn(x: i64) -> i64 {
        x - 1
    }

    #[jit_release_gil]
    pub fn gil_fn(x: i64) -> i64 {
        x * 3
    }

    #[jit_loop_invariant]
    pub fn invariant_fn(x: i64) -> i64 {
        x / 2
    }
}

#[test]
fn test_all_attribute_types_discovered() {
    let helpers = multi_attr_module::__MAJIT_DISCOVERED_HELPERS;
    assert_eq!(helpers.len(), 5);
    assert!(helpers.contains(&"pure_fn"));
    assert!(helpers.contains(&"opaque_fn"));
    assert!(helpers.contains(&"force_fn"));
    assert!(helpers.contains(&"gil_fn"));
    assert!(helpers.contains(&"invariant_fn"));
}

#[test]
fn test_all_attribute_policies() {
    let policies = multi_attr_module::__MAJIT_HELPER_POLICIES;
    assert_eq!(policies.len(), 5);
    assert!(policies.contains(&("pure_fn", "elidable")));
    assert!(policies.contains(&("opaque_fn", "dont_look_inside")));
    assert!(policies.contains(&("force_fn", "jit_may_force")));
    assert!(policies.contains(&("gil_fn", "jit_release_gil")));
    assert!(policies.contains(&("invariant_fn", "jit_loop_invariant")));
}

#[test]
fn test_multi_attr_functions_callable() {
    assert_eq!(multi_attr_module::pure_fn(5), 10);
    assert_eq!(multi_attr_module::opaque_fn(5), 6);
    assert_eq!(multi_attr_module::force_fn(5), 4);
    assert_eq!(multi_attr_module::gil_fn(5), 15);
    assert_eq!(multi_attr_module::invariant_fn(10), 5);
}

// `#[jit_module]` discovers JIT helpers inside `impl` blocks. Both
// inherent (`impl Foo { fn ... }`) and trait-impl
// (`impl Trait for Foo { fn ... }`) methods land in the same registry
// via the structured `__majit_helper_impl_trace_fnaddrs()` registry,
// keyed by `impl_type_joined / method` that matches the parser's
// `self_ty_root` canonicalization (parse.rs:702, lib.rs:406-433) —
// RPython `call.py:174-187 getfunctionptr(graph)` parity.
//
// `#[unroll_safe]` is used here because it is one of the JIT attribute
// macros that does not generate out-of-scope trampolines; applying it
// on an impl method simply re-emits the method body.  Instance methods
// (`&self` / `&mut self` / `self`) are also exercised — Rust allows
// `<Type>::method as fn(&Type)` coercion, and RPython upstream treats
// `getfunctionptr(graph)` uniformly across free fns and methods.
#[jit_module]
mod impl_walk_module {
    use majit_macros::unroll_safe;

    pub struct Adder {
        pub value: i64,
    }

    impl Adder {
        #[unroll_safe]
        pub fn add(a: i64, b: i64) -> i64 {
            a + b
        }

        #[unroll_safe]
        pub fn bump(&self, x: i64) -> i64 {
            self.value + x
        }

        // No JIT attribute — must be skipped by discovery.
        pub fn ignore_me(_x: i64) -> i64 {
            0
        }
    }
}

#[test]
fn test_jit_module_discovers_impl_methods_including_receivers() {
    let helpers = impl_walk_module::__MAJIT_DISCOVERED_HELPERS;
    // Both the associated fn and the `&self` method must land in the
    // registry — discovery no longer excludes receiver methods.
    assert!(
        helpers.contains(&"Adder::add"),
        "free-signature impl method must be discovered, got {helpers:?}"
    );
    assert!(
        helpers.contains(&"Adder::bump"),
        "&self instance method must be discovered, got {helpers:?}"
    );
    assert!(!helpers.contains(&"Adder::ignore_me"));
    assert_eq!(helpers.len(), 2);

    let policies = impl_walk_module::__MAJIT_HELPER_POLICIES;
    assert!(policies.contains(&("Adder::add", "unroll_safe")));
    assert!(policies.contains(&("Adder::bump", "unroll_safe")));
}

#[test]
fn test_jit_module_emits_structured_impl_trace_fnaddrs() {
    let entries = impl_walk_module::__majit_helper_impl_trace_fnaddrs();
    // Entries are
    //   `(module_path_with_crate, impl_type_as_written, method, fnaddr)`
    // 4-tuples. The codewriter applies `qualify_type_name`
    // (front/ast.rs:106) using `module_path_with_crate` to decide
    // whether to prepend the module prefix.
    assert_eq!(entries.len(), 2);

    let expected_module_path = concat!(module_path!(), "::impl_walk_module");

    let add_entry = entries
        .iter()
        .find(|(_, ty, m, _)| *ty == "Adder" && *m == "add")
        .expect("Adder::add entry");
    assert_eq!(add_entry.0, expected_module_path);
    assert_eq!(
        add_entry.3,
        impl_walk_module::Adder::add as *const () as usize as i64,
    );

    let bump_entry = entries
        .iter()
        .find(|(_, ty, m, _)| *ty == "Adder" && *m == "bump")
        .expect("Adder::bump entry");
    assert_eq!(bump_entry.0, expected_module_path);
    // Casting a `&self` associated fn to a plain `*const ()` works —
    // confirms Rust allows the coercion the reviewer called out.
    assert_eq!(
        bump_entry.3,
        impl_walk_module::Adder::bump as *const () as usize as i64,
    );
}

// Trait-impl disambiguation: when a type implements a trait method
// whose name could collide with an inherent method (or with another
// trait's method), the macro must emit `<Type as Trait>::method` to
// keep the fnaddr cast unambiguous. RPython `getfunctionptr(graph)`
// (call.py:174) does not need this because it uses graph identity;
// pyre's Rust-layer registry does need it.
pub trait NameCollider {
    fn conflict(&self) -> i64;
}

#[jit_module]
mod trait_impl_module {
    use super::NameCollider;
    use majit_macros::unroll_safe;

    pub struct Widget {
        pub value: i64,
    }

    // Inherent method with the same name as the trait's — without
    // `<Widget as NameCollider>::conflict` disambiguation the fnaddr
    // cast would be rejected by rustc.
    impl Widget {
        #[unroll_safe]
        pub fn conflict(&self) -> i64 {
            self.value * 10
        }
    }

    impl NameCollider for Widget {
        #[unroll_safe]
        fn conflict(&self) -> i64 {
            self.value + 1
        }
    }
}

#[test]
fn test_jit_module_disambiguates_trait_impl_with_as_trait_cast() {
    let entries = trait_impl_module::__majit_helper_impl_trace_fnaddrs();
    assert_eq!(entries.len(), 2, "both methods discovered: {entries:?}");

    let widget = trait_impl_module::Widget { value: 3 };
    // Find both entries.
    let mut seen_inherent = false;
    let mut seen_trait = false;
    for (_module_path, ty, method, fnaddr) in &entries {
        assert_eq!(*ty, "Widget");
        assert_eq!(*method, "conflict");
        // The fnaddr is one of the two method addresses; call through
        // the address to verify which.
        let fp: fn(&trait_impl_module::Widget) -> i64 =
            unsafe { std::mem::transmute(*fnaddr as usize) };
        let result = fp(&widget);
        if result == 30 {
            seen_inherent = true;
        } else if result == 4 {
            seen_trait = true;
        } else {
            panic!("unexpected conflict() result {result}");
        }
    }
    assert!(seen_inherent, "inherent Widget::conflict entry missing");
    assert!(
        seen_trait,
        "<Widget as NameCollider>::conflict entry missing"
    );
}

// ── `#[jit_elidable]` impl-method attachment ──────────────────────
//
// `#[elidable]` / `#[elidable_cannot_raise]` and friends emit a
// module-level trampoline (`__majit_call_target_*`), so they cannot be
// attached inside an `impl` block (probe result: `not found in this
// scope`).  `#[jit_elidable]` (lib.rs:993) is a pure pass-through and
// only relies on `front::ast::collect_jit_hints`'s hint flip, so it can
// safely sit on impl methods.  This fixture verifies that live wire.
#[jit_module]
mod elidable_method_module {
    use majit_macros::jit_elidable;

    pub struct PureCalc {
        pub seed: i64,
    }

    impl PureCalc {
        // Free-style impl method (no receiver).
        #[jit_elidable]
        pub fn compute_xor(a: i64, b: i64) -> i64 {
            a ^ b
        }

        // `&self` instance method.
        #[jit_elidable]
        pub fn shifted(&self, x: i64) -> i64 {
            self.seed.wrapping_add(x)
        }
    }
}

#[test]
fn test_jit_elidable_on_impl_methods_is_discovered() {
    let helpers = elidable_method_module::__MAJIT_DISCOVERED_HELPERS;
    assert!(
        helpers.contains(&"PureCalc::compute_xor"),
        "free-style elidable method must be discovered, got {helpers:?}"
    );
    assert!(
        helpers.contains(&"PureCalc::shifted"),
        "&self elidable method must be discovered, got {helpers:?}"
    );

    let policies = elidable_method_module::__MAJIT_HELPER_POLICIES;
    // jit_module records the raw attribute name; normalisation happens
    // in `front::ast::collect_jit_hints` (front/ast.rs:1971), which
    // flips "jit_elidable" → "elidable" before mark_elidable consumes
    // the hint.
    assert!(policies.contains(&("PureCalc::compute_xor", "jit_elidable")));
    assert!(policies.contains(&("PureCalc::shifted", "jit_elidable")));
}

#[test]
fn test_jit_elidable_method_runtime_callable() {
    // `#[jit_elidable]` is pass-through, so runtime behaviour is unchanged.
    assert_eq!(
        elidable_method_module::PureCalc::compute_xor(0xff, 0x0f),
        0xf0
    );
    let calc = elidable_method_module::PureCalc { seed: 100 };
    assert_eq!(calc.shifted(7), 107);
}

// ── pass-through attribute on a *free* fn under `#[jit_module]` ───
//
// `#[jit_elidable]` (lib.rs:1008) is a pure pass-through and emits no
// `__majit_call_policy_<name>()` trampoline.  `__majit_helper_trace_fnaddrs()`
// must therefore record the function's direct address (impl_addr_expr's
// pass-through branch) instead of routing through the missing policy fn —
// without that branch the `#[jit_module]` expansion fails to compile with
// `cannot find function __majit_call_policy_*`.  Other pass-through attrs
// (`#[unroll_safe]`, `#[not_in_trace]`) must take the same direct path; the
// fixture exercises one of each to pin the static surface in tests.
#[jit_module]
mod passthrough_free_fn_module {
    use majit_macros::{jit_elidable, not_in_trace, unroll_safe};

    #[jit_elidable]
    pub fn pure_xor(a: i64, b: i64) -> i64 {
        a ^ b
    }

    #[unroll_safe]
    pub fn unrolled(x: i64) -> i64 {
        x.wrapping_add(1)
    }

    #[not_in_trace]
    pub fn out_of_trace(x: i64) -> i64 {
        x.wrapping_mul(2)
    }
}

#[test]
fn test_passthrough_free_fn_discovery_uses_direct_fn_address() {
    let helpers = passthrough_free_fn_module::__MAJIT_DISCOVERED_HELPERS;
    assert!(helpers.contains(&"pure_xor"));
    assert!(helpers.contains(&"unrolled"));
    assert!(helpers.contains(&"out_of_trace"));

    let policies = passthrough_free_fn_module::__MAJIT_HELPER_POLICIES;
    // jit_module records the raw attribute name; normalisation lives
    // in front/ast.rs:2147.
    assert!(policies.contains(&("pure_xor", "jit_elidable")));
    assert!(policies.contains(&("unrolled", "unroll_safe")));
    assert!(policies.contains(&("out_of_trace", "not_in_trace")));

    let trace_fnaddrs = passthrough_free_fn_module::__majit_helper_trace_fnaddrs();
    assert_eq!(trace_fnaddrs.len(), 3);
    // No `__majit_call_policy_*` exists for any of these, so the only
    // legal address is the function's direct cast.
    assert!(trace_fnaddrs.iter().any(|(path, addr)| {
        *path == concat!(module_path!(), "::passthrough_free_fn_module::pure_xor")
            && *addr == passthrough_free_fn_module::pure_xor as *const () as usize as i64
    }));
    assert!(trace_fnaddrs.iter().any(|(path, addr)| {
        *path == concat!(module_path!(), "::passthrough_free_fn_module::unrolled")
            && *addr == passthrough_free_fn_module::unrolled as *const () as usize as i64
    }));
    assert!(trace_fnaddrs.iter().any(|(path, addr)| {
        *path == concat!(module_path!(), "::passthrough_free_fn_module::out_of_trace")
            && *addr == passthrough_free_fn_module::out_of_trace as *const () as usize as i64
    }));
}

#[test]
fn test_passthrough_free_fn_callable() {
    assert_eq!(passthrough_free_fn_module::pure_xor(0b1010, 0b0110), 0b1100);
    assert_eq!(passthrough_free_fn_module::unrolled(7), 8);
    assert_eq!(passthrough_free_fn_module::out_of_trace(5), 10);
}
