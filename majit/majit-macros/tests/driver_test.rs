use majit_macros::{
    dont_look_inside, elidable, elidable_cannot_raise, elidable_or_memerror, jit_driver,
};

#[jit_driver(greens = [pc, code], reds = [frame])]
struct MyDriver;

#[test]
fn test_driver_greens() {
    assert_eq!(MyDriver::GREENS, &["pc", "code"]);
}

#[test]
fn test_driver_reds() {
    assert_eq!(MyDriver::REDS, &["frame"]);
}

#[test]
fn test_driver_num_greens() {
    assert_eq!(MyDriver::NUM_GREENS, 2);
}

#[test]
fn test_driver_num_reds() {
    assert_eq!(MyDriver::NUM_REDS, 1);
}

#[test]
fn test_driver_num_vars() {
    assert_eq!(MyDriver::NUM_VARS, 3);
}

#[jit_driver(greens = [pc], reds = [frame, stack])]
struct SingleGreenDriver;

#[test]
fn test_single_green_driver() {
    assert_eq!(SingleGreenDriver::GREENS, &["pc"]);
    assert_eq!(SingleGreenDriver::REDS, &["frame", "stack"]);
    assert_eq!(SingleGreenDriver::NUM_GREENS, 1);
    assert_eq!(SingleGreenDriver::NUM_REDS, 2);
    assert_eq!(SingleGreenDriver::NUM_VARS, 3);
}

#[elidable]
fn compute(x: i64) -> i64 {
    x * x + 1
}

#[test]
fn test_elidable_function() {
    assert_eq!(compute(5), 26);
    assert_eq!(compute(0), 1);
    assert_eq!(compute(-3), 10);
    // EF_ELIDABLE_CAN_RAISE — call.py:297 `elif cr:` branch.
    let (policy, _, _, _, _) = __majit_call_policy_compute();
    assert_eq!(policy, 3u8);
}

#[elidable_cannot_raise]
fn compute_pure(x: i64) -> i64 {
    x * 2
}

#[test]
fn test_elidable_cannot_raise_function() {
    assert_eq!(compute_pure(7), 14);
    // EF_ELIDABLE_CANNOT_RAISE — call.py:299 `else` branch.
    let (policy, _, _, _, _) = __majit_call_policy_compute_pure();
    assert_eq!(policy, 19u8);
}

#[elidable_or_memerror]
fn compute_memerror(x: i64) -> i64 {
    x + 100
}

#[test]
fn test_elidable_or_memerror_function() {
    assert_eq!(compute_memerror(7), 107);
    // EF_ELIDABLE_OR_MEMORYERROR — call.py:295 `if cr == "mem":`.
    let (policy, _, _, _, _) = __majit_call_policy_compute_memerror();
    assert_eq!(policy, 20u8);
}

#[dont_look_inside]
fn opaque_call(x: i64, y: i64) -> i64 {
    x + y
}

#[test]
fn test_opaque_function() {
    assert_eq!(opaque_call(2, 3), 5);
    assert_eq!(opaque_call(-1, 1), 0);
}
