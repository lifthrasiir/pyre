//! Slice 4.2 regression pin: `ExceptionState` clone/clear preserves
//! pending state.
//!
//! Multi-frame guard-fail resume threads `&ExceptionState` through
//! `restore_guard_failure_with_session_cache` (`jit_state.rs:885-991`)
//! into `restore_reconstructed_frame_values_with_metadata`. The resume
//! side cannot drain pending state from the live interpreter — the
//! borrow is read-only via `&ExceptionState`. This test pins the
//! invariant by exercising the underlying `ExceptionState` API: a
//! clone of a pending exception remains pending; clearing the clone
//! does not affect the original. Mirrors the value-semantics
//! guarantee that `pyjitpl.py:2479-2538 finishframe_exception` relies
//! on across frame boundaries.
use majit_metainterp::blackhole::ExceptionState;

#[test]
fn exception_state_clone_preserves_pending_state() {
    let mut exc = ExceptionState::default();
    exc.set(0xdead_beef_i64, 0xfeed_face_i64);

    let snapshot = exc.clone();
    assert!(snapshot.is_pending());

    let (cls, val) = snapshot.clone().clear();
    assert_eq!(cls, 0xdead_beef_i64);
    assert_eq!(val, 0xfeed_face_i64);

    // Clearing the snapshot's clone must not drain the live exception.
    assert!(exc.is_pending());
    assert_eq!(exc.exc_class, 0xdead_beef_i64);
    assert_eq!(exc.exc_value, 0xfeed_face_i64);
}
