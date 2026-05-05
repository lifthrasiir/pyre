//! No direct RPython equivalent — thread-local I/O buffer for compiled
//! loops (RPython interpreter writes to stdout directly).

use std::cell::RefCell;
use std::fmt;
use std::io::{self, Write};

use majit_ir::{OpCode, Type};

use crate::call_descr::{CANNOT_RAISE_NO_HEAP_EFFECT_INFO, make_call_descr_with_effect};
use crate::trace_ctx::TraceCtx;

// ── Thread-local I/O buffer ──────────────────────────────────────────
//
// JIT-compiled loops call extern "C" shim functions that write to this
// buffer instead of directly to stdout. At each successful loop iteration,
// commit flushes the buffer. On guard failure, discard clears it.

thread_local! {
    static JIT_IO_BUFFER: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(4096));
}

/// Write raw bytes to the JIT I/O buffer.
pub fn io_buffer_write(data: &[u8]) {
    JIT_IO_BUFFER.with(|buf| {
        buf.borrow_mut().extend_from_slice(data);
    });
}

/// Write formatted output to the JIT I/O buffer.
pub fn io_buffer_write_fmt(args: fmt::Arguments<'_>) {
    JIT_IO_BUFFER.with(|buf| {
        let _ = buf.borrow_mut().write_fmt(args);
    });
}

/// Flush the JIT I/O buffer to stdout.
pub fn io_buffer_commit() {
    JIT_IO_BUFFER.with(|buf| {
        let mut b = buf.borrow_mut();
        if !b.is_empty() {
            let stdout = io::stdout();
            let mut out = stdout.lock();
            let _ = out.write_all(&b);
            let _ = out.flush();
            b.clear();
        }
    });
}

/// Discard the JIT I/O buffer contents.
pub fn io_buffer_discard() {
    JIT_IO_BUFFER.with(|buf| {
        buf.borrow_mut().clear();
    });
}

// ── Reusable I/O shims ───────────────────────────────────────────────
//
// Common JIT I/O operations that interpreters can use directly via
// io_shims in `#[jit_interp]` instead of defining their own.

/// Encode an i64 as decimal into a fixed buffer, returning the used slice.
pub fn encode_decimal_i64(value: i64, buf: &mut [u8; 20]) -> &[u8] {
    if value == 0 {
        buf[19] = b'0';
        return &buf[19..];
    }
    let negative = value < 0;
    // Use u64 for magnitude to handle i64::MIN correctly
    let mut v: u64 = if negative {
        (value as u64).wrapping_neg()
    } else {
        value as u64
    };
    let mut pos = 20;
    while v > 0 {
        pos -= 1;
        buf[pos] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    if negative {
        pos -= 1;
        buf[pos] = b'-';
    }
    &buf[pos..]
}

/// Write an i64 as its decimal string representation to the JIT I/O buffer.
///
/// Suitable as an `extern "C"` shim for JIT-compiled numeric output.
pub extern "C" fn jit_write_number_i64(value: i64) {
    match value {
        0 => io_buffer_write(b"0"),
        1 => io_buffer_write(b"1"),
        -1 => io_buffer_write(b"-1"),
        _ => {
            let mut buf = [0u8; 20];
            io_buffer_write(encode_decimal_i64(value, &mut buf));
        }
    }
}

/// Write an i64 interpreted as a Unicode codepoint (UTF-8 encoded) to the
/// JIT I/O buffer.
///
/// Suitable as an `extern "C"` shim for JIT-compiled character output.
pub extern "C" fn jit_write_utf8_codepoint(value: i64) {
    if (0..=0x7F).contains(&value) {
        io_buffer_write(&[value as u8]);
        return;
    }
    if let Some(c) = char::from_u32(value as u32) {
        let mut buf = [0u8; 4];
        io_buffer_write(c.encode_utf8(&mut buf).as_bytes());
    } else {
        io_buffer_write("\u{FFFD}".as_bytes());
    }
}

// The actual extern "C" function that commit_io CallN calls.
extern "C" fn jit_commit_io_shim() {
    io_buffer_commit();
}

/// Emit a CallN to commit the I/O buffer in the trace.
///
/// Should be called right before returning `TraceAction::CloseLoop`,
/// so that each successful loop iteration flushes its I/O.
pub fn emit_commit_io(ctx: &mut TraceCtx) {
    let func_ref = ctx.const_int(jit_commit_io_shim as *const () as usize as i64);
    // `call.py:303 getcalldescr`'s `else` branch — `_canraise(op) == False`
    // for `jit_commit_io_shim`: it flushes a thread-local `Vec<u8>` to
    // stdout (`io_buffer_commit`); no allocation that can `MemoryError`,
    // no Python-level dispatch that can raise. `EF_CANNOT_RAISE` lets
    // `do_residual_call` skip the trailing `GUARD_NO_EXCEPTION`
    // (`pyjitpl.py:2111-2115`).
    let descr = make_call_descr_with_effect(&[], Type::Void, CANNOT_RAISE_NO_HEAP_EFFECT_INFO);
    ctx.record_op_with_descr(OpCode::CallN, &[func_ref], descr);
}
