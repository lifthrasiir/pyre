//! C-ABI call stub dispatch shared between backends.
//!
//! `bh_call_i_dispatch` mirrors `rpython/jit/backend/llmodel.py:816 call_stub_i` —
//! the arity-table that materializes a typed `extern "C" fn` from a raw funcptr
//! and forwards integer + float register files independently per the SysV /
//! AAPCS C ABI.
//!
//! `bh_call_v_dispatch` is the void-return parallel of `bh_call_i_dispatch`,
//! mirroring `rpython/jit/backend/llsupport/llmodel.py:834 bh_call_v` /
//! `descr.py:598-612 create_call_stub` where `RESULT == lltype.Void` produces
//! a stub whose generated function signature returns nothing. Using a real
//! `extern "C" fn(...) -> ()` transmute matches the C ABI of genuinely void
//! callees instead of reading whatever rax/x0 happens to carry.
//!
//! `bh_call_v_dispatch` mirrors `bh_call_i_dispatch`'s arity table verbatim
//! through the shared `dispatch_arity_body!` macro so callers see identical
//! arity coverage regardless of return type.

/// Arity-table body shared by `bh_call_i_dispatch` and `bh_call_v_dispatch`.
/// `$ret` plugs into both the function-pointer signature and the dispatch
/// function's return type so each unit-arm just evaluates `f(...)` and
/// returns the produced value (`i64` or `()`).
///
/// `descr.py:598-612 create_call_stub` parity: a single ARGS×RESULT shape
/// per dispatch; we enumerate the same shape twice (once per RESULT) instead
/// of generating one stub per descriptor.
macro_rules! dispatch_arity_body {
    ($func:ident, $int_args:ident, $float_args:ident, $ret:ty) => {{
        type I = i64;
        type F = f64;
        match ($int_args.len(), $float_args.len()) {
            // No float args — integer-only calls (0..=16 to match
            // `pyjitpl/dispatch.rs::call_int_function` /
            // `call_void_function` MAX_HOST_CALL_ARITY = 16).
            (0, 0) => {
                let f: unsafe extern "C" fn() -> $ret = std::mem::transmute($func);
                f()
            }
            (1, 0) => {
                let f: unsafe extern "C" fn(I) -> $ret = std::mem::transmute($func);
                f($int_args[0])
            }
            (2, 0) => {
                let f: unsafe extern "C" fn(I, I) -> $ret = std::mem::transmute($func);
                f($int_args[0], $int_args[1])
            }
            (3, 0) => {
                let f: unsafe extern "C" fn(I, I, I) -> $ret = std::mem::transmute($func);
                f($int_args[0], $int_args[1], $int_args[2])
            }
            (4, 0) => {
                let f: unsafe extern "C" fn(I, I, I, I) -> $ret = std::mem::transmute($func);
                f($int_args[0], $int_args[1], $int_args[2], $int_args[3])
            }
            (5, 0) => {
                let f: unsafe extern "C" fn(I, I, I, I, I) -> $ret = std::mem::transmute($func);
                f(
                    $int_args[0],
                    $int_args[1],
                    $int_args[2],
                    $int_args[3],
                    $int_args[4],
                )
            }
            (6, 0) => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I) -> $ret = std::mem::transmute($func);
                f(
                    $int_args[0],
                    $int_args[1],
                    $int_args[2],
                    $int_args[3],
                    $int_args[4],
                    $int_args[5],
                )
            }
            (7, 0) => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $int_args[0],
                    $int_args[1],
                    $int_args[2],
                    $int_args[3],
                    $int_args[4],
                    $int_args[5],
                    $int_args[6],
                )
            }
            (8, 0) => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $int_args[0],
                    $int_args[1],
                    $int_args[2],
                    $int_args[3],
                    $int_args[4],
                    $int_args[5],
                    $int_args[6],
                    $int_args[7],
                )
            }
            (9, 0) => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $int_args[0],
                    $int_args[1],
                    $int_args[2],
                    $int_args[3],
                    $int_args[4],
                    $int_args[5],
                    $int_args[6],
                    $int_args[7],
                    $int_args[8],
                )
            }
            (10, 0) => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $int_args[0],
                    $int_args[1],
                    $int_args[2],
                    $int_args[3],
                    $int_args[4],
                    $int_args[5],
                    $int_args[6],
                    $int_args[7],
                    $int_args[8],
                    $int_args[9],
                )
            }
            (11, 0) => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $int_args[0],
                    $int_args[1],
                    $int_args[2],
                    $int_args[3],
                    $int_args[4],
                    $int_args[5],
                    $int_args[6],
                    $int_args[7],
                    $int_args[8],
                    $int_args[9],
                    $int_args[10],
                )
            }
            (12, 0) => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $int_args[0],
                    $int_args[1],
                    $int_args[2],
                    $int_args[3],
                    $int_args[4],
                    $int_args[5],
                    $int_args[6],
                    $int_args[7],
                    $int_args[8],
                    $int_args[9],
                    $int_args[10],
                    $int_args[11],
                )
            }
            (13, 0) => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $int_args[0],
                    $int_args[1],
                    $int_args[2],
                    $int_args[3],
                    $int_args[4],
                    $int_args[5],
                    $int_args[6],
                    $int_args[7],
                    $int_args[8],
                    $int_args[9],
                    $int_args[10],
                    $int_args[11],
                    $int_args[12],
                )
            }
            (14, 0) => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $int_args[0],
                    $int_args[1],
                    $int_args[2],
                    $int_args[3],
                    $int_args[4],
                    $int_args[5],
                    $int_args[6],
                    $int_args[7],
                    $int_args[8],
                    $int_args[9],
                    $int_args[10],
                    $int_args[11],
                    $int_args[12],
                    $int_args[13],
                )
            }
            (15, 0) => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $int_args[0],
                    $int_args[1],
                    $int_args[2],
                    $int_args[3],
                    $int_args[4],
                    $int_args[5],
                    $int_args[6],
                    $int_args[7],
                    $int_args[8],
                    $int_args[9],
                    $int_args[10],
                    $int_args[11],
                    $int_args[12],
                    $int_args[13],
                    $int_args[14],
                )
            }
            (16, 0) => {
                let f: unsafe extern "C" fn(
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                ) -> $ret = std::mem::transmute($func);
                f(
                    $int_args[0],
                    $int_args[1],
                    $int_args[2],
                    $int_args[3],
                    $int_args[4],
                    $int_args[5],
                    $int_args[6],
                    $int_args[7],
                    $int_args[8],
                    $int_args[9],
                    $int_args[10],
                    $int_args[11],
                    $int_args[12],
                    $int_args[13],
                    $int_args[14],
                    $int_args[15],
                )
            }
            // Float-only calls.
            (0, 1) => {
                let f: unsafe extern "C" fn(F) -> $ret = std::mem::transmute($func);
                f($float_args[0])
            }
            (0, 2) => {
                let f: unsafe extern "C" fn(F, F) -> $ret = std::mem::transmute($func);
                f($float_args[0], $float_args[1])
            }
            (0, 3) => {
                let f: unsafe extern "C" fn(F, F, F) -> $ret = std::mem::transmute($func);
                f($float_args[0], $float_args[1], $float_args[2])
            }
            (0, 4) => {
                let f: unsafe extern "C" fn(F, F, F, F) -> $ret = std::mem::transmute($func);
                f(
                    $float_args[0],
                    $float_args[1],
                    $float_args[2],
                    $float_args[3],
                )
            }
            // Mixed int + float calls.
            (1, 1) => {
                let f: unsafe extern "C" fn(I, F) -> $ret = std::mem::transmute($func);
                f($int_args[0], $float_args[0])
            }
            (2, 1) => {
                let f: unsafe extern "C" fn(I, I, F) -> $ret = std::mem::transmute($func);
                f($int_args[0], $int_args[1], $float_args[0])
            }
            (1, 2) => {
                let f: unsafe extern "C" fn(I, F, F) -> $ret = std::mem::transmute($func);
                f($int_args[0], $float_args[0], $float_args[1])
            }
            (2, 2) => {
                let f: unsafe extern "C" fn(I, I, F, F) -> $ret = std::mem::transmute($func);
                f($int_args[0], $int_args[1], $float_args[0], $float_args[1])
            }
            (3, 1) => {
                let f: unsafe extern "C" fn(I, I, I, F) -> $ret = std::mem::transmute($func);
                f($int_args[0], $int_args[1], $int_args[2], $float_args[0])
            }
            (4, 1) => {
                let f: unsafe extern "C" fn(I, I, I, I, F) -> $ret = std::mem::transmute($func);
                f(
                    $int_args[0],
                    $int_args[1],
                    $int_args[2],
                    $int_args[3],
                    $float_args[0],
                )
            }
            (3, 2) => {
                let f: unsafe extern "C" fn(I, I, I, F, F) -> $ret = std::mem::transmute($func);
                f(
                    $int_args[0],
                    $int_args[1],
                    $int_args[2],
                    $float_args[0],
                    $float_args[1],
                )
            }
            (1, 3) => {
                let f: unsafe extern "C" fn(I, F, F, F) -> $ret = std::mem::transmute($func);
                f($int_args[0], $float_args[0], $float_args[1], $float_args[2])
            }
            (ni, nf) => {
                // PRE-EXISTING-ADAPTATION: upstream
                // `rpython/jit/backend/llsupport/descr.py:590-602
                // create_call_stub` generates a per-calldescr stub at
                // translation time so any (ni, nf) combination has a
                // matching extern "C" signature.  Rust has no
                // translation-time codegen equivalent, so the dispatch
                // is a hand-rolled arity table.  Convergence path:
                // wire libffi (or an ABI adapter) so any arity is
                // dispatchable; until then, callees outside the
                // table panic instead of silently corrupting registers.
                panic!(
                    "bh_call dispatch: unsupported arg combination ({ni} ints, {nf} floats); \
                     needs libffi for general dispatch"
                );
            }
        }
    }};
}

/// llmodel.py:816 call_stub_i: ABI-correct dispatch with separate int/float
/// register files. On ARM64/x86-64, integer args go to x0-x7 / rdi,rsi,... and
/// float args go to d0-d7 / xmm0-xmm7 independently.
///
/// Safety: `func` must be a valid function pointer matching the described ABI
/// — i.e. `extern "C" fn(I × ints, F × floats) -> i64` for the (ints, floats)
/// arity recovered from `int_args.len()` / `float_args.len()`.
pub unsafe fn bh_call_i_dispatch(func: usize, int_args: &[i64], float_args: &[f64]) -> i64 {
    unsafe { dispatch_arity_body!(func, int_args, float_args, i64) }
}

/// llmodel.py:834 bh_call_v: void-typed parallel of `bh_call_i_dispatch`.
///
/// Safety: `func` must be a valid function pointer matching the described
/// ABI — i.e. `extern "C" fn(I × ints, F × floats) -> ()` for the recovered
/// (ints, floats) arity. `descr.py:590-602 create_call_stub` builds a real
/// void-returning stub for `RESULT == lltype.Void`; calling such a function
/// through an `i64`-returning transmute reads garbage from rax/x0, so the
/// canonical `BC_RESIDUAL_CALL_*_V` blackhole/trace path must use this
/// dispatcher rather than `bh_call_i_dispatch`.
pub unsafe fn bh_call_v_dispatch(func: usize, int_args: &[i64], float_args: &[f64]) {
    unsafe { dispatch_arity_body!(func, int_args, float_args, ()) }
}

/// llmodel.py:825 bh_call_f: f64-typed parallel of `bh_call_i_dispatch`.
///
/// Safety: `func` must be a valid function pointer matching the described
/// ABI — i.e. `extern "C" fn(I × ints, F × floats) -> f64` for the recovered
/// (ints, floats) arity. `descr.py:590-602 create_call_stub` generates a
/// real f64-returning stub for `RESULT == lltype.Float`; the C ABI returns
/// f64 in xmm0 / d0 rather than rax / x0, so an `i64`-typed transmute
/// would read uninitialized integer-bank state.
pub unsafe fn bh_call_f_dispatch(func: usize, int_args: &[i64], float_args: &[f64]) -> f64 {
    unsafe { dispatch_arity_body!(func, int_args, float_args, f64) }
}

/// Bucket `args_i` / `args_r` / `args_f` slices into the (int, float) shape the
/// dispatch table expects, following `calldescr.arg_classes` order.
///
/// `arg_classes` is the per-argument class string from
/// `majit_translate::jitcode::BhCallDescr`. RPython
/// `rpython/jit/backend/llsupport/descr.py:545-571 create_call_stub`'s
/// `process(c)` defines the storage bank vs. C ABI register file mapping:
///
/// | class | storage bank | C ABI type           | register file |
/// |-------|--------------|----------------------|---------------|
/// | `i`   | `args_i`     | `lltype.Signed`      | int           |
/// | `r`   | `args_r`     | `llmemory.GCREF`     | int           |
/// | `f`   | `args_f`     | `lltype.Float`       | float         |
/// | `L`   | `args_f`     | `lltype.SignedLongLong` | int        |
/// | `S`   | `args_i`     | `lltype.SingleFloat` | float (f32)   |
///
/// Note the asymmetry: `L` is stored in the float bank (PyPy `process('L')`
/// rewrites `c = 'f'` for the storage lookup) yet passed in an integer
/// register, while `S` is stored in the int bank (PyPy
/// `int2singlefloat(args_i[..])`) yet passed in a float register as a
/// 32-bit value.
///
/// `S` currently panics: pyre's dispatch table only emits `extern "C" fn(.., f64, ..)`
/// arms, so an `f32` ABI cannot be transmuted accurately (a 64-bit movsd
/// vs. a 32-bit movss to the same xmm/d register file). Pyre's
/// `type_to_argclass` (`majit-translate/src/jit_codewriter/call.rs:190-197`)
/// never produces `S`, so the panic is unreachable from in-tree callers
/// today; reaching it requires a foreign-supplied calldescr (e.g. a
/// build-time bincode embed loaded from RPython).
///
/// Mirrors `rpython/jit/backend/llsupport/descr.py:614-620 verify_types`:
/// the per-class counts in `arg_classes` must match the corresponding list
/// length, and any unknown class is a codegen bug.
pub fn collect_call_args(
    arg_classes: &str,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
) -> (Vec<i64>, Vec<f64>) {
    // descr.py:614-620 verify_types parity: assert per-class counts.
    let count_i: usize = arg_classes
        .chars()
        .filter(|c| matches!(c, 'i' | 'S'))
        .count();
    let count_r: usize = arg_classes.chars().filter(|c| *c == 'r').count();
    let count_f: usize = arg_classes
        .chars()
        .filter(|c| matches!(c, 'f' | 'L'))
        .count();
    let len_i = args_i.map_or(0, <[i64]>::len);
    let len_r = args_r.map_or(0, <[i64]>::len);
    let len_f = args_f.map_or(0, <[i64]>::len);
    assert_eq!(
        count_i, len_i,
        "BhCallDescr.verify_types: arg_classes={arg_classes:?} has {count_i} int slots, args_i has {len_i}"
    );
    assert_eq!(
        count_r, len_r,
        "BhCallDescr.verify_types: arg_classes={arg_classes:?} has {count_r} ref slots, args_r has {len_r}"
    );
    assert_eq!(
        count_f, len_f,
        "BhCallDescr.verify_types: arg_classes={arg_classes:?} has {count_f} float slots, args_f has {len_f}"
    );

    let mut int_args: Vec<i64> = Vec::with_capacity(count_i + count_r);
    let mut float_args: Vec<f64> = Vec::with_capacity(count_f);
    let mut ii = 0usize;
    let mut ri = 0usize;
    let mut fi = 0usize;
    for c in arg_classes.chars() {
        match c {
            'i' => {
                int_args.push(args_i.expect("BhCallDescr.collect_call_args: args_i missing")[ii]);
                ii += 1;
            }
            'r' => {
                int_args.push(args_r.expect("BhCallDescr.collect_call_args: args_r missing")[ri]);
                ri += 1;
            }
            'f' => {
                let bits = args_f.expect("BhCallDescr.collect_call_args: args_f missing")[fi];
                float_args.push(f64::from_bits(bits as u64));
                fi += 1;
            }
            'L' => {
                // descr.py:546-548 process('L'): storage bank = `args_f`
                // (PyPy rewrites `c = 'f'` for the lookup); FUNC parameter
                // type = `lltype.SignedLongLong` → C `long long` →
                // 8-byte int dispatched in an integer register.
                int_args.push(args_f.expect("BhCallDescr.collect_call_args: args_f missing")[fi]);
                fi += 1;
            }
            'S' => {
                // descr.py:551-552 process('S'): storage bank = `args_i`
                // (PyPy reads via `int2singlefloat(args_i[..])`); FUNC
                // parameter type = `lltype.SingleFloat` → C `float` →
                // 32-bit float dispatched in an xmm/d register. pyre's
                // dispatch table emits only `extern "C" fn(.., f64, ..)`
                // arms, so transmuting f32 through f64 would mismatch the
                // C ABI (movss vs. movsd to the same register file).
                let _ = (ii, args_i);
                panic!(
                    "BhCallDescr.collect_call_args: 'S' (SingleFloat) ABI \
                     requires f32-aware dispatch; pyre's dispatch table \
                     only supports f64. arg_classes={arg_classes:?}"
                );
            }
            other => panic!(
                "BhCallDescr.collect_call_args: unsupported arg class {other:?} \
                 in arg_classes={arg_classes:?}"
            ),
        }
    }
    (int_args, float_args)
}
