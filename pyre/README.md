# pyre

[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/2fAUZ49JX3)

**Py**thon **Re**written — a no-GIL Python implementation in Rust, with a meta-tracing JIT compiler ported from PyPy.

## Why pyre?

PyPy proved that a meta-tracing JIT can make Python fast. pyre takes that proven architecture and rebuilds it in Rust — gaining memory safety, no GIL, and a modern toolchain, while keeping the same optimization pipeline that makes PyPy fast.

The key insight: pyre's JIT framework [MaJIT](../majit/) handles tracing, optimization, and native code generation. This means the pyre interpreter itself can stay close to a straightforward Rust program that executes Python bytecodes, while MaJIT provides the tracing JIT machinery around it. In the same way that PyPy is "just a Python interpreter" that RPython makes fast, pyre is "just a Rust interpreter" that MaJIT makes fast.

## Status

pyre is under active development. Loop tracing and function inlining work. On integer-heavy workloads, current results still trail PyPy but increasingly follow it. Many Python features are not yet implemented.

## Benchmarks

Measured on Apple M-series, single core:

| Benchmark | pyre (JIT) | PyPy 7.3 | CPython 3.14 | vs PyPy | vs CPython |
|-----------|-----------|----------|--------------|---------|-------------|
| int_loop | 0.06s | 0.04s | 1.86s | 1.5x slower | 31.0x faster |
| fib_loop | 0.08s | 0.06s | 0.11s | 1.3x slower | 1.4x faster |
| inline_helper | 0.04s | 0.04s | 1.45s | parity | 36.3x faster |
| fib_recursive | 0.16s | 0.08s | 0.87s | 2.0x slower | 5.4x faster |
| nbody | 1.90s | 0.03s | 0.21s | 63.3x slower | 9.0x slower |
| fannkuch | 2.37s | 0.05s | 0.26s | 47.4x slower | 9.1x slower |
| raise_catch | 0.34s | 0.01s | 0.06s | 34.0x slower | 5.7x slower |
| spectral_norm | 0.24s | 0.01s | 0.04s | 24.0x slower | 6.0x slower |

Integer-heavy benchmarks where the JIT fires still trail PyPy, but the gap is smaller there than on float-heavy or exception-heavy workloads. Float-heavy workloads (nbody, spectral_norm) and exception-heavy paths (raise_catch) run correctly but are not yet JIT-compiled — they fall back to the interpreter.

Run `python pyre/check.py` to reproduce all benchmarks with CPython / PyPy / pyre comparison on your machine.

## Installation

### Homebrew

```sh
brew install youknowone/tap/pyrex
```

The formula lives in the [youknowone/homebrew-tap](https://github.com/youknowone/homebrew-tap/tree/main/Formula).

### Prebuilt binaries

Download a prebuilt binary from the [GitHub releases page](https://github.com/youknowone/pyre/releases).

### Cargo

```sh
cargo install pyrex
```

## Building from source

```sh
cargo build --release -p pyrex
./target/release/pyre script.py
```

## How it works

pyre follows PyPy's meta-tracing approach:

1. The **interpreter** (`pyre-interpreter`) executes Python bytecodes normally.
2. When a loop or function becomes hot, **MaJIT** records the interpreter's execution as a linear trace of IR operations.
3. The trace passes through an **8-pass optimizer** — the same pipeline as PyPy: IntBounds, Rewrite, Virtualize, String, Pure, Guard, Simplify, Heap.
4. The optimized IR is compiled to **native machine code** via Cranelift.
5. Subsequent executions of that path run the compiled code directly. Guard failures fall back to the interpreter.

### Function inlining

During loop tracing, pyre traces *through* function call boundaries. A call to `add(a, b)` in the loop body becomes `IntAddOvf(a_raw, b_raw)` in the compiled trace — no function call overhead, no frame allocation.

### no-GIL

pyre has no Global Interpreter Lock. RPython/PyPy features that depend on the GIL have no equivalent trigger path in pyre. The API surfaces are kept for naming parity with the original codebase but have no production call sites.

## Crate structure

```
pyre/
├── pyre-object      # Python object types (W_IntObject, W_FloatObject, W_ListObject, ...)
├── pyre-bytecode    # Bytecode definitions (re-exports RustPython compiler-core)
├── pyre-interpreter # Object space, interpreter frame, eval loop, opcode dispatch
├── pyre-jit         # JIT integration — trace recording, call bridges, inlining
└── pyrex            # Executable entry point (builds the `pyre` binary)
```

## Relationship to PyPy

pyre is a structural port of PyPy's interpreter (`pypy/interpreter/` and `pypy/objspace/`). Every module, type, and function in the original Python codebase exists in the Rust port under the same name at the same relative location — only snake_case conversion is applied to method names. This naming parity makes it possible to read the PyPy source alongside pyre and see exactly what each piece corresponds to.

## Key differences from PyPy

- **No annotator/rtyper.** RPython translates Python to C via a whole-program type inferencer. pyre runs `majit-analyze` at `cargo build` time instead — same role, but static analysis on Rust source rather than RPython translation.
- **Proc macros instead of decorators.** `@jit.elidable` becomes `#[elidable]`, `driver.jit_merge_point(...)` becomes `jit_merge_point!`. Same semantics, Rust syntax.
- **No GIL.** pyre is free-threaded from day one. GIL-dependent code paths in PyPy (heapcache resets on GIL release, `release_gil` effect info, etc.) simply don't exist.
- **Python 3.14, not 2.7.** PyPy's main branch targets Python 2.7/3.10. pyre targets CPython 3.14 bytecodes directly, using RustPython's compiler frontend.

## Relationship to MaJIT

[MaJIT](../majit/) (**M**eta-tr**A**cing **JIT**) is a standalone Rust port of RPython's JIT infrastructure. It is a general-purpose framework for Rust bytecode interpreters that integrate with its tracing interface. pyre is MaJIT's primary consumer, but MaJIT has no dependency on pyre.

## Roadmap

What's next, roughly in priority order:

- **Float-heavy JIT paths** — nbody and fannkuch run correctly but don't yet JIT-compile float operations; closing this gap is the biggest performance unlock remaining.
- **More Python built-ins** — str methods, dict operations, list comprehensions, generators.
- **Exception-heavy JIT** — raise/catch inside traced loops (raise_catch benchmark works interpreted, JIT bridge in progress).
- **Multi-threaded execution** — the no-GIL foundation is there; actual parallel thread scheduling is not.
- **CPython C extension compatibility** — long-term goal, likely via HPy or similar ABI layer.

## Name

**pyrex** = **pyre** e**x**ecutable. The `pyrex` crate builds the `pyre` command-line binary.

## License

MIT — same as [PyPy](https://github.com/pypy/pypy/blob/main/LICENSE).
