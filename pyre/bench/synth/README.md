# Pyre Synthetic Benchmark Suite

This directory contains small deterministic benchmarks grouped by common Python
language features.  They are meant to expose Pyre/PyPy parity gaps by comparing
stdout and runtime across interpreters.

Run all cases with CPython only:

```sh
python3 pyre/check_synthetic.py
```

Compare against PyPy and a Pyre binary:

```sh
python3 pyre/check_synthetic.py --pypy pypy3 --pyre ./target/release/pyre-dynasm
```

Each benchmark prints a stable checksum.  A Pyre failure is useful signal: it
marks a feature category that needs trace, optimizer, backend, or frontend
parity work.

