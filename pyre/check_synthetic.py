#!/usr/bin/env python3
"""Run synthetic Python feature benchmarks across interpreters.

This runner is intentionally separate from pyre/check.py.  pyre/check.py is a
pre-merge gate with tuned thresholds; this file is a parity discovery tool.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BENCH_DIR = ROOT / "bench" / "synth"


def green(s):
    return f"\033[32m{s}\033[0m"


def red(s):
    return f"\033[31m{s}\033[0m"


def dim(s):
    return f"\033[2m{s}\033[0m"


def run(args, timeout):
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, "", "timeout", timeout
    except FileNotFoundError as e:
        elapsed = time.perf_counter() - start
        return 127, "", str(e), elapsed
    elapsed = time.perf_counter() - start
    return proc.returncode, proc.stdout.replace("\r\n", "\n"), proc.stderr, elapsed


def iter_benches(pattern):
    benches = []
    for path in sorted(BENCH_DIR.glob(pattern)):
        if path.name == "README.md":
            continue
        benches.append(path)
    return benches


def main():
    parser = argparse.ArgumentParser(description="Synthetic Pyre parity benchmark runner")
    parser.add_argument("--python", default=os.environ.get("PYRE_SYNTH_PYTHON") or sys.executable)
    parser.add_argument("--pypy", default=os.environ.get("PYRE_SYNTH_PYPY") or shutil.which("pypy3") or "")
    parser.add_argument("--pyre", default=os.environ.get("PYRE_SYNTH_PYRE") or "")
    parser.add_argument("--pattern", default="*.py")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    interpreters = [("python", args.python)]
    if args.pypy:
        interpreters.append(("pypy", args.pypy))
    if args.pyre:
        interpreters.append(("pyre", args.pyre))

    benches = iter_benches(args.pattern)
    if not benches:
        print(f"no benchmarks matched {args.pattern!r} under {BENCH_DIR}")
        return 2

    failures = []
    print("Synthetic parity benchmarks")
    print(f"benchmarks: {len(benches)}")
    print("interpreters: " + " ".join(name for name, _ in interpreters))
    print()

    for bench in benches:
        print(bench.name)
        baseline = None
        for name, exe in interpreters:
            rc, out, err, elapsed = run([exe, str(bench)], args.timeout)
            if baseline is None and name == "python" and rc == 0:
                baseline = out

            status = green("PASS")
            detail = out.strip().splitlines()[-1] if out.strip() else ""
            if rc != 0:
                status = red("FAIL")
                last_error = err.strip().splitlines()[-1] if err.strip() else ""
                detail = f"exit={rc} {last_error}"
                failures.append((bench.name, name, "exit", rc))
            elif baseline is not None and out != baseline:
                status = red("WRONG")
                detail = f"got={out.strip()!r} expected={baseline.strip()!r}"
                failures.append((bench.name, name, "wrong-output", 0))
            print(f"  {name:<8s} {status:<14s} {elapsed:6.2f}s  {detail}")
        print()

    if failures:
        print(red("FAILED"))
        for bench, name, kind, rc in failures:
            print(f"  {bench}: {name} {kind} {rc}")
        return 1

    print(green("ALL PASSED"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
