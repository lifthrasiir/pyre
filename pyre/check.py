#!/usr/bin/env python3
"""pyre pre-merge check: correctness + regression guard + comparison

Cross-platform Python translation of pyre/check.sh.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

EXE = ".exe" if sys.platform == "win32" else ""
PYTHON3 = os.environ.get("PYRE_CHECK_PYTHON3") or (
    "python3" if shutil.which("python3") else "python"
)
PYPY3 = os.environ.get("PYRE_CHECK_PYPY3") or (
    "pypy3" if shutil.which("pypy3") else "pypy"
)

BENCH_DIR = "pyre/bench"
SNAP_DIR = "pyre/check.snap"
BENCH_COMPARE_BUFFER_S = 0.005

CARGO_CONFIG = {
    "dynasm": {
        "extra": ["--no-default-features", "--features", "dynasm"],
        "bin": "pyre-dynasm",
    },
    "cranelift": {
        "extra": ["--no-default-features", "--features", "cranelift"],
        "bin": "pyre-cranelift",
    },
}

# ── ANSI helpers ─────────────────────────────────────────────────────

def red(s):    return f"\033[31m{s}\033[0m"
def green(s):  return f"\033[32m{s}\033[0m"
def dim(s):    return f"\033[2m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"

# ── Child-process user CPU time ──────────────────────────────────────

def _run_timed_unix(args, timeout_s):
    import resource
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        proc = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return "", 0.0, 124
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    utime = max(after.ru_utime - before.ru_utime, 0.0)
    return proc.stdout.decode("utf-8", errors="replace"), utime, proc.returncode


def _run_timed_win32(args, timeout_s):
    import ctypes
    from ctypes import wintypes

    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        stdout_bytes, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return "", 0.0, 124

    utime = 0.0
    try:
        ct = wintypes.FILETIME()
        et = wintypes.FILETIME()
        kt = wintypes.FILETIME()
        ut = wintypes.FILETIME()
        handle = int(proc._handle)
        if ctypes.windll.kernel32.GetProcessTimes(
            handle,
            ctypes.byref(ct), ctypes.byref(et),
            ctypes.byref(kt), ctypes.byref(ut),
        ):
            utime = ((ut.dwHighDateTime << 32) | ut.dwLowDateTime) / 1e7
    except Exception:
        pass
    return stdout_bytes.decode("utf-8", errors="replace"), utime, proc.returncode


def run_timed(args, timeout_s=None):
    """Run *args*, return (stdout_str, user_cpu_seconds, returncode).

    returncode 124 = timeout (matching coreutils convention).
    """
    if sys.platform == "win32":
        return _run_timed_win32(args, timeout_s)
    return _run_timed_unix(args, timeout_s)

# ── Helpers ──────────────────────────────────────────────────────────

def scaled_timeout(base, scale):
    v = base * scale
    return int(v) if v == int(v) else float(f"{v:.3f}".rstrip("0").rstrip("."))


def fmt_time(t):
    if t is None or t == "-":
        return "-"
    return f"{t}s"


def default_binary(backend):
    name = CARGO_CONFIG[backend]["bin"]
    return f"./target/release/{name}{EXE}"

# ── Check runner ─────────────────────────────────────────────────────

class Check:
    def __init__(self, args):
        self.args = args
        self.results = []
        self.comparisons = []
        self.dynasm_pass = self.dynasm_fail = 0
        self.cranelift_pass = self.cranelift_fail = 0
        self.dynasm_pyre = ""
        self.cranelift_pyre = ""
        self.snapshot_diffs = []
        self.snapshot_missing = []

    # ── backend helpers ──

    def enabled(self, backend):
        return bool(self._pyre(backend))

    def _pyre(self, backend):
        return self.dynasm_pyre if backend == "dynasm" else self.cranelift_pyre

    def _timeout_scale(self, backend):
        if backend == "dynasm" and self.args.dynasm_timeout_scale is not None:
            return self.args.dynasm_timeout_scale
        if backend == "cranelift" and self.args.cranelift_timeout_scale is not None:
            return self.args.cranelift_timeout_scale
        return self.args.timeout_scale

    def _set_pyre(self, backend, path):
        if backend == "dynasm":
            self.dynasm_pyre = path
        else:
            self.cranelift_pyre = path

    # ── comparison table ──

    def _comp_index(self, name):
        for i, c in enumerate(self.comparisons):
            if c["name"] == name:
                return i
        return -1

    def _append_comparison(self, backend, name, t_cpython, t_pypy, pyre_field, note=""):
        idx = self._comp_index(name)
        if idx == -1:
            entry = {
                "name": name,
                "cpython": fmt_time(t_cpython),
                "pypy": fmt_time(t_pypy),
                "dynasm": "-",
                "cranelift": "-",
            }
            self.comparisons.append(entry)
            idx = len(self.comparisons) - 1
        else:
            self.comparisons[idx]["cpython"] = fmt_time(t_cpython)
            self.comparisons[idx]["pypy"] = fmt_time(t_pypy)

        cell = pyre_field
        if note:
            note = note.strip("()")
            if note.endswith(" vs pypy"):
                note = note[: -len(" vs pypy")]
            cell = f"{pyre_field:>6s}   {note:>5s}"
        self.comparisons[idx][backend] = cell

    # ── record result ──

    def _record(self, backend, passed, name, detail):
        if passed:
            if backend == "dynasm":
                self.dynasm_pass += 1
            else:
                self.cranelift_pass += 1
        else:
            self.results.append(f"{red('FAIL')} {backend} {name}  {detail}")
            if backend == "dynasm":
                self.dynasm_fail += 1
            else:
                self.cranelift_fail += 1

    # ── snapshot gate ──

    def _snapshot_path(self, backend, name, suffix):
        return Path(SNAP_DIR) / backend / f"{name}.{suffix}"

    def _apply_snapshot_gate(self, backend, name, output, elapsed):
        status, reason = "ok", ""
        out_path = self._snapshot_path(backend, name, "out")
        time_path = self._snapshot_path(backend, name, "time")

        if self.args.snapshot_mode == "record":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output, encoding="utf-8")
            time_path.write_text(f"{elapsed:.2f}", encoding="utf-8")

        if self.args.snapshot_mode == "diff":
            if not out_path.exists():
                self.snapshot_missing.append(f"{backend}/{name}")
            else:
                saved_out = out_path.read_text(encoding="utf-8")
                if output != saved_out:
                    self.snapshot_diffs.append(f"{backend}/{name}")
                    return "fail", "snapshot output diff"

        if (
            self.args.threshold is not None
            and elapsed is not None
            and elapsed != "-"
            and time_path.exists()
        ):
            saved_time_str = time_path.read_text(encoding="utf-8").strip()
            if saved_time_str and saved_time_str != "-":
                saved_time = float(saved_time_str)
                limit = saved_time * (1 + self.args.threshold / 100.0)
                if float(elapsed) > limit:
                    return "fail", f"threshold {elapsed:.2f}s > baseline {saved_time_str}s +{self.args.threshold}%"

        return status, reason

    # ── build ──

    def build_backend(self, backend):
        cfg = CARGO_CONFIG[backend]
        print(f"Building {cfg['bin']} (release, backend={backend})...")
        cmd = [
            "cargo", "build", "--release", "-p", "pyrex",
            "--bin", cfg["bin"], *cfg["extra"],
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        lines = (proc.stdout or "").strip().splitlines() + (proc.stderr or "").strip().splitlines()
        if lines:
            print(lines[-1])
        if proc.returncode != 0:
            print(f"ERROR: cargo build failed (exit {proc.returncode})")
            if proc.stderr:
                print(proc.stderr[-500:])
            sys.exit(1)

    # ── warmup ──

    def warmup(self, script):
        sys.stdout.write(f"  {'warmup':<10s}")
        sys.stdout.flush()
        for runner in [PYTHON3, PYPY3]:
            try:
                subprocess.run(
                    [runner, script],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            except Exception:
                pass
        for backend in ("dynasm", "cranelift"):
            if self.enabled(backend):
                try:
                    subprocess.run(
                        [self._pyre(backend), script],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=30,
                    )
                except Exception:
                    pass
        print(dim("done"))

    # ── single-backend bench run ──

    def _run_backend_bench(
        self, backend, name, script, timeout,
        vs_cpython, vs_pypy, t_cpython, t_pypy, pypy_output,
    ):
        pyre_bin = self._pyre(backend)
        effective_timeout = scaled_timeout(timeout, self._timeout_scale(backend))

        sys.stdout.write(f"    {backend:<10s}")
        sys.stdout.flush()

        output, elapsed, code = run_timed(
            [pyre_bin, script], timeout_s=effective_timeout,
        )

        if code != 0:
            if code == 124:
                self._record(backend, False, name, f"timeout (>{effective_timeout}s)")
                print(f"{red('TIMEOUT')}  >{effective_timeout}s")
            else:
                self._record(backend, False, name, f"crash (exit {code})")
                print(f"{red('CRASH')} (exit {code})")
            self._append_comparison(backend, name, t_cpython, t_pypy, "FAIL")
            return

        if output != pypy_output:
            exp = pypy_output[:60]
            act = output[:60]
            self._record(backend, False, name, "wrong output")
            print(f"{red('WRONG')}  got: {act} expected(pypy): {exp}")
            self._append_comparison(backend, name, t_cpython, t_pypy, "WRONG")
            return

        ratio = "-"
        if t_pypy not in (None, "-") and float(t_pypy) > 0 and elapsed > 0:
            ratio = f"{elapsed / float(t_pypy):.1f}x"

        if vs_cpython and t_cpython not in (None, "-"):
            if elapsed > float(t_cpython) * vs_cpython + BENCH_COMPARE_BUFFER_S:
                self._record(
                    backend, False, name,
                    f"{elapsed:.2f}s > cpython {t_cpython}s x{vs_cpython}",
                )
                print(f"{red('SLOWER')}  pyre {elapsed:.2f}s > cpython {t_cpython}s x{vs_cpython}")
                self._append_comparison(
                    backend, name, t_cpython, t_pypy,
                    fmt_time(f"{elapsed:.2f}"), f"({ratio} vs pypy)",
                )
                return

        if vs_pypy and t_pypy not in (None, "-"):
            if elapsed > float(t_pypy) * vs_pypy + BENCH_COMPARE_BUFFER_S:
                self._record(
                    backend, False, name,
                    f"{elapsed:.2f}s > pypy {t_pypy}s x{vs_pypy}",
                )
                print(f"{red('SLOWER')}  pyre {elapsed:.2f}s > pypy {t_pypy}s x{vs_pypy}")
                self._append_comparison(
                    backend, name, t_cpython, t_pypy,
                    fmt_time(f"{elapsed:.2f}"), f"({ratio} vs pypy)",
                )
                return

        snap_status, snap_reason = self._apply_snapshot_gate(
            backend, name, output, elapsed,
        )
        if snap_status == "fail":
            self._record(backend, False, name, snap_reason)
            print(f"{red('SNAPDIFF')}  {snap_reason}")
            self._append_comparison(backend, name, t_cpython, t_pypy, "SNAPDIFF")
            return

        self._record(backend, True, name, f"{elapsed:.2f}s")
        print(f"{green('PASS')}  {elapsed:.2f}s")
        self._append_comparison(
            backend, name, t_cpython, t_pypy,
            fmt_time(f"{elapsed:.2f}"), f"({ratio} vs pypy)",
        )

    # ── top-level bench entry ──

    def run_bench(
        self, name, script, timeout,
        dynasm_vs_cpython=None, dynasm_vs_pypy=None,
        cranelift_vs_cpython=None, cranelift_vs_pypy=None,
        skip_backends=(),
    ):
        need_cpython = False
        if (
            self.enabled("dynasm")
            and "dynasm" not in skip_backends
            and dynasm_vs_cpython
        ):
            need_cpython = True
        if (
            self.enabled("cranelift")
            and "cranelift" not in skip_backends
            and cranelift_vs_cpython
        ):
            need_cpython = True

        print(f"  {name}")

        t_cpython = "-"
        cpython_code = 0
        if need_cpython:
            sys.stdout.write(f"    {'cpython':<10s}")
            sys.stdout.flush()
            cpython_output, t_cpu, cpython_code = run_timed([PYTHON3, script])
            t_cpython = f"{t_cpu:.2f}"
            if cpython_code != 0:
                print(f"{red('CRASH')} (exit {cpython_code})")
            else:
                print(f"{dim('done')}  {t_cpython}s")

        sys.stdout.write(f"    {'pypy':<10s}")
        sys.stdout.flush()
        pypy_output, pypy_cpu, pypy_code = run_timed([PYPY3, script])
        t_pypy = f"{pypy_cpu:.2f}" if pypy_code == 0 else "-"
        if pypy_code != 0:
            print(f"{red('CRASH')} (exit {pypy_code})")
            for backend in ("dynasm", "cranelift"):
                if self.enabled(backend):
                    self._record(backend, False, name, "pypy crash")
                    self._append_comparison(backend, name, fmt_time(t_cpython), "-", "FAIL")
            return
        print(f"{dim('done')}  {t_pypy}s")

        for backend, vs_cpython, vs_pypy in [
            ("dynasm", dynasm_vs_cpython, dynasm_vs_pypy),
            ("cranelift", cranelift_vs_cpython, cranelift_vs_pypy),
        ]:
            if not self.enabled(backend):
                continue
            if backend in skip_backends:
                sys.stdout.write(f"    {backend:<10s}")
                print(dim("skip"))
                self._append_comparison(
                    backend, name, fmt_time(t_cpython), fmt_time(t_pypy), "skip",
                )
                continue
            if vs_cpython and cpython_code != 0:
                sys.stdout.write(f"    {backend:<10s}")
                print(f"{red('FAIL')}  missing cpython baseline")
                self._record(backend, False, name, "cpython crash")
                self._append_comparison(backend, name, "-", fmt_time(t_pypy), "FAIL")
                continue
            self._run_backend_bench(
                backend, name, script, timeout,
                vs_cpython, vs_pypy, t_cpython, t_pypy, pypy_output,
            )

    # ── printing ──

    def print_backend_config(self):
        parts = []
        for b in ("dynasm", "cranelift"):
            if self.enabled(b):
                parts.append(f"{b}={self._pyre(b)}(x{self._timeout_scale(b)})")
        if parts:
            print(f"backend: {' '.join(parts)}")

    def print_comparison_table(self):
        if not self.comparisons:
            return
        both = self.enabled("dynasm") and self.enabled("cranelift")
        dynasm_only = self.enabled("dynasm") and not self.enabled("cranelift")
        cranelift_only = self.enabled("cranelift") and not self.enabled("dynasm")

        print(bold("Comparison"))

        if both:
            print(f"  {'benchmark':<15s} {'cpython':>8s} {'pypy':>8s} {'dynasm':>18s} {'cranelift':>18s}")
            print("  " + "─" * 78)
            for c in self.comparisons:
                print(
                    f"  {c['name']:<15s} {c['cpython']:>8s} {c['pypy']:>8s}"
                    f" {c['dynasm']:>18s} {c['cranelift']:>18s}"
                )
        elif dynasm_only:
            print(f"  {'benchmark':<15s} {'cpython':>8s} {'pypy':>8s} {'dynasm':>18s}")
            print("  " + "─" * 56)
            for c in self.comparisons:
                print(
                    f"  {c['name']:<15s} {c['cpython']:>8s} {c['pypy']:>8s}"
                    f" {c['dynasm']:>18s}"
                )
        elif cranelift_only:
            print(f"  {'benchmark':<15s} {'cpython':>8s} {'pypy':>8s} {'cranelift':>18s}")
            print("  " + "─" * 56)
            for c in self.comparisons:
                print(
                    f"  {c['name']:<15s} {c['cpython']:>8s} {c['pypy']:>8s}"
                    f" {c['cranelift']:>18s}"
                )

    def print_summary(self):
        print()
        if self.results:
            print("─" * 33)
            for r in self.results:
                print(f"  {r}")
            print("─" * 33)

        failed_runs = 0
        enabled_runs = 0
        for b in ("dynasm", "cranelift"):
            if not self.enabled(b):
                continue
            enabled_runs += 1
            fail = self.dynasm_fail if b == "dynasm" else self.cranelift_fail
            if fail > 0:
                failed_runs += 1

        self.print_comparison_table()
        print()

        if self.args.snapshot_mode or self.args.threshold is not None:
            if self.args.snapshot_mode == "record":
                print(dim(f"snapshot recorded under {SNAP_DIR}/"))
            elif self.args.snapshot_mode == "diff":
                if self.snapshot_diffs:
                    print(
                        f"{red('snapshot diff')}: {len(self.snapshot_diffs)} bench(es)"
                        f" — {' '.join(self.snapshot_diffs)}"
                    )
                if self.snapshot_missing:
                    print(
                        f"{dim('snapshot missing')}: {len(self.snapshot_missing)} bench(es)"
                        f" — {' '.join(self.snapshot_missing)}"
                    )
                if not self.snapshot_diffs and not self.snapshot_missing:
                    print(dim("snapshot diff: clean"))
            if self.args.threshold is not None:
                print(dim(f"threshold: ±{self.args.threshold}% vs baseline"))
            print()

        for b in ("dynasm", "cranelift"):
            if not self.enabled(b):
                continue
            p = self.dynasm_pass if b == "dynasm" else self.cranelift_pass
            f = self.dynasm_fail if b == "dynasm" else self.cranelift_fail
            if f > 0:
                print(f"{red('FAILED')}: {b} {f} failed, {p} passed")
            else:
                print(f"{green('ALL PASSED')}: {b} {p}/{p}")

        if failed_runs > 0:
            print(f"{red('FAILED')}: {failed_runs} backend run(s) failed")
        else:
            print(f"{green('ALL PASSED')}: {enabled_runs}/{enabled_runs} backend run(s)")

        return 1 if failed_runs > 0 else 0


# ── Argument parsing ─────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="pyre pre-merge check: correctness + regression guard + comparison",
    )
    parser.add_argument("--backend", choices=["dynasm", "cranelift"], default="")
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    parser.add_argument("--dynasm-timeout-scale", type=float, default=None)
    parser.add_argument("--cranelift-timeout-scale", type=float, default=None)
    parser.add_argument("--snapshot", dest="snapshot_mode", action="store_const", const="record")
    parser.add_argument("--snapshot-diff", dest="snapshot_mode", action="store_const", const="diff")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("pyre_path", nargs="?", default="")
    args = parser.parse_args()

    if args.pyre_path and not args.backend:
        parser.error("[path/to/pyre] requires --backend when running a single binary")

    if args.snapshot_mode == "record" and args.threshold is not None:
        print("NOTE: --threshold ignored in --snapshot record mode")
        args.threshold = None

    return args


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    chk = Check(args)

    backends = [args.backend] if args.backend else ["dynasm", "cranelift"]

    for backend in backends:
        chk.build_backend(backend)
        pyre_bin = args.pyre_path if args.pyre_path else default_binary(backend)
        if not Path(pyre_bin).exists():
            alt = pyre_bin + EXE
            if Path(alt).exists():
                pyre_bin = alt
        if not os.access(pyre_bin, os.X_OK) and not Path(pyre_bin).exists():
            print(f"ERROR: build failed for backend '{backend}' (missing executable: {pyre_bin})")
            sys.exit(1)
        chk._set_pyre(backend, pyre_bin)

    print()
    print(bold("pyre pre-merge check"))
    chk.print_backend_config()
    print()
    chk.warmup(f"{BENCH_DIR}/int_loop.py")
    print()

    B = BENCH_DIR

    #             name              script                          timeout  d_vs_cp  d_vs_py  c_vs_cp  c_vs_py  skip
    chk.run_bench("int_loop",       f"{B}/int_loop.py",             5,       None,    1.5,     None,    1.5)
    chk.run_bench("float_loop",     f"{B}/float_loop.py",           5,       None,    1.0,     None,    2.5)
    chk.run_bench("fib_loop",       f"{B}/fib_loop.py",             5,       None,    1.5,     1.0,     None)
    chk.run_bench("inline_helper",  f"{B}/inline_helper.py",        5,       None,    1.0,     None,    1.0)
    chk.run_bench("fib_recursive",  f"{B}/fib_recursive.py",        5,       1.5,     None,    1,       8)
    chk.run_bench("nested_loop",    f"{B}/nested_loop.py",          5,       None,    2,       None,    2)
    chk.run_bench("raise_catch",    f"{B}/raise_catch_loop.py",     6,       None,    None,    None,    None)
    chk.run_bench("spectral_norm",  f"{B}/spectral_norm.py",       10,       10,      None,    10,      None)
    chk.run_bench("nbody",          f"{B}/nbody_50k.py",            5,       15,      None,    15,      None)
    chk.run_bench("fannkuch",       f"{B}/fannkuch.py",             5,       None,    None,    None,    None)
    chk.run_bench("list_reverse",   f"{B}/list_reverse.py",         5,       10,      None,    10,      None)
    chk.run_bench("list_pop_append",f"{B}/list_pop_append.py",      5,       15,      None,    15,      None)
    chk.run_bench("list_insert",    f"{B}/list_insert.py",          5,       None,    1.5,     None,    1.5)
    chk.run_bench("list_setslice",  f"{B}/list_setslice.py",        5,       7,       None,    7,       None)

    rc = chk.print_summary()
    sys.exit(rc)


if __name__ == "__main__":
    main()
