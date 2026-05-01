//! Port of `rpython/config/support.py`.
//!
//! Upstream is 50 LOC of two small helpers
//! (`detect_number_of_processors`, `detect_pax`) imported by
//! `translationoption.py` to populate the `make_jobs` option default
//! and consumed by the C backend's PaX-aware compile flags.

use std::env;
use std::fs;
use std::process::Command;

/// Upstream `support.py:7-29 detect_number_of_processors`. Returns the
/// number of processors to use as the `make -j` argument.
///
/// Body shape mirrors upstream `:7-29` line-by-line:
///
/// ```python
/// def detect_number_of_processors(filename_or_file='/proc/cpuinfo'):
///     if os.environ.get('MAKEFLAGS'):
///         return 1
///     if sys.platform == 'darwin':
///         return sysctl_get_cpu_count('/usr/sbin/sysctl')
///     elif sys.platform.startswith('freebsd'):
///         return sysctl_get_cpu_count('/sbin/sysctl')
///     elif not sys.platform.startswith('linux'):
///         return 1
///     try:
///         ...
///         count = max([... for line in f if line.startswith('processor')]) + 1
///         if count >= 4:
///             return max(count // 2, 3)
///         else:
///             return count
///     except:
///         return 1
/// ```
///
/// Linux branch parses `/proc/cpuinfo` line-by-line for `processor:`
/// entries — same source upstream reads. macOS / FreeBSD shell out to
/// `sysctl -n hw.ncpu`. The bare `except` at `:28-29` becomes a
/// catch-all returning `1`.
pub fn detect_number_of_processors() -> i64 {
    detect_number_of_processors_with_path("/proc/cpuinfo")
}

/// Test-friendly variant of [`detect_number_of_processors`]. Mirrors
/// upstream's `filename_or_file='/proc/cpuinfo'` parameter so a
/// fixture file can be supplied.
pub fn detect_number_of_processors_with_path(filename: &str) -> i64 {
    // Upstream `:8-9`: `if os.environ.get('MAKEFLAGS'): return 1`.
    // Python's truthiness on `dict.get(...)` treats both "absent" and
    // "empty string" as falsy. `env::var_os(...).is_some()` would treat
    // an empty `MAKEFLAGS` env var as truthy, which diverges. Match
    // upstream by checking for a non-empty value.
    if matches!(env::var("MAKEFLAGS"), Ok(value) if !value.is_empty()) {
        return 1;
    }
    // Upstream `:10-11`: `sys.platform == 'darwin' →
    // sysctl_get_cpu_count('/usr/sbin/sysctl')` — raw hw.ncpu, no cap.
    if cfg!(target_os = "macos") {
        return sysctl_get_cpu_count("/usr/sbin/sysctl");
    }
    // Upstream `:12-13`: `sys.platform.startswith('freebsd') →
    // sysctl_get_cpu_count('/sbin/sysctl')` — raw hw.ncpu, no cap.
    if cfg!(target_os = "freebsd") {
        return sysctl_get_cpu_count("/sbin/sysctl");
    }
    // Upstream `:14-15`: `not sys.platform.startswith('linux') →
    // return 1   # implement me`.
    if !cfg!(target_os = "linux") {
        return 1;
    }
    // Upstream `:16-29`: Linux /proc/cpuinfo path with half-load cap.
    // Upstream `:28-29 except: return 1`. Parse failure → fallback 1.
    let count = match read_cpu_count_from_cpuinfo(filename) {
        Some(c) => c,
        None => return 1,
    };
    if count >= 4 {
        (count / 2).max(3)
    } else {
        count
    }
}

/// Upstream `support.py:31-37 sysctl_get_cpu_count(cmd, name='hw.ncpu')`:
///
/// ```python
/// def sysctl_get_cpu_count(cmd, name='hw.ncpu'):
///     try:
///         proc = subprocess.Popen([cmd, '-n', name], stdout=subprocess.PIPE)
///         count = proc.communicate()[0]
///         return int(count)
///     except (OSError, ValueError):
///         return 1
/// ```
fn sysctl_get_cpu_count(cmd: &str) -> i64 {
    let output = match Command::new(cmd).args(["-n", "hw.ncpu"]).output() {
        Ok(o) => o,
        Err(_) => return 1, // Upstream `:36` OSError.
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    match stdout.trim().parse::<i64>() {
        Ok(n) => n,
        Err(_) => 1, // Upstream `:36` ValueError.
    }
}

/// Linux-only CPU count. Upstream `support.py:16-29` parses
/// `/proc/cpuinfo`:
///
/// ```python
/// f = open('/proc/cpuinfo', 'r')
/// count = max([int(re.split('processor[^\d]*(\d+)', line)[1])
///             for line in f.readlines()
///             if line.startswith('processor')]) + 1
/// ```
///
/// Returns `None` on any failure (matches upstream's bare `except:` at
/// `:28-29` returning `1`).
fn read_cpu_count_from_cpuinfo(filename: &str) -> Option<i64> {
    let contents = fs::read_to_string(filename).ok()?;
    let mut max_id: Option<i64> = None;
    for line in contents.lines() {
        // Upstream `:23 if line.startswith('processor')`.
        if !line.starts_with("processor") {
            continue;
        }
        // Upstream `:21 re.split('processor[^\d]*(\d+)', line)[1]`.
        // Splitting on a regex that captures a number group returns
        // [prefix, number, rest]. Index [1] is the captured digits.
        // Walk past `processor`, skip non-digits, parse the next run
        // of digits.
        let after = &line["processor".len()..];
        let digits: String = after
            .chars()
            .skip_while(|c| !c.is_ascii_digit())
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if digits.is_empty() {
            // Upstream `re.split` on a line with no digits raises
            // IndexError on `[1]`, hitting the bare `except:`.
            return None;
        }
        let id: i64 = digits.parse().ok()?;
        max_id = Some(max_id.map_or(id, |m| m.max(id)));
    }
    // Upstream `max([...]) + 1`; an empty list raises ValueError →
    // bare `except:` → return 1. Surface as `None` here.
    max_id.map(|m| m + 1)
}

/// Upstream `support.py:39-50 detect_pax`. Returns `True` when the
/// running Linux kernel has PaX protection enabled. `support.py:46`
/// reads `/proc/self/status` and checks the substring `"PaX"`. On
/// non-Linux platforms upstream short-circuits to `False`.
///
/// Upstream's `with open(...) as fd: data = fd.read()` propagates
/// `IOError` on failure. The Rust port mirrors that by returning
/// `Result<bool, io::Error>` so the C-backend's PaX-aware compile
/// rules cannot silently drop the protection check when
/// `/proc/self/status` is unreadable.
pub fn detect_pax() -> std::io::Result<bool> {
    detect_pax_with_path("/proc/self/status")
}

/// Test-friendly variant matching upstream's open-and-substring
/// pattern but accepting an explicit path so fixtures can drive it.
pub fn detect_pax_with_path(path: &str) -> std::io::Result<bool> {
    if !cfg!(target_os = "linux") {
        return Ok(false);
    }
    let data = fs::read_to_string(path)?;
    Ok(data.contains("PaX"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    };

    /// Serialise every test that touches `MAKEFLAGS` so the
    /// remove-var/read-var window is not raced by another test
    /// flipping the same global. cargo's default test runner is multi-
    /// threaded; without this lock the macOS / Linux sub-tests below
    /// observe `MAKEFLAGS=-j4` set by the dedicated MAKEFLAGS test.
    static MAKEFLAGS_LOCK: Mutex<()> = Mutex::new(());
    static TEMP_FIXTURE_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_fixture_path(name: &str) -> std::path::PathBuf {
        let id = TEMP_FIXTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("majit-{name}-{}-{id}", std::process::id()))
    }

    #[test]
    fn detect_number_of_processors_returns_positive() {
        // Regardless of MAKEFLAGS state, the result must be >= 1.
        let n = detect_number_of_processors();
        assert!(n >= 1, "expected >=1, got {}", n);
    }

    #[test]
    fn detect_number_of_processors_respects_makeflags() {
        // SAFETY of set_env: serialised against other env-touching
        // tests via [`MAKEFLAGS_LOCK`] so concurrent tests in this
        // module never observe a partially-mutated env.
        //
        // Upstream `support.py:8-9` short-circuits on any truthy
        // MAKEFLAGS.
        let _guard = MAKEFLAGS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = env::var_os("MAKEFLAGS");
        unsafe {
            env::set_var("MAKEFLAGS", "-j4");
        }
        let n = detect_number_of_processors();
        unsafe {
            match prev {
                Some(v) => env::set_var("MAKEFLAGS", v),
                None => env::remove_var("MAKEFLAGS"),
            }
        }
        assert_eq!(n, 1, "MAKEFLAGS set must short-circuit to 1");
    }

    /// Upstream `support.py:10-13` — macOS / FreeBSD return the raw
    /// `hw.ncpu` count, NOT the half-load-capped value. Pin the parity
    /// behaviour with a platform-gated test that checks `count == raw`
    /// when `count` is in the no-cap range (>= 1) on macOS and FreeBSD.
    #[cfg(any(target_os = "macos", target_os = "freebsd"))]
    #[test]
    fn detect_number_of_processors_macos_freebsd_uses_raw_count() {
        let _guard = MAKEFLAGS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = env::var_os("MAKEFLAGS");
        unsafe {
            env::remove_var("MAKEFLAGS");
        }
        let n = detect_number_of_processors();
        // Read the same source upstream uses so the "raw" baseline
        // matches the production read path.
        let cmd = if cfg!(target_os = "macos") {
            "/usr/sbin/sysctl"
        } else {
            "/sbin/sysctl"
        };
        let raw = sysctl_get_cpu_count(cmd);
        unsafe {
            match prev {
                Some(v) => env::set_var("MAKEFLAGS", v),
                None => env::remove_var("MAKEFLAGS"),
            }
        }
        assert_eq!(
            n, raw,
            "macOS / FreeBSD must return raw hw.ncpu without half-load cap"
        );
    }

    /// Upstream `support.py:23-27` — Linux applies the half-load cap
    /// when `count >= 4`. Pinned with the same shape as upstream's
    /// `if count >= 4: return max(count // 2, 3)`.
    #[cfg(target_os = "linux")]
    #[test]
    fn detect_number_of_processors_linux_applies_half_load_cap() {
        let _guard = MAKEFLAGS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = env::var_os("MAKEFLAGS");
        unsafe {
            env::remove_var("MAKEFLAGS");
        }
        let n = detect_number_of_processors();
        // Read the same source upstream parses so the "raw" baseline
        // matches the production read path.
        let raw = read_cpu_count_from_cpuinfo("/proc/cpuinfo").unwrap_or(1);
        unsafe {
            match prev {
                Some(v) => env::set_var("MAKEFLAGS", v),
                None => env::remove_var("MAKEFLAGS"),
            }
        }
        let expected = if raw >= 4 { (raw / 2).max(3) } else { raw };
        assert_eq!(n, expected, "Linux applies half-load cap when count >= 4");
    }

    /// Upstream `support.py:21-23` — `int(re.split('processor[^\d]*(\d+)',
    /// line)[1])` over lines starting with `'processor'`. Pin the
    /// parser against a synthetic fixture so the regex behaviour is
    /// observable independent of the host machine.
    #[test]
    fn read_cpu_count_from_cpuinfo_parses_processor_max_id_plus_one() {
        let path = temp_fixture_path("cpuinfo-fixture-basic");
        let _ = fs::write(
            &path,
            "processor\t: 0\nvendor_id\t: AuthenticAMD\nprocessor\t: 7\n",
        );
        let count = read_cpu_count_from_cpuinfo(path.to_str().unwrap());
        let _ = fs::remove_file(&path);
        // max(0, 7) + 1 = 8, mirroring upstream's `max([...]) + 1`.
        assert_eq!(count, Some(8));
    }

    /// Upstream's bare `except:` at `:28-29` collapses every parse
    /// failure to `1`. The Rust port surfaces `None` from the parser
    /// and the caller turns that into `1`.
    #[test]
    fn read_cpu_count_from_cpuinfo_returns_none_for_missing_file() {
        let path = temp_fixture_path("cpuinfo-fixture-missing");
        let _ = fs::remove_file(&path);
        assert!(read_cpu_count_from_cpuinfo(path.to_str().unwrap()).is_none());
    }

    /// Upstream `:21-23` only counts `'processor'`-prefix lines.
    /// Garbage lines must not poison the max.
    #[test]
    fn read_cpu_count_from_cpuinfo_ignores_non_processor_lines() {
        let path = temp_fixture_path("cpuinfo-fixture-mixed");
        let _ = fs::write(
            &path,
            "model name\t: Threadripper\nprocessor\t: 3\nflags\t: fpu vme de\n",
        );
        let count = read_cpu_count_from_cpuinfo(path.to_str().unwrap());
        let _ = fs::remove_file(&path);
        assert_eq!(count, Some(4));
    }

    /// Upstream's regex `'processor[^\d]*(\d+)'` requires a digit run
    /// after the `processor` literal. A line missing the digit hits
    /// `re.split(...)[1]` IndexError, which the bare `except:`
    /// collapses to `1`. The Rust parser surfaces `None`.
    #[test]
    fn read_cpu_count_from_cpuinfo_returns_none_for_processor_line_without_digits() {
        let path = temp_fixture_path("cpuinfo-fixture-no-digits");
        let _ = fs::write(&path, "processor\t: not-a-number\n");
        let count = read_cpu_count_from_cpuinfo(path.to_str().unwrap());
        let _ = fs::remove_file(&path);
        assert!(count.is_none());
    }

    /// Upstream `support.py:39-50 detect_pax`: substring match against
    /// the contents of `/proc/self/status`.
    #[test]
    fn detect_pax_returns_true_when_status_contains_pax() {
        let path = temp_fixture_path("pax-fixture-positive");
        let _ = fs::write(
            &path,
            "Name:\tcat\nState:\tR (running)\nPaX:\tPemRs\n".as_bytes(),
        );
        let result = detect_pax_with_path(path.to_str().unwrap()).expect("path readable");
        let _ = fs::remove_file(&path);
        if cfg!(target_os = "linux") {
            assert!(result, "PaX-marked status must report true on linux");
        } else {
            assert!(!result, "non-linux short-circuits to false");
        }
    }

    #[test]
    fn detect_pax_returns_false_when_status_lacks_pax() {
        let path = temp_fixture_path("pax-fixture-negative");
        let _ = fs::write(&path, "Name:\tcat\nState:\tR (running)\n".as_bytes());
        let result = detect_pax_with_path(path.to_str().unwrap()).expect("path readable");
        let _ = fs::remove_file(&path);
        assert!(!result, "absent PaX entry must surface false");
    }

    /// Upstream's `with open('/proc/self/status') as fd: data = fd.read()`
    /// propagates `IOError` on failure. The Rust port must surface the
    /// error rather than silently treat a missing file as "no PaX".
    #[test]
    fn detect_pax_propagates_missing_file_on_linux() {
        let path = temp_fixture_path("pax-fixture-absent");
        let _ = fs::remove_file(&path);
        let result = detect_pax_with_path(path.to_str().unwrap());
        if cfg!(target_os = "linux") {
            assert!(
                result.is_err(),
                "missing /proc/self/status surrogate must surface an io::Error",
            );
        } else {
            assert!(
                matches!(result, Ok(false)),
                "non-linux short-circuit must return Ok(false)",
            );
        }
    }
}
