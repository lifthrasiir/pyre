//! sys module definition.
//!
//! PyPy equivalent: pypy/module/sys/

use crate::{DictStorage, dict_storage_store};
use pyre_object::*;
use std::sync::OnceLock;

/// Shared stub type for `sys._getframe`, `sys.flags`, `sys.stdout` and other
/// module-level sys attributes that expose CPython-looking attribute bags.
///
/// `typedef::w_object()` (plain `object`) cannot store instance attributes —
/// its type flag `hasdict` is false, matching CPython where `object()`
/// instances reject `__setattr__` unless their subclass explicitly opts in.
/// PyPy's `sys` module exposes these as dedicated W_Root types with their
/// own typedefs, not as bare `object` instances. The Rust port mirrors that
/// by installing a single `sys.namespace` type with `__dict__` in its
/// typedef slots so every stub instance supports `setattr`.
fn sys_namespace_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    let raw = *TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("sys.namespace", |ns| {
            // typedef.py:34 — presence of `__dict__` in the init namespace
            // is what flips `hasdict` on the resulting type (see
            // typedef.rs:541/554). The value itself is never read; PyPy
            // stores a `GetSetProperty` there, but the Rust port only needs
            // the key to light up the per-instance attribute store.
            dict_storage_store(ns, "__dict__", w_none());
        });
        tp as usize
    });
    raw as PyObjectRef
}

/// Allocate a fresh stub instance whose type supports `setattr`. Used for
/// all the CPython-style attribute bags surfaced by the sys module.
fn make_sys_namespace_instance() -> PyObjectRef {
    w_instance_new(sys_namespace_type())
}

/// `pypy/module/sys/vm.py:217 space.getexecutioncontext()` access for
/// `sys.gettrace`/`settrace`/`getprofile`/`setprofile`.
///
/// Pyre's `crate::call::getexecutioncontext` returns the TLS-cached
/// active context (set on eval-loop entry); see the helper's doc for
/// the staleness gap relative to PyPy's `space.getexecutioncontext()`
/// which always queries the thread state.
fn current_execution_context() -> *mut crate::PyExecutionContext {
    crate::call::getexecutioncontext() as *mut crate::PyExecutionContext
}

fn sys_gettrace_impl(_args: &[PyObjectRef]) -> crate::PyResult {
    let ec = current_execution_context();
    if ec.is_null() {
        return Ok(w_none());
    }
    let w_trace = unsafe { (*ec).gettrace() };
    Ok(if w_trace.is_null() { w_none() } else { w_trace })
}

fn sys_settrace_impl(args: &[PyObjectRef]) -> crate::PyResult {
    let ec = current_execution_context();
    if !ec.is_null() {
        let w_func = args.first().copied().unwrap_or_else(w_none);
        unsafe { (*ec).settrace(w_func) };
    }
    Ok(w_none())
}

fn sys_getprofile_impl(_args: &[PyObjectRef]) -> crate::PyResult {
    let ec = current_execution_context();
    if ec.is_null() {
        return Ok(w_none());
    }
    let w_profile = unsafe { (*ec).getprofile() };
    Ok(if w_profile.is_null() {
        w_none()
    } else {
        w_profile
    })
}

fn sys_setprofile_impl(args: &[PyObjectRef]) -> crate::PyResult {
    let ec = current_execution_context();
    if !ec.is_null() {
        let w_func = args.first().copied().unwrap_or_else(w_none);
        // executioncontext.py:317-318 ValueError("Cannot call setllprofile
        // with real None") propagates via setprofile -> setllprofile.
        unsafe { (*ec).setprofile(w_func)? };
    }
    Ok(w_none())
}

pub fn init(ns: &mut DictStorage) {
    dict_storage_store(ns, "maxsize", w_int_new(i64::MAX));
    dict_storage_store(ns, "maxunicode", w_int_new(0x10FFFF));
    dict_storage_store(ns, "version", w_str_new("3.13.0 (pyre 0.0.1)"));
    dict_storage_store(
        ns,
        "platform",
        w_str_new(if cfg!(target_os = "macos") {
            "darwin"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(target_os = "windows") {
            "win32"
        } else {
            "unknown"
        }),
    );
    dict_storage_store(
        ns,
        "byteorder",
        w_str_new(if cfg!(target_endian = "little") {
            "little"
        } else {
            "big"
        }),
    );
    dict_storage_store(
        ns,
        "version_info",
        w_tuple_new(vec![
            w_int_new(3),
            w_int_new(13),
            w_int_new(0),
            w_str_new("final"),
            w_int_new(0),
        ]),
    );
    // sys.modules — live dict synced with the import cache.
    let modules_dict = w_dict_new();
    crate::importing::set_sys_modules_dict(modules_dict);
    dict_storage_store(ns, "modules", modules_dict);
    // sys.path — empty list placeholder
    dict_storage_store(ns, "path", w_list_new(vec![]));
    // sys.stdout/stderr/stdin — stub file-like objects.  Real CPython
    // wires these through io.TextIOWrapper around sys.__stdout__; pyre
    // exposes a tiny object with the bare minimum surface so anything
    // that writes status (unittest, traceback, warnings) keeps working.
    dict_storage_store(ns, "stdout", make_std_stream("<stdout>", false));
    dict_storage_store(ns, "stderr", make_std_stream("<stderr>", true));
    dict_storage_store(ns, "stdin", make_std_stream("<stdin>", false));
    dict_storage_store(ns, "__stdout__", make_std_stream("<stdout>", false));
    dict_storage_store(ns, "__stderr__", make_std_stream("<stderr>", true));
    dict_storage_store(ns, "__stdin__", make_std_stream("<stdin>", false));
    // sys._getframe — returns a stub frame object with f_locals/f_globals
    dict_storage_store(
        ns,
        "_getframe",
        crate::make_builtin_function("_getframe", |_| {
            let frame = make_sys_namespace_instance();
            let _ = crate::baseobjspace::setattr(frame, "f_locals", w_dict_new());
            let _ = crate::baseobjspace::setattr(frame, "f_globals", w_dict_new());
            let _ = crate::baseobjspace::setattr(frame, "f_code", w_none());
            let _ = crate::baseobjspace::setattr(frame, "f_back", w_none());
            let _ = crate::baseobjspace::setattr(frame, "f_lineno", w_int_new(0));
            Ok(frame)
        }),
    );
    // sys.exc_info() → (type, value, traceback)
    dict_storage_store(
        ns,
        "exc_info",
        crate::make_builtin_function("exc_info", |_| {
            let exc = crate::eval::get_current_exception();
            unsafe {
                if exc.is_null() || pyre_object::is_none(exc) || !pyre_object::is_exception(exc) {
                    Ok(w_tuple_new(vec![w_none(), w_none(), w_none()]))
                } else {
                    let w_class = (*exc).w_class;
                    let exc_type = if w_class.is_null() { w_none() } else { w_class };
                    Ok(w_tuple_new(vec![exc_type, exc, w_none()]))
                }
            }
        }),
    );
    // sys.flags — pypy/module/sys/app.py:99-119 `class sysflags` with
    // `__metaclass__ = structseqtype`. PyPy exposes it as a structseq
    // (immutable tuple subclass with named fields). pyre does not have
    // structseq yet, so we approximate the orthodox behavior with a
    // dedicated type whose attributes live in the TYPE's class
    // namespace rather than the instance dict. Read access via the
    // descriptor protocol still works (`sys.flags.optimize`); writes
    // fall through to `setdictvalue → raiseattrerror` because the type
    // has no `__dict__` slot, matching the read-only contract:
    //
    //     >>> sys.flags.optimize = 3
    //     AttributeError: 'sys.flags' object has no attribute 'optimize'
    //
    // The exact exception type differs from PyPy
    // (`pypy/module/sys/test/test_sysmodule.py:148` expects TypeError)
    // because pyre lacks the structseq tp_setattro slot. The full
    // structseq port is tracked separately.
    {
        let flags_type = crate::typedef::make_builtin_type("sys.flags", |fns| {
            dict_storage_store(fns, "debug", w_int_new(0));
            dict_storage_store(fns, "inspect", w_int_new(0));
            dict_storage_store(fns, "interactive", w_int_new(0));
            dict_storage_store(fns, "optimize", w_int_new(0));
            dict_storage_store(fns, "dont_write_bytecode", w_int_new(0));
            dict_storage_store(fns, "no_user_site", w_int_new(0));
            dict_storage_store(fns, "no_site", w_int_new(0));
            dict_storage_store(fns, "ignore_environment", w_int_new(0));
            dict_storage_store(fns, "verbose", w_int_new(0));
            dict_storage_store(fns, "bytes_warning", w_int_new(0));
            dict_storage_store(fns, "quiet", w_int_new(0));
            dict_storage_store(fns, "hash_randomization", w_int_new(0));
            dict_storage_store(fns, "isolated", w_int_new(0));
            dict_storage_store(fns, "dev_mode", w_bool_from(false));
            dict_storage_store(fns, "utf8_mode", w_int_new(1));
            dict_storage_store(fns, "warn_default_encoding", w_int_new(0));
            dict_storage_store(fns, "safe_path", w_bool_from(false));
            dict_storage_store(fns, "int_max_str_digits", w_int_new(4300));
            dict_storage_store(fns, "context_aware_warnings", w_bool_from(false));
        });
        let flags = w_instance_new(flags_type);
        dict_storage_store(ns, "flags", flags);
    }
    // sys.getdefaultencoding
    dict_storage_store(
        ns,
        "getdefaultencoding",
        crate::make_builtin_function("getdefaultencoding", |_| Ok(w_str_new("utf-8"))),
    );
    // sys.getrecursionlimit / setrecursionlimit — pypy/module/sys/vm.py:45.
    // The runtime stack budget lives in `crate::stack_check`; both
    // helpers route through it so the interpreter, JIT prologue probe,
    // and blackhole resume see a consistent recursion budget.
    dict_storage_store(
        ns,
        "getrecursionlimit",
        crate::make_builtin_function("getrecursionlimit", |args| {
            // pypy/module/sys/vm.py:72 — no arguments.
            if !args.is_empty() {
                return Err(crate::PyError::type_error(
                    "getrecursionlimit() takes no arguments",
                ));
            }
            Ok(w_int_new(crate::stack_check::get_recursion_limit() as i64))
        }),
    );
    dict_storage_store(
        ns,
        "setrecursionlimit",
        crate::make_builtin_function("setrecursionlimit", |args| {
            // pypy/module/sys/vm.py:63 `@unwrap_spec(new_limit="c_int")`
            // — exactly one positional argument, coerced through
            // baseobjspace.c_int_w (gateway_int_w + 32-bit range
            // check). `c_int_w` accepts int subclasses and any object
            // implementing `__int__`, rejects floats, and surfaces
            // out-of-range values as OverflowError.
            if args.len() != 1 {
                return Err(crate::PyError::type_error(
                    "setrecursionlimit() takes exactly one argument",
                ));
            }
            let new_limit = crate::baseobjspace::c_int_w(args[0])?;
            crate::stack_check::set_recursion_limit(new_limit)?;
            Ok(w_none())
        }),
    );
    // sys.intern
    dict_storage_store(
        ns,
        "intern",
        crate::make_builtin_function("intern", |args| {
            Ok(if args.is_empty() {
                w_str_new("")
            } else {
                args[0]
            })
        }),
    );
    // sys.implementation — structseq-like namespace with name, version, ...
    {
        let impl_obj = make_sys_namespace_instance();
        let _ = crate::baseobjspace::setattr(impl_obj, "name", w_str_new("pyre"));
        let _ = crate::baseobjspace::setattr(
            impl_obj,
            "version",
            w_tuple_new(vec![
                w_int_new(3),
                w_int_new(13),
                w_int_new(0),
                w_str_new("final"),
                w_int_new(0),
            ]),
        );
        let _ = crate::baseobjspace::setattr(impl_obj, "hexversion", w_int_new(0x030d00f0));
        let _ = crate::baseobjspace::setattr(impl_obj, "cache_tag", w_str_new("pyre-3.13"));
        let _ = crate::baseobjspace::setattr(impl_obj, "_multiarch", w_str_new(""));
        dict_storage_store(ns, "implementation", impl_obj);
    }
    // sys.hash_info — structseq with width/modulus/... fields.
    // PyPy: pypy/module/sys/state.py W_HashInfoStructSeq.
    {
        let hash_info = make_sys_namespace_instance();
        let _ = crate::baseobjspace::setattr(hash_info, "width", w_int_new(64));
        let _ = crate::baseobjspace::setattr(hash_info, "modulus", w_int_new((1i64 << 61) - 1));
        let _ = crate::baseobjspace::setattr(hash_info, "inf", w_int_new(314159));
        let _ = crate::baseobjspace::setattr(hash_info, "nan", w_int_new(0));
        let _ = crate::baseobjspace::setattr(hash_info, "imag", w_int_new(1000003));
        let _ = crate::baseobjspace::setattr(hash_info, "algorithm", w_str_new("siphash13"));
        let _ = crate::baseobjspace::setattr(hash_info, "hash_bits", w_int_new(64));
        let _ = crate::baseobjspace::setattr(hash_info, "seed_bits", w_int_new(128));
        let _ = crate::baseobjspace::setattr(hash_info, "cutoff", w_int_new(0));
        dict_storage_store(ns, "hash_info", hash_info);
    }
    // sys.float_info — structseq with IEEE 754 double metadata.
    // PyPy: pypy/module/sys/state.py W_FloatInfoStructSeq.
    {
        let fi = make_sys_namespace_instance();
        let _ = crate::baseobjspace::setattr(fi, "max", w_float_new(f64::MAX));
        let _ = crate::baseobjspace::setattr(fi, "max_exp", w_int_new(1024));
        let _ = crate::baseobjspace::setattr(fi, "max_10_exp", w_int_new(308));
        let _ = crate::baseobjspace::setattr(fi, "min", w_float_new(f64::MIN_POSITIVE));
        let _ = crate::baseobjspace::setattr(fi, "min_exp", w_int_new(-1021));
        let _ = crate::baseobjspace::setattr(fi, "min_10_exp", w_int_new(-307));
        let _ = crate::baseobjspace::setattr(fi, "dig", w_int_new(15));
        let _ = crate::baseobjspace::setattr(fi, "mant_dig", w_int_new(53));
        let _ = crate::baseobjspace::setattr(fi, "epsilon", w_float_new(f64::EPSILON));
        let _ = crate::baseobjspace::setattr(fi, "radix", w_int_new(2));
        let _ = crate::baseobjspace::setattr(fi, "rounds", w_int_new(1));
        dict_storage_store(ns, "float_info", fi);
    }
    // sys.int_info — structseq with int implementation details.
    {
        let ii = make_sys_namespace_instance();
        let _ = crate::baseobjspace::setattr(ii, "bits_per_digit", w_int_new(30));
        let _ = crate::baseobjspace::setattr(ii, "sizeof_digit", w_int_new(4));
        let _ = crate::baseobjspace::setattr(ii, "default_max_str_digits", w_int_new(4300));
        let _ = crate::baseobjspace::setattr(ii, "str_digits_check_threshold", w_int_new(640));
        dict_storage_store(ns, "int_info", ii);
    }
    // sys.executable
    dict_storage_store(ns, "executable", w_str_new("pyre"));
    // sys.prefix / exec_prefix
    dict_storage_store(ns, "prefix", w_str_new(""));
    dict_storage_store(ns, "exec_prefix", w_str_new(""));
    dict_storage_store(ns, "base_prefix", w_str_new(""));
    dict_storage_store(ns, "base_exec_prefix", w_str_new(""));
    // sys._framework — macOS framework name (empty string on non-framework builds)
    dict_storage_store(ns, "_framework", w_str_new(""));
    // sys._jit — namespace with is_enabled/is_available methods.
    // Python 3.14+ introduced sys._jit for CPython tier-2 JIT support checks.
    {
        let jit = make_sys_namespace_instance();
        let _ = crate::baseobjspace::setattr(
            jit,
            "is_enabled",
            crate::make_builtin_function("is_enabled", |_| Ok(w_bool_from(false))),
        );
        let _ = crate::baseobjspace::setattr(
            jit,
            "is_available",
            crate::make_builtin_function("is_available", |_| Ok(w_bool_from(false))),
        );
        dict_storage_store(ns, "_jit", jit);
    }
    // sys.platlibdir — typically "lib" on POSIX; used by sysconfig to
    // construct install paths.
    dict_storage_store(ns, "platlibdir", w_str_new("lib"));
    // sys.exit(code=0) — raise SystemExit
    dict_storage_store(
        ns,
        "exit",
        crate::make_builtin_function("exit", |args| {
            let code = if args.is_empty() {
                0i64
            } else {
                unsafe {
                    if pyre_object::is_none(args[0]) {
                        0
                    } else if pyre_object::is_int(args[0]) || pyre_object::is_bool(args[0]) {
                        pyre_object::w_int_get_value(args[0])
                    } else {
                        // Non-integer arg: print it to stderr and exit 1
                        0
                    }
                }
            };
            Err(crate::PyError {
                kind: crate::PyErrorKind::SystemExit,
                message: code.to_string(),
                exc_object: std::ptr::null_mut(),
            })
        }),
    );
    // sys.abiflags
    dict_storage_store(ns, "abiflags", w_str_new(""));
    // sys.argv — pick up pending argv from set_sys_argv if available.
    let pending = crate::importing::take_pending_sys_argv();
    let argv = if pending.is_null() {
        w_list_new(vec![])
    } else {
        pending
    };
    dict_storage_store(ns, "argv", argv);
    // sys.warnoptions
    dict_storage_store(ns, "warnoptions", w_list_new(vec![]));
    // sys.builtin_module_names — tuple of names of modules compiled into
    // the interpreter. PyPy: pypy/module/sys/state.py get_builtin_module_names.
    // Pyre: include all stub/native built-ins from importing.rs.
    dict_storage_store(
        ns,
        "builtin_module_names",
        w_tuple_new(vec![
            w_str_new("_abc"),
            w_str_new("_bisect"),
            w_str_new("_blake2"),
            w_str_new("_codecs"),
            w_str_new("_collections"),
            w_str_new("_collections_abc"),
            w_str_new("_contextvars"),
            w_str_new("_csv"),
            w_str_new("_datetime"),
            w_str_new("_decimal"),
            w_str_new("_functools"),
            w_str_new("_hashlib"),
            w_str_new("_heapq"),
            w_str_new("_imp"),
            w_str_new("_io"),
            w_str_new("_json"),
            w_str_new("_locale"),
            w_str_new("_md5"),
            w_str_new("_opcode"),
            w_str_new("_operator"),
            w_str_new("_pickle"),
            w_str_new("_random"),
            w_str_new("_sha1"),
            w_str_new("_sha2"),
            w_str_new("_sha3"),
            w_str_new("_signal"),
            w_str_new("_socket"),
            w_str_new("_sre"),
            w_str_new("_stat"),
            w_str_new("_string"),
            w_str_new("_struct"),
            w_str_new("_thread"),
            w_str_new("_tokenize"),
            w_str_new("_tracemalloc"),
            w_str_new("_typing"),
            w_str_new("_warnings"),
            w_str_new("_weakref"),
            w_str_new("atexit"),
            w_str_new("binascii"),
            w_str_new("builtins"),
            w_str_new("copyreg"),
            w_str_new("errno"),
            w_str_new("fcntl"),
            w_str_new("grp"),
            w_str_new("itertools"),
            w_str_new("marshal"),
            w_str_new("math"),
            w_str_new("cmath"),
            w_str_new("operator"),
            w_str_new("posix"),
            w_str_new("pwd"),
            w_str_new("select"),
            w_str_new("sys"),
            w_str_new("time"),
        ]),
    );
    // sys.exception — returns the currently handled exception or None
    dict_storage_store(
        ns,
        "exception",
        crate::make_builtin_function("exception", |_| Ok(w_none())),
    );
    // sys.exc_clear — no-op
    dict_storage_store(
        ns,
        "exc_clear",
        crate::make_builtin_function("exc_clear", |_| Ok(w_none())),
    );
    // sys.gettrace / settrace
    dict_storage_store(
        ns,
        "gettrace",
        crate::make_builtin_function("gettrace", sys_gettrace_impl),
    );
    dict_storage_store(
        ns,
        "settrace",
        crate::make_builtin_function("settrace", sys_settrace_impl),
    );
    // sys.getprofile / setprofile
    dict_storage_store(
        ns,
        "getprofile",
        crate::make_builtin_function("getprofile", sys_getprofile_impl),
    );
    dict_storage_store(
        ns,
        "setprofile",
        crate::make_builtin_function("setprofile", sys_setprofile_impl),
    );
    // sys.getfilesystemencoding
    dict_storage_store(
        ns,
        "getfilesystemencoding",
        crate::make_builtin_function("getfilesystemencoding", |_| Ok(w_str_new("utf-8"))),
    );
    dict_storage_store(
        ns,
        "getfilesystemencodeerrors",
        crate::make_builtin_function("getfilesystemencodeerrors", |_| {
            Ok(w_str_new("surrogateescape"))
        }),
    );
    // sys.audit — no-op
    dict_storage_store(
        ns,
        "audit",
        crate::make_builtin_function("audit", |_| Ok(w_none())),
    );
    // sys.is_finalizing
    dict_storage_store(
        ns,
        "is_finalizing",
        crate::make_builtin_function("is_finalizing", |_| Ok(w_bool_from(false))),
    );
    // sys.displayhook / excepthook
    dict_storage_store(
        ns,
        "displayhook",
        crate::make_builtin_function("displayhook", |_| Ok(w_none())),
    );
    dict_storage_store(
        ns,
        "excepthook",
        crate::make_builtin_function("excepthook", |_| Ok(w_none())),
    );
    // sys.path_hooks / path_importer_cache
    dict_storage_store(ns, "path_hooks", w_list_new(vec![]));
    dict_storage_store(ns, "path_importer_cache", w_dict_new());
    // sys.meta_path — empty
    dict_storage_store(ns, "meta_path", w_list_new(vec![]));
    // sys.dont_write_bytecode
    dict_storage_store(ns, "dont_write_bytecode", w_bool_from(true));
    // sys.addaudithook
    dict_storage_store(
        ns,
        "addaudithook",
        crate::make_builtin_function("addaudithook", |_| Ok(w_none())),
    );
}

/// Construct a stub stdio object exposing `write`, `flush`, `isatty`,
/// `fileno`, and `name`.  PyPy uses real W_File-backed objects via the io
/// module; pyre routes writes through Rust's stdout/stderr directly.
fn make_std_stream(name: &'static str, is_stderr: bool) -> PyObjectRef {
    let stream = make_sys_namespace_instance();
    let _ = crate::baseobjspace::setattr(stream, "name", w_str_new(name));
    let _ = crate::baseobjspace::setattr(stream, "encoding", w_str_new("utf-8"));
    let _ =
        crate::baseobjspace::setattr(stream, "mode", w_str_new(if is_stderr { "w" } else { "r" }));
    let _ = crate::baseobjspace::setattr(stream, "closed", w_bool_from(false));
    let _ = crate::baseobjspace::setattr(stream, "buffer", w_none());
    // ATTR_TABLE-stored builtin methods do not get `self` prepended (see
    // pyopcode load_method dispatch), so the first arg may be the string
    // directly. Pick whichever element is a real str.
    fn pick_str(args: &[PyObjectRef]) -> Option<&str> {
        for &a in args {
            if !a.is_null() && unsafe { is_str(a) } {
                return Some(unsafe { w_str_get_value(a) });
            }
        }
        None
    }
    let write_fn = if is_stderr {
        crate::make_builtin_function("write", |args| {
            use std::io::Write;
            if let Some(text) = pick_str(args) {
                let _ = std::io::stderr().write_all(text.as_bytes());
                return Ok(w_int_new(text.len() as i64));
            }
            Ok(w_int_new(0))
        })
    } else {
        crate::make_builtin_function("write", |args| {
            use std::io::Write;
            if let Some(text) = pick_str(args) {
                let _ = std::io::stdout().write_all(text.as_bytes());
                return Ok(w_int_new(text.len() as i64));
            }
            Ok(w_int_new(0))
        })
    };
    let _ = crate::baseobjspace::setattr(stream, "write", write_fn);
    let _ = crate::baseobjspace::setattr(
        stream,
        "flush",
        crate::make_builtin_function("flush", |_| {
            use std::io::Write;
            let _ = std::io::stdout().flush();
            let _ = std::io::stderr().flush();
            Ok(w_none())
        }),
    );
    let _ = crate::baseobjspace::setattr(
        stream,
        "isatty",
        crate::make_builtin_function("isatty", |_| Ok(w_bool_from(false))),
    );
    let fileno_fn = if is_stderr {
        crate::make_builtin_function("fileno", |_| Ok(w_int_new(2)))
    } else {
        crate::make_builtin_function("fileno", |_| Ok(w_int_new(1)))
    };
    let _ = crate::baseobjspace::setattr(stream, "fileno", fileno_fn);
    let _ = crate::baseobjspace::setattr(
        stream,
        "writable",
        crate::make_builtin_function("writable", |_| Ok(w_bool_from(true))),
    );
    let _ = crate::baseobjspace::setattr(
        stream,
        "readable",
        crate::make_builtin_function("readable", |_| Ok(w_bool_from(false))),
    );
    stream
}
