//! Module importing — PyPy equivalent: pypy/module/imp/importing.py
//!
//! Implements the import machinery:
//! - `importhook()` — main entry point (called by IMPORT_NAME opcode)
//! - `find_module()` — locate a .py file on sys.path
//! - `load_source_module()` — compile and execute a .py file
//! - `check_sys_modules()` — consult the module cache
//! - `import_all_from()` — IMPORT_STAR handler

use std::cell::RefCell;
use std::collections::HashMap;
#[cfg(feature = "host_env")]
use std::path::{Path, PathBuf};

use crate::{CodeObject, Mode, compile_source_with_filename};
use crate::{DictStorage, PyExecutionContext, dict_storage_load, dict_storage_store};
use pyre_object::*;

use crate::pyframe::PyFrame;

// ── sys.modules cache ────────────────────────────────────────────────
// PyPy equivalent: space.sys.get('modules') — a dict mapping module names
// to module objects. We use a thread-local HashMap<String, PyObjectRef>.

thread_local! {
    static SYS_MODULES: RefCell<HashMap<String, PyObjectRef>> = RefCell::new(HashMap::new());
    /// The Python-visible `sys.modules` dict. Kept in sync with SYS_MODULES
    /// so that `sys.modules['name']` lookups work from Python code.
    static SYS_MODULES_DICT: std::cell::Cell<PyObjectRef> = const { std::cell::Cell::new(pyre_object::PY_NULL) };
    /// sys.path equivalent — list of directories to search for modules.
    #[cfg(feature = "host_env")]
    static SYS_PATH: RefCell<Vec<PathBuf>> = RefCell::new(Vec::new());
    /// Builtin modules registry — PyPy equivalent: space.builtin_modules
    ///
    /// Maps module name → initializer function that populates a DictStorage.
    /// Each builtin module is lazily created on first import.
    static BUILTIN_MODULES: RefCell<HashMap<&'static str, fn(&mut DictStorage)>> =
        RefCell::new(HashMap::new());
}

// ── builtin module registry ──────────────────────────────────────────
// PyPy equivalent: space.builtin_modules dict + MixedModule.interpleveldefs

/// Register a builtin module initializer.
///
/// PyPy equivalent: Module.install() → space.builtin_modules[name] = mod
pub fn register_builtin_module(name: &'static str, init: fn(&mut DictStorage)) {
    BUILTIN_MODULES.with(|m| {
        m.borrow_mut().insert(name, init);
    });
}

/// Install all standard builtin modules.
///
/// PyPy equivalent: baseobjspace.py `make_builtins()` +
/// `install_mixedmodule()` for each module in objspace.usemodules.
pub fn install_builtin_modules() {
    register_builtin_module("math", crate::module::math::moduledef::init);
    register_builtin_module("cmath", crate::module::math::cmath_moduledef::init);
    register_builtin_module("time", crate::module::time::moduledef::init);
    register_builtin_module("sys", crate::module::sys::moduledef::init);
    register_builtin_module("operator", crate::module::operator::moduledef::init);
    register_builtin_module("_operator", crate::module::operator::moduledef::init);
    register_builtin_module("builtins", crate::module::__builtin__::moduledef::init);
    register_builtin_module("_io", crate::module::_io::moduledef::init);
    register_builtin_module("_sre", crate::module::_sre::moduledef::init);

    // Minimal C-extension stubs required for stdlib import chains.
    // PyPy: these are all implemented as mixed modules under pypy/module/.
    register_builtin_module("_weakref", crate::module::_weakref::moduledef::init);
    register_builtin_module("_abc", init_abc);
    register_builtin_module("_functools", init_functools);
    register_builtin_module("_thread", init_thread);
    register_builtin_module("itertools", init_itertools);
    register_builtin_module("_contextvars", init_contextvars);
    register_builtin_module("copyreg", init_copyreg);
    register_builtin_module("_codecs", init_codecs);
    register_builtin_module("posix", init_posix);
    register_builtin_module("errno", init_errno);
    register_builtin_module("_collections", init_collections_c);
    register_builtin_module("_ast", init_ast);
    register_builtin_module("_opcode", init_opcode_c);
    register_builtin_module("_imp", init_imp);
    register_builtin_module("importlib.machinery", init_importlib_machinery);
    register_builtin_module("importlib", init_importlib_pkg);
    register_builtin_module("importlib.util", init_importlib_util);
    register_builtin_module("importlib.abc", init_importlib_abc);
    register_builtin_module("_signal", init_signal_stub);
    register_builtin_module("atexit", init_atexit);
    register_builtin_module("pwd", init_pwd);
    register_builtin_module("_locale", init_locale);
    register_builtin_module("_random", init_random);
    register_builtin_module("_struct", init_struct);
    register_builtin_module("gc", init_gc);
    register_builtin_module("unicodedata", init_unicodedata);
    // `_sysconfigdata_{abiflags}_{platform}_{multiarch}` is a generated
    // Python module containing `build_time_vars = {...}` that sysconfig
    // imports from `_init_posix`. We stub it out with an empty dict so
    // `sysconfig.get_config_vars()` returns an empty mapping.
    // PyPy equivalent: pypy/tool/build_cffi_imports.py creates the same
    // file during translation.
    register_builtin_module("_sysconfigdata__darwin_", init_sysconfigdata_empty);
    register_builtin_module("_sysconfigdata__linux_", init_sysconfigdata_empty);
    register_builtin_module(
        "_sysconfigdata__linux_x86_64-linux-gnu",
        init_sysconfigdata_empty,
    );
    register_builtin_module(
        "_sysconfigdata__linux_aarch64-linux-gnu",
        init_sysconfigdata_empty,
    );
    // _opcode_metadata.py exists in the stdlib; load the real file instead.
    for name in &[
        "_string",
        "_warnings",
        "_heapq",
        "_tokenize",
        "_typing",
        "_bisect",
        "binascii",
        "_hashlib",
        "_sha2",
        "_md5",
        "_sha1",
        "_sha3",
        "_blake2",
        "_decimal",
        "_pickle",
        "_datetime",
        "_json",
        "_csv",
        "marshal",
        "fcntl",
        "grp",
        "select",
        "_socket",
        "_tracemalloc",
        "_stat",
        "_asyncio",
        "_queue",
        "_zoneinfo",
        "array",
        "zlib",
    ] {
        register_builtin_module(name, empty_module_init);
    }
}

/// Empty module initializer for C-extension stubs.
fn empty_module_init(_ns: &mut DictStorage) {}

/// gc module stub — enough to let `import gc` succeed.
fn init_gc(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "collect",
        crate::make_builtin_function_with_arity("collect", |_| Ok(pyre_object::w_int_new(0)), 1),
    );
    crate::dict_storage_store(
        ns,
        "disable",
        crate::make_builtin_function_with_arity("disable", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "enable",
        crate::make_builtin_function_with_arity("enable", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "isenabled",
        crate::make_builtin_function_with_arity(
            "isenabled",
            |_| Ok(pyre_object::w_bool_from(false)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_objects",
        crate::make_builtin_function_with_arity(
            "get_objects",
            |_| Ok(pyre_object::w_list_new(vec![])),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_referrers",
        crate::make_builtin_function("get_referrers", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
    crate::dict_storage_store(
        ns,
        "get_referents",
        crate::make_builtin_function("get_referents", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
    crate::dict_storage_store(
        ns,
        "set_threshold",
        crate::make_builtin_function_with_arity("set_threshold", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "get_threshold",
        crate::make_builtin_function_with_arity(
            "get_threshold",
            |_| {
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_int_new(700),
                    pyre_object::w_int_new(10),
                    pyre_object::w_int_new(10),
                ]))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_count",
        crate::make_builtin_function_with_arity(
            "get_count",
            |_| {
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_int_new(0),
                    pyre_object::w_int_new(0),
                    pyre_object::w_int_new(0),
                ]))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "is_tracked",
        crate::make_builtin_function_with_arity(
            "is_tracked",
            |_| Ok(pyre_object::w_bool_from(false)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "is_finalized",
        crate::make_builtin_function_with_arity(
            "is_finalized",
            |_| Ok(pyre_object::w_bool_from(false)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "freeze",
        crate::make_builtin_function_with_arity("freeze", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(ns, "callbacks", pyre_object::w_list_new(vec![]));
    crate::dict_storage_store(ns, "garbage", pyre_object::w_list_new(vec![]));
    crate::dict_storage_store(ns, "DEBUG_STATS", pyre_object::w_int_new(1));
    crate::dict_storage_store(ns, "DEBUG_COLLECTABLE", pyre_object::w_int_new(2));
    crate::dict_storage_store(ns, "DEBUG_UNCOLLECTABLE", pyre_object::w_int_new(4));
    crate::dict_storage_store(ns, "DEBUG_SAVEALL", pyre_object::w_int_new(32));
    crate::dict_storage_store(ns, "DEBUG_LEAK", pyre_object::w_int_new(38));
}

/// unicodedata module stub — provides normalize() and category().
fn init_unicodedata(ns: &mut DictStorage) {
    // unicodedata.normalize(form, unistr) → unistr (stub: returns input unchanged)
    crate::dict_storage_store(
        ns,
        "normalize",
        crate::make_builtin_function_with_arity(
            "normalize",
            |args| {
                if args.len() >= 2 {
                    Ok(args[1])
                } else {
                    Ok(pyre_object::w_str_new(""))
                }
            },
            2,
        ),
    );
    // unicodedata.category(chr) → str (stub: returns "Cn" = unassigned)
    crate::dict_storage_store(
        ns,
        "category",
        crate::make_builtin_function_with_arity(
            "category",
            |_| Ok(pyre_object::w_str_new("Cn")),
            1,
        ),
    );
    // unicodedata.name(chr, default=None) → str
    crate::dict_storage_store(
        ns,
        "name",
        crate::make_builtin_function("name", |args| {
            if args.len() >= 2 {
                Ok(args[1])
            } else {
                Err(crate::PyError::value_error("no such name"))
            }
        }),
    );
    // unicodedata.lookup(name) → chr
    crate::dict_storage_store(
        ns,
        "lookup",
        crate::make_builtin_function_with_arity(
            "lookup",
            |_| Err(crate::PyError::key_error("character not found")),
            1,
        ),
    );
    // unicodedata.decimal(chr, default=None) → int
    crate::dict_storage_store(
        ns,
        "decimal",
        crate::make_builtin_function("decimal", |args| {
            if args.len() >= 2 {
                Ok(args[1])
            } else {
                Err(crate::PyError::value_error("not a decimal"))
            }
        }),
    );
    // unicodedata.numeric(chr, default=None) → float
    crate::dict_storage_store(
        ns,
        "numeric",
        crate::make_builtin_function("numeric", |args| {
            if args.len() >= 2 {
                Ok(args[1])
            } else {
                Err(crate::PyError::value_error("not a numeric character"))
            }
        }),
    );
    // unicodedata.unidata_version
    crate::dict_storage_store(ns, "unidata_version", pyre_object::w_str_new("15.1.0"));
    // unicodedata.ucd_3_2_0 — alias for the module itself (used by IDNA)
    // We store a sentinel; os_helper only checks that the module imported.
}

/// `_struct` C-extension stub — PyPy: pypy/module/struct/interp_struct.py.
///
/// Implements just enough to let `struct.py` load: `pack`, `unpack`,
/// `calcsize`, `_clearcache`, and the `error` type. Each packer handles
/// the format codes pyre actually uses during import (`<q`, `<d`, etc.).
fn init_struct(ns: &mut DictStorage) {
    fn parse_format(fmt: &str) -> (char, Vec<char>) {
        // Returns (byte_order, codes).
        let chars = fmt.chars();
        let first = chars.clone().next().unwrap_or('@');
        let (endian, rest) = if matches!(first, '<' | '>' | '!' | '=' | '@') {
            (first, chars.skip(1).collect::<String>())
        } else {
            ('@', fmt.to_string())
        };
        (
            endian,
            rest.chars().filter(|c| !c.is_ascii_whitespace()).collect(),
        )
    }
    fn code_size(c: char) -> usize {
        match c {
            'b' | 'B' | 'c' | '?' | 'x' => 1,
            'h' | 'H' => 2,
            'i' | 'I' | 'l' | 'L' | 'f' => 4,
            'q' | 'Q' | 'd' | 'n' | 'N' => 8,
            'e' => 2,
            _ => 0,
        }
    }
    crate::dict_storage_store(
        ns,
        "_clearcache",
        crate::make_builtin_function_with_arity("_clearcache", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(ns, "error", crate::typedef::w_object());
    crate::dict_storage_store(
        ns,
        "calcsize",
        crate::make_builtin_function_with_arity(
            "calcsize",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_int_new(0));
                }
                let fmt = unsafe {
                    if pyre_object::is_str(args[0]) {
                        pyre_object::w_str_get_value(args[0]).to_string()
                    } else if pyre_object::bytesobject::is_bytes_like(args[0]) {
                        let data = pyre_object::bytesobject::bytes_like_data(args[0]);
                        String::from_utf8_lossy(data).into_owned()
                    } else {
                        return Err(crate::PyError::type_error("calcsize: format must be str"));
                    }
                };
                let (_, codes) = parse_format(&fmt);
                let total: usize = codes.iter().copied().map(code_size).sum();
                Ok(pyre_object::w_int_new(total as i64))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "pack",
        crate::make_builtin_function("pack", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_bytes_from_bytes(&[]));
            }
            let fmt = unsafe {
                if pyre_object::is_str(args[0]) {
                    pyre_object::w_str_get_value(args[0]).to_string()
                } else {
                    return Err(crate::PyError::type_error("pack: format must be str"));
                }
            };
            let (endian, codes) = parse_format(&fmt);
            let little = matches!(endian, '<' | '=' | '@');
            let mut out = Vec::new();
            for (i, code) in codes.iter().enumerate() {
                let arg = args.get(i + 1).copied().unwrap_or(pyre_object::w_none());
                match *code {
                    'b' | 'B' => {
                        let v = unsafe { pyre_object::w_int_get_value(arg) } as i8;
                        out.push(v as u8);
                    }
                    'h' | 'H' => {
                        let v = unsafe { pyre_object::w_int_get_value(arg) } as i16;
                        let bytes = if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        };
                        out.extend_from_slice(&bytes);
                    }
                    'i' | 'I' | 'l' | 'L' => {
                        let v = unsafe { pyre_object::w_int_get_value(arg) } as i32;
                        let bytes = if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        };
                        out.extend_from_slice(&bytes);
                    }
                    'q' | 'Q' | 'n' | 'N' => {
                        let v = unsafe { pyre_object::w_int_get_value(arg) };
                        let bytes = if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        };
                        out.extend_from_slice(&bytes);
                    }
                    'f' => {
                        let v = unsafe {
                            if pyre_object::is_float(arg) {
                                pyre_object::w_float_get_value(arg) as f32
                            } else {
                                pyre_object::w_int_get_value(arg) as f32
                            }
                        };
                        let bytes = if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        };
                        out.extend_from_slice(&bytes);
                    }
                    'd' => {
                        let v = unsafe {
                            if pyre_object::is_float(arg) {
                                pyre_object::w_float_get_value(arg)
                            } else {
                                pyre_object::w_int_get_value(arg) as f64
                            }
                        };
                        let bytes = if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        };
                        out.extend_from_slice(&bytes);
                    }
                    _ => {}
                }
            }
            Ok(pyre_object::w_bytes_from_bytes(&out))
        }),
    );
    crate::dict_storage_store(
        ns,
        "unpack",
        crate::make_builtin_function_with_arity(
            "unpack",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("unpack requires (fmt, buffer)"));
                }
                let fmt = unsafe { pyre_object::w_str_get_value(args[0]).to_string() };
                let buf = unsafe {
                    if pyre_object::bytesobject::is_bytes_like(args[1]) {
                        pyre_object::bytesobject::bytes_like_data(args[1]).to_vec()
                    } else {
                        return Err(crate::PyError::type_error(
                            "unpack: buffer must be bytes-like",
                        ));
                    }
                };
                let (endian, codes) = parse_format(&fmt);
                let little = matches!(endian, '<' | '=' | '@');
                let mut out = Vec::new();
                let mut pos = 0usize;
                for code in codes {
                    match code {
                        'b' | 'B' => {
                            if pos >= buf.len() {
                                break;
                            }
                            out.push(pyre_object::w_int_new(buf[pos] as i8 as i64));
                            pos += 1;
                        }
                        'h' | 'H' => {
                            if pos + 2 > buf.len() {
                                break;
                            }
                            let chunk = [buf[pos], buf[pos + 1]];
                            let v = if little {
                                i16::from_le_bytes(chunk)
                            } else {
                                i16::from_be_bytes(chunk)
                            };
                            out.push(pyre_object::w_int_new(v as i64));
                            pos += 2;
                        }
                        'i' | 'I' | 'l' | 'L' => {
                            if pos + 4 > buf.len() {
                                break;
                            }
                            let chunk = [buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]];
                            let v = if little {
                                i32::from_le_bytes(chunk)
                            } else {
                                i32::from_be_bytes(chunk)
                            };
                            out.push(pyre_object::w_int_new(v as i64));
                            pos += 4;
                        }
                        'q' | 'Q' | 'n' | 'N' => {
                            if pos + 8 > buf.len() {
                                break;
                            }
                            let chunk: [u8; 8] = buf[pos..pos + 8].try_into().unwrap();
                            let v = if little {
                                i64::from_le_bytes(chunk)
                            } else {
                                i64::from_be_bytes(chunk)
                            };
                            out.push(pyre_object::w_int_new(v));
                            pos += 8;
                        }
                        'f' => {
                            if pos + 4 > buf.len() {
                                break;
                            }
                            let chunk = [buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]];
                            let v = if little {
                                f32::from_le_bytes(chunk)
                            } else {
                                f32::from_be_bytes(chunk)
                            };
                            out.push(pyre_object::w_float_new(v as f64));
                            pos += 4;
                        }
                        'd' => {
                            if pos + 8 > buf.len() {
                                break;
                            }
                            let chunk: [u8; 8] = buf[pos..pos + 8].try_into().unwrap();
                            let v = if little {
                                f64::from_le_bytes(chunk)
                            } else {
                                f64::from_be_bytes(chunk)
                            };
                            out.push(pyre_object::w_float_new(v));
                            pos += 8;
                        }
                        _ => {}
                    }
                }
                Ok(pyre_object::w_tuple_new(out))
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "unpack_from",
        crate::make_builtin_function("unpack_from", |_| Ok(pyre_object::w_tuple_new(vec![]))),
    );
    crate::dict_storage_store(
        ns,
        "iter_unpack",
        crate::make_builtin_function_with_arity(
            "iter_unpack",
            |_| Ok(pyre_object::w_list_new(vec![])),
            2,
        ),
    );
    // Struct class — minimal constructor returning instance with format
    // attribute. Used by struct.Struct(fmt).pack/unpack.
    crate::dict_storage_store(
        ns,
        "Struct",
        crate::make_builtin_function_with_arity(
            "Struct",
            |args| {
                let fmt = args.first().copied().unwrap_or(pyre_object::w_str_new(""));
                let obj = pyre_object::w_instance_new(crate::typedef::w_object());
                let _ = crate::baseobjspace::setattr(obj, "format", fmt);
                Ok(obj)
            },
            1,
        ),
    );
}

/// `_random` C-extension stub — PyPy: pypy/module/_random/interp_random.py.
///
/// Provides a minimal `Random` class that wraps a very small linear
/// congruential generator. Good enough for `random.py` to construct a
/// `random._inst` at module import time; real tests can then use the
/// Python `random.Random` subclass as a drop-in.
fn init_random(ns: &mut DictStorage) {
    fn random_type() -> PyObjectRef {
        thread_local! {
            static RANDOM_TYPE: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
        }
        RANDOM_TYPE.with(|c| {
            *c.get_or_init(|| {
                let tp = crate::typedef::make_builtin_type("_random.Random", |ns| {
                    // random_method_* are defined in importing.rs; routing
                    // through make_builtin_function binds them as unbound
                    // methods so `rand.random()` calls pass `self` as args[0].
                    crate::dict_storage_store(
                        ns,
                        "__init__",
                        crate::make_builtin_function("__init__", |args| {
                            let seed = if args.len() >= 2 {
                                unsafe {
                                    if pyre_object::is_int(args[1]) {
                                        pyre_object::w_int_get_value(args[1]) as u64
                                    } else {
                                        0x1234_5678
                                    }
                                }
                            } else {
                                0x1234_5678
                            };
                            let _ = crate::baseobjspace::setattr(
                                args[0],
                                "__rand_state__",
                                pyre_object::w_int_new(seed as i64),
                            );
                            Ok(pyre_object::w_none())
                        }),
                    );
                    crate::dict_storage_store(
                        ns,
                        "seed",
                        crate::make_builtin_function("seed", |args| {
                            let seed = if args.len() >= 2 {
                                unsafe {
                                    if pyre_object::is_int(args[1]) {
                                        pyre_object::w_int_get_value(args[1]) as u64
                                    } else {
                                        0x1234_5678
                                    }
                                }
                            } else {
                                0x1234_5678
                            };
                            let _ = crate::baseobjspace::setattr(
                                args[0],
                                "__rand_state__",
                                pyre_object::w_int_new(seed as i64),
                            );
                            Ok(pyre_object::w_none())
                        }),
                    );
                    crate::dict_storage_store(
                        ns,
                        "random",
                        crate::make_builtin_function_with_arity(
                            "random",
                            |args| {
                                // Tiny xorshift PRNG — ok for import-time construction.
                                let self_obj = args[0];
                                let state =
                                    crate::baseobjspace::getattr(self_obj, "__rand_state__")
                                        .ok()
                                        .map(|v| unsafe { pyre_object::w_int_get_value(v) as u64 })
                                        .unwrap_or(0x1234_5678);
                                let mut x = state;
                                x ^= x << 13;
                                x ^= x >> 7;
                                x ^= x << 17;
                                let _ = crate::baseobjspace::setattr(
                                    self_obj,
                                    "__rand_state__",
                                    pyre_object::w_int_new(x as i64),
                                );
                                Ok(pyre_object::w_float_new((x as f64) / (u64::MAX as f64)))
                            },
                            1,
                        ),
                    );
                    crate::dict_storage_store(
                        ns,
                        "getrandbits",
                        crate::make_builtin_function("getrandbits", |args| {
                            let k = if args.len() >= 2 {
                                unsafe { pyre_object::w_int_get_value(args[1]) as u32 }
                            } else {
                                32
                            };
                            let state = crate::baseobjspace::getattr(args[0], "__rand_state__")
                                .ok()
                                .map(|v| unsafe { pyre_object::w_int_get_value(v) as u64 })
                                .unwrap_or(0x1234_5678);
                            let mut x = state;
                            x ^= x << 13;
                            x ^= x >> 7;
                            x ^= x << 17;
                            let _ = crate::baseobjspace::setattr(
                                args[0],
                                "__rand_state__",
                                pyre_object::w_int_new(x as i64),
                            );
                            let mask = if k >= 64 { u64::MAX } else { (1u64 << k) - 1 };
                            Ok(pyre_object::w_int_new((x & mask) as i64))
                        }),
                    );
                    crate::dict_storage_store(
                        ns,
                        "getstate",
                        crate::make_builtin_function_with_arity(
                            "getstate",
                            |args| {
                                let state = crate::baseobjspace::getattr(args[0], "__rand_state__")
                                    .unwrap_or_else(|_| pyre_object::w_int_new(0));
                                Ok(pyre_object::w_tuple_new(vec![state]))
                            },
                            1,
                        ),
                    );
                    crate::dict_storage_store(
                        ns,
                        "setstate",
                        crate::make_builtin_function_with_arity(
                            "setstate",
                            |args| {
                                if args.len() >= 2 {
                                    unsafe {
                                        if pyre_object::is_tuple(args[1])
                                            && pyre_object::w_tuple_len(args[1]) >= 1
                                        {
                                            if let Some(state) =
                                                pyre_object::w_tuple_getitem(args[1], 0)
                                            {
                                                let _ = crate::baseobjspace::setattr(
                                                    args[0],
                                                    "__rand_state__",
                                                    state,
                                                );
                                            }
                                        }
                                    }
                                }
                                Ok(pyre_object::w_none())
                            },
                            2,
                        ),
                    );
                });
                unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
                tp
            })
        })
    }
    crate::dict_storage_store(ns, "Random", random_type());
}

/// `_locale` C-extension stub — PyPy: pypy/module/_locale/.
///
/// Provides the 'C' locale defaults so locale.py's `from _locale import *`
/// succeeds and Lib/locale.py exposes working `localeconv`/`setlocale`.
/// This mirrors the `except ImportError` fallback in the stdlib's
/// `locale` module, but routed through pyre's builtin-module registry
/// so a single import succeeds.
fn init_locale(ns: &mut DictStorage) {
    // Locale category constants — match POSIX values.
    crate::dict_storage_store(ns, "LC_CTYPE", pyre_object::w_int_new(0));
    crate::dict_storage_store(ns, "LC_NUMERIC", pyre_object::w_int_new(1));
    crate::dict_storage_store(ns, "LC_TIME", pyre_object::w_int_new(2));
    crate::dict_storage_store(ns, "LC_COLLATE", pyre_object::w_int_new(3));
    crate::dict_storage_store(ns, "LC_MONETARY", pyre_object::w_int_new(4));
    crate::dict_storage_store(ns, "LC_MESSAGES", pyre_object::w_int_new(5));
    crate::dict_storage_store(ns, "LC_ALL", pyre_object::w_int_new(6));
    crate::dict_storage_store(ns, "CHAR_MAX", pyre_object::w_int_new(127));
    // Error alias — locale.py does `Error = ValueError` when _locale is
    // missing; here we expose a real placeholder that is a str so that
    // `except _locale.Error` still compiles (match falls through).
    crate::dict_storage_store(ns, "Error", pyre_object::w_str_new("Error"));

    // localeconv() — returns the 'C' locale parameters as a dict.
    crate::dict_storage_store(
        ns,
        "localeconv",
        crate::make_builtin_function_with_arity(
            "localeconv",
            |_| {
                let d = pyre_object::w_dict_new();
                unsafe {
                    pyre_object::w_dict_setitem_str(
                        d,
                        "grouping",
                        pyre_object::w_list_new(vec![pyre_object::w_int_new(127)]),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "currency_symbol",
                        pyre_object::w_str_new(""),
                    );
                    pyre_object::w_dict_setitem_str(d, "n_sign_posn", pyre_object::w_int_new(127));
                    pyre_object::w_dict_setitem_str(
                        d,
                        "p_cs_precedes",
                        pyre_object::w_int_new(127),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "n_cs_precedes",
                        pyre_object::w_int_new(127),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "mon_grouping",
                        pyre_object::w_list_new(vec![]),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "n_sep_by_space",
                        pyre_object::w_int_new(127),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "decimal_point",
                        pyre_object::w_str_new("."),
                    );
                    pyre_object::w_dict_setitem_str(d, "negative_sign", pyre_object::w_str_new(""));
                    pyre_object::w_dict_setitem_str(d, "positive_sign", pyre_object::w_str_new(""));
                    pyre_object::w_dict_setitem_str(
                        d,
                        "p_sep_by_space",
                        pyre_object::w_int_new(127),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "int_curr_symbol",
                        pyre_object::w_str_new(""),
                    );
                    pyre_object::w_dict_setitem_str(d, "p_sign_posn", pyre_object::w_int_new(127));
                    pyre_object::w_dict_setitem_str(d, "thousands_sep", pyre_object::w_str_new(""));
                    pyre_object::w_dict_setitem_str(
                        d,
                        "mon_thousands_sep",
                        pyre_object::w_str_new(""),
                    );
                    pyre_object::w_dict_setitem_str(d, "frac_digits", pyre_object::w_int_new(127));
                    pyre_object::w_dict_setitem_str(
                        d,
                        "mon_decimal_point",
                        pyre_object::w_str_new(""),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "int_frac_digits",
                        pyre_object::w_int_new(127),
                    );
                }
                Ok(d)
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "setlocale",
        crate::make_builtin_function("setlocale", |_| Ok(pyre_object::w_str_new("C"))),
    );
    crate::dict_storage_store(
        ns,
        "nl_langinfo",
        crate::make_builtin_function_with_arity(
            "nl_langinfo",
            |_| Ok(pyre_object::w_str_new("")),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "strcoll",
        crate::make_builtin_function_with_arity(
            "strcoll",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_int_new(0));
                }
                unsafe {
                    if pyre_object::is_str(args[0]) && pyre_object::is_str(args[1]) {
                        let a = pyre_object::w_str_get_value(args[0]);
                        let b = pyre_object::w_str_get_value(args[1]);
                        return Ok(pyre_object::w_int_new(a.cmp(b) as i64));
                    }
                }
                Ok(pyre_object::w_int_new(0))
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "strxfrm",
        crate::make_builtin_function_with_arity(
            "strxfrm",
            |args| Ok(args.first().copied().unwrap_or(pyre_object::w_str_new(""))),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getencoding",
        crate::make_builtin_function_with_arity(
            "getencoding",
            |_| Ok(pyre_object::w_str_new("utf-8")),
            0,
        ),
    );
}

/// `_sysconfigdata_*` stub — sysconfig imports this generated module to
/// read the CPython build variables. We expose a minimal `build_time_vars`
/// dict that lets sysconfig initialize without crashing.
fn init_sysconfigdata_empty(ns: &mut DictStorage) {
    let vars = pyre_object::w_dict_new();
    // A few keys are load-bearing — sysconfig.get_config_vars() populates
    // them, but an import-time crash hits on 'Py_GIL_DISABLED' and
    // similar. Leave the dict empty; .get('X') returns None for unknown
    // keys which every caller already handles.
    crate::dict_storage_store(ns, "build_time_vars", vars);
}

/// Shared `posix.stat_result` builtin type — a plain instance bag with
/// hasdict so that `st_mode`, `st_ino`, etc. attributes can be set from
/// Rust when building stat results. PyPy builds a structseq subclass with
/// named fields; this is the pyre approximation.
fn stat_result_type() -> PyObjectRef {
    thread_local! {
        static STAT_RESULT_TYPE: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    }
    STAT_RESULT_TYPE.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("stat_result", |_ns| {});
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

/// pwd module — PyPy: pypy/module/pwd/interp_pwd.py.
///
/// getpwuid / getpwnam return a struct_passwd tuple with named fields via
/// libc's getpwuid(3) / getpwnam(3). The result has the same layout as
/// CPython's pwd.struct_passwd: (pw_name, pw_passwd, pw_uid, pw_gid,
/// pw_gecos, pw_dir, pw_shell).
fn init_pwd(ns: &mut DictStorage) {
    #[cfg(unix)]
    unsafe extern "C" {
        fn getpwuid(uid: u32) -> *mut Passwd;
        fn getpwnam(name: *const std::os::raw::c_char) -> *mut Passwd;
    }
    #[cfg(unix)]
    #[repr(C)]
    struct Passwd {
        pw_name: *const std::os::raw::c_char,
        pw_passwd: *const std::os::raw::c_char,
        pw_uid: u32,
        pw_gid: u32,
        pw_change: i64,
        pw_class: *const std::os::raw::c_char,
        pw_gecos: *const std::os::raw::c_char,
        pw_dir: *const std::os::raw::c_char,
        pw_shell: *const std::os::raw::c_char,
        pw_expire: i64,
    }
    #[cfg(unix)]
    unsafe fn c_str(ptr: *const std::os::raw::c_char) -> String {
        unsafe {
            if ptr.is_null() {
                return String::new();
            }
            let cstr = std::ffi::CStr::from_ptr(ptr);
            cstr.to_string_lossy().into_owned()
        }
    }
    #[cfg(unix)]
    unsafe fn make_struct_passwd(pw: *mut Passwd) -> pyre_object::PyObjectRef {
        unsafe {
            let pw = &*pw;
            pyre_object::w_tuple_new(vec![
                pyre_object::w_str_new(&c_str(pw.pw_name)),
                pyre_object::w_str_new(&c_str(pw.pw_passwd)),
                pyre_object::w_int_new(pw.pw_uid as i64),
                pyre_object::w_int_new(pw.pw_gid as i64),
                pyre_object::w_str_new(&c_str(pw.pw_gecos)),
                pyre_object::w_str_new(&c_str(pw.pw_dir)),
                pyre_object::w_str_new(&c_str(pw.pw_shell)),
            ])
        }
    }
    crate::dict_storage_store(
        ns,
        "getpwuid",
        crate::make_builtin_function_with_arity(
            "getpwuid",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("getpwuid() missing argument"));
                }
                #[cfg(unix)]
                unsafe {
                    if !pyre_object::is_int(args[0]) {
                        return Err(crate::PyError::type_error(
                            "getpwuid(): uid should be an integer",
                        ));
                    }
                    let uid = pyre_object::w_int_get_value(args[0]) as u32;
                    let pw = getpwuid(uid);
                    if pw.is_null() {
                        return Err(crate::PyError::key_error(format!(
                            "getpwuid(): uid not found: {}",
                            uid
                        )));
                    }
                    return Ok(make_struct_passwd(pw));
                }
                #[cfg(not(unix))]
                Err(crate::PyError::key_error("getpwuid(): uid not found"))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getpwnam",
        crate::make_builtin_function_with_arity(
            "getpwnam",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("getpwnam() missing argument"));
                }
                #[cfg(unix)]
                unsafe {
                    if !pyre_object::is_str(args[0]) {
                        return Err(crate::PyError::type_error(
                            "getpwnam(): name should be a string",
                        ));
                    }
                    let name = pyre_object::w_str_get_value(args[0]);
                    let cname = std::ffi::CString::new(name).map_err(|_| {
                        crate::PyError::value_error("getpwnam(): embedded null character in name")
                    })?;
                    let pw = getpwnam(cname.as_ptr());
                    if pw.is_null() {
                        return Err(crate::PyError::key_error(format!(
                            "getpwnam(): name not found: {}",
                            name
                        )));
                    }
                    return Ok(make_struct_passwd(pw));
                }
                #[cfg(not(unix))]
                Err(crate::PyError::key_error("getpwnam(): name not found"))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getpwall",
        crate::make_builtin_function_with_arity(
            "getpwall",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
}

/// atexit stub — PyPy: pypy/module/atexit/. Single-threaded pyre doesn't
/// actually run the registered callbacks on shutdown yet; `register` accepts
/// any callable and returns it so `@atexit.register` decorators work.
fn init_atexit(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "register",
        crate::make_builtin_function("register", |args| {
            // Return the function so `@atexit.register` decorator form works.
            Ok(args.first().copied().unwrap_or(pyre_object::w_none()))
        }),
    );
    crate::dict_storage_store(
        ns,
        "unregister",
        crate::make_builtin_function_with_arity("unregister", |_| Ok(pyre_object::w_none()), 1),
    );
    crate::dict_storage_store(
        ns,
        "_run_exitfuncs",
        crate::make_builtin_function_with_arity("_run_exitfuncs", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "_clear",
        crate::make_builtin_function_with_arity("_clear", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "_ncallbacks",
        crate::make_builtin_function_with_arity(
            "_ncallbacks",
            |_| Ok(pyre_object::w_int_new(0)),
            0,
        ),
    );
}

/// _signal module stub — PyPy: pypy/module/signal/. Provides the signal()
/// function and SIG_DFL/SIG_IGN constants that signal.py wraps.
fn init_signal_stub(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "signal",
        crate::make_builtin_function("signal", |args| {
            // signal(signalnum, handler) — return previous handler (None stub).
            Ok(args.get(1).copied().unwrap_or(pyre_object::w_none()))
        }),
    );
    crate::dict_storage_store(
        ns,
        "getsignal",
        crate::make_builtin_function_with_arity("getsignal", |_| Ok(pyre_object::w_none()), 1),
    );
    crate::dict_storage_store(
        ns,
        "default_int_handler",
        crate::make_builtin_function_with_arity(
            "default_int_handler",
            |_| Ok(pyre_object::w_none()),
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "set_wakeup_fd",
        crate::make_builtin_function("set_wakeup_fd", |_| Ok(pyre_object::w_int_new(-1))),
    );
    crate::dict_storage_store(ns, "SIG_DFL", pyre_object::w_int_new(0));
    crate::dict_storage_store(ns, "SIG_IGN", pyre_object::w_int_new(1));
    // Common signal numbers (POSIX subset).
    crate::dict_storage_store(ns, "SIGINT", pyre_object::w_int_new(2));
    crate::dict_storage_store(ns, "SIGTERM", pyre_object::w_int_new(15));
    crate::dict_storage_store(ns, "SIGHUP", pyre_object::w_int_new(1));
    crate::dict_storage_store(ns, "SIGQUIT", pyre_object::w_int_new(3));
    crate::dict_storage_store(ns, "SIGKILL", pyre_object::w_int_new(9));
    crate::dict_storage_store(ns, "SIGUSR1", pyre_object::w_int_new(30));
    crate::dict_storage_store(ns, "SIGUSR2", pyre_object::w_int_new(31));
    crate::dict_storage_store(ns, "SIGPIPE", pyre_object::w_int_new(13));
    crate::dict_storage_store(ns, "SIGALRM", pyre_object::w_int_new(14));
    crate::dict_storage_store(ns, "SIGCHLD", pyre_object::w_int_new(20));
    crate::dict_storage_store(ns, "NSIG", pyre_object::w_int_new(64));
}

/// itertools stub
fn init_itertools(ns: &mut DictStorage) {
    // chain(*iterables) → flat iterator
    crate::dict_storage_store(
        ns,
        "chain",
        crate::make_builtin_function("chain", |args| {
            let mut items = Vec::new();
            for &arg in args {
                items.extend(crate::builtins::collect_iterable(arg)?);
            }
            let n = items.len();
            let list = pyre_object::w_list_new(items);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // starmap stub
    crate::dict_storage_store(
        ns,
        "starmap",
        crate::make_builtin_function_with_arity(
            "starmap",
            |_| Ok(pyre_object::w_list_new(vec![])),
            2,
        ),
    );
    // count(start=0, step=1) — PyPy: W_Count___new__
    //
    //     def W_Count___new__(space, w_subtype, w_start=0, w_step=1):
    //         return W_Count(space, w_start, w_step)
    crate::dict_storage_store(
        ns,
        "count",
        crate::make_builtin_function("count", |args| {
            let w_start = args.first().copied().unwrap_or(pyre_object::w_int_new(0));
            let w_step = args.get(1).copied().unwrap_or(pyre_object::w_int_new(1));
            Ok(pyre_object::itertoolsmodule::w_count_new(w_start, w_step))
        }),
    );
    // repeat(obj, times=None) — PyPy: W_Repeat___new__
    //
    //     def W_Repeat___new__(space, w_subtype, w_obj, w_times=None):
    //         return W_Repeat(space, w_obj, w_times)
    crate::dict_storage_store(
        ns,
        "repeat",
        crate::make_builtin_function("repeat", |args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error(
                    "repeat() missing 'object' argument",
                ));
            }
            let w_obj = args[0];
            let w_times = if args.len() >= 2 {
                unsafe {
                    if pyre_object::is_int(args[1]) {
                        Some(pyre_object::w_int_get_value(args[1]))
                    } else {
                        None
                    }
                }
            } else {
                None
            };
            Ok(pyre_object::itertoolsmodule::w_repeat_new(w_obj, w_times))
        }),
    );
    // islice
    crate::dict_storage_store(
        ns,
        "islice",
        crate::make_builtin_function("islice", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
    // groupby
    crate::dict_storage_store(
        ns,
        "groupby",
        crate::make_builtin_function("groupby", |_| Ok(pyre_object::w_none())),
    );
    // permutations(iterable, r=None) — PyPy: pypy/module/itertools/interp_itertools.py
    crate::dict_storage_store(
        ns,
        "permutations",
        crate::make_builtin_function("permutations", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            let pool = crate::builtins::collect_iterable(args[0])?;
            let n = pool.len();
            let r = if args.len() >= 2 {
                unsafe {
                    if pyre_object::is_int(args[1]) {
                        pyre_object::w_int_get_value(args[1]) as usize
                    } else {
                        n
                    }
                }
            } else {
                n
            };
            if r > n {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            // Heap/Lehmer would be clearer; use a recursive closure-free helper.
            fn perms(
                pool: &[pyre_object::PyObjectRef],
                r: usize,
            ) -> Vec<Vec<pyre_object::PyObjectRef>> {
                if r == 0 {
                    return vec![vec![]];
                }
                let mut out = Vec::new();
                for i in 0..pool.len() {
                    let mut rest: Vec<_> = pool.to_vec();
                    let head = rest.remove(i);
                    for mut tail in perms(&rest, r - 1) {
                        let mut v = vec![head];
                        v.append(&mut tail);
                        out.push(v);
                    }
                }
                out
            }
            let all = perms(&pool, r);
            let tuples: Vec<_> = all.into_iter().map(pyre_object::w_tuple_new).collect();
            let n = tuples.len();
            let list = pyre_object::w_list_new(tuples);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // combinations(iterable, r)
    crate::dict_storage_store(
        ns,
        "combinations",
        crate::make_builtin_function_with_arity(
            "combinations",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_list_new(vec![]));
                }
                let pool = crate::builtins::collect_iterable(args[0])?;
                let r = unsafe { pyre_object::w_int_get_value(args[1]) as usize };
                if r > pool.len() {
                    return Ok(pyre_object::w_list_new(vec![]));
                }
                fn combs(
                    pool: &[pyre_object::PyObjectRef],
                    r: usize,
                    start: usize,
                ) -> Vec<Vec<pyre_object::PyObjectRef>> {
                    if r == 0 {
                        return vec![vec![]];
                    }
                    let mut out = Vec::new();
                    for i in start..pool.len() {
                        for mut tail in combs(pool, r - 1, i + 1) {
                            let mut v = vec![pool[i]];
                            v.append(&mut tail);
                            out.push(v);
                        }
                    }
                    out
                }
                let all = combs(&pool, r, 0);
                let tuples: Vec<_> = all.into_iter().map(pyre_object::w_tuple_new).collect();
                let n = tuples.len();
                let list = pyre_object::w_list_new(tuples);
                Ok(pyre_object::w_seq_iter_new(list, n))
            },
            2,
        ),
    );
    // product(*iterables, repeat=1)
    crate::dict_storage_store(
        ns,
        "product",
        crate::make_builtin_function("product", |args| {
            let pools: Vec<Vec<_>> = args
                .iter()
                .map(|&a| crate::builtins::collect_iterable(a))
                .collect::<Result<_, _>>()?;
            let mut result: Vec<Vec<pyre_object::PyObjectRef>> = vec![vec![]];
            for pool in &pools {
                let mut new_result = Vec::with_capacity(result.len() * pool.len());
                for existing in &result {
                    for &item in pool {
                        let mut v = existing.clone();
                        v.push(item);
                        new_result.push(v);
                    }
                }
                result = new_result;
            }
            let tuples: Vec<_> = result.into_iter().map(pyre_object::w_tuple_new).collect();
            let n = tuples.len();
            let list = pyre_object::w_list_new(tuples);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // zip_longest(*iterables, fillvalue=None)
    crate::dict_storage_store(
        ns,
        "zip_longest",
        crate::make_builtin_function("zip_longest", |args| {
            let pools: Vec<Vec<_>> = args
                .iter()
                .map(|&a| crate::builtins::collect_iterable(a))
                .collect::<Result<_, _>>()?;
            let max_len = pools.iter().map(|p| p.len()).max().unwrap_or(0);
            let fill = pyre_object::w_none();
            let mut tuples = Vec::with_capacity(max_len);
            for i in 0..max_len {
                let row: Vec<_> = pools
                    .iter()
                    .map(|p| if i < p.len() { p[i] } else { fill })
                    .collect();
                tuples.push(pyre_object::w_tuple_new(row));
            }
            let n = tuples.len();
            let list = pyre_object::w_list_new(tuples);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // accumulate(iterable) — sums only, PyPy interp_itertools W_Accumulate.
    crate::dict_storage_store(
        ns,
        "accumulate",
        crate::make_builtin_function("accumulate", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            let items = crate::builtins::collect_iterable(args[0])?;
            let mut out = Vec::with_capacity(items.len());
            let mut acc: Option<pyre_object::PyObjectRef> = None;
            for item in items {
                acc = Some(match acc {
                    None => item,
                    Some(prev) => crate::baseobjspace::add(prev, item)?,
                });
                out.push(acc.unwrap());
            }
            let n = out.len();
            let list = pyre_object::w_list_new(out);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // compress(data, selectors)
    crate::dict_storage_store(
        ns,
        "compress",
        crate::make_builtin_function_with_arity(
            "compress",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_list_new(vec![]));
                }
                let data = crate::builtins::collect_iterable(args[0])?;
                let selectors = crate::builtins::collect_iterable(args[1])?;
                let mut out = Vec::new();
                for (d, s) in data.iter().zip(selectors.iter()) {
                    if crate::baseobjspace::is_true(*s) {
                        out.push(*d);
                    }
                }
                let n = out.len();
                let list = pyre_object::w_list_new(out);
                Ok(pyre_object::w_seq_iter_new(list, n))
            },
            2,
        ),
    );
}

/// _contextvars stub
fn init_contextvars(ns: &mut DictStorage) {
    // ContextVar(name, *, default=_MISSING) — context variable
    crate::dict_storage_store(
        ns,
        "ContextVar",
        crate::make_builtin_function("ContextVar", |args| {
            // Return stub object with get/set methods
            let obj = pyre_object::w_instance_new(crate::typedef::w_object());
            if !args.is_empty() {
                let _ = crate::baseobjspace::setattr(obj, "name", args[0]);
            }
            // get() returns default or raises LookupError
            let _ = crate::baseobjspace::setattr(
                obj,
                "get",
                crate::make_builtin_function("get", |args| {
                    // Return default if provided
                    if args.len() > 1 {
                        Ok(args[1])
                    } else {
                        Ok(pyre_object::w_none())
                    }
                }),
            );
            let _ = crate::baseobjspace::setattr(
                obj,
                "set",
                crate::make_builtin_function_with_arity("set", |_| Ok(pyre_object::w_none()), 2),
            );
            Ok(obj)
        }),
    );
    crate::dict_storage_store(
        ns,
        "Context",
        crate::make_builtin_function_with_arity("Context", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "Token",
        crate::make_builtin_function_with_arity("Token", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "copy_context",
        crate::make_builtin_function_with_arity("copy_context", |_| Ok(pyre_object::w_none()), 0),
    );
}

/// _abc stub — PyPy: pypy/module/_abc/
fn init_abc(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "get_cache_token",
        crate::make_builtin_function_with_arity(
            "get_cache_token",
            |_| Ok(pyre_object::w_int_new(0)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_abc_init",
        crate::make_builtin_function_with_arity("_abc_init", |_| Ok(pyre_object::w_none()), 1),
    );
    crate::dict_storage_store(
        ns,
        "_abc_register",
        crate::make_builtin_function_with_arity("_abc_register", |_| Ok(pyre_object::w_none()), 2),
    );
    // _abc_instancecheck(cls, instance) — CPython: Modules/_abc.c _abc__abc_instancecheck.
    //
    // ABCMeta.__instancecheck__ (abc.py:119) delegates here. The canonical
    // behaviour: walk type(instance).__mro__ looking for cls (direct
    // subclass), then consult cls._abc_registry for virtual subclasses
    // registered via `cls.register(subclass)`. Our previous stub
    // unconditionally returned False, which broke
    // `isinstance(Fraction(1,2), numbers.Rational)`.
    crate::dict_storage_store(
        ns,
        "_abc_instancecheck",
        crate::make_builtin_function_with_arity(
            "_abc_instancecheck",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let cls = args[0];
                let instance = args[1];
                unsafe {
                    Ok(pyre_object::w_bool_from(crate::baseobjspace::isinstance_w(
                        instance, cls,
                    )))
                }
            },
            2,
        ),
    );
    // _abc_subclasscheck(cls, subclass) — CPython: Modules/_abc.c _abc__abc_subclasscheck.
    crate::dict_storage_store(
        ns,
        "_abc_subclasscheck",
        crate::make_builtin_function_with_arity(
            "_abc_subclasscheck",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let cls = args[0];
                let subclass = args[1];
                unsafe {
                    // Walk subclass.__mro__ looking for cls.
                    let mro_ptr = pyre_object::w_type_get_mro(subclass);
                    if !mro_ptr.is_null() {
                        for &t in &*mro_ptr {
                            if std::ptr::eq(t, cls) {
                                return Ok(pyre_object::w_bool_from(true));
                            }
                        }
                    }
                }
                Ok(pyre_object::w_bool_from(false))
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_get_dump",
        crate::make_builtin_function_with_arity(
            "_get_dump",
            |_| Ok(pyre_object::w_tuple_new(vec![])),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_reset_registry",
        crate::make_builtin_function_with_arity(
            "_reset_registry",
            |_| Ok(pyre_object::w_none()),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_reset_caches",
        crate::make_builtin_function_with_arity("_reset_caches", |_| Ok(pyre_object::w_none()), 1),
    );
}

/// _functools stub
fn init_functools(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "reduce",
        crate::make_builtin_function("reduce", |_| {
            Err(crate::PyError::type_error("reduce not implemented"))
        }),
    );
    // functools.cmp_to_key(cmp) — returns a callable that wraps a value in
    // an opaque key. For sorting str / int / tuple of those (the only paths
    // pyre's stdlib actually exercises), the items are already comparable,
    // so an identity key gives the same ordering as `cmp(a, b)` would.
    crate::dict_storage_store(
        ns,
        "cmp_to_key",
        crate::make_builtin_function_with_arity(
            "cmp_to_key",
            |_args| {
                Ok(crate::make_builtin_function_with_arity(
                    "cmp_to_key.K",
                    |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
                    1,
                ))
            },
            1,
        ),
    );
}

/// Lock methods — PyPy: pypy/module/thread/os_lock.py W_Lock / W_RLock
///
/// Single-threaded pyre: state lives in the instance dict as `_locked_count`.
/// Methods increment/decrement this counter so Condition/RLock ownership
/// checks see the correct state.
fn init_lock_type(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "__enter__",
        crate::make_builtin_function_with_arity(
            "__enter__",
            |args| {
                if let Some(&obj) = args.first() {
                    lock_acquire_impl(obj)?;
                }
                Ok(args.first().copied().unwrap_or(pyre_object::w_none()))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__exit__",
        crate::make_builtin_function("__exit__", |args| {
            if let Some(&obj) = args.first() {
                lock_release_impl(obj)?;
            }
            Ok(pyre_object::w_bool_from(false))
        }),
    );
    // descr_lock_acquire — PyPy: os_lock.Lock.descr_lock_acquire
    crate::dict_storage_store(
        ns,
        "acquire",
        crate::make_builtin_function("acquire", |args| {
            let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            lock_acquire_impl(obj)?;
            Ok(pyre_object::w_bool_from(true))
        }),
    );
    // descr_lock_release — PyPy: os_lock.Lock.descr_lock_release
    crate::dict_storage_store(
        ns,
        "release",
        crate::make_builtin_function_with_arity(
            "release",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                lock_release_impl(obj)?;
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    // descr_lock_locked — PyPy: os_lock.Lock.descr_lock_locked
    crate::dict_storage_store(
        ns,
        "locked",
        crate::make_builtin_function_with_arity(
            "locked",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(pyre_object::w_bool_from(lock_count(obj) > 0))
            },
            1,
        ),
    );
    // _is_owned — used by RLock/Condition in threading.py
    crate::dict_storage_store(
        ns,
        "_is_owned",
        crate::make_builtin_function_with_arity(
            "_is_owned",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(pyre_object::w_bool_from(lock_count(obj) > 0))
            },
            1,
        ),
    );
    // _at_fork_reinit — PyPy: os_lock.Lock._at_fork_reinit (reset to unlocked)
    crate::dict_storage_store(
        ns,
        "_at_fork_reinit",
        crate::make_builtin_function_with_arity(
            "_at_fork_reinit",
            |args| {
                if let Some(&obj) = args.first() {
                    lock_set_count(obj, 0);
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
}

/// Read the lock's internal count. Single-threaded: 0 = unlocked, >0 = locked.
fn lock_count(obj: pyre_object::PyObjectRef) -> i64 {
    let w_dict = crate::baseobjspace::getdict(obj);
    if w_dict.is_null() {
        return 0;
    }
    if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(w_dict, "_locked_count") } {
        unsafe {
            if pyre_object::is_int(v) {
                return pyre_object::w_int_get_value(v);
            }
        }
    }
    0
}

fn lock_set_count(obj: pyre_object::PyObjectRef, v: i64) {
    let w_dict = crate::baseobjspace::getdict(obj);
    if w_dict.is_null() {
        return;
    }
    unsafe {
        pyre_object::w_dict_setitem_str(w_dict, "_locked_count", pyre_object::w_int_new(v));
    }
}

fn lock_acquire_impl(obj: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
    lock_set_count(obj, lock_count(obj) + 1);
    Ok(())
}

fn lock_release_impl(obj: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
    let cur = lock_count(obj);
    if cur <= 0 {
        return Err(crate::PyError::runtime_error("release unlocked lock"));
    }
    lock_set_count(obj, cur - 1);
    Ok(())
}

thread_local! {
    static LOCK_TYPE_OBJ: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    static THREAD_HANDLE_TYPE_OBJ: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
}

fn lock_type() -> PyObjectRef {
    LOCK_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("lock", init_lock_type);
            // Store per-instance `_locked_count` in the instance dict.
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

fn thread_handle_type() -> PyObjectRef {
    THREAD_HANDLE_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            crate::typedef::make_builtin_type("_ThreadHandle", |ns| {
                crate::dict_storage_store(
                    ns,
                    "is_done",
                    crate::make_builtin_function_with_arity(
                        "is_done",
                        |_| Ok(pyre_object::w_bool_from(true)),
                        1,
                    ),
                );
                crate::dict_storage_store(
                    ns,
                    "join",
                    crate::make_builtin_function("join", |_| Ok(pyre_object::w_none())),
                );
                crate::dict_storage_store(
                    ns,
                    "set_result",
                    crate::make_builtin_function_with_arity(
                        "set_result",
                        |_| Ok(pyre_object::w_none()),
                        2,
                    ),
                );
                crate::dict_storage_store(
                    ns,
                    "_set_done",
                    crate::make_builtin_function_with_arity(
                        "_set_done",
                        |_| Ok(pyre_object::w_none()),
                        1,
                    ),
                );
            })
        })
    })
}

/// _thread stub
fn init_thread(ns: &mut DictStorage) {
    let lock_tp = lock_type();
    crate::dict_storage_store(ns, "LockType", lock_tp);
    crate::dict_storage_store(
        ns,
        "RLock",
        crate::make_builtin_function_with_arity(
            "RLock",
            |_| Ok(pyre_object::w_instance_new(lock_type())),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "allocate_lock",
        crate::make_builtin_function_with_arity(
            "allocate_lock",
            |_| Ok(pyre_object::w_instance_new(lock_type())),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_ident",
        crate::make_builtin_function_with_arity("get_ident", |_| Ok(pyre_object::w_int_new(1)), 0),
    );
    crate::dict_storage_store(
        ns,
        "_count",
        crate::make_builtin_function_with_arity("_count", |_| Ok(pyre_object::w_int_new(1)), 0),
    );
    crate::dict_storage_store(ns, "TIMEOUT_MAX", pyre_object::w_float_new(f64::MAX));
    crate::dict_storage_store(ns, "error", crate::typedef::w_object());
    crate::dict_storage_store(
        ns,
        "start_joinable_thread",
        crate::make_builtin_function("start_joinable_thread", |_| Ok(pyre_object::w_int_new(0))),
    );
    crate::dict_storage_store(
        ns,
        "_set_sentinel",
        crate::make_builtin_function_with_arity(
            "_set_sentinel",
            |_| Ok(pyre_object::w_instance_new(lock_type())),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "stack_size",
        crate::make_builtin_function_with_arity("stack_size", |_| Ok(pyre_object::w_int_new(0)), 1),
    );
    crate::dict_storage_store(
        ns,
        "_is_main_interpreter",
        crate::make_builtin_function_with_arity(
            "_is_main_interpreter",
            |_| Ok(pyre_object::w_bool_from(true)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "daemon_threads_allowed",
        crate::make_builtin_function_with_arity(
            "daemon_threads_allowed",
            |_| Ok(pyre_object::w_bool_from(true)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_shutdown",
        crate::make_builtin_function_with_arity("_shutdown", |_| Ok(pyre_object::w_none()), 0),
    );
    // _make_thread_handle / _ThreadHandle — threading.py:40-41
    crate::dict_storage_store(ns, "_ThreadHandle", thread_handle_type());
    crate::dict_storage_store(
        ns,
        "_make_thread_handle",
        crate::make_builtin_function_with_arity(
            "_make_thread_handle",
            |_| Ok(pyre_object::w_instance_new(thread_handle_type())),
            1,
        ),
    );
    // _get_main_thread_ident — threading.py:43
    crate::dict_storage_store(
        ns,
        "_get_main_thread_ident",
        crate::make_builtin_function_with_arity(
            "_get_main_thread_ident",
            |_| Ok(pyre_object::w_int_new(1)),
            0,
        ),
    );
    // get_native_id — threading.py:46
    crate::dict_storage_store(
        ns,
        "get_native_id",
        crate::make_builtin_function_with_arity(
            "get_native_id",
            |_| Ok(pyre_object::w_int_new(1)),
            0,
        ),
    );
    // set_name — threading.py:52
    crate::dict_storage_store(
        ns,
        "set_name",
        crate::make_builtin_function_with_arity("set_name", |_| Ok(pyre_object::w_none()), 1),
    );
    // _excepthook — threading.py:1262
    crate::dict_storage_store(
        ns,
        "_excepthook",
        crate::make_builtin_function_with_arity("_excepthook", |_| Ok(pyre_object::w_none()), 1),
    );
    // _local — PyPy: pypy/module/thread/os_local.py Local
    // Thread-local data. Single-threaded: equivalent to a plain object with dict.
    crate::dict_storage_store(ns, "_local", local_type());
}

fn local_type() -> PyObjectRef {
    thread_local! {
        static LOCAL_TYPE_OBJ: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    }
    LOCAL_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("_local", |_ns| {});
            // Instances need __dict__ for per-thread attribute storage.
            // PyPy: os_local.py Local has getdict(space) → w_dict
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

/// posix stub — PyPy: pypy/module/posix/ interp_posix.py
///
/// Provides the minimal surface that os.py module init needs to succeed.
/// Real posix calls are not implemented — they raise or return defaults.
fn init_posix(ns: &mut DictStorage) {
    // environ — dict populated from the host environment.
    // PyPy equivalent: posix.State.startup → _convertenviron copies
    // os.environ.items() into w_environ at interpreter startup.
    let w_environ = pyre_object::w_dict_new();
    #[cfg(feature = "host_env")]
    {
        // On POSIX, posix.environ stores bytes → bytes. os.py's
        // _create_environ_mapping wraps this dict in an _Environ object that
        // encodes/decodes via surrogateescape when accessed.
        for (key, value) in std::env::vars_os() {
            let k_bytes = key.as_encoded_bytes();
            let v_bytes = value.as_encoded_bytes();
            unsafe {
                pyre_object::w_dict_store(
                    w_environ,
                    pyre_object::w_bytes_from_bytes(k_bytes),
                    pyre_object::w_bytes_from_bytes(v_bytes),
                );
            }
        }
    }
    crate::dict_storage_store(ns, "environ", w_environ);
    // _have_functions — list of HAVE_* macro names that were defined at
    // build time. os.py uses this to populate the supports_* capability
    // sets. Advertising a representative subset lets os.py module init
    // complete successfully.
    crate::dict_storage_store(
        ns,
        "_have_functions",
        pyre_object::w_list_new(vec![
            pyre_object::w_str_new("HAVE_FACCESSAT"),
            pyre_object::w_str_new("HAVE_FCHDIR"),
            pyre_object::w_str_new("HAVE_FCHMOD"),
            pyre_object::w_str_new("HAVE_FCHMODAT"),
            pyre_object::w_str_new("HAVE_FCHOWN"),
            pyre_object::w_str_new("HAVE_FCHOWNAT"),
            pyre_object::w_str_new("HAVE_FDOPENDIR"),
            pyre_object::w_str_new("HAVE_FEXECVE"),
            pyre_object::w_str_new("HAVE_FPATHCONF"),
            pyre_object::w_str_new("HAVE_FSTATAT"),
            pyre_object::w_str_new("HAVE_FSTATVFS"),
            pyre_object::w_str_new("HAVE_FTRUNCATE"),
            pyre_object::w_str_new("HAVE_FUTIMENS"),
            pyre_object::w_str_new("HAVE_FUTIMES"),
            pyre_object::w_str_new("HAVE_FUTIMESAT"),
            pyre_object::w_str_new("HAVE_LINKAT"),
            pyre_object::w_str_new("HAVE_LSTAT"),
            pyre_object::w_str_new("HAVE_MKDIRAT"),
            pyre_object::w_str_new("HAVE_MKFIFOAT"),
            pyre_object::w_str_new("HAVE_MKNODAT"),
            pyre_object::w_str_new("HAVE_OPENAT"),
            pyre_object::w_str_new("HAVE_READLINKAT"),
            pyre_object::w_str_new("HAVE_RENAMEAT"),
            pyre_object::w_str_new("HAVE_SYMLINKAT"),
            pyre_object::w_str_new("HAVE_UNLINKAT"),
            pyre_object::w_str_new("HAVE_UTIMENSAT"),
        ]),
    );
    // POSIX constants — real libc values (cross-platform subset).
    for (name, val) in [
        // F_OK/R_OK/W_OK/X_OK: Windows doesn't have them in libc crate,
        // define standard POSIX values directly.
        #[cfg(unix)]
        ("F_OK", libc::F_OK as i64),
        #[cfg(not(unix))]
        ("F_OK", 0i64),
        #[cfg(unix)]
        ("R_OK", libc::R_OK as i64),
        #[cfg(not(unix))]
        ("R_OK", 4i64),
        #[cfg(unix)]
        ("W_OK", libc::W_OK as i64),
        #[cfg(not(unix))]
        ("W_OK", 2i64),
        #[cfg(unix)]
        ("X_OK", libc::X_OK as i64),
        #[cfg(not(unix))]
        ("X_OK", 1i64),
        ("O_RDONLY", libc::O_RDONLY as i64),
        ("O_WRONLY", libc::O_WRONLY as i64),
        ("O_RDWR", libc::O_RDWR as i64),
        ("O_APPEND", libc::O_APPEND as i64),
        ("O_CREAT", libc::O_CREAT as i64),
        ("O_EXCL", libc::O_EXCL as i64),
        ("O_TRUNC", libc::O_TRUNC as i64),
        // O_NONBLOCK, O_DSYNC, O_SYNC are Unix-only.
        #[cfg(unix)]
        ("O_NONBLOCK", libc::O_NONBLOCK as i64),
        #[cfg(not(unix))]
        ("O_NONBLOCK", 0i64),
        #[cfg(unix)]
        ("O_NDELAY", libc::O_NONBLOCK as i64),
        #[cfg(not(unix))]
        ("O_NDELAY", 0i64),
        #[cfg(unix)]
        ("O_DSYNC", libc::O_DSYNC as i64),
        #[cfg(not(unix))]
        ("O_DSYNC", 0i64),
        #[cfg(unix)]
        ("O_SYNC", libc::O_SYNC as i64),
        #[cfg(not(unix))]
        ("O_SYNC", 0i64),
        ("SEEK_SET", libc::SEEK_SET as i64),
        ("SEEK_CUR", libc::SEEK_CUR as i64),
        ("SEEK_END", libc::SEEK_END as i64),
    ] {
        crate::dict_storage_store(ns, name, pyre_object::w_int_new(val));
    }
    // Non-critical constants — zero stubs are fine for os.py init.
    for name in [
        "EX_OK",
        "EX_USAGE",
        "EX_DATAERR",
        "EX_NOINPUT",
        "EX_NOUSER",
        "EX_NOHOST",
        "EX_UNAVAILABLE",
        "EX_SOFTWARE",
        "EX_OSERR",
        "EX_OSFILE",
        "EX_CANTCREAT",
        "EX_IOERR",
        "EX_TEMPFAIL",
        "EX_PROTOCOL",
        "EX_NOPERM",
        "EX_CONFIG",
        "WNOHANG",
        "WCONTINUED",
        "WUNTRACED",
        "P_WAIT",
        "P_NOWAIT",
        "P_NOWAITO",
        "ST_RDONLY",
        "ST_NOSUID",
        "SCHED_OTHER",
        "SCHED_FIFO",
        "SCHED_RR",
        "SCHED_BATCH",
        "SCHED_IDLE",
        "RTLD_LAZY",
        "RTLD_NOW",
        "RTLD_GLOBAL",
        "RTLD_LOCAL",
        "RTLD_NODELETE",
        "RTLD_NOLOAD",
        "RTLD_DEEPBIND",
        "PRIO_PROCESS",
        "PRIO_PGRP",
        "PRIO_USER",
    ] {
        crate::dict_storage_store(ns, name, pyre_object::w_int_new(0));
    }
    // Remaining noop stubs — functions os.py references at module level.
    // Functions with real implementations are registered individually below.
    for name in [
        "fstatat",
        "statvfs",
        "fstatvfs",
        "dup",
        "dup2",
        "chdir",
        "fchdir",
        "link",
        "symlink",
        "readlink",
        "chmod",
        "fchmod",
        "lchmod",
        "chown",
        "fchown",
        "lchown",
        "access",
        "faccessat",
        "chflags",
        "lchflags",
        "utime",
        "futimens",
        "futimes",
        "scandir",
        "fdopendir",
        "execve",
        "execv",
        "fork",
        "forkpty",
        "wait",
        "waitpid",
        "truncate",
        "ftruncate",
        "pathconf",
        "fpathconf",
        "getppid",
        "setuid",
        "setgid",
        "setsid",
        "setpgid",
        "setreuid",
        "setregid",
        "getgroups",
        "setgroups",
        "getpgrp",
        "setpgrp",
        "getpgid",
        "umask",
        "getlogin",
        "nice",
        "pipe",
        "pipe2",
        "dup3",
        "fsync",
        "fdatasync",
        "mkfifo",
        "mknod",
        "major",
        "minor",
        "makedev",
        "get_inheritable",
        "set_inheritable",
        "get_blocking",
        "set_blocking",
        // "get_terminal_size" — implemented below
        "cpu_count",
        "getloadavg",
        "kill",
        "killpg",
        "getpriority",
        "setpriority",
        "sched_get_priority_max",
        "sched_get_priority_min",
        "sched_getparam",
        "sched_setparam",
        "sched_getscheduler",
        "sched_setscheduler",
        "sched_yield",
        "confstr",
        "confstr_names",
        "sysconf",
        "sysconf_names",
        "pathconf_names",
        "setenv",
        "unsetenv",
        "putenv",
        "device_encoding",
        "ttyname",
        "openpty",
        "login_tty",
        "tcgetpgrp",
        "tcsetpgrp",
        "ctermid",
        "get_exec_path",
        "WIFEXITED",
        "WEXITSTATUS",
        "WIFSIGNALED",
        "WTERMSIG",
        "WIFSTOPPED",
        "WSTOPSIG",
        "WEXITED",
        "WNOWAIT",
        "WSTOPPED",
        "waitstatus_to_exitcode",
        "_exit",
        "_cpu_count",
        "register_at_fork",
        "abort",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "system",
        "popen",
    ] {
        crate::dict_storage_store(
            ns,
            name,
            crate::make_builtin_function(name, |_| Ok(pyre_object::w_none())),
        );
    }

    // ── Helper: extract a filesystem path (str or bytes) from a PyObjectRef ──
    fn extract_path(obj: pyre_object::PyObjectRef) -> Result<String, crate::PyError> {
        unsafe {
            if pyre_object::is_str(obj) {
                return Ok(pyre_object::w_str_get_value(obj).to_string());
            }
            if pyre_object::bytesobject::is_bytes_like(obj) {
                let data = pyre_object::bytesobject::bytes_like_data(obj);
                return Ok(String::from_utf8_lossy(data).into_owned());
            }
        }
        if let Ok(fspath) = crate::baseobjspace::getattr(obj, "__fspath__") {
            let result = crate::call_function(fspath, &[obj]);
            if !result.is_null() && unsafe { pyre_object::is_str(result) } {
                return Ok(unsafe { pyre_object::w_str_get_value(result).to_string() });
            }
        }
        Err(crate::PyError::type_error(
            "expected str, bytes or os.PathLike",
        ))
    }

    // ── Helper: convert std::io::Error → PyError (OSError) ──
    fn io_err(e: std::io::Error, path: &str) -> crate::PyError {
        crate::PyError::os_error_with_errno(
            e.raw_os_error().unwrap_or(0),
            format!("{}: '{}'", e, path),
        )
    }

    // ── posix.open(path, flags, mode=0o777) → fd ──
    crate::dict_storage_store(
        ns,
        "open",
        crate::make_builtin_function("open", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error(
                    "open() requires at least 2 arguments",
                ));
            }
            let path = extract_path(args[0])?;
            let flags = unsafe { pyre_object::w_int_get_value(args[1]) } as libc::c_int;
            let mode: u32 = if args.len() >= 3 {
                (unsafe { pyre_object::w_int_get_value(args[2]) }) as u32
            } else {
                0o777
            };
            let c_path = std::ffi::CString::new(path.as_bytes())
                .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
            let fd = unsafe { libc::open(c_path.as_ptr(), flags, mode as libc::c_uint) };
            if fd < 0 {
                return Err(io_err(std::io::Error::last_os_error(), &path));
            }
            Ok(pyre_object::w_int_new(fd as i64))
        }),
    );

    // ── posix.close(fd) ──
    crate::dict_storage_store(
        ns,
        "close",
        crate::make_builtin_function_with_arity(
            "close",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("close() requires 1 argument"));
                }
                let fd = unsafe { pyre_object::w_int_get_value(args[0]) } as libc::c_int;
                let ret = unsafe { libc::close(fd) };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), ""));
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // ── posix.read(fd, n) → bytes ──
    crate::dict_storage_store(
        ns,
        "read",
        crate::make_builtin_function_with_arity(
            "read",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("read() requires 2 arguments"));
                }
                let fd = unsafe { pyre_object::w_int_get_value(args[0]) } as libc::c_int;
                let n = unsafe { pyre_object::w_int_get_value(args[1]) } as usize;
                let mut buf = vec![0u8; n];
                let ret = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, n as _) };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), ""));
                }
                buf.truncate(ret as usize);
                Ok(pyre_object::w_bytes_from_bytes(&buf))
            },
            2,
        ),
    );

    // ── posix.write(fd, data) → nbytes ──
    crate::dict_storage_store(
        ns,
        "write",
        crate::make_builtin_function_with_arity(
            "write",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("write() requires 2 arguments"));
                }
                let fd = unsafe { pyre_object::w_int_get_value(args[0]) } as libc::c_int;
                let data = unsafe {
                    if pyre_object::bytesobject::is_bytes_like(args[1]) {
                        pyre_object::bytesobject::bytes_like_data(args[1]).to_vec()
                    } else if pyre_object::is_str(args[1]) {
                        pyre_object::w_str_get_value(args[1]).as_bytes().to_vec()
                    } else {
                        return Err(crate::PyError::type_error(
                            "write() arg 2 must be bytes-like",
                        ));
                    }
                };
                let ret = unsafe {
                    libc::write(fd, data.as_ptr() as *const libc::c_void, data.len() as _)
                };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), ""));
                }
                Ok(pyre_object::w_int_new(ret as i64))
            },
            2,
        ),
    );

    // ── posix.lseek(fd, offset, whence) → position ──
    crate::dict_storage_store(
        ns,
        "lseek",
        crate::make_builtin_function_with_arity(
            "lseek",
            |args| {
                if args.len() < 3 {
                    return Err(crate::PyError::type_error("lseek() requires 3 arguments"));
                }
                let fd = unsafe { pyre_object::w_int_get_value(args[0]) } as libc::c_int;
                let offset = unsafe { pyre_object::w_int_get_value(args[1]) } as libc::off_t;
                let whence = unsafe { pyre_object::w_int_get_value(args[2]) } as libc::c_int;
                let ret = unsafe { libc::lseek(fd, offset, whence) };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), ""));
                }
                Ok(pyre_object::w_int_new(ret as i64))
            },
            3,
        ),
    );

    // ── posix.unlink(path) / posix.remove(path) ──
    fn posix_unlink(
        args: &[pyre_object::PyObjectRef],
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        if args.is_empty() {
            return Err(crate::PyError::type_error("unlink() requires 1 argument"));
        }
        let path = extract_path(args[0])?;
        let c_path = std::ffi::CString::new(path.as_bytes())
            .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
        let ret = unsafe { libc::unlink(c_path.as_ptr()) };
        if ret < 0 {
            return Err(io_err(std::io::Error::last_os_error(), &path));
        }
        Ok(pyre_object::w_none())
    }
    crate::dict_storage_store(
        ns,
        "unlink",
        crate::make_builtin_function_with_arity("unlink", posix_unlink, 1),
    );
    crate::dict_storage_store(
        ns,
        "remove",
        crate::make_builtin_function_with_arity("remove", posix_unlink, 1),
    );

    // ── posix.mkdir(path, mode=0o777) ──
    crate::dict_storage_store(
        ns,
        "mkdir",
        crate::make_builtin_function("mkdir", |args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error("mkdir() requires 1 argument"));
            }
            let path = extract_path(args[0])?;
            let _mode: u32 = if args.len() >= 2 {
                (unsafe { pyre_object::w_int_get_value(args[1]) }) as u32
            } else {
                0o777
            };
            let c_path = std::ffi::CString::new(path.as_bytes())
                .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
            #[cfg(unix)]
            let ret = unsafe { libc::mkdir(c_path.as_ptr(), _mode as libc::mode_t) };
            #[cfg(windows)]
            let ret = unsafe { libc::mkdir(c_path.as_ptr()) };
            if ret < 0 {
                return Err(io_err(std::io::Error::last_os_error(), &path));
            }
            Ok(pyre_object::w_none())
        }),
    );

    // ── posix.rmdir(path) ──
    crate::dict_storage_store(
        ns,
        "rmdir",
        crate::make_builtin_function_with_arity(
            "rmdir",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("rmdir() requires 1 argument"));
                }
                let path = extract_path(args[0])?;
                let c_path = std::ffi::CString::new(path.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                let ret = unsafe { libc::rmdir(c_path.as_ptr()) };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), &path));
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // ── posix.rename(src, dst) ──
    crate::dict_storage_store(
        ns,
        "rename",
        crate::make_builtin_function_with_arity(
            "rename",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("rename() requires 2 arguments"));
                }
                let src = extract_path(args[0])?;
                let dst = extract_path(args[1])?;
                std::fs::rename(&src, &dst).map_err(|e| io_err(e, &src))?;
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    // ── posix.listdir(path=".") → list of str ──
    crate::dict_storage_store(
        ns,
        "listdir",
        crate::make_builtin_function("listdir", |args| {
            let path = if args.is_empty() || unsafe { pyre_object::is_none(args[0]) } {
                ".".to_string()
            } else {
                extract_path(args[0])?
            };
            let entries = std::fs::read_dir(&path).map_err(|e| io_err(e, &path))?;
            let mut items = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|e| io_err(e, &path))?;
                let name = entry.file_name();
                items.push(pyre_object::w_str_new(&name.to_string_lossy()));
            }
            Ok(pyre_object::w_list_new(items))
        }),
    );

    // ── posix.isatty(fd) → bool ──
    crate::dict_storage_store(
        ns,
        "isatty",
        crate::make_builtin_function_with_arity(
            "isatty",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let fd = unsafe { pyre_object::w_int_get_value(args[0]) } as libc::c_int;
                let ret = unsafe { libc::isatty(fd) };
                Ok(pyre_object::w_bool_from(ret != 0))
            },
            1,
        ),
    );

    // ── posix.urandom(n) → bytes ──
    crate::dict_storage_store(
        ns,
        "urandom",
        crate::make_builtin_function_with_arity(
            "urandom",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("urandom() requires 1 argument"));
                }
                let n = unsafe { pyre_object::w_int_get_value(args[0]) } as usize;
                let mut buf = vec![0u8; n];
                // Use /dev/urandom on Unix
                #[cfg(unix)]
                {
                    use std::io::Read;
                    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
                        let _ = f.read_exact(&mut buf);
                    }
                }
                Ok(pyre_object::w_bytes_from_bytes(&buf))
            },
            1,
        ),
    );
    // os.terminal_size — namedtuple-like type with columns/lines.
    // Uses stat_result_type (hasdict instance) so setattr works.
    fn make_terminal_size(cols: i64, lines: i64) -> pyre_object::PyObjectRef {
        let instance = pyre_object::w_instance_new(stat_result_type());
        let _ = crate::baseobjspace::setattr(instance, "columns", pyre_object::w_int_new(cols));
        let _ = crate::baseobjspace::setattr(instance, "lines", pyre_object::w_int_new(lines));
        instance
    }
    let terminal_size_type = crate::typedef::make_builtin_type("terminal_size", |ns| {
        crate::dict_storage_store(
            ns,
            "__new__",
            crate::make_builtin_function("__new__", |args| {
                let (cols, rows) = if args.len() >= 2 {
                    let seq = args[1];
                    unsafe {
                        if pyre_object::is_tuple(seq) {
                            let c = pyre_object::w_tuple_getitem(seq, 0)
                                .map(|v| pyre_object::w_int_get_value(v))
                                .unwrap_or(80);
                            let r = pyre_object::w_tuple_getitem(seq, 1)
                                .map(|v| pyre_object::w_int_get_value(v))
                                .unwrap_or(24);
                            (c, r)
                        } else {
                            (80, 24)
                        }
                    }
                } else {
                    (80, 24)
                };
                Ok(make_terminal_size(cols, rows))
            }),
        );
    });
    crate::dict_storage_store(ns, "terminal_size", terminal_size_type);

    // ── posix.get_terminal_size(fd=1) → os.terminal_size(columns, lines) ──
    crate::dict_storage_store(
        ns,
        "get_terminal_size",
        crate::make_builtin_function_with_arity(
            "get_terminal_size",
            |_args| {
                let (cols, rows) = {
                    #[cfg(unix)]
                    {
                        let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
                        let ret = unsafe { libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) };
                        if ret == 0 && ws.ws_col > 0 {
                            (ws.ws_col as i64, ws.ws_row as i64)
                        } else {
                            (80, 24)
                        }
                    }
                    #[cfg(not(unix))]
                    {
                        (80, 24)
                    }
                };
                let result = pyre_object::w_tuple_new(vec![
                    pyre_object::w_int_new(cols),
                    pyre_object::w_int_new(rows),
                ]);
                let wrapper = pyre_object::w_instance_new(stat_result_type());
                let _ =
                    crate::baseobjspace::setattr(wrapper, "columns", pyre_object::w_int_new(cols));
                let _ =
                    crate::baseobjspace::setattr(wrapper, "lines", pyre_object::w_int_new(rows));
                let _ = crate::baseobjspace::setattr(wrapper, "__tuple__", result);
                Ok(wrapper)
            },
            0,
        ),
    );
    // os.fspath() — PyPy: posixmodule.c posix_fspath. Returns the argument
    // unchanged for str/bytes/bytearray (the protocol's identity case);
    // any other object would normally trigger __fspath__ but we don't
    // model that protocol yet.
    crate::dict_storage_store(
        ns,
        "fspath",
        crate::make_builtin_function_with_arity(
            "fspath",
            |args| {
                let arg = args.first().copied().unwrap_or(pyre_object::w_none());
                unsafe {
                    if pyre_object::is_str(arg) || pyre_object::bytesobject::is_bytes_like(arg) {
                        return Ok(arg);
                    }
                }
                // Try __fspath__ — for pathlib.Path-like objects.
                if let Ok(method) = crate::baseobjspace::getattr(arg, "__fspath__") {
                    let result = crate::call_function(method, &[arg]);
                    if !result.is_null() {
                        return Ok(result);
                    }
                }
                Ok(arg)
            },
            1,
        ),
    );
    // os.stat / os.lstat / os.fstat — return stat_result structseq.
    // PyPy: posixmodule.c posix_do_stat → build_stat_result.
    //
    // The returned object is a tuple subclass with named attributes
    // (st_mode, st_ino, ...). We expose it as a plain instance with
    // attributes so that both `os.stat(p).st_mode` and
    // `os.stat(p)[0]` work.
    fn make_stat_result(meta: &std::fs::Metadata) -> pyre_object::PyObjectRef {
        // Extract stat fields in a cross-platform way.
        #[cfg(unix)]
        let (
            st_mode,
            st_ino,
            st_dev,
            st_nlink,
            st_uid,
            st_gid,
            st_size,
            st_atime,
            st_mtime,
            st_ctime,
            st_atime_ns,
            st_mtime_ns,
            st_ctime_ns,
        ) = {
            use std::os::unix::fs::MetadataExt;
            (
                meta.mode() as i64,
                meta.ino() as i64,
                meta.dev() as i64,
                meta.nlink() as i64,
                meta.uid() as i64,
                meta.gid() as i64,
                meta.size() as i64,
                meta.atime(),
                meta.mtime(),
                meta.ctime(),
                meta.atime() * 1_000_000_000 + meta.atime_nsec(),
                meta.mtime() * 1_000_000_000 + meta.mtime_nsec(),
                meta.ctime() * 1_000_000_000 + meta.ctime_nsec(),
            )
        };
        #[cfg(windows)]
        let (
            st_mode,
            st_ino,
            st_dev,
            st_nlink,
            st_uid,
            st_gid,
            st_size,
            st_atime,
            st_mtime,
            st_ctime,
            st_atime_ns,
            st_mtime_ns,
            st_ctime_ns,
        ) = {
            use std::os::windows::fs::MetadataExt;
            let ft = meta.file_type();
            let attrs = meta.file_attributes();
            let mode: i64 = if ft.is_symlink() {
                // S_IFLNK | 0o777
                0o120777
            } else if ft.is_dir() {
                0o40755
            } else if attrs & 0x1 != 0 {
                // FILE_ATTRIBUTE_READONLY
                0o100444
            } else {
                0o100644
            };
            let size = meta.file_size() as i64;
            // Windows FILETIME is 100-ns intervals since 1601-01-01.
            // Convert to Unix epoch seconds.
            const EPOCH_DIFF: i64 = 11_644_473_600;
            let atime_secs = (meta.last_access_time() as i64 / 10_000_000) - EPOCH_DIFF;
            let mtime_secs = (meta.last_write_time() as i64 / 10_000_000) - EPOCH_DIFF;
            let ctime_secs = (meta.creation_time() as i64 / 10_000_000) - EPOCH_DIFF;
            let atime_ns =
                ((meta.last_access_time() as i64 % 10_000_000) * 100) + atime_secs * 1_000_000_000;
            let mtime_ns =
                ((meta.last_write_time() as i64 % 10_000_000) * 100) + mtime_secs * 1_000_000_000;
            let ctime_ns =
                ((meta.creation_time() as i64 % 10_000_000) * 100) + ctime_secs * 1_000_000_000;
            (
                mode, 0i64, // st_ino — not available on Windows
                0i64, // st_dev
                1i64, // nlink — not easily available on stable Windows
                0i64, // st_uid
                0i64, // st_gid
                size, atime_secs, mtime_secs, ctime_secs, atime_ns, mtime_ns, ctime_ns,
            )
        };

        let tuple = pyre_object::w_tuple_new(vec![
            pyre_object::w_int_new(st_mode),
            pyre_object::w_int_new(st_ino),
            pyre_object::w_int_new(st_dev),
            pyre_object::w_int_new(st_nlink),
            pyre_object::w_int_new(st_uid),
            pyre_object::w_int_new(st_gid),
            pyre_object::w_int_new(st_size),
            pyre_object::w_int_new(st_atime),
            pyre_object::w_int_new(st_mtime),
            pyre_object::w_int_new(st_ctime),
        ]);
        // Attach st_* attributes via a wrapping instance.
        let wrapper = pyre_object::w_instance_new(stat_result_type());
        let _ = crate::baseobjspace::setattr(wrapper, "__tuple__", tuple);
        let _ = crate::baseobjspace::setattr(wrapper, "st_mode", pyre_object::w_int_new(st_mode));
        let _ = crate::baseobjspace::setattr(wrapper, "st_ino", pyre_object::w_int_new(st_ino));
        let _ = crate::baseobjspace::setattr(wrapper, "st_dev", pyre_object::w_int_new(st_dev));
        let _ = crate::baseobjspace::setattr(wrapper, "st_nlink", pyre_object::w_int_new(st_nlink));
        let _ = crate::baseobjspace::setattr(wrapper, "st_uid", pyre_object::w_int_new(st_uid));
        let _ = crate::baseobjspace::setattr(wrapper, "st_gid", pyre_object::w_int_new(st_gid));
        let _ = crate::baseobjspace::setattr(wrapper, "st_size", pyre_object::w_int_new(st_size));
        let _ = crate::baseobjspace::setattr(
            wrapper,
            "st_atime",
            pyre_object::w_float_new(st_atime as f64),
        );
        let _ = crate::baseobjspace::setattr(
            wrapper,
            "st_mtime",
            pyre_object::w_float_new(st_mtime as f64),
        );
        let _ = crate::baseobjspace::setattr(
            wrapper,
            "st_ctime",
            pyre_object::w_float_new(st_ctime as f64),
        );
        let _ = crate::baseobjspace::setattr(
            wrapper,
            "st_atime_ns",
            pyre_object::w_int_new(st_atime_ns),
        );
        let _ = crate::baseobjspace::setattr(
            wrapper,
            "st_mtime_ns",
            pyre_object::w_int_new(st_mtime_ns),
        );
        let _ = crate::baseobjspace::setattr(
            wrapper,
            "st_ctime_ns",
            pyre_object::w_int_new(st_ctime_ns),
        );
        wrapper
    }
    fn stat_impl(
        args: &[pyre_object::PyObjectRef],
        follow_symlinks: bool,
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        if args.is_empty() {
            return Err(crate::PyError::type_error("stat() missing argument"));
        }
        let path_obj = args[0];
        let path_str = unsafe {
            if pyre_object::is_str(path_obj) {
                pyre_object::w_str_get_value(path_obj).to_string()
            } else if pyre_object::bytesobject::is_bytes_like(path_obj) {
                let data = pyre_object::bytesobject::bytes_like_data(path_obj);
                String::from_utf8_lossy(data).into_owned()
            } else if let Ok(fspath) = crate::baseobjspace::getattr(path_obj, "__fspath__") {
                let result = crate::call_function(fspath, &[path_obj]);
                if !result.is_null() && pyre_object::is_str(result) {
                    pyre_object::w_str_get_value(result).to_string()
                } else {
                    return Err(crate::PyError::type_error(
                        "stat: path should be string, bytes, os.PathLike",
                    ));
                }
            } else {
                return Err(crate::PyError::type_error(
                    "stat: path should be string, bytes, os.PathLike",
                ));
            }
        };
        let meta = if follow_symlinks {
            std::fs::metadata(&path_str)
        } else {
            std::fs::symlink_metadata(&path_str)
        };
        match meta {
            Ok(m) => Ok(make_stat_result(&m)),
            Err(e) => {
                let kind = e.raw_os_error().unwrap_or(2);
                Err(crate::PyError::os_error_with_errno(
                    kind,
                    format!("{}: '{}'", e, path_str),
                ))
            }
        }
    }
    // os.uname() — returns structseq (sysname, nodename, release, version, machine).
    crate::dict_storage_store(
        ns,
        "uname",
        crate::make_builtin_function_with_arity(
            "uname",
            |_| {
                let wrapper = pyre_object::w_instance_new(stat_result_type());
                let sysname = std::env::consts::OS.to_string();
                let machine = std::env::consts::ARCH.to_string();
                let _ = crate::baseobjspace::setattr(
                    wrapper,
                    "sysname",
                    pyre_object::w_str_new(&sysname),
                );
                let _ =
                    crate::baseobjspace::setattr(wrapper, "nodename", pyre_object::w_str_new(""));
                let _ =
                    crate::baseobjspace::setattr(wrapper, "release", pyre_object::w_str_new(""));
                let _ =
                    crate::baseobjspace::setattr(wrapper, "version", pyre_object::w_str_new(""));
                let _ = crate::baseobjspace::setattr(
                    wrapper,
                    "machine",
                    pyre_object::w_str_new(&machine),
                );
                Ok(wrapper)
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "stat",
        crate::make_builtin_function_with_arity("stat", |args| stat_impl(args, true), 1),
    );
    crate::dict_storage_store(
        ns,
        "lstat",
        crate::make_builtin_function_with_arity("lstat", |args| stat_impl(args, false), 1),
    );
    crate::dict_storage_store(
        ns,
        "fstat",
        crate::make_builtin_function_with_arity(
            "fstat",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("fstat() missing argument"));
                }
                let fd = unsafe { pyre_object::w_int_get_value(args[0]) } as i32;
                #[cfg(unix)]
                {
                    use std::os::unix::io::FromRawFd;
                    let f = unsafe { std::fs::File::from_raw_fd(fd) };
                    let meta = f.metadata();
                    let _ = std::mem::ManuallyDrop::new(f); // don't close
                    match meta {
                        Ok(m) => Ok(make_stat_result(&m)),
                        Err(e) => Err(crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(9),
                            format!("{}", e),
                        )),
                    }
                }
                #[cfg(not(unix))]
                Err(crate::PyError::os_error_with_errno(
                    9,
                    "fstat unsupported".to_string(),
                ))
            },
            1,
        ),
    );
    // stat_result type — simple instance with hasdict so setattr works.
    // Exported so that `posix.stat_result` can be looked up.
    crate::dict_storage_store(ns, "stat_result", stat_result_type());
    // os.getcwd() — PyPy: posixmodule.c posix_getcwd.
    crate::dict_storage_store(
        ns,
        "getcwd",
        crate::make_builtin_function_with_arity(
            "getcwd",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    if let Ok(cwd) = std::env::current_dir() {
                        return Ok(pyre_object::w_str_new(&cwd.to_string_lossy()));
                    }
                }
                Ok(pyre_object::w_str_new(""))
            },
            0,
        ),
    );
    // os.getcwdb() — bytes form of getcwd.
    crate::dict_storage_store(
        ns,
        "getcwdb",
        crate::make_builtin_function_with_arity(
            "getcwdb",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    if let Ok(cwd) = std::env::current_dir() {
                        return Ok(pyre_object::w_bytes_from_bytes(
                            cwd.as_os_str().as_encoded_bytes(),
                        ));
                    }
                }
                Ok(pyre_object::w_bytes_from_bytes(b""))
            },
            0,
        ),
    );
    // os.getuid / geteuid / getgid / getegid — real syscalls.
    #[cfg(unix)]
    unsafe extern "C" {
        fn getuid() -> u32;
        fn geteuid() -> u32;
        fn getgid() -> u32;
        fn getegid() -> u32;
    }
    crate::dict_storage_store(
        ns,
        "getuid",
        crate::make_builtin_function_with_arity(
            "getuid",
            |_| {
                #[cfg(unix)]
                unsafe {
                    return Ok(pyre_object::w_int_new(getuid() as i64));
                }
                #[cfg(not(unix))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "geteuid",
        crate::make_builtin_function_with_arity(
            "geteuid",
            |_| {
                #[cfg(unix)]
                unsafe {
                    return Ok(pyre_object::w_int_new(geteuid() as i64));
                }
                #[cfg(not(unix))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getgid",
        crate::make_builtin_function_with_arity(
            "getgid",
            |_| {
                #[cfg(unix)]
                unsafe {
                    return Ok(pyre_object::w_int_new(getgid() as i64));
                }
                #[cfg(not(unix))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getegid",
        crate::make_builtin_function_with_arity(
            "getegid",
            |_| {
                #[cfg(unix)]
                unsafe {
                    return Ok(pyre_object::w_int_new(getegid() as i64));
                }
                #[cfg(not(unix))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    // os.getpid — std::process::id.
    crate::dict_storage_store(
        ns,
        "getpid",
        crate::make_builtin_function_with_arity(
            "getpid",
            |_| Ok(pyre_object::w_int_new(std::process::id() as i64)),
            0,
        ),
    );
    // os.environ lookups from setenv / unsetenv / putenv / getenv — mutate
    // posix.environ (the dict) rather than calling libc; os.py writes back
    // into that dict in its _Environ wrapper.
    crate::dict_storage_store(
        ns,
        "getenv",
        crate::make_builtin_function("getenv", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            let key = unsafe {
                if pyre_object::is_str(args[0]) {
                    pyre_object::w_str_get_value(args[0]).to_string()
                } else {
                    return Ok(pyre_object::w_none());
                }
            };
            #[cfg(feature = "host_env")]
            {
                if let Ok(value) = std::env::var(&key) {
                    return Ok(pyre_object::w_str_new(&value));
                }
            }
            if args.len() >= 2 {
                Ok(args[1])
            } else {
                Ok(pyre_object::w_none())
            }
        }),
    );
    crate::dict_storage_store(ns, "error", crate::typedef::w_object());
}

/// _collections C-extension stub — PyPy: pypy/module/_collections/
/// Provides the C-accelerated deque/defaultdict/OrderedDict types.
/// Our stubs are backed by lists/dicts, which is correct semantically
/// but not performant. PyPy's W_Deque is a doubly-linked block list.
fn init_collections_c(ns: &mut DictStorage) {
    // deque(iterable=(), maxlen=None) — returns a list that we alias as deque.
    // Sufficient for collections.py's MutableSequence.register(deque).
    let deque_type = crate::typedef::make_builtin_type("deque", init_deque_type);
    crate::dict_storage_store(ns, "deque", deque_type);
    // _deque_iterator — reuse object (just a type sentinel)
    crate::dict_storage_store(ns, "_deque_iterator", crate::typedef::w_object());
    // defaultdict — returns a dict-like instance
    let defaultdict_type = crate::typedef::make_builtin_type("defaultdict", init_defaultdict_type);
    crate::dict_storage_store(ns, "defaultdict", defaultdict_type);
    // OrderedDict — same as dict for our purposes
    crate::dict_storage_store(ns, "OrderedDict", crate::typedef::w_type());
}

/// deque methods — PyPy: pypy/module/_collections/interp_deque.py W_Deque
fn init_deque_type(ns: &mut DictStorage) {
    // __init__(self, iterable=(), maxlen=None) — store items as __data__ list
    crate::dict_storage_store(
        ns,
        "__init__",
        crate::make_builtin_function("__init__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            let self_obj = args[0];
            let items: Vec<_> = if args.len() >= 2 {
                crate::builtins::collect_iterable(args[1]).unwrap_or_default()
            } else {
                Vec::new()
            };
            let list = pyre_object::w_list_new(items);
            let _ = crate::baseobjspace::setattr(self_obj, "__data__", list);
            let _ = crate::baseobjspace::setattr(
                self_obj,
                "maxlen",
                if args.len() >= 3 {
                    args[2]
                } else {
                    pyre_object::w_none()
                },
            );
            Ok(pyre_object::w_none())
        }),
    );
    crate::dict_storage_store(
        ns,
        "append",
        crate::make_builtin_function_with_arity(
            "append",
            |args| {
                if args.len() >= 2 {
                    if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                        unsafe { pyre_object::w_list_append(data, args[1]) };
                    }
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "appendleft",
        crate::make_builtin_function_with_arity(
            "appendleft",
            |args| {
                if args.len() >= 2 {
                    if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                        unsafe {
                            let n = pyre_object::w_list_len(data);
                            let mut items: Vec<_> = (0..n)
                                .filter_map(|i| pyre_object::w_list_getitem(data, i as i64))
                                .collect();
                            items.insert(0, args[1]);
                            let new_list = pyre_object::w_list_new(items);
                            let _ = crate::baseobjspace::setattr(args[0], "__data__", new_list);
                        }
                    }
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "pop",
        crate::make_builtin_function_with_arity(
            "pop",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_none());
                }
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    unsafe {
                        let n = pyre_object::w_list_len(data);
                        if n > 0 {
                            let item = pyre_object::w_list_getitem(data, (n - 1) as i64)
                                .unwrap_or(pyre_object::w_none());
                            let items: Vec<_> = (0..n - 1)
                                .filter_map(|i| pyre_object::w_list_getitem(data, i as i64))
                                .collect();
                            let new_list = pyre_object::w_list_new(items);
                            let _ = crate::baseobjspace::setattr(args[0], "__data__", new_list);
                            return Ok(item);
                        }
                    }
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "popleft",
        crate::make_builtin_function_with_arity(
            "popleft",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_none());
                }
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    unsafe {
                        let n = pyre_object::w_list_len(data);
                        if n > 0 {
                            let item = pyre_object::w_list_getitem(data, 0)
                                .unwrap_or(pyre_object::w_none());
                            let items: Vec<_> = (1..n)
                                .filter_map(|i| pyre_object::w_list_getitem(data, i as i64))
                                .collect();
                            let new_list = pyre_object::w_list_new(items);
                            let _ = crate::baseobjspace::setattr(args[0], "__data__", new_list);
                            return Ok(item);
                        }
                    }
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "clear",
        crate::make_builtin_function_with_arity(
            "clear",
            |args| {
                if !args.is_empty() {
                    let _ = crate::baseobjspace::setattr(
                        args[0],
                        "__data__",
                        pyre_object::w_list_new(vec![]),
                    );
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "extend",
        crate::make_builtin_function_with_arity(
            "extend",
            |args| {
                if args.len() >= 2 {
                    let items = crate::builtins::collect_iterable(args[1])?;
                    if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                        for item in items {
                            unsafe { pyre_object::w_list_append(data, item) };
                        }
                    }
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__len__",
        crate::make_builtin_function_with_arity(
            "__len__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_int_new(0));
                }
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    return Ok(pyre_object::w_int_new(
                        unsafe { pyre_object::w_list_len(data) } as i64,
                    ));
                }
                Ok(pyre_object::w_int_new(0))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__iter__",
        crate::make_builtin_function_with_arity(
            "__iter__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_seq_iter_new(
                        pyre_object::w_list_new(vec![]),
                        0,
                    ));
                }
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    return crate::baseobjspace::iter(data);
                }
                Ok(pyre_object::w_seq_iter_new(
                    pyre_object::w_list_new(vec![]),
                    0,
                ))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__getitem__",
        crate::make_builtin_function_with_arity(
            "__getitem__",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_none());
                }
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    return crate::baseobjspace::getitem(data, args[1]);
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
}

/// defaultdict — PyPy: pypy/module/_collections/interp_defaultdict.py
fn init_defaultdict_type(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "__init__",
        crate::make_builtin_function("__init__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            let self_obj = args[0];
            let factory = if args.len() >= 2 {
                args[1]
            } else {
                pyre_object::w_none()
            };
            let _ = crate::baseobjspace::setattr(self_obj, "default_factory", factory);
            let _ = crate::baseobjspace::setattr(self_obj, "__data__", pyre_object::w_dict_new());
            Ok(pyre_object::w_none())
        }),
    );
    crate::dict_storage_store(
        ns,
        "__getitem__",
        crate::make_builtin_function_with_arity(
            "__getitem__",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_none());
                }
                let self_obj = args[0];
                let key = args[1];
                if let Ok(data) = crate::baseobjspace::getattr(self_obj, "__data__") {
                    unsafe {
                        if let Some(v) = pyre_object::w_dict_lookup(data, key) {
                            return Ok(v);
                        }
                    }
                    // Not present — try factory
                    if let Ok(factory) = crate::baseobjspace::getattr(self_obj, "default_factory") {
                        if !factory.is_null() && !unsafe { pyre_object::is_none(factory) } {
                            // Can't easily call factory without frame — return None.
                            let default = pyre_object::w_none();
                            unsafe { pyre_object::w_dict_store(data, key, default) };
                            return Ok(default);
                        }
                    }
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__setitem__",
        crate::make_builtin_function_with_arity(
            "__setitem__",
            |args| {
                if args.len() >= 3 {
                    if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                        unsafe { pyre_object::w_dict_store(data, args[1], args[2]) };
                    }
                }
                Ok(pyre_object::w_none())
            },
            3,
        ),
    );
}

/// _opcode stub — PyPy: pypy/module/_opcode (CPython's opcode introspection).
/// opcode.py requires stack_effect + has_arg/has_const/has_name/has_jump and
/// related classifiers. Our stubs return neutral values; full implementations
/// would mirror CPython Python/compile.c.
fn init_opcode_c(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "stack_effect",
        crate::make_builtin_function_with_arity(
            "stack_effect",
            |_| Ok(pyre_object::w_int_new(0)),
            3,
        ),
    );
    for name in [
        "has_arg",
        "has_const",
        "has_name",
        "has_jump",
        "has_jrel",
        "has_jabs",
        "has_free",
        "has_local",
        "has_exc",
    ] {
        crate::dict_storage_store(
            ns,
            name,
            crate::make_builtin_function_with_arity(
                name,
                |_| Ok(pyre_object::w_bool_from(false)),
                0,
            ),
        );
    }
    crate::dict_storage_store(
        ns,
        "get_executor",
        crate::make_builtin_function_with_arity("get_executor", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "get_specialization_stats",
        crate::make_builtin_function_with_arity(
            "get_specialization_stats",
            |_| Ok(pyre_object::w_dict_new()),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_intrinsic1_descs",
        crate::make_builtin_function_with_arity(
            "get_intrinsic1_descs",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_intrinsic2_descs",
        crate::make_builtin_function_with_arity(
            "get_intrinsic2_descs",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_opname",
        crate::make_builtin_function_with_arity(
            "get_opname",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_str_new("<0>"));
                }
                let code = unsafe { pyre_object::w_int_get_value(args[0]) };
                Ok(pyre_object::w_str_new(&format!("<{code}>")))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_nb_ops",
        crate::make_builtin_function_with_arity(
            "get_nb_ops",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_special_method_names",
        crate::make_builtin_function_with_arity(
            "get_special_method_names",
            |_| {
                Ok(pyre_object::w_list_new(vec![
                    pyre_object::w_str_new("__enter__"),
                    pyre_object::w_str_new("__exit__"),
                    pyre_object::w_str_new("__aenter__"),
                    pyre_object::w_str_new("__aexit__"),
                ]))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_executor_count",
        crate::make_builtin_function_with_arity(
            "get_executor_count",
            |_| Ok(pyre_object::w_int_new(0)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_hot_code",
        crate::make_builtin_function_with_arity(
            "get_hot_code",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
}

/// importlib stub — PyPy: pypy/module/importlib/
/// Avoid loading the real importlib.__init__ since it drags in
/// _bootstrap and _bootstrap_external.
fn init_importlib_pkg(ns: &mut DictStorage) {
    // importlib.import_module(name, package=None) — return an imported
    // module by name. PyPy: Lib/importlib/__init__.py import_module →
    // _bootstrap._gcd_import. We defer to the interpreter's importhook
    // since it handles both builtins and source modules.
    crate::dict_storage_store(
        ns,
        "import_module",
        crate::make_builtin_function("import_module", |args| {
            let name = args.first().copied().unwrap_or(pyre_object::w_none());
            unsafe {
                if !pyre_object::is_str(name) {
                    return Err(crate::PyError::type_error(
                        "import_module: name must be str",
                    ));
                }
                let name_str = pyre_object::w_str_get_value(name).to_string();
                crate::importing::importhook(
                    &name_str,
                    pyre_object::w_none(),
                    pyre_object::w_list_new(vec![pyre_object::w_str_new("*")]),
                    0,
                    std::ptr::null(),
                )
            }
        }),
    );
    crate::dict_storage_store(
        ns,
        "invalidate_caches",
        crate::make_builtin_function_with_arity(
            "invalidate_caches",
            |_| Ok(pyre_object::w_none()),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "reload",
        crate::make_builtin_function_with_arity(
            "reload",
            |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        ),
    );
    // Mark as a package so dotted imports treat it as such.
    crate::dict_storage_store(ns, "__path__", pyre_object::w_list_new(vec![]));
}

/// importlib.util stub — minimal subset.
fn init_importlib_util(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "spec_from_file_location",
        crate::make_builtin_function_with_arity(
            "spec_from_file_location",
            |_| Ok(pyre_object::w_none()),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "module_from_spec",
        crate::make_builtin_function_with_arity(
            "module_from_spec",
            |_| Ok(pyre_object::w_none()),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "find_spec",
        crate::make_builtin_function_with_arity("find_spec", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "resolve_name",
        crate::make_builtin_function("resolve_name", |args| {
            Ok(args.first().copied().unwrap_or(pyre_object::w_str_new("")))
        }),
    );
    crate::dict_storage_store(ns, "MAGIC_NUMBER", pyre_object::w_int_new(0));
}

/// importlib.abc stub — abstract base classes.
fn init_importlib_abc(ns: &mut DictStorage) {
    for name in [
        "Loader",
        "Finder",
        "MetaPathFinder",
        "PathEntryFinder",
        "ResourceLoader",
        "InspectLoader",
        "ExecutionLoader",
        "FileLoader",
        "SourceLoader",
    ] {
        crate::dict_storage_store(ns, name, crate::typedef::w_object());
    }
}

/// importlib.machinery stub — provides the names inspect.py references.
/// PyPy ships the real importlib; we shortcut it with a stub so pyre does
/// not have to execute _bootstrap_external.
fn init_importlib_machinery(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "SOURCE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".py")]),
    );
    crate::dict_storage_store(
        ns,
        "BYTECODE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".pyc")]),
    );
    crate::dict_storage_store(
        ns,
        "EXTENSION_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".so")]),
    );
    crate::dict_storage_store(
        ns,
        "DEBUG_BYTECODE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".pyc")]),
    );
    crate::dict_storage_store(
        ns,
        "OPTIMIZED_BYTECODE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".pyc")]),
    );
    crate::dict_storage_store(
        ns,
        "all_suffixes",
        crate::make_builtin_function_with_arity(
            "all_suffixes",
            |_| {
                Ok(pyre_object::w_list_new(vec![
                    pyre_object::w_str_new(".py"),
                    pyre_object::w_str_new(".pyc"),
                    pyre_object::w_str_new(".so"),
                ]))
            },
            0,
        ),
    );
    crate::dict_storage_store(ns, "ModuleSpec", crate::typedef::w_object());
    crate::dict_storage_store(ns, "BuiltinImporter", crate::typedef::w_object());
    crate::dict_storage_store(ns, "FrozenImporter", crate::typedef::w_object());
    crate::dict_storage_store(ns, "PathFinder", crate::typedef::w_object());
    crate::dict_storage_store(ns, "FileFinder", crate::typedef::w_object());
    crate::dict_storage_store(ns, "SourceFileLoader", crate::typedef::w_object());
    crate::dict_storage_store(ns, "SourcelessFileLoader", crate::typedef::w_object());
    crate::dict_storage_store(ns, "ExtensionFileLoader", crate::typedef::w_object());
    crate::dict_storage_store(ns, "AppleFrameworkLoader", crate::typedef::w_object());
    crate::dict_storage_store(ns, "NamespaceLoader", crate::typedef::w_object());
    crate::dict_storage_store(ns, "WindowsRegistryFinder", crate::typedef::w_object());
}

/// _imp stub — PyPy: pypy/module/imp/
///
/// Minimal subset required by importlib._bootstrap to decide which loader
/// handles a name. We report every name we know about as a builtin so
/// pyre's own registrations remain authoritative.
fn init_imp(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "is_builtin",
        crate::make_builtin_function_with_arity(
            "is_builtin",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_int_new(0));
                }
                let name = unsafe {
                    if pyre_object::is_str(args[0]) {
                        pyre_object::w_str_get_value(args[0])
                    } else {
                        return Ok(pyre_object::w_int_new(0));
                    }
                };
                let is_builtin = BUILTIN_MODULES.with(|m| m.borrow().contains_key(name));
                Ok(pyre_object::w_int_new(if is_builtin { 1 } else { 0 }))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "is_frozen",
        crate::make_builtin_function_with_arity(
            "is_frozen",
            |_| Ok(pyre_object::w_bool_from(false)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "is_frozen_package",
        crate::make_builtin_function_with_arity(
            "is_frozen_package",
            |_| Ok(pyre_object::w_bool_from(false)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_frozen_object",
        crate::make_builtin_function_with_arity(
            "get_frozen_object",
            |_| Ok(pyre_object::w_none()),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "create_builtin",
        crate::make_builtin_function_with_arity(
            "create_builtin",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_none());
                }
                Ok(args[0])
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "exec_builtin",
        crate::make_builtin_function_with_arity(
            "exec_builtin",
            |_| Ok(pyre_object::w_int_new(0)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "exec_dynamic",
        crate::make_builtin_function_with_arity(
            "exec_dynamic",
            |_| Ok(pyre_object::w_int_new(0)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "acquire_lock",
        crate::make_builtin_function_with_arity("acquire_lock", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "release_lock",
        crate::make_builtin_function_with_arity("release_lock", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "lock_held",
        crate::make_builtin_function_with_arity(
            "lock_held",
            |_| Ok(pyre_object::w_bool_from(false)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_fix_co_filename",
        crate::make_builtin_function_with_arity(
            "_fix_co_filename",
            |_| Ok(pyre_object::w_none()),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "extension_suffixes",
        crate::make_builtin_function_with_arity(
            "extension_suffixes",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "source_hash",
        crate::make_builtin_function_with_arity(
            "source_hash",
            |_| Ok(pyre_object::w_int_new(0)),
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "check_hash_based_pycs",
        pyre_object::w_str_new("default"),
    );
    crate::dict_storage_store(ns, "pyc_magic_number_token", pyre_object::w_int_new(3495));
}

/// _ast stub — PyPy: pypy/module/_ast/
///
/// Exposes the AST node type hierarchy as plain type stubs. Our stubs are
/// enough to satisfy `from _ast import *` in `ast.py` and class body
/// references like `class slice(AST)`. Actual AST construction is not
/// supported because pyre uses RustPython's compiler.
fn init_ast(ns: &mut DictStorage) {
    let ast_names: &[&str] = &[
        "AST",
        "mod",
        "Module",
        "Interactive",
        "Expression",
        "FunctionType",
        "stmt",
        "FunctionDef",
        "AsyncFunctionDef",
        "ClassDef",
        "Return",
        "Delete",
        "Assign",
        "TypeAlias",
        "AugAssign",
        "AnnAssign",
        "For",
        "AsyncFor",
        "While",
        "If",
        "With",
        "AsyncWith",
        "Match",
        "Raise",
        "Try",
        "TryStar",
        "Assert",
        "Import",
        "ImportFrom",
        "Global",
        "Nonlocal",
        "Expr",
        "Pass",
        "Break",
        "Continue",
        "expr",
        "BoolOp",
        "NamedExpr",
        "BinOp",
        "UnaryOp",
        "Lambda",
        "IfExp",
        "Dict",
        "Set",
        "ListComp",
        "SetComp",
        "DictComp",
        "GeneratorExp",
        "Await",
        "Yield",
        "YieldFrom",
        "Compare",
        "Call",
        "FormattedValue",
        "JoinedStr",
        "Constant",
        "Attribute",
        "Subscript",
        "Starred",
        "Name",
        "List",
        "Tuple",
        "Slice",
        "expr_context",
        "Load",
        "Store",
        "Del",
        "boolop",
        "And",
        "Or",
        "operator",
        "Add",
        "Sub",
        "Mult",
        "MatMult",
        "Div",
        "Mod",
        "Pow",
        "LShift",
        "RShift",
        "BitOr",
        "BitXor",
        "BitAnd",
        "FloorDiv",
        "unaryop",
        "Invert",
        "Not",
        "UAdd",
        "USub",
        "cmpop",
        "Eq",
        "NotEq",
        "Lt",
        "LtE",
        "Gt",
        "GtE",
        "Is",
        "IsNot",
        "In",
        "NotIn",
        "comprehension",
        "excepthandler",
        "ExceptHandler",
        "arguments",
        "arg",
        "keyword",
        "alias",
        "withitem",
        "match_case",
        "pattern",
        "MatchValue",
        "MatchSingleton",
        "MatchSequence",
        "MatchMapping",
        "MatchClass",
        "MatchStar",
        "MatchAs",
        "MatchOr",
        "type_ignore",
        "TypeIgnore",
        "type_param",
        "TypeVar",
        "ParamSpec",
        "TypeVarTuple",
        // Flags used by ast.parse()
        "PyCF_ONLY_AST",
        "PyCF_OPTIMIZED_AST",
        "PyCF_TYPE_COMMENTS",
        "PyCF_ALLOW_TOP_LEVEL_AWAIT",
    ];
    for name in ast_names {
        if name.starts_with("PyCF") {
            crate::dict_storage_store(ns, name, pyre_object::w_int_new(0));
        } else {
            crate::dict_storage_store(ns, name, crate::typedef::make_builtin_type(name, |_| {}));
        }
    }
}

/// errno stub — PyPy: pypy/module/errno/
fn init_errno(ns: &mut DictStorage) {
    for (name, value) in [
        ("EPERM", 1),
        ("ENOENT", 2),
        ("ESRCH", 3),
        ("EINTR", 4),
        ("EIO", 5),
        ("ENXIO", 6),
        ("E2BIG", 7),
        ("ENOEXEC", 8),
        ("EBADF", 9),
        ("ECHILD", 10),
        ("EAGAIN", 35),
        ("EWOULDBLOCK", 35),
        ("ENOMEM", 12),
        ("EACCES", 13),
        ("EFAULT", 14),
        ("ENOTBLK", 15),
        ("EBUSY", 16),
        ("EEXIST", 17),
        ("EXDEV", 18),
        ("ENODEV", 19),
        ("ENOTDIR", 20),
        ("EISDIR", 21),
        ("EINVAL", 22),
        ("ENFILE", 23),
        ("EMFILE", 24),
        ("ENOTTY", 25),
        ("ETXTBSY", 26),
        ("EFBIG", 27),
        ("ENOSPC", 28),
        ("ESPIPE", 29),
        ("EROFS", 30),
        ("EMLINK", 31),
        ("EPIPE", 32),
        ("EDOM", 33),
        ("ERANGE", 34),
        ("EDEADLK", 11),
        ("ENAMETOOLONG", 63),
        ("ENOLCK", 77),
        ("ENOSYS", 78),
        ("ENOTEMPTY", 66),
        ("ELOOP", 62),
        ("ENOMSG", 91),
        ("EIDRM", 90),
        ("EBADMSG", 94),
        ("EMULTIHOP", 95),
        ("ENODATA", 96),
        ("ENOLINK", 97),
        ("ENOSR", 98),
        ("ENOSTR", 99),
        ("EOVERFLOW", 84),
        ("EPROTO", 100),
        ("ETIME", 101),
        ("EDESTADDRREQ", 39),
        ("EAFNOSUPPORT", 47),
        ("EALREADY", 37),
        ("EDQUOT", 69),
    ] {
        crate::dict_storage_store(ns, name, pyre_object::w_int_new(value));
    }
    crate::dict_storage_store(ns, "errorcode", pyre_object::w_dict_new());
}

/// _codecs stub — PyPy: pypy/module/_codecs/
///
/// Provides lookup_error/register_error and encode/decode no-op stubs so
/// codecs.py module init runs to completion.
fn init_codecs(ns: &mut DictStorage) {
    // lookup_error(name) — returns an error handler for the given error
    // strategy. Pyre returns a pass-through lambda that never fires because
    // we don't encounter encoding errors in the pure-Python stdlib paths
    // we exercise so far.
    crate::dict_storage_store(
        ns,
        "lookup_error",
        crate::make_builtin_function_with_arity(
            "lookup_error",
            |_| {
                Ok(crate::make_builtin_function_with_arity(
                    "error_handler",
                    |args| {
                        Ok(if args.is_empty() {
                            pyre_object::w_none()
                        } else {
                            args[0]
                        })
                    },
                    1,
                ))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "register_error",
        crate::make_builtin_function_with_arity("register_error", |_| Ok(pyre_object::w_none()), 2),
    );
    crate::dict_storage_store(
        ns,
        "register",
        crate::make_builtin_function_with_arity("register", |_| Ok(pyre_object::w_none()), 1),
    );
    crate::dict_storage_store(
        ns,
        "lookup",
        crate::make_builtin_function_with_arity("lookup", |_| Ok(pyre_object::w_none()), 1),
    );
    // encode/decode — return input unchanged. Matches PyPy _codecs.encode
    // when the codec is the identity.
    let identity = crate::make_builtin_function_with_arity(
        "identity",
        |args| {
            Ok(if args.is_empty() {
                pyre_object::w_none()
            } else {
                args[0]
            })
        },
        1,
    );
    crate::dict_storage_store(ns, "encode", identity);
    crate::dict_storage_store(ns, "decode", identity);
    crate::dict_storage_store(ns, "_forget_codec", identity);
    crate::dict_storage_store(
        ns,
        "charmap_build",
        crate::make_builtin_function_with_arity(
            "charmap_build",
            |_| Ok(pyre_object::w_dict_new()),
            1,
        ),
    );
}

/// copyreg stub — PyPy: pypy/module/copyreg/
fn init_copyreg(ns: &mut DictStorage) {
    // copyreg.pickle(type, reduce_func, constructor=None) — register a
    // pickle reducer. Stub: ignore (pyre doesn't support pickle).
    crate::dict_storage_store(
        ns,
        "pickle",
        crate::make_builtin_function_with_arity("pickle", |_| Ok(pyre_object::w_none()), 3),
    );
    crate::dict_storage_store(ns, "dispatch_table", pyre_object::w_dict_new());
}

/// Try to load a builtin module by name.
///
/// PyPy equivalent: find_module() → C_BUILTIN path →
/// getbuiltinmodule() → Module.__init__ + startup()
fn load_builtin_module(name: &str) -> Option<PyObjectRef> {
    let init_fn = BUILTIN_MODULES.with(|m| m.borrow().get(name).copied())?;

    let mut namespace = Box::new(DictStorage::new());
    namespace.fix_ptr();

    // Set __name__ (PyPy: Module.__init__ sets __name__)
    let name_obj = pyre_object::w_str_new(name);
    dict_storage_store(&mut namespace, "__name__", name_obj);

    // Run module-specific initializer (PyPy: interpleveldefs)
    init_fn(&mut namespace);

    let ns_ptr = Box::into_raw(namespace);
    // PyPy `module.py:77 Module.getdict()` parity: reuse the canonical
    // W_DictObject paired with this storage (`dict_storage_to_dict`
    // lazy mirror_target registration); avoid allocating a sibling
    // W_DictObject through `w_module_new` that would shadow the
    // canonical identity.
    let canonical = crate::baseobjspace::dict_storage_to_dict(ns_ptr);
    let module = pyre_object::w_module_new_aliasing_dict(name, ns_ptr as *mut u8, canonical);
    Some(module)
}

/// Initialize sys.path with the directory containing the main script.
///
/// PyPy equivalent: sys.path is populated at startup with the script
/// directory, then PYTHONPATH entries, then the stdlib.
#[cfg(feature = "host_env")]
pub fn init_sys_path(script_dir: &Path) {
    // Register builtin modules (PyPy: make_builtins / setup_builtin_modules)
    install_builtin_modules();

    SYS_PATH.with(|p| {
        let mut path = p.borrow_mut();
        path.clear();
        // Script directory first (PyPy: first entry in sys.path)
        path.push(script_dir.to_path_buf());
        // Current working directory as fallback
        if let Ok(cwd) = std::env::current_dir() {
            if cwd != script_dir {
                path.push(cwd);
            }
        }
        // CPython stdlib path is detected lazily on first stdlib import
        // to avoid spawning python3 subprocess on every startup.
        // See find_module() → ensure_stdlib_path().
    });
}

/// Detect CPython stdlib path via `python3 -c "import sysconfig; ..."`.
///
/// PyPy equivalent: initpath.py scans for lib-python/X.Y at startup.
#[cfg(feature = "host_env")]
fn detect_stdlib_path() -> Option<PathBuf> {
    // Try PYRE_STDLIB env var first
    if let Ok(p) = std::env::var("PYRE_STDLIB") {
        let path = PathBuf::from(p);
        if path.is_dir() {
            return Some(path);
        }
    }
    // Auto-detect via python3
    let output = std::process::Command::new("python3")
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['stdlib'])",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8(output.stdout).ok()?;
    let path = PathBuf::from(s.trim());
    if path.is_dir() { Some(path) } else { None }
}

/// Add a directory to sys.path.
#[cfg(feature = "host_env")]
pub fn add_sys_path(dir: &Path) {
    SYS_PATH.with(|p| {
        let mut path = p.borrow_mut();
        let pb = dir.to_path_buf();
        if !path.contains(&pb) {
            path.push(pb);
        }
    });
}

// ── check_sys_modules ────────────────────────────────────────────────
// PyPy equivalent: importing.py `check_sys_modules(space, w_modulename)`

fn check_sys_modules(name: &str) -> Option<PyObjectRef> {
    // Consult the Python-visible sys.modules dict first so that user code
    // writing `sys.modules['foo'] = mod` is immediately visible to imports.
    // PyPy: importing.py check_sys_modules reads space.sys.get('modules').
    let key = pyre_object::w_str_new(name);
    let dict = SYS_MODULES_DICT.with(|d| d.get());
    if !dict.is_null() {
        if let Some(m) = unsafe { pyre_object::w_dict_lookup(dict, key) } {
            if !m.is_null() && !unsafe { pyre_object::is_none(m) } {
                return Some(m);
            }
        }
    }
    SYS_MODULES.with(|m| m.borrow().get(name).copied())
}

pub fn set_sys_module(name: &str, module: PyObjectRef) {
    SYS_MODULES.with(|m| {
        m.borrow_mut().insert(name.to_string(), module);
    });
    // Keep the Python-visible sys.modules dict in sync.
    SYS_MODULES_DICT.with(|d| {
        let dict = d.get();
        if !dict.is_null() {
            unsafe {
                pyre_object::w_dict_store(dict, pyre_object::w_str_new(name), module);
            }
        }
    });
}

/// Set the Python-visible sys.modules dict reference. Called during sys
/// module initialization so subsequent set_sys_module calls keep it in sync.
/// Also copies all previously cached modules into the dict.
/// Set sys.argv from a list of strings.
/// Must be called after the first `import sys` has run (e.g. after
/// `run_source` compiles the module-level code).
pub fn set_sys_argv(args: &[String]) {
    let items: Vec<pyre_object::PyObjectRef> =
        args.iter().map(|s| pyre_object::w_str_new(s)).collect();
    let argv = pyre_object::w_list_new(items);
    SYS_ARGV_PENDING.with(|p| p.set(argv));
}

thread_local! {
    static SYS_ARGV_PENDING: std::cell::Cell<pyre_object::PyObjectRef> =
        const { std::cell::Cell::new(pyre_object::PY_NULL) };
}

/// Called from sys module init to pick up any pending argv.
pub fn take_pending_sys_argv() -> pyre_object::PyObjectRef {
    SYS_ARGV_PENDING.with(|p| {
        let v = p.get();
        p.set(pyre_object::PY_NULL);
        v
    })
}

pub fn set_sys_modules_dict(dict: PyObjectRef) {
    SYS_MODULES_DICT.with(|d| d.set(dict));
    // Populate with all modules already in the cache.
    SYS_MODULES.with(|m| {
        for (name, &module) in m.borrow().iter() {
            unsafe {
                pyre_object::w_dict_store(dict, pyre_object::w_str_new(name), module);
            }
        }
    });
}

// ── find_module ──────────────────────────────────────────────────────
// PyPy equivalent: importing.py `find_module()`
// Searches sys.path for `<partname>.py` or `<partname>/__init__.py` (package).

#[derive(Debug)]
enum FindInfo {
    /// A .py source file was found.
    #[cfg(feature = "host_env")]
    SourceFile { pathname: PathBuf },
    /// A package directory with __init__.py was found.
    #[cfg(feature = "host_env")]
    Package { dirpath: PathBuf },
    /// A builtin (Rust-implemented) module was found.
    /// PyPy equivalent: C_BUILTIN modtype in find_module()
    Builtin,
}

#[cfg(feature = "host_env")]
fn find_module(partname: &str) -> Option<FindInfo> {
    // Check builtin modules first (PyPy: space.builtin_modules check in find_module)
    let is_builtin = BUILTIN_MODULES.with(|m| m.borrow().contains_key(partname));
    if is_builtin {
        return Some(FindInfo::Builtin);
    }

    // Try sys.path first
    if let Some(info) = find_in_sys_path(partname) {
        return Some(info);
    }

    // Lazy stdlib detection — only on first miss (avoid python3 spawn at startup)
    ensure_stdlib_path();
    return find_in_sys_path(partname);
}

#[cfg(not(feature = "host_env"))]
fn find_module(partname: &str) -> Option<FindInfo> {
    let is_builtin = BUILTIN_MODULES.with(|m| m.borrow().contains_key(partname));
    if is_builtin {
        return Some(FindInfo::Builtin);
    }
    None
}

/// Detect and add CPython stdlib to sys.path (once).
#[cfg(feature = "host_env")]
fn ensure_stdlib_path() {
    thread_local! {
        static DONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    DONE.with(|d| {
        if d.get() {
            return;
        }
        d.set(true);
        if let Some(stdlib) = detect_stdlib_path() {
            add_sys_path(&stdlib);
        }
    });
}

#[cfg(feature = "host_env")]
fn find_in_sys_path(partname: &str) -> Option<FindInfo> {
    SYS_PATH.with(|p| {
        let path = p.borrow();
        for dir in path.iter() {
            // Check for package: <dir>/<partname>/__init__.py
            let pkg_dir = dir.join(partname);
            let init_file = pkg_dir.join("__init__.py");
            if init_file.is_file() {
                return Some(FindInfo::Package { dirpath: pkg_dir });
            }

            // Check for source file: <dir>/<partname>.py
            let source_file = dir.join(format!("{partname}.py"));
            if source_file.is_file() {
                return Some(FindInfo::SourceFile {
                    pathname: source_file,
                });
            }
        }
        None
    })
}

// ── parse_source_module ──────────────────────────────────────────────
// PyPy equivalent: importing.py `parse_source_module(space, pathname, source)`

fn parse_source_module(pathname: &str, source: &str) -> Result<CodeObject, String> {
    compile_source_with_filename(source, Mode::Exec, pathname)
}

// ── exec_code_module ─────────────────────────────────────────────────
// PyPy equivalent: importing.py `exec_code_module(space, w_mod, code_w,
//                                  pathname, cpathname, write_paths=True)`
//
// Mirrors `pypy/module/imp/importing.py:269-300` line-by-line:
//   w_dict = space.getattr(w_mod, '__dict__')                       # ns
//   space.call_method(w_dict, 'setdefault',
//                     '__builtins__', space.builtin)
//   if write_paths:
//       space.setitem(w_dict, '__file__', w_pathname)
//       space.setitem(w_dict, '__cached__', w_cpathname)
//       _fix_up_module(d, name, pathname, cpathname)               # appexec
//   code_w.exec_code(space, w_dict, w_dict)
//
// `pathname` is `None` for callers that do not have a filesystem path
// (REPL `__main__`, builtin module bootstrap), matching PyPy's
// `write_paths=False` shape.  `cpathname` is `None` when no `.pyc` cache
// is available (pyre has no .pyc cache today, so all reachable callers
// pass `None` here — kept as a parameter so the signature mirrors PyPy
// instead of erasing the field).

fn exec_code_module(
    code: CodeObject,
    namespace: *mut DictStorage,
    execution_context: *const PyExecutionContext,
    pathname: Option<&str>,
    cpathname: Option<&str>,
) -> Result<PyObjectRef, crate::PyError> {
    // importing.py:272-274 — setdefault('__builtins__', space.builtin).
    // `fresh_dict_storage` already seeds `__builtins__` for module-shape
    // namespaces; the explicit setdefault here mirrors PyPy's defensive
    // call so callers that hand in a pre-built storage (future
    // `_imp.exec_dynamic`-style entry) still inherit the builtins
    // pointer with no surprises.
    {
        let ns = unsafe { &mut *namespace };
        if crate::dict_storage_get(ns, "__builtins__").is_none() {
            let ctx = unsafe { &*execution_context };
            let w_builtin = ctx.get_builtin();
            if !w_builtin.is_null() {
                crate::dict_storage_store(ns, "__builtins__", w_builtin);
            }
        }
    }
    // importing.py:275-298 write_paths block.  Pyre callers always pass
    // `Some(pathname)` for source-file imports and `None` for the
    // `write_paths=False` shape (REPL, builtin bootstrap).
    if let Some(p) = pathname {
        let ns = unsafe { &mut *namespace };
        // importing.py:284 setitem('__file__', w_pathname).
        let w_pathname = pyre_object::w_str_new(p);
        crate::dict_storage_store(ns, "__file__", w_pathname);
        // importing.py:285 setitem('__cached__', w_cpathname).  PyPy
        // surfaces `space.w_None` when `cpathname is None`, i.e. the
        // import was not satisfied from a `.pyc`.  Pyre has no .pyc
        // path today so reachable callers still hit the None arm.
        let w_cpathname = match cpathname {
            Some(c) => pyre_object::w_str_new(c),
            None => pyre_object::w_none(),
        };
        crate::dict_storage_store(ns, "__cached__", w_cpathname);
        // importing.py:286-298 — `_fix_up_module(d, name, pathname,
        // cpathname)`.  PyPy's `_fix_up_module`
        // (`lib-python/3/importlib/_bootstrap_external.py:1728`) sets
        // `__spec__`/`__loader__`/`__file__`/`__cached__` from the
        // app-level `SourceFileLoader` + `spec_from_file_location`
        // helpers.  Pyre lacks the importlib bootstrap machinery
        // (`SourceFileLoader`, `ModuleSpec`, `spec_from_file_location`
        // are not yet ported), so as a PRE-EXISTING-ADAPTATION we seed
        // `__loader__`/`__spec__` with `None` only when missing —
        // matching PyPy's `if not loader / if not spec` guards
        // (_bootstrap_external.py:1732, 1739).  When the importlib
        // app-level layer lands, the `None` arms will collapse onto the
        // mechanical PyPy port.
        if crate::dict_storage_get(ns, "__loader__").is_none() {
            crate::dict_storage_store(ns, "__loader__", pyre_object::w_none());
        }
        if crate::dict_storage_get(ns, "__spec__").is_none() {
            crate::dict_storage_store(ns, "__spec__", pyre_object::w_none());
        }
    }
    let code_ptr = Box::into_raw(Box::new(code));
    let w_code = crate::w_code_new(code_ptr as *const ());
    // importing.py:300 code_w.exec_code(space, w_dict, w_dict) → eval.py:31-33
    // Code.exec_code → space.createframe(...) + frame.run().  Surface
    // initialize_frame_scopes' freevar/closure mismatch (TypeError /
    // ValueError per pyframe.py:242-253) as PyError so the importer
    // reports it instead of panicking.  Route through run() so the
    // GENERATOR / COROUTINE / ASYNC_GENERATOR dispatch in
    // pyframe.py:268-273 holds for the import path too.
    let mut frame = crate::createframe(w_code as *const (), namespace, execution_context, None)?;
    frame.run()
}

// ── load_source_module ───────────────────────────────────────────────
// PyPy equivalent: importing.py `load_source_module()`
//
// Parse + execute a .py source file, producing a module object.

#[cfg(feature = "host_env")]
fn load_source_module(
    modulename: &str,
    pathname: &Path,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    let source = std::fs::read_to_string(pathname).map_err(|e| {
        crate::PyError::new(
            crate::PyErrorKind::ImportError,
            format!("cannot read '{}': {e}", pathname.display()),
        )
    })?;

    let pathname_str = pathname.to_string_lossy();
    let code = parse_source_module(&pathname_str, &source).map_err(|e| {
        crate::PyError::new(
            crate::PyErrorKind::ImportError,
            format!("cannot compile '{}': {e}", pathname.display()),
        )
    })?;

    // Create a fresh namespace for the module, seeded with builtins.
    // PyPy equivalent: Module.__init__ creates w_dict = space.newdict()
    // then exec_code_module sets __builtins__ and runs code in w_dict.
    let ctx = unsafe { &*execution_context };
    let mut namespace = Box::new(ctx.fresh_dict_storage());
    namespace.fix_ptr();

    // PyPy `interpreter/module.py:Module.__init__` seeds `__name__` on
    // the module's w_dict.  `w_module_new(modulename, ns_ptr)` below
    // does that via `w_dict_setitem_str("__name__", ...)` which the
    // storage proxy mirrors back into `namespace`, so an explicit
    // dict_storage_store here would be redundant.
    //
    // `__file__`/`__cached__` setting moved into `exec_code_module`
    // (`importing.py:284-285`) so the per-module attribute seeding
    // mirrors the PyPy call order.
    //
    // `__package__` is set by PyPy `interp_import._prepare_module`
    // (`pypy/module/imp/interp_import.py`); pyre has no `_prepare_module`
    // yet, so we still seed it here as a PRE-EXISTING-ADAPTATION until
    // the prepare-module path is ported.
    let pkg = if let Some(dot) = modulename.rfind('.') {
        &modulename[..dot]
    } else {
        modulename
    };
    crate::dict_storage_store(&mut namespace, "__package__", pyre_object::w_str_new(pkg));

    let ns_ptr = Box::into_raw(namespace);

    // Create the module object BEFORE execution and register in sys.modules.
    // PyPy: load_source_module → set_sys_modules BEFORE exec_code_module.
    // This prevents infinite recursion on circular imports.
    //
    // Reuse the canonical W_DictObject paired with the storage
    // (`dict_storage_to_dict` lazy mirror_target registration) so the
    // module's `w_dict` is the same identity that `function.__globals__`
    // and `globals()` will return for code executing in this module.
    let canonical = crate::baseobjspace::dict_storage_to_dict(ns_ptr);
    let module = pyre_object::w_module_new_aliasing_dict(modulename, ns_ptr as *mut u8, canonical);
    set_sys_module(modulename, module);

    // PyPy `importing.py:300` passes `pathname`/`cpathname` to
    // `exec_code_module`; pyre has no .pyc cache today so cpathname is
    // always None, matching the PyPy `cpathname is None` arm at line
    // 282-283.
    exec_code_module(code, ns_ptr, execution_context, Some(&pathname_str), None)?;

    // Module-level code may have rewritten `sys.modules[name]` (the
    // `decimal` → `_pydecimal` pattern, or PyPy's `_cffi_backend` style
    // late rewiring). Honour that — PyPy: interp_import.importhook
    // reads sys.modules again after exec_code_module via importcache.
    if let Some(replaced) = check_sys_modules(modulename) {
        if !std::ptr::eq(replaced, module) {
            return Ok(replaced);
        }
    }

    Ok(module)
}

// ── load_package ─────────────────────────────────────────────────────
// PyPy equivalent: load_module with PKG_DIRECTORY modtype

#[cfg(feature = "host_env")]
fn load_package(
    modulename: &str,
    dirpath: &Path,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // Add package directory to sys.path BEFORE executing __init__.py,
    // so that relative sub-imports within the package can find siblings.
    // PyPy: sets __path__ on module before exec.
    add_sys_path(dirpath);

    let init_path = dirpath.join("__init__.py");
    let module = load_source_module(modulename, &init_path, execution_context)?;

    // Set __path__ and __package__ on the module namespace via
    // `module.w_dict` so storage-backed and dict-subclass-backed Modules
    // both observe the writes (`pypy/module/__builtin__/moduledef.py:102-103
    // Module(space, None, w_builtin)`).  When the dict is storage-backed
    // the proxy store hook propagates the entry into the underlying
    // DictStorage; when it's a subclass instance the write lands in the
    // entries Vec where the subclass's `__init__` placed any seeded keys.
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
    let path_str = pyre_object::w_str_new(&dirpath.to_string_lossy());
    let path_list = pyre_object::w_list_new(vec![path_str]);
    unsafe {
        if !w_dict.is_null() && pyre_object::is_dict(w_dict) {
            pyre_object::dictobject::w_dict_setitem_str(w_dict, "__path__", path_list);
            pyre_object::dictobject::w_dict_setitem_str(
                w_dict,
                "__package__",
                pyre_object::w_str_new(modulename),
            );
        }
    }

    Ok(module)
}

// ── load_part ────────────────────────────────────────────────────────
// PyPy equivalent: importing.py `load_part()`

fn load_part(
    modulename: &str,
    partname: &str,
    execution_context: *const PyExecutionContext,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    // Check sys.modules cache first
    if let Some(cached) = check_sys_modules(modulename) {
        return Ok(Some(cached));
    }

    // Try a full-name builtin match first so dotted stubs like
    // `importlib.machinery` can override the filesystem search.
    // PyPy: interp_import.importhook consults sys.builtin_module_names by
    // the fully-qualified name.
    let full_is_builtin = BUILTIN_MODULES.with(|m| m.borrow().contains_key(modulename));
    if full_is_builtin {
        let m = load_builtin_module(modulename).ok_or_else(|| crate::PyError {
            kind: crate::PyErrorKind::ImportError,
            message: format!("builtin module '{modulename}' failed to initialize"),
            exc_object: std::ptr::null_mut(),
        })?;
        set_sys_module(modulename, m);
        return Ok(Some(m));
    }

    // Find the module on disk
    let find_info = find_module(partname);
    let Some(info) = find_info else {
        return Ok(None);
    };

    let module = match info {
        #[cfg(feature = "host_env")]
        FindInfo::SourceFile { pathname } => {
            match load_source_module(modulename, &pathname, execution_context) {
                Ok(m) => m,
                Err(e) => {
                    return Err(e);
                }
            }
        }
        #[cfg(feature = "host_env")]
        FindInfo::Package { dirpath } => load_package(modulename, &dirpath, execution_context)?,
        FindInfo::Builtin => {
            // PyPy: getbuiltinmodule() path
            let m = load_builtin_module(partname).ok_or_else(|| crate::PyError {
                kind: crate::PyErrorKind::ImportError,
                message: format!("builtin module '{modulename}' failed to initialize"),
                exc_object: std::ptr::null_mut(),
            })?;
            // Store builtin modules in cache immediately
            set_sys_module(modulename, m);
            m
        }
    };

    Ok(Some(module))
}

// ── _absolute_import ─────────────────────────────────────────────────
// PyPy equivalent: importing.py `_absolute_import()`

fn absolute_import(
    modulename: &str,
    w_fromlist: PyObjectRef,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    let parts: Vec<&str> = modulename.split('.').collect();
    let mut first: Option<PyObjectRef> = None;
    let mut prefix = Vec::new();

    for (level, &part) in parts.iter().enumerate() {
        prefix.push(part);
        let full_name = prefix.join(".");
        let w_mod = load_part(&full_name, part, execution_context)?;
        let Some(module) = w_mod else {
            return Err(crate::PyError::new(
                crate::PyErrorKind::ImportError,
                format!("No module named '{modulename}'"),
            ));
        };
        if level == 0 {
            first = Some(module);
        }
    }

    // PyPy: if w_fromlist is not None, return the leaf module.
    // Otherwise, return the first (top-level) module.
    if !w_fromlist.is_null() && !unsafe { is_none(w_fromlist) } {
        // `from X.Y import Z` → return the leaf module (Y)
        if let Some(cached) = check_sys_modules(modulename) {
            return Ok(cached);
        }
    }

    // `import X.Y` → return the top-level module (X)
    first.ok_or_else(|| {
        crate::PyError::new(
            crate::PyErrorKind::ImportError,
            format!("No module named '{modulename}'"),
        )
    })
}

// ── importhook ───────────────────────────────────────────────────────
// PyPy equivalent: importing.py `importhook()`
//
// Main entry point called by the IMPORT_NAME opcode.
// Stack: [level, fromlist] → [module]

pub fn importhook(
    name: &str,
    w_globals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    if name.is_empty() && level < 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            "Empty module name",
        ));
    }

    if level > 0 {
        return relative_import(name, w_globals, w_fromlist, level, execution_context);
    }

    absolute_import(name, w_fromlist, execution_context)
}

/// Relative import: `from .foo import bar` (level=1), `from ..foo import bar` (level=2).
///
/// PyPy: importing.py `_relative_import()`.
/// Resolves the package base from __package__ or __name__ in w_globals,
/// strips `level - 1` trailing components, then does absolute import.
fn relative_import(
    name: &str,
    w_globals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // Get the package name from the calling module's globals.
    // PyPy: pkgname = globals.get('__package__') or globals.get('__name__')
    let package = resolve_package_name(w_globals);
    let package = package.ok_or_else(|| crate::PyError {
        kind: crate::PyErrorKind::ImportError,
        message: "attempted relative import with no known parent package".to_string(),
        exc_object: std::ptr::null_mut(),
    })?;

    // Strip (level - 1) trailing components from package
    // PyPy: for dotted name "a.b.c" with level=2, strip "c" → "a.b", then strip "b" → "a"
    let mut parts: Vec<&str> = package.split('.').collect();
    let strips = (level - 1) as usize;
    if strips >= parts.len() {
        return Err(crate::PyError {
            kind: crate::PyErrorKind::ImportError,
            message: format!(
                "attempted relative import beyond top-level package (package='{package}', level={level})"
            ),
            exc_object: std::ptr::null_mut(),
        });
    }
    for _ in 0..strips {
        parts.pop();
    }
    let base = parts.join(".");

    // Build the fully-qualified module name
    let fqn = if name.is_empty() {
        base.clone()
    } else {
        format!("{base}.{name}")
    };

    absolute_import(&fqn, w_fromlist, execution_context)
}

/// Extract the package name from the calling module's globals namespace.
///
/// PyPy: importing.py — checks __package__ first, falls back to __name__,
/// strips the last component if __name__ has dots (module in a package).
fn resolve_package_name(w_globals: PyObjectRef) -> Option<String> {
    if w_globals.is_null() {
        return None;
    }
    let ns = w_globals as *const crate::DictStorage;
    let ns = unsafe { &*ns };

    // Try __package__ first (PyPy: space.finditem_str(w_globals, '__package__'))
    if let Some(&pkg) = ns.get("__package__") {
        if !pkg.is_null() && unsafe { pyre_object::is_str(pkg) } {
            let s = unsafe { pyre_object::w_str_get_value(pkg) };
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }

    // Fallback: __name__ (for modules inside packages)
    if let Some(&name_obj) = ns.get("__name__") {
        if !name_obj.is_null() && unsafe { pyre_object::is_str(name_obj) } {
            let name = unsafe { pyre_object::w_str_get_value(name_obj) };
            // If the module has a __path__, it's a package — use __name__ as-is
            if ns.get("__path__").is_some() {
                return Some(name.to_string());
            }
            // Otherwise strip the last component (module name within package)
            if let Some(dot) = name.rfind('.') {
                return Some(name[..dot].to_string());
            }
        }
    }

    None
}

// ── import_from ──────────────────────────────────────────────────────
// PyPy equivalent: pyopcode.py `IMPORT_FROM`
//
// Get an attribute from the module on TOS. Like `space.getattr(w_module, w_name)`.

pub fn import_from(
    module: PyObjectRef,
    name: &str,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // First try the module's namespace dict (PyPy: space.getattr → w_dict lookup).
    // Routed through `w_module.w_dict` so dict-subclass-backed Modules
    // (`pypy/module/__builtin__/moduledef.py:102-103`) honour their
    // `__getitem__` overrides via the same lookup path.
    if unsafe { is_module(module) } {
        let w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
        if !w_dict.is_null() && unsafe { pyre_object::is_dict(w_dict) } {
            if let Some(value) = unsafe { pyre_object::w_dict_getitem_str(w_dict, name) } {
                return Ok(value);
            }
        }
    }

    // Fallback: try getattr (for non-module objects or attrs set via setattr)
    if let Ok(value) = crate::baseobjspace::getattr(module, name) {
        return Ok(value);
    }

    // PyPy: pyopcode.py _import_from — try importing as a submodule.
    // Build fullname = module.__name__ + "." + name and import it.
    // Same `w_dict` routing as the first lookup so dict-subclass-backed
    // Modules' submodule fallback honours overridden `__getitem__`.
    if unsafe { is_module(module) } {
        let w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
        if !w_dict.is_null() && unsafe { pyre_object::is_dict(w_dict) } {
            if let Some(modname_obj) =
                unsafe { pyre_object::w_dict_getitem_str(w_dict, "__name__") }
            {
                if !modname_obj.is_null() && unsafe { pyre_object::is_str(modname_obj) } {
                    let modname = unsafe { pyre_object::w_str_get_value(modname_obj) };
                    let fullname = format!("{modname}.{name}");
                    if importhook(
                        &fullname,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        0,
                        execution_context,
                    )
                    .is_ok()
                    {
                        // importhook returns the top-level module when
                        // fromlist is empty. Retrieve the actual leaf
                        // module from sys.modules.
                        if let Some(submod) = check_sys_modules(&fullname) {
                            unsafe {
                                pyre_object::dictobject::w_dict_setitem_str(w_dict, name, submod);
                            }
                            return Ok(submod);
                        }
                    }
                }
            }
        }
    }

    Err(crate::PyError::new(
        crate::PyErrorKind::ImportError,
        format!("cannot import name '{name}'"),
    ))
}

// ── import_all_from ──────────────────────────────────────────────────
// PyPy equivalent: pyopcode.py:2221-2258 `import_all_from(module,
// into_locals)` (applevel function called by IMPORT_STAR).

fn type_name_for_err(w_obj: PyObjectRef) -> String {
    unsafe {
        match crate::typedef::r#type(w_obj) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => (*(*w_obj).ob_type).name.to_string(),
        }
    }
}

/// pypy/interpreter/pyopcode.py:2221-2258 `import_all_from` — applevel
/// driver.  Iterates `for name in all:` lazily via `space.iter` /
/// `space.next`, applies the per-name str check + leading-underscore
/// filter, and invokes `write` once per accepted name.  Used by the
/// `*mut DictStorage` and generic-mapping wrappers below.
///
/// ```python
/// try:
///     all = module.__all__
/// except AttributeError:
///     try:
///         dict = module.__dict__
///     except AttributeError:
///         raise ImportError("from-import-* object has no __dict__ "
///                           "and no __all__")
///     all = dict.keys()
///     skip_leading_underscores = True
/// else:
///     skip_leading_underscores = False
///
/// module_name = module.__name__
/// if not isinstance(module_name, str):
///     raise TypeError("module __name__ must be a string, not %s",
///                     type(module_name).__name__)
///
/// for name in all:
///     if not isinstance(name, str):
///         ...  # raise TypeError ("Item in <m>.__all__ ..." or
///              #                  "Key in <m>.__dict__ ...")
///     if skip_leading_underscores and name and name[0] == '_':
///         continue
///     into_locals[name] = getattr(module, name)
/// ```
fn import_all_from_each<F>(module: PyObjectRef, mut write: F) -> Result<(), crate::PyError>
where
    F: FnMut(&str, PyObjectRef) -> Result<(), crate::PyError>,
{
    let (w_iterable, skip_leading_underscores) =
        match crate::baseobjspace::getattr(module, "__all__") {
            Ok(w_all) => (w_all, false),
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
                // pyopcode.py:2225-2230 — `dict = module.__dict__; all = dict.keys()`.
                // `space.getattr(module, '__dict__')` so any object exposing
                // `__dict__` (Module, class, instance with `__dict__`,
                // bytes-keyed proxies, ...) participates.
                match crate::baseobjspace::getattr(module, "__dict__") {
                    Ok(w_dict) => {
                        let w_keys_method = crate::baseobjspace::getattr(w_dict, "keys")?;
                        // pyopcode.py:2230 `all = dict.keys()` — pyre's
                        // `call_function` stashes errors as PY_NULL; use
                        // `call_and_check` so a misbehaving `keys()` (or
                        // `__getattr__`-installed override) raises here
                        // rather than handing a bogus iterable to
                        // `space.iter` below.
                        let w_keys = crate::builtins::call_and_check(w_keys_method, &[])?;
                        (w_keys, true)
                    }
                    Err(e2) if e2.kind == crate::PyErrorKind::AttributeError => {
                        return Err(crate::PyError::new(
                            crate::PyErrorKind::ImportError,
                            "from-import-* object has no __dict__ and no __all__".to_string(),
                        ));
                    }
                    Err(e2) => return Err(e2),
                }
            }
            Err(e) => return Err(e),
        };

    // pyopcode.py:2235-2237 — `module_name = module.__name__` with str check.
    let module_name_w = crate::baseobjspace::getattr(module, "__name__")?;
    if !unsafe { is_str(module_name_w) } {
        return Err(crate::PyError::type_error(format!(
            "module __name__ must be a string, not {}",
            type_name_for_err(module_name_w),
        )));
    }
    let module_name = unsafe { pyre_object::w_str_get_value(module_name_w) }.to_string();

    // pyopcode.py:2239 — `for name in all:` lazy iteration.
    let w_iter = crate::baseobjspace::iter(w_iterable)?;
    loop {
        let w_name = match crate::baseobjspace::next(w_iter) {
            Ok(v) => v,
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        };
        // pyopcode.py:2240-2255 — per-name str check.
        if !unsafe { is_str(w_name) } {
            let (container, accessor) = if skip_leading_underscores {
                ("__dict__", "Key")
            } else {
                ("__all__", "Item")
            };
            return Err(crate::PyError::type_error(format!(
                "{accessor} in {module_name}.{container} must be str, not {}",
                type_name_for_err(w_name),
            )));
        }
        let name = unsafe { pyre_object::w_str_get_value(w_name) }.to_string();
        // pyopcode.py:2256-2257 — leading-underscore filter (only for
        // the `__dict__.keys()` fallback).
        if skip_leading_underscores && name.starts_with('_') {
            continue;
        }
        // pyopcode.py:2258 — `into_locals[name] = getattr(module, name)`.
        let value = crate::baseobjspace::getattr(module, &name)?;
        write(&name, value)?;
    }
    Ok(())
}

/// pypy/interpreter/pyopcode.py:2221-2258 `import_all_from` —
/// `*mut DictStorage` (dict locals fast path) target variant.
pub fn import_all_from(
    module: PyObjectRef,
    into_namespace: *mut DictStorage,
) -> Result<(), crate::PyError> {
    let dst_ns = unsafe { &mut *into_namespace };
    import_all_from_each(module, |name, value| {
        dict_storage_store(dst_ns, name, value);
        Ok(())
    })
}

/// pypy/interpreter/pyopcode.py:2221-2258 `import_all_from` — generic
/// mapping (`PyObjectRef`) target variant.  Errors from `__setitem__`
/// propagate (CPython behaviour: a misbehaving mapping surfaces its
/// TypeError / KeyError to the caller).
pub fn import_all_from_w(
    module: PyObjectRef,
    into_locals: PyObjectRef,
) -> Result<(), crate::PyError> {
    import_all_from_each(module, |name, value| {
        crate::baseobjspace::setitem(into_locals, unsafe { pyre_object::w_str_new(name) }, value)?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sys_modules_cache() {
        let sentinel = w_none();
        set_sys_module("test_cached", sentinel);
        let cached = check_sys_modules("test_cached");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), sentinel);
    }

    #[test]
    fn test_find_module_nonexistent() {
        // Should not find a module that doesn't exist
        let result = find_module("__nonexistent_pyre_test_module__");
        assert!(result.is_none());
    }
}
