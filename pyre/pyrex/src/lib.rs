//! pyre — A Rust meta-tracing JIT Python interpreter.

use std::path::Path;
use std::rc::Rc;

use lexopt::Arg::*;
use lexopt::ValueExt;

use pyre_interpreter::call;
use pyre_interpreter::importing;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::*;
use pyre_interpreter::{PyDisplay, PyExecutionContext};
use pyre_jit::eval::eval_with_jit;

mod repl;
mod repl_readline;

enum RunMode {
    Script(String),
    Command(String),
    Repl,
}

fn usage(binary_name: &str) -> String {
    format!(
        "\
usage: {binary_name} [option] ... [-c cmd | file | -] [arg] ...
Options:
-c cmd : program passed in as string (terminates option list)
-h     : print this help message and exit (also --help)
-i     : inspect interactively after running script
-O     : optimize (no-op, reserved for compatibility)
-q     : don't print version on interactive startup
-V     : print the Python version number and exit (also --version)
file   : program read from script file
-      : program read from stdin (default; interactive mode if a tty)
arg ...: arguments passed to program in sys.argv[1:]
"
    )
}

fn parse_args(binary_name: &str) -> Result<(RunMode, bool, bool), lexopt::Error> {
    let mut parser = lexopt::Parser::from_env();
    let mut inspect = false;
    let mut quiet = false;

    while let Some(arg) = parser.next()? {
        match arg {
            Short('c') => {
                let cmd = parser.value()?.string()?;
                return Ok((RunMode::Command(cmd), inspect, quiet));
            }
            Short('h') | Long("help") => {
                print!("{}", usage(binary_name));
                std::process::exit(0);
            }
            Short('i') => inspect = true,
            Short('O') => {} // no-op
            Short('q') => quiet = true,
            Short('V') | Long("version") => {
                println!("{binary_name} 0.0.1");
                std::process::exit(0);
            }
            Value(script) => {
                let script = script.string()?;
                let mode = if script == "-" {
                    RunMode::Repl
                } else {
                    RunMode::Script(script)
                };
                return Ok((mode, inspect, quiet));
            }
            _ => return Err(arg.unexpected()),
        }
    }
    Ok((RunMode::Repl, inspect, quiet))
}

pub fn main_entry(binary_name: &'static str) {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| real_main(binary_name))
        .expect("spawn main thread")
        .join()
        .unwrap();
}

fn real_main(binary_name: &str) {
    // Suppress panic messages for InvalidLoop — these are caught by
    // catch_unwind in the JIT optimizer but the default panic hook still
    // prints to stderr, making it look like a crash.
    // RPython: InvalidLoop is a silent exception, not an error.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Only print if it's NOT an InvalidLoop (which is caught by catch_unwind)
        let payload = info.payload();
        if payload
            .downcast_ref::<majit_metainterp::optimize::InvalidLoop>()
            .is_none()
        {
            default_hook(info);
        }
    }));
    let (mode, inspect, quiet) = match parse_args(binary_name) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{binary_name}: {e}");
            std::process::exit(2);
        }
    };

    // pypy/interpreter/app_main.py:824-825 parity for the standalone
    // launcher: untranslated hosts need a higher startup recursion limit
    // than translated PyPy's default 1000 because each host-language
    // frame is larger. Pyre is likewise running on Rust frames here, so
    // raise the startup budget before executing user code.
    pyre_interpreter::stack_check::set_recursion_limit(5000)
        .expect("startup recursion limit must be applicable");

    // Eagerly install pyre-jit's hooks into pyre-interpreter so that
    // sys.settrace / set_jit_param routing is live from the very first
    // user statement, not only after the first JIT-traced bytecode.
    pyre_jit::eval::init_jit_hooks();

    match mode {
        RunMode::Command(cmd) => {
            // Initialize sys.path with CWD for -c mode.
            let cwd = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
            importing::init_sys_path(&cwd);
            importing::set_sys_argv(&["-c".to_string()]);
            run_source(&cmd, Mode::Exec);
            if inspect {
                repl::run_repl(true);
            }
        }
        RunMode::Script(path) => {
            let source = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("{binary_name}: cannot open '{path}': {e}");
                    std::process::exit(1);
                }
            };
            // Initialize sys.path with the script's directory.
            let script_dir = Path::new(&path)
                .parent()
                .unwrap_or(Path::new("."))
                .canonicalize()
                .unwrap_or_else(|_| Path::new(".").to_path_buf());
            importing::init_sys_path(&script_dir);
            // Collect remaining CLI args for sys.argv.
            let argv = vec![path.clone()];
            // lexopt consumed script name; remaining values go to sys.argv[1:]
            importing::set_sys_argv(&argv);
            run_source(&source, Mode::Exec);
            if inspect {
                repl::run_repl(true);
            }
        }
        RunMode::Repl => {
            // Initialize sys.path with CWD for REPL mode.
            let cwd = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
            importing::init_sys_path(&cwd);
            repl::run_repl(quiet);
        }
    }
}

fn run_source(source: &str, mode: Mode) {
    let code = match compile_source(source, mode) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // Register __build_class__ callback (PyPy: setup_builtin_modules)
    call::register_build_class();

    let execution_context = Rc::new(PyExecutionContext::default());
    // Set execution context for __build_class__ to use
    call::set_build_class_exec_ctx(Rc::as_ptr(&execution_context));
    // Eagerly seed the LAST_EXEC_CTX TLS slot so that
    // `space.getexecutioncontext()` (sys.settrace/setprofile/getframe)
    // returns the live ExecutionContext from the very first user
    // statement, not only after the first `eval_frame_plain` entry
    // updates the slot.  Mirrors PyPy's `space.threadlocals` always
    // holding the active EC for the current thread.
    call::set_last_exec_ctx(Rc::as_ptr(&execution_context));
    let mut frame = PyFrame::new_with_context(code, execution_context);

    // Register __main__ module in sys.modules — PyPy: app_main sets
    // sys.modules['__main__'] before executing user code so that
    // enum.global_enum and similar introspection works.
    let main_module =
        pyre_object::moduleobject::w_module_new("__main__", frame.w_globals as *mut u8);
    importing::set_sys_module("__main__", main_module);

    match eval_with_jit(&mut frame) {
        Ok(result) => {
            if !result.is_null() && !unsafe { pyre_object::is_none(result) } {
                println!("{}", PyDisplay(result));
            }
        }
        Err(e) => {
            if e.kind == pyre_interpreter::PyErrorKind::SystemExit {
                let code: i32 = e.message.parse().unwrap_or(0);
                std::process::exit(code);
            }
            pyre_interpreter::eprint_exception(&e, true);
            std::process::exit(1);
        }
    }
}
