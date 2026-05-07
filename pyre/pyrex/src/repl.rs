use std::path::PathBuf;
use std::rc::Rc;

use rustpython_compiler::{
    CompileError, Mode, ParseError,
    parser::{InterpolatedStringErrorType, LexicalErrorType, ParseErrorType},
};

use pyre_interpreter::call;
use pyre_interpreter::importing;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{DictStorage, PyDisplay, PyError, PyExecutionContext, dict_storage_store};
use pyre_jit::eval::eval_with_jit;

use crate::repl_readline::{Readline, ReadlineResult};

const DEFAULT_PRIMARY_PROMPT: &str = ">>> ";
const DEFAULT_SECONDARY_PROMPT: &str = "... ";

enum ShellCompileAction {
    Execute(pyre_interpreter::CodeObject),
    Ignore,
    ContinueBlock,
    ContinueLine,
    CompileErr(String),
}

enum ShellExecResult {
    Ok,
    ContinueBlock,
    ContinueLine,
    RuntimeErr(PyError),
    CompileErr(String),
}

struct ReplRuntime {
    ctx_ptr: *const PyExecutionContext,
    namespace: *mut DictStorage,
    sys_module: pyre_object::PyObjectRef,
}

pub fn run_repl(quiet: bool) {
    let mut repl = Readline::new();
    let history_path = repl_history_path();

    if let Err(err) = repl.load_history(&history_path) {
        eprintln!("pyre: could not load REPL history: {err}");
    }

    let execution_context = Rc::new(PyExecutionContext::default());
    call::register_build_class();
    call::set_build_class_exec_ctx(Rc::as_ptr(&execution_context));

    let mut namespace = Box::new(execution_context.fresh_dict_storage());
    namespace.fix_ptr();
    dict_storage_store(
        &mut namespace,
        "__name__",
        pyre_object::w_str_new("__main__"),
    );
    let namespace = Box::into_raw(namespace);

    let main_module = pyre_object::moduleobject::w_module_new("__main__", namespace as *mut u8);
    // Bind storage→W_DictObject back-mirror so REPL STORE_NAME writes
    // surface in `__main__.__dict__`'s keys/items/lookups (PyPy
    // `module.py:77 Module.getdict()` parity).
    unsafe {
        pyre_interpreter::bind_module_back_mirror(namespace, main_module);
    }
    importing::set_sys_module("__main__", main_module);

    let sys_module = match importing::importhook(
        "sys",
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        0,
        Rc::as_ptr(&execution_context),
    ) {
        Ok(module) => module,
        Err(err) => {
            pyre_interpreter::eprint_exception(&err, true);
            return;
        }
    };
    configure_sys_for_repl(sys_module);

    let runtime = ReplRuntime {
        ctx_ptr: Rc::into_raw(Rc::clone(&execution_context)),
        namespace,
        sys_module,
    };

    if !quiet {
        println!("pyre 0.0.1 (Rust meta-tracing JIT)");
        println!("Type \"exit()\" or Ctrl-D to exit.");
    }

    let mut full_input = String::new();
    let mut continuing_block = false;
    let mut continuing_line = false;

    loop {
        let prompt = if continuing_block || continuing_line {
            load_prompt(&runtime, "ps2", DEFAULT_SECONDARY_PROMPT)
        } else {
            load_prompt(&runtime, "ps1", DEFAULT_PRIMARY_PROMPT)
        };
        continuing_line = false;

        let line = match repl.readline(&prompt) {
            ReadlineResult::Line(line) => line,
            ReadlineResult::Interrupt => {
                eprintln!("KeyboardInterrupt");
                full_input.clear();
                continuing_block = false;
                continuing_line = false;
                continue;
            }
            ReadlineResult::Eof => {
                println!();
                break;
            }
            ReadlineResult::Io(err) => {
                eprintln!("pyre: REPL I/O error: {err}");
                break;
            }
            ReadlineResult::Other(err) => {
                eprintln!("pyre: REPL error: {err}");
                break;
            }
        };

        if let Err(err) = repl.add_history_entry(line.trim_end()) {
            eprintln!("pyre: could not record REPL history: {err}");
        }

        let empty_line_given = line.is_empty();
        if full_input.is_empty() {
            full_input = line;
        } else {
            full_input.push_str(&line);
        }
        full_input.push('\n');

        match shell_exec(&runtime, &full_input, empty_line_given, continuing_block) {
            ShellExecResult::Ok => {
                if continuing_block {
                    if empty_line_given {
                        continuing_block = false;
                        full_input.clear();
                    }
                } else {
                    full_input.clear();
                }
            }
            ShellExecResult::ContinueLine => {
                continuing_line = true;
            }
            ShellExecResult::ContinueBlock => {
                continuing_block = true;
            }
            ShellExecResult::CompileErr(err) => {
                eprintln!("{err}");
                full_input.clear();
                continuing_block = false;
                continuing_line = false;
            }
            ShellExecResult::RuntimeErr(err) => {
                pyre_interpreter::eprint_exception(&err, true);
                full_input.clear();
                continuing_block = false;
                continuing_line = false;
            }
        }
    }

    if let Err(err) = repl.save_history(&history_path) {
        eprintln!("pyre: could not save REPL history: {err}");
    }
}

fn configure_sys_for_repl(sys_module: pyre_object::PyObjectRef) {
    ensure_sys_prompt(sys_module, "ps1", DEFAULT_PRIMARY_PROMPT);
    ensure_sys_prompt(sys_module, "ps2", DEFAULT_SECONDARY_PROMPT);

    if let Ok(flags) = pyre_interpreter::baseobjspace::getattr(sys_module, "flags") {
        let _ = pyre_interpreter::baseobjspace::setattr(
            flags,
            "interactive",
            pyre_object::w_int_new(1),
        );
        let _ =
            pyre_interpreter::baseobjspace::setattr(flags, "inspect", pyre_object::w_int_new(1));
    }
}

fn ensure_sys_prompt(sys_module: pyre_object::PyObjectRef, name: &str, fallback: &str) {
    if pyre_interpreter::baseobjspace::getattr(sys_module, name).is_err() {
        let _ = pyre_interpreter::baseobjspace::setattr(
            sys_module,
            name,
            pyre_object::w_str_new(fallback),
        );
    }
}

fn load_prompt(runtime: &ReplRuntime, name: &str, fallback: &str) -> String {
    read_prompt(runtime.sys_module, name).unwrap_or_else(|| fallback.to_string())
}

fn read_prompt(sys_module: pyre_object::PyObjectRef, name: &str) -> Option<String> {
    let prompt = pyre_interpreter::baseobjspace::getattr(sys_module, name).ok()?;
    if prompt.is_null() || unsafe { pyre_object::is_none(prompt) } {
        return None;
    }
    Some(unsafe { pyre_interpreter::py_str(prompt) })
}

fn shell_exec(
    runtime: &ReplRuntime,
    source: &str,
    empty_line_given: bool,
    continuing_block: bool,
) -> ShellExecResult {
    match compile_repl_input(source, empty_line_given, continuing_block) {
        ShellCompileAction::Execute(code) => {
            let code_ptr = Box::into_raw(Box::new(code));
            let w_code = pyre_interpreter::pycode::w_code_new(code_ptr as *const ());
            let mut frame = match pyre_interpreter::createframe(
                w_code as *const (),
                runtime.namespace,
                runtime.ctx_ptr,
                None,
            ) {
                Ok(frame) => frame,
                Err(err) => return ShellExecResult::RuntimeErr(err),
            };
            match eval_with_jit(&mut frame) {
                Ok(result) => {
                    if !result.is_null() && !unsafe { pyre_object::is_none(result) } {
                        println!("{}", PyDisplay(result));
                    }
                    ShellExecResult::Ok
                }
                Err(err) => ShellExecResult::RuntimeErr(err),
            }
        }
        ShellCompileAction::Ignore => ShellExecResult::Ok,
        ShellCompileAction::ContinueBlock => ShellExecResult::ContinueBlock,
        ShellCompileAction::ContinueLine => ShellExecResult::ContinueLine,
        ShellCompileAction::CompileErr(err) => ShellExecResult::CompileErr(err),
    }
}

fn compile_repl_input(
    source: &str,
    empty_line_given: bool,
    continuing_block: bool,
) -> ShellCompileAction {
    #[cfg(windows)]
    let normalized = source.replace("\r\n", "\n");
    #[cfg(windows)]
    let source = normalized.as_str();

    match pyre_interpreter::rp_compile(source, Mode::Single, "<stdin>".into(), Default::default()) {
        Ok(code) => {
            if empty_line_given || !continuing_block {
                ShellCompileAction::Execute(code)
            } else {
                ShellCompileAction::Ignore
            }
        }
        Err(CompileError::Parse(ParseError {
            error: ParseErrorType::Lexical(LexicalErrorType::Eof),
            ..
        })) => ShellCompileAction::ContinueLine,
        Err(CompileError::Parse(ParseError {
            error:
                ParseErrorType::Lexical(LexicalErrorType::FStringError(
                    InterpolatedStringErrorType::UnterminatedTripleQuotedString,
                )),
            ..
        })) => ShellCompileAction::ContinueLine,
        Err(CompileError::Parse(ParseError {
            is_unclosed_bracket: true,
            ..
        })) => ShellCompileAction::ContinueLine,
        Err(err) => {
            if let CompileError::Parse(ParseError {
                error: ParseErrorType::Lexical(LexicalErrorType::UnclosedStringError),
                raw_location,
                ..
            }) = &err
            {
                let loc = raw_location.start().to_usize();
                let mut iter = source.chars();
                if let Some(quote) = iter.nth(loc)
                    && iter.next() == Some(quote)
                    && iter.next() == Some(quote)
                {
                    return ShellCompileAction::ContinueLine;
                }
            }

            let bad_error = match &err {
                CompileError::Parse(parse_err) => match &parse_err.error {
                    ParseErrorType::Lexical(LexicalErrorType::IndentationError) => continuing_block,
                    ParseErrorType::OtherError(msg) => {
                        !msg.starts_with("Expected an indented block")
                    }
                    _ => true,
                },
                _ => true,
            };

            if empty_line_given || bad_error {
                ShellCompileAction::CompileErr(format!("compile error: {err}"))
            } else {
                ShellCompileAction::ContinueBlock
            }
        }
    }
}

fn repl_history_path() -> PathBuf {
    match dirs::config_dir() {
        Some(mut path) => {
            path.push("pyre");
            path.push("repl_history.txt");
            path
        }
        None => PathBuf::from(".pyre_repl_history.txt"),
    }
}

#[cfg(test)]
mod tests {
    use super::{ShellCompileAction, compile_repl_input, read_prompt};

    #[test]
    fn incomplete_block_continues() {
        let result = compile_repl_input("if True:\n", false, false);
        assert!(matches!(result, ShellCompileAction::ContinueBlock));
    }

    #[test]
    fn incomplete_line_continues() {
        let result = compile_repl_input("x = (\n", false, false);
        assert!(matches!(result, ShellCompileAction::ContinueLine));
    }

    #[test]
    fn completed_block_waits_for_blank_line() {
        let result = compile_repl_input("if True:\n    1\n", false, true);
        assert!(matches!(result, ShellCompileAction::Ignore));
    }

    #[test]
    fn reads_prompt_from_sys_module() {
        let mut namespace = Box::new(pyre_interpreter::DictStorage::default());
        namespace.fix_ptr();
        pyre_interpreter::dict_storage_store(&mut namespace, "ps1", pyre_object::w_str_new("py> "));
        let ns_ptr = Box::into_raw(namespace);
        let sys_module = pyre_object::moduleobject::w_module_new("sys", ns_ptr as *mut u8);
        // bind_module_back_mirror pre-populates the W_DictObject entries
        // Vec from the storage so getattr → finditem_str(w_dict, ...)
        // sees the pre-existing `ps1` binding without depending on the
        // storage-proxy lookup hook (not registered in this test).
        unsafe {
            pyre_interpreter::bind_module_back_mirror(ns_ptr, sys_module);
        }

        assert_eq!(read_prompt(sys_module, "ps1").as_deref(), Some("py> "));
    }
}
