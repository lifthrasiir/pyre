use wasm_bindgen::prelude::*;

use pyre_interpreter::*;

use std::cell::RefCell;
use std::sync::Once;

static PANIC_HOOK: Once = Once::new();

fn install_panic_hook() {
    PANIC_HOOK.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            let msg = format!("[pyre panic] {info}");
            OUTPUT_BUF.with(|buf| buf.borrow_mut().push_str(&msg));
        }));
    });
}

thread_local! {
    static OUTPUT_BUF: RefCell<String> = RefCell::new(String::new());
}

fn install_wasm_print_hook() {
    pyre_interpreter::set_print_hook(|s| {
        OUTPUT_BUF.with(|buf| buf.borrow_mut().push_str(s));
    });
}

/// Run a Python source string and return the output as a string.
#[wasm_bindgen]
pub fn run_python(source: &str) -> String {
    install_panic_hook();
    pyre_interpreter::importing::install_builtin_modules();
    install_wasm_print_hook();
    OUTPUT_BUF.with(|buf| buf.borrow_mut().clear());

    let code = match compile_source(source, Mode::Exec) {
        Ok(code) => code,
        Err(e) => return format!("SyntaxError: {e}"),
    };

    let execution_context = std::rc::Rc::new(PyExecutionContext::default());
    let mut frame =
        match pyre_interpreter::pyframe::PyFrame::new_with_context(code, execution_context) {
            Ok(frame) => frame,
            Err(e) => return format!("Error: {e}"),
        };

    // catch_unwind to capture panics from JIT as error messages
    let eval_result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pyre_jit::eval::eval_with_jit(&mut frame)
    })) {
        Ok(r) => r,
        Err(_) => {
            let panic_msg = OUTPUT_BUF.with(|buf| buf.borrow().clone());
            return if panic_msg.is_empty() {
                "[pyre] unknown panic".to_string()
            } else {
                panic_msg
            };
        }
    };

    let mut output = OUTPUT_BUF.with(|buf| buf.borrow().clone());

    match eval_result {
        Ok(result) => {
            if !result.is_null() && !unsafe { pyre_object::is_none(result) } {
                if !output.is_empty() && !output.ends_with('\n') {
                    output.push('\n');
                }
                output.push_str(&format!("{}", PyDisplay(result)));
            }
        }
        Err(e) => {
            if !output.is_empty() && !output.ends_with('\n') {
                output.push('\n');
            }
            output.push_str(&format!("Error: {e}"));
        }
    }

    output
}
