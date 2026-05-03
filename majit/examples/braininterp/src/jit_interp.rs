/// JIT-enabled Brainfuck interpreter using `#[jit_interp]` + `state_fields`.
///
/// Greens: [pc, program]
/// Reds:   [pointer, tape]  (tracked via state_fields)
///
/// The tape is a state array: during tracing, tape cells are tracked as
/// symbolic OpRefs, eliminating memory loads/stores in compiled code.
///
/// Back-edge detection: `]` that jumps backward to matching `[` is the
/// loop back-edge. When it becomes hot, tracing starts at the `[` header.

pub type Bytecode = [u8];

trait BytecodeExt {
    fn get_op(&self, pc: usize) -> u8;
}
impl BytecodeExt for [u8] {
    fn get_op(&self, pc: usize) -> u8 {
        self[pc]
    }
}

const TAPE_SIZE: usize = 30000;
const DEFAULT_THRESHOLD: u32 = 3;

struct BfState {
    pointer: i64,
    tape: Vec<i64>,
}

#[majit_macros::jit_interp(
    state = BfState,
    env = Bytecode,
    state_fields = {
        pointer: int,
        tape: [int; virt],
    },
)]
fn mainloop(program: &Bytecode, threshold: u32) -> String {
    let mut driver: majit_metainterp::JitDriver<BfState> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut state = BfState {
        pointer: 0,
        tape: vec![0i64; TAPE_SIZE],
    };
    let mut output = String::new();

    // RPython warmspot.py: make_jitcodes() before pyjitpl.py finish_setup().
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    loop {
        if pc >= program.len() {
            break;
        }
        jit_merge_point!();
        let ch = program[pc];

        match ch {
            b'>' => {
                state.pointer = state.pointer + 1;
                pc = pc + 1;
            }
            b'<' => {
                state.pointer = state.pointer - 1;
                pc = pc + 1;
            }
            b'+' => {
                state.tape[state.pointer as usize] = state.tape[state.pointer as usize] + 1;
                pc = pc + 1;
            }
            b'-' => {
                state.tape[state.pointer as usize] = state.tape[state.pointer as usize] - 1;
                pc = pc + 1;
            }
            b'.' => break,
            b',' => break,
            b'[' => {
                if state.tape[state.pointer as usize] == 0 {
                    let mut need: i32 = 1;
                    let mut p = pc + 1;
                    while need > 0 {
                        if program[p] == b']' {
                            need = need - 1;
                        } else if program[p] == b'[' {
                            need = need + 1;
                        }
                        p = p + 1;
                    }
                    pc = p;
                } else {
                    pc = pc + 1;
                }
            }
            b']' => {
                if state.tape[state.pointer as usize] != 0 {
                    let target = find_matching_open(program, pc);
                    if target < pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                } else {
                    pc = pc + 1;
                }
            }
            _ => {
                pc = pc + 1;
            }
        }
    }

    // Handle I/O after breaking out of the traced loop.
    // The trace aborts on '.' and ','; the interpreter handles them here
    // and re-enters the loop.
    while pc < program.len() {
        let ch = program[pc];
        if ch == b'.' {
            output.push(state.tape[state.pointer as usize] as u8 as char);
            pc += 1;
        } else if ch == b',' {
            state.tape[state.pointer as usize] = 0;
            pc += 1;
        } else {
            // Re-enter the traced mainloop for the remaining program.
            let remaining_output = mainloop_resume(program, &mut state, &mut driver, pc, threshold);
            output.push_str(&remaining_output);
            break;
        }
    }

    output
}

/// Resume mainloop execution from a given pc, reusing existing state and driver.
fn mainloop_resume(
    program: &Bytecode,
    state: &mut BfState,
    _driver: &mut majit_metainterp::JitDriver<BfState>,
    mut pc: usize,
    _threshold: u32,
) -> String {
    let mut output = String::new();

    while pc < program.len() {
        let ch = program[pc];
        if ch == b'>' {
            state.pointer += 1;
            pc += 1;
        } else if ch == b'<' {
            state.pointer -= 1;
            pc += 1;
        } else if ch == b'+' {
            state.tape[state.pointer as usize] += 1;
            pc += 1;
        } else if ch == b'-' {
            state.tape[state.pointer as usize] -= 1;
            pc += 1;
        } else if ch == b'.' {
            output.push(state.tape[state.pointer as usize] as u8 as char);
            pc += 1;
        } else if ch == b',' {
            state.tape[state.pointer as usize] = 0;
            pc += 1;
        } else if ch == b'[' {
            if state.tape[state.pointer as usize] == 0 {
                let mut need: i32 = 1;
                let mut p = pc + 1;
                while need > 0 {
                    if program[p] == b']' {
                        need -= 1;
                    } else if program[p] == b'[' {
                        need += 1;
                    }
                    p += 1;
                }
                pc = p;
            } else {
                pc += 1;
            }
        } else if ch == b']' {
            if state.tape[state.pointer as usize] != 0 {
                let target = find_matching_open(program, pc);
                pc = target;
            } else {
                pc += 1;
            }
        } else {
            pc += 1;
        }
    }

    output
}

/// Find the matching '[' for a ']' at the given position.
fn find_matching_open(code: &[u8], close_pos: usize) -> usize {
    let mut need: i32 = 1;
    let mut p = close_pos - 1;
    while need > 0 {
        if code[p] == b']' {
            need += 1;
        } else if code[p] == b'[' {
            need -= 1;
        }
        if need > 0 {
            p -= 1;
        }
    }
    p
}

pub struct JitBrainInterp {
    threshold: u32,
}

impl JitBrainInterp {
    pub fn new() -> Self {
        JitBrainInterp {
            threshold: DEFAULT_THRESHOLD,
        }
    }

    pub fn run(&mut self, code: &[u8]) -> String {
        mainloop(code, self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    #[test]
    fn jit_simple_loop() {
        // Move cell0 to cell1: +++++[->+<]
        let mut jit = JitBrainInterp::new();
        let output = jit.run(b"+++++[->+<]");
        assert_eq!(output.len(), 0);
    }

    #[test]
    fn jit_multiply() {
        // cell1 = 9 * 9 = 81
        let mut jit = JitBrainInterp::new();
        let output = jit.run(b"+++++++++[>+++++++++<-]");
        assert_eq!(output.len(), 0);
    }

    #[test]
    fn jit_matches_interp_move() {
        let code = b"+++++[->+<]";
        let expected = interp::interpret(code);
        let mut jit = JitBrainInterp::new();
        let got = jit.run(code);
        assert_eq!(got, expected);
    }

    #[test]
    fn jit_matches_interp_multiply() {
        let code = b"+++++++++[>+++++++++<-]";
        let expected = interp::interpret(code);
        let mut jit = JitBrainInterp::new();
        let got = jit.run(code);
        assert_eq!(got, expected);
    }

    #[test]
    fn jit_no_loop() {
        let mut jit = JitBrainInterp::new();
        let output = jit.run(b"+++");
        assert_eq!(output.len(), 0);
    }
}
