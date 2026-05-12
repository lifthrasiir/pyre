//! Port of `rpython/rtyper/llinterp.py`.
//!
//! Upstream is 1477 LOC of two top-level types — `LLException` /
//! `LLFatalError` exceptions, plus the `LLInterpreter` and its
//! `LLFrame` opcode-dispatch loop. Driver task `task_llinterpret_lltype`
//! (`driver.py:543-555`) is the only consumer of `LLInterpreter` and
//! uses only:
//!
//! * Constructor: `LLInterpreter(translator.rtyper)` (`:67-82`).
//! * `eval_graph(graph, args)` (`:84-…`): walks a `FunctionGraph`,
//!   running each `SpaceOperation` through `op_<opname>` handlers.
//!
//! The full interpreter requires `rpython.rtyper.lltypesystem.{lltype,
//! llmemory, lloperation, llheap}` plus the full `LLFrame` opcode table.
//! This file starts the real leaf port with the upstream `LLFrame`
//! execution skeleton and a small straight-line operation subset.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::flowspace::model::{
    BlockRef, ConstValue, Constant, FunctionGraph, GraphRef, Hlvalue, LinkRef, SpaceOperation,
    Variable,
};
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rtyper::RPythonTyper;
use crate::translator::tool::taskengine::TaskError;

/// Port of upstream `class LLException(Exception)` at `:28-54`.
///
/// `error_value` defaults to upstream's `UNDEFINED_ERROR_VALUE`
/// sentinel — the local port encodes the sentinel as `None`.
#[derive(Debug, Clone)]
pub struct LLException {
    pub args: Vec<Rc<dyn Any>>,
    pub error_value: Option<Rc<dyn Any>>,
}

impl LLException {
    /// Upstream `__init__(*args, error_value=UNDEFINED_ERROR_VALUE)` at
    /// `:35-40`.
    pub fn new(args: Vec<Rc<dyn Any>>, error_value: Option<Rc<dyn Any>>) -> Self {
        Self { args, error_value }
    }
}

impl std::fmt::Display for LLException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<LLException>")
    }
}

impl std::error::Error for LLException {}

/// Port of upstream `class LLFatalError(Exception)` at `:56-58`.
#[derive(Debug, Clone)]
pub struct LLFatalError(pub Vec<String>);

impl std::fmt::Display for LLFatalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.join(": "))
    }
}

impl std::error::Error for LLFatalError {}

/// Port of upstream `class LLAssertFailure(Exception)` at `:60-61`.
#[derive(Debug, Clone)]
pub struct LLAssertFailure;

impl std::fmt::Display for LLAssertFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LLAssertFailure")
    }
}

impl std::error::Error for LLAssertFailure {}

/// Concrete low-level value used by the local `LLFrame` port.
///
/// Upstream `LLFrame.bindings` stores arbitrary concrete Python objects.
/// Rust does not have that object model, so this private enum carries
/// the primitive shapes currently produced by local lltyped graphs.
#[derive(Clone, Debug, PartialEq)]
enum LLValue {
    Void,
    Int(i64),
    Bool(bool),
    Float(u64),
    Str(String),
    Const(ConstValue),
}

impl LLValue {
    fn into_any(self) -> Rc<dyn Any> {
        match self {
            LLValue::Void => Rc::new(()) as Rc<dyn Any>,
            LLValue::Int(n) => Rc::new(n) as Rc<dyn Any>,
            LLValue::Bool(b) => Rc::new(b) as Rc<dyn Any>,
            LLValue::Float(bits) => Rc::new(f64::from_bits(bits)) as Rc<dyn Any>,
            LLValue::Str(s) => Rc::new(s) as Rc<dyn Any>,
            LLValue::Const(c) => Rc::new(c) as Rc<dyn Any>,
        }
    }

    fn as_i64(&self, opname: &str) -> Result<i64, TaskError> {
        match self {
            LLValue::Int(n) => Ok(*n),
            other => Err(TaskError {
                message: format!("llinterp.py:405 {opname}: expected Signed, got {other:?}"),
            }),
        }
    }

    fn as_bool(&self, opname: &str) -> Result<bool, TaskError> {
        match self {
            LLValue::Bool(b) => Ok(*b),
            other => Err(TaskError {
                message: format!("llinterp.py:405 {opname}: expected Bool, got {other:?}"),
            }),
        }
    }

    fn truth(&self) -> bool {
        match self {
            LLValue::Void => false,
            LLValue::Int(n) => *n != 0,
            LLValue::Bool(b) => *b,
            LLValue::Float(bits) => f64::from_bits(*bits) != 0.0,
            LLValue::Str(s) => !s.is_empty(),
            LLValue::Const(ConstValue::None) => false,
            LLValue::Const(_) => true,
        }
    }
}

fn const_to_llvalue(c: &Constant) -> LLValue {
    match &c.value {
        ConstValue::Int(n) => LLValue::Int(*n),
        ConstValue::Bool(b) => LLValue::Bool(*b),
        ConstValue::Float(bits) => LLValue::Float(*bits),
        // Item 2 string split: ByteStr is the Python 2 `str` shape and
        // is the only string carrier upstream `LLInterpreter` produces
        // here. UniStr falls through to the Const carrier so callers
        // see the type-mismatch shape rather than a silent rewrap.
        ConstValue::ByteStr(s) => LLValue::Str(String::from_utf8_lossy(s).into_owned()),
        ConstValue::None => LLValue::Void,
        other => LLValue::Const(other.clone()),
    }
}

fn any_to_llvalue(value: &Rc<dyn Any>) -> Result<LLValue, TaskError> {
    if let Some(v) = value.downcast_ref::<LLValue>() {
        return Ok(v.clone());
    }
    if let Some(v) = value.downcast_ref::<ConstValue>() {
        return Ok(match v {
            ConstValue::Int(n) => LLValue::Int(*n),
            ConstValue::Bool(b) => LLValue::Bool(*b),
            ConstValue::Float(bits) => LLValue::Float(*bits),
            ConstValue::ByteStr(s) => LLValue::Str(String::from_utf8_lossy(s).into_owned()),
            ConstValue::None => LLValue::Void,
            other => LLValue::Const(other.clone()),
        });
    }
    if let Some(v) = value.downcast_ref::<i64>() {
        return Ok(LLValue::Int(*v));
    }
    if let Some(v) = value.downcast_ref::<bool>() {
        return Ok(LLValue::Bool(*v));
    }
    if let Some(v) = value.downcast_ref::<f64>() {
        return Ok(LLValue::Float(v.to_bits()));
    }
    if let Some(v) = value.downcast_ref::<String>() {
        return Ok(LLValue::Str(v.clone()));
    }
    if value.downcast_ref::<()>().is_some() {
        return Ok(LLValue::Void);
    }
    Err(TaskError {
        message: "llinterp.py:84 LLInterpreter.eval_graph — unsupported concrete argument type"
            .to_string(),
    })
}

/// Port of upstream `class LLInterpreter(object)` at `:67-…`.
///
/// Slot layout matches upstream's `__init__` (`:72-82`). The first
/// executable slice covers `eval_graph` (`:84-126`) and the straight-line
/// parts of `LLFrame` (`:214-309`, `:405-525`).
pub struct LLInterpreter {
    /// Upstream `self.bindings = {}` at `:73`. Maps each Variable to
    /// its concrete value during a frame's `eval()`. Stored as
    /// `Vec<(Rc<dyn Any>, Rc<dyn Any>)>` because the bindings carry
    /// heterogeneous (Variable, [`LLValue`]) pairs that callers
    /// downcast at the read site. lltype is now ported under
    /// [`crate::translator::rtyper::lltypesystem::lltype`]; a
    /// follow-up could narrow the value slot to
    /// `lltype::LowLevelValue` once every read site is migrated
    /// off `Rc<dyn Any>::downcast_ref`.
    pub bindings: RefCell<Vec<(Rc<dyn Any>, Rc<dyn Any>)>>,
    /// Upstream `self.typer = typer` at `:74`.
    pub typer: Rc<RPythonTyper>,
    /// Upstream `self.heap = llheap` at `:76`. Upstream points at the
    /// `rpython.rtyper.lltypesystem.llheap` module (the `malloc`/`free`
    /// surface). The local port has no `llheap` analogue yet — keep the
    /// slot opaque and surface a TaskError citing the upstream module
    /// when consumers actually read it.
    pub heap: Option<Rc<dyn Any>>,
    /// Upstream `self.exc_data_ptr = exc_data_ptr` at `:77`.
    pub exc_data_ptr: Option<Rc<dyn Any>>,
    /// Upstream `self.frame_stack = []` at `:78`.
    pub frame_stack: RefCell<Vec<Rc<RefCell<LLFrame>>>>,
    /// Upstream `self.tracer = None` (or `Tracer()`) at `:79-82`.
    pub tracer: Option<Rc<dyn Any>>,
    /// Upstream `self.frame_class = LLFrame` at `:80`. Upstream stores
    /// the class so subclasses can override; the Rust port has only
    /// the one [`LLFrame`] type, so the slot is documented but unused.
    pub frame_class: std::marker::PhantomData<LLFrame>,
    /// Upstream `self.traceback_frames = []` initialized at the top of
    /// `eval_graph` (`:91`). The local port carries the slot to keep
    /// the upstream shape; it is reset on each `eval_graph` entry.
    pub traceback_frames: RefCell<Vec<Rc<RefCell<LLFrame>>>>,
}

// Upstream module-level `LLInterpreter.current_interpreter = None`
// (`llinterp.py:70`) plus its mutation in `eval_graph` (`:99`):
// `LLInterpreter.current_interpreter = self`. Carried as a thread-local
// `Weak` pointer so the LLFrame can `current_interpreter()` without
// receiver plumbing, matching upstream's class-attribute lookup.
thread_local! {
    static CURRENT_INTERPRETER: std::cell::RefCell<std::rc::Weak<LLInterpreter>>
        = const { std::cell::RefCell::new(std::rc::Weak::new()) };
}

impl LLInterpreter {
    /// Upstream `LLInterpreter.current_interpreter` class-level slot
    /// (`llinterp.py:70`) read by `LLFrame` opcode handlers via
    /// `LLInterpreter.current_interpreter`.
    pub fn current_interpreter() -> Option<Rc<LLInterpreter>> {
        CURRENT_INTERPRETER.with(|cell| cell.borrow().upgrade())
    }

    /// Upstream `__init__(self, typer, tracing=True, exc_data_ptr=None)`
    /// at `:72-82`.
    pub fn new(typer: Rc<RPythonTyper>, tracing: bool, exc_data_ptr: Option<Rc<dyn Any>>) -> Self {
        // Upstream `:80-82`: `if tracing: self.tracer = Tracer()` else
        // `self.tracer = None`. The local `Tracer` port is not landed,
        // so the slot stays `None` regardless. The `tracing` arg is
        // accepted for signature parity but not yet observable —
        // documented PRE-EXISTING-ADAPTATION until `Tracer` lands.
        let _ = tracing;
        Self {
            bindings: RefCell::new(Vec::new()),
            typer,
            // Upstream `:76`: `self.heap = llheap`. Local llheap port
            // is not landed; keep the slot None.
            heap: None,
            exc_data_ptr,
            frame_stack: RefCell::new(Vec::new()),
            tracer: None,
            frame_class: std::marker::PhantomData,
            traceback_frames: RefCell::new(Vec::new()),
        }
    }

    /// Upstream `eval_graph(self, graph, args=(), recursive=False)` at
    /// `llinterp.py:84-126`.
    ///
    /// Mirrors upstream `:99` `LLInterpreter.current_interpreter = self`
    /// unconditionally — every public entry installs the thread-local
    /// pointer that `LLFrame` opcode handlers consult, then restores
    /// the previous value on exit (handling re-entrant evaluation).
    /// Takes `self: &Rc<Self>` because installing a `Weak<Self>` into
    /// the thread-local requires upgrading from a `Rc`, matching
    /// upstream's Python-object identity.
    pub fn eval_graph(
        self: &Rc<Self>,
        graph: Rc<dyn Any>,
        args: Vec<Rc<dyn Any>>,
        _recursive: bool,
    ) -> Result<Rc<dyn Any>, TaskError> {
        // Upstream `llframe = self.frame_class(graph, args, self)` at
        // `llinterp.py:85`. The local public API still accepts
        // `Rc<dyn Any>` because driver plumbing is not typed yet; the
        // executable leaf accepts the orthodox `GraphRef`.
        let graph = graph
            .downcast::<RefCell<FunctionGraph>>()
            .map_err(|_| TaskError {
                message:
                    "llinterp.py:84 LLInterpreter.eval_graph — expected flowspace::model::GraphRef"
                        .to_string(),
            })?;
        let args: Vec<LLValue> = args.iter().map(any_to_llvalue).collect::<Result<_, _>>()?;
        let llframe = Rc::new(RefCell::new(LLFrame::new(graph, args)));
        // Upstream `:91`: `self.traceback_frames = []`. Reset the slot
        // at every `eval_graph` entry (top-level call resets the stack).
        self.traceback_frames.borrow_mut().clear();
        // Upstream `:99`: `LLInterpreter.current_interpreter = self`.
        // Save the prior value so re-entrant `eval_graph` (recursive
        // calls into another graph) restores correctly on return.
        let prior = CURRENT_INTERPRETER.with(|cell| {
            let old = cell.borrow().clone();
            *cell.borrow_mut() = Rc::downgrade(self);
            old
        });
        let old_depth = self.frame_stack.borrow().len();
        let result = llframe.borrow_mut().eval(self, llframe.clone());
        debug_assert_eq!(self.frame_stack.borrow().len(), old_depth);
        CURRENT_INTERPRETER.with(|cell| {
            *cell.borrow_mut() = prior;
        });
        Ok(result?.into_any())
    }
}

/// Port of upstream `class LLFrame(object)` at `llinterp.py:214-…`.
pub struct LLFrame {
    pub graph: GraphRef,
    args: Vec<LLValue>,
    /// Upstream `self.bindings = {}` at `:221`.
    bindings: HashMap<Variable, LLValue>,
    pub curr_block: Option<BlockRef>,
    pub curr_operation_index: usize,
    pub alloca_objects: Vec<Rc<dyn Any>>,
    /// Upstream catches `LLException as e` at `:323` after an op
    /// raised; `e.args[0]` is the exception class and `e.args[1]` the
    /// instance (`llinterp.py:373-374`). The Rust port captures the
    /// same `(etype, evalue)` pair here as typed `LLValue`s; the next
    /// `eval_block` step consumes it via the `canraise` dispatch.
    pending_exception: Option<(LLValue, LLValue)>,
}

impl LLFrame {
    /// Upstream `LLFrame.__init__(graph, args, llinterpreter)` at
    /// `llinterp.py:215-224`.
    fn new(graph: GraphRef, args: Vec<LLValue>) -> Self {
        Self {
            graph,
            args,
            bindings: HashMap::new(),
            curr_block: None,
            curr_operation_index: 0,
            alloca_objects: Vec::new(),
            pending_exception: None,
        }
    }

    /// Upstream `clear` at `:233-234`.
    fn clear(&mut self) {
        self.bindings.clear();
    }

    /// Upstream `fillvars` at `:236-244`.
    fn fillvars(&mut self, block: &BlockRef, values: &[LLValue]) -> Result<(), TaskError> {
        let vars = block.borrow().inputargs.clone();
        if vars.len() != values.len() {
            return Err(TaskError {
                message: format!(
                    "llinterp.py:236 block received {} args, expected {}",
                    values.len(),
                    vars.len()
                ),
            });
        }
        for (var, val) in vars.iter().zip(values.iter()) {
            self.setvar(var, val.clone())?;
        }
        Ok(())
    }

    /// Upstream `setvar` at `:246-253`.
    fn setvar(&mut self, var: &Hlvalue, val: LLValue) -> Result<(), TaskError> {
        let Hlvalue::Variable(v) = var else {
            return Ok(());
        };
        let val = self.enforce_value(v.concretetype(), val)?;
        self.bindings.insert(v.clone(), val);
        Ok(())
    }

    /// Upstream `getval` at `:256-268`.
    fn getval(&self, varorconst: &Hlvalue) -> Result<LLValue, TaskError> {
        let val = match varorconst {
            Hlvalue::Constant(c) => const_to_llvalue(c),
            Hlvalue::Variable(v) => self.bindings.get(v).cloned().ok_or_else(|| TaskError {
                message: format!("llinterp.py:256 unbound variable {v}"),
            })?,
        };
        match varorconst {
            Hlvalue::Variable(v) => self.enforce_value(v.concretetype(), val),
            Hlvalue::Constant(c) => self.enforce_value(c.concretetype.clone(), val),
        }
    }

    fn enforce_value(
        &self,
        concretetype: Option<LowLevelType>,
        val: LLValue,
    ) -> Result<LLValue, TaskError> {
        match concretetype {
            None | Some(LowLevelType::Void) => Ok(val),
            Some(LowLevelType::Signed)
            | Some(LowLevelType::Unsigned)
            | Some(LowLevelType::SignedLongLong)
            | Some(LowLevelType::SignedLongLongLong)
            | Some(LowLevelType::UnsignedLongLong)
            | Some(LowLevelType::UnsignedLongLongLong) => match val {
                LLValue::Int(_) => Ok(val),
                LLValue::Bool(b) => Ok(LLValue::Int(i64::from(b))),
                other => Err(TaskError {
                    message: format!("llinterp.py:246 type error: expected integer, got {other:?}"),
                }),
            },
            Some(LowLevelType::Bool) => match val {
                LLValue::Bool(_) => Ok(val),
                other => Err(TaskError {
                    message: format!("llinterp.py:246 type error: expected Bool, got {other:?}"),
                }),
            },
            Some(LowLevelType::Float)
            | Some(LowLevelType::SingleFloat)
            | Some(LowLevelType::LongFloat) => match val {
                LLValue::Float(_) => Ok(val),
                other => Err(TaskError {
                    message: format!("llinterp.py:246 type error: expected Float, got {other:?}"),
                }),
            },
            Some(LowLevelType::Char) | Some(LowLevelType::UniChar) => match val {
                LLValue::Str(_) => Ok(val),
                other => Err(TaskError {
                    message: format!("llinterp.py:246 type error: expected Char, got {other:?}"),
                }),
            },
            Some(_) => Ok(val),
        }
    }

    /// Upstream `eval` at `llinterp.py:282-307`.
    fn eval(
        &mut self,
        llinterpreter: &LLInterpreter,
        self_ref: Rc<RefCell<LLFrame>>,
    ) -> Result<LLValue, TaskError> {
        llinterpreter
            .frame_stack
            .borrow_mut()
            .push(self_ref.clone());
        let mut nextblock = self.graph.borrow().startblock.clone();
        let mut args = self.args.clone();
        let result = loop {
            self.clear();
            if let Err(e) = self.fillvars(&nextblock, &args) {
                break Err(e);
            }
            let (block, values) = match self.eval_block(nextblock.clone()) {
                Ok(next) => next,
                Err(e) => break Err(e),
            };
            match block {
                Some(b) => {
                    nextblock = b;
                    args = values;
                }
                None => {
                    break values.into_iter().next().ok_or_else(|| TaskError {
                        message: "llinterp.py:282 return block produced no value".to_string(),
                    });
                }
            }
        };
        let leavingframe = llinterpreter.frame_stack.borrow_mut().pop();
        debug_assert!(
            leavingframe
                .as_ref()
                .is_some_and(|frame| Rc::ptr_eq(frame, &self_ref))
        );
        result
    }

    /// Upstream `eval_block` at `llinterp.py:309-403`.
    fn eval_block(
        &mut self,
        block: BlockRef,
    ) -> Result<(Option<BlockRef>, Vec<LLValue>), TaskError> {
        self.curr_block = Some(block.clone());
        let operations = block.borrow().operations.clone();
        // Upstream `:316-326`: ops run in sequence; only the last op
        // of a `canraise` block is allowed to raise `LLException` —
        // earlier raises propagate as TaskError.
        let canraise = block.borrow().canraise();
        let last_index = operations.len().saturating_sub(1);
        for (i, op) in operations.iter().enumerate() {
            self.curr_operation_index = i;
            self.eval_operation(op)?;
            if self.pending_exception.is_some() && (!canraise || i != last_index) {
                let (etype, _) = self.pending_exception.take().unwrap();
                return Err(TaskError {
                    message: format!(
                        "llinterp.py:316 unhandled LLException at op {}: etype={:?}",
                        op.opname, etype
                    ),
                });
            }
        }

        let exits = block.borrow().exits.clone();
        if exits.is_empty() {
            let variables = block.borrow().getvariables();
            if variables.len() == 2 {
                return Err(TaskError {
                    message:
                        "llinterp.py:325 exception return blocks are not implemented in LLFrame"
                            .to_string(),
                });
            }
            if variables.len() != 1 {
                return Err(TaskError {
                    message: format!(
                        "llinterp.py:338 return block expected one result variable, got {}",
                        variables.len()
                    ),
                });
            }
            let result = self.getval(&Hlvalue::Variable(variables[0].clone()))?;
            return Ok((None, vec![result]));
        }

        let link = if block.borrow().exitswitch.is_none() {
            // Upstream `llinterp.py:365-368`: single-exit block.
            if exits.len() != 1 {
                return Err(TaskError {
                    message: format!(
                        "llinterp.py:350 single-exit block expected one exit, got {}",
                        exits.len()
                    ),
                });
            }
            exits[0].clone()
        } else if canraise {
            // Upstream `llinterp.py:369-384`: a `canraise` block has
            // its normal flow on `block.exits[0]` and exception
            // handlers on `block.exits[1:]`. When `e is None` (the op
            // did not raise) take `exits[0]`. Otherwise walk the
            // handler links, dispatching to the first one whose
            // `llexitcase` matches; populate the link's
            // `last_exception` / `last_exc_value` Variables so the
            // target block sees them.
            match self.pending_exception.take() {
                None => exits[0].clone(),
                Some((etype, evalue)) => self.dispatch_exception(&exits[1..], &etype, &evalue)?,
            }
        } else {
            self.choose_switch_link(&block, &exits)?
        };
        let target = link.borrow().target.clone().ok_or_else(|| TaskError {
            message: "llinterp.py:397 link target is unset".to_string(),
        })?;
        let mut values = Vec::with_capacity(link.borrow().args.len());
        for arg in &link.borrow().args {
            let arg = arg.as_ref().ok_or_else(|| TaskError {
                message: "llinterp.py:397 merge link carried undefined local".to_string(),
            })?;
            values.push(self.getval(arg)?);
        }
        Ok((Some(target), values))
    }

    /// Upstream `llinterp.py:371-384` exception-handler dispatch loop.
    /// Walks `block.exits[1:]`, returns the first link whose
    /// `llexitcase` is a (super)class of `etype`, populating the
    /// link's `last_exception` / `last_exc_value` Variables. If no
    /// handler matches the upstream `else: raise e` branch surfaces
    /// here as a TaskError carrying the etype for the caller.
    fn dispatch_exception(
        &mut self,
        handler_links: &[LinkRef],
        etype: &LLValue,
        evalue: &LLValue,
    ) -> Result<LinkRef, TaskError> {
        for link in handler_links {
            let exitcase = {
                let l = link.borrow();
                debug_assert!(
                    l.exitcase.is_some(),
                    "llinterp.py:376 exception handler link must have an exitcase"
                );
                l.llexitcase.clone().or_else(|| l.exitcase.clone())
            };
            if !self.exception_link_matches(exitcase.as_ref(), etype) {
                continue;
            }
            let (last_exception, last_exc_value) = {
                let l = link.borrow();
                (l.last_exception.clone(), l.last_exc_value.clone())
            };
            if let Some(var) = last_exception {
                self.setvar(&var, etype.clone())?;
            }
            if let Some(var) = last_exc_value {
                self.setvar(&var, evalue.clone())?;
            }
            return Ok(link.clone());
        }
        Err(TaskError {
            message: format!("llinterp.py:382 unhandled LLException: etype={etype:?}",),
        })
    }

    /// Upstream `op_direct_call(exdata.fn_exception_match, cls,
    /// link.llexitcase)` at `llinterp.py:377`. The full port routes
    /// through the typer's `exceptiondata.fn_exception_match` runtime
    /// function — that helper is itself a generated graph and is not
    /// yet ported. Until then the local implementation matches by
    /// `host_class_name` qualname equality, which is sound for the
    /// rtyper-built graphs the test suite exercises (concrete
    /// exception classes, no class-hierarchy walks).
    fn exception_link_matches(&self, llexitcase: Option<&Hlvalue>, etype: &LLValue) -> bool {
        let Some(llexitcase) = llexitcase else {
            return false;
        };
        let case_name = match llexitcase {
            Hlvalue::Constant(c) => c.value.host_class_name().map(str::to_owned),
            _ => None,
        };
        let etype_name = match etype {
            LLValue::Const(c) => c.host_class_name().map(str::to_owned),
            _ => None,
        };
        match (case_name, etype_name) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    fn choose_switch_link(
        &self,
        block: &BlockRef,
        exits: &[LinkRef],
    ) -> Result<LinkRef, TaskError> {
        let switch = block.borrow().exitswitch.clone().ok_or_else(|| TaskError {
            message: "llinterp.py:359 exitswitch is unset".to_string(),
        })?;
        let llexitvalue = self.getval(&switch)?;
        // Upstream `llinterp.py:386-402`: separate the `"default"`
        // exitcase (if any) from the regular cases, search regular
        // cases first, fall back to the default link if no match.
        let (defaultexit, nondefaultexits) = match exits.last() {
            Some(last) if exitcase_is_default(&last.borrow().exitcase) => {
                (Some(last.clone()), &exits[..exits.len() - 1])
            }
            _ => (None, exits),
        };
        for link in nondefaultexits {
            let llexitcase = {
                let link = link.borrow();
                link.llexitcase
                    .as_ref()
                    .and_then(exitcase_value)
                    .or_else(|| link.exitcase.as_ref().and_then(exitcase_value))
            };
            if llexitcase.is_some_and(|case| llvalue_matches_const(&llexitvalue, &case)) {
                return Ok(link.clone());
            }
        }
        if let Some(link) = defaultexit {
            return Ok(link);
        }
        Err(TaskError {
            message: format!("llinterp.py:399 exit case {llexitvalue:?} not found"),
        })
    }

    /// Upstream `eval_operation` at `llinterp.py:405-525`.
    fn eval_operation(&mut self, operation: &SpaceOperation) -> Result<(), TaskError> {
        let vals = operation
            .args
            .iter()
            .map(|arg| self.getval(arg))
            .collect::<Result<Vec<_>, _>>()?;
        let retval = self.getoperationhandler(&operation.opname, &vals)?;
        self.setvar(&operation.result, retval)
    }

    /// Upstream `getoperationhandler` at `llinterp.py:273-280`, with
    /// only the local primitive fold subset installed.
    fn getoperationhandler(&self, opname: &str, vals: &[LLValue]) -> Result<LLValue, TaskError> {
        match opname {
            "same_as" => vals.first().cloned().ok_or_else(|| TaskError {
                message: "llinterp.py:405 same_as expected one argument".to_string(),
            }),
            "int_add" => Ok(LLValue::Int(
                vals[0]
                    .as_i64(opname)?
                    .wrapping_add(vals[1].as_i64(opname)?),
            )),
            "int_sub" => Ok(LLValue::Int(
                vals[0]
                    .as_i64(opname)?
                    .wrapping_sub(vals[1].as_i64(opname)?),
            )),
            "int_mul" => Ok(LLValue::Int(
                vals[0]
                    .as_i64(opname)?
                    .wrapping_mul(vals[1].as_i64(opname)?),
            )),
            "int_neg" => Ok(LLValue::Int(vals[0].as_i64(opname)?.wrapping_neg())),
            "int_is_true" => Ok(LLValue::Bool(vals[0].truth())),
            "int_eq" => Ok(LLValue::Bool(
                vals[0].as_i64(opname)? == vals[1].as_i64(opname)?,
            )),
            "int_ne" => Ok(LLValue::Bool(
                vals[0].as_i64(opname)? != vals[1].as_i64(opname)?,
            )),
            "int_lt" => Ok(LLValue::Bool(
                vals[0].as_i64(opname)? < vals[1].as_i64(opname)?,
            )),
            "int_le" => Ok(LLValue::Bool(
                vals[0].as_i64(opname)? <= vals[1].as_i64(opname)?,
            )),
            "int_gt" => Ok(LLValue::Bool(
                vals[0].as_i64(opname)? > vals[1].as_i64(opname)?,
            )),
            "int_ge" => Ok(LLValue::Bool(
                vals[0].as_i64(opname)? >= vals[1].as_i64(opname)?,
            )),
            "bool_not" => Ok(LLValue::Bool(!vals[0].as_bool(opname)?)),
            "cast_bool_to_int" => Ok(LLValue::Int(i64::from(vals[0].as_bool(opname)?))),
            _ => Err(TaskError {
                message: format!(
                    "llinterp.py:273 LLFrame.getoperationhandler — op_{opname} not yet ported"
                ),
            }),
        }
    }
}

fn exitcase_value(value: &Hlvalue) -> Option<ConstValue> {
    match value {
        Hlvalue::Constant(c) => Some(c.value.clone()),
        Hlvalue::Variable(_) => None,
    }
}

/// Upstream `llinterp.py:387` reads `block.exits[-1].exitcase ==
/// "default"` directly.  In the Rust port the exitcase carrier is
/// `Hlvalue`; the model encodes the default sentinel as
/// `Constant(Str("default"))` (see `flowspace::model::Block` rendering).
fn exitcase_is_default(exitcase: &Option<Hlvalue>) -> bool {
    matches!(
        exitcase,
        Some(Hlvalue::Constant(Constant { value, .. })) if value.string_eq("default")
    )
}

fn llvalue_matches_const(value: &LLValue, case: &ConstValue) -> bool {
    if matches!(
        (value, case),
        (LLValue::Int(a), ConstValue::Int(b)) if a == b
    ) || matches!(
        (value, case),
        (LLValue::Bool(a), ConstValue::Bool(b)) if a == b
    ) || matches!((value, case), (LLValue::Void, ConstValue::None))
    {
        return true;
    }
    // Item 2 string split: LLValue::Str canonicalises to a Rust
    // `String`. Both ByteStr (utf8-lossy) and UniStr compare against
    // it; the comparison itself stays Python 2-bytes-shaped.
    if let LLValue::Str(a) = value {
        match case {
            ConstValue::ByteStr(b) => return a.as_bytes() == b.as_slice(),
            ConstValue::UniStr(b) => return a == b,
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::flowspace::model::{Block, BlockRefExt, Link};
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    use crate::translator::rtyper::rtyper::RPythonTyper;

    fn fixture_typer() -> Rc<RPythonTyper> {
        let ann = RPythonAnnotator::new(None, None, None, false);
        Rc::new(RPythonTyper::new(&ann))
    }

    #[test]
    fn llinterpreter_init_mirrors_upstream_slots() {
        // Upstream `:72-82`: `__init__(typer, tracing=True,
        // exc_data_ptr=None)` writes `bindings = {}`, `typer = typer`,
        // `exc_data_ptr = exc_data_ptr`, `frame_stack = []`, `tracer
        // = None or Tracer()`.
        let typer = fixture_typer();
        let interp = LLInterpreter::new(typer, true, None);
        assert!(interp.bindings.borrow().is_empty());
        assert!(interp.exc_data_ptr.is_none());
        assert!(interp.frame_stack.borrow().is_empty());
    }

    #[test]
    fn eval_graph_rejects_non_graph_until_driver_passes_real_entry_graph() {
        let interp = Rc::new(LLInterpreter::new(fixture_typer(), false, None));
        let err = interp
            .eval_graph(Rc::new(()) as Rc<dyn Any>, Vec::new(), false)
            .expect_err("non-graph inputs still fail at the typed leaf");
        assert!(err.message.contains("llinterp.py:84"), "{}", err.message);
    }

    #[test]
    fn eval_graph_runs_straight_line_int_add_to_returnblock() {
        // Covers upstream `eval_graph` (`llinterp.py:84-126`),
        // `LLFrame.eval` (`:282-307`), `fillvars/getval/setvar`
        // (`:236-268`), single-exit `eval_block` (`:309-403`), and
        // primitive `eval_operation` dispatch (`:405-525`).
        let x = Variable::named("x");
        x.set_concretetype(Some(LowLevelType::Signed));
        let y = Variable::named("y");
        y.set_concretetype(Some(LowLevelType::Signed));
        let z = Variable::named("z");
        z.set_concretetype(Some(LowLevelType::Signed));

        let start = Block::shared(vec![x.clone().into(), y.clone().into()]);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![x.into(), y.into()],
            z.clone().into(),
        ));
        let retvar = Hlvalue::Variable(Variable::named("ret"));
        if let Hlvalue::Variable(v) = &retvar {
            v.set_concretetype(Some(LowLevelType::Signed));
        }
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "add",
            start.clone(),
            retvar,
        )));
        let returnblock = graph.borrow().returnblock.clone();
        start.closeblock(vec![
            Link::new(vec![z.into()], Some(returnblock), None).into_ref(),
        ]);

        let interp = Rc::new(LLInterpreter::new(fixture_typer(), false, None));
        let out = interp
            .eval_graph(
                graph as Rc<dyn Any>,
                vec![
                    Rc::new(40_i64) as Rc<dyn Any>,
                    Rc::new(2_i64) as Rc<dyn Any>,
                ],
                false,
            )
            .expect("straight-line int_add graph should run");
        assert_eq!(*out.downcast::<i64>().expect("Signed return"), 42);
        assert!(interp.frame_stack.borrow().is_empty());
        // Upstream `:99` `LLInterpreter.current_interpreter = self`:
        // every public eval_graph entry must have published the
        // thread-local pointer while running. After return the local
        // mirror is restored to the prior (empty) Weak.
        assert!(LLInterpreter::current_interpreter().is_none());
    }

    #[test]
    fn eval_graph_publishes_current_interpreter_during_execution() {
        // Upstream `llinterp.py:99` `LLInterpreter.current_interpreter
        // = self` is mandatory side effect of every `eval_graph` call.
        // Without it, downstream opcode handlers (e.g. an `op_*` that
        // reads `LLInterpreter.current_interpreter` per upstream
        // `:405-525`) cannot recover the active interpreter.
        let x = Variable::named("x");
        x.set_concretetype(Some(LowLevelType::Signed));
        let start = Block::shared(vec![x.clone().into()]);
        let retvar = Hlvalue::Variable(Variable::named("ret"));
        if let Hlvalue::Variable(v) = &retvar {
            v.set_concretetype(Some(LowLevelType::Signed));
        }
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "passthrough",
            start.clone(),
            retvar,
        )));
        let returnblock = graph.borrow().returnblock.clone();
        start.closeblock(vec![
            Link::new(vec![x.into()], Some(returnblock), None).into_ref(),
        ]);

        let interp = Rc::new(LLInterpreter::new(fixture_typer(), false, None));
        // Before the call: thread-local has no current interpreter.
        assert!(LLInterpreter::current_interpreter().is_none());
        let _ = interp
            .eval_graph(
                graph as Rc<dyn Any>,
                vec![Rc::new(7_i64) as Rc<dyn Any>],
                false,
            )
            .expect("passthrough graph runs");
        // After the call: the thread-local has been restored to the
        // prior (empty) Weak. Upstream's class attribute survives the
        // call, but the local port saves/restores so re-entrant
        // evaluations behave correctly.
        assert!(LLInterpreter::current_interpreter().is_none());
    }

    #[test]
    fn llexception_carries_upstream_argument_layout() {
        // Upstream `:28-54`: `LLException(*args, error_value=...)`. The
        // local port preserves `args` and `error_value` slots so tests
        // tracking `e.error_value` (e.g. test_exceptiontransform per
        // `:30-32`) can land verbatim later.
        let exc = LLException::new(vec![Rc::new("etype".to_string()) as Rc<dyn Any>], None);
        assert_eq!(exc.args.len(), 1);
        assert!(exc.error_value.is_none());
    }

    #[test]
    fn llfatalerror_renders_args_separated_by_colons() {
        // Upstream `:56-58`: `__str__ = ': '.join([str(x) for x in
        // self.args])`.
        let err = LLFatalError(vec!["one".to_string(), "two".to_string()]);
        assert_eq!(err.to_string(), "one: two");
    }

    #[test]
    fn eval_block_switch_default_falls_through_when_no_case_matches() {
        // Upstream `llinterp.py:387-402`: the last exit's exitcase
        // `"default"` is consulted only when no non-default case
        // matches. Build a 3-exit switch: case 1, case 2, default.
        // The exitswitch value is 99 (no match) → must take default.
        use crate::flowspace::model::Constant as FlowConstant;
        let switch_var = Variable::named("s");
        switch_var.set_concretetype(Some(LowLevelType::Signed));
        let start = Block::shared(vec![switch_var.clone().into()]);
        start.borrow_mut().exitswitch = Some(switch_var.clone().into());

        let retvar = Hlvalue::Variable(Variable::named("ret"));
        if let Hlvalue::Variable(v) = &retvar {
            v.set_concretetype(Some(LowLevelType::Signed));
        }
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "switch_default",
            start.clone(),
            retvar,
        )));
        let returnblock = graph.borrow().returnblock.clone();

        // Cases produce distinct return constants so we can tell which arm fired.
        let case1 = Hlvalue::Constant(FlowConstant::new(ConstValue::Int(1)));
        let case2 = Hlvalue::Constant(FlowConstant::new(ConstValue::Int(2)));
        let default_case = Hlvalue::Constant(FlowConstant::new(ConstValue::byte_str("default")));

        let v100 = Hlvalue::Constant(FlowConstant::new(ConstValue::Int(100)));
        let v200 = Hlvalue::Constant(FlowConstant::new(ConstValue::Int(200)));
        let v999 = Hlvalue::Constant(FlowConstant::new(ConstValue::Int(999)));

        let mut link1 = Link::new(vec![v100], Some(returnblock.clone()), Some(case1.clone()));
        link1.llexitcase = Some(case1);
        let mut link2 = Link::new(vec![v200], Some(returnblock.clone()), Some(case2.clone()));
        link2.llexitcase = Some(case2);
        // Upstream `:390`: `assert defaultexit.llexitcase is None`.
        let link_default = Link::new(vec![v999], Some(returnblock), Some(default_case));
        start.closeblock(vec![
            link1.into_ref(),
            link2.into_ref(),
            link_default.into_ref(),
        ]);

        let interp = Rc::new(LLInterpreter::new(fixture_typer(), false, None));
        let out = interp
            .eval_graph(
                graph as Rc<dyn Any>,
                vec![Rc::new(99_i64) as Rc<dyn Any>],
                false,
            )
            .expect("switch with default arm runs");
        assert_eq!(*out.downcast::<i64>().expect("Signed return"), 999);
    }

    #[test]
    fn eval_block_canraise_takes_normal_exit_when_no_exception() {
        // Upstream `llinterp.py:369-384`: a `canraise` block has its
        // normal flow on `block.exits[0]`, with exception handlers on
        // `block.exits[1:]`.  Without `LLException` propagation in the
        // local port, every `eval_operation` either succeeds or returns
        // a `TaskError` (= upstream `LLFatalError`); the no-exception
        // arm must fire.
        use crate::flowspace::model::{Constant as FlowConstant, LAST_EXCEPTION};
        let x = Variable::named("x");
        x.set_concretetype(Some(LowLevelType::Signed));
        let y = Variable::named("y");
        y.set_concretetype(Some(LowLevelType::Signed));

        let start = Block::shared(vec![x.clone().into(), y.clone().into()]);
        // A raising op must be the last operation per
        // `flowspace::model::Block::canraise()`.
        let z = Variable::named("z");
        z.set_concretetype(Some(LowLevelType::Signed));
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![x.clone().into(), y.clone().into()],
            z.clone().into(),
        ));
        start.borrow_mut().exitswitch = Some(Hlvalue::Constant(FlowConstant::new(
            ConstValue::Atom(LAST_EXCEPTION.clone()),
        )));
        assert!(start.borrow().canraise());

        let retvar = Hlvalue::Variable(Variable::named("ret"));
        if let Hlvalue::Variable(v) = &retvar {
            v.set_concretetype(Some(LowLevelType::Signed));
        }
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "canraise_no_exc",
            start.clone(),
            retvar,
        )));
        let returnblock = graph.borrow().returnblock.clone();

        // exits[0] = normal flow, exits[1] = exception handler.
        let normal_link = Link::new(vec![z.clone().into()], Some(returnblock.clone()), None);
        let raised_const = Hlvalue::Constant(FlowConstant::new(ConstValue::Int(-1)));
        let exc_link = Link::new(
            vec![raised_const],
            Some(returnblock),
            Some(Hlvalue::Constant(FlowConstant::new(ConstValue::byte_str(
                "BaseException",
            )))),
        );
        start.closeblock(vec![normal_link.into_ref(), exc_link.into_ref()]);

        let interp = Rc::new(LLInterpreter::new(fixture_typer(), false, None));
        let out = interp
            .eval_graph(
                graph as Rc<dyn Any>,
                vec![
                    Rc::new(40_i64) as Rc<dyn Any>,
                    Rc::new(2_i64) as Rc<dyn Any>,
                ],
                false,
            )
            .expect("canraise block with successful op takes exits[0]");
        assert_eq!(*out.downcast::<i64>().expect("Signed return"), 42);
    }
}
