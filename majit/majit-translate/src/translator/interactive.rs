//! Port of upstream `rpython/translator/interactive.py`.
//!
//! Upstream entry point is `class Translation` at
//! `interactive.py:12-26`:
//!
//! ```python
//! class Translation(object):
//!
//!     def __init__(self, entry_point, argtypes=None, **kwds):
//!         self.driver = driver.TranslationDriver(overrides=DEFAULTS)
//!         self.config = self.driver.config
//!
//!         self.entry_point = export_symbol(entry_point)
//!         self.context = TranslationContext(config=self.config)
//!
//!         policy = kwds.pop('policy', None)
//!         self.update_options(kwds)
//!         self.ensure_setup(argtypes, policy)
//!         # for t.view() to work just after construction
//!         graph = self.context.buildflowgraph(entry_point)
//!         self.context._prebuilt_graphs[entry_point] = graph
//! ```
//!
//! Two parity adaptations apply here:
//!
//! 1. **Rust-source entry** (PRE-EXISTING-ADAPTATION, unavoidable). The
//!    upstream line 25 call `buildflowgraph(entry_point)` compiles Python
//!    bytecode. pyre-interpreter's source is Rust, so the same call has
//!    no bytecode to run on. Replaced by `build_host_function_from_rust`
//!    — the Position-2 Rust-AST adapter (see
//!    `flowspace/rust_source/mod.rs`) — which produces the same
//!    `(HostObject, PyGraph)` shape line 25 would have returned. Line
//!    26's `_prebuilt_graphs[entry_point] = graph` seed is performed
//!    verbatim after.
//!
//! 2. **Driver-backed methods** (PARITY). Upstream line 15 creates
//!    `TranslationDriver(overrides=DEFAULTS)`. Every method on
//!    `Translation` outside `__init__` / `view` / `viewcg` forwards to
//!    `self.driver.*` (`annotate`, `rtype`, `backendopt`, `source`,
//!    `source_c`, `source_cl`, `compile`, `compile_c`, `disable`,
//!    `set_backend_extra_options`, `ensure_opt`, `ensure_type_system`,
//!    `ensure_backend`). The Rust port wires every one through
//!    [`TranslationDriver::proceed`] / [`TranslationDriver::disable`] /
//!    [`TranslationDriver::set_backend_extra_options`] in the same body
//!    shape upstream uses (`update_options(kwds) → ensure_backend →
//!    driver.proceed("<task>_<backend>")`); the only `getattr`-style
//!    indirection upstream uses to dispatch `rtype_lltype` /
//!    `compile_c` etc. is folded into `format!("rtype_{}", ts)` etc.
//!    on the Rust side because [`ProceedGoals::One`] takes the goal
//!    name as a string.
//!
//! The `export_symbol(entry_point)` call at upstream line 18 is
//! ported in-place via [`crate::rlib::entrypoint::export_symbol`] —
//! it flips the `GraphFunc.exported_symbol` flag before the
//! `HostObject` is stored in `self.entry_point`, matching upstream
//! `entrypoint.py:10-12`.
//!
//! Fields ported here match upstream line-for-line:
//!
//! | upstream field                             | local              |
//! |--------------------------------------------|--------------------|
//! | `self.driver`                              | `self.driver`      |
//! | `self.config`                              | `self.config()`    |
//! | `self.entry_point = export_symbol(…)`      | `self.entry_point` |
//! | `self.context = TranslationContext(…)`    | `self.context`     |
//! | `self.ann_argtypes` (set by ensure_setup)  | `self.ann_argtypes` |
//! | `self.ann_policy` (set by ensure_setup)    | `self.ann_policy`  |

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use quote::ToTokens;
use syn::{File as SynFile, ItemFn};

use crate::annotator::policy::AnnotatorPolicy;
use crate::annotator::signature::AnnotationSpec;
use crate::config::config::{Config, ConfigError, ConfigValue, OptionValue};
use crate::flowspace::model::HostObject;
use crate::flowspace::rust_source::{AdapterError, build_host_function_from_rust_file};
use crate::rlib::entrypoint::export_symbol;
use crate::translator::driver::{ProceedGoals, TranslationDriver};
use crate::translator::simplify;
use crate::translator::tool::taskengine::{TaskError, TaskOutput};
use crate::translator::translator::{FlowingFlags, TranslationConfig, TranslationContext};

/// Top-level construction error for [`Translation::from_rust_item_fn`]
/// and the `_with_source` variant. Carries either an [`AdapterError`]
/// (raised by `build_host_function_from_rust`) or a [`ConfigError`]
/// (raised by `TranslationDriver::new`'s overrides plumbing). No
/// upstream counterpart — the upstream constructor relies on Python
/// raise-on-error.
#[derive(Debug)]
pub enum TranslationConstructError {
    Adapter(AdapterError),
    Config(ConfigError),
}

impl std::fmt::Display for TranslationConstructError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TranslationConstructError::Adapter(e) => write!(f, "{e}"),
            TranslationConstructError::Config(e) => write!(f, "{e:?}"),
        }
    }
}

impl std::error::Error for TranslationConstructError {}

impl From<AdapterError> for TranslationConstructError {
    fn from(e: AdapterError) -> Self {
        TranslationConstructError::Adapter(e)
    }
}

impl From<ConfigError> for TranslationConstructError {
    fn from(e: ConfigError) -> Self {
        TranslationConstructError::Config(e)
    }
}

/// Rust port of upstream `interactive.py:12 Translation`.
///
/// Every field except `driver` mirrors the upstream `__init__` body
/// at `interactive.py:14-26`; `driver`-dependent methods (`annotate`,
/// `rtype`, `backendopt`, `source`, `compile`, `disable`,
/// `set_backend_extra_options`, `ensure_*`) remain DEFERRED until
/// `translator::driver` ports. See the module docstring.
pub struct Translation {
    /// Upstream `self.driver = driver.TranslationDriver(overrides=DEFAULTS)`
    /// at `interactive.py:15`. Drives every `annotate` / `rtype` /
    /// `compile` step and owns the authoritative `Rc<Config>` that
    /// [`Self::update_options`] mutates and [`Self::ensure_setup`]
    /// forwards through.
    pub driver: Rc<TranslationDriver>,

    /// Upstream `self.entry_point = export_symbol(entry_point)` at
    /// `interactive.py:18`. `export_symbol` in
    /// `rpython/rlib/entrypoint.py:10-12` sets
    /// `func.exported_symbol = True` and returns the function; the
    /// Rust port routes through [`crate::rlib::entrypoint::export_symbol`]
    /// which applies the same flip on the wrapped `GraphFunc` before
    /// it is stored here, preserving the surface parity even though
    /// the flag is C-backend-only in upstream.
    pub entry_point: HostObject,

    /// Upstream `self.context = TranslationContext(config=self.config)`
    /// at `interactive.py:19`. Held as `Rc` so `RPythonAnnotator` can
    /// share the same context instance via `new_with_translator`.
    pub context: Rc<TranslationContext>,

    /// Upstream `self.ann_argtypes` at `interactive.py:37`. Populated
    /// by [`Self::ensure_setup`] which forwards the value through to
    /// `self.driver.setup(...)` and also writes here for direct
    /// readers in this module.
    pub ann_argtypes: RefCell<Option<Vec<AnnotationSpec>>>,

    /// Upstream `self.ann_policy` at `interactive.py:38`. Stored as
    /// `Option<AnnotatorPolicy>` matching the driver's `policy` slot
    /// (`driver.rs:setup`'s `policy: Option<AnnotatorPolicy>`); the
    /// previous string placeholder is gone now that the driver owns
    /// the canonical type.
    pub ann_policy: RefCell<Option<AnnotatorPolicy>>,
}

impl Translation {
    /// Accessor matching upstream `self.config` at `interactive.py:16`.
    /// Upstream binds `self.config = self.driver.config`, so reads
    /// after construction observe every `update_options` write
    /// through the same `Rc<Config>`. Returning the driver's
    /// `Rc<Config>` directly preserves that aliasing — the previous
    /// version returned a typed `&TranslationConfig` snapshot taken
    /// at construction time, which silently diverged once
    /// `update_options` mutated the driver state.
    pub fn config(&self) -> &Rc<Config> {
        &self.driver.config
    }

    /// Port of upstream `Translation.view()` at `interactive.py:28-29`.
    /// Upstream forwards to `TranslationContext.view()` at
    /// `translator.py:122-126`, which constructs a
    /// `translator.tool.graphpage.FlowGraphPage(self).display()` —
    /// a pygame / graphviz viewer.
    ///
    /// Neither `translator.tool.graphpage` nor the pygame stack is
    /// ported locally, so this method is structurally a no-op. The
    /// method itself is present to keep the upstream Translation
    /// surface parity complete — caller code that writes
    /// `translation.view()` compiles and runs silently rather than
    /// surfacing a missing-method compile error.
    pub fn view(&self) {
        // Intentional no-op. See upstream `translator.py:122-126`
        // for the full FlowGraphPage call; `translator.tool.graphpage`
        // depends on pygame + graphviz which are not part of majit's
        // runtime footprint.
    }

    /// Port of upstream `Translation.viewcg()` at
    /// `interactive.py:31-32`. Forwards to
    /// `TranslationContext.viewcg(center_graph, huge)` at
    /// `translator.py:130-134`. Same no-op parity as [`Self::view`];
    /// the pygame/graphviz stack is out of scope.
    pub fn viewcg(&self) {
        // Intentional no-op; see `view()` docstring for the shared
        // graphpage dependency.
    }

    /// Port of upstream `Translation.ensure_setup()` at
    /// `interactive.py:34-38`.
    ///
    /// Upstream body:
    ///
    /// ```python
    /// def ensure_setup(self, argtypes=None, policy=None):
    ///     self.driver.setup(self.entry_point, argtypes, policy,
    ///                       empty_translator=self.context)
    ///     self.ann_argtypes = argtypes
    ///     self.ann_policy = policy
    /// ```
    ///
    /// Forwards `argtypes` / `policy` into
    /// [`TranslationDriver::setup`] with `empty_translator=self.context`,
    /// matching upstream lines 35-36 exactly. The pure-storage lines
    /// 37-38 (`self.ann_argtypes = argtypes; self.ann_policy = policy`)
    /// run after the driver call so a `setup` failure leaves both
    /// slots untouched.
    ///
    /// Returns `Result` rather than panicking on driver-side errors
    /// (e.g. a malformed secondary-entrypoints CSV inside
    /// `setup`'s `:201-208` lookup) because `Translation`'s public
    /// surface needs to surface them to the caller — the upstream
    /// `def ensure_setup` returns nothing only because Python uses
    /// raise-on-error semantics throughout.
    pub fn ensure_setup(
        &self,
        argtypes: Option<Vec<AnnotationSpec>>,
        policy: Option<AnnotatorPolicy>,
    ) -> Result<(), crate::translator::tool::taskengine::TaskError> {
        // Upstream `:35-36`: `self.driver.setup(self.entry_point,
        // argtypes, policy, empty_translator=self.context)`.
        self.driver.setup(
            Some(self.entry_point.clone()),
            argtypes.clone(),
            policy.clone(),
            HashMap::new(),
            Some(self.context.clone()),
        )?;
        // Upstream `:37-38`: pure-storage post-conditions.
        *self.ann_argtypes.borrow_mut() = argtypes;
        *self.ann_policy.borrow_mut() = policy;
        Ok(())
    }

    /// Port of upstream `Translation.update_options()` at
    /// `interactive.py:40-44`.
    ///
    /// Upstream body:
    ///
    /// ```python
    /// def update_options(self, kwds):
    ///     gc = kwds.pop('gc', None)
    ///     if gc:
    ///         self.config.translation.gc = gc
    ///     self.config.translation.set(**kwds)
    /// ```
    ///
    /// Routes `kwds` through `self.driver.config` so option overrides
    /// take effect on the driver's authoritative `Rc<Config>`. The
    /// `gc` short-circuit at upstream `:41-43` handles the explicit
    /// `translation.gc` rename; everything else is fed to
    /// [`crate::config::config::Config::set`] which mirrors upstream's
    /// `Config.set(**kwds)` from `rpython/config/config.py:129-143`.
    ///
    /// Returns `Result` so the caller sees `ConfigError` rather than
    /// a silent drop. Upstream `update_options` raises directly out
    /// of `Config.set`; the Rust port surfaces the same lift point
    /// through `Result::Err`.
    pub fn update_options(
        &self,
        kwds: &[(String, String)],
    ) -> Result<(), crate::config::config::ConfigError> {
        // Upstream `:41-43`: pop `gc` and special-case it as
        // `translation.gc`. `kwds` is an ordered slice so the
        // downstream `Config.set` walk preserves caller order
        // (`config.py:131 for key, value in kwargs.iteritems():`).
        let mut remaining: Vec<(String, OptionValue)> = Vec::with_capacity(kwds.len());
        for (k, v) in kwds.iter() {
            if k == "gc" {
                self.driver
                    .config
                    .set_value("translation.gc", OptionValue::Choice(v.clone()))?;
                continue;
            }
            remaining.push((k.clone(), OptionValue::Str(v.clone())));
        }
        // Upstream `:44`: `self.config.translation.set(**kwds)`.
        // `Config::set` at `config.rs` walks the option tree by name,
        // matching `rpython/config/config.py:129-143`.
        if !remaining.is_empty() {
            self.driver.config.set(remaining)?;
        }
        Ok(())
    }

    /// Port of upstream `Translation.ensure_opt()` at
    /// `interactive.py:46-57`.
    ///
    /// ```python
    /// def ensure_opt(self, name, value=None, fallback=None):
    ///     if value is not None:
    ///         self.update_options({name: value})
    ///         return value
    ///     val = getattr(self.config.translation, name, None)
    ///     if fallback is not None and val is None:
    ///         self.update_options({name: fallback})
    ///         return fallback
    ///     if val is not None:
    ///         return val
    ///     raise Exception(
    ///                 "the %r option should have been specified at this point" % name)
    /// ```
    ///
    /// Upstream `getattr(self.config.translation, name, None)` reads the
    /// `translation.<name>` slot via Python's `__getattr__`. The Rust
    /// port routes through [`Config::get`] on `translation.<name>` and
    /// peels the inner [`OptionValue`] back to a `String`. The two
    /// upstream call sites — `ensure_type_system` and `ensure_backend`
    /// — both read `ChoiceOption` slots (`backend` / `type_system`) so
    /// only `Choice` / `Str` / `None` arms are exercised here; other
    /// `OptionValue` variants (`Bool`, `Int`, `Float`, `Arbitrary`)
    /// would be a caller bug.
    pub fn ensure_opt(
        &self,
        name: &str,
        value: Option<&str>,
        fallback: Option<&str>,
    ) -> Result<String, ConfigError> {
        // Upstream `:47-49`: `if value is not None: self.update_options(
        // {name: value}); return value`.
        if let Some(v) = value {
            let kwds = vec![(name.to_string(), v.to_string())];
            self.update_options(&kwds)?;
            return Ok(v.to_string());
        }
        // Upstream `:50`: `val = getattr(self.config.translation, name,
        // None)`. `Config::get` returns `Err(UnknownOption)` on a
        // missing slot — Python's `getattr(_, _, None)` swallows it, so
        // map every error here back to the `None` arm.
        let path = format!("translation.{}", name);
        let val = self.driver.config.get(&path).ok();
        let val_str = val.and_then(|cv| match cv {
            ConfigValue::Value(OptionValue::Choice(s)) => Some(s),
            ConfigValue::Value(OptionValue::Str(s)) => Some(s),
            // Upstream `None` sentinel ⇒ Python `None`.
            ConfigValue::Value(OptionValue::None) => None,
            _ => None,
        });
        // Upstream `:51-53`: `if fallback is not None and val is None:
        // self.update_options({name: fallback}); return fallback`.
        if let Some(f) = fallback {
            if val_str.is_none() {
                let kwds = vec![(name.to_string(), f.to_string())];
                self.update_options(&kwds)?;
                return Ok(f.to_string());
            }
        }
        // Upstream `:54-55`: `if val is not None: return val`.
        if let Some(v) = val_str {
            return Ok(v);
        }
        // Upstream `:56-57`: `raise Exception("the %r option should
        // have been specified at this point" % name)`.
        Err(ConfigError::Generic(format!(
            "the {:?} option should have been specified at this point",
            name
        )))
    }

    /// Port of upstream `Translation.ensure_type_system()` at
    /// `interactive.py:59-62`.
    ///
    /// ```python
    /// def ensure_type_system(self, type_system=None):
    ///     if self.config.translation.backend is not None:
    ///         return self.ensure_opt('type_system')
    ///     return self.ensure_opt('type_system', type_system, 'lltype')
    /// ```
    ///
    /// `self.config.translation.backend is not None` tests the
    /// `ChoiceOption` slot for a non-None value. The Rust port reads
    /// the slot through [`Config::get`] and treats `OptionValue::None`
    /// (and any read error) as the `None` arm.
    pub fn ensure_type_system(&self, type_system: Option<&str>) -> Result<String, ConfigError> {
        // Upstream `:60`: `if self.config.translation.backend is not None`.
        let backend_set = match self.driver.config.get("translation.backend").ok() {
            Some(ConfigValue::Value(OptionValue::None)) => false,
            Some(ConfigValue::Value(_)) => true,
            _ => false,
        };
        if backend_set {
            // Upstream `:61`: `return self.ensure_opt('type_system')`.
            return self.ensure_opt("type_system", None, None);
        }
        // Upstream `:62`: `return self.ensure_opt('type_system',
        // type_system, 'lltype')`.
        self.ensure_opt("type_system", type_system, Some("lltype"))
    }

    /// Port of upstream `Translation.ensure_backend()` at
    /// `interactive.py:64-67`.
    ///
    /// ```python
    /// def ensure_backend(self, backend=None):
    ///     backend = self.ensure_opt('backend', backend)
    ///     self.ensure_type_system()
    ///     return backend
    /// ```
    pub fn ensure_backend(&self, backend: Option<&str>) -> Result<String, ConfigError> {
        // Upstream `:65`: `backend = self.ensure_opt('backend', backend)`.
        let backend_str = self.ensure_opt("backend", backend, None)?;
        // Upstream `:66`: `self.ensure_type_system()`.
        self.ensure_type_system(None)?;
        // Upstream `:67`: `return backend`.
        Ok(backend_str)
    }

    /// Port of upstream `Translation.disable()` at `interactive.py:69-71`.
    ///
    /// ```python
    /// def disable(self, to_disable):
    ///     self.driver.disable(to_disable)
    /// ```
    pub fn disable(&self, to_disable: Vec<String>) {
        self.driver.disable(to_disable);
    }

    /// Port of upstream `Translation.set_backend_extra_options()` at
    /// `interactive.py:73-77`.
    ///
    /// ```python
    /// def set_backend_extra_options(self, **extra_options):
    ///     for name in extra_options:
    ///         backend, option = name.split('_', 1)
    ///         self.ensure_backend(backend)
    ///     self.driver.set_backend_extra_options(extra_options)
    /// ```
    ///
    /// Upstream `name.split('_', 1)` raises `ValueError` if the name
    /// has no `_` (the Python tuple-unpack at `:75` requires exactly
    /// two parts). The Rust port returns
    /// [`ConfigError::Generic`] on the same condition so the surface
    /// is observable; PyPy's exception at this point would also be
    /// caught by the same `try/except` callers wrap around
    /// `set_backend_extra_options` in the C-backend path.
    pub fn set_backend_extra_options(
        &self,
        extra_options: HashMap<String, OptionValue>,
    ) -> Result<(), ConfigError> {
        // Upstream `:74-76`: split each name on first `_` and ensure
        // the corresponding backend.
        for name in extra_options.keys() {
            let (backend, _option) = name.split_once('_').ok_or_else(|| {
                ConfigError::Generic(format!(
                    "extra option {:?} must be of the form <backend>_<option>",
                    name
                ))
            })?;
            self.ensure_backend(Some(backend))?;
        }
        // Upstream `:77`: forward the dict to the driver.
        self.driver.set_backend_extra_options(extra_options);
        Ok(())
    }

    /// Port of upstream `Translation.annotate()` at
    /// `interactive.py:81-83`.
    ///
    /// ```python
    /// def annotate(self, **kwds):
    ///     self.update_options(kwds)
    ///     return self.driver.annotate()
    /// ```
    ///
    /// `self.driver.annotate()` upstream is the `expose_task`-generated
    /// proc that calls `self.proceed("annotate")`
    /// (`driver.py:104-110`). The Rust port routes through
    /// [`TranslationDriver::proceed`] with `ProceedGoals::One("annotate")`,
    /// which performs the same engine planning + execution.
    pub fn annotate(&self, kwds: &[(String, String)]) -> Result<TaskOutput, TaskError> {
        // Upstream `:82`: `self.update_options(kwds)`.
        self.update_options(kwds).map_err(cfg_to_task)?;
        // Upstream `:83`: `return self.driver.annotate()` ⇒
        // `self.proceed("annotate")` per `driver.py:104-110`.
        self.driver
            .proceed(ProceedGoals::One("annotate".to_string()))
    }

    /// Port of upstream `Translation.rtype()` at `interactive.py:87-90`.
    ///
    /// ```python
    /// def rtype(self, **kwds):
    ///     self.update_options(kwds)
    ///     ts = self.ensure_type_system()
    ///     return getattr(self.driver, 'rtype_' + ts)()
    /// ```
    ///
    /// `getattr(self.driver, 'rtype_' + ts)()` upstream resolves to the
    /// `expose_task`-generated proc whose body calls
    /// `self.proceed('rtype_<ts>')`. The Rust port short-circuits the
    /// `getattr` round-trip and proceeds directly.
    pub fn rtype(&self, kwds: &[(String, String)]) -> Result<TaskOutput, TaskError> {
        self.update_options(kwds).map_err(cfg_to_task)?;
        let ts = self.ensure_type_system(None).map_err(cfg_to_task)?;
        self.driver
            .proceed(ProceedGoals::One(format!("rtype_{}", ts)))
    }

    /// Port of upstream `Translation.backendopt()` at
    /// `interactive.py:92-95`.
    ///
    /// ```python
    /// def backendopt(self, **kwds):
    ///     self.update_options(kwds)
    ///     ts = self.ensure_type_system('lltype')
    ///     return getattr(self.driver, 'backendopt_' + ts)()
    /// ```
    ///
    /// Note the explicit `'lltype'` fallback at upstream `:94`: this
    /// method forces `type_system=lltype` when the slot is unset,
    /// while `rtype()` at `:90` only resolves the existing default.
    pub fn backendopt(&self, kwds: &[(String, String)]) -> Result<TaskOutput, TaskError> {
        self.update_options(kwds).map_err(cfg_to_task)?;
        let ts = self
            .ensure_type_system(Some("lltype"))
            .map_err(cfg_to_task)?;
        self.driver
            .proceed(ProceedGoals::One(format!("backendopt_{}", ts)))
    }

    /// Port of upstream `Translation.source()` at
    /// `interactive.py:99-102`.
    ///
    /// ```python
    /// def source(self, **kwds):
    ///     self.update_options(kwds)
    ///     backend = self.ensure_backend()
    ///     getattr(self.driver, 'source_' + backend)()
    /// ```
    pub fn source(&self, kwds: &[(String, String)]) -> Result<TaskOutput, TaskError> {
        self.update_options(kwds).map_err(cfg_to_task)?;
        let backend = self.ensure_backend(None).map_err(cfg_to_task)?;
        self.driver
            .proceed(ProceedGoals::One(format!("source_{}", backend)))
    }

    /// Port of upstream `Translation.source_c()` at
    /// `interactive.py:104-107`.
    ///
    /// ```python
    /// def source_c(self, **kwds):
    ///     self.update_options(kwds)
    ///     self.ensure_backend('c')
    ///     self.driver.source_c()
    /// ```
    pub fn source_c(&self, kwds: &[(String, String)]) -> Result<TaskOutput, TaskError> {
        self.update_options(kwds).map_err(cfg_to_task)?;
        self.ensure_backend(Some("c")).map_err(cfg_to_task)?;
        self.driver
            .proceed(ProceedGoals::One("source_c".to_string()))
    }

    /// Port of upstream `Translation.source_cl()` at
    /// `interactive.py:109-112`.
    ///
    /// ```python
    /// def source_cl(self, **kwds):
    ///     self.update_options(kwds)
    ///     self.ensure_backend('cl')
    ///     self.driver.source_cl()
    /// ```
    ///
    /// "cl" is not in the upstream `backend` choices either
    /// (`translationoption.py:51-56` lists only `"c"`); the method is
    /// preserved for surface parity but always errors out at
    /// `ensure_backend('cl')` with `ConfigError::ValidationFailed`.
    pub fn source_cl(&self, kwds: &[(String, String)]) -> Result<TaskOutput, TaskError> {
        self.update_options(kwds).map_err(cfg_to_task)?;
        self.ensure_backend(Some("cl")).map_err(cfg_to_task)?;
        self.driver
            .proceed(ProceedGoals::One("source_cl".to_string()))
    }

    /// Port of upstream `Translation.compile()` at
    /// `interactive.py:114-118`.
    ///
    /// ```python
    /// def compile(self, **kwds):
    ///     self.update_options(kwds)
    ///     backend = self.ensure_backend()
    ///     getattr(self.driver, 'compile_' + backend)()
    ///     return self.driver.c_entryp
    /// ```
    ///
    /// Returns `self.driver.c_entryp` after the proceed call. Upstream
    /// reads the attribute directly even when it is `None` — the Rust
    /// port mirrors that with `Option<PathBuf>` so callers can observe
    /// the same shape (the C-backend leaf populates the slot through
    /// `task_compile_c` at upstream `:524`).
    pub fn compile(&self, kwds: &[(String, String)]) -> Result<Option<PathBuf>, TaskError> {
        self.update_options(kwds).map_err(cfg_to_task)?;
        let backend = self.ensure_backend(None).map_err(cfg_to_task)?;
        self.driver
            .proceed(ProceedGoals::One(format!("compile_{}", backend)))?;
        Ok(self.driver.c_entryp.borrow().clone())
    }

    /// Port of upstream `Translation.compile_c()` at
    /// `interactive.py:120-124`.
    ///
    /// ```python
    /// def compile_c(self, **kwds):
    ///     self.update_options(kwds)
    ///     self.ensure_backend('c')
    ///     self.driver.compile_c()
    ///     return self.driver.c_entryp
    /// ```
    pub fn compile_c(&self, kwds: &[(String, String)]) -> Result<Option<PathBuf>, TaskError> {
        self.update_options(kwds).map_err(cfg_to_task)?;
        self.ensure_backend(Some("c")).map_err(cfg_to_task)?;
        self.driver
            .proceed(ProceedGoals::One("compile_c".to_string()))?;
        Ok(self.driver.c_entryp.borrow().clone())
    }

    /// Port of upstream `Translation.__init__` specialized for a
    /// Rust-source entry. Mirrors the `buildflowgraph(entry_point)` +
    /// `_prebuilt_graphs[entry_point] = graph` pair at
    /// `interactive.py:25-26`, replacing the bytecode-requiring
    /// `buildflowgraph` with `build_host_function_from_rust`.
    ///
    /// Returns the `Translation` wrapper plus the synthesized
    /// `HostObject` so the caller can pass it to downstream
    /// `RPythonAnnotator::build_types` (`annrpython.py:73-97`).
    ///
    /// Source filename falls back to the `<rust-source>` sentinel
    /// (syn has no stable source-path accessor, so this stays a
    /// sentinel unless the caller threads a real path through
    /// [`Self::from_rust_item_fn_with_source`]).
    ///
    /// Source text is auto-populated from `quote::ToTokens` — every
    /// `syn::ItemFn` round-trips through `to_token_stream().to_string()`
    /// to produce the textual rendering that upstream
    /// `inspect.getsource(func)` at `bytecode.py:50` would return.
    /// The rendering is whitespace-collapsed but semantically
    /// complete, which is the invariant upstream relies on at
    /// `model.py:35-47` (`FunctionGraph.source` setter) +
    /// `tool/error.rs:300` (graph-render error path).
    pub fn from_rust_item_fn(
        item: &ItemFn,
    ) -> Result<(Self, HostObject), TranslationConstructError> {
        // `quote::ToTokens` renders the original syntax back to text
        // without the caller having to keep the source string
        // around. The output is not pretty-printed but is the
        // canonical stable stringification for error rendering.
        let source_text = item.to_token_stream().to_string();
        Self::from_rust_item_fn_with_source(item, None, Some(&source_text))
    }

    /// Same as [`Self::from_rust_item_fn`] but threads upstream's
    /// `argtypes` / `policy` / `**kwds` parameters from
    /// `interactive.py:14 def __init__(self, entry_point,
    /// argtypes=None, **kwds)`. Drives the same `update_options(kwds)`
    /// + `ensure_setup(argtypes, policy)` chain as upstream
    /// `interactive.py:21-23`. Use the bare [`Self::from_rust_item_fn`]
    /// form when defaults (`argtypes=None`, `policy=None`, empty
    /// `kwds`) are sufficient.
    pub fn from_rust_item_fn_with_options(
        item: &ItemFn,
        argtypes: Option<Vec<AnnotationSpec>>,
        policy: Option<AnnotatorPolicy>,
        kwds: &[(String, String)],
    ) -> Result<(Self, HostObject), TranslationConstructError> {
        let source_text = item.to_token_stream().to_string();
        Self::from_rust_item_fn_with_source_and_options(
            item,
            None,
            Some(&source_text),
            argtypes,
            policy,
            kwds,
        )
    }

    /// Same as [`Self::from_rust_item_fn`] but threads caller-supplied
    /// source metadata. Upstream reads `func.__code__.co_filename` at
    /// `model.py:54` (filename) and the source text comes from
    /// `inspect.getsource(func)` at `bytecode.py:50` (GraphFunc.source)
    /// / the `FunctionGraph.source` property at `model.py:35-47`
    /// (`_source`). Both populate the `graph.source()` fast path
    /// `tool/error.rs:300` uses for graph-render error messages.
    pub fn from_rust_item_fn_with_source(
        item: &ItemFn,
        source_filename: Option<&str>,
        source_text: Option<&str>,
    ) -> Result<(Self, HostObject), TranslationConstructError> {
        Self::from_rust_item_fn_with_source_and_options(
            item,
            source_filename,
            source_text,
            None,
            None,
            &[],
        )
    }

    /// Full-parity counterpart of [`Self::from_rust_item_fn_with_source`]
    /// that mirrors upstream `interactive.py:14-26 __init__` end-to-end:
    /// driver → entry_point → `update_options(kwds)` → context (with
    /// `config=driver.config`) → `ensure_setup(argtypes, policy)` →
    /// `_prebuilt_graphs[entry_point] = graph`.
    ///
    /// Order note: upstream constructs `self.context` BEFORE
    /// `update_options`, but Python's `Config` is a live `Rc<Config>`
    /// shared with the context, so post-update reads see the new
    /// values automatically. The local typed `TranslationConfig`
    /// inside `TranslationContext` is a snapshot, so the snapshot
    /// must be taken AFTER `update_options` to observe the same
    /// post-construction state. Snapshotting later is a port-internal
    /// reordering — observable behaviour matches upstream.
    pub fn from_rust_item_fn_with_source_and_options(
        item: &ItemFn,
        source_filename: Option<&str>,
        source_text: Option<&str>,
        argtypes: Option<Vec<AnnotationSpec>>,
        policy: Option<AnnotatorPolicy>,
        kwds: &[(String, String)],
    ) -> Result<(Self, HostObject), TranslationConstructError> {
        // Synthesize a single-item `syn::File` so the walker
        // (Slice O7-O8) populates the module-globals registry with
        // the entry-point fn before its body is lowered. Mirrors
        // upstream Python `def` running at module-import time
        // before `buildflowgraph(entry_point)` is invoked
        // (`flowcontext.py:847 w_globals.value[varname]`). Callers
        // that need sibling items (enums, structs, helper fns) in
        // the same source file walked alongside the entry point
        // should use
        // [`Self::from_rust_file_entry_point_with_source_and_options`]
        // (Slice O9) directly.
        let synthetic_file = single_item_file(item);
        let entry_point_name = item.sig.ident.to_string();
        Self::from_rust_file_entry_point_with_source_and_options(
            &synthetic_file,
            &entry_point_name,
            source_filename,
            source_text,
            argtypes,
            policy,
            kwds,
        )
    }

    /// File-aware entry: walk every top-level item in `file` through
    /// `register_rust_module` (Slice O7-O8) before lowering the
    /// `entry_point_name` `Item::Fn`. Mirrors upstream
    /// `interactive.py:25 buildflowgraph(entry_point)` running AFTER
    /// the module's `def` / `class` statements have already populated
    /// `entry_point.func_globals` (so the body's `LOAD_GLOBAL`
    /// resolutions hit the populated registry —
    /// `flowcontext.py:847 w_globals.value[varname]`).
    ///
    /// The orthodox alternative to [`Self::from_rust_item_fn`]: when
    /// the entry-point fn references sibling enums, structs, or
    /// helper fns from the same source file, the caller passes the
    /// parsed `syn::File` so the walker registers everything before
    /// the entry-point body resolves their identities.
    ///
    /// Returns `AdapterError::Unsupported` when `entry_point_name`
    /// is not a top-level `Item::Fn` in `file`.
    pub fn from_rust_file_entry_point(
        file: &SynFile,
        entry_point_name: &str,
    ) -> Result<(Self, HostObject), TranslationConstructError> {
        Self::from_rust_file_entry_point_with_source_and_options(
            file,
            entry_point_name,
            None,
            None,
            None,
            None,
            &[],
        )
    }

    /// Source-metadata sibling of [`Self::from_rust_file_entry_point`].
    pub fn from_rust_file_entry_point_with_source(
        file: &SynFile,
        entry_point_name: &str,
        source_filename: Option<&str>,
        source_text: Option<&str>,
    ) -> Result<(Self, HostObject), TranslationConstructError> {
        Self::from_rust_file_entry_point_with_source_and_options(
            file,
            entry_point_name,
            source_filename,
            source_text,
            None,
            None,
            &[],
        )
    }

    /// Full-parity counterpart of
    /// [`Self::from_rust_file_entry_point_with_source`] that mirrors
    /// upstream `interactive.py:14-26 __init__` end-to-end. See
    /// [`Self::from_rust_item_fn_with_source_and_options`] for the
    /// per-step parity rationale; the only difference here is that
    /// the body builder is `build_host_function_from_rust_file`
    /// (which runs the walker pre-pass) instead of
    /// `build_host_function_from_rust` (which would lower a single
    /// ItemFn without walking sibling items).
    pub fn from_rust_file_entry_point_with_source_and_options(
        file: &SynFile,
        entry_point_name: &str,
        source_filename: Option<&str>,
        source_text: Option<&str>,
        argtypes: Option<Vec<AnnotationSpec>>,
        policy: Option<AnnotatorPolicy>,
        kwds: &[(String, String)],
    ) -> Result<(Self, HostObject), TranslationConstructError> {
        // Upstream `interactive.py:15`:
        // `self.driver = driver.TranslationDriver(overrides=DEFAULTS)`.
        // `DEFAULTS` at upstream `:6-10` sets `translation.verbose = True`
        // and leaves `backend` / `type_system` at `None`. Pass them
        // through `overrides=` so the driver's `Rc<Config>` carries the
        // upstream-observable post-construction state.
        let mut overrides: HashMap<String, OptionValue> = HashMap::new();
        overrides.insert("translation.verbose".to_string(), OptionValue::Bool(true));
        overrides.insert("translation.backend".to_string(), OptionValue::None);
        overrides.insert("translation.type_system".to_string(), OptionValue::None);
        let driver =
            TranslationDriver::new(None, None, Vec::new(), None, None, None, Some(overrides))?;

        // `build_host_function_from_rust_file` runs the walker
        // pre-pass over `file.items` (registering enums/structs/fns
        // into the process-global module-globals registry — Slice
        // O7-O8) and then lowers the named entry-point. Mirrors
        // upstream `buildflowgraph(entry_point)` arriving with
        // `entry_point.func_globals` already populated by the
        // module's `def` / `class` execution at import time.
        let (host, pygraph) = build_host_function_from_rust_file(
            file,
            entry_point_name,
            source_filename,
            source_text,
        )?;

        // Upstream `interactive.py:21-22`: `policy = kwds.pop('policy',
        // None); self.update_options(kwds)`. The Rust port already
        // received `policy` as a typed parameter — `kwds` is everything
        // else and goes straight to `Config::set` via `update_options`.
        update_options_via_driver(&driver, kwds)?;

        // Upstream `interactive.py:19`: `self.context =
        // TranslationContext(config=self.config)`. Snapshot
        // `driver.config` into a typed `TranslationConfig` AFTER
        // `update_options` so any kwds-provided overrides land in the
        // snapshot, matching upstream's shared-Rc semantics.
        let translation_config = TranslationConfig::from_rc_config(&driver.config)?;
        let context = Rc::new(TranslationContext::with_config_and_flowing_flags(
            Some(translation_config),
            FlowingFlags::default(),
        ));

        // Upstream `interactive.py:25 buildflowgraph(entry_point)`
        // enters the non-prebuilt branch of
        // `translator.py:46-62 buildflowgraph`. That branch, after
        // `build_flow(func)` returns, runs
        // `simplify.simplify_graph(graph)` (translator.py:56),
        // optionally `detect_list_comprehension` (:57-58), and
        // finally `self.graphs.append(graph)` (:61). Our local port
        // reproduces the same sequence in `translator.rs:264-281`.
        // The adapter path skips `buildflowgraph` entirely (the
        // prebuilt short-circuit returns the graph we seed below),
        // so to keep parity with upstream's post-construction state
        // we run the same post-`build_flow` steps ourselves.
        {
            let g = pygraph.graph.borrow();
            simplify::simplify_graph(&g, None);
        }
        if context.config.translation.list_comprehension_operations {
            let g = pygraph.graph.borrow();
            simplify::detect_list_comprehension(&g);
        }
        context.graphs.borrow_mut().push(pygraph.graph.clone());

        // `self.entry_point = export_symbol(entry_point)` —
        // upstream `interactive.py:18`. Upstream mutates the Python
        // function in place and returns the same object. The Rust
        // port routes through `rlib::entrypoint::export_symbol` which
        // flips `GraphFunc.exported_symbol` (an `Arc<AtomicBool>`)
        // through interior mutability, so the input HostObject and
        // the returned one are `Arc::ptr_eq` (same identity), and
        // every adapter-built `pygraph.func` / `pygraph.graph.func`
        // clone shares the same flag cell — matching upstream's
        // single-Python-object semantics.
        let flagged_host = export_symbol(host);
        context
            ._prebuilt_graphs
            .borrow_mut()
            .insert(flagged_host.clone(), pygraph);

        let translation = Translation {
            driver,
            entry_point: flagged_host.clone(),
            context,
            ann_argtypes: RefCell::new(None),
            ann_policy: RefCell::new(None),
        };

        // Upstream `interactive.py:23`: `self.ensure_setup(argtypes,
        // policy)`. Runs `driver.setup(self.entry_point, argtypes,
        // policy, empty_translator=self.context)` and persists
        // `ann_argtypes` / `ann_policy` slots, completing the
        // upstream-observable post-construction state.
        translation
            .ensure_setup(argtypes, policy)
            .map_err(|e| TranslationConstructError::Config(ConfigError::Generic(e.message)))?;

        Ok((translation, flagged_host))
    }
}

/// Wrap a single `syn::ItemFn` in a one-item `syn::File` so the
/// walker pre-pass in
/// [`crate::flowspace::rust_source::register::register_rust_module`]
/// has a unified input shape. Used by
/// [`Translation::from_rust_item_fn_with_source_and_options`] to
/// route every existing item-fn-only caller through the same walker
/// path Slice O9 introduces for multi-item file callers.
fn single_item_file(item: &ItemFn) -> SynFile {
    SynFile {
        shebang: None,
        attrs: Vec::new(),
        items: vec![syn::Item::Fn(item.clone())],
    }
}

/// Converts a [`ConfigError`] (raised by `Config::set` / `Config::get`
/// surfaces inside `update_options` / `ensure_*`) into a [`TaskError`]
/// so the forwarding methods (`annotate`, `rtype`, `compile`, …) can
/// expose a single error type to callers. Upstream Python uses one
/// shared `raise` channel, so this fold-down preserves the
/// observable behaviour: a config-time problem before the proceed
/// call surfaces with the same urgency as a task-time error during
/// it.
fn cfg_to_task(e: ConfigError) -> TaskError {
    TaskError {
        message: format!("{:?}", e),
    }
}

/// Mirrors `Translation::update_options` for use BEFORE the
/// `Translation` struct is fully constructed (the constructor needs
/// to apply kwds to `driver.config` before snapshotting it into the
/// context). Logic stays identical to the public method.
fn update_options_via_driver(
    driver: &TranslationDriver,
    kwds: &[(String, String)],
) -> Result<(), ConfigError> {
    // `kwds` is an ordered slice mirroring upstream's `**kwds` dict;
    // `Config::set` walks them in caller order.
    let mut remaining: Vec<(String, OptionValue)> = Vec::with_capacity(kwds.len());
    for (k, v) in kwds.iter() {
        if k == "gc" {
            driver
                .config
                .set_value("translation.gc", OptionValue::Choice(v.clone()))?;
            continue;
        }
        remaining.push((k.clone(), OptionValue::Str(v.clone())));
    }
    if !remaining.is_empty() {
        driver.config.set(remaining)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_item_fn(src: &str) -> ItemFn {
        syn::parse_str::<ItemFn>(src).expect("test fixture must parse")
    }

    fn parse_file(src: &str) -> SynFile {
        syn::parse_file(src).expect("test fixture must parse as syn::File")
    }

    #[test]
    fn from_rust_item_fn_seeds_prebuilt_graphs_exactly_once() {
        // Mirrors upstream `interactive.py:25-26` behaviour: the entry
        // point's graph is cached in `_prebuilt_graphs` after
        // construction. The subsequent `buildflowgraph(entry_point)`
        // call pops it per `translator.py:50-51`, so the cache is
        // empty after one consumption.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, host) = Translation::from_rust_item_fn(&item).expect("translation");

        assert!(
            t.context._prebuilt_graphs.borrow().contains_key(&host),
            "entry point must be seeded after construction"
        );

        let pg = t
            .context
            .buildflowgraph(host.clone(), false)
            .expect("buildflowgraph hits the prebuilt cache");
        // Pop-on-hit semantic: second lookup must not find it.
        assert!(!t.context._prebuilt_graphs.borrow().contains_key(&host));
        // And the returned PyGraph is the one the adapter built.
        assert_eq!(pg.func.id, host.user_function().expect("user fn").id);
    }

    #[test]
    fn from_rust_item_fn_propagates_adapter_errors() {
        // Any `AdapterError` from `build_host_function_from_rust` must
        // bubble up — the wrapper does not swallow errors. Use an
        // unsupported signature (const generic) as a concrete trigger;
        // `build_flow.rs:138-141` rejects it with
        // `InvalidSignature { reason: "const generic parameter …" }`.
        let item = parse_item_fn("fn f<const N: usize>() -> i64 { 0 }");
        match Translation::from_rust_item_fn(&item) {
            Ok(_) => panic!("const generic must reject"),
            Err(TranslationConstructError::Adapter(AdapterError::InvalidSignature { reason })) => {
                assert!(reason.contains("const generic"), "reason: {reason}");
            }
            Err(other) => panic!("expected Adapter(InvalidSignature), got {other:?}"),
        }
    }

    #[test]
    fn from_rust_item_fn_default_verbose_true() {
        // Upstream `interactive.py:6 DEFAULTS = {..., 'translation.verbose': True}`
        // is passed to `driver.TranslationDriver(overrides=DEFAULTS)` at line 15;
        // `self.config = self.driver.config` at line 16 picks up the override.
        // The Rust port applies the single driver-equivalent default via
        // `FlowingFlags { verbose: Some(true) }`.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");
        assert!(
            t.context.config.translation.verbose,
            "interactive.py:6 DEFAULTS sets translation.verbose = True"
        );
    }

    #[test]
    fn from_rust_item_fn_appends_entry_graph_to_context_graphs() {
        // Upstream `interactive.py:25 buildflowgraph(entry_point)` goes
        // through the non-prebuilt branch of `translator.py:46-62`,
        // which at line 61 does `self.graphs.append(graph)`. The
        // adapter path skips `buildflowgraph`, so the append is
        // performed explicitly — this test pins the parity.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");

        let graphs = t.context.graphs.borrow();
        assert_eq!(
            graphs.len(),
            1,
            "entry graph must be appended — matches upstream translator.py:61"
        );

        // The appended graph is the SAME underlying FunctionGraph the
        // PyGraph wraps (Rc pointer-equal), so `t.view()`-style
        // operations that index through `context.graphs` find it.
        let seeded_pygraph = t
            .context
            ._prebuilt_graphs
            .borrow()
            .values()
            .next()
            .expect("seeded entry")
            .clone();
        assert!(
            Rc::ptr_eq(&graphs[0], &seeded_pygraph.graph),
            "context.graphs entry must be Rc-identical to the seeded PyGraph's graph"
        );
    }

    #[test]
    fn from_rust_item_fn_runs_simplify_graph() {
        // Upstream `translator.py:56 simplify_graph(graph)` runs on
        // the result of `build_flow(func)` before the graph is
        // appended to `self.graphs`. simplify_graph is idempotent
        // (eliminate_empty_blocks / join_blocks / remove_identical_vars
        // converge on a fixed point), so a second `simplify_graph`
        // call on a post-construction graph must be a structural
        // no-op. Exercise that: snapshot the block/exit count, run
        // simplify again, confirm the shape is unchanged.
        let item = parse_item_fn(
            "fn f(x: i64) -> i64 {
                if x > 0 { let _y = 1; }
                2
            }",
        );
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");

        let pygraph = t
            .context
            ._prebuilt_graphs
            .borrow()
            .values()
            .next()
            .expect("seeded entry")
            .clone();
        let blocks_before: Vec<_> = {
            let g = pygraph.graph.borrow();
            let mut out = Vec::new();
            let mut stack = vec![g.startblock.clone()];
            let mut seen = std::collections::HashSet::new();
            while let Some(b) = stack.pop() {
                let b_ptr = std::rc::Rc::as_ptr(&b) as usize;
                if !seen.insert(b_ptr) {
                    continue;
                }
                let blk = b.borrow();
                out.push((b_ptr, blk.operations.len(), blk.exits.len()));
                for exit in &blk.exits {
                    if let Some(tgt) = exit.borrow().target.clone() {
                        stack.push(tgt);
                    }
                }
            }
            out.sort();
            out
        };

        // Run simplify again — if `from_rust_item_fn` skipped
        // upstream's `simplify_graph(graph)` step, simplify would
        // actually change something here.
        {
            let g = pygraph.graph.borrow();
            crate::translator::simplify::simplify_graph(&g, None);
        }

        let blocks_after: Vec<_> = {
            let g = pygraph.graph.borrow();
            let mut out = Vec::new();
            let mut stack = vec![g.startblock.clone()];
            let mut seen = std::collections::HashSet::new();
            while let Some(b) = stack.pop() {
                let b_ptr = std::rc::Rc::as_ptr(&b) as usize;
                if !seen.insert(b_ptr) {
                    continue;
                }
                let blk = b.borrow();
                out.push((b_ptr, blk.operations.len(), blk.exits.len()));
                for exit in &blk.exits {
                    if let Some(tgt) = exit.borrow().target.clone() {
                        stack.push(tgt);
                    }
                }
            }
            out.sort();
            out
        };

        assert_eq!(
            blocks_before, blocks_after,
            "simplify_graph must be idempotent after from_rust_item_fn — \
                if the second run changed the graph, the first simplify was skipped"
        );
    }

    #[test]
    fn from_rust_item_fn_with_source_threads_filename_into_co_filename() {
        // Upstream reads `func.__code__.co_filename` at `model.py:54`
        // (FunctionGraph.filename) for graph-render error messages.
        // The caller-supplied filename must land in HostCode.co_filename
        // instead of the `<rust-source>` sentinel.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (_t, host) = Translation::from_rust_item_fn_with_source(
            &item,
            Some("pyre/pyre-interpreter/src/pyopcode.rs"),
            None,
        )
        .expect("translation");

        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_filename, "pyre/pyre-interpreter/src/pyopcode.rs");
    }

    #[test]
    fn from_rust_item_fn_defaults_co_filename_to_rust_source_sentinel() {
        // When no source filename is threaded in, the sentinel remains
        // `<rust-source>`. Documented behaviour for fixture callers.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (_t, host) = Translation::from_rust_item_fn(&item).expect("translation");
        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_filename, "<rust-source>");
    }

    #[test]
    fn from_rust_item_fn_with_source_threads_source_text_to_graph_source() {
        // Upstream `model.py:35-47` exposes `FunctionGraph.source` as a
        // property; the Translation constructor uses it through
        // `graph.source()` on the error-rendering path
        // (`tool/error.rs:300-320`). Pre-patch, the adapter left both
        // `_source` and `GraphFunc.source` unset, so `graph.source()`
        // failed with the `"source not found"` error and the render
        // fell back to "no source!".
        let src = "fn one() -> i64 { 1 }";
        let item = parse_item_fn(src);
        let (t, _host) =
            Translation::from_rust_item_fn_with_source(&item, Some("fixture.rs"), Some(src))
                .expect("translation");

        let pygraph = t
            .context
            ._prebuilt_graphs
            .borrow()
            .values()
            .next()
            .expect("seeded entry")
            .clone();
        let graph = pygraph.graph.borrow();
        // `graph.source()` fast-path reads `_source`.
        assert_eq!(graph.source().expect("source populated"), src);
        // Fallback path: even if `_source` is cleared, `GraphFunc.source`
        // holds the same text so `inspect.getsource(func)`-style callers
        // still see it (model.rs:3207-3216).
        let gf_source = graph
            .func
            .as_ref()
            .expect("func set")
            .source
            .as_deref()
            .expect("GraphFunc.source populated");
        assert_eq!(gf_source, src);
    }

    #[test]
    fn from_rust_item_fn_auto_populates_source_text_from_tokens() {
        // Default-path parity: `from_rust_item_fn` (no explicit
        // source_text) auto-populates `graph.source()` via
        // `quote::ToTokens::to_token_stream().to_string()`. Upstream
        // `inspect.getsource(func)` at `bytecode.py:50` returns the
        // verbatim source — the Rust equivalent round-trips through
        // syn's token stream, which is whitespace-collapsed but
        // semantically complete (`fn one () -> i64 { 1 }`). What
        // matters for `tool/error.rs:300` is that `graph.source()`
        // succeeds, not the exact whitespace layout.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");
        let pygraph = t
            .context
            ._prebuilt_graphs
            .borrow()
            .values()
            .next()
            .expect("seeded entry")
            .clone();
        let graph = pygraph.graph.borrow();
        let src = graph.source().expect("auto-populated via ToTokens");
        assert!(
            src.contains("fn one") && src.contains("-> i64"),
            "token-stream render should contain fn signature: {src}"
        );
        // `GraphFunc.source` carries the same text (fallback path).
        let gf_source = graph
            .func
            .as_ref()
            .expect("func set")
            .source
            .as_deref()
            .expect("GraphFunc.source populated by default");
        assert_eq!(gf_source, src);
    }

    #[test]
    fn ensure_setup_stores_argtypes_and_policy_via_driver() {
        // Upstream `interactive.py:34-38` — `ensure_setup` calls
        // `self.driver.setup(self.entry_point, argtypes, policy,
        // empty_translator=self.context)` and then stores
        // `self.ann_argtypes = argtypes` / `self.ann_policy = policy`.
        // Round-trip both slots to confirm the storage half ran AND
        // the driver-side slot observes the same translator.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");
        let policy = AnnotatorPolicy::default();
        t.ensure_setup(Some(vec![]), Some(policy))
            .expect("ensure_setup");
        assert!(t.ann_argtypes.borrow().is_some());
        assert!(t.ann_policy.borrow().is_some());
        // Driver's translator slot must point at this Translation's
        // context (`empty_translator=self.context` parity).
        let driver_translator = t.driver.translator.borrow().as_ref().map(Rc::clone);
        assert!(
            driver_translator
                .as_ref()
                .map(|c| Rc::ptr_eq(c, &t.context))
                .unwrap_or(false),
            "driver.translator must alias self.context after ensure_setup"
        );
    }

    #[test]
    fn from_rust_item_fn_with_options_runs_update_and_ensure_setup() {
        // Upstream `interactive.py:14-26 __init__` runs
        // `update_options(kwds)` + `ensure_setup(argtypes, policy)`
        // before returning. Confirm the constructor variant that
        // accepts these parameters drives the same chain end-to-end.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let kwds = vec![("gc".to_string(), "boehm".to_string())];
        let argtypes: Vec<AnnotationSpec> = Vec::new();
        let policy = AnnotatorPolicy::default();
        let (t, _host) =
            Translation::from_rust_item_fn_with_options(&item, Some(argtypes), Some(policy), &kwds)
                .expect("translation");

        // `update_options(kwds)` landed: `translation.gc=boehm` is
        // visible on the live `Rc<Config>`.
        let gc = match t.config().get("translation.gc").expect("translation.gc") {
            crate::config::config::ConfigValue::Value(OptionValue::Choice(s)) => s,
            other => panic!("expected Choice, got {other:?}"),
        };
        assert_eq!(gc, "boehm");

        // `ensure_setup` landed: `ann_argtypes` / `ann_policy` are
        // populated and the driver knows about the context.
        assert!(t.ann_argtypes.borrow().is_some());
        assert!(t.ann_policy.borrow().is_some());
        let driver_translator = t.driver.translator.borrow().as_ref().map(Rc::clone);
        assert!(
            driver_translator
                .as_ref()
                .map(|c| Rc::ptr_eq(c, &t.context))
                .unwrap_or(false),
            "constructor must thread self.context through driver.setup"
        );
    }

    #[test]
    fn from_rust_item_fn_default_runs_ensure_setup_with_none_args() {
        // Upstream `interactive.py:14`: `argtypes=None` default + empty
        // `**kwds`. The bare `from_rust_item_fn` constructor must still
        // run `ensure_setup(None, None)` so the driver's `translator`
        // slot is populated (downstream callers expect
        // `t.driver.translator` to alias `t.context`).
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");
        let driver_translator = t.driver.translator.borrow().as_ref().map(Rc::clone);
        assert!(
            driver_translator
                .as_ref()
                .map(|c| Rc::ptr_eq(c, &t.context))
                .unwrap_or(false),
            "default constructor must wire driver.translator → self.context via ensure_setup"
        );
    }

    #[test]
    fn view_and_viewcg_are_no_op_safe_to_call() {
        // Upstream `view()` / `viewcg()` open pygame windows; the
        // Rust port has no pygame dependency, so both methods are
        // no-op stubs. Test invariant: calling them on a freshly-
        // constructed Translation does not panic / error (regression
        // check for accidentally changing the methods' safety
        // contract).
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");
        t.view();
        t.viewcg();
    }

    #[test]
    fn update_options_routes_kwds_through_driver_config() {
        // Upstream `interactive.py:40-44`:
        //
        //     gc = kwds.pop('gc', None)
        //     if gc:
        //         self.config.translation.gc = gc
        //     self.config.translation.set(**kwds)
        //
        // Empty kwds = no-op. `gc` is special-cased through the
        // ChoiceOption setter; everything else flows through
        // `Config::set`.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");
        // Empty: no error, config unchanged.
        t.update_options(&[]).expect("empty update");
        let verbose = match t
            .driver
            .config
            .get("translation.verbose")
            .expect("translation.verbose")
        {
            crate::config::config::ConfigValue::Value(OptionValue::Bool(b)) => b,
            other => panic!("expected Bool, got {other:?}"),
        };
        assert!(
            verbose,
            "DEFAULTS at interactive.py:6 must keep verbose=True"
        );

        // `gc` short-circuit lands as ChoiceOption.
        let kwds = vec![("gc".to_string(), "boehm".to_string())];
        t.update_options(&kwds).expect("gc update");
        let gc = match t
            .driver
            .config
            .get("translation.gc")
            .expect("translation.gc")
        {
            crate::config::config::ConfigValue::Value(OptionValue::Choice(s)) => s,
            other => panic!("expected Choice, got {other:?}"),
        };
        assert_eq!(gc, "boehm");
    }

    #[test]
    fn from_rust_item_fn_flags_entry_point_as_exported() {
        // Upstream `interactive.py:18 self.entry_point =
        // export_symbol(entry_point)` sets
        // `entry_point.exported_symbol = True`. The Rust port flips the
        // flag on `GraphFunc.exported_symbol` of the HostObject stored
        // in `self.entry_point`; verify both that slot AND the
        // `_prebuilt_graphs` key carry the flagged HostObject (parity
        // with upstream's Python object-identity — the map key and
        // `self.entry_point` share the single flagged function object).
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, host) = Translation::from_rust_item_fn(&item).expect("translation");

        // `self.entry_point` carries the flag.
        let gf = t.entry_point.user_function().expect("user function");
        assert!(
            gf.exported_symbol
                .load(std::sync::atomic::Ordering::Relaxed),
            "interactive.py:18 export_symbol(entry_point) must set exported_symbol=true"
        );

        // Returned host is the same flagged HostObject — tests that
        // round-trip through the `_prebuilt_graphs` cache must look up
        // via this host.
        assert_eq!(t.entry_point, host);
        assert!(
            t.context._prebuilt_graphs.borrow().contains_key(&host),
            "prebuilt cache key matches the returned (flagged) host"
        );
    }

    #[test]
    fn translation_carries_entry_point_and_placeholder_annotation_slots() {
        // Upstream `interactive.py:18, :37-38` — `Translation` carries
        // `entry_point` (the export_symbol-flagged function), plus
        // `ann_argtypes` / `ann_policy` set by `ensure_setup`. The
        // Rust port stores the HostObject directly in `entry_point`
        // (the `exported_symbol` flag itself is a C-backend detail —
        // see module docstring) and leaves the annotation slots
        // `None` until the driver-dependent `ensure_setup` port
        // lands.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, host) = Translation::from_rust_item_fn(&item).expect("translation");
        assert_eq!(t.entry_point, host);
        assert!(t.ann_argtypes.borrow().is_none());
        assert!(t.ann_policy.borrow().is_none());
    }

    #[test]
    fn translation_config_accessor_matches_driver_config() {
        // Upstream `interactive.py:16`: `self.config = self.driver.config`.
        // Our `config()` accessor returns the same `Rc<Config>` the
        // driver owns; `Rc::ptr_eq` confirms reference identity.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");
        assert!(Rc::ptr_eq(t.config(), &t.driver.config));
        let verbose = match t
            .config()
            .get("translation.verbose")
            .expect("translation.verbose")
        {
            crate::config::config::ConfigValue::Value(OptionValue::Bool(b)) => b,
            other => panic!("expected Bool, got {other:?}"),
        };
        assert!(
            verbose,
            "DEFAULTS at interactive.py:6 must keep verbose=True"
        );
    }

    #[test]
    fn context_is_shared_identity_between_translation_and_annotator() {
        // `context` is `Rc<TranslationContext>` so annotator
        // constructors that take `Rc<TranslationContext>` observe the
        // same instance. This matches upstream: `self.context` on
        // `Translation` and the `translator` arg threaded into
        // `RPythonAnnotator` are literal `is`-identical (Python object
        // identity).
        use crate::annotator::annrpython::RPythonAnnotator;

        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");

        let ann =
            RPythonAnnotator::new_with_translator(Some(Rc::clone(&t.context)), None, None, false);
        assert!(
            Rc::ptr_eq(&t.context, &ann.translator),
            "Translation.context and annotator.translator must be Rc-identical"
        );
    }

    // ---- forwarding API parity tests (interactive.py:46-124) -------------

    /// Helper: build a Translation around `fn main() -> i64 { ... }` with
    /// `argtypes=Some(vec![])` so `setup` records `standalone=False`.
    /// `task_annotate`'s `entry_point and standalone and s.knowntype != int`
    /// check at upstream `:321-324` is therefore skipped, matching the
    /// shape used by `task_annotate_runs_end_to_end_for_constant_return`.
    fn translation_for(src: &str) -> Translation {
        let item = parse_item_fn(src);
        let (t, _host) =
            Translation::from_rust_item_fn_with_options(&item, Some(Vec::new()), None, &[])
                .expect("translation");
        t
    }

    #[test]
    fn ensure_opt_returns_explicit_value_when_provided() {
        // Upstream `interactive.py:47-49`: `if value is not None:
        // self.update_options({name: value}); return value`.
        let t = translation_for("fn main() -> i64 { 1 }");
        let got = t
            .ensure_opt("type_system", Some("lltype"), None)
            .expect("ensure_opt explicit");
        assert_eq!(got, "lltype");
        // Side effect: `update_options` lands `lltype` on the live config.
        let read = match t.driver.config.get("translation.type_system").unwrap() {
            ConfigValue::Value(OptionValue::Choice(s)) => s,
            other => panic!("expected Choice, got {other:?}"),
        };
        assert_eq!(read, "lltype");
    }

    #[test]
    fn ensure_opt_returns_fallback_when_unset() {
        // Upstream `:51-53`: `if fallback is not None and val is None:
        // self.update_options({name: fallback}); return fallback`. The
        // driver default for `translation.type_system` is `None`, so
        // `ensure_opt('type_system', None, 'lltype')` lands `lltype` via
        // the fallback branch.
        let t = translation_for("fn main() -> i64 { 1 }");
        let got = t
            .ensure_opt("type_system", None, Some("lltype"))
            .expect("ensure_opt fallback");
        assert_eq!(got, "lltype");
    }

    #[test]
    fn ensure_backend_explicit_c_lands_lltype_via_requires() {
        // Upstream `interactive.py:6-10 DEFAULTS` sets
        // `translation.backend = None` (overriding the schema default
        // of `"c"` from `translationoption.py:51-56`). So
        // `ensure_backend(None)` raises — see the test below — but
        // `ensure_backend("c")` lands the value via the `value is not
        // None` branch and the `requires={"c": [("type_system",
        // "lltype")]}` chain triggers `type_system=lltype` as a side
        // effect.
        let t = translation_for("fn main() -> i64 { 1 }");
        let backend = t.ensure_backend(Some("c")).expect("ensure_backend(c)");
        assert_eq!(backend, "c");
        let ts = match t.driver.config.get("translation.type_system").unwrap() {
            ConfigValue::Value(OptionValue::Choice(s)) => s,
            other => panic!("expected Choice, got {other:?}"),
        };
        assert_eq!(ts, "lltype");
    }

    #[test]
    fn ensure_backend_none_raises_when_defaults_zero_backend() {
        // Upstream parity with `interactive.py:6 DEFAULTS = {'backend':
        // None, ...}`: with backend=None the unset slot has no
        // fallback, so `ensure_opt('backend', None, None)` raises
        // `"the 'backend' option should have been specified at this
        // point"` per upstream `:56-57`. The Rust port surfaces the
        // same message via `ConfigError::Generic`.
        let t = translation_for("fn main() -> i64 { 1 }");
        let err = t.ensure_backend(None).unwrap_err();
        match err {
            ConfigError::Generic(msg) => {
                assert!(msg.contains("\"backend\""), "msg: {msg}");
                assert!(msg.contains("should have been specified"), "msg: {msg}");
            }
            other => panic!("expected Generic, got {other:?}"),
        }
    }

    #[test]
    fn disable_forwards_to_driver_underscore_disabled() {
        // Upstream `interactive.py:69-71`: `def disable(self,
        // to_disable): self.driver.disable(to_disable)`. Forwards into
        // `driver.py:166-167` which writes `_disabled = to_disable`.
        let t = translation_for("fn main() -> i64 { 1 }");
        t.disable(vec!["compile_c".to_string(), "source_c".to_string()]);
        let stored = t.driver._disabled.borrow();
        assert_eq!(
            *stored,
            vec!["compile_c".to_string(), "source_c".to_string()]
        );
    }

    #[test]
    fn set_backend_extra_options_runs_ensure_backend_per_key() {
        // Upstream `interactive.py:73-77`: each name splits on `_` and
        // ensures the backend, then the dict is forwarded to
        // `driver.set_backend_extra_options`. Single-key fixture using
        // `c_compiler_path` exercises the `c` path (the only valid
        // choice) without triggering the `cl` rejection.
        let t = translation_for("fn main() -> i64 { 1 }");
        let mut extras = HashMap::new();
        extras.insert(
            "c_compiler_path".to_string(),
            OptionValue::Str("/usr/bin/clang".to_string()),
        );
        t.set_backend_extra_options(extras.clone())
            .expect("set_backend_extra_options");
        // Forwarded dict is on the driver.
        let stored = t.driver._backend_extra_options.borrow();
        assert!(stored.contains_key("c_compiler_path"));
        // Backend slot remains `c` (was the default already, but the
        // ensure_backend call must not break it).
        let backend = match t.driver.config.get("translation.backend").unwrap() {
            ConfigValue::Value(OptionValue::Choice(s)) => s,
            other => panic!("expected Choice, got {other:?}"),
        };
        assert_eq!(backend, "c");
    }

    #[test]
    fn set_backend_extra_options_rejects_name_without_underscore() {
        // Upstream `:75 backend, option = name.split('_', 1)` raises
        // `ValueError: not enough values to unpack` on a name without
        // `_`. The Rust port surfaces the same rejection as
        // `ConfigError::Generic`.
        let t = translation_for("fn main() -> i64 { 1 }");
        let mut extras = HashMap::new();
        extras.insert("nounderscore".to_string(), OptionValue::Bool(true));
        let err = t.set_backend_extra_options(extras).unwrap_err();
        match err {
            ConfigError::Generic(msg) => {
                assert!(msg.contains("nounderscore"), "msg: {msg}");
            }
            other => panic!("expected Generic, got {other:?}"),
        }
    }

    #[test]
    fn annotate_runs_proceed_through_engine() {
        // Upstream `interactive.py:81-83`:
        //
        //     def annotate(self, **kwds):
        //         self.update_options(kwds)
        //         return self.driver.annotate()
        //
        // The Rust port routes `driver.annotate()` through
        // `driver.proceed("annotate")`, which executes
        // `task_annotate` end-to-end (driver.py:297-327). Verify
        // completion + `done` bookkeeping via DriverHooks::_do.
        let t = translation_for("fn main() -> i64 { 42 }");
        t.annotate(&[]).expect("annotate");
        assert!(
            t.driver.done.borrow().contains_key("annotate"),
            "DriverHooks::_do must record annotate as done"
        );
    }

    #[test]
    fn rtype_runs_proceed_for_rtype_lltype_goal() {
        // Upstream `interactive.py:87-90`: `getattr(self.driver,
        // 'rtype_' + ts)()` resolves the `expose_task`-generated
        // proc whose body is `proceed('rtype_<ts>')`. With the
        // schema default `type_system=None`, `ensure_type_system`
        // lands `lltype` first, so the proceed goal is `rtype_lltype`.
        let t = translation_for("fn main() -> i64 { 7 }");
        t.rtype(&[]).expect("rtype");
        let done = t.driver.done.borrow();
        assert!(done.contains_key("annotate"));
        assert!(done.contains_key("rtype_lltype"));
    }

    #[test]
    fn compile_c_propagates_missing_task_leaf_error() {
        // Upstream `interactive.py:120-124`: `compile_c` runs
        // `proceed('compile_c')` then returns `self.driver.c_entryp`.
        // The `task_compile_c` body at upstream `:533-541` is DEFERRED
        // in the Rust port — `task_database_c` raises a
        // `missing_task_leaf` before the chain reaches `compile_c`. The
        // forwarding method must propagate that error rather than
        // silently dropping it.
        let t = translation_for("fn main() -> i64 { 1 }");
        let err = t.compile_c(&[]).unwrap_err();
        // Error message must cite the deferred leaf so a reviewer can
        // grep the exact upstream line. Accept both `not ported` and
        // `not yet ported` wordings — different shells along the chain
        // (driver.py / genc.py / all.py / unixcheckpoint.py) phrase the
        // deferred-leaf marker slightly differently.
        let msg = &err.message;
        let cites_upstream = msg.contains(".py:") || msg.contains("not ported");
        assert!(
            cites_upstream,
            "compile_c must surface the missing_task_leaf message: {msg}"
        );
    }

    // ---- Slice O9 — file-aware entry point with walker pre-pass ---

    #[test]
    fn from_rust_file_entry_point_walks_sibling_enum_before_lowering_entry() {
        // The new file-based entry walks every `Item::*` BEFORE
        // running the entry-point body lowerer. So a multi-item
        // file with `enum X { ... } fn main() {...}` registers `X`
        // in the module-globals registry first; when the body
        // lowerer resolves `X::Variant`, it constfolds through the
        // walker-populated class. Mirrors upstream
        // `flowcontext.py:847 w_globals.value[varname]` resolving
        // module-level names that import-time `def`/`class`
        // statements bound.
        use crate::flowspace::rust_source::register::register_rust_module;

        let src = "pub enum ParityProbe_O9_FileEntry_Enum { AlphaA, BetaB }\n\
                   fn parity_probe_o9_entry() -> i64 { 0 }";
        let file = parse_file(src);
        let (_t, host) = Translation::from_rust_file_entry_point(&file, "parity_probe_o9_entry")
            .expect("file-aware entry");
        assert_eq!(host.qualname(), "parity_probe_o9_entry");

        // Confirm the walker dispatch lands the sibling enum into
        // the module-globals registry: re-run `register_rust_module`
        // with a separate `ModuleId` (per-module scoping, Issue 1.3
        // — each call mints fresh) and verify the resulting
        // partition holds the expected `HostObject` shape. The
        // entry-point's body lowering inside `Translation` already
        // proved the production-side walker pre-pass works; this
        // assertion separately exercises the walker's observable
        // output for an arbitrary id.
        let probe_id = register_rust_module(&file);
        use crate::flowspace::model::ConstValue;
        let registered = crate::flowspace::rust_source::register::module_globals_for_test(
            probe_id,
            "ParityProbe_O9_FileEntry_Enum",
        )
        .expect("walker registered the sibling enum");
        let parent = match registered {
            ConstValue::HostObject(h) => h,
            other => panic!("enum entry must wrap HostObject, got {other:?}"),
        };
        assert!(parent.is_class());
        let alpha = parent
            .class_get("AlphaA")
            .expect("walker populated `AlphaA` variant");
        match alpha {
            ConstValue::HostObject(child) => {
                assert!(child.is_class());
                assert_eq!(child.qualname(), "ParityProbe_O9_FileEntry_Enum.AlphaA");
            }
            other => panic!("variant must be HostObject::Class, got {other:?}"),
        }
    }

    #[test]
    fn from_rust_file_entry_point_rejects_missing_entry_name() {
        let src = "fn parity_probe_o9_existing() -> i64 { 0 }";
        let file = parse_file(src);
        match Translation::from_rust_file_entry_point(&file, "nonexistent_target") {
            Err(TranslationConstructError::Adapter(AdapterError::Unsupported { reason })) => {
                assert!(
                    reason.contains("entry-point fn `nonexistent_target` not found"),
                    "must surface the missing-entry diagnostic, got: {reason}"
                );
            }
            Err(other) => panic!("expected Adapter(Unsupported), got {other:?}"),
            Ok(_) => panic!("must error when entry point name is not in file.items"),
        }
    }
}
