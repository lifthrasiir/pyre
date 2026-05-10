//! pypy/module/_weakref/moduledef.py
//!
//! ```python
//! from pypy.interpreter.mixedmodule import MixedModule
//!
//! class Module(MixedModule):
//!     appleveldefs = {
//!     }
//!     interpleveldefs = {
//!         'ref': 'interp__weakref.W_Weakref',
//!         'getweakrefcount': 'interp__weakref.getweakrefcount',
//!         'getweakrefs': 'interp__weakref.getweakrefs',
//!         'ReferenceType': 'interp__weakref.W_Weakref',
//!         'ProxyType': 'interp__weakref.W_Proxy',
//!         'CallableProxyType': 'interp__weakref.W_CallableProxy',
//!         'proxy': 'interp__weakref.proxy'
//!     }
//! ```

use crate::DictStorage;
use crate::make_module_builtin_function_with_arity;
use crate::module::_weakref::interp_weakref;

pub fn init(ns: &mut DictStorage) {
    // pypy/module/_weakref/moduledef.py:6-13 interpleveldefs:
    //   'ref'                : 'interp__weakref.W_Weakref',
    //   'getweakrefcount'    : 'interp__weakref.getweakrefcount',
    //   'getweakrefs'        : 'interp__weakref.getweakrefs',
    //   'ReferenceType'      : 'interp__weakref.W_Weakref',
    //   'ProxyType'          : 'interp__weakref.W_Proxy',
    //   'CallableProxyType'  : 'interp__weakref.W_CallableProxy',
    //   'proxy'              : 'interp__weakref.proxy'
    let weakref_type = interp_weakref::weakref_type();
    let proxy_type = interp_weakref::proxy_type();
    let callable_proxy_type = interp_weakref::callable_proxy_type();
    crate::dict_storage_store(ns, "ref", weakref_type);
    crate::dict_storage_store(ns, "ReferenceType", weakref_type);
    crate::dict_storage_store(ns, "ProxyType", proxy_type);
    crate::dict_storage_store(ns, "CallableProxyType", callable_proxy_type);
    crate::dict_storage_store(
        ns,
        "proxy",
        crate::make_builtin_function("proxy", interp_weakref::proxy),
    );
    crate::dict_storage_store(
        ns,
        "getweakrefcount",
        make_module_builtin_function_with_arity(
            "getweakrefcount",
            interp_weakref::getweakrefcount,
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getweakrefs",
        make_module_builtin_function_with_arity("getweakrefs", interp_weakref::getweakrefs, 1),
    );
    // CPython-specific helper used by weakref.py to clean up dead refs
    // from WeakValueDictionary. PyPy doesn't expose it because PyPy's
    // weakrefs auto-clean. pyre stubs it as a no-op.
    crate::dict_storage_store(
        ns,
        "_remove_dead_weakref",
        make_module_builtin_function_with_arity(
            "_remove_dead_weakref",
            |_| Ok(pyre_object::w_none()),
            2,
        ),
    );
}
