use std::fmt;

use pyre_object::excobject::EXCEPTION_TYPE;
use pyre_object::pyobject::{
    BOOL_TYPE, ELLIPSIS_TYPE, FLOAT_TYPE, INSTANCE_TYPE, INT_TYPE, LONG_TYPE, MODULE_TYPE,
    NONE_TYPE, PyObjectRef, PyType, STR_TYPE, TYPE_TYPE,
};

use crate::{
    BUILTIN_CODE_TYPE, BUILTIN_FUNCTION_TYPE, FUNCTION_TYPE, builtin_code_name, function_get_name,
};

/// Try to call a dunder method (__repr__, __str__, etc.) on an instance.
///
/// PyPy: `ObjSpace.call_function(space.lookup(w_obj, name), w_obj)`
/// Uses the unified `call_function` instead of a dedicated callback.
fn try_call_dunder(obj: PyObjectRef, name: &str) -> Option<String> {
    unsafe {
        if !pyre_object::is_instance(obj) {
            return None;
        }
        let method = crate::baseobjspace::lookup(obj, name)?;
        if method.is_null() {
            return None;
        }
        let result = crate::call_function(method, &[obj]);
        if result.is_null() {
            return None;
        }
        if pyre_object::is_str(result) {
            return Some(pyre_object::w_str_get_value(result).to_string());
        }
    }
    None
}

/// Format a PyObjectRef for debug display.
///
/// # Safety
/// `obj` must be a valid pointer to a known Python object type.
pub unsafe fn py_repr(obj: PyObjectRef) -> String {
    let obj = crate::baseobjspace::unwrap_cell(obj);
    if obj.is_null() {
        return "NULL".to_string();
    }
    unsafe {
        let tp = (*obj).ob_type;
        if std::ptr::eq(tp, &INT_TYPE as *const PyType) {
            let int_obj = obj as *const pyre_object::intobject::W_IntObject;
            format!("{}", (*int_obj).intval)
        } else if std::ptr::eq(tp, &FLOAT_TYPE as *const PyType) {
            let float_obj = obj as *const pyre_object::floatobject::W_FloatObject;
            let val = (*float_obj).floatval;
            if val.fract() == 0.0 && val.is_finite() {
                format!("{val:.1}")
            } else {
                format!("{val}")
            }
        } else if std::ptr::eq(tp, &LONG_TYPE as *const PyType) {
            let long_obj = obj as *const pyre_object::longobject::W_LongObject;
            format!("{}", &*(*long_obj).value)
        } else if std::ptr::eq(tp, &BOOL_TYPE as *const PyType) {
            let bool_obj = obj as *const pyre_object::boolobject::W_BoolObject;
            if (*bool_obj).boolval {
                "True".to_string()
            } else {
                "False".to_string()
            }
        } else if std::ptr::eq(tp, &pyre_object::pyobject::LIST_TYPE as *const PyType) {
            let n = pyre_object::w_list_len(obj);
            let mut parts = Vec::with_capacity(n);
            for i in 0..n {
                if let Some(item) = pyre_object::w_list_getitem(obj, i as i64) {
                    parts.push(py_repr(item));
                }
            }
            format!("[{}]", parts.join(", "))
        } else if std::ptr::eq(tp, &pyre_object::pyobject::TUPLE_TYPE as *const PyType) {
            let n = pyre_object::w_tuple_len(obj);
            let mut parts = Vec::with_capacity(n);
            for i in 0..n {
                if let Some(item) = pyre_object::w_tuple_getitem(obj, i as i64) {
                    parts.push(py_repr(item));
                }
            }
            if n == 1 {
                format!("({},)", parts[0])
            } else {
                format!("({})", parts.join(", "))
            }
        } else if std::ptr::eq(tp, &pyre_object::pyobject::DICT_TYPE as *const PyType) {
            let d = &*(obj as *const pyre_object::dictobject::W_DictObject);
            let entries = &*d.entries;
            let mut parts = Vec::with_capacity(entries.len());
            for &(k, v) in entries {
                parts.push(format!("{}: {}", py_repr(k), py_repr(v)));
            }
            format!("{{{}}}", parts.join(", "))
        } else if std::ptr::eq(tp, &STR_TYPE as *const PyType) {
            let str_obj = obj as *const pyre_object::strobject::W_StrObject;
            format!("'{}'", &*(*str_obj).value)
        } else if std::ptr::eq(tp, &NONE_TYPE as *const PyType) {
            "None".to_string()
        } else if std::ptr::eq(
            tp,
            &pyre_object::pyobject::NOTIMPLEMENTED_TYPE as *const PyType,
        ) {
            "NotImplemented".to_string()
        } else if std::ptr::eq(tp, &ELLIPSIS_TYPE as *const PyType) {
            "Ellipsis".to_string()
        } else if std::ptr::eq(tp, &BUILTIN_CODE_TYPE as *const PyType) {
            // Raw BuiltinCode objects (Code-level, not normally user-visible)
            let name = builtin_code_name(obj);
            format!("<code {name}>")
        } else if std::ptr::eq(tp, &BUILTIN_FUNCTION_TYPE as *const PyType) {
            // function.py:721 BuiltinFunction.descr_function_repr
            let name = function_get_name(obj);
            format!("<built-in function {name}>")
        } else if std::ptr::eq(tp, &FUNCTION_TYPE as *const PyType) {
            // function.py:268-269 Function.descr_function_repr —
            // FunctionWithFixedCode inherits this and reports as <function>.
            let name = function_get_name(obj);
            format!("<function {name}>")
        } else if std::ptr::eq(tp, &EXCEPTION_TYPE as *const PyType) {
            let msg = pyre_object::excobject::w_exception_get_message(obj);
            msg.to_string()
        } else if std::ptr::eq(tp, &TYPE_TYPE as *const PyType) {
            let name = pyre_object::w_type_get_name(obj);
            format!("<class '{name}'>")
        } else if std::ptr::eq(tp, &pyre_object::UNION_TYPE as *const PyType) {
            // PyPy: UnionType.__repr__ → " | ".join([_repr_item(x) for x in self.__args__])
            let args = pyre_object::w_union_get_args(obj);
            let n = pyre_object::w_tuple_len(args);
            let mut parts = Vec::with_capacity(n);
            for i in 0..n {
                if let Some(item) = pyre_object::w_tuple_getitem(args, i as i64) {
                    if pyre_object::is_none(item) {
                        parts.push("None".to_string());
                    } else if pyre_object::is_type(item) {
                        parts.push(pyre_object::w_type_get_name(item).to_string());
                    } else {
                        parts.push(py_repr(item));
                    }
                }
            }
            parts.join(" | ")
        } else if std::ptr::eq(tp, &MODULE_TYPE as *const PyType) {
            let name = pyre_object::w_module_get_name(obj);
            format!("<module '{name}'>")
        } else if std::ptr::eq(
            tp,
            &pyre_object::pyobject::MAPPING_PROXY_TYPE as *const PyType,
        ) {
            // `pypy/objspace/std/dictproxyobject.py:47 descr_repr` →
            // `b"mappingproxy(%s)" % space.utf8_w(space.repr(self.w_mapping))`.
            let inner = pyre_object::w_dict_proxy_get_mapping(obj);
            format!("mappingproxy({})", py_repr(inner))
        } else if std::ptr::eq(tp, &INSTANCE_TYPE as *const PyType) {
            // Try __repr__ first, then __str__
            if let Some(s) = try_call_dunder(obj, "__repr__") {
                return s;
            }
            if let Some(s) = try_call_dunder(obj, "__str__") {
                return s;
            }
            let w_type = pyre_object::w_instance_get_type(obj);
            let name = pyre_object::w_type_get_name(w_type);
            format!("<{name} object at {obj:?}>")
        } else {
            format!("<{} object at {:?}>", (*tp).name, obj)
        }
    }
}

/// Format for str() — tries __str__ first, then __repr__.
pub unsafe fn py_str(obj: PyObjectRef) -> String {
    unsafe {
        let obj = crate::baseobjspace::unwrap_cell(obj);
        if obj.is_null() {
            return "NULL".to_string();
        }
        let tp = (*obj).ob_type;
        // For strings, return the value directly (no quotes).
        if std::ptr::eq(tp, &STR_TYPE as *const PyType) {
            return pyre_object::w_str_get_value(obj).to_string();
        }
        if std::ptr::eq(tp, &INSTANCE_TYPE as *const PyType) {
            if let Some(s) = try_call_dunder(obj, "__str__") {
                return s;
            }
            if let Some(s) = try_call_dunder(obj, "__repr__") {
                return s;
            }
        }
        py_repr(obj)
    }
}

/// Display wrapper for PyObjectRef.
pub struct PyDisplay(pub PyObjectRef);

impl fmt::Display for PyDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_null() {
            write!(f, "NULL")
        } else {
            write!(f, "{}", unsafe { py_str(self.0) })
        }
    }
}
