use js_sys::{Array, Function, Object, Promise, Reflect};
use wasm_bindgen::{JsCast, JsValue};

pub(crate) fn object() -> Object {
    Object::new()
}

pub(crate) fn set(target: &Object, key: &str, value: impl AsRef<JsValue>) {
    Reflect::set(target, &JsValue::from_str(key), value.as_ref())
        .unwrap_or_else(|error| panic!("failed to set WebGPU descriptor field {key}: {error:?}"));
}

pub(crate) fn set_opt_str(target: &Object, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        set(target, key, JsValue::from_str(value));
    }
}

pub(crate) fn get(target: &JsValue, key: &str) -> JsValue {
    Reflect::get(target, &JsValue::from_str(key))
        .unwrap_or_else(|error| panic!("failed to read WebGPU property {key}: {error:?}"))
}

pub(crate) fn get_opt(target: &JsValue, key: &str) -> Option<JsValue> {
    let value = Reflect::get(target, &JsValue::from_str(key)).ok()?;
    (!value.is_null() && !value.is_undefined()).then_some(value)
}

pub(crate) fn call(target: &JsValue, method: &str, args: &[JsValue]) -> JsValue {
    let function: Function = get(target, method)
        .dyn_into()
        .unwrap_or_else(|_| panic!("WebGPU property {method} is not callable"));
    let array = Array::new();
    for argument in args {
        array.push(argument);
    }
    function
        .apply(target, &array)
        .unwrap_or_else(|error| panic!("WebGPU call {method} failed: {error:?}"))
}

pub(crate) fn call_result(
    target: &JsValue,
    method: &str,
    args: &[JsValue],
) -> Result<JsValue, JsValue> {
    let function: Function = get(target, method).dyn_into().map_err(|value| value)?;
    let array = Array::new();
    for argument in args {
        array.push(argument);
    }
    function.apply(target, &array)
}

pub(crate) fn call_promise(target: &JsValue, method: &str, args: &[JsValue]) -> Promise {
    call(target, method, args)
        .dyn_into()
        .unwrap_or_else(|_| panic!("WebGPU call {method} did not return a Promise"))
}

pub(crate) fn array(values: impl IntoIterator<Item = JsValue>) -> Array {
    let result = Array::new();
    for value in values {
        result.push(&value);
    }
    result
}

pub(crate) fn number(value: impl Into<f64>) -> JsValue {
    JsValue::from_f64(value.into())
}

pub(crate) fn bool_value(value: bool) -> JsValue {
    JsValue::from_bool(value)
}

pub(crate) fn string(value: &str) -> JsValue {
    JsValue::from_str(value)
}

pub(crate) fn error_string(value: &JsValue) -> String {
    get_opt(value, "message")
        .and_then(|message| message.as_string())
        .or_else(|| value.as_string())
        .unwrap_or_else(|| format!("{value:?}"))
}
