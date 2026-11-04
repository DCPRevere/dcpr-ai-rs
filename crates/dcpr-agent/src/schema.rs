//! JSON Schema helpers for tool arguments: derive a schema from a Rust type,
//! coerce the scalar spellings models commonly emit, and validate arguments
//! before a tool performs any I/O.

use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::ToolInputError;

/// Generate the exact JSON Schema advertised to the model from a Rust input
/// type, without root metadata. The same as [`dcpr_llm::json_schema_for`].
pub fn typed_parameters<T: JsonSchema>() -> Value {
    dcpr_llm::json_schema_for::<T>()
}

/// Coerce the few safe scalar spellings models commonly emit, then decode and
/// re-encode through the same Rust type used to generate the provider schema.
/// The returned JSON is canonical, and is the only version a host should
/// authorise, record and execute.
pub fn prepare_typed_args<T>(args: Value) -> Result<Value, ToolInputError>
where
    T: DeserializeOwned + Serialize + JsonSchema,
{
    // Typed inputs decode through Serde: its aliases are part of the accepted
    // wire contract even when JSON Schema can advertise only canonical spellings.
    let typed: T = decode_typed_args(args)?;
    serde_json::to_value(typed).map_err(|error| ToolInputError::new(error.to_string()))
}

/// Coerce scalar spellings against `T`'s schema, then decode into `T`.
pub fn decode_typed_args<T>(mut args: Value) -> Result<T, ToolInputError>
where
    T: DeserializeOwned + JsonSchema,
{
    let schema = typed_parameters::<T>();
    coerce_args_to_schema(&mut args, &schema);
    serde_json::from_value(args).map_err(|error| ToolInputError::new(error.to_string()))
}

/// Coerce model-supplied arguments to the types the tool's schema declares.
///
/// Models — and the Hermes-XML fallback parser — frequently emit numbers and
/// booleans as JSON STRINGS (`"30"`, `"true"`) or, on the XML path, the parser
/// upcasts a string to a number. Either way a consumer doing `.as_i64()` /
/// `.as_bool()` gets `None` and silently falls back to a default or no-op (a
/// scheduled job that never runs, a confirmation flag that never confirms).
///
/// Driven by the tool's own `parameters` schema (the single source of truth), we
/// reconcile each property to its declared type. Critically we only convert a
/// STRING into a number/bool when the schema asks for one — a `type:string`
/// field is left EXACTLY as sent, so a numeric-looking text value ("100") or
/// title is never corrupted into a number. Unknown/unschema'd keys pass through
/// untouched.
pub fn coerce_args_to_schema(args: &mut Value, schema: &Value) {
    let Some(props) = schema.get("properties").and_then(Value::as_object) else {
        return;
    };
    let Some(obj) = args.as_object_mut() else {
        return;
    };
    for (key, pschema) in props {
        let Some(declared) = declared_schema_type(pschema) else {
            continue;
        };
        let Some(current) = obj.get_mut(key) else {
            continue;
        };
        // Recurse into structured values so a string-typed number nested inside an
        // object or array is coerced too. A nested `interval` read as an integer
        // would otherwise stay "2" and silently fall back to a default.
        match declared {
            "object" => {
                coerce_args_to_schema(current, pschema);
                continue;
            }
            "array" => {
                if let (Some(items_schema), Some(arr)) =
                    (pschema.get("items"), current.as_array_mut())
                {
                    // If the items schema declares a SCALAR type, each element is a
                    // string the model/parser may have given us for a number/bool —
                    // coerce it directly (recursing would no-op on non-objects, so
                    // `["9","17"]` for items:{integer} would stay strings). For a
                    // structured item type (object/array), recurse as before.
                    let item_type = declared_schema_type(items_schema);
                    if matches!(item_type, Some("integer" | "number" | "boolean")) {
                        let item_type = item_type.unwrap();
                        for el in arr.iter_mut() {
                            if let Some(v) = coerce_scalar(el, item_type) {
                                *el = v;
                            }
                        }
                    } else {
                        for el in arr.iter_mut() {
                            coerce_args_to_schema(el, items_schema);
                        }
                    }
                }
                continue;
            }
            _ => {}
        }
        if let Some(v) = coerce_scalar(current, declared) {
            obj.insert(key.clone(), v);
        }
    }
}

/// Enforce the provider-visible input contract before a tool can perform I/O.
/// Provider-side schema validation is useful but not a trust boundary: routed
/// models and compatibility transports can still return malformed arguments.
/// Tool objects are closed by default here, even when an older handwritten
/// schema omitted `additionalProperties: false`.
pub fn validate_args_against_schema(
    value: &Value,
    schema: &Value,
    path: &str,
) -> Result<(), ToolInputError> {
    if let Some(options) = schema.get("anyOf").and_then(Value::as_array) {
        if options
            .iter()
            .any(|option| validate_args_against_schema(value, option, path).is_ok())
        {
            return Ok(());
        }
        return Err(ToolInputError::new(format!(
            "{path} does not match any allowed shape"
        )));
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array)
        && !values.contains(value)
    {
        return Err(ToolInputError::new(format!(
            "{path} is not an allowed value"
        )));
    }
    let accepts_null = schema
        .get("type")
        .and_then(Value::as_array)
        .is_some_and(|types| types.iter().any(|kind| kind == "null"));
    if value.is_null() && accepts_null {
        return Ok(());
    }
    match declared_schema_type(schema) {
        Some("object") => {
            let object = value
                .as_object()
                .ok_or_else(|| ToolInputError::new(format!("{path} must be an object")))?;
            let properties = schema
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            if let Some(required) = schema.get("required").and_then(Value::as_array) {
                for key in required.iter().filter_map(Value::as_str) {
                    if !object.contains_key(key) {
                        return Err(ToolInputError::new(format!("{path}.{key} is required")));
                    }
                }
            }
            for (key, child) in object {
                let Some(child_schema) = properties.get(key) else {
                    return Err(ToolInputError::new(format!(
                        "{path}.{key} is not a recognised field"
                    )));
                };
                validate_args_against_schema(child, child_schema, &format!("{path}.{key}"))?;
            }
        }
        Some("array") => {
            let array = value
                .as_array()
                .ok_or_else(|| ToolInputError::new(format!("{path} must be an array")))?;
            if let Some(item_schema) = schema.get("items") {
                for (index, item) in array.iter().enumerate() {
                    validate_args_against_schema(item, item_schema, &format!("{path}[{index}]"))?;
                }
            }
        }
        Some("string") if !value.is_string() => {
            return Err(ToolInputError::new(format!("{path} must be a string")));
        }
        Some("integer") if !value.is_i64() && !value.is_u64() => {
            return Err(ToolInputError::new(format!("{path} must be an integer")));
        }
        Some("number") if !value.is_number() => {
            return Err(ToolInputError::new(format!("{path} must be a number")));
        }
        Some("boolean") if !value.is_boolean() => {
            return Err(ToolInputError::new(format!("{path} must be a boolean")));
        }
        _ => {}
    }
    Ok(())
}

/// Return the concrete JSON type from either a simple `"type":"object"` schema
/// or the nullable form (`"type":["object","null"]`) generated for `Option<T>`.
fn declared_schema_type(schema: &Value) -> Option<&str> {
    let value = schema.get("type")?;
    value.as_str().or_else(|| {
        value
            .as_array()?
            .iter()
            .filter_map(Value::as_str)
            .find(|kind| *kind != "null")
    })
}

/// Reinterpret a STRING value as the scalar type `declared` asks for
/// (integer/number/boolean). Returns `Some(new)` only when `value` is a string
/// that parses cleanly; returns `None` for a non-string (never down/up-casts a
/// value that already has a JSON type — a real number stays a number) or an
/// unparseable/`string`/unknown case (left verbatim, no false zero).
fn coerce_scalar(value: &Value, declared: &str) -> Option<Value> {
    let s = value.as_str()?.trim();
    match declared {
        "integer" => s.parse::<i64>().ok().map(Value::from),
        "number" => s
            .parse::<f64>()
            .ok()
            .filter(|f| f.is_finite())
            .map(Value::from),
        "boolean" => match s.to_ascii_lowercase().as_str() {
            "true" => Some(Value::Bool(true)),
            "false" => Some(Value::Bool(false)),
            _ => None,
        },
        _ => None, // "string" / unknown: leave verbatim.
    }
}
