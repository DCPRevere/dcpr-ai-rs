//! Typed structured output: derive a JSON Schema from a Rust type, ask the
//! model for it, and decode the reply.

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{ChatModel, Completion, Error, Result};

/// The JSON Schema for `T`, without root metadata such as `$schema` and
/// `title`, which waste provider context. Subschemas are inlined so the result
/// is self-contained.
pub fn json_schema_for<T: JsonSchema>() -> Value {
    let settings = schemars::generate::SchemaSettings::draft2020_12().with(|settings| {
        settings.meta_schema = None;
        settings.inline_subschemas = true;
    });
    let mut schema = serde_json::to_value(settings.into_generator().into_root_schema_for::<T>())
        .expect("schemars always serializes a generated schema");
    if let Some(object) = schema.as_object_mut() {
        object.remove("$schema");
        object.remove("title");
    }
    schema
}

/// The schema for `T` in the form strict structured output requires (OpenAI
/// `strict: true`, and Anthropic's compatible endpoint, which accepts only
/// strict schemas): every object lists all its properties in `required` and
/// sets `additionalProperties: false`. A property that was optional becomes
/// nullable instead, so `Option` fields still decode. Numeric bounds and
/// formats (`minimum`, `format: uint32` and so on) are dropped, because
/// Anthropic rejects them; decoding into `T` still enforces them.
pub fn strict_json_schema_for<T: JsonSchema>() -> Value {
    let mut schema = json_schema_for::<T>();
    make_strict(&mut schema);
    schema
}

fn make_strict(schema: &mut Value) {
    let Some(object) = schema.as_object_mut() else {
        return;
    };
    for key in [
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "format",
    ] {
        object.remove(key);
    }
    let required: std::collections::HashSet<String> = object
        .get("required")
        .and_then(Value::as_array)
        .map(|names| {
            names
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    if let Some(Value::Object(properties)) = object.get_mut("properties") {
        for (name, property) in properties.iter_mut() {
            if !required.contains(name) {
                make_nullable(property);
            }
            make_strict(property);
        }
        let names: Vec<Value> = properties.keys().cloned().map(Value::String).collect();
        object.insert("required".into(), Value::Array(names));
        object.insert("additionalProperties".into(), Value::Bool(false));
    } else if object.get("type").and_then(Value::as_str) == Some("object") {
        object.insert("additionalProperties".into(), Value::Bool(false));
    }
    for key in ["items", "additionalItems", "not"] {
        if let Some(inner) = object.get_mut(key) {
            make_strict(inner);
        }
    }
    for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
        if let Some(Value::Array(variants)) = object.get_mut(key) {
            variants.iter_mut().for_each(make_strict);
        }
    }
    for key in ["$defs", "definitions"] {
        if let Some(Value::Object(defs)) = object.get_mut(key) {
            defs.values_mut().for_each(make_strict);
        }
    }
}

fn make_nullable(schema: &mut Value) {
    let null = Value::String("null".into());
    match schema.get_mut("type") {
        Some(Value::String(kind)) if kind != "null" => {
            let kind = Value::String(kind.clone());
            schema["type"] = Value::Array(vec![kind, null]);
        }
        Some(Value::Array(kinds)) if !kinds.contains(&null) => kinds.push(null),
        Some(_) => {}
        None => {
            if let Some(Value::Array(variants)) = schema.get_mut("anyOf") {
                if !variants.iter().any(|v| v["type"] == "null") {
                    variants.push(serde_json::json!({"type": "null"}));
                }
            } else {
                *schema = serde_json::json!({"anyOf": [schema.clone(), {"type": "null"}]});
            }
        }
    }
}

/// Ask `model` for a reply shaped like `T` and decode it.
///
/// The schema is derived from `T` and sent as `response_format: json_schema`.
/// A reply that does not decode is a transient error, since another attempt or
/// another model usually succeeds.
///
/// ```no_run
/// # async fn example(model: &dyn dcpr_llm::ChatModel) -> dcpr_llm::Result<()> {
/// #[derive(serde::Deserialize, schemars::JsonSchema)]
/// struct Verdict {
///     approved: bool,
///     reason: String,
/// }
///
/// let verdict: Verdict = dcpr_llm::complete_typed(
///     model,
///     "user-42",
///     "Review the request. Reply as JSON.",
///     "May I borrow the rover tomorrow?",
///     200,
/// )
/// .await?;
/// # Ok(()) }
/// ```
pub async fn complete_typed<T, C, M>(
    model: &M,
    caller: &C,
    system: &str,
    user: &str,
    max_tokens: u32,
) -> Result<T>
where
    T: DeserializeOwned + JsonSchema,
    C: ?Sized + Sync,
    M: ChatModel<C> + ?Sized,
{
    let schema = json_schema_for::<T>();
    let name = schema_name::<T>();
    let request = Completion::new(system, user, max_tokens).with_schema(&name, &schema);
    let reply = model.complete(caller, request).await?;
    decode_json(&reply)
}

/// Like [`complete_typed`], with the provider enforcing the schema exactly
/// ([`strict_json_schema_for`] and `strict: true`). Use it on OpenAI, where it
/// guarantees the shape, and on Anthropic's compatible endpoint, which accepts
/// nothing else.
pub async fn complete_typed_strict<T, C, M>(
    model: &M,
    caller: &C,
    system: &str,
    user: &str,
    max_tokens: u32,
) -> Result<T>
where
    T: DeserializeOwned + JsonSchema,
    C: ?Sized + Sync,
    M: ChatModel<C> + ?Sized,
{
    let schema = strict_json_schema_for::<T>();
    let name = schema_name::<T>();
    let request = Completion::new(system, user, max_tokens)
        .with_schema(&name, &schema)
        .strict();
    let reply = model.complete(caller, request).await?;
    decode_json(&reply)
}

/// Decode a JSON reply, tolerating a Markdown code fence around it.
pub fn decode_json<T: DeserializeOwned>(reply: &str) -> Result<T> {
    let text = reply.trim();
    let text = text
        .strip_prefix("```json")
        .or_else(|| text.strip_prefix("```"))
        .and_then(|inner| inner.trim_end().strip_suffix("```"))
        .unwrap_or(text)
        .trim();
    serde_json::from_str(text).map_err(|error| {
        let snippet: String = text.chars().take(200).collect();
        Error::transient(format!(
            "reply did not match the requested schema ({error}): {snippet}"
        ))
    })
}

/// A provider-safe schema name from the type name: its last path segment,
/// limited to letters, digits, `_` and `-`, at most 64 characters.
fn schema_name<T>() -> String {
    let full = std::any::type_name::<T>();
    let base = full.split('<').next().unwrap_or(full);
    let last = base.rsplit("::").next().unwrap_or(base);
    let name: String = last
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .take(64)
        .collect();
    if name.is_empty() {
        "response".to_string()
    } else {
        name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, serde::Deserialize, schemars::JsonSchema)]
    struct Harvest {
        bed: u32,
        ready: bool,
    }

    #[test]
    fn schemas_drop_root_metadata_and_names_are_provider_safe() {
        let schema = json_schema_for::<Harvest>();
        assert!(schema.get("$schema").is_none() && schema.get("title").is_none());
        assert_eq!(schema["properties"]["bed"]["type"], "integer");
        assert_eq!(schema_name::<Harvest>(), "Harvest");
        assert_eq!(schema_name::<Vec<Harvest>>(), "Vec");
    }

    #[derive(Debug, PartialEq, serde::Deserialize, schemars::JsonSchema)]
    struct Plan {
        bed: u32,
        note: Option<String>,
        steps: Vec<Step>,
    }

    #[derive(Debug, PartialEq, serde::Deserialize, schemars::JsonSchema)]
    struct Step {
        action: String,
        minutes: Option<u32>,
    }

    #[test]
    fn strict_schemas_require_every_property_and_close_every_object() {
        let schema = strict_json_schema_for::<Plan>();
        fn check(schema: &Value) {
            if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
                assert_eq!(schema["additionalProperties"], false, "{schema}");
                let required = schema["required"].as_array().unwrap();
                assert_eq!(required.len(), properties.len(), "{schema}");
                properties.values().for_each(check);
            }
            if let Some(items) = schema.get("items") {
                check(items);
            }
            for variants in ["anyOf", "oneOf"] {
                schema[variants]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .for_each(check);
            }
        }
        check(&schema);
        assert!(!schema.to_string().contains("minimum"), "{schema}");
        // An optional field may now be null, and still decodes when it is.
        assert!(schema["properties"]["note"].to_string().contains("null"));
        let plan: Plan = decode_json(
            r#"{"bed": 2, "note": null, "steps": [{"action": "water", "minutes": null}]}"#,
        )
        .unwrap();
        assert_eq!(plan.steps[0].minutes, None);
    }

    #[test]
    fn replies_decode_with_or_without_a_code_fence() {
        let plain: Harvest = decode_json(r#"{"bed": 4, "ready": true}"#).unwrap();
        let fenced: Harvest = decode_json("```json\n{\"bed\": 4, \"ready\": true}\n```").unwrap();
        assert_eq!(plain, fenced);
        let err = decode_json::<Harvest>("Bed four is ready.").unwrap_err();
        assert!(err.is_retryable(), "{err}");
    }
}
