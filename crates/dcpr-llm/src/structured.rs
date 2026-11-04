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

    #[test]
    fn replies_decode_with_or_without_a_code_fence() {
        let plain: Harvest = decode_json(r#"{"bed": 4, "ready": true}"#).unwrap();
        let fenced: Harvest = decode_json("```json\n{\"bed\": 4, \"ready\": true}\n```").unwrap();
        assert_eq!(plain, fenced);
        let err = decode_json::<Harvest>("Bed four is ready.").unwrap_err();
        assert!(err.is_retryable(), "{err}");
    }
}
