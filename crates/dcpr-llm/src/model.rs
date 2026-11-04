//! The model trait and the caller identity it is generic over.

use async_trait::async_trait;
use std::fmt;
use std::sync::Arc;

use crate::{ChatStep, Error, Result, SystemPrompt, TextStreamSink, ToolTurn};

/// Who a request is for. The library never interprets a caller beyond these
/// methods; a [`UsageSink`](crate::UsageSink) receives it unchanged so an
/// application can attribute cost to its own accounts.
pub trait Caller: fmt::Display + Send + Sync {
    /// A stable key for this caller. OpenAI-compatible clients hash it into
    /// OpenRouter's session and end-user identifiers, and key cached reasoning
    /// by it. Defaults to the `Display` form.
    fn scope(&self) -> String {
        self.to_string()
    }

    /// Output-token ceiling for a tool-using step. `None` uses the client
    /// default. Raise it for callers whose tool arguments are large.
    fn tool_step_output_tokens(&self) -> Option<u32> {
        None
    }
}

impl Caller for str {}
impl Caller for String {}
impl<T: Caller + ?Sized> Caller for &T {
    fn scope(&self) -> String {
        (**self).scope()
    }
    fn tool_step_output_tokens(&self) -> Option<u32> {
        (**self).tool_step_output_tokens()
    }
}

/// A small structured completion: one system prompt and one user message.
#[derive(Debug, Clone, Copy)]
pub struct Completion<'a> {
    pub system: &'a str,
    pub user: &'a str,
    pub max_tokens: u32,
    /// Ask for provider-enforced JSON object output. Use only when the reply is
    /// parsed as a JSON object.
    pub json_object: bool,
    /// Ask for output matching a JSON Schema. Takes precedence over
    /// `json_object`.
    pub schema: Option<ResponseSchema<'a>>,
}

/// A JSON Schema the reply must follow (`response_format: json_schema`).
#[derive(Debug, Clone, Copy)]
pub struct ResponseSchema<'a> {
    /// A short name for the schema: letters, digits, `_` and `-`.
    pub name: &'a str,
    pub schema: &'a serde_json::Value,
    /// Ask the provider to enforce the schema exactly. OpenAI's strict mode
    /// requires every property to be listed in `required` and
    /// `additionalProperties: false` on every object.
    pub strict: bool,
}

impl<'a> Completion<'a> {
    pub fn new(system: &'a str, user: &'a str, max_tokens: u32) -> Self {
        Self {
            system,
            user,
            max_tokens,
            json_object: false,
            schema: None,
        }
    }

    pub fn json(mut self) -> Self {
        self.json_object = true;
        self
    }

    /// Ask for output matching `schema`. See [`crate::complete_typed`] to
    /// derive the schema from a Rust type and decode the reply.
    pub fn with_schema(mut self, name: &'a str, schema: &'a serde_json::Value) -> Self {
        self.schema = Some(ResponseSchema {
            name,
            schema,
            strict: false,
        });
        self
    }

    /// Ask the provider to enforce the schema exactly. No effect without one.
    pub fn strict(mut self) -> Self {
        if let Some(schema) = &mut self.schema {
            schema.strict = true;
        }
        self
    }
}

/// The capabilities an agent needs from a model.
#[async_trait]
pub trait ChatModel<C: ?Sized + Sync = str>: Send + Sync {
    /// A human-readable identifier for the model in use.
    fn model_id(&self) -> &str;

    /// Whether [`chat_step`](Self::chat_step) is implemented.
    fn supports_tools(&self) -> bool {
        true
    }

    /// One step of a tool-using turn: given the system prompt, the conversation
    /// so far and provider-neutral tool specs (`{name, description,
    /// parameters}`), return final text or tool calls.
    async fn chat_step(
        &self,
        caller: &C,
        system: SystemPrompt<'_>,
        history: &[ToolTurn],
        tools: &[serde_json::Value],
    ) -> Result<ChatStep>;

    /// Streaming variant of [`chat_step`](Self::chat_step). Visible text is
    /// sent to `sink` as it arrives. The default does not stream.
    async fn chat_step_stream(
        &self,
        caller: &C,
        system: SystemPrompt<'_>,
        history: &[ToolTurn],
        tools: &[serde_json::Value],
        _sink: Arc<dyn TextStreamSink>,
    ) -> Result<ChatStep> {
        self.chat_step(caller, system, history, tools).await
    }

    /// A small deterministic completion, such as classification or field
    /// extraction. Returns the trimmed reply text.
    async fn complete(&self, _caller: &C, _request: Completion<'_>) -> Result<String> {
        Err(Error::hard(format!(
            "{} does not support completions",
            self.model_id()
        )))
    }
}
