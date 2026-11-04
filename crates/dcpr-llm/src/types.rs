//! Provider-neutral conversation and tool-call types.

/// Receives user-visible text as a provider generates it. Implementations must
/// never receive reasoning or function-call arguments.
pub trait TextStreamSink: Send + Sync {
    /// Discard text streamed so far, for example before a retry.
    fn reset(&self);
    fn delta(&self, text: &str);
}

/// A tool call the model wants to make: provider call id, tool name, arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// One step of a tool-using conversation: the model produced final text, or it
/// wants to call tools, which the caller executes and feeds back.
#[derive(Debug, Clone)]
pub enum ChatStep {
    Text(String),
    ToolCalls(Vec<ToolCall>),
}

/// An image attached to a user turn, for vision models.
#[derive(Debug, Clone)]
pub struct InputImage {
    /// For example "image/jpeg" or "image/png".
    pub mime: String,
    /// Base64-encoded image bytes.
    pub data_b64: String,
}

/// A turn in the tool-loop conversation history.
#[derive(Debug, Clone)]
pub enum ToolTurn {
    /// The user's message.
    User(String),
    /// The user's message with attached images.
    UserWithImages(String, Vec<InputImage>),
    /// A plain assistant reply, for example when replaying stored history.
    Assistant(String),
    /// The assistant requested these tool calls.
    AssistantToolCalls(Vec<ToolCall>),
    /// Results for tool calls, as (call id, output) pairs.
    ToolResults(Vec<(String, String)>),
}

/// Trusted instructions with an optional boundary after their fixed prefix.
/// Providers may use the boundary for prompt caching. Providers without
/// explicit cache boundaries receive the same full text.
#[derive(Debug, Clone, Copy)]
pub struct SystemPrompt<'a> {
    text: &'a str,
    cacheable_prefix_len: usize,
}

impl<'a> SystemPrompt<'a> {
    /// # Panics
    ///
    /// Panics if `prefix_len` is not on a UTF-8 character boundary.
    pub fn with_cacheable_prefix(text: &'a str, prefix_len: usize) -> Self {
        assert!(
            text.is_char_boundary(prefix_len),
            "invalid system prompt cache boundary"
        );
        Self {
            text,
            cacheable_prefix_len: prefix_len,
        }
    }

    pub fn as_str(self) -> &'a str {
        self.text
    }

    /// The fixed prefix and the remaining suffix, when a boundary was set.
    pub fn cacheable_prefix(self) -> Option<(&'a str, &'a str)> {
        (self.cacheable_prefix_len > 0).then(|| self.text.split_at(self.cacheable_prefix_len))
    }
}

impl<'a> From<&'a str> for SystemPrompt<'a> {
    fn from(text: &'a str) -> Self {
        Self {
            text,
            cacheable_prefix_len: 0,
        }
    }
}
