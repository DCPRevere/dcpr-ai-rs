# dcpr-ai-rs

Rust building blocks for tool-calling LLM agents.

| Crate | What it gives you |
| --- | --- |
| [`dcpr-llm`](crates/dcpr-llm) | Provider-neutral chat types, the `ChatModel` trait, an OpenAI-compatible client with OpenRouter support, a retrying fallback chain, and a scripted model for tests. |
| [`dcpr-agent`](crates/dcpr-agent) | A typed `Tool` contract with outcome statuses and argument validation, a tool-calling loop whose decisions are made by a `TurnPolicy`, and a ready-made `ToolAgent`. |

## dcpr-llm

```rust
use dcpr_llm::{ChatModel, Completion, NoUsage, OpenAiCompat};
use std::sync::Arc;

let model = OpenAiCompat::<str>::new(
    "https://openrouter.ai/api/v1",
    std::env::var("OPENROUTER_API_KEY")?,
    "anthropic/claude-haiku-5.5",
    Arc::new(NoUsage),
)
.with_responses_api(true);

let answer = model.complete("user-42", Completion::new("Reply with one word.", "Capital of France?", 16)).await?;
```

- **Callers.** Every request names a `Caller`. A `UsageSink` receives each call's token usage and the provider's reported cost, tagged with that caller, so you can bill or cap your own accounts.
- **OpenAI-compatible endpoints** use Chat Completions. **OpenRouter** can also use the Responses API for tool steps, with server-side tool search for deferred tools, and reasoning items echoed back between tool rounds.
- **Streaming.** `chat_step_stream` sends visible text to a `TextStreamSink` on both transports. Tool-call fragments are assembled before they reach you, and tool calls a model writes as Hermes XML in its text are never streamed as prose.
- **Structured output.** `complete_typed::<T>()` derives a JSON Schema from `T`, sends it as `response_format: json_schema`, and decodes the reply. On OpenRouter it routes only to endpoints that honour the schema. Pass your own schema with `Completion::with_schema`.
- **Reasoning.** Set an effort with `with_reasoning_effort`, or an exact token budget with `with_reasoning_budget`. A reasoning step that fails or returns nothing is retried once without reasoning.
- **Stop reasons.** A reply cut off by the token ceiling or a content filter, an incomplete Responses reply, or a refusal becomes `Error::Stopped` with a `StopReason`, instead of partial text. Early stops are retryable; refusals are not.
- **Robustness.** Errors are classified as transient, hard, context overflow or stopped. Tool calls the provider garbled are recovered from Hermes XML in the content, and malformed arguments are kept so the host can return a validation result.
- **Privacy.** OpenRouter requests always set `provider.data_collection = "deny"`. Session and end-user identifiers are hashed from the caller scope.
- **Prompt caching.** Mark a fixed prefix with `SystemPrompt::with_cacheable_prefix`. Anthropic models on OpenRouter get explicit cache breakpoints on both transports.
- **Rate limits.** `with_max_concurrent_requests` caps requests in flight across a client and its clones.
- **`Fallback`** tries a chain of models. Each gets one jittered retry of a transient failure, honouring `Retry-After`. A context overflow fails fast. A `ChainHealth` hook sees every outcome.

## dcpr-agent

```rust
use async_trait::async_trait;
use dcpr_agent::{Runtime, Tool, ToolAgent, ToolResult};
use serde_json::{Value, json};

struct App;
impl Runtime for App {
    type Context = ();        // passed to every call, e.g. the signed-in user
    type Error = String;
    type Extra = ();          // host data on results, never sent to the model
}

struct Clock;

#[async_trait]
impl Tool<App> for Clock {
    fn name(&self) -> &str { "current_time" }
    fn description(&self) -> &str { "The current UTC time." }
    fn parameters(&self) -> Value { json!({"type": "object", "properties": {}}) }
    async fn call(&self, _: &(), _: Value) -> Result<ToolResult, String> {
        Ok(ToolResult::observed("12:00 UTC"))
    }
}

let agent = ToolAgent::new("You are a helpful assistant.", vec![Box::new(Clock)]);
let run = agent.run(&model, "user-42", &(), vec![dcpr_llm::ToolTurn::User("What time is it?".into())]).await?;
println!("{}", run.reply);
```

- **Tools** declare a JSON Schema. Arguments are coerced and validated against it before a tool runs; unknown fields are rejected. Derive the schema and decoder from one Rust type with `typed_parameters` and `prepare_typed_args`.
- **Results** carry a `ToolOutcomeStatus` (`observed`, `applied`, `no_change`, `blocked` and so on), evidence, and an optional receipt that is safe to show the user. Reads are never reported as changes. Untrusted content is fenced before the model sees it.
- **`ToolAgent`** runs independent shared reads concurrently, reuses the result of an identical repeated call, and withdraws tools on the last step to ask for a final answer. `run_streaming` streams the final answer as it is written.
- **`run_turn` and `TurnPolicy`** are for agents that need their own rules. The loop owns the step budget, the deadline (which also abandons a slow model call) and streaming. The policy builds each step's prompt and tool list, executes calls, decides when a reply is final, and says what to return when time or steps run out.

## Testing

`ScriptedModel` replays a fixed sequence of text replies, tool calls and errors, and records what each step was sent, so a whole agent turn can be tested without a network.

## Licence

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
