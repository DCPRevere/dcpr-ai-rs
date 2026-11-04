# dcpr-ai-rs

Rust building blocks for tool-calling LLM agents.

| Crate | What it gives you |
| --- | --- |
| [`dcpr-llm`](crates/dcpr-llm) | Provider-neutral chat types, the `ChatModel` trait, an OpenAI-compatible client with OpenRouter support, Batch API support, a retrying fallback chain, and a scripted model for tests. |
| [`dcpr-agent`](crates/dcpr-agent) | A typed `Tool` contract with outcome statuses and argument validation, a tool-calling loop whose decisions are made by a `TurnPolicy`, a ready-made `ToolAgent`, and an MCP bridge. |

## Install

```toml
[dependencies]
dcpr-llm = { git = "https://github.com/DCPRevere/dcpr-ai-rs", tag = "v0.3.0" }
dcpr-agent = { git = "https://github.com/DCPRevere/dcpr-ai-rs", tag = "v0.3.0", features = ["mcp"] }
```

Optional features: `dcpr-llm/batch` (the OpenAI Batch API, through `async-openai`) and `dcpr-agent/mcp` (MCP, through `rmcp`).

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
- **Providers.** Verified live against OpenAI, OpenRouter, Anthropic's and Gemini's OpenAI-compatible endpoints, and Ollama. Tool steps use the Responses API on OpenAI by default (from gpt-5.5 on, its Chat Completions refuses function tools while the model reasons) and Chat Completions elsewhere; `with_responses_api` chooses explicitly, and OpenRouter supports both. Each host gets its own rules: OpenAI's `max_completion_tokens`, effort fields and item ids; Gemini's thought signatures; dropping `temperature` where a model rejects it. OpenRouter adds server-side tool search for deferred tools, and reasoning is echoed back between tool rounds on both.
- **Streaming.** `chat_step_stream` sends visible text to a `TextStreamSink` on both transports. Tool-call fragments are assembled before they reach you, and tool calls a model writes as Hermes XML in its text are never streamed as prose.
- **Structured output.** `complete_typed::<T>()` derives a JSON Schema from `T`, sends it as `response_format: json_schema`, and decodes the reply. On OpenRouter it routes only to endpoints that honour the schema. `complete_typed_strict` sends a strict schema (`strict_json_schema_for`), which OpenAI enforces exactly and Anthropic's endpoint requires. Pass your own schema with `Completion::with_schema`.
- **Reasoning.** Set an effort with `with_reasoning_effort`, or an exact token budget with `with_reasoning_budget` (OpenAI has no budget: it asks for medium effort). The output ceiling grows to leave room for reasoning. A reasoning step that fails or returns nothing is retried once without reasoning. Small completions use the lowest effort a model accepts.
- **Stop reasons.** A reply cut off by the token ceiling or a content filter, an incomplete Responses reply, or a refusal becomes `Error::Stopped` with a `StopReason`, instead of partial text, on tool steps and completions alike. Early stops are retryable; refusals are not.
- **Robustness.** Errors are classified as transient, hard, context overflow or stopped. Tool calls the provider garbled are recovered from Hermes XML in the content, and malformed arguments are kept so the host can return a validation result.
- **Privacy.** OpenRouter requests always set `provider.data_collection = "deny"`. Session and end-user identifiers are hashed from the caller scope.
- **Prompt caching.** Mark a fixed prefix with `SystemPrompt::with_cacheable_prefix`. Anthropic models on OpenRouter get explicit cache breakpoints on both transports.
- **Rate limits.** `with_max_concurrent_requests` caps requests in flight across a client and its clones. OpenAI's spent-credit 429 (`insufficient_quota`) is a hard error, not a retry.
- **Cost.** OpenRouter reports each call's cost. For other providers, `PricedUsage` prices calls from a `PriceTable` of list prices (built in, refreshable from OpenRouter's catalogue) before they reach your sink.
- **Batches** (feature `batch`). `batch_completion` and `batch_step` build requests exactly as the live calls do; `submit_batch`, `wait_for_batch` and `batch_results` run them through any endpoint with the OpenAI Batch API. With the Responses API on, tool steps target `/v1/responses`; keep them in a different batch from completions. Each reply goes through the live checks: usage is recorded, early stops and refusals become errors, and tool calls are parsed. File and batch calls go through `async-openai`.
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
- **`ToolAgent`** runs independent shared reads concurrently, reuses the result of an identical repeated call, and withdraws tools on the last step to ask for a final answer. `run_streaming` streams the final answer as it is written. Each tool output is capped at 16,000 characters, keeping its start and end (`with_tool_output_limit`, or `Tool::output_limit` per tool).
- **History.** `HistoryCompactor` keeps a long turn inside a token budget without losing the task: the user's request is pinned, recent tool exchanges stay verbatim, and older ones are condensed into evidence notes, written by a model with `compact_with_model`. `ToolAgent::with_history_budget` compacts before each step and once more after a context overflow. `cap_output` shortens any text the same way.
- **`run_turn` and `TurnPolicy`** are for agents that need their own rules. The loop owns the step budget, the deadline (which also abandons a slow model call) and streaming. The policy builds each step's prompt and tool list, executes calls, decides when a reply is final, and says what to return when time or steps run out.

### MCP

The `mcp` feature bridges `Tool` and the Model Context Protocol through [`rmcp`](https://crates.io/crates/rmcp), the official Rust SDK. Transports and sessions stay in `rmcp`.

- **`McpTool`** wraps each tool on a connected server as a `Tool`, so an agent calls it like a local one. `with_prefix` keeps names from several servers apart. Output is fenced as untrusted unless you mark the server `trusted()`, and tools the server does not mark read-only count as actions.
- **`ToolServer`** is an `rmcp` `ServerHandler` that serves your `Tool`s over stdio, streamable HTTP or any other `rmcp` transport. A context function sees each request, including HTTP parts, so it can authenticate the caller or refuse. Results carry the typed status and evidence as structured content.

```rust
use dcpr_agent::mcp::{McpTool, ToolServer};
use rmcp::ServiceExt;

// Call a server's tools from an agent.
let client = ().serve(rmcp::transport::TokioChildProcess::new(tokio::process::Command::new("weather-mcp"))?).await?;
let tools: Vec<Box<dyn Tool<App>>> = McpTool::list(&client).await?
    .into_iter().map(|tool| tool.with_prefix("weather").boxed()).collect();

// Serve your own tools over stdio.
ToolServer::<App>::with_shared_context(vec![Box::new(Clock)], Arc::new(()))
    .serve(rmcp::transport::stdio()).await?.waiting().await?;
```

## Testing

`ScriptedModel` replays a fixed sequence of text replies, tool calls and errors, for tool steps and completions alike, and records what each was sent, so a whole agent turn can be tested without a network.

Live OpenAI checks are opt-in: `DCPR_LIVE=1 OPENAI_API_KEY=… cargo test -p dcpr-llm --test live_openai`.

## Licence

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
