# Changelog

## 0.3.0 (unreleased)

Works against OpenAI directly, and is verified live against OpenAI,
OpenRouter, Anthropic's and Gemini's OpenAI-compatible endpoints, and Ollama.

### dcpr-llm

Fixed

- OpenAI rejected every request. Requests to `api.openai.com` now send
  `max_completion_tokens`, drop `temperature` for reasoning models, and carry
  the configured effort as `reasoning_effort` (Chat Completions) or
  `reasoning.effort` (Responses), with output headroom for the reasoning.
  Small completions use effort `low`, the one every OpenAI reasoning model
  accepts.
- The Responses API now works on OpenAI. A replayed `function_call` carries
  OpenAI's own `fc_…` item id, or none, never the call id. Encrypted
  reasoning is requested and echoed between tool steps.
- OpenRouter models whose reasoning cannot be turned off (`openai/gpt-6.1-sol`,
  `openai/gpt-5-mini`, the o-series) failed every call. They now get low effort,
  and any other model that answers "Reasoning is mandatory" is learned and
  retried once.
- Gemini 3 tool loops failed on the second step. Each tool call's
  `thought_signature` is now sent back, with Gemini's documented placeholder
  for history from before the client started.
- `temperature` is dropped, on every host, for models that reject it: Claude
  Opus 4.7 and later, every Claude 5.x model, and OpenAI reasoning models. On
  OpenRouter, `require_parameters` otherwise found no endpoint for them.
- An empty `tools` list is left out; Anthropic's endpoint rejected it.
- OpenAI's spent-credit 429 (`insufficient_quota`) is a hard error, not a retry.
- `complete()` reports a reply cut off by the token ceiling or a content
  filter as `Error::Stopped`, as tool steps already did, instead of returning
  partial text. Raise the ceiling, or ask for a short reply in the prompt.
- Gemini's array-wrapped error bodies are read like any other.
- Error messages from non-OpenRouter hosts no longer say "OpenRouter Responses".

Added

- `PriceTable`, `ModelPrice` and `PricedUsage`: list prices for providers that
  do not report cost, built in and refreshable from OpenRouter's catalogue.
- `strict_json_schema_for` and `complete_typed_strict`, for OpenAI strict mode
  and Anthropic's endpoint.
- `OpenAiCompat::with_completion_timeout` (default 30 seconds).
- Requests to OpenAI carry hashed `safety_identifier` and `prompt_cache_key`
  values derived from the caller scope.
- Batch tool steps use `/v1/responses` when the Responses API is on.
- `ScriptedModel` answers `complete()` from its script.
- An opt-in live test against OpenAI (`DCPR_LIVE=1`).

Changed

- Tool steps on `api.openai.com` use the Responses API unless
  `with_responses_api(false)` says otherwise. Other hosts keep Chat
  Completions by default.
- `with_fallback_models` warns off OpenRouter, where it has no effect.
- OpenRouter tool search is only added on OpenRouter; elsewhere deferred tools
  are sent in full.

### dcpr-agent

Added

- `HistoryCompactor`: shrinks a turn's history to a token budget while keeping
  the task, condensing old tool exchanges into evidence notes, optionally
  written by a model (`compact_with_model`).
- `cap_output`, `estimate_tokens` and `turn_tokens`.
- `ToolAgent::with_history_budget` and `with_compactor`; a context overflow
  compacts and retries once.
- `Tool::output_limit` and `ToolAgent::with_tool_output_limit`.

Changed

- `ToolAgent` caps each tool output the model sees at 16,000 characters,
  keeping its start and end. `with_tool_output_limit(None)` restores whole
  outputs. The host's `ExecutedCall` results are unchanged.
