# Provider verification, 2026-11-04

A live probe (`/tmp/dcpr-probe`, outside the repo) ran 19 scenarios per
configuration: `complete` (16 and 256 tokens), JSON mode, typed, strict and
strict-derived schemas, a truncated completion, text and tool steps, history
replay, a leading assistant turn, the agent loop (single, parallel and
streamed), streamed text and tools, an image, a bad key and a bad model.
Extra scenarios ran for the new features: strict typed output, pricing, and
model-written compaction.

Each provider was called with an ordinary API key; Ollama ran locally.

## Before and after (passing scenarios out of 19)

| Configuration | Before | After |
| --- | --- | --- |
| OpenAI gpt-6.1-sol, Responses | 9 | 17 |
| OpenAI gpt-6.1-sol, Chat Completions | 2 | 10 |
| OpenAI gpt-5.5, Chat Completions | 2 | 17 |
| OpenAI gpt-4.1-mini, Chat Completions | 16 | 17 |
| OpenAI gpt-6.1-terra / 6-sol / 6-luna / 5.6-sol, Responses | — | 17 each |
| OpenRouter openai/gpt-6.1-sol, both transports | 2 | 18 |
| OpenRouter claude-haiku-5.5, both transports | 18 / 17 | 18 / 18 |
| OpenRouter claude-sonnet-5.5 | — | 18 |
| OpenRouter gemini-3.8-flash, qwen3.6, glm-5.3-flash | 18 (qwen) | 18 each |
| Anthropic direct, claude-haiku-5-5 | 8 | 16 |
| Anthropic direct, claude-sonnet-5-5 / opus-5-5 | — | 16 |
| Gemini direct, gemini-3.5-flash-lite | 14 | 17 |
| Ollama llama3.1 | 15 | 15 |

New-feature scenarios (strict typed, pricing, model compaction, agent) pass
on all 8 configurations tried: OpenAI 6.1-sol, 6-luna and 4.1-mini; OpenRouter
6.1-sol and Haiku; Anthropic Haiku and Sonnet; Gemini 3.5-flash-lite.

## Measured provider rules

OpenAI, direct:

- Every reasoning model rejects `max_tokens` and `temperature: 0`.
- `reasoning_effort`: every reasoning model accepts `low`. `none` is rejected
  by gpt-6.1-sol and gpt-5-mini; `minimal` by gpt-5.5 and later. gpt-4.1
  rejects the field entirely.
- Function tools on Chat Completions while reasoning: refused from gpt-5.5 on
  (gpt-5.5, 5.6-sol, 6-sol, 6-luna and 6.1-terra need effort `none`;
  gpt-6.1-sol cannot use them at all). gpt-5-mini still allows them. So
  OpenAI tool steps need `with_responses_api(true)`.
- Responses rejects a replayed `function_call` whose `id` is not an `fc_…` id.
- Spent credit is a 429 with `type: insufficient_quota`,
  `code: credit_balance_exhausted`.

OpenRouter: `reasoning: {enabled: false}` is rejected ("Reasoning is
mandatory") for exactly the OpenAI models without a `none` effort.
`require_parameters` plus `temperature` finds no endpoint for Sonnet 5.5 or
gpt-6.1-sol.

Anthropic: `temperature` is rejected from Opus 4.7 on, including every 5.x
model. Opus 4.6 and Sonnet 4.6 accept it.

Gemini 3: a replayed tool call needs its `thought_signature`. One-word
completions took 15 to 20 seconds under load, at any effort. gemini-3.8-flash
returned 503 "high demand" for most of the day.

## What still fails, and why

- **OpenAI Chat Completions with tools on gpt-6.x and 5.6**: the API refuses
  them. The Responses API is now the default on OpenAI, so this only happens
  with `with_responses_api(false)`. With defaults, gpt-6.1-sol and gpt-6-luna
  pass 17/19 and gpt-4.1-mini 18/19.
- **`truncated_complete`**: `complete()` now returns `Error::Stopped` for a
  cut-off reply. The probe still marks it failed where the reply was never
  cut off: OpenAI reasoning models get 4,096 tokens of headroom, and
  OpenRouter returned whole replies for Sonnet 5.5 and gpt-6.1-sol.
- **`typed_strict_derived`**: strict mode with the plain `json_schema_for`
  schema fails on OpenAI and Anthropic, as documented. Use
  `complete_typed_strict` or `strict_json_schema_for`.
- **Anthropic direct, `complete_json` and non-strict `typed`**: that endpoint
  only takes strict `json_schema`. Use `complete_typed_strict` there.
- **Ollama**: llama3.1 has no vision and ignores the API key; one model-quality
  miss on the leading-assistant case.
- **Gemini 3.8-flash**: Google capacity (503).

## Findings left for later

- `Usage` has no reasoning-token field; reasoning is inside `output_tokens`.
  Adding one breaks struct literals, so it needs a minor release of its own.
- Hosts other than `api.openai.com` serving OpenAI models (proxies, LiteLLM,
  Azure) still get Chat Completions' `max_tokens`. A `with_host` override
  would cover them.
- Batch usage is recorded at full price; batch rows cost half.
- Claude on OpenRouter's Responses API sometimes distrusts the injected
  replay-boundary message when history starts with an assistant turn (seen
  once in three runs).
- The compactor estimates tokens at four characters each. Pair it with
  `PriceTable::context_tokens` to set a budget from the model's window.
