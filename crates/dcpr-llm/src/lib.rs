//! Provider-neutral chat types for tool-calling LLM agents, plus an
//! OpenAI-compatible client with first-class OpenRouter support.
//!
//! - [`ChatModel`] is the one trait an agent needs from a model: a single
//!   tool-calling step, an optional streaming variant, and small structured
//!   completions.
//! - [`OpenAiCompat`] speaks Chat Completions to any OpenAI-compatible endpoint
//!   and the Responses API to OpenRouter, with reasoning control, server-side
//!   model fallback, prompt caching for Anthropic models, and Hermes-XML
//!   tool-call recovery.
//! - [`Fallback`] retries transient failures and falls through a chain of
//!   models.
//! - [`complete_typed`] asks for a reply matching a JSON Schema derived from a
//!   Rust type, and decodes it.
//! - Batch API support (feature `batch`) builds batched requests exactly as
//!   live ones and reads replies back through the same checks.
//! - [`ScriptedModel`] replays a fixed script of steps for deterministic tests.
//!
//! - [`PriceTable`] prices calls from list prices for providers that do not
//!   report cost, such as OpenAI; [`PricedUsage`] applies it to a sink.
//!
//! Callers are generic. A [`Caller`] says who a request is for, so usage can be
//! attributed through a [`UsageSink`] without the library knowing anything about
//! accounts or tenants.

mod error;
mod fallback;
mod model;
pub mod openai_compat;
mod pricing;
mod scripted;
mod structured;
mod types;
mod usage;

pub use error::{Error, Result, StopReason, is_context_overflow_body};
pub use fallback::{ChainHealth, Fallback, Retryable};
pub use model::{Caller, ChatModel, Completion, ResponseSchema};
#[cfg(feature = "batch")]
pub use openai_compat::{BatchItem, BatchJob, BatchRequest, BatchStatus};
pub use openai_compat::{OpenAiCompat, OpenRouterProviderSort, RequestSize, parse_retry_after};
pub use pricing::{ModelPrice, PriceTable, PricedUsage};
pub use scripted::ScriptedModel;
pub use structured::{
    complete_typed, complete_typed_strict, decode_json, json_schema_for, strict_json_schema_for,
};
pub use types::{ChatStep, InputImage, SystemPrompt, TextStreamSink, ToolCall, ToolTurn};
pub use usage::{NoUsage, Usage, UsageSink, openrouter_cost_cents};
