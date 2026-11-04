//! A typed tool contract and a hookable tool-calling loop for LLM agents,
//! built on [`dcpr_llm`].
//!
//! - [`Tool`] is what a model may call. Tools are generic over a [`Runtime`],
//!   which names the host's context, error and result-extra types.
//! - [`ToolResult`] carries a typed [`ToolOutcomeStatus`] and evidence, so an
//!   agent can tell a read from a change and never claim work that did not
//!   happen.
//! - [`run_turn`] drives a turn and hands every decision to a [`TurnPolicy`].
//! - [`ToolAgent`] is a ready-made policy for the common case.
//! - `mcp` (feature `mcp`) bridges tools to and from MCP servers through
//!   `rmcp`, the official Rust SDK.

mod agent;
#[cfg(feature = "mcp")]
pub mod mcp;
mod schema;
mod tool;
mod turn;

pub use agent::{AgentError, AgentRun, ExecutedCall, ToolAgent};
pub use schema::{
    coerce_args_to_schema, decode_typed_args, prepare_typed_args, typed_parameters,
    validate_args_against_schema,
};
pub use tool::{
    ResultExtra, Runtime, SharedCall, Tool, ToolConcurrency, ToolInputError, ToolOutcomeStatus,
    ToolResult, call_shared, confirmed_output, fence_untrusted, find,
};
pub use turn::{Flow, LoopEvent, StepInfo, StepRequest, TurnLimits, TurnPolicy, run_turn};

pub use dcpr_llm;
