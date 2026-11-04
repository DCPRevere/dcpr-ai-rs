//! A bridge between [`Tool`] and the Model Context Protocol, built on
//! [`rmcp`], the official Rust MCP SDK.
//!
//! The protocol, its transports and sessions all stay in `rmcp`. This module
//! only converts between the two tool contracts:
//!
//! - [`McpTool`] wraps a tool on a connected MCP server as a [`Tool`], so an
//!   agent can call it like a local one.
//! - [`ToolServer`] is an `rmcp` [`ServerHandler`] that serves a set of
//!   [`Tool`]s over whichever `rmcp` transport the host picks.
//!
//! ```no_run
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! use dcpr_agent::mcp::McpTool;
//! use rmcp::ServiceExt;
//! use rmcp::transport::TokioChildProcess;
//!
//! # struct App;
//! # impl dcpr_agent::Runtime for App { type Context = (); type Error = String; type Extra = (); }
//! let client = ()
//!     .serve(TokioChildProcess::new(tokio::process::Command::new("weather-mcp"))?)
//!     .await?;
//! let tools: Vec<Box<dyn dcpr_agent::Tool<App>>> = McpTool::list(&client)
//!     .await?
//!     .into_iter()
//!     .map(|tool| tool.with_prefix("weather").boxed())
//!     .collect();
//! # Ok(()) }
//! ```

use crate::{ResultExtra, Runtime, Tool, ToolOutcomeStatus, ToolResult};
use async_trait::async_trait;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, JsonObject,
    ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerConfig, ToolAnnotations,
};
use rmcp::service::{RequestContext, RoleClient, RoleServer};
use rmcp::{ErrorData, Peer, ServerHandler, ServiceError};
use serde_json::{Value, json};
use std::sync::Arc;

/// A tool on a connected MCP server, callable as a [`Tool`] for any
/// [`Runtime`].
///
/// Results are marked untrusted by default, since a third-party server's
/// output must never be read as instructions. Transport failures and tool
/// errors become [`ToolOutcomeStatus::Failed`] results the model can see.
#[derive(Clone)]
pub struct McpTool {
    peer: Peer<RoleClient>,
    name: String,
    remote_name: String,
    description: String,
    parameters: Value,
    read_only: bool,
    trusted: bool,
}

impl McpTool {
    /// Every tool the server lists, following pagination.
    pub async fn list(peer: &Peer<RoleClient>) -> Result<Vec<Self>, ServiceError> {
        Ok(peer
            .list_all_tools()
            .await?
            .into_iter()
            .map(|tool| Self::new(peer.clone(), tool))
            .collect())
    }

    /// Wrap one tool definition from `peer`.
    pub fn new(peer: Peer<RoleClient>, tool: rmcp::model::Tool) -> Self {
        let read_only = tool
            .annotations
            .as_ref()
            .and_then(|annotations| annotations.read_only_hint)
            .unwrap_or(false);
        Self {
            peer,
            name: tool.name.to_string(),
            remote_name: tool.name.to_string(),
            description: tool.description.as_deref().unwrap_or_default().to_string(),
            parameters: Value::Object((*tool.input_schema).clone()),
            read_only,
            trusted: false,
        }
    }

    /// Expose the tool as `{prefix}__{name}`, so tools from several servers
    /// cannot collide. Calls still use the server's own name.
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.name = format!("{prefix}__{}", self.remote_name);
        self
    }

    /// Treat this server's output as trusted, so it is not fenced as
    /// third-party data. Only for servers you operate.
    pub fn trusted(mut self) -> Self {
        self.trusted = true;
        self
    }

    /// The tool's name on its server.
    pub fn remote_name(&self) -> &str {
        &self.remote_name
    }

    pub fn boxed<R: Runtime>(self) -> Box<dyn Tool<R>> {
        Box::new(self)
    }

    fn failed<X: ResultExtra>(&self, message: impl std::fmt::Display) -> ToolResult<X> {
        ToolResult::terminal(
            ToolOutcomeStatus::Failed,
            format!("{} failed: {message}", self.name),
        )
    }
}

#[async_trait]
impl<R: Runtime> Tool<R> for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    /// A server that does not mark a tool read-only may change state with it.
    fn is_action(&self) -> bool {
        !self.read_only
    }

    async fn call(&self, _ctx: &R::Context, args: Value) -> Result<ToolResult<R::Extra>, R::Error> {
        let mut params = CallToolRequestParams::new(self.remote_name.clone());
        if let Value::Object(arguments) = args {
            params = params.with_arguments(arguments);
        }
        let result = match self.peer.call_tool_once(params).await {
            Ok(CallToolResponse::Complete(result)) => result,
            Ok(CallToolResponse::InputRequired(_)) => {
                return Ok(self.failed("the server asked for input this agent cannot give"));
            }
            Ok(CallToolResponse::Task(_)) => {
                return Ok(
                    self.failed("the server started a background task, which is not supported")
                );
            }
            Ok(_) => {
                return Ok(self.failed("the server sent a response this agent does not understand"));
            }
            Err(error) => return Ok(self.failed(error)),
        };
        let output = result_text(&result);
        if result.is_error == Some(true) {
            return Ok(self.failed(output));
        }
        let mut tool_result = ToolResult::text(output);
        if let Some(structured) = result.structured_content {
            tool_result = tool_result.with_evidence(json!({ "structured_content": structured }));
        }
        if !self.trusted {
            tool_result = tool_result.with_untrusted_content();
        }
        Ok(tool_result)
    }
}

/// The text of a call result. Non-text blocks are named, not dropped, so the
/// model knows they existed. Structured content stands in for empty text.
fn result_text(result: &CallToolResult) -> String {
    let parts: Vec<String> = result
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::Text(text) => text.text.clone(),
            ContentBlock::Image(image) => format!("[image: {}]", image.mime_type),
            ContentBlock::Audio(audio) => format!("[audio: {}]", audio.mime_type),
            ContentBlock::Resource(_) => "[embedded resource]".to_string(),
            ContentBlock::ResourceLink(link) => format!("[resource: {}]", link.uri),
            _ => "[unsupported content]".to_string(),
        })
        .collect();
    if parts.iter().all(|part| part.trim().is_empty())
        && let Some(structured) = &result.structured_content
    {
        return structured.to_string();
    }
    parts.join("\n")
}

type ContextFn<R> = dyn Fn(&RequestContext<RoleServer>) -> Result<Arc<<R as Runtime>::Context>, ErrorData>
    + Send
    + Sync;

/// Serves [`Tool`]s as an MCP server. Pass it to any `rmcp` transport, for
/// example `rmcp::transport::stdio` or the streamable HTTP service.
///
/// Each call builds its context with the function given to [`Self::new`],
/// which sees the request, including HTTP request parts that `rmcp` puts in
/// [`RequestContext::extensions`]. Return an error there to refuse the call,
/// for example when authentication fails.
pub struct ToolServer<R: Runtime> {
    tools: Arc<[Box<dyn Tool<R>>]>,
    context: Arc<ContextFn<R>>,
    info: ServerConfig,
}

impl<R: Runtime> Clone for ToolServer<R> {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
            context: self.context.clone(),
            info: self.info.clone(),
        }
    }
}

impl<R: Runtime> ToolServer<R> {
    pub fn new(
        tools: Vec<Box<dyn Tool<R>>>,
        context: impl Fn(&RequestContext<RoleServer>) -> Result<Arc<R::Context>, ErrorData>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            tools: tools.into(),
            context: Arc::new(context),
            info: ServerConfig::new(ServerCapabilities::builder().enable_tools().build()),
        }
    }

    /// Every call shares one context.
    pub fn with_shared_context(tools: Vec<Box<dyn Tool<R>>>, context: Arc<R::Context>) -> Self {
        Self::new(tools, move |_| Ok(context.clone()))
    }

    /// Replace the server's name, version, instructions and capabilities.
    pub fn with_info(mut self, info: ServerConfig) -> Self {
        self.info = info;
        self
    }

    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.info = self.info.with_instructions(instructions);
        self
    }

    fn tool(&self, name: &str) -> Option<&dyn Tool<R>> {
        crate::find(&self.tools, name)
    }
}

/// The MCP definition of a tool. Read-only unless it is an action.
pub fn mcp_definition<R: Runtime>(tool: &dyn Tool<R>) -> rmcp::model::Tool {
    let schema: JsonObject = match tool.parameters() {
        Value::Object(schema) => schema,
        _ => JsonObject::new(),
    };
    rmcp::model::Tool::new(
        tool.name().to_string(),
        tool.description().to_string(),
        schema,
    )
    .with_annotations(ToolAnnotations::new().read_only(!tool.is_action()))
}

/// The MCP result for a [`ToolResult`]. The text is the tool's output, fenced
/// when untrusted. The structured content carries the typed status and
/// evidence. Any status other than a read, a change or no change is an error.
pub fn mcp_result<X: ResultExtra>(result: &ToolResult<X>) -> CallToolResult {
    let text = if result.is_untrusted() {
        X::fence_untrusted(&result.output)
    } else {
        result.output.clone()
    };
    let succeeded = matches!(
        result.status,
        ToolOutcomeStatus::Observed | ToolOutcomeStatus::Applied | ToolOutcomeStatus::NoChange
    );
    let mut mcp = if succeeded {
        CallToolResult::success(vec![ContentBlock::text(text)])
    } else {
        CallToolResult::error(vec![ContentBlock::text(text)])
    };
    mcp.structured_content = Some(json!({
        "status": result.status,
        "state_changed": result.changed(),
        "evidence": if result.is_untrusted() { Value::Null } else { result.evidence.clone() },
        "user_receipt": result.user_receipt,
    }));
    mcp
}

impl<R: Runtime> ServerHandler for ToolServer<R> {
    fn get_info(&self) -> ServerConfig {
        self.info.clone()
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(
            self.tools
                .iter()
                .map(|tool| mcp_definition(tool.as_ref()))
                .collect(),
        ))
    }

    fn get_tool(&self, name: &str) -> Option<rmcp::model::Tool> {
        self.tool(name).map(mcp_definition)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let Some(tool) = self.tool(&request.name) else {
            return Err(ErrorData::invalid_params(
                format!("unknown tool: {}", request.name),
                None,
            ));
        };
        let ctx = (self.context)(&context)?;
        let args = match tool
            .prepare_arguments(Value::Object(request.arguments.unwrap_or_default()))
        {
            Ok(args) => args,
            // Spec: input validation errors are tool errors, so the model can
            // correct itself, not protocol errors.
            Err(error) => {
                let result =
                    CallToolResult::error(vec![ContentBlock::text(error.for_model(tool.name()))]);
                return Ok(result.into());
            }
        };
        let outcome = tokio::select! {
            outcome = tool.call(&ctx, args) => outcome,
            () = context.ct.cancelled() => {
                return Err(ErrorData::internal_error("request cancelled", None));
            }
        };
        let result = match outcome {
            Ok(result) => mcp_result(&result.with_runtime_semantics(tool.is_action())),
            Err(error) => {
                tracing::warn!(tool = %request.name, "tool call failed: {error}");
                CallToolResult::error(vec![ContentBlock::text(format!(
                    "{} failed: {error}",
                    request.name
                ))])
            }
        };
        Ok(result.into())
    }
}
