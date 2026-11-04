//! A ready-made agent: a system prompt, a set of tools, and the standard
//! execution rules.

use async_trait::async_trait;
use dcpr_llm::{ChatModel, TextStreamSink, ToolCall, ToolTurn};
use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    Flow, HistoryCompactor, LoopEvent, Runtime, SharedCall, StepInfo, StepRequest, Tool,
    ToolConcurrency, ToolOutcomeStatus, ToolResult, TurnLimits, TurnPolicy, call_shared,
    cap_output, estimate_tokens, find, run_turn,
};

/// Default cap on one tool output, in characters (about 4,000 tokens).
pub const DEFAULT_TOOL_OUTPUT_CHARS: usize = 16_000;

/// A completed call, as executed this turn.
#[derive(Debug, Clone)]
pub struct ExecutedCall<X> {
    pub call: ToolCall,
    pub result: ToolResult<X>,
}

/// The outcome of [`ToolAgent::run`].
#[derive(Debug, Clone)]
pub struct AgentRun<X> {
    pub reply: String,
    /// Every tool call this turn, in order, with the arguments actually used.
    pub calls: Vec<ExecutedCall<X>>,
    pub steps: usize,
}

impl<X> AgentRun<X> {
    /// Whether any action changed state.
    pub fn acted(&self) -> bool {
        self.calls
            .iter()
            .any(|c| c.result.status == ToolOutcomeStatus::Applied)
    }
}

/// An agent with a fixed system prompt and tool set.
///
/// Each step offers every tool. Arguments are prepared through
/// [`Tool::prepare_arguments`] before execution. A batch runs concurrently
/// only when every call is to a [`ToolConcurrency::Shared`] tool. An identical
/// repeat of a call this turn gets the earlier result back with a
/// [`ToolOutcomeStatus::Reused`] status instead of running again; a successful
/// action clears cached reads, since state may have changed. On the last step
/// tools are withdrawn and the model is asked for a final answer.
///
/// Each tool output the model sees is capped, keeping its start and end (see
/// [`with_tool_output_limit`](Self::with_tool_output_limit)). With
/// [`with_history_budget`](Self::with_history_budget) the history is compacted
/// before each step, keeping the user's request, and a context overflow
/// compacts harder and tries again.
pub struct ToolAgent<R: Runtime> {
    system: String,
    tools: Vec<Box<dyn Tool<R>>>,
    max_steps: usize,
    timeout: Option<std::time::Duration>,
    output_limit: Option<usize>,
    compactor: Option<HistoryCompactor>,
}

impl<R: Runtime> ToolAgent<R> {
    pub fn new(system: impl Into<String>, tools: Vec<Box<dyn Tool<R>>>) -> Self {
        Self {
            system: system.into(),
            tools,
            max_steps: 10,
            timeout: None,
            output_limit: Some(DEFAULT_TOOL_OUTPUT_CHARS),
            compactor: None,
        }
    }

    /// Most characters of one tool output the model sees, unless the tool
    /// sets its own [`Tool::output_limit`]. Defaults to
    /// [`DEFAULT_TOOL_OUTPUT_CHARS`]; `None` sends outputs whole.
    pub fn with_tool_output_limit(mut self, chars: Option<usize>) -> Self {
        self.output_limit = chars;
        self
    }

    /// Compact the history before any step whose estimate exceeds `tokens`.
    /// See [`HistoryCompactor`].
    pub fn with_history_budget(self, tokens: usize) -> Self {
        self.with_compactor(HistoryCompactor::new(tokens))
    }

    pub fn with_compactor(mut self, compactor: HistoryCompactor) -> Self {
        self.compactor = Some(compactor);
        self
    }

    /// Model steps allowed per turn. Defaults to 10.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps.max(1);
        self
    }

    /// Wall-clock budget per turn.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn tools(&self) -> &[Box<dyn Tool<R>>] {
        &self.tools
    }

    /// Run one turn. `history` holds earlier turns and ends with the user's
    /// message.
    pub async fn run<C, M>(
        &self,
        model: &M,
        caller: &C,
        ctx: &R::Context,
        history: Vec<ToolTurn>,
    ) -> Result<AgentRun<R::Extra>, AgentError<R::Error>>
    where
        C: ?Sized + Sync,
        M: ChatModel<C> + ?Sized,
    {
        self.run_inner(model, caller, ctx, history, None).await
    }

    /// Like [`run`](Self::run), streaming visible reply text to `sink` as the
    /// model writes it. The sink is reset when a step turns out to be tool
    /// calls, so only the final answer remains.
    pub async fn run_streaming<C, M>(
        &self,
        model: &M,
        caller: &C,
        ctx: &R::Context,
        history: Vec<ToolTurn>,
        sink: Arc<dyn TextStreamSink>,
    ) -> Result<AgentRun<R::Extra>, AgentError<R::Error>>
    where
        C: ?Sized + Sync,
        M: ChatModel<C> + ?Sized,
    {
        self.run_inner(model, caller, ctx, history, Some(sink))
            .await
    }

    async fn run_inner<C, M>(
        &self,
        model: &M,
        caller: &C,
        ctx: &R::Context,
        mut history: Vec<ToolTurn>,
        sink: Option<Arc<dyn TextStreamSink>>,
    ) -> Result<AgentRun<R::Extra>, AgentError<R::Error>>
    where
        C: ?Sized + Sync,
        M: ChatModel<C> + ?Sized,
    {
        // The request that started this turn, pinned through compaction.
        let task = history
            .iter()
            .rposition(|turn| matches!(turn, ToolTurn::User(_) | ToolTurn::UserWithImages(..)));
        let mut policy = Policy {
            sink,
            agent: self,
            ctx,
            task,
            overflow_retries: 0,
            specs: self.tools.iter().map(|t| t.spec()).collect(),
            calls: Vec::new(),
            reads: HashMap::new(),
            actions: HashMap::new(),
            steps: 0,
        };
        let mut limits = TurnLimits::steps(self.max_steps);
        if let Some(timeout) = self.timeout {
            limits = limits.with_timeout(timeout);
        }
        run_turn(model, caller, &mut policy, &mut history, limits).await
    }
}

/// Why a turn failed.
#[derive(Debug)]
pub enum AgentError<E> {
    Model(dcpr_llm::Error),
    Tool(E),
}

impl<E> From<dcpr_llm::Error> for AgentError<E> {
    fn from(error: dcpr_llm::Error) -> Self {
        Self::Model(error)
    }
}

impl<E: std::fmt::Display> std::fmt::Display for AgentError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(e) => e.fmt(f),
            Self::Tool(e) => write!(f, "tool error: {e}"),
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for AgentError<E> {}

struct Policy<'a, R: Runtime> {
    agent: &'a ToolAgent<R>,
    ctx: &'a R::Context,
    specs: Vec<serde_json::Value>,
    calls: Vec<ExecutedCall<R::Extra>>,
    reads: HashMap<String, String>,
    actions: HashMap<String, String>,
    steps: usize,
    sink: Option<Arc<dyn TextStreamSink>>,
    task: Option<usize>,
    overflow_retries: usize,
}

fn call_key(name: &str, args: &serde_json::Value) -> String {
    format!("{name}::{args}")
}

impl<R: Runtime> Policy<'_, R> {
    /// What the model sees of a result: its rendering, with the output
    /// capped first so the envelope stays whole.
    fn render(&self, tool: Option<&dyn Tool<R>>, result: &ToolResult<R::Extra>) -> String {
        let limit = tool
            .and_then(Tool::output_limit)
            .or(self.agent.output_limit);
        match limit.map(|limit| cap_output(&result.output, limit)) {
            Some(std::borrow::Cow::Owned(output)) => {
                let mut capped = result.clone();
                capped.output = output;
                capped.for_model()
            }
            _ => result.for_model(),
        }
    }

    fn compact(&mut self, history: &mut Vec<ToolTurn>, compactor: &HistoryCompactor) -> bool {
        let report = compactor.compact(history, self.task);
        self.task = report.task_index;
        if report.changed() {
            tracing::info!(
                tokens_before = report.tokens_before,
                tokens_after = report.tokens_after,
                capped = report.capped_outputs,
                folded = report.folded_exchanges,
                digested = report.digested_turns,
                "compacted the turn history"
            );
        }
        report.changed()
    }

    async fn execute(&mut self, call: &ToolCall) -> (ToolCall, ToolResult<R::Extra>) {
        let Some(tool) = find(&self.agent.tools, &call.name) else {
            let valid = self
                .agent
                .tools
                .iter()
                .map(|t| t.name())
                .collect::<Vec<_>>()
                .join(", ");
            return (
                call.clone(),
                ToolResult::terminal(
                    ToolOutcomeStatus::UnknownTool,
                    format!("unknown tool: {}. Valid tools are: {valid}.", call.name),
                ),
            );
        };
        let args = match tool.prepare_arguments(call.arguments.clone()) {
            Ok(args) => args,
            Err(error) => {
                return (
                    call.clone(),
                    ToolResult::terminal(
                        ToolOutcomeStatus::InvalidArguments,
                        error.for_model(&call.name),
                    ),
                );
            }
        };
        let prepared = ToolCall {
            arguments: args.clone(),
            ..call.clone()
        };
        let key = call_key(&call.name, &args);
        let cache = if tool.is_action() {
            &self.actions
        } else {
            &self.reads
        };
        if let Some(previous) = cache.get(&key) {
            let result = ToolResult::terminal(
                ToolOutcomeStatus::Reused,
                format!(
                    "You already called {} with these exact arguments this turn. The earlier result was: {previous}. Do not repeat it; move on or reply to the user.",
                    call.name
                ),
            );
            return (prepared, result);
        }
        let result = match tool.call(self.ctx, args).await {
            Ok(result) => result.with_runtime_semantics(tool.is_action()),
            Err(error) => {
                tracing::warn!(tool = %call.name, "tool call failed: {error}");
                ToolResult::terminal(
                    ToolOutcomeStatus::Failed,
                    format!("{} failed: {error}", call.name),
                )
            }
        };
        let rendered = self.render(Some(tool), &result);
        if tool.is_action() {
            if result.changed() {
                self.reads.clear();
            }
            self.actions.insert(key, rendered);
        } else {
            self.reads.insert(key, rendered);
        }
        (prepared, result)
    }

    fn request(&self, tools: Vec<serde_json::Value>) -> StepRequest {
        let mut request = StepRequest::new(self.agent.system.clone(), tools);
        request.sink = self.sink.clone();
        request
    }

    fn record(
        &mut self,
        executed: Vec<(ToolCall, ToolResult<R::Extra>)>,
        history: &mut Vec<ToolTurn>,
    ) {
        history.push(ToolTurn::AssistantToolCalls(
            executed.iter().map(|(c, _)| c.clone()).collect(),
        ));
        let results = executed
            .iter()
            .map(|(c, r)| {
                let tool = find(&self.agent.tools, &c.name);
                (c.id.clone(), self.render(tool, r))
            })
            .collect();
        history.push(ToolTurn::ToolResults(results));
        self.calls.extend(
            executed
                .into_iter()
                .map(|(call, result)| ExecutedCall { call, result }),
        );
    }
}

#[async_trait]
impl<C: ?Sized + Sync, R: Runtime> TurnPolicy<C> for Policy<'_, R> {
    type Output = AgentRun<R::Extra>;
    type Error = AgentError<R::Error>;

    async fn prepare_step(
        &mut self,
        step: StepInfo,
        history: &mut Vec<ToolTurn>,
    ) -> Result<StepRequest, Self::Error> {
        self.steps = step.number();
        if let Some(compactor) = &self.agent.compactor {
            self.compact(history, &compactor.clone());
        }
        if step.is_last() && !self.calls.is_empty() {
            history.push(ToolTurn::User(
                "Give the user one final answer now. Use the tool results above and do not claim anything they did not establish."
                    .to_string(),
            ));
            return Ok(self.request(Vec::new()));
        }
        Ok(self.request(self.specs.clone()))
    }

    async fn on_text(
        &mut self,
        _step: StepInfo,
        text: String,
        _history: &mut Vec<ToolTurn>,
    ) -> Result<Flow<Self::Output>, Self::Error> {
        Ok(Flow::Finish(AgentRun {
            reply: text,
            calls: std::mem::take(&mut self.calls),
            steps: self.steps,
        }))
    }

    async fn on_tool_calls(
        &mut self,
        _step: StepInfo,
        calls: Vec<ToolCall>,
        history: &mut Vec<ToolTurn>,
    ) -> Result<Flow<Self::Output>, Self::Error> {
        // Text streamed before tool calls was provisional, not the answer.
        if let Some(sink) = &self.sink {
            sink.reset();
        }
        let agent = self.agent;
        let tools = &agent.tools;
        let shared = calls.len() > 1
            && calls.iter().all(|call| {
                find(tools, &call.name).is_some_and(|t| t.concurrency() == ToolConcurrency::Shared)
            });
        let prepared = if shared {
            calls
                .iter()
                .map(|call| {
                    let tool = find(tools, &call.name)?;
                    let args = tool.prepare_arguments(call.arguments.clone()).ok()?;
                    let key = call_key(&call.name, &args);
                    (!self.reads.contains_key(&key)).then_some((tool, args))
                })
                .collect::<Option<Vec<_>>>()
        } else {
            None
        };
        // Distinct calls only: duplicates go through the sequential path so
        // the second copy is reported as reused.
        let distinct = prepared.as_ref().is_some_and(|p| {
            let mut keys = std::collections::HashSet::new();
            p.iter()
                .zip(&calls)
                .all(|((_, args), call)| keys.insert(call_key(&call.name, args)))
        });
        let executed = if let (Some(prepared), true) = (prepared, distinct) {
            let batch = prepared
                .iter()
                .map(|(tool, args)| SharedCall {
                    tool: *tool,
                    ctx: self.ctx,
                    arguments: args.clone(),
                })
                .collect();
            let results = call_shared(batch).await;
            let mut executed = Vec::new();
            for ((call, (tool, args)), result) in calls.iter().zip(&prepared).zip(results) {
                let result = match result {
                    Ok(result) => result.with_runtime_semantics(tool.is_action()),
                    Err(error) => ToolResult::terminal(
                        ToolOutcomeStatus::Failed,
                        format!("{} failed: {error}", call.name),
                    ),
                };
                let rendered = self.render(Some(*tool), &result);
                self.reads.insert(call_key(&call.name, args), rendered);
                executed.push((
                    ToolCall {
                        arguments: args.clone(),
                        ..call.clone()
                    },
                    result,
                ));
            }
            executed
        } else {
            let mut executed = Vec::new();
            for call in &calls {
                executed.push(self.execute(call).await);
            }
            executed
        };
        self.record(executed, history);
        Ok(Flow::Continue)
    }

    async fn on_model_error(
        &mut self,
        _step: StepInfo,
        error: dcpr_llm::Error,
        history: &mut Vec<ToolTurn>,
    ) -> Result<Flow<Self::Output>, Self::Error> {
        // A context overflow: compact to half the current size and try once
        // more, when there is a budget to compact with.
        if error.is_context_overflow()
            && self.overflow_retries == 0
            && let Some(compactor) = &self.agent.compactor
        {
            let mut tighter = compactor.clone();
            tighter.max_tokens = tighter.max_tokens.min(estimate_tokens(history) / 2);
            self.overflow_retries += 1;
            if self.compact(history, &tighter) {
                return Ok(Flow::Continue);
            }
        }
        Err(error.into())
    }

    fn observe(&mut self, event: LoopEvent<'_>) {
        if let LoopEvent::ProviderCompleted {
            step,
            elapsed,
            error: Some(error),
            ..
        } = event
        {
            tracing::warn!(
                step = step.number(),
                elapsed_ms = elapsed.as_millis() as u64,
                "model step failed: {error}"
            );
        }
    }
}
