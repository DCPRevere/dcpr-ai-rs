//! The tool-calling loop, with every decision delegated to a [`TurnPolicy`].
//!
//! [`run_turn`] owns the mechanics that every agent shares: the step budget, a
//! wall-clock deadline that also bounds each provider call, streaming, and the
//! dispatch of each model step to the policy. The policy owns everything that
//! makes an agent particular: the system prompt and tools for each step, how
//! tool calls execute, when a text reply is final, and what to return when the
//! budget or deadline runs out.

use async_trait::async_trait;
use dcpr_llm::{ChatModel, ChatStep, SystemPrompt, TextStreamSink, ToolCall, ToolTurn};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Where the loop is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepInfo {
    /// Zero-based step index.
    pub index: usize,
    pub max_steps: usize,
}

impl StepInfo {
    /// One-based step number, for logs.
    pub fn number(self) -> usize {
        self.index + 1
    }

    /// Whether this is the last step the budget allows.
    pub fn is_last(self) -> bool {
        self.index + 1 == self.max_steps
    }
}

/// What to send the model on one step.
pub struct StepRequest {
    pub system: String,
    /// Length of the fixed prefix of `system`, for provider prompt caching.
    /// Zero for none.
    pub cacheable_prefix_len: usize,
    /// Provider-neutral tool specs offered this step. Empty asks for text.
    pub tools: Vec<serde_json::Value>,
    /// Stream visible text here when set.
    pub sink: Option<Arc<dyn TextStreamSink>>,
}

impl StepRequest {
    pub fn new(system: impl Into<String>, tools: Vec<serde_json::Value>) -> Self {
        Self {
            system: system.into(),
            cacheable_prefix_len: 0,
            tools,
            sink: None,
        }
    }
}

/// Whether the loop continues to another model step.
#[derive(Debug)]
pub enum Flow<O> {
    Continue,
    Finish(O),
}

/// Progress events, for timing and observability.
#[derive(Debug)]
pub enum LoopEvent<'a> {
    ProviderStarted {
        step: StepInfo,
    },
    ProviderCompleted {
        step: StepInfo,
        elapsed: Duration,
        tools_offered: usize,
        /// The model's error, if the call failed.
        error: Option<&'a dcpr_llm::Error>,
        /// The call was abandoned at the turn deadline.
        timed_out: bool,
    },
}

/// The decisions an agent makes around each model step.
#[async_trait]
pub trait TurnPolicy<C: ?Sized + Sync>: Send {
    type Output: Send;
    type Error: From<dcpr_llm::Error> + Send;

    /// Build the request for this step. History may be amended first, for
    /// example to add a final-answer instruction on the last step.
    async fn prepare_step(
        &mut self,
        step: StepInfo,
        history: &mut Vec<ToolTurn>,
    ) -> Result<StepRequest, Self::Error>;

    /// The model returned text.
    async fn on_text(
        &mut self,
        step: StepInfo,
        text: String,
        history: &mut Vec<ToolTurn>,
    ) -> Result<Flow<Self::Output>, Self::Error>;

    /// The model asked for tool calls. Execute them and append the calls and
    /// their results to `history`, or finish.
    async fn on_tool_calls(
        &mut self,
        step: StepInfo,
        calls: Vec<ToolCall>,
        history: &mut Vec<ToolTurn>,
    ) -> Result<Flow<Self::Output>, Self::Error>;

    /// The model call failed. The default propagates the error. Override to
    /// shrink history after a context overflow, or to report work already done.
    async fn on_model_error(
        &mut self,
        step: StepInfo,
        error: dcpr_llm::Error,
        history: &mut Vec<ToolTurn>,
    ) -> Result<Flow<Self::Output>, Self::Error> {
        let _ = (step, history);
        Err(error.into())
    }

    /// The deadline passed, before a step or during its model call.
    async fn on_deadline(
        &mut self,
        step: StepInfo,
        history: &mut Vec<ToolTurn>,
    ) -> Result<Self::Output, Self::Error> {
        let _ = (step, history);
        Err(dcpr_llm::Error::transient("tool loop timed out").into())
    }

    /// Every step was used without a final answer.
    async fn on_step_limit(
        &mut self,
        history: &mut Vec<ToolTurn>,
    ) -> Result<Self::Output, Self::Error> {
        let _ = history;
        Err(dcpr_llm::Error::hard("tool loop did not converge").into())
    }

    fn observe(&mut self, event: LoopEvent<'_>) {
        let _ = event;
    }
}

/// Bounds on one turn.
#[derive(Debug, Clone, Copy)]
pub struct TurnLimits {
    pub max_steps: usize,
    /// When set, no step starts after this instant, and a model call running
    /// at this instant is abandoned.
    pub deadline: Option<Instant>,
}

impl TurnLimits {
    pub fn steps(max_steps: usize) -> Self {
        Self {
            max_steps,
            deadline: None,
        }
    }

    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    pub fn with_timeout(self, timeout: Duration) -> Self {
        self.with_deadline(Instant::now() + timeout)
    }
}

/// Drive one turn: call the model, hand each step to `policy`, and repeat until
/// the policy finishes or a limit is reached.
pub async fn run_turn<C, M, P>(
    model: &M,
    caller: &C,
    policy: &mut P,
    history: &mut Vec<ToolTurn>,
    limits: TurnLimits,
) -> Result<P::Output, P::Error>
where
    C: ?Sized + Sync,
    M: ChatModel<C> + ?Sized,
    P: TurnPolicy<C>,
{
    for index in 0..limits.max_steps {
        let step = StepInfo {
            index,
            max_steps: limits.max_steps,
        };
        if limits
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return policy.on_deadline(step, history).await;
        }
        let request = policy.prepare_step(step, history).await?;
        let system =
            SystemPrompt::with_cacheable_prefix(&request.system, request.cacheable_prefix_len);
        policy.observe(LoopEvent::ProviderStarted { step });
        let started = Instant::now();
        let call = async {
            match request.sink.clone() {
                Some(sink) => {
                    model
                        .chat_step_stream(caller, system, history, &request.tools, sink)
                        .await
                }
                None => {
                    model
                        .chat_step(caller, system, history, &request.tools)
                        .await
                }
            }
        };
        let result = match limits.deadline {
            Some(deadline) => match tokio::time::timeout_at(deadline.into(), call).await {
                Ok(result) => result,
                Err(_) => {
                    policy.observe(LoopEvent::ProviderCompleted {
                        step,
                        elapsed: started.elapsed(),
                        tools_offered: request.tools.len(),
                        error: None,
                        timed_out: true,
                    });
                    return policy.on_deadline(step, history).await;
                }
            },
            None => call.await,
        };
        policy.observe(LoopEvent::ProviderCompleted {
            step,
            elapsed: started.elapsed(),
            tools_offered: request.tools.len(),
            error: result.as_ref().err(),
            timed_out: false,
        });
        let flow = match result {
            Err(error) => policy.on_model_error(step, error, history).await?,
            Ok(ChatStep::Text(text)) => policy.on_text(step, text, history).await?,
            Ok(ChatStep::ToolCalls(calls)) => policy.on_tool_calls(step, calls, history).await?,
        };
        if let Flow::Finish(output) = flow {
            return Ok(output);
        }
    }
    policy.on_step_limit(history).await
}
