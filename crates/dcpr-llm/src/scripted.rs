//! A deterministic model that replays scripted steps, for tests.

use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::Mutex;

use crate::{ChatModel, ChatStep, Error, Result, SystemPrompt, ToolCall, ToolTurn};

/// Replays a fixed script of [`ChatStep`]s, one per `chat_step` call.
///
/// Build it with the `push_*` methods. It panics if the conversation asks for
/// more steps than were scripted, which means the test is wrong or the loop
/// does not terminate. [`remaining`](Self::remaining) lets a test assert that
/// the whole script was used.
pub struct ScriptedModel {
    steps: Mutex<VecDeque<Result<ChatStep>>>,
    tool_names_seen: Mutex<Vec<Vec<String>>>,
    systems_seen: Mutex<Vec<String>>,
    cacheable_prefix_lengths_seen: Mutex<Vec<usize>>,
    histories_seen: Mutex<Vec<Vec<ToolTurn>>>,
}

impl ScriptedModel {
    pub fn new() -> Self {
        Self {
            steps: Mutex::new(VecDeque::new()),
            tool_names_seen: Mutex::new(Vec::new()),
            systems_seen: Mutex::new(Vec::new()),
            cacheable_prefix_lengths_seen: Mutex::new(Vec::new()),
            histories_seen: Mutex::new(Vec::new()),
        }
    }

    /// Push a plain-text assistant reply.
    pub fn push_text(&mut self, text: impl Into<String>) -> &mut Self {
        self.steps
            .get_mut()
            .unwrap()
            .push_back(Ok(ChatStep::Text(text.into())));
        self
    }

    /// Push a single tool call. Ids are assigned in push order (`call_1`,
    /// `call_2`, ...).
    pub fn push_tool_call(
        &mut self,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> &mut Self {
        self.push_tool_calls(vec![(name.into(), arguments)])
    }

    /// Push one assistant turn that requests several tool calls at once.
    pub fn push_tool_calls(&mut self, calls: Vec<(String, serde_json::Value)>) -> &mut Self {
        let q = self.steps.get_mut().unwrap();
        let base = q.len();
        let calls = calls
            .into_iter()
            .enumerate()
            .map(|(i, (name, arguments))| ToolCall {
                id: format!("call_{}", base + i + 1),
                name,
                arguments,
            })
            .collect();
        q.push_back(Ok(ChatStep::ToolCalls(calls)));
        self
    }

    /// Push a failing step, for example a simulated 529 or timeout.
    pub fn push_error(&mut self, error: impl Into<Error>) -> &mut Self {
        self.steps.get_mut().unwrap().push_back(Err(error.into()));
        self
    }

    /// Scripted steps not yet consumed.
    pub fn remaining(&self) -> usize {
        self.steps.lock().unwrap().len()
    }

    /// Tool names offered on each step.
    pub fn tool_names_seen(&self) -> Vec<Vec<String>> {
        self.tool_names_seen.lock().unwrap().clone()
    }

    /// System prompts sent on each step.
    pub fn systems_seen(&self) -> Vec<String> {
        self.systems_seen.lock().unwrap().clone()
    }

    pub fn cacheable_prefix_lengths_seen(&self) -> Vec<usize> {
        self.cacheable_prefix_lengths_seen.lock().unwrap().clone()
    }

    /// Conversation history sent on each step.
    pub fn histories_seen(&self) -> Vec<Vec<ToolTurn>> {
        self.histories_seen.lock().unwrap().clone()
    }
}

impl Default for ScriptedModel {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<C: ?Sized + Sync> ChatModel<C> for ScriptedModel {
    fn model_id(&self) -> &str {
        "scripted"
    }

    async fn chat_step(
        &self,
        _caller: &C,
        system: SystemPrompt<'_>,
        history: &[ToolTurn],
        tools: &[serde_json::Value],
    ) -> Result<ChatStep> {
        self.systems_seen
            .lock()
            .unwrap()
            .push(system.as_str().to_string());
        self.cacheable_prefix_lengths_seen.lock().unwrap().push(
            system
                .cacheable_prefix()
                .map_or(0, |(prefix, _)| prefix.len()),
        );
        self.histories_seen.lock().unwrap().push(history.to_vec());
        self.tool_names_seen.lock().unwrap().push(
            tools
                .iter()
                .filter_map(|spec| spec.get("name").and_then(serde_json::Value::as_str))
                .map(str::to_owned)
                .collect(),
        );
        self.steps.lock().unwrap().pop_front().unwrap_or_else(|| {
            panic!(
                "ScriptedModel ran out of scripted steps: the conversation asked for more \
                 chat_step calls than were pushed. Script another step, or the loop is \
                 not terminating."
            )
        })
    }
}
