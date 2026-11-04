//! A one-tool agent, run against a scripted model so it needs no network.
//!
//! `cargo run -p dcpr-agent --example clock`

use async_trait::async_trait;
use dcpr_agent::dcpr_llm::{ScriptedModel, ToolTurn};
use dcpr_agent::{Runtime, Tool, ToolAgent, ToolResult};
use serde_json::{Value, json};

struct App;

impl Runtime for App {
    type Context = ();
    type Error = String;
    type Extra = ();
}

struct Clock;

#[async_trait]
impl Tool<App> for Clock {
    fn name(&self) -> &str {
        "current_time"
    }
    fn description(&self) -> &str {
        "The current UTC time."
    }
    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    async fn call(&self, _: &(), _: Value) -> Result<ToolResult, String> {
        Ok(ToolResult::observed("12:00 UTC"))
    }
}

#[tokio::main]
async fn main() {
    // Swap in `dcpr_llm::OpenAiCompat` for a real model.
    let mut model = ScriptedModel::new();
    model
        .push_tool_call("current_time", json!({}))
        .push_text("It is noon UTC.");

    let agent = ToolAgent::new("You are a helpful assistant.", vec![Box::new(Clock)]);
    let history = vec![ToolTurn::User("What time is it?".into())];
    let run = agent
        .run(&model, "user-42", &(), history)
        .await
        .expect("turn succeeds");
    println!(
        "{} ({} steps, {} tool calls)",
        run.reply,
        run.steps,
        run.calls.len()
    );
}
