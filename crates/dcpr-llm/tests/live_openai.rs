//! Live checks against OpenAI's own API. Opt-in: set `DCPR_LIVE=1` and
//! `OPENAI_API_KEY`. `DCPR_LIVE_OPENAI_MODEL` overrides the model, which
//! defaults to `gpt-6.1-sol`. Without them each test skips with a message.
//!
//! `DCPR_LIVE=1 OPENAI_API_KEY=… cargo test -p dcpr-llm --test live_openai -- --nocapture`

use dcpr_llm::{ChatModel, ChatStep, Completion, NoUsage, OpenAiCompat, ToolTurn};
use serde_json::json;
use std::sync::Arc;

fn client() -> Option<OpenAiCompat> {
    let key = std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|key| !key.is_empty());
    let (true, Some(key)) = (std::env::var("DCPR_LIVE").as_deref() == Ok("1"), key) else {
        eprintln!("skipped: set DCPR_LIVE=1 and OPENAI_API_KEY to run live OpenAI tests");
        return None;
    };
    let model = std::env::var("DCPR_LIVE_OPENAI_MODEL").unwrap_or_else(|_| "gpt-6.1-sol".into());
    Some(
        OpenAiCompat::new("https://api.openai.com/v1", key, model, Arc::new(NoUsage))
            .with_responses_api(true),
    )
}

#[tokio::test]
async fn a_small_completion_returns_text() {
    let Some(model) = client() else { return };
    let reply = model
        .complete(
            "live-test",
            Completion::new("Reply with one word.", "What is the capital of France?", 64),
        )
        .await
        .expect("completion succeeds");
    assert!(!reply.is_empty());
    assert!(reply.to_lowercase().contains("paris"), "{reply}");
}

#[tokio::test]
async fn a_tool_call_its_result_and_a_text_answer() {
    let Some(model) = client() else { return };
    let tools = [json!({
        "name": "get_weather",
        "description": "Current weather for one city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    })];
    let system = "Use the tool for weather. Be brief.".into();
    let mut history = vec![ToolTurn::User("What's the weather in Paris?".into())];

    let step = model
        .chat_step("live-test", system, &history, &tools)
        .await
        .expect("first step succeeds");
    let ChatStep::ToolCalls(calls) = step else {
        panic!("expected a tool call, got {step:?}");
    };
    assert_eq!(calls[0].name, "get_weather");
    let results = calls
        .iter()
        .map(|call| (call.id.clone(), "Paris: 18C, sunny".to_string()))
        .collect();
    history.push(ToolTurn::AssistantToolCalls(calls));
    history.push(ToolTurn::ToolResults(results));

    let step = model
        .chat_step("live-test", system, &history, &tools)
        .await
        .expect("second step succeeds");
    let ChatStep::Text(answer) = step else {
        panic!("expected a text answer, got {step:?}");
    };
    assert!(answer.contains("18"), "{answer}");
}
