//! Request shapes for hosts other than OpenRouter: OpenAI's own API and
//! Gemini's OpenAI-compatible endpoint. The mock server is mounted under a
//! path naming the host, so host detection sees the real service.

use super::*;
use crate::{Completion, NoUsage, ToolTurn};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn openai(server: &MockServer, model: &str) -> OpenAiCompat {
    OpenAiCompat::new(
        format!("{}/api.openai.com/v1", server.uri()),
        "k",
        model,
        Arc::new(NoUsage),
    )
}

fn bodies(requests: &[Request]) -> Vec<Value> {
    requests
        .iter()
        .map(|request| serde_json::from_slice(&request.body).unwrap())
        .collect()
}

fn assert_no_openrouter_fields(body: &Value) {
    for field in ["provider", "models", "session_id", "user", "max_tokens"] {
        assert!(body.get(field).is_none(), "{field} sent to OpenAI: {body}");
    }
    // Hashed per-caller fields, never the caller's own identifier.
    for field in ["safety_identifier", "prompt_cache_key"] {
        let value = body[field].as_str().unwrap_or_default();
        assert_eq!(value.len(), 64, "{field}: {body}");
        assert!(!value.contains("caller"));
    }
    let tools = body["tools"].as_array().into_iter().flatten();
    for tool in tools {
        let kind = tool["type"].as_str().unwrap_or_default();
        assert!(!kind.starts_with("openrouter:"), "{tool}");
        assert!(tool.get("defer_loading").is_none(), "{tool}");
    }
}

fn chat_reply(content: &str) -> Value {
    json!({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]})
}

#[tokio::test]
async fn openai_completions_use_its_token_field_and_reasoning_rules() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api.openai.com/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_reply("Paris")))
        .mount(&server)
        .await;
    for model in ["gpt-6.1-sol", "gpt-4.1-mini"] {
        let reply = openai(&server, model)
            .with_fallback_models(vec!["gpt-5.5".into()])
            .with_reasoning_effort(Some("high".into()))
            .complete(
                "caller",
                Completion::new("One word.", "Capital of France?", 64),
            )
            .await
            .unwrap();
        assert_eq!(reply, "Paris");
    }
    let bodies = bodies(&server.received_requests().await.unwrap());
    let (reasoning, plain) = (&bodies[0], &bodies[1]);
    for body in &bodies {
        assert_no_openrouter_fields(body);
        assert!(body.get("reasoning").is_none(), "{body}");
    }
    // A reasoning model: low effort, headroom for it, and no temperature.
    assert_eq!(reasoning["reasoning_effort"], "low");
    assert_eq!(reasoning["max_completion_tokens"], 64 + 4_096);
    assert!(reasoning.get("temperature").is_none());
    // gpt-4.1 keeps temperature and rejects any reasoning field.
    assert_eq!(plain["max_completion_tokens"], 64);
    assert_eq!(plain["temperature"], 0);
    assert!(plain.get("reasoning_effort").is_none());
}

#[tokio::test]
async fn an_empty_reasoning_completion_is_a_max_tokens_stop() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": ""}}]
        })))
        .mount(&server)
        .await;
    let error = openai(&server, "gpt-6.1-sol")
        .complete("caller", Completion::new("s", "u", 16))
        .await
        .unwrap_err();
    assert_eq!(error.stop_reason(), Some(StopReason::MaxTokens), "{error}");
}

#[tokio::test]
async fn a_cut_off_completion_is_a_max_tokens_stop_even_with_text() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "The fox had"}}]
        })))
        .mount(&server)
        .await;
    let error = openai(&server, "gpt-4.1-mini")
        .complete("caller", Completion::new("s", "u", 4))
        .await
        .unwrap_err();
    assert_eq!(error.stop_reason(), Some(StopReason::MaxTokens), "{error}");
    assert!(error.is_retryable());
}

#[tokio::test]
async fn openai_tool_steps_default_to_the_responses_api() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api.openai.com/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "completed",
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hi."}]}],
        })))
        .expect(1)
        .mount(&server)
        .await;
    let step = openai(&server, "gpt-6.1-sol")
        .chat_step("caller", "s".into(), &[ToolTurn::User("Hi".into())], &[])
        .await
        .unwrap();
    assert!(matches!(step, ChatStep::Text(text) if text == "Hi."));
    // Other hosts keep Chat Completions unless asked.
    let other: OpenAiCompat =
        OpenAiCompat::new("https://example.test/v1", "k", "m", Arc::new(NoUsage));
    assert!(!other.uses_responses());
    assert!(other.with_responses_api(true).uses_responses());
}

#[tokio::test]
async fn openai_chat_tool_steps_send_the_configured_effort() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_reply("Done.")))
        .mount(&server)
        .await;
    let tools = [json!({"name": "list_beds", "description": "List beds."})];
    openai(&server, "gpt-5-mini")
        .with_responses_api(false)
        .with_reasoning_effort(Some("low".into()))
        .chat_step(
            "caller",
            "Tend the garden.".into(),
            &[ToolTurn::User("Which beds?".into())],
            &tools,
        )
        .await
        .unwrap();
    let body = &bodies(&server.received_requests().await.unwrap())[0];
    assert_no_openrouter_fields(body);
    assert_eq!(body["reasoning_effort"], "low");
    assert_eq!(body["max_completion_tokens"], 4_096 + 24_000);
    assert!(body.get("temperature").is_none());
}

#[tokio::test]
async fn a_three_step_openai_responses_loop_replays_openai_item_ids_and_reasoning() {
    let server = MockServer::start().await;
    let call = json!({
        "status": "completed",
        "output": [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "sealed", "summary": []},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1",
             "name": "list_beds", "arguments": "{}", "status": "completed"},
        ],
    });
    let answer = json!({
        "status": "completed",
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "Bed 4 is free."}]}],
    });
    Mock::given(method("POST"))
        .and(path("/api.openai.com/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(call))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api.openai.com/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(answer))
        .mount(&server)
        .await;
    let client = openai(&server, "gpt-6.1-sol")
        .with_responses_api(true)
        .with_reasoning_effort(Some("medium".into()));
    let tools = [
        json!({"name": "list_beds", "description": "List beds."}),
        json!({"name": "rare_tool", "description": "Rarely used.", "defer_loading": true}),
    ];
    let system = SystemPrompt::from("Tend the garden.");
    let mut history = vec![ToolTurn::User("Which bed is free?".into())];
    let ChatStep::ToolCalls(calls) = client
        .chat_step("caller", system, &history, &tools)
        .await
        .unwrap()
    else {
        panic!("expected a tool call");
    };
    assert_eq!(calls[0].id, "call_1");
    history.push(ToolTurn::AssistantToolCalls(calls));
    history.push(ToolTurn::ToolResults(vec![(
        "call_1".into(),
        "bed 4".into(),
    )]));
    let ChatStep::Text(text) = client
        .chat_step("caller", system, &history, &tools)
        .await
        .unwrap()
    else {
        panic!("expected text");
    };
    assert_eq!(text, "Bed 4 is free.");

    let bodies = bodies(&server.received_requests().await.unwrap());
    for body in &bodies {
        assert_no_openrouter_fields(body);
        assert_eq!(body["reasoning"], json!({"effort": "medium"}));
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(body["max_output_tokens"], 4_096 + 24_000);
        assert_eq!(body["store"], false);
        assert_eq!(body["tools"].as_array().unwrap().len(), 2);
    }
    let input = bodies[1]["input"].as_array().unwrap();
    assert_eq!(input[1]["type"], "reasoning");
    assert_eq!(input[1]["encrypted_content"], "sealed");
    assert_eq!(input[2]["type"], "function_call");
    assert_eq!(input[2]["id"], "fc_1");
    assert_eq!(input[2]["call_id"], "call_1");
    assert_eq!(input[3]["type"], "function_call_output");

    // History replayed by a fresh client has no remembered item id, so the
    // call id is never sent in its place.
    let fresh = openai(&server, "gpt-6.1-sol").with_responses_api(true);
    fresh
        .chat_step("caller", system, &history, &tools)
        .await
        .unwrap();
    let replay = &bodies_after(&server, 2).await[0];
    let call = replay["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "function_call")
        .unwrap();
    assert!(call.get("id").is_none(), "{call}");
    assert_eq!(call["call_id"], "call_1");
    // No effort configured: the model's default applies, with headroom.
    assert!(replay.get("reasoning").is_none());
    assert_eq!(replay["max_output_tokens"], 4_096 + 24_000);
}

async fn bodies_after(server: &MockServer, skip: usize) -> Vec<Value> {
    bodies(&server.received_requests().await.unwrap()[skip..])
}

#[tokio::test]
async fn openai_quota_exhaustion_is_not_retried() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({"error": {
            "message": "You have no credits remaining.",
            "type": "insufficient_quota",
            "code": "credit_balance_exhausted",
        }})))
        .mount(&server)
        .await;
    let error = openai(&server, "gpt-4.1-mini")
        .complete("caller", Completion::new("s", "u", 8))
        .await
        .unwrap_err();
    assert!(!error.is_retryable(), "{error}");
    assert!(error.to_string().contains("insufficient_quota"), "{error}");
    // An ordinary rate limit stays retryable.
    assert!(Error::from_status(429, "Rate limit reached [rate_limit_exceeded]").is_retryable());
}

#[tokio::test]
async fn gemini_tool_calls_carry_their_thought_signature_back() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": null,
                "tool_calls": [{
                    "id": "g1", "type": "function",
                    "function": {"name": "list_beds", "arguments": "{}"},
                    "extra_content": {"google": {"thought_signature": "sig-1"}},
                }],
            }}],
        })))
        .mount(&server)
        .await;
    let gemini = |model: &str| {
        OpenAiCompat::<str>::new(
            format!(
                "{}/generativelanguage.googleapis.com/v1beta/openai",
                server.uri()
            ),
            "k",
            model,
            Arc::new(NoUsage),
        )
    };
    let client = gemini("gemini-3.5-flash-lite");
    let tools = [json!({"name": "list_beds", "description": "List beds."})];
    let mut history = vec![ToolTurn::User("Which beds?".into())];
    let ChatStep::ToolCalls(calls) = client
        .chat_step("caller", "s".into(), &history, &tools)
        .await
        .unwrap()
    else {
        panic!("expected a tool call");
    };
    history.push(ToolTurn::AssistantToolCalls(calls));
    history.push(ToolTurn::ToolResults(vec![("g1".into(), "bed 4".into())]));
    client
        .chat_step("caller", "s".into(), &history, &tools)
        .await
        .unwrap();
    // A fresh client no longer knows the signature and sends Gemini's
    // documented placeholder.
    gemini("gemini-3.5-flash-lite")
        .chat_step("caller", "s".into(), &history, &[])
        .await
        .unwrap();

    let bodies = bodies(&server.received_requests().await.unwrap());
    let replayed = |body: &Value| body["messages"][2]["tool_calls"][0]["extra_content"].clone();
    assert_eq!(
        replayed(&bodies[1]),
        json!({"google": {"thought_signature": "sig-1"}})
    );
    assert_eq!(
        replayed(&bodies[2]),
        json!({"google": {"thought_signature": "skip_thought_signature_validator"}})
    );
    // An empty tool list is left out rather than sent as `[]`.
    assert!(bodies[2].get("tools").is_none());
    assert_eq!(bodies[1]["max_tokens"], 4_096);
}

#[tokio::test]
async fn openrouter_learns_a_model_that_cannot_disable_reasoning() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({"error": {
            "message": "Reasoning is mandatory for this endpoint and cannot be disabled.",
            "code": 400,
        }})))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_reply("Paris")))
        .mount(&server)
        .await;
    let client: OpenAiCompat = OpenAiCompat::new(
        format!("{}/openrouter", server.uri()),
        "k",
        "vendor/new-reasoner",
        Arc::new(NoUsage),
    );
    for _ in 0..2 {
        let reply = client
            .complete("caller", Completion::new("s", "u", 16))
            .await
            .unwrap();
        assert_eq!(reply, "Paris");
    }
    let bodies = bodies(&server.received_requests().await.unwrap());
    assert_eq!(
        bodies.len(),
        3,
        "one failure, one retry, then straight through"
    );
    assert_eq!(bodies[0]["reasoning"], json!({"enabled": false}));
    assert_eq!(bodies[1]["reasoning"], json!({"effort": "low"}));
    assert_eq!(bodies[2]["reasoning"], json!({"effort": "low"}));
}

#[test]
fn openai_models_on_openrouter_that_need_reasoning_get_low_effort() {
    let client: OpenAiCompat = OpenAiCompat::new(
        "https://openrouter.ai/api/v1",
        "k",
        "openai/gpt-6.1-sol",
        Arc::new(NoUsage),
    );
    let schema = json!({"type": "object"});
    let body = client.completion_body(
        "caller",
        &Completion::new("s", "u", 16).with_schema("reply", &schema),
    );
    assert_eq!(body["reasoning"], json!({"effort": "low"}));
    // `require_parameters` would find no endpoint that takes temperature.
    assert!(body.get("temperature").is_none());
    assert_eq!(body["provider"]["require_parameters"], true);
    // A plain helper on a model that can disable reasoning is unchanged.
    let luna: OpenAiCompat = OpenAiCompat::new(
        "https://openrouter.ai/api/v1",
        "k",
        "openai/gpt-6-luna",
        Arc::new(NoUsage),
    );
    let body = luna.completion_body("caller", &Completion::new("s", "u", 16));
    assert_eq!(body["reasoning"], json!({"enabled": false}));
    // OpenAI reasoning models take no temperature on any host.
    assert!(body.get("temperature").is_none());
}

#[test]
fn claude_models_that_reject_temperature_never_get_it_off_openrouter() {
    for (model, has_temperature) in [("claude-sonnet-5-5", false), ("claude-opus-4-6", true)] {
        let client: OpenAiCompat = OpenAiCompat::new(
            "https://api.anthropic.com/v1",
            "k",
            model,
            Arc::new(NoUsage),
        );
        let body = client.completion_body("caller", &Completion::new("s", "u", 16));
        assert_eq!(
            body.get("temperature").is_some(),
            has_temperature,
            "{model}"
        );
    }
}

#[cfg(feature = "batch")]
#[tokio::test]
async fn responses_batch_steps_target_the_responses_endpoint_and_stay_apart() {
    let client: OpenAiCompat = OpenAiCompat::new(
        "http://127.0.0.1:9/api.openai.com/v1",
        "k",
        "gpt-6.1-sol",
        Arc::new(NoUsage),
    )
    .with_responses_api(true);
    let tools = [json!({"name": "list_beds", "description": "List beds."})];
    let step = client.batch_step(
        "caller",
        "step",
        "s".into(),
        &[ToolTurn::User("Which beds?".into())],
        &tools,
    );
    let line = step.line();
    assert_eq!(line["url"], "/v1/responses");
    assert_eq!(line["body"]["max_output_tokens"], 4_096 + 24_000);
    assert_eq!(line["body"]["tools"][0]["name"], "list_beds");
    let completion = client.batch_completion("caller", "helper", Completion::new("s", "u", 8));
    assert_eq!(completion.line()["url"], "/v1/chat/completions");
    let error = client.submit_batch(&[step, completion]).await.unwrap_err();
    assert!(error.to_string().contains("one endpoint"), "{error}");
}
