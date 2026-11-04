use super::*;
use crate::{Completion, InputImage, NoUsage, ToolTurn, Usage, UsageSink};

/// (caller, model, usage, reported cost in cents)
type UsageRecord = (String, String, Usage, Option<f64>);

#[derive(Default)]
struct RecordingUsage(Mutex<Vec<UsageRecord>>);

impl UsageSink<str> for RecordingUsage {
    fn record(&self, caller: &str, model: &str, usage: Usage, cost: Option<f64>) {
        self.0
            .lock()
            .unwrap()
            .push((caller.to_string(), model.to_string(), usage, cost));
    }
}
use std::sync::Mutex;

#[test]
fn request_sizes_separate_deferred_schemas_without_exposing_content() {
    let initial = json!({"name": "read", "description": "visible"});
    let deferred = json!({"name": "rare", "defer_loading": true});
    let input = json!([{"role": "user", "content": "private"}]);
    let body = json!({
        "instructions": "é", "input": input,
        "tools": [initial, deferred],
    });
    let mut split = body.clone();
    anthropic::RequestPolicy::for_models("anthropic/claude-haiku-5.5", &[])
        .apply_responses_instructions(&mut split, SystemPrompt::with_cacheable_prefix("é", 2));
    let sizes = [&body, &split, &json!({"instructions": "", "input": []})]
        .into_iter()
        .map(request_size)
        .map(|s| {
            (
                s.instructions_bytes,
                s.history_bytes,
                s.initial_tools_bytes,
                s.deferred_tools_bytes,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sizes,
        vec![
            (
                2,
                input.to_string().len(),
                initial.to_string().len(),
                deferred.to_string().len()
            ),
            (
                2,
                input.to_string().len(),
                initial.to_string().len(),
                deferred.to_string().len()
            ),
            (0, 2, 0, 0),
        ]
    );
}

#[test]
fn request_observer_receives_each_responses_request_size() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let provider = provider_at("https://openrouter.ai/api/v1")
        .with_request_observer(move |size| sink.lock().unwrap().push(size));
    provider.observe_request_size(&json!({"instructions": "abc", "input": []}));
    assert_eq!(seen.lock().unwrap()[0].instructions_bytes, 3);
}

#[derive(Default)]
struct RecordingTextSink(Mutex<Vec<String>>);

impl crate::TextStreamSink for RecordingTextSink {
    fn reset(&self) {
        self.0.lock().unwrap().clear();
    }

    fn delta(&self, text: &str) {
        self.0.lock().unwrap().push(text.to_string());
    }
}

#[test]
fn responses_stream_emits_only_visible_text_across_fragmented_frames() {
    let wire = concat!(
        "event: response.output_text.delta\r\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Café \"}\r\n\r\n",
        "event: response.reasoning_text.delta\n",
        "data: {\"type\":\"response.reasoning_text.delta\",\"delta\":\"private reasoning\"}\n\n",
        "event: response.function_call_arguments.delta\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"secret\\\":true}\"}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"answer\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"model\":\"test\",\"output\":[],\"usage\":{}}}\n\n",
        "data: [DONE]\n\n"
    );
    let sink = RecordingTextSink::default();
    let mut accumulator = ResponsesStreamAccumulator::default();
    for chunk in wire.as_bytes().chunks(7) {
        accumulator.push(chunk, &sink).unwrap();
    }
    accumulator.finish(&sink).unwrap();

    assert_eq!(sink.0.lock().unwrap().concat(), "Café answer");
    assert_eq!(accumulator.completed.unwrap()["status"], "completed");
}

#[test]
fn oai_function_puts_the_real_schema_under_parameters() {
    // The production outage: the model received tools with NO declared
    // parameters (the schema shipped under the wrong key) and guessed argument
    // names — deepseek confabulated `text` for water_bed's `body`. Drive
    // this from the ACTUAL neutral spec shape Tool::spec() produces, so a
    // regression in EITHER spec() or oai_function fails here. Assert the schema
    // lands under `function.parameters` with `body`/`when` required and that no
    // Anthropic `input_schema` key leaks onto the OpenAI wire.
    let spec = json!({
        "name": "water_bed",
        "description": "Schedule a watering.",
        "parameters": {
            "type": "object",
            "properties": { "body": {"type":"string"}, "when": {"type":"string"} },
            "required": ["body", "when"]
        }
    });
    let f = oai_function(&spec);
    assert_eq!(f["type"], "function");
    assert_eq!(f["function"]["name"], "water_bed");
    assert_eq!(
        f["function"]["parameters"]["required"],
        json!(["body", "when"])
    );
    assert!(
        f["function"]["parameters"]["properties"]
            .get("body")
            .is_some(),
        "the real `body` parameter must reach the model"
    );
    assert!(
        f["function"].get("input_schema").is_none(),
        "input_schema must not be sent"
    );
    assert!(f.get("input_schema").is_none());
}

#[test]
fn oai_function_supplies_an_empty_schema_when_a_tool_has_no_params() {
    // A param-less tool (e.g. list_seedlings) still needs a well-formed
    // `parameters` object, never a missing or null schema.
    let spec = json!({ "name": "list_seedlings", "description": "List seedlings." });
    let f = oai_function(&spec);
    assert_eq!(
        f["function"]["parameters"],
        json!({ "type": "object", "properties": {} })
    );
}

#[test]
fn responses_function_preserves_native_deferral() {
    let spec = json!({
        "name": "grant_docking",
        "description": "Grant a supply ship docking access.",
        "parameters": {"type": "object", "properties": {}},
        "defer_loading": true
    });
    let tool = responses_function(&spec);
    assert_eq!(tool["type"], "function");
    assert_eq!(tool["name"], "grant_docking");
    assert_eq!(tool["defer_loading"], true);
    assert!(
        tool.get("function").is_none(),
        "Responses tools use the flat shape"
    );
}

#[test]
fn responses_payload_parses_parallel_function_calls_and_text() {
    let calls = json!({
        "output": [
            {"type":"function_call","id":"fc1","call_id":"c1","name":"list_seedlings","arguments":"{}"},
            {"type":"function_call","id":"fc2","call_id":"c2","name":"water_bed","arguments":"{\"body\":\"bed 2\",\"when\":\"tomorrow\"}"}
        ]
    });
    match interpret_responses_payload(&calls) {
        crate::ChatStep::ToolCalls(calls) => {
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[1].arguments["body"], "bed 2");
        }
        other => panic!("expected tool calls, got {other:?}"),
    }

    let text = json!({"output":[{"type":"message","content":[
        {"type":"output_text","text":"Done.","annotations":[]}
    ]}]});
    assert!(matches!(
        interpret_responses_payload(&text),
        crate::ChatStep::Text(ref value) if value == "Done."
    ));
}

#[test]
fn responses_input_preserves_a_complete_parallel_tool_round_trip() {
    let turns = vec![
        ToolTurn::User("Plant moon basil and show my waterings".into()),
        ToolTurn::AssistantToolCalls(vec![
            crate::ToolCall {
                id: "call_seedling".into(),
                name: "plant_seedling".into(),
                arguments: json!({"title": "moon basil"}),
            },
            crate::ToolCall {
                id: "call_waterings".into(),
                name: "list_waterings".into(),
                arguments: json!({}),
            },
        ]),
        ToolTurn::ToolResults(vec![
            ("call_seedling".into(), "{\"status\":\"applied\"}".into()),
            ("call_waterings".into(), "{\"items\":[]}".into()),
        ]),
    ];
    let input = responses_input(&turns);
    assert_eq!(input.len(), 5);
    assert_eq!(input[1]["type"], "function_call");
    assert_eq!(input[1]["call_id"], "call_seedling");
    assert_eq!(input[2]["call_id"], "call_waterings");
    assert_eq!(input[3]["type"], "function_call_output");
    assert_eq!(input[3]["call_id"], "call_seedling");
    assert_eq!(input[4]["call_id"], "call_waterings");
}

#[test]
fn responses_input_preserves_a_delivered_reply_at_the_replay_boundary() {
    let input = responses_input(&[
        ToolTurn::Assistant("My earlier answer was wrong.".into()),
        ToolTurn::User("Please correct that answer.".into()),
    ]);
    assert_eq!(input.len(), 3);
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[1]["role"], "assistant");
    assert_eq!(input[1]["status"], "completed");
    assert_eq!(input[1]["content"][0]["type"], "output_text");
    assert_eq!(
        input[1]["content"][0]["text"],
        "My earlier answer was wrong."
    );
    assert_eq!(
        input[2]["content"][0]["text"],
        "Please correct that answer."
    );
}

#[tokio::test]
async fn responses_request_replays_complete_assistant_history() {
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openrouter/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "completed", "output": [{
                "type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Here is the correction.", "annotations": []}
                ]
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let provider = wiremock_provider(&server.uri()).with_responses_api(true);
    let history = vec![
        ToolTurn::User("Can I do this?".into()),
        ToolTurn::Assistant("My earlier answer was wrong.".into()),
        ToolTurn::User("Please correct that answer.".into()),
        ToolTurn::Assistant("I will check the source.".into()),
        ToolTurn::User("What did you conclude?".into()),
    ];
    provider
        .chat_step("caller", "test".into(), &history, &[])
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: Value = requests[0].body_json().unwrap();
    let assistant = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["role"] == "assistant")
        .collect::<Vec<_>>();
    assert_eq!(assistant.len(), 2);
    for item in &assistant {
        assert_eq!(item["status"], "completed");
        assert!(item["id"].as_str().is_some_and(|id| id.starts_with("msg_")));
        assert_eq!(item["content"][0]["type"], "output_text");
    }
    assert_ne!(assistant[0]["id"], assistant[1]["id"]);
    assert_eq!(
        assistant[0]["content"][0]["text"],
        "My earlier answer was wrong."
    );
    assert_eq!(
        assistant[1]["content"][0]["text"],
        "I will check the source."
    );
    assert_eq!(body["input"], json!(responses_input(&history)));
}

#[test]
fn responses_payload_ignores_reasoning_and_accepts_changed_output_order() {
    let payload = json!({
        "output": [
            {"type":"reasoning","id":"rs_1","summary":[]},
            {"type":"message","role":"assistant","content":[
                {"type":"output_text","text":"I will check both.","annotations":[]}
            ]},
            {"type":"function_call","id":"fc_2","call_id":"call_2","name":"list_seedlings","arguments":{}},
            {"type":"function_call","id":"fc_1","call_id":"call_1","name":"list_waterings","arguments":null}
        ]
    });
    match interpret_responses_payload(&payload) {
        crate::ChatStep::ToolCalls(calls) => {
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].name, "list_seedlings");
            assert_eq!(calls[0].arguments, json!({}));
            assert_eq!(calls[1].name, "list_waterings");
            assert_eq!(calls[1].arguments, json!({}));
        }
        other => panic!("expected tool calls, got {other:?}"),
    }
}

#[test]
fn responses_payload_keeps_malformed_arguments_for_tool_validation() {
    let payload = json!({"output":[{
        "type":"function_call",
        "call_id":"call_1",
        "name":"water_bed",
        "arguments":"{\"body\":"
    }]});
    match interpret_responses_payload(&payload) {
        crate::ChatStep::ToolCalls(calls) => {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].arguments, "{\"body\":");
        }
        other => panic!("expected retained malformed call, got {other:?}"),
    }
}

#[tokio::test]
async fn responses_transport_uses_server_side_tool_search() {
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openrouter/responses"))
        .and(body_partial_json(json!({
            "instructions": "sys",
            "tools": [
                {"type":"openrouter:tool_search"},
                {"type":"function","name":"grant_docking","defer_loading":true}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1",
            "model": "m",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "grant_docking",
                "arguments": "{}"
            }],
            "usage": {"input_tokens": 10, "output_tokens": 4}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = wiremock_provider(&server.uri()).with_responses_api(true);
    let specs = [json!({
        "name": "grant_docking",
        "description": "Grant docking access.",
        "parameters": {"type":"object","properties":{}},
        "defer_loading": true
    })];
    let step = provider
        .chat_step(
            "caller",
            "sys".into(),
            &[ToolTurn::User("Let the supply ship dock".into())],
            &specs,
        )
        .await
        .unwrap();
    assert!(matches!(
        step,
        crate::ChatStep::ToolCalls(ref calls)
            if calls.len() == 1 && calls[0].name == "grant_docking"
    ));
}

#[tokio::test]
async fn byok_upstream_charge_is_reported_with_usage() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openrouter/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1",
            "model": "anthropic/claude-haiku-5.5",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done."}]
            }],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "cost": 0.0005,
                "is_byok": true,
                "cost_details": {"upstream_inference_cost": 0.01}
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let usage = Arc::new(RecordingUsage::default());
    let provider = OpenAiCompat::new(
        format!("{}/openrouter", server.uri()),
        "k",
        "m",
        usage.clone(),
    )
    .with_responses_api(true);
    provider
        .chat_step("caller", "sys".into(), &[ToolTurn::User("Hi".into())], &[])
        .await
        .unwrap();

    let records = usage.0.lock().unwrap();
    assert_eq!(records.len(), 1);
    let (caller, model, tokens, cents) = &records[0];
    assert_eq!(
        (caller.as_str(), model.as_str()),
        ("caller", "anthropic/claude-haiku-5.5")
    );
    assert_eq!((tokens.input_tokens, tokens.output_tokens), (10, 4));
    let cents = cents.unwrap();
    assert!(
        (cents - 1.05).abs() < 1e-9,
        "fee plus upstream charge, got {cents}"
    );
}

#[tokio::test]
async fn anthropic_cache_prefix_survives_changing_context_on_both_transports() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const PREFIX: &str = "Fixed trusted instructions.\n";
    for (model, openrouter) in [
        ("anthropic/claude-haiku-5.5", true),
        ("~anthropic/claude-sonnet-latest", true),
        ("z-ai/glm-5.3-flash", true),
        ("anthropic/claude-haiku-5.5", false),
    ] {
        for responses_api in [true, false] {
            let server = MockServer::start().await;
            let endpoint = match (openrouter, responses_api) {
                (true, true) => "/openrouter/responses",
                (true, false) => "/openrouter/chat/completions",
                (false, true) => "/responses",
                (false, false) => "/chat/completions",
            };
            let response = if responses_api {
                json!({"status": "completed", "output": [{
                    "type": "message", "content": [{"type": "output_text", "text": "Here."}]
                }]})
            } else {
                json!({"choices": [{"finish_reason": "stop", "message": {"content": "Here."}}]})
            };
            Mock::given(method("POST"))
                .and(path(endpoint))
                .respond_with(ResponseTemplate::new(200).set_body_json(response))
                .expect(2)
                .mount(&server)
                .await;
            let mut provider = wiremock_provider(&server.uri()).with_responses_api(responses_api);
            provider.model = model.into();
            if !openrouter {
                provider.base_url = server.uri();
            }
            let caller = "caller";
            let history = [
                ToolTurn::Assistant("Earlier reply.".into()),
                ToolTurn::User("Continue.".into()),
            ];
            let tools = [
                json!({"name": "list_seedlings", "description": "List seedlings.",
                "parameters": {"type": "object", "properties": {}}, "defer_loading":true}),
            ];
            let prompts = [
                format!("{PREFIX}GREENHOUSE: Bay 7. CLOCK: 09:00. EXECUTION: none."),
                format!(
                    "{PREFIX}GREENHOUSE: Bay 9. CLOCK: 10:00. EXECUTION: plant_seedling applied."
                ),
            ];
            for text in &prompts {
                provider
                    .chat_step(
                        caller,
                        SystemPrompt::with_cacheable_prefix(text, PREFIX.len()),
                        &history,
                        &tools,
                    )
                    .await
                    .unwrap();
            }
            let requests = server.received_requests().await.unwrap();
            let bodies = requests
                .iter()
                .map(|r| serde_json::from_slice::<Value>(&r.body).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(bodies[0]["tools"], bodies[1]["tools"]);
            assert_eq!(bodies[0]["session_id"], bodies[1]["session_id"]);
            for (index, body) in bodies.iter().enumerate() {
                if !responses_api {
                    assert_eq!(body["tools"].as_array().unwrap().len(), 1);
                    assert_eq!(body["tools"][0]["type"], "function");
                    assert_eq!(body["tools"][0]["function"]["name"], "list_seedlings");
                    assert!(body["tools"][0].get("defer_loading").is_none());
                    assert!(body["tools"][0]["function"].get("defer_loading").is_none());
                }
                if openrouter && anthropic::is_model(model) {
                    assert_eq!(body["cache_control"], json!({"type": "ephemeral"}));
                    let content = if responses_api {
                        assert!(
                            body.get("instructions").is_none(),
                            "volatile root instructions would precede the fixed prefix"
                        );
                        assert_eq!(body["input"][0]["role"], "system");
                        // Leading assistant history still has its neutral user boundary.
                        assert_eq!(body["input"][1]["role"], "user");
                        assert_eq!(body["input"][2]["role"], "assistant");
                        &body["input"][0]["content"]
                    } else {
                        assert_eq!(body["messages"][0]["role"], "system");
                        &body["messages"][0]["content"]
                    };
                    assert_eq!(content[0]["text"], PREFIX);
                    assert_eq!(content[1]["text"], &prompts[index][PREFIX.len()..]);
                    assert_eq!(content.as_array().unwrap().len(), 2);
                    if responses_api {
                        assert_eq!(content[0]["type"], "input_text");
                        assert_eq!(
                            content[0]["prompt_cache_breakpoint"],
                            json!({"mode": "explicit"})
                        );
                        assert!(content[0].get("cache_control").is_none());
                    } else {
                        assert_eq!(content[0]["type"], "text");
                        assert_eq!(content[0]["cache_control"], json!({"type": "ephemeral"}));
                    }
                    assert!(content[1].get("prompt_cache_breakpoint").is_none());
                    assert!(content[1].get("cache_control").is_none());
                } else if responses_api {
                    assert_eq!(body["instructions"], prompts[index]);
                    assert!(
                        body["input"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .all(|item| item["role"] != "system")
                    );
                } else {
                    assert_eq!(body["messages"][0]["content"], prompts[index]);
                }
            }
        }
    }
}

#[tokio::test]
async fn responses_catalogue_policy_follows_selected_vision_and_fallback_models() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const GLM: &str = "z-ai/glm-5.3-flash";
    const HAIKU: &str = "anthropic/claude-haiku-5.5";
    for (primary, vision, fallbacks, image, selected, eager) in [
        (GLM, None, vec![], false, GLM, false),
        (HAIKU, None, vec![GLM], false, HAIKU, true),
        (GLM, None, vec![HAIKU], false, GLM, false),
        (GLM, Some(HAIKU), vec![GLM], true, HAIKU, true),
        (HAIKU, Some(GLM), vec![GLM], true, GLM, false),
        (HAIKU, Some(GLM), vec![HAIKU], true, GLM, false),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openrouter/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "status":"completed", "output":[{"type":"message", "content":[
                    {"type":"output_text", "text":"Here."}
                ]}]
            })))
            .expect(1)
            .mount(&server)
            .await;
        let mut provider = wiremock_provider(&server.uri())
            .with_responses_api(true)
            .with_vision_model(vision.map(str::to_string))
            .with_fallback_models(fallbacks.iter().map(|m| m.to_string()).collect());
        provider.model = primary.into();
        let history = [if image {
            image_turn()
        } else {
            ToolTurn::User("Continue.".into())
        }];
        let tools = [
            json!({"name":"list_seedlings", "description":"Read permitted seedlings.",
                "parameters":{"type":"object","properties":{}}}),
            json!({"name":"grant_docking", "description":"A permitted specialist tool.",
                "parameters":{"type":"object","properties":{}}, "defer_loading":true}),
        ];
        provider
            .chat_step(
                "caller",
                SystemPrompt::with_cacheable_prefix("fixed\ndynamic", 5),
                &history,
                &tools,
            )
            .await
            .unwrap();
        let requests = server.received_requests().await.unwrap();
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["model"], selected);
        assert!(body["provider"].get("zdr").is_none());
        assert_eq!(body["provider"]["data_collection"], "deny");
        if !fallbacks.is_empty() {
            let models = std::iter::once(selected)
                .chain(fallbacks.iter().copied())
                .collect::<Vec<_>>();
            assert_eq!(body["models"], json!(models));
        }
        let actual_tools = body["tools"].as_array().unwrap();
        let names = actual_tools
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec!["list_seedlings", "grant_docking"],
            "only the supplied tools may be sent, in the supplied order"
        );
        if eager {
            assert_eq!(actual_tools.len(), 2);
            assert!(
                actual_tools
                    .iter()
                    .all(|tool| tool["type"] == "function" && tool.get("defer_loading").is_none())
            );
            assert_eq!(body["input"][0]["role"], "system");
            assert_eq!(
                body["input"][0]["content"][0]["prompt_cache_breakpoint"],
                json!({"mode":"explicit"})
            );
            assert_eq!(body["cache_control"], json!({"type":"ephemeral"}));
        } else {
            assert_eq!(actual_tools.len(), 3);
            assert_eq!(actual_tools[0]["type"], "openrouter:tool_search");
            assert_eq!(actual_tools[2]["defer_loading"], true);
            assert_eq!(body["instructions"], "fixed\ndynamic");
            assert!(body.get("cache_control").is_none());
        }
        if image {
            let expected = if selected == GLM {
                json!({"effort":"low"})
            } else {
                json!({"enabled":false})
            };
            assert_eq!(body["reasoning"], expected);
        }
    }
}

#[tokio::test]
async fn anthropic_eager_catalogue_and_cache_boundary_survive_empty_reasoning_retry() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openrouter/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status":"completed", "output":[]
        })))
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/openrouter/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status":"completed", "output":[{"type":"message", "content":[
                {"type":"output_text", "text":"Here."}
            ]}]
        })))
        .with_priority(6)
        .expect(1)
        .mount(&server)
        .await;
    let mut provider = wiremock_provider(&server.uri())
        .with_responses_api(true)
        .with_reasoning_effort(Some("high".into()));
    provider.model = "anthropic/claude-haiku-5.5".into();
    let caller = "caller";
    let old_call = crate::ToolCall {
        id: "old-call".into(),
        name: "grant_docking".into(),
        arguments: json!({}),
    };
    let reasoning = json!({"type":"reasoning", "id":"old-reasoning", "encrypted_content":"opaque"});
    provider.remember_reasoning(
        caller,
        &json!({"output":[reasoning]}),
        &crate::ChatStep::ToolCalls(vec![old_call.clone()]),
    );
    let history = [
        ToolTurn::User("Earlier request.".into()),
        ToolTurn::AssistantToolCalls(vec![old_call]),
        ToolTurn::ToolResults(vec![("old-call".into(), "Observed.".into())]),
        ToolTurn::User("Continue.".into()),
    ];
    let tools = [
        json!({"name":"grant_docking", "description":"Permitted specialist.",
        "parameters":{"type":"object","properties":{}}, "defer_loading":true}),
    ];
    provider
        .chat_step(
            caller,
            SystemPrompt::with_cacheable_prefix("fixed\ndynamic", 5),
            &history,
            &tools,
        )
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let bodies = requests
        .iter()
        .map(|r| serde_json::from_slice::<Value>(&r.body).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(bodies[0]["tools"], bodies[1]["tools"]);
    for body in &bodies {
        assert_eq!(body["tools"].as_array().unwrap().len(), 1);
        assert_eq!(body["tools"][0]["name"], "grant_docking");
        assert!(body["tools"][0].get("defer_loading").is_none());
        assert_eq!(body["input"][0]["role"], "system");
        assert_eq!(
            body["input"][0]["content"][0]["prompt_cache_breakpoint"],
            json!({"mode":"explicit"})
        );
        assert_eq!(body["cache_control"], json!({"type":"ephemeral"}));
        assert!(body["provider"].get("zdr").is_none());
    }
    assert!(
        bodies[0]["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["type"] == "reasoning")
    );
    assert!(
        bodies[1]["input"]
            .as_array()
            .unwrap()
            .iter()
            .all(|item| item["type"] != "reasoning")
    );
    assert_eq!(bodies[1]["reasoning"], json!({"enabled":false}));
}

#[tokio::test]
async fn reasoning_is_echoed_before_its_calls_for_the_same_caller_only() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    let reasoning = json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"});
    Mock::given(method("POST"))
        .and(path("/openrouter/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "completed", "model": "m", "usage": {},
            "output": [
                reasoning,
                {"type": "function_call", "call_id": "c1", "name": "list_seedlings", "arguments": "{}"}
            ]
        })))
        .mount(&server)
        .await;
    let provider = wiremock_provider(&server.uri())
        .with_responses_api(true)
        .with_reasoning_effort(Some("high".into()));
    let tools = [
        json!({"name": "list_seedlings", "description": "List.", "parameters": {"type": "object", "properties": {}}}),
    ];
    let crew = "crew-a";
    let first = vec![ToolTurn::User("What is on my list?".into())];
    let step = provider
        .chat_step(crew, "sys".into(), &first, &tools)
        .await
        .unwrap();
    let crate::ChatStep::ToolCalls(calls) = step else {
        panic!("expected calls")
    };
    let mut next = first.clone();
    next.push(ToolTurn::AssistantToolCalls(calls));
    next.push(ToolTurn::ToolResults(vec![(
        "c1".into(),
        "No seedlings.".into(),
    )]));
    provider
        .chat_step(crew, "sys".into(), &next, &tools)
        .await
        .unwrap();
    provider
        .chat_step("caller", "sys".into(), &next, &tools)
        .await
        .unwrap();

    let bodies = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|request| serde_json::from_slice::<Value>(&request.body).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(bodies[0]["include"], json!(["reasoning.encrypted_content"]));
    let same = bodies[1]["input"].as_array().unwrap();
    let call = same
        .iter()
        .position(|item| item["type"] == "function_call")
        .unwrap();
    assert_eq!(
        same[call - 1],
        reasoning,
        "reasoning precedes the call it produced"
    );
    let other = bodies[2]["input"].as_array().unwrap();
    assert!(
        other.iter().all(|item| item["type"] != "reasoning"),
        "another crew never receives this crew's reasoning"
    );
}

#[tokio::test]
async fn failed_requests_name_their_openrouter_generation() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openrouter/responses"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("x-generation-id", "gen-123")
                .set_body_json(json!({"error": {"code": 429, "message": "Slow down"}})),
        )
        .mount(&server)
        .await;
    let provider = wiremock_provider(&server.uri()).with_responses_api(true);
    let caller = "caller";
    let sent = provider
        .responses_step_send(caller, "m", &json!({}))
        .await
        .unwrap_err();
    let streamed = provider
        .responses_step_stream(caller, "m", &json!({}), &RecordingTextSink::default())
        .await
        .unwrap_err();
    for error in [sent, streamed] {
        assert!(error.to_string().contains("generation gen-123"), "{error}");
        assert!(error.is_retryable());
    }
}

#[tokio::test]
async fn responses_transport_requests_streaming_and_returns_the_completed_reply() {
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello \"}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"there.\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"m\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello there.\",\"annotations\":[]}]}],\"usage\":{\"input_tokens\":10,\"output_tokens\":3}}}\n\n"
    );
    Mock::given(method("POST"))
        .and(path("/openrouter/responses"))
        .and(body_partial_json(json!({ "stream": true, "store": false })))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .expect(1)
        .mount(&server)
        .await;

    let mut provider = wiremock_provider(&server.uri()).with_responses_api(true);
    provider.model = "anthropic/claude-haiku-5.5".into();
    let sink = Arc::new(RecordingTextSink::default());
    let step = provider
        .chat_step_stream(
            "caller",
            SystemPrompt::with_cacheable_prefix("sys\nvolatile context", 3),
            &[ToolTurn::User("Hello".into())],
            &[],
            sink.clone(),
        )
        .await
        .unwrap();

    assert!(matches!(step, crate::ChatStep::Text(ref text) if text == "Hello there."));
    assert_eq!(sink.0.lock().unwrap().concat(), "Hello there.");
    let requests = server.received_requests().await.unwrap();
    let request: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert!(request.get("instructions").is_none());
    assert_eq!(request["input"][0]["role"], "system");
    assert_eq!(request["input"][0]["content"][0]["text"], "sys");
    assert_eq!(
        request["input"][0]["content"][0]["prompt_cache_breakpoint"],
        json!({"mode": "explicit"})
    );
}

#[tokio::test]
async fn responses_terminal_event_does_not_wait_for_connection_close() {
    use futures::StreamExt;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router = axum::Router::new().route(
        "/openrouter/responses",
        axum::routing::post(|| async {
            let payload = json!({
                "type": "response.completed",
                "response": {
                    "status": "completed", "model": "m", "usage": {},
                    "output": [{"type": "message", "content": [
                        {"type": "output_text", "text": "Finished"}
                    ]}]
                }
            });
            let event = format!("data: {payload}\n\n");
            let stream = futures::stream::once(async move { Ok::<_, std::io::Error>(event) })
                .chain(futures::stream::pending());
            axum::response::Response::builder()
                .header("content-type", "text/event-stream")
                .body(axum::body::Body::from_stream(stream))
                .unwrap()
        }),
    );
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let provider = wiremock_provider(&format!("http://{address}")).with_responses_api(true);
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        provider.chat_step_stream(
            "caller",
            "sys".into(),
            &[ToolTurn::User("Hello".into())],
            &[],
            Arc::new(RecordingTextSink::default()),
        ),
    )
    .await;
    server.abort();
    let step = result
        .expect("completed response must not wait for HTTP EOF")
        .unwrap();
    assert!(matches!(step, crate::ChatStep::Text(ref text) if text == "Finished"));
}

#[test]
fn responses_string_codes_and_error_types_map_to_documented_statuses() {
    let failed = |code: &str, error_type: Option<&str>| {
        let (status, message) =
            responses_error(&json!({"code": code, "message": "failed"}), error_type);
        Error::from_status(status, message)
    };
    // The reduced code alone collapses context overflow into invalid_prompt;
    // the canonical type lets the tool loop shrink history and retry.
    let overflow = failed("invalid_prompt", Some("context_length_exceeded"));
    assert!(overflow.is_context_overflow(), "{overflow}");
    assert!(!failed("invalid_prompt", None).is_retryable());
    assert!(failed("rate_limit_exceeded", None).is_retryable());
    // server_error hides authentication; the canonical type does not.
    assert!(!failed("server_error", Some("authentication")).is_retryable());
    assert!(failed("server_error", Some("provider_overloaded")).is_retryable());
    assert!(failed("server_error", None).is_retryable());
    assert_eq!(openrouter_error_status(&json!({"code": 429}), None), 429);
    assert_eq!(openrouter_error_status(&json!({"code": "503"}), None), 503);
    assert_eq!(
        openrouter_error_status(
            &json!({"code": "x", "metadata": {"error_type": "timeout"}}),
            None
        ),
        504
    );
}

#[test]
fn a_failed_streamed_response_keeps_its_canonical_error_type() {
    let mut accumulator = ResponsesStreamAccumulator::default();
    let event = json!({
        "type": "error", "code": "invalid_prompt",
        "message": "Input too large", "error_type": "context_length_exceeded"
    });
    let error = accumulator
        .push(
            format!("data: {event}\n\n").as_bytes(),
            &RecordingTextSink::default(),
        )
        .unwrap_err();
    assert!(error.is_context_overflow(), "{error}");

    let provider = provider_at("https://openrouter.ai/api/v1");
    let payload = json!({
        "status": "failed", "error_type": "authentication",
        "error": {"code": "server_error", "message": "Invalid credentials"}
    });
    let error = provider
        .finish_responses_payload("caller", "m", payload, None, std::time::Instant::now())
        .unwrap_err();
    assert!(!error.is_retryable(), "{error}");
}

#[tokio::test]
async fn a_responses_in_flight_402_is_retryable_with_or_without_metadata() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let message = "This request would exceed your available credits given your current in-flight requests. Retry after in-flight requests settle, or add credits.";
    for body in [
        // The documented Responses shape: metadata beside error.
        json!({
            "error": {"code": 402, "message": "Payment required"},
            "metadata": {"limit_source": "openrouter_in_flight_budget"}
        }),
        // The shape observed live on 2026-09-27: no metadata at all.
        json!({"error": {"code": 402, "message": message}}),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openrouter/responses"))
            .respond_with(ResponseTemplate::new(402).set_body_json(body))
            .mount(&server)
            .await;
        let provider = wiremock_provider(&server.uri()).with_responses_api(true);
        let caller = "caller";
        let sent = provider
            .responses_step_send(caller, "m", &json!({}))
            .await
            .unwrap_err();
        let streamed = provider
            .responses_step_stream(caller, "m", &json!({}), &RecordingTextSink::default())
            .await
            .unwrap_err();
        for error in [sent, streamed] {
            assert!(error.is_retryable(), "{error}");
        }
    }
}

#[test]
fn http_error_messages_keep_openrouter_limit_metadata() {
    let body = json!({"error": {"message": "Payment required", "metadata": {
        "limit_source": "openrouter_in_flight_budget"
    }}});
    let message = http_error_message(&body.to_string());
    assert_eq!(message, "Payment required [openrouter_in_flight_budget]");
    assert!(Error::from_status(402, message).is_retryable());
    assert_eq!(
        http_error_message("<html>Bad gateway</html>"),
        "<html>Bad gateway</html>"
    );
}

#[tokio::test]
async fn a_stalled_responses_stream_fails_as_retryable() {
    use futures::StreamExt;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router = axum::Router::new().route(
        "/openrouter/responses",
        axum::routing::post(|| async {
            // One real event, then only keep-alive comments: no progress.
            let started = "data: {\"type\":\"response.created\"}\n\n".to_string();
            let keepalive =
                futures::stream::repeat_with(|| ": OPENROUTER PROCESSING\n\n".to_string()).then(
                    |comment| async move {
                        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                        comment
                    },
                );
            let stream = futures::stream::once(async move { started })
                .chain(keepalive)
                .map(Ok::<_, std::io::Error>);
            axum::response::Response::builder()
                .header("content-type", "text/event-stream")
                .body(axum::body::Body::from_stream(stream))
                .unwrap()
        }),
    );
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let provider = wiremock_provider(&format!("http://{address}"))
        .with_responses_api(true)
        .with_stream_stall(std::time::Duration::from_millis(300));
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        provider.responses_step_stream("caller", "m", &json!({}), &RecordingTextSink::default()),
    )
    .await;
    server.abort();
    let error = result
        .expect("a stalled stream must not wait for the request timeout")
        .unwrap_err();
    assert!(error.is_retryable(), "{error}");
    assert!(error.to_string().contains("stalled"), "{error}");
}

#[tokio::test]
async fn responses_http_errors_keep_budget_reason_in_both_transports() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    for (metadata, retryable) in [
        (json!({"reason": "in_flight_budget_exhausted"}), true),
        (json!({"limit_source": "openrouter_in_flight_budget"}), true),
        (json!({"limit_source": "openrouter_key_limit"}), false),
        (json!({"limit_source": "openrouter_credits"}), false),
        (json!({}), false),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openrouter/responses"))
            .respond_with(ResponseTemplate::new(402).set_body_json(json!({"error": {
                "message": "Payment required", "metadata": metadata
            }})))
            .expect(2)
            .mount(&server)
            .await;
        let provider = wiremock_provider(&server.uri()).with_responses_api(true);
        let caller = "caller";
        let plain = provider
            .responses_step_send(caller, "m", &json!({}))
            .await
            .unwrap_err();
        let streamed = provider
            .responses_step_stream(caller, "m", &json!({}), &RecordingTextSink::default())
            .await
            .unwrap_err();
        assert_eq!(plain.is_retryable(), retryable);
        assert_eq!(streamed.is_retryable(), retryable);
    }
}

#[tokio::test]
async fn responses_transport_rejects_incomplete_and_failed_payloads() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    for payload in [
        json!({
            "id":"resp_incomplete",
            "status":"incomplete",
            "incomplete_details":{"reason":"max_output_tokens"}
        }),
        json!({
            "id":"resp_failed",
            "status":"failed",
            "error":{"message":"provider unavailable"}
        }),
        json!({
            "id":"resp_cancelled",
            "status":"cancelled"
        }),
        json!({
            "id":"resp_missing_status",
            "output":[]
        }),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openrouter/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(payload))
            .expect(1)
            .mount(&server)
            .await;
        let provider = wiremock_provider(&server.uri()).with_responses_api(true);
        let error = provider
            .chat_step(
                "caller",
                "sys".into(),
                &[ToolTurn::User("hello".into())],
                &[],
            )
            .await
            .expect_err("non-completed Responses payload must fail closed");
        assert!(error.to_string().contains("Responses"), "{error}");
    }
}

#[tokio::test]
async fn responses_transport_replays_calls_and_results_without_previous_response_id() {
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openrouter/responses"))
        .and(body_partial_json(json!({
            "store": false,
            "input": [
                {"type":"message","role":"user"},
                {"type":"function_call","call_id":"call_1","name":"plant_seedling"},
                {"type":"function_call_output","call_id":"call_1","output":"{\"status\":\"applied\"}"}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id":"resp_2",
            "model":"m",
            "status":"completed",
            "output":[{"type":"message","content":[
                {"type":"output_text","text":"Added.","annotations":[]}
            ]}],
            "usage":{"input_tokens":12,"output_tokens":2}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let provider = wiremock_provider(&server.uri()).with_responses_api(true);
    let history = [
        ToolTurn::User("Plant moon basil".into()),
        ToolTurn::AssistantToolCalls(vec![crate::ToolCall {
            id: "call_1".into(),
            name: "plant_seedling".into(),
            arguments: json!({"title":"moon basil"}),
        }]),
        ToolTurn::ToolResults(vec![("call_1".into(), "{\"status\":\"applied\"}".into())]),
    ];
    let step = provider
        .chat_step("caller", "sys".into(), &history, &[])
        .await
        .unwrap();
    assert!(matches!(step, crate::ChatStep::Text(ref text) if text == "Added."));
}

#[test]
fn parses_hermes_xml_tool_call_from_content() {
    // The exact Hermes-XML shape these open models emit in `content`
    // (OpenRouter truncates its own `tool_calls` conversion of this, so the
    // XML is authoritative).
    let content = "<tool_call>\n<function=start_growth_cycle>\n\
        <parameter=which>St Gerard Majella</parameter>\n\
        <parameter=when>today</parameter>\n\
        <parameter=today>2026-06-19</parameter>\n\
        </function>\n</tool_call>";
    let calls = parse_hermes_tool_calls(content).expect("a tool call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "start_growth_cycle");
    assert_eq!(calls[0].arguments["which"], "St Gerard Majella");
    assert_eq!(calls[0].arguments["when"], "today");
    assert_eq!(calls[0].arguments["today"], "2026-06-19");
}

#[test]
fn parses_multiple_hermes_tool_calls() {
    let content = "<tool_call>\n<function=complete_cycle_day>\n\
        <parameter=date>2026-06-19</parameter>\n</function>\n</tool_call>\n\
        <tool_call>\n<function=complete_cycle_day>\n\
        <parameter=date>2026-06-19</parameter>\n</function>\n</tool_call>";
    let calls = parse_hermes_tool_calls(content).expect("two calls");
    assert_eq!(calls.len(), 2);
    assert!(calls.iter().all(|c| c.name == "complete_cycle_day"));
    // Synthesized ids must be distinct so tool-result turns don't collide.
    assert_ne!(calls[0].id, calls[1].id);
}

#[test]
fn hermes_call_with_no_parameters() {
    let content = "<tool_call>\n<function=list_seedlings>\n</function>\n</tool_call>";
    let calls = parse_hermes_tool_calls(content).expect("a call");
    assert_eq!(calls[0].name, "list_seedlings");
    assert!(calls[0].arguments.as_object().unwrap().is_empty());
}

#[test]
fn plain_text_is_not_a_tool_call() {
    assert!(parse_hermes_tool_calls("Started the growth cycle. Day 1: …").is_none());
    assert!(parse_hermes_tool_calls("").is_none());
}

#[test]
fn clean_native_tool_calls_win_and_content_is_ignored() {
    use crate::ChatStep;
    // A clean-JSON model (deepseek): valid tool_calls. Even if its prose in
    // `content` literally contained a <tool_call> block, the native call must
    // be used and the XML fallback must NOT fire — no interference.
    let msg = json!({
        "content": "Sure — calling <tool_call><function=other>...</tool_call> for you.",
        "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": { "name": "plant_seedling", "arguments": "{\"title\": \"dwarf kale\"}" }
        }]
    });
    let (step, parse) = interpret_tool_message(&msg);
    assert_eq!(
        parse,
        ToolParse::Native,
        "a clean model must not hit the XML fallback"
    );
    match step {
        ChatStep::ToolCalls(calls) => {
            assert_eq!(calls.len(), 1);
            assert_eq!(
                calls[0].name, "plant_seedling",
                "the NATIVE call wins, not the XML in content"
            );
            assert_eq!(calls[0].arguments["title"], "dwarf kale");
        }
        other => panic!("expected the native tool call, got {other:?}"),
    }
}

#[test]
fn xml_fallback_only_fires_when_native_calls_are_unusable() {
    use crate::ChatStep;
    // The truncation case: tool_calls present but arguments truncated; the
    // complete call is the XML in content. The fallback recovers it.
    let msg = json!({
        "content": "<tool_call><function=plant_seedling><parameter=title>dwarf kale</parameter></function></tool_call>",
        "tool_calls": [{
            "id": "c", "type": "function",
            "function": { "name": "plant_seedling", "arguments": "{\"title\": " }  // truncated
        }]
    });
    let (step, parse) = interpret_tool_message(&msg);
    assert_eq!(
        parse,
        ToolParse::XmlAfterMalformedNative,
        "recovered from XML, and flagged it"
    );
    match step {
        ChatStep::ToolCalls(calls) => assert_eq!(calls[0].arguments["title"], "dwarf kale"),
        other => panic!("expected XML recovery, got {other:?}"),
    }

    // A plain text reply with no tool_calls and no XML stays text (Native).
    let msg = json!({ "content": "Started the growth cycle. Day 1: …" });
    let (step, parse) = interpret_tool_message(&msg);
    assert!(matches!(step, ChatStep::Text(_)));
    assert_eq!(parse, ToolParse::Native);
}

#[test]
fn empty_arguments_string_is_treated_as_an_empty_object_not_dropped() {
    use crate::ChatStep;
    // Some providers send arguments:"" (not "{}") for a param-less tool. That
    // call must SURVIVE as an empty-object call, or list_seedlings / a scheduled
    // job's reads silently never run.
    let msg = json!({
        "tool_calls": [{
            "id": "c1", "type": "function",
            "function": { "name": "list_seedlings", "arguments": "" }
        }]
    });
    let (step, parse) = interpret_tool_message(&msg);
    assert_eq!(parse, ToolParse::Native);
    match step {
        ChatStep::ToolCalls(calls) => {
            assert_eq!(calls.len(), 1, "the param-less call must not be dropped");
            assert_eq!(calls[0].name, "list_seedlings");
            assert!(calls[0].arguments.is_object());
        }
        other => panic!("expected the call to survive, got {other:?}"),
    }
}

#[test]
fn missing_native_id_is_synthesized_distinctly() {
    use crate::ChatStep;
    // Two parallel native calls with no id must not collapse to the same
    // empty tool_call_id (strict providers reject/mis-attribute that).
    let msg = json!({
        "tool_calls": [
            { "type": "function", "function": { "name": "list_seedlings", "arguments": "{}" } },
            { "type": "function", "function": { "name": "list_intentions", "arguments": "{}" } }
        ]
    });
    let (step, _) = interpret_tool_message(&msg);
    match step {
        ChatStep::ToolCalls(calls) => {
            assert_eq!(calls.len(), 2);
            assert!(
                !calls[0].id.is_empty() && !calls[1].id.is_empty(),
                "ids synthesized"
            );
            assert_ne!(calls[0].id, calls[1].id, "ids must be distinct");
        }
        other => panic!("expected two calls, got {other:?}"),
    }
}

#[test]
fn malformed_native_with_no_xml_is_preserved_for_a_paired_validation_result() {
    use crate::ChatStep;
    // Truncated native args and no XML to recover from: keep the named call
    // with its raw arguments. The dispatch boundary will reject the non-object
    // value and feed one invalid_arguments result back to the model.
    let msg = json!({
        "content": "",
        "tool_calls": [{
            "id": "c", "type": "function",
            "function": { "name": "plant_seedling", "arguments": "{\"title\": " }  // truncated, no XML
        }]
    });
    let (step, parse) = interpret_tool_message(&msg);
    assert_eq!(parse, ToolParse::InvalidNativeArguments);
    match step {
        ChatStep::ToolCalls(calls) => {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].id, "c");
            assert_eq!(calls[0].name, "plant_seedling");
            assert!(calls[0].arguments.is_string());
        }
        other => panic!("expected the invalid call to survive, got {other:?}"),
    }
}

#[test]
fn partial_malformed_parallel_calls_all_receive_a_lifecycle() {
    use crate::ChatStep;
    // The second call is truncated. A partial XML copy must not displace the
    // native batch: both IDs survive, so one call executes and the other gets
    // a deterministic validation result.
    let msg = json!({
        "content": "<tool_call><function=plant_seedling><parameter=title>dwarf kale</parameter></function></tool_call>",
        "tool_calls": [
            { "id": "c1", "type": "function",
              "function": { "name": "plant_seedling", "arguments": "{\"title\": \"dwarf kale\"}" } },
            { "id": "c2", "type": "function",
              "function": { "name": "water_bed", "arguments": "{\"body\": " } }  // truncated
        ]
    });
    let (step, parse) = interpret_tool_message(&msg);
    assert_eq!(
        parse,
        ToolParse::PartialInvalidNativeArguments,
        "a malformed sibling must be observable without losing its lifecycle"
    );
    match step {
        ChatStep::ToolCalls(calls) => {
            assert_eq!(calls.len(), 2, "every named call is kept");
            assert_eq!(calls[0].name, "plant_seedling");
            assert!(calls[0].arguments.is_object());
            assert_eq!(calls[1].name, "water_bed");
            assert!(calls[1].arguments.is_string());
        }
        other => panic!("expected the complete native call batch, got {other:?}"),
    }

    // If XML holds the complete batch it wins, recovering the truncated call.
    let msg = json!({
        "content": "<tool_call><function=plant_seedling><parameter=title>dwarf kale</parameter></function></tool_call>\
                    <tool_call><function=water_bed><parameter=body>bed 4</parameter></function></tool_call>",
        "tool_calls": [
            { "id": "c1", "type": "function",
              "function": { "name": "plant_seedling", "arguments": "{\"title\": \"dwarf kale\"}" } },
            { "id": "c2", "type": "function",
              "function": { "name": "water_bed", "arguments": "{\"body\": " } }  // truncated
        ]
    });
    let (step, parse) = interpret_tool_message(&msg);
    assert_eq!(
        parse,
        ToolParse::XmlAfterMalformedNative,
        "XML wins when it recovers the complete batch"
    );
    match step {
        ChatStep::ToolCalls(calls) => {
            assert_eq!(calls.len(), 2, "both calls recovered from XML")
        }
        other => panic!("expected XML recovery of both calls, got {other:?}"),
    }
}

#[test]
fn duplicate_native_call_ids_are_made_unique_without_losing_calls() {
    use crate::ChatStep;
    let msg = json!({
        "tool_calls": [
            { "id": "same", "type": "function",
              "function": { "name": "list_seedlings", "arguments": "{}" } },
            { "id": "same", "type": "function",
              "function": { "name": "list_intentions", "arguments": "{}" } }
        ]
    });
    let (step, parse) = interpret_tool_message(&msg);
    assert_eq!(parse, ToolParse::Native);
    match step {
        ChatStep::ToolCalls(calls) => {
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0].id, "same");
            assert_eq!(calls[1].id, "same_1");
        }
        other => panic!("expected two calls, got {other:?}"),
    }
}

#[test]
fn direct_object_arguments_from_a_compatible_provider_are_accepted() {
    use crate::ChatStep;
    let msg = json!({
        "tool_calls": [{
            "id": "c1", "type": "function",
            "function": { "name": "plant_seedling", "arguments": {"title": "dwarf kale"} }
        }]
    });
    let (step, parse) = interpret_tool_message(&msg);
    assert_eq!(parse, ToolParse::Native);
    assert!(matches!(step, ChatStep::ToolCalls(ref calls)
        if calls[0].arguments == json!({"title": "dwarf kale"})));
}

#[test]
fn hermes_params_stay_verbatim_strings() {
    // The XML parser must NOT guess types — every value stays a string, so a
    // numeric-looking string field (a text field "100") is never corrupted.
    // Type reconciliation happens later, schema-aware, at dispatch.
    let content = "<tool_call>\n<function=recommend_cultivar>\n\
        <parameter=today>2026-06-19</parameter>\n\
        <parameter=days_ahead>30</parameter>\n\
        <parameter=heirloom>true</parameter>\n</function>\n</tool_call>";
    let calls = parse_hermes_tool_calls(content).unwrap();
    let a = &calls[0].arguments;
    assert_eq!(
        a["days_ahead"].as_str(),
        Some("30"),
        "kept verbatim as a string"
    );
    assert_eq!(
        a["heirloom"].as_str(),
        Some("true"),
        "kept verbatim as a string"
    );
    assert_eq!(
        a["today"].as_str(),
        Some("2026-06-19"),
        "date stays a string"
    );
}

fn provider(vision: Option<&str>) -> OpenAiCompat {
    let usage = Arc::new(NoUsage);
    OpenAiCompat::new("https://example.test/v1", "k", "text-model", usage)
        .with_vision_model(vision.map(str::to_string))
}

fn image_turn() -> ToolTurn {
    ToolTurn::UserWithImages(
        "what is this?".into(),
        vec![InputImage {
            mime: "image/png".into(),
            data_b64: "aGk=".into(),
        }],
    )
}

fn openrouter(models: Vec<&str>) -> OpenAiCompat {
    let usage = Arc::new(NoUsage);
    OpenAiCompat::new("https://openrouter.ai/api/v1", "k", "primary/model", usage)
        .with_fallback_models(models.into_iter().map(str::to_string).collect())
}

#[test]
fn fallback_models_become_an_ordered_list_on_openrouter() {
    let p = openrouter(vec!["vendor/backup-1", "vendor/backup-2"]);
    let mut body = json!({ "model": "primary/model" });
    p.apply_provider_quirks(&mut body, None);
    // Primary first, then the fallbacks in order.
    assert_eq!(
        body["models"],
        json!(["primary/model", "vendor/backup-1", "vendor/backup-2"])
    );
}

#[test]
fn no_models_array_without_fallbacks_or_off_openrouter() {
    // No fallbacks configured → no `models` field at all.
    let p = openrouter(vec![]);
    let mut body = json!({ "model": "primary/model" });
    p.apply_provider_quirks(&mut body, None);
    assert!(body.get("models").is_none());
    // Fallbacks set but NOT OpenRouter → still no `models` (OpenAI rejects it).
    let p = provider(None);
    let p = p.with_fallback_models(vec!["a".into()]);
    let mut body = json!({ "model": "text-model" });
    p.apply_provider_quirks(&mut body, None);
    assert!(body.get("models").is_none());
}

#[test]
fn text_turns_use_the_default_model() {
    let p = provider(Some("vision-model"));
    assert_eq!(p.model_for(&[ToolTurn::User("hi".into())]), "text-model");
}

#[test]
fn image_turns_use_the_vision_model_when_configured() {
    let p = provider(Some("vision-model"));
    assert_eq!(
        p.model_for(&[ToolTurn::User("hi".into()), image_turn()]),
        "vision-model"
    );
}

#[test]
fn image_turns_fall_back_to_the_default_without_a_vision_model() {
    let p = provider(None);
    assert_eq!(p.model_for(&[image_turn()]), "text-model");
}

fn provider_at(base_url: &str) -> OpenAiCompat {
    let usage = Arc::new(NoUsage);
    OpenAiCompat::new(base_url, "k", "m", usage)
}

#[test]
fn openrouter_requests_disable_reasoning() {
    let p = provider_at("https://openrouter.ai/api/v1");
    let mut body = json!({ "model": "m" });
    p.apply_provider_quirks(&mut body, None);
    assert_eq!(body["reasoning"], json!({ "enabled": false }));
    // Data-collection-deny rides on the `provider` object; ZDR is not
    // requested because it excludes Anthropic BYOK.
    assert_eq!(
        body["provider"],
        json!({ "data_collection": "deny", "sort": "throughput" })
    );
    // Usage and cost are always returned; the old opt-in flag is deprecated.
    assert!(body.get("usage").is_none());
}

#[test]
fn haiku_enables_caching_and_omits_sampling_without_changing_privacy() {
    let provider = provider_at("https://openrouter.ai/api/v1")
        .with_openrouter_provider_ignore(vec!["amazon-bedrock".into()]);
    for limit in ["max_tokens", "max_output_tokens"] {
        let mut body = json!({"model":"anthropic/claude-haiku-5.5", "temperature":0});
        body[limit] = json!(200);
        provider.apply_provider_quirks(&mut body, None);
        assert!(body.get("temperature").is_none());
        assert_eq!(body["cache_control"], json!({"type":"ephemeral"}));
        assert_eq!(body["reasoning"], json!({"enabled":false}));
        assert!(body["provider"].get("zdr").is_none());
        assert_eq!(body["provider"]["data_collection"], "deny");
        assert_eq!(body["provider"]["ignore"], json!(["amazon-bedrock"]));
        assert_eq!(body[limit], 200);
    }
    let mut other = json!({"model":"some/model", "temperature":0});
    provider.apply_provider_quirks(&mut other, None);
    assert_eq!(other["temperature"], 0);
    assert!(other.get("cache_control").is_none());
    let non_router = provider_at("https://api.openai.com/v1")
        .with_openrouter_provider_ignore(vec!["amazon-bedrock".into()]);
    let mut body = json!({"model":"m"});
    non_router.apply_provider_quirks(&mut body, None);
    assert!(body.get("provider").is_none());
}

#[tokio::test]
async fn glm_completion_with_haiku_fallback_keeps_string_messages_and_omits_temperature() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"finish_reason":"stop", "message":{"content":"Europe/London"}}]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let mut provider = wiremock_provider(&server.uri())
        .with_fallback_models(vec!["anthropic/claude-haiku-5.5".into()]);
    provider.model = "z-ai/glm-5.3-flash".into();
    assert_eq!(
        provider
            .complete(
                "caller",
                Completion::new("Name the IANA timezone.", "London time", 24)
            )
            .await
            .unwrap(),
        "Europe/London"
    );
    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert!(body.get("temperature").is_none());
    assert!(body.get("cache_control").is_none());
    assert_eq!(
        body["models"],
        json!(["z-ai/glm-5.3-flash", "anthropic/claude-haiku-5.5"])
    );
    assert!(
        body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["content"].is_string())
    );
}

#[tokio::test]
async fn haiku_completion_sends_a_valid_cached_request() {
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .and(body_partial_json(json!({
            "model":"anthropic/claude-haiku-5.5",
            "cache_control":{"type":"ephemeral"},
            "provider":{"ignore":["amazon-bedrock"],"data_collection":"deny"},
            "reasoning":{"enabled":false}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"finish_reason":"stop","message":{"content":"Europe/London"}}]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let mut provider = wiremock_provider(&server.uri())
        .with_openrouter_provider_ignore(vec!["amazon-bedrock".into()]);
    provider.model = "anthropic/claude-haiku-5.5".into();
    assert_eq!(
        provider
            .complete(
                "caller",
                Completion::new("Name the IANA timezone.", "London time", 24)
            )
            .await
            .unwrap(),
        "Europe/London"
    );
    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert!(body.get("temperature").is_none());
}

#[test]
fn openrouter_session_is_stable_and_does_not_expose_the_scope() {
    let p = provider_at("https://openrouter.ai/api/v1");
    let mut first = json!({});
    let mut second = json!({});
    p.apply_openrouter_session(&mut first, "crew-secret-id", "safety_identifier");
    p.apply_openrouter_session(&mut second, "crew-secret-id", "safety_identifier");
    assert_eq!(first["session_id"], second["session_id"]);
    let session = first["session_id"].as_str().unwrap();
    assert_eq!(session.len(), 64);
    assert!(!session.contains("crew-secret-id"));
    // A stable per-crew abuse identity, distinct from the routing key.
    assert_eq!(first["safety_identifier"], second["safety_identifier"]);
    let end_user = first["safety_identifier"].as_str().unwrap();
    assert_eq!(end_user.len(), 64);
    assert_ne!(end_user, session);
    let mut other = json!({});
    p.apply_openrouter_session(&mut other, "another-crew", "safety_identifier");
    assert_ne!(other["safety_identifier"], first["safety_identifier"]);
    let mut chat = json!({});
    p.apply_openrouter_session(&mut chat, "crew-secret-id", "user");
    assert_eq!(chat["user"], first["safety_identifier"]);

    let non_router = provider_at("https://api.openai.com/v1");
    let mut body = json!({});
    non_router.apply_openrouter_session(&mut body, "scope", "user");
    assert!(body.get("session_id").is_none());
    assert!(body.get("user").is_none());
}

#[test]
fn openrouter_reasoning_effort_replaces_the_disable_and_adds_headroom() {
    let p = provider_at("https://openrouter.ai/api/v1").with_reasoning_effort(Some("max".into()));
    let mut body = json!({ "model": "m", "max_tokens": 500 });
    p.apply_provider_quirks(&mut body, p.reasoning_effort.as_deref());
    assert_eq!(body["reasoning"], json!({ "effort": "max" }));
    // Reasoning tokens count toward max_tokens — the budget grows so
    // content isn't starved to empty on hard prompts.
    assert_eq!(body["max_tokens"], json!(24_500));
    // The privacy floor is untouched by the reasoning mode.
    assert_eq!(
        body["provider"],
        json!({ "data_collection": "deny", "sort": "throughput" })
    );

    // The SMALL-call path pins reasoning off even when an effort is
    // configured: classify/timezone must never pay reasoning latency or
    // route into the reasoning pool (they are the degraded path).
    let mut small = json!({ "model": "m", "max_tokens": 256 });
    p.apply_provider_quirks(&mut small, None);
    assert_eq!(small["reasoning"], json!({ "enabled": false }));
    assert_eq!(
        small["max_tokens"],
        json!(256),
        "no headroom inflation on small calls"
    );
}

#[test]
fn mandatory_reasoning_model_uses_supported_low_effort_on_small_calls_and_retries() {
    let p = provider_at("https://openrouter.ai/api/v1");
    let mut body = json!({ "model": "z-ai/glm-5.3-flash", "max_tokens": 24 });
    p.apply_provider_quirks(&mut body, None);
    assert_eq!(body["reasoning"], json!({ "effort": "low" }));
    assert_eq!(body["max_tokens"], json!(24_024));
    let mut retry = json!({ "model": "z-ai/glm-5.3-flash", "max_tokens": 24 });
    p.apply_provider_quirks(&mut retry, Some("minimal"));
    assert_eq!(retry["reasoning"], json!({ "effort": "low" }));
}

#[test]
fn openai_requests_carry_no_reasoning_field() {
    let p = provider_at("https://api.openai.com/v1")
        .with_openrouter_provider_sort(OpenRouterProviderSort::Price);
    let mut body = json!({ "model": "m" });
    p.apply_provider_quirks(&mut body, None);
    assert!(
        body.get("reasoning").is_none(),
        "OpenAI rejects unknown params"
    );
    // OpenAI proper also rejects the unknown `provider` param entirely.
    assert!(
        body.get("provider").is_none(),
        "OpenAI rejects unknown params"
    );
}

#[test]
fn responses_preserve_temporary_budget_reason_for_retry() {
    let provider = provider_at("https://openrouter.ai/api/v1");
    for (reason, retryable) in [(Some("in_flight_budget_exhausted"), true), (None, false)] {
        let payload = json!({"error": {
            "code": 402,
            "message": "Payment required",
            "metadata": {"reason": reason}
        }});
        let caller = "background";
        let error = provider
            .finish_responses_payload(
                caller,
                "z-ai/glm-5.3-flash",
                payload.clone(),
                Some(7),
                std::time::Instant::now(),
            )
            .unwrap_err();
        assert_eq!(error.is_retryable(), retryable);
        assert_eq!(
            embedded_error(&payload, Some(7)).unwrap().is_retryable(),
            retryable
        );
        let http_error = Error::from_status_with_retry_after(
            402,
            provider_error_message(&payload["error"]),
            Some(7),
        );
        assert_eq!(http_error.is_retryable(), retryable);
    }
}

#[test]
fn embedded_errors_map_by_their_wrapped_status() {
    // The live shape: HTTP 200, body = {"error": {...}} and no choices.
    let payload = json!({
        "error": { "message": "x is temporarily rate-limited upstream", "code": 429 }
    });
    let e = embedded_error(&payload, Some(7)).expect("an embedded 429 is an error");
    assert!(e.is_retryable(), "a wrapped 429 must stay retryable: {e}");
    assert_eq!(e.retry_after_secs(), Some(7), "Retry-After carried through");

    // A wrapped hard failure stays hard — no retry burn on a 400.
    let payload = json!({ "error": { "message": "bad request", "code": 400 } });
    assert!(!embedded_error(&payload, None).unwrap().is_retryable());

    // A healthy response is never mistaken for an embedded error, even if
    // the provider echoes an `error: null`-ish field alongside choices.
    let payload = json!({ "choices": [{ "message": { "content": "hi" } }] });
    assert!(embedded_error(&payload, None).is_none());

    // No choices and NO error object → not this shape (falls through to
    // the API-change path with a body snippet).
    assert!(embedded_error(&json!({}), None).is_none());
}

#[test]
fn empty_messages_are_degenerate_but_tool_call_messages_are_not() {
    assert!(message_is_empty(&json!({ "content": "" })));
    assert!(message_is_empty(
        &json!({ "content": null, "reasoning": "…thought…" })
    ));
    assert!(message_is_empty(&json!({ "content": "  \n" })));
    // A tool-call message legitimately has empty content — NOT degenerate.
    assert!(!message_is_empty(&json!({
        "content": "",
        "tool_calls": [{ "function": { "name": "list_seedlings", "arguments": "{}" } }]
    })));
    assert!(!message_is_empty(&json!({ "content": "Here." })));
}

/// A provider pointed at a wiremock server, on a path containing
/// "openrouter" so the OpenRouter quirks (reasoning field, privacy) apply.
fn wiremock_provider(uri: &str) -> OpenAiCompat {
    let usage = Arc::new(NoUsage);
    OpenAiCompat::new(format!("{uri}/openrouter"), "k", "m", usage)
}

#[tokio::test]
async fn openrouter_provider_sort_preserves_privacy_and_reasoning_on_both_transports() {
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    for (sort, expected_sort) in [
        (OpenRouterProviderSort::Throughput, "throughput"),
        (OpenRouterProviderSort::Price, "price"),
    ] {
        for responses_api in [false, true] {
            let server = MockServer::start().await;
            let endpoint = if responses_api {
                "/openrouter/responses"
            } else {
                "/openrouter/chat/completions"
            };
            let response = if responses_api {
                json!({
                    "status": "completed",
                    "output": [{"type": "message", "content": [{
                        "type": "output_text", "text": "Here."
                    }]}]
                })
            } else {
                json!({
                    "choices": [{"finish_reason": "stop", "message": {"content": "Here."}}]
                })
            };
            Mock::given(method("POST"))
                .and(path(endpoint))
                .and(body_partial_json(json!({
                    "provider": {
                        "data_collection": "deny",
                        "sort": expected_sort
                    },
                    "reasoning": {"effort": "high"},
                    "model": "m"
                })))
                .respond_with(ResponseTemplate::new(200).set_body_json(response))
                .expect(1)
                .mount(&server)
                .await;

            let mut provider = wiremock_provider(&server.uri())
                .with_responses_api(responses_api)
                .with_reasoning_effort(Some("high".into()));
            if sort == OpenRouterProviderSort::Price {
                provider = provider.with_openrouter_provider_sort(sort);
            }
            let step = provider
                .chat_step("caller", "sys".into(), &[ToolTurn::User("hi".into())], &[])
                .await
                .unwrap();
            assert!(matches!(step, crate::ChatStep::Text(text) if text == "Here."));
        }
    }
}

#[tokio::test]
async fn chat_step_retries_without_reasoning_when_the_reasoning_pool_rate_limits() {
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    // Attempt 1 (reasoning effort on the wire): OpenRouter wraps the
    // upstream rate limit in an HTTP 200 — the exact live failure shape.
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .and(body_partial_json(
            json!({ "reasoning": { "effort": "max" } }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "error": { "message": "m is temporarily rate-limited upstream", "code": 429 }
        })))
        .expect(1)
        .mount(&server)
        .await;
    // Attempt 2 must carry reasoning DISABLED and the un-inflated budget —
    // the turn is salvaged without reasoning, privacy floor untouched.
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .and(body_partial_json(
            json!({ "reasoning": { "enabled": false }, "max_tokens": 1024 }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "finish_reason": "stop", "message": { "content": "Here." } }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let p = wiremock_provider(&server.uri()).with_reasoning_effort(Some("max".into()));
    let step = p
        .chat_step("caller", "sys".into(), &[ToolTurn::User("hi".into())], &[])
        .await
        .expect("the no-reasoning retry salvages the turn");
    match step {
        crate::ChatStep::Text(t) => assert_eq!(t, "Here."),
        other => panic!("expected the salvaged text reply, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_step_retries_without_reasoning_after_a_hard_reasoning_rejection() {
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .and(body_partial_json(
            json!({ "reasoning": { "effort": "max" } }),
        ))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": { "message": "reasoning is not supported by this upstream" }
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .and(body_partial_json(
            json!({ "reasoning": { "enabled": false } }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "finish_reason": "stop", "message": { "content": "Plain." } }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let p = wiremock_provider(&server.uri()).with_reasoning_effort(Some("max".into()));
    let step = p
        .chat_step("caller", "sys".into(), &[ToolTurn::User("hi".into())], &[])
        .await
        .expect("a hard reasoning-only rejection is salvageable");
    assert!(matches!(step, crate::ChatStep::Text(ref t) if t == "Plain."));
}

#[tokio::test]
async fn chat_step_discards_a_length_truncated_reply_and_retries_without_reasoning() {
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .and(body_partial_json(
            json!({ "reasoning": { "effort": "max" } }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "finish_reason": "length", "message": {
                "content": "Cultivar logged. For the watering: (Also"
            } }]
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .and(body_partial_json(json!({ "reasoning": { "enabled": false } })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "finish_reason": "stop", "message": {
                "content": "Cultivar logged. What watering time do you want? Bay 7 added Kelp trellis for Saturday."
            } }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let p = wiremock_provider(&server.uri()).with_reasoning_effort(Some("max".into()));
    let step = p
        .chat_step(
            "caller",
            "sys".into(),
            &[ToolTurn::User("compound".into())],
            &[],
        )
        .await
        .expect("the complete no-reasoning retry replaces the truncated reply");
    assert!(matches!(step, crate::ChatStep::Text(ref text) if text.contains("Kelp trellis")));
}

#[tokio::test]
async fn image_turns_use_the_vision_model_without_reasoning() {
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .and(body_partial_json(json!({
            "model": "vision/model",
            "reasoning": { "enabled": false },
            "max_tokens": 1024
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "finish_reason": "stop", "message": { "content": "I see it." } }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let p = wiremock_provider(&server.uri())
        .with_vision_model(Some("vision/model".into()))
        .with_reasoning_effort(Some("max".into()));
    let history = [ToolTurn::UserWithImages(
        "What is this?".into(),
        vec![crate::InputImage {
            mime: "image/png".into(),
            data_b64: "AA==".into(),
        }],
    )];
    let step = p
        .chat_step("caller", "sys".into(), &history, &[])
        .await
        .expect("vision requests do not inherit text-model reasoning");
    assert!(matches!(step, crate::ChatStep::Text(ref t) if t == "I see it."));
}

#[tokio::test]
async fn chat_step_retries_without_reasoning_when_content_comes_back_empty() {
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    // Attempt 1: 200 OK, choices present, but the reasoning stream ate the
    // reply — no tool calls, empty content (the other live Luna shape).
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .and(body_partial_json(
            json!({ "reasoning": { "effort": "max" } }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "finish_reason": "stop",
                          "message": { "content": "", "reasoning": "…thinking…" } }]
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .and(body_partial_json(
            json!({ "reasoning": { "enabled": false } }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "finish_reason": "stop", "message": { "content": "Recovered." } }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let p = wiremock_provider(&server.uri()).with_reasoning_effort(Some("max".into()));
    let step = p
        .chat_step("caller", "sys".into(), &[ToolTurn::User("hi".into())], &[])
        .await
        .expect("the empty reply is retried, not surfaced");
    match step {
        crate::ChatStep::Text(t) => assert_eq!(t, "Recovered."),
        other => panic!("expected the recovered text reply, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_step_surfaces_an_embedded_429_as_retryable_when_reasoning_is_off() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    // Prod shape (no reasoning configured): the embedded 429 must come out
    // RETRYABLE so FallbackLlm backs off and retries — and exactly ONE
    // request is sent (no reasoning to drop, so no second attempt here).
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "error": { "message": "m is temporarily rate-limited upstream", "code": 429 }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let p = wiremock_provider(&server.uri());
    let err = p
        .chat_step("caller", "sys".into(), &[ToolTurn::User("hi".into())], &[])
        .await
        .expect_err("the wrapped 429 is an error, not an empty reply");
    assert!(
        err.is_retryable(),
        "wrapped 429 must be retryable, got: {err}"
    );
}

fn sse(events: &[&str]) -> String {
    events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>()
        + "data: [DONE]\n\n"
}

async fn stream_server(body: String) -> wiremock::MockServer {
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .and(body_partial_json(
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(body, "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;
    server
}

fn streamed_text(sink: &RecordingTextSink) -> String {
    sink.0.lock().unwrap().concat()
}

#[tokio::test]
async fn chat_completions_stream_text_and_assemble_tool_call_fragments() {
    let server = stream_server(sse(&[
        r#"{"model":"vendor/ran","choices":[{"delta":{"content":"Checking "}}]}"#,
        r#"{"choices":[{"delta":{"content":"bed 4."}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"water_bed","arguments":"{\"be"}}]}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"d\":4}"}}]},"finish_reason":"tool_calls"}]}"#,
        r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#,
    ]))
    .await;
    let usage = Arc::new(RecordingUsage::default());
    let provider = OpenAiCompat::new(
        format!("{}/openrouter", server.uri()),
        "k",
        "m",
        usage.clone(),
    );
    let sink = Arc::new(RecordingTextSink::default());
    let step = provider
        .chat_step_stream(
            "crew-a",
            "sys".into(),
            &[ToolTurn::User("Water bed 4".into())],
            &[],
            sink.clone(),
        )
        .await
        .unwrap();
    let ChatStep::ToolCalls(calls) = step else {
        panic!("expected a tool call")
    };
    assert_eq!(
        (calls[0].id.as_str(), calls[0].name.as_str()),
        ("c1", "water_bed")
    );
    assert_eq!(calls[0].arguments, json!({"bed": 4}));
    assert_eq!(streamed_text(&sink), "Checking bed 4.");
    let records = usage.0.lock().unwrap();
    assert_eq!(
        (
            records[0].1.as_str(),
            records[0].2.input_tokens,
            records[0].2.output_tokens
        ),
        ("vendor/ran", 10, 5)
    );
}

#[tokio::test]
async fn streamed_hermes_xml_tool_calls_never_reach_the_user() {
    let server = stream_server(sse(&[
        r#"{"choices":[{"delta":{"content":"  <tool"}}]}"#,
        r#"{"choices":[{"delta":{"content":"_call><function=water_bed><parameter=bed>4</parameter></function></tool_call>"},"finish_reason":"stop"}]}"#,
    ]))
    .await;
    let sink = Arc::new(RecordingTextSink::default());
    let step = wiremock_provider(&server.uri())
        .chat_step_stream(
            "crew-a",
            "sys".into(),
            &[ToolTurn::User("Water bed 4".into())],
            &[],
            sink.clone(),
        )
        .await
        .unwrap();
    assert!(matches!(step, ChatStep::ToolCalls(ref calls) if calls[0].name == "water_bed"));
    assert_eq!(streamed_text(&sink), "");
}

#[tokio::test]
async fn a_streamed_reply_cut_off_at_the_token_ceiling_is_a_typed_stop() {
    let server = stream_server(sse(&[
        r#"{"choices":[{"delta":{"content":"Bed 4 needs"},"finish_reason":"length"}]}"#,
    ]))
    .await;
    let sink = Arc::new(RecordingTextSink::default());
    let err = wiremock_provider(&server.uri())
        .chat_step_stream(
            "crew-a",
            "sys".into(),
            &[ToolTurn::User("Status?".into())],
            &[],
            sink,
        )
        .await
        .unwrap_err();
    assert_eq!(err.stop_reason(), Some(crate::StopReason::MaxTokens));
    assert!(err.is_retryable());
}

#[test]
fn a_short_ambiguous_reply_is_still_delivered() {
    let sink = RecordingTextSink::default();
    let mut visible = VisibleText::default();
    visible.push("<");
    visible.flush(&sink, false);
    assert_eq!(streamed_text(&sink), "");
    visible.flush(&sink, true);
    assert_eq!(streamed_text(&sink), "<");
}

#[test]
fn incomplete_responses_name_their_stop_reason() {
    let provider = provider_at("https://openrouter.ai/api/v1");
    for (reason, expected) in [
        ("max_output_tokens", crate::StopReason::MaxTokens),
        ("content_filter", crate::StopReason::ContentFilter),
        ("something_new", crate::StopReason::Incomplete),
    ] {
        let payload = json!({"status": "incomplete", "incomplete_details": {"reason": reason}});
        let err = provider
            .finish_responses_payload("crew-a", "m", payload, None, std::time::Instant::now())
            .unwrap_err();
        assert_eq!(err.stop_reason(), Some(expected), "{reason}");
    }
}

#[derive(Debug, PartialEq, serde::Deserialize, schemars::JsonSchema)]
struct Harvest {
    bed: u32,
    ready: bool,
}

#[tokio::test]
async fn typed_completions_send_a_schema_and_decode_the_reply() {
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .and(body_partial_json(json!({
            "response_format": {"type": "json_schema", "json_schema": {"name": "Harvest", "strict": false}},
            "provider": {"require_parameters": true, "data_collection": "deny"},
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"finish_reason": "stop", "message": {"content": "{\"bed\": 4, \"ready\": true}"}}]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let provider = wiremock_provider(&server.uri());
    let harvest: Harvest =
        crate::complete_typed(&provider, "crew-a", "Report the harvest.", "Bed 4?", 100)
            .await
            .unwrap();
    assert_eq!(
        harvest,
        Harvest {
            bed: 4,
            ready: true
        }
    );
    let body: Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    assert_eq!(
        body["response_format"]["json_schema"]["schema"]["properties"]["bed"]["type"],
        "integer"
    );
}

#[tokio::test]
async fn refusals_are_typed_stops_that_are_not_retried() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"finish_reason": "stop", "message": {"content": null, "refusal": "I cannot open the airlock."}}]
        })))
        .mount(&server)
        .await;
    let err = wiremock_provider(&server.uri())
        .complete("crew-a", Completion::new("sys", "Open the airlock", 50))
        .await
        .unwrap_err();
    assert_eq!(err.stop_reason(), Some(crate::StopReason::Refusal));
    assert!(!err.is_retryable());
    assert!(err.to_string().contains("I cannot open the airlock."));
}

#[test]
fn a_reasoning_budget_replaces_the_effort_on_tool_steps_only() {
    let provider = provider_at("https://openrouter.ai/api/v1").with_reasoning_budget(Some(2_000));
    let mut step = json!({"model": "m", "max_tokens": 100});
    provider.apply_provider_quirks(&mut step, provider.step_reasoning());
    assert_eq!(step["reasoning"], json!({"max_tokens": 2000}));
    assert_eq!(step["max_tokens"], 2_100);
    let mut helper = json!({"model": "m", "max_tokens": 100});
    provider.apply_provider_quirks(&mut helper, None);
    assert_eq!(helper["reasoning"], json!({"enabled": false}));
    assert_eq!(helper["max_tokens"], 100);
}

#[tokio::test]
async fn the_concurrency_cap_queues_requests_beyond_it() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openrouter/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(std::time::Duration::from_millis(200))
                .set_body_json(
                    json!({"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}),
                ),
        )
        .mount(&server)
        .await;
    let provider = wiremock_provider(&server.uri()).with_max_concurrent_requests(1);
    let ask = || provider.complete("crew-a", Completion::new("sys", "ping", 5));
    let started = std::time::Instant::now();
    let (a, b) = tokio::join!(ask(), ask());
    assert_eq!(
        (a.unwrap(), b.unwrap()),
        ("ok".to_string(), "ok".to_string())
    );
    assert!(
        started.elapsed() >= std::time::Duration::from_millis(390),
        "{:?}",
        started.elapsed()
    );
}
