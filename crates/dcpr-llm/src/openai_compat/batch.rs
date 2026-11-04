//! Batch API support (feature `batch`), on [`async_openai`]'s files and
//! batches clients.
//!
//! A batched request is built exactly as the live call would be, and each
//! reply goes through the same checks: usage is recorded, early stops and
//! refusals become errors, and tool calls are parsed. Batches run at a
//! discount, in up to 24 hours, on endpoints that implement the OpenAI Batch
//! API. OpenRouter does not.
//!
//! A client on the Responses API (the default on OpenAI, see
//! [`OpenAiCompat::with_responses_api`]) builds tool steps for
//! the `/v1/responses` batch endpoint, which OpenAI's gpt-6 models need for
//! function tools. One batch holds one endpoint, so keep those steps apart from
//! completions, which always use Chat Completions.

use super::{OpenAiCompat, interpret_responses_payload, interpret_tool_message};
use crate::{Caller, ChatStep, Completion, Error, Result, SystemPrompt, ToolTurn};
use async_openai::config::OpenAIConfig;
use async_openai::error::OpenAIError;
use async_openai::types::InputSource;
use async_openai::types::batches::{
    Batch, BatchCompletionWindow, BatchEndpoint, BatchRequest as CreateBatch,
    BatchStatus as WireStatus,
};
use async_openai::types::files::{CreateFileRequest, FileInput, FilePurpose};
use serde_json::{Value, json};
use std::time::Duration;

const ENDPOINT: &str = "/v1/chat/completions";
const RESPONSES_ENDPOINT: &str = "/v1/responses";

/// One request in a batch: a caller-chosen id and the request body.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    pub custom_id: String,
    body: Value,
    responses: bool,
}

impl BatchRequest {
    fn url(&self) -> &'static str {
        if self.responses {
            RESPONSES_ENDPOINT
        } else {
            ENDPOINT
        }
    }

    pub(super) fn line(&self) -> Value {
        json!({
            "custom_id": self.custom_id,
            "method": "POST",
            "url": self.url(),
            "body": self.body,
        })
    }
}

/// Where a batch is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BatchStatus {
    Validating,
    InProgress,
    Finalizing,
    Completed,
    Failed,
    Expired,
    Cancelling,
    Cancelled,
}

impl BatchStatus {
    /// The batch will not change again. Completed, expired and cancelled
    /// batches may still hold results.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Expired | Self::Cancelled
        )
    }
}

impl From<WireStatus> for BatchStatus {
    fn from(status: WireStatus) -> Self {
        match status {
            WireStatus::Validating => Self::Validating,
            WireStatus::InProgress => Self::InProgress,
            WireStatus::Finalizing => Self::Finalizing,
            WireStatus::Completed => Self::Completed,
            WireStatus::Failed => Self::Failed,
            WireStatus::Expired => Self::Expired,
            WireStatus::Cancelling => Self::Cancelling,
            WireStatus::Cancelled => Self::Cancelled,
        }
    }
}

/// A submitted batch.
#[derive(Debug, Clone)]
pub struct BatchJob {
    pub id: String,
    pub status: BatchStatus,
    pub total: u32,
    pub completed: u32,
    pub failed: u32,
    /// Why the batch itself failed, when it did.
    pub errors: Vec<String>,
    output_file_id: Option<String>,
    error_file_id: Option<String>,
}

impl From<Batch> for BatchJob {
    fn from(batch: Batch) -> Self {
        let counts = batch.request_counts.unwrap_or_default();
        Self {
            id: batch.id,
            status: batch.status.into(),
            total: counts.total,
            completed: counts.completed,
            failed: counts.failed,
            errors: batch
                .errors
                .map(|errors| {
                    errors
                        .data
                        .into_iter()
                        .map(|error| format!("{}: {}", error.code, error.message))
                        .collect()
                })
                .unwrap_or_default(),
            output_file_id: batch.output_file_id,
            error_file_id: batch.error_file_id,
        }
    }
}

/// One request's outcome.
#[derive(Debug)]
pub struct BatchItem {
    pub custom_id: String,
    pub result: Result<ChatStep>,
}

impl BatchItem {
    /// The reply text, for a request built with
    /// [`OpenAiCompat::batch_completion`].
    pub fn text(self) -> Result<String> {
        match self.result? {
            ChatStep::Text(text) => Ok(text.trim().to_string()),
            ChatStep::ToolCalls(_) => Err(Error::Protocol(format!(
                "batch item {} asked for tools on a completion",
                self.custom_id
            ))),
        }
    }
}

fn batch_error(error: OpenAIError) -> Error {
    match error {
        OpenAIError::ApiError(response) => Error::from_status(
            response.status_code.as_u16(),
            format!("batch API {response}"),
        ),
        OpenAIError::Reqwest(error) => match error.status() {
            Some(status) => Error::from_status(status.as_u16(), format!("batch API: {error}")),
            None => Error::transient(format!("batch API request failed: {error}")),
        },
        OpenAIError::JSONDeserialize(error, _) => {
            Error::Protocol(format!("batch API returned unexpected JSON: {error}"))
        }
        other => Error::hard(format!("batch API: {other}")),
    }
}

impl<C: Caller + ?Sized> OpenAiCompat<C> {
    fn batch_client(&self) -> async_openai::Client<OpenAIConfig> {
        let config = OpenAIConfig::new()
            .with_api_base(self.base_url.trim_end_matches('/'))
            .with_api_key(self.bearer.clone());
        // Files can be large: allow far longer than a chat call.
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .expect("reqwest client builds");
        async_openai::Client::build(http, config)
    }

    /// A one-shot completion to add to a batch, built as
    /// [`ChatModel::complete`](crate::ChatModel::complete) builds it.
    pub fn batch_completion(
        &self,
        caller: &C,
        custom_id: impl Into<String>,
        request: Completion<'_>,
    ) -> BatchRequest {
        BatchRequest {
            custom_id: custom_id.into(),
            body: self.completion_body(caller, &request),
            responses: false,
        }
    }

    /// A tool step to add to a batch, built as
    /// [`ChatModel::chat_step`](crate::ChatModel::chat_step) builds it: on
    /// the Responses API when the client uses it, else Chat Completions. A
    /// batch has no second attempt, so a failed reasoning step is not retried
    /// without reasoning.
    pub fn batch_step(
        &self,
        caller: &C,
        custom_id: impl Into<String>,
        system: SystemPrompt<'_>,
        history: &[ToolTurn],
        tools: &[Value],
    ) -> BatchRequest {
        let reasoning = self.chat_step_reasoning(history);
        let mut body = if self.uses_responses() {
            self.responses_base(caller, system, history, tools).body
        } else {
            self.chat_step_base(caller, system, history, tools).1
        };
        self.apply_provider_quirks(&mut body, reasoning);
        BatchRequest {
            custom_id: custom_id.into(),
            body,
            responses: self.uses_responses(),
        }
    }

    /// Upload `requests` and start a batch with a 24-hour window.
    pub async fn submit_batch(&self, requests: &[BatchRequest]) -> Result<BatchJob> {
        if requests.is_empty() {
            return Err(Error::hard("a batch needs at least one request"));
        }
        let responses = requests[0].responses;
        if requests
            .iter()
            .any(|request| request.responses != responses)
        {
            return Err(Error::hard(
                "a batch holds one endpoint: submit Responses tool steps and Chat Completions requests separately",
            ));
        }
        let mut seen = std::collections::HashSet::new();
        if let Some(duplicate) = requests.iter().find(|r| !seen.insert(&r.custom_id)) {
            return Err(Error::hard(format!(
                "batch custom_id {} is used twice",
                duplicate.custom_id
            )));
        }
        let mut jsonl = String::new();
        for request in requests {
            jsonl.push_str(&request.line().to_string());
            jsonl.push('\n');
        }
        let client = self.batch_client();
        let file = client
            .files()
            .create(CreateFileRequest {
                file: FileInput {
                    source: InputSource::VecU8 {
                        filename: "batch.jsonl".to_string(),
                        vec: jsonl.into_bytes(),
                    },
                },
                purpose: FilePurpose::Batch,
                expires_after: None,
            })
            .await
            .map_err(batch_error)?;
        let batch = client
            .batches()
            .create(CreateBatch {
                input_file_id: file.id,
                endpoint: if responses {
                    BatchEndpoint::V1Responses
                } else {
                    BatchEndpoint::V1ChatCompletions
                },
                completion_window: BatchCompletionWindow::W24H,
                metadata: None,
                output_expires_after: None,
            })
            .await
            .map_err(batch_error)?;
        Ok(batch.into())
    }

    pub async fn batch_status(&self, id: &str) -> Result<BatchJob> {
        let batch = self
            .batch_client()
            .batches()
            .retrieve(id)
            .await
            .map_err(batch_error)?;
        Ok(batch.into())
    }

    /// Ask the provider to stop a batch. Finished requests keep their results.
    pub async fn cancel_batch(&self, id: &str) -> Result<BatchJob> {
        let batch = self
            .batch_client()
            .batches()
            .cancel(id)
            .await
            .map_err(batch_error)?;
        Ok(batch.into())
    }

    /// Poll every `interval` until the batch is terminal.
    pub async fn wait_for_batch(&self, id: &str, interval: Duration) -> Result<BatchJob> {
        loop {
            let job = self.batch_status(id).await?;
            if job.status.is_terminal() {
                return Ok(job);
            }
            tokio::time::sleep(interval).await;
        }
    }

    /// Every result the batch has, successes and failures, in file order.
    /// Usage is recorded against `caller` as each reply is read. Batch
    /// discounts are not applied to the recorded usage.
    pub async fn batch_results(&self, caller: &C, job: &BatchJob) -> Result<Vec<BatchItem>> {
        let client = self.batch_client();
        let mut items = Vec::new();
        for file in [&job.output_file_id, &job.error_file_id]
            .into_iter()
            .flatten()
        {
            let bytes = client.files().content(file).await.map_err(batch_error)?;
            let text = std::str::from_utf8(&bytes)
                .map_err(|e| Error::Protocol(format!("batch file {file} is not UTF-8: {e}")))?;
            for line in text.lines().filter(|line| !line.trim().is_empty()) {
                items.push(self.batch_item(caller, line)?);
            }
        }
        Ok(items)
    }

    fn batch_item(&self, caller: &C, line: &str) -> Result<BatchItem> {
        let row: Value = serde_json::from_str(line)
            .map_err(|e| Error::Protocol(format!("batch result line is not JSON: {e}")))?;
        let custom_id = row["custom_id"].as_str().unwrap_or_default().to_string();
        let result = (|| {
            if let Some(error) = row["error"].as_object() {
                return Err(Error::hard(format!(
                    "batch request failed: {}",
                    error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown error")
                )));
            }
            let response = &row["response"];
            let status = response["status_code"].as_u64().unwrap_or(0) as u16;
            let body = response["body"].clone();
            if !(200..300).contains(&status) {
                let message = body["error"]["message"]
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| body.to_string());
                return Err(Error::from_status(
                    status,
                    format!("batch request {status}: {message}"),
                ));
            }
            let model = body["model"].as_str().unwrap_or(&self.model).to_string();
            if body["object"] == "response" || body.get("output").is_some() {
                let payload = self.finish_responses_payload(
                    caller,
                    &model,
                    body,
                    None,
                    std::time::Instant::now(),
                )?;
                return Ok(interpret_responses_payload(&payload));
            }
            let raw = body.to_string();
            let message = self.finish_chat_payload(
                caller,
                &model,
                body,
                None,
                std::time::Instant::now(),
                &raw,
            )?;
            Ok(interpret_tool_message(&message).0)
        })();
        Ok(BatchItem { custom_id, result })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Usage, UsageSink};
    use std::sync::{Arc, Mutex};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[derive(Default)]
    struct Recorded(Mutex<Vec<(String, String, Usage)>>);

    impl UsageSink<str> for Recorded {
        fn record(&self, caller: &str, model: &str, usage: Usage, _: Option<f64>) {
            self.0
                .lock()
                .unwrap()
                .push((caller.to_string(), model.to_string(), usage));
        }
    }

    fn batch(status: &str, extra: Value) -> Value {
        let mut batch = json!({
            "id": "batch_1",
            "object": "batch",
            "endpoint": ENDPOINT,
            "input_file_id": "file-in",
            "completion_window": "24h",
            "status": status,
            "created_at": 1,
            "request_counts": {"total": 3, "completed": 2, "failed": 1},
        });
        for (key, value) in extra.as_object().unwrap() {
            batch[key] = value.clone();
        }
        batch
    }

    fn reply(custom_id: &str, message: Value) -> String {
        json!({
            "id": format!("req_{custom_id}"),
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "request_id": "r",
                "body": {
                    "model": "orchard-model",
                    "choices": [{"finish_reason": "stop", "message": message}],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 5},
                },
            },
            "error": null,
        })
        .to_string()
    }

    #[tokio::test]
    async fn a_batch_is_submitted_polled_and_read_back() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/files"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "file-in", "object": "file", "bytes": 10, "created_at": 1,
                "filename": "batch.jsonl", "purpose": "batch",
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/batches"))
            .respond_with(ResponseTemplate::new(200).set_body_json(batch("validating", json!({}))))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/batches/batch_1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(batch(
                "completed",
                json!({"output_file_id": "file-out", "error_file_id": "file-err"}),
            )))
            .mount(&server)
            .await;
        let output = [
            reply(
                "pear",
                json!({"role": "assistant", "content": " Conference pears. "}),
            ),
            reply(
                "graft",
                json!({"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "graft_scion", "arguments": "{\"rootstock\":\"M9\"}"},
                }]}),
            ),
        ]
        .join("\n");
        Mock::given(method("GET"))
            .and(path("/v1/files/file-out/content"))
            .respond_with(ResponseTemplate::new(200).set_body_string(output))
            .mount(&server)
            .await;
        let failed = json!({
            "custom_id": "plum",
            "response": {"status_code": 400, "body": {"error": {"message": "bad plum"}}},
            "error": null,
        });
        Mock::given(method("GET"))
            .and(path("/v1/files/file-err/content"))
            .respond_with(ResponseTemplate::new(200).set_body_string(failed.to_string()))
            .mount(&server)
            .await;

        let usage = Arc::new(Recorded::default());
        let client: OpenAiCompat = OpenAiCompat::new(
            format!("{}/v1", server.uri()),
            "k",
            "orchard-model",
            usage.clone(),
        );
        let tools = [
            json!({"name": "graft_scion", "description": "Graft a scion.", "parameters": {"type": "object"}}),
        ];
        let requests = [
            client.batch_completion(
                "orchard",
                "pear",
                Completion::new("Name a pear.", "One.", 16),
            ),
            client.batch_step(
                "orchard",
                "graft",
                SystemPrompt::from("You tend an orchard."),
                &[ToolTurn::User("Graft onto M9.".into())],
                &tools,
            ),
            client.batch_completion(
                "orchard",
                "plum",
                Completion::new("Name a plum.", "One.", 16),
            ),
        ];
        let job = client.submit_batch(&requests).await.unwrap();
        assert_eq!(job.status, BatchStatus::Validating);

        let job = client
            .wait_for_batch(&job.id, Duration::from_millis(1))
            .await
            .unwrap();
        assert_eq!((job.total, job.completed, job.failed), (3, 2, 1));

        let mut items = client
            .batch_results("orchard", &job)
            .await
            .unwrap()
            .into_iter();
        let pear = items.next().unwrap();
        assert_eq!(pear.custom_id, "pear");
        assert_eq!(pear.text().unwrap(), "Conference pears.");
        let graft = items.next().unwrap();
        match graft.result.unwrap() {
            ChatStep::ToolCalls(calls) => {
                assert_eq!(calls[0].name, "graft_scion");
                assert_eq!(calls[0].arguments, json!({"rootstock": "M9"}));
            }
            other => panic!("expected tool calls, got {other:?}"),
        }
        let plum = items.next().unwrap();
        assert_eq!(plum.custom_id, "plum");
        let error = plum.result.unwrap_err();
        assert!(error.to_string().contains("bad plum"), "{error}");
        assert!(items.next().is_none());

        {
            let recorded = usage.0.lock().unwrap();
            assert_eq!(recorded.len(), 2);
            assert_eq!(recorded[0].0, "orchard");
            assert_eq!(recorded[0].1, "orchard-model");
            assert_eq!(recorded[0].2.input_tokens, 40);
        }

        let upload = server.received_requests().await.unwrap();
        let upload = String::from_utf8_lossy(&upload[0].body);
        assert_eq!(
            upload.matches("\"url\":\"/v1/chat/completions\"").count(),
            3
        );
        assert!(upload.contains("\"custom_id\":\"graft\""));
        assert!(upload.contains("graft_scion"));
    }

    #[tokio::test]
    async fn duplicate_ids_are_refused_before_upload() {
        let client: OpenAiCompat =
            OpenAiCompat::new("http://127.0.0.1:9/v1", "k", "m", Arc::new(crate::NoUsage));
        let request = client.batch_completion("orchard", "same", Completion::new("s", "u", 8));
        let error = client
            .submit_batch(&[request.clone(), request])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("used twice"), "{error}");
    }
}
