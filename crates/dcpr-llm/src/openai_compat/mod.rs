//! OpenAI-compatible client. Tool steps use Chat Completions by default, or
//! the Responses API on hosts that implement it
//! ([`OpenAiCompat::with_responses_api`]). Small structured completions always
//! use Chat Completions.
//! - **OpenAI**: `https://api.openai.com/v1`, `Authorization: Bearer <api key>`.
//!   Turn on the Responses API for tool steps: from gpt-5.5 on, Chat
//!   Completions refuses function tools while the model reasons.
//! - **OpenRouter**: `https://openrouter.ai/api/v1`, `Authorization: Bearer <key>`.
//! - **Gemini**: `https://generativelanguage.googleapis.com/v1beta/openai`,
//!   Chat Completions only.
//!
//! Every successful call reports its token usage to the configured
//! [`UsageSink`], tagged with the caller.

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use std::sync::Arc;

mod anthropic;
#[cfg(feature = "batch")]
mod batch;
mod openai;
use crate::{
    Caller, ChatModel, ChatStep, Completion, StopReason, SystemPrompt, TextStreamSink, ToolCall,
    ToolTurn, Usage, UsageSink,
};
use crate::{Error, Result};
#[cfg(feature = "batch")]
pub use batch::{BatchItem, BatchJob, BatchRequest, BatchStatus};

/// Whole interactive turns have their own shorter deadline. Do not repeatedly
/// abort and restart a progressing reasoning stream before that deadline.
/// Long-running background callers can legitimately use this budget.
const CHAT_STEP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(240);
const RESPONSES_MAX_EVENT_BYTES: usize = 1024 * 1024;
/// A streamed step that produces no event for this long has stalled. Healthy
/// steps finish well inside it (the slowest of ~1,600 measured steps took
/// 33s), and keep-alive comments are not events. Failing here leaves time for
/// the fallback chain to retry within the interactive turn deadline.
const RESPONSES_STREAM_STALL: std::time::Duration = std::time::Duration::from_secs(60);
/// Reasoning items kept for echoing back on the next tool step. An interactive
/// turn ends well inside the lifetime; the entry cap bounds memory.
const REASONING_ITEM_LIFETIME: std::time::Duration = std::time::Duration::from_secs(600);
const REASONING_ITEM_ENTRIES: usize = 512;

/// The reasoning that led to one batch of tool calls. OpenRouter recommends
/// echoing it before those calls on the next step so the model continues its
/// reasoning instead of starting again. Requests are stateless (`store: false`),
/// so the items are kept here, keyed by caller scope and the batch's first call id.
/// https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning
struct ReasoningEntry {
    scope: String,
    first_call_id: String,
    items: Vec<Value>,
    /// OpenAI's own `fc_…` item id for each call id in the batch. OpenAI
    /// rejects a replayed `function_call` whose `id` is anything else.
    item_ids: Vec<(String, String)>,
    stored: std::time::Instant,
}
/// Provider data attached to one tool call that must be sent back with it,
/// such as Gemini's `extra_content.google.thought_signature`. Gemini 3 rejects
/// a replayed call without its signature.
struct CallExtraEntry {
    scope: String,
    call_id: String,
    extra: Value,
    stored: std::time::Instant,
}

/// Gemini's documented placeholder signature for a replayed function call
/// whose real signature is no longer known, such as history restored from
/// storage. https://ai.google.dev/gemini-api/docs/thought-signatures
const GEMINI_SKIP_SIGNATURE: &str = "skip_thought_signature_validator";

/// Output headroom added to the ceiling when a request reasons, so reasoning
/// cannot consume the reply.
const REASONING_HEADROOM: u64 = 24_000;

/// Output headroom for a small completion's low-effort reasoning on OpenAI.
const HELPER_REASONING_HEADROOM: u64 = 4_096;

/// Default total time for a small completion.
const COMPLETION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Internal marker: reasoning is set by token budget, not effort.
const REASONING_BY_BUDGET: &str = "budget";

/// The service behind the base URL, for its request rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Host {
    OpenRouter,
    OpenAi,
    Gemini,
    Other,
}

impl Host {
    fn detect(base_url: &str) -> Self {
        if base_url.contains("openrouter") {
            Self::OpenRouter
        } else if base_url.contains("api.openai.com") {
            Self::OpenAi
        } else if base_url.contains("generativelanguage.googleapis.com") {
            Self::Gemini
        } else {
            Self::Other
        }
    }
}
const RESPONSES_TEXT_FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_millis(32);

/// Parse a `Retry-After` header (seconds form) into a bounded number of seconds.
/// Providers overwhelmingly send the delta-seconds form on a 429/529; the
/// HTTP-date form is rare here and deliberately ignored (we fall back to jitter).
/// Capped at 5 minutes so a hostile/absurd value can't wedge the retry loop.
pub fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|secs| secs.min(300))
}

/// OpenRouter's id for one generation, returned on every endpoint. Failures
/// carry it so a stalled or failed request can be found in OpenRouter's logs.
fn generation_id(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get("x-generation-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn tag_generation(error: Error, generation: Option<&str>) -> Error {
    error.with_generation(generation)
}

/// OpenRouter endpoint ordering. Throughput is the default; price ordering is
/// available for explicit provider comparisons.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OpenRouterProviderSort {
    #[default]
    Throughput,
    Price,
}

impl OpenRouterProviderSort {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Throughput => "throughput",
            Self::Price => "price",
        }
    }
}

/// Serialized request sizes for one Responses step. Bytes, not tokens, and
/// never request content. Reported through
/// [`OpenAiCompat::with_request_observer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestSize {
    pub instructions_bytes: usize,
    pub history_bytes: usize,
    pub initial_tools_bytes: usize,
    pub deferred_tools_bytes: usize,
    pub initially_loaded_tools: usize,
    pub deferred_tools: usize,
    pub tool_search_enabled: bool,
}

/// A Responses tool-step body and its tool catalogue's shape.
struct ResponsesBase {
    body: Value,
    deferred_names: std::collections::HashSet<String>,
    initially_loaded_tools: usize,
    deferred_tools: usize,
}

type RequestObserver = Arc<dyn Fn(RequestSize) + Send + Sync>;

/// An OpenAI-compatible chat client, generic over the [`Caller`] type its
/// [`UsageSink`] attributes cost to.
pub struct OpenAiCompat<C: ?Sized = str> {
    client: reqwest::Client,
    base_url: String,
    bearer: String,
    model: String,
    /// Model used for turns that carry images (e.g. a multimodal OpenRouter
    /// model alongside a cheaper text-only default). None => use `model`.
    vision_model: Option<String>,
    /// OpenRouter app attribution: `HTTP-Referer` and `X-OpenRouter-Title`.
    referer: Option<String>,
    app_title: Option<String>,
    /// Prefix hashed with the caller scope into OpenRouter session and end-user
    /// identifiers, so two applications sharing an account never collide.
    session_namespace: String,
    request_observer: Option<RequestObserver>,
    /// OpenRouter fallback models, in priority order, tried after `model` if it
    /// errors (rate-limit, downtime, a model being deprecated). Sent as the
    /// `models` array; OpenRouter does the failover server-side and bills for
    /// whichever model actually ran. Empty (the default) = no fallback. Only
    /// honoured on the OpenRouter endpoint (OpenAI proper rejects the field).
    fallback_models: Vec<String>,
    /// OpenRouter reasoning effort (e.g. "high", "max"). None normally means
    /// reasoning disabled; models which require reasoning use supported low effort.
    reasoning_effort: Option<String>,
    /// OpenRouter reasoning token budget. Takes precedence over the effort.
    reasoning_budget: Option<u32>,
    openrouter_provider_sort: OpenRouterProviderSort,
    openrouter_provider_ignore: Vec<String>,
    /// Use the Responses API for tool-using turns, on hosts that implement
    /// it (OpenRouter and OpenAI).
    responses_api: Option<bool>,
    stream_stall: std::time::Duration,
    reasoning_items: Arc<std::sync::Mutex<std::collections::VecDeque<ReasoningEntry>>>,
    call_extras: Arc<std::sync::Mutex<std::collections::VecDeque<CallExtraEntry>>>,
    /// OpenRouter models found to reject `reasoning: {enabled: false}`.
    mandatory_reasoning: Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
    /// Caps requests in flight from this client and its clones.
    limiter: Option<Arc<tokio::sync::Semaphore>>,
    completion_timeout: std::time::Duration,
    usage: Arc<dyn UsageSink<C>>,
}

impl<C: ?Sized> Clone for OpenAiCompat<C> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            base_url: self.base_url.clone(),
            bearer: self.bearer.clone(),
            model: self.model.clone(),
            vision_model: self.vision_model.clone(),
            referer: self.referer.clone(),
            app_title: self.app_title.clone(),
            session_namespace: self.session_namespace.clone(),
            request_observer: self.request_observer.clone(),
            fallback_models: self.fallback_models.clone(),
            reasoning_effort: self.reasoning_effort.clone(),
            reasoning_budget: self.reasoning_budget,
            openrouter_provider_sort: self.openrouter_provider_sort,
            openrouter_provider_ignore: self.openrouter_provider_ignore.clone(),
            responses_api: self.responses_api,
            stream_stall: self.stream_stall,
            reasoning_items: self.reasoning_items.clone(),
            call_extras: self.call_extras.clone(),
            mandatory_reasoning: self.mandatory_reasoning.clone(),
            limiter: self.limiter.clone(),
            completion_timeout: self.completion_timeout,
            usage: self.usage.clone(),
        }
    }
}

impl<C: Caller + ?Sized> OpenAiCompat<C> {
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let mut req = self
            .client
            .post(format!("{}/{path}", self.base_url.trim_end_matches('/')))
            .header("authorization", format!("Bearer {}", self.bearer))
            .header("content-type", "application/json");
        if let Some(referer) = &self.referer {
            req = req.header("HTTP-Referer", referer);
        }
        if let Some(title) = &self.app_title {
            req = req.header("X-OpenRouter-Title", title);
        }
        req
    }

    /// `end_user_field` is `safety_identifier` on the Responses API and `user`
    /// on Chat Completions.
    fn host(&self) -> Host {
        Host::detect(&self.base_url)
    }

    fn is_openrouter(&self) -> bool {
        self.host() == Host::OpenRouter
    }

    /// The name used in error messages for the Responses transport.
    fn responses_label(&self) -> &'static str {
        if self.is_openrouter() {
            "OpenRouter Responses"
        } else {
            "Responses API"
        }
    }

    fn apply_openrouter_session(&self, body: &mut Value, scope: &str, end_user_field: &str) {
        if self.host() == Host::OpenAi {
            self.apply_openai_identity(body, scope);
            return;
        }
        if !self.is_openrouter() {
            return;
        }
        // OpenRouter uses this opaque key to keep a conversation on the same
        // upstream endpoint from its first successful request. That improves
        // cache hits across agent tool rounds without disclosing the caller's
        // identifiers to the routing layer.
        use sha2::{Digest, Sha256};
        let namespace = &self.session_namespace;
        let digest = Sha256::digest(format!("{namespace}-openrouter-session:{scope}").as_bytes());
        let session_id: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        body["session_id"] = json!(session_id);
        // Without a per-user identifier, a provider's abuse block applies to the
        // whole OpenRouter account. A separate pseudonymous hash keeps one
        // caller's block from silencing every other caller.
        let digest = Sha256::digest(format!("{namespace}-openrouter-user:{scope}").as_bytes());
        let end_user: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        body[end_user_field] = json!(end_user);
    }

    /// OpenAI's per-caller fields, hashed like OpenRouter's: a
    /// `safety_identifier`, so an abuse block lands on one caller and not the
    /// whole organisation, and a `prompt_cache_key`, so one caller's requests
    /// reach the same cache.
    fn apply_openai_identity(&self, body: &mut Value, scope: &str) {
        use sha2::{Digest, Sha256};
        let namespace = &self.session_namespace;
        let hash = |kind: &str| -> String {
            Sha256::digest(format!("{namespace}-openai-{kind}:{scope}").as_bytes())
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect()
        };
        body["safety_identifier"] = json!(hash("user"));
        body["prompt_cache_key"] = json!(hash("cache"));
    }

    fn remember_reasoning(&self, scope: &str, payload: &Value, step: &ChatStep) {
        let ChatStep::ToolCalls(calls) = step else {
            return;
        };
        let Some(first) = calls.first() else { return };
        let output = payload["output"].as_array().into_iter().flatten();
        let items = output
            .clone()
            .filter(|item| item["type"] == "reasoning")
            .cloned()
            .collect::<Vec<_>>();
        // OpenRouter accepts the call id as the item id; OpenAI needs its own.
        let item_ids = if self.is_openrouter() {
            Vec::new()
        } else {
            output
                .filter(|item| item["type"] == "function_call")
                .filter_map(|item| {
                    Some((
                        item["call_id"].as_str()?.to_string(),
                        item["id"].as_str()?.to_string(),
                    ))
                })
                .collect()
        };
        if items.is_empty() && item_ids.is_empty() {
            return;
        }
        let Ok(mut entries) = self.reasoning_items.lock() else {
            return;
        };
        entries.retain(|entry| entry.stored.elapsed() < REASONING_ITEM_LIFETIME);
        while entries.len() >= REASONING_ITEM_ENTRIES {
            entries.pop_front();
        }
        entries.push_back(ReasoningEntry {
            scope: scope.to_string(),
            first_call_id: first.id.clone(),
            items,
            item_ids,
            stored: std::time::Instant::now(),
        });
    }

    /// Insert remembered reasoning immediately before the calls it produced.
    /// Off OpenRouter, a replayed call carries OpenAI's remembered item id, or
    /// no `id` at all, since OpenAI rejects one that is not its own.
    fn restore_reasoning(&self, scope: &str, input: Vec<Value>) -> Vec<Value> {
        let Ok(entries) = self.reasoning_items.lock() else {
            return input;
        };
        let live = |entry: &&ReasoningEntry| {
            entry.scope == scope && entry.stored.elapsed() < REASONING_ITEM_LIFETIME
        };
        let mut restored = Vec::with_capacity(input.len());
        for mut item in input {
            if item["type"] == "function_call" {
                let call_id = item["call_id"].as_str().unwrap_or_default().to_string();
                if let Some(entry) = entries
                    .iter()
                    .filter(live)
                    .find(|entry| entry.first_call_id == call_id)
                {
                    restored.extend(entry.items.iter().cloned());
                }
                if !self.is_openrouter() {
                    let item_id = entries.iter().filter(live).find_map(|entry| {
                        entry
                            .item_ids
                            .iter()
                            .find(|(call, _)| *call == call_id)
                            .map(|(_, id)| id.clone())
                    });
                    match item_id {
                        Some(id) => item["id"] = json!(id),
                        None => {
                            item.as_object_mut().map(|item| item.remove("id"));
                        }
                    }
                }
            }
            restored.push(item);
        }
        restored
    }

    /// Keep each tool call's provider data (Gemini's thought signature) to
    /// send back when the call is replayed.
    fn remember_call_extras(&self, scope: &str, message: &Value) {
        let extras = message["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|call| call["extra_content"].is_object())
            .filter_map(|call| Some((call["id"].as_str()?, call["extra_content"].clone())))
            .collect::<Vec<_>>();
        if extras.is_empty() {
            return;
        }
        let Ok(mut entries) = self.call_extras.lock() else {
            return;
        };
        entries.retain(|entry| entry.stored.elapsed() < REASONING_ITEM_LIFETIME);
        for (call_id, extra) in extras {
            while entries.len() >= REASONING_ITEM_ENTRIES {
                entries.pop_front();
            }
            entries.push_back(CallExtraEntry {
                scope: scope.to_string(),
                call_id: call_id.to_string(),
                extra,
                stored: std::time::Instant::now(),
            });
        }
    }

    /// The provider data to send back with a replayed call. Gemini requires a
    /// signature on the first call of each batch; when the real one is gone,
    /// its documented placeholder is used.
    fn call_extra(&self, scope: &str, call_id: &str, first_in_batch: bool) -> Option<Value> {
        let remembered = self.call_extras.lock().ok().and_then(|entries| {
            entries
                .iter()
                .find(|entry| {
                    entry.scope == scope
                        && entry.call_id == call_id
                        && entry.stored.elapsed() < REASONING_ITEM_LIFETIME
                })
                .map(|entry| entry.extra.clone())
        });
        remembered.or_else(|| {
            (self.host() == Host::Gemini && first_in_batch)
                .then(|| json!({"google": {"thought_signature": GEMINI_SKIP_SIGNATURE}}))
        })
    }

    pub fn new(
        base_url: impl Into<String>,
        bearer: impl Into<String>,
        model: impl Into<String>,
        usage: Arc<dyn UsageSink<C>>,
    ) -> Self {
        let client = reqwest::Client::builder()
            // Default total timeout for small completions (classification,
            // extraction, max_tokens 12-256). These should never take 30s.
            // chat_step overrides this per-request with a far larger budget
            // (see CHAT_STEP_TIMEOUT), because a 4k+-token generation on an
            // OpenRouter-served model routinely runs past 30s and killing it
            // just retries into the same wall.
            .timeout(std::time::Duration::from_secs(30))
            // A stalled CONNECT is a real failure worth failing fast on, even on
            // the long chat_step budget — separate it from the body-read timeout.
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("reqwest client builds");
        Self {
            client,
            base_url: base_url.into(),
            bearer: bearer.into(),
            model: model.into(),
            vision_model: None,
            referer: None,
            app_title: None,
            session_namespace: "dcpr-llm".to_string(),
            request_observer: None,
            fallback_models: Vec::new(),
            reasoning_effort: None,
            reasoning_budget: None,
            openrouter_provider_sort: OpenRouterProviderSort::default(),
            openrouter_provider_ignore: Vec::new(),
            responses_api: None,
            stream_stall: RESPONSES_STREAM_STALL,
            reasoning_items: Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new())),
            call_extras: Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new())),
            mandatory_reasoning: Arc::new(std::sync::Mutex::new(Default::default())),
            limiter: None,
            completion_timeout: COMPLETION_TIMEOUT,
            usage,
        }
    }

    /// Set OpenRouter fallback models (priority order). No-op effect off the
    /// OpenRouter endpoint. The primary `model` is always tried first; these
    /// follow it.
    pub fn with_fallback_models(mut self, models: Vec<String>) -> Self {
        if !models.is_empty() && !self.is_openrouter() {
            tracing::warn!(
                base_url = %self.base_url,
                "fallback models are only honoured on OpenRouter; use dcpr_llm::Fallback for client-side failover"
            );
        }
        self.fallback_models = models;
        self
    }

    /// Enable OpenRouter reasoning at the given effort for `chat_step`.
    /// Small deterministic calls normally disable it, but models which reject
    /// no-reasoning requests use supported low effort instead.
    /// Enable OpenRouter reasoning with an exact token budget for tool steps
    /// (`reasoning.max_tokens`) instead of an effort level. The output ceiling
    /// grows by the budget, so reasoning cannot consume the reply.
    pub fn with_reasoning_budget(mut self, tokens: Option<u32>) -> Self {
        self.reasoning_budget = tokens;
        self
    }

    /// The reasoning setting for a tool step: the effort, or a marker that
    /// the budget applies.
    fn step_reasoning(&self) -> Option<&str> {
        self.reasoning_effort
            .as_deref()
            .or(self.reasoning_budget.map(|_| REASONING_BY_BUDGET))
    }

    pub fn with_reasoning_effort(mut self, effort: Option<String>) -> Self {
        self.reasoning_effort = effort;
        self
    }

    /// Select OpenRouter's endpoint ordering without changing model, reasoning,
    /// or the mandatory privacy policy. Has no effect on other endpoints.
    pub fn with_openrouter_provider_sort(mut self, sort: OpenRouterProviderSort) -> Self {
        self.openrouter_provider_sort = sort;
        self
    }

    /// Exclude explicitly configured upstream providers without relaxing privacy.
    pub fn with_openrouter_provider_ignore(mut self, providers: Vec<String>) -> Self {
        self.openrouter_provider_ignore = providers;
        self
    }

    /// How long a Responses stream may go without an event before it fails as
    /// retryable. Defaults to 60 seconds.
    pub fn with_stream_stall(mut self, stall: std::time::Duration) -> Self {
        self.stream_stall = stall;
        self
    }

    /// Allow at most `max` requests in flight across this client and its
    /// clones. Further calls wait for a slot. Use it to stay under a
    /// provider's concurrency limit instead of meeting it as 429s.
    pub fn with_max_concurrent_requests(mut self, max: usize) -> Self {
        self.limiter = Some(Arc::new(tokio::sync::Semaphore::new(max.max(1))));
        self
    }

    /// Total time allowed for one [`complete`](ChatModel::complete) call.
    /// Defaults to 30 seconds. Gemini under load has needed 15 to 20 seconds
    /// for a one-word answer.
    pub fn with_completion_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.completion_timeout = timeout;
        self
    }

    async fn permit(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        match &self.limiter {
            Some(limiter) => limiter.clone().acquire_owned().await.ok(),
            None => None,
        }
    }

    /// Use the Responses API for tool steps (`true`) or Chat Completions
    /// (`false`). By default OpenAI's own API uses Responses, since from
    /// gpt-5.5 on its Chat Completions refuses function tools while a model
    /// reasons, and every other host uses Chat Completions.
    pub fn with_responses_api(mut self, enabled: bool) -> Self {
        self.responses_api = Some(enabled);
        self
    }

    /// Whether tool steps use the Responses API.
    fn uses_responses(&self) -> bool {
        self.responses_api
            .unwrap_or_else(|| self.host() == Host::OpenAi)
    }

    /// Send OpenRouter app attribution as `HTTP-Referer`.
    pub fn with_referer(mut self, referer: impl Into<String>) -> Self {
        self.referer = Some(referer.into());
        self
    }

    /// Send OpenRouter app attribution as `X-OpenRouter-Title`.
    pub fn with_app_title(mut self, title: impl Into<String>) -> Self {
        self.app_title = Some(title.into());
        self
    }

    /// Namespace hashed into OpenRouter session and end-user identifiers.
    /// Defaults to `dcpr-llm`. Changing it moves every caller to a new session.
    pub fn with_session_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.session_namespace = namespace.into();
        self
    }

    /// Observe the serialized size of each Responses request.
    pub fn with_request_observer(
        mut self,
        observer: impl Fn(RequestSize) + Send + Sync + 'static,
    ) -> Self {
        self.request_observer = Some(Arc::new(observer));
        self
    }

    fn observe_request_size(&self, body: &Value) {
        if let Some(observer) = &self.request_observer {
            observer(request_size(body));
        }
    }

    fn anthropic_policy(&self, model: &str) -> anthropic::RequestPolicy {
        if self.is_openrouter() {
            anthropic::RequestPolicy::for_models(model, &self.fallback_models)
        } else {
            anthropic::RequestPolicy::default()
        }
    }

    /// OpenRouter turns reasoning ON by default for reasoning-capable models.
    /// The reasoning stream then consumes the token
    /// budget and `message.content` comes back empty — and this provider reads
    /// only `content`. So reasoning is always EXPLICIT here: `reasoning` names
    /// the effort for THIS request (chat_step passes the configured effort;
    /// the small deterministic calls and the low-cost retry pass None). Models
    /// known to require reasoning receive supported low effort on that path. OpenAI
    /// proper rejects unknown parameters, so the field is sent to OpenRouter
    /// only.
    fn apply_provider_quirks(&self, body: &mut Value, reasoning: Option<&str>) {
        if self.host() == Host::OpenAi {
            self.apply_openai_quirks(body, reasoning);
            return;
        }
        if self.is_openrouter() {
            let model = body["model"].as_str().unwrap_or(&self.model);
            let mandatory_reasoning = reasoning_required(model)
                || openai::reasoning_is_mandatory(model)
                || self
                    .mandatory_reasoning
                    .lock()
                    .is_ok_and(|models| models.contains(model));
            self.anthropic_policy(model).apply_parameters(body);
            // GLM Flash advertises low/high/max, not the gateway's minimal
            // level. Use its lowest supported effort for small calls/retries.
            let effective_reasoning = if mandatory_reasoning {
                Some(match reasoning {
                    None | Some("minimal") => "low",
                    Some(effort) => effort,
                })
            } else {
                reasoning
            };
            match effective_reasoning {
                Some(effort) => {
                    // A budget applies only to the step that asked for
                    // reasoning; a mandatory-reasoning helper keeps its effort.
                    let budget = self.reasoning_budget.filter(|_| reasoning.is_some());
                    let headroom = match budget {
                        Some(tokens) => {
                            body["reasoning"] = json!({ "max_tokens": tokens });
                            u64::from(tokens)
                        }
                        None => {
                            let effort = if effort == REASONING_BY_BUDGET {
                                "medium"
                            } else {
                                effort
                            };
                            body["reasoning"] = json!({ "effort": effort });
                            REASONING_HEADROOM
                        }
                    };
                    let limit_key = if body.get("max_output_tokens").is_some() {
                        "max_output_tokens"
                    } else {
                        "max_tokens"
                    };
                    let mt = body[limit_key].as_u64().unwrap_or(0);
                    body[limit_key] = json!(mt + headroom);
                }
                None => body["reasoning"] = json!({ "enabled": false }),
            }
            // Privacy floor: route only to hosts that do not train on prompts.
            // This backs up the account-level setting and is enforced on every
            // request.
            // Per-request `zdr` is deliberately NOT set: no Anthropic endpoint
            // qualifies, so it excluded BYOK Haiku and sent every turn to the
            // fallbacks on OpenRouter credit.
            // `sort: throughput` disables OpenRouter's price-weighted load
            // balancing and bypasses Auto Exacto. It is the default because
            // default routing favours cheaper quantized endpoints, which gave
            // measurably worse tool-calling and slower replies in agent tests.
            // Errors still fall back to the next provider in sorted order.
            // https://openrouter.ai/docs/guides/routing/auto-exacto
            body["provider"] = json!({
                "data_collection": "deny",
                "sort": self.openrouter_provider_sort.as_str(),
            });
            if !self.openrouter_provider_ignore.is_empty() {
                body["provider"]["ignore"] = json!(self.openrouter_provider_ignore);
            }
            // Fallback models: try `model` first, then each fallback in order, so
            // a deprecated/down/rate-limited primary doesn't take us offline.
            // OpenRouter does the failover server-side and bills the model that
            // actually ran (returned in the response's `model`).
            if !self.fallback_models.is_empty() {
                let primary = body["model"].as_str().unwrap_or(&self.model).to_string();
                let mut models = vec![primary];
                models.extend(self.fallback_models.iter().cloned());
                body["models"] = json!(models);
            }
        }
    }

    /// OpenAI's own API. Every model takes `max_completion_tokens`; reasoning
    /// models also drop `temperature`, take the effort as `reasoning_effort`
    /// (Chat Completions) or `reasoning.effort` (Responses), and get output
    /// headroom for their reasoning, which they do even at their default
    /// effort. `None` leaves the model's default effort. OpenAI has no
    /// reasoning token budget, so a budget asks for medium effort and becomes
    /// the headroom. Non-reasoning models reject `reasoning_effort`, so a
    /// configured effort is not sent to them.
    fn apply_openai_quirks(&self, body: &mut Value, reasoning: Option<&str>) {
        let Some(object) = body.as_object_mut() else {
            return;
        };
        if let Some(limit) = object.remove("max_tokens") {
            object.insert("max_completion_tokens".into(), limit);
        }
        let model = object
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(&self.model);
        if !openai::is_reasoning_model(model) {
            return;
        }
        object.remove("temperature");
        let responses = object.contains_key("input");
        if let Some(effort) = reasoning {
            let effort = if effort == REASONING_BY_BUDGET {
                "medium"
            } else {
                effort
            };
            if responses {
                object.insert("reasoning".into(), json!({ "effort": effort }));
            } else {
                object.insert("reasoning_effort".into(), json!(effort));
            }
        }
        let headroom = match (reasoning, self.reasoning_budget) {
            (Some(_), Some(budget)) => u64::from(budget),
            _ => REASONING_HEADROOM,
        };
        let limit_key = if responses {
            "max_output_tokens"
        } else {
            "max_completion_tokens"
        };
        if let Some(limit) = object.get(limit_key).and_then(Value::as_u64) {
            object.insert(limit_key.into(), json!(limit + headroom));
        }
    }

    /// Whether this request reasons on OpenAI and so can return encrypted
    /// reasoning to echo on the next step.
    fn openai_reasons(&self, model: &str) -> bool {
        self.host() == Host::OpenAi && openai::is_reasoning_model(model)
    }

    /// Remember an OpenRouter model that refused to run without reasoning, so
    /// its later small calls use low effort instead of failing again.
    fn learn_mandatory_reasoning(&self, model: &str, error: &Error) -> bool {
        let mandatory = self.is_openrouter()
            && error
                .to_string()
                .contains("Reasoning is mandatory for this endpoint");
        if mandatory && let Ok(mut models) = self.mandatory_reasoning.lock() {
            tracing::warn!(model = %model, "model requires reasoning; small calls will use low effort");
            models.insert(model.to_string());
        }
        mandatory
    }

    /// Route image-bearing turns to a different (multimodal) model.
    pub fn with_vision_model(mut self, model: Option<String>) -> Self {
        self.vision_model = model;
        self
    }

    /// The model for this request: the vision model when any turn carries an
    /// image and one is configured, else the default.
    fn model_for(&self, history: &[ToolTurn]) -> &str {
        let has_images = history
            .iter()
            .any(|t| matches!(t, ToolTurn::UserWithImages(_, imgs) if !imgs.is_empty()));
        match (&self.vision_model, has_images) {
            (Some(v), true) => v,
            _ => &self.model,
        }
    }

    /// One chat-completions POST for `chat_step`: status and embedded-error
    /// mapping, usage recording, truncation WARNs. Returns the response
    /// `message` object. Split out so a reasoning-enabled step can re-send the
    /// same turn without reasoning when the first attempt fails or comes back
    /// empty.
    async fn chat_completions_step_send(
        &self,
        caller: &C,
        model: &str,
        body: &Value,
        sink: Option<&dyn TextStreamSink>,
    ) -> Result<Value> {
        let _permit = self.permit().await;
        if let Some(sink) = sink {
            return self
                .chat_completions_step_stream(caller, model, body, sink)
                .await;
        }
        let req = self.post("chat/completions");
        tracing::debug!(caller = %caller, model = %model, "openai-compat tools request");
        let started = std::time::Instant::now();
        // The agent bounds the complete interactive turn separately.
        let resp = req
            .timeout(CHAT_STEP_TIMEOUT)
            .json(body)
            .send()
            .await
            .map_err(|e| {
                tracing::warn!(
                    caller = %caller,
                    model = %model,
                    latency_ms = started.elapsed().as_millis() as u64,
                    "openai-compat tools request failed: {e}"
                );
                Error::transient(format!("openai-compat tools request failed: {e}"))
            })?;
        // Status FIRST, then body: a proxy/gateway 5xx (or some 4xx) often returns
        // HTML/plain text, which parsed as JSON would surface as a status-less
        // "bad json" transient and mis-retry a hard failure. Read the Retry-After
        // off the response headers before consuming the body so a 429/529 backs
        // off exactly as long as the provider asked.
        let status = resp.status();
        let retry_after = parse_retry_after(resp.headers());
        let generation = generation_id(resp.headers());
        let raw = resp
            .text()
            .await
            .map_err(|e| Error::transient(format!("openai-compat tools read body: {e}")))?;
        if !status.is_success() {
            let msg = http_error_message(&raw);
            tracing::warn!(
                caller = %caller,
                model = %model,
                status = status.as_u16(),
                latency_ms = started.elapsed().as_millis() as u64,
                "openai-compat tools call failed: {msg}"
            );
            return Err(tag_generation(
                Error::from_status_with_retry_after(
                    status.as_u16(),
                    format!("openai-compat {status}: {msg}"),
                    retry_after,
                ),
                generation.as_deref(),
            ));
        }
        let payload: Value = serde_json::from_str(&raw)
            .map_err(|e| Error::transient(format!("openai-compat tools bad json: {e}")))?;
        self.finish_chat_payload(caller, model, payload, retry_after, started, &raw)
            .map_err(|error| tag_generation(error, generation.as_deref()))
    }

    /// Checks shared by streamed and buffered Chat Completions steps:
    /// embedded errors, usage, early stops, and the reply message.
    fn finish_chat_payload(
        &self,
        caller: &C,
        model: &str,
        payload: Value,
        retry_after: Option<u64>,
        started: std::time::Instant,
        raw: &str,
    ) -> Result<Value> {
        // OpenRouter can wrap an upstream failure in an HTTP 200 whose body is
        // just `{"error": {...}}` with no choices (seen live: "temporarily
        // rate-limited upstream", code 429). Map it by its embedded code so a
        // wrapped 429/5xx is retried with backoff, never hard-failed.
        if let Some(e) = embedded_error(&payload, retry_after) {
            tracing::warn!(
                caller = %caller, model = %model,
                "openai-compat 200 OK with embedded error: {e}"
            );
            return Err(e);
        }
        if let Some(u) = payload.get("usage") {
            // Attribute to the model that actually ran (may be a fallback).
            let ran = payload["model"].as_str().unwrap_or(model);
            // Cached prompt tokens: OpenAI/OpenRouter report them under
            // prompt_tokens_details.cached_tokens; DeepSeek uses
            // prompt_cache_hit_tokens. Count them SEPARATELY and subtract from
            // input so the discounted cache rate applies without double counting.
            let prompt = u["prompt_tokens"].as_u64().unwrap_or(0);
            let cached = u["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .or_else(|| u["prompt_cache_hit_tokens"].as_u64())
                .unwrap_or(0)
                .min(prompt);
            // OpenRouter's reported cost, including any BYOK upstream charge,
            // is recorded as truth over the table.
            let provider_cost = crate::openrouter_cost_cents(u);
            self.usage.record(
                caller,
                ran,
                Usage {
                    input_tokens: prompt - cached,
                    output_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
                    cache_read_tokens: cached,
                    cache_write_tokens: u["prompt_tokens_details"]["cache_write_tokens"]
                        .as_u64()
                        .unwrap_or(0)
                        .min(prompt - cached),
                },
                provider_cost,
            );
        }

        // A `length`/content-filter finish is not a usable answer. Returning its
        // partial text produced fluent-looking replies that silently dropped the
        // final clause of compound requests. Surface it as retryable so chat_step
        // can retry a reasoning-heavy call without reasoning and provider fallback
        // can try another route when reasoning is already off.
        match payload["choices"][0]["finish_reason"].as_str() {
            Some("length") => {
                return Err(Error::stopped(
                    StopReason::MaxTokens,
                    "openai-compat reply hit the max_tokens ceiling; partial reply discarded",
                ));
            }
            Some("content_filter" | "content-filter") => {
                return Err(Error::stopped(
                    StopReason::ContentFilter,
                    "openai-compat reply stopped on a content filter; partial reply discarded",
                ));
            }
            _ => {}
        }
        if let Some(refusal) = refusal_of(&payload["choices"][0]["message"]) {
            return Err(Error::stopped(StopReason::Refusal, refusal));
        }

        let msg = payload
            .get("choices")
            .and_then(|c| c.get(0))
            .map(|c| c["message"].clone())
            .ok_or_else(|| {
                // Keep a body snippet: a bare "no choices" hides the real cause,
                // such as the embedded-429 shape.
                let snippet: String = raw.chars().take(300).collect();
                tracing::warn!(
                    caller = %caller, model = %model, body = %snippet,
                    "openai-compat 200 OK but response had no choices (API change?)"
                );
                Error::Protocol(format!("openai-compat: response had no choices: {snippet}"))
            })?;
        tracing::debug!(
            caller = %caller,
            requested_model = %model,
            ran_model = payload["model"].as_str().unwrap_or(model),
            latency_ms = started.elapsed().as_millis() as u64,
            in_tokens = payload["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
            cached_tokens = payload["usage"]["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0),
            cache_write_tokens = payload["usage"]["prompt_tokens_details"]["cache_write_tokens"]
                .as_u64()
                .unwrap_or(0),
            out_tokens = payload["usage"]["completion_tokens"].as_u64().unwrap_or(0),
            reasoning_tokens = payload["usage"]["completion_tokens_details"]["reasoning_tokens"]
                .as_u64()
                .unwrap_or(0),
            finish_reason = payload["choices"][0]["finish_reason"].as_str().unwrap_or("unknown"),
            "openai-compat tools call ok"
        );
        Ok(msg)
    }

    /// One streamed Chat Completions request. Visible text goes to `sink` as
    /// it arrives; tool-call fragments are assembled into the same message a
    /// buffered request returns, so both paths share every later check.
    async fn chat_completions_step_stream(
        &self,
        caller: &C,
        model: &str,
        body: &Value,
        sink: &dyn TextStreamSink,
    ) -> Result<Value> {
        let mut body = body.clone();
        body["stream"] = json!(true);
        body["stream_options"] = json!({ "include_usage": true });
        let started = std::time::Instant::now();
        let stalled = |events: usize| {
            tracing::warn!(
                caller = %caller,
                model = %model,
                elapsed_ms = started.elapsed().as_millis(),
                events,
                "openai-compat stream stalled without a new event"
            );
            Error::transient("openai-compat stream stalled")
        };
        let response = tokio::time::timeout(
            self.stream_stall,
            self.post("chat/completions")
                .timeout(CHAT_STEP_TIMEOUT)
                .json(&body)
                .send(),
        )
        .await
        .map_err(|_| stalled(0))?
        .map_err(|error| Error::transient(format!("openai-compat stream failed: {error}")))?;
        let status = response.status();
        let retry_after = parse_retry_after(response.headers());
        let generation = generation_id(response.headers());
        let tag = |error| tag_generation(error, generation.as_deref());
        if !status.is_success() {
            let raw = response.text().await.unwrap_or_default();
            let message = http_error_message(&raw);
            return Err(tag(Error::from_status_with_retry_after(
                status.as_u16(),
                format!("openai-compat {status}: {message}"),
                retry_after,
            )));
        }
        let mut accumulator = ChatStreamAccumulator::default();
        let mut chunks = response.bytes_stream();
        let mut last_event = std::time::Instant::now();
        let mut events_seen = 0;
        while !accumulator.done {
            let wait = self.stream_stall.saturating_sub(last_event.elapsed());
            let Some(chunk) = tokio::time::timeout(wait, chunks.next())
                .await
                .map_err(|_| tag(stalled(accumulator.events)))?
            else {
                break;
            };
            let chunk = chunk.map_err(|error| {
                tag(Error::transient(format!(
                    "openai-compat stream read: {error}"
                )))
            })?;
            accumulator.push(&chunk, sink).map_err(tag)?;
            if accumulator.events != events_seen {
                events_seen = accumulator.events;
                last_event = std::time::Instant::now();
            }
        }
        accumulator.finish(sink).map_err(tag)?;
        let payload = accumulator.payload(model);
        let raw = payload.to_string();
        self.finish_chat_payload(caller, model, payload, retry_after, started, &raw)
            .map_err(tag)
    }

    fn finish_responses_payload(
        &self,
        caller: &C,
        model: &str,
        payload: Value,
        retry_after: Option<u64>,
        started: std::time::Instant,
    ) -> Result<Value> {
        let label = self.responses_label();
        if payload.get("error").is_some_and(|error| !error.is_null()) {
            let (code, message) = responses_error(
                &responses_body_error(&payload).unwrap_or_default(),
                payload["error_type"].as_str(),
            );
            return Err(Error::from_status_with_retry_after(
                code,
                format!("{label} embedded error {code}: {message}"),
                retry_after,
            ));
        }
        if let Some(usage) = payload.get("usage") {
            let input = usage["input_tokens"].as_u64().unwrap_or(0);
            let cached = usage["input_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0)
                .min(input);
            let ran = payload["model"].as_str().unwrap_or(model);
            self.usage.record(
                caller,
                ran,
                Usage {
                    input_tokens: input - cached,
                    output_tokens: usage["output_tokens"].as_u64().unwrap_or(0),
                    cache_read_tokens: cached,
                    cache_write_tokens: usage["input_tokens_details"]["cache_write_tokens"]
                        .as_u64()
                        .unwrap_or(0)
                        .min(input - cached),
                },
                crate::openrouter_cost_cents(usage),
            );
        }
        if payload["status"] == "incomplete" {
            let reason = payload["incomplete_details"]["reason"]
                .as_str()
                .unwrap_or("unknown reason");
            let stop = match reason {
                "max_output_tokens" | "max_tokens" | "length" => StopReason::MaxTokens,
                "content_filter" => StopReason::ContentFilter,
                _ => StopReason::Incomplete,
            };
            return Err(Error::stopped(
                stop,
                format!("{label} reply incomplete: {reason}"),
            ));
        }
        if payload["status"] == "failed" {
            let message = payload["error"]["message"]
                .as_str()
                .unwrap_or("unknown failure");
            return Err(Error::transient(format!("{label} failed: {message}")));
        }
        let status = payload["status"]
            .as_str()
            .ok_or_else(|| Error::transient(format!("{label} reply omitted its status")))?;
        if status != "completed" {
            return Err(Error::transient(format!(
                "{label} returned terminally unusable status: {status}"
            )));
        }
        tracing::debug!(
            caller = %caller,
            requested_model = %model,
            ran_model = payload["model"].as_str().unwrap_or(model),
            latency_ms = started.elapsed().as_millis() as u64,
            in_tokens = payload["usage"]["input_tokens"].as_u64().unwrap_or(0),
            cached_tokens = payload["usage"]["input_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0),
            out_tokens = payload["usage"]["output_tokens"].as_u64().unwrap_or(0),
            reasoning_tokens = payload["usage"]["output_tokens_details"]["reasoning_tokens"].as_u64().unwrap_or(0),
            "{label} call ok"
        );
        Ok(payload)
    }

    /// One OpenRouter Responses API request for a tool-loop step. The response
    /// payload is returned intact because function calls and final text are
    /// sibling output items rather than one Chat Completions message.
    async fn responses_step_send(&self, caller: &C, model: &str, body: &Value) -> Result<Value> {
        let label = self.responses_label();
        let _permit = self.permit().await;
        let req = self.post("responses");
        tracing::debug!(caller = %caller, model = %model, "{label} request");
        let started = std::time::Instant::now();
        let resp = req
            .timeout(CHAT_STEP_TIMEOUT)
            .json(body)
            .send()
            .await
            .map_err(|e| {
                tracing::warn!(
                    caller = %caller,
                    model = %model,
                    latency_ms = started.elapsed().as_millis() as u64,
                    "{label} request failed: {e}"
                );
                Error::transient(format!("{label} request failed: {e}"))
            })?;
        let status = resp.status();
        let retry_after = parse_retry_after(resp.headers());
        let generation = generation_id(resp.headers());
        let raw = resp
            .text()
            .await
            .map_err(|e| Error::transient(format!("{label} read body: {e}")))?;
        if !status.is_success() {
            let parsed = serde_json::from_str::<Value>(&raw).ok();
            let msg = parsed
                .as_ref()
                .and_then(responses_body_error)
                .map(|error| provider_error_message(&error))
                .unwrap_or_else(|| raw.chars().take(300).collect());
            tracing::warn!(
                caller = %caller,
                model = %model,
                status = status.as_u16(),
                latency_ms = started.elapsed().as_millis() as u64,
                "{label} call failed: {msg}"
            );
            return Err(tag_generation(
                Error::from_status_with_retry_after(
                    status.as_u16(),
                    format!("{label} {status}: {msg}"),
                    retry_after,
                ),
                generation.as_deref(),
            ));
        }
        let payload: Value = serde_json::from_str(&raw)
            .map_err(|e| Error::transient(format!("{label} bad json: {e}")))?;
        self.finish_responses_payload(caller, model, payload, retry_after, started)
            .map_err(|error| tag_generation(error, generation.as_deref()))
    }

    async fn responses_step_stream(
        &self,
        caller: &C,
        model: &str,
        body: &Value,
        sink: &dyn TextStreamSink,
    ) -> Result<Value> {
        let label = self.responses_label();
        let _permit = self.permit().await;
        let req = self.post("responses");
        let started = std::time::Instant::now();
        let stalled = |events: usize| {
            tracing::warn!(
                caller = %caller,
                elapsed_ms = started.elapsed().as_millis(),
                events,
                "Responses stream stalled without a new event"
            );
            Error::transient(format!("{label} stream stalled"))
        };
        let response = tokio::time::timeout(
            self.stream_stall,
            req.timeout(CHAT_STEP_TIMEOUT).json(body).send(),
        )
        .await
        .map_err(|_| stalled(0))?
        .map_err(|error| Error::transient(format!("{label} stream failed: {error}")))?;
        let status = response.status();
        let retry_after = parse_retry_after(response.headers());
        let generation = generation_id(response.headers());
        let tag = |error| tag_generation(error, generation.as_deref());
        if !status.is_success() {
            let raw = response.text().await.unwrap_or_default();
            let parsed = serde_json::from_str::<Value>(&raw).ok();
            let message = parsed
                .as_ref()
                .and_then(responses_body_error)
                .map(|error| provider_error_message(&error))
                .unwrap_or_else(|| raw.chars().take(300).collect());
            return Err(tag(Error::from_status_with_retry_after(
                status.as_u16(),
                format!("{label} {status}: {message}"),
                retry_after,
            )));
        }

        let mut accumulator = ResponsesStreamAccumulator {
            label,
            ..Default::default()
        };
        let mut chunks = response.bytes_stream();
        let mut last_event = std::time::Instant::now();
        let mut events_seen = 0;
        loop {
            let wait = self.stream_stall.saturating_sub(last_event.elapsed());
            let Some(chunk) = tokio::time::timeout(wait, chunks.next())
                .await
                .map_err(|_| tag(stalled(accumulator.events)))?
            else {
                break;
            };
            let chunk = chunk.map_err(|error| {
                tracing::warn!(
                    caller = %caller,
                    timed_out = error.is_timeout(),
                    elapsed_ms = started.elapsed().as_millis(),
                    events = accumulator.events,
                    reasoning_events = accumulator.reasoning_events,
                    tool_events = accumulator.tool_events,
                    text_bytes = accumulator.text_bytes,
                    generation = generation.as_deref().unwrap_or_default(),
                    "Responses stream interrupted before its terminal event"
                );
                tag(Error::transient(format!("{label} stream read: {error}")))
            })?;
            accumulator.push(&chunk, sink).map_err(tag)?;
            if accumulator.events != events_seen {
                events_seen = accumulator.events;
                last_event = std::time::Instant::now();
            }
            // A terminal Responses event completes the operation. Waiting for
            // HTTP EOF can discard a completed reply when a proxy keeps SSE open.
            if accumulator.completed.is_some() {
                break;
            }
        }
        accumulator.finish(sink).map_err(tag)?;
        let payload = accumulator.completed.ok_or_else(|| {
            tag(Error::transient(format!(
                "{label} stream ended without a completed response"
            )))
        })?;
        self.finish_responses_payload(caller, model, payload, retry_after, started)
            .map_err(tag)
    }

    async fn responses_step_send_mode(
        &self,
        caller: &C,
        model: &str,
        body: &Value,
        stream_sink: Option<&dyn TextStreamSink>,
    ) -> Result<Value> {
        match stream_sink {
            Some(sink) => self.responses_step_stream(caller, model, body, sink).await,
            None => self.responses_step_send(caller, model, body).await,
        }
    }

    /// The Responses body for a tool step, before reasoning and provider
    /// quirks, with the tool catalogue's shape for telemetry.
    fn responses_base(
        &self,
        caller: &C,
        system: SystemPrompt<'_>,
        history: &[ToolTurn],
        tools: &[Value],
    ) -> ResponsesBase {
        let model = self.model_for(history);
        let max_output_tokens = if tools.is_empty() {
            1_024
        } else {
            caller.tool_step_output_tokens().unwrap_or(4_096)
        };
        let mut response_tools = tools.iter().map(responses_function).collect::<Vec<_>>();
        let anthropic_policy = self.anthropic_policy(model);
        anthropic_policy.prepare_tools(&mut response_tools);
        let deferred_tools = response_tools
            .iter()
            .filter(|tool| tool["defer_loading"] == true)
            .count();
        let deferred_names = response_tools
            .iter()
            .filter(|tool| tool["defer_loading"] == true)
            .filter_map(|tool| tool["name"].as_str().map(str::to_owned))
            .collect::<std::collections::HashSet<_>>();
        if deferred_tools > 0 {
            if self.is_openrouter() {
                response_tools.insert(0, json!({"type": "openrouter:tool_search"}));
            } else {
                // Tool search is OpenRouter's. Elsewhere, send the full catalogue.
                for tool in &mut response_tools {
                    tool.as_object_mut()
                        .map(|tool| tool.remove("defer_loading"));
                }
            }
        }
        let (deferred_tools, deferred_names) = if self.is_openrouter() {
            (deferred_tools, deferred_names)
        } else {
            (0, Default::default())
        };
        let initially_loaded_tools = response_tools
            .iter()
            .filter(|tool| tool["type"] == "function" && tool["defer_loading"] != true)
            .count();
        let mut base = json!({
            "model": model,
            "instructions": system.as_str(),
            "input": self.restore_reasoning(&caller.scope(), responses_input(history)),
            "max_output_tokens": max_output_tokens,
            "parallel_tool_calls": true,
            "store": false,
        });
        anthropic_policy.apply_responses_instructions(&mut base, system);
        if response_tools.is_empty() {
            base["tool_choice"] = json!("none");
        } else {
            base["tools"] = Value::Array(response_tools);
            base["tool_choice"] = json!("auto");
        }
        self.apply_openrouter_session(&mut base, &caller.scope(), "safety_identifier");
        ResponsesBase {
            body: base,
            deferred_names,
            initially_loaded_tools,
            deferred_tools,
        }
    }

    async fn responses_chat_step(
        &self,
        caller: &C,
        system: SystemPrompt<'_>,
        history: &[ToolTurn],
        tools: &[Value],
        stream_sink: Option<Arc<dyn TextStreamSink>>,
    ) -> Result<ChatStep> {
        let label = self.responses_label();
        let model = self.model_for(history);
        let ResponsesBase {
            body: base,
            deferred_names,
            initially_loaded_tools,
            deferred_tools,
        } = self.responses_base(caller, system, history, tools);
        self.observe_request_size(&base);

        tracing::debug!(
            caller = %caller,
            model = %model,
            initially_loaded_tools,
            deferred_tools,
            tool_search_enabled = deferred_tools > 0,
            tool_schema_bytes = tools.iter().map(|tool| tool.to_string().len()).sum::<usize>(),
            "{label} tool catalogue prepared"
        );

        let has_images = history
            .iter()
            .any(|turn| matches!(turn, ToolTurn::UserWithImages(_, images) if !images.is_empty()));
        let reasoning = if has_images {
            None
        } else {
            self.step_reasoning()
        };
        let mut body = base.clone();
        self.apply_provider_quirks(&mut body, reasoning);
        // OpenAI reasoning models reason at their default effort when none is
        // configured, so their reasoning is always worth carrying forward.
        let carries_reasoning =
            (reasoning.is_some() && self.is_openrouter()) || self.openai_reasons(model);
        if carries_reasoning {
            // Stateless requests only return reasoning that can be echoed back
            // when its encrypted form is requested explicitly.
            body["include"] = json!(["reasoning.encrypted_content"]);
        }
        if stream_sink.is_some() {
            body["stream"] = Value::Bool(true);
        }
        let mut payload = self
            .responses_step_send_mode(caller, model, &body, stream_sink.as_deref())
            .await;
        // A valid-but-empty reasoning response can be retried once without
        // reasoning. Transport, authentication, rate-limit, and protocol errors
        // belong to FallbackLlm; retrying them here would create a second,
        // overlapping retry policy.
        if reasoning.is_some() && payload.as_ref().is_ok_and(responses_payload_is_empty) {
            tracing::warn!(
                caller = %caller,
                model = %model,
                "reasoning Responses step returned no content; retrying without reasoning"
            );
            let mut plain = base;
            // The retry disables reasoning, so echoed reasoning items go too.
            if let Some(input) = plain["input"].as_array_mut() {
                input.retain(|item| item["type"] != "reasoning");
            }
            self.apply_provider_quirks(&mut plain, None);
            if stream_sink.is_some() {
                plain["stream"] = Value::Bool(true);
                if let Some(sink) = stream_sink.as_deref() {
                    sink.reset();
                }
            }
            payload = self
                .responses_step_send_mode(caller, model, &plain, stream_sink.as_deref())
                .await;
        } else if reasoning.is_none()
            && let Err(error) = &payload
            && self.learn_mandatory_reasoning(model, error)
        {
            let mut low = base;
            self.apply_provider_quirks(&mut low, None);
            if stream_sink.is_some() {
                low["stream"] = Value::Bool(true);
            }
            payload = self
                .responses_step_send_mode(caller, model, &low, stream_sink.as_deref())
                .await;
        }
        let payload = payload?;
        let output = payload["output"].as_array();
        let function_calls = output
            .into_iter()
            .flatten()
            .filter(|item| item["type"] == "function_call")
            .filter_map(|item| item["name"].as_str())
            .collect::<Vec<_>>();
        let tool_search_items = output
            .into_iter()
            .flatten()
            .filter(|item| {
                item["type"]
                    .as_str()
                    .is_some_and(|kind| kind.contains("tool_search"))
            })
            .count();
        let discovered_calls = function_calls
            .iter()
            .filter(|name| deferred_names.contains(**name))
            .count();
        tracing::info!(
            caller = %caller,
            function_calls = function_calls.len(),
            function_names = ?function_calls,
            discovered_calls,
            tool_search_items,
            tool_search_enabled = deferred_tools > 0,
            "{label} tool selection"
        );
        let step = interpret_responses_payload(&payload);
        if reasoning.is_some() || !self.is_openrouter() {
            self.remember_reasoning(&caller.scope(), &payload, &step);
        }
        Ok(step)
    }

    /// The Chat Completions body for a one-shot completion.
    fn completion_body(&self, caller: &C, request: &Completion<'_>) -> Value {
        let Completion {
            system,
            user,
            max_tokens,
            json_object,
            schema,
        } = *request;
        let mut body = json!({
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": user },
            ],
        });
        self.apply_openrouter_session(&mut body, &caller.scope(), "user");
        if let Some(schema) = schema {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": {
                    "name": schema.name,
                    "strict": schema.strict,
                    "schema": schema.schema,
                },
            });
        } else if json_object {
            // OpenRouter/OpenAI JSON mode: eliminates the malformed-JSON class
            // instead of retrying it. Guarded by the caller flag so a bare-value
            // call never gets forced into object shape.
            body["response_format"] = json!({ "type": "json_object" });
        }
        // Small deterministic calls never reason,
        // even when a reasoning effort is configured for chat: reasoning adds
        // tens of seconds to a 24-token extraction, and it routes into the
        // reasoning provider pool, which rate-limits far earlier than the
        // plain pool. Small calls are often the fallback a host relies on
        // when a large call has already failed.
        self.apply_provider_quirks(&mut body, None);
        if self.openai_reasons(&self.model) {
            // OpenAI reasoning models cannot all turn reasoning off; `low` is
            // the one effort every one of them accepts. Its reasoning is short,
            // so the headroom is smaller than a tool step's.
            body["reasoning_effort"] = json!("low");
            body["max_completion_tokens"] =
                json!(u64::from(max_tokens) + HELPER_REASONING_HEADROOM);
        }
        let rejects_temperature =
            anthropic::rejects_temperature(&self.model) || openai::is_reasoning_model(&self.model);
        // Every host: direct APIs reject it, and OpenRouter with
        // `require_parameters` (below) finds no endpoint at all.
        if rejects_temperature {
            body.as_object_mut().map(|body| body.remove("temperature"));
        }
        if schema.is_some() && self.is_openrouter() {
            // Route only to endpoints that honour `response_format`, so a
            // schema is never silently ignored.
            body["provider"]["require_parameters"] = json!(true);
        }
        body
    }

    async fn complete_request(&self, caller: &C, request: Completion<'_>) -> Result<String> {
        match self.complete_once(caller, &request).await {
            Err(error) if self.learn_mandatory_reasoning(&self.model, &error) => {
                self.complete_once(caller, &request).await
            }
            other => other,
        }
    }

    async fn complete_once(&self, caller: &C, request: &Completion<'_>) -> Result<String> {
        let _permit = self.permit().await;
        let body = self.completion_body(caller, request);

        let req = self.post("chat/completions");

        tracing::debug!(caller = %caller, model = %self.model, "openai-compat request");
        // Time the round-trip so prod logs show LLM latency (a common cause of
        // "the assistant went quiet").
        let started = std::time::Instant::now();
        let timeout = self.completion_timeout;
        let resp = req.timeout(timeout).json(&body).send().await.map_err(|e| {
            tracing::warn!(
                caller = %caller, model = %self.model,
                latency_ms = started.elapsed().as_millis() as u64,
                "openai-compat request failed (transient): {e}"
            );
            if e.is_timeout() {
                Error::transient(format!("openai-compat request timed out after {timeout:?}"))
            } else {
                Error::transient(format!("openai-compat request failed: {e}"))
            }
        })?;
        // Status BEFORE JSON: a gateway 5xx returns non-JSON, which parsed as JSON
        // would surface as a status-less "bad json" transient and mis-retry a hard
        // failure. Read Retry-After off the headers so a 429/529 backs off as asked.
        let status = resp.status();
        let retry_after = parse_retry_after(resp.headers());
        let generation = generation_id(resp.headers());
        let raw = resp.text().await.map_err(|e| {
            tracing::warn!(
                caller = %caller, model = %self.model, status = status.as_u16(),
                "openai-compat could not read response body: {e}"
            );
            Error::transient(format!("openai-compat read body: {e}"))
        })?;
        if !status.is_success() {
            let msg = http_error_message(&raw);
            // The single most useful prod signal: which model, what HTTP status,
            // and the provider's own error message (rate limit, auth, overload…).
            tracing::warn!(
                caller = %caller, model = %self.model, status = status.as_u16(),
                latency_ms = started.elapsed().as_millis() as u64,
                "openai-compat call failed: {msg}"
            );
            return Err(tag_generation(
                Error::from_status_with_retry_after(
                    status.as_u16(),
                    format!("openai-compat {status}: {msg}"),
                    retry_after,
                ),
                generation.as_deref(),
            ));
        }
        let payload: Value = serde_json::from_str(&raw).map_err(|e| {
            tracing::warn!(
                caller = %caller, model = %self.model, status = status.as_u16(),
                "openai-compat response was not valid JSON: {e}"
            );
            Error::transient(format!("openai-compat bad json: {e}"))
        })?;
        // OpenRouter can wrap an upstream failure in an HTTP 200 whose body is
        // just `{"error": {...}}` with no choices (seen live: "temporarily
        // rate-limited upstream", code 429). Map it by its embedded code so a
        // wrapped 429/5xx is retried with backoff instead of hard-failing.
        if let Some(e) = embedded_error(&payload, retry_after) {
            tracing::warn!(
                caller = %caller, model = %self.model,
                latency_ms = started.elapsed().as_millis() as u64,
                "openai-compat 200 OK with embedded error: {e}"
            );
            return Err(e);
        }

        if let Some(u) = payload.get("usage") {
            let prompt = u["prompt_tokens"].as_u64().unwrap_or(0);
            let cached = u["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .or_else(|| u["prompt_cache_hit_tokens"].as_u64())
                .unwrap_or(0)
                .min(prompt);
            let usage = Usage {
                input_tokens: prompt - cached,
                output_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
                cache_read_tokens: cached,
                cache_write_tokens: u["prompt_tokens_details"]["cache_write_tokens"]
                    .as_u64()
                    .unwrap_or(0)
                    .min(prompt - cached),
            };
            // Attribute usage to the model that ACTUALLY ran (the response's
            // `model`), which may be a fallback rather than the configured one.
            let ran = payload["model"].as_str().unwrap_or(&self.model);
            let provider_cost = crate::openrouter_cost_cents(u);
            self.usage.record(caller, ran, usage, provider_cost);
        }

        // A 200 with no `choices` is a malformed/changed response, not an empty
        // reply: surface it instead of silently returning "" (which downstream
        // would mistake for the model having nothing to say).
        let message = payload
            .get("choices")
            .and_then(|c| c.get(0))
            .map(|c| &c["message"])
            .ok_or_else(|| {
                // Keep a body snippet: a bare "no choices" hides the real
                // cause, such as the embedded-429 shape.
                let snippet: String = raw.chars().take(300).collect();
                tracing::warn!(
                    caller = %caller, model = %self.model, body = %snippet,
                    "openai-compat 200 OK but response had no choices (API change?)"
                );
                Error::Protocol(format!("openai-compat: response had no choices: {snippet}"))
            })?;
        // Success at DEBUG: keeps the prod INFO stream clean but lets us turn on
        // per-call latency and token visibility when diagnosing.
        tracing::debug!(
            caller = %caller, model = %self.model,
            latency_ms = started.elapsed().as_millis() as u64,
            in_tokens = payload["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
            out_tokens = payload["usage"]["completion_tokens"].as_u64().unwrap_or(0),
            "openai-compat call ok"
        );
        if let Some(refusal) = refusal_of(message) {
            return Err(Error::stopped(StopReason::Refusal, refusal));
        }
        // A cut-off reply is not an answer: partial text reads as complete,
        // and an empty one as the model having nothing to say. Raise the
        // ceiling, or ask for a shorter reply in the prompt.
        let stopped = match payload["choices"][0]["finish_reason"].as_str() {
            Some("length") => Some((
                StopReason::MaxTokens,
                "openai-compat reply hit the max_tokens ceiling; partial reply discarded",
            )),
            Some("content_filter" | "content-filter") => Some((
                StopReason::ContentFilter,
                "openai-compat reply stopped on a content filter; partial reply discarded",
            )),
            _ => None,
        };
        if let Some((reason, message)) = stopped {
            return Err(tag_generation(
                Error::stopped(reason, message),
                generation.as_deref(),
            ));
        }
        Ok(message["content"].as_str().unwrap_or("").trim().to_string())
    }
}

impl<C: Caller + ?Sized> OpenAiCompat<C> {
    /// The Chat Completions model and body for a tool step, before reasoning
    /// and provider quirks.
    fn chat_step_base(
        &self,
        caller: &C,
        system: SystemPrompt<'_>,
        history: &[ToolTurn],
        tools: &[Value],
    ) -> (String, Value) {
        // Build the OpenAI messages array from the turn history.
        let mut messages: Vec<Value> =
            vec![json!({ "role": "system", "content": system.as_str() })];
        if let Some(blocks) = self
            .anthropic_policy(self.model_for(history))
            .chat_cache_blocks(system)
        {
            messages[0]["content"] = json!(blocks);
        }
        for turn in history {
            match turn {
                ToolTurn::User(text) => messages.push(json!({ "role": "user", "content": text })),
                ToolTurn::UserWithImages(text, images) => {
                    let mut parts: Vec<Value> = images
                        .iter()
                        .map(|img| {
                            json!({"type":"image_url","image_url":{
                                "url": format!("data:{};base64,{}", img.mime, img.data_b64)
                            }})
                        })
                        .collect();
                    parts.push(json!({"type":"text","text":text}));
                    messages.push(json!({ "role": "user", "content": parts }));
                }
                ToolTurn::Assistant(text) => {
                    messages.push(json!({ "role": "assistant", "content": text }));
                }
                ToolTurn::AssistantToolCalls(calls) => {
                    let scope = caller.scope();
                    let tool_calls: Vec<Value> = calls
                        .iter()
                        .enumerate()
                        .map(|(index, c)| {
                            let mut call = json!({
                                "id": c.id,
                                "type": "function",
                                "function": { "name": c.name, "arguments": c.arguments.to_string() }
                            });
                            if let Some(extra) = self.call_extra(&scope, &c.id, index == 0) {
                                call["extra_content"] = extra;
                            }
                            call
                        })
                        .collect();
                    messages.push(
                        json!({ "role": "assistant", "content": null, "tool_calls": tool_calls }),
                    );
                }
                ToolTurn::ToolResults(results) => {
                    for (id, out) in results {
                        messages
                            .push(json!({ "role": "tool", "tool_call_id": id, "content": out }));
                    }
                }
            }
        }

        // Wrap each provider-neutral spec into the OpenAI `function` shape, with
        // the schema under `function.parameters` (see oai_function). Getting this
        // key wrong leaves the model with no declared parameters and forces it to
        // guess argument names.
        let oai_tools: Vec<Value> = tools.iter().map(oai_function).collect();

        let model = self.model_for(history).to_string();
        // No-tool calls are small structured helpers such as classification
        // and extraction. They return a few fields of JSON and must not inherit
        // a large ceiling: a huge ceiling let a malformed reviewer ramble until
        // the whole interactive turn timed out. Ordinary chat gets 4k, ample
        // prose and tool-call headroom. A caller whose tool arguments are large
        // raises it through `Caller::tool_step_output_tokens`.
        let max_tokens = if tools.is_empty() {
            1_024
        } else {
            caller.tool_step_output_tokens().unwrap_or(4_096)
        };
        let mut base = json!({
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        });
        // Some hosts (Anthropic's) reject an empty `tools` list.
        if !oai_tools.is_empty() {
            base["tools"] = json!(oai_tools);
        }
        self.apply_openrouter_session(&mut base, &caller.scope(), "user");
        (model, base)
    }

    /// The reasoning setting for a tool step. Image turns never reason.
    fn chat_step_reasoning(&self, history: &[ToolTurn]) -> Option<&str> {
        let has_images = history
            .iter()
            .any(|t| matches!(t, ToolTurn::UserWithImages(_, images) if !images.is_empty()));
        if has_images {
            None
        } else {
            self.step_reasoning()
        }
    }

    /// One Chat Completions tool step, streamed to `sink` when one is given.
    async fn chat_completions_chat_step(
        &self,
        caller: &C,
        system: SystemPrompt<'_>,
        history: &[ToolTurn],
        tools: &[Value],
        sink: Option<Arc<dyn TextStreamSink>>,
    ) -> Result<ChatStep> {
        let (model, base) = self.chat_step_base(caller, system, history, tools);
        let model = model.as_str();

        // Attempt 1 carries the configured reasoning effort for text turns.
        // Vision models are configured independently and may reject reasoning,
        // so image-bearing turns explicitly disable it. Reasoning is the
        // EXPENDABLE layer of a chat step: when the reasoning-enabled call fails
        // (the reasoning provider pool rate-limits upstream long before
        // the plain pool — seen live as an embedded 429 on nearly half a
        // session's turns) or degenerate (the reasoning stream consumed the
        // reply and `content` is empty), the SAME turn is re-sent once with
        // reasoning disabled. The privacy floor is never traded away —
        // only the effort is. A failure of BOTH attempts propagates normally.
        let reasoning = self.chat_step_reasoning(history);
        let mut body = base.clone();
        self.apply_provider_quirks(&mut body, reasoning);
        let mut msg = self
            .chat_completions_step_send(caller, model, &body, sink.as_deref())
            .await;
        if reasoning.is_some()
            && match &msg {
                // Reasoning-only failures are often hard 400s: an upstream may
                // reject reasoning or the inflated token ceiling. The same base
                // request can still work, so retry every failed reasoning call
                // exactly once without it. Even auth/config failures remain
                // bounded to these two attempts.
                Err(_) => true,
                Ok(m) => message_is_empty(m),
            }
        {
            tracing::warn!(
                caller = %caller, model = %model,
                "reasoning chat step failed or returned no content — retrying once without reasoning"
            );
            let mut plain = base;
            self.apply_provider_quirks(&mut plain, None);
            if let Some(sink) = &sink {
                sink.reset();
            }
            msg = self
                .chat_completions_step_send(caller, model, &plain, sink.as_deref())
                .await;
        } else if reasoning.is_none()
            && let Err(error) = &msg
            && self.learn_mandatory_reasoning(model, error)
        {
            let mut low = base;
            self.apply_provider_quirks(&mut low, None);
            msg = self
                .chat_completions_step_send(caller, model, &low, sink.as_deref())
                .await;
        }
        let msg = msg?;
        self.remember_call_extras(&caller.scope(), &msg);
        let (step, parse) = interpret_tool_message(&msg);
        // Surface the XML-recovery path: it means the provider's native
        // tool_calls were unusable. Silent before, this is exactly the failure
        // that took a long hunt to find — a WARN here makes a recurrence (new
        // model, provider regression) visible in prod immediately.
        if parse != ToolParse::Native {
            // InvalidNativeArguments means the native calls were retained for
            // explicit validation results; the XML variants mean a complete XML
            // copy recovered the batch. Either path remains visible in telemetry.
            let detail = match parse {
                ToolParse::InvalidNativeArguments => {
                    "native tool-call arguments were malformed — preserving the call for an explicit invalid-arguments result"
                }
                ToolParse::PartialInvalidNativeArguments => {
                    "some parallel native tool-call arguments were malformed — preserving every call for paired results"
                }
                _ => {
                    "recovered tool call from Hermes XML — provider's native tool_calls were unusable"
                }
            };
            tracing::warn!(
                caller = %caller, model = %model, parse = ?parse,
                "{detail}"
            );
        }
        Ok(step)
    }
}

/// Models which reject OpenRouter's explicit `{enabled:false}`. Keep this
/// narrow: enabling reasoning changes both latency and routing, so an unrelated
/// model must not inherit it merely because it happens to support reasoning.
fn reasoning_required(model: &str) -> bool {
    model == "z-ai/glm-5.3-flash" || model.starts_with("z-ai/glm-5.3-flash:")
}

#[async_trait]
impl<C: Caller + ?Sized> ChatModel<C> for OpenAiCompat<C> {
    fn model_id(&self) -> &str {
        &self.model
    }

    async fn complete(&self, caller: &C, request: Completion<'_>) -> Result<String> {
        self.complete_request(caller, request).await
    }

    fn supports_tools(&self) -> bool {
        true
    }

    async fn chat_step(
        &self,
        caller: &C,
        system: SystemPrompt<'_>,
        history: &[ToolTurn],
        tools: &[Value],
    ) -> Result<ChatStep> {
        if self.uses_responses() {
            return self
                .responses_chat_step(caller, system, history, tools, None)
                .await;
        }
        self.chat_completions_chat_step(caller, system, history, tools, None)
            .await
    }

    async fn chat_step_stream(
        &self,
        caller: &C,
        system: SystemPrompt<'_>,
        history: &[ToolTurn],
        tools: &[Value],
        sink: Arc<dyn TextStreamSink>,
    ) -> Result<ChatStep> {
        if self.uses_responses() {
            self.responses_chat_step(caller, system, history, tools, Some(sink))
                .await
        } else {
            self.chat_completions_chat_step(caller, system, history, tools, Some(sink))
                .await
        }
    }
}

/// Turn one response `message` object into a `ChatStep`. A complete native
/// OpenAI `tool_calls` batch is authoritative. If some arguments are malformed,
/// complete Hermes XML may recover the whole batch; otherwise native calls are
/// preserved in order so dispatch can return explicit validation results.
///
/// The Hermes-XML fallback only fires when there is no usable native call:
/// several open models on OpenRouter (qwen, mistral, and others) emit Hermes-XML
/// tool calls in `content`, and OpenRouter's conversion into `tool_calls` can
/// come back TRUNCATED (`{"which": ` — with finish_reason=tool_calls, so not a
/// length cutoff; verified by raw replay, same class as RooCode #6630). The
/// complete call is the XML in `content`, so we recover it there.
/// How a tool call was recovered from the response — for observability. The XML
/// variants mean the provider's native `tool_calls` field was unusable and we
/// fell back to parsing Hermes XML out of `content`. A spike in these (esp.
/// `XmlAfterMalformedNative`) signals the OpenRouter/provider truncation bug
/// recurring — the thing that was silently breaking tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolParse {
    /// Clean native `tool_calls` (or a plain-text reply) — the healthy path.
    Native,
    /// Native `tool_calls` were present but malformed; recovered from XML.
    XmlAfterMalformedNative,
    /// No native `tool_calls`; the call was Hermes XML in `content`.
    XmlNoNative,
    /// Every named native call had malformed arguments and no complete XML copy
    /// recovered the batch. The calls remain in the step with their raw argument
    /// strings, so dispatch returns one paired `invalid_arguments` result per call.
    InvalidNativeArguments,
    /// Some native calls parsed and some had malformed arguments. All calls remain
    /// in provider order; malformed calls receive paired validation failures while
    /// valid siblings execute normally.
    PartialInvalidNativeArguments,
}

/// OpenRouter can deliver an upstream failure as an HTTP **200** whose body is
/// just `{"error": {"message": …, "code": 429}}` with no `choices` (seen live
/// when the reasoning pool for a model was rate-limited upstream). Map it exactly as if the embedded code had been the
/// HTTP status, so a wrapped 429/5xx stays RETRYABLE and the fallback chain
/// backs off instead of hard-failing the turn.
/// None = the response is not this shape (healthy, or no error object).
fn embedded_error(payload: &Value, retry_after_secs: Option<u64>) -> Option<Error> {
    if payload["choices"].as_array().is_some_and(|a| !a.is_empty()) {
        return None;
    }
    let err = payload.get("error")?;
    let code = openrouter_error_status(err, None);
    let msg = provider_error_message(err);
    Some(Error::from_status_with_retry_after(
        code,
        format!("openai-compat embedded error {code}: {msg}"),
        retry_after_secs,
    ))
}

/// Keep the machine-readable reason used by retry classification. The human
/// message alone cannot distinguish temporary reservations from spent credit.
fn provider_error_message(error: &Value) -> String {
    let mut message = error["message"]
        .as_str()
        .unwrap_or("(no message)")
        .to_string();
    push_error_metadata(&mut message, error);
    message
}

/// The HTTP status of an OpenRouter error object. Chat Completions uses a
/// numeric `code`; the Responses API uses a reduced string `code` plus a
/// precise canonical `error_type` (top level on a failed response, or in
/// `error.metadata`). An unknown error is safer retried once than hard-failed.
/// https://openrouter.ai/docs/api_reference/errors-and-debugging#typed-error-codes
fn openrouter_error_status(error: &Value, error_type: Option<&str>) -> u16 {
    if let Some(code) = error["code"]
        .as_u64()
        .and_then(|code| u16::try_from(code).ok())
        .or_else(|| error["code"].as_str().and_then(|code| code.parse().ok()))
    {
        return code;
    }
    let kind = error_type
        .or_else(|| error["metadata"]["error_type"].as_str())
        .or_else(|| error["code"].as_str())
        .unwrap_or_default();
    match kind {
        "context_length_exceeded"
        | "max_tokens_exceeded"
        | "token_limit_exceeded"
        | "string_too_long"
        | "invalid_request"
        | "invalid_prompt"
        | "invalid_image"
        | "image_too_large"
        | "image_too_small"
        | "unsupported_image_format"
        | "image_download_failed" => 400,
        "authentication" => 401,
        "payment_required" => 402,
        "permission_denied"
        | "content_policy_violation"
        | "image_content_policy_violation"
        | "refusal" => 403,
        "not_found" | "image_not_found" => 404,
        "precondition_failed" => 412,
        "payload_too_large" => 413,
        "unprocessable" => 422,
        "rate_limit_exceeded" => 429,
        "provider_unavailable" => 502,
        "provider_overloaded" => 503,
        "timeout" => 504,
        _ => 500,
    }
}

/// The error object from a Responses API body. That API places `metadata`
/// beside `error` rather than inside it (Chat Completions nests it), so the
/// machine-readable cause, such as a temporary in-flight 402, would otherwise
/// be lost and a retryable error treated as hard.
fn responses_body_error(body: &Value) -> Option<Value> {
    let mut error = body.get("error").filter(|error| error.is_object())?.clone();
    if error.get("metadata").is_none_or(Value::is_null)
        && let Some(metadata) = body.get("metadata").filter(|metadata| metadata.is_object())
    {
        error["metadata"] = metadata.clone();
    }
    Some(error)
}

/// A Responses API error object with the response's top-level `error_type`
/// echoed into the message, so overflow detection sees the canonical type.
fn responses_error(error: &Value, error_type: Option<&str>) -> (u16, String) {
    let mut message = provider_error_message(error);
    if let Some(error_type) = error_type.filter(|kind| !message.contains(*kind)) {
        message.push_str(" [");
        message.push_str(error_type);
        message.push(']');
    }
    (openrouter_error_status(error, error_type), message)
}

/// A non-2xx body: the provider's error message when it is JSON, otherwise the
/// start of the raw body (a gateway's HTML or plain text).
fn http_error_message(raw: &str) -> String {
    let parsed = serde_json::from_str::<Value>(raw).ok();
    // Gemini wraps its error object in a one-element array.
    let body = parsed
        .as_ref()
        .map(|body| body.get(0).filter(|_| body.is_array()).unwrap_or(body));
    let error = body.map(|body| &body["error"]);
    let mut message = error
        .and_then(|error| error["message"].as_str())
        .map(str::to_string)
        .unwrap_or_else(|| raw.chars().take(200).collect());
    if let Some(error) = error {
        push_error_metadata(&mut message, error);
    }
    message
}

/// OpenRouter's machine-readable cause. 402s carry `limit_source` (for example
/// `openrouter_in_flight_budget`); older payloads used `reason`. Echo both so
/// retry classification never depends on the free-text message.
fn push_error_metadata(message: &mut String, error: &Value) {
    for key in ["reason", "limit_source", "error_type"] {
        if let Some(value) = error["metadata"][key].as_str() {
            message.push_str(" [");
            message.push_str(value);
            message.push(']');
        }
    }
    // OpenAI's own machine-readable cause, such as `insufficient_quota`,
    // which separates spent credit from a rate limit on the same 429.
    for key in ["type", "code"] {
        if let Some(value) = error[key]
            .as_str()
            .filter(|value| QUOTA_CODES.contains(value) && !message.contains(*value))
        {
            message.push_str(" [");
            message.push_str(value);
            message.push(']');
        }
    }
}

/// OpenAI error types and codes for exhausted credit.
const QUOTA_CODES: [&str; 2] = ["insufficient_quota", "credit_balance_exhausted"];

/// A response `message` carrying neither tool calls nor text — the degenerate
/// reply a reasoning model produces when the reasoning stream ate the whole
/// turn. Distinct from a legitimate tool-call message, whose `content` is
/// also empty.
/// A model's explicit refusal, which OpenAI returns beside an empty content
/// when it declines to follow a structured-output request.
fn refusal_of(message: &Value) -> Option<String> {
    message["refusal"]
        .as_str()
        .map(str::trim)
        .filter(|refusal| !refusal.is_empty())
        .map(|refusal| format!("model refused: {refusal}"))
}

fn message_is_empty(msg: &Value) -> bool {
    let no_calls = msg["tool_calls"].as_array().is_none_or(|a| a.is_empty());
    no_calls && msg["content"].as_str().unwrap_or("").trim().is_empty()
}

fn request_size(body: &Value) -> RequestSize {
    let mut initial_tools_bytes = 0;
    let mut deferred_tools_bytes = 0;
    let mut initially_loaded_tools = 0;
    let mut deferred_tools = 0;
    let mut tool_search_enabled = false;
    for tool in body["tools"].as_array().into_iter().flatten() {
        tool_search_enabled |= tool["type"] == "openrouter:tool_search";
        if tool["defer_loading"] == true {
            deferred_tools_bytes += tool.to_string().len();
            deferred_tools += 1;
        } else {
            initial_tools_bytes += tool.to_string().len();
            if tool["type"] == "function" {
                initially_loaded_tools += 1;
            }
        }
    }
    let mut instructions_bytes = body["instructions"].as_str().unwrap_or_default().len();
    let mut history = Vec::new();
    for item in body["input"].as_array().into_iter().flatten() {
        if matches!(item["role"].as_str(), Some("system" | "developer")) {
            instructions_bytes += item["content"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|block| block["text"].as_str())
                .map(str::len)
                .sum::<usize>();
        } else {
            history.push(item);
        }
    }
    RequestSize {
        instructions_bytes,
        history_bytes: json!(history).to_string().len(),
        initial_tools_bytes,
        deferred_tools_bytes,
        initially_loaded_tools,
        deferred_tools,
        tool_search_enabled,
    }
}

fn responses_function(spec: &Value) -> Value {
    let mut tool = json!({
        "type": "function",
        "name": spec["name"],
        "description": spec["description"],
        "parameters": spec.get("parameters").cloned().unwrap_or_else(|| {
            json!({ "type": "object", "properties": {} })
        }),
    });
    if spec["defer_loading"] == true {
        tool["defer_loading"] = Value::Bool(true);
    }
    tool
}

fn responses_input(history: &[ToolTurn]) -> Vec<Value> {
    use ToolTurn;

    let mut input = Vec::new();
    // A bounded replay may start with an earlier assistant reply. Claude via
    // OpenRouter treats a leading assistant item as instruction context rather
    // than a delivered reply. Establish the replay boundary without inventing
    // the missing user's words, and keep the assistant item in its original role.
    if matches!(history.first(), Some(ToolTurn::Assistant(_))) {
        input.push(json!({
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "[Replay boundary: the earlier user message is outside the saved window. The next assistant message was previously delivered in this conversation.]"}],
        }));
    }
    for (index, turn) in history.iter().enumerate() {
        match turn {
            ToolTurn::User(text) => input.push(json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            })),
            ToolTurn::UserWithImages(text, images) => {
                let mut content = vec![json!({"type": "input_text", "text": text})];
                content.extend(images.iter().map(|image| {
                    json!({
                        "type": "input_image",
                        "image_url": format!("data:{};base64,{}", image.mime, image.data_b64),
                    })
                }));
                input.push(json!({"type": "message", "role": "user", "content": content}));
            }
            ToolTurn::Assistant(text) => input.push(json!({
                "type": "message",
                "role": "assistant",
                // OpenRouter requires both fields on assistant history items.
                // Deterministic ids preserve caching across stateless requests.
                "id": format!("msg_history_{index}"),
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            })),
            ToolTurn::AssistantToolCalls(calls) => input.extend(calls.iter().map(|call| {
                json!({
                    "type": "function_call",
                    "status": "completed",
                    "id": call.id,
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments.to_string(),
                })
            })),
            ToolTurn::ToolResults(results) => input.extend(results.iter().map(|(id, output)| {
                json!({
                    "type": "function_call_output",
                    "call_id": id,
                    "output": output,
                })
            })),
        }
    }
    input
}

fn responses_payload_is_empty(payload: &Value) -> bool {
    let output = payload["output"].as_array();
    let has_call =
        output.is_some_and(|items| items.iter().any(|item| item["type"] == "function_call"));
    if has_call {
        return false;
    }
    responses_text(payload).trim().is_empty()
}

/// Incremental decoder for a Responses API event stream. Only output-text
/// deltas cross this boundary. Reasoning and function-call arguments remain
/// server-side even when a provider interleaves them with visible text.
#[derive(Default)]
struct ResponsesStreamAccumulator {
    /// The transport's name for error messages.
    label: &'static str,
    pending: Vec<u8>,
    completed: Option<Value>,
    visible_pending: String,
    last_visible_emit: Option<std::time::Instant>,
    events: usize,
    reasoning_events: usize,
    tool_events: usize,
    text_bytes: usize,
}

impl ResponsesStreamAccumulator {
    fn push(&mut self, chunk: &[u8], sink: &dyn TextStreamSink) -> Result<()> {
        self.pending.extend_from_slice(chunk);
        while let Some((at, separator_len)) = sse_frame_boundary(&self.pending) {
            let frame = self.pending[..at].to_vec();
            self.pending.drain(..at + separator_len);
            self.consume_frame(&frame)?;
        }
        if self.pending.len() > RESPONSES_MAX_EVENT_BYTES {
            return Err(Error::transient(format!(
                "{} stream contained an oversized event",
                self.label
            )));
        }
        self.flush_visible(sink, false);
        Ok(())
    }

    fn finish(&mut self, sink: &dyn TextStreamSink) -> Result<()> {
        if !self.pending.iter().all(u8::is_ascii_whitespace) {
            let frame = std::mem::take(&mut self.pending);
            self.consume_frame(&frame)?;
        }
        self.flush_visible(sink, true);
        Ok(())
    }

    fn flush_visible(&mut self, sink: &dyn TextStreamSink, force: bool) {
        if self.visible_pending.is_empty()
            || (!force
                && self
                    .last_visible_emit
                    .is_some_and(|last| last.elapsed() < RESPONSES_TEXT_FLUSH_INTERVAL))
        {
            return;
        }
        let delta = std::mem::take(&mut self.visible_pending);
        sink.delta(&delta);
        self.last_visible_emit = Some(std::time::Instant::now());
    }

    fn consume_frame(&mut self, frame: &[u8]) -> Result<()> {
        let frame = std::str::from_utf8(frame).map_err(|error| {
            Error::transient(format!("Responses stream was not UTF-8: {error}"))
        })?;
        let data = frame
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(|line| line.strip_prefix(' ').unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n");
        if data.is_empty() || data == "[DONE]" {
            return Ok(());
        }
        let event: Value = serde_json::from_str(&data).map_err(|error| {
            Error::transient(format!("{} stream bad json: {error}", self.label))
        })?;
        self.events += 1;
        let kind = event["type"].as_str().unwrap_or_default();
        if kind.starts_with("response.reasoning") {
            self.reasoning_events += 1;
        }
        if kind.starts_with("response.function_call") || kind.starts_with("response.tool_search") {
            self.tool_events += 1;
        }
        match kind {
            "response.output_text.delta" => {
                if let Some(delta) = event["delta"].as_str() {
                    self.text_bytes += delta.len();
                    self.visible_pending.push_str(delta);
                }
            }
            "response.completed" | "response.failed" | "response.incomplete" => {
                if let Some(response) = event.get("response") {
                    self.completed = Some(response.clone());
                }
            }
            "error" => {
                // The event itself is the error object: `code` is a string such
                // as `rate_limit_exceeded`, with `message` beside it.
                let (code, message) = responses_error(&event, event["error_type"].as_str());
                return Err(Error::from_status(
                    code,
                    format!("{} stream error {code}: {message}", self.label),
                ));
            }
            _ => {}
        }
        Ok(())
    }
}

/// Text bound for the user, held back while it could still be a Hermes XML
/// tool call. Some models write tool calls into `content`; those must never
/// reach the user as prose.
#[derive(Default)]
struct VisibleText {
    /// `Some(true)` forwards text, `Some(false)` suppresses a tool call.
    forward: Option<bool>,
    held: String,
    pending: String,
    last_emit: Option<std::time::Instant>,
}

impl VisibleText {
    const TOOL_MARKERS: [&'static str; 2] = ["<tool_call", "<function="];

    fn push(&mut self, text: &str) {
        match self.forward {
            Some(true) => self.pending.push_str(text),
            Some(false) => {}
            None => {
                self.held.push_str(text);
                let start = self.held.trim_start();
                if start.is_empty() {
                    return;
                }
                if Self::TOOL_MARKERS.iter().any(|m| start.starts_with(m)) {
                    self.forward = Some(false);
                    self.held.clear();
                } else if !Self::TOOL_MARKERS.iter().any(|m| m.starts_with(start)) {
                    self.forward = Some(true);
                    self.pending = std::mem::take(&mut self.held);
                }
            }
        }
    }

    fn flush(&mut self, sink: &dyn TextStreamSink, force: bool) {
        if force && self.forward.is_none() {
            // The reply ended while still ambiguous, so it was never a tool
            // call.
            self.forward = Some(true);
            self.pending = std::mem::take(&mut self.held);
        }
        if self.pending.is_empty()
            || (!force
                && self
                    .last_emit
                    .is_some_and(|last| last.elapsed() < RESPONSES_TEXT_FLUSH_INTERVAL))
        {
            return;
        }
        sink.delta(&std::mem::take(&mut self.pending));
        self.last_emit = Some(std::time::Instant::now());
    }
}

/// A tool call assembled from streamed fragments.
#[derive(Default)]
struct StreamedCall {
    id: Option<String>,
    name: String,
    arguments: String,
    extra_content: Option<Value>,
}

/// Incremental decoder for a Chat Completions event stream.
#[derive(Default)]
struct ChatStreamAccumulator {
    pending: Vec<u8>,
    done: bool,
    events: usize,
    model: Option<String>,
    usage: Option<Value>,
    finish_reason: Option<String>,
    content: String,
    refusal: String,
    calls: Vec<StreamedCall>,
    visible: VisibleText,
}

impl ChatStreamAccumulator {
    fn push(&mut self, chunk: &[u8], sink: &dyn TextStreamSink) -> Result<()> {
        self.pending.extend_from_slice(chunk);
        while let Some((at, separator_len)) = sse_frame_boundary(&self.pending) {
            let frame = self.pending[..at].to_vec();
            self.pending.drain(..at + separator_len);
            self.consume_frame(&frame)?;
        }
        if self.pending.len() > RESPONSES_MAX_EVENT_BYTES {
            return Err(Error::transient(
                "openai-compat stream contained an oversized event",
            ));
        }
        self.visible.flush(sink, false);
        Ok(())
    }

    fn finish(&mut self, sink: &dyn TextStreamSink) -> Result<()> {
        if !self.pending.iter().all(u8::is_ascii_whitespace) {
            let frame = std::mem::take(&mut self.pending);
            self.consume_frame(&frame)?;
        }
        self.visible.flush(sink, true);
        Ok(())
    }

    fn consume_frame(&mut self, frame: &[u8]) -> Result<()> {
        let frame = std::str::from_utf8(frame).map_err(|error| {
            Error::transient(format!("openai-compat stream was not UTF-8: {error}"))
        })?;
        let data = frame
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(|line| line.strip_prefix(' ').unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n");
        if data.is_empty() {
            return Ok(());
        }
        if data == "[DONE]" {
            self.done = true;
            return Ok(());
        }
        let event: Value = serde_json::from_str(&data)
            .map_err(|error| Error::transient(format!("openai-compat stream bad json: {error}")))?;
        self.events += 1;
        if let Some(error) = embedded_error(&event, None) {
            return Err(error);
        }
        if let Some(model) = event["model"].as_str() {
            self.model = Some(model.to_string());
        }
        if event.get("usage").is_some_and(|usage| !usage.is_null()) {
            self.usage = Some(event["usage"].clone());
        }
        let Some(choice) = event["choices"].get(0) else {
            return Ok(());
        };
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.finish_reason = Some(reason.to_string());
        }
        let delta = &choice["delta"];
        if let Some(text) = delta["content"].as_str() {
            self.content.push_str(text);
            self.visible.push(text);
        }
        if let Some(refusal) = delta["refusal"].as_str() {
            self.refusal.push_str(refusal);
        }
        for (position, fragment) in delta["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
        {
            let index = fragment["index"].as_u64().map_or(position, |i| i as usize);
            if self.calls.len() <= index {
                self.calls.resize_with(index + 1, StreamedCall::default);
            }
            let call = &mut self.calls[index];
            if let Some(id) = fragment["id"].as_str().filter(|id| !id.is_empty()) {
                call.id = Some(id.to_string());
            }
            if let Some(name) = fragment["function"]["name"].as_str() {
                call.name.push_str(name);
            }
            if fragment["extra_content"].is_object() {
                call.extra_content = Some(fragment["extra_content"].clone());
            }
            match &fragment["function"]["arguments"] {
                Value::String(part) => call.arguments.push_str(part),
                Value::Null => {}
                whole => call.arguments = whole.to_string(),
            }
        }
        Ok(())
    }

    /// The buffered-response shape of everything received.
    fn payload(&self, requested_model: &str) -> Value {
        let mut message = json!({
            "role": "assistant",
            "content": if self.content.is_empty() { Value::Null } else { json!(self.content) },
        });
        if !self.refusal.is_empty() {
            message["refusal"] = json!(self.refusal);
        }
        let calls = self
            .calls
            .iter()
            .filter(|call| !call.name.is_empty())
            .map(|call| {
                let mut value = json!({
                    "id": call.id,
                    "type": "function",
                    "function": { "name": call.name, "arguments": call.arguments },
                });
                if let Some(extra) = &call.extra_content {
                    value["extra_content"] = extra.clone();
                }
                value
            })
            .collect::<Vec<_>>();
        if !calls.is_empty() {
            message["tool_calls"] = json!(calls);
        }
        let mut payload = json!({
            "model": self.model.as_deref().unwrap_or(requested_model),
            "choices": [{ "finish_reason": self.finish_reason, "message": message }],
        });
        if let Some(usage) = &self.usage {
            payload["usage"] = usage.clone();
        }
        payload
    }
}

fn sse_frame_boundary(bytes: &[u8]) -> Option<(usize, usize)> {
    let lf = bytes
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|at| (at, 2));
    let crlf = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|at| (at, 4));
    match (lf, crlf) {
        (Some(left), Some(right)) => Some(if left.0 < right.0 { left } else { right }),
        (some @ Some(_), None) | (None, some @ Some(_)) => some,
        (None, None) => None,
    }
}

fn responses_text(payload: &Value) -> String {
    if let Some(text) = payload["output_text"]
        .as_str()
        .filter(|text| !text.trim().is_empty())
    {
        return text.trim().to_string();
    }
    payload["output"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|item| item["type"] == "message")
        .flat_map(|item| item["content"].as_array().into_iter().flatten())
        .filter(|part| part["type"] == "output_text")
        .filter_map(|part| part["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn interpret_responses_payload(payload: &Value) -> ChatStep {
    use crate::{ChatStep, ToolCall};

    let mut calls = Vec::new();
    let mut used_ids = std::collections::HashSet::new();
    for (index, item) in payload["output"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        if item["type"] != "function_call" {
            continue;
        }
        let Some(name) = item["name"].as_str().filter(|name| !name.is_empty()) else {
            continue;
        };
        let raw = &item["arguments"];
        let arguments = match raw {
            Value::String(value) if value.trim().is_empty() => json!({}),
            Value::String(value) => serde_json::from_str(value).unwrap_or_else(|_| raw.clone()),
            Value::Null => json!({}),
            other => other.clone(),
        };
        let base = item["call_id"]
            .as_str()
            .or_else(|| item["id"].as_str())
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("call_{name}_{index}"));
        let mut id = base.clone();
        let mut suffix = 1usize;
        while !used_ids.insert(id.clone()) {
            id = format!("{base}_{suffix}");
            suffix += 1;
        }
        calls.push(ToolCall {
            id,
            name: name.to_string(),
            arguments,
        });
    }
    if calls.is_empty() {
        ChatStep::Text(responses_text(payload))
    } else {
        ChatStep::ToolCalls(calls)
    }
}

/// Wrap a provider-neutral tool spec ({name, description, parameters}) into an
/// OpenAI/OpenRouter `function` tool. The schema MUST land under
/// `function.parameters` — the key the OpenAI tool-calling contract reads it
/// from. (Shipping it under any other key, such as Anthropic's `input_schema`,
/// leaves the model with no declared parameters and forces it to guess argument
/// names.) A spec without a schema still gets a valid
/// empty object schema so the function is well-formed.
fn oai_function(spec: &Value) -> Value {
    let parameters = match spec.get("parameters") {
        Some(s) if !s.is_null() => s.clone(),
        _ => json!({ "type": "object", "properties": {} }),
    };
    json!({
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec["description"],
            "parameters": parameters,
        }
    })
}

/// The Chat Completions wire form of a provider-neutral tool spec
/// (`{name, description, parameters}`), as sent in the `tools` array. Exposed
/// so hosts can assert their tool schemas survive the provider boundary.
pub fn chat_completions_tool(spec: &Value) -> Value {
    oai_function(spec)
}

fn interpret_tool_message(msg: &Value) -> (ChatStep, ToolParse) {
    use crate::{ChatStep, ToolCall};
    let content = msg["content"].as_str().unwrap_or("");
    if let Some(tcs) = msg["tool_calls"].as_array().filter(|a| !a.is_empty()) {
        // Count named entries (an entry with no name is not a call the model made,
        // just filler) so a truncation-drop shows up as survivors < named.
        let named = tcs
            .iter()
            .filter(|c| !c["function"]["name"].as_str().unwrap_or("").is_empty())
            .count();
        let mut calls = Vec::with_capacity(named);
        let mut malformed = 0usize;
        let mut used_ids = std::collections::HashSet::new();
        for (i, c) in tcs.iter().enumerate() {
            let name = c["function"]["name"].as_str().unwrap_or("");
            if name.is_empty() {
                continue;
            }
            // OpenAI specifies a JSON-encoded string. A few compatible providers
            // return the object directly, which is unambiguous and safe to accept.
            // Preserve any malformed value verbatim instead of dropping the call:
            // tools take an object, so the host's tool boundary
            // will reject it and return a paired, actionable validation result.
            let raw_arguments = &c["function"]["arguments"];
            let (arguments, invalid) = match raw_arguments {
                Value::String(raw) if raw.trim().is_empty() => {
                    (Value::Object(Default::default()), false)
                }
                Value::String(raw) => match serde_json::from_str(raw) {
                    Ok(arguments) => (arguments, false),
                    Err(_) => (Value::String(raw.clone()), true),
                },
                Value::Object(_) => (raw_arguments.clone(), false),
                Value::Null => (Value::Object(Default::default()), false),
                other => (other.clone(), true),
            };
            malformed += usize::from(invalid);

            // Provider IDs are untrusted transport data. Missing or duplicate IDs
            // would make result rows ambiguous, so allocate a unique internal ID
            // while keeping the provider's first usable ID unchanged.
            let base_id = match c["id"].as_str().unwrap_or("") {
                "" => format!("call_{name}_{i}"),
                id => id.to_string(),
            };
            let mut id = base_id.clone();
            let mut suffix = 1usize;
            while !used_ids.insert(id.clone()) {
                id = format!("{base_id}_{suffix}");
                suffix += 1;
            }
            calls.push(ToolCall {
                id,
                name: name.to_string(),
                arguments,
            });
        }
        // ALL named calls parsed cleanly — this wins outright, content is never
        // consulted. (Precedence: clean native beats any XML in content.)
        if !calls.is_empty() && malformed == 0 {
            return (ChatStep::ToolCalls(calls), ToolParse::Native);
        }
        // Some (or all) named calls were MALFORMED (the OpenRouter arguments-
        // truncation bug). The complete calls may be the Hermes-XML in content, so
        // consult it. Prefer XML only when it recovers the COMPLETE named batch;
        // replacing two retained native calls with one XML call would lose a side
        // effect just as surely as dropping malformed native arguments did.
        let xml_recovered = parse_hermes_tool_calls(content);
        if let Some(xml_calls) = &xml_recovered
            && xml_calls.len() >= named
        {
            return (
                ChatStep::ToolCalls(xml_calls.clone()),
                ToolParse::XmlAfterMalformedNative,
            );
        }
        let parse = if malformed == calls.len() {
            ToolParse::InvalidNativeArguments
        } else {
            ToolParse::PartialInvalidNativeArguments
        };
        return (ChatStep::ToolCalls(calls), parse);
    }
    // No native tool_calls at all. The model may still have emitted a Hermes-XML
    // call in `content`; otherwise it's a plain text reply.
    if let Some(xml_calls) = parse_hermes_tool_calls(content) {
        return (ChatStep::ToolCalls(xml_calls), ToolParse::XmlNoNative);
    }
    (
        ChatStep::Text(content.trim().to_string()),
        ToolParse::Native,
    )
}

/// Parse Hermes / Qwen-style XML tool calls that open models emit in the message
/// `content` instead of (or alongside a broken copy of) the OpenAI `tool_calls`
/// field. The shape is:
///   <tool_call>
///   <function=water_bed>
///   <parameter=bed>4</parameter>
///   <parameter=when>today</parameter>
///   </function>
///   </tool_call>
/// Returns None when no `<tool_call>` block is present. Best-effort and tolerant
/// of whitespace; values are taken verbatim and assembled into a JSON object.
fn parse_hermes_tool_calls(content: &str) -> Option<Vec<ToolCall>> {
    use ToolCall;
    if !content.contains("<tool_call>") {
        return None;
    }
    let mut calls = Vec::new();
    // Each <tool_call> ... </tool_call> block (tolerate a missing close at EOF).
    for block in content.split("<tool_call>").skip(1) {
        let block = block.split("</tool_call>").next().unwrap_or(block);
        // Function name: <function=NAME>
        let Some(name) = block
            .split_once("<function=")
            .and_then(|(_, rest)| rest.split_once('>').map(|(n, _)| n.trim()))
            .filter(|n| !n.is_empty())
        else {
            continue;
        };
        // Parameters: each <parameter=KEY>VALUE</parameter>. XML carries no type,
        // so EVERY value arrives as a string. We deliberately do NOT guess types
        // here — blindly upcasting "100" to a number would corrupt a string field
        // (a numeric-looking text field would read back empty). Instead we keep
        // every value verbatim as a string; the agent reconciles each to the
        // tool's declared type via coerce_args_to_schema at dispatch, which only
        // numifies a value when the schema asks for a number/bool.
        let mut args = serde_json::Map::new();
        for piece in block.split("<parameter=").skip(1) {
            let Some((key, after)) = piece.split_once('>') else {
                continue;
            };
            let value = after.split("</parameter>").next().unwrap_or("").trim();
            args.insert(key.trim().to_string(), Value::String(value.to_string()));
        }
        calls.push(ToolCall {
            // Open models rarely supply a call id; synthesize a stable-ish one so
            // the tool-result turn can reference it.
            id: format!("call_{}_{}", name, calls.len()),
            name: name.to_string(),
            arguments: Value::Object(args),
        });
    }
    if calls.is_empty() { None } else { Some(calls) }
}

#[cfg(test)]
mod host_tests;
#[cfg(test)]
mod tests;
