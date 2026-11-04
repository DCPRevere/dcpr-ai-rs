//! A fallback chain of models.
//!
//! Tries models in order. Each model gets one retry of a transient failure
//! (429, 5xx, timeout) after a bounded, jittered backoff. A hard failure (401,
//! 400, model not found) or an exhausted retry falls through to the next model.
//! A context overflow fails fast: the same prompt would overflow everywhere.

use async_trait::async_trait;
use rand::RngExt;
use std::sync::Arc;

use crate::{
    ChatModel, ChatStep, Completion, Error, Result, SystemPrompt, TextStreamSink, ToolTurn,
};

/// Retry budget for transient failures, per model.
const MAX_RETRIES: u32 = 1;
/// Base backoff. Rate limits rarely clear in under a second, so the base is
/// seconds-scale. Each attempt doubles the ceiling, and the sleep is a full
/// jittered draw within it.
const BASE_DELAY_MS: u64 = 1000;
/// Cap on any single backoff, so a generous `Retry-After` cannot hang a turn.
const MAX_DELAY_MS: u64 = 30_000;

/// Honour the provider's `Retry-After` when present (capped). Otherwise draw
/// uniformly from an exponentially growing window, which de-correlates
/// concurrent retries. `attempt` is 1-based.
fn backoff_delay_ms(attempt: u32, retry_after_secs: Option<u64>) -> u64 {
    if let Some(secs) = retry_after_secs {
        return (secs.saturating_mul(1000)).min(MAX_DELAY_MS);
    }
    let ceiling = BASE_DELAY_MS
        .saturating_mul(1u64 << (attempt - 1))
        .min(MAX_DELAY_MS);
    rand::rng().random_range(0..=ceiling)
}

/// How an error steers the fallback chain. Implemented for [`Error`];
/// applications implement it for their own error types so
/// [`Fallback::run`] can drive their own operations.
pub trait Retryable {
    fn is_retryable(&self) -> bool;
    fn is_context_overflow(&self) -> bool;
    fn retry_after_secs(&self) -> Option<u64>;
}

impl Retryable for Error {
    fn is_retryable(&self) -> bool {
        Error::is_retryable(self)
    }
    fn is_context_overflow(&self) -> bool {
        Error::is_context_overflow(self)
    }
    fn retry_after_secs(&self) -> Option<u64> {
        Error::retry_after_secs(self)
    }
}

/// Observes the outcome of each top-level operation, for example to alert an
/// operator during a sustained outage and clear the alert on recovery.
#[async_trait]
pub trait ChainHealth: Send + Sync {
    async fn record_success(&self);
    /// Every model in the chain failed this operation.
    async fn record_failure(&self, error: &str);
}

/// A chain of models tried in order. `M` is usually a trait object such as
/// `dyn ChatModel<C>` or an application trait that extends it.
pub struct Fallback<M: ?Sized> {
    chain: Vec<Arc<M>>,
    sleep_between_retries: bool,
    health: Option<Arc<dyn ChainHealth>>,
}

impl<M: ?Sized + Send + Sync> Fallback<M> {
    pub fn new(chain: Vec<Arc<M>>) -> Self {
        Self {
            chain,
            sleep_between_retries: true,
            health: None,
        }
    }

    /// Report each operation's outcome to `health`.
    pub fn with_health(mut self, health: Arc<dyn ChainHealth>) -> Self {
        self.health = Some(health);
        self
    }

    /// Retry immediately instead of sleeping. Intended for tests.
    pub fn without_retry_sleep(mut self) -> Self {
        self.sleep_between_retries = false;
        self
    }

    pub fn models(&self) -> &[Arc<M>] {
        &self.chain
    }

    /// Run `op` against each model in turn, with retry and fallthrough. `name`
    /// labels a model in logs.
    pub async fn run<T, E, F, Fut>(
        &self,
        name: impl Fn(&M) -> String,
        op: F,
    ) -> std::result::Result<T, E>
    where
        E: Retryable + std::fmt::Display + From<Error>,
        F: Fn(Arc<M>) -> Fut,
        Fut: std::future::Future<Output = std::result::Result<T, E>>,
    {
        self.run_over(self.chain.iter(), "no models configured", name, op)
            .await
    }

    /// The retry, backoff and fallthrough engine over `models`.
    pub async fn run_over<'a, T, E, F, Fut, I>(
        &self,
        models: I,
        exhausted: &'static str,
        name: impl Fn(&M) -> String,
        op: F,
    ) -> std::result::Result<T, E>
    where
        M: 'a,
        I: Iterator<Item = &'a Arc<M>>,
        E: Retryable + std::fmt::Display + From<Error>,
        F: Fn(Arc<M>) -> Fut,
        Fut: std::future::Future<Output = std::result::Result<T, E>>,
    {
        let mut last_err: Option<E> = None;
        for model in models {
            let label = name(model);
            let mut attempt = 0;
            loop {
                match op(model.clone()).await {
                    Ok(v) => {
                        if let Some(h) = &self.health {
                            h.record_success().await;
                        }
                        return Ok(v);
                    }
                    // The same oversized prompt would fail on every model, so
                    // neither retry nor fall through. The caller shrinks the
                    // prompt. Not counted as an outage.
                    Err(e) if e.is_context_overflow() => {
                        tracing::warn!(
                            model = %label,
                            "LLM context overflow: failing fast to caller (no retry, no fallthrough): {e}"
                        );
                        return Err(e);
                    }
                    Err(e) if e.is_retryable() && attempt < MAX_RETRIES => {
                        attempt += 1;
                        let retry_after = e.retry_after_secs();
                        let delay = backoff_delay_ms(attempt, retry_after);
                        if self.sleep_between_retries {
                            tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                        }
                        tracing::warn!(
                            model = %label,
                            attempt,
                            delay_ms = delay,
                            retry_after_secs = retry_after,
                            "LLM transient failure, retrying: {e}"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            model = %label,
                            "LLM failure, falling through to next model: {e}"
                        );
                        last_err = Some(e);
                        break;
                    }
                }
            }
        }
        let err = last_err.unwrap_or_else(|| Error::hard(exhausted).into());
        if let Some(h) = &self.health {
            h.record_failure(&err.to_string()).await;
        }
        Err(err)
    }
}

#[async_trait]
impl<C, M> ChatModel<C> for Fallback<M>
where
    C: ?Sized + Sync,
    M: ChatModel<C> + ?Sized,
{
    fn model_id(&self) -> &str {
        self.chain
            .first()
            .map(|m| m.model_id())
            .unwrap_or("fallback-empty")
    }

    fn supports_tools(&self) -> bool {
        self.chain.iter().any(|m| m.supports_tools())
    }

    async fn chat_step(
        &self,
        caller: &C,
        system: SystemPrompt<'_>,
        history: &[ToolTurn],
        tools: &[serde_json::Value],
    ) -> Result<ChatStep> {
        self.run_over(
            self.chain.iter().filter(|m| m.supports_tools()),
            "no tool-capable models",
            |m| m.model_id().to_string(),
            |m| async move { m.chat_step(caller, system, history, tools).await },
        )
        .await
    }

    async fn chat_step_stream(
        &self,
        caller: &C,
        system: SystemPrompt<'_>,
        history: &[ToolTurn],
        tools: &[serde_json::Value],
        sink: Arc<dyn TextStreamSink>,
    ) -> Result<ChatStep> {
        self.run_over(
            self.chain.iter().filter(|m| m.supports_tools()),
            "no tool-capable models",
            |m| m.model_id().to_string(),
            |m| {
                let sink = sink.clone();
                async move {
                    // A failed model may have streamed a partial reply.
                    sink.reset();
                    m.chat_step_stream(caller, system, history, tools, sink)
                        .await
                }
            },
        )
        .await
    }

    async fn complete(&self, caller: &C, request: Completion<'_>) -> Result<String> {
        self.run(
            |m| m.model_id().to_string(),
            |m| async move { m.complete(caller, request).await },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct Stub {
        id: String,
        calls: Arc<AtomicU32>,
        fail_times: u32,
        retryable: bool,
        succeeds_eventually: bool,
    }

    #[async_trait]
    impl ChatModel for Stub {
        fn model_id(&self) -> &str {
            &self.id
        }
        async fn chat_step(
            &self,
            _: &str,
            _: SystemPrompt<'_>,
            _: &[ToolTurn],
            _: &[serde_json::Value],
        ) -> Result<ChatStep> {
            unreachable!()
        }
        async fn complete(&self, _: &str, _: Completion<'_>) -> Result<String> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_times {
                return Err(if self.retryable {
                    Error::transient(format!("{} transient", self.id))
                } else {
                    Error::hard(format!("{} hard", self.id))
                });
            }
            if self.succeeds_eventually {
                Ok(self.id.clone())
            } else {
                Err(Error::hard("never succeeds"))
            }
        }
    }

    fn stub(
        id: &str,
        calls: &Arc<AtomicU32>,
        fail_times: u32,
        retryable: bool,
        succeeds: bool,
    ) -> Arc<dyn ChatModel> {
        Arc::new(Stub {
            id: id.into(),
            calls: calls.clone(),
            fail_times,
            retryable,
            succeeds_eventually: succeeds,
        })
    }

    fn counter() -> Arc<AtomicU32> {
        Arc::new(AtomicU32::new(0))
    }

    fn ask() -> Completion<'static> {
        Completion::new("sys", "user", 10)
    }

    struct StreamingStub {
        id: &'static str,
        delta: &'static str,
        fails: bool,
    }

    #[async_trait]
    impl ChatModel for StreamingStub {
        fn model_id(&self) -> &str {
            self.id
        }
        async fn chat_step(
            &self,
            _: &str,
            _: SystemPrompt<'_>,
            _: &[ToolTurn],
            _: &[serde_json::Value],
        ) -> Result<ChatStep> {
            unreachable!()
        }
        async fn chat_step_stream(
            &self,
            _: &str,
            _: SystemPrompt<'_>,
            _: &[ToolTurn],
            _: &[serde_json::Value],
            sink: Arc<dyn TextStreamSink>,
        ) -> Result<ChatStep> {
            sink.delta(self.delta);
            if self.fails {
                Err(Error::hard("failed after a partial reply"))
            } else {
                Ok(ChatStep::Text(self.delta.into()))
            }
        }
    }

    #[derive(Default)]
    struct CurrentText(Mutex<String>);

    impl TextStreamSink for CurrentText {
        fn reset(&self) {
            self.0.lock().unwrap().clear();
        }
        fn delta(&self, text: &str) {
            self.0.lock().unwrap().push_str(text);
        }
    }

    #[tokio::test]
    async fn fallback_clears_a_failed_partial_reply() {
        let chain: Vec<Arc<dyn ChatModel>> = vec![
            Arc::new(StreamingStub {
                id: "failed",
                delta: "stale",
                fails: true,
            }),
            Arc::new(StreamingStub {
                id: "healthy",
                delta: "fresh",
                fails: false,
            }),
        ];
        let fallback = Fallback::new(chain).without_retry_sleep();
        let sink = Arc::new(CurrentText::default());
        let reply = fallback
            .chat_step_stream("caller", "sys".into(), &[], &[], sink.clone())
            .await
            .unwrap();
        assert!(matches!(reply, ChatStep::Text(ref text) if text == "fresh"));
        assert_eq!(sink.0.lock().unwrap().as_str(), "fresh");
    }

    #[test]
    fn retry_after_wins_over_exponential_backoff_and_is_capped() {
        assert_eq!(backoff_delay_ms(1, Some(3)), 3_000);
        assert_eq!(backoff_delay_ms(1, Some(600)), MAX_DELAY_MS);
    }

    #[test]
    fn jittered_backoff_stays_within_the_exponential_ceiling() {
        for attempt in 1..=6 {
            let ceiling = (BASE_DELAY_MS * (1u64 << (attempt - 1))).min(MAX_DELAY_MS);
            for _ in 0..100 {
                let d = backoff_delay_ms(attempt, None);
                assert!(
                    d <= ceiling,
                    "attempt {attempt}: {d} exceeds ceiling {ceiling}"
                );
            }
        }
        const _: () = assert!(BASE_DELAY_MS >= 1000);
    }

    #[tokio::test]
    async fn retries_transient_then_succeeds_on_same_model() {
        let c = counter();
        let f = Fallback::new(vec![stub("p1", &c, 1, true, true)]).without_retry_sleep();
        assert_eq!(f.complete("caller", ask()).await.unwrap(), "p1");
        assert_eq!(c.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn hard_failure_falls_through_immediately() {
        let (c1, c2) = (counter(), counter());
        let f = Fallback::new(vec![
            stub("p1", &c1, 1, false, true),
            stub("p2", &c2, 0, true, true),
        ])
        .without_retry_sleep();
        assert_eq!(f.complete("caller", ask()).await.unwrap(), "p2");
        assert_eq!(c1.load(Ordering::SeqCst), 1);
        assert_eq!(c2.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn exhausted_transient_falls_through_to_next() {
        let (c1, c2) = (counter(), counter());
        let f = Fallback::new(vec![
            stub("p1", &c1, 99, true, false),
            stub("p2", &c2, 0, true, true),
        ])
        .without_retry_sleep();
        f.complete("caller", ask()).await.unwrap();
        assert_eq!(c1.load(Ordering::SeqCst), 1 + MAX_RETRIES);
        assert_eq!(c2.load(Ordering::SeqCst), 1);
    }

    #[derive(Default)]
    struct Health {
        successes: AtomicU32,
        failures: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl ChainHealth for Health {
        async fn record_success(&self) {
            self.successes.fetch_add(1, Ordering::SeqCst);
        }
        async fn record_failure(&self, error: &str) {
            self.failures.lock().unwrap().push(error.to_string());
        }
    }

    #[tokio::test]
    async fn health_sees_each_operation_outcome_once() {
        let health = Arc::new(Health::default());
        let dead = Fallback::new(vec![stub("dead", &counter(), 99, false, false)])
            .without_retry_sleep()
            .with_health(health.clone());
        assert!(dead.complete("caller", ask()).await.is_err());
        assert_eq!(health.failures.lock().unwrap().len(), 1);
        let good = Fallback::new(vec![stub("good", &counter(), 1, true, true)])
            .without_retry_sleep()
            .with_health(health.clone());
        good.complete("caller", ask()).await.unwrap();
        assert_eq!(health.successes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn context_overflow_fails_fast_without_fallthrough() {
        struct Overflow(Arc<AtomicU32>);
        #[async_trait]
        impl ChatModel for Overflow {
            fn model_id(&self) -> &str {
                "overflow"
            }
            async fn chat_step(
                &self,
                _: &str,
                _: SystemPrompt<'_>,
                _: &[ToolTurn],
                _: &[serde_json::Value],
            ) -> Result<ChatStep> {
                unreachable!()
            }
            async fn complete(&self, _: &str, _: Completion<'_>) -> Result<String> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Err(Error::from_status(
                    400,
                    "maximum context length is 8192 tokens",
                ))
            }
        }
        let (c1, c2) = (counter(), counter());
        let chain: Vec<Arc<dyn ChatModel>> = vec![
            Arc::new(Overflow(c1.clone())),
            stub("p2", &c2, 0, true, true),
        ];
        let f = Fallback::new(chain).without_retry_sleep();
        let err = f.complete("caller", ask()).await.unwrap_err();
        assert!(err.is_context_overflow(), "{err}");
        assert_eq!(c1.load(Ordering::SeqCst), 1);
        assert_eq!(c2.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn errors_only_when_all_exhausted() {
        let f = Fallback::new(vec![
            stub("p1", &counter(), 99, false, false),
            stub("p2", &counter(), 99, false, false),
        ])
        .without_retry_sleep();
        assert!(f.complete("caller", ask()).await.is_err());
        let empty: Fallback<dyn ChatModel> = Fallback::new(vec![]);
        assert!(empty.complete("caller", ask()).await.is_err());
    }
}
