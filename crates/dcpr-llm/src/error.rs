//! Errors from model calls, classified for retry and fallback decisions.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// A failed model call.
#[derive(Debug, Clone, Error)]
pub enum Error {
    /// A transport or provider failure. `retryable` marks a transient fault
    /// (429, 408, 5xx, timeouts, overload, a temporary OpenRouter in-flight
    /// budget). `retry_after_secs` carries the provider's `Retry-After` when it
    /// sent one, so a retry can wait exactly as long as asked.
    #[error("llm error ({}): {message}", if *.retryable { "transient" } else { "hard" })]
    Call {
        message: String,
        retryable: bool,
        retry_after_secs: Option<u64>,
    },

    /// The prompt exceeded the model's context window. Retrying the same prompt
    /// always fails, but a smaller one can succeed, so callers should shrink the
    /// prompt rather than retry or fall through to another model.
    #[error("llm context overflow: {message}")]
    ContextOverflow { message: String },

    /// The provider answered with a response this client could not use, such
    /// as a 200 with no choices.
    #[error("llm error: {0}")]
    Protocol(String),

    /// The model stopped before producing a usable answer. The partial reply
    /// is discarded: fluent but truncated text silently drops the end of a
    /// request. Retryable unless the model refused.
    #[error("llm stopped ({reason}): {message}")]
    Stopped { reason: StopReason, message: String },
}

/// Why a model stopped without a usable answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StopReason {
    /// The output-token ceiling was reached. Raise it, or reduce reasoning.
    MaxTokens,
    /// A provider content filter stopped the reply.
    ContentFilter,
    /// The model declined to answer, for example under a structured-output
    /// schema.
    Refusal,
    /// The provider reported the reply incomplete for another reason.
    Incomplete,
}

impl StopReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MaxTokens => "max_tokens",
            Self::ContentFilter => "content_filter",
            Self::Refusal => "refusal",
            Self::Incomplete => "incomplete",
        }
    }
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Error {
    /// A transient failure worth retrying or falling through.
    pub fn transient(message: impl Into<String>) -> Self {
        Self::Call {
            message: message.into(),
            retryable: true,
            retry_after_secs: None,
        }
    }

    /// A hard failure. Do not retry the same model.
    pub fn hard(message: impl Into<String>) -> Self {
        Self::Call {
            message: message.into(),
            retryable: false,
            retry_after_secs: None,
        }
    }

    pub fn context_overflow(message: impl Into<String>) -> Self {
        Self::ContextOverflow {
            message: message.into(),
        }
    }

    /// The model stopped early for `reason`.
    pub fn stopped(reason: StopReason, message: impl Into<String>) -> Self {
        Self::Stopped {
            reason,
            message: message.into(),
        }
    }

    /// A transient fault, or an early stop that a retry or another model may
    /// avoid. A refusal is not retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Call { retryable, .. } => *retryable,
            Self::Stopped { reason, .. } => *reason != StopReason::Refusal,
            _ => false,
        }
    }

    /// Why the model stopped early, when it did.
    pub fn stop_reason(&self) -> Option<StopReason> {
        match self {
            Self::Stopped { reason, .. } => Some(*reason),
            _ => None,
        }
    }

    pub fn is_context_overflow(&self) -> bool {
        matches!(self, Self::ContextOverflow { .. })
    }

    /// The provider's `Retry-After` in seconds on a transient failure.
    pub fn retry_after_secs(&self) -> Option<u64> {
        match self {
            Self::Call {
                retry_after_secs, ..
            } => *retry_after_secs,
            _ => None,
        }
    }

    /// Classify an HTTP status into a retryable or hard error.
    pub fn from_status(status: u16, message: impl Into<String>) -> Self {
        Self::from_status_with_retry_after(status, message, None)
    }

    /// Like [`from_status`](Self::from_status), keeping the provider's parsed
    /// `Retry-After`.
    pub fn from_status_with_retry_after(
        status: u16,
        message: impl Into<String>,
        retry_after_secs: Option<u64>,
    ) -> Self {
        let message = message.into();
        // A context-window overflow arrives as a 400 naming the token or context
        // limit, or rarely as a 413. Restricting this to 400/413 keeps a 429 or
        // 5xx whose body mentions "context" retryable.
        if (status == 400 || status == 413) && is_context_overflow_body(&message) {
            return Self::ContextOverflow { message };
        }
        // OpenRouter uses 402 both for exhausted credit (hard) and for a
        // temporary in-flight budget cap while other requests are still running.
        // Only the latter is retryable. Its machine-readable `limit_source`
        // (`reason` in older payloads) is echoed in the message by the client.
        // The Responses API omits that metadata, so its fixed "in-flight
        // requests" wording is the remaining signal.
        let lower = message.to_ascii_lowercase();
        let temporary_in_flight_budget = status == 402
            && (lower.contains("openrouter_in_flight_budget")
                || lower.contains("in_flight_budget_exhausted")
                || lower.contains("in-flight requests"));
        // OpenAI sends spent credit as a 429 too. Retrying cannot help.
        let quota_exhausted = status == 429
            && (lower.contains("insufficient_quota") || lower.contains("credit_balance_exhausted"));
        let retryable = temporary_in_flight_budget
            || (status == 429 && !quota_exhausted)
            || status == 408
            || (500..=599).contains(&status);
        Self::Call {
            message,
            retryable,
            // A Retry-After is only meaningful on a retryable status. Dropping it
            // on a hard failure keeps it from stalling a fallthrough.
            retry_after_secs: if retryable { retry_after_secs } else { None },
        }
    }

    /// Append an OpenRouter generation id so a failed request can be found in
    /// the provider's logs.
    pub fn with_generation(self, generation: Option<&str>) -> Self {
        let Some(generation) = generation else {
            return self;
        };
        match self {
            Self::Call {
                message,
                retryable,
                retry_after_secs,
            } => Self::Call {
                message: format!("{message} (generation {generation})"),
                retryable,
                retry_after_secs,
            },
            Self::ContextOverflow { message } => Self::ContextOverflow {
                message: format!("{message} (generation {generation})"),
            },
            Self::Stopped { reason, message } => Self::Stopped {
                reason,
                message: format!("{message} (generation {generation})"),
            },
            other => other,
        }
    }
}

/// Whether a provider error body describes a context-window or token-limit
/// overflow rather than any other 4xx. Matches the phrasings used by the
/// OpenAI-compatible family (OpenRouter forwards upstream text), DeepSeek and
/// Anthropic. The match is deliberately broad: shrinking and retrying once is
/// safe on a false positive, whereas a miss leaves a large conversation failing
/// permanently.
pub fn is_context_overflow_body(body: &str) -> bool {
    let b = body.to_ascii_lowercase();
    const MARKERS: &[&str] = &[
        "context_length_exceeded",
        "context length exceeded",
        "maximum context length",
        "maximum context",
        "context window",
        "context_window_exceeded",
        "too many tokens",
        "too many input tokens",
        "reduce the length",
        "prompt is too long",
        "input is too long",
        "exceeds the maximum",
        "exceeds the context",
        "exceed context limit",
        "string too long",
    ];
    MARKERS.iter().any(|m| b.contains(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_bodies_classify_as_context_overflow() {
        let bodies = [
            "openai-compat 400 Bad Request: This model's maximum context length is \
             128000 tokens. However, your messages resulted in 190234 tokens. \
             Please reduce the length of the messages.",
            r#"{"error":{"code":"context_length_exceeded","message":"too many tokens"}}"#,
            "anthropic 400 Bad Request: invalid_request_error: prompt is too long: \
             250000 tokens > 200000 maximum",
            "provider 400: input is too long for the requested model",
        ];
        for b in bodies {
            let e = Error::from_status(400, b);
            assert!(e.is_context_overflow(), "should be context overflow: {b}");
            assert!(!e.is_retryable(), "overflow must not be retryable: {b}");
        }
    }

    #[test]
    fn ordinary_400_is_not_context_overflow() {
        let e = Error::from_status(400, "400 Bad Request: model `foo` not found");
        assert!(!e.is_context_overflow());
        assert!(!e.is_retryable());
    }

    #[test]
    fn transient_statuses_stay_retryable_even_mentioning_context() {
        for status in [429u16, 408, 500, 503, 529] {
            let e = Error::from_status(status, "upstream error: maximum context length reached");
            assert!(e.is_retryable(), "status {status} must stay retryable");
            assert!(
                !e.is_context_overflow(),
                "status {status} is not an overflow"
            );
        }
    }

    #[test]
    fn only_the_temporary_form_of_payment_required_is_retryable() {
        let temporary = Error::from_status(
            402,
            r#"{"reason":"in_flight_budget_exhausted","remedy_hint":"Retry after in-flight requests settle"}"#,
        );
        assert!(temporary.is_retryable());
        let documented = Error::from_status(402, "Payment required [openrouter_in_flight_budget]");
        assert!(documented.is_retryable());
        let responses = Error::from_status(
            402,
            "OpenRouter Responses 402 Payment Required: This request would exceed your available credits given your current in-flight requests. Retry after in-flight requests settle, or add credits.",
        );
        assert!(responses.is_retryable());
        assert!(!Error::from_status(402, "Insufficient credits").is_retryable());
        for source in ["openrouter_key_limit", "openrouter_credits"] {
            let hard = Error::from_status(402, format!("Payment required [{source}]"));
            assert!(!hard.is_retryable(), "{source}");
        }
    }

    #[test]
    fn retry_after_is_dropped_on_hard_failures() {
        assert_eq!(
            Error::from_status_with_retry_after(429, "slow down", Some(7)).retry_after_secs(),
            Some(7)
        );
        assert_eq!(
            Error::from_status_with_retry_after(401, "bad key", Some(7)).retry_after_secs(),
            None
        );
    }

    #[test]
    fn early_stops_are_retryable_except_refusals() {
        let truncated = Error::stopped(StopReason::MaxTokens, "cut off");
        assert!(truncated.is_retryable());
        assert_eq!(truncated.stop_reason(), Some(StopReason::MaxTokens));
        assert_eq!(truncated.to_string(), "llm stopped (max_tokens): cut off");
        assert!(!Error::stopped(StopReason::Refusal, "declined").is_retryable());
        assert_eq!(Error::hard("x").stop_reason(), None);
    }

    #[test]
    fn overflow_body_matcher_is_case_insensitive_and_specific() {
        assert!(is_context_overflow_body("CONTEXT_LENGTH_EXCEEDED"));
        assert!(is_context_overflow_body("Maximum Context Length is 8192"));
        assert!(!is_context_overflow_body("rate limit exceeded"));
        assert!(!is_context_overflow_body("invalid api key"));
    }
}
