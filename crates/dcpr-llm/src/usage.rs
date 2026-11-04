//! Token usage reporting.

use serde::{Deserialize, Serialize};

/// Token usage from a single model call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Uncached input tokens.
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Cache-read tokens (Anthropic `cache_read_input_tokens`, OpenAI and
    /// OpenRouter `prompt_tokens_details.cached_tokens`, DeepSeek
    /// `prompt_cache_hit_tokens`). Counted separately from `input_tokens`,
    /// because they are billed at a discount.
    pub cache_read_tokens: u64,
    /// Tokens written to the prompt cache, already included in `input_tokens`.
    #[serde(default)]
    pub cache_write_tokens: u64,
}

/// Receives the usage of every successful model call, tagged with its caller
/// and the model that actually ran (which may be a server-side fallback).
pub trait UsageSink<C: ?Sized>: Send + Sync {
    /// `reported_cost_cents` is the provider's own cost in US cents when it
    /// reports one (OpenRouter's `usage.cost`, including any BYOK upstream
    /// charge).
    fn record(&self, caller: &C, model: &str, usage: Usage, reported_cost_cents: Option<f64>);
}

/// A sink that discards usage.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoUsage;

impl<C: ?Sized> UsageSink<C> for NoUsage {
    fn record(&self, _: &C, _: &str, _: Usage, _: Option<f64>) {}
}

/// OpenRouter's reported cost for one call, in US cents. A BYOK call reports
/// the upstream provider's charge separately; when that is missing the cost is
/// unknown and `None` is returned so a price table can be used instead.
pub fn openrouter_cost_cents(usage: &serde_json::Value) -> Option<f64> {
    let cost = usage["cost"].as_f64()?;
    let upstream = usage["cost_details"]["upstream_inference_cost"].as_f64();
    let usd = match (usage["is_byok"].as_bool(), upstream) {
        (Some(false), _) => cost,
        (Some(true), None) => return None,
        (_, upstream) => cost + upstream.unwrap_or(0.0),
    };
    Some(usd * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn byok_cost_includes_the_upstream_charge() {
        let usage = json!({"cost": 0.001, "is_byok": true, "cost_details": {"upstream_inference_cost": 0.02}});
        assert!((openrouter_cost_cents(&usage).unwrap() - 2.1).abs() < 1e-9);
        assert_eq!(
            openrouter_cost_cents(&json!({"cost": 0.001, "is_byok": true})),
            None
        );
        assert!(
            (openrouter_cost_cents(&json!({"cost": 0.5, "is_byok": false})).unwrap() - 50.0).abs()
                < 1e-9
        );
        assert_eq!(openrouter_cost_cents(&json!({})), None);
    }
}
