//! OpenAI model rules, shared by requests to `api.openai.com` and by OpenAI
//! models routed through OpenRouter.
//!
//! Measured against the live API on 2026-11-04:
//! - Reasoning models (o-series, gpt-5 and later) reject `max_tokens` and
//!   `temperature: 0`. `max_completion_tokens` works on every current model.
//! - Every reasoning model accepts effort `low`. `none` and `minimal` are each
//!   rejected by some of them (`gpt-6.1-sol` takes neither), and older models
//!   such as `gpt-4.1` reject `reasoning_effort` outright.
//! - From gpt-5.5 on, Chat Completions refuses function tools while
//!   reasoning is on, so tool steps belong on the Responses API.

/// The model name without an `openai/` routing prefix or `:variant` suffix.
fn base(model: &str) -> &str {
    let model = model.strip_prefix("openai/").unwrap_or(model);
    model.split(':').next().unwrap_or(model)
}

/// The leading major version of a `gpt-N…` name.
fn gpt_major(name: &str) -> Option<u32> {
    let digits: String = name
        .strip_prefix("gpt-")?
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

fn is_o_series(name: &str) -> bool {
    ["o1", "o3", "o4"]
        .iter()
        .any(|family| name == *family || name.starts_with(&format!("{family}-")))
}

/// An OpenAI model that reasons: the o-series, and gpt-5 and later except
/// the `-chat` variants.
pub(super) fn is_reasoning_model(model: &str) -> bool {
    let name = base(model);
    is_o_series(name)
        || (gpt_major(name).is_some_and(|major| major >= 5) && !name.contains("-chat"))
}

/// OpenAI reasoning models that accept neither effort `none` nor a request to
/// disable reasoning: the o-series, the gpt-5.0 family, and `gpt-6.1-sol`.
pub(super) fn reasoning_is_mandatory(model: &str) -> bool {
    let name = base(model);
    let gpt_5_0 = name == "gpt-5" || name.starts_with("gpt-5-");
    is_reasoning_model(model) && (is_o_series(name) || gpt_5_0 || name.starts_with("gpt-6.1-sol"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_models_are_recognised_with_and_without_a_routing_prefix() {
        for model in [
            "gpt-6.1-sol",
            "openai/gpt-6.1-sol",
            "gpt-5.5",
            "gpt-5-mini",
            "o3",
            "o4-mini",
            "openai/gpt-10:nitro",
        ] {
            assert!(is_reasoning_model(model), "{model}");
        }
        for model in [
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-5-chat-latest",
            "o1x",
            "anthropic/claude-haiku-5.5",
        ] {
            assert!(!is_reasoning_model(model), "{model}");
        }
    }

    #[test]
    fn mandatory_reasoning_covers_only_models_without_a_none_effort() {
        for model in ["gpt-6.1-sol", "openai/gpt-5-mini", "gpt-5", "o3-mini"] {
            assert!(reasoning_is_mandatory(model), "{model}");
        }
        for model in [
            "gpt-5.5",
            "gpt-6-luna",
            "openai/gpt-6.1-terra",
            "gpt-4.1-mini",
        ] {
            assert!(!reasoning_is_mandatory(model), "{model}");
        }
    }
}
