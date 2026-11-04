//! Anthropic request policies for OpenRouter's compatible APIs.
//!
//! Transport, response parsing, and billing stay in the shared adapter. These
//! policies change how already-permitted instructions and tools are presented.

use serde_json::{Value, json};

use crate::SystemPrompt;

/// Claude models that reject `temperature`: Opus 4.7 and later, every 5.x
/// model, and the `-latest` aliases, which point at them. Measured on
/// Anthropic's API on 2026-11-04; Opus 4.6 and Sonnet 4.6 still accept it.
pub(super) fn rejects_temperature(model: &str) -> bool {
    let name = model.rsplit('/').next().unwrap_or(model);
    let name = name.split(':').next().unwrap_or(name);
    let Some(rest) = name.strip_prefix("claude-") else {
        return false;
    };
    if rest.ends_with("-latest") {
        return true;
    }
    // claude-<family>-<major>[-.]<minor>[-<date>]
    let mut numbers = rest
        .split(['-', '.'])
        .filter_map(|part| part.parse::<u32>().ok())
        .filter(|number| *number < 100);
    let Some(major) = numbers.next() else {
        return false;
    };
    let minor = numbers.next().unwrap_or(0);
    (major, minor) >= (4, 7)
}

pub(super) fn is_model(model: &str) -> bool {
    model.starts_with("anthropic/") || model.starts_with("~anthropic/")
}

/// Cache and catalogue policies follow the selected request model. Sampling
/// restrictions cover every server-side candidate, since all share one body.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RequestPolicy {
    enabled: bool,
    omit_temperature: bool,
}

impl RequestPolicy {
    pub(super) fn for_models(selected: &str, fallbacks: &[String]) -> Self {
        let candidates = std::iter::once(selected).chain(fallbacks.iter().map(String::as_str));
        let mut policy = Self {
            enabled: is_model(selected),
            omit_temperature: false,
        };
        for model in candidates.filter(|model| is_model(model)) {
            policy.omit_temperature |= rejects_temperature(model);
        }
        policy
    }

    pub(super) fn apply_parameters(self, body: &mut Value) {
        if self.enabled {
            body["cache_control"] = json!({"type": "ephemeral"});
        }
        if self.omit_temperature {
            // Opus 4.7 and later, and every 5.x model, reject any temperature.
            body.as_object_mut().unwrap().remove("temperature");
        }
    }

    pub(super) fn prepare_tools(self, tools: &mut [Value]) {
        if self.enabled {
            // An upstream trace showed OpenRouter Tool Search dropping explicit
            // cache markers on Google-hosted Haiku. Send the full catalogue
            // until that path preserves them. Never add a tool here:
            // the host has already filtered this list for authority/prerequisites.
            for tool in tools {
                tool.as_object_mut().unwrap().remove("defer_loading");
            }
        }
    }

    /// Responses cannot mark part of root instructions. Keep the fixed prefix
    /// and dynamic suffix in ordered system text blocks before the history.
    pub(super) fn apply_responses_instructions(self, body: &mut Value, system: SystemPrompt<'_>) {
        if !self.enabled {
            return;
        }
        let Some((prefix, suffix)) = system.cacheable_prefix() else {
            return;
        };
        let mut blocks = vec![json!({
            "type": "input_text", "text": prefix,
            "prompt_cache_breakpoint": {"mode": "explicit"},
        })];
        if !suffix.is_empty() {
            blocks.push(json!({"type": "input_text", "text": suffix}));
        }
        body["input"].as_array_mut().unwrap().insert(
            0,
            json!({
                "type": "message", "role": "system", "content": blocks,
            }),
        );
        body.as_object_mut().unwrap().remove("instructions");
    }

    pub(super) fn chat_cache_blocks(self, system: SystemPrompt<'_>) -> Option<Vec<Value>> {
        if !self.enabled {
            return None;
        }
        let (prefix, suffix) = system.cacheable_prefix()?;
        let mut blocks = vec![json!({
            "type": "text", "text": prefix,
            "cache_control": {"type": "ephemeral"},
        })];
        if !suffix.is_empty() {
            blocks.push(json!({"type": "text", "text": suffix}));
        }
        Some(blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_anthropic_candidates_change_the_request_policy() {
        let tools = vec![json!({"type":"function", "name":"permitted_read", "defer_loading":true})];
        for model in [
            "z-ai/glm-5.3-flash",
            "openai/gpt-5.4",
            "other/anthropic-lookalike",
        ] {
            let policy = RequestPolicy::for_models(model, &[]);
            let mut body = json!({"temperature":0, "instructions":"fixed", "input":[]});
            let original = body.clone();
            policy.apply_parameters(&mut body);
            policy.apply_responses_instructions(
                &mut body,
                SystemPrompt::with_cacheable_prefix("fixed", 5),
            );
            assert_eq!(body, original);
            let mut actual_tools = tools.clone();
            policy.prepare_tools(&mut actual_tools);
            assert_eq!(actual_tools, tools);
        }
    }

    #[test]
    fn eager_catalogue_preserves_exactly_the_host_permitted_tools_and_order() {
        let mut tools = vec![
            json!({"type":"function", "name":"common", "parameters":{}}),
            json!({"type":"function", "name":"specialist", "parameters":{}, "defer_loading":true}),
        ];
        RequestPolicy::for_models("anthropic/claude-haiku-5.5", &[]).prepare_tools(&mut tools);
        assert_eq!(
            tools,
            vec![
                json!({"type":"function", "name":"common", "parameters":{}}),
                json!({"type":"function", "name":"specialist", "parameters":{}}),
            ]
        );
    }

    #[test]
    fn haiku_fallback_sampling_does_not_change_a_glm_catalogue_or_instruction_format() {
        for (selected, fallbacks) in [
            (
                "z-ai/glm-5.3-flash",
                vec!["anthropic/claude-haiku-5.5".into()],
            ),
            (
                "anthropic/claude-sonnet-5.5",
                vec![
                    "z-ai/glm-5.3-flash".into(),
                    "~anthropic/claude-haiku-5.5".into(),
                ],
            ),
        ] {
            let policy = RequestPolicy::for_models(selected, &fallbacks);
            let mut body = json!({"temperature":0});
            policy.apply_parameters(&mut body);
            assert!(body.get("temperature").is_none());
            assert_eq!(body.get("cache_control").is_some(), is_model(selected));
            let mut tools = vec![json!({"name":"already_permitted", "defer_loading":true})];
            policy.prepare_tools(&mut tools);
            assert_eq!(tools[0].get("defer_loading").is_none(), is_model(selected));
            let mut instructions = json!({"instructions":"fixed\ndynamic", "input":[]});
            policy.apply_responses_instructions(
                &mut instructions,
                SystemPrompt::with_cacheable_prefix("fixed\ndynamic", 5),
            );
            assert_eq!(
                instructions.get("instructions").is_none(),
                is_model(selected)
            );
        }
    }

    #[test]
    fn haiku_sampling_covers_explicit_versions_aliases_and_routing_variants() {
        for model in [
            "anthropic/claude-haiku-5.5:nitro",
            "~anthropic/claude-haiku-5-5",
            "~anthropic/claude-haiku-latest",
            "anthropic/claude-haiku-latest:floor",
        ] {
            let policy = RequestPolicy::for_models("z-ai/glm-5.3-flash", &[model.into()]);
            let mut body = json!({"temperature":0});
            policy.apply_parameters(&mut body);
            assert!(body.get("temperature").is_none(), "{model}");
            assert!(body.get("cache_control").is_none());
        }
    }

    #[test]
    fn temperature_is_rejected_from_opus_4_7_on() {
        for model in [
            "claude-haiku-5-5",
            "anthropic/claude-sonnet-5.5",
            "claude-opus-4-7",
            "claude-fable-5-1",
            "~anthropic/claude-sonnet-latest",
            "claude-opus-5:thinking",
        ] {
            assert!(rejects_temperature(model), "{model}");
        }
        for model in [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-opus-4-5-20251101",
            "gpt-6.1-sol",
        ] {
            assert!(!rejects_temperature(model), "{model}");
        }
    }

    #[test]
    fn older_claude_retains_sampling_and_unmarked_instructions_remain_intact() {
        // Sonnet 4.6 still accepts temperature; Sonnet 5.5 (and the `-latest`
        // alias pointing at it) rejects it on Anthropic's API.
        let policy = RequestPolicy::for_models("anthropic/claude-sonnet-4.6", &[]);
        let mut body = json!({"temperature":0, "instructions":"unmarked", "input":[]});
        policy.apply_parameters(&mut body);
        policy.apply_responses_instructions(&mut body, "unmarked".into());
        assert_eq!(body["temperature"], 0);
        let mut latest = json!({"temperature":0});
        RequestPolicy::for_models("~anthropic/claude-sonnet-latest", &[])
            .apply_parameters(&mut latest);
        assert!(latest.get("temperature").is_none());
        assert_eq!(body["instructions"], "unmarked");
        assert_eq!(body["input"], json!([]));
    }
}
