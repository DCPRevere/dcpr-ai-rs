//! Anthropic request policies for OpenRouter's compatible APIs.
//!
//! Transport, response parsing, and billing stay in the shared adapter. These
//! policies change how already-permitted instructions and tools are presented.

use serde_json::{Value, json};

use crate::SystemPrompt;

/// Haiku 5.5 rejects non-default sampling parameters.
fn is_haiku_5_5(model: &str) -> bool {
    model.contains("haiku-5.5") || model.contains("haiku-5-5")
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
            policy.omit_temperature |= is_haiku_5_5(model)
                || model
                    .split(':')
                    .next()
                    .is_some_and(|name| name.ends_with("/claude-haiku-latest"));
        }
        policy
    }

    pub(super) fn apply_parameters(self, body: &mut Value) {
        if self.enabled {
            body["cache_control"] = json!({"type": "ephemeral"});
        }
        if self.omit_temperature {
            // Haiku 5.5 rejects non-default sampling, including helper temperature 0.
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
    fn sonnet_retains_sampling_and_unmarked_instructions_remain_intact() {
        let policy = RequestPolicy::for_models("~anthropic/claude-sonnet-latest", &[]);
        let mut body = json!({"temperature":0, "instructions":"unmarked", "input":[]});
        policy.apply_parameters(&mut body);
        policy.apply_responses_instructions(&mut body, "unmarked".into());
        assert_eq!(body["temperature"], 0);
        assert_eq!(body["instructions"], "unmarked");
        assert_eq!(body["input"], json!([]));
    }
}
