//! Chat template formatting.
//!
//! Converts OpenAI-style `[{role, content}]` messages into the prompt string
//! expected by each model family.  Two strategies:
//!
//! * **[`AutoChatTemplate`]** — uses the Jinja `chat_template` from
//!   `tokenizer_config.json` (works for Qwen, Llama, Mistral, …).
//! * **[`HunyuanChatTemplate`]** — hardcoded template for Hunyuan models.

use crate::openai_api::ChatMessage;
use crate::reasoning::ThinkingOptions;
use crane_core::autotokenizer::AutoTokenizer;

// ─────────────────────────────────────────────────────────────
//  Trait
// ─────────────────────────────────────────────────────────────

/// Formats chat messages into a model-specific prompt string.
pub trait ChatTemplateProcessor: Send + Sync {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String, String>;

    /// Render with reasoning controls. Defaults to ignoring them, which is
    /// correct for templates that have no notion of thinking (Hunyuan).
    fn apply_with_thinking(
        &self,
        messages: &[ChatMessage],
        _opts: &ThinkingOptions,
    ) -> Result<String, String> {
        self.apply(messages)
    }
}

// ─────────────────────────────────────────────────────────────
//  AutoChatTemplate (Jinja-based)
// ─────────────────────────────────────────────────────────────

/// Uses [`AutoTokenizer`]'s Jinja `chat_template` from `tokenizer_config.json`.
pub struct AutoChatTemplate {
    tokenizer: AutoTokenizer,
}

impl AutoChatTemplate {
    pub fn new(model_path: &str) -> Result<Self, String> {
        let tokenizer = AutoTokenizer::from_pretrained(model_path, None)
            .map_err(|e| format!("Failed to load AutoTokenizer: {e}"))?;
        Ok(Self { tokenizer })
    }
}

impl ChatTemplateProcessor for AutoChatTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String, String> {
        self.apply_with_thinking(messages, &ThinkingOptions::default())
    }

    fn apply_with_thinking(
        &self,
        messages: &[ChatMessage],
        opts: &ThinkingOptions,
    ) -> Result<String, String> {
        // Build the list of {role, content} values expected by the Jinja template.
        let template_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.text_content(),
                })
            })
            .collect();

        self.tokenizer
            .apply_chat_template_full(
                &template_messages,
                Option::<&serde_json::Value>::None,
                true,
                opts.enable_thinking,
                opts.reasoning_effort.as_deref(),
            )
            // Qwen 3.6+ templates `raise_exception` on an unsupported
            // reasoning_effort, so this is a user-input error, not a bug.
            .map_err(|e| format!("Chat template error: {e}"))
    }
}

// ─────────────────────────────────────────────────────────────
//  HunyuanChatTemplate (hardcoded)
// ─────────────────────────────────────────────────────────────

/// Hardcoded chat template for Hunyuan Dense models.
pub struct HunyuanChatTemplate;

impl ChatTemplateProcessor for HunyuanChatTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String, String> {
        const BOS: &str = "<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}>";
        const USER: &str = "<\u{ff5c}hy_User\u{ff5c}>";
        const ASSISTANT: &str = "<\u{ff5c}hy_Assistant\u{ff5c}>";
        const EOS: &str = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}2\u{ff5c}>";
        const SEP: &str = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}3\u{ff5c}>";

        let mut result = String::new();
        result.push_str(BOS);

        let (system_msg, loop_messages) = if !messages.is_empty() && messages[0].role == "system" {
            (Some(messages[0].text_content()), &messages[1..])
        } else {
            (None, &messages[..])
        };

        if let Some(sys) = system_msg {
            result.push_str(&sys);
            result.push_str(SEP);
        }

        for msg in loop_messages {
            match msg.role.as_str() {
                "user" => {
                    result.push_str(USER);
                    result.push_str(&msg.text_content());
                }
                "assistant" => {
                    result.push_str(ASSISTANT);
                    result.push_str(&msg.text_content());
                    result.push_str(EOS);
                }
                _ => {}
            }
        }

        result.push_str(ASSISTANT);
        Ok(result)
    }
}
