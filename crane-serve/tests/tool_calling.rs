//! Tool calling, end to end against a real checkpoint's chat template.
//!
//! Two halves have to line up for a tool turn to work, and this checks both
//! against the model's own Jinja template rather than a hand-written mock:
//!
//! 1. **Out** — `tools` reach the prompt, and an assistant turn that already
//!    called a tool renders back into the transcript. If it does not, the model
//!    sees its own call vanish from the history and calls again forever.
//! 2. **Back** — what the model emits parses into OpenAI `tool_calls`.
//!
//! ```sh
//! CRANE_QWEN38_MODEL=~/models/Qwen3.8-27B-Q4_K_M.gguf \
//!   cargo test -p crane-serve --test tool_calling -- --ignored --nocapture
//! ```

use crane_serve::chat_template::{AutoChatTemplate, ChatTemplateProcessor, RenderOptions};
use crane_serve::openai_api::{
    ChatMessage, ChatMessageContent, FunctionCall, FunctionDef, Tool, ToolCall,
};
use crane_serve::tools::parse_output;

/// What the Qwen 3.5/3.6/3.8 and Ornith templates instruct the model to emit.
const MODEL_REPLY: &str = "Let me look that up.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

fn template() -> Option<AutoChatTemplate> {
    let path = std::env::var("CRANE_QWEN38_MODEL").ok()?;
    let path = match path.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => path,
    };
    Some(AutoChatTemplate::new(&path).expect("load chat template"))
}

fn weather_tool() -> Tool {
    Tool {
        kind: "function".into(),
        function: FunctionDef {
            name: "get_weather".into(),
            description: Some("Get the current weather for a city".into()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            })),
        },
    }
}

fn msg(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: ChatMessageContent::Text(content.into()),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

#[test]
#[ignore = "requires CRANE_QWEN38_MODEL"]
fn tools_reach_the_prompt() {
    let Some(t) = template() else {
        eprintln!("skipped: set CRANE_QWEN38_MODEL");
        return;
    };
    let tools = [weather_tool()];
    let opts = RenderOptions {
        tools: Some(&tools),
        ..Default::default()
    };
    let prompt = t
        .apply_with(&[msg("user", "What is the weather in Paris?")], &opts)
        .unwrap();

    assert!(prompt.contains("<tools>"), "no tool block rendered");
    assert!(prompt.contains("get_weather"));
    assert!(prompt.contains("Get the current weather for a city"));
    // The template teaches the model the call syntax; without it the model has
    // no way to know what to emit.
    assert!(
        prompt.contains("<function=example_function_name>"),
        "no format instruction"
    );

    // And with no tools, none of that appears.
    let bare = t
        .apply_with(&[msg("user", "hi")], &RenderOptions::default())
        .unwrap();
    assert!(
        !bare.contains("<tools>"),
        "tool block rendered without tools"
    );
}

/// The full agentic loop: call → result → follow-up prompt.
#[test]
#[ignore = "requires CRANE_QWEN38_MODEL"]
fn assistant_tool_calls_and_results_round_trip() {
    let Some(t) = template() else {
        eprintln!("skipped: set CRANE_QWEN38_MODEL");
        return;
    };

    // Step 1: what the model said, parsed the way the server parses it.
    let parsed = parse_output(MODEL_REPLY);
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.content, "Let me look that up.");

    // Step 2: replay that turn plus the tool's result.
    let mut assistant = msg("assistant", &parsed.content);
    assistant.tool_calls = Some(parsed.tool_calls.clone());
    let mut result = msg("tool", r#"{"temperature_c": 18, "conditions": "cloudy"}"#);
    result.tool_call_id = Some(parsed.tool_calls[0].id.clone());
    result.name = Some("get_weather".into());

    let tools = [weather_tool()];
    let opts = RenderOptions {
        tools: Some(&tools),
        ..Default::default()
    };
    let prompt = t
        .apply_with(
            &[
                msg("user", "What is the weather in Paris?"),
                assistant,
                result,
            ],
            &opts,
        )
        .unwrap();

    println!(
        "--- rendered follow-up prompt (tail) ---\n{}",
        &prompt[prompt.len().saturating_sub(600)..]
    );

    // The assistant's call must be back in the transcript, in the template's
    // own syntax — not as the raw text we parsed it from.
    assert!(
        prompt.contains("<function=get_weather>"),
        "tool call not replayed"
    );
    assert!(
        prompt.contains("<parameter=city>"),
        "arguments not replayed"
    );
    assert!(prompt.contains("Paris"));
    // And the tool's answer must come back as a tool_response turn.
    assert!(
        prompt.contains("<tool_response>"),
        "tool result not rendered"
    );
    assert!(prompt.contains("cloudy"));
}

/// `arguments` crosses the wire as a JSON string but the templates iterate it
/// (`tool_call.arguments|items`), so the conversion must happen before render.
/// A string would make the filter fail and take the whole request with it.
#[test]
#[ignore = "requires CRANE_QWEN38_MODEL"]
fn json_string_arguments_render_as_parameters() {
    let Some(t) = template() else {
        eprintln!("skipped: set CRANE_QWEN38_MODEL");
        return;
    };
    let mut assistant = msg("assistant", "");
    assistant.tool_calls = Some(vec![ToolCall {
        id: "call_0".into(),
        kind: "function".into(),
        function: FunctionCall {
            name: "search".into(),
            // Exactly what an OpenAI client sends back: a string, two keys.
            arguments: r#"{"query":"rust lang","limit":5}"#.into(),
        },
    }]);

    let prompt = t
        .apply_with(
            &[msg("user", "find something"), assistant],
            &RenderOptions::default(),
        )
        .expect("template must accept string-encoded arguments");

    assert!(prompt.contains("<parameter=query>"));
    assert!(prompt.contains("rust lang"));
    assert!(prompt.contains("<parameter=limit>"));
    assert!(prompt.contains('5'));
}
