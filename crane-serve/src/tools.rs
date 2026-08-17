//! Tool (function) calling: parsing what the model emits back into OpenAI shape.
//!
//! The request side needs no code — `tools` is handed to the Jinja chat
//! template verbatim (`tool | tojson`), and the template owns the prompt
//! format. The response side does: the model answers in the template's own
//! syntax, and clients expect `message.tool_calls`.
//!
//! Qwen 3.5/3.6/3.8 and Ornith all specify the same XML-ish grammar, quoted
//! from the Qwen 3.8 template itself:
//!
//! ```text
//! <tool_call>
//! <function=example_function_name>
//! <parameter=example_parameter_1>
//! value_1
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Note this is *not* the `{"name": …, "arguments": …}` JSON that older Qwen
//! releases used — parameter values are raw text, newline-delimited, and may
//! span multiple lines.
//!
//! The template also permits prose before a call ("You may provide optional
//! reasoning for your function call in natural language BEFORE the function
//! call, but NOT after"), so surrounding text is preserved as `content`.

use serde_json::{Map, Value};

use crate::openai_api::{FunctionCall, ToolCall};

const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";
const FN_OPEN: &str = "<function=";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";

/// Everything the model produced, split into prose and calls.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ParsedOutput {
    /// Text outside any `<tool_call>` block, trimmed.
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

impl ParsedOutput {
    pub fn has_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

/// Split `text` into prose and tool calls.
///
/// An unterminated `<tool_call>` is dropped rather than guessed at — that
/// happens when generation hits the token limit mid-call, and a half-parsed
/// call would be executed by the client as if it were complete.
pub fn parse_output(text: &str) -> ParsedOutput {
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    let mut rest = text;

    while let Some(start) = rest.find(CALL_OPEN) {
        content.push_str(&rest[..start]);
        let after = &rest[start + CALL_OPEN.len()..];
        match after.find(CALL_CLOSE) {
            Some(end) => {
                if let Some(call) = parse_call(&after[..end], tool_calls.len()) {
                    tool_calls.push(call);
                }
                rest = &after[end + CALL_CLOSE.len()..];
            },
            None => {
                // Truncated mid-call: the prose already copied above is kept,
                // the fragment is dropped, and nothing after it can exist.
                rest = "";
                break;
            },
        }
    }
    content.push_str(rest);

    ParsedOutput {
        content: content.trim().to_string(),
        tool_calls,
    }
}

/// Parse one `<function=NAME>…</function>` body into an OpenAI tool call.
fn parse_call(block: &str, index: usize) -> Option<ToolCall> {
    let name_start = block.find(FN_OPEN)? + FN_OPEN.len();
    let name_end = block[name_start..].find('>')? + name_start;
    let name = block[name_start..name_end].trim();
    if name.is_empty() {
        return None;
    }

    let mut args = Map::new();
    let mut rest = &block[name_end..];
    while let Some(p) = rest.find(PARAM_OPEN) {
        let after = &rest[p + PARAM_OPEN.len()..];
        let Some(key_end) = after.find('>') else {
            break;
        };
        let key = after[..key_end].trim().to_string();
        let value_region = &after[key_end + 1..];
        let Some(value_end) = value_region.find(PARAM_CLOSE) else {
            break;
        };
        // The template wraps values in newlines (`>\nVALUE\n</parameter>`),
        // but a value may itself span lines, so only the framing is trimmed.
        args.insert(key, coerce(value_region[..value_end].trim()));
        rest = &value_region[value_end + PARAM_CLOSE.len()..];
    }

    Some(ToolCall {
        // OpenAI requires an id so tool results can be correlated back. The
        // template does not emit one, so synthesize a stable per-message id.
        id: format!("call_{index}"),
        kind: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            // OpenAI carries arguments as a JSON *string*, not an object.
            arguments: Value::Object(args).to_string(),
        },
    })
}

/// Parameter values arrive as text. Recover JSON scalars so a client that
/// deserializes `arguments` sees `{"count": 3}` rather than `{"count": "3"}`,
/// while anything that is not valid JSON stays a string.
fn coerce(raw: &str) -> Value {
    match serde_json::from_str::<Value>(raw) {
        // A bare word like `Paris` is not JSON; `3`, `true`, `[1,2]` are.
        Ok(v) if !v.is_string() => v,
        _ => Value::String(raw.to_string()),
    }
}

/// Streaming filter that keeps tool-call markup out of `content` deltas.
///
/// A tool call cannot be streamed incrementally the way text can: the client
/// needs a complete, parseable call before it can run anything, and OpenAI's
/// own incremental `tool_calls` deltas assume a JSON grammar this template
/// does not use. So text streams normally until `<tool_call>` appears, after
/// which everything is buffered and the finished calls are emitted in one
/// delta at the end.
#[derive(Default)]
pub struct ToolCallStream {
    /// Text held back because it might be the start of `<tool_call>`.
    pending: String,
    /// Everything from the first `<tool_call>` onward.
    buffered: String,
    in_call: bool,
}

impl ToolCallStream {
    /// Feed a decoded content delta; returns the part safe to stream now.
    pub fn push(&mut self, text: &str) -> String {
        if self.in_call {
            self.buffered.push_str(text);
            return String::new();
        }
        self.pending.push_str(text);

        if let Some(idx) = self.pending.find(CALL_OPEN) {
            let emit = self.pending[..idx].to_string();
            self.buffered = self.pending[idx..].to_string();
            self.pending.clear();
            self.in_call = true;
            return emit;
        }
        // Hold back only a suffix that could still become `<tool_call>`.
        let keep = crate::reasoning::partial_tag_suffix_len(&self.pending, CALL_OPEN);
        let split = self.pending.len() - keep;
        let emit = self.pending[..split].to_string();
        self.pending = self.pending[split..].to_string();
        emit
    }

    /// Flush: any trailing content, plus the parsed calls.
    pub fn finish(&mut self) -> (String, Vec<ToolCall>) {
        if !self.in_call {
            return (std::mem::take(&mut self.pending), Vec::new());
        }
        let parsed = parse_output(&std::mem::take(&mut self.buffered));
        (parsed.content, parsed.tool_calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact shape the Qwen 3.8 template instructs the model to produce.
    const CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

    #[test]
    fn parses_a_single_call() {
        let out = parse_output(CALL);
        assert_eq!(out.tool_calls.len(), 1);
        let c = &out.tool_calls[0];
        assert_eq!(c.function.name, "get_weather");
        assert_eq!(c.function.arguments, r#"{"city":"Paris"}"#);
        assert_eq!(c.kind, "function");
        assert_eq!(out.content, "");
    }

    /// The template explicitly allows reasoning before a call, so prose must
    /// survive as content rather than being swallowed.
    #[test]
    fn keeps_prose_around_calls() {
        let out = parse_output(&format!("Let me check that.\n{CALL}"));
        assert_eq!(out.content, "Let me check that.");
        assert_eq!(out.tool_calls.len(), 1);
    }

    #[test]
    fn parses_multiple_calls_with_distinct_ids() {
        let two = format!("{CALL}\n{CALL}");
        let out = parse_output(&two);
        assert_eq!(out.tool_calls.len(), 2);
        assert_ne!(out.tool_calls[0].id, out.tool_calls[1].id);
    }

    #[test]
    fn parses_multiple_parameters() {
        let text = "<tool_call>\n<function=search>\n<parameter=query>\nrust lang\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>";
        let out = parse_output(text);
        let args: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust lang");
        // Numeric-looking values become JSON numbers, not strings.
        assert_eq!(args["limit"], 5);
    }

    /// Values may span lines — the template's own example says so.
    #[test]
    fn preserves_multiline_parameter_values() {
        let text = "<tool_call>\n<function=write>\n<parameter=body>\nline one\nline two\n</parameter>\n</function>\n</tool_call>";
        let out = parse_output(text);
        let args: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(args["body"], "line one\nline two");
    }

    /// Hitting the token limit mid-call must not yield a half-built call the
    /// client would then execute.
    #[test]
    fn drops_an_unterminated_call() {
        let out =
            parse_output("Checking.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nPar");
        assert!(out.tool_calls.is_empty());
        assert_eq!(out.content, "Checking.");
    }

    #[test]
    fn plain_text_has_no_calls() {
        let out = parse_output("Just an ordinary answer.");
        assert!(!out.has_calls());
        assert_eq!(out.content, "Just an ordinary answer.");
    }

    /// A malformed block (no `<function=`) is skipped, not turned into a call
    /// with an empty name.
    #[test]
    fn skips_a_block_without_a_function_name() {
        let out = parse_output("<tool_call>\ngarbage\n</tool_call>");
        assert!(out.tool_calls.is_empty());
    }

    // ── streaming ──

    /// Markup must never reach the client as content, even when the opening
    /// tag is split across token boundaries.
    #[test]
    fn stream_withholds_tool_markup() {
        let mut s = ToolCallStream::default();
        let mut streamed = String::new();
        for tok in [
            "Let me ",
            "check.",
            "\n<",
            "tool",
            "_call>",
            "\n<function=get_weather>\n",
            "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
        ] {
            streamed.push_str(&s.push(tok));
        }
        let (tail, calls) = s.finish();
        streamed.push_str(&tail);

        assert_eq!(streamed.trim(), "Let me check.");
        assert!(
            !streamed.contains("tool_call"),
            "markup leaked: {streamed:?}"
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    /// Ordinary replies must still stream token by token — a `<` that never
    /// becomes a tag has to be released, not swallowed.
    #[test]
    fn stream_passes_plain_text_through() {
        let mut s = ToolCallStream::default();
        let mut out = String::new();
        for tok in ["a < b", " and ", "c > d"] {
            out.push_str(&s.push(tok));
        }
        let (tail, calls) = s.finish();
        out.push_str(&tail);
        assert_eq!(out, "a < b and c > d");
        assert!(calls.is_empty());
    }
}
