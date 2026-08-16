//! End-to-end thinking control against a real checkpoint's chat template.
//!
//! Renders the prompt through the model's own embedded Jinja template for each
//! `enable_thinking` / `reasoning_effort` combination, then runs a synthetic
//! completion back through the splitter — the same pair of steps the server
//! performs, minus the GPU.
//!
//! Gated on a checkpoint path (`.gguf` file, or a directory with
//! `tokenizer_config.json` / `chat_template.jinja`):
//!
//! ```sh
//! CRANE_QWEN38_MODEL=~/models/Qwen3.8-27B-Q4_K_M.gguf \
//!     cargo test -p crane-serve --test thinking_control -- --ignored --nocapture
//! ```

use crane_core::autotokenizer::AutoTokenizer;
use crane_serve::reasoning::{split_complete, ThinkingOptions, THINK_CLOSE, THINK_OPEN};

fn load_tokenizer() -> Option<AutoTokenizer> {
    let path = std::env::var("CRANE_QWEN38_MODEL").ok()?;
    let path = shellexpand(&path);
    let tok = if path.ends_with(".gguf") {
        AutoTokenizer::from_gguf(&path).expect("load tokenizer from GGUF")
    } else {
        AutoTokenizer::from_pretrained(&path, None).expect("load tokenizer from directory")
    };
    Some(tok)
}

fn shellexpand(p: &str) -> String {
    match p.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => p.to_string(),
    }
}

fn render(tok: &AutoTokenizer, opts: &ThinkingOptions) -> String {
    let messages = serde_json::json!([{"role": "user", "content": "What is 2+2?"}]);
    tok.apply_chat_template_full(
        &messages,
        Option::<&serde_json::Value>::None,
        true,
        opts.enable_thinking,
        opts.reasoning_effort.as_deref(),
    )
    .expect("render chat template")
}

/// Whether the rendered prompt leaves a `<think>` block open — the condition
/// the splitter keys off.
fn leaves_think_open(prompt: &str) -> bool {
    match (prompt.rfind(THINK_OPEN), prompt.rfind(THINK_CLOSE)) {
        (Some(o), Some(c)) => o > c,
        (Some(_), None) => true,
        (None, _) => false,
    }
}

#[test]
#[ignore = "requires CRANE_QWEN38_MODEL"]
fn thinking_on_leaves_block_open_and_splitter_recovers_the_answer() {
    let Some(tok) = load_tokenizer() else {
        eprintln!("skipped: set CRANE_QWEN38_MODEL");
        return;
    };

    let prompt = render(&tok, &ThinkingOptions::default());
    println!("--- default (thinking on) tail ---\n{}", tail(&prompt));
    assert!(
        leaves_think_open(&prompt),
        "expected a dangling <think> with thinking on"
    );

    // The completion therefore contains only the CLOSING tag.
    let output = "Two plus two is four.\n</think>\n\n4";
    let (reasoning, content) = split_complete(&prompt, output);
    assert_eq!(reasoning.as_deref(), Some("Two plus two is four."));
    assert_eq!(content, "4");
}

#[test]
#[ignore = "requires CRANE_QWEN38_MODEL"]
fn thinking_off_pre_closes_the_block_so_output_is_pure_content() {
    let Some(tok) = load_tokenizer() else {
        eprintln!("skipped: set CRANE_QWEN38_MODEL");
        return;
    };

    let opts = ThinkingOptions { enable_thinking: Some(false), reasoning_effort: None };
    let prompt = render(&tok, &opts);
    println!("--- enable_thinking=false tail ---\n{}", tail(&prompt));
    assert!(
        !leaves_think_open(&prompt),
        "expected a pre-closed <think></think> with thinking off"
    );

    let (reasoning, content) = split_complete(&prompt, "4");
    assert_eq!(reasoning, None, "no reasoning block should be reported");
    assert_eq!(content, "4");
}

/// Each supported effort must reach the template and produce a distinct
/// prompt.
///
/// Note the asymmetry, which is the template's design rather than a bug:
/// `xhigh` and `low` each inject a "Reasoning effort is set to …" sentence,
/// while **`medium` injects nothing** — it is the neutral baseline. So
/// `medium` cannot be detected by looking for its own name in the prompt.
#[test]
#[ignore = "requires CRANE_QWEN38_MODEL"]
fn reasoning_effort_reaches_the_template() {
    let Some(tok) = load_tokenizer() else {
        eprintln!("skipped: set CRANE_QWEN38_MODEL");
        return;
    };

    let effort_prompt = |effort: &str| {
        render(
            &tok,
            &ThinkingOptions {
                enable_thinking: None,
                reasoning_effort: Some(effort.to_string()),
            },
        )
    };

    let (low, medium, xhigh) = (effort_prompt("low"), effort_prompt("medium"), effort_prompt("xhigh"));
    for (name, prompt) in [("low", &low), ("medium", &medium), ("xhigh", &xhigh)] {
        println!("--- reasoning_effort={name} ---\n{}", first_system_line(prompt));
    }

    assert!(low.contains("Reasoning effort is set to low"));
    assert!(xhigh.contains("Reasoning effort is set to xhigh"));
    assert!(
        !medium.contains("Reasoning effort is set to"),
        "medium is the neutral baseline and should inject no instruction"
    );
    assert_ne!(low, medium);
    assert_ne!(medium, xhigh);

    // Undefined effort must fall through to the template's own default,
    // which for Qwen 3.8 is xhigh.
    assert_eq!(render(&tok, &ThinkingOptions::default()), xhigh);

    let bogus = tok.apply_chat_template_full(
        &serde_json::json!([{"role": "user", "content": "hi"}]),
        Option::<&serde_json::Value>::None,
        true,
        None,
        Some("turbo"),
    );
    assert!(bogus.is_err(), "unsupported effort should be rejected by the template");
}

fn tail(s: &str) -> String {
    let n = s.chars().count().saturating_sub(120);
    s.chars().skip(n).collect::<String>().replace('\n', "\\n")
}

fn first_system_line(s: &str) -> String {
    s.lines()
        .find(|l| l.contains("Reasoning effort"))
        .unwrap_or("<no reasoning instruction found>")
        .chars()
        .take(140)
        .collect()
}
