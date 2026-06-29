//! Tiny CLI: load a Qwen 3.5 model and dump first-step top-K next-token logits.
//!
//! Usage:
//!   cargo run --release -p crane-core --bin qwen35_debug -- \
//!     --model-path /path/to/Qwen3.5-0.8B --ids 151644,872,9707,151645,151644,77091,198,271,152198,271,151645
//!
//! Use the `--ids` flag to pass raw token IDs (the chat template is bypassed
//! — we feed the same prompt tokens HF would). Compare the top-K against
//! `compare_qwen35.py` output to localize correctness drift.
//!
//! Add `--layer-stats` to print per-layer hidden-state stats to localize
//! divergence.

use anyhow::{Context, Result};
use candle_core::Device;
use crane_core::models::qwen3_5::Model;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = String::new();
    let mut ids_csv = String::new();
    let mut topk: usize = 10;
    let mut layer_stats = false;
    let mut device_arg = String::from("cpu");
    let mut dtype_arg = String::from("f32");
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-path" => model_path = args[i + 1].clone(),
            "--ids" => ids_csv = args[i + 1].clone(),
            "--topk" => topk = args[i + 1].parse().unwrap_or(10),
            "--device" => device_arg = args[i + 1].clone(),
            "--dtype" => dtype_arg = args[i + 1].clone(),
            "--layer-stats" => {
                layer_stats = true;
                i += 1;
                continue;
            }
            _ => {}
        }
        i += 2;
    }
    let has_prompt = std::env::args().any(|a| a == "--prompt" || a == "--messages");
    if model_path.is_empty() || (ids_csv.is_empty() && !has_prompt) {
        eprintln!("Usage: qwen35_debug --model-path <dir> (--ids <csv> | --prompt <text>) [--topk N] [--gen N] [--layer-stats]");
        anyhow::bail!("--model-path and one of --ids/--prompt required");
    }

    let ids: Vec<u32> = ids_csv
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse::<u32>())
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("parse ids: {e}"))?;

    let arg_after = |name: &str| {
        std::env::args()
            .position(|a| a == name)
            .and_then(|i| std::env::args().nth(i + 1))
    };
    let gen_n = arg_after("--gen").and_then(|s| s.parse::<usize>().ok());

    // Chat-template rendering via Crane's AutoTokenizer (the SDK path):
    //   --prompt "text"           single user message
    //   --messages '<json array>' full conversation
    //   --tools '<json array>'    optional OpenAI-style tool specs
    //   --render-only             print the rendered prompt and exit (no model)
    let prompt = arg_after("--prompt");
    let messages_json = arg_after("--messages");
    let tools_json = arg_after("--tools");
    let render_only = std::env::args().any(|a| a == "--render-only");

    let rendered = if prompt.is_some() || messages_json.is_some() {
        let tok = crane_core::autotokenizer::AutoTokenizer::from_pretrained(&model_path, None)
            .map_err(|e| anyhow::anyhow!("autotokenizer: {e}"))?;
        let messages: serde_json::Value = match messages_json {
            Some(j) => serde_json::from_str(&j).context("parse --messages json")?,
            None => serde_json::json!([{ "role": "user", "content": prompt.unwrap() }]),
        };
        let tools: Option<serde_json::Value> = match tools_json {
            Some(j) => Some(serde_json::from_str(&j).context("parse --tools json")?),
            None => None,
        };
        let r = tok
            .apply_chat_template_with_tools(&messages, tools, true)
            .map_err(|e| anyhow::anyhow!("apply_chat_template: {e}"))?;
        println!("=== rendered prompt ===\n{r}\n=== end rendered ===");
        Some(r)
    } else {
        None
    };
    if render_only {
        return Ok(());
    }

    let device = match device_arg.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::new_cuda(0).context("init CUDA device")?,
        "metal" => Device::new_metal(0).context("init Metal device")?,
        other => anyhow::bail!("unknown --device {other} (cpu|cuda|metal)"),
    };
    let dtype = match dtype_arg.as_str() {
        "f32" => candle_core::DType::F32,
        "f16" => candle_core::DType::F16,
        "bf16" => candle_core::DType::BF16,
        other => anyhow::bail!("unknown --dtype {other} (f32|f16|bf16)"),
    };
    println!("[crane] device={device_arg} dtype={dtype_arg}");
    let mut model = Model::new(&model_path, &device, &dtype).context("loading model")?;

    let ids = match rendered {
        Some(r) => model.prepare_inputs(&r).context("tokenize prompt")?,
        None => ids,
    };

    if let Some(max_new) = gen_n {
        use crane_core::generation::based::ModelForCausalLM;
        use crane_core::generation::GenerationConfig;
        let cfg = GenerationConfig {
            max_new_tokens: max_new,
            temperature: None, // greedy (argmax)
            report_speed: true,
            ..Default::default()
        };
        let out = model.generate(&ids, &cfg, None).context("generate")?;
        let new_ids = &out[ids.len()..];
        let text = model
            .tokenizer
            .tokenizer
            .decode(new_ids, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        println!("[crane] generated {} tokens:\n{text}", new_ids.len());
    } else if layer_stats {
        model.debug_layer_stats(&ids)?;
    } else {
        model.debug_topk(&ids, topk)?;
    }
    Ok(())
}