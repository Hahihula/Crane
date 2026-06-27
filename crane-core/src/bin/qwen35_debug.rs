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
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-path" => model_path = args[i + 1].clone(),
            "--ids" => ids_csv = args[i + 1].clone(),
            "--topk" => topk = args[i + 1].parse().unwrap_or(10),
            "--layer-stats" => layer_stats = true,
            _ => {}
        }
        i += 2;
    }
    if model_path.is_empty() || ids_csv.is_empty() {
        eprintln!("Usage: qwen35_debug --model-path <dir> --ids <csv> [--topk N] [--layer-stats]");
        anyhow::bail!("--model-path and --ids required");
    }

    let ids: Vec<u32> = ids_csv
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse::<u32>())
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("parse ids: {e}"))?;
    println!("[crane] input_ids: {ids:?}");

    let device = Device::Cpu;
    let dtype = candle_core::DType::F32;
    let mut model = Model::new(&model_path, &device, &dtype).context("loading model")?;
    if layer_stats {
        model.debug_layer_stats(&ids)?;
    } else {
        model.debug_topk(&ids, topk)?;
    }
    Ok(())
}