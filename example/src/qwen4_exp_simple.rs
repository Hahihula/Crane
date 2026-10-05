//! Greedy text generation with a Qwen4-Exp GGUF (e.g.
//! `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`).
//!
//! Usage:
//!   cargo run --release --features sycl --bin `qwen4_exp_simple` -- \
//!       /path/to/Model-00001-of-00002.gguf "Write a Rust function that reverses a string."
//!
//! Pass the first shard; the n-gram table in the second one is memory-mapped
//! from next to it.

use std::io::Write;
use std::time::Instant;

use anyhow::Result;
use crane_core::models::Device;
use crane_core::models::qwen4_exp::Model;

fn main() -> Result<()> {
    crane_core::utils::sycl_env::ensure_sycl_runtime_env();
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: qwen4_exp_simple <first-shard.gguf> [prompt]"))?;
    let prompt = args
        .next()
        .unwrap_or_else(|| "Write a Rust function that reverses a string.".to_string());
    let max_new_tokens: usize = std::env::var("MAX_NEW_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);

    #[cfg(feature = "sycl")]
    let device = Device::new_sycl(0)?;
    #[cfg(not(feature = "sycl"))]
    let device = Device::cuda_if_available(0)?;
    eprintln!("Device: {device:?}");

    let t = Instant::now();
    let mut model = Model::from_gguf_file(std::path::Path::new(&path), &device)?;
    eprintln!("Loaded in {:.1}s", t.elapsed().as_secs_f64());

    // ChatML with an empty think block, i.e. thinking disabled.
    let text = format!(
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let ids = model
        .tokenizer
        .tokenizer
        .encode(text, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    let t = Instant::now();
    let mut logits = model.forward_step(&ids, 0)?;
    device.synchronize()?;
    let prefill = t.elapsed().as_secs_f64();
    eprintln!(
        "Prefill: {} tokens in {prefill:.2}s ({:.1} tok/s)",
        ids.len(),
        ids.len() as f64 / prefill
    );

    let eos = model.eos_token_ids().to_vec();
    let mut pos = ids.len();
    let mut generated = 0usize;
    let t = Instant::now();
    for _ in 0..max_new_tokens {
        let next = logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;
        if eos.contains(&next) {
            break;
        }
        if let Some(piece) = model.tokenizer.next_token(next)? {
            print!("{piece}");
            std::io::stdout().flush()?;
        }
        logits = model.forward_step(&[next], pos)?;
        pos += 1;
        generated += 1;
    }
    if let Some(rest) = model.tokenizer.decode_rest()? {
        print!("{rest}");
    }
    println!();
    let decode = t.elapsed().as_secs_f64();
    eprintln!(
        "Decode: {generated} tokens in {decode:.2}s ({:.2} tok/s)",
        generated as f64 / decode.max(1e-9)
    );
    Ok(())
}
