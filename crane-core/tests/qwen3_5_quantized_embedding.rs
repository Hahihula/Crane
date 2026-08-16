//! Equivalence check for the quantized embedding table.
//!
//! GGUF checkpoints used to have `token_embd.weight` dequantized to the
//! compute dtype at load; it now stays quantized and only the gathered rows
//! are dequantized. This test loads the *same* file both ways — the new path,
//! and the old one via `CRANE_EMBED_DENSE=1` — and compares them.
//!
//! What each assertion is worth depends on how the checkpoint ties its
//! weights, and the difference is worth stating precisely:
//!
//! * **Untied** (Qwen 3.6/3.8-27B): only the embedding changes, and the row
//!   gather decodes the very same blocks the bulk dequantization would, so the
//!   two paths are *bit-identical* — greedy decoding matches token for token.
//! * **Tied** (Qwen 3.5 0.8B/4B): the output projection is that same table, so
//!   keeping it quantized also turns the lm_head into a `QMatMul`. That is a
//!   genuinely different arithmetic path (activations are quantized for the
//!   integer dot product), so logits shift by a hair and greedy decoding can
//!   flip a near-tie. Hence the cosine bound rather than exact equality.
//!
//! ```bash
//! CRANE_QWEN35_GGUF=/path/to/Qwen3.5-0.8B-Q8_0.gguf \
//!   cargo test -p crane-core --release --test qwen3_5_quantized_embedding -- \
//!     --ignored --nocapture --test-threads=1
//! ```

use candle_core::{DType, Device, Tensor};
use crane_core::generation::based::ModelForCausalLM;
use crane_core::generation::GenerationConfig;
use crane_core::models::qwen3_5::{Model, ModelFormat};

const PROMPT: &str =
    "<|im_start|>user\nBriefly explain what a crane (the bird) looks like.<|im_end|>\n<|im_start|>assistant\n";
const MAX_NEW_TOKENS: usize = 32;

fn device_and_dtype() -> (Device, DType) {
    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        return (Device::new_cuda(0).unwrap(), DType::F16);
    }
    (Device::Cpu, DType::F32)
}

/// Greedy, so two runs on one machine are directly comparable.
fn greedy_config() -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: MAX_NEW_TOKENS,
        temperature: None,
        top_p: None,
        ..Default::default()
    }
}

/// Load `gguf` with the embedding table dense or quantized, and return both
/// the prefill logits and a greedy continuation.
fn run(gguf: &str, dense: bool) -> (Vec<f32>, Vec<u32>) {
    // SAFETY: single-threaded test (`--test-threads=1`); the variable is read
    // once, during model construction on the next line.
    unsafe {
        if dense {
            std::env::set_var("CRANE_EMBED_DENSE", "1");
        } else {
            std::env::remove_var("CRANE_EMBED_DENSE");
        }
    }
    let (device, dtype) = device_and_dtype();
    let mut model =
        Model::new_with_format(gguf, &device, &dtype, ModelFormat::Gguf).expect("load GGUF model");
    unsafe { std::env::remove_var("CRANE_EMBED_DENSE") };

    let ids = model.prepare_inputs(PROMPT).expect("tokenize prompt");
    let logits = to_f32(&model.forward_step(&ids, 0).expect("prefill"));

    let tokens = model
        .generate(&ids, &greedy_config(), None)
        .expect("generation failed");
    let generated = tokens[ids.len()..].to_vec();
    assert!(!generated.is_empty(), "no tokens generated");
    (logits, generated)
}

fn to_f32(t: &Tensor) -> Vec<f32> {
    t.flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

#[test]
#[ignore = "needs a local Qwen3.5-family GGUF (CRANE_QWEN35_GGUF)"]
fn quantized_embedding_matches_dense_embedding() {
    let Ok(gguf) = std::env::var("CRANE_QWEN35_GGUF") else {
        eprintln!("skipped: set CRANE_QWEN35_GGUF");
        return;
    };

    let (q_logits, q_tokens) = run(&gguf, false);
    let (d_logits, d_tokens) = run(&gguf, true);
    assert_eq!(q_logits.len(), d_logits.len(), "vocab size mismatch");

    let cos = cosine(&q_logits, &d_logits);
    let max_abs = q_logits
        .iter()
        .zip(&d_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let first_diff = q_tokens.iter().zip(&d_tokens).position(|(a, b)| a != b);

    println!("prefill logits cosine : {cos:.9}");
    println!("prefill logits max|Δ| : {max_abs:.6}");
    match first_diff {
        None => println!("greedy tokens         : identical for all {} tokens", q_tokens.len()),
        Some(i) => println!(
            "greedy tokens         : diverge at {i}/{} ({} vs {})",
            q_tokens.len(),
            q_tokens[i],
            d_tokens[i]
        ),
    }

    // The embedding gather itself is exact; on a tied checkpoint the lm_head
    // also becomes quantized, which moves logits slightly but must not change
    // what the model is saying.
    assert!(
        cos > 0.9999,
        "quantized embedding path diverged from dense: cosine {cos:.9}"
    );
}
