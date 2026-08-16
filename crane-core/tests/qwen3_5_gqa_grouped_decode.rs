//! Equivalence of the grouped-GQA decode path with the legacy expansion path.
//!
//! `FullAttention::forward` used to materialize `k_rep`, `v_rep` and `k_t` at
//! full context width on every decoded token — three O(context) copies per
//! full-attention layer. The decode path now folds the `n_rep` query heads that
//! share a KV head into the matmul's row dimension instead, reading K/V once at
//! their stored width.
//!
//! The reformulation is supposed to be exact, so this pins that down two ways:
//! the raw tensor algebra, and end-to-end greedy decoding of a real checkpoint
//! with `CRANE_ATTN_EXPAND=1` selecting the old path in the same binary.
//!
//! ```sh
//! CRANE_QWEN35_GGUF=/path/to/Qwen3.5-4B-Q6_K.gguf \
//!   cargo test -p crane-core --release --features cuda \
//!     --test qwen3_5_gqa_grouped_decode -- --ignored --nocapture --test-threads=1
//! ```

use candle_core::{DType, Device, Tensor, D};
use crane_core::generation::based::ModelForCausalLM;
use crane_core::generation::GenerationConfig;
use crane_core::models::qwen3_5::{Model, ModelFormat};

const PROMPT: &str =
    "<|im_start|>user\nBriefly explain what a crane (the bird) looks like.<|im_end|>\n<|im_start|>assistant\n";
const MAX_NEW_TOKENS: usize = 48;

fn device() -> Device {
    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        return Device::new_cuda(0).unwrap();
    }
    Device::Cpu
}

/// The two formulations, on the same random Q/K/V.
///
/// Runs on whatever device is available — the algebra is device-independent,
/// so this is useful even on a CPU-only machine.
#[test]
fn grouped_decode_matches_expansion_math() -> candle_core::Result<()> {
    let dev = device();
    // Qwen3.8-27B geometry: 24 query heads over 4 KV heads.
    let (b, kv_heads, n_rep, d, s) = (1usize, 4usize, 6usize, 256usize, 733usize);
    let heads = kv_heads * n_rep;
    let scale = 1.0 / (d as f64).sqrt();

    let q = Tensor::randn(0f32, 1f32, (b, heads, 1, d), &dev)?;
    let k = Tensor::randn(0f32, 1f32, (b, kv_heads, s, d), &dev)?;
    let v = Tensor::randn(0f32, 1f32, (b, kv_heads, s, d), &dev)?;

    // ── legacy: expand K/V to 24 heads, then standard SDPA ──
    let k_rep = k
        .unsqueeze(2)?
        .expand((b, kv_heads, n_rep, s, d))?
        .contiguous()?
        .reshape((b, heads, s, d))?;
    let v_rep = v
        .unsqueeze(2)?
        .expand((b, kv_heads, n_rep, s, d))?
        .contiguous()?
        .reshape((b, heads, s, d))?;
    let k_t = k_rep.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    let logits = (q.matmul(&k_t)? * scale)?;
    let w = candle_nn::ops::softmax_last_dim(&logits)?;
    let expected = w.matmul(&v_rep)?.reshape((b, 1, heads * d))?;

    // ── grouped: fold n_rep into the matmul rows, K/V read at stored width ──
    let q_g = (q.reshape((b, kv_heads, n_rep, d))? * scale)?;
    let k_t2 = k.transpose(2, 3)?;
    let w2 = candle_nn::ops::softmax_last_dim(&q_g.matmul(&k_t2)?)?;
    let got = w2.matmul(&v)?.reshape((b, 1, heads * d))?;

    assert_eq!(got.dims(), expected.dims());
    let max_abs = (got - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("grouped vs expansion, max|Δ| = {max_abs:.3e}");
    // Same operations in a different association order; only float reassociation
    // should separate them.
    assert!(max_abs < 1e-4, "grouped decode diverged from expansion: {max_abs:e}");
    Ok(())
}

/// Head order must survive the regroup: `[B, H, 1, D]` reshaped through
/// `[B, kv_heads, n_rep, D]` and back has to land each head where `o_proj`
/// expects it. A transposition here would still produce plausible text, so it
/// is checked directly rather than left to the eyeball.
#[test]
fn regroup_preserves_head_order() -> candle_core::Result<()> {
    let dev = device();
    let (b, kv_heads, n_rep, d) = (1usize, 4usize, 6usize, 8usize);
    let heads = kv_heads * n_rep;

    // Head i carries the constant value i, so any permutation is visible.
    let per_head: Vec<f32> = (0..heads)
        .flat_map(|h| std::iter::repeat(h as f32).take(d))
        .collect();
    let x = Tensor::from_vec(per_head.clone(), (b, heads, 1, d), &dev)?;

    let regrouped = x.reshape((b, kv_heads, n_rep, d))?.reshape((b, 1, heads * d))?;
    let flat = regrouped.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(flat, per_head, "head order changed across the regroup");
    Ok(())
}

fn greedy(gguf: &str, legacy: bool) -> Vec<u32> {
    // SAFETY: single-threaded test (`--test-threads=1`); read once per forward,
    // memoized in a OnceLock inside the model, so it must be set before load.
    unsafe {
        if legacy {
            std::env::set_var("CRANE_ATTN_EXPAND", "1");
        } else {
            std::env::remove_var("CRANE_ATTN_EXPAND");
        }
    }
    let dev = device();
    let dtype = if dev.is_cpu() { DType::F32 } else { DType::F16 };
    let mut model = Model::new_with_format(gguf, &dev, &dtype, ModelFormat::Gguf).expect("load");
    let ids = model.prepare_inputs(PROMPT).expect("tokenize");
    let cfg = GenerationConfig {
        max_new_tokens: MAX_NEW_TOKENS,
        temperature: None,
        top_p: None,
        ..Default::default()
    };
    let out = model.generate(&ids, &cfg, None).expect("generate");
    out[ids.len()..].to_vec()
}

/// End-to-end: the same checkpoint decoded both ways must say the same thing.
#[test]
#[ignore = "needs a local Qwen3.5-family GGUF (CRANE_QWEN35_GGUF)"]
fn grouped_decode_matches_expansion_end_to_end() {
    let Ok(gguf) = std::env::var("CRANE_QWEN35_GGUF") else {
        eprintln!("skipped: set CRANE_QWEN35_GGUF");
        return;
    };
    let grouped = greedy(&gguf, false);
    let legacy = greedy(&gguf, true);
    unsafe { std::env::remove_var("CRANE_ATTN_EXPAND") };

    println!("grouped: {grouped:?}");
    println!("legacy : {legacy:?}");
    assert_eq!(
        grouped, legacy,
        "grouped decode changed greedy output vs the expansion path"
    );
}
