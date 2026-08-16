//! End-to-end generation checks for the Qwen 3.5 weight-loading paths
//! (safetensors, in-situ quantized, GGUF).
//!
//! These need local checkpoints, so they are `#[ignore]`d by default and
//! resolve their inputs from env vars:
//!
//! ```bash
//! CRANE_QWEN35_DIR=/path/to/Qwen3.5-4B \
//! CRANE_QWEN35_GGUF=/path/to/Qwen3.5-4B-Q6_K.gguf \
//!   cargo test -p crane-core --release --test qwen3_5_quant -- \
//!     --ignored --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` is required on GPU: each test loads its own copy of the
//! model, and several 4B copies at once will exhaust a 24 GB card. The failure
//! surfaces as `CUDA_ERROR_OUT_OF_MEMORY` in whichever test loses the race, so
//! it looks like a flaky/unrelated regression.
//!
//! Point `CRANE_QWEN35_*` at the **4B**, not the 0.8B. The 0.8B (and 2B) have
//! `linear_num_key_heads == linear_num_value_heads` → `v_per_group == 1`, which
//! makes the GDN key→value head expansion a no-op and hides any head-pairing
//! bug; the 4B is 16/32 (`v_per_group == 2`) and does exercise it.
//!
//! Greedy decoding (temperature = None) makes runs on the same machine
//! byte-comparable, which is how the LinearLayer refactor and the quantized
//! paths are validated against the bf16 baseline.

use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device};
use crane_core::generation::based::ModelForCausalLM;
use crane_core::generation::GenerationConfig;
use crane_core::models::qwen3_5::{Model, ModelFormat};

const PROMPT: &str = "<|im_start|>user\nBriefly explain what a crane (the bird) looks like.<|im_end|>\n<|im_start|>assistant\n";
const MAX_NEW_TOKENS: usize = 48;

fn device_and_dtype() -> (Device, DType) {
    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        return (Device::new_cuda(0).unwrap(), DType::F16);
    }
    if candle_core::utils::metal_is_available() {
        return (Device::new_metal(0).unwrap(), DType::F16);
    }
    (Device::Cpu, DType::F32)
}

fn greedy_config() -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: MAX_NEW_TOKENS,
        temperature: None,
        top_p: None,
        report_speed: true,
        ..Default::default()
    }
}

fn run_greedy(model: &mut Model, label: &str) -> Vec<u32> {
    let input_ids = model.prepare_inputs(PROMPT).expect("tokenize prompt");
    let tokens = model
        .generate(&input_ids, &greedy_config(), None)
        .expect("generation failed");
    let generated = &tokens[input_ids.len()..];
    let text = model
        .tokenizer
        .tokenizer
        .decode(generated, true)
        .unwrap_or_default();
    println!("[{label}] {} new tokens: {generated:?}", generated.len());
    println!("[{label}] text: {text}");
    assert!(!generated.is_empty(), "no tokens generated");
    generated.to_vec()
}

fn model_dir() -> String {
    std::env::var("CRANE_QWEN35_DIR").expect("set CRANE_QWEN35_DIR to a Qwen3.5 checkpoint dir")
}

#[test]
#[ignore = "needs a local Qwen3.5 checkpoint (CRANE_QWEN35_DIR)"]
fn greedy_safetensors() {
    let (device, dtype) = device_and_dtype();
    let mut model = Model::new(&model_dir(), &device, &dtype).expect("load safetensors model");
    run_greedy(&mut model, "safetensors");
}

/// Diagnostic for https://github.com/lucasjinreal/Crane/issues/88 (long-prompt
/// gibberish on GGUF). Set `CRANE_QWEN35_DEBUG_LAYERS=1` alongside this to dump
/// per-layer last-position hidden-state stats and compare where GGUF and
/// safetensors start to diverge on a long, well-formed prompt (not the
/// truncated-markdown repro, which confounds prompt-corruption with length).
#[test]
#[ignore = "needs a local Qwen3.5 checkpoint (CRANE_QWEN35_DIR and/or CRANE_QWEN35_GGUF)"]
fn long_prompt_divergence() {
    let para = "The quick brown fox jumps over the lazy dog near the river bank while the sun sets slowly behind the distant mountains, painting the sky in brilliant shades of orange and purple. ";
    let prompt = format!(
        "<|im_start|>user\nHere is some background reading:\n\n{}\n\nIn one short sentence, what color is the sky at sunset in the text above?<|im_end|>\n<|im_start|>assistant\n",
        para.repeat(100)
    );

    let which = std::env::var("CRANE_LONG_PROMPT_WHICH").unwrap_or_else(|_| "gguf".to_string());
    let (device, dtype) = device_and_dtype();
    let mut model = if which == "gguf" {
        let path = std::env::var("CRANE_QWEN35_GGUF").expect("set CRANE_QWEN35_GGUF");
        Model::new_with_options(&path, &device, &dtype, ModelFormat::Auto, None).expect("load gguf")
    } else {
        Model::new(&model_dir(), &device, &dtype).expect("load safetensors")
    };

    let input_ids = model.prepare_inputs(&prompt).expect("tokenize");
    println!("[{which}] prompt tokens: {}", input_ids.len());
    let cfg = GenerationConfig {
        max_new_tokens: 20,
        temperature: None,
        top_p: None,
        report_speed: true,
        ..Default::default()
    };
    let tokens = model.generate(&input_ids, &cfg, None).expect("generate");
    let generated = &tokens[input_ids.len()..];
    let text = model.tokenizer.tokenizer.decode(generated, true).unwrap_or_default();
    println!("[{which}] text: {text}");
}

/// Direct side-by-side: load both GGUF and safetensors, run the identical
/// long prompt through `forward_step` (one prefill call each, no
/// generation loop), and compare the last-position logits. Cosine
/// similarity catches a *directional* corruption (wrong pairing/order —
/// same magnitude, wrong correspondence, exactly the signature of the
/// already-fixed VHeadOrder bug) that aggregate min/max/mean stats can't.
#[test]
#[ignore = "needs CRANE_QWEN35_DIR and CRANE_QWEN35_GGUF, both Qwen3.5-4B"]
fn long_prompt_logit_cosine() {
    let para = "The quick brown fox jumps over the lazy dog near the river bank while the sun sets slowly behind the distant mountains, painting the sky in brilliant shades of orange and purple. ";
    let prompt = format!(
        "<|im_start|>user\nHere is some background reading:\n\n{}\n\nIn one short sentence, what color is the sky at sunset in the text above?<|im_end|>\n<|im_start|>assistant\n",
        para.repeat(100)
    );

    let (device, dtype) = device_and_dtype();
    let mut st_model = Model::new(&model_dir(), &device, &dtype).expect("load safetensors");
    let st_ids = st_model.prepare_inputs(&prompt).expect("tokenize st");
    let st_logits = st_model.forward_step(&st_ids, 0).expect("st forward");

    let gguf_path = std::env::var("CRANE_QWEN35_GGUF").expect("set CRANE_QWEN35_GGUF");
    let mut gguf_model =
        Model::new_with_options(&gguf_path, &device, &dtype, ModelFormat::Auto, None)
            .expect("load gguf");
    let gguf_ids = gguf_model.prepare_inputs(&prompt).expect("tokenize gguf");
    let gguf_logits = gguf_model.forward_step(&gguf_ids, 0).expect("gguf forward");

    println!("st tokens={} gguf tokens={}", st_ids.len(), gguf_ids.len());

    let st_v = st_logits.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let gguf_v = gguf_logits.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(st_v.len(), gguf_v.len(), "vocab size mismatch");

    let dot: f64 = st_v.iter().zip(&gguf_v).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
    let na: f64 = st_v.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = gguf_v.iter().map(|b| f64::from(*b).powi(2)).sum::<f64>().sqrt();
    let cosine = dot / (na * nb);
    println!("last-position logits cosine similarity: {cosine:.6}");
    // The bug fixed for issue #88 (ssm_a fed through -exp() twice) collapsed
    // this to ~0.32 on a 3.4k-token prompt. Fixed, it lands around ~0.80 —
    // well short of the ~0.998 an ISQ-vs-bf16 control gets (see
    // `long_prompt_isq_q6k_vs_bf16` history in git blame), because llama.cpp's
    // Q6_K quantizer and candle's aren't bit-identical, not because of a
    // remaining bug: greedy decoding on the real issue #88 repro and a
    // 218–13648 token length ramp both matched the safetensors answer
    // token-for-token after this fix. 0.5 comfortably separates "fixed" from
    // "double-exponentiated decay gate regressed".
    assert!(
        cosine > 0.5,
        "GGUF/safetensors last-position logits diverged too much on a long prompt \
         (cosine={cosine:.4}); this is the signature of \
         https://github.com/lucasjinreal/Crane/issues/88 (double-exponentiated \
         ssm_a decay gate, or a quantized ssm_beta/ssm_alpha) — see \
         DecoderLayer::from_gguf's LinearAttention branch in modeling.rs"
    );

    let top = |v: &[f32], n: usize| -> Vec<(usize, f32)> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        idx.into_iter().take(n).map(|i| (i, v[i])).collect()
    };
    println!("st   top5: {:?}", top(&st_v, 5));
    println!("gguf top5: {:?}", top(&gguf_v, 5));
}

fn run_isq(quant: GgmlDType, label: &str) -> Vec<u32> {
    let (device, dtype) = device_and_dtype();
    let mut model = Model::new_with_options(
        &model_dir(),
        &device,
        &dtype,
        ModelFormat::Auto,
        Some(quant),
    )
    .expect("load ISQ model");
    run_greedy(&mut model, label)
}

#[test]
#[ignore = "needs a local Qwen3.5 checkpoint (CRANE_QWEN35_DIR)"]
fn greedy_isq_q8_0() {
    run_isq(GgmlDType::Q8_0, "isq-q8_0");
}

#[test]
#[ignore = "needs a local Qwen3.5 checkpoint (CRANE_QWEN35_DIR)"]
fn greedy_isq_q4k() {
    run_isq(GgmlDType::Q4K, "isq-q4k");
}

#[test]
#[ignore = "needs a local Qwen3.5 GGUF (CRANE_QWEN35_GGUF) — tokenizer is read from the GGUF itself"]
fn greedy_gguf() {
    let path = std::env::var("CRANE_QWEN35_GGUF").expect("set CRANE_QWEN35_GGUF to a .gguf file");
    let (device, dtype) = device_and_dtype();
    let mut model = Model::new_with_options(&path, &device, &dtype, ModelFormat::Auto, None)
        .expect("load GGUF model");
    run_greedy(&mut model, "gguf");
}

/// Loads the GGUF from a fresh tempdir that has NO sibling `tokenizer.json` /
/// `chat_template.jinja` / `tokenizer_config.json`. Regression test for the
/// GGUF-embedded-tokenizer path; if `Model::from_gguf_file` regresses to
/// requiring a sibling tokenizer, this test will fail with a "Tokenizer not
/// found" error.
#[test]
#[ignore = "needs a local Qwen3.5 GGUF (CRANE_QWEN35_GGUF); copies to a fresh tempdir"]
fn greedy_gguf_isolated() {
    let path = std::env::var("CRANE_QWEN35_GGUF").expect("set CRANE_QWEN35_GGUF to a .gguf file");
    let dir = tempfile::tempdir().expect("tempdir");
    let dest = dir.path().join("qwen35.gguf");
    std::fs::copy(&path, &dest).expect("copy gguf to tempdir");

    let (device, dtype) = device_and_dtype();
    let mut model = Model::new_with_options(
        dest.to_str().unwrap(),
        &device,
        &dtype,
        ModelFormat::Auto,
        None,
    )
    .expect("load GGUF from tempdir (no sibling files)");
    run_greedy(&mut model, "gguf-isolated");
}

/// Regression test for the GDN value-head permutation bug: llama.cpp's GGUF
/// converter stores the linear-attention `num_v_heads` axis (A_log, dt_bias,
/// β, α, and the V/Z column blocks of the QKV/gate projections) in a
/// different head order than HF/Crane's `repeat_kv_heads` expects (see
/// `unchunk_value_heads` in `models/qwen3_5/modeling.rs`). Left unfixed the
/// model still runs but produces fluent-looking, quickly-degenerating
/// garbage.
///
/// This used to assert an exact-token greedy match over the first 8 tokens,
/// but reasoning models routinely sit right on a decision boundary (emit a
/// visible chain-of-thought vs. close `<think>` immediately) where genuine
/// Q6_K-vs-bf16 quantization noise — not structural corruption — is enough
/// to flip the argmax on token 2. Comparing first-position logits by cosine
/// similarity (the same technique `long_prompt_logit_cosine` uses) is more
/// robust, but even that has a lower noise floor than you'd expect at a
/// single early position: measured ~0.65–0.69 on this real (non-ISQ) Q6_K
/// checkpoint regardless of the issue #88 fix, since at token 0 the GDN
/// recurrence hasn't run long enough for that bug to compound — this test
/// predates #88 and was never sensitive to it. What it does catch is gross
/// structural corruption: the value-head permutation bug it was written for
/// scrambled logits toward 0, not down to 0.6.
#[test]
#[ignore = "needs CRANE_QWEN35_DIR (safetensors) + CRANE_QWEN35_GGUF"]
fn gguf_matches_safetensors_prefix() {
    let (device, dtype) = device_and_dtype();
    let mut st_model = Model::new(&model_dir(), &device, &dtype).expect("load safetensors model");
    let st_ids = st_model.prepare_inputs(PROMPT).expect("tokenize st");
    let st_logits = st_model.forward_step(&st_ids, 0).expect("st forward");

    let path = std::env::var("CRANE_QWEN35_GGUF").expect("set CRANE_QWEN35_GGUF to a .gguf file");
    let mut gguf_model = Model::new_with_options(&path, &device, &dtype, ModelFormat::Auto, None)
        .expect("load GGUF model");
    let gguf_ids = gguf_model.prepare_inputs(PROMPT).expect("tokenize gguf");
    let gguf_logits = gguf_model.forward_step(&gguf_ids, 0).expect("gguf forward");

    let st_v = st_logits.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let gguf_v = gguf_logits.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(st_v.len(), gguf_v.len(), "vocab size mismatch");

    let dot: f64 = st_v.iter().zip(&gguf_v).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
    let na: f64 = st_v.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = gguf_v.iter().map(|b| f64::from(*b).powi(2)).sum::<f64>().sqrt();
    let cosine = dot / (na * nb);
    println!("first-position logits cosine similarity: {cosine:.6}");
    assert!(
        cosine > 0.3,
        "GGUF and safetensors first-token logits diverged too far (cosine={cosine:.4}); \
         this is the signature of the value-head permutation bug (or a regression of its fix)"
    );
}

/// Diagnostic: print every metadata key and tensor name in a GGUF file.
/// Used to pin down the converter's naming scheme for the hybrid arch.
#[test]
#[ignore = "needs a local Qwen3.5 GGUF (CRANE_QWEN35_GGUF)"]
fn dump_gguf_header() {
    let path = std::env::var("CRANE_QWEN35_GGUF").expect("set CRANE_QWEN35_GGUF to a .gguf file");
    let mut file = std::fs::File::open(&path).expect("open gguf");
    let ct = candle_core::quantized::gguf_file::Content::read(&mut file).expect("parse gguf");

    let mut keys: Vec<_> = ct.metadata.keys().collect();
    keys.sort();
    for k in keys {
        let v = &ct.metadata[k];
        let vs = format!("{v:?}");
        let vs = if vs.len() > 120 { format!("{}…", &vs[..120]) } else { vs };
        println!("meta  {k} = {vs}");
    }
    let mut names: Vec<_> = ct.tensor_infos.iter().collect();
    names.sort_by(|a, b| a.0.cmp(b.0));
    for (name, info) in names {
        println!("tensor  {name}  {:?}  {:?}", info.shape, info.ggml_dtype);
    }

    // Norm means reveal whether the converter folded the Gemma-style `+1`
    // into the stored weights (safetensors norms average ~0.24).
    for name in [
        "blk.0.attn_norm.weight",
        "blk.0.post_attention_norm.weight",
        "blk.0.ssm_norm.weight",
        "blk.3.attn_q_norm.weight",
        "output_norm.weight",
    ] {
        let t = ct
            .tensor(&mut file, name, &Device::Cpu)
            .expect("load norm tensor");
        let mean = t
            .dequantize(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        println!("norm-mean  {name} = {mean:.4}");
    }
}

/// Documents why the GGUF value-head order is handled by adapting the Q/K
/// expansion (`VHeadOrder::Chunked`) instead of permuting the affected
/// weights at load time: permuting requires dequantize → reorder →
/// re-quantize, and `quantize` is NOT idempotent for k-quants. This measures
/// the cost of a bare round-trip (no permutation at all) — Q6_K loses ~1% of
/// the weight range, the same order as its original quantization error, while
/// Q8_0 happens to round-trip exactly.
///
/// If you are tempted to "simplify" the fix by reordering weights on load,
/// run this first.
#[test]
#[ignore = "needs a local Qwen3.5 GGUF (CRANE_QWEN35_GGUF)"]
fn requantization_roundtrip_error() {
    use candle_core::quantized::QTensor;

    let path = std::env::var("CRANE_QWEN35_GGUF").expect("set CRANE_QWEN35_GGUF");
    let mut file = std::fs::File::open(&path).unwrap();
    let ct = candle_core::quantized::gguf_file::Content::read(&mut file).unwrap();

    for name in [
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_out.weight",
    ] {
        let qt = ct.tensor(&mut file, name, &Device::Cpu).unwrap();
        let ggml_dtype = qt.dtype();
        let orig = qt.dequantize(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap();

        // Round-trip WITHOUT any permutation: pure quantize(dequantize(x)) cost.
        let rt = QTensor::quantize(&orig, ggml_dtype)
            .unwrap()
            .dequantize(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let err = (&rt - &orig).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        let scale = orig.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        println!(
            "{name}: dtype={ggml_dtype:?} block={} max_abs_roundtrip_err={err:.8} weight_max={scale:.6} rel={:.8}",
            ggml_dtype.block_size(),
            err / scale
        );
    }
}
