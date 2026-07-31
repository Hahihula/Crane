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
/// garbage. Greedy decoding from the same prompt on the safetensors and GGUF
/// paths should therefore agree on at least the first several tokens
/// (exact bit-match isn't expected long-run — Q6_K/Q8_0 quantization noise
/// legitimately diverges greedy argmax after enough tokens, same as any
/// quantized dense model).
#[test]
#[ignore = "needs CRANE_QWEN35_DIR (safetensors) + CRANE_QWEN35_GGUF"]
fn gguf_matches_safetensors_prefix() {
    let st_tokens = {
        let (device, dtype) = device_and_dtype();
        let mut model = Model::new(&model_dir(), &device, &dtype).expect("load safetensors model");
        run_greedy(&mut model, "safetensors")
    };
    let gguf_tokens = {
        let path = std::env::var("CRANE_QWEN35_GGUF").expect("set CRANE_QWEN35_GGUF to a .gguf file");
        let (device, dtype) = device_and_dtype();
        let mut model = Model::new_with_options(&path, &device, &dtype, ModelFormat::Auto, None)
            .expect("load GGUF model");
        run_greedy(&mut model, "gguf")
    };

    const PREFIX_LEN: usize = 8;
    assert_eq!(
        &st_tokens[..PREFIX_LEN],
        &gguf_tokens[..PREFIX_LEN],
        "GGUF and safetensors greedy decoding diverged within the first {PREFIX_LEN} tokens; \
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
