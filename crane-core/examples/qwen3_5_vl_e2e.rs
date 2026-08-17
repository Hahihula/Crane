//! End-to-end smoke test for Qwen 3.5 VL: load a multimodal checkpoint,
//! preprocess an image, build the input with `<|image_pad|>` placeholders,
//! run prefill, and print the top-K argmax tokens. If the top-1 is a
//! plausible next word (English / Chinese), the splice + 3D-MRoPE math
//! is right; if it's `<|im_end|>` or NaN, something's off.
//!
//! Gated by `CRANE_QWEN35_VL_DIR`. Optional `CRANE_QWEN35_VL_IMAGE` to
//! point at a PNG/JPEG (defaults to `/tmp/test_image.png`).

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};

use crane_core::models::qwen3_5::Qwen3_5VLModel;
use crane_core::models::qwen3_5::processor::load_preprocessor_config;

const TEST_IMAGE: &str = "/tmp/test_image.png";

fn device_and_dtype() -> (candle_core::Device, DType) {
    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        return (candle_core::Device::new_cuda(0).unwrap(), DType::F16);
    }
    (candle_core::Device::Cpu, DType::F32)
}

fn main() -> Result<()> {
    let dir = std::env::var("CRANE_QWEN35_VL_DIR")
        .context("set CRANE_QWEN35_VL_DIR to a Qwen3.5 multimodal checkpoint dir")?;
    let image_path =
        std::env::var("CRANE_QWEN35_VL_IMAGE").unwrap_or_else(|_| TEST_IMAGE.to_string());

    println!("[qwen3_5_vl_e2e] loading model from {dir}");
    let (device, dtype) = device_and_dtype();
    let mut model = Qwen3_5VLModel::new(&dir, &device, &dtype)?;

    println!("[qwen3_5_vl_e2e] loading image from {image_path}");
    let img = image::open(&image_path).context("open image")?;
    let pp = load_preprocessor_config(&dir)?;
    let processed = pp.process(&img, &device, dtype)?;
    let (t, h, w) = processed.grid_thw;
    println!(
        "[qwen3_5_vl_e2e] preprocessed: grid_thw=({t},{h},{w}), {} patches",
        processed.pixel_values.dim(0)?
    );

    let merge = pp.merge_size;
    let n_image_tokens = ((h / merge as u32) * (w / merge as u32)) as usize;

    let img_token_str = "<|image_pad|>";
    let img_pad_id = model
        .tokenizer
        .tokenizer
        .get_vocab(true)
        .get(img_token_str)
        .copied()
        .context("<|image_pad|> not in vocab")? as u32;

    let prompt_text = format!(
        "<|im_start|>user\n<|vision_start|>{img_token_str}<|vision_end|>Briefly: what is in this image? Answer directly without thinking.<|im_end|>\n<|im_start|>assistant\n"
    );

    let base_ids = model
        .tokenizer
        .tokenizer
        .encode(prompt_text, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    let mut input_ids: Vec<u32> = Vec::with_capacity(base_ids.len() - 1 + n_image_tokens);
    let mut inserted = false;
    for &tok in &base_ids {
        if tok == img_pad_id {
            if inserted {
                anyhow::bail!("duplicate <|image_pad|> in prompt");
            }
            for _ in 0..n_image_tokens {
                input_ids.push(img_pad_id);
            }
            inserted = true;
        } else {
            input_ids.push(tok);
        }
    }
    if !inserted {
        anyhow::bail!("<|image_pad|> missing from encoded prompt");
    }
    println!(
        "[qwen3_5_vl_e2e] input_ids length: {} ({} image + {} text)",
        input_ids.len(),
        n_image_tokens,
        input_ids.len() - n_image_tokens
    );

    let input_tensor = Tensor::from_vec(input_ids.clone(), (1usize, input_ids.len()), &device)?;
    let grid_tensor = Tensor::from_vec(vec![t, h, w], (1usize, 3usize), &device)?;

    println!("[qwen3_5_vl_e2e] running prefill...");
    let prefill_start = std::time::Instant::now();
    let logits = model.forward(
        &input_tensor,
        Some(&processed.pixel_values),
        Some(&grid_tensor),
        0,
    )?;
    let prefill_ms = prefill_start.elapsed().as_millis();
    println!(
        "[qwen3_5_vl_e2e] prefill done in {prefill_ms} ms, logits shape {:?}",
        logits.dims()
    );

    // Decode a few tokens past the <think> marker to show actual content.
    let mut generated = vec![248_068u32]; // <think> (the model's first impulse)
    let mut cur_logits = logits;
    let mut cur_pos = input_ids.len(); // next position to decode
    let max_steps: usize = std::env::var("CRANE_QWEN35_VL_MAXTOK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    for step in 1..max_steps {
        let v: Vec<f32> = cur_logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1()?;
        let argmax = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .0 as u32;
        let txt = model
            .tokenizer
            .tokenizer
            .decode(&[argmax], true)
            .unwrap_or_default();
        println!("[qwen3_5_vl_e2e] step {step:2}: id={argmax:6}  text={txt:?}");
        if argmax == 248_044 || argmax == 248_045 || argmax == 248_046 {
            println!("[qwen3_5_vl_e2e] hit EOS");
            break;
        }
        generated.push(argmax);
        // Decode the next token (KV cache already populated by the prefill).
        cur_logits = model.decode_step(argmax, cur_pos)?;
        cur_pos += 1;
    }

    let full = model
        .tokenizer
        .tokenizer
        .decode(&generated, true)
        .unwrap_or_default();
    println!("[qwen3_5_vl_e2e] decoded text:\n{full}");
    Ok(())
}
