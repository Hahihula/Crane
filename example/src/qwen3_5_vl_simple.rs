//! Simple Qwen 3.5 vision-language example using crane-core directly.
//!
//! Describes an image with a multimodal Qwen 3.5 checkpoint
//! (`Qwen3_5ForConditionalGeneration` — a text-only checkpoint has no
//! `vision_config` and will be rejected at load).
//!
//! Usage:
//!   cargo run --bin qwen3_5_vl_simple -- /path/to/Qwen3.5-4B image.png
//!   cargo run --bin qwen3_5_vl_simple --features cuda -- <model> <image> "what colour is the bird?"

use anyhow::{Context, Result};
use crane_core::models::DType;
use crane_core::models::qwen3_5::{Qwen3_5VLModel, VlGenerationConfig};
use std::io::Write;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .unwrap_or_else(|| "model/Qwen3.5-4B".to_string());
    let image_path = args
        .next()
        .context("usage: qwen3_5_vl_simple <model-dir> <image> [prompt]")?;
    let prompt = args
        .next()
        .unwrap_or_else(|| "Briefly describe this image.".to_string());

    let device = crane_core::models::Device::cuda_if_available(0)?;
    let dtype = if device.is_cuda() {
        DType::F16
    } else {
        DType::F32
    };

    eprintln!("Loading Qwen 3.5 VL from: {model_path}");
    eprintln!("Device: {device:?}, dtype: {dtype:?}");
    let mut model = Qwen3_5VLModel::new(&model_path, &device, &dtype)?;

    let image = image::open(&image_path).with_context(|| format!("open {image_path}"))?;
    eprintln!("Image: {image_path} ({}x{})", image.width(), image.height());
    eprintln!("Prompt: {prompt}\n");

    let config = VlGenerationConfig {
        max_new_tokens: 256,
        ..Default::default()
    };

    // Stream tokens as they are decoded, then keep the assembled answer.
    let started = std::time::Instant::now();
    let answer = model.generate(Some(&image), &prompt, &config, |tok| {
        print!("{tok}");
        let _ = std::io::stdout().flush();
    })?;
    println!();

    eprintln!(
        "\n--- {} chars in {:.2?} ---",
        answer.len(),
        started.elapsed()
    );
    Ok(())
}
