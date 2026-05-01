//! Gemma4 Vision-Language Model test example.

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use clap::Parser;
use std::path::PathBuf;

use crane_core::models::gemma4::vision::{load_and_preprocess_image, ImagePreprocessConfig};
use crane_core::models::gemma4::vlm::Gemma4VLModel;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "checkpoints/gemma-4-E2B")]
    model_path: PathBuf,
    #[arg(long, default_value = "data/images/seal.png")]
    image_path: PathBuf,
    #[arg(long, default_value = "Describe this image in one sentence.")]
    prompt: String,
    #[arg(long, default_value = "50")]
    max_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = Device::cuda_if_available(0)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    eprintln!("Loading Gemma4-VLM from: {:?}", args.model_path);
    eprintln!("Device: {:?}, dtype: {:?}", device, dtype);

    let mut vlm = Gemma4VLModel::new(args.model_path.to_str().unwrap(), &device, &dtype)?;
    eprintln!("Model loaded successfully");

    eprintln!("Loading image: {:?}", args.image_path);
    let preprocess_config = ImagePreprocessConfig::default();
    let preprocessed = load_and_preprocess_image(&args.image_path, &preprocess_config, &device)?;
    eprintln!(
        "Image preprocessed: {} patches, {} image tokens",
        preprocessed.pixel_values.dim(1)?,
        preprocessed.num_image_tokens
    );

    let image_embeds = vlm.encode_image(
        &preprocessed.pixel_values,
        &preprocessed.pixel_position_ids,
        &preprocessed.padding_positions,
        Some(preprocessed.num_image_tokens),
    )?;

    // Encode the text prompt
    let text_ids = vlm
        .tokenizer
        .tokenizer
        .encode(args.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {}", e))?
        .get_ids()
        .to_vec();

    // Build VLM prompt: BOS + image tokens + text tokens + EOS
    let num_img_to_use = preprocessed.num_image_tokens;
    let image_token_id = vlm.image_token_id;
    let mut vlm_prompt = vec![2u32]; // BOS
    for _ in 0..num_img_to_use {
        vlm_prompt.push(image_token_id);
    }
    vlm_prompt.extend_from_slice(&text_ids);
    vlm_prompt.push(1); // EOS

    vlm.clear_kv_cache();
    // Store image embeddings for use during generation
    vlm.store_image_embeddings(&image_embeds, num_img_to_use);

    let vlm_input = Tensor::new(vlm_prompt.as_slice(), &device)?.unsqueeze(0)?;
    let image_embeds_sliced = image_embeds.narrow(1, 0, num_img_to_use)?;
    let vlm_logits = vlm.forward(&vlm_input, Some(&image_embeds_sliced), 0)?;
    let vlm_logits = vlm_logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

    // Generate tokens
    let mut tokens = vlm_prompt.clone();
    let mut next_t = candle_nn::ops::softmax_last_dim(&vlm_logits)?
        .argmax(D::Minus1)?
        .to_scalar::<u32>()?;

    print!("Description: ");
    for _ in 0..args.max_tokens {
        if next_t == 1 {
            break;
        }
        let input = Tensor::new(&[next_t], &device)?.unsqueeze(0)?;
        let logits = vlm.forward(&input, None, tokens.len() - 1)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        next_t = candle_nn::ops::softmax_last_dim(&logits)?
            .argmax(D::Minus1)?
            .to_scalar::<u32>()?;
        if let Ok(dec) = vlm.tokenizer.tokenizer.decode(&[next_t], true) {
            print!("{}", dec);
        }
        tokens.push(next_t);
    }
    println!();

    Ok(())
}
