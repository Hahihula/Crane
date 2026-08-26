//! Qwen3-ASR 0.6B transcription example.
//!
//! The audio is decoded, mixed down to mono, and resampled to 16 kHz through
//! Crane's audio SDK before it is passed to the unified [`crane::audio::Asr`]
//! interface.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p crane-examples --bin asr_qwen3_simple --release -- \
//!   vendor/Qwen3-ASR-0.6B data/audio/sample.wav
//! ```
//!
//! Both arguments are optional. The defaults are `vendor/Qwen3-ASR-0.6B` and
//! `data/audio/sample.wav`.

use anyhow::{Context, Result};
use crane::audio::{Asr, TranscribeOptions, load_wav_f32};
use crane_core::models::{DType, Device, qwen3_asr};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .unwrap_or_else(|| "vendor/Qwen3-ASR-0.6B".to_string());
    let audio_path = args
        .next()
        .unwrap_or_else(|| "data/audio/sample.wav".to_string());
    anyhow::ensure!(
        args.next().is_none(),
        "usage: asr_qwen3_simple [MODEL_DIR] [AUDIO.wav]"
    );

    let device = pick_device();
    let dtype = pick_dtype(&device);
    println!("Loading Qwen3-ASR 0.6B from {model_path}");
    println!("Device: {device:?}  dtype: {dtype:?}");

    // `qwen3_asr::Model` implements Crane's model-agnostic `Asr` trait.
    let mut model = qwen3_asr::Model::new(&model_path, &device, &dtype)
        .with_context(|| format!("load Qwen3-ASR checkpoint from {model_path}"))?;

    let audio = load_wav_f32(&audio_path, model.input_sample_rate())
        .with_context(|| format!("decode audio file {audio_path}"))?;
    println!(
        "Transcribing {audio_path} ({} samples at {} Hz)...",
        audio.len(),
        model.input_sample_rate()
    );

    let transcript = Asr::transcribe(&mut model, &audio, &TranscribeOptions::default())
        .context("transcribe audio")?;
    println!("\nTranscription: {}", transcript.text);

    Ok(())
}

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(device) = Device::new_cuda(0) {
        return device;
    }

    #[cfg(all(not(feature = "cuda"), target_os = "macos"))]
    if let Ok(device) = Device::new_metal(0) {
        return device;
    }

    #[cfg(all(not(feature = "cuda"), not(target_os = "macos"), feature = "rocm"))]
    if let Ok(device) = Device::new_rocm(0) {
        return device;
    }

    Device::Cpu
}

fn pick_dtype(device: &Device) -> DType {
    if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::F16
    }
}
