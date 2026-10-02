//! `KugelAudio` Simple Example
//!
//! Generates speech from text using `KugelAudio`
//! (`kugelaudio/kugelaudio-0-open`, a post-trained fine-tune of Microsoft's
//! `VibeVoice`): zero-shot with the model's built-in voice, or conditioned on
//! a reference clip for voice cloning (raw audio, resampled to 24kHz — no
//! prompt transcript needed).
//!
//! Unlike Crane's other TTS checkpoints, `KugelAudio` ships **no tokenizer of
//! its own** — it needs a Qwen2-VL-family `tokenizer.json` (only the
//! Qwen2-VL vocab defines `<|vision_start/end/pad|>` at the fixed ids this
//! port hardcodes as speech control tokens). `KugelAudioModel::from_pretrained`
//! loads it automatically from `<model_path>/tokenizer.json`. See
//! `crane_core::models::kugelaudio::prompt` for the full explanation.
//!
//! # Usage
//!
//! ```bash
//! # Zero-shot, built-in example sentence
//! cargo run --bin kugelaudio_simple --release --features cuda -- \
//!     checkpoints/kugelaudio-0-open
//!
//! # Voice cloning from a reference clip (no transcript needed)
//! cargo run --bin kugelaudio_simple --release --features cuda -- \
//!     checkpoints/kugelaudio-0-open "Text to speak" --ref-wav ref.wav
//!
//! # Preset voice shipped with the checkpoint (--ref-wav and --voice are
//! # mutually exclusive)
//! cargo run --bin kugelaudio_simple --release --features cuda -- \
//!     checkpoints/kugelaudio-0-open "Text to speak" --voice english_female
//!
//! # macOS: use --features metal instead of --features cuda
//!
//! # Low-VRAM/low-RAM: 4-bit in-situ quantization of the decoder backbone
//! # (~7B parameters, the bulk of the ~18.7GB checkpoint) — works with
//! # either --features cuda or --features metal
//! cargo run --bin kugelaudio_simple --release --features metal -- \
//!     checkpoints/kugelaudio-0-open --quant q4_0
//! ```
//!
//! **Quantization caveat**: `--quant`/`CRANE_ISQ` is unit-tested on CPU
//! and Metal with synthetic weights, but loading the real checkpoint
//! end-to-end on a memory-constrained machine has not been verified. On an
//! 18GB Mac, attempts to load (even at `--quant q4_0`) repeatedly exhausted
//! system memory badly enough to crash the whole machine — the loading
//! path allocates in bursts that can outrun an external memory watchdog.
//! Watch memory closely (or use a tool that can hard-kill the process) rather
//! than assuming `q4_0` is safe on a tight-memory machine.

#![allow(clippy::doc_markdown)] // KugelAudio / VibeVoice are external names
#![allow(clippy::cast_precision_loss)] // u128 nanos / SAMPLE_RATE: bounded
#![allow(clippy::cast_possible_truncation)] // u128->u64 for nanos; PID u32->u64
#![allow(clippy::too_many_lines)] // main() does device setup, load, generate, save

use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "KugelAudio TTS demo: zero-shot or reference-audio-conditioned (voice cloning)")]
struct Args {
    /// Path to the KugelAudio checkpoint directory (must contain
    /// config.json + model-*.safetensors / model.safetensors.index.json,
    /// plus a Qwen2-VL-family tokenizer.json — see module doc comment).
    model_path: String,
    /// Text to synthesize.
    #[arg(default_value = "Hello! I am Crane, an ultra-fast inference engine written in Rust.")]
    text: String,
    /// Reference audio clip for voice cloning (raw audio, resampled to
    /// 24kHz internally — no transcript needed). Mutually exclusive with
    /// --voice.
    #[arg(long, conflicts_with = "voice")]
    ref_wav: Option<String>,
    /// Preset voice name from the checkpoint's voices/ directory (e.g.
    /// "default", "clear", "english_female", "english_male"). Mutually
    /// exclusive with --ref-wav.
    #[arg(long)]
    voice: Option<String>,
    #[arg(long, default_value = "data/audio/output")]
    output_dir: String,
    /// CFG scale (1.0 disables CFG). Matches the checkpoint's own default.
    #[arg(long, default_value_t = 3.0)]
    cfg_scale: f64,
    #[arg(long, default_value_t = 2048)]
    max_new_tokens: usize,
    /// Sample instead of greedy-decoding the control-token stream.
    #[arg(long)]
    do_sample: bool,
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,
    /// Force CPU/F32 compute instead of the per-device default (BF16 on
    /// CUDA, F16 on Metal).
    #[arg(long)]
    cpu: bool,
    /// Seed for the diffusion-sampling noise. Defaults to a fresh random
    /// value every launch.
    #[arg(long)]
    seed: Option<u64>,
    /// In-situ-quantize the decoder backbone (the ~7B-parameter Qwen2
    /// stack, the bulk of the ~18.7GB checkpoint) to this `GgmlDType` as it
    /// loads, e.g. `q4_0` for 4-bit — the lever for low-VRAM GPUs or
    /// low-RAM machines. One of `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`,
    /// `q2k`, `q3k`, `q4k`, `q5k`, `q6k`. Works on CUDA and Metal. Falls
    /// back to `CRANE_ISQ` when unset.
    #[arg(long)]
    quant: Option<String>,
}

fn main() -> anyhow::Result<()> {
    use crane_core::candle_core::{DType, Device, Tensor};
    use crane_core::models::kugelaudio::prompt::SAMPLE_RATE;
    use crane_core::models::kugelaudio::{KugelAudioGenerationConfig, KugelAudioModel};
    use std::time::{SystemTime, UNIX_EPOCH};

    let args = Args::parse();

    let (device, dtype) = if args.cpu {
        (Device::Cpu, DType::F32)
    } else {
        #[cfg(feature = "cuda")]
        {
            (Device::new_cuda(0).unwrap_or(Device::Cpu), DType::BF16)
        }
        #[cfg(all(target_os = "macos", not(feature = "cuda")))]
        {
            (Device::new_metal(0).unwrap_or(Device::Cpu), DType::F16)
        }
        #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
        {
            (Device::Cpu, DType::F32)
        }
    };
    if matches!(device, Device::Cpu) {
        eprintln!(
            "WARNING: KugelAudio on CPU will be slow (28-layer decoder + 20-step diffusion per frame). GPU strongly recommended."
        );
    } else {
        // candle's CUDA/Metal backends default to a fixed RNG seed
        // (299792458), so the diffusion noise would be identical on every
        // launch without this. CPU's RNG is OS-entropy-seeded per run.
        let seed = args.seed.unwrap_or_else(|| {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
            nanos ^ (std::process::id() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        });
        println!("Seed: {seed}");
        device.set_seed(seed)?;
    }

    let quant = match args.quant.as_deref() {
        Some(name) => Some(
            crane_core::ops::linear::parse_ggml_dtype(name)
                .map_err(|e| anyhow::anyhow!("invalid --quant: {e}"))?,
        ),
        None => std::env::var("CRANE_ISQ")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(|s| crane_core::ops::linear::parse_ggml_dtype(&s))
            .transpose()
            .map_err(|e| anyhow::anyhow!("invalid CRANE_ISQ: {e}"))?,
    };

    let voice_waveform = args
        .ref_wav
        .as_deref()
        .map(|path| -> anyhow::Result<_> {
            let samples = crane::audio::load_wav_f32(path, SAMPLE_RATE)?;
            let n = samples.len();
            let waveform = Tensor::from_vec(samples, (1, 1, n), &device)?;
            Ok((waveform, n))
        })
        .transpose()?;

    println!("Loading KugelAudio from: {}", args.model_path);
    println!("Device: {device:?}  dtype: {dtype:?}  quant: {quant:?}");
    let mut model =
        KugelAudioModel::from_pretrained_with_quant(&args.model_path, &device, dtype, quant)?;

    // Tensor::clone() is Arc::clone (no data copy) -- needed here so the
    // preset's borrowed tensors outlive the &self lookup, since generate()
    // below needs &mut self.
    let voice_latents = args
        .voice
        .as_deref()
        .map(|name| -> anyhow::Result<_> {
            let voice = model.available_voices().get(name).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown voice '{name}' (available: {:?})",
                    model.available_voices().keys().collect::<Vec<_>>()
                )
            })?;
            Ok((voice.acoustic_mean.clone(), voice.semantic_mean.clone()))
        })
        .transpose()?;

    println!(
        "Mode: {}",
        if let Some(name) = args.voice.as_deref() {
            format!("preset voice '{name}'")
        } else if voice_waveform.is_some() {
            "voice cloning (--ref-wav)".to_string()
        } else {
            "zero-shot".to_string()
        }
    );
    println!("Text: {}", args.text);

    let prompt = if let Some(name) = args.voice.as_deref() {
        model.build_prompt_for_voice(&args.text, name)?
    } else {
        model.build_prompt(&args.text, voice_waveform.as_ref().map(|(_, n)| *n))?
    };

    let gen_cfg = KugelAudioGenerationConfig {
        cfg_scale: args.cfg_scale,
        max_new_tokens: args.max_new_tokens,
        do_sample: args.do_sample,
        temperature: args.temperature,
    };

    let start = std::time::Instant::now();
    let out = model.generate(
        &prompt,
        voice_latents.as_ref().map(|(a, s)| (a, s)),
        voice_waveform.as_ref().map(|(w, _)| w),
        &gen_cfg,
    )?;
    println!(
        "Generated {} control tokens, {:.2}s audio in {:.1?}",
        out.token_ids.len(),
        out.audio.len() as f32 / SAMPLE_RATE as f32,
        start.elapsed()
    );

    if out.audio.is_empty() {
        anyhow::bail!(
            "generation produced no audio (model emitted speech_end/eos before any diffusion token)"
        );
    }

    std::fs::create_dir_all(&args.output_dir)?;
    let output_path = format!("{}/kugelaudio_output.wav", args.output_dir);
    let n = out.audio.len();
    let wav = Tensor::from_vec(out.audio, (1, 1, n), &device)?;
    let saved_path = crane::audio::save_wav(&wav, &output_path, SAMPLE_RATE)?;
    println!("Saved {saved_path}");

    Ok(())
}
