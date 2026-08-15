//! MuScriptor: transcribe audio to MIDI.
//!
//! Minimal CLI for now — `--list-fixtures` works without the model
//! loaded, `--transcribe <wav>` needs a `model_dir` argument that
//! points at a directory containing `config.json` +
//! `model.safetensors`.
//!
//! See `data/audio/muscriptor/README.md` for the fixture format and
//! the CC0 Musopen sourcing rules.

use std::path::Path;

use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "MuScriptor (muscriptor) audio-to-MIDI transcription CLI")]
struct Args {
    /// Path to the MuScriptor checkpoint directory. Must contain
    /// `config.json` and `model.safetensors`.
    #[arg(long, required_unless_present = "list_fixtures")]
    model_dir: Option<String>,

    /// Print a status line for each fixture file under
    /// `data/audio/muscriptor/` and exit. Does not load the model.
    #[arg(long)]
    list_fixtures: bool,

    /// Path to an audio file to transcribe (WAV / MP3 / FLAC / OGG —
    /// anything ffmpeg can decode). Resampled internally to 16 kHz.
    /// Audio of any length is supported — internally split into
    /// consecutive 5 s chunks with tie-prologue forcing across chunk
    /// boundaries, matching the upstream's default `prelude_forcing`.
    #[arg(long)]
    transcribe: Option<String>,

    /// Seconds of audio to feed the model, starting at `--offset`.
    /// Omit to transcribe to the end of the file.
    #[arg(long)]
    duration: Option<f32>,

    /// Optional space-separated comma list of instrument group names
    /// to hard-mask during decoding (verbatim entries of the MT3
    /// vocabulary). See
    /// <https://huggingface.co/MuScriptor/muscriptor-large> for the
    /// list.
    #[arg(long, value_delimiter = ',')]
    instruments: Vec<String>,

    /// Path to write the resulting MIDI file. Defaults to
    /// `data/audio/output/muscriptor_<input-stem>.mid`.
    #[arg(long)]
    output: Option<String>,

    /// Start the audio at this offset (seconds). Useful for skipping
    /// past the silence/intro of a long recording.
    #[arg(long, default_value_t = 0.0)]
    offset: f32,

    /// Sample instead of greedy-decode (temperature/top-k/top-p below
    /// only take effect with this on).
    #[arg(long)]
    use_sampling: bool,

    /// Softmax temperature for `--use-sampling`.
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    /// Top-k filter for `--use-sampling` (0 = disabled).
    #[arg(long, default_value_t = 0)]
    top_k: usize,

    /// Top-p / nucleus filter for `--use-sampling` (0.0 = disabled;
    /// takes priority over `--top-k` when both are set, matching the
    /// upstream).
    #[arg(long, default_value_t = 0.0)]
    top_p: f32,

    /// RNG seed for `--use-sampling`.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Max tokens generated per 5 s chunk (roughly bounds how dense a
    /// chunk's transcription can be before it's cut off).
    #[arg(long, default_value_t = crane_core::models::muscriptor::Model::default_max_gen_len())]
    max_gen_len: usize,

    /// Transformer compute dtype: f32 (default), f16, or bf16. Roughly
    /// halves transformer weight + KV-cache VRAM at f16/bf16 vs f32. The
    /// mel/class conditioners always stay f32 regardless (numerically
    /// required, and a tiny fraction of total weights either way).
    #[arg(long, default_value = "f32")]
    dtype: String,

    /// In-situ-quantize the transformer's linear projections (attention
    /// in/out, FFN, LM head) to this GGML level — e.g. `q8_0`, `q4k`.
    /// Stacks with `--dtype`. Omit to keep them at `--dtype` precision.
    #[arg(long)]
    quant: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.list_fixtures {
        return list_fixtures();
    }

    let model_dir = args
        .model_dir
        .as_deref()
        .context("--model-dir is required for --transcribe")?;
    let wav_path = args
        .transcribe
        .as_deref()
        .context("--transcribe is required")?;

    let device = pick_device();
    let dtype = parse_dtype(&args.dtype)?;
    let quant = args
        .quant
        .as_deref()
        .map(crane_core::ops::linear::parse_ggml_dtype)
        .transpose()?;

    eprintln!(
        "loading MuScriptor from {model_dir} on {device:?} (dtype={dtype:?}, quant={:?})",
        args.quant
    );
    let transcription = crane_core::models::muscriptor::TranscriptionModel::load_with_options(
        model_dir, &device, dtype, quant,
    )?;

    let (raw_samples, sample_rate) = read_wav_mono_f32(wav_path, args.offset, args.duration)?;
    eprintln!(
        "{}: {} samples @ {} Hz fed (offset={:.2}s, duration={:.2}s)",
        wav_path,
        raw_samples.len(),
        sample_rate,
        args.offset,
        raw_samples.len() as f32 / sample_rate as f32
    );

    let cfg = crane_core::models::muscriptor::TranscribeConfig {
        instruments: if args.instruments.is_empty() {
            None
        } else {
            Some(args.instruments.clone())
        },
        hard_mask_instruments: !args.instruments.is_empty(),
        use_sampling: args.use_sampling,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
        max_gen_len: args.max_gen_len,
        ..Default::default()
    };

    let bytes = transcription.transcribe_to_midi(&raw_samples, sample_rate, &cfg)?;
    let output = args.output.unwrap_or_else(|| {
        let stem = Path::new(wav_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("muscriptor");
        format!("data/audio/output/muscriptor_{stem}.mid")
    });
    if let Some(parent) = Path::new(&output).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&output, &bytes)
        .with_context(|| format!("write {output}"))?;
    eprintln!("wrote {} ({} bytes)", output, bytes.len());
    Ok(())
}

fn list_fixtures() -> Result<()> {
    let dir = Path::new("data/audio/muscriptor");
    if !dir.exists() {
        eprintln!(
            "no fixtures installed at {}\n\
             see {}/README.md for download instructions (CC0 Musopen).\n",
            dir.display(),
            dir.display()
        );
        std::process::exit(2);
    }
    let mut any = false;
    for entry in std::fs::read_dir(dir)?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("wav") {
            any = true;
            let metadata = std::fs::metadata(&path).ok();
            let size = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
            println!(
                "{}  ({} bytes)",
                path.file_name().unwrap().to_string_lossy(),
                size
            );
        }
    }
    if !any {
        eprintln!(
            "fixture directory {} exists but contains no .wav files;\n\
             see {}/README.md for download instructions.",
            dir.display(),
            dir.display()
        );
        std::process::exit(2);
    }
    Ok(())
}

fn pick_device() -> candle_core::Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = candle_core::Device::new_cuda(0) {
        return d;
    }
    #[cfg(feature = "metal")]
    if let Ok(d) = candle_core::Device::new_metal(0) {
        return d;
    }
    candle_core::Device::Cpu
}

fn parse_dtype(name: &str) -> Result<candle_core::DType> {
    match name.to_lowercase().as_str() {
        "f32" | "fp32" => Ok(candle_core::DType::F32),
        "f16" | "fp16" | "half" => Ok(candle_core::DType::F16),
        "bf16" => Ok(candle_core::DType::BF16),
        other => anyhow::bail!("unsupported --dtype '{other}' (expected f32, f16 or bf16)"),
    }
}

/// Read a mono-f32 PCM WAV (or any audio format ffmpeg can decode,
/// via a temporary WAV round-trip). `offset` and `duration` are
/// applied at the decode layer (for non-WAV inputs, ffmpeg does the
/// trimming; for WAV inputs, we trim in Rust after the read).
/// `duration = None` reads to the end of the file. Returns `(samples,
/// sample_rate)` — the ffmpeg path always resamples to 16 kHz, but a raw
/// `.wav` input is read at its own native rate, so the caller must not
/// assume 16 kHz.
fn read_wav_mono_f32(path: &str, offset: f32, duration: Option<f32>) -> Result<(Vec<f32>, u32)> {
    // Non-WAV inputs: decode via ffmpeg into a temp WAV. ffmpeg handles
    // MP3, FLAC, OGG, M4A, etc. with no extra Rust deps.
    let lower = path.to_ascii_lowercase();
    let wav_path = if lower.ends_with(".wav") {
        path.to_string()
    } else {
        let tmp = std::env::temp_dir().join(format!(
            "muscriptor-{}.wav",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let mut cmd = std::process::Command::new("ffmpeg");
        cmd.args(["-y", "-loglevel", "error", "-ss", &format!("{offset}"), "-i", path]);
        if let Some(d) = duration.filter(|&d| d > 0.0) {
            cmd.args(["-t", &format!("{d}")]);
        }
        cmd.args([
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            tmp.to_str().context("tmp path")?,
        ]);
        let status = cmd.status().with_context(|| format!("spawn ffmpeg for {path}"))?;
        if !status.success() {
            anyhow::bail!("ffmpeg failed to decode {path} (exit {status})");
        }
        tmp.to_str().context("tmp path")?.to_string()
    };

    let reader = hound::WavReader::open(&wav_path)
        .with_context(|| format!("open {wav_path}"))?;
    let spec = reader.spec();
    let samples_f32: Vec<f32> = match (spec.bits_per_sample, spec.sample_format) {
        (16, _) => reader
            .into_samples::<i16>()
            .map(|s| match s {
                Ok(x) => Ok(x as f32 / i16::MAX as f32),
                Err(e) => Err(e),
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("decode {wav_path}: {e}"))?,
        (32, hound::SampleFormat::Float) => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap_or(0.0))
            .collect(),
        _ => anyhow::bail!(
            "unsupported WAV format: {}-bit {:?}",
            spec.bits_per_sample,
            spec.sample_format
        ),
    };
    // Mono-ize: average the channels.
    let channels = spec.channels as usize;
    let mono: Vec<f32> = if channels == 1 {
        samples_f32
    } else {
        let frames = samples_f32.len() / channels;
        let mut mono = Vec::with_capacity(frames);
        for i in 0..frames {
            let mut s = 0.0f32;
            for c in 0..channels {
                s += samples_f32[i * channels + c];
            }
            mono.push(s / channels as f32);
        }
        mono
    };

    // Apply offset/duration (WAV inputs are not trimmed by ffmpeg, so do it
    // in Rust, at the WAV's own sample rate — `main` passes `sr` through to
    // `transcribe_to_midi`, which resamples to 16 kHz itself if needed).
    let sr = spec.sample_rate as f32;
    let start = (offset * sr).max(0.0) as usize;
    let Some(duration) = duration else {
        // No --duration: everything from `start` to the end of the file.
        let out = mono.get(start..).unwrap_or_default().to_vec();
        return Ok((out, spec.sample_rate));
    };
    if start >= mono.len() {
        let pad = (duration * sr).max(160.0) as usize;
        return Ok((vec![0.0_f32; pad.min(spec.sample_rate as usize)], spec.sample_rate));
    }
    let take = (duration * sr).max(160.0) as usize;
    let mut out = mono[start..(start + take).min(mono.len())].to_vec();
    if out.len() < take {
        out.resize(take, 0.0);
    }
    Ok((out, spec.sample_rate))
}
