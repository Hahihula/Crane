// SPDX-License-Identifier: MIT

//! Preset voice loading for `KugelAudioModel`.
//!
//! The reference checkpoint ships a `voices/` directory with a handful of
//! pre-encoded reference voices: each `voices/<name>.pt` is a `torch.save`'d
//! dict holding the cached `encode_acoustic`/`encode_semantic` output of some
//! reference clip (pre-`scale_acoustic_latent`, pre-connector), indexed by
//! `voices/voices.json`. A checkpoint with no `voices/` directory simply has
//! no preset voices — not an error.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

use super::config::KugelAudioConfig;
use crate::models::modules::voice_embedding::load_pt_tensor_bytes;

/// One preset voice: pre-encoded acoustic/semantic latents plus metadata
/// from `voices.json`.
#[derive(Debug)]
pub struct KugelAudioVoice {
    /// `[1, acoustic_vae_dim, T]`, BF16, pre-connector — same shape
    /// `KugelAudioModel::encode_acoustic` returns.
    pub acoustic_mean: Tensor,
    /// `[1, semantic_vae_dim, T]`, BF16, pre-connector — same shape
    /// `KugelAudioModel::encode_semantic` returns.
    pub semantic_mean: Tensor,
    /// ISO-639-1-ish language code this preset was recorded in (e.g. `"de"`).
    pub language: String,
    /// Human-readable description from `voices.json` (e.g. "Calm female narrator").
    pub description: String,
}

#[derive(Debug, Clone, Deserialize)]
struct VoiceEntry {
    file: String,
    description: String,
    language: String,
}

/// Load every preset voice from `<model_dir>/voices/voices.json`.
///
/// Returns an empty map if the checkpoint ships no `voices/voices.json` —
/// not all checkpoints (e.g. fine-tunes) include preset voices.
///
/// # Errors
///
/// Returns an error if `voices.json` exists but is malformed, a referenced
/// `.pt` file is missing or not a valid `PyTorch` ZIP archive, or a loaded
/// tensor's byte length isn't an exact multiple of its expected element size.
pub fn load_voices(
    model_dir: &Path,
    config: &KugelAudioConfig,
    device: &Device,
) -> Result<HashMap<String, KugelAudioVoice>> {
    let voices_dir = model_dir.join("voices");
    let index_path = voices_dir.join("voices.json");
    if !index_path.exists() {
        return Ok(HashMap::new());
    }

    let index_bytes = std::fs::read(&index_path)
        .with_context(|| format!("failed to read {}", index_path.display()))?;
    let entries: HashMap<String, VoiceEntry> = serde_json::from_slice(&index_bytes)
        .with_context(|| format!("failed to parse {}", index_path.display()))?;

    let canon_voices = voices_dir
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", voices_dir.display()))?;

    let mut voices = HashMap::with_capacity(entries.len());
    for (name, entry) in entries {
        let pt_path = voices_dir.join(&entry.file);
        let canon_pt = pt_path
            .canonicalize()
            .with_context(|| format!("voice '{name}': failed to resolve {}", pt_path.display()))?;
        anyhow::ensure!(
            canon_pt.starts_with(&canon_voices),
            "voice '{name}': file path escapes voices directory: {}",
            entry.file,
        );
        let acoustic_mean = load_latent_tensor(&pt_path, 0, config.acoustic_vae_dim, device)
            .with_context(|| {
                format!(
                    "voice '{name}': loading acoustic_mean from {}",
                    pt_path.display()
                )
            })?;
        // Index 1 is `acoustic_std`, unused: KugelAudioModel's encoders
        // always take the latent mean, never sample the distribution.
        let semantic_mean = load_latent_tensor(&pt_path, 2, config.semantic_vae_dim, device)
            .with_context(|| {
                format!(
                    "voice '{name}': loading semantic_mean from {}",
                    pt_path.display()
                )
            })?;

        voices.insert(
            name,
            KugelAudioVoice {
                acoustic_mean,
                semantic_mean,
                language: entry.language,
                description: entry.description,
            },
        );
    }

    Ok(voices)
}

/// Read the `index`-th tensor storage from a voice `.pt` file and reshape it
/// to `[1, dim, T]` BF16, inferring `T` from the byte length.
fn load_latent_tensor(path: &Path, index: usize, dim: usize, device: &Device) -> Result<Tensor> {
    let bytes = load_pt_tensor_bytes(path, index)?;

    // Each BF16 element is 2 bytes; the tensor is [1, dim, T].
    anyhow::ensure!(
        dim > 0,
        "vae dim is 0 for voice tensor at {}",
        path.display()
    );
    anyhow::ensure!(
        bytes.len() % (dim * 2) == 0,
        "tensor size {} is not a multiple of {} (dim * 2 bytes)",
        bytes.len(),
        dim * 2,
    );
    let t = bytes.len() / (dim * 2);

    Tensor::from_raw_buffer(&bytes, DType::BF16, &[1, dim, t], device)
        .context("failed to create latent tensor from raw bytes")
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::*;

    const MINIMAL_CONFIG_JSON: &str = r#"{
      "decoder_config": {
        "vocab_size": 1, "hidden_size": 1, "intermediate_size": 1,
        "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
        "max_position_embeddings": 1, "rope_theta": 1.0, "rms_norm_eps": 1e-6,
        "hidden_act": "silu"
      },
      "acoustic_tokenizer_config": {
        "channels": 1, "vae_dim": 64, "encoder_n_filters": 1, "encoder_ratios": [1],
        "encoder_depths": "1", "causal": true, "conv_bias": true, "conv_norm": "none",
        "pad_mode": "constant", "layernorm": "RMSNorm", "layernorm_eps": 1e-5,
        "layernorm_elementwise_affine": true, "mixer_layer": "depthwise_conv",
        "layer_scale_init_value": 1e-6, "disable_last_norm": true
      },
      "semantic_tokenizer_config": {
        "channels": 1, "vae_dim": 128, "encoder_n_filters": 1, "encoder_ratios": [1],
        "encoder_depths": "1", "causal": true, "conv_bias": true, "conv_norm": "none",
        "pad_mode": "constant", "layernorm": "RMSNorm", "layernorm_eps": 1e-5,
        "layernorm_elementwise_affine": true, "mixer_layer": "depthwise_conv",
        "layer_scale_init_value": 1e-6, "disable_last_norm": true
      },
      "diffusion_head_config": {
        "hidden_size": 1, "latent_size": 1, "head_layers": 1, "head_ffn_ratio": 1.0,
        "rms_norm_eps": 1e-5, "prediction_type": "v_prediction",
        "ddpm_beta_schedule": "cosine", "ddpm_num_steps": 1, "ddpm_num_inference_steps": 1,
        "ddpm_algorithm_type": "sde-dpmsolver++"
      },
      "acoustic_vae_dim": 64,
      "semantic_vae_dim": 128
    }"#;

    // Verifies a missing voices/voices.json yields an empty map rather than
    // an error, since not all checkpoints ship preset voices.
    #[test]
    fn test_load_voices_missing_index_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let config: KugelAudioConfig = serde_json::from_str(MINIMAL_CONFIG_JSON).unwrap();
        let voices = load_voices(dir.path(), &config, &Device::Cpu).unwrap();
        assert!(voices.is_empty());
    }

    // Verifies the raw tensor bytes from a voice .pt entry are reshaped to
    // [1, dim, T] with T inferred from the byte length.
    #[test]
    fn test_load_latent_tensor_reshapes_correctly() {
        let dim = 4;
        let t = 3;
        let raw = vec![0u8; dim * t * 2]; // all-zero BF16

        let mut buf = Vec::new();
        {
            let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("tmpabc/data/0", opts).unwrap();
            zip.write_all(&raw).unwrap();
            zip.finish().unwrap();
        }
        let tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.as_file().write_all(&buf).unwrap();

        let tensor = load_latent_tensor(tmp.path(), 0, dim, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[1, dim, t]);
        assert_eq!(tensor.dtype(), DType::BF16);
    }

    // Verifies dim=0 returns an error instead of panicking on modulo-by-zero.
    #[test]
    fn test_load_latent_tensor_dim_zero_errors() {
        let raw = vec![0u8; 8];
        let mut buf = Vec::new();
        {
            let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("tmpabc/data/0", opts).unwrap();
            zip.write_all(&raw).unwrap();
            zip.finish().unwrap();
        }
        let tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.as_file().write_all(&buf).unwrap();

        let err = load_latent_tensor(tmp.path(), 0, 0, &Device::Cpu).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("vae dim is 0"), "unexpected error: {msg}");
    }

    // Verifies a voices.json entry with a path-traversal component is rejected.
    #[test]
    fn test_load_voices_rejects_path_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let voices_dir = dir.path().join("voices");
        std::fs::create_dir(&voices_dir).unwrap();

        // Place a .pt file *outside* the voices directory.
        let escape_path = dir.path().join("escape.pt");
        std::fs::write(&escape_path, b"dummy").unwrap();

        let index = r#"{"evil": {"file": "../escape.pt", "description": "x", "language": "en"}}"#;
        std::fs::write(voices_dir.join("voices.json"), index).unwrap();

        let config: KugelAudioConfig = serde_json::from_str(MINIMAL_CONFIG_JSON).unwrap();
        let err = load_voices(dir.path(), &config, &Device::Cpu).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("escapes voices directory"),
            "unexpected error: {msg}"
        );
    }
}
