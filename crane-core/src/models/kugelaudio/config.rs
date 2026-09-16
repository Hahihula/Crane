//! HF-compatible config types for KugelAudio (`kugelaudio/kugelaudio-0-open`).
//!
//! KugelAudio is a post-trained fine-tune of Microsoft's VibeVoice
//! (`model_type: "kugelaudio"`, `KugelAudioForConditionalGeneration` in the
//! checkpoint's `config.json`) — a dense Qwen2 decoder backbone plus two
//! causal-conv VAE "tokenizers" (acoustic + semantic) and a small adaLN
//! diffusion head that predicts continuous speech latents autoregressively.
//! Ported against `kugelaudio-0-open`'s bundled source (a renamed copy of
//! `microsoft/VibeVoice`'s `modular_vibevoice_*.py` and a vendored
//! `diffusers` DPM-Solver).

#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::doc_markdown)] // KugelAudio / VibeVoice / Microsoft are external names

use serde::Deserialize;

/// Qwen2 decoder config (28 layers / 3584 hidden / GQA 28:4 for the
/// reference 7B checkpoint). Kept separate from `qwen25::qwen2::Config` —
/// the shared struct's non-`Option` `sliding_window: usize` can't represent
/// this checkpoint's `sliding_window: null` (Qwen2, not Qwen2.5).
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub hidden_act: candle_nn::Activation,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl DecoderConfig {
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Shared `acoustic_tokenizer_config`/`semantic_tokenizer_config` shape:
/// causal-conv VAE encoder/decoder stacks (see `conv_layers.rs`). Both
/// configs use this struct — the semantic side leaves decoder-only fields
/// at inert defaults since it only encodes.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerConfig {
    pub channels: usize,
    /// Latent dimension (64 acoustic, 128 semantic for the reference
    /// checkpoint). Python calls this `dimension`.
    pub vae_dim: usize,
    pub encoder_n_filters: usize,
    pub encoder_ratios: Vec<usize>,
    /// Dash-separated stage depths, parsed by [`Self::encoder_depths_vec`].
    pub encoder_depths: String,
    #[serde(default)]
    pub decoder_n_filters: Option<usize>,
    #[serde(default)]
    pub decoder_ratios: Option<Vec<usize>>,
    /// `None` → defaults to `reversed(encoder_depths)`.
    #[serde(default)]
    pub decoder_depths: Option<String>,
    pub causal: bool,
    pub conv_bias: bool,
    /// `"none" | "weight_norm" | "spectral_norm" | "layer_norm" | "time_group_norm"`.
    /// Reference checkpoint: `"none"` (no reparametrization). Other values
    /// are not implemented in `conv_layers.rs`.
    pub conv_norm: String,
    /// `"constant" | "reflect"`. Reference checkpoint: `"constant"`.
    pub pad_mode: String,
    /// `"LN" | "RMSNorm"`. Reference checkpoint: `"RMSNorm"`.
    pub layernorm: String,
    pub layernorm_eps: f64,
    pub layernorm_elementwise_affine: bool,
    /// `"conv" | "depthwise_conv"`. Reference: `"depthwise_conv"`.
    pub mixer_layer: String,
    /// Nonzero enables per-block learnable `LayerScale` (`gamma`/`ffn_gamma`).
    pub layer_scale_init_value: f64,
    pub disable_last_norm: bool,
    /// Fixed reconstruction std for acoustic VAE Gaussian sampling.
    /// Unused by the semantic tokenizer.
    #[serde(default)]
    pub fix_std: f64,
    /// Unused at inference — `encode()` returns the distribution mean.
    #[serde(default)]
    pub std_dist_type: String,
}

impl TokenizerConfig {
    #[must_use]
    pub fn encoder_depths_vec(&self) -> Vec<usize> {
        parse_dash_depths(&self.encoder_depths)
    }

    #[must_use]
    pub fn decoder_depths_vec(&self) -> Vec<usize> {
        if let Some(s) = &self.decoder_depths {
            return parse_dash_depths(s);
        }
        let mut d = self.encoder_depths_vec();
        d.reverse();
        d
    }
}

fn parse_dash_depths(s: &str) -> Vec<usize> {
    s.split('-').filter_map(|d| d.parse().ok()).collect()
}

/// AdaLN-modulated FFN stack that predicts the next acoustic latent's
/// noise/velocity conditioned on the decoder's hidden state. See
/// `diffusion_head.rs`.
#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionHeadConfig {
    pub hidden_size: usize,
    pub latent_size: usize,
    pub head_layers: usize,
    pub head_ffn_ratio: f64,
    pub rms_norm_eps: f64,
    /// Reference checkpoint: `"v_prediction"`.
    pub prediction_type: String,
    /// Reference checkpoint: `"cosine"` (the only schedule implemented in
    /// `dpm_solver.rs`).
    pub ddpm_beta_schedule: String,
    pub ddpm_num_steps: usize,
    pub ddpm_num_inference_steps: usize,
    /// Reference checkpoint: `"sde-dpmsolver++"` (the only algorithm
    /// implemented in `dpm_solver.rs`).
    pub ddpm_algorithm_type: String,
}

/// Top-level KugelAudio `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct KugelAudioConfig {
    pub decoder_config: DecoderConfig,
    pub acoustic_tokenizer_config: TokenizerConfig,
    pub semantic_tokenizer_config: TokenizerConfig,
    pub diffusion_head_config: DiffusionHeadConfig,
    pub acoustic_vae_dim: usize,
    pub semantic_vae_dim: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

/// Load `config.json` for a KugelAudio checkpoint.
///
/// # Errors
///
/// Returns a candle error if the file can't be read or parsed.
pub fn load_config(path: &str) -> candle_core::Result<KugelAudioConfig> {
    let data = std::fs::read(path)
        .map_err(|e| candle_core::Error::Msg(format!("read config {path}: {e}")))?;
    let cfg: KugelAudioConfig = serde_json::from_slice(&data)
        .map_err(|e| candle_core::Error::Msg(format!("parse config {path}: {e}")))?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact `config.json` published for `kugelaudio/kugelaudio-0-open` —
    /// regression test against silent schema drift, also documents the real
    /// shape.
    const REAL_CONFIG_JSON: &str = r#"{
      "acostic_vae_dim": 64,
      "acoustic_tokenizer_config": {
        "causal": true, "channels": 1, "conv_bias": true, "conv_norm": "none",
        "corpus_normalize": 0.0, "decoder_depths": null, "decoder_n_filters": 32,
        "decoder_ratios": [8, 5, 5, 4, 2, 2], "disable_last_norm": true,
        "encoder_depths": "3-3-3-3-3-3-8", "encoder_n_filters": 32,
        "encoder_ratios": [8, 5, 5, 4, 2, 2], "fix_std": 0.5,
        "layer_scale_init_value": 1e-06, "layernorm": "RMSNorm",
        "layernorm_elementwise_affine": true, "layernorm_eps": 1e-05,
        "mixer_layer": "depthwise_conv", "model_type": "kugelaudio_acoustic_tokenizer",
        "pad_mode": "constant", "std_dist_type": "gaussian", "torch_dtype": "bfloat16",
        "vae_dim": 64, "weight_init_value": 0.01
      },
      "acoustic_vae_dim": 64,
      "architectures": ["KugelAudioForConditionalGeneration"],
      "decoder_config": {
        "attention_dropout": 0.0, "hidden_act": "silu", "hidden_size": 3584,
        "initializer_range": 0.02, "intermediate_size": 18944,
        "max_position_embeddings": 32768, "max_window_layers": 28,
        "model_type": "qwen2", "num_attention_heads": 28, "num_hidden_layers": 28,
        "num_key_value_heads": 4, "rms_norm_eps": 1e-06, "rope_scaling": null,
        "rope_theta": 1000000.0, "sliding_window": null, "torch_dtype": "bfloat16",
        "use_cache": true, "use_mrope": false, "use_sliding_window": false,
        "vocab_size": 152064
      },
      "diffusion_head_config": {
        "ddpm_algorithm_type": "sde-dpmsolver++", "ddpm_batch_mul": 4,
        "ddpm_beta_schedule": "cosine", "ddpm_num_inference_steps": 20,
        "ddpm_num_steps": 1000, "diffusion_type": "ddpm", "head_ffn_ratio": 3.0,
        "head_layers": 4, "hidden_size": 3584, "latent_size": 64,
        "model_type": "kugelaudio_diffusion_head", "prediction_type": "v_prediction",
        "rms_norm_eps": 1e-05, "speech_vae_dim": 64, "torch_dtype": "bfloat16"
      },
      "model_type": "kugelaudio",
      "semantic_tokenizer_config": {
        "causal": true, "channels": 1, "conv_bias": true, "conv_norm": "none",
        "corpus_normalize": 0.0, "disable_last_norm": true,
        "encoder_depths": "3-3-3-3-3-3-8", "encoder_n_filters": 32,
        "encoder_ratios": [8, 5, 5, 4, 2, 2], "fix_std": 0,
        "layer_scale_init_value": 1e-06, "layernorm": "RMSNorm",
        "layernorm_elementwise_affine": true, "layernorm_eps": 1e-05,
        "mixer_layer": "depthwise_conv", "model_type": "kugelaudio_semantic_tokenizer",
        "pad_mode": "constant", "std_dist_type": "none", "torch_dtype": "bfloat16",
        "vae_dim": 128, "weight_init_value": 0.01
      },
      "semantic_vae_dim": 128,
      "tie_word_embeddings": false,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.52.0.dev0",
      "ddpm_inference_steps": 20
    }"#;

    #[test]
    fn parses_real_checkpoint_config() {
        let cfg: KugelAudioConfig = serde_json::from_str(REAL_CONFIG_JSON).expect("parse");
        assert_eq!(cfg.decoder_config.hidden_size, 3584);
        assert_eq!(cfg.decoder_config.num_hidden_layers, 28);
        assert_eq!(cfg.decoder_config.num_attention_heads, 28);
        assert_eq!(cfg.decoder_config.num_key_value_heads, 4);
        assert_eq!(cfg.decoder_config.head_dim(), 128);
        assert_eq!(cfg.acoustic_vae_dim, 64);
        assert_eq!(cfg.semantic_vae_dim, 128);
        assert_eq!(
            cfg.acoustic_tokenizer_config.encoder_depths_vec(),
            vec![3, 3, 3, 3, 3, 3, 8]
        );
        assert_eq!(
            cfg.acoustic_tokenizer_config.decoder_depths_vec(),
            vec![8, 3, 3, 3, 3, 3, 3]
        );
        assert_eq!(cfg.acoustic_tokenizer_config.vae_dim, 64);
        assert_eq!(cfg.semantic_tokenizer_config.vae_dim, 128);
        assert_eq!(cfg.diffusion_head_config.latent_size, 64);
        assert_eq!(cfg.diffusion_head_config.head_layers, 4);
        assert!((cfg.diffusion_head_config.head_ffn_ratio - 3.0).abs() < 1e-9);
        assert_eq!(cfg.diffusion_head_config.prediction_type, "v_prediction");
        assert_eq!(
            cfg.diffusion_head_config.ddpm_algorithm_type,
            "sde-dpmsolver++"
        );
    }
}
