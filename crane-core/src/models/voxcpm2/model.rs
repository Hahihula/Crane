//! VoxCPM2 orchestration: loads all five sub-networks and implements the
//! **zero-shot-only** generation loop (no reference-audio conditioning /
//! voice cloning, no streaming — see the crate-level scope note in `mod.rs`).
//! Port of `voxcpm2.py`'s `VoxCPM2Model._generate`'s zero-shot branch +
//! `_inference`.
//!
//! Two exactness details worth flagging for future readers:
//! - `lm_hidden` is **not** FSQ-quantized coming out of the text-only
//!   prefill (the reference only applies FSQ where `audio_mask == 1`, and
//!   zero-shot's prefix is all-text) but **is** FSQ'd after every subsequent
//!   generated step — this asymmetry is real, not an oversight, and is
//!   preserved here.
//! - The reference always runs `feat_encoder` on the (all-zero, for
//!   zero-shot) prefix audio features even though the result is immediately
//!   multiplied by an all-zero `audio_mask`. This crate skips that call and
//!   substitutes an exact zero tensor directly — `0.0 * finite == 0.0`
//!   exactly in IEEE 754, so this is bit-identical to the reference, not an
//!   approximation, while skipping real compute.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, linear_no_bias, Activation, Linear, VarBuilder};
use serde::Deserialize;

use super::audio_vae::AudioVaeDecoder;
use super::cfm::UnifiedCfm;
use super::config::load_config;
use super::fsq::ScalarQuantizationLayer;
use super::local_dit::VoxCpmLocDit;
use super::local_encoder::VoxCpmLocEnc;
use super::minicpm4::MiniCpm4Model;
use super::tokenizer::VoxCpm2Tokenizer;

/// Hardcoded in the Python (`VoxCPM2Model.__init__`'s
/// `self.audio_start_token = 101`), not config-derived.
const AUDIO_START_TOKEN: u32 = 101;

#[derive(Debug, Clone)]
pub struct VoxCpm2GenerationConfig {
    /// Won't stop before this many patches even if the stop head fires.
    pub min_len: usize,
    /// Hard cap on generated patches.
    pub max_len: usize,
    /// Euler steps per patch in the flow-matching sampler.
    pub inference_timesteps: usize,
    /// Classifier-free guidance strength (`dit_config.cfm_config.inference_cfg_rate`
    /// in the checkpoint; exposed here since callers commonly override it).
    pub cfg_value: f64,
}

impl Default for VoxCpm2GenerationConfig {
    fn default() -> Self {
        Self { min_len: 2, max_len: 2000, inference_timesteps: 10, cfg_value: 2.0 }
    }
}

/// Shape/runtime fields read out of `config.json`'s generic
/// `audio_vae_config` block (kept as `serde_json::Value` in
/// [`VoxCpm2Config`] since the AudioVAE decoder only needs a handful of its
/// fields — see `config.rs`'s module docs).
#[derive(Debug, Clone, Deserialize)]
struct AudioVaeShapeConfig {
    latent_dim: usize,
    decoder_dim: usize,
    decoder_rates: Vec<usize>,
    sr_bin_boundaries: Vec<i64>,
    out_sample_rate: i64,
}

pub struct VoxCpm2Model {
    tokenizer: VoxCpm2Tokenizer,
    base_lm: MiniCpm4Model,
    residual_lm: MiniCpm4Model,
    feat_encoder: VoxCpmLocEnc,
    feat_decoder: UnifiedCfm,
    fsq_layer: ScalarQuantizationLayer,
    enc_to_lm_proj: Linear,
    lm_to_dit_proj: Linear,
    res_to_dit_proj: Linear,
    fusion_concat_proj: Linear,
    stop_proj: Linear,
    stop_head: Linear,
    audio_vae: AudioVaeDecoder,
    patch_size: usize,
    feat_dim: usize,
    lm_use_mup: bool,
    lm_scale_emb: f64,
    device: Device,
    dtype: DType,
    pub sample_rate: u32,
}

impl VoxCpm2Model {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let cfg = load_config(&format!("{model_path}/config.json"))
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("load VoxCPM2 config.json")?;

        let tokenizer_path = format!("{model_path}/tokenizer.json");
        let tokenizer = VoxCpm2Tokenizer::from_file(&tokenizer_path)?;

        let weights_path = format!("{model_path}/model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], *dtype, device) }
            .context("mmap model.safetensors")?;

        let table_len = cfg.max_length;
        let base_lm = MiniCpm4Model::new(&cfg.lm_config, table_len, vb.pp("base_lm"))?;
        let residual_cfg = cfg.lm_config.derive_residual_lm(cfg.residual_lm_num_layers, cfg.residual_lm_no_rope);
        let residual_lm = MiniCpm4Model::new(&residual_cfg, table_len, vb.pp("residual_lm"))?;

        let encoder_cfg = cfg.lm_config.derive(&cfg.encoder_config);
        let feat_encoder = VoxCpmLocEnc::new(&encoder_cfg, cfg.feat_dim, table_len, vb.pp("feat_encoder"))?;

        let dit_cfg = cfg.lm_config.derive(&cfg.dit_config.shape);
        let estimator = VoxCpmLocDit::new(&dit_cfg, cfg.feat_dim, table_len, vb.pp("feat_decoder").pp("estimator"))?;
        let feat_decoder = UnifiedCfm::new(estimator, cfg.dit_config.mean_mode);

        let hidden = cfg.lm_config.hidden_size;
        let fsq_layer = ScalarQuantizationLayer::new(
            hidden,
            hidden,
            cfg.scalar_quantization_latent_dim,
            cfg.scalar_quantization_scale,
            vb.pp("fsq_layer"),
        )?;
        let enc_to_lm_proj = linear(cfg.encoder_config.hidden_dim, hidden, vb.pp("enc_to_lm_proj"))?;
        let lm_to_dit_proj = linear(hidden, cfg.dit_config.shape.hidden_dim, vb.pp("lm_to_dit_proj"))?;
        let res_to_dit_proj = linear(hidden, cfg.dit_config.shape.hidden_dim, vb.pp("res_to_dit_proj"))?;
        let fusion_concat_proj = linear(hidden * 2, hidden, vb.pp("fusion_concat_proj"))?;
        let stop_proj = linear(hidden, hidden, vb.pp("stop_proj"))?;
        let stop_head = linear_no_bias(hidden, 2, vb.pp("stop_head"))?;

        let avc: AudioVaeShapeConfig =
            serde_json::from_value(cfg.audio_vae_config.clone()).context("parse audio_vae_config")?;
        let vae_weights_path = format!("{model_path}/audiovae.safetensors");
        let vae_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[vae_weights_path], DType::F32, device) }
            .context("mmap audiovae.safetensors (run the .pth -> safetensors conversion first)")?;
        let audio_vae = AudioVaeDecoder::new(
            avc.latent_dim,
            avc.decoder_dim,
            &avc.decoder_rates,
            avc.sr_bin_boundaries,
            avc.out_sample_rate,
            vae_vb.pp("decoder"),
        )?;

        Ok(Self {
            tokenizer,
            base_lm,
            residual_lm,
            feat_encoder,
            feat_decoder,
            fsq_layer,
            enc_to_lm_proj,
            lm_to_dit_proj,
            res_to_dit_proj,
            fusion_concat_proj,
            stop_proj,
            stop_head,
            audio_vae,
            patch_size: cfg.patch_size,
            feat_dim: cfg.feat_dim,
            lm_use_mup: cfg.lm_config.use_mup,
            lm_scale_emb: cfg.lm_config.scale_emb,
            device: device.clone(),
            dtype: *dtype,
            sample_rate: avc.out_sample_rate as u32,
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_lm.clear_kv_cache();
        self.residual_lm.clear_kv_cache();
    }

    /// Zero-shot text-to-speech: `text` in, a `[1, 1, T]` f32 waveform in
    /// `[-1, 1]` at [`Self::sample_rate`] out.
    pub fn generate_speech(&mut self, text: &str, cfg: &VoxCpm2GenerationConfig) -> Result<Tensor> {
        self.clear_kv_cache();

        let mut ids = self.tokenizer.encode(text)?;
        ids.push(AUDIO_START_TOKEN);
        let text_len = ids.len();
        let ids_tensor = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?; // [1, T]

        let embed_tokens = self.base_lm.embed_tokens.as_ref().context("base_lm has no embed_tokens")?;
        let text_embed = embed_tokens.forward(&ids_tensor)?.to_dtype(self.dtype)?; // [1, T, H]
        let text_embed =
            if self.lm_use_mup { (text_embed * self.lm_scale_emb)? } else { text_embed };

        // combined_embed == text_embed exactly: zero-shot's prefix has no
        // audio positions (audio_mask is all-zero throughout).
        let enc_outputs = self.base_lm.forward(&text_embed, true)?; // [1, T, H]
        // No FSQ here — see module docs.
        let mut lm_hidden = enc_outputs.narrow(1, text_len - 1, 1)?.squeeze(1)?; // [1, H]

        // feat_mask is all-zero, so the `feat_mask * feat_embed` term is an
        // exact zero tensor — skip running feat_encoder on the (all-zero)
        // prefix entirely rather than computing then discarding it.
        let zeros_feat_embed = enc_outputs.zeros_like()?;
        let residual_in = self.fusion_concat_proj.forward(&Tensor::cat(&[&enc_outputs, &zeros_feat_embed], 2)?)?;
        let residual_outputs = self.residual_lm.forward(&residual_in, true)?;
        let mut residual_hidden = residual_outputs.narrow(1, text_len - 1, 1)?.squeeze(1)?; // [1, H]

        let mut prefix_feat_cond = Tensor::zeros((1, self.patch_size, self.feat_dim), self.dtype, &self.device)?;
        let mut generated: Vec<Tensor> = Vec::new();

        for step in 0..cfg.max_len {
            let dit_h1 = self.lm_to_dit_proj.forward(&lm_hidden)?;
            let dit_h2 = self.res_to_dit_proj.forward(&residual_hidden)?;
            let dit_hidden = Tensor::cat(&[&dit_h1, &dit_h2], 1)?; // [1, 2*dit_hidden]

            let cond = prefix_feat_cond.transpose(1, 2)?.contiguous()?; // [1, D, P]
            let pred_feat = self.feat_decoder.forward(
                &dit_hidden,
                cfg.inference_timesteps,
                self.patch_size,
                &cond,
                cfg.cfg_value,
                1.0,
                1.0,
                true,
            )?; // [1, D, P]
            let pred_feat = pred_feat.transpose(1, 2)?.contiguous()?; // [1, P, D]

            let curr_embed = self.feat_encoder.forward(&pred_feat.unsqueeze(1)?)?; // [1, 1, H_enc]
            let curr_embed = self.enc_to_lm_proj.forward(&curr_embed)?; // [1, 1, H]

            generated.push(pred_feat.clone());
            prefix_feat_cond = pred_feat;

            let stop_hidden = Activation::Silu.forward(&self.stop_proj.forward(&lm_hidden)?)?;
            let stop_logits = self.stop_head.forward(&stop_hidden)?; // [1, 2]
            let stop_flag = stop_logits.argmax(1)?.reshape(())?.to_scalar::<u32>()?;
            if step > cfg.min_len && stop_flag == 1 {
                break;
            }

            let step_embed = curr_embed.squeeze(1)?; // [1, H]
            let position = text_len + step;
            let next_lm_hidden = self.base_lm.forward_step(&step_embed, position)?; // [1, H]
            lm_hidden = self.fsq_layer.forward(&next_lm_hidden)?; // FSQ applied for every generated step.
            let residual_input = self.fusion_concat_proj.forward(&Tensor::cat(&[&lm_hidden, &step_embed], 1)?)?;
            residual_hidden = self.residual_lm.forward_step(&residual_input, position)?;
        }

        anyhow::ensure!(!generated.is_empty(), "generated zero audio patches");

        // "b t p d -> b d (t p)": stack the per-step patches into a time
        // axis, then permute+flatten so channels lead and (step, within-patch)
        // collapse into one axis with `step` outer / `within-patch` inner.
        let stacked_terms: Vec<Tensor> = generated.iter().map(|p| p.unsqueeze(1)).collect::<candle_core::Result<_>>()?;
        let stacked = Tensor::cat(&stacked_terms, 1)?; // [1, n_steps, P, D]
        let (_b, n_steps, p, d) = stacked.dims4()?;
        let latent = stacked.permute((0, 3, 1, 2))?.contiguous()?.reshape((1, d, n_steps * p))?;

        self.audio_vae.decode(&latent.to_dtype(DType::F32)?).map_err(|e| anyhow::anyhow!("{e}"))
    }
}
