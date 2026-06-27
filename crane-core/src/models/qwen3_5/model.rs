//! Top-level Qwen 3.5 text-only transformer + the high-level `Model`
//! wrapper used by the engine (config + weights + tokenizer).

use std::io::Write;

use anyhow::{Context, Error as E, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{embedding, rms_norm, Embedding, RmsNorm, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

use super::config::{load_config, Config, TextConfig};
use super::modeling::{resolve_tied, DecoderLayer, MRotaryEmbedding};
use crate::generation::based::ModelForCausalLM;
use crate::generation::GenerationConfig;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::utils;

/// Text-only Qwen 3.5 transformer.
///
/// `gdn_caches` is indexed by layer; `None` for full-attention blocks, `Some`
/// for linear-attention blocks. The engine is responsible for cloning/saving
/// these caches across context switches (continuous batching).
pub struct Qwen3_5TextModel {
    cfg: TextConfig,
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Tensor,
    rotary: MRotaryEmbedding,
    gdn_caches: Vec<Option<crate::gdn::GdnLayerCache>>,
    device: Device,
    dtype: DType,
}

impl Qwen3_5TextModel {
    /// Load a text-only Qwen 3.5 model from a HF checkpoint directory.
    ///
    /// The HF layout has a top-level `language_model.*` prefix when the model
    /// was saved with `Qwen3_5ForConditionalGeneration`. We probe for that
    /// prefix and fall back to a flat layout.
    pub fn new(cfg: &Config, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let text_cfg = cfg.text().clone();
        let vb_lm = if vb.contains_tensor("language_model.model.embed_tokens.weight") {
            vb.pp("language_model").pp("model")
        } else {
            vb.pp("model")
        };

        let embed_tokens = embedding(text_cfg.vocab_size, text_cfg.hidden_size, vb_lm.pp("embed_tokens"))?;

        let layer_types = text_cfg.layer_types();
        let mut layers = Vec::with_capacity(text_cfg.num_hidden_layers);
        for (idx, &layer_type) in layer_types.iter().enumerate() {
            layers.push(DecoderLayer::load(&text_cfg, layer_type, vb_lm.pp("layers").pp(idx))?);
        }

        let norm = rms_norm(text_cfg.hidden_size, text_cfg.rms_norm_eps, vb_lm.pp("norm"))?;

        let embed_weight = embed_tokens.embeddings().clone();
        let lm_head_weight = vb_lm.get(text_cfg.vocab_size, "lm_head").ok();
        let lm_head = resolve_tied(cfg.tie_word_embeddings, embed_weight, lm_head_weight);

        let rotary = MRotaryEmbedding::new(&text_cfg, device)?;

        // Pre-allocate GDN caches for linear-attention layers.
        let mut gdn_caches = Vec::with_capacity(layers.len());
        for layer in &layers {
            if layer.is_linear() {
                gdn_caches.push(Some(crate::gdn::GdnLayerCache::new(
                    &text_cfg, dtype, device,
                )?));
            } else {
                gdn_caches.push(None);
            }
        }

        Ok(Self {
            cfg: text_cfg,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
            gdn_caches,
            device: device.clone(),
            dtype,
        })
    }

    pub fn config(&self) -> &TextConfig {
        &self.cfg
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn lm_head(&self) -> &Tensor {
        &self.lm_head
    }

    /// Reset all per-layer GDN caches. Called between unrelated requests
    /// sharing the same pre-allocated layer set.
    pub fn reset_gdn_caches(&mut self) -> Result<()> {
        for slot in self.gdn_caches.iter_mut().flatten() {
            slot.reset()?;
        }
        Ok(())
    }

    /// Forward pass over `input_ids` of shape `[B, S]`. `start_pos` is the
    /// absolute position of the first token (used for rotary slicing).
    /// `attention_mask` is broadcastable to `[B, 1, S, S_total]` (or `None`).
    ///
    /// Returns logits of shape `[B, S, vocab_size]`.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        start_pos: usize,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let is_decode_step = seq_len == 1;

        let mut xs = self.embed_tokens.forward(input_ids)?;

        let (cos, sin) = self.rotary.cos_sin(start_pos, seq_len)?;
        let rot_dim = self.rotary.rot_dim();

        for i in 0..self.layers.len() {
            let layer = &mut self.layers[i];
            let cache_slot = self.gdn_caches[i].as_mut();
            xs = layer.forward(
                &xs,
                &cos,
                &sin,
                rot_dim,
                attention_mask,
                cache_slot,
                is_decode_step,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        // Matmul with lm_head: [B, S, H] @ [V, H]^T = [B, S, V]
        let logits = xs.matmul(&self.lm_head.t()?)?;
        Ok(logits)
    }
}

// ─────────────────────────────────────────────────────────────────────
//  High-level Model wrapper (used by crane-serve's engine)
// ─────────────────────────────────────────────────────────────────────

/// Format of model weights on disk. Qwen 3.5 currently only ships
/// safetensors; GGUF support is left for a future PR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Auto,
    Safetensors,
}

/// Public-facing `Model` for Qwen 3.5 text-only inference.
///
/// Mirrors the structure of `crane_core::models::qwen3::Model`:
/// holds a tokenizer, device, dtype, and the inner [`Qwen3_5TextModel`].
pub struct Model {
    pub tokenizer: TokenOutputStream,
    pub device: Device,
    pub dtype: DType,
    inner: Qwen3_5TextModel,
}

impl Model {
    /// Load a Qwen 3.5 model from a HF checkpoint directory.
    ///
    /// Phase 1A: only the safetensors path is wired. GGUF support lands later.
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        Self::new_with_format(model_path, device, dtype, ModelFormat::Auto)
    }

    pub fn new_with_format(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        format: ModelFormat,
    ) -> Result<Self> {
        let format = match format {
            ModelFormat::Auto => ModelFormat::Safetensors,
            other => other,
        };
        match format {
            ModelFormat::Safetensors => Self::from_pretrained(model_path, device, dtype),
            // GGUF support is a follow-up — the dense checkpoints ship as
            // safetensors, and the GDN path's lazy-eviction story is enough
            // complexity for one PR.
            ModelFormat::Auto => unreachable!("Auto resolves to Safetensors above"),
        }
    }

    fn from_pretrained(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let filenames = utils::get_safetensors_files(model_path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }?;

        let config_path = std::path::Path::new(model_path).join("config.json");
        let cfg = load_config(config_path.to_str().context("non-UTF8 model path")?)?;

        let inner = Qwen3_5TextModel::new(&cfg, vb, device, *dtype)?;

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype: *dtype,
            inner,
        })
    }

    /// Run a single forward step, returning raw logits `[1, S, vocab]`.
    pub fn forward_step(
        &mut self,
        input_ids: &[u32],
        start_pos: usize,
    ) -> Result<Tensor> {
        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        Ok(self.inner.forward(&input, start_pos, None)?)
    }

    /// Reset all per-layer GDN caches (between unrelated requests).
    pub fn clear_kv_cache(&mut self) {
        self.inner
            .reset_gdn_caches()
            .expect("GDN cache reset failed");
    }

    pub fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    /// Warm up the model with a small forward pass.
    pub fn warmup(&mut self) {
        if let Err(e) = self.generate(
            &[45, 546, 456],
            &GenerationConfig::with_max_tokens(5),
            None,
        ) {
            eprintln!("warmup failed (non-fatal): {e}");
        }
        self.clear_kv_cache();
    }
}

impl ModelForCausalLM for Model {
    fn device(&self) -> &Device {
        &self.device
    }

    fn generate(
        &mut self,
        input_ids: &[u32],
        config: &GenerationConfig,
        mut streamer: Option<&mut dyn crate::generation::streamer::TokenStreamer>,
    ) -> Result<Vec<u32>> {
        self.tokenizer.clear();
        self.clear_kv_cache();

        let mut logits_processor =
            LogitsProcessor::new(1024, config.temperature, config.top_p);

        let mut tokens = input_ids.to_vec();
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        // Qwen 3.5 / Qwen 3 EOS tokens: 151645 () and 151643 ().
        let eos_token: Option<u32> = config
            .eos_token_id
            .or_else(|| self.tokenizer.get_token(""))
            .or_else(|| self.tokenizer.get_token(""));
        let mut streamer_finalized = false;

        let start_gen = std::time::Instant::now();
        for index in 0..config.max_new_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            let logits = self.forward_step(ctxt, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            let logits = if config.repetition_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(config.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    config.repetition_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;

            if eos_token == Some(next_token) {
                if let Some(ref mut s) = streamer {
                    s.finalize()?;
                }
                streamer_finalized = true;
                break;
            }

            if let Some(ref mut s) = streamer {
                s.append(next_token)?;
            }
        }

        let dt = start_gen.elapsed();
        if config.report_speed {
            println!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)\n",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
        }
        let _ = streamer_finalized;
        Ok(tokens)
    }
}