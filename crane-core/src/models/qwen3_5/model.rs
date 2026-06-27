//! Top-level Qwen 3.5 text-only transformer.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, rms_norm, Embedding, RmsNorm, VarBuilder};

use super::config::{Config, TextConfig};
use super::modeling::{resolve_tied, DecoderLayer, MRotaryEmbedding};

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