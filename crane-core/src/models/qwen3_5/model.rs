//! Top-level Qwen 3.5 text-only transformer + the high-level `Model`
//! wrapper used by the engine (config + weights + tokenizer).

use std::io::Write;

use anyhow::{Context, Error as E, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{embedding, Embedding, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

use super::config::{load_config, Config, TextConfig};
use super::modeling::{resolve_tied, DecoderLayer, MRotaryEmbedding, Qwen35RmsNorm};
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
    norm: Qwen35RmsNorm,
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
        // HF saves Qwen 3.5 weights with the prefix
        //   model.language_model.layers.{i}.{linear_attn,self_attn}.*
        // (the `model.` is the inner multimodal `Qwen3_5Model`, of which
        // `language_model` is the text component). Older checkpoints and
        // standalone text exports may drop the leading `model.`.
        let vb_lm = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            vb.pp("model").pp("language_model")
        } else if vb.contains_tensor("language_model.embed_tokens.weight") {
            vb.pp("language_model")
        } else if vb.contains_tensor("model.embed_tokens.weight") {
            vb.pp("model")
        } else {
            // Last-resort: assume a flat layout, no prefix.
            vb.clone()
        };

        let embed_tokens = embedding(text_cfg.vocab_size, text_cfg.hidden_size, vb_lm.pp("embed_tokens"))?;

        let layer_types = text_cfg.layer_types();
        let mut layers = Vec::with_capacity(text_cfg.num_hidden_layers);
        for (idx, &layer_type) in layer_types.iter().enumerate() {
            layers.push(DecoderLayer::load(&text_cfg, layer_type, vb_lm.pp("layers").pp(idx))?);
        }

        let norm = Qwen35RmsNorm::load(text_cfg.hidden_size, text_cfg.rms_norm_eps, vb_lm.pp("norm"))?;
        if std::env::var("CRANE_DEBUG_NORM").is_ok() {
            if let Ok(w) = norm.weight().to_dtype(DType::F32) {
                if let Ok(v) = w.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                    let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mean = v.iter().sum::<f32>() / v.len() as f32;
                    eprintln!("[crane-norm] FINAL norm.weight: shape={:?}, min={mn}, max={mx}, mean={mean}", w.dims());
                }
            }
        }

        let embed_weight = embed_tokens.embeddings().clone();
        let lm_head_weight = vb_lm.get(text_cfg.vocab_size, "lm_head").ok();
        // Store `lm_head` already transposed (`[H, V]`) so the per-step matmul
        // doesn't have to. `resolve_tied` returns either the dedicated lm_head
        // weight or — when `tie_word_embeddings: true` — the embed table.
        let lm_head_raw = resolve_tied(cfg.tie_word_embeddings, embed_weight, lm_head_weight);
        let lm_head = lm_head_raw.t()?.contiguous()?.clone();

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

        // If no mask was supplied, build a causal mask so prefill doesn't
        // leak future tokens into past positions. (Decode is `seq_len==1`,
        // where every position trivially attends only to itself — the mask is
        // a no-op there.)
        let mask = match attention_mask {
            Some(m) => Some(m.clone()),
            None if !is_decode_step => {
                let total = start_pos + seq_len;
                Some(build_causal_mask(seq_len, total, xs.device(), xs.dtype())?)
            }
            None => None,
        };
        let mask_ref = mask.as_ref();

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
                mask_ref,
                cache_slot,
                is_decode_step,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        // Matmul with lm_head on the LAST position only. The engine's
        // sampling step expects `[B, V]` (1D logits for the next token).
        let (b, s, _) = xs.dims3()?;
        let last = xs.narrow(1, s - 1, 1)?.reshape((b, ()))?;
        let logits = last.matmul(&self.lm_head)?;
        Ok(logits)
    }
}

/// Build a causal mask of shape `[1, 1, S_q, S_k]` where row `i` has `-inf`
/// for columns `j > start_pos + i` (future tokens). Returned as f32 cast to
/// the model's dtype.
fn build_causal_mask(
    seq_q: usize,
    total_k: usize,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> candle_core::Result<candle_core::Tensor> {
    let mut data: Vec<f32> = Vec::with_capacity(total_k * total_k);
    for q in 0..total_k {
        for k in 0..total_k {
            data.push(if k > q { f32::NEG_INFINITY } else { 0.0 });
        }
    }
    let full = candle_core::Tensor::from_vec(data, (total_k, total_k), device)?
        .to_dtype(dtype)?;
    Ok(full.narrow(0, 0, seq_q)?.unsqueeze(0)?.unsqueeze(0)?)
}

fn print_layer_stats(name: &str, t: &candle_core::Tensor) -> anyhow::Result<()> {
    let v = t
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let finite: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let nan = v.iter().filter(|x| x.is_nan()).count();
    let inf = v.iter().filter(|x| x.is_infinite()).count();
    let (mn, mx, mean) = if finite.is_empty() {
        (f32::NAN, f32::NAN, f32::NAN)
    } else {
        let mn = finite.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = finite.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let avg = finite.iter().sum::<f32>() / finite.len() as f32;
        (mn, mx, avg)
    };
    println!(
        "[crane] {name}: shape={:?}, NaN={}, Inf={}, min={:.4}, max={:.4}, mean={:.4}",
        t.dims(),
        nan,
        inf,
        mn,
        mx,
        mean,
    );
    Ok(())
}

fn print_layer_stats_per_pos(name: &str, t: &candle_core::Tensor) -> anyhow::Result<()> {
    // t is [B, S, H]; print per-position min/max for the last batch.
    let dims = t.dims();
    if dims.len() < 3 {
        return Ok(());
    }
    let s = dims[1];
    let h = dims[2];
    let flat = t.to_dtype(candle_core::DType::F32)?.reshape((dims[0], s, h))?;
    let data = flat.squeeze(0)?.to_vec2::<f32>()?;
    println!("[crane] {name} per-position (S={s}, H={h}):");
    for p in 0..s {
        let row = &data[p];
        let mn = row.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = row.iter().sum::<f32>() / row.len() as f32;
        println!("  pos {}: min={}, max={}, mean={}", p, mn, mx, mean);
    }
    Ok(())
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

    /// Tokenize a prompt string into input IDs (mirrors `qwen3::Model`).
    pub fn prepare_inputs(&self, inputs: &str) -> Result<Vec<u32>> {
        let input_ids = self
            .tokenizer
            .tokenizer
            .encode(inputs, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        Ok(input_ids)
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
        // Run a tiny decode-only forward (single token) so warmup succeeds
        // regardless of how the generate loop is implemented.
        if let Err(e) = self.generate(
            &[45],
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

        // The full-attention layers have no KV cache yet, so incremental decode
        // (feeding only the last token) starves them of history. Until a KV
        // cache lands, recompute the full context each step (O(n^2) but
        // correct): reset the GDN caches and re-run the whole prefix. Default
        // ON for correctness; set CRANE_INCREMENTAL_DECODE=1 to opt into the
        // (currently incorrect) fast path for benchmarking.
        let full_recompute = std::env::var("CRANE_INCREMENTAL_DECODE").is_err();

        let start_gen = std::time::Instant::now();
        for index in 0..config.max_new_tokens {
            let context_size = if index > 0 && !full_recompute { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            if full_recompute {
                self.clear_kv_cache();
            }
            let logits = self.forward_step(ctxt, start_pos)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

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

// ─────────────────────────────────────────────────────────────────────
//  Debug helpers
// ─────────────────────────────────────────────────────────────────────

impl Model {
    /// Run a single forward on `input_ids` and print the top-K next-token
    /// IDs + logits + decoded tokens. Used for correctness debugging
    /// against an HF Transformers reference. Not part of the public SDK.
    #[doc(hidden)]
    pub fn debug_topk(&mut self, input_ids: &[u32], k: usize) -> anyhow::Result<()> {
        use anyhow::Context;
        let logits = self
            .forward_step(input_ids, 0)
            .context("forward_step failed")?;
        let logits = logits.squeeze(0)?.to_dtype(candle_core::DType::F32)?;
        let v = logits.to_vec1::<f32>().context("to_vec1")?;
        let nan_count = v.iter().filter(|x| x.is_nan()).count();
        let inf_count = v.iter().filter(|x| x.is_infinite()).count();
        let finite: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
        let (min, max, mean) = if finite.is_empty() {
            (f32::NAN, f32::NAN, f32::NAN)
        } else {
            let mn = finite.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = finite.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let avg = finite.iter().sum::<f32>() / finite.len() as f32;
            (mn, mx, avg)
        };
        println!(
            "[crane] logits stats: shape=[{}], NaN={}, Inf={}, finite_min={:.4}, finite_max={:.4}, finite_mean={:.4}",
            v.len(),
            nan_count,
            inf_count,
            min,
            max,
            mean,
        );

        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
        let topk: Vec<(u32, f32)> = idx
            .iter()
            .take(k)
            .map(|&i| (i as u32, v[i]))
            .collect();
        println!("[crane] step-0 argmax: id={} logit={:.4}", topk[0].0, topk[0].1);
        println!("[crane] step-0 top-{} (id, logit, decoded):", k);
        for (id, logit) in &topk {
            let decoded = self
                .tokenizer
                .tokenizer
                .id_to_token(*id)
                .unwrap_or_else(|| "<unk>".into());
            println!("  {} ({:.4}) {:?}", id, logit, decoded);
        }
        Ok(())
    }

    /// Dump the hidden-state stats after each layer + the post-norm state.
    /// Used to localize which layer first diverges from the HF reference.
    #[doc(hidden)]
    pub fn debug_layer_stats(&mut self, input_ids: &[u32]) -> anyhow::Result<()> {
        use anyhow::Context;
        let seq_len = input_ids.len();
        let input = candle_core::Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let mut xs = self.inner.embed_tokens.forward(&input)?;
        print_layer_stats("after_embed", &xs)?;
        print_layer_stats_per_pos("after_embed", &xs)?;

        let (cos, sin) = self.inner.rotary.cos_sin(0, seq_len)?;
        let rot_dim = self.inner.rotary.rot_dim();
        let mask = build_causal_mask(seq_len, seq_len, &self.device, self.dtype)?;
        let mask_ref: Option<&candle_core::Tensor> = if seq_len > 1 {
            Some(&mask)
        } else {
            None
        };

        for i in 0..self.inner.layers.len() {
            let layer = &mut self.inner.layers[i];
            let cache_slot = self.inner.gdn_caches[i].as_mut();
            xs = layer.forward(
                &xs,
                &cos,
                &sin,
                rot_dim,
                mask_ref,
                cache_slot,
                seq_len == 1,
            )?;
            if i == 0 || i == 3 || i == 11 || i == 23 {
                print_layer_stats_per_pos(&format!("after_layer_{}", i), &xs)?;
            }
        }
        let xs = self.inner.norm.forward(&xs)?;
        print_layer_stats("after_norm", &xs)?;
        print_layer_stats_per_pos("after_norm", &xs)?;
        let _ = (cos, sin);
        Ok(())
    }
}