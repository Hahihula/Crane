// SPDX-License-Identifier: MIT

//! QSA indexer: picks which cached tokens each query of an indexed-attention
//! layer may attend to. Keys are mean-pooled over blocks of `compress_ratio`
//! tokens, scored against the query with a `ReLU`'d multi-head dot product, and
//! the best `budget / compress_ratio` complete blocks are kept together with
//! the incomplete tail block. Reference: `Qwen4ExpTextQSAIndexer` in
//! transformers `modeling_qwen4_exp.py`.

use candle_core::{DType, Module, Result, Tensor, bail};
use candle_nn::VarBuilder;

use super::config::IndexerConfig;
use crate::models::qwen3_5::{Qwen35RmsNorm, apply_mrope};
use crate::ops::fused_ops::qsa_mask::qsa_block_mask;
use crate::ops::linear::LinearLayer;
use crate::quantized::gguf_file::Gguf;

/// Per-sequence indexer cache: the raw (un-normalized, un-rotated) key of
/// every token seen so far, `[tokens, head_dim]`.
#[derive(Default)]
pub struct IndexerCache {
    keys: Option<Tensor>,
}

impl IndexerCache {
    /// Cached tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.as_ref().map_or(0, |k| k.dim(0).unwrap_or(0))
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct QsaIndexer {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    q_norm: Qwen35RmsNorm,
    k_norm: Qwen35RmsNorm,
    cfg: IndexerConfig,
}

impl QsaIndexer {
    /// Load from a HF checkpoint (`vb` scoped to `layers.N.self_attn.indexer`),
    /// splitting the fused `index_qk_proj` into its query and key halves.
    ///
    /// # Errors
    ///
    /// Returns an error if a weight is missing or has the wrong shape.
    pub fn load(cfg: IndexerConfig, hidden: usize, eps: f64, vb: &VarBuilder) -> Result<Self> {
        let q_width = cfg.n_heads * cfg.head_dim;
        let qk = vb.get((q_width + cfg.head_dim, hidden), "index_qk_proj.weight")?;
        let linear = |w: Tensor| LinearLayer::Standard(candle_nn::Linear::new(w, None));
        Ok(Self {
            q_proj: linear(qk.narrow(0, 0, q_width)?),
            k_proj: linear(qk.narrow(0, q_width, cfg.head_dim)?),
            q_norm: Qwen35RmsNorm::load(cfg.head_dim, eps, &vb.pp("q_layernorm"))?,
            k_norm: Qwen35RmsNorm::load(cfg.head_dim, eps, &vb.pp("k_layernorm"))?,
            cfg,
        })
    }

    /// Load layer `layer_idx`'s indexer from GGUF (`blk.N.indexer.{q,k}_proj`
    /// and the `+1`-folded `{q,k}_norm`).
    ///
    /// # Errors
    ///
    /// Returns an error if a tensor is missing.
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        cfg: IndexerConfig,
        eps: f64,
        gg: &mut Gguf<R>,
        layer_idx: usize,
    ) -> Result<Self> {
        let name = |t: &str| format!("blk.{layer_idx}.indexer.{t}.weight");
        Ok(Self::new(
            cfg,
            gg.linear_compact(&name("q_proj"))?,
            gg.linear_compact(&name("k_proj"))?,
            Qwen35RmsNorm::from_folded(gg.dequant_tensor(&name("q_norm"))?, eps),
            Qwen35RmsNorm::from_folded(gg.dequant_tensor(&name("k_norm"))?, eps),
        ))
    }

    /// Assemble from separately loaded projections (GGUF keeps them apart).
    #[must_use]
    pub fn new(
        cfg: IndexerConfig,
        q_proj: LinearLayer,
        k_proj: LinearLayer,
        q_norm: Qwen35RmsNorm,
        k_norm: Qwen35RmsNorm,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            q_norm,
            k_norm,
            cfg,
        }
    }

    /// Select, for each of the `seq` new queries of one sequence, the cells of
    /// the attention cache it may attend to.
    ///
    /// `x` is the block input `[seq, hidden]` for cache positions
    /// `cache.len()..cache.len() + seq`. `cos`/`sin` are the rotary tables of
    /// every cell up to and including the new ones (`[cells, rot_dim / 2]`,
    /// as [`crate::models::qwen3_5::MRotaryEmbedding`] produces them).
    ///
    /// Returns an additive mask `[seq, cells]`: `0` where attention is
    /// allowed, `-inf` elsewhere. It is already causal.
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes disagree or a tensor op fails.
    pub fn select(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rot_dim: usize,
        cache: &mut IndexerCache,
    ) -> Result<Tensor> {
        let IndexerConfig {
            n_heads,
            head_dim,
            budget,
            compress_ratio: ratio,
        } = self.cfg;
        let (seq, _) = x.dims2()?;
        let start = cache.len();
        let cells = start + seq;
        if cos.dim(0)? < cells {
            bail!("rotary tables cover {} cells, need {cells}", cos.dim(0)?)
        }

        let new_keys = self.k_proj.forward(x)?;
        let keys = match cache.keys.take() {
            Some(prev) => Tensor::cat(&[&prev, &new_keys], 0)?,
            None => new_keys,
        };
        cache.keys = Some(keys.clone());

        // Every query sees a prefix of the cache, so the complete blocks it
        // sees are a prefix of the blocks complete over the whole cache. While
        // they fit the budget, every query keeps every visible cell and the
        // scores are not needed.
        let blocks = cells / ratio;
        let keep_blocks = budget / ratio;
        let scores = if blocks <= keep_blocks {
            None
        } else {
            // Queries: per-head norm, then rotary at their own positions.
            let q = self
                .q_norm
                .forward(&self.q_proj.forward(x)?.reshape((seq, n_heads, head_dim))?)?
                .transpose(0, 1)?
                .unsqueeze(0)?;
            let q = apply_mrope(
                &q,
                &cos.narrow(0, start, seq)?,
                &sin.narrow(0, start, seq)?,
                rot_dim,
            )?
            .squeeze(0)?; // [heads, seq, head_dim]

            let pooled = keys
                .narrow(0, 0, blocks * ratio)?
                .to_dtype(DType::F32)?
                .reshape((blocks, ratio, head_dim))?
                .mean(1)?
                .to_dtype(keys.dtype())?;
            let pooled = self.k_norm.forward(&pooled)?;
            // Each block is rotated at the position of its first token.
            let starts: Vec<u32> = (0..blocks)
                .map(|b| u32::try_from(b * ratio))
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let starts = Tensor::from_vec(starts, blocks, cos.device())?;
            let pooled = apply_mrope(
                &pooled.unsqueeze(0)?.unsqueeze(0)?,
                &cos.index_select(&starts, 0)?,
                &sin.index_select(&starts, 0)?,
                rot_dim,
            )?
            .squeeze(0)?
            .squeeze(0)?; // [blocks, head_dim]

            #[allow(clippy::cast_precision_loss)]
            let scale = 1.0 / (head_dim as f64).sqrt();
            let per_head = q
                .to_dtype(DType::F32)?
                .broadcast_matmul(&pooled.to_dtype(DType::F32)?.t()?)?; // [heads, seq, blocks]
            Some((per_head.relu()?.sum(0)? * scale)?.contiguous()?)
        };
        // Top-k over the scores stays on the device where a kernel exists.
        qsa_block_mask(scores.as_ref(), seq, start, ratio, keep_blocks, x.device())
    }
}
