// SPDX-License-Identifier: MIT

//! Hyper-connections: the residual stream is `hc_count` parallel copies of
//! the hidden state (`[.., hc_count * hidden]`). Before each block a gated
//! mixer collapses them to one input; after it, the block output is scattered
//! back into every stream with a per-stream weight. Reference:
//! `Qwen4ExpTextGatedResidual` in transformers `modeling_qwen4_exp.py`.

use candle_core::{D, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::ops::hyper_connection as hc_ops;
use crate::ops::linear::{LinearLayer, linear_layer};
use crate::quantized::gguf_file::Gguf;

/// `RMSNorm` applied to each `group`-wide slice of the last dimension on its
/// own, each with its own gain (HF `Qwen4ExpTextRMSNorm(group_size=..)`).
///
/// Gains follow the unit-offset convention of [`crate::models::qwen3_5::Qwen35RmsNorm`]:
/// HF stores `w` and applies `1 + w`.
#[derive(Clone)]
pub struct GroupedRmsNorm {
    /// `[groups, group]`, the `+1` already folded in.
    alpha: Tensor,
    eps: f32,
}

impl GroupedRmsNorm {
    /// Load `weight` (`[groups * group]`, unit offset not yet applied).
    ///
    /// # Errors
    ///
    /// Returns an error if the weight is missing or has the wrong size.
    pub fn load(groups: usize, group: usize, eps: f64, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get(groups * group, "weight")?;
        Self::from_folded(&weight.affine(1.0, 1.0)?, groups, group, eps)
    }

    /// Construct from a gain that already includes the `+1` (GGUF layout).
    ///
    /// # Errors
    ///
    /// Returns an error if `alpha` does not hold `groups * group` values.
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_folded(alpha: &Tensor, groups: usize, group: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            alpha: alpha.reshape((groups, group))?,
            eps: eps as f32,
        })
    }
}

impl Module for GroupedRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        hc_ops::grouped_rms_norm(x, &self.alpha, self.eps)
    }
}

/// One gated hyper-connection mixer: [`Self::mix`] before a block, and
/// [`combine`] with its injection weights after it.
pub struct GatedResidual {
    norm: GroupedRmsNorm,
    down: LinearLayer,
    up: LinearLayer,
    /// Per-stream scatter weights; `None` on the final mixer, which only
    /// collapses the streams (and doubles as the output norm).
    inject: Option<LinearLayer>,
    hc_count: usize,
}

impl GatedResidual {
    /// Load from a HF checkpoint (`vb` scoped to e.g.
    /// `layers.N.attn_hyper_connection` or `hyper_connection_mixer`).
    ///
    /// # Errors
    ///
    /// Returns an error if a weight is missing or has the wrong shape.
    pub fn load(
        hidden: usize,
        hc_count: usize,
        low_rank: usize,
        eps: f64,
        with_inject: bool,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let wide = hc_count * hidden;
        Ok(Self {
            norm: GroupedRmsNorm::load(hc_count, hidden, eps, &vb.pp("hc_norm"))?,
            down: linear_layer(wide, low_rank, vb.pp("input_mix_weight_down"), None)?,
            up: linear_layer(low_rank, wide, vb.pp("input_mix_weight_up"), None)?,
            inject: with_inject
                .then(|| linear_layer(wide, hc_count, vb.pp("block_inject_weight"), None))
                .transpose()?,
            hc_count,
        })
    }

    /// Load from GGUF, where `prefix` names the mixer (`blk.N.hc_attn`,
    /// `blk.N.hc_ffn`, or `output_hc` for the final one): `{prefix}_norm`
    /// (`[hc, hidden]`, `+1` folded in), `{prefix}_down`, `{prefix}_up` and,
    /// with `with_inject`, `{prefix}_inject`.
    ///
    /// # Errors
    ///
    /// Returns an error if a tensor is missing or has the wrong shape.
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        hidden: usize,
        hc_count: usize,
        eps: f64,
        with_inject: bool,
    ) -> Result<Self> {
        let norm = gg.dequant_tensor(&format!("{prefix}_norm.weight"))?;
        Ok(Self {
            norm: GroupedRmsNorm::from_folded(&norm, hc_count, hidden, eps)?,
            down: gg.linear_compact(&format!("{prefix}_down.weight"))?,
            up: gg.linear_compact(&format!("{prefix}_up.weight"))?,
            inject: with_inject
                .then(|| gg.linear_compact(&format!("{prefix}_inject.weight")))
                .transpose()?,
            hc_count,
        })
    }

    /// Collapse the streams `[.., hc_count * hidden]` into one block input
    /// `[.., hidden]`. Also returns the [`Injection`] for [`combine`], or
    /// `None` for the final mixer.
    ///
    /// # Errors
    ///
    /// Returns an error if a projection or tensor op fails.
    pub fn mix(&self, streams: &Tensor) -> Result<(Tensor, Option<Injection>)> {
        #[allow(clippy::cast_precision_loss)]
        let inv_hc = 1.0 / self.hc_count as f64;
        let normed = self.norm.forward(streams)?;
        let low = hc_ops::scaled_silu(&self.down.forward(&normed)?, inv_hc)?;
        let gate = self.up.forward(&low)?;
        let mixed = hc_ops::gated_stream_mean(&gate, &normed, self.hc_count)?;
        let inject = self
            .inject
            .as_ref()
            .map(|inject| {
                Ok::<_, candle_core::Error>(Injection {
                    logits: inject.forward(&normed)?,
                    scale: inv_hc,
                })
            })
            .transpose()?;
        Ok((mixed, inject))
    }
}

/// Per-stream scatter weights of one block, `2 * sigmoid(logits * scale)`,
/// kept as raw logits so [`combine`] applies them in the same kernel.
pub struct Injection {
    /// `[.., hc_count]`.
    logits: Tensor,
    scale: f64,
}

impl Injection {
    /// The weights themselves, `[.., hc_count]`.
    ///
    /// # Errors
    ///
    /// Returns an error if a tensor op fails.
    pub fn weights(&self) -> Result<Tensor> {
        candle_nn::ops::sigmoid(&(&self.logits * self.scale)?)? * 2.0
    }
}

/// Add `block_out` (`[.., hidden]`) into every stream of `streams`
/// (`[.., hc_count * hidden]`), scaled per stream by `injection`.
///
/// # Errors
///
/// Returns an error if the shapes disagree.
pub fn combine(streams: &Tensor, block_out: &Tensor, injection: &Injection) -> Result<Tensor> {
    hc_ops::gated_combine(streams, block_out, &injection.logits, injection.scale)
}

/// The initial residual: `hc_count` copies of the token embedding.
///
/// # Errors
///
/// Returns an error if the tensor op fails.
pub fn expand_streams(embeddings: &Tensor, hc_count: usize) -> Result<Tensor> {
    let copies = vec![embeddings; hc_count];
    Tensor::cat(&copies, D::Minus1)
}
