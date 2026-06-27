//! Qwen 3.5 transformer layer: hybrid full-attention + linear-attention stack.
//!
//! Layout (per layer):
//!   residual = input_layernorm(x)
//!   attn_or_gdn_out = full_or_linear(residual, …)   // dispatched by layer_types
//!   x = x + attn_or_gdn_out
//!   residual2 = post_attention_layernorm(x)
//!   x = x + mlp(residual2)
//!
//! The full-attention path uses MRoPE-interleaved rotary embeddings and gated
//! output (`attn_output_gate: true` in the config). The linear-attention path
//! uses [`crate::gdn::GatedDeltaNet`].
//!
//! Per-layer GDN state is held by [`super::Qwen3_5TextModel`] (not by the
//! layer), so that continuous-batching can save/restore state per request.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, RmsNorm, VarBuilder};

use super::config::{LayerType, TextConfig};
use crate::gdn::{GatedDeltaNet, GdnDims, GdnInputProjectionKind, GdnLayerCache};

// ── MRoPE rotary embedding ─────────────────────────────────────────────

/// Multimodal Rotary Position Embedding (interleaved variant, used by Qwen 3.5).
///
/// For text-only inference the position_ids are identical across all three
/// sections, so this reduces to standard RoPE applied to the first
/// `rot_dim = head_dim * partial_rotary_factor` components of each head.
///
/// Precomputes `cos` and `sin` tables of shape `[max_pos, rot_dim/2]`. Use
/// [`apply_mrope`] to rotate a query/key tensor.
pub struct MRotaryEmbedding {
    cos_table: Tensor,
    sin_table: Tensor,
    rot_dim: usize,
}

impl MRotaryEmbedding {
    pub fn new(cfg: &TextConfig, device: &Device) -> Result<Self> {
        let rot_dim = cfg.rot_dim();
        let head_dim = cfg.head_dim;
        let base = cfg.rope_theta() as f32;
        let max_pos = cfg.max_position_embeddings;

        // cos/sin tables must have shape `[S, head_dim/2]` to satisfy
        // `candle_nn::rotary_emb::rope`. The first `rot_dim / 2` entries are
        // populated with the actual rotary frequencies; the remaining
        // entries are padded with `cos=1, sin=0` so the trailing dimensions
        // pass through the rotation unchanged.
        let half_head = head_dim / 2;
        let half_rot = rot_dim / 2;
        let inv: Vec<f32> = (0..half_rot)
            .map(|i| 1.0 / base.powf(i as f32 * 2.0 / rot_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_pos).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?; // [max_pos, half_rot]

        let cos_rot = freqs.cos()?.contiguous()?;
        let sin_rot = freqs.sin()?.contiguous()?;

        // Pad to [max_pos, head_dim/2] with cos=1, sin=0 in the unused slots.
        let pad = half_head - half_rot;
        let cos_table = if pad > 0 {
            let ones = Tensor::ones((max_pos, pad), DType::F32, device)?;
            Tensor::cat(&[&cos_rot, &ones], D::Minus1)?.contiguous()?
        } else {
            cos_rot
        };
        let sin_table = if pad > 0 {
            let zeros = Tensor::zeros((max_pos, pad), DType::F32, device)?;
            Tensor::cat(&[&sin_rot, &zeros], D::Minus1)?.contiguous()?
        } else {
            sin_rot
        };

        Ok(Self {
            cos_table,
            sin_table,
            rot_dim,
        })
    }

    /// Slice cos/sin for positions `[start, start+seq_len)`.
    pub fn cos_sin(&self, start: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos_table.narrow(0, start, seq_len)?;
        let sin = self.sin_table.narrow(0, start, seq_len)?;
        Ok((cos, sin))
    }

    pub fn rot_dim(&self) -> usize {
        self.rot_dim
    }
}

/// Apply rotary embeddings to a query/key tensor `[B, H, S, D]`.
///
/// Only the first `rot_dim` components of each head are rotated. The remaining
/// components are passed through unchanged. `cos`/`sin` tables have shape
/// `[S, D/2]`; the trailing `(D - rot_dim) / 2` entries are padding
/// (`cos=1, `sin=0`) so the rope op leaves those dimensions alone.
pub fn apply_mrope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rot_dim: usize,
) -> Result<Tensor> {
    let (_b, _h, seq_len, head_dim) = x.dims4()?;
    let device = x.device();
    let dtype = x.dtype();

    // cos/sin already have shape [S, head_dim/2] (padded by the embedding
    // module). Just pass through to `candle_nn::rotary_emb::rope`.
    let _ = (seq_len, dtype, rot_dim);
    candle_nn::rotary_emb::rope(x, cos, sin)
}

// ── Full-attention layer ────────────────────────────────────────────────

/// Standard softmax attention layer for Qwen 3.5's `full_attention` blocks.
///
/// Differences from the regular Qwen 3 attention:
/// - `q_proj` outputs `num_heads * head_dim * 2`; the second half is a sigmoid
///   gate applied to the attention output.
/// - Per-head QK-norm is always present (`q_norm`, `k_norm` of size `head_dim`).
/// - RoPE is MRoPE-interleaved applied only to the first `rot_dim` components.
pub struct FullAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    has_output_gate: bool,
}

impl FullAttention {
    pub fn load(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_out = if cfg.attn_output_gate {
            num_heads * head_dim * 2
        } else {
            num_heads * head_dim
        };
        let q_proj = linear_no_bias(cfg.hidden_size, q_out, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            has_output_gate: cfg.attn_output_gate,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rot_dim: usize,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q_out = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let (q, gate) = if self.has_output_gate {
            let half = self.num_heads * self.head_dim;
            let q = q_out.narrow(D::Minus1, 0, half)?;
            let gate = q_out.narrow(D::Minus1, half, half)?;
            (q, Some(gate))
        } else {
            (q_out, None)
        };

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = apply_mrope(&q, cos, sin, rot_dim)?;
        let k = apply_mrope(&k, cos, sin, rot_dim)?;

        let n_rep = self.num_heads / self.num_kv_heads;
        let scale = 1.0 / (self.head_dim as f64).sqrt();

        let k_rep = if n_rep > 1 {
            let (b, kv_heads, s, d) = k.dims4()?;
            k.unsqueeze(2)?
                .expand((b, kv_heads, n_rep, s, d))?
                .reshape((b, self.num_heads, s, d))?
        } else {
            k
        };
        let v_rep = if n_rep > 1 {
            let (b, kv_heads, s, d) = v.dims4()?;
            v.unsqueeze(2)?
                .expand((b, kv_heads, n_rep, s, d))?
                .reshape((b, self.num_heads, s, d))?
        } else {
            v
        };

        let attn_weights = (q.matmul(&k_rep.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let y = attn_weights.matmul(&v_rep)?;

        let y = y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;

        let y = if let Some(g) = gate {
            let gate = candle_nn::ops::sigmoid(&g.to_dtype(y.dtype())?)?;
            y.broadcast_mul(&gate)?
        } else {
            y
        };

        self.o_proj.forward(&y)
    }
}

// ── MLP ────────────────────────────────────────────────────────────────

/// Standard SwiGLU MLP: `down(silu(gate(x)) * up(x))`.
pub struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Mlp {
    pub fn load(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if std::env::var("CRANE_DEBUG_LAYER").is_ok() {
            let xn = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let (mn, mx) = xn
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            eprintln!("[crane-mlp] input: min={mn}, max={mx}");
            let gw = self.gate_proj.weight().to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let (gmn, gmx) = gw.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            eprintln!("[crane-mlp] gate_proj.weight: shape={:?}, min={gmn}, max={gmx}", self.gate_proj.weight().dims());
            let dw = self.down_proj.weight().to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let (dmn, dmx) = dw.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            eprintln!("[crane-mlp] down_proj.weight: shape={:?}, min={dmn}, max={dmx}", self.down_proj.weight().dims());
        }
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let h = gate.broadcast_mul(&up)?;
        let out = self.down_proj.forward(&h)?;
        if std::env::var("CRANE_DEBUG_LAYER").is_ok() {
            let hn = h.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let (hmn, hmx) = hn.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            eprintln!("[crane-mlp] post_silu_mul: min={hmn}, max={hmx}");
            let on = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let (omn, omx) = on.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            eprintln!("[crane-mlp] post_down_proj: min={omn}, max={omx}");
        }
        Ok(out)
    }
}

// ── DecoderLayer ────────────────────────────────────────────────────────

/// One transformer block. `LayerImpl` selects which attention path runs.
///
/// Per-layer GDN state is passed in via the [`DecoderLayer::forward`] signature
/// (a `&mut Option<GdnLayerCache>`); the model holds the canonical cache array.
pub struct DecoderLayer {
    layer_impl: LayerImpl,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    /// Pre-computed dims for the GDN path. `None` for full-attention blocks.
    gdn_dims: Option<GdnDims>,
}

enum LayerImpl {
    FullAttention(FullAttention),
    LinearAttention(GatedDeltaNet),
}

impl DecoderLayer {
    pub fn load(
        cfg: &TextConfig,
        layer_type: LayerType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = Mlp::load(cfg, vb.pp("mlp"))?;

        let (layer_impl, gdn_dims) = match layer_type {
            LayerType::FullAttention => (
                LayerImpl::FullAttention(FullAttention::load(cfg, vb.pp("self_attn"))?),
                None,
            ),
            LayerType::LinearAttention => {
                let dims = GdnDims::new(cfg);
                let gdn = GatedDeltaNet::load(vb, cfg, GdnInputProjectionKind::Split)?;
                (LayerImpl::LinearAttention(gdn), Some(dims))
            }
        };

        Ok(Self {
            layer_impl,
            input_layernorm,
            post_attention_layernorm,
            mlp,
            gdn_dims,
        })
    }

    pub fn is_linear(&self) -> bool {
        matches!(self.layer_impl, LayerImpl::LinearAttention(_))
    }

    /// Forward pass. `gdn_cache` is `Some(&mut cache)` only for
    /// `LinearAttention` blocks; pass `None` for full-attention blocks.
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rot_dim: usize,
        attention_mask: Option<&Tensor>,
        gdn_cache: Option<&mut GdnLayerCache>,
        is_decode_step: bool,
    ) -> Result<Tensor> {
        let residual = x;
        let normed = self.input_layernorm.forward(x)?;
        if std::env::var("CRANE_DEBUG_LAYER").is_ok() {
            let mn = normed.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let min = mn.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = mn.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[crane-layer] post_input_layernorm: min={min}, max={max}");
        }

        let attn_out = match &self.layer_impl {
            LayerImpl::FullAttention(attn) => {
                debug_assert!(gdn_cache.is_none(), "full-attention layer should not receive a GDN cache");
                attn.forward(&normed, cos, sin, rot_dim, attention_mask)?
            }
            LayerImpl::LinearAttention(gdn) => {
                let cache = gdn_cache.ok_or_else(|| {
                    candle_core::Error::Msg("GDN cache missing for linear-attention layer".into())
                })?;
                let dims = self.gdn_dims.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("GDN dims missing for linear-attention layer".into())
                })?;
                gdn.forward(&normed, dims, cache, is_decode_step)?
            }
        };
        if std::env::var("CRANE_DEBUG_LAYER").is_ok() {
            let mn = attn_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let min = mn.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = mn.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[crane-layer] post_attn_or_gdn: min={min}, max={max}");
        }

        let x = (residual + attn_out)?;
        if std::env::var("CRANE_DEBUG_LAYER").is_ok() {
            let mn = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let min = mn.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = mn.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[crane-layer] post_residual1: min={min}, max={max}");
        }

        let residual2 = &x;
        let normed2 = self.post_attention_layernorm.forward(&x)?;
        if std::env::var("CRANE_DEBUG_LAYER").is_ok() {
            let mn = normed2.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let min = mn.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = mn.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[crane-layer] post_post_attention_layernorm: min={min}, max={max}");
        }

        let mlp_out = self.mlp.forward(&normed2)?;
        if std::env::var("CRANE_DEBUG_LAYER").is_ok() {
            let mn = mlp_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let min = mn.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = mn.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[crane-layer] post_mlp: min={min}, max={max}");
        }

        residual2 + mlp_out
    }
}

/// Choose between a dedicated `lm_head` weight and the tied `embed_tokens`
/// weight, based on `tie_word_embeddings`.
pub fn resolve_tied(tie: bool, embed_weight: Tensor, lm_head_weight: Option<Tensor>) -> Tensor {
    match lm_head_weight {
        Some(w) => w,
        None if tie => embed_weight,
        None => embed_weight,
    }
}

// Touch unused imports so the file type-checks even before all paths are wired.
#[allow(dead_code)]
fn _touch_unused(_: DType, _: Device) {}