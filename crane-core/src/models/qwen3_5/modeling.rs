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
use candle_nn::{linear_no_bias, VarBuilder};

// ── Qwen 3.5 RMSNorm (unit-offset) ───────────────────────────────────────

/// RMSNorm as used by Qwen 3.5: `x / rms(x) * (1 + weight)`.
///
/// Unlike the standard (Llama/Qwen3) RMSNorm — which scales by `weight` — Qwen
/// 3.5 adds a unit offset (`1 + weight`, Gemma-style). HF source:
/// `output = self._norm(x.float()) * (1.0 + self.weight.float())`. The stored
/// weights have mean ~0.24, so omitting the `+1` shrinks every normalized
/// activation ~5x and compounds across layers. Computed in f32 then cast back.
///
/// This is NOT used by the GDN gated norm ([`crate::gdn::RmsNormGated`]), which
/// scales by plain `weight` in HF.
#[derive(Clone)]
pub struct Qwen35RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen35RmsNorm {
    pub fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Module for Qwen35RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        // `1 + weight` (unit offset), in f32.
        let scale = self.weight.to_dtype(DType::F32)?.affine(1.0, 1.0)?;
        x_normed.broadcast_mul(&scale)?.to_dtype(dtype)
    }
}

use super::config::{LayerType, TextConfig};
use super::kv_cache::KvCache;
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
        let base = cfg.rope_theta() as f32;
        let max_pos = cfg.max_position_embeddings;

        // cos/sin tables have shape `[S, rot_dim/2]` — exactly the slice of the
        // head that receives rotary embeddings. `apply_mrope` rotates only the
        // first `rot_dim` components of each head (HF's partial-rotary scheme:
        // `q_rot = q[..., :rot_dim]`), so the tables must NOT be padded to
        // `head_dim/2`. Padding them to the full head and rotating the whole
        // head (the previous approach) pairs dim `i` with dim `i+head_dim/2`,
        // whereas HF pairs `i` with `i+rot_dim/2` inside the rotary slice — a
        // different rotation entirely.
        let half_rot = rot_dim / 2;
        let inv: Vec<f32> = (0..half_rot)
            .map(|i| 1.0 / base.powf(i as f32 * 2.0 / rot_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_pos).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?; // [max_pos, half_rot]

        let cos_table = freqs.cos()?.contiguous()?;
        let sin_table = freqs.sin()?.contiguous()?;

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
/// Only the first `rot_dim` components of each head are rotated; the remaining
/// `D - rot_dim` components are passed through unchanged. This mirrors HF's
/// partial-rotary scheme (`q_rot = q[..., :rot_dim]`, `q_pass = q[..., rot_dim:]`).
/// `cos`/`sin` tables have shape `[S, rot_dim/2]`.
///
/// `candle_nn::rotary_emb::rope` is the rotate-half (non-interleaved / GPT-NeoX)
/// variant — it pairs component `i` with `i + rot_dim/2` *within the slice we
/// hand it*, which matches HF's `rotate_half` over the rotary slice. We must
/// slice first; rotating the full head would pair `i` with `i + head_dim/2`.
pub fn apply_mrope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rot_dim: usize,
) -> Result<Tensor> {
    let (_b, _h, _seq_len, head_dim) = x.dims4()?;
    if rot_dim == head_dim {
        return candle_nn::rotary_emb::rope(&x.contiguous()?, cos, sin);
    }
    let x_rot = x.narrow(D::Minus1, 0, rot_dim)?.contiguous()?;
    let x_pass = x.narrow(D::Minus1, rot_dim, head_dim - rot_dim)?;
    let x_rot = candle_nn::rotary_emb::rope(&x_rot, cos, sin)?;
    Tensor::cat(&[&x_rot, &x_pass], D::Minus1)?.contiguous()
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
    q_norm: Qwen35RmsNorm,
    k_norm: Qwen35RmsNorm,
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

        let q_norm = Qwen35RmsNorm::load(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Qwen35RmsNorm::load(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

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
        kv_cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let debug_attn = std::env::var("CRANE_DEBUG_ATTN").is_ok();

        if debug_attn {
            dump_to_file("layer_input", x);
        }

        let q_out = self.q_proj.forward(x)?;
        let k_proj_out = self.k_proj.forward(x)?;
        let v_proj_out = self.v_proj.forward(x)?;
        if debug_attn {
            dump_to_file("q_proj", &q_out);
            dump_to_file("k_proj", &k_proj_out);
            dump_to_file("v_proj", &v_proj_out);
        }
        let k = k_proj_out;
        let v = v_proj_out;

        let (q, gate) = if self.has_output_gate {
            // HF splits `[query | gate]` PER HEAD, not on the flat axis:
            //   q_proj(x).view(B, S, num_heads, head_dim*2).chunk(2, dim=-1)
            // so for each head the first `head_dim` is the query and the next
            // `head_dim` is the gate. Splitting the flat 4096 in half instead
            // interleaves heads' query/gate and scrambles q_norm. Re-flatten
            // both back to `[B, S, num_heads*head_dim]` in head order.
            let flat = self.num_heads * self.head_dim;
            let qh = q_out.reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?;
            let q = qh
                .narrow(D::Minus1, 0, self.head_dim)?
                .contiguous()?
                .reshape((b_sz, seq_len, flat))?;
            let gate = qh
                .narrow(D::Minus1, self.head_dim, self.head_dim)?
                .contiguous()?
                .reshape((b_sz, seq_len, flat))?;
            (q, Some(gate))
        } else {
            (q_out, None)
        };

        if debug_attn {
            let w = self.q_proj.weight().to_dtype(DType::F32)?;
            dump_to_file("q_proj_weight", &w);
            let w = self.k_proj.weight().to_dtype(DType::F32)?;
            dump_to_file("k_proj_weight", &w);
            let w = self.v_proj.weight().to_dtype(DType::F32)?;
            dump_to_file("v_proj_weight", &w);
        }

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
        if debug_attn { dump_to_file("q_norm", &q); dump_to_file("k_norm", &k); }

        let q = apply_mrope(&q, cos, sin, rot_dim)?;
        let k = apply_mrope(&k, cos, sin, rot_dim)?;
        if debug_attn { dump_to_file("q_after_mrope", &q); dump_to_file("k_after_mrope", &k); }

        // Append this step's K/V to the cache (post-RoPE, pre-GQA-expand) and
        // continue with the full cached K/V. During incremental decode this is
        // what lets the attention see the whole context from a single token.
        let (k, v) = match kv_cache {
            Some(cache) => cache.append(&k, &v)?,
            None => (k, v),
        };

        let n_rep = self.num_heads / self.num_kv_heads;
        let scale = 1.0 / (self.head_dim as f64).sqrt();

        let k_rep = if n_rep > 1 {
            let (b, kv_heads, s, d) = k.dims4()?;
            k.unsqueeze(2)?
                .expand((b, kv_heads, n_rep, s, d))?
                .contiguous()?
                .reshape((b, self.num_heads, s, d))?
        } else {
            k
        };
        let v_rep = if n_rep > 1 {
            let (b, kv_heads, s, d) = v.dims4()?;
            v.unsqueeze(2)?
                .expand((b, kv_heads, n_rep, s, d))?
                .contiguous()?
                .reshape((b, self.num_heads, s, d))?
        } else {
            v
        };
        if debug_attn { dump_to_file("k_rep", &k_rep); dump_to_file("v_rep", &v_rep); }

        let k_t = k_rep.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn_logits = (q.matmul(&k_t)? * scale)?;
        if debug_attn { dump_to_file("attn_logits", &attn_logits); }
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights_with_mask(&attn_logits, mask)?,
            None => attn_logits,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        if debug_attn { dump_to_file("attn_weights_post_softmax", &attn_weights); }
        let y = attn_weights.matmul(&v_rep)?;
        if debug_attn { dump_to_file("attn_output_pre_gate", &y); }

        let y = y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;

        let y = if let Some(g) = gate {
            let gate = candle_nn::ops::sigmoid(&g.to_dtype(y.dtype())?)?;
            y.broadcast_mul(&gate)?
        } else {
            y
        };
        if debug_attn { dump_to_file("attn_output_post_gate", &y); }

        self.o_proj.forward(&y)
    }
}

fn attn_weights_with_mask(
    attn_logits: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    // HF applies the mask (shape `[B, 1, S_q, S_k]` for additive causal mask)
    // via `attn_weights + mask` and softmax. We do the same.
    Ok(attn_logits.broadcast_add(mask)?)
}

fn dump_to_file(name: &str, t: &candle_core::Tensor) {
    use std::io::Write;
    let dir = "/tmp/opencode/crane_attn_dump";
    let _ = std::fs::create_dir_all(dir);
    let v = match t.to_dtype(candle_core::DType::F32).and_then(|x| x.flatten_all()) {
        Ok(v) => v,
        Err(_) => return,
    };
    let data: Vec<f32> = match v.to_vec1() {
        Ok(d) => d,
        Err(_) => return,
    };
    let path = format!("{dir}/{name}.bin");
    let mut f = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(_) => return,
    };
    // Header: name, shape (4 dims), dtype tag (0=f32)
    let dims = t.dims();
    let name_bytes = name.as_bytes();
    let _ = f.write_all(&(name_bytes.len() as u32).to_le_bytes());
    let _ = f.write_all(name_bytes);
    let _ = f.write_all(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        let _ = f.write_all(&(*d as u64).to_le_bytes());
    }
    for x in &data {
        let _ = f.write_all(&x.to_le_bytes());
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
    input_layernorm: Qwen35RmsNorm,
    post_attention_layernorm: Qwen35RmsNorm,
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
        let input_layernorm = Qwen35RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        if std::env::var("CRANE_DEBUG_NORM").is_ok() {
            if let Ok(w) = input_layernorm.weight().to_dtype(DType::F32) {
                if let Ok(v) = w.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                    let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mean = v.iter().sum::<f32>() / v.len() as f32;
                    eprintln!("[crane-norm] input_layernorm.weight: shape={:?}, min={mn}, max={mx}, mean={mean}", w.dims());
                }
            }
        }
        let post_attention_layernorm = Qwen35RmsNorm::load(
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

    /// Forward pass. For `LinearAttention` blocks pass `Some(gdn_cache)` and
    /// `None` for `attn_cache`; for `FullAttention` blocks pass `Some(attn_cache)`
    /// and `None` for `gdn_cache`.
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rot_dim: usize,
        attention_mask: Option<&Tensor>,
        gdn_cache: Option<&mut GdnLayerCache>,
        attn_cache: Option<&mut KvCache>,
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
        if std::env::var("CRANE_DEBUG_ATTN").is_ok() {
            let w = self.input_layernorm.weight().to_dtype(DType::F32)?;
            dump_to_file("input_layernorm_weight", &w);
            dump_to_file("input_layernorm_output", &normed);
            // Sanity: confirm they're different
            let w_min = w.to_vec1::<f32>()?.iter().cloned().fold(f32::INFINITY, f32::min);
            let w_max = w.to_vec1::<f32>()?.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let w_mean = w.to_vec1::<f32>()?.iter().sum::<f32>() / w.to_vec1::<f32>()?.len() as f32;
            eprintln!("[crane-norm] LAYER 3 input_layernorm.weight: min={w_min:.4}, max={w_max:.4}, mean={w_mean:.4}");
            let xn = normed.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let xn_min = xn.iter().cloned().fold(f32::INFINITY, f32::min);
            let xn_max = xn.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[crane-norm] LAYER 3 normed: min={xn_min:.4}, max={xn_max:.4}");
            let xi = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let xi_min = xi.iter().cloned().fold(f32::INFINITY, f32::min);
            let xi_max = xi.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[crane-norm] LAYER 3 input (pre-norm): min={xi_min:.4}, max={xi_max:.4}");
        }

        let attn_out = match &self.layer_impl {
            LayerImpl::FullAttention(attn) => {
                debug_assert!(gdn_cache.is_none(), "full-attention layer should not receive a GDN cache");
                attn.forward(&normed, cos, sin, rot_dim, attention_mask, attn_cache)?
            }
            LayerImpl::LinearAttention(gdn) => {
                debug_assert!(attn_cache.is_none(), "linear-attention layer should not receive a KV cache");
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
            let v = attn_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let (mn, mx) = v.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| (lo.min(x), hi.max(x)));
            eprintln!("[crane-layer] post_attn_or_gdn (flat): min={mn}, max={mx}");
            // Per-position for the last batch.
            let dims = attn_out.dims();
            if dims.len() >= 3 {
                let flat = attn_out.to_dtype(DType::F32)?.reshape((dims[0], dims[1], dims[2]))?;
                let data = flat.squeeze(0)?.to_vec2::<f32>()?;
                for p in 0..dims[1].min(3) {
                    let row = &data[p];
                    let mn = row.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    eprintln!("[crane-layer] post_attn_or_gdn pos {p}: min={mn}, max={mx}");
                }
                if dims[1] > 3 {
                    let p = dims[1] - 1;
                    let row = &data[p];
                    let mn = row.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    eprintln!("[crane-layer] post_attn_or_gdn pos {p} (last): min={mn}, max={mx}");
                }
            }
        }
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