//! The diffusion head: predicts noise/velocity for the next acoustic latent,
//! conditioned on the decoder's hidden state and a diffusion timestep.
//!
//! Port of `kugelaudio_open.models.diffusion_head` (a renamed copy of
//! `microsoft/VibeVoice`'s `modular_vibevoice_diffusion_head.py`). Not a full
//! DiT — `head_layers` plain adaLN-modulated FFN blocks over a flat
//! `[N, latent_size]` batch of noisy latents, no self-attention, no
//! patchification.
//!
//! Weight-name note: `nn.Sequential(SiLU, Linear)` puts the activation at
//! Sequential index `0` (no params) and the `Linear` at `1` — hence
//! `adaLN_modulation.1.*` (not `.0.*`), and `t_embedder.mlp.{0,2}.*`.

#![allow(clippy::needless_pass_by_value)] // VarBuilder by-value is the candle idiom
#![allow(clippy::cast_precision_loss)] // exp() / pos encoding math
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // f64->usize for ffn_dim
#![allow(clippy::module_name_repetitions)] // HeadLayer/FinalLayer/etc. are common names
#![allow(clippy::missing_errors_doc)] // Result-returning helpers: errors are candle tensor errors
#![allow(clippy::missing_panics_doc)] // expect() in tests only
#![allow(clippy::must_use_candidate)] // getters are conventionally used at call sites
#![allow(clippy::doc_markdown)] // adaLN/ SiLU are math names, not generic Markdown text
#![allow(clippy::similar_names)] // proj field names match PyTorch's source
#![allow(clippy::pedantic)] // trailing catch-all after targeted allows above

use candle_core::{D, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::models::with_tracing::{Linear, RmsNorm, linear_no_bias};

use super::config::DiffusionHeadConfig;

const FREQUENCY_EMBEDDING_SIZE: usize = 256;

/// `x * (1 + scale) + shift`.
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let ones_plus_scale = (scale + 1.0)?;
    &x.broadcast_mul(&ones_plus_scale)? + shift
}

/// Sinusoidal timestep embedding for even `dim` (always true here).
fn timestep_embedding(t: &Tensor, dim: usize, device: &candle_core::Device) -> Result<Tensor> {
    let half = dim / 2;
    let max_period = 10_000f64;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(max_period.ln()) * (i as f64) / (half as f64)).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(freqs, half, device)?;
    let t = t.to_dtype(candle_core::DType::F32)?;
    let n = t.dim(0)?;
    let args = t
        .reshape((n, 1))?
        .broadcast_mul(&freqs.reshape((1, half))?)?;
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)
}

struct TimestepEmbedder {
    mlp0: Linear,
    mlp2: Linear,
}

impl TimestepEmbedder {
    fn load(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let vb_mlp = vb.pp("mlp");
        Ok(Self {
            mlp0: linear_no_bias(FREQUENCY_EMBEDDING_SIZE, hidden_size, vb_mlp.pp(0))?,
            mlp2: linear_no_bias(hidden_size, hidden_size, vb_mlp.pp(2))?,
        })
    }

    fn forward(&self, t: &Tensor, device: &candle_core::Device) -> Result<Tensor> {
        let t_freq = timestep_embedding(t, FREQUENCY_EMBEDDING_SIZE, device)?
            .to_dtype(self.mlp0.weight().dtype())?;
        let h = self.mlp0.forward(&t_freq)?.silu()?;
        self.mlp2.forward(&h)
    }
}

/// SwiGLU FFN, all projections bias-free.
struct FeedForwardNetwork {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl FeedForwardNetwork {
    fn load(embed_dim: usize, ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(embed_dim, ffn_dim, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(embed_dim, ffn_dim, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(ffn_dim, embed_dim, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj
            .forward(&crate::ops::fused_ops::swiglu::swiglu(&gate, &up)?)
    }
}

/// AdaLN-modulated FFN block: `x + gate * ffn(modulate(norm(x), shift, scale))`,
/// with `(shift, scale, gate) = adaLN_modulation(c).chunk(3)`.
struct HeadLayer {
    ffn: FeedForwardNetwork,
    norm: RmsNorm,
    ada_ln: Linear,
    embed_dim: usize,
}

impl HeadLayer {
    fn load(
        embed_dim: usize,
        ffn_dim: usize,
        cond_dim: usize,
        norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            ffn: FeedForwardNetwork::load(embed_dim, ffn_dim, vb.pp("ffn"))?,
            norm: RmsNorm::new(embed_dim, norm_eps, vb.pp("norm"))?,
            ada_ln: linear_no_bias(cond_dim, 3 * embed_dim, vb.pp("adaLN_modulation").pp(1))?,
            embed_dim,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let c = c.silu()?;
        let mod_params = self.ada_ln.forward(&c)?;
        let shift = mod_params.narrow(D::Minus1, 0, self.embed_dim)?;
        let scale = mod_params.narrow(D::Minus1, self.embed_dim, self.embed_dim)?;
        let gate = mod_params.narrow(D::Minus1, 2 * self.embed_dim, self.embed_dim)?;
        let normed = self.norm.forward(x)?;
        let modulated = modulate(&normed, &shift, &scale)?;
        let ffn_out = self.ffn.forward(&modulated)?;
        x + (gate * ffn_out)?
    }
}

/// Output projection: `linear(modulate(norm_final(x), shift, scale))`, with
/// `(shift, scale) = adaLN_modulation(c).chunk(2)`. `norm_final` has
/// `elementwise_affine=False` (plain RMS, no learned scale).
struct FinalLayer {
    norm_eps: f64,
    hidden_size: usize,
    linear: Linear,
    ada_ln: Linear,
}

impl FinalLayer {
    fn load(
        hidden_size: usize,
        output_size: usize,
        cond_size: usize,
        norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            norm_eps,
            hidden_size,
            linear: linear_no_bias(hidden_size, output_size, vb.pp("linear"))?,
            ada_ln: linear_no_bias(cond_size, 2 * hidden_size, vb.pp("adaLN_modulation").pp(1))?,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let c = c.silu()?;
        let mod_params = self.ada_ln.forward(&c)?;
        let shift = mod_params.narrow(D::Minus1, 0, self.hidden_size)?;
        let scale = mod_params.narrow(D::Minus1, self.hidden_size, self.hidden_size)?;
        // RMS without learned scale.
        let normed = {
            let x32 = x.to_dtype(candle_core::DType::F32)?;
            let ms = x32.sqr()?.mean_keepdim(D::Minus1)?;
            let rstd = (ms + self.norm_eps)?.recip()?.sqrt()?;
            x32.broadcast_mul(&rstd)?.to_dtype(x.dtype())?
        };
        let modulated = modulate(&normed, &shift, &scale)?;
        self.linear.forward(&modulated)
    }
}

/// Projects noisy latents + condition + timestep through `head_layers`
/// [`HeadLayer`]s, then [`FinalLayer`] back to `latent_size`.
pub struct DiffusionHead {
    noisy_images_proj: Linear,
    cond_proj: Linear,
    t_embedder: TimestepEmbedder,
    layers: Vec<HeadLayer>,
    final_layer: FinalLayer,
}

impl DiffusionHead {
    pub fn load(cfg: &DiffusionHeadConfig, vb: VarBuilder) -> Result<Self> {
        let cond_dim = cfg.hidden_size;
        let ffn_dim = (cfg.hidden_size as f64 * cfg.head_ffn_ratio).round() as usize;
        let noisy_images_proj =
            linear_no_bias(cfg.latent_size, cfg.hidden_size, vb.pp("noisy_images_proj"))?;
        let cond_proj = linear_no_bias(cfg.hidden_size, cond_dim, vb.pp("cond_proj"))?;
        let t_embedder = TimestepEmbedder::load(cond_dim, vb.pp("t_embedder"))?;
        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.head_layers);
        for i in 0..cfg.head_layers {
            layers.push(HeadLayer::load(
                cfg.hidden_size,
                ffn_dim,
                cond_dim,
                cfg.rms_norm_eps,
                vb_layers.pp(i),
            )?);
        }
        let final_layer = FinalLayer::load(
            cfg.hidden_size,
            cfg.latent_size,
            cond_dim,
            cfg.rms_norm_eps,
            vb.pp("final_layer"),
        )?;
        Ok(Self {
            noisy_images_proj,
            cond_proj,
            t_embedder,
            layers,
            final_layer,
        })
    }

    /// `noisy_latents`: `[N, latent_size]`. `timesteps`: `[N]`. `condition`:
    /// `[N, hidden_size]` — decoder hidden state at each latent's
    /// generating position. Returns `[N, latent_size]` — predicted
    /// noise/velocity (interpretation per `prediction_type`; see
    /// `dpm_solver.rs`).
    pub fn forward(
        &self,
        noisy_latents: &Tensor,
        timesteps: &Tensor,
        condition: &Tensor,
    ) -> Result<Tensor> {
        let x = self.noisy_images_proj.forward(noisy_latents)?;
        let t = self.t_embedder.forward(timesteps, x.device())?;
        let cond = self.cond_proj.forward(condition)?;
        let c = (cond + t)?;
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &c)?;
        }
        self.final_layer.forward(&x, &c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    fn small_cfg() -> DiffusionHeadConfig {
        DiffusionHeadConfig {
            hidden_size: 16,
            latent_size: 8,
            head_layers: 2,
            head_ffn_ratio: 3.0,
            rms_norm_eps: 1e-5,
            prediction_type: "v_prediction".to_string(),
            ddpm_beta_schedule: "cosine".to_string(),
            ddpm_num_steps: 1000,
            ddpm_num_inference_steps: 20,
            ddpm_algorithm_type: "sde-dpmsolver++".to_string(),
        }
    }

    fn make_vb(cfg: &DiffusionHeadConfig, device: &Device) -> VarBuilder<'static> {
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let fill = |shape: &[usize]| -> Tensor {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| 0.01 * (i as f32 + 1.0)).collect();
            Tensor::from_vec(data, shape, device).unwrap()
        };
        let ones = |n: usize| Tensor::ones(n, DType::F32, device).unwrap();
        let h = cfg.hidden_size;
        let l = cfg.latent_size;
        let ffn_dim = (h as f64 * cfg.head_ffn_ratio).round() as usize;

        t.insert("noisy_images_proj.weight".into(), fill(&[h, l]));
        t.insert("cond_proj.weight".into(), fill(&[h, h]));
        t.insert("t_embedder.mlp.0.weight".into(), fill(&[h, 256]));
        t.insert("t_embedder.mlp.2.weight".into(), fill(&[h, h]));

        for i in 0..cfg.head_layers {
            let p = format!("layers.{i}");
            t.insert(format!("{p}.ffn.gate_proj.weight"), fill(&[ffn_dim, h]));
            t.insert(format!("{p}.ffn.up_proj.weight"), fill(&[ffn_dim, h]));
            t.insert(format!("{p}.ffn.down_proj.weight"), fill(&[h, ffn_dim]));
            t.insert(format!("{p}.norm.weight"), ones(h));
            t.insert(format!("{p}.adaLN_modulation.1.weight"), fill(&[3 * h, h]));
        }
        t.insert("final_layer.linear.weight".into(), fill(&[l, h]));
        t.insert(
            "final_layer.adaLN_modulation.1.weight".into(),
            fill(&[2 * h, h]),
        );

        VarBuilder::from_tensors(t, DType::F32, device)
    }

    #[test]
    fn forward_shape_and_finite() {
        let device = Device::Cpu;
        let cfg = small_cfg();
        let vb = make_vb(&cfg, &device);
        let head = DiffusionHead::load(&cfg, vb).expect("load");

        let n = 5usize;
        let noisy = Tensor::rand(-1f32, 1f32, (n, cfg.latent_size), &device).unwrap();
        let t = Tensor::from_vec(
            (0..n).map(|i| i as f32 * 50.0).collect::<Vec<_>>(),
            n,
            &device,
        )
        .unwrap();
        let cond = Tensor::rand(-1f32, 1f32, (n, cfg.hidden_size), &device).unwrap();

        let out = head.forward(&noisy, &t, &cond).expect("forward");
        assert_eq!(out.dims(), &[n, cfg.latent_size]);
        let max_abs: f32 = out.abs().unwrap().max_all().unwrap().to_scalar().unwrap();
        assert!(max_abs.is_finite());
    }

    /// Same forward pass on Metal (skipped where Metal isn't available). The
    /// adaLN/SiLU/RMSNorm ops have no CUDA-only equivalent, but this is the
    /// only place that exercises them on Metal.
    #[test]
    fn forward_shape_and_finite_on_metal() {
        if !candle_core::utils::metal_is_available() {
            return;
        }
        let device = Device::new_metal(0).expect("metal device");
        let cfg = small_cfg();
        let vb = make_vb(&cfg, &device);
        let head = DiffusionHead::load(&cfg, vb).expect("load");

        let n = 5usize;
        let noisy = Tensor::rand(-1f32, 1f32, (n, cfg.latent_size), &device).unwrap();
        let t = Tensor::from_vec(
            (0..n).map(|i| i as f32 * 50.0).collect::<Vec<_>>(),
            n,
            &device,
        )
        .unwrap();
        let cond = Tensor::rand(-1f32, 1f32, (n, cfg.hidden_size), &device).unwrap();

        let out = head.forward(&noisy, &t, &cond).expect("forward");
        assert_eq!(out.dims(), &[n, cfg.latent_size]);
        let max_abs: f32 = out.abs().unwrap().max_all().unwrap().to_scalar().unwrap();
        assert!(max_abs.is_finite());
    }
}
