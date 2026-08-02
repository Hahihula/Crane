//! Scalar-quantization bottleneck applied to `base_lm`'s hidden state at
//! audio positions — VoxCPM2's "tokenizer-free" discretization: a
//! differentiable-in-training, hard-round-in-inference bottleneck instead of
//! a VQ codebook lookup. Port of `layers/scalar_quantization_layer.py`
//! (inference path only — no straight-through estimator, this crate never
//! trains).

use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

pub struct ScalarQuantizationLayer {
    in_proj: Linear,
    out_proj: Linear,
    scale: f64,
}

impl ScalarQuantizationLayer {
    pub fn new(in_dim: usize, out_dim: usize, latent_dim: usize, scale: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            in_proj: linear(in_dim, latent_dim, vb.pp("in_proj"))?,
            out_proj: linear(latent_dim, out_dim, vb.pp("out_proj"))?,
            scale: scale as f64,
        })
    }

    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = self.in_proj.forward(hidden)?.tanh()?;
        let hidden = ((hidden * self.scale)?.round()? / self.scale)?;
        self.out_proj.forward(&hidden)
    }
}
