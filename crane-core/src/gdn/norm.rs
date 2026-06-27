//! RMSNorm with a SiLU-gated output: `y = (x / rms(x)) * weight * silu(gate)`.
//!
//! This is the output-side normalization used by every Gated Delta Net layer in
//! Qwen 3.5 — the `z` projection (silu-gate) modulates the normalized
//! recurrence output before it goes through `out_proj`.

use candle_core::{Module, Result, Tensor};
use candle_nn::{rms_norm, RmsNorm, VarBuilder};

/// `RmsNorm(x) * silu(z)` with a learned per-channel weight.
pub struct RmsNormGated {
    norm: RmsNorm,
}

impl RmsNormGated {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let norm = rms_norm(size, eps, vb)?;
        Ok(Self { norm })
    }

    /// Forward pass. `x` and `gate` must share shape `[..., size]` and dtype.
    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(candle_core::DType::F32)?;
        let gate = gate.to_dtype(candle_core::DType::F32)?;
        let normalized = self.norm.forward(&x)?;
        let silu_gate = candle_nn::ops::silu(&gate)?;
        let out = normalized.broadcast_mul(&silu_gate)?;
        out.to_dtype(dtype)
    }

    /// Length of the learned per-channel weight vector. Exposed so that
    /// callers (e.g. `GatedDeltaNet`) can recover the per-head value dim
    /// without a separate config field.
    pub fn weight_len(&self) -> usize {
        self.norm.weight().dim(0).unwrap_or(0)
    }
}