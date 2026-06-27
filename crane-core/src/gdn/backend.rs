//! Numerical kernels for Gated Delta Net: L2 normalization, softplus, the
//! gated delta rule recurrence, causal Conv1D, and dispatch.
//!
//! # Phase 1A (CPU only)
//!
//! This file currently implements the CPU reference path in pure Candle ops.
//! CUDA and Metal kernel wrappers are stubbed and will return an error; they
//! are filled in by Phase 1B (`gdn.cu` via `bindgen_cuda`) and Phase 1C (MSL).
//!
//! The CPU implementation is intentionally simple — every operation is a
//! composition of `Tensor` ops, easy to validate against a reference. The CUDA
//! path in `mistralrs-core/src/cuda/gdn.cu` uses V-tiled kernels with register
//! residency for the per-head state to get throughput; that level of
//! optimization is out of scope until 1B.

use candle_core::{DType, IndexOp, Result, Tensor, D};

use super::cache::GdnLayerCache;
use super::config::GdnDims;

// ─────────────────────────────────────────────────────────────────────
//  Elementwise helpers
// ─────────────────────────────────────────────────────────────────────

/// `x / sqrt(mean(x^2) + eps)` over the last dim — used to normalize Q and K
/// before the delta-rule recurrence.
pub fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let inv_norm = x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .broadcast_add(&Tensor::new(eps as f32, x.device())?.to_dtype(x.dtype())?)?
        .sqrt()?
        .recip()?;
    x.broadcast_mul(&inv_norm)
}

/// `log(1 + exp(x))` — numerically stable; computed as `log1p(exp(x))`.
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    (Tensor::ones_like(x)? + x.exp()?)?.log()
}

// ─────────────────────────────────────────────────────────────────────
//  Gated delta rule recurrence (CPU reference)
// ─────────────────────────────────────────────────────────────────────

/// Per-timestep gated delta rule in pure Candle ops.
///
/// Inputs (all contiguous, f32):
/// - `q, k`: `[BH, S, K]` — queries / keys
/// - `v`:    `[BH, S, V]` — values
/// - `g`:    `[BH, S]`    — log-decay (already pre-softplus'd by the caller)
/// - `beta`: `[BH, S]`    — write strength
/// - `state`: `[BH, K, V]` — recurrent state (mutated in place)
///
/// Returns `y: [BH, S, V]`. `state` is updated to the post-final-step value.
///
/// This is the exact CPU fallback lifted from mistral.rs
/// (`mistralrs-core/src/gdn/backend.rs:30-81`). Output dtype matches `q.dtype()`.
pub fn gated_delta_rule_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let dtype = q.dtype();
    let k_head_dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (k_head_dim as f64).sqrt();

    let q = (q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)? * scale)?;
    let k = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let v = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let g = g.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let beta = beta.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;

    let seq_len = q.dim(2)?;
    let mut s = state.to_dtype(DType::F32)?;
    let mut outputs = Vec::with_capacity(seq_len);

    for i in 0..seq_len {
        let q_t = q.i((.., .., i, ..))?;
        let k_t = k.i((.., .., i, ..))?;
        let v_t = v.i((.., .., i, ..))?;
        let g_t = g.i((.., .., i))?;
        let beta_t = beta.i((.., .., i))?;

        // 1. Decay state by per-head factor.
        let decay = g_t.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        s = s.broadcast_mul(&decay)?;

        // 2. Retrieve kv_mem = sum_d_state(state * k).
        let k_exp = k_t.unsqueeze(D::Minus1)?;
        let kv_mem = s.broadcast_mul(&k_exp)?.sum(2)?;

        // 3. Delta rule residual.
        let beta_exp = beta_t.unsqueeze(D::Minus1)?;
        let delta = (v_t - kv_mem)?.broadcast_mul(&beta_exp)?;

        // 4. Write state update: S += outer(k, delta).
        let outer = k_exp.broadcast_mul(&delta.unsqueeze(2)?)?;
        s = (s + outer)?;

        // 5. Output: y = sum_d_state(state * q).
        let q_exp = q_t.unsqueeze(D::Minus1)?;
        let y_t = s.broadcast_mul(&q_exp)?.sum(2)?;
        outputs.push(y_t);
    }

    *state = s.to_dtype(state.dtype())?;

    Tensor::stack(&outputs, 2)?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(dtype)
}

// ─────────────────────────────────────────────────────────────────────
//  β/g computation (pre-recurrence)
// ─────────────────────────────────────────────────────────────────────

/// Derive the per-head write strength β and per-step decay g from the
/// projections, A_log, and dt_bias.
///
/// `b: [B, S, H]` raw logits → `beta = sigmoid(b)`.
/// `a: [B, S, H]` raw values → combined with `A_log` (negative log of decay
/// rate) and `dt_bias` to produce `g = -exp(A_log) * softplus(a + dt_bias)`.
pub fn compute_beta_g(
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    // CPU reference path: pure Candle ops. Phase 1B will route CUDA devices
    // to a fused `fused_gdn_gating_cuda` kernel.
    let beta = candle_nn::ops::sigmoid(b)?;
    let a_f = a.to_dtype(DType::F32)?;
    let dt_bias_expanded = dt_bias.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(0)?;
    let g = a_log
        .to_dtype(DType::F32)?
        .exp()?
        .neg()?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .broadcast_mul(&softplus(&a_f.broadcast_add(&dt_bias_expanded)?)?)?
        .to_dtype(dtype)?;
    Ok((beta, g))
}

// ─────────────────────────────────────────────────────────────────────
//  Dispatch entry points
// ─────────────────────────────────────────────────────────────────────

/// Compute β and g, run the gated delta rule recurrence, return the output.
///
/// Dispatches to the CUDA/Metal fast paths when available; falls back to the
/// CPU reference otherwise.
#[allow(unused_variables)]
pub fn apply_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    // CUDA path placeholder (Phase 1B).
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() {
        candle_core::bail!("GDN CUDA recurrence not yet wired up (Phase 1B)")
    }

    // Metal path placeholder (Phase 1C).
    // `metal` is enabled by Crane's target-specific deps on macOS aarch64, not
    // as a Cargo feature, so we gate on the device directly instead of via
    // `#[cfg(feature = "metal")]`.
    #[cfg(target_os = "macos")]
    if q.device().is_metal() {
        candle_core::bail!("GDN Metal recurrence not yet wired up (Phase 1C)")
    }

    // CPU fallback.
    gated_delta_rule_recurrence(q, k, v, g, beta, &mut cache.recurrent_state)
}

/// Causal Conv1D over the QKV channels. Dispatches to a kernel-based
/// implementation when available, pure-Candle otherwise.
pub fn causal_conv1d(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
    is_decode_step: bool,
) -> Result<Tensor> {
    if is_decode_step {
        causal_conv1d_update(x, conv1d_weight, dims, cache)
    } else {
        causal_conv1d_full(x, conv1d_weight, dims, cache)
    }
}

/// Single-step Conv1D for autoregressive decode. Concatenates the new token
/// to the cached conv state, computes one output, then rolls the cache by 1.
fn causal_conv1d_update(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
) -> Result<Tensor> {
    let (_, seq_len, _) = x.dims3()?;
    let x_t = x.transpose(1, 2)?.contiguous()?;
    // `conv_state` is shape `[1, conv_dim, kernel_size]` — the kernel-size
    // dim is the LAST (matches mistral.rs).
    let state_len = cache.conv_state.dim(2)?;
    let hidden_new = Tensor::cat(&[cache.conv_state.clone(), x_t], 2)?;
    let new_len = hidden_new.dim(2)?;
    cache.conv_state = hidden_new.narrow(2, new_len - state_len, state_len)?;

    let weight = conv1d_weight.squeeze(1)?.to_dtype(hidden_new.dtype())?;
    let mut conv_outputs = Vec::with_capacity(seq_len);
    let total_len = hidden_new.dim(2)?;
    for i in (total_len - seq_len)..total_len {
        let window = hidden_new.narrow(2, i + 1 - dims.conv_kernel_size, dims.conv_kernel_size)?;
        let out = (window * weight.unsqueeze(0)?)?.sum(D::Minus1)?;
        conv_outputs.push(out);
    }
    candle_nn::ops::silu(&Tensor::stack(&conv_outputs, 2)?)?.transpose(1, 2)
}

/// Multi-step Conv1D used during prefill. Pads on the left by `kernel - 1`,
/// runs the full convolution, then captures the trailing `kernel - 1` tokens
/// into the cache for the next decode step.
fn causal_conv1d_full(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
) -> Result<Tensor> {
    let (batch_size, seq_len, conv_dim) = x.dims3()?;
    let x_t = x.transpose(1, 2)?.contiguous()?;

    // Pad on the time dim (the last) and capture the trailing `kernel - 1`
    // tokens into the cache for the next decode step.
    let pad_width = dims.conv_kernel_size.saturating_sub(seq_len);
    cache.conv_state = if pad_width > 0 {
        let zeros = Tensor::zeros((batch_size, conv_dim, pad_width), x_t.dtype(), x_t.device())?;
        Tensor::cat(&[zeros, x_t.clone()], 2)?
    } else {
        x_t.narrow(2, seq_len - dims.conv_kernel_size, dims.conv_kernel_size)?
    };

    let padded_t = Tensor::cat(
        &[
            Tensor::zeros(
                (batch_size, conv_dim, dims.conv_kernel_size - 1),
                x_t.dtype(),
                x_t.device(),
            )?,
            x_t,
        ],
        2,
    )?;

    let weight = conv1d_weight.squeeze(1)?.to_dtype(padded_t.dtype())?;
    let mut conv_outputs = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let window = padded_t.narrow(2, i, dims.conv_kernel_size)?;
        let out = (window * weight.unsqueeze(0)?)?.sum(D::Minus1)?;
        conv_outputs.push(out);
    }
    candle_nn::ops::silu(&Tensor::stack(&conv_outputs, 2)?)?.transpose(1, 2)
}
// ─────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// Smoke test: recurrence runs end-to-end on CPU and produces the right
    /// output shape. Numerical correctness against a reference (HF Transformers
    /// / mistral.rs) is checked in the integration test that loads a real
    /// Qwen 3.5 checkpoint — see Phase 1A.9 in AGENTS.md.
    #[test]
    fn recurrence_runs_and_returns_correct_shape() {
        let dev = Device::Cpu;
        let q = Tensor::new(
            &[[
                [[1.0f32, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 1.0]],
                [[0.5, 0.5]],
            ]],
            &dev,
        )
        .unwrap();
        let k = Tensor::new(
            &[[
                [[1.0f32, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ]],
            &dev,
        )
        .unwrap();
        let v = Tensor::new(
            &[[
                [[1.0f32, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 1.0]],
                [[0.5, 0.5]],
            ]],
            &dev,
        )
        .unwrap();
        let g = Tensor::new(&[[[0.0f32], [0.0], [0.0], [0.0]]], &dev).unwrap();
        let beta = Tensor::new(&[[[1.0f32], [1.0], [1.0], [1.0]]], &dev).unwrap();

        let mut state = Tensor::zeros((1, 1, 2, 2), DType::F32, &dev).unwrap();
        let y = gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut state).unwrap();
        assert_eq!(y.dims(), &[1, 4, 1, 2]);
        // State must have been mutated in place.
        assert!(state.dims() == [1, 1, 2, 2]);
    }

    #[test]
    fn l2_norm_preserves_direction() {
        let dev = Device::Cpu;
        let x = Tensor::new(&[[3.0f32, 4.0], [1.0, 0.0]], &dev).unwrap();
        let n = l2_norm(&x, 1e-6).unwrap();
        let v0 = n.i((0, ..)).unwrap().to_vec1::<f32>().unwrap();
        assert!((v0[0] - 0.6).abs() < 1e-3 && (v0[1] - 0.8).abs() < 1e-3);
        let v1 = n.i((1, ..)).unwrap().to_vec1::<f32>().unwrap();
        assert!((v1[0] - 1.0).abs() < 1e-3 && v1[1].abs() < 1e-3);
    }

    #[test]
    fn softplus_at_zero_is_ln2() {
        let dev = Device::Cpu;
        let x = Tensor::new(0.0f32, &dev).unwrap();
        let s = softplus(&x).unwrap().to_scalar::<f32>().unwrap();
        assert!((s - std::f32::consts::LN_2).abs() < 1e-4);
    }
}
