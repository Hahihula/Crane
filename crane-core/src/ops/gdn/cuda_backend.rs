//! CUDA launcher for the fused Gated Delta Net recurrence kernel.
//!
//! Collapses the per-timestep Candle op graph into a single kernel launch
//! (`kernels/cuda/gdn.cu`). Inputs must be contiguous f32 CUDA tensors in the
//! layouts documented on [`gdn_recurrence_cuda`]; `q` is expected pre-scaled
//! by `1/sqrt(K)` (the caller does this, matching the CPU reference).

use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::op::BackpropOp;
use candle_core::{CudaStorage, Result, Storage, Tensor};

use super::chunked::MAX_CHUNK_SIZE;

// PTX compiled from kernels/cuda/gdn.cu — embedded at build time.
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

const MODULE_NAME: &str = "crane_gdn";

/// Run the gated delta rule recurrence on CUDA.
///
/// Shapes: `q,k = [BH,S,K]`, `v = [BH,S,V]`, `g,beta = [BH,S]`,
/// `state = [BH,K,V]`. Returns `(y = [BH,S,V], state_out = [BH,K,V])`.
pub fn gdn_recurrence_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (bh, s, kdim) = q.dims3()?;
    let vdim = v.dim(2)?;
    if kdim > 256 {
        candle_core::bail!("gdn cuda kernel supports head_k_dim <= 256, got {kdim}");
    }
    let dev = q.device().as_cuda_device()?.clone();

    // Hold the storage guards alive until after the launch; the slice views
    // borrow from them.
    let (q_s, q_l) = q.storage_and_layout();
    let q_sl = match &*q_s {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("gdn: q must be a cuda tensor"),
    };
    let (k_s, k_l) = k.storage_and_layout();
    let k_sl = match &*k_s {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("gdn: k must be a cuda tensor"),
    };
    let (v_s, v_l) = v.storage_and_layout();
    let v_sl = match &*v_s {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("gdn: v must be a cuda tensor"),
    };
    let (g_s, g_l) = g.storage_and_layout();
    let g_sl = match &*g_s {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("gdn: g must be a cuda tensor"),
    };
    let (beta_s, beta_l) = beta.storage_and_layout();
    let beta_sl = match &*beta_s {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("gdn: beta must be a cuda tensor"),
    };
    let (state_s, state_l) = state.storage_and_layout();
    let state_sl = match &*state_s {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("gdn: state must be a cuda tensor"),
    };

    // Offsets (tensors are passed contiguous; honor any start offset anyway).
    let q_v = q_sl.slice(q_l.start_offset()..);
    let k_v = k_sl.slice(k_l.start_offset()..);
    let v_v = v_sl.slice(v_l.start_offset()..);
    let g_v = g_sl.slice(g_l.start_offset()..);
    let beta_v = beta_sl.slice(beta_l.start_offset()..);
    let state_v = state_sl.slice(state_l.start_offset()..);

    let y_buf = unsafe { dev.alloc::<f32>(bh * s * vdim) }?;
    let state_out_buf = unsafe { dev.alloc::<f32>(bh * kdim * vdim) }?;

    // V columns are independent and can be tiled across blocks, but for these
    // shapes one block per head (V_TILE == V, ~4 warps) hides the serial
    // inner-loop latency best, so default there. Tunable via CRANE_GDN_VTILE.
    let v_tile = std::env::var("CRANE_GDN_VTILE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|t| *t > 0)
        .unwrap_or(vdim)
        .min(vdim);
    let tiles = vdim.div_ceil(v_tile);
    let cfg = LaunchConfig {
        grid_dim: ((bh * tiles) as u32, 1, 1),
        block_dim: (v_tile as u32, 1, 1),
        shared_mem_bytes: (2 * kdim * std::mem::size_of::<f32>()) as u32,
    };

    // Specialized register-resident kernel for K==128 (state column in
    // registers); runtime-K fallback otherwise (local memory).
    let (bh_i, s_i, k_i, v_i, vt_i) =
        (bh as i32, s as i32, kdim as i32, vdim as i32, v_tile as i32);
    if kdim == 128 {
        let func = dev.get_or_load_custom_func("gdn_recurrence_f32_k128", MODULE_NAME, ptx::GDN)?;
        let mut builder = func.builder();
        builder.arg(&q_v);
        builder.arg(&k_v);
        builder.arg(&v_v);
        builder.arg(&g_v);
        builder.arg(&beta_v);
        builder.arg(&state_v);
        builder.arg(&state_out_buf);
        builder.arg(&y_buf);
        builder.arg(&bh_i);
        builder.arg(&s_i);
        builder.arg(&v_i);
        builder.arg(&vt_i);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let func = dev.get_or_load_custom_func("gdn_recurrence_f32", MODULE_NAME, ptx::GDN)?;
        let mut builder = func.builder();
        builder.arg(&q_v);
        builder.arg(&k_v);
        builder.arg(&v_v);
        builder.arg(&g_v);
        builder.arg(&beta_v);
        builder.arg(&state_v);
        builder.arg(&state_out_buf);
        builder.arg(&y_buf);
        builder.arg(&bh_i);
        builder.arg(&s_i);
        builder.arg(&k_i);
        builder.arg(&v_i);
        builder.arg(&vt_i);
        unsafe { builder.launch(cfg) }.w()?;
    }

    let y = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(y_buf, dev.clone())),
        (bh, s, vdim),
        BackpropOp::none(),
        false,
    );
    let state_out = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(state_out_buf, dev.clone())),
        (bh, kdim, vdim),
        BackpropOp::none(),
        false,
    );
    Ok((y, state_out))
}

/// Batched inverse of a unit lower-triangular `[B_all,cs,cs]` matrix on CUDA:
/// `T_inv = (I + kb)^-1`. Used by the chunked gated-delta-rule prefill path
/// (`ops/gdn/chunked.rs`) in place of a 63-step host-orchestrated Candle loop.
///
/// # Errors
///
/// Returns an error if `cs > 78` (the kernel's shared-memory limit without a
/// dynamic-shared-memory opt-in), if `kb` is not a contiguous f32 CUDA
/// tensor, or if the launch fails.
pub fn gdn_chunk_tri_inverse_cuda(kb: &Tensor) -> Result<Tensor> {
    tri_inverse(kb, false)
}
pub fn gdn_chunk_tri_inverse_torch_cuda(kb: &Tensor) -> Result<Tensor> {
    tri_inverse(kb, true)
}
fn tri_inverse(kb: &Tensor, torch: bool) -> Result<Tensor> {
    let (b_all, cs, cs2) = kb.dims3()?;
    debug_assert_eq!(cs, cs2, "gdn chunk tri inverse: kb must be square");
    if cs > MAX_CHUNK_SIZE {
        candle_core::bail!(
            "gdn chunk tri inverse kernel supports chunk size <= {MAX_CHUNK_SIZE}, got {cs}"
        );
    }
    let dev = kb.device().as_cuda_device()?.clone();

    let (kb_s, kb_l) = kb.storage_and_layout();
    let kb_sl = match &*kb_s {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("gdn chunk tri inverse: kb must be a cuda tensor"),
    };
    let kb_v = kb_sl.slice(kb_l.start_offset()..);

    let t_inv_buf = unsafe { dev.alloc::<f32>(b_all * cs * cs) }?;

    let cfg = LaunchConfig {
        grid_dim: (b_all as u32, 1, 1),
        block_dim: (cs as u32, 1, 1),
        shared_mem_bytes: (2 * cs * cs * std::mem::size_of::<f32>()) as u32,
    };

    #[allow(clippy::cast_possible_wrap)]
    let cs_i = cs as i32;
    let name = if torch {
        "gdn_chunk_tri_inverse_torch_f32"
    } else {
        "gdn_chunk_tri_inverse_f32"
    };
    let func = dev.get_or_load_custom_func(name, MODULE_NAME, ptx::GDN)?;
    let mut builder = func.builder();
    builder.arg(&kb_v);
    builder.arg(&t_inv_buf);
    builder.arg(&cs_i);
    unsafe { builder.launch(cfg) }.w()?;

    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(t_inv_buf, dev)),
        (b_all, cs, cs),
        BackpropOp::none(),
        false,
    ))
}

/// F32 rsqrt (kind 0) or stable softplus (kind 1), matching Torch math.
pub fn float_math(xs: &Tensor, kind: i32) -> Result<Tensor> {
    let xs = xs.contiguous()?;
    let dev = xs.device().as_cuda_device()?.clone();
    let (storage, layout) = xs.storage_and_layout();
    let slice = match &*storage {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("gdn math requires CUDA F32"),
    };
    let input = slice.slice(layout.start_offset()..);
    let n = xs.elem_count() as u32;
    let out = unsafe { dev.alloc::<f32>(n as usize) }?;
    let name = if kind == 2 {
        "gdn_l2_bf16_128"
    } else {
        "gdn_float_math"
    };
    let func = dev.get_or_load_custom_func(name, MODULE_NAME, ptx::GDN)?;
    let mut builder = func.builder();
    builder.arg(&input);
    builder.arg(&out);
    builder.arg(&n);
    builder.arg(&kind);
    let threads = if kind == 2 { n / 4 } else { n };
    unsafe { builder.launch(LaunchConfig::for_num_elems(threads)) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)),
        xs.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}
