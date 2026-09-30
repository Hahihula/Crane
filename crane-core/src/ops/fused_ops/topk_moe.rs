// SPDX-License-Identifier: MIT
//! Fused top-K `MoE` routing kernel (Phase 8b): replaces the softmax -> `arg_sort` -> narrow ->
//! gather -> optional-normalize chain in `SparseMoeBlock::forward` (`crate::models::modules::moe`)
//! with a single kernel launch on CUDA/`ROCm`.
//!
//! One warp per token, softmax computed in registers via `__shfl_xor_sync` butterfly reductions
//! (every lane ends up with the same max/sum, unlike `__shfl_down_sync`, which only gives the
//! result to lane 0), then `top_k` rounds of warp-cooperative iterative argmax.
//!
//! Backends, selected like the rest of `fused_ops`: CUDA (PTX built by `build.rs` from
//! `kernels/cuda/topk_moe.cu`), `ROCm` (the same `.cu` source, compiled by `hipcc` on first
//! use), SYCL (`kernels/sycl/fused_ops.cpp`) and Metal (`kernels/metal/topk_moe.metal`). CPU,
//! and any expert count above [`MAX_FUSED_EXPERTS`], fall back to the plain candle op chain --
//! the same one this replaces on GPU.
//!
//! Direct dispatch rather than a `CustomOp` impl: the op returns two tensors of different dtypes
//! (`U32` ids, `F32` weights), past what `CustomOp1`/`CustomOp2` support.
//!
//! The fused kernel matches the portable path's softmax at `f32` precision, but its top-k
//! tie-break (smaller expert id wins) can differ from the portable path's `arg_sort_last_dim`,
//! which is documented as unstable with no tie-order guarantee -- neither order is more "correct"
//! and no downstream consumer depends on a particular one.

use candle_core::{D, DType, Result, Tensor};

/// Largest expert count the fused kernels support: `MAX_EPT * WARP_SIZE` in
/// `kernels/cuda/topk_moe.cu` (16 * 32), and the same 32 * 16 in the SYCL
/// and Metal ports. Above this, callers fall back to the portable chain
/// regardless of device. Only read from the GPU dispatch branches below.
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "sycl",
    feature = "metal"
))]
const MAX_FUSED_EXPERTS: usize = 512;

/// Fused top-K `MoE` routing: softmax over `logits`' experts, select the
/// `top_k` highest-probability experts per token, and optionally renormalize
/// their weights to sum to 1.
///
/// `logits` is `(num_tokens, num_experts)`, `F32`. Returns `(topk_ids,
/// topk_weights)`, both `(num_tokens, top_k)` -- `topk_ids` is `U32`,
/// `topk_weights` is `F32`.
///
/// # Errors
///
/// Returns an error if `logits` is not a 2D `F32` tensor, if `top_k` is zero
/// or exceeds the expert count, or if the underlying kernel launch or tensor
/// ops fail.
pub fn topk_moe_routing(
    logits: &Tensor,
    top_k: usize,
    norm_topk_prob: bool,
) -> Result<(Tensor, Tensor)> {
    if logits.rank() != 2 {
        candle_core::bail!(
            "topk_moe_routing expects a 2D (tokens, experts) tensor, got rank {}",
            logits.rank()
        );
    }
    if logits.dtype() != DType::F32 {
        candle_core::bail!(
            "topk_moe_routing expects F32 logits, got {:?}",
            logits.dtype()
        );
    }
    let n_experts = logits.dim(1)?;
    if top_k == 0 || top_k > n_experts {
        candle_core::bail!("topk_moe_routing: top_k {top_k} must be in 1..={n_experts}");
    }

    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() && n_experts <= MAX_FUSED_EXPERTS {
        return cuda_impl::topk_moe_routing(logits, top_k, norm_topk_prob);
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    if logits.device().is_rocm() && n_experts <= MAX_FUSED_EXPERTS {
        return rocm_impl::topk_moe_routing(logits, top_k, norm_topk_prob);
    }

    #[cfg(feature = "sycl")]
    if logits.device().is_sycl() && n_experts <= MAX_FUSED_EXPERTS {
        return sycl_impl::topk_moe_routing(logits, top_k, norm_topk_prob);
    }

    #[cfg(feature = "metal")]
    if logits.device().is_metal() && n_experts <= MAX_FUSED_EXPERTS {
        return metal_impl::topk_moe_routing(logits, top_k, norm_topk_prob);
    }

    portable_topk_moe_routing(logits, top_k, norm_topk_prob)
}

#[cfg(feature = "metal")]
mod metal_impl {
    //! Launcher for `topk_moe_f32` (`kernels/metal/topk_moe.metal`), the
    //! Metal port of the SYCL kernel.

    use candle_core::{DType, Result, Tensor};
    use candle_metal_kernels::metal::ComputeCommandEncoder;
    use objc2_metal::MTLSize;

    use crate::ops::metal_util;

    /// `TopkParams` in the kernel.
    #[repr(C)]
    struct TopkParams {
        n_tokens: i32,
        n_experts: i32,
        top_k: i32,
        norm_topk: i32,
    }

    /// See [`super::topk_moe_routing`]. `logits` must already be validated
    /// (2D, `F32`, `top_k` in range, at most `MAX_FUSED_EXPERTS` experts).
    pub fn topk_moe_routing(
        logits: &Tensor,
        top_k: usize,
        norm_topk_prob: bool,
    ) -> Result<(Tensor, Tensor)> {
        let logits = logits.contiguous()?;
        let (n_tokens, n_experts) = logits.dims2()?;
        let dev = logits.device().as_metal_device()?.clone();
        let pipeline = metal_util::pipeline(
            &dev,
            "topk_moe",
            || include_str!("../../../kernels/metal/topk_moe.metal").to_string(),
            "topk_moe_f32",
        )?;
        let int = |n: usize| {
            i32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("{n} exceeds i32")))
        };
        let params = TopkParams {
            n_tokens: int(n_tokens)?,
            n_experts: int(n_experts)?,
            top_k: int(top_k)?,
            norm_topk: i32::from(norm_topk_prob),
        };
        let n = n_tokens * top_k;
        let ids = metal_util::output(&dev, n, DType::U32, "topk_moe_ids")?;
        let weights = metal_util::output(&dev, n, DType::F32, "topk_moe_weights")?;
        let (storage, _) = logits.storage_and_layout();
        {
            let (buf, offset) = metal_util::buffer(&storage, &logits, 0, "logits")?;
            let encoder = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = encoder.as_ref();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_input_buffer(0, Some(buf), offset);
            enc.set_output_buffer(1, Some(&ids), 0);
            enc.set_output_buffer(2, Some(&weights), 0);
            enc.set_bytes(3, &params);
            // One 32-lane SIMD-group per token.
            enc.dispatch_thread_groups(
                MTLSize {
                    width: n_tokens,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
        }
        drop(storage);
        Ok((
            metal_util::wrap(&dev, ids, (n_tokens, top_k), DType::U32),
            metal_util::wrap(&dev, weights, (n_tokens, top_k), DType::F32),
        ))
    }
}

#[cfg(feature = "sycl")]
mod sycl_impl {
    //! Launcher for `crane_topk_moe_sycl` (`kernels/sycl/fused_ops.cpp`),
    //! the SYCL port of the CUDA kernel.

    use std::ffi::c_void;

    use candle_core::op::BackpropOp;
    use candle_core::{DType, Result, Storage, SyclStorage, Tensor};

    // libcrane_gdn_sycl.so — linked by build.rs when `--features sycl`.
    unsafe extern "C" {
        fn crane_topk_moe_sycl(
            queue: *mut c_void,
            logits: *const f32,
            out_ids: *mut u32,
            out_weights: *mut f32,
            n_tokens: i32,
            n_experts: i32,
            top_k: i32,
            norm_topk: i32,
        ) -> i32;
    }

    /// See [`super::topk_moe_routing`]. `logits` must already be validated
    /// (2D, `F32`, `top_k` in range) by the caller.
    pub fn topk_moe_routing(
        logits: &Tensor,
        top_k: usize,
        norm_topk_prob: bool,
    ) -> Result<(Tensor, Tensor)> {
        let logits = logits.contiguous()?;
        let (n_tokens, n_experts) = logits.dims2()?;
        let dev = logits.device().as_sycl_device()?.clone();
        let int = |n: usize| {
            i32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("{n} exceeds i32")))
        };

        let (storage, layout) = logits.storage_and_layout();
        let src = match &*storage {
            Storage::Sycl(st) => unsafe {
                st.buf().as_ptr().cast::<f32>().add(layout.start_offset())
            },
            _ => candle_core::bail!("topk_moe_routing: logits must be a sycl tensor"),
        };
        let n = n_tokens * top_k;
        let ids = dev.alloc_bytes(n * DType::U32.size_in_bytes())?;
        let weights = dev.alloc_bytes(n * DType::F32.size_in_bytes())?;
        let status = unsafe {
            crane_topk_moe_sycl(
                dev.queue().native_ptr(),
                src,
                ids.as_mut_ptr().cast::<u32>(),
                weights.as_mut_ptr().cast::<f32>(),
                int(n_tokens)?,
                int(n_experts)?,
                int(top_k)?,
                i32::from(norm_topk_prob),
            )
        };
        drop(storage);
        if status != 0 {
            candle_core::bail!("crane_topk_moe_sycl failed (status {status})");
        }
        let wrap = |buf, dtype| {
            Tensor::from_storage(
                Storage::Sycl(SyclStorage::from_buffer(&dev, buf, dtype, n)),
                (n_tokens, top_k),
                BackpropOp::none(),
                false,
            )
        };
        Ok((wrap(ids, DType::U32), wrap(weights, DType::F32)))
    }
}

/// CPU fallback, and the GPU fallback for expert counts above
/// [`MAX_FUSED_EXPERTS`]: the softmax -> `arg_sort` -> narrow -> gather chain
/// `SparseMoeBlock::forward` used before this module existed.
fn portable_topk_moe_routing(
    logits: &Tensor,
    top_k: usize,
    norm_topk_prob: bool,
) -> Result<(Tensor, Tensor)> {
    let probs = candle_nn::ops::softmax_last_dim(logits)?;
    let topk_ids = probs
        .arg_sort_last_dim(false)?
        .narrow(D::Minus1, 0, top_k)?
        .contiguous()?;
    let mut topk_weights = probs.gather(&topk_ids, D::Minus1)?;
    if norm_topk_prob {
        let sum = topk_weights.sum_keepdim(D::Minus1)?;
        topk_weights = topk_weights.broadcast_div(&sum)?;
    }
    Ok((topk_ids, topk_weights))
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    //! CUDA launcher for `topk_moe_routing_f32`
    //! (`kernels/cuda/topk_moe.cu`). Direct dispatch, mirroring
    //! `super::super::quant_attn::cuda_impl`'s pattern: this op returns two
    //! differently-typed output tensors, past `CustomOp1`/`CustomOp2`'s
    //! single-output limit.

    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{Result, Storage, Tensor};

    mod ptx {
        include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
    }

    const MODULE_NAME: &str = "crane_topk_moe";

    /// See [`super::topk_moe_routing`]. `logits` must already be validated
    /// (2D, `F32`, `top_k` in range) by the caller.
    pub fn topk_moe_routing(
        logits: &Tensor,
        top_k: usize,
        norm_topk_prob: bool,
    ) -> Result<(Tensor, Tensor)> {
        let logits = logits.contiguous()?;
        let (n_tokens, n_experts) = logits.dims2()?;
        let dev = logits.device().as_cuda_device()?.clone();

        let (l_s, l_l) = logits.storage_and_layout();
        let l_slice = match &*l_s {
            Storage::Cuda(c) => match &c.slice {
                CudaStorageSlice::F32(s) => s.slice(l_l.start_offset()..),
                _ => candle_core::bail!("topk_moe_routing: logits must be f32"),
            },
            _ => candle_core::bail!("topk_moe_routing: logits must be a cuda tensor"),
        };

        let func =
            dev.get_or_load_custom_func("topk_moe_routing_f32", MODULE_NAME, ptx::TOPK_MOE)?;

        let ids_dst = unsafe { dev.alloc::<u32>(n_tokens * top_k)? };
        let weights_dst = unsafe { dev.alloc::<f32>(n_tokens * top_k)? };

        let cfg = LaunchConfig {
            grid_dim: (n_tokens as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        // n_experts and top_k are model dims, far below i32::MAX.
        let (n_experts_i, top_k_i) = (n_experts as i32, top_k as i32);
        let norm_i = i32::from(norm_topk_prob);

        let mut builder = func.builder();
        builder.arg(&l_slice);
        builder.arg(&ids_dst);
        builder.arg(&weights_dst);
        builder.arg(&n_experts_i);
        builder.arg(&top_k_i);
        builder.arg(&norm_i);
        // SAFETY: the argument list matches `topk_moe_routing_f32`'s
        // signature in `kernels/cuda/topk_moe.cu`; the grid launches one
        // block (one warp) per token, covering every output row.
        unsafe { builder.launch(cfg) }.w()?;

        let ids = Tensor::from_storage(
            Storage::Cuda(CudaStorage {
                slice: CudaStorageSlice::U32(ids_dst),
                device: dev.clone(),
            }),
            (n_tokens, top_k),
            BackpropOp::none(),
            false,
        );
        let weights = Tensor::from_storage(
            Storage::Cuda(CudaStorage {
                slice: CudaStorageSlice::F32(weights_dst),
                device: dev.clone(),
            }),
            (n_tokens, top_k),
            BackpropOp::none(),
            false,
        );
        Ok((ids, weights))
    }
}

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
mod rocm_impl {
    //! ROCm/HIP launcher, the counterpart of [`super::cuda_impl`] against the
    //! same `kernels/cuda/topk_moe.cu` -- candle compiles it with `hipcc` on
    //! first use and caches the code object. Mirrors
    //! `super::super::quant_attn::rocm_impl`'s use of `crate::ops::rocm`'s
    //! shared pointer/launch/wrap helpers.

    use candle_core::{DType, Result, Tensor};

    use crate::ops::rocm::{self, arg};

    const MODULE_NAME: &str = "crane_topk_moe";
    const SOURCE: &str = include_str!("../../../kernels/cuda/topk_moe.cu");

    /// See [`super::topk_moe_routing`]. `logits` must already be validated
    /// (2D, `F32`, `top_k` in range) by the caller.
    pub fn topk_moe_routing(
        logits: &Tensor,
        top_k: usize,
        norm_topk_prob: bool,
    ) -> Result<(Tensor, Tensor)> {
        let logits = logits.contiguous()?;
        let (n_tokens, n_experts) = logits.dims2()?;
        let dev = logits.device().as_rocm_device()?.clone();

        let (l_s, l_l) = logits.storage_and_layout();
        let l_p = rocm::device_ptr(&l_s, l_l, DType::F32, "topk_moe_routing logits")?;

        let ids_dst = dev.alloc::<u32>(n_tokens * top_k)?;
        let weights_dst = dev.alloc::<f32>(n_tokens * top_k)?;
        let ids_p = ids_dst.as_ptr();
        let weights_p = weights_dst.as_ptr();

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        // n_experts and top_k are model dims, far below i32::MAX.
        let (n_experts_i, top_k_i) = (n_experts as i32, top_k as i32);
        let norm_i = i32::from(norm_topk_prob);
        #[allow(clippy::cast_possible_truncation)]
        // n_tokens is a batch size, far below u32::MAX.
        let n_tokens_u32 = n_tokens as u32;

        let mut args = vec![
            arg(&l_p),
            arg(&ids_p),
            arg(&weights_p),
            arg(&n_experts_i),
            arg(&top_k_i),
            arg(&norm_i),
        ];
        // SAFETY: the argument list matches `topk_moe_routing_f32`'s
        // signature in `kernels/cuda/topk_moe.cu`; the grid launches one
        // block (one warp) per token, covering every output row.
        unsafe {
            rocm::launch(
                &dev,
                MODULE_NAME,
                "topk_moe_routing_f32",
                SOURCE,
                n_tokens_u32,
                32,
                0,
                &mut args,
            )?;
        }

        let ids = rocm::wrap_u32(ids_dst, &dev, (n_tokens, top_k));
        let weights = rocm::wrap_f32(weights_dst, &dev, (n_tokens, top_k));
        Ok((ids, weights))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    /// Fused routing (here, the portable CPU path) must select the same
    /// experts and weights as computing softmax/top-k by hand.
    #[test]
    fn matches_reference_topk_softmax() {
        let logits = Tensor::new(
            &[[1.0f32, 3.0, 0.5, 2.0, -1.0], [0.1, 0.1, 5.0, 0.2, 0.3]],
            &Device::Cpu,
        )
        .unwrap();
        let (ids, weights) = topk_moe_routing(&logits, 2, false).unwrap();
        let ids = ids.to_vec2::<u32>().unwrap();
        let weights = weights.to_vec2::<f32>().unwrap();

        // Row 0: logits descending are index 1 (3.0), then index 3 (2.0).
        assert_eq!(ids[0], [1, 3]);
        // Row 1: index 2 (5.0) dominates, then a tie among the rest broken
        // by softmax value; index 4 (0.3) is the next-highest logit.
        assert_eq!(ids[1], [2, 4]);

        // Weights must equal softmax(logits) at those indices.
        let expected_row0_softmax = {
            let exp: Vec<f64> = [1.0f64, 3.0, 0.5, 2.0, -1.0]
                .iter()
                .map(|v| v.exp())
                .collect();
            let sum: f64 = exp.iter().sum();
            exp.iter().map(|v| v / sum).collect::<Vec<_>>()
        };
        assert!((f64::from(weights[0][0]) - expected_row0_softmax[1]).abs() < 1e-5);
        assert!((f64::from(weights[0][1]) - expected_row0_softmax[3]).abs() < 1e-5);
    }

    /// `norm_topk_prob` must renormalize the selected weights to sum to 1.
    #[test]
    fn norm_topk_prob_normalizes_to_one() {
        let logits = Tensor::new(&[[1.0f32, 3.0, 0.5, 2.0, -1.0]], &Device::Cpu).unwrap();
        let (_ids, weights) = topk_moe_routing(&logits, 3, true).unwrap();
        let weights = weights.to_vec2::<f32>().unwrap();
        let sum: f32 = weights[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    /// `top_k` must be validated against the expert count before dispatch.
    #[test]
    fn rejects_top_k_larger_than_experts() {
        let logits = Tensor::new(&[[1.0f32, 2.0, 3.0]], &Device::Cpu).unwrap();
        assert!(topk_moe_routing(&logits, 4, false).is_err());
        assert!(topk_moe_routing(&logits, 0, false).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_kernel_matches_portable_routing() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        kernel_matches_portable_routing(&device);
    }

    #[cfg(feature = "sycl")]
    #[test]
    fn sycl_kernel_matches_portable_routing() {
        if !candle_core::utils::sycl_is_available() {
            return;
        }
        kernel_matches_portable_routing(&Device::new_sycl(0).unwrap());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_kernel_matches_portable_routing() {
        if !candle_core::utils::metal_is_available() {
            return;
        }
        kernel_matches_portable_routing(&Device::new_metal(0).unwrap());
    }

    /// The kernel on `device` (SYCL or Metal) against the portable chain, at Qwen3.8-Flash-Next's
    /// 256 experts / top-10 and the 512-expert maximum. Random logits have
    /// no ties, so the selections must agree exactly.
    #[cfg(any(feature = "sycl", feature = "metal", feature = "cuda"))]
    fn kernel_matches_portable_routing(device: &Device) {
        for (tokens, experts, top_k) in [
            (1usize, 256usize, 10usize),
            (37, 256, 10),
            (5, 512, 8),
            (3, 60, 4),
        ] {
            let logits = Tensor::randn(0f32, 2.0, (tokens, experts), &Device::Cpu).unwrap();
            for norm in [false, true] {
                let (want_ids, want_w) = portable_topk_moe_routing(&logits, top_k, norm).unwrap();
                let (ids, w) =
                    topk_moe_routing(&logits.to_device(device).unwrap(), top_k, norm).unwrap();
                assert_eq!(
                    ids.to_device(&Device::Cpu)
                        .unwrap()
                        .to_vec2::<u32>()
                        .unwrap(),
                    want_ids.to_vec2::<u32>().unwrap(),
                    "ids, {experts} experts, norm={norm}"
                );
                let diff = (w.to_device(&Device::Cpu).unwrap() - want_w)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                assert!(diff < 1e-5, "weights differ by {diff}");
            }
        }
    }
}
