// SPDX-License-Identifier: MIT

//! Block selection of the QSA indexer ([`crate::models::qwen4_exp::indexer`]):
//! from per-block scores, the additive attention mask that keeps each query's
//! `keep_blocks` best visible blocks plus its incomplete tail block.
//!
//! On CUDA (`kernels/cuda/qsa_mask.cu`), Metal (`kernels/metal/qsa_mask.metal`)
//! and SYCL (`kernels/sycl/qsa_mask.cpp`) the scores never leave the device. On other backends the scores are read
//! back and selected on the host, which costs a device sync per call.
//!
//! Ties go to the earlier block, on both paths.

use candle_core::{DType, Device, Result, Tensor, bail};

/// The additive mask `[seq, cells]` for `seq` queries at cache positions
/// `start..start + seq`: `0` on the kept blocks (`ratio` cells each) and on the
/// tail cells up to the query itself, `-inf` elsewhere.
///
/// `scores` is `[seq, blocks]` `F32` (`blocks = cells / ratio`); it is only read
/// for queries that see more than `keep_blocks` complete blocks, and may be
/// `None` when no query does.
///
/// # Errors
///
/// Returns an error if the shapes disagree or a kernel fails.
pub fn qsa_block_mask(
    scores: Option<&Tensor>,
    seq: usize,
    start: usize,
    ratio: usize,
    keep_blocks: usize,
    device: &Device,
) -> Result<Tensor> {
    let cells = start + seq;
    let blocks = cells / ratio;
    if let Some(s) = scores {
        if s.dims() != [seq, blocks] || s.dtype() != DType::F32 {
            bail!(
                "qsa scores must be F32 [{seq}, {blocks}], got {:?} {:?}",
                s.dtype(),
                s.dims()
            )
        }
    } else if blocks > keep_blocks {
        bail!("qsa_block_mask needs scores: {blocks} blocks exceed the {keep_blocks} kept")
    }

    #[cfg(feature = "cuda")]
    if let Some(s) = scores
        && s.device().is_cuda()
    {
        return cuda::qsa_block_mask(s, seq, start, cells, ratio, keep_blocks);
    }
    #[cfg(feature = "metal")]
    if let Some(s) = scores
        && s.device().is_metal()
    {
        return metal::qsa_block_mask(s, seq, start, cells, ratio, keep_blocks);
    }
    #[cfg(feature = "sycl")]
    if let Some(s) = scores
        && s.device().is_sycl()
    {
        return sycl::qsa_block_mask(s, seq, start, cells, ratio, keep_blocks);
    }

    let host_scores = scores
        .map(|s| s.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>())
        .transpose()?;
    let mask = host_mask(
        host_scores.as_deref(),
        seq,
        start,
        ratio,
        keep_blocks,
        blocks,
    );
    Tensor::from_vec(mask, (seq, cells), device)
}

/// The reference selection, on the host.
fn host_mask(
    scores: Option<&[f32]>,
    seq: usize,
    start: usize,
    ratio: usize,
    keep_blocks: usize,
    blocks: usize,
) -> Vec<f32> {
    let cells = start + seq;
    let mut mask = vec![f32::NEG_INFINITY; seq * cells];
    for i in 0..seq {
        let pos = start + i;
        let visible_blocks = (pos + 1) / ratio;
        let row = &mut mask[i * cells..(i + 1) * cells];
        let mut order: Vec<usize> = (0..visible_blocks).collect();
        if visible_blocks > keep_blocks {
            let s =
                &scores.expect("checked by the caller")[i * blocks..i * blocks + visible_blocks];
            // Highest score first; ties keep the earlier block.
            order.sort_by(|&a, &b| s[b].total_cmp(&s[a]).then(a.cmp(&b)));
            order.truncate(keep_blocks);
        }
        for b in order {
            row[b * ratio..(b + 1) * ratio].fill(0.0);
        }
        row[visible_blocks * ratio..=pos].fill(0.0);
    }
    mask
}

#[cfg(feature = "sycl")]
mod sycl {
    use std::ffi::c_void;

    use candle_core::op::BackpropOp;
    use candle_core::{DType, Result, Storage, SyclStorage, Tensor};

    // libcrane_gdn_sycl.so — linked by build.rs when `--features sycl`.
    unsafe extern "C" {
        fn crane_qsa_mask_sycl(
            queue: *mut c_void,
            scores: *const f32,
            mask: *mut f32,
            seq: i32,
            blocks: i32,
            cells: i32,
            start: i32,
            ratio: i32,
            keep: i32,
        ) -> i32;
    }

    fn int(n: usize) -> Result<i32> {
        i32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("{n} exceeds i32")))
    }

    pub fn qsa_block_mask(
        scores: &Tensor,
        seq: usize,
        start: usize,
        cells: usize,
        ratio: usize,
        keep_blocks: usize,
    ) -> Result<Tensor> {
        let dev = scores.device().as_sycl_device()?.clone();
        let scores = scores.contiguous()?;
        let (storage, layout) = scores.storage_and_layout();
        let src = match &*storage {
            Storage::Sycl(st) => unsafe {
                st.buf().as_ptr().cast::<f32>().add(layout.start_offset())
            },
            _ => candle_core::bail!("qsa scores must be on SYCL"),
        };
        let n = seq * cells;
        let out = dev.alloc_bytes(n * DType::F32.size_in_bytes())?;
        let status = unsafe {
            crane_qsa_mask_sycl(
                dev.queue().native_ptr(),
                src,
                out.as_mut_ptr().cast::<f32>(),
                int(seq)?,
                int(cells / ratio)?,
                int(cells)?,
                int(start)?,
                int(ratio)?,
                int(keep_blocks)?,
            )
        };
        drop(storage);
        if status != 0 {
            candle_core::bail!("crane_qsa_mask_sycl failed (status {status})");
        }
        Ok(Tensor::from_storage(
            Storage::Sycl(SyclStorage::from_buffer(&dev, out, DType::F32, n)),
            (seq, cells),
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{Result, Storage, Tensor};

    mod ptx {
        include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
    }

    const MODULE_NAME: &str = "crane_qsa_mask";
    /// `QSA_THREADS` in the kernel.
    const THREADS: u32 = 256;

    fn int(n: usize) -> Result<i32> {
        i32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("{n} exceeds i32")))
    }

    pub fn qsa_block_mask(
        scores: &Tensor,
        seq: usize,
        start: usize,
        cells: usize,
        ratio: usize,
        keep_blocks: usize,
    ) -> Result<Tensor> {
        let dev = scores.device().as_cuda_device()?.clone();
        let scores = scores.contiguous()?;
        let blocks = cells / ratio;
        let (blocks_i, cells_i, start_i, ratio_i, keep_i) = (
            int(blocks)?,
            int(cells)?,
            int(start)?,
            int(ratio)?,
            int(keep_blocks)?,
        );
        let func = dev.get_or_load_custom_func("qsa_topk_mask_f32", MODULE_NAME, ptx::QSA_MASK)?;
        let (storage, layout) = scores.storage_and_layout();
        let Storage::Cuda(cuda) = &*storage else {
            candle_core::bail!("qsa scores must be on CUDA")
        };
        let view = cuda.as_cuda_slice::<f32>()?.slice(layout.start_offset()..);
        let out = unsafe { dev.alloc::<f32>(seq * cells) }?;
        let mut b = func.builder();
        b.arg(&view)
            .arg(&out)
            .arg(&blocks_i)
            .arg(&cells_i)
            .arg(&start_i)
            .arg(&ratio_i)
            .arg(&keep_i);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (int(seq)? as u32, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        drop(storage);
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)),
            (seq, cells),
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(feature = "metal")]
mod metal {
    use candle_core::{DType, Result, Tensor};
    use candle_metal_kernels::metal::ComputeCommandEncoder;
    use objc2_metal::MTLSize;

    use crate::ops::metal_util;

    /// `QSA_THREADS` in the kernel.
    const THREADS: usize = 256;

    /// `QsaParams` in the kernel.
    #[repr(C)]
    struct QsaParams {
        blocks: i32,
        cells: i32,
        start: i32,
        ratio: i32,
        keep: i32,
    }

    fn int(n: usize) -> Result<i32> {
        i32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("{n} exceeds i32")))
    }

    pub fn qsa_block_mask(
        scores: &Tensor,
        seq: usize,
        start: usize,
        cells: usize,
        ratio: usize,
        keep_blocks: usize,
    ) -> Result<Tensor> {
        let dev = scores.device().as_metal_device()?.clone();
        let scores = scores.contiguous()?;
        let params = QsaParams {
            blocks: int(cells / ratio)?,
            cells: int(cells)?,
            start: int(start)?,
            ratio: int(ratio)?,
            keep: int(keep_blocks)?,
        };
        let pipeline = metal_util::pipeline(
            &dev,
            "qsa_mask",
            || include_str!("../../../kernels/metal/qsa_mask.metal").to_string(),
            "qsa_topk_mask_f32",
        )?;
        let out = metal_util::output(&dev, seq * cells, DType::F32, "qsa_mask")?;
        let (storage, _) = scores.storage_and_layout();
        {
            let (buf, offset) = metal_util::buffer(&storage, &scores, 0, "qsa scores")?;
            let encoder = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = encoder.as_ref();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_input_buffer(0, Some(buf), offset);
            enc.set_output_buffer(1, Some(&out), 0);
            enc.set_bytes(2, &params);
            // One threadgroup per query row.
            enc.dispatch_thread_groups(
                MTLSize {
                    width: seq,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
        }
        drop(storage);
        Ok(metal_util::wrap(&dev, out, (seq, cells), DType::F32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(feature = "cuda", feature = "metal", feature = "sycl"))]
    /// Random scores quantized to a few levels (so ties are plentiful) and
    /// some exact zeros, as a `ReLU`'d score matrix has.
    fn scores(seq: usize, blocks: usize) -> Tensor {
        let raw = Tensor::randn(0f32, 1.0, (seq, blocks), &Device::Cpu).unwrap();
        (raw * 2.0).unwrap().relu().unwrap().floor().unwrap()
    }

    #[cfg(any(feature = "cuda", feature = "metal", feature = "sycl"))]
    fn check(device: &Device) {
        // Prefill chunks (rows see different block counts), decode, and a
        // row count above the kernel's thread count.
        for (seq, start, ratio, keep) in [
            (1usize, 4095usize, 4usize, 64usize),
            (7, 2000, 4, 100),
            (33, 600, 8, 16),
            (5, 3000, 4, 1000),
            (300, 1500, 4, 128),
            (4, 40, 4, 64),
        ] {
            let cells = start + seq;
            let blocks = cells / ratio;
            let s = scores(seq, blocks);
            let need = blocks > keep;
            let want = qsa_block_mask(need.then_some(&s), seq, start, ratio, keep, &Device::Cpu)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let on = s.to_device(device).unwrap();
            let got = qsa_block_mask(need.then_some(&on), seq, start, ratio, keep, device)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            assert_eq!(
                got, want,
                "seq={seq} start={start} ratio={ratio} keep={keep}"
            );
        }
    }

    #[test]
    fn cpu_keeps_best_blocks_and_tail() {
        // 3 queries at positions 8..11, ratio 4, keep 1: the best of blocks
        // 0 and 1 (scores 1 vs 5; 11 cells = 2 complete blocks), plus the tail.
        let s = Tensor::new(&[[1f32, 5.0], [1.0, 5.0], [1.0, 5.0]], &Device::Cpu).unwrap();
        let mask = qsa_block_mask(Some(&s), 3, 8, 4, 1, &Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let ninf = f32::NEG_INFINITY;
        // Row 0 (pos 8): blocks visible = 2 -> keep block 1; tail = cell 8.
        assert_eq!(
            mask[0],
            [ninf, ninf, ninf, ninf, 0., 0., 0., 0., 0., ninf, ninf]
        );
        // Row 2 (pos 10): still 2 complete blocks, tail = cells 8..=10.
        assert_eq!(
            mask[2],
            [ninf, ninf, ninf, ninf, 0., 0., 0., 0., 0., 0., 0.]
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_kernel_matches_host_selection() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        for _ in 0..20 {
            check(&device);
        }
    }

    #[cfg(feature = "sycl")]
    #[test]
    fn sycl_kernel_matches_host_selection() {
        if !candle_core::utils::sycl_is_available() {
            return;
        }
        let device = Device::new_sycl(0).unwrap();
        for _ in 0..20 {
            check(&device);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_kernel_matches_host_selection() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        for _ in 0..20 {
            check(&device);
        }
    }
}
