// SPDX-License-Identifier: MIT

//! Launcher for `kernels/cuda/quant_ternary.cu` on ROCm/HIP.
//!
//! The same source the CUDA build compiles to PTX; here candle compiles it
//! with `hipcc` on first use and caches the code object on disk (see
//! [`crate::ops::rocm`]). Every intrinsic the kernels need (`__shfl_down_sync`,
//! `__shfl_sync`, `__syncthreads`, `__half2float`, dynamic shared memory) is
//! either native on `ROCm` or already bridged by candle-rocm's HIP shim, so
//! this mirrors [`super::cuda`] exactly rather than needing a kernel rewrite.

use candle_core::rocm_backend::rocm_rs;
use candle_core::{DType, Result, Tensor, bail};
use rocm_rs::hip::Dim3;

use crate::ops::rocm::{self, arg, wrap_f32};
use crate::quantized::ternary::{GdnPermutation, HadamardMode, TernaryEncoding};

const MODULE_NAME: &str = "crane_quant_ternary";
const SOURCE: &str = include_str!("../../../kernels/cuda/quant_ternary.cu");

/// `input` (`[rows, cols]`, `ROCm` f32) times the packed ternary weight
/// (`[output_rows, cols]`), returning `[rows, output_rows]` f32.
///
/// Mirrors [`super::cuda::linear_f32`]: optionally applies a forward
/// Hadamard transform (sign multiply + FWHT) to `input` before the ternary
/// matvec, decoding `packed` per the given [`TernaryEncoding`].
///
/// # Errors
///
/// Returns an error if `mode` is [`HadamardMode::Inverse`], if any tensor is
/// not on a `ROCm` device with the expected dtype, or if a kernel launch fails.
#[allow(clippy::too_many_arguments)]
pub fn linear_f32(
    input: &Tensor,
    packed: &Tensor,
    encoding: TernaryEncoding,
    output_rows: usize,
    cols: usize,
    signs: &Tensor,
    block_size: usize,
    mode: HadamardMode,
    gdn_permutation: Option<GdnPermutation>,
) -> Result<Tensor> {
    if mode == HadamardMode::Inverse {
        bail!("inverse Hadamard is only valid after an embedding lookup")
    }
    let input = input.contiguous()?;
    let rows = input.elem_count() / cols;
    let dev = input.device().as_rocm_device()?.clone();

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_ptr = rocm::device_ptr(
        &input_storage,
        input_layout,
        DType::F32,
        "quant_ternary input",
    )?;

    let transformed_buf;
    let matmul_input_ptr = if mode == HadamardMode::Forward {
        let (sign_storage, sign_layout) = signs.storage_and_layout();
        let sign_ptr = rocm::device_ptr(
            &sign_storage,
            sign_layout,
            DType::F32,
            "quant_ternary signs",
        )?;
        transformed_buf = dev.alloc::<f32>(rows * cols)?;
        let transformed_ptr = transformed_buf.as_ptr();
        let chunks = rows * (cols / block_size);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let (rows_i, cols_i, block_size_i) = (rows as i32, cols as i32, block_size as i32);
        let (perm_hd, perm_nk, perm_rep) = gdn_permutation.map_or((0, 0, 0), |p| {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let (hd, nk, rep) = (p.head_dim as i32, p.key_heads as i32, p.repeats as i32);
            (hd, nk, rep)
        });
        let mut args = [
            arg(&input_ptr),
            arg(&sign_ptr),
            arg(&transformed_ptr),
            arg(&rows_i),
            arg(&cols_i),
            arg(&block_size_i),
            arg(&perm_hd),
            arg(&perm_nk),
            arg(&perm_rep),
        ];
        #[allow(clippy::cast_possible_truncation)]
        let grid = Dim3::new_1d(chunks as u32);
        let block = Dim3::new_1d(256);
        #[allow(clippy::cast_possible_truncation)]
        let shared_mem_bytes = (block_size * std::mem::size_of::<f32>()) as u32;
        let func =
            dev.get_or_load_custom_func("quant_ternary_hadamard_f32", MODULE_NAME, SOURCE)?;
        func.launch(grid, block, shared_mem_bytes, Some(dev.stream()), &mut args)
            .map_err(|e| {
                candle_core::Error::Msg(format!("quant_ternary_hadamard_f32 launch failed: {e}"))
            })?;
        drop(sign_storage);
        transformed_ptr
    } else {
        input_ptr
    };

    let (packed_storage, packed_layout) = packed.storage_and_layout();
    let packed_ptr = rocm::device_ptr(
        &packed_storage,
        packed_layout,
        DType::U8,
        "quant_ternary packed weight",
    )?;

    let output_buf = dev.alloc::<f32>(rows * output_rows)?;
    let output_ptr = output_buf.as_ptr();
    let batch4 = rows >= 4;
    let kernel = match (encoding, batch4) {
        (TernaryEncoding::Pq2_0, false) => "quant_ternary_pq2_matvec_f32",
        (TernaryEncoding::Ptq1_0, false) => "quant_ternary_ptq1_matvec_f32",
        (TernaryEncoding::Pq2_0, true) => "quant_ternary_pq2_matvec_batch4_f32",
        (TernaryEncoding::Ptq1_0, true) => "quant_ternary_ptq1_matvec_batch4_f32",
    };
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let (rows_i, output_rows_i, cols_i) = (rows as i32, output_rows as i32, cols as i32);
    let mut args = [
        arg(&packed_ptr),
        arg(&matmul_input_ptr),
        arg(&output_ptr),
        arg(&rows_i),
        arg(&output_rows_i),
        arg(&cols_i),
    ];
    #[allow(clippy::cast_possible_truncation)]
    let grid = Dim3::new_2d(
        output_rows.div_ceil(4) as u32,
        if batch4 { rows.div_ceil(4) } else { rows } as u32,
    );
    let block = Dim3::new_1d(128);
    let func = dev.get_or_load_custom_func(kernel, MODULE_NAME, SOURCE)?;
    func.launch(grid, block, 0, Some(dev.stream()), &mut args)
        .map_err(|e| candle_core::Error::Msg(format!("{kernel} launch failed: {e}")))?;
    drop(input_storage);
    drop(packed_storage);

    let mut dims = input.dims().to_vec();
    *dims.last_mut().unwrap() = output_rows;
    Ok(wrap_f32(output_buf, &dev, dims))
}
