// SPDX-License-Identifier: MIT

use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::op::BackpropOp;
use candle_core::{CudaStorage, Result, Storage, Tensor, bail};

use crate::quantized::ternary::{GdnPermutation, HadamardMode, TernaryEncoding};

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

const MODULE_NAME: &str = "crane_quant_ternary";

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
    let dev = input.device().as_cuda_device()?.clone();

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_slice = match &*input_storage {
        Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
        _ => bail!("quant_ternary input must be CUDA f32"),
    };
    let input_slice = input_slice.slice(input_layout.start_offset()..);

    let transformed_buf;
    let matmul_input = if mode == HadamardMode::Forward {
        let (sign_storage, sign_layout) = signs.storage_and_layout();
        let sign_slice = match &*sign_storage {
            Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
            _ => bail!("quant_ternary signs must be CUDA f32"),
        };
        let sign_slice = sign_slice.slice(sign_layout.start_offset()..);
        transformed_buf = unsafe { dev.alloc::<f32>(rows * cols) }?;
        let func = dev.get_or_load_custom_func(
            "quant_ternary_hadamard_f32",
            MODULE_NAME,
            ptx::QUANT_TERNARY,
        )?;
        let chunks = rows * (cols / block_size);
        let rows_i = rows as i32;
        let cols_i = cols as i32;
        let block_size_i = block_size as i32;
        let (perm_hd, perm_nk, perm_rep) = gdn_permutation.map_or((0, 0, 0), |p| {
            (p.head_dim as i32, p.key_heads as i32, p.repeats as i32)
        });
        let mut builder = func.builder();
        builder.arg(&input_slice);
        builder.arg(&sign_slice);
        builder.arg(&transformed_buf);
        builder.arg(&rows_i);
        builder.arg(&cols_i);
        builder.arg(&block_size_i);
        builder.arg(&perm_hd);
        builder.arg(&perm_nk);
        builder.arg(&perm_rep);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (chunks as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: (block_size * std::mem::size_of::<f32>()) as u32,
            })
        }
        .w()?;
        transformed_buf.as_view()
    } else {
        input_slice
    };

    let (packed_storage, packed_layout) = packed.storage_and_layout();
    let packed_slice = match &*packed_storage {
        Storage::Cuda(storage) => storage.as_cuda_slice::<u8>()?,
        _ => bail!("quant_ternary packed weight must be CUDA u8"),
    };
    let packed_slice = packed_slice.slice(packed_layout.start_offset()..);
    let output_buf = unsafe { dev.alloc::<f32>(rows * output_rows) }?;
    let kernel = match encoding {
        TernaryEncoding::Pq2_0 => "quant_ternary_pq2_matvec_f32",
        TernaryEncoding::Ptq1_0 => "quant_ternary_ptq1_matvec_f32",
    };
    let func = dev.get_or_load_custom_func(kernel, MODULE_NAME, ptx::QUANT_TERNARY)?;
    let rows_i = rows as i32;
    let output_rows_i = output_rows as i32;
    let cols_i = cols as i32;
    let mut builder = func.builder();
    builder.arg(&packed_slice);
    builder.arg(&matmul_input);
    builder.arg(&output_buf);
    builder.arg(&rows_i);
    builder.arg(&output_rows_i);
    builder.arg(&cols_i);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (output_rows as u32, rows as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;

    let mut dims = input.dims().to_vec();
    *dims.last_mut().unwrap() = output_rows;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output_buf, dev)),
        dims,
        BackpropOp::none(),
        false,
    ))
}
