// SPDX-License-Identifier: MIT

//! Launchers for `kernels/metal/quant_iq.metal`.
//!
//! Mirrors [`super::sycl`]: activations are not quantized to int8 for a
//! `dp4a`-style dot product — both entry points decode weights on the fly and
//! accumulate a plain float dot product. Every [`IQuantType`] has a Metal
//! kernel, including the by-expert-id matvec packed `MoE` experts need.
//! F32 / F16 / BF16 outputs are supported; on older Macs without `bfloat`
//! (Metal < 3.0) the BF16 kernels are absent from the compiled library and
//! the BF16 path returns an error so the caller can fall back to the CPU
//! path.
//!
//! The MSL source is compiled at runtime, once per device, by
//! [`metal_util::pipeline`] — no `.metallib` to ship.

use candle_core::{DType, Result, Tensor};
use candle_metal_kernels::metal::ComputeCommandEncoder;
use objc2_metal::MTLSize;

use crate::ops::metal_util::{self, buffer, float_tag};
use crate::quantized::iquant::IQuantType;

const MSL_SRC: &str = include_str!("../../../kernels/metal/quant_iq.metal");
/// The codebooks, shared with the SYCL kernels (see [`msl_source`]).
const GRIDS_H: &str = include_str!("../../../kernels/sycl/iq_grids.h");

/// Output rows (SIMD-groups of 32) per matvec threadgroup; `ROWS_PER_TG` in
/// the kernel.
const ROWS_PER_TG: usize = 4;

/// Pair count from which [`matvec_indexed`] switches from the by-id matvec
/// (every pair decodes its expert on its own, ids stay on the device) to
/// [`super::indexed_via_gemm`] (one host sync for the ids, then each routed
/// expert is decoded once and multiplied by candle's Metal GEMM). The same
/// crossover as SYCL's; not tuned on Apple GPUs yet.
const GEMM_MIN_PAIRS: usize = 2048;

/// `MatvecParams` in the kernel.
#[repr(C)]
struct MatvecParams {
    expert_stride: u64,
    x_div: i32,
    pairs: i32,
    out_rows: i32,
    cols: i32,
    has_ids: i32,
}

/// `DequantParams` in the kernel.
#[repr(C)]
struct DequantParams {
    expert_stride: u64,
    n_mats: i32,
    n_rows: i32,
    cols: i32,
    has_ids: i32,
}

/// The type part of the kernel names (`IQ_ALL_TYPES` in `quant_iq.metal`).
fn type_tag(ty: IQuantType) -> &'static str {
    match ty {
        IQuantType::Iq4Nl => "iq4_nl",
        IQuantType::Iq4Xs => "iq4_xs",
        IQuantType::Iq2S => "iq2_s",
        IQuantType::Iq3Xxs => "iq3_xxs",
        IQuantType::Iq3S => "iq3_s",
        IQuantType::Q2_0 => "q2_0",
        // `IQuantType::has_native_kernel` keeps k-quants off this backend.
        IQuantType::Q4K | IQuantType::Q5K | IQuantType::Q6K => {
            unreachable!("no Metal kernel for {}", ty.name())
        },
    }
}

fn to_i32(n: usize, what: &str) -> Result<i32> {
    i32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("{what} {n} exceeds i32")))
}

/// `quant_iq.metal` with the SYCL codebook header spliced in for its
/// `#include "iq_grids.h"`: the header's preprocessor lines are dropped and
/// its `inline constexpr uintN_t` tables become `constant` MSL arrays.
fn msl_source() -> String {
    let grids = GRIDS_H
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            l.replace("inline constexpr uint64_t", "constant ulong")
                .replace("inline constexpr uint32_t", "constant uint")
        })
        .collect::<Vec<_>>()
        .join("\n");
    MSL_SRC.replace("#include \"iq_grids.h\"", &grids)
}

/// `input` (`[rows, cols]`, Metal, any float dtype) times the transposed
/// packed weight (`[output_rows, cols]`), returning `[rows, output_rows]` in
/// `out_dtype`.
///
/// Meant for decode-sized `rows`; prefill should use [`dequantize`] and a
/// regular matmul instead (see `IQuantLinear::forward_chunked`).
///
/// # Errors
///
/// Returns an error if the tensors are not on a Metal device, `out_dtype`
/// isn't F32/F16/BF16, the BF16 kernel is missing on older Metal, or the
/// kernel launch fails.
pub fn matvec(
    input: &Tensor,
    packed: &Tensor,
    ty: IQuantType,
    output_rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    let rows = input.elem_count() / cols;
    let out = launch_matvec(
        input,
        packed,
        ty,
        None,
        1,
        rows,
        output_rows,
        cols,
        out_dtype,
    )?;
    let mut dims = input.dims().to_vec();
    *dims.last_mut().unwrap() = output_rows;
    out.reshape(dims)
}

/// `MoE` matmul by expert id: for each of the `ids.len()` pairs `p`, row `p`
/// of the result is expert `ids[p]`'s `[output_rows, cols]` matrix (from the
/// packed `[experts, output_rows, cols]` tensor) times activation row
/// `p / x_div` of `input` (`[_, cols]`). Returns `[ids.len(), output_rows]`.
///
/// Below [`GEMM_MIN_PAIRS`] `ids` (`U32`, same device) is never read on the
/// host.
///
/// # Errors
///
/// Returns an error if the tensors are not on a Metal device, `ids` is not
/// `U32`, `out_dtype` isn't F32/F16/BF16, or the kernel launch fails.
#[allow(clippy::too_many_arguments)]
pub fn matvec_indexed(
    input: &Tensor,
    packed: &Tensor,
    ty: IQuantType,
    ids: &Tensor,
    x_div: usize,
    output_rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    if ids.dtype() != DType::U32 {
        candle_core::bail!("expert ids must be U32, got {:?}", ids.dtype())
    }
    let ids = ids.flatten_all()?.contiguous()?;
    let pairs = ids.elem_count();
    if pairs >= GEMM_MIN_PAIRS {
        return super::indexed_via_gemm(input, &ids, x_div, output_rows, cols, out_dtype, |e| {
            dequantize_experts(packed, ty, e, output_rows, cols, DType::F16)
        });
    }
    launch_matvec(
        input,
        packed,
        ty,
        Some(&ids),
        x_div,
        pairs,
        output_rows,
        cols,
        out_dtype,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_matvec(
    input: &Tensor,
    packed: &Tensor,
    ty: IQuantType,
    ids: Option<&Tensor>,
    x_div: usize,
    pairs: usize,
    output_rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    let dev = input.device().as_metal_device()?.clone();
    let pipeline = metal_util::pipeline(
        &dev,
        "quant_iq",
        msl_source,
        &format!("matvec_{}_{}", type_tag(ty), float_tag(out_dtype)?),
    )?;
    let params = MatvecParams {
        expert_stride: (output_rows * (cols / ty.block_size()) * ty.block_bytes()) as u64,
        x_div: to_i32(x_div.max(1), "x_div")?,
        pairs: to_i32(pairs, "matvec rows")?,
        out_rows: to_i32(output_rows, "matvec output rows")?,
        cols: to_i32(cols, "matvec cols")?,
        has_ids: i32::from(ids.is_some()),
    };

    // Activations are F32 in the kernel (matches the SYCL path).
    let input = input.to_dtype(DType::F32)?.contiguous()?;
    let (input_storage, _) = input.storage_and_layout();
    let (input_buf, input_offset) = buffer(&input_storage, &input, 0, "matvec input")?;
    let (packed_storage, _) = packed.storage_and_layout();
    let (packed_buf, packed_offset) = buffer(&packed_storage, packed, 0, "matvec weight")?;
    let ids_storage = ids.map(Tensor::storage_and_layout);

    let out_el = pairs * output_rows;
    let out_buf = metal_util::output(&dev, out_el, out_dtype, "iq_matvec_out")?;
    {
        let encoder = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_input_buffer(0, Some(packed_buf), packed_offset);
        match (&ids_storage, ids) {
            (Some((storage, _)), Some(ids)) => {
                let (buf, offset) = buffer(storage, ids, 0, "expert ids")?;
                enc.set_input_buffer(1, Some(buf), offset);
            },
            // Unread without ids; any valid buffer will do.
            _ => enc.set_input_buffer(1, Some(packed_buf), packed_offset),
        }
        enc.set_input_buffer(2, Some(input_buf), input_offset);
        enc.set_output_buffer(3, Some(&out_buf), 0);
        enc.set_bytes(4, &params);
        enc.dispatch_thread_groups(
            MTLSize {
                width: output_rows.div_ceil(ROWS_PER_TG),
                height: pairs,
                depth: 1,
            },
            MTLSize {
                width: ROWS_PER_TG * 32,
                height: 1,
                depth: 1,
            },
        );
    }
    drop(input_storage);
    drop(packed_storage);
    drop(ids_storage);

    Ok(metal_util::wrap(
        &dev,
        out_buf,
        (pairs, output_rows),
        out_dtype,
    ))
}

/// Expand weight rows `row_start..row_start + n_rows` of a packed `[_, cols]`
/// tensor into a dense `[n_rows, cols]` tensor of `dtype`.
///
/// # Errors
///
/// Returns an error if `packed` is not on a Metal device, `dtype` isn't
/// F32/F16/BF16, the BF16 kernel is missing on older Metal, or the kernel
/// launch fails.
pub fn dequantize(
    packed: &Tensor,
    ty: IQuantType,
    row_start: usize,
    n_rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Tensor> {
    let row_bytes = cols / ty.block_size() * ty.block_bytes();
    launch_dequant(
        packed,
        row_start * row_bytes,
        ty,
        None,
        1,
        n_rows,
        cols,
        dtype,
    )?
    .reshape((n_rows, cols))
}

/// Decode experts `ids` (`U32`, on the device) of a packed `[experts, rows,
/// cols]` tensor into a dense `[ids.len(), rows, cols]` tensor of `dtype`,
/// in one launch.
///
/// # Errors
///
/// Returns an error if the tensors are not on a Metal device, `dtype` isn't
/// F32/F16/BF16, or the kernel launch fails.
pub fn dequantize_experts(
    packed: &Tensor,
    ty: IQuantType,
    ids: &Tensor,
    rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Tensor> {
    let n = ids.elem_count();
    launch_dequant(packed, 0, ty, Some(ids), n, rows, cols, dtype)?.reshape((n, rows, cols))
}

#[allow(clippy::too_many_arguments)]
fn launch_dequant(
    packed: &Tensor,
    byte_offset: usize,
    ty: IQuantType,
    ids: Option<&Tensor>,
    n_mats: usize,
    n_rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Tensor> {
    let dev = packed.device().as_metal_device()?.clone();
    let pipeline = metal_util::pipeline(
        &dev,
        "quant_iq",
        msl_source,
        &format!("dequant_{}_{}", type_tag(ty), float_tag(dtype)?),
    )?;
    let params = DequantParams {
        expert_stride: (n_rows * (cols / ty.block_size()) * ty.block_bytes()) as u64,
        n_mats: to_i32(n_mats, "dequantize matrices")?,
        n_rows: to_i32(n_rows, "dequantize rows")?,
        cols: to_i32(cols, "dequantize cols")?,
        has_ids: i32::from(ids.is_some()),
    };

    let (packed_storage, _) = packed.storage_and_layout();
    let (packed_buf, packed_offset) =
        buffer(&packed_storage, packed, byte_offset, "dequantize weight")?;
    let ids = ids.map(|t| t.flatten_all()?.contiguous()).transpose()?;
    let ids_storage = ids.as_ref().map(Tensor::storage_and_layout);

    let out_el = n_mats * n_rows * cols;
    let out_buf = metal_util::output(&dev, out_el, dtype, "iq_dequant_out")?;
    {
        let encoder = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_input_buffer(0, Some(packed_buf), packed_offset);
        match (&ids_storage, &ids) {
            (Some((storage, _)), Some(ids)) => {
                let (buf, offset) = buffer(storage, ids, 0, "expert ids")?;
                enc.set_input_buffer(1, Some(buf), offset);
            },
            _ => enc.set_input_buffer(1, Some(packed_buf), packed_offset),
        }
        enc.set_output_buffer(2, Some(&out_buf), 0);
        enc.set_bytes(3, &params);
        enc.dispatch_threads(
            MTLSize {
                width: cols / 32,
                height: n_mats * n_rows,
                depth: 1,
            },
            MTLSize {
                width: 8,
                height: 8,
                depth: 1,
            },
        );
    }
    drop(packed_storage);
    drop(ids_storage);

    Ok(metal_util::wrap(&dev, out_buf, out_el, dtype))
}

#[cfg(test)]
mod tests {
    #[test]
    fn grids_splice_into_msl() {
        let src = super::msl_source();
        assert!(!src.contains("#include \"iq_grids.h\""));
        assert!(!src.contains("#pragma once"));
        assert!(!src.contains("constexpr"));
        for table in [
            "constant ulong iq2s_grid[1024]",
            "constant uint iq3xxs_grid[256]",
            "constant uint iq3s_grid[512]",
        ] {
            assert!(src.contains(table), "{table} missing");
        }
    }
}
