// SPDX-License-Identifier: MIT

//! Launchers for `kernels/sycl/quant_iq.cpp`.
//!
//! Unlike the CUDA path ([`super::cuda`]), there is no int8-activation fast
//! path (no portable `dp4a` equivalent across Intel GPU generations) — both
//! entry points decode weights on the fly and accumulate a plain float dot
//! product. Every [`IQuantType`] has a SYCL kernel, including the lower-bit
//! types that CUDA/Metal still re-quantize. Only F32/F16 outputs are
//! supported, matching the rest of the SYCL fused-op surface
//! (`ops/fused_ops/sycl_impl.rs`); a BF16 request falls back to
//! [`crate::quantized::iquant::IQuantLinear`]'s generic CPU dequantize path.

use std::ffi::c_void;

use candle_core::op::BackpropOp;
use candle_core::{DType, Result, Storage, SyclStorage, Tensor};

use crate::quantized::iquant::IQuantType;

// libcrane_gdn_sycl.so — linked by build.rs when `--features sycl`.
unsafe extern "C" {
    fn crane_iq_matvec_sycl(
        queue: *mut c_void,
        ty: i32,
        dtype: i32,
        in_dtype: i32,
        packed: *const c_void,
        expert_stride: usize,
        ids: *const c_void,
        x_div: i32,
        input: *const c_void,
        output: *mut c_void,
        pairs: i32,
        out_rows: i32,
        cols: i32,
    ) -> i32;

    fn crane_iq_dequant_sycl(
        queue: *mut c_void,
        ty: i32,
        dtype: i32,
        packed: *const c_void,
        expert_stride: usize,
        ids: *const c_void,
        n_mats: i32,
        output: *mut c_void,
        n_rows: i32,
        cols: i32,
    ) -> i32;
}

/// The kernels' type tag (`enum Ty` in `quant_iq.cpp`).
fn type_tag(ty: IQuantType) -> i32 {
    match ty {
        IQuantType::Iq4Nl => 0,
        IQuantType::Iq4Xs => 1,
        IQuantType::Iq2S => 2,
        IQuantType::Iq3Xxs => 3,
        IQuantType::Iq3S => 4,
        IQuantType::Q2_0 => 5,
        IQuantType::Q4K => 6,
        IQuantType::Q5K => 7,
        IQuantType::Q6K => 8,
    }
}

fn dtype_tag(dtype: DType) -> Result<i32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        other => candle_core::bail!("i-quant SYCL kernels do not support {other:?} output"),
    }
}

fn byte_ptr(storage: &Storage, byte_offset: usize, name: &str) -> Result<*const c_void> {
    match storage {
        Storage::Sycl(st) => {
            Ok(unsafe { (st.buf().as_ptr() as *const u8).add(byte_offset) as *const c_void })
        },
        _ => candle_core::bail!("i-quant SYCL kernel: {name} must be a sycl tensor"),
    }
}

fn to_i32(n: usize, what: &str) -> Result<i32> {
    i32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("{what} {n} exceeds i32")))
}

/// `input` (`[rows, cols]`, SYCL, any float dtype) times the transposed
/// packed weight (`[output_rows, cols]`), returning `[rows, output_rows]` in
/// `out_dtype`.
///
/// Meant for decode-sized `rows`; prefill should use [`dequantize`] and a
/// regular matmul instead (see `IQuantLinear::forward_chunked`).
///
/// # Errors
///
/// Returns an error if the tensors are not on a SYCL device, `out_dtype`
/// isn't F32/F16, or the kernel launch fails.
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
/// `ids` is `U32` on the same device and never read on the host. Prefill-sized
/// routings take [`super::GemmPlan`] instead (see `IQuantExperts`).
///
/// # Errors
///
/// Returns an error if the tensors are not on a SYCL device, `ids` is not
/// `U32`, `out_dtype` isn't F32/F16, or the kernel launch fails.
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
    let dev = input.device().as_sycl_device()?.clone();
    let queue = dev.queue().native_ptr();
    let dtype = dtype_tag(out_dtype)?;
    // F16 activations go in as they are; anything else is widened to F32.
    let in_dtype = if input.dtype() == DType::F16 {
        DType::F16
    } else {
        DType::F32
    };
    let input = input.to_dtype(in_dtype)?.contiguous()?;
    let expert_stride = output_rows * (cols / ty.block_size()) * ty.block_bytes();

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_ptr = byte_ptr(
        &input_storage,
        input_layout.start_offset() * in_dtype.size_in_bytes(),
        "matvec input",
    )?;
    let (packed_storage, packed_layout) = packed.storage_and_layout();
    let packed_ptr = byte_ptr(
        &packed_storage,
        packed_layout.start_offset(),
        "matvec weight",
    )?;
    let ids_storage = ids.map(Tensor::storage_and_layout);
    let ids_ptr = match &ids_storage {
        Some((storage, layout)) => byte_ptr(
            storage,
            layout.start_offset() * DType::U32.size_in_bytes(),
            "expert ids",
        )?,
        None => std::ptr::null(),
    };

    let out_el = pairs * output_rows;
    let out_buf = dev.alloc_bytes(out_el * out_dtype.size_in_bytes())?;

    let status = unsafe {
        crane_iq_matvec_sycl(
            queue,
            type_tag(ty),
            dtype,
            dtype_tag(in_dtype)?,
            packed_ptr,
            expert_stride,
            ids_ptr,
            to_i32(x_div.max(1), "x_div")?,
            input_ptr,
            out_buf.as_mut_ptr(),
            to_i32(pairs, "matvec rows")?,
            to_i32(output_rows, "matvec output rows")?,
            to_i32(cols, "matvec cols")?,
        )
    };
    drop(input_storage);
    drop(packed_storage);
    drop(ids_storage);
    if status != 0 {
        candle_core::bail!("crane_iq_matvec_sycl failed (status {status})");
    }

    Ok(Tensor::from_storage(
        Storage::Sycl(SyclStorage::from_buffer(&dev, out_buf, out_dtype, out_el)),
        (pairs, output_rows),
        BackpropOp::none(),
        false,
    ))
}

/// Expand weight rows `row_start..row_start + n_rows` of a packed `[_, cols]`
/// tensor into a dense `[n_rows, cols]` tensor of `dtype`.
///
/// # Errors
///
/// Returns an error if `packed` is not on a SYCL device, `dtype` isn't
/// F32/F16, or the kernel launch fails.
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
/// Returns an error if the tensors are not on a SYCL device, `dtype` isn't
/// F32/F16, or the kernel launch fails.
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
    let dev = packed.device().as_sycl_device()?.clone();
    let queue = dev.queue().native_ptr();
    let dtype_i = dtype_tag(dtype)?;
    let expert_stride = n_rows * (cols / ty.block_size()) * ty.block_bytes();

    let (packed_storage, packed_layout) = packed.storage_and_layout();
    let packed_ptr = byte_ptr(
        &packed_storage,
        packed_layout.start_offset() + byte_offset,
        "dequantize weight",
    )?;
    let ids = ids.map(|t| t.flatten_all()?.contiguous()).transpose()?;
    let ids_storage = ids.as_ref().map(Tensor::storage_and_layout);
    let ids_ptr = match &ids_storage {
        Some((storage, layout)) => byte_ptr(
            storage,
            layout.start_offset() * DType::U32.size_in_bytes(),
            "expert ids",
        )?,
        None => std::ptr::null(),
    };

    let out_el = n_mats * n_rows * cols;
    let out_buf = dev.alloc_bytes(out_el * dtype.size_in_bytes())?;
    let status = unsafe {
        crane_iq_dequant_sycl(
            queue,
            type_tag(ty),
            dtype_i,
            packed_ptr,
            expert_stride,
            ids_ptr,
            to_i32(n_mats, "dequantize matrices")?,
            out_buf.as_mut_ptr(),
            to_i32(n_rows, "dequantize rows")?,
            to_i32(cols, "dequantize cols")?,
        )
    };
    drop(packed_storage);
    drop(ids_storage);
    if status != 0 {
        candle_core::bail!("crane_iq_dequant_sycl failed (status {status})");
    }
    Ok(Tensor::from_storage(
        Storage::Sycl(SyclStorage::from_buffer(&dev, out_buf, dtype, out_el)),
        out_el,
        BackpropOp::none(),
        false,
    ))
}
