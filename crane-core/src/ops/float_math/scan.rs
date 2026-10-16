//! Inclusive F32 prefix scan, avoiding triangular-matrix multiplication.
use candle_core::{DType, Result, Tensor};
pub fn cumsum_last_dim(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != DType::F32 {
        candle_core::bail!("F32 prefix scan requires F32 input");
    }
    let axis = x
        .rank()
        .checked_sub(1)
        .ok_or_else(|| candle_core::Error::Msg("prefix scan needs a dimension".into()))?;
    if x.elem_count() == 0 {
        return Ok(x.clone());
    }
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        return cuda_scan(x);
    }
    x.cumsum(axis)
}
#[cfg(feature = "cuda")]
fn cuda_scan(x: &Tensor) -> Result<Tensor> {
    use candle_core::cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    };
    use candle_core::{CudaStorage, Storage, op::BackpropOp};
    let x = x.contiguous()?;
    let count = u32::try_from(x.elem_count()).map_err(candle_core::Error::wrap)?;
    let width = u32::try_from(x.dim(candle_core::D::Minus1)?).map_err(candle_core::Error::wrap)?;
    let rows = count / width;
    let log_width = 32 - (width - 1).leading_zeros();
    let log_rows = 32 - (rows - 1).leading_zeros();
    let log_threads = ((9 + log_width as i32 - log_rows as i32) / 2).clamp(4, 9) as u32;
    let tx = 1u32 << log_threads;
    let ty = 512 / tx;
    let dev = x.device().as_cuda_device()?.clone();
    let (storage, layout) = x.storage_and_layout();
    let slice = match &*storage {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => unreachable!(),
    };
    let input = slice.slice(layout.start_offset()..);
    let out = unsafe { dev.alloc::<f32>(count as usize) }?;
    let function = dev.get_or_load_custom_func(
        "prefix_scan_f32",
        "crane_float_math",
        super::ptx::FLOAT_MATH,
    )?;
    let mut builder = function.builder();
    builder.arg(&input);
    builder.arg(&out);
    builder.arg(&rows);
    builder.arg(&width);
    builder.arg(&log_threads);
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(ty), 1, 1),
        block_dim: (tx, ty, 1),
        shared_mem_bytes: 4096,
    };
    // SAFETY: rows*width contiguous F32 values and 2*512 shared elements.
    unsafe { builder.launch(config) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)),
        x.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    fn check(device: &Device) -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., -4., 8., -2.], [5., -5., 1., 0., 3.]], device)?;
        assert_eq!(
            cumsum_last_dim(&x)?.to_vec2::<f32>()?,
            vec![vec![1., 3., -1., 7., 5.], vec![5., 0., 1., 1., 4.]]
        );
        Ok(())
    }
    #[test]
    fn cpu_prefix_values() -> Result<()> {
        check(&Device::Cpu)
    }
    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_prefix_values() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        check(&device)
    }
}
