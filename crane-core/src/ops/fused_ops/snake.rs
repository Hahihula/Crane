// SPDX-License-Identifier: MIT
//! Crane Added 20260806: fused Snake periodic activation, reusable across
//! callers (currently the ONNX evaluator's `Snake` op).
//!
//! Implements `snake(x, alpha) = x + sin(alpha * x)^2 / alpha` as a single-pass
//! `CustomOp2` kernel, avoiding the 5-op decomposition (`Mul`, `Sin`, `Pow`,
//! `Mul`, `Add`) that ONNX exporters emit. Each of those ops is a full
//! read-and-write pass over the whole tensor; fusing them into one pass avoids
//! materializing 4 intermediate tensors, which matters once tensors are large
//! enough to blow past cache (see `ONNX_SPEEDUP.md`). `cpu_fwd` is always
//! compiled; `cuda_fwd` is gated behind the `cuda` feature and dispatches to
//! the kernel compiled from `kernels/cuda/snake.cu`, following the
//! `FusedSiluMul` pattern in `cuda_impl.rs`. `rocm_fwd` is gated behind the
//! `rocm` feature and runs the *same* `.cu` source through `hipcc` at
//! runtime (see [`crate::ops::rocm`]). Callers broadcast `x`/`alpha` to
//! matching shapes before calling `snake()`.
//!
//! Metal has no fused kernel here, and `CustomOp2`'s default `metal_fwd`
//! just errors out ("no metal implementation for snake") — so [`snake`]
//! routes Metal around `SnakeOp` entirely, to the portable
//! `x + sin(alpha * x)^2 / alpha` chain built from candle's own
//! `sin`/`powf`/broadcast ops, which already has a real Metal backend. Same
//! pattern as [`super::swiglu`].
//!
//! SYCL has a real fused kernel (`kernels/sycl/fused_ops.cpp`'s
//! `crane_snake_sycl`, built into `libcrane_gdn_sycl.so` by `build.rs`),
//! dispatched directly from [`snake`] rather than through `SnakeOp` — same
//! raw-buffer-pointer FFI pattern as [`super::swiglu`]'s `sycl_kernel`
//! module (SYCL storage has no per-dtype slice enum for `sycl_fwd` to match
//! on, unlike CUDA's `CudaStorageSlice`).

#[cfg(feature = "cuda")]
use candle_core::DType;
#[cfg(feature = "cuda")]
use candle_core::backend::BackendStorage;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
use candle_core::rocm_backend::RocmStorage;
use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor, WithDType};

// PTX compiled from kernels/cuda/snake.cu — embedded at build time.
#[cfg(feature = "cuda")]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

#[cfg(feature = "cuda")]
const MODULE_NAME: &str = "crane_snake";

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
const ROCM_MODULE_NAME: &str = "crane_snake";
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
const ROCM_SOURCE: &str = include_str!("../../../kernels/cuda/snake.cu");

/// SYCL launcher for `crane_snake_sycl` (`kernels/sycl/fused_ops.cpp`).
/// Raw-buffer-pointer FFI, not a `CustomOp2::sycl_fwd` impl — see the module
/// doc comment for why.
#[cfg(feature = "sycl")]
mod sycl_kernel {
    use std::ffi::c_void;

    use candle_core::op::BackpropOp;
    use candle_core::{DType, Result, Storage, SyclStorage, Tensor};

    unsafe extern "C" {
        fn crane_snake_sycl(
            queue: *mut c_void,
            dtype: i32,
            x: *const c_void,
            alpha: *const c_void,
            out: *mut c_void,
            n: i64,
        ) -> i32;
    }

    /// Dtype tags match `crane_snake_sycl`'s `CRANE_FSM_*` enum.
    fn dtype_tag(dtype: DType) -> Result<i32> {
        Ok(match dtype {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::BF16 => 2,
            dt => candle_core::bail!("snake: unsupported dtype {dt:?} on SYCL"),
        })
    }

    pub fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let dtype_tag = dtype_tag(dtype)?;
        let n = x.elem_count();

        // Flat kernel indexes a contiguous buffer; broadcast alpha (e.g. the
        // real model's `[1, C, 1]` vs `x`'s `[1, C, T]`) is compacted first,
        // matching the CUDA/ROCm branch below.
        let x = x.contiguous()?;
        let alpha = alpha.contiguous()?;

        let dev = x.device().as_sycl_device()?.clone();
        let queue = dev.queue().native_ptr();

        let (x_s, x_l) = x.storage_and_layout();
        let (alpha_s, alpha_l) = alpha.storage_and_layout();
        let ptr = |s: &Storage, offset: usize, name: &str| -> Result<*const c_void> {
            match s {
                Storage::Sycl(st) => Ok(unsafe {
                    (st.buf().as_ptr() as *const u8).add(offset * dtype.size_in_bytes())
                        as *const c_void
                }),
                _ => candle_core::bail!("snake: {name} must be a sycl tensor"),
            }
        };
        let x_ptr = ptr(&x_s, x_l.start_offset(), "x")?;
        let alpha_ptr = ptr(&alpha_s, alpha_l.start_offset(), "alpha")?;

        let out_buf = dev.alloc_bytes(n * dtype.size_in_bytes())?;
        let status = unsafe {
            crane_snake_sycl(
                queue,
                dtype_tag,
                x_ptr,
                alpha_ptr,
                out_buf.as_mut_ptr(),
                n as i64,
            )
        };
        if status != 0 {
            candle_core::bail!("crane_snake_sycl failed (status {status})");
        }

        let storage = Storage::Sycl(SyclStorage::from_buffer(&dev, out_buf, dtype, n));
        Ok(Tensor::from_storage(
            storage,
            x_l.shape().clone(),
            BackpropOp::none(),
            false,
        ))
    }
}

/// Fused Snake activation: `x + sin(alpha * x)^2 / alpha`.
struct SnakeOp;

impl CustomOp2 for SnakeOp {
    fn name(&self) -> &'static str {
        "snake"
    }

    fn cpu_fwd(
        &self,
        s_x: &CpuStorage,
        l_x: &Layout,
        s_alpha: &CpuStorage,
        l_alpha: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: WithDType>(
            x: &[T],
            l_x: &Layout,
            alpha: &[T],
            l_alpha: &Layout,
        ) -> (CpuStorage, Shape) {
            let dst =
                candle_core::cpu_backend::binary_map(l_x, l_alpha, x, alpha, |x_val, a_val| {
                    let x = x_val.to_f64();
                    let a = a_val.to_f64();
                    let sin_ax = (a * x).sin();
                    T::from_f64(x + sin_ax * sin_ax / a)
                });
            (T::to_cpu_storage_owned(dst), l_x.shape().clone())
        }

        if l_x.shape() != l_alpha.shape() {
            candle_core::bail!("snake: x and alpha must have the same shape");
        }

        match (s_x, s_alpha) {
            (CpuStorage::BF16(x), CpuStorage::BF16(alpha)) => Ok(inner(x, l_x, alpha, l_alpha)),
            (CpuStorage::F16(x), CpuStorage::F16(alpha)) => Ok(inner(x, l_x, alpha, l_alpha)),
            (CpuStorage::F32(x), CpuStorage::F32(alpha)) => Ok(inner(x, l_x, alpha, l_alpha)),
            (CpuStorage::F64(x), CpuStorage::F64(alpha)) => Ok(inner(x, l_x, alpha, l_alpha)),
            _ => candle_core::bail!("unsupported or mismatched dtypes for Snake"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_x: &CudaStorage,
        l_x: &Layout,
        s_alpha: &CudaStorage,
        l_alpha: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = s_x.device();
        let n = l_x.shape().elem_count();

        let (xo1, xo2) = l_x
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("snake: x must be contiguous".into()))?;
        let (ao1, ao2) = l_alpha
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("snake: alpha must be contiguous".into()))?;
        if xo2 - xo1 != n || ao2 - ao1 != n {
            candle_core::bail!("snake: x and alpha must have the same element count");
        }

        let fn_name = match s_x.dtype() {
            DType::BF16 => "snake_bf16",
            DType::F16 => "snake_f16",
            DType::F32 => "snake_f32",
            dt => candle_core::bail!("snake: unsupported dtype {dt:?}"),
        };
        let func = dev.get_or_load_custom_func(fn_name, MODULE_NAME, ptx::SNAKE)?;

        let n_u32 = n as u32;
        let block_size = 256u32;
        let grid_size = n_u32.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let slice = match (&s_x.slice, &s_alpha.slice) {
            (CudaStorageSlice::BF16(x), CudaStorageSlice::BF16(alpha)) => {
                let x = x.slice(xo1..xo2);
                let alpha = alpha.slice(ao1..ao2);
                let dst = unsafe { dev.alloc::<half::bf16>(n)? };
                let mut builder = func.builder();
                builder.arg(&x);
                builder.arg(&alpha);
                builder.arg(&dst);
                builder.arg(&n_u32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::BF16(dst)
            },
            (CudaStorageSlice::F16(x), CudaStorageSlice::F16(alpha)) => {
                let x = x.slice(xo1..xo2);
                let alpha = alpha.slice(ao1..ao2);
                let dst = unsafe { dev.alloc::<half::f16>(n)? };
                let mut builder = func.builder();
                builder.arg(&x);
                builder.arg(&alpha);
                builder.arg(&dst);
                builder.arg(&n_u32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F16(dst)
            },
            (CudaStorageSlice::F32(x), CudaStorageSlice::F32(alpha)) => {
                let x = x.slice(xo1..xo2);
                let alpha = alpha.slice(ao1..ao2);
                let dst = unsafe { dev.alloc::<f32>(n)? };
                let mut builder = func.builder();
                builder.arg(&x);
                builder.arg(&alpha);
                builder.arg(&dst);
                builder.arg(&n_u32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F32(dst)
            },
            _ => candle_core::bail!("snake: unsupported or mismatched CUDA storage types"),
        };

        let dst = CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l_x.shape().clone()))
    }

    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    fn rocm_fwd(
        &self,
        s_x: &RocmStorage,
        l_x: &Layout,
        s_alpha: &RocmStorage,
        l_alpha: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        // SAFETY: snake_{bf16,f16,f32} in ROCM_SOURCE take (const T*, const
        // T*, T*, uint32_t) for dtype T, matching binary_elementwise_fwd's
        // contract.
        unsafe {
            crate::ops::rocm::binary_elementwise_fwd(
                s_x,
                l_x,
                s_alpha,
                l_alpha,
                ROCM_MODULE_NAME,
                "snake",
                ROCM_SOURCE,
            )
        }
    }
}

/// Fused `Snake` activation: `x + sin(alpha * x)^2 / alpha`.
///
/// Computes the periodic activation used by BigVGAN-family vocoder decoders
/// (see Liu et al.) in a single pass over the data, rather than through the
/// decomposed `Mul(alpha,x) -> Sin -> Pow(_,2) -> Mul(1/alpha,_) -> Add(x,_)`
/// chain that ONNX exporters emit. `x` and `alpha` must already have matching
/// shapes (broadcast by the caller); both are made contiguous here so the
/// CUDA kernel can index them as flat buffers. At `alpha == 0` this produces
/// `NaN` (`sin(0)^2 / 0 = 0 / 0`), identical to the naive decomposition's
/// `Div`-by-zero — not a fusion-specific bug.
///
/// # Errors
///
/// Returns an error if `x`/`alpha` have a dtype other than `BF16`/`F16`/
/// `F32`/`F64` (`cpu_fwd`), or — on CUDA — other than `BF16`/`F16`/`F32`,
/// or if either input's element count doesn't match after broadcasting.
pub fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "sycl")]
    if x.device().is_sycl() {
        return sycl_kernel::snake(x, alpha);
    }
    // No fused kernel on Metal — `SnakeOp` only implements
    // cpu_fwd/cuda_fwd/rocm_fwd, so routing this device through
    // `apply_op2_no_bwd` would hit `CustomOp2`'s default `metal_fwd`, which
    // unconditionally errors.
    if x.device().is_metal() {
        return portable_snake(x, alpha);
    }
    let x = x.contiguous()?;
    let alpha = alpha.contiguous()?;
    x.apply_op2_no_bwd(&alpha, &SnakeOp)
}

/// `x + sin(alpha * x)^2 / alpha` via candle's own ops, used on devices
/// `SnakeOp` has no fused kernel for (Metal, SYCL).
fn portable_snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let sin_ax = x.broadcast_mul(alpha)?.sin()?;
    x.broadcast_add(&sin_ax.powf(2.0)?.broadcast_div(alpha)?)
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::{portable_snake, snake};

    /// Naive reference: the 5-op decomposition the ONNX exporter emits.
    fn naive_snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
        let sin_sq = x.broadcast_mul(alpha)?.sin()?.powf(2.0)?;
        x.broadcast_add(&sin_sq.broadcast_div(alpha)?)
    }

    // Verifies the fused kernel matches the naive 5-op decomposition for a
    // handful of representative 1-D f32 values.
    #[test]
    fn snake_matches_naive_1d_f32() -> Result<()> {
        let x = Tensor::new(&[0.0f32, 1.0, -1.0, 2.5, -0.5], &Device::Cpu)?;
        let alpha = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 1.5], &Device::Cpu)?;

        let got = snake(&x, &alpha)?.to_vec1::<f32>()?;
        let expected = naive_snake(&x, &alpha)?.to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6);
        }
        Ok(())
    }

    // Verifies the fused kernel matches the naive decomposition under the
    // real model's broadcast shape: alpha [1, C, 1] against x [1, C, T].
    #[test]
    fn snake_broadcast_shape() -> Result<()> {
        let device = Device::Cpu;
        let alpha = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?.reshape((1, 4, 1))?;
        let x_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.6).collect();
        let x = Tensor::new(x_data.as_slice(), &device)?.reshape((1, 4, 8))?;

        let shape = x.shape();
        let alpha_b = alpha.broadcast_as(shape)?;

        let got = snake(&x, &alpha_b)?.flatten_all()?.to_vec1::<f32>()?;
        let expected = naive_snake(&x, &alpha)?.flatten_all()?.to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5);
        }
        Ok(())
    }

    // Verifies the F64 dtype branch is exercised and matches the naive
    // decomposition.
    #[test]
    fn snake_f64() -> Result<()> {
        let x = Tensor::new(&[0.0f64, 1.0, -1.0, 2.5], &Device::Cpu)?;
        let alpha = Tensor::new(&[1.0f64, 2.0, 0.5, 3.0], &Device::Cpu)?;

        let got = snake(&x, &alpha)?.to_vec1::<f64>()?;
        let expected = naive_snake(&x, &alpha)?.to_vec1::<f64>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < f64::EPSILON * 100.0);
        }
        Ok(())
    }

    // Verifies that an unsupported dtype returns an error rather than
    // panicking.
    #[test]
    fn snake_unsupported_dtype_errors() -> Result<()> {
        let x = Tensor::new(&[1u32, 2, 3], &Device::Cpu)?;
        let alpha = Tensor::new(&[1u32, 2, 3], &Device::Cpu)?;

        let err = snake(&x, &alpha).expect_err("unsupported dtype must error");

        assert!(
            err.to_string().contains("unsupported"),
            "unexpected error message: {err}"
        );
        Ok(())
    }

    // Degenerate case: as alpha approaches zero, sin(a*x)^2/a approaches
    // a*x^2 (by the small-angle limit), so the result must stay finite
    // rather than blowing up to NaN/Inf.
    #[test]
    fn snake_alpha_near_zero() -> Result<()> {
        let x = Tensor::new(&[1.0f32], &Device::Cpu)?;
        let alpha = Tensor::new(&[1e-10f32], &Device::Cpu)?;

        let got = snake(&x, &alpha)?.to_vec1::<f32>()?;

        assert!(got[0].is_finite());
        assert!((got[0] - 1.0).abs() < 1e-6);
        Ok(())
    }

    // Spec-correct behavior: NaN inputs produce NaN outputs, non-NaN lanes
    // are unaffected.
    #[test]
    fn snake_nan_passthrough() -> Result<()> {
        let x = Tensor::new(&[f32::NAN, 1.0], &Device::Cpu)?;
        let alpha = Tensor::new(&[1.0f32, 1.0], &Device::Cpu)?;

        let got = snake(&x, &alpha)?.to_vec1::<f32>()?;
        let expected = naive_snake(&x, &alpha)?.to_vec1::<f32>()?;

        assert!(got[0].is_nan());
        assert!((got[1] - expected[1]).abs() < 1e-6);
        Ok(())
    }

    // Boundary case: alpha == 0 divides by zero (sin(0)^2 / 0 = 0 / 0),
    // producing NaN identically to the naive decomposition's Div — not a
    // fusion-specific bug.
    #[test]
    fn snake_alpha_zero_produces_nan() -> Result<()> {
        let x = Tensor::new(&[1.0f32], &Device::Cpu)?;
        let alpha = Tensor::new(&[0.0f32], &Device::Cpu)?;

        let got = snake(&x, &alpha)?.to_vec1::<f32>()?;
        let expected = naive_snake(&x, &alpha)?.to_vec1::<f32>()?;

        assert!(got[0].is_nan());
        assert!(expected[0].is_nan());
        Ok(())
    }

    // Verifies the BF16 dtype branch is exercised and matches the naive
    // decomposition within BF16's precision.
    #[test]
    fn snake_bf16() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.0f32, 1.0, -1.0, 2.5], &device)?.to_dtype(DType::BF16)?;
        let alpha = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?.to_dtype(DType::BF16)?;

        let got = snake(&x, &alpha)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let expected = naive_snake(&x, &alpha)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-2, "got {g}, expected {e}");
        }
        Ok(())
    }

    // Verifies the F16 dtype branch is exercised and matches the naive
    // decomposition within F16's precision.
    #[test]
    fn snake_f16() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.0f32, 1.0, -1.0, 2.5], &device)?.to_dtype(DType::F16)?;
        let alpha = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?.to_dtype(DType::F16)?;

        let got = snake(&x, &alpha)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let expected = naive_snake(&x, &alpha)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-3, "got {g}, expected {e}");
        }
        Ok(())
    }

    // Verifies the Metal/SYCL fallback path (exercised directly here, since
    // this CPU-only test suite can't reach it through `snake()`'s device
    // check) matches the fused kernel / naive reference.
    #[test]
    fn portable_snake_matches_naive() -> Result<()> {
        let x = Tensor::new(&[0.0f32, 1.0, -1.0, 2.5, -0.5], &Device::Cpu)?;
        let alpha = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 1.5], &Device::Cpu)?;

        let got = portable_snake(&x, &alpha)?.to_vec1::<f32>()?;
        let expected = naive_snake(&x, &alpha)?.to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
        Ok(())
    }
}
