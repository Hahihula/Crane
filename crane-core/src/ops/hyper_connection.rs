// SPDX-License-Identifier: MIT

//! Element-wise pieces of the hyper-connection mixer
//! ([`crate::models::qwen4_exp::hyper_connection`]).
//!
//! The residual is `groups` parallel streams of `group` values in the last
//! dimension. Decode runs the mixer 97 times per token on Qwen3.8-Flash-Next,
//! so each op chain here is one kernel on SYCL (`kernels/sycl/
//! hyper_connection.cpp`), Metal (`kernels/metal/hyper_connection.metal`) and
//! CUDA (`kernels/cuda/hyper_connection.cu`); elsewhere it is the equivalent
//! candle op chain.

#[cfg(feature = "sycl")]
use candle_core::DType;
use candle_core::{D, Result, Tensor};

/// `RMSNorm` over each `group`-wide slice of the last dimension, scaled by
/// that slice's gain (`alpha`: `[groups, group]`).
///
/// # Errors
///
/// Returns an error if the shapes disagree or a kernel fails.
pub fn grouped_rms_norm(x: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let (groups, group) = alpha.dims2()?;
    #[cfg(feature = "sycl")]
    if let Some(dtype) = sycl_dtype(x) {
        let rows = rows_of(x, groups * group)?;
        let x = x.contiguous()?;
        let alpha = alpha.to_dtype(x.dtype())?.contiguous()?;
        return sycl::launch(&x, x.shape().clone(), |q, out| unsafe {
            Ok(sycl::crane_hc_norm_sycl(
                q,
                dtype,
                sycl::ptr(&x)?,
                sycl::ptr(&alpha)?,
                out,
                rows,
                sycl::int(groups)?,
                sycl::int(group)?,
                eps,
            ))
        });
    }
    #[cfg(feature = "cuda")]
    if cuda::supports(x) {
        let x = x.contiguous()?;
        let alpha = alpha.to_dtype(x.dtype())?.contiguous()?;
        return cuda::norm(&x, &alpha, groups, group, eps);
    }
    #[cfg(feature = "metal")]
    if metal::supports(x) {
        let rows = x.elem_count() / (groups * group);
        let x = x.contiguous()?;
        let alpha = alpha.to_dtype(x.dtype())?.contiguous()?;
        let params = metal::HcParams {
            rows: metal::uint(rows)?,
            groups: metal::uint(groups)?,
            group: metal::uint(group)?,
            eps,
            ..metal::HcParams::default()
        };
        return metal::launch(
            "hc_norm",
            &[&x, &alpha],
            x.shape().clone(),
            &params,
            metal::Grid::Groups(rows * groups, metal::NORM_TG),
        );
    }
    let mut grouped = x.dims()[..x.rank() - 1].to_vec();
    grouped.extend([groups, group]);
    let ones = Tensor::ones(group, x.dtype(), x.device())?;
    // candle's fused `rms_norm` accumulates in f32 whatever the dtype.
    candle_nn::ops::rms_norm(&x.reshape(grouped)?.contiguous()?, &ones, eps)?
        .broadcast_mul(&alpha.to_dtype(x.dtype())?)?
        .reshape(x.shape())
}

/// `silu(x * scale)`.
///
/// # Errors
///
/// Returns an error if a kernel fails.
pub fn scaled_silu(x: &Tensor, scale: f64) -> Result<Tensor> {
    #[cfg(feature = "sycl")]
    if let Some(dtype) = sycl_dtype(x) {
        let x = x.contiguous()?;
        #[allow(clippy::cast_possible_truncation)]
        return sycl::launch(&x, x.shape().clone(), |q, out| unsafe {
            Ok(sycl::crane_hc_low_sycl(
                q,
                dtype,
                sycl::ptr(&x)?,
                out,
                x.elem_count(),
                scale as f32,
            ))
        });
    }
    #[cfg(feature = "cuda")]
    if cuda::supports(x) {
        return cuda::low(&x.contiguous()?, scale);
    }
    #[cfg(feature = "metal")]
    if metal::supports(x) {
        let x = x.contiguous()?;
        let n = x.elem_count();
        #[allow(clippy::cast_possible_truncation)]
        let params = metal::HcParams {
            n: metal::uint(n)?,
            scale: scale as f32,
            ..metal::HcParams::default()
        };
        return metal::launch(
            "hc_low",
            &[&x],
            x.shape().clone(),
            &params,
            metal::Grid::Threads([n, 1, 1]),
        );
    }
    candle_nn::ops::silu(&(x * scale)?)
}

/// Collapse the streams: `mean over g of sigmoid(gate[.., g, :]) *
/// normed[.., g, :]`, from `[.., groups * group]` to `[.., group]`.
///
/// # Errors
///
/// Returns an error if the shapes disagree or a kernel fails.
pub fn gated_stream_mean(gate: &Tensor, normed: &Tensor, groups: usize) -> Result<Tensor> {
    let wide = *gate.dims().last().unwrap_or(&0);
    let group = wide / groups;
    let mut out_dims = gate.dims().to_vec();
    if let Some(last) = out_dims.last_mut() {
        *last = group;
    }
    #[cfg(feature = "sycl")]
    if let Some(dtype) = sycl_dtype(gate) {
        let rows = rows_of(gate, wide)?;
        let gate = gate.contiguous()?;
        let normed = normed.to_dtype(gate.dtype())?.contiguous()?;
        return sycl::launch(&gate, out_dims.into(), |q, out| unsafe {
            Ok(sycl::crane_hc_mix_sycl(
                q,
                dtype,
                sycl::ptr(&gate)?,
                sycl::ptr(&normed)?,
                out,
                rows,
                sycl::int(groups)?,
                sycl::int(group)?,
            ))
        });
    }
    #[cfg(feature = "cuda")]
    if cuda::supports(gate) {
        let gate = gate.contiguous()?;
        let normed = normed.to_dtype(gate.dtype())?.contiguous()?;
        return cuda::mix(&gate, &normed, groups, group);
    }
    #[cfg(feature = "metal")]
    if metal::supports(gate) {
        let rows = gate.elem_count() / wide.max(1);
        let gate = gate.contiguous()?;
        let normed = normed.to_dtype(gate.dtype())?.contiguous()?;
        let params = metal::HcParams {
            rows: metal::uint(rows)?,
            groups: metal::uint(groups)?,
            group: metal::uint(group)?,
            ..metal::HcParams::default()
        };
        return metal::launch(
            "hc_mix",
            &[&gate, &normed],
            out_dims.into(),
            &params,
            metal::Grid::Threads([group, rows, 1]),
        );
    }
    let mut split = gate.dims()[..gate.rank() - 1].to_vec();
    split.extend([groups, group]);
    (candle_nn::ops::sigmoid(gate)? * normed)?
        .reshape(split)?
        .mean(D::Minus2)
}

/// Scatter a block output back into the streams: `streams[.., g, :] +
/// block * 2 * sigmoid(logits[.., g] * scale)`, with `streams` `[..,
/// groups * group]`, `block` `[.., group]` and `logits` `[.., groups]`.
///
/// # Errors
///
/// Returns an error if the shapes disagree or a kernel fails.
pub fn gated_combine(
    streams: &Tensor,
    block: &Tensor,
    logits: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    #[cfg(feature = "sycl")]
    if let Some(dtype) = sycl_dtype(streams) {
        let groups = *logits.dims().last().unwrap_or(&0);
        let group = *block.dims().last().unwrap_or(&0);
        let rows = rows_of(streams, groups * group)?;
        let streams = streams.contiguous()?;
        let block = block.to_dtype(streams.dtype())?.contiguous()?;
        let logits = logits.to_dtype(streams.dtype())?.contiguous()?;
        #[allow(clippy::cast_possible_truncation)]
        return sycl::launch(&streams, streams.shape().clone(), |q, out| unsafe {
            Ok(sycl::crane_hc_combine_sycl(
                q,
                dtype,
                sycl::ptr(&streams)?,
                sycl::ptr(&block)?,
                sycl::ptr(&logits)?,
                out,
                rows,
                sycl::int(groups)?,
                sycl::int(group)?,
                scale as f32,
            ))
        });
    }
    #[cfg(feature = "cuda")]
    if cuda::supports(streams) {
        let groups = *logits.dims().last().unwrap_or(&0);
        let group = *block.dims().last().unwrap_or(&0);
        let streams = streams.contiguous()?;
        let block = block.to_dtype(streams.dtype())?.contiguous()?;
        let logits = logits.to_dtype(streams.dtype())?.contiguous()?;
        return cuda::combine(&streams, &block, &logits, groups, group, scale);
    }
    #[cfg(feature = "metal")]
    if metal::supports(streams) {
        let groups = *logits.dims().last().unwrap_or(&0);
        let group = *block.dims().last().unwrap_or(&0);
        let rows = streams.elem_count() / (groups * group).max(1);
        let streams = streams.contiguous()?;
        let block = block.to_dtype(streams.dtype())?.contiguous()?;
        let logits = logits.to_dtype(streams.dtype())?.contiguous()?;
        #[allow(clippy::cast_possible_truncation)]
        let params = metal::HcParams {
            rows: metal::uint(rows)?,
            groups: metal::uint(groups)?,
            group: metal::uint(group)?,
            scale: scale as f32,
            ..metal::HcParams::default()
        };
        return metal::launch(
            "hc_combine",
            &[&streams, &block, &logits],
            streams.shape().clone(),
            &params,
            metal::Grid::Threads([group, groups, rows]),
        );
    }
    let weights = (candle_nn::ops::sigmoid(&(logits * scale)?)? * 2.0)?;
    let injected = block
        .unsqueeze(D::Minus2)?
        .broadcast_mul(&weights.to_dtype(block.dtype())?.unsqueeze(D::Minus1)?)?
        .flatten_from(D::Minus2)?;
    streams + injected.to_dtype(streams.dtype())?
}

/// The kernels' dtype tag, if `x` can take the SYCL path.
#[cfg(feature = "sycl")]
fn sycl_dtype(x: &Tensor) -> Option<i32> {
    if !x.device().is_sycl() {
        return None;
    }
    match x.dtype() {
        DType::F32 => Some(0),
        DType::F16 => Some(1),
        _ => None,
    }
}

#[cfg(feature = "sycl")]
fn rows_of(x: &Tensor, width: usize) -> Result<i32> {
    if width == 0 || x.dims().last() != Some(&width) {
        candle_core::bail!(
            "hyper-connection op: expected width {width}, got {:?}",
            x.dims()
        )
    }
    sycl::int(x.elem_count() / width)
}

#[cfg(feature = "cuda")]
mod cuda {
    //! Launchers for `kernels/cuda/hyper_connection.cu`.

    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaDType, CudaStorage, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{DType, Result, Storage, Tensor};

    mod ptx {
        include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
    }

    const MODULE_NAME: &str = "crane_hyper_connection";
    /// `HC_NORM_THREADS` in the kernel.
    const NORM_THREADS: u32 = 256;
    const THREADS: usize = 256;

    /// Whether `x` can take the CUDA path.
    pub fn supports(x: &Tensor) -> bool {
        x.device().is_cuda() && matches!(x.dtype(), DType::F32 | DType::F16 | DType::BF16)
    }

    fn tag(dtype: DType) -> &'static str {
        match dtype {
            DType::F32 => "f32",
            DType::F16 => "f16",
            _ => "bf16",
        }
    }

    fn int(n: usize) -> Result<i32> {
        i32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("{n} exceeds i32")))
    }

    fn one_d(n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (n.div_ceil(THREADS).max(1) as u32, 1, 1),
            block_dim: (THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Dispatch `$body` with `$t` bound to the element type of `$x`.
    macro_rules! typed {
        ($x:expr, $t:ident => $body:expr) => {
            match $x.dtype() {
                DType::F32 => {
                    type $t = f32;
                    $body
                },
                DType::F16 => {
                    type $t = half::f16;
                    $body
                },
                _ => {
                    type $t = half::bf16;
                    $body
                },
            }
        };
    }

    fn view<'a, T: CudaDType>(
        storage: &'a Storage,
        offset: usize,
    ) -> Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, T>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("hyper-connection kernel input must be on CUDA")
        };
        Ok(cuda.as_cuda_slice::<T>()?.slice(offset..))
    }

    fn wrap<T: CudaDType>(
        buf: candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
        like: &Tensor,
        shape: &[usize],
    ) -> Result<Tensor> {
        let dev = like.device().as_cuda_device()?.clone();
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(buf, dev)),
            shape.to_vec(),
            BackpropOp::none(),
            false,
        ))
    }

    pub fn norm(
        x: &Tensor,
        alpha: &Tensor,
        groups: usize,
        group: usize,
        eps: f32,
    ) -> Result<Tensor> {
        let dev = x.device().as_cuda_device()?.clone();
        let func = dev.get_or_load_custom_func(
            &format!("hc_norm_{}", tag(x.dtype())),
            MODULE_NAME,
            ptx::HYPER_CONNECTION,
        )?;
        let rows_groups = x.elem_count() / group;
        let (groups_i, group_i) = (int(groups)?, int(group)?);
        typed!(x, T => {
            let (xs, xl) = x.storage_and_layout();
            let (als, all) = alpha.storage_and_layout();
            let xv = view::<T>(&xs, xl.start_offset())?;
            let av = view::<T>(&als, all.start_offset())?;
            let out = unsafe { dev.alloc::<T>(x.elem_count()) }?;
            let mut b = func.builder();
            b.arg(&xv).arg(&av).arg(&out).arg(&groups_i).arg(&group_i).arg(&eps);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (int(rows_groups)? as u32, 1, 1),
                    block_dim: (NORM_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .w()?;
            drop((xs, als));
            wrap(out, x, x.dims())
        })
    }

    pub fn low(x: &Tensor, scale: f64) -> Result<Tensor> {
        let dev = x.device().as_cuda_device()?.clone();
        let func = dev.get_or_load_custom_func(
            &format!("hc_low_{}", tag(x.dtype())),
            MODULE_NAME,
            ptx::HYPER_CONNECTION,
        )?;
        let n = x.elem_count();
        #[allow(clippy::cast_possible_truncation)]
        let scale = scale as f32;
        typed!(x, T => {
            let (xs, xl) = x.storage_and_layout();
            let xv = view::<T>(&xs, xl.start_offset())?;
            let out = unsafe { dev.alloc::<T>(n) }?;
            let mut b = func.builder();
            b.arg(&xv).arg(&out).arg(&n).arg(&scale);
            unsafe { b.launch(one_d(n)) }.w()?;
            drop(xs);
            wrap(out, x, x.dims())
        })
    }

    pub fn mix(gate: &Tensor, normed: &Tensor, groups: usize, group: usize) -> Result<Tensor> {
        let dev = gate.device().as_cuda_device()?.clone();
        let func = dev.get_or_load_custom_func(
            &format!("hc_mix_{}", tag(gate.dtype())),
            MODULE_NAME,
            ptx::HYPER_CONNECTION,
        )?;
        let rows = gate.elem_count() / (groups * group).max(1);
        let (groups_i, group_i) = (int(groups)?, int(group)?);
        let mut dims = gate.dims().to_vec();
        if let Some(last) = dims.last_mut() {
            *last = group;
        }
        typed!(gate, T => {
            let (gs, gl) = gate.storage_and_layout();
            let (ns, nl) = normed.storage_and_layout();
            let gv = view::<T>(&gs, gl.start_offset())?;
            let nv = view::<T>(&ns, nl.start_offset())?;
            let out = unsafe { dev.alloc::<T>(rows * group) }?;
            let mut b = func.builder();
            b.arg(&gv).arg(&nv).arg(&out).arg(&rows).arg(&groups_i).arg(&group_i);
            unsafe { b.launch(one_d(rows * group)) }.w()?;
            drop((gs, ns));
            wrap(out, gate, &dims)
        })
    }

    pub fn combine(
        streams: &Tensor,
        block: &Tensor,
        logits: &Tensor,
        groups: usize,
        group: usize,
        scale: f64,
    ) -> Result<Tensor> {
        let dev = streams.device().as_cuda_device()?.clone();
        let func = dev.get_or_load_custom_func(
            &format!("hc_combine_{}", tag(streams.dtype())),
            MODULE_NAME,
            ptx::HYPER_CONNECTION,
        )?;
        let n = streams.elem_count();
        let rows = n / (groups * group).max(1);
        let (groups_i, group_i) = (int(groups)?, int(group)?);
        #[allow(clippy::cast_possible_truncation)]
        let scale = scale as f32;
        typed!(streams, T => {
            let (ss, sl) = streams.storage_and_layout();
            let (bs, bl) = block.storage_and_layout();
            let (ls, ll) = logits.storage_and_layout();
            let sv = view::<T>(&ss, sl.start_offset())?;
            let bv = view::<T>(&bs, bl.start_offset())?;
            let lv = view::<T>(&ls, ll.start_offset())?;
            let out = unsafe { dev.alloc::<T>(n) }?;
            let mut b = func.builder();
            b.arg(&sv).arg(&bv).arg(&lv).arg(&out).arg(&rows).arg(&groups_i).arg(&group_i).arg(&scale);
            unsafe { b.launch(one_d(n)) }.w()?;
            drop((ss, bs, ls));
            wrap(out, streams, streams.dims())
        })
    }
}

#[cfg(feature = "metal")]
mod metal {
    use candle_core::{DType, Result, Shape, Tensor};
    use candle_metal_kernels::metal::ComputeCommandEncoder;
    use objc2_metal::MTLSize;

    use crate::ops::metal_util;

    /// Threads per `hc_norm` threadgroup (`NORM_TG` in the kernel).
    pub const NORM_TG: usize = 256;

    /// `HcParams` in `kernels/metal/hyper_connection.metal`.
    #[repr(C)]
    #[derive(Default)]
    pub struct HcParams {
        pub rows: u32,
        pub groups: u32,
        pub group: u32,
        pub n: u32,
        pub eps: f32,
        pub scale: f32,
    }

    pub enum Grid {
        /// `count` threadgroups of `width` threads.
        Groups(usize, usize),
        /// One thread per grid point.
        Threads([usize; 3]),
    }

    /// Whether `x` can take the Metal path.
    pub fn supports(x: &Tensor) -> bool {
        x.device().is_metal() && matches!(x.dtype(), DType::F32 | DType::F16 | DType::BF16)
    }

    pub fn uint(n: usize) -> Result<u32> {
        u32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("{n} exceeds u32")))
    }

    /// Run `{kernel}_{dtype}` on contiguous `inputs` (buffers `0..`, then
    /// the output, then `params`) into a new tensor of `shape`, typed like
    /// `inputs[0]`.
    pub fn launch(
        kernel: &str,
        inputs: &[&Tensor],
        shape: Shape,
        params: &HcParams,
        grid: Grid,
    ) -> Result<Tensor> {
        let dev = inputs[0].device().as_metal_device()?.clone();
        let dtype = inputs[0].dtype();
        let name = format!("{kernel}_{}", metal_util::float_tag(dtype)?);
        let pipeline = metal_util::pipeline(
            &dev,
            "hyper_connection",
            || include_str!("../../kernels/metal/hyper_connection.metal").to_string(),
            &name,
        )?;
        let n = shape.elem_count();
        let out = metal_util::output(&dev, n, dtype, kernel)?;
        let guards: Vec<_> = inputs.iter().map(|t| t.storage_and_layout().0).collect();
        {
            let encoder = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = encoder.as_ref();
            enc.set_compute_pipeline_state(&pipeline);
            for (i, (guard, t)) in guards.iter().zip(inputs).enumerate() {
                let (buf, offset) = metal_util::buffer(guard, t, 0, kernel)?;
                enc.set_input_buffer(i, Some(buf), offset);
            }
            enc.set_output_buffer(inputs.len(), Some(&out), 0);
            enc.set_bytes(inputs.len() + 1, params);
            let size = |[width, height, depth]: [usize; 3]| MTLSize {
                width,
                height,
                depth,
            };
            match grid {
                Grid::Groups(count, width) => {
                    enc.dispatch_thread_groups(size([count, 1, 1]), size([width, 1, 1]));
                },
                Grid::Threads(dims) => {
                    let width = dims[0].clamp(1, 256);
                    enc.dispatch_threads(size(dims), size([width, 1, 1]));
                },
            }
        }
        drop(guards);
        Ok(metal_util::wrap(&dev, out, shape, dtype))
    }
}

#[cfg(feature = "sycl")]
mod sycl {
    use std::ffi::c_void;

    use candle_core::op::BackpropOp;
    use candle_core::{Result, Shape, Storage, SyclStorage, Tensor};

    // libcrane_gdn_sycl.so — linked by build.rs when `--features sycl`.
    unsafe extern "C" {
        pub fn crane_hc_norm_sycl(
            q: *mut c_void,
            dtype: i32,
            x: *const c_void,
            alpha: *const c_void,
            out: *mut c_void,
            rows: i32,
            groups: i32,
            group: i32,
            eps: f32,
        ) -> i32;
        pub fn crane_hc_low_sycl(
            q: *mut c_void,
            dtype: i32,
            low: *const c_void,
            out: *mut c_void,
            n: usize,
            scale: f32,
        ) -> i32;
        pub fn crane_hc_mix_sycl(
            q: *mut c_void,
            dtype: i32,
            gate: *const c_void,
            normed: *const c_void,
            out: *mut c_void,
            rows: i32,
            groups: i32,
            group: i32,
        ) -> i32;
        pub fn crane_hc_combine_sycl(
            q: *mut c_void,
            dtype: i32,
            streams: *const c_void,
            block: *const c_void,
            logits: *const c_void,
            out: *mut c_void,
            rows: i32,
            groups: i32,
            group: i32,
            scale: f32,
        ) -> i32;
    }

    pub fn int(n: usize) -> Result<i32> {
        i32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("{n} exceeds i32")))
    }

    /// Device pointer to the first element of contiguous `t`.
    pub fn ptr(t: &Tensor) -> Result<*const c_void> {
        let (storage, layout) = t.storage_and_layout();
        match &*storage {
            Storage::Sycl(st) => Ok(unsafe {
                st.buf()
                    .as_ptr()
                    .cast::<u8>()
                    .add(layout.start_offset() * t.dtype().size_in_bytes())
                    .cast::<c_void>()
            }),
            _ => candle_core::bail!("hyper-connection op: expected a sycl tensor"),
        }
    }

    /// Allocate an output like `like` with `shape` and run `kernel` into it.
    pub fn launch(
        like: &Tensor,
        shape: Shape,
        kernel: impl FnOnce(*mut c_void, *mut c_void) -> Result<i32>,
    ) -> Result<Tensor> {
        let dev = like.device().as_sycl_device()?.clone();
        let dtype = like.dtype();
        let n = shape.elem_count();
        let out = dev.alloc_bytes(n * dtype.size_in_bytes())?;
        let status = kernel(dev.queue().native_ptr(), out.as_mut_ptr())?;
        if status != 0 {
            candle_core::bail!("hyper-connection SYCL kernel failed (status {status})");
        }
        Ok(Tensor::from_storage(
            Storage::Sycl(SyclStorage::from_buffer(&dev, out, dtype, n)),
            shape,
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(all(test, any(feature = "sycl", feature = "metal", feature = "cuda")))]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    /// Each SYCL kernel against the portable op chain (run on the CPU).
    #[cfg(feature = "sycl")]
    #[test]
    fn sycl_kernels_match_portable_chains() -> Result<()> {
        if !candle_core::utils::sycl_is_available() {
            return Ok(());
        }
        device_matches_cpu(&Device::new_sycl(0)?)
    }

    /// The CUDA kernels against the portable op chains (run on the CPU).
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_kernels_match_portable_chains() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        device_matches_cpu(&device)
    }

    /// The portable op chains on Metal against the CPU.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_matches_cpu() -> Result<()> {
        if !candle_core::utils::metal_is_available() {
            return Ok(());
        }
        device_matches_cpu(&Device::new_metal(0)?)
    }

    fn device_matches_cpu(device: &Device) -> Result<()> {
        let cpu = Device::Cpu;
        // Flash-Next's 4 streams of 2560, and an odd width.
        for (rows, groups, group) in [(3usize, 4usize, 2560usize), (5, 4, 100)] {
            let wide = groups * group;
            let x = Tensor::randn(0f32, 1.0, (rows, wide), &cpu)?;
            let gate = Tensor::randn(0f32, 1.0, (rows, wide), &cpu)?;
            let alpha = (Tensor::randn(0f32, 0.1, (groups, group), &cpu)? + 1.0)?;
            let block = Tensor::randn(0f32, 1.0, (rows, group), &cpu)?;
            let logits = Tensor::randn(0f32, 1.0, (rows, groups), &cpu)?;
            let low = Tensor::randn(0f32, 1.0, (rows, 320), &cpu)?;

            let mut dtypes = vec![(DType::F32, 1e-5f32), (DType::F16, 5e-3)];
            if device.is_metal() || device.is_cuda() {
                dtypes.push((DType::BF16, 3e-2));
            }
            for (dtype, tol) in dtypes {
                let on = |t: &Tensor| t.to_dtype(dtype)?.to_device(device);
                let check = |got: Tensor, want: Tensor, what: &str| -> Result<()> {
                    let got = got.to_device(&cpu)?.to_dtype(DType::F32)?;
                    let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
                    let diff = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
                    assert!(
                        diff / scale < tol,
                        "{what} {dtype:?} [{rows}, {groups}x{group}]: rel diff {}",
                        diff / scale
                    );
                    Ok(())
                };
                check(
                    super::grouped_rms_norm(&on(&x)?, &on(&alpha)?, 1e-6)?,
                    super::grouped_rms_norm(&x, &alpha, 1e-6)?,
                    "norm",
                )?;
                check(
                    super::scaled_silu(&on(&low)?, 0.25)?,
                    super::scaled_silu(&low, 0.25)?,
                    "silu",
                )?;
                check(
                    super::gated_stream_mean(&on(&gate)?, &on(&x)?, groups)?,
                    super::gated_stream_mean(&gate, &x, groups)?,
                    "mix",
                )?;
                check(
                    super::gated_combine(&on(&x)?, &on(&block)?, &on(&logits)?, 0.25)?,
                    super::gated_combine(&x, &block, &logits, 0.25)?,
                    "combine",
                )?;
            }
        }
        Ok(())
    }
}
