// SPDX-License-Identifier: MIT
//! Fused `SwiGLU` activation: `silu(gate) * up` in a single pass.
//!
//! Replaces the two-op `Silu.forward(&gate)? * up` chain with one kernel
//! launch, eliminating the intermediate tensor that `Silu` allocates.
//! `cpu_fwd` is always compiled; `cuda_fwd` is gated behind the `cuda`
//! feature and dispatches to the kernel compiled from
//! `kernels/cuda/swiglu.cu`, following the [`super::snake`] pattern.
//! `rocm_fwd` is gated behind the `rocm` feature and runs the *same*
//! `.cu` source through `hipcc` at runtime (see [`crate::ops::rocm`]).
//!
//! `metal_fwd` is gated behind the `metal` feature and runs
//! `kernels/metal/fused_ops.metal`'s `crane_swiglu_*` kernels (one per
//! supported dtype). The .metal source is JIT-compiled via
//! `candle_metal_kernels::metal::Device::new_library_with_source(...)`
//! on first dispatch and cached in a per-`(device, kernel)` `OnceLock`
//! keyed on `MetalDevice.registry_id()`.
//!
//! SYCL has a real fused kernel (`kernels/sycl/fused_ops.cpp`'s
//! `crane_swiglu_sycl`, built into `libcrane_gdn_sycl.so` by `build.rs`),
//! dispatched directly from [`swiglu`] rather than through `SwigluOp` (SYCL
//! storage is an untyped device buffer + `DType` tag, not the per-dtype
//! `CudaStorageSlice`-style enum `CustomOp2::sycl_fwd` would need to match
//! on) — same raw-buffer-pointer pattern as `sycl_impl::fused_silu_mul` and
//! `gdn::sycl_backend::gdn_recurrence_sycl`.

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

#[cfg(feature = "cuda")]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

#[cfg(feature = "cuda")]
const MODULE_NAME: &str = "crane_swiglu";

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
const ROCM_MODULE_NAME: &str = "crane_swiglu";
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
const ROCM_SOURCE: &str = include_str!("../../../kernels/cuda/swiglu.cu");

/// SYCL launcher for `crane_swiglu_sycl` (`kernels/sycl/fused_ops.cpp`).
/// Raw-buffer-pointer FFI, not a `CustomOp2::sycl_fwd` impl — see the module
/// doc comment for why.
#[cfg(feature = "sycl")]
mod sycl_kernel {
    use std::ffi::c_void;

    use candle_core::op::BackpropOp;
    use candle_core::{DType, Result, Storage, SyclStorage, Tensor};

    unsafe extern "C" {
        fn crane_swiglu_sycl(
            queue: *mut c_void,
            dtype: i32,
            gate: *const c_void,
            up: *const c_void,
            out: *mut c_void,
            n: i64,
        ) -> i32;
    }

    /// Dtype tags match `crane_swiglu_sycl`'s `CRANE_FSM_*` enum.
    fn dtype_tag(dtype: DType) -> Result<i32> {
        Ok(match dtype {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::BF16 => 2,
            dt => candle_core::bail!("swiglu: unsupported dtype {dt:?} on SYCL"),
        })
    }

    pub fn swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        let dtype = gate.dtype();
        let dtype_tag = dtype_tag(dtype)?;
        let n = gate.elem_count();

        // Flat kernel indexes a contiguous buffer; broadcasting/narrow'd
        // views are compacted first, matching the CUDA/ROCm branch below.
        let gate = gate.contiguous()?;
        let up = up.contiguous()?;

        let dev = gate.device().as_sycl_device()?.clone();
        let queue = dev.queue().native_ptr();

        let (gate_s, gate_l) = gate.storage_and_layout();
        let (up_s, up_l) = up.storage_and_layout();
        let ptr = |s: &Storage, offset: usize, name: &str| -> Result<*const c_void> {
            match s {
                Storage::Sycl(st) => Ok(unsafe {
                    (st.buf().as_ptr() as *const u8).add(offset * dtype.size_in_bytes())
                        as *const c_void
                }),
                _ => candle_core::bail!("swiglu: {name} must be a sycl tensor"),
            }
        };
        let gate_ptr = ptr(&gate_s, gate_l.start_offset(), "gate")?;
        let up_ptr = ptr(&up_s, up_l.start_offset(), "up")?;

        let out_buf = dev.alloc_bytes(n * dtype.size_in_bytes())?;
        let status = unsafe {
            crane_swiglu_sycl(
                queue,
                dtype_tag,
                gate_ptr,
                up_ptr,
                out_buf.as_mut_ptr(),
                n as i64,
            )
        };
        if status != 0 {
            candle_core::bail!("crane_swiglu_sycl failed (status {status})");
        }

        let storage = Storage::Sycl(SyclStorage::from_buffer(&dev, out_buf, dtype, n));
        Ok(Tensor::from_storage(
            storage,
            gate_l.shape().clone(),
            BackpropOp::none(),
            false,
        ))
    }
}

/// Metal launcher for `crane_swiglu_{f32,f16,bf16}`
/// (`kernels/metal/fused_ops.metal`). The same source also contains
/// `crane_snake_*`, which is dispatched by [`super::snake::metal_kernel`].
///
/// Raw-buffer-pointer FFI through candle-core's `MetalDevice`, not a
/// `CustomOp2::metal_fwd` impl — same direct-dispatch pattern as the SYCL
/// kernel above. The `.metal` source is JIT-compiled into an `MTLLibrary`
/// once per `MetalDevice` (keyed on `registry_id()`) and cached in a
/// process-global `OnceLock`-guarded `HashMap`. Per-kernel-name
/// `MTLComputePipelineState` is cached the same way, so the cost of the
/// first dispatch on each dtype is one Metal shader compile + one pipeline
/// build, and subsequent dispatches are pure buffer-binding + dispatch.
#[cfg(feature = "metal")]
mod metal_kernel {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use candle_core::backend::BackendStorage;
    use candle_core::metal_backend::{MetalDevice, MetalStorage};
    use candle_core::op::BackpropOp;
    use candle_core::{DType, Result, Storage, Tensor};
    use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline, Library};

    /// Compiled-in Metal source, see `kernels/metal/fused_ops.metal`.
    pub(super) const SOURCE: &str = include_str!("../../../kernels/metal/fused_ops.metal");

    /// Per-device compiled library. One `MTLLibrary` per `registry_id`,
    /// shared across all kernel names on that device — Metal's compiler
    /// caches the parsed / SPIR-V-like intermediate, so reusing one
    /// library is significantly cheaper than one library per kernel.
    static LIBRARIES: OnceLock<Mutex<HashMap<u64, Arc<Library>>>> = OnceLock::new();

    /// Per-(device, kernel) compiled pipeline. Built lazily on first
    /// dispatch; never invalidated (the kernel binary is determined
    /// entirely by kernel source + device, neither of which changes).
    static PIPELINES: OnceLock<Mutex<HashMap<(u64, &'static str), Arc<ComputePipeline>>>> =
        OnceLock::new();

    /// Compile `SOURCE` once per device, return the cached `Library`.
    fn get_library(device: &MetalDevice) -> Result<Arc<Library>> {
        let registry_id = device.registry_id();
        let mtx = LIBRARIES.get_or_init(|| Mutex::new(HashMap::new()));
        {
            let cache = mtx.lock().expect("metal library cache poisoned");
            if let Some(lib) = cache.get(&registry_id) {
                return Ok(Arc::clone(lib));
            }
        }
        let lib = device
            .device()
            .new_library_with_source(SOURCE, None)
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "swiglu: failed to compile metal `fused_ops.metal`: {e}"
                ))
            })?;
        let arc = Arc::new(lib);
        mtx.lock()
            .expect("metal library cache poisoned")
            .insert(registry_id, Arc::clone(&arc));
        Ok(arc)
    }

    /// Get or compile the pipeline for `(device, kernel_name)`.
    fn get_pipeline(
        device: &MetalDevice,
        kernel_name: &'static str,
    ) -> Result<Arc<ComputePipeline>> {
        let registry_id = device.registry_id();
        let mtx = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
        {
            let cache = mtx.lock().expect("metal pipeline cache poisoned");
            if let Some(p) = cache.get(&(registry_id, kernel_name)) {
                return Ok(Arc::clone(p));
            }
        }
        let lib = get_library(device)?;
        let func = lib.get_function(kernel_name, None).map_err(|e| {
            candle_core::Error::Msg(format!(
                "swiglu: missing kernel `{kernel_name}` in compiled Metal library: {e}"
            ))
        })?;
        let pipeline = device
            .device()
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "swiglu: pipeline build for `{kernel_name}` failed: {e}"
                ))
            })?;
        let arc = Arc::new(pipeline);
        mtx.lock()
            .expect("metal pipeline cache poisoned")
            .insert((registry_id, kernel_name), Arc::clone(&arc));
        Ok(arc)
    }

    /// `(kernel_name, dtype_size)` lookup for the dtypes `crane_swiglu_*`
    /// supports.
    fn dtype_kernel(dtype: DType) -> Option<(&'static str, usize)> {
        match dtype {
            DType::F32 => Some(("crane_swiglu_f32", 4)),
            DType::F16 => Some(("crane_swiglu_f16", 2)),
            DType::BF16 => Some(("crane_swiglu_bf16", 2)),
            _ => None,
        }
    }

    /// Compute (threadgroups, threads_per_threadgroup) for a flat one-D
    /// dispatch with `n` total elements and `pipeline.max_total_threads_per_threadgroup()`
    /// as the upper bound on `width`.
    fn dispatch_dims(
        n: usize,
        pipeline: &ComputePipeline,
    ) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
        let width = pipeline
            .max_total_threads_per_threadgroup()
            .max(1)
            .min(n.max(1));
        let count = n.div_ceil(width).max(1);
        (
            objc2_metal::MTLSize {
                width: count,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        )
    }

    /// Fused `silu(gate) * up` on Metal. Mirrors the SYCL
    /// [`sycl_kernel::swiglu`] contract: `gate`/`up` must already be
    /// matching-shape; we make both contiguous here (matching the
    /// CUDA/ROCm path) so the kernel's flat `tid -> i` index lines up.
    pub fn swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        if gate.shape() != up.shape() {
            candle_core::bail!("swiglu: gate and up must have the same shape");
        }
        let dtype = gate.dtype();
        let (kernel_name, dtype_size) = dtype_kernel(dtype).ok_or_else(|| {
            candle_core::Error::Msg(format!("swiglu: unsupported dtype {dtype:?} on Metal"))
        })?;

        let n = gate.elem_count();
        // Flat kernel indexes a contiguous buffer; broadcasting / narrow'd
        // views are compacted first, matching the CUDA/ROCm/SYCL branches.
        let gate = gate.contiguous()?;
        let up = up.contiguous()?;

        let (gate_s, gate_l) = gate.storage_and_layout();
        let (up_s, up_l) = up.storage_and_layout();
        let gate_storage = match &*gate_s {
            Storage::Metal(s) => s,
            _ => {
                candle_core::bail!("swiglu: gate must be a metal tensor");
            },
        };
        let up_storage = match &*up_s {
            Storage::Metal(s) => s,
            _ => {
                candle_core::bail!("swiglu: up must be a metal tensor");
            },
        };

        let device = gate_storage.device().clone();
        let pipeline = get_pipeline(&device, kernel_name)?;
        let (grid, tgp) = dispatch_dims(n, &pipeline);

        let dst = device.new_buffer(n, dtype, "crane_swiglu.out")?;
        // Scope the encoder so its borrow of `device` ends before we move
        // `device` into `MetalStorage::new` below. Encoded work is committed
        // when `encoder`/`CommandsGuard` drops (held by `Commands` internally),
        // so the buffer-bound command buffer runs after this scope closes —
        // which is the right time relative to the `dst` `Arc<Buffer>` we're
        // handing back into a `Tensor`.
        {
            // One command encoder for the whole dispatch. The
            // `CommandsGuard` returned by `command_encoder()` is a shared
            // encoder guarded by a `Mutex` inside `Commands`, so multiple
            // threads encoding in parallel simply queue up against the same
            // in-flight command buffer — that's exactly what every other
            // candle Metal backend op does. We unwrap the inner
            // `ComputeCommandEncoder` once via `.as_ref()` because only the
            // guard exposes `set_compute_pipeline_state` / `set_label`; the
            // buffer-binding and dispatch methods all live on
            // `ComputeCommandEncoder` itself.
            let encoder_guard = device.command_encoder()?;
            let encoder: &ComputeCommandEncoder = encoder_guard.as_ref();
            encoder.set_compute_pipeline_state(&pipeline);
            // Device buffer bindings are byte offsets, but Metal re-validates
            // the underlying CPU-readable pointer — we don't read it, so
            // passing `None` for the buffer arg is *not* correct here. The
            // buffer base is bound; the kernel multiplies `tid` itself.
            encoder.set_input_buffer(
                0,
                Some(gate_storage.buffer()),
                gate_l.start_offset() * dtype_size,
            );
            encoder.set_input_buffer(
                1,
                Some(up_storage.buffer()),
                up_l.start_offset() * dtype_size,
            );
            // Output buffer is registered via the dedicated
            // `set_output_buffer` path so candle-metal-kernels'
            // hazard-tracking fence database marks the page as written —
            // required under HazardTrackingModeUntracked.
            encoder.set_output_buffer(2, Some(&dst), 0);
            // Scalar `n` packed into the kernel's `constant uint &n` slot.
            let n_u32 = u32::try_from(n).map_err(|_| {
                candle_core::Error::Msg(format!(
                    "swiglu: {n} elements exceeds u32::MAX for the Metal kernel"
                ))
            })?;
            encoder.set_bytes(3, &n_u32);
            encoder.dispatch_thread_groups(grid, tgp);
        }

        let storage = Storage::Metal(MetalStorage::new(dst, device, n, dtype));
        Ok(Tensor::from_storage(
            storage,
            gate_l.shape().clone(),
            BackpropOp::none(),
            false,
        ))
    }
}

/// Fused `SwiGLU`: `silu(gate) * up`.
struct SwigluOp;

impl CustomOp2 for SwigluOp {
    fn name(&self) -> &'static str {
        "swiglu"
    }

    fn cpu_fwd(
        &self,
        s_gate: &CpuStorage,
        l_gate: &Layout,
        s_up: &CpuStorage,
        l_up: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: WithDType>(
            gate: &[T],
            l_gate: &Layout,
            up: &[T],
            l_up: &Layout,
        ) -> (CpuStorage, Shape) {
            let dst =
                candle_core::cpu_backend::binary_map(l_gate, l_up, gate, up, |g_val, u_val| {
                    let g = g_val.to_f64();
                    let u = u_val.to_f64();
                    T::from_f64((g / (1.0 + (-g).exp())) * u)
                });
            (T::to_cpu_storage_owned(dst), l_gate.shape().clone())
        }

        match (s_gate, s_up) {
            (CpuStorage::BF16(gate), CpuStorage::BF16(up)) => Ok(inner(gate, l_gate, up, l_up)),
            (CpuStorage::F16(gate), CpuStorage::F16(up)) => Ok(inner(gate, l_gate, up, l_up)),
            (CpuStorage::F32(gate), CpuStorage::F32(up)) => Ok(inner(gate, l_gate, up, l_up)),
            (CpuStorage::F64(gate), CpuStorage::F64(up)) => Ok(inner(gate, l_gate, up, l_up)),
            _ => candle_core::bail!("swiglu: unsupported or mismatched dtypes"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_gate: &CudaStorage,
        l_gate: &Layout,
        s_up: &CudaStorage,
        l_up: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = s_gate.device();
        let n = l_gate.shape().elem_count();

        let (go1, go2) = l_gate
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("swiglu: gate must be contiguous".into()))?;
        let (uo1, uo2) = l_up
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("swiglu: up must be contiguous".into()))?;
        if go2 - go1 != n || uo2 - uo1 != n {
            candle_core::bail!("swiglu: gate and up must have the same element count");
        }

        let fn_name = match s_gate.dtype() {
            DType::BF16 => "swiglu_bf16",
            DType::F16 => "swiglu_f16",
            DType::F32 => "swiglu_f32",
            dt => candle_core::bail!("swiglu: unsupported dtype {dt:?}"),
        };
        let func = dev.get_or_load_custom_func(fn_name, MODULE_NAME, ptx::SWIGLU)?;

        let n_u32 = n as u32;
        let block_size = 256u32;
        let grid_size = n_u32.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let slice = match (&s_gate.slice, &s_up.slice) {
            (CudaStorageSlice::BF16(gate), CudaStorageSlice::BF16(up)) => {
                let gate = gate.slice(go1..go2);
                let up = up.slice(uo1..uo2);
                let dst = unsafe { dev.alloc::<half::bf16>(n)? };
                let mut builder = func.builder();
                builder.arg(&gate);
                builder.arg(&up);
                builder.arg(&dst);
                builder.arg(&n_u32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::BF16(dst)
            },
            (CudaStorageSlice::F16(gate), CudaStorageSlice::F16(up)) => {
                let gate = gate.slice(go1..go2);
                let up = up.slice(uo1..uo2);
                let dst = unsafe { dev.alloc::<half::f16>(n)? };
                let mut builder = func.builder();
                builder.arg(&gate);
                builder.arg(&up);
                builder.arg(&dst);
                builder.arg(&n_u32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F16(dst)
            },
            (CudaStorageSlice::F32(gate), CudaStorageSlice::F32(up)) => {
                let gate = gate.slice(go1..go2);
                let up = up.slice(uo1..uo2);
                let dst = unsafe { dev.alloc::<f32>(n)? };
                let mut builder = func.builder();
                builder.arg(&gate);
                builder.arg(&up);
                builder.arg(&dst);
                builder.arg(&n_u32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F32(dst)
            },
            _ => candle_core::bail!("swiglu: unsupported or mismatched CUDA storage types"),
        };

        let dst = CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l_gate.shape().clone()))
    }

    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    fn rocm_fwd(
        &self,
        s_gate: &RocmStorage,
        l_gate: &Layout,
        s_up: &RocmStorage,
        l_up: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        // SAFETY: swiglu_{bf16,f16,f32} in ROCM_SOURCE take (const T*, const
        // T*, T*, uint32_t) for dtype T, matching binary_elementwise_fwd's
        // contract.
        unsafe {
            crate::ops::rocm::binary_elementwise_fwd(
                s_gate,
                l_gate,
                s_up,
                l_up,
                ROCM_MODULE_NAME,
                "swiglu",
                ROCM_SOURCE,
            )
        }
    }
}

/// Fused `SwiGLU` activation: `silu(gate) * up`.
///
/// Computes `gate / (1 + exp(-gate)) * up` in a single pass, avoiding the
/// intermediate tensor that a separate `Silu` + `Mul` chain allocates.
/// `gate` and `up` must have the same shape. On CPU, `cpu_fwd` walks each
/// input's own strides via `binary_map`, so non-contiguous views (e.g. a
/// `narrow` of a shared `gate_up` tensor) are passed through as-is. On
/// CUDA/ROCm/Metal the kernel indexes flat buffers, so both inputs are made
/// contiguous first.
///
/// # Errors
///
/// Returns an error if `gate`/`up` have a dtype other than `BF16`/`F16`/
/// `F32`/`F64` (`cpu_fwd`), or on CUDA/ROCm/Metal other than `BF16`/`F16`/
/// `F32`, or if the shapes don't match.
pub fn swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.shape() != up.shape() {
        candle_core::bail!("swiglu: gate and up must have the same shape");
    }
    let device = gate.device();
    #[cfg(feature = "sycl")]
    if device.is_sycl() {
        return sycl_kernel::swiglu(gate, up);
    }
    #[cfg(feature = "metal")]
    if device.is_metal() {
        return metal_kernel::swiglu(gate, up);
    }
    if device.is_cpu() {
        return gate.apply_op2_no_bwd(up, &SwigluOp);
    }
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    gate.apply_op2_no_bwd(&up, &SwigluOp)
}

/// `silu(gate) * up` via candle's own ops. Kept around as the CPU-side
/// reference the kernel-correctness test compares against, and nothing
/// else: every device `swiglu()` runs on (CPU, CUDA, ROCm, SYCL, Metal)
/// has its own dispatch path now (portable dispatch was the previous
/// Metal and SYCL fallback, but both now have real fused Metal and SYCL
/// kernels). Both `Silu` and broadcast multiply are native candle ops with
/// real implementations on every backend, unlike `SwigluOp`'s hand-written
/// `CustomOp2`.
#[cfg_attr(not(test), allow(dead_code))]
fn portable_swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    candle_nn::ops::silu(gate)?.broadcast_mul(up)
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::{portable_swiglu, swiglu};

    /// Naive reference: separate silu + mul.
    fn naive_swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        let activated = candle_nn::ops::silu(gate)?;
        activated.broadcast_mul(up)
    }

    // Verifies the fused kernel matches silu(gate) * up for 1-D f32 values.
    #[test]
    fn swiglu_matches_naive_1d_f32() -> Result<()> {
        let gate = Tensor::new(&[0.0f32, 1.0, -1.0, 2.5, -0.5], &Device::Cpu)?;
        let up = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 1.5], &Device::Cpu)?;

        let got = swiglu(&gate, &up)?.to_vec1::<f32>()?;
        let expected = naive_swiglu(&gate, &up)?.to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
        Ok(())
    }

    // Verifies correctness on a 2-D shape (simulates a batch of tokens).
    #[test]
    fn swiglu_2d_shape() -> Result<()> {
        let gate = Tensor::new(&[[0.5f32, -0.3, 1.2], [2.0, -1.0, 0.0]], &Device::Cpu)?;
        let up = Tensor::new(&[[1.0f32, 0.8, -0.5], [0.3, 2.0, 1.0]], &Device::Cpu)?;

        let got = swiglu(&gate, &up)?.flatten_all()?.to_vec1::<f32>()?;
        let expected = naive_swiglu(&gate, &up)?.flatten_all()?.to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
        Ok(())
    }

    // Verifies the BF16 dtype branch.
    #[test]
    fn swiglu_bf16() -> Result<()> {
        let device = Device::Cpu;
        let gate = Tensor::new(&[0.0f32, 1.0, -1.0, 2.5], &device)?.to_dtype(DType::BF16)?;
        let up = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?.to_dtype(DType::BF16)?;

        let got = swiglu(&gate, &up)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let expected = naive_swiglu(&gate, &up)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-2, "got {g}, expected {e}");
        }
        Ok(())
    }

    // Verifies the F16 dtype branch.
    #[test]
    fn swiglu_f16() -> Result<()> {
        let device = Device::Cpu;
        let gate = Tensor::new(&[0.0f32, 1.0, -1.0, 2.5], &device)?.to_dtype(DType::F16)?;
        let up = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?.to_dtype(DType::F16)?;

        let got = swiglu(&gate, &up)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let expected = naive_swiglu(&gate, &up)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-3, "got {g}, expected {e}");
        }
        Ok(())
    }

    // Verifies the F64 dtype branch.
    #[test]
    fn swiglu_f64() -> Result<()> {
        let gate = Tensor::new(&[0.0f64, 1.0, -1.0, 2.5], &Device::Cpu)?;
        let up = Tensor::new(&[1.0f64, 2.0, 0.5, 3.0], &Device::Cpu)?;

        let got = swiglu(&gate, &up)?.to_vec1::<f64>()?;
        let expected = naive_swiglu(&gate, &up)?.to_vec1::<f64>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < f64::EPSILON * 100.0,
                "got {g}, expected {e}"
            );
        }
        Ok(())
    }

    // Verifies that unsupported dtypes return an error.
    #[test]
    fn swiglu_unsupported_dtype_errors() -> Result<()> {
        let gate = Tensor::new(&[1u32, 2, 3], &Device::Cpu)?;
        let up = Tensor::new(&[1u32, 2, 3], &Device::Cpu)?;

        let err = swiglu(&gate, &up).expect_err("unsupported dtype must error");

        assert!(
            err.to_string().contains("unsupported"),
            "unexpected error message: {err}"
        );
        Ok(())
    }

    // Verifies mismatched shapes error even when element counts match.
    #[test]
    fn swiglu_shape_mismatch_errors() -> Result<()> {
        let gate = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu)?;
        let up = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)?;

        let err = swiglu(&gate, &up).expect_err("mismatched shapes must error");

        assert!(
            err.to_string().contains("same shape"),
            "unexpected error message: {err}"
        );
        Ok(())
    }

    // Verifies silu(0) * up == 0 (gate=0 means sigmoid(0)=0.5, so silu(0)=0).
    #[test]
    fn swiglu_gate_zero() -> Result<()> {
        let gate = Tensor::new(&[0.0f32], &Device::Cpu)?;
        let up = Tensor::new(&[42.0f32], &Device::Cpu)?;

        let got = swiglu(&gate, &up)?.to_vec1::<f32>()?;

        assert!((got[0]).abs() < 1e-10);
        Ok(())
    }

    // Verifies NaN gate produces NaN output.
    #[test]
    fn swiglu_nan_passthrough() -> Result<()> {
        let gate = Tensor::new(&[f32::NAN, 1.0], &Device::Cpu)?;
        let up = Tensor::new(&[1.0f32, 1.0], &Device::Cpu)?;

        let got = swiglu(&gate, &up)?.to_vec1::<f32>()?;
        let expected = naive_swiglu(&gate, &up)?.to_vec1::<f32>()?;

        assert!(got[0].is_nan());
        assert!((got[1] - expected[1]).abs() < 1e-6);
        Ok(())
    }

    // Verifies non-contiguous inputs (e.g. narrow()'d halves of a shared
    // gate_up tensor, as used by the MoE expert forward paths) produce the
    // same result as contiguous ones.
    #[test]
    fn swiglu_non_contiguous_narrow() -> Result<()> {
        let gate_up = Tensor::new(
            &[
                [0.5f32, -0.3, 1.2, 1.0, 0.8, -0.5],
                [2.0, -1.0, 0.0, 0.3, 2.0, 1.0],
            ],
            &Device::Cpu,
        )?;
        let gate = gate_up.narrow(1, 0, 3)?;
        let up = gate_up.narrow(1, 3, 3)?;
        assert!(!gate.is_contiguous());
        assert!(!up.is_contiguous());

        let got = swiglu(&gate, &up)?.flatten_all()?.to_vec1::<f32>()?;
        let expected = naive_swiglu(&gate, &up)?.flatten_all()?.to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
        Ok(())
    }

    // Verifies the Metal/SYCL fallback path (exercised directly here, since
    // this CPU-only test suite can't reach it through `swiglu()`'s device
    // check) matches the fused kernel / naive reference.
    #[test]
    fn portable_swiglu_matches_naive() -> Result<()> {
        let gate = Tensor::new(&[0.0f32, 1.0, -1.0, 2.5, -0.5], &Device::Cpu)?;
        let up = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 1.5], &Device::Cpu)?;

        let got = portable_swiglu(&gate, &up)?.to_vec1::<f32>()?;
        let expected = naive_swiglu(&gate, &up)?.to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
        Ok(())
    }
}
