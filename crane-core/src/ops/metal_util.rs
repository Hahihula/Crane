// SPDX-License-Identifier: MIT

//! Plumbing shared by the direct-dispatch Metal kernels (`kernels/metal/*.metal`):
//! runtime compilation of an MSL source into an `MTLLibrary` (once per
//! device and library), a per-kernel `ComputePipeline` cache, and buffer /
//! output helpers.
//!
//! Kernels are dispatched on candle's own command encoder
//! ([`MetalDevice::command_encoder`]), so they order with candle's ops and
//! its hazard tracking sees every buffer bound through
//! `set_input_buffer` / `set_output_buffer`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::metal_backend::{MetalDevice, MetalStorage};
use candle_core::op::BackpropOp;
use candle_core::{DType, Result, Shape, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline, Library};
use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode};

type Key = (u64, &'static str);

/// The pipeline of kernel `name` in library `library` (compiled from
/// `source()` on first use on `device`).
///
/// # Errors
///
/// Returns an error if the source fails to compile or has no kernel `name`
/// (e.g. a `bf16` kernel on Metal < 3.0).
pub fn pipeline(
    device: &MetalDevice,
    library: &'static str,
    source: impl FnOnce() -> String,
    name: &str,
) -> Result<ComputePipeline> {
    static LIBRARIES: OnceLock<Mutex<HashMap<Key, Library>>> = OnceLock::new();
    static PIPELINES: OnceLock<Mutex<HashMap<(u64, String), ComputePipeline>>> = OnceLock::new();

    let id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (id, format!("{library}::{name}"));
    if let Some(p) = pipelines.lock().unwrap().get(&key) {
        return Ok(p.clone());
    }
    let lib = {
        let mut libs = LIBRARIES
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap();
        if let Some(lib) = libs.get(&(id, library)) {
            lib.clone()
        } else {
            let opts = MTLCompileOptions::new();
            opts.setMathMode(MTLMathMode::Fast);
            opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Fast);
            let lib = device
                .device()
                .new_library_with_source(&source(), Some(&opts))
                .map_err(|e| candle_core::Error::Msg(format!("Metal {library}: compile: {e}")))?;
            libs.insert((id, library), lib.clone());
            lib
        }
    };
    let function = lib
        .get_function(name, None)
        .map_err(|e| candle_core::Error::Msg(format!("Metal {library}: {name}: {e}")))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| candle_core::Error::Msg(format!("Metal {library}: pipeline {name}: {e}")))?;
    pipelines.lock().unwrap().insert(key, pipeline.clone());
    Ok(pipeline)
}

/// Kernel-name suffix for a float dtype.
///
/// # Errors
///
/// Returns an error for anything but F32/F16/BF16.
pub fn float_tag(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        other => candle_core::bail!("Metal kernels do not support {other:?}"),
    }
}

/// The Metal buffer behind `t` (whose storage guard is `storage`) and the
/// byte offset of its first element plus `extra_bytes`.
///
/// # Errors
///
/// Returns an error if `t` is not a Metal tensor.
pub fn buffer<'a>(
    storage: &'a Storage,
    t: &Tensor,
    extra_bytes: usize,
    name: &str,
) -> Result<(&'a Buffer, usize)> {
    let Storage::Metal(st) = storage else {
        candle_core::bail!("Metal kernel: {name} must be a metal tensor")
    };
    let offset = t.layout().start_offset() * t.dtype().size_in_bytes();
    Ok((st.buffer(), offset + extra_bytes))
}

/// A fresh output buffer for `n` elements of `dtype`.
///
/// # Errors
///
/// Returns an error if the allocation fails.
pub fn output(dev: &MetalDevice, n: usize, dtype: DType, label: &str) -> Result<Arc<Buffer>> {
    dev.new_buffer_builder()
        .with_size_for(n, dtype)
        .with_label(label)
        .build()
}

/// Wrap a kernel's output buffer as a tensor of `shape`.
pub fn wrap(dev: &MetalDevice, buf: Arc<Buffer>, shape: impl Into<Shape>, dtype: DType) -> Tensor {
    let shape = shape.into();
    Tensor::from_storage(
        Storage::Metal(MetalStorage::new(
            buf,
            dev.clone(),
            shape.elem_count(),
            dtype,
        )),
        shape,
        BackpropOp::none(),
        false,
    )
}
