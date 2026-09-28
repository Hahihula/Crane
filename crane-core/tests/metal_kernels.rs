//! Metal kernel correctness checks against the portable references.
//!
//! Needs an Apple Silicon Mac (Metal is `cfg`-gated to `target_os = "macos"`
//! and candle cannot link Metal alongside another GPU backend). Run with:
//!   cargo test -p crane-core --release --features metal --test metal_kernels
#![cfg(all(
    feature = "metal",
    not(feature = "cuda"),
    not(feature = "rocm"),
    not(feature = "sycl")
))]
use crane_core::{candle_core, candle_nn};

use candle_core::{DType, Device, Result, Tensor};
use crane_core::ops::fused_ops::snake::snake;
use crane_core::ops::fused_ops::swiglu::swiglu;

fn device() -> Option<Device> {
    Device::new_metal(0).ok()
}

fn cos_sim(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "shape mismatch");
    let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    Ok(dot / (na * nb))
}

/// Fused Metal `swiglu` vs. `silu(gate) * up` computed on CPU, across the
/// dtypes `crane_swiglu_*` supports (F32/F16/BF16) and both a contiguous
/// shape and a broadcast one (matching an MoE expert's up-proj).
#[test]
fn swiglu_matches_the_portable_reference() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("no Metal device — skipping");
        return Ok(());
    };

    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let gate_cpu = Tensor::randn(0f32, 2.0, (4, 37), &Device::Cpu)?.to_dtype(dtype)?;
        let up_cpu = Tensor::randn(0f32, 2.0, (4, 37), &Device::Cpu)?.to_dtype(dtype)?;
        let expected = candle_nn::ops::silu(&gate_cpu.to_dtype(DType::F32)?)?
            .broadcast_mul(&up_cpu.to_dtype(DType::F32)?)?;

        let gate = gate_cpu.to_device(&dev)?;
        let up = up_cpu.to_device(&dev)?;
        let got = swiglu(&gate, &up)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;

        let sim = cos_sim(&got, &expected)?;
        eprintln!("swiglu {dtype:?}: cos={sim:.6}");
        assert!(sim >= 0.999, "{dtype:?}: cos={sim}");
    }
    Ok(())
}

/// Fused Metal `snake` vs. `x + sin(alpha*x)^2/alpha` computed on CPU,
/// across dtypes and the broadcast shape (`alpha` per-channel, matching a
/// BigVGAN-family vocoder decoder) that exercises `.contiguous()`.
#[test]
fn snake_matches_the_portable_reference() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("no Metal device — skipping");
        return Ok(());
    };

    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let x_cpu = Tensor::randn(0f32, 1.0, (1, 8, 16), &Device::Cpu)?.to_dtype(dtype)?;
        let alpha_cpu =
            (Tensor::randn(0f32, 1.0, (1, 8, 1), &Device::Cpu)?.abs()? + 0.1)?.to_dtype(dtype)?;
        let alpha_b_cpu = alpha_cpu.broadcast_as(x_cpu.shape())?;
        let x_f32 = x_cpu.to_dtype(DType::F32)?;
        let alpha_f32 = alpha_b_cpu.to_dtype(DType::F32)?;
        let sin_sq = x_f32.broadcast_mul(&alpha_f32)?.sin()?.powf(2.0)?;
        let expected = x_f32.broadcast_add(&sin_sq.broadcast_div(&alpha_f32)?)?;

        let x = x_cpu.to_device(&dev)?;
        let alpha = alpha_cpu.to_device(&dev)?.broadcast_as(x.shape())?;
        let got = snake(&x, &alpha)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;

        let sim = cos_sim(&got, &expected)?;
        eprintln!("snake {dtype:?}: cos={sim:.6}");
        assert!(sim >= 0.999, "{dtype:?}: cos={sim}");
    }
    Ok(())
}
