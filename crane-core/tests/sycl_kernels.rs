//! SYCL kernel correctness checks against the portable references.
//!
//! Needs an Intel GPU + oneAPI. Run inside the `docker/sycl` dev container:
//!   cargo test -p crane-core --release --features sycl --test sycl_kernels
#![cfg(all(feature = "sycl", not(feature = "cuda"), not(feature = "rocm")))]
use crane_core::{candle_core, candle_nn};

use candle_core::{DType, Device, Result, Tensor};
use crane_core::ops::fused_ops::snake::snake;
use crane_core::ops::fused_ops::swiglu::swiglu;
use crane_core::ops::gdn::{gated_delta_rule_recurrence, gdn_recurrence_sycl};

fn device() -> Option<Device> {
    Device::new_sycl(0).ok()
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

/// The fused SYCL recurrence against the op-by-op reference it replaces, for
/// both a prefill-length sequence and single-token decode.
#[test]
fn gdn_recurrence_matches_the_portable_reference() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("no SYCL device — skipping");
        return Ok(());
    };

    for kdim in [128usize, 64] {
        for s in [24usize, 1] {
            let (b, h, vdim) = (1usize, 4usize, 128usize);

            let q = Tensor::randn(0f32, 1.0, (b, s, h, kdim), &dev)?;
            let k = Tensor::randn(0f32, 1.0, (b, s, h, kdim), &dev)?;
            let v = Tensor::randn(0f32, 1.0, (b, s, h, vdim), &dev)?;
            let g = Tensor::randn(0f32, 1.0, (b, s, h), &dev)?.affine(0.01, -0.05)?;
            let beta = candle_nn::ops::sigmoid(&Tensor::randn(0f32, 1.0, (b, s, h), &dev)?)?;
            let state = Tensor::randn(0f32, 0.1, (b, h, kdim, vdim), &dev)?;

            let mut state_ref = state.clone();
            let y_ref = gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut state_ref)?;

            let bh = b * h;
            let flat3 = |t: &Tensor| -> Result<Tensor> {
                t.transpose(1, 2)?.contiguous()?.reshape((bh, s, ()))
            };
            let flat2 = |t: &Tensor| -> Result<Tensor> {
                t.transpose(1, 2)?.contiguous()?.reshape((bh, s))
            };
            let scale = 1.0 / (kdim as f64).sqrt();
            let (y, state_out) = gdn_recurrence_sycl(
                &flat3(&q)?.affine(scale, 0.0)?,
                &flat3(&k)?,
                &flat3(&v)?,
                &flat2(&g)?,
                &flat2(&beta)?,
                &state.reshape((bh, kdim, vdim))?.contiguous()?,
            )?;
            let y = y.reshape((b, h, s, vdim))?.transpose(1, 2)?.contiguous()?;

            let sim_y = cos_sim(&y, &y_ref)?;
            let sim_state = cos_sim(&state_out.reshape((b, h, kdim, vdim))?, &state_ref)?;
            eprintln!("K={kdim} S={s}: y_cos={sim_y:.5} state_cos={sim_state:.5}");
            assert!(sim_y >= 0.999, "K={kdim} S={s}: y cos={sim_y}");
            assert!(sim_state >= 0.999, "K={kdim} S={s}: state cos={sim_state}");
        }
    }
    Ok(())
}

/// The fused SYCL `swiglu` kernel against `silu(gate) * up` computed on CPU,
/// across the dtypes `crane_swiglu_sycl` supports (F32/F16/BF16) and both a
/// contiguous shape and a broadcast one (matching an MoE expert's up-proj).
#[test]
fn swiglu_matches_the_portable_reference() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("no SYCL device — skipping");
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

/// The fused SYCL `snake` kernel against `x + sin(alpha*x)^2/alpha` computed
/// on CPU, across dtypes and the broadcast shape (`alpha` per-channel,
/// matching a BigVGAN-family vocoder decoder) that exercises `.contiguous()`.
#[test]
fn snake_matches_the_portable_reference() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("no SYCL device — skipping");
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
