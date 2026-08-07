// SPDX-License-Identifier: MIT
//! Crane Added 20260731: ONNX `Atan` as a native eval op.
//!
//! `candle_core` has no native `.atan()` tensor method (its `UnaryOp` enum
//! covers `Exp`/`Log`/`Sin`/`Cos`/… but not `atan`), so this is implemented
//! via `CustomOp1` operating directly on tensor storage. CPU-only: only
//! `cpu_fwd` is implemented, so evaluating it on a CUDA or Metal tensor
//! returns a runtime error rather than running on-device. Upstream candle
//! has an open PR adding `atan`/`atan2`
//! (<https://github.com/huggingface/candle/pull/3338>); once that ships in a
//! released version this crate upgrades to, this file can be replaced with a
//! direct tensor method call. `Atan2` (a fused, reusable op with CUDA+CPU
//! dispatch) lives in [`crate::ops::fused_ops::atan2`].

use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor, WithDType};

/// Element-wise `atan`. Spec-correct: NaN inputs produce NaN outputs (the
/// ONNX spec does not define NaN-clamping behavior for `Atan`). CPU-only.
struct AtanOp;

impl CustomOp1 for AtanOp {
    fn name(&self) -> &'static str {
        "atan"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        fn inner<T: WithDType>(src: &[T], layout: &Layout) -> (CpuStorage, Shape) {
            let dst = candle_core::cpu_backend::unary_map(src, layout, |v| {
                T::from_f64(v.to_f64().atan())
            });
            (T::to_cpu_storage_owned(dst), layout.shape().clone())
        }

        match storage {
            CpuStorage::BF16(s) => Ok(inner(s, layout)),
            CpuStorage::F16(s) => Ok(inner(s, layout)),
            CpuStorage::F32(s) => Ok(inner(s, layout)),
            CpuStorage::F64(s) => Ok(inner(s, layout)),
            _ => candle_core::bail!("unsupported dtype for Atan"),
        }
    }
}

/// ONNX `Atan`: element-wise arctangent.
pub(crate) fn atan(input: &Tensor) -> Result<Tensor> {
    input.apply_op1_no_bwd(&AtanOp)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use super::atan;

    // Verifies that atan matches the standard library f32::atan for a range
    // of representative inputs.
    #[test]
    fn atan_matches_std_atan_elementwise() -> Result<()> {
        let values = [0.0f32, 1.0, -1.0, 10.0, -10.0, 0.5];
        let x = Tensor::new(&values, &Device::Cpu)?;

        let y = atan(&x)?;

        let got = y.to_vec1::<f32>()?;
        let expected: Vec<f32> = values.iter().copied().map(f32::atan).collect();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < f32::EPSILON);
        }
        Ok(())
    }

    // Verifies that the F64 dtype branch is exercised and round-trips
    // without precision loss.
    #[test]
    fn atan_f64() -> Result<()> {
        let values = [0.0f64, 1.0, -1.0, 10.0, -10.0, 0.5];
        let x = Tensor::new(&values, &Device::Cpu)?;

        let y = atan(&x)?;

        let got = y.to_vec1::<f64>()?;
        let expected: Vec<f64> = values.iter().copied().map(f64::atan).collect();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < f64::EPSILON);
        }
        Ok(())
    }

    // Verifies that an unsupported dtype returns an error rather than
    // panicking.
    #[test]
    fn atan_unsupported_dtype_errors() -> Result<()> {
        let x = Tensor::new(&[1u32, 2, 3], &Device::Cpu)?;

        let err = atan(&x).expect_err("unsupported dtype must error");

        assert!(
            err.to_string().contains("unsupported dtype"),
            "unexpected error message: {err}"
        );
        Ok(())
    }

    // Spec-correct behavior: NaN inputs produce NaN outputs (no clamping).
    #[test]
    fn atan_preserves_nan() -> Result<()> {
        let x = Tensor::new(&[f32::NAN, 1.0, f32::NAN], &Device::Cpu)?;

        let y = atan(&x)?;

        let got = y.to_vec1::<f32>()?;
        assert!(got[0].is_nan());
        assert!((got[1] - 1.0f32.atan()).abs() < f32::EPSILON);
        assert!(got[2].is_nan());
        Ok(())
    }

    // Verifies that positive and negative infinity produce the expected
    // asymptotic values (±π/2).
    #[test]
    fn atan_infinity() -> Result<()> {
        let x = Tensor::new(&[f32::INFINITY, f32::NEG_INFINITY], &Device::Cpu)?;

        let y = atan(&x)?;

        let got = y.to_vec1::<f32>()?;
        let half_pi = std::f32::consts::FRAC_PI_2;
        assert!((got[0] - half_pi).abs() < 1e-7);
        assert!((got[1] + half_pi).abs() < 1e-7);
        Ok(())
    }
}
