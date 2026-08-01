// SPDX-License-Identifier: MIT
//! Crane Added 20260731: ONNX `Atan` and `Atan2` as native eval ops.
//!
//! `candle_core` has no native `.atan()` or `.atan2()` tensor methods (its
//! `UnaryOp` enum covers `Exp`/`Log`/`Sin`/`Cos`/… but not `atan`), so these
//! are implemented via `CustomOp1`/`CustomOp2` operating directly on tensor
//! storage. Both ops are CPU-only: only `cpu_fwd` is implemented, so
//! evaluating them on a CUDA or Metal tensor returns a runtime error rather
//! than running on-device. Upstream candle has an open PR adding
//! `atan`/`atan2` (<https://github.com/huggingface/candle/pull/3338>); once
//! that ships in a released version this crate upgrades to, this file can be
//! replaced with direct tensor method calls.

use candle_core::{CpuStorage, CustomOp1, CustomOp2, Layout, Result, Shape, Tensor, WithDType};

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

/// Element-wise `atan2(y, x)`. IEEE 754 compliant: `atan2(0, 0) = 0`,
/// handling the zero-magnitude case without a special branch. CPU-only.
struct Atan2Op;

impl CustomOp2 for Atan2Op {
    fn name(&self) -> &'static str {
        "atan2"
    }

    fn cpu_fwd(
        &self,
        s_y: &CpuStorage,
        l_y: &Layout,
        s_x: &CpuStorage,
        l_x: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: WithDType>(
            y: &[T],
            l_y: &Layout,
            x: &[T],
            l_x: &Layout,
        ) -> (CpuStorage, Shape) {
            let dst = candle_core::cpu_backend::binary_map(l_y, l_x, y, x, |a, b| {
                T::from_f64(a.to_f64().atan2(b.to_f64()))
            });
            (T::to_cpu_storage_owned(dst), l_y.shape().clone())
        }

        match (s_y, s_x) {
            (CpuStorage::BF16(y), CpuStorage::BF16(x)) => Ok(inner(y, l_y, x, l_x)),
            (CpuStorage::F16(y), CpuStorage::F16(x)) => Ok(inner(y, l_y, x, l_x)),
            (CpuStorage::F32(y), CpuStorage::F32(x)) => Ok(inner(y, l_y, x, l_x)),
            (CpuStorage::F64(y), CpuStorage::F64(x)) => Ok(inner(y, l_y, x, l_x)),
            _ => candle_core::bail!("unsupported or mismatched dtypes for Atan2"),
        }
    }
}

/// Fused `Atan2`: element-wise `atan2(y, x)`.
///
/// Computes the two-argument arctangent directly rather than through the
/// decomposed `Div(y,x) → Atan → quadrant-correction Where` chain that ONNX
/// exporters emit for opsets without a native `Atan2` op. This avoids the
/// numerical instability that decomposition introduces near the origin (where
/// `Div(0,0) = NaN` and the quadrant-decision `Less(x, 0)` is noise-sensitive).
pub(crate) fn atan2(y: &Tensor, x: &Tensor) -> Result<Tensor> {
    y.apply_op2_no_bwd(x, &Atan2Op)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use super::{atan, atan2};

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

    // Verifies all four quadrants of atan2.
    #[test]
    fn atan2_all_quadrants() -> Result<()> {
        let y_vals = Tensor::new(&[1.0f32, 1.0, -1.0, -1.0], &Device::Cpu)?;
        let x_vals = Tensor::new(&[1.0f32, -1.0, -1.0, 1.0], &Device::Cpu)?;

        let result = atan2(&y_vals, &x_vals)?;

        let got = result.to_vec1::<f32>()?;
        let expected: Vec<f32> = vec![
            1.0f32.atan2(1.0),
            1.0f32.atan2(-1.0),
            (-1.0f32).atan2(-1.0),
            (-1.0f32).atan2(1.0),
        ];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < f32::EPSILON);
        }
        Ok(())
    }

    // IEEE 754: atan2(0, 0) = 0.
    #[test]
    fn atan2_zero_zero_is_zero() -> Result<()> {
        let y = Tensor::new(&[0.0f32], &Device::Cpu)?;
        let x = Tensor::new(&[0.0f32], &Device::Cpu)?;

        let result = atan2(&y, &x)?;

        assert_eq!(result.to_vec1::<f32>()?, vec![0.0]);
        Ok(())
    }

    // Verifies atan2 on the axes (y=0 or x=0).
    #[test]
    fn atan2_on_axes() -> Result<()> {
        let y_vals = Tensor::new(&[0.0f32, 1.0, 0.0, -1.0], &Device::Cpu)?;
        let x_vals = Tensor::new(&[1.0f32, 0.0, -1.0, 0.0], &Device::Cpu)?;

        let result = atan2(&y_vals, &x_vals)?;

        let got = result.to_vec1::<f32>()?;
        let expected: Vec<f32> = vec![
            0.0f32.atan2(1.0),
            1.0f32.atan2(0.0),
            0.0f32.atan2(-1.0),
            (-1.0f32).atan2(0.0),
        ];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < f32::EPSILON);
        }
        Ok(())
    }
}
