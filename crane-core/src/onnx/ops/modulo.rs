// SPDX-License-Identifier: MIT
//! Crane Added 20260731: ONNX `Mod` as a native eval op.

use candle_core::{DType, Result, Tensor, bail};

use crate::onnx::proto::{self, NodeProto};

/// ONNX `Mod`: elementwise modulo of `a` by `b`.
///
/// The `fmod` attribute (default `0`) selects the sign convention: `fmod=0`
/// uses Python/NumPy semantics where the result takes the sign of the
/// divisor (`a - floor(a / b) * b`); `fmod=1` uses C/Rust `%` semantics
/// where the result takes the sign of the dividend (`a - trunc(a / b) * b`).
/// candle has no built-in remainder or truncation op, so both are computed
/// via division in F64 followed by `floor` (or a round-trip through `I64`
/// for truncation), then cast back to the input dtype.
pub(crate) fn modulo(node: &NodeProto, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let fmod = int_attribute(node, "fmod", 0)?;

    let dtype = a.dtype();
    let a_f64 = a.to_dtype(DType::F64)?;
    let b_f64 = b.to_dtype(DType::F64)?;
    let quotient = a_f64.broadcast_div(&b_f64)?;

    let truncated_quotient = match fmod {
        0 => quotient.floor()?,
        1 => quotient.to_dtype(DType::I64)?.to_dtype(DType::F64)?,
        other => bail!(
            "Mod node '{}' has unsupported fmod value {other}; only 0 or 1 are valid",
            node.name
        ),
    };

    let remainder = (a_f64 - truncated_quotient.broadcast_mul(&b_f64)?)?;
    remainder.to_dtype(dtype)
}

fn int_attribute(node: &NodeProto, name: &str, default: i64) -> Result<i64> {
    let Some(attribute) = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == name)
    else {
        return Ok(default);
    };
    if attribute.r#type() != proto::attribute_proto::AttributeType::Int {
        bail!(
            "Mod node '{}' has a non-INT '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type(),
        );
    }
    Ok(attribute.i)
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use crate::onnx::proto::{AttributeProto, NodeProto, attribute_proto::AttributeType};

    use super::modulo;

    fn node_with_fmod(fmod: i64) -> NodeProto {
        NodeProto {
            name: "Mod.0".to_string(),
            attribute: vec![AttributeProto {
                name: "fmod".to_string(),
                r#type: AttributeType::Int as i32,
                i: fmod,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn fmod_zero_takes_sign_of_divisor() -> Result<()> {
        // Python/NumPy semantics: -7 % 3 == 2, 7 % 3 == 1.
        let a = Tensor::new(&[-7i64, 7], &Device::Cpu)?;
        let b = Tensor::new(&[3i64, 3], &Device::Cpu)?;
        let node = node_with_fmod(0);

        let output = modulo(&node, &a, &b)?;

        assert_eq!(output.to_vec1::<i64>()?, vec![2, 1]);
        Ok(())
    }

    #[test]
    fn fmod_one_takes_sign_of_dividend() -> Result<()> {
        // C/Rust semantics: -7 % 3 == -1, 7 % 3 == 1.
        let a = Tensor::new(&[-7i64, 7], &Device::Cpu)?;
        let b = Tensor::new(&[3i64, 3], &Device::Cpu)?;
        let node = node_with_fmod(1);

        let output = modulo(&node, &a, &b)?;

        assert_eq!(output.to_vec1::<i64>()?, vec![-1, 1]);
        Ok(())
    }

    #[test]
    fn broadcasts_scalar_divisor() -> Result<()> {
        // A scalar divisor should broadcast against a vector dividend.
        let a = Tensor::new(&[10i64, 11, 12], &Device::Cpu)?;
        let b = Tensor::new(3i64, &Device::Cpu)?;
        let node = node_with_fmod(0);

        let output = modulo(&node, &a, &b)?;

        assert_eq!(output.dims(), &[3]);
        assert_eq!(output.to_vec1::<i64>()?, vec![1, 2, 0]);
        Ok(())
    }

    #[test]
    fn float_inputs_preserve_dtype() -> Result<()> {
        // Mod on float inputs must return a float tensor, not an int one.
        let a = Tensor::new(&[5.5f32], &Device::Cpu)?;
        let b = Tensor::new(&[2.0f32], &Device::Cpu)?;
        let node = node_with_fmod(1);

        let output = modulo(&node, &a, &b)?;

        assert_eq!(output.dtype(), DType::F32);
        let got = output.to_vec1::<f32>()?;
        assert!((got[0] - 1.5).abs() < 1e-5, "{got:?}");
        Ok(())
    }
}
