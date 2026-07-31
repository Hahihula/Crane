// SPDX-License-Identifier: MIT
//! Crane Added 20260731: ONNX `LayerNormalization` as a native eval op.

use candle_core::{DType, Result, Tensor, bail};

use crate::onnx::proto::{self, NodeProto};

/// ONNX `LayerNormalization`: normalizes `x` over its trailing dimensions.
///
/// `axis` selects the first normalized dimension (default `-1`); every
/// dimension from `axis` to the end of the tensor is reduced over. Both
/// positive and negative axis values are accepted, per the ONNX spec (e.g.
/// `axis=0` normalizes over every dimension). `epsilon` (default `1e-5`) is
/// added to the variance before the square root to avoid division by zero.
/// The optional ONNX `stash_type`
/// attribute is ignored: computation runs in the input's own dtype, except
/// that F16/BF16 inputs are promoted to F32 for the mean/variance
/// accumulation to avoid overflow, matching `candle_nn::LayerNorm`.
pub(crate) fn layer_norm(
    node: &NodeProto,
    x: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let axis = int_attribute(node, "axis", -1)?;
    let epsilon = float_attribute(node, "epsilon", 1e-5)?;

    let rank = x.rank();
    let normalized_axis = x.normalize_axis(axis)?;
    let reduce_axes: Vec<usize> = (normalized_axis..rank).collect();

    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        dtype => dtype,
    };
    let x_internal = x.to_dtype(internal_dtype)?;

    let mean = x_internal.mean_keepdim(reduce_axes.as_slice())?;
    let centered = x_internal.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(reduce_axes.as_slice())?;
    let normalized = centered.broadcast_div(&(variance + epsilon)?.sqrt()?)?;

    let output = normalized.to_dtype(x_dtype)?.broadcast_mul(scale)?;
    match bias {
        Some(bias) => output.broadcast_add(bias),
        None => Ok(output),
    }
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
            "LayerNormalization node '{}' has a non-INT '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type(),
        );
    }
    Ok(attribute.i)
}

fn float_attribute(node: &NodeProto, name: &str, default: f64) -> Result<f64> {
    let Some(attribute) = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == name)
    else {
        return Ok(default);
    };
    if attribute.r#type() != proto::attribute_proto::AttributeType::Float {
        bail!(
            "LayerNormalization node '{}' has a non-FLOAT '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type(),
        );
    }
    Ok(f64::from(attribute.f))
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use crate::onnx::proto::{AttributeProto, NodeProto, attribute_proto::AttributeType};

    use super::layer_norm;

    fn node_with_axis_epsilon(axis: i64, epsilon: f32) -> NodeProto {
        NodeProto {
            name: "LayerNormalization.0".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "axis".to_string(),
                    r#type: AttributeType::Int as i32,
                    i: axis,
                    ..Default::default()
                },
                AttributeProto {
                    name: "epsilon".to_string(),
                    r#type: AttributeType::Float as i32,
                    f: epsilon,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn axis_minus_one_matches_manual_layer_norm() -> Result<()> {
        // Normalizes a [2, 4] tensor over its last dim and checks the result
        // against the standard layer-norm formula computed by hand.
        let x = Tensor::new(
            &[[1.0f32, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            &Device::Cpu,
        )?;
        let scale = Tensor::new(&[1.5f32, 0.5, 2.0, 1.0], &Device::Cpu)?;
        let bias = Tensor::new(&[0.1f32, -0.2, 0.3, 0.0], &Device::Cpu)?;
        let node = node_with_axis_epsilon(-1, 1e-5);

        let output = layer_norm(&node, &x, &scale, Some(&bias))?;

        let mean = 2.5f32;
        let variance = 1.25f32;
        let std_dev = (variance + 1e-5).sqrt();
        let expected_row0 = [
            (1.0 - mean) / std_dev * 1.5 + 0.1,
            (2.0 - mean) / std_dev * 0.5 - 0.2,
            (3.0 - mean) / std_dev * 2.0 + 0.3,
            (4.0 - mean) / std_dev * 1.0 + 0.0,
        ];
        let got: Vec<Vec<f32>> = output.to_vec2()?;
        for (got, expected) in got[0].iter().zip(expected_row0.iter()) {
            assert!((got - expected).abs() < 1e-4, "{got} vs {expected}");
        }
        Ok(())
    }

    #[test]
    fn axis_minus_two_reduces_over_last_two_dims() -> Result<()> {
        // With axis=-2 on a [2, 3, 4] tensor, normalization spans dims 1 and
        // 2 together rather than just the trailing dim.
        let x = Tensor::arange(0f32, 24f32, &Device::Cpu)?.reshape((2, 3, 4))?;
        let scale = Tensor::ones((3, 4), DType::F32, &Device::Cpu)?;
        let node = node_with_axis_epsilon(-2, 1e-5);

        let output = layer_norm(&node, &x, &scale, None)?;

        assert_eq!(output.dims(), &[2, 3, 4]);
        let mean_over_slice = output.mean(vec![1, 2])?.to_vec1::<f32>()?;
        for value in mean_over_slice {
            assert!(value.abs() < 1e-4, "{value}");
        }
        Ok(())
    }

    #[test]
    fn positive_axis_matches_equivalent_negative_axis() -> Result<()> {
        // axis=1 on a [2, 3, 4] tensor reduces over dims 1 and 2, the same
        // dims as axis=-2 — positive axis values must be accepted too.
        let x = Tensor::arange(0f32, 24f32, &Device::Cpu)?.reshape((2, 3, 4))?;
        let scale = Tensor::ones((3, 4), DType::F32, &Device::Cpu)?;
        let node_positive = node_with_axis_epsilon(1, 1e-5);
        let node_negative = node_with_axis_epsilon(-2, 1e-5);

        let output_positive = layer_norm(&node_positive, &x, &scale, None)?;
        let output_negative = layer_norm(&node_negative, &x, &scale, None)?;

        let got_positive: Vec<f32> = output_positive.flatten_all()?.to_vec1()?;
        let got_negative: Vec<f32> = output_negative.flatten_all()?.to_vec1()?;
        for (positive, negative) in got_positive.iter().zip(got_negative.iter()) {
            assert!(
                (positive - negative).abs() < 1e-6,
                "{positive} vs {negative}"
            );
        }
        Ok(())
    }

    #[test]
    fn missing_bias_is_optional() -> Result<()> {
        // ONNX allows LayerNormalization with only X and Scale; bias must
        // not be required.
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &Device::Cpu)?;
        let scale = Tensor::ones(4, DType::F32, &Device::Cpu)?;
        let node = node_with_axis_epsilon(-1, 1e-5);

        let output = layer_norm(&node, &x, &scale, None)?;
        assert_eq!(output.dims(), &[1, 4]);
        Ok(())
    }
}
