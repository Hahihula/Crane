//! Crane Added 20260731: ONNX Squeeze across legacy and current opsets.

use candle_core::{Result, Tensor, bail};

use crate::onnx::proto::{self, NodeProto};

pub(crate) fn squeeze(
    node: &NodeProto,
    input: &Tensor,
    axes_input: Option<&Tensor>,
) -> Result<Tensor> {
    let attribute_axes = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == "axes");
    let mut axes = if let Some(attribute) = attribute_axes {
        if attribute.r#type() != proto::attribute_proto::AttributeType::Ints {
            bail!(
                "Squeeze node '{}' has a non-integer 'axes' attribute ({:?})",
                node.name,
                attribute.r#type()
            );
        }
        normalize_axes(input, &attribute.ints)?
    } else if let Some(axes) = axes_input {
        normalize_axes(input, &axes.to_vec1::<i64>()?)?
    } else {
        input
            .dims()
            .iter()
            .enumerate()
            .filter_map(|(axis, &dimension)| (dimension == 1).then_some(axis))
            .collect()
    };

    axes.sort_unstable();
    axes.dedup();
    let mut output = input.clone();
    for axis in axes.into_iter().rev() {
        output = output.squeeze(axis)?;
    }
    Ok(output)
}

fn normalize_axes(input: &Tensor, axes: &[i64]) -> Result<Vec<usize>> {
    axes.iter()
        .map(|&axis| input.normalize_axis(axis))
        .collect()
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use crate::onnx::proto::{AttributeProto, NodeProto, attribute_proto::AttributeType};

    use super::squeeze;

    #[test]
    fn legacy_axes_attribute_can_remove_leading_dimension() -> Result<()> {
        let input = Tensor::zeros((1, 12, 8, 95, 15), DType::F32, &Device::Cpu)?;
        let node = NodeProto {
            name: "Squeeze.2".to_string(),
            attribute: vec![AttributeProto {
                name: "axes".to_string(),
                r#type: AttributeType::Ints as i32,
                ints: vec![0],
                ..Default::default()
            }],
            ..Default::default()
        };
        let output = squeeze(&node, &input, None)?;
        assert_eq!(output.dims(), &[12, 8, 95, 15]);
        Ok(())
    }
}
