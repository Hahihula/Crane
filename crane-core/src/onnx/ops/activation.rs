//! Crane Added 20260731: ONNX activation operators not yet in upstream eval.

use candle_core::{Result, Tensor, bail};

use crate::onnx::proto::{self, NodeProto};

/// ONNX HardSigmoid: `clip(alpha * x + beta, 0, 1)`.
///
/// ONNX defaults are `alpha = 0.2` and `beta = 0.5`; do not use a fixed
/// backend hard-sigmoid because exported models may override those attributes.
pub(crate) fn hard_sigmoid(node: &NodeProto, input: &Tensor) -> Result<Tensor> {
    let alpha = float_attribute(node, "alpha", 0.2)?;
    let beta = float_attribute(node, "beta", 0.5)?;
    input.affine(alpha, beta)?.clamp(0.0, 1.0)
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
            "HardSigmoid node '{}' has a non-float '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type(),
        );
    }
    Ok(f64::from(attribute.f))
}
