//! Crane Added 20260731: ONNX pooling operators not yet in upstream eval.

use candle_core::{Result, Tensor, bail};

use crate::onnx::proto::NodeProto;

/// Reduces all spatial axes, preserving their rank: `[N, C, D0, ...]` becomes
/// `[N, C, 1, ...]`, as required by ONNX GlobalAveragePool.
pub(crate) fn global_average_pool(node: &NodeProto, input: &Tensor) -> Result<Tensor> {
    if input.rank() < 3 {
        bail!(
            "GlobalAveragePool node '{}' requires rank >= 3, got rank {}",
            node.name,
            input.rank(),
        );
    }
    input.mean_keepdim((2..input.rank()).collect::<Vec<_>>())
}
