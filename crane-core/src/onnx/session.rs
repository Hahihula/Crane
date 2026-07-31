//! Reusable execution state for Crane's ONNX evaluator.

use std::collections::HashMap;

use candle_core::{Result, Tensor};

use super::{eval, proto};

/// An ONNX model whose initializer tensors are decoded once at load time.
///
/// The graph is still executed by [`eval::simple_eval`], keeping Crane's
/// reusable state separate from the vendored evaluator implementation.
pub struct Session {
    model: proto::ModelProto,
    initializers: HashMap<String, Tensor>,
}

impl Session {
    pub fn new(mut model: proto::ModelProto) -> Result<Self> {
        let graph = model
            .graph
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("no graph defined in proto".to_string()))?;
        let mut initializers = HashMap::with_capacity(graph.initializer.len());

        // Crane Added 20260731: cache parameter tensors instead of decoding
        // the protobuf initializer payload again on every forward pass.
        for initializer in std::mem::take(&mut graph.initializer) {
            let tensor = eval::get_tensor(&initializer, &initializer.name)?;
            initializers.insert(initializer.name, tensor);
        }

        Ok(Self {
            model,
            initializers,
        })
    }

    pub fn run(&self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let mut values = inputs;
        values.extend(
            self.initializers
                .iter()
                .map(|(name, tensor)| (name.clone(), tensor.clone())),
        );
        eval::simple_eval(&self.model, values)
    }
}
