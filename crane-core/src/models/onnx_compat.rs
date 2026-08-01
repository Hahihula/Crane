// SPDX-License-Identifier: MIT

//! Load-time ONNX graph rewrites that work around gaps in
//! `crate::onnx::eval`'s vendored evaluator, for models (currently Kokoro
//! TTS) whose export hits an op `eval.rs` either can't run at all or runs
//! incorrectly.
//!
//! Two different kinds of gap exist, and only one of them belongs here:
//!
//! - **Ops `eval.rs` runs *incorrectly*** — the op is implemented, but its
//!   implementation is wrong for some input shape, dtype, or attribute
//!   combination. These are fixable by rewriting the graph, at load time,
//!   into a decomposition of ops `eval.rs` already runs *correctly*. That
//!   rewrite is [`rewrite_unsupported_ops`]'s job.
//! - **Ops `eval.rs` can't run at all** — no dispatch arm exists for the op
//!   (e.g. a real DSP computation like `Resize` `mode="linear"`, which has
//!   no equivalent decomposition into other ONNX ops). Those need a native
//!   Rust implementation spliced into the graph via segmentation instead —
//!   see `crate::models::kokoro_tts::native_ops`.
//!
//! This module intentionally never modifies `crate::onnx::eval` itself:
//! upstream Crane doesn't want per-model workarounds baked into the shared
//! evaluator, so every fix here is a graph transformation applied once,
//! before `crate::onnx::simple_eval` (or `native_ops`'s segmented calls
//! into it) ever sees the graph.
//!
//! # Gaps fixed here
//!
//! - **`Trilu` produces NaN on `+/-inf` inputs.** `eval.rs`'s `Trilu`
//!   computes `input * mask`, and `0 * inf` is `NaN` in IEEE 754 — so any
//!   masked-out entry of an `f32::INFINITY`/`f32::NEG_INFINITY`-valued
//!   input (e.g. an additive attention mask before a softmax) becomes NaN
//!   instead of the intended value. [`expand_trilu`] rewrites `Trilu` into
//!   a `Where`-based selection between `data` and a same-dtype zero
//!   tensor, which never multiplies and so never produces this NaN.

use anyhow::{Context, Result};

use crate::onnx::proto::attribute_proto::AttributeType;
use crate::onnx::proto::tensor_proto::DataType;
use crate::onnx::proto::{AttributeProto, GraphProto, NodeProto, TensorProto};

/// Rewrites every node in `graph` that unmodified `crate::onnx::eval`
/// handles incorrectly into a decomposition it handles correctly. Runs
/// once, at model load time, before the graph (or its segments) is passed
/// to `crate::onnx::simple_eval`.
pub(crate) fn rewrite_unsupported_ops(graph: &mut GraphProto) -> Result<()> {
    let orig_nodes = std::mem::take(&mut graph.node);
    let mut new_nodes = Vec::with_capacity(orig_nodes.len());

    for node in &orig_nodes {
        match node.op_type.as_str() {
            "Trilu" => expand_trilu(node, graph, &mut new_nodes)?,
            _ => new_nodes.push(node.clone()),
        }
    }

    graph.node = new_nodes;
    Ok(())
}

/// Rewrites `node` (a `Trilu`) in place into a small subgraph that avoids
/// the NaN `crate::onnx::eval`'s `Trilu` produces via `data * mask` when
/// `data` contains `+/-inf` and a masked-out entry is 0.
///
/// Instead of multiplying, the rewrite selects between `data` and a
/// same-dtype zero tensor with `Where`, driven by a 0/1 mask computed by
/// applying the *same* `Trilu` node — diagonal offset (`k`) and `upper`
/// attribute preserved verbatim, including a non-constant `k` input, since
/// nothing here needs `k`'s value at rewrite time — to an all-ones tensor
/// of `data`'s shape. `Trilu` applied to an all-ones/all-finite tensor can
/// never produce NaN, since it has no infinities to mishandle. Since the
/// mask is computed by `eval.rs`'s own unmodified `Trilu`, this rewrite
/// doesn't change (or fix) that op's existing rank-2-only mask broadcast —
/// a batched (rank > 2) `Trilu` input already fails in unmodified
/// `eval.rs` today, NaN or not, and continues to fail the same way here.
///
/// The zero-fill's data type is read from `graph.value_info` when `data`'s
/// declared type is present there; falls back to `Float` otherwise
/// (matching `eval.rs`'s own float-only defaults elsewhere, e.g.
/// `ConstantOfShape`'s default `value`). See [`scalar_value_attribute`] for
/// which dtypes are natively represented.
fn expand_trilu(node: &NodeProto, graph: &GraphProto, new_nodes: &mut Vec<NodeProto>) -> Result<()> {
    let data = node
        .input
        .first()
        .cloned()
        .context("Trilu node has no data input")?;
    let output = node
        .output
        .first()
        .cloned()
        .context("Trilu node has no output")?;

    let shape_name = format!("{output}__onnx_compat_shape");
    new_nodes.push(unary_node("Shape", &data, &shape_name));

    let ones_name = format!("{output}__onnx_compat_ones");
    new_nodes.push(constant_of_shape_node(
        &shape_name,
        &ones_name,
        scalar_value_attribute(DataType::Float as i32, 1.0),
    ));

    let mask_name = format!("{output}__onnx_compat_mask");
    let mut mask_node = node.clone();
    mask_node.input[0] = ones_name;
    mask_node.output = vec![mask_name.clone()];
    new_nodes.push(mask_node);

    let mask_bool_name = format!("{output}__onnx_compat_mask_bool");
    new_nodes.push(cast_node(&mask_name, &mask_bool_name, DataType::Bool));

    let data_dtype = declared_dtype(graph, &data).unwrap_or(DataType::Float as i32);
    let zero_name = format!("{output}__onnx_compat_zero");
    new_nodes.push(constant_of_shape_node(
        &shape_name,
        &zero_name,
        scalar_value_attribute(data_dtype, 0.0),
    ));

    new_nodes.push(NodeProto {
        name: node.name.clone(),
        op_type: "Where".to_string(),
        input: vec![mask_bool_name, data, zero_name],
        output: vec![output],
        ..Default::default()
    });

    Ok(())
}

/// Looks up `name`'s declared tensor type among `graph`'s `value_info`,
/// `input`, and `output` lists.
fn tensor_type_info<'a>(
    graph: &'a GraphProto,
    name: &str,
) -> Option<&'a crate::onnx::proto::type_proto::Tensor> {
    graph
        .value_info
        .iter()
        .chain(graph.input.iter())
        .chain(graph.output.iter())
        .find(|value_info| value_info.name == name)
        .and_then(|value_info| value_info.r#type.as_ref())
        .and_then(|type_proto| type_proto.value.as_ref())
        .and_then(|value| match value {
            crate::onnx::proto::type_proto::Value::TensorType(tensor_type) => Some(tensor_type),
            _ => None,
        })
}

/// Looks up `name`'s declared ONNX element type (as a raw `TensorProto`
/// `DataType` value) among `graph`'s `value_info`, `input`, and `output`
/// lists. Returns `None` when no declaration is present, which is common —
/// exporters often only populate `value_info` for a subset of tensors.
fn declared_dtype(graph: &GraphProto, name: &str) -> Option<i32> {
    tensor_type_info(graph, name).map(|tensor_type| tensor_type.elem_type)
}

/// Builds a single-input, single-output node with no attributes.
fn unary_node(op_type: &str, input: &str, output: &str) -> NodeProto {
    NodeProto {
        op_type: op_type.to_string(),
        input: vec![input.to_string()],
        output: vec![output.to_string()],
        ..Default::default()
    }
}

/// Builds a `ConstantOfShape` node reading its shape from `shape_input`
/// and filling with `value` (the node's `"value"` attribute).
fn constant_of_shape_node(shape_input: &str, output: &str, value: AttributeProto) -> NodeProto {
    NodeProto {
        op_type: "ConstantOfShape".to_string(),
        input: vec![shape_input.to_string()],
        output: vec![output.to_string()],
        attribute: vec![value],
        ..Default::default()
    }
}

/// Builds a `Cast` node converting `input` to ONNX element type `to`.
fn cast_node(input: &str, output: &str, to: DataType) -> NodeProto {
    NodeProto {
        op_type: "Cast".to_string(),
        input: vec![input.to_string()],
        output: vec![output.to_string()],
        attribute: vec![AttributeProto {
            name: "to".to_string(),
            r#type: AttributeType::Int as i32,
            i: to as i64,
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// Builds a scalar-tensor `"value"` attribute (as used by
/// `ConstantOfShape`) holding `value` encoded as `data_type`.
///
/// `eval.rs`'s `TENSOR`-attribute decoder only reads a tensor's `raw_data`
/// bytes (never the type-specific `float_data`/`int32_data`/`int64_data`
/// fields `crate::onnx::eval::get_tensor` accepts for initializers), and
/// has no `Int32` case at all — so a declared `Int32` dtype is represented
/// here as `Int64` instead, matching how `get_tensor` already promotes
/// ONNX `Int32` *tensors* (as opposed to this *attribute* path) to
/// candle's `I64` dtype elsewhere in this evaluator. Any other declared
/// type falls back to `Float`, since `value` only ever needs to
/// distinguish "zero" from "one" for the rewrites in this module.
fn scalar_value_attribute(data_type: i32, value: f64) -> AttributeProto {
    let (resolved_type, raw_data) = match DataType::try_from(data_type) {
        Ok(DataType::Int64 | DataType::Int32) => {
            #[allow(clippy::cast_possible_truncation)]
            let bytes = (value as i64).to_le_bytes().to_vec();
            (DataType::Int64, bytes)
        },
        Ok(DataType::Double) => (DataType::Double, value.to_le_bytes().to_vec()),
        _ => {
            #[allow(clippy::cast_possible_truncation)]
            let bytes = (value as f32).to_le_bytes().to_vec();
            (DataType::Float, bytes)
        },
    };
    AttributeProto {
        name: "value".to_string(),
        r#type: AttributeType::Tensor as i32,
        t: Some(TensorProto {
            data_type: resolved_type as i32,
            raw_data,
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Device, Tensor};

    use super::*;
    use crate::onnx::proto::{ModelProto, ValueInfoProto, type_proto};

    fn trilu_node(data: &str, k: Option<&str>, upper: i64, output: &str) -> NodeProto {
        let mut input = vec![data.to_string()];
        if let Some(k) = k {
            input.push(k.to_string());
        }
        NodeProto {
            op_type: "Trilu".to_string(),
            input,
            output: vec![output.to_string()],
            attribute: vec![AttributeProto {
                name: "upper".to_string(),
                r#type: AttributeType::Int as i32,
                i: upper,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn run_graph(graph: GraphProto, inputs: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        let model = ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        crate::onnx::simple_eval(&model, inputs).expect("simple_eval should succeed")
    }

    fn declare_tensor_type(name: &str, elem_type: DataType) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(crate::onnx::proto::TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: elem_type as i32,
                    ..Default::default()
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    // Upper-triangular Trilu on a +/-inf input must zero the masked-out
    // lower triangle instead of producing NaN via `-inf * 0`.
    #[test]
    fn trilu_upper_zeroes_masked_entries_without_nan() {
        let mut graph = GraphProto {
            node: vec![trilu_node("data", None, 1, "out")],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let data = Tensor::from_vec(
            vec![f32::INFINITY, 1.0, f32::NEG_INFINITY, 2.0],
            (2, 2),
            &Device::Cpu,
        )
        .unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);

        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![f32::INFINITY, 1.0], vec![0.0, 2.0]]);
    }

    // Lower-triangular Trilu (upper=0) must keep the diagonal and below,
    // zeroing the strictly-upper entries.
    #[test]
    fn trilu_lower_zeroes_masked_entries_without_nan() {
        let mut graph = GraphProto {
            node: vec![trilu_node("data", None, 0, "out")],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let data = Tensor::from_vec(
            vec![1.0, f32::INFINITY, f32::NEG_INFINITY, 2.0],
            (2, 2),
            &Device::Cpu,
        )
        .unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);

        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![1.0, 0.0], vec![f32::NEG_INFINITY, 2.0]]);
    }

    // A non-zero diagonal offset `k` (here as a graph input, i.e. not
    // resolvable at rewrite time) must still be honored, since the
    // rewrite forwards `k` verbatim into the mask's Trilu instead of
    // requiring it to be a compile-time constant.
    #[test]
    fn trilu_diagonal_offset_from_dynamic_input_is_honored() {
        let mut graph = GraphProto {
            node: vec![trilu_node("data", Some("k"), 1, "out")],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let data = Tensor::from_vec(
            vec![f32::INFINITY, 1.0, 2.0, f32::NEG_INFINITY, 3.0, 4.0],
            (2, 3),
            &Device::Cpu,
        )
        .unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);
        inputs.insert("k".to_string(), Tensor::new(1i64, &Device::Cpu).unwrap());

        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap().to_vec2::<f32>().unwrap();
        // upper=1, k=1 keeps j >= i+1: row0 keeps col>=1, row1 keeps col>=2.
        assert_eq!(out, vec![vec![0.0, 1.0, 2.0], vec![0.0, 0.0, 4.0]]);
    }

    // The motivating fix: a non-f32 (int32) Trilu input must not have its
    // zero-fill hardcoded to Float, which would produce a dtype-mismatched
    // `Where` node. `data`'s declared type comes from `graph.value_info`.
    #[test]
    fn trilu_non_f32_input_uses_declared_dtype_for_zero_fill() {
        let mut graph = GraphProto {
            node: vec![trilu_node("data", None, 0, "out")],
            value_info: vec![declare_tensor_type("data", DataType::Int32)],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let data = Tensor::from_vec(vec![5i64, 6, 7, 8], (2, 2), &Device::Cpu).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);

        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap();
        assert_eq!(out.dtype(), candle_core::DType::I64);
        assert_eq!(out.to_vec2::<i64>().unwrap(), vec![vec![5, 0], vec![7, 8]]);
    }

    // A graph with no Trilu nodes must pass through unchanged.
    #[test]
    fn rewrite_is_a_no_op_without_trilu_nodes() {
        let mut graph = GraphProto {
            node: vec![unary_node("Identity", "x", "y")],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Identity");
    }
}
