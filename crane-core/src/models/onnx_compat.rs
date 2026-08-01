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

use std::collections::HashMap;

use anyhow::{Context, Result};
use candle_core::DType;

use crate::onnx::proto::attribute_proto::AttributeType;
use crate::onnx::proto::tensor_proto::DataType;
use crate::onnx::proto::{AttributeProto, GraphProto, NodeProto, TensorProto};

/// Whether a rewrite function actually rewrote its node in place, so
/// [`rewrite_unsupported_ops`] knows whether to also keep the original
/// node or discard it in favor of the rewrite's replacement(s).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Rewritten {
    Yes,
    No,
}

/// Rewrites every node in `graph` that unmodified `crate::onnx::eval`
/// handles incorrectly into a decomposition it handles correctly. Runs
/// once, at model load time, before the graph (or its segments) is passed
/// to `crate::onnx::simple_eval`.
pub(crate) fn rewrite_unsupported_ops(graph: &mut GraphProto) -> Result<()> {
    let constants = collect_constant_i64_values(graph);
    let orig_nodes = std::mem::take(&mut graph.node);
    let mut new_nodes = Vec::with_capacity(orig_nodes.len());

    for node in &orig_nodes {
        match node.op_type.as_str() {
            "Trilu" => expand_trilu(node, graph, &mut new_nodes)?,
            "ReduceSum" => {
                if let Rewritten::No =
                    fix_reduce_sum_negative_axes(node, graph, &constants, &mut new_nodes)
                {
                    new_nodes.push(node.clone());
                }
            },
            "ReduceMean" => {
                if let Rewritten::No = fix_reduce_mean_axes_input(node, &constants, &mut new_nodes)
                {
                    new_nodes.push(node.clone());
                }
            },
            "CumSum" => fix_int_cumsum(node, graph, &mut new_nodes),
            _ => new_nodes.push(node.clone()),
        }
    }

    graph.node = new_nodes;
    Ok(())
}

/// Collects every graph initializer and `Constant`-node output that decodes
/// to an integer tensor, flattened to `Vec<i64>` via
/// [`crate::onnx::eval::get_tensor`] (which also upcasts `Int32` to `i64`).
/// Used to resolve small integer inputs (like `axes`) to compile-time
/// constants when possible, so rewrites can fall back to a (larger, but
/// universally correct) dynamic subgraph only when genuinely needed.
fn collect_constant_i64_values(graph: &GraphProto) -> HashMap<String, Vec<i64>> {
    let mut constants = HashMap::new();
    for initializer in &graph.initializer {
        if let Some(values) = tensor_proto_to_i64_vec(initializer, &initializer.name) {
            constants.insert(initializer.name.clone(), values);
        }
    }
    for node in &graph.node {
        if node.op_type != "Constant" || node.output.len() != 1 {
            continue;
        }
        let Some(value_attr) = node.attribute.iter().find(|attr| attr.name == "value") else {
            continue;
        };
        let Some(tensor_proto) = &value_attr.t else {
            continue;
        };
        if let Some(values) = tensor_proto_to_i64_vec(tensor_proto, &node.output[0]) {
            constants.insert(node.output[0].clone(), values);
        }
    }
    constants
}

/// Decodes `tensor_proto` and flattens it to a `Vec<i64>`, or `None` if it
/// can't be decoded or isn't an integer tensor.
fn tensor_proto_to_i64_vec(tensor_proto: &TensorProto, name: &str) -> Option<Vec<i64>> {
    let tensor = crate::onnx::eval::get_tensor(tensor_proto, name).ok()?;
    let tensor = tensor.flatten_all().ok()?;
    tensor.to_dtype(DType::I64).ok()?.to_vec1::<i64>().ok()
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

/// Looks up `name`'s declared rank (number of dimensions) among `graph`'s
/// `value_info`, `input`, and `output` lists. Returns `None` when no
/// declaration is present, or the declaration has no shape at all.
fn declared_rank(graph: &GraphProto, name: &str) -> Option<usize> {
    tensor_type_info(graph, name)?
        .shape
        .as_ref()
        .map(|shape| shape.dim.len())
}

/// Rewrites a `ReduceSum` node whose `axes` input may contain negative
/// values into one whose `axes` input is guaranteed non-negative, working
/// around `eval.rs`'s `ReduceSum` casting axes directly via `x as usize`
/// with no negative-axis normalization at all (unlike `ReduceMean`, which
/// does normalize negative axes, but only in its older attribute form).
///
/// When `axes` resolves to a compile-time constant *and* `data`'s rank is
/// declared in `graph.value_info`, normalizes `axes` directly and swaps in
/// a replacement `Constant` node — cheaper, and produces a smaller graph.
/// Otherwise (axes computed dynamically elsewhere in the graph, or a
/// constant axes value whose rank isn't declared) emits a small subgraph
/// that normalizes at runtime: `fixed = axes < 0 ? axes + rank : axes`,
/// using `Size(Shape(data))` for `rank` — this covers every case, just
/// with a larger graph than the constant-resolution path needs.
///
/// Returns [`Rewritten::No`] (leaving `node` untouched) when `axes` is
/// absent, empty, or already proven non-negative by a resolved constant.
fn fix_reduce_sum_negative_axes(
    node: &NodeProto,
    graph: &GraphProto,
    constants: &HashMap<String, Vec<i64>>,
    new_nodes: &mut Vec<NodeProto>,
) -> Rewritten {
    let Some(axes_name) = node.input.get(1).filter(|name| !name.is_empty()) else {
        return Rewritten::No;
    };
    let data = &node.input[0];
    let output = &node.output[0];

    if let Some(axes_values) = constants.get(axes_name) {
        if axes_values.iter().all(|&axis| axis >= 0) {
            return Rewritten::No;
        }
        if let Some(rank) = declared_rank(graph, data) {
            #[allow(clippy::cast_possible_wrap)]
            let rank = rank as i64;
            let normalized = axes_values
                .iter()
                .map(|&axis| if axis < 0 { axis + rank } else { axis })
                .collect::<Vec<_>>();
            let fixed_axes_name = format!("{output}__onnx_compat_axes_fixed");
            #[allow(clippy::cast_possible_wrap)]
            let dims = vec![normalized.len() as i64];
            new_nodes.push(int64_constant_node(&fixed_axes_name, dims, normalized));
            let mut rewritten = node.clone();
            rewritten.input[1] = fixed_axes_name;
            new_nodes.push(rewritten);
            return Rewritten::Yes;
        }
    }

    let fixed_axes_name =
        push_dynamic_axis_normalization(data, axes_name, output, "axes", new_nodes);
    let mut rewritten = node.clone();
    rewritten.input[1] = fixed_axes_name;
    new_nodes.push(rewritten);
    Rewritten::Yes
}

/// Pushes nodes computing `axis < 0 ? axis + rank : axis` at runtime —
/// `rank` via `Size(Shape(data))` — and returns the name of the resulting
/// non-negative axis/axes tensor. Shared by every rewrite that needs to
/// normalize a negative axis input dynamically, since `data`'s rank isn't
/// known until the graph actually runs.
///
/// `label` only distinguishes this call site's intermediate tensor names
/// from another rewrite's in the same graph (e.g. `"axes"` vs `"axis"`);
/// it has no effect on the computation.
fn push_dynamic_axis_normalization(
    data: &str,
    axis_name: &str,
    output: &str,
    label: &str,
    new_nodes: &mut Vec<NodeProto>,
) -> String {
    let zero_name = format!("{output}__onnx_compat_zero_{label}");
    new_nodes.push(int64_constant_node(&zero_name, vec![], vec![0]));

    let shape_name = format!("{output}__onnx_compat_data_shape_{label}");
    new_nodes.push(unary_node("Shape", data, &shape_name));

    let rank_name = format!("{output}__onnx_compat_rank_{label}");
    new_nodes.push(unary_node("Size", &shape_name, &rank_name));

    let is_negative_name = format!("{output}__onnx_compat_{label}_negative");
    new_nodes.push(binary_node("Less", axis_name, &zero_name, &is_negative_name));

    let adjusted_name = format!("{output}__onnx_compat_{label}_adjusted");
    new_nodes.push(binary_node("Add", axis_name, &rank_name, &adjusted_name));

    let fixed_name = format!("{output}__onnx_compat_{label}_fixed");
    new_nodes.push(NodeProto {
        op_type: "Where".to_string(),
        input: vec![is_negative_name, adjusted_name, axis_name.to_string()],
        output: vec![fixed_name.clone()],
        ..Default::default()
    });
    fixed_name
}

/// Rewrites a `ReduceMean` node passing `axes` as an opset-18+ input into
/// the older attribute form, working around `eval.rs`'s `ReduceMean`
/// reading *only* the `axes` attribute and never an `axes` input at all —
/// an axes-as-input node silently reduces over every axis instead of the
/// intended ones.
///
/// Unlike [`fix_reduce_sum_negative_axes`], this can only resolve `axes`
/// to a compile-time constant: `eval.rs`'s `ReduceMean` has no path to
/// accept a non-attribute axes value, so a dynamic-subgraph fallback
/// (which still has to produce *some* form eval.rs reads) isn't possible
/// here — that's inherent to targeting unmodified `eval.rs`, not a bug in
/// this rewrite. Negative entries in `axes` don't need normalizing before
/// becoming the attribute: `eval.rs`'s `ReduceMean` already normalizes
/// negative axes in its attribute-reading path.
///
/// Known limitation inherited from `eval.rs`, not fixed here: `eval.rs`'s
/// `ReduceMean` never reads `noop_with_empty_axes` at all, so a node with
/// no axes input/attribute and `noop_with_empty_axes=1` still incorrectly
/// reduces every axis instead of being a no-op. Not fixed because doing so
/// means rewriting an absent/empty-axes node into an `Identity`, and
/// Kokoro's export always passes non-empty axes as an input for every one
/// of its `ReduceMean` nodes, so the gap doesn't manifest in practice.
///
/// Returns [`Rewritten::No`] (leaving `node` untouched) when `axes` is
/// absent, empty, or not a compile-time constant.
fn fix_reduce_mean_axes_input(
    node: &NodeProto,
    constants: &HashMap<String, Vec<i64>>,
    new_nodes: &mut Vec<NodeProto>,
) -> Rewritten {
    let Some(axes_name) = node.input.get(1).filter(|name| !name.is_empty()) else {
        return Rewritten::No;
    };
    let Some(axes_values) = constants.get(axes_name) else {
        return Rewritten::No;
    };

    let mut rewritten = node.clone();
    rewritten.input.truncate(1);
    rewritten.attribute.push(AttributeProto {
        name: "axes".to_string(),
        r#type: AttributeType::Ints as i32,
        ints: axes_values.clone(),
        ..Default::default()
    });
    new_nodes.push(rewritten);
    Rewritten::Yes
}

/// Rewrites every `CumSum` node to fix two independent gaps in `eval.rs`'s
/// `CumSum`, unconditionally (there's no "already fine" case worth
/// detecting, unlike this module's other rewrites):
///
/// - `candle_core::Tensor::cumsum` is a matmul-based implementation that
///   only supports floating-point dtypes, so an int64 `data` input fails
///   outright. Wrapping `data` in `Cast(to=Double)` before, and back to
///   its original declared dtype (or `Double`, if undeclared) after,
///   routes every dtype through the same working float path. `Double` (not
///   `Float`) matches the precision the dropped `ops/cumsum.rs`
///   implementation used, since `Float`/f32 loses exactness above 2^24 —
///   plausible for cumulative sums longer than a couple thousand terms.
/// - `axis` is an `eval.rs` *input* (not an attribute), cast via
///   `to_dtype(DType::U32)` then to `usize` with no negative-axis
///   normalization at all — the same wraparound bug
///   [`fix_reduce_sum_negative_axes`] fixes for `ReduceSum`. Always
///   normalized dynamically via [`push_dynamic_axis_normalization`], since
///   (unlike `ReduceSum`) there's no meaningfully cheaper constant-
///   resolution path worth adding just for this.
///
/// `exclusive`/`reverse` attributes, if present, are preserved verbatim on
/// the rewritten node — this rewrite only touches `data` and `axis`.
fn fix_int_cumsum(node: &NodeProto, graph: &GraphProto, new_nodes: &mut Vec<NodeProto>) {
    let data = &node.input[0];
    let axis = &node.input[1];
    let output = &node.output[0];

    let cast_in_name = format!("{output}__onnx_compat_cumsum_in");
    new_nodes.push(cast_node(data, &cast_in_name, DataType::Double));

    let fixed_axis = push_dynamic_axis_normalization(data, axis, output, "axis", new_nodes);

    let cumsum_out_name = format!("{output}__onnx_compat_cumsum_out");
    let mut rewritten = node.clone();
    rewritten.input = vec![cast_in_name, fixed_axis];
    rewritten.output = vec![cumsum_out_name.clone()];
    new_nodes.push(rewritten);

    let back_dtype = declared_dtype(graph, data)
        .and_then(|dt| DataType::try_from(dt).ok())
        .unwrap_or(DataType::Double);
    new_nodes.push(cast_node(&cumsum_out_name, output, back_dtype));
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

/// Builds a two-input, single-output node with no attributes.
fn binary_node(op_type: &str, a: &str, b: &str, output: &str) -> NodeProto {
    NodeProto {
        op_type: op_type.to_string(),
        input: vec![a.to_string(), b.to_string()],
        output: vec![output.to_string()],
        ..Default::default()
    }
}

/// Builds a `Constant` node holding an `int64` tensor. Unlike
/// [`scalar_value_attribute`] (used for `ConstantOfShape`'s `"value"`,
/// which `eval.rs` decodes via a `raw_data`-only path), `eval.rs`'s
/// `Constant` handler decodes its `"value"` attribute with
/// [`crate::onnx::eval::get_tensor`], which reads the type-specific
/// `int64_data` field directly.
fn int64_constant_node(output: &str, dims: Vec<i64>, values: Vec<i64>) -> NodeProto {
    NodeProto {
        op_type: "Constant".to_string(),
        output: vec![output.to_string()],
        attribute: vec![AttributeProto {
            name: "value".to_string(),
            r#type: AttributeType::Tensor as i32,
            t: Some(TensorProto {
                data_type: DataType::Int64 as i32,
                dims,
                int64_data: values,
                ..Default::default()
            }),
            ..Default::default()
        }],
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

    fn reduce_sum_node(data: &str, axes: &str, output: &str) -> NodeProto {
        NodeProto {
            op_type: "ReduceSum".to_string(),
            input: vec![data.to_string(), axes.to_string()],
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    fn int64_initializer(name: &str, dims: Vec<i64>, values: Vec<i64>) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: DataType::Int64 as i32,
            dims,
            int64_data: values,
            ..Default::default()
        }
    }

    fn declare_rank(name: &str, rank: usize) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(crate::onnx::proto::TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: DataType::Float as i32,
                    shape: Some(crate::onnx::proto::TensorShapeProto {
                        dim: vec![Default::default(); rank],
                    }),
                    ..Default::default()
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn data_2x3() -> Tensor {
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &Device::Cpu).unwrap()
    }

    // Kokoro's actual case: a constant negative axis (`[-1]`), with data's
    // rank declared in value_info, takes the cheap constant-resolution
    // path (a replacement `Constant` node, not a dynamic subgraph).
    #[test]
    fn reduce_sum_constant_negative_axis_with_declared_rank_uses_constant_path() {
        let mut graph = GraphProto {
            node: vec![reduce_sum_node("data", "axes", "out")],
            initializer: vec![int64_initializer("axes", vec![1], vec![-1])],
            value_info: vec![declare_rank("data", 2)],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        // The cheap path swaps in a `Constant` node instead of the dynamic
        // Shape/Size/Less/Add/Where subgraph.
        assert!(
            graph
                .node
                .iter()
                .any(|n| n.op_type == "Constant" && !n.output.is_empty())
        );
        assert!(!graph.node.iter().any(|n| n.op_type == "Shape"));

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data_2x3());
        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![6.0], vec![15.0]]);
    }

    // A constant, already non-negative axis must be left untouched.
    #[test]
    fn reduce_sum_constant_positive_axis_is_a_no_op() {
        let mut graph = GraphProto {
            node: vec![reduce_sum_node("data", "axes", "out")],
            initializer: vec![int64_initializer("axes", vec![1], vec![1])],
            value_info: vec![declare_rank("data", 2)],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "ReduceSum");
        assert_eq!(graph.node[0].input[1], "axes");
    }

    // A dynamic (non-constant) axes input containing a negative value must
    // still be normalized correctly, proving the runtime Shape/Size/Less/
    // Add/Where subgraph — not just the constant-resolution path — works.
    #[test]
    fn reduce_sum_dynamic_negative_axis_is_normalized_at_runtime() {
        let mut graph = GraphProto {
            node: vec![reduce_sum_node("data", "axes", "out")],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        // No declared rank and no constant axes forces the dynamic path.
        assert!(graph.node.iter().any(|n| n.op_type == "Shape"));

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data_2x3());
        inputs.insert(
            "axes".to_string(),
            Tensor::from_vec(vec![-1i64], 1, &Device::Cpu).unwrap(),
        );
        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![6.0], vec![15.0]]);
    }

    fn reduce_mean_node(data: &str, axes: Option<&str>, output: &str) -> NodeProto {
        let mut input = vec![data.to_string()];
        if let Some(axes) = axes {
            input.push(axes.to_string());
        }
        NodeProto {
            op_type: "ReduceMean".to_string(),
            input,
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    // Kokoro's actual case: a constant positive axes input must be
    // converted to the attribute form eval.rs's ReduceMean actually reads.
    #[test]
    fn reduce_mean_constant_positive_axis_input_becomes_attribute() {
        let mut graph = GraphProto {
            node: vec![reduce_mean_node("data", Some("axes"), "out")],
            initializer: vec![int64_initializer("axes", vec![1], vec![1])],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].input.len(), 1);
        assert!(graph.node[0].attribute.iter().any(|a| a.name == "axes"));

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data_2x3());
        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![2.0], vec![5.0]]);
    }

    // A constant negative axes input must also convert correctly — eval.rs
    // already normalizes negative axes once they're in the attribute form.
    #[test]
    fn reduce_mean_constant_negative_axis_input_becomes_attribute() {
        let mut graph = GraphProto {
            node: vec![reduce_mean_node("data", Some("axes"), "out")],
            initializer: vec![int64_initializer("axes", vec![1], vec![-1])],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data_2x3());
        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![2.0], vec![5.0]]);
    }

    // A ReduceMean with no axes input at all (attribute-only form, or a
    // deliberate full reduction) must be left untouched.
    #[test]
    fn reduce_mean_without_axes_input_is_a_no_op() {
        let mut graph = GraphProto {
            node: vec![reduce_mean_node("data", None, "out")],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].input.len(), 1);
        assert!(graph.node[0].attribute.is_empty());
    }

    // A non-constant axes input can't be resolved at rewrite time, since
    // eval.rs's ReduceMean has no path to accept axes as anything but an
    // attribute — the node must be left untouched (conservative, documented
    // limitation) rather than guessing.
    #[test]
    fn reduce_mean_non_constant_axes_input_is_a_no_op() {
        let mut graph = GraphProto {
            node: vec![reduce_mean_node("data", Some("axes"), "out")],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].input, vec!["data".to_string(), "axes".to_string()]);
    }

    fn cumsum_node(
        data: &str,
        axis: &str,
        output: &str,
        exclusive: Option<i64>,
        reverse: Option<i64>,
    ) -> NodeProto {
        let mut attribute = vec![];
        if let Some(value) = exclusive {
            attribute.push(AttributeProto {
                name: "exclusive".to_string(),
                r#type: AttributeType::Int as i32,
                i: value,
                ..Default::default()
            });
        }
        if let Some(value) = reverse {
            attribute.push(AttributeProto {
                name: "reverse".to_string(),
                r#type: AttributeType::Int as i32,
                i: value,
                ..Default::default()
            });
        }
        NodeProto {
            op_type: "CumSum".to_string(),
            input: vec![data.to_string(), axis.to_string()],
            output: vec![output.to_string()],
            attribute,
            ..Default::default()
        }
    }

    // Kokoro's actual case: int64 data with a negative axis. Both bugs
    // (float-only cumsum, un-normalized negative axis) are exercised at
    // once, and the output must come back as int64, not float.
    #[test]
    fn cumsum_int64_negative_axis_produces_correct_int64_output() {
        let mut graph = GraphProto {
            node: vec![cumsum_node("data", "axis", "out", None, None)],
            value_info: vec![declare_tensor_type("data", DataType::Int64)],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let data = Tensor::from_vec(vec![1i64, 2, 3, 4, 5, 6], (2, 3), &Device::Cpu).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);
        inputs.insert("axis".to_string(), Tensor::new(-1i64, &Device::Cpu).unwrap());

        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap();
        assert_eq!(out.dtype(), candle_core::DType::I64);
        assert_eq!(out.to_vec2::<i64>().unwrap(), vec![vec![1, 3, 6], vec![4, 9, 15]]);
    }

    // Float data with a positive axis must round-trip through the Double
    // intermediate cast without changing the result or the output dtype.
    #[test]
    fn cumsum_float_data_round_trips_through_double_precision() {
        let mut graph = GraphProto {
            node: vec![cumsum_node("data", "axis", "out", None, None)],
            value_info: vec![declare_tensor_type("data", DataType::Float)],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let data =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &Device::Cpu).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);
        inputs.insert("axis".to_string(), Tensor::new(1i64, &Device::Cpu).unwrap());

        let values = run_graph(graph, inputs);
        let out = values.get("out").unwrap();
        assert_eq!(out.dtype(), candle_core::DType::F32);
        assert_eq!(out.to_vec2::<f32>().unwrap(), vec![vec![1.0, 3.0, 6.0], vec![
            4.0, 9.0, 15.0
        ]]);
    }

    // An `exclusive`/`reverse` attribute must survive the rewrite verbatim
    // rather than being silently dropped — eval.rs still rejects
    // `exclusive != 0` explicitly, so the error proves the attribute made
    // it onto the rewritten node instead of being lost.
    #[test]
    fn cumsum_exclusive_attribute_is_preserved_and_still_rejected() {
        let mut graph = GraphProto {
            node: vec![cumsum_node("data", "axis", "out", Some(1), None)],
            output: vec![ValueInfoProto {
                name: "out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        rewrite_unsupported_ops(&mut graph).unwrap();

        let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &Device::Cpu).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);
        inputs.insert("axis".to_string(), Tensor::new(0i64, &Device::Cpu).unwrap());

        let model = ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        let err = crate::onnx::simple_eval(&model, inputs).unwrap_err();
        assert!(err.to_string().contains("exclusive"));
    }
}
