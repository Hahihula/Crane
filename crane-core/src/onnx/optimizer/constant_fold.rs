use std::collections::HashMap;

use candle_core::{Result, Tensor};

use super::super::{eval, proto};

const FOLDABLE_OPERATORS: &[&str] = &[
    "Add",
    "Cast",
    "Concat",
    "Constant",
    "Div",
    "Gather",
    "Identity",
    "Mul",
    "Reshape",
    "Shape",
    "Size",
    "Slice",
    "Squeeze",
    "Sub",
    "Transpose",
    "Unsqueeze",
];

pub(super) fn fold_constants(
    graph: &mut proto::GraphProto,
    constants: &mut HashMap<String, Tensor>,
    max_folded_elements: usize,
) -> Result<usize> {
    let mut folded = 0;
    let mut remaining = Vec::with_capacity(graph.node.len());

    for node in std::mem::take(&mut graph.node) {
        let can_fold = is_standard_domain(&node.domain)
            && FOLDABLE_OPERATORS.contains(&node.op_type.as_str())
            && node
                .input
                .iter()
                .all(|name| name.is_empty() || constants.contains_key(name))
            && node.output.iter().all(|name| !name.is_empty());
        if !can_fold {
            remaining.push(node);
            continue;
        }

        let inputs = node
            .input
            .iter()
            .filter(|name| !name.is_empty())
            .filter_map(|name| {
                constants
                    .get(name)
                    .map(|tensor| (name.clone(), tensor.clone()))
            })
            .collect::<HashMap<_, _>>();
        let model = proto::ModelProto {
            graph: Some(proto::GraphProto {
                node: vec![node.clone()],
                output: node
                    .output
                    .iter()
                    .map(|name| proto::ValueInfoProto {
                        name: name.clone(),
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let values = match eval::simple_eval(&model, inputs) {
            Ok(values) => values,
            Err(_) => {
                remaining.push(node);
                continue;
            },
        };
        let Some(outputs) = node
            .output
            .iter()
            .map(|name| {
                values
                    .get(name)
                    .map(|tensor| (name.clone(), tensor.clone()))
            })
            .collect::<Option<Vec<_>>>()
        else {
            remaining.push(node);
            continue;
        };
        let Some(element_count) = outputs.iter().try_fold(0usize, |total, (_, tensor)| {
            total.checked_add(tensor.elem_count())
        }) else {
            remaining.push(node);
            continue;
        };
        if element_count > max_folded_elements {
            remaining.push(node);
            continue;
        }

        constants.extend(outputs);
        folded += 1;
    }

    graph.node = remaining;
    Ok(folded)
}

fn is_standard_domain(domain: &str) -> bool {
    domain.is_empty() || domain == "ai.onnx"
}
