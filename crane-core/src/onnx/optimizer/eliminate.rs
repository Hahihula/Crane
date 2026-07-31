use std::collections::{HashMap, HashSet};

use super::super::proto::{GraphProto, NodeProto};

pub(super) fn eliminate_alias_nodes(graph: &mut GraphProto) -> usize {
    let public_outputs = graph
        .output
        .iter()
        .map(|output| output.name.as_str())
        .collect::<HashSet<_>>();
    let mut aliases = HashMap::<String, String>::new();
    let mut remaining = Vec::with_capacity(graph.node.len());
    let mut removed = 0;

    for mut node in std::mem::take(&mut graph.node) {
        for input in &mut node.input {
            *input = resolve_alias(input, &aliases);
        }
        let alias = alias_input(&node);
        let output = node.output.first();
        if let (Some(input), Some(output)) = (alias, output)
            && node.output.len() == 1
            && !output.is_empty()
            && !public_outputs.contains(output.as_str())
        {
            aliases.insert(output.clone(), resolve_alias(input, &aliases));
            removed += 1;
        } else {
            remaining.push(node);
        }
    }

    graph.node = remaining;
    removed
}

pub(super) fn contains_subgraphs(graph: &GraphProto) -> bool {
    graph.node.iter().any(|node| {
        node.attribute
            .iter()
            .any(|attribute| attribute.g.is_some() || !attribute.graphs.is_empty())
    })
}

pub(super) fn eliminate_dead_nodes(graph: &mut GraphProto) -> usize {
    let mut needed = graph
        .output
        .iter()
        .map(|output| output.name.clone())
        .collect::<HashSet<_>>();
    let mut keep = vec![false; graph.node.len()];

    for (index, node) in graph.node.iter().enumerate().rev() {
        if node.output.iter().any(|output| needed.contains(output)) {
            keep[index] = true;
            needed.extend(node.input.iter().filter(|name| !name.is_empty()).cloned());
        }
    }

    let before = graph.node.len();
    graph.node = std::mem::take(&mut graph.node)
        .into_iter()
        .zip(keep)
        .filter_map(|(node, keep)| keep.then_some(node))
        .collect();
    before - graph.node.len()
}

fn alias_input(node: &NodeProto) -> Option<&str> {
    if !is_standard_domain(&node.domain) {
        return None;
    }
    match node.op_type.as_str() {
        "Identity" if node.input.len() == 1 => Some(node.input[0].as_str()),
        "Concat" => {
            let mut inputs = node.input.iter().filter(|input| !input.is_empty());
            let input = inputs.next()?;
            inputs.next().is_none().then_some(input.as_str())
        },
        "Transpose" if identity_permutation(node) => Some(node.input.first()?.as_str()),
        _ => None,
    }
}

fn identity_permutation(node: &NodeProto) -> bool {
    let Some(permutation) = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == "perm")
    else {
        return false;
    };
    !permutation.ints.is_empty()
        && permutation
            .ints
            .iter()
            .enumerate()
            .all(|(axis, &value)| value == axis as i64)
}

fn resolve_alias(name: &str, aliases: &HashMap<String, String>) -> String {
    let mut resolved = name;
    while let Some(next) = aliases.get(resolved) {
        resolved = next;
    }
    resolved.to_string()
}

fn is_standard_domain(domain: &str) -> bool {
    domain.is_empty() || domain == "ai.onnx"
}
