// SPDX-License-Identifier: MIT
//! Fuses the decomposed `atan2(y, x)` pattern into a single `Atan2` node.
//!
//! ONNX exporters targeting opsets without a native `Atan2` op (notably
//! `PyTorch`'s `torch.onnx.export`) emit the two-argument arctangent as a
//! multi-node decomposition:
//!
//! ```text
//! Div(y, x) → Atan → inner Where(Greater(y, 0), Add(atan, π), Sub(atan, π))
//!           → outer Where(Less(x, 0), inner_where, atan) → result
//! ```
//!
//! This decomposition is numerically fragile near the origin: `Div(0, 0)`
//! produces `NaN`, and the quadrant-correction `Less(x, 0)` flips
//! unpredictably when `x` is near zero — both problems that `f32::atan2`
//! handles correctly in a single call. This pass recognizes the terminal
//! `Where` node of that decomposition and replaces the entire subgraph with
//! a single `Atan2(y, x)` node. Dead intermediate nodes (`Div`, `Atan`,
//! `Less`, `Greater`, `Add`, `Sub`, inner `Where`) are left for the existing
//! DCE pass in [`super::eliminate`] to clean up.

use std::collections::HashMap;

use super::super::proto::{GraphProto, NodeProto};
use super::collect_producers;

/// Fuses every decomposed `atan2` pattern in `graph` into a single `Atan2`
/// node, returning the number of fusions performed.
///
/// Intended to run once at optimization time (before constant folding),
/// not per inference call.
pub(crate) fn fuse_atan2_decomposition(graph: &mut GraphProto) -> usize {
    let producers = collect_producers(&graph.node);
    let mut fused = 0;

    for node in &mut graph.node {
        if node.op_type == "Where" && try_fuse_where(node, &producers) {
            fused += 1;
        }
    }

    fused
}

/// Attempts to match `node` (a `Where`) against the `atan2` quadrant-
/// correction decomposition and, if it matches, rewrites `node` in place
/// to an `Atan2(y, x)` node with the same output name. Returns `true` on
/// a successful rewrite.
///
/// The matched shape (backward from the terminal `Where`):
///
/// 1. `Where(cond, true_branch, false_branch)` where `false_branch` is the
///    raw `Atan` result (the "no correction needed" path).
/// 2. `cond` is produced by `Less(x, zero)` — the quadrant check.
/// 3. `false_branch` is produced by `Atan(div_out)`.
/// 4. `div_out` is produced by `Div(y, x)` where `x` is the same tensor
///    as the `Less` node's first input.
/// 5. `true_branch` is produced by an inner `Where(inner_cond, add, sub)` —
///    the quadrant-correction branch.
/// 6. `inner_cond` is produced by `Greater(y, _)`, using the same `y` as
///    the `Div` node.
/// 7. `add` is produced by `Add` taking the `Atan` result as an input.
/// 8. `sub` is produced by `Sub` taking the `Atan` result as an input.
fn try_fuse_where(node: &mut NodeProto, producers: &HashMap<String, NodeProto>) -> bool {
    if node.input.len() != 3 || node.output.len() != 1 {
        return false;
    }
    let cond = &node.input[0];
    let true_val = &node.input[1];
    let false_val = &node.input[2];

    let Some(less) = producers.get(cond) else { return false };
    if less.op_type != "Less" || less.input.len() != 2 {
        return false;
    }
    let x = &less.input[0];

    let Some(atan) = producers.get(false_val) else { return false };
    if atan.op_type != "Atan" || atan.input.len() != 1 {
        return false;
    }
    let atan_output = &atan.output[0];

    let Some(div) = producers.get(&atan.input[0]) else { return false };
    if div.op_type != "Div" || div.input.len() != 2 || div.input[1] != *x {
        return false;
    }
    let y = &div.input[0];

    let Some(inner_where) = producers.get(true_val) else { return false };
    if inner_where.op_type != "Where" || inner_where.input.len() != 3 {
        return false;
    }

    let Some(greater) = producers.get(&inner_where.input[0]) else { return false };
    if greater.op_type != "Greater" || greater.input.len() != 2 || greater.input[0] != *y {
        return false;
    }

    let Some(add) = producers.get(&inner_where.input[1]) else { return false };
    if add.op_type != "Add" || !add.input.contains(atan_output) {
        return false;
    }

    let Some(sub) = producers.get(&inner_where.input[2]) else { return false };
    if sub.op_type != "Sub" || !sub.input.contains(atan_output) {
        return false;
    }

    node.op_type = "Atan2".to_string();
    node.input = vec![y.clone(), x.clone()];
    node.name = if node.name.is_empty() {
        "fused_atan2".to_string()
    } else {
        format!("{}/fused_atan2", node.name)
    };

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_node(op_type: &str, a: &str, b: &str, output: &str) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            input: vec![a.to_string(), b.to_string()],
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    fn unary_node(op_type: &str, input: &str, output: &str) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            input: vec![input.to_string()],
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    fn where_node(name: &str, cond: &str, t: &str, f: &str, output: &str) -> NodeProto {
        NodeProto {
            name: name.to_string(),
            op_type: "Where".to_string(),
            input: vec![cond.to_string(), t.to_string(), f.to_string()],
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    /// Builds the full `atan2(y, x)` quadrant-correction decomposition.
    fn atan2_decomposition(y: &str, x: &str, output: &str) -> Vec<NodeProto> {
        vec![
            binary_node("Div", y, x, "div"),
            unary_node("Atan", "div", "atan"),
            binary_node("Greater", y, "zero", "greater"),
            binary_node("Add", "atan", "pi", "add_pi"),
            binary_node("Sub", "atan", "pi", "sub_pi"),
            where_node("inner_where", "greater", "add_pi", "sub_pi", "where0"),
            binary_node("Less", x, "zero", "less"),
            where_node("outer_where", "less", "where0", "atan", output),
        ]
    }

    // The motivating case: the full atan2 decomposition is recognized and
    // the terminal Where is rewritten to a single Atan2(y, x) node.
    #[test]
    fn fuses_full_atan2_decomposition() {
        let mut graph = GraphProto {
            node: atan2_decomposition("imag", "real", "result"),
            ..Default::default()
        };

        let fused = fuse_atan2_decomposition(&mut graph);

        assert_eq!(fused, 1);
        let atan2_node = graph
            .node
            .iter()
            .find(|n| n.op_type == "Atan2")
            .expect("should have an Atan2 node");
        assert_eq!(atan2_node.input, vec!["imag", "real"]);
        assert_eq!(atan2_node.output, vec!["result"]);
    }

    // Unrelated Where nodes must be left completely unchanged.
    #[test]
    fn leaves_unrelated_where_unchanged() {
        let mut graph = GraphProto {
            node: vec![where_node("w", "cond", "a", "b", "y")],
            ..Default::default()
        };

        let fused = fuse_atan2_decomposition(&mut graph);

        assert_eq!(fused, 0);
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Where");
    }

    // Multiple independent atan2 decompositions in the same graph should
    // each be fused.
    #[test]
    fn fuses_multiple_decompositions() {
        let mut nodes = Vec::new();

        let first = atan2_decomposition("imag1", "real1", "result1");
        nodes.extend(first);

        nodes.push(binary_node("Div", "imag2", "real2", "div2"));
        nodes.push(unary_node("Atan", "div2", "atan2_"));
        nodes.push(binary_node("Greater", "imag2", "zero", "greater2"));
        nodes.push(binary_node("Add", "atan2_", "pi", "add_pi2"));
        nodes.push(binary_node("Sub", "atan2_", "pi", "sub_pi2"));
        nodes.push(where_node("inner2", "greater2", "add_pi2", "sub_pi2", "where02"));
        nodes.push(binary_node("Less", "real2", "zero", "less2"));
        nodes.push(where_node("outer2", "less2", "where02", "atan2_", "result2"));

        let mut graph = GraphProto { node: nodes, ..Default::default() };

        let fused = fuse_atan2_decomposition(&mut graph);
        assert_eq!(fused, 2);
    }

    // A Where node whose condition is Less but false branch is not an Atan
    // (e.g. the inner Where of the decomposition) should not be fused.
    #[test]
    fn does_not_fuse_inner_where() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Less", "x", "zero", "less"),
                binary_node("Add", "a", "b", "add"),
                where_node("w", "less", "add", "c", "y"),
            ],
            ..Default::default()
        };

        let fused = fuse_atan2_decomposition(&mut graph);
        assert_eq!(fused, 0);
    }

    // A Where node whose condition/false-branch match the outer atan2 shape
    // (Less/Atan/Div) but whose true-branch is not the inner quadrant-
    // correction Where must not be fused.
    #[test]
    fn does_not_fuse_wrong_true_branch() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Div", "y", "x", "div"),
                unary_node("Atan", "div", "atan"),
                binary_node("Less", "x", "zero", "less"),
                binary_node("Add", "unrelated_a", "unrelated_b", "not_inner_where"),
                where_node("w", "less", "not_inner_where", "atan", "result"),
            ],
            ..Default::default()
        };

        let fused = fuse_atan2_decomposition(&mut graph);
        assert_eq!(fused, 0);
    }
}
