// SPDX-License-Identifier: MIT
//! Fuses the decomposed BigVGAN-style `Snake` periodic activation into a
//! single `Snake` node.
//!
//! ONNX exporters emit `snake(x, alpha) = x + sin(alpha * x)^2 / alpha`
//! (Liu et al., used in Kokoro's HiFiGAN-style vocoder decoder) as a five-op
//! decomposition:
//!
//! ```text
//! Mul(alpha, x) → Sin → Pow(_, 2) → Mul(inv_alpha, _) → Add(x, _) → result
//! ```
//!
//! Each op is a full read-and-write pass over the whole tensor; on the
//! decoder's largest resblocks this becomes memory-bandwidth-bound once the
//! intermediate tensors exceed cache, causing synthesis time to blow up
//! superlinearly with input length (see `ONNX_SPEEDUP.md`). This pass
//! recognizes the terminal `Add` node of that decomposition and replaces the
//! entire subgraph with a single `Snake(x, alpha)` node, evaluated by the
//! single-pass `CustomOp2` kernel in [`crate::ops::fused_ops::snake`]. Dead
//! intermediate nodes (`Mul`, `Sin`, `Pow`) are left for the existing DCE
//! pass in [`super::eliminate`] to clean up.

use std::collections::HashMap;

use super::super::proto::{GraphProto, NodeProto};
use super::collect_producers;

/// Fuses every decomposed `Snake` pattern in `graph` into a single `Snake`
/// node, returning the number of fusions performed.
///
/// Intended to run once at optimization time (before constant folding),
/// not per inference call.
pub(crate) fn fuse_snake_decomposition(graph: &mut GraphProto) -> usize {
    let producers = collect_producers(&graph.node);
    let mut fused = 0;

    for node in &mut graph.node {
        if node.op_type == "Add" && try_fuse_add(node, &producers) {
            fused += 1;
        }
    }

    fused
}

/// Given a commutative binary node's two inputs, returns
/// `(matched_input, other_input)` where `matched_input` is produced by a
/// node of type `op_type`. Tries both operand positions since exporters may
/// emit either order, returning the first match if both operands happen to
/// qualify. Returns `None` if neither input matches or `inputs` isn't
/// exactly 2 elements.
fn find_producer_input<'a>(
    inputs: &'a [String],
    producers: &HashMap<String, NodeProto>,
    op_type: &str,
) -> Option<(&'a String, &'a String)> {
    let [a, b] = inputs else { return None };
    if producers.get(a).is_some_and(|p| p.op_type == op_type) {
        return Some((a, b));
    }
    if producers.get(b).is_some_and(|p| p.op_type == op_type) {
        return Some((b, a));
    }
    None
}

/// Given a commutative binary node's two inputs, returns the input that
/// isn't `target`, or `None` if `target` isn't among exactly 2 inputs.
fn other_input<'a>(inputs: &'a [String], target: &str) -> Option<&'a String> {
    let [a, b] = inputs else { return None };
    if a == target {
        return Some(b);
    }
    if b == target {
        return Some(a);
    }
    None
}

/// Attempts to match `node` (an `Add`) against the `Snake` decomposition
/// and, if it matches, rewrites `node` in place to a `Snake(x, alpha)` node
/// with the same output name. Returns `true` on a successful rewrite.
///
/// The matched shape (backward from the terminal `Add`, trying both operand
/// orders since `Add`/`Mul` are commutative):
///
/// 1. `Add(x, mul2_out)` — `x` is one operand, the other is the tail of the
///    `sin^2/alpha` chain.
/// 2. `mul2_out` is produced by `Mul(inv_alpha, pow_out)`.
/// 3. `pow_out` is produced by `Pow(sin_out, exponent)`.
/// 4. `sin_out` is produced by `Sin(mul1_out)`.
/// 5. `mul1_out` is produced by `Mul(alpha, x)` — one operand is the same
///    `x` tensor as the outer `Add`'s.
fn try_fuse_add(node: &mut NodeProto, producers: &HashMap<String, NodeProto>) -> bool {
    let [in0, in1] = node.input.as_slice() else { return false };
    if node.output.len() != 1 {
        return false;
    }

    let matched = try_match_chain(in0, in1, producers).or_else(|| try_match_chain(in1, in0, producers));
    let Some((x, alpha)) = matched else { return false };

    node.op_type = "Snake".to_string();
    node.input = vec![x, alpha];
    node.name = if node.name.is_empty() {
        "fused_snake".to_string()
    } else {
        format!("{}/fused_snake", node.name)
    };
    true
}

/// Walks backward from `mul2_name` (the `Add`'s non-`x` operand) through
/// `Mul → Pow → Sin → Mul` looking for a `Mul` whose operands are `alpha`
/// and `x`. `inv_alpha`, the `Pow` exponent, and `alpha` itself must each be
/// initializers/constants (absent from `producers`, i.e. not produced by
/// any node) rather than dynamically computed — the fused kernel hardcodes
/// both the square and the division by `alpha`, so a dynamic exponent or an
/// `inv_alpha` unrelated to `alpha` would silently compute the wrong
/// result. Returns `(x, alpha)` on a match.
fn try_match_chain(
    x: &str,
    mul2_name: &str,
    producers: &HashMap<String, NodeProto>,
) -> Option<(String, String)> {
    let mul2 = producers.get(mul2_name)?;
    if mul2.op_type != "Mul" {
        return None;
    }
    let (pow_name, inv_alpha) = find_producer_input(&mul2.input, producers, "Pow")?;
    if producers.contains_key(inv_alpha) {
        return None;
    }

    let pow = &producers[pow_name];
    let (sin_name, exponent) = find_producer_input(&pow.input, producers, "Sin")?;
    if producers.contains_key(exponent) {
        return None;
    }

    let sin = &producers[sin_name];
    let [mul1_name] = sin.input.as_slice() else { return None };

    let mul1 = producers.get(mul1_name)?;
    if mul1.op_type != "Mul" {
        return None;
    }
    let alpha = other_input(&mul1.input, x)?;
    if producers.contains_key(alpha) {
        return None;
    }

    Some((x.to_string(), alpha.clone()))
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

    /// Builds the full `snake(x, alpha)` decomposition:
    /// `Mul(alpha, x) -> Sin -> Pow(_, 2) -> Mul(inv_alpha, _) -> Add(x, _)`.
    /// `suffix` disambiguates intermediate tensor names across multiple
    /// independent instances in the same graph.
    fn snake_decomposition(
        x: &str,
        alpha: &str,
        inv_alpha: &str,
        output: &str,
        suffix: &str,
    ) -> Vec<NodeProto> {
        vec![
            binary_node("Mul", alpha, x, &format!("mul1{suffix}")),
            unary_node("Sin", &format!("mul1{suffix}"), &format!("sin{suffix}")),
            binary_node("Pow", &format!("sin{suffix}"), "two", &format!("pow{suffix}")),
            binary_node("Mul", inv_alpha, &format!("pow{suffix}"), &format!("mul2{suffix}")),
            binary_node("Add", x, &format!("mul2{suffix}"), output),
        ]
    }

    // The motivating case: the full Snake decomposition is recognized and
    // the terminal Add is rewritten to a single Snake(x, alpha) node.
    #[test]
    fn fuses_full_snake_decomposition() {
        let mut graph = GraphProto {
            node: snake_decomposition("x", "alpha", "inv_alpha", "result", ""),
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);

        assert_eq!(fused, 1);
        let snake_node = graph
            .node
            .iter()
            .find(|n| n.op_type == "Snake")
            .expect("should have a Snake node");
        assert_eq!(snake_node.input, vec!["x", "alpha"]);
        assert_eq!(snake_node.output, vec!["result"]);
    }

    // Exporters may emit the Add/Mul operands in either order since both
    // ops are commutative; the reversed order must fuse identically.
    #[test]
    fn fuses_with_reversed_commutative_operands() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "x", "alpha", "mul1"),
                unary_node("Sin", "mul1", "sin"),
                binary_node("Pow", "sin", "two", "pow"),
                binary_node("Mul", "pow", "inv_alpha", "mul2"),
                binary_node("Add", "mul2", "x", "result"),
            ],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);

        assert_eq!(fused, 1);
        let snake_node = graph
            .node
            .iter()
            .find(|n| n.op_type == "Snake")
            .expect("should have a Snake node");
        assert_eq!(snake_node.input, vec!["x", "alpha"]);
    }

    // Unrelated Add nodes must be left completely unchanged.
    #[test]
    fn leaves_unrelated_add_unchanged() {
        let mut graph = GraphProto {
            node: vec![binary_node("Add", "a", "b", "y")],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);

        assert_eq!(fused, 0);
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Add");
    }

    // Multiple independent Snake decompositions in the same graph should
    // each be fused.
    #[test]
    fn fuses_multiple_decompositions() {
        let mut nodes = snake_decomposition("x1", "alpha1", "inv_alpha1", "result1", "_a");
        nodes.extend(snake_decomposition("x2", "alpha2", "inv_alpha2", "result2", "_b"));

        let mut graph = GraphProto { node: nodes, ..Default::default() };

        let fused = fuse_snake_decomposition(&mut graph);
        assert_eq!(fused, 2);
    }

    // A Pow node whose non-exponent input is not a Sin output must not be
    // fused — the chain is broken.
    #[test]
    fn does_not_fuse_wrong_pow_input() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "alpha", "x", "mul1"),
                binary_node("Add", "mul1", "other", "not_sin"),
                binary_node("Pow", "not_sin", "two", "pow"),
                binary_node("Mul", "inv_alpha", "pow", "mul2"),
                binary_node("Add", "x", "mul2", "result"),
            ],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);
        assert_eq!(fused, 0);
    }

    // The innermost Mul must actually reference the same `x` tensor as the
    // outer Add — a decomposition for a *different* tensor must not fuse.
    #[test]
    fn does_not_fuse_when_x_mismatch() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "alpha", "unrelated_x", "mul1"),
                unary_node("Sin", "mul1", "sin"),
                binary_node("Pow", "sin", "two", "pow"),
                binary_node("Mul", "inv_alpha", "pow", "mul2"),
                binary_node("Add", "x", "mul2", "result"),
            ],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);
        assert_eq!(fused, 0);
    }

    // The Pow exponent must be a constant (absent from producers), not a
    // dynamically computed value — the fused kernel hardcodes the square.
    #[test]
    fn does_not_fuse_dynamic_exponent() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "alpha", "x", "mul1"),
                unary_node("Sin", "mul1", "sin"),
                unary_node("Identity", "dynamic_source", "two"),
                binary_node("Pow", "sin", "two", "pow"),
                binary_node("Mul", "inv_alpha", "pow", "mul2"),
                binary_node("Add", "x", "mul2", "result"),
            ],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);
        assert_eq!(fused, 0);
    }

    // inv_alpha must be a constant (absent from producers), not derived
    // from an unrelated dynamic computation — the fused kernel hardcodes
    // division by `alpha`, so a dynamic inv_alpha could silently diverge.
    #[test]
    fn does_not_fuse_dynamic_inv_alpha() {
        let mut graph = GraphProto {
            node: vec![
                binary_node("Mul", "alpha", "x", "mul1"),
                unary_node("Sin", "mul1", "sin"),
                binary_node("Pow", "sin", "two", "pow"),
                unary_node("Identity", "dynamic_source", "inv_alpha"),
                binary_node("Mul", "inv_alpha", "pow", "mul2"),
                binary_node("Add", "x", "mul2", "result"),
            ],
            ..Default::default()
        };

        let fused = fuse_snake_decomposition(&mut graph);
        assert_eq!(fused, 0);
    }
}
