// SPDX-License-Identifier: MIT

//! Crane Added 20260804: evaluator-loop-wide bookkeeping for ONNX subgraph
//! (e.g. "If" branch) value capture and eviction, shared by `eval.rs`'s
//! main per-node cleanup loop and by its `"If"` op handler.

use super::eval::Value;
use super::proto::GraphProto;
use std::collections::{HashMap, HashSet};

// Crane Added 20260804: every name a subgraph's own nodes reference as an
// input but don't themselves produce — i.e. a value captured from whatever
// scope the subgraph is nested in, per ONNX subgraph scoping rules.
// Occurrences are not deduplicated: a name referenced N times contributes N
// entries, matching how the caller's `remaining_uses` counts every
// occurrence individually.
pub(crate) fn captured_names(subgraph: &GraphProto) -> Vec<&str> {
    let locally_produced: HashSet<&str> =
        subgraph.node.iter().flat_map(|node| node.output.iter().map(String::as_str)).collect();
    subgraph
        .node
        .iter()
        .flat_map(|node| node.input.iter())
        .filter(|input| !input.is_empty())
        .map(String::as_str)
        .filter(|input| !locally_produced.contains(input))
        .collect()
}

// Crane Added 20260806: every captured reference (see `captured_names`) in
// `subgraph` itself plus every subgraph nested anywhere inside it (e.g. an
// inner "If" node's then_branch/else_branch). This is the single traversal
// shared by both the up-front counting in `count_nested_subgraph_captures`
// and the post-run release in the "If" op handler, so a name counted for a
// given subgraph tree is always released against that same tree.
pub(crate) fn collect_all_captures<'a>(subgraph: &'a GraphProto, result: &mut Vec<&'a str>) {
    result.extend(captured_names(subgraph));
    for node in &subgraph.node {
        for attribute in &node.attribute {
            let Some(nested) = &attribute.g else { continue };
            collect_all_captures(nested, result);
        }
    }
}

// Crane Added 20260804: recursively adds, into `counts`, one entry per
// captured reference (see `captured_names`) in every subgraph nested
// anywhere under `graph` (e.g. an "If" node's then_branch/else_branch,
// including branches nested inside other branches). Counted once per
// subgraph a capture appears in, even across mutually-exclusive branches of
// the same "If" (at most one of which actually executes) — an intentional
// over-count, never an under-count, since which branch gets taken isn't
// known until evaluation. The "If" op handler releases the matching count
// for both the taken branch (once it has run) and the untaken branch (since
// it never runs at all) via the same `collect_all_captures` traversal used
// here, so every increment added below has exactly one release.
pub(crate) fn count_nested_subgraph_captures<'a>(
    graph: &'a GraphProto,
    counts: &mut HashMap<&'a str, usize>,
) {
    for node in &graph.node {
        for attribute in &node.attribute {
            let Some(subgraph) = &attribute.g else { continue };
            let mut captures = Vec::new();
            collect_all_captures(subgraph, &mut captures);
            for name in captures {
                *counts.entry(name).or_default() += 1;
            }
        }
    }
}

// Crane Added 20260804: decrements `remaining_uses` for each of `names`,
// evicting a value from `values` once its count reaches zero — unless it's
// a graph output or was inherited from an enclosing scope. Shared by the
// per-node cleanup loop in `simple_eval_` and by "If"'s handling of its
// taken branch's captured names (see `captured_names`), so both paths
// agree on exactly when a value is safe to free.
pub(crate) fn release_names_if_done<'a>(
    names: impl IntoIterator<Item = &'a str>,
    remaining_uses: &mut HashMap<&'a str, usize>,
    graph_outputs: &HashSet<&str>,
    inherited_values: &HashSet<String>,
    values: &mut HashMap<String, Value>,
) {
    for name in names {
        if let Some(count) = remaining_uses.get_mut(name) {
            *count -= 1;
            if *count == 0 && !graph_outputs.contains(name) && !inherited_values.contains(name) {
                values.remove(name);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Device, Result};

    use super::super::eval::{simple_eval, simple_eval_};
    use super::Value;
    use crate::onnx::proto::attribute_proto::AttributeType;
    use crate::onnx::proto::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};

    #[test]
    fn if_branch_does_not_evict_value_outer_graph_still_needs() -> Result<()> {
        // Regression test: an "If" branch recursively calls simple_eval_
        // sharing the same `values` map as the enclosing graph. Its own
        // last-use cleanup is scoped only to the branch's own node list, so
        // without the inherited_values guard it would free an outer-scope
        // value ("x") the moment the branch's *local* view of its uses hits
        // zero — even though a node after the "If" still needs it.
        let then_branch = GraphProto {
            node: vec![NodeProto {
                op_type: "Identity".to_string(),
                input: vec!["x".to_string()],
                output: vec!["branch_out".to_string()],
                ..Default::default()
            }],
            output: vec![ValueInfoProto { name: "branch_out".to_string(), ..Default::default() }],
            ..Default::default()
        };

        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![
                    ValueInfoProto { name: "x".to_string(), ..Default::default() },
                    ValueInfoProto { name: "cond".to_string(), ..Default::default() },
                ],
                node: vec![
                    NodeProto {
                        op_type: "If".to_string(),
                        input: vec!["cond".to_string()],
                        output: vec!["if_out".to_string()],
                        attribute: vec![AttributeProto {
                            name: "then_branch".to_string(),
                            r#type: AttributeType::Graph as i32,
                            g: Some(then_branch),
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                    // Runs after the "If" node and needs "x" again — this is
                    // exactly what the buggy version evicted out from under.
                    NodeProto {
                        op_type: "Identity".to_string(),
                        input: vec!["x".to_string()],
                        output: vec!["y2".to_string()],
                        ..Default::default()
                    },
                ],
                output: vec![
                    ValueInfoProto { name: "if_out".to_string(), ..Default::default() },
                    ValueInfoProto { name: "y2".to_string(), ..Default::default() },
                ],
                ..Default::default()
            }),
            ..Default::default()
        };

        let x = Value::new(&[1f32, 2., 3.], &Device::Cpu)?;
        let cond = Value::new(&[1u8], &Device::Cpu)?;
        let outputs = simple_eval(
            &model,
            [("x".to_string(), x), ("cond".to_string(), cond)].into(),
        )?;

        assert_eq!(outputs["if_out"].to_vec1::<f32>()?, vec![1., 2., 3.]);
        assert_eq!(outputs["y2"].to_vec1::<f32>()?, vec![1., 2., 3.]);
        Ok(())
    }

    #[test]
    fn if_branch_capture_survives_an_earlier_top_level_consumer() -> Result<()> {
        // Regression test for the actual production bug (not just the
        // inherited_values guard in eval.rs): a value ("x") produced early,
        // with exactly one *top-level* consumer, plus an "If" branch that
        // references it purely by outer-scope capture — never as a direct
        // input to the "If" node itself, per ONNX subgraph scoping rules.
        // The flat scan building `remaining_uses` can't see that capture at
        // all, so without count_nested_subgraph_captures, x's count hits
        // zero (and gets evicted) right after its one top-level consumer
        // runs — well before the "If" node, which also needs it, is ever
        // reached.
        let then_branch = GraphProto {
            node: vec![NodeProto {
                op_type: "Identity".to_string(),
                input: vec!["x".to_string()],
                output: vec!["branch_out".to_string()],
                ..Default::default()
            }],
            output: vec![ValueInfoProto { name: "branch_out".to_string(), ..Default::default() }],
            ..Default::default()
        };

        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![
                    ValueInfoProto { name: "raw".to_string(), ..Default::default() },
                    ValueInfoProto { name: "cond".to_string(), ..Default::default() },
                ],
                node: vec![
                    NodeProto {
                        op_type: "Identity".to_string(),
                        input: vec!["raw".to_string()],
                        output: vec!["x".to_string()],
                        ..Default::default()
                    },
                    // x's only *top-level* consumer — its count would hit
                    // zero here under the buggy (pre-fix) counting.
                    NodeProto {
                        op_type: "Identity".to_string(),
                        input: vec!["x".to_string()],
                        output: vec!["discard".to_string()],
                        ..Default::default()
                    },
                    // "x" is never listed as this node's own input — only
                    // referenced inside then_branch by outer-scope capture.
                    NodeProto {
                        op_type: "If".to_string(),
                        input: vec!["cond".to_string()],
                        output: vec!["if_out".to_string()],
                        attribute: vec![AttributeProto {
                            name: "then_branch".to_string(),
                            r#type: AttributeType::Graph as i32,
                            g: Some(then_branch),
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                output: vec![ValueInfoProto { name: "if_out".to_string(), ..Default::default() }],
                ..Default::default()
            }),
            ..Default::default()
        };

        let raw = Value::new(&[1f32, 2., 3.], &Device::Cpu)?;
        let cond = Value::new(&[1u8], &Device::Cpu)?;
        let outputs = simple_eval(
            &model,
            [("raw".to_string(), raw), ("cond".to_string(), cond)].into(),
        )?;

        assert_eq!(outputs["if_out"].to_vec1::<f32>()?, vec![1., 2., 3.]);
        Ok(())
    }

    #[test]
    fn if_branch_releases_capture_shared_by_both_branches() -> Result<()> {
        // Regression test: count_nested_subgraph_captures deliberately
        // over-counts a name captured by *both* then_branch and
        // else_branch, since which one is taken isn't known until
        // evaluation. The "If" handler must release the untaken branch's
        // share of that over-count too, not just the taken branch's — else
        // the count never reaches zero and "x" is retained in `values` for
        // the rest of evaluation even though nothing needs it anymore.
        // Calls simple_eval_ directly (rather than the public simple_eval
        // wrapper) so the shared `values` map can be inspected after the
        // call to confirm "x" was actually evicted.
        let then_branch = GraphProto {
            node: vec![NodeProto {
                op_type: "Identity".to_string(),
                input: vec!["x".to_string()],
                output: vec!["branch_out".to_string()],
                ..Default::default()
            }],
            output: vec![ValueInfoProto { name: "branch_out".to_string(), ..Default::default() }],
            ..Default::default()
        };
        let else_branch = GraphProto {
            node: vec![NodeProto {
                op_type: "Identity".to_string(),
                input: vec!["x".to_string()],
                output: vec!["branch_out".to_string()],
                ..Default::default()
            }],
            output: vec![ValueInfoProto { name: "branch_out".to_string(), ..Default::default() }],
            ..Default::default()
        };

        let graph = GraphProto {
            input: vec![
                ValueInfoProto { name: "raw".to_string(), ..Default::default() },
                ValueInfoProto { name: "cond".to_string(), ..Default::default() },
            ],
            node: vec![
                NodeProto {
                    op_type: "Identity".to_string(),
                    input: vec!["raw".to_string()],
                    output: vec!["x".to_string()],
                    ..Default::default()
                },
                // x's only *top-level* consumer.
                NodeProto {
                    op_type: "Identity".to_string(),
                    input: vec!["x".to_string()],
                    output: vec!["discard".to_string()],
                    ..Default::default()
                },
                // "x" is never listed as this node's own input — both
                // branches reference it purely by outer-scope capture.
                NodeProto {
                    op_type: "If".to_string(),
                    input: vec!["cond".to_string()],
                    output: vec!["if_out".to_string()],
                    attribute: vec![
                        AttributeProto {
                            name: "then_branch".to_string(),
                            r#type: AttributeType::Graph as i32,
                            g: Some(then_branch),
                            ..Default::default()
                        },
                        AttributeProto {
                            name: "else_branch".to_string(),
                            r#type: AttributeType::Graph as i32,
                            g: Some(else_branch),
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
            ],
            output: vec![ValueInfoProto { name: "if_out".to_string(), ..Default::default() }],
            ..Default::default()
        };

        let mut values: HashMap<String, Value> = HashMap::new();
        values.insert("raw".to_string(), Value::new(&[1f32, 2., 3.], &Device::Cpu)?);
        values.insert("cond".to_string(), Value::new(&[1u8], &Device::Cpu)?);
        let outputs = simple_eval_(&graph, &mut values)?;

        assert_eq!(outputs["if_out"].to_vec1::<f32>()?, vec![1., 2., 3.]);
        assert!(
            !values.contains_key("x"),
            "\"x\" should have been evicted once both branches' over-counted \
             captures were released"
        );
        Ok(())
    }

    #[test]
    fn if_branch_releases_capture_from_a_nested_if() -> Result<()> {
        // Regression test: count_nested_subgraph_captures recurses into an
        // "If" nested inside another "If"'s branch, so a name captured only
        // by the innermost branch still gets counted at the outermost
        // scope. The release side must recurse the same way — releasing
        // only the taken branch's *own* (non-nested) captured_names misses
        // "x" entirely here, since "x" is never referenced directly inside
        // the outer branch, only inside the inner branch nested within it.
        let inner_branch = GraphProto {
            node: vec![NodeProto {
                op_type: "Identity".to_string(),
                input: vec!["x".to_string()],
                output: vec!["inner_branch_out".to_string()],
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "inner_branch_out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let outer_branch = GraphProto {
            node: vec![NodeProto {
                op_type: "If".to_string(),
                input: vec!["inner_cond".to_string()],
                output: vec!["outer_branch_out".to_string()],
                attribute: vec![AttributeProto {
                    name: "then_branch".to_string(),
                    r#type: AttributeType::Graph as i32,
                    g: Some(inner_branch),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "outer_branch_out".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };

        let graph = GraphProto {
            input: vec![
                ValueInfoProto { name: "raw".to_string(), ..Default::default() },
                ValueInfoProto { name: "cond".to_string(), ..Default::default() },
                ValueInfoProto { name: "inner_cond".to_string(), ..Default::default() },
            ],
            node: vec![
                NodeProto {
                    op_type: "Identity".to_string(),
                    input: vec!["raw".to_string()],
                    output: vec!["x".to_string()],
                    ..Default::default()
                },
                // x's only *top-level* consumer.
                NodeProto {
                    op_type: "Identity".to_string(),
                    input: vec!["x".to_string()],
                    output: vec!["discard".to_string()],
                    ..Default::default()
                },
                NodeProto {
                    op_type: "If".to_string(),
                    input: vec!["cond".to_string()],
                    output: vec!["if_out".to_string()],
                    attribute: vec![AttributeProto {
                        name: "then_branch".to_string(),
                        r#type: AttributeType::Graph as i32,
                        g: Some(outer_branch),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            output: vec![ValueInfoProto { name: "if_out".to_string(), ..Default::default() }],
            ..Default::default()
        };

        let mut values: HashMap<String, Value> = HashMap::new();
        values.insert("raw".to_string(), Value::new(&[1f32, 2., 3.], &Device::Cpu)?);
        values.insert("cond".to_string(), Value::new(&[1u8], &Device::Cpu)?);
        values.insert("inner_cond".to_string(), Value::new(&[1u8], &Device::Cpu)?);
        let outputs = simple_eval_(&graph, &mut values)?;

        assert_eq!(outputs["if_out"].to_vec1::<f32>()?, vec![1., 2., 3.]);
        assert!(
            !values.contains_key("x"),
            "\"x\" should have been evicted once the nested If's capture was released"
        );
        Ok(())
    }
}
