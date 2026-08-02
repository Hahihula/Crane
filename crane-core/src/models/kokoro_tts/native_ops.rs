// SPDX-License-Identifier: MIT

//! Native-Rust replacements for ONNX ops `crate::onnx::simple_eval` can't
//! run at all:
//!
//! - `Resize` with `mode="linear"` — `candle-onnx`'s `"Resize"` handler
//!   only implements `mode="nearest"` (a *different*, rank-4-only gap in
//!   that same path is fixed by `crate::onnx::optimizer`'s `compat`
//!   submodule instead, since it's a rewritable shape problem, not a
//!   missing computation). See [`NativeLinearResize`].
//!
//! [`extract_segments`] splits Kokoro's ONNX graph into segments at each
//! such node, decoding its static parameters (weights, window
//! coefficients, scale factors, ...) once at load time into a
//! [`SpecialNode`]. `Model::generate_speech` runs `crate::onnx::simple_eval`
//! per segment, computing each `SpecialNode` natively in between and
//! feeding its result back in as a plain tensor for the next segment.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use crate::onnx::proto::{GraphProto, NodeProto};

/// Looks up `name` in `graph.initializer` and decodes it into a `Tensor`
/// via `crate::onnx`'s own tensor decoder.
fn decode_initializer(graph: &GraphProto, name: Option<&str>, what: &str) -> Result<Tensor> {
    let name = name.filter(|s| !s.is_empty()).with_context(|| format!("missing {what} input name"))?;
    let proto = graph
        .initializer
        .iter()
        .find(|i| i.name == name)
        .with_context(|| format!("{what} {name:?} not found in graph initializers"))?;
    Ok(crate::onnx::eval::get_tensor(proto, name)?)
}

/// A `mode="linear"` `Resize` node's static parameters, extracted once at
/// load time. The evaluator's `"Resize"` handler only implements
/// `mode="nearest"` (see `crate::onnx::optimizer`'s `compat` submodule doc
/// for the separate rank-4-only gap in that path) — `"linear"` bails
/// unconditionally, so this computes ONNX's half-pixel linear resize
/// directly.
///
/// Scoped to what Kokoro's graph actually needs: exactly one axis has a
/// non-`1.0` `scales` entry (every other axis is left unchanged), and
/// `coordinate_transformation_mode="half_pixel"` (the ONNX default for
/// `linear` mode). Both observed nodes fit this shape — one downsamples,
/// one upsamples, by the same factor, in a harmonic source-signal
/// generator — but this isn't a general N-linear implementation.
pub struct NativeLinearResize {
    data_input: String,
    output: String,
    /// The single axis with a non-`1.0` scale.
    axis: usize,
    /// That axis's scale factor (may be `> 1` to upsample or `< 1` to
    /// downsample); output length is `floor(input_length * scale)`, per
    /// the ONNX `Resize` spec.
    scale: f32,
    /// Length of the decoded `scales` array, checked against the data
    /// input's actual rank in [`Self::compute`]. `axis` is found by
    /// scanning `scales` alone, with nothing to check it was actually
    /// indexing the right tensor — a `scales` array shorter than `data`'s
    /// rank (e.g. one meant to pair with an `axes` input, which this
    /// implementation doesn't support) would otherwise have `axis`
    /// silently resolve against the wrong dimension.
    scales_len: usize,
}

impl NativeLinearResize {
    fn build(node: &NodeProto, graph: &GraphProto) -> Result<Self> {
        let data_input = node.input.first().cloned().context("Resize node missing data input")?;
        let output = node.output.first().cloned().context("Resize node missing output")?;

        let ctm = node.attribute.iter().find(|a| a.name == "coordinate_transformation_mode");
        if ctm.is_some_and(|a| a.s != b"half_pixel") {
            bail!(
                "Resize node {:?}: only coordinate_transformation_mode=\"half_pixel\" is \
                 supported for linear-mode resize",
                node.name
            );
        }

        let scales_name = node.input.get(2).map(String::as_str).filter(|s| !s.is_empty());
        let Some(scales_name) = scales_name else {
            bail!("Resize node {:?}: only a 'scales' input is supported (not 'sizes')", node.name);
        };
        let scales = decode_initializer(graph, Some(scales_name), "Resize scales")?.to_vec1::<f32>()?;

        let mut axis = None;
        for (i, &s) in scales.iter().enumerate() {
            if (s - 1.0).abs() > 1e-6 {
                if axis.is_some() {
                    bail!(
                        "Resize node {:?}: more than one axis has a non-1.0 scale {scales:?}; \
                         only single-axis linear resize is supported",
                        node.name
                    );
                }
                axis = Some(i);
            }
        }
        let axis =
            axis.with_context(|| format!("Resize node {:?}: no axis has a non-1.0 scale", node.name))?;

        Ok(Self { data_input, output, axis, scale: scales[axis], scales_len: scales.len() })
    }

    /// Computes ONNX's half-pixel linear resize along [`Self::axis`] using
    /// `index_select` gathers and broadcast arithmetic, keeping the data in
    /// tensor storage throughout (no host-side `Vec` round-trip). For output
    /// index `i`, the corresponding input coordinate is
    /// `(i + 0.5) / scale - 0.5` (boundary-clamped), matching
    /// `coordinate_transformation_mode="half_pixel"`.
    ///
    /// # Errors
    ///
    /// Returns an error if `values` is missing this node's data input, or
    /// if the decoded `scales` array's length doesn't match the data
    /// input's actual rank.
    fn compute(&self, values: &HashMap<String, Tensor>) -> Result<Tensor> {
        let x = values
            .get(&self.data_input)
            .with_context(|| format!("missing Resize input {:?}", self.data_input))?;
        if self.scales_len != x.rank() {
            bail!(
                "Resize node: scales has {} value(s) but input {:?} has rank {}",
                self.scales_len,
                self.data_input,
                x.rank()
            );
        }
        let in_len = x.dim(self.axis)?;
        // `scale` and `in_len` are both bounded, ordinary model dimensions
        // (at most a few hundred thousand samples), so this product stays
        // far inside f32's exact-integer range and the truncating cast to
        // usize matches ONNX Resize's `floor(input_dim * scale)` spec.
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let out_len = (in_len as f32 * self.scale) as usize;

        // `in_len` is a real tensor dimension (well under i64::MAX in
        // practice), so this cast never wraps.
        #[allow(clippy::cast_possible_wrap)]
        let max_idx = in_len as i64 - 1;

        let mut floor_idx = Vec::with_capacity(out_len);
        let mut ceil_idx = Vec::with_capacity(out_len);
        let mut fracs = Vec::with_capacity(out_len);
        for i in 0..out_len {
            #[allow(clippy::cast_precision_loss)]
            let coord = (i as f32 + 0.5) / self.scale - 0.5;
            let f = coord.floor();
            #[allow(clippy::cast_possible_truncation)]
            let f_i64 = f as i64;
            floor_idx.push(f_i64.clamp(0, max_idx));
            ceil_idx.push((f_i64 + 1).clamp(0, max_idx));
            fracs.push(coord - f);
        }

        let dev = x.device();
        let floor_t = Tensor::new(floor_idx, dev)?;
        let ceil_t = Tensor::new(ceil_idx, dev)?;

        let left = x.index_select(&floor_t, self.axis)?;
        let right = x.index_select(&ceil_t, self.axis)?;

        let mut frac_shape = vec![1usize; x.rank()];
        frac_shape[self.axis] = out_len;
        let frac_t = Tensor::new(fracs, dev)?.reshape(frac_shape)?;

        Ok(left.broadcast_add(&right.broadcast_sub(&left)?.broadcast_mul(&frac_t)?)?)
    }
}

/// One graph-splitting boundary: either op type this module knows how to
/// compute natively. A single variant today (only `Resize` `mode="linear"`
/// still needs segmentation — see the module doc) rather than a bare
/// `NativeLinearResize` wrapper, to keep this shape ready for a future
/// native op without another restructuring.
pub enum SpecialNode {
    /// See [`NativeLinearResize`].
    LinearResize(NativeLinearResize),
}

impl SpecialNode {
    /// The node's output name — the accumulated value map is populated
    /// under this name after [`Self::compute`].
    #[must_use]
    pub fn output(&self) -> &str {
        match self {
            Self::LinearResize(r) => &r.output,
        }
    }

    /// Computes this node's result natively.
    ///
    /// # Errors
    ///
    /// See [`NativeLinearResize::compute`].
    pub fn compute(&self, values: &HashMap<String, Tensor>) -> Result<Tensor> {
        match self {
            Self::LinearResize(r) => r.compute(values),
        }
    }
}

/// `true` if `node` is a `Resize` node whose `mode` attribute is present
/// and not `"nearest"` — i.e. a mode `crate::onnx::simple_eval` can never
/// run, regardless of rank (unlike the rank-4-only gap `crate::onnx::optimizer`'s
/// `compat` submodule fixes for `mode="nearest"`).
fn is_unsupported_resize_mode(node: &NodeProto) -> bool {
    node.attribute.iter().find(|a| a.name == "mode").is_some_and(|a| a.s != b"nearest")
}

/// Splits `graph.node` into segments at each linear-`Resize` node (in
/// original order), returning the segments (one more than the number of
/// special nodes found) and each special node's extracted [`SpecialNode`].
///
/// # Errors
///
/// Returns an error if a linear-mode `Resize` node's inputs/parameters
/// can't be resolved — see [`NativeLinearResize::build`].
pub fn extract_segments(graph: &GraphProto) -> Result<(Vec<Vec<NodeProto>>, Vec<SpecialNode>)> {
    let mut segments = Vec::new();
    let mut special_nodes = Vec::new();
    let mut current = Vec::new();
    for node in &graph.node {
        match node.op_type.as_str() {
            "Resize" if is_unsupported_resize_mode(node) => {
                special_nodes.push(SpecialNode::LinearResize(NativeLinearResize::build(node, graph)?));
                segments.push(std::mem::take(&mut current));
            }
            _ => current.push(node.clone()),
        }
    }
    segments.push(current);
    Ok((segments, special_nodes))
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use crate::onnx::proto::AttributeProto;
    use crate::onnx::proto::attribute_proto::AttributeType;

    use super::*;

    fn values_with(name: &str, tensor: Tensor) -> HashMap<String, Tensor> {
        HashMap::from([(name.to_string(), tensor)])
    }

    #[test]
    fn linear_resize_upsample_doubles_length() {
        let node = NativeLinearResize {
            data_input: "x".to_string(),
            output: "y".to_string(),
            axis: 1,
            scale: 2.0,
            scales_len: 2,
        };
        // Shape [1, 4]: a simple ramp, easy to hand-check half-pixel linear
        // interpolation against.
        let x = Tensor::new(&[[0f32, 10., 20., 30.]], &Device::Cpu).unwrap();
        let values = values_with("x", x);

        let y = node.compute(&values).unwrap();
        assert_eq!(y.dims(), &[1, 8]);
        let got = y.to_vec2::<f32>().unwrap()[0].clone();
        // coord(i) = (i + 0.5) / 2 - 0.5; e.g. i=0 -> coord=-0.25 -> clamps
        // to index 0 -> value 0.0; i=1 -> coord=0.25 -> lerp(0,10,0.25)=2.5.
        let expected = [0.0f32, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 30.0];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-4, "got {got:?}, expected {expected:?}");
        }
    }

    #[test]
    fn linear_resize_downsample_halves_length() {
        let node = NativeLinearResize {
            data_input: "x".to_string(),
            output: "y".to_string(),
            axis: 1,
            scale: 0.5,
            scales_len: 2,
        };
        let x = Tensor::new(&[[0f32, 10., 20., 30.]], &Device::Cpu).unwrap();
        let values = values_with("x", x);

        let y = node.compute(&values).unwrap();
        assert_eq!(y.dims(), &[1, 2]);
    }

    // A `scales` array shorter than the input's actual rank must now error
    // instead of silently resolving `axis` against the wrong dimension.
    #[test]
    fn linear_resize_scales_length_mismatch_errors() {
        let node = NativeLinearResize {
            data_input: "x".to_string(),
            output: "y".to_string(),
            axis: 1,
            scale: 2.0,
            scales_len: 2,
        };
        // Rank 3, but scales_len claims rank 2.
        let x = Tensor::new(&[[[0f32, 10., 20., 30.]]], &Device::Cpu).unwrap();
        let values = values_with("x", x);

        let err = node.compute(&values).unwrap_err();
        assert!(err.to_string().contains("scales"));
    }

    #[test]
    fn is_unsupported_resize_mode_detects_linear_not_nearest_or_default() {
        let mode_attr = |mode: &[u8]| AttributeProto {
            name: "mode".to_string(),
            r#type: AttributeType::String as i32,
            s: mode.to_vec(),
            ..Default::default()
        };
        let linear = NodeProto { attribute: vec![mode_attr(b"linear")], ..Default::default() };
        let nearest = NodeProto { attribute: vec![mode_attr(b"nearest")], ..Default::default() };
        let no_mode = NodeProto::default();

        assert!(is_unsupported_resize_mode(&linear));
        assert!(!is_unsupported_resize_mode(&nearest));
        assert!(!is_unsupported_resize_mode(&no_mode));
    }

    fn passthrough_node(name: &str) -> NodeProto {
        NodeProto {
            op_type: "Identity".to_string(),
            name: name.to_string(),
            input: vec![format!("{name}_in")],
            output: vec![format!("{name}_out")],
            ..Default::default()
        }
    }

    #[test]
    fn extract_segments_with_no_special_nodes_returns_one_segment() {
        let graph = GraphProto { node: vec![passthrough_node("a"), passthrough_node("b")], ..Default::default() };
        let (segments, special_nodes) = extract_segments(&graph).unwrap();
        assert_eq!(segments.len(), 1);
        assert!(special_nodes.is_empty());
        assert_eq!(segments[0].len(), 2);
    }

}
