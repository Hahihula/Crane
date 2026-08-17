// SPDX-License-Identifier: MIT
//! Crane Added 20260731: ONNX `NonZero` as a native eval op.

use candle_core::{DType, Result, Tensor};

use crate::onnx::proto::NodeProto;

/// ONNX `NonZero`: returns the multi-indices of `input`'s nonzero elements.
///
/// Per the ONNX spec, the output is an `int64` tensor of shape
/// `[input_rank, num_nonzero]`, where column `i` holds the multi-index (in
/// row-major order) of the `i`-th nonzero element. `input` is flattened and
/// cast to `F32` so that any upstream dtype's "nonzero" is checked uniformly
/// as `!= 0.0` -- this also covers the common case of a comparison op
/// (`Greater`/`Less`/etc.) feeding `NonZero` with a 0/1-valued `U8` tensor.
pub(crate) fn nonzero(_node: &NodeProto, input: &Tensor) -> Result<Tensor> {
    let dims = input.dims().to_vec();
    let rank = dims.len();
    let flat: Vec<f32> = input
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;

    // Standard unravel-index: for a fixed `flat_idx`, walking `d` from the
    // last dimension backward and repeatedly taking `% dims[d]` then
    // dividing recovers each dimension's index directly (each `indices[d]`
    // is written by-index, not appended in traversal order, so visiting
    // dimensions in this order doesn't affect the final per-dimension
    // ordering across nonzero elements -- only the outer ascending
    // `flat_idx` loop does, matching the ONNX spec's row-major output
    // order).
    let mut indices: Vec<Vec<i64>> = vec![Vec::new(); rank];
    for (flat_idx, &v) in flat.iter().enumerate() {
        if v == 0.0 {
            continue;
        }
        let mut remaining = flat_idx;
        for d in (0..rank).rev() {
            // `dims[d]` is a real tensor dimension, always far below
            // i64::MAX, so this cast never wraps.
            #[allow(clippy::cast_possible_wrap)]
            let dim_idx = (remaining % dims[d]) as i64;
            indices[d].push(dim_idx);
            remaining /= dims[d];
        }
    }

    let num_nonzero = indices.first().map_or(0, Vec::len);
    let mut flat_out = Vec::with_capacity(rank * num_nonzero);
    for row in indices {
        flat_out.extend(row);
    }
    Tensor::from_vec(flat_out, (rank, num_nonzero), input.device())
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use crate::onnx::proto::NodeProto;

    use super::nonzero;

    fn node() -> NodeProto {
        NodeProto {
            name: "NonZero.0".to_string(),
            ..Default::default()
        }
    }

    // The motivating shape: a 2D mask produces row-major multi-indices.
    #[test]
    fn nonzero_2d_multi_indices() -> Result<()> {
        // [[0, 1, 0], [1, 0, 1]] -> nonzero at (0,1), (1,0), (1,2)
        let x = Tensor::new(&[[0f32, 1., 0.], [1., 0., 1.]], &Device::Cpu)?;

        let y = nonzero(&node(), &x)?;

        assert_eq!(y.dims(), &[2, 3]);
        let rows = y.to_vec2::<i64>()?;
        assert_eq!(rows[0], vec![0, 1, 1]); // row indices
        assert_eq!(rows[1], vec![1, 0, 2]); // col indices
        Ok(())
    }

    #[test]
    fn nonzero_all_zero_input_returns_empty() -> Result<()> {
        let x = Tensor::zeros((2, 2), candle_core::DType::F32, &Device::Cpu)?;

        let y = nonzero(&node(), &x)?;

        assert_eq!(y.dims(), &[2, 0]);
        Ok(())
    }

    #[test]
    fn nonzero_all_nonzero_input_returns_every_index() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?;

        let y = nonzero(&node(), &x)?;

        assert_eq!(y.dims(), &[2, 4]);
        let rows = y.to_vec2::<i64>()?;
        assert_eq!(rows[0], vec![0, 0, 1, 1]);
        assert_eq!(rows[1], vec![0, 1, 0, 1]);
        Ok(())
    }

    #[test]
    fn nonzero_1d_input() -> Result<()> {
        let x = Tensor::new(&[0f32, 5., 0., 7.], &Device::Cpu)?;

        let y = nonzero(&node(), &x)?;

        assert_eq!(y.dims(), &[1, 2]);
        assert_eq!(y.to_vec2::<i64>()?[0], vec![1, 3]);
        Ok(())
    }

    // NonZero commonly consumes the U8 0/1 output of a comparison op
    // (Greater/Less/etc.); the F32 cast must still treat any nonzero U8
    // value as "nonzero".
    #[test]
    fn nonzero_u8_comparison_output() -> Result<()> {
        let x = Tensor::new(&[[0u8, 1], [1, 0]], &Device::Cpu)?;

        let y = nonzero(&node(), &x)?;

        assert_eq!(y.dims(), &[2, 2]);
        let rows = y.to_vec2::<i64>()?;
        assert_eq!(rows[0], vec![0, 1]);
        assert_eq!(rows[1], vec![1, 0]);
        Ok(())
    }
}
