// SPDX-License-Identifier: MIT

//! Native i-quant inference kernels (see `crate::quantized::iquant`).

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(feature = "sycl")]
pub mod sycl;

/// Experts decoded per batched GEMM in [`indexed_via_gemm`]; bounds the
/// transient f16 weights (32 x 640 x 2560 x 2 B = 105 MB for
/// Qwen3.8-Flash-Next).
#[cfg(any(feature = "sycl", feature = "metal", feature = "cuda"))]
const GEMM_EXPERT_BATCH: usize = 32;

/// Zero-padded activation rows per batched GEMM in [`indexed_via_gemm`]:
/// a batch holds `experts x largest pair count` rows, so with skewed routing
/// (a few experts taking most tokens) an unbounded batch pads every expert to
/// the busiest one's count. 8192 rows of a 2560-wide projection are 42 MB of
/// f16 activations, and as much again for the output.
#[cfg(any(feature = "sycl", feature = "metal", feature = "cuda"))]
const GEMM_ROW_BUDGET: usize = 8192;

/// Prefill path of the backends' `matvec_indexed` (`MoE` matmul by expert
/// id, see [`sycl::matvec_indexed`](self::sycl) /
/// [`metal::matvec_indexed`](self::metal)). Pairs are grouped by expert;
/// experts are processed in batches sorted by pair count (so padding stays
/// small), each at most [`GEMM_EXPERT_BATCH`] experts and
/// [`GEMM_ROW_BUDGET`] padded rows: the batch's weights are decoded to f16
/// in one launch by `dequantize_experts(ids)` (`[batch, output_rows, cols]`),
/// each expert's activation rows are gathered into a zero-padded `[batch,
/// max_pairs, cols]` block, and one batched GEMM produces every pair's
/// output. Only the real (unpadded) rows are kept per batch, so what
/// accumulates is `pairs x output_rows`; a final gather puts them back in
/// pair order.
///
/// Costs one host sync for `ids` (`U32`, flattened). Every batch is planned
/// on the host from that one read and all index arrays go to the device in a
/// single upload before any batch is queued: an upload waits for the queue
/// to drain, so uploading per batch would stall the device once per batch.
#[cfg(any(feature = "sycl", feature = "metal", feature = "cuda"))]
pub(crate) fn indexed_via_gemm(
    input: &candle_core::Tensor,
    ids: &candle_core::Tensor,
    x_div: usize,
    output_rows: usize,
    cols: usize,
    out_dtype: candle_core::DType,
    dequantize_experts: impl Fn(&candle_core::Tensor) -> candle_core::Result<candle_core::Tensor>,
) -> candle_core::Result<candle_core::Tensor> {
    use candle_core::{DType, Tensor};

    let device = input.device();
    let host_ids = ids.to_vec1::<u32>()?;
    let pairs = host_ids.len();
    let x_div = x_div.max(1);
    let mut by_expert: std::collections::BTreeMap<u32, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (p, &e) in host_ids.iter().enumerate() {
        by_expert.entry(e).or_default().push(p);
    }
    let mut experts: Vec<(u32, Vec<usize>)> = by_expert.into_iter().collect();
    experts.sort_by_key(|(_, members)| std::cmp::Reverse(members.len()));

    let x_rows = input.dim(0)?;
    let to_u32 = |n: usize| u32::try_from(n).map_err(|e| candle_core::Error::Msg(e.to_string()));
    // Activation row that padding slots gather: an extra zero row.
    let zero_row = to_u32(x_rows)?;

    // Plan every batch, appending its index arrays to one host buffer:
    // expert ids, padded gather rows, and the real rows of its output.
    let mut index = Vec::with_capacity(experts.len() + 3 * pairs);
    // (expert ids, gather rows, real rows) offsets into `index`, plus the
    // batch size and its padded pair count.
    let mut batches = Vec::new();
    // Where each pair's result lands in the concatenated batch outputs.
    let mut position = vec![0u32; pairs];
    let mut offset = 0usize;
    let mut next = 0usize;
    while next < experts.len() {
        // Sorted by count, so the first expert of a batch sets its padding.
        let max_n = experts[next].1.len();
        let take = (GEMM_ROW_BUDGET / max_n)
            .clamp(1, GEMM_EXPERT_BATCH)
            .min(experts.len() - next);
        let batch = &experts[next..next + take];
        next += take;

        let ids_at = index.len();
        index.extend(batch.iter().map(|(e, _)| *e));
        let gather_at = index.len();
        index.resize(gather_at + take * max_n, zero_row);
        let real_at = index.len();
        for (b, (_, members)) in batch.iter().enumerate() {
            for (i, &p) in members.iter().enumerate() {
                index[gather_at + b * max_n + i] = to_u32(p / x_div)?;
                position[p] = to_u32(offset + index.len() - real_at)?;
                index.push(to_u32(b * max_n + i)?);
            }
        }
        offset += index.len() - real_at;
        batches.push((
            ids_at,
            gather_at,
            real_at,
            take,
            max_n,
            index.len() - real_at,
        ));
    }
    let position_at = index.len();
    index.extend(position);
    let index_len = index.len();
    let index = Tensor::from_vec(index, index_len, device)?;

    // Activations in f16 with the zero row appended.
    let x = Tensor::cat(
        &[
            &input.to_dtype(DType::F16)?,
            &Tensor::zeros((1, cols), DType::F16, device)?,
        ],
        0,
    )?;
    let mut outputs = Vec::with_capacity(batches.len());
    for (ids_at, gather_at, real_at, take, max_n, n_real) in batches {
        let weights = dequantize_experts(&index.narrow(0, ids_at, take)?)?;
        let xb = x
            .index_select(&index.narrow(0, gather_at, take * max_n)?, 0)?
            .reshape((take, max_n, cols))?;
        let y = xb
            .matmul(&weights.transpose(1, 2)?)? // [batch, max_n, rows]
            .reshape((take * max_n, output_rows))?;
        outputs.push(y.index_select(&index.narrow(0, real_at, n_real)?, 0)?);
    }
    let all = Tensor::cat(&outputs, 0)?;
    drop(outputs);
    let out = all
        .index_select(&index.narrow(0, position_at, pairs)?, 0)?
        .to_dtype(out_dtype)?;
    // Batches differ in size, so their freed temporaries would otherwise pile
    // up in the SYCL backend's caches (a no-op elsewhere).
    crate::device::release_cached_memory(device);
    Ok(out)
}
