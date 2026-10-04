// SPDX-License-Identifier: MIT

//! Native i-quant inference kernels (see `crate::quantized::iquant`).

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub mod rocm;
#[cfg(feature = "sycl")]
pub mod sycl;

/// Routed (token, expert) pairs from which the by-id `MoE` matmul takes the
/// batched-GEMM path ([`GemmPlan`]: one host sync for the ids, then each
/// routed expert is decoded once and multiplied by the backend's GEMM)
/// instead of the backends' by-id matvec (every pair decodes its expert on
/// its own, ids stay on the device).
///
/// The GEMM path costs roughly a fixed ~9 ms per projection once most
/// experts are routed (decoding them dominates) plus ~3 ms per 1000 tokens;
/// the matvec grows linearly from ~45 us per token. On an Arc Pro B70 with
/// Qwen3.8-Flash-Next shapes (256 experts, top-10) they cross at ~200-250
/// tokens, i.e. ~2k pairs. CUDA, Metal and `ROCm` use the same crossover;
/// not tuned there yet.
#[cfg(any(
    feature = "sycl",
    feature = "metal",
    feature = "cuda",
    feature = "rocm"
))]
pub(crate) const GEMM_MIN_PAIRS: usize = 2048;

/// Experts decoded per batched GEMM in [`GemmPlan::run`]; bounds the
/// transient f16 weights (32 x 640 x 2560 x 2 B = 105 MB for
/// Qwen3.8-Flash-Next).
#[cfg(any(
    feature = "sycl",
    feature = "metal",
    feature = "cuda",
    feature = "rocm"
))]
const GEMM_EXPERT_BATCH: usize = 32;

/// Zero-padded activation rows per batched GEMM in [`GemmPlan::run`]:
/// a batch holds `experts x largest pair count` rows, so with skewed routing
/// (a few experts taking most tokens) an unbounded batch pads every expert to
/// the busiest one's count. 8192 rows of a 2560-wide projection are 42 MB of
/// f16 activations, and as much again for the output.
#[cfg(any(
    feature = "sycl",
    feature = "metal",
    feature = "cuda",
    feature = "rocm"
))]
const GEMM_ROW_BUDGET: usize = 8192;

/// The prefill path of the by-id `MoE` matmul, planned once per routing.
///
/// Pairs are grouped by expert; experts are processed in batches sorted by
/// pair count (so padding stays small), each at most [`GEMM_EXPERT_BATCH`]
/// experts and [`GEMM_ROW_BUDGET`] padded rows: the batch's weights are
/// decoded to f16 in one launch (`[batch, output_rows, cols]`), each expert's
/// activation rows are gathered into a zero-padded `[batch, max_pairs, cols]`
/// block, and one batched GEMM produces every pair's output. Only the real
/// (unpadded) rows are kept per batch and concatenated: that is *plan order*,
/// which [`Self::to_pair_order`] (or a fused consumer reading
/// [`Self::position`]) maps back to pair order.
///
/// Building the plan costs the one host sync for the ids; every batch is
/// planned from that read and all index arrays go to the device in a single
/// upload (an upload waits for the queue to drain, so uploading per batch
/// would stall the device once per batch). A `MoE` layer's gate, up and down
/// projections share one routing, so they share one plan
/// (`PackedIQuantExperts`): gate and up run together on one gather of the
/// input, and down reads their (elementwise-combined) output in plan order,
/// so nothing is reordered until the final combine.
#[cfg(any(
    feature = "sycl",
    feature = "metal",
    feature = "cuda",
    feature = "rocm"
))]
pub(crate) struct GemmPlan {
    /// Every batch's index arrays (see [`PlanBatch`]) plus `position`.
    index: candle_core::Tensor,
    batches: Vec<PlanBatch>,
    /// Pair `p`'s row in plan order is `index[position_at + p]`.
    position_at: usize,
    pairs: usize,
    /// The input layouts planned for, with their row counts;
    /// [`PlanBatch::gather_at`] follows this order.
    layouts: Vec<(PlanLayout, usize)>,
}

/// Decodes the listed experts (`U32` ids) of one projection to f16
/// `[n, output_rows, cols]`, for [`GemmPlan::run`].
#[cfg(any(
    feature = "sycl",
    feature = "metal",
    feature = "cuda",
    feature = "rocm"
))]
pub(crate) type DequantizeExperts<'a> =
    &'a dyn Fn(&candle_core::Tensor) -> candle_core::Result<candle_core::Tensor>;

/// How a [`GemmPlan::run`] input's rows relate to the routed pairs.
#[cfg(any(
    feature = "sycl",
    feature = "metal",
    feature = "cuda",
    feature = "rocm"
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlanLayout {
    /// Pair `p` reads row `p / x_div`: `top_k` for one row per token, 1 for
    /// one row per pair (in pair order).
    Rows(usize),
    /// One row per pair in plan order: the output of an earlier
    /// [`GemmPlan::run`] on the same plan.
    PlanOrder,
}

/// One batch of [`GemmPlan`]: offsets into its `index` tensor.
#[cfg(any(
    feature = "sycl",
    feature = "metal",
    feature = "cuda",
    feature = "rocm"
))]
struct PlanBatch {
    /// `experts` expert ids.
    ids_at: usize,
    experts: usize,
    /// Padded pairs per expert.
    max_n: usize,
    /// Per planned layout, `experts * max_n` input rows to gather; the
    /// padding slots gather an appended zero row.
    gather_at: Vec<usize>,
    /// `n_real` rows of the padded output that hold real pairs.
    real_at: usize,
    n_real: usize,
}

#[cfg(any(
    feature = "sycl",
    feature = "metal",
    feature = "cuda",
    feature = "rocm"
))]
impl GemmPlan {
    /// Plan the routing `ids` (`U32`, flattened: pair `p` is expert `ids[p]`)
    /// for inputs laid out as each of `layouts`.
    ///
    /// # Errors
    ///
    /// Returns an error if `ids` cannot be read or an index exceeds `u32`.
    pub(crate) fn new(
        ids: &candle_core::Tensor,
        layouts: &[PlanLayout],
    ) -> candle_core::Result<Self> {
        let device = ids.device();
        let host_ids = ids.flatten_all()?.to_vec1::<u32>()?;
        let pairs = host_ids.len();
        let mut by_expert: std::collections::BTreeMap<u32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (p, &e) in host_ids.iter().enumerate() {
            by_expert.entry(e).or_default().push(p);
        }
        let mut experts: Vec<(u32, Vec<usize>)> = by_expert.into_iter().collect();
        experts.sort_by_key(|(_, members)| std::cmp::Reverse(members.len()));

        let to_u32 =
            |n: usize| u32::try_from(n).map_err(|e| candle_core::Error::Msg(e.to_string()));
        let layouts: Vec<(PlanLayout, usize)> = layouts
            .iter()
            .map(|&l| match l {
                PlanLayout::Rows(d) => (PlanLayout::Rows(d.max(1)), pairs.div_ceil(d.max(1))),
                PlanLayout::PlanOrder => (l, pairs),
            })
            .collect();

        let mut index = Vec::with_capacity(experts.len() + (layouts.len() + 2) * pairs);
        let mut batches = Vec::new();
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
            let mut gather_at = Vec::with_capacity(layouts.len());
            for &(layout, x_rows) in &layouts {
                let at = index.len();
                // Padding slots read the zero row appended after the input.
                index.resize(at + take * max_n, to_u32(x_rows)?);
                // Real rows are numbered in this same (expert, member) order.
                let mut real = offset;
                for (b, (_, members)) in batch.iter().enumerate() {
                    for (i, &p) in members.iter().enumerate() {
                        index[at + b * max_n + i] = to_u32(match layout {
                            PlanLayout::Rows(x_div) => p / x_div,
                            PlanLayout::PlanOrder => real,
                        })?;
                        real += 1;
                    }
                }
                gather_at.push(at);
            }
            let real_at = index.len();
            for (b, (_, members)) in batch.iter().enumerate() {
                for (i, &p) in members.iter().enumerate() {
                    position[p] = to_u32(offset + index.len() - real_at)?;
                    index.push(to_u32(b * max_n + i)?);
                }
            }
            let n_real = index.len() - real_at;
            offset += n_real;
            batches.push(PlanBatch {
                ids_at,
                experts: take,
                max_n,
                gather_at,
                real_at,
                n_real,
            });
        }
        let position_at = index.len();
        index.extend(position);
        let len = index.len();
        Ok(Self {
            index: candle_core::Tensor::from_vec(index, len, device)?,
            batches,
            position_at,
            pairs,
            layouts,
        })
    }

    /// Each pair's row in plan order (`U32`, `[pairs]`).
    ///
    /// # Errors
    ///
    /// Returns an error if the slice cannot be taken.
    pub(crate) fn position(&self) -> candle_core::Result<candle_core::Tensor> {
        self.index.narrow(0, self.position_at, self.pairs)
    }

    /// A plan-order `[pairs, _]` tensor back in pair order.
    ///
    /// # Errors
    ///
    /// Returns an error if the gather fails.
    pub(crate) fn to_pair_order(
        &self,
        t: &candle_core::Tensor,
    ) -> candle_core::Result<candle_core::Tensor> {
        t.index_select(&self.position()?, 0)
    }

    /// Projections over the plan, one per entry of `dequantize_experts`:
    /// each output is `[pairs, output_rows]` f16 in plan order, pair `p`
    /// being expert `ids[p]` applied to its row of `input` (`[_, cols]`, laid
    /// out as `layout`, at least the planned rows), and
    /// `dequantize_experts[i](ids)` decodes the listed experts of projection
    /// `i` to f16 `[n, output_rows, cols]`. Several projections of the same
    /// input (a `MoE` layer's gate and up) share the input conversion and
    /// each batch's activation gather.
    ///
    /// # Errors
    ///
    /// Returns an error if `layout` was not planned for, `input` is too
    /// short, or a kernel fails.
    pub(crate) fn run(
        &self,
        input: &candle_core::Tensor,
        layout: PlanLayout,
        output_rows: usize,
        cols: usize,
        dequantize_experts: &[DequantizeExperts<'_>],
    ) -> candle_core::Result<Vec<candle_core::Tensor>> {
        use candle_core::{DType, Tensor};

        let layout = match layout {
            PlanLayout::Rows(d) => PlanLayout::Rows(d.max(1)),
            PlanLayout::PlanOrder => PlanLayout::PlanOrder,
        };
        let Some(slot) = self.layouts.iter().position(|&(l, _)| l == layout) else {
            candle_core::bail!("GEMM plan has no {layout:?} input layout")
        };
        let x_rows = self.layouts[slot].1;
        if input.dim(0)? < x_rows {
            candle_core::bail!("GEMM plan needs {x_rows} input rows, got {}", input.dim(0)?)
        }
        let device = input.device();
        // Activations in f16 with the zero row appended.
        let x = Tensor::cat(
            &[
                &input.narrow(0, 0, x_rows)?.to_dtype(DType::F16)?,
                &Tensor::zeros((1, cols), DType::F16, device)?,
            ],
            0,
        )?;
        let index = &self.index;
        let mut outputs: Vec<Vec<Tensor>> = dequantize_experts
            .iter()
            .map(|_| Vec::with_capacity(self.batches.len()))
            .collect();
        for batch in &self.batches {
            let (take, max_n) = (batch.experts, batch.max_n);
            let ids = index.narrow(0, batch.ids_at, take)?;
            let xb = x
                .index_select(&index.narrow(0, batch.gather_at[slot], take * max_n)?, 0)?
                .reshape((take, max_n, cols))?;
            let real = index.narrow(0, batch.real_at, batch.n_real)?;
            for (dequantize, out) in dequantize_experts.iter().zip(&mut outputs) {
                let weights = dequantize(&ids)?;
                let y = xb
                    .matmul(&weights.transpose(1, 2)?)? // [batch, max_n, rows]
                    .reshape((take * max_n, output_rows))?;
                out.push(y.index_select(&real, 0)?);
            }
        }
        let outs = outputs
            .into_iter()
            .map(|parts| Tensor::cat(&parts, 0))
            .collect::<candle_core::Result<Vec<_>>>()?;
        // Batches differ in size, so their freed temporaries would otherwise
        // pile up in the SYCL backend's caches (a no-op elsewhere).
        crate::device::release_cached_memory(device);
        Ok(outs)
    }
}
