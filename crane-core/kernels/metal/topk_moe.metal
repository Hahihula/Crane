// Fused top-K `MoE` routing for the Metal backend, driven by
// `ops/fused_ops/topk_moe.rs` — the counterpart of `crane_topk_moe_sycl`
// (`kernels/sycl/fused_ops.cpp`) and `kernels/cuda/topk_moe.cu`.
//
// Softmax over a token's expert logits, `top_k` rounds of SIMD-group argmax
// (ties to the smaller expert id), optional renormalization. One 32-lane
// SIMD-group per token; up to 32 * 16 = 512 experts (`MAX_FUSED_EXPERTS` in
// `topk_moe.rs`). `logits` [n_tokens, n_experts] F32 in, `out_ids`
// [n_tokens, top_k] U32 and `out_weights` [n_tokens, top_k] F32 out.

#include <metal_stdlib>
using namespace metal;

constant int SGW = 32;
constant int MAX_EPT = 16;

// Mirrors `TopkParams` in `topk_moe.rs`.
struct TopkParams {
    int n_tokens;
    int n_experts;
    int top_k;
    int norm_topk;
};

kernel void topk_moe_f32(
    device const float *logits [[buffer(0)]],
    device uint *out_ids [[buffer(1)]],
    device float *out_weights [[buffer(2)]],
    constant TopkParams &p [[buffer(3)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane_u [[thread_index_in_simdgroup]]) {
    if (int(token) >= p.n_tokens) {
        return;
    }
    const int lane = int(lane_u);
    const int n_experts = p.n_experts;
    device const float *row = logits + size_t(token) * n_experts;
    // Every loop over `wt` runs the full MAX_EPT with a guard, so it unrolls
    // and `wt` stays in registers (a dynamic index would spill it to slow
    // thread memory, which dominates the single-SIMD-group decode case).
    float wt[MAX_EPT];
    float mx = -INFINITY;
    for (int i = 0; i < MAX_EPT; ++i) {
        const int e = lane + i * SGW;
        wt[i] = e < n_experts ? row[e] : -INFINITY;
        mx = fmax(mx, wt[i]);
    }
    mx = simd_max(mx);
    float sum = 0.f;
    for (int i = 0; i < MAX_EPT; ++i) {
        const int e = lane + i * SGW;
        wt[i] = e < n_experts ? exp(wt[i] - mx) : 0.f;
        sum += wt[i];
    }
    sum = simd_sum(sum);
    const float inv_sum = 1.f / sum;
    for (int i = 0; i < MAX_EPT; ++i) {
        const int e = lane + i * SGW;
        wt[i] *= inv_sum;
        // NaN never wins a comparison, so it would be re-selected forever;
        // padding lanes must never be picked.
        if (isnan(wt[i]) || e >= n_experts) {
            wt[i] = -INFINITY;
        }
    }
    device uint *ids = out_ids + size_t(token) * p.top_k;
    device float *ws = out_weights + size_t(token) * p.top_k;
    float picked = 0.f;
    for (int k = 0; k < p.top_k; ++k) {
        float best = -INFINITY;
        int best_e = n_experts;
        for (int i = 0; i < MAX_EPT; ++i) {
            const int e = lane + i * SGW;
            if (wt[i] > best || (wt[i] == best && e < best_e)) {
                best = wt[i];
                best_e = e;
            }
        }
        for (ushort mask = SGW / 2; mask > 0; mask /= 2) {
            const float ov = simd_shuffle_xor(best, mask);
            const int oe = simd_shuffle_xor(best_e, mask);
            if (ov > best || (ov == best && oe < best_e)) {
                best = ov;
                best_e = oe;
            }
        }
        if (lane == 0) {
            ids[k] = uint(best_e);
            ws[k] = best;
        }
        picked += best;
        for (int i = 0; i < MAX_EPT; ++i) {
            if (lane + i * SGW == best_e) {
                wt[i] = -INFINITY;
            }
        }
    }
    if (p.norm_topk && lane == 0) {
        const float inv = 1.f / (picked > 0.f ? picked : 1.f);
        for (int k = 0; k < p.top_k; ++k) {
            ws[k] *= inv;
        }
    }
}
