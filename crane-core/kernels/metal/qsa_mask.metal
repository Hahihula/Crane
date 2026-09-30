// QSA indexer block selection (Qwen4-Exp, `models/qwen4_exp/indexer.rs`) for
// the Metal backend, driven by `ops/fused_ops/qsa_mask.rs` — the counterpart
// of `kernels/cuda/qsa_mask.cu`. From per-block scores `[seq, blocks]`, build
// the additive attention mask `[seq, cells]`: 0 on the `keep` best complete
// blocks each query can see plus its incomplete tail, -inf elsewhere.
//
// One threadgroup per query row. The `keep`-th largest score is found by a
// bitwise search over order-preserving integer keys (32 passes, each counting
// keys >= candidate across the row); blocks above it are all kept and ties at
// it are kept earliest-block-first, the total order the host path uses (score
// descending, then block ascending). Counts are reduced with `simd_sum` and a
// per-SIMD-group partial array rather than atomics, so the kernel is
// deterministic.

#include <metal_stdlib>
using namespace metal;

constant uint QSA_THREADS = 256;  // `THREADS` in qsa_mask.rs
constant uint QSA_SIMDS = QSA_THREADS / 32;

// Mirrors `QsaParams` in `qsa_mask.rs`.
struct QsaParams {
    int blocks;
    int cells;
    int start;
    int ratio;
    int keep;
};

// Order-preserving float -> uint (larger float, larger key).
inline uint qsa_key(float v) {
    const uint u = as_type<uint>(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Sum of `local` over the threadgroup; every thread gets the result.
inline int tg_sum(int local, threadgroup int *partial, uint sg, uint lane) {
    local = simd_sum(local);
    if (lane == 0) {
        partial[sg] = local;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int total = 0;
    for (uint w = 0; w < QSA_SIMDS; ++w) {
        total += partial[w];
    }
    // `partial` is reused by the next call.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

kernel void qsa_topk_mask_f32(
    device const float *scores [[buffer(0)]],
    device float *mask [[buffer(1)]],
    constant QsaParams &p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup int partial[QSA_SIMDS];

    const int pos = p.start + int(row);
    const int visible = (pos + 1) / p.ratio;  // complete blocks this query sees
    device const float *s = scores + size_t(row) * p.blocks;
    device float *out = mask + size_t(row) * p.cells;

    for (int c = int(tid); c < p.cells; c += QSA_THREADS) {
        out[c] = -INFINITY;
    }
    // Other threads write the tail (and kept blocks) over cells filled above.
    threadgroup_barrier(mem_flags::mem_device);
    // The incomplete tail block, up to and including the query itself.
    for (int c = visible * p.ratio + int(tid); c <= pos; c += QSA_THREADS) {
        out[c] = 0.f;
    }

    uint threshold = 0;  // key of the keep-th largest score
    int above = 0;       // blocks strictly above it
    const bool select_all = visible <= p.keep;
    if (!select_all) {
        uint prefix = 0;
        for (int bit = 31; bit >= 0; --bit) {
            const uint candidate = prefix | (1u << bit);
            int local = 0;
            for (int b = int(tid); b < visible; b += QSA_THREADS) {
                local += qsa_key(s[b]) >= candidate;
            }
            if (tg_sum(local, partial, sg, lane) >= p.keep) {
                prefix = candidate;
            }
        }
        threshold = prefix;
        int local = 0;
        for (int b = int(tid); b < visible; b += QSA_THREADS) {
            local += qsa_key(s[b]) > threshold;
        }
        above = tg_sum(local, partial, sg, lane);
    }
    const int need_eq = select_all ? 0 : p.keep - above;

    // Walk the row in chunks of QSA_THREADS blocks so ties at the threshold
    // are taken in block order.
    int taken = 0;
    for (int base = 0; base < visible; base += QSA_THREADS) {
        const int b = base + int(tid);
        bool selected = false;
        bool eq = false;
        if (b < visible) {
            if (select_all) {
                selected = true;
            } else {
                const uint k = qsa_key(s[b]);
                selected = k > threshold;
                eq = k == threshold;
            }
        }
        const int in_simd = simd_prefix_exclusive_sum(int(eq));
        const int simd_total = simd_sum(int(eq));
        if (lane == 0) {
            partial[sg] = simd_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int before = taken + in_simd;
        int chunk_total = 0;
        for (uint w = 0; w < QSA_SIMDS; ++w) {
            if (w < sg) {
                before += partial[w];
            }
            chunk_total += partial[w];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (eq && before < need_eq) {
            selected = true;
        }
        if (selected) {
            for (int c = b * p.ratio; c < (b + 1) * p.ratio; ++c) {
                out[c] = 0.f;
            }
        }
        taken += chunk_total;
    }
}
