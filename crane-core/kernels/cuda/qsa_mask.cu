// SPDX-License-Identifier: MIT
//
// QSA indexer block selection (Qwen4-Exp, `models/qwen4_exp/indexer.rs`): from
// per-block scores `[seq, blocks]`, build the additive attention mask
// `[seq, cells]` -- 0 on the `keep` best complete blocks each query can see
// plus its incomplete tail, -inf elsewhere. Replaces a host round-trip of the
// scores and a per-row sort on every decode step.
//
// One thread block per query row. The `keep`-th largest score is found by a
// bitwise search over order-preserving integer keys (32 passes, each counting
// keys >= candidate across the row); blocks above it are all kept and ties at
// it are kept earliest-block-first, the total order the host path uses (score
// descending, then block ascending).

#include <stdint.h>

#define QSA_THREADS 256
#define QSA_WARPS (QSA_THREADS / 32)

// Order-preserving float -> uint (larger float, larger key).
__device__ __forceinline__ uint32_t qsa_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

extern "C" __global__ void qsa_topk_mask_f32(
    const float * scores, float * mask, int blocks, int cells, int start, int ratio, int keep) {
    __shared__ int count;
    __shared__ int warp_eq[QSA_WARPS];
    __shared__ int taken;

    const int row = blockIdx.x;
    const int pos = start + row;
    const int visible = (pos + 1) / ratio;  // complete blocks this query sees
    const float * s = scores + size_t(row) * blocks;
    float * out = mask + size_t(row) * cells;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    for (int c = tid; c < cells; c += QSA_THREADS) {
        out[c] = -INFINITY;
    }
    // Other threads write the tail (and kept blocks) over cells filled above.
    __syncthreads();
    // The incomplete tail block, up to and including the query itself.
    for (int c = visible * ratio + tid; c <= pos; c += QSA_THREADS) {
        out[c] = 0.f;
    }

    uint32_t threshold = 0;  // key of the keep-th largest score
    const bool select_all = visible <= keep;
    if (!select_all) {
        uint32_t prefix = 0;
        for (int bit = 31; bit >= 0; --bit) {
            const uint32_t candidate = prefix | (1u << bit);
            if (tid == 0) {
                count = 0;
            }
            __syncthreads();
            int local = 0;
            for (int b = tid; b < visible; b += QSA_THREADS) {
                local += qsa_key(s[b]) >= candidate;
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                local += __shfl_down_sync(0xffffffffu, local, offset);
            }
            if (lane == 0) {
                atomicAdd(&count, local);
            }
            __syncthreads();
            if (count >= keep) {
                prefix = candidate;
            }
            __syncthreads();
        }
        threshold = prefix;

        // Blocks strictly above the threshold.
        if (tid == 0) {
            count = 0;
        }
        __syncthreads();
        int local = 0;
        for (int b = tid; b < visible; b += QSA_THREADS) {
            local += qsa_key(s[b]) > threshold;
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local += __shfl_down_sync(0xffffffffu, local, offset);
        }
        if (lane == 0) {
            atomicAdd(&count, local);
        }
        __syncthreads();
    }
    __syncthreads();
    const int need_eq = select_all ? 0 : keep - count;
    if (tid == 0) {
        taken = 0;
    }
    __syncthreads();

    // Walk the row in chunks of QSA_THREADS blocks so ties at the threshold
    // are taken in block order.
    for (int base = 0; base < visible; base += QSA_THREADS) {
        const int b = base + tid;
        bool selected = false;
        bool eq = false;
        if (b < visible) {
            if (select_all) {
                selected = true;
            } else {
                const uint32_t k = qsa_key(s[b]);
                selected = k > threshold;
                eq = k == threshold;
            }
        }
        const uint32_t ballot = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) {
            warp_eq[warp] = __popc(ballot);
        }
        __syncthreads();
        int before = taken;
        for (int w = 0; w < warp; ++w) {
            before += warp_eq[w];
        }
        before += __popc(ballot & ((1u << lane) - 1));
        if (eq && before < need_eq) {
            selected = true;
        }
        if (selected) {
            for (int c = b * ratio; c < (b + 1) * ratio; ++c) {
                out[c] = 0.f;
            }
        }
        __syncthreads();
        if (tid == 0) {
            int total = 0;
            for (int w = 0; w < QSA_WARPS; ++w) {
                total += warp_eq[w];
            }
            taken += total;
        }
        __syncthreads();
    }
}
