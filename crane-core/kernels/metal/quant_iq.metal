// llama.cpp i-quant weight kernels for the Metal backend: IQ4_NL, IQ4_XS,
// IQ2_S, IQ3_XXS, IQ3_S and Q2_0, plus the k-quants Q4_K, Q5_K and Q6_K for
// packed MoE experts — the counterpart of
// `kernels/sycl/quant_iq.cpp` (see `crane-core/src/quantized/iquant.rs` for
// the block layouts). Compiled to an `MTLLibrary` at runtime by
// `ops/quant_iq/metal.rs` via `Device::new_library_with_source` (no
// `.metallib` to ship); the launcher splices the codebooks of
// `kernels/sycl/iq_grids.h` in place of the `#include` below, so both
// backends read one copy of the tables.
//
// Every format splits a row into 32-value chunks that decode on their own (an
// IQ4_NL block, a Q2_0 half-block, or one 32-value group of a 256-value
// super-block), so both entry points are written against one per-type
// `decode32`:
//
// - `matvec`: one 32-lane SIMD-group per output row; lanes stride over the
//   row's chunks, decode, dot against the f32 activations, and `simd_sum`.
//   With an expert-id buffer it is the `MoE` "matmul by id": input row `p`
//   uses expert `ids[p]`'s slice of a packed `[experts, rows, cols]` tensor
//   and activation row `p / x_div`, so routing never leaves the device.
// - `dequant`: one thread per chunk, optionally over a list of experts,
//   feeding prefill-sized matmuls.
//
// Activations are not quantized to int8; the dot product is plain f32.
// Decoders follow the CPU reference's float operation order, so `dequant`
// is bit-identical to it. F32, F16 and (when `__HAVE_BFLOAT__` is defined,
// i.e. Metal 3.0+) BF16 outputs are supported.

#include <metal_stdlib>
using namespace metal;

#include "iq_grids.h"

using namespace crane_iq;

constant int QK_K = 256;
constant int ROWS_PER_TG = 4;  // output rows (SIMD-groups) per threadgroup

// Type tags; the kernel names use `type_tag` in `ops/quant_iq/metal.rs`.
enum { IQ4_NL = 0, IQ4_XS = 1, IQ2_S = 2, IQ3_XXS = 3, IQ3_S = 4, Q2_0 = 5, Q4_K = 6, Q5_K = 7, Q6_K = 8 };

template <int TY> inline int block_values() {
    return TY == IQ4_NL ? 32 : TY == Q2_0 ? 64 : QK_K;
}
template <int TY> inline int block_bytes() {
    switch (TY) {
    case IQ4_NL: return 2 + 16;
    case IQ4_XS: return 2 + 2 + QK_K / 64 + QK_K / 2;
    case IQ2_S: return 2 + QK_K / 4 + QK_K / 16;
    case IQ3_XXS: return 2 + 3 * QK_K / 8;
    case IQ3_S: return 2 + 13 * QK_K / 32 + QK_K / 64;
    case Q4_K: return 2 + 2 + 12 + QK_K / 2;
    case Q5_K: return 2 + 2 + 12 + QK_K / 8 + QK_K / 2;
    case Q6_K: return QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
    default: return 2 + 64 / 4; // Q2_0
    }
}

// IQ4_NL / IQ4_XS codebook (mirrors `KVALUES_IQ4NL` in iquant.rs).
constant float kvalues_iq4nl[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
       1.f,   13.f,  25.f,  38.f,  53.f,  69.f,  89.f, 113.f,
};

// Decode a little-endian f16 (2 bytes, no alignment assumed).
inline float load_half(const device uchar *p) {
    return float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1]) << 8))));
}

inline float grid_byte(ulong entry, int j) { return float((entry >> (8 * j)) & 0xff); }
inline float signed_val(float v, uchar signs, int j) { return (signs >> j) & 1 ? -v : v; }
// ggml `ksigns_iq2xs`: seven sign bits plus one keeping the negatives even.
inline uchar ksigns(uint bits7) {
    const uint b = bits7 & 127;
    return uchar(b | ((popcount(b) & 1) << 7));
}

// Decodes chunk `c` (values 32c..32c+31) of `row` into `w`.
template <int TY> inline void decode32(const device uchar *row, int c, thread float *w) {
    if (TY == IQ4_NL) {
        const device uchar *blk = row + size_t(c) * block_bytes<TY>();
        const float d = load_half(blk);
        for (int j = 0; j < 16; ++j) {
            w[j] = d * kvalues_iq4nl[blk[2 + j] & 0xf];
            w[j + 16] = d * kvalues_iq4nl[blk[2 + j] >> 4];
        }
    } else if (TY == Q2_0) {
        const device uchar *blk = row + size_t(c / 2) * block_bytes<TY>();
        const float d = load_half(blk);
        const device uchar *qs = blk + 2 + 8 * (c % 2);
        for (int j = 0; j < 32; ++j) {
            w[j] = d * (float((qs[j / 4] >> (2 * (j % 4))) & 3) - 1.f);
        }
    } else {
        const device uchar *blk = row + size_t(c / 8) * block_bytes<TY>();
        const int ib = c % 8;
        // Q6_K keeps its scale after the values.
        const float d = load_half(blk + (TY == Q6_K ? block_bytes<TY>() - 2 : 0));
        if (TY == IQ4_XS) {
            const int scales_h = int(blk[2]) | (int(blk[3]) << 8);
            const int lo = (blk[4 + ib / 2] >> (4 * (ib % 2))) & 0xf;
            const int hi = (scales_h >> (2 * ib)) & 3;
            const float dl = d * float((lo | (hi << 4)) - 32);
            const device uchar *q = blk + 4 + QK_K / 64 + 16 * ib;
            for (int j = 0; j < 16; ++j) {
                w[j] = dl * kvalues_iq4nl[q[j] & 0xf];
                w[j + 16] = dl * kvalues_iq4nl[q[j] >> 4];
            }
        } else if (TY == IQ2_S) {
            const device uchar *qs = blk + 2;
            const device uchar *signs = qs + QK_K / 8;
            const device uchar *qh = blk + 2 + QK_K / 4;
            const uchar sc = blk[2 + QK_K / 4 + QK_K / 32 + ib];
            const float db0 = d * (0.5f + float(sc & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + float(sc >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const int idx = qs[4 * ib + l] | ((int(qh[ib]) << (8 - 2 * l)) & 0x300);
                const ulong g = iq2s_grid[idx];
                const uchar s = signs[4 * ib + l];
                const float db = l < 2 ? db0 : db1;
                for (int j = 0; j < 8; ++j) {
                    w[8 * l + j] = signed_val(db * grid_byte(g, j), s, j);
                }
            }
        } else if (TY == IQ3_XXS) {
            const device uchar *qs = blk + 2 + 8 * ib;
            const device uchar *sas = blk + 2 + QK_K / 4 + 4 * ib;
            const uint aux = uint(sas[0]) | (uint(sas[1]) << 8) | (uint(sas[2]) << 16) |
                             (uint(sas[3]) << 24);
            const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uchar s = ksigns(aux >> (7 * l));
                const uint g1 = iq3xxs_grid[qs[2 * l]];
                const uint g2 = iq3xxs_grid[qs[2 * l + 1]];
                for (int j = 0; j < 4; ++j) {
                    w[8 * l + j] = signed_val(db * grid_byte(g1, j), s, j);
                    w[8 * l + 4 + j] = signed_val(db * grid_byte(g2, j), s, j + 4);
                }
            }
        } else if (TY == Q4_K || TY == Q5_K) {
            // ggml `get_scale_min_k4`: eight 6-bit scale/min pairs in 12 bytes.
            const device uchar *sc = blk + 4;
            int s, m;
            if (ib < 4) {
                s = sc[ib] & 63;
                m = sc[ib + 4] & 63;
            } else {
                s = (sc[ib + 4] & 0xf) | ((sc[ib - 4] >> 6) << 4);
                m = (sc[ib + 4] >> 4) | ((sc[ib] >> 6) << 4);
            }
            const float dl = d * float(s);
            const float ml = load_half(blk + 2) * float(m);
            // Sub-blocks 2k and 2k+1 share 32 bytes: low nibbles, then high.
            const device uchar *q = blk + (TY == Q5_K ? 16 + QK_K / 8 : 16) + 32 * (ib / 2);
            const int shift = 4 * (ib % 2);
            for (int j = 0; j < 32; ++j) {
                int v = (q[j] >> shift) & 0xf;
                if (TY == Q5_K) {
                    v |= ((blk[16 + j] >> ib) & 1) << 4;
                }
                w[j] = dl * float(v) - ml;
            }
        } else if (TY == Q6_K) {
            // Each half of the block covers 128 values in four groups of 32:
            // group k takes the low (k < 2) or high nibbles of 32 `ql` bytes
            // and bits 2k..2k+1 of 32 `qh` bytes, one i8 scale per 16 values.
            const int half_idx = ib / 4;
            const int k = ib % 4;
            const device uchar *ql = blk + 64 * half_idx + 32 * (k & 1);
            const device uchar *qh = blk + QK_K / 2 + 32 * half_idx;
            const device char *sc =
                reinterpret_cast<const device char *>(blk + QK_K / 2 + QK_K / 4) + 8 * half_idx + 2 * k;
            for (int j = 0; j < 32; ++j) {
                const int v = ((ql[j] >> (4 * (k >> 1))) & 0xf) | (((qh[j] >> (2 * k)) & 3) << 4);
                w[j] = d * float(sc[j / 16]) * float(v - 32);
            }
        } else { // IQ3_S
            const device uchar *qs = blk + 2 + 8 * ib;
            const int qh = blk[2 + QK_K / 4 + ib];
            const device uchar *signs = blk + 2 + QK_K / 4 + QK_K / 32 + 4 * ib;
            const int sc =
                (blk[2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + ib / 2] >> (4 * (ib % 2))) & 0xf;
            const float db = d * float(1 + 2 * sc);
            for (int l = 0; l < 4; ++l) {
                const uint g1 = iq3s_grid[qs[2 * l] | ((qh << (8 - 2 * l)) & 256)];
                const uint g2 = iq3s_grid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
                for (int j = 0; j < 4; ++j) {
                    w[8 * l + j] = signed_val(db * grid_byte(g1, j), signs[l], j);
                    w[8 * l + 4 + j] = signed_val(db * grid_byte(g2, j), signs[l], j + 4);
                }
            }
        }
    }
}

template <int TY> inline size_t row_bytes(int cols) {
    return size_t(cols / block_values<TY>()) * block_bytes<TY>();
}

// Mirrors `MatvecParams` in `ops/quant_iq/metal.rs`.
struct MatvecParams {
    ulong expert_stride; // bytes between experts of `packed`
    int x_div;           // activation row of pair `p` is `p / x_div`
    int pairs;
    int out_rows;
    int cols;
    int has_ids;         // 0: `ids` is unbound, every pair uses `packed` itself
};

// `output[p, r] = dot(W_e[r], input[p / x_div])` for `p < pairs`, where
// `W_e` is expert `e = ids[p]` of `packed`, or the only matrix without ids.
// Threadgroups: (ceil(out_rows / ROWS_PER_TG), pairs), ROWS_PER_TG * 32 threads.
template <int TY, typename T>
kernel void iq_matvec(
    const device uchar *packed [[buffer(0)]],
    const device uint *ids [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device T *output [[buffer(3)]],
    constant MatvecParams &p [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int pair = int(tg.y);
    const int row = int(tg.x) * ROWS_PER_TG + int(sg);
    // Uniform across the SIMD-group, so the reduction below stays legal.
    if (row >= p.out_rows) {
        return;
    }
    const size_t expert = p.has_ids ? size_t(ids[pair]) : 0;
    const device uchar *r = packed + expert * p.expert_stride + size_t(row) * row_bytes<TY>(p.cols);
    const device float *x = input + size_t(pair / p.x_div) * p.cols;
    const int chunks = p.cols / 32;
    float acc = 0.f;
    float w[32];
    for (int c = int(lane); c < chunks; c += 32) {
        decode32<TY>(r, c, w);
        const device float *xc = x + 32 * c;
        for (int j = 0; j < 32; ++j) {
            acc += w[j] * xc[j];
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        output[size_t(pair) * p.out_rows + row] = T(acc);
    }
}

// Mirrors `DequantParams` in `ops/quant_iq/metal.rs`.
struct DequantParams {
    ulong expert_stride;
    int n_mats;
    int n_rows;
    int cols;
    int has_ids;
};

// Decodes `n_mats` matrices of `n_rows` x `cols` into `[n_mats, n_rows,
// cols]`: matrix `m` is expert `ids[m]` of `packed`, or the only one without
// ids. One thread per 32-value chunk; grid (cols / 32, n_mats * n_rows).
template <int TY, typename T>
kernel void iq_dequant(
    const device uchar *packed [[buffer(0)]],
    const device uint *ids [[buffer(1)]],
    device T *output [[buffer(2)]],
    constant DequantParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    const int c = int(gid.x);
    const size_t mat_row = gid.y;
    if (c >= p.cols / 32 || mat_row >= size_t(p.n_mats) * size_t(p.n_rows)) {
        return;
    }
    const size_t m = mat_row / size_t(p.n_rows);
    const size_t row = mat_row % size_t(p.n_rows);
    const size_t expert = p.has_ids ? size_t(ids[m]) : 0;
    float w[32];
    decode32<TY>(packed + expert * p.expert_stride + row * row_bytes<TY>(p.cols), c, w);
    device T *out = output + mat_row * size_t(p.cols) + 32 * size_t(c);
    for (int j = 0; j < 32; ++j) {
        out[j] = T(w[j]);
    }
}

#define IQ_KERNELS(TY, NAME, T, TNAME)                                                     \
    template [[host_name("matvec_" NAME "_" TNAME)]] kernel void iq_matvec<TY, T>(          \
        const device uchar *, const device uint *, const device float *, device T *,       \
        constant MatvecParams &, uint2, uint, uint);                                       \
    template [[host_name("dequant_" NAME "_" TNAME)]] kernel void iq_dequant<TY, T>(        \
        const device uchar *, const device uint *, device T *, constant DequantParams &,   \
        uint2);

#define IQ_ALL_TYPES(T, TNAME)                 \
    IQ_KERNELS(IQ4_NL, "iq4_nl", T, TNAME)     \
    IQ_KERNELS(IQ4_XS, "iq4_xs", T, TNAME)     \
    IQ_KERNELS(IQ2_S, "iq2_s", T, TNAME)       \
    IQ_KERNELS(IQ3_XXS, "iq3_xxs", T, TNAME)   \
    IQ_KERNELS(IQ3_S, "iq3_s", T, TNAME)       \
    IQ_KERNELS(Q2_0, "q2_0", T, TNAME)         \
    IQ_KERNELS(Q4_K, "q4_k", T, TNAME)         \
    IQ_KERNELS(Q5_K, "q5_k", T, TNAME)         \
    IQ_KERNELS(Q6_K, "q6_k", T, TNAME)

IQ_ALL_TYPES(float, "f32")
IQ_ALL_TYPES(half, "f16")
#if defined(__HAVE_BFLOAT__)
IQ_ALL_TYPES(bfloat, "bf16")
#endif
