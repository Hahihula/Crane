// Hyper-connection mixer kernels (Qwen4-Exp) for the Metal backend, driven by
// `ops/hyper_connection.rs` — the counterpart of
// `kernels/sycl/hyper_connection.cpp`. Compiled at runtime through
// `ops/metal_util.rs`.
//
// The residual is `groups` parallel streams of `group` values per row
// (`[rows, groups * group]`). Each kernel replaces a chain of small candle
// ops that decode runs 97 times per token; all accumulate in f32. F32, F16
// and (Metal 3.0+) BF16 tensors are supported.

#include <metal_stdlib>
using namespace metal;

inline float sigmoidf(float x) { return 1.f / (1.f + exp(-x)); }

// Mirrors `HcParams` in `ops/hyper_connection.rs`.
struct HcParams {
    uint rows;
    uint groups;
    uint group;
    uint n;       // element count, for the element-wise kernels
    float eps;    // norm
    float scale;  // low / combine
};

constant uint NORM_TG = 256;

// out = x / rms(x over each group) * alpha[g]; one threadgroup of NORM_TG
// threads per (row, group).
template <typename T>
kernel void hc_norm(
    device const T *x [[buffer(0)]],
    device const T *alpha [[buffer(1)]],
    device T *out [[buffer(2)]],
    constant HcParams &p [[buffer(3)]],
    uint rg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[NORM_TG / 32];
    const uint g = rg % p.groups;
    device const T *xr = x + size_t(rg) * p.group;
    float ss = 0.f;
    for (uint j = tid; j < p.group; j += NORM_TG) {
        const float v = float(xr[j]);
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0) {
        partial[sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = 0.f;
    for (uint i = 0; i < NORM_TG / 32; ++i) {
        ss += partial[i];
    }
    const float inv = rsqrt(ss / float(p.group) + p.eps);
    device const T *a = alpha + size_t(g) * p.group;
    device T *o = out + size_t(rg) * p.group;
    for (uint j = tid; j < p.group; j += NORM_TG) {
        o[j] = T(float(xr[j]) * inv * float(a[j]));
    }
}

// out[i] = silu(low[i] * scale).
template <typename T>
kernel void hc_low(
    device const T *low [[buffer(0)]],
    device T *out [[buffer(1)]],
    constant HcParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    const float v = float(low[i]) * p.scale;
    out[i] = T(v * sigmoidf(v));
}

// out[r, j] = mean over g of sigmoid(gate[r, g, j]) * normed[r, g, j].
// Grid (group, rows).
template <typename T>
kernel void hc_mix(
    device const T *gate [[buffer(0)]],
    device const T *normed [[buffer(1)]],
    device T *out [[buffer(2)]],
    constant HcParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x;
    const uint r = gid.y;
    if (j >= p.group || r >= p.rows) {
        return;
    }
    const size_t base = size_t(r) * p.groups * p.group + j;
    float acc = 0.f;
    for (uint g = 0; g < p.groups; ++g) {
        const size_t k = base + size_t(g) * p.group;
        acc += sigmoidf(float(gate[k])) * float(normed[k]);
    }
    out[size_t(r) * p.group + j] = T(acc / float(p.groups));
}

// out[r, g, j] = streams[r, g, j] + block[r, j] * 2 * sigmoid(logits[r, g] * scale).
// Grid (group, groups, rows).
template <typename T>
kernel void hc_combine(
    device const T *streams [[buffer(0)]],
    device const T *block [[buffer(1)]],
    device const T *logits [[buffer(2)]],
    device T *out [[buffer(3)]],
    constant HcParams &p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
    const uint j = gid.x;
    const uint g = gid.y;
    const uint r = gid.z;
    if (j >= p.group || g >= p.groups || r >= p.rows) {
        return;
    }
    const float w = 2.f * sigmoidf(float(logits[size_t(r) * p.groups + g]) * p.scale);
    const size_t k = (size_t(r) * p.groups + g) * p.group + j;
    out[k] = T(float(streams[k]) + float(block[size_t(r) * p.group + j]) * w);
}

#define HC_KERNELS(T, TNAME)                                                                  \
    template [[host_name("hc_norm_" TNAME)]] kernel void hc_norm<T>(                          \
        device const T *, device const T *, device T *, constant HcParams &, uint, uint, uint, \
        uint);                                                                                \
    template [[host_name("hc_low_" TNAME)]] kernel void hc_low<T>(                            \
        device const T *, device T *, constant HcParams &, uint);                             \
    template [[host_name("hc_mix_" TNAME)]] kernel void hc_mix<T>(                            \
        device const T *, device const T *, device T *, constant HcParams &, uint2);          \
    template [[host_name("hc_combine_" TNAME)]] kernel void hc_combine<T>(                    \
        device const T *, device const T *, device const T *, device T *, constant HcParams &, \
        uint3);

HC_KERNELS(float, "f32")
HC_KERNELS(half, "f16")
#if defined(__HAVE_BFLOAT__)
HC_KERNELS(bfloat, "bf16")
#endif
