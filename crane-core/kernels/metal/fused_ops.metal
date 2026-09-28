// SPDX-License-Identifier: MIT
// Fused `SwiGLU` and `Snake` activation kernels for Metal.
//
// Complements the CUDA (`kernels/cuda/swiglu.cu` / `snake.cu`, PTX via
// `bindgen_cuda`), ROCm (same `.cu` source through `hipcc`) and SYCL
// (`kernels/sycl/fused_ops.cpp`, compiled to `libcrane_gdn_sycl.so` by
// `build.rs`) implementations. Compiled JIT by
// `crane-core/src/ops/fused_ops/{swiglu,snake}.rs`'s `metal_kernel` module
// via `MetalDevice::device().new_library_with_source(...)` on first use,
// and cached in a per-`(device, kernel)` `OnceLock` keyed on
// `device.registry_id()`. The kernel arguments are scalar dimension + two
// raw `device` pointers (input A, input B, output), so each dispatch is
// cheap to encode and the source is exactly one `[[kernel]]` function per
// (op, dtype).
//
// Each kernel is a flat one-thread-per-element pass. The caller
// (Rust-side) makes `gate`/`up` (or `x`/`alpha`) contiguous and
// matching-shape before dispatch — same contract as the CUDA/ROCm kernels.
// F16/BF16 inputs are widened to F32 for the inner computation, then
// narrowed back; this matches the CUDA/ROCm kernels and is what makes the
// activation spec-correct (e.g. `swiglu` in F16 with the same `g`/`u`
// values produces the same output as the F32 pipeline). NaN/Inf propagate
// the same way.

#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------------
// SwiGLU: out[i] = (g[i] / (1 + exp(-g[i]))) * u[i]
// ----------------------------------------------------------------------------

kernel void crane_swiglu_f32(
    constant uint &n,
    device const float *gate,
    device const float *up,
    device float *out,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float g = gate[tid];
    float u = up[tid];
    out[tid] = (g / (1.0f + metal::exp(-g))) * u;
}

kernel void crane_swiglu_f16(
    constant uint &n,
    device const half *gate,
    device const half *up,
    device half *out,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float g = static_cast<float>(gate[tid]);
    float u = static_cast<float>(up[tid]);
    out[tid] = static_cast<half>((g / (1.0f + metal::exp(-g))) * u);
}

kernel void crane_swiglu_bf16(
    constant uint &n,
    device const bfloat *gate,
    device const bfloat *up,
    device bfloat *out,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float g = static_cast<float>(gate[tid]);
    float u = static_cast<float>(up[tid]);
    out[tid] = static_cast<bfloat>((g / (1.0f + metal::exp(-g))) * u);
}

// ----------------------------------------------------------------------------
// Snake (periodic): out[i] = x[i] + sin(alpha[i] * x[i])^2 / alpha[i]
// ----------------------------------------------------------------------------

kernel void crane_snake_f32(
    constant uint &n,
    device const float *x,
    device const float *alpha,
    device float *out,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float xv = x[tid];
    float av = alpha[tid];
    float s = metal::precise::sin(av * xv);
    out[tid] = xv + (s * s) / av;
}

kernel void crane_snake_f16(
    constant uint &n,
    device const half *x,
    device const half *alpha,
    device half *out,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float xv = static_cast<float>(x[tid]);
    float av = static_cast<float>(alpha[tid]);
    float s = metal::precise::sin(av * xv);
    out[tid] = static_cast<half>(xv + (s * s) / av);
}

kernel void crane_snake_bf16(
    constant uint &n,
    device const bfloat *x,
    device const bfloat *alpha,
    device bfloat *out,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float xv = static_cast<float>(x[tid]);
    float av = static_cast<float>(alpha[tid]);
    float s = metal::precise::sin(av * xv);
    out[tid] = static_cast<bfloat>(xv + (s * s) / av);
}
