/**
 * Fused SwiGLU activation kernel: silu(gate) * up.
 *
 * Targets: sm_80+ (Ampere & newer, bf16 support), and AMD GPUs via HIP — the
 * ROCm launcher (`ops/rocm.rs`'s `launch_binary_elementwise`) hands this same
 * source to hipcc at runtime.
 *
 * Caller makes gate/up contiguous and matching-shape before dispatch, so
 * this is a flat one-thread-per-element kernel over n elements.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

extern "C" __global__ void swiglu_f32(
    const float * __restrict__ gate,
    const float * __restrict__ up,
    float       * __restrict__ dst,
    const uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = fast_silu(gate[idx]) * up[idx];
    }
}

extern "C" __global__ void swiglu_f16(
    const __half * __restrict__ gate,
    const __half * __restrict__ up,
    __half       * __restrict__ dst,
    const uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        dst[idx] = __float2half(fast_silu(g) * u);
    }
}

extern "C" __global__ void swiglu_bf16(
    const __nv_bfloat16 * __restrict__ gate,
    const __nv_bfloat16 * __restrict__ up,
    __nv_bfloat16       * __restrict__ dst,
    const uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = __bfloat162float(gate[idx]);
        float u = __bfloat162float(up[idx]);
        dst[idx] = __float2bfloat16(fast_silu(g) * u);
    }
}
