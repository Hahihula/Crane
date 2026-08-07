/**
 * Fused Atan2 kernel.
 *
 * Targets: sm_80+ (Ampere & newer, bf16 support)
 *
 * atan2(y, x): element-wise two-argument arctangent.
 *
 * Caller pre-broadcasts y/x to matching shapes and passes contiguous
 * buffers, so this is a flat one-thread-per-element kernel over n elements.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void atan2_f32(
    const float * __restrict__ y,
    const float * __restrict__ x,
    float       * __restrict__ dst,
    const uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = atan2f(y[idx], x[idx]);
    }
}

extern "C" __global__ void atan2_f16(
    const __half * __restrict__ y,
    const __half * __restrict__ x,
    __half       * __restrict__ dst,
    const uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float yv = __half2float(y[idx]);
        float xv = __half2float(x[idx]);
        dst[idx] = __float2half(atan2f(yv, xv));
    }
}

extern "C" __global__ void atan2_bf16(
    const __nv_bfloat16 * __restrict__ y,
    const __nv_bfloat16 * __restrict__ x,
    __nv_bfloat16       * __restrict__ dst,
    const uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float yv = __bfloat162float(y[idx]);
        float xv = __bfloat162float(x[idx]);
        dst[idx] = __float2bfloat16(atan2f(yv, xv));
    }
}
