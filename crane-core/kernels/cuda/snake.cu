/**
 * Fused Snake periodic activation kernel.
 *
 * Targets: sm_80+ (Ampere & newer, bf16 support)
 *
 * snake(x, alpha) = x + sin(alpha * x)^2 / alpha
 *
 * Caller pre-broadcasts x/alpha to matching shapes and passes contiguous
 * buffers, so this is a flat one-thread-per-element kernel over n elements.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void snake_f32(
    const float * __restrict__ x,
    const float * __restrict__ alpha,
    float       * __restrict__ dst,
    const uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xv = x[idx];
        float av = alpha[idx];
        float s = sinf(av * xv);
        dst[idx] = xv + (s * s) / av;
    }
}

extern "C" __global__ void snake_f16(
    const __half * __restrict__ x,
    const __half * __restrict__ alpha,
    __half       * __restrict__ dst,
    const uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xv = __half2float(x[idx]);
        float av = __half2float(alpha[idx]);
        float s = sinf(av * xv);
        dst[idx] = __float2half(xv + (s * s) / av);
    }
}

extern "C" __global__ void snake_bf16(
    const __nv_bfloat16 * __restrict__ x,
    const __nv_bfloat16 * __restrict__ alpha,
    __nv_bfloat16       * __restrict__ dst,
    const uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xv = __bfloat162float(x[idx]);
        float av = __bfloat162float(alpha[idx]);
        float s = sinf(av * xv);
        dst[idx] = __float2bfloat16(xv + (s * s) / av);
    }
}
