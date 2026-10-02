// Fused Atan2 for the Metal backend — the counterpart of
// `kernels/cuda/atan2.cu`, driven by `ops/fused_ops/atan2.rs`.
//
// atan2(y, x), element-wise. The caller pre-broadcasts y/x to matching
// shapes and passes contiguous buffers, so this is a flat
// one-thread-per-element kernel over n elements. F16/BF16 are widened to f32.
// `precise::atan2` because the library is compiled with fast math, whose
// `atan2` is not IEEE-accurate (signed zeros, the axes).

#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void atan2_kernel(
    device const T *y [[buffer(0)]],
    device const T *x [[buffer(1)]],
    device T *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= n) {
        return;
    }
    out[i] = T(precise::atan2(float(y[i]), float(x[i])));
}

#define ATAN2_KERNEL(T, TNAME)                                                      \
    template [[host_name("atan2_" TNAME)]] kernel void atan2_kernel<T>(              \
        device const T *, device const T *, device T *, constant uint &, uint);

ATAN2_KERNEL(float, "f32")
ATAN2_KERNEL(half, "f16")
#if defined(__HAVE_BFLOAT__)
ATAN2_KERNEL(bfloat, "bf16")
#endif
