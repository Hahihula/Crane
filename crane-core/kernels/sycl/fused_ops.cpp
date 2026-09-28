// Fused elementwise kernels for the Intel SYCL backend — the counterpart of
// `kernels/cuda/fused_ops.cu`. Built into `libcrane_gdn_sycl.so` by
// `crane-core/build.rs` (icpx, `--features sycl` only) and driven by
// `ops/fused_ops/sycl_impl.rs`.
//
// `fused_silu_mul` collapses the `narrow(gate) + narrow(up) + silu + mul` op
// chain (candle: 2 real kernel launches after narrow's free views — SiLU then
// multiply) into one submission, mirroring the CUDA/ROCm `fused_silu_mul_*`
// kernels.
//
// `swiglu`/`snake` are the SYCL counterparts of `kernels/cuda/swiglu.cu` /
// `kernels/cuda/snake.cu` — flat one-work-item-per-element kernels, driven by
// `ops/fused_ops/swiglu.rs` / `ops/fused_ops/snake.rs` directly (not through
// `sycl_impl.rs`, since those ops aren't part of the shared fused-op registry
// `sycl_impl.rs` wraps).
#include <sycl/sycl.hpp>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

// dtype tags match `ops/fused_ops/sycl_impl.rs`, `swiglu.rs`, `snake.rs`.
enum { CRANE_FSM_F32 = 0, CRANE_FSM_F16 = 1, CRANE_FSM_BF16 = 2 };

template <typename T>
void fused_silu_mul_launch(sycl::queue &q, const T *gu, T *out,
                           long long n_rows, int isz) {
  const long long total = n_rows * static_cast<long long>(isz);
  q.parallel_for(sycl::range<1>(static_cast<size_t>(total)),
                 [=](sycl::id<1> idx) {
                   long long i = idx[0];
                   long long row = i / isz;
                   long long col = i % isz;
                   const T *base = gu + row * (2LL * isz);
                   float g = static_cast<float>(base[col]);
                   float u = static_cast<float>(base[isz + col]);
                   float silu = g / (1.0f + sycl::exp(-g));
                   out[i] = static_cast<T>(silu * u);
                 });
}

// swiglu(gate, up) = silu(gate) * up, flat over n elements.
template <typename T>
void swiglu_launch(sycl::queue &q, const T *gate, const T *up, T *out,
                   long long n) {
  q.parallel_for(sycl::range<1>(static_cast<size_t>(n)),
                 [=](sycl::id<1> idx) {
                   long long i = idx[0];
                   float g = static_cast<float>(gate[i]);
                   float u = static_cast<float>(up[i]);
                   float silu = g / (1.0f + sycl::exp(-g));
                   out[i] = static_cast<T>(silu * u);
                 });
}

// snake(x, alpha) = x + sin(alpha * x)^2 / alpha, flat over n elements.
template <typename T>
void snake_launch(sycl::queue &q, const T *x, const T *alpha, T *out,
                  long long n) {
  q.parallel_for(sycl::range<1>(static_cast<size_t>(n)),
                 [=](sycl::id<1> idx) {
                   long long i = idx[0];
                   float xv = static_cast<float>(x[i]);
                   float av = static_cast<float>(alpha[i]);
                   float s = sycl::sin(av * xv);
                   out[i] = static_cast<T>(xv + (s * s) / av);
                 });
}

} // namespace

extern "C" int crane_swiglu_sycl(void *queue, int dtype, const void *gate,
                                const void *up, void *out, long long n) {
  try {
    auto &sq = *static_cast<sycl::queue *>(queue);
    switch (dtype) {
    case CRANE_FSM_F32:
      swiglu_launch<float>(sq, static_cast<const float *>(gate),
                           static_cast<const float *>(up),
                           static_cast<float *>(out), n);
      return 0;
    case CRANE_FSM_F16:
      swiglu_launch<sycl::half>(sq, static_cast<const sycl::half *>(gate),
                                static_cast<const sycl::half *>(up),
                                static_cast<sycl::half *>(out), n);
      return 0;
    case CRANE_FSM_BF16:
      swiglu_launch<bf16>(sq, static_cast<const bf16 *>(gate),
                          static_cast<const bf16 *>(up),
                          static_cast<bf16 *>(out), n);
      return 0;
    default:
      return 2; // unsupported dtype
    }
  } catch (const sycl::exception &) {
    return 1;
  } catch (...) {
    return 1;
  }
}

extern "C" int crane_snake_sycl(void *queue, int dtype, const void *x,
                               const void *alpha, void *out, long long n) {
  try {
    auto &sq = *static_cast<sycl::queue *>(queue);
    switch (dtype) {
    case CRANE_FSM_F32:
      snake_launch<float>(sq, static_cast<const float *>(x),
                          static_cast<const float *>(alpha),
                          static_cast<float *>(out), n);
      return 0;
    case CRANE_FSM_F16:
      snake_launch<sycl::half>(sq, static_cast<const sycl::half *>(x),
                               static_cast<const sycl::half *>(alpha),
                               static_cast<sycl::half *>(out), n);
      return 0;
    case CRANE_FSM_BF16:
      snake_launch<bf16>(sq, static_cast<const bf16 *>(x),
                         static_cast<const bf16 *>(alpha),
                         static_cast<bf16 *>(out), n);
      return 0;
    default:
      return 2; // unsupported dtype
    }
  } catch (const sycl::exception &) {
    return 1;
  } catch (...) {
    return 1;
  }
}

extern "C" int crane_fused_silu_mul_sycl(void *queue, int dtype,
                                        const void *gate_up, void *out,
                                        long long n_rows, int intermediate_size) {
  try {
    auto &sq = *static_cast<sycl::queue *>(queue);
    switch (dtype) {
    case CRANE_FSM_F32:
      fused_silu_mul_launch<float>(sq, static_cast<const float *>(gate_up),
                                   static_cast<float *>(out), n_rows,
                                   intermediate_size);
      return 0;
    case CRANE_FSM_F16:
      fused_silu_mul_launch<sycl::half>(
          sq, static_cast<const sycl::half *>(gate_up),
          static_cast<sycl::half *>(out), n_rows, intermediate_size);
      return 0;
    default:
      return 2; // unsupported dtype
    }
  } catch (const sycl::exception &) {
    return 1;
  } catch (...) {
    return 1;
  }
}
