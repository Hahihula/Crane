// Hyper-connection mixer kernels (Qwen4-Exp) for the Intel SYCL backend,
// driven by `ops/hyper_connection.rs`. Built into `libcrane_gdn_sycl.so` by
// `crane-core/build.rs` (icpx, `--features sycl` only).
//
// The residual is `groups` parallel streams of `group` values per row
// (`[rows, groups * group]`). Each entry point replaces a chain of small
// candle ops that decode runs 97 times per token; all accumulate in f32.
#include <sycl/sycl.hpp>

#include <cstdio>

namespace {

enum { OUT_F32 = 0, OUT_F16 = 1 };

inline float sigmoidf(float x) { return 1.f / (1.f + sycl::exp(-x)); }

// out = x / rms(x over each group) * alpha[g], one work-group per (row, group).
template <typename T>
void norm_launch(sycl::queue &q, const T *x, const T *alpha, T *out, int rows, int groups,
                 int group, float eps) {
  constexpr int WG = 256;
  q.parallel_for(sycl::nd_range<1>(size_t(rows) * groups * WG, WG), [=](sycl::nd_item<1> it) {
    const size_t rg = it.get_group_linear_id();
    const int g = int(rg % size_t(groups));
    const T *xr = x + rg * group;
    float ss = 0.f;
    for (int j = int(it.get_local_linear_id()); j < group; j += WG) {
      const float v = static_cast<float>(xr[j]);
      ss += v * v;
    }
    ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
    const float inv = sycl::rsqrt(ss / float(group) + eps);
    const T *a = alpha + size_t(g) * group;
    T *o = out + rg * group;
    for (int j = int(it.get_local_linear_id()); j < group; j += WG) {
      o[j] = static_cast<T>(static_cast<float>(xr[j]) * inv * static_cast<float>(a[j]));
    }
  });
}

// out[r, j] = silu(low[r, j] * scale).
template <typename T>
void low_launch(sycl::queue &q, const T *low, T *out, size_t n, float scale) {
  q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
    const float v = static_cast<float>(low[i]) * scale;
    out[i] = static_cast<T>(v * sigmoidf(v));
  });
}

// out[r, j] = mean over g of sigmoid(gate[r, g, j]) * normed[r, g, j].
template <typename T>
void mix_launch(sycl::queue &q, const T *gate, const T *normed, T *out, int rows, int groups,
                int group) {
  q.parallel_for(sycl::range<2>(size_t(rows), size_t(group)), [=](sycl::id<2> idx) {
    const size_t r = idx[0];
    const size_t j = idx[1];
    const size_t base = r * size_t(groups) * group + j;
    float acc = 0.f;
    for (int g = 0; g < groups; ++g) {
      const size_t k = base + size_t(g) * group;
      acc += sigmoidf(static_cast<float>(gate[k])) * static_cast<float>(normed[k]);
    }
    out[r * group + j] = static_cast<T>(acc / float(groups));
  });
}

// out[r, g, j] = streams[r, g, j] + block[r, j] * 2 * sigmoid(logits[r, g] * scale).
template <typename T>
void combine_launch(sycl::queue &q, const T *streams, const T *block, const T *logits, T *out,
                    int rows, int groups, int group, float scale) {
  q.parallel_for(sycl::range<3>(size_t(rows), size_t(groups), size_t(group)),
                 [=](sycl::id<3> idx) {
                   const size_t r = idx[0], g = idx[1], j = idx[2];
                   const float w =
                       2.f * sigmoidf(static_cast<float>(logits[r * groups + g]) * scale);
                   const size_t k = (r * groups + g) * group + j;
                   out[k] = static_cast<T>(static_cast<float>(streams[k]) +
                                           static_cast<float>(block[r * group + j]) * w);
                 });
}

template <typename F> int run(int dtype, F &&f) {
  try {
    if (dtype == OUT_F32) {
      f(float{});
    } else if (dtype == OUT_F16) {
      f(sycl::half{});
    } else {
      return 2;
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[crane sycl] %s: %s\n", __func__, e.what());
    return 1;
  } catch (...) {
    return 1;
  }
}

} // namespace

extern "C" int crane_hc_norm_sycl(void *queue, int dtype, const void *x, const void *alpha,
                                  void *out, int rows, int groups, int group, float eps) {
  auto &q = *static_cast<sycl::queue *>(queue);
  return run(dtype, [&](auto t) {
    using T = decltype(t);
    norm_launch<T>(q, static_cast<const T *>(x), static_cast<const T *>(alpha),
                   static_cast<T *>(out), rows, groups, group, eps);
  });
}

extern "C" int crane_hc_low_sycl(void *queue, int dtype, const void *low, void *out, size_t n,
                                 float scale) {
  auto &q = *static_cast<sycl::queue *>(queue);
  return run(dtype, [&](auto t) {
    using T = decltype(t);
    low_launch<T>(q, static_cast<const T *>(low), static_cast<T *>(out), n, scale);
  });
}

extern "C" int crane_hc_mix_sycl(void *queue, int dtype, const void *gate, const void *normed,
                                 void *out, int rows, int groups, int group) {
  auto &q = *static_cast<sycl::queue *>(queue);
  return run(dtype, [&](auto t) {
    using T = decltype(t);
    mix_launch<T>(q, static_cast<const T *>(gate), static_cast<const T *>(normed),
                  static_cast<T *>(out), rows, groups, group);
  });
}

extern "C" int crane_hc_combine_sycl(void *queue, int dtype, const void *streams,
                                     const void *block, const void *logits, void *out, int rows,
                                     int groups, int group, float scale) {
  auto &q = *static_cast<sycl::queue *>(queue);
  return run(dtype, [&](auto t) {
    using T = decltype(t);
    combine_launch<T>(q, static_cast<const T *>(streams), static_cast<const T *>(block),
                      static_cast<const T *>(logits), static_cast<T *>(out), rows, groups, group,
                      scale);
  });
}
