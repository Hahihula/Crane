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

#include <cstdio>

#include <cfloat>
#include <cmath>
#include <cstdint>

#include <cfloat>
#include <cmath>
#include <cstdint>

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
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[crane sycl] %s: %s\n", __func__, e.what());
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
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[crane sycl] %s: %s\n", __func__, e.what());
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
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[crane sycl] %s: %s\n", __func__, e.what());
    return 1;
  } catch (...) {
    return 1;
  }
}

// Fused top-K MoE routing, the SYCL port of `kernels/cuda/topk_moe.cu`:
// softmax over a token's expert logits, `top_k` rounds of sub-group argmax
// (ties to the smaller expert id), optional renormalization. One 16-lane
// sub-group per token; up to 32 * 16 = 512 experts (see `topk_moe.rs`'s
// `MAX_FUSED_EXPERTS`). `logits` [n_tokens, n_experts] F32 in, `out_ids`
// [n_tokens, top_k] U32 and `out_weights` [n_tokens, top_k] F32 out.
extern "C" int crane_topk_moe_sycl(void *queue, const float *logits, uint32_t *out_ids,
                                   float *out_weights, int n_tokens, int n_experts, int top_k,
                                   int norm_topk) {
  constexpr int SGW = 16;
  constexpr int MAX_EPT = 32;
  try {
    auto &q = *static_cast<sycl::queue *>(queue);
    q.parallel_for(
        sycl::nd_range<1>(size_t(n_tokens) * SGW, SGW),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SGW)]] {
          const auto sg = it.get_sub_group();
          const int token = int(it.get_group_linear_id());
          const int lane = int(sg.get_local_linear_id());
          const int ept = (n_experts + SGW - 1) / SGW;
          const float *row = logits + size_t(token) * n_experts;
          float wt[MAX_EPT];
          for (int i = 0; i < MAX_EPT; ++i) {
            const int e = lane + i * SGW;
            wt[i] = (i < ept && e < n_experts) ? row[e] : -INFINITY;
          }
          float mx = -INFINITY;
          for (int i = 0; i < ept; ++i) {
            mx = sycl::fmax(mx, wt[i]);
          }
          mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
          float sum = 0.f;
          for (int i = 0; i < ept; ++i) {
            const int e = lane + i * SGW;
            wt[i] = e < n_experts ? sycl::exp(wt[i] - mx) : 0.f;
            sum += wt[i];
          }
          sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
          for (int i = 0; i < ept; ++i) {
            wt[i] /= sum;
            // NaN never wins a comparison, so it would be re-selected forever.
            if (sycl::isnan(wt[i])) {
              wt[i] = -FLT_MAX;
            }
          }
          uint32_t *ids = out_ids + size_t(token) * top_k;
          float *ws = out_weights + size_t(token) * top_k;
          float picked = 0.f;
          for (int k = 0; k < top_k; ++k) {
            float best = -INFINITY;
            int best_e = n_experts;
            for (int i = 0; i < ept; ++i) {
              const int e = lane + i * SGW;
              if (e < n_experts && (wt[i] > best || (wt[i] == best && e < best_e))) {
                best = wt[i];
                best_e = e;
              }
            }
            for (int mask = SGW / 2; mask > 0; mask /= 2) {
              const float ov = sycl::permute_group_by_xor(sg, best, mask);
              const int oe = sycl::permute_group_by_xor(sg, best_e, mask);
              if (ov > best || (ov == best && oe < best_e)) {
                best = ov;
                best_e = oe;
              }
            }
            if (lane == 0) {
              ids[k] = uint32_t(best_e);
              ws[k] = best;
            }
            picked += best;
            if (best_e % SGW == lane) {
              wt[best_e / SGW] = -INFINITY;
            }
          }
          if (norm_topk && lane == 0) {
            const float inv = 1.f / (picked > 0.f ? picked : 1.f);
            for (int k = 0; k < top_k; ++k) {
              ws[k] *= inv;
            }
          }
        });
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[crane sycl] %s: %s\n", __func__, e.what());
    return 1;
  } catch (...) {
    return 1;
  }
}

// Element-wise `atan2(y, x)`, the SYCL counterpart of
// `kernels/cuda/atan2.cu` / `kernels/metal/atan2.metal`, computed in f32
// (`sycl::atan2` keeps IEEE quadrants and signed zeros; `atan2(0, 0) = 0`).
// dtype tags: 0 = f32, 1 = f16.
extern "C" int crane_atan2_sycl(void *queue, int dtype, const void *y, const void *x, void *out,
                                size_t n) {
  try {
    auto &q = *static_cast<sycl::queue *>(queue);
    auto run = [&](auto t) {
      using T = decltype(t);
      const T *yp = static_cast<const T *>(y);
      const T *xp = static_cast<const T *>(x);
      T *op = static_cast<T *>(out);
      q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        op[i] = static_cast<T>(sycl::atan2(static_cast<float>(yp[i]), static_cast<float>(xp[i])));
      });
    };
    if (dtype == 0) {
      run(float{});
    } else if (dtype == 1) {
      run(sycl::half{});
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

// `MoE` combine (`ops/fused_ops/moe_combine.rs`): `out[t, h] = sum_k w[t, k] *
// y[row(t, k), h]`, with `row(t, k) = rows[t * K + k]`, or `t * K + k` when
// `rows` is null. `y` is f32 (`y_f16 == 0`) or f16; `out` likewise by
// `out_f16`. The sum runs over `k` in order, in f32, as the op chain it
// replaces does. Returns 0 on success, 1 on a SYCL error.
extern "C" int crane_moe_combine_sycl(void *queue, const void *y, int y_f16, const uint32_t *rows,
                                      const float *w, void *out, int out_f16, int tokens, int K,
                                      int H) {
  try {
    auto &q = *static_cast<sycl::queue *>(queue);
    auto run = [&]<typename Y, typename O>(const Y *yp, O *op) {
      q.parallel_for(sycl::range<2>(size_t(tokens), size_t(H)), [=](sycl::id<2> id) {
        const size_t t = id[0], h = id[1];
        float acc = 0.f;
        for (int k = 0; k < K; ++k) {
          const size_t p = t * size_t(K) + size_t(k);
          const size_t r = rows ? size_t(rows[p]) : p;
          acc += w[p] * static_cast<float>(yp[r * size_t(H) + h]);
        }
        op[t * size_t(H) + h] = static_cast<O>(acc);
      });
    };
    using half = sycl::half;
    if (y_f16) {
      const auto *yp = static_cast<const half *>(y);
      out_f16 ? run(yp, static_cast<half *>(out)) : run(yp, static_cast<float *>(out));
    } else {
      const auto *yp = static_cast<const float *>(y);
      out_f16 ? run(yp, static_cast<half *>(out)) : run(yp, static_cast<float *>(out));
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[crane sycl] %s: %s\n", __func__, e.what());
    return 1;
  } catch (...) {
    return 1;
  }
}
