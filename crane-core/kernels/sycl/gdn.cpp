// Fused Gated Delta Net recurrence for the Intel SYCL backend — the counterpart
// of `kernels/cuda/gdn.cu`. Same math, same contiguous-f32 `[BH, S, *]` layouts.
// Built into `libcrane_gdn_sycl.so` by `crane-core/build.rs` (icpx, `--features
// sycl` only) and driven by `ops/gdn/sycl_backend.rs`.
//
// Each value column of each (batch*head) is an independent sequential
// recurrence, so one launch steps through the whole sequence, collapsing the
// per-timestep candle op graph into one submission per GDN layer per pass.
//
// For K = 64 / 128 / 256 a 16-lane sub-group owns four columns and splits
// their K rows across its lanes (`gdn_sg_launch`): the state stays in
// registers and there are 16x more work-items than columns. On an Arc Pro B70
// at 32 heads of 128 x 128 that is 0.9 us per token against 18 us for the
// per-column kernel (`gdn_launch`), where each work-item held a whole
// 128-float column — every register a lane has, so it spilled — and only
// 4096 work-items ran. Other K keep the per-column kernel.
//
// Layouts (all contiguous f32):
//   q, k     : [BH, S, K]   (q already pre-scaled by 1/sqrt(K) by the caller)
//   v, y     : [BH, S, V]
//   g, beta  : [BH, S]      (g is the log-decay; decay = exp(g))
//   state    : [BH, K, V]
//
// Recurrence per timestep t (matches the CPU reference exactly):
//   S      *= exp(g_t)
//   kv_mem  = sum_k S[k,:] * k_t[k]
//   delta   = (v_t - kv_mem) * beta_t
//   S[k,:] += k_t[k] * delta
//   y_t     = sum_k S[k,:] * q_t[k]
#include <sycl/sycl.hpp>

#include <cstdio>

#define GDN_MAX_K 256

namespace {

constexpr int SG = 16;        // sub-group width
constexpr int SG_PER_WG = 4;  // sub-groups per work-group

// Sub-group kernel: one 16-lane sub-group owns `C` state columns of one
// (batch*head), and lane `l` holds rows `l, l + 16, ...` (`R = K / 16` of
// them) of each, so the state stays in registers and `BH * V / C` sub-groups
// run in parallel. The two dot products per step (`S^T k`, `S^T q`) are
// sub-group reductions.
template <int R, int C>
void gdn_sg_launch(sycl::queue &q, const float *qp, const float *kp, const float *vp,
                   const float *gp, const float *bp, const float *st_in, float *st_out,
                   float *yp, int BH, int S, int V) {
  constexpr int K = R * SG;
  const int tiles = V / C;
  const size_t sgs = size_t(BH) * size_t(tiles);
  const size_t wgs = (sgs + SG_PER_WG - 1) / SG_PER_WG;
  q.parallel_for(
      sycl::nd_range<1>(wgs * SG_PER_WG * SG, SG_PER_WG * SG),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const size_t id = it.get_group(0) * SG_PER_WG + sg.get_group_linear_id();
        // Uniform across the sub-group, so the reductions below stay legal.
        if (id >= sgs) {
          return;
        }
        const int lane = int(sg.get_local_linear_id());
        const size_t bh = id / size_t(tiles);
        const int col0 = int(id % size_t(tiles)) * C;

        float st[C][R];
        const float *sti = st_in + bh * K * V;
        for (int r = 0; r < R; ++r) {
          for (int c = 0; c < C; ++c) {
            st[c][r] = sti[size_t(lane + SG * r) * V + col0 + c];
          }
        }
        const float *qb = qp + bh * S * K;
        const float *kb = kp + bh * S * K;
        const float *vb = vp + bh * S * V;
        const float *gb = gp + bh * S;
        const float *bb = bp + bh * S;
        float *yb = yp + bh * S * V;

        for (int t = 0; t < S; ++t) {
          const float decay = sycl::exp(gb[t]);
          const float beta_t = bb[t];
          float kt[R], qt[R];
          for (int r = 0; r < R; ++r) {
            kt[r] = kb[size_t(t) * K + lane + SG * r];
            qt[r] = qb[size_t(t) * K + lane + SG * r];
          }
          float kv[C];
          for (int c = 0; c < C; ++c) {
            float acc = 0.f;
            for (int r = 0; r < R; ++r) {
              st[c][r] *= decay;
              acc += st[c][r] * kt[r];
            }
            kv[c] = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
          }
          float y[C];
          for (int c = 0; c < C; ++c) {
            const float delta = (vb[size_t(t) * V + col0 + c] - kv[c]) * beta_t;
            float acc = 0.f;
            for (int r = 0; r < R; ++r) {
              st[c][r] += kt[r] * delta;
              acc += st[c][r] * qt[r];
            }
            y[c] = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
          }
          for (int c = 0; c < C; ++c) {
            if (lane == c) {
              yb[size_t(t) * V + col0 + c] = y[c];
            }
          }
        }

        float *sto = st_out + bh * K * V;
        for (int r = 0; r < R; ++r) {
          for (int c = 0; c < C; ++c) {
            sto[size_t(lane + SG * r) * V + col0 + c] = st[c][r];
          }
        }
      });
}

// Columns per sub-group: on the B70, 4 beats 1, 2 and 8 (0.9 vs 1.5, 1.0
// and 2.2 us per token at 128 x 128): fewer leaves the reductions' latency
// exposed, more spills.
constexpr int SG_COLS = 4;

// The fused form of `gdn_sg_launch` for the model's own tensors: it reads Q,
// K and V straight from the conv output and computes everything the caller
// would otherwise launch one candle op each for (~20 per GDN layer per
// token, which bounds decode on SYCL): the key-head to value-head mapping,
// the L2 norms of Q and K, the `1/sqrt(K)` query scale, `beta = sigmoid(b)`
// and `g = -exp(A_log) * softplus(a + dt_bias)`. Activations and `y` are `T`
// (the model dtype); the state and all arithmetic are f32.
//
//   qkv     : [B, S, conv_dim]  Q (Hk heads), then K (Hk heads), then V
//   a, b    : [B, S, Hv]
//   neg_a, dt_bias : [Hv] f32 (`-exp(A_log)`, `dt_bias`)
//   y       : [B, S, Hv, V]
//   state   : [B, Hv, K, V] f32
//
// Value head `h` reads key head `h % Hk` when `chunked` (llama.cpp GGUF
// order), `h / (Hv / Hk)` otherwise (HF order).
template <int R, int C, typename T>
void gdn_fused_launch(sycl::queue &q, const T *qkv, const T *ap, const T *bp, const float *neg_a,
                      const float *dt_bias, const float *st_in, float *st_out, T *yp, int B,
                      int S, int Hk, int Hv, int V, int conv_dim, int key_dim, bool chunked) {
  constexpr int K = R * SG;
  const int tiles = V / C;
  const size_t sgs = size_t(B) * Hv * tiles;
  const size_t wgs = (sgs + SG_PER_WG - 1) / SG_PER_WG;
  const int per_group = Hv / Hk;
  q.parallel_for(
      sycl::nd_range<1>(wgs * SG_PER_WG * SG, SG_PER_WG * SG),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const size_t id = it.get_group(0) * SG_PER_WG + sg.get_group_linear_id();
        if (id >= sgs) {
          return;
        }
        const int lane = int(sg.get_local_linear_id());
        const size_t bh = id / size_t(tiles);
        const int col0 = int(id % size_t(tiles)) * C;
        const int b = int(bh / size_t(Hv));
        const int h = int(bh % size_t(Hv));
        const int kh = chunked ? h % Hk : h / per_group;
        const float q_scale = sycl::rsqrt(float(K));
        const float na = neg_a[h];
        const float dtb = dt_bias[h];

        float st[C][R];
        const float *sti = st_in + bh * K * V;
        for (int r = 0; r < R; ++r) {
          for (int c = 0; c < C; ++c) {
            st[c][r] = sti[size_t(lane + SG * r) * V + col0 + c];
          }
        }

        for (int t = 0; t < S; ++t) {
          const size_t row = size_t(b) * S + t;
          const T *xr = qkv + row * conv_dim;
          float kt[R], qt[R];
          float ssk = 0.f, ssq = 0.f;
          for (int r = 0; r < R; ++r) {
            kt[r] = static_cast<float>(xr[key_dim + kh * K + lane + SG * r]);
            qt[r] = static_cast<float>(xr[kh * K + lane + SG * r]);
            ssk += kt[r] * kt[r];
            ssq += qt[r] * qt[r];
          }
          const float k_inv = sycl::rsqrt(sycl::reduce_over_group(sg, ssk, sycl::plus<float>()) + 1e-6f);
          const float q_inv =
              q_scale * sycl::rsqrt(sycl::reduce_over_group(sg, ssq, sycl::plus<float>()) + 1e-6f);
          for (int r = 0; r < R; ++r) {
            kt[r] *= k_inv;
            qt[r] *= q_inv;
          }
          const float beta_t = 1.f / (1.f + sycl::exp(-static_cast<float>(bp[row * Hv + h])));
          const float x = static_cast<float>(ap[row * Hv + h]) + dtb;
          // softplus, without the overflow of a literal log(1 + exp(x)).
          const float softplus = x > 20.f ? x : sycl::log1p(sycl::exp(x));
          const float decay = sycl::exp(na * softplus);

          const T *vr = xr + 2 * key_dim + size_t(h) * V + col0;
          float kv[C];
          for (int c = 0; c < C; ++c) {
            float acc = 0.f;
            for (int r = 0; r < R; ++r) {
              st[c][r] *= decay;
              acc += st[c][r] * kt[r];
            }
            kv[c] = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
          }
          float y[C];
          for (int c = 0; c < C; ++c) {
            const float delta = (static_cast<float>(vr[c]) - kv[c]) * beta_t;
            float acc = 0.f;
            for (int r = 0; r < R; ++r) {
              st[c][r] += kt[r] * delta;
              acc += st[c][r] * qt[r];
            }
            y[c] = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
          }
          T *yr = yp + (row * Hv + h) * V + col0;
          for (int c = 0; c < C; ++c) {
            if (lane == c) {
              yr[c] = static_cast<T>(y[c]);
            }
          }
        }

        float *sto = st_out + bh * K * V;
        for (int r = 0; r < R; ++r) {
          for (int c = 0; c < C; ++c) {
            sto[size_t(lane + SG * r) * V + col0 + c] = st[c][r];
          }
        }
      });
}

} // namespace

template <int KT>
static void gdn_launch(sycl::queue &q, const float *qp, const float *kp,
                       const float *vp, const float *gp, const float *bp,
                       const float *st_in, float *st_out, float *yp, int BH,
                       int S, int Kr, int V, int V_TILE) {
  const int K = (KT > 0) ? KT : Kr;
  const int tiles = (V + V_TILE - 1) / V_TILE;
  const std::size_t local = static_cast<std::size_t>(V_TILE);
  const std::size_t groups = static_cast<std::size_t>(BH) * tiles;

  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * local),
                          sycl::range<1>(local)),
        [=](sycl::nd_item<1> it) {
          const int gid = static_cast<int>(it.get_global_id(0));
          const int bh = gid / V_TILE / tiles;
          const int rem = gid / V_TILE % tiles;
          const int lid = gid % V_TILE;
          const int vcol = rem * V_TILE + lid;
          if (bh >= BH || vcol >= V)
            return;

          float Scol[(KT > 0) ? KT : GDN_MAX_K];
          const float *sti = st_in + static_cast<long long>(bh) * K * V;
          for (int kk = 0; kk < K; ++kk)
            Scol[kk] = sti[kk * V + vcol];

          const float *qb = qp + static_cast<long long>(bh) * S * K;
          const float *kb = kp + static_cast<long long>(bh) * S * K;
          const float *vb = vp + static_cast<long long>(bh) * S * V;
          const float *gb = gp + static_cast<long long>(bh) * S;
          const float *bb = bp + static_cast<long long>(bh) * S;
          float *yb = yp + static_cast<long long>(bh) * S * V;

          for (int t = 0; t < S; ++t) {
            const float decay = sycl::exp(gb[t]);
            const float beta_t = bb[t];
            const float v_t = vb[t * V + vcol];
            const float *kt = kb + static_cast<long long>(t) * K;
            const float *qt = qb + static_cast<long long>(t) * K;

            float kv = 0.f;
            for (int kk = 0; kk < K; ++kk) {
              Scol[kk] *= decay;
              kv += Scol[kk] * kt[kk];
            }
            const float delta = (v_t - kv) * beta_t;

            float y = 0.f;
            for (int kk = 0; kk < K; ++kk) {
              Scol[kk] += kt[kk] * delta;
              y += Scol[kk] * qt[kk];
            }
            yb[t * V + vcol] = y;
          }

          float *sto = st_out + static_cast<long long>(bh) * K * V;
          for (int kk = 0; kk < K; ++kk)
            sto[kk * V + vcol] = Scol[kk];
        });
  });
}

extern "C" int crane_gdn_recurrence_sycl(void *queue, const float *q,
                                        const float *k, const float *v,
                                        const float *g, const float *beta,
                                        const float *state_in, float *state_out,
                                        float *y, int BH, int S, int K, int V,
                                        int V_TILE) {
  try {
    auto &sq = *static_cast<sycl::queue *>(queue);
    if (V_TILE <= 0 || V_TILE > V)
      V_TILE = V;
    if (V % SG_COLS == 0 && (K == 64 || K == 128 || K == 256)) {
      if (K == 64)
        gdn_sg_launch<4, SG_COLS>(sq, q, k, v, g, beta, state_in, state_out, y, BH, S, V);
      else if (K == 128)
        gdn_sg_launch<8, SG_COLS>(sq, q, k, v, g, beta, state_in, state_out, y, BH, S, V);
      else
        gdn_sg_launch<16, SG_COLS>(sq, q, k, v, g, beta, state_in, state_out, y, BH, S, V);
      return 0;
    }
    if (K == 128)
      gdn_launch<128>(sq, q, k, v, g, beta, state_in, state_out, y, BH, S, 128, V,
                      V_TILE);
    else if (K > 0 && K <= GDN_MAX_K)
      gdn_launch<0>(sq, q, k, v, g, beta, state_in, state_out, y, BH, S, K, V,
                    V_TILE);
    else
      return 2; // unsupported head_k_dim
    // No wait: candle's SYCL queue is in-order, so the result buffers are
    // correctly ordered against every later op the launcher queues on it.
    return 0;
  } catch (const sycl::exception &) {
    return 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[crane sycl] %s: %s\n", __func__, e.what());
    return 1;
  } catch (...) {
    return 1;
  }
}

// The fused step (`gdn_fused_launch`). `dtype` tags `qkv`, `a`, `b` and `y`:
// 0 = f32, 1 = f16. Returns 0 on success, 1 on a SYCL error, 2 for an
// unsupported head dim, value width or dtype (the caller then takes the
// unfused path).
extern "C" int crane_gdn_fused_sycl(void *queue, int dtype, const void *qkv, const void *a,
                                    const void *b, const float *neg_a, const float *dt_bias,
                                    const float *state_in, float *state_out, void *y, int B,
                                    int S, int Hk, int Hv, int K, int V, int conv_dim,
                                    int key_dim, int chunked) {
  try {
    auto &sq = *static_cast<sycl::queue *>(queue);
    if (V % SG_COLS != 0 || Hk <= 0 || Hv % Hk != 0 || (dtype != 0 && dtype != 1)) {
      return 2;
    }
    auto run = [&]<int R, typename T>() {
      gdn_fused_launch<R, SG_COLS, T>(sq, static_cast<const T *>(qkv), static_cast<const T *>(a),
                                      static_cast<const T *>(b), neg_a, dt_bias, state_in,
                                      state_out, static_cast<T *>(y), B, S, Hk, Hv, V, conv_dim,
                                      key_dim, chunked != 0);
    };
    auto by_dtype = [&]<int R>() {
      if (dtype == 0) {
        run.template operator()<R, float>();
      } else {
        run.template operator()<R, sycl::half>();
      }
    };
    switch (K) {
    case 64: by_dtype.template operator()<4>(); return 0;
    case 128: by_dtype.template operator()<8>(); return 0;
    case 256: by_dtype.template operator()<16>(); return 0;
    default: return 2;
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[crane sycl] %s: %s\n", __func__, e.what());
    return 1;
  } catch (...) {
    return 1;
  }
}
