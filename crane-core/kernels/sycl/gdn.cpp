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
