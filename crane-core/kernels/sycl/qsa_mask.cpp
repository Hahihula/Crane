// QSA indexer block selection (Qwen4-Exp) for the Intel SYCL backend, the
// counterpart of `kernels/cuda/qsa_mask.cu` / `kernels/metal/qsa_mask.metal`.
// Driven by `ops/fused_ops/qsa_mask.rs`; built into `libcrane_gdn_sycl.so`
// by `crane-core/build.rs` (icpx, `--features sycl` only).
//
// From per-block scores `[seq, blocks]`, build the additive attention mask
// `[seq, cells]`: 0 on the `keep` best complete blocks each query can see
// plus its incomplete tail, -inf elsewhere. One work-group per query row. The
// `keep`-th largest score is found by a bitwise search over order-preserving
// integer keys (32 passes, each counting keys >= candidate across the row);
// blocks above it are all kept and ties at it are kept earliest-block-first,
// the total order the host path uses (score descending, then block ascending).
#include <sycl/sycl.hpp>

#include <cstdio>

#include <cmath>
#include <cstdint>

namespace {

constexpr int WG = 256;

// Order-preserving float -> uint (larger float, larger key).
inline uint32_t qsa_key(float v) {
  const uint32_t u = sycl::bit_cast<uint32_t>(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

} // namespace

extern "C" int crane_qsa_mask_sycl(void *queue, const float *scores, float *mask, int seq,
                                   int blocks, int cells, int start, int ratio, int keep) {
  try {
    auto &q = *static_cast<sycl::queue *>(queue);
    q.parallel_for(sycl::nd_range<1>(size_t(seq) * WG, WG), [=](sycl::nd_item<1> it) {
      const auto g = it.get_group();
      const int row = int(it.get_group_linear_id());
      const int tid = int(it.get_local_linear_id());
      const int pos = start + row;
      const int visible = (pos + 1) / ratio; // complete blocks this query sees
      const float *s = scores + size_t(row) * blocks;
      float *out = mask + size_t(row) * cells;

      for (int c = tid; c < cells; c += WG) {
        out[c] = -INFINITY;
      }
      // Kept blocks and the tail are written over the cells filled above.
      sycl::group_barrier(g);
      // The incomplete tail block, up to and including the query itself.
      for (int c = visible * ratio + tid; c <= pos; c += WG) {
        out[c] = 0.f;
      }

      // `visible` is uniform per row, so every branch below is too.
      const bool select_all = visible <= keep;
      uint32_t threshold = 0; // key of the keep-th largest score
      int above = 0;          // blocks strictly above it
      if (!select_all) {
        uint32_t prefix = 0;
        for (int bit = 31; bit >= 0; --bit) {
          const uint32_t candidate = prefix | (1u << bit);
          int local = 0;
          for (int b = tid; b < visible; b += WG) {
            local += qsa_key(s[b]) >= candidate;
          }
          if (sycl::reduce_over_group(g, local, sycl::plus<int>()) >= keep) {
            prefix = candidate;
          }
        }
        threshold = prefix;
        int local = 0;
        for (int b = tid; b < visible; b += WG) {
          local += qsa_key(s[b]) > threshold;
        }
        above = sycl::reduce_over_group(g, local, sycl::plus<int>());
      }
      const int need_eq = select_all ? 0 : keep - above;

      // Walk the row in chunks of WG blocks so ties at the threshold are
      // taken in block order.
      int taken = 0;
      for (int base = 0; base < visible; base += WG) {
        const int b = base + tid;
        bool selected = false;
        bool eq = false;
        if (b < visible) {
          if (select_all) {
            selected = true;
          } else {
            const uint32_t k = qsa_key(s[b]);
            selected = k > threshold;
            eq = k == threshold;
          }
        }
        const int before =
            taken + sycl::exclusive_scan_over_group(g, int(eq), sycl::plus<int>());
        if (eq && before < need_eq) {
          selected = true;
        }
        if (selected) {
          for (int c = b * ratio; c < (b + 1) * ratio; ++c) {
            out[c] = 0.f;
          }
        }
        taken += sycl::reduce_over_group(g, int(eq), sycl::plus<int>());
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
