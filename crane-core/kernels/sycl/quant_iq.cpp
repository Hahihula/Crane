// llama.cpp i-quant weight kernels for the Intel SYCL backend: IQ4_NL,
// IQ4_XS, IQ2_S, IQ3_XXS, IQ3_S and Q2_0 (see
// `crane-core/src/quantized/iquant.rs` for the block layouts). Built into
// `libcrane_gdn_sycl.so` by `crane-core/build.rs` (icpx, `--features sycl`
// only) and driven by `ops/quant_iq/sycl.rs`.
//
// Every format here splits a row into 32-value chunks that decode on their
// own (an IQ4_NL block, a Q2_0 half-block, or one sub-block of a 256-value
// super-block), so both entry points are written against one per-type
// `decode32`:
//
// - `matvec`: one 16-lane sub-group per output row; lanes stride over the
//   row's chunks, decode, dot against the f32 activations, and reduce. With
//   an expert-id array it is the `MoE` "matmul by id": input row `p` uses
//   expert `ids[p]`'s slice of a packed `[experts, rows, cols]` tensor and
//   activation row `p / x_div`, so routing never leaves the device.
// - `dequant`: one work-item per chunk, optionally over a list of experts,
//   feeding prefill-sized matmuls (oneMKL GEMMs on the XMX engines).
//
// Activations are not quantized to int8 (no portable `dp4a` across Intel GPU
// generations); the dot product is plain f32. Decoders follow the CPU
// reference's float operation order, so `dequant` is bit-identical to it.
#include <sycl/sycl.hpp>

#include <cstdio>

#include "iq_grids.h"

namespace {

using namespace crane_iq;

constexpr int QK_K = 256;
constexpr int SG = 16;          // sub-group width
constexpr int ROWS_PER_WG = 4;  // output rows (sub-groups) per work-group

// Type tags match `IQuantType::sycl_tag` in `ops/quant_iq/sycl.rs`.
enum Ty { IQ4_NL = 0, IQ4_XS = 1, IQ2_S = 2, IQ3_XXS = 3, IQ3_S = 4, Q2_0 = 5 };
// dtype tags match `dtype_tag` there.
enum { OUT_F32 = 0, OUT_F16 = 1 };

template <int TY> constexpr int block_values() {
  return TY == IQ4_NL ? 32 : TY == Q2_0 ? 64 : QK_K;
}
template <int TY> constexpr int block_bytes() {
  switch (TY) {
  case IQ4_NL: return 2 + 16;
  case IQ4_XS: return 2 + 2 + QK_K / 64 + QK_K / 2;
  case IQ2_S: return 2 + QK_K / 4 + QK_K / 16;
  case IQ3_XXS: return 2 + 3 * QK_K / 8;
  case IQ3_S: return 2 + 13 * QK_K / 32 + QK_K / 64;
  default: return 2 + 64 / 4; // Q2_0
  }
}

inline constexpr float kvalues_iq4nl[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f,
                                            -22.f,  -10.f,  1.f,   13.f,  25.f,  38.f,
                                            53.f,   69.f,   89.f,  113.f};

// `p` points at a little-endian f16; no alignment assumed.
inline float load_half(const uint8_t *p) {
  sycl::half h;
  auto *hp = reinterpret_cast<uint8_t *>(&h);
  hp[0] = p[0];
  hp[1] = p[1];
  return static_cast<float>(h);
}

inline float grid_byte(uint64_t entry, int j) { return float((entry >> (8 * j)) & 0xff); }
inline float signed_val(float v, uint8_t signs, int j) { return (signs >> j) & 1 ? -v : v; }
// ggml `ksigns_iq2xs`: seven sign bits plus one keeping the negatives even.
inline uint8_t ksigns(uint32_t bits7) {
  const uint32_t b = bits7 & 127;
  return uint8_t(b | ((sycl::popcount(b) & 1) << 7));
}

// Decodes chunk `c` (values 32c..32c+31) of `row` into `w`.
template <int TY> inline void decode32(const uint8_t *row, int c, float *w) {
  if constexpr (TY == IQ4_NL) {
    const uint8_t *blk = row + size_t(c) * block_bytes<TY>();
    const float d = load_half(blk);
    for (int j = 0; j < 16; ++j) {
      w[j] = d * kvalues_iq4nl[blk[2 + j] & 0xf];
      w[j + 16] = d * kvalues_iq4nl[blk[2 + j] >> 4];
    }
  } else if constexpr (TY == Q2_0) {
    const uint8_t *blk = row + size_t(c / 2) * block_bytes<TY>();
    const float d = load_half(blk);
    const uint8_t *qs = blk + 2 + 8 * (c % 2);
    for (int j = 0; j < 32; ++j) {
      w[j] = d * (float((qs[j / 4] >> (2 * (j % 4))) & 3) - 1.f);
    }
  } else {
    const uint8_t *blk = row + size_t(c / 8) * block_bytes<TY>();
    const int ib = c % 8;
    const float d = load_half(blk);
    if constexpr (TY == IQ4_XS) {
      const int scales_h = int(blk[2]) | (int(blk[3]) << 8);
      const int lo = (blk[4 + ib / 2] >> (4 * (ib % 2))) & 0xf;
      const int hi = (scales_h >> (2 * ib)) & 3;
      const float dl = d * float((lo | (hi << 4)) - 32);
      const uint8_t *q = blk + 4 + QK_K / 64 + 16 * ib;
      for (int j = 0; j < 16; ++j) {
        w[j] = dl * kvalues_iq4nl[q[j] & 0xf];
        w[j + 16] = dl * kvalues_iq4nl[q[j] >> 4];
      }
    } else if constexpr (TY == IQ2_S) {
      const uint8_t *qs = blk + 2;
      const uint8_t *signs = qs + QK_K / 8;
      const uint8_t *qh = blk + 2 + QK_K / 4;
      const uint8_t sc = blk[2 + QK_K / 4 + QK_K / 32 + ib];
      const float db[2] = {d * (0.5f + float(sc & 0xf)) * 0.25f,
                           d * (0.5f + float(sc >> 4)) * 0.25f};
      for (int l = 0; l < 4; ++l) {
        const int idx = qs[4 * ib + l] | ((int(qh[ib]) << (8 - 2 * l)) & 0x300);
        const uint64_t g = iq2s_grid[idx];
        const uint8_t s = signs[4 * ib + l];
        for (int j = 0; j < 8; ++j) {
          w[8 * l + j] = signed_val(db[l / 2] * grid_byte(g, j), s, j);
        }
      }
    } else if constexpr (TY == IQ3_XXS) {
      const uint8_t *qs = blk + 2 + 8 * ib;
      const uint8_t *sas = blk + 2 + QK_K / 4 + 4 * ib;
      const uint32_t aux =
          uint32_t(sas[0]) | (uint32_t(sas[1]) << 8) | (uint32_t(sas[2]) << 16) | (uint32_t(sas[3]) << 24);
      const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
      for (int l = 0; l < 4; ++l) {
        const uint8_t s = ksigns(aux >> (7 * l));
        const uint64_t g1 = iq3xxs_grid[qs[2 * l]];
        const uint64_t g2 = iq3xxs_grid[qs[2 * l + 1]];
        for (int j = 0; j < 4; ++j) {
          w[8 * l + j] = signed_val(db * grid_byte(g1, j), s, j);
          w[8 * l + 4 + j] = signed_val(db * grid_byte(g2, j), s, j + 4);
        }
      }
    } else { // IQ3_S
      const uint8_t *qs = blk + 2 + 8 * ib;
      const int qh = blk[2 + QK_K / 4 + ib];
      const uint8_t *signs = blk + 2 + QK_K / 4 + QK_K / 32 + 4 * ib;
      const int sc = (blk[2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + ib / 2] >> (4 * (ib % 2))) & 0xf;
      const float db = d * float(1 + 2 * sc);
      for (int l = 0; l < 4; ++l) {
        const uint64_t g1 = iq3s_grid[qs[2 * l] | ((qh << (8 - 2 * l)) & 256)];
        const uint64_t g2 = iq3s_grid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
        for (int j = 0; j < 4; ++j) {
          w[8 * l + j] = signed_val(db * grid_byte(g1, j), signs[l], j);
          w[8 * l + 4 + j] = signed_val(db * grid_byte(g2, j), signs[l], j + 4);
        }
      }
    }
  }
}

template <int TY> inline size_t row_bytes(int cols) {
  return size_t(cols / block_values<TY>()) * block_bytes<TY>();
}

template <int TY, typename T>
void matvec_launch(sycl::queue &q, const uint8_t *packed, size_t expert_stride,
                   const uint32_t *ids, int x_div, const float *input, T *output,
                   int pairs, int out_rows, int cols) {
  const size_t rb = row_bytes<TY>(cols);
  const int chunks = cols / 32;
  const size_t row_groups = size_t((out_rows + ROWS_PER_WG - 1) / ROWS_PER_WG);
  q.parallel_for(
      sycl::nd_range<2>({size_t(pairs), row_groups * ROWS_PER_WG * SG}, {1, ROWS_PER_WG * SG}),
      [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int p = int(it.get_global_id(0));
        const int row = int(it.get_group(1)) * ROWS_PER_WG + int(sg.get_group_linear_id());
        // Uniform across the sub-group, so the reduction below stays legal.
        if (row >= out_rows) {
          return;
        }
        const int lane = int(sg.get_local_linear_id());
        const size_t expert = ids ? size_t(ids[p]) : 0;
        const uint8_t *r = packed + expert * expert_stride + size_t(row) * rb;
        const float *x = input + size_t(p / x_div) * cols;
        float acc = 0.f;
        float w[32];
        for (int c = lane; c < chunks; c += SG) {
          decode32<TY>(r, c, w);
          const float *xc = x + 32 * c;
          for (int j = 0; j < 32; ++j) {
            acc += w[j] * xc[j];
          }
        }
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (lane == 0) {
          output[size_t(p) * out_rows + row] = static_cast<T>(acc);
        }
      });
}

// Stores the 32 decoded values with vector stores (the output chunk is
// 32-element aligned); scalar stores of f16 halve the kernel's throughput.
template <typename T> inline void store32(T *out, const float *w) {
  constexpr int V = 16 / sizeof(T);
  for (int j = 0; j < 32; j += V) {
    sycl::vec<T, V> v;
    for (int i = 0; i < V; ++i) {
      v[i] = static_cast<T>(w[j + i]);
    }
    *reinterpret_cast<sycl::vec<T, V> *>(out + j) = v;
  }
}

// Decodes `n_mats` matrices of `n_rows` x `cols`: matrix `m` is expert
// `ids[m]` of `packed` (`expert_stride` bytes apart), or the only one when
// `ids` is null. Output is `[n_mats, n_rows, cols]`.
template <int TY, typename T>
void dequant_launch(sycl::queue &q, const uint8_t *packed, size_t expert_stride,
                    const uint32_t *ids, T *output, int n_mats, int n_rows, int cols) {
  const size_t rb = row_bytes<TY>(cols);
  const int chunks = cols / 32;
  q.parallel_for(sycl::range<2>(size_t(n_mats) * n_rows, size_t(chunks)), [=](sycl::id<2> idx) {
    const size_t mat_row = idx[0];
    const int c = int(idx[1]);
    const size_t m = mat_row / size_t(n_rows);
    const size_t row = mat_row % size_t(n_rows);
    const size_t expert = ids ? size_t(ids[m]) : 0;
    float w[32];
    decode32<TY>(packed + expert * expert_stride + row * rb, c, w);
    store32(output + mat_row * cols + 32 * c, w);
  });
}

// Calls `f.template operator()<TY>()` for the runtime type tag; false if unknown.
template <typename F> bool dispatch_ty(int ty, F &&f) {
  switch (ty) {
  case IQ4_NL: f.template operator()<IQ4_NL>(); return true;
  case IQ4_XS: f.template operator()<IQ4_XS>(); return true;
  case IQ2_S: f.template operator()<IQ2_S>(); return true;
  case IQ3_XXS: f.template operator()<IQ3_XXS>(); return true;
  case IQ3_S: f.template operator()<IQ3_S>(); return true;
  case Q2_0: f.template operator()<Q2_0>(); return true;
  default: return false;
  }
}

} // namespace

// `output[p, r] = dot(W_e[r], input[p / x_div])` for `p < pairs`, where
// `W_e` is expert `e = ids[p]` of `packed` (`expert_stride` bytes apart), or
// the only matrix when `ids` is null. Returns 0 on success, 1 on a SYCL
// error, 2 for an unknown dtype or type tag.
extern "C" int crane_iq_matvec_sycl(void *queue, int ty, int dtype, const void *packed,
                                    size_t expert_stride, const void *ids, int x_div,
                                    const void *input, void *output, int pairs,
                                    int out_rows, int cols) {
  try {
    auto &sq = *static_cast<sycl::queue *>(queue);
    const auto *p = static_cast<const uint8_t *>(packed);
    const auto *id = static_cast<const uint32_t *>(ids);
    const auto *x = static_cast<const float *>(input);
    bool ok = false;
    if (dtype == OUT_F32) {
      ok = dispatch_ty(ty, [&]<int TY>() {
        matvec_launch<TY, float>(sq, p, expert_stride, id, x_div, x,
                                 static_cast<float *>(output), pairs, out_rows, cols);
      });
    } else if (dtype == OUT_F16) {
      ok = dispatch_ty(ty, [&]<int TY>() {
        matvec_launch<TY, sycl::half>(sq, p, expert_stride, id, x_div, x,
                                      static_cast<sycl::half *>(output), pairs, out_rows, cols);
      });
    }
    return ok ? 0 : 2;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[crane sycl] %s: %s\n", __func__, e.what());
    return 1;
  } catch (...) {
    return 1;
  }
}

// Decodes `n_mats` packed `[n_rows, cols]` matrices into `output`: expert
// `ids[m]` of `packed` for each `m`, or just `packed` when `ids` is null.
extern "C" int crane_iq_dequant_sycl(void *queue, int ty, int dtype, const void *packed,
                                     size_t expert_stride, const void *ids, int n_mats,
                                     void *output, int n_rows, int cols) {
  try {
    auto &sq = *static_cast<sycl::queue *>(queue);
    const auto *p = static_cast<const uint8_t *>(packed);
    const auto *id = static_cast<const uint32_t *>(ids);
    bool ok = false;
    if (dtype == OUT_F32) {
      ok = dispatch_ty(ty, [&]<int TY>() {
        dequant_launch<TY, float>(sq, p, expert_stride, id, static_cast<float *>(output),
                                  n_mats, n_rows, cols);
      });
    } else if (dtype == OUT_F16) {
      ok = dispatch_ty(ty, [&]<int TY>() {
        dequant_launch<TY, sycl::half>(sq, p, expert_stride, id,
                                       static_cast<sycl::half *>(output), n_mats, n_rows, cols);
      });
    }
    return ok ? 0 : 2;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[crane sycl] %s: %s\n", __func__, e.what());
    return 1;
  } catch (...) {
    return 1;
  }
}
