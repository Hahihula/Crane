// SPDX-License-Identifier: MIT

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

struct quant_ternary_pq2_block {
    uint16_t d;
    uint8_t qs[32];
};

struct quant_ternary_ptq1_block {
    uint8_t qs[24];
    uint8_t qh[2];
    uint16_t d;
};

__device__ __forceinline__ float quant_ternary_half(uint16_t bits) {
    return __half2float(*reinterpret_cast<const half *>(&bits));
}

__device__ __forceinline__ int quant_ternary_ptq1_trit(const quant_ternary_ptq1_block * block, int e) {
    uint8_t value;
    int digit;
    if (e < 80) {
        value = block->qs[e & 15];
        digit = e >> 4;
    } else if (e < 120) {
        const int t = e - 80;
        value = block->qs[16 + (t & 7)];
        digit = t >> 3;
    } else {
        const int t = e - 120;
        value = block->qh[t & 1];
        digit = t >> 1;
    }
    // The PTQ1 packing uses at most five base-3 digits. Spell this out so
    // decode does not carry a tiny variable-trip loop in every dot product.
    if (digit > 0) value *= 3;
    if (digit > 1) value *= 3;
    if (digit > 2) value *= 3;
    if (digit > 3) value *= 3;
    return int((uint16_t(value) * 3) >> 8) - 1;
}

__device__ __forceinline__ float quant_ternary_block_sum(float value) {
    constexpr unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    __shared__ float warp_sums[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(mask, value, offset);
        }
    }
    return value;
}

extern "C" __global__ void quant_ternary_hadamard_f32(
    const float * src,
    const float * signs,
    float * dst,
    int rows,
    int width,
    int block_size,
    int perm_hd,
    int perm_nk,
    int perm_rep) {
    extern __shared__ float values[];
    const int chunk = blockIdx.x;
    const int row = chunk / (width / block_size);
    const int block = chunk % (width / block_size);
    if (row >= rows) return;
    const int base = row * width + block * block_size;
    const float scale = rsqrtf(float(block_size));
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        const int dst_col = block * block_size + i;
        int src_col = dst_col;
        if (perm_rep > 1) {
            const int dim = dst_col % perm_hd;
            const int tmp = dst_col / perm_hd;
            const int rep = tmp % perm_rep;
            const int key = tmp / perm_rep;
            src_col = dim + perm_hd * (key + perm_nk * rep);
        }
        values[i] = src[row * width + src_col] * signs[dst_col] * scale;
    }
    __syncthreads();
    for (int h = 1; h < block_size; h <<= 1) {
        for (int idx = threadIdx.x; idx < block_size / 2; idx += blockDim.x) {
            const int j = (idx / h) * 2 * h + idx % h;
            const float x = values[j];
            const float y = values[j + h];
            values[j] = x + y;
            values[j + h] = x - y;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        dst[base + i] = values[i];
    }
}

extern "C" __global__ void quant_ternary_pq2_matvec_f32(
    const uint8_t * packed,
    const float * input,
    float * output,
    int input_rows,
    int output_rows,
    int cols) {
    const int out = blockIdx.x;
    const int row = blockIdx.y;
    if (out >= output_rows || row >= input_rows) return;
    const auto * blocks = reinterpret_cast<const quant_ternary_pq2_block *>(packed);
    const int blocks_per_row = cols / 128;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const auto * block = blocks + out * blocks_per_row + col / 128;
        const int e = col & 127;
        const int q = (block->qs[e >> 2] >> (2 * (e & 3))) & 3;
        sum += float(q - 1) * quant_ternary_half(block->d) * input[row * cols + col];
    }
    sum = quant_ternary_block_sum(sum);
    if (threadIdx.x == 0) output[row * output_rows + out] = sum;
}

extern "C" __global__ void quant_ternary_ptq1_matvec_f32(
    const uint8_t * packed,
    const float * input,
    float * output,
    int input_rows,
    int output_rows,
    int cols) {
    const int out = blockIdx.x;
    const int row = blockIdx.y;
    if (out >= output_rows || row >= input_rows) return;
    const auto * blocks = reinterpret_cast<const quant_ternary_ptq1_block *>(packed);
    const int blocks_per_row = cols / 128;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const auto * block = blocks + out * blocks_per_row + col / 128;
        sum += float(quant_ternary_ptq1_trit(block, col & 127)) * quant_ternary_half(block->d) * input[row * cols + col];
    }
    sum = quant_ternary_block_sum(sum);
    if (threadIdx.x == 0) output[row * output_rows + out] = sum;
}
