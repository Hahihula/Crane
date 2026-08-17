//! Cost of the GQA expansion in the Qwen 3.5/3.8 decode attention path.
//!
//! `FullAttention::forward` materializes three full-context tensors per
//! full-attention layer per decoded token: `k_rep` and `v_rep` (the 4→24 head
//! GQA expansion) and `k_t` (a transposed copy of `k_rep`). All three are
//! O(context length), so their cost grows linearly with depth even though a
//! decode step processes a single token — which is why Qwen3.8-27B decode
//! falls from 16.9 t/s at 512 tokens of context to 8.6 t/s at 4096, while
//! llama.cpp (which never materializes the expansion) stays flat.
//!
//! ```sh
//! cargo test -p crane-core --release --features cuda \
//!   --test qwen3_5_gqa_expand_cost -- --ignored --nocapture
//! ```
use candle_core::{D, DType, Device, Tensor};

// Qwen3.8-27B full-attention geometry.
const KV_HEADS: usize = 4;
const N_REP: usize = 6; // 24 query heads / 4 KV heads
const HEAD_DIM: usize = 256;
const FULL_ATTN_LAYERS: usize = 16;

#[test]
#[ignore = "needs a CUDA device"]
fn gqa_expansion_cost_grows_linearly_with_context() {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipped: no CUDA device ({e})");
            return;
        },
    };

    println!(
        "{:>6} | {:>13} | {:>12} | {:>12} | {:>7}",
        "ctx", "bytes/token", "expand ms", "grouped ms", "saved"
    );
    for &s in &[312usize, 1412, 2912, 4096] {
        let k = Tensor::zeros((1, KV_HEADS, s, HEAD_DIM), DType::BF16, &dev).unwrap();
        let v = Tensor::zeros((1, KV_HEADS, s, HEAD_DIM), DType::BF16, &dev).unwrap();
        // Decode: one query token, 24 heads.
        let q = Tensor::zeros((1, KV_HEADS * N_REP, 1, HEAD_DIM), DType::BF16, &dev).unwrap();

        let bench = |f: &dyn Fn()| -> f64 {
            // Warm the allocator so this times the steady state, not first touch.
            for _ in 0..3 {
                f();
            }
            dev.synchronize().unwrap();
            let iters = 10;
            let t0 = std::time::Instant::now();
            for _ in 0..iters * FULL_ATTN_LAYERS {
                f();
            }
            dev.synchronize().unwrap();
            t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
        };

        let expand_ms = bench(&|| expand_once(&k, &v));
        let grouped_ms = bench(&|| grouped_once(&q, &k, &v));

        // 3 materialized tensors, each [1, 24, s, 256] bf16, per layer.
        let bytes = 3.0 * (N_REP * KV_HEADS * s * HEAD_DIM * 2) as f64 * FULL_ATTN_LAYERS as f64;
        println!(
            "{s:>6} | {:>10.1} MB | {expand_ms:>9.2} ms | {grouped_ms:>9.2} ms | {:>6.1}x",
            bytes / 1e6,
            expand_ms / grouped_ms
        );
    }
}

/// The expansion-free equivalent: fold the `n_rep` query heads that share a KV
/// head into the matmul's row dimension, so K/V are read once at their stored
/// 4-head width instead of being broadcast to 24 and copied.
fn grouped_once(q: &Tensor, k: &Tensor, v: &Tensor) {
    let (b, kv, s, d) = k.dims4().unwrap();
    // [b, 24, 1, d] -> [b, 4, 6, d]: the 6 query heads of one KV group become
    // rows of a single small matmul against that group's K.
    let qg = q.reshape((b, kv, N_REP, d)).unwrap();
    let kt = k
        .transpose(D::Minus2, D::Minus1)
        .unwrap()
        .contiguous()
        .unwrap();
    let scores = qg.matmul(&kt).unwrap(); // [b, 4, 6, s]
    let w = candle_nn::ops::softmax_last_dim(&scores).unwrap();
    let _y = w.matmul(v).unwrap(); // [b, 4, 6, d]
    let _ = s;
}

fn expand_once(k: &Tensor, v: &Tensor) {
    let (b, kv, s, d) = k.dims4().unwrap();
    let k_rep = k
        .unsqueeze(2)
        .unwrap()
        .expand((b, kv, N_REP, s, d))
        .unwrap()
        .contiguous()
        .unwrap()
        .reshape((b, kv * N_REP, s, d))
        .unwrap();
    let _v_rep = v
        .unsqueeze(2)
        .unwrap()
        .expand((b, kv, N_REP, s, d))
        .unwrap()
        .contiguous()
        .unwrap()
        .reshape((b, kv * N_REP, s, d))
        .unwrap();
    let _k_t = k_rep
        .transpose(D::Minus2, D::Minus1)
        .unwrap()
        .contiguous()
        .unwrap();
}
