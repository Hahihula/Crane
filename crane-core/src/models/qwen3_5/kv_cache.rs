//! K/V cache for the full-attention layers of Qwen 3.5 / Ornith.
//!
//! The hybrid model only needs this for the 1-in-4 full-attention blocks; the
//! linear-attention (GDN) blocks carry a constant-size recurrent state instead
//! (see [`crate::gdn::GdnLayerCache`]), so the context-growing part of the
//! cache lives in just these layers. At long context that K/V dominates memory,
//! which is why quantizing it lets a single agent hold much more context
//! locally (e.g. Ornith-9B's full 262K window on a 24 GB GPU).
//!
//! # Backends behind one contract
//!
//! [`KvCacheBackend`] is the seam: a backend stores the cache however it likes
//! but must, on [`append`](KvCacheBackend::append), take the new post-RoPE
//! `k`/`v` (`[B, num_kv_heads, S, head_dim]`) and return the *full* `k`/`v`
//! spanning all cached positions, **in the compute dtype**, ready for attention.
//! Attention logic never sees the storage representation.
//!
//! - [`FpKvCache`] — lossless f16/bf16 store (default).
//! - [`Int8KvCache`] — per-token symmetric int8 (~2x smaller), dequantized to
//!   the compute dtype on read.
//! - Future: int4-packed (~4x), and rotation-based codecs (rotorquant-style)
//!   for models whose usable window is much larger (≈1M tokens) where 2-3 bit
//!   needs the rotation to stay accurate. Each is just another
//!   `KvCacheBackend` + enum variant.

use candle_core::{DType, Result, Tensor, D};

/// Headroom (in positions) added when (re)allocating, to amortize growth.
const ROOM: usize = 256;

/// Contract every K/V cache backend honors. See the module docs.
pub trait KvCacheBackend {
    /// Append this step's `k`/`v` and return the full cached `(k, v)` in the
    /// compute dtype (`[B, num_kv_heads, seq_len + S, head_dim]`).
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)>;
    /// Drop all cached state (between unrelated requests).
    fn reset(&mut self);
    /// Number of cached positions.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Which cache representation to use. Selected once per model load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheKind {
    /// Lossless f16/bf16.
    Fp,
    /// Per-token symmetric int8 (~2x smaller).
    Int8,
}

impl KvCacheKind {
    /// Read from `CRANE_KV_QUANT` (`int8` → Int8, anything else → Fp).
    pub fn from_env() -> Self {
        match std::env::var("CRANE_KV_QUANT").as_deref() {
            Ok("int8") => Self::Int8,
            _ => Self::Fp,
        }
    }
}

/// Per-layer K/V cache. A thin enum dispatcher over the concrete backends so
/// `FullAttention` holds one type regardless of representation.
#[derive(Debug)]
pub enum KvCache {
    Fp(FpKvCache),
    Int8(Int8KvCache),
}

impl KvCache {
    pub fn new(kind: KvCacheKind) -> Self {
        match kind {
            KvCacheKind::Fp => Self::Fp(FpKvCache::new()),
            KvCacheKind::Int8 => Self::Int8(Int8KvCache::new()),
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Fp(c) => c.append(k, v),
            Self::Int8(c) => c.append(k, v),
        }
    }

    pub fn reset(&mut self) {
        match self {
            Self::Fp(c) => c.reset(),
            Self::Int8(c) => c.reset(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Fp(c) => c.len(),
            Self::Int8(c) => c.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new(KvCacheKind::Fp)
    }
}

// ── Growth helper (shared by all backends) ────────────────────────────────

/// Append `new` along the time dim (2) into a pre-allocated buffer, growing
/// with `ROOM` headroom on overflow, and return the filled `[.., filled+S, ..]`
/// view. Works for any rank-4 tensor (codes `[B,H,S,D]` or scales `[B,H,S,1]`).
fn grow_append(buf: &mut Option<Tensor>, new: &Tensor, filled: usize) -> Result<Tensor> {
    let new = new.contiguous()?;
    let add = new.dim(2)?;
    let total = filled + add;
    match buf.take() {
        None => {
            let (b, h, _s, d) = new.dims4()?;
            let store = Tensor::zeros((b, h, add + ROOM, d), new.dtype(), new.device())?;
            store.slice_set(&new, 2, 0)?;
            let view = store.narrow(2, 0, add)?;
            *buf = Some(store);
            Ok(view)
        }
        Some(store) => {
            if total <= store.dim(2)? {
                store.slice_set(&new, 2, filled)?;
                let view = store.narrow(2, 0, total)?;
                *buf = Some(store);
                Ok(view)
            } else {
                let cur = store.narrow(2, 0, filled)?;
                let full = Tensor::cat(&[&cur, &new], 2)?;
                let (b, h, t, d) = full.dims4()?;
                let grown = Tensor::zeros((b, h, t + ROOM, d), new.dtype(), new.device())?;
                grown.slice_set(&full, 2, 0)?;
                *buf = Some(grown);
                Ok(full)
            }
        }
    }
}

// ── Fp backend (lossless) ─────────────────────────────────────────────────

/// Lossless f16/bf16 cache: pre-allocated buffers written with `slice_set`
/// (O(new tokens), not `cat`), grown with fixed headroom on overflow.
#[derive(Debug, Default)]
pub struct FpKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    seq_len: usize,
}

impl FpKvCache {
    pub fn new() -> Self {
        Self::default()
    }
}

impl KvCacheBackend for FpKvCache {
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let add = k.dim(2)?;
        let k_full = grow_append(&mut self.k, k, self.seq_len)?;
        let v_full = grow_append(&mut self.v, v, self.seq_len)?;
        self.seq_len += add;
        Ok((k_full, v_full))
    }

    fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.seq_len = 0;
    }

    fn len(&self) -> usize {
        self.seq_len
    }
}

// ── Int8 backend (per-token symmetric) ────────────────────────────────────

/// Per-token symmetric int8 K/V cache. Each `[B,H,S,head_dim]` slice is stored
/// as u8 codes plus an f32 per-token scale (`amax / 127`); on read the filled
/// span is dequantized to the compute dtype, so attention is unchanged. ~2x
/// smaller than f16 (the f32 scale adds ~1.5% at head_dim=256).
///
/// Note: read dequantizes the whole filled cache each step — this trades some
/// decode bandwidth for the memory win that lets long context fit at all. A
/// fused dequantize-in-attention kernel is the perf follow-up.
#[derive(Debug, Default)]
pub struct Int8KvCache {
    k_codes: Option<Tensor>,
    k_scale: Option<Tensor>,
    v_codes: Option<Tensor>,
    v_scale: Option<Tensor>,
    seq_len: usize,
    /// Compute/return dtype (set on first append).
    dtype: Option<DType>,
}

impl Int8KvCache {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Quantize `[B,H,S,D]` per-token (symmetric): returns `(u8 codes, f32 scale)`.
/// `scale = amax/127 (+eps)` guarantees `|x/scale| <= 127`, so no clamp needed;
/// codes are stored as `q + 128` in `[1,255]`.
fn quantize_per_token(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let x = x.to_dtype(DType::F32)?;
    let amax = x.abs()?.max_keepdim(D::Minus1)?; // [B,H,S,1]
    let scale = amax.affine(1.0 / 127.0, 1e-8)?; // amax/127 + eps
    let q = x.broadcast_div(&scale)?.round()?;
    let codes = q.affine(1.0, 128.0)?.to_dtype(DType::U8)?; // q + 128
    Ok((codes, scale))
}

/// Inverse of [`quantize_per_token`] into `dtype`.
fn dequantize_per_token(codes: &Tensor, scale: &Tensor, dtype: DType) -> Result<Tensor> {
    codes
        .to_dtype(DType::F32)?
        .affine(1.0, -128.0)? // codes - 128
        .broadcast_mul(scale)?
        .to_dtype(dtype)
}

impl KvCacheBackend for Int8KvCache {
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let dtype = *self.dtype.get_or_insert(k.dtype());
        let add = k.dim(2)?;
        let filled = self.seq_len;

        let (kc, ks) = quantize_per_token(k)?;
        let (vc, vs) = quantize_per_token(v)?;

        let kc_full = grow_append(&mut self.k_codes, &kc, filled)?;
        let ks_full = grow_append(&mut self.k_scale, &ks, filled)?;
        let vc_full = grow_append(&mut self.v_codes, &vc, filled)?;
        let vs_full = grow_append(&mut self.v_scale, &vs, filled)?;
        self.seq_len += add;

        let k_full = dequantize_per_token(&kc_full, &ks_full, dtype)?;
        let v_full = dequantize_per_token(&vc_full, &vs_full, dtype)?;
        Ok((k_full, v_full))
    }

    fn reset(&mut self) {
        self.k_codes = None;
        self.k_scale = None;
        self.v_codes = None;
        self.v_scale = None;
        self.seq_len = 0;
        self.dtype = None;
    }

    fn len(&self) -> usize {
        self.seq_len
    }
}
