// SPDX-License-Identifier: MIT

//! Per-Layer Embedding (PLE): hashed token n-grams looked up in a huge
//! embedding table, gated by the residual streams, then smoothed by a dilated
//! depthwise causal convolution. Reference: `Qwen4ExpTextNGramEmbedding` and
//! `Qwen4ExpTextPLELayer` in transformers `modeling_qwen4_exp.py`.
//!
//! The table (~320M rows in the released checkpoint) never goes to the
//! device: rows are hashed and gathered on the host from a memory-mapped GGUF
//! shard ([`NgramTable::open_gguf`]), and only the gathered rows are uploaded.

use std::collections::HashMap;
use std::path::Path;

use candle_core::quantized::gguf_file::Value;
use candle_core::{D, Device, Module, Result, Tensor, bail};
use candle_nn::VarBuilder;

use super::config::TextConfig;
use super::hyper_connection::GroupedRmsNorm;
use crate::ops::linear::{LinearLayer, linear_layer};
use crate::quantized::extended_gguf::read_content;
use crate::quantized::gguf_file::Gguf;
use crate::quantized::gguf_metadata::GgufMetadata;
use crate::quantized::iquant::IQuantType;

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
/// Seed stride between PLE modules.
const PRIME_1: u64 = 10_007;

fn splitmix64(value: u64) -> u64 {
    let mut v = value.wrapping_add(SPLITMIX_GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(SPLITMIX_M1);
    v = (v ^ (v >> 27)).wrapping_mul(SPLITMIX_M2);
    v ^ (v >> 31)
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    (3..)
        .step_by(2)
        .take_while(|d| d * d <= n)
        .all(|d| !n.is_multiple_of(d))
}

/// Maps each token (with its predecessors) to one table row per hashed head.
///
/// For every order `n` in `2..=ngram_size` the window's tokens are mixed as
/// `t[p]*m[0] ^ t[p-1]*m[1] ^ .. ^ t[p-n+1]*m[n-1]`, and each of that order's
/// `heads_per_ngram` heads takes `mixed % vocab[h] + offset[h]`. An EOS among
/// the predecessors cuts the window: it and everything before read as EOS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramHash {
    multipliers: Vec<u64>,
    head_vocab_sizes: Vec<u64>,
    head_offsets: Vec<u64>,
    heads_per_ngram: usize,
    eos: u32,
}

impl NgramHash {
    /// Derive the constants the way HF does from the config (seeded
    /// splitmix64 multipliers, consecutive primes from
    /// `ngram_vocab_size_base` as head vocab sizes). `ple_index` counts PLE
    /// modules, not layers.
    ///
    /// # Errors
    ///
    /// Returns an error if the config has no PLE EOS id.
    pub fn from_config(cfg: &TextConfig, ple_index: u64) -> Result<Self> {
        let Some(eos) = cfg.ple_eos_token_id() else {
            bail!("PLE needs an eos_token_id")
        };
        let vocab = (cfg.vocab_size as u64).max(1);
        let half_bound = ((i64::MAX as u64 / vocab) / 2).max(1);
        let base_seed = cfg.seed.wrapping_add(PRIME_1.wrapping_mul(ple_index));
        let multipliers = (1..=cfg.ngram_size as u64)
            .map(|i| {
                let v = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(i));
                2 * (splitmix64(v) % half_bound) + 1
            })
            .collect();

        let heads = cfg.ngram_heads();
        // Head h of this module takes the (ple_index * heads + h + 1)-th prime
        // at or above the base.
        let mut primes = (cfg.ngram_vocab_size_base..).filter(|&n| is_prime(n));
        let skip = usize::try_from(ple_index).unwrap_or(usize::MAX) * heads;
        let head_vocab_sizes: Vec<u64> = primes.by_ref().skip(skip).take(heads).collect();
        let head_offsets = head_vocab_sizes
            .iter()
            .scan(0u64, |total, size| {
                let offset = *total;
                *total += size;
                Some(offset)
            })
            .collect();
        Ok(Self {
            multipliers,
            head_vocab_sizes,
            head_offsets,
            heads_per_ngram: cfg.heads_per_ngram,
            eos,
        })
    }

    /// Read the constants llama.cpp's converter stores in the GGUF
    /// (`qwen4exp.ple.*`).
    ///
    /// # Errors
    ///
    /// Returns an error if a key is missing or the arrays disagree in length.
    pub fn from_gguf(md: &HashMap<String, Value>) -> Result<Self> {
        let md = GgufMetadata(md);
        let key = |k: &str| format!("qwen4exp.ple.{k}");
        let multipliers = md.u64s(&key("layer_multipliers"))?;
        let head_vocab_sizes = md.u64s(&key("head_vocab_sizes"))?;
        let head_offsets = md.u64s(&key("head_offsets"))?;
        let heads_per_ngram = md.usize(&key("heads_per_ngram"))?;
        let ngram_size = md.usize(&key("ngram_size"))?;
        let heads = (ngram_size - 1) * heads_per_ngram;
        if multipliers.len() < ngram_size
            || head_vocab_sizes.len() < heads
            || head_offsets.len() < heads
        {
            bail!("qwen4exp.ple.* arrays are shorter than the {heads} heads they describe")
        }
        Ok(Self {
            multipliers: multipliers[..ngram_size].to_vec(),
            head_vocab_sizes: head_vocab_sizes[..heads].to_vec(),
            head_offsets: head_offsets[..heads].to_vec(),
            heads_per_ngram,
            eos: md.u32(&key("eos_token_id"))?,
        })
    }

    /// Number of hashed heads (rows gathered per token).
    #[must_use]
    pub fn heads(&self) -> usize {
        self.head_vocab_sizes.len()
    }

    /// Rows the table must at least have.
    #[must_use]
    pub fn min_rows(&self) -> u64 {
        self.head_offsets
            .iter()
            .zip(&self.head_vocab_sizes)
            .map(|(o, s)| o + s)
            .max()
            .unwrap_or(0)
    }

    /// Fresh per-sequence window: every predecessor reads as EOS.
    #[must_use]
    pub fn new_window(&self) -> NgramWindow {
        NgramWindow(vec![self.eos; self.multipliers.len() - 1])
    }

    /// Table rows for `tokens` (`[tokens.len() * heads]`, heads innermost),
    /// advancing `window` past them.
    ///
    /// # Errors
    ///
    /// Returns an error if a row index does not fit `u32`.
    pub fn rows(&self, window: &mut NgramWindow, tokens: &[u32]) -> Result<Vec<u32>> {
        let n_prev = self.multipliers.len() - 1;
        let mut history = std::mem::take(&mut window.0);
        history.extend_from_slice(tokens);
        let mut rows = Vec::with_capacity(tokens.len() * self.heads());
        let mut ctx = vec![0u64; self.multipliers.len()];
        for pos in n_prev..history.len() {
            ctx[0] = u64::from(history[pos]);
            let mut cut = false;
            for s in 1..=n_prev {
                let t = history[pos - s];
                cut = cut || t == self.eos;
                ctx[s] = u64::from(if cut { self.eos } else { t });
            }
            let mut mixed = ctx[0].wrapping_mul(self.multipliers[0]);
            for (order, heads) in self
                .head_vocab_sizes
                .chunks(self.heads_per_ngram)
                .zip(self.head_offsets.chunks(self.heads_per_ngram))
                .enumerate()
            {
                // Order `order + 2` extends the previous order's mix by one token.
                let j = order + 1;
                mixed ^= ctx[j].wrapping_mul(self.multipliers[j]);
                for (size, offset) in heads.0.iter().zip(heads.1) {
                    let row = mixed % size + offset;
                    rows.push(u32::try_from(row).map_err(|_| {
                        candle_core::Error::Msg(format!("PLE row {row} does not fit u32"))
                    })?);
                }
            }
        }
        window.0 = history.split_off(history.len() - n_prev);
        Ok(rows)
    }
}

/// The last `ngram_size - 1` tokens of a sequence, oldest first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramWindow(Vec<u32>);

/// Where the n-gram embedding rows live.
pub enum NgramTable {
    /// A dense `[rows, head_dim]` tensor (HF checkpoints; tests).
    Dense(Tensor),
    /// A memory-mapped GGUF tensor, decoded one row at a time on gather.
    Packed(PackedTable),
}

/// Where a packed table's bytes are kept (`CRANE_PLE_RAM`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    /// Memory-mapped: rows are paged in from disk on demand, and the page
    /// cache may drop them again under memory pressure.
    Mapped,
    /// Copied into process memory at load: lookups never touch the disk.
    Ram,
    /// [`Self::Ram`] when the host has the table's size plus
    /// [`RAM_HEADROOM`] available, else [`Self::Mapped`].
    Auto,
}

/// Host memory left free after copying a table into RAM under
/// [`Residency::Auto`].
pub const RAM_HEADROOM: u64 = 4 << 30;

impl Residency {
    /// Parse `CRANE_PLE_RAM`: unset or `auto` → [`Self::Auto`], `1`/`on`/
    /// `true` → [`Self::Ram`], `0`/`off`/`false` → [`Self::Mapped`].
    ///
    /// # Errors
    ///
    /// Returns an error for any other value.
    pub fn from_env() -> Result<Self> {
        let Ok(value) = std::env::var("CRANE_PLE_RAM") else {
            return Ok(Self::Auto);
        };
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "1" | "on" | "true" => Ok(Self::Ram),
            "0" | "off" | "false" => Ok(Self::Mapped),
            other => bail!("CRANE_PLE_RAM: expected auto, 1 or 0, got {other:?}"),
        }
    }
}

/// The bytes of a [`PackedTable`].
enum TableBytes {
    Mapped(memmap2::Mmap),
    Ram(Vec<u8>),
}

impl TableBytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mapped(map) => map,
            Self::Ram(bytes) => bytes,
        }
    }
}

/// Rows of a packed GGUF tensor, decoded one row at a time on gather.
pub struct PackedTable {
    bytes: TableBytes,
    /// Byte offset of row 0 within `bytes`.
    start: usize,
    rows: usize,
    head_dim: usize,
    ty: IQuantType,
}

impl NgramTable {
    /// Open `name` from the GGUF file at `path` (the released model keeps
    /// `per_layer_token_embd.weight`, `IQ4_NL`, alone in its second shard),
    /// memory-mapped or copied into RAM per `residency`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be mapped, the tensor is missing,
    /// its type is not one [`IQuantType`] decodes, or [`Residency::Ram`] was
    /// requested and the copy cannot be allocated.
    pub fn open_gguf(path: &Path, name: &str, residency: Residency) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: the mapping is read-only; the model file is not expected to
        // change while it is loaded, as with every other mmapped GGUF here.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let (content, extended) = read_content(&map)?;
        let Some(info) = extended.iquant.get(name) else {
            bail!(
                "{}: {name} is missing or not in a packed type Crane decodes",
                path.display()
            )
        };
        let [rows, head_dim] = info.shape[..] else {
            bail!("{name} should be 2-D, got {:?}", info.shape)
        };
        if !head_dim.is_multiple_of(info.ty.block_size()) {
            bail!(
                "{name} rows of {head_dim} are not whole {} blocks",
                info.ty.name()
            )
        }
        let start = usize::try_from(content.tensor_data_offset + info.offset)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let len = rows * (head_dim / info.ty.block_size()) * info.ty.block_bytes();
        if start + len > map.len() {
            bail!("{name} extends past the end of {}", path.display())
        }

        let to_ram = match residency {
            Residency::Mapped => false,
            Residency::Ram => true,
            Residency::Auto => crate::device::available_host_memory()
                .is_some_and(|avail| avail >= len as u64 + RAM_HEADROOM),
        };
        #[allow(clippy::cast_precision_loss)] // log output only
        let gib = |b: usize| b as f64 / f64::from(1u32 << 30);
        let (bytes, start) = if to_ram {
            let t = std::time::Instant::now();
            let mut owned = Vec::new();
            match owned.try_reserve_exact(len) {
                Ok(()) => {
                    #[cfg(unix)]
                    let _ = map.advise(memmap2::Advice::Sequential);
                    owned.extend_from_slice(&map[start..start + len]);
                    eprintln!(
                        "[qwen4_exp] {name}: {:.1} GiB copied to RAM in {:.1}s",
                        gib(len),
                        t.elapsed().as_secs_f64()
                    );
                    (TableBytes::Ram(owned), 0)
                },
                Err(e) if residency == Residency::Ram => {
                    bail!("cannot hold {name} ({:.1} GiB) in RAM: {e}", gib(len))
                },
                Err(_) => {
                    eprintln!(
                        "[qwen4_exp] {name}: RAM copy failed to allocate, memory-mapping instead"
                    );
                    (TableBytes::Mapped(map), start)
                },
            }
        } else {
            eprintln!(
                "[qwen4_exp] {name}: {:.1} GiB memory-mapped from {} \
                 (CRANE_PLE_RAM=1 copies it to RAM)",
                gib(len),
                path.display()
            );
            (TableBytes::Mapped(map), start)
        };
        Ok(Self::Packed(PackedTable {
            bytes,
            start,
            rows,
            head_dim,
            ty: info.ty,
        }))
    }

    /// Number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        match self {
            Self::Dense(t) => t.dim(0).unwrap_or(0),
            Self::Packed(p) => p.rows,
        }
    }

    /// Width of one row (one head's embedding).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        match self {
            Self::Dense(t) => t.dim(1).unwrap_or(0),
            Self::Packed(p) => p.head_dim,
        }
    }

    /// Rows `ids` as an `f32` `[ids.len(), head_dim]` tensor on `device`.
    ///
    /// # Errors
    ///
    /// Returns an error if an id is out of range.
    pub fn gather(&self, ids: &[u32], device: &Device) -> Result<Tensor> {
        if let Some(&bad) = ids.iter().find(|&&id| id as usize >= self.rows()) {
            bail!("PLE row {bad} out of range for {} rows", self.rows())
        }
        match self {
            Self::Dense(table) => {
                let ids = Tensor::from_slice(ids, ids.len(), table.device())?;
                table
                    .index_select(&ids, 0)?
                    .to_dtype(candle_core::DType::F32)?
                    .to_device(device)
            },
            Self::Packed(p) => {
                let row_bytes = p.row_bytes();
                let mut out = vec![0f32; ids.len() * p.head_dim];
                for (&id, dst) in ids.iter().zip(out.chunks_exact_mut(p.head_dim)) {
                    let at = p.start + id as usize * row_bytes;
                    p.ty.dequantize(&p.bytes.as_slice()[at..at + row_bytes], dst);
                }
                Tensor::from_vec(out, (ids.len(), p.head_dim), device)
            },
        }
    }
}

impl PackedTable {
    fn row_bytes(&self) -> usize {
        self.head_dim / self.ty.block_size() * self.ty.block_bytes()
    }
}

/// Per-sequence PLE state: the hash window and the conv history.
pub struct PleState {
    window: NgramWindow,
    /// Last `(kernel - 1) * dilation` conv inputs `[hist, hc * hidden]`;
    /// `None` before the first token (reads as zeros).
    conv: Option<Tensor>,
}

impl PleState {
    #[must_use]
    pub fn new(hash: &NgramHash) -> Self {
        Self {
            window: hash.new_window(),
            conv: None,
        }
    }
}

/// The PLE module of one decoder layer.
pub struct PleLayer {
    hash: NgramHash,
    table: NgramTable,
    key_proj: LinearLayer,
    value_proj: LinearLayer,
    norm_key: GroupedRmsNorm,
    norm_query: GroupedRmsNorm,
    norm_conv: GroupedRmsNorm,
    /// Depthwise kernel `[hc * hidden, kernel]`.
    conv_weight: Tensor,
    dilation: usize,
    hc_count: usize,
    hidden: usize,
}

impl PleLayer {
    /// Load from a HF checkpoint (`vb` scoped to `layers.N.ple`), with the
    /// n-gram table read densely from `ple_embedding.ngram_embedding`.
    ///
    /// # Errors
    ///
    /// Returns an error if a weight is missing or has the wrong shape.
    pub fn load(cfg: &TextConfig, ple_index: u64, vb: &VarBuilder) -> Result<Self> {
        let hash = NgramHash::from_config(cfg, ple_index)?;
        let table = vb
            .pp("ple_embedding.ngram_embedding")
            .get_unchecked("weight")?;
        let table = NgramTable::Dense(table);
        let (hidden, hc) = (cfg.hidden_size, cfg.hc_count);
        let wide = hc * hidden;
        let embed = cfg.ple_embed_dim();
        let conv_weight = vb
            .get((wide, 1, cfg.ple_conv_kernel_size), "conv1d.weight")?
            .squeeze(1)?;
        Self::new(
            hash,
            table,
            linear_layer(embed, wide, vb.pp("key_proj"), None)?,
            linear_layer(embed, hidden, vb.pp("value_proj"), None)?,
            [
                GroupedRmsNorm::load(hc, hidden, cfg.rms_norm_eps, &vb.pp("norm_key"))?,
                GroupedRmsNorm::load(hc, hidden, cfg.rms_norm_eps, &vb.pp("norm_query"))?,
                GroupedRmsNorm::load(hc, hidden, cfg.rms_norm_eps, &vb.pp("norm_conv"))?,
            ],
            conv_weight,
            cfg,
        )
    }

    /// Load layer `layer_idx`'s PLE module from GGUF (`blk.N.ple_*`, norms
    /// with `+1` folded in, hash constants from `qwen4exp.ple.*`), reading
    /// n-gram rows from `table` (see [`NgramTable::open_gguf`]).
    ///
    /// # Errors
    ///
    /// Returns an error if a tensor or metadata key is missing, or the table
    /// does not match the hash.
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        cfg: &TextConfig,
        gg: &mut Gguf<R>,
        layer_idx: usize,
        table: NgramTable,
    ) -> Result<Self> {
        let hash = NgramHash::from_gguf(gg.metadata())?;
        let name = |t: &str| format!("blk.{layer_idx}.ple_{t}.weight");
        let (hc, hidden, eps) = (cfg.hc_count, cfg.hidden_size, cfg.rms_norm_eps);
        let mut norm = |t: &str| -> Result<GroupedRmsNorm> {
            GroupedRmsNorm::from_folded(&gg.dequant_tensor(&name(t))?, hc, hidden, eps)
        };
        let norms = [norm("norm_key")?, norm("norm_query")?, norm("norm_conv")?];
        Self::new(
            hash,
            table,
            gg.linear_compact(&name("key"))?,
            gg.linear_compact(&name("value"))?,
            norms,
            gg.dequant_tensor(&name("conv1d"))?,
            cfg,
        )
    }

    /// Assemble from loaded parts; checks that the table matches the hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the table is too small or has the wrong width.
    pub fn new(
        hash: NgramHash,
        table: NgramTable,
        key_proj: LinearLayer,
        value_proj: LinearLayer,
        [norm_key, norm_query, norm_conv]: [GroupedRmsNorm; 3],
        conv_weight: Tensor,
        cfg: &TextConfig,
    ) -> Result<Self> {
        if (table.rows() as u64) < hash.min_rows() {
            bail!(
                "PLE table has {} rows, the hash needs {}",
                table.rows(),
                hash.min_rows()
            )
        }
        if table.head_dim() * hash.heads() != cfg.ple_embed_dim() {
            bail!(
                "PLE table rows of {} x {} heads do not make ple_embed_dim {}",
                table.head_dim(),
                hash.heads(),
                cfg.ple_embed_dim()
            )
        }
        Ok(Self {
            hash,
            table,
            key_proj,
            value_proj,
            norm_key,
            norm_query,
            norm_conv,
            conv_weight,
            // The conv is dilated by the n-gram size.
            dilation: cfg.ngram_size,
            hc_count: cfg.hc_count,
            hidden: cfg.hidden_size,
        })
    }

    #[must_use]
    pub fn hash(&self) -> &NgramHash {
        &self.hash
    }

    /// Gathered n-gram embeddings `[tokens.len(), ple_embed_dim]` for the
    /// next `tokens` of a sequence, advancing its hash window.
    ///
    /// # Errors
    ///
    /// Returns an error if hashing or the gather fails.
    pub fn embed(&self, state: &mut PleState, tokens: &[u32], device: &Device) -> Result<Tensor> {
        let rows = self.hash.rows(&mut state.window, tokens)?;
        self.table
            .gather(&rows, device)?
            .reshape((tokens.len(), self.hash.heads() * self.table.head_dim()))
    }

    /// PLE contribution for one sequence, to be added to `streams`
    /// (`[seq, hc * hidden]`). `embeddings` come from [`Self::embed`].
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes disagree or a tensor op fails.
    pub fn forward(
        &self,
        streams: &Tensor,
        embeddings: &Tensor,
        state: &mut PleState,
    ) -> Result<Tensor> {
        let (seq, wide) = streams.dims2()?;
        let dtype = streams.dtype();
        let embeddings = embeddings.to_dtype(dtype)?;
        let per_stream = (seq, self.hc_count, self.hidden);

        let key = self
            .norm_key
            .forward(&self.key_proj.forward(&embeddings)?)?
            .reshape(per_stream)?;
        let query = self.norm_query.forward(streams)?.reshape(per_stream)?;
        let value = self.value_proj.forward(&embeddings)?;

        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (self.hidden as f64).sqrt();
        let score = ((key * query)?.sum_keepdim(D::Minus1)? * scale)?;
        // Signed square root before the sigmoid.
        let score = (score.abs()?.clamp(1e-6, f64::INFINITY)?.sqrt()? * score.sign()?)?;
        let gated = candle_nn::ops::sigmoid(&score)?
            .broadcast_mul(&value.unsqueeze(1)?)?
            .reshape((seq, wide))?;

        let conv = self.dilated_conv(&self.norm_conv.forward(&gated)?, state)?;
        gated + candle_nn::ops::silu(&conv)?
    }

    /// Depthwise causal conv over time, dilated by `self.dilation`:
    /// `out[t] = sum_k w[:, k] * x[t - (K - 1 - k) * dilation]`, with the
    /// history of earlier calls in front so chunked prefill matches one shot.
    fn dilated_conv(&self, x: &Tensor, state: &mut PleState) -> Result<Tensor> {
        let (seq, wide) = x.dims2()?;
        let kernel = self.conv_weight.dim(1)?;
        let hist = (kernel - 1) * self.dilation;
        let prev = match state.conv.take() {
            Some(prev) => prev,
            None => Tensor::zeros((hist, wide), x.dtype(), x.device())?,
        };
        let padded = Tensor::cat(&[&prev, x], 0)?;
        let weight = self.conv_weight.to_dtype(x.dtype())?;
        let mut out: Option<Tensor> = None;
        for k in 0..kernel {
            let tap = padded
                .narrow(0, k * self.dilation, seq)?
                .broadcast_mul(&weight.narrow(1, k, 1)?.squeeze(1)?)?;
            out = Some(match out {
                Some(acc) => (acc + tap)?,
                None => tap,
            });
        }
        state.conv = Some(padded.narrow(0, seq, hist)?.contiguous()?);
        out.ok_or_else(|| candle_core::Error::Msg("PLE conv kernel is empty".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLASH_NEXT: &str = r#"{
        "vocab_size": 248320, "hidden_size": 2560, "num_hidden_layers": 4,
        "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
        "rms_norm_eps": 1e-06, "max_position_embeddings": 4096,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention",
            "linear_attention"],
        "rope_parameters": {"mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
            "rope_theta": 10000000},
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_num_key_heads": 16, "linear_num_value_heads": 48, "output_gate_type": "sigmoid",
        "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640, "hc_count": 4, "hc_lowrank": 320,
        "ple_layer_ids": [2], "ple_embed_dim": 2560, "ngram_size": 3, "heads_per_ngram": 8,
        "ngram_vocab_size_base": 20000000, "eos_token_id": 248044
    }"#;

    /// The constants llama.cpp's converter wrote into the released GGUF
    /// (`qwen4exp.ple.*` of `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002`),
    /// which the HF derivation from `seed` / `ngram_vocab_size_base` must
    /// reproduce exactly.
    #[test]
    fn config_derivation_matches_released_gguf_constants() {
        let cfg = TextConfig::from_json(FLASH_NEXT).unwrap();
        let hash = NgramHash::from_config(&cfg, 0).unwrap();
        assert_eq!(
            hash.multipliers,
            [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
        );
        assert_eq!(
            hash.head_vocab_sizes,
            [
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
                20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
                20_000_161, 20_000_171
            ]
        );
        assert_eq!(
            hash.head_offsets,
            [
                0,
                20_000_003,
                40_000_026,
                60_000_059,
                80_000_106,
                100_000_165,
                120_000_228,
                140_000_297,
                160_000_374,
                180_000_455,
                200_000_548,
                220_000_655,
                240_000_802,
                260_000_955,
                280_001_114,
                300_001_275
            ]
        );
        assert_eq!(hash.eos, 248_044);
        // The released table is padded up to a multiple of 128 rows.
        assert_eq!(hash.min_rows().div_ceil(128) * 128, 320_001_536);
    }

    fn small_hash() -> NgramHash {
        NgramHash {
            multipliers: vec![3, 5, 7],
            head_vocab_sizes: vec![101, 103, 107, 109],
            head_offsets: vec![0, 101, 204, 311],
            heads_per_ngram: 2,
            eos: 9,
        }
    }

    #[test]
    fn eos_cuts_the_window_but_not_its_own_context() {
        let hash = small_hash();
        let rows = hash.rows(&mut hash.new_window(), &[1, 2, 9, 4, 5]).unwrap();
        let row = |ctx: [u64; 3], order: usize, head: usize| {
            let mut mixed = (ctx[0] * 3) ^ (ctx[1] * 5);
            if order == 3 {
                mixed ^= ctx[2] * 7;
            }
            let h = (order - 2) * 2 + head;
            u32::try_from(mixed % hash.head_vocab_sizes[h] + hash.head_offsets[h]).unwrap()
        };
        let expect = |ctx: [u64; 3]| {
            [
                row(ctx, 2, 0),
                row(ctx, 2, 1),
                row(ctx, 3, 0),
                row(ctx, 3, 1),
            ]
        };
        // The sequence starts with an all-EOS history; EOS (9) at position 2
        // keeps its predecessors, and cuts everything before it for token 4.
        let want: Vec<u32> = [[1, 9, 9], [2, 1, 9], [9, 2, 1], [4, 9, 9], [5, 4, 9]]
            .into_iter()
            .flat_map(expect)
            .collect();
        assert_eq!(rows, want);
    }

    #[test]
    fn chunked_hashing_matches_one_shot() {
        let hash = small_hash();
        let tokens = [4, 8, 1, 9, 2, 2, 7, 3, 9, 9, 6];
        let whole = hash.rows(&mut hash.new_window(), &tokens).unwrap();
        for split in 0..tokens.len() {
            let mut window = hash.new_window();
            let mut rows = hash.rows(&mut window, &tokens[..split]).unwrap();
            for &t in &tokens[split..] {
                rows.extend(hash.rows(&mut window, &[t]).unwrap());
            }
            assert_eq!(rows, whole, "split at {split}");
        }
    }

    #[test]
    fn packed_gather_decodes_the_requested_rows() -> Result<()> {
        use half::f16;
        // Two IQ4_NL rows of 32 values each, written as a minimal GGUF.
        let mut data = Vec::new();
        for d in [0.5f32, 2.0] {
            data.extend(f16::from_f32(d).to_le_bytes());
            data.extend((0u8..16).map(|j| j | ((15 - j) << 4)));
        }
        let file = crate::quantized::test_util::write_gguf(&[(
            "per_layer_token_embd.weight",
            &[2, 32],
            20, // IQ4_NL
            data.clone(),
        )]);

        let path = std::env::temp_dir().join(format!("crane_ple_{}.gguf", std::process::id()));
        std::fs::write(&path, &file)?;
        let tables: Vec<Result<NgramTable>> = [Residency::Mapped, Residency::Ram]
            .into_iter()
            .map(|r| NgramTable::open_gguf(&path, "per_layer_token_embd.weight", r))
            .collect();
        std::fs::remove_file(&path)?;

        let mut want = [[0f32; 32]; 2];
        IQuantType::Iq4Nl.dequantize(&data[..18], &mut want[0]);
        IQuantType::Iq4Nl.dequantize(&data[18..], &mut want[1]);
        for table in tables {
            let table = table?;
            assert_eq!((table.rows(), table.head_dim()), (2, 32));
            let got = table.gather(&[1, 0, 1], &Device::Cpu)?.to_vec2::<f32>()?;
            assert_eq!(got, [want[1].to_vec(), want[0].to_vec(), want[1].to_vec()]);
            assert!(table.gather(&[2], &Device::Cpu).is_err());
        }
        Ok(())
    }
}
