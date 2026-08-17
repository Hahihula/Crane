//! Variant + DSP parameters loaded from the upstream `config.json`.
//!
//! ```json
//! {
//!   "model_type": "muscriptor",
//!   "variant": "small" | "medium" | "large",
//!   "dim": 768,            // 1024 / 1536 for medium / large
//!   "num_heads": 12,       // 16 / 24
//!   "num_layers": 14,      // 24 / 48
//!   "card": 1393           // vocabulary size (always 1393 on the three published variants)
//! }
//! ```
//!
//! The DSP constants (`sample_rate`, `n_fft`, `hop_length`, `n_mels`,
//! `frame_rate`) aren't in the JSON — they're fixed by the upstream
//! training setup, not configurable per variant. They live here as
//! constants next to [`VariantConfig`] so consumers don't have to look
//! up a separate module to find them.

use candle_core::DType;
use serde::Deserialize;

/// Sample rate (Hz) the upstream MelSpectrogramConditioner expects.
pub const SAMPLE_RATE: usize = 16_000;
/// Segment duration (seconds) the model emits tokens over per forward
/// pass. Matches `_SEGMENT_DURATION` in `transcription_model.py`.
pub const SEGMENT_DURATION: f32 = 5.0;
/// FFT size for the mel-spec STFT.
pub const N_FFT: usize = 2048;
/// STFT hop length in samples. With `SAMPLE_RATE = 16000` and `HOP_LENGTH
/// = 160`, the mel frame rate is exactly `100` Hz.
pub const HOP_LENGTH: usize = 160;
/// Number of mel filterbanks. Audio's `[B, T, n_mels]` mel features are
/// linearly projected to `dim` by the conditioner.
pub const N_MELS: usize = 512;
/// Frame rate the `Shift` tokens are denominated in (100 Hz). Set here
/// as a sanity-cross-check for [`crate::models::muscriptor::mt3::MT3Tokenizer`].
pub const FRAME_RATE_HZ: u32 = 100;

/// Raw deserialized `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct RawConfig {
    #[serde(rename = "model_type")]
    pub _model_type: String,
    #[serde(default)]
    pub variant: Option<String>,
    pub dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub card: usize,
}

/// Loaded per-variant config. Holds the architecture parameters that
/// actually differ between small / medium / large.
#[derive(Debug, Clone)]
pub struct VariantConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub card: usize,
    /// Sinusoidal position `max_period` (10 000, matching the upstream).
    pub max_period: f64,
    /// FFN expansion factor — upstream hard-codes `hidden_scale = 4`,
    /// i.e. `dim_feedforward = 4 * dim`.
    pub hidden_scale: usize,
    /// Variant name (e.g. `"small"`); surfaced in logs / errors.
    pub variant: String,
}

impl VariantConfig {
    /// Build a [`VariantConfig`] from a parsed [`RawConfig`]. Validates
    /// that the dimensions are internally consistent (heads divide dim,
    /// card matches one of the published sizes).
    ///
    /// `card` is the LM head's output dim. The MT3 tokenizer itself is
    /// fixed at 1393 entries; medium/large allocate two extra
    /// "reserved / OOV" logits that are forced to -inf during
    /// decoding — see `LMModel::generate`.
    pub fn from_raw(raw: RawConfig) -> Result<Self, String> {
        if raw.dim % raw.num_heads != 0 {
            return Err(format!(
                "dim ({}) is not divisible by num_heads ({})",
                raw.dim, raw.num_heads
            ));
        }
        if !matches!(raw.card, 1393 | 1395) {
            return Err(format!(
                "unsupported card={} (expected 1393 for small or 1395 for medium/large)",
                raw.card
            ));
        }
        Ok(Self {
            dim: raw.dim,
            num_heads: raw.num_heads,
            num_layers: raw.num_layers,
            card: raw.card,
            max_period: 10_000.0,
            hidden_scale: 4,
            variant: raw.variant.unwrap_or_else(|| "unknown".to_string()),
        })
    }

    /// Parse directly from a JSON byte slice.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let raw: RawConfig =
            serde_json::from_slice(bytes).map_err(|e| format!("config.json: {e}"))?;
        Self::from_raw(raw)
    }

    /// Per-head dimension. `dim / num_heads` — panics only on malformed
    /// configs that passed `from_raw`.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        assert!(self.dim % self.num_heads == 0);
        self.dim / self.num_heads
    }

    /// FFN intermediate size.
    #[must_use]
    pub fn dim_feedforward(&self) -> usize {
        self.hidden_scale * self.dim
    }

    /// Conservative dtype for the *compute* path. Conditioning
    /// (mel-spec) always stays in fp32 (log-of-tiny mel bins underflows
    /// in fp16); the transformer itself can run in fp16/bf16 with
    /// autocast. Helper for callers that aren't sure what to pick.
    #[must_use]
    pub fn default_compute_dtype(&self) -> DType {
        DType::F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_small_config() {
        let cfg = VariantConfig::from_json_bytes(
            br#"{"model_type":"muscriptor","variant":"small","dim":768,"num_heads":12,"num_layers":14,"card":1393}"#,
        )
        .unwrap();
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.dim_feedforward(), 3072);
    }

    #[test]
    fn rejects_bad_card() {
        let cfg = VariantConfig::from_json_bytes(
            br#"{"dim":768,"num_heads":12,"num_layers":14,"card":9999"}"#,
        );
        assert!(cfg.is_err());
    }
}
