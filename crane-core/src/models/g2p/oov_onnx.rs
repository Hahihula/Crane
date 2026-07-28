// SPDX-License-Identifier: MIT

//! OOV (out-of-vocabulary) ONNX config parsing and model loading.
//!
//! Loads a Moonshine-TTS-format OOV model: an `onnx-config.json` sidecar
//! (char/phoneme vocabularies, sequence limits, special token IDs) plus a
//! `model.onnx` encoder-decoder transformer, used as a fallback tier when a
//! word misses the lexicon and hand-written rules can't confidently resolve
//! it. Also implements the autoregressive greedy-decode inference loop:
//! one `simple_eval()` call per output phoneme token, feeding the growing
//! decoder sequence back as input.
//!
//! **Known `candle-onnx` gap:** `candle-onnx` 0.11's `simple_eval()` does
//! not implement the ONNX `LayerNormalization` op (verified against its
//! `eval.rs` op-dispatch table, which has `BatchNormalization` but no
//! `LayerNormalization` arm). The shipped Moonshine-TTS English OOV model is
//! a standard pre/post-LayerNorm transformer, so [`Model::predict_phonemes`]
//! currently errors on it before completing a single encoder layer. The
//! decode loop and tensor plumbing here are otherwise verified correct (see
//! the `oov_onnx` and `english` unit tests), but end-to-end OOV inference
//! against a real model does not work until this is fixed — either upstream
//! in `candle-onnx`, or locally by rewriting `LayerNormalization` nodes into
//! the primitive ops `candle-onnx` does support (`ReduceMean`, `Sub`, `Pow`,
//! `Add`, `Sqrt`, `Div`, `Mul`) before calling `simple_eval`. Callers must
//! not assume a loaded OOV model will actually produce output — see
//! `EnglishG2p::try_oov_model`, which treats inference errors as a tier miss
//! rather than a hard failure for exactly this reason.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{Device, IndexOp, Tensor};
use serde::Deserialize;

/// Only schema version this parser understands.
const EXPECTED_SCHEMA_VERSION: u64 = 1;

const TOKEN_PAD: &str = "<pad>";
const TOKEN_UNK: &str = "<unk>";
const TOKEN_BOS: &str = "<bos>";
const TOKEN_EOS: &str = "<eos>";

/// Recognized `model_kind` values in `onnx-config.json`.
#[derive(Deserialize)]
enum ModelKind {
    /// OOV encoder-decoder fallback model.
    #[serde(rename = "oov")]
    Oov,
}

/// Raw shape of `onnx-config.json`, deserialized before validation.
#[derive(Deserialize)]
struct RawOovOnnxConfig {
    config_schema_version: u64,
    /// Only present so deserialization rejects unrecognized `model_kind`
    /// values; never read once parsing succeeds.
    #[allow(dead_code)]
    model_kind: ModelKind,
    char_vocab: HashMap<String, i64>,
    phoneme_vocab: HashMap<String, i64>,
    train_config: RawTrainConfig,
    oov_index: RawOovIndex,
}

/// Training hyperparameters relevant at inference time.
#[derive(Deserialize)]
struct RawTrainConfig {
    max_seq_len: usize,
}

/// OOV-specific config relevant at inference time.
#[derive(Deserialize)]
struct RawOovIndex {
    max_phoneme_len: usize,
}

/// Parsed and validated OOV ONNX config: vocabularies, sequence limits, and
/// special token IDs needed to run the encoder-decoder model.
#[derive(Debug)]
pub struct Config {
    /// Grapheme character to encoder input ID (excludes special tokens).
    pub char_to_id: HashMap<char, i64>,
    /// IPA phoneme token (may be multi-codepoint, e.g. `ˈeɪ`) to decoder
    /// token ID.
    pub phoneme_to_id: HashMap<String, i64>,
    /// Decoder token ID to IPA phoneme token, indexed by ID.
    pub id_to_phoneme: Vec<String>,
    /// Maximum encoder input sequence length.
    pub max_seq_len: usize,
    /// Maximum decoder output sequence length.
    pub max_phoneme_len: usize,
    /// Encoder padding token ID.
    pub char_pad_id: i64,
    /// Encoder unknown-character token ID.
    pub char_unk_id: i64,
    /// Decoder start-of-sequence token ID.
    pub phoneme_bos_id: i64,
    /// Decoder end-of-sequence token ID.
    pub phoneme_eos_id: i64,
    /// Decoder padding token ID.
    pub phoneme_pad_id: i64,
}

impl Config {
    /// Parses and validates an `onnx-config.json` document.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is malformed, the schema version or
    /// `model_kind` don't match what this parser supports, `max_seq_len` or
    /// `max_phoneme_len` is 0, a required special token is missing from
    /// either vocabulary, a `char_vocab` key (other than a special token) is
    /// not exactly one character, or `phoneme_vocab`'s IDs aren't a
    /// contiguous range starting at 0.
    ///
    /// `phoneme_bos_id`/`phoneme_eos_id` are legitimately similar names for
    /// distinct decoder special tokens, not a typo risk.
    #[allow(clippy::similar_names)]
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: RawOovOnnxConfig =
            serde_json::from_str(json).context("failed to parse OOV onnx-config.json")?;

        if raw.config_schema_version != EXPECTED_SCHEMA_VERSION {
            bail!(
                "unsupported OOV config schema version {} (expected {EXPECTED_SCHEMA_VERSION})",
                raw.config_schema_version
            );
        }

        if raw.train_config.max_seq_len == 0 {
            bail!("train_config.max_seq_len must be at least 1");
        }
        if raw.oov_index.max_phoneme_len == 0 {
            bail!("oov_index.max_phoneme_len must be at least 1");
        }

        let char_pad_id = required_token_id(&raw.char_vocab, TOKEN_PAD, "char_vocab")?;
        let char_unk_id = required_token_id(&raw.char_vocab, TOKEN_UNK, "char_vocab")?;
        let phoneme_bos_id = required_token_id(&raw.phoneme_vocab, TOKEN_BOS, "phoneme_vocab")?;
        let phoneme_eos_id = required_token_id(&raw.phoneme_vocab, TOKEN_EOS, "phoneme_vocab")?;
        let phoneme_pad_id = required_token_id(&raw.phoneme_vocab, TOKEN_PAD, "phoneme_vocab")?;

        let char_to_id = build_char_to_id(&raw.char_vocab)?;
        let id_to_phoneme = build_id_to_phoneme(&raw.phoneme_vocab)?;

        Ok(Self {
            char_to_id,
            phoneme_to_id: raw.phoneme_vocab,
            id_to_phoneme,
            max_seq_len: raw.train_config.max_seq_len,
            max_phoneme_len: raw.oov_index.max_phoneme_len,
            char_pad_id,
            char_unk_id,
            phoneme_bos_id,
            phoneme_eos_id,
            phoneme_pad_id,
        })
    }

    /// Tokenizes `word` into encoder input IDs, one per character.
    ///
    /// Characters not present in `char_to_id` map to `char_unk_id`. The
    /// result is truncated to `max_seq_len` if `word` has more characters
    /// than the encoder can accept.
    #[must_use]
    pub fn encode_word(&self, word: &str) -> Vec<i64> {
        let mut ids: Vec<i64> = word
            .chars()
            .map(|c| self.char_to_id.get(&c).copied().unwrap_or(self.char_unk_id))
            .collect();
        ids.truncate(self.max_seq_len);
        ids
    }
}

/// Looks up a required special token's ID in a vocabulary, erroring with the
/// vocabulary's name if the token is missing.
fn required_token_id(vocab: &HashMap<String, i64>, token: &str, vocab_name: &str) -> Result<i64> {
    vocab
        .get(token)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("{vocab_name} is missing required token {token:?}"))
}

/// Returns `true` for special tokens like `<pad>`, which aren't grapheme
/// characters and are excluded from [`Config::char_to_id`].
fn is_special_token(key: &str) -> bool {
    key.starts_with('<') && key.ends_with('>')
}

/// Builds the grapheme-character-to-ID map from `char_vocab`, skipping
/// special tokens and erroring if any remaining key isn't a single character.
fn build_char_to_id(char_vocab: &HashMap<String, i64>) -> Result<HashMap<char, i64>> {
    let mut char_to_id = HashMap::with_capacity(char_vocab.len());
    for (key, &id) in char_vocab {
        if is_special_token(key) {
            continue;
        }
        let mut chars = key.chars();
        let (Some(c), None) = (chars.next(), chars.next()) else {
            bail!("char_vocab key {key:?} is not a single character");
        };
        char_to_id.insert(c, id);
    }
    Ok(char_to_id)
}

/// Inverts `phoneme_vocab` into a dense `Vec<String>` indexed by token ID.
fn build_id_to_phoneme(phoneme_vocab: &HashMap<String, i64>) -> Result<Vec<String>> {
    let Some(&max_id) = phoneme_vocab.values().max() else {
        bail!("phoneme_vocab is empty");
    };
    let max_id = usize::try_from(max_id).context("phoneme_vocab contains a negative token ID")?;
    if max_id + 1 != phoneme_vocab.len() {
        bail!(
            "phoneme_vocab IDs must be a contiguous range from 0 to {max_id}, but {} entries were found",
            phoneme_vocab.len()
        );
    }
    let mut id_to_phoneme = vec![String::new(); max_id + 1];
    for (token, &id) in phoneme_vocab {
        let idx = usize::try_from(id)
            .with_context(|| format!("phoneme_vocab entry {token:?} has a negative ID"))?;
        id_to_phoneme[idx].clone_from(token);
    }
    Ok(id_to_phoneme)
}

/// Returns the index of the largest value in `logits`, for greedy decoding.
///
/// # Errors
///
/// Returns an error if `logits` is empty or every element is non-finite
/// (e.g. `NaN`), since neither case has a well-defined largest element —
/// silently falling back to index 0 would risk misinterpreting a malformed
/// model output as a real predicted token.
fn greedy_argmax(logits: &[f32]) -> Result<i64> {
    let mut best: Option<(usize, f32)> = None;
    for (idx, &score) in logits.iter().enumerate() {
        if score > best.map_or(f32::NEG_INFINITY, |(_, best_score)| best_score) {
            best = Some((idx, score));
        }
    }
    let (idx, _) = best.ok_or_else(|| anyhow::anyhow!("logits are empty or entirely non-finite"))?;
    // `idx` is bounded by the phoneme vocab size (131 for the shipped
    // English model, always far under i64::MAX), so this cast never
    // truncates.
    #[allow(clippy::cast_possible_wrap)]
    let idx = idx as i64;
    Ok(idx)
}

/// Loaded OOV encoder-decoder model: parsed config plus the ONNX graph.
///
/// Autoregressive decoding is not implemented here — this type only covers
/// loading the config and the ONNX model file.
pub struct Model {
    /// Parsed and validated `onnx-config.json`.
    pub config: Config,
    /// Loaded ONNX graph for the encoder-decoder model.
    pub model: crate::onnx::proto::ModelProto,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("max_seq_len", &self.config.max_seq_len)
            .field("max_phoneme_len", &self.config.max_phoneme_len)
            .field("char_vocab_len", &self.config.char_to_id.len())
            .field("phoneme_vocab_len", &self.config.id_to_phoneme.len())
            .finish_non_exhaustive()
    }
}

impl Model {
    /// Loads an OOV model from a directory containing `onnx-config.json` and
    /// `model.onnx`.
    ///
    /// # Errors
    ///
    /// Returns an error if either file is missing or malformed.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("onnx-config.json");
        let config_json = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let config = Config::from_json(&config_json)?;

        let onnx_path = model_dir.join("model.onnx");
        let model = crate::onnx::read_file(&onnx_path)
            .with_context(|| format!("failed to read {}", onnx_path.display()))?;

        Ok(Self { config, model })
    }

    /// Runs autoregressive greedy decoding for a single word, returning the
    /// predicted IPA string.
    ///
    /// Returns an empty string for an empty `word` or if the model emits
    /// end-of-sequence immediately.
    ///
    /// The decoder input/mask scratch buffers (`dec_ids`/`dec_mask`) are
    /// allocated once and mutated in place as each new token is produced,
    /// rather than rebuilt from scratch every step. This loop still performs
    /// per-step allocations that buffer reuse alone can't avoid — a fresh
    /// `Tensor` copy of each buffer, a new input `HashMap`, and a new
    /// `Vec<f32>` for the logits row — since `crate::onnx::simple_eval`
    /// takes owned/borrowed tensors and a fresh `HashMap` each call.
    ///
    /// This also does not use a KV-cache: every step re-sends the full
    /// `[1, max_phoneme_len]` decoder sequence rather than just the newest
    /// token (contrast with the incremental `past_key_values` decoding in
    /// `moonshine_asr::model::generate`), so total decode cost is
    /// `O(max_phoneme_len^2)`, not linear. This follows from the shipped
    /// ONNX model's exported graph, which has no cache inputs, not from a
    /// choice made here.
    ///
    /// # Errors
    ///
    /// Returns an error if ONNX inference fails, the model's `logits`
    /// output is missing, has an unexpected shape, or is empty/entirely
    /// non-finite, or a predicted token ID falls outside the phoneme
    /// vocabulary. In particular, this errors on the shipped Moonshine-TTS
    /// English model today — see the `candle-onnx` `LayerNormalization` gap
    /// documented at the top of this module.
    pub fn predict_phonemes(&self, word: &str) -> Result<String> {
        if word.is_empty() {
            return Ok(String::new());
        }

        let device = Device::Cpu;
        let max_seq_len = self.config.max_seq_len;
        let max_phoneme_len = self.config.max_phoneme_len;

        let encoded = self.config.encode_word(word);
        let mut enc_ids = vec![self.config.char_pad_id; max_seq_len];
        let mut enc_mask = vec![0i64; max_seq_len];
        for (i, &id) in encoded.iter().enumerate() {
            enc_ids[i] = id;
            enc_mask[i] = 1;
        }
        let enc_ids_tensor = Tensor::from_slice(&enc_ids, &[1, max_seq_len], &device)?;
        let enc_mask_tensor = Tensor::from_slice(&enc_mask, &[1, max_seq_len], &device)?;

        // Preallocated once, mutated in place per decode step instead of
        // being rebuilt — see the per-step allocation caveats on
        // `predict_phonemes` above.
        let mut dec_ids = vec![self.config.phoneme_pad_id; max_phoneme_len];
        let mut dec_mask = vec![0i64; max_phoneme_len];
        dec_ids[0] = self.config.phoneme_bos_id;
        dec_mask[0] = 1;
        let mut cursor = 1usize;

        while cursor < max_phoneme_len {
            let dec_ids_tensor = Tensor::from_slice(&dec_ids, &[1, max_phoneme_len], &device)?;
            let dec_mask_tensor = Tensor::from_slice(&dec_mask, &[1, max_phoneme_len], &device)?;

            // Encoder tensors are shared across every decode step; `clone()`
            // bumps the tensor's internal Arc, it does not copy the
            // underlying data.
            let inputs = HashMap::from([
                ("encoder_input_ids".to_string(), enc_ids_tensor.clone()),
                ("encoder_attention_mask".to_string(), enc_mask_tensor.clone()),
                ("decoder_input_ids".to_string(), dec_ids_tensor),
                ("decoder_attention_mask".to_string(), dec_mask_tensor),
            ]);

            let outputs = crate::onnx::simple_eval(&self.model, inputs)?;
            let logits = outputs
                .get("logits")
                .ok_or_else(|| anyhow::anyhow!("OOV model returned no logits output"))?;

            let last_step = logits.i((0, cursor - 1, ..))?.to_vec1::<f32>()?;
            let predicted_id = greedy_argmax(&last_step)?;

            if predicted_id == self.config.phoneme_eos_id
                || predicted_id == self.config.phoneme_pad_id
            {
                break;
            }

            dec_ids[cursor] = predicted_id;
            dec_mask[cursor] = 1;
            cursor += 1;
        }

        let mut ipa = String::new();
        for &token_id in &dec_ids[1..cursor] {
            let idx = usize::try_from(token_id)
                .with_context(|| format!("negative phoneme token ID {token_id}"))?;
            let phoneme = self
                .config
                .id_to_phoneme
                .get(idx)
                .with_context(|| format!("phoneme token ID {token_id} out of range"))?;
            ipa.push_str(phoneme);
        }
        Ok(ipa)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config_json() -> String {
        r#"{
            "config_schema_version": 1,
            "model_kind": "oov",
            "char_vocab": {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3},
            "phoneme_vocab": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "æ": 4},
            "train_config": {"max_seq_len": 64},
            "oov_index": {"max_phoneme_len": 64}
        }"#
        .to_string()
    }

    #[test]
    fn parses_valid_config() {
        let config = Config::from_json(&valid_config_json()).unwrap();
        assert_eq!(config.char_to_id.len(), 2);
        assert_eq!(config.char_to_id.get(&'a'), Some(&2));
        assert_eq!(config.char_to_id.get(&'b'), Some(&3));
        assert_eq!(config.char_pad_id, 0);
        assert_eq!(config.char_unk_id, 1);
        assert_eq!(config.phoneme_bos_id, 2);
        assert_eq!(config.phoneme_eos_id, 3);
        assert_eq!(config.phoneme_pad_id, 0);
        assert_eq!(config.max_seq_len, 64);
        assert_eq!(config.max_phoneme_len, 64);
    }

    #[test]
    fn id_to_phoneme_round_trips_with_phoneme_to_id() {
        let config = Config::from_json(&valid_config_json()).unwrap();
        for (token, &id) in &config.phoneme_to_id {
            let idx = usize::try_from(id).unwrap();
            assert_eq!(&config.id_to_phoneme[idx], token);
        }
    }

    #[test]
    fn rejects_wrong_schema_version() {
        let json = valid_config_json()
            .replace("\"config_schema_version\": 1", "\"config_schema_version\": 2");
        let err = Config::from_json(&json).unwrap_err();
        assert!(err.to_string().contains("schema version"));
    }

    #[test]
    fn rejects_wrong_model_kind() {
        let json = valid_config_json().replace("\"model_kind\": \"oov\"", "\"model_kind\": \"other\"");
        let err = Config::from_json(&json).unwrap_err();
        assert!(format!("{err:?}").contains("unknown variant"));
    }

    #[test]
    fn rejects_missing_special_token() {
        let json = valid_config_json().replace("\"<bos>\": 2, ", "");
        let err = Config::from_json(&json).unwrap_err();
        assert!(err.to_string().contains("<bos>"));
    }

    #[test]
    fn rejects_multi_character_char_vocab_key() {
        let json = valid_config_json().replace("\"a\": 2", "\"ab\": 2");
        let err = Config::from_json(&json).unwrap_err();
        assert!(err.to_string().contains("not a single character"));
    }

    #[test]
    fn rejects_non_contiguous_phoneme_vocab_ids() {
        let json = valid_config_json().replace("\"æ\": 4", "\"æ\": 5");
        let err = Config::from_json(&json).unwrap_err();
        assert!(err.to_string().contains("contiguous range"));
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(Config::from_json("not json").is_err());
    }

    #[test]
    fn rejects_zero_max_seq_len() {
        let json = valid_config_json().replace("\"max_seq_len\": 64", "\"max_seq_len\": 0");
        let err = Config::from_json(&json).unwrap_err();
        assert!(err.to_string().contains("max_seq_len"));
    }

    #[test]
    fn rejects_zero_max_phoneme_len() {
        let json =
            valid_config_json().replace("\"max_phoneme_len\": 64", "\"max_phoneme_len\": 0");
        let err = Config::from_json(&json).unwrap_err();
        assert!(err.to_string().contains("max_phoneme_len"));
    }

    #[test]
    fn load_reports_missing_config_file() {
        let missing_dir = std::env::temp_dir().join("crane_oov_onnx_missing_dir_for_test");
        let err = Model::load(&missing_dir).unwrap_err();
        assert!(err.to_string().contains("onnx-config.json"));
    }

    #[test]
    fn encode_word_maps_known_chars() {
        let config = Config::from_json(&valid_config_json()).unwrap();
        assert_eq!(config.encode_word("ab"), vec![2, 3]);
    }

    #[test]
    fn encode_word_maps_unknown_char_to_unk() {
        let config = Config::from_json(&valid_config_json()).unwrap();
        assert_eq!(config.encode_word("az"), vec![2, config.char_unk_id]);
    }

    #[test]
    fn encode_word_truncates_to_max_seq_len() {
        let json = valid_config_json().replace("\"max_seq_len\": 64", "\"max_seq_len\": 3");
        let config = Config::from_json(&json).unwrap();
        assert_eq!(config.encode_word("aaaaa").len(), 3);
    }

    #[test]
    fn encode_word_empty_returns_empty() {
        let config = Config::from_json(&valid_config_json()).unwrap();
        assert!(config.encode_word("").is_empty());
    }

    #[test]
    fn model_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Model>();
    }

    #[test]
    fn greedy_argmax_picks_largest_score() {
        assert_eq!(greedy_argmax(&[0.1, 0.9, 0.4]).unwrap(), 1);
    }

    #[test]
    fn greedy_argmax_rejects_empty_logits() {
        assert!(greedy_argmax(&[]).is_err());
    }

    #[test]
    fn greedy_argmax_rejects_all_nan_logits() {
        assert!(greedy_argmax(&[f32::NAN, f32::NAN]).is_err());
    }
}
