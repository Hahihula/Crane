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
//! `LayerNormalization`, `Mod`, and `Trilu` — all needed by the shipped
//! Moonshine-TTS English OOV model — are natively supported by
//! [`crate::onnx`]'s vendored evaluator, so no load-time graph rewrite is
//! needed here. Callers should still not assume ONNX inference can never
//! fail for other reasons — see `EnglishG2p::text_to_ipa`, which treats
//! inference errors as a tier miss rather than a hard failure.

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
    /// Superseded on the production path by [`Self::predict_phonemes_batch`],
    /// which `EnglishG2p::text_to_ipa` calls instead so multiple OOV words
    /// share decode steps; this method is kept as its sequential-decode
    /// reference implementation and as the correctness oracle in
    /// batch-vs-sequential regression tests.
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

    /// Runs autoregressive greedy decoding for multiple words in one
    /// synchronized batch: a single `simple_eval()` call per decode step is
    /// shared across every word in `words`, instead of a full decode loop
    /// per word (see [`Self::predict_phonemes`]). For `N` words where the
    /// longest needs `M` decode steps, this is `M` total `simple_eval()`
    /// calls instead of up to `N × M` — the real latency win for a request
    /// containing several OOV words. Words finish decoding (hit
    /// `<eos>`/`<pad>`) at different steps; a finished word's decoder row
    /// stops advancing while still-active rows continue, so the loop runs
    /// until every word is done or `max_phoneme_len` is reached.
    ///
    /// Returns one entry per input word, in the same order: `None` for an
    /// empty word, one that decodes to an empty result, or one whose row hit
    /// a decode error (an out-of-vocabulary predicted token ID) — matching
    /// [`Self::predict_phonemes`]'s per-word contract (see
    /// `EnglishG2p::text_to_ipa`, which treats both identically). Only a
    /// failure that aborts the whole batch (malformed tensors, a missing
    /// `logits` output) degrades every word to `None`; a single row's decode
    /// error does not affect its sibling rows — one bad word must not cost
    /// the rest of the batch a result it already computed correctly.
    #[must_use]
    pub fn predict_phonemes_batch(&self, words: &[&str]) -> Vec<Option<String>> {
        if words.is_empty() {
            return Vec::new();
        }
        self.predict_phonemes_batch_inner(words)
            .unwrap_or_else(|_| vec![None; words.len()])
    }

    /// Implements [`Self::predict_phonemes_batch`]; split out so batch-setup
    /// failures (tensor construction, `simple_eval`, a missing `logits`
    /// output) can be caught with `?` and mapped to an all-`None` result by
    /// the caller, while per-row decode and reconstruction failures are
    /// handled locally within this function and never propagate past a
    /// single row.
    fn predict_phonemes_batch_inner(&self, words: &[&str]) -> Result<Vec<Option<String>>> {
        let mut results = vec![None; words.len()];
        let active: Vec<usize> = words
            .iter()
            .enumerate()
            .filter(|(_, word)| !word.is_empty())
            .map(|(i, _)| i)
            .collect();
        if active.is_empty() {
            return Ok(results);
        }

        let device = Device::Cpu;
        let max_seq_len = self.config.max_seq_len;
        let max_phoneme_len = self.config.max_phoneme_len;
        let batch_size = active.len();

        let mut enc_ids = vec![self.config.char_pad_id; batch_size * max_seq_len];
        let mut enc_mask = vec![0i64; batch_size * max_seq_len];
        for (b, &word_idx) in active.iter().enumerate() {
            let encoded = self.config.encode_word(words[word_idx]);
            let row = b * max_seq_len;
            for (i, &id) in encoded.iter().enumerate() {
                enc_ids[row + i] = id;
                enc_mask[row + i] = 1;
            }
        }
        let enc_ids_tensor = Tensor::from_slice(&enc_ids, &[batch_size, max_seq_len], &device)?;
        let enc_mask_tensor = Tensor::from_slice(&enc_mask, &[batch_size, max_seq_len], &device)?;

        // Preallocated once, mutated in place per decode step — same
        // rationale as the scratch buffers in `predict_phonemes`.
        let mut dec_ids = vec![self.config.phoneme_pad_id; batch_size * max_phoneme_len];
        let mut dec_mask = vec![0i64; batch_size * max_phoneme_len];
        for b in 0..batch_size {
            dec_ids[b * max_phoneme_len] = self.config.phoneme_bos_id;
            dec_mask[b * max_phoneme_len] = 1;
        }
        // Number of decoder positions filled so far per row (mirrors
        // `predict_phonemes`'s single `cursor`, one per batch row).
        let mut filled = vec![1usize; batch_size];
        let mut finished = vec![false; batch_size];

        let mut step = 1usize;
        while step < max_phoneme_len && finished.contains(&false) {
            let dec_ids_tensor =
                Tensor::from_slice(&dec_ids, &[batch_size, max_phoneme_len], &device)?;
            let dec_mask_tensor =
                Tensor::from_slice(&dec_mask, &[batch_size, max_phoneme_len], &device)?;

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

            // One slice + one bulk copy for the whole batch instead of one
            // per active row — `step_logits[b]` is row `b`'s slice over the
            // vocab dimension at this step.
            let step_logits = logits.narrow(1, step - 1, 1)?.squeeze(1)?.to_vec2::<f32>()?;

            for (b, done) in finished.iter_mut().enumerate() {
                if *done {
                    continue;
                }
                // A single row's logits being empty/non-finite must not
                // abort the rest of the batch — treat it as that word's
                // decode finishing (early) rather than propagating.
                let Ok(predicted_id) = greedy_argmax(&step_logits[b]) else {
                    *done = true;
                    continue;
                };

                if predicted_id == self.config.phoneme_eos_id
                    || predicted_id == self.config.phoneme_pad_id
                {
                    *done = true;
                    continue;
                }

                dec_ids[b * max_phoneme_len + step] = predicted_id;
                dec_mask[b * max_phoneme_len + step] = 1;
                filled[b] = step + 1;
            }
            step += 1;
        }

        for (b, &word_idx) in active.iter().enumerate() {
            let row = b * max_phoneme_len;
            // A bad token ID in this row alone must not discard the other
            // rows' already-decoded results, so failures here resolve to
            // `None` for this word rather than propagating with `?`.
            let ipa: Option<String> = (|| {
                let mut ipa = String::new();
                for &token_id in &dec_ids[row + 1..row + filled[b]] {
                    let idx = usize::try_from(token_id).ok()?;
                    ipa.push_str(self.config.id_to_phoneme.get(idx)?);
                }
                (!ipa.is_empty()).then_some(ipa)
            })();
            results[word_idx] = ipa;
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx::proto::attribute_proto::AttributeType;
    use crate::onnx::proto::tensor_proto::DataType;
    use crate::onnx::proto::{AttributeProto, GraphProto, NodeProto, TensorProto, ValueInfoProto};

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

    /// A `Model` wrapping a valid config but an empty ONNX graph, so any
    /// `simple_eval()` call errors (there's no `logits` output to find).
    /// Used to verify batch-wide failures degrade to `None` per word
    /// instead of panicking or propagating.
    fn model_with_empty_graph() -> Model {
        Model {
            config: Config::from_json(&valid_config_json()).unwrap(),
            model: crate::onnx::proto::ModelProto::default(),
        }
    }

    #[test]
    fn predict_phonemes_batch_empty_input_returns_empty_vec() {
        let model = model_with_empty_graph();
        assert!(model.predict_phonemes_batch(&[]).is_empty());
    }

    #[test]
    fn predict_phonemes_batch_failing_model_returns_none_for_each_word() {
        let model = model_with_empty_graph();
        let results = model.predict_phonemes_batch(&["ab", "ba"]);
        assert_eq!(results, vec![None, None]);
    }

    #[test]
    fn predict_phonemes_batch_all_empty_words_returns_none_without_inference() {
        // Empty words are filtered out before any tensor is built, so this
        // must succeed even though the graph can't run inference.
        let model = model_with_empty_graph();
        let results = model.predict_phonemes_batch(&["", ""]);
        assert_eq!(results, vec![None, None]);
    }

    fn float_const_node(output: &str, dims: Vec<i64>, data: &[f32]) -> NodeProto {
        NodeProto {
            op_type: "Constant".to_string(),
            output: vec![output.to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: AttributeType::Tensor as i32,
                t: Some(TensorProto {
                    dims,
                    data_type: DataType::Float as i32,
                    raw_data: data.iter().flat_map(|f| f.to_le_bytes()).collect(),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn small_phoneme_len_config_json() -> String {
        r#"{
            "config_schema_version": 1,
            "model_kind": "oov",
            "char_vocab": {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3},
            "phoneme_vocab": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "æ": 4},
            "train_config": {"max_seq_len": 8},
            "oov_index": {"max_phoneme_len": 4}
        }"#
        .to_string()
    }

    /// A `Model` whose graph is a single `Constant` node emitting a fixed
    /// `[2, 4, 5]` logits tensor (batch, `max_phoneme_len`, vocab) regardless
    /// of its inputs, so decoding is fully deterministic without needing a
    /// real transformer. Vocab IDs follow [`small_phoneme_len_config_json`]:
    /// `<pad>`=0, `<unk>`=1, `<bos>`=2, `<eos>`=3, `æ`=4. Row 0 predicts `æ`
    /// then `<eos>` (finishes after 1 phoneme); row 1 predicts `æ`, `æ`,
    /// then `<eos>` (finishes after 2 phonemes) — used to prove a row that
    /// finishes early doesn't affect a still-active row's decoding.
    fn interleaved_finish_model() -> Model {
        const PHONEME_LEN: usize = 4;
        const VOCAB: usize = 5;
        let idx = |row: usize, pos: usize, id: usize| (row * PHONEME_LEN + pos) * VOCAB + id;

        let mut logits = vec![0f32; 2 * PHONEME_LEN * VOCAB];
        logits[idx(0, 0, 4)] = 1.0; // row 0, step 0 -> æ
        logits[idx(0, 1, 3)] = 1.0; // row 0, step 1 -> <eos>
        logits[idx(1, 0, 4)] = 1.0; // row 1, step 0 -> æ
        logits[idx(1, 1, 4)] = 1.0; // row 1, step 1 -> æ
        logits[idx(1, 2, 3)] = 1.0; // row 1, step 2 -> <eos>

        let mut graph = GraphProto {
            node: vec![float_const_node(
                "logits",
                vec![2, PHONEME_LEN as i64, VOCAB as i64],
                &logits,
            )],
            output: vec![ValueInfoProto {
                name: "logits".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let model = crate::onnx::proto::ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        Model {
            config: Config::from_json(&small_phoneme_len_config_json()).unwrap(),
            model,
        }
    }

    #[test]
    fn predict_phonemes_batch_interleaved_finish() {
        let model = interleaved_finish_model();
        let results = model.predict_phonemes_batch(&["a", "ab"]);
        assert_eq!(results, vec![Some("æ".to_string()), Some("ææ".to_string())]);
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

    /// Real-model regression test: verifies `predict_phonemes_batch`
    /// produces the same IPA as running `predict_phonemes` on each word
    /// individually, i.e. that batching decode steps together doesn't
    /// change the result. Needs a real OOV model on disk, so `#[ignore]`d
    /// by default. Run with:
    ///
    /// ```sh
    /// CRANE_G2P_EN_US_DIR=/path/to/en_us \
    ///   cargo test -p crane-core --features onnx -- \
    ///   predict_phonemes_batch_matches_sequential --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a local G2P model directory (CRANE_G2P_EN_US_DIR)"]
    fn predict_phonemes_batch_matches_sequential() {
        let model_dir: std::path::PathBuf = std::env::var("CRANE_G2P_EN_US_DIR")
            .expect("set CRANE_G2P_EN_US_DIR to an en_us G2P model directory")
            .into();
        let model = Model::load(&model_dir.join("oov")).expect("load OOV model");

        let words = ["zoinks", "archaeopteryx", "wibbly"];
        let sequential: Vec<Option<String>> = words
            .iter()
            .map(|word| model.predict_phonemes(word).ok().filter(|ipa| !ipa.is_empty()))
            .collect();
        let batched = model.predict_phonemes_batch(&words);

        assert_eq!(batched, sequential);
    }
}
