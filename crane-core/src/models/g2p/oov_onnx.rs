// SPDX-License-Identifier: MIT

//! OOV (out-of-vocabulary) ONNX config parsing and model loading.
//!
//! Loads a Moonshine-TTS-format OOV model: an `onnx-config.json` sidecar
//! (char/phoneme vocabularies, sequence limits, special token IDs) plus a
//! `model.onnx` encoder-decoder transformer, used as a fallback tier when a
//! word misses the lexicon and hand-written rules can't confidently resolve
//! it. Also implements the autoregressive decode inference loop: one
//! `simple_eval()` call per output phoneme token, feeding the growing
//! decoder sequence back as input. [`Model::predict_phonemes`] is greedy
//! (single best token per step, kept as the sequential reference/
//! correctness oracle); [`Model::predict_phonemes_batch`], the production
//! path, uses beam search (see [`DEFAULT_BEAM_WIDTH`]) for better accuracy.
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

/// Default beam width for beam search decoding in [`Model::predict_phonemes_batch`].  Width 3 is
/// expected to capture most of the accuracy gain over greedy decoding for this model's small
/// (131-token) vocabulary while keeping the `O(beam_width * max_phoneme_len^2)` decode cost (no
/// KV-cache) modest.
const DEFAULT_BEAM_WIDTH: usize = 3;

/// Length penalty exponent for length-normalized beam scoring, from Wu et al. (2016) ("Google's
/// Neural Machine Translation System"). Applied only when selecting the final best beam per word,
/// not during per-step pruning — mid-search pruning compares partial hypotheses whose eventual
/// length isn't known yet, so normalizing by a not-yet-final length would be premature; final
/// selection is the point where every beam's length is fixed and normalization is meaningful.
const LENGTH_NORM_ALPHA: f32 = 0.6;

/// Computes log-softmax over `logits`, returning one log-probability per
/// input element.
///
/// Uses the numerically stable formulation `log_softmax(x)_i = x_i -
/// max(x) - log(sum_j(exp(x_j - max(x))))`, matching the model's `f32`
/// logit precision. Used by beam search to turn raw logits into
/// log-probabilities that can be summed across decode steps (log-space
/// avoids the underflow that multiplying per-step probabilities directly
/// would risk over many steps).
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_sum_exp = sum_exp.ln() + max;
    logits.iter().map(|&x| x - log_sum_exp).collect()
}

/// Returns the `k` largest `(index, score)` pairs from `scores`, sorted
/// descending by score (ties broken by lower index). Returns fewer than
/// `k` pairs if `scores` has fewer than `k` elements.
///
/// `scores` is small in every caller (the 131-token phoneme vocab, or a
/// beam-width multiple of it), so a full sort is simpler than a partial
/// selection algorithm and costs nothing measurable at this scale.
fn top_k(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    indexed.truncate(k);
    indexed
}

/// Applies the Wu et al. (2016) length penalty to a cumulative beam
/// log-probability `score`, for comparing beams of different lengths when
/// picking the final best beam. `length` is the number of generated
/// phoneme tokens (excluding BOS).
fn length_normalized_score(score: f32, length: usize) -> f32 {
    // `length_f32.powf(LENGTH_NORM_ALPHA)` is 0.0 at length 0, so dividing
    // by it would produce +/-inf or NaN; return the (unpenalized) score
    // unchanged instead.
    if length == 0 {
        return score;
    }
    // Phoneme sequences are always far under 2^24, so this conversion is
    // exact.
    #[allow(clippy::cast_precision_loss)]
    let length_f32 = length as f32;
    score / length_f32.powf(LENGTH_NORM_ALPHA)
}

/// One candidate sequence in beam search decoding for a single word, used
/// internally by [`Model::predict_phonemes_batch`].
struct Beam {
    /// Decoder token IDs generated so far, excluding the implicit leading
    /// `<bos>`. Grows by at most one token per decode step.
    tokens: Vec<i64>,
    /// Cumulative log-probability (sum of each step's log-softmax score for
    /// the chosen token). Not length-normalized — see
    /// [`length_normalized_score`], which is only applied when picking the
    /// final best beam, not during per-step pruning.
    score: f32,
    /// `true` once this beam has produced `<eos>` or `<pad>`; a finished
    /// beam is carried forward unchanged in subsequent steps rather than
    /// being expanded further.
    finished: bool,
}

/// Loaded OOV encoder-decoder model: parsed config plus the ONNX session.
///
/// Autoregressive decoding is not implemented here — this type only covers
/// loading the config and the ONNX model file.
pub struct Model {
    /// Parsed and validated `onnx-config.json`.
    pub config: Config,
    /// ONNX session for the encoder-decoder model, with initializer tensors
    /// decoded once at load time rather than on every decode-step
    /// `simple_eval()` call — see [`crate::onnx::Session`].
    pub session: crate::onnx::Session,
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
        let session = crate::onnx::Session::new(model)
            .with_context(|| format!("building ONNX session from {}", onnx_path.display()))?;

        Ok(Self { config, session })
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

            let outputs = self.session.run(inputs)?;
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

    /// Runs autoregressive beam search decoding (width
    /// [`DEFAULT_BEAM_WIDTH`]) for multiple words in one synchronized batch:
    /// a single `simple_eval()` call per decode step covers every word's
    /// beams at once, instead of a full decode loop per word (see
    /// [`Self::predict_phonemes`], which stays greedy as the sequential
    /// reference/correctness oracle). For `N` words where the longest needs
    /// `M` decode steps, this is `M` total `simple_eval()` calls instead of
    /// up to `N × beam_width × M` — the real latency win for a request
    /// containing several OOV words. Each word's `beam_width` candidate
    /// sequences occupy contiguous rows in the effective batch (`N ×
    /// beam_width` rows total); a word's beams individually finish decoding
    /// (hit `<eos>`/`<pad>`) at different steps and are carried forward
    /// unchanged once finished, so the loop runs until every word's beams
    /// are all finished or `max_phoneme_len` is reached. The final IPA per
    /// word comes from the beam with the highest length-normalized score
    /// (see [`length_normalized_score`]) among that word's beams.
    ///
    /// Returns one entry per input word, in the same order: `None` for an
    /// empty word, one whose best beam decodes to an empty result, or one
    /// whose best beam hit a decode error (an out-of-vocabulary predicted
    /// token ID) — matching [`Self::predict_phonemes`]'s per-word contract
    /// (see `EnglishG2p::text_to_ipa`, which treats both identically). Only a
    /// failure that aborts the whole batch (malformed tensors, a missing
    /// `logits` output) degrades every word to `None`; a single word's
    /// decode error does not affect its sibling words — one bad word must
    /// not cost the rest of the batch a result it already computed
    /// correctly.
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
        let num_words = active.len();
        let beam_width = DEFAULT_BEAM_WIDTH;
        let eff_batch = num_words * beam_width;

        // Encoder inputs: each word's encoding is replicated across its
        // `beam_width` rows in the effective batch, since beam search runs
        // `beam_width` parallel decode candidates per word through the same
        // encoder input.
        let mut enc_ids = vec![self.config.char_pad_id; eff_batch * max_seq_len];
        let mut enc_mask = vec![0i64; eff_batch * max_seq_len];
        for (w, &word_idx) in active.iter().enumerate() {
            let encoded = self.config.encode_word(words[word_idx]);
            for k in 0..beam_width {
                let row = (w * beam_width + k) * max_seq_len;
                for (i, &id) in encoded.iter().enumerate() {
                    enc_ids[row + i] = id;
                    enc_mask[row + i] = 1;
                }
            }
        }
        let enc_ids_tensor = Tensor::from_slice(&enc_ids, &[eff_batch, max_seq_len], &device)?;
        let enc_mask_tensor = Tensor::from_slice(&enc_mask, &[eff_batch, max_seq_len], &device)?;

        // Preallocated once, refilled from `beams` at the top of each
        // decode step — same rationale as the scratch buffers in
        // `predict_phonemes`.
        let mut dec_ids = vec![self.config.phoneme_pad_id; eff_batch * max_phoneme_len];
        let mut dec_mask = vec![0i64; eff_batch * max_phoneme_len];

        // Per-word beam state. Each word starts with a single beam (empty
        // token sequence); the first decode step's top-`beam_width`
        // expansion grows it to `beam_width` distinct beams. Until then,
        // rows `1..beam_width` of a word's slice in the effective batch are
        // unused scratch space — the buffer-refresh loop below fills them
        // with a duplicate of row 0 so `simple_eval`'s fixed batch shape is
        // satisfied, but their (redundant) logits are never consulted for
        // candidate generation, since that iterates over `beams[w]` itself
        // (length 1 at step 1), not over the fixed `beam_width` row slots.
        let mut beams: Vec<Vec<Beam>> = (0..num_words)
            .map(|_| vec![Beam { tokens: Vec::new(), score: 0.0, finished: false }])
            .collect();

        let mut step = 1usize;
        while step < max_phoneme_len
            && beams.iter().any(|word_beams| word_beams.iter().any(|b| !b.finished))
        {
            Self::refill_decoder_buffers(
                &beams,
                beam_width,
                max_phoneme_len,
                &self.config,
                &mut dec_ids,
                &mut dec_mask,
            );

            let dec_ids_tensor =
                Tensor::from_slice(&dec_ids, &[eff_batch, max_phoneme_len], &device)?;
            let dec_mask_tensor =
                Tensor::from_slice(&dec_mask, &[eff_batch, max_phoneme_len], &device)?;

            // Encoder tensors are shared across every decode step; `clone()`
            // bumps the tensor's internal Arc, it does not copy the
            // underlying data.
            let inputs = HashMap::from([
                ("encoder_input_ids".to_string(), enc_ids_tensor.clone()),
                ("encoder_attention_mask".to_string(), enc_mask_tensor.clone()),
                ("decoder_input_ids".to_string(), dec_ids_tensor),
                ("decoder_attention_mask".to_string(), dec_mask_tensor),
            ]);

            let outputs = self.session.run(inputs)?;
            let logits = outputs
                .get("logits")
                .ok_or_else(|| anyhow::anyhow!("OOV model returned no logits output"))?;

            // One slice + one bulk copy for the whole effective batch
            // instead of one per row — `step_logits[row]` is that row's
            // slice over the vocab dimension at this step.
            let step_logits = logits.narrow(1, step - 1, 1)?.squeeze(1)?.to_vec2::<f32>()?;

            for (w, beams_w) in beams.iter_mut().enumerate() {
                Self::expand_word_beams(&self.config, beams_w, w, beam_width, &step_logits);
            }
            step += 1;
        }

        for (w, &word_idx) in active.iter().enumerate() {
            let best = beams[w].iter().max_by(|a, b| {
                length_normalized_score(a.score, a.tokens.len())
                    .total_cmp(&length_normalized_score(b.score, b.tokens.len()))
            });
            // A bad token ID in this word's best beam alone must not
            // discard the other words' already-decoded results, so
            // failures here resolve to `None` for this word rather than
            // propagating with `?`.
            let ipa: Option<String> = best.and_then(|beam| {
                let mut ipa = String::new();
                for &token_id in &beam.tokens {
                    let idx = usize::try_from(token_id).ok()?;
                    ipa.push_str(self.config.id_to_phoneme.get(idx)?);
                }
                (!ipa.is_empty()).then_some(ipa)
            });
            results[word_idx] = ipa;
        }

        Ok(results)
    }

    /// Refills `dec_ids`/`dec_mask` from `beams`'s current token sequences,
    /// ahead of a decode step's `simple_eval` call. A beam slot with no
    /// beam yet (before the first step's top-`beam_width` expansion, see
    /// [`Self::predict_phonemes_batch_inner`]) is filled with an empty
    /// sequence (just `<bos>`), duplicating row 0's content, so
    /// `simple_eval`'s fixed batch shape is satisfied even though that
    /// row's logits are never consulted for candidate generation.
    fn refill_decoder_buffers(
        beams: &[Vec<Beam>],
        beam_width: usize,
        max_phoneme_len: usize,
        config: &Config,
        dec_ids: &mut [i64],
        dec_mask: &mut [i64],
    ) {
        for (w, beams_w) in beams.iter().enumerate() {
            for k in 0..beam_width {
                let row = w * beam_width + k;
                let base = row * max_phoneme_len;
                dec_ids[base] = config.phoneme_bos_id;
                dec_mask[base] = 1;
                let tokens = beams_w.get(k).map_or(&[][..], |b| b.tokens.as_slice());
                debug_assert!(
                    tokens.len() < max_phoneme_len,
                    "beam token count must leave room for the leading <bos>"
                );
                let mut pos = 1usize;
                for &tok in tokens {
                    dec_ids[base + pos] = tok;
                    dec_mask[base + pos] = 1;
                    pos += 1;
                }
                for i in pos..max_phoneme_len {
                    dec_ids[base + i] = config.phoneme_pad_id;
                    dec_mask[base + i] = 0;
                }
            }
        }
    }

    /// Expands the word at `active_idx` (an index into the active-words
    /// list, not the original `words`/`results` slices)'s beams by one
    /// decode step using `step_logits` (already narrowed to this step's
    /// `[eff_batch, vocab]` slice), replacing `beams_w` in place with the
    /// new top-`beam_width` beams. No-op if all of the word's beams are
    /// already finished.
    ///
    /// Candidates are `(parent_beam_index, token, cumulative_score)`.
    /// `token` is `None` for a finished beam carried forward unchanged
    /// (competing on its already-fixed score against fresh expansions of
    /// the still-active beams). Restricting each active beam to its own
    /// top-`beam_width` tokens (rather than every vocab token) is sound:
    /// the overall top-`beam_width` candidates across all beams must be a
    /// subset of the union of each beam's own top-`beam_width` (a k-way
    /// merge argument), and it keeps the candidate pool small
    /// (`beam_width^2` instead of `beam_width * vocab`).
    fn expand_word_beams(
        config: &Config,
        beams_w: &mut Vec<Beam>,
        active_idx: usize,
        beam_width: usize,
        step_logits: &[Vec<f32>],
    ) {
        if beams_w.iter().all(|b| b.finished) {
            return;
        }

        let mut candidates: Vec<(usize, Option<i64>, f32)> = Vec::new();
        for (k, beam) in beams_w.iter().enumerate() {
            if beam.finished {
                candidates.push((k, None, beam.score));
                continue;
            }
            let row = active_idx * beam_width + k;
            let log_probs = log_softmax(&step_logits[row]);
            for (token_id, log_p) in top_k(&log_probs, beam_width) {
                // `token_id` is bounded by the phoneme vocab size (131 for
                // the shipped English model), always far under i64::MAX.
                #[allow(clippy::cast_possible_wrap)]
                let token_id = token_id as i64;
                candidates.push((k, Some(token_id), beam.score + log_p));
            }
        }
        // Tie-break by lower parent beam index, matching `top_k`'s
        // tie-break, so ordering stays deterministic even if this is later
        // changed to `sort_unstable_by` (which offers no tie stability).
        candidates.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        candidates.truncate(beam_width);

        let mut new_beams = Vec::with_capacity(beam_width);
        for (parent_k, token, score) in candidates {
            let parent_tokens = &beams_w[parent_k].tokens;
            // `parent_tokens` is at most `max_phoneme_len` long, and this
            // clone runs at most `beam_width` times per decode step; the
            // `simple_eval()` call dominates per-step cost by orders of
            // magnitude, so cloning here isn't worth the complexity of
            // avoiding it (e.g. `Rc`-sharing tokens across beams).
            match token {
                None => new_beams.push(Beam { tokens: parent_tokens.clone(), score, finished: true }),
                Some(token_id) => {
                    let is_end =
                        token_id == config.phoneme_eos_id || token_id == config.phoneme_pad_id;
                    let mut tokens = parent_tokens.clone();
                    if !is_end {
                        tokens.push(token_id);
                    }
                    new_beams.push(Beam { tokens, score, finished: is_end });
                }
            }
        }
        *beams_w = new_beams;
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
        let model = crate::onnx::proto::ModelProto {
            graph: Some(GraphProto::default()),
            ..Default::default()
        };
        Model {
            config: Config::from_json(&valid_config_json()).unwrap(),
            session: crate::onnx::Session::new(model).unwrap(),
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
    /// `[2 * DEFAULT_BEAM_WIDTH, 4, 5]` logits tensor (effective batch,
    /// `max_phoneme_len`, vocab) regardless of its inputs, so decoding is
    /// fully deterministic without needing a real transformer. Vocab IDs
    /// follow [`small_phoneme_len_config_json`]: `<pad>`=0, `<unk>`=1,
    /// `<bos>`=2, `<eos>`=3, `æ`=4.
    ///
    /// Beam search gives each word `DEFAULT_BEAM_WIDTH` rows in the
    /// effective batch; since this constant graph can't vary its output by
    /// which beam row is asking (unlike a real model, whose output would
    /// depend on that row's actual decoded-so-far sequence), every row in a
    /// word's group is given the same fixed per-step logits. The first
    /// `DEFAULT_BEAM_WIDTH` rows (word "a") predict `æ` then `<eos>`
    /// (finishes after 1 phoneme); the next `DEFAULT_BEAM_WIDTH` rows (word
    /// "ab") predict `æ`, `æ`, then `<eos>` (finishes after 2 phonemes) —
    /// used to prove a word that finishes early doesn't affect a
    /// still-active word's decoding. Every non-target vocab entry gets a
    /// large negative logit so beam search's other `DEFAULT_BEAM_WIDTH - 1`
    /// decoy beams per word never outscore the intended path, keeping the
    /// final best-beam selection deterministic.
    fn interleaved_finish_model() -> Model {
        const PHONEME_LEN: usize = 4;
        const VOCAB: usize = 5;
        const NEG: f32 = -1e9;
        let beam_width = DEFAULT_BEAM_WIDTH;
        let eff_batch = 2 * beam_width;
        let idx = |row: usize, pos: usize, id: usize| (row * PHONEME_LEN + pos) * VOCAB + id;

        let mut logits = vec![NEG; eff_batch * PHONEME_LEN * VOCAB];
        for k in 0..beam_width {
            logits[idx(k, 0, 4)] = 1.0; // word "a": step 0 -> æ
            logits[idx(k, 1, 3)] = 1.0; // word "a": step 1 -> <eos>

            let row = beam_width + k;
            logits[idx(row, 0, 4)] = 1.0; // word "ab": step 0 -> æ
            logits[idx(row, 1, 4)] = 1.0; // word "ab": step 1 -> æ
            logits[idx(row, 2, 3)] = 1.0; // word "ab": step 2 -> <eos>
        }

        let graph = GraphProto {
            node: vec![float_const_node(
                "logits",
                vec![eff_batch as i64, PHONEME_LEN as i64, VOCAB as i64],
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
            session: crate::onnx::Session::new(model).unwrap(),
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

    #[test]
    fn log_softmax_sums_to_one_in_probability_space() {
        let log_probs = log_softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = log_probs.iter().map(|&lp| lp.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    #[test]
    fn log_softmax_preserves_relative_order() {
        let log_probs = log_softmax(&[0.1, 0.9, 0.4]);
        assert!(log_probs[1] > log_probs[2]);
        assert!(log_probs[2] > log_probs[0]);
    }

    #[test]
    fn log_softmax_uniform_input_gives_uniform_output() {
        let log_probs = log_softmax(&[1.0, 1.0, 1.0, 1.0]);
        let expected = 0.25f32.ln();
        for lp in log_probs {
            assert!((lp - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn top_k_returns_largest_first() {
        let result = top_k(&[0.1, 0.9, 0.4, 0.7], 2);
        assert_eq!(result, vec![(1, 0.9), (3, 0.7)]);
    }

    #[test]
    fn top_k_breaks_ties_by_lower_index() {
        let result = top_k(&[0.5, 0.5, 0.1], 2);
        assert_eq!(result, vec![(0, 0.5), (1, 0.5)]);
    }

    #[test]
    fn top_k_saturates_at_input_length() {
        let result = top_k(&[0.1, 0.2], 5);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn top_k_zero_returns_empty() {
        assert!(top_k(&[0.1, 0.2], 0).is_empty());
    }

    #[test]
    fn length_normalized_score_zero_length_returns_score_unchanged() {
        assert!((length_normalized_score(-3.0, 0) - (-3.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn length_normalized_score_reduces_magnitude_of_negative_score() {
        // A negative score divided by length^alpha (>1 for length>1) moves
        // toward zero, i.e. its magnitude shrinks.
        let normalized = length_normalized_score(-4.0, 4);
        assert!(normalized > -4.0);
        assert!(normalized < 0.0);
    }

    /// A `Model` whose graph is a single `Constant` node emitting a fixed
    /// `[DEFAULT_BEAM_WIDTH, 4, 5]` logits tensor for a single word: every
    /// row (one per beam) carries the same clearly-peaked per-step pattern
    /// (`æ`, `æ`, then `<eos>`), with every non-target vocab entry set to a
    /// large negative logit. Used to verify beam search degenerates to the
    /// same result as greedy decoding when the model has no real
    /// ambiguity in its predictions.
    fn peaked_single_word_model() -> Model {
        const PHONEME_LEN: usize = 4;
        const VOCAB: usize = 5;
        const NEG: f32 = -1e9;
        let beam_width = DEFAULT_BEAM_WIDTH;
        let idx = |row: usize, pos: usize, id: usize| (row * PHONEME_LEN + pos) * VOCAB + id;

        let mut logits = vec![NEG; beam_width * PHONEME_LEN * VOCAB];
        for k in 0..beam_width {
            logits[idx(k, 0, 4)] = 1.0; // step 0 -> æ
            logits[idx(k, 1, 4)] = 1.0; // step 1 -> æ
            logits[idx(k, 2, 3)] = 1.0; // step 2 -> <eos>
        }

        let graph = GraphProto {
            node: vec![float_const_node(
                "logits",
                vec![beam_width as i64, PHONEME_LEN as i64, VOCAB as i64],
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
            session: crate::onnx::Session::new(model).unwrap(),
        }
    }

    #[test]
    fn beam_search_degenerates_to_greedy_on_peaked_logits() {
        let model = peaked_single_word_model();
        let greedy = model.predict_phonemes("a").unwrap();
        let beam = model.predict_phonemes_batch(&["a"]);
        assert_eq!(beam, vec![Some(greedy)]);
    }

    /// A `Model` whose graph is a single `Constant` node where `<eos>`
    /// dominates the very first decode step for every beam row, so the
    /// whole beam collapses to an empty decoded sequence immediately. Used
    /// to verify [`Model::predict_phonemes_batch`] returns `None` for a
    /// word whose best beam decodes to nothing, matching
    /// [`Model::predict_phonemes`]'s "empty result -> `None`" contract.
    fn all_eos_first_step_model() -> Model {
        const PHONEME_LEN: usize = 4;
        const VOCAB: usize = 5;
        const NEG: f32 = -1e9;
        let beam_width = DEFAULT_BEAM_WIDTH;
        let idx = |row: usize, pos: usize, id: usize| (row * PHONEME_LEN + pos) * VOCAB + id;

        let mut logits = vec![NEG; beam_width * PHONEME_LEN * VOCAB];
        for row in 0..beam_width {
            logits[idx(row, 0, 3)] = 10.0; // <eos>, clearly dominant
            logits[idx(row, 0, 0)] = 5.0; // <pad>, second highest, also ends the beam
        }

        let graph = GraphProto {
            node: vec![float_const_node(
                "logits",
                vec![beam_width as i64, PHONEME_LEN as i64, VOCAB as i64],
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
            session: crate::onnx::Session::new(model).unwrap(),
        }
    }

    #[test]
    fn beam_search_all_eos_first_step() {
        let model = all_eos_first_step_model();
        let results = model.predict_phonemes_batch(&["x"]);
        assert_eq!(results, vec![None]);
    }

    fn two_phoneme_config_json() -> String {
        r#"{
            "config_schema_version": 1,
            "model_kind": "oov",
            "char_vocab": {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3},
            "phoneme_vocab": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "æ": 4, "ə": 5},
            "train_config": {"max_seq_len": 8},
            "oov_index": {"max_phoneme_len": 4}
        }"#
        .to_string()
    }

    /// A `Model` whose graph is a single `Constant` node, over a
    /// `DEFAULT_BEAM_WIDTH`-row effective batch for a single word,
    /// deliberately constructed so the greedy-optimal first token (`æ`)
    /// leads to a lower total joint log-probability than the second-best
    /// first token (`ə`), which leads to a high-confidence continuation.
    /// Used to prove beam search explores and keeps the globally better
    /// path instead of committing to the locally-best (greedy) first
    /// choice.
    ///
    /// Step 0 gives every row `æ` (logit 2.0, the greedy pick) and `ə`
    /// (logit 1.9, a close second) — everything else is strongly
    /// negative. The gap must stay small: `ə`'s only advantage is its
    /// confident step-1 continuation, so if `æ` starts too far ahead at
    /// step 0, no later step can close the difference. After step 0, beam
    /// search's rank-0 beam (`æ`) occupies
    /// row 0 and rank-1 (`ə`) occupies row 1. Step 1's row 0 gives that
    /// beam a costly, evenly-split choice between `<unk>` and `<eos>`
    /// (roughly halving its already-lower joint probability), while row 1
    /// gives the `ə` beam a confident, near-certain continuation into
    /// `æ`. By step 2, the `ə`-first beam's joint log-probability has
    /// overtaken the `æ`-first beam's, so it wins the final
    /// length-normalized comparison — even though `æ` looked better after
    /// a single step, which is exactly what greedy (always following row
    /// 0) commits to.
    fn diverging_paths_model() -> Model {
        const PHONEME_LEN: usize = 4;
        const VOCAB: usize = 6;
        const NEG: f32 = -1e9;
        let beam_width = DEFAULT_BEAM_WIDTH;
        let idx = |row: usize, pos: usize, id: usize| (row * PHONEME_LEN + pos) * VOCAB + id;

        let mut logits = vec![NEG; beam_width * PHONEME_LEN * VOCAB];

        // Step 0 (only row 0 is real at this point — the sole starting
        // beam): "æ" (id 4) is the greedy pick, "ə" (id 5) a close second.
        for row in 0..beam_width {
            logits[idx(row, 0, 4)] = 2.0;
            logits[idx(row, 0, 5)] = 1.9;
        }

        // Step 1, row 0: whichever beam ranks 0 after step 0 (the "æ"
        // beam) sees a costly, evenly-split choice between "<unk>" and
        // "<eos>".
        logits[idx(0, 1, 1)] = 0.0; // <unk>
        logits[idx(0, 1, 3)] = 0.0; // <eos>

        // Step 1, row 1: whichever beam ranks 1 after step 0 (the "ə"
        // beam) sees a confident continuation into "æ".
        logits[idx(1, 1, 4)] = 5.0;

        // Step 2: whichever beam now occupies row 0 (the "ə" + "æ" beam,
        // which overtakes rank 0 after step 1) confidently finishes.
        logits[idx(0, 2, 3)] = 5.0;
        // Row 1's beam (the "æ" + "<unk>" alternative) also confidently
        // finishes, so it doesn't linger as an unfinished distraction.
        logits[idx(1, 2, 3)] = 5.0;

        let graph = GraphProto {
            node: vec![float_const_node(
                "logits",
                vec![beam_width as i64, PHONEME_LEN as i64, VOCAB as i64],
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
            config: Config::from_json(&two_phoneme_config_json()).unwrap(),
            session: crate::onnx::Session::new(model).unwrap(),
        }
    }

    #[test]
    fn beam_search_finds_better_path() {
        let model = diverging_paths_model();

        // Greedy always follows row 0, so it commits to "æ" at step 0 (the
        // higher of the two step-0 logits) and never reconsiders "ə".
        let greedy = model.predict_phonemes("a").unwrap();
        assert!(greedy.starts_with('æ'));

        // Beam search keeps both step-0 candidates alive; "ə" (rank 1)
        // leads to a confident "æ" continuation whose joint log-probability
        // ends up higher than "æ"'s own costly, evenly-split continuation
        // — so the best beam switches to the "ə"-first path.
        let beam = model.predict_phonemes_batch(&["a"]);
        assert_eq!(beam, vec![Some("əæ".to_string())]);
    }

    /// Real-model regression test: verifies that batching multiple OOV
    /// words into one `predict_phonemes_batch` call produces the same
    /// per-word IPA as calling `predict_phonemes_batch` once per word (each
    /// its own single-word beam search) — i.e. that batching decode steps
    /// together across words doesn't change beam search's result. Does
    /// *not* compare against the greedy `predict_phonemes` oracle, since
    /// beam search is expected to (and, per the CER benchmark, should)
    /// diverge from greedy on some words. Needs a real OOV model on disk,
    /// so `#[ignore]`d by default. Run with:
    ///
    /// ```sh
    /// CRANE_G2P_EN_US_DIR=/path/to/en_us \
    ///   cargo test -p crane-core --features onnx -- \
    ///   predict_phonemes_batch_matches_single_word_batches --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a local G2P model directory (CRANE_G2P_EN_US_DIR)"]
    fn predict_phonemes_batch_matches_single_word_batches() {
        let model_dir: std::path::PathBuf = std::env::var("CRANE_G2P_EN_US_DIR")
            .expect("set CRANE_G2P_EN_US_DIR to an en_us G2P model directory")
            .into();
        let model = Model::load(&model_dir.join("oov")).expect("load OOV model");

        let words = ["zoinks", "archaeopteryx", "wibbly"];
        let per_word: Vec<Option<String>> =
            words.iter().flat_map(|&word| model.predict_phonemes_batch(&[word])).collect();
        let batched = model.predict_phonemes_batch(&words);

        assert_eq!(batched, per_word);
    }
}
