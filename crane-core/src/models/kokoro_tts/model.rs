// SPDX-License-Identifier: MIT

//! Kokoro TTS `Model`: config/vocab parsing, voice loading, and the ONNX
//! forward pass.
//!
//! [`Model::new`] loads the ONNX graph, the phoneme vocabulary, discovers
//! available voice names, and lazily loads/caches per-voice style embeddings
//! from their `.bin` files. [`Model::generate_speech`] phonemizes text with a
//! caller-supplied [`Phonemizer`], normalizes the result to Kokoro's phoneme
//! inventory, and runs the ONNX forward pass to produce PCM audio.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

use crate::generation::SpeechOptions;
use crate::models::g2p::Phonemizer;
use crate::models::g2p::ipa_postprocess::IpaNormalizer;

use super::ipa::build_kokoro_normalizer;

/// Kokoro always outputs mono PCM at 24 kHz.
const KOKORO_SAMPLE_RATE: u32 = 24_000;

/// Style embedding dimension. Not present in any shipped config file —
/// inferred from the ONNX graph's `style` input shape (`[1, 256]`) and from
/// voice tensor byte sizes (510 × 256 float32 for `af_heart.bin`) and
/// hardcoded here.
const KOKORO_STYLE_DIM: usize = 256;

/// Maximum number of phoneme codepoints per Kokoro ONNX call. Voice style
/// matrices have on the order of ~510 rows (one style vector per possible
/// phoneme-count index) and the model rejects longer token sequences, so
/// longer input is split into chunks of at most this many phoneme
/// codepoints each — matching Moonshine's `chunk_phonemes()` default.
const MAX_PHONEME_CODEPOINTS: usize = 510;

/// The `model_type` value every known Kokoro ONNX export's `config.json`
/// carries. `config.json` otherwise has no other keys to validate against.
const EXPECTED_MODEL_TYPE: &str = "style_text_to_speech_2";

/// Voice loaded eagerly at construction to fail fast on a broken model
/// directory. Falls back to the first discovered voice if this one isn't
/// present.
const DEFAULT_VOICE: &str = "af_heart";

/// Language substituted for the `"auto"` sentinel (or an empty string) in
/// [`Model::generate_speech`], since Kokoro's phonemizer has no
/// language-detection of its own — unlike Qwen3-TTS, whose codec can infer
/// language from the text itself when `"auto"` is requested. Currently the
/// only language [`MoonshineG2p`](crate::models::g2p::MoonshineG2p) loads is
/// `en_us`, so this is also the only sensible default.
const DEFAULT_LANGUAGE: &str = "en_us";

/// Maps an ISO 639-1 language code — the format the `Tts` trait boundary
/// standardizes on (see `crane::audio::tts_qwen3`'s `language_code_to_name`
/// for the Qwen3 equivalent) — to the language identifier
/// [`MoonshineG2p`](crate::models::g2p::MoonshineG2p) actually loads (e.g.
/// `"en_us"`).
///
/// Codes not in the mapping — including already-resolved identifiers like
/// `"en_us"` — pass through unchanged, so both formats work. Only `"en"` is
/// mapped today because `en_us` is the only language
/// [`SUPPORTED_LANGUAGES`](crate::models::g2p::languages::SUPPORTED_LANGUAGES)
/// lists; extend this as more languages are added.
fn iso_code_to_language(code: &str) -> &str {
    match code {
        "en" => DEFAULT_LANGUAGE,
        other => other,
    }
}

/// Minimal `config.json` shape: only `model_type` is present in real exports.
#[derive(Debug, Deserialize)]
struct ConfigJson {
    model_type: String,
}

/// The `model` field of `tokenizer.json`, containing the phoneme vocabulary.
#[derive(Debug, Deserialize)]
struct TokenizerModel {
    vocab: HashMap<String, i64>,
}

/// Minimal `tokenizer.json` shape: only `model.vocab` is needed.
#[derive(Debug, Deserialize)]
struct TokenizerJson {
    model: TokenizerModel,
}

/// Minimal `tokenizer_config.json` shape: only the max sequence length is
/// needed.
#[derive(Debug, Deserialize)]
struct TokenizerConfigJson {
    model_max_length: usize,
}

/// Reads `tokenizer.json` and converts its `model.vocab` map into
/// `HashMap<char, i64>`, since every Kokoro vocab key is a single codepoint
/// (verified against the real 115-entry vocab shipped in this model's
/// `tokenizer.json`).
fn parse_vocab(path: &Path) -> Result<HashMap<char, i64>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading tokenizer.json at {}", path.display()))?;
    let parsed: TokenizerJson = serde_json::from_str(&raw)
        .with_context(|| format!("parsing tokenizer.json at {}", path.display()))?;

    let mut vocab = HashMap::with_capacity(parsed.model.vocab.len());
    for (key, id) in parsed.model.vocab {
        let mut chars = key.chars();
        let c = chars.next().context("vocab key must not be empty")?;
        if chars.next().is_some() {
            bail!("vocab key {key:?} is not a single codepoint");
        }
        if vocab.insert(c, id).is_some() {
            bail!("vocab key {key:?} duplicates an already-seen codepoint {c:?}");
        }
    }
    Ok(vocab)
}

/// Reads `tokenizer_config.json` and returns `model_max_length`.
fn parse_max_seq_len(path: &Path) -> Result<usize> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading tokenizer_config.json at {}", path.display()))?;
    let parsed: TokenizerConfigJson = serde_json::from_str(&raw)
        .with_context(|| format!("parsing tokenizer_config.json at {}", path.display()))?;
    Ok(parsed.model_max_length)
}

/// Validates `config.json`'s `model_type` matches the expected Kokoro value.
fn validate_config(path: &Path) -> Result<()> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading config.json at {}", path.display()))?;
    let parsed: ConfigJson = serde_json::from_str(&raw)
        .with_context(|| format!("parsing config.json at {}", path.display()))?;
    if parsed.model_type != EXPECTED_MODEL_TYPE {
        bail!(
            "unexpected Kokoro model_type {:?}, expected {:?}",
            parsed.model_type,
            EXPECTED_MODEL_TYPE
        );
    }
    Ok(())
}

/// Discovers available voice names from `.bin` files in `voice_dir`, without
/// reading their contents. Returns names sorted for deterministic output.
fn discover_voices(voice_dir: &Path) -> Result<Vec<String>> {
    let mut voices = Vec::new();
    let entries = std::fs::read_dir(voice_dir)
        .with_context(|| format!("reading voices directory {}", voice_dir.display()))?;
    for entry in entries {
        let entry =
            entry.with_context(|| format!("reading entry in {}", voice_dir.display()))?;
        let path = entry.path();
        if entry.file_type()?.is_file()
            && path.extension().and_then(|ext| ext.to_str()) == Some("bin")
            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
        {
            voices.push(stem.to_string());
        }
    }
    voices.sort();
    Ok(voices)
}

/// Parses a `.bin` voice style embedding file: headerless raw little-endian
/// float32, row-major, `style_dim` columns. Row count is derived from the
/// file size rather than assumed fixed, since shipped voices are not all the
/// same row count (`af.bin` is 1024 rows; most others are 1020).
fn load_voice_bin(path: &Path, style_dim: usize) -> Result<Tensor> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading voice file {}", path.display()))?;

    if style_dim == 0 {
        bail!("style_dim must be non-zero");
    }
    if bytes.is_empty() {
        bail!("voice file {} is empty", path.display());
    }
    let row_bytes = style_dim * size_of::<f32>();
    if bytes.len() % row_bytes != 0 {
        bail!(
            "voice file {} has {} bytes, not a multiple of style_dim {} * 4 bytes",
            path.display(),
            bytes.len(),
            style_dim
        );
    }
    let rows = bytes.len() / row_bytes;

    let data: Vec<f32> = bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| {
            let raw: [u8; 4] = chunk.try_into().expect("chunks_exact(4) yields 4 bytes");
            f32::from_le_bytes(raw)
        })
        .collect();

    Tensor::from_vec(data, (rows, style_dim), &Device::Cpu)
        .with_context(|| format!("building tensor from voice file {}", path.display()))
}

/// Resolves the `language` parameter passed to [`Model::generate_speech`]:
/// `"auto"` (case-insensitive) or an empty string become [`DEFAULT_LANGUAGE`],
/// since Kokoro's phonemizer has no language-detection of its own; otherwise
/// the value is passed through [`iso_code_to_language`] so ISO 639-1 codes
/// (e.g. `"en"`) resolve to the identifier the phonemizer actually loaded.
fn resolve_language(language: &str) -> &str {
    if language.is_empty() || language.eq_ignore_ascii_case("auto") {
        DEFAULT_LANGUAGE
    } else {
        iso_code_to_language(language)
    }
}

/// Splits a normalized phoneme string into chunks of at most `max_cp`
/// codepoints each, preferring to cut at a space within the current window
/// so words stay whole where possible. Empty pieces (leading/trailing
/// whitespace collapse) are dropped. Ported from Moonshine's
/// `chunk_phonemes()`.
fn chunk_phonemes(phonemes: &str, max_cp: usize) -> Vec<String> {
    debug_assert!(max_cp > 0, "max_cp must be > 0 to avoid an infinite loop");
    let chars: Vec<char> = phonemes.chars().collect();
    if chars.len() <= max_cp {
        let piece = phonemes.trim();
        return if piece.is_empty() { Vec::new() } else { vec![piece.to_string()] };
    }

    let mut chunks = Vec::new();
    let mut rest: &[char] = &chars;
    while !rest.is_empty() {
        if rest.len() <= max_cp {
            let piece: String = rest.iter().collect::<String>();
            push_trimmed(&mut chunks, &piece);
            break;
        }

        let window_len = (max_cp + 1).min(rest.len());
        let window = &rest[..window_len];
        let cut = window
            .iter()
            .rposition(|&c| c == ' ')
            .filter(|&pos| pos > 0)
            .unwrap_or(max_cp);

        let piece: String = rest[..cut].iter().collect::<String>();
        push_trimmed(&mut chunks, &piece);

        let mut next_start = cut;
        while next_start < rest.len() && rest[next_start] == ' ' {
            next_start += 1;
        }
        rest = &rest[next_start..];
    }
    chunks
}

/// Pushes `piece` onto `chunks` after trimming, unless it's empty.
fn push_trimmed(chunks: &mut Vec<String>, piece: &str) {
    let trimmed = piece.trim();
    if !trimmed.is_empty() {
        chunks.push(trimmed.to_string());
    }
}

/// Kokoro-82M ONNX text-to-speech model.
///
/// Loaded once via [`Model::new`] and reused across `generate_speech()`
/// calls — the ONNX graph, phoneme vocab, and voice list are all immutable
/// after construction. Voice style embeddings are loaded lazily on first use
/// via [`Model::voice`] and cached for the lifetime of the model.
pub struct Model {
    /// Loaded ONNX model graph, run in full via `crate::onnx::simple_eval()`
    /// on each [`Model::generate_speech`] call.
    onnx_graph: crate::onnx::proto::ModelProto,
    /// ONNX graph initializers (model weights) pre-decoded from their
    /// protobuf `TensorProto` representation into candle `Tensor`s once in
    /// [`Model::new`], so each `simple_eval()` call doesn't re-decode them.
    /// The raw initializers in `onnx_graph.graph.initializer` are cleared
    /// after decoding.
    decoded_initializers: HashMap<String, Tensor>,
    /// Phoneme character to token ID, from `tokenizer.json`'s `model.vocab`.
    vocab: HashMap<char, i64>,
    /// Lazily loaded voice style embeddings, keyed by voice name (e.g.
    /// `"af_heart"`). One cell per name discovered at construction time;
    /// each is populated on first [`Model::voice`] call for that name.
    voices: HashMap<String, OnceLock<Tensor>>,
    /// Directory containing per-voice `.bin` style embedding files.
    voice_dir: PathBuf,
    /// Voice names discovered at construction time (`.bin` filenames minus
    /// the extension), independent of whether they've been loaded yet.
    available_voices: Vec<String>,
    /// Style embedding dimension. See [`KOKORO_STYLE_DIM`].
    style_dim: usize,
    /// Maximum phoneme sequence length, from `tokenizer_config.json`.
    max_seq_len: usize,
    /// Token ID for the pad/BOS/EOS token `$`, read from `vocab`.
    pad_token_id: i64,
    /// Lazily built per-language IPA normalizers (see
    /// [`build_kokoro_normalizer`]), keyed by language identifier (e.g.
    /// `"en_us"`). Each is built once from [`Self::vocab`] on first
    /// [`Model::generate_speech`] call for that language and reused after.
    normalizers: HashMap<String, IpaNormalizer>,
}

impl Model {
    /// Loads a Kokoro model from `model_path`, which must contain
    /// `config.json`, `tokenizer.json`, `tokenizer_config.json`,
    /// `onnx/model.onnx`, and a `voices/` directory.
    ///
    /// `device` and `dtype` are accepted for API compatibility with
    /// `create_tts()` but unused: `crate::onnx::simple_eval()` runs on CPU
    /// only, regardless of the configured device.
    ///
    /// # Errors
    ///
    /// Returns an error if any required file is missing or malformed, or if
    /// `config.json`'s `model_type` doesn't match the expected Kokoro value.
    ///
    /// # Panics
    ///
    /// Panics if `default_voice_name` (derived from `available_voices`,
    /// checked non-empty above) is somehow absent from the `voices` map
    /// built from that same list — not reachable in practice.
    pub fn new(model_path: &str, _device: &Device, _dtype: &DType) -> Result<Self> {
        let root = Path::new(model_path);

        validate_config(&root.join("config.json"))?;
        let vocab = parse_vocab(&root.join("tokenizer.json"))?;
        let max_seq_len = parse_max_seq_len(&root.join("tokenizer_config.json"))?;

        let pad_token_id = *vocab
            .get(&'$')
            .context("Kokoro vocab is missing the pad/BOS/EOS token '$'")?;

        let voice_dir = root.join("voices");
        let available_voices = discover_voices(&voice_dir)?;

        let onnx_path = root.join("onnx").join("model.onnx");
        let mut onnx_graph = crate::onnx::read_file(&onnx_path)
            .with_context(|| format!("loading Kokoro ONNX model from {}", onnx_path.display()))?;
        let graph = onnx_graph.graph.as_mut().context("Kokoro ONNX model has no graph")?;

        let mut decoded_initializers = HashMap::with_capacity(graph.initializer.len());
        for t in &graph.initializer {
            let tensor = crate::onnx::eval::get_tensor(t, t.name.as_str())
                .with_context(|| format!("decoding ONNX initializer {:?}", t.name))?;
            decoded_initializers.insert(t.name.clone(), tensor);
        }
        // Fuses decomposed atan2(imag, real) patterns (Div → Atan →
        // quadrant-correction Where) into single Atan2 nodes, rewrites ops
        // unmodified `simple_eval()` handles incorrectly (Trilu NaN on
        // +/-inf inputs, and more added as Kokoro's export needs them) into
        // decompositions it handles correctly, and folds/eliminates the
        // resulting dead and constant nodes — see `crate::onnx::optimizer`
        // and its `compat` submodule doc.
        let onnx_options = crate::onnx::SessionOptions::default();
        crate::onnx::optimize(graph, &mut decoded_initializers, &onnx_options)
            .context("optimizing Kokoro ONNX graph")?;
        graph.initializer.clear();

        // Fail fast on a broken model directory by loading one voice now,
        // rather than deferring every voice load to first request.
        let default_voice_name = available_voices
            .iter()
            .find(|name| name.as_str() == DEFAULT_VOICE)
            .or_else(|| available_voices.first())
            .context("Kokoro voices directory contains no voice files")?
            .clone();
        let default_voice = load_voice_bin(
            &voice_dir.join(format!("{default_voice_name}.bin")),
            KOKORO_STYLE_DIM,
        )?;

        let voices: HashMap<String, OnceLock<Tensor>> = available_voices
            .iter()
            .map(|name| (name.clone(), OnceLock::new()))
            .collect();
        voices
            .get(&default_voice_name)
            .expect("default voice is in available_voices")
            .get_or_init(|| default_voice);

        Ok(Self {
            onnx_graph,
            decoded_initializers,
            vocab,
            voices,
            voice_dir,
            available_voices,
            style_dim: KOKORO_STYLE_DIM,
            max_seq_len,
            pad_token_id,
            normalizers: HashMap::new(),
        })
    }

    /// Sample rate of generated audio: always 24 kHz.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        KOKORO_SAMPLE_RATE
    }

    /// Names of voices discovered under the model's `voices/` directory,
    /// independent of whether they've been loaded into memory yet.
    #[must_use]
    pub fn available_voices(&self) -> &[String] {
        &self.available_voices
    }

    /// The phoneme character to token ID vocabulary, from `tokenizer.json`.
    #[must_use]
    pub fn vocab(&self) -> &HashMap<char, i64> {
        &self.vocab
    }

    /// Returns the style embedding tensor for `name`, loading and caching it
    /// from its `.bin` file on first request.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` isn't among [`Model::available_voices`], or
    /// if its `.bin` file can't be read or parsed.
    pub fn voice(&self, name: &str) -> Result<Tensor> {
        let cell = self.voices.get(name).with_context(|| {
            format!(
                "unknown Kokoro voice {name:?}; available voices: {}",
                self.available_voices.join(", ")
            )
        })?;
        if let Some(tensor) = cell.get() {
            return Ok(tensor.clone());
        }

        // `OnceLock::get_or_try_init` is still unstable, so a fallible load
        // can't be wired directly into initialization. A concurrent first
        // request for the same voice may load its `.bin` file twice, but
        // `get_or_init` guarantees only one tensor is ever cached.
        let tensor = load_voice_bin(&self.voice_dir.join(format!("{name}.bin")), self.style_dim)?;
        Ok(cell.get_or_init(|| tensor).clone())
    }

    /// Returns the [`IpaNormalizer`] for `language`, building and caching it
    /// on first request. Building only needs [`Self::vocab`] (already
    /// loaded at construction), so this never touches disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the normalizer's replacement table fails to
    /// compile — see [`build_kokoro_normalizer`].
    fn normalizer_for(&mut self, language: &str) -> Result<&IpaNormalizer> {
        use std::collections::hash_map::Entry;
        match self.normalizers.entry(language.to_string()) {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => {
                let normalizer = build_kokoro_normalizer(language, &self.vocab)?;
                Ok(e.insert(normalizer))
            }
        }
    }

    /// Converts a normalized phoneme string into Kokoro token IDs, mirroring
    /// Moonshine's `phoneme_str_to_input_ids()`: the pad/BOS/EOS token is
    /// prepended and appended, and codepoints missing from [`Self::vocab`]
    /// are skipped (unreachable in practice, since `IpaNormalizer::normalize`
    /// already drops anything outside the vocab it was built from).
    fn phonemes_to_ids(&self, phonemes: &str) -> Vec<i64> {
        let mut ids = vec![self.pad_token_id];
        ids.extend(phonemes.chars().filter_map(|c| self.vocab.get(&c).copied()));
        ids.push(self.pad_token_id);
        ids
    }

    /// Generates speech from `text`: phonemizes it with `phonemizer`,
    /// normalizes the IPA to Kokoro's phoneme inventory, and runs the ONNX
    /// forward pass once per chunk (see [`MAX_PHONEME_CODEPOINTS`]),
    /// concatenating the resulting waveform chunks.
    ///
    /// `voice` selects the style embedding by name (see
    /// [`Model::available_voices`]); `None` uses [`DEFAULT_VOICE`]. Per
    /// chunk, the style row is selected by the chunk's phoneme codepoint
    /// count, clamped to the voice matrix's row count, matching Moonshine's
    /// `KokoroTtsEngine::synthesize`.
    ///
    /// `language` of `"auto"` (case-insensitive) or `""` is substituted with
    /// [`DEFAULT_LANGUAGE`] before being passed to `phonemizer` — Kokoro has
    /// no language-detection of its own, unlike Qwen3-TTS's codec. Any other
    /// value is resolved through [`iso_code_to_language`] first, so ISO
    /// 639-1 codes (e.g. `"en"`) work alongside `phonemizer`'s own
    /// identifiers (e.g. `"en_us"`).
    ///
    /// `opts` is accepted for API compatibility with the other TTS models'
    /// `generate_speech` signature but unused: Kokoro is a feed-forward
    /// model, so `max_new_tokens`, `temperature`, `top_p`, and
    /// `repetition_penalty` don't apply.
    ///
    /// # Errors
    ///
    /// Returns an error if phonemization fails, if `voice` isn't among
    /// [`Model::available_voices`], if a chunk's token sequence exceeds
    /// [`Self::max_seq_len`], or if ONNX inference fails.
    ///
    /// # Panics
    ///
    /// Never panics: the `expect()` on the single-chunk fast path is only
    /// reached when `waveform_chunks.len() == 1`, which the surrounding
    /// `if` already guarantees is non-empty.
    pub fn generate_speech(
        &mut self,
        text: &str,
        language: &str,
        voice: Option<&str>,
        phonemizer: &dyn Phonemizer,
        _opts: &SpeechOptions,
    ) -> Result<(Tensor, u32)> {
        let voice_name = voice.unwrap_or(DEFAULT_VOICE);
        let voice_tensor = self.voice(voice_name)?;
        let voice_rows = voice_tensor.dim(0)?;

        let language = resolve_language(language);
        let ipa = phonemizer.text_to_ipa(text, language)?;
        let normalizer = self.normalizer_for(language)?;
        let phonemes = normalizer.normalize(&ipa);

        let chunks = chunk_phonemes(&phonemes, MAX_PHONEME_CODEPOINTS);
        if chunks.is_empty() {
            bail!("no phonemes produced for input text {text:?}");
        }

        let mut waveform_chunks = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            let ids = self.phonemes_to_ids(chunk);
            if ids.len() > self.max_seq_len {
                bail!(
                    "phoneme chunk has {} tokens, exceeding Kokoro's max sequence length of {}",
                    ids.len(),
                    self.max_seq_len
                );
            }
            let ntok = ids.len();
            let input_ids = Tensor::from_vec(ids, (1, ntok), &Device::Cpu)?;

            let codepoint_count = chunk.chars().count().max(1);
            let row_idx = codepoint_count.min(voice_rows) - 1;
            let style = voice_tensor.narrow(0, row_idx, 1)?;

            let speed = Tensor::from_vec(vec![1f32], (1,), &Device::Cpu)?;

            let mut inputs = HashMap::new();
            inputs.insert("input_ids".to_string(), input_ids);
            inputs.insert("style".to_string(), style);
            inputs.insert("speed".to_string(), speed);
            inputs.extend(self.decoded_initializers.iter().map(|(k, v)| (k.clone(), v.clone())));

            let mut values = crate::onnx::simple_eval(&self.onnx_graph, inputs)?;
            let waveform =
                values.remove("waveform").context("Kokoro ONNX output is missing 'waveform'")?;
            waveform_chunks.push(waveform);
        }

        let waveform = if waveform_chunks.len() == 1 {
            waveform_chunks.into_iter().next().expect("checked len == 1")
        } else {
            Tensor::cat(&waveform_chunks, 1)?
        };

        Ok((waveform, KOKORO_SAMPLE_RATE))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    fn write_temp_json(contents: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        file
    }

    #[test]
    fn parse_vocab_from_json() {
        let file = write_temp_json(
            r#"{"model": {"vocab": {"$": 0, "a": 43, "ˈ": 156}}}"#,
        );
        let vocab = parse_vocab(file.path()).unwrap();
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab[&'$'], 0);
        assert_eq!(vocab[&'a'], 43);
        assert_eq!(vocab[&'\u{02c8}'], 156);
    }

    #[test]
    fn parse_vocab_rejects_multi_codepoint_key() {
        let file = write_temp_json(r#"{"model": {"vocab": {"ab": 0}}}"#);
        let err = parse_vocab(file.path()).unwrap_err();
        assert!(err.to_string().contains("not a single codepoint"));
    }

    #[test]
    fn parse_vocab_rejects_empty_key() {
        let file = write_temp_json(r#"{"model": {"vocab": {"": 0}}}"#);
        assert!(parse_vocab(file.path()).is_err());
    }

    #[test]
    fn parse_tokenizer_config() {
        let file = write_temp_json(
            r#"{"model_max_length": 512, "pad_token": "$", "tokenizer_class": "PreTrainedTokenizer", "unk_token": "$"}"#,
        );
        assert_eq!(parse_max_seq_len(file.path()).unwrap(), 512);
    }

    #[test]
    fn validate_config_accepts_expected_model_type() {
        let file = write_temp_json(r#"{"model_type": "style_text_to_speech_2"}"#);
        assert!(validate_config(file.path()).is_ok());
    }

    #[test]
    fn validate_config_rejects_unexpected_model_type() {
        let file = write_temp_json(r#"{"model_type": "something_else"}"#);
        let err = validate_config(file.path()).unwrap_err();
        assert!(err.to_string().contains("unexpected Kokoro model_type"));
    }

    #[test]
    fn discover_voices_finds_bin_files_sorted() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("af_heart.bin"), []).unwrap();
        std::fs::write(dir.path().join("am_adam.bin"), []).unwrap();
        std::fs::write(dir.path().join("README.md"), []).unwrap();
        let voices = discover_voices(dir.path()).unwrap();
        assert_eq!(voices, vec!["af_heart".to_string(), "am_adam".to_string()]);
    }

    #[test]
    fn discover_voices_errors_on_missing_dir() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does_not_exist");
        assert!(discover_voices(&missing).is_err());
    }

    #[test]
    fn discover_voices_ignores_directories() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("af_heart.bin"), []).unwrap();
        std::fs::create_dir(dir.path().join("fake.bin")).unwrap();
        let voices = discover_voices(dir.path()).unwrap();
        assert_eq!(voices, vec!["af_heart".to_string()]);
    }

    fn write_voice_bin(values: &[f32]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for v in values {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        file
    }

    /// Builds a `Model` without going through `Model::new`, so voice-loading
    /// tests don't need a real ONNX file, config, or tokenizer on disk.
    fn test_model(voice_dir: &Path, available_voices: Vec<String>) -> Model {
        let voices = available_voices
            .iter()
            .map(|name| (name.clone(), OnceLock::new()))
            .collect();
        Model {
            onnx_graph: crate::onnx::proto::ModelProto::default(),
            decoded_initializers: HashMap::new(),
            vocab: HashMap::new(),
            voices,
            voice_dir: voice_dir.to_path_buf(),
            available_voices,
            style_dim: 2,
            max_seq_len: 0,
            pad_token_id: 0,
            normalizers: HashMap::new(),
        }
    }

    /// A small hand-picked vocab covering the phonemes used by the
    /// `phonemes_to_ids`/`chunk_phonemes` tests below: `$` (pad/BOS/EOS),
    /// `a`, `b`, and a space.
    fn small_vocab() -> HashMap<char, i64> {
        [('$', 0), ('a', 1), ('b', 2), (' ', 3)].into_iter().collect()
    }

    fn test_model_with_vocab(vocab: HashMap<char, i64>) -> Model {
        let mut model = test_model(Path::new("/nonexistent"), Vec::new());
        model.pad_token_id = *vocab.get(&'$').unwrap_or(&0);
        model.vocab = vocab;
        model
    }

    #[test]
    fn load_voice_bin_parses_raw_f32() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let file = write_voice_bin(&values);
        let tensor = load_voice_bin(file.path(), 2).unwrap();
        assert_eq!(tensor.dims(), &[4, 2]);
        let data = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(data, values);
    }

    #[test]
    fn load_voice_bin_rejects_misaligned_file() {
        let file = write_voice_bin(&[1.0, 2.0, 3.0]);
        let err = load_voice_bin(file.path(), 2).unwrap_err();
        assert!(err.to_string().contains("not a multiple"));
    }

    #[test]
    fn load_voice_bin_rejects_empty_file() {
        let file = NamedTempFile::new().unwrap();
        assert!(load_voice_bin(file.path(), 2).is_err());
    }

    #[test]
    fn load_voice_bin_rejects_zero_style_dim() {
        let file = write_voice_bin(&[1.0, 2.0]);
        let err = load_voice_bin(file.path(), 0).unwrap_err();
        assert!(err.to_string().contains("style_dim must be non-zero"));
    }

    #[test]
    fn voice_returns_cached_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(dir.path().join("test_voice.bin"), bytes).unwrap();
        let model = test_model(dir.path(), vec!["test_voice".to_string()]);

        let first = model.voice("test_voice").unwrap();
        let second = model.voice("test_voice").unwrap();
        assert_eq!(first.dims(), &[2, 2]);
        assert_eq!(second.dims(), &[2, 2]);
        assert!(model.voices.get("test_voice").unwrap().get().is_some());
    }

    #[test]
    fn voice_errors_on_missing_bin_file() {
        let dir = tempfile::tempdir().unwrap();
        let model = test_model(dir.path(), vec!["ghost_voice".to_string()]);
        assert!(model.voice("ghost_voice").is_err());
    }

    #[test]
    fn voice_rejects_unknown_name() {
        let dir = tempfile::tempdir().unwrap();
        let model = test_model(dir.path(), vec!["known".to_string()]);
        let err = model.voice("unknown").unwrap_err();
        assert!(err.to_string().contains("unknown Kokoro voice"));
        assert!(err.to_string().contains("known"));
    }

    /// Requires a real Kokoro model directory, e.g.
    /// `models/tts/Kokoro-82M-v1.0-ONNX`, passed via `CRANE_KOKORO_DIR`.
    #[test]
    #[ignore]
    fn new_loads_real_model() {
        let dir = std::env::var("CRANE_KOKORO_DIR")
            .expect("set CRANE_KOKORO_DIR to a real Kokoro model directory");
        let model = Model::new(&dir, &Device::Cpu, &DType::F32).unwrap();
        assert_eq!(model.sample_rate(), 24_000);
        assert!(!model.available_voices().is_empty());
        assert!(model.vocab().contains_key(&'$'));
        assert_eq!(model.pad_token_id, 0);

        let voice = model.voice("af_heart").unwrap();
        assert_eq!(voice.dims(), &[510, 256]);
    }

    #[test]
    fn phonemes_to_ids_basic() {
        let model = test_model_with_vocab(small_vocab());
        assert_eq!(model.phonemes_to_ids("ab"), vec![0, 1, 2, 0]);
    }

    #[test]
    fn phonemes_to_ids_skips_unknown_codepoints() {
        let model = test_model_with_vocab(small_vocab());
        // 'z' is not in the vocab, so it's dropped rather than erroring.
        assert_eq!(model.phonemes_to_ids("azb"), vec![0, 1, 2, 0]);
    }

    #[test]
    fn phonemes_to_ids_empty_input() {
        let model = test_model_with_vocab(small_vocab());
        assert_eq!(model.phonemes_to_ids(""), vec![0, 0]);
    }

    #[test]
    fn chunk_phonemes_short_input_is_one_chunk() {
        assert_eq!(chunk_phonemes("hello world", 510), vec!["hello world".to_string()]);
    }

    #[test]
    fn chunk_phonemes_trims_and_drops_empty() {
        assert_eq!(chunk_phonemes("  hello  ", 510), vec!["hello".to_string()]);
        assert!(chunk_phonemes("", 510).is_empty());
        assert!(chunk_phonemes("   ", 510).is_empty());
    }

    #[test]
    fn chunk_phonemes_splits_long_input_at_space_boundary() {
        // 6 codepoints per word, 5 words = 30 codepoints, split into
        // chunks of at most 10 codepoints (a word plus a space is exactly
        // 7 codepoints, so each chunk should hold exactly one word).
        let words: Vec<&str> = std::iter::repeat_n("abcdef", 5).collect();
        let phonemes = words.join(" ");
        let chunks = chunk_phonemes(&phonemes, 10);
        assert_eq!(chunks, vec!["abcdef".to_string(); 5]);
    }

    #[test]
    fn chunk_phonemes_never_exceeds_max_cp() {
        let phonemes = "a".repeat(25);
        let chunks = chunk_phonemes(&phonemes, 10);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 10, "chunk {chunk:?} exceeds max_cp");
        }
        assert_eq!(chunks.concat().len(), 25);
    }

    #[test]
    fn resolve_language_auto_and_empty_default_to_en_us() {
        assert_eq!(resolve_language("auto"), "en_us");
        assert_eq!(resolve_language("AUTO"), "en_us");
        assert_eq!(resolve_language("Auto"), "en_us");
        assert_eq!(resolve_language(""), "en_us");
    }

    #[test]
    fn resolve_language_passes_through_explicit_language() {
        assert_eq!(resolve_language("en_us"), "en_us");
        assert_eq!(resolve_language("de"), "de");
    }

    #[test]
    fn resolve_language_maps_iso_code_to_language_identifier() {
        assert_eq!(resolve_language("en"), "en_us");
    }

    #[test]
    fn iso_code_to_language_known_and_passthrough() {
        assert_eq!(iso_code_to_language("en"), "en_us");
        // Unmapped codes (not yet supported languages) pass through
        // unchanged.
        assert_eq!(iso_code_to_language("de"), "de");
        // Already-resolved identifiers pass through unchanged too.
        assert_eq!(iso_code_to_language("en_us"), "en_us");
    }

    #[test]
    fn normalizer_for_builds_and_caches() {
        // Neither 'a' nor 'b' appears in any Kokoro replacement pattern, and
        // a non-`en`-prefixed language skips the rhotic-vowel pairs, so
        // "ab" should normalize to itself unchanged.
        let mut model = test_model_with_vocab(small_vocab());

        assert!(model.normalizers.is_empty());
        let normalized = model.normalizer_for("de_test").unwrap().normalize("ab");
        assert_eq!(normalized, "ab");
        assert_eq!(model.normalizers.len(), 1);

        // Second call for the same language reuses the cached normalizer.
        model.normalizer_for("de_test").unwrap();
        assert_eq!(model.normalizers.len(), 1);
    }

    /// Requires real Kokoro model assets (`CRANE_KOKORO_DIR`) and a real
    /// English G2P lexicon (`CRANE_G2P_EN_US_DIR`, no OOV model needed since
    /// "hello" and "world" are common lexicon entries).
    #[test]
    #[ignore = "needs CRANE_KOKORO_DIR and CRANE_G2P_EN_US_DIR"]
    fn generate_speech_real_model() {
        use crate::models::g2p::MoonshineG2p;
        use crate::models::g2p::languages::LanguageG2p;
        use crate::models::g2p::languages::english::EnglishG2p;

        let kokoro_dir = std::env::var("CRANE_KOKORO_DIR")
            .expect("set CRANE_KOKORO_DIR to a real Kokoro model directory");
        let g2p_dir = std::env::var("CRANE_G2P_EN_US_DIR")
            .expect("set CRANE_G2P_EN_US_DIR to an en_us G2P model directory");

        let dict_path = std::path::Path::new(&g2p_dir).join("dict_filtered_heteronyms.tsv");
        let dict_tsv = std::fs::read_to_string(&dict_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", dict_path.display()));
        let english = EnglishG2p::new(&dict_tsv, None, false).expect("build EnglishG2p");
        let mut phonemizer = MoonshineG2p::new();
        phonemizer.add_language(LanguageG2p::English(english));

        let mut model = Model::new(&kokoro_dir, &Device::Cpu, &DType::F32).unwrap();
        let (waveform, sample_rate) = model
            .generate_speech(
                "Hello world",
                "en_us",
                Some("af_heart"),
                &phonemizer,
                &SpeechOptions::default(),
            )
            .unwrap();

        assert_eq!(sample_rate, 24_000);
        let samples = waveform.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // "Hello world" should produce somewhere between 0.3s and 5s of
        // audio at 24 kHz -- a loose sanity bound, not an exact figure.
        assert!(
            samples.len() > 7_000 && samples.len() < 120_000,
            "unexpected sample count: {}",
            samples.len()
        );
        // `s != 0.0` alone would also be true for NaN (NaN compares unequal
        // to everything, including itself), so a fully-NaN waveform would
        // slip past that check alone -- require finite, non-zero samples.
        assert!(samples.iter().all(|s| s.is_finite()), "waveform contains NaN or infinite samples");
        assert!(samples.iter().any(|&s| s != 0.0), "waveform is all zeros");
    }

    /// Longer, punctuated multi-word input than [`generate_speech_real_model`]
    /// — regression test for a phase-computation NaN that only manifested on
    /// inputs producing enough STFT frames to hit an exact zero-magnitude
    /// bin (`0.0/0.0` feeding `Atan`). See
    /// `crate::onnx::optimizer::fuse_atan2`'s doc comment for the full
    /// story.
    #[test]
    #[ignore = "needs CRANE_KOKORO_DIR and CRANE_G2P_EN_US_DIR"]
    fn generate_speech_real_model_longer_sentence_is_not_nan() {
        use crate::models::g2p::MoonshineG2p;
        use crate::models::g2p::languages::LanguageG2p;
        use crate::models::g2p::languages::english::EnglishG2p;

        let kokoro_dir = std::env::var("CRANE_KOKORO_DIR")
            .expect("set CRANE_KOKORO_DIR to a real Kokoro model directory");
        let g2p_dir = std::env::var("CRANE_G2P_EN_US_DIR")
            .expect("set CRANE_G2P_EN_US_DIR to an en_us G2P model directory");

        let dict_path = std::path::Path::new(&g2p_dir).join("dict_filtered_heteronyms.tsv");
        let dict_tsv = std::fs::read_to_string(&dict_path).unwrap();
        let english = EnglishG2p::new(&dict_tsv, None, false).unwrap();
        let mut phonemizer = MoonshineG2p::new();
        phonemizer.add_language(LanguageG2p::English(english));

        let mut model = Model::new(&kokoro_dir, &Device::Cpu, &DType::F32).unwrap();
        let (waveform, sample_rate) = model
            .generate_speech(
                "Hello world, how are you?",
                "en_us",
                Some("af_heart"),
                &phonemizer,
                &SpeechOptions::default(),
            )
            .unwrap();

        assert_eq!(sample_rate, 24_000);
        let samples = waveform.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(samples.iter().all(|s| s.is_finite()), "waveform contains NaN or infinite samples");
        assert!(samples.iter().any(|&s| s != 0.0), "waveform is all zeros");
    }
}
