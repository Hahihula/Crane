// SPDX-License-Identifier: MIT

//! Kokoro TTS `Model`: config/vocab parsing, voice loading, and the model
//! skeleton.
//!
//! The ONNX forward pass itself is wired up in a later step; this step
//! loads the ONNX graph, the phoneme vocabulary, discovers available voice
//! names, and lazily loads/caches per-voice style embeddings from their
//! `.bin` files.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

/// Kokoro always outputs mono PCM at 24 kHz.
const KOKORO_SAMPLE_RATE: u32 = 24_000;

/// Style embedding dimension. Not present in any shipped config file —
/// inferred from voice tensor shape (1020 × 128 float32 per `.bin` file)
/// and hardcoded here.
const KOKORO_STYLE_DIM: usize = 128;

/// The `model_type` value every known Kokoro ONNX export's `config.json`
/// carries. `config.json` otherwise has no other keys to validate against.
const EXPECTED_MODEL_TYPE: &str = "style_text_to_speech_2";

/// Voice loaded eagerly at construction to fail fast on a broken model
/// directory. Falls back to the first discovered voice if this one isn't
/// present.
const DEFAULT_VOICE: &str = "af_heart";

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

/// Kokoro-82M ONNX text-to-speech model.
///
/// Loaded once via [`Model::new`] and reused across `generate_speech()`
/// calls — the ONNX graph, phoneme vocab, and voice list are all immutable
/// after construction. Voice style embeddings are loaded lazily on first use
/// via [`Model::voice`] and cached for the lifetime of the model.
pub struct Model {
    /// Loaded ONNX model graph, run via `crate::onnx::simple_eval()`.
    ///
    /// Not yet read: the forward pass is wired up in a later step.
    #[allow(dead_code)]
    onnx_graph: crate::onnx::proto::ModelProto,
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
    ///
    /// Not yet read: used to bound the phoneme ID buffer in a later step.
    #[allow(dead_code)]
    max_seq_len: usize,
    /// Token ID for the pad/BOS/EOS token `$`, read from `vocab`.
    ///
    /// Not yet read: used when building phoneme ID sequences in a later
    /// step.
    #[allow(dead_code)]
    pad_token_id: i64,
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
        let onnx_graph = crate::onnx::read_file(&onnx_path)
            .with_context(|| format!("loading Kokoro ONNX model from {}", onnx_path.display()))?;

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
            vocab,
            voices,
            voice_dir,
            available_voices,
            style_dim: KOKORO_STYLE_DIM,
            max_seq_len,
            pad_token_id,
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
            vocab: HashMap::new(),
            voices,
            voice_dir: voice_dir.to_path_buf(),
            available_voices,
            style_dim: 2,
            max_seq_len: 0,
            pad_token_id: 0,
        }
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
        assert_eq!(voice.dims(), &[1020, 128]);
    }
}
