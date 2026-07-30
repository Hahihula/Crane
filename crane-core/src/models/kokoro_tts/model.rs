// SPDX-License-Identifier: MIT

//! Kokoro TTS `Model`: config/vocab parsing and the model skeleton.
//!
//! Voice loading (`.bin` style embeddings) is added in a later step; this
//! step only loads the ONNX graph, the phoneme vocabulary, and discovers
//! available voice names without reading their contents.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

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

    parsed
        .model
        .vocab
        .into_iter()
        .map(|(key, id)| {
            let mut chars = key.chars();
            let c = chars
                .next()
                .with_context(|| "vocab key must not be empty".to_string())?;
            if chars.next().is_some() {
                bail!("vocab key {key:?} is not a single codepoint");
            }
            Ok((c, id))
        })
        .collect()
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
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("bin")
            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
        {
            voices.push(stem.to_string());
        }
    }
    voices.sort();
    Ok(voices)
}

/// Kokoro-82M ONNX text-to-speech model.
///
/// Loaded once via [`Model::new`] and reused across `generate_speech()`
/// calls — the ONNX graph, phoneme vocab, and voice list are all immutable
/// after construction. Voice style embeddings are loaded lazily on first use
/// (a later step); this struct only tracks their names and directory here.
pub struct Model {
    /// Loaded ONNX model graph, run via `crate::onnx::simple_eval()`.
    ///
    /// Not yet read: the forward pass is wired up in a later step.
    #[allow(dead_code)]
    onnx_graph: crate::onnx::proto::ModelProto,
    /// Phoneme character to token ID, from `tokenizer.json`'s `model.vocab`.
    vocab: HashMap<char, i64>,
    /// Lazily loaded voice style embeddings, keyed by voice name (e.g.
    /// `"af_heart"`).
    ///
    /// Not yet read: populated on first use starting in a later step.
    #[allow(dead_code)]
    voices: HashMap<String, Tensor>,
    /// Directory containing per-voice `.bin` style embedding files.
    ///
    /// Not yet read: used to lazily load a voice starting in a later step.
    #[allow(dead_code)]
    voice_dir: PathBuf,
    /// Voice names discovered at construction time (`.bin` filenames minus
    /// the extension), independent of whether they've been loaded yet.
    available_voices: Vec<String>,
    /// Style embedding dimension. See [`KOKORO_STYLE_DIM`].
    ///
    /// Not yet read: used to validate loaded voice tensor shape in a later
    /// step.
    #[allow(dead_code)]
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
    pub fn new(model_path: &str, _device: &Device, _dtype: &DType) -> Result<Self> {
        let root = Path::new(model_path);

        validate_config(&root.join("config.json"))?;
        let vocab = parse_vocab(&root.join("tokenizer.json"))?;
        let max_seq_len = parse_max_seq_len(&root.join("tokenizer_config.json"))?;

        let pad_token_id = *vocab
            .get(&'$')
            .with_context(|| "Kokoro vocab is missing the pad/BOS/EOS token '$'".to_string())?;

        let voice_dir = root.join("voices");
        let available_voices = discover_voices(&voice_dir)?;

        let onnx_path = root.join("onnx").join("model.onnx");
        let onnx_graph = crate::onnx::read_file(&onnx_path)
            .with_context(|| format!("loading Kokoro ONNX model from {}", onnx_path.display()))?;

        Ok(Self {
            onnx_graph,
            vocab,
            voices: HashMap::new(),
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
    }
}
