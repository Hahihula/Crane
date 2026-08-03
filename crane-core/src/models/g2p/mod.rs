// SPDX-License-Identifier: MIT

//! Grapheme-to-phoneme phonemization: shared trait, per-language dispatch,
//! lexicon storage, text normalization, and benchmarking utilities.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};

use languages::{LanguageG2p, SUPPORTED_LANGUAGES};
use languages::english::EnglishG2p;

pub mod benchmark;
pub mod ipa_postprocess;
pub mod languages;
pub mod lexicon;
pub mod numeral_expand;
pub mod oov_onnx;
pub mod text_normalize;

/// Converts text to phoneme representation for TTS models.
///
/// Each TTS model family (Kokoro, Piper, Qwen3-TTS, etc.) expects phonemes
/// in a specific format. Implementations handle:
/// 1. Text normalization (numbers, abbreviations, punctuation)
/// 2. Grapheme-to-phoneme conversion (lexicon lookup + rules/OOV model)
/// 3. Output formatting (IPA string, phoneme IDs, etc.)
pub trait Phonemizer: Send {
    /// Converts `text` to an IPA phoneme string for the given `language`.
    ///
    /// # Errors
    ///
    /// Returns an error if `language` is not supported, or if phonemization
    /// fails.
    fn text_to_ipa(&self, text: &str, language: &str) -> Result<String>;

    /// Languages this phonemizer implementation can support (e.g.
    /// `["en_us"]`), not necessarily the languages currently loaded — a
    /// language listed here can still fail `text_to_ipa` if it hasn't been
    /// loaded yet.
    fn supported_languages(&self) -> &[&str];
}

/// Moonshine-style grapheme-to-phoneme phonemizer.
///
/// Holds one [`LanguageG2p`] engine per language, constructed once at model
/// load time and reused across every `generate_speech()` call — never
/// re-created per request.
#[derive(Default)]
pub struct MoonshineG2p {
    /// Loaded per-language engines, keyed by language identifier (e.g. `"en_us"`).
    /// Keys are always drawn from [`SUPPORTED_LANGUAGES`], so `&'static str`
    /// avoids an owned-`String` allocation per loaded language.
    languages: HashMap<&'static str, LanguageG2p>,
}

impl MoonshineG2p {
    /// Creates a phonemizer with no languages loaded yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a language engine, keyed by [`LanguageG2p::language`].
    ///
    /// Replaces any engine previously registered for the same language,
    /// returning it.
    pub fn add_language(&mut self, engine: LanguageG2p) -> Option<LanguageG2p> {
        debug_assert!(
            SUPPORTED_LANGUAGES.contains(&engine.language()),
            "{:?} is not in SUPPORTED_LANGUAGES",
            engine.language()
        );
        self.languages.insert(engine.language(), engine)
    }

    /// Builds a phonemizer from a G2P asset directory, following Moonshine's
    /// per-language layout: `{g2p_dir}/<language>/dict_filtered_heteronyms.tsv`
    /// plus an optional `{g2p_dir}/<language>/oov/` model directory
    /// (`model.onnx` + `onnx-config.json`).
    ///
    /// Currently loads `en_us` only, per [`SUPPORTED_LANGUAGES`]; German and
    /// French will add their own subdirectory checks here in Phase 2. The
    /// OOV model is optional — if `{g2p_dir}/en_us/oov/` is absent or fails
    /// to load, the engine silently falls back to lexicon + hand rules only,
    /// matching [`EnglishG2p::text_to_ipa`]'s own swallow-and-fall-through
    /// behavior for OOV inference errors.
    ///
    /// # Errors
    ///
    /// Returns an error if `{g2p_dir}/en_us/dict_filtered_heteronyms.tsv` is
    /// missing or malformed — the lexicon, unlike the OOV model, is required.
    pub fn from_g2p_dir(g2p_dir: &Path) -> Result<Self> {
        let en_us_dir = g2p_dir.join("en_us");
        let dict_path = en_us_dir.join("dict_filtered_heteronyms.tsv");
        let lexicon_tsv = std::fs::read_to_string(&dict_path)
            .with_context(|| format!("reading G2P lexicon at {}", dict_path.display()))?;

        let oov_dir = en_us_dir.join("oov");
        let oov_model = if oov_dir.is_dir() { oov_onnx::Model::load(&oov_dir).ok() } else { None };

        let english = EnglishG2p::new(&lexicon_tsv, oov_model, false)?;
        let mut phonemizer = Self::new();
        phonemizer.add_language(LanguageG2p::English(english));
        Ok(phonemizer)
    }
}

impl Phonemizer for MoonshineG2p {
    fn text_to_ipa(&self, text: &str, language: &str) -> Result<String> {
        let Some(engine) = self.languages.get(language) else {
            bail!("unsupported language: {language:?}");
        };
        engine.text_to_ipa(text)
    }

    fn supported_languages(&self) -> &[&str] {
        SUPPORTED_LANGUAGES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn moonshine_g2p_is_send() {
        assert_send::<MoonshineG2p>();
    }

    #[test]
    fn new_phonemizer_has_no_languages_loaded() {
        let phonemizer = MoonshineG2p::new();
        assert!(phonemizer.text_to_ipa("hello", "en_us").is_err());
    }

    #[test]
    fn add_language_enables_text_to_ipa() {
        let mut phonemizer = MoonshineG2p::new();
        let english =
            languages::english::EnglishG2p::new("hello\thəlˈoʊ\n", None, false).unwrap();
        phonemizer.add_language(LanguageG2p::English(english));

        assert_eq!(phonemizer.text_to_ipa("hello", "en_us").unwrap(), "həlˈoʊ");
        assert!(phonemizer.text_to_ipa("hello", "de").is_err());
    }

    #[test]
    fn supported_languages_matches_registry() {
        let phonemizer = MoonshineG2p::new();
        assert_eq!(phonemizer.supported_languages(), SUPPORTED_LANGUAGES);
    }

    // Verifies from_g2p_dir loads a lexicon-only en_us directory (no oov/
    // subdirectory) and phonemizes via the resulting engine.
    #[test]
    fn from_g2p_dir_loads_lexicon_without_oov() {
        let dir = tempfile::tempdir().unwrap();
        let en_us_dir = dir.path().join("en_us");
        std::fs::create_dir_all(&en_us_dir).unwrap();
        std::fs::write(en_us_dir.join("dict_filtered_heteronyms.tsv"), "hello\thəlˈoʊ\n").unwrap();

        let phonemizer = MoonshineG2p::from_g2p_dir(dir.path()).unwrap();
        assert_eq!(phonemizer.text_to_ipa("hello", "en_us").unwrap(), "həlˈoʊ");
    }

    // Verifies a missing lexicon file is a hard error, since it's the one
    // required asset (unlike the optional OOV model).
    #[test]
    fn from_g2p_dir_errors_when_lexicon_missing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("en_us")).unwrap();
        assert!(MoonshineG2p::from_g2p_dir(dir.path()).is_err());
    }

    // Verifies a present-but-broken oov/ directory is skipped rather than
    // failing the whole load, matching the OOV-is-optional design.
    #[test]
    fn from_g2p_dir_skips_broken_oov_directory() {
        let dir = tempfile::tempdir().unwrap();
        let en_us_dir = dir.path().join("en_us");
        std::fs::create_dir_all(en_us_dir.join("oov")).unwrap();
        std::fs::write(en_us_dir.join("dict_filtered_heteronyms.tsv"), "hello\thəlˈoʊ\n").unwrap();

        let phonemizer = MoonshineG2p::from_g2p_dir(dir.path()).unwrap();
        assert_eq!(phonemizer.text_to_ipa("hello", "en_us").unwrap(), "həlˈoʊ");
    }
}
