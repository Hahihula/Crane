// SPDX-License-Identifier: MIT

//! Grapheme-to-phoneme phonemization: shared trait, per-language dispatch,
//! lexicon storage, text normalization, and benchmarking utilities.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, bail};

use languages::{LanguageG2p, SUPPORTED_LANGUAGES};
use languages::english::EnglishG2p;
use languages::german::GermanG2p;

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
    /// per-language layout: one subdirectory per language under `g2p_dir`,
    /// each with its own lexicon filename convention (English:
    /// `en_us/dict_filtered_heteronyms.tsv` plus an optional `en_us/oov/`
    /// model directory (`model.onnx` + `onnx-config.json`); German:
    /// `de/dict.tsv`, no OOV directory).
    ///
    /// Each language subdirectory is tried independently and skipped if its
    /// required lexicon is missing — a language simply isn't loaded if its
    /// assets aren't present, rather than failing the whole phonemizer. If a
    /// lexicon is present but malformed, a warning is printed to stderr and
    /// the language is still skipped, since that case likely indicates a
    /// broken asset bundle rather than an intentionally-absent language.
    /// English's OOV model is optional on top of that: if `en_us/oov/` is
    /// absent or fails to load, the engine silently falls back to lexicon +
    /// hand rules only, matching [`EnglishG2p::text_to_ipa`]'s own
    /// swallow-and-fall-through behavior for OOV inference errors.
    ///
    /// # Errors
    ///
    /// Returns an error if no language's assets could be loaded at all.
    pub fn from_g2p_dir(g2p_dir: &Path) -> Result<Self> {
        let mut phonemizer = Self::new();

        if let Some(english) = Self::load_english(g2p_dir) {
            phonemizer.add_language(LanguageG2p::English(Box::new(english)));
        }
        if let Some(german) = Self::load_german(g2p_dir) {
            phonemizer.add_language(LanguageG2p::German(german));
        }

        if phonemizer.languages.is_empty() {
            bail!("no G2P language assets found under {}", g2p_dir.display());
        }

        Ok(phonemizer)
    }

    /// Loads the `en_us` engine from `{g2p_dir}/en_us/`. Returns `None`
    /// silently if the lexicon file is missing; if it is present but fails
    /// to parse, prints a warning to stderr and returns `None`.
    fn load_english(g2p_dir: &Path) -> Option<EnglishG2p> {
        let en_us_dir = g2p_dir.join("en_us");
        let dict_path = en_us_dir.join("dict_filtered_heteronyms.tsv");
        let lexicon_tsv = std::fs::read_to_string(&dict_path).ok()?;

        let oov_dir = en_us_dir.join("oov");
        let oov_model = if oov_dir.is_dir() { oov_onnx::Model::load(&oov_dir).ok() } else { None };

        match EnglishG2p::new(&lexicon_tsv, oov_model, false) {
            Ok(engine) => Some(engine),
            Err(err) => {
                eprintln!("warning: failed to parse G2P lexicon at {}: {err}", dict_path.display());
                None
            }
        }
    }

    /// Loads the `de` engine from `{g2p_dir}/de/`. Returns `None` silently if
    /// the lexicon file is missing; if it is present but fails to parse,
    /// prints a warning to stderr and returns `None`.
    fn load_german(g2p_dir: &Path) -> Option<GermanG2p> {
        let dict_path = g2p_dir.join("de").join("dict.tsv");
        let lexicon_tsv = std::fs::read_to_string(&dict_path).ok()?;
        match GermanG2p::new(&lexicon_tsv) {
            Ok(engine) => Some(engine),
            Err(err) => {
                eprintln!("warning: failed to parse G2P lexicon at {}: {err}", dict_path.display());
                None
            }
        }
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
        phonemizer.add_language(LanguageG2p::English(Box::new(english)));

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

    // Verifies from_g2p_dir loads a German-only directory (de/dict.tsv, no
    // en_us/ subdirectory at all) and phonemizes via the resulting engine.
    #[test]
    fn from_g2p_dir_loads_german_only() {
        let dir = tempfile::tempdir().unwrap();
        let de_dir = dir.path().join("de");
        std::fs::create_dir_all(&de_dir).unwrap();
        std::fs::write(de_dir.join("dict.tsv"), "Haus\thaʊ̯s\n").unwrap();

        let phonemizer = MoonshineG2p::from_g2p_dir(dir.path()).unwrap();
        assert_eq!(phonemizer.text_to_ipa("Haus", "de").unwrap(), "haʊ̯s");
        assert!(phonemizer.text_to_ipa("hello", "en_us").is_err());
    }

    // Verifies from_g2p_dir loads both languages independently when both
    // subdirectories are present.
    #[test]
    fn from_g2p_dir_loads_both_languages() {
        let dir = tempfile::tempdir().unwrap();
        let en_us_dir = dir.path().join("en_us");
        std::fs::create_dir_all(&en_us_dir).unwrap();
        std::fs::write(en_us_dir.join("dict_filtered_heteronyms.tsv"), "hello\thəlˈoʊ\n").unwrap();
        let de_dir = dir.path().join("de");
        std::fs::create_dir_all(&de_dir).unwrap();
        std::fs::write(de_dir.join("dict.tsv"), "Haus\thaʊ̯s\n").unwrap();

        let phonemizer = MoonshineG2p::from_g2p_dir(dir.path()).unwrap();
        assert_eq!(phonemizer.text_to_ipa("hello", "en_us").unwrap(), "həlˈoʊ");
        assert_eq!(phonemizer.text_to_ipa("Haus", "de").unwrap(), "haʊ̯s");
    }

    // Verifies from_g2p_dir errors only when zero languages load — an empty
    // directory (no en_us/ or de/ subdirectory at all) is the hard-error case.
    #[test]
    fn from_g2p_dir_errors_when_no_language_loads() {
        let dir = tempfile::tempdir().unwrap();
        assert!(MoonshineG2p::from_g2p_dir(dir.path()).is_err());
    }

    // Verifies a present-but-malformed en_us lexicon is skipped (not a hard
    // error) as long as another language loads successfully.
    #[test]
    fn from_g2p_dir_skips_malformed_english_lexicon() {
        let dir = tempfile::tempdir().unwrap();
        let en_us_dir = dir.path().join("en_us");
        std::fs::create_dir_all(&en_us_dir).unwrap();
        std::fs::write(en_us_dir.join("dict_filtered_heteronyms.tsv"), "no-tab-here\n").unwrap();
        let de_dir = dir.path().join("de");
        std::fs::create_dir_all(&de_dir).unwrap();
        std::fs::write(de_dir.join("dict.tsv"), "Haus\thaʊ̯s\n").unwrap();

        let phonemizer = MoonshineG2p::from_g2p_dir(dir.path()).unwrap();
        assert_eq!(phonemizer.text_to_ipa("Haus", "de").unwrap(), "haʊ̯s");
        assert!(phonemizer.text_to_ipa("hello", "en_us").is_err());
    }

    // Verifies a present-but-malformed de lexicon is skipped (not a hard
    // error) as long as another language loads successfully.
    #[test]
    fn from_g2p_dir_skips_malformed_german_lexicon() {
        let dir = tempfile::tempdir().unwrap();
        let en_us_dir = dir.path().join("en_us");
        std::fs::create_dir_all(&en_us_dir).unwrap();
        std::fs::write(en_us_dir.join("dict_filtered_heteronyms.tsv"), "hello\thəlˈoʊ\n").unwrap();
        let de_dir = dir.path().join("de");
        std::fs::create_dir_all(&de_dir).unwrap();
        std::fs::write(de_dir.join("dict.tsv"), "no-tab-here\n").unwrap();

        let phonemizer = MoonshineG2p::from_g2p_dir(dir.path()).unwrap();
        assert_eq!(phonemizer.text_to_ipa("hello", "en_us").unwrap(), "həlˈoʊ");
        assert!(phonemizer.text_to_ipa("Haus", "de").is_err());
    }
}
