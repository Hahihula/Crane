// SPDX-License-Identifier: MIT

//! Grapheme-to-phoneme phonemization: shared trait, per-language dispatch,
//! lexicon storage, text normalization, and benchmarking utilities.

use std::collections::HashMap;

use anyhow::{Result, bail};

use languages::{LanguageG2p, SUPPORTED_LANGUAGES};

pub mod benchmark;
pub mod ipa_postprocess;
pub mod languages;
pub mod lexicon;
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
}
