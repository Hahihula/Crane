// SPDX-License-Identifier: MIT

//! Per-language grapheme-to-phoneme engines, statically dispatched.
//!
//! [`LanguageG2p`] is a closed enum with one variant per supported language.
//! Using an enum instead of `Box<dyn LanguageG2p>` lets the compiler inline
//! lookup and rule logic on the hot `text_to_ipa` call path instead of going
//! through a vtable on every call.

use anyhow::{Result, bail};

/// Language identifiers currently registered in [`LanguageG2p`].
///
/// Grows as languages are implemented, per the rollout order in the G2P
/// design: English, then German, then French.
pub const SUPPORTED_LANGUAGES: &[&str] = &["en_us"];

/// One language's grapheme-to-phoneme engine.
///
/// Each variant wraps a per-language engine (lexicon lookup, rules, and
/// optionally an OOV fallback model). Variants are added incrementally as
/// each language is implemented.
#[derive(Debug)]
pub enum LanguageG2p {
    /// English (`en_us`). Placeholder variant — the lexicon/rules engine is
    /// not implemented yet; this establishes the enum shape for later steps.
    English,
}

impl LanguageG2p {
    /// Converts `text` to an IPA phoneme string using this language's engine.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying language engine is not yet
    /// implemented, or if it fails to phonemize the input.
    pub fn text_to_ipa(&self, text: &str) -> Result<String> {
        match self {
            Self::English => {
                bail!("English G2P engine is not yet implemented (input: {text:?})")
            }
        }
    }

    /// Returns the language identifier this engine handles (e.g. `"en_us"`).
    #[must_use]
    pub fn language(&self) -> &'static str {
        match self {
            Self::English => "en_us",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn language_g2p_is_send_sync() {
        assert_send_sync::<LanguageG2p>();
    }

    #[test]
    fn english_language_identifier() {
        assert_eq!(LanguageG2p::English.language(), "en_us");
    }

    #[test]
    fn english_text_to_ipa_not_yet_implemented() {
        assert!(LanguageG2p::English.text_to_ipa("hello").is_err());
    }

    #[test]
    fn supported_languages_lists_english() {
        assert_eq!(SUPPORTED_LANGUAGES, &["en_us"]);
    }
}
