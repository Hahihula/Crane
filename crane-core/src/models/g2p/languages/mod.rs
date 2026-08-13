// SPDX-License-Identifier: MIT

//! Per-language grapheme-to-phoneme engines, statically dispatched.
//!
//! [`LanguageG2p`] is a closed enum with one variant per supported language.
//! Using an enum instead of `Box<dyn LanguageG2p>` lets the compiler inline
//! lookup and rule logic on the hot `text_to_ipa` call path instead of going
//! through a vtable on every call.

use anyhow::Result;

use english::EnglishG2p;
use german::GermanG2p;

pub mod english;
mod english_numerals;
mod english_rules;
pub mod german;
mod german_compound;
mod german_numerals;
mod german_rules;

/// Language identifiers currently registered in [`LanguageG2p`].
///
/// Grows as languages are implemented, per the rollout order in the G2P
/// design: English, then German, then French.
pub const SUPPORTED_LANGUAGES: &[&str] = &["en_us", "de"];

/// One language's grapheme-to-phoneme engine.
///
/// Each variant wraps a per-language engine (lexicon lookup, rules, and
/// optionally an OOV fallback model). Variants are added incrementally as
/// each language is implemented.
///
/// `text_to_ipa` takes `&self`, but engines with an interior-mutable cache
/// (e.g. [`EnglishG2p`]'s OOV result cache) assume calls are driven from a
/// single thread at a time, matching Crane's existing single-thread-per-model
/// TTS serving pattern. `LanguageG2p` is `Send + Sync`, so it is safe to
/// share across threads, but callers wrapping it in `Arc` for concurrent
/// calls should expect lock contention on that cache rather than free
/// parallelism.
#[derive(Debug)]
pub enum LanguageG2p {
    /// English (`en_us`) engine. Boxed: `EnglishG2p` (lexicon + OOV model +
    /// LRU cache) is far larger than `GermanG2p`, and an unboxed variant
    /// would size every `LanguageG2p` value to the largest variant.
    English(Box<EnglishG2p>),
    /// German (`de`) engine.
    German(GermanG2p),
}

impl LanguageG2p {
    /// Converts `text` to an IPA phoneme string using this language's engine.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying language engine fails to
    /// phonemize the input.
    pub fn text_to_ipa(&self, text: &str) -> Result<String> {
        match self {
            Self::English(engine) => engine.text_to_ipa(text),
            Self::German(engine) => engine.text_to_ipa(text),
        }
    }

    /// Returns the language identifier this engine handles (e.g. `"en_us"`).
    #[must_use]
    pub fn language(&self) -> &'static str {
        match self {
            Self::English(_) => "en_us",
            Self::German(_) => "de",
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

    fn test_english_language() -> LanguageG2p {
        LanguageG2p::English(Box::new(EnglishG2p::new("hello\thəlˈoʊ\n", None, false).unwrap()))
    }

    fn test_german_language() -> LanguageG2p {
        LanguageG2p::German(GermanG2p::new("Haus\thaʊ̯s\n").unwrap())
    }

    #[test]
    fn english_language_identifier() {
        assert_eq!(test_english_language().language(), "en_us");
    }

    #[test]
    fn english_text_to_ipa_delegates_to_engine() {
        assert_eq!(test_english_language().text_to_ipa("hello").unwrap(), "həlˈoʊ");
    }

    #[test]
    fn german_language_identifier() {
        assert_eq!(test_german_language().language(), "de");
    }

    #[test]
    fn german_text_to_ipa_delegates_to_engine() {
        assert_eq!(test_german_language().text_to_ipa("Haus").unwrap(), "haʊ̯s");
    }

    #[test]
    fn supported_languages_lists_english_and_german() {
        assert_eq!(SUPPORTED_LANGUAGES, &["en_us", "de"]);
    }
}
