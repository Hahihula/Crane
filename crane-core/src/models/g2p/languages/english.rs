// SPDX-License-Identifier: MIT

//! English (`en_us`) grapheme-to-phoneme engine.
//!
//! Currently a single tier: lexicon lookup. Words missing from the lexicon
//! fall back to their normalized spelling as a placeholder — the OOV model
//! and hand-written rule fallbacks are added in later steps.

use anyhow::Result;

use crate::models::g2p::lexicon::Lexicon;
use crate::models::g2p::text_normalize::normalize_word_for_lookup;

/// English grapheme-to-phoneme engine: lexicon lookup only, for now.
#[derive(Debug)]
pub struct EnglishG2p {
    /// Word-to-IPA lexicon, built from a `word\tIPA` TSV at construction.
    lexicon: Lexicon,
}

impl EnglishG2p {
    /// Builds an English G2P engine from lexicon TSV content
    /// (`word\tIPA` lines, no header).
    ///
    /// # Errors
    ///
    /// Returns an error if `lexicon_tsv` is malformed — see
    /// [`Lexicon::from_tsv`].
    pub fn new(lexicon_tsv: &str) -> Result<Self> {
        Ok(Self {
            lexicon: Lexicon::from_tsv(lexicon_tsv)?,
        })
    }

    /// Converts `text` to a space-joined IPA phoneme string.
    ///
    /// Each word is normalized and looked up in the lexicon. Words not
    /// found in the lexicon pass through as their normalized spelling
    /// (placeholder until the OOV model and hand-written rules land).
    /// Tokens that normalize to nothing (all-punctuation) are skipped.
    ///
    /// # Errors
    ///
    /// Currently infallible; returns `Result` for forward compatibility
    /// with later tiers (OOV model inference) that can fail.
    pub fn text_to_ipa(&self, text: &str) -> Result<String> {
        let mut ipa = String::with_capacity(text.len());
        for token in text.split_ascii_whitespace() {
            let word = normalize_word_for_lookup(token);
            if word.is_empty() {
                continue;
            }
            if !ipa.is_empty() {
                ipa.push(' ');
            }
            match self.lexicon.get(&word) {
                Some(word_ipa) => ipa.push_str(word_ipa),
                None => ipa.push_str(&word),
            }
        }
        Ok(ipa)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_engine() -> EnglishG2p {
        let tsv = "hello\thəlˈoʊ\nworld\twˈɜɹld\n";
        EnglishG2p::new(tsv).unwrap()
    }

    #[test]
    fn lexicon_hit_returns_ipa() {
        let engine = test_engine();
        assert_eq!(engine.text_to_ipa("hello").unwrap(), "həlˈoʊ");
    }

    #[test]
    fn lexicon_miss_returns_normalized_word() {
        let engine = test_engine();
        assert_eq!(engine.text_to_ipa("xyzzy").unwrap(), "xyzzy");
    }

    #[test]
    fn multi_word_input_joins_with_spaces() {
        let engine = test_engine();
        assert_eq!(engine.text_to_ipa("Hello, world!").unwrap(), "həlˈoʊ wˈɜɹld");
    }

    #[test]
    fn empty_input_returns_empty_string() {
        let engine = test_engine();
        assert_eq!(engine.text_to_ipa("").unwrap(), "");
    }

    #[test]
    fn punctuation_only_input_returns_empty_string() {
        let engine = test_engine();
        assert_eq!(engine.text_to_ipa("---").unwrap(), "");
    }

    #[test]
    fn mixed_hit_and_miss_words() {
        let engine = test_engine();
        assert_eq!(engine.text_to_ipa("hello there").unwrap(), "həlˈoʊ there");
    }

    #[test]
    fn new_propagates_malformed_lexicon_error() {
        assert!(EnglishG2p::new("no-tab-here\n").is_err());
    }
}
