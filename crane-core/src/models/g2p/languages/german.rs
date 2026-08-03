// SPDX-License-Identifier: MIT

//! German (`de`) grapheme-to-phoneme engine.
//!
//! Two tiers: case-cascading lexicon lookup, then hand-written
//! letter-to-sound rules as the final fallback. Only the lexicon-lookup
//! tier exists so far — a lookup miss currently contributes nothing to the
//! output, since there is no rule fallback wired in yet. `GermanG2p` is also
//! not yet reachable through [`LanguageG2p`](super::LanguageG2p) — a
//! `LanguageG2p::German` variant will be added once the rule fallback tier
//! lands.

use anyhow::Result;

use crate::models::g2p::lexicon::Lexicon;
use crate::models::g2p::text_normalize::{split_text_to_words, trim_edge_punctuation};

/// German grapheme-to-phoneme engine: case-cascading lexicon lookup, then
/// hand-written rule fallback (rules not yet wired in — see the module
/// docs).
///
/// Unlike English's engine, there is no out-of-vocabulary model tier for
/// German and therefore no interior mutability: the lexicon is immutable
/// after construction, so `text_to_ipa` needs no locking.
#[derive(Debug)]
pub struct GermanG2p {
    /// Word-to-IPA lexicon, built from a `word\tIPA` TSV at construction.
    /// Unlike English's all-lowercase lexicon, German's preserves the
    /// source dictionary's original casing — see [`lookup_cascade`].
    lexicon: Lexicon,
}

impl GermanG2p {
    /// Builds a German G2P engine from lexicon TSV content (`word\tIPA`
    /// lines, no header).
    ///
    /// # Errors
    ///
    /// Returns an error if `lexicon_tsv` is malformed — see
    /// [`Lexicon::from_tsv`].
    pub fn new(lexicon_tsv: &str) -> Result<Self> {
        Ok(Self { lexicon: Lexicon::from_tsv(lexicon_tsv)? })
    }

    /// Converts `text` to a space-joined IPA phoneme string.
    ///
    /// Each word has attached punctuation trimmed from its edges (case
    /// preserved — see [`trim_edge_punctuation`]) before being resolved via
    /// [`lookup_cascade`]. A word that normalizes to nothing (all
    /// punctuation) or that misses every case-cascade tier is skipped
    /// entirely for now, rather than erroring or inserting a placeholder —
    /// the hand-written rule fallback that will close the lookup-miss gap
    /// lands in a later step.
    ///
    /// # Errors
    ///
    /// Currently infallible; returns `Result` to satisfy the
    /// [`Phonemizer`](crate::models::g2p::Phonemizer) trait contract other
    /// language engines rely on.
    pub fn text_to_ipa(&self, text: &str) -> Result<String> {
        let mut ipa = String::with_capacity(text.len());
        for word in split_text_to_words(text) {
            let word = trim_edge_punctuation(word);
            if word.is_empty() {
                continue;
            }
            let Some(word_ipa) = lookup_cascade(&self.lexicon, word) else {
                continue;
            };
            if !ipa.is_empty() {
                ipa.push(' ');
            }
            ipa.push_str(word_ipa);
        }
        Ok(ipa)
    }
}

/// Looks up `word` in `lexicon`, trying progressively more aggressive case
/// transforms until one hits: the exact surface form, then title-case
/// (first codepoint uppercased, rest unchanged), then fully lowercased.
/// Returns `None` if all three miss.
///
/// The German lexicon is not uniformly lowercase like English's — many
/// nouns only have a capitalized entry, while some compound-internal forms
/// are only lowercase — so case-folding every lookup key (as English does)
/// would silently miss whichever form isn't present. The exact-form attempt
/// is zero-allocation; the fallback attempts allocate a transformed key
/// lazily, one at a time, since Unicode case mapping is not guaranteed to
/// be a fixed-length, allocation-free transform in general. A fallback tier
/// is skipped entirely (no allocation, no repeat lookup) when it would
/// reproduce a key already tried: title-casing is a no-op when `word`
/// already starts uppercase, and lowercasing is a no-op when `word` has no
/// uppercase characters at all.
fn lookup_cascade<'a>(lexicon: &'a Lexicon, word: &str) -> Option<&'a str> {
    if let Some(ipa) = lexicon.get(word) {
        return Some(ipa);
    }
    let starts_uppercase = word.chars().next().is_some_and(char::is_uppercase);
    // Kept as a nested `if` rather than a `&&`-chain so the two conditions
    // stay independently steppable/inspectable in a debugger.
    #[allow(clippy::collapsible_if)]
    if !starts_uppercase {
        if let Some(ipa) = lexicon.get(&title_case(word)) {
            return Some(ipa);
        }
    }
    if word.chars().any(char::is_uppercase) {
        return lexicon.get(&word.to_lowercase());
    }
    None
}

/// Uppercases the first codepoint of `word` and leaves the rest unchanged —
/// not a full "capitalize", which would also lowercase the remainder.
///
/// `char::to_uppercase` is not guaranteed 1:1 (e.g. `ß` uppercases to the
/// two-character `"SS"`), but German words never begin with `ß`, so that
/// expansion is not reachable here.
fn title_case(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_propagates_malformed_lexicon_error() {
        assert!(GermanG2p::new("no-tab-here\n").is_err());
    }

    #[test]
    fn lookup_cascade_exact_case_hit() {
        let lexicon = Lexicon::from_tsv("Haus\thaʊ̯s\n").unwrap();
        assert_eq!(lookup_cascade(&lexicon, "Haus"), Some("haʊ̯s"));
    }

    #[test]
    fn lookup_cascade_title_case_hit() {
        // Lexicon only has the capitalized entry; a lowercase surface form
        // misses the exact attempt but hits on the title-case fallback.
        let lexicon = Lexicon::from_tsv("Haus\thaʊ̯s\n").unwrap();
        assert_eq!(lookup_cascade(&lexicon, "haus"), Some("haʊ̯s"));
    }

    #[test]
    fn lookup_cascade_lowercase_hit() {
        // Lexicon only has the lowercase entry (e.g. a verb, never
        // capitalized in the dictionary); a sentence-initial capitalized
        // surface form misses exact and title-case (title-casing an
        // already-capitalized word is a no-op) but hits on lowercasing.
        let lexicon = Lexicon::from_tsv("laufen\tˈlaʊ̯fn̩\n").unwrap();
        assert_eq!(lookup_cascade(&lexicon, "Laufen"), Some("ˈlaʊ̯fn̩"));
    }

    #[test]
    fn lookup_cascade_miss_returns_none() {
        let lexicon = Lexicon::from_tsv("Haus\thaʊ̯s\n").unwrap();
        assert_eq!(lookup_cascade(&lexicon, "Fenster"), None);
    }

    #[test]
    fn text_to_ipa_single_word_hit() {
        let engine = GermanG2p::new("Haus\thaʊ̯s\n").unwrap();
        assert_eq!(engine.text_to_ipa("Haus").unwrap(), "haʊ̯s");
    }

    #[test]
    fn text_to_ipa_multi_word_input_joins_with_spaces() {
        let engine = GermanG2p::new("Haus\thaʊ̯s\nFenster\tˈfɛnstɐ\n").unwrap();
        assert_eq!(engine.text_to_ipa("Haus Fenster").unwrap(), "haʊ̯s ˈfɛnstɐ");
    }

    #[test]
    fn text_to_ipa_miss_word_is_skipped_without_double_space() {
        let engine = GermanG2p::new("Haus\thaʊ̯s\nFenster\tˈfɛnstɐ\n").unwrap();
        assert_eq!(
            engine.text_to_ipa("Haus Schublade Fenster").unwrap(),
            "haʊ̯s ˈfɛnstɐ"
        );
    }

    #[test]
    fn text_to_ipa_empty_input_returns_empty_string() {
        let engine = GermanG2p::new("Haus\thaʊ̯s\n").unwrap();
        assert_eq!(engine.text_to_ipa("").unwrap(), "");
    }

    #[test]
    fn text_to_ipa_trims_trailing_punctuation_before_lookup() {
        let engine = GermanG2p::new("Haus\thaʊ̯s\n").unwrap();
        assert_eq!(engine.text_to_ipa("Haus.").unwrap(), "haʊ̯s");
    }

    #[test]
    fn text_to_ipa_multi_word_with_punctuation_joins_with_spaces() {
        let engine = GermanG2p::new("Haus\thaʊ̯s\nFenster\tˈfɛnstɐ\n").unwrap();
        assert_eq!(
            engine.text_to_ipa("Haus, Fenster.").unwrap(),
            "haʊ̯s ˈfɛnstɐ"
        );
    }

    #[test]
    fn text_to_ipa_all_punctuation_word_is_skipped() {
        let engine = GermanG2p::new("Haus\thaʊ̯s\n").unwrap();
        assert_eq!(engine.text_to_ipa("Haus ---").unwrap(), "haʊ̯s");
    }
}
