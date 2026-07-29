// SPDX-License-Identifier: MIT

//! English Kokoro IPA replacement tables and normalizer construction.
//!
//! Ported from Moonshine's `apply_diphthong_map()`: multi-codepoint IPA
//! sequences the G2P engine produces are collapsed into the single-codepoint
//! tokens Kokoro's phoneme vocabulary expects.

use std::collections::HashMap;

use anyhow::Result;

use crate::models::g2p::ipa_postprocess::IpaNormalizer;

/// IPA replacement pairs shared by every Kokoro language.
///
/// Affricate ligatures (with and without the IPA tie bar `U+0361`, since a
/// G2P engine may emit either form) and diphthong-to-letter aliases. Does
/// not include the two rhotic vowel expansions in
/// [`EN_KOKORO_REPLACEMENTS`] — non-English G2P engines don't produce those
/// codepoints.
pub const BASE_KOKORO_REPLACEMENTS: &[(&str, &str)] = &[
    ("t\u{0361}ʃ", "ʧ"), // t + tie bar + ʃ  → U+02A7
    ("d\u{0361}ʒ", "ʤ"), // d + tie bar + ʒ  → U+02A4
    ("tʃ", "ʧ"),         // affricate without tie bar
    ("dʒ", "ʤ"),         // affricate without tie bar
    ("eɪ", "A"),         // FACE diphthong
    ("aɪ", "I"),         // PRICE diphthong
    ("aʊ", "W"),         // MOUTH diphthong
    ("oʊ", "O"),         // GOAT diphthong
    ("əʊ", "Q"),         // GOAT (reduced) diphthong
    ("ɔɪ", "Y"),         // CHOICE diphthong
];

/// IPA replacement pairs for English (US/GB) Kokoro normalization.
///
/// [`BASE_KOKORO_REPLACEMENTS`] plus two English-specific rhotic vowel
/// expansions: `ɝ` (stressed) → `ɜɹ` and `ɚ` (unstressed) → `əɹ`.
///
/// All entries are NFC-normalized (verified by the debug assertion in
/// [`IpaNormalizer::new`]).
pub const EN_KOKORO_REPLACEMENTS: &[(&str, &str)] = &[
    ("t\u{0361}ʃ", "ʧ"),
    ("d\u{0361}ʒ", "ʤ"),
    ("tʃ", "ʧ"),
    ("dʒ", "ʤ"),
    ("eɪ", "A"),
    ("aɪ", "I"),
    ("aʊ", "W"),
    ("oʊ", "O"),
    ("əʊ", "Q"),
    ("ɔɪ", "Y"),
    ("ɝ", "ɜɹ"), // rhotic stressed vowel (English only)
    ("ɚ", "əɹ"), // rhotic unstressed vowel (English only)
];

/// Builds an [`IpaNormalizer`] for the Kokoro vocoder in the given language.
///
/// `"en_us"` uses [`EN_KOKORO_REPLACEMENTS`] (includes the rhotic vowel
/// expansions); every other language uses [`BASE_KOKORO_REPLACEMENTS`].
/// `vocab` is Kokoro's phoneme vocabulary (from `tokenizer.json`) — its keys
/// become the accepted codepoint set, and any codepoint outside it is
/// dropped rather than coerced (Kokoro uses an empty coercion pool).
///
/// # Errors
///
/// Returns an error if the replacement table fails to compile into an
/// Aho-Corasick automaton.
#[allow(clippy::implicit_hasher)]
pub fn build_kokoro_normalizer(
    language: &str,
    vocab: &HashMap<char, i64>,
) -> Result<IpaNormalizer> {
    let replacements: &[(&str, &str)] = match language {
        "en_us" => EN_KOKORO_REPLACEMENTS,
        _ => BASE_KOKORO_REPLACEMENTS,
    };
    let vocab_chars: Vec<char> = vocab.keys().copied().collect();
    IpaNormalizer::new(replacements, vocab_chars, Vec::new())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use unicode_normalization::is_nfc;

    use super::*;

    /// Representative subset of Kokoro's 115-entry `tokenizer.json` vocab,
    /// sufficient to exercise `EN_KOKORO_REPLACEMENTS`. `g` (U+0067) is
    /// deliberately absent — Kokoro uses `ɡ` (U+0261) instead.
    fn en_test_vocab() -> HashMap<char, i64> {
        [
            ('$', 0),
            (';', 1),
            (':', 2),
            (',', 3),
            ('.', 4),
            ('!', 5),
            ('?', 6),
            (' ', 16),
            ('A', 24),
            ('I', 25),
            ('O', 31),
            ('Q', 33),
            ('W', 39),
            ('Y', 41),
            ('a', 43),
            ('b', 44),
            ('c', 45),
            ('d', 46),
            ('e', 47),
            ('f', 48),
            ('h', 50),
            ('i', 51),
            ('j', 52),
            ('k', 53),
            ('l', 54),
            ('m', 55),
            ('n', 56),
            ('o', 57),
            ('p', 58),
            ('r', 60),
            ('s', 61),
            ('t', 62),
            ('u', 63),
            ('v', 64),
            ('w', 65),
            ('z', 68),
            ('\u{0251}', 69),  // ɑ
            ('\u{00e6}', 72),  // æ
            ('\u{0254}', 76),  // ɔ
            ('\u{00f0}', 81),  // ð
            ('\u{02a4}', 82),  // ʤ
            ('\u{0259}', 83),  // ə
            ('\u{025b}', 86),  // ɛ
            ('\u{025c}', 87),  // ɜ
            ('\u{026a}', 102), // ɪ
            ('\u{014b}', 112), // ŋ
            ('\u{03b8}', 119), // θ
            ('\u{0279}', 123), // ɹ
            ('\u{0283}', 131), // ʃ
            ('\u{02a7}', 133), // ʧ
            ('\u{028a}', 135), // ʊ
            ('\u{028c}', 138), // ʌ
            ('\u{0292}', 147), // ʒ
            ('\u{0294}', 148), // ʔ
            ('\u{02c8}', 156), // ˈ
            ('\u{02cc}', 157), // ˌ
            ('\u{02d0}', 158), // ː
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn en_kokoro_replacements_are_nfc() {
        for (from, to) in EN_KOKORO_REPLACEMENTS {
            assert!(is_nfc(from), "from pattern {from:?} is not NFC");
            assert!(is_nfc(to), "replacement {to:?} is not NFC");
        }
    }

    #[test]
    fn en_kokoro_replacements_no_duplicates() {
        let mut seen = HashSet::new();
        for (from, _) in EN_KOKORO_REPLACEMENTS {
            assert!(seen.insert(*from), "duplicate from pattern: {from:?}");
        }
    }

    #[test]
    fn base_kokoro_is_prefix_of_en_kokoro() {
        assert_eq!(
            BASE_KOKORO_REPLACEMENTS,
            &EN_KOKORO_REPLACEMENTS[..BASE_KOKORO_REPLACEMENTS.len()]
        );
    }

    #[test]
    fn en_kokoro_replaces_tie_bar_affricate() {
        let vocab = en_test_vocab();
        let norm = build_kokoro_normalizer("en_us", &vocab).unwrap();
        assert_eq!(norm.normalize("t\u{0361}ʃ"), "ʧ");
        assert_eq!(norm.normalize("d\u{0361}ʒ"), "ʤ");
    }

    #[test]
    fn en_kokoro_replaces_no_tie_bar_affricate() {
        let vocab = en_test_vocab();
        let norm = build_kokoro_normalizer("en_us", &vocab).unwrap();
        assert_eq!(norm.normalize("tʃ"), "ʧ");
        assert_eq!(norm.normalize("dʒ"), "ʤ");
    }

    #[test]
    fn en_kokoro_replaces_diphthongs() {
        let vocab = en_test_vocab();
        let norm = build_kokoro_normalizer("en_us", &vocab).unwrap();
        assert_eq!(norm.normalize("eɪ"), "A");
        assert_eq!(norm.normalize("aɪ"), "I");
        assert_eq!(norm.normalize("aʊ"), "W");
        assert_eq!(norm.normalize("oʊ"), "O");
        assert_eq!(norm.normalize("əʊ"), "Q");
        assert_eq!(norm.normalize("ɔɪ"), "Y");
    }

    #[test]
    fn en_kokoro_expands_rhotic_vowels() {
        let vocab = en_test_vocab();
        let norm = build_kokoro_normalizer("en_us", &vocab).unwrap();
        assert_eq!(norm.normalize("ɝ"), "ɜɹ");
        assert_eq!(norm.normalize("ɚ"), "əɹ");
    }

    #[test]
    fn non_english_kokoro_skips_rhotic_pairs() {
        let vocab = en_test_vocab();
        let norm = build_kokoro_normalizer("fr_fr", &vocab).unwrap();
        // ɝ is not in the replacement table for non-English and not in the
        // vocab, so with an empty coerce pool it gets dropped.
        assert_eq!(norm.normalize("ɝ"), "");
        // Shared diphthong replacements still apply.
        assert_eq!(norm.normalize("eɪ"), "A");
    }

    #[test]
    fn en_kokoro_drops_unknown_codepoints() {
        let vocab = en_test_vocab();
        let norm = build_kokoro_normalizer("en_us", &vocab).unwrap();
        // 'g' (U+0067) is not in the Kokoro vocab (Kokoro uses ɡ U+0261).
        assert_eq!(norm.normalize("gab"), "ab");
    }

    #[test]
    fn en_kokoro_full_word_normalization() {
        let vocab = en_test_vocab();
        let norm = build_kokoro_normalizer("en_us", &vocab).unwrap();
        // "teacher": tˈiːtʃɚ -> tˈiːʧəɹ after affricate + rhotic replacement.
        assert_eq!(norm.normalize("tˈiːtʃɚ"), "tˈiːʧəɹ");
    }
}
