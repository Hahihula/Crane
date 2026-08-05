// SPDX-License-Identifier: MIT

//! English and German Kokoro IPA replacement tables and normalizer
//! construction.
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
/// [`EN_EXTRA_KOKORO_REPLACEMENTS`] — non-English G2P engines don't produce
/// those codepoints.
pub const SHARED_KOKORO_REPLACEMENTS: &[(&str, &str)] = &[
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

/// Extra IPA replacement pairs for English Kokoro normalization.
///
/// Appended to [`SHARED_KOKORO_REPLACEMENTS`] for any language tag starting
/// with `en`: the two rhotic vowel expansions `ɝ` (stressed) → `ɜɹ` and `ɚ`
/// (unstressed) → `əɹ`.
///
/// All entries are NFC-normalized (verified by the debug assertion in
/// [`IpaNormalizer::new`]).
pub const EN_EXTRA_KOKORO_REPLACEMENTS: &[(&str, &str)] = &[
    ("ɝ", "ɜɹ"), // rhotic stressed vowel (English only)
    ("ɚ", "əɹ"), // rhotic unstressed vowel (English only)
];

/// Extra IPA replacement pairs for German Kokoro normalization.
///
/// Appended to [`SHARED_KOKORO_REPLACEMENTS`] when `language == "de"`.
/// Kokoro's vocab has dedicated single-token ligatures for the `/ts/` and
/// `/dz/` affricates and for a geminate sibilant, but German's G2P output
/// produces these as multi-codepoint sequences; the model was trained on
/// the single-token forms, so leaving them unreplaced renders as separate
/// English-like phonemes rather than the intended affricate. `ʏ` (near-close
/// near-front rounded vowel) has no slot in Kokoro's vocab at all — `y` is
/// the exact substitution the voice's own training data used.
///
/// The `ts`/`dz` pairs are listed in both tie-bar (as produced directly by
/// the lexicon) and plain (as produced by the hand-rule fallback) forms,
/// since either could reach the normalizer depending on which G2P tier
/// produced the word. **Must stay German-only, never shared**: English's
/// `ts` (e.g. "cats") is a genuine two-phoneme /t/+/s/ sequence, not an
/// affricate — the opposite of German's `/ts/`.
///
/// All entries are NFC-normalized (verified by the debug assertion in
/// [`IpaNormalizer::new`]).
pub const DE_EXTRA_KOKORO_REPLACEMENTS: &[(&str, &str)] = &[
    ("t\u{0361}s", "ʦ"), // voiceless alveolar affricate, tie-bar form
    ("d\u{0361}z", "ʣ"), // voiced alveolar affricate, tie-bar form
    ("ts", "ʦ"),         // voiceless alveolar affricate, plain form
    ("dz", "ʣ"),         // voiced alveolar affricate, plain form
    ("ss", "S"),         // geminate sibilant
    ("ʏ", "y"),          // near-close near-front rounded vowel, not in Kokoro's vocab
];

/// Builds an [`IpaNormalizer`] for the Kokoro vocoder in the given language.
///
/// Any language tag starting with `en` (`en_us`, `en_gb`, ...) additionally
/// gets [`EN_EXTRA_KOKORO_REPLACEMENTS`]'s rhotic vowel expansions, and
/// `"de"` gets [`DE_EXTRA_KOKORO_REPLACEMENTS`]'s affricate/geminate
/// ligatures and `ʏ` mapping, on top of [`SHARED_KOKORO_REPLACEMENTS`];
/// every other language uses [`SHARED_KOKORO_REPLACEMENTS`] alone. `vocab`
/// is Kokoro's phoneme vocabulary (from `tokenizer.json`) — its keys become
/// the accepted codepoint set, and any codepoint outside it is dropped
/// rather than coerced (Kokoro uses an empty coercion pool).
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
    let mut replacements = SHARED_KOKORO_REPLACEMENTS.to_vec();
    if language.starts_with("en") {
        replacements.extend_from_slice(EN_EXTRA_KOKORO_REPLACEMENTS);
    } else if language == "de" {
        replacements.extend_from_slice(DE_EXTRA_KOKORO_REPLACEMENTS);
    }
    let vocab_chars: Vec<char> = vocab.keys().copied().collect();
    IpaNormalizer::new(&replacements, vocab_chars, Vec::new())
}

/// German IPA vowel characters `reposition_stress_before_vowel` scans for.
///
/// Taken from the actual character inventory of `g2p/de_de/test.tsv` in the
/// `crane-local-ai/test-data` HuggingFace dataset (the Duden/CELEX-style
/// reference corpus `dict.tsv` is drawn from), plus the orthographic
/// umlauts `ä ö ü` defensively (Crane's own G2P rules convert these to
/// their IPA equivalents before this function ever sees them, but there's
/// no harm in recognizing them too).
fn is_stress_target_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'e'
            | 'i'
            | 'o'
            | 'u'
            | 'y'
            | 'ä'
            | 'ö'
            | 'ü'
            | 'ø'
            | 'œ'
            | 'ɐ'
            | 'ɑ'
            | 'ɒ'
            | 'ɔ'
            | 'ə'
            | 'ɛ'
            | 'ɨ'
            | 'ɪ'
            | 'ʊ'
            | 'ʏ'
    )
}

/// Repositions every primary (`ˈ`, U+02C8) and secondary (`ˌ`, U+02CC)
/// stress mark in `ipa` from immediately before its syllable's entire onset
/// consonant cluster to immediately before that syllable's vowel.
///
/// German-specific, called only for `language == "de"`: German's G2P
/// dictionary (`dict.tsv`, sourced from Moonshine-TTS, see `MOONSHINE_DE.md`
/// at the repo root for the upstream bug report) and its
/// `de_test.tsv`-benchmarked hand-rule fallback both place stress before the
/// onset cluster (e.g. `"klettern"` -> `"ˈklɛtɐn"`), a legitimate
/// dictionary convention in its own right but not the one the actual
/// fine-tuned Kokoro German checkpoint was trained on — Crane's own English
/// dictionary already stresses immediately before the vowel instead (e.g.
/// `"teacher"` -> `"tˈitʃɚ"`, never `"ˈtitʃɚ"`), matching standard
/// espeak-ng-derived phonemization. For an *internal* syllable this is only
/// a few characters' difference, but when the stressed syllable is
/// word-initial (very common in German), the onset-cluster convention
/// leaves a bare stress mark as the literal first phoneme of the word,
/// immediately adjacent to the preceding space/silence with no consonant in
/// between — a pattern the model most likely never saw in training,
/// producing an audible artifact right at that word's boundary.
///
/// Implemented as a dedicated scanning function rather than an
/// [`IpaNormalizer`] replacement-table entry: that normalizer only does
/// fixed-pattern substitution (see its doc comment in
/// `crate::models::g2p::ipa_postprocess`), but "move this mark to just
/// before the next vowel" is inherently context-scanning, not a fixed
/// find/replace pair.
///
/// Idempotent on already-correct input: if a stress mark is already
/// immediately followed by a vowel (zero-consonant onset), the scan finds
/// that vowel at the very next position and re-emits the mark and vowel in
/// the same order, unchanged — so this keeps working with no double-effect
/// if `dict.tsv` is ever corrected upstream to this same convention.
pub(crate) fn reposition_stress_before_vowel(ipa: &str) -> String {
    let chars: Vec<char> = ipa.chars().collect();
    let mut out = String::with_capacity(ipa.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == 'ˈ' || c == 'ˌ' {
            // Scan forward for this syllable's vowel, but never past a word
            // boundary or another stress mark — those mean this syllable's
            // onset produced no recognized vowel at all, so the mark is
            // left where it is rather than swallowing a following word or
            // syllable's phonemes.
            let mut j = i + 1;
            while j < chars.len()
                && !is_stress_target_vowel(chars[j])
                && chars[j] != ' '
                && chars[j] != 'ˈ'
                && chars[j] != 'ˌ'
            {
                j += 1;
            }
            if j < chars.len() && is_stress_target_vowel(chars[j]) {
                out.extend(&chars[i + 1..j]); // onset consonant(s), unchanged
                out.push(c); // stress mark, now after the onset
                out.push(chars[j]); // the vowel
                i = j + 1;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use unicode_normalization::is_nfc;

    use super::*;

    /// Representative subset of Kokoro's 115-entry `tokenizer.json` vocab,
    /// sufficient to exercise `EN_EXTRA_KOKORO_REPLACEMENTS`. `g` (U+0067)
    /// is deliberately absent — Kokoro uses `ɡ` (U+0261) instead.
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
    fn kokoro_replacements_are_nfc() {
        for (from, to) in SHARED_KOKORO_REPLACEMENTS
            .iter()
            .chain(EN_EXTRA_KOKORO_REPLACEMENTS)
            .chain(DE_EXTRA_KOKORO_REPLACEMENTS)
        {
            assert!(is_nfc(from), "from pattern {from:?} is not NFC");
            assert!(is_nfc(to), "replacement {to:?} is not NFC");
        }
    }

    #[test]
    fn kokoro_replacements_no_duplicates() {
        let mut seen = HashSet::new();
        for (from, _) in SHARED_KOKORO_REPLACEMENTS
            .iter()
            .chain(EN_EXTRA_KOKORO_REPLACEMENTS)
            .chain(DE_EXTRA_KOKORO_REPLACEMENTS)
        {
            assert!(seen.insert(*from), "duplicate from pattern: {from:?}");
        }
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

    /// Minimal vocab sufficient to exercise
    /// [`DE_EXTRA_KOKORO_REPLACEMENTS`]'s targeted unit tests below.
    fn de_test_vocab() -> HashMap<char, i64> {
        [
            ('a', 43),
            ('y', 67),
            ('\u{0053}', 35),  // S
            ('\u{02a3}', 18),  // ʣ
            ('\u{02a6}', 20),  // ʦ
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn de_kokoro_replaces_tie_bar_ts_affricate() {
        let vocab = de_test_vocab();
        let norm = build_kokoro_normalizer("de", &vocab).unwrap();
        assert_eq!(norm.normalize("t\u{0361}s"), "ʦ");
    }

    #[test]
    fn de_kokoro_replaces_plain_ts_affricate() {
        let vocab = de_test_vocab();
        let norm = build_kokoro_normalizer("de", &vocab).unwrap();
        assert_eq!(norm.normalize("ts"), "ʦ");
    }

    #[test]
    fn de_kokoro_replaces_dz_affricate() {
        let vocab = de_test_vocab();
        let norm = build_kokoro_normalizer("de", &vocab).unwrap();
        assert_eq!(norm.normalize("d\u{0361}z"), "ʣ");
        assert_eq!(norm.normalize("dz"), "ʣ");
    }

    #[test]
    fn de_kokoro_replaces_geminate_ss() {
        let vocab = de_test_vocab();
        let norm = build_kokoro_normalizer("de", &vocab).unwrap();
        assert_eq!(norm.normalize("ss"), "S");
    }

    #[test]
    fn de_kokoro_maps_near_close_rounded_vowel() {
        let vocab = de_test_vocab();
        let norm = build_kokoro_normalizer("de", &vocab).unwrap();
        assert_eq!(norm.normalize("ʏ"), "y");
    }

    #[test]
    fn de_kokoro_skips_english_rhotic_pairs() {
        let vocab = en_test_vocab();
        let norm = build_kokoro_normalizer("de", &vocab).unwrap();
        // ɝ is not in DE_EXTRA_KOKORO_REPLACEMENTS and not in the vocab, so
        // with an empty coerce pool it gets dropped.
        assert_eq!(norm.normalize("ɝ"), "");
        // Shared diphthong replacements still apply.
        assert_eq!(norm.normalize("eɪ"), "A");
    }

    /// Maps each character of normalized IPA to its Kokoro phoneme ID,
    /// mirroring Moonshine's `phoneme_str_to_input_ids()`: a `0` (pad/BOS)
    /// is prepended and appended, and characters missing from `vocab` are
    /// skipped rather than erroring (unreachable in practice here, since
    /// `normalize()` already dropped anything outside `vocab`).
    fn phoneme_str_to_input_ids(phonemes: &str, vocab: &HashMap<char, i64>) -> Vec<i64> {
        let mut ids = vec![0i64];
        ids.extend(phonemes.chars().filter_map(|c| vocab.get(&c).copied()));
        ids.push(0);
        ids
    }

    /// Full real Kokoro vocabulary (114 entries) copied from Moonshine's
    /// `kokoro/config.json`, used only by the reference-corpus test below —
    /// distinct from [`en_test_vocab`]'s 55-entry representative subset used
    /// by the targeted pattern tests above.
    fn full_en_kokoro_vocab() -> HashMap<char, i64> {
        crate::test_data::load_kokoro_vocab().unwrap()
    }

    /// Correctness benchmark: the (`en_us`, Kokoro) [`IpaNormalizer`] must
    /// reproduce Moonshine's `normalize_ipa_to_kokoro()` +
    /// `phoneme_str_to_input_ids()` output exactly, on a fixed corpus of
    /// realistic G2P output plus hand-crafted edge cases. See
    /// the `crane-local-ai/test-data` dataset's `g2p/README.md` for the
    /// corpus's provenance — expected output was generated by an
    /// independent Python reimplementation of the C++ pipeline, not by
    /// running the C++ itself.
    #[test]
    #[ignore = "needs crane-local-ai/test-data (CRANE_TEST_DATA_DIR or network)"]
    fn en_kokoro_normalizer_matches_reference_corpus() {
        let vocab = full_en_kokoro_vocab();
        let norm = build_kokoro_normalizer("en_us", &vocab).unwrap();

        let path =
            crate::test_data::get_test_data_file("g2p/en_us/kokoro_normalizer_ref.tsv").unwrap();
        let tsv = std::fs::read_to_string(&path).unwrap();
        let mut checked = 0;
        for line in tsv.lines() {
            let mut fields = line.splitn(3, '\t');
            let raw_ipa = fields.next().expect("missing raw_ipa field");
            let expected_normalized = fields.next().expect("missing expected_normalized field");
            let expected_ids_field = fields.next().expect("missing expected_ids field");
            let expected_ids: Vec<i64> = expected_ids_field
                .split(',')
                .map(|s| s.parse().expect("phoneme id must be a valid i64"))
                .collect();

            let normalized = norm.normalize(raw_ipa);
            assert_eq!(
                normalized, expected_normalized,
                "normalize() mismatch for raw IPA {raw_ipa:?}"
            );

            let ids = phoneme_str_to_input_ids(&normalized, &vocab);
            assert_eq!(ids, expected_ids, "phoneme ID mismatch for raw IPA {raw_ipa:?}");

            checked += 1;
        }

        assert_eq!(checked, 71, "expected exactly 71 corpus entries, found {checked}");
    }

    /// Correctness benchmark: the (`de`, Kokoro) [`IpaNormalizer`] must
    /// collapse German-specific affricate/geminate sequences and the `ʏ`
    /// gap correctly, on a corpus of realistic G2P output (drawn from
    /// `g2p/de_de/test.tsv` lexicon entries) plus hand-crafted edge cases. See
    /// the `crane-local-ai/test-data` dataset's `g2p/README.md` for the
    /// corpus's provenance — expected output was hand-derived from
    /// [`DE_EXTRA_KOKORO_REPLACEMENTS`] and [`SHARED_KOKORO_REPLACEMENTS`]
    /// plus the vocab-filter algorithm, not by running `normalize()` itself.
    #[test]
    #[ignore = "needs crane-local-ai/test-data (CRANE_TEST_DATA_DIR or network)"]
    fn de_kokoro_normalizer_matches_reference_corpus() {
        let vocab = full_en_kokoro_vocab();
        let norm = build_kokoro_normalizer("de", &vocab).unwrap();

        let path =
            crate::test_data::get_test_data_file("g2p/de_de/kokoro_normalizer_ref.tsv").unwrap();
        let tsv = std::fs::read_to_string(&path).unwrap();
        let mut checked = 0;
        for line in tsv.lines() {
            let mut fields = line.splitn(3, '\t');
            let raw_ipa = fields.next().expect("missing raw_ipa field");
            let expected_normalized = fields.next().expect("missing expected_normalized field");
            let expected_ids_field = fields.next().expect("missing expected_ids field");
            let expected_ids: Vec<i64> = expected_ids_field
                .split(',')
                .map(|s| s.parse().expect("phoneme id must be a valid i64"))
                .collect();

            let normalized = norm.normalize(raw_ipa);
            assert_eq!(
                normalized, expected_normalized,
                "normalize() mismatch for raw IPA {raw_ipa:?}"
            );

            let ids = phoneme_str_to_input_ids(&normalized, &vocab);
            assert_eq!(ids, expected_ids, "phoneme ID mismatch for raw IPA {raw_ipa:?}");

            checked += 1;
        }

        assert_eq!(checked, 32, "expected exactly 32 corpus entries, found {checked}");
    }

    #[test]
    fn reposition_stress_moves_mark_past_single_onset_consonant() {
        // The exact "wäre" case from the reported bug: a bare word-initial
        // stress mark before a single-consonant onset must move to sit
        // right before the vowel.
        assert_eq!(reposition_stress_before_vowel("ˈvɛːʁə"), "vˈɛːʁə");
    }

    #[test]
    fn reposition_stress_moves_mark_past_single_consonant_onset_servus() {
        // The exact "Servus" case: also exercises a vowel immediately
        // followed by a combining diacritic (the non-syllabic offglide
        // marker U+032F on "ɐ̯"), which must survive untouched after the
        // vowel it modifies.
        assert_eq!(reposition_stress_before_vowel("ˈseɐ̯vus"), "sˈeɐ̯vus");
    }

    #[test]
    fn reposition_stress_moves_mark_past_multi_consonant_onset() {
        // Multi-consonant onset, the case cited in the commit that
        // introduced the current before-onset convention ("klettern" ->
        // "ˈklɛtɐn").
        assert_eq!(reposition_stress_before_vowel("ˈklɛtɐn"), "klˈɛtɐn");
    }

    #[test]
    fn reposition_stress_is_noop_for_vowel_initial_syllable() {
        // An empty onset (the stressed syllable is vowel-initial) means
        // "before the onset" and "before the vowel" are the same position —
        // nothing to move.
        assert_eq!(reposition_stress_before_vowel("ˈapfl̩"), "ˈapfl̩");
    }

    #[test]
    fn reposition_stress_handles_secondary_stress() {
        // Both primary and secondary marks are repositioned independently.
        // From the real de_test.tsv entry for "3-wöchigen".
        assert_eq!(
            reposition_stress_before_vowel("ˈdʁaɪ̯ˌvœçɪɡn̩"),
            "dʁˈaɪ̯vˌœçɪɡn̩"
        );
    }

    #[test]
    fn reposition_stress_does_not_cross_word_boundary() {
        // A word with no stress mark ("wie") must pass through unchanged,
        // and a following word needing repositioning ("wäre") must still
        // get it — the scan must restart cleanly after the space rather
        // than reading ahead into or across it.
        assert_eq!(reposition_stress_before_vowel("viː ˈvɛːʁə"), "viː vˈɛːʁə");
    }

    #[test]
    fn reposition_stress_leaves_dangling_mark_unmoved_if_no_vowel_found() {
        // Defensive fallback: a stress mark followed only by consonants
        // until end-of-string should never happen in real G2P output, but
        // must not panic or corrupt the string either way.
        assert_eq!(reposition_stress_before_vowel("ˈkt"), "ˈkt");
    }

    #[test]
    fn reposition_stress_is_idempotent_on_already_correct_input() {
        // Forward-compatibility guarantee: if dict.tsv is ever corrected
        // upstream to already place stress immediately before the vowel
        // (zero consonants between mark and vowel), this function must
        // leave it unchanged rather than double-applying anything.
        assert_eq!(reposition_stress_before_vowel("sˈeɐ̯vus"), "sˈeɐ̯vus");
        assert_eq!(reposition_stress_before_vowel("vˈɛːʁə"), "vˈɛːʁə");
    }
}
