// SPDX-License-Identifier: MIT

//! G2P accuracy benchmark harness.
//!
//! Measures predicted IPA against reference IPA using character error rate
//! (CER): the Levenshtein edit distance between predicted and expected IPA
//! strings, divided by the expected string's length. This is independent of
//! any vocoder — it isolates G2P accuracy from downstream synthesis quality.

use std::collections::HashMap;

/// External reference CER for the held-out `en_us` test set
/// (`g2p/en_us/test.tsv` in the `crane-local-ai/test-data` HuggingFace dataset).
/// This is the regression threshold `en_us` CER must stay at or below once the lexicon/OOV/rules
/// tiers exist to benchmark against.
pub const REFERENCE_CER_EN_US: f64 = 0.2558;

/// Regression threshold for the held-out `de` test set
/// (`g2p/de_de/test.tsv` in the `crane-local-ai/test-data` HuggingFace
/// dataset): `GermanG2p`'s own measured CER
/// (lexicon + compound decomposition + hand rules, with those words excluded
/// from the lexicon so they exercise the fallback tiers) must stay at or
/// below this value. Originally recorded as the external
/// moonshine-tts `german-rule-g2p-cli.cpp` reference engine's rule-only CER
/// (0.4390) before `GermanG2p` existed to benchmark against; tightened to
/// Crane's own measured CER once it did, matching how `REFERENCE_CER_EN_US`
/// tracks `EnglishG2p`'s own pipeline rather than an external baseline.
pub const REFERENCE_CER_DE: f64 = 0.2830;

/// Per-word benchmark result.
pub struct WordResult {
    /// The input word.
    pub word: String,
    /// The reference (ground-truth) IPA transcription.
    pub expected_ipa: String,
    /// The phonemizer's predicted IPA transcription.
    pub predicted_ipa: String,
    /// Codepoint-level Levenshtein distance between predicted and expected.
    pub levenshtein: usize,
    /// Character error rate for this word: `levenshtein / expected.chars().count()`.
    pub cer: f64,
}

/// Aggregate G2P accuracy benchmark result for one language.
pub struct G2pBenchmarkResult {
    /// Language identifier the benchmark was run for (e.g. `"en_us"`).
    pub language: String,
    /// Total number of words evaluated.
    pub total_words: usize,
    /// Sum of per-word Levenshtein distances across all evaluated words.
    pub total_errors: usize,
    /// Overall character error rate, micro-averaged: total edit distance
    /// divided by total reference length across the whole corpus.
    pub cer: f64,
    /// Per-word results, in the order the test pairs were provided.
    pub per_word: Vec<WordResult>,
}

/// Computes the Levenshtein edit distance between two strings, operating on
/// Unicode codepoints rather than bytes.
///
/// IPA strings use multi-byte UTF-8 characters (e.g. `ˈ`, `ʃ`, `ŋ`), so a
/// byte-wise distance would overcount edits. Uses the standard
/// Wagner-Fischer dynamic program with a single-row buffer.
#[must_use]
pub fn levenshtein_codepoints(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();

    // Iterate over the shorter string in the inner loop to keep the buffer small.
    let (short, long) = if a.len() <= b.len() { (&a, &b) } else { (&b, &a) };

    let mut prev_row: Vec<usize> = (0..=short.len()).collect();
    let mut curr_row = vec![0usize; short.len() + 1];

    for (i, &long_ch) in long.iter().enumerate() {
        curr_row[0] = i + 1;
        for (j, &short_ch) in short.iter().enumerate() {
            let cost = usize::from(long_ch != short_ch);
            curr_row[j + 1] = (prev_row[j + 1] + 1)
                .min(curr_row[j] + 1)
                .min(prev_row[j] + cost);
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[short.len()]
}

/// Runs a direct G2P accuracy benchmark: compares `predictions` against
/// `test_pairs` (the reference IPA) and computes per-word and aggregate CER.
///
/// `predictions` and `test_pairs` are both `(word, ipa)` pairs. A word
/// present in `test_pairs` but missing from `predictions` is scored as a
/// full miss (Levenshtein distance equal to the reference length, CER of
/// 1.0), so a partial phonemizer can still be benchmarked.
#[must_use]
pub fn benchmark_g2p(
    predictions: &[(&str, &str)],
    test_pairs: &[(&str, &str)],
    language: &str,
) -> G2pBenchmarkResult {
    let predictions_by_word: HashMap<&str, &str> = predictions.iter().copied().collect();

    let mut total_errors = 0usize;
    let mut total_reference_len = 0usize;
    let mut per_word = Vec::with_capacity(test_pairs.len());

    for &(word, expected_ipa) in test_pairs {
        let predicted_ipa = predictions_by_word.get(word).copied().unwrap_or("");

        let levenshtein = levenshtein_codepoints(predicted_ipa, expected_ipa);
        let reference_len = expected_ipa.chars().count();
        // IPA word lengths and edit distances are tiny (well under 2^53), so
        // the usize -> f64 conversion below is exact.
        #[allow(clippy::cast_precision_loss)]
        let cer = levenshtein as f64 / reference_len.max(1) as f64;

        total_errors += levenshtein;
        total_reference_len += reference_len;

        per_word.push(WordResult {
            word: word.to_string(),
            expected_ipa: expected_ipa.to_string(),
            predicted_ipa: predicted_ipa.to_string(),
            levenshtein,
            cer,
        });
    }

    // Corpus-wide totals stay well under 2^53, so this conversion is exact.
    #[allow(clippy::cast_precision_loss)]
    let cer = total_errors as f64 / total_reference_len.max(1) as f64;

    G2pBenchmarkResult {
        language: language.to_string(),
        total_words: test_pairs.len(),
        total_errors,
        cer,
        per_word,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_identical_strings_is_zero() {
        assert_eq!(levenshtein_codepoints("ˈɑkən", "ˈɑkən"), 0);
    }

    #[test]
    fn levenshtein_empty_vs_nonempty_equals_length() {
        assert_eq!(levenshtein_codepoints("", "ˈɑkən"), 5);
        assert_eq!(levenshtein_codepoints("ˈɑkən", ""), 5);
    }

    #[test]
    fn levenshtein_both_empty_is_zero() {
        assert_eq!(levenshtein_codepoints("", ""), 0);
    }

    #[test]
    fn levenshtein_single_substitution() {
        assert_eq!(levenshtein_codepoints("kat", "kot"), 1);
    }

    #[test]
    fn levenshtein_counts_codepoints_not_bytes() {
        // Every char here is a multi-byte UTF-8 IPA codepoint; a byte-wise
        // distance would wildly overcount.
        assert_eq!(levenshtein_codepoints("ˈæbɪlsən", "ˈæbɪlsən"), 0);
        assert_eq!(levenshtein_codepoints("ˈæbɪlsən", "ˈæbɪlsɛn"), 1);
    }

    #[test]
    fn levenshtein_is_symmetric() {
        assert_eq!(
            levenshtein_codepoints("əbˈændənz", "æbdˈʌkʃən"),
            levenshtein_codepoints("æbdˈʌkʃən", "əbˈændənz")
        );
    }

    #[test]
    fn benchmark_perfect_match_has_zero_cer() {
        let pairs = [("read", "riːd")];
        let result = benchmark_g2p(&pairs, &pairs, "en_us");

        assert_eq!(result.total_words, 1);
        assert_eq!(result.total_errors, 0);
        assert!((result.cer - 0.0).abs() < f64::EPSILON);
        assert_eq!(result.per_word[0].levenshtein, 0);
        assert!((result.per_word[0].cer - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn benchmark_missing_prediction_is_full_miss() {
        let predictions: [(&str, &str); 0] = [];
        let test_pairs = [("read", "riːd")];
        let result = benchmark_g2p(&predictions, &test_pairs, "en_us");

        assert_eq!(result.per_word[0].predicted_ipa, "");
        assert_eq!(result.per_word[0].levenshtein, 4);
        assert!((result.per_word[0].cer - 1.0).abs() < f64::EPSILON);
        assert!((result.cer - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn benchmark_partial_mismatch_computes_expected_cer() {
        let predictions = [("read", "rɛd")];
        let test_pairs = [("read", "riːd")];
        let result = benchmark_g2p(&predictions, &test_pairs, "en_us");

        // "rɛd" vs "riːd": r-matches, ɛ vs i (sub), missing ː (del), d matches = 2 edits.
        let expected_distance = levenshtein_codepoints("rɛd", "riːd");
        assert_eq!(result.per_word[0].levenshtein, expected_distance);
        // expected_distance is a tiny Levenshtein distance (well under 2^53),
        // so this conversion is exact.
        #[allow(clippy::cast_precision_loss)]
        let expected_cer = expected_distance as f64 / 4.0;
        assert!((result.per_word[0].cer - expected_cer).abs() < f64::EPSILON);
    }

    #[test]
    fn benchmark_aggregates_cer_across_words() {
        let predictions = [("aachen", "ˈɑkən"), ("aase", "ˈɑt")];
        let test_pairs = [("aachen", "ˈɑkən"), ("aase", "ˈɑs")];
        let result = benchmark_g2p(&predictions, &test_pairs, "en_us");

        assert_eq!(result.total_words, 2);
        // "aachen" is a perfect match (0 errors), "aase" has 1 substitution.
        assert_eq!(result.total_errors, 1);
        // Aggregate CER = total_errors / total_reference_len = 1 / (5 + 3).
        let expected_cer = 1.0 / 8.0;
        assert!((result.cer - expected_cer).abs() < f64::EPSILON);
    }
}
