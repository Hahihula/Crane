// SPDX-License-Identifier: MIT

//! German (`de`) grapheme-to-phoneme engine.
//!
//! Three tiers: case-cascading lexicon lookup, then compound-word
//! decomposition (see [`german_compound`](super::german_compound)) for a
//! whole-word miss, then hand-written letter-to-sound rules (see
//! [`german_rules`](super::german_rules)) as the final fallback when neither
//! finds anything. `GermanG2p` is not yet reachable through
//! [`LanguageG2p`](super::LanguageG2p) — a `LanguageG2p::German` variant
//! lands in a later step.

use anyhow::Result;

use crate::models::g2p::lexicon::Lexicon;
use crate::models::g2p::numeral_expand::expand_numerals;
use crate::models::g2p::text_normalize::{split_text_to_words, trim_edge_punctuation};

use super::german_numerals::GermanNumerals;

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
    /// Digit runs are expanded to their German cardinal spelling (see
    /// [`expand_numerals`]/[`GermanNumerals`]) before word splitting, so
    /// downstream lookup never sees raw digits. Each resulting word has
    /// attached punctuation trimmed from its edges (case preserved — see
    /// [`trim_edge_punctuation`]) before being resolved via
    /// [`lookup_cascade`]; a lexicon miss tries
    /// [`decompose`](super::german_compound::decompose) (compound-word
    /// splitting), and only falls through to the
    /// [`hand_rules_ipa`](super::german_rules::hand_rules_ipa) rule engine
    /// if that also finds nothing. A word that normalizes to nothing (all
    /// punctuation) or that produces no IPA from any tier is skipped
    /// entirely, rather than erroring or inserting a placeholder.
    ///
    /// # Errors
    ///
    /// Currently infallible; returns `Result` to satisfy the
    /// [`Phonemizer`](crate::models::g2p::Phonemizer) trait contract other
    /// language engines rely on.
    pub fn text_to_ipa(&self, text: &str) -> Result<String> {
        let text = expand_numerals(text, &GermanNumerals);
        let text: &str = &text;

        let mut ipa = String::with_capacity(text.len());
        for word in split_text_to_words(text) {
            let word = trim_edge_punctuation(word);
            if word.is_empty() {
                continue;
            }
            let owned_ipa;
            let word_ipa = if let Some(ipa) = lookup_cascade(&self.lexicon, word) {
                ipa
            } else if let Some(compound_ipa) = super::german_compound::decompose(&self.lexicon, word) {
                owned_ipa = compound_ipa;
                owned_ipa.as_str()
            } else {
                owned_ipa = super::german_rules::hand_rules_ipa(word);
                if owned_ipa.is_empty() {
                    continue;
                }
                owned_ipa.as_str()
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
pub(super) fn lookup_cascade<'a>(lexicon: &'a Lexicon, word: &str) -> Option<&'a str> {
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
    fn text_to_ipa_miss_word_falls_through_to_hand_rules() {
        // "Schublade" isn't in the lexicon, so it now falls through to the
        // hand-rule engine instead of being silently skipped.
        let engine = GermanG2p::new("Haus\thaʊ̯s\nFenster\tˈfɛnstɐ\n").unwrap();
        let ipa = engine.text_to_ipa("Haus Schublade Fenster").unwrap();
        let words: Vec<&str> = ipa.split(' ').collect();
        assert_eq!(words.first(), Some(&"haʊ̯s"));
        assert_eq!(words.last(), Some(&"ˈfɛnstɐ"));
        assert!(ipa.contains('ʃ'), "Schublade should produce ʃ from sch: {ipa}");
    }

    #[test]
    fn text_to_ipa_miss_word_falls_through_to_compound_decomposition() {
        // "Handschuhfach" isn't in the lexicon whole, but decomposes into
        // "Hand" + "Schuhfach", both of which are — so it resolves via
        // compound decomposition rather than falling all the way through
        // to hand rules.
        let engine = GermanG2p::new("Hand\thant\nSchuhfach\tʃuːfax\n").unwrap();
        assert_eq!(engine.text_to_ipa("Handschuhfach").unwrap(), "hantʃuːfax");
    }

    #[test]
    fn text_to_ipa_expands_numerals_before_lexicon_lookup() {
        let tsv = "Ich\tʔɪç\nhabe\tˈhaːbə\neinundzwanzig\tˈaɪ̯nʔʊntˌt͡svanˌt͡sɪç\nKatzen\tˈkatsn̩\n";
        let engine = GermanG2p::new(tsv).unwrap();
        assert_eq!(
            engine.text_to_ipa("Ich habe 21 Katzen").unwrap(),
            "ʔɪç ˈhaːbə ˈaɪ̯nʔʊntˌt͡svanˌt͡sɪç ˈkatsn̩"
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

    // CER benchmark: measures the compound-decomposition and hand-rules
    // tiers' accuracy against the checked-in `REFERENCE_CER_DE` threshold.
    // Needs a real lexicon on disk, so this is `#[ignore]`d by default. Run
    // with:
    //
    // ```sh
    // CRANE_G2P_DE_DIR=/path/to/de \
    //   cargo test -p crane-core -- \
    //   models::g2p::languages::german::tests::cer_benchmark --ignored --nocapture
    // ```

    use crate::models::g2p::benchmark::{G2pBenchmarkResult, REFERENCE_CER_DE, benchmark_g2p};

    fn model_dir() -> std::path::PathBuf {
        std::env::var("CRANE_G2P_DE_DIR")
            .expect("set CRANE_G2P_DE_DIR to a German G2P model directory")
            .into()
    }

    fn parse_word_ipa_tsv(tsv: &str) -> Vec<(String, String)> {
        tsv.lines()
            .filter_map(|line| line.split_once('\t'))
            .map(|(word, ipa)| (word.to_string(), ipa.to_string()))
            .collect()
    }

    /// Builds a `GermanG2p` from the on-disk lexicon with the held-out test
    /// words removed and runs it over the test set, reporting the resulting
    /// CER.
    ///
    /// Test words are excluded from the lexicon before construction so the
    /// benchmark measures the compound-decomposition and hand-rules tiers
    /// rather than trivial lexicon hits — a lexicon hit on a held-out word
    /// would trivially match and mask any regression in the fallback tiers.
    fn run_cer_benchmark() -> G2pBenchmarkResult {
        let test_data_path =
            crate::test_data::get_test_data_file("g2p/de_de/test.tsv").expect("fetch test.tsv");
        let test_tsv = std::fs::read_to_string(&test_data_path).expect("read test.tsv");
        let test_pairs = parse_word_ipa_tsv(&test_tsv);
        let test_words: std::collections::HashSet<&str> =
            test_pairs.iter().map(|(word, _)| word.as_str()).collect();

        let dict_path = model_dir().join("dict.tsv");
        let dict_tsv = std::fs::read_to_string(&dict_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", dict_path.display()));
        let held_out_dict: String = dict_tsv
            .lines()
            .filter(|line| match line.split_once('\t') {
                Some((word, _)) => !test_words.contains(word),
                None => true,
            })
            .collect::<Vec<_>>()
            .join("\n");

        let engine = GermanG2p::new(&held_out_dict).expect("build GermanG2p");

        let predictions: Vec<(String, String)> = test_pairs
            .iter()
            .map(|(word, _)| (word.clone(), engine.text_to_ipa(word).expect("text_to_ipa")))
            .collect();
        let predictions_ref: Vec<(&str, &str)> = predictions
            .iter()
            .map(|(word, ipa)| (word.as_str(), ipa.as_str()))
            .collect();
        let test_pairs_ref: Vec<(&str, &str)> = test_pairs
            .iter()
            .map(|(word, ipa)| (word.as_str(), ipa.as_str()))
            .collect();

        let result = benchmark_g2p(&predictions_ref, &test_pairs_ref, "de");
        println!(
            "de CER: {:.4} ({} errors / {} words)",
            result.cer, result.total_errors, result.total_words
        );
        result
    }

    #[test]
    #[ignore = "needs a local G2P lexicon directory (CRANE_G2P_DE_DIR)"]
    fn cer_benchmark() {
        let result = run_cer_benchmark();
        assert!(
            result.cer <= REFERENCE_CER_DE,
            "CER {:.4} exceeds reference {REFERENCE_CER_DE:.4}",
            result.cer
        );
    }
}
