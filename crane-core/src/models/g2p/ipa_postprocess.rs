// SPDX-License-Identifier: MIT

//! Single-pass IPA postprocessing: pattern replacement, vocab filtering, and
//! unknown-codepoint coercion, compiled once per (language, vocoder) pair.
//!
//! Replaces the reference implementation's 15-50 sequential full-string
//! passes with a fixed number of passes regardless of table size: one NFC
//! pass, one Aho-Corasick multi-pattern replacement pass, and one combined
//! vocab-filter/whitespace-collapse/coercion pass.

use std::collections::HashSet;

use aho_corasick::{AhoCorasick, MatchKind};
use anyhow::{Context, Result};
use unicode_normalization::{UnicodeNormalization, is_nfc};

/// Compiled IPA normalizer for a specific (language, vocoder) pair.
///
/// Built once at model load time from that pair's replacement table, vocab,
/// and (for Piper) unknown-codepoint coercion pool. `normalize()` then runs
/// in a fixed number of passes independent of table size.
pub struct IpaNormalizer {
    /// Compiled multi-pattern automaton, leftmost-longest matching so that
    /// e.g. `"eɪ"` matches before a lone `"e"` would. Replacement is a
    /// single non-cascading pass over the original text: a pattern is
    /// matched against the input only, never against another pattern's
    /// replacement output. A replacement table ported from a reference
    /// implementation that relies on one rule's output being caught by a
    /// later rule will not reproduce that behavior here.
    replacer: AhoCorasick,
    /// Replacement string for each pattern, indexed by the automaton's
    /// pattern ID.
    replacements: Vec<String>,
    /// Sorted, deduped codepoints the target vocoder accepts; checked via
    /// binary search. Anything else is coerced via `coerce_pool` or dropped.
    vocab: Vec<char>,
    /// Sorted, deduped pool of codepoints to coerce unknown output into via
    /// nearest-neighbor (by raw Unicode scalar value, not phonetic
    /// similarity) binary search. Empty means "drop unknown codepoints"
    /// (the Kokoro case; Piper populates this from its `phoneme_id_map`).
    coerce_pool: Vec<char>,
}

impl IpaNormalizer {
    /// Builds a normalizer from a replacement table, an accepted vocab, and
    /// an optional coercion pool.
    ///
    /// `replacements` pairs are compiled into a single Aho-Corasick
    /// automaton at construction time — this is the one-time cost that lets
    /// `normalize()` do a single scan per call. `vocab` and `coerce_pool`
    /// are sorted and deduped here so `normalize()` can binary-search them.
    ///
    /// Every `from`/`to` string in `replacements` must already be in NFC
    /// form: `normalize()` runs NFC once, up front, and never again, so a
    /// decomposed pattern will not match the NFC'd input and a decomposed
    /// replacement will leak non-NFC output downstream. Debug builds assert
    /// this and that no two `from` patterns are identical.
    ///
    /// # Errors
    ///
    /// Returns an error if `replacements` contains a pattern the
    /// Aho-Corasick builder rejects (e.g. an empty pattern).
    pub fn new(
        replacements: &[(&str, &str)],
        mut vocab: Vec<char>,
        mut coerce_pool: Vec<char>,
    ) -> Result<Self> {
        debug_assert!(
            replacements
                .iter()
                .all(|(from, to)| is_nfc(from) && is_nfc(to)),
            "IPA replacement patterns and replacements must be NFC-normalized"
        );
        debug_assert!(
            {
                let mut seen = HashSet::new();
                replacements.iter().all(|(from, _)| seen.insert(*from))
            },
            "IPA replacement table must not contain duplicate patterns"
        );

        let patterns: Vec<&str> = replacements.iter().map(|(from, _)| *from).collect();
        let replacer = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .context("IPA replacement patterns must compile into a valid automaton")?;

        vocab.sort_unstable();
        vocab.dedup();
        coerce_pool.sort_unstable();
        coerce_pool.dedup();

        Ok(Self {
            replacer,
            replacements: replacements
                .iter()
                .map(|(_, to)| (*to).to_string())
                .collect(),
            vocab,
            coerce_pool,
        })
    }

    /// Normalizes raw IPA into the target vocoder's phoneme inventory.
    ///
    /// Three passes: NFC (via a lazy iterator, no intermediate allocation),
    /// a single Aho-Corasick replacement scan, then a combined vocab-filter,
    /// whitespace-collapse, and coercion pass.
    ///
    /// Every out-of-vocab codepoint is coerced via `closest_codepoint`,
    /// including zero-width combining marks (e.g. IPA diacritics): a mark
    /// with no vocab entry is replaced by whichever spacing character is
    /// numerically nearest rather than being dropped, which can insert a
    /// stray visible character where an invisible diacritic was expected.
    #[must_use]
    pub fn normalize(&self, ipa: &str) -> String {
        let nfc: String = ipa.nfc().collect();
        let replaced = self.replacer.replace_all(&nfc, &self.replacements);

        let mut out = String::with_capacity(replaced.len());
        let mut pending_space = false;
        for c in replaced.chars() {
            if c.is_whitespace() {
                pending_space = !out.is_empty();
                continue;
            }

            let kept = if self.vocab.binary_search(&c).is_ok() {
                Some(c)
            } else {
                self.closest_codepoint(c)
            };

            let Some(kept) = kept else {
                continue;
            };

            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(kept);
        }

        out
    }

    /// Finds the codepoint in `coerce_pool` nearest to `c` by raw Unicode
    /// scalar value (not phonetic or visual similarity) via binary search —
    /// a deterministic best-effort fallback, not a phonetically informed
    /// one. Returns `None` if the pool is empty.
    fn closest_codepoint(&self, c: char) -> Option<char> {
        if self.coerce_pool.is_empty() {
            return None;
        }

        let idx = self.coerce_pool.partition_point(|&pool_c| pool_c < c);

        let after = self.coerce_pool.get(idx).copied();
        let before = idx
            .checked_sub(1)
            .and_then(|i| self.coerce_pool.get(i).copied());

        match (before, after) {
            (Some(b), Some(a)) => {
                if (c as u32).abs_diff(b as u32) <= (a as u32).abs_diff(c as u32) {
                    Some(b)
                } else {
                    Some(a)
                }
            },
            (Some(b), None) => Some(b),
            (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn ipa_normalizer_is_send_and_sync() {
        assert_send_sync::<IpaNormalizer>();
    }

    #[test]
    fn basic_replacement() {
        let normalizer =
            IpaNormalizer::new(&[("a", "x"), ("b", "y")], vec!['x', 'y'], Vec::new()).unwrap();
        assert_eq!(normalizer.normalize("ab"), "xy");
    }

    #[test]
    fn leftmost_longest_prefers_longer_pattern() {
        let normalizer =
            IpaNormalizer::new(&[("ab", "X"), ("a", "Y")], vec!['X', 'Y', 'b'], Vec::new())
                .unwrap();
        assert_eq!(normalizer.normalize("ab"), "X");
    }

    #[test]
    fn vocab_filter_drops_unknown_codepoints() {
        let normalizer = IpaNormalizer::new(&[], vec!['a', 'b', 'c'], Vec::new()).unwrap();
        assert_eq!(normalizer.normalize("abd c"), "ab c");
    }

    #[test]
    fn whitespace_runs_collapse_to_single_space() {
        let normalizer = IpaNormalizer::new(&[], vec!['a', 'b'], Vec::new()).unwrap();
        assert_eq!(normalizer.normalize("  a   b  "), "a b");
    }

    #[test]
    fn nfc_normalizes_decomposed_input() {
        let composed = "\u{e9}"; // é
        let decomposed = "e\u{301}"; // e + combining acute accent
        let normalizer =
            IpaNormalizer::new(&[], vec![composed.chars().next().unwrap()], Vec::new()).unwrap();
        assert_eq!(normalizer.normalize(decomposed), composed);
    }

    #[test]
    fn coerce_picks_nearest_pool_codepoint() {
        // Pool contains 'a' (0x61) and 'e' (0x65); 'c' (0x63) is 2 away from
        // both, so the tie-break (<=) should pick 'a' (before).
        let normalizer = IpaNormalizer::new(&[], vec!['a', 'e'], vec!['a', 'e']).unwrap();
        assert_eq!(normalizer.normalize("c"), "a");

        // 'd' (0x64) is 1 away from 'e' and 3 away from 'a' -> nearest is 'e'.
        assert_eq!(normalizer.normalize("d"), "e");
    }

    #[test]
    fn coerce_with_empty_pool_drops_unknown() {
        let normalizer = IpaNormalizer::new(&[], vec!['a'], Vec::new()).unwrap();
        assert_eq!(normalizer.normalize("ab"), "a");
    }

    #[test]
    fn empty_input_returns_empty_string() {
        let normalizer = IpaNormalizer::new(&[], Vec::new(), Vec::new()).unwrap();
        assert_eq!(normalizer.normalize(""), "");
    }

    #[test]
    fn no_replacements_passes_through_vocab_filter() {
        let normalizer = IpaNormalizer::new(&[], vec!['h', 'e', 'l', 'o'], Vec::new()).unwrap();
        assert_eq!(normalizer.normalize("hello"), "hello");
    }
}
