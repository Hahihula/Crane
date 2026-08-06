// SPDX-License-Identifier: MIT

//! FST-backed word-to-IPA lexicon for the G2P engine.
//!
//! Compiles a TSV dictionary (`word\tIPA` lines) into a finite-state
//! transducer ([`fst::Map`]) with a contiguous IPA byte buffer. Lookups are
//! zero-allocation and return `&str` slices borrowed from the buffer.
//!
//! Words with multiple pronunciations (heteronyms, e.g. "read") keep every
//! distinct IPA alternative rather than collapsing to one: alternatives are
//! sorted lexicographically and packed into one buffer span per word,
//! separated by `\0` (IPA transcriptions never contain a NUL byte). [`get`](Lexicon::get)
//! returns the lexicographically-first alternative; [`get_all`](Lexicon::get_all)
//! returns all of them for callers that need dialect-aware disambiguation.

use anyhow::{Result, bail};

/// Number of bits used to encode the IPA buffer offset in a packed value.
const OFFSET_BITS: u32 = 40;
/// Number of bits used to encode the IPA string length in a packed value.
const LENGTH_BITS: u32 = 24;
/// Bitmask covering the low [`LENGTH_BITS`] bits of a packed value.
const LENGTH_MASK: u64 = (1 << LENGTH_BITS) - 1;
/// Largest offset representable in [`OFFSET_BITS`] bits.
const MAX_OFFSET: usize = (1 << OFFSET_BITS) - 1;
/// Largest length representable in [`LENGTH_BITS`] bits.
const MAX_LENGTH: usize = (1 << LENGTH_BITS) - 1;

/// Packs an `(offset, length)` pair into a single `u64` FST value.
///
/// The high [`OFFSET_BITS`] bits hold the offset, the low [`LENGTH_BITS`]
/// bits hold the length. Callers must ensure both values fit within their
/// allotted bit widths (checked by [`Lexicon::from_tsv`] before calling).
fn pack(offset: usize, length: usize) -> u64 {
    ((offset as u64) << LENGTH_BITS) | (length as u64)
}

/// Unpacks a `u64` FST value into an `(offset, length)` pair.
fn unpack(value: u64) -> (usize, usize) {
    let offset = value >> LENGTH_BITS;
    let length = value & LENGTH_MASK;
    // fst values are always produced by `pack`, which only ever encodes
    // values that originated as `usize` and fit within OFFSET_BITS/LENGTH_BITS,
    // so widening back to usize here is lossless on all supported platforms.
    #[allow(clippy::cast_possible_truncation)]
    let unpacked = (offset as usize, length as usize);
    unpacked
}

/// Compiled word-to-IPA lexicon backed by a finite-state transducer.
///
/// Stores words as FST keys mapping to packed `(offset, length)` values that
/// point into a contiguous IPA byte buffer. Constructed from a TSV file at
/// model load time. `fst::Map<Vec<u8>>` and `String` are both `Send + Sync`,
/// so `Lexicon` is safe to share via `Arc` across TTS worker threads without
/// any locking on the read path.
pub struct Lexicon {
    /// The compiled finite-state transducer mapping words to packed values.
    fst: fst::Map<Vec<u8>>,
    /// Concatenated IPA strings for all entries; individual entries are
    /// recovered by slicing at the `(offset, length)` stored in the FST.
    ipa_buffer: String,
}

impl std::fmt::Debug for Lexicon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lexicon")
            .field("len", &self.fst.len())
            .finish_non_exhaustive()
    }
}

impl Lexicon {
    /// Builds a lexicon from TSV content (`word\tIPA` lines, no header).
    ///
    /// Input does not need to be pre-sorted; entries are sorted
    /// lexicographically by word before the FST is built, since `fst`
    /// requires keys inserted in strictly increasing order. Words with
    /// multiple distinct IPA transcriptions (heteronyms) keep every
    /// alternative, stored in lexicographic order — see the module docs.
    ///
    /// # Errors
    ///
    /// Returns an error if any non-empty line is missing its tab separator,
    /// or if the input is larger than the packed value format can address.
    pub fn from_tsv(tsv: &str) -> Result<Self> {
        let mut entries: Vec<(&str, &str)> = Vec::new();
        for (line_num, line) in tsv.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let Some((word, ipa)) = line.split_once('\t') else {
                bail!("line {}: missing tab separator: {line:?}", line_num + 1);
            };
            entries.push((word, ipa));
        }

        // Sort by (word, ipa) so each word's alternatives are adjacent and
        // lexicographically ordered, then drop exact (word, ipa) duplicates.
        entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()).then(a.1.cmp(b.1)));
        entries.dedup();

        let capacity = entries.iter().map(|(_, ipa)| ipa.len() + 1).sum();
        let mut ipa_buffer = String::with_capacity(capacity);
        let mut builder = fst::MapBuilder::memory();

        for word_entries in entries.chunk_by(|a, b| a.0 == b.0) {
            let word = word_entries[0].0;
            let offset = ipa_buffer.len();
            if offset > MAX_OFFSET {
                bail!("IPA buffer offset exceeds {OFFSET_BITS}-bit capacity ({offset} bytes)");
            }
            for (i, &(_, ipa)) in word_entries.iter().enumerate() {
                if i > 0 {
                    ipa_buffer.push('\0');
                }
                ipa_buffer.push_str(ipa);
            }
            let length = ipa_buffer.len() - offset;
            if length > MAX_LENGTH {
                bail!("IPA string length exceeds {LENGTH_BITS}-bit capacity ({length} bytes)");
            }
            builder.insert(word.as_bytes(), pack(offset, length))?;
        }

        let fst = builder.into_map();
        Ok(Self { fst, ipa_buffer })
    }

    /// Looks up the IPA transcription for `word`.
    ///
    /// For a heteronym with multiple pronunciations, returns the
    /// lexicographically-first alternative — see [`get_all`](Self::get_all)
    /// for dialect-aware disambiguation among all alternatives. Returns
    /// `None` if the word is not in the lexicon. The returned `&str` borrows
    /// from the lexicon's internal buffer — no allocation.
    ///
    /// # Panics
    ///
    /// Panics if the internal FST contains a value that points outside the
    /// IPA buffer. This cannot happen when the lexicon is built via
    /// [`from_tsv`](Self::from_tsv).
    #[must_use]
    pub fn get(&self, word: &str) -> Option<&str> {
        let mut alts = self.get_all(word)?;
        // `from_tsv` never produces an empty alternative span for a word
        // that has an FST entry, so this always yields at least one item.
        alts.next()
    }

    /// Returns all IPA alternatives for `word`, in lexicographic order.
    ///
    /// Most words have exactly one alternative; heteronyms (e.g. "read")
    /// have two or more. Returns `None` if the word is not in the lexicon.
    ///
    /// # Panics
    ///
    /// Panics if the internal FST contains a value that points outside the
    /// IPA buffer. This cannot happen when the lexicon is built via
    /// [`from_tsv`](Self::from_tsv).
    #[must_use]
    pub fn get_all(&self, word: &str) -> Option<impl Iterator<Item = &str>> {
        let packed = self.fst.get(word.as_bytes())?;
        let (offset, length) = unpack(packed);
        Some(self.ipa_buffer[offset..offset + length].split('\0'))
    }

    /// Returns the number of entries in the lexicon.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fst.len()
    }

    /// Returns `true` if the lexicon has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fst.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn lexicon_is_send_sync() {
        assert_send_sync::<Lexicon>();
    }

    #[test]
    fn from_tsv_and_lookup_known_words() {
        let tsv = "aachen\tˈɑkən\naase\tˈɑs\nabandons\təbˈændənz\n\
                   abelson\tˈæbɪlsən\nabzug\tˈæbzˌʌɡ\n";
        let lexicon = Lexicon::from_tsv(tsv).unwrap();
        assert_eq!(lexicon.get("aachen"), Some("ˈɑkən"));
        assert_eq!(lexicon.get("aase"), Some("ˈɑs"));
        assert_eq!(lexicon.get("abandons"), Some("əbˈændənz"));
        assert_eq!(lexicon.get("abelson"), Some("ˈæbɪlsən"));
        assert_eq!(lexicon.get("abzug"), Some("ˈæbzˌʌɡ"));
    }

    #[test]
    fn lookup_missing_word_returns_none() {
        let lexicon = Lexicon::from_tsv("aachen\tˈɑkən\n").unwrap();
        assert_eq!(lexicon.get("nonexistent"), None);
    }

    #[test]
    fn heteronym_entries_stored_and_returned() {
        let tsv = "read\trˈɛd\nread\trˈid\n";
        let lexicon = Lexicon::from_tsv(tsv).unwrap();
        // get() returns the lexicographically-first alternative.
        assert_eq!(lexicon.get("read"), Some("rˈid"));
        assert_eq!(lexicon.len(), 1);
        // get_all() returns every alternative, sorted.
        let alts: Vec<&str> = lexicon.get_all("read").unwrap().collect();
        assert_eq!(alts, vec!["rˈid", "rˈɛd"]);
    }

    #[test]
    fn heteronym_with_three_alternatives() {
        let tsv = "minute\tmˈɪnət\nminute\tmaɪnjˈut\nminute\tmaɪnˈut\n";
        let lexicon = Lexicon::from_tsv(tsv).unwrap();
        let alts: Vec<&str> = lexicon.get_all("minute").unwrap().collect();
        assert_eq!(alts, vec!["maɪnjˈut", "maɪnˈut", "mˈɪnət"]);
        assert_eq!(lexicon.get("minute"), Some("maɪnjˈut"));
    }

    #[test]
    fn identical_duplicate_ipa_lines_collapse_to_one_alternative() {
        let tsv = "hello\thəlˈoʊ\nhello\thəlˈoʊ\n";
        let lexicon = Lexicon::from_tsv(tsv).unwrap();
        let alts: Vec<&str> = lexicon.get_all("hello").unwrap().collect();
        assert_eq!(alts, vec!["həlˈoʊ"]);
    }

    #[test]
    fn get_all_returns_none_for_missing_word() {
        let lexicon = Lexicon::from_tsv("aachen\tˈɑkən\n").unwrap();
        assert!(lexicon.get_all("nonexistent").is_none());
    }

    #[test]
    fn unsorted_input_is_handled() {
        let tsv = "zebra\tzˈibɹə\naachen\tˈɑkən\nmango\tmˈæŋɡoʊ\n";
        let lexicon = Lexicon::from_tsv(tsv).unwrap();
        assert_eq!(lexicon.get("zebra"), Some("zˈibɹə"));
        assert_eq!(lexicon.get("aachen"), Some("ˈɑkən"));
        assert_eq!(lexicon.get("mango"), Some("mˈæŋɡoʊ"));
    }

    #[test]
    fn empty_tsv_produces_empty_lexicon() {
        let lexicon = Lexicon::from_tsv("").unwrap();
        assert_eq!(lexicon.len(), 0);
        assert!(lexicon.is_empty());
        assert_eq!(lexicon.get("anything"), None);
    }

    #[test]
    fn malformed_line_returns_error() {
        let tsv = "aachen\tˈɑkən\nmissing_tab_line\nabase\tˈeɪs\n";
        let err = Lexicon::from_tsv(tsv).unwrap_err();
        assert!(err.to_string().contains("line 2"));
    }

    #[test]
    fn len_and_is_empty() {
        let tsv = "aachen\tˈɑkən\naase\tˈɑs\n";
        let lexicon = Lexicon::from_tsv(tsv).unwrap();
        assert_eq!(lexicon.len(), 2);
        assert!(!lexicon.is_empty());
    }

    #[test]
    #[ignore = "needs crane-local-ai/test-data (CRANE_TEST_DATA_DIR or network)"]
    fn test_tsv_fixture_lookups() {
        let path = crate::test_data::get_test_data_file("g2p/en_us/test.tsv").unwrap();
        let tsv = std::fs::read_to_string(&path).unwrap();
        let lexicon = Lexicon::from_tsv(&tsv).unwrap();
        // 5000 lines, but a handful of words (e.g. "read"-style heteronyms)
        // appear twice with different IPA, so unique word entries are fewer
        // — each still keeps both alternatives, see below.
        assert_eq!(lexicon.len(), 4988);
        assert_eq!(lexicon.get("aachen"), Some("ˈɑkən"));
        assert_eq!(lexicon.get("zynda"), Some("zˈɪndə"));
        assert_eq!(lexicon.get("abandons"), Some("əbˈændənz"));
        assert_eq!(lexicon.get("not_in_lexicon"), None);
        // "tanzania" is one of the fixture's real heteronyms.
        let alts: Vec<&str> = lexicon.get_all("tanzania").unwrap().collect();
        assert_eq!(alts, vec!["tænzˈeɪniə", "tˌænzənˈiə"]);
    }
}
