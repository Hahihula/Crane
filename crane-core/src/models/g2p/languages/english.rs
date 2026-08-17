// SPDX-License-Identifier: MIT

//! English (`en_us`) grapheme-to-phoneme engine.
//!
//! Three tiers: lexicon lookup, then OOV ONNX model inference (when a model
//! is loaded and produces non-empty output), then hand-written
//! letter-to-sound rules as the final fallback.

use std::borrow::Cow;
use std::num::NonZeroUsize;
use std::sync::{Mutex, PoisonError};

use anyhow::Result;
use lru::LruCache;

use crate::models::g2p::lexicon::Lexicon;
use crate::models::g2p::numeral_expand::expand_numerals;
use crate::models::g2p::oov_onnx;
use crate::models::g2p::text_normalize::normalize_word_for_lookup;

use super::english_numerals::EnglishNumerals;
use super::english_rules::hand_oov_rules_ipa;

/// Default capacity of the per-engine OOV result cache — see
/// [`EnglishG2p::oov_cache`].
const DEFAULT_OOV_CACHE_CAPACITY: NonZeroUsize = NonZeroUsize::new(10_000).unwrap();

/// English grapheme-to-phoneme engine: lexicon lookup, then OOV model
/// fallback, then hand-written rule fallback.
pub struct EnglishG2p {
    /// Word-to-IPA lexicon, built from a `word\tIPA` TSV at construction.
    lexicon: Lexicon,
    /// OOV ONNX fallback model, tried between the lexicon and hand rules.
    /// `None` when no OOV model is available for this deployment.
    oov_model: Option<oov_onnx::Model>,
    /// Cache of resolved OOV words, so repeated words (proper nouns, brand
    /// names, domain terms) don't re-run ONNX inference. `Mutex`-guarded for
    /// interior mutability since `text_to_ipa` takes `&self`. This relies on
    /// callers driving `text_to_ipa` from a single thread at a time — see
    /// the note on [`LanguageG2p`](super::LanguageG2p) — since TTS runs on a
    /// single dedicated thread per the existing Crane TTS serving pattern.
    /// Only successful, non-empty OOV results are cached — see
    /// [`Self::text_to_ipa`].
    oov_cache: Mutex<LruCache<String, String>>,
    /// When a lexicon hit has multiple pronunciations (a heteronym) whose
    /// alternatives split along US/UK dialect lines, prefer the UK-style
    /// alternative over the US-style one — see [`pick_heteronym_ipa`].
    prefer_british_heteronyms: bool,
}

impl std::fmt::Debug for EnglishG2p {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnglishG2p")
            .field("lexicon", &self.lexicon)
            .field("oov_model", &self.oov_model)
            .field(
                "oov_cache_len",
                &self.oov_cache.lock().map(|c| c.len()).unwrap_or_default(),
            )
            .field("prefer_british_heteronyms", &self.prefer_british_heteronyms)
            .finish()
    }
}

impl EnglishG2p {
    /// Builds an English G2P engine from lexicon TSV content
    /// (`word\tIPA` lines, no header) and an optional OOV fallback model.
    ///
    /// `prefer_british_heteronyms` selects the UK-style pronunciation for
    /// heteronyms whose alternatives split along US/UK dialect lines (see
    /// [`pick_heteronym_ipa`]); pass `false` for US English.
    ///
    /// # Errors
    ///
    /// Returns an error if `lexicon_tsv` is malformed — see
    /// [`Lexicon::from_tsv`].
    pub fn new(
        lexicon_tsv: &str,
        oov_model: Option<oov_onnx::Model>,
        prefer_british_heteronyms: bool,
    ) -> Result<Self> {
        Ok(Self {
            lexicon: Lexicon::from_tsv(lexicon_tsv)?,
            oov_model,
            oov_cache: Mutex::new(LruCache::new(DEFAULT_OOV_CACHE_CAPACITY)),
            prefer_british_heteronyms,
        })
    }

    /// Converts `text` to a space-joined IPA phoneme string.
    ///
    /// Digit runs are expanded to cardinal number words first (see
    /// [`expand_numerals`]), so `"21"` phonemizes as "twenty one" rather than
    /// falling through to hand rules on the literal digit characters.
    ///
    /// Each word is normalized and looked up in the lexicon. On a miss, the
    /// OOV model (if loaded) is tried next; if it's absent, errors, or
    /// returns empty output, the hand-written rule engine is the final
    /// fallback. Tokens that normalize to nothing (all-punctuation) are
    /// skipped.
    ///
    /// Runs in three passes rather than resolving each word as it's seen:
    /// (1) classify every word as a lexicon hit, an OOV-cache hit, an
    /// over-length/no-model case that goes straight to hand rules, or a
    /// pending OOV lookup; (2) run every pending OOV word through
    /// [`oov_onnx::Model::predict_phonemes_batch`] in a single batched call
    /// instead of one autoregressive decode per word; (3) reassemble the
    /// output in original word order. This is what lets multiple OOV words
    /// in one `text_to_ipa` call share decode steps instead of each paying
    /// for its own full decode loop — see
    /// [`oov_onnx::Model::predict_phonemes_batch`] for why that's the real
    /// latency win.
    ///
    /// # Errors
    ///
    /// Currently infallible; returns `Result` to satisfy the
    /// [`Phonemizer`](crate::models::g2p::Phonemizer) trait contract other
    /// language engines rely on.
    pub fn text_to_ipa(&self, text: &str) -> Result<String> {
        /// Per-word classification decided in the first pass, resolved into
        /// IPA in the third. `Lexicon` borrows directly from `self.lexicon`;
        /// the others own their data since it comes from the cache or a
        /// batched inference call made after the borrow would need to end.
        enum WordSource<'a> {
            Lexicon(&'a str),
            Cached(String),
            Oov(usize),
            HandRules,
        }

        let text = expand_numerals(text, &EnglishNumerals);
        let text: &str = &text;

        let words: Vec<Cow<'_, str>> = text
            .split_ascii_whitespace()
            .map(normalize_word_for_lookup)
            .filter(|word| !word.is_empty())
            .collect();

        let mut sources = Vec::with_capacity(words.len());
        let mut oov_words: Vec<&str> = Vec::new();
        for word in &words {
            let word: &str = word;
            if let Some(alts) = self.lexicon.get_all(word) {
                let word_ipa = pick_heteronym_ipa(alts, self.prefer_british_heteronyms);
                sources.push(WordSource::Lexicon(word_ipa));
                continue;
            }
            let Some(model) = self.oov_model.as_ref() else {
                sources.push(WordSource::HandRules);
                continue;
            };
            if word.chars().count() > model.config.max_seq_len {
                sources.push(WordSource::HandRules);
                continue;
            }
            let cached = self
                .oov_cache
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(word)
                .cloned();
            if let Some(cached) = cached {
                sources.push(WordSource::Cached(cached));
                continue;
            }
            // A word repeated within one `text_to_ipa` call (e.g. "wibbly
            // wibbly") shares a single batch slot instead of decoding twice
            // — the cache above isn't populated until after the batch runs,
            // so it can't catch this on its own.
            if let Some(idx) = oov_words.iter().position(|&w| w == word) {
                sources.push(WordSource::Oov(idx));
                continue;
            }
            sources.push(WordSource::Oov(oov_words.len()));
            oov_words.push(word);
        }

        let oov_results = match self.oov_model.as_ref() {
            Some(model) if !oov_words.is_empty() => {
                let results = model.predict_phonemes_batch(&oov_words);
                let mut cache = self
                    .oov_cache
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                for (&word, result) in oov_words.iter().zip(&results) {
                    if let Some(ipa) = result {
                        cache.put(word.to_owned(), ipa.clone());
                    }
                }
                results
            },
            _ => Vec::new(),
        };

        let mut ipa = String::with_capacity(text.len());
        for (word, source) in words.iter().zip(sources) {
            let word: &str = word;
            if !ipa.is_empty() {
                ipa.push(' ');
            }
            match source {
                WordSource::Lexicon(word_ipa) => ipa.push_str(word_ipa),
                WordSource::Cached(cached) => ipa.push_str(&cached),
                WordSource::Oov(idx) => match &oov_results[idx] {
                    Some(word_ipa) => ipa.push_str(word_ipa),
                    None => ipa.push_str(&hand_oov_rules_ipa(word)),
                },
                WordSource::HandRules => ipa.push_str(&hand_oov_rules_ipa(word)),
            }
        }
        Ok(ipa)
    }
}

/// Picks among a heteronym's IPA alternatives, matching Moonshine's
/// `pick_english_heteronym_ipa()`.
///
/// Some CMU-style heteronyms (e.g. "tomato") have alternatives that split
/// along US/UK dialect lines: a US-style reading with a stressed `ˈeɪ`
/// diphthong, and a UK-style reading with a stressed open-back `ˈɑ` (and no
/// `ˈeɪ`). Only when *both* patterns are present among the alternatives does
/// `prefer_british` decide which one wins; otherwise (most heteronyms, e.g.
/// "read"/"lead", don't follow this dialect split at all) this falls back to
/// the lexicographically-first alternative. `alternatives` is assumed
/// already sorted, matching [`Lexicon::get_all`]'s contract.
fn pick_heteronym_ipa<'a>(
    alternatives: impl Iterator<Item = &'a str>,
    prefer_british: bool,
) -> &'a str {
    let alts: Vec<&str> = alternatives.collect();
    let Some(&first) = alts.first() else {
        return "";
    };
    if alts.len() == 1 {
        return first;
    }

    let american_reading_present = alts.iter().copied().any(has_stressed_ei);
    let british_reading_present = alts.iter().copied().any(has_stressed_open_back);
    if american_reading_present && british_reading_present {
        let picked = if prefer_british {
            alts.iter().copied().find(|&s| has_stressed_open_back(s))
        } else {
            alts.iter().copied().find(|&s| has_stressed_ei(s))
        };
        if let Some(s) = picked {
            return s;
        }
    }
    first
}

/// `true` if `s` has a US-style stressed `ˈeɪ` diphthong — see
/// [`pick_heteronym_ipa`].
fn has_stressed_ei(s: &str) -> bool {
    s.contains("ˈeɪ")
}

/// `true` if `s` has a UK-style stressed open-back `ˈɑ` and no `ˈeɪ` — see
/// [`pick_heteronym_ipa`].
fn has_stressed_open_back(s: &str) -> bool {
    s.contains("ˈɑ") && !has_stressed_ei(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_engine() -> EnglishG2p {
        let tsv = "hello\thəlˈoʊ\nworld\twˈɜɹld\n";
        EnglishG2p::new(tsv, None, false).unwrap()
    }

    #[test]
    fn lexicon_hit_returns_ipa() {
        let engine = test_engine();
        assert_eq!(engine.text_to_ipa("hello").unwrap(), "həlˈoʊ");
    }

    #[test]
    fn lexicon_miss_falls_back_to_hand_rules() {
        let engine = test_engine();
        assert_eq!(engine.text_to_ipa("xyzzy").unwrap(), "ksɪzˈaɪ");
    }

    #[test]
    fn multi_word_input_joins_with_spaces() {
        let engine = test_engine();
        assert_eq!(
            engine.text_to_ipa("Hello, world!").unwrap(),
            "həlˈoʊ wˈɜɹld"
        );
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
        assert_eq!(engine.text_to_ipa("hello there").unwrap(), "həlˈoʊ ðˈɛɹ");
    }

    #[test]
    fn new_propagates_malformed_lexicon_error() {
        assert!(EnglishG2p::new("no-tab-here\n", None, false).is_err());
    }

    #[test]
    fn numeral_in_text_expands_before_lexicon_lookup() {
        let tsv = "twenty\ttwˈɛnti\none\twˈʌn\n";
        let engine = EnglishG2p::new(tsv, None, false).unwrap();
        assert_eq!(engine.text_to_ipa("21").unwrap(), "twˈɛnti wˈʌn");
    }

    #[test]
    fn no_numerals_unchanged() {
        let engine = test_engine();
        assert_eq!(engine.text_to_ipa("hello world").unwrap(), "həlˈoʊ wˈɜɹld");
    }

    #[test]
    fn numeral_mixed_with_words() {
        let engine = test_engine();
        let expected = format!("həlˈoʊ {}", hand_oov_rules_ipa("five"));
        assert_eq!(engine.text_to_ipa("hello 5").unwrap(), expected);
    }

    #[test]
    fn heteronym_us_uk_split_picks_stressed_ei_by_default() {
        let tsv = "tomato\ttəmˈeɪtˌoʊ\ntomato\ttəmˈɑtˌoʊ\n";
        let engine = EnglishG2p::new(tsv, None, false).unwrap();
        assert_eq!(engine.text_to_ipa("tomato").unwrap(), "təmˈeɪtˌoʊ");
    }

    #[test]
    fn heteronym_us_uk_split_picks_stressed_open_back_when_prefer_british() {
        let tsv = "tomato\ttəmˈeɪtˌoʊ\ntomato\ttəmˈɑtˌoʊ\n";
        let engine = EnglishG2p::new(tsv, None, true).unwrap();
        assert_eq!(engine.text_to_ipa("tomato").unwrap(), "təmˈɑtˌoʊ");
    }

    #[test]
    fn heteronym_without_us_uk_pattern_falls_back_to_lexicographic_first() {
        // "read"/"lead"-style heteronyms don't follow the US/UK stress
        // split, so both prefer_british settings must agree on the
        // lexicographically-first alternative.
        let tsv = "read\trˈɛd\nread\trˈid\n";
        for prefer_british in [false, true] {
            let engine = EnglishG2p::new(tsv, None, prefer_british).unwrap();
            assert_eq!(engine.text_to_ipa("read").unwrap(), "rˈid");
        }
    }

    #[test]
    fn pick_heteronym_ipa_single_alternative_passes_through() {
        assert_eq!(pick_heteronym_ipa(["həlˈoʊ"].into_iter(), false), "həlˈoʊ");
    }

    #[test]
    fn pick_heteronym_ipa_empty_input_returns_empty_string() {
        assert_eq!(pick_heteronym_ipa(std::iter::empty(), false), "");
    }

    /// An OOV model wrapping an empty ONNX graph, so `predict_phonemes`
    /// always errors (`simple_eval` has no `logits` output to find). Used to
    /// verify that OOV inference failures fall through to hand rules instead
    /// of aborting `text_to_ipa` for the whole input.
    fn failing_oov_model() -> oov_onnx::Model {
        let config_json = r#"{
            "config_schema_version": 1,
            "model_kind": "oov",
            "char_vocab": {"<pad>": 0, "<unk>": 1, "x": 2, "y": 3, "z": 4},
            "phoneme_vocab": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3},
            "train_config": {"max_seq_len": 64},
            "oov_index": {"max_phoneme_len": 64}
        }"#;
        let model = crate::onnx::proto::ModelProto {
            graph: Some(crate::onnx::proto::GraphProto::default()),
            ..Default::default()
        };
        oov_onnx::Model {
            config: oov_onnx::Config::from_json(config_json).unwrap(),
            session: crate::onnx::Session::new(model).unwrap(),
        }
    }

    #[test]
    fn oov_inference_error_falls_back_to_hand_rules_not_propagated() {
        let engine = EnglishG2p::new(
            "hello\thəlˈoʊ\nworld\twˈɜɹld\n",
            Some(failing_oov_model()),
            false,
        )
        .unwrap();
        // "xyzzy" misses the lexicon; the OOV model errors on an empty
        // graph, so this must still succeed via hand rules, not propagate
        // the ONNX error out of text_to_ipa.
        assert_eq!(engine.text_to_ipa("xyzzy").unwrap(), "ksɪzˈaɪ");
    }

    #[test]
    fn multiple_oov_words_all_fall_back_to_hand_rules_when_model_fails() {
        let engine = EnglishG2p::new(
            "hello\thəlˈoʊ\nworld\twˈɜɹld\n",
            Some(failing_oov_model()),
            false,
        )
        .unwrap();
        // Both "xyzzy" and "wibbly" miss the lexicon, so both are batched
        // into a single OOV inference call; that call errors on the empty
        // graph, and both words must still fall back to hand rules rather
        // than one word's failure aborting the whole `text_to_ipa` call.
        let expected = format!(
            "{} {}",
            hand_oov_rules_ipa("xyzzy"),
            hand_oov_rules_ipa("wibbly")
        );
        assert_eq!(engine.text_to_ipa("xyzzy wibbly").unwrap(), expected);
    }

    #[test]
    fn oov_cache_returns_cached_result_on_repeat() {
        // A model must be loaded (even one that always errors) for
        // text_to_ipa's classification pass to consult the cache at all —
        // see `oov_model_skipped_when_word_exceeds_max_seq_len`.
        let engine = EnglishG2p::new(
            "hello\thəlˈoʊ\nworld\twˈɜɹld\n",
            Some(failing_oov_model()),
            false,
        )
        .unwrap();
        // Seed the cache directly with an IPA that differs from what hand
        // rules would produce for "xyzzy" ("ksɪzˈaɪ"), so a cache hit is
        // distinguishable from falling through to hand rules.
        engine
            .oov_cache
            .lock()
            .unwrap()
            .put("xyzzy".to_string(), "kˈæʃd".to_string());
        assert_eq!(engine.text_to_ipa("xyzzy").unwrap(), "kˈæʃd");
    }

    #[test]
    fn oov_cache_not_populated_on_model_miss() {
        let engine = EnglishG2p::new("hello\thəlˈoʊ\n", Some(failing_oov_model()), false).unwrap();
        // "xyzzy" misses the lexicon and the failing OOV model errors, so
        // hand rules produce the result — but a failed inference must not
        // pollute the cache.
        assert_eq!(engine.text_to_ipa("xyzzy").unwrap(), "ksɪzˈaɪ");
    }

    #[test]
    fn oov_model_skipped_when_word_exceeds_max_seq_len() {
        let engine = EnglishG2p::new("hello\thəlˈoʊ\n", Some(failing_oov_model()), false).unwrap();
        // failing_oov_model's max_seq_len is 64.
        let long_word = "x".repeat(100);
        // Seed the cache as if a prior call had succeeded, proving the
        // length guard short-circuits before the cache is even consulted.
        engine
            .oov_cache
            .lock()
            .unwrap()
            .put(long_word.clone(), "shouldnotappear".to_string());
        assert_eq!(
            engine.text_to_ipa(&long_word).unwrap(),
            hand_oov_rules_ipa(&long_word)
        );
    }

    // CER benchmark: measures the OOV tier's contribution against the
    // checked-in `REFERENCE_CER_EN_US` threshold. Needs a real lexicon +
    // OOV model on disk, so these are `#[ignore]`d by default. Run with:
    //
    // ```sh
    // CRANE_G2P_EN_US_DIR=/path/to/en_us \
    //   cargo test -p crane-core --features onnx -- \
    //   models::g2p::languages::english::tests::cer_benchmark --ignored --nocapture
    // ```

    use crate::models::g2p::benchmark::{G2pBenchmarkResult, REFERENCE_CER_EN_US, benchmark_g2p};

    fn model_dir() -> std::path::PathBuf {
        std::env::var("CRANE_G2P_EN_US_DIR")
            .expect("set CRANE_G2P_EN_US_DIR to an en_us G2P model directory")
            .into()
    }

    fn parse_word_ipa_tsv(tsv: &str) -> Vec<(String, String)> {
        tsv.lines()
            .filter_map(|line| line.split_once('\t'))
            .map(|(word, ipa)| (word.to_string(), ipa.to_string()))
            .collect()
    }

    /// Builds an `EnglishG2p` from the on-disk lexicon with the held-out
    /// test words removed and runs it over the test set, reporting the
    /// resulting CER. Does not assert against `REFERENCE_CER_EN_US` itself
    /// — callers decide whether that's meaningful for their configuration
    /// (see `cer_benchmark_with_oov` vs `cer_benchmark_without_oov`).
    ///
    /// Test words are excluded from the lexicon before construction so the
    /// benchmark measures the OOV/rules tiers rather than trivial lexicon
    /// hits — a lexicon hit on a held-out word would trivially match and
    /// mask any regression in the fallback tiers.
    fn run_cer_benchmark(oov_model: Option<oov_onnx::Model>, label: &str) -> G2pBenchmarkResult {
        let test_data_path =
            crate::test_data::get_test_data_file("g2p/en_us/test.tsv").expect("fetch test.tsv");
        let test_tsv = std::fs::read_to_string(&test_data_path).expect("read test.tsv");
        let test_pairs = parse_word_ipa_tsv(&test_tsv);
        let test_words: std::collections::HashSet<&str> =
            test_pairs.iter().map(|(word, _)| word.as_str()).collect();

        let dict_path = model_dir().join("dict_filtered_heteronyms.tsv");
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

        let engine = EnglishG2p::new(&held_out_dict, oov_model, false).expect("build EnglishG2p");

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

        let result = benchmark_g2p(&predictions_ref, &test_pairs_ref, "en_us");
        println!(
            "[{label}] en_us CER: {:.4} ({} errors / {} words)",
            result.cer, result.total_errors, result.total_words
        );
        result
    }

    #[test]
    #[ignore = "needs a local G2P model directory (CRANE_G2P_EN_US_DIR)"]
    fn cer_benchmark_with_oov() {
        let oov_model = oov_onnx::Model::load(&model_dir().join("oov")).expect("load OOV model");
        let result = run_cer_benchmark(Some(oov_model), "with-oov");
        assert!(
            result.cer <= REFERENCE_CER_EN_US,
            "CER {:.4} exceeds reference {REFERENCE_CER_EN_US:.4}",
            result.cer
        );
    }

    #[test]
    #[ignore = "needs a local G2P model directory (CRANE_G2P_EN_US_DIR)"]
    fn cer_benchmark_without_oov() {
        // Informational only: REFERENCE_CER_EN_US reflects the full
        // lexicon+OOV+rules pipeline. With no OOV model, held-out words
        // fall straight to hand rules, which is expected to score well
        // above that reference -- this variant exists to show the OOV
        // tier's contribution by comparison, not to pass a regression gate
        // on its own.
        run_cer_benchmark(None, "without-oov");
    }

    #[test]
    #[ignore = "needs a local G2P model directory (CRANE_G2P_EN_US_DIR)"]
    fn oov_cache_hit_avoids_rerunning_inference() {
        let oov_model = oov_onnx::Model::load(&model_dir().join("oov")).expect("load OOV model");
        // Lexicon without "zoinks" so it's forced through the OOV tier.
        let engine = EnglishG2p::new("hello\thəlˈoʊ\n", Some(oov_model), false).unwrap();

        let first = engine.text_to_ipa("zoinks").expect("text_to_ipa");
        assert_eq!(engine.oov_cache.lock().unwrap().len(), 1);

        let second = engine.text_to_ipa("zoinks").expect("text_to_ipa");
        assert_eq!(first, second);
        assert_eq!(engine.oov_cache.lock().unwrap().len(), 1);
    }

    #[test]
    #[ignore = "needs a local G2P model directory (CRANE_G2P_EN_US_DIR)"]
    fn multiple_oov_words_batch_matches_per_word_inference() {
        // Two engines sharing no state, each with their own freshly loaded
        // OOV model: one resolves "zoinks" and "archaeopteryx" together in
        // a single `text_to_ipa` call (batched), the other resolves them
        // one at a time in separate calls (each its own unbatched decode).
        // Both must produce identical IPA per word, proving batching
        // multiple OOV words into one call doesn't change the result.
        let lexicon = "hello\thəlˈoʊ\n";
        let batched_engine = EnglishG2p::new(
            lexicon,
            Some(oov_onnx::Model::load(&model_dir().join("oov")).expect("load OOV model")),
            false,
        )
        .unwrap();
        let sequential_engine = EnglishG2p::new(
            lexicon,
            Some(oov_onnx::Model::load(&model_dir().join("oov")).expect("load OOV model")),
            false,
        )
        .unwrap();

        let batched = batched_engine
            .text_to_ipa("zoinks archaeopteryx")
            .expect("text_to_ipa");
        let sequential = format!(
            "{} {}",
            sequential_engine
                .text_to_ipa("zoinks")
                .expect("text_to_ipa"),
            sequential_engine
                .text_to_ipa("archaeopteryx")
                .expect("text_to_ipa")
        );

        assert_eq!(batched, sequential);
    }
}
