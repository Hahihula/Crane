// SPDX-License-Identifier: MIT

//! English (`en_us`) grapheme-to-phoneme engine.
//!
//! Three tiers: lexicon lookup, then OOV ONNX model inference (when a model
//! is loaded and produces non-empty output), then hand-written
//! letter-to-sound rules as the final fallback.

use anyhow::Result;

use crate::models::g2p::lexicon::Lexicon;
use crate::models::g2p::oov_onnx;
use crate::models::g2p::text_normalize::normalize_word_for_lookup;

use super::english_rules::hand_oov_rules_ipa;

/// English grapheme-to-phoneme engine: lexicon lookup, then OOV model
/// fallback, then hand-written rule fallback.
#[derive(Debug)]
pub struct EnglishG2p {
    /// Word-to-IPA lexicon, built from a `word\tIPA` TSV at construction.
    lexicon: Lexicon,
    /// OOV ONNX fallback model, tried between the lexicon and hand rules.
    /// `None` when no OOV model is available for this deployment.
    oov_model: Option<oov_onnx::Model>,
}

impl EnglishG2p {
    /// Builds an English G2P engine from lexicon TSV content
    /// (`word\tIPA` lines, no header) and an optional OOV fallback model.
    ///
    /// # Errors
    ///
    /// Returns an error if `lexicon_tsv` is malformed — see
    /// [`Lexicon::from_tsv`].
    pub fn new(lexicon_tsv: &str, oov_model: Option<oov_onnx::Model>) -> Result<Self> {
        Ok(Self {
            lexicon: Lexicon::from_tsv(lexicon_tsv)?,
            oov_model,
        })
    }

    /// Converts `text` to a space-joined IPA phoneme string.
    ///
    /// Each word is normalized and looked up in the lexicon. On a miss, the
    /// OOV model (if loaded) is tried next; if it's absent, errors, or
    /// returns empty output, the hand-written rule engine is the final
    /// fallback. Tokens that normalize to nothing (all-punctuation) are
    /// skipped.
    ///
    /// # Errors
    ///
    /// Currently infallible; returns `Result` to satisfy the
    /// [`Phonemizer`](crate::models::g2p::Phonemizer) trait contract other
    /// language engines rely on.
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
            if let Some(word_ipa) = self.lexicon.get(&word) {
                ipa.push_str(word_ipa);
            } else if let Some(oov_ipa) = self.try_oov_model(&word) {
                ipa.push_str(&oov_ipa);
            } else {
                ipa.push_str(&hand_oov_rules_ipa(&word));
            }
        }
        Ok(ipa)
    }

    /// Attempts OOV model inference for `word`.
    ///
    /// Returns `None` if no OOV model is loaded, if inference fails, or if
    /// the model produces an empty result — all three fall through to hand
    /// rules. A single word's OOV failure must never abort phonemization of
    /// the rest of the sentence, so inference errors are swallowed here
    /// rather than propagated. In practice this currently means every OOV
    /// call falls through: the shipped Moonshine-TTS English model always
    /// errors on `candle-onnx`'s missing `LayerNormalization` support (see
    /// the module doc on [`oov_onnx`](crate::models::g2p::oov_onnx)), so
    /// this tier is a documented no-op against real assets until that's
    /// fixed, not a silently-broken one.
    fn try_oov_model(&self, word: &str) -> Option<String> {
        let model = self.oov_model.as_ref()?;
        let ipa = model.predict_phonemes(word).ok()?;
        if ipa.is_empty() {
            return None;
        }
        Some(ipa)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_engine() -> EnglishG2p {
        let tsv = "hello\thəlˈoʊ\nworld\twˈɜɹld\n";
        EnglishG2p::new(tsv, None).unwrap()
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
        assert_eq!(engine.text_to_ipa("hello there").unwrap(), "həlˈoʊ ðˈɛɹ");
    }

    #[test]
    fn new_propagates_malformed_lexicon_error() {
        assert!(EnglishG2p::new("no-tab-here\n", None).is_err());
    }

    #[test]
    fn try_oov_model_returns_none_when_no_model_loaded() {
        let engine = test_engine();
        assert!(engine.try_oov_model("xyzzy").is_none());
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
        oov_onnx::Model {
            config: oov_onnx::Config::from_json(config_json).unwrap(),
            model: crate::onnx::proto::ModelProto::default(),
        }
    }

    #[test]
    fn oov_inference_error_falls_back_to_hand_rules_not_propagated() {
        let engine = EnglishG2p::new(
            "hello\thəlˈoʊ\nworld\twˈɜɹld\n",
            Some(failing_oov_model()),
        )
        .unwrap();
        // "xyzzy" misses the lexicon; the OOV model errors on an empty
        // graph, so this must still succeed via hand rules, not propagate
        // the ONNX error out of text_to_ipa.
        assert_eq!(engine.text_to_ipa("xyzzy").unwrap(), "ksɪzˈaɪ");
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

    use crate::models::g2p::benchmark::{REFERENCE_CER_EN_US, benchmark_g2p};

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
    /// test words removed, runs it over the test set, and asserts the
    /// resulting CER is at or below `REFERENCE_CER_EN_US`.
    ///
    /// Test words are excluded from the lexicon before construction so the
    /// benchmark measures the OOV/rules tiers rather than trivial lexicon
    /// hits — a lexicon hit on a held-out word would trivially match and
    /// mask any regression in the fallback tiers.
    fn run_cer_benchmark(oov_model: Option<oov_onnx::Model>, label: &str) {
        let test_pairs =
            parse_word_ipa_tsv(include_str!("../../../../tests/data/g2p/en_us_test.tsv"));
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

        let engine = EnglishG2p::new(&held_out_dict, oov_model).expect("build EnglishG2p");

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
        assert!(
            result.cer <= REFERENCE_CER_EN_US,
            "[{label}] CER {:.4} exceeds reference {REFERENCE_CER_EN_US:.4}",
            result.cer
        );
    }

    #[test]
    #[ignore = "needs a local G2P model directory (CRANE_G2P_EN_US_DIR)"]
    fn cer_benchmark_with_oov() {
        let oov_model =
            oov_onnx::Model::load(&model_dir().join("oov")).expect("load OOV model");
        run_cer_benchmark(Some(oov_model), "with-oov");
    }

    #[test]
    #[ignore = "needs a local G2P model directory (CRANE_G2P_EN_US_DIR)"]
    fn cer_benchmark_without_oov() {
        run_cer_benchmark(None, "without-oov");
    }
}
