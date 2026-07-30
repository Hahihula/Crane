// SPDX-License-Identifier: MIT

//! Performance benchmarks for the G2P hot path: `EnglishG2p::text_to_ipa()`
//! (the full phonemization pipeline), `IpaNormalizer::normalize()` (IPA
//! postprocessing for the Kokoro vocoder), and the `text_normalize` helpers
//! (`split_text_to_words`, `normalize_word_for_lookup`) both of those call
//! into. All run once per TTS request and gate time-to-first-audio.
//!
//! `text_to_ipa` uses a small inline lexicon and no OOV model, so it
//! measures dispatch overhead in isolation. `text_to_ipa_full_lexicon`
//! loads a real `en_us` lexicon and OOV ONNX model directory from
//! `CRANE_G2P_EN_US_DIR` for a production-representative comparison
//! point; it silently skips (no benchmark registered, non-error exit) when
//! that variable isn't set, so it's opt-in and doesn't affect the default
//! `cargo bench` run.

use std::collections::HashMap;
use std::hint::black_box;
use std::path::PathBuf;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use crane_core::models::g2p::languages::english::EnglishG2p;
use crane_core::models::g2p::oov_onnx;
use crane_core::models::g2p::text_normalize::{normalize_word_for_lookup, split_text_to_words};
use crane_core::models::kokoro_tts::build_kokoro_normalizer;

/// Small hand-picked lexicon covering every word used by the benchmarks
/// below, so `text_to_ipa` exercises the lexicon-hit fast path without
/// loading the real 133K-entry English TSV or an OOV ONNX model -- neither
/// is needed to measure G2P dispatch overhead, and requiring them would make
/// this benchmark depend on external model assets.
const INLINE_LEXICON: &str = "\
hello\thəlˈoʊ
the\tðə
quick\tkwˈɪk
brown\tbɹˈaʊn
fox\tfˈɑks
jumps\tdʒˈʌmps
over\tˈoʊvɚ
lazy\tlˈeɪzi
dog\tdˈɔɡ
";

/// Parses the real 114-entry Kokoro phoneme vocabulary shipped as a test
/// fixture, for building a realistic `IpaNormalizer` in the benchmarks below.
fn kokoro_vocab() -> HashMap<char, i64> {
    let json = include_str!("../tests/data/g2p/kokoro_vocab.json");
    let raw: HashMap<String, i64> = serde_json::from_str(json).unwrap();
    raw.into_iter()
        .map(|(k, v)| {
            let mut chars = k.chars();
            let c = chars.next().expect("vocab key must not be empty");
            assert!(chars.next().is_none(), "vocab key {k:?} is not a single codepoint");
            (c, v)
        })
        .collect()
}

fn bench_text_to_ipa(c: &mut Criterion) {
    let engine = EnglishG2p::new(INLINE_LEXICON, None, false).unwrap();
    let mut group = c.benchmark_group("text_to_ipa");
    group.throughput(Throughput::Elements(1));

    group.bench_function("single_known_word", |b| {
        b.iter(|| engine.text_to_ipa(black_box("hello")).unwrap());
    });

    group.bench_function("known_sentence", |b| {
        b.iter(|| {
            engine
                .text_to_ipa(black_box("the quick brown fox jumps over the lazy dog"))
                .unwrap()
        });
    });

    group.bench_function("single_unknown_word", |b| {
        b.iter(|| engine.text_to_ipa(black_box("zoinks")).unwrap());
    });

    group.finish();
}

/// Directory of a real `en_us` G2P model (lexicon + OOV ONNX model), or
/// `None` if `CRANE_G2P_EN_US_DIR` isn't set. This benchmark group depends
/// on external model assets not checked into the repo, so it's opt-in.
fn g2p_model_dir() -> Option<PathBuf> {
    std::env::var("CRANE_G2P_EN_US_DIR").ok().map(PathBuf::from)
}

/// Full-lexicon variant of [`bench_text_to_ipa`], loading the real
/// ~133K-entry English lexicon and OOV ONNX model instead of the 9-word
/// inline lexicon, so these numbers are directly comparable to Moonshine's
/// C++ G2P baseline (recorded against the same real assets). Skipped when
/// `CRANE_G2P_EN_US_DIR` isn't set.
fn bench_text_to_ipa_full_lexicon(c: &mut Criterion) {
    let Some(model_dir) = g2p_model_dir() else {
        eprintln!(
            "skipping text_to_ipa_full_lexicon: set CRANE_G2P_EN_US_DIR to an en_us G2P model \
             directory to run it"
        );
        return;
    };

    let dict_path = model_dir.join("dict_filtered_heteronyms.tsv");
    let dict_tsv = std::fs::read_to_string(&dict_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", dict_path.display()));
    let oov_model =
        oov_onnx::Model::load(&model_dir.join("oov")).expect("load OOV model");
    let engine = EnglishG2p::new(&dict_tsv, Some(oov_model), false).expect("build EnglishG2p");

    let corpus: Vec<&str> = include_str!("../tests/data/g2p/en_us_test.tsv")
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .map(|(word, _)| word)
        .collect();

    let mut group = c.benchmark_group("text_to_ipa_full_lexicon");
    group.throughput(Throughput::Elements(1));

    group.bench_function("single_known_word", |b| {
        b.iter(|| engine.text_to_ipa(black_box("hello")).unwrap());
    });

    group.bench_function("known_sentence", |b| {
        b.iter(|| {
            engine
                .text_to_ipa(black_box("the quick brown fox jumps over the lazy dog"))
                .unwrap()
        });
    });

    // Unlike `bench_text_to_ipa`'s `single_unknown_word` (no OOV model, so
    // every call takes the same hand-rules path), this engine has a real OOV
    // model and therefore a live OOV LRU cache: reusing the same word across
    // iterations would hit the cache after the first call and measure
    // cache-hit latency instead of ONNX inference. A fresh nonsense word per
    // iteration guarantees a lexicon miss and a cache miss every time.
    let unknown_word_counter = std::cell::Cell::new(0u64);
    group.bench_function("single_unknown_word", |b| {
        b.iter(|| {
            let n = unknown_word_counter.get();
            unknown_word_counter.set(n + 1);
            let word = format!("zqxoovbench{n}");
            engine.text_to_ipa(black_box(word.as_str())).unwrap()
        });
    });

    group.finish();

    let mut corpus_group = c.benchmark_group("text_to_ipa_full_lexicon_corpus");
    corpus_group.throughput(Throughput::Elements(corpus.len() as u64));
    corpus_group.bench_function("corpus_5000_words", |b| {
        b.iter(|| {
            for word in &corpus {
                engine.text_to_ipa(black_box(word)).unwrap();
            }
        });
    });
    corpus_group.finish();
}

fn bench_ipa_normalize(c: &mut Criterion) {
    let vocab = kokoro_vocab();
    let normalizer = build_kokoro_normalizer("en_us", &vocab).unwrap();
    let mut group = c.benchmark_group("ipa_normalize");
    group.throughput(Throughput::Elements(1));

    group.bench_function("short_word", |b| {
        b.iter(|| normalizer.normalize(black_box("həlˈoʊ")));
    });

    // The double space between "dʒˌæpənˈiz" and "lˈɪt" is intentional: it
    // exercises `normalize()`'s whitespace-run-collapsing pass.
    let long_sentence =
        "sˈɛndʒ nˈoʊ vælkˈaɪɹaɪɑː θɹˈi ˌʌnɹɪkˈɔɹdɪd kɹˈɑnɪkəlz dʒˌæpənˈiz  lˈɪt";
    group.bench_function("long_sentence", |b| {
        b.iter(|| normalizer.normalize(black_box(long_sentence)));
    });

    group.finish();
}

fn bench_text_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_normalize");
    group.throughput(Throughput::Elements(1));

    group.bench_function("split_text_to_words", |b| {
        b.iter(|| split_text_to_words(black_box("the quick brown fox jumps over the lazy dog")));
    });

    group.bench_function("normalize_word_for_lookup", |b| {
        b.iter(|| normalize_word_for_lookup(black_box("Hello,")));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_text_to_ipa,
    bench_text_to_ipa_full_lexicon,
    bench_ipa_normalize,
    bench_text_normalize
);
criterion_main!(benches);
