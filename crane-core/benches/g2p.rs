// SPDX-License-Identifier: MIT

//! Performance benchmarks for the G2P hot path: `EnglishG2p::text_to_ipa()`
//! (the full phonemization pipeline), `IpaNormalizer::normalize()` (IPA
//! postprocessing for the Kokoro vocoder), and the `text_normalize` helpers
//! (`split_text_to_words`, `normalize_word_for_lookup`) both of those call
//! into. All run once per TTS request and gate time-to-first-audio.
//!
//! This sets up the `criterion` harness only -- no baseline numbers are
//! recorded here. A later step runs Moonshine's C++ G2P CLI on the same
//! inputs and records its latency/throughput as the regression baseline.

use std::collections::HashMap;
use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use crane_core::models::g2p::languages::english::EnglishG2p;
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

criterion_group!(benches, bench_text_to_ipa, bench_ipa_normalize, bench_text_normalize);
criterion_main!(benches);
