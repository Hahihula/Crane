# G2P (Grapheme-to-Phoneme)

G2P turns written text into phonemes. A phoneme is a unit of sound, like
the "k" sound in "cat". Text-to-speech models don't read letters. They
read phonemes. So before Crane can speak a sentence, this module has to
convert it from text to phonemes first.

Phonemes are written using IPA, the International Phonetic Alphabet. IPA
gives every sound in every language its own symbol. That's what lets one
system handle English, German, and other languages without inventing a
new alphabet for each one.

For each supported language, G2P has a lexicon: a dictionary that maps
words to their IPA pronunciation. The lexicon is backed by
[`lexicon.rs`](lexicon.rs)'s FST (a compact, fast lookup structure). Not
every word is in the dictionary. For words the lexicon doesn't cover, each
language falls back to hand-written rules and/or a small neural model. See
[`languages/`](languages) for that per-language logic.

## Dictionary sources

Lexicon data comes from third-party dictionaries. Scripts in
[`data/g2p/`](../../../../data/g2p) at the repo root convert each one into
the plain `word<TAB>ipa` format `Lexicon::from_tsv` expects:

- **German** (`de`): extracted directly from a German Wiktionary XML dump
  by `extract-de-wiktionary-ipa.py`. Wiktionary is the ultimate source for
  the German data most other IPA dictionaries redistribute, including
  `open-dict-data/ipa-dict`. Extracting it ourselves gets the license
  right. It's CC BY-SA 4.0, with proper attribution, instead of a
  downstream copy that mislabels it.
- **English** (`en_us`): converted from `open-dict-data/ipa-dict`'s
  `en_US.txt` by `ipa-dict-to-tsv.py`. This specific language file is
  MIT-licensed.

Each conversion script writes a `<output>.PROVENANCE.md` next to its TSV.
That file records the exact source and license. License and attribution
can vary per language file, even within the same upstream project. The
German case above is a real example of that. Check the provenance file
before adding a new language's dictionary to a model's assets, and before
publishing a converted dictionary anywhere.

## Dictionary IPA isn't what every model expects

These dictionaries transcribe pronunciation using standard,
linguistics-reference-style IPA conventions. For example, the
primary/secondary stress marks `ˈ`/`ˌ` are placed before a syllable's
entire onset consonant cluster. Affricates and diphthongs are written as
multi-codepoint sequences like `t͡ʃ` or `aɪ`. That's the right choice for
the lexicon itself. It's what lets `benchmark.rs`'s CER benchmark validate
G2P output against a reference corpus written in the same convention. It
also keeps the lexicon usable by any future consumer, not just one
specific model.

But a TTS model was trained on whatever phoneme convention its own
training pipeline produced. That's frequently not this dictionary
convention. For example, Kokoro's German checkpoint was fine-tuned on
espeak-ng-style phonemization. Espeak-ng places stress immediately before
the vowel, not before the onset cluster. Kokoro's vocabulary also has
single-codepoint tokens for affricates and diphthongs that dictionary IPA
spells with multiple codepoints. Feeding a model raw dictionary IPA it
wasn't trained on produces audible artifacts. See the reposition-stress
fix in `kokoro_tts/ipa.rs` and `kokoro_tts/model.rs` for a real example: a
word-initial stress mark with no preceding consonant synthesized as a
spurious vowel.

**Each model implementation is responsible for bridging this gap itself,
in its own module, not in `g2p/`.** For example, `kokoro_tts/ipa.rs`
builds an [`IpaNormalizer`](ipa_postprocess.rs) with Kokoro-vocab-specific
replacement tables
(`SHARED_KOKORO_REPLACEMENTS`/`EN_EXTRA_KOKORO_REPLACEMENTS`/`DE_EXTRA_KOKORO_REPLACEMENTS`).
It also has a dedicated `reposition_stress_before_vowel()` scanning
function for the stress-placement mismatch, applied to German only, right
before phonemes are handed to the model.

When wiring up a new phoneme-consuming model, check what phoneme
convention that model's own training data actually used. Don't just check
whether it's IPA. Do this before assuming G2P's lexicon output can be fed
to it directly. Then add the model's own normalization step alongside its
other model-specific code, the same way `kokoro_tts` does. Don't modify
`g2p/`'s shared lexicon output to suit one model.
