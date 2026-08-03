# Kokoro TTS

Kokoro is a lightweight, open-weight text-to-speech model. This module is
Crane's native Rust implementation: text goes in, spoken audio comes out,
with no espeak or Python dependency.

Text is first turned into phonemes (the sound units that make up speech)
by Crane's own grapheme-to-phoneme (G2P) engine. Those phonemes then run
through Kokoro's ONNX model to produce audio.

## Kudos

Kudos to [Kokoro-TTS](https://github.com/hexgrad/kokoro) for the voice
model, and to [Moonshine-TTS](https://github.com/moonshine-ai/moonshine)
for the original espeak-free G2P approach that served as the blueprint for
this module's phonemizer.

## What we improved

Crane's G2P engine reuses Moonshine's rules but changes how they run:

- **Aho-Corasick automaton.** IPA cleanup applies dozens of text
  replacements (e.g. turning `"eɪ"` into a single symbol). Moonshine
  checks each replacement one at a time. Crane compiles them all into one
  automaton that finds every match in a single pass over the text, so
  the cost stays flat no matter how many replacement rules exist.

- **Beam search for unknown words.** Words missing from the pronunciation
  dictionary go through a neural model that predicts phonemes letter by
  letter. Instead of keeping only the single best guess at each step,
  Crane keeps the top 3 candidate pronunciations in parallel and picks
  the best-scoring one at the end. This gives more accurate pronunciations
  for unfamiliar words, at a small, fixed extra cost.

- **Cache for unknown words.** Once an unknown word has been through beam
  search, its pronunciation is kept in memory so the next time that word
  shows up, it's a lookup instead of a full model run. This cache lives as
  long as the process does, so it helps with more than just one long piece
  of text — a name or brand word that keeps reappearing across many
  separate requests to a running server gets the speedup too.

Together, these keep the G2P step fast enough that the full pipeline runs
faster than real time, on CPU alone.
