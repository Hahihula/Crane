# MuScriptor

Automatic music transcription: audio (WAV) → MIDI (multi-track SMF).
Decoder-only transformer with a mel-spectrogram prefix conditioner.

## License

The code in this directory is part of Crane and follows Crane's standard
MIT-style license.

**The upstream *weights* are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) —
non-commercial use only.** This module loads user-supplied weights;
Crane does not redistribute them. See the upstream HuggingFace
organization for the model cards and license terms:
<https://huggingface.co/MuScriptor>.

## Paper

MuScriptor: An Open Model for Multi-Instrument Music Transcription
— Rouard, Krause, Roebel, Simon-Gabriel, Défossez (2026).
arXiv: <https://arxiv.org/abs/2607.08168>

## Variants

| Variant    | Params | Layers | Dim | Heads | HF repo                                           |
|------------|-------:|-------:|----:|------:|---------------------------------------------------|
| `small`    |  ≈100M |     14 | 768 |    12 | <https://huggingface.co/MuScriptor/muscriptor-small>    |
| `medium`   |  ≈300M |     24 | 1024|    16 | <https://huggingface.co/MuScriptor/muscriptor-medium>  |
| `large`    | ≈1.3B  |     48 | 1536|    24 | <https://huggingface.co/MuScriptor/muscriptor-large>   |

All variants share the same input pipeline, tokenizer, and training
recipe; they differ only in depth, width, and head count.

## Architecture (per variant)

- **Self-attention**: plain multi-head attention (no GQA, no RoPE, no
  Q/K-norm). Sinusoidal positions are added in fp32 to the input
  embeddings at the stack level — the KV cache carries no positional
  information, so position re-flows on each forward.
- **FFN**: `dim × 4` hidden size, no bias, GELU activation.
- **Norm**: pre-norm `LayerNorm`, `eps = 1e-5`. No RMSNorm, no QK-norm.
- **Streaming KV cache**: pre-allocated `[B, H, max_seq, D]` buffers
  with host-side `seq_len` cursor — avoids the per-step `.item()` sync
  that a device-side counter would force. Bottom-right-aligned causal
  attention (mask-free for `T_q == 1` decode and `T_q == T_k` prefill;
  explicit bottom-right causal mask for the rectangular case).
- **LM head**: untied `nn.Linear(dim, card)` with `card = 1393`. Token
  embedding has `card + 1 = 1394` rows; the +1 is a reserved
  `zero_idx` slot returned as a zero vector by `ScaledEmbedding`
  (never produced by the generator).

## Conditioning

| Key                | Shape       | Source                                                                                              |
|--------------------|-------------|-----------------------------------------------------------------------------------------------------|
| `self_wav`         | `[B, T, dim]` | log-magnitude mel-spectrogram of the audio chunk (`n_fft=2048`, `hop=160` → 100 Hz × 512 bins) |
| `instrument_group` | `[B, L, dim]` | one or more class IDs from [`MT3_FULL_PLUS_GROUP_NAMES`](crate::mt3)                             |
| `dataset_name`     | `[B, L, dim]` | always `None` at inference (CFG null condition)                                                   |

The mel filterbank is loaded from the checkpoint
(`condition_provider.conditioners.self_wav.mel_spec_transform.mel_scale.fb`)
rather than reconstructed — matching what `muscriptor` saves at train
time.

## Tokenization

MT3-style MIDI event vocabulary, fixed layout:

```
indices [0,    3) → PAD / EOS / UNK  (special tokens)
indices [3, 1004) → shift 0..1000
indices [1004, 1132) → pitch 0..127
indices [1132, 1134) → velocity 0..1
indices [1134, 1135) → tie
indices [1135, 1265) → program 0..129
indices [1265, 1393) → drum 0..127
```

Total = 1393. The `card` field in the checkpoint is authoritative and is
validated to be 1393 (small) or 1395 (medium/large — two extra
reserved/OOV logit slots, masked out at generation time so they can
never be sampled) at load time. See [`mt3.rs`](crate::mt3) for the
decode table, instrument-group resolution helpers, and forbidden-token
masking.

## Inference pattern (audio of any length)

`example/src/muscriptor_transcribe.rs` drives `TranscriptionModel`:

1. **Load weights** — reads `config.json` + `model.safetensors`
   from `--model-dir`. Handles the upstream `emb.0.weight` →
   `emb.weight` remap and pulls the filterbank + class-embedding
   buffers directly from the safetensors map (they're not
   `Parameter`s in the upstream so `VarBuilder::pp` can't see them).
2. **Read WAV** — `hound` decode (or ffmpeg for other formats),
   mono-ize, leave sample rate intact.
3. **Run `TranscriptionModel::transcribe_to_midi`** — splits the
   audio into consecutive `SEGMENT_DURATION` (5 s) chunks (each
   resampled to 16 kHz and zero-padded if it's the last, short one)
   and, for each chunk in order:
   - builds the mel + instrument-group conditioning for that chunk;
   - **tie-prologue forcing**: every chunk after the first
     teacher-forces the `(program, pitch)` pairs still sounding at
     the end of the *previous* chunk as its opening tokens (via
     `mt3::tie_section_token_ids`), so a note straddling a chunk
     boundary stays attributed to the same instrument instead of the
     model re-guessing it on the other side. There is no cross-chunk
     KV cache — each chunk gets an independent forward pass; this
     forced prologue is the *only* thing carrying continuity across
     chunks, matching the upstream's default `prelude_forcing=True`;
   - runs the autoregressive generate loop (greedy, or sampled via
     `TranscribeConfig::use_sampling` + temperature/top-k/top-p) up
     to `max_gen_len` tokens, decoding each token through
     `OpenNoteTracker` — Program/Velocity tokens are *state*, Pitch
     is the *trigger* that opens or closes a note at
     `(current_program, pitch)` against whatever velocity is
     currently active (a retrigger closes-then-reopens in one step);
     Drum tokens are instantaneous hits.
4. **Write Standard MIDI File** (type 1, 120 BPM, 480 ticks/beat) via
   the in-repo writer, splitting notes onto per-program tracks named
   from the decoded instrument group (`mt3::instrument_name_for_program`,
   falling back to `program_<n>` for an unmapped program).

## Not implemented

* **Classifier-free guidance** (`cfg_coef != 1.0` is rejected at
  generation time). The upstream runs a doubled cond/uncond batch
  through a doubled KV cache; `LMModel::generate` here is
  single-sequence only, so this would need real surgery to the
  `LayerState`/cache plumbing rather than a local tweak.
* **Beam search** — greedy or sampled (temperature/top-k/top-p, via
  `candle_transformers::generation::LogitsProcessor`) decoding only.
* **Dedicated int8/int4 KV-cache quantization** (the `CRANE_KV_QUANT`
  scheme Qwen 3.5 has). Not implemented because it isn't the actual VRAM
  driver here: weights dominate (≈5.1 GB of `large`'s ≈7.1 GB f32 peak),
  and `--dtype f16` already halves the KV cache for free as a side effect
  of the transformer's compute dtype — see below.

None of these change the architecture; each is a plausible follow-up.

## Precision / quantization

`Model::new_with_options(path, device, dtype, quant)` — the transformer
(embedding table, attention/FFN projections, LM head, and therefore the KV
cache, which is allocated at `dtype`) follows `dtype`; `quant`
additionally in-situ-quantizes every big `Linear` to a GGML level (same
levels as Qwen 3.5's `--quant`: `q4_0`, `q8_0`, `q4k`, `q6k`, …), stacking
with `dtype`. The mel-spectrogram and class-embedding conditioners
**always load and compute in F32 regardless of `dtype`/`quant`** — the
conditioning pipeline is < 2M params either way, nothing worth quantizing,
and log-mel of quiet passages genuinely underflows in fp16.
`LMModel::forward` casts the conditioner's F32 output to the transformer's
dtype right at the splice point, so this costs nothing at the seam.

CLI: `--dtype f32|f16|bf16` (default `f32`) and `--quant q4k|q8_0|…`
(default: unset, i.e. `dtype` precision).

Measured on the RTX 3090, `large`, the 48 s / 10-chunk Toccata (see the
root README's MuScriptor section for the exact command and output files):

| Config             | Peak VRAM | Time     |
|---------------------|----------:|---------:|
| f32 (default)       |   ≈7.1 GB |   ≈27 s  |
| f16                 |   ≈3.9 GB |   ≈24 s  |
| f16 + `q4k`         |   ≈2.2 GB |   ≈51 s  |

`--quant` trades VRAM for time, not just precision: `QMatMul`'s forward
casts to F32, does the dequantized matmul, casts back — worthwhile
bandwidth savings at prefill-sized batches, but at this model's
single-token (`T_q == 1`) decode steps the per-step dequant/cast overhead
dominates and generation gets *slower*, not faster (matches this repo's
general decode-is-dispatch-bound finding for hybrid/quantized models — see
the root README's `2026.08.02` update). Reach for `--quant` when VRAM is
the hard constraint and you can tolerate the extra wall-clock; reach for
plain `--dtype f16` when it isn't (smaller *and* mildly faster, no
extra overhead).

## Performance notes

* CPU (f32): small (100M) model ≈ 7 minutes per 5-second chunk on a
  single thread (no fused SDPA). Large (1.3B) is 30+ minutes per chunk.
* CUDA (RTX 3090, f32): small ≈ 0.5 s/chunk, large ≈ 2.5-3 s/chunk — the
  48 s / 10-chunk Toccata transcribes in ≈ 5 s (small) / ≈ 27-40 s
  (large; run-to-run variance is real — see below). See the precision
  table above for `--dtype`/`--quant` deltas.
* Metal: not runtime-verified (no Apple hardware in this repo's dev
  environment), but the module has no CUDA-specific code paths — every
  op used (`slice_set`, `where_cond`, `matmul`, `softmax_last_dim`,
  the mel FFT which always runs on CPU via `rustfft` regardless of
  device) is a standard candle primitive already exercised by this
  repo's other Metal-supported models. `--features metal` builds
  cleanly on this Linux box up to the point where `candle-core`'s
  `metal` feature pulls in `objc2`, which refuses to compile for any
  non-Apple target (`compile_error!` from the crate itself) — that's
  an Apple-toolchain requirement common to every model in this repo,
  not something specific to MuScriptor.
* **VRAM does not grow with input length.** Each 5 s chunk gets an
  independent forward pass with a freshly-allocated, fixed-size KV cache
  (`prepend_total + 1 + max_gen_len` rows — constant across chunks,
  since every chunk padded to `SEGMENT_DURATION` has ~the same mel-frame
  count); nothing accumulates across chunks except the tiny open-notes
  tracker. Verified by watching `nvidia-smi` per-process VRAM over a
  live 4+ minute `large` transcription of a real ~4 minute song: flat at
  ~7.1 GB the entire run, no upward trend. Wall-clock time *does* scale
  linearly with length (more chunks, same per-chunk cost), which is the
  expected, unavoidable kind of "grows with length."
* **Greedy decoding is not bit-reproducible run-to-run on CUDA.**
  Floating-point reduction order in CUDA kernels isn't fixed, so tiny
  numerical differences between runs can flip an argmax tie at some
  step deep in a ~2000-token chunk and cascade into a different (still
  valid) transcription. Expect this, not a bug, when diffing two runs
  of the same config.

## Layout

| File                    | What                                                                              |
|-------------------------|-----------------------------------------------------------------------------------|
| `mod.rs`                | module re-exports                                                                |
| `config.rs`             | `VariantConfig`, DSP constants (`SAMPLE_RATE`, `N_FFT`, `HOP_LENGTH`, `N_MELS`)  |
| `mt3.rs`                | MT3 vocabulary, MT3Tokenizer, instrument-group helpers                          |
| `midi.rs`               | minimal Standard MIDI File (SMF type-1) writer                                   |
| `transformer.rs`        | streaming MHA, sinusoidal pos embedding, layer + stack                           |
| `conditioner.rs`        | mel-spec + class conditioners, ConditioningProvider                              |
| `model.rs`              | LMModel, Model (weight loading), TranscriptionModel                              |
