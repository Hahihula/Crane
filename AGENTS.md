# AGENTS.md — Crane

Crane is a Rust workspace (`edition = "2024"`) that runs LLM/MLLM/TTS
inference on top of the `candle` framework. There are **no CI workflows,
no `clippy.toml`, no `rustfmt.toml`, no pre-commit config** — verify your
changes by building and running tests yourself.

## Workspace layout

The Cargo workspace at `Cargo.toml` has four members:

| Crate            | Path             | Role                                                                  |
|------------------|------------------|-----------------------------------------------------------------------|
| `crane-core`     | `crane-core/`    | Library: model implementations, autotokenizer, fused CUDA kernels.    |
| `crane`          | `crane/`         | High-level SDK (`chat/vision/audio/multimodal/llm/common`).           |
| `crane-serve`    | `crane-serve/`   | OpenAI & SGLang-compatible HTTP server (axum + continuous-batching).  |
| `crane-examples` | `example/`       | Demo binaries; targets declared in `example/Cargo.toml`.              |

## Build

```bash
# CPU
cargo build --release

# NVIDIA GPU — also pulls in bindgen_cuda and compiles crane-core/kernels/*.cu
cargo build --release --features cuda
```

`install.sh` auto-detects macOS / Linux+CUDA / Linux-CPU and runs the right
`cargo build -p crane --features ...`.

## Tests

```bash
cargo test -p crane-serve    # engine, scheduler, model_factory, API types
cargo test -p crane-core     # autotokenizer, generation, qwen3_5
```

`scripts/test_*.sh` and `tests/test_cv*.py` are **manual** model-output sanity
scripts (PyTorch comparison); not a CI test suite.

## Qwen 3.5 / Ornith (hybrid Mamba / Transformer)

Lives in `crane-core/src/models/qwen3_5/` and `crane-core/src/gdn/`. The
**base implementation was added in #36**; this branch (this PR) layers
optimizations and tool-calling on top.

Verified end-to-end against `Qwen3.5-0.8B` (CPU, CUDA, Metal). Prefill
argmax matches HuggingFace Transformers bit-exactly in f32/f16/bf16
(`token 283 " ="` on a 512-token prefill) and decoding is coherent.
KV cache, fused CUDA GDN kernel, tool-calling, and quantized caches
all work — see below.

### Environment variables

| Variable               | Default  | What it does                                                                      |
|------------------------|----------|-----------------------------------------------------------------------------------|
| `CRANE_GDN_PORTABLE`   | `unset`  | If set, GDN recurrence runs op-by-op in Candle instead of the fused CUDA kernel.  |
| `CRANE_KV_QUANT`       | unset    | `int8` → per-token symmetric int8 K/V cache (≈2× memory). `int4` → packed int4 (≈4×). |
| `CRANE_FULL_RECOMPUTE` | unset    | If set, generation reset-and-reprocesses the prefix each step (O(n²); debugging cross-check for the incremental path). |

### GGUF loading (qwen3_5)

The `qwen35` GGUF loader reads the tokenizer, EOS / PAD ids, special-token
registry and the Jinja chat template directly from the file's metadata
(`tokenizer.ggml.tokens`, `tokenizer.ggml.merges`, `tokenizer.ggml.token_type`,
`tokenizer.ggml.{eos,padding}_token_id`, `tokenizer.chat_template`). **No
sibling `tokenizer.json` / `chat_template.jinja` is required.** Older
quantizers that omit `tokenizer.ggml.tokens` / `merges` automatically fall
back to a sibling `tokenizer.json` if one is present.

This avoids a tokenizer-mismatch trap: a `tokenizer.json` copied from the
sister safetensors checkpoint encodes chat-template tokens to different IDs
than the ones llama.cpp quantized against, which used to send the GDN
recurrence into a degenerate loop on `/v1/chat/completions`. Using the
embedded tokenizer keeps the IDs canonical.

### Vision (qwen3_5_vl)

Multimodal Qwen 3.5 (`Qwen3_5ForConditionalGeneration`) checkpoints run
end-to-end on `--model-type qwen3_5_vl`. The pipeline:

- **Image preprocessing** (`models/qwen3_5/processor.rs`): HF
  `Qwen2VLImageProcessor` smart-resize (resize so total pixels fall in
  `[min_pixels, max_pixels]` AND both dims are multiples of
  `patch * spatial_merge = 32`), normalize by `image_mean` / `image_std`,
  reshape into `[num_patches, temporal_patch * in_channels * patch * patch]`.
- **Vision tower** (`models/qwen3_5/vision.rs`): full Qwen ViT lifted
  from `qwen3_vl::vision` — Conv3d patch embed, 12 transformer blocks,
  per-block LayerNorm, fast attention, `PatchMerger` MLP that
  2×2-spatially-merges and projects to the text hidden size.
- **Multimodal decoder** (`models/qwen3_5/vlm.rs`): the existing
  `Qwen3_5TextModel` is reused unchanged; image embeddings are spliced
  over the `<|image_pad|>` placeholder positions and per-token 3D
  position ids (T/H/W) drive MRoPE.

**MRoPE-3D positions:** `MRotaryEmbedding::cos_sin_with_position_ids`
takes a `[3, S]` tensor of T/H/W positions and produces the
interleaved-MRoPE cos/sin tables of shape `[S, rot_dim/2]`. Inside
`Qwen3_5TextModel::forward_embeds` the new path replaces the
1D-position `cos_sin(start, seq_len)` call when 3D positions are
provided; the text-only path is unaffected (all existing 67+115+6 tests
still pass). The per-axis slice is half the doubled section size because
candle's `rotary_emb::rope` pairs `i` with `i + rot_dim/2` inside the
rotary slice — pair-duplicating the per-axis tables to length `rot_dim`
is NOT needed.

**API surface (crane-serve):**
- `--model-type qwen3_5_vl` triggers auto-detection from
  `architectures: ["Qwen3_5ForConditionalGeneration"]` or `vision_config`
  presence on `model_type: qwen3_5`.
- `POST /v1/chat/completions` accepts OpenAI-style multimodal payloads
  with `image_url` content parts (remote URL or `data:image/...;base64,...`
  inline — the VLM handler in `handlers/vlm.rs` decodes the latter).
- The image is preprocessed on the engine thread (not the request
  thread), prefill runs once, then `decode_step` advances one token at a
  time reusing the KV cache.

**Verified end-to-end:** `crane-core/examples/qwen3_5_vl_e2e.rs` (model
side) and `crane-serve/examples/qwen3_5_vl_chat.rs` (server side). Both
produce coherent descriptions of a real PNG.

### Measured performance (RTX 3090, `Qwen3.5-0.8B`, single sequence)

| Configuration                              | Prefill 512 tok | Recurrence only | KV cache at 4 K tokens |
|--------------------------------------------|----------------:|----------------:|-----------------------:|
| Portable op-by-op reference                |     1.00×       |         1.00×   |                  n/a   |
| Fused CUDA `gdn_recurrence_f32_k<K>`       |     ~1.8×       |        ~3.0×    |                  n/a   |
| Fused + register-resident K=128            |     ~5×         |        ~7.8×    |                  n/a   |

KV-cache memory scaling at context S, full-attention layers only, bf16:

| Cache representation | Bytes / K (full-attention K+V, all layers) | vs fp16 |
|----------------------|--------------------------------------------:|--------:|
| fp16 / bf16          |                                  ~112 MiB  |    1.0× |
| int8 (per-token)     |                                   ~63 MiB  |    ~0.56× |
| int4 (nibble-packed) |                                   ~35 MiB  |    ~0.31× |

Measured with the `attn_cache_bytes()` helper on `Qwen3_5TextModel`
(see `crane-core/src/models/qwen3_5/model.rs`); numbers are for the
Qwen3.5-0.8B architecture (24 layers, full-attention every 4th).
The GDN layers contribute a constant ~`O(L · K · V · num_heads)` regardless
of context size, so the full-attention cache dominates at long context —
which is why int4 helps the most on the 0.8B/4B tier.

`gdn_bench` (CUDA only) isolates the fused recurrence:

```
cargo run -p crane-core --release --features cuda --bin gdn_bench
# GDN recurrence  BH=16 S=512 K=128 V=128  ->  0.32 ms/iter   (X GFLOP/s)
```

### Known implementation quirks

- **`1/sqrt(K)` Q-scale inside the recurrence.** HF applies
  `query *= 1/sqrt(head_k_dim)` before the gated delta rule. Omitting it
  leaves recurrence output a factor of `sqrt(K)` too large *and* the error
  is not washed out downstream (gated RMSNorm's `eps` and the silu gate
  make it observable). Matches mistral.rs's reference.
- **Unit-offset RMSNorm.** `Qwen35RmsNorm` scales by `(1 + weight)`
  (Gemma-style), not `weight`. Plain candle `RmsNorm` would shrink every
  normalized activation ~5× and compound across layers.
- **Per-head `[query | gate]` split.** Full attention's q_proj must be
  viewed as `[B, S, heads, 2*head_dim]` then chunked on the head dim — not
  on the flat axis — otherwise `q_norm` sees scrambled (q, gate) halves.
- **Partial-rotary MRoPE rotates only the first `rot_dim` components.**
  Slice → rope → concat, matching `apply_rotary_pos_emb` in HF.
- **Causal mask required during prefill.** Built automatically when no
  mask is supplied (`qwen3_5::model::build_causal_mask`).
- **Hybrid caches are not interchangeable.** GDN carries a constant-size
  recurrent state per layer; full-attention carries the per-token K/V
  cache. The two are reset together via `Model::clear_kv_cache`.
- **`attn_output_gate: true`.** Qwen 3.5 applies a sigmoid gate to the
  attention output before `o_proj`. Honored.
- **Multi-id EOS.** Qwen 3.5 / Ornith declare `eos_token_id` as a list
  (e.g. `[248044, 248046]`). Read from `generation_config.json` (preferred)
  then `config.json`.

### Tool calling (Ornith)

The chat template renders a `# Tools` system block and expects
`tool_call{...} / tool` turns. `AutoTokenizer::apply_chat_template_with_tools`
exposes this with byte-identical output to HuggingFace's tokenizer
(`raise_exception` + Python-style `tojson` filters, `serde_json` with
`preserve_order`). See `example/src/ornith_tools.rs` for an end-to-end demo.

### Limitation

The `qwen3_5` backend caps `max_concurrent=1` — KV swap and batched decode
aren't implemented yet (hybrid layer types complicate a generic GPU-batched
implementation).

## Key files to read before changing things

- `crane-core/build.rs` — builds CUDA PTX when `cuda` is on; add
  `crane-core/kernels/<file>.cu` and rebuild.
- `crane-serve/src/engine/backend.rs` — per-model `ModelBackend` impls.
- `crane-serve/src/engine/model_factory.rs` — auto-detect + factory.
- `crane-core/src/models/qwen3_5/{model,modeling,kv_cache}.rs` and
  `crane-core/src/gdn/{backend,cuda_backend}.rs` for any Qwen 3.5 change.
