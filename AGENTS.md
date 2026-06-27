# AGENTS.md — Crane

Crane is a Rust workspace (`edition = "2024"`) that runs LLM/MLLM/TTS inference on top of the `candle` framework. There are **no CI workflows, no `clippy.toml`, no `rustfmt.toml`, no pre-commit config** — verify your changes by building and running tests yourself.

## Workspace layout

The Cargo workspace at `Cargo.toml` has **four members**:

| Crate | Path | Role |
|---|---|---|
| `crane-core` | `crane-core/` | Library: model implementations, autotokenizer, fused CUDA kernels, generation utilities. `pub mod` list lives in `crane-core/src/models/mod.rs`. |
| `crane` | `crane/` | High-level SDK (`pub mod chat/vision/audio/multimodal/llm/common`, prelude in `src/lib.rs`). Its `src/main.rs` is a thin wrapper that calls `crane_serve::run(...)`. |
| `crane-serve` | `crane-serve/` | OpenAI & SGLang compatible HTTP server built on axum + a continuous-batching inference engine (`src/engine/`). |
| `crane-examples` | `example/` | Demo binaries — `chat_simple`, `chat_streaming`, `asr_simple`, `vision_simple`, `ocr_simple`, `bm_resize`, `tts_simple`, `tts_custom_voice`, `tts_voice_clone`, `gemma4_simple`, `hunyuan_simple`. Bin targets are declared in `example/Cargo.toml`. |

## ⚠️ The README is partially stale — the package was renamed

Top-level `README.md` and `crane-serve/README.md` still reference **`crane-oai`** throughout (binary name, package name, `cargo build -p crane-oai`, etc.). **There is no `crane-oai` package or binary in the source.** After the rename:

- Server package is `crane-serve` (`cargo build -p crane-serve`).
- Built server binary is `target/release/crane-serve` (or `target/release/crane` via the wrapper).
- Tests: `cargo test -p crane-serve`, `cargo test -p crane-core`.
- The old `target/release/crane-oai` artifact is a leftover from a previous build and is **not** produced by current sources.

If a command from the README says `crane-oai`, translate it to `crane` or `crane-serve`.

## Build

```bash
# CPU (default)
cargo build --release

# NVIDIA GPU — also pulls in bindgen_cuda and compiles crane-core/kernels/*.cu
cargo build --release --features cuda

# Apple Silicon — Metal is enabled automatically via target-specific deps
# in crane-core/Cargo.toml; no feature flag needed.
```

Feature flags live in `crane-core/Cargo.toml` and propagate through `crane/Cargo.toml` (`cuda`, `cudnn`, `mkl`, `onnx`). The `onnx` feature gates the `moonshine_asr`, `silero_vad`, and `snac_onnx` modules — required for the `asr_simple` example: `cargo run -p crane-examples --features onnx --bin asr_simple --release`.

`install.sh` auto-detects macOS / Linux+CUDA / Linux-CPU and runs the right `cargo build -p crane --features ...`. It writes the binary to `target/release/crane`.

## Run the server

```bash
cargo run -p crane --release -- --model-path /path/to/Qwen2.5-7B-Instruct
# or directly:
cargo run -p crane-serve --release -- -m /path/to/model -p 8080
```

The CLI is defined in `crane-serve/src/lib.rs` (`Args`). Model type is auto-detected from `<model>/config.json` (`model_type` / `architectures`); override with `--model-type {hunyuan,qwen25,qwen3,qwen3_tts,paddleocr_vl,gemma4,gemma4_vl}`. See `crane-serve/src/engine/model_factory.rs:31` (`ModelType::from_str`).

GPU memory control: `--gpu-memory-limit 8G` (or `0.7` for 70% of VRAM) + `--max-seq-len`. When the KV budget is exceeded the engine preempts the longest-output sequence and re-prefills later — see `InferenceEngine::evict_if_needed` in `crane-serve/src/engine/mod.rs`.

## Tests

```bash
cargo test -p crane-serve    # engine/scheduler/stats/sequence/model_factory + openai/sglang API types
cargo test -p crane-core     # autotokenizer, generation
```

Unit tests live next to the code: `crane-serve/src/engine/{stats,sequence,types,model_factory}.rs`, `crane-serve/src/{openai_api,sglang_api}.rs`, `crane-core/src/{autotokenizer,generation/based}.rs`. No integration tests, no doctests on private APIs.

`scripts/test_*.sh` and `tests/test_cv*.py` are **manual** model-output sanity scripts (e.g. compare against PyTorch), not a CI test suite.

## Adding a new model

Per the top-level README contribution guide, for a new architecture drop the implementation into `crane-core/src/models/<name>/` with its own `mod.rs`, and register the module in `crane-core/src/models/mod.rs`. Reference impls: `crane-core/src/models/qwen25/` (text-only, sequential decode), `crane-core/src/models/qwen3/` (full features), `crane-core/src/models/qwen3_tts/` (multi-stage pipeline). For a new multimodal arch that composes existing pieces, see `crane-core/src/models/hunyuanocr/` and the VLM wiring in `crane-serve/src/engine/backend.rs`.

## Key files to read before changing things

- `crane-core/build.rs` — builds CUDA PTX when `cuda` is on; add `crane-core/kernels/<file>.cu` and rebuild.
- `crane-serve/src/lib.rs` — server entry point, route table, `AppState`.
- `crane-serve/src/engine/mod.rs` — continuous-batching loop, KV eviction, memory gate.
- `crane-serve/src/engine/backend.rs` — per-model `ModelBackend` impls (batch decode, KV swap support differ per model).
- `crane-serve/src/engine/model_factory.rs` — auto-detect + factory.
- `crane-serve/src/handlers/{openai,sglang,tts,vlm,sse}.rs` — HTTP handlers.

## Environment variables

All four control sampling behavior in the inference engine (see `crane-serve/src/engine/sampling.rs`):

- `CRANE_FORCE_GPU_TOPK` (default `0`)
- `CRANE_TOPP_FALLBACK_TOPK` (default `64`)
- `CRANE_TOPK_SAMPLE_ON_CPU` (default `0`)
- `CRANE_SAMPLE_TRACE` (default `0`) — verbose sampling timing logs.

## Conventions worth knowing

- TTS models (`qwen3_tts`) run on a **dedicated std::thread**, not the continuous-batching engine. VLM requests are routed via separate mpsc channels (PaddleOCR-VL → `vlm_tx`, Gemma4-VL → `gemma4_vlm_tx`, Qwen3-TTS → `tts_tx`) in `crane-serve/src/lib.rs:run`.
- `crane-core` uses `minijinja` for chat-template rendering; Hunyuan has a hard-coded template in `crane-serve/src/chat_template.rs`.
- The repo checks in `Cargo.lock` (root) but the `.gitignore` lists `/target`, `checkpoints/`, `vendor/`, `*.onnx`, `outputs/`, `*.bin`. Models and ONNX files are not in git.
- Output from TTS examples lands in `data/audio/output/`; raw audio clips in `data/audio/` are not all gitignored (only a few regenerated ones in `data/.gitignore`).