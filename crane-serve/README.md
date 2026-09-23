# crane-serve

An OpenAI & SGLang compatible inference API server built on the [Crane](../README.md) framework, with continuous batching support.

## Features

- **OpenAI-compatible API** — Chat Completions, Text Completions, Text-to-Speech, Models, Tokenize/Detokenize
- **SGLang native API** — `/generate`, `/model_info`, `/server_info` and related endpoints
- **Continuous batching** — Dedicated inference thread with prefill-priority scheduling, dynamic KV memory budget, and automatic sequence eviction/recovery
- **Multi-model support** — Auto-detects and loads Hunyuan Dense, Qwen 2.5, Qwen 3, Qwen 3.5 (hybrid GDN + softmax), **Bonsai 2 Ternary 27B (PTQ1_0/PQ2_0 GGUF)**, Qwen3-TTS, Voxtral TTS
- **Qwen3-TTS** — Full two-level TTS inference (Talker + Code Predictor) with native Candle speech-tokenizer decoder (ONNX optional fallback); exposes OpenAI-compatible `/v1/audio/speech`
- **Voxtral TTS** — 4B-parameter Mistral-based TTS with 20 multilingual voice embeddings, flow-matching acoustic model, and codec decoder; exposes OpenAI-compatible `/v1/audio/speech`
- **Tool / function calling** — OpenAI-shaped `tools`, `tool_calls` and `finish_reason: "tool_calls"`, streaming included; the prompt syntax comes from the model's own chat template
- **Reasoning control** — `enable_thinking` / `reasoning_effort` per request, with the `<think>` scratchpad separated out of `content` into `reasoning_content`
- **Streaming** — SSE (Server-Sent Events) token streaming
- **Cross-platform acceleration** — CPU / CUDA / Apple Metal, selected automatically

## Quick Start

### Build

```bash
# CPU only
cargo build -p crane-serve --release

# CUDA (NVIDIA GPU)
cargo build -p crane-serve --release --features cuda
```

### Run

```bash
# Auto-detect model type and device
crane --model-path /path/to/model

# Specify model type and port
crane --model-path /path/to/Qwen2.5-7B-Instruct \
    --model-type qwen25 \
    --port 8000

# GGUF weights
crane --model-path /path/to/model.gguf \
    --format gguf

# Bonsai 2 Ternary 27B — architecture and PTQ1_0/PQ2_0 are auto-detected
./target/release/crane-serve \
    --model-path checkpoints/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    --port 8000

# Force CPU
crane --model-path /path/to/model --cpu
```

### Built-in browser UI

The API server remains headless by default. Add `--ui` to serve the built-in
React interface at the server root, without starting a separate frontend
process:

```bash
crane-serve --model-path /path/to/model --ui
```

The UI selects its view from the loaded model: ASR models show an audio upload
and transcription screen; language models show chat; vision-language models
also expose an image attachment control. The frontend source lives in `ui/`.
After changing it, rebuild its embedded production assets with
`npm run build --prefix crane-serve/ui` before building `crane-serve`.

Microphone transcription requires a secure browser origin: use
`http://localhost:<port>` when opening the UI on the server machine, or HTTPS
when accessing it through a LAN IP or hostname. Browsers intentionally block
microphone access from plain HTTP remote origins.

## Text-to-Speech and Speech Recognition

Crane supports Qwen3-TTS, Voxtral TTS, and Qwen3-ASR, all exposed through
OpenAI-compatible endpoints. See
[Audio: Text-to-Speech and Speech Recognition](docs/audio.md) for setup,
voice lists, generation parameters, and troubleshooting.

## GPU Deployment

Build with `--features cuda`, control VRAM usage, and use GGUF
quantized models on GPU: see [GPU Deployment](docs/gpu.md).

## CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | *(required)* | Path to model directory or GGUF file |
| `--model-type` | `auto` | Architecture: `auto`, `hunyuan`, `qwen25`, `qwen3`, `qwen3_5`, `qwen3_5_vl`, `qwen3_tts`, `voxtral_tts` (aliases: `voxtral`, `voxtral-tts`, `voxtral_4b`) |
| `--model-name` | directory name | Model name shown in API responses |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8080` | Bind port |
| `--cpu` | `false` | Force CPU even when a GPU is available |
| `--max-concurrent` | `16` | Hard cap on concurrently decoding sequences. Actual concurrency may be lower when `--gpu-memory-limit` is active. |
| `--decode-tokens-per-seq` | `16` | Max decode rounds per scheduling step. Higher = less scheduling overhead, higher TTFT for queued requests. |
| `--format` | `auto` | Weight format: `auto`, `safetensors`, `gguf` |
| `--max-seq-len` | `0` | Max sequence length (prompt + generation); `0` = unlimited |
| `--gpu-memory-limit` | *(none)* | VRAM cap: absolute (`5G`, `8G`, `5120M`) or fractional (`0.7` = 70% of total) |
| `--text-only` | `false` | Qwen 3.5-VL / Ornith only: opt out of the vision tower and load the same checkpoint as a plain text model (skips the ~600M-param ViT entirely — no extra VRAM — and unlocks `--quant`, which the VLM path doesn't support). Vision-capable checkpoints load with vision by default; this flag is the opt-out. |
| `--api-key` | *(none)* | API key required for non-exempt endpoints; repeatable to configure multiple valid keys. Pass with no value (`--api-key`) to generate a random key printed to stdout at startup. Also settable via `CRANE_API_KEY` (always requires a value there). Unset means open access. |
| `--api-key-file` | *(none)* | File with one API key per line (`#`-prefixed lines are comments); combines with `--api-key`. Also settable via `CRANE_API_KEY_FILE`. |

### Parameter tuning guide

| Goal | Recommendation |
|------|----------------|
| Constrained VRAM (≤12 GB) | Set `--gpu-memory-limit`; use `--max-concurrent 4–8` as a safety ceiling |
| Maximum throughput | Increase `--decode-tokens-per-seq` to `32` to reduce scheduling round-trips |
| Lowest time-to-first-token | Decrease `--decode-tokens-per-seq` to `4–8` so prefill slots in sooner |
| Long context generation | Set `--max-seq-len` to avoid unbounded KV growth |

### Authentication

Disabled by default (open access). Configure `--api-key` (repeatable) and/or
`--api-key-file` to require a valid key on every request:

```bash
crane-serve --model-path /path/to/model --api-key sk-mysecretkey

# Generate a random key at startup instead of choosing one
crane-serve --model-path /path/to/model --api-key
```

Once at least one key is configured, requests must present it via either
header:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-mysecretkey" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Hi"}]}'

# or
curl http://localhost:8080/v1/chat/completions \
  -H "X-Api-Key: sk-mysecretkey" \
  ...
```

A missing or invalid key returns `401` with an OpenAI-shaped error body.
`/health`, `/v1/stats`, `/` and `/ui/*` stay reachable without a key so
monitoring and the browser UI shell keep working — note the UI's own API
calls still require one, so `--ui` combined with `--api-key` isn't usable
from the browser today. Binding `--host 0.0.0.0` with an API key configured
logs a startup warning, since keys travel in cleartext without TLS in front
of the server.

## API Reference

Full endpoint documentation, request/response fields, and curl examples
for the OpenAI-compatible, SGLang-compatible, and management endpoints:
see [API Reference](docs/api-reference.md).

## Using the OpenAI SDK

crane-serve implements the OpenAI API shape, so the official `openai`
Python package works unmodified. See
[Using the OpenAI SDK](docs/openai-sdk.md) for chat, tool calling, and
TTS examples.

## Source Structure

```
crane-serve/src/
├── main.rs              # CLI entry point, AppState, route registration
├── openai_api.rs        # OpenAI request/response types (incl. SpeechRequest)
├── sglang_api.rs        # SGLang native API types
├── chat_template.rs     # Chat template rendering (Jinja / Hunyuan hard-coded)
├── reasoning.rs         # Thinking control + <think> / content separation
├── tools.rs             # Tool-call parsing (incl. streaming filter)
├── handlers/
│   ├── common.rs        # /health, /v1/stats
│   ├── openai.rs        # OpenAI endpoint handlers
│   ├── sglang.rs        # SGLang endpoint handlers
│   ├── tts.rs           # /v1/audio/speech handler (Qwen3-TTS, Voxtral TTS)
│   ├── vlm.rs           # VLM handler (PaddleOCR-VL)
│   └── sse.rs           # SSE stream builder
└── engine/
    ├── mod.rs           # InferenceEngine core loop (continuous batching)
    ├── types.rs         # EngineRequest / EngineResponse / EngineHandle
    ├── stats.rs         # Lock-free atomic counters
    ├── sampling.rs      # Token sampling (top-k/p, Gumbel-max, repetition penalty)
    ├── scheduler.rs     # Prefill-priority scheduler (dynamic effective_max_running cap)
    ├── sequence.rs      # Sequence lifecycle management
    ├── backend.rs       # ModelBackend trait and per-model implementations
    └── model_factory.rs # Model auto-detection and factory (incl. Qwen3TTS, VoxtralTTS)
```

## Model Backend Support

| Model | Batch decode | KV Swap | Formats | Notes |
|-------|-------------|---------|---------|-------|
| Hunyuan Dense | ✅ | ✅ | Safetensors / GGUF | KV pre-alloc, GQA 4D matmul, RoPE cache growth |
| Qwen 3 | ✅ | ✅ | Safetensors / GGUF | + QK Norm 4D, GGUF quantization |
| Qwen 3.5 / 3.6 / 3.8 | ❌ | ❌ | Safetensors / GGUF | Hybrid GDN + softmax attention; CUDA fused recurrence kernel; `max_concurrent=1`. Qwen 3.6-27B and 3.8-27B are the same architecture scaled up and load on this path (they declare `model_type: "qwen3_5"`). Tool calling and reasoning control supported. |
| Qwen 2.5 | sequential | ❌ | Safetensors | — |
| **Qwen3-TTS** | N/A | N/A | Safetensors (ONNX fallback optional) | Dedicated thread; no continuous batching; voice cloning supported |
| **Voxtral TTS** | N/A | N/A | Safetensors | Dedicated thread; 37-codebook codec; no voice cloning |

Model type is auto-detected from `config.json` / `params.json` (`model_type` / `architectures`), from a `.gguf` header's `general.architecture`, or can be set explicitly with `--model-type`.

Tool calling and reasoning control are **template-driven**, not model-type-driven: any checkpoint whose chat template defines a tool protocol and/or a `<think>` block gets them, and a template that defines neither simply ignores the corresponding request fields.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CRANE_SAMPLE_TRACE` | `0` | Verbose sampling timing logs |
| `CRANE_KV_QUANT` | unset | Qwen 3.5 family K/V cache: `int8` (~2x smaller) or `int4` (~4x smaller) |
| `CRANE_EMBED_DENSE` | `0` | GGUF: dequantize the whole embedding table at load instead of gathering rows (pre-optimization behaviour; costs ~1.7 GiB on Qwen 3.8-27B) |
| `CRANE_PROF` | `0` | Per-forward-pass profiler: splits kernel *submission* time from wall time after a device sync |

GPU-specific environment variables (`CRANE_FORCE_GPU_TOPK`,
`CRANE_TOPP_FALLBACK_TOPK`, `CRANE_TOPK_SAMPLE_ON_CPU`) are documented in
[GPU Deployment](docs/gpu.md).

## Notes

- **API key authentication is opt-in** — unset by default (open access); see [Authentication](#authentication) for `--api-key`/`--api-key-file`.
- **KV eviction is lossless** — Evicted sequences preserve their full state and resume automatically; in-flight requests are not dropped or errored.
- **`--max-seq-len 0`** means no limit. On constrained hardware, always set an explicit value to avoid runaway memory growth.
- **`--decode-tokens-per-seq`** controls decode rounds per engine step, not per request. Requests always complete fully regardless of this value.
- **Qwen3-TTS and Voxtral TTS run on a dedicated thread** — No continuous batching; each `/v1/audio/speech` request is processed sequentially. Concurrent requests are queued in an unbounded channel.
- **Qwen3-TTS decoder backend** — The speech-tokenizer decoder (codes → waveform) uses native Candle by default. ONNX export is optional as a compatibility fallback.
- **Voxtral TTS uses greedy decoding** — `temperature`, `top_p`, and `repetition_penalty` are accepted for API compatibility but do not affect the output.

## Testing

```bash
# All unit tests
cargo test -p crane-serve
cargo test -p crane-core

# Specific modules
cargo test -p crane-serve engine::scheduler
cargo test -p crane-serve openai_api::tests
cargo test -p crane-serve sglang_api::tests
cargo test -p crane-serve tools          # tool-call parsing + streaming filter
cargo test -p crane-serve reasoning      # <think> separation, thinking options
cargo test -p crane-core autotokenizer
```

Tool calling and reasoning control are additionally checked against a real
checkpoint's own chat template. These need a local model, so they are
`#[ignore]`d and read it from an env var:

```bash
CRANE_QWEN38_MODEL=/path/to/Qwen3.8-27B-Q4_K_M.gguf \
  cargo test -p crane-serve --test tool_calling -- --ignored --nocapture
CRANE_QWEN38_MODEL=/path/to/Qwen3.8-27B-Q4_K_M.gguf \
  cargo test -p crane-serve --test thinking_control -- --ignored --nocapture
```

`tool_calling` renders a full agentic turn (tools out, assistant `tool_calls`
replayed, `<tool_response>` back in) through the model's template, which is
what catches a template contract change that unit tests with a mock cannot.

## License

MIT
