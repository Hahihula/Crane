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

## CUDA Usage

> **Note:** CUDA support requires the `cuda` feature flag at build time (see above). The server automatically uses the first available CUDA device.

### Basic CUDA inference

```bash
crane-serve --model-path /path/to/Qwen3-8B-Instruct
```

On CUDA, `model_info` will report the device as `Cuda(0)` (or `Cuda(1)`, etc.).

### GPU memory control

GPU memory grows as KV caches accumulate. Use `--gpu-memory-limit` to keep usage bounded:

```bash
# Hard cap at 8 GB — recommended starting point for a 12 GB GPU
crane-serve --model-path /path/to/model \
    --gpu-memory-limit 8G \
    --max-seq-len 4096

# Cap at 5 GB for 8 GB VRAM cards
crane-serve --model-path /path/to/model \
    --gpu-memory-limit 5G \
    --max-seq-len 2048 \
    --max-concurrent 4

# Use 75% of total VRAM
crane-serve --model-path /path/to/model \
    --gpu-memory-limit 0.75
```

When the KV memory budget is exceeded, the engine evicts the longest-output sequence (preserving its state), tightens the concurrency cap, and resumes that sequence automatically once load subsides. This avoids OOM without crashing the server.

**Recommended values by GPU size:**

| GPU VRAM | `--gpu-memory-limit` | `--max-seq-len` |
|----------|---------------------|----------------|
| 8 GB     | `6G` or `0.7`       | `2048`         |
| 12 GB    | `8G` or `0.7`       | `4096`         |
| 24 GB    | `20G` or `0.8`      | `8192`         |
| 48 GB+   | *(omit)*            | *(omit)*       |

### GGUF quantized models on CUDA

GGUF quantization roughly halves VRAM usage compared to FP16:

```bash
crane-serve --model-path /path/to/Qwen3-8B-Q4_K_M.gguf \
    --format gguf \
    --gpu-memory-limit 8G
```

### Multi-GPU note

Currently crane-serve runs on a single CUDA device (device 0). Multi-GPU tensor parallelism is not yet supported.

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

### OpenAI-compatible

#### `POST /v1/audio/speech` — Text-to-Speech

Synthesizes speech from text using either Qwen3-TTS or Voxtral TTS. Returns a WAV audio file (or raw PCM, depending on `response_format`).

```bash
# Voxtral TTS — German female voice
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxtral",
    "input": "Guten Morgen, wie geht es Ihnen?",
    "voice": "de_female",
    "language": "german",
    "response_format": "wav"
  }' \
  --output speech.wav

# Qwen3-TTS — Chinese
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "今天天气真好，我们去公园吧。",
    "voice": "Chelsie",
    "language": "chinese"
  }' \
  --output speech.wav

# Qwen3-TTS — English with higher temperature
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "Hello! This is Crane, an ultra-fast inference framework written in Rust.",
    "voice": "Chelsie",
    "language": "english",
    "temperature": 0.8,
    "max_tokens": 2048
  }' \
  --output hello.wav

# Qwen3-TTS — Voice cloning (Base model)
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "そんな何もない今日が 少しだけでもいい日になったと思えたら",
    "language": "japanese",
    "reference_audio": "data/audio/kinsenka_3.wav",
    "reference_text": "こうして君に直接ありがとうを言える時間をくれたこと それが多分一番私は嬉しい"
  }' \
  --output voice_clone.wav

# Auto-detect language
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "This is a bilingual test. 这是一个中英文混合测试。",
    "language": "auto"
  }' \
  --output bilingual.wav
```

**Request fields:**

| Field | Type | Default | Qwen3-TTS | Voxtral TTS | Description |
|-------|------|---------|-----------|-------------|-------------|
| `model` | string | — | ✅ | ✅ | Model name (e.g. `"Qwen3-TTS"`, `"voxtral"`) |
| `input` | string | — | ✅ | ✅ | The text to synthesize. UTF-8, up to a few thousand characters |
| `voice` | string | `null` | ✅ speaker name | ✅ embedding name | Qwen3: speaker from `config.json` (e.g. `"Serena"`). Voxtral: voice embedding (e.g. `"de_female"`). `null` uses default |
| `language` | string | `"auto"` | ✅ used | accepted | Language hint: `"english"`, `"german"`, `"french"`, `"chinese"`, `"japanese"`, etc. |
| `instructions` | string | `null` | ✅ | accepted | Optional system-level prompt to guide speaking style |
| `response_format` | string | `"wav"` | ✅ | ✅ | Output audio format: `"wav"` or `"pcm"` (raw 16-bit LE at 24 kHz). `"mp3"`, `"opus"`, `"aac"`, `"flac"` return `400` |
| `speed` | float | `1.0` | reserved | reserved | Speaking speed multiplier (reserved, not yet applied) |
| `temperature` | float | `0.9` | ✅ used | accepted (no effect) | Sampling temperature. Lower = more deterministic (Qwen3 only) |
| `top_p` | float | `null` | ✅ used | accepted (no effect) | Nucleus sampling threshold; `null` or `1.0` disables filtering (Qwen3 only) |
| `repetition_penalty` | float | `1.05` | ✅ used | accepted (no effect) | Repetition penalty for codec token generation (Qwen3 only) |
| `max_tokens` | int | `8192` | ✅ | ✅ | Max codec tokens. Qwen3: ~83 ms/token at 12 Hz. Voxtral: ~80 ms/frame at 12.5 Hz |
| `reference_audio` | string | `null` | ✅ Base model only | ❌ not supported | Local path to reference WAV for voice cloning |
| `reference_text` | string | `null` | ✅ Base model only | ❌ not supported | Transcript of the reference audio (required with `reference_audio`) |

**Response:** Binary audio bytes with `Content-Type: audio/wav` (`response_format="wav"`) or `audio/pcm` (`response_format="pcm"`).

**Approximate duration cap:** `max_tokens / 12` seconds (e.g. `8192` tokens ≈ 683 seconds, `2048` ≈ 171 seconds).

#### `POST /v1/chat/completions`

```bash
# Non-streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'

# Streaming (SSE)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Beyond the standard OpenAI fields, the chat endpoint accepts:

| Field | Meaning |
|:------|:--------|
| `tools` | Function specs. Passed to the model's chat template verbatim, so the prompt syntax is whatever that model was trained on. |
| `tool_choice` | `"none"` renders no tool block. Other values are accepted but **advisory** — forcing a particular call would need constrained decoding, which the engine does not implement. |
| `chat_template_kwargs` | Extra template variables (vLLM/SGLang convention), e.g. `{"enable_thinking": false}`. |
| `reasoning_effort` | OpenAI's top-level budget. `chat_template_kwargs` wins if both set it. |

#### Tool / function calling

Supported for any model whose chat template defines a tool protocol — the
Qwen 3.5 / 3.6 / 3.8 and Ornith families all do, and share one syntax
(`<tool_call><function=NAME><parameter=KEY>…`). crane-serve parses that back
into OpenAI's `tool_calls`, so clients need no model-specific handling.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "id": "call_0",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

Run the tool, then send the result back as a `tool` message. **Echo the
assistant's `tool_calls` turn back too** — the template re-renders it into the
transcript, and without it the model cannot see that it already called the tool
and will call it again:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris right now?"},
      {"role": "assistant", "content": null, "tool_calls": [
        {"id": "call_0", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}
      ]},
      {"role": "tool", "tool_call_id": "call_0", "name": "get_weather",
       "content": "{\"temperature_c\": 18, \"conditions\": \"cloudy\"}"}
    ],
    "tools": [{"type": "function", "function": {"name": "get_weather",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
      "required": ["city"]}}}]
  }'
# → "The current weather in Paris is: Temperature 18°C, Conditions: Cloudy"
```

Notes:

- `arguments` is a **JSON-encoded string**, per the OpenAI wire format — parse
  it client-side. Values that look like JSON scalars are recovered as such, so
  a numeric argument arrives as `{"limit": 5}` rather than `{"limit": "5"}`.
- Call `id`s are synthesized (`call_0`, `call_1`, …) because the template does
  not emit them; they are stable within one response, which is all
  `tool_call_id` correlation needs.
- **Streaming**: tool-call markup never appears in `content` deltas. A complete
  call is emitted as a single `tool_calls` delta before the terminal chunk,
  because a partially-streamed call is not something a client can safely run.
- If generation stops mid-call (token limit), the fragment is **discarded**
  rather than half-parsed, and `finish_reason` stays `length`.

#### Reasoning control (Qwen 3.5 / 3.6 / 3.8)

Reasoning models emit a `<think>` scratchpad. crane-serve splits it out of
`content` into `reasoning_content` (streaming too), so the answer alone is what
a client displays.

```bash
# Thinking on (the family default) — answer in content, scratchpad separate
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "What is 2+2?"}]}'
# → {"message": {"content": "4", "reasoning_content": "Two plus two is four."}}

# Shorter thinking — much lower latency
  -d '{..., "reasoning_effort": "low"}'

# Off entirely
  -d '{..., "chat_template_kwargs": {"enable_thinking": false}}'
```

`reasoning_effort` takes `low`, `medium` or `xhigh`. Qwen 3.8 defaults to
**`xhigh`**, the longest budget, so a modest `max_tokens` can be consumed
entirely by the scratchpad. Note `medium` is the neutral baseline and injects
no instruction — only `low` and `xhigh` add one.

#### Multimodal / Vision (PaddleOCR-VL-1.5)

For VLM requests, use an array in `content` to provide the image URL and the prompt text:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "paddleocr_vl-1.5",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://i0.hdslb.com/bfs/new_dyn/1824ac967aca31d7ac9da4fdda678c4639471072.png"}},
          {"type": "text", "text": "OCR:"}
        ]
      }
    ],
    "max_tokens": 1024
  }'
```

**Request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | — | Model name |
| `messages` | array | — | `[{role, content}]`. `content` can be a string, or an array of `{"type": "text", "text": "..."}` and `{"type": "image_url", "image_url": {"url": "..."}}` items for VLM models. |
| `max_tokens` | int | `512` | Max tokens to generate |
| `temperature` | float | `0.8` | Sampling temperature; `0` = greedy |
| `top_p` | float | `0.95` | Nucleus sampling threshold |
| `top_k` | int | `40` | Top-k sampling |
| `repetition_penalty` | float | `1.05` | Repetition penalty |
| `stream` | bool | `false` | Enable SSE streaming |
| `stream_options` | object | — | `{"include_usage": true}` to include token counts in the final chunk |
| `seed` | int | — | Random seed for reproducibility |

#### `POST /v1/completions`

Raw text completion (no chat template applied).

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-7B-Instruct", "prompt": "The capital of France is", "max_tokens": 64}'
```

`prompt` accepts a single string or an array of strings (concatenated).

#### `GET /v1/models` · `GET /v1/models/:model_id`

List available models or fetch metadata for a specific one.

#### `POST /v1/tokenize` · `POST /v1/detokenize`

```bash
# Text → token IDs
curl http://localhost:8080/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'

# Token IDs → text
curl http://localhost:8080/v1/detokenize \
  -H "Content-Type: application/json" \
  -d '{"tokens": [9707, 1917]}'
```

### SGLang-compatible

#### `POST /generate`

```bash
curl http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The meaning of life is",
    "sampling_params": {"max_new_tokens": 128, "temperature": 0.8, "top_p": 0.95}
  }'
```

#### Multimodal / Vision (PaddleOCR-VL-1.5)

To run a multimodal inference request with a VLM (PaddleOCR-VL), include the `image_url` parameter:

```bash
curl http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "OCR:",
    "image_url": "https://i0.hdslb.com/bfs/new_dyn/1824ac967aca31d7ac9da4fdda678c4639471072.png",
    "sampling_params": {"max_new_tokens": 1024}
  }'
```

**`sampling_params` fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_new_tokens` | int | `128` | Max tokens to generate |
| `temperature` | float | `0.8` | Sampling temperature |
| `top_p` | float | `0.95` | Nucleus sampling |
| `top_k` | int | `20` | Top-k sampling |
| `repetition_penalty` | float | `1.0` | Repetition penalty |
| `stop` | string/array | — | Stop string(s) |
| `stop_token_ids` | array | — | Stop token IDs |
| `seed` | int | — | Random seed |
| `n` | int | `1` | Number of parallel completions |

#### `GET /model_info`

Returns model metadata including device (`Cuda(0)`, `Metal(0)`, `Cpu`).

#### `GET /server_info`

Returns server config and live engine stats.

```json
{
  "version": "0.1.0",
  "model_path": "/models/Qwen2.5-7B-Instruct",
  "max_concurrent": 16,
  "decode_tokens_per_seq": 16,
  "stats": {
    "total_requests": 42,
    "completed_requests": 40,
    "avg_decode_tokens_per_sec": 35.2,
    "active_sequences": 2,
    "waiting_sequences": 0
  }
}
```

#### `GET /health_generate`

Deep health check — runs a 1-token inference probe with a 30-second timeout.

#### `POST /abort_request`

Cancel an in-flight request by ID.

```bash
curl http://localhost:8080/abort_request \
  -H "Content-Type: application/json" \
  -d '{"rid": "gen-xxxx-xxxx"}'
```

### Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check, returns `{"status": "ok"}` |
| `/v1/stats` | GET | Engine stats snapshot (requests, throughput, active sequences) |
| `/flush_cache` | POST | KV cache flush (reserved for compatibility; caches are managed automatically) |

## Using with OpenAI SDK

### Chat (LLM)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",  # only checked if --api-key/--api-key-file is configured
)

response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Tool calling (LLM)

The standard OpenAI agentic loop works unmodified:

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

def get_weather(city):
    return {"temperature_c": 18, "conditions": "cloudy"}

messages = [{"role": "user", "content": "What is the weather in Paris?"}]

while True:
    reply = client.chat.completions.create(
        model="qwen3.8-27b", messages=messages, tools=TOOLS,
    ).choices[0].message

    if not reply.tool_calls:
        print(reply.content)
        break

    # Append the assistant turn *including* tool_calls — the template replays
    # it into the transcript, and dropping it makes the model call again.
    messages.append(reply.model_dump(exclude_none=True))
    for call in reply.tool_calls:
        args = json.loads(call.function.arguments)   # arguments is a JSON string
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.function.name,
            "content": json.dumps(get_weather(**args)),
        })
```

Reasoning models additionally expose the scratchpad on the message:

```python
reply = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    reasoning_effort="low",                       # low | medium | xhigh
    # or: extra_body={"chat_template_kwargs": {"enable_thinking": False}},
).choices[0].message

print(reply.content)                              # "4"
print(getattr(reply, "reasoning_content", None))  # the <think> scratchpad
```

### Text-to-Speech

#### Voxtral TTS

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

# German female voice
response = client.audio.speech.create(
    model="voxtral",
    voice="de_female",
    input="Hallo, wie geht es Ihnen heute?",
    extra_body={"language": "german", "response_format": "wav"},
)
response.stream_to_file("output.wav")

# English neutral voice
response = client.audio.speech.create(
    model="voxtral",
    voice="neutral_female",
    input="Hello, this is a test of the Voxtral TTS engine.",
    extra_body={"response_format": "wav"},
)
response.stream_to_file("english.wav")
```

#### Qwen3-TTS

> Start crane-serve with `--model-type qwen3_tts` and a Qwen3-TTS checkpoint.

**CustomVoice model (predefined speakers):**

```python
from openai import OpenAI
import pathlib

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

# --- Basic synthesis (returns WAV bytes) ---
response = client.audio.speech.create(
    model="Qwen3-TTS",
    voice="Chelsie",
    input="今天天气真好，我们去公园吧！",
    extra_body={"language": "chinese"},
)
response.stream_to_file("output.wav")

# --- English ---
response = client.audio.speech.create(
    model="Qwen3-TTS",
    voice="Ethan",
    input="Hello, this is a test of the Crane TTS engine.",
    extra_body={"language": "english", "temperature": 0.7},
)
response.stream_to_file("english.wav")
```

**Base model (voice cloning):**

```python
from openai import OpenAI
import pathlib

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

# --- Voice clone: synthesize new text in the reference speaker's voice ---
response = client.audio.speech.create(
    model="Qwen3-TTS",
    voice="clone",  # voice field is ignored in voice-clone mode
    input="そんな何もない今日が 少しだけでもいい日になったと思えたら",
    extra_body={
        "language": "japanese",
        "reference_audio": "data/audio/kinsenka_3.wav",
        "reference_text": "こうして君に直接ありがとうを言える時間をくれたこと それが多分一番私は嬉しい",
    },
)
response.stream_to_file("voice_clone.wav")
```

**Low-level requests (requests library):**

```python
import requests, pathlib

# Voxtral TTS
r = requests.post(
    "http://localhost:8080/v1/audio/speech",
    json={
        "model": "voxtral",
        "input": "Hello, how are you today?",
        "voice": "neutral_female",
        "response_format": "wav",
    },
)
r.raise_for_status()
pathlib.Path("speech.wav").write_bytes(r.content)

# Qwen3-TTS CustomVoice
r = requests.post(
    "http://localhost:8080/v1/audio/speech",
    json={
        "model": "Qwen3-TTS",
        "input": "今天天气真好，我们去公园吧！",
        "voice": "Chelsie",
        "language": "chinese",
        "temperature": 0.7,
        "max_tokens": 2048,
    },
)
r.raise_for_status()
pathlib.Path("speech.wav").write_bytes(r.content)

# Qwen3-TTS voice clone
r = requests.post(
    "http://localhost:8080/v1/audio/speech",
    json={
        "model": "Qwen3-TTS",
        "input": "こんにちは、今日はいい天気ですね。",
        "language": "japanese",
        "reference_audio": "data/audio/kinsenka_3.wav",
        "reference_text": "Reference transcript here",
        "max_tokens": 2048,
    },
)
r.raise_for_status()
pathlib.Path("voice_clone.wav").write_bytes(r.content)
```

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
| `CRANE_FORCE_GPU_TOPK` | `0` | Force GPU top-k even for large vocabularies |
| `CRANE_TOPP_FALLBACK_TOPK` | `64` | k value for GPU top-k fallback |
| `CRANE_TOPK_SAMPLE_ON_CPU` | `0` | Sample on CPU after GPU top-k |
| `CRANE_SAMPLE_TRACE` | `0` | Verbose sampling timing logs |
| `CRANE_KV_QUANT` | unset | Qwen 3.5 family K/V cache: `int8` (~2x smaller) or `int4` (~4x smaller) |
| `CRANE_EMBED_DENSE` | `0` | GGUF: dequantize the whole embedding table at load instead of gathering rows (pre-optimization behaviour; costs ~1.7 GiB on Qwen 3.8-27B) |
| `CRANE_PROF` | `0` | Per-forward-pass profiler: splits kernel *submission* time from wall time after a device sync |

## Notes

- **API key authentication is opt-in** — unset by default (open access); see [Authentication](#authentication) for `--api-key`/`--api-key-file`.
- **Single CUDA device** — The server uses CUDA device 0. Multi-GPU tensor parallelism is not yet supported.
- **KV eviction is lossless** — Evicted sequences preserve their full state and resume automatically; in-flight requests are not dropped or errored.
- **`--max-seq-len 0`** means no limit. On constrained hardware, always set an explicit value to avoid runaway memory growth.
- **GGUF quantization** is supported for Hunyuan Dense and Qwen 3. Qwen 2.5 requires Safetensors format.
- **`--decode-tokens-per-seq`** controls decode rounds per engine step, not per request. Requests always complete fully regardless of this value.
- **Log diagnostics** — The startup log prints `kv_bytes` and `kv_budget`. Monitor these to validate your `--gpu-memory-limit` headroom.
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
