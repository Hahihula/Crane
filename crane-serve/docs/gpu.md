# GPU Deployment

CUDA support requires the `cuda` feature flag at build time:

```bash
cargo build -p crane-serve --release --features cuda
```

The server automatically uses the first available CUDA device.

## Basic CUDA inference

```bash
crane-serve --model-path /path/to/Qwen3-8B-Instruct
```

`model_info` reports the device as `Cuda(0)` (or `Cuda(1)`, etc.).

## GPU memory control

GPU memory grows as KV caches accumulate. Use `--gpu-memory-limit` to keep
usage bounded:

```bash
# Hard cap at 8 GB — recommended starting point for a 12 GB GPU
crane-serve --model-path /path/to/model \
    --gpu-memory-limit 8G \
    --context 4K

# Cap at 5 GB for 8 GB VRAM cards
crane-serve --model-path /path/to/model \
    --gpu-memory-limit 5G \
    --context 2K \
    --max-concurrent 4

# Use 75% of total VRAM
crane-serve --model-path /path/to/model \
    --gpu-memory-limit 0.75
```

When the KV memory budget is exceeded, the engine evicts the longest-output
sequence, preserving its state, and tightens the concurrency cap. It resumes
the evicted sequence automatically once load subsides. This avoids OOM
without crashing the server.

**Recommended values by GPU size:**

| GPU VRAM | `--gpu-memory-limit` | `--context` |
|----------|---------------------|-------------|
| 8 GB     | `6G` or `0.7`       | `2K`        |
| 12 GB    | `8G` or `0.7`       | `4K`        |
| 24 GB    | `20G` or `0.8`      | `8K`        |
| 48 GB+   | *(omit)*            | *(omit)*    |

The startup log prints `kv_bytes` and `kv_budget`. Monitor these to
validate your `--gpu-memory-limit` headroom.

## GGUF quantized models on CUDA

GGUF quantization roughly halves VRAM usage compared to FP16:

```bash
crane-serve --model-path /path/to/Qwen3-8B-Q4_K_M.gguf \
    --format gguf \
    --gpu-memory-limit 8G
```

GGUF quantization is supported for Hunyuan Dense and Qwen 3. Qwen 2.5
requires the Safetensors format.

## Multi-GPU

crane-serve runs on a single CUDA device (device 0). Multi-GPU tensor
parallelism is not yet supported.

## Environment variables

These tune GPU-side sampling. They rarely need changing.

| Variable | Default | Description |
|----------|---------|-------------|
| `CRANE_FORCE_GPU_TOPK` | `0` | Force GPU top-k even for large vocabularies |
| `CRANE_TOPP_FALLBACK_TOPK` | `64` | k value for GPU top-k fallback |
| `CRANE_TOPK_SAMPLE_ON_CPU` | `0` | Sample on CPU after GPU top-k |
