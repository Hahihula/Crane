# Candle Transformers removal inventory

`crane-core` still depends on `candle-transformers`. Every source module that
uses it is marked with `TODO(candle-transformers-removal)` so the remaining
surface can be found with:

```bash
grep -RIn "TODO(candle-transformers-removal)" crane-core
```

## Model and component dependencies

These usages require a Crane-native model or component before the dependency
can be removed.

| Crane model | Source | Candle Transformers API | Replacement direction |
|---|---|---|---|
| Qwen 2.5 | `src/models/qwen25/model.rs` | `models::qwen2` and `models::qwen2_moe` | Use/finish the existing local `qwen25/qwen2.rs` implementation; add a native MoE implementation. |
| PaddleOCR-VL | `src/models/paddleocr_vl/model.rs` | `models::paddleocr_vl::{Config, PaddleOCRVLModel}` | Port the complete model and config into Crane. |
| Qwen3-TTS speech tokenizer | `src/models/qwen3_tts/speech_tokenizer_v2.rs` | Mimi SeaNet encoder, projected transformer, downsampler, and quantizer | Port the required Mimi encoder and quantization components. |

## Generation-helper-only dependencies

These models are implemented in Crane and only use Candle Transformers for
sampling and/or repetition penalty helpers:

| Crane model | Source | Helpers used |
|---|---|---|
| Gemma 4 | `src/models/gemma4/model.rs` | `LogitsProcessor`, `apply_repeat_penalty` |
| Hunyuan Dense | `src/models/hunyuan_dense/model.rs` | `LogitsProcessor`, `apply_repeat_penalty` |
| Qwen 3 | `src/models/qwen3/model.rs` | `LogitsProcessor`, `apply_repeat_penalty` |
| Qwen 3.5 | `src/models/qwen3_5/model.rs` | `LogitsProcessor`, `apply_repeat_penalty` |
| Qwen3-ASR | `src/models/qwen3_asr/model.rs` | `LogitsProcessor`, `apply_repeat_penalty` |
| Qwen3-TTS modeling | `src/models/qwen3_tts/modeling.rs` | `LogitsProcessor`, `Sampling::TopKThenTopP`, `apply_repeat_penalty` |
| Qwen 2.5 generation wrapper | `src/models/qwen25/model.rs` | `LogitsProcessor`, `apply_repeat_penalty` |

Crane already has a native repetition-penalty function in
`src/models/utils.rs`. The remaining prerequisite for this group is a native
logits processor covering greedy, temperature, top-k, and top-p sampling.

## Feature forwarding

After all source imports are removed, delete `candle-transformers` from
`crane-core/Cargo.toml` and remove its `metal`, `cuda`, `mkl`, and `accelerate`
feature forwarding entries. Then remove the corresponding direct dependency
and forwarding entries from `crane-serve/Cargo.toml`.
