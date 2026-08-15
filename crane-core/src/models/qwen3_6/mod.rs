//! Qwen 3.6 / Qwen 3.8 — no code of their own.
//!
//! Both are the Qwen 3.5 architecture scaled up, and say so themselves: the
//! 27B checkpoints declare `model_type: "qwen3_5"` /
//! `architectures: ["Qwen3_5ForConditionalGeneration"]`, and their GGUF
//! conversions carry `general.architecture = "qwen35"`. Qwen 3.6-27B and
//! Qwen 3.8-27B have byte-identical `text_config` blocks.
//!
//! Load them through [`crate::models::qwen3_5`]; every difference from a
//! Qwen 3.5 checkpoint is a config value (64 layers, `hidden_size` 5120,
//! 24 query / 4 KV heads, 48 GDN value heads so `v_per_group == 3`, untied
//! `lm_head`, ViT depth 27 / hidden 1152, and an explicit
//! `output_gate_type: "swish"` naming the GDN gate that module already
//! implements).
//!
//! Still open, and the only genuinely new capability these checkpoints ship:
//! the MTP draft head (`mtp.*` weights in safetensors, a separate
//! `mtp-*.gguf` alongside the main file) for speculative decoding. Those
//! tensors are simply not loaded today.
