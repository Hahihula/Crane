//! Qwen 3.5 dense text-only model.
//!
//! Hybrid Mamba/Transformer: every 4th layer is full (softmax) attention,
//! the other 3 are linear attention via [`crate::gdn::GatedDeltaNet`].
//!
//! See [`config::TextConfig`] for the HF schema mapping and
//! [`model::Qwen3_5TextModel`] for the high-level entry point.
//!
//! Phase 1A scope: CPU inference, dense checkpoints only (no MoE), text-only
//! weights (the `Qwen3_5ForConditionalGeneration` multimodal class is
//! supported for weight loading, but vision weights are ignored).

mod config;
mod model;
mod modeling;

pub use config::{load_config, Config, LayerType, TextConfig};
pub use model::{Model, ModelFormat, Qwen3_5TextModel};
pub use modeling::{apply_mrope, DecoderLayer, FullAttention, Mlp, MRotaryEmbedding, Qwen35RmsNorm};