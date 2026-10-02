//! Qwen4-Exp (`qwen4_exp` / GGUF `qwen4exp`), e.g. Qwen3.8-Flash-Next.
//!
//! A Qwen 3.5-style hybrid (three gated-delta-net layers per softmax layer,
//! routed `MoE` with a shared expert everywhere) with three additions, each
//! its own module here:
//!
//! - [`hyper_connection`]: the residual is `hc_count` parallel streams, mixed
//!   into each block's input and scattered back after it.
//! - [`ple`]: Per-Layer Embedding — hashed token n-grams from a huge table
//!   that stays on the host, injected into the streams of one layer.
//! - [`indexer`]: the QSA indexer restricting each softmax layer to a budget
//!   of pooled key blocks.
//!
//! The gated delta net is [`crate::ops::gdn::GatedDeltaNet`] with a sigmoid
//! output gate, and the `MoE` is [`crate::models::modules::moe::SparseMoeBlock`].
//! Reference: transformers `models/qwen4_exp`.

pub mod config;
pub mod hyper_connection;
pub mod indexer;
pub mod model;
pub mod ple;

pub use config::{IndexerConfig, LayerType, TextConfig};
pub use model::{Model, Qwen4ExpTextModel};
