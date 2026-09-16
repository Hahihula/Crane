//! `KugelAudio` (`kugelaudio/kugelaudio-0-open`): a post-trained fine-tune of
//! Microsoft's `VibeVoice` architecture — a `Qwen2` decoder that autoregressively
//! interleaves text tokens with continuous speech latents from a diffusion
//! head, encoded/decoded through causal-conv VAE "tokenizers".
//!
//! See `config.rs` for the architecture summary, `model.rs` for what's wired
//! up vs. deferred.

#![allow(clippy::doc_markdown)] // KugelAudio / VibeVoice / Qwen2 are external names

pub mod config;
pub mod connector;
pub mod conv_layers;
pub mod decoder;
pub mod diffusion_head;
pub mod dpm_solver;
pub mod model;
pub mod prompt;

pub use config::{KugelAudioConfig, load_config};
pub use model::{KugelAudioGenerationConfig, KugelAudioGenerationOutput, KugelAudioModel};
pub use prompt::{PromptResult, build_prompt};
