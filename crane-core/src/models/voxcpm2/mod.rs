//! VoxCPM2 (OpenBMB): tokenizer-free diffusion-autoregressive TTS.
//!
//! Five sub-networks — `base_lm`/`residual_lm` (causal, share
//! [`minicpm4::MiniCpm4Model`]), `feat_encoder`/`feat_decoder` (non-causal,
//! same shared stack), and a separate DAC-style `audio_vae` codec — see
//! `config.rs` for the architecture summary and `model.rs` for the
//! generation loop.
//!
//! Covers **zero-shot generation and reference-audio conditioning (voice
//! cloning) with prompt-cache support** — see [`model::VoxCpm2Conditioning`]
//! for the four supported modes and [`model::VoxCpm2PromptCache`] for
//! multi-turn reuse — plus **incremental streaming generation** via
//! [`model::VoxCpm2Model::generate_speech_streaming`] /
//! [`model::VoxCpm2SpeechStream`].

pub mod audio_vae;
pub mod cfm;
pub mod config;
pub mod fsq;
pub mod local_dit;
pub mod local_encoder;
pub mod minicpm4;
pub mod model;
pub mod tokenizer;

pub use config::{VoxCpm2Config, load_config};
pub use model::{
    VoxCpm2Conditioning, VoxCpm2GenerationConfig, VoxCpm2Model, VoxCpm2PromptCache,
    VoxCpm2SpeechStream, VoxCpm2StreamConfig,
};
