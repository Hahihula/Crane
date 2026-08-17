//! MuScriptor: multi-instrument audio-to-MIDI transcription.
//!
//! Decoder-only transformer with a mel-spectrogram **prefix conditioner**
//! prepended to the token sequence. Auto-regressively predicts MT3-style
//! MIDI event tokens over a 1393-entry vocabulary. See the module README
//! (`README.md`) for architecture and license notes — the upstream
//! *weights* are CC BY-NC 4.0; the *code* here is Crane's standard MIT.
//!
//! Reading order (top-down = dependency order):
//!   `mt3`        → MT3Tokenizer (vocab + encode/decode + tie section)
//!   `midi`       → minimal Standard MIDI File writer
//!   `transformer` → streaming MHA + sinusoidal pos + transformer layers
//!   `conditioner` → mel + class prefix conditioners
//!   `config`     → variant (small/medium/large) + DSP params
//!   `model`      → LMModel assembly + generate loop + public `TranscriptionModel`

// Many of these items are scaffolding for follow-up PRs (CFG plumbing,
// tie-section forcing, polyphonic note reconstruction, etc.).
// `#[allow(dead_code)]` keeps the workspace warning-clean while the
// adjacent code paths stay invisible to the compiler.
#![allow(dead_code)]

mod conditioner;
mod config;
mod midi;
mod model;
mod mt3;
mod transformer;

pub use conditioner::{
    ClassConditioner, ConditioningAttributes, ConditioningProvider, MelSpectrogramConditioner,
    WavCondition,
};
pub use config::VariantConfig;
pub use midi::{MidiNote, MidiWriter};
pub use model::{LMModel, Model, NoteEvent, TranscribeConfig, TranscriptionModel};
pub use mt3::{
    DRUM_PROGRAM, MT3_FULL_PLUS_GROUP_NAMES, MT3Tokenizer, Token, TokenIter,
    instrument_group_from_names, resolve_instrument_names,
};
pub use transformer::{LayerState, TransformerState, create_sin_embedding};
