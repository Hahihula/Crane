// SPDX-License-Identifier: MIT

//! Kokoro TTS model support: IPA postprocessing for now, ONNX inference and
//! voice loading in a later step.

mod ipa;

pub use ipa::build_kokoro_normalizer;
