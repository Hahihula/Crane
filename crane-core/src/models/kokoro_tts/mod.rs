// SPDX-License-Identifier: MIT

//! Kokoro TTS model support: IPA postprocessing and config/vocab parsing for
//! now, voice loading and ONNX inference in later steps.

mod ipa;
mod model;

pub use ipa::build_kokoro_normalizer;
pub use model::Model;
