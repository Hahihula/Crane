// SPDX-License-Identifier: MIT

//! [`Tts`] trait implementation for Kokoro TTS.
//!
//! Unlike the other `Tts` impls, Kokoro's inherent `generate_speech` takes an
//! extra `&dyn Phonemizer` parameter (see
//! [`crane_core::models::kokoro_tts::Model::generate_speech`]), so this
//! wraps the model together with an already-built phonemizer rather than
//! implementing `Tts` directly on the model type.

use anyhow::Result;
use candle_core::Tensor;
use crane_core::generation::SpeechOptions;
use crane_core::models::g2p::Phonemizer;
use crane_core::models::kokoro_tts;

use super::pcm::AudioInfo;
use super::tts::{Tts, VoiceInfo};

/// Maps a Kokoro voice name's single-character language prefix to an ISO
/// 639-1 code, per Kokoro's documented voice naming convention (e.g.
/// `af_heart` -> American English, `bf_emma` -> British English).
///
/// `d` -> German is this codebase's own extension: upstream Kokoro/Misaki
/// never assigned that prefix to a language, so it's free to repurpose for
/// `df_kerstin`, the German voice routed through `GermanG2p`.
///
/// Falls back to `"en"` for an unrecognized or missing prefix.
fn voice_name_language(name: &str) -> &'static str {
    match name.as_bytes().first() {
        Some(b'a' | b'b') => "en",
        Some(b'd') => "de",
        Some(b'e') => "es",
        Some(b'f') => "fr",
        Some(b'h') => "hi",
        Some(b'i') => "it",
        Some(b'j') => "ja",
        Some(b'k') => "ko",
        Some(b'p') => "pt",
        Some(b'z') => "zh",
        _ => "en",
    }
}

/// Kokoro TTS, paired with the [`Phonemizer`] its `generate_speech` needs.
///
/// The phonemizer is supplied by the caller rather than located from
/// `--model-path`: `--model-path` points at a Kokoro-only directory with no
/// G2P assets, so locating a lexicon/OOV model is left to callers that know
/// where their G2P data lives (e.g. crane-wyoming's multi-directory model
/// layout).
pub struct KokoroTts {
    model: kokoro_tts::Model,
    phonemizer: Box<dyn Phonemizer + Send>,
}

impl KokoroTts {
    /// Wraps an already-constructed Kokoro `Model` and `Phonemizer`.
    #[must_use]
    pub fn new(model: kokoro_tts::Model, phonemizer: Box<dyn Phonemizer + Send>) -> Self {
        Self { model, phonemizer }
    }
}

impl Tts for KokoroTts {
    fn audio_info(&self) -> AudioInfo {
        AudioInfo {
            sample_rate: self.model.sample_rate(),
            channels: 1,
            bits_per_sample: 16,
        }
    }

    /// Returns a [`VoiceInfo`] for each voice discovered under the model's
    /// `voices/` directory, with the language derived from the voice name's
    /// single-character prefix (see [`voice_name_language`]).
    fn voices(&self) -> Vec<VoiceInfo> {
        self.model
            .available_voices()
            .iter()
            .map(|name| VoiceInfo {
                name: name.clone(),
                languages: vec![voice_name_language(name).to_string()],
            })
            .collect()
    }

    /// Delegates to the inherent [`kokoro_tts::Model::generate_speech`],
    /// supplying the stored phonemizer and discarding the sample rate
    /// (available via [`Tts::audio_info`] instead).
    fn generate_speech(
        &mut self,
        text: &str,
        language: &str,
        voice: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<Tensor> {
        let (tensor, _sample_rate) =
            self.model
                .generate_speech(text, language, voice, self.phonemizer.as_ref(), opts)?;
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::voice_name_language;

    // Verifies every documented Kokoro language-prefix character maps to
    // its correct ISO 639-1 code.
    #[test]
    fn test_voice_name_language_known_prefixes() {
        assert_eq!(voice_name_language("af_heart"), "en");
        assert_eq!(voice_name_language("bf_emma"), "en");
        assert_eq!(voice_name_language("df_kerstin"), "de");
        assert_eq!(voice_name_language("ef_dora"), "es");
        assert_eq!(voice_name_language("ff_siwis"), "fr");
        assert_eq!(voice_name_language("hf_alpha"), "hi");
        assert_eq!(voice_name_language("if_sara"), "it");
        assert_eq!(voice_name_language("jf_alpha"), "ja");
        assert_eq!(voice_name_language("kf_dahye"), "ko");
        assert_eq!(voice_name_language("pf_dora"), "pt");
        assert_eq!(voice_name_language("zf_xiaobei"), "zh");
    }

    // Verifies unrecognized or empty voice names fall back to English
    // rather than panicking on an out-of-range prefix lookup.
    #[test]
    fn test_voice_name_language_unknown_falls_back_to_english() {
        assert_eq!(voice_name_language("qf_unknown"), "en");
        assert_eq!(voice_name_language(""), "en");
    }
}
