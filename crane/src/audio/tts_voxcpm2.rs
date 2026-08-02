//! [`Tts`] trait implementation for [`crane_core::models::voxcpm2::VoxCpm2Model`].
//!
//! Zero-shot only for this pass (see the crate-level scope note on
//! `crane_core::models::voxcpm2`) — `voices()` returns no presets and
//! `generate_voice_clone`/`generate_speech_stream` are left at their trait
//! defaults (`bail!` and single-chunk-wrap respectively), both of which
//! already match this model's current capability.

use anyhow::Result;
use candle_core::Tensor;
use crane_core::generation::SpeechOptions;
use crane_core::models::voxcpm2::{VoxCpm2GenerationConfig, VoxCpm2Model};

use super::pcm::AudioInfo;
use super::tts::{Tts, VoiceInfo};

impl Tts for VoxCpm2Model {
    fn audio_info(&self) -> AudioInfo {
        AudioInfo { sample_rate: self.sample_rate, channels: 1, bits_per_sample: 16 }
    }

    /// No discrete presets — VoxCPM2 is zero-shot per-utterance (or, in
    /// modes not implemented by this pass, cloned from reference audio).
    fn voices(&self) -> Vec<VoiceInfo> {
        vec![]
    }

    /// `language`/`voice` are unused: VoxCPM2's zero-shot path infers
    /// prosody/language from the text itself and has no voice selection.
    fn generate_speech(
        &mut self,
        text: &str,
        _language: &str,
        _voice: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<Tensor> {
        // `max_new_tokens` doc says "codec frames"; VoxCPM2's closest analog
        // is its own generation-step count (each step yields one 4-frame
        // latent patch) — pass through directly as an upper bound rather
        // than inventing an unjustified conversion factor. The model's own
        // stop head almost always ends generation well before this cap.
        let cfg = VoxCpm2GenerationConfig { max_len: opts.max_new_tokens.max(1), ..Default::default() };
        VoxCpm2Model::generate_speech(self, text, &cfg)
    }
}
