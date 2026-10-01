// SPDX-License-Identifier: MIT

//! [`Tts`] trait implementation for [`crane_core::models::kugelaudio::KugelAudioModel`].

use std::collections::HashMap;

use anyhow::Result;
use candle_core::{Device, Tensor};
use crane_core::candle_core;
use crane_core::generation::SpeechOptions;
use crane_core::models::kugelaudio::{
    KugelAudioGenerationConfig, KugelAudioModel, KugelAudioVoice,
};

use super::pcm::{AudioInfo, load_wav_f32};
use super::tts::{Tts, VoiceInfo};

/// Normalizes a language code to its base ISO 639-1 subtag.
///
/// Lowercases the input, splits on `-`/`_`, and keeps the first component:
/// `"de-DE"` and `"de_DE"` both become `"de"`. Each `Tts` backend owns its
/// own language-format translation rather than sharing one, since different
/// models expect different internal formats (see `tts_qwen3.rs`'s
/// `language_code_to_name` for the same pattern applied differently).
fn normalize_language_code(code: &str) -> String {
    let lower = code.to_lowercase();
    match lower.split_once(['-', '_']) {
        Some((head, _)) => head.to_string(),
        None => lower,
    }
}

/// Resolves which preset voice (if any) `generate_speech` should condition on,
/// returning its name and Arc-cloned (cheap, no data copy) latent tensors.
///
/// Resolution order: an explicit `voice` name wins outright (error if it is
/// not a loaded preset); else the first preset whose declared language
/// matches `language`; else the preset literally named `"clear"`; else
/// `None`, meaning the caller should fall back to zero-shot generation
/// (either no presets were loaded, or none matched and there is no
/// `"clear"` preset).
fn resolve_voice(
    voices: &HashMap<String, KugelAudioVoice>,
    language: &str,
    voice: Option<&str>,
) -> Result<Option<(String, Tensor, Tensor)>> {
    if voices.is_empty() {
        return Ok(None);
    }

    if let Some(name) = voice {
        let Some((key, preset)) = voices.get_key_value(name) else {
            let available: Vec<&str> = voices.keys().map(String::as_str).collect();
            anyhow::bail!(
                "unknown KugelAudio voice '{name}' (available: {})",
                available.join(", ")
            );
        };
        return Ok(Some((
            key.clone(),
            preset.acoustic_mean.clone(),
            preset.semantic_mean.clone(),
        )));
    }

    let lang = normalize_language_code(language);
    if let Some((name, preset)) = voices
        .iter()
        .find(|(_, v)| normalize_language_code(&v.language) == lang)
    {
        return Ok(Some((
            name.clone(),
            preset.acoustic_mean.clone(),
            preset.semantic_mean.clone(),
        )));
    }

    Ok(voices.get_key_value("clear").map(|(key, preset)| {
        (
            key.clone(),
            preset.acoustic_mean.clone(),
            preset.semantic_mean.clone(),
        )
    }))
}

/// Builds a [`KugelAudioGenerationConfig`] from the shared [`SpeechOptions`].
///
/// `opts.top_p`, `opts.repetition_penalty`, and `opts.cfm_steps` have no
/// `KugelAudio` equivalent and are ignored, matching how other backends
/// ignore options that don't apply to them (e.g. `VoxCPM2` ignores `temperature`).
/// Temperature `0.0` means greedy decoding, matching `SpeechOptions`'
/// convention elsewhere.
fn gen_config(opts: &SpeechOptions) -> KugelAudioGenerationConfig {
    KugelAudioGenerationConfig {
        cfg_scale: opts
            .cfg_scale
            .unwrap_or(KugelAudioGenerationConfig::default().cfg_scale),
        max_new_tokens: opts.max_new_tokens,
        do_sample: opts.temperature > 0.0,
        temperature: opts.temperature,
    }
}

/// Wraps a raw mono f32 PCM buffer as a `[1, n]` tensor for the [`Tts`] trait's return type.
fn wrap_audio(audio: Vec<f32>, device: &Device) -> Result<Tensor> {
    let n = audio.len();
    Ok(Tensor::from_vec(audio, (1, n), device)?)
}

impl Tts for KugelAudioModel {
    fn audio_info(&self) -> AudioInfo {
        AudioInfo {
            sample_rate: self.sample_rate(),
            channels: 1,
            bits_per_sample: 16,
        }
    }

    /// Returns a [`VoiceInfo`] for each preset voice loaded from
    /// `voices/voices.json`. Empty if the checkpoint ships no presets.
    fn voices(&self) -> Vec<VoiceInfo> {
        self.available_voices()
            .iter()
            .map(|(name, voice)| VoiceInfo {
                name: name.clone(),
                languages: vec![voice.language.clone()],
            })
            .collect()
    }

    fn supports_voice_cloning(&self) -> bool {
        true
    }

    /// Resolves a preset voice (see [`resolve_voice`]) and generates with its
    /// pre-encoded latents, or falls back to zero-shot generation when no
    /// preset applies.
    fn generate_speech(
        &mut self,
        text: &str,
        language: &str,
        voice: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<Tensor> {
        let cfg = gen_config(opts);
        let resolved = resolve_voice(self.available_voices(), language, voice)?;

        let Some((voice_name, acoustic, semantic)) = resolved else {
            let prompt = self.build_prompt(text, None)?;
            let output = self.generate(&prompt, None, None, &cfg)?;
            return wrap_audio(output.audio, self.device());
        };

        let prompt = self.build_prompt_for_voice(text, &voice_name)?;
        let output = self.generate(&prompt, Some((&acoustic, &semantic)), None, &cfg)?;
        wrap_audio(output.audio, self.device())
    }

    /// Loads `ref_audio` at the model's own sample rate and conditions
    /// generation on the raw waveform. `language` and `ref_text` are unused:
    /// `KugelAudio` has no internal language-conditioning signal (language
    /// follows the input text), and its voice cloning path conditions on
    /// audio directly rather than a reference transcript.
    fn generate_voice_clone(
        &mut self,
        text: &str,
        _language: &str,
        ref_audio: &str,
        _ref_text: &str,
        opts: &SpeechOptions,
    ) -> Result<Tensor> {
        let samples = load_wav_f32(ref_audio, self.sample_rate())?;
        let n = samples.len();
        let waveform = Tensor::from_vec(samples, (1, 1, n), self.device())?;
        let prompt = self.build_prompt(text, Some(n))?;
        let cfg = gen_config(opts);
        let output = self.generate(&prompt, None, Some(&waveform), &cfg)?;
        wrap_audio(output.audio, self.device())
    }

    // generate_speech_stream: uses the trait default (wraps generate_speech
    // in a single-chunk stream). KugelAudio's diffusion-based pipeline
    // produces the full waveform before returning, so there is no
    // incremental output to stream.
}

#[cfg(test)]
mod tests {
    use super::candle_core::DType;
    use super::{Device, HashMap, KugelAudioVoice, Tensor, normalize_language_code, resolve_voice};

    fn dummy_voice(language: &str) -> KugelAudioVoice {
        let t = Tensor::zeros((1, 1, 0), DType::BF16, &Device::Cpu).unwrap();
        KugelAudioVoice {
            acoustic_mean: t.clone(),
            semantic_mean: t,
            language: language.to_string(),
            description: String::new(),
        }
    }

    #[test]
    fn resolve_voice_empty_map_returns_none() {
        let voices = HashMap::new();
        assert!(resolve_voice(&voices, "en", None).unwrap().is_none());
        assert!(resolve_voice(&voices, "en", Some("alice")).is_ok());
    }

    #[test]
    fn resolve_voice_explicit_name_hit() {
        let mut voices = HashMap::new();
        voices.insert("alice".to_string(), dummy_voice("en"));
        let (name, _, _) = resolve_voice(&voices, "fr", Some("alice"))
            .unwrap()
            .unwrap();
        assert_eq!(name, "alice");
    }

    #[test]
    fn resolve_voice_explicit_name_miss_errors() {
        let mut voices = HashMap::new();
        voices.insert("alice".to_string(), dummy_voice("en"));
        let err = resolve_voice(&voices, "en", Some("bob")).unwrap_err();
        assert!(err.to_string().contains("unknown KugelAudio voice 'bob'"));
        assert!(err.to_string().contains("alice"));
    }

    #[test]
    fn resolve_voice_language_match_falls_back() {
        let mut voices = HashMap::new();
        voices.insert("bruno".to_string(), dummy_voice("de-DE"));
        let (name, _, _) = resolve_voice(&voices, "de", None).unwrap().unwrap();
        assert_eq!(name, "bruno");
    }

    #[test]
    fn resolve_voice_clear_fallback_when_no_language_match() {
        let mut voices = HashMap::new();
        voices.insert("clear".to_string(), dummy_voice("en"));
        voices.insert("bruno".to_string(), dummy_voice("de"));
        let (name, _, _) = resolve_voice(&voices, "fr", None).unwrap().unwrap();
        assert_eq!(name, "clear");
    }

    #[test]
    fn resolve_voice_no_match_and_no_clear_returns_none() {
        let mut voices = HashMap::new();
        voices.insert("bruno".to_string(), dummy_voice("de"));
        assert!(resolve_voice(&voices, "fr", None).unwrap().is_none());
    }

    #[test]
    fn normalize_language_code_strips_region() {
        assert_eq!(normalize_language_code("de-DE"), "de");
        assert_eq!(normalize_language_code("en_US"), "en");
        assert_eq!(normalize_language_code("zh-Hans-CN"), "zh");
    }

    #[test]
    fn normalize_language_code_lowercases() {
        assert_eq!(normalize_language_code("DE"), "de");
        assert_eq!(normalize_language_code("En-US"), "en");
    }

    #[test]
    fn normalize_language_code_passthrough() {
        assert_eq!(normalize_language_code("de"), "de");
        assert_eq!(normalize_language_code("en"), "en");
        assert_eq!(normalize_language_code("auto"), "auto");
    }

    #[test]
    fn normalize_language_code_empty() {
        assert_eq!(normalize_language_code(""), "");
    }
}
