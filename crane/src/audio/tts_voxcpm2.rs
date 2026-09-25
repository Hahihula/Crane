//! VoxCPM2 TTS adapter with persistent built-in voice embeddings.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::SystemTime;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use crane_core::candle_core;
use crane_core::generation::SpeechOptions;
use crane_core::models::voxcpm2::{
    VoxCpm2Conditioning, VoxCpm2GenerationConfig, VoxCpm2Model, VoxCpm2StreamConfig,
};

use super::pcm::{AudioInfo, load_audio_f32, load_wav_f32};
use super::tts::{Tts, TtsStream, VoiceInfo};

const CACHE_DIR: &str = ".voxcpm2-cache";
const CACHE_TENSOR: &str = "embedding";

/// VoxCPM2 plus reference-audio embeddings loaded once at startup.
pub struct VoxCpm2Tts {
    model: VoxCpm2Model,
    voices: BTreeMap<String, Tensor>,
    voice_names: Vec<String>,
}

impl VoxCpm2Tts {
    pub fn new(
        model: VoxCpm2Model,
        model_path: &Path,
        voice_dir: Option<&Path>,
        device: &Device,
    ) -> Result<Self> {
        let mut this = Self {
            model,
            voices: BTreeMap::new(),
            voice_names: Vec::new(),
        };
        if let Some(voice_dir) = voice_dir {
            this.load_builtin_voices(model_path, voice_dir, device)?;
        }
        Ok(this)
    }

    fn load_builtin_voices(
        &mut self,
        model_path: &Path,
        voice_dir: &Path,
        device: &Device,
    ) -> Result<()> {
        if !voice_dir.exists() {
            eprintln!(
                "[voxcpm2] built-in voice directory does not exist: {}",
                voice_dir.display()
            );
            return Ok(());
        }

        let cache_dir = voice_dir.join(CACHE_DIR);
        fs::create_dir_all(&cache_dir).with_context(|| {
            format!(
                "create VoxCPM2 voice cache directory {}",
                cache_dir.display()
            )
        })?;

        let mut sources = fs::read_dir(voice_dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| is_supported_audio(path))
            .collect::<Vec<_>>();
        sources.sort();

        for source in sources {
            let file_name = source
                .file_name()
                .and_then(|value| value.to_str())
                .context("voice filename is not valid UTF-8")?
                .to_string();
            let stem = source
                .file_stem()
                .and_then(|value| value.to_str())
                .context("voice filename has no stem")?
                .to_string();
            let cache_path = cache_dir.join(format!("{file_name}.safetensors"));

            let embedding = if cache_is_fresh(&cache_path, &source, model_path) {
                match candle_core::safetensors::load(&cache_path, device).and_then(|mut tensors| {
                    tensors.remove(CACHE_TENSOR).ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "missing tensor {CACHE_TENSOR:?} in {}",
                            cache_path.display()
                        ))
                        .bt()
                    })
                }) {
                    Ok(tensor) => {
                        eprintln!("[voxcpm2] loaded built-in voice cache: {file_name}");
                        tensor
                    },
                    Err(err) => {
                        eprintln!(
                            "[voxcpm2] rebuilding invalid voice cache {}: {err}",
                            cache_path.display()
                        );
                        self.encode_and_cache_voice(&source, &cache_path)?
                    },
                }
            } else {
                self.encode_and_cache_voice(&source, &cache_path)?
            };

            // The extension-free stem is the public voice name. Keep the
            // exact filename as a backwards-compatible lookup alias only.
            self.voices.insert(stem.clone(), embedding.clone());
            self.voices.insert(file_name, embedding);
            self.voice_names.push(stem);
        }

        eprintln!(
            "[voxcpm2] {} built-in voice(s) ready from {}",
            self.voice_names.len(),
            voice_dir.display()
        );
        Ok(())
    }

    fn encode_and_cache_voice(&self, source: &Path, cache_path: &Path) -> Result<Tensor> {
        let source_str = source.to_string_lossy();
        eprintln!("[voxcpm2] encoding built-in voice: {}", source.display());
        let samples = load_audio_f32(&source_str, self.model.encoder_sample_rate())?;
        let embedding = self.model.encode_reference_audio(&samples, false)?;
        embedding
            .to_device(&Device::Cpu)?
            .save_safetensors(CACHE_TENSOR, cache_path)?;
        eprintln!(
            "[voxcpm2] saved built-in voice cache: {}",
            cache_path.display()
        );
        Ok(embedding)
    }

    fn conditioning(&self, voice: Option<&str>) -> Result<VoxCpm2Conditioning> {
        match voice {
            None => Ok(VoxCpm2Conditioning::ZeroShot),
            Some(name) => self
                .voices
                .get(name)
                .cloned()
                .map(VoxCpm2Conditioning::Reference)
                .with_context(|| {
                    format!(
                        "unknown VoxCPM2 voice {name:?}; available voices: {}",
                        self.voice_names.join(", ")
                    )
                }),
        }
    }
}

fn is_supported_audio(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|ext| {
                matches!(
                    ext.to_ascii_lowercase().as_str(),
                    "wav" | "mp3" | "flac" | "ogg" | "m4a" | "aac"
                )
            })
}

fn modified(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).ok()?.modified().ok()
}

fn cache_is_fresh(cache: &Path, source: &Path, model_path: &Path) -> bool {
    let Some(cache_time) = modified(cache) else {
        return false;
    };
    [
        source.to_path_buf(),
        model_path.join("config.json"),
        model_path.join("model.safetensors"),
        model_path.join("audiovae.safetensors"),
    ]
    .iter()
    .filter_map(|path| modified(path))
    .all(|time| time <= cache_time)
}

fn gen_config(opts: &SpeechOptions) -> VoxCpm2GenerationConfig {
    let defaults = VoxCpm2GenerationConfig::default();
    VoxCpm2GenerationConfig {
        max_len: opts.max_new_tokens.max(1),
        inference_timesteps: opts
            .cfm_steps
            .map_or(defaults.inference_timesteps, |steps| steps.max(1)),
        cfg_value: opts.cfg_scale.unwrap_or(defaults.cfg_value),
        ..defaults
    }
}

impl Tts for VoxCpm2Tts {
    fn audio_info(&self) -> AudioInfo {
        AudioInfo {
            sample_rate: self.model.sample_rate,
            channels: 1,
            bits_per_sample: 16,
        }
    }

    fn voices(&self) -> Vec<VoiceInfo> {
        self.voice_names
            .iter()
            .map(|name| VoiceInfo {
                name: name.clone(),
                languages: vec![],
            })
            .collect()
    }

    fn supports_voice_cloning(&self) -> bool {
        true
    }

    fn generate_speech(
        &mut self,
        text: &str,
        _language: &str,
        voice: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<Tensor> {
        let conditioning = self.conditioning(voice)?;
        self.model
            .generate_speech_conditioned(text, &conditioning, &gen_config(opts))
    }

    fn generate_voice_clone(
        &mut self,
        text: &str,
        _language: &str,
        ref_audio: &str,
        ref_text: &str,
        opts: &SpeechOptions,
    ) -> Result<Tensor> {
        let samples = load_wav_f32(ref_audio, self.model.encoder_sample_rate())?;
        let prompt_feat = self.model.encode_reference_audio(&samples, true)?;
        let conditioning = VoxCpm2Conditioning::Continuation {
            prompt_text: ref_text.to_string(),
            prompt_feat,
        };
        self.model
            .generate_speech_conditioned(text, &conditioning, &gen_config(opts))
    }

    fn generate_speech_stream(
        &mut self,
        text: &str,
        _language: &str,
        voice: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<TtsStream<'_>> {
        let audio_info = self.audio_info();
        let conditioning = self.conditioning(voice)?;
        let cfg = gen_config(opts);
        let stream = self.model.generate_speech_streaming(
            text,
            &conditioning,
            &cfg,
            VoxCpm2StreamConfig::default(),
        )?;
        Ok(TtsStream::new(audio_info, stream))
    }
}
