//! `KugelAudioModel`: loads and wires every sub-network from a checkpoint
//! directory, plus the mechanical building blocks a generation loop needs
//! (prefill/decode forward, one diffusion-denoising step, tokenizer
//! encode/decode). All weight-name prefixes verified against the real
//! `kugelaudio/kugelaudio-0-open` `model.safetensors.index.json` (1205
//! tensors), not just the Python source:
//!
//! ```text
//! lm_head.weight
//! model.language_model.{embed_tokens,layers,norm}.*
//! model.acoustic_tokenizer.{encoder,decoder}.*
//! model.semantic_tokenizer.encoder.*
//! model.{acoustic,semantic}_connector.{fc1,fc2,norm}.*
//! model.prediction_head.{cond_proj,final_layer,layers,noisy_images_proj,t_embedder}.*
//! model.speech_scaling_factor, model.speech_bias_factor   (scalar buffers)
//! ```
//!
//! The `generate()` loop is a batch-size-1 simplification of
//! `kugelaudio_inference.py`'s version (the batched version's per-step
//! cache splice has no equivalent in the shared
//! [`crate::models::modules::attention::GqaAttention`], append-only). The
//! simplification is exact for the single-utterance case.

#![allow(clippy::needless_pass_by_value)] // VarBuilder by-value is the candle idiom
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // i64 timestep -> f32: bounded 0-999
#![allow(clippy::format_in_format_args)] // "kugelaudio: ... {e}" inline format
#![allow(clippy::too_many_lines)] // generate() is necessarily long
#![allow(clippy::cast_precision_loss)] // tensor dim / ratio math
#![allow(clippy::module_name_repetitions)] // KugelAudioModel etc.
#![allow(clippy::missing_errors_doc)] // Result-returning helpers: errors are candle tensor errors
#![allow(clippy::missing_panics_doc)] // expect() in tests / debug_asserts only
#![allow(clippy::must_use_candidate)] // getters are conventionally used at call sites
#![allow(clippy::doc_markdown)] // KugelAudio is the model name, not generic Markdown text
#![allow(
    clippy::explicit_iter_loop,
    clippy::explicit_counter_loop,
    clippy::float_cmp
)]
// //: stylistic; matches the codebase's prevailing style elsewhere

use anyhow::{Context, Result};
use candle_core::quantized::GgmlDType;
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;

use crate::utils::utils::get_safetensors_files;

use super::config::{KugelAudioConfig, load_config};
use super::connector::SpeechConnector;
use super::conv_layers::{TokenizerDecoder, TokenizerEncoder, load_decoder, load_encoder};
use super::decoder::KugelAudioDecoder;
use super::diffusion_head::DiffusionHead;
use super::dpm_solver::DpmSolverScheduler;
use super::prompt::PromptResult;

use crate::models::with_tracing::{Linear, linear_no_bias};

/// Speech-related special token ids. Hardcoded in the Python (not in
/// `config.json`) — reuse Qwen2's vision special tokens since KugelAudio
/// ships no tokenizer of its own.
pub mod special_tokens {
    pub const SPEECH_START_ID: u32 = 151_652;
    pub const SPEECH_END_ID: u32 = 151_653;
    pub const SPEECH_DIFFUSION_ID: u32 = 151_654;
    pub const EOS_TOKEN_ID: u32 = 151_643;
}

/// Every sub-network wired to one checkpoint's weights, plus the config
/// values callers need.
pub struct KugelAudioModel {
    pub config: KugelAudioConfig,
    decoder: KugelAudioDecoder,
    lm_head: Linear,
    acoustic_encoder: TokenizerEncoder,
    acoustic_decoder: TokenizerDecoder,
    semantic_encoder: TokenizerEncoder,
    acoustic_connector: SpeechConnector,
    semantic_connector: SpeechConnector,
    diffusion_head: DiffusionHead,
    /// `model.speech_scaling_factor` / `model.speech_bias_factor`: training-
    /// time scalar buffers (`1/std`, `-mean` of the training acoustic-latent
    /// distribution). Applied as `(latent + bias) * scale` before the
    /// acoustic connector, inverted after diffusion sampling. Always
    /// present (non-NaN) in a trained checkpoint — the Python's NaN-guarded
    /// "skip scaling" branch only matters mid-training.
    speech_scaling_factor: f64,
    speech_bias_factor: f64,
    device: Device,
    dtype: DType,
}

/// Read the in-situ quantization level from `CRANE_ISQ` (e.g. `q4_0`,
/// `q8_0`). Invalid values abort with a clear message rather than silently
/// loading fp.
fn isq_from_env() -> Option<GgmlDType> {
    let name = std::env::var("CRANE_ISQ").ok()?;
    if name.trim().is_empty() {
        return None;
    }
    match crate::ops::linear::parse_ggml_dtype(&name) {
        Ok(dt) => Some(dt),
        Err(e) => panic!("invalid CRANE_ISQ: {e}"),
    }
}

impl KugelAudioModel {
    /// Load every sub-network from `model_dir` (a directory containing
    /// `config.json` and `model.safetensors.index.json` + shards). Verified
    /// against the real checkpoint in `crane-core/tests/kugelaudio_load.rs`.
    ///
    /// In-situ quantization is picked up from `CRANE_ISQ`; use
    /// [`Self::from_pretrained_with_quant`] to set it explicitly.
    ///
    /// # Errors
    ///
    /// Returns an error if the config can't be read, weights can't be
    /// mmaped, or any sub-network fails to construct.
    pub fn from_pretrained(model_dir: &str, device: &Device, dtype: DType) -> Result<Self> {
        Self::from_pretrained_with_quant(model_dir, device, dtype, isq_from_env())
    }

    /// Like [`Self::from_pretrained`], but with `quant: Some(dtype)` quantizes
    /// the decoder backbone's Q/K/V/O and MLP projections in-situ. The
    /// ~7B-parameter Qwen2 decoder is the bulk of the ~18.7GB checkpoint, so
    /// this is the lever for low-VRAM/low-RAM machines (works identically on
    /// CUDA and Metal — see `decoder.rs`'s module doc comment). The VAE
    /// tokenizers and diffusion head stay at full precision.
    ///
    /// # Errors
    ///
    /// Returns an error if the config can't be read, weights can't be
    /// mmaped, or any sub-network fails to construct.
    pub fn from_pretrained_with_quant(
        model_dir: &str,
        device: &Device,
        dtype: DType,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let config_path = std::path::Path::new(model_dir).join("config.json");
        let config = load_config(config_path.to_str().context("non-UTF8 model path")?)
            .context("kugelaudio: load config.json")?;

        let filenames = get_safetensors_files(model_dir)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device)? };
        let vb_model = vb.pp("model");

        // A second, CPU-scoped handle onto the same checkpoint (cheap: same
        // mmap) so the decoder's quantized path can read each big matmul
        // weight without ever materializing it on `device` first. See
        // `KugelAudioDecoder::new_with_quant`.
        let decoder_quant = match quant {
            None => None,
            Some(dt) => {
                eprintln!("[kugelaudio] in-situ quantization enabled: {dt:?}");
                let vb_cpu = unsafe {
                    VarBuilder::from_mmaped_safetensors(&filenames, dtype, &Device::Cpu)?
                };
                Some((dt, vb_cpu.pp("model").pp("language_model")))
            },
        };
        let decoder = KugelAudioDecoder::new_with_quant(
            &config.decoder_config,
            vb_model.pp("language_model"),
            decoder_quant,
        )
        .context("kugelaudio: load language_model")?;
        // Dense even under ISQ — see `KugelAudioDecoder::new_with_quant`.
        let lm_head = linear_no_bias(
            config.decoder_config.hidden_size,
            config.decoder_config.vocab_size,
            vb.pp("lm_head"),
        )
        .context("kugelaudio: load lm_head")?;

        let vb_acoustic = vb_model.pp("acoustic_tokenizer");
        let acoustic_encoder =
            load_encoder(&config.acoustic_tokenizer_config, vb_acoustic.pp("encoder"))
                .context("kugelaudio: load acoustic_tokenizer.encoder")?;
        let acoustic_decoder =
            load_decoder(&config.acoustic_tokenizer_config, vb_acoustic.pp("decoder"))
                .context("kugelaudio: load acoustic_tokenizer.decoder")?;

        let semantic_encoder = load_encoder(
            &config.semantic_tokenizer_config,
            vb_model.pp("semantic_tokenizer").pp("encoder"),
        )
        .context("kugelaudio: load semantic_tokenizer.encoder")?;

        let acoustic_connector = SpeechConnector::load(
            config.acoustic_vae_dim,
            config.decoder_config.hidden_size,
            vb_model.pp("acoustic_connector"),
        )
        .context("kugelaudio: load acoustic_connector")?;
        let semantic_connector = SpeechConnector::load(
            config.semantic_vae_dim,
            config.decoder_config.hidden_size,
            vb_model.pp("semantic_connector"),
        )
        .context("kugelaudio: load semantic_connector")?;

        let diffusion_head = DiffusionHead::load(
            &config.diffusion_head_config,
            vb_model.pp("prediction_head"),
        )
        .context("kugelaudio: load prediction_head")?;

        let speech_scaling_factor: f64 = vb_model
            .get((), "speech_scaling_factor")
            .context("kugelaudio: load speech_scaling_factor")?
            .to_dtype(DType::F64)?
            .to_scalar()?;
        let speech_bias_factor: f64 = vb_model
            .get((), "speech_bias_factor")
            .context("kugelaudio: load speech_bias_factor")?
            .to_dtype(DType::F64)?
            .to_scalar()?;

        Ok(Self {
            config,
            decoder,
            lm_head,
            acoustic_encoder,
            acoustic_decoder,
            semantic_encoder,
            acoustic_connector,
            semantic_connector,
            diffusion_head,
            speech_scaling_factor,
            speech_bias_factor,
            device: device.clone(),
            dtype,
        })
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Text-token embedding lookup. `input_ids`: `[batch, seq_len]` →
    /// `[batch, seq_len, hidden_size]`.
    pub fn embed_text_tokens(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        self.decoder.embed_tokens(input_ids)
    }

    /// Run the decoder backbone on pre-built input embeddings and project
    /// to vocabulary logits at every position. `seqlen_offset` is the
    /// number of already-cached positions (0 for a fresh prefill).
    /// Returns `(hidden_states, logits)`, both `[batch, seq_len, *]`.
    pub fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let hidden = self.decoder.forward_embeds(inputs_embeds, seqlen_offset)?;
        let logits = self.lm_head.forward(&hidden)?;
        Ok((hidden, logits))
    }

    pub fn clear_kv_cache(&mut self) {
        self.decoder.clear_kv_cache();
    }

    /// Encode a raw waveform through the acoustic tokenizer, returning the
    /// **unscaled** latent mean (this port always takes the distribution's
    /// mean rather than the Gaussian-sampled path — see
    /// `config.rs`'s `TokenizerConfig::std_dist_type`). `[batch, vae_dim, T]`.
    pub fn encode_acoustic(&self, waveform: &Tensor) -> candle_core::Result<Tensor> {
        self.acoustic_encoder
            .encode(&waveform.to_dtype(self.dtype)?)
    }

    /// Encode a raw waveform through the semantic tokenizer.
    /// `[batch, semantic_vae_dim, T]`.
    pub fn encode_semantic(&self, waveform: &Tensor) -> candle_core::Result<Tensor> {
        self.semantic_encoder
            .encode(&waveform.to_dtype(self.dtype)?)
    }

    /// Decode acoustic latents (`[batch, vae_dim, T]`, **already** run
    /// through [`Self::unscale_acoustic_latent`] if they came from
    /// diffusion sampling) back to a waveform (`[batch, 1, num_samples]`).
    pub fn decode_acoustic(&self, latents: &Tensor) -> candle_core::Result<Tensor> {
        self.acoustic_decoder.decode(&latents.to_dtype(self.dtype)?)
    }

    /// `(latent + speech_bias_factor) * speech_scaling_factor` — applied
    /// before the acoustic connector (voice-prompt conditioning).
    pub fn scale_acoustic_latent(&self, latent: &Tensor) -> candle_core::Result<Tensor> {
        latent.affine(
            self.speech_scaling_factor,
            self.speech_bias_factor * self.speech_scaling_factor,
        )
    }

    /// `latent / speech_scaling_factor - speech_bias_factor` — inverts
    /// [`Self::scale_acoustic_latent`], applied to diffusion-sampled
    /// latents before decode/re-encode.
    pub fn unscale_acoustic_latent(&self, latent: &Tensor) -> candle_core::Result<Tensor> {
        latent.affine(1.0 / self.speech_scaling_factor, -self.speech_bias_factor)
    }

    pub fn acoustic_connector(&self) -> &SpeechConnector {
        &self.acoustic_connector
    }

    pub fn semantic_connector(&self) -> &SpeechConnector {
        &self.semantic_connector
    }

    /// Build a fresh [`DpmSolverScheduler`] and run the full
    /// `ddpm_num_inference_steps`-step denoising chain, producing one
    /// speech-latent vector per row of `condition`.
    ///
    /// `condition`: `[N, hidden_size]` decoder hidden states. No CFG here —
    /// interpolating between `condition` and `neg_condition` is the
    /// caller's responsibility (see [`Self::sample_speech_latents_cfg`]).
    /// Returns `[N, latent_size]`, still scaled; callers must apply
    /// [`Self::unscale_acoustic_latent`] before decoding to audio.
    pub fn sample_speech_latents(&self, condition: &Tensor) -> candle_core::Result<Tensor> {
        let mut scheduler =
            DpmSolverScheduler::new(&self.config.diffusion_head_config).map_err(|e| {
                candle_core::Error::Msg(format!("kugelaudio: build DPM scheduler: {e}"))
            })?;
        scheduler.set_timesteps(self.config.diffusion_head_config.ddpm_num_inference_steps);

        let n = condition.dim(0)?;
        let latent_size = self.config.diffusion_head_config.latent_size;
        let mut sample = Tensor::randn(0f32, 1f32, (n, latent_size), condition.device())?
            .to_dtype(condition.dtype())?;

        for &t in scheduler.timesteps().to_vec().iter() {
            let timesteps = Tensor::full(t as f32, n, condition.device())?;
            let eps = self
                .diffusion_head
                .forward(&sample, &timesteps, condition)?;
            sample = scheduler.step(&eps, &sample)?;
        }
        Ok(sample)
    }

    /// Classifier-free-guided variant of [`Self::sample_speech_latents`],
    /// `N == 1` (batch>1 not implemented — see [`Self::generate`]).
    /// `condition`/`neg_condition`: `[1, hidden_size]`. Returns
    /// `[1, latent_size]`, still scaled.
    ///
    /// Deviates from `kugelaudio_inference.py`'s CFG path: the Python
    /// maintains *two* noised rows (`speech = torch.randn(2, vae_dim)`)
    /// through the whole loop, feeding only row 0 to the diffusion head
    /// (duplicated into both rows) and discarding row 1 at the end. This
    /// port only tracks the one trajectory actually returned, duplicating
    /// on the fly for the conditional/unconditional forward pass —
    /// mathematically identical, no wasted half.
    pub fn sample_speech_latents_cfg(
        &self,
        condition: &Tensor,
        neg_condition: &Tensor,
        cfg_scale: f64,
    ) -> candle_core::Result<Tensor> {
        let mut scheduler =
            DpmSolverScheduler::new(&self.config.diffusion_head_config).map_err(|e| {
                candle_core::Error::Msg(format!("kugelaudio: build DPM scheduler: {e}"))
            })?;
        scheduler.set_timesteps(self.config.diffusion_head_config.ddpm_num_inference_steps);

        let latent_size = self.config.diffusion_head_config.latent_size;
        let combined_condition = Tensor::cat(&[condition, neg_condition], 0)?; // [2, hidden]
        let mut sample = Tensor::randn(0f32, 1f32, (1, latent_size), condition.device())?
            .to_dtype(condition.dtype())?;

        for &t in scheduler.timesteps().to_vec().iter() {
            let combined_sample = Tensor::cat(&[&sample, &sample], 0)?; // [2, latent]
            let timesteps = Tensor::full(t as f32, 2, condition.device())?;
            let eps =
                self.diffusion_head
                    .forward(&combined_sample, &timesteps, &combined_condition)?; // [2, latent]
            let cond_eps = eps.narrow(0, 0, 1)?;
            let uncond_eps = eps.narrow(0, 1, 1)?;
            let half_eps = (&uncond_eps + ((&cond_eps - &uncond_eps)? * cfg_scale)?)?;
            sample = scheduler.step(&half_eps, &sample)?;
        }
        Ok(sample)
    }

    /// Splice `replacement` (`[T, hidden_size]`) into `embeds`
    /// (`[1, N, hidden_size]`) at the single contiguous run of `true`
    /// positions in `mask` (length `N`) — `build_prompt`'s voice-prompt
    /// placeholder run. No-op if `mask` is all `false`.
    fn splice_speech_embeds(
        embeds: &Tensor,
        mask: &[bool],
        replacement: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let Some(start) = mask.iter().position(|&b| b) else {
            return Ok(embeds.clone());
        };
        let count = mask.iter().filter(|&&b| b).count();
        let n = mask.len();
        let before = embeds.narrow(1, 0, start)?;
        let after = embeds.narrow(1, start + count, n - start - count)?;
        let replacement = replacement.unsqueeze(0)?;
        Tensor::cat(&[&before, &replacement, &after], 1)
    }

    /// Pad/truncate `x` (`[1, T, dim]`) along time to `target_len`. A
    /// no-op for this checkpoint (both tokenizers share `encoder_ratios`).
    fn align_time_len(x: &Tensor, target_len: usize) -> candle_core::Result<Tensor> {
        let t = x.dim(1)?;
        match t.cmp(&target_len) {
            std::cmp::Ordering::Equal => Ok(x.clone()),
            std::cmp::Ordering::Greater => x.narrow(1, 0, target_len),
            std::cmp::Ordering::Less => {
                let dim = x.dim(2)?;
                let pad = Tensor::zeros((1, target_len - t, dim), x.dtype(), x.device())?;
                Tensor::cat(&[x, &pad], 1)
            },
        }
    }

    /// Generate speech for `prompt`, batch size 1 only.
    ///
    /// Simplified batch-1 port of `kugelaudio_inference.py`'s `generate()`
    /// (see module doc comment for why the batched version's KV cache
    /// splice isn't ported). The simplification is exact for the
    /// single-utterance case: with one sample, the Python's per-step
    /// "is this sample mid-diffusion or not" bookkeeping collapses to a
    /// single always-true-or-false flag, and the retroactive cache
    /// correction it drives is dead code when `speech_start` occurs at
    /// most once and `speech_end`/`eos` are terminal. Multi-speaker
    /// prompts that re-emit `speech_start` mid-generation are not ported.
    ///
    /// No streaming tokenizer cache either (see `conv_layers.rs`): every
    /// diffusion step re-decodes the entire accumulated latent sequence
    /// through the acoustic decoder and re-encodes the entire resulting
    /// waveform through the semantic encoder from scratch, keeping only
    /// the newest frame's output — correct (causal convs give identical
    /// results for a position whether computed via a streaming cache or
    /// full recompute) but `O(steps²)` instead of `O(steps)`. Fine for
    /// short clips; revisit for long-form generation.
    #[allow(clippy::too_many_lines)]
    pub fn generate(
        &mut self,
        prompt: &PromptResult,
        voice_waveform: Option<&Tensor>,
        cfg: &KugelAudioGenerationConfig,
    ) -> Result<KugelAudioGenerationOutput> {
        use special_tokens::{EOS_TOKEN_ID, SPEECH_DIFFUSION_ID, SPEECH_END_ID, SPEECH_START_ID};

        self.clear_kv_cache();
        let device = self.device.clone();

        let ids_u32: Vec<u32> = prompt.token_ids.clone();
        let ids_tensor = Tensor::from_vec(ids_u32.clone(), (1, ids_u32.len()), &device)?;
        let mut text_embeds = self.embed_text_tokens(&ids_tensor)?;

        if prompt.voice_frame_count > 0 {
            let waveform = voice_waveform
                .context("kugelaudio generate: prompt has voice-prompt frames but no voice_waveform was given")?;
            let acoustic = self
                .encode_acoustic(waveform)?
                .transpose(1, 2)?
                .contiguous()?; // [1, T, vae_dim]
            let acoustic_scaled = self.scale_acoustic_latent(&acoustic)?;
            let semantic = self
                .encode_semantic(waveform)?
                .transpose(1, 2)?
                .contiguous()?; // [1, T_sem, sem_dim]
            let semantic = Self::align_time_len(&semantic, acoustic_scaled.dim(1)?)?;
            let acoustic_embed = self
                .acoustic_connector
                .forward(&acoustic_scaled.squeeze(0)?)?;
            let semantic_embed = self.semantic_connector.forward(&semantic.squeeze(0)?)?;
            let combined = (acoustic_embed + semantic_embed)?; // [T, hidden]
            text_embeds =
                Self::splice_speech_embeds(&text_embeds, &prompt.speech_input_mask, &combined)?;
        }

        let embed_one = |model: &Self, id: u32| -> candle_core::Result<Tensor> {
            let t = Tensor::from_vec(vec![id], (1, 1), &device)?;
            model.embed_text_tokens(&t)
        };
        let speech_start_embed = embed_one(self, SPEECH_START_ID)?;
        let inputs_embeds = Tensor::cat(&[&text_embeds, &speech_start_embed], 1)?;
        let mut generated_ids: Vec<u32> = vec![SPEECH_START_ID];

        let (mut hidden, mut logits) = self.forward(&inputs_embeds, 0)?;
        let mut pos_seqlen = inputs_embeds.dim(1)?;

        let use_cfg = cfg.cfg_scale != 1.0;
        let mut neg_decoder = self.decoder.clone();
        let mut neg_last_embed = speech_start_embed;
        let mut neg_seqlen = 0usize;

        let mut logits_processor =
            LogitsProcessor::new(42, cfg.do_sample.then_some(cfg.temperature), None);

        let candidate_ids = [
            SPEECH_START_ID,
            SPEECH_END_ID,
            SPEECH_DIFFUSION_ID,
            EOS_TOKEN_ID,
        ];

        let mut all_latents: Vec<Tensor> = Vec::new();
        let mut audio_samples: Vec<f32> = Vec::new();
        let mut prev_audio_len = 0usize;

        // CRANE_KUGELAUDIO_PROFILE=1: per-component wall-clock breakdown to
        // stderr at end of generation.
        let profile = std::env::var("CRANE_KUGELAUDIO_PROFILE").is_ok();
        let mut t_decoder_main = std::time::Duration::ZERO;
        let mut t_decoder_neg = std::time::Duration::ZERO;
        let mut t_diffusion = std::time::Duration::ZERO;
        let mut t_decode_acoustic = std::time::Duration::ZERO;
        let mut t_encode_semantic = std::time::Duration::ZERO;
        let mut n_diffusion_steps = 0usize;

        for _step in 0..cfg.max_new_tokens {
            let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.flatten_all()?;
            let candidate_values: Vec<f32> = candidate_ids
                .iter()
                .map(|&id| {
                    last_logits
                        .narrow(0, id as usize, 1)?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()
                        .map(|v| v[0])
                })
                .collect::<candle_core::Result<_>>()?;
            let candidate_logits =
                Tensor::from_vec(candidate_values, candidate_ids.len(), &device)?;
            let next_token = candidate_ids[logits_processor.sample(&candidate_logits)? as usize];

            generated_ids.push(next_token);
            if next_token == EOS_TOKEN_ID || next_token == SPEECH_END_ID {
                break;
            }

            let next_embed = if next_token == SPEECH_DIFFUSION_ID {
                let condition = hidden.narrow(1, hidden.dim(1)? - 1, 1)?.squeeze(1)?; // [1, hidden]

                let speech_latent_scaled = if use_cfg {
                    let t0 = std::time::Instant::now();
                    let neg_hidden = neg_decoder.forward_embeds(&neg_last_embed, neg_seqlen)?;
                    if profile {
                        self.device.synchronize()?;
                        t_decoder_neg += t0.elapsed();
                    }
                    neg_seqlen += 1;
                    let neg_condition = neg_hidden
                        .narrow(1, neg_hidden.dim(1)? - 1, 1)?
                        .squeeze(1)?;
                    let t0 = std::time::Instant::now();
                    let out =
                        self.sample_speech_latents_cfg(&condition, &neg_condition, cfg.cfg_scale)?;
                    if profile {
                        self.device.synchronize()?;
                        t_diffusion += t0.elapsed();
                    }
                    out
                } else {
                    let t0 = std::time::Instant::now();
                    let out = self.sample_speech_latents(&condition)?;
                    if profile {
                        self.device.synchronize()?;
                        t_diffusion += t0.elapsed();
                    }
                    out
                };
                n_diffusion_steps += 1;
                let speech_latent = self.unscale_acoustic_latent(&speech_latent_scaled)?;
                all_latents.push(speech_latent);

                let latents_bct = Tensor::stack(&all_latents, 1)?
                    .transpose(1, 2)?
                    .contiguous()?; // [1, vae_dim, T]
                let t0 = std::time::Instant::now();
                let full_audio = self.decode_acoustic(&latents_bct)?; // [1, 1, num_samples]
                if profile {
                    self.device.synchronize()?;
                    t_decode_acoustic += t0.elapsed();
                }
                let num_samples = full_audio.dim(2)?;
                let new_chunk =
                    full_audio.narrow(D::Minus1, prev_audio_len, num_samples - prev_audio_len)?;
                let chunk: Vec<f32> = new_chunk.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
                audio_samples.extend(chunk);
                prev_audio_len = num_samples;

                let t0 = std::time::Instant::now();
                let full_semantic = self.encode_semantic(&full_audio)?; // [1, sem_dim, T_sem]
                if profile {
                    self.device.synchronize()?;
                    t_encode_semantic += t0.elapsed();
                }
                let t_sem = full_semantic.dim(2)?;
                let last_semantic = full_semantic
                    .narrow(D::Minus1, t_sem - 1, 1)?
                    .transpose(1, 2)?
                    .contiguous()?
                    .squeeze(1)?; // [1, sem_dim]

                let acoustic_embed = self.acoustic_connector.forward(&speech_latent_scaled)?;
                let semantic_embed = self.semantic_connector.forward(&last_semantic)?;
                (acoustic_embed + semantic_embed)?.unsqueeze(1)? // [1, 1, hidden]
            } else {
                embed_one(self, next_token)?
            };

            let t0 = std::time::Instant::now();
            let (h, l) = self.forward(&next_embed, pos_seqlen)?;
            if profile {
                self.device.synchronize()?;
                t_decoder_main += t0.elapsed();
            }
            hidden = h;
            logits = l;
            pos_seqlen += 1;

            if use_cfg && next_token == SPEECH_DIFFUSION_ID {
                neg_last_embed = next_embed;
            }
        }

        if profile {
            eprintln!(
                "kugelaudio profile ({n_diffusion_steps} diffusion frames): \
                 decoder_main={t_decoder_main:.2?} decoder_neg={t_decoder_neg:.2?} \
                 diffusion_sampling={t_diffusion:.2?} decode_acoustic={t_decode_acoustic:.2?} \
                 encode_semantic={t_encode_semantic:.2?}"
            );
        }

        Ok(KugelAudioGenerationOutput {
            token_ids: generated_ids,
            audio: audio_samples,
        })
    }
}

/// Sampling/stopping knobs for [`KugelAudioModel::generate`]. Defaults
/// match `kugelaudio_inference.py`'s `generate()` signature.
#[derive(Debug, Clone)]
pub struct KugelAudioGenerationConfig {
    /// CFG strength. `1.0` disables CFG (single forward pass per diffusion
    /// step, no negative decoder stream).
    pub cfg_scale: f64,
    pub max_new_tokens: usize,
    pub do_sample: bool,
    pub temperature: f64,
}

impl Default for KugelAudioGenerationConfig {
    fn default() -> Self {
        Self {
            cfg_scale: 3.0,
            max_new_tokens: 2048,
            do_sample: false,
            temperature: 1.0,
        }
    }
}

/// Result of [`KugelAudioModel::generate`].
pub struct KugelAudioGenerationOutput {
    /// Generated continuation only (control/diffusion-placeholder token ids),
    /// starting with `speech_start_id` — **not** including the prompt.
    pub token_ids: Vec<u32>,
    /// Mono 24kHz waveform, concatenated across every generated frame.
    /// Empty if generation stopped before any diffusion token was produced.
    pub audio: Vec<f32>,
}

#[cfg(test)]
mod tests {
    // Real-checkpoint loading is exercised by
    // `crane-core/tests/kugelaudio_load.rs` (integration test — requires a
    // local clone of `kugelaudio/kugelaudio-0-open`).
}
