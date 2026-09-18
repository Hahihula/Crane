//! End-to-end sanity check for `KugelAudio`'s `generate()`: builds a real
//! prompt with a real Qwen2-VL-family tokenizer, runs it through the real
//! checkpoint, verifies the output is well-formed (valid token sequence,
//! finite non-empty audio) — not output *correctness* (no HF reference
//! comparison yet).
//!
//! Gated by `CRANE_KUGELAUDIO_DIR` and `CRANE_KUGELAUDIO_TOKENIZER` (see
//! `prompt.rs` for why the tokenizer must be Qwen2-VL-family).

#![allow(clippy::doc_markdown)] // KugelAudio is the model name, not generic Markdown text

#[test]
#[ignore = "needs a local KugelAudio checkpoint + tokenizer.json (CRANE_KUGELAUDIO_DIR, CRANE_KUGELAUDIO_TOKENIZER)"]
fn kugelaudio_generate_is_well_formed() {
    #[cfg(feature = "cuda")]
    use crane_core::cuda_is_available;
    use crane_core::models::kugelaudio::model::special_tokens::{
        EOS_TOKEN_ID, SPEECH_DIFFUSION_ID, SPEECH_END_ID, SPEECH_START_ID,
    };
    use crane_core::models::kugelaudio::{
        KugelAudioGenerationConfig, KugelAudioModel, build_prompt,
    };
    use crane_core::{DType, Device};
    use tokenizers::Tokenizer;

    let dir = std::env::var("CRANE_KUGELAUDIO_DIR").expect("set CRANE_KUGELAUDIO_DIR");
    let tokenizer_path =
        std::env::var("CRANE_KUGELAUDIO_TOKENIZER").expect("set CRANE_KUGELAUDIO_TOKENIZER");

    // CUDA → CUDA BF16; macOS → Metal F16; everything else → CPU F32.
    #[cfg(feature = "cuda")]
    let (device, dtype) = if cuda_is_available() {
        (Device::new_cuda(0).unwrap(), DType::BF16)
    } else {
        (Device::Cpu, DType::F32)
    };
    #[cfg(all(target_os = "macos", not(feature = "cuda")))]
    let (device, dtype) = (Device::new_metal(0).unwrap_or(Device::Cpu), DType::F16);
    #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
    let (device, dtype) = (Device::Cpu, DType::F32);

    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("load tokenizer.json");
    let prompt =
        build_prompt(&tokenizer, "Hello there, this is a short test.", None).expect("build_prompt");
    assert!(prompt.speech_input_mask.iter().all(|&b| !b));

    let mut model =
        KugelAudioModel::from_pretrained(&dir, &device, dtype).expect("from_pretrained");

    let gen_cfg = KugelAudioGenerationConfig {
        cfg_scale: 1.0, // no CFG for this smoke run
        max_new_tokens: 15,
        do_sample: false,
        temperature: 1.0,
    };
    let out = model.generate(&prompt, None, &gen_cfg).expect("generate");

    eprintln!(
        "generated {} control tokens, {} audio samples",
        out.token_ids.len(),
        out.audio.len()
    );

    assert_eq!(out.token_ids[0], SPEECH_START_ID);
    let valid = [
        SPEECH_START_ID,
        SPEECH_END_ID,
        SPEECH_DIFFUSION_ID,
        EOS_TOKEN_ID,
    ];
    for &id in &out.token_ids {
        assert!(
            valid.contains(&id),
            "generated token {id} outside the constrained set {valid:?}"
        );
    }

    if !out.audio.is_empty() {
        let max_abs = out.audio.iter().fold(0f32, |m, &x| m.max(x.abs()));
        assert!(max_abs.is_finite(), "audio must be finite");
        assert!(max_abs > 0.0, "audio must not be silent");
    }
}
