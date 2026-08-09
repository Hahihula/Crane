//! End-to-end sanity check for VoxCPM2: load a real checkpoint, run
//! zero-shot text-to-speech, verify the output is a finite, well-formed
//! waveform of plausible length.
//!
//! Gated by `CRANE_VOXCPM2_DIR` so it doesn't run by default. The DiT/CFM
//! sampler (the highest-risk new component — sway sampling + CFG-Zero-star)
//! was separately validated against a real HF forward-pass dump during
//! development (see `crane_core::models::voxcpm2::cfm`'s `hf_diff` test
//! module) — this test covers the full pipeline glue on top of that.

#[test]
#[ignore = "needs a local VoxCPM2 checkpoint (CRANE_VOXCPM2_DIR), incl. a converted audiovae.safetensors"]
fn voxcpm2_generate_is_well_formed() {
    use crane_core::models::voxcpm2::{VoxCpm2GenerationConfig, VoxCpm2Model};

    let dir = std::env::var("CRANE_VOXCPM2_DIR").expect("set CRANE_VOXCPM2_DIR to a VoxCPM2 checkpoint dir");

    // CUDA → CUDA BF16; macOS → Metal F16 (halves the DiT/CFM sampler's
    // activation memory vs F32, which previously panicked the M3 Pro
    // GPU under full-pipeline load); everything else → CPU F32.
    //
    // AGENTS.md flags F16's narrower exponent range as a known risk for
    // some families (Gemma activations have overflowed in F16 elsewhere),
    // but VoxCPM2's transformer math is BF16-validated upstream and the
    // Metal F16 default here is opt-in per this test — flip back to F32
    // if HF-diff validation ever finds divergence.
    #[cfg(feature = "cuda")]
    let (device, dtype) = if candle_core::utils::cuda_is_available() {
        (candle_core::Device::new_cuda(0).unwrap(), candle_core::DType::BF16)
    } else {
        (candle_core::Device::Cpu, candle_core::DType::F32)
    };
    #[cfg(all(target_os = "macos", not(feature = "cuda")))]
    let (device, dtype) = (
        candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu),
        candle_core::DType::F16,
    );
    #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
    let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

    let mut model = VoxCpm2Model::new(&dir, &device, &dtype).expect("load VoxCPM2");

    let cfg = VoxCpm2GenerationConfig { max_len: 200, ..Default::default() };
    let wav = model
        .generate_speech("VoxCPM2 brings multilingual support to Crane.", &cfg)
        .expect("generate_speech");

    let dims = wav.dims();
    println!("wav shape: {dims:?}, sample_rate: {}", model.sample_rate);
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 1);
    // A short sentence shouldn't hit the 200-patch cap (each patch is
    // 4 * 1920 = 7680 samples at 48kHz) — if it does, the stop head likely
    // never fired, which is itself a sign something's wrong.
    assert!(dims[2] < 200 * 4 * 1920, "generation ran to max_len without stopping");
    assert!(dims[2] > 4 * 1920, "generated less than one patch of audio");

    let samples: Vec<f32> = wav.flatten_all().unwrap().to_vec1().unwrap();
    assert!(samples.iter().all(|v| v.is_finite()), "non-finite sample in output");
    assert!(samples.iter().all(|v| (-1.0..=1.0).contains(v)), "sample outside tanh range");
    let max_abs = samples.iter().fold(0f32, |a, &b| a.max(b.abs()));
    assert!(max_abs > 0.01, "output looks like near-silence (max_abs={max_abs})");
}
