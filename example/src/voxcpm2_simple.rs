//! VoxCPM2 Simple Example
//!
//! Generates speech from text using VoxCPM2 (OpenBMB), zero-shot only —
//! this Crane port does not implement reference-audio conditioning (voice
//! cloning) or streaming generation. See
//! `crane_core::models::voxcpm2` module docs for the current scope.
//!
//! Expects `model.safetensors` + a pre-converted `audiovae.safetensors`
//! (converted once from the upstream `audiovae.pth`, see the module docs)
//! alongside `config.json`/`tokenizer.json` in the checkpoint directory.
//!
//! # Usage
//!
//! ```bash
//! # Run default examples
//! cargo run --bin voxcpm2_simple --release -- checkpoints/VoxCPM2
//!
//! # Synthesize custom text
//! cargo run --bin voxcpm2_simple --release -- checkpoints/VoxCPM2 "Your custom text here"
//! ```

fn main() -> anyhow::Result<()> {
    use crane_core::models::voxcpm2::{VoxCpm2GenerationConfig, VoxCpm2Model};
    use crane_core::models::{DType, Device};

    let args: Vec<String> = std::env::args().collect();

    let model_path = if args.len() >= 2 {
        args[1].clone()
    } else {
        eprintln!("error: missing checkpoint directory argument");
        eprintln!();
        eprintln!("Usage:");
        eprintln!("  cargo run --bin voxcpm2_simple --release -- <checkpoint_dir> [text]");
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  <checkpoint_dir>  Path to the VoxCPM2 checkpoint directory");
        eprintln!("                    (must contain model.safetensors, audiovae.safetensors,");
        eprintln!("                     config.json, and tokenizer.json)");
        eprintln!("  [text]            Optional custom text to synthesize. If omitted,");
        eprintln!("                    the built-in example sentences are used.");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  cargo run --bin voxcpm2_simple --release -- checkpoints/VoxCPM2");
        eprintln!(
            "  cargo run --bin voxcpm2_simple --release -- checkpoints/VoxCPM2 \"Hello world\""
        );
        std::process::exit(1);
    };

    let custom_text = if args.len() > 2 {
        Some(args[2..].join(" "))
    } else {
        None
    };

    let device = {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        }
        #[cfg(all(target_os = "macos", not(feature = "cuda")))]
        {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        }
        #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
        {
            Device::Cpu
        }
    };
    let dtype = {
        #[cfg(feature = "cuda")]
        {
            DType::BF16
        }
        #[cfg(not(feature = "cuda"))]
        {
            DType::F32
        }
    };

    if matches!(device, Device::Cpu) {
        eprintln!(
            "WARNING: VoxCPM2 on CPU will be slow (multiple transformer passes per audio patch). GPU strongly recommended."
        );
    }

    println!("Loading VoxCPM2 from: {model_path}");
    println!("Device: {device:?}  dtype: {dtype:?}");

    let mut model = VoxCpm2Model::new(&model_path, &device, &dtype)?;
    println!("Sample rate: {} Hz", model.sample_rate);

    let output_dir = "data/audio/output";
    std::fs::create_dir_all(output_dir)?;

    let examples: Vec<(&str, &str)> = if let Some(ref text) = custom_text {
        vec![(text.as_str(), "voxcpm2_custom.wav")]
    } else {
        vec![
            (
                "Hello! I am Crane, an ultra-fast inference engine written in Rust.",
                "voxcpm2_en.wav",
            ),
            (
                "VoxCPM2 supports thirty languages with tokenizer-free speech generation.",
                "voxcpm2_en_2.wav",
            ),
        ]
    };

    let cfg = VoxCpm2GenerationConfig::default();

    for (i, (text, filename)) in examples.iter().enumerate() {
        println!("\n[{}/{}]", i + 1, examples.len());
        println!("  Text: {text}");

        let start = std::time::Instant::now();
        let wav = model.generate_speech(text, &cfg)?;
        let output_path = format!("{output_dir}/{filename}");
        let saved_path = crane::audio::save_wav(&wav, &output_path, model.sample_rate)?;
        let elapsed = start.elapsed();
        println!("  Saved {saved_path} in {elapsed:.1?}");
    }

    println!("\nDone!");
    Ok(())
}
