use crane::common::config::{CommonConfig, DataType, DeviceConfig};
use crane::llm::LlmModelType;
use crane::prelude::*;
use std::env;
use std::time::Instant;

fn main() -> CraneResult<()> {
    let args: Vec<String> = env::args().collect();
    let image_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("data/images/test_chart.png");

    let config = CommonConfig {
        model_path: args
            .get(2)
            .cloned()
            .unwrap_or_else(|| "checkpoints/PaddleOCRv6".to_string()),
        model_type: if args.get(3).map(String::as_str) == Some("vl") {
            LlmModelType::PaddleOcrVl
        } else {
            LlmModelType::PaddleOcrV6
        },
        device: DeviceConfig::Cpu,
        // device: DeviceConfig::Cuda(0),
        // dtype: DataType::BF16,
        dtype: DataType::F32,
        max_memory: None,
    };

    let mut ocr_client = OcrClient::new(config)?;

    let warmup_runs = env::var("CRANE_OCR_WARMUP_RUNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    if warmup_runs > 0 {
        println!("Warming up OCR for {warmup_runs} run(s)...");
        for run in 0..warmup_runs {
            let started = Instant::now();
            let _ = ocr_client.extract_text_with_locations(image_path)?;
            println!(
                "  warmup {}/{}: {:.3}s",
                run + 1,
                warmup_runs,
                started.elapsed().as_secs_f64()
            );
        }
    }

    println!("Performing OCR on image: {}", image_path);

    // Model construction is intentionally outside this timing boundary.
    let forward_started = Instant::now();
    let result = ocr_client.extract_text_with_locations(image_path)?;
    let forward_elapsed = forward_started.elapsed();
    println!(
        "OCR forward time (model load excluded): {:.3}s",
        forward_elapsed.as_secs_f64()
    );
    println!("Detected {} text region(s)", result.regions.len());
    for region in &result.regions {
        println!(
            "  [{}, {}, {}, {}] {:.3}: {}",
            region.left, region.top, region.right, region.bottom, region.confidence, region.text
        );
    }
    println!("OCR result:\n{}", result.text);

    Ok(())
}
