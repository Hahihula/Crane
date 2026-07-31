use crane::common::config::{CommonConfig, DataType, DeviceConfig};
use crane::llm::LlmModelType;
use crane::prelude::*;
use std::env;

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
        // dtype: DataType::BF16,
        dtype: DataType::F32,
        max_memory: None,
    };

    let mut ocr_client = OcrClient::new(config)?;

    println!("Performing OCR on image: {}", image_path);

    let result = ocr_client.extract_text_with_locations(image_path)?;
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
