//! Simple Chat Example
//!
//! This example shows how to create a basic chat application using the Crane SDK.

use crane::common::config::{CommonConfig, DataType, DeviceConfig};
use crane::llm::{GenerationConfig, LlmModelType};
use crane::prelude::*;

fn main() -> CraneResult<()> {
    // Create a simple chat configuration.
    //
    // Qwen 3.5 (hybrid Gated Delta Net + attention) is CPU-only for now, so use
    // DeviceConfig::Cpu (dtype is forced to F32 on CPU). For Qwen3 / Qwen2.5 you
    // can switch model_type and use DeviceConfig::Metal / Cuda(0) with F16.
    let config = ChatConfig {
        common: CommonConfig {
            // Update this path to your local Qwen 3.5 checkpoint.
            model_path: "/home/hahihula/mywork/ai/additional_models/Qwen3.5-0.8B".to_string(),
            model_type: LlmModelType::Qwen35,
            device: DeviceConfig::Cpu,
            dtype: DataType::F32,
            max_memory: None,
        },
        generation: GenerationConfig {
            max_new_tokens: 48, // Keep responses short for demo (CPU, O(n^2) decode)
            temperature: Some(0.7),
            ..Default::default()
        },
        max_history_turns: 4,
        enable_streaming: true, // Enable streaming for real-time responses
    };

    // Create a new chat client
    let mut chat_client = ChatClient::new(config)?;

    // Send a simple message and get a response
    let response = chat_client.send_message("Tell me a joke about Rust.")?;
    println!("AI Response: {}", response);

    Ok(())
}
