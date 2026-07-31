//! PaddleOCR v6 ONNX forward models.
//!
//! The complete detector/recognizer pipeline lives beside this file in
//! `pipeline.rs`; higher-level crates only select and invoke the backend.

use anyhow::{Context, Result, anyhow};
use candle_core::Tensor;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use crate::onnx::{self, proto::ModelProto};

pub const DEFAULT_CHECKPOINT_DIR: &str = "checkpoints/PaddleOCRv6";
pub const DETECTOR_FILE: &str = "pp-ocrv6_small_det.onnx";
pub const RECOGNIZER_FILE: &str = "pp-ocrv6_small_rec.onnx";
pub const DICTIONARY_FILE: &str = "ppocrv6_dict.txt";

/// Small PaddleOCR v6 detector/recognizer pair backed by Crane ONNX.
pub struct PaddleOcrV6 {
    detector: ModelProto,
    recognizer: ModelProto,
    detector_input: String,
    recognizer_input: String,
    detector_output: String,
    recognizer_output: String,
}

impl PaddleOcrV6 {
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let det_path = path.join(DETECTOR_FILE);
        let rec_path = path.join(RECOGNIZER_FILE);
        if !det_path.is_file() || !rec_path.is_file() {
            return Err(anyhow!(
                "PaddleOCR v6 checkpoint is incomplete at {}. Expected {} and {}. Download it with: modelscope download --model nextbig/PaddleOCR_v6 --local_dir {}",
                path.display(),
                DETECTOR_FILE,
                RECOGNIZER_FILE,
                path.display()
            ));
        }
        let detector = onnx::read_file(&det_path).with_context(|| {
            format!(
                "failed to load PaddleOCR v6 detector {}",
                det_path.display()
            )
        })?;
        let recognizer = onnx::read_file(&rec_path).with_context(|| {
            format!(
                "failed to load PaddleOCR v6 recognizer {}",
                rec_path.display()
            )
        })?;
        let (detector_input, detector_output) = io_names(&detector)?;
        let (recognizer_input, recognizer_output) = io_names(&recognizer)?;
        Ok(Self {
            detector,
            recognizer,
            detector_input,
            recognizer_input,
            detector_output,
            recognizer_output,
        })
    }

    pub fn detect(&self, image: &Tensor) -> Result<Tensor> {
        forward(
            &self.detector,
            &self.detector_input,
            &self.detector_output,
            image,
        )
    }

    pub fn recognize(&self, image: &Tensor) -> Result<Tensor> {
        forward(
            &self.recognizer,
            &self.recognizer_input,
            &self.recognizer_output,
            image,
        )
    }
}

fn io_names(model: &ModelProto) -> Result<(String, String)> {
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| anyhow!("ONNX model has no graph"))?;
    let initializers = graph
        .initializer
        .iter()
        .map(|value| value.name.as_str())
        .collect::<HashSet<_>>();
    let input = graph
        .input
        .iter()
        .find(|value| !initializers.contains(value.name.as_str()))
        .ok_or_else(|| anyhow!("ONNX graph has no non-initializer input"))?
        .name
        .clone();
    let output = graph
        .output
        .first()
        .ok_or_else(|| anyhow!("ONNX graph has no output"))?
        .name
        .clone();
    Ok((input, output))
}

fn forward(
    model: &ModelProto,
    input_name: &str,
    output_name: &str,
    input: &Tensor,
) -> Result<Tensor> {
    let values = onnx::simple_eval(
        model,
        HashMap::from([(input_name.to_owned(), input.clone())]),
    )?;
    values
        .get(output_name)
        .cloned()
        .ok_or_else(|| anyhow!("ONNX graph did not produce '{output_name}'"))
}
