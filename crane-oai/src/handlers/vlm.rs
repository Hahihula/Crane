//! VLM (Vision-Language Model) handlers for PaddleOCR-VL and Gemma4VL.
//!
//! These handlers bypass the text-only engine and use VLM models directly
//! for image+text inference.

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
};

use crane_core::models::paddleocr_vl::OcrTask;

use crate::openai_api::*;
use crate::sglang_api::*;
use crate::{make_error, now_epoch, AppState};

// ─────────────────────────────────────────────────────────────
//  PaddleOCR-VL Request Channel
// ─────────────────────────────────────────────────────────────

pub enum VlmRequest {
    /// Non-streaming request
    Recognize {
        img_path: std::path::PathBuf,
        task: OcrTask,
        max_tokens: usize,
        tx: tokio::sync::oneshot::Sender<Result<String, String>>,
    },
    /// Streaming request
    RecognizeStream {
        img_path: std::path::PathBuf,
        task: OcrTask,
        max_tokens: usize,
        token_tx: tokio::sync::mpsc::UnboundedSender<String>,
        done_tx: tokio::sync::oneshot::Sender<Result<(), String>>,
    },
}

// ─────────────────────────────────────────────────────────────
//  Image downloading
// ─────────────────────────────────────────────────────────────

/// Download an image from a URL to a temporary file.
/// Returns the path to the temp file (the file persists until the TempDir is dropped).
async fn download_image(url: &str) -> Result<(tempfile::TempDir, std::path::PathBuf), String> {
    let dir = tempfile::TempDir::new()
        .map_err(|e| format!("Failed to create temp dir: {e}"))?;

    let client = reqwest::Client::new();
    let resp = client
        .get(url)
        .header("User-Agent", "crane-oai/0.1")
        .send()
        .await
        .map_err(|e| format!("Failed to download image from '{}': {e}", url))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Image download failed (HTTP {}): {}",
            resp.status(),
            url
        ));
    }

    // Determine extension from content-type or URL.
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "".to_string());

    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("Failed to read image bytes: {e}"))?;

    let ext = if content_type.contains("image/png") {
        "png"
    } else if content_type.contains("image/webp") {
        "webp"
    } else if content_type.contains("image/jpeg") || content_type.contains("image/jpg") {
        "jpg"
    } else {
        // Fallback to URL extension
        let url_lower = url.to_lowercase();
        if url_lower.contains(".png") {
            "png"
        } else if url_lower.contains(".webp") {
            "webp"
        } else {
            "jpg" // safe default
        }
    };

    let img_path = dir.path().join(format!("image.{ext}"));

    std::fs::write(&img_path, &bytes)
        .map_err(|e| format!("Failed to write image to temp file: {e}"))?;

    Ok((dir, img_path))
}

/// Determine the OCR task from the text prompt.
fn detect_ocr_task(text: &str) -> OcrTask {
    let text_lower = text.to_lowercase();
    if text_lower.contains("table") {
        OcrTask::Table
    } else if text_lower.contains("formula") {
        OcrTask::Formula
    } else if text_lower.contains("chart") {
        OcrTask::Chart
    } else {
        OcrTask::Ocr
    }
}

// ─────────────────────────────────────────────────────────────
//  Chat Completions (VLM)
// ─────────────────────────────────────────────────────────────

/// Extract the first image URL and text prompt from chat messages.
fn extract_image_and_text(
    messages: &[crate::openai_api::ChatMessage],
    model_name: &str,
) -> Result<(String, String), (StatusCode, Json<ErrorResponse>)> {
    let mut image_urls = Vec::new();
    let mut text_prompt = String::new();

    for msg in messages {
        if msg.role == "user" {
            image_urls.extend(msg.image_urls());
            let text = msg.text_content();
            if !text.is_empty() {
                text_prompt = text;
            }
        }
    }

    if image_urls.is_empty() {
        if model_name == "Gemma4-VL" {
            return Err(make_error(
                StatusCode::BAD_REQUEST,
                "No image_url found in messages. Gemma4-VL requires at least one image.",
            ));
        }
        return Err(make_error(
            StatusCode::BAD_REQUEST,
            &format!("No image_url found in messages. {model_name} requires at least one image."),
        ));
    }

    Ok((image_urls.swap_remove(0), text_prompt))
}

/// Extract text prompt only (for models like Gemma4-VL that require images).
fn extract_text_only(
    messages: &[crate::openai_api::ChatMessage],
) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    let mut text_prompt = String::new();
    for msg in messages {
        if msg.role == "user" {
            let text = msg.text_content();
            if !text.is_empty() {
                text_prompt = text;
            }
        }
    }
    Ok(text_prompt)
}

/// Extract first image URL only, if any.
fn extract_first_image_url(
    messages: &[crate::openai_api::ChatMessage],
) -> Option<String> {
    for msg in messages {
        if msg.role == "user" {
            let mut urls = msg.image_urls();
            if !urls.is_empty() {
                return Some(urls.swap_remove(0));
            }
        }
    }
    None
}

/// Decode a base64 data URL into bytes and return (extension, bytes).
fn decode_base64_image(data_url: &str) -> Result<(&'static str, Vec<u8>), String> {
    let data_url = data_url.trim();
    let comma_pos = data_url.find(',').ok_or_else(|| format!("Invalid base64 data URL: missing comma"))?;
    let header = &data_url[..comma_pos];
    let encoded = &data_url[comma_pos + 1..];

    let mime = header
        .strip_prefix("data:")
        .and_then(|s| s.strip_suffix(";base64"))
        .unwrap_or("image/png");

    let bytes = base64::decode(encoded)
        .map_err(|e| format!("Failed to decode base64: {e}"))?;

    let ext = if mime.contains("png") {
        "png"
    } else if mime.contains("webp") {
        "webp"
    } else if mime.contains("jpeg") || mime.contains("jpg") {
        "jpg"
    } else {
        "png"
    };

    Ok((ext, bytes))
}

/// Download or decode an image (supports URL or base64 data URL).
async fn get_image(
    url_or_data: &str,
) -> Result<(tempfile::TempDir, std::path::PathBuf), String> {
    if url_or_data.starts_with("data:") {
        let (ext, bytes) = decode_base64_image(url_or_data)?;
        let dir = tempfile::TempDir::new()
            .map_err(|e| format!("Failed to create temp dir: {e}"))?;
        let img_path = dir.path().join(format!("image.{ext}"));
        std::fs::write(&img_path, &bytes)
            .map_err(|e| format!("Failed to write image to temp file: {e}"))?;
        Ok((dir, img_path))
    } else {
        download_image(url_or_data).await
    }
}

// ─────────────────────────────────────────────────────────────
//  VLM Chat Completions (PaddleOCR-VL)
// ─────────────────────────────────────────────────────────────

/// VLM-aware chat completions handler.
///
/// Extracts image URLs and text from multimodal messages, downloads
/// images, and runs PaddleOCR-VL inference.
pub async fn vlm_chat_completions(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let vlm_tx = state.vlm_tx.as_ref().ok_or_else(|| {
        make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM model not loaded")
    })?;

    let (image_url, text_prompt) = extract_image_and_text(&req.messages, "PaddleOCR-VL")?;
    let image_url = &image_url;

    // Get image (URL or base64 data URL).
    let (temp_dir, img_path) = get_image(image_url)
        .await
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &e))?;

    let task = detect_ocr_task(&text_prompt);
    let max_tokens = req.max_tokens;
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    if req.stream {
        // Streaming mode
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let (done_tx, _done_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

        if vlm_tx.send(VlmRequest::RecognizeStream {
            img_path,
            task,
            max_tokens,
            token_tx: tx,
            done_tx,
        }).is_err() {
            return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM engine thread crashed"));
        }

        let model_name = state.model_name.clone();
        let created = now_epoch();

        let stream = async_stream::stream! {
            // Role announcement chunk.
            let first_chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".into(),
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: Some("assistant".into()),
                        content: None,
                    },
                    finish_reason: None,
                }],
                usage: None,
            };
            yield Ok::<_, std::convert::Infallible>(Event::default().json_data(&first_chunk).unwrap());

            let mut completion_tokens = 0usize;

            // Stream tokens.
            while let Some(text) = rx.recv().await {
                completion_tokens += 1;
                let chunk = ChatCompletionChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: None,
                            content: Some(text),
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                };
                yield Ok(Event::default().json_data(&chunk).unwrap());
            }

            // Finish chunk.
            let finish_chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".into(),
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".into()),
                }],
                usage: None,
            };
            yield Ok(Event::default().json_data(&finish_chunk).unwrap());
            yield Ok(Event::default().data("[DONE]"));
        };

        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Non-streaming mode
        let (tx, rx) = tokio::sync::oneshot::channel();
        if vlm_tx.send(VlmRequest::Recognize {
            img_path,
            task,
            max_tokens,
            tx,
        }).is_err() {
            return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM engine thread crashed"));
        }

        let result = rx.await
            .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("VLM task dropped: {e}")))?
            .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("VLM inference failed: {e}")))?;

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".into(),
            created: now_epoch(),
            model: state.model_name.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: ChatMessageContent::Text(result),
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        };
        Ok(Json(response).into_response())
    }
}

// ─────────────────────────────────────────────────────────────
//  /generate (VLM)
// ─────────────────────────────────────────────────────────────

/// VLM-aware generate handler for SGLang-style `/generate`.
pub async fn vlm_generate(
    state: Arc<AppState>,
    req: GenerateRequest,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let vlm_tx = state.vlm_tx.as_ref().ok_or_else(|| {
        make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM model not loaded")
    })?;

    let image_url = req.image_url.as_deref().ok_or_else(|| {
        make_error(
            StatusCode::BAD_REQUEST,
            "PaddleOCR-VL requires 'image_url' in the generate request",
        )
    })?;

    // Get image (URL or base64 data URL).
    let (temp_dir, img_path) = get_image(image_url)
        .await
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &e))?;

    let text_prompt = req.text.as_deref().unwrap_or("OCR:");
    let task = detect_ocr_task(text_prompt);
    let max_tokens = req.sampling_params.max_new_tokens;
    let request_id = req
        .rid
        .unwrap_or_else(|| format!("gen-{}", uuid::Uuid::new_v4()));

    if req.stream {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let (done_tx, _done_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

        if vlm_tx.send(VlmRequest::RecognizeStream {
            img_path,
            task,
            max_tokens,
            token_tx: tx,
            done_tx,
        }).is_err() {
            return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM engine thread crashed"));
        }

        let rid = request_id.clone();
        let stream = async_stream::stream! {
            while let Some(text) = rx.recv().await {
                let chunk = GenerateStreamChunk {
                    text,
                    meta_info: None,
                };
                yield Ok::<_, std::convert::Infallible>(Event::default().json_data(&chunk).unwrap());
            }

            // Final chunk with meta.
            let final_chunk = GenerateStreamChunk {
                text: String::new(),
                meta_info: Some(GenerateMetaInfo {
                    id: rid,
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    finish_reason: "stop".into(),
                }),
            };
            yield Ok(Event::default().json_data(&final_chunk).unwrap());
            yield Ok(Event::default().data("[DONE]"));
        };

        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if vlm_tx.send(VlmRequest::Recognize {
            img_path,
            task,
            max_tokens,
            tx,
        }).is_err() {
            return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM engine thread crashed"));
        }

        let result = rx.await
            .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("VLM task dropped: {e}")))?
            .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("VLM inference failed: {e}")))?;

        let response = GenerateResponse {
            text: result,
            meta_info: GenerateMetaInfo {
                id: request_id,
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: "stop".into(),
            },
        };

        Ok(Json(response).into_response())
    }
}

// ─────────────────────────────────────────────────────────────
//  Gemma4 VLM
// ─────────────────────────────────────────────────────────────

pub struct Gemma4VlmRequest {
    /// MUST stay alive until the VLM thread finishes.
    pub temp_dir: Option<tempfile::TempDir>,
    pub img_path: std::path::PathBuf,
    pub text_prompt: String,
    pub max_tokens: usize,
    pub tx: tokio::sync::oneshot::Sender<Result<String, String>>,
}

/// Gemma4VL chat completions handler.
///
/// For text-only requests, creates a 224x224 black PNG placeholder.
/// The VLM needs a non-trivial image to generate non-zero image tokens.
/// temp_dir is sent through the channel to stay alive until the thread finishes.
pub async fn gemma4_vlm_chat_completions(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let g4vlm_tx = state.gemma4_vlm_tx.as_ref().ok_or_else(|| {
        make_error(StatusCode::INTERNAL_SERVER_ERROR, "Gemma4 VLM model not loaded")
    })?;

    let image_url_opt = extract_first_image_url(&req.messages);
    let text_prompt = extract_text_only(&req.messages)?;

    // temp_dir MUST stay alive until the VLM thread finishes.
    // We send it through the channel so it stays in scope.
    let mut temp_dir: Option<tempfile::TempDir> = None;
    let img_path: std::path::PathBuf = match image_url_opt {
        Some(url) => {
            let dir = tempfile::TempDir::new()
                .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("Failed to create temp dir: {e}")))?;
            temp_dir = Some(dir);
            let img_path = temp_dir.as_ref().unwrap().path().join("image.png");
            if url.starts_with("data:") {
                let (ext, bytes) = decode_base64_image(&url)
                    .map_err(|e| make_error(StatusCode::BAD_REQUEST, &e))?;
                let img_path = temp_dir.as_ref().unwrap().path().join(format!("image.{ext}"));
                std::fs::write(&img_path, &bytes)
                    .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Failed to write image: {e}")))?;
                img_path
            } else {
                let client = reqwest::Client::new();
                let resp = client.get(&url).send().await
                    .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Failed to download image: {e}")))?;
                let bytes = resp.bytes().await
                    .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Failed to read image bytes: {e}")))?;
                std::fs::write(&img_path, &bytes)
                    .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Failed to write image: {e}")))?;
                img_path
            }
        }
        None => {
            // Text-only: create a 224x224 black PNG placeholder.
            // A 1x1 PNG produces 0 image tokens; 224x224 produces non-zero.
            let dir = tempfile::TempDir::new()
                .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("Failed to create temp dir: {e}")))?;
            temp_dir = Some(dir);
            let placeholder = temp_dir.as_ref().unwrap().path().join("placeholder.png");
            // 224x224 black PNG (valid PNG, black pixels -> zero image embeddings)
            let png_data = base64::decode("iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAAAqUlEQVR4nO3BMQEAAADCoPVPbQlPoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAvgZM/gABE/clzwAAAABJRU5ErkJggg==")
                .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("Invalid placeholder: {e}")))?;
            std::fs::write(&placeholder, png_data)
                .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("Failed to write placeholder: {e}")))?;
            placeholder
        }
    };

    let max_tokens = req.max_tokens;
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    let (tx, rx) = tokio::sync::oneshot::channel();
    if g4vlm_tx
        .send(Gemma4VlmRequest {
            temp_dir,
            img_path,
            text_prompt,
            max_tokens,
            tx,
        })
        .is_err()
    {
        return Err(make_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Gemma4 VLM engine thread crashed",
        ));
    }

    let result = rx
        .await
        .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("Gemma4 VLM task dropped: {e}")))?
        .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("Gemma4 VLM inference failed: {e}")))?;

    let response = ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".into(),
        created: now_epoch(),
        model: state.model_name.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: ChatMessageContent::Text(result),
            },
            finish_reason: Some("stop".into()),
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };
    Ok(Json(response).into_response())
}