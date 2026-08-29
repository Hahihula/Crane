//! Built-in browser UI, deliberately kept separate from inference handlers.
//!
//! The routes in this module are only mounted when `crane-serve --ui` is used.

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::{StatusCode, header},
    response::{Html, IntoResponse},
};
use serde::Serialize;

use crate::AppState;

#[derive(Serialize)]
pub struct UiConfig {
    pub mode: &'static str,
    pub multimodal: bool,
    pub model_name: String,
    pub model_type: String,
}

/// The UI shell. Its JavaScript fetches `/ui/config` before rendering, so one
/// static asset can adapt to both text/VLM and ASR servers.
pub async fn index() -> impl IntoResponse {
    Html(include_str!("../ui/dist/index.html"))
}

/// Build artifacts are embedded at compile time, keeping `crane-serve --ui`
/// self-contained instead of requiring a Node/Vite process at runtime.
pub async fn asset(uri: axum::http::Uri) -> impl IntoResponse {
    match uri.path() {
        "/ui/assets/app.js" => (
            [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
            include_str!("../ui/dist/assets/app.js"),
        )
            .into_response(),
        "/ui/assets/index.css" => (
            [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
            include_str!("../ui/dist/assets/index.css"),
        )
            .into_response(),
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

pub async fn config(State(state): State<Arc<AppState>>) -> Json<UiConfig> {
    let multimodal = state.vlm_tx.is_some()
        || state.gemma4_vlm_tx.is_some()
        || state.qwen3_5_vlm_tx.is_some()
        || state.minicpm_v_vlm_tx.is_some();
    Json(UiConfig {
        mode: if state.asr_tx.is_some() {
            "asr"
        } else if state.tts_tx.is_some() {
            "tts"
        } else {
            "chat"
        },
        multimodal,
        model_name: state.model_name.clone(),
        model_type: state.model_type_name.clone(),
    })
}
