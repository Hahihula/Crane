//! Common handlers: health, stats.

use std::sync::Arc;

use axum::{Json, extract::State, response::IntoResponse};
use serde_json::json;

use crate::AppState;

/// `GET /health`
pub async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

/// `GET /v1/stats`
pub async fn stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let snap = state
        .engine
        .as_ref()
        .map(|e| e.stats.snapshot())
        .unwrap_or_default();
    Json(snap)
}
