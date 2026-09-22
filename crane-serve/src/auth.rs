// SPDX-License-Identifier: MIT

//! API key authentication middleware.
//!
//! Enforcement is skipped entirely when `AppState::api_keys` is empty
//! (open access, the default). Otherwise requests to non-exempt paths must
//! present a configured key via `Authorization: Bearer <key>` or
//! `X-Api-Key: <key>`.

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use subtle::{Choice, ConstantTimeEq};

use crate::AppState;
use crate::openai_api::{ErrorDetail, ErrorResponse};

/// Paths reachable without an API key even when authentication is enabled:
/// health checks, stats, and the UI shell (whose own API calls still require
/// a key). `/health_generate` is deliberately not exempt — unlike `/health`,
/// it runs a real generation through the engine and would otherwise let
/// unauthenticated callers consume inference capacity. `/v1/stats` is exempt
/// because it exposes only aggregate engine counters (throughput, queue
/// depth) with no per-request content, so monitoring dashboards can poll it
/// without credentials.
///
/// This list must stay in sync with the routes registered in
/// `build_router_with_ui` (`lib.rs`) — it isn't derived from the router
/// table. `auth_middleware_tests` in `lib.rs` exercises the current mapping;
/// review both when adding a route under `/ui/` or reusing `/`.
const EXEMPT_PATHS: &[&str] = &["/health", "/v1/stats", "/"];

/// Path prefixes reachable without an API key.
const EXEMPT_PREFIXES: &[&str] = &["/ui/"];

fn is_exempt(path: &str) -> bool {
    EXEMPT_PATHS.contains(&path) || EXEMPT_PREFIXES.iter().any(|p| path.starts_with(p))
}

/// Extracts the request's bearer token, checking `Authorization: Bearer
/// <key>` first and falling back to `X-Api-Key` if the `Authorization`
/// header is absent or not a Bearer token. The `Bearer` scheme is matched
/// case-insensitively, per RFC 7235's auth-scheme grammar.
fn extract_api_key(headers: &axum::http::HeaderMap) -> Option<&str> {
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split_once(' '))
        .and_then(|(scheme, token)| scheme.eq_ignore_ascii_case("bearer").then_some(token));
    if let Some(bearer) = bearer {
        return Some(bearer);
    }
    headers.get("X-Api-Key")?.to_str().ok()
}

/// Membership check against the configured keys. Each comparison is
/// constant-time for keys of equal length (`ct_eq` still returns early on a
/// length mismatch), and every configured key is compared — the result
/// isn't combined with short-circuiting — so a mismatch doesn't leak which
/// configured key index it was compared against.
fn key_matches(candidate: &str, keys: &[String]) -> bool {
    let mut result = Choice::from(0u8);
    for key in keys {
        result |= candidate.as_bytes().ct_eq(key.as_bytes());
    }
    result.into()
}

fn unauthorized() -> Response {
    let mut response = (
        StatusCode::UNAUTHORIZED,
        axum::Json(ErrorResponse {
            error: ErrorDetail {
                message: "Invalid API Key".to_string(),
                r#type: "invalid_request_error".to_string(),
                code: Some("invalid_api_key".to_string()),
            },
        }),
    )
        .into_response();
    // RFC 7235 requires a 401 response to carry `WWW-Authenticate`.
    response.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        header::HeaderValue::from_static("Bearer"),
    );
    response
}

/// Axum middleware enforcing API key authentication. Registered via
/// [`axum::middleware::from_fn_with_state`] on the whole router; exempt
/// paths and open-access (no keys configured) are handled internally so the
/// router itself doesn't need to be split.
pub async fn require_api_key(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    if state.api_keys.is_empty() || is_exempt(request.uri().path()) {
        return next.run(request).await;
    }

    match extract_api_key(request.headers()) {
        Some(key) if key_matches(key, &state.api_keys) => next.run(request).await,
        _ => unauthorized(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exempt_paths_recognized() {
        assert!(is_exempt("/health"));
        assert!(is_exempt("/v1/stats"));
        assert!(is_exempt("/"));
        assert!(is_exempt("/ui/config"));
        assert!(is_exempt("/ui/assets/app.js"));
    }

    #[test]
    fn non_exempt_paths_rejected() {
        assert!(!is_exempt("/v1/chat/completions"));
        assert!(!is_exempt("/v1/models"));
        assert!(!is_exempt("/generate"));
        assert!(!is_exempt("/health_generate"));
    }

    #[test]
    fn extracts_bearer_token() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Bearer sk-test".parse().unwrap());
        assert_eq!(extract_api_key(&headers), Some("sk-test"));
    }

    #[test]
    fn extracts_bearer_token_case_insensitive() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "bearer sk-test".parse().unwrap());
        assert_eq!(extract_api_key(&headers), Some("sk-test"));

        let mut headers = axum::http::HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "BEARER sk-test".parse().unwrap());
        assert_eq!(extract_api_key(&headers), Some("sk-test"));
    }

    #[test]
    fn extracts_x_api_key_fallback() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("X-Api-Key", "sk-test".parse().unwrap());
        assert_eq!(extract_api_key(&headers), Some("sk-test"));
    }

    #[test]
    fn authorization_header_takes_precedence() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Bearer sk-bearer".parse().unwrap());
        headers.insert("X-Api-Key", "sk-fallback".parse().unwrap());
        assert_eq!(extract_api_key(&headers), Some("sk-bearer"));
    }

    #[test]
    fn missing_headers_yield_none() {
        let headers = axum::http::HeaderMap::new();
        assert_eq!(extract_api_key(&headers), None);
    }

    #[test]
    fn non_bearer_authorization_yields_none() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Basic sk-test".parse().unwrap());
        assert_eq!(extract_api_key(&headers), None);
    }

    #[test]
    fn non_bearer_auth_falls_through_to_x_api_key() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Basic sk-test".parse().unwrap());
        headers.insert("X-Api-Key", "sk-fallback".parse().unwrap());
        assert_eq!(extract_api_key(&headers), Some("sk-fallback"));
    }

    #[test]
    fn key_matches_checks_all_configured_keys() {
        let keys = vec!["key1".to_string(), "key2".to_string()];
        assert!(key_matches("key1", &keys));
        assert!(key_matches("key2", &keys));
        assert!(!key_matches("key3", &keys));
    }

    #[test]
    fn key_matches_rejects_empty_candidate() {
        let keys = vec!["key1".to_string()];
        assert!(!key_matches("", &keys));
    }

    #[test]
    fn unauthorized_sets_www_authenticate() {
        let response = unauthorized();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers().get(header::WWW_AUTHENTICATE).unwrap(),
            "Bearer"
        );
    }
}
