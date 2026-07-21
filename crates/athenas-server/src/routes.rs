use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    middleware::from_fn_with_state,
    response::sse::{Event, KeepAlive},
    response::{IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Semaphore};
use tower_http::{
    compression::CompressionLayer, cors::CorsLayer, limit::RequestBodyLimitLayer,
    timeout::TimeoutLayer, trace::TraceLayer,
};

use athenas_core::ServerConfig;
use athenas_inference::{Backend, ChatMessage, ChatRequest, CompletionRequest, Role, StreamChunk};

use crate::metrics::{metrics_middleware, SharedMetrics};
use crate::middleware::{rate_limit_middleware, SharedRateLimiter};

type SharedBackend = Arc<Mutex<Box<dyn Backend>>>;

#[derive(Clone)]
struct AppState {
    backend: SharedBackend,
    api_key: Option<String>,
    metrics: SharedMetrics,
    semaphore: Arc<Semaphore>,
    start_time: std::time::Instant,
}

pub fn create_router(
    backend: SharedBackend,
    metrics: SharedMetrics,
    semaphore: Arc<Semaphore>,
    rate_limiter: SharedRateLimiter,
    config: &ServerConfig,
) -> Router {
    let state = AppState {
        backend,
        api_key: config.api_key.clone(),
        metrics: metrics.clone(),
        semaphore,
        start_time: std::time::Instant::now(),
    };

    let mut router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(list_models))
        .route("/v1/health", get(health))
        .route("/v1/ready", get(ready))
        .route("/health", get(health))
        .route("/metrics", get(metrics_endpoint));

    if config.cors_enabled {
        router = router.layer(CorsLayer::permissive());
    }

    if config.enable_compression {
        router = router.layer(CompressionLayer::new());
    }

    router = router
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(config.request_timeout_secs),
        ))
        .layer(RequestBodyLimitLayer::new(
            (config.max_body_size_mb as usize) * 1024 * 1024,
        ))
        .layer(TraceLayer::new_for_http())
        .layer(from_fn_with_state(rate_limiter, rate_limit_middleware))
        .layer(from_fn_with_state(metrics, metrics_middleware))
        .layer(tower_http::request_id::SetRequestIdLayer::x_request_id(
            tower_http::request_id::MakeRequestUuid,
        ));

    router.with_state(state)
}

fn check_auth(headers: &HeaderMap, api_key: &Option<String>) -> bool {
    if api_key.is_none() {
        return true;
    }
    let expected = api_key.as_ref().unwrap();
    if let Some(auth) = headers.get("authorization") {
        if let Ok(auth_str) = auth.to_str() {
            if auth_str.starts_with("Bearer ") {
                return &auth_str[7..] == expected;
            }
        }
    }
    false
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let backend = state.backend.lock().await;
    let model_info = backend.model_info();
    let uptime = state.start_time.elapsed().as_secs();

    let mut json = serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": uptime,
        "backend": backend.name(),
        "model_loaded": backend.is_loaded(),
    });

    if let Some(info) = model_info {
        json["model"] = serde_json::json!({
            "name": info.name,
            "context_size": info.context_size,
            "gpu_layers": info.gpu_layers,
            "backend": info.backend_name,
        });
    }

    Json(json)
}

async fn ready(State(state): State<AppState>) -> Response {
    let backend = state.backend.lock().await;
    if backend.is_loaded() {
        Json(serde_json::json!({"status": "ready"})).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "not_ready", "reason": "no model loaded"})),
        )
            .into_response()
    }
}

async fn metrics_endpoint() -> impl IntoResponse {
    let body = crate::metrics::Metrics::render();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

async fn list_models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !check_auth(&headers, &state.api_key) {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let backend = state.backend.lock().await;
    if let Some(info) = backend.model_info() {
        Json(serde_json::json!({
            "object": "list",
            "data": [{
                "id": info.name,
                "object": "model",
                "created": chrono::Utc::now().timestamp(),
                "owned_by": "athenas-studio",
            }]
        }))
        .into_response()
    } else {
        Json(serde_json::json!({
            "object": "list",
            "data": []
        }))
        .into_response()
    }
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatCompletionMessage>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    stream: Option<bool>,
    stop: Option<serde_json::Value>,
    seed: Option<u64>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ChatCompletionMessage {
    role: String,
    content: String,
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if !check_auth(&headers, &state.api_key) {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let _permit = match state.semaphore.acquire().await {
        Ok(permit) => permit,
        Err(_) => {
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
    };

    let messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "system" => Role::System,
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                _ => Role::User,
            };
            ChatMessage {
                role,
                content: m.content.clone(),
            }
        })
        .collect();

    let stop = match req.stop {
        Some(serde_json::Value::String(s)) => Some(vec![s]),
        Some(serde_json::Value::Array(arr)) => Some(
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
        ),
        _ => None,
    };

    let chat_req = ChatRequest {
        model: req.model.unwrap_or_default(),
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens.or(req.max_completion_tokens),
        stream: req.stream.unwrap_or(false),
        stop,
        seed: req.seed,
    };

    let backend = state.backend.lock().await;

    if chat_req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamChunk>(100);
        let model_name = backend
            .model_info()
            .map(|i| i.name.clone())
            .unwrap_or_default();
        drop(backend);

        let backend2 = state.backend.clone();
        let metrics = state.metrics.clone();
        tokio::spawn(async move {
            let b = backend2.lock().await;
            if let Err(e) = b.chat_stream(chat_req, tx).await {
                tracing::error!("Stream error: {}", e);
            }
        });

        let stream = async_stream::stream! {
            let mut rx = rx;
            while let Some(chunk) = rx.recv().await {
                if chunk.done {
                    if let Some(stats) = &chunk.stats {
                        metrics.record_tokens(&model_name, stats.tokens_prompt, stats.tokens_generated);
                    }
                    let json = serde_json::json!({
                        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                        "object": "chat.completion.chunk",
                        "created": chrono::Utc::now().timestamp(),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    });
                    yield Ok::<Event, std::convert::Infallible>(
                        Event::default().data(json.to_string())
                    );
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                let json = serde_json::json!({
                    "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    "object": "chat.completion.chunk",
                    "created": chrono::Utc::now().timestamp(),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk.text},
                        "finish_reason": null,
                    }],
                });
                yield Ok::<Event, std::convert::Infallible>(
                    Event::default().data(json.to_string())
                );
            }
        };

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        match backend.chat(chat_req).await {
            Ok(resp) => {
                state.metrics.record_tokens(
                    &resp.model,
                    resp.stats.tokens_prompt,
                    resp.stats.tokens_generated,
                );
                let json = serde_json::json!({
                    "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    "object": "chat.completion",
                    "created": chrono::Utc::now().timestamp(),
                    "model": resp.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": resp.message.content,
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": resp.stats.tokens_prompt,
                        "completion_tokens": resp.stats.tokens_generated,
                        "total_tokens": resp.stats.tokens_prompt + resp.stats.tokens_generated,
                    },
                });
                Json(json).into_response()
            }
            Err(e) => {
                tracing::error!("Chat completion error: {}", e);
                state
                    .metrics
                    .errors_total
                    .with_label_values(&["/v1/chat/completions", "inference"])
                    .inc();
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct CompletionRequestBody {
    model: Option<String>,
    prompt: String,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    stream: Option<bool>,
    stop: Option<serde_json::Value>,
    seed: Option<u64>,
}

async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CompletionRequestBody>,
) -> Response {
    if !check_auth(&headers, &state.api_key) {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let _permit = match state.semaphore.acquire().await {
        Ok(permit) => permit,
        Err(_) => {
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
    };

    let stop = match req.stop {
        Some(serde_json::Value::String(s)) => Some(vec![s]),
        Some(serde_json::Value::Array(arr)) => Some(
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
        ),
        _ => None,
    };

    let comp_req = CompletionRequest {
        model: req.model.unwrap_or_default(),
        prompt: req.prompt,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        stream: req.stream.unwrap_or(false),
        stop,
        seed: req.seed,
    };

    let backend = state.backend.lock().await;

    if comp_req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamChunk>(100);
        let model_name = backend
            .model_info()
            .map(|i| i.name.clone())
            .unwrap_or_default();
        drop(backend);

        let backend2 = state.backend.clone();
        let metrics = state.metrics.clone();
        tokio::spawn(async move {
            let b = backend2.lock().await;
            if let Err(e) = b.complete_stream(comp_req, tx).await {
                tracing::error!("Stream error: {}", e);
            }
        });

        let stream = async_stream::stream! {
            let mut rx = rx;
            while let Some(chunk) = rx.recv().await {
                if chunk.done {
                    if let Some(stats) = &chunk.stats {
                        metrics.record_tokens(&model_name, stats.tokens_prompt, stats.tokens_generated);
                    }
                    let json = serde_json::json!({
                        "id": format!("cmpl-{}", uuid::Uuid::new_v4()),
                        "object": "text_completion",
                        "created": chrono::Utc::now().timestamp(),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "text": "",
                            "finish_reason": "stop",
                        }],
                    });
                    yield Ok::<Event, std::convert::Infallible>(
                        Event::default().data(json.to_string())
                    );
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                let json = serde_json::json!({
                    "id": format!("cmpl-{}", uuid::Uuid::new_v4()),
                    "object": "text_completion",
                    "created": chrono::Utc::now().timestamp(),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "text": chunk.text,
                        "finish_reason": null,
                    }],
                });
                yield Ok::<Event, std::convert::Infallible>(
                    Event::default().data(json.to_string())
                );
            }
        };

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        match backend.complete(comp_req).await {
            Ok(resp) => {
                state.metrics.record_tokens(
                    &resp.model,
                    resp.stats.tokens_prompt,
                    resp.stats.tokens_generated,
                );
                let json = serde_json::json!({
                    "id": format!("cmpl-{}", uuid::Uuid::new_v4()),
                    "object": "text_completion",
                    "created": chrono::Utc::now().timestamp(),
                    "model": resp.model,
                    "choices": [{
                        "index": 0,
                        "text": resp.text,
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": resp.stats.tokens_prompt,
                        "completion_tokens": resp.stats.tokens_generated,
                        "total_tokens": resp.stats.tokens_prompt + resp.stats.tokens_generated,
                    },
                });
                Json(json).into_response()
            }
            Err(e) => {
                tracing::error!("Completion error: {}", e);
                state
                    .metrics
                    .errors_total
                    .with_label_values(&["/v1/completions", "inference"])
                    .inc();
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            }
        }
    }
}
