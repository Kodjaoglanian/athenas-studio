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
use tokio::sync::Semaphore;
use tower_http::{
    compression::CompressionLayer, cors::CorsLayer, limit::RequestBodyLimitLayer,
    timeout::TimeoutLayer, trace::TraceLayer,
};

use athenas_core::ServerConfig;
use athenas_inference::{
    BackendFactory, ChatMessage, ChatRequest, CompletionRequest, ContentPart, MessageContent,
    ModelLoadConfig, Role, StreamChunk,
};

use crate::metrics::{metrics_middleware, SharedMetrics};
use crate::middleware::{rate_limit_middleware, SharedRateLimiter};
use crate::model_manager::SharedModelManager;

#[derive(Clone)]
struct AppState {
    model_manager: SharedModelManager,
    api_key: Option<String>,
    metrics: SharedMetrics,
    semaphore: Arc<Semaphore>,
    start_time: std::time::Instant,
}

pub fn create_router(
    model_manager: SharedModelManager,
    metrics: SharedMetrics,
    semaphore: Arc<Semaphore>,
    rate_limiter: SharedRateLimiter,
    config: &ServerConfig,
) -> Router {
    let state = AppState {
        model_manager,
        api_key: config.api_key.clone(),
        metrics: metrics.clone(),
        semaphore,
        start_time: std::time::Instant::now(),
    };

    let mut router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(list_models))
        .route("/v1/models/load", post(load_model_endpoint))
        .route("/v1/models/unload", post(unload_model_endpoint))
        .route("/v1/files", post(upload_file))
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
    let mgr = state.model_manager.lock().await;
    let uptime = state.start_time.elapsed().as_secs();
    let models = mgr.list();

    let mut json = serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": uptime,
        "models_loaded": mgr.count(),
        "default_model": mgr.default_id(),
    });

    if !models.is_empty() {
        json["models"] = serde_json::json!(models
            .iter()
            .map(|m| serde_json::json!({
                "id": m.id,
                "name": m.model_info.name,
                "context_size": m.model_info.context_size,
                "gpu_layers": m.model_info.gpu_layers,
                "backend": m.backend_name,
            }))
            .collect::<Vec<_>>());
    }

    Json(json)
}

async fn ready(State(state): State<AppState>) -> Response {
    let mgr = state.model_manager.lock().await;
    if mgr.has_models() {
        Json(serde_json::json!({"status": "ready", "models_loaded": mgr.count()})).into_response()
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

    let mgr = state.model_manager.lock().await;
    let models = mgr.list();
    let data: Vec<_> = models
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "object": "model",
                "created": chrono::Utc::now().timestamp(),
                "owned_by": "athenas-studio",
                "backend": m.backend_name,
                "context_size": m.model_info.context_size,
                "gpu_layers": m.model_info.gpu_layers,
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": data
    }))
    .into_response()
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
    #[serde(deserialize_with = "deserialize_content")]
    content: MessageContent,
}

/// Deserialize content that can be either a string or an array of content parts
fn deserialize_content<'de, D>(deserializer: D) -> Result<MessageContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    if value.is_string() {
        Ok(MessageContent::Text(
            value.as_str().unwrap_or("").to_string(),
        ))
    } else if value.is_array() {
        let parts: Vec<ContentPart> =
            serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(MessageContent::Parts(parts))
    } else {
        Ok(MessageContent::Text(String::new()))
    }
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

    let model_id = chat_req.model.clone();

    let mgr = state.model_manager.lock().await;
    let backend = match mgr.get(Some(model_id.as_str())) {
        Some(b) => b,
        None => {
            let available: Vec<_> = mgr.list().iter().map(|m| m.id.clone()).collect();
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!("Model '{}' not loaded. Available models: {:?}",
                        model_id, available)
                })),
            )
                .into_response();
        }
    };

    if chat_req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamChunk>(100);
        let model_name = backend
            .model_info()
            .map(|i| i.name.clone())
            .unwrap_or_default();
        drop(mgr);

        let mgr2 = state.model_manager.clone();
        let metrics = state.metrics.clone();
        let model_id_clone = model_id.clone();
        tokio::spawn(async move {
            let m = mgr2.lock().await;
            if let Some(b) = m.get(Some(model_id_clone.as_str())) {
                if let Err(e) = b.chat_stream(chat_req, tx).await {
                    tracing::error!("Stream error: {}", e);
                }
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
                            "content": resp.message.content.as_text(),
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

    let model_id = comp_req.model.clone();

    let mgr = state.model_manager.lock().await;
    let backend = match mgr.get(Some(model_id.as_str())) {
        Some(b) => b,
        None => {
            let available: Vec<_> = mgr.list().iter().map(|m| m.id.clone()).collect();
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!("Model '{}' not loaded. Available models: {:?}",
                        model_id, available)
                })),
            )
                .into_response();
        }
    };

    if comp_req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamChunk>(100);
        let model_name = backend
            .model_info()
            .map(|i| i.name.clone())
            .unwrap_or_default();
        drop(mgr);

        let mgr2 = state.model_manager.clone();
        let metrics = state.metrics.clone();
        let model_id_clone = model_id.clone();
        tokio::spawn(async move {
            let m = mgr2.lock().await;
            if let Some(b) = m.get(Some(model_id_clone.as_str())) {
                if let Err(e) = b.complete_stream(comp_req, tx).await {
                    tracing::error!("Stream error: {}", e);
                }
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

/// Upload an image file for use in multimodal chat completions.
/// Returns a data URI that can be used as image_url in chat messages.
async fn upload_file(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: axum::extract::Multipart,
) -> Response {
    if !check_auth(&headers, &state.api_key) {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mut file_data: Option<Vec<u8>> = None;
    let mut filename: Option<String> = None;
    let mut content_type: Option<String> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            filename = field.file_name().map(|s| s.to_string());
            content_type = field.content_type().map(|s| s.to_string());
            match field.bytes().await {
                Ok(bytes) => {
                    file_data = Some(bytes.to_vec());
                }
                Err(e) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({"error": format!("Failed to read file: {}", e)})),
                    )
                        .into_response();
                }
            }
        }
    }

    let data = match file_data {
        Some(d) => d,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "No file uploaded. Use multipart form with 'file' field."})),
            )
                .into_response();
        }
    };

    let mime = content_type.unwrap_or_else(|| {
        let name = filename.as_deref().unwrap_or("");
        if name.ends_with(".png") {
            "image/png".to_string()
        } else if name.ends_with(".jpg") || name.ends_with(".jpeg") {
            "image/jpeg".to_string()
        } else if name.ends_with(".gif") {
            "image/gif".to_string()
        } else if name.ends_with(".webp") {
            "image/webp".to_string()
        } else if name.ends_with(".bmp") {
            "image/bmp".to_string()
        } else {
            "application/octet-stream".to_string()
        }
    });

    if !mime.starts_with("image/") {
        return (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Json(serde_json::json!({"error": format!("Only image files are supported, got: {}", mime)})),
        )
            .into_response();
    }

    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &data);
    let data_uri = format!("data:{};base64,{}", mime, b64);

    let file_id = uuid::Uuid::new_v4().to_string();
    let response = serde_json::json!({
        "id": file_id,
        "object": "file",
        "bytes": data.len(),
        "created_at": chrono::Utc::now().timestamp(),
        "filename": filename.unwrap_or_else(|| "upload".to_string()),
        "purpose": "vision",
        "url": data_uri,
    });

    Json(response).into_response()
}

// ── Multi-model management endpoints ──────────────────────────────────

#[derive(Debug, Deserialize)]
struct LoadModelRequest {
    model_path: String,
    backend: Option<String>,
    gpu_layers: Option<i32>,
    context_size: Option<u32>,
    batch_size: Option<u32>,
    threads: Option<u32>,
    flash_attention: Option<bool>,
    reasoning_enabled: Option<bool>,
    reasoning_budget: Option<i32>,
    set_default: Option<bool>,
}

async fn load_model_endpoint(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    if !check_auth(&headers, &state.api_key) {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let backend_type = match req.backend.as_deref() {
        Some("llama.cpp") | Some("llamacpp") | Some("llama") => athenas_core::BackendType::LlamaCpp,
        Some("vllm") => athenas_core::BackendType::Vllm,
        _ => athenas_core::BackendType::Auto,
    };

    let hardware = match athenas_core::HardwareDetector::detect() {
        Ok(h) => h,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Hardware detection failed: {}", e)})),
            )
                .into_response();
        }
    };

    let mut backend = match BackendFactory::create(backend_type, &hardware) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Failed to create backend: {}", e)})),
            )
                .into_response();
        }
    };

    let load_config = ModelLoadConfig {
        model_path: req.model_path.clone(),
        gpu_layers: req.gpu_layers.unwrap_or(-1),
        context_size: req.context_size.unwrap_or(4096),
        batch_size: req.batch_size.unwrap_or(512),
        threads: req.threads.unwrap_or(0),
        flash_attention: req.flash_attention.unwrap_or(true),
        use_mmap: true,
        use_mlock: false,
        reasoning_enabled: req.reasoning_enabled.unwrap_or(false),
        reasoning_budget: req.reasoning_budget.unwrap_or(-1),
    };

    if let Err(e) = backend.load_model(load_config).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Failed to load model: {}", e)})),
        )
            .into_response();
    }

    let model_info = backend.model_info();
    let model_name = model_info
        .as_ref()
        .map(|i| i.name.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let backend_name = backend.name().to_string();

    let mut mgr = state.model_manager.lock().await;
    let model_id = mgr.add(backend);

    if req.set_default.unwrap_or(true) {
        let _ = mgr.set_default(&model_id);
    }

    let count = mgr.count();

    Json(serde_json::json!({
        "status": "loaded",
        "model_id": model_id,
        "model_name": model_name,
        "backend": backend_name,
        "models_loaded": count,
        "is_default": mgr.default_id() == Some(&model_id),
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
struct UnloadModelRequest {
    model_id: String,
}

async fn unload_model_endpoint(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<UnloadModelRequest>,
) -> Response {
    if !check_auth(&headers, &state.api_key) {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mut mgr = state.model_manager.lock().await;

    match mgr.remove(&req.model_id).await {
        Ok(()) => {
            let count = mgr.count();
            let default = mgr.default_id().map(|s| s.to_string());
            Json(serde_json::json!({
                "status": "unloaded",
                "model_id": req.model_id,
                "models_loaded": count,
                "default_model": default,
            }))
            .into_response()
        }
        Err(e) => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": e}))).into_response(),
    }
}
