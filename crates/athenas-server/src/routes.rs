use std::sync::Arc;
use tokio::sync::Mutex;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse},
    response::sse::{Event, KeepAlive},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tracing::error;

use athenas_inference::{
    Backend, ChatMessage, ChatRequest, CompletionRequest, Role, StreamChunk,
};

type SharedBackend = Arc<Mutex<Box<dyn Backend>>>;

#[derive(Clone)]
struct AppState {
    backend: SharedBackend,
    api_key: Option<String>,
}

pub fn create_router(backend: SharedBackend, cors_enabled: bool, api_key: Option<String>) -> Router {
    let state = AppState { backend, api_key };

    let mut router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(list_models))
        .route("/v1/health", get(health))
        .route("/health", get(health));

    if cors_enabled {
        router = router.layer(CorsLayer::permissive());
    }

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

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn list_models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
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
        })).into_response()
    } else {
        Json(serde_json::json!({
            "object": "list",
            "data": []
        })).into_response()
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

    let messages: Vec<ChatMessage> = req.messages.iter().map(|m| {
        let role = match m.role.as_str() {
            "system" => Role::System,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::User,
        };
        ChatMessage { role, content: m.content.clone() }
    }).collect();

    let stop = match req.stop {
        Some(serde_json::Value::String(s)) => Some(vec![s]),
        Some(serde_json::Value::Array(arr)) => {
            Some(arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        }
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
        let model_name = backend.model_info().map(|i| i.name.clone()).unwrap_or_default();
        drop(backend);

        let backend2 = state.backend.clone();
        tokio::spawn(async move {
            let b = backend2.lock().await;
            if let Err(e) = b.chat_stream(chat_req, tx).await {
                error!("Stream error: {}", e);
            }
        });

        let stream = async_stream::stream! {
            let mut rx = rx;
            while let Some(chunk) = rx.recv().await {
                if chunk.done {
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

        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        match backend.chat(chat_req).await {
            Ok(resp) => {
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
                error!("Chat completion error: {}", e);
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

    let stop = match req.stop {
        Some(serde_json::Value::String(s)) => Some(vec![s]),
        Some(serde_json::Value::Array(arr)) => {
            Some(arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        }
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
        let model_name = backend.model_info().map(|i| i.name.clone()).unwrap_or_default();
        drop(backend);

        let backend2 = state.backend.clone();
        tokio::spawn(async move {
            let b = backend2.lock().await;
            if let Err(e) = b.complete_stream(comp_req, tx).await {
                error!("Stream error: {}", e);
            }
        });

        let stream = async_stream::stream! {
            let mut rx = rx;
            while let Some(chunk) = rx.recv().await {
                if chunk.done {
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

        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        match backend.complete(comp_req).await {
            Ok(resp) => {
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
                error!("Completion error: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            }
        }
    }
}
